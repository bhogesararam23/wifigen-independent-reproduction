#!/usr/bin/env python3
"""Train my first WiFi-to-image reconstruction model.

This model is deliberately smaller than the StyleGAN-based model in the paper.
I am using it first to check that the data format, training loop, and IoU
calculation work before trying a larger architecture.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class NpzDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.glob("*.npz"))
        if not self.paths: raise FileNotFoundError(f"I could not find any .npz samples under {root}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        d = np.load(self.paths[i])
        x = torch.from_numpy(d["wifi"].astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(d["mask"].astype(np.float32)).unsqueeze(0)
        return x, y


class WiFiReconstructor(nn.Module):
    def __init__(self, out_size: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 5 * 5, 256 * 8 * 8), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 96, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 1),
        )
    def forward(self, x):
        return self.decoder(self.fc(self.encoder(x)))


def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > threshold)
    inter = (pred & (target > 0.5)).sum().item()
    union = (pred | (target > 0.5)).sum().item()
    return float(inter / union) if union else 1.0


def run(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = NpzDataset(Path(args.data))
    n_val = max(1, int(len(ds) * args.val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = WiFiReconstructor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    history = []
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            dice = 1.0 - (2 * (probs * y).sum(dim=(1,2,3)) + 1.0) / (probs.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) + 1.0)
            loss = bce(logits, y) + args.dice_weight * dice.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * x.size(0)
        model.eval(); val_loss = 0.0; val_iou = 0.0; count = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device); logits = model(x)
                val_loss += bce(logits, y).item() * x.size(0)
                val_iou += iou_from_logits(logits, y) * x.size(0); count += x.size(0)
        row = {"epoch": epoch, "train_bce": train_loss / n_train, "val_bce": val_loss / count, "val_iou": val_iou / count, "device": str(device)}
        history.append(row); print(json.dumps(row))
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, out / "checkpoint.pt")
        if epoch == args.epochs:
            with torch.no_grad():
                x, y = next(iter(val_dl)); pred = (torch.sigmoid(model(x.to(device))).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
                Image.fromarray((255 * (1 - pred)).astype(np.uint8)).save(out / "prediction.png")
                Image.fromarray((255 * (1 - y.numpy()[0, 0])).astype(np.uint8)).save(out / "target.png")
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", default="run"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--pos-weight", type=float, default=4.0); p.add_argument("--dice-weight", type=float, default=1.0); p.add_argument("--val-fraction", type=float, default=0.2); p.add_argument("--seed", type=int, default=2024); p.add_argument("--cpu", action="store_true"); run(p.parse_args())
