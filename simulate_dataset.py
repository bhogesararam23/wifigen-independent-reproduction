#!/usr/bin/env python3
"""My simplified simulator for the WiFi-to-image experiment.

I am not trying to reproduce the authors' exact electromagnetic solver here.
I use a 2-D single-scattering approximation so I can understand and test the
complete pipeline. For each transmitter/receiver pair, I add the contributions
from occupied pixels using their bistatic path length and inverse-distance
attenuation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image

C = 299_792_458.0
FREQ_HZ = 2.4e9
LAMBDA = C / FREQ_HZ


def boundary_nodes(n: int = 20, side: float = 3.0) -> np.ndarray:
    """Put n evenly spaced WiFi nodes around the square boundary."""
    pts = []
    # Avoid duplicating corners by parameterizing the perimeter.
    for i in range(n):
        t = i / n
        u = 4 * side * t
        if u < side:
            p = (u, 0.0)
        elif u < 2 * side:
            p = (side, u - side)
        elif u < 3 * side:
            p = (3 * side - u, side)
        else:
            p = (0.0, 4 * side - u)
        pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def shape_mask(shape: str, rng: np.random.Generator, n: int = 256, side: float = 3.0) -> np.ndarray:
    """Generate one of my randomly sized and positioned shape masks."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    x = (x + 0.5) / n * side
    y = (y + 0.5) / n * side
    cx = rng.uniform(0.35 * side, 0.65 * side)
    cy = rng.uniform(0.35 * side, 0.65 * side)
    radius = rng.uniform(0.10 * side, 0.22 * side)
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = x - cx, y - cy
    xr, yr = c * dx + s * dy, -s * dx + c * dy

    if shape == "circle":
        mask = xr**2 + yr**2 <= radius**2
    elif shape in {"square", "rectangle"}:
        width = rng.uniform(1.3, 2.0) * radius
        height = rng.uniform(0.8, 1.6) * radius
        mask = (np.abs(xr) <= width) & (np.abs(yr) <= height)
    elif shape == "triangle":
        # Equilateral triangle centered at origin, then rotated above.
        h = radius * 2.0
        v1 = np.array([0.0, 2 * h / 3])
        v2 = np.array([-h / np.sqrt(3), -h / 3])
        v3 = np.array([h / np.sqrt(3), -h / 3])
        den = ((v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1]))
        a = ((v2[1] - v3[1]) * (xr - v3[0]) + (v3[0] - v2[0]) * (yr - v3[1])) / den
        b = ((v3[1] - v1[1]) * (xr - v3[0]) + (v1[0] - v3[0]) * (yr - v3[1])) / den
        cc = 1 - a - b
        mask = (a >= 0) & (b >= 0) & (cc >= 0)
    elif shape == "ring":
        outer = xr**2 + yr**2 <= radius**2
        inner_r = rng.uniform(0.35, 0.60) * radius
        inner = xr**2 + yr**2 < inner_r**2
        mask = outer & ~inner
    else:
        raise ValueError(f"unknown shape: {shape}")
    return mask.astype(np.uint8)


def wifi_power(mask: np.ndarray, nodes: np.ndarray, side: float = 3.0,
               eps_r: complex = 4 + 0.4j, noise_db: float = 0.0,
               grid_stride: int = 4) -> np.ndarray:
    """Simulate a 19 x 20 power matrix with a coherent scattered field."""
    n = mask.shape[0]
    ys, xs = np.nonzero(mask[::grid_stride, ::grid_stride])
    xs = (xs * grid_stride + 0.5) / n * side
    ys = (ys * grid_stride + 0.5) / n * side
    pix = np.column_stack([xs, ys])
    if len(pix) == 0:
        pix = np.array([[side / 2, side / 2]])
    # Approximate cell area, reduced by stride because we subsample pixels.
    cell_area = (side / n) ** 2 * grid_stride**2
    k = 2 * np.pi / LAMBDA
    contrast = (eps_r - 1) / (eps_r + 1)
    power = np.zeros((len(nodes) - 1, len(nodes)), dtype=np.float64)
    for tx in range(len(nodes)):
        txp = nodes[tx]
        receivers = [i for i in range(len(nodes)) if i != tx]
        rt = np.linalg.norm(pix - txp[None, :], axis=1) + 1e-6
        direct_dist = np.linalg.norm(nodes[receivers] - txp[None, :], axis=1) + 1e-6
        direct = np.exp(-1j * k * direct_dist) / direct_dist
        for row, rx in enumerate(receivers):
            rrx = np.linalg.norm(pix - nodes[rx][None, :], axis=1) + 1e-6
            scattered = np.sum(np.exp(-1j * k * (rt + rrx)) / np.sqrt(rt * rrx))
            field = direct[row] + 0.08 * contrast * cell_area * scattered
            power[row, tx] = np.abs(field) ** 2
    # Normalize each sample into a stable numerical range, preserving structure.
    power = np.log1p(power)
    power = (power - power.mean()) / (power.std() + 1e-8)
    if noise_db > 0:
        sigma = noise_db / 20 * math.log(10)
        power = power + np.random.default_rng().normal(0, sigma, power.shape)
    return power.astype(np.float32)


def save_sample(out: Path, idx: int, rng: np.random.Generator, image_size: int, noise_db: float) -> dict:
    shapes = ["circle", "rectangle", "triangle", "ring"]
    shape = shapes[idx % len(shapes)]
    mask = shape_mask(shape, rng, image_size)
    nodes = boundary_nodes(20)
    signal = wifi_power(mask, nodes, noise_db=noise_db)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"sample_{idx:06d}.npz", wifi=signal, mask=mask, nodes=nodes)
    # Black object on white background matches the paper's binary-mask description.
    Image.fromarray((255 * (1 - mask)).astype(np.uint8)).save(out / f"sample_{idx:06d}.png")
    return {"index": idx, "shape": shape, "wifi_shape": list(signal.shape), "mask_shape": list(mask.shape)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("independent_data"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--noise-db", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    metadata = {"paper_frequency_hz": FREQ_HZ, "wavelength_m": LAMBDA, "doi_m": [3.0, 3.0], "nodes": 20, "noise_db": args.noise_db, "seed": args.seed, "samples": []}
    for i in range(args.samples):
        metadata["samples"].append(save_sample(args.out, i, rng, args.image_size, args.noise_db))
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
