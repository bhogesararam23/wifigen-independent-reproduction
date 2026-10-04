# My WiFi-GEN Reproduction Project

## What I am trying to reproduce

This project is my attempt to understand and reproduce the paper **WiFi-GEN: High-Resolution Indoor Imaging from WiFi Signals Using Generative AI**. The main idea is to use WiFi power measurements to reconstruct the shape of an object inside a room. In the paper, the input is a `19 × 20` WiFi power matrix and the output is a `256 × 256` image of the object.

I could not obtain the authors’ original dataset or pretrained model, so this is not an exact reproduction of their final numerical results. Instead, I reproduced the overall problem setup and built my own simplified simulator and reconstruction model. I have kept the assumptions visible so that I can improve them later.

The paper uses 20 WiFi nodes around a `3 m × 3 m` area, four object types, and a dataset of 80,000 signal-image pairs. It reports a learning rate of `0.0001`, batch size `8`, 50 training epochs, and a combined L2 and LPIPS loss with LPIPS weight `0.8` [1]. The authors’ code is available in their GitHub repository [2].

## Files in this project

| File or folder | What it is for |
|---|---|
| `simulate_dataset.py` | Creates random shapes and their simulated WiFi measurements |
| `train_model.py` | Trains my smaller WiFi-to-image neural network |
| `smoke_data/` | A 64-sample dataset used to check that everything works |
| `train_data_large/` | The full 80,000-sample simulated dataset matching the paper's dataset size |
| `smoke_run/` | Output checkpoint and logs from the smoke test run |
| `independent_run/` | Output directory for larger training runs |
| `experiment_report.md` | My notes, experimental results, and dataset documentation |

## Installing the packages

I tested the independent code with Python 3.11. The basic packages can be installed with:

```bash
sudo pip3 install scipy torch torchvision scikit-learn tqdm
```

The official repository uses an older Python and PyTorch setup, so I kept it separate from this implementation instead of forcing both projects to use exactly the same environment.

## Generating the simulated data

The following command generates a small dataset for testing:

```bash
python simulate_dataset.py \
  --out smoke_data \
  --samples 64 \
  --seed 2024 \
  --noise-db 0.02
```

To match the scale of the dataset in the paper, I generated a full 80,000-sample dataset:

```bash
python simulate_dataset.py \
  --out train_data_large \
  --samples 80000 \
  --seed 2024 \
  --noise-db 0.5
```

This produced 80,000 `.npz` files and 80,000 corresponding `.png` mask previews under `train_data_large/`, along with `metadata.json`. The samples are evenly balanced with 20,000 examples for each shape class (circle, rectangle, triangle, and ring). Each `.npz` file contains three arrays: `wifi` with shape `(19, 20)`, `mask` with shape `(256, 256)`, and `nodes` containing the 20 boundary-node coordinates. I set `--noise-db 0.5` here because the initial 0.02 dB setting produced virtually no change after normalization.

Because 80,000 samples produce over 160,000 individual files, I intentionally exclude `train_data_large/` and `.npz` files from git in `.gitignore`. Tracking this many small files directly in git would slow down repository operations and bloat the commit history unnecessarily. Because the simulator uses a deterministic random seed (`--seed 2024`), anyone running the command above can reproduce the exact same dataset locally.

## My simplified WiFi simulator

I placed 20 nodes evenly around the boundary of a square room. For each transmitter, the other 19 nodes are treated as receivers. The object is represented by a binary mask, and I calculate an approximate scattered complex field by summing contributions from the occupied pixels.

The phase depends on the total distance from the transmitter to the object and then to the receiver. I use inverse-distance attenuation, add the direct signal, convert the resulting field to power, and normalize the matrix. Optional Gaussian noise can then be added.

This is a simplified single-scattering approximation. I am using it to understand the data-generation process and to test the learning pipeline. I am **not** claiming that it is the same electromagnetic solver used by the paper’s authors.

## Training my reconstruction model

Before attempting a longer run, I recommend checking the code with a one-epoch smoke test:

```bash
python train_model.py \
  --data smoke_data \
  --out smoke_run \
  --epochs 1 \
  --batch-size 2 \
  --cpu
```

To train on the full 80,000-sample dataset:

```bash
python train_model.py \
  --data train_data_large \
  --out independent_run \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4
```

My model is smaller than the StyleGAN-based WiFi-GEN architecture in the paper. It first extracts features from the `19 × 20` matrix, expands them into a latent representation, and then upsamples them to produce the `256 × 256` mask. I use weighted binary cross-entropy together with Dice loss because the object occupies much less area than the background.

## First results

Using 64 samples and five epochs, the best validation IoU was approximately `0.327`. This was only a small debugging experiment. The model was able to learn something, but the result should not be compared directly with the paper’s reported IoU of `0.795`.

My first no-noise comparison was almost identical to the noisy run. The reason is that I initially used only `0.02 dB` of noise, which was too small to have a visible effect after normalization. In the next experiment, I should define noise using an explicit SNR or use a larger, justified noise level.

## Limitations I need to keep in mind

The most important missing information is the authors’ original 80,000-example dataset, pretrained checkpoint, exact electromagnetic simulator, exact node ordering, object-size distribution, noise model, and detailed training configuration. Because of this, my current result should be described as an **independent baseline implementation**, not as an exact replication of the paper’s reported performance.

I have now generated my own 80,000-sample dataset to match the paper’s data scale. The next improvements I would make are GPU-accelerated training for 50 epochs, a fixed held-out test set, per-shape IoU for circles, rectangles, triangles, and rings, a better noise model, and a higher-capacity network. I would only report FID after deciding whether the usual image feature extractor is appropriate for these binary synthetic masks.

## Checking the official code later

If I obtain the authors’ dataset or checkpoint later, I can test the official implementation separately:

```bash
gh repo clone CNFightingSjy/WiFiGEN WiFiGEN-official
cd WiFiGEN-official
python3 tools/check_artifacts.py /path/to/data4psp
```

The official loader expects this directory structure:

```text
data4psp/
  train/ptot/*.mat
  train/rgb_epsono/*
  test/ptot/*.mat
  test/rgb_epsono/*
```

Each MATLAB file should contain a variable named `data` with shape `(19, 20)`.

## References

[1]: https://arxiv.org/html/2401.04317v2 "WiFi-GEN: High-Resolution Indoor Imaging from WiFi Signals Using Generative AI"
[2]: https://github.com/CNFightingSjy/WiFiGEN "Official WiFiGEN implementation"
