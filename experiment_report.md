# My First WiFi-GEN Reproduction Experiment

## Purpose

I started this experiment to understand the WiFi-GEN paper and to see whether I could build the complete signal-to-image pipeline myself. Since I did not have the authors’ original dataset or checkpoint, I treated this as an independent implementation rather than claiming an exact reproduction.

## What I implemented

I generated four types of binary shapes: circles, rectangles, triangles, and rings. Each shape was given a random size, position, and orientation. I then placed 20 WiFi nodes around a square domain of size `3 m × 3 m` and generated one `19 × 20` power matrix for each shape.

The forward model is intentionally simple. I calculate a coherent scattered field using the distance from the transmitter to each occupied pixel and from that pixel to the receiver. I combine this with a direct signal, convert it to power, normalize it, and optionally add noise. This gives me a reproducible way to test the machine-learning part of the project, but it is not a replacement for a full electromagnetic solver.

For the reconstruction model, I used a small convolutional network instead of trying to copy the full StyleGAN-based architecture immediately. The network receives the WiFi matrix and produces a `256 × 256` mask. I used weighted binary cross-entropy and Dice loss because most output pixels are background pixels.

## Dataset used for the first run

I generated 64 samples with random seed `2024`. The four shapes appear in a repeating order, so the dataset contains the same number of examples for each class. I used an 80/20 random train-validation split.

| Setting | Value |
|---|---:|
| Number of samples | 64 |
| Image size | 256 × 256 |
| WiFi input size | 19 × 20 |
| Number of nodes | 20 |
| Domain size | 3 m × 3 m |
| Random seed | 2024 |
| Training epochs | 5 |
| Batch size | 4 |
| Learning rate | 0.0001 |

## Results

The first ordinary BCE run tended to predict mostly background. Its best validation IoU was about `0.318`, but the final IoU dropped because the thresholded predictions became too small.

I then added positive-pixel weighting and Dice loss. This behaved better and reached a best validation IoU of approximately `0.327` after five epochs. The final IoU was about `0.310`.

| Experiment | Best validation IoU | Final validation IoU |
|---|---:|---:|
| BCE only | 0.318 | 0.081 |
| Weighted BCE + Dice, noise 0.02 dB | 0.327 | 0.310 |
| Weighted BCE + Dice, no noise | 0.327 | 0.310 |

The noisy and no-noise runs were almost identical. After checking the code, I realized that `0.02 dB` is a very small perturbation after the signal normalization step. Therefore, I do not consider this a meaningful noise-ablation result yet. I should repeat it with an SNR-based noise definition or with a larger noise level and report that choice clearly.

## Comparison with the paper

The paper reports an overall IoU of `0.795` for WiFi-GEN and gives shape-specific IoUs of `0.841` for circles, `0.836` for rectangles, `0.777` for triangles, and `0.728` for rings [1]. My result of `0.327` is much lower, but this comparison is not a controlled reproduction because my simulator, model, dataset size, loss, and training schedule are different.

The result is still useful because it confirms that the full pipeline runs: a shape is converted into a WiFi matrix, the model receives the matrix, and the model produces an image-shaped output that can be evaluated with IoU.

## Scaling up to 80,000 simulated samples

After verifying that my pipeline ran end-to-end with the 64-sample test, I simulated a full dataset of 80,000 samples to match the dataset size described in the WiFi-GEN paper.

I made one key change to the simulation parameters: I increased the noise level from `0.02 dB` to `0.5 dB`. In my initial experiment, `0.02 dB` was so small that it caused almost no difference after normalizing the power matrix. Setting the noise to `0.5 dB` introduces meaningful variation across transmitter-receiver pairs while keeping the shape signatures identifiable.

### Dataset summary

I saved the entire dataset under `train_data_large/`.

| Setting | Value |
|---|---:|
| Output folder | `train_data_large/` |
| Total samples | 80,000 |
| Total files | 160,001 (80,000 `.npz`, 80,000 `.png`, 1 `metadata.json`) |
| Dataset disk size | ~268 MB |
| Image size | 256 × 256 |
| WiFi input size | 19 × 20 |
| Number of boundary nodes | 20 |
| Room domain size | 3 m × 3 m |
| Random seed | 2024 |
| Noise level | 0.5 dB |
| Circles | 20,000 (25.0%) |
| Rectangles | 20,000 (25.0%) |
| Triangles | 20,000 (25.0%) |
| Rings | 20,000 (25.0%) |

Because my simulation script alternates through the four shape generators in order, the dataset is balanced across all four classes. Each sample contains:
- A compressed `.npz` file containing the `19 × 20` normalized WiFi power matrix (`wifi`), the `256 × 256` ground-truth binary mask (`mask`), and the 20 boundary coordinates (`nodes`).
- A corresponding `.png` preview image of the mask (inverted so the object is black and the background is white, matching the paper).
- An entry in `metadata.json` documenting the shape class and array dimensions.

I verified the dataset by loading the files with my `NpzDataset` class in PyTorch, confirming that all 80,000 samples load cleanly without missing indices or corrupt files. I keep `train_data_large/` in `.gitignore` rather than committing it to git, since tracking 160,001 files would bloat the repository and the entire dataset can be regenerated deterministically at any time using the simulation command.

### Next steps for training

With 80,000 samples, the reconstruction network will have enough diverse shapes, scales, and positions to learn generalizable features rather than overfitting to a small handful of masks.

Because I am currently running on a CPU without a dedicated GPU, training on all 80,000 samples for 50 epochs with batch size 8 would take tens of hours. My plan is to run this full training run on a machine with a CUDA-enabled GPU using the paper’s hyperparameter settings (batch size `8`, learning rate `0.0001`, and 50 epochs). I also plan to evaluate per-shape IoUs separately for circles, rectangles, triangles, and rings so that I can see which geometries are easiest to reconstruct from the boundary power measurements.

## Problems that remain and next steps

The biggest missing piece remains the authors’ original data and simulator: their exact electromagnetic solver, node ordering, object size distribution, and full StyleGAN-based WiFi-GEN architecture. Because of this, my results represent an independent implementation of the idea rather than an exact numerical replication.

Having simulated the 80,000-sample dataset, my next priorities are:
1. Running the 50-epoch training on a GPU with batch size `8` and learning rate `0.0001`.
2. Setting aside a fixed held-out test split rather than relying only on random validation folds.
3. Calculating per-shape IoU for circles, rectangles, triangles, and rings to compare directly against the paper’s shape breakdown (`0.841`, `0.836`, `0.777`, and `0.728`).
4. Saving representative predicted masks alongside their ground-truth shapes to inspect where the reconstruction errors happen.

## Conclusion

At this stage, I have a working independent baseline and a complete 80,000-sample simulated dataset matching the scale of the WiFi-GEN paper. The simulation, dataset loading, and reconstruction pipelines all run end-to-end. I have documented every assumption and parameter choice so that I can improve the physical approximation and model architecture step by step.

## References

- [1] [WiFi-GEN: High-Resolution Indoor Imaging from WiFi Signals Using Generative AI](https://arxiv.org/html/2401.04317v2)

[1]: https://arxiv.org/html/2401.04317v2 "WiFi-GEN: High-Resolution Indoor Imaging from WiFi Signals Using Generative AI"
