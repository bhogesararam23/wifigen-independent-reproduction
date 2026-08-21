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

## Problems that remain

The biggest issue is that the exact physical data-generation process is not available. I also do not know the authors’ exact node coordinates, object distributions, noise settings, preprocessing, or checkpoint initialization. In addition, my model is much smaller than the paper’s StyleGAN-based WiFi-GEN model.

The next experiments should use more samples, a separate test set, longer training, per-shape IoU, threshold selection on the validation set, and a stronger simulator. I should also save representative WiFi matrices and predicted masks so that I can inspect whether the model is learning object location, object size, or only the average shape.

## Conclusion

At this stage, I would describe the project as a working independent baseline and a learning implementation of the paper’s main idea. It is not yet an exact reproduction of the reported numbers. The code and assumptions are documented so that I can improve the physical model and network step by step instead of hiding the missing information.

## Reference

[1]: https://arxiv.org/html/2401.04317v2 "WiFi-GEN: High-Resolution Indoor Imaging from WiFi Signals Using Generative AI"
