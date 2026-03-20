# Model selection and optimization (task 2b write-up)

## Problem geometry

The released pairs are **not** two 125×125 grids with different “effective” resolution: `X_jets_LR` is stored at **64×64** and `X_jets` at **125×125**, both with **three channels** (calorimeter-style jet images). The generator therefore performs a **non-integer upscale** (factor ≈ 1.95). A **bicubic resize** to 125×125 is used as an explicit **coarse baseline inside the network**; the CNN predicts a **residual** added to that baseline. This stabilizes optimization because the network only refines an already sensible global layout.

## Generator (SRGAN-inspired)

- **Residual backbone** with **InstanceNorm** and ReLU: batch statistics are a poor fit for small GAN batches; instance normalization tracks per-map statistics and is common in SR GANs.
- **Depth / width** (`--n-res`, `--g-feats`): deeper generators can sharpen details but increase compute and instability on small data caps; defaults are chosen so CPU smoke runs complete quickly while preserving the SRGAN pattern (head → residual stack → tail conv).
- **Output parameterization**: `output = bicubic(LR→HR) + residual` keeps pixel errors bounded early in training compared with predicting HR from scratch.

## Discriminator (PatchGAN + class conditioning)

- **PatchGAN** stacks stride-2 blocks so the discriminator reasons about local texture patches, which matches the SRGAN literature and jet images’ local energy deposits.
- **Class conditioning**: quark vs gluon labels are injected by **scaling the last feature map** with an embedding vector (`feat * (1 + tanh(emb))`). This avoids spatial mismatches from an auxiliary projection head while still giving the discriminator class-aware cues. (A naïve projection sum mixed incompatible tensor shapes; the multiplicative form keeps a single coherent forward path.)

## Losses

- **Adversarial**: **Least-Squares GAN** objectives for `D` and `G` (stable compared with vanilla cross-entropy in many SR setups).
- **Content**: **L1** reconstruction on the normalized HR target. The relative weight `--lambda-adv` is kept **small** (default `1e-3`) so the generator prioritizes faithfulness to ground truth, with adversarial pressure refining high-frequency structure.
- **Physics-aware regularization**: a small **energy-matching L1 term** on denormalized total image energy reduces the failure mode where PSNR/SSIM look strong but the generated jet systematically overshoots or undershoots the HR energy scale.

## Optimization

- **Adam** with **higher learning rate for D** than G (`lr_d = 4e-4`, `lr_g = 1e-4`): a mild **two-timescale** setup that reduces generator collapse when D trains too fast.
- **Alternating updates**: one D step per G step; no historical replay buffers (kept simple for the evaluation task).
- **AMP** (`--amp` on CUDA): optional mixed precision for faster training on GPUs.

## Preprocessing

- Per-channel **mean/std** computed on the **training split only** and applied to both LR and HR. This equalizes channel scales before the L1 term dominates early training.

## Evaluation choices

- **PSNR / SSIM** on denormalized non-negative clips (`relu` prior to metrics) quantify pixel fidelity vs the HR target.
- **Bicubic baseline comparison** keeps the GAN honest: if SR does not beat a trivial resize, the adversarial part is not buying much.
- **Energy ratio**, **energy MAE**, and **radial profiles** probe whether super-resolution preserves coarse **physics-aware summaries**, not only perceptual quality.
- **Linear probe** on channel means is intentionally weak; it is a sanity check rather than a full quark/gluon classifier. Low accuracy simply indicates that **global channel means are not sufficient** for separation on this subset.

## What to try next (beyond the minimal deliverable)

- **Perceptual / feature loss** using a small CNN trained on jet images.
- **Multi-scale D** or **spectral normalization** if adversarial training oscillates.
- Calibrate **per-channel** or **per-class** energy penalties if one particle type drifts more than the other.
- Full-dataset training with **larger `--g-feats` / `--n-res`** once VRAM/time allow.
