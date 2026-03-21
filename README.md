# CMS_E2E

**ML4SCI GSoC 2026 — Task 2b:** super-resolution of CMS-style **jet images** with a **GAN** (low-res $\to$ high-res), evaluation vs.\ a bicubic baseline and physics-aware energy metrics.

**Write-up (PDF + LaTeX):** [`docs/GSoC26_Task2b_Report.pdf`](docs/GSoC26_Task2b_Report.pdf) · [`docs/GSoC26_Task2b_Report.tex`](docs/GSoC26_Task2b_Report.tex)

---

## What this repo contains

- **`super_resolution/`** — data loading, SRGAN-style generator/discriminator, training (`train.py`), evaluation JSON (`evaluate.py`). See [`super_resolution/README.md`](super_resolution/README.md) and [`super_resolution/MODEL_DISCUSSION.md`](super_resolution/MODEL_DISCUSSION.md).
- **Reference runs** — `super_resolution/runs/full_run/` and `super_resolution/runs/ablation_no_energy/` (metrics, `RUN_NOTES.md`).
