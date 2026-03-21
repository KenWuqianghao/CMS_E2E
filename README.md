# CMS_E2E

**ML4SCI GSoC 2026 — Task 2b:** super-resolution of CMS-style **jet images** with a **GAN** (low-res $\to$ high-res), evaluation vs.\ a bicubic baseline and physics-aware energy metrics.

**Write-up (PDF + LaTeX):** [`docs/GSoC26_Task2b_Report.pdf`](docs/GSoC26_Task2b_Report.pdf) · [`docs/GSoC26_Task2b_Report.tex`](docs/GSoC26_Task2b_Report.tex)

---

## What this repo contains

- **`super_resolution/`** — data loading, SRGAN-style generator/discriminator, training (`train.py`), evaluation JSON (`evaluate.py`). See [`super_resolution/README.md`](super_resolution/README.md) and [`super_resolution/MODEL_DISCUSSION.md`](super_resolution/MODEL_DISCUSSION.md).
- **Reference runs** — `super_resolution/runs/full_run/` and `super_resolution/runs/ablation_no_energy/` (metrics, `RUN_NOTES.md`).

---

## Data

Parquet shards from the task CERNBox link are **not** committed (`.parquet` is gitignored). Point `--data-dir` at the folder containing `*LR*.parquet` files.

---

## Task 2c (separate repository)

Work for **Task 2c** (HEPTAPOD / agentic CMS layout) lives in a **different** GitHub repository — the HEPTAPOD **fork**, not this repo:

```bash
git clone -b gsoc26-cms-restructure https://github.com/KenWuqianghao/heptapod.git
```

Upstream: [tonymenzo/heptapod](https://github.com/tonymenzo/heptapod). Task 2c PDF and docs are under **`heptapod/docs/`** on branch `gsoc26-cms-restructure`.
