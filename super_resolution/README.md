# CMS jet super-resolution (GSoC task 2b)

Trains an **SRGAN-style** model to map **low-resolution** calorimeter jet images `X_jets_LR` **(3×64×64)** to high-resolution `X_jets` **(3×125×125)** for quark/gluon labelled samples from the ML4SCI parquet release (see the GSoC PDF for the CERNBox link).

The training loop reports what reviewers care about:

- whether the model beats a plain **bicubic** baseline on image metrics
- whether outputs preserve **total deposited energy**, not only PSNR/SSIM

## Training protocols

Two tiers are documented on purpose:

1. **Reference run (host-constrained)** — Use this when you have limited RAM/disk (~tens of GB). Training uses **materialized subsets** (`--max-train-samples`, `--max-val-samples`) and fewer epochs so a laptop can finish. Metrics in this repo were produced this way; see `super_resolution/runs/full_run/RUN_NOTES.md` and `super_resolution/runs/ablation_no_energy/RUN_NOTES.md`. **Evaluation still uses the full test split** unless you pass `--max-test-samples`.
2. **Full-scale run** — When you have **≥~25 GB RAM or disk** for memmap (or enough RAM to hold the full training tensor cache), omit the `--max-*` caps, prefer `--memmap-train` if materializing everything, and use **`--epochs 20`** as in the command block below. That matches the original “large experiment” intent from the task description.

A small **smoke run** (hundreds of samples, few epochs) is fine for debugging the pipeline; do not confuse it with the reference or full-scale protocols above.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r super_resolution/requirements.txt
```

Place all `*LR*.parquet` shards in a single directory (the default layout uses the project root).

## Train

**Reference run** (subset + fewer epochs; reproduces checked-in `metrics_report.json` when data/split match):

```bash
python3 super_resolution/train.py \
  --data-dir /path/to/parquet_dir \
  --out-dir super_resolution/runs/full_run \
  --epochs 8 \
  --batch-size 8 \
  --g-feats 64 \
  --d-feats 64 \
  --n-res 8 \
  --lambda-energy 0.05 \
  --max-train-samples 16000 \
  --max-val-samples 4000 \
  --amp
```

**No-energy ablation** (same caps, turn off energy penalty):

```bash
python3 super_resolution/train.py \
  --data-dir /path/to/parquet_dir \
  --out-dir super_resolution/runs/ablation_no_energy \
  --epochs 8 \
  --batch-size 8 \
  --g-feats 64 \
  --d-feats 64 \
  --n-res 8 \
  --lambda-energy 0 \
  --max-train-samples 16000 \
  --max-val-samples 4000 \
  --amp
```

**Full-scale** (no subset caps; long training):

```bash
python3 super_resolution/train.py \
  --data-dir /path/to/parquet_dir \
  --out-dir super_resolution/runs/my_run \
  --epochs 20 \
  --batch-size 8 \
  --g-feats 64 \
  --d-feats 64 \
  --n-res 8 \
  --lambda-energy 0.05 \
  --amp
```

Optional: add `--memmap-train` under `my_run/` when memory is tight; see `train.py` help.

- **`--max-train-samples` / `--max-val-samples`**: cap how many jets are copied into dense arrays (each jet needs on the order of **~90 KB** once cached).
- **Checkpoints**: `checkpoint_best.pt` (lowest val L1), `checkpoint_last.pt`.
- **`history.csv`**: per-epoch `val_l1`, bicubic `val_bicubic_l1`, and `val_energy_mae`.
- **`--lambda-energy`**: energy-matching penalty on the generator; set to `0` for the ablation.

## Evaluate

```bash
python3 super_resolution/evaluate.py \
  --checkpoint super_resolution/runs/full_run/checkpoint_best.pt \
  --data-dir /path/to/parquet_dir \
  --save-report super_resolution/runs/full_run/metrics_report.json
```

Reports the SR model and the **bicubic** baseline:

- **PSNR / SSIM**
- **total-energy ratio** and **energy MAE**
- **mean radial profile** L1 distance
- a toy **linear probe** on channel-wise means

The file is **JSON** (use `.json` in `--save-report`) so it can be dropped into a report or notebook.

## Why materialized subsets?

Each public shard is stored as **one Parquet row group**, so naive per-row reads pull large chunks into memory. The loader streams record batches and **copies only the indices** you train on into dense NumPy arrays. For many shards, set **`JET_SR_MATERIALIZE_WORKERS`** (e.g. `4`) to decode distinct files in parallel during that copy step.

## Model and optimization discussion

See [MODEL_DISCUSSION.md](MODEL_DISCUSSION.md) for architecture choices, losses, and tuning notes for the write-up expected by the task.
