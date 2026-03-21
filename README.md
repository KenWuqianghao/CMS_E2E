# CMS_E2E

This repository holds **ML4SCI GSoC 2026** work for two **separate** evaluation tasks: **Task 2b** (jet image super-resolution with a GAN) and **Task 2c** (HEPTAPOD fork: CMS-oriented agent layout and tooling).

**Formal write-up (PDF + LaTeX):** see [`docs/GSoC26_CMS_E2E_Report.pdf`](docs/GSoC26_CMS_E2E_Report.pdf) and [`docs/GSoC26_CMS_E2E_Report.tex`](docs/GSoC26_CMS_E2E_Report.tex). The sections below mirror that document in Markdown.

---

## Task 2b — Super-resolution at the CMS detector

### What was done

- Implemented an **SRGAN-style** pipeline: **generator** maps low-resolution jets `X_jets_LR` (**3×64×64**) to high-resolution `X_jets` (**3×125×125**), with **bicubic upsampling + residual CNN**; **PatchGAN discriminator** with **quark/gluon class conditioning**.
- **Losses:** least-squares GAN objectives, **L1** on normalized HR, small adversarial weight, optional **energy-matching** L1 on denormalized total deposited energy (`--lambda-energy`).
- **Data:** stratified **80/10/10** train/val/test on label `y`; Parquet loader **streams batches** (shards have one row group each) and materializes only requested indices; optional **`--memmap-train`** and parallel materialization via `JET_SR_MATERIALIZE_WORKERS`.
- **Training logs:** `history.csv` with `val_l1`, `val_bicubic_l1`, `val_energy_mae`; checkpoints `checkpoint_best.pt` / `checkpoint_last.pt`.
- **Evaluation:** `evaluate.py` writes JSON (e.g. `metrics_report.json`) with SR vs **bicubic** baseline: PSNR/SSIM, energy ratio & MAE, radial-profile L1, linear probe on channel means.

### Where to look

| Item | Location |
|------|----------|
| Code | [`super_resolution/`](super_resolution/) (`data.py`, `models.py`, `train.py`, `evaluate.py`) |
| Usage & commands | [`super_resolution/README.md`](super_resolution/README.md) |
| Model / optimization discussion | [`super_resolution/MODEL_DISCUSSION.md`](super_resolution/MODEL_DISCUSSION.md) |
| Reference run notes | [`super_resolution/runs/full_run/RUN_NOTES.md`](super_resolution/runs/full_run/RUN_NOTES.md) |
| Energy ablation notes | [`super_resolution/runs/ablation_no_energy/RUN_NOTES.md`](super_resolution/runs/ablation_no_energy/RUN_NOTES.md) |
| Example metrics | `super_resolution/runs/full_run/metrics_report.json`, `super_resolution/runs/ablation_no_energy/metrics_report.json` |

### Reference metrics (illustrative)

On the **full test split** ($n=13\,932$), the checked-in **full_run** report shows SR vs bicubic **similar PSNR/SSIM**, while **energy MAE** is ~**1.2** (SR) vs ~**21.3** (bicubic); radial profile L1 vs HR is lower for SR than for bicubic. The **no-energy ablation** shows much worse energy scale (see its `metrics_report.json`). Details and tables are in the PDF.

### Data

Parquet shards from the task CERNBox link are **not** committed (large `.parquet` is gitignored). Place them in a directory and point `--data-dir` at it.

---

## Task 2c — Agentic AI for HEP analyses (HEPTAPOD)

### What was done

- Work lives on a **fork** of upstream HEPTAPOD (this repo does **not** vendor the clone; see `.gitignore` entry for `heptapod/`).
- Branch **`gsoc26-cms-restructure`** adds a **parallel layout** `cms_agent/` for CMS-oriented agents: agents contract, workflows / run-card examples, **tool inventory** (`TOOL_INVENTORY.md`), configs (allowlists), knowledge stubs, eval fixtures, and a **minimal NumPy histogram adapter** so contracts can be tested without CMSSW/coffea.
- Top-level narrative: **`ARCHITECTURE_CMS_AGENT.md`** in the fork (orchestration vs existing `tools/`, links to Orchestral AI, MCP, paper arXiv:2512.15867).
- **Per evaluation instructions:** do **not** open a PR to **upstream** HEPTAPOD unless the program asks.

### Clone the fork

```bash
git clone -b gsoc26-cms-restructure https://github.com/KenWuqianghao/heptapod.git heptapod
```

Upstream for comparison: [tonymenzo/heptapod](https://github.com/tonymenzo/heptapod).

### Where to look (after cloning)

| Item | Location |
|------|----------|
| Architecture overview | `heptapod/ARCHITECTURE_CMS_AGENT.md` |
| CMS scaffold | `heptapod/cms_agent/` |
| Tool inventory | `heptapod/cms_agent/TOOL_INVENTORY.md` |

---

## Large files and secrets

- **Parquet** datasets are gitignored.
- **HEPTAPOD** clone is gitignored; use your fork next to this project.
- Do not commit API keys or personal handoff folders (see `.gitignore`).
