# CMS_E2E workspace

Artifacts for **ML4SCI GSoC 2026** evaluation tasks **2b** (jet super-resolution GAN) and **2c** (HEPTAPOD CMS agent layout).

## Super-resolution (2b)

Code and documentation live in [`super_resolution/`](super_resolution/README.md). Reference **2b** checkpoints and `metrics_report.json` files are under `super_resolution/runs/full_run/` and `super_resolution/runs/ablation_no_energy/` (see per-run `RUN_NOTES.md`); the README distinguishes **host-constrained** training from optional **full-scale** runs.

## Agentic AI / HEPTAPOD (2c)

Task **2c** work uses **HEPTAPOD**. This GitHub repo does **not** include the `heptapod/` tree (it is a large nested clone with its own history). Clone it next to this project:

```bash
git clone -b gsoc26-cms-restructure https://github.com/tonymenzo/heptapod.git heptapod
```

Then read `heptapod/ARCHITECTURE_CMS_AGENT.md` and `heptapod/cms_agent/`. **Do not open a pull request** to the main HEPTAPOD repository for the evaluation—keep changes in your local clone or fork.

## Large data files

Parquet shards from the task CERNBox link are **not** committed here; keep them alongside this repo as you already do locally.
