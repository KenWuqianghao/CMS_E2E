# full_run (host-constrained)

- **Disk**: ~7 GB free prevented full memmap materialization (~24 GB) for all ~111k training indices.
- **Training data**: `--max-train-samples 16000` and `--max-val-samples 4000` (stratified split unchanged; only materialized prefix).
- **Epochs**: **8** (not 20) so training + eval + ablation finish on this machine in one session; increase locally when you have time.

Re-run without caps when you have **≥25 GB RAM or disk** for memmap: omit `--max-*`, add `--memmap-train`, and use `--epochs 20` per README.
