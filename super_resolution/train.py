"""Train SRGAN on CMS jet LR->HR pairs."""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import JetSRDataset, load_all_labels, stratified_split_indices
from models import PatchDiscriminator, SRGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _pick_device(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_subset_indices(split_indices: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples >= len(split_indices):
        return split_indices
    return split_indices[:max_samples]


def denorm(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return x * s + m


def batch_total_energy(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(x, 0.0).sum(dim=(1, 2, 3))


def append_epoch_log(csv_path: Path, row: dict[str, float | int]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = _pick_device(args)

    data_dir = os.path.abspath(args.data_dir)
    print(f"[startup] device={device} data_dir={data_dir}", flush=True)
    print("[startup] loading labels...", flush=True)
    labels = load_all_labels(data_dir)
    print(f"[startup] loaded {len(labels)} labels", flush=True)
    print("[startup] building stratified split...", flush=True)
    split = stratified_split_indices(
        labels,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(
        f"[startup] split sizes train={len(split.train)} val={len(split.val)} test={len(split.test)}",
        flush=True,
    )
    train_ix = iter_subset_indices(split.train, args.max_train_samples)
    val_ix = iter_subset_indices(split.val, args.max_val_samples)
    print(
        f"[startup] selected subsets train={len(train_ix)} val={len(val_ix)}",
        flush=True,
    )

    train_mmap_dir = (
        str(Path(args.out_dir) / "train_materialized") if args.memmap_train else None
    )
    print("[startup] materializing training subset...", flush=True)
    if train_mmap_dir:
        print(f"[startup] using on-disk memmap under {train_mmap_dir}", flush=True)
    train_ds = JetSRDataset(
        data_dir,
        index_subset=train_ix,
        memmap_dir=train_mmap_dir,
        memmap_prefix="train",
    )
    mean, std = train_ds.channel_mean, train_ds.channel_std
    print("[startup] materializing validation subset...", flush=True)
    val_ds = JetSRDataset(data_dir, index_subset=val_ix, channel_mean=mean, channel_std=std)
    print("[startup] datasets ready; creating loaders...", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    G = SRGenerator(in_ch=3, feats=args.g_feats, n_res=args.n_res).to(device)
    D = PatchDiscriminator(in_ch=3, feats=args.d_feats, n_classes=2).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    amp_enabled = args.amp and device.type == "cuda"
    scaler_g = amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = amp.GradScaler("cuda", enabled=amp_enabled)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            # --- Update D ---
            with amp.autocast("cuda", enabled=amp_enabled):
                with torch.no_grad():
                    fake = G(lr)
                logits_real = D(hr, y)
                logits_fake = D(fake.detach(), y)
                d_loss = torch.mean((logits_real - 1) ** 2) + torch.mean(logits_fake**2)

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # --- Update G ---
            with amp.autocast("cuda", enabled=amp_enabled):
                fake = G(lr)
                logits_fake = D(fake, y)
                adv = torch.mean((logits_fake - 1) ** 2)
                l1 = F.l1_loss(fake, hr)
                fake_denorm = denorm(fake, mean, std)
                hr_denorm = denorm(hr, mean, std)
                energy_fake = batch_total_energy(fake_denorm)
                energy_hr = batch_total_energy(hr_denorm)
                energy_loss = F.l1_loss(energy_fake, energy_hr)
                g_loss = (
                    args.lambda_adv * adv
                    + args.lambda_l1 * l1
                    + args.lambda_energy * energy_loss
                )

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            pbar.set_postfix(
                d=float(d_loss.item()),
                g=float(g_loss.item()),
                l1=float(l1.item()),
                e=float(energy_loss.item()),
            )

        # validation metrics against both HR and bicubic baseline
        G.eval()
        val_l1 = 0.0
        val_energy_mae = 0.0
        val_bicubic_l1 = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                lr = batch["lr"].to(device, non_blocking=True)
                hr = batch["hr"].to(device, non_blocking=True)
                fake = G(lr)
                bicubic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
                val_l1 += F.l1_loss(fake, hr).item() * lr.size(0)
                val_bicubic_l1 += F.l1_loss(bicubic, hr).item() * lr.size(0)
                energy_fake = batch_total_energy(denorm(fake, mean, std))
                energy_hr = batch_total_energy(denorm(hr, mean, std))
                val_energy_mae += F.l1_loss(energy_fake, energy_hr).item() * lr.size(0)
                n += lr.size(0)
        val_l1 /= max(n, 1)
        val_bicubic_l1 /= max(n, 1)
        val_energy_mae /= max(n, 1)
        print(
            f"epoch {epoch} val_l1 {val_l1:.6f} "
            f"val_bicubic_l1 {val_bicubic_l1:.6f} "
            f"val_energy_mae {val_energy_mae:.6f}"
        )
        append_epoch_log(
            history_path,
            {
                "epoch": epoch,
                "val_l1": val_l1,
                "val_bicubic_l1": val_bicubic_l1,
                "val_energy_mae": val_energy_mae,
            },
        )
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(
                {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "channel_mean": mean,
                    "channel_std": std,
                    "epoch": epoch,
                    "val_l1": val_l1,
                    "val_bicubic_l1": val_bicubic_l1,
                    "val_energy_mae": val_energy_mae,
                    "split_sizes": {
                        "train": int(len(train_ix)),
                        "val": int(len(val_ix)),
                        "test": int(len(split.test)),
                    },
                    "args": vars(args),
                },
                out_dir / "checkpoint_best.pt",
            )
        torch.save(
            {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "channel_mean": mean,
                "channel_std": std,
                "epoch": epoch,
                "split_sizes": {
                    "train": int(len(train_ix)),
                    "val": int(len(val_ix)),
                    "test": int(len(split.test)),
                },
                "args": vars(args),
            },
            out_dir / "checkpoint_last.pt",
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CMS jet SRGAN training")
    p.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing *LR*.parquet shards",
    )
    p.add_argument("--out-dir", type=str, default="sr_runs/run1")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=4e-4)
    p.add_argument("--lambda-adv", type=float, default=1e-3)
    p.add_argument("--lambda-l1", type=float, default=1.0)
    p.add_argument("--lambda-energy", type=float, default=0.05)
    p.add_argument("--g-feats", type=int, default=64)
    p.add_argument("--d-feats", type=int, default=64)
    p.add_argument("--n-res", type=int, default=8)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    p.add_argument(
        "--memmap-train",
        action="store_true",
        help="Stream training tensors to np.memmap under out-dir/train_materialized (~25GB for full CMS split) to fit RAM.",
    )
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
