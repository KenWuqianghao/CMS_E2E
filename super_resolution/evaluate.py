"""Evaluate SR model: PSNR, SSIM, simple physics summaries, optional linear probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data import JetSRDataset, load_all_labels, stratified_split_indices
from models import SRGenerator


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    m = mean.view(1, 3, 1, 1).to(x.device)
    s = std.view(1, 3, 1, 1).to(x.device)
    return x * s + m


def total_energy(img_chw: np.ndarray) -> float:
    """Sum over channels and pixels (ROI energy proxy)."""
    return float(np.sum(np.maximum(img_chw, 0.0)))


def radial_profile(img_chw: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Mean intensity in annuli around center (per channel averaged)."""
    _, h, w = img_chw.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max() + 1e-6
    bins = (r / r_max * n_bins).astype(np.int32).clip(0, n_bins - 1)
    gray = img_chw.mean(axis=0)
    prof = np.array([gray[bins == k].mean() if np.any(bins == k) else 0.0 for k in range(n_bins)])
    return prof


def summarize_distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def classify_energy_ratio(mean_ratio: float, tolerance: float) -> str:
    if abs(mean_ratio - 1.0) <= tolerance:
        return "good"
    if mean_ratio > 1.0:
        return "high_bias"
    return "low_bias"


def _pick_device(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_eval(args: argparse.Namespace) -> None:
    device = _pick_device(args)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mean_np = ckpt["channel_mean"]
    std_np = ckpt["channel_std"]
    mean = torch.from_numpy(np.asarray(mean_np, dtype=np.float32))
    std = torch.from_numpy(np.asarray(std_np, dtype=np.float32))
    arch = ckpt.get("args") or {}
    g_feats = int(arch.get("g_feats", args.g_feats))
    n_res = int(arch.get("n_res", args.n_res))

    G = SRGenerator(in_ch=3, feats=g_feats, n_res=n_res).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    data_dir = os.path.abspath(args.data_dir)
    labels = load_all_labels(data_dir)
    split = stratified_split_indices(
        labels,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    test_ix = split.test
    if args.max_test_samples is not None:
        test_ix = test_ix[: args.max_test_samples]

    ds = JetSRDataset(
        data_dir,
        index_subset=test_ix,
        channel_mean=mean_np,
        channel_std=std_np,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    psnrs: list[float] = []
    ssims: list[float] = []
    bicubic_psnrs: list[float] = []
    bicubic_ssims: list[float] = []
    e_hr: list[float] = []
    e_sr: list[float] = []
    e_bicubic: list[float] = []
    labels_list: list[int] = []
    profiles_hr: list[np.ndarray] = []
    profiles_sr: list[np.ndarray] = []
    profiles_bicubic: list[np.ndarray] = []
    feats_hr: list[np.ndarray] = []
    feats_sr: list[np.ndarray] = []
    feats_bicubic: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            y = batch["y"].cpu().numpy()
            fake = G(lr)
            bicubic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

            hr_d = denorm(hr, mean, std).cpu().numpy()
            sr_d = denorm(fake, mean, std).cpu().numpy()
            bc_d = denorm(bicubic, mean, std).cpu().numpy()

            for i in range(hr_d.shape[0]):
                h = np.clip(hr_d[i], 0.0, None)
                s = np.clip(sr_d[i], 0.0, None)
                b = np.clip(bc_d[i], 0.0, None)
                psnrs.append(
                    peak_signal_noise_ratio(h, s, data_range=max(h.max(), s.max(), 1e-6))
                )
                ssims.append(
                    structural_similarity(
                        h.transpose(1, 2, 0).mean(axis=2),
                        s.transpose(1, 2, 0).mean(axis=2),
                        data_range=max(h.max(), s.max(), 1e-6),
                    )
                )
                bicubic_psnrs.append(
                    peak_signal_noise_ratio(h, b, data_range=max(h.max(), b.max(), 1e-6))
                )
                bicubic_ssims.append(
                    structural_similarity(
                        h.transpose(1, 2, 0).mean(axis=2),
                        b.transpose(1, 2, 0).mean(axis=2),
                        data_range=max(h.max(), b.max(), 1e-6),
                    )
                )
                e_hr.append(total_energy(hr_d[i]))
                e_sr.append(total_energy(sr_d[i]))
                e_bicubic.append(total_energy(bc_d[i]))
                labels_list.append(int(y[i]))
                profiles_hr.append(radial_profile(h))
                profiles_sr.append(radial_profile(s))
                profiles_bicubic.append(radial_profile(b))
                feats_hr.append(h.mean(axis=(1, 2)))
                feats_sr.append(s.mean(axis=(1, 2)))
                feats_bicubic.append(b.mean(axis=(1, 2)))

    psnr_stats = summarize_distribution(psnrs)
    ssim_stats = summarize_distribution(ssims)
    bicubic_psnr_stats = summarize_distribution(bicubic_psnrs)
    bicubic_ssim_stats = summarize_distribution(bicubic_ssims)
    print("PSNR:", psnr_stats)
    print("SSIM:", ssim_stats)
    print("Bicubic PSNR:", bicubic_psnr_stats)
    print("Bicubic SSIM:", bicubic_ssim_stats)

    e_hr_a = np.asarray(e_hr)
    e_sr_a = np.asarray(e_sr)
    e_bc_a = np.asarray(e_bicubic)
    sr_energy_ratio = e_sr_a / np.maximum(e_hr_a, 1e-6)
    bicubic_energy_ratio = e_bc_a / np.maximum(e_hr_a, 1e-6)
    sr_energy_ratio_mean = float(sr_energy_ratio.mean())
    bicubic_energy_ratio_mean = float(bicubic_energy_ratio.mean())
    sr_energy_mae = float(np.mean(np.abs(e_sr_a - e_hr_a)))
    bicubic_energy_mae = float(np.mean(np.abs(e_bc_a - e_hr_a)))
    print("Total energy ratio mean (SR/HR):", sr_energy_ratio_mean)
    print("Total energy ratio mean (bicubic/HR):", bicubic_energy_ratio_mean)
    print("Energy MAE (SR vs HR):", sr_energy_mae)
    print("Energy MAE (bicubic vs HR):", bicubic_energy_mae)

    prof_hr = np.stack(profiles_hr).mean(axis=0)
    prof_sr = np.stack(profiles_sr).mean(axis=0)
    prof_bc = np.stack(profiles_bicubic).mean(axis=0)
    prof_l1 = float(np.mean(np.abs(prof_hr - prof_sr)))
    prof_l1_bicubic = float(np.mean(np.abs(prof_hr - prof_bc)))
    print("Mean abs diff of mean radial profile:", prof_l1)
    print("Mean abs diff of mean radial profile (bicubic):", prof_l1_bicubic)

    # Linear probe on channel means: can we still separate classes?
    Y = np.asarray(labels_list, dtype=np.int64)
    X_hr = np.stack(feats_hr)
    X_sr = np.stack(feats_sr)
    X_bicubic = np.stack(feats_bicubic)

    def acc_logreg(X: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return 0.5
        # simple manual split 80/20
        rng = np.random.default_rng(0)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_tr = int(0.8 * len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        # logistic regression via torch
        w = torch.zeros(X.shape[1] + 1, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        Xt = torch.from_numpy(Xtr.astype(np.float32))
        yt = torch.from_numpy(ytr.astype(np.float32))
        for _ in range(200):
            logits = (Xt * w[:-1]).sum(dim=1) + w[-1]
            loss = F.binary_cross_entropy_with_logits(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            Xe = torch.from_numpy(Xte.astype(np.float32))
            logits = (Xe * w[:-1]).sum(dim=1) + w[-1]
            pred = (torch.sigmoid(logits) > 0.5).long().numpy()
            return float((pred == yte).mean())

    print("Linear probe acc (HR mean features):", acc_logreg(X_hr, Y))
    print("Linear probe acc (SR mean features):", acc_logreg(X_sr, Y))
    print("Linear probe acc (bicubic mean features):", acc_logreg(X_bicubic, Y))

    summary = {
        "checkpoint": args.checkpoint,
        "n_test": len(psnrs),
        "psnr": psnr_stats,
        "ssim": ssim_stats,
        "bicubic_psnr": bicubic_psnr_stats,
        "bicubic_ssim": bicubic_ssim_stats,
        "energy_ratio_mean": sr_energy_ratio_mean,
        "energy_ratio_bias_flag": classify_energy_ratio(sr_energy_ratio_mean, args.energy_ratio_tolerance),
        "bicubic_energy_ratio_mean": bicubic_energy_ratio_mean,
        "energy_mae": sr_energy_mae,
        "bicubic_energy_mae": bicubic_energy_mae,
        "radial_profile_l1_mean": prof_l1,
        "bicubic_radial_profile_l1_mean": prof_l1_bicubic,
        "linear_probe_hr": acc_logreg(X_hr, Y),
        "linear_probe_sr": acc_logreg(X_sr, Y),
        "linear_probe_bicubic": acc_logreg(X_bicubic, Y),
    }

    if args.save_report:
        out = Path(args.save_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("Wrote", out)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SR evaluation metrics")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=".")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--g-feats", type=int, default=64)
    p.add_argument("--n-res", type=int, default=8)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--energy-ratio-tolerance", type=float, default=0.1)
    p.add_argument("--save-report", type=str, default=None)
    return p


if __name__ == "__main__":
    run_eval(build_argparser().parse_args())
