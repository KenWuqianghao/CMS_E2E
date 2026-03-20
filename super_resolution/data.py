"""Load CMS jet image super-resolution pairs from ML4SCI parquet shards."""
from __future__ import annotations

import bisect
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def list_parquet_files(data_dir: str) -> list[str]:
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet") and "LR" in f
    )
    if not files:
        raise FileNotFoundError(
            f"No *LR*.parquet files under {data_dir}. "
            "Download the CERNBox dataset from the GSoC task PDF."
        )
    return files


def load_all_labels(data_dir: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for fp in list_parquet_files(data_dir):
        col = pq.read_table(fp, columns=["y"])["y"]
        parts.append(np.asarray(col, dtype=np.int64).reshape(-1))
    return np.concatenate(parts, axis=0)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split_indices(
    labels: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> SplitIndices:
    """Stratified split of row indices 0..N-1 by class label."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.")
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_parts.append(cls_idx[:n_train])
        val_parts.append(cls_idx[n_train : n_train + n_val])
        test_parts.append(cls_idx[n_train + n_val :])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndices(train=train, val=val, test=test)


class JetParquetIndex:
    """Maps flat row index -> parquet file and row offset within that file."""

    def __init__(self, data_dir: str) -> None:
        self._files = list_parquet_files(data_dir)
        self._cum_rows: list[int] = []
        t = 0
        for fp in self._files:
            n = int(pq.ParquetFile(fp).metadata.num_rows)
            t += n
            self._cum_rows.append(t)

    @property
    def total_rows(self) -> int:
        return int(self._cum_rows[-1])

    def resolve(self, flat_i: int) -> tuple[int, int]:
        """Return (file_index, row_within_file)."""
        j = bisect.bisect_right(self._cum_rows, flat_i)
        prev = self._cum_rows[j - 1] if j > 0 else 0
        row = flat_i - prev
        return j, row


def _assign_batch_rows(
    lr_out: np.ndarray,
    hr_out: np.ndarray,
    y_out: np.ndarray,
    out_pos_slice: np.ndarray,
    lr_col: pa.Array,
    hr_col: pa.Array,
    y_col: pa.Array,
) -> None:
    """Bulk convert taken Arrow columns into preallocated NumPy outputs."""
    lr_list = lr_col.to_pylist()
    hr_list = hr_col.to_pylist()
    op = out_pos_slice.astype(np.int64, copy=False)
    lr_out[op] = np.stack([np.asarray(x, dtype=np.float32) for x in lr_list], axis=0)
    hr_out[op] = np.stack([np.asarray(x, dtype=np.float32) for x in hr_list], axis=0)
    y_out[op] = np.asarray(y_col.to_pylist(), dtype=np.int64)


def _materialize_from_one_file(
    path: str,
    needed_rows: np.ndarray,
    out_positions: np.ndarray,
    lr_out: np.ndarray,
    hr_out: np.ndarray,
    y_out: np.ndarray,
    batch_read_rows: int,
    cols: list[str],
) -> None:
    """Stream one shard; fill lr_out/hr_out/y_out at disjoint out_positions."""
    needed_rows = np.asarray(needed_rows, dtype=np.int64)
    out_positions = np.asarray(out_positions, dtype=np.int64)
    pf = pq.ParquetFile(path)
    row_base = 0
    cursor = 0
    idx_type = pa.int32()
    for batch in pf.iter_batches(batch_size=batch_read_rows, columns=cols):
        batch_end = row_base + batch.num_rows
        start = cursor
        while start < len(needed_rows) and needed_rows[start] < row_base:
            start += 1
        end = start
        while end < len(needed_rows) and needed_rows[end] < batch_end:
            end += 1
        if end > start:
            local_rows = needed_rows[start:end] - row_base
            take_idx = pa.array(np.asarray(local_rows, dtype=np.int32), type=idx_type)
            sub = batch.take(take_idx)
            _assign_batch_rows(
                lr_out,
                hr_out,
                y_out,
                out_positions[start:end],
                sub.column(0),
                sub.column(1),
                sub.column(2),
            )
            cursor = end
        row_base += batch.num_rows
        if cursor >= len(needed_rows):
            break


def _materialize_workers_default() -> int:
    raw = os.environ.get("JET_SR_MATERIALIZE_WORKERS", "1").strip()
    try:
        n = max(1, int(raw))
    except ValueError:
        n = 1
    return n


def _materialize_flat_indices(
    data_dir: str,
    flat_indices: np.ndarray,
    batch_read_rows: int = 4096,
    materialize_workers: int | None = None,
    *,
    lr_out: np.ndarray | None = None,
    hr_out: np.ndarray | None = None,
    y_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream each Parquet file once in fixed-size record batches and copy only the
    requested rows into dense NumPy arrays (avoids loading full shards when only a
    subset is needed for training).

    ``batch_read_rows`` is larger than the Parquet default to reduce Python overhead
    in the scan loop; it does not change which Parquet pages are decoded for a given
    row, only how many rows are delivered per RecordBatch.

    Set ``materialize_workers`` > 1 or env ``JET_SR_MATERIALIZE_WORKERS`` to decode
    distinct shards in parallel (writes use disjoint output indices).
    """
    workers = (
        materialize_workers
        if materialize_workers is not None
        else _materialize_workers_default()
    )

    index = JetParquetIndex(data_dir)
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    n = len(flat_indices)
    if lr_out is None:
        lr_out = np.empty((n, 3, 64, 64), dtype=np.float32)
    if hr_out is None:
        hr_out = np.empty((n, 3, 125, 125), dtype=np.float32)
    if y_out is None:
        y_out = np.empty((n,), dtype=np.int64)
    if lr_out.shape != (n, 3, 64, 64) or hr_out.shape != (n, 3, 125, 125) or y_out.shape != (n,):
        raise ValueError("Preallocated outputs must match index_subset length and channel shapes.")

    by_file: dict[int, list[tuple[int, int]]] = {}
    for out_pos, flat in enumerate(flat_indices):
        fi, local = index.resolve(int(flat))
        by_file.setdefault(fi, []).append((local, out_pos))

    cols = ["X_jets_LR", "X_jets", "y"]
    file_jobs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for fi, pairs in by_file.items():
        path = index._files[fi]
        pairs.sort(key=lambda item: item[0])
        needed_rows = np.asarray([local_row for local_row, _ in pairs], dtype=np.int64)
        out_positions = np.asarray([out_pos for _, out_pos in pairs], dtype=np.int64)
        file_jobs.append((path, needed_rows, out_positions))

    if workers <= 1:
        for path, needed_rows, out_positions in file_jobs:
            _materialize_from_one_file(
                path,
                needed_rows,
                out_positions,
                lr_out,
                hr_out,
                y_out,
                batch_read_rows,
                cols,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _materialize_from_one_file,
                    path,
                    needed_rows,
                    out_positions,
                    lr_out,
                    hr_out,
                    y_out,
                    batch_read_rows,
                    cols,
                )
                for path, needed_rows, out_positions in file_jobs
            ]
            for fut in as_completed(futs):
                fut.result()
    return lr_out, hr_out, y_out


def _channel_mean_std_from_tensors(lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([lr.reshape(len(lr), 3, -1), hr.reshape(len(hr), 3, -1)], axis=2)
    x = x.astype(np.float64)
    mean = x.mean(axis=(0, 2)).astype(np.float32)
    std = x.std(axis=(0, 2)).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def _channel_mean_std_chunked(
    lr: np.ndarray,
    hr: np.ndarray,
    chunk_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Same statistics as _channel_mean_std_from_tensors, but in row chunks (for np.memmap)."""
    n = int(len(lr))
    n_pix = 64 * 64 + 125 * 125
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    for i in range(0, n, chunk_rows):
        sl = slice(i, min(i + chunk_rows, n))
        lc = lr[sl].astype(np.float64, copy=False).reshape(-1, 3, 64 * 64)
        hc = hr[sl].astype(np.float64, copy=False).reshape(-1, 3, 125 * 125)
        x = np.concatenate([lc, hc], axis=2)
        sum_c += x.sum(axis=(0, 2))
        sumsq_c += np.square(x).sum(axis=(0, 2))
    denom = float(n * n_pix)
    mean = (sum_c / denom).astype(np.float32)
    var = (sumsq_c / denom) - np.square((sum_c / denom).astype(np.float64))
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class JetSRDataset(Dataset):
    """LR (3,64,64), HR (3,125,125), label y in {0,1}. Subsets are materialized into RAM."""

    def __init__(
        self,
        data_dir: str,
        index_subset: np.ndarray | None = None,
        channel_mean: np.ndarray | None = None,
        channel_std: np.ndarray | None = None,
        batch_read_rows: int = 4096,
        materialize_workers: int | None = None,
        memmap_dir: str | Path | None = None,
        memmap_prefix: str = "data",
        stats_chunk_rows: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self._index_map: np.ndarray | None = None
        self._lr_mmap_path: str | None = None
        self._hr_mmap_path: str | None = None
        if index_subset is not None:
            idx = np.asarray(index_subset, dtype=np.int64)
            n = len(idx)
            workers_eff = materialize_workers
            if memmap_dir is not None:
                # np.memmap concurrent writes from multiple threads are unsafe.
                workers_eff = 1
                root = Path(memmap_dir)
                root.mkdir(parents=True, exist_ok=True)
                lr_path = str(root / f"{memmap_prefix}_lr.mmap")
                hr_path = str(root / f"{memmap_prefix}_hr.mmap")
                self._lr_mmap_path = lr_path
                self._hr_mmap_path = hr_path
                lr_mm = np.memmap(lr_path, dtype=np.float32, mode="w+", shape=(n, 3, 64, 64))
                hr_mm = np.memmap(hr_path, dtype=np.float32, mode="w+", shape=(n, 3, 125, 125))
                y_arr = np.empty((n,), dtype=np.int64)
                _materialize_flat_indices(
                    data_dir,
                    idx,
                    batch_read_rows=batch_read_rows,
                    materialize_workers=workers_eff,
                    lr_out=lr_mm,
                    hr_out=hr_mm,
                    y_out=y_arr,
                )
                lr_mm.flush()
                hr_mm.flush()
                self._lr = lr_mm
                self._hr = hr_mm
                self._y = y_arr
            else:
                self._lr, self._hr, self._y = _materialize_flat_indices(
                    data_dir,
                    idx,
                    batch_read_rows=batch_read_rows,
                    materialize_workers=materialize_workers,
                )
            self._index_map = idx
        else:
            raise ValueError("index_subset is required (full-scan Dataset not implemented).")

        if channel_mean is not None and channel_std is not None:
            self.channel_mean = np.asarray(channel_mean, dtype=np.float32)
            self.channel_std = np.asarray(channel_std, dtype=np.float32)
        else:
            if memmap_dir is not None:
                mean, std = _channel_mean_std_chunked(
                    self._lr, self._hr, chunk_rows=stats_chunk_rows
                )
            else:
                mean, std = _channel_mean_std_from_tensors(self._lr, self._hr)
            self.channel_mean = mean
            self.channel_std = std

    def __len__(self) -> int:
        return int(len(self._y))

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.channel_mean is None or self.channel_std is None:
            return x
        m = self.channel_mean.reshape(3, 1, 1)
        s = self.channel_std.reshape(3, 1, 1)
        return ((x - m) / s).astype(np.float32)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        lr_n = self._normalize(self._lr[i])
        hr_n = self._normalize(self._hr[i])
        y = int(self._y[i])
        flat = int(self._index_map[i]) if self._index_map is not None else i
        return {
            "lr": torch.from_numpy(lr_n),
            "hr": torch.from_numpy(hr_n),
            "y": torch.tensor(y, dtype=torch.long),
            "idx": torch.tensor(flat, dtype=torch.long),
        }
