"""k-mer feature construction, TF-IDF (PlasFlow) and standardisation (Zou)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def revcomp_index_table(k: int) -> np.ndarray:
    idx = np.arange(4 ** k, dtype=np.int64)
    rc = np.zeros_like(idx)
    tmp = idx.copy()
    for _ in range(k):
        rc = rc * 4 + (3 - (tmp & 3))
        tmp >>= 2
    return rc


def canonical_colmap(k: int) -> np.ndarray:
    """Map every k-mer index to its canonical column, ordered alphabetically.

    Base-4 index order equals lexicographic k-mer order, so sorting the distinct
    canonical representatives reproduces the column order KMC emits.
    """
    idx = np.arange(4 ** k, dtype=np.int64)
    canon = np.minimum(idx, revcomp_index_table(k))
    uniq = np.unique(canon)
    dense = np.zeros(4 ** k, dtype=np.int32)
    dense[uniq] = np.arange(len(uniq), dtype=np.int32)
    return dense[canon]


def kmer_strings(k: int, columns: np.ndarray | None = None) -> list[str]:
    bases = "ACGT"
    out = []
    keep = np.arange(4 ** k) if columns is None else columns
    for i in keep:
        s = ""
        for j in range(k):
            s = bases[(int(i) >> (2 * j)) & 3] + s
        out.append(s)
    return out


def n_features(k: int, canonical: bool) -> int:
    if not canonical:
        return 4 ** k
    palindromes = 4 ** (k // 2) if k % 2 == 0 else 0
    return (4 ** k + palindromes) // 2


def safe_chunk_rows(n_cols: int, seq_len: int, budget_bytes: int = 100_000_000) -> int:
    """Rows per chunk that keep count_kmers within a memory budget.

    Cost is driven by two different things: the per-window index arrays, which scale
    with sequence length and not with k, and the bincount output, which scales with
    the column count. Sizing on columns alone blows up for small k.
    """
    per_row = 32 * seq_len + 8 * n_cols
    return max(32, int(budget_bytes // per_row))


def count_kmers(codes: np.ndarray, k: int, colmap: np.ndarray | None, n_cols: int) -> np.ndarray:
    """Counts per row; windows containing a non-ACGT base are skipped."""
    n, length = codes.shape
    width = length - k + 1
    idx = np.zeros((n, width), dtype=np.int32)
    bad = np.zeros((n, width), dtype=bool)
    for j in range(k):
        c = codes[:, j:j + width]
        bad |= c == 4
        idx *= 4
        idx += np.where(c == 4, 0, c).astype(np.int32)
    if colmap is not None:
        idx = colmap[idx]
    flat = idx.astype(np.int64) + (np.arange(n, dtype=np.int64) * n_cols)[:, None]
    flat = flat[~bad]
    counts = np.bincount(flat, minlength=n * n_cols).reshape(n, n_cols)
    return counts.astype(np.uint16)


def build_features(seq_path: Path, n_rows: int, frag_len: int, k: int, canonical: bool,
                   out_path: Path, order: np.ndarray | None = None,
                   chunk_rows: int = 512, log=print) -> Path:
    meta = out_path.with_suffix(".json")
    n_cols = n_features(k, canonical)
    if out_path.exists() and meta.exists():
        done = json.loads(meta.read_text())
        if done == {"n": n_rows, "cols": n_cols, "k": k, "canonical": canonical}:
            return out_path

    colmap = canonical_colmap(k) if canonical else None
    seq = np.memmap(seq_path, dtype=np.uint8, mode="r", shape=(n_rows, frag_len))
    feats = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n_rows, n_cols))
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        rows = np.asarray(seq[start:stop]) if order is None else np.asarray(seq[order[start:stop]])
        feats[start:stop] = count_kmers(rows, k, colmap, n_cols)
        if (start // chunk_rows) % 100 == 0:
            log(f"    {out_path.name}: {stop}/{n_rows}")
    feats.flush()
    del feats, seq
    meta.write_text(json.dumps({"n": n_rows, "cols": n_cols, "k": k, "canonical": canonical}))
    return out_path


def _iter_blocks(path: Path, n_rows: int, n_cols: int, block_bytes: int):
    mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_rows, n_cols))
    step = max(1, int(block_bytes // (n_cols * 2)))
    for s in range(0, n_rows, step):
        yield np.asarray(mm[s:s + step])


class TfidfTransform:
    """sklearn TfidfTransformer(smooth_idf=False) semantics: idf = ln(n/df) + 1, then L2."""

    def __init__(self, idf: np.ndarray):
        self.idf = idf.astype(np.float32)

    @classmethod
    def fit(cls, path, n_rows, n_cols, block_bytes):
        df = np.zeros(n_cols, dtype=np.int64)
        for block in _iter_blocks(path, n_rows, n_cols, block_bytes):
            df += (block > 0).sum(axis=0)
        idf = np.log(n_rows / np.maximum(df, 1)) + 1.0
        return cls(idf)

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        x = counts.astype(np.float32) * self.idf
        norm = np.sqrt((x * x).sum(axis=1, keepdims=True))
        np.maximum(norm, 1e-12, out=norm)
        return x / norm

    def save(self, path):
        np.save(path, self.idf)

    @classmethod
    def load(cls, path):
        return cls(np.load(path))


class StandardTransform:
    """sklearn StandardScaler semantics, including scale=1 for zero-variance columns."""

    def __init__(self, mean, scale):
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit(cls, path, n_rows, n_cols, block_bytes):
        total = np.zeros(n_cols, dtype=np.float64)
        total_sq = np.zeros(n_cols, dtype=np.float64)
        for block in _iter_blocks(path, n_rows, n_cols, block_bytes):
            b = block.astype(np.float64)
            total += b.sum(axis=0)
            total_sq += (b * b).sum(axis=0)
        mean = total / n_rows
        var = np.maximum(total_sq / n_rows - mean ** 2, 0.0)
        scale = np.sqrt(var)
        scale[scale == 0.0] = 1.0
        return cls(mean, scale)

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        return (counts.astype(np.float32) - self.mean) / self.scale

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["scale"])
