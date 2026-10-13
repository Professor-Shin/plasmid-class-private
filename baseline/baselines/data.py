"""Dataset access: FASTA -> encoded memmap, split tables, minibatch streaming."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "validation", "test")

# 4 marks any non-ACGT base; k-mer windows containing it are dropped, which is
# what Biostrings::oligonucleotideFrequency and KMC both do.
CODE_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    CODE_LUT[_b] = _i
    CODE_LUT[_b + 32] = _i


def load_config(path="config.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(path).resolve().parent
    for key in ("dataset_dir", "cache_dir", "results_dir"):
        cfg[key] = (root / cfg[key]).resolve()
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def load_table(dataset_dir: Path, split: str) -> pd.DataFrame:
    cols = ["fragment_id", "replicon_id", "assembly_accession", "replicon_type",
            "label", "length", "cluster_id", "gc_content"]
    path = Path(dataset_dir) / f"{split}.csv"
    # length_bucket exists only in the mixed-length dataset, so the fixed-5-kb
    # dataset keeps loading through this same path.
    available = pd.read_csv(path, nrows=0).columns
    if "length_bucket" in available:
        cols = cols + ["length_bucket"]
    return pd.read_csv(path, usecols=cols)


def iter_fasta(path: Path):
    name, parts = None, []
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    yield name, b"".join(parts)
                name, parts = line[1:].split()[0].decode(), []
            else:
                parts.append(line.strip())
    if name is not None:
        yield name, b"".join(parts)


def encode_split(dataset_dir: Path, cache_dir: Path, split: str, frag_len: int,
                 table: pd.DataFrame) -> Path:
    """Write an (n_fragments, frag_len) uint8 memmap of 0-3 base codes.

    This fixed-width staging buffer only works for a fixed-length dataset. On the
    mixed-length dataset the k-mer spectrum is counted per fragment instead (see
    run_composition.kmer_freq), because the spectrum's width is 4^k whatever the
    sequence length is - it is only this intermediate that needs one length.
    """
    out = Path(cache_dir) / f"seq_{split}.u8"
    meta = out.with_suffix(".json")
    n = len(table)
    if out.exists() and meta.exists() and json.loads(meta.read_text())["n"] == n:
        return out

    lengths = table["length"].to_numpy()
    if (lengths != frag_len).any():
        raise ValueError(
            f"{split}: encode_split needs every fragment to be {frag_len} bp, but the "
            f"dataset holds lengths {lengths.min()}-{lengths.max()}. Use the per-fragment "
            "k-mer path (run_composition.py) for a mixed-length dataset.")

    mm = np.memmap(out, dtype=np.uint8, mode="w+", shape=(n, frag_len))
    expected = table["fragment_id"].to_numpy()
    for i, (name, seq) in enumerate(iter_fasta(Path(dataset_dir) / f"{split}.fasta")):
        if name != expected[i]:
            raise ValueError(f"{split}: FASTA record {i} is {name}, CSV says {expected[i]}")
        mm[i] = CODE_LUT[np.frombuffer(seq, dtype=np.uint8)]
    if i + 1 != n:
        raise ValueError(f"{split}: FASTA holds {i + 1} records, CSV holds {n}")
    mm.flush()
    del mm
    meta.write_text(json.dumps({"n": n, "length": frag_len}))
    return out


def balanced_subset(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a 1:1 chromosome/plasmid subsample, as used by both papers."""
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    keep = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, keep])
    idx.sort()
    return idx


class MemmapStream:
    """Minibatches over a row-major uint16 feature memmap.

    Train features are stored in a pre-shuffled row order, so shuffling whole
    blocks plus rows inside a block approximates a full shuffle while keeping
    reads sequential - the machine cannot hold a 11 GB feature matrix in RAM.
    """

    def __init__(self, path, n_rows, n_cols, labels, batch_size,
                 block_bytes=67_108_864, subset=None, transform=None, seed=0):
        self.mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_rows, n_cols))
        self.batch_size = batch_size
        self.transform = transform
        self.rng = np.random.default_rng(seed)
        self.block_rows = max(batch_size, int(block_bytes // (n_cols * 2)))
        if subset is None:
            self.buf, self.labels, self.n = None, labels, n_rows
        else:
            self.buf = np.asarray(self.mm[subset])
            self.labels, self.n = labels[subset], len(subset)

    def _emit(self, X, y, order):
        for i in range(0, len(order), self.batch_size):
            sel = order[i:i + self.batch_size]
            xb = X[sel]
            yield (self.transform(xb) if self.transform else xb), y[sel]

    def epoch(self, shuffle=True):
        if self.buf is not None:
            order = self.rng.permutation(self.n) if shuffle else np.arange(self.n)
            yield from self._emit(self.buf, self.labels, order)
            return
        starts = list(range(0, self.n, self.block_rows))
        if shuffle:
            self.rng.shuffle(starts)
        for s in starts:
            block = np.asarray(self.mm[s:s + self.block_rows])
            y = self.labels[s:s + self.block_rows]
            order = self.rng.permutation(len(block)) if shuffle else np.arange(len(block))
            yield from self._emit(block, y, order)

    def forever(self):
        while True:
            yield from self.epoch()

    def steps_per_epoch(self) -> int:
        return int(np.ceil(self.n / self.batch_size))
