"""Dataset access and the stratified subsampling that keeps ESM-2 affordable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "validation", "test")


def load_config(path="config.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(path).resolve().parent
    for key in ("dataset_dir", "baseline_dir", "cache_dir", "results_dir"):
        cfg[key] = (root / cfg[key]).resolve()
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def load_table(dataset_dir: Path, split: str) -> pd.DataFrame:
    cols = ["fragment_id", "replicon_id", "assembly_accession", "replicon_type",
            "label", "length", "cluster_id", "gc_content"]
    return pd.read_csv(Path(dataset_dir) / f"{split}.csv", usecols=cols)


def stratified_subsample(table: pd.DataFrame, wanted: dict, seed: int) -> pd.DataFrame:
    """Sample each class separately and attach the weight each row stands for.

    A row sampled from a class of N members into a sample of n carries weight N/n, so
    weighted metrics estimate what the metric would be on the whole split.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for label, key in ((1, "plasmid"), (0, "chromosome")):
        pool = table.index[table["label"] == label].to_numpy()
        take = min(int(wanted[key]), len(pool))
        picked = np.sort(rng.choice(pool, size=take, replace=False))
        part = table.loc[picked].copy()
        part["weight"] = len(pool) / take
        part["row_in_split"] = picked
        parts.append(part)
    out = pd.concat(parts).sort_values("row_in_split").reset_index(drop=True)
    return out


def read_fasta_subset(fasta_path: Path, wanted_ids) -> dict[str, str]:
    """Stream the FASTA once and keep only the requested records."""
    wanted = set(wanted_ids)
    found, name, parts = {}, None, []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name in wanted:
                    found[name] = "".join(parts)
                name, parts = line[1:].split()[0].strip(), []
            else:
                parts.append(line.strip())
    if name in wanted:
        found[name] = "".join(parts)
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} fragment ids not found in {fasta_path.name}")
    return found


def save_embeddings(path: Path, table: pd.DataFrame, embeddings: list[np.ndarray]):
    counts = np.array([e.shape[0] for e in embeddings], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez(path,
             data=np.concatenate(embeddings, axis=0).astype(np.float32),
             offsets=offsets,
             # pandas returns object arrays for string columns; cast so the npz
             # stays readable without pickle.
             fragment_id=table["fragment_id"].to_numpy().astype(str),
             label=table["label"].to_numpy().astype(np.int64),
             weight=table["weight"].to_numpy().astype(np.float64),
             row_in_split=table["row_in_split"].to_numpy().astype(np.int64),
             replicon_id=table["replicon_id"].to_numpy().astype(str))


class EmbeddingSet:
    """Ragged (n_orfs, hidden) embeddings for a set of fragments."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.data = d["data"]
        self.offsets = d["offsets"]
        self.fragment_id = d["fragment_id"].astype(str)
        self.label = d["label"]
        self.weight = d["weight"]
        self.row_in_split = d["row_in_split"]
        self.replicon_id = d["replicon_id"].astype(str)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i) -> np.ndarray:
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    @property
    def n_orfs(self) -> np.ndarray:
        return np.diff(self.offsets)
