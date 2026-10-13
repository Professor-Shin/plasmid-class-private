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
    path = Path(dataset_dir) / f"{split}.csv"
    # length_bucket exists only in the mixed-length dataset, so the fixed-5-kb
    # dataset keeps loading through this same code path.
    if "length_bucket" in pd.read_csv(path, nrows=0).columns:
        cols = cols + ["length_bucket"]
    return pd.read_csv(path, usecols=cols)


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


def save_embeddings(path: Path, table: pd.DataFrame, embeddings: list[np.ndarray],
                    aux: np.ndarray | None = None, aux_names: list[str] | None = None,
                    n_orfs_real: np.ndarray | None = None):
    counts = np.array([e.shape[0] for e in embeddings], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    # `counts` is 1 for an ORF-free fragment because a zero vector stands in for it,
    # so the true per-fragment ORF count has to be passed in separately.
    real = counts if n_orfs_real is None else np.asarray(n_orfs_real, dtype=np.int64)
    payload = dict(
        data=np.concatenate(embeddings, axis=0).astype(np.float32),
        offsets=offsets,
        # pandas returns object arrays for string columns; cast so the npz
        # stays readable without pickle.
        fragment_id=table["fragment_id"].to_numpy().astype(str),
        label=table["label"].to_numpy().astype(np.int64),
        weight=table["weight"].to_numpy().astype(np.float64),
        row_in_split=table["row_in_split"].to_numpy().astype(np.int64),
        replicon_id=table["replicon_id"].to_numpy().astype(str),
        length=table["length"].to_numpy().astype(np.int64),
        n_orfs_real=real,
    )
    if "length_bucket" in table.columns:
        payload["length_bucket"] = table["length_bucket"].to_numpy().astype(str)
    if aux is not None:
        payload["aux"] = aux.astype(np.float32)
        payload["aux_names"] = np.asarray(aux_names or [], dtype=str)
    np.savez(path, **payload)


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
        self.length = d["length"] if "length" in d.files else None
        self.length_bucket = (d["length_bucket"].astype(str)
                              if "length_bucket" in d.files else None)
        # Fragments with no ORF carry one zero vector, so n_orfs would read 1 for
        # them; n_orfs_real keeps the true count for the ORF-free breakdown.
        self.n_orfs_real = d["n_orfs_real"] if "n_orfs_real" in d.files else None
        self.aux = d["aux"] if "aux" in d.files else None
        self.aux_names = (list(d["aux_names"].astype(str))
                          if "aux_names" in d.files else [])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i) -> np.ndarray:
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    @property
    def n_orfs(self) -> np.ndarray:
        return np.diff(self.offsets)
