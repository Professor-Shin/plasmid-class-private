"""Dataset access, subsampling, and the embedding cache.

Two things changed when this moved to a GPU server, both backward compatible:

* ``subsample`` may now be ``None``, meaning "use the whole split". The laptop had
  to sample each class and carry an importance weight per row; with the full split
  every weight is 1.0 and the weighted metrics in metrics.py reduce to ordinary
  ones without any change there.

* the cache has a second, memory-mappable format. The original single ``.npz``
  per split is read back by ``np.load`` into RAM in one piece, which is fine for
  the ~2,000-fragment samples the laptop could embed but not for the full
  112,929 fragments with the ORF cap removed (~3 GB of embeddings). The new format
  is a plain ``.npy`` matrix read with ``mmap_mode="r"`` plus a small metadata
  ``.npz``. Both formats are still readable, so every cache file produced on the
  laptop keeps working.
"""

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


def full_split(table: pd.DataFrame) -> pd.DataFrame:
    """Every row of the split, with the weight and index columns a sample carries.

    Weight 1.0 is not a placeholder: it is the correct importance weight when
    nothing was sampled away, so the weighted metrics stay valid and the two
    regimes are directly comparable.
    """
    out = table.copy()
    out["weight"] = 1.0
    out["row_in_split"] = np.arange(len(out), dtype=np.int64)
    return out.reset_index(drop=True)


def stratified_subsample(table: pd.DataFrame, wanted: dict | None, seed: int) -> pd.DataFrame:
    """Sample each class separately and attach the weight each row stands for.

    A row sampled from a class of N members into a sample of n carries weight N/n, so
    weighted metrics estimate what the metric would be on the whole split.
    ``wanted=None`` skips sampling entirely and returns the full split.
    """
    if wanted is None:
        return full_split(table)
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


def iter_fasta_subset(fasta_path: Path, wanted_ids):
    """Yield (id, sequence) for the requested records without holding them all.

    read_fasta_subset builds a dict of every sequence it was asked for, which is
    1.7 GB when the request is a whole split. The embedding pass only needs one
    fragment at a time, so it iterates instead; the records come out in FASTA
    order, which is the split's own row order.
    """
    wanted = set(wanted_ids)
    name, parts, seen = None, [], 0
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name in wanted:
                    yield name, "".join(parts)
                    seen += 1
                name, parts = line[1:].split()[0].strip(), []
            else:
                parts.append(line.strip())
    if name in wanted:
        yield name, "".join(parts)
        seen += 1
    if seen != len(wanted):
        raise ValueError(f"{len(wanted) - seen} fragment ids not found in "
                         f"{Path(fasta_path).name}")


# --------------------------------------------------------------------------- #
# the embedding cache
# --------------------------------------------------------------------------- #

def embedding_paths(cache_dir: Path, split: str, shard: int | None = None) -> dict:
    """Where one split's (or one shard's) cache files live, in both formats."""
    tag = split if shard is None else f"{split}.shard{shard:04d}"
    base = Path(cache_dir)
    return {
        "tag": tag,
        "legacy": base / f"emb_{tag}.npz",
        "data": base / f"emb_{tag}.data.npy",
        "meta": base / f"emb_{tag}.meta.npz",
    }


def is_cached(cache_dir: Path, split: str, shard: int | None = None) -> bool:
    p = embedding_paths(cache_dir, split, shard)
    return p["legacy"].exists() or (p["data"].exists() and p["meta"].exists())


META_KEYS = ("offsets", "fragment_id", "label", "weight", "row_in_split",
             "replicon_id", "length", "n_orfs_real", "length_bucket",
             "aux", "aux_names")


def _meta_payload(table: pd.DataFrame, counts: np.ndarray, offsets: np.ndarray,
                  aux, aux_names, n_orfs_real) -> dict:
    # `counts` is 1 for an ORF-free fragment because a zero vector stands in for
    # it, so the true per-fragment ORF count has to be passed in separately.
    real = counts if n_orfs_real is None else np.asarray(n_orfs_real, dtype=np.int64)
    payload = dict(
        offsets=offsets,
        # pandas returns object arrays for string columns; cast so the npz stays
        # readable without pickle.
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
        payload["aux"] = np.asarray(aux, dtype=np.float32)
        payload["aux_names"] = np.asarray(aux_names or [], dtype=str)
    return payload


def save_embeddings(path: Path, table: pd.DataFrame, embeddings: list[np.ndarray],
                    aux: np.ndarray | None = None, aux_names: list[str] | None = None,
                    n_orfs_real: np.ndarray | None = None):
    """The original single-file format, kept so old cache files stay reproducible."""
    counts = np.array([e.shape[0] for e in embeddings], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    payload = _meta_payload(table, counts, offsets, aux, aux_names, n_orfs_real)
    payload["data"] = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.savez(path, **payload)


def save_embeddings_mmap(paths: dict, table: pd.DataFrame, data: np.ndarray,
                         counts: np.ndarray, aux=None, aux_names=None,
                         n_orfs_real=None):
    """Matrix to a plain .npy (memory-mappable), everything else to a small .npz.

    `data` is the already-concatenated (total_orfs, hidden) matrix and `counts` the
    per-fragment row count, so a caller streaming fragment by fragment never has
    to hold a list of per-fragment arrays.
    """
    offsets = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))])
    if offsets[-1] != len(data):
        raise ValueError(f"counts sum to {offsets[-1]} rows but data has {len(data)}")
    payload = _meta_payload(table, np.asarray(counts, dtype=np.int64), offsets,
                            aux, aux_names, n_orfs_real)
    payload["dtype"] = np.asarray(str(data.dtype))
    payload["hidden"] = np.asarray(data.shape[1])
    np.save(paths["data"], data)
    np.savez(paths["meta"], **payload)


class EmbeddingSet:
    """Ragged (n_orfs, hidden) embeddings for a set of fragments.

    Accepts either format: a legacy ``emb_{split}.npz`` path, or the dict returned
    by ``embedding_paths``. In the new format the matrix is memory-mapped, so a
    3 GB cache costs no resident memory until rows are actually touched.
    """

    def __init__(self, source):
        if isinstance(source, dict):
            paths = source
        else:
            source = Path(source)
            paths = {"legacy": source,
                     "data": source.with_suffix("").with_suffix(".data.npy"),
                     "meta": source.with_suffix("").with_suffix(".meta.npz")}

        if paths.get("data") and Path(paths["data"]).exists():
            self.path = Path(paths["data"])
            self.mmapped = True
            self.data = np.load(paths["data"], mmap_mode="r")
            meta = np.load(paths["meta"], allow_pickle=False)
        elif paths.get("legacy") and Path(paths["legacy"]).exists():
            self.path = Path(paths["legacy"])
            self.mmapped = False
            meta = np.load(paths["legacy"], allow_pickle=True)
            self.data = meta["data"]
        else:
            raise FileNotFoundError(
                f"no embedding cache at {paths.get('data')} or {paths.get('legacy')}")

        self.offsets = meta["offsets"]
        self.fragment_id = meta["fragment_id"].astype(str)
        self.label = meta["label"]
        self.weight = meta["weight"]
        self.row_in_split = meta["row_in_split"]
        self.replicon_id = meta["replicon_id"].astype(str)
        self.length = meta["length"] if "length" in meta.files else None
        self.length_bucket = (meta["length_bucket"].astype(str)
                              if "length_bucket" in meta.files else None)
        # Fragments with no ORF carry one zero vector, so n_orfs would read 1 for
        # them; n_orfs_real keeps the true count for the ORF-free breakdown.
        self.n_orfs_real = meta["n_orfs_real"] if "n_orfs_real" in meta.files else None
        self.aux = meta["aux"] if "aux" in meta.files else None
        self.aux_names = (list(meta["aux_names"].astype(str))
                          if "aux_names" in meta.files else [])

    def close(self):
        """Release the memory map, so the file can be replaced or deleted.

        Needed only when something is about to remove the cache files - on
        Windows an open mapping blocks the unlink outright, and everywhere else
        it leaks a descriptor.
        """
        if getattr(self, "mmapped", False):
            self.data = None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i) -> np.ndarray:
        return np.asarray(self.data[self.offsets[i]:self.offsets[i + 1]])

    @property
    def hidden_size(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_orfs(self) -> np.ndarray:
        return np.diff(self.offsets)


def open_split(cache_dir: Path, split: str) -> EmbeddingSet:
    """Load a split's cache in whichever format is present."""
    return EmbeddingSet(embedding_paths(cache_dir, split))
