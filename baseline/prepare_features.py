"""Step 1: encode the dataset FASTA files and precompute every k-mer matrix.

Writes, into cache/:
  seq_{split}.u8              base codes, (n_fragments, 5000) uint8
  feat_{split}_k{K}{f|c}.u16  k-mer counts, (n_fragments, n_features) uint16
  train_perm.npy              row order used for the train matrices
  labels_{split}.npy          labels aligned to those rows
  tfidf_k{K}.npy / zou_scaler.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import data, features  # noqa: E402

SPECS = [(5, False), (6, False), (7, False), (6, True)]


def feature_path(cache: Path, split: str, k: int, canonical: bool) -> Path:
    return cache / f"feat_{split}_k{k}{'c' if canonical else 'f'}.u16"


def verify_canonical_order(reference: Path):
    colmap = features.canonical_colmap(6)
    columns = np.array([np.flatnonzero(colmap == c)[0] for c in range(colmap.max() + 1)])
    mine = features.kmer_strings(6, columns)
    kmc = reference.read_text().split()
    if mine != kmc:
        raise SystemExit("canonical 6-mer column order does not match KMC's kmer_order.csv")
    print(f"  canonical 6-mer order matches KMC reference ({len(kmc)} columns)")


def main():
    cfg = data.load_config(Path(__file__).resolve().parent / "config.yaml")
    cache, frag_len = cfg["cache_dir"], cfg["fragment_length"]
    rng = np.random.default_rng(cfg["random_seed"])

    print("verifying k-mer column order")
    verify_canonical_order(Path(__file__).resolve().parent / "reference" / "kmc_kmer_order.csv")

    tables = {}
    for split in data.SPLITS:
        t0 = time.time()
        tables[split] = data.load_table(cfg["dataset_dir"], split)
        data.encode_split(cfg["dataset_dir"], cache, split, frag_len, tables[split])
        print(f"encoded {split}: {len(tables[split])} fragments "
              f"({(tables[split]['label'] == 1).sum()} plasmid) in {time.time() - t0:.1f}s")

    perm = rng.permutation(len(tables["train"]))
    np.save(cache / "train_perm.npy", perm)
    np.save(cache / "labels_train.npy", tables["train"]["label"].to_numpy()[perm].astype(np.int64))
    for split in ("validation", "test"):
        np.save(cache / f"labels_{split}.npy", tables[split]["label"].to_numpy().astype(np.int64))

    for k, canonical in SPECS:
        n_cols = features.n_features(k, canonical)
        print(f"k={k} canonical={canonical} -> {n_cols} features")
        for split in data.SPLITS:
            t0 = time.time()
            out = feature_path(cache, split, k, canonical)
            existed = out.with_suffix('.json').exists()   # meta is written only on success
            features.build_features(
                cache / f"seq_{split}.u8", len(tables[split]), frag_len, k, canonical, out,
                order=perm if split == "train" else None,
                chunk_rows=features.safe_chunk_rows(n_cols, frag_len),
            )
            print(f"  {split}: {out.stat().st_size / 2**30:.2f} GiB"
                  f"{' (cached)' if existed else f' in {time.time() - t0:.1f}s'}")

        n_train = len(tables["train"])
        train_path = feature_path(cache, "train", k, canonical)
        if canonical:
            scaler = features.StandardTransform.fit(train_path, n_train, n_cols, cfg["block_bytes"])
            scaler.save(cache / "zou_scaler.npz")
            print("  fitted StandardScaler on train")
        else:
            tfidf = features.TfidfTransform.fit(train_path, n_train, n_cols, cfg["block_bytes"])
            tfidf.save(cache / f"tfidf_k{k}.npy")
            print("  fitted TF-IDF (smooth_idf=False) on train")

    # PlasClass concatenates canonical k-mer frequencies for k=3..7. Its scaler is
    # fitted on the assembled matrix in run_plasclass.py, not per k.
    extra = [k for k in cfg["plasclass"]["ks"] if (k, True) not in SPECS]
    print(f"PlasClass canonical k-mer features (k={extra} in addition to k=6)")
    for k in extra:
        n_cols = features.n_features(k, True)
        for split in data.SPLITS:
            t0 = time.time()
            out = feature_path(cache, split, k, True)
            existed = out.with_suffix('.json').exists()   # meta is written only on success
            features.build_features(
                cache / f"seq_{split}.u8", len(tables[split]), frag_len, k, True, out,
                order=perm if split == "train" else None,
                chunk_rows=features.safe_chunk_rows(n_cols, frag_len),
            )
            print(f"  k={k} {split}: {n_cols} features, {out.stat().st_size / 2**30:.2f} GiB"
                  f"{' (cached)' if existed else f' in {time.time() - t0:.1f}s'}")


if __name__ == "__main__":
    main()
