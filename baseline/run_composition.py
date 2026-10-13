"""Composition baselines on the mixed-length dataset: how much is just GC?

Four logistic-regression models over the FULL dataset (no subsampling), each
answering a different part of the expert's second and third comments:

``gc_only``     4 GC features. The floor: how far does GC content alone get?
``ori_only``    8 origin-of-replication composition features alone.
``gc+ori``      both, i.e. the whole non-coding composition proxy.
``kmer5``       canonical 5-mer frequencies (512 dims), the classical
                composition baseline that PlasFlow/PlasClass are built on.
``kmer5+gc+ori``  whether the explicit GC/ori features add anything a k-mer
                spectrum does not already carry implicitly.

The feature definitions are imported from esm2/composition.py rather than
re-implemented, so the baselines and the ESM variants are provably scoring the
same quantities.

Everything streams: one pass over each split's FASTA builds the feature matrix,
which is small (512 floats per fragment) even though the sequences are not.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Single source of truth for the biological feature definitions.
sys.path.insert(0, str(HERE.parent / "esm2-attention"))
from baselines import data, features, metrics  # noqa: E402
from esm2 import composition  # noqa: E402

KMER_K = 5
SPLITS = ("train", "validation", "test")


def kmer_freq(codes: np.ndarray, k: int, colmap: np.ndarray, n_cols: int) -> np.ndarray:
    """Canonical k-mer frequencies for one variable-length fragment.

    Windows containing a non-ACGT base are skipped, matching what KMC and
    Biostrings::oligonucleotideFrequency do and what features.count_kmers does
    for the fixed-length case.
    """
    if len(codes) < k:
        return np.zeros(n_cols, dtype=np.float32)
    view = sliding_window_view(codes, k)
    clean = (view != 4).all(1)
    idx = np.zeros(view.shape[0], dtype=np.int64)
    for j in range(k):
        idx = idx * 4 + view[:, j]
    idx = idx[clean]
    if not len(idx):
        return np.zeros(n_cols, dtype=np.float32)
    counts = np.bincount(colmap[idx], minlength=n_cols).astype(np.float32)
    return counts / counts.sum()


def build_split(dataset_dir: Path, cache: Path, split: str, table: pd.DataFrame,
                colmap: np.ndarray, n_cols: int, log) -> tuple[np.ndarray, np.ndarray]:
    """Return (kmer matrix, composition matrix) aligned to `table`'s row order."""
    k_path, c_path = cache / f"comp_kmer{KMER_K}_{split}.npy", cache / f"comp_aux_{split}.npy"
    if k_path.exists() and c_path.exists():
        km, aux = np.load(k_path), np.load(c_path)
        if len(km) == len(table):
            log(f"  {split}: cached ({km.shape}, {aux.shape})")
            return km, aux

    expected = table["fragment_id"].to_numpy()
    km = np.zeros((len(table), n_cols), dtype=np.float32)
    aux = np.zeros((len(table), composition.feature_dim(composition.ALL_BLOCKS)),
                   dtype=np.float32)
    t0 = time.time()
    for i, (name, seq) in enumerate(data.iter_fasta(Path(dataset_dir) / f"{split}.fasta")):
        if i >= len(table):
            raise ValueError(f"{split}: FASTA has more records than the CSV")
        if name != expected[i]:
            raise ValueError(f"{split}: FASTA record {i} is {name}, CSV says {expected[i]}")
        codes = composition.CODE_LUT[np.frombuffer(seq, dtype=np.uint8)]
        km[i] = kmer_freq(codes, KMER_K, colmap, n_cols)
        aux[i] = composition.fragment_features(seq.decode("ascii", "replace"),
                                               composition.ALL_BLOCKS)
        if (i + 1) % 20000 == 0:
            log(f"  {split}: {i + 1}/{len(table)} ({time.time() - t0:.0f}s)")
    if i + 1 != len(table):
        raise ValueError(f"{split}: FASTA holds {i + 1} records, CSV holds {len(table)}")
    np.save(k_path, km)
    np.save(c_path, aux)
    log(f"  {split}: built in {time.time() - t0:.0f}s")
    return km, aux


def main():
    cfg = data.load_config(HERE / "config.yaml")
    dataset_dir, cache = Path(cfg["dataset_dir"]), cfg["cache_dir"]
    results = cfg["results_dir"]
    (results).mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    n_cols = features.n_features(KMER_K, True)
    colmap = features.canonical_colmap(KMER_K)
    log(f"canonical {KMER_K}-mer features: {n_cols}")

    tables, km, aux = {}, {}, {}
    for split in SPLITS:
        tables[split] = data.load_table(dataset_dir, split)
        if "length_bucket" not in tables[split].columns:
            raise SystemExit("dataset has no length_bucket column; rebuild step07 first")
        log(f"{split}: {len(tables[split])} fragments "
            f"({int((tables[split]['label'] == 1).sum())} plasmid)")
        km[split], aux[split] = build_split(dataset_dir, cache, split, tables[split],
                                            colmap, n_cols, log)

    gc_cols = composition.block_columns(["gc"])
    ori_cols = composition.block_columns(["ori"])
    names = composition.all_feature_names()
    designs = {
        "gc_only": lambda s: aux[s][:, gc_cols],
        "ori_only": lambda s: aux[s][:, ori_cols],
        "gc+ori": lambda s: aux[s],
        f"kmer{KMER_K}": lambda s: km[s],
        f"kmer{KMER_K}+gc+ori": lambda s: np.hstack([km[s], aux[s]]),
    }

    payload = {"dataset_dir": str(dataset_dir), "kmer_k": KMER_K,
               "aux_feature_names": names, "models": {}}
    y = {s: tables[s]["label"].to_numpy() for s in SPLITS}

    for name, build in designs.items():
        t0 = time.time()
        Xtr = build("train")
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        clf.fit(scaler.transform(Xtr), y["train"])

        proba = {s: clf.predict_proba(scaler.transform(build(s)))[:, 1] for s in SPLITS}
        tuned = metrics.best_f1_threshold(y["validation"], proba["validation"])
        entry = {
            "n_features": int(Xtr.shape[1]),
            "train_seconds": round(time.time() - t0, 1),
            "tuned_threshold": tuned,
            "test": metrics.binary_metrics(y["test"], proba["test"], 0.5),
            "test_tuned": metrics.binary_metrics(y["test"], proba["test"], tuned),
            "test_replicon": metrics.replicon_metrics(tables["test"], proba["test"], 0.5),
            "test_by_length": metrics.length_bucket_metrics(
                tables["test"]["length_bucket"], y["test"], proba["test"], tuned
            ).to_dict("records"),
        }
        if name in ("gc_only", "gc+ori", "ori_only"):
            cols = (gc_cols if name == "gc_only"
                    else ori_cols if name == "ori_only" else np.arange(len(names)))
            entry["coefficients"] = {names[c]: round(float(v), 4)
                                     for c, v in zip(cols, clf.coef_[0])}
        payload["models"][name] = entry
        np.save(results / f"proba_{name.replace('+', '_')}_test.npy", proba["test"])
        m, mt = entry["test"], entry["test_tuned"]
        log(f"{name:22s} feats={Xtr.shape[1]:5d} "
            f"test mcc={m['mcc']:.4f} f1={m['f1']:.4f} auprc={m['auprc']:.4f} | "
            f"tuned mcc={mt['mcc']:.4f} f1={mt['f1']:.4f} | "
            f"replicon mcc={entry['test_replicon']['mcc']:.4f} "
            f"({time.time() - t0:.0f}s)")

    out = results / "composition_baselines.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"\nwrote {out}")

    rows = []
    for name, e in payload["models"].items():
        rows.append({"model": name, "n_features": e["n_features"],
                     "mcc": round(e["test"]["mcc"], 4), "f1": round(e["test"]["f1"], 4),
                     "auprc": round(e["test"]["auprc"], 4),
                     "mcc_tuned": round(e["test_tuned"]["mcc"], 4),
                     "f1_tuned": round(e["test_tuned"]["f1"], 4),
                     "replicon_mcc": round(e["test_replicon"]["mcc"], 4)})
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "composition_baselines.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
