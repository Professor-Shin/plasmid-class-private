"""Baseline 3: PlasClass (Pellow, Zorea & Shamir, PLoS Comput Biol 2020) retrained.

Canonical k-mer frequencies for k = 3..7, each k normalised to sum to one and then
concatenated (32 + 136 + 512 + 2080 + 8192 = 10952 features), standardised, and fed to
an L2 logistic regression - the architecture in Shamir-Lab/PlasClass `plasclass/train.py`.

PlasClass trains one model per sequence-length bin. Every fragment here is 5000 bp and
`plasclass._get_scale(5000)` routes that to the 1000 bp model, so exactly one model is
trained. Its training code also samples an equal number of fragments per class, so only
the balanced regime is run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import data, features, metrics, train  # noqa: E402
from prepare_features import feature_path  # noqa: E402


class PlasClassFeatures:
    """Assemble the concatenated per-k canonical frequency vector from count memmaps."""

    def __init__(self, cache: Path, split: str, n_rows: int, ks):
        self.ks = list(ks)
        self.widths = [features.n_features(k, True) for k in self.ks]
        self.n_cols = sum(self.widths)
        self.n_rows = n_rows
        self.mms = [np.memmap(feature_path(cache, split, k, True), dtype=np.uint16,
                              mode="r", shape=(n_rows, w))
                    for k, w in zip(self.ks, self.widths)]

    def _fill(self, rows, out):
        start = 0
        for mm, width in zip(self.mms, self.widths):
            counts = np.asarray(mm[rows]).astype(np.float32)
            totals = counts.sum(axis=1, keepdims=True)
            np.maximum(totals, 1.0, out=totals)
            out[:, start:start + width] = counts / totals
            start += width

    def take(self, rows, chunk_rows=2000) -> np.ndarray:
        rows = np.asarray(rows)
        out = np.empty((len(rows), self.n_cols), dtype=np.float32)
        for i in range(0, len(rows), chunk_rows):
            self._fill(rows[i:i + chunk_rows], out[i:i + chunk_rows])
        return out

    def chunks(self, chunk_rows=2000):
        buf = np.empty((chunk_rows, self.n_cols), dtype=np.float32)
        for start in range(0, self.n_rows, chunk_rows):
            stop = min(start + chunk_rows, self.n_rows)
            view = buf[:stop - start]
            self._fill(slice(start, stop), view)
            yield view


def predict_split(clf, scaler, feats: PlasClassFeatures, chunk_rows: int) -> np.ndarray:
    out = np.empty(feats.n_rows, dtype=np.float64)
    at = 0
    for block in feats.chunks(chunk_rows):
        scaled = scaler.transform(block)
        out[at:at + len(block)] = clf.predict_proba(scaled)[:, 1]
        at += len(block)
    return out


def compare_solvers(X, y, n, log) -> dict:
    """The released models use liblinear; float64 for 10952 features does not fit in
    RAM here, so lbfgs on float32 is used instead. Both minimise the same L2 logistic
    objective - this checks on a subset that they agree."""
    idx = np.concatenate([np.flatnonzero(y == 1)[:n // 2], np.flatnonzero(y == 0)[:n // 2]])
    sub, sub_y = X[idx], y[idx]
    fits = {}
    for solver in ("liblinear", "lbfgs"):
        t0 = time.time()
        clf = LogisticRegression(solver=solver, C=1.0, max_iter=1000).fit(sub, sub_y)
        fits[solver] = (clf.predict_proba(sub)[:, 1], clf.coef_.ravel(), time.time() - t0)
    p_lib, c_lib, _ = fits["liblinear"]
    p_lbf, c_lbf, _ = fits["lbfgs"]
    out = {"n_rows": int(len(idx)),
           "max_abs_prob_diff": float(np.abs(p_lib - p_lbf).max()),
           "mean_abs_prob_diff": float(np.abs(p_lib - p_lbf).mean()),
           "coef_pearson_r": float(np.corrcoef(c_lib, c_lbf)[0, 1]),
           "agreement_at_0.5": float(((p_lib >= 0.5) == (p_lbf >= 0.5)).mean())}
    log(f"  solver check on {out['n_rows']} rows: max|dp|={out['max_abs_prob_diff']:.4f} "
        f"mean|dp|={out['mean_abs_prob_diff']:.5f} coef r={out['coef_pearson_r']:.5f} "
        f"label agreement={out['agreement_at_0.5']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-solver-check", action="store_true")
    args = ap.parse_args()

    cfg = data.load_config(HERE / "config.yaml")
    pc = cfg["plasclass"]
    cache, results = cfg["cache_dir"], cfg["results_dir"]
    (results / "probs").mkdir(exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    tables = {s: data.load_table(cfg["dataset_dir"], s) for s in data.SPLITS}
    labels = {s: np.load(cache / f"labels_{s}.npy") for s in data.SPLITS}
    feats = {s: PlasClassFeatures(cache, s, len(labels[s]), pc["ks"]) for s in data.SPLITS}
    log(f"PlasClass features: k={pc['ks']} -> {feats['train'].n_cols} columns "
        f"(length bin {pc['length_bin']} bp)")

    regime = "balanced"
    subset = data.balanced_subset(labels["train"], np.random.default_rng(cfg["random_seed"]))
    y_train = labels["train"][subset]
    log(f"=== PlasClass | regime={regime} | train fragments={len(subset)} "
        f"({int(y_train.sum())} plasmid) ===")

    t0 = time.time()
    X = feats["train"].take(subset, pc["chunk_rows"])
    log(f"  assembled train matrix {X.shape} ({X.nbytes / 2**30:.2f} GiB) "
        f"in {time.time() - t0:.1f}s")

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X, copy=False)

    solver_check = None if args.skip_solver_check else compare_solvers(X, y_train, 3000, log)

    t0 = time.time()
    clf = LogisticRegression(solver=pc["solver"], C=pc["C"], max_iter=pc["max_iter"])
    clf.fit(X, y_train)
    train_seconds = time.time() - t0
    log(f"  fitted {pc['solver']} in {train_seconds:.1f}s, n_iter={int(clf.n_iter_[0])}")
    del X

    probs = {}
    for split in ("validation", "test"):
        t0 = time.time()
        probs[split] = predict_split(clf, scaler, feats[split], pc["chunk_rows"])
        np.save(results / "probs" / f"plasclass_{regime}_lr_{split}.npy", probs[split])
        log(f"  predicted {split} ({len(probs[split])} fragments) in {time.time() - t0:.1f}s")

    tuned = metrics.best_f1_threshold(labels["validation"], probs["validation"])
    payload = {
        "method": "PlasClass (Pellow, Zorea & Shamir 2020), retrained",
        "regime": regime,
        "n_train_fragments": int(len(subset)),
        "n_features": int(feats["train"].n_cols),
        "per_model": {"logreg": {"solver": pc["solver"], "C": pc["C"],
                                 "n_iter": int(clf.n_iter_[0]),
                                 "train_seconds": train_seconds,
                                 "test": metrics.binary_metrics(labels["test"], probs["test"])}},
        "solver_equivalence_check": solver_check,
        "ensemble": {
            "validation": metrics.binary_metrics(labels["validation"], probs["validation"]),
            "test": metrics.binary_metrics(labels["test"], probs["test"]),
            "test_tuned_threshold": metrics.binary_metrics(labels["test"], probs["test"], tuned),
            "test_replicon_level": metrics.replicon_metrics(tables["test"], probs["test"]),
        },
    }
    train.save_json(results / f"plasclass_{regime}.json", payload)
    t = payload["ensemble"]["test"]
    log(f"  TEST: acc={t['accuracy']:.4f} f1={t['f1']:.4f} recall={t['recall']:.4f} "
        f"precision={t['precision']:.4f} auprc={t['auprc']:.4f} mcc={t['mcc']:.4f}")


if __name__ == "__main__":
    main()
