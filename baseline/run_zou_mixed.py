"""Zou et al. (PLoS ONE 2022), retrained on the MIXED-LENGTH dataset.

run_zou.py trains the same architecture on the original fixed-5-kb dataset and
writes into results/. This script is its mixed-length counterpart and writes
into results_mixed/, so that Zou can be compared with Platon, MOB-suite, PLASMe,
PlasmidHunter and the ESM-2 ladder on the one dataset they all see.

Three things differ from run_zou.py, and only three:

1. **Where the k-mers come from.** The fixed-length path stages a rectangular
   (n_fragments, 5000) base-code buffer and counts k-mers across it in blocks
   (prepare_features.py). A dataset whose fragments run from 1 kb to 100 kb has
   no such rectangle, so each fragment is counted on its own with
   features.count_kmers_one - verified to agree with the fixed-length counter
   element for element, including how both skip windows with a non-ACGT base.

2. **Counts or frequencies.** Zou's pipeline feeds KMC *counts* into a standard
   scaler. On fixed 5-kb fragments a count and a frequency differ only by a
   constant, so the choice is invisible; on this dataset a count carries the
   fragment's length in it, and a 100 kb fragment's counts are ~100x a 1 kb
   fragment's whatever their composition. That is a real confound and not one to
   resolve silently, so both are trained:

       zou             raw canonical 6-mer counts - the published pipeline,
                       applied to this input as-is. The headline row.
       zou_lengthnorm  the same counts divided by their own sum, which removes
                       the length term and leaves only composition.

   If the two agree, length was not doing the work. If `zou` is much the better,
   it is reading length - which is worth knowing, because none of the other four
   tools can.

3. **It all fits in memory.** 112,929 x 2080 float32 is 940 MB for the whole
   dataset, against 11 GB for the 5-kb dataset's feature matrix, so the block
   streaming of baselines/data.MemmapStream buys nothing here and the plainer
   in-memory loop is easier to check.

Everything else - the 256-256-128-128-32-10-1 architecture with its dropout and
L2, Adam at 1e-3, batch 256, BCE, early stopping on validation, the paper's
three-vote ensemble over three seeds, and both the balanced and full class
regimes - is the shared code in baselines/, unchanged.

    python baseline/run_zou_mixed.py                      # both regimes, both variants
    python baseline/run_zou_mixed.py --max-epochs 2       # smoke test
    python baseline/run_zou_mixed.py --variants zou       # counts only
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import data, features, metrics, models, train  # noqa: E402

K = 6
CANONICAL = True
SPLITS = ("train", "validation", "test")

# name -> how the cached counts are turned into model input. Both go through the
# same StandardScaler afterwards, exactly as Zou's model_train.py does.
VARIANTS = {
    "zou": lambda counts: counts,
    "zou_lengthnorm": lambda counts: counts / np.maximum(
        counts.sum(axis=1, keepdims=True), 1.0),
}


def peak_rss_mb() -> float:
    """Peak resident memory of this process, in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_counts(dataset_dir: Path, cache: Path, split: str, table: pd.DataFrame,
                 colmap: np.ndarray, n_cols: int, log) -> np.ndarray:
    """Canonical 6-mer counts per fragment, aligned to `table`'s row order."""
    path = cache / f"zou_k{K}{'c' if CANONICAL else 'f'}_{split}.npy"
    if path.exists():
        counts = np.load(path)
        if counts.shape == (len(table), n_cols):
            log(f"  {split}: cached {counts.shape}")
            return counts
        log(f"  {split}: cache has shape {counts.shape}, rebuilding")

    expected = table["fragment_id"].to_numpy()
    out = np.zeros((len(table), n_cols), dtype=np.float32)
    t0 = time.time()
    i = -1
    for i, (name, seq) in enumerate(data.iter_fasta(Path(dataset_dir) / f"{split}.fasta")):
        if i >= len(table):
            raise SystemExit(f"{split}: FASTA has more records than the CSV")
        # Same guard every other script in this tree uses: the FASTA and the CSV
        # must be in the same order, or every label below is attached to the
        # wrong sequence and nothing downstream would notice.
        if name != expected[i]:
            raise SystemExit(f"{split}: FASTA record {i} is {name}, CSV says {expected[i]}")
        codes = data.CODE_LUT[np.frombuffer(seq, dtype=np.uint8)]
        out[i] = features.count_kmers_one(codes, K, colmap, n_cols)
        if (i + 1) % 20000 == 0:
            log(f"  {split}: {i + 1:,}/{len(table):,} ({time.time() - t0:.0f}s)")
    if i + 1 != len(table):
        raise SystemExit(f"{split}: FASTA holds {i + 1} records, CSV holds {len(table)}")
    np.save(path, out)
    log(f"  {split}: counted in {time.time() - t0:.0f}s -> {path.name} "
        f"({out.nbytes / 1e6:.0f} MB)")
    return out


def predict(model, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[s:s + batch]))).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def monitor_score(kind: str, y: np.ndarray, proba: np.ndarray) -> float:
    if kind == "accuracy":
        return float(((proba >= 0.5).astype(int) == y).mean())
    return float(metrics.average_precision_score(y, proba))


def train_one(z, Xtr, ytr, Xval, yval, seed, limits, monitor, log):
    train.set_seed(seed)
    model = models.ZouMLP(Xtr.shape[1], tuple(z["dropout"]))
    opt = model.optimizer(z["learning_rate"], z["l2"])
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, history = -np.inf, None, 0, []

    for epoch in range(1, limits["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(order), z["batch_size"]):
            sel = order[s:s + z["batch_size"]]
            opt.zero_grad()
            loss = criterion(model(torch.from_numpy(Xtr[sel])),
                             torch.from_numpy(ytr[sel]).float())
            loss.backward()
            opt.step()
            losses.append(loss.item())
        score = monitor_score(monitor, yval, predict(model, Xval))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        f"val_{monitor}": score})
        log(f"    epoch {epoch}/{limits['max_epochs']} loss={np.mean(losses):.4f} "
            f"val_{monitor}={score:.4f} ({time.time() - t0:.0f}s)")
        if score > best + z["min_delta"]:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= limits["patience"]:
            log(f"    early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--regimes", nargs="*", default=None,
                    help="subset of config's regimes; the summary files are then "
                         "rewritten with only what this run produced")
    ap.add_argument("--variants", nargs="*", default=None, choices=list(VARIANTS),
                    help="subset of the feature variants; same caveat as --regimes")
    ap.add_argument("--max-epochs", type=int, help="override config (smoke tests)")
    args = ap.parse_args()

    cfg = data.load_config(args.config or HERE / "config.yaml")
    z = cfg["zou"]
    if args.max_epochs:
        for regime in ("balanced", "full"):
            z[regime]["max_epochs"] = args.max_epochs
            z[regime]["patience"] = min(z[regime]["patience"], args.max_epochs)
    train.set_threads(z.get("torch_threads"))
    dataset_dir, cache, results = cfg["dataset_dir"], cfg["cache_dir"], cfg["results_dir"]
    log = lambda m: print(m, flush=True)  # noqa: E731
    run_t0 = time.time()

    n_cols = features.n_features(K, CANONICAL)
    colmap = features.canonical_colmap(K) if CANONICAL else None
    # The column order has to be KMC's, because that is the order Zou's released
    # model and its kmer_order.csv use. prepare_features.py checks this for the
    # fixed-length path; the same check belongs here.
    reference = HERE / "reference" / "kmc_kmer_order.csv"
    if reference.exists():
        columns = np.array([np.flatnonzero(colmap == c)[0] for c in range(colmap.max() + 1)])
        if features.kmer_strings(K, columns) != reference.read_text().split():
            raise SystemExit("canonical 6-mer column order does not match KMC's "
                             f"{reference.name}; the feature columns would be permuted")
        log(f"canonical {K}-mer column order matches {reference.name} ({n_cols} columns)")

    tables, counts, labels = {}, {}, {}
    for split in SPLITS:
        tables[split] = data.load_table(dataset_dir, split)
        if "length_bucket" not in tables[split].columns:
            raise SystemExit("dataset has no length_bucket column; this script is for "
                             "the mixed-length dataset - use run_zou.py for the 5-kb one")
        labels[split] = tables[split]["label"].to_numpy()
        log(f"{split}: {len(tables[split]):,} fragments "
            f"({int(labels[split].sum()):,} plasmid, "
            f"{labels[split].mean():.2%}), "
            f"{tables[split]['length'].sum() / 1e6:,.0f} Mbp")
        counts[split] = build_counts(dataset_dir, cache, split, tables[split],
                                     colmap, n_cols, log)

    payload = {
        "method": "Zou et al. 2022 (BV-BRC zou-plasmid-prediction), retrained",
        "dataset_dir": str(dataset_dir),
        "kmer_k": K, "canonical": CANONICAL, "n_features": int(n_cols),
        "upstream": cfg["upstream"]["zou"],
        "note": ("Retrained on the mixed-length dataset. The `zou` variant feeds raw "
                 "canonical 6-mer counts to the published architecture, as the paper "
                 "does; on variable-length fragments those counts carry the fragment "
                 "length. `zou_lengthnorm` divides each fragment's counts by their own "
                 "sum, which removes the length term. The gap between them is how much "
                 "of the counts model is reading length rather than composition."),
        "models": {},
    }
    rows = []
    regimes = args.regimes or cfg["regimes"]
    variants = args.variants or list(VARIANTS)
    if set(variants) != set(VARIANTS) or set(regimes) != set(cfg["regimes"]):
        # zou_mixed.json/.csv are rewritten from this run alone, not merged into
        # whatever was there - merging would silently mix results from two code
        # versions. Say so, because a filtered run quietly dropping rows from the
        # report is exactly the kind of thing nobody notices.
        log(f"NOTE: running only variants={variants} regimes={regimes}. "
            "zou_mixed.json and zou_mixed.csv will be REWRITTEN with just these, "
            "dropping anything a previous run left there. Re-run without "
            "--variants/--regimes before building the report.")

    for variant in variants:
        transform_counts = VARIANTS[variant]
        X = {s: transform_counts(counts[s]).astype(np.float32) for s in SPLITS}
        for regime in regimes:
            seed_rng = np.random.default_rng(cfg["random_seed"])
            if regime == "balanced":
                subset = data.balanced_subset(labels["train"], seed_rng)
                val_subset = data.balanced_subset(labels["validation"], seed_rng)
                monitor = "accuracy"
            else:
                subset = np.arange(len(labels["train"]))
                val_subset = np.arange(len(labels["validation"]))
                # Accuracy is degenerate at 2.5% prevalence, so model selection
                # uses AUPRC - the same substitution run_zou.py makes.
                monitor = "auprc"
            limits = z[regime]

            # The scaler is fitted on the rows the model actually trains on, which
            # for the balanced regime is the subsample, not the whole split.
            Xtr = X["train"][subset]
            mean = Xtr.mean(axis=0)
            scale = Xtr.std(axis=0)
            scale[scale == 0.0] = 1.0
            std = lambda a: ((a - mean) / scale).astype(np.float32)  # noqa: E731
            Xtr_s = std(Xtr)
            Xval_fit_s = std(X["validation"][val_subset])
            yval_fit = labels["validation"][val_subset]
            Xval_s, Xtest_s = std(X["validation"]), std(X["test"])

            log(f"\n=== {variant} | regime={regime} | train fragments={len(subset):,} "
                f"| monitor=val_{monitor} | max_epochs={limits['max_epochs']} ===")
            t_regime = time.time()
            probs = {"validation": [], "test": []}
            per_model = {}
            for i in range(z["n_models"]):
                seed = cfg["random_seed"] + 100 * i
                log(f"  model {i + 1}/{z['n_models']} (seed {seed})")
                t0 = time.time()
                model, history, best_epoch = train_one(
                    z, Xtr_s, labels["train"][subset], Xval_fit_s, yval_fit,
                    seed, limits, monitor, log)
                probs["validation"].append(predict(model, Xval_s))
                probs["test"].append(predict(model, Xtest_s))
                per_model[f"model{i}"] = {
                    "seed": seed, "best_epoch": best_epoch,
                    "train_seconds": round(time.time() - t0, 1), "history": history,
                    "test": metrics.binary_metrics(labels["test"], probs["test"][-1]),
                }
                m = per_model[f"model{i}"]["test"]
                log(f"    test: auprc={m['auprc']:.4f} f1={m['f1']:.4f} "
                    f"mcc={m['mcc']:.4f} recall={m['recall']:.4f}")

            mean_prob = {s: np.mean(probs[s], axis=0) for s in probs}
            # The paper's rule is a majority vote over the three binarised
            # predictions, which is not the same as thresholding the mean - both
            # are reported rather than picking one.
            votes = (np.mean([(p >= 0.5).astype(float) for p in probs["test"]], axis=0)
                     >= 0.5).astype(float)
            tuned = metrics.best_f1_threshold(labels["validation"], mean_prob["validation"])
            # Chosen on VALIDATION, so unlike the four published tools this one
            # is an operating point that could have been picked in advance, not
            # a ceiling. By MCC rather than F1 because MCC is what the table
            # compares, and at 1:40 the two optima are not the same threshold.
            tuned_mcc = metrics.best_mcc_threshold(labels["validation"],
                                                   mean_prob["validation"])
            entry = {
                "variant": variant, "regime": regime,
                "n_train_fragments": int(len(subset)),
                "monitor": f"val_{monitor}",
                "tuned_threshold": float(tuned),
                "regime_seconds": round(time.time() - t_regime, 1),
                "per_model": per_model,
                "validation": metrics.binary_metrics(labels["validation"],
                                                     mean_prob["validation"]),
                "test": metrics.binary_metrics(labels["test"], mean_prob["test"]),
                "test_three_vote": metrics.binary_metrics(labels["test"], votes),
                "test_tuned": metrics.binary_metrics(labels["test"], mean_prob["test"], tuned),
                "test_tuned_mcc": metrics.binary_metrics(labels["test"],
                                                         mean_prob["test"], tuned_mcc),
                "tuned_threshold_f1": float(tuned),
                "tuned_threshold_mcc": float(tuned_mcc),
                "test_replicon": metrics.replicon_metrics(tables["test"], mean_prob["test"]),
                "test_by_length": metrics.length_bucket_metrics(
                    tables["test"]["length_bucket"], labels["test"], mean_prob["test"],
                    tuned).to_dict("records"),
            }
            payload["models"][f"{variant}|{regime}"] = entry
            np.save(results / f"proba_{variant}_{regime}_test.npy", mean_prob["test"])

            t, tt = entry["test"], entry["test_tuned"]
            rows.append({"model": variant, "regime": regime,
                         "n_features": int(n_cols),
                         "n_train": int(len(subset)),
                         "mcc": round(t["mcc"], 4), "f1": round(t["f1"], 4),
                         "precision": round(t["precision"], 4),
                         "recall": round(t["recall"], 4),
                         "specificity": round(t["specificity"], 4),
                         "auprc": round(t["auprc"], 4), "auroc": round(t["auroc"], 4),
                         "mcc_tuned": round(entry["test_tuned_mcc"]["mcc"], 4),
                         "tuned_threshold": round(float(tuned_mcc), 4),
                         "f1_tuned": round(tt["f1"], 4),
                         "mcc_three_vote": round(entry["test_three_vote"]["mcc"], 4),
                         "replicon_mcc": round(entry["test_replicon"]["mcc"], 4),
                         "train_seconds": entry["regime_seconds"]})
            log(f"  ENSEMBLE test: mcc={t['mcc']:.4f} f1={t['f1']:.4f} "
                f"auprc={t['auprc']:.4f} recall={t['recall']:.4f} "
                f"precision={t['precision']:.4f} | tuned mcc={tt['mcc']:.4f} "
                f"| replicon mcc={entry['test_replicon']['mcc']:.4f} "
                f"({entry['regime_seconds']:.0f}s)")

    # The fifth baseline is the only one that is trained here, so its cost has to
    # come from the script rather than from a per-shard runtime record. Same three
    # quantities server/_timed_run.py collects for the other four.
    payload["runtime"] = {
        "tool": "zou",
        "wall_seconds": round(time.time() - run_t0, 1),
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "torch_threads": torch.get_num_threads(),
        "host": platform.node(),
        "n_fragments_featurised": int(sum(len(tables[s]) for s in SPLITS)),
        "bp_featurised": int(sum(int(tables[s]["length"].sum()) for s in SPLITS)),
    }

    out = results / "zou_mixed.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "zou_mixed.csv", index=False)
    log(f"\nwrote {out}")
    log(f"wrote {results / 'zou_mixed.csv'}")
    print()
    print(summary.to_string(index=False))
    print(f"\ntotal {payload['runtime']['wall_seconds']:,.0f}s, peak "
          f"{peak_rss_mb():,.0f} MiB on {torch.get_num_threads()} torch threads")
    print("`zou` is the published pipeline as-is (raw counts, so fragment length is "
          "part of the input); `zou_lengthnorm` is the same model on per-fragment "
          "frequencies. Compare the two before quoting either.")


if __name__ == "__main__":
    main()
