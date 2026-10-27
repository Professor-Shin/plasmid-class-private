"""How much of the signal is just GC? Accuracy vs plasmid-host GC divergence.

The expert asked us to consider GC content; the sharper question a reviewer will
ask is whether the models are really detecting "plasmid" or merely "DNA whose
composition differs from the chromosome it sits next to". If accuracy on plasmid
fragments rises with |plasmid GC - host chromosome GC|, the models are leaning on
compositional divergence, which also predicts the known failure mode: plasmids
that have been resident long enough to ameliorate to their host's composition
should be missed.

For every plasmid in the test split we take its replicon-level GC and the GC of a
chromosome from the SAME assembly (its host), bin plasmid fragments by that gap,
and report per-bin recall plus the rank correlation between gap and correctness.

Inputs are whatever prediction files exist, so this runs against the composition
baselines alone or alongside the ESM variants.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset pipeline1" / "data" / "dataset_mixed"
REPLICONS = ROOT / "dataset pipeline1" / "data" / "metadata" / "replicons.csv"
# Which run to analyse. The server writes results_server/cache_server while the
# laptop wrote results_mixed/cache_mixed, and reading one while the rest of the
# report shows the other is how a figure ends up describing the wrong run.
BASELINE_RESULTS = ROOT / "baseline" / os.environ.get("BASE_RESULTS", "results_mixed")
ESM_RESULTS = ROOT / "esm2-attention" / os.environ.get("ESM_RESULTS", "results_mixed")
ESM_CACHE = ROOT / "esm2-attention" / os.environ.get("ESM_CACHE", "cache_mixed")
OUT = ROOT / "analysis" / "results"

GAP_BINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 1.0]


def host_gc_table() -> pd.DataFrame:
    """Per-replicon GC alongside the GC of a chromosome from the same assembly."""
    rep = pd.read_csv(REPLICONS, keep_default_na=False,
                      usecols=["replicon_id", "assembly_accession", "replicon_type",
                               "label", "length", "gc_content"])
    chrom = rep[rep["replicon_type"] == "chromosome"]
    # Longest chromosome of the assembly is the host reference; secondary
    # chromosomes/chromids would otherwise make the "host GC" ambiguous.
    host = (chrom.sort_values("length", ascending=False)
                 .groupby("assembly_accession")
                 .agg(host_gc=("gc_content", "first"),
                      host_chrom_len=("length", "first")).reset_index())
    out = rep.merge(host, on="assembly_accession", how="left")
    out["gc_gap"] = (out["gc_content"] - out["host_gc"]).abs()
    out["gc_signed"] = out["gc_content"] - out["host_gc"]
    return out[["replicon_id", "assembly_accession", "replicon_type", "gc_content",
                "host_gc", "gc_gap", "gc_signed"]]


def load_predictions() -> dict[str, pd.DataFrame]:
    """name -> DataFrame[fragment_id, proba]. Aligned by construction or by id."""
    preds: dict[str, pd.DataFrame] = {}
    test = pd.read_csv(DATASET / "test.csv", usecols=["fragment_id"])
    for path in sorted(BASELINE_RESULTS.glob("proba_*_test.npy")):
        proba = np.load(path)
        if len(proba) != len(test):
            print(f"  skip {path.name}: {len(proba)} rows vs {len(test)} test fragments")
            continue
        name = path.stem.replace("proba_", "").replace("_test", "")
        preds[name] = pd.DataFrame({"fragment_id": test["fragment_id"], "proba": proba})

    sub = ESM_CACHE / "subsample_test.csv"
    if sub.exists():
        ids = pd.read_csv(sub, usecols=["fragment_id"])["fragment_id"]
        for path in sorted((ESM_RESULTS / "probs").glob("*_ensemble_test.npy")):
            proba = np.load(path)
            if len(proba) != len(ids):
                continue
            name = path.stem.replace("_ensemble_test", "")
            preds[name] = pd.DataFrame({"fragment_id": ids, "proba": proba})
    return preds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frag = pd.read_csv(DATASET / "test.csv",
                       usecols=["fragment_id", "replicon_id", "assembly_accession",
                                "label", "length", "length_bucket", "gc_content"])
    gc = host_gc_table()
    frag = frag.merge(gc.drop(columns=["gc_content", "assembly_accession"]),
                      on="replicon_id", how="left")

    plasmid = frag[frag["label"] == 1]
    have_host = plasmid["host_gc"].notna()
    print(f"test plasmid fragments: {len(plasmid)}; "
          f"with a host chromosome in the dataset: {int(have_host.sum())} "
          f"({have_host.mean():.1%})")
    print(f"plasmid GC gap |plasmid - host|: median "
          f"{plasmid.loc[have_host, 'gc_gap'].median():.4f}, "
          f"p90 {plasmid.loc[have_host, 'gc_gap'].quantile(0.9):.4f}")
    print(f"signed gap (plasmid - host): mean "
          f"{plasmid.loc[have_host, 'gc_signed'].mean():+.4f} "
          f"(negative = plasmid is AT-richer than its host)")

    preds = load_predictions()
    if not preds:
        sys.exit("no prediction files found; run run_composition.py / train_eval.py first")
    print(f"prediction sets: {', '.join(preds)}")

    payload = {
        "n_test_plasmid_fragments": int(len(plasmid)),
        "plasmid_fragments_with_host": int(have_host.sum()),
        "gc_gap_median": float(plasmid.loc[have_host, "gc_gap"].median()),
        "gc_signed_mean": float(plasmid.loc[have_host, "gc_signed"].mean()),
        "models": {},
    }
    rows = []
    for name, pred in preds.items():
        df = plasmid.merge(pred, on="fragment_id", how="inner").dropna(subset=["host_gc"])
        if len(df) < 30:
            continue
        # Recall at 0.5: was this plasmid fragment actually recovered?
        df = df.assign(correct=(df["proba"] >= 0.5).astype(int))
        rho, p = stats.spearmanr(df["gc_gap"], df["proba"])
        rho_c, p_c = stats.spearmanr(df["gc_gap"], df["correct"])
        binned = pd.cut(df["gc_gap"], GAP_BINS, include_lowest=True)
        per_bin = (df.groupby(binned, observed=True)
                     .agg(n=("correct", "size"), recall=("correct", "mean"),
                          mean_proba=("proba", "mean"),
                          mean_gap=("gc_gap", "mean")).reset_index())
        per_bin["gc_gap_bin"] = per_bin["gc_gap"].astype(str)
        payload["models"][name] = {
            "n_plasmid_fragments_scored": int(len(df)),
            "spearman_gap_vs_proba": {"rho": float(rho), "p": float(p)},
            "spearman_gap_vs_correct": {"rho": float(rho_c), "p": float(p_c)},
            "recall_by_gc_gap": per_bin.drop(columns=["gc_gap"]).to_dict("records"),
        }
        for r in per_bin.itertuples():
            rows.append({"model": name, "gc_gap_bin": str(r.gc_gap), "n": int(r.n),
                         "recall": round(float(r.recall), 4),
                         "mean_proba": round(float(r.mean_proba), 4)})
        print(f"\n{name}: n={len(df)}  spearman(gap, p_plasmid)={rho:+.3f} (p={p:.2g})  "
              f"spearman(gap, correct)={rho_c:+.3f} (p={p_c:.2g})")
        print(per_bin[["gc_gap_bin", "n", "recall", "mean_proba"]].to_string(index=False))

    (OUT / "gc_signal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "gc_signal_recall_by_gap.csv", index=False)
    print(f"\nwrote {OUT / 'gc_signal.json'}")


if __name__ == "__main__":
    main()
