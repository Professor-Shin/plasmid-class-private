"""Score the Platon and MOB-suite predictions with the retrained baselines' metrics.

Every metric function here is imported verbatim from `baseline/baselines/metrics.py`,
the module that produced the PlasFlow / Zou / PlasClass / composition numbers, so
the two new baselines land in the same table with no risk of a different MCC
convention or replicon aggregation creeping in.

Three things this reports that a single accuracy number would hide:

1. **Unclassified policies.** Both tools can decline to classify a fragment -
   Platon when it finds no marker proteins, mob_recon when a contig is below its
   length cutoff. Counting those as chromosome, as plasmid, or excluding them
   gives three very different headline numbers, so all three are reported
   together with the coverage, exactly as plasflow_threshold_rule already does
   for PlasFlow's unclassified band.

2. **Hard labels are not probabilities.** mob_recon returns an assignment and
   Platon's filter modes return a decision. AUROC/AUPRC computed from those are
   label-dominated and are flagged `score_is_hard_label: true`. Platon's RDS is
   a genuine continuous score, so it gets its own threshold-free entry.

3. **An oracle threshold, labelled as such.** Only the test split is run (both
   tools are untrained, so a validation run would buy nothing), which means there
   is no held-out split on which to tune a threshold. The best-F1 threshold on
   the test split is reported under `test_oracle_tuned` as an upper bound, never
   as a comparable operating point.

Writes results_dir/tool_baselines.{json,csv}, in the same shape as
composition_baselines.{json,csv} so analysis/dump_tables.py can read both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import cfg, rules  # noqa: E402

# Stand-ins for "no score at all" on Platon's RDS scale, which is unbounded and
# whose chromosome cutoff is -7.9 - filling an unscored fragment with 0.0, as one
# would for a probability, would place it ABOVE that cutoff and so call it
# plasmid, which is the opposite of the "chromosome" policy.
RDS_FLOOR = -1e6
RDS_CEILING = 1e6


def _policy_view(y, proba, reported, table, policy, fill_low, fill_high):
    """Apply one unclassified policy, returning the rows and scores to score on."""
    proba = np.asarray(proba, dtype=float)
    if policy == "exclude":
        keep = np.asarray(reported, dtype=bool)
        return y[keep], proba[keep], table.loc[keep], keep
    filled = proba.copy()
    # The fill has to be on the SCORE's own scale, not on [0, 1]: what makes a
    # fragment "called chromosome" is being below the threshold being applied.
    filled[~np.asarray(reported, dtype=bool)] = (
        fill_low if policy == "chromosome" else fill_high)
    keep = np.ones(len(proba), dtype=bool)
    return y, filled, table, keep


def score_one(metrics, name, table, proba, reported, threshold, *,
              hard_label, policy, oracle_threshold=True,
              fill_low=0.0, fill_high=1.0) -> dict:
    y = table["label"].to_numpy()
    y_p, p_p, t_p, keep = _policy_view(y, proba, reported, table, policy,
                                       fill_low, fill_high)
    coverage = float(np.mean(reported))
    bounded = (fill_low, fill_high) == (0.0, 1.0)

    entry = {
        "model": name,
        "policy": policy,
        "score_is_hard_label": bool(hard_label),
        "threshold": float(threshold),
        "coverage": coverage,
        "n_scored": int(len(y_p)),
        "n_unclassified": int(len(y) - int(np.sum(reported))),
        "replicon_aggregation": "mean score" if bounded else "majority vote",
    }
    if len(y_p) == 0 or len(np.unique(y_p)) < 2:
        entry["note"] = "too few fragments or a single class after the policy"
        return entry

    entry["test"] = metrics.binary_metrics(y_p, p_p, threshold)
    # replicon_metrics averages the fragment scores of a replicon and then
    # thresholds. That is right for a probability, but Platon's RDS is unbounded
    # and unclassified fragments carry a +/-1e6 sentinel, so one sentinel would
    # dominate a replicon's mean and decide it on its own. For such a score the
    # fragment decisions are aggregated instead, which makes it a majority vote.
    replicon_scores = p_p if bounded else (p_p >= threshold).astype(float)
    replicon_threshold = threshold if bounded else 0.5
    entry["test_replicon"] = metrics.replicon_metrics(t_p, replicon_scores,
                                                      replicon_threshold)
    if "length_bucket" in t_p.columns:
        entry["test_by_length"] = metrics.length_bucket_metrics(
            t_p["length_bucket"], y_p, p_p, threshold).to_dict("records")
    if oracle_threshold:
        # Tuned ON THE TEST SPLIT, because no other split was run. An upper bound,
        # not an operating point that could be chosen in advance.
        tuned = metrics.best_f1_threshold(y_p, p_p)
        entry["oracle_threshold"] = float(tuned)
        entry["test_oracle_tuned"] = metrics.binary_metrics(y_p, p_p, tuned)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    conf = cfg.load_config(args.config)
    _, metrics = cfg.use_baseline_metrics()
    split = args.split or conf["split"]
    results = Path(conf["results_dir"])
    policies = conf["unclassified_policies"]
    log = lambda m: print(m, flush=True)  # noqa: E731

    payload = {
        "split": split,
        "dataset_dir": str(conf["dataset_dir"]),
        "upstream": conf["upstream"],
        "note": ("Platon and MOB-suite are database- and rule-based and are run as "
                 "published, not retrained. Fragments are presented to them grouped "
                 "by assembly, i.e. as the contigs of one fragmented draft assembly, "
                 "which is the input both tools were designed for. Fragments are "
                 "linear sub-sequences of a replicon, so they carry no circularity "
                 "and usually no oriT or rep hallmark; both tools are therefore "
                 "expected to lean chromosomal, and the replicon-level numbers are "
                 "the fairer comparison."),
        "models": {},
    }
    rows = []

    def record(entry):
        payload["models"][f"{entry['model']}|{entry['policy']}"] = entry
        if "test" not in entry:
            return
        m, r = entry["test"], entry.get("test_replicon", {})
        rows.append({
            "model": entry["model"],
            "policy": entry["policy"],
            "hard_label_score": entry["score_is_hard_label"],
            "coverage": round(entry["coverage"], 4),
            "n_scored": entry["n_scored"],
            "mcc": round(m["mcc"], 4),
            "f1": round(m["f1"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "specificity": round(m["specificity"], 4),
            "auprc": round(m["auprc"], 4),
            "auroc": round(m["auroc"], 4),
            "mcc_oracle": round(entry.get("test_oracle_tuned", {}).get("mcc", float("nan")), 4),
            "replicon_mcc": round(r.get("mcc", float("nan")), 4),
        })
        log(f"{entry['model']:24s} {entry['policy']:10s} "
            f"cov={entry['coverage']:.3f} mcc={m['mcc']:.4f} f1={m['f1']:.4f} "
            f"prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"auprc={m['auprc']:.4f} | replicon mcc={r.get('mcc', float('nan')):.4f}")

    # ------------------------------------------------------------------ Platon --
    platon_path = results / f"predictions_platon_{split}.csv"
    if platon_path.exists():
        pl = pd.read_csv(platon_path)
        reported = pl["reported"].to_numpy(dtype=bool)
        log(f"\n=== Platon: {len(pl):,} fragments, {reported.sum():,} characterised "
            f"({reported.mean():.2%} coverage)")
        for mode in conf["platon"]["modes"]:
            col = f"pred_{mode}"
            if col not in pl.columns or (pl[col] < 0).all():
                log(f"  mode {mode}: no predictions in the table, skipped")
                continue
            pred = pl[col].to_numpy()
            for policy in policies:
                record(score_one(metrics, f"platon_{mode}", pl,
                                 np.clip(pred, 0, 1).astype(float), pred >= 0,
                                 0.5, hard_label=True, policy=policy))

        # The RDS as a continuous score: the one threshold-free comparison available
        # for either tool. Scored at Platon's own two published cutoffs plus AUROC
        # and AUPRC, which do not depend on a threshold at all.
        rds = pl["rds"].to_numpy(dtype=float)
        for label, thr in (("specificity_cutoff", rules.SPECIFICITY_CUTOFF),
                           ("sensitivity_cutoff", rules.SENSITIVITY_CUTOFF)):
            for policy in policies:
                record(score_one(metrics, f"platon_rds@{label}", pl, rds, reported,
                                 thr, hard_label=False, policy=policy,
                                 fill_low=RDS_FLOOR, fill_high=RDS_CEILING))
    else:
        log(f"no {platon_path.name}; run collect_platon.py first")

    # ---------------------------------------------------------------- MOB-suite --
    mob_path = results / f"predictions_mobsuite_{split}.csv"
    if mob_path.exists():
        mb = pd.read_csv(mob_path)
        reported = mb["reported"].to_numpy(dtype=bool)
        log(f"\n=== MOB-recon: {len(mb):,} fragments, {reported.sum():,} reported "
            f"({reported.mean():.2%} coverage)")
        pred = mb["pred"].to_numpy()
        for policy in policies:
            record(score_one(metrics, "mobsuite_molecule_type", mb,
                             np.clip(pred, 0, 1).astype(float), pred >= 0,
                             0.5, hard_label=True, policy=policy))
        if "score" in mb.columns and mb["score"].notna().any():
            scores = np.nan_to_num(mb["score"].to_numpy(dtype=float), nan=0.0)
            for policy in policies:
                record(score_one(metrics, "mobsuite_mash_ranked", mb, scores, reported,
                                 0.5, hard_label=True, policy=policy))
    else:
        log(f"no {mob_path.name}; run collect_mobsuite.py first")

    if not rows:
        raise SystemExit("nothing scored - no prediction tables found")

    out = results / "tool_baselines.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "tool_baselines.csv", index=False)
    log(f"\nwrote {out}")
    log(f"wrote {results / 'tool_baselines.csv'}")
    print()
    print(summary.to_string(index=False))
    print("\nhard_label_score=True means AUROC/AUPRC for that row are derived from a "
          "decision, not a probability, and are label-dominated.")
    print("mcc_oracle uses a threshold tuned on the test split itself: an upper "
          "bound, not a comparable operating point.")


if __name__ == "__main__":
    main()
