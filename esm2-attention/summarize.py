"""Step 3: score the k-mer baselines on ESM's exact evaluation fragments and tabulate.

The baselines' per-fragment probabilities were saved by ../baseline in the order of
the full test split, so they can be restricted to exactly the fragments ESM-2 was
evaluated on and re-scored with the same weights and the same threshold protocol.

This previously read `results/esm2_attention.json` with the keys
`ensemble_best_epoch` / `test_final`, which is the schema of the original 5-kb run.
train_eval.py now writes `esm2_variants.json`, where each variant carries
`ensemble_final` / `ensemble_best` and (on the server config) a `regimes` block, so
the reader below follows that shape and reports the whole ladder rather than one
model. Run it with the same --config as the training step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import data, metrics  # noqa: E402

BASELINES = [
    ("PlasFlow (vote of 3) - balanced", "plasflow_balanced_vote"),
    ("PlasFlow (vote of 3) - full", "plasflow_full_vote"),
    ("Zou et al. (mean of 3) - balanced", "zou_balanced_mean"),
    ("Zou et al. (mean of 3) - full", "zou_full_mean"),
    ("PlasClass (logistic regression) - balanced", "plasclass_balanced_lr"),
]
COMPOSITION = [
    ("GC only (logistic regression)", "proba_gc_only_test.npy"),
    ("ori only (logistic regression)", "proba_ori_only_test.npy"),
    ("5-mer (logistic regression)", "proba_kmer5_test.npy"),
    ("5-mer + GC + ori", "proba_kmer5_gc_ori_test.npy"),
]
MAIN = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
        "auroc", "auprc"]


def row(label, m, keys=MAIN):
    return "| " + label + " | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|", *rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()

    cfg = data.load_config(args.config)
    results, baseline_dir = cfg["results_dir"], cfg["baseline_dir"]
    payload_path = results / "esm2_variants.json"
    if not payload_path.exists():
        raise SystemExit(f"no {payload_path} - run train_eval.py with the same --config first")
    esm = json.loads(payload_path.read_text(encoding="utf-8"))

    sets = {s: data.open_split(cfg["cache_dir"], s) for s in ("validation", "test")}
    test, val = sets["test"], sets["validation"]
    headline = esm.get("headline_regime")

    sampled = ("the whole test split" if np.allclose(test.weight, 1.0)
               else f"an evaluation subsample of the test split")
    out = ["# ESM-2 attention vs. the k-mer baselines", "",
           f"Scored on {sampled}: **{len(test):,} fragments** "
           f"({int(test.label.sum())} plasmid + {int((test.label == 0).sum())} chromosome). "
           f"Every metric is weighted back to the full split "
           f"({test.weight.sum():,.0f} fragments, "
           f"{test.weight[test.label == 1].sum():,.0f} plasmid = "
           f"{100 * test.weight[test.label == 1].sum() / test.weight.sum():.3f}%).", "",
           "All models below are scored on exactly these fragments with the same "
           "threshold protocol: the decision threshold is chosen on the weighted "
           "validation set and then applied to the test set.", ""]
    if headline:
        out += [f"ESM-2 rows are the **{headline}** regime unless stated otherwise; "
                f"regimes run: {', '.join(esm.get('regimes', []))}.", ""]

    # ---------------------------------------------------------------- baselines --
    scored = {}
    for label, stem in BASELINES:
        tp = baseline_dir / "probs" / f"{stem}_test.npy"
        vp = baseline_dir / "probs" / f"{stem}_validation.npy"
        if not tp.exists() or not vp.exists():
            continue
        p_test = np.load(tp)[test.row_in_split]
        p_val = np.load(vp)[val.row_in_split]
        tuned = metrics.best_f1_threshold(val.label, p_val, val.weight)
        scored[label] = {
            "half": metrics.binary_metrics(test.label, p_test, 0.5, test.weight),
            "tuned": metrics.binary_metrics(test.label, p_test, tuned, test.weight),
            "replicon": metrics.replicon_metrics(test.replicon_id, test.label, p_test),
            "threshold": tuned,
        }
    # The composition baselines were fitted on the full mixed-length dataset and
    # saved test-split probabilities only, so they get the 0.5 row and no tuned row.
    for label, fname in COMPOSITION:
        p = baseline_dir / fname
        if not p.exists():
            continue
        p_test = np.load(p)[test.row_in_split]
        scored[label] = {
            "half": metrics.binary_metrics(test.label, p_test, 0.5, test.weight),
            "replicon": metrics.replicon_metrics(test.replicon_id, test.label, p_test),
        }

    # --------------------------------------------------------------- ESM ladder --
    def variant_entry(name, regime=None):
        entry = esm["variants"][name]
        if regime is None or regime == headline:
            return entry
        return entry["regimes"][regime]

    names = list(esm["variants"])
    out += ["## 1. Head-to-head at the default threshold 0.5", ""]
    rows = [row("always-chromosome", metrics.binary_metrics(
        test.label, np.zeros(len(test)) + 1e-9, 0.5, test.weight))]
    for label, s in scored.items():
        rows.append(row(label, s["half"]))
    for name in names:
        e = variant_entry(name)
        rows.append(row(f"**{name} (ensemble, best epoch)**", e["ensemble_best"]["weighted"]))
        rows.append(row(f"{name} (ensemble, final epoch)", e["ensemble_final"]["weighted"]))
    out += [md_table(["model", *MAIN], rows), ""]

    out += ["## 2. Head-to-head at the validation-tuned threshold", ""]
    rows = []
    for label, s in scored.items():
        if "tuned" in s:
            rows.append(row(f"{label} @ {s['threshold']:.3f}", s["tuned"]))
    for name in names:
        e = variant_entry(name)["ensemble_best"]
        rows.append(row(f"**{name} (best epoch) @ {e['tuned_threshold']:.3f}**",
                        e["weighted_tuned"]))
    out += [md_table(["model", *MAIN], rows), ""]

    out += ["## 3. ESM-2 run-to-run spread (individual seeds, weighted, threshold 0.5)", ""]
    rows = []
    for name in names:
        for v in variant_entry(name)["per_seed"].values():
            rows.append(row(f"{name} seed {v['seed']} "
                            f"(best epoch {v['best_epoch']} of {v.get('epochs_run', '?')})",
                            v["test"]["weighted"]))
        rows.append(row(f"**{name} ensemble (mean probability)**",
                        variant_entry(name)["ensemble_best"]["weighted"]))
    out += [md_table(["model", *MAIN], rows), ""]

    out += ["## 4. Replicon-level (mean probability over a replicon's fragments)", ""]
    rows = [row(label, s["replicon"]) for label, s in scored.items()]
    for name in names:
        rows.append(row(f"**{name} (best epoch)**",
                        variant_entry(name)["ensemble_best"]["replicon_level"]))
    out += [md_table(["model", *MAIN], rows), ""]

    if len(esm.get("regimes", [])) > 1:
        out += ["## 5. Class-balance regimes", "",
                "`full` trains on the split's natural plasmid rate with "
                "`pos_weight = n_neg/n_pos`, so the training distribution matches the "
                "evaluation distribution. `balanced` resamples to 1:1, which is the "
                "condition the laptop's stratified subsample imposed implicitly.", ""]
        rows = []
        for name in names:
            for regime in esm["regimes"]:
                e = variant_entry(name, regime)["ensemble_best"]
                rows.append(row(f"{name} [{regime}]", e["weighted"]))
        out += [md_table(["model", *MAIN], rows), ""]

    out += ["## 6. Fragments with no callable ORF", "",
            "The ORF pathway is blind to these by construction; only the auxiliary "
            "whole-fragment features can separate them.", ""]
    rows = []
    for name in names:
        free = variant_entry(name)["ensemble_best"].get("orf_free", {})
        if "mcc" in free:
            rows.append(row(f"{name} (n={free['n']})", free))
    out += ([md_table(["model", *MAIN], rows), ""] if rows
            else ["_no ORF-free fragments with both classes present_", ""])

    out += ["## 7. Set sizes and ORF statistics", "",
            md_table(["split", "fragments", "plasmid", "effective size", "ORFs per fragment"],
                     ["| " + s + " | "
                      f"{v.get('n_used', v.get('n_sampled')):,} | {v['n_plasmid']:,} | "
                      f"{v['effective_n']:,.0f} | {v['orfs_per_fragment']:.2f} |"
                      for s, v in esm["splits"].items()]), ""]

    epoch_seconds = [e["seconds"] for name in names
                     for v in variant_entry(name)["per_seed"].values()
                     for e in v["history"]]
    seed_seconds = [v["train_seconds"] for name in names
                    for v in variant_entry(name)["per_seed"].values()]
    out += ["## 8. Cost", "",
            md_table(["item", "value"],
                     ["| ESM-2 embedding | see the embedding job log |",
                      f"| device | {esm.get('device', 'cpu')} |",
                      f"| hidden size | {esm.get('hidden_size', '?')} |",
                      "| attention-head training, median seconds per seed | "
                      f"{np.median(seed_seconds):.0f} s |",
                      f"| median seconds per training epoch | {np.median(epoch_seconds):.0f} s |",
                      f"| seeds per variant | {len(variant_entry(names[0])['per_seed'])} |",
                      f"| variants x regimes | {len(names)} x {len(esm.get('regimes', [1]))} |"]),
            ""]

    (results / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
