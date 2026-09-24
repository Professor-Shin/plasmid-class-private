"""Step 3: score the two baselines on the identical evaluation subsample and tabulate.

The baselines' per-fragment probabilities were saved by ../baseline in the order of
the full test split, so they can be restricted to exactly the fragments ESM-2 was
evaluated on and re-scored with the same weights and the same threshold protocol.
"""

from __future__ import annotations

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
MAIN = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
        "auroc", "auprc"]


def row(label, m, keys=MAIN):
    return "| " + label + " | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|", *rows])


def main():
    cfg = data.load_config(HERE / "config.yaml")
    results, baseline_dir = cfg["results_dir"], cfg["baseline_dir"]
    sets = {s: data.EmbeddingSet(cfg["cache_dir"] / f"emb_{s}.npz")
            for s in ("validation", "test")}
    esm = json.loads((results / "esm2_attention.json").read_text(encoding="utf-8"))

    test, val = sets["test"], sets["validation"]
    out = ["# ESM-2 attention vs. the k-mer baselines", "",
           f"Evaluation subsample of the test split: **{len(test)} fragments** "
           f"({int(test.label.sum())} plasmid + {int((test.label == 0).sum())} chromosome). "
           f"Every metric is weighted back to the full split "
           f"({test.weight.sum():,.0f} fragments, "
           f"{test.weight[test.label == 1].sum():,.0f} plasmid = "
           f"{100 * test.weight[test.label == 1].sum() / test.weight.sum():.3f}%), "
           "so the numbers estimate performance on the whole test split.", "",
           "All models below are scored on exactly these fragments with the same "
           "threshold protocol: the decision threshold is chosen on the weighted "
           "validation subsample and then applied to the test subsample.", ""]

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

    esm_ens = esm["ensemble_final"]
    esm_best = esm["ensemble_best_epoch"]

    out += ["## 1. Head-to-head at the default threshold 0.5", ""]
    rows = [row("always-chromosome", metrics.binary_metrics(
        test.label, np.zeros(len(test)) + 1e-9, 0.5, test.weight))]
    for label, s in scored.items():
        rows.append(row(label, s["half"]))
    rows.append(row("**ESM-2 + attention (mean of 3, final epoch)**", esm_ens["weighted"]))
    rows.append(row("ESM-2 + attention (mean of 3, best epoch)", esm_best["weighted"]))
    out += [table(["model", *MAIN], rows), ""]

    out += ["## 2. Head-to-head at the validation-tuned threshold", ""]
    rows = []
    for label, s in scored.items():
        rows.append(row(f"{label} @ {s['threshold']:.3f}", s["tuned"]))
    rows.append(row(f"**ESM-2 + attention (final) @ {esm_ens['tuned_threshold']:.3f}**",
                    esm_ens["weighted_tuned"]))
    rows.append(row(f"ESM-2 + attention (best epoch) @ {esm_best['tuned_threshold']:.3f}",
                    esm_best["weighted_tuned"]))
    out += [table(["model", *MAIN], rows), ""]

    out += ["## 3. ESM-2 run-to-run spread (individual seeds, weighted, threshold 0.5)", ""]
    rows = [row(f"seed {v['seed']} (best epoch {v['best_epoch']})",
                v["test_final"]["weighted"]) for v in esm["per_seed"].values()]
    rows.append(row("**ensemble (mean probability)**", esm_ens["weighted"]))
    out += [table(["model", *MAIN], rows), ""]

    out += ["## 4. Replicon-level (mean probability over a replicon's sampled fragments)", ""]
    rows = [row(label, s["replicon"]) for label, s in scored.items()]
    rows.append(row("**ESM-2 + attention (final)**", esm_ens["replicon_level"]))
    out += [table(["model", *MAIN], rows), ""]

    out += ["## 5. Unweighted view of the sampled fragments (not prevalence-corrected)", "",
            "Useful only for checking the weighting; the plasmid share of the subsample "
            f"is {100 * test.label.mean():.1f}%, not the real 2.345%.", "",
            table(["model", *MAIN],
                  [row("ESM-2 + attention (final)", esm_ens["as_sampled"])]), ""]

    out += ["## 6. Evaluation subsample and ORF statistics", "",
            table(["split", "sampled", "plasmid", "effective size", "ORFs per fragment"],
                  ["| " + s + " | " + f"{v['n_sampled']} | {v['n_plasmid']} | "
                   f"{v['effective_n']:,.0f} | {v['orfs_per_fragment']:.2f} |"
                   for s, v in esm["splits"].items()]), ""]

    # The machine suspended during one seed, so wall-clock per seed is reported as a
    # median and the epoch timings are the honest per-epoch cost.
    epoch_seconds = [e["seconds"] for v in esm["per_seed"].values() for e in v["history"]]
    out += ["## 7. Cost", "",
            table(["item", "value"],
                  ["| ESM-2 embedding (all three splits) | see results/extract_features.log |",
                   "| attention-head training, median seconds per seed | "
                   f"{np.median([v['train_seconds'] for v in esm['per_seed'].values()]):.0f} s |",
                   f"| median seconds per training epoch | {np.median(epoch_seconds):.0f} s |",
                   f"| seeds trained | {len(esm['per_seed'])} |"]), ""]

    (results / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
