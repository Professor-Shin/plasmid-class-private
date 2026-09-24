"""Step 3: turn the result JSON files into the markdown tables used by REPORT.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import data, metrics  # noqa: E402

RUNS = [("plasflow", "balanced"), ("plasflow", "full"), ("zou", "balanced"), ("zou", "full"),
        ("plasclass", "balanced")]
NAMES = {"plasflow": "PlasFlow (vote of 3)", "zou": "Zou et al. (mean of 3)",
         "plasclass": "PlasClass (logistic regression)"}
STEMS = {"plasflow": "vote", "zou": "mean", "plasclass": "lr"}
REGIME = {"balanced": "A: balanced", "full": "B: full/imbalanced"}
MAIN = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
        "auroc", "auprc"]


def row(label, m, keys=MAIN):
    return "| " + label + " | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |"


def table(header, rows):
    line = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    return "\n".join([line, sep, *rows])


def main():
    cfg = data.load_config(HERE / "config.yaml")
    results = cfg["results_dir"]
    loaded = {}
    for method, regime in RUNS:
        path = results / f"{method}_{regime}.json"
        if path.exists():
            loaded[(method, regime)] = json.loads(path.read_text(encoding="utf-8"))

    test = data.load_table(cfg["dataset_dir"], "test")
    y = test["label"].to_numpy()
    out = ["# Baseline results", "",
           f"Test split: {len(y)} fragments, {int(y.sum())} plasmid "
           f"({y.mean() * 100:.3f}%), {int((y == 0).sum())} chromosome.", ""]

    out += ["## 1. Fragment-level test performance (ensemble, threshold 0.5)", ""]
    rows = [row("always-chromosome", metrics.binary_metrics(y, np.zeros(len(y)) + 1e-9))]
    for key, res in loaded.items():
        rows.append(row(f"{NAMES[key[0]]} - regime {REGIME[key[1]]}", res["ensemble"]["test"]))
    out += [table(["model", *MAIN], rows), ""]

    # The published accuracies were measured on class-balanced test sets, so a
    # balanced subsample of our test split is the only comparable number.
    balanced = data.balanced_subset(y, np.random.default_rng(cfg["random_seed"]))
    rows = []
    for (method, regime), res in loaded.items():
        path = results / "probs" / f"{method}_{regime}_{STEMS[method]}_test.npy"
        if path.exists():
            p = np.load(path)[balanced]
            rows.append(row(f"{NAMES[method]} - regime {REGIME[regime]}",
                            metrics.binary_metrics(y[balanced], p)))
    out += [f"## 2. Balanced test subsample ({len(balanced)} fragments, 1:1)", "",
            "Published accuracies (PlasFlow ~96%, Zou et al. 89-92%) were measured on "
            "balanced data; this table is the comparable view of our runs.", "",
            table(["model", *MAIN], rows), ""]

    out += ["## 3. Confusion matrices on the test split", "",
            table(["model", "TN", "FP", "FN", "TP", "plasmid recall", "plasmid precision"],
                  ["| " + f"{NAMES[k[0]]} - {REGIME[k[1]]}" + " | " +
                   " | ".join(str(r["ensemble"]["test"][c]) for c in ("tn", "fp", "fn", "tp")) +
                   f" | {r['ensemble']['test']['recall']:.4f} | "
                   f"{r['ensemble']['test']['precision']:.4f} |"
                   for k, r in loaded.items()]), ""]

    out += ["## 4. Individual members of each ensemble (test split)", ""]
    rows = []
    for (method, regime), res in loaded.items():
        for name, m in res["per_model"].items():
            rows.append(row(f"{method} {REGIME[regime]} - {name}", m["test"]))
    out += [table(["model", *MAIN], rows), ""]

    out += ["## 5. Decision-threshold variants (test split)", ""]
    rows = []
    for (method, regime), res in loaded.items():
        ens = res["ensemble"]
        rows.append(row(f"{method} {REGIME[regime]} - threshold 0.5", ens["test"]))
        tuned = ens["test_tuned_threshold"]
        rows.append(row(f"{method} {REGIME[regime]} - threshold {tuned['threshold']:.3f} "
                        f"(best F1 on validation)", tuned))
        for key, label in (("test_three_vote", "three-vote rule"),
                           ("test_majority_vote", "majority vote of members")):
            if key in ens:
                rows.append(row(f"{method} {REGIME[regime]} - {label}", ens[key]))
    out += [table(["model", *MAIN], rows), ""]

    pf = [(k, r) for k, r in loaded.items() if k[0] == "plasflow"]
    if pf:
        out += ["## 6. PlasFlow probability-threshold rule (0.7)", "",
                table(["model", "classified", "unclassified %", "unclassified % (plasmid)",
                       "accuracy on classified", "F1 on classified", "recall on classified"],
                      ["| " + f"{REGIME[k[1]]}" + " | " +
                       f"{r['ensemble']['test_threshold_rule']['n_classified']} | "
                       f"{r['ensemble']['test_threshold_rule']['unclassified_rate'] * 100:.2f} | "
                       f"{r['ensemble']['test_threshold_rule']['unclassified_rate_plasmid'] * 100:.2f} | "
                       f"{r['ensemble']['test_threshold_rule'].get('classified_accuracy', float('nan')):.4f} | "
                       f"{r['ensemble']['test_threshold_rule'].get('classified_f1', float('nan')):.4f} | "
                       f"{r['ensemble']['test_threshold_rule'].get('classified_recall', float('nan')):.4f} |"
                       for k, r in pf]), ""]

    out += ["## 7. Replicon-level performance (mean fragment probability per replicon)", ""]
    rows = [row(f"{NAMES[k[0]]} - {REGIME[k[1]]}", r["ensemble"]["test_replicon_level"])
            for k, r in loaded.items()]
    out += [table(["model", *MAIN], rows), ""]

    out += ["## 8. Training cost", "",
            table(["run", "train fragments", "members", "total training seconds"],
                  ["| " + f"{NAMES[k[0]]} - {REGIME[k[1]]}" + " | "
                   f"{r['n_train_fragments']} | {len(r['per_model'])} | "
                   f"{sum(m['train_seconds'] for m in r['per_model'].values()):.0f} |"
                   for k, r in loaded.items()]), ""]

    (results / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
