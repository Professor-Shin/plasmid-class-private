"""Render every result file as a markdown table, for pasting into the report.

Keeps the report's numbers mechanically transcribed from the artefacts rather
than retyped. Run after step07, run_composition.py, train_eval.py and
gc_signal.py have produced their outputs; missing files are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "dataset pipeline1" / "reports"
# The directory names are overridable so the same script can render the laptop's
# results (the defaults) or the server run's, which writes to results_server:
#   ESM_RESULTS=results_server python analysis/dump_tables.py
BASE = ROOT / "baseline" / os.environ.get("BASE_RESULTS", "results_mixed")
ESM = ROOT / "esm2-attention" / os.environ.get("ESM_RESULTS", "results_mixed")
ANA = ROOT / "analysis" / "results"

pd.set_option("display.width", 200)


def _cell(v, floatfmt="%.4f") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, (int,)) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("int")):
        return f"{int(v):,}"
    if isinstance(v, float):
        return floatfmt % v
    return str(v)


def to_md(df: pd.DataFrame, floatfmt="%.4f", index=False) -> str:
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dependency)."""
    d = df.reset_index() if index else df
    head = list(d.columns)
    lines = ["| " + " | ".join(str(h) for h in head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)


def md(df: pd.DataFrame, cols=None, floatfmt="%.4f") -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return to_md(d, floatfmt)


def section(title: str):
    print(f"\n\n{'#' * 3} {title}\n")


def maybe(path: Path):
    if not path.exists():
        print(f"_(missing: {path.relative_to(ROOT)})_")
        return None
    return path


if __name__ == "__main__":
    section("Dataset: per-split summary")
    if p := maybe(REPORTS / "dataset_summary_mixed.csv"):
        print(md(pd.read_csv(p), ["split", "fragments", "fragments_chromosome",
                                  "fragments_plasmid", "plasmid_fraction", "replicons",
                                  "chromosomes", "plasmids", "species", "total_bp"]))

    section("Dataset: fragment-length distribution")
    if p := maybe(REPORTS / "fragment_length_distribution_mixed.csv"):
        print(md(pd.read_csv(p), ["length_bucket", "fragments", "fragment_fraction",
                                  "fragments_plasmid", "plasmid_fraction",
                                  "plasmid_replicons", "mean_length", "mean_gc",
                                  "n_train", "n_validation", "n_test"]))

    section("Dataset: length quantiles (from the run manifest)")
    if p := maybe(REPORTS / "run_manifest.json"):
        man = json.loads(p.read_text())
        for key in man.get("steps", {}):
            if key.startswith("step07") and man["steps"][key].get("length_stats"):
                st = man["steps"][key]["length_stats"]
                print(f"`{key}`: " + ", ".join(f"{k}={v:,}" for k, v in st.items()))
                ls = man["steps"][key].get("length_sampling")
                if ls:
                    print(f"\nsampling: `{ls}`")

    section("Composition baselines (full test split)")
    if p := maybe(BASE / "composition_baselines.csv"):
        print(md(pd.read_csv(p)))

    section("Composition baselines: GC / ori coefficients")
    if p := maybe(BASE / "composition_baselines.json"):
        js = json.loads(p.read_text())
        for name, e in js["models"].items():
            if "coefficients" in e:
                co = pd.Series(e["coefficients"]).sort_values(key=abs, ascending=False)
                print(f"\n**{name}** (standardised logistic-regression coefficients)\n")
                print(to_md(co.to_frame("coef").rename_axis("feature"), "%.3f", index=True))

    section("Composition baselines: by length bucket")
    if p := maybe(BASE / "composition_baselines.json"):
        js = json.loads(p.read_text())
        rows = []
        for name, e in js["models"].items():
            for r in e.get("test_by_length", []):
                rows.append({"model": name, "length_bucket": r["length_bucket"],
                             "n": r.get("n"), "mcc": r.get("mcc"), "f1": r.get("f1")})
        if rows:
            df = pd.DataFrame(rows)
            print(to_md(df.pivot(index="length_bucket", columns="model",
                                values="mcc").rename_axis("length_bucket"), index=True))

    section("Published tools as baselines: Platon and MOB-recon (test)")
    if p := maybe(BASE / "tool_baselines.csv"):
        df = pd.read_csv(p)
        print(md(df, ["model", "policy", "coverage", "n_scored", "mcc", "f1",
                      "precision", "recall", "specificity", "auprc",
                      "replicon_mcc", "hard_label_score"]))
        print("\n`policy` is how fragments the tool declined to classify are "
              "counted. `hard_label_score` marks rows whose AUROC/AUPRC come from "
              "a decision rather than a probability, so they are label-dominated. "
              "Both tools expect assembled contigs; these fragments are linear "
              "sub-sequences with no circularity and usually no oriT or rep "
              "hallmark, which is why the replicon-level column is the fairer "
              "comparison.")

    section("Platon and MOB-recon by length bucket (MCC, headline policy)")
    if p := maybe(BASE / "tool_baselines.json"):
        js = json.loads(p.read_text())
        rows = []
        for key, e in js["models"].items():
            if not key.endswith("|chromosome") or "test_by_length" not in e:
                continue
            for r in e["test_by_length"]:
                rows.append({"model": e["model"], "length_bucket": r["length_bucket"],
                             "n": r.get("n"), "mcc": r.get("mcc")})
        if rows:
            df = pd.DataFrame(rows)
            print(to_md(df.pivot(index="length_bucket", columns="model",
                                values="mcc").rename_axis("length_bucket"), index=True))
        else:
            print("_(no by-length breakdown in tool_baselines.json)_")

    section("ESM-2 feature ladder (test, weighted)")
    if p := maybe(ESM / "esm2_variants.csv"):
        print(md(pd.read_csv(p)))

    section("ESM-2 ladder by length bucket (MCC)")
    if p := maybe(ESM / "esm2_by_length.csv"):
        df = pd.read_csv(p)
        # The server run trains the ladder under more than one class-balance
        # regime, so the table carries a regime column and the pivot has to pick
        # one. The headline regime is the first one train_eval.py ran.
        if "regime" in df.columns and df["regime"].nunique() > 1:
            headline = df["regime"].iloc[0]
            print(f"_(regime: {headline}; the full table also holds "
                  f"{', '.join(r for r in df['regime'].unique() if r != headline)})_\n")
            df = df[df["regime"] == headline]
        print(to_md(df.pivot(index="length_bucket", columns="variant",
                            values="mcc").rename_axis("length_bucket"), index=True))

    section("ESM-2: replicon level and ORF-free subset")
    if p := maybe(ESM / "esm2_variants.json"):
        js = json.loads(p.read_text())
        rows = []
        for name, v in js["variants"].items():
            e = v["ensemble"]
            row = {"variant": name,
                   "frag_mcc": e["weighted"]["mcc"], "frag_f1": e["weighted"]["f1"],
                   "replicon_mcc": e["replicon_level"]["mcc"],
                   "replicon_f1": e["replicon_level"]["f1"]}
            of = e.get("orf_free", {})
            row["orf_free_n"] = of.get("n")
            row["orf_free_mcc"] = of.get("mcc")
            rows.append(row)
        print(md(pd.DataFrame(rows)))
        print("\nsplits: " + json.dumps(js["splits"]))

    section("GC signal: recall vs plasmid-host GC gap")
    if p := maybe(ANA / "gc_signal.json"):
        js = json.loads(p.read_text())
        print(f"test plasmid fragments: {js['n_test_plasmid_fragments']}, "
              f"with host chromosome: {js['plasmid_fragments_with_host']}, "
              f"median |GC gap| {js['gc_gap_median']:.4f}, "
              f"mean signed gap {js['gc_signed_mean']:+.4f}\n")
        rows = []
        for name, e in js["models"].items():
            rows.append({"model": name,
                         "n": e["n_plasmid_fragments_scored"],
                         "rho(gap, p_plasmid)": e["spearman_gap_vs_proba"]["rho"],
                         "p": e["spearman_gap_vs_proba"]["p"],
                         "rho(gap, correct)": e["spearman_gap_vs_correct"]["rho"]})
        print(md(pd.DataFrame(rows)))
    if p := maybe(ANA / "gc_signal_recall_by_gap.csv"):
        df = pd.read_csv(p)
        print()
        print(to_md(df.pivot(index="gc_gap_bin", columns="model",
                            values="recall").rename_axis("gc_gap_bin"), "%.3f", index=True))
