"""Render every result file as a markdown table, for pasting into the report.

Keeps the report's numbers mechanically transcribed from the artefacts rather
than retyped. Run after step07, run_composition.py, train_eval.py and
gc_signal.py have produced their outputs; missing files are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "dataset pipeline1" / "reports"
# The directory names are overridable so the same script can render the laptop's
# results (the defaults) or the server run's, which writes to results_server:
#   ESM_RESULTS=results_server python analysis/dump_tables.py
BASE = ROOT / "baseline" / os.environ.get("BASE_RESULTS", "results_mixed")
ESM = ROOT / "esm2-attention" / os.environ.get("ESM_RESULTS", "results_mixed")
ANA = ROOT / "analysis" / "results"

# The five baselines the report compares, in the order they are discussed, each
# named by the ONE row that is its published default operating point. Anything
# else in tool_baselines.csv is an alternative mode or an unclassified policy and
# belongs in the detailed table below, not in the headline.
HEADLINE = [
    ("Platon",        "platon_accuracy",        "chromosome", "filter mode `accuracy`, the tool's default"),
    ("MOB-suite",     "mobsuite_molecule_type", "chromosome", "`mob_recon` molecule_type"),
    ("PLASMe",        "plasme_balance",         "chromosome", "preset `balance` = the CLI defaults"),
    ("PlasmidHunter", "plasmidhunter",          "chromosome", "posterior >= 0.5"),
]

pd.set_option("display.width", 200)


def _cell(v, floatfmt="%.4f") -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
        return "-"
    # bool is a subclass of int, so it has to be caught first or True prints as 1.
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
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


def _precision_at_balance(recall: float, specificity: float) -> float | None:
    """What this row's precision would be on a 1:1 test set.

    Sensitivity and specificity do not depend on prevalence; precision does.
    The four tools publish their numbers on balanced benchmarks (PLASMe 6,674
    plasmid vs 6,674 chromosome fragments, PlasmidHunter 3,000 vs 3,000), while
    this test split is the 1:40 that genomes actually are. Without this column
    a reader compares 0.07 here against 0.72 in the paper and concludes the tool
    is broken, when the two numbers are the same classifier measured under
    different base rates.
    """
    try:
        denom = recall + (1.0 - specificity)
        return round(recall / denom, 4) if denom > 0 else None
    except TypeError:
        return None


def _zou_n_scored(model: str, regime: str) -> int | None:
    """How many test fragments the Zou row was scored on, from its own JSON.

    The CSV carries n_train but not the test count, and leaving the column
    empty invited the reading that Zou was scored on something other than the
    16,944 fragments every other row uses.
    """
    path = BASE / "zou_mixed.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
        return int(blob["models"][f"{model}|{regime}"]["test"]["n"])
    except (KeyError, ValueError, TypeError):
        return None


def headline_table() -> pd.DataFrame:
    """One row per baseline at its published default, for the summary table.

    Built from the same artefacts as the detailed tables below - nothing is
    retyped - but reduced to one comparable line each, because the full tool
    table is 3 policies x up to 5 modes x 4 tools and no one can read a ranking
    out of 50 rows.
    """
    rows = []
    tools = BASE / "tool_baselines.csv"
    if tools.exists():
        df = pd.read_csv(tools)
        for label, model, policy, note in HEADLINE:
            hit = df[(df["model"] == model) & (df["policy"] == policy)]
            if hit.empty:
                rows.append({"baseline": label, "note": f"not run ({note})"})
                continue
            r = hit.iloc[0]
            rows.append({
                "baseline": label, "trained here": "no",
                "coverage": r["coverage"], "n_scored": int(r["n_scored"]),
                "mcc": r["mcc"], "f1": r["f1"], "precision": r["precision"],
                "recall": r["recall"], "specificity": r.get("specificity"),
                "precision@1:1": _precision_at_balance(r["recall"],
                                                       r.get("specificity")),
                "auprc": r["auprc"],
                "replicon_mcc": r["replicon_mcc"],
                # A hard label has no threshold to move, so there is no ceiling
                # to quote - printing mcc back again would read as if tuning
                # had been tried and had changed nothing.
                "mcc tuned": (None if r["hard_label_score"]
                              else r.get("mcc_oracle")),
                "tuned on": ("-" if r["hard_label_score"] else "test (ceiling)"),
                "real probability": "no" if r["hard_label_score"] else "yes",
                "note": note})
    zou = BASE / "zou_mixed.csv"
    if zou.exists():
        df = pd.read_csv(zou)
        # `balanced` because that is the regime the paper trains under, and this
        # table is each baseline at the operating point its authors ship.
        #
        # The earlier choice here was `full`, on the reasoning that the other
        # four tools meet the test split at its natural 40:1 and so should this
        # one. That reasoning does not hold: `regime` is the class balance at
        # TRAINING time, and every regime is scored on the same 16,944 test
        # fragments either way (zou_mixed.json: test.n = 16944 for all four).
        # What `full` changes is that an unweighted model under 40:1 stops
        # predicting the positive class at all - MCC 0.0000 at recall 0.0000 -
        # so the headline was showing a collapsed model rather than the
        # architecture. Both regimes stay in the detailed Zou table below.
        pick = df[(df["model"] == "zou") & (df["regime"] == "balanced")]
        pick = pick if not pick.empty else df.head(1)
        if not pick.empty:
            r = pick.iloc[0]
            rows.append({
                "baseline": "Zou et al.", "trained here": "yes",
                "coverage": 1.0, "n_scored": _zou_n_scored(r["model"], r["regime"]),
                "mcc": r["mcc"], "f1": r["f1"], "precision": r["precision"],
                "recall": r["recall"], "specificity": r.get("specificity"),
                "precision@1:1": _precision_at_balance(r["recall"],
                                                       r.get("specificity")),
                "auprc": r["auprc"],
                "replicon_mcc": r["replicon_mcc"],
                "mcc tuned": r.get("mcc_tuned"),
                "tuned on": "validation",
                "real probability": "yes",
                "note": f"retrained, regime {r['regime']}, raw 6-mer counts"})
    out = pd.DataFrame(rows)
    # A baseline that was not run contributes a row with no n_scored, which turns
    # the whole column to float and prints 16,944 as 16944.0000. Int64 is the
    # nullable integer type, so the missing one stays missing and the rest stay
    # integers.
    if "n_scored" in out.columns:
        out["n_scored"] = out["n_scored"].astype("Int64")
    return out


if __name__ == "__main__":
    section("Headline: the five baselines at their published operating points")
    hl = headline_table()
    if hl.empty:
        print("_(no baseline results yet)_")
    else:
        print(md(hl, ["baseline", "trained here", "coverage", "n_scored", "mcc", "f1",
                      "precision", "recall", "auprc", "replicon_mcc",
                      "real probability", "note"]))
        print("\nOne row per baseline, at the operating point its authors ship. "
              "`coverage` is the share of test fragments the tool was willing to "
              "classify; fragments it declined are counted as chromosome here, "
              "which is the conservative choice (the other two policies are in the "
              "detailed table). `real probability` says whether AUPRC for that row "
              "comes from a score or from a binary decision - where it is `no`, the "
              "AUPRC is label-dominated and not comparable to the rows where it is "
              "`yes`. `replicon_mcc` aggregates a replicon's fragments before "
              "scoring and is the fairer comparison for the four tools that expect "
              "assembled contigs. Four of the five are run as published; Zou et al. "
              "is retrained on this dataset, which is an advantage the other four "
              "do not have.")
        print("\n`precision@1:1` is this row's precision recomputed for a test set "
              "with equal numbers of plasmid and chromosome fragments, which is "
              "what these tools were benchmarked on (PLASMe 6,674 vs 6,674; "
              "PlasmidHunter 3,000 vs 3,000) and is the only way to compare "
              "against the numbers in their papers. Sensitivity and specificity "
              "carry over unchanged between base rates; precision does not. "
              "`mcc tuned` is the best MCC the row's score reaches at any "
              "threshold, and `tuned on` says what that is worth: for the four "
              "tools the threshold is fitted on the test split itself, because "
              "none of them is trained here and no other split was run, so it is "
              "a ceiling and not an operating point. Zou's is fitted on "
              "validation and is therefore a choice that could have been made in "
              "advance. A dash means the row is a hard label with no threshold "
              "to move.")

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

    section("Dataset: class balance at fragment, replicon and assembly level")
    if p := maybe(ANA / "dataset_class_balance.csv"):
        print(md(pd.read_csv(p), ["split", "fragments", "fragments_plasmid",
                                  "fragment_plasmid_rate", "imbalance_1_to_n",
                                  "replicons", "replicons_plasmid",
                                  "replicon_plasmid_rate", "assemblies",
                                  "assemblies_with_plasmid", "species", "clusters",
                                  "total_bp", "plasmid_bp", "plasmid_bp_rate"]))
        print("\nThe prevalence a model sees depends on the level it is scored at: "
              "fragments are roughly 1 plasmid in 40, replicons closer to 1 in 2.4. "
              "A fragment-level MCC and a replicon-level MCC are therefore not two "
              "views of the same difficulty, and the headline table reports both.")

    section("Dataset: fragment length by split and class")
    if p := maybe(ANA / "dataset_length_quantiles.csv"):
        print(md(pd.read_csv(p), ["split", "class", "n", "min", "p1", "p10", "p25",
                                  "p50", "p75", "p90", "p99", "max", "mean",
                                  "total_Mbp"], floatfmt="%.1f"))
        print("\nLengths were sampled log-uniformly between 1 kb and 100 kb "
              "independently of class, and no fragment was truncated after the "
              "fact (`sampled_length == length` everywhere). Plasmid quantiles "
              "still sit lower, because the sampler can only draw a length the "
              "replicon can supply and most plasmids are small: 173 of the 381 "
              "plasmid replicons hold under 50 kb in total, against 0 of the 527 "
              "chromosomes - see `replicons_under_50kb` in the replicon-structure "
              "table. Length is therefore weakly informative about class on this "
              "dataset, which is why the Zou baseline is reported both on raw "
              "k-mer counts (which carry length) and on frequencies (which do not).")

    section("Dataset: GC content by class")
    if p := maybe(ANA / "dataset_gc_by_class.csv"):
        print(md(pd.read_csv(p), ["length_bucket", "chromosome_n", "chromosome_gc_mean",
                                  "chromosome_gc_sd", "plasmid_n", "plasmid_gc_mean",
                                  "plasmid_gc_sd", "gc_gap", "gc_gap_sd_units"]))
        print("\n`gc_gap` is the plasmid mean minus the chromosome mean, and "
              "`gc_gap_sd_units` divides it by the pooled within-class spread. That "
              "second number is the one that bounds a GC-only classifier: a gap far "
              "below one standard deviation cannot separate the classes however it "
              "is thresholded, which is what the gc_only row of the composition "
              "table shows from the other direction.")

    section("Dataset: how fragments stack into replicons and isolates")
    if p := maybe(ANA / "dataset_replicon_structure.csv"):
        print(md(pd.read_csv(p), ["split", "class", "replicons",
                                  "frags_per_replicon_min", "frags_per_replicon_median",
                                  "frags_per_replicon_max", "replicon_kbp_median",
                                  "replicon_kbp_max", "replicons_under_50kb",
                                  "replicons_under_50kb_share",
                                  "replicons_per_assembly_median",
                                  "replicons_per_assembly_max"], floatfmt="%.1f"))
        print("\nThe replicon-level metrics average this many fragments per replicon "
              "before thresholding, so a replicon built from 200 fragments is a far "
              "more reliable call than one built from 2 - and plasmid replicons are "
              "the short ones.")

    section("Dataset: what a per-assembly shard actually contains")
    if p := maybe(ANA / "dataset_shard_composition.csv"):
        print(md(pd.read_csv(p), ["split", "shards", "chromosome_only", "plasmid_only",
                                  "chromosome_and_plasmid", "complete_isolates",
                                  "replicon_coverage_median", "replicon_coverage_min",
                                  "fragments_per_shard_median", "Mbp_per_shard_median"],
                 floatfmt="%.3f"))
        print("\n**Read the four published tools against this table.** They are "
              "given one shard at a time as \"the contigs of one isolate\", but the "
              "dataset splits at replicon level, not assembly level, so a test shard "
              "holds only the replicons of that isolate that landed in test. Most "
              "test shards are single-class: a plasmid-only shard has no chromosome "
              "to be confused with and is an easier input than a real draft assembly, "
              "a chromosome-only shard is a harder one, and only a handful look like "
              "what mob_recon and Platon were built for. The splits themselves are "
              "sound - no cluster and no replicon crosses a split, which is what "
              "prevents leakage - this is a limitation of the per-assembly framing, "
              "not of the dataset.")

    section("Dataset: taxonomic composition (top genera)")
    if p := maybe(ANA / "dataset_taxonomy.csv"):
        print(md(pd.read_csv(p), ["genus", "fragments", "fragment_share", "assemblies",
                                  "species", "plasmid_fragments", "n_train",
                                  "n_validation", "n_test"]))
        print("\nPLASMe selects a Transformer per taxonomic order and MOB-suite's "
              "clusters are built from RefSeq, so which genera dominate is part of "
              "how their numbers should be read.")

    section("Dataset: integrity and quality")
    if p := maybe(ANA / "dataset_stats.json"):
        ov = json.loads(p.read_text()).get("split_overlap", {})
        rows = []
        for key in ("cluster_id", "replicon_id", "assembly_accession", "species_taxid"):
            if key not in ov:
                continue
            rows.append({"key": key, "distinct": ov[key]["total_distinct"],
                         **{f"shared {k}": v for k, v in ov[key]["shared"].items()}})
        if rows:
            print(md(pd.DataFrame(rows)))
        print(f"\nleakage check (clusters and replicons must be disjoint): "
              f"**{'passed' if ov.get('leakage_ok') else 'FAILED'}**. Assemblies and "
              "species are expected to be shared, because the split is at cluster "
              "level and an isolate's replicons can fall on either side of it.")
    if p := maybe(ANA / "dataset_quality.csv"):
        print()
        print(md(pd.read_csv(p), ["split", "fragments", "with_non_acgt", "non_acgt_max",
                                  "non_acgt_mean", "length_below_request",
                                  "length_below_request_plasmid"], floatfmt="%.6f"))
        print("\n`length_below_request` counts fragments the sampler wanted longer "
              "than the replicon could supply. It is zero throughout, which says "
              "the sampler drew within each replicon's capacity instead of "
              "truncating afterwards - a non-zero value here would mean the "
              "dataset build changed.")

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

    section("Published tools as baselines: every mode and policy (test)")
    if p := maybe(BASE / "tool_baselines.csv"):
        df = pd.read_csv(p)
        print(md(df, ["model", "policy", "coverage", "n_scored", "mcc", "f1",
                      "precision", "recall", "specificity", "auprc",
                      "replicon_mcc", "mcc_oracle", "oracle_threshold",
                      "hard_label_score"]))
        print("\n`policy` is how fragments the tool declined to classify are "
              "counted. `hard_label_score` marks rows whose AUROC/AUPRC come from "
              "a decision rather than a probability, so they are label-dominated. "
              "Both tools expect assembled contigs; these fragments are linear "
              "sub-sequences with no circularity and usually no oriT or rep "
              "hallmark, which is why the replicon-level column is the fairer "
              "comparison.")

    section("Published tools by length bucket (MCC, headline policy)")
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

    section("Zou et al., retrained on this dataset (test)")
    if p := maybe(BASE / "zou_mixed.csv"):
        print(md(pd.read_csv(p), ["model", "regime", "n_features", "n_train", "mcc",
                                  "f1", "precision", "recall", "specificity", "auprc",
                                  "auroc", "mcc_tuned", "tuned_threshold",
                                  "mcc_three_vote",
                                  "replicon_mcc", "train_seconds"]))
        print("\n`zou` feeds raw canonical 6-mer counts to the published "
              "architecture, which is what the paper does - but on variable-length "
              "fragments a count carries the fragment's length, so part of what the "
              "model reads is length rather than composition. `zou_lengthnorm` is the "
              "same model on per-fragment frequencies, with the length term removed. "
              "The gap between the two rows is how much of the counts model was "
              "length. `regime` is the class balance it trained under: `full` is the "
              "dataset's natural 40:1, `balanced` a 1:1 subsample as the paper used. "
              "`mcc_three_vote` is the paper's majority vote over the three seeds, "
              "which is not the same as thresholding their mean.")

    section("What each baseline cost to run")
    if p := maybe(BASE / "tool_runtime.csv"):
        print(md(pd.read_csv(p), ["tool", "shards", "wall_hours_total",
                                  "cpu_hours_total", "cpu_per_wall", "peak_rss_gib",
                                  "cpu_s_per_Mbp", "wall_s_per_Mbp", "Mbp_scored",
                                  "fragments_scored", "wall_only", "source"],
                 floatfmt="%.2f"))
        print("\n`cpu_s_per_Mbp` is the column to compare on: wall clock depends on "
              "how many shards were allowed to run at once, CPU seconds per megabase "
              "does not. `cpu_per_wall` above 1 means the tool really used the "
              "threads it was given. `wall_only` marks a tool whose record came from "
              "banner timestamps in its shard logs rather than from the timing "
              "wrapper, so it has no CPU or memory figure - not that it used none. "
              "Zou is the odd row out: it is trained rather than invoked per shard, "
              "and its figure covers featurising all three splits plus every regime "
              "and variant, so it is not a per-shard inference cost.")

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

    # Every section above is what the models scored. These last ones are how many
    # records went through them and what the cluster spent doing it - the numbers
    # the dataset and methods sections need, which no result file carries.
    if os.environ.get("PROVENANCE", "1") != "0":
        try:
            import run_provenance
        except Exception as exc:                        # noqa: BLE001
            print(f"\n\n_(run_provenance unavailable: {exc})_")
        else:
            run_provenance.render(no_sacct=os.environ.get("NO_SACCT") == "1")
