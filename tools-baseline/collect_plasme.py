"""Gather every PLASMe shard into one fragment-level prediction table.

Reads out_dir/plasme/<shard>/PLASMe_candidate.csv - the table PLASMe leaves in
its temporary directory, which the runner keeps on purpose. PLASMe's own
`<output>_report.csv` lists only the contigs it called plasmid, so it carries no
denominator and no rejected contig; the candidate table has one row per contig
that aligned to the reference plasmid database, with the BLASTn identity and
coverage and, where the Transformer stage ran, its probability.

Every published decision mode is a threshold triple over those three numbers
(toolbase/rules.PLASME_MODES), so a single run yields all three offline, exactly
as --characterize does for Platon.

Writes results_dir/predictions_plasme_<split>.csv with, per fragment:
    reported            whether PLASMe aligned the fragment at all
    identity, coverage  the best BLASTn hit's identity and coverage
    transformer_score   the Transformer probability, or -1 when it did not run
    has_transformer     whether that probability is a real probability
    pred_<mode>         0/1 for high_sensitivity, balance, high_precision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import cfg, parse, rules  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--allow-missing-shards", action="store_true",
                    help="continue when some shards produced no output (debugging only)")
    args = ap.parse_args()

    conf = cfg.load_config(args.config)
    split = args.split or conf["split"]
    shard_dir, out_dir = Path(conf["shard_dir"]), Path(conf["out_dir"]) / "plasme"
    results = Path(conf["results_dir"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    index = pd.read_csv(shard_dir / f"fragment_index_{split}.csv")
    manifest = pd.read_csv(shard_dir / "manifest.csv")
    log(f"{split}: {len(index):,} fragments across {len(manifest)} shards")

    frames, empty, absent, mismatched = [], [], [], []
    for shard in manifest["shard"]:
        # The runner writes a .done sentinel after a successful run, so a shard
        # that legitimately aligned nothing is distinguishable from one that
        # never ran. Without it an empty candidate table and a missing job look
        # identical, and a crashed shard would silently become "all chromosome".
        done = out_dir / str(shard) / ".done"
        table = out_dir / str(shard) / "PLASMe_candidate.csv"
        if not done.exists():
            absent.append(shard)
            continue
        df = parse.read_plasme_candidate(table)
        if df.empty:
            empty.append(shard)
            continue
        df["shard"] = shard
        frames.append(df)

        # Verify the offline rule against PLASMe's OWN answer, for this shard.
        # 16_plasme.sh runs at the CLI defaults, which are the `balance` preset,
        # and PLASMe writes the contigs it called into <output>_report.csv - so
        # every shard carries its own ground truth for one of the three modes
        # and checking it costs nothing. server/13_smoke_tools.sh runs the same
        # check before the array starts; this one covers all 116 shards rather
        # than the two the gate can afford.
        report = parse.read_plasme_report(out_dir / str(shard) / "plasmid.fna_report.csv")
        if not report.empty or (out_dir / str(shard) / "plasmid.fna_report.csv").exists():
            called = set(report["contig_id"].astype(str)) if not report.empty else set()
            derived = rules.plasme_labels(df, *rules.PLASME_MODES["balance"])
            truth = df["contig_id"].astype(str).isin(called).to_numpy().astype(np.int8)
            if not np.array_equal(derived, truth):
                mismatched.append((shard, int((derived != truth).sum()), len(df)))

    if mismatched:
        total = sum(n for _, n, _ in mismatched)
        raise SystemExit(
            f"FATAL: the offline PLASMe rule disagrees with PLASMe's own report on "
            f"{len(mismatched)} shard(s), {total} contig(s) in all. First few: "
            f"{mismatched[:5]}\n"
            "  The three preset modes are re-derived from the candidate table on the "
            "assumption that they are thresholds over identity, coverage and the "
            "Transformer probability (toolbase/rules.PLASME_MODES, transcribed from "
            "PLASMe.py:plasme_output). That assumption has broken - most likely the "
            "installed PLASMe is a different version.\n"
            "  Run tools-baseline/verify_plasme_modes.py on one of those shards to "
            "see the disagreeing contigs, and do not score this run until it passes.")

    if absent:
        msg = (f"{len(absent)} shard(s) have no finished PLASMe run: "
               f"{absent[:5]}{' ...' if len(absent) > 5 else ''}")
        if not args.allow_missing_shards:
            raise SystemExit(
                f"FATAL: {msg}\n"
                "  Check server/logs/plasme-serial-<task>.out for those shards and "
                "resubmit (16_plasme.sh skips shards that already carry a .done "
                "sentinel), or pass --allow-missing-shards to score an incomplete run.")
        log(f"WARNING: {msg}")
    if empty:
        log(f"note: {len(empty)} shard(s) aligned no contig to the reference database")

    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=list(parse.PLASME_ALIASES) + ["shard"])
    calls = calls.rename(columns={"contig_id": "fragment_id"})
    log(f"PLASMe aligned {len(calls):,} contigs; columns present: "
        f"{sorted(c for c in calls.columns if c != 'shard')}")
    log(f"offline mode rule verified against PLASMe's own report on "
        f"{len(frames)} shard(s): all agree")

    dup = calls["fragment_id"].duplicated()
    if dup.any():
        raise SystemExit(f"FATAL: {int(dup.sum())} fragment ids appear in more than one "
                         "PLASMe result; the shards overlap, which they must not")

    unknown = set(calls["fragment_id"]) - set(index["fragment_id"])
    if unknown:
        raise SystemExit(
            f"FATAL: {len(unknown)} contig id(s) in the PLASMe tables are not "
            f"fragment ids of the {split} split, e.g. {sorted(unknown)[:3]}.")

    merged = index.merge(calls.drop(columns=["shard"]), on="fragment_id", how="left")
    merged["reported"] = merged["identity"].notna()
    if len(merged) != len(index):
        raise SystemExit("FATAL: the join changed the row count")

    n_rep = int(merged["reported"].sum())
    n_miss = len(merged) - n_rep
    log(f"reported {n_rep:,} | unclassified {n_miss:,} | total {len(merged):,}")
    if n_rep + n_miss != len(index):
        raise SystemExit("FATAL: classified + unclassified does not equal the fragment count")

    # A contig PLASMe aligned but did not pass to the Transformer carries -1,
    # which is a decision, not a probability. Keeping the two apart is what lets
    # score_tools.py report a genuinely threshold-free PLASMe number alongside
    # the label-dominated ones.
    score = pd.to_numeric(merged.get("transformer_score"), errors="coerce").to_numpy(
        dtype=float) if "transformer_score" in merged.columns else np.full(len(merged), np.nan)
    merged["transformer_score"] = score
    merged["has_transformer"] = merged["reported"].to_numpy() & (score > rules.PLASME_NO_TRANSFORMER)
    log(f"  Transformer scored {int(merged['has_transformer'].sum()):,} of {n_rep:,} "
        f"aligned contigs; the rest were decided by BLASTn alone")

    for mode, thresholds in rules.PLASME_MODES.items():
        pred = np.full(len(merged), -1, dtype=np.int8)
        mask = merged["reported"].to_numpy()
        if mask.any():
            pred[mask] = rules.plasme_labels(merged.loc[mask], *thresholds)
        merged[f"pred_{mode}"] = pred
        called = pred >= 0
        share = (pred[called] == 1).mean() if called.any() else float("nan")
        log(f"  mode {mode:17s} (i>={thresholds[0]}, c>={thresholds[1]}, "
            f"p>{thresholds[2]}): plasmid {int((pred == 1).sum()):5d} / "
            f"{int(called.sum()):5d} aligned ({share:.2%})")

    keep = [c for c in ("fragment_id", "shard", "replicon_id", "assembly_accession",
                        "label", "length", "length_bucket", "reported", "identity",
                        "coverage", "transformer_score", "has_transformer", "order",
                        "overlap") if c in merged.columns]
    keep += [f"pred_{m}" for m in rules.PLASME_MODES]
    out = results / f"predictions_plasme_{split}.csv"
    merged[keep].to_csv(out, index=False)
    log(f"\nwrote {out}")

    if merged["has_transformer"].any():
        tr = merged.loc[merged["has_transformer"], "transformer_score"]
        log(f"Transformer probability: min {tr.min():.3f} median {tr.median():.3f} "
            f"max {tr.max():.3f}")
        by_label = merged.loc[merged["has_transformer"]].groupby("label")[
            "transformer_score"].median()
        log(f"  median probability by true label: {by_label.to_dict()}")


if __name__ == "__main__":
    main()
