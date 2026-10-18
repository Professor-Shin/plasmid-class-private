"""Gather every Platon shard into one fragment-level prediction table.

Reads out_dir/platon/<shard>/<shard>.tsv for each shard in the manifest, joins it
back onto fragment_index so that nothing can quietly go missing, and applies the
three published filter modes (toolbase/rules.py) to the single --characterize run.

Writes results_dir/predictions_platon_<split>.csv with, per fragment:
    reported            whether Platon characterised it at all
    rds                 the replicon distribution score - a continuous score, so
                        Platon can be scored threshold-free like the other baselines
    pred_<mode>         0/1 for each of sensitivity, accuracy, specificity
    the hallmark columns the accuracy rule consumes, kept for auditability
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
    shard_dir, out_dir = Path(conf["shard_dir"]), Path(conf["out_dir"]) / "platon"
    results = Path(conf["results_dir"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    index = pd.read_csv(shard_dir / f"fragment_index_{split}.csv")
    manifest = pd.read_csv(shard_dir / "manifest.csv")
    log(f"{split}: {len(index):,} fragments across {len(manifest)} shards")

    frames, empty, absent = [], [], []
    for shard in manifest["shard"]:
        tsv = out_dir / str(shard) / f"{shard}.tsv"
        if not tsv.exists():
            absent.append(shard)
            continue
        df = parse.read_platon_tsv(tsv)
        if df.empty:
            empty.append(shard)
            continue
        df["shard"] = shard
        frames.append(df)

    if absent:
        msg = (f"{len(absent)} shard(s) produced no Platon TSV: "
               f"{absent[:5]}{' ...' if len(absent) > 5 else ''}")
        if not args.allow_missing_shards:
            raise SystemExit(
                f"FATAL: {msg}\n"
                "  Check server/logs/platon-*_<task>.err for those array tasks and "
                "resubmit them (11_platon.sh skips shards that already have output), "
                "or pass --allow-missing-shards to score an incomplete run.")
        log(f"WARNING: {msg}")
    if empty:
        log(f"note: {len(empty)} shard(s) returned an empty TSV (no contig characterised)")

    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=list(parse.PLATON_ALIASES) + ["shard"])
    calls = calls.rename(columns={"contig_id": "fragment_id"})
    log(f"Platon characterised {len(calls):,} contigs; columns present: "
        f"{sorted(c for c in calls.columns if c != 'shard')}")

    dup = calls["fragment_id"].duplicated()
    if dup.any():
        raise SystemExit(f"FATAL: {int(dup.sum())} fragment ids appear in more than one "
                         "Platon result; the shards overlap, which they must not")

    # A left join is the whole point: a fragment Platon never reported stays in the
    # table with reported=False instead of disappearing from the denominator.
    merged = index.merge(calls.drop(columns=["shard"]), on="fragment_id", how="left")
    merged["reported"] = merged["rds"].notna()
    if len(merged) != len(index):
        raise SystemExit("FATAL: the join changed the row count")

    n_rep = int(merged["reported"].sum())
    n_miss = len(merged) - n_rep
    log(f"reported {n_rep:,} | unclassified {n_miss:,} | total {len(merged):,}")
    if n_rep + n_miss != len(index):
        raise SystemExit("FATAL: classified + unclassified does not equal the fragment count")

    # Modes are applied only to the rows Platon actually characterised; the rest
    # carry -1 so score_tools.py can apply each unclassified policy explicitly.
    for mode in conf["platon"]["modes"]:
        missing = rules.missing_for_mode(merged.loc[merged["reported"]], mode)
        col = f"pred_{mode}"
        if missing:
            log(f"  mode {mode:12s} SKIPPED - TSV lacks {missing} "
                "(was Platon run without --characterize?)")
            merged[col] = -1
            continue
        pred = np.full(len(merged), -1, dtype=np.int8)
        sub = merged.loc[merged["reported"]]
        pred[merged["reported"].to_numpy()] = rules.platon_labels(sub, mode)
        merged[col] = pred
        called = pred >= 0
        log(f"  mode {mode:12s} plasmid {int((pred == 1).sum()):5d} / "
            f"{int(called.sum()):5d} characterised "
            f"({(pred[called] == 1).mean():.2%})")

    keep = [c for c in ("fragment_id", "shard", "replicon_id", "assembly_accession",
                        "label", "length", "length_bucket", "reported", "rds", "n_orfs",
                        "circularity", "inc_types", "n_replication", "n_mobilization",
                        "n_orit", "n_conjugation", "n_rrna", "n_plasmid_hits")
            if c in merged.columns]
    keep += [f"pred_{m}" for m in conf["platon"]["modes"]]
    out = results / f"predictions_platon_{split}.csv"
    merged[keep].to_csv(out, index=False)
    log(f"\nwrote {out}")

    if merged["reported"].any():
        rds = merged.loc[merged["reported"], "rds"]
        log(f"RDS over characterised contigs: min {rds.min():.2f} "
            f"median {rds.median():.2f} max {rds.max():.2f}")
        by_label = merged.loc[merged["reported"]].groupby("label")["rds"].median()
        log(f"  median RDS by true label: {by_label.to_dict()}")


if __name__ == "__main__":
    main()
