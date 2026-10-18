"""Gather every mob_recon shard into one fragment-level prediction table.

Reads out_dir/mobsuite/<shard>/contig_report.txt for each shard in the manifest.
The join against fragment_index is not cosmetic here: mob_recon drops contigs
below its --min_length (1000 bp by default), so its report is not guaranteed to
cover every input fragment, and a fragment it dropped must be counted as
unclassified rather than removed from the denominator.

Writes results_dir/predictions_mobsuite_<split>.csv with, per fragment:
    reported            whether mob_recon returned a row for it
    pred                1 where molecule_type == Plasmid
    score               a RANKING heuristic, not a probability (see rules.py)
    the cluster and mobility columns, kept for auditability
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
    ap.add_argument("--allow-missing-shards", action="store_true")
    args = ap.parse_args()

    conf = cfg.load_config(args.config)
    split = args.split or conf["split"]
    shard_dir, out_dir = Path(conf["shard_dir"]), Path(conf["out_dir"]) / "mobsuite"
    results = Path(conf["results_dir"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    index = pd.read_csv(shard_dir / f"fragment_index_{split}.csv")
    manifest = pd.read_csv(shard_dir / "manifest.csv")
    log(f"{split}: {len(index):,} fragments across {len(manifest)} shards")

    frames, absent = [], []
    for shard in manifest["shard"]:
        report = out_dir / str(shard) / "contig_report.txt"
        if not report.exists():
            absent.append(shard)
            continue
        df = parse.read_mob_contig_report(report)
        if df.empty:
            continue
        df["shard"] = shard
        frames.append(df)

    if absent:
        msg = (f"{len(absent)} shard(s) produced no contig_report.txt: "
               f"{absent[:5]}{' ...' if len(absent) > 5 else ''}")
        if not args.allow_missing_shards:
            raise SystemExit(
                f"FATAL: {msg}\n"
                "  Check server/logs/mobsuite-*_<task>.err for those array tasks and "
                "resubmit them (12_mobsuite.sh skips shards that already have output), "
                "or pass --allow-missing-shards to score an incomplete run.")
        log(f"WARNING: {msg}")

    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=list(parse.MOB_ALIASES) + ["shard"])
    calls = calls.rename(columns={"contig_id": "fragment_id"})
    log(f"mob_recon reported {len(calls):,} contigs; columns present: "
        f"{sorted(c for c in calls.columns if c != 'shard')}")

    dup = calls["fragment_id"].duplicated()
    if dup.any():
        raise SystemExit(f"FATAL: {int(dup.sum())} fragment ids appear in more than one "
                         "mob_recon result; the shards overlap, which they must not")

    unknown = set(calls["fragment_id"]) - set(index["fragment_id"])
    if unknown:
        raise SystemExit(
            f"FATAL: {len(unknown)} contig id(s) in the mob_recon reports are not "
            f"fragment ids of the {split} split, e.g. {sorted(unknown)[:3]}. "
            "mob_recon may have renamed contigs; the join key is unusable.")

    merged = index.merge(calls.drop(columns=["shard"]), on="fragment_id", how="left")
    merged["reported"] = merged["molecule_type"].notna()
    if len(merged) != len(index):
        raise SystemExit("FATAL: the join changed the row count")

    n_rep = int(merged["reported"].sum())
    n_miss = len(merged) - n_rep
    log(f"reported {n_rep:,} | unclassified {n_miss:,} | total {len(merged):,}")
    if n_rep + n_miss != len(index):
        raise SystemExit("FATAL: classified + unclassified does not equal the fragment count")
    if n_miss:
        dropped = merged.loc[~merged["reported"], "length"]
        log(f"  dropped fragment lengths: min {dropped.min()} median "
            f"{dropped.median():.0f} max {dropped.max()} "
            f"(min_length is configured as {conf['mobsuite']['min_length']})")

    pred = np.full(len(merged), -1, dtype=np.int8)
    score = np.full(len(merged), np.nan)
    mask = merged["reported"].to_numpy()
    if mask.any():
        sub = merged.loc[mask]
        labels = rules.mobsuite_labels(sub)
        pred[mask] = labels
        score[mask] = rules.mobsuite_score(sub, labels)
        log(f"  plasmid calls {int((labels == 1).sum()):5d} / {int(mask.sum()):5d} "
            f"reported ({labels.mean():.2%})")
    merged["pred"] = pred
    merged["score"] = score

    keep = [c for c in ("fragment_id", "shard", "replicon_id", "assembly_accession",
                        "label", "length", "length_bucket", "reported", "pred", "score",
                        "molecule_type", "primary_cluster", "secondary_cluster",
                        "circularity", "rep_types", "relaxase_types", "orit_types",
                        "predicted_mobility", "mash_neighbor", "mash_distance",
                        "mash_identification")
            if c in merged.columns]
    out = results / f"predictions_mobsuite_{split}.csv"
    merged[keep].to_csv(out, index=False)
    log(f"\nwrote {out}")

    if "primary_cluster" in merged.columns and mask.any():
        clusters = parse.is_present(merged.loc[mask, "primary_cluster"])
        log(f"fragments assigned to a plasmid cluster: {int(clusters.sum()):,}")


if __name__ == "__main__":
    main()
