"""Gather every PlasmidHunter shard into one fragment-level prediction table.

Reads out_dir/plasmidhunter/<shard>/predictions.tsv for each shard in the
manifest. PlasmidHunter builds its feature vector from DIAMOND hits of the
contig's Prodigal-predicted proteins, so a contig whose proteins hit nothing in
the database produces no row at all - the same situation as mob_recon's length
cutoff, and handled the same way: a left join against the fragment index keeps
the fragment with reported=False instead of removing it from the denominator.

Unlike Platon's filter modes and mob_recon's assignment, PlasmidHunter's output
IS a probability (a Gaussian naive Bayes posterior), so this is the one published
tool here that can be scored threshold-free without a ranking heuristic.

Writes results_dir/predictions_plasmidhunter_<split>.csv with, per fragment:
    reported            whether PlasmidHunter returned a row for it
    pred                its own class call (0 chromosome / 1 plasmid)
    proba_plasmid       the posterior probability of the plasmid class
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
    shard_dir = Path(conf["shard_dir"])
    out_dir = Path(conf["out_dir"]) / "plasmidhunter"
    results = Path(conf["results_dir"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    index = pd.read_csv(shard_dir / f"fragment_index_{split}.csv")
    manifest = pd.read_csv(shard_dir / "manifest.csv")
    log(f"{split}: {len(index):,} fragments across {len(manifest)} shards")

    frames, empty, absent = [], [], []
    for shard in manifest["shard"]:
        done = out_dir / str(shard) / ".done"
        tsv = out_dir / str(shard) / "predictions.tsv"
        if not done.exists():
            absent.append(shard)
            continue
        df = parse.read_plasmidhunter_tsv(tsv)
        if df.empty:
            empty.append(shard)
            continue
        df["shard"] = shard
        frames.append(df)

    if absent:
        msg = (f"{len(absent)} shard(s) have no finished PlasmidHunter run: "
               f"{absent[:5]}{' ...' if len(absent) > 5 else ''}")
        if not args.allow_missing_shards:
            raise SystemExit(
                f"FATAL: {msg}\n"
                "  Check server/logs/plasmidhunter-serial-<task>.out for those "
                "shards and resubmit (17_plasmidhunter.sh skips shards that "
                "already carry a .done sentinel), or pass --allow-missing-shards "
                "to score an incomplete run.")
        log(f"WARNING: {msg}")
    if empty:
        log(f"note: {len(empty)} shard(s) returned no scored contig")

    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["contig_id", *parse.PLASMIDHUNTER_ALIASES, "shard"])
    calls = calls.rename(columns={"contig_id": "fragment_id"})
    log(f"PlasmidHunter scored {len(calls):,} contigs; columns present: "
        f"{sorted(c for c in calls.columns if c != 'shard')}")

    dup = calls["fragment_id"].duplicated()
    if dup.any():
        raise SystemExit(f"FATAL: {int(dup.sum())} fragment ids appear in more than one "
                         "PlasmidHunter result; the shards overlap, which they must not")

    # PlasmidHunter recovers the contig id by stripping Prodigal's trailing
    # "_<n>" from each protein name. Our fragment ids contain exactly one
    # underscore ("fragment_00000567"), so that reconstruction is exact - but it
    # is an assumption about someone else's string handling, so it is checked
    # rather than assumed.
    unknown = set(calls["fragment_id"]) - set(index["fragment_id"])
    if unknown:
        raise SystemExit(
            f"FATAL: {len(unknown)} contig id(s) in the PlasmidHunter output are "
            f"not fragment ids of the {split} split, e.g. {sorted(unknown)[:3]}. "
            "PlasmidHunter reconstructs the contig id from Prodigal's protein "
            "names by dropping the last '_<n>'; if fragment ids ever gain or lose "
            "an underscore that reconstruction breaks and the join key is unusable.")

    merged = index.merge(calls.drop(columns=["shard"]), on="fragment_id", how="left")
    merged["reported"] = merged["pred"].notna()
    if len(merged) != len(index):
        raise SystemExit("FATAL: the join changed the row count")

    n_rep = int(merged["reported"].sum())
    n_miss = len(merged) - n_rep
    log(f"reported {n_rep:,} | unclassified {n_miss:,} | total {len(merged):,}")
    if n_rep + n_miss != len(index):
        raise SystemExit("FATAL: classified + unclassified does not equal the fragment count")
    if n_miss:
        dropped = merged.loc[~merged["reported"], "length"]
        log(f"  unscored fragment lengths: min {dropped.min()} median "
            f"{dropped.median():.0f} max {dropped.max()} "
            f"(PlasmidHunter drops contigs below 1000 bp and any contig whose "
            f"proteins produce no DIAMOND hit)")
        short = int((dropped < 1000).sum())
        log(f"  of those, {short} are below PlasmidHunter's own 1000 bp cutoff; "
            f"the remaining {len(dropped) - short} produced no gene hit")

    pred = np.full(len(merged), -1, dtype=np.int8)
    proba = np.full(len(merged), np.nan)
    mask = merged["reported"].to_numpy()
    if mask.any():
        sub = merged.loc[mask]
        labels = rules.plasmidhunter_labels(sub)
        pred[mask] = labels
        proba[mask] = pd.to_numeric(sub["proba_plasmid"], errors="coerce").to_numpy()
        log(f"  plasmid calls {int((labels == 1).sum()):5d} / {int(mask.sum()):5d} "
            f"scored ({labels.mean():.2%})")
    merged["pred"] = pred
    merged["proba_plasmid"] = proba

    keep = [c for c in ("fragment_id", "shard", "replicon_id", "assembly_accession",
                        "label", "length", "length_bucket", "reported", "pred",
                        "proba_chromosome", "proba_plasmid") if c in merged.columns]
    out = results / f"predictions_plasmidhunter_{split}.csv"
    merged[keep].to_csv(out, index=False)
    log(f"\nwrote {out}")

    if mask.any():
        pr = merged.loc[mask, "proba_plasmid"]
        log(f"posterior probability: min {pr.min():.3f} median {pr.median():.3f} "
            f"max {pr.max():.3f}")
        by_label = merged.loc[mask].groupby("label")["proba_plasmid"].median()
        log(f"  median probability by true label: {by_label.to_dict()}")


if __name__ == "__main__":
    main()
