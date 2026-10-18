"""Split one dataset split into per-assembly FASTA files, one per Slurm array task.

Why per assembly and not per fragment or one big file: Platon and MOB-recon are
built to take *the contigs of one draft assembly*. The mixed-length dataset splits
at cluster/replicon level, so every fragment of an assembly is in the same split
and the fragments tile each replicon without overlap - which means an assembly's
fragments are precisely a fragmented draft assembly of that isolate. Feeding them
together is therefore the faithful use of both tools, and it also happens to give
116 shards of roughly one genome each, which is a convenient array size.

Outputs in shard_dir:
    <assembly_accession>.fasta   the shard, FASTA with fragment ids as contig ids
    manifest.csv                 one row per shard; Slurm array index i reads line i+1
    fragment_index.csv           one row per fragment; the join key for the collectors

The FASTA is streamed, never loaded: test.fasta alone is 367 MB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None,
                    help="override config's split (test by default)")
    ap.add_argument("--force", action="store_true",
                    help="rewrite shards that already exist")
    args = ap.parse_args()

    conf = cfg.load_config(args.config)
    data, _ = cfg.use_baseline_metrics()
    split = args.split or conf["split"]
    dataset_dir, shard_dir = Path(conf["dataset_dir"]), Path(conf["shard_dir"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    table = data.load_table(dataset_dir, split)
    key = conf["shard_by"]
    log(f"{split}: {len(table):,} fragments, {table[key].nunique()} {key} values, "
        f"{int((table['label'] == 1).sum())} plasmid "
        f"({(table['label'] == 1).mean():.2%}), {table['length'].sum() / 1e6:.1f} Mbp")

    # The fragment-level index every collector joins against. Writing it here, from
    # the same table that drove the sharding, is what lets the collectors assert
    # that no fragment was lost between sharding and scoring.
    index_cols = [c for c in ("fragment_id", "replicon_id", "assembly_accession",
                              "replicon_type", "label", "length", "length_bucket",
                              "cluster_id", "gc_content") if c in table.columns]
    index = table[index_cols].copy()
    index["shard"] = table[key]
    index.to_csv(shard_dir / f"fragment_index_{split}.csv", index=False)
    log(f"wrote {shard_dir / f'fragment_index_{split}.csv'}")

    # Fragment id -> shard, so the single streaming pass can route each record.
    shard_of = dict(zip(table["fragment_id"], table[key]))
    order = list(dict.fromkeys(table[key]))          # first-appearance order, stable

    existing = {p.stem for p in shard_dir.glob("*.fasta")}
    if existing and not args.force:
        wanted = set(order)
        if wanted <= existing:
            log(f"all {len(wanted)} shards already present; use --force to rewrite")
            write = False
        else:
            log(f"{len(wanted - existing)} of {len(wanted)} shards missing; rewriting all "
                "(a partially written shard cannot be detected from its name alone)")
            write = True
    else:
        write = True

    if write:
        for stale in shard_dir.glob("*.fasta"):
            stale.unlink()
        t0 = time.time()
        handles = {}
        try:
            written, expected = 0, table["fragment_id"].to_numpy()
            for i, (name, seq) in enumerate(data.iter_fasta(dataset_dir / f"{split}.fasta")):
                # Same guard the other scripts use: the FASTA and the CSV must be in
                # the same order, or every downstream label would be shifted.
                if i >= len(expected) or name != expected[i]:
                    raise ValueError(
                        f"{split}: FASTA record {i} is {name}, CSV says "
                        f"{expected[i] if i < len(expected) else '<past end>'}")
                shard = shard_of[name]
                fh = handles.get(shard)
                if fh is None:
                    fh = handles[shard] = open(shard_dir / f"{shard}.fasta", "wb")
                fh.write(b">" + name.encode() + b"\n")
                for off in range(0, len(seq), 70):
                    fh.write(seq[off:off + 70] + b"\n")
                written += 1
                if written % 5000 == 0:
                    log(f"  {written:,}/{len(expected):,} fragments "
                        f"({time.time() - t0:.0f}s)")
            if written != len(expected):
                raise ValueError(f"{split}: wrote {written} fragments, CSV holds "
                                 f"{len(expected)}")
        finally:
            for fh in handles.values():
                fh.close()
        log(f"wrote {len(handles)} shards in {time.time() - t0:.0f}s")

    # The manifest IS the array mapping: task $SLURM_ARRAY_TASK_ID reads line i+1.
    rows = []
    for shard in order:
        sel = table[key] == shard
        path = shard_dir / f"{shard}.fasta"
        rows.append({
            "shard": shard,
            "n_fragments": int(sel.sum()),
            "n_plasmid": int((table.loc[sel, "label"] == 1).sum()),
            "bp": int(table.loc[sel, "length"].sum()),
            "fasta": path.name,
            "fasta_bytes": path.stat().st_size if path.exists() else 0,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(shard_dir / "manifest.csv", index=False)

    if manifest["n_fragments"].sum() != len(table):
        raise SystemExit("manifest does not account for every fragment")
    log(f"\nwrote {shard_dir / 'manifest.csv'}: {len(manifest)} shards, "
        f"{manifest['n_fragments'].sum():,} fragments, {manifest['bp'].sum() / 1e6:.1f} Mbp")
    log(f"  fragments/shard  min {manifest['n_fragments'].min()} "
        f"median {manifest['n_fragments'].median():.0f} "
        f"max {manifest['n_fragments'].max()}")
    log(f"  Mbp/shard        min {manifest['bp'].min() / 1e6:.2f} "
        f"median {manifest['bp'].median() / 1e6:.2f} "
        f"max {manifest['bp'].max() / 1e6:.2f}")
    log(f"\nsubmit the arrays as 1-{len(manifest)}")


if __name__ == "__main__":
    main()
