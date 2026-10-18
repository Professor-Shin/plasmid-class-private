"""Prove that one `--characterize` run can stand in for a real `--mode accuracy` run.

The whole Platon plan rests on a saving: `--characterize` reports RDS and every
hallmark column for every contig, so the three published filter modes can be
applied offline (toolbase/rules.py) instead of running Platon three times. That
saving is only legitimate if the re-derived labels are identical to the ones
Platon itself produces.

This compares them directly. A `--mode accuracy` run partitions the input into
<prefix>.plasmid.fasta and <prefix>.chromosome.fasta, which is Platon's own
answer; the accuracy rule is applied to the characterize TSV of the same input
and the two label vectors must match exactly.

Exits non-zero on any disagreement, and prints the offending contigs with the
fields the rule consumed so the discrepancy can be read off directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import parse, rules  # noqa: E402


def fasta_ids(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    ids = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.add(line[1:].split()[0].strip())
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, help="TSV from the --characterize run")
    ap.add_argument("--plasmid-fasta", required=True,
                    help="<prefix>.plasmid.fasta from the --mode accuracy run")
    ap.add_argument("--chromosome-fasta", required=True,
                    help="<prefix>.chromosome.fasta from the --mode accuracy run")
    ap.add_argument("--mode", default="accuracy", choices=rules.MODES)
    args = ap.parse_args()

    df = parse.read_platon_tsv(Path(args.tsv))
    if df.empty:
        raise SystemExit(f"FATAL: {args.tsv} is empty - the characterize run produced "
                         "nothing to compare against")
    print(f"characterize TSV: {len(df)} contigs, columns "
          f"{sorted(c for c in df.columns)}")

    plasmid = fasta_ids(Path(args.plasmid_fasta))
    chromosome = fasta_ids(Path(args.chromosome_fasta))
    print(f"platon --mode {args.mode}: {len(plasmid)} plasmid, "
          f"{len(chromosome)} chromosome contigs")
    if not plasmid and not chromosome:
        raise SystemExit("FATAL: both output FASTAs of the filtered run are empty or "
                         "missing; nothing to verify against")

    derived = rules.platon_labels(df, args.mode)
    ids = df["contig_id"].astype(str).to_numpy()

    truth = np.full(len(ids), -1, dtype=np.int8)
    for i, cid in enumerate(ids):
        if cid in plasmid:
            truth[i] = 1
        elif cid in chromosome:
            truth[i] = 0

    unseen = int((truth < 0).sum())
    if unseen:
        print(f"WARNING: {unseen} contig(s) of the characterize run appear in neither "
              f"output FASTA of the filtered run; they are excluded from the comparison")
    compare = truth >= 0
    if not compare.any():
        raise SystemExit("FATAL: no contigs in common between the two runs")

    agree = derived[compare] == truth[compare]
    n = int(compare.sum())
    print(f"\ncompared {n} contigs: {int(agree.sum())} agree, "
          f"{int((~agree).sum())} disagree")

    if agree.all():
        print(f"\nPASS: the re-derived '{args.mode}' rule reproduces Platon's own "
              "classification exactly.")
        print("One --characterize run per shard is therefore sufficient for all three "
              "filter modes.")
        return

    bad = np.flatnonzero(compare)[~agree]
    cols = [c for c in ("contig_id", "rds", "circularity", "inc_types", "n_replication",
                        "n_mobilization", "n_orit", "n_rrna", "n_plasmid_hits")
            if c in df.columns]
    print("\nFAIL: the following contigs were labelled differently\n")
    detail = df.iloc[bad][cols].copy()
    detail["platon_said"] = truth[bad]
    detail["rule_said"] = derived[bad]
    print(detail.to_string(index=False))
    print("\nDo NOT trust the offline rule. Re-run the array with "
          "`--mode accuracy` instead of `--characterize` (the fallback in section "
          "2.2 of the plan) and score Platon's own output FASTAs directly.")
    sys.exit(1)


if __name__ == "__main__":
    main()
