"""Prove that one PLASMe run can stand in for all three of its preset modes.

The PLASMe plan rests on the same saving the Platon plan does. PLASMe's
`-m/--mode` presets are three threshold triples over the BLASTn identity, the
BLASTn coverage and the Transformer probability, and all three of those numbers
are written for every aligned contig into <temp>/PLASMe_candidate.csv. So the
three modes can be applied offline (toolbase/rules.PLASME_MODES) from one run
instead of running PLASMe three times against a 12 GB database.

That saving is only legitimate if the re-derived labels match what PLASMe itself
writes. PLASMe's `<output>_report.csv` lists exactly the contigs it called
plasmid, at whatever thresholds the run used, so it is the ground truth for one
mode per run. A run at the CLI defaults (-i 0.9 -c 0.9 -p 0.5) is the `balance`
preset, which is what 16_plasme.sh runs, so that is the mode checked by default.

Exits non-zero on any disagreement, printing the offending contigs with the
three numbers the rule consumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import parse, rules  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True,
                    help="<temp>/PLASMe_candidate.csv from the run")
    ap.add_argument("--report", required=True,
                    help="<output>_report.csv from the same run")
    ap.add_argument("--mode", default="balance", choices=list(rules.PLASME_MODES),
                    help="the preset the run used; 'balance' == the CLI defaults")
    args = ap.parse_args()

    df = parse.read_plasme_candidate(Path(args.candidate))
    if df.empty:
        raise SystemExit(f"FATAL: {args.candidate} is empty - the run aligned nothing, "
                         "so there is nothing to verify against")
    thresholds = rules.PLASME_MODES[args.mode]
    print(f"candidate table: {len(df)} aligned contigs")
    print(f"mode '{args.mode}': identity >= {thresholds[0]}, coverage >= "
          f"{thresholds[1]}, probability > {thresholds[2]}")
    scored = int((df['transformer_score'] > rules.PLASME_NO_TRANSFORMER).sum())
    print(f"  of those, {scored} carry a Transformer probability and "
          f"{len(df) - scored} were decided by BLASTn alone")

    report = parse.read_plasme_report(Path(args.report))
    called = set(report["contig_id"].astype(str)) if not report.empty else set()
    print(f"PLASMe's own report: {len(called)} contigs called plasmid")

    derived = rules.plasme_labels(df, *thresholds)
    ids = df["contig_id"].astype(str).to_numpy()
    truth = np.array([1 if cid in called else 0 for cid in ids], dtype=np.int8)

    # Every contig PLASMe reported must be one it aligned; if the report names a
    # contig the candidate table does not, the two files are not from one run.
    stray = called - set(ids)
    if stray:
        raise SystemExit(
            f"FATAL: {len(stray)} contig(s) in {args.report} are absent from "
            f"{args.candidate}, e.g. {sorted(stray)[:3]}. The two files are not "
            "from the same run, so the comparison is meaningless.")

    agree = derived == truth
    print(f"\ncompared {len(ids)} contigs: {int(agree.sum())} agree, "
          f"{int((~agree).sum())} disagree")

    if agree.all():
        print(f"\nPASS: the re-derived '{args.mode}' rule reproduces PLASMe's own "
              "classification exactly.")
        print("One PLASMe run per shard is therefore sufficient for all three preset "
              "modes, and its Transformer probability can be used as a continuous "
              "score on top.")
        return

    bad = np.flatnonzero(~agree)
    detail = df.iloc[bad][["contig_id", "identity", "coverage", "transformer_score"]].copy()
    detail["plasme_said"] = truth[bad]
    detail["rule_said"] = derived[bad]
    print("\nFAIL: the following contigs were labelled differently\n")
    print(detail.to_string(index=False))
    print("\nDo NOT trust the offline rule. Either re-read the decision in "
          "PLASMe.py:plasme_output() against toolbase/rules.plasme_labels, or run "
          "PLASMe once per mode with -m and score each run's own report.")
    sys.exit(1)


if __name__ == "__main__":
    main()
