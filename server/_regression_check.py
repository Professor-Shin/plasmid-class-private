"""Re-run the ORIGINAL configuration and compare with the published laptop numbers.

The move to the server rewrote the attention pooling (to accept padded batches),
the training loop (mini-batches, pos_weight, early stopping) and the embedding
cache format. Every one of those could change the numbers while still running
without error, and the result would look like a scientific finding rather than a
refactoring bug.

So: train_eval.py is run against `config.yaml` - batch_size 1, max_orfs 10, the
small stratified subsample, 2 seeds, no regimes - reading the laptop's own cached
embeddings, and the resulting MCC per variant is compared with
esm2-attention/results_mixed/esm2_variants.csv, which is what the report cites.

Exits non-zero if any variant moves by more than the seed-to-seed spread that the
original run itself recorded (or --tolerance, whichever is larger).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ESM = ROOT / "esm2-attention"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="absolute MCC tolerance floor (default 0.02)")
    ap.add_argument("--reference", default=str(ESM / "results_mixed" / "esm2_variants.csv"))
    args = ap.parse_args()

    reference = Path(args.reference)
    if not reference.exists():
        print(f"SKIPPED: no reference at {reference}. rsync "
              "esm2-attention/results_mixed/ from the laptop to enable this check.")
        return

    want = pd.read_csv(reference)
    want = want[want["checkpoint"] == "best"].set_index("variant")["mcc"]
    print(f"reference (laptop, best checkpoint):\n{want.to_string()}\n")

    # Re-run the original config into a separate results directory so the laptop's
    # artefacts are never overwritten by the check that validates them.
    out_dir = ESM / "results_regression"
    cfg_path = ESM / "config_regression.yaml"
    original = (ESM / "config.yaml").read_text(encoding="utf-8")
    cfg_path.write_text(
        original.replace('results_dir: "results_mixed"', 'results_dir: "results_regression"'),
        encoding="utf-8")
    print(f"running train_eval.py with {cfg_path.name} "
          f"(original config, results into {out_dir.name}/)")

    proc = subprocess.run(
        [sys.executable, "-u", "train_eval.py", "--config", str(cfg_path),
         "--device", "cpu"],
        cwd=ESM, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"FATAL: the regression run itself failed (exit {proc.returncode})")

    got_path = out_dir / "esm2_variants.csv"
    if not got_path.exists():
        raise SystemExit(f"FATAL: the regression run wrote no {got_path}")
    got = pd.read_csv(got_path)
    got = got[got["checkpoint"] == "best"].set_index("variant")["mcc"]

    # The original run's own seed-to-seed spread is the honest tolerance: a
    # difference smaller than the noise the published numbers already carry cannot
    # be distinguished from it.
    payload = json.loads((ESM / "results_mixed" / "esm2_variants.json")
                         .read_text(encoding="utf-8"))
    spreads = {}
    for name, entry in payload.get("variants", {}).items():
        per_seed = entry.get("per_seed", {})
        mccs = [v["test"]["weighted"]["mcc"] for v in per_seed.values()
                if "test" in v]
        spreads[name] = (max(mccs) - min(mccs)) if len(mccs) > 1 else 0.0

    print("\nvariant                  reference      rerun       diff   tolerance  verdict")
    failures = []
    for name in want.index:
        if name not in got.index:
            failures.append(f"{name}: missing from the rerun")
            continue
        diff = abs(got[name] - want[name])
        tol = max(args.tolerance, spreads.get(name, 0.0))
        ok = diff <= tol
        print(f"{name:24s} {want[name]:9.4f} {got[name]:10.4f} {diff:10.4f} "
              f"{tol:10.4f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{name}: {want[name]:.4f} -> {got[name]:.4f} "
                            f"(diff {diff:.4f} > tolerance {tol:.4f})")

    if failures:
        print("\nFAIL: the refactor changed the published numbers:")
        for line in failures:
            print(f"  {line}")
        print("\nDo not interpret the server run as a result until this passes. The "
              "likely suspects are esm2/model.py (attention masking), train_eval.py "
              "(the batch_size==1 path must stay the prototype's) and esm2/data.py "
              "(cache ordering).")
        sys.exit(1)

    print("\nPASS: the refactored code reproduces the published numbers within the "
          "original run's own seed spread.")


if __name__ == "__main__":
    main()
