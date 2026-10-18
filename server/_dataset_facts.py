"""Print the dataset counts the server plan is sized for, and fail if they moved.

These are the numbers every resource request in server/*.sh was derived from:
116 assemblies becomes the Platon/MOB-suite array size, and 362 Mbp is what the
2-hour per-task time limit was estimated from. If the dataset is ever rebuilt,
this check fails and the array sizes in submit_all.sh have to be updated with it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

DATASET = Path(os.environ.get("DATASET_DIR", "dataset pipeline1/data/dataset_mixed"))

EXPECTED = {
    "train": {"fragments": 79094},
    "validation": {"fragments": 16891},
    "test": {"fragments": 16944, "assemblies": 116, "replicons": 129, "plasmid": 416},
}

failures = []
for split in ("train", "validation", "test"):
    table = pd.read_csv(DATASET / f"{split}.csv",
                        usecols=["assembly_accession", "replicon_id", "label", "length"])
    facts = {
        "fragments": len(table),
        "assemblies": table["assembly_accession"].nunique(),
        "replicons": table["replicon_id"].nunique(),
        "plasmid": int((table["label"] == 1).sum()),
    }
    per_assembly = table.groupby("assembly_accession").size()
    print(f"{split:11s} {facts['fragments']:7,d} fragments | "
          f"{facts['assemblies']:4d} assemblies | {facts['replicons']:4d} replicons | "
          f"{facts['plasmid']:5d} plasmid ({facts['plasmid'] / len(table):.2%}) | "
          f"{table['length'].sum() / 1e6:7.1f} Mbp | "
          f"fragments/assembly min {per_assembly.min()} "
          f"median {per_assembly.median():.0f} max {per_assembly.max()}")
    for key, want in EXPECTED[split].items():
        if facts[key] != want:
            failures.append(f"{split}.{key}: expected {want}, found {facts[key]}")

print()
if failures:
    print("MISMATCH against the counts server/*.sh was sized for:")
    for line in failures:
        print(f"  {line}")
    print("\nUpdate N_SHARDS in server/submit_all.sh (and the plan) before submitting.")
    sys.exit(1)

print("dataset matches the counts the Slurm requests were sized for")
print(f"  Platon / MOB-suite array size = {EXPECTED['test']['assemblies']} "
      "(one task per test assembly)")
