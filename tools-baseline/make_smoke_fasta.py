"""Build a tiny mixed-class FASTA to prove both tools are installed and working.

Picks a handful of plasmid and chromosome fragments from the split, preferring
short ones so the smoke test finishes in a couple of minutes. Both classes are
present on purpose: a tool that silently calls everything chromosome would still
"succeed" on a plasmid-only input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conf = cfg.load_config(args.config)
    data, _ = cfg.use_baseline_metrics()
    split = args.split or conf["split"]
    table = data.load_table(Path(conf["dataset_dir"]), split)

    picked = []
    for label in (1, 0):
        sub = table[table["label"] == label].nsmallest(args.n_per_class, "length")
        picked.append(sub)
        print(f"  label {label}: {len(sub)} fragments, "
              f"lengths {sub['length'].min()}-{sub['length'].max()} bp")
    wanted = {row.fragment_id: row.label
              for part in picked for row in part.itertuples()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "wb") as fh:
        for name, seq in data.iter_fasta(Path(conf["dataset_dir"]) / f"{split}.fasta"):
            if name not in wanted:
                continue
            fh.write(b">" + name.encode() + b"\n")
            for off in range(0, len(seq), 70):
                fh.write(seq[off:off + 70] + b"\n")
            written += 1
            if written == len(wanted):
                break
    if written != len(wanted):
        raise SystemExit(f"only found {written} of {len(wanted)} fragments in the FASTA")

    truth = out.with_suffix(".labels.csv")
    with open(truth, "w", encoding="utf-8") as fh:
        fh.write("fragment_id,label\n")
        for frag_id, label in wanted.items():
            fh.write(f"{frag_id},{label}\n")
    print(f"wrote {out} ({written} fragments) and {truth}")


if __name__ == "__main__":
    main()
