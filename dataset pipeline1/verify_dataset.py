"""Independent verification of the final dataset.

Deliberately re-derives its checks from the output files and the original
FASTA rather than reusing the pipeline's own bookkeeping, so a bug in the
pipeline cannot hide itself.

    python verify_dataset.py
"""
from __future__ import annotations

import gzip
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.utils import load_config, min_fragment_length, path_for, report_tag, read_fasta

SPLITS = ("train", "validation", "test")
failures: list[str] = []
notes: list[str] = []


def check(condition: bool, ok: str, bad: str) -> None:
    (notes if condition else failures).append(ok if condition else bad)


def main() -> int:
    cfg = load_config()
    random.seed(int(cfg["random_seed"]))
    meta = path_for(cfg, "metadata_dir")
    fcfg = cfg["fragments"]
    mixed = fcfg.get("mode") == "random_length"
    tag = report_tag(cfg)
    dataset = path_for(cfg, fcfg.get("output_dir_key", "dataset_dir"))
    frag_len = min_fragment_length(cfg)

    frames = {s: pd.read_csv(dataset / f"{s}.csv") for s in SPLITS}
    allf = pd.concat(frames.values(), ignore_index=True)
    replicons = pd.read_csv(meta / "replicons.csv")
    clusters = pd.read_csv(meta / f"clusters{tag}.csv", keep_default_na=False)

    print(f"Fragments: {len(allf):,} across {len(SPLITS)} splits")

    # 1. fragment ids unique and matching the FASTA exactly
    check(not allf["fragment_id"].duplicated().any(),
          "fragment_id values are unique across all splits",
          "DUPLICATE fragment_id values found")
    for s in SPLITS:
        ids_fa = [l[1:].strip() for l in open(dataset / f"{s}.fasta", encoding="utf-8")
                  if l.startswith(">")]
        check(len(ids_fa) == len(set(ids_fa)) and set(ids_fa) == set(frames[s]["fragment_id"]),
              f"{s}: FASTA ids match CSV exactly ({len(ids_fa):,})",
              f"{s}: FASTA/CSV id mismatch")

    # 2. no replicon, cluster or assembly-cluster crosses a split boundary
    for key in ("replicon_id", "cluster_id"):
        spread = allf.groupby(key)["split"].nunique()
        check(int((spread > 1).sum()) == 0,
              f"no {key} appears in more than one split",
              f"LEAKAGE: {int((spread > 1).sum())} {key} values span multiple splits")

    # 3. every fragment is traceable to a real replicon and assembly
    known = set(replicons["replicon_id"])
    check(set(allf["replicon_id"]).issubset(known),
          "every fragment_id traces to a replicon in replicons.csv",
          "fragments reference replicon_ids absent from replicons.csv")
    rep_by_id = replicons.set_index("replicon_id")
    mismatched_label = sum(
        1 for rid, lab in allf.groupby("replicon_id")["label"].first().items()
        if int(rep_by_id.at[rid, "label"]) != int(lab)
    )
    check(mismatched_label == 0,
          "fragment labels agree with their parent replicon labels",
          f"{mismatched_label} replicons have fragments labelled inconsistently")

    # 4. labels are binary and match replicon_type
    check(set(allf["label"].unique()) <= {0, 1},
          "labels are strictly 0/1",
          f"unexpected label values: {sorted(set(allf['label'].unique()))}")
    bad_type = allf[((allf["label"] == 0) & (allf["replicon_type"] != "chromosome"))
                    | ((allf["label"] == 1) & (allf["replicon_type"] != "plasmid"))]
    check(len(bad_type) == 0,
          "label 0 == chromosome and label 1 == plasmid throughout",
          f"{len(bad_type)} fragments whose label disagrees with replicon_type")
    check("unknown" not in set(allf["replicon_type"]),
          "no 'unknown' replicon leaked into the binary dataset",
          "'unknown' replicons are present in the dataset")

    # 5. fragment lengths are in range and coordinates are consistent
    if mixed:
        lo, hi = frag_len, int(fcfg["length_sampling"]["max"])
        check(bool((allf["length"] >= lo).all()),
              f"every fragment is at least {lo} bp",
              f"fragments below {lo} bp: {int((allf['length'] < lo).sum())}")
        # a tail absorbed into its window may exceed the sampling maximum
        check(bool((allf["length"] <= 2 * hi).all()),
              f"no fragment exceeds twice the sampling maximum ({2 * hi} bp)",
              f"fragment longer than {2 * hi} bp: {int(allf['length'].max())}")
    else:
        check(set(allf["length"].unique()) == {frag_len},
              f"every fragment is exactly {frag_len} bp",
              f"fragment lengths vary: {sorted(set(allf['length'].unique()))[:5]}")
    check(bool(((allf["end"] - allf["start"]) == allf["length"]).all()),
          "end - start == length for every fragment",
          "some fragments have inconsistent start/end coordinates")

    # 6. fragments do not overlap within a replicon (non_overlapping strategy)
    if mixed or cfg["fragments"]["strategy"] == "non_overlapping":
        sample = random.sample(sorted(set(allf["replicon_id"])), min(50, allf["replicon_id"].nunique()))
        overlaps = 0
        for rid in sample:
            starts = sorted(allf[allf["replicon_id"] == rid]["start"])
            if any(b - a < frag_len for a, b in zip(starts, starts[1:])):
                overlaps += 1
        check(overlaps == 0,
              f"no overlapping fragments in {len(sample)} sampled replicons",
              f"{overlaps} sampled replicons contain overlapping fragments")

    # 7. THE important one: do the emitted sequences actually match the source
    #    genome at the recorded coordinates?
    sample_rows = allf.sample(n=min(25, len(allf)), random_state=int(cfg["random_seed"]))
    seqs: dict[str, str] = {}
    for s in SPLITS:
        want = set(sample_rows[sample_rows["split"] == s]["fragment_id"])
        if not want:
            continue
        cur = None
        for line in open(dataset / f"{s}.fasta", encoding="utf-8"):
            if line.startswith(">"):
                cur = line[1:].strip() if line[1:].strip() in want else None
            elif cur:
                seqs[cur] = seqs.get(cur, "") + line.strip()

    verified = mismatch = 0
    for row in sample_rows.itertuples(index=False):
        src = rep_by_id.at[row.replicon_id, "source_fasta"]
        target = rep_by_id.at[row.replicon_id, "sequence_id"]
        for defline, seq in read_fasta(src):
            if defline.split()[0] != target:
                continue
            expected = seq[row.start:row.end].upper()
            if expected == seqs.get(row.fragment_id):
                verified += 1
            else:
                mismatch += 1
                failures.append(f"{row.fragment_id}: sequence does not match "
                                f"{target}[{row.start}:{row.end}]")
            break
    check(mismatch == 0,
          f"{verified} randomly sampled fragments byte-match their source genome slice",
          f"{mismatch} sampled fragments do not match their source coordinates")

    # 8. split proportions and class balance
    total = len(allf)
    print("\nsplit        fragments    share   plasmid%   replicons  assemblies  species")
    for s in SPLITS:
        f = frames[s]
        print(f"{s:<12}{len(f):>10,}{len(f)/total:>9.4f}{(f['label']==1).mean()*100:>10.3f}"
              f"{f['replicon_id'].nunique():>12,}{f['assembly_accession'].nunique():>12,}"
              f"{f['species_taxid'].nunique():>9,}")
    balances = [float((frames[s]["label"] == 1).mean()) for s in SPLITS]
    check(max(balances) - min(balances) < 0.01,
          f"plasmid fraction is consistent across splits (spread {max(balances)-min(balances):.5f})",
          f"class balance differs markedly across splits: {balances}")
    for s in SPLITS:
        target = float(cfg["split"][f"{s}_ratio"])
        actual = len(frames[s]) / total
        check(abs(actual - target) < 0.02,
              f"{s} share {actual:.4f} is within 0.02 of the target {target}",
              f"{s} share {actual:.4f} deviates from target {target}")

    # 9. nothing silently lost between replicons and fragments
    eligible = replicons[(replicons["replicon_type"].isin(["chromosome", "plasmid"]))
                         & (replicons["length"] >= frag_len)]
    check(allf["replicon_id"].nunique() == len(eligible),
          f"all {len(eligible)} eligible replicons produced fragments",
          f"{len(eligible) - allf['replicon_id'].nunique()} eligible replicons produced none")

    # ------------------------------------------------------------------ report
    print("\nPASSED")
    for n in notes:
        print(f"  [ok] {n}")
    if failures:
        print("\nFAILED")
        for f in failures:
            print(f"  [!!] {f}")
        return 1
    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
