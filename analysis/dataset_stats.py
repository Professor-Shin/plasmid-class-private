"""Dataset statistics for the report's dataset section.

`dataset pipeline1/reports/` already holds what the build pipeline wrote while
it was building: per-split counts and the fragment-length distribution. This
adds what a reader of the results table needs in order to interpret them, which
the build never had to compute:

  * class balance at all three levels - fragment, replicon, assembly - because
    the headline MCC is a fragment number and the fair comparison is a replicon
    number, and the two have very different prevalences.
  * the length distribution per split AND per class, since every tool here
    degrades with length and the question "do plasmid fragments differ in
    length from chromosome fragments?" decides whether length is a confound.
  * GC by class and by length bucket - the composition baselines exist to ask
    how much of the signal is GC, and this is the input to that question.
  * how fragments stack into replicons and replicons into assemblies, which is
    what the replicon-level aggregation averages over.
  * taxonomic composition, and whether any cluster, replicon or assembly appears
    in more than one split. The splitting was done at cluster level, so clusters
    and replicons must be disjoint and that is asserted; assemblies are NOT, and
    how far they are not is one of the more consequential facts about this
    dataset - see shard_composition below.
  * what a per-assembly shard actually contains. Because the split is at
    replicon/cluster level, one isolate's chromosome can sit in train while one
    of its plasmids sits in test, so a test-split shard is a SUBSET of a draft
    assembly and some shards are plasmid-only. Platon and mob_recon are built
    around the contigs of one complete isolate, so this is a real limitation on
    how their numbers can be read, and it is measured rather than assumed away.

Everything is computed from the split CSVs alone - no FASTA, no model output -
so this runs anywhere the dataset is checked out, and it is deterministic.

    python analysis/dataset_stats.py
    python analysis/dataset_stats.py --dataset "dataset pipeline1/data/dataset_mixed"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset pipeline1" / "data" / "dataset_mixed"
OUT = ROOT / "analysis" / "results"
SPLITS = ("train", "validation", "test")
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
CLASS_NAME = {0: "chromosome", 1: "plasmid"}


def load(dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = dataset_dir / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        df = pd.read_csv(path)
        df["split"] = split            # trust the file name, not the column
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["class"] = df["label"].map(CLASS_NAME)
    return df


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence at fragment, replicon and assembly level, per split."""
    rows = []
    for split in list(SPLITS) + ["all"]:
        d = df if split == "all" else df[df["split"] == split]
        rep = d.groupby("replicon_id")["label"].first()
        # An assembly is "plasmid-bearing" if any of its replicons is a plasmid;
        # an assembly is never itself one class or the other.
        asm = d.groupby("assembly_accession")["label"].max()
        bp = d.groupby("label")["length"].sum()
        rows.append({
            "split": split,
            "fragments": len(d),
            "fragments_plasmid": int((d["label"] == 1).sum()),
            "fragment_plasmid_rate": round(float((d["label"] == 1).mean()), 5),
            "imbalance_1_to_n": round(float((d["label"] == 0).sum()
                                            / max((d["label"] == 1).sum(), 1)), 1),
            "replicons": int(rep.size),
            "replicons_plasmid": int((rep == 1).sum()),
            "replicon_plasmid_rate": round(float((rep == 1).mean()), 5),
            "assemblies": int(asm.size),
            "assemblies_with_plasmid": int((asm == 1).sum()),
            "species": int(d["species_taxid"].nunique()),
            "clusters": int(d["cluster_id"].nunique()),
            "total_bp": int(d["length"].sum()),
            "plasmid_bp": int(bp.get(1, 0)),
            "plasmid_bp_rate": round(float(bp.get(1, 0) / max(d["length"].sum(), 1)), 5),
        })
    return pd.DataFrame(rows)


def length_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Length quantiles per split and class - the confound check."""
    rows = []
    for split in list(SPLITS) + ["all"]:
        d = df if split == "all" else df[df["split"] == split]
        for cls in ("chromosome", "plasmid", "both"):
            sub = d if cls == "both" else d[d["class"] == cls]
            if sub.empty:
                continue
            q = sub["length"].quantile(QUANTILES)
            rows.append({
                "split": split, "class": cls, "n": len(sub),
                "min": int(sub["length"].min()), "max": int(sub["length"].max()),
                "mean": round(float(sub["length"].mean()), 1),
                **{f"p{int(p * 100)}": int(q.loc[p]) for p in QUANTILES},
                "total_Mbp": round(float(sub["length"].sum()) / 1e6, 1),
            })
    return pd.DataFrame(rows)


def length_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Per length bucket: how many fragments, how many plasmid, in each split.

    This is the table every by-length result in the report is read against: an
    MCC in the 1-2kb bucket is computed on whatever number of plasmid fragments
    this says the bucket holds, and some of those counts are small.
    """
    order = ["1-2kb", "2-5kb", "5-10kb", "10-25kb", "25-50kb", "50-100kb"]
    if "length_bucket" not in df.columns:
        return pd.DataFrame()
    present = [b for b in order if (df["length_bucket"] == b).any()]
    present += [b for b in df["length_bucket"].dropna().unique() if b not in order]
    rows = []
    for bucket in present:
        d = df[df["length_bucket"] == bucket]
        row = {"length_bucket": bucket, "fragments": len(d),
               "fragment_share": round(len(d) / len(df), 4),
               "fragments_plasmid": int((d["label"] == 1).sum()),
               "plasmid_rate": round(float((d["label"] == 1).mean()), 5),
               "plasmid_replicons": int(d.loc[d["label"] == 1, "replicon_id"].nunique()),
               "mean_length": int(d["length"].mean()),
               "mean_gc": round(float(d["gc_content"].mean()), 4)}
        for split in SPLITS:
            s = d[d["split"] == split]
            row[f"n_{split}"] = len(s)
            row[f"plasmid_{split}"] = int((s["label"] == 1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def gc_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """GC per class, overall and per length bucket.

    The gap between the two class means is the ceiling on what a GC-only model
    can do, and the per-bucket rows show whether that gap survives at the short
    lengths where a single fragment's GC is noisy.
    """
    rows = []
    for bucket in ["all"] + sorted(df["length_bucket"].dropna().unique().tolist()
                                   if "length_bucket" in df.columns else []):
        d = df if bucket == "all" else df[df["length_bucket"] == bucket]
        stats = d.groupby("class")["gc_content"].agg(["count", "mean", "std", "median"])
        row = {"length_bucket": bucket}
        for cls in ("chromosome", "plasmid"):
            if cls in stats.index:
                row[f"{cls}_n"] = int(stats.loc[cls, "count"])
                row[f"{cls}_gc_mean"] = round(float(stats.loc[cls, "mean"]), 4)
                row[f"{cls}_gc_sd"] = round(float(stats.loc[cls, "std"]), 4)
        if {"chromosome", "plasmid"} <= set(stats.index):
            row["gc_gap"] = round(float(stats.loc["plasmid", "mean"]
                                        - stats.loc["chromosome", "mean"]), 4)
            # In units of the within-class spread: a gap of 0.02 means something
            # different when the sd is 0.01 than when it is 0.08.
            pooled = float(np.sqrt((stats.loc["plasmid", "std"] ** 2
                                    + stats.loc["chromosome", "std"] ** 2) / 2))
            row["gc_gap_sd_units"] = round(row["gc_gap"] / pooled, 3) if pooled else None
        rows.append(row)
    return pd.DataFrame(rows)


def replicon_structure(df: pd.DataFrame) -> pd.DataFrame:
    """How many fragments make up a replicon, and how many replicons an assembly."""
    rows = []
    for split in list(SPLITS) + ["all"]:
        d = df if split == "all" else df[df["split"] == split]
        per_rep = d.groupby("replicon_id").agg(n=("fragment_id", "size"),
                                               label=("label", "first"),
                                               bp=("length", "sum"))
        per_asm = d.groupby("assembly_accession")["replicon_id"].nunique()
        for cls in ("chromosome", "plasmid"):
            sel = per_rep[per_rep["label"] == (1 if cls == "plasmid" else 0)]
            if sel.empty:
                continue
            rows.append({
                "split": split, "class": cls, "replicons": len(sel),
                "frags_per_replicon_min": int(sel["n"].min()),
                "frags_per_replicon_median": float(sel["n"].median()),
                "frags_per_replicon_max": int(sel["n"].max()),
                "replicon_kbp_median": round(float(sel["bp"].median()) / 1e3, 1),
                "replicon_kbp_max": round(float(sel["bp"].max()) / 1e3, 1),
                # Why the two classes' fragment-length distributions differ at
                # all: the sampler draws a length the replicon can actually
                # supply (sampled_length == length everywhere, see quality()),
                # so a replicon shorter than the upper end of the range simply
                # cannot contribute a long fragment. Most plasmids are.
                "replicons_under_50kb": int((sel["bp"] < 50_000).sum()),
                "replicons_under_50kb_share": round(
                    float((sel["bp"] < 50_000).mean()), 3),
                "replicons_per_assembly_median": float(per_asm.median()),
                "replicons_per_assembly_max": int(per_asm.max()),
            })
    return pd.DataFrame(rows)


def taxonomy(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """The most represented organisms, and how they fall across the splits.

    A test split dominated by one genus would make every number in the report a
    statement about that genus, so the shape of this table is part of reading
    the results.
    """
    df = df.copy()
    df["genus"] = df["organism_name"].astype(str).str.split().str[0]
    rows = []
    for genus, d in df.groupby("genus"):
        row = {"genus": genus, "fragments": len(d),
               "fragment_share": round(len(d) / len(df), 4),
               "assemblies": int(d["assembly_accession"].nunique()),
               "species": int(d["species_taxid"].nunique()),
               "plasmid_fragments": int((d["label"] == 1).sum())}
        for split in SPLITS:
            row[f"n_{split}"] = int((d["split"] == split).sum())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("fragments", ascending=False)
    return out.head(top).reset_index(drop=True)


def split_overlap(df: pd.DataFrame) -> dict:
    """Leakage check: is any cluster, replicon or assembly in two splits?

    The pipeline splits at CLUSTER level, where a cluster is a group of
    replicons within Mash-ANI 0.95 of each other. So:

      cluster_id, replicon_id   must be disjoint across splits. A shared one
                                would mean the same sequence, or a near-identical
                                one, was both trained on and tested on, and every
                                number in the report would be inflated. Asserted.
      assembly_accession        may be shared, and is: an isolate's chromosome
                                and its plasmids are different replicons in
                                different clusters, so they can land in different
                                splits. Reported, not asserted - but it is why
                                shard_composition exists.
      species_taxid             may be shared. Splitting on species would have
                                left too few of them. Reported.
    """
    out = {}
    for key in ("cluster_id", "replicon_id", "assembly_accession", "species_taxid"):
        by_split = {s: set(df.loc[df["split"] == s, key]) for s in SPLITS}
        shared = {}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                shared[f"{a}&{b}"] = len(by_split[a] & by_split[b])
        out[key] = {"per_split": {s: len(v) for s, v in by_split.items()},
                    "shared": shared,
                    "total_distinct": int(df[key].nunique())}
    out["leakage_ok"] = all(v == 0 for k in ("cluster_id", "replicon_id")
                            for v in out[k]["shared"].values())
    return out


def shard_composition(df: pd.DataFrame) -> pd.DataFrame:
    """What one per-assembly shard actually holds, per split.

    tools-baseline/shard_by_assembly.py groups a split's fragments by assembly
    and hands each group to Platon and mob_recon as "the contigs of one draft
    assembly". That framing is only partly true: the split is at replicon level,
    so a shard holds the replicons of that assembly THAT LANDED IN THIS SPLIT,
    which can be a chromosome without its plasmids, plasmids without their
    chromosome, or the whole isolate.

    Both tools reason over an isolate's full contig set - mob_recon reconstructs
    plasmids from it, and Platon's RDS is a per-contig statistic but its
    accuracy mode leans on hallmarks that a lone plasmid contig may well carry
    while a lone chromosome fragment does not. A plasmid-only shard is therefore
    an easier input than a real draft assembly, and a chromosome-only shard a
    harder one. The counts below are how much of each the test split is.
    """
    # Replicons per assembly across the WHOLE dataset, so "how much of this
    # isolate is here" is measurable from inside one split.
    total_reps = df.groupby("assembly_accession")["replicon_id"].nunique()
    rows = []
    for split in list(SPLITS) + ["all"]:
        d = df if split == "all" else df[df["split"] == split]
        per_asm = d.groupby("assembly_accession").agg(
            replicons=("replicon_id", "nunique"),
            has_chromosome=("label", lambda x: bool((x == 0).any())),
            has_plasmid=("label", lambda x: bool((x == 1).any())),
            fragments=("fragment_id", "size"),
            bp=("length", "sum"))
        if per_asm.empty:
            continue
        frac = per_asm["replicons"] / total_reps.reindex(per_asm.index)
        both = per_asm["has_chromosome"] & per_asm["has_plasmid"]
        rows.append({
            "split": split,
            "shards": int(len(per_asm)),
            "chromosome_only": int((per_asm["has_chromosome"] & ~per_asm["has_plasmid"]).sum()),
            "plasmid_only": int((~per_asm["has_chromosome"] & per_asm["has_plasmid"]).sum()),
            "chromosome_and_plasmid": int(both.sum()),
            "complete_isolates": int((frac >= 1.0).sum()),
            "replicon_coverage_median": round(float(frac.median()), 3),
            "replicon_coverage_min": round(float(frac.min()), 3),
            "fragments_per_shard_median": float(per_asm["fragments"].median()),
            "Mbp_per_shard_median": round(float(per_asm["bp"].median()) / 1e6, 2),
        })
    return pd.DataFrame(rows)


def quality(df: pd.DataFrame) -> pd.DataFrame:
    """Non-ACGT content and whether the sampled length was achievable.

    `sampled_length` is what the sampler asked for and `length` is what the
    replicon supplied. They are equal everywhere in this dataset, which says the
    sampler drew within each replicon's capacity rather than truncating after
    the fact - so a non-zero count here would mean the build changed.

    That makes replicon length, not truncation, the reason the two classes'
    fragment-length distributions differ: see `replicons_under_50kb` in the
    replicon-structure table.
    """
    rows = []
    for split in list(SPLITS) + ["all"]:
        d = df if split == "all" else df[df["split"] == split]
        truncated = (d["sampled_length"] != d["length"]) if "sampled_length" in d else pd.Series(dtype=bool)
        rows.append({
            "split": split, "fragments": len(d),
            "with_non_acgt": int((d["non_acgt_fraction"] > 0).sum()),
            "non_acgt_max": round(float(d["non_acgt_fraction"].max()), 6),
            "non_acgt_mean": round(float(d["non_acgt_fraction"].mean()), 8),
            "length_below_request": int(truncated.sum()) if len(truncated) else None,
            "length_below_request_plasmid": (
                int((truncated & (d["label"] == 1)).sum()) if len(truncated) else None),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--top-genera", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.dataset))
    print(f"{len(df):,} fragments from {Path(args.dataset)}")

    tables = {
        "dataset_class_balance": class_balance(df),
        "dataset_length_quantiles": length_distribution(df),
        "dataset_length_buckets": length_buckets(df),
        "dataset_gc_by_class": gc_by_class(df),
        "dataset_replicon_structure": replicon_structure(df),
        "dataset_taxonomy": taxonomy(df, args.top_genera),
        "dataset_quality": quality(df),
        "dataset_shard_composition": shard_composition(df),
    }
    payload = {"dataset_dir": str(Path(args.dataset)), "fragments": int(len(df))}
    for name, table in tables.items():
        if table.empty:
            continue
        table.to_csv(out / f"{name}.csv", index=False)
        payload[name] = table.to_dict("records")
        print(f"  wrote {name}.csv ({len(table)} rows)")

    payload["split_overlap"] = split_overlap(df)
    (out / "dataset_stats.json").write_text(json.dumps(payload, indent=2, default=str),
                                            encoding="utf-8")
    print(f"  wrote dataset_stats.json")

    ov = payload["split_overlap"]
    if not ov["leakage_ok"]:
        print("\nSPLIT INTEGRITY FAILED - a cluster or replicon is in two splits:")
        for key in ("cluster_id", "replicon_id"):
            for pair, n in ov[key]["shared"].items():
                if n:
                    print(f"  {key}: {n} shared between {pair}")
        print("  Every number in the report is invalid until this is fixed.")
        return 1
    print("\nsplit integrity: no cluster and no replicon appears in two splits")

    asm_shared = ov["assembly_accession"]["shared"]
    if any(asm_shared.values()):
        print(f"assemblies spanning two splits: {asm_shared} "
              f"(of {ov['assembly_accession']['total_distinct']})")
        print("  Expected: the split is at replicon/cluster level, so an isolate's")
        print("  chromosome and its plasmids can land in different splits. It does")
        print("  mean a per-assembly shard is a SUBSET of a draft assembly - see the")
        print("  dataset_shard_composition table for how many shards hold only one")
        print("  class, which is the form Platon and mob_recon were NOT designed for.")
    print(f"species shared between splits (expected, not an error): "
          f"{ov['species_taxid']['shared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
