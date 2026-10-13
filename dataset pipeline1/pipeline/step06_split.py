"""Step 6 - train / validation / test split at the CLUSTER level.

Splitting happens before any fragment exists. The unit is the similarity
cluster from step 5 (a cluster of one replicon is just that replicon), so:

  * every fragment of a replicon lands in exactly one split;
  * identical and near-identical parent replicons cannot straddle splits.

Quota filling is class-aware. Clusters are bucketed by the replicon types they
contain (chromosome-only / plasmid-only / mixed) and each bucket is filled
towards the 70/15/15 target independently, weighted by the number of fragments
the cluster will eventually produce. That keeps the class balance of the
splits close to the balance of the corpus without touching the corpus itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (StageLog, get_logger, load_config, min_fragment_length, path_for,
                    report_tag, update_manifest)

SPLITS = ("train", "validation", "test")


def cluster_weight(group: pd.DataFrame, cfg: dict) -> int:
    basis = cfg["split"]["balance_by"]
    if basis == "replicon_count":
        return len(group)
    if basis == "bases":
        return int(group["length"].sum())
    return int((group["length"] // min_fragment_length(cfg)).sum())


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step06.split")
    slog = StageLog("step06_split")

    meta_dir = path_for(cfg, "metadata_dir")
    tag = report_tag(cfg)
    clusters = pd.read_csv(meta_dir / f"clusters{tag}.csv", keep_default_na=False)

    n_all = len(clusters)
    usable = clusters[
        (clusters["cluster_id"].astype(str).str.len() > 0)
        & clusters["replicon_type"].isin(["chromosome", "plasmid"])
    ].copy()
    slog.record("replicons available for splitting", n_all, len(usable),
                "unknown / sub-fragment-length replicons excluded")

    ratios = {
        "train": float(cfg["split"]["train_ratio"]),
        "validation": float(cfg["split"]["validation_ratio"]),
        "test": float(cfg["split"]["test_ratio"]),
    }
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total_ratio}")

    # --------------------------------------- optional: keep assemblies intact
    # The instruction's split unit is the parent replicon / cluster, so by
    # default an assembly's chromosome and its plasmids may land in different
    # splits - they are different molecules carrying different labels, so this
    # is not fragment-level leakage. Some designs prefer the stricter rule that
    # no strain contributes sequence to more than one split; that is this flag.
    usable["split_group_id"] = usable["cluster_id"]
    if bool(cfg["split"].get("keep_assembly_together", False)):
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for cid in usable["cluster_id"].unique():
            parent.setdefault(cid, cid)
        for _, grp in usable.groupby("assembly_accession", sort=True):
            cids = sorted(set(grp["cluster_id"]))
            for other in cids[1:]:
                union(cids[0], other)
        usable["split_group_id"] = [find(c) for c in usable["cluster_id"]]
        logger.info("keep_assembly_together: %d clusters merged into %d split groups",
                    usable["cluster_id"].nunique(), usable["split_group_id"].nunique())

    # ------------------------------------------------- describe every cluster
    rows = []
    for cid, group in usable.groupby("split_group_id", sort=True):
        types = set(group["replicon_type"])
        bucket = "mixed" if len(types) > 1 else next(iter(types))
        rows.append({
            "split_group_id": cid,
            "cluster_ids": ";".join(sorted(set(group["cluster_id"]))),
            "bucket": bucket,
            "n_replicons": len(group),
            "n_chromosomes": int((group["replicon_type"] == "chromosome").sum()),
            "n_plasmids": int((group["replicon_type"] == "plasmid").sum()),
            "n_assemblies": int(group["assembly_accession"].nunique()),
            "total_bp": int(group["length"].sum()),
            "weight": cluster_weight(group, cfg),
        })
    cdf = pd.DataFrame(rows)
    logger.info("Splitting %d clusters (%d replicons) by %s",
                len(cdf), len(usable), cfg["split"]["balance_by"])

    # ------------------------------------------------------ greedy quota fill
    rng = np.random.default_rng(int(cfg["random_seed"]))
    assignment: dict[str, str] = {}

    for bucket, bdf in cdf.groupby("bucket", sort=True):
        bdf = bdf.sort_values("split_group_id", kind="mergesort")
        order = rng.permutation(len(bdf))
        bdf = bdf.iloc[order]
        # Largest clusters first: placing the heavy ones while all quotas are
        # still open keeps the final proportions close to the target.
        bdf = bdf.sort_values("weight", ascending=False, kind="mergesort")

        total_w = int(bdf["weight"].sum())
        quota = {s: ratios[s] * total_w for s in SPLITS}
        filled = {s: 0 for s in SPLITS}
        for row in bdf.itertuples(index=False):
            # Assign to whichever split is furthest below its quota.
            deficit = {s: (quota[s] - filled[s]) / max(quota[s], 1e-9) for s in SPLITS}
            target = max(SPLITS, key=lambda s: (deficit[s], -SPLITS.index(s)))
            assignment[row.split_group_id] = target
            filled[target] += int(row.weight)
        logger.info("  bucket %-11s clusters=%4d weight=%8d -> %s",
                    bucket, len(bdf), total_w,
                    {s: filled[s] for s in SPLITS})

    usable["split"] = usable["split_group_id"].map(assignment)
    cdf["split"] = cdf["split_group_id"].map(assignment)

    # -------------------------------------------------------------- outputs
    split_dir = path_for(cfg, "split_dir")
    cols = ["replicon_id", "assembly_accession", "sequence_id", "organism_name",
            "taxid", "species_taxid", "replicon_type", "label", "length",
            "cluster_id", "split_group_id", "split", "sequence_sha256", "source_fasta"]
    for s in SPLITS:
        sub = usable[usable["split"] == s][cols].sort_values("replicon_id")
        sub.to_csv(split_dir / f"{s}_replicons{tag}.csv", index=False)
        logger.info("Wrote %s (%d replicons)", split_dir / f"{s}_replicons{tag}.csv", len(sub))

    all_path = meta_dir / f"replicons_split{tag}.csv"
    usable[cols].sort_values(["split", "replicon_id"]).to_csv(all_path, index=False)
    cdf.sort_values("split_group_id").to_csv(meta_dir / f"cluster_assignments{tag}.csv", index=False)

    # ----------------------------------------------------------- leakage check
    leaks = usable.groupby("cluster_id")["split"].nunique()
    n_leaks = int((leaks > 1).sum())
    if n_leaks:
        raise RuntimeError(f"{n_leaks} clusters were assigned to more than one split")
    if int(usable.groupby("replicon_id")["split"].nunique().max()) > 1:
        raise RuntimeError("a replicon was assigned to more than one split")

    frag_len = min_fragment_length(cfg)
    usable["expected_fragments"] = usable["length"] // frag_len
    summary_rows = []
    for s in SPLITS:
        sub = usable[usable["split"] == s]
        chrom = sub[sub["replicon_type"] == "chromosome"]
        plas = sub[sub["replicon_type"] == "plasmid"]
        summary_rows.append({
            "split": s,
            "target_ratio": ratios[s],
            "clusters": int(sub["cluster_id"].nunique()),
            "split_groups": int(sub["split_group_id"].nunique()),
            "assemblies": int(sub["assembly_accession"].nunique()),
            "replicons": len(sub),
            "chromosomes": len(chrom),
            "plasmids": len(plas),
            "total_bp": int(sub["length"].sum()),
            "expected_fragments": int(sub["expected_fragments"].sum()),
            "expected_fragments_chromosome": int(chrom["length"].sum() // frag_len) if len(chrom) else 0,
            "expected_fragments_plasmid": int(plas["length"].sum() // frag_len) if len(plas) else 0,
        })
    summary = pd.DataFrame(summary_rows)
    total_frag = summary["expected_fragments"].sum()
    summary["actual_fragment_ratio"] = (summary["expected_fragments"] / max(total_frag, 1)).round(4)

    reports = path_for(cfg, "reports_dir")
    summary.to_csv(reports / f"split_summary{tag}.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))

    # Assemblies must not straddle splits either - they normally cannot,
    # because the whole assembly's replicons share cluster membership only by
    # similarity, so this is reported rather than enforced.
    asm_splits = usable.groupby("assembly_accession")["split"].nunique()
    n_asm_multi = int((asm_splits > 1).sum())
    logger.info("Assemblies whose replicons span >1 split: %d "
                "(expected: chromosome and plasmid of one assembly may cluster apart)",
                n_asm_multi)

    slog.record("replicons assigned to a split", len(usable), len(usable))
    slog.flush(cfg)
    update_manifest(cfg, "step06_split", {
        "unit": cfg["split"]["unit"],
        "ratios": ratios,
        "balance_by": cfg["split"]["balance_by"],
        "random_seed": cfg["random_seed"],
        "keep_assembly_together": bool(cfg["split"].get("keep_assembly_together", False)),
        "split_groups": int(usable["split_group_id"].nunique()),
        "clusters_assigned": int(usable["cluster_id"].nunique()),
        "replicons_assigned": int(len(usable)),
        "clusters_spanning_multiple_splits": n_leaks,
        "assemblies_spanning_multiple_splits": n_asm_multi,
        "per_split": summary.to_dict("records"),
    })
    return usable


if __name__ == "__main__":
    run()
