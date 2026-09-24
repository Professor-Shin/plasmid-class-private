"""Step 4 - data quality control over the selected assemblies and replicons.

Produces reports/data_qc.csv (machine readable metric table) and
reports/data_qc.txt (human readable summary). Nothing is discarded here: QC
only measures and reports, so that later filtering decisions are explicit.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import StageLog, get_logger, load_config, path_for, update_manifest

CHROMOSOME, PLASMID, UNKNOWN = "chromosome", "plasmid", "unknown"


def length_stats(series: pd.Series) -> dict:
    if series.empty:
        return {k: 0 for k in ("n", "min", "p25", "median", "mean", "p75", "p95", "max", "total_bp")}
    arr = series.to_numpy(dtype=np.int64)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.median(arr)),
        "mean": int(arr.mean()),
        "p75": int(np.percentile(arr, 75)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "total_bp": int(arr.sum()),
    }


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step04.qc")
    slog = StageLog("step04_qc")

    meta_dir = path_for(cfg, "metadata_dir")
    reports = path_for(cfg, "reports_dir")
    assemblies = pd.read_csv(meta_dir / "assemblies.csv", dtype=str, keep_default_na=False)
    status = pd.read_csv(meta_dir / "download_status.csv", dtype=str, keep_default_na=False)
    rep = pd.read_csv(meta_dir / "replicons.csv", keep_default_na=False)

    frag_len = int(cfg["fragments"]["length"])
    short_thr = int(cfg["replicons"]["short_replicon_threshold"])

    n_selected = len(assemblies)
    n_ok = int((status["status"] != "failed").sum())
    n_failed = int((status["status"] == "failed").sum())

    chrom = rep[rep["replicon_type"] == CHROMOSOME]
    plas = rep[rep["replicon_type"] == PLASMID]
    unk = rep[rep["replicon_type"] == UNKNOWN]

    by_asm = rep.groupby("assembly_accession")["replicon_type"]
    has_chr = by_asm.apply(lambda s: (s == CHROMOSOME).any())
    has_pla = by_asm.apply(lambda s: (s == PLASMID).any())
    n_plasmids_per_asm = rep[rep["replicon_type"] == PLASMID].groupby("assembly_accession").size()

    # Duplicate sequences (identical SHA-256 over the uppercased sequence).
    dup_groups = rep.groupby("sequence_sha256").size()
    dup_groups = dup_groups[dup_groups > 1]
    n_dup_records = int(dup_groups.sum() - len(dup_groups)) if len(dup_groups) else 0

    non_acgt = rep[rep["non_acgt_count"] > 0]
    short_reps = rep[rep["length"] < short_thr]
    too_short_for_frag = rep[rep["length"] < frag_len]

    metrics: list[dict] = []

    def add(metric: str, value, group: str = "", note: str = "") -> None:
        metrics.append({"metric": metric, "group": group, "value": value, "note": note})

    add("assemblies_selected", n_selected, "assemblies")
    add("assemblies_downloaded", n_ok, "assemblies")
    add("assemblies_failed_download", n_failed, "assemblies")
    add("assemblies_parsed", int(rep["assembly_accession"].nunique()), "assemblies")
    add("distinct_species_taxid", int(assemblies["species_taxid"].nunique()), "assemblies")
    add("fasta_records_total", len(rep), "replicons", "one FASTA record = one replicon")
    add("chromosomes", len(chrom), "replicons")
    add("plasmids", len(plas), "replicons")
    add("unknown_replicons", len(unk), "replicons", "excluded from the binary ML dataset")
    add("assemblies_with_chromosome_and_plasmid", int((has_chr & has_pla).sum()), "assemblies")
    add("assemblies_with_multiple_plasmids", int((n_plasmids_per_asm > 1).sum()), "assemblies")
    add("assemblies_with_no_plasmid", int((~has_pla).sum()), "assemblies")
    add("assemblies_with_multiple_chromosomes",
        int((rep[rep["replicon_type"] == CHROMOSOME].groupby("assembly_accession").size() > 1).sum()),
        "assemblies")
    add("max_plasmids_in_one_assembly",
        int(n_plasmids_per_asm.max()) if len(n_plasmids_per_asm) else 0, "assemblies")
    add("duplicate_sequence_groups", int(len(dup_groups)), "sequences",
        "identical SHA-256 sequences shared by >1 replicon")
    add("duplicate_replicon_records", n_dup_records, "sequences",
        "records beyond the first in each duplicate group")
    add("replicons_with_non_acgt", len(non_acgt), "sequences")
    add("non_acgt_bases_total", int(rep["non_acgt_count"].sum()), "sequences")
    add("max_non_acgt_fraction_in_a_replicon",
        float(rep["non_acgt_fraction"].max()) if len(rep) else 0.0, "sequences")
    add("replicons_shorter_than_threshold", len(short_reps), "sequences",
        f"length < {short_thr} bp")
    add("replicons_shorter_than_fragment_length", len(too_short_for_frag), "sequences",
        f"length < {frag_len} bp - cannot yield a fragment")
    add("classification_by_assembly_report",
        int((rep["classification_source"] == "assembly_report").sum()), "classification")
    add("classification_by_defline",
        int((rep["classification_source"] == "defline").sum()), "classification")

    for name, subset in ((CHROMOSOME, chrom), (PLASMID, plas), (UNKNOWN, unk)):
        for k, v in length_stats(subset["length"]).items():
            add(f"length_{k}", v, f"{name}_length_distribution")
        est = (subset["length"] // frag_len).sum() if len(subset) else 0
        add("expected_fragments", int(est), f"{name}_length_distribution",
            f"non-overlapping {frag_len} bp fragments")

    qc = pd.DataFrame(metrics)
    qc_csv = reports / "data_qc.csv"
    qc.to_csv(qc_csv, index=False)

    # Per-assembly QC detail, useful when chasing a specific accession.
    per_asm = (
        rep.groupby("assembly_accession")
        .agg(
            organism_name=("organism_name", "first"),
            taxid=("taxid", "first"),
            replicons=("replicon_id", "size"),
            chromosomes=("replicon_type", lambda s: int((s == CHROMOSOME).sum())),
            plasmids=("replicon_type", lambda s: int((s == PLASMID).sum())),
            unknown=("replicon_type", lambda s: int((s == UNKNOWN).sum())),
            total_bp=("length", "sum"),
            non_acgt=("non_acgt_count", "sum"),
        )
        .reset_index()
    )
    per_asm.to_csv(reports / "qc_per_assembly.csv", index=False)

    if len(dup_groups):
        dups = rep[rep["sequence_sha256"].isin(dup_groups.index)].sort_values("sequence_sha256")
        dups.to_csv(reports / "qc_duplicate_sequences.csv", index=False)

    # ------------------------------------------------------------------ text
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("DATA QUALITY REPORT - RefSeq plasmid vs chromosome dataset")
    w("=" * 78)
    w("")
    w("1.  Assemblies selected .................. %d" % n_selected)
    w("2.  Assemblies downloaded ................ %d" % n_ok)
    w("3.  Failed downloads ..................... %d" % n_failed)
    w("4.  FASTA records (replicons) ............ %d" % len(rep))
    w("5.  Chromosomes .......................... %d" % len(chrom))
    w("6.  Plasmids ............................. %d" % len(plas))
    w("7.  Unknown replicons .................... %d" % len(unk))
    w("")
    w("8.  Chromosome length distribution (bp)")
    for k, v in length_stats(chrom["length"]).items():
        w("        %-10s %s" % (k, f"{v:,}"))
    w("")
    w("9.  Plasmid length distribution (bp)")
    for k, v in length_stats(plas["length"]).items():
        w("        %-10s %s" % (k, f"{v:,}"))
    w("")
    w("10. Assemblies with both chromosome and plasmid ... %d / %d (%.1f%%)"
      % (int((has_chr & has_pla).sum()), n_selected,
         100.0 * int((has_chr & has_pla).sum()) / max(n_selected, 1)))
    w("11. Assemblies with multiple plasmids ............. %d (max %d in one assembly)"
      % (int((n_plasmids_per_asm > 1).sum()),
         int(n_plasmids_per_asm.max()) if len(n_plasmids_per_asm) else 0))
    w("12. Duplicate sequences ........................... %d groups, %d redundant records"
      % (len(dup_groups), n_dup_records))
    w("13. Invalid / non-ACGT characters ................. %d replicons, %d bases (max fraction %.6f)"
      % (len(non_acgt), int(rep["non_acgt_count"].sum()),
         float(rep["non_acgt_fraction"].max()) if len(rep) else 0.0))
    w("14. Extremely short replicons ..................... %d < %d bp; %d < %d bp (unusable for fragments)"
      % (len(short_reps), short_thr, len(too_short_for_frag), frag_len))
    w("")
    w("Replicon type evidence")
    for src, cnt in rep["classification_source"].value_counts().items():
        w("        %-20s %d" % (src, cnt))
    w("")
    w("Nothing was discarded by this step. 'unknown' replicons stay in")
    w("metadata/replicons.csv and reports/unknown_replicons.csv; they are")
    w("excluded only from the binary train/val/test dataset.")
    w("")
    w("Expected non-overlapping %d bp fragments: chromosome=%s plasmid=%s"
      % (frag_len,
         f"{int((chrom['length'] // frag_len).sum()):,}",
         f"{int((plas['length'] // frag_len).sum()):,}"))
    w("=" * 78)

    qc_txt = reports / "data_qc.txt"
    qc_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", qc_csv, qc_txt)
    print("\n".join(lines))

    slog.record("replicons examined", len(rep), len(rep), "QC discards nothing")
    slog.flush(cfg)
    update_manifest(cfg, "step04_qc", {
        "qc_csv": str(qc_csv),
        "qc_txt": str(qc_txt),
        "chromosomes": len(chrom),
        "plasmids": len(plas),
        "unknown": len(unk),
        "duplicate_sequence_groups": int(len(dup_groups)),
        "replicons_with_non_acgt": len(non_acgt),
    })
    return qc


if __name__ == "__main__":
    run()
