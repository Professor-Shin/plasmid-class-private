"""Step 7 - generate 5-kb genomic fragments and write the final dataset.

These are 5-kb GENOMIC FRAGMENTS: fixed-length slices of complete
chromosomes and plasmids. They are deliberately NOT called contigs - a slice
of a finished replicon has none of the properties of a real assembly contig
(no assembly errors, no coverage-driven breakpoints, no chimeras). Simulated
contigs, if ever needed, belong in a separate experiment.

Strategy: ``fragments.strategy`` in config.yaml (default ``non_overlapping``,
step == length). The trailing remainder shorter than one fragment is dropped.

Splits were decided in step 6, at the parent-replicon/cluster level, so every
fragment simply inherits the split of its parent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    StageLog,
    get_logger,
    load_config,
    path_for,
    read_fasta,
    update_manifest,
    write_fasta_record,
)

SPLITS = ("train", "validation", "test")
FRAGMENT_COLUMNS = [
    "fragment_id", "replicon_id", "assembly_accession", "sequence_id",
    "organism_name", "taxid", "replicon_type", "label", "start", "end",
    "length", "split",
]
EXTRA_COLUMNS = ["species_taxid", "cluster_id", "gc_content", "non_acgt_fraction"]


def fragment_offsets(length: int, frag_len: int, step: int, drop_partial: bool) -> range:
    if length < frag_len:
        return range(0)
    last = length - frag_len
    return range(0, last + 1, step) if drop_partial else range(0, length, step)


def base_stats(seq: str) -> tuple[float, float]:
    arr = np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)
    n = arr.size
    if n == 0:
        return 0.0, 1.0
    a = int((arr == 65).sum()); c = int((arr == 67).sum())
    g = int((arr == 71).sum()); t = int((arr == 84).sum())
    acgt = a + c + g + t
    gc = (g + c) / acgt if acgt else 0.0
    return round(gc, 6), round((n - acgt) / n, 6)


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step07.fragments")
    slog = StageLog("step07_fragments")

    fcfg = cfg["fragments"]
    frag_len = int(fcfg["length"])
    step = int(fcfg.get("step", frag_len))
    drop_partial = bool(fcfg.get("drop_last_partial", True))
    max_bad = float(fcfg.get("max_non_acgt_fraction", 0.05))
    if fcfg["strategy"] == "non_overlapping" and step != frag_len:
        raise ValueError("strategy 'non_overlapping' requires step == length")

    meta_dir = path_for(cfg, "metadata_dir")
    dataset_dir = path_for(cfg, "dataset_dir")
    split_df = pd.read_csv(meta_dir / "replicons_split.csv", keep_default_na=False)
    logger.info("Fragmenting %d parent replicons (strategy=%s, length=%d, step=%d)",
                len(split_df), fcfg["strategy"], frag_len, step)

    # Group by FASTA file so each file is read exactly once.
    by_file: dict[str, list] = {}
    for row in split_df.itertuples(index=False):
        by_file.setdefault(row.source_fasta, []).append(row)

    handles = {s: open(dataset_dir / f"{s}.fasta", "w", encoding="utf-8", newline="\n")
               for s in SPLITS}
    fragment_rows: list[dict] = []
    counter = 0
    dropped_bad_bases = 0
    replicons_done = 0
    replicons_no_fragment = 0

    try:
        for fasta_path in sorted(by_file):
            wanted = {r.sequence_id: r for r in by_file[fasta_path]}
            for defline, seq in read_fasta(fasta_path):
                row = wanted.get(defline.split()[0])
                if row is None:
                    continue
                offsets = fragment_offsets(len(seq), frag_len, step, drop_partial)
                if len(offsets) == 0:
                    replicons_no_fragment += 1
                    continue
                for start in offsets:
                    end = min(start + frag_len, len(seq))
                    piece = seq[start:end].upper()
                    if drop_partial and len(piece) < frag_len:
                        continue
                    gc, bad_frac = base_stats(piece)
                    if bad_frac > max_bad:
                        dropped_bad_bases += 1
                        continue
                    counter += 1
                    frag_id = f"fragment_{counter:08d}"
                    fragment_rows.append({
                        "fragment_id": frag_id,
                        "replicon_id": row.replicon_id,
                        "assembly_accession": row.assembly_accession,
                        "sequence_id": row.sequence_id,
                        "organism_name": row.organism_name,
                        "taxid": row.taxid,
                        "replicon_type": row.replicon_type,
                        "label": int(row.label),
                        "start": int(start),
                        "end": int(end),
                        "length": len(piece),
                        "split": row.split,
                        "species_taxid": row.species_taxid,
                        "cluster_id": row.cluster_id,
                        "gc_content": gc,
                        "non_acgt_fraction": bad_frac,
                    })
                    write_fasta_record(handles[row.split], frag_id, piece)
                replicons_done += 1
                if replicons_done % 200 == 0:
                    logger.info("  %d replicons -> %d fragments", replicons_done, counter)
    finally:
        for fh in handles.values():
            fh.close()

    frags = pd.DataFrame(fragment_rows, columns=FRAGMENT_COLUMNS + EXTRA_COLUMNS)
    frags_path = meta_dir / "fragments.csv"
    frags.to_csv(frags_path, index=False)
    logger.info("Wrote %s (%d fragments)", frags_path, len(frags))

    for s in SPLITS:
        sub = frags[frags["split"] == s]
        sub.to_csv(dataset_dir / f"{s}.csv", index=False)
        logger.info("%-11s %8d fragments (chromosome=%d plasmid=%d)",
                    s, len(sub), int((sub["label"] == 0).sum()), int((sub["label"] == 1).sum()))

    # -------------------------------------------------------------- integrity
    problems: list[str] = []
    for s in SPLITS:
        ids_csv = set(pd.read_csv(dataset_dir / f"{s}.csv", usecols=["fragment_id"])["fragment_id"])
        ids_fa = {line[1:].strip() for line in
                  open(dataset_dir / f"{s}.fasta", encoding="utf-8") if line.startswith(">")}
        if ids_csv != ids_fa:
            problems.append(f"{s}: FASTA/CSV id mismatch "
                            f"({len(ids_fa - ids_csv)} fasta-only, {len(ids_csv - ids_fa)} csv-only)")
    overlap = frags.groupby("replicon_id")["split"].nunique()
    if int((overlap > 1).sum()):
        problems.append("replicons whose fragments span multiple splits: "
                        f"{int((overlap > 1).sum())}")
    if frags["fragment_id"].duplicated().any():
        problems.append("duplicate fragment_id values")
    if problems:
        raise RuntimeError("dataset integrity checks failed: " + "; ".join(problems))
    logger.info("Integrity checks passed: fragment_id sets match between FASTA and CSV, "
                "no replicon spans multiple splits, no duplicate ids")

    # ---------------------------------------------------------------- summary
    summary_rows = []
    for s in SPLITS:
        sub = frags[frags["split"] == s]
        summary_rows.append({
            "split": s,
            "fragments": len(sub),
            "fragments_chromosome": int((sub["label"] == 0).sum()),
            "fragments_plasmid": int((sub["label"] == 1).sum()),
            "plasmid_fraction": round(float((sub["label"] == 1).mean()), 6) if len(sub) else 0.0,
            "replicons": int(sub["replicon_id"].nunique()),
            "chromosomes": int(sub[sub["label"] == 0]["replicon_id"].nunique()),
            "plasmids": int(sub[sub["label"] == 1]["replicon_id"].nunique()),
            "assemblies": int(sub["assembly_accession"].nunique()),
            "clusters": int(sub["cluster_id"].nunique()),
            "species": int(sub["species_taxid"].nunique()),
            "total_bp": int(sub["length"].sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary["fragment_ratio"] = (summary["fragments"] / max(len(frags), 1)).round(4)
    reports = path_for(cfg, "reports_dir")
    summary.to_csv(reports / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))

    slog.record("parent replicons fragmented", len(split_df), replicons_done,
                f"{replicons_no_fragment} produced no fragment (< {frag_len} bp)")
    slog.record("candidate fragments kept", len(frags) + dropped_bad_bases, len(frags),
                f"{dropped_bad_bases} dropped: non-ACGT fraction > {max_bad}")
    slog.flush(cfg)
    update_manifest(cfg, "step07_fragments", {
        "fragment_length": frag_len,
        "strategy": fcfg["strategy"],
        "step": step,
        "drop_last_partial": drop_partial,
        "max_non_acgt_fraction": max_bad,
        "fragments_total": int(len(frags)),
        "fragments_dropped_non_acgt": dropped_bad_bases,
        "replicons_fragmented": replicons_done,
        "replicons_without_fragments": replicons_no_fragment,
        "per_split": summary.to_dict("records"),
        "fragments_csv": str(frags_path),
        "dataset_dir": str(dataset_dir),
    })
    return frags


if __name__ == "__main__":
    run()
