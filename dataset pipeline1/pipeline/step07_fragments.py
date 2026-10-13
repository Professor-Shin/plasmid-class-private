"""Step 7 - generate genomic fragments and write the final dataset.

These are GENOMIC FRAGMENTS: slices of complete chromosomes and plasmids. They
are deliberately NOT called contigs - a slice of a finished replicon has none of
the properties of a real assembly contig (no assembly errors, no coverage-driven
breakpoints, no chimeras). Simulated contigs belong in a separate experiment.

Two modes, selected by ``fragments.mode`` in config.yaml:

``fixed``
    The original behaviour: non-overlapping windows of ``fragments.length`` bp,
    trailing remainder dropped. Kept so the published 5-kb dataset still rebuilds.

``random_length``
    Each window's length is drawn from ``fragments.length_sampling`` (log-uniform
    by default), so no single fragment length is privileged. A tail too short to
    stand alone is absorbed into the previous window, and a replicon shorter than
    the sampled length becomes a single fragment at its true size.

The expert review prompted ``random_length``: 5 kb was a round number with no
biological meaning, and a 5-kb floor emitted no fragment at all for the 10.2% of
plasmids shorter than 5 kb (down to 1119 bp), which silently removed exactly the
small ColE1/rolling-circle plasmids from the problem.

Splits were decided in step 6, at the parent-replicon/cluster level, so every
fragment inherits the split of its parent - a replicon and all of its windows
always land in the same split, whatever their lengths.
"""
from __future__ import annotations

import hashlib
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
EXTRA_COLUMNS = ["species_taxid", "cluster_id", "gc_content", "non_acgt_fraction",
                 "sampled_length", "length_bucket"]

# Evaluation buckets for the length-stratified report. Upper bound is exclusive;
# the last bucket is open-ended so a window at the sampling maximum still lands.
LENGTH_BUCKETS = [(1000, 2000), (2000, 5000), (5000, 10000),
                  (10000, 25000), (25000, 50000), (50000, float("inf"))]


def bucket_label(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n < hi:
            return f"{lo // 1000}-{'100' if hi == float('inf') else hi // 1000}kb"
    return "under-1kb"


def fragment_offsets(length: int, frag_len: int, step: int, drop_partial: bool) -> range:
    if length < frag_len:
        return range(0)
    last = length - frag_len
    return range(0, last + 1, step) if drop_partial else range(0, length, step)


def replicon_rng(seed: int, replicon_id: str) -> np.random.Generator:
    """A generator keyed to one replicon, so the dataset rebuilds byte-for-byte.

    Seeding from the replicon id rather than from a single stream means the
    windows of one replicon do not depend on how many replicons were processed
    before it, so the FASTA read order cannot change the result.
    """
    digest = hashlib.sha256(replicon_id.encode("utf-8")).digest()[:8]
    return np.random.default_rng([int(seed), int.from_bytes(digest, "little")])


def sample_length(rng: np.random.Generator, lmin: int, lmax: int, dist: str) -> int:
    if dist == "log_uniform":
        return int(round(float(np.exp(rng.uniform(np.log(lmin), np.log(lmax))))))
    if dist == "uniform":
        return int(rng.integers(lmin, lmax + 1))
    raise ValueError(f"unknown length distribution {dist!r}")


def random_windows(seq_len: int, rng: np.random.Generator, lmin: int, lmax: int,
                   dist: str) -> list[tuple[int, int]]:
    """Tile [0, seq_len) with non-overlapping windows of randomly drawn length.

    A remainder that would be shorter than ``lmin`` is absorbed into the window
    before it instead of being discarded, so no sequence is thrown away and a
    replicon shorter than the drawn length survives whole - that is what lets
    sub-5-kb plasmids into the dataset at their true size.
    """
    windows: list[tuple[int, int]] = []
    pos = 0
    while seq_len - pos >= lmin:
        remaining = seq_len - pos
        drawn = sample_length(rng, lmin, lmax, dist)
        taken = remaining if (drawn >= remaining or remaining - drawn < lmin) else drawn
        windows.append((pos, taken))
        pos += taken
    return windows


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
    mode = fcfg.get("mode", "fixed")
    frag_len = int(fcfg["length"])
    step = int(fcfg.get("step", frag_len))
    drop_partial = bool(fcfg.get("drop_last_partial", True))
    max_bad = float(fcfg.get("max_non_acgt_fraction", 0.05))
    if mode == "fixed" and fcfg["strategy"] == "non_overlapping" and step != frag_len:
        raise ValueError("strategy 'non_overlapping' requires step == length")

    lcfg = fcfg.get("length_sampling") or {}
    lmin = int(lcfg.get("min", 1000))
    lmax = int(lcfg.get("max", 100000))
    ldist = str(lcfg.get("dist", "log_uniform"))
    seed = int(cfg.get("random_seed", 42))
    if mode == "random_length" and not 0 < lmin < lmax:
        raise ValueError("length_sampling requires 0 < min < max")

    meta_dir = path_for(cfg, "metadata_dir")
    dataset_dir = path_for(cfg, fcfg.get("output_dir_key", "dataset_dir"))
    tag = fcfg.get("report_tag", "")
    split_df = pd.read_csv(meta_dir / f"replicons_split{tag}.csv", keep_default_na=False)
    if mode == "random_length":
        logger.info("Fragmenting %d parent replicons (mode=random_length, "
                    "%s length in [%d, %d] bp, seed=%d) -> %s",
                    len(split_df), ldist, lmin, lmax, seed, dataset_dir)
    else:
        logger.info("Fragmenting %d parent replicons (strategy=%s, length=%d, step=%d) -> %s",
                    len(split_df), fcfg["strategy"], frag_len, step, dataset_dir)

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
                if mode == "random_length":
                    rng = replicon_rng(seed, str(row.replicon_id))
                    windows = random_windows(len(seq), rng, lmin, lmax, ldist)
                else:
                    windows = [(s, frag_len)
                               for s in fragment_offsets(len(seq), frag_len, step, drop_partial)]
                if len(windows) == 0:
                    replicons_no_fragment += 1
                    continue
                for start, want in windows:
                    end = min(start + want, len(seq))
                    piece = seq[start:end].upper()
                    if mode == "fixed" and drop_partial and len(piece) < frag_len:
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
                        "sampled_length": int(want),
                        "length_bucket": bucket_label(len(piece)),
                    })
                    write_fasta_record(handles[row.split], frag_id, piece)
                replicons_done += 1
                if replicons_done % 200 == 0:
                    logger.info("  %d replicons -> %d fragments", replicons_done, counter)
    finally:
        for fh in handles.values():
            fh.close()

    frags = pd.DataFrame(fragment_rows, columns=FRAGMENT_COLUMNS + EXTRA_COLUMNS)
    frags_path = meta_dir / f"fragments{tag}.csv"
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
    summary.to_csv(reports / f"dataset_summary{tag}.csv", index=False)
    print(summary.to_string(index=False))

    # ------------------------------------------------- length distribution report
    # The expert's first comment was that 5 kb is a magic number, so the realised
    # spread of fragment lengths is a reported artefact, not an internal detail.
    order = [f"{lo // 1000}-{'100' if hi == float('inf') else hi // 1000}kb"
             for lo, hi in LENGTH_BUCKETS]
    dist_rows = []
    for bucket in order:
        sub = frags[frags["length_bucket"] == bucket]
        if not len(sub):
            continue
        dist_rows.append({
            "length_bucket": bucket,
            "fragments": len(sub),
            "fragment_fraction": round(len(sub) / max(len(frags), 1), 6),
            "fragments_chromosome": int((sub["label"] == 0).sum()),
            "fragments_plasmid": int((sub["label"] == 1).sum()),
            "plasmid_fraction": round(float((sub["label"] == 1).mean()), 6),
            "total_bp": int(sub["length"].sum()),
            "replicons": int(sub["replicon_id"].nunique()),
            "plasmid_replicons": int(sub[sub["label"] == 1]["replicon_id"].nunique()),
            "mean_length": int(sub["length"].mean()),
            "mean_gc": round(float(sub["gc_content"].mean()), 6),
            **{f"n_{s}": int((sub["split"] == s).sum()) for s in SPLITS},
        })
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(reports / f"fragment_length_distribution{tag}.csv", index=False)

    lengths = frags["length"].to_numpy()
    quantiles = {f"p{int(q * 100)}": int(np.quantile(lengths, q))
                 for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    length_stats = {"n": int(len(lengths)), "min": int(lengths.min()),
                    "max": int(lengths.max()), "mean": int(lengths.mean()),
                    **quantiles}
    logger.info("Fragment length distribution: %s", length_stats)
    print("\nfragment length distribution by bucket")
    print(dist.to_string(index=False))

    slog.record("parent replicons fragmented", len(split_df), replicons_done,
                f"{replicons_no_fragment} produced no fragment "
                f"(< {lmin if mode == 'random_length' else frag_len} bp)")
    slog.record("candidate fragments kept", len(frags) + dropped_bad_bases, len(frags),
                f"{dropped_bad_bases} dropped: non-ACGT fraction > {max_bad}")
    slog.flush(cfg)
    update_manifest(cfg, f"step07_fragments{tag}", {
        "mode": mode,
        "fragment_length": frag_len if mode == "fixed" else None,
        "length_sampling": {"dist": ldist, "min": lmin, "max": lmax,
                            "seed": seed} if mode == "random_length" else None,
        "strategy": fcfg["strategy"],
        "step": step,
        "drop_last_partial": drop_partial,
        "max_non_acgt_fraction": max_bad,
        "fragments_total": int(len(frags)),
        "fragments_dropped_non_acgt": dropped_bad_bases,
        "replicons_fragmented": replicons_done,
        "replicons_without_fragments": replicons_no_fragment,
        "length_stats": length_stats,
        "length_distribution": dist.to_dict("records"),
        "per_split": summary.to_dict("records"),
        "fragments_csv": str(frags_path),
        "dataset_dir": str(dataset_dir),
    })
    return frags


if __name__ == "__main__":
    run()
