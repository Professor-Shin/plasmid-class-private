"""Step 1 - robustly parse assembly_summary.txt and select assemblies.

The NCBI assembly summary is a tab-delimited table whose header line is itself
a comment line starting with "#assembly_accession". The first line of the file
is a *different* comment, so the header row has to be located rather than
assumed, and ``pd.read_csv(..., comment="#")`` must not be used because it
would strip the header away along with the comments.
"""
from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import StageLog, get_logger, load_config, path_for, resolve, update_manifest

HEADER_TOKEN = "assembly_accession"

# Fields the instruction requires us to retain at minimum. Everything else in
# the file is preserved too - unused metadata is still written out.
REQUIRED_FIELDS = [
    "assembly_accession",
    "bioproject",
    "biosample",
    "organism_name",
    "taxid",
    "species_taxid",
    "refseq_category",
    "assembly_level",
    "version_status",
    "seq_rel_date",
    "ftp_path",
]


def find_header_line(path: Path, max_scan: int = 200) -> tuple[int, list[str]]:
    """Return (0-based line index, column names) of the real header row."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            if idx > max_scan:
                break
            stripped = line.lstrip("#").strip()
            if stripped.startswith(HEADER_TOKEN):
                cols = [c.strip() for c in stripped.split("\t")]
                if len(cols) >= 10:
                    return idx, cols
    raise ValueError(
        f"Could not locate a header row starting with '#{HEADER_TOKEN}' in {path}"
    )


def load_assembly_summary(path: Path, logger) -> pd.DataFrame:
    header_idx, columns = find_header_line(path)
    logger.info("Header row found at line %d with %d columns", header_idx + 1, len(columns))
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        skiprows=header_idx + 1,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        engine="c",
        on_bad_lines="warn",
    )
    logger.info("Parsed %d assembly rows", len(df))
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"assembly_summary.txt is missing required columns: {missing}")
    return df


def stage_assembly_summary(cfg: dict, logger) -> Path:
    """Ensure data/raw/assembly_summary.txt exists (hardlink, else copy)."""
    dest = resolve(cfg, cfg["source"]["assembly_summary"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    src = resolve(cfg, "assembly_summary.txt")
    if not src.exists():
        raise FileNotFoundError(
            f"{dest} not found and no assembly_summary.txt at the project root. "
            f"Download it from {cfg['source']['url']}"
        )
    try:
        os.link(src, dest)
        logger.info("Hardlinked %s -> %s", src, dest)
    except OSError:
        shutil.copy2(src, dest)
        logger.info("Copied %s -> %s", src, dest)
    return dest


def sample_assemblies(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    """Deterministically reduce the filtered set to a workable size.

    species_balanced: keep at most ``max_per_species`` assemblies per
    species_taxid - which stops over-represented species (E. coli,
    S. enterica, K. pneumoniae) from dominating - then take a seeded random
    subsample down to ``max_assemblies``.
    """
    scfg = cfg["sampling"]
    if not scfg.get("enabled", False):
        logger.info("Sampling disabled - using all %d filtered assemblies", len(df))
        return df

    seed = int(cfg["random_seed"])
    rng = np.random.default_rng(seed)
    # Sort first so the row order of the input file can never affect the result.
    df = df.sort_values("assembly_accession", kind="mergesort").reset_index(drop=True)

    if scfg["strategy"] == "species_balanced":
        per_species = int(scfg["max_per_species"])
        keep_idx: list[int] = []
        for _, group in df.groupby("species_taxid", sort=True):
            idx = group.index.to_numpy()
            if len(idx) > per_species:
                idx = rng.choice(idx, size=per_species, replace=False)
            keep_idx.extend(int(i) for i in idx)
        out = df.loc[sorted(keep_idx)]
        logger.info(
            "Species balancing: %d -> %d assemblies (<= %d per species_taxid, %d species)",
            len(df), len(out), per_species, out["species_taxid"].nunique(),
        )
    elif scfg["strategy"] == "random":
        out = df
    else:
        raise ValueError(f"Unknown sampling strategy: {scfg['strategy']}")

    max_n = scfg.get("max_assemblies")
    if max_n and len(out) > int(max_n):
        pick = rng.choice(out.index.to_numpy(), size=int(max_n), replace=False)
        out = out.loc[sorted(int(i) for i in pick)]
        logger.info("Subsampled to max_assemblies=%d (seed=%d)", int(max_n), seed)
    return out.sort_values("assembly_accession", kind="mergesort").reset_index(drop=True)


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step01.filter")
    slog = StageLog("step01_filter_assemblies")

    summary_path = stage_assembly_summary(cfg, logger)
    df = load_assembly_summary(summary_path, logger)
    n_total = len(df)
    slog.record("parsed_assembly_summary", n_total, n_total, str(summary_path))

    f = cfg["filter"]
    cur = df

    n = len(cur)
    cur = cur[cur["assembly_level"] == f["assembly_level"]]
    slog.record(f"assembly_level == {f['assembly_level']!r}", n, len(cur))
    logger.info("assembly_level filter: %d -> %d", n, len(cur))

    n = len(cur)
    cur = cur[cur["version_status"] == f["version_status"]]
    slog.record(f"version_status == {f['version_status']!r}", n, len(cur))
    logger.info("version_status filter: %d -> %d", n, len(cur))

    if f.get("group") and "group" in cur.columns:
        n = len(cur)
        cur = cur[cur["group"] == f["group"]]
        slog.record(f"group == {f['group']!r}", n, len(cur))
        logger.info("group filter: %d -> %d", n, len(cur))

    if f.get("require_refseq_accession", True):
        n = len(cur)
        cur = cur[cur["assembly_accession"].str.startswith("GCF_")]
        slog.record("RefSeq accession (GCF_*)", n, len(cur))
        logger.info("RefSeq accession filter: %d -> %d", n, len(cur))

    if f.get("require_ftp_path", True):
        n = len(cur)
        cur = cur[(cur["ftp_path"] != "na") & (cur["ftp_path"].str.len() > 0)]
        slog.record("usable ftp_path", n, len(cur))
        logger.info("ftp_path filter: %d -> %d", n, len(cur))

    if f.get("drop_excluded_from_refseq", True) and "excluded_from_refseq" in cur.columns:
        n = len(cur)
        cur = cur[cur["excluded_from_refseq"].isin(["na", ""])]
        slog.record("excluded_from_refseq == na", n, len(cur))
        logger.info("excluded_from_refseq filter: %d -> %d", n, len(cur))

    filtered = cur.reset_index(drop=True)
    meta_dir = path_for(cfg, "metadata_dir")

    # Full filtered population, kept so the sampling frame is traceable.
    all_path = meta_dir / "assemblies_filtered_all.csv"
    filtered.to_csv(all_path, index=False)
    logger.info("Wrote %s (%d rows)", all_path, len(filtered))

    n = len(filtered)
    selected = sample_assemblies(filtered, cfg, logger)
    strategy = cfg["sampling"]["strategy"] if cfg["sampling"]["enabled"] else "disabled"
    slog.record(f"sampling ({strategy})", n, len(selected))

    # Required fields first; every other original column is preserved after them.
    ordered = REQUIRED_FIELDS + [c for c in selected.columns if c not in REQUIRED_FIELDS]
    selected = selected[ordered]
    out_path = meta_dir / "assemblies.csv"
    selected.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d selected assemblies)", out_path, len(selected))

    slog.flush(cfg)
    dl_date = cfg["source"].get("download_date") or datetime.fromtimestamp(
        summary_path.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")
    update_manifest(
        cfg,
        "step01_filter_assemblies",
        {
            "assembly_summary_path": str(summary_path),
            "assembly_summary_download_date": dl_date,
            "assembly_summary_bytes": summary_path.stat().st_size,
            "rows_in_summary": n_total,
            "filter_criteria": f,
            "assemblies_after_filtering": len(filtered),
            "sampling": cfg["sampling"],
            "assemblies_selected": len(selected),
            "distinct_species_taxid_selected": int(selected["species_taxid"].nunique()),
        },
    )
    return selected


if __name__ == "__main__":
    run()
