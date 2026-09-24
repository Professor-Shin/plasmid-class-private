"""Shared helpers: config loading, logging, stage accounting, FASTA I/O."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- config
def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict, *parts: str) -> Path:
    """Resolve a project-relative path to an absolute Path."""
    return Path(cfg["_project_root"]).joinpath(*parts)


def path_for(cfg: dict, key: str) -> Path:
    p = resolve(cfg, cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- logging
def get_logger(cfg: dict, name: str) -> logging.Logger:
    log_dir = path_for(cfg, "logs_dir")
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s")

    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- stage accounting
@dataclass
class StageLog:
    """Records how many records enter and leave every filtering stage.

    The instruction requires reporting data loss at every stage, so each
    step appends its counts here and they are flushed to
    reports/pipeline_stages.csv.
    """

    step: str
    rows: list = field(default_factory=list)

    def record(self, stage: str, n_in: int, n_out: int, note: str = "") -> None:
        self.rows.append(
            {
                "step": self.step,
                "stage": stage,
                "n_in": n_in,
                "n_out": n_out,
                "n_lost": n_in - n_out,
                "note": note,
            }
        )

    def flush(self, cfg: dict) -> Path:
        out = path_for(cfg, "reports_dir") / "pipeline_stages.csv"
        write_header = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["timestamp", "step", "stage", "n_in", "n_out", "n_lost", "note"]
            )
            if write_header:
                w.writeheader()
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for r in self.rows:
                w.writerow({"timestamp": ts, **r})
        return out


# --------------------------------------------------------------------------- manifest
def update_manifest(cfg: dict, step: str, payload: dict) -> Path:
    """Accumulate provenance for the whole run in reports/run_manifest.json."""
    out = path_for(cfg, "reports_dir") / "run_manifest.json"
    manifest = {}
    if out.exists():
        with open(out, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    manifest.setdefault("project_name", cfg.get("project_name"))
    manifest.setdefault("random_seed", cfg.get("random_seed"))
    manifest.setdefault("source", cfg.get("source"))
    manifest["software"] = software_versions()
    manifest.setdefault("steps", {})
    manifest["steps"][step] = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    return out


def software_versions() -> dict:
    vers = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pipeline": __import__("pipeline").__version__,
    }
    for mod in ("pandas", "numpy", "Bio", "yaml"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = "not installed"
    return vers


# --------------------------------------------------------------------------- FASTA
def read_fasta(path: str | os.PathLike):
    """Stream (header, sequence) pairs from a plain or gzipped FASTA file.

    Implemented directly rather than via Bio.SeqIO because the pipeline only
    needs the raw string and this keeps memory flat over large genomes.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def write_fasta_record(fh, seq_id: str, sequence: str, width: int = 70) -> None:
    fh.write(f">{seq_id}\n")
    for i in range(0, len(sequence), width):
        fh.write(sequence[i : i + width] + "\n")


def sha256_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("ascii", "replace")).hexdigest()


def md5_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
