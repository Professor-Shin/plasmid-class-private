"""Step 2 - download *_genomic.fna.gz (and the assembly report) per assembly.

One directory per assembly under data/fasta/. Files are never merged at this
stage. Downloads are resumable in the sense that an assembly whose files are
already present and verified is skipped entirely, and partial downloads are
written to a .part file so an interrupted run can never leave a truncated
.fna.gz behind that a later run would trust.
"""
from __future__ import annotations

import concurrent.futures as futures
import csv
import gzip
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .utils import StageLog, get_logger, load_config, md5_file, path_for, update_manifest

USER_AGENT = "refseq-plasmid-chromosome-pipeline/1.0 (academic dataset build)"


def https_url(ftp_path: str, cfg: dict) -> str:
    """NCBI's ftp_path column may use ftp:// - HTTPS is more reliable."""
    base = cfg["source"]["fasta_base_url"].rstrip("/")
    p = ftp_path.strip().rstrip("/")
    for prefix in ("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov", "http://ftp.ncbi.nlm.nih.gov"):
        if p.startswith(prefix):
            return base + p[len(prefix):]
    return p


def fetch(url: str, dest: Path, cfg: dict) -> None:
    """Download url to dest atomically, with retries and backoff."""
    dcfg = cfg["download"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, int(dcfg["retries"]) + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=int(dcfg["timeout_seconds"])) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt < int(dcfg["retries"]):
                time.sleep(float(dcfg["backoff_seconds"]) * attempt)
    raise RuntimeError(f"{url}: {last_err}")


def gzip_is_readable(path: Path) -> bool:
    """Cheap integrity check: the gzip member must decompress end-to-end."""
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 22):
                pass
        return True
    except Exception:
        return False


def parse_md5sums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    sums[Path(parts[1]).name] = parts[0]
    except OSError:
        pass
    return sums


def download_assembly(row: dict, cfg: dict) -> dict:
    """Fetch one assembly's files. Returns a status record."""
    acc = row["assembly_accession"]
    out_dir = path_for(cfg, "fasta_dir") / acc
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = https_url(row["ftp_path"], cfg)
    asm_name = base_url.rstrip("/").split("/")[-1]  # e.g. GCF_000005845.2_ASM584v2
    fna_name = f"{asm_name}_genomic.fna.gz"
    rpt_name = f"{asm_name}_assembly_report.txt"
    fna_path = out_dir / fna_name
    rpt_path = out_dir / rpt_name

    rec = {
        "assembly_accession": acc,
        "organism_name": row.get("organism_name", ""),
        "url": f"{base_url}/{fna_name}",
        "fasta_path": str(fna_path),
        "assembly_report_path": "",
        "status": "",
        "bytes": 0,
        "md5_verified": False,
        "skipped_existing": False,
        "error": "",
    }

    dcfg = cfg["download"]
    try:
        md5s: dict[str, str] = {}
        if dcfg.get("verify_md5", True):
            md5_path = out_dir / "md5checksums.txt"
            if not md5_path.exists():
                try:
                    fetch(f"{base_url}/md5checksums.txt", md5_path, cfg)
                except Exception:  # noqa: BLE001 - md5 file is a nicety, not a blocker
                    pass
            md5s = parse_md5sums(md5_path)

        expected = md5s.get(fna_name)
        if fna_path.exists() and fna_path.stat().st_size > 0:
            ok = md5_file(fna_path) == expected if expected else gzip_is_readable(fna_path)
            if ok:
                rec.update(status="skipped", skipped_existing=True,
                           bytes=fna_path.stat().st_size, md5_verified=bool(expected))
            else:
                fna_path.unlink(missing_ok=True)

        if rec["status"] != "skipped":
            fetch(rec["url"], fna_path, cfg)
            if expected and md5_file(fna_path) != expected:
                raise RuntimeError("md5 mismatch after download")
            if not expected and not gzip_is_readable(fna_path):
                raise RuntimeError("downloaded file is not a readable gzip stream")
            rec.update(status="downloaded", bytes=fna_path.stat().st_size,
                       md5_verified=bool(expected))

        # The assembly report carries NCBI's own molecule assignment
        # (chromosome / plasmid) per sequence - the authoritative label source.
        if dcfg.get("fetch_assembly_report", True):
            if not rpt_path.exists() or rpt_path.stat().st_size == 0:
                try:
                    fetch(f"{base_url}/{rpt_name}", rpt_path, cfg)
                except Exception:  # noqa: BLE001 - fall back to defline parsing
                    rpt_path.unlink(missing_ok=True)
            if rpt_path.exists():
                rec["assembly_report_path"] = str(rpt_path)
    except Exception as exc:  # noqa: BLE001
        rec.update(status="failed", error=str(exc)[:300])
    return rec


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step02.download")
    slog = StageLog("step02_download_fasta")

    meta_dir = path_for(cfg, "metadata_dir")
    assemblies = pd.read_csv(meta_dir / "assemblies.csv", dtype=str, keep_default_na=False)
    rows = assemblies.to_dict("records")
    logger.info("Downloading FASTA for %d assemblies (workers=%d)",
                len(rows), cfg["download"]["workers"])

    results: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=int(cfg["download"]["workers"])) as pool:
        pending = {pool.submit(download_assembly, r, cfg): r["assembly_accession"] for r in rows}
        for i, fut in enumerate(futures.as_completed(pending), start=1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(rows):
                ok = sum(1 for r in results if r["status"] in ("downloaded", "skipped"))
                logger.info("Progress %d/%d (ok=%d, failed=%d)", i, len(rows), ok, len(results) - ok)

    res = pd.DataFrame(results).sort_values("assembly_accession").reset_index(drop=True)
    status_path = meta_dir / "download_status.csv"
    res.to_csv(status_path, index=False)

    failed = res[res["status"] == "failed"]
    fail_path = path_for(cfg, "reports_dir") / "download_failures.csv"
    failed.to_csv(fail_path, index=False)

    n_ok = int((res["status"] != "failed").sum())
    n_dl = int((res["status"] == "downloaded").sum())
    n_skip = int((res["status"] == "skipped").sum())
    n_rpt = int((res["assembly_report_path"].astype(str).str.len() > 0).sum())
    total_bytes = int(res["bytes"].sum())

    slog.record("selected assemblies", len(rows), len(rows))
    slog.record("successfully obtained FASTA", len(rows), n_ok,
                f"{n_dl} downloaded, {n_skip} already present")
    slog.flush(cfg)

    logger.info("Downloaded=%d skipped=%d failed=%d assembly_reports=%d total=%.2f GB",
                n_dl, n_skip, len(failed), n_rpt, total_bytes / 1e9)
    if len(failed):
        logger.warning("Failed downloads listed in %s", fail_path)

    update_manifest(cfg, "step02_download_fasta", {
        "assemblies_requested": len(rows),
        "downloaded_this_run": n_dl,
        "already_present": n_skip,
        "failed": int(len(failed)),
        "assembly_reports_available": n_rpt,
        "total_fasta_bytes": total_bytes,
        "md5_verified": int(res["md5_verified"].astype(bool).sum()),
        "download_status_csv": str(status_path),
        "failures_csv": str(fail_path),
    })
    return res


if __name__ == "__main__":
    run()
