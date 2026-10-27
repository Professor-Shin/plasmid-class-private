"""How much data went through the run, and what it cost to run it.

dump_tables.py renders what the models *scored*; nothing in it says how many
records were actually processed end to end, nor what the cluster spent doing it.
A thesis needs both: the dataset section needs ORF and amino-acid counts, not
just fragment counts, and the methods section needs wall clock, CPU-hours,
GPU-hours and peak memory per stage.

Three sources, none of which needs the run repeated:

  * the embedding cache (``emb_*.meta.npz``) - fragments, true ORF counts,
    stored rows, ORF-free fragments, bytes on disk. Authoritative, because it is
    what the model was actually trained on.
  * the shard logs in ``server/logs`` - ORF-calling seconds, embedding minutes
    and mean ORF length per shard, which is the only place the amino-acid total
    can be recovered from.
  * ``sacct`` - the accounting database, which keeps elapsed time, allocated
    TRES and peak RSS long after the job is gone. Job ids come from the log file
    names, so only this project's jobs are reported.

    python analysis/run_provenance.py                  # markdown + a json dump
    ESM_CACHE=cache_mixed python analysis/run_provenance.py --no-sacct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "esm2-attention" / os.environ.get("ESM_CACHE", "cache_server")
ESM = ROOT / "esm2-attention" / os.environ.get("ESM_RESULTS", "results_server")
LOGS = Path(os.environ.get("SERVER_LOGS", ROOT / "server" / "logs"))
ANA = ROOT / "analysis" / "results"
SPLITS = ("train", "validation", "test")

# "test.shard0003: 491234 ORFs from 6274 fragments in 83s | per fragment mean
#  78.30 median 74 max 412 | ORF aa mean 243 median 191 max 2734 | ..."
ORF_LINE = re.compile(
    r"^\s*(?P<tag>\S+?):\s+(?P<orfs>\d+) ORFs from (?P<frags>\d+) fragments in "
    r"(?P<secs>[\d.]+)s\b.*?ORF aa mean (?P<aa_mean>[\d.]+)")
EMB_LINE = re.compile(r"^\s*(?P<tag>\S+?):\s+embedded in (?P<minutes>[\d.]+) min")
# esm-embed-411706.out, tools-serial-411701.out - but not the per-shard logs
# (platon-serial-001.out, esm-embed-411706-test-shard0003.out). The leading digit
# cannot be a zero: Slurm ids are never zero-padded, the shard numbers always are.
JOB_LOG = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z][A-Za-z0-9_]*)*)"
                     r"-(?P<jid>[1-9]\d*)\.(?:out|err)$")


def human(n: float, unit="B") -> str:
    for suffix in ("", "Ki", "Mi", "Gi", "Ti"):
        if abs(n) < 1024:
            return f"{n:.1f} {suffix}{unit}"
        n /= 1024
    return f"{n:.1f} Pi{unit}"


def table(rows: list[dict], floatfmt="%.2f") -> str:
    if not rows:
        return "_(nothing to report)_"
    head = list(rows[0])
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        cells = []
        for h in head:
            v = r.get(h)
            if v is None:
                cells.append("-")
            elif isinstance(v, bool):
                cells.append(str(v))
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            elif isinstance(v, float):
                cells.append(floatfmt % v)
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------- records
def cache_files() -> list[tuple[str, Path]]:
    """(tag, metadata file) for both cache layouts.

    The server run writes the split .data.npy / .meta.npz pair; the laptop's
    cache_mixed is the older single emb_<tag>.npz, which holds the same metadata
    keys alongside the matrix. Reading both means this table can be produced for
    either run.
    """
    out = []
    for f in sorted(CACHE.glob("emb_*.npz")):
        if f.name.endswith(".meta.npz"):
            out.append((f.name[len("emb_"):-len(".meta.npz")], f))
        else:
            out.append((f.name[len("emb_"):-len(".npz")], f))
    return out


def scan_cache() -> tuple[list[dict], dict]:
    """Per-shard record counts straight out of the cache metadata."""
    shards = []
    for tag, meta in cache_files():
        with np.load(meta, allow_pickle=False) as z:
            real = z["n_orfs_real"].astype(np.int64) if "n_orfs_real" in z.files else None
            offsets = z["offsets"].astype(np.int64)
            lengths = z["length"].astype(np.int64) if "length" in z.files else None
            hidden = int(z["hidden"]) if "hidden" in z.files else None
            dtype = str(z["dtype"]) if "dtype" in z.files else None
            n_frag = len(z["fragment_id"])
        # the matrix lives beside the metadata in the split layout, and inside
        # the same file in the legacy one - counting the sibling unconditionally
        # would double the legacy cache's size
        extra = 0
        if meta.name.endswith(".meta.npz"):
            data_npy = meta.with_name(meta.name[:-len(".meta.npz")] + ".data.npy")
            extra = data_npy.stat().st_size if data_npy.exists() else 0
        shards.append({
            "shard": tag,
            "split": tag.split(".")[0],
            "fragments": int(n_frag),
            "orfs": int(real.sum()) if real is not None else None,
            "orf_free_fragments": int((real == 0).sum()) if real is not None else None,
            "embedding_rows": int(offsets[-1]),
            "bp": int(lengths.sum()) if lengths is not None else None,
            "hidden": hidden,
            "stored_dtype": dtype,
            "bytes": extra + meta.stat().st_size,
        })
    totals = {k: sum(s[k] or 0 for s in shards)
              for k in ("fragments", "orfs", "orf_free_fragments",
                        "embedding_rows", "bp", "bytes")}
    return shards, totals


def scan_shard_logs() -> dict[str, dict]:
    """ORF-calling and embedding time per shard, plus the mean ORF length.

    The amino-acid total - the number that actually sizes the transformer work -
    is nowhere in the cache: only the per-shard log kept the mean ORF length, so
    it is recovered as orfs x mean length and reported as an estimate.
    """
    per: dict[str, dict] = {}
    for log in sorted(LOGS.glob("esm-embed-*.out")):
        for line in log.read_text(errors="replace").splitlines():
            if m := ORF_LINE.match(line):
                e = per.setdefault(m["tag"], {})
                e["orf_seconds"] = float(m["secs"])
                e["orfs_logged"] = int(m["orfs"])
                e["orf_aa_mean"] = float(m["aa_mean"])
                e["log"] = log.name
            elif m := EMB_LINE.match(line):
                per.setdefault(m["tag"], {})["embed_minutes"] = float(m["minutes"])
    for e in per.values():
        if "orfs_logged" in e and "orf_aa_mean" in e:
            e["amino_acids_est"] = int(round(e["orfs_logged"] * e["orf_aa_mean"]))
    return per


# --------------------------------------------------------------- compute
def job_ids() -> dict[str, str]:
    """{job id: job name} for every Slurm job that left a log in server/logs."""
    found: dict[str, str] = {}
    if LOGS.is_dir():
        for f in LOGS.iterdir():
            if m := JOB_LOG.match(f.name):
                found.setdefault(m["jid"], m["name"])
    return found


SACCT_FIELDS = ("JobID,JobName,State,Partition,QOS,NNodes,Elapsed,TotalCPU,"
                "AllocTRES,MaxRSS,NodeList,ExitCode,Submit,Start,End")


def sacct(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    try:
        raw = subprocess.run(["sacct", "-j", ",".join(sorted(ids)),
                              "--format=" + SACCT_FIELDS,
                              "--parsable2", "--noheader"],
                             capture_output=True, text=True, timeout=120, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"_(sacct unavailable: {exc}; run this on the cluster to fill in "
              f"the compute table)_\n")
        return []
    cols = SACCT_FIELDS.split(",")
    return [dict(zip(cols, line.split("|"))) for line in raw.stdout.splitlines() if line]


def hms_to_hours(s: str) -> float | None:
    """Slurm writes D-HH:MM:SS, HH:MM:SS, MM:SS.mmm - all of them turn up."""
    if not s or s in ("", "INVALID"):
        return None
    days, _, rest = s.partition("-")
    if not rest:
        days, rest = "0", days
    parts = [float(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3:]
    return float(days) * 24 + h + m / 60 + sec / 3600


def tres(alloc: str) -> dict[str, str]:
    out = {}
    for item in (alloc or "").split(","):
        k, _, v = item.partition("=")
        if k:
            out[k] = v
    return out


def rss_bytes(s: str) -> int | None:
    if not s:
        return None
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(float(s) * 1024)       # sacct's bare numbers are KiB


def summarise_jobs(rows: list[dict], names: dict[str, str]) -> list[dict]:
    """One line per job: the parent row's time, the steps' peak memory."""
    parents = {r["JobID"]: r for r in rows if "." not in r["JobID"]}
    peak: dict[str, int] = {}
    for r in rows:
        base = r["JobID"].split(".")[0]
        if (b := rss_bytes(r.get("MaxRSS", ""))) is not None:
            peak[base] = max(peak.get(base, 0), b)
    out = []
    for jid, r in sorted(parents.items(), key=lambda kv: kv[0]):
        hours = hms_to_hours(r["Elapsed"]) or 0.0
        alloc = tres(r.get("AllocTRES", ""))
        cpus = int(alloc.get("cpu", 0) or 0)
        # AllocTRES carries both the generic gres/gpu=1 and the typed
        # gres/gpu:7g.40gb=1; counting every gres/gpu* key would double it.
        gpus = int(alloc.get("gres/gpu", 0) or 0)
        if not gpus:
            gpus = sum(int(v) for k, v in alloc.items()
                       if k.startswith("gres/gpu") and v.isdigit())
        out.append({
            "job": jid,
            "name": r["JobName"] or names.get(jid, ""),
            "state": r["State"].split(" ")[0],
            "partition": r.get("Partition", ""),
            "qos": r.get("QOS", ""),
            "node": r.get("NodeList", ""),
            "elapsed": r["Elapsed"],
            "cpus": cpus,
            "cpu_hours_alloc": round(cpus * hours, 2),
            "cpu_hours_used": round(hms_to_hours(r.get("TotalCPU", "")) or 0.0, 2),
            "gpus": gpus,
            "gpu_hours": round(gpus * hours, 2),
            "peak_rss": human(peak[jid]) if jid in peak else None,
            "exit": r.get("ExitCode", ""),
        })
    return out


# --------------------------------------------------------------- disk
DISK_TARGETS = (
    ("dataset (csv + fasta)", "dataset pipeline1/data/dataset"),
    ("per-tool shards", "server/shards"),
    ("embedding cache", f"esm2-attention/{CACHE.name}"),
    ("ESM-2 checkpoints", f"esm2-attention/{ESM.name}/checkpoints"),
    ("ESM-2 results", f"esm2-attention/{ESM.name}"),
    ("tool baselines", "baseline"),
    ("slurm logs", "server/logs"),
)


def disk() -> list[dict]:
    rows = []
    for label, rel in DISK_TARGETS:
        p = ROOT / rel
        if not p.exists():
            continue
        total = files = 0
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
                files += 1
        rows.append({"what": label, "path": rel, "files": files,
                     "size": human(total), "bytes": total})
    return rows


# --------------------------------------------------------------- render
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-sacct", action="store_true",
                    help="skip the Slurm accounting table")
    ap.add_argument("--json", default=str(ANA / "run_provenance.json"))
    args = ap.parse_args()
    render(no_sacct=args.no_sacct, json_out=Path(args.json))
    return 0


def render(no_sacct: bool = False, json_out: Path | None = None) -> dict:
    doc: dict = {"captured_at": datetime.now(timezone.utc).isoformat(),
                 "cache_dir": str(CACHE.relative_to(ROOT))}

    try:
        shards, totals = scan_cache()
    except Exception as exc:                                    # noqa: BLE001
        print(f"\n\n_(could not read {CACHE.name}: {exc!r})_")
        shards, totals = [], {}
    try:
        logs = scan_shard_logs()
    except Exception as exc:                                    # noqa: BLE001
        print(f"\n\n_(could not parse shard logs in {LOGS}: {exc!r})_")
        logs = {}
    for s in shards:
        s.update({k: v for k, v in logs.get(s["shard"], {}).items() if k != "log"})

    print("\n\n### Run provenance: records processed\n")
    if not shards:
        print(f"_(missing: esm2-attention/{CACHE.name} holds no emb_*.meta.npz yet)_")
    else:
        per_split = []
        for split in SPLITS:
            part = [s for s in shards if s["split"] == split]
            if not part:
                continue
            per_split.append({
                "split": split,
                "shards": len(part),
                "fragments": sum(s["fragments"] for s in part),
                "bp": sum(s["bp"] or 0 for s in part),
                "ORFs": sum(s["orfs"] or 0 for s in part),
                "ORF-free frags": sum(s["orf_free_fragments"] or 0 for s in part),
                "ORFs/fragment": (sum(s["orfs"] or 0 for s in part)
                                  / max(sum(s["fragments"] for s in part), 1)),
                "amino acids (est)": sum(s.get("amino_acids_est", 0) for s in part) or None,
                "embedding rows": sum(s["embedding_rows"] for s in part),
                "cache": human(sum(s["bytes"] for s in part)),
                "ORF calling (h)": sum(s.get("orf_seconds", 0) for s in part) / 3600 or None,
                "embedding (h)": sum(s.get("embed_minutes", 0) for s in part) / 60 or None,
            })
        grand = {"split": "**all**", "shards": len(shards)}
        for k in ("fragments", "bp", "ORFs", "ORF-free frags", "embedding rows"):
            grand[k] = sum(r[k] for r in per_split)
        grand["ORFs/fragment"] = grand["ORFs"] / max(grand["fragments"], 1)
        grand["amino acids (est)"] = sum(r["amino acids (est)"] or 0 for r in per_split) or None
        grand["cache"] = human(totals["bytes"])
        grand["ORF calling (h)"] = sum(r["ORF calling (h)"] or 0 for r in per_split) or None
        grand["embedding (h)"] = sum(r["embedding (h)"] or 0 for r in per_split) or None
        print(table(per_split + [grand]))
        hidden = {s["hidden"] for s in shards if s["hidden"]}
        dtypes = {s["stored_dtype"] for s in shards if s["stored_dtype"]}
        print(f"\nembeddings: {'/'.join(str(h) for h in sorted(hidden)) or '?'} dimensions, "
              f"stored as {'/'.join(sorted(dtypes)) or '?'}. "
              "`ORF-free frags` are fragments represented by a single zero vector, "
              "which is why `embedding rows` exceeds `ORFs`. "
              "`amino acids (est)` is ORFs x the mean ORF length recorded in the "
              "shard log - the cache keeps counts, not lengths.")
        doc["records"] = {"per_shard": shards, "totals": totals}

    print("\n\n### Run provenance: compute\n")
    names = job_ids()
    jobs = [] if no_sacct else summarise_jobs(sacct(list(names)), names)
    if jobs:
        print(table(jobs))
        cpu_h = sum(j["cpu_hours_alloc"] for j in jobs)
        gpu_h = sum(j["gpu_hours"] for j in jobs)
        used = sum(j["cpu_hours_used"] for j in jobs)
        print(f"\n**totals: {cpu_h:,.1f} CPU-hours allocated, {used:,.1f} actually "
              f"consumed ({used / max(cpu_h, 1e-9) * 100:.0f}% efficiency), "
              f"{gpu_h:,.1f} GPU-hours over {len(jobs)} jobs.** "
              "`cpu_hours_alloc` is cpus x elapsed, which is what the share of the "
              "cluster cost; `cpu_hours_used` is Slurm's TotalCPU. A job that waits "
              "on one GPU while holding 16 cores shows a low efficiency by design.")
        doc["compute"] = jobs
    else:
        print(f"_(no Slurm accounting; {len(names)} job id(s) found in "
              f"{LOGS.name}: {', '.join(sorted(names)) or 'none'})_")

    print("\n\n### Run provenance: disk footprint\n")
    rows = disk()
    print(table(rows))
    doc["disk"] = rows

    print("\n\n### Run provenance: software and hardware\n")
    envf = next((p for p in (ESM / "environment.json",
                             ROOT / "esm2-attention" / "results" / "environment.json")
                 if p.exists()), None)
    if envf is None:
        print("_(missing: no environment.json - run "
              "`python esm2-attention/capture_environment.py` on the machine that "
              "produced the results)_")
    else:
        env = json.loads(envf.read_text())
        hw, osx = env.get("hardware", {}), env.get("os", {})
        print(f"captured `{envf.relative_to(ROOT)}` at {env.get('captured_at', '?')}\n")
        print(table([{
            "host": f"{osx.get('system', '?')} {osx.get('version', '')}".strip(),
            "cpu": hw.get("cpu_model", "?"),
            "cores": f"{hw.get('physical_cores', '?')}p/{hw.get('logical_cores', '?')}l",
            "ram": human(hw.get("ram_bytes", 0)),
            "python": (env.get("python") or {}).get("version", "?"),
            "torch": (env.get("torch") or {}).get("version", "?"),
            "cuda": str((env.get("torch") or {}).get("cuda_available", "?")),
        }]))
        if pk := env.get("packages"):
            print("\npackages: " + ", ".join(f"`{k} {v}`" for k, v in pk.items()))
        doc["environment_file"] = str(envf.relative_to(ROOT))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(doc, indent=2, default=str))
        print(f"\n_(machine-readable copy: {json_out.relative_to(ROOT)})_")
    return doc


if __name__ == "__main__":
    raise SystemExit(main())
