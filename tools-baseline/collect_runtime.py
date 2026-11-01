"""What each baseline cost to run, per tool and per shard.

`sacct` accounts per JOB, and 15_tools_serial.sh deliberately puts all 116
shards of a tool inside one job (cpu40h rejects a 116-element array), so Slurm
can say what one tool cost in total but cannot separate a slow assembly from a
fast one, and cannot separate two tools that shared an allocation at all. This
assembles the per-tool table from the finest-grained source available for each:

  1. ``out/<tool>/<shard>/runtime.json`` - written by server/_timed_run.py around
     every tool invocation: wall clock, user+sys CPU seconds and peak RSS of the
     whole process tree. This is the good source, and the only one that carries
     CPU time and memory.

  2. ``server/logs/<tool>-serial-<nnn>.out`` - the two `banner` timestamps a
     per-shard runner prints. Wall clock only, and only to the second. This is
     the fallback for shards that were run before the timing wrapper existed:
     Platon and MOB-suite were already finished by then, and re-running them
     purely to collect timings would overwrite results the report is built on.

  3. ``<results>/zou_mixed.json`` - Zou is the one baseline that is trained
     rather than invoked per shard, so its cost is recorded by its own script.

A tool whose numbers come from source 2 is marked `wall_only` in the output, so
no one reads a missing CPU column as zero.

    python tools-baseline/collect_runtime.py
    python tools-baseline/collect_runtime.py --logs ../server/logs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolbase import cfg  # noqa: E402

# Tool directory under out/ -> the name used in the report.
TOOLS = {
    "platon": "platon",
    "mobsuite": "mob_recon",
    "plasme": "plasme",
    "plasmidhunter": "plasmidhunter",
}

# `    host=cuaim-fat-02 job=411700 task=none at 2026-10-28T01:02:03+07:00`
BANNER_TIME = re.compile(r"^\s+host=\S+ job=\S+ task=\S+ at (?P<ts>\S+)\s*$")


def from_runtime_json(tool_dir: Path) -> list[dict]:
    out = []
    for path in sorted(tool_dir.glob("*/runtime.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError) as exc:                        # noqa: BLE001
            print(f"  WARN: unreadable {path}: {exc}")
            continue
        rec["shard"] = rec.get("shard") or path.parent.name
        rec["source"] = "runtime.json"
        out.append(rec)
    return out


def from_shard_logs(logs: Path, tool: str) -> list[dict]:
    """Wall clock per shard, from the opening and closing banner timestamps.

    A shard that was skipped (its output already existed) prints one banner and
    exits, so a log with fewer than two timestamps is not a run and is left out
    rather than being counted as a zero-second one.
    """
    out = []
    for path in sorted(logs.glob(f"{tool}-serial-*.out")):
        stamps = [m.group("ts") for m in
                  (BANNER_TIME.match(line) for line in path.read_text(
                      encoding="utf-8", errors="replace").splitlines()) if m]
        if len(stamps) < 2:
            continue
        shard = ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if m := re.match(r"^=== \S+ \| shard (\S+)", line):
                shard = m.group(1)
                break
        try:
            t0 = datetime.fromisoformat(stamps[0])
            t1 = datetime.fromisoformat(stamps[-1])
        except ValueError:
            continue
        out.append({"tool": tool, "shard": shard or path.stem,
                    "wall_seconds": (t1 - t0).total_seconds(),
                    "started_at": stamps[0], "finished_at": stamps[-1],
                    "source": f"log:{path.name}"})
    return out


def summarise(tool: str, records: list[dict], manifest: pd.DataFrame) -> dict:
    df = pd.DataFrame(records).reset_index(drop=True)

    def column(name):
        """The record's own value where it has one, the manifest's otherwise.

        A runtime.json carries the shard's size because the runner passed it in;
        a log-derived record does not, and the manifest is then the only place
        to get it. Both paths must end with the same denominator or the per-Mbp
        columns would not be comparable between tools.
        """
        from_manifest = df["shard"].map(dict(zip(manifest["shard"], manifest[name])))
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(from_manifest)
        return pd.to_numeric(from_manifest, errors="coerce")

    size_bp = column("bp")
    size_frags = column("n_fragments") if "n_contigs" not in df.columns else (
        pd.to_numeric(df["n_contigs"], errors="coerce").fillna(
            df["shard"].map(dict(zip(manifest["shard"], manifest["n_fragments"])))))
    wall = pd.to_numeric(df.get("wall_seconds"), errors="coerce")
    cpu = pd.to_numeric(df["cpu_seconds"], errors="coerce") if "cpu_seconds" in df else None
    rss = pd.to_numeric(df["max_rss_kb"], errors="coerce") if "max_rss_kb" in df else None
    total_bp = float(size_bp.sum(skipna=True))
    has_cpu = cpu is not None and cpu.notna().any()

    row = {
        "tool": TOOLS.get(tool, tool),
        "shards": int(len(df)),
        "wall_only": not has_cpu,
        "wall_hours_total": round(float(wall.sum()) / 3600, 3),
        "wall_s_median": round(float(wall.median()), 1),
        "wall_s_max": round(float(wall.max()), 1),
        "slowest_shard": str(df.loc[wall.idxmax(), "shard"]) if wall.notna().any() else None,
        "cpu_hours_total": round(float(cpu.sum()) / 3600, 3) if has_cpu else None,
        # > 1 means the tool really did use the threads it was given. Platon and
        # mob_recon both shell out to multithreaded aligners; PlasmidHunter forks
        # one prodigal per split instead, and PLASMe's Transformer stage is
        # single-threaded whatever --thread says.
        "cpu_per_wall": round(float(cpu.sum() / wall.sum()), 2) if has_cpu and wall.sum() else None,
        "peak_rss_gib": round(float(rss.max()) / 1024 / 1024, 2) if rss is not None and rss.notna().any() else None,
        # The comparable numbers: everything above scales with how much of the
        # cluster the tool was handed, these do not.
        "cpu_s_per_Mbp": round(float(cpu.sum()) / (total_bp / 1e6), 2) if has_cpu and total_bp else None,
        "wall_s_per_Mbp": round(float(wall.sum()) / (total_bp / 1e6), 2) if total_bp else None,
        "Mbp_scored": round(total_bp / 1e6, 1),
        "fragments_scored": int(size_frags.sum(skipna=True)),
        "source": "runtime.json" if has_cpu else "shard logs (wall clock only)",
    }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--logs", default=None, help="directory holding the per-shard logs")
    args = ap.parse_args()

    conf = cfg.load_config(args.config)
    out_dir, results = Path(conf["out_dir"]), Path(conf["results_dir"])
    logs = Path(args.logs) if args.logs else HERE.parent / "server" / "logs"
    manifest_path = Path(conf["shard_dir"]) / "manifest.csv"
    manifest = (pd.read_csv(manifest_path) if manifest_path.exists()
                else pd.DataFrame(columns=["shard", "n_fragments", "bp"]))
    log = lambda m: print(m, flush=True)  # noqa: E731

    rows, detail = [], []
    for tool in TOOLS:
        records = from_runtime_json(out_dir / tool)
        source = "runtime.json"
        if not records and logs.is_dir():
            records = from_shard_logs(logs, tool)
            source = "shard logs"
        if not records:
            log(f"{tool:14s} no timing found (no runtime.json under "
                f"{(out_dir / tool)} and no {tool}-serial-*.out in {logs})")
            continue
        for r in records:
            r.setdefault("tool", tool)
        detail.extend(records)
        rows.append(summarise(tool, records, manifest))
        log(f"{tool:14s} {len(records):3d} shards from {source}")

    # Zou is trained, not invoked per shard, so its own script records the cost.
    zou_path = results / "zou_mixed.json"
    if zou_path.exists():
        rt = json.loads(zou_path.read_text()).get("runtime", {})
        if rt:
            bp = rt.get("bp_featurised") or 0
            rows.append({
                "tool": "zou (retrained)",
                "shards": None, "wall_only": False,
                "wall_hours_total": round(rt["wall_seconds"] / 3600, 3),
                "wall_s_median": None, "wall_s_max": None, "slowest_shard": None,
                # One process, so wall x threads is the honest allocation cost -
                # the same convention sacct's cpu_hours_alloc uses for the jobs.
                "cpu_hours_total": round(rt["wall_seconds"] * rt.get("torch_threads", 1) / 3600, 3),
                "cpu_per_wall": float(rt.get("torch_threads", 1)),
                "peak_rss_gib": round(rt["max_rss_kb"] / 1024 / 1024, 2),
                "cpu_s_per_Mbp": round(rt["wall_seconds"] * rt.get("torch_threads", 1)
                                       / (bp / 1e6), 2) if bp else None,
                "wall_s_per_Mbp": round(rt["wall_seconds"] / (bp / 1e6), 2) if bp else None,
                "Mbp_scored": round(bp / 1e6, 1),
                "fragments_scored": rt.get("n_fragments_featurised"),
                "source": "zou_mixed.json (train + featurise, all three splits)",
            })
            log(f"{'zou':14s} from {zou_path.name}")

    if not rows:
        raise SystemExit("no timing information found for any tool")

    summary = pd.DataFrame(rows)
    summary.to_csv(results / "tool_runtime.csv", index=False)
    (results / "tool_runtime.json").write_text(
        json.dumps({"per_tool": rows, "per_shard": detail}, indent=2, default=str),
        encoding="utf-8")
    log(f"\nwrote {results / 'tool_runtime.csv'}")
    log(f"wrote {results / 'tool_runtime.json'} ({len(detail)} per-shard records)")
    print()
    print(summary.to_string(index=False))
    print("\nwall_only=True means the record came from banner timestamps in the shard "
          "logs, which carry no CPU time and no memory - not that the tool used none.")
    print("cpu_s_per_Mbp is the comparable column: wall clock depends on how many "
          "shards were run at once, CPU seconds per megabase does not.")
    print("zou is trained rather than invoked per shard, and its figure covers "
          "featurising all three splits plus every regime and variant it trained, "
          "so it is not on the same footing as a per-shard inference cost.")


if __name__ == "__main__":
    main()
