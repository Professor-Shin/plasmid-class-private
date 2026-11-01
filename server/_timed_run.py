#!/usr/bin/env python3
"""Run one command and write what it cost next to its output.

    python server/_timed_run.py --json OUT/runtime.json --tool plasme \
        --shard GCF_000013865.1 --threads 8 --contigs 146 --bp 3112044 \
        -- plasme-command --with --its --own --flags

Why a wrapper instead of reading Slurm's accounting: `sacct` reports one row per
JOB, and the serial runner puts all 116 shards of a tool inside one job, so
sacct can say what Platon cost in total but not what it cost per assembly, and
it cannot separate two tools that share an allocation at all. This records the
same three quantities - wall clock, CPU seconds, peak resident memory - per
invocation, which is the granularity the per-tool comparison needs.

`resource.getrusage(RUSAGE_CHILDREN)` is used rather than parsing
`/usr/bin/time`: GNU time is not guaranteed present on a compute node, its
output format differs between releases, and some builds report peak RSS in
pages while claiming kilobytes. getrusage is the same syscall underneath,
returns a struct, and covers the whole process tree the child spawns - which
matters here because every one of these tools shells out to prodigal, BLAST,
DIAMOND or mash to do the real work.

The JSON it writes is never the gate on success; the runner still checks the
tool's own output file. A failed run still gets a runtime record, with its
non-zero exit code, so a crash that took two hours is visible in the report
rather than missing from it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="where to write the runtime record")
    ap.add_argument("--tool", required=True)
    ap.add_argument("--shard", default="")
    ap.add_argument("--threads", type=int, default=0,
                    help="threads the tool was asked for, so CPU seconds can be read "
                         "against the parallelism it was given")
    ap.add_argument("--contigs", type=int, default=0)
    ap.add_argument("--bp", type=int, default=0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("FATAL: no command given after --", file=sys.stderr)
        return 2

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    rc = subprocess.call(cmd)
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)

    # utime/stime accumulate across every child this process has reaped, so the
    # delta is this command's own. ru_maxrss is a maximum, not a total, so it is
    # taken as-is - with one child, that maximum IS this command's peak. Linux
    # reports it in kibibytes.
    user = after.ru_utime - before.ru_utime
    system = after.ru_stime - before.ru_stime
    record = {
        "tool": args.tool,
        "shard": args.shard,
        "command": " ".join(shlex.quote(c) for c in cmd),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall, 3),
        "user_seconds": round(user, 3),
        "sys_seconds": round(system, 3),
        "cpu_seconds": round(user + system, 3),
        # > 1 means the tool really did use several cores; ~1 means it was
        # serial whatever --threads said, which is worth knowing before
        # comparing two tools' wall clocks.
        "cpu_efficiency": round((user + system) / wall, 3) if wall > 0 else None,
        "max_rss_kb": int(after.ru_maxrss),
        "threads_requested": args.threads or None,
        "n_contigs": args.contigs or None,
        "bp": args.bp or None,
        "bp_per_cpu_second": (round(args.bp / (user + system), 1)
                              if args.bp and (user + system) > 0 else None),
        "exit_code": rc,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  timed| {args.tool} {args.shard}: wall {wall:,.1f}s "
          f"cpu {user + system:,.1f}s peak {after.ru_maxrss / 1024:,.0f} MiB "
          f"exit {rc}  -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
