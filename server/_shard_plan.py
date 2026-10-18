"""Print the embedding work plan as `split shard n_shards cached` lines.

20_esm_embed.sh loops over these and runs one srun per shard, so each shard gets
its own log and a job that hits the time limit can be resubmitted and pick up
exactly where it stopped. Reading the plan from the config here keeps the shell
script free of YAML parsing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ESM = Path(__file__).resolve().parent.parent / "esm2-attention"
sys.path.insert(0, str(ESM))
from esm2 import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ESM / "config_server.yaml"))
    ap.add_argument("--pending-only", action="store_true",
                    help="omit shards whose cache already exists")
    args = ap.parse_args()

    cfg = data.load_config(args.config)
    shards_cfg = cfg["esm"].get("shards") or {}
    for split in data.SPLITS:
        n = int(shards_cfg.get(split, 1))
        for shard in range(n):
            cached = data.is_cached(cfg["cache_dir"], split, None if n == 1 else shard)
            if args.pending_only and cached:
                continue
            print(f"{split} {shard} {n} {'cached' if cached else 'pending'}")


if __name__ == "__main__":
    main()
