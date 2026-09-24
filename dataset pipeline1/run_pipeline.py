"""Run the RefSeq plasmid-vs-chromosome dataset pipeline end to end.

    python run_pipeline.py                  # all steps
    python run_pipeline.py --steps 3 4 5    # selected steps
    python run_pipeline.py --config other.yaml
    python run_pipeline.py --from 5         # step 5 onwards

Every step is idempotent: already-downloaded FASTA is reused, and the
metadata each step reads is the file the previous step wrote, so a run can be
resumed after an interruption.
"""
from __future__ import annotations

import argparse
import sys
import time

from pipeline import (
    step01_filter_assemblies,
    step02_download_fasta,
    step03_parse_replicons,
    step04_qc,
    step05_cluster,
    step06_split,
    step07_fragments,
)
from pipeline.utils import get_logger, load_config

STEPS = {
    1: ("filter assemblies", step01_filter_assemblies.run),
    2: ("download genomic FASTA", step02_download_fasta.run),
    3: ("parse replicons", step03_parse_replicons.run),
    4: ("data QC", step04_qc.run),
    5: ("deduplicate / cluster", step05_cluster.run),
    6: ("split parent replicons", step06_split.run),
    7: ("generate 5-kb fragments", step07_fragments.run),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--steps", type=int, nargs="+", choices=sorted(STEPS),
                    help="run only these steps")
    ap.add_argument("--from", dest="from_step", type=int, choices=sorted(STEPS),
                    help="run this step and everything after it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger(cfg, "run_pipeline")

    if args.steps:
        chosen = sorted(set(args.steps))
    elif args.from_step:
        chosen = [s for s in sorted(STEPS) if s >= args.from_step]
    else:
        chosen = sorted(STEPS)

    logger.info("Running steps %s with config %s", chosen, cfg["_config_path"])
    for s in chosen:
        name, fn = STEPS[s]
        logger.info("=" * 70)
        logger.info("STEP %d - %s", s, name)
        logger.info("=" * 70)
        t0 = time.time()
        fn(cfg)
        logger.info("STEP %d finished in %.1f s", s, time.time() - t0)
    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
