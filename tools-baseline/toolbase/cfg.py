"""Config loading and the import path shared by the tools-baseline scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent          # tools-baseline/
ROOT = HERE.parent                                     # repository root

PATH_KEYS = ("dataset_dir", "results_dir", "shard_dir", "out_dir")


def load_config(path: str | Path = None) -> dict:
    """Resolve every path key relative to the config file, as the other configs do.

    `baseline/baselines/data.py` and `esm2-attention/esm2/data.py` both resolve
    their relative paths against their own config file's directory, which is what
    keeps the tree portable despite the space in "dataset pipeline1". This follows
    the same convention so a server checkout needs no path edits.
    """
    path = Path(path or HERE / "config.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent
    for key in PATH_KEYS:
        cfg[key] = (root / cfg[key]).resolve()
    cfg["shard_dir"].mkdir(parents=True, exist_ok=True)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def use_baseline_metrics():
    """Import the metric functions the retrained baselines were scored with.

    The point of reusing `baseline/baselines/metrics.py` verbatim rather than
    writing new metric code is that Platon and MOB-suite then land in the same
    table as PlasFlow, Zou, PlasClass and the composition models with no risk of
    a different definition of MCC, a different confusion-matrix orientation or a
    different replicon aggregation creeping in.
    """
    baseline = str(ROOT / "baseline")
    if baseline not in sys.path:
        sys.path.insert(0, baseline)
    from baselines import data, metrics  # noqa: E402
    return data, metrics
