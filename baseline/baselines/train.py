"""Shared helpers for the two baseline training scripts."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .data import MemmapStream


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_threads(n: int | None):
    # Tiny batches lose more to OpenMP synchronisation than they gain from cores,
    # so the thread count is tuned per model rather than set to os.cpu_count().
    torch.set_num_threads(n or os.cpu_count() or 4)


def predict_proba(model, path, n_rows, n_cols, labels, transform, block_bytes,
                  subset=None, batch_size=512, binary=False) -> np.ndarray:
    stream = MemmapStream(path, n_rows, n_cols, labels, batch_size,
                          block_bytes=block_bytes, subset=subset, transform=transform)
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in stream.epoch(shuffle=False):
            logits = model(torch.from_numpy(xb))
            p = torch.sigmoid(logits) if binary else torch.softmax(logits, dim=1)[:, 1]
            out.append(p.numpy())
    return np.concatenate(out)


def save_json(path: Path, payload: dict):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
