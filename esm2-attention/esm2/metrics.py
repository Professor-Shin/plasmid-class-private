"""Metrics that re-weight a stratified sample back to the full split (plasmid = positive)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)


def binary_metrics(y_true, proba, threshold=0.5, weight=None) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1],
                                      sample_weight=weight).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tn + fp + fn + tp
    return {
        "threshold": float(threshold),
        "n_sampled": int(len(y_true)),
        "n_effective": float(total),
        "n_plasmid_effective": float(tp + fn),
        "accuracy": float((tp + tn) / total),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, pred, sample_weight=weight))
        if len(np.unique(pred)) > 1 else 0.0,
        "auroc": float(roc_auc_score(y_true, proba, sample_weight=weight)),
        "auprc": float(average_precision_score(y_true, proba, sample_weight=weight)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def best_f1_threshold(y_true, proba, weight=None) -> float:
    proba = np.asarray(proba)
    grid = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, 501)))
    w = np.ones(len(y_true)) if weight is None else np.asarray(weight)
    y = np.asarray(y_true)
    best, best_t = -1.0, 0.5
    for t in grid:
        pred = proba >= t
        tp = w[pred & (y == 1)].sum()
        fp = w[pred & (y == 0)].sum()
        fn = w[~pred & (y == 1)].sum()
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if score > best:
            best, best_t = score, float(t)
    return best_t


def replicon_metrics(replicon_id, y_true, proba, threshold=0.5) -> dict:
    """Average a replicon's sampled fragment probabilities, then score replicons."""
    df = pd.DataFrame({"replicon_id": replicon_id, "label": y_true, "proba": proba})
    agg = df.groupby("replicon_id").agg(label=("label", "first"), proba=("proba", "mean"))
    return binary_metrics(agg["label"].to_numpy(), agg["proba"].to_numpy(), threshold)
