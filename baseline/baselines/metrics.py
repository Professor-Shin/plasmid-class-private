"""Evaluation of plasmid-vs-chromosome predictions (plasmid = positive class)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_plasmid": int((y_true == 1).sum()),
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "auroc": float(roc_auc_score(y_true, proba)),
        "auprc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, 501)))
    scores = []
    for t in grid:
        pred = proba >= t
        tp = float((pred & (y_true == 1)).sum())
        fp = float((pred & (y_true == 0)).sum())
        fn = float((~pred & (y_true == 1)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(grid[int(np.argmax(scores))])


def plasflow_threshold_rule(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """PlasFlow.py leaves a sequence unclassified when no class passes the threshold."""
    confident = np.maximum(proba, 1 - proba) >= threshold
    out = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_classified": int(confident.sum()),
        "unclassified_rate": float(1 - confident.mean()),
        "unclassified_rate_plasmid": float(1 - confident[y_true == 1].mean()),
        "unclassified_rate_chromosome": float(1 - confident[y_true == 0].mean()),
    }
    if confident.sum() and len(np.unique(y_true[confident])) > 1:
        sub = binary_metrics(y_true[confident], proba[confident], 0.5)
        out.update({f"classified_{k}": v for k, v in sub.items()})
    return out


def replicon_metrics(table: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Aggregate fragment probabilities to their parent replicon (mean probability)."""
    df = pd.DataFrame({"replicon_id": table["replicon_id"].to_numpy(),
                       "label": table["label"].to_numpy(), "proba": proba})
    agg = df.groupby("replicon_id").agg(label=("label", "first"), proba=("proba", "mean"))
    return binary_metrics(agg["label"].to_numpy(), agg["proba"].to_numpy(), threshold)
