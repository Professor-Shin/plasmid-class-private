"""Baseline 2: Zou et al. (PLoS ONE 2022) retrained on our dataset.

Seven-layer MLP (256-256-128-128-32-10-1) with dropout and L2 on standardised
canonical 6-mer counts (2080 features, KMC column order), Adam lr=1e-3,
batch 256, binary cross-entropy, early stopping on the validation split, and
the paper's three-vote ensemble over three independently trained models.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import data, features, metrics, models, train  # noqa: E402
from prepare_features import feature_path  # noqa: E402


def monitor_score(kind, y, proba):
    if kind == "accuracy":
        return float(((proba >= 0.5).astype(int) == y).mean())
    return float(metrics.average_precision_score(y, proba))


def train_one(cfg, n_cols, paths, labels, transform, subset, val_subset, seed, limits,
              monitor, log):
    z = cfg["zou"]
    train.set_seed(seed)
    stream = data.MemmapStream(paths["train"], len(labels["train"]), n_cols, labels["train"],
                               z["batch_size"], block_bytes=cfg["block_bytes"], subset=subset,
                               transform=transform, seed=seed)
    model = models.ZouMLP(n_cols, tuple(z["dropout"]))
    opt = model.optimizer(z["learning_rate"], z["l2"])
    criterion = nn.BCEWithLogitsLoss()

    y_val = labels["validation"] if val_subset is None else labels["validation"][val_subset]
    best, best_state, best_epoch, history = -np.inf, None, 0, []
    for epoch in range(1, limits["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in stream.epoch():
            opt.zero_grad()
            loss = criterion(model(torch.from_numpy(xb)), torch.from_numpy(yb).float())
            loss.backward()
            opt.step()
            losses.append(loss.item())
        proba = train.predict_proba(model, paths["validation"], len(labels["validation"]), n_cols,
                                    labels["validation"], transform, cfg["block_bytes"],
                                    subset=val_subset, binary=True)
        score = monitor_score(monitor, y_val, proba)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        f"val_{monitor}": score})
        log(f"    epoch {epoch}/{limits['max_epochs']} loss={np.mean(losses):.4f} "
            f"val_{monitor}={score:.4f} ({time.time() - t0:.0f}s)")
        if score > best + z["min_delta"]:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= limits["patience"]:
            log(f"    early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--max-epochs", type=int, help="override config (smoke tests)")
    args = ap.parse_args()

    cfg = data.load_config(HERE / "config.yaml")
    if args.max_epochs:
        for regime in cfg["zou"]["balanced"], cfg["zou"]["full"]:
            regime["max_epochs"] = args.max_epochs
    train.set_threads(cfg["zou"].get("torch_threads"))
    cache, results = cfg["cache_dir"], cfg["results_dir"]
    (results / "probs").mkdir(exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    z = cfg["zou"]
    n_cols = features.n_features(z["k"], z["canonical"])
    transform = features.StandardTransform.load(cache / "zou_scaler.npz")
    paths = {s: feature_path(cache, s, z["k"], z["canonical"]) for s in data.SPLITS}
    tables = {s: data.load_table(cfg["dataset_dir"], s) for s in data.SPLITS}
    labels = {s: np.load(cache / f"labels_{s}.npy") for s in data.SPLITS}

    for regime in (args.regimes or cfg["regimes"]):
        seed_rng = np.random.default_rng(cfg["random_seed"])
        if regime == "balanced":
            subset = data.balanced_subset(labels["train"], seed_rng)
            val_subset = data.balanced_subset(labels["validation"], seed_rng)
            monitor = "accuracy"
        else:
            subset, val_subset = None, None
            # Accuracy is degenerate at 2.3% prevalence, so model selection uses AUPRC.
            monitor = "auprc"
        limits = z[regime]
        n_train = len(subset) if subset is not None else len(labels["train"])
        log(f"\n=== Zou et al. | regime={regime} | train fragments={n_train} | "
            f"monitor=val_{monitor} | max_epochs={limits['max_epochs']} ===")

        probs = {"validation": [], "test": []}
        per_model = {}
        for i in range(z["n_models"]):
            seed = cfg["random_seed"] + 100 * i
            log(f"  model {i + 1}/{z['n_models']} (seed {seed})")
            t0 = time.time()
            model, history, best_epoch = train_one(cfg, n_cols, paths, labels, transform, subset,
                                                   val_subset, seed, limits, monitor, log)
            for split in ("validation", "test"):
                p = train.predict_proba(model, paths[split], len(labels[split]), n_cols,
                                        labels[split], transform, cfg["block_bytes"], binary=True)
                np.save(results / "probs" / f"zou_{regime}_m{i}_{split}.npy", p)
                probs[split].append(p)
            per_model[f"model{i}"] = {
                "seed": seed, "best_epoch": best_epoch, "train_seconds": time.time() - t0,
                "history": history,
                "test": metrics.binary_metrics(labels["test"], probs["test"][-1]),
            }
            m = per_model[f"model{i}"]["test"]
            log(f"    test: auprc={m['auprc']:.4f} f1={m['f1']:.4f} recall={m['recall']:.4f}")

        mean_prob = {s: np.mean(probs[s], axis=0) for s in probs}
        # Paper's three-vote rule: majority over the three binarised predictions.
        votes = {s: (np.mean([(p >= 0.5).astype(float) for p in probs[s]], axis=0) >= 0.5).astype(float)
                 for s in probs}
        for s in mean_prob:
            np.save(results / "probs" / f"zou_{regime}_mean_{s}.npy", mean_prob[s])
        tuned = metrics.best_f1_threshold(labels["validation"], mean_prob["validation"])
        payload = {
            "method": "Zou et al. 2022 (BV-BRC zou-plasmid-prediction), retrained",
            "regime": regime,
            "n_train_fragments": int(n_train),
            "monitor": f"val_{monitor}",
            "per_model": per_model,
            "ensemble": {
                "validation": metrics.binary_metrics(labels["validation"], mean_prob["validation"]),
                "test": metrics.binary_metrics(labels["test"], mean_prob["test"]),
                "test_three_vote": metrics.binary_metrics(labels["test"], votes["test"]),
                "test_tuned_threshold": metrics.binary_metrics(labels["test"], mean_prob["test"], tuned),
                "test_replicon_level": metrics.replicon_metrics(tables["test"], mean_prob["test"]),
            },
        }
        train.save_json(results / f"zou_{regime}.json", payload)
        t = payload["ensemble"]["test"]
        log(f"  ENSEMBLE test: acc={t['accuracy']:.4f} f1={t['f1']:.4f} recall={t['recall']:.4f} "
            f"precision={t['precision']:.4f} auprc={t['auprc']:.4f} mcc={t['mcc']:.4f}")


if __name__ == "__main__":
    main()
