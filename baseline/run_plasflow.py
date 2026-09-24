"""Baseline 1: PlasFlow (Krawczyk et al., NAR 2018) retrained on our dataset.

Three DNNClassifiers (5-mer [20,20], 6-mer [20,20], 7-mer [30], ReLU) on TF-IDF
transformed oligonucleotide counts, Adagrad lr=0.05, batch 32, 50,000 steps,
combined by averaging their class probabilities.
"""

from __future__ import annotations

import argparse
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


def train_one(cfg, hidden, n_cols, paths, labels, transform, subset, eval_rows, seed, log):
    pf = cfg["plasflow"]
    train.set_seed(seed)
    stream = data.MemmapStream(paths["train"], len(labels["train"]), n_cols, labels["train"],
                               pf["batch_size"], block_bytes=cfg["block_bytes"], subset=subset,
                               transform=transform, seed=seed)
    model = models.PlasFlowDNN(n_cols, hidden)
    opt = model.optimizer(pf["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    history, running, t0 = [], [], time.time()
    batches = stream.forever()
    model.train()
    for step in range(1, pf["max_steps"] + 1):
        xb, yb = next(batches)
        opt.zero_grad()
        loss = criterion(model(torch.from_numpy(xb)), torch.from_numpy(yb))
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % pf["eval_every"] == 0:
            proba = train.predict_proba(model, paths["validation"], len(labels["validation"]),
                                        n_cols, labels["validation"], transform,
                                        cfg["block_bytes"], subset=eval_rows)
            y = labels["validation"][eval_rows]
            auprc = metrics.average_precision_score(y, proba) if y.sum() else float("nan")
            history.append({"step": step, "train_loss": float(np.mean(running)),
                            "val_auprc": float(auprc)})
            log(f"    step {step}/{pf['max_steps']} loss={np.mean(running):.4f} "
                f"val_auprc={auprc:.4f} ({time.time() - t0:.0f}s)")
            running = []
            model.train()
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, help="override config (smoke tests)")
    args = ap.parse_args()

    cfg = data.load_config(HERE / "config.yaml")
    if args.max_steps:
        cfg["plasflow"]["max_steps"] = args.max_steps
        cfg["plasflow"]["eval_every"] = max(1, args.max_steps // 2)
    train.set_threads(cfg["plasflow"].get("torch_threads"))
    cache, results = cfg["cache_dir"], cfg["results_dir"]
    (results / "probs").mkdir(exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    tables = {s: data.load_table(cfg["dataset_dir"], s) for s in data.SPLITS}
    labels = {s: np.load(cache / f"labels_{s}.npy") for s in data.SPLITS}
    rng = np.random.default_rng(cfg["random_seed"])

    for regime in (args.regimes or cfg["regimes"]):
        if regime == "balanced":
            subset = data.balanced_subset(labels["train"], np.random.default_rng(cfg["random_seed"]))
            eval_rows = data.balanced_subset(labels["validation"],
                                             np.random.default_rng(cfg["random_seed"]))
        else:
            subset = None
            eval_rows = np.sort(rng.choice(len(labels["validation"]),
                                           size=cfg["plasflow"]["eval_subset"], replace=False))
        n_train = len(subset) if subset is not None else len(labels["train"])
        log(f"\n=== PlasFlow | regime={regime} | train fragments={n_train} "
            f"({int(labels['train'][subset].sum()) if subset is not None else int(labels['train'].sum())} plasmid) ===")

        probs = {"validation": [], "test": []}
        per_model, histories = {}, {}
        for spec in cfg["plasflow"]["models"]:
            k, hidden = spec["k"], list(spec["hidden"])
            n_cols = features.n_features(k, False)
            transform = features.TfidfTransform.load(cache / f"tfidf_k{k}.npy")
            paths = {s: feature_path(cache, s, k, False) for s in data.SPLITS}
            log(f"  model k={k} hidden={hidden} features={n_cols}")
            t0 = time.time()
            model, history = train_one(cfg, hidden, n_cols, paths, labels, transform,
                                       subset, eval_rows, cfg["random_seed"] + k, log)
            histories[f"k{k}"] = history
            for split in ("validation", "test"):
                p = train.predict_proba(model, paths[split], len(labels[split]), n_cols,
                                        labels[split], transform, cfg["block_bytes"])
                np.save(results / "probs" / f"plasflow_{regime}_k{k}_{split}.npy", p)
                probs[split].append(p)
            per_model[f"k{k}"] = {
                "hidden": hidden, "n_features": n_cols, "train_seconds": time.time() - t0,
                "test": metrics.binary_metrics(labels["test"], probs["test"][-1]),
            }
            log(f"    k={k} test: auprc={per_model[f'k{k}']['test']['auprc']:.4f} "
                f"f1={per_model[f'k{k}']['test']['f1']:.4f} "
                f"recall={per_model[f'k{k}']['test']['recall']:.4f}")

        vote = {s: np.mean(probs[s], axis=0) for s in probs}
        # TF_Vote_Classifier.predict is a majority vote over the members' argmax,
        # while its reported probabilities are the average - both are kept here.
        majority = (np.mean([(p >= 0.5).astype(float) for p in probs["test"]], axis=0) >= 0.5)
        for split in vote:
            np.save(results / "probs" / f"plasflow_{regime}_vote_{split}.npy", vote[split])
        tuned = metrics.best_f1_threshold(labels["validation"], vote["validation"])
        payload = {
            "method": "PlasFlow (Krawczyk et al. 2018), retrained",
            "regime": regime,
            "n_train_fragments": int(n_train),
            "per_model": per_model,
            "history": histories,
            "ensemble": {
                "validation": metrics.binary_metrics(labels["validation"], vote["validation"]),
                "test": metrics.binary_metrics(labels["test"], vote["test"]),
                "test_majority_vote": metrics.binary_metrics(labels["test"], majority.astype(float)),
                "test_tuned_threshold": metrics.binary_metrics(labels["test"], vote["test"], tuned),
                "test_threshold_rule": metrics.plasflow_threshold_rule(
                    labels["test"], vote["test"], cfg["plasflow"]["threshold"]),
                "test_replicon_level": metrics.replicon_metrics(tables["test"], vote["test"]),
            },
        }
        train.save_json(results / f"plasflow_{regime}.json", payload)
        t = payload["ensemble"]["test"]
        log(f"  VOTE test: acc={t['accuracy']:.4f} f1={t['f1']:.4f} recall={t['recall']:.4f} "
            f"precision={t['precision']:.4f} auprc={t['auprc']:.4f} mcc={t['mcc']:.4f}")


if __name__ == "__main__":
    main()
