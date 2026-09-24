"""Step 2: train the attention classifier on the cached ORF embeddings and evaluate it.

Training follows the prototype: AdamW(lr=1e-3, weight_decay=1e-4), BCEWithLogitsLoss,
one fragment per optimisation step, a fixed 15 epochs with no early stopping. Several
seeds are trained so the run-to-run spread is visible, and their mean probability is
reported as the ensemble.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import data, metrics, model as model_mod  # noqa: E402


def as_tensors(embeddings):
    return [torch.from_numpy(np.ascontiguousarray(embeddings[i]))
            for i in range(len(embeddings))]


def predict(net, tensors) -> np.ndarray:
    net.eval()
    out = np.empty(len(tensors), dtype=np.float64)
    with torch.no_grad():
        for i, x in enumerate(tensors):
            out[i] = torch.sigmoid(net(x)).item()
    return out


def train_one(cfg, train_x, train_y, val_x, val_set, seed, log):
    t = cfg["train"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    net = model_mod.EsmAttentionClassifier(cfg["esm"]["hidden_size"],
                                           t["classifier_hidden"], t["dropout"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=t["learning_rate"],
                                  weight_decay=t["weight_decay"])

    history, best, best_state, best_epoch = [], -np.inf, None, 0
    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        net.train()
        order = rng.permutation(len(train_x))
        total = 0.0
        for i in order:
            optimizer.zero_grad()
            logit = net(train_x[i]).reshape(1)
            loss = criterion(logit, train_y[i].reshape(1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_proba = predict(net, val_x)
        auprc = metrics.average_precision_score(val_set.label, val_proba,
                                                sample_weight=val_set.weight)
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "val_auprc_weighted": float(auprc),
                        "seconds": time.time() - t0})
        log(f"    epoch {epoch}/{t['epochs']} loss={total / len(order):.4f} "
            f"val_auprc={auprc:.4f} ({time.time() - t0:.0f}s)")
        if auprc > best:
            best, best_epoch = auprc, epoch
            best_state = copy.deepcopy(net.state_dict())
    return net, best_state, best_epoch, history


def evaluate(name, y, proba, weight, replicon_id, tuned_threshold):
    return {
        "weighted": metrics.binary_metrics(y, proba, 0.5, weight),
        "weighted_tuned": metrics.binary_metrics(y, proba, tuned_threshold, weight),
        "as_sampled": metrics.binary_metrics(y, proba, 0.5),
        "replicon_level": metrics.replicon_metrics(replicon_id, y, proba),
        "name": name,
    }


def main():
    cfg = data.load_config(HERE / "config.yaml")
    torch.set_num_threads(cfg["train"]["torch_threads"])
    cache, results = cfg["cache_dir"], cfg["results_dir"]
    (results / "probs").mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    sets = {s: data.EmbeddingSet(cache / f"emb_{s}.npz") for s in data.SPLITS}
    tensors = {s: as_tensors(sets[s]) for s in data.SPLITS}
    train_y = torch.tensor(sets["train"].label, dtype=torch.float32)
    for s, es in sets.items():
        log(f"{s}: {len(es)} fragments, {int(es.label.sum())} plasmid, "
            f"{es.n_orfs.mean():.2f} ORFs/fragment, "
            f"effective size {es.weight.sum():,.0f}")

    per_seed, final_probs, best_probs = {}, {"validation": [], "test": []}, \
        {"validation": [], "test": []}
    for i in range(cfg["train"]["n_seeds"]):
        seed = cfg["random_seed"] + 100 * i
        log(f"\n=== model {i + 1}/{cfg['train']['n_seeds']} (seed {seed}) ===")
        t0 = time.time()
        net, best_state, best_epoch, history = train_one(
            cfg, tensors["train"], train_y, tensors["validation"], sets["validation"],
            seed, log)
        train_seconds = time.time() - t0

        probs = {s: predict(net, tensors[s]) for s in ("validation", "test")}
        final_net_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
        best = {s: predict(net, tensors[s]) for s in ("validation", "test")}
        net.load_state_dict(final_net_state)

        for split in ("validation", "test"):
            final_probs[split].append(probs[split])
            best_probs[split].append(best[split])
            np.save(results / "probs" / f"esm2_seed{seed}_final_{split}.npy", probs[split])
            np.save(results / "probs" / f"esm2_seed{seed}_best_{split}.npy", best[split])

        tuned = metrics.best_f1_threshold(sets["validation"].label, probs["validation"],
                                          sets["validation"].weight)
        per_seed[f"seed{seed}"] = {
            "seed": seed, "best_epoch": best_epoch, "train_seconds": train_seconds,
            "history": history, "tuned_threshold": tuned,
            "test_final": evaluate("final", sets["test"].label, probs["test"],
                                   sets["test"].weight, sets["test"].replicon_id, tuned),
        }
        m = per_seed[f"seed{seed}"]["test_final"]["weighted"]
        log(f"  test (weighted): auprc={m['auprc']:.4f} f1={m['f1']:.4f} "
            f"recall={m['recall']:.4f} mcc={m['mcc']:.4f}")

    payload = {"method": "ESM-2 (esm2_t6_8M_UR50D) ORF embeddings + attention pooling",
               "config": {k: cfg[k] for k in ("esm", "orf", "subsample", "train")},
               "splits": {s: {"n_sampled": len(sets[s]),
                              "n_plasmid": int(sets[s].label.sum()),
                              "effective_n": float(sets[s].weight.sum()),
                              "orfs_per_fragment": float(sets[s].n_orfs.mean())}
                          for s in data.SPLITS},
               "per_seed": per_seed}

    for tag, store in (("final", final_probs), ("best_epoch", best_probs)):
        mean = {s: np.mean(store[s], axis=0) for s in store}
        np.save(results / "probs" / f"esm2_ensemble_{tag}_test.npy", mean["test"])
        np.save(results / "probs" / f"esm2_ensemble_{tag}_validation.npy", mean["validation"])
        tuned = metrics.best_f1_threshold(sets["validation"].label, mean["validation"],
                                          sets["validation"].weight)
        payload[f"ensemble_{tag}"] = {
            "tuned_threshold": tuned,
            **evaluate(tag, sets["test"].label, mean["test"], sets["test"].weight,
                       sets["test"].replicon_id, tuned),
        }
        m = payload[f"ensemble_{tag}"]["weighted"]
        log(f"ENSEMBLE ({tag}) test weighted: acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
            f"recall={m['recall']:.4f} precision={m['precision']:.4f} "
            f"auprc={m['auprc']:.4f} mcc={m['mcc']:.4f}")

    (results / "esm2_attention.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")
    log(f"wrote {results / 'esm2_attention.json'}")


if __name__ == "__main__":
    main()
