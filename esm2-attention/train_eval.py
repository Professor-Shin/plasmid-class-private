"""Step 2: train the feature ladder on the cached ORF embeddings and evaluate it.

Every variant in ``variants:`` reuses the SAME cached ESM-2 embeddings and differs
only in which whole-fragment auxiliary columns the classifier head receives:

    esm2         ORF attention only - the original model, blind to non-coding DNA
    esm2+gc      + GC content and its local heterogeneity
    esm2+ori     + origin-of-replication composition proxy
    esm2+gc+ori  + both

Training follows the prototype: AdamW(lr=1e-3, weight_decay=1e-4),
BCEWithLogitsLoss, one fragment per optimisation step, a fixed number of epochs
with no early stopping. Several seeds are trained so the run-to-run spread is
visible, and their mean probability is reported as the ensemble.

Results are reported three ways, because the expert's comments are about three
different things: overall (weighted back to the split's true composition), per
fragment-length bucket (is the model length-robust?), and per replicon.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import composition, data, metrics, model as model_mod  # noqa: E402


def as_tensors(embeddings):
    return [torch.from_numpy(np.ascontiguousarray(embeddings[i]))
            for i in range(len(embeddings))]


def aux_tensors(sets, blocks):
    """Select the variant's auxiliary columns and standardise on train statistics.

    Returns (per-split list of tensors or None, dim, feature names).
    """
    cols = composition.block_columns(blocks)
    if len(cols) == 0:
        return {s: None for s in sets}, 0, []
    train = sets["train"].aux[:, cols].astype(np.float64)
    mean, std = train.mean(0), train.std(0)
    std[std == 0.0] = 1.0
    out = {}
    for split, es in sets.items():
        z = (es.aux[:, cols].astype(np.float64) - mean) / std
        out[split] = [torch.tensor(r, dtype=torch.float32) for r in z]
    names = [composition.all_feature_names()[c] for c in cols]
    return out, len(cols), names


def predict(net, tensors, aux) -> np.ndarray:
    net.eval()
    out = np.empty(len(tensors), dtype=np.float64)
    with torch.no_grad():
        for i, x in enumerate(tensors):
            logit = net(x) if aux is None else net(x, aux[i])
            out[i] = torch.sigmoid(logit).item()
    return out


def train_one(cfg, aux_dim, train_x, train_aux, train_y, val_x, val_aux, val_set,
              seed, log):
    t = cfg["train"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    net = model_mod.EsmAttentionClassifier(cfg["esm"]["hidden_size"],
                                           t["classifier_hidden"], t["dropout"],
                                           aux_dim=aux_dim)
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
            logit = (net(train_x[i]) if train_aux is None
                     else net(train_x[i], train_aux[i])).reshape(1)
            loss = criterion(logit, train_y[i].reshape(1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_proba = predict(net, val_x, val_aux)
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


def evaluate(name, es, proba, tuned_threshold):
    """Overall, length-stratified and replicon-level scores for one prediction set."""
    out = {
        "name": name,
        "weighted": metrics.binary_metrics(es.label, proba, 0.5, es.weight),
        "weighted_tuned": metrics.binary_metrics(es.label, proba, tuned_threshold, es.weight),
        "as_sampled": metrics.binary_metrics(es.label, proba, 0.5),
        "replicon_level": metrics.replicon_metrics(es.replicon_id, es.label, proba),
    }
    if es.length_bucket is not None:
        out["by_length"] = metrics.length_bucket_metrics(
            es.length_bucket, es.label, proba, es.weight, tuned_threshold).to_dict("records")
    # Fragments with no callable ORF are exactly where the ORF-only model is blind.
    if es.n_orfs_real is not None:
        free = es.n_orfs_real == 0
        out["orf_free"] = {"n": int(free.sum())}
        if free.sum() and len(np.unique(es.label[free])) > 1:
            out["orf_free"].update(
                metrics.binary_metrics(es.label[free], proba[free], 0.5, es.weight[free]))
    return out


def run_variant(cfg, sets, tensors, name, blocks, log):
    aux, aux_dim, aux_names = aux_tensors(sets, blocks)
    log(f"\n{'=' * 70}\nvariant {name}: aux_dim={aux_dim} "
        f"{'(' + ', '.join(aux_names) + ')' if aux_names else '(ORF embeddings only)'}")
    train_y = torch.tensor(sets["train"].label, dtype=torch.float32)

    # Two checkpoints are reported per seed. The model overfits (train loss keeps
    # falling while validation AUPRC plateaus), so the last epoch is not the best
    # epoch, and which one you report changes the ranking of the variants. "best"
    # is early stopping done after the fact: the epoch with the highest weighted
    # validation AUPRC.
    per_seed = {}
    store = {tag: {"validation": [], "test": []} for tag in ("final", "best")}
    for i in range(cfg["train"]["n_seeds"]):
        seed = cfg["random_seed"] + 100 * i
        log(f"  --- seed {seed} ({i + 1}/{cfg['train']['n_seeds']})")
        t0 = time.time()
        net, best_state, best_epoch, history = train_one(
            cfg, aux_dim, tensors["train"], aux["train"], train_y,
            tensors["validation"], aux["validation"], sets["validation"], seed, log)

        probs = {s: predict(net, tensors[s], aux[s]) for s in ("validation", "test")}
        final_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
        best_probs = {s: predict(net, tensors[s], aux[s]) for s in ("validation", "test")}
        net.load_state_dict(final_state)

        for split in ("validation", "test"):
            store["final"][split].append(probs[split])
            store["best"][split].append(best_probs[split])
        tuned = metrics.best_f1_threshold(sets["validation"].label, probs["validation"],
                                          sets["validation"].weight)
        per_seed[f"seed{seed}"] = {
            "seed": seed, "best_epoch": best_epoch,
            "train_seconds": round(time.time() - t0, 1),
            "history": history, "tuned_threshold": tuned,
            "test": evaluate("final", sets["test"], probs["test"], tuned),
        }
        m = per_seed[f"seed{seed}"]["test"]["weighted"]
        log(f"    test (weighted, final epoch): auprc={m['auprc']:.4f} "
            f"f1={m['f1']:.4f} mcc={m['mcc']:.4f} | best epoch was {best_epoch}")

    out = {"aux_dim": aux_dim, "aux_names": aux_names, "blocks": list(blocks),
           "per_seed": per_seed}
    means = {}
    for tag in ("final", "best"):
        mean = {s: np.mean(store[tag][s], axis=0) for s in store[tag]}
        means[tag] = mean
        tuned = metrics.best_f1_threshold(sets["validation"].label, mean["validation"],
                                          sets["validation"].weight)
        ens = evaluate(f"ensemble_{tag}", sets["test"], mean["test"], tuned)
        ens["tuned_threshold"] = tuned
        out[f"ensemble_{tag}"] = ens
        m, mt = ens["weighted"], ens["weighted_tuned"]
        log(f"  ENSEMBLE[{tag:5s}] test weighted: f1={m['f1']:.4f} mcc={m['mcc']:.4f} "
            f"auprc={m['auprc']:.4f} | tuned f1={mt['f1']:.4f} mcc={mt['mcc']:.4f} | "
            f"replicon mcc={ens['replicon_level']['mcc']:.4f}")
    out["ensemble"] = out["ensemble_best"]   # headline = early-stopped checkpoint
    return out, means


def main():
    cfg = data.load_config(HERE / "config.yaml")
    torch.set_num_threads(cfg["train"]["torch_threads"])
    cache, results = cfg["cache_dir"], cfg["results_dir"]
    (results / "probs").mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    sets = {s: data.EmbeddingSet(cache / f"emb_{s}.npz") for s in data.SPLITS}
    tensors = {s: as_tensors(sets[s]) for s in data.SPLITS}
    for s, es in sets.items():
        orf_free = int((es.n_orfs_real == 0).sum()) if es.n_orfs_real is not None else -1
        log(f"{s}: {len(es)} fragments, {int(es.label.sum())} plasmid, "
            f"{es.n_orfs.mean():.2f} ORFs/fragment, {orf_free} ORF-free, "
            f"effective size {es.weight.sum():,.0f}")

    payload = {"method": "ESM-2 (esm2_t6_8M_UR50D) ORF embeddings + attention pooling, "
                         "with optional whole-fragment GC / origin composition features",
               "dataset_dir": str(cfg["dataset_dir"]),
               "config": {k: cfg[k] for k in ("esm", "orf", "subsample", "train", "variants")},
               "splits": {s: {"n_sampled": len(sets[s]),
                              "n_plasmid": int(sets[s].label.sum()),
                              "effective_n": float(sets[s].weight.sum()),
                              "orfs_per_fragment": float(sets[s].n_orfs.mean())}
                          for s in data.SPLITS},
               "variants": {}}

    rows = []
    for name, blocks in cfg["variants"].items():
        entry, means = run_variant(cfg, sets, tensors, name, blocks, log)
        payload["variants"][name] = entry
        slug = name.replace("+", "_")
        for tag in ("final", "best"):
            np.save(results / "probs" / f"{slug}_ensemble_{tag}_test.npy", means[tag]["test"])
        # gc_signal.py reads this name; point it at the headline checkpoint
        np.save(results / "probs" / f"{slug}_ensemble_test.npy", means["best"]["test"])
        for tag in ("best", "final"):
            e = entry[f"ensemble_{tag}"]
            rows.append({"variant": name, "aux_dim": entry["aux_dim"], "checkpoint": tag,
                         "mcc": round(e["weighted"]["mcc"], 4),
                         "f1": round(e["weighted"]["f1"], 4),
                         "auprc": round(e["weighted"]["auprc"], 4),
                         "mcc_tuned": round(e["weighted_tuned"]["mcc"], 4),
                         "f1_tuned": round(e["weighted_tuned"]["f1"], 4),
                         "replicon_mcc": round(e["replicon_level"]["mcc"], 4)})

    (results / "esm2_variants.json").write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "esm2_variants.csv", index=False)
    log(f"\nwrote {results / 'esm2_variants.json'}")
    print("\n" + summary.to_string(index=False))

    # length-stratified table across variants, the answer to "5 kb is arbitrary"
    long_rows = []
    for name, entry in payload["variants"].items():
        for r in entry["ensemble_best"].get("by_length", []):
            long_rows.append({"variant": name, "length_bucket": r["length_bucket"],
                              "n_sampled": r.get("n_sampled"),
                              "mcc": round(r.get("mcc", float("nan")), 4),
                              "f1": round(r.get("f1", float("nan")), 4),
                              "auprc": round(r.get("auprc", float("nan")), 4)})
    if long_rows:
        by_len = pd.DataFrame(long_rows)
        by_len.to_csv(results / "esm2_by_length.csv", index=False)
        print("\nby length bucket (tuned threshold)")
        print(by_len.pivot(index="length_bucket", columns="variant",
                           values="mcc").to_string())


if __name__ == "__main__":
    main()
