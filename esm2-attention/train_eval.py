"""Step 2: train the feature ladder on the cached ORF embeddings and evaluate it.

Every variant in ``variants:`` reuses the SAME cached ESM-2 embeddings and differs
only in which whole-fragment auxiliary columns the classifier head receives:

    esm2         ORF attention only - the original model, blind to non-coding DNA
    esm2+gc      + GC content and its local heterogeneity
    esm2+ori     + origin-of-replication composition proxy
    esm2+gc+ori  + both

Results are reported three ways, because the expert's comments are about three
different things: overall (weighted back to the split's true composition), per
fragment-length bucket (is the model length-robust?), and per replicon.

Server additions, all opt-in through the config so that running the original
``config.yaml`` reproduces the original numbers exactly:

* **Mini-batching.** ``train.batch_size > 1`` pads a batch of fragments and masks
  the attention logits. ``batch_size: 1`` keeps the prototype's per-fragment path
  untouched, and verify_components.py asserts the two agree to 1e-5.
* **Regimes.** ``train.regimes`` runs the same ladder under more than one class
  balance. ``full`` trains on the natural 2.8% plasmid rate with
  ``pos_weight = n_neg/n_pos``, so the training distribution finally matches the
  evaluation distribution; ``balanced`` reproduces the 1:1 condition the laptop's
  subsample forced. The first regime listed is the headline.
* **Early stopping.** ``train.patience`` stops a seed once the weighted validation
  AUPRC has not improved for that many epochs, instead of always running a fixed
  budget and picking the best epoch after the fact.
* **Checkpoints.** ``train.save_checkpoints`` writes the best state of every seed
  to results_dir/checkpoints. Nothing in the laptop code called torch.save, so
  every trained head was thrown away when the process exited.
"""

from __future__ import annotations

import argparse
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
from esm2 import composition, data, embed, metrics, model as model_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #

def as_tensors(embeddings):
    """Materialise every fragment as a tensor (the original, small-cache path)."""
    return [torch.from_numpy(np.ascontiguousarray(embeddings[i]))
            for i in range(len(embeddings))]


def aux_tensors(sets, blocks):
    """Select the variant's auxiliary columns and standardise on train statistics.

    Returns (per-split matrix or None, dim, feature names).
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
        out[split] = torch.tensor(z, dtype=torch.float32)
    names = [composition.all_feature_names()[c] for c in cols]
    return out, len(cols), names


def make_batch(es, rows, aux, device):
    """Pad a set of fragments into (batch, max_orfs, hidden) plus a validity mask."""
    chunks = [np.asarray(es[i], dtype=np.float32) for i in rows]
    lengths = [len(c) for c in chunks]
    width = max(lengths)
    x = torch.zeros(len(rows), width, es.hidden_size, dtype=torch.float32)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for j, chunk in enumerate(chunks):
        x[j, :len(chunk)] = torch.from_numpy(chunk)
        mask[j, :len(chunk)] = True
    x, mask = x.to(device), mask.to(device)
    a = None if aux is None else aux[list(rows)].to(device)
    return x, mask, a


def batched_order(es, order, batch_size, rng, bucket=32):
    """Group the epoch's fragments into padded batches, bucketing by ORF count.

    Fragments in this dataset carry between 1 and a few hundred ORFs, so random
    batches would be mostly padding. Sorting within a window of `bucket` batches
    and then shuffling the batch order keeps the sampling effectively random while
    cutting the padded volume by an order of magnitude.
    """
    n_orfs = es.n_orfs
    batches = []
    window = batch_size * bucket
    for start in range(0, len(order), window):
        chunk = order[start:start + window]
        chunk = chunk[np.argsort(n_orfs[chunk], kind="stable")]
        for i in range(0, len(chunk), batch_size):
            batches.append(chunk[i:i + batch_size])
    rng.shuffle(batches)
    return batches


def predict(net, es, aux, device, batch_size=256) -> np.ndarray:
    net.eval()
    out = np.empty(len(es), dtype=np.float64)
    order = np.argsort(es.n_orfs, kind="stable")      # minimal padding, any order
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            rows = order[i:i + batch_size]
            x, mask, a = make_batch(es, rows, aux, device)
            logits = net(x, a, mask)
            out[rows] = torch.sigmoid(logits).reshape(-1).cpu().numpy()
    return out


def predict_legacy(net, tensors, aux, device) -> np.ndarray:
    """One fragment at a time, exactly as the prototype evaluated."""
    net.eval()
    out = np.empty(len(tensors), dtype=np.float64)
    with torch.no_grad():
        for i, x in enumerate(tensors):
            a = None if aux is None else aux[i]
            logit = net(x.to(device)) if a is None else net(x.to(device), a.to(device))
            out[i] = torch.sigmoid(logit).item()
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def regime_rows(es, regime, cfg, seed) -> np.ndarray:
    """Which training rows a regime uses, and nothing about the loss."""
    if cfg["train"].get("class_balance", {}).get(regime) != "resample":
        return np.arange(len(es))
    # The laptop's implicit condition: an equal number of each class. Downsampled
    # once per seed rather than per epoch, so a seed sees a fixed training set.
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(es.label == 1)
    neg = np.flatnonzero(es.label == 0)
    keep = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    rows = np.concatenate([pos, keep])
    rows.sort()
    return rows


def make_criterion(es, rows, regime, cfg, device):
    """BCE, with pos_weight when the regime trains on the natural imbalance."""
    if cfg["train"].get("class_balance", {}).get(regime) != "pos_weight":
        return nn.BCEWithLogitsLoss(), None
    labels = es.label[rows]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        return nn.BCEWithLogitsLoss(), None
    # This is what lets training keep the evaluation distribution instead of
    # resampling to 1:1: the minority class is reweighted in the loss rather than
    # over-represented in the data.
    w = float(n_neg) / float(n_pos)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=device)), w


def train_one(cfg, aux_dim, sets, aux, regime, seed, device, log, tensors=None):
    t = cfg["train"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_es, val_es = sets["train"], sets["validation"]
    rows = regime_rows(train_es, regime, cfg, seed)
    criterion, pos_weight = make_criterion(train_es, rows, regime, cfg, device)
    batch_size = int(t.get("batch_size", 1))
    patience = t.get("patience")

    net = model_mod.EsmAttentionClassifier(train_es.hidden_size,
                                           t["classifier_hidden"], t["dropout"],
                                           aux_dim=aux_dim).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=t["learning_rate"],
                                  weight_decay=t["weight_decay"])
    y_all = torch.tensor(train_es.label, dtype=torch.float32, device=device)

    log(f"    rows {len(rows):,}/{len(train_es):,} "
        f"({int((train_es.label[rows] == 1).sum()):,} plasmid, "
        f"{(train_es.label[rows] == 1).mean():.2%}) | batch {batch_size} | "
        f"pos_weight {pos_weight if pos_weight else '-'} | "
        f"patience {patience if patience else '-'}")

    history, best, best_state, best_epoch = [], -np.inf, None, 0
    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        net.train()
        order = rows[rng.permutation(len(rows))]
        total, n_steps = 0.0, 0
        if batch_size == 1:
            # The prototype's path, kept bit-for-bit so config.yaml reproduces.
            for i in order:
                optimizer.zero_grad()
                x = tensors["train"][i].to(device)
                a = None if aux["train"] is None else aux["train"][i].to(device)
                logit = (net(x) if a is None else net(x, a)).reshape(1)
                loss = criterion(logit, y_all[i].reshape(1))
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_steps += 1
        else:
            for batch in batched_order(train_es, order, batch_size, rng):
                optimizer.zero_grad()
                x, mask, a = make_batch(train_es, batch, aux["train"], device)
                logits = net(x, a, mask).reshape(-1)
                loss = criterion(logits, y_all[batch])
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_steps += 1

        if batch_size == 1:
            val_proba = predict_legacy(net, tensors["validation"], aux["validation"], device)
        else:
            val_proba = predict(net, val_es, aux["validation"], device)
        auprc = metrics.average_precision_score(val_es.label, val_proba,
                                                sample_weight=val_es.weight)
        history.append({"epoch": epoch, "train_loss": total / max(n_steps, 1),
                        "val_auprc_weighted": float(auprc),
                        "seconds": time.time() - t0})
        log(f"    epoch {epoch}/{t['epochs']} loss={total / max(n_steps, 1):.4f} "
            f"val_auprc={auprc:.4f} ({time.time() - t0:.0f}s)")
        if auprc > best:
            best, best_epoch = auprc, epoch
            best_state = copy.deepcopy(net.state_dict())
        elif patience and epoch - best_epoch >= int(patience):
            log(f"    early stop: no improvement for {patience} epochs "
                f"(best was epoch {best_epoch}, auprc {best:.4f})")
            break
    return net, best_state, best_epoch, history


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

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


def run_variant(cfg, sets, name, blocks, regime, device, log, tensors=None):
    aux, aux_dim, aux_names = aux_tensors(sets, blocks)
    log(f"\n{'=' * 70}\nvariant {name} [regime {regime}]: aux_dim={aux_dim} "
        f"{'(' + ', '.join(aux_names) + ')' if aux_names else '(ORF embeddings only)'}")
    batch_size = int(cfg["train"].get("batch_size", 1))

    # Two checkpoints are reported per seed. The model overfits (train loss keeps
    # falling while validation AUPRC plateaus), so the last epoch is not the best
    # epoch, and which one you report changes the ranking of the variants. "best"
    # is the epoch with the highest weighted validation AUPRC.
    per_seed = {}
    store = {tag: {"validation": [], "test": []} for tag in ("final", "best")}
    ckpt_dir = Path(cfg["results_dir"]) / "checkpoints"
    for i in range(cfg["train"]["n_seeds"]):
        seed = cfg["random_seed"] + 100 * i
        log(f"  --- seed {seed} ({i + 1}/{cfg['train']['n_seeds']})")
        t0 = time.time()
        net, best_state, best_epoch, history = train_one(
            cfg, aux_dim, sets, aux, regime, seed, device, log, tensors)

        def infer(split):
            if batch_size == 1:
                return predict_legacy(net, tensors[split], aux[split], device)
            return predict(net, sets[split], aux[split], device)

        probs = {s: infer(s) for s in ("validation", "test")}
        final_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
        best_probs = {s: infer(s) for s in ("validation", "test")}
        net.load_state_dict(final_state)

        if cfg["train"].get("save_checkpoints"):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            slug = name.replace("+", "_")
            torch.save({"state_dict": best_state, "variant": name, "regime": regime,
                        "seed": seed, "best_epoch": best_epoch, "aux_dim": aux_dim,
                        "aux_names": aux_names, "hidden_size": sets["train"].hidden_size,
                        "config": {k: cfg[k] for k in ("esm", "orf", "train")}},
                       ckpt_dir / f"{regime}_{slug}_seed{seed}.pt")

        for split in ("validation", "test"):
            store["final"][split].append(probs[split])
            store["best"][split].append(best_probs[split])
        tuned = metrics.best_f1_threshold(sets["validation"].label, probs["validation"],
                                          sets["validation"].weight)
        per_seed[f"seed{seed}"] = {
            "seed": seed, "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_seconds": round(time.time() - t0, 1),
            "history": history, "tuned_threshold": tuned,
            "test": evaluate("final", sets["test"], probs["test"], tuned),
        }
        m = per_seed[f"seed{seed}"]["test"]["weighted"]
        log(f"    test (weighted, final epoch): auprc={m['auprc']:.4f} "
            f"f1={m['f1']:.4f} mcc={m['mcc']:.4f} | best epoch was {best_epoch}")

    out = {"aux_dim": aux_dim, "aux_names": aux_names, "blocks": list(blocks),
           "regime": regime, "per_seed": per_seed}
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    args = ap.parse_args()

    cfg = data.load_config(args.config)
    torch.set_num_threads(cfg["train"]["torch_threads"])
    cache, results = cfg["cache_dir"], cfg["results_dir"]
    (results / "probs").mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    device = embed.pick_device(args.device or cfg["train"].get("device", "cpu"))
    log(f"device: {embed.describe_device(device)}")

    sets = {s: data.open_split(cache, s) for s in data.SPLITS}
    for s, es in sets.items():
        orf_free = int((es.n_orfs_real == 0).sum()) if es.n_orfs_real is not None else -1
        log(f"{s}: {len(es):,} fragments, {int(es.label.sum()):,} plasmid "
            f"({es.label.mean():.2%}), {es.n_orfs.mean():.2f} ORFs/fragment, "
            f"{orf_free} ORF-free, effective size {es.weight.sum():,.0f}"
            f"{' [mmap]' if es.mmapped else ''}")
    hidden = sets["train"].hidden_size
    if any(sets[s].hidden_size != hidden for s in sets):
        raise SystemExit("the three splits were embedded with different models")

    batch_size = int(cfg["train"].get("batch_size", 1))
    tensors = None
    if batch_size == 1:
        # Only the per-fragment path needs every fragment resident; it is the
        # small-cache configuration by definition.
        log("batch_size=1: materialising all fragments (the prototype's path)")
        tensors = {s: as_tensors(sets[s]) for s in data.SPLITS}

    regimes = list(cfg["train"].get("regimes") or ["default"])
    headline = regimes[0]
    payload = {"method": f"ESM-2 ({cfg['esm']['model_name']}) ORF embeddings + attention "
                         "pooling, with optional whole-fragment GC / origin "
                         "composition features",
               "dataset_dir": str(cfg["dataset_dir"]),
               "device": device,
               "hidden_size": hidden,
               "regimes": regimes,
               "headline_regime": headline,
               "config": {k: cfg[k] for k in ("esm", "orf", "subsample", "train",
                                              "variants")},
               "splits": {s: {"n_used": len(sets[s]),
                              "n_plasmid": int(sets[s].label.sum()),
                              "plasmid_rate": float(sets[s].label.mean()),
                              "effective_n": float(sets[s].weight.sum()),
                              "orfs_per_fragment": float(sets[s].n_orfs.mean())}
                          for s in data.SPLITS},
               "variants": {}}

    rows = []
    for name, blocks in cfg["variants"].items():
        slug = name.replace("+", "_")
        entry = {"regimes": {}}
        for regime in regimes:
            sub, means = run_variant(cfg, sets, name, blocks, regime, device, log, tensors)
            entry["regimes"][regime] = sub
            for tag in ("final", "best"):
                np.save(results / "probs" / f"{slug}_{regime}_ensemble_{tag}_test.npy",
                        means[tag]["test"])
            if regime == headline:
                # The headline regime also keeps the flat, un-nested shape and the
                # plain file name, because analysis/gc_signal.py and
                # analysis/dump_tables.py read those.
                entry.update(sub)
                for tag in ("final", "best"):
                    np.save(results / "probs" / f"{slug}_ensemble_{tag}_test.npy",
                            means[tag]["test"])
                np.save(results / "probs" / f"{slug}_ensemble_test.npy",
                        means["best"]["test"])
            for tag in ("best", "final"):
                e = sub[f"ensemble_{tag}"]
                rows.append({"variant": name, "regime": regime,
                             "aux_dim": sub["aux_dim"], "checkpoint": tag,
                             "mcc": round(e["weighted"]["mcc"], 4),
                             "f1": round(e["weighted"]["f1"], 4),
                             "auprc": round(e["weighted"]["auprc"], 4),
                             "mcc_tuned": round(e["weighted_tuned"]["mcc"], 4),
                             "f1_tuned": round(e["weighted_tuned"]["f1"], 4),
                             "replicon_mcc": round(e["replicon_level"]["mcc"], 4)})
        payload["variants"][name] = entry

    (results / "esm2_variants.json").write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "esm2_variants.csv", index=False)
    log(f"\nwrote {results / 'esm2_variants.json'}")
    print("\n" + summary.to_string(index=False))

    # length-stratified table across variants, the answer to "5 kb is arbitrary"
    long_rows = []
    for name, entry in payload["variants"].items():
        for regime, sub in entry["regimes"].items():
            for r in sub["ensemble_best"].get("by_length", []):
                long_rows.append({"variant": name, "regime": regime,
                                  "length_bucket": r["length_bucket"],
                                  "n_sampled": r.get("n_sampled"),
                                  "mcc": round(r.get("mcc", float("nan")), 4),
                                  "f1": round(r.get("f1", float("nan")), 4),
                                  "auprc": round(r.get("auprc", float("nan")), 4)})
    if long_rows:
        by_len = pd.DataFrame(long_rows)
        by_len.to_csv(results / "esm2_by_length.csv", index=False)
        print(f"\nby length bucket (tuned threshold, regime {headline})")
        print(by_len[by_len["regime"] == headline]
              .pivot(index="length_bucket", columns="variant", values="mcc")
              .to_string())


if __name__ == "__main__":
    main()
