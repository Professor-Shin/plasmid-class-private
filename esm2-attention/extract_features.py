"""Step 1: subsample each split, call ORFs, and embed every ORF with ESM-2.

Writes cache/subsample_{split}.csv (the exact fragments used, with their weights)
and cache/emb_{split}.npz (ragged per-fragment ORF embeddings plus the
whole-fragment auxiliary features).

This is the expensive step: about 0.44 s per 5-kb fragment on a 4-core CPU, and
more on the mixed-length dataset because the ten longest ORFs of a long fragment
are themselves longer. It runs ONCE: every variant of the classifier
(esm2, +gc, +ori, +gc+ori) reuses these embeddings and simply selects which
auxiliary columns it consumes, so changing the feature set costs no ESM time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import composition, data, embed, orfs  # noqa: E402


def build_split(cfg, split, tokenizer, model, log):
    cache = cfg["cache_dir"]
    out = cache / f"emb_{split}.npz"
    if out.exists():
        log(f"{split}: cached, skipping")
        return

    table = data.load_table(cfg["dataset_dir"], split)
    picked = data.stratified_subsample(table, cfg["subsample"][split], cfg["random_seed"])
    picked.to_csv(cache / f"subsample_{split}.csv", index=False)
    weights = picked.groupby("label")["weight"].first().to_dict()
    log(f"{split}: sampled {len(picked)} of {len(table)} fragments "
        f"({int((picked['label'] == 1).sum())} plasmid, "
        f"{int((picked['label'] == 0).sum())} chromosome); "
        f"weights plasmid={weights.get(1, 0):.4f} chromosome={weights.get(0, 0):.4f}")

    t0 = time.time()
    seqs = data.read_fasta_subset(Path(cfg["dataset_dir"]) / f"{split}.fasta",
                                  picked["fragment_id"])
    log(f"{split}: read FASTA subset in {time.time() - t0:.1f}s")

    t0 = time.time()
    peptides, counts, aux_rows = [], [], []
    for frag_id in picked["fragment_id"]:
        seq = seqs[frag_id]
        found = orfs.extract_orfs(seq, cfg["orf"]["min_length"], cfg["orf"]["max_orfs"])
        peptides.extend(found)
        counts.append(len(found))
        # Whole-fragment features: the only path by which non-coding sequence
        # (origin of replication included) reaches the classifier.
        aux_rows.append(composition.fragment_features(seq, composition.ALL_BLOCKS))
    counts = np.array(counts, dtype=np.int64)
    aux = np.vstack(aux_rows).astype(np.float32)
    aux_names = composition.all_feature_names()
    log(f"{split}: auxiliary features {aux.shape} "
        f"({len(aux_names)} cols: {', '.join(aux_names)})")
    bounds = np.concatenate([[0], np.cumsum(counts)])
    empty = int((counts == 0).sum())
    lengths = np.array([len(p) for p in peptides])
    log(f"{split}: {len(peptides)} ORFs from {len(picked)} fragments in "
        f"{time.time() - t0:.1f}s | per fragment mean "
        f"{len(peptides) / len(picked):.2f} | ORF aa mean {lengths.mean():.0f} "
        f"median {np.median(lengths):.0f} max {lengths.max()} | "
        f"fragments with no ORF: {empty}")

    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        now = time.time()
        if now - last[0] > 60 or done == total:
            last[0] = now
            rate = done / max(now - t0, 1e-9)
            log(f"    {split}: {done}/{total} ORFs embedded "
                f"({rate:.0f}/s, eta {(total - done) / max(rate, 1e-9) / 60:.1f} min)")

    matrix = embed.embed_peptides(peptides, tokenizer, model, cfg["esm"]["hidden_size"],
                                  cfg["esm"]["max_length"], cfg["esm"]["batch_tokens"],
                                  progress)
    log(f"{split}: embedded in {(time.time() - t0) / 60:.1f} min "
        f"({(time.time() - t0) / len(picked):.3f} s/fragment)")

    hidden = cfg["esm"]["hidden_size"]
    zeros = np.zeros((1, hidden), dtype=np.float32)
    # The prototype represents an ORF-free sequence with a single zero vector.
    grouped = [matrix[bounds[i]:bounds[i + 1]] if counts[i] else zeros
               for i in range(len(picked))]
    data.save_embeddings(out, picked, grouped, aux=aux, aux_names=aux_names,
                         n_orfs_real=counts)
    log(f"{split}: wrote {out.name} ({out.stat().st_size / 2**20:.1f} MiB)")


def main():
    cfg = data.load_config(HERE / "config.yaml")
    torch.set_num_threads(cfg["esm"]["torch_threads"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    log(f"loading {cfg['esm']['model_name']}")
    tokenizer, model = embed.load_esm(cfg["esm"]["model_name"])
    log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, "
        f"hidden {model.config.hidden_size}, threads {torch.get_num_threads()}")

    t0 = time.time()
    for split in data.SPLITS:
        build_split(cfg, split, tokenizer, model, log)
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
