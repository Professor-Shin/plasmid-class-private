"""Step 1: call ORFs and embed every ORF with ESM-2, one shard at a time.

Writes, per split (or per shard of a split):
    cache/subsample_{split}.csv     the exact fragments used, with their weights
    cache/emb_{tag}.data.npy        the ORF embedding matrix (memory-mappable)
    cache/emb_{tag}.meta.npz        offsets, labels, weights, auxiliary features

This is the expensive step and it runs ONCE: every variant of the classifier
(esm2, +gc, +ori, +gc+ori) reuses these embeddings and simply selects which
auxiliary columns it consumes, so changing the feature set costs no ESM time.

Three changes let it process the whole 112,929-fragment dataset with the ORF cap
removed, where the laptop could only manage a ~2,000-fragment stratified sample:

* **GPU.** `--device auto` uses CUDA when a GPU is allocated, with bfloat16
  autocast for the transformer and a float32 mean pool.
* **Parallel ORF calling.** With `max_orfs` unlimited the dataset yields 8.79 M
  ORFs (measured), so ORF calling runs in a worker pool that keeps the GPU fed.
  The translation itself is not the bottleneck - it costs about 1.3 CPU-hours in
  total, because Biopython's translate is C-backed - the 2.13 G amino acids of
  transformer work are.
* **Shards.** `--shard i --n-shards N` processes a contiguous block of the split
  and writes it separately; `merge_shards.py` concatenates them in row order, so
  the result does not depend on how many shards were used. A shard already on
  disk is skipped, which makes the embedding pass resumable - on the laptop an
  interrupted split had to start over from zero, and the test split alone took
  233 minutes.

Usage:
    python extract_features.py                                  # all splits, one shard each
    python extract_features.py --split train --shard 3 --n-shards 10
    python extract_features.py --config config_server.yaml --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import composition, data, embed, orfs  # noqa: E402

# Set once per worker process by _init_worker; the pool is created with a
# fork/spawn-safe initialiser rather than closing over the config.
_ORF_CFG: dict = {}


def _init_worker(orf_cfg: dict):
    global _ORF_CFG
    _ORF_CFG = orf_cfg
    # Each worker is pure Python string work; BLAS threads would only fight the
    # other workers for cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _orf_job(item):
    """One fragment: six-frame ORFs plus the whole-fragment composition features."""
    frag_id, seq = item
    found = orfs.extract_orfs(seq, _ORF_CFG["min_length"], _ORF_CFG["max_orfs"],
                              _ORF_CFG.get("max_orfs_hard_cap"))
    aux = composition.fragment_features(seq, composition.ALL_BLOCKS)
    return frag_id, found, aux


def shard_slice(n_rows: int, shard: int, n_shards: int) -> slice:
    """Contiguous block for one shard. Contiguous, so merging restores row order."""
    if not 0 <= shard < n_shards:
        raise SystemExit(f"--shard must be in [0, {n_shards}), got {shard}")
    edges = np.linspace(0, n_rows, n_shards + 1).astype(int)
    return slice(int(edges[shard]), int(edges[shard + 1]))


def build_shard(cfg, split, shard, n_shards, tokenizer, model, device, log):
    cache = cfg["cache_dir"]
    paths = data.embedding_paths(cache, split, None if n_shards == 1 else shard)
    if data.is_cached(cache, split, None if n_shards == 1 else shard):
        log(f"{paths['tag']}: cached, skipping")
        return

    table = data.load_table(cfg["dataset_dir"], split)
    wanted = (cfg.get("subsample") or {}).get(split) if cfg.get("subsample") else None
    picked = data.stratified_subsample(table, wanted, cfg["random_seed"])
    if shard == 0 or n_shards == 1:
        picked.to_csv(cache / f"subsample_{split}.csv", index=False)
    weights = picked.groupby("label")["weight"].first().to_dict()
    log(f"{split}: using {len(picked)} of {len(table)} fragments "
        f"({int((picked['label'] == 1).sum())} plasmid, "
        f"{int((picked['label'] == 0).sum())} chromosome); "
        f"weights plasmid={weights.get(1, 0):.4f} chromosome={weights.get(0, 0):.4f}"
        + ("  [FULL SPLIT]" if wanted is None else "  [stratified sample]"))

    block = picked.iloc[shard_slice(len(picked), shard, n_shards)].reset_index(drop=True)
    log(f"{paths['tag']}: shard {shard + 1}/{n_shards} -> rows "
        f"{shard_slice(len(picked), shard, n_shards)} ({len(block)} fragments, "
        f"{block['length'].sum() / 1e6:.1f} Mbp)")

    orf_cfg = dict(cfg["orf"])
    workers = int(cfg["esm"].get("orf_workers") or 0)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 2)

    t0 = time.time()
    wanted_ids = block["fragment_id"].tolist()
    stream = data.iter_fasta_subset(Path(cfg["dataset_dir"]) / f"{split}.fasta", wanted_ids)

    peptides: list[str] = []
    counts: list[int] = []
    aux_rows: list[np.ndarray] = []
    seen_ids: list[str] = []

    if workers == 1:
        results = (_orf_job(item) for item in stream)
        _init_worker(orf_cfg)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=(orf_cfg,))
        # imap, not imap_unordered: the row order of the shard has to survive,
        # because labels and auxiliary features are matched by position.
        results = pool.imap(_orf_job, stream, chunksize=4)
    log(f"{paths['tag']}: calling ORFs with {workers} worker(s), "
        f"max_orfs={orf_cfg['max_orfs']} "
        f"(hard cap {orf_cfg.get('max_orfs_hard_cap')})")

    try:
        for i, (frag_id, found, aux) in enumerate(results):
            seen_ids.append(frag_id)
            peptides.extend(found)
            counts.append(len(found))
            # Whole-fragment features: the only path by which non-coding sequence
            # (origin of replication included) reaches the classifier.
            aux_rows.append(aux)
            if (i + 1) % 5000 == 0:
                log(f"    {paths['tag']}: {i + 1}/{len(block)} fragments, "
                    f"{len(peptides)} ORFs ({time.time() - t0:.0f}s)")
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    if seen_ids != wanted_ids:
        raise ValueError(f"{paths['tag']}: ORF results came back in the wrong order; "
                         "the embedding rows would not match their labels")

    counts = np.array(counts, dtype=np.int64)
    aux = np.vstack(aux_rows).astype(np.float32)
    aux_names = composition.all_feature_names()
    lengths = np.array([len(p) for p in peptides]) if peptides else np.zeros(1)
    log(f"{paths['tag']}: {len(peptides)} ORFs from {len(block)} fragments in "
        f"{time.time() - t0:.0f}s | per fragment mean {len(peptides) / max(len(block), 1):.2f} "
        f"median {np.median(counts):.0f} max {counts.max() if len(counts) else 0} | "
        f"ORF aa mean {lengths.mean():.0f} median {np.median(lengths):.0f} "
        f"max {lengths.max():.0f} | fragments with no ORF: {int((counts == 0).sum())} | "
        f"auxiliary features {aux.shape}")

    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        now = time.time()
        if now - last[0] > 60 or done == total:
            last[0] = now
            rate = done / max(now - t0, 1e-9)
            log(f"    {paths['tag']}: {done}/{total} ORFs embedded "
                f"({rate:.0f}/s, eta {(total - done) / max(rate, 1e-9) / 60:.1f} min)")

    hidden = int(model.config.hidden_size)
    store_dtype = np.float16 if cfg["esm"].get("store_float16") else np.float32
    matrix = embed.embed_peptides(peptides, tokenizer, model, hidden,
                                  cfg["esm"]["max_length"], cfg["esm"]["batch_tokens"],
                                  progress, device=device,
                                  amp=bool(cfg["esm"].get("amp")),
                                  out_dtype=store_dtype)
    log(f"{paths['tag']}: embedded in {(time.time() - t0) / 60:.1f} min "
        f"({(time.time() - t0) / max(len(block), 1):.3f} s/fragment, "
        f"stored as {np.dtype(store_dtype).name})")

    # The prototype represents an ORF-free sequence with a single zero vector, so
    # every fragment occupies at least one row and the attention pooling always has
    # something to weight.
    stored_counts = np.maximum(counts, 1)
    total_rows = int(stored_counts.sum())
    out = np.zeros((total_rows, hidden), dtype=store_dtype)
    src = np.concatenate([[0], np.cumsum(counts)])
    dst = np.concatenate([[0], np.cumsum(stored_counts)])
    for i in range(len(block)):
        if counts[i]:
            out[dst[i]:dst[i + 1]] = matrix[src[i]:src[i + 1]]
    del matrix

    data.save_embeddings_mmap(paths, block, out, stored_counts, aux=aux,
                              aux_names=aux_names, n_orfs_real=counts)
    size = paths["data"].stat().st_size + paths["meta"].stat().st_size
    log(f"{paths['tag']}: wrote {paths['data'].name} + {paths['meta'].name} "
        f"({size / 2**20:.1f} MiB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--split", default=None, choices=list(data.SPLITS),
                    help="one split; default is all three in order")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=None,
                    help="default comes from esm.shards in the config, else 1")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    args = ap.parse_args()

    cfg = data.load_config(args.config)
    torch.set_num_threads(cfg["esm"]["torch_threads"])
    log = lambda m: print(m, flush=True)  # noqa: E731

    device = embed.pick_device(args.device or cfg["esm"].get("device", "auto"))
    log(f"device: {embed.describe_device(device)}")
    log(f"loading {cfg['esm']['model_name']}")
    tokenizer, model = embed.load_esm(cfg["esm"]["model_name"], device)
    hidden = int(model.config.hidden_size)
    want = cfg["esm"].get("hidden_size")
    if want and int(want) != hidden:
        raise SystemExit(
            f"config says esm.hidden_size={want} but {cfg['esm']['model_name']} has "
            f"hidden_size={hidden}. Fix the config (or remove the key - it is read "
            f"from the checkpoint).")
    log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, "
        f"hidden {hidden}, amp={bool(cfg['esm'].get('amp')) and device.startswith('cuda')}")

    splits = (args.split,) if args.split else data.SPLITS
    shards_cfg = cfg["esm"].get("shards") or {}
    t0 = time.time()
    for split in splits:
        if args.n_shards:
            # an explicit --n-shards means "do exactly this one shard", which is
            # how the Slurm script walks them one srun at a time
            n_shards, shards = args.n_shards, (args.shard,)
        else:
            # otherwise walk every shard the config asks for, skipping cached ones
            n_shards = int(shards_cfg.get(split, 1))
            shards = range(n_shards)
        for shard in shards:
            build_shard(cfg, split, shard, n_shards, tokenizer, model, device, log)
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
