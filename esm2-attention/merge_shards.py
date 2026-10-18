"""Concatenate per-shard embedding caches into one cache per split.

The shards written by extract_features.py are contiguous blocks of the split in
row order, so merging is a concatenation in shard order and the merged cache is
identical whatever number of shards produced it. That property is asserted here
rather than assumed: the merged `row_in_split` must come out strictly increasing,
which is only true if no block was skipped, duplicated or reordered.

    python merge_shards.py --config config_server.yaml
    python merge_shards.py --config config_server.yaml --split train --keep-shards

By default the shard files are deleted once the merged cache is written and
verified, because they are an exact duplicate of it and project storage is the
binding constraint (see section 4 of the plan).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import data  # noqa: E402


def merge_split(cfg, split, n_shards, keep_shards, log) -> bool:
    cache = cfg["cache_dir"]
    target = data.embedding_paths(cache, split)
    if data.is_cached(cache, split):
        log(f"{split}: merged cache already present, skipping")
        return True

    shards = [data.embedding_paths(cache, split, i) for i in range(n_shards)]
    missing = [p["tag"] for p in shards if not (p["data"].exists() and p["meta"].exists())]
    if missing:
        log(f"{split}: NOT merged - {len(missing)} shard(s) missing: {missing}")
        log(f"  resubmit the embedding job; it skips the shards already on disk")
        return False

    t0 = time.time()
    metas = [np.load(p["meta"], allow_pickle=False) for p in shards]
    rows = sum(int(m["offsets"][-1]) for m in metas)
    hidden = int(metas[0]["hidden"])
    dtype = np.dtype(str(metas[0]["dtype"]))
    for i, m in enumerate(metas):
        if int(m["hidden"]) != hidden or np.dtype(str(m["dtype"])) != dtype:
            raise SystemExit(
                f"{split}: shard {i} holds {m['hidden']}-d {m['dtype']} embeddings but "
                f"shard 0 holds {hidden}-d {dtype}. The shards were produced by "
                "different configs; delete the cache and re-embed.")
    log(f"{split}: merging {n_shards} shards -> {rows:,} ORF rows x {hidden} "
        f"({dtype.name}, {rows * hidden * dtype.itemsize / 2**30:.2f} GiB)")

    # Written through a memmap so the merged matrix never has to fit in RAM twice.
    out = np.lib.format.open_memmap(target["data"], mode="w+", dtype=dtype,
                                    shape=(rows, hidden))
    at = 0
    counts, meta_parts = [], {k: [] for k in data.META_KEYS if k != "offsets"}
    for i, (p, m) in enumerate(zip(shards, metas)):
        block = np.load(p["data"], mmap_mode="r")
        out[at:at + len(block)] = block
        at += len(block)
        counts.append(np.diff(m["offsets"]))
        for key in list(meta_parts):
            if key in m.files:
                meta_parts[key].append(m[key])
            else:
                meta_parts.pop(key, None)
        del block
        log(f"  shard {i}: {int(m['offsets'][-1]):,} rows, "
            f"{len(m['label']):,} fragments")
    out.flush()
    del out
    if at != rows:
        raise SystemExit(f"{split}: wrote {at} rows, expected {rows}")

    merged = {key: np.concatenate(parts) for key, parts in meta_parts.items()}
    merged["offsets"] = np.concatenate([[0], np.cumsum(np.concatenate(counts))])
    merged["dtype"] = np.asarray(dtype.name)
    merged["hidden"] = np.asarray(hidden)

    # The ordering guarantee, checked rather than trusted.
    order = merged["row_in_split"]
    if not np.all(np.diff(order) > 0):
        raise SystemExit(
            f"{split}: merged row_in_split is not strictly increasing, so the shards "
            "were not contiguous blocks in row order. The merged cache would not "
            "match the split's labels - delete the shards and re-embed.")
    if len(merged["label"]) != len(merged["offsets"]) - 1:
        raise SystemExit(f"{split}: {len(merged['label'])} fragments but "
                         f"{len(merged['offsets']) - 1} offset intervals")

    np.savez(target["meta"], **merged)
    log(f"{split}: wrote {target['data'].name} + {target['meta'].name} in "
        f"{time.time() - t0:.0f}s ({len(merged['label']):,} fragments)")

    # np.load on an .npz returns a lazily-read archive that holds the file open.
    # Those handles have to be released before the shard files can be removed -
    # on Linux the unlink would succeed anyway and leak the descriptors, on
    # Windows it fails outright.
    for m in metas:
        m.close()
    metas.clear()

    es = data.open_split(cache, split)
    log(f"{split}: verified - {len(es):,} fragments, {es.n_orfs.sum():,} ORF rows, "
        f"hidden {es.hidden_size}, {int(es.label.sum()):,} plasmid "
        f"({es.label.mean():.2%}), effective size {es.weight.sum():,.0f}")
    es.close()

    if not keep_shards:
        for p in shards:
            p["data"].unlink(missing_ok=True)
            p["meta"].unlink(missing_ok=True)
        log(f"{split}: removed {n_shards} shard files (exact duplicates of the merge)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--split", default=None, choices=list(data.SPLITS))
    ap.add_argument("--keep-shards", action="store_true",
                    help="do not delete the per-shard files after merging")
    args = ap.parse_args()

    cfg = data.load_config(args.config)
    log = lambda m: print(m, flush=True)  # noqa: E731
    shards_cfg = cfg["esm"].get("shards") or {}

    ok = True
    for split in ((args.split,) if args.split else data.SPLITS):
        n_shards = int(shards_cfg.get(split, 1))
        if n_shards == 1:
            log(f"{split}: single-shard config, nothing to merge")
            continue
        ok &= merge_split(cfg, split, n_shards, args.keep_shards, log)
    if not ok:
        raise SystemExit("one or more splits could not be merged (see above)")


if __name__ == "__main__":
    main()
