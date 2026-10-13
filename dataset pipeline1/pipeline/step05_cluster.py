"""Step 5 - redundancy analysis over PARENT REPLICONS (before splitting).

Why this exists
---------------
Splitting fragments at random leaks information: two fragments from the same
replicon - or from two near-identical replicons of closely related strains -
would end up in train and test simultaneously and inflate the measured
accuracy. So redundancy is resolved here, at the parent-replicon level, and
the split in step 6 operates on the clusters produced here.

Method
------
MinHash (Mash-style) sketches over canonical k-mers, no external binaries:

  * each replicon is reduced to the ``sketch_size`` smallest 64-bit hashes of
    its canonical k-mers (k = ``kmer_size``);
  * the Jaccard index between two bottom-k sketches estimates the k-mer
    Jaccard index J of the full sequences;
  * the Mash distance D = -(1/k) * ln(2J / (1+J)) estimates ANI as 1 - D.

Candidate pairs come from an inverted index on sketch hashes, so only
replicons sharing at least one sketched k-mer are ever compared.

Clustering criterion (A CHOSEN EXPERIMENTAL PARAMETER, NOT A UNIVERSAL
BIOLOGICAL TRUTH): two replicons are linked when

    estimated ANI >= similarity.ani_threshold          (default 0.95)
    AND min(len)/max(len) >= similarity.min_length_ratio (default 0.90)

The length ratio stands in for alignment coverage, which a sketch cannot
measure directly. Single-linkage union-find turns the links into clusters.
Exact duplicates (identical SHA-256) are always merged regardless of the
thresholds.

Chromosomes and plasmids are compared against each other as well. Plasmid and
chromosomal sequence are NOT always dissimilar: horizontal gene transfer,
mobile elements, integrated prophage, and AMR cassettes produce genuine shared
sequence, and any cross-type link found here is reported rather than assumed
away.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (StageLog, get_logger, load_config, min_fragment_length, path_for,
                    read_fasta, report_tag, update_manifest)

MASK64 = (1 << 64) - 1
COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _hash64(value: int) -> int:
    """SplitMix64 finalizer - fast, well-distributed, and deterministic."""
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return value ^ (value >> 31)


_CODE = np.full(256, 255, dtype=np.uint8)
for i, base in enumerate("ACGT"):
    _CODE[ord(base)] = i
    _CODE[ord(base.lower())] = i


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finalizer over a uint64 array (wrap-around is intended)."""
    with np.errstate(over="ignore"):
        v = values.astype(np.uint64, copy=True)
        v ^= v >> np.uint64(30)
        v *= np.uint64(0xBF58476D1CE4E5B9)
        v ^= v >> np.uint64(27)
        v *= np.uint64(0x94D049BB133111EB)
        v ^= v >> np.uint64(31)
    return v


def _window_values(base: np.ndarray, k: int, forward: bool) -> np.ndarray:
    """2-bit-packed values of every length-k window of ``base``.

    Built by doubling - A(a+b)[i] = A(a)[i]*4**b + A(b)[i+a] for the forward
    strand, A(a+b)[i] = A(a)[i] + 4**a * A(b)[i+a] for the reverse-complement
    strand - which costs O(n log k) vector operations instead of the O(n*k) of
    a correlation, and never allocates a float64 copy of the genome.
    """
    n = base.size
    powers: dict[int, np.ndarray] = {1: base}
    m = 1
    while m * 2 <= k:
        a = powers[m]
        L = n - 2 * m + 1
        if L <= 0:
            break
        if forward:
            powers[2 * m] = a[:L] * np.uint64(4 ** m) + a[m : m + L]
        else:
            powers[2 * m] = a[:L] + a[m : m + L] * np.uint64(4 ** m)
        m *= 2

    acc: np.ndarray | None = None
    acc_len = 0
    for bit in sorted(powers, reverse=True):
        if not (bit & k):
            continue
        if acc is None:
            acc, acc_len = powers[bit], bit
            continue
        L = n - (acc_len + bit) + 1
        if L <= 0:
            return np.empty(0, dtype=np.uint64)
        if forward:
            acc = acc[:L] * np.uint64(4 ** bit) + powers[bit][acc_len : acc_len + L]
        else:
            acc = acc[:L] + powers[bit][acc_len : acc_len + L] * np.uint64(4 ** acc_len)
        acc_len += bit
    if acc is None:
        return np.empty(0, dtype=np.uint64)
    return acc[: n - k + 1]


def sketch(sequence: str, k: int, size: int, chunk: int = 4_000_000) -> np.ndarray:
    """Bottom-k MinHash sketch over canonical k-mers, as a sorted uint64 array.

    The sequence is processed in overlapping chunks and only the running
    bottom-``size`` hashes are kept, so peak memory stays flat no matter how
    long the replicon is (some bacterial chromosomes exceed 13 Mbp).
    """
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    n = raw.size
    if n < k:
        return np.empty(0, dtype=np.uint64)

    best = np.empty(0, dtype=np.uint64)
    step = max(chunk, k)
    for start in range(0, max(n - k + 1, 1), step):
        stop = min(start + step + k - 1, n)
        codes = _CODE[raw[start:stop]]
        if codes.size < k:
            break
        invalid = codes == 255
        base = np.where(invalid, 0, codes).astype(np.uint64)

        fwd = _window_values(base, k, forward=True)
        rev = _window_values(np.uint64(3) - base, k, forward=False)
        if fwd.size == 0:
            continue

        # Drop every window containing a non-ACGT base.
        if invalid.any():
            bad = np.cumsum(np.concatenate(([0], invalid.astype(np.int64))))
            keep = (bad[k:] - bad[:-k]) == 0
            keep = keep[: fwd.size]
            fwd, rev = fwd[keep], rev[keep]
            if fwd.size == 0:
                continue

        hashed = _splitmix64(np.minimum(fwd, rev))
        del fwd, rev, base, codes, invalid
        # Only the smallest `size` distinct hashes matter, so pre-select a
        # small candidate set in O(n) instead of sorting millions of values.
        # Falls back to the exact full sort in the (practically impossible)
        # case that the candidates do not contain `size` distinct hashes.
        if hashed.size > 4 * size:
            cand = np.unique(np.partition(hashed, 4 * size)[: 4 * size])
            hashed = cand if cand.size >= size else np.unique(hashed)
        else:
            hashed = np.unique(hashed)
        merged = np.union1d(best, hashed[:size])
        best = merged[:size] if merged.size > size else merged
    return best


def sketch_jaccard(a: np.ndarray, b: np.ndarray, size: int) -> float:
    """Jaccard estimate from two bottom-k sketches (merge-and-truncate)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    m = min(size, a.size, b.size)
    union = np.union1d(a, b)[:m]
    shared = np.intersect1d(union, np.intersect1d(a, b, assume_unique=True),
                            assume_unique=True).size
    return shared / m if m else 0.0


def sketch_containment(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Estimated containment of the smaller sketch in the larger.

    Returns ``(estimate, sample_size)``. Jaccard collapses when two sequences
    differ greatly in length, so a short replicon that is essentially a
    subsequence of a long one scores low; containment does not. Both sketches
    are truncated to the hash range they both cover before the fraction is
    taken.

    ``sample_size`` is how many hashes that shared range actually contains,
    and it matters: for very unequal sequences (a 10-kb plasmid against a
    5-Mb chromosome) the shared range can hold a handful of hashes, and a
    1-of-1 hit would read as containment 1.0. Callers must require a
    reasonable sample before trusting the estimate.
    """
    if a.size == 0 or b.size == 0:
        return 0.0, 0
    small, large = (a, b) if a.size <= b.size else (b, a)
    threshold = min(small[-1], large[-1])
    s = small[small <= threshold]
    l = large[large <= threshold]
    if s.size == 0:
        return 0.0, 0
    return float(np.intersect1d(s, l, assume_unique=True).size) / float(s.size), int(s.size)


def mash_ani(jaccard: float, k: int) -> float:
    if jaccard <= 0.0:
        return 0.0
    if jaccard >= 1.0:
        return 1.0
    distance = -math.log(2.0 * jaccard / (1.0 + jaccard)) / k
    return max(0.0, 1.0 - distance)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step05.cluster")
    slog = StageLog("step05_cluster")
    scfg = cfg["similarity"]

    meta_dir = path_for(cfg, "metadata_dir")
    rep = pd.read_csv(meta_dir / "replicons.csv", keep_default_na=False)

    # Only replicons that can actually enter the binary dataset are clustered.
    usable = rep[rep["replicon_type"].isin(["chromosome", "plasmid"])].reset_index(drop=True)
    n_before = len(usable)
    min_len = min_fragment_length(cfg)
    usable = usable[usable["length"] >= min_len].reset_index(drop=True)
    slog.record("replicons eligible for clustering", n_before, len(usable),
                f"length >= shortest emittable fragment ({min_len} bp)")
    logger.info("Clustering %d eligible replicons (of %d chromosome/plasmid records)",
                len(usable), n_before)

    uf = UnionFind(len(usable))
    idx_of = {rid: i for i, rid in enumerate(usable["replicon_id"])}

    # ---------------------------------------------------------- exact duplicates
    n_exact_links = 0
    if scfg.get("always_cluster_exact_duplicates", True):
        for _, group in usable.groupby("sequence_sha256"):
            ids = [idx_of[r] for r in group["replicon_id"]]
            for other in ids[1:]:
                uf.union(ids[0], other)
                n_exact_links += 1
    logger.info("Exact-duplicate links: %d", n_exact_links)

    pairs: list[dict] = []
    if scfg.get("enabled", True):
        k = int(scfg["kmer_size"])
        size = int(scfg["sketch_size"])
        logger.info("Sketching (k=%d, sketch_size=%d)...", k, size)

        # Sketches are cached on disk keyed by the sketching parameters, so an
        # interrupted run - or a re-run with a different ANI threshold - does
        # not have to re-read every genome.
        cache_path = meta_dir / f"sketch_cache_k{k}_s{size}.npz"
        cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            with np.load(cache_path) as npz:
                cache = {name: npz[name] for name in npz.files}
            logger.info("Loaded %d cached sketches from %s", len(cache), cache_path)

        sketches: list[np.ndarray] = [np.empty(0, dtype=np.uint64)] * len(usable)
        wanted: dict[str, dict[str, int]] = defaultdict(dict)
        for pos, row in enumerate(usable.itertuples(index=False)):
            if row.replicon_id in cache:
                sketches[pos] = cache[row.replicon_id]
            else:
                wanted[row.source_fasta][row.sequence_id] = pos

        ids = usable["replicon_id"].tolist()
        done, todo = 0, sum(len(v) for v in wanted.values())
        logger.info("Sketching %d replicons (%d served from cache)",
                    todo, len(usable) - todo)
        for fasta_path, want in wanted.items():
            for defline, seq in read_fasta(fasta_path):
                pos = want.get(defline.split()[0])
                if pos is None:
                    continue
                sketches[pos] = sketch(seq, k, size)
                cache[ids[pos]] = sketches[pos]
                done += 1
                if done % 100 == 0:
                    logger.info("  sketched %d/%d replicons", done, todo)
        if todo:
            np.savez_compressed(cache_path, **cache)
            logger.info("Wrote sketch cache %s (%d entries)", cache_path, len(cache))

        # ------------------------------------------------ candidate generation
        inverted: dict[int, list[int]] = defaultdict(list)
        for i, sk in enumerate(sketches):
            for h in sk.tolist():
                inverted[h].append(i)
        candidates: set[tuple[int, int]] = set()
        for members in inverted.values():
            if 1 < len(members) <= 500:  # ultra-common hashes carry no signal
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        candidates.add((members[a], members[b]))
        logger.info("Candidate pairs sharing >=1 sketched k-mer: %d", len(candidates))

        thr = float(scfg["ani_threshold"])
        min_ratio = float(scfg["min_length_ratio"])
        cross_ok = bool(scfg.get("compare_across_replicon_types", True))
        lengths = usable["length"].to_numpy()
        types = usable["replicon_type"].to_numpy()

        cont_thr = float(scfg.get("containment_threshold", 0.90))
        min_cont_sample = int(scfg.get("min_containment_sample", 50))
        for a, b in sorted(candidates):
            if not cross_ok and types[a] != types[b]:
                continue
            j = sketch_jaccard(sketches[a], sketches[b], size)
            if j <= 0:
                continue
            ani = mash_ani(j, k)
            ratio = min(lengths[a], lengths[b]) / max(lengths[a], lengths[b])
            containment, cont_sample = sketch_containment(sketches[a], sketches[b])
            # Two ways to be redundant: near-identical over comparable lengths,
            # or the shorter replicon being largely contained in the longer one
            # (a small plasmid that is a subsequence of a bigger one would
            # otherwise sail past a Jaccard/length-ratio test). The containment
            # estimate is only trusted when it rests on enough shared-range
            # hashes to mean anything.
            by_ani = ani >= thr and ratio >= min_ratio
            by_containment = containment >= cont_thr and cont_sample >= min_cont_sample
            linked = by_ani or by_containment
            if ani >= thr - 0.05 or containment >= cont_thr - 0.1 or linked:
                pairs.append({
                    "replicon_id_a": usable.at[a, "replicon_id"],
                    "replicon_id_b": usable.at[b, "replicon_id"],
                    "type_a": types[a], "type_b": types[b],
                    "length_a": int(lengths[a]), "length_b": int(lengths[b]),
                    "jaccard_estimate": round(j, 6),
                    "ani_estimate": round(ani, 6),
                    "length_ratio": round(float(ratio), 6),
                    "containment_estimate": round(containment, 6),
                    "containment_sample": cont_sample,
                    "linked": bool(linked),
                    "link_reason": ("ani+length_ratio" if by_ani and not by_containment
                                    else "containment" if by_containment and not by_ani
                                    else "ani+containment" if linked else ""),
                    "cross_type": bool(types[a] != types[b]),
                    "same_assembly": bool(usable.at[a, "assembly_accession"]
                                          == usable.at[b, "assembly_accession"]),
                })
            if linked:
                uf.union(a, b)

    pairs_df = pd.DataFrame(pairs)
    reports = path_for(cfg, "reports_dir")
    pairs_path = reports / f"similarity_pairs{report_tag(cfg)}.csv"
    pairs_df.to_csv(pairs_path, index=False)

    # ------------------------------------------------------------- clusters
    roots = [uf.find(i) for i in range(len(usable))]
    root_to_cluster = {r: f"cluster_{n:06d}" for n, r in enumerate(sorted(set(roots)), start=1)}
    usable["cluster_id"] = [root_to_cluster[r] for r in roots]

    # Replicons excluded from clustering still get a row, each in its own
    # singleton cluster, so nothing silently disappears.
    excluded = rep[~rep["replicon_id"].isin(set(usable["replicon_id"]))].copy()
    excluded["cluster_id"] = ""

    out_cols = ["replicon_id", "assembly_accession", "sequence_id", "organism_name",
                "taxid", "species_taxid", "replicon_type", "label", "length",
                "sequence_sha256", "cluster_id", "source_fasta"]
    clustered = pd.concat([usable[out_cols], excluded[out_cols]], ignore_index=True)
    clustered = clustered.sort_values(["cluster_id", "replicon_id"]).reset_index(drop=True)
    clusters_path = meta_dir / f"clusters{report_tag(cfg)}.csv"
    clustered.to_csv(clusters_path, index=False)

    sizes = usable.groupby("cluster_id").size()
    n_clusters = int(sizes.size)
    n_multi = int((sizes > 1).sum())
    n_linked_pairs = int(pairs_df["linked"].sum()) if len(pairs_df) else 0
    cross = pairs_df[pairs_df["cross_type"]] if len(pairs_df) else pairs_df
    n_cross_linked = int(cross["linked"].sum()) if len(cross) else 0
    linked_df = pairs_df[pairs_df["linked"]] if len(pairs_df) else pairs_df
    n_by_ani = int((linked_df["link_reason"] == "ani+length_ratio").sum()) if len(linked_df) else 0
    n_by_cont = int((linked_df["link_reason"] == "containment").sum()) if len(linked_df) else 0
    n_same_asm = int(linked_df["same_assembly"].sum()) if len(linked_df) else 0

    summary = [
        "=" * 78,
        "REDUNDANCY / SIMILARITY ANALYSIS (parent replicons, before splitting)",
        "=" * 78,
        "",
        f"Method .................... MinHash sketching + Mash ANI estimate",
        f"k-mer size ................ {scfg['kmer_size']}",
        f"sketch size ............... {scfg['sketch_size']}",
        f"ANI threshold ............. {scfg['ani_threshold']}",
        f"min length ratio .......... {scfg['min_length_ratio']} (coverage proxy)",
        f"containment threshold ..... {scfg.get('containment_threshold', 0.90)}",
        f"linkage ................... {scfg['linkage']}",
        "",
        f"Replicons clustered ....... {len(usable)}",
        f"Clusters .................. {n_clusters}",
        f"Multi-member clusters ..... {n_multi}",
        f"Largest cluster ........... {int(sizes.max()) if n_clusters else 0} replicons",
        f"Exact-duplicate links ..... {n_exact_links}",
        f"Similarity-linked pairs ... {n_linked_pairs}",
        f"  by ANI + length ratio ... {n_by_ani}",
        f"  by containment .......... {n_by_cont}",
        f"  of which cross-type ..... {n_cross_linked} (chromosome <-> plasmid)",
        f"  of which same assembly .. {n_same_asm}",
        f"Redundancy removed from",
        f"  the splitting unit ...... {len(usable) - n_clusters} replicons absorbed into clusters",
        "",
        "INTERPRETATION / CAVEATS",
        f"  * Two replicons are linked when EITHER estimated ANI >= {scfg['ani_threshold']}"
        " with a",
        f"    length ratio >= {scfg['min_length_ratio']}, OR the shorter one's containment"
        " in the longer",
        f"    one is >= {scfg.get('containment_threshold', 0.90)}. Both are EXPERIMENTAL"
        " CRITERIA chosen for this",
        "    dataset, not universally correct biological thresholds. Changing",
        "    them changes the split, so they are recorded in config.yaml and in",
        "    reports/run_manifest.json.",
        "  * Containment exists because Jaccard collapses for unequal lengths:",
        "    a small plasmid that is essentially a subsequence of a larger",
        "    replicon would otherwise be free to land in a different split.",
        "  * MinHash estimates whole-sequence k-mer similarity. It is not an",
        "    alignment, so the length ratio only approximates coverage.",
        f"  * Containment is only trusted on >= {scfg.get('min_containment_sample', 50)}"
        " hashes of shared sketch range.",
        "    It is therefore under-powered for extremely unequal pairs (a 10-kb",
        "    plasmid inside a 5-Mb chromosome): such containment would need an",
        "    alignment to establish, and is not claimed here either way.",
        "  * Chromosomes and plasmids WERE compared against each other. Shared",
        "    sequence between them is biologically real (HGT, mobile elements,",
        "    prophage, AMR genes); any cross-type link is listed in",
        "    reports/similarity_pairs.csv rather than assumed impossible.",
        "  * Clustering guarantees only that identical and near-identical",
        "    parent replicons never straddle train/validation/test. It does not",
        "    claim the splits are free of all homology.",
        "=" * 78,
    ]
    summary_path = reports / f"similarity_clusters{report_tag(cfg)}.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))

    slog.record("replicons -> clusters (splitting units)", len(usable), n_clusters,
                f"single-linkage at ANI>={scfg['ani_threshold']}")
    slog.flush(cfg)
    update_manifest(cfg, "step05_cluster", {
        "method": scfg["method"],
        "parameters": scfg,
        "replicons_clustered": len(usable),
        "clusters": n_clusters,
        "multi_member_clusters": n_multi,
        "exact_duplicate_links": n_exact_links,
        "similarity_linked_pairs": n_linked_pairs,
        "cross_type_linked_pairs": n_cross_linked,
        "clusters_csv": str(clusters_path),
        "pairs_csv": str(pairs_path),
    })
    return clustered


if __name__ == "__main__":
    run()
