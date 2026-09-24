"""Correctness checks for the feature pipeline, run before any training.

Checks the vectorised k-mer counter against a brute-force count, the canonical
column order against KMC's own ordering, and the two transforms against the
scikit-learn classes the original code used.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import features  # noqa: E402
from baselines.data import CODE_LUT  # noqa: E402


def brute_force_counts(seqs, k, canonical):
    bases = "ACGT"
    comp = str.maketrans("ACGT", "TGCA")
    if canonical:
        colmap = features.canonical_colmap(k)
        cols = np.array([np.flatnonzero(colmap == c)[0] for c in range(colmap.max() + 1)])
        names = features.kmer_strings(k, cols)
    else:
        names = features.kmer_strings(k)
    order = {name: i for i, name in enumerate(names)}
    out = np.zeros((len(seqs), len(names)), dtype=np.int64)
    for r, s in enumerate(seqs):
        counter = Counter()
        for i in range(len(s) - k + 1):
            word = s[i:i + k]
            if any(c not in bases for c in word):
                continue
            if canonical:
                word = min(word, word.translate(comp)[::-1])
            counter[word] += 1
        for word, n in counter.items():
            out[r, order[word]] = n
    return out


def random_sequences(rng, n, length):
    alphabet = np.array(list("ACGTN"))
    p = [0.245, 0.245, 0.245, 0.245, 0.02]
    return ["".join(rng.choice(alphabet, size=length, p=p)) for _ in range(n)]


def main():
    rng = np.random.default_rng(0)
    seqs = random_sequences(rng, 8, 400)
    codes = np.stack([CODE_LUT[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])

    for k, canonical in [(3, False), (5, False), (6, False), (7, False), (6, True), (4, True)]:
        n_cols = features.n_features(k, canonical)
        colmap = features.canonical_colmap(k) if canonical else None
        got = features.count_kmers(codes, k, colmap, n_cols)
        want = brute_force_counts(seqs, k, canonical)
        assert got.shape == (len(seqs), n_cols), (got.shape, n_cols)
        assert np.array_equal(got, want), f"k-mer counts differ for k={k} canonical={canonical}"
        expected_positions = [sum(1 for i in range(len(s) - k + 1)
                                  if "N" not in s[i:i + k]) for s in seqs]
        assert np.array_equal(got.sum(axis=1), expected_positions), "N-containing windows counted"
        print(f"  k={k} canonical={canonical}: {n_cols} columns, counts match brute force")

    kmc = (HERE / "reference" / "kmc_kmer_order.csv").read_text().split()
    colmap = features.canonical_colmap(6)
    cols = np.array([np.flatnonzero(colmap == c)[0] for c in range(colmap.max() + 1)])
    assert features.kmer_strings(6, cols) == kmc, "canonical order differs from KMC"
    print(f"  canonical 6-mer column order identical to KMC kmer_order.csv ({len(kmc)} columns)")

    counts = features.count_kmers(codes, 5, None, 1024)
    tmp = HERE / "cache" / "_verify.u16"
    tmp.parent.mkdir(exist_ok=True)
    np.memmap(tmp, dtype=np.uint16, mode="w+", shape=counts.shape)[:] = counts

    idf = features.TfidfTransform.fit(tmp, *counts.shape, 1 << 20)
    want = TfidfTransformer(smooth_idf=False).fit_transform(counts).toarray()
    assert np.allclose(idf(counts), want, atol=1e-5), "TF-IDF differs from sklearn"
    print("  TF-IDF matches sklearn TfidfTransformer(smooth_idf=False)")

    scaler = features.StandardTransform.fit(tmp, *counts.shape, 1 << 20)
    want = StandardScaler().fit_transform(counts.astype(np.float64))
    assert np.allclose(scaler(counts), want, atol=1e-4), "scaling differs from sklearn"
    print("  standardisation matches sklearn StandardScaler")
    tmp.unlink()
    print("all feature checks passed")


if __name__ == "__main__":
    main()
