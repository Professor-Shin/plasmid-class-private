"""Whole-fragment composition features: the non-coding side of the signal.

The ORF/ESM-2 pathway only ever sees translated peptides, so anything a plasmid
encodes in non-coding DNA - above all the origin of replication - is invisible to
it, and an ORF-free fragment reaches the classifier as a single zero vector.
These features are the cheap, annotation-free stand-in, and each one is here for
a mechanistic reason rather than because it was available:

``gc`` block (4 dims)
    Overall GC plus how much GC varies locally. A plasmid that arrived by recent
    horizontal transfer has not yet ameliorated to its host's GC, and cargo
    regions (AMR, IS) make plasmid GC more heterogeneous than chromosomal GC.

``ori`` block (8 dims)
    - **Cumulative GC skew**, (G-C) integrated along the sequence. Leading and
      lagging strands accumulate different mutational biases, so cumulative skew
      turns at the replication origin and again at the terminus: a window holding
      an origin shows an inflection instead of a monotone ramp. Amplitude and the
      normalised positions of the turn are what we keep.
    - **AT-richness of the most AT-rich window**. Origins carry DNA-unwinding
      elements (DUEs) that are AT-rich so the duplex opens at lower energy cost.
    - **DnaA-box density**. Replication initiator boxes matching the TTATCCACA
      consensus cluster at oriC and at many plasmid origins; counted on both
      strands allowing one mismatch.
    - **Duplicated-k-mer fraction**. Plasmid oriV binds its Rep protein at
      iterons - tandem direct repeats - and repeat-dense DNA is also a general
      mobile-element signature.

None of this *detects* an origin; it measures the composition an origin leaves
behind. Replacing the proxy with real oriC/oriV calls later changes only this
module, not the model that consumes it.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 4 marks any non-ACGT base, matching the convention in baselines/data.py.
CODE_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    CODE_LUT[_b] = _i
    CODE_LUT[_b + 32] = _i

A, C, G, T = 0, 1, 2, 3

# DnaA box consensus (E. coli oriC R-boxes) and its reverse complement, so a box
# is found whichever strand it sits on.
_DNAA_BOX = np.array([T, T, A, T, C, C, A, C, A], dtype=np.uint8)
_DNAA_BOX_RC = np.array([T, G, T, G, G, A, T, A, A], dtype=np.uint8)

WINDOW = 200          # bp, local-composition window (DUE-scale)
N_CHUNKS = 20         # equal chunks for length-independent heterogeneity stats
REPEAT_K = 12         # k for the duplicated-k-mer (iteron/repeat) statistic

GC_DIM = 4
ORI_DIM = 8
BLOCK_DIMS = {"gc": GC_DIM, "ori": ORI_DIM}
# Fixed storage order. Every fragment is extracted once with ALL blocks, and a
# variant then selects its columns, so the expensive ESM pass is never repeated
# just to change which auxiliary features the classifier sees.
ALL_BLOCKS = ("gc", "ori")


def encode(seq: str) -> np.ndarray:
    return CODE_LUT[np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)]


def _windows(codes: np.ndarray, width: int) -> np.ndarray:
    """Non-overlapping windows of `width`, dropping a short tail."""
    n = (len(codes) // width) * width
    if n == 0:
        return codes.reshape(1, -1)
    return codes[:n].reshape(-1, width)


def _frac(counts: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.divide(counts, total, out=np.zeros_like(counts, dtype=np.float64),
                     where=total > 0)


def gc_features(codes: np.ndarray) -> np.ndarray:
    g = codes == G
    c = codes == C
    acgt = codes != 4
    gc = float(_frac(np.array([(g | c).sum()]), np.array([acgt.sum()]))[0])

    win = _windows(codes, WINDOW)
    w_gc = _frac(((win == G) | (win == C)).sum(1).astype(np.float64),
                 (win != 4).sum(1).astype(np.float64))
    chunks = np.array_split(codes, min(N_CHUNKS, max(1, len(codes) // 50)))
    c_gc = np.array([_frac(np.array([float(((ch == G) | (ch == C)).sum())]),
                           np.array([float((ch != 4).sum())]))[0] for ch in chunks])
    return np.array([gc, float(c_gc.std()), float(w_gc.min()), float(w_gc.max())],
                    dtype=np.float32)


def ori_features(codes: np.ndarray) -> np.ndarray:
    n = len(codes)
    g = codes == G
    c = codes == C
    a = codes == A
    t = codes == T

    # ---- cumulative GC skew: turns at the origin and at the terminus
    step = g.astype(np.int32) - c.astype(np.int32)
    cum = np.cumsum(step)
    amplitude = float(cum.max() - cum.min()) / max(n, 1)
    argmin = float(cum.argmin()) / max(n - 1, 1)
    argmax = float(cum.argmax()) / max(n - 1, 1)

    ng, nc, na, nt = int(g.sum()), int(c.sum()), int(a.sum()), int(t.sum())
    gc_skew = (ng - nc) / (ng + nc) if ng + nc else 0.0
    at_skew = (na - nt) / (na + nt) if na + nt else 0.0

    # ---- most AT-rich window: the DUE signature
    win = _windows(codes, WINDOW)
    w_at = _frac(((win == A) | (win == T)).sum(1).astype(np.float64),
                 (win != 4).sum(1).astype(np.float64))
    max_at = float(w_at.max())

    # ---- DnaA boxes, both strands, Hamming distance <= 1
    if n >= _DNAA_BOX.size:
        view = sliding_window_view(codes, _DNAA_BOX.size)
        hits = ((view != _DNAA_BOX).sum(1) <= 1) | ((view != _DNAA_BOX_RC).sum(1) <= 1)
        dnaa_per_kb = float(hits.sum()) / (n / 1000.0)
    else:
        dnaa_per_kb = 0.0

    # ---- duplicated k-mer fraction: iterons and other direct repeats
    dup_frac = 0.0
    if n >= REPEAT_K:
        view = sliding_window_view(codes, REPEAT_K)
        clean = (view != 4).all(1)
        if clean.any():
            vals = np.zeros(view.shape[0], dtype=np.int64)
            for j in range(REPEAT_K):
                vals = vals * 4 + view[:, j]
            vals = vals[clean]
            _, counts = np.unique(vals, return_counts=True)
            dup_frac = float(counts[counts > 1].sum()) / len(vals)

    return np.array([amplitude, argmin, argmax, gc_skew, at_skew,
                     max_at, dnaa_per_kb, dup_frac], dtype=np.float32)


def feature_dim(blocks) -> int:
    return sum(BLOCK_DIMS[b] for b in blocks)


def fragment_features(seq: str, blocks=ALL_BLOCKS) -> np.ndarray:
    """Concatenate the requested blocks for one fragment, in ALL_BLOCKS order."""
    codes = encode(seq)
    parts = []
    for block in ALL_BLOCKS:             # fixed order, independent of caller order
        if block in blocks:
            parts.append(gc_features(codes) if block == "gc" else ori_features(codes))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


FEATURE_NAMES = {
    "gc": ["gc", "gc_chunk_std", "gc_window_min", "gc_window_max"],
    "ori": ["skew_amplitude", "skew_argmin", "skew_argmax", "gc_skew", "at_skew",
            "max_at_window", "dnaa_boxes_per_kb", "dup_kmer_frac"],
}


def all_feature_names() -> list[str]:
    return [n for b in ALL_BLOCKS for n in FEATURE_NAMES[b]]


def block_columns(blocks) -> np.ndarray:
    """Column indices of `blocks` inside a full ALL_BLOCKS feature vector."""
    idx: list[int] = []
    offset = 0
    for block in ALL_BLOCKS:
        width = BLOCK_DIMS[block]
        if block in blocks:
            idx.extend(range(offset, offset + width))
        offset += width
    return np.array(idx, dtype=np.int64)
