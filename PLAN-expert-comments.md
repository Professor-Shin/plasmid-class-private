# Plan — Addressing the biology expert's three comments

## Context

A domain expert reviewed the plasmid-vs-chromosome project and raised three points:

1. **5 kb fragment length is a "magic number"** — justify against biology or span many lengths.
2. **GC content should be considered** explicitly.
3. **Non-coding regions carry plasmid signal** — notably the *origin of replication* (oriV/oriT) —
   so the model must be able to see non-coding DNA, not only translated ORFs.

Two design decisions (agreed with the project owner) reframe the response:

- **One mixed-length dataset, length-stratified at evaluation** — not one model per length. Each
  replicon is tiled into non-overlapping windows whose length is **randomly sampled** (so no single
  length is privileged, which is the actual answer to "magic number"); small replicons (<1 kb floor
  impossible here — smallest plasmid is 1119 bp) are fed at their true size, rescuing the 39 small
  plasmids (10.2%) currently dropped. Every model trains on this one set; metrics are then broken
  down by length bucket to show how length affects performance.
- **The ESM-2 sequence-embedding model is the core contribution**; the work is a *feature ladder* on
  top of it: `esm2 → +gc → +ori → +gc+ori → +dnabert → +all`. PlasFlow/Zou/PlasClass become external
  baselines, retrained on the same mixed-length set for a fair comparison.

Why the code rewards this: the **ESM-2 path is already length-agnostic** — it reads variable-length
sequences, extracts ORFs, and attention-pools over a *variable* ORF count; `fragment_length: 5000`
in `esm2-attention/config.yaml` is only recorded in `environment.json`, never used in training
(`train_eval.py`, `extract_features.py`). So mixed-length training needs **no ESM architecture
change** — only the new feature branches. The **baselines** currently hard-reject any length ≠ 5000
(`baseline/baselines/data.py:66-67`) because they stage raw sequence in a fixed `(n, frag_len)`
memmap; but k-mer *frequency* width is `4^k` regardless of length, so the only change is to count
k-mers per fragment directly.

---

## Work item 1 — Mixed-length dataset by random window length

**Fragmentation (rewrite `dataset pipeline1/pipeline/step07_fragments.py`).**
Steps 1–6 and the cluster→split map in `replicons_split.csv` stay **fixed** (no leakage: a replicon
and all its windows live in one split). For each replicon, tile left-to-right with non-overlapping
windows; for each window sample a target length `L ~ log-uniform[1 kb, 100 kb]`, clamped to the
remaining sequence. A trailing remainder below the 1 kb floor is dropped; a whole replicon shorter
than the sampled `L` becomes one fragment at its true length. Keep the existing per-fragment
`gc_content`/`non_acgt_fraction` computation and the `max_non_acgt_fraction` filter
(`base_stats`, step07 lines 48-57). Record both the sampled `target_length` and actual `length`
columns.

**Reproducibility:** seed the per-window RNG deterministically from `random_seed` + a hash of
`replicon_id`, so the dataset rebuilds byte-for-byte (the config's reproducibility contract).

**Show the distribution (explicit request).** Add a reporting step (extend the step07 summary or a new
`analysis/length_distribution.py`) that emits: histogram of realized fragment `length`; per
length-bucket counts of fragments, total bp, and plasmid fraction; and the same split by class.
Buckets: `1–2, 2–5, 5–10, 10–25, 25–50, 50–100 kb`. This is a deliverable, not just a sanity check.

**Config.** Replace the fixed `fragments.length/step` block in `dataset pipeline1/config.yaml`
(lines 116-127) with `length_sampling: {dist: log_uniform, min: 1000, max: 100000}` plus the
existing `drop`/`max_non_acgt_fraction` keys.

---

## Work item 2 — Make every model length-robust + the GC baseline/analysis

**Baselines on mixed length.** Replace the fixed-length raw-sequence staging with per-fragment k-mer
counting: iterate fragments, encode each to base codes, count k-mers into the existing fixed-width
`(n, 4^k)` feature memmap. Reuse the counting core in `baseline/baselines/features.py:count_kmers`
(lines 65-81); the only change is feeding it one variable-length fragment at a time instead of a
fixed `(n, frag_len)` block. Everything downstream (TF-IDF / StandardScaler / `MemmapStream`) is
unchanged because the feature matrix stays fixed-width. Drop the `len(seq) != frag_len` guard in
`encode_split` (data.py lines 52-74). Re-run PlasFlow, Zou, PlasClass on the new set.

**GC-only baseline (floor).** New `baseline/run_gc.py`: logistic regression on `gc_content` alone
(straight from the CSV column — no memmap), reported through the existing
`baseline/baselines/metrics.py` / `summarize.py`. Establishes "how much is just GC?"

**GC-vs-accuracy analysis.** New `analysis/gc_signal.py`: per test replicon compute
|plasmid GC − host-chromosome GC| (host GC is in `data/metadata/replicons.csv`) and correlate with
per-fragment correctness — direct evidence of how much the models lean on GC (starred Q1 / H1).

---

## Work item 3 — ESM-2 feature ladder (the core development)

All variants keep the existing ORF-attention branch (`esm2-attention/esm2/model.py`,
`EsmAttentionClassifier`) and concatenate extra **whole-fragment** features onto the pooled 320-d
ORF vector *before* the classifier head. Because these features see the entire fragment, they also
compensate for the `max_orfs: 10` cap that under-samples coding content on long fragments — and they
fill the current zero-vector hole for ORF-free fragments.

| Variant | Added to pooled ORF embedding |
|---|---|
| `esm2` (base) | nothing (current model) |
| `esm2+gc` | `gc_content` (+ fragment GC value) |
| `esm2+ori` | ori-oriented composition vector (see below) |
| `esm2+gc+ori` | both |
| `esm2+dnabert` | pooled whole-fragment DNA-LM embedding |
| `esm2+all` | gc + ori + dnabert |

**`+ori` = composition proxy (now), true ori (HPC later).** We do **not** annotate a real origin of
replication yet. `+ori` is a small, cheap vector of non-coding signals that origins are known to
carry, computed from the fragment sequence:
- **Cumulative GC-skew** summary — `(G−C)/(G+C)` integrated along the sequence; its sign flips at the
  replication origin/terminus because leading/lagging strands have different mutational bias.
  Features: skew range, and position of the cumulative min/max.
- **AT-richness** of the most AT-rich window — DnaA-unwinding elements / DUEs at origins are AT-rich
  to ease strand opening.
- **DnaA-box motif density** — origins cluster the ~9-mer `TTATCCACA`-like boxes.
- **Short tandem direct-repeat (iteron) density** — plasmid oriV binds Rep via iteron repeats.

These are exactly the non-coding features the protein/ORF pathway is blind to — the expert's point —
and each has a mechanistic reason, so the choice is defensible in the write-up. A **true-ori** version
(e.g. oriC/oriV detection via an external tool or database) is deferred to the HPC run and noted as
future work; the `+ori` branch is built so its input vector can later be swapped for true-ori features
without touching the model.

**`+dnabert`** = a parallel branch embedding the whole fragment with a small DNA language model
(DNABERT-2 or a 50 M Nucleotide-Transformer), chunk-and-pooled for long fragments. Heaviest piece;
precompute once on the subsample like ESM. Flag as the likely-HPC item for the full run.

**Implementation.** Compute the gc/ori vectors in `esm2-attention/extract_features.py` (cheap, from
the already-read sequence) and store them beside the ORF embeddings in the npz; load them in
`EmbeddingSet` (`esm2-attention/esm2/data.py` lines 85-106); give `EsmAttentionClassifier` an
`aux_dim` and concatenate before the head; select the active branches via a config key
(e.g. `features: [gc, ori]`) so one code path produces all six variants. DNA-LM embeddings come from
a new `esm2-attention/extract_dna_lm.py`.

---

## Cross-cutting — length-stratified evaluation

Add a shared helper (extend both `summarize.py` files) that buckets **test** fragments by `length`
(same buckets as Work item 1) and reports fragment-level and replicon-level MCC/F1 per bucket for
**every** model and variant. This is the headline table answering comment 1: performance as a
function of length, for one length-robust model, rather than a chosen magic number.

## Compute realism (8 GB RAM, CPU-only)

- ESM embedding cost per fragment is capped by `max_orfs: 10`, so mixed lengths don't blow up the
  budget; keep the subsample sizes in `esm2-attention/config.yaml` (lines 36-39).
- `+dnabert` (and true-ori) are the heavy pieces → earmarked for HPC; the CPU plan can validate them
  on a small subsample first.
- **Validate small before full:** rebuild the dataset, run the GC baseline and `esm2`/`esm2+gc`/
  `esm2+ori` at reduced subsample to confirm wiring, then scale up.

## Time budget & 4-hour triage (8 GB / 4-core CPU)

Rough estimates; the ESM embedding pass is the long pole and is shared by all `esm2(+gc/+ori)` variants.

| Task | Est. time | In the 4 h run? |
|---|---|---|
| Dataset rebuild — mixed-length `step07` | 15–30 min | ✅ |
| Length-distribution report | ~5 min | ✅ |
| GC-only baseline (`run_gc.py`, from CSV column) | ~10 min | ✅ |
| **ESM embedding pass** (~22.7k subsampled frags @ 0.44 s) | **2.5–3 h** | ✅ (shared by 4 variants) |
| Train `esm2`, `+gc`, `+ori`, `+gc+ori` (reuse embeddings) | ~15–20 min each | ✅ |
| Length-stratified `summarize` + `gc_signal.py` | ~15 min | ✅ |
| `esm2+dnabert` / `esm2+all` (DNA-LM embed pass) | +3–6 h | ❌ defer → HPC |
| Full baseline retrain (PlasFlow/Zou/PlasClass on 482k) | 3–6 h | ❌ defer / overnight |
| `+ori` with a **true** origin detector | setup-heavy | ❌ defer → HPC |

**To fit 4 h with margin:** trim the chromosome subsample for this run (train 4000→2500, val
4000→2000, test-chrom 8000→4000) → embedding drops to ~1.5–2 h; and set `n_seeds: 1` for the first
pass (restore 3 seeds on HPC). That leaves comfortable room for all four light variants + analysis.

**Delivered in 4 h:** mixed-length dataset + length distribution; length-stratified results for
`esm2`, `+gc`, `+ori`, `+gc+ori`; GC-only floor; GC-vs-accuracy analysis. **Deferred:** `+dnabert`,
`+all`, full-dataset baseline retrain, true-ori — all flagged for HPC / overnight.

## Suggested execution order

1. Random-length `step07` rewrite + config + length-distribution report (cheap; step 7 only).
2. Baseline mixed-length counting change; retrain PlasFlow/Zou/PlasClass + GC-only baseline.
3. ESM aux-feature plumbing; build `esm2`, `+gc`, `+ori`, `+gc+ori`.
4. Length-stratified `summarize` + `gc_signal.py` analysis; assemble the headline length tables.
5. `+dnabert` (and later true-ori) — CPU smoke test, then HPC for the full run.

## Verification

- **Dataset:** step07's integrity checks still pass (FASTA/CSV id match, no replicon spans splits, no
  duplicate ids); a second build with the same config reproduces identical files; the length
  histogram + per-bucket class table render and look sane (small plasmids now present).
- **Length-robustness:** baselines and ESM train and evaluate end-to-end on mixed lengths with no
  fixed-length error; the per-bucket metric table is produced for all models.
- **GC:** GC-only floor reported beside k-mer baselines; `gc_signal.py` gives the accuracy-vs-GC-gap
  correlation.
- **Feature ladder:** the six variants train from one code path via the `features:` config; compare
  them at fragment and replicon level, with attention to ORF-free and small-plasmid fragments; confirm
  `+ori`/`+dnabert` help where the ORF-only model is weakest.
- Re-run the existing `summarize.py` / REPORT generation so results land in the format used for the
  expert discussion (`QUESTIONS-FOR-EXPERT.md`, REPORTs).
