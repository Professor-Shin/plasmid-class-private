# Proposed model: ESM-2 ORF embeddings + trainable attention pooling

The architecture prototyped in `../old resource/` (`09_train_esm2.py`,
`10_esm_attention.py`), re-implemented and trained on the current RefSeq 5-kb
fragment dataset from `../dataset pipeline1`, and compared head-to-head with the
three k-mer baselines in `../baseline` (PlasFlow, Zou et al., PlasClass) on exactly
the same evaluation fragments.

```
5-kb DNA fragment
   -> six-frame translation, split at stop codons, keep peptides >= 100 aa (top 10)
   -> ESM-2 (facebook/esm2_t6_8M_UR50D), mean-pool residues -> 320-d per ORF
   -> trainable attention pooling over the fragment's ORFs -> 320-d context
   -> Linear(320,64) + ReLU + Dropout(0.3) + Linear(64,1) -> plasmid probability
```

## Run

```bash
py -3.13 verify_components.py    # ORF calling, embedding and pooling vs. the prototype
py -3.13 extract_features.py     # the expensive step: ESM-2 over every ORF (~1-2 h)
py -3.13 train_eval.py           # 3 seeds x 15 epochs on the cached embeddings
py -3.13 summarize.py            # re-scores the baselines on the same fragments
py -3.13 capture_environment.py  # results/environment.json
```

Requirements: Python 3.13 with `torch`, `transformers`, `biopython`, `numpy`,
`pandas`, `scikit-learn`, `pyyaml`. CPU is enough. The ESM-2 checkpoint (~31 MB) is
downloaded once by `transformers` and then read from the local Hugging Face cache.

## Why the dataset is subsampled

ESM-2 costs about 0.44 s per 5-kb fragment on this machine, so embedding all 482,161
fragments would take roughly 60 hours. Each split is therefore sampled **per class**,
and every metric is re-weighted by how many fragments of that class each sampled row
stands for, so the reported numbers estimate performance on the full split rather than
on the subsample. Sizes live in `config.yaml`; see `REPORT.md` section 3.3.

Because the baselines saved a probability for every test fragment, they can be
re-scored on precisely the fragments ESM-2 saw — the comparison is paired, not
approximate.

## Layout

```
esm2/               ORF calling, ESM-2 embedding, the classifier, weighted metrics
extract_features.py subsample -> ORFs -> ESM-2 embeddings (cache/emb_*.npz)
train_eval.py       trains the attention head, evaluates, writes results/esm2_attention.json
summarize.py        head-to-head tables against ../baseline -> results/summary.md
verify_components.py equivalence checks against the prototype implementation
capture_environment.py hardware, versions, ESM-2 weight fingerprint, checksums
cache/              embeddings and the exact subsample tables (git-ignored)
results/            metrics, per-fragment probabilities, summary.md, environment.json
```
