# Baselines: PlasFlow and Zou et al., retrained on our dataset

Two published plasmid-vs-chromosome classifiers, re-implemented from their
released source code and **retrained from scratch** on the 5-kb fragment dataset
produced by `../dataset pipeline1`:

| baseline | paper | source code |
|---|---|---|
| PlasFlow | Krawczyk, Lipinski & Dziembowski, *NAR* 2018;46(6):e35 | <https://github.com/smaegol/PlasFlow> |
| Zou et al. | Zou, Nguyen, Overbeek, Cao & Davis, *PLoS ONE* 2022;17(12):e0279280 | <https://github.com/BV-BRC-dependencies/zou-plasmid-prediction> |
| PlasClass | Pellow, Zorea, Probst et al., *PLoS Comput Biol* 2020;16(10):e1007781 | <https://github.com/Shamir-Lab/PlasClass> |

Retraining rather than running the released weights is deliberate: both published
models were fitted on public plasmid/chromosome collections that overlap RefSeq,
so their pretrained weights would have seen our test replicons. `REPORT.md` holds the
full write-up: section 3 records the exact environment the published numbers came from,
and section 4.4 lists every deviation from the original protocols.

## Run

```bash
py -3.13 verify_features.py     # correctness checks for the k-mer/transform code
py -3.13 prepare_features.py    # encode FASTA, build every k-mer matrix (~25 GB in cache/)
py -3.13 run_plasflow.py        # 3 models x 2 training regimes
py -3.13 run_zou.py             # 3 models x 2 training regimes
py -3.13 run_plasclass.py       # one logistic regression (balanced regime only)
py -3.13 summarize.py           # results/summary.md
py -3.13 capture_environment.py # results/environment.json (hardware, versions, checksums)
```

Requirements: Python 3.13 with `numpy`, `pandas`, `scikit-learn`, `torch` (CPU is
enough), `pyyaml`. Everything is driven by `config.yaml`; all runs are seeded.
The exact versions, hardware, upstream commits and dataset checksums behind the
published numbers are in `results/environment.json` and REPORT.md section 3.
Budget ~40 GB of free disk and ~80 minutes on a 4-core CPU.

## Layout

```
baselines/           data access, k-mer features, model definitions, metrics
prepare_features.py  FASTA -> uint8 base codes -> uint16 k-mer count memmaps
run_plasflow.py      TF-IDF + DNNClassifier voting ensemble (k = 5, 6, 7)
run_zou.py           standardised canonical 6-mers + 7-layer MLP, three-vote ensemble
run_plasclass.py     canonical k-mer frequencies for k=3..7 + L2 logistic regression
summarize.py         result JSON -> markdown tables
verify_features.py   k-mer counts vs brute force, column order vs KMC, transforms vs sklearn
capture_environment.py  records hardware, versions, upstream commits and checksums
reference/           KMC canonical 6-mer column order shipped with the Zou repository
cache/               generated features (git-ignored, regenerable)
results/             metrics JSON, per-fragment probabilities, summary.md
```

## Two training regimes

Both baselines are trained twice and always evaluated on the same untouched,
naturally imbalanced test split:

- **Regime A (balanced)** — chromosome fragments downsampled 1:1 against plasmid
  fragments, which is the class ratio both papers trained on.
- **Regime B (full)** — the complete training split at its natural 2.345% plasmid
  rate, i.e. the condition a model trained on this dataset actually faces.

PlasClass has regime A only: its own training code always samples an equal number of
fragments per class, and a full-split feature matrix (337,512 x 10,952) needs 14.8 GB,
which does not fit on this machine.

## Memory note

The machine this was developed on has 8 GB RAM, so nothing is held in memory:
features live in `uint16` memmaps, train rows are stored pre-shuffled so that
block-wise reading approximates a full shuffle, and every transform is applied
per minibatch.
