# RefSeq Plasmid vs Chromosome Dataset Pipeline

A reproducible data-preparation pipeline that turns NCBI RefSeq bacterial
assembly metadata into an ML-ready dataset of **5-kb genomic fragments**
labelled *chromosome* (0) or *plasmid* (1).

Every fragment is traceable back to its parent replicon, its parent assembly,
and the exact NCBI accession it came from.

---

## 1. Quick start

```bash
python run_pipeline.py                 # all 7 steps
python run_pipeline.py --steps 3 4     # only steps 3 and 4
python run_pipeline.py --from 5        # step 5 onwards
```

Requirements: Python 3.11+, `pandas`, `numpy`, `pyyaml` (Biopython is optional;
FASTA parsing is implemented directly for streaming performance). No external
bioinformatics binaries are needed — sequence similarity uses a MinHash
implementation in `pipeline/step05_cluster.py`.

`assembly_summary.txt` must be present at the project root or at
`data/raw/assembly_summary.txt`. Source:
<https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt>

---

## 2. Pipeline

| Step | Module | Output |
|------|--------|--------|
| 1 | `step01_filter_assemblies.py` | `data/metadata/assemblies.csv`, `assemblies_filtered_all.csv` |
| 2 | `step02_download_fasta.py` | `data/fasta/<GCF_...>/*_genomic.fna.gz`, `download_status.csv` |
| 3 | `step03_parse_replicons.py` | `data/metadata/replicons.csv`, `reports/unknown_replicons.csv` |
| 4 | `step04_qc.py` | `reports/data_qc.csv`, `reports/data_qc.txt`, `qc_per_assembly.csv` |
| 5 | `step05_cluster.py` | `data/metadata/clusters.csv`, `reports/similarity_*.{csv,txt}` |
| 6 | `step06_split.py` | `data/split/{train,validation,test}_replicons.csv` |
| 7 | `step07_fragments.py` | `data/dataset/{train,validation,test}.{fasta,csv}`, `fragments.csv` |

Cross-cutting outputs: `reports/pipeline_stages.csv` (records *in → out → lost*
for every filtering stage), `reports/run_manifest.json` (full provenance:
source URL, snapshot date, parameters, software versions, per-step counts),
`logs/pipeline.log`.

---

## 3. Methodological decisions

Each of these is configurable in `config.yaml` and recorded in the run
manifest.

### 3.1 Header detection

`assembly_summary.txt` starts with a comment line, then a header line that is
*also* a comment (`#assembly_accession ...`). The parser scans for the line
whose content begins with `assembly_accession` and uses it as the header.
`pd.read_csv(..., comment="#")` is deliberately **not** used, because it would
delete the header along with the comments.

### 3.2 Assembly filtering

`assembly_level == "Complete Genome"` and `version_status == "latest"`.
Complete genomes are required because replicon identity (which sequence *is*
a plasmid) is only reliable in finished assemblies.

`refseq_category` is **not** filtered. Restricting to reference/representative
genomes would collapse biological diversity and systematically bias plasmid
content, which is precisely the signal being modelled.

Additionally dropped: non-RefSeq accessions (GenBank `GCA_*`), rows without a
usable `ftp_path`, and rows NCBI itself flags in `excluded_from_refseq`. All
original metadata columns are preserved in the output.

### 3.3 Sampling (computational-cost decision)

RefSeq currently holds ~66k Complete Genome/latest bacterial assemblies
(~100 GB of FASTA). The default configuration samples **500 assemblies**,
at most **one per `species_taxid`**, with `random_seed: 42`.

*Consequence:* the dataset is species-balanced rather than
RefSeq-frequency-weighted. This suppresses the massive over-representation of
clinical species (*E. coli*, *S. enterica*, *K. pneumoniae*) that would
otherwise dominate, at the cost of not reflecting RefSeq's real composition.
Set `sampling.enabled: false` to use every filtered assembly, or raise
`max_assemblies` / `max_per_species`.

### 3.4 Replicon classification

FASTA record order is **never** used. Classification precedence:

1. **`*_assembly_report.txt`** — NCBI's own `Sequence-Role` and
   `Assigned-Molecule-Location/Type` fields (authoritative);
2. **FASTA definition line keywords** — `plasmid`/`megaplasmid` →
   plasmid; `chromosome`/`chromid` → chromosome; phage, prophage, virus,
   mitochondrion, plastid → `unknown`;
3. a single-record complete-genome assembly with no plasmid keyword is treated
   as the chromosome (toggle:
   `replicons.single_record_complete_genome_is_chromosome`);
4. otherwise → **`unknown`**.

`unknown` replicons are never coerced into a class. They stay in
`replicons.csv`, are listed in `reports/unknown_replicons.csv`, and are
excluded only from the binary dataset. Every replicon row carries
`classification_source` and `classification_evidence` so any label can be
audited.

### 3.5 Redundancy analysis — before splitting

MinHash sketches (canonical k-mers, `k=21`, 2000 hashes) give a Mash-style ANI
estimate; candidate pairs come from an inverted index over sketch hashes.
Replicons are linked when **either**

```
estimated ANI >= 0.95   AND   min(len)/max(len) >= 0.90     # near-identical
containment of the shorter in the longer >= 0.90            # subsequence
```

and single-linkage union-find turns links into clusters. Identical sequences
(SHA-256) are always merged.

The containment rule exists because Jaccard collapses for unequal lengths: the
run found a 39-kb plasmid at 99.4% ANI to a 72-kb plasmid whose length ratio
(0.54) would have let it cross splits. Containment is only trusted when the
shared sketch range holds at least `min_containment_sample` (50) hashes — a
9.6-kb plasmid against a 4.9-Mbp chromosome otherwise scored containment 1.0
off a one- or two-hash sample. That guard also means containment is
**under-powered for extremely unequal pairs**; establishing whether a small
element really sits inside a chromosome needs an alignment, and the pipeline
claims nothing either way.

> **This threshold is a chosen experimental criterion, not a universally
> correct biological standard.** A sketch is not an alignment, so the length
> ratio is only a proxy for coverage. Changing the threshold changes the split;
> that is why it lives in `config.yaml` and is echoed into the manifest and
> `reports/similarity_clusters.txt`.

Chromosomes and plasmids **are** compared against each other. Plasmid and
chromosomal sequence are not always dissimilar — horizontal gene transfer,
mobile elements, integrated prophage and AMR cassettes create genuine shared
sequence. Any cross-type link is reported in `reports/similarity_pairs.csv`
rather than assumed impossible.

### 3.6 Splitting

Splitting happens **at the cluster level, before any fragment exists**
(70 / 15 / 15, seed 42). All fragments of a replicon therefore share one split,
and near-identical parent replicons cannot straddle train and test.

Clusters are bucketed by the replicon types they contain (chromosome-only /
plasmid-only / mixed) and each bucket is filled towards the target ratios
independently, weighted by the number of fragments each cluster will produce.
This keeps class balance comparable across splits without resampling the
corpus.

Note that an assembly's chromosome and its plasmids may legitimately land in
different splits: the split unit is the *replicon cluster*, not the assembly.
Because a plasmid and its host chromosome are different molecules carrying
different labels, this is not fragment-level leakage — and where they do share
sequence (HGT, mobile elements), the shared region is labelled 0 in one split
and 1 in the other, which penalises a model rather than flattering it. The
count is reported in the manifest so the choice stays visible, and
`split.keep_assembly_together: true` enforces the stricter rule that no strain
contributes sequence to more than one split.

### 3.7 Fragment generation

`fragment_length = 5000`, strategy **`non_overlapping`** (step = 5000), trailing
remainder shorter than 5000 bp dropped, replicons shorter than 5000 bp produce
no fragments. Fragments with more than 5% non-ACGT characters are dropped and
counted.

These are **5-kb genomic fragments**, not "real contigs". A slice of a finished
replicon has no assembly errors, no coverage-driven breakpoints and no
chimeras. Simulated contigs would be a separate, clearly distinguished
experiment.

### 3.8 Class balance

The natural RefSeq imbalance is preserved and reported, not engineered away —
chromosomes contribute vastly more 5-kb fragments than plasmids do
(`fragments.rebalance_classes: false`). Handle it at training time with class
weights, focal loss, or a balanced sampler; the per-split plasmid fraction is
in `reports/dataset_summary.csv`.

---

## 4. Output schema

`data/dataset/{split}.csv` — one row per fragment:

```
fragment_id, replicon_id, assembly_accession, sequence_id, organism_name,
taxid, replicon_type, label, start, end, length, split,
species_taxid, cluster_id, gc_content, non_acgt_fraction
```

`data/dataset/{split}.fasta` — the FASTA id is exactly the `fragment_id`:

```
>fragment_00000001
ATCG...
```

Step 7 fails loudly if the FASTA and CSV id sets disagree, if a replicon's
fragments span multiple splits, or if any `fragment_id` repeats.

### Population characterisation

```bash
python analyze_population.py
```

Describes the **whole filtered population** (all 66,230 assemblies), not just
the sampled subset. It resolves the domain of every distinct `species_taxid`
against NCBI Taxonomy — necessary because the `group` column reads "bacteria"
for every row purely because of the FTP directory it came from — and derives
chromosome/plasmid counts from NCBI's bulk `GENOME_REPORTS/prokaryotes.txt`
instead of downloading ~100 GB of FASTA. Writes
`reports/population_characterisation.txt` and `reports/population_replicons.csv`;
the taxonomy lookups are cached in `data/metadata/taxonomy_species.csv`.

### Independent verification

```bash
python verify_dataset.py
```

Re-derives its checks from the output files and the original genomes rather
than reusing the pipeline's bookkeeping, so a bug in the pipeline cannot hide
itself. It checks id uniqueness and FASTA/CSV agreement, that no replicon or
cluster crosses a split boundary, that labels agree with `replicon_type` and
with the parent replicon, that no `unknown` replicon leaked in, fragment
lengths and coordinate consistency, non-overlap, split proportions and class
balance, that every eligible replicon produced fragments — and, most
importantly, that randomly sampled fragment sequences byte-match the slice of
their source genome at the recorded coordinates.

---

## 5. Reproducibility

The same `config.yaml` plus the same `assembly_summary.txt` snapshot reproduces
the same dataset: all sampling and splitting draws come from
`numpy.random.default_rng(random_seed)` after a deterministic sort, so neither
file order nor thread scheduling can affect the result.

`reports/run_manifest.json` records the source URL, the snapshot's download
date and size, filtering criteria, sampling parameters, download counts,
classification-source counts, similarity method and thresholds, split ratios,
fragment strategy, the random seed, and the Python/pandas/numpy versions used.

Caveat: RefSeq is a moving target. Reproducing a run on a *newer*
`assembly_summary.txt` will select different assemblies. Archive the snapshot
(`data/raw/assembly_summary.txt`) alongside the dataset.
