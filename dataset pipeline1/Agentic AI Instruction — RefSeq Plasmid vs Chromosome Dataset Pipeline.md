# Task: Build a Reproducible RefSeq Dataset Pipeline for Plasmid vs Chromosome Classification

You are an agentic AI responsible for designing and implementing a reproducible bioinformatics data-preparation pipeline for a research project on **bacterial plasmid vs chromosome sequence classification using machine learning**.

The goal is to transform NCBI RefSeq assembly metadata and genomic FASTA files into a clean, traceable dataset suitable for ML experiments.

---

## 1. Overall Objective

Build the following pipeline:

```text
NCBI RefSeq Bacteria
        │
        ▼
assembly_summary.txt
        │
        ▼
Filter assemblies
        │
        ├── RefSeq
        ├── Bacteria
        ├── Complete Genome
        └── latest version
        │
        ▼
Download genomic FASTA
        │
        ▼
Parse biological replicons
        │
        ├── chromosome
        └── plasmid
        │
        ▼
Create replicon metadata
        │
        ▼
Deduplicate / identify highly similar replicons
        │
        ▼
Split at parent replicon / cluster level
        │
        ├── train
        ├── validation
        └── test
        │
        ▼
Generate 5-kb genomic fragments
        │
        ▼
Final ML dataset
```

The final dataset must be reproducible and every fragment must be traceable back to its original NCBI assembly and biological replicon.

---

# 2. Input

The primary metadata file is:

```text
assembly_summary.txt
```

Expected source:

```text
https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt
```

The pipeline should NOT assume that the first row of the file is the header.

The header begins with:

```text
# assembly_accession
```

Therefore, detect the header row robustly before loading the table.

Do NOT use:

```python
pd.read_csv(..., comment="#")
```

because this removes the actual header.

The pipeline should correctly parse the NCBI tab-delimited assembly summary format.

---

# 3. Assembly Filtering

Initially select assemblies satisfying:

```text
assembly_level == "Complete Genome"
version_status == "latest"
```

The dataset should primarily use **RefSeq bacterial assemblies**.

Do not restrict the dataset only to `reference genome` or `representative genome`, because this can unnecessarily reduce biological diversity.

Preserve the original metadata fields even if they are not immediately used.

At minimum, retain:

```text
assembly_accession
bioproject
biosample
organism_name
taxid
species_taxid
refseq_category
assembly_level
version_status
seq_rel_date
ftp_path
```

Create:

```text
metadata/assemblies.csv
```

This file represents the set of assemblies selected for the study.

---

# 4. Download Genomic FASTA

For each selected assembly, use its `ftp_path` to locate:

```text
*_genomic.fna.gz
```

Do not merge all FASTA files into a single file at this stage.

Store files by assembly:

```text
data/
└── fasta/
    ├── GCF_xxxxx/
    │   └── *_genomic.fna.gz
    ├── GCF_yyyyy/
    │   └── *_genomic.fna.gz
    └── ...
```

The pipeline should support:

- resumable downloads
- skipping files that already exist
- compressed FASTA files
- logging failed downloads
- retrying failed downloads where appropriate

Do not redownload existing files unnecessarily.

---

# 5. Parse FASTA Records

Parse every genomic FASTA file.

Important:

A FASTA file may contain multiple biological replicons:

```text
assembly
├── chromosome
├── chromosome
├── plasmid
├── plasmid
└── plasmid
```

Do NOT assume:

```text
first FASTA record = chromosome
second FASTA record = plasmid
```

Replicon type must be determined using NCBI metadata and/or FASTA definition lines.

The classification must distinguish at least:

```text
chromosome
plasmid
unknown
```

Do not silently force ambiguous records into chromosome or plasmid.

If a record cannot be confidently classified, mark it as:

```text
unknown
```

and report it.

---

# 6. Create Replicon Metadata

Create:

```text
metadata/replicons.csv
```

Each row should represent **one biological replicon**, not one fragment.

Recommended columns:

```text
replicon_id
assembly_accession
sequence_id
organism_name
taxid
species_taxid
replicon_type
label
length
source_fasta
```

Use:

```text
chromosome → label 0
plasmid    → label 1
unknown    → exclude from binary ML dataset
```

Example:

```text
replicon_id,assembly_accession,organism_name,replicon_type,label,length
GCF_xxx_chr1,GCF_xxx,Escherichia coli,chromosome,0,4600000
GCF_xxx_p1,GCF_xxx,Escherichia coli,plasmid,1,90000
```

Every replicon must retain its parent:

```text
assembly_accession
```

This is essential for preventing data leakage.

---

# 7. Data Quality Checks

Before generating fragments, perform quality-control checks.

Report:

1. Number of selected assemblies
2. Number of assemblies successfully downloaded
3. Number of failed downloads
4. Number of FASTA records
5. Number of chromosomes
6. Number of plasmids
7. Number of unknown replicons
8. Length distribution of chromosomes
9. Length distribution of plasmids
10. Number of assemblies containing both chromosome and plasmid
11. Number of assemblies containing multiple plasmids
12. Duplicate sequence counts
13. Invalid/non-ACGT characters
14. Extremely short replicons

Do not silently discard records.

Create a QC report such as:

```text
reports/data_qc.csv
```

and/or:

```text
reports/data_qc.txt
```

---

# 8. Deduplication and Sequence Similarity

Sequence redundancy must be considered before train/validation/test splitting.

Important:

Do NOT randomly split generated fragments.

Bad:

```text
complete genomes
      ↓
generate fragments
      ↓
random train/test split
```

This can cause severe data leakage because fragments from the same biological replicon may occur in both train and test.

Instead:

```text
parent replicons
      ↓
deduplication / similarity analysis
      ↓
split parent replicons
      ↓
generate fragments
```

At minimum, identical parent sequences must never be split across train and test.

If feasible, perform sequence similarity clustering before splitting.

The similarity threshold must be treated as a methodological parameter, not an unquestioned universal standard.

If using a threshold such as 90% identity over 90% coverage, explicitly document that it is a chosen experimental criterion rather than a universally correct biological threshold.

Do not claim that plasmid and chromosome sequences are always dissimilar.

Shared sequence caused by horizontal gene transfer, mobile elements, phage-related regions, antimicrobial-resistance genes, and homologous regions can occur.

---

# 9. Train / Validation / Test Split

Split at the **parent replicon level**.

Never split individual fragments independently.

Preferred structure:

```text
Replicon A → TRAIN
Replicon B → TRAIN
Replicon C → VALIDATION
Replicon D → TEST
```

All fragments generated from one parent replicon must belong to the same split.

If clustering is performed:

```text
sequence clusters
      ↓
split clusters
      ↓
generate fragments
```

This is preferred because highly similar replicons should not be distributed across different splits.

Use an approximately:

```text
70% train
15% validation
15% test
```

split unless there is a strong methodological reason to use another ratio.

The split must preserve reasonable class balance.

Also report the number of:

- assemblies
- replicons
- chromosomes
- plasmids
- fragments

in each split.

---

# 10. Generate 5-kb Genomic Fragments

The initial ML dataset should use:

```text
fragment length = 5000 bp
```

These sequences should be described as:

> 5-kb genomic fragments

Do NOT call them "real contigs".

A sequence created by slicing a complete chromosome/plasmid is not a real assembly contig.

If the experiment later requires simulated contigs, implement that as a separate experiment and clearly distinguish it from simple fixed-length genomic fragmentation.

---

# 11. Fragment Generation

For every parent replicon:

```text
length >= 5000 bp
```

generate 5-kb fragments.

Preserve metadata:

```text
fragment_id
replicon_id
assembly_accession
sequence_id
organism_name
taxid
replicon_type
label
start
end
length
split
```

Example:

```text
fragment_000001
GCF_xxx_chr1
GCF_xxx
CPxxxx
Escherichia coli
562
chromosome
0
0
5000
5000
train
```

Avoid generating unnecessary overlapping fragments unless the experimental design explicitly requires them.

The fragment-generation strategy must be documented, for example:

```text
non-overlapping
```

or:

```text
random sampling
```

or:

```text
sliding window
```

Do not silently choose a strategy.

---

# 12. Final Dataset

Create:

```text
dataset/
├── train.fasta
├── train.csv
├── validation.fasta
├── validation.csv
├── test.fasta
└── test.csv
```

FASTA:

```text
>fragment_000001
ATCG...
```

CSV:

```text
fragment_id
replicon_id
assembly_accession
sequence_id
organism_name
taxid
replicon_type
label
start
end
length
split
```

The FASTA ID must exactly match the `fragment_id` in the CSV.

---

# 13. Reproducibility Requirements

The pipeline must save:

```text
config.yaml
```

containing all important parameters:

```yaml
assembly_level: "Complete Genome"
version_status: "latest"
fragment_length: 5000
train_ratio: 0.70
validation_ratio: 0.15
test_ratio: 0.15
random_seed: 42
```

Also record:

- NCBI download date
- source URL
- number of assemblies
- filtering criteria
- software versions
- similarity/clustering method
- similarity thresholds
- fragment-generation method
- random seed

The same configuration should reproduce the same dataset.

---

# 14. Recommended Directory Structure

Use:

```text
data/
│
├── raw/
│   └── assembly_summary.txt
│
├── metadata/
│   ├── assemblies.csv
│   ├── replicons.csv
│   └── fragments.csv
│
├── fasta/
│   ├── GCF_xxxxx/
│   ├── GCF_yyyyy/
│   └── ...
│
├── split/
│   ├── train_replicons.csv
│   ├── validation_replicons.csv
│   └── test_replicons.csv
│
└── dataset/
    ├── train.fasta
    ├── train.csv
    ├── validation.fasta
    ├── validation.csv
    ├── test.fasta
    └── test.csv

reports/
├── data_qc.csv
└── data_qc.txt

config.yaml
```

---

# 15. Important Scientific Constraints

These constraints are mandatory.

### Never:

- Randomly split fragments generated from the same parent replicon.
- Assume every FASTA record is a contig.
- Assume FASTA record order indicates chromosome/plasmid.
- Call simple sequence slicing "real contigs".
- Silently discard ambiguous replicons.
- Mix train/test fragments from highly similar parent sequences without documenting it.
- Use metadata from a different assembly as a substitute for the requested assembly.
- Claim a similarity threshold is universally correct without evidence.

### Always:

- Preserve NCBI accession IDs.
- Preserve parent assembly information.
- Preserve sequence IDs.
- Keep biological labels separate from fragment-generation metadata.
- Perform splitting before fragment generation.
- Log every filtering step.
- Make the pipeline reproducible.
- Report data loss at every stage.

---

# 16. Expected Final Workflow

The completed implementation should follow:

```text
assembly_summary.txt
        ↓
robust parsing
        ↓
Complete Genome + latest
        ↓
assemblies.csv
        ↓
download *_genomic.fna.gz
        ↓
parse FASTA
        ↓
identify biological replicons
        ↓
replicons.csv
        ↓
QC
        ↓
deduplicate / cluster
        ↓
split parent replicons
        ↓
generate 5-kb fragments
        ↓
fragments.csv
        ↓
train.fasta + train.csv
validation.fasta + validation.csv
test.fasta + test.csv
        ↓
Final ML-ready dataset
```

---

# 17. Agent Behavior

You are not merely writing code. You are responsible for making scientifically defensible dataset-preparation decisions.

When a methodological decision is ambiguous:

1. Identify the ambiguity.
2. Explain its scientific consequence.
3. Check authoritative NCBI documentation and relevant plasmid-classification literature when appropriate.
4. Choose a reasonable default.
5. Make the choice configurable.
6. Document the decision.

Prefer simple, robust implementations over unnecessarily complicated pipelines.

Do not optimize for the largest possible dataset.

Optimize for:

```text
reproducibility
+
traceability
+
low data leakage
+
biological validity
+
reasonable computational cost
```

The final output should be suitable as the data-preparation component of a research project/paper on bacterial plasmid vs chromosome classification.