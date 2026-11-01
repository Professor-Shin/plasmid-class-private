#!/bin/bash -l
#SBATCH --job-name=plasme
#SBATCH --output=server/logs/plasme-%A_%a.out
#SBATCH --error=server/logs/plasme-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-116%5
#
# One array task per assembly shard, the same structure and the same 5 x 8 = 40
# CPU reasoning as 11_platon.sh. PLASMe does not need the assembly grouping the
# way Platon and mob_recon do - it classifies each contig on its own - but using
# the same shards keeps the input to all four published tools byte-identical,
# which is the only way the per-tool cost numbers mean anything.
#
# PLASMe runs on CPU here. Its second stage is a Transformer and would be faster
# on a GPU, but its first stage is BLASTn against a 12 GB plasmid database and
# its third is DIAMOND, both CPU-bound, and the GPU QoS caps the wall clock at
# 8 hours against cpu40h's 4 days. Running it beside Platon and mob_recon on the
# same partition also makes the measured time and memory directly comparable.
#
# WHY THE TEMPORARY DIRECTORY IS KEPT: PLASMe's visible output is a FASTA of the
# contigs it called plasmid plus a `_report.csv` listing only those contigs -
# neither carries a denominator or a rejected contig. The file with one row per
# ALIGNED contig, and with the BLASTn identity/coverage and the Transformer
# probability on it, is <temp>/PLASMe_candidate.csv. All three published preset
# modes are threshold triples over those three numbers, so keeping that one file
# turns a single run into all three modes plus a continuous score, exactly as
# --characterize does for Platon.
#
# Resumable: a shard whose output directory carries a .done sentinel is skipped.
# The sentinel, not the presence of the CSV, is the test - PLASMe calls exit(0)
# on several of its own error paths, so "the file exists" is not proof it ran.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

MANIFEST="$SHARD_DIR/manifest.csv"
TASK="${SLURM_ARRAY_TASK_ID:-1}"
SHARD="$(shard_for_task "$MANIFEST" "$TASK")"
if [[ -z "$SHARD" ]]; then
  echo "FATAL: no shard on manifest line $((TASK + 1)) of $MANIFEST" >&2
  exit 1
fi

OUT="$TOOL_OUT/plasme/$SHARD"
FASTA="$SHARD_DIR/$SHARD.fasta"

banner "plasme | shard $SHARD (task $TASK)"

if [[ -f "$OUT/.done" ]]; then
  echo "SKIP: $OUT/.done exists ($(wc -l < "$OUT/PLASMe_candidate.csv" 2>/dev/null || echo 0) candidate rows)"
  exit 0
fi
if [[ ! -s "$FASTA" ]]; then
  echo "FATAL: missing or empty shard FASTA $FASTA - run 10_shard.sh" >&2
  exit 1
fi

cuaim_modules
cuaim_activate "$ENV_PLASME"
require_dirs "$PLASME_DB"
if [[ ! -f "$PLASME_DB/plsdb_Mar30.dmnd" ]]; then
  echo "FATAL: $PLASME_DB holds no plsdb_Mar30.dmnd, so the database was never" >&2
  echo "       built. Re-run 00_setup_envs.sh; do NOT let a compute node build it" >&2
  echo "       (116 tasks would each try to build the same 12 GB index)." >&2
  exit 1
fi

N_CONTIGS="$(shard_field "$MANIFEST" "$SHARD" n_fragments)"
N_BP="$(shard_field "$MANIFEST" "$SHARD" bp)"
echo "  input    $FASTA ($N_CONTIGS contigs, $N_BP bp, $(du -h "$FASTA" | cut -f1))"
echo "  database $PLASME_DB"
echo "  threads  $CPUS_PER_TOOL"
echo "  device   CPU (torch sees no GPU on the $PART_CPU partition, by design)"

# Staged, then moved on success. PLASMe refuses to start when its output file
# already exists ("Please rename the output file"), so the stage must be gone
# before it runs, not merely empty.
STAGE="$TOOL_OUT/plasme/.stage-$SHARD-${SLURM_JOB_ID:-$$}"
rm -rf "$STAGE"
mkdir -p "$STAGE/temp"

# PLASMe writes its own temp files relative to --temp and nothing relative to the
# working directory, but it does resolve its DEFAULT database against the cwd, so
# the run happens inside the stage to keep any stray file out of the repo.
RC=0
( cd "$STAGE" && timed_tool "$STAGE/runtime.json" plasme "$SHARD" "$N_CONTIGS" "$N_BP" -- \
    "$FASTA" "$STAGE/plasmid.fna" \
    --database "$PLASME_DB" \
    --thread "$CPUS_PER_TOOL" \
    --temp "$STAGE/temp" ) || RC=$?

CAND="$STAGE/temp/PLASMe_candidate.csv"
if [[ $RC -ne 0 ]]; then
  echo "FATAL: PLASMe exited $RC on shard $SHARD" >&2
  ls -la "$STAGE" "$STAGE/temp" >&2 || true
  exit 1
fi
if [[ ! -s "$CAND" ]]; then
  # PLASMe exits 0 when its database is missing or its output path is taken, so
  # the absence of this file - not the exit code - is what catches a non-run.
  echo "FATAL: PLASMe produced no $CAND (it exits 0 on several error paths, so" >&2
  echo "       check the output above for 'Database directory does not exist' or" >&2
  echo "       'The output file already exists')" >&2
  ls -la "$STAGE" "$STAGE/temp" >&2 || true
  exit 1
fi

mkdir -p "$OUT"
cp -f "$CAND" "$OUT/PLASMe_candidate.csv"
[[ -f "$STAGE/plasmid.fna" ]]         && cp -f "$STAGE/plasmid.fna" "$OUT/"
[[ -f "$STAGE/plasmid.fna_report.csv" ]] && cp -f "$STAGE/plasmid.fna_report.csv" "$OUT/"
[[ -f "$STAGE/runtime.json" ]]        && cp -f "$STAGE/runtime.json" "$OUT/"

# The BLASTn table against a 12 GB plasmid database is the one intermediate worth
# keeping for an audit and the one large enough to matter; everything else is
# re-derivable from it in seconds. 116 shards x an uncompressed copy would be the
# largest thing this pipeline writes.
if [[ -s "$STAGE/temp/blastn.csv" ]]; then
  gzip -c "$STAGE/temp/blastn.csv" > "$OUT/blastn.csv.gz"
fi
if [[ -n "${PLASME_KEEP_TEMP:-}" ]]; then
  cp -r "$STAGE/temp" "$OUT/temp"
  echo "  PLASME_KEEP_TEMP set: kept the whole temp directory"
fi
rm -rf "$STAGE"

ROWS=$(( $(wc -l < "$OUT/PLASMe_candidate.csv") - 1 ))
CALLED=$(( $(wc -l < "$OUT/plasmid.fna_report.csv" 2>/dev/null || echo 1) - 1 ))
echo "  aligned contigs: $ROWS / $N_CONTIGS"
echo "  called plasmid at the CLI defaults (i>=0.9, c>=0.9, p>0.5): $CALLED"
echo "  (the other two preset modes are re-derived offline from the same table)"
touch "$OUT/.done"
banner "plasme done | shard $SHARD"
