#!/bin/bash -l
#SBATCH --job-name=platon
#SBATCH --output=server/logs/platon-%A_%a.out
#SBATCH --error=server/logs/platon-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-116%5
#
# One array task per assembly shard. submit_all.sh overrides the array range,
# partition, qos and account from env.sh on the command line, because #SBATCH
# directives cannot read variables - the values above are only sane defaults.
#
# Why --array=...%5 : the documented cpu40h QoS grants 40 CPUs, and the manual
# (5.1.2) warns limits may apply to a user's COMBINED activity. 5 tasks x 8 CPUs
# stays inside 40 whether the cap is per job or aggregate.
#
# Why --characterize : it deactivates Platon's filters and reports RDS and every
# hallmark column for EVERY contig. Without it the TSV holds only the contigs
# already classified as plasmid, and neither a continuous score nor the other two
# filter modes could be recovered. The three published modes are then re-derived
# offline by tools-baseline/toolbase/rules.py, and 13_smoke_tools.sh proves that
# re-derivation against a real `--mode accuracy` run before this is trusted.
#
# Resumable: a shard whose TSV already exists is skipped, so a task that hit the
# time limit or a node failure can be resubmitted with the same command.

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

OUT="$TOOL_OUT/platon/$SHARD"
FASTA="$SHARD_DIR/$SHARD.fasta"

banner "platon | shard $SHARD (task $TASK)"

if [[ -s "$OUT/$SHARD.tsv" ]]; then
  echo "SKIP: $OUT/$SHARD.tsv already exists ($(wc -l < "$OUT/$SHARD.tsv") lines)"
  exit 0
fi
if [[ ! -s "$FASTA" ]]; then
  echo "FATAL: missing or empty shard FASTA $FASTA - run 10_shard.sh" >&2
  exit 1
fi

cuaim_modules
cuaim_activate "$ENV_PLATON"
require_dirs "$PLATON_DB"
mkdir -p "$OUT"

echo "  input    $FASTA ($(grep -c '^>' "$FASTA") contigs, $(du -h "$FASTA" | cut -f1))"
echo "  database $PLATON_DB"
echo "  threads  $CPUS_PER_TOOL"

# Write into a scratch directory and move it into place only on success, so an
# interrupted task never leaves a half-written TSV that the skip check above
# would mistake for a finished shard.
STAGE="$TOOL_OUT/platon/.stage-$SHARD-${SLURM_JOB_ID:-$$}"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# Called directly rather than through srun: run_tool is a shell function (it
# switches between conda and a Singularity image on one variable) and srun execs
# a binary, so it cannot launch one. A single-task array element needs no job
# step of its own - the manual's own batch example in 4.4.5 calls its program
# directly too.
run_tool platon \
     --db "$PLATON_DB" \
     --output "$STAGE" \
     --prefix "$SHARD" \
     --characterize \
     --threads "$CPUS_PER_TOOL" \
     --verbose \
     "$FASTA"

if [[ ! -s "$STAGE/$SHARD.tsv" ]]; then
  echo "FATAL: platon produced no $SHARD.tsv in $STAGE" >&2
  ls -la "$STAGE" >&2 || true
  exit 1
fi

mkdir -p "$OUT"
mv -f "$STAGE"/* "$OUT"/
rmdir "$STAGE"

echo "  rows in TSV: $(( $(wc -l < "$OUT/$SHARD.tsv") - 1 ))"
banner "platon done | shard $SHARD"
