#!/bin/bash -l
#SBATCH --job-name=plasmidhunter
#SBATCH --output=server/logs/plasmidhunter-%A_%a.out
#SBATCH --error=server/logs/plasmidhunter-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-116%5
#
# One array task per assembly shard, the same structure and the same 5 x 8 = 40
# CPU reasoning as 11_platon.sh. Like PLASMe, PlasmidHunter classifies each
# contig independently and does not need the assembly grouping; it is used anyway
# so that all four published tools see byte-identical input.
#
# PlasmidHunter is the only one of the four whose output is a calibrated
# probability (a Gaussian naive Bayes posterior over a gene-content profile), so
# it is the only one that lands on a ROC curve without a ranking heuristic.
#
# Two upstream behaviours the wrapper has to work around:
#
#   1. `-o` is DELETED (shutil.rmtree) at the start of every run, so the output
#      directory must be a disposable stage, never the final one.
#   2. It downloads its DIAMOND database into its own site-packages on first use.
#      116 tasks starting at once would each try that, and a compute node may
#      have no internet at all - so 00_setup_envs.sh fetches it in advance and
#      this script refuses to start if it is not already there.
#
# Resumable: a shard whose output directory carries a .done sentinel is skipped.

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

OUT="$TOOL_OUT/plasmidhunter/$SHARD"
FASTA="$SHARD_DIR/$SHARD.fasta"

banner "plasmidhunter | shard $SHARD (task $TASK)"

if [[ -f "$OUT/.done" ]]; then
  echo "SKIP: $OUT/.done exists ($(( $(wc -l < "$OUT/predictions.tsv" 2>/dev/null || echo 1) - 1 )) scored contigs)"
  exit 0
fi
if [[ ! -s "$FASTA" ]]; then
  echo "FATAL: missing or empty shard FASTA $FASTA - run 10_shard.sh" >&2
  exit 1
fi

cuaim_modules
cuaim_activate "$ENV_PH"

# The tool resolves its database against its own installed package, so ask the
# package where that is rather than trusting the path guessed in env.sh - a
# different Python minor version moves site-packages.
PH_PKG="$(python -c 'import os, PlasmidHunter; print(os.path.dirname(PlasmidHunter.__file__))')"
PH_DBFILE="$PH_PKG/database/database.dmnd"
if [[ ! -s "$PH_DBFILE" ]]; then
  echo "FATAL: PlasmidHunter's DIAMOND database is not installed at" >&2
  echo "       $PH_DBFILE" >&2
  echo "       It downloads this on first use, which 116 parallel tasks must not" >&2
  echo "       do (and a compute node may have no internet). Run:" >&2
  echo "         bash server/00_setup_envs.sh" >&2
  exit 1
fi

N_CONTIGS="$(shard_field "$MANIFEST" "$SHARD" n_fragments)"
N_BP="$(shard_field "$MANIFEST" "$SHARD" bp)"
echo "  input    $FASTA ($N_CONTIGS contigs, $N_BP bp, $(du -h "$FASTA" | cut -f1))"
echo "  database $PH_DBFILE ($(du -h "$PH_DBFILE" | cut -f1))"
echo "  threads  $CPUS_PER_TOOL  (prodigal workers and DIAMOND threads alike)"

STAGE="$TOOL_OUT/plasmidhunter/.stage-$SHARD-${SLURM_JOB_ID:-$$}"
rm -rf "$STAGE"
mkdir -p "$(dirname "$STAGE")"

# -o is the directory the tool wipes and recreates, so it is given the stage.
# TMPDIR already points inside $PROJECT (env.sh), which keeps the split FASTAs
# and the merged protein file off the small home quota.
RC=0
timed_tool "$STAGE.runtime.json" plasmidhunter "$SHARD" "$N_CONTIGS" "$N_BP" -- \
    --infile "$FASTA" \
    --cpu "$CPUS_PER_TOOL" \
    --outdir "$STAGE" || RC=$?

if [[ $RC -ne 0 ]]; then
  echo "FATAL: plasmidhunter exited $RC on shard $SHARD" >&2
  ls -la "$STAGE" >&2 || true
  exit 1
fi
if [[ ! -s "$STAGE/predictions.tsv" ]]; then
  echo "FATAL: plasmidhunter produced no predictions.tsv in $STAGE" >&2
  ls -la "$STAGE" >&2 || true
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
mv "$STAGE" "$OUT"
[[ -f "$STAGE.runtime.json" ]] && mv -f "$STAGE.runtime.json" "$OUT/runtime.json"

SCORED=$(( $(wc -l < "$OUT/predictions.tsv") - 1 ))
echo "  scored contigs: $SCORED / $N_CONTIGS"
if [[ "$SCORED" -lt "$N_CONTIGS" ]]; then
  echo "  note: $(( N_CONTIGS - SCORED )) contig(s) produced no DIAMOND gene hit and"
  echo "        carry no row; collect_plasmidhunter.py re-joins them as unclassified"
fi
touch "$OUT/.done"
banner "plasmidhunter done | shard $SHARD"
