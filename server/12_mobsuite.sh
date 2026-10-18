#!/bin/bash
#SBATCH --job-name=mobsuite
#SBATCH --output=server/logs/mobsuite-%A_%a.out
#SBATCH --error=server/logs/mobsuite-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-116%5
#
# One array task per assembly shard, same structure and the same reasoning as
# 11_platon.sh: 5 x 8 CPUs stays inside the documented 40-CPU cpu40h limit
# whether that cap is per job or aggregate, and a shard whose contig_report.txt
# already exists is skipped so the array is resumable.
#
# mob_recon reconstructs plasmids from the contigs of ONE isolate, which is why
# the shard is an assembly rather than an arbitrary batch of fragments. Note it
# discards contigs below its --min_length (1000 bp by default); the collector
# re-joins those as unclassified instead of dropping them from the denominator.

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

OUT="$TOOL_OUT/mobsuite/$SHARD"
FASTA="$SHARD_DIR/$SHARD.fasta"

banner "mob_recon | shard $SHARD (task $TASK)"

if [[ -s "$OUT/contig_report.txt" ]]; then
  echo "SKIP: $OUT/contig_report.txt already exists"
  exit 0
fi
if [[ ! -s "$FASTA" ]]; then
  echo "FATAL: missing or empty shard FASTA $FASTA - run 10_shard.sh" >&2
  exit 1
fi

cuaim_modules
cuaim_activate "$ENV_MOB"
require_dirs "$MOB_DB"
mkdir -p "$(dirname "$OUT")"

# The flag that points mob_recon at a database directory has been renamed between
# releases, and the upstream README does not document every option, so discover
# it instead of guessing. 00_setup_envs.sh saved the full --help for reference.
MOB_HELP="$(mob_recon --help 2>&1 || true)"
DB_ARGS=()
for f in --database_directory -d; do
  if grep -q -- "$f" <<<"$MOB_HELP"; then
    DB_ARGS=("$f" "$MOB_DB")
    break
  fi
done
if [[ ${#DB_ARGS[@]} -eq 0 ]]; then
  echo "  note: mob_recon exposes no database-directory flag; relying on its default"
  echo "        location. If it tries to download the database on a compute node"
  echo "        without internet, this task will fail here."
fi

THREAD_ARGS=()
for f in --num_threads -n; do
  if grep -q -- "$f" <<<"$MOB_HELP"; then
    THREAD_ARGS=("$f" "$CPUS_PER_TOOL")
    break
  fi
done

MIN_LEN="$(awk -F': *' '/min_length/ {print $2; exit}' "$PROJECT/tools-baseline/config.yaml" 2>/dev/null || echo 1000)"
LEN_ARGS=()
if grep -q -- "--min_length" <<<"$MOB_HELP"; then
  LEN_ARGS=(--min_length "${MIN_LEN:-1000}")
fi

echo "  input    $FASTA ($(grep -c '^>' "$FASTA") contigs, $(du -h "$FASTA" | cut -f1))"
echo "  database ${DB_ARGS[*]:-<tool default>}"
echo "  threads  ${THREAD_ARGS[*]:-<tool default>}"
echo "  min len  ${LEN_ARGS[*]:-<tool default>}"

# Staged, then moved on success, so an interrupted task cannot leave a partial
# contig_report.txt that the skip check above would treat as finished.
STAGE="$TOOL_OUT/mobsuite/.stage-$SHARD-${SLURM_JOB_ID:-$$}"
rm -rf "$STAGE"

run_tool mob_recon \
     --infile "$FASTA" \
     --outdir "$STAGE" \
     "${DB_ARGS[@]}" "${THREAD_ARGS[@]}" "${LEN_ARGS[@]}" \
     --force

if [[ ! -s "$STAGE/contig_report.txt" ]]; then
  echo "FATAL: mob_recon produced no contig_report.txt in $STAGE" >&2
  ls -la "$STAGE" >&2 || true
  exit 1
fi

rm -rf "$OUT"
mv "$STAGE" "$OUT"

echo "  rows in contig_report: $(( $(wc -l < "$OUT/contig_report.txt") - 1 ))"
banner "mob_recon done | shard $SHARD"
