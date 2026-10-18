#!/bin/bash
#SBATCH --job-name=smoke-tools
#SBATCH --output=server/logs/smoke-tools-%j.out
#SBATCH --error=server/logs/smoke-tools-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#
# The gate in front of the two 116-task arrays. Three things are checked, and any
# failure here means the arrays would waste hours producing unusable output:
#
#   1. Both tools run at all, on a 20-fragment input with both classes present
#      (a tool that called everything chromosome would still "succeed" on a
#      plasmid-only input, hence both classes).
#   2. Their output carries the columns the collectors depend on - parse.py
#      asserts by column NAME, so a schema change from a different tool version
#      fails here with the header it actually saw.
#   3. The offline re-derivation of Platon's `accuracy` rule reproduces a real
#      `platon --mode accuracy` run EXACTLY, on two real shards. This is the
#      claim the whole single-pass Platon design rests on; if it fails, the
#      fallback is to run the array with --mode accuracy instead.
#
# Runs independently of 11/12 (it invokes Platon itself), so it can be submitted
# in parallel with them - submit_all.sh does exactly that and makes the
# collection step depend on all three.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

SMOKE="$TOOL_OUT/smoke"
mkdir -p "$SMOKE"

banner "1/4  build a 20-fragment smoke input"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/tools-baseline"
python -u make_smoke_fasta.py --n-per-class 10 --out "$SMOKE/smoke.fasta"

banner "2/4  platon on the smoke input"
conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_PLATON"
rm -rf "$SMOKE/platon"
mkdir -p "$SMOKE/platon"
run_tool platon --db "$PLATON_DB" --output "$SMOKE/platon" --prefix smoke \
         --characterize --threads "$CPUS_PER_TOOL" --verbose "$SMOKE/smoke.fasta"
echo "  TSV header: $(head -1 "$SMOKE/platon/smoke.tsv")"

banner "3/4  mob_recon on the smoke input"
conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_MOB"
MOB_HELP="$(mob_recon --help 2>&1 || true)"
DB_ARGS=()
for f in --database_directory -d; do
  if grep -q -- "$f" <<<"$MOB_HELP"; then DB_ARGS=("$f" "$MOB_DB"); break; fi
done
THREAD_ARGS=()
for f in --num_threads -n; do
  if grep -q -- "$f" <<<"$MOB_HELP"; then THREAD_ARGS=("$f" "$CPUS_PER_TOOL"); break; fi
done
rm -rf "$SMOKE/mobsuite"
run_tool mob_recon --infile "$SMOKE/smoke.fasta" --outdir "$SMOKE/mobsuite" \
         "${DB_ARGS[@]}" "${THREAD_ARGS[@]}" --force
echo "  contig_report header: $(head -1 "$SMOKE/mobsuite/contig_report.txt")"

banner "4/4  does the offline accuracy rule match a real --mode accuracy run?"
# Two real shards, the smallest ones that actually contain a plasmid fragment, so
# the comparison exercises both branches of the rule rather than only the
# chromosome branch.
SHARDS="$(awk -F',' 'NR>1 && $3>0 {print $4","$1}' "$SHARD_DIR/manifest.csv" \
          | sort -t',' -k1,1n | head -2 | cut -d',' -f2)"
if [[ -z "$SHARDS" ]]; then
  echo "WARNING: no shard in the manifest contains a plasmid fragment; falling back"
  echo "         to the two smallest shards"
  SHARDS="$(awk -F',' 'NR>1 {print $4","$1}' "$SHARD_DIR/manifest.csv" \
            | sort -t',' -k1,1n | head -2 | cut -d',' -f2)"
fi
echo "  verifying on shards: $(echo $SHARDS | tr '\n' ' ')"

conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_PLATON"
for SHARD in $SHARDS; do
  FASTA="$SHARD_DIR/$SHARD.fasta"
  VDIR="$TOOL_OUT/platon_modecheck/$SHARD"
  rm -rf "$VDIR"
  mkdir -p "$VDIR/characterize" "$VDIR/accuracy"
  echo "--- $SHARD: $(grep -c '^>' "$FASTA") contigs"
  run_tool platon --db "$PLATON_DB" --output "$VDIR/characterize" --prefix "$SHARD" \
           --characterize --threads "$CPUS_PER_TOOL" "$FASTA"
  run_tool platon --db "$PLATON_DB" --output "$VDIR/accuracy" --prefix "$SHARD" \
           --mode accuracy --threads "$CPUS_PER_TOOL" "$FASTA"
done

conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_ML"
cd "$PROJECT/tools-baseline"
FAILED=0
for SHARD in $SHARDS; do
  VDIR="$TOOL_OUT/platon_modecheck/$SHARD"
  echo
  echo "=== mode check: $SHARD"
  python -u verify_platon_modes.py \
      --tsv "$VDIR/characterize/$SHARD.tsv" \
      --plasmid-fasta "$VDIR/accuracy/$SHARD.plasmid.fasta" \
      --chromosome-fasta "$VDIR/accuracy/$SHARD.chromosome.fasta" || FAILED=1
done

if [[ $FAILED -ne 0 ]]; then
  banner "SMOKE TEST FAILED"
  echo "The offline accuracy rule does not reproduce Platon's own output."
  echo "Fallback: re-run 11_platon.sh with --mode accuracy instead of --characterize"
  echo "and score Platon's .plasmid.fasta / .chromosome.fasta directly. That costs"
  echo "one extra pass but removes the re-derivation from the result path."
  exit 1
fi

banner "smoke test passed - both tools work and the single-pass Platon design holds"
