#!/bin/bash -l
#SBATCH --job-name=smoke-tools
#SBATCH --output=server/logs/smoke-tools-%j.out
#SBATCH --error=server/logs/smoke-tools-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#
# The gate in front of the four 116-shard tool runs. Three things are checked,
# and any failure here means those runs would waste hours producing unusable
# output:
#
#   1. All four tools run at all, on a 20-fragment input with both classes
#      present (a tool that called everything chromosome would still "succeed"
#      on a plasmid-only input, hence both classes).
#   2. Their output carries the columns the collectors depend on - parse.py
#      asserts by column NAME, so a schema change from a different tool version
#      fails here with the header it actually saw.
#   3. The two offline rule re-derivations hold:
#        - Platon's `accuracy` rule reproduces a real `platon --mode accuracy`
#          run EXACTLY, on two real shards;
#        - PLASMe's `balance` preset reproduces PLASMe's own report.
#      Each is the claim a single-pass design rests on. If Platon's fails, the
#      fallback is to run the array with --mode accuracy instead; if PLASMe's
#      fails, to run it once per preset with -m.
#      (collect_plasme.py re-checks PLASMe's on all 116 shards afterwards, since
#      every shard carries its own report; the gate exists to catch it in two
#      minutes rather than two hours.)
#
# Runs independently of the tool jobs (it invokes the tools itself), so it can be
# submitted in parallel with them - submit_all.sh does exactly that and makes the
# collection step depend on all of them.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

SMOKE="$TOOL_OUT/smoke"
mkdir -p "$SMOKE"

banner "1/6  build a 20-fragment smoke input"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/tools-baseline"
python -u make_smoke_fasta.py --n-per-class 10 --out "$SMOKE/smoke.fasta"

banner "2/6  platon on the smoke input"
conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_PLATON"
rm -rf "$SMOKE/platon"
mkdir -p "$SMOKE/platon"
run_tool platon --db "$PLATON_DB" --output "$SMOKE/platon" --prefix smoke \
         --characterize --threads "$CPUS_PER_TOOL" --verbose "$SMOKE/smoke.fasta"
echo "  TSV header: $(head -1 "$SMOKE/platon/smoke.tsv")"

banner "3/6  mob_recon on the smoke input"
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

banner "4/6  PLASMe on the smoke input (and does its offline rule match?)"
conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_PLASME"
if [[ ! -f "$PLASME_DB/plsdb_Mar30.dmnd" ]]; then
  echo "FATAL: $PLASME_DB holds no built database - run 00_setup_envs.sh" >&2
  exit 1
fi
rm -rf "$SMOKE/plasme"
mkdir -p "$SMOKE/plasme/temp"
# Run at the CLI defaults, which ARE the `balance` preset, so the report this
# writes is ground truth for exactly the mode verify_plasme_modes.py checks.
( cd "$SMOKE/plasme" && run_tool plasme "$SMOKE/smoke.fasta" "$SMOKE/plasme/plasmid.fna" \
      --database "$PLASME_DB" --thread "$CPUS_PER_TOOL" --temp "$SMOKE/plasme/temp" )
if [[ ! -s "$SMOKE/plasme/temp/PLASMe_candidate.csv" ]]; then
  echo "FATAL: PLASMe wrote no candidate table. It exits 0 on several of its own" >&2
  echo "       error paths, so read the output above rather than the exit code." >&2
  exit 1
fi
echo "  candidate header: $(head -1 "$SMOKE/plasme/temp/PLASMe_candidate.csv")"
echo "  report header:    $(head -1 "$SMOKE/plasme/plasmid.fna_report.csv" 2>/dev/null || echo '<no report>')"

banner "5/6  PlasmidHunter on the smoke input"
conda deactivate 2>/dev/null || true
cuaim_activate "$ENV_PH"
PH_PKG="$(python -c 'import os, PlasmidHunter; print(os.path.dirname(PlasmidHunter.__file__))')"
if [[ ! -s "$PH_PKG/database/database.dmnd" ]]; then
  echo "FATAL: PlasmidHunter's DIAMOND database is missing from $PH_PKG/database -" >&2
  echo "       run 00_setup_envs.sh. Letting it download on a compute node means" >&2
  echo "       116 simultaneous downloads, or none at all if the node is offline." >&2
  exit 1
fi
rm -rf "$SMOKE/plasmidhunter"
run_tool plasmidhunter --infile "$SMOKE/smoke.fasta" --cpu "$CPUS_PER_TOOL" \
         --outdir "$SMOKE/plasmidhunter"
[[ -s "$SMOKE/plasmidhunter/predictions.tsv" ]] || {
  echo "FATAL: plasmidhunter wrote no predictions.tsv" >&2; exit 1; }
echo "  predictions header: $(head -1 "$SMOKE/plasmidhunter/predictions.tsv")"

banner "6/6  does the offline accuracy rule match a real --mode accuracy run?"
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
  echo "=== Platon mode check: $SHARD"
  python -u verify_platon_modes.py \
      --tsv "$VDIR/characterize/$SHARD.tsv" \
      --plasmid-fasta "$VDIR/accuracy/$SHARD.plasmid.fasta" \
      --chromosome-fasta "$VDIR/accuracy/$SHARD.chromosome.fasta" || FAILED=1
done

echo
echo "=== PLASMe mode check: the smoke run"
# The smoke run used the CLI defaults, i.e. the `balance` preset, and PLASMe
# wrote both the candidate table and its own report - so one run is both the
# input and the ground truth for this check.
python -u verify_plasme_modes.py \
    --candidate "$SMOKE/plasme/temp/PLASMe_candidate.csv" \
    --report "$SMOKE/plasme/plasmid.fna_report.csv" \
    --mode balance || FAILED=1

echo
echo "=== PlasmidHunter schema check"
python -u - "$SMOKE/plasmidhunter/predictions.tsv" <<'PY' || FAILED=1
import sys
from pathlib import Path
from toolbase import parse, rules
df = parse.read_plasmidhunter_tsv(Path(sys.argv[1]))
print(f"parsed {len(df)} rows, columns {sorted(df.columns)}")
missing = [c for c in ("contig_id", "pred", "proba_plasmid") if c not in df.columns]
if missing:
    raise SystemExit(f"FAIL: predictions.tsv lacks {missing} after aliasing")
# plasmidhunter_labels raises if the class column is not the argmax of the two
# probability columns, which is the one invariant the collector relies on.
labels = rules.plasmidhunter_labels(df)
print(f"PASS: class column agrees with its own probabilities; "
      f"{int((labels == 1).sum())} of {len(labels)} called plasmid")
PY

if [[ $FAILED -ne 0 ]]; then
  banner "SMOKE TEST FAILED"
  echo "At least one offline rule does not reproduce its tool's own output; the"
  echo "lines above name which. Fallbacks:"
  echo "  Platon: re-run 11_platon.sh with --mode accuracy instead of --characterize"
  echo "          and score its .plasmid.fasta / .chromosome.fasta directly."
  echo "  PLASMe: re-run 16_plasme.sh once per preset with -m <mode> and score each"
  echo "          run's own _report.csv."
  echo "Either costs an extra pass but removes the re-derivation from the result path."
  exit 1
fi

banner "smoke test passed - all four tools work and both single-pass designs hold"
