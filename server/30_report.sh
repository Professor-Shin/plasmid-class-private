#!/bin/bash -l
#SBATCH --job-name=report
#SBATCH --output=server/logs/report-%j.out
#SBATCH --error=server/logs/report-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# The last step: one table holding every baseline plus the ESM-2 ladder.
#
# gc_signal.py answers "is the signal just GC?" and dump_tables.py renders every
# artefact as markdown. Both read the _mixed directory names by default, so
# ESM_RESULTS points them at the server run's output.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

banner "render the report tables"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT"

mkdir -p "$SERVER_LOGS"
OUT="$SERVER_LOGS/report_$(date +%Y%m%d_%H%M).md"

echo "--- GC-signal analysis"
# Analyse the run this report is about. results_mixed from the laptop is tracked
# in git, so without these overrides gc_signal.py would quietly describe the
# laptop's capped-ORF ladder inside a report about the server run.
ESM_RES_DIR="${ESM_RESULTS:-results_server}"
ESM_CACHE_DIR="${ESM_CACHE:-cache_server}"
if [[ -f "$PROJECT/esm2-attention/$ESM_RES_DIR/esm2_variants.json" ]]; then
  echo "    reading esm2-attention/$ESM_RES_DIR and $ESM_CACHE_DIR"
  BASE_RESULTS="${BASE_RESULTS:-results_mixed}" \
  ESM_RESULTS="$ESM_RES_DIR" ESM_CACHE="$ESM_CACHE_DIR" \
    srun python -u analysis/gc_signal.py || echo "    (gc_signal.py failed; non-fatal)"
else
  echo "    SKIPPED: no esm2-attention/$ESM_RES_DIR/esm2_variants.json yet."
  echo "    Set ESM_RESULTS/ESM_CACHE to analyse a different run."
fi

echo
echo "--- every artefact as markdown -> $OUT"
ESM_RESULTS="${ESM_RESULTS:-results_server}" \
BASE_RESULTS="${BASE_RESULTS:-results_mixed}" \
  srun python -u analysis/dump_tables.py | tee "$OUT"

banner "report written"
echo "$OUT"
echo
echo "Copy it to the laptop with:"
echo "  scp -i <key> <user>@161.200.107.77:$OUT ."
