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
# gc_signal.py hardcodes the _mixed directories; run it only when they are the
# ones that were just produced. It is an analysis of the laptop run's artefacts,
# not a dependency of anything below.
if [[ -f "$PROJECT/esm2-attention/results_mixed/esm2_variants.json" ]]; then
  srun python -u analysis/gc_signal.py || echo "    (gc_signal.py failed; non-fatal)"
else
  echo "    SKIPPED: analysis/gc_signal.py reads esm2-attention/results_mixed,"
  echo "    which this run does not write (it writes results_server). Point it at"
  echo "    the server results by editing its ESM_RESULTS/ESM_CACHE constants if"
  echo "    the GC analysis is wanted for the server ladder too."
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
