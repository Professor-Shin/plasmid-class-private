#!/bin/bash -l
#SBATCH --job-name=collect-tools
#SBATCH --output=server/logs/collect-tools-%j.out
#SBATCH --error=server/logs/collect-tools-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Turn 116 x 4 tool output directories into four fragment-level prediction tables,
# one scored summary and one cost table, using the same metric functions the
# retrained baselines were scored with (baseline/baselines/metrics.py, imported
# verbatim) so all five baselines land in one table with one definition of MCC.
#
# The collectors fail loudly if any shard is missing, if classified +
# unclassified does not add up to the fragment count, or if PLASMe's offline mode
# rule disagrees with PLASMe's own report on any shard - rather than quietly
# scoring a partial or mis-derived run. A baseline computed on 90% of the test
# split would be worse than no baseline at all.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

banner "collect and score the four published tools"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/tools-baseline"

echo "--- 1. Platon"
srun python -u collect_platon.py

echo
echo "--- 2. MOB-suite"
srun python -u collect_mobsuite.py

echo
echo
echo "--- 3. PLASMe"
srun python -u collect_plasme.py

echo
echo "--- 4. PlasmidHunter"
srun python -u collect_plasmidhunter.py

echo
echo "--- 5. score all four with the retrained baselines' metrics"
srun python -u score_tools.py

echo
echo "--- 6. what each tool cost (wall clock, CPU seconds, peak RSS)"
# Non-fatal: a missing timing record is a gap in the methods section, not a
# reason to throw away the results that were just scored.
srun python -u collect_runtime.py --logs "$SERVER_LOGS" \
  || echo "    (collect_runtime.py failed; non-fatal, the scores above stand)"

banner "tool baselines scored"
echo "wrote into $PROJECT/baseline/results_mixed/:"
ls -la "$PROJECT/baseline/results_mixed/" \
  | grep -E "tool_baselines|tool_runtime|predictions_" || true
