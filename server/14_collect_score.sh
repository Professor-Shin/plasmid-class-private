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
# Turn 116 x 2 tool output directories into two fragment-level prediction tables
# and one scored summary, using the same metric functions the retrained baselines
# were scored with (baseline/baselines/metrics.py, imported verbatim).
#
# The collectors fail loudly if any shard is missing or if classified +
# unclassified does not add up to the fragment count, rather than quietly scoring
# a partial run - a baseline computed on 90% of the test split would be worse
# than no baseline at all.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

banner "collect and score Platon + MOB-suite"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/tools-baseline"

echo "--- 1. Platon"
srun python -u collect_platon.py

echo
echo "--- 2. MOB-suite"
srun python -u collect_mobsuite.py

echo
echo "--- 3. score both with the retrained baselines' metrics"
srun python -u score_tools.py

banner "tool baselines scored"
echo "wrote into $PROJECT/baseline/results_mixed/:"
ls -la "$PROJECT/baseline/results_mixed/" | grep -E "tool_baselines|predictions_" || true
