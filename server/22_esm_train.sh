#!/bin/bash -l
#SBATCH --job-name=esm-train
#SBATCH --output=server/logs/esm-train-%j.out
#SBATCH --error=server/logs/esm-train-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu40g
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#
# Train the four-variant ladder on the cached embeddings, under both class-balance
# regimes, with five seeds each, then score it against the k-mer baselines.
#
# The regression check comes first and is the important part of this script: the
# same code is run against the ORIGINAL config.yaml, on the laptop's own cache if
# it was copied up, and the result is compared with the numbers already in the
# report. The refactor that made this run possible touched the attention pooling,
# the training loop and the cache format, and a silent change there would look
# like a scientific result. Set SKIP_REGRESSION=1 to bypass it once it has passed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

CONFIG="${ESM_CONFIG:-$PROJECT/esm2-attention/config_server.yaml}"

banner "ESM-2 training | config $(basename "$CONFIG")"
cuaim_modules gpu
cuaim_activate "$ENV_ML"
nvidia-smi || { echo "FATAL: no GPU in this allocation" >&2; exit 1; }
cd "$PROJECT/esm2-attention"

echo
echo "--- component equivalence checks (batched vs per-fragment, GPU vs CPU)"
srun python -u verify_components.py --config "$CONFIG"

if [[ -z "${SKIP_REGRESSION:-}" ]]; then
  echo
  echo "--- regression check against the laptop's published numbers"
  if [[ -d cache_mixed ]] && [[ -n "$(ls -A cache_mixed 2>/dev/null)" ]]; then
    srun python -u "$PROJECT/server/_regression_check.py"
  else
    echo "    SKIPPED: esm2-attention/cache_mixed is not on the server, so the"
    echo "    laptop's exact embeddings are unavailable to re-train against."
    echo "    The equivalence checks above still cover the refactor; to run the"
    echo "    full check, rsync esm2-attention/cache_mixed/ and resubmit."
  fi
fi

echo
echo "--- train the ladder"
srun python -u train_eval.py --config "$CONFIG" --device cuda

echo
echo "--- score against the k-mer baselines"
srun python -u summarize.py --config "$CONFIG"

echo
echo "--- record the environment"
srun python -u capture_environment.py || echo "    (capture_environment.py failed; non-fatal)"

banner "training complete"
RESULTS="$(python -c "
import sys; sys.path.insert(0, '.')
from esm2 import data
print(data.load_config('$CONFIG')['results_dir'])")"
echo "results in $RESULTS"
ls -la "$RESULTS" | sed 's/^/  /'
