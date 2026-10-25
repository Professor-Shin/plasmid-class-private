#!/bin/bash -l
#SBATCH --job-name=esm-embed
#SBATCH --output=server/logs/esm-embed-%j.out
#SBATCH --error=server/logs/esm-embed-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu40g
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#
# Embed every ORF of every fragment with ESM-2.
#
# One job, not a job array: the documented QoS tiers grant ONE GPU per job
# (gpu40g24m is the only exception, with two), so shards here exist for
# resumability rather than parallelism. Each shard is a separate srun with its own
# log, and a shard whose cache already exists is skipped - so if this job hits the
# time limit or a node dies, resubmitting the identical command continues from the
# next unfinished shard. The laptop had no such property: an interrupted split
# restarted from zero, and the test split alone took 233 minutes.
#
# Scale, measured on the real dataset rather than estimated: with the ORF cap
# removed the 112,929 fragments yield 8.79 M ORFs, i.e. 2.13 G amino acids to
# push through the transformer, and a 5.2 GiB float16 embedding cache. The
# six-frame translation itself is cheap (~1.3 CPU-hours in total, since
# Biopython's translate is C-backed), so the GPU is the bottleneck and the 14 ORF
# workers exist only to keep it fed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

CONFIG="${ESM_CONFIG:-$PROJECT/esm2-attention/config_server.yaml}"

banner "ESM-2 embedding | config $(basename "$CONFIG")"
cuaim_modules gpu
cuaim_activate "$ENV_ML"

nvidia-smi || { echo "FATAL: no GPU in this allocation" >&2; exit 1; }
python - <<'PY'
import torch
assert torch.cuda.is_available(), (
    "torch.cuda.is_available() is False inside a GPU allocation. The torch wheel "
    "does not match the driver - reinstall with the index-url 00_setup_envs.sh "
    "selected (see $PROJECT/.cuda_tag).")
print(f"  torch {torch.__version__} sees {torch.cuda.get_device_name(0)}")
PY

cd "$PROJECT/esm2-attention"

echo
echo "--- work plan"
python -u "$PROJECT/server/_shard_plan.py" --config "$CONFIG" | sed 's/^/    /'
PENDING="$(python -u "$PROJECT/server/_shard_plan.py" --config "$CONFIG" --pending-only)"
if [[ -z "$PENDING" ]]; then
  banner "nothing to do - every shard is already cached"
  exit 0
fi
echo
echo "--- $(wc -l <<<"$PENDING") shard(s) pending"

mkdir -p "$SERVER_LOGS"
while read -r SPLIT SHARD NSHARDS _; do
  [[ -z "$SPLIT" ]] && continue
  TAG="$SPLIT-shard$SHARD"
  echo
  echo "=== embedding $TAG of $NSHARDS at $(date -Is)"
  # Blocking srun per shard: the next one starts only when this one has finished,
  # so a time-limit kill loses at most one shard.
  srun --output="$SERVER_LOGS/esm-embed-${SLURM_JOB_ID:-local}-$TAG.out" \
       --error="$SERVER_LOGS/esm-embed-${SLURM_JOB_ID:-local}-$TAG.err" \
       python -u extract_features.py --config "$CONFIG" \
              --split "$SPLIT" --shard "$SHARD" --n-shards "$NSHARDS" \
              --device cuda
  echo "    done; log: $SERVER_LOGS/esm-embed-${SLURM_JOB_ID:-local}-$TAG.out"
  tail -3 "$SERVER_LOGS/esm-embed-${SLURM_JOB_ID:-local}-$TAG.out" | sed 's/^/    /'
done <<<"$PENDING"

echo
echo "--- remaining work after this job"
python -u "$PROJECT/server/_shard_plan.py" --config "$CONFIG" --pending-only | sed 's/^/    /'
du -sh "$PROJECT/esm2-attention/cache_server" 2>/dev/null | sed 's/^/    /'

banner "embedding job finished"
echo "If any shard is still pending above, resubmit this identical script - it"
echo "skips what is already cached."
