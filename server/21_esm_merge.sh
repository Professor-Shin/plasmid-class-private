#!/bin/bash
#SBATCH --job-name=esm-merge
#SBATCH --output=server/logs/esm-merge-%j.out
#SBATCH --error=server/logs/esm-merge-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Concatenate the per-shard embedding caches into one cache per split. CPU only -
# it is file I/O, so it does not need to hold the GPU allocation.
#
# merge_shards.py refuses to merge an incomplete set and asserts that the merged
# row order is strictly increasing, which is the property that makes the result
# independent of how many shards were used. The shard files are deleted after a
# verified merge because they duplicate it exactly and project storage is the
# binding constraint.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

CONFIG="${ESM_CONFIG:-$PROJECT/esm2-attention/config_server.yaml}"
KEEP="${KEEP_SHARDS:-}"

banner "merge embedding shards"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/esm2-attention"

df -h "$PROJECT" | sed 's/^/  df| /'
srun python -u merge_shards.py --config "$CONFIG" ${KEEP:+--keep-shards}
df -h "$PROJECT" | sed 's/^/  df| /'

banner "merge complete"
ls -la cache_server/ | grep -E "emb_.*(data.npy|meta.npz)" | sed 's/^/  /'
