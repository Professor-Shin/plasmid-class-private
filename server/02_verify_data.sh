#!/bin/bash
#SBATCH --job-name=verify-data
#SBATCH --output=server/logs/verify-data-%j.out
#SBATCH --error=server/logs/verify-data-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# Proves the server holds the same dataset the current report numbers came from,
# before any compute is spent on it. Submit with:
#
#   sbatch server/02_verify_data.sh
#
# (submit_all.sh overrides partition/qos/account from env.sh on the command line,
#  since #SBATCH directives cannot read variables)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

banner "verify transferred data"

require_dirs "$DATASET_DIR" "$PROJECT/dataset pipeline1" "$PROJECT/baseline"

echo "--- 1. checksums against the manifest built on the laptop"
if [[ -f server/dataset_md5.txt ]]; then
  ( cd "$PROJECT" && md5sum -c "$PROJECT/server/dataset_md5.txt" )
  echo "    all dataset checksums match the laptop"
else
  echo "FATAL: server/dataset_md5.txt missing - re-run 01_sync_from_laptop.sh" >&2
  exit 1
fi

echo "--- 2. the dataset pipeline's own independent re-derivation"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/dataset pipeline1"
srun python -u verify_dataset.py
cd "$PROJECT"

echo "--- 3. the counts this plan is sized for"
srun python -u server/_dataset_facts.py

banner "data verified"
