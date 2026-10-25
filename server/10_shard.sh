#!/bin/bash -l
#SBATCH --job-name=shard
#SBATCH --output=server/logs/shard-%j.out
#SBATCH --error=server/logs/shard-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# Turn the test split into one FASTA per assembly, i.e. one fragmented draft
# assembly per Slurm array task. Streams the 367 MB FASTA once; it is submitted
# as a job rather than run on the login node because the manual (2.1) reserves
# the login node for file management and submission.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

banner "shard the test split by assembly"
cuaim_modules
cuaim_activate "$ENV_ML"

cd "$PROJECT/tools-baseline"
srun python -u shard_by_assembly.py

echo
echo "The array size for 11_platon.sh / 12_mobsuite.sh is the shard count above."
echo "submit_all.sh reads it straight out of shards/manifest.csv."
