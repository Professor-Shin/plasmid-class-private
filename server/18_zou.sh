#!/bin/bash -l
#SBATCH --job-name=zou
#SBATCH --output=server/logs/zou-%j.out
#SBATCH --error=server/logs/zou-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# The fifth baseline: Zou et al. 2022, retrained on the mixed-length dataset.
#
# It is the odd one out in two ways and both are deliberate:
#
#   * It is TRAINED here, where Platon, MOB-suite, PLASMe and PlasmidHunter are
#     run as published. That is not a level playing field and the report says so
#     - but the alternative is worse. baseline/results/zou_*.json holds a Zou run
#     on the OLD fixed-5-kb dataset (test n=72,332), whose numbers cannot be put
#     in a table beside results measured on 16,944 mixed-length fragments.
#   * It needs train and validation, not just test, so it reads the whole
#     dataset rather than the per-assembly shards. Nothing here depends on
#     10_shard.sh, which is why submit_all.sh runs it in parallel with the tools.
#
# CPU, not GPU: the model is a 2080 -> 256 -> ... -> 1 MLP over at most 79,094
# rows. The arithmetic is trivial next to the k-mer counting pass that feeds it,
# and a GPU allocation would spend its 8-hour wall clock waiting on single-thread
# Python. 16 CPUs here, against the 40 the tool jobs take, so this can run
# alongside them inside the cpu40h cap rather than queueing behind them.
#
# Resumable in the part that takes the time: the 6-mer counts are cached as
# baseline/cache_mixed/zou_k6c_<split>.npy and reused, so a resubmit after a
# failed training re-reads them instead of re-counting 2.4 Gbp.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

banner "Zou et al., retrained on the mixed-length dataset"
cuaim_modules
cuaim_activate "$ENV_ML"
cd "$PROJECT/baseline"

require_dirs "$DATASET_DIR"
for split in train validation test; do
  for ext in csv fasta; do
    [[ -s "$DATASET_DIR/$split.$ext" ]] || {
      echo "FATAL: missing $DATASET_DIR/$split.$ext" >&2
      echo "       Unlike the four published tools, Zou trains, so it needs every" >&2
      echo "       split - not just test. Run 01_sync_from_laptop.sh." >&2
      exit 1; }
  done
done

# torch_threads in config.yaml was benchmarked on the laptop for batch-256 steps;
# the allocation here is larger, so let the job's own CPU count win unless the
# caller says otherwise. Beyond ~8 threads these batches lose more to OpenMP
# synchronisation than they gain, hence the cap.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( ${SLURM_CPUS_PER_TASK:-8} < 8 ? ${SLURM_CPUS_PER_TASK:-8} : 8 ))}"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS of ${SLURM_CPUS_PER_TASK:-?} allocated"

srun python -u run_zou_mixed.py ${ZOU_ARGS:-}

banner "Zou retrained"
ls -la "$PROJECT/baseline/results_mixed/" | grep -E "zou_mixed" || true
echo
echo "Both variants are in the table: 'zou' is the published pipeline on raw"
echo "6-mer counts, 'zou_lengthnorm' the same model on per-fragment frequencies."
echo "If they differ much, the counts model is reading fragment length."
