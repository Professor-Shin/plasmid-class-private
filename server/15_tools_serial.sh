#!/bin/bash -l
#SBATCH --job-name=tools-serial
#SBATCH --output=server/logs/tools-serial-%j.out
#SBATCH --error=server/logs/tools-serial-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#
# All shards of one tool inside ONE Slurm job, instead of a 116-element array.
#
# Why this exists: cpu40h rejects the array with QOSMaxSubmitJobPerUserLimit,
# because every array element counts as a submitted job against that limit.
# The manual's own answer to "a numbered series of programs that must fit one
# allocation" is the loop in section 5.3.2, which is what this is - with the
# shards run a few at a time so the 40 CPUs are not left idle.
#
# It calls 11_platon.sh / 12_mobsuite.sh unchanged, once per shard, exactly as
# an array element would. Those scripts skip a shard whose output already
# exists, so this is as resumable as the array was: submit it again and it picks
# up where it stopped.
#
#   sbatch --account=$CUAIM_ACCOUNT --dependency=afterok:<shard job> \
#          server/15_tools_serial.sh platon
#   sbatch --account=$CUAIM_ACCOUNT --dependency=afterok:<shard job> \
#          server/15_tools_serial.sh mobsuite
#
# TOOL_PARALLEL shards run at once, each with CPUS_PER_TOOL threads; keep the
# product at or below the CPUs this job asked for.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source server/env.sh
source server/lib.sh

TOOL="${1:-${TOOL:-platon}}"
case "$TOOL" in
  platon)   SCRIPT="server/11_platon.sh" ;;
  mobsuite) SCRIPT="server/12_mobsuite.sh" ;;
  *) echo "FATAL: usage: $0 platon|mobsuite" >&2; exit 2 ;;
esac

MANIFEST="$SHARD_DIR/manifest.csv"
[[ -f "$MANIFEST" ]] || { echo "FATAL: no manifest $MANIFEST - run 10_shard.sh" >&2; exit 1; }
N="$(( $(wc -l < "$MANIFEST") - 1 ))"
PAR="${TOOL_PARALLEL:-$MAX_PARALLEL}"
CPUS="${SLURM_CPUS_PER_TASK:-$(( PAR * CPUS_PER_TOOL ))}"

banner "$TOOL | $N shards, $PAR at a time, $CPUS_PER_TOOL threads each ($CPUS cpus allocated)"
if (( PAR * CPUS_PER_TOOL > CPUS )); then
  echo "WARN: $PAR x $CPUS_PER_TOOL threads exceeds the $CPUS cpus allocated;" >&2
  echo "      lower TOOL_PARALLEL or raise --cpus-per-task" >&2
fi
mkdir -p "$SERVER_LOGS"

# One log per shard, named the way the array's logs were, so the troubleshooting
# steps in MANUAL.md still find what they describe.
seq 1 "$N" | xargs -P "$PAR" -I{} bash -c '
    shard="$1"; script="$2"; logs="$3"; tool="$4"
    if SLURM_ARRAY_TASK_ID="$shard" bash "$script" \
         > "$logs/$tool-serial-$(printf %03d "$shard").out" 2>&1; then
      echo "  ok   shard $shard"
    else
      echo "  FAIL shard $shard (see $logs/$tool-serial-$(printf %03d "$shard").out)"
      exit 1
    fi
  ' _ {} "$SCRIPT" "$SERVER_LOGS" "$TOOL" && rc=0 || rc=$?

if (( rc != 0 )); then
  echo
  echo "FATAL: at least one shard failed; the lines above name them." >&2
  echo "       Finished shards are kept, so resubmitting this script retries" >&2
  echo "       only what is missing." >&2
  exit 1
fi

banner "$TOOL complete - all $N shards have output"
