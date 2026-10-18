#!/bin/bash
# Submit the whole pipeline with the right dependencies, from the login node.
#
#   bash server/submit_all.sh                # everything
#   bash server/submit_all.sh tools          # only Platon + MOB-suite
#   bash server/submit_all.sh esm            # only the ESM-2 chain
#   DRY_RUN=1 bash server/submit_all.sh      # print the sbatch commands only
#
# Resource values come from env.sh and are passed on the sbatch COMMAND LINE,
# which overrides the #SBATCH directives inside each script - those directives
# cannot read variables, so they only carry defaults.
#
#   10_shard ──┬─ 11_platon[1-N%P] ──┬─ 14_collect_score ─┐
#              ├─ 12_mobsuite[1-N%P] ┤                    ├─ 30_report
#              └─ 13_smoke_tools ────┘                    │
#   20_esm_embed ── 21_esm_merge ── 22_esm_train ─────────┘
#
# 13_smoke_tools is a GATE and runs in parallel with the arrays rather than
# before them: it re-runs Platon in both modes on two shards and proves that the
# offline re-derivation of the accuracy rule is exact. 14_collect_score depends on
# it with afterok, so if the gate fails nothing is scored.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

WHAT="${1:-all}"
cd "$PROJECT"
mkdir -p "$SERVER_LOGS"

ACCOUNT_FLAG=()
[[ -n "$CUAIM_ACCOUNT" ]] && ACCOUNT_FLAG=(--account="$CUAIM_ACCOUNT")

CPU_FLAGS=(--partition="$PART_CPU" --qos="$QOS_CPU" "${ACCOUNT_FLAG[@]}")
GPU_FLAGS=(--partition="$PART_GPU" --qos="$QOS_GPU" --gres="$GRES_GPU"
           --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM" "${ACCOUNT_FLAG[@]}")
TOOL_FLAGS=(--cpus-per-task="$CPUS_PER_TOOL" --mem="$MEM_PER_TOOL"
            --time="$TIME_PER_TOOL")

submit() {   # submit <description> <sbatch args...> ; echoes the job id
  local desc="$1"; shift
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY  sbatch $*" >&2
    echo "0"
    return
  fi
  local out
  out="$(sbatch "$@")"
  local id="${out##* }"
  printf '%-22s job %s\n' "$desc" "$id" >&2
  echo "$id"
}

echo "=== submitting on $(hostname) as $(whoami)"
echo "    project  $PROJECT"
echo "    cpu      partition=$PART_CPU qos=$QOS_CPU  ${CPUS_PER_TOOL} cpus x ${MAX_PARALLEL} parallel"
echo "    gpu      partition=$PART_GPU qos=$QOS_GPU gres=$GRES_GPU"
echo

TOOLS_LAST=""
if [[ "$WHAT" == "all" || "$WHAT" == "tools" ]]; then
  SHARD_ID="$(submit "10_shard" "${CPU_FLAGS[@]}" --cpus-per-task=2 --mem=8G \
                     --time=01:00:00 "$HERE/10_shard.sh")"

  # The array size is the number of shards, which is the number of assemblies in
  # the test split. Read from the manifest when it already exists; otherwise fall
  # back to the count _dataset_facts.py asserts, since 10_shard has not run yet.
  N_SHARDS=116
  if [[ -f "$SHARD_DIR/manifest.csv" ]]; then
    N_SHARDS="$(( $(wc -l < "$SHARD_DIR/manifest.csv") - 1 ))"
    echo "    shards   $N_SHARDS (from manifest.csv)"
  else
    echo "    shards   $N_SHARDS (default; manifest.csv not written yet)"
    echo "             if 10_shard reports a different count, cancel the arrays and"
    echo "             resubmit this script - the manifest will then be read directly"
  fi

  PLATON_ID="$(submit "11_platon[$N_SHARDS]" "${CPU_FLAGS[@]}" "${TOOL_FLAGS[@]}" \
                      --array="1-${N_SHARDS}%${MAX_PARALLEL}" \
                      --dependency="afterok:$SHARD_ID" "$HERE/11_platon.sh")"
  MOB_ID="$(submit "12_mobsuite[$N_SHARDS]" "${CPU_FLAGS[@]}" "${TOOL_FLAGS[@]}" \
                   --array="1-${N_SHARDS}%${MAX_PARALLEL}" \
                   --dependency="afterok:$SHARD_ID" "$HERE/12_mobsuite.sh")"
  SMOKE_ID="$(submit "13_smoke_tools" "${CPU_FLAGS[@]}" \
                     --cpus-per-task="$CPUS_PER_TOOL" --mem="$MEM_PER_TOOL" \
                     --time=03:00:00 --dependency="afterok:$SHARD_ID" \
                     "$HERE/13_smoke_tools.sh")"
  COLLECT_ID="$(submit "14_collect_score" "${CPU_FLAGS[@]}" --cpus-per-task=4 \
                       --mem=16G --time=01:00:00 \
                       --dependency="afterok:$PLATON_ID:$MOB_ID:$SMOKE_ID" \
                       "$HERE/14_collect_score.sh")"
  TOOLS_LAST="$COLLECT_ID"
fi

ESM_LAST=""
if [[ "$WHAT" == "all" || "$WHAT" == "esm" ]]; then
  EMBED_ID="$(submit "20_esm_embed" "${GPU_FLAGS[@]}" --time="$TIME_EMBED" \
                     "$HERE/20_esm_embed.sh")"
  MERGE_ID="$(submit "21_esm_merge" "${CPU_FLAGS[@]}" --cpus-per-task=4 --mem=32G \
                     --time=02:00:00 --dependency="afterok:$EMBED_ID" \
                     "$HERE/21_esm_merge.sh")"
  TRAIN_ID="$(submit "22_esm_train" "${GPU_FLAGS[@]}" --time="$TIME_TRAIN" \
                     --dependency="afterok:$MERGE_ID" "$HERE/22_esm_train.sh")"
  ESM_LAST="$TRAIN_ID"
fi

DEPS=""
for id in $TOOLS_LAST $ESM_LAST; do
  [[ -n "$id" && "$id" != "0" ]] && DEPS="${DEPS:+$DEPS:}$id"
done
if [[ -n "$DEPS" ]]; then
  submit "30_report" "${CPU_FLAGS[@]}" --cpus-per-task=4 --mem=16G --time=02:00:00 \
         --dependency="afterok:$DEPS" "$HERE/30_report.sh" >/dev/null
fi

echo
echo "=== submitted. Watch with:"
echo "    squeue -u \"\$USER\" -o '%.18i %.9P %.12j %.8T %.10M %.9l %R %b'"
echo "=== if a job fails:"
echo "    sacct -j <id> --format=JobID,JobName%18,State,ExitCode,Elapsed"
echo "    the arrays and the embedding job both SKIP finished work, so the fix is"
echo "    usually to resubmit the same script rather than to start over:"
echo "      sbatch ${CPU_FLAGS[*]} ${TOOL_FLAGS[*]} --array=1-N%${MAX_PARALLEL} server/11_platon.sh"
echo "      sbatch ${GPU_FLAGS[*]} --time=$TIME_EMBED server/20_esm_embed.sh"
echo "=== cancel everything:  scancel -u \"\$USER\""
