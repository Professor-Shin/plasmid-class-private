#!/bin/bash
# Run this ON THE LAPTOP (Git Bash), with the Chula VPN connected.
#
#   CUAIM_USER=<your-cuaim-user> CUAIM_KEY=~/.ssh/id_rsa_cuaim \
#   PROJECT=/data/project/<group>/<dir> bash server/01_sync_from_laptop.sh
#
# Copies the dataset and the existing result files to CU-AIM. The dataset is
# COPIED rather than regenerated on the server on purpose: step01/step02 of
# "dataset pipeline1" download from NCBI, whose assembly_summary.txt changes
# every week, so re-running the pipeline there would risk building a different
# dataset and invalidating every number already in the report.
#
# What is deliberately NOT copied:
#   baseline/cache*                    32 GB, regenerable, unused by this plan
#   dataset pipeline1/data/fasta       3 GB of per-assembly downloads, already
#                                      distilled into the fragment FASTAs
#   dataset pipeline1/data/raw         450 MB of NCBI metadata tables

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "$HERE/.." && pwd)"

: "${CUAIM_HOST:=161.200.107.77}"
: "${CUAIM_USER:?set CUAIM_USER to your CU-AIM username}"
: "${CUAIM_KEY:=$HOME/.ssh/id_rsa_cuaim}"
: "${PROJECT:?set PROJECT to the assigned /data/project path (see probe_server.sh)}"

SSH_OPTS=(-i "$CUAIM_KEY")
REMOTE="$CUAIM_USER@$CUAIM_HOST"

echo "=== syncing to $REMOTE:$PROJECT"
echo "    from $LOCAL_ROOT"
echo "    (VPN must be connected - see the manual, section 1.1)"

# The fragment FASTAs and CSVs are the only inputs every downstream step needs.
# Keep the layout byte-identical to the laptop, including the space in the
# directory name, so that no existing config.yaml has to be edited.
ITEMS=(
  "dataset pipeline1/data/dataset_mixed/train.fasta"
  "dataset pipeline1/data/dataset_mixed/train.csv"
  "dataset pipeline1/data/dataset_mixed/validation.fasta"
  "dataset pipeline1/data/dataset_mixed/validation.csv"
  "dataset pipeline1/data/dataset_mixed/test.fasta"
  "dataset pipeline1/data/dataset_mixed/test.csv"
  "dataset pipeline1/data/metadata/replicons.csv"
  "dataset pipeline1/data/metadata/fragments_mixed.csv"
  "dataset pipeline1/data/split"
  "dataset pipeline1/reports/run_manifest.json"
  "dataset pipeline1/reports/dataset_summary_mixed.csv"
  "baseline/results_mixed"
  "esm2-attention/results_mixed"
  # The laptop's embedding cache (~42 MB) is sent so 22_esm_train can rerun the
  # ORIGINAL config on the server and confirm the refactor still reproduces the
  # published MCCs. At <2% of this transfer it is not worth skipping, and with the
  # server on a different torch/GPU it is the only check that the port is faithful.
  "esm2-attention/cache_mixed"
)

# A checksum manifest is built here, on the machine that produced the data, and
# verified on the server by 02_verify_data.sh. This is what proves the server is
# scoring the same bytes the current report numbers came from.
MANIFEST="$HERE/dataset_md5.txt"
echo "=== building $MANIFEST"
: > "$MANIFEST"
for f in "${ITEMS[@]:0:8}"; do
  ( cd "$LOCAL_ROOT" && md5sum "$f" ) >> "$MANIFEST"
done
cat "$MANIFEST"

echo "=== creating remote directories"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$PROJECT'"

if command -v rsync >/dev/null; then
  echo "=== rsync (resumable: re-run after a dropped VPN and it continues)"
  for item in "${ITEMS[@]}" "server/dataset_md5.txt"; do
    echo "--- $item"
    rsync -av --partial --progress --mkpath \
          -e "ssh ${SSH_OPTS[*]}" \
          "$LOCAL_ROOT/$item" "$REMOTE:$PROJECT/$(dirname "$item")/"
  done
else
  echo "=== rsync not found; falling back to scp (NOT resumable)"
  echo "    Git Bash often ships without rsync. If the VPN drops mid-transfer,"
  echo "    install rsync (e.g. via MSYS2 pacman) or re-run this whole script."
  for item in "${ITEMS[@]}" "server/dataset_md5.txt"; do
    echo "--- $item"
    ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$PROJECT/$(dirname "$item")'"
    scp "${SSH_OPTS[@]}" -r "$LOCAL_ROOT/$item" "$REMOTE:$PROJECT/$(dirname "$item")/"
  done
fi

echo
echo "=== done. The code itself is NOT copied here - clone it on the server:"
echo "    ssh ${SSH_OPTS[*]} $REMOTE"
echo "    cd '$PROJECT' && git clone <this repo> . 2>/dev/null || git pull"
echo "=== then verify the transfer:  sbatch server/02_verify_data.sh"
