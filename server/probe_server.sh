#!/bin/bash
# Read-only survey of CU-AIM. Run this FIRST, on the login node, and paste the
# answers into env.sh:
#
#   bash server/probe_server.sh 2>&1 | tee server/logs/probe.txt
#
# Nothing here writes, submits or installs. The three answers that decide whether
# the plan runs as designed are printed again at the end:
#   (1) the project directory this account can write to, and its free space
#   (2) the QoS list actually granted to this account
#   (3) whether the login node can reach the internet (tool databases are ~7 GB)

set -u
say() { echo; echo "######## $* ########"; }
try() { echo "\$ $*"; "$@" 2>&1 | sed 's/^/  /' || echo "  (command unavailable or failed: $*)"; }

echo "CU-AIM probe | user=$(whoami) host=$(hostname) date=$(date -Is)"

say "1. account, group and QoS (CU-AIM helper commands)"
try check_qos
try check_group
try sacctmgr show assoc "user=$(whoami)" format=account,user%15,qos%60

say "2. partitions, nodes and GPU resources"
try sinfo -o "%.15R %.8D %.10m %.5c %7z %40G %N"
echo "\$ sinfo -h -o %R | sort -u   (partition names for PART_CPU / PART_GPU)"
sinfo -h -o "%R" 2>/dev/null | sort -u | sed 's/^/  /' || echo "  (sinfo unavailable)"
echo "\$ sinfo -h -o %G | sort -u   (GRES strings for GRES_GPU)"
sinfo -h -o "%G" 2>/dev/null | sort -u | sed 's/^/  /' || echo "  (sinfo unavailable)"

say "3. scheduler limits (the Platon/MOB arrays need MaxArraySize >= 116)"
try bash -c 'scontrol show config | grep -i -E "MaxArraySize|MaxJobCount|MaxSubmitJobs|DefMemPerCPU"'

say "4. software modules"
try bash -c 'module avail 2>&1 | tr "\t" "\n" | grep -i -E "anaconda|miniconda|cuda|singularity|apptainer|python" | sort -u'
for c in conda mamba singularity apptainer rsync git curl wget; do
  printf '  %-12s %s\n' "$c" "$(command -v $c 2>/dev/null || echo '-- not found --')"
done

say "5. storage and quota  (THE constraint: /data/home is small, do not use it)"
try df -h /data/home /data/project "$HOME"
try bash -c 'quota -s 2>/dev/null || echo "(no quota command)"'
echo "\$ project directories this account can see"
ls -ld /data/project/*/ 2>/dev/null | sed 's/^/  /' || echo "  (cannot list /data/project)"
echo "\$ writable candidates under /data/project"
for d in /data/project/*/ /data/project/*/*/; do
  [[ -d "$d" && -w "$d" ]] && echo "  WRITABLE  $d"
done 2>/dev/null
echo
echo "  This plan needs roughly 30 GB:"
echo "    dataset (rsync)   2.5 GB      Platon DB        2.8 GB"
echo "    MOB-suite DB      ~4 GB       ESM embeddings   3-5 GB"
echo "    conda envs        ~8 GB       logs/outputs     ~1 GB"
echo "  If the free space is under ~30 GB, use the fallback ladder in section 4"
echo "  of the plan (float16 embeddings first, then drop ESM shards after merge)."

say "6. internet access from the login node (databases are downloaded here)"
for url in https://zenodo.org https://huggingface.co https://conda.anaconda.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$url" 2>/dev/null || echo "---")
  printf '  %-32s HTTP %s\n' "$url" "$code"
done

say "7. GPU visibility from the login node"
echo "  (a login node normally has NO GPU - 'command not found' or 'no devices' here"
echo "   is expected and not a problem; the real check runs inside a GPU allocation)"
try nvidia-smi

say "8. this account's recent Slurm history"
try bash -c 'sacct -u "$(whoami)" --start=$(date -d "7 days ago" +%F) -o JobID,JobName%20,State,Elapsed,NodeList 2>/dev/null | head -20'

say "SUMMARY - copy these into server/env.sh"
echo "  PROJECT=         <a writable /data/project path from section 5>"
echo "  CUAIM_ACCOUNT=   <the account column from check_group / sacctmgr, section 1>"
echo "  PART_CPU=        <cpu-ish partition from section 2>   QOS_CPU=cpu40h"
echo "  PART_GPU=        <gpu partition from section 2>       QOS_GPU=gpu40g"
echo "  GRES_GPU=        <e.g. gpu:7g.40gb:1, from section 2>"
echo "  MODULE_CONDA=    <Anaconda3 or the name shown in section 4>"
echo "  MODULE_CUDA=     <CUDA/11.8.0 or the name shown in section 4>"
echo
echo "Then: 1) edit server/env.sh   2) bash server/00_setup_envs.sh   3) bash server/01_sync_from_laptop.sh (on the laptop)"
