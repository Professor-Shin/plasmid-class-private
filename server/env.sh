#!/bin/bash
# Every CU-AIM-specific knob lives here. All other scripts `source` this file and
# nothing else, so moving to a different account, project directory or QoS is a
# one-file edit.
#
# Fill the <...> placeholders from `bash server/probe_server.sh` output BEFORE
# submitting anything. Values are documented in survival-manual-cuaim.pdf:
#   QoS table    section 3.2.2
#   partitions   section 3.3
#   modules      section 4.4
#   storage      section 3.1  (/data/home has a small quota - do not use it)

# ---------------------------------------------------------------- identity ----
# Assigned project directory. The whole repository lives here, keeping the same
# layout as the laptop (including the space in "dataset pipeline1") so that no
# existing config.yaml needs editing.
: "${PROJECT:=/data/project/<group>/<firstname>_<initial>_plasmid}"

# Slurm account from `check_group`. Left empty, --account is simply not passed.
: "${CUAIM_ACCOUNT:=}"

# ------------------------------------------------------------------- Slurm ----
# CPU work: Platon and MOB-suite. cpu40h is documented as 40 CPUs / 1 TB / 4 days.
: "${QOS_CPU:=cpu40h}"
: "${PART_CPU:=cpu}"
: "${CPUS_PER_TOOL:=8}"
# 5 x 8 = 40 CPUs. The manual (5.1.2) warns that limits may apply to the COMBINED
# activity of a user or QoS, so the array is throttled to stay inside 40 either way.
: "${MAX_PARALLEL:=5}"
: "${MEM_PER_TOOL:=16G}"
: "${TIME_PER_TOOL:=02:00:00}"

# GPU work: ESM-2. gpu40g is documented as 16 CPUs / 1x7g.40gb / 64 GB / 8 days.
# 7g.40gb is a whole A100-40GB, so there is no MIG memory ceiling to design around.
: "${QOS_GPU:=gpu40g}"
: "${PART_GPU:=gpu}"
: "${GRES_GPU:=gpu:7g.40gb:1}"
: "${GPU_CPUS:=16}"
: "${GPU_MEM:=60G}"
: "${TIME_EMBED:=2-00:00:00}"
: "${TIME_TRAIN:=1-00:00:00}"

# ----------------------------------------------------------------- modules ----
: "${MODULE_CONDA:=Anaconda3}"
: "${MODULE_CUDA:=CUDA/11.8.0}"

# ------------------------------------------------- environments and databases -
: "${ENV_ML:=$PROJECT/envs/plasmid-ml}"
: "${ENV_PLATON:=$PROJECT/envs/platon}"
: "${ENV_MOB:=$PROJECT/envs/mobsuite}"
: "${PLATON_DB:=$PROJECT/db/platon}"
: "${MOB_DB:=$PROJECT/db/mobsuite}"

# "conda" (default) or "singularity". The manual (chapter 6) says to reach for a
# container only when conda cannot provide compatible dependencies; run_tool() in
# lib.sh switches on this single variable.
: "${TOOL_RUNTIME:=conda}"
: "${SIF_PLATON:=$PROJECT/containers/platon.sif}"
: "${SIF_MOB:=$PROJECT/containers/mobsuite.sif}"

# Platon's database is versioned separately from the tool and an older Platon
# cannot read a newer database. Override if the project README names a newer one.
: "${PLATON_DB_URL:=https://zenodo.org/record/4066768/files/db.tar.gz}"
: "${MOB_DB_URL:=https://zenodo.org/records/10304948/files/data.tar.gz}"

# --------------------------------------------------- keep /data/home empty ----
# THE trap of this whole setup: `conda create` defaults to ~/.conda/envs and pip,
# HuggingFace and Singularity all cache under $HOME. Three conda environments
# (~8 GB) plus two tool databases (~7 GB) blow the home quota instantly.
export CONDA_ENVS_DIRS="$PROJECT/envs"
export CONDA_PKGS_DIRS="$PROJECT/.conda-pkgs"
export PIP_CACHE_DIR="$PROJECT/.pip-cache"
export HF_HOME="$PROJECT/.hf"
export TMPDIR="$PROJECT/tmp"
export SINGULARITY_CACHEDIR="$PROJECT/.singularity-cache"
export SINGULARITY_TMPDIR="$PROJECT/.singularity-tmp"

# -------------------------------------------------------------- work layout ---
export REPO="$PROJECT"
export DATASET_DIR="$PROJECT/dataset pipeline1/data/dataset_mixed"
export SHARD_DIR="$PROJECT/tools-baseline/shards"
export TOOL_OUT="$PROJECT/tools-baseline/out"
export SERVER_LOGS="$PROJECT/server/logs"

export PROJECT CUAIM_ACCOUNT QOS_CPU PART_CPU CPUS_PER_TOOL MAX_PARALLEL \
       MEM_PER_TOOL TIME_PER_TOOL QOS_GPU PART_GPU GRES_GPU GPU_CPUS GPU_MEM \
       TIME_EMBED TIME_TRAIN MODULE_CONDA MODULE_CUDA ENV_ML ENV_PLATON ENV_MOB \
       PLATON_DB MOB_DB TOOL_RUNTIME SIF_PLATON SIF_MOB PLATON_DB_URL MOB_DB_URL
