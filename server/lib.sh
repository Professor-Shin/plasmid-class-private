#!/bin/bash
# Shared helpers for every CU-AIM job script. Source env.sh first.

# `module` is a shell function defined by /etc/profile.d, which only a login
# shell sources. Every job script therefore uses `#!/bin/bash -l` (as all the
# manual's listings do), but `bash script.sh` from an already-set-up shell does
# not pass the function on to the child, so re-create it from MODULESHOME when
# it is missing rather than failing with "module: command not found".
cuaim_init_modules() {
  declare -F module >/dev/null && return 0
  command -v module >/dev/null && return 0
  local init
  for init in "${MODULESHOME:-/usr/share/Modules}/init/bash" \
              /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
              /usr/share/lmod/lmod/init/bash; do
    [[ -r "$init" ]] && { source "$init"; declare -F module >/dev/null && return 0; }
  done
  echo "FATAL: 'module' is unavailable and no modules init script was found." >&2
  echo "       Run this script with bash -l, or set MODULESHOME." >&2
  return 1
}

# Load the shared software stack. A batch job starts in a fresh non-interactive
# shell and does NOT inherit the modules of the terminal that submitted it
# (manual 4.4.5), so every job script must call this itself.
cuaim_modules() {
  cuaim_init_modules || return 1
  module purge 2>/dev/null || true
  module load "$MODULE_CONDA" || { echo "FATAL: module load $MODULE_CONDA failed" >&2; return 1; }
  if [[ "${1:-}" == "gpu" ]]; then
    module load "$MODULE_CUDA" || echo "WARN: module load $MODULE_CUDA failed (torch wheels ship their own runtime)" >&2
  fi
  eval "$(conda shell.bash hook)" || return 1
  module list 2>&1 | sed 's/^/  module| /'
}

# Activate a conda environment by absolute prefix. Using --prefix paths keeps the
# environments in project storage instead of the small home quota.
cuaim_activate() {
  local prefix="$1"
  [[ -d "$prefix" ]] || { echo "FATAL: no conda env at $prefix - run 00_setup_envs.sh" >&2; return 1; }
  conda activate "$prefix" || return 1
  echo "  python| $(command -v python) ($(python --version 2>&1))"
}

# Run one of the external tools under whichever runtime env.sh selects, so that
# falling back from bioconda to a Singularity image is a one-variable change.
#   run_tool platon <args...>   |   run_tool mob_recon <args...>
run_tool() {
  local tool="$1"; shift
  if [[ "$TOOL_RUNTIME" == "singularity" ]]; then
    local sif
    case "$tool" in
      platon)    sif="$SIF_PLATON" ;;
      mob_recon) sif="$SIF_MOB" ;;
      *) echo "FATAL: run_tool does not know '$tool'" >&2; return 2 ;;
    esac
    [[ -f "$sif" ]] || { echo "FATAL: missing image $sif" >&2; return 1; }
    singularity exec --bind "$PROJECT:$PROJECT:rw" "$sif" "$tool" "$@"
  else
    command -v "$tool" >/dev/null || { echo "FATAL: $tool not on PATH" >&2; return 1; }
    "$tool" "$@"
  fi
}

# One array task maps to one line of the shard manifest. Line 1 is the header, so
# array index i reads line i+1 and the array is submitted as 1-N.
shard_for_task() {
  local manifest="$1" idx="$2"
  [[ -f "$manifest" ]] || { echo "FATAL: no manifest $manifest - run 10_shard.sh" >&2; return 1; }
  awk -F',' -v n="$((idx + 1))" 'NR==n {print $1}' "$manifest"
}

# Fail loudly and early rather than producing a half-written result directory.
require_dirs() {
  local d
  for d in "$@"; do
    [[ -d "$d" ]] || { echo "FATAL: missing directory: $d" >&2; return 1; }
  done
}

banner() { echo; echo "=== $* ==="; echo "    host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} at $(date -Is)"; }
