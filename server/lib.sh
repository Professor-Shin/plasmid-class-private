#!/bin/bash
# Shared helpers for every CU-AIM job script. Source env.sh first.

# Where this file lives, resolved once at source time. Everything here that
# needs a sibling script uses this rather than $PROJECT/$REPO: those are
# configuration and can be wrong or still a placeholder, whereas this is simply
# where the code being executed actually is.
CUAIM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
      plasme|plasmidhunter)
        echo "FATAL: no Singularity image is defined for '$tool'. Neither upstream" >&2
        echo "       publishes one, so these two run under conda only - set" >&2
        echo "       TOOL_RUNTIME=conda for them." >&2
        return 2 ;;
      *) echo "FATAL: run_tool does not know '$tool'" >&2; return 2 ;;
    esac
    [[ -f "$sif" ]] || { echo "FATAL: missing image $sif" >&2; return 1; }
    singularity exec --bind "$PROJECT:$PROJECT:rw" "$sif" "$tool" "$@"
  elif [[ "$tool" == "plasme" ]]; then
    # PLASMe is a git checkout, not an installed command: PLASMe.py imports
    # bio_script / trans_model / trans_data as siblings, which Python resolves
    # from the SCRIPT's directory, so invoking it by absolute path works from
    # any working directory - but invoking a copy of it would not.
    [[ -f "$PLASME_DIR/PLASMe.py" ]] || {
      echo "FATAL: no PLASMe.py at $PLASME_DIR - run 00_setup_envs.sh" >&2; return 1; }
    python "$PLASME_DIR/PLASMe.py" "$@"
  else
    command -v "$tool" >/dev/null || { echo "FATAL: $tool not on PATH" >&2; return 1; }
    "$tool" "$@"
  fi
}

# run_tool, with wall clock, CPU seconds and peak RSS recorded to a JSON file.
# Used by every per-shard tool runner so the report can compare what the five
# baselines cost on the same input, which sacct cannot do: it accounts per JOB,
# and the serial runner puts all 116 shards of a tool in one job.
#
#   timed_tool <runtime.json> <tool> <shard> <contigs> <bp> -- <args...>
timed_tool() {
  local json="$1" tool="$2" shard="$3" contigs="$4" bp="$5"; shift 5
  [[ "${1:-}" == "--" ]] && shift
  local wrapper="$CUAIM_LIB_DIR/_timed_run.py"
  if [[ ! -f "$wrapper" ]]; then
    echo "WARN: $wrapper missing; running untimed" >&2
    run_tool "$tool" "$@"
    return
  fi
  # run_tool is a shell FUNCTION (it picks conda vs Singularity vs a path
  # invocation), and the wrapper execs a program, so the function is re-entered
  # through a child bash rather than passed as a command. `bash -l` is not used:
  # the environment is already set up by the caller and a login shell would
  # reset PATH out from under the activated conda env.
  python "$wrapper" --json "$json" --tool "$tool" --shard "$shard" \
      --threads "${CPUS_PER_TOOL:-0}" --contigs "${contigs:-0}" --bp "${bp:-0}" \
      -- bash -c '
        set -euo pipefail
        source "$1/env.sh"
        source "$1/lib.sh"
        shift
        run_tool "$@"
      ' _ "$CUAIM_LIB_DIR" "$tool" "$@"
}

# One field of one shard's manifest row, by column NAME: the manifest's column
# order is not a contract, and the timing record needs n_fragments and bp.
shard_field() {
  local manifest="$1" shard="$2" column="$3"
  awk -F',' -v s="$shard" -v c="$column" '
    NR==1 { for (i = 1; i <= NF; i++) if ($i == c) col = i; next }
    $1 == s && col { print $col; exit }' "$manifest"
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
