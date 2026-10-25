#!/bin/bash -l
# One-time setup: creates the three conda environments in project storage and
# downloads the two tool databases plus the ESM-2 weights.
#
# Run it inside a small CPU allocation, NOT bare on the login node. The manual
# (2.1) reserves the login node for file management and submission, and a
# bioconda solve is heavy enough that the login node kills it (exit 137):
#
#   cd "$PROJECT" && source server/env.sh
#   srun -p "$PART_CPU" --qos="$QOS_CPU" -c 4 --mem=16G -t 2:00:00 --pty bash -l
#   cd "$PROJECT" && bash server/00_setup_envs.sh 2>&1 | tee server/logs/setup.txt
#
# The downloads work either way; if the compute nodes turn out to have no
# internet access (probe_server.sh section 6), run the script once on the login
# node to fetch the databases, then again inside the allocation for the solves -
# each step is skipped once its own output is in place.
#
# Safe to re-run: every step is skipped if its output already exists, so an
# interrupted download or a failed solve can simply be retried.
#
# Why three environments: MOB-suite pins numpy<1.23.5 and pandas<=1.5.3, which
# cannot coexist with the numpy 2.4.4 / pandas 3.0.2 this project's own code is
# pinned to, and Platon pins its own biopython and blast versions on top.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source "$HERE/lib.sh"

if [[ "$PROJECT" == *"<group>"* ]]; then
  echo "FATAL: PROJECT in server/env.sh is still a placeholder." >&2
  echo "       Run 'bash server/probe_server.sh' and fill it in first." >&2
  exit 2
fi

banner "setup | PROJECT=$PROJECT"
mkdir -p "$PROJECT"/envs "$PROJECT"/db "$PROJECT"/tmp "$PROJECT"/containers \
         "$PROJECT"/.conda-pkgs "$PROJECT"/.pip-cache "$PROJECT"/.hf "$SERVER_LOGS"
df -h "$PROJECT" | sed 's/^/  df| /'

cuaim_modules gpu

# A faster solver if the cluster has one: conda's classic solver can take 20+
# minutes on the bioconda dependency graphs below AND needs several GB of RAM,
# which is what gets a solve killed on a login node (seen: "Killed" with exit
# 137 while solving mob_suite on cuaim-frontend1). libmamba ships with recent
# Anaconda3 modules and solves the same specs in a fraction of the memory.
SOLVER=(conda)
SOLVER_ARGS=()
if command -v mamba >/dev/null; then
  SOLVER=(mamba)
  echo "  using mamba for environment solves"
elif conda create --solver=libmamba --help >/dev/null 2>&1; then
  # The flag only parses when conda-libmamba-solver is actually installed.
  SOLVER_ARGS=(--solver=libmamba)
  echo "  using conda's libmamba solver for environment solves"
else
  echo "  WARN: only conda's classic solver is available - it may need several GB"
  echo "        of RAM and 20+ minutes per environment. CU-AIM ships Anaconda3/2021.11"
  echo "        (conda 4.x), which has no libmamba plugin, so the fix is either an"
  echo "        srun allocation or a standalone micromamba - see MANUAL.md step 2."
fi

# An env counts as done only when mark_ok has written its .setup_ok sentinel,
# which the caller does after proving the env actually works. A half-built env
# from an interrupted or failed solve has no sentinel, so a rerun rebuilds it
# instead of skipping over something broken (the bug the reviewer flagged: the old
# check skipped on "python exists", which a failed torch install leaves behind).
create_env() {         # create_env <prefix> <description> <packages...>
  local prefix="$1" desc="$2"
  shift 2
  if [[ -f "$prefix/.setup_ok" ]]; then
    echo "  SKIP  $desc already set up and verified ($prefix)"
    return 0
  fi
  echo "  CREATE $desc -> $prefix"
  rm -rf "$prefix"                 # drop any half-built leftover before rebuilding
  local rc=0
  # ${a[@]+...}: bash 4.2 (CentOS 7, the glibc 2.17 host) treats an empty array
  # as unbound under `set -u`.
  "${SOLVER[@]}" create -y ${SOLVER_ARGS[@]+"${SOLVER_ARGS[@]}"} -p "$prefix" \
    -c conda-forge -c bioconda "$@" || rc=$?
  if (( rc == 137 || rc == 139 )); then
    echo "FATAL: the solve for '$desc' was killed by the kernel (exit $rc) - it ran" >&2
    echo "       out of memory, not out of packages. This is what a login node does" >&2
    echo "       to a classic-solver bioconda solve. Two fixes, either is enough:" >&2
    echo "         1) rerun this script inside an allocation (manual 4.2.1):" >&2
    echo "              srun -p \$PART_CPU --qos=\$QOS_CPU -c 4 --mem=16G -t 2:00:00 --pty bash -l" >&2
    echo "         2) install a leaner solver once, then rerun:" >&2
    echo "              conda install -n base -c conda-forge conda-libmamba-solver" >&2
    return "$rc"
  fi
  return "$rc"
}

mark_ok() { touch "$1/.setup_ok"; }

banner "1/6  conda environment: plasmid-ml (this project's own code + ESM-2)"
# Python 3.11, not 3.13: it pairs with the glibc-2.17 torch 2.5.1 wheel selected below
create_env "$ENV_ML" "plasmid-ml" python=3.11

# torch version is capped by the system glibc, not only by the driver. PyTorch
# 2.6 moved its Linux wheels to manylinux_2_28 (needs glibc >= 2.28); CU-AIM runs
# glibc 2.17 - the same reason the manual pins VS Code to <= 1.85.2, since 1.86+
# needs glibc 2.28 too. On glibc 2.17 a pip-installed 2.6+ wheel installs but
# fails to load, so torch is capped at 2.5.1, the last release built for 2.17.
GLIBC="$(ldd --version 2>/dev/null | sed -n '1s/.*[^0-9.]\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
TORCH_SPEC="torch"
CU_OPTIONS="cu126 cu124 cu121 cu118"      # newest first
if [[ -n "$GLIBC" ]]; then
  GMAJ="${GLIBC%%.*}"; GMIN="${GLIBC##*.}"
  echo "  system glibc: $GLIBC"
  if (( GMAJ < 2 || (GMAJ == 2 && GMIN < 28) )); then
    TORCH_SPEC="torch==2.5.1"
    CU_OPTIONS="cu124 cu121 cu118"        # 2.5.1 ships these, not cu126
    echo "  glibc < 2.28 -> capping torch at 2.5.1 (last glibc-2.17 wheel)"
  fi
else
  echo "  WARN: could not read glibc (ldd missing?); not capping the torch version"
fi

# Pick the newest CUDA wheel the driver supports, among the options glibc allows.
CUDA_TAG="${CU_OPTIONS%% *}"              # best available, as a safe default
if command -v nvidia-smi >/dev/null; then
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
  if [[ -n "${DRIVER_CUDA:-}" ]]; then
    MAJ="${DRIVER_CUDA%%.*}"; MIN="${DRIVER_CUDA##*.}"
    want=118
    (( MAJ > 12 || (MAJ == 12 && MIN >= 1) )) && want=121
    (( MAJ > 12 || (MAJ == 12 && MIN >= 4) )) && want=124
    (( MAJ > 12 || (MAJ == 12 && MIN >= 6) )) && want=126
    for c in $CU_OPTIONS; do
      if (( ${c#cu} <= want )); then CUDA_TAG="$c"; break; fi
    done
    echo "  driver CUDA $DRIVER_CUDA -> torch wheel index $CUDA_TAG"
  fi
else
  echo "  nvidia-smi not on the login node (expected) -> defaulting to $CUDA_TAG"
fi
{ echo "CUDA_TAG=$CUDA_TAG"; echo "TORCH_SPEC=$TORCH_SPEC"; echo "GLIBC=${GLIBC:-unknown}"; } \
    > "$PROJECT/.cuda_tag"

if ! "$ENV_ML/bin/python" -c "import torch" 2>/dev/null; then
  echo "  installing $TORCH_SPEC from the $CUDA_TAG wheel index"
  "$ENV_ML/bin/python" -m pip install --no-cache-dir \
      --index-url "https://download.pytorch.org/whl/$CUDA_TAG" "$TORCH_SPEC"
else
  echo "  SKIP  torch already installed"
fi
# --only-binary=:all: is important on glibc 2.17: if a pinned version has no
# manylinux2014 wheel, pip fails fast with a clear "no matching distribution"
# instead of trying to compile from source and dying in a wall of C errors. The
# fix in that case is to relax that one pin - the project code uses none of the
# APIs that moved, and transformers 5.x runs on torch 2.5.
"$ENV_ML/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
    -r "$HERE/requirements-plasmid-ml.txt"
# A dependency could try to drag torch back up to a glibc-2.28 wheel; confirm it
# did not, since that reintroduces the exact failure this whole section avoids.
"$ENV_ML/bin/python" - <<'PY'
import torch
assert torch.__version__.startswith("2.5."), \
    f"a dependency changed torch to {torch.__version__}; pin it back to 2.5.1"
print(f"  torch still {torch.__version__}")
PY
"$ENV_ML/bin/python" "$HERE/_check_ml_env.py"
# A full version lock, for the paper's Methods and to document why the server's
# regression diff is not exactly zero (different torch/numpy than the laptop). It
# lives under server/ so it can be committed; envs/ itself is gitignored.
"$ENV_ML/bin/python" -m pip freeze > "$REPO/server/plasmid-ml-server.lock.txt"
echo "  wrote server/plasmid-ml-server.lock.txt ($(wc -l < "$REPO/server/plasmid-ml-server.lock.txt") packages) - commit it"
mark_ok "$ENV_ML"

banner "2/6  conda environment: platon (pulls prodigal, diamond, hmmer, blast, mummer)"
create_env "$ENV_PLATON" "platon" platon
# Verify the binary actually runs before marking the env done, so a half-failed
# solve is rebuilt on the next run instead of silently skipped.
"$ENV_PLATON/bin/platon" --help >/dev/null 2>&1 && mark_ok "$ENV_PLATON"   || { echo "FATAL: platon did not install correctly in $ENV_PLATON" >&2; exit 1; }

banner "3/6  conda environment: mobsuite (pulls blast+, mash)"
create_env "$ENV_MOB" "mobsuite" mob_suite
"$ENV_MOB/bin/mob_recon" --help >/dev/null 2>&1 && mark_ok "$ENV_MOB"   || { echo "FATAL: mob_recon did not install correctly in $ENV_MOB" >&2; exit 1; }

banner "4/6  Platon database"
# Platon's database is versioned separately from the tool and an older Platon
# cannot read a newer database, so the tool version is recorded next to the DB.
if [[ -d "$PLATON_DB" ]] && [[ -n "$(ls -A "$PLATON_DB" 2>/dev/null)" ]]; then
  echo "  SKIP  $PLATON_DB already populated ($(du -sh "$PLATON_DB" | cut -f1))"
else
  mkdir -p "$PLATON_DB"
  TARBALL="$PROJECT/tmp/platon-db.tar.gz"
  echo "  downloading $PLATON_DB_URL (~1.6 GB)"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$TARBALL" "$PLATON_DB_URL"
  echo "  extracting (~2.8 GB)"
  tar -xzf "$TARBALL" -C "$PROJECT/tmp"
  # the archive unpacks to a single directory (db/); move its contents up
  SRC="$(find "$PROJECT/tmp" -maxdepth 2 -type d -name db | head -1)"
  if [[ -z "$SRC" ]]; then
    echo "FATAL: no db/ directory inside the archive" >&2
    exit 1
  fi
  mv "$SRC"/* "$PLATON_DB"/
  rm -rf "$TARBALL" "$SRC"
fi
PLATON_VER="$("$ENV_PLATON/bin/platon" --version 2>&1 | head -1 || echo unknown)"
{
  echo "database url:   $PLATON_DB_URL"
  echo "platon version: $PLATON_VER"
  echo "downloaded:     $(date -Is)"
} > "$PLATON_DB/PROVENANCE.txt"
echo "  platon version: $PLATON_VER"
echo "  NOTE: the Platon documentation states that an older Platon release cannot"
echo "        read the latest database. If the version above does not match the"
echo "        database in env.sh, check the project GitHub for the current database"
echo "        and re-run with PLATON_DB_URL=<new url>."

banner "5/6  MOB-suite database"
if [[ -d "$MOB_DB" ]] && [[ -n "$(ls -A "$MOB_DB" 2>/dev/null)" ]]; then
  echo "  SKIP  $MOB_DB already populated ($(du -sh "$MOB_DB" | cut -f1))"
else
  mkdir -p "$MOB_DB"
  # mob_init downloads and indexes the databases itself. The flag name has changed
  # between releases, so discover it from --help instead of guessing.
  MOB_HELP="$("$ENV_MOB/bin/mob_init" --help 2>&1 || true)"
  DBFLAG=""
  for f in --database_directory -d; do
    if grep -q -- "$f" <<<"$MOB_HELP"; then DBFLAG="$f"; break; fi
  done
  if [[ -n "$DBFLAG" ]]; then
    echo "  running: mob_init $DBFLAG $MOB_DB"
    "$ENV_MOB/bin/mob_init" "$DBFLAG" "$MOB_DB"
  else
    echo "  mob_init exposes no database-directory flag; using the manual archive"
    TARBALL="$PROJECT/tmp/mob-db.tar.gz"
    curl -fL --retry 3 --retry-delay 5 -C - -o "$TARBALL" "$MOB_DB_URL"
    tar -xzf "$TARBALL" -C "$MOB_DB" --strip-components=1
    rm -f "$TARBALL"
    echo "  NOTE: a manually extracted database may still need mob_init to build"
    echo "        its BLAST/mash indexes. 13_smoke_tools.sh will catch it if so."
  fi
fi
echo "  mob_recon version: $("$ENV_MOB/bin/mob_recon" --version 2>&1 | head -1 || echo unknown)"

banner "6/6  pre-download ESM-2 weights (compute nodes may have no internet)"
"$ENV_ML/bin/python" "$HERE/_cache_esm_weights.py"

# Record the exact CLI of both tools. The collectors parse output BY COLUMN NAME
# and assert the columns they need, so a schema change fails loudly instead of
# silently producing wrong numbers - this file is the reference for debugging it.
HELPLOG="$SERVER_LOGS/tool_help_$(date +%Y%m%d).txt"
{
  echo "### captured $(date -Is) on $(hostname)"
  for exe in "$ENV_PLATON/bin/platon" "$ENV_MOB/bin/mob_recon" "$ENV_MOB/bin/mob_init"; do
    echo
    echo "### $exe --version"
    "$exe" --version 2>&1 | head -3 || true
    echo
    echo "### $exe --help"
    "$exe" --help 2>&1 || true
  done
} > "$HELPLOG"
echo "  wrote $HELPLOG"

banner "setup complete"
du -sh "$PROJECT"/envs/* "$PLATON_DB" "$MOB_DB" "$HF_HOME" 2>/dev/null | sed 's/^/  /'
df -h "$PROJECT" | sed 's/^/  df| /'
echo
echo "Next steps:"
echo "  1) on the LAPTOP:  bash server/01_sync_from_laptop.sh   (dataset, ~2.5 GB)"
echo "  2) on CU-AIM:      sbatch server/02_verify_data.sh      (checks it arrived intact)"
echo "  3) on CU-AIM:      bash server/submit_all.sh            (submits the pipeline)"
