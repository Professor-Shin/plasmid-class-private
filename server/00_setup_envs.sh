#!/bin/bash
# One-time setup, run on the CU-AIM LOGIN NODE (not under Slurm): creates the
# three conda environments in project storage and downloads the two tool
# databases plus the ESM-2 weights.
#
#   bash server/00_setup_envs.sh 2>&1 | tee server/logs/setup.txt
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
# minutes on the bioconda dependency graphs below.
SOLVER=(conda)
if command -v mamba >/dev/null; then
  SOLVER=(mamba)
  echo "  using mamba for environment solves"
fi

create_env() {         # create_env <prefix> <description> <packages...>
  local prefix="$1" desc="$2"
  shift 2
  if [[ -x "$prefix/bin/python" ]]; then
    echo "  SKIP  $desc already exists at $prefix"
    return 0
  fi
  echo "  CREATE $desc -> $prefix"
  "${SOLVER[@]}" create -y -p "$prefix" -c conda-forge -c bioconda "$@"
}

banner "1/6  conda environment: plasmid-ml (this project's own code + ESM-2)"
create_env "$ENV_ML" "plasmid-ml" python=3.13

# torch: pick the CUDA wheel the installed driver can actually run. nvidia-smi
# reports the newest CUDA the driver supports (manual 4.4.2). A login node usually
# has no GPU, so fall back to cu126 - the version the manual's own Singularity
# example installs.
CUDA_TAG="cu126"
if command -v nvidia-smi >/dev/null; then
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
  if [[ -n "${DRIVER_CUDA:-}" ]]; then
    MAJ="${DRIVER_CUDA%%.*}"
    MIN="${DRIVER_CUDA##*.}"
    if   (( MAJ > 12 || (MAJ == 12 && MIN >= 6) )); then CUDA_TAG="cu126"
    elif (( MAJ == 12 ));                           then CUDA_TAG="cu121"
    else                                                 CUDA_TAG="cu118"
    fi
    echo "  driver supports CUDA $DRIVER_CUDA -> torch wheel index $CUDA_TAG"
  fi
else
  echo "  nvidia-smi not on the login node (expected) -> defaulting to $CUDA_TAG"
fi
echo "CUDA_TAG=$CUDA_TAG" > "$PROJECT/.cuda_tag"

if ! "$ENV_ML/bin/python" -c "import torch" 2>/dev/null; then
  echo "  installing torch from the $CUDA_TAG wheel index"
  "$ENV_ML/bin/python" -m pip install --no-cache-dir \
      --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch
else
  echo "  SKIP  torch already installed"
fi
"$ENV_ML/bin/python" -m pip install --no-cache-dir -r "$HERE/requirements-plasmid-ml.txt"
"$ENV_ML/bin/python" "$HERE/_check_ml_env.py"

banner "2/6  conda environment: platon (pulls prodigal, diamond, hmmer, blast, mummer)"
create_env "$ENV_PLATON" "platon" platon

banner "3/6  conda environment: mobsuite (pulls blast+, mash)"
create_env "$ENV_MOB" "mobsuite" mob_suite

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
