#!/bin/bash -l
# One-time setup: creates the five conda environments in project storage and
# downloads the four tool databases plus the ESM-2 weights.
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
# Why five environments: MOB-suite pins numpy<1.23.5 and pandas<=1.5.3, which
# cannot coexist with the numpy 2.4.4 / pandas 3.0.2 this project's own code is
# pinned to; Platon pins its own biopython and blast versions on top; PLASMe
# needs Python 3.9 with torch and its own BLAST/DIAMOND; and PlasmidHunter's
# setup.py demands `python_requires == 3.10.*` with numpy==1.25.0 and
# scikit-learn==1.3.2 exactly. No two of those five sets of pins are satisfiable
# together, which is why each tool gets its own prefix rather than one shared env.
#
# Disk: envs ~14 GB, databases ~22 GB (Platon 2.8 + MOB 2.9 + PLASMe ~16 once the
# 12.4 GB archive is expanded and indexed + PlasmidHunter ~0.4), ESM weights
# 0.2 GB. The PLASMe archive is deleted after its indexes are built.

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

banner "1/10  conda environment: plasmid-ml (this project's own code + ESM-2)"
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

banner "2/10  conda environment: platon (pulls prodigal, diamond, hmmer, blast, mummer)"
create_env "$ENV_PLATON" "platon" platon
# Verify the binary actually runs before marking the env done, so a half-failed
# solve is rebuilt on the next run instead of silently skipped.
"$ENV_PLATON/bin/platon" --help >/dev/null 2>&1 && mark_ok "$ENV_PLATON"   || { echo "FATAL: platon did not install correctly in $ENV_PLATON" >&2; exit 1; }

banner "3/10  conda environment: mobsuite (pulls blast+, mash)"
create_env "$ENV_MOB" "mobsuite" mob_suite
"$ENV_MOB/bin/mob_recon" --help >/dev/null 2>&1 && mark_ok "$ENV_MOB"   || { echo "FATAL: mob_recon did not install correctly in $ENV_MOB" >&2; exit 1; }

banner "4/10  conda environment: plasme (Python 3.9 + BLAST + DIAMOND + prodigal)"
# PLASMe is distributed as a git checkout, not a package: PLASMe.py imports
# bio_script, trans_model and trans_data as sibling modules, so the repository
# itself has to be on disk and invoked by path.
if [[ -d "$PLASME_DIR/.git" ]]; then
  echo "  SKIP  $PLASME_DIR already cloned ($(git -C "$PLASME_DIR" rev-parse --short HEAD))"
else
  mkdir -p "$(dirname "$PLASME_DIR")"
  git clone --depth 1 --branch "$PLASME_REPO_REF" "$PLASME_REPO_URL" "$PLASME_DIR"
fi
echo "  PLASMe commit: $(git -C "$PLASME_DIR" rev-parse HEAD) ($(git -C "$PLASME_DIR" log -1 --format=%cI))"

# Upstream's plasme.yaml pins pytorch=1.11.0 WITH cudatoolkit=10.2.89, a ~2 GB
# GPU runtime this cluster has no use for (PLASMe runs on the CPU partition
# here - see 16_plasme.sh) and which conda would have to solve against a CUDA
# 12.4 driver. The binaries that decide PLASMe's output - BLAST, DIAMOND,
# prodigal - are pinned to upstream's versions; torch is installed separately
# below, because the solve for it is the fragile part and pip is deterministic.
create_env "$ENV_PLASME" "plasme" python=3.9 \
    "blast=2.13.0" "diamond=2.0.15" "prodigal=2.6.3" \
    "biopython=1.80" "numpy=1.23" "pandas=1.5" requests

# torch capped BELOW 2.6 for two independent reasons, either of which alone
# would be enough: glibc 2.17 (see the ENV_ML section above - 2.6+ wheels are
# manylinux_2_28 and will not load), and torch 2.6 flipping `torch.load`'s
# weights_only default to True, which refuses the pickled state dicts in
# PLASMe's DB/trans_model/*.pt. The CPU wheel index is deliberate: a CUDA build
# would pull ~2.5 GB of runtime the CPU partition never touches.
if ! "$ENV_PLASME/bin/python" -c "import torch" 2>/dev/null; then
  echo "  installing torch==2.5.1 (CPU wheel index) into the plasme env"
  "$ENV_PLASME/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
else
  echo "  SKIP  torch already installed in the plasme env"
fi
"$ENV_PLASME/bin/python" - <<'PY'
import torch
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
assert (major, minor) < (2, 6), (
    f"torch {torch.__version__} defaults torch.load(weights_only=True), which "
    "cannot read PLASMe's pickled Transformer checkpoints. Pin it below 2.6.")
print(f"  torch {torch.__version__} (CPU), torch.load default is safe for PLASMe")
PY
for exe in blastn makeblastdb diamond prodigal; do
  "$ENV_PLASME/bin/$exe" -h >/dev/null 2>&1 || "$ENV_PLASME/bin/$exe" --help >/dev/null 2>&1 \
    || { echo "FATAL: $exe did not install correctly in $ENV_PLASME" >&2; exit 1; }
done
mark_ok "$ENV_PLASME"

banner "5/10  conda environment: plasmidhunter (Python 3.10 + DIAMOND + prodigal)"
# setup.py says python_requires == 3.10.*, numpy == 1.25.0, scikit-learn == 1.3.2
# - exact pins, not ranges, so this env can hold nothing else of ours. diamond is
# pinned to 2.1.8 because the README names that version and a DIAMOND database
# built by one major version is not readable by an older one.
create_env "$ENV_PH" "plasmidhunter" python=3.10 "diamond=2.1.8" prodigal
if ! "$ENV_PH/bin/python" -c "import PlasmidHunter" 2>/dev/null; then
  "$ENV_PH/bin/python" -m pip install --no-cache-dir plasmidhunter
else
  echo "  SKIP  plasmidhunter already installed"
fi
"$ENV_PH/bin/plasmidhunter" -h >/dev/null 2>&1 \
  || { echo "FATAL: plasmidhunter did not install correctly in $ENV_PH" >&2; exit 1; }
echo "  plasmidhunter version: $("$ENV_PH/bin/python" -c 'import importlib.metadata as m; print(m.version("plasmidhunter"))' 2>&1 | head -1)"
mark_ok "$ENV_PH"

banner "6/10  Platon database"
# Platon's database is versioned separately from the tool and an older Platon
# cannot read a newer database, so the tool version is recorded next to the DB.
# A sentinel, not "the directory has something in it": a download or extraction
# that died half way leaves plenty of files behind, and skipping over that is how
# a broken database reaches the array jobs.
if [[ -f "$PLATON_DB/.setup_ok" ]]; then
  echo "  SKIP  $PLATON_DB already populated and verified ($(du -sh "$PLATON_DB" | cut -f1))"
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
  mark_ok "$PLATON_DB"
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

banner "7/10  MOB-suite database"
if [[ -f "$MOB_DB/.setup_ok" ]]; then
  echo "  SKIP  $MOB_DB already populated and verified ($(du -sh "$MOB_DB" | cut -f1))"
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
    # mob_init shells out to makeblastdb, blastn and mash BY BARE NAME, so the
    # environment's bin must be on PATH - calling mob_init by absolute path is
    # not enough. Without this it downloads the 450 MB archive, fails while
    # building the indexes, and reports it as "check your internet connection"
    # (the real errno is EACCES on 'makeblastdb').
    PATH="$ENV_MOB/bin:$PATH" "$ENV_MOB/bin/mob_init" "$DBFLAG" "$MOB_DB"
    mark_ok "$MOB_DB"
  else
    echo "  mob_init exposes no database-directory flag; using the manual archive"
    TARBALL="$PROJECT/tmp/mob-db.tar.gz"
    curl -fL --retry 3 --retry-delay 5 -C - -o "$TARBALL" "$MOB_DB_URL"
    tar -xzf "$TARBALL" -C "$MOB_DB" --strip-components=1
    rm -f "$TARBALL"
    # Deliberately NOT marked done: a hand-extracted archive has no BLAST/mash
    # indexes, so the next run should try mob_init again.
    echo "  NOTE: a manually extracted database may still need mob_init to build"
    echo "        its BLAST/mash indexes. 13_smoke_tools.sh will catch it if so."
  fi
fi
echo "  mob_recon version: $("$ENV_MOB/bin/mob_recon" --version 2>&1 | head -1 || echo unknown)"

banner "8/10  PLASMe database (12.4 GB download, ~16 GB once indexed)"
# PLASMe_db.py downloads with `requests` into whatever directory the script sits
# in and refuses to resume a partial file, so the download is done with curl -C -
# here and only the indexing is left to PLASMe's own code path. The md5 is the
# one PLASMe_db.py itself verifies against; a truncated 12 GB download that
# still unzips would otherwise be discovered 116 shards later.
if [[ -f "$PLASME_DB/.setup_ok" ]]; then
  echo "  SKIP  $PLASME_DB already populated and verified ($(du -sh "$PLASME_DB" | cut -f1))"
else
  mkdir -p "$PLASME_DB"
  ZIP="$PROJECT/tmp/PLASMe-DB.zip"
  if [[ ! -f "$PLASME_DB/plsdb_Mar30.dmnd" ]]; then
    if [[ ! -s "$ZIP" ]] || [[ "$(md5sum "$ZIP" | cut -d' ' -f1)" != "$PLASME_DB_MD5" ]]; then
      echo "  downloading $PLASME_DB_URL (12.4 GB; resumable, safe to re-run)"
      curl -fL --retry 5 --retry-delay 10 -C - -o "$ZIP" "$PLASME_DB_URL"
      echo "  verifying md5 (expected $PLASME_DB_MD5)"
      GOT="$(md5sum "$ZIP" | cut -d' ' -f1)"
      if [[ "$GOT" != "$PLASME_DB_MD5" ]]; then
        echo "FATAL: PLASMe DB.zip md5 is $GOT, expected $PLASME_DB_MD5." >&2
        echo "       Delete $ZIP and re-run; curl -C - resumes, so a half file" >&2
        echo "       that keeps failing means the mirror changed, not the link." >&2
        exit 1
      fi
    else
      echo "  SKIP  $ZIP already downloaded and md5-verified"
    fi
    echo "  extracting into $PLASME_DB"
    # The archive contains a single DB/ directory; unpack to tmp and move its
    # contents up, so PLASME_DB can be anywhere rather than having to be a
    # directory literally named DB next to PLASMe.py.
    rm -rf "$PROJECT/tmp/plasme-db-extract"
    mkdir -p "$PROJECT/tmp/plasme-db-extract"
    unzip -q "$ZIP" -d "$PROJECT/tmp/plasme-db-extract"
    SRC="$(find "$PROJECT/tmp/plasme-db-extract" -maxdepth 2 -type d -name DB | head -1)"
    [[ -n "$SRC" ]] || { echo "FATAL: no DB/ directory inside the archive" >&2; exit 1; }
    mv "$SRC"/* "$PLASME_DB"/
    rm -rf "$PROJECT/tmp/plasme-db-extract"

    # The inner plsdb.zip holds the reference FASTA and its protein translation;
    # PLASMe builds the BLAST and DIAMOND indexes from them and then deletes
    # both. Doing it HERE, once on the login node, is the whole point: left to
    # itself PLASMe builds the index on first use, and 116 array tasks starting
    # together would each try to build the same 12 GB index into the same path.
    echo "  building the BLAST and DIAMOND indexes (several minutes)"
    PATH="$ENV_PLASME/bin:$PATH" "$ENV_PLASME/bin/python" - "$PLASME_DB" "${SLURM_CPUS_PER_TASK:-8}" <<'PY'
import subprocess, shutil, sys, os
db_dir, threads = sys.argv[1], sys.argv[2]
inner = os.path.join(db_dir, "plsdb.zip")
if os.path.exists(inner):
    print(f"  unpacking {inner}")
    shutil.unpack_archive(inner, db_dir)
    os.remove(inner)
for cmd in (f"diamond makedb --in {db_dir}/plsdb_Mar30.fna.aa -d {db_dir}/plsdb_Mar30 -p {threads}",
            f"makeblastdb -in {db_dir}/plsdb_Mar30.fna -dbtype nucl -out {db_dir}/plsdb_Mar30"):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)
for leftover in ("plsdb_Mar30.fna", "plsdb_Mar30.fna.aa"):
    path = os.path.join(db_dir, leftover)
    if os.path.exists(path):
        os.remove(path)
PY
    rm -f "$ZIP"
  fi
  for f in plsdb_Mar30.dmnd plsdb_taxon.tsv plas_overlap.csv; do
    [[ -e "$PLASME_DB/$f" ]] || { echo "FATAL: $PLASME_DB/$f is missing after the build" >&2; exit 1; }
  done
  [[ -d "$PLASME_DB/trans_model" ]] || { echo "FATAL: $PLASME_DB/trans_model is missing" >&2; exit 1; }
  echo "  Transformer checkpoints: $(ls "$PLASME_DB/trans_model" | wc -l) (one per taxonomic order, plus unified)"
  mark_ok "$PLASME_DB"
fi
{
  echo "database url: $PLASME_DB_URL"
  echo "md5:          $PLASME_DB_MD5"
  echo "PLASMe commit: $(git -C "$PLASME_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "built:        $(date -Is)"
} > "$PLASME_DB/PROVENANCE.txt"

banner "9/10  PlasmidHunter database (DIAMOND, ~0.4 GB)"
# PlasmidHunter downloads this into its OWN site-packages the first time it is
# run, with no locking and no resume. On a compute node with no internet that is
# a hard failure, and with internet it is 116 simultaneous downloads into one
# path. Fetching it here removes both; 17_plasmidhunter.sh refuses to start
# until it is in place.
PH_PKG="$("$ENV_PH/bin/python" -c 'import os, PlasmidHunter; print(os.path.dirname(PlasmidHunter.__file__))')"
PH_DB_DIR="$PH_PKG/database"
if [[ -s "$PH_DB_DIR/database.dmnd" ]]; then
  echo "  SKIP  $PH_DB_DIR/database.dmnd already present ($(du -h "$PH_DB_DIR/database.dmnd" | cut -f1))"
else
  mkdir -p "$PH_DB_DIR"
  echo "  downloading $PH_DB_URL"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$PH_DB_DIR/database.dmnd.gz" "$PH_DB_URL"
  gunzip -f "$PH_DB_DIR/database.dmnd.gz"
  [[ -s "$PH_DB_DIR/database.dmnd" ]] || { echo "FATAL: PlasmidHunter database did not unpack" >&2; exit 1; }
  echo "  installed $PH_DB_DIR/database.dmnd ($(du -h "$PH_DB_DIR/database.dmnd" | cut -f1))"
fi
# A DIAMOND database built by a newer major version cannot be read by an older
# one, and this file was built by whoever published it - so check it opens with
# the diamond in THIS environment rather than finding out 116 shards in.
"$ENV_PH/bin/diamond" dbinfo --db "$PH_DB_DIR/database.dmnd" 2>&1 | sed 's/^/  dbinfo| /' \
  || { echo "FATAL: diamond $("$ENV_PH/bin/diamond" --version 2>&1 | head -1) cannot read" >&2
       echo "       $PH_DB_DIR/database.dmnd - the published database was built by a" >&2
       echo "       different DIAMOND major version. Pin diamond to match it." >&2; exit 1; }

banner "10/10  pre-download ESM-2 weights (compute nodes may have no internet)"
"$ENV_ML/bin/python" "$HERE/_cache_esm_weights.py"

# Record the exact CLI of both tools. The collectors parse output BY COLUMN NAME
# and assert the columns they need, so a schema change fails loudly instead of
# silently producing wrong numbers - this file is the reference for debugging it.
HELPLOG="$SERVER_LOGS/tool_help_$(date +%Y%m%d).txt"
{
  echo "### captured $(date -Is) on $(hostname)"
  for exe in "$ENV_PLATON/bin/platon" "$ENV_MOB/bin/mob_recon" "$ENV_MOB/bin/mob_init" \
             "$ENV_PH/bin/plasmidhunter"; do
    echo
    echo "### $exe --version"
    "$exe" --version 2>&1 | head -3 || true
    echo
    echo "### $exe --help"
    "$exe" --help 2>&1 || true
  done
  echo
  echo "### PLASMe (git checkout, invoked by path)"
  echo "commit $(git -C "$PLASME_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  "$ENV_PLASME/bin/python" "$PLASME_DIR/PLASMe.py" --version 2>&1 | head -3 || true
  "$ENV_PLASME/bin/python" "$PLASME_DIR/PLASMe.py" --help 2>&1 || true
} > "$HELPLOG"
echo "  wrote $HELPLOG"

banner "setup complete"
du -sh "$PROJECT"/envs/* "$PLATON_DB" "$MOB_DB" "$PLASME_DB" "$PLASME_DIR" \
      "$HF_HOME" 2>/dev/null | sed 's/^/  /'
df -h "$PROJECT" | sed 's/^/  df| /'
echo
echo "Next steps:"
echo "  1) on the LAPTOP:  bash server/01_sync_from_laptop.sh   (dataset, ~2.5 GB)"
echo "  2) on CU-AIM:      sbatch server/02_verify_data.sh      (checks it arrived intact)"
echo "  3) on CU-AIM:      bash server/submit_all.sh            (submits the pipeline)"
