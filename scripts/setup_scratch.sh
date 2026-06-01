#!/usr/bin/env bash
# =============================================================================
# setup_scratch.sh  —  UCL TSG GPU Workstation Environment Bootstrap
# =============================================================================
#
# PURPOSE
# -------
# Provisions a Python virtual environment and ALL dependency caches in the
# local scratch disk (/scratch0/$USER/), completely bypassing the 10GB network
# quota on your persistent home directory (~/).
#
# UCL TSG QUOTA ARCHITECTURE
# --------------------------
# Your ~/  is a network-mounted NFS share with a hard 10GB quota.
# PyTorch alone is ~2-3GB. Installing it into ~/  would exhaust your quota.
#
# /scratch0/$USER/ is a local NVME disk (typically 1TB+) with NO quota limit.
# It is wiped when your 72-hour GPU booking expires, so we only store
# re-creatable artefacts here (venv, pip cache, model cache).
#
# SCRATCH LAYOUT (created by this script, wiped after booking)
# ------------------------------------------------------------
#   /scratch0/$USER/
#     arm_tracking_venv/  <- Python venv with all deps (~3-5GB)
#     .cache/
#       uv/               <- uv package cache (avoids re-downloading wheels)
#       pip/              <- pip fallback cache
#       torch/            <- PyTorch hub cache (backbone weights)
#
# USAGE
# -----
#   bash ~/arm_path_tracking/scripts/setup_scratch.sh
#
# Run this ONCE per GPU booking. If your booking expires and you get a new
# machine, run it again — the scratch space will be empty.
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, or pipe failures

# ---------------------------------------------------------------------------
# 0. CONFIGURATION — adjust these paths if your layout differs
# ---------------------------------------------------------------------------

# The UCL scratch disk — this is local to the GPU node, no quota
SCRATCH_BASE="/scratch0/${USER}"

# Where the uv-managed venv will live (on fast local disk)
VENV_DIR="${SCRATCH_BASE}/arm_tracking_venv"

# All caches redirected to scratch — CRITICAL to set before any install step
CACHE_DIR="${SCRATCH_BASE}/.cache"

# Your project root (adjust if different)
PROJECT_ROOT="${HOME}/Documents/arm-path-tracking"

# ---------------------------------------------------------------------------
# 1. REDIRECT ALL CACHES TO SCRATCH — set BEFORE touching pip/uv
# ---------------------------------------------------------------------------
# These variables must be exported so every child process (pip, uv, torch)
# inherits them.

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"

# PyTorch hub cache — backbone weights land here.
# Without this, torch.hub defaults to ~/.cache/torch which is on the 10GB NFS
# home quota and will exhaust it when model weights download.
export TORCH_HOME="${CACHE_DIR}/torch"

# Tell uv to create the project's venv in scratch rather than in project dir
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

# MuJoCo rendering backend for headless training
export MUJOCO_GL="egl"

echo "================================================================="
echo "  Arm Path Tracking — UCL TSG GPU Setup"
echo "================================================================="
echo "  User       : ${USER}"
echo "  Scratch    : ${SCRATCH_BASE}"
echo "  Venv       : ${VENV_DIR}"
echo "  Project    : ${PROJECT_ROOT}"
echo "================================================================="

# ---------------------------------------------------------------------------
# 2. CREATE SCRATCH DIRECTORY TREE
# ---------------------------------------------------------------------------

echo ""
echo "[1/4] Creating scratch directory structure..."

mkdir -p "${VENV_DIR}"
mkdir -p "${CACHE_DIR}/pip"
mkdir -p "${CACHE_DIR}/uv"
mkdir -p "${CACHE_DIR}/torch/hub/checkpoints"

# Ensure outputs directory exists in project (for checkpoints/logs)
mkdir -p "${PROJECT_ROOT}/outputs"

echo "    OK — scratch directories created."

# ---------------------------------------------------------------------------
# 3. VERIFY PRE-CONDITIONS
# ---------------------------------------------------------------------------

echo ""
echo "[2/4] Verifying pre-conditions..."

# Check that the project exists
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo ""
    echo "ERROR: Cannot find project at ${PROJECT_ROOT}"
    echo "       Expected pyproject.toml not found."
    echo ""
    exit 1
fi
echo "    Project found at ${PROJECT_ROOT}"

# Check uv is available — install if not
if ! command -v uv &>/dev/null; then
    echo "    uv not found in PATH. Installing uv to scratch space..."
    export CARGO_HOME="${SCRATCH_BASE}/.cargo"
    curl -LsSf https://astral.sh/uv/install.sh | \
        env INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR="${SCRATCH_BASE}/bin" sh
    export PATH="${SCRATCH_BASE}/bin:${PATH}"
    echo "    uv installed to ${SCRATCH_BASE}/bin/uv"
else
    echo "    uv found: $(which uv) — $(uv --version)"
fi

# Confirm CUDA is visible
if command -v nvidia-smi &>/dev/null; then
    echo "    GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | \
        sed 's/^/        /'
else
    echo "    WARNING: nvidia-smi not found. Training will use CPU."
fi

# ---------------------------------------------------------------------------
# 4. INSTALL DEPENDENCIES INTO SCRATCH VENV
# ---------------------------------------------------------------------------

echo ""
echo "[3/4] Installing dependencies into scratch venv..."
echo "      (Downloads PyTorch ~2GB + MuJoCo — takes 3-10 min)"
echo "      Venv target : ${VENV_DIR}"
echo "      Cache target: ${UV_CACHE_DIR}"
echo ""

cd "${PROJECT_ROOT}"

# Sync the project with all dependencies
# Using --extra dev for testing tools, --extra viz if you need plotting
uv sync --extra dev 2>&1 | tee "${SCRATCH_BASE}/setup_install.log"

# Verify PyTorch + CUDA
echo ""
echo "    Verifying PyTorch + CUDA inside the new venv..."
uv run python -c "
import torch
print(f'    torch version  : {torch.__version__}')
print(f'    CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    CUDA device    : {torch.cuda.get_device_name(0)}')
    print(f'    CUDA version   : {torch.version.cuda}')
else:
    print('    WARNING: CUDA not available — training will use CPU')
"

# Verify MuJoCo
echo ""
echo "    Verifying MuJoCo..."
uv run python -c "
import mujoco
print(f'    MuJoCo version : {mujoco.__version__}')
"

# ---------------------------------------------------------------------------
# 5. WRITE ENVIRONMENT ACTIVATION HELPERS
# ---------------------------------------------------------------------------

ACTIVATE_SHIM="${SCRATCH_BASE}/activate_arm_tracking.sh"
ACTIVATE_CSH="${SCRATCH_BASE}/activate_arm_tracking.csh"

# Resolve the uv binary location
UV_BIN_DIR="$(dirname "$(command -v uv)")"

# --- bash shim ---
cat > "${ACTIVATE_SHIM}" << SHIM_EOF
# Auto-generated by setup_scratch.sh — source in bash
# Usage:  source ${ACTIVATE_SHIM}

export PATH="${UV_BIN_DIR}:\$PATH"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export TORCH_HOME="${CACHE_DIR}/torch"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export MUJOCO_GL="egl"

echo "Arm tracking environment activated."
echo "  Venv  : ${VENV_DIR}"
echo "  Torch : ${CACHE_DIR}/torch"
SHIM_EOF

# --- tcsh shim ---
cat > "${ACTIVATE_CSH}" << CSH_EOF
# Auto-generated by setup_scratch.sh — source in tcsh
# Usage:  source ${ACTIVATE_CSH}

setenv PATH "${UV_BIN_DIR}:\$PATH"
setenv PIP_CACHE_DIR "${CACHE_DIR}/pip"
setenv UV_CACHE_DIR "${CACHE_DIR}/uv"
setenv TORCH_HOME "${CACHE_DIR}/torch"
setenv UV_PROJECT_ENVIRONMENT "${VENV_DIR}"
setenv MUJOCO_GL "egl"

echo "Arm tracking environment activated."
echo "  Venv  : ${VENV_DIR}"
CSH_EOF

chmod +x "${ACTIVATE_SHIM}" "${ACTIVATE_CSH}"

echo ""
echo "[4/4] Environment activation shims written:"
echo "      bash  — ${ACTIVATE_SHIM}"
echo "      tcsh  — ${ACTIVATE_CSH}"

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

echo ""
echo "================================================================="
echo "  Setup complete."
echo "================================================================="
echo ""
echo "  TO RUN TRAINING:"
echo ""
echo "    cd ${PROJECT_ROOT}"
echo "    source ${ACTIVATE_SHIM}"
echo "    uv run python scripts/train_circle.py"
echo ""
echo "  Or run in background with nohup:"
echo ""
echo "    nohup uv run python scripts/train_circle.py > train.log 2>&1 &"
echo ""
echo "  To activate the env interactively:"
echo "    bash shell : source ${ACTIVATE_SHIM}"
echo "    tcsh shell : source ${ACTIVATE_CSH}"
echo ""
echo "  REMINDER: This scratch space will be wiped when your 72-hour"
echo "  GPU booking expires. Re-run setup_scratch.sh on your next booking."
echo "  Your outputs in ${PROJECT_ROOT}/outputs/ are in home (persistent)."
echo "================================================================="
