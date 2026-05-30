#!/bin/bash
# =============================================================================
# train_trm_sudoku_original_a100.sbatch — TRM Sudoku training (original config)
#
# This keeps the original training configuration from the upstream procedure
# and forces execution on A100 GPUs only.
#
# Usage: sbatch scripts/initial/train_trm_sudoku_original_a100.sbatch [mlp|att]
# =============================================================================
#SBATCH --job-name=trm_sudoku_orig_a100
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodelist=cn[7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/initial/trm_sudoku_orig_a100_%j.out

set -euo pipefail
trap 'echo "ERROR at line ${LINENO}: command failed: ${BASH_COMMAND}" >&2' ERR
echo "Job started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)"
echo "Detected GPU memory: ${GPU_MEM_MB} MB"
if ! echo "${GPU_NAME}" | grep -q "A100"; then
    echo "ERROR: Expected an A100 GPU, got '${GPU_NAME}'."
    exit 1
fi
if [ "${GPU_MEM_MB}" -lt 39000 ]; then
    echo "ERROR: Expected >=40GB VRAM on A100, got ${GPU_MEM_MB} MB."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- Variant selection ----------
VARIANT="${1:-mlp}"  # default to mlp, pass "att" for attention variant
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
echo "eval_interval=${EVAL_INTERVAL}"

# ---------- Modules & Environment ----------
unset -f module 2>/dev/null || true

if ! command -v module >/dev/null 2>&1; then
    set +u
    [ -f /etc/profile ] && source /etc/profile || true
    set -u
fi
if ! command -v module >/dev/null 2>&1; then
    set +u
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
    set -u
fi
if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: Environment Modules is not available in this job shell."
    echo "Continuing with direct Anaconda fallback."
else
    module purge || true
    module use /opt/itu/easybuild/modules/all || true
    module load Anaconda3 || true
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not available via module. Trying direct ITU Anaconda path..."

    set +u
    ANACONDA_CONDA_BIN=""
    for candidate in /opt/itu/easybuild/software/Anaconda3/*/bin/conda; do
        if [ -x "$candidate" ]; then
            ANACONDA_CONDA_BIN="$candidate"
        fi
    done

    if [ -n "$ANACONDA_CONDA_BIN" ]; then
        __conda_setup="$($ANACONDA_CONDA_BIN shell.bash hook 2>/dev/null || true)"
        if [ -n "$__conda_setup" ]; then
            eval "$__conda_setup"
        else
            CONDA_SH="${ANACONDA_CONDA_BIN%/bin/conda}/etc/profile.d/conda.sh"
            if [ -f "$CONDA_SH" ]; then
                . "$CONDA_SH"
            else
                export PATH="${ANACONDA_CONDA_BIN%/bin/conda}/bin:$PATH"
            fi
        fi
        unset __conda_setup
    fi
    set -u
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda still not found after module/direct fallback."
    module list || true
    exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate trm-env
set -u

TORCH_LIB_DIR="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

echo "Conda env active: ${CONDA_DEFAULT_ENV:-unknown}"
python --version

# Ensure adam-atan2 backend is importable
if ! python - <<'PY'
import adam_atan2_backend
print("adam_atan2_backend OK")
PY
then
    echo "adam_atan2_backend missing - reinstalling adam-atan2 in trm-env..."

    if ! command -v nvcc >/dev/null 2>&1; then
        module load CUDA >/dev/null 2>&1 || true
    fi
    if command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    elif [ -d /usr/local/cuda ]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "ERROR: CUDA toolkit not found (nvcc missing), cannot build adam-atan2 backend."
        exit 1
    fi
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    echo "Using CUDA_HOME=${CUDA_HOME} for adam-atan2 build"

    if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ] && [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
        export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
        export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
        export CUDAHOSTCXX="$CXX"
    else
        echo "Conda GCC toolchain not found; attempting install (one-time)."
        set +u
        conda install -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13 || true
        set -u
        if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ] && [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
            export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
            export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
            export CUDAHOSTCXX="$CXX"
        fi
    fi
    echo "Using CC=${CC:-unset}"
    echo "Using CXX=${CXX:-unset}"

    pip install --upgrade pip wheel setuptools
    pip install --no-cache-dir --force-reinstall --no-build-isolation adam-atan2

    python - <<'PY'
import adam_atan2_backend
print("adam_atan2_backend OK after reinstall")
PY
fi

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# ---------- Navigate to TRM codebase ----------
cd "$ROOT/code/initial/TinyRecursiveModels"

# ---------- Dataset location ----------
TRM_DATA_DIR="${TRM_DATA_DIR:-$ROOT/data/initial/sudoku_grid/trm_training}"
echo "TRM dataset dir: $TRM_DATA_DIR"

# ---------- Checkpoint destination ----------
CHECKPOINT_ROOT="$ROOT/checkpoints/initial"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/trm-sudoku-orig-a100/${VARIANT}"
CHECKPOINT_EVERY_EVAL="${CHECKPOINT_EVERY_EVAL:-false}"
mkdir -p "$CHECKPOINT_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "checkpoint_every_eval=${CHECKPOINT_EVERY_EVAL}"

# ---------- Step 1: Validate dataset location ----------
if [ ! -f "$TRM_DATA_DIR/train/dataset.json" ]; then
    echo "ERROR: Expected TRM training dataset at $TRM_DATA_DIR/train/dataset.json"
    echo "Set TRM_DATA_DIR to your prepared dataset root under $ROOT/data/initial/sudoku_grid/trm_training"
    exit 1
fi
if [ ! -f "$TRM_DATA_DIR/test/dataset.json" ]; then
    echo "WARNING: No test split found at $TRM_DATA_DIR/test/dataset.json"
    echo "Training can run, but held-out evaluation inside pretrain.py may be unavailable."
fi

# ---------- Step 2: Train TRM (original hyperparameter flags) ----------
if [ "$VARIANT" = "mlp" ]; then
    run_name="pretrain_mlp_t_sudoku_orig_a100"
    python pretrain.py \
        arch=trm \
        data_paths="[$TRM_DATA_DIR]" \
        evaluators="[]" \
        epochs=50000 eval_interval=${EVAL_INTERVAL} \
        lr=1e-4 puzzle_emb_lr=1e-4 \
        weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.mlp_t=True arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        +run_name=${run_name} ema=True \
        +checkpoint_path=${CHECKPOINT_DIR} \
        checkpoint_every_eval=${CHECKPOINT_EVERY_EVAL}
elif [ "$VARIANT" = "att" ]; then
    run_name="pretrain_att_sudoku_orig_a100"
    python pretrain.py \
        arch=trm \
        data_paths="[$TRM_DATA_DIR]" \
        evaluators="[]" \
        epochs=50000 eval_interval=${EVAL_INTERVAL} \
        lr=1e-4 puzzle_emb_lr=1e-4 \
        weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        +run_name=${run_name} ema=True \
        +checkpoint_path=${CHECKPOINT_DIR} \
        checkpoint_every_eval=${CHECKPOINT_EVERY_EVAL}
else
    echo "Unknown variant: $VARIANT (use 'mlp' or 'att')"
    exit 1
fi

echo "Job finished at $(date)"
