#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment setup for thesis-trm-llm on ITU HPC
# Run inside an interactive srun session, NOT on login node:
#   srun --partition=acltr --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash
#   bash scripts/initial/setup_env.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Temporary directory for pip (required on HPC, see usingpip.md)
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# Load Anaconda
module purge
module use /opt/itu/easybuild/modules/all
module load Anaconda3

# Conda activate hooks may reference unset vars on HPC (e.g. QT_XCB_GL_INTEGRATION)
# so relax nounset only for conda hook/activate calls.
set +u
eval "$(conda shell.bash hook)"
set -u

echo "============================================"
echo "Setting up TRM environment (trm-env)"
echo "============================================"

# Create TRM conda env
if ! conda info --envs | grep -q "trm-env"; then
    set +u
    conda create -n trm-env python=3.10 -y
    set -u
fi
set +u
conda activate trm-env
set -u

# Ensure torch shared libraries (e.g., libc10.so) are discoverable at runtime
TORCH_LIB_DIR="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Install PyTorch nightly (CUDA 12.6) — per TRM README
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Ensure CUDA toolkit paths are available for building CUDA extensions (adam-atan2)
if ! command -v nvcc >/dev/null 2>&1; then
    module load CUDA >/dev/null 2>&1 || true
fi
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME="/usr/local/cuda"
else
    echo "ERROR: CUDA toolkit not found (nvcc missing)."
    echo "Run this script inside an srun allocation with CUDA available."
    exit 1
fi
export CUDACXX="${CUDA_HOME}/bin/nvcc"
echo "Using CUDA_HOME=${CUDA_HOME}"

# Use conda GCC toolchain for torch CUDA extension builds.
# This avoids cluster-system GCC/CUDA compatibility issues (e.g., CUDA 12.9 requires GCC < 14).
set +u
conda install -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13 || true
set -u
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ] && [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX"
fi
echo "Using CC=${CC:-unset}"
echo "Using CXX=${CXX:-unset}"

# Install TRM requirements
pip install -r "$PROJECT_DIR/code/initial/TinyRecursiveModels/requirements.txt"

# Ensure adam-atan2 backend is present (requirements may install package without compiled backend)
if ! python - <<'PY'
import adam_atan2_backend
print("adam_atan2_backend OK")
PY
then
    echo "adam_atan2_backend missing - forcing rebuild of adam-atan2..."
    pip install --upgrade pip wheel setuptools
    pip install --no-cache-dir --force-reinstall --no-build-isolation adam-atan2

    python - <<'PY'
import adam_atan2_backend
print("adam_atan2_backend OK after rebuild")
PY
fi

echo "TRM environment (trm-env) setup complete."

echo ""
echo "============================================"
echo "Setting up LLM environment (llm-env)"
echo "============================================"

# Create LLM conda env for Qwen3-1.7B and NVIDIA API work
if ! conda info --envs | grep -q "llm-env"; then
    set +u
    conda create -n llm-env python=3.10 -y
    set -u
fi
set +u
conda activate llm-env
set -u

pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

TORCH_LIB_DIR="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

pip install transformers>=4.53.0 accelerate huggingface_hub
pip install openai  # For NVIDIA API (OpenAI-compatible endpoint)
pip install datasets pandas tqdm

echo "LLM environment (llm-env) setup complete."

echo ""
echo "============================================"
echo "Setting up wandb (optional)"
echo "============================================"
echo "Run 'wandb login YOUR-KEY' in each env if you want W&B logging."
echo ""
echo "All environments ready. Activate with:"
echo "  conda activate trm-env   # For TRM training"
echo "  conda activate llm-env   # For Qwen3-1.7B / NVIDIA API"
