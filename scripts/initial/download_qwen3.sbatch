#!/bin/bash
# =============================================================================
# download_qwen3.sbatch — Download Qwen3-1.7B to HPC
#
# Model: Qwen/Qwen3-1.7B (Apache 2.0, no gating)
# Size: ~3-4 GB in BF16 safetensors
# Docs: https://huggingface.co/Qwen/Qwen3-1.7B
#
# Usage: sbatch scripts/initial/download_qwen3.sbatch
# =============================================================================
#SBATCH --job-name=download_qwen3
#SBATCH --partition=cores
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/initial/download_qwen3_%j.out

set -euo pipefail
echo "Download started on $(hostname) at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$ROOT/logs/initial"

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
if command -v module >/dev/null 2>&1; then
    module purge || true
    module use /opt/itu/easybuild/modules/all || true
    module load Anaconda3 || true
fi

if ! command -v conda >/dev/null 2>&1; then
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
    echo "ERROR: conda not found after module/direct fallback."
    module list || true
    exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate llm-env
set -u

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# ---------- Download ----------
MODEL_ID="Qwen/Qwen3-1.7B"
DEST_DIR="$ROOT/checkpoints/initial/Qwen3-1.7B"

echo "Downloading $MODEL_ID to $DEST_DIR"

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_ID}',
    local_dir='${DEST_DIR}',
    local_dir_use_symlinks=False,
)
print('Download complete.')
"

echo "Model saved to $DEST_DIR"
echo "Download finished at $(date)"
