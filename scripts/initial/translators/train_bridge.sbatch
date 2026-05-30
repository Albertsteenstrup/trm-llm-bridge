#!/bin/bash
# =============================================================================
# train_bridge.sbatch — Train Q-Former bridge translator (Path 1)
#
# Trains a lightweight cross-attention module that reads frozen LLM hidden
# states and classifies each Sudoku cell as empty or digit 1-9.
#
# Usage:
#   sbatch scripts/initial/translators/train_bridge.sbatch                               # default: qformer + default ckpt dir
#   sbatch scripts/initial/translators/train_bridge.sbatch linear                        # linear probe baseline
#   sbatch scripts/initial/translators/train_bridge.sbatch mlp                           # LLaVA-style MLP
#   sbatch scripts/initial/translators/train_bridge.sbatch qformer                       # Q-Former baseline
#   sbatch scripts/initial/translators/train_bridge.sbatch transnar_trm                  # TransNAR-style gated interleave scaffold
#   sbatch scripts/initial/translators/train_bridge.sbatch qformer checkpoints/initial/translator_bridge/1K-500epochs
#   sbatch scripts/initial/translators/train_bridge.sbatch qformer checkpoints/initial/translator_bridge/1K-500epochs 100
#
# Optional env vars:
#   DATASET=data/initial/sudoku_synthetic/rule/sudoku_nl_diverse.json
#   EPOCHS=15
#   BATCH_SIZE=16
#   LR=3e-4
#   PATIENCE=8
#   LLM_MODEL=Qwen/Qwen3-1.7B   # or local path to a downloaded model
#   LLM_LAYER=-1
#   TRANSNAR_RECURRENCE_STEPS=1
#   CHECKPOINT_DIR=checkpoints/initial/translator_bridge/1K-500epochs
#   EVAL_ONLY=/path/to/checkpoint.pt
#   RESUME=1              # 1=auto-resume from last_<arch>.pt (default), 0=start fresh
#   RESUME_FROM=/path/to/checkpoint.pt  # explicit checkpoint to resume from
#   WANDB=0               # set to 1 to enable Weights & Biases
#   WANDB_PROJECT=trm-llm-translator-bridge
#   WANDB_RUN_NAME=
#   WANDB_ENTITY=
# =============================================================================
#SBATCH --job-name=bridge_translator
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/initial/bridge_translator_%j.out
#SBATCH --error=logs/initial/bridge_translator_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ---------- Variant selection ----------
ARCH="${1:-qformer}"
CHECKPOINT_DIR_ARG="${2:-}"
PATIENCE_ARG="${3:-}"
DEFAULT_CHECKPOINT_DIR="checkpoints/initial/translator_bridge/1K-500epochs"
CHECKPOINT_DIR_EFFECTIVE="${CHECKPOINT_DIR_ARG:-${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}}"
PATIENCE_EFFECTIVE="${PATIENCE_ARG:-${PATIENCE:-}}"
echo "Architecture: $ARCH"
echo "Checkpoint dir: $CHECKPOINT_DIR_EFFECTIVE"
if [ -n "$PATIENCE_EFFECTIVE" ]; then
    echo "Patience: $PATIENCE_EFFECTIVE"
fi

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
    exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate llm-env
set -u

echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
python --version

if [ "${WANDB:-0}" = "1" ]; then
    echo "WANDB enabled. Verifying wandb package..."
    if ! python - <<'PY'
import wandb
print("wandb OK")
PY
    then
        echo "wandb missing in env '${CONDA_DEFAULT_ENV:-unknown}' - installing via python -m pip..."
        python -m pip install --upgrade pip setuptools wheel
        python -m pip install --no-cache-dir wandb
        if ! python - <<'PY'
import wandb
print("wandb OK after install")
PY
        then
            echo "ERROR: WANDB=1 but wandb could not be imported after install in env '${CONDA_DEFAULT_ENV:-unknown}'."
            echo "Try manually on login node: conda activate llm-env && python -m pip install wandb"
            exit 1
        fi
    fi
fi

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/train_translator_bridge.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
SCRIPT="$ROOT/code/initial/integration/train_translator_bridge.py"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  SCRIPT_DIR=$SCRIPT_DIR"
    echo "  ROOT=$ROOT"
    exit 1
fi

# ---------- Build command ----------
CMD="python $SCRIPT --arch $ARCH"

if [ -n "${DATASET:-}" ]; then
    CMD="$CMD --dataset $DATASET"
fi
if [ -n "${EPOCHS:-}" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi
if [ -n "${BATCH_SIZE:-}" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi
if [ -n "${LR:-}" ]; then
    CMD="$CMD --lr $LR"
fi
if [ -n "$PATIENCE_EFFECTIVE" ]; then
    CMD="$CMD --patience $PATIENCE_EFFECTIVE"
fi
if [ -n "${LLM_MODEL:-}" ]; then
    CMD="$CMD --llm-model $LLM_MODEL"
fi
if [ -n "${LLM_LAYER:-}" ]; then
    CMD="$CMD --llm-layer $LLM_LAYER"
fi
if [ -n "${TRANSNAR_RECURRENCE_STEPS:-}" ]; then
    CMD="$CMD --transnar-recurrence-steps $TRANSNAR_RECURRENCE_STEPS"
fi
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR_EFFECTIVE"
if [ -n "${EVAL_ONLY:-}" ]; then
    CMD="$CMD --eval-only $EVAL_ONLY"
fi
if [ -n "${RESUME_FROM:-}" ]; then
    CMD="$CMD --resume-from $RESUME_FROM"
elif [ "${RESUME:-1}" = "0" ]; then
    CMD="$CMD --no-auto-resume"
fi
if [ "${WANDB:-0}" = "1" ]; then
    CMD="$CMD --wandb"
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
    CMD="$CMD --wandb-project $WANDB_PROJECT"
fi
if [ -n "${WANDB_RUN_NAME:-}" ]; then
    CMD="$CMD --wandb-run-name $WANDB_RUN_NAME"
fi
if [ -n "${WANDB_ENTITY:-}" ]; then
    CMD="$CMD --wandb-entity $WANDB_ENTITY"
fi

echo "Running: $CMD"
eval "$CMD"

echo "Job finished at $(date)"
