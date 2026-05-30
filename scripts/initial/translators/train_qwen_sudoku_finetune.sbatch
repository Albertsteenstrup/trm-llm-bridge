#!/bin/bash
# =============================================================================
# train_qwen_sudoku_finetune.sbatch
#
# Fine-tunes Qwen3-1.7B to solve Sudoku directly from NL clue descriptions,
# using the same training distribution family as the bridge experiments.
#
# Usage:
#   sbatch scripts/initial/translators/train_qwen_sudoku_finetune.sbatch
#
# Optional env vars:
#   DATASET=data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse.json
#   DATASET=data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json
#   QWEN_MODEL=checkpoints/initial/Qwen3-1.7B
#   CHECKPOINT_DIR=checkpoints/initial/qwen_sudoku_ft
#   EPOCHS=20
#   BATCH_SIZE=2
#   EVAL_BATCH_SIZE=4
#   GRAD_ACCUM=8
#   LR=2e-5
#   MAX_LENGTH=768
#   MAX_NEW_TOKENS=128
#   SAVE_EVERY_EPOCH=10
#   PATIENCE=6
#   RESUME=1
#   RESUME_FROM=/path/to/checkpoint_epoch010.pt
#   WANDB=1
#   WANDB_PROJECT=trm-llm-qwen-solver-ft
#   WANDB_RUN_NAME=qwen-solver-ft-diverse
#   WANDB_ENTITY=
# =============================================================================

#SBATCH --job-name=qwen_sudoku_ft
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodelist=cn[7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/initial/qwen_sudoku_ft_%j.out
#SBATCH --error=logs/initial/qwen_sudoku_ft_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

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

if [ "${WANDB:-1}" = "1" ]; then
    if ! python - <<'PY'
import wandb
print("wandb OK")
PY
    then
        echo "wandb missing - installing in llm-env..."
        python -m pip install --upgrade pip setuptools wheel
        python -m pip install --no-cache-dir wandb
    fi
fi

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/qwen_sudoku_finetune.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
SCRIPT="$ROOT/code/initial/integration/qwen_sudoku_finetune.py"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

DATASET="${DATASET:-$ROOT/data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse.json}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/checkpoints/initial/Qwen3-1.7B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/initial/qwen_sudoku_ft}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/initial}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"
PATIENCE="${PATIENCE:-6}"
EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
SEED="${SEED:-42}"
LLM_DTYPE="${LLM_DTYPE:-auto}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-qwen-solver-ft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_ft_${SLURM_JOB_ID}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$ROOT/logs/initial"

CMD=(
    python "$SCRIPT" train
    --base-model "$QWEN_MODEL"
    --dataset "$DATASET"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --log-dir "$LOG_DIR"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --max-length "$MAX_LENGTH"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --eval-split "$EVAL_SPLIT"
    --seed "$SEED"
    --save-every-epoch "$SAVE_EVERY_EPOCH"
    --patience "$PATIENCE"
    --llm-dtype "$LLM_DTYPE"
    --grad-clip-norm "$GRAD_CLIP_NORM"
)

if [ -n "${RESUME_FROM:-}" ]; then
    CMD+=(--resume-from "$RESUME_FROM")
elif [ "${RESUME:-1}" = "0" ]; then
    CMD+=(--no-auto-resume)
fi
if [ "$WANDB" = "1" ]; then
    CMD+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
    if [ -n "$WANDB_ENTITY" ]; then
        CMD+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Job finished at $(date)"
