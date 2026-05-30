#!/bin/bash
# =============================================================================
# train_qwen_sudoku_grid_finetune_diverse_then_messy.sbatch
#
# Fine-tunes Qwen3-1.7B as a Sudoku text-to-grid parser:
#   Phase 1: diverse 10K natural-language clues -> initial 81-char grid
#   Phase 2: continue from diverse weights on messy 10K clues
#
# The resulting messy checkpoint can be evaluated as:
#   Qwen grid-FT -> parsed givens grid -> TRM solver
#
# Usage:
#   sbatch scripts/initial/translators/train_qwen_sudoku_grid_finetune_diverse_then_messy.sbatch
#
# Optional env vars:
#   QWEN_MODEL=checkpoints/initial/Qwen3-1.7B
#   DIVERSE_DATASET=data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json
#   MESSY_DATASET=data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json
#   OUT_ROOT=checkpoints/initial/qwen_sudoku_grid_ft
#   DIVERSE_EPOCHS=5
#   MESSY_EPOCHS=2
#   BATCH_SIZE=2
#   EVAL_BATCH_SIZE=4
#   GRAD_ACCUM=8
#   LR=2e-5
#   MAX_LENGTH=768
#   MAX_NEW_TOKENS=96
#   WANDB=1
# =============================================================================

#SBATCH --job-name=qwen_grid_ft_dm
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/initial/qwen_grid_ft_dm_%j.out
#SBATCH --error=logs/initial/qwen_grid_ft_dm_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

unset -f module 2>/dev/null || true
if ! command -v module >/dev/null 2>&1; then
    set +u
    [ -f /etc/profile ] && source /etc/profile || true
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
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/qwen_sudoku_grid_finetune.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
SCRIPT="$ROOT/code/initial/integration/qwen_sudoku_grid_finetune.py"

QWEN_MODEL="${QWEN_MODEL:-$ROOT/checkpoints/initial/Qwen3-1.7B}"
DIVERSE_DATASET="${DIVERSE_DATASET:-$ROOT/data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json}"
MESSY_DATASET="${MESSY_DATASET:-$ROOT/data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/checkpoints/initial/qwen_sudoku_grid_ft}"
DIVERSE_DIR="${DIVERSE_DIR:-$OUT_ROOT/diverse10k}"
MESSY_DIR="${MESSY_DIR:-$OUT_ROOT/diverse_then_messy10k}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/initial}"

DIVERSE_EPOCHS="${DIVERSE_EPOCHS:-5}"
MESSY_EPOCHS="${MESSY_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
DIVERSE_PATIENCE="${DIVERSE_PATIENCE:-3}"
MESSY_PATIENCE="${MESSY_PATIENCE:-2}"
EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
SEED="${SEED:-42}"
LLM_DTYPE="${LLM_DTYPE:-auto}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-translator-bridge}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$DIVERSE_DIR" "$MESSY_DIR" "$LOG_DIR" "$ROOT/logs/initial"

for path in "$SCRIPT" "$DIVERSE_DATASET" "$MESSY_DATASET"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing required path: $path"
        exit 1
    fi
done
if [ ! -d "$QWEN_MODEL" ]; then
    echo "ERROR: base Qwen model dir not found: $QWEN_MODEL"
    exit 1
fi

run_train() {
    local dataset="$1"
    local checkpoint_dir="$2"
    local epochs="$3"
    local patience="$4"
    local run_name="$5"
    shift 5

    local cmd=(
        python "$SCRIPT" train
        --base-model "$QWEN_MODEL"
        --dataset "$dataset"
        --checkpoint-dir "$checkpoint_dir"
        --log-dir "$LOG_DIR"
        --epochs "$epochs"
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
        --patience "$patience"
        --llm-dtype "$LLM_DTYPE"
        --grad-clip-norm "$GRAD_CLIP_NORM"
        "$@"
    )

    if [ "$WANDB" = "1" ]; then
        cmd+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name")
        if [ -n "$WANDB_ENTITY" ]; then
            cmd+=(--wandb-entity "$WANDB_ENTITY")
        fi
    fi

    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
}

echo
echo "=== Phase 1/2: Diverse grid parser fine-tuning ==="
run_train "$DIVERSE_DATASET" "$DIVERSE_DIR" "$DIVERSE_EPOCHS" "$DIVERSE_PATIENCE" "qwen-grid-diverse10k-${SLURM_JOB_ID:-local}"

DIVERSE_BEST="$DIVERSE_DIR/best_model.pt"
if [ ! -f "$DIVERSE_BEST" ]; then
    echo "ERROR: diverse phase did not write best checkpoint: $DIVERSE_BEST"
    exit 1
fi

echo
echo "=== Phase 2/2: Continue grid parser fine-tuning on messy ==="
run_train "$MESSY_DATASET" "$MESSY_DIR" "$MESSY_EPOCHS" "$MESSY_PATIENCE" "qwen-grid-diverse-then-messy10k-${SLURM_JOB_ID:-local}" \
    --resume-from "$DIVERSE_BEST" \
    --weights-only-resume \
    --no-auto-resume

echo
echo "Diverse best checkpoint: $DIVERSE_BEST"
echo "Messy best checkpoint: $MESSY_DIR/best_model.pt"
echo "Job finished at $(date)"
