#!/bin/bash
# =============================================================================
# evaluate_qwen_sudoku_finetune.sbatch
#
# Evaluates frozen or fine-tuned Qwen3-1.7B on direct Sudoku solving from NL.
#
# Usage:
#   sbatch scripts/initial/translators/evaluate_qwen_sudoku_finetune.sbatch
#
# Optional env vars:
#   INPUT_JSON=data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json
#   QWEN_MODEL=checkpoints/initial/Qwen3-1.7B
#   QWEN_CHECKPOINT=checkpoints/initial/qwen_sudoku_ft/best_model.pt   # leave empty for frozen Qwen
#   MAX_SAMPLES=1000
#   EVAL_BATCH_SIZE=4
#   PREDICTIONS_JSON=results/qwen_sudoku_ft/solver_predictions_diverse_v3.json
#   METRICS_JSON=results/qwen_sudoku_ft/solver_metrics_diverse_v3.json
#   BASELINE_METRICS_JSON=results/test_pipeline_finetuned/pipeline_metrics_corrected2.json
#   WANDB=1
#   WANDB_PROJECT=trm-llm-qwen-ft-eval
#   WANDB_RUN_NAME=qwen-solver-diverse-v3
# =============================================================================

#SBATCH --job-name=eval_qwen_ft
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodelist=cn[7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/initial/eval_qwen_ft_%j.out
#SBATCH --error=logs/initial/eval_qwen_ft_%j.err

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

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/qwen_sudoku_finetune.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
SCRIPT="$ROOT/code/initial/integration/qwen_sudoku_finetune.py"

INPUT_JSON="${INPUT_JSON:-$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/checkpoints/initial/Qwen3-1.7B}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LLM_DTYPE="${LLM_DTYPE:-auto}"
SEED="${SEED:-42}"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/qwen_sudoku_ft}"
PREDICTIONS_JSON="${PREDICTIONS_JSON:-$RESULTS_DIR/solver_predictions_diverse_v3.json}"
METRICS_JSON="${METRICS_JSON:-$RESULTS_DIR/solver_metrics_diverse_v3.json}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-qwen-ft-eval}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_solver_eval_${SLURM_JOB_ID}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$RESULTS_DIR" "$ROOT/logs/initial"

CMD=(
    python "$SCRIPT" eval
    --base-model "$QWEN_MODEL"
    --input-json "$INPUT_JSON"
    --predictions-json "$PREDICTIONS_JSON"
    --metrics-json "$METRICS_JSON"
    --max-samples "$MAX_SAMPLES"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
    --llm-dtype "$LLM_DTYPE"
)

if [ -n "$QWEN_CHECKPOINT" ]; then
    CMD+=(--checkpoint "$QWEN_CHECKPOINT")
fi
if [ -n "$BASELINE_METRICS_JSON" ]; then
    CMD+=(--baseline-metrics-json "$BASELINE_METRICS_JSON")
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
