#!/bin/bash
# =============================================================================
# run_test_pipeline_qwen_grid_ft_trm.sbatch
#
# Evaluates:
#   1) Qwen grid-FT -> parsed givens grid
#   2) TRM checkpoint inference on the parsed grids
#   3) Final bridge-style + TRM metrics
#
# Usage:
#   sbatch scripts/initial/translators/run_test_pipeline_qwen_grid_ft_trm.sbatch
#
# Optional env vars:
#   INPUT_JSON=data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json
#   TEST_PROFILE=corrected2
#   MAX_SAMPLES=1000
#   QWEN_CHECKPOINT=checkpoints/initial/qwen_sudoku_grid_ft/diverse_then_messy10k/best_model.pt
# =============================================================================

#SBATCH --job-name=test_qwen_grid_trm
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/initial/test_qwen_grid_trm_%j.out
#SBATCH --error=logs/initial/test_qwen_grid_trm_%j.err

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

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/qwen_sudoku_grid_finetune.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

GRID_EVAL_PY="$ROOT/code/initial/integration/qwen_sudoku_grid_finetune.py"
FINALIZE_PY="$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py"
TRM_PY="$ROOT/code/initial/run_trm_checkpoint_inference.py"

TEST_PROFILE="${TEST_PROFILE:-corrected2}"
case "$TEST_PROFILE" in
    diverse)
        INPUT_DEFAULT="$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json"
        STEM_DEFAULT="diverse_v3"
        ;;
    messy)
        INPUT_DEFAULT="$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_messy_test.json"
        STEM_DEFAULT="messy_test"
        ;;
    corrected2|ood)
        INPUT_DEFAULT="$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json"
        STEM_DEFAULT="corrected2"
        ;;
    *)
        echo "ERROR: unknown TEST_PROFILE='$TEST_PROFILE' (valid: diverse, messy, corrected2)"
        exit 1
        ;;
esac

INPUT_JSON="${INPUT_JSON:-$INPUT_DEFAULT}"
STEM="${STEM:-$STEM_DEFAULT}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/checkpoints/initial/Qwen3-1.7B}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-$ROOT/checkpoints/initial/qwen_sudoku_grid_ft/diverse_then_messy10k/best_model.pt}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
LLM_DTYPE="${LLM_DTYPE:-auto}"
SEED="${SEED:-42}"

TRM_CHECKPOINT_DIR="${TRM_CHECKPOINT_DIR:-$ROOT/checkpoints/initial/trm-sudoku}"
TRM_CHECKPOINT_FILE="${TRM_CHECKPOINT_FILE:-$ROOT/checkpoints/initial/trm-sudoku/step_65100}"
TRM_ROOT="${TRM_ROOT:-$ROOT/code/initial/TinyRecursiveModels}"
TRM_REQUIRE_TEST_PATH="${TRM_REQUIRE_TEST_PATH:-/data/initial/sudoku_grid/test/}"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/test_pipeline_qwen_grid_ft}"
BRIDGE_OUTPUT_JSON="${BRIDGE_OUTPUT_JSON:-$RESULTS_DIR/grid_predictions_${STEM}.json}"
GRID_METRICS_JSON="${GRID_METRICS_JSON:-$RESULTS_DIR/grid_metrics_${STEM}.json}"
TRM_INPUT_JSON="${TRM_INPUT_JSON:-$ROOT/data/initial/sudoku_grid/test/test_pipeline_qwen_grid_ft_input_${STEM}.json}"
TRM_PREDICTIONS_JSON="${TRM_PREDICTIONS_JSON:-$RESULTS_DIR/trm_predictions_on_qwen_grid_ft_${STEM}.json}"
METRICS_JSON="${METRICS_JSON:-$RESULTS_DIR/pipeline_metrics_${STEM}.json}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-translator-bridge}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
GRID_WANDB_RUN_NAME="${GRID_WANDB_RUN_NAME:-qwen-grid-ft-parse-${STEM}-${SLURM_JOB_ID:-local}}"
FINAL_WANDB_RUN_NAME="${FINAL_WANDB_RUN_NAME:-qwen-grid-ft-trm-${STEM}-${SLURM_JOB_ID:-local}}"

mkdir -p "$RESULTS_DIR" "$ROOT/logs/initial" "$(dirname "$TRM_INPUT_JSON")"

for path in "$GRID_EVAL_PY" "$FINALIZE_PY" "$TRM_PY" "$INPUT_JSON"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing required path: $path"
        exit 1
    fi
done
if [ ! -d "$QWEN_MODEL" ]; then
    echo "ERROR: base Qwen model dir not found: $QWEN_MODEL"
    exit 1
fi
if [ ! -f "$QWEN_CHECKPOINT" ]; then
    echo "ERROR: fine-tuned grid checkpoint not found: $QWEN_CHECKPOINT"
    exit 1
fi

echo "INPUT_JSON=$INPUT_JSON"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "QWEN_MODEL=$QWEN_MODEL"
echo "QWEN_CHECKPOINT=$QWEN_CHECKPOINT"
echo "TRM_CHECKPOINT_DIR=$TRM_CHECKPOINT_DIR"
echo "TRM_CHECKPOINT_FILE=${TRM_CHECKPOINT_FILE:-<auto-latest>}"
echo "BRIDGE_OUTPUT_JSON=$BRIDGE_OUTPUT_JSON"
echo "TRM_INPUT_JSON=$TRM_INPUT_JSON"
echo "TRM_PREDICTIONS_JSON=$TRM_PREDICTIONS_JSON"
echo "METRICS_JSON=$METRICS_JSON"

echo
echo "=== Stage 1/3: Qwen grid parser inference ==="
CMD_GRID=(
    python "$GRID_EVAL_PY" eval
    --base-model "$QWEN_MODEL"
    --checkpoint "$QWEN_CHECKPOINT"
    --input-json "$INPUT_JSON"
    --max-samples "$MAX_SAMPLES"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
    --llm-dtype "$LLM_DTYPE"
    --predictions-json "$BRIDGE_OUTPUT_JSON"
    --bridge-output-json "$BRIDGE_OUTPUT_JSON"
    --trm-input-json "$TRM_INPUT_JSON"
    --metrics-json "$GRID_METRICS_JSON"
)
if [ "$WANDB" = "1" ]; then
    CMD_GRID+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$GRID_WANDB_RUN_NAME")
    if [ -n "$WANDB_ENTITY" ]; then
        CMD_GRID+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi
printf '%q ' "${CMD_GRID[@]}"
echo
"${CMD_GRID[@]}"

echo
echo "=== Stage 2/3: TRM inference on parsed grids ==="
CMD_TRM=(
    conda run -n trm-env python "$TRM_PY"
    --checkpoint-dir "$TRM_CHECKPOINT_DIR"
    --trm-root "$TRM_ROOT"
    --test-json "$TRM_INPUT_JSON"
    --output-json "$TRM_PREDICTIONS_JSON"
    --max-puzzles "$MAX_SAMPLES"
)
if [ -n "$TRM_CHECKPOINT_FILE" ]; then
    CMD_TRM+=(--checkpoint-file "$TRM_CHECKPOINT_FILE")
fi
if [ -n "$TRM_REQUIRE_TEST_PATH" ]; then
    CMD_TRM+=(--require-test-path "$TRM_REQUIRE_TEST_PATH")
fi
printf '%q ' "${CMD_TRM[@]}"
echo
"${CMD_TRM[@]}"

echo
echo "=== Stage 3/3: Finalize metrics ==="
CMD_FINAL=(
    python "$FINALIZE_PY"
    --stage finalize
    --bridge-output-json "$BRIDGE_OUTPUT_JSON"
    --trm-predictions-json "$TRM_PREDICTIONS_JSON"
    --metrics-json "$METRICS_JSON"
)
if [ "$WANDB" = "1" ]; then
    CMD_FINAL+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$FINAL_WANDB_RUN_NAME")
fi
printf '%q ' "${CMD_FINAL[@]}"
echo
"${CMD_FINAL[@]}"

echo
echo "Done. Grid metrics: $GRID_METRICS_JSON"
echo "Done. Final pipeline metrics: $METRICS_JSON"
echo "Job finished at $(date)"
