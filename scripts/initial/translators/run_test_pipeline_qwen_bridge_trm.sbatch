#!/bin/bash
# =============================================================================
# run_test_pipeline_qwen_bridge_trm.sbatch
#
# Runs inference-only pipeline:
#   1) Qwen3-1.7B + translator bridge checkpoint -> translated Sudoku grids
#   2) TRM checkpoint inference on translated grids
#   3) Final metrics aggregation (bridge + TRM)
#
# Usage:
#   sbatch scripts/initial/translators/run_test_pipeline_qwen_bridge_trm.sbatch
#
# Optional env vars:
#   INPUT_JSON=data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json
#   NL_JSON=   # optional override/mapping file; usually leave empty when INPUT_JSON already has nl_description
#   MAX_SAMPLES=1000
#   QWEN_MODEL=checkpoints/initial/Qwen3-1.7B
#   BRIDGE_CHECKPOINT=checkpoints/initial/translator_bridge/10K/best_qformer.pt
#   BRIDGE_ARCH=          # optional if checkpoint does not store architecture metadata
#   TRM_CHECKPOINT_DIR=checkpoints/initial/trm-sudoku
#   TRM_CHECKPOINT_FILE=checkpoints/initial/trm-sudoku/step_65100
#   WANDB=1
#   WANDB_PROJECT=trm-llm-test-pipeline
# =============================================================================

#SBATCH --job-name=test_pipe_qwen_bridge_trm
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodelist=cn[7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/initial/test_pipe_qwen_bridge_trm_%j.out
#SBATCH --error=logs/initial/test_pipe_qwen_bridge_trm_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

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

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Under Slurm, BASH_SOURCE may point to a temporary spool script path.
# Prefer submit directory, then fall back to repo-relative path from script.
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

if [ ! -f "$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py" ]; then
    echo "ERROR: Could not resolve repository root."
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  SCRIPT_DIR=$SCRIPT_DIR"
    echo "  ROOT=$ROOT"
    exit 1
fi

cd "$ROOT"
PIPELINE_PY="$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py"

INPUT_JSON="${INPUT_JSON:-$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json}"
NL_JSON="${NL_JSON:-}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
REQUIRE_NL_DESCRIPTION="${REQUIRE_NL_DESCRIPTION:-1}"

QWEN_MODEL="${QWEN_MODEL:-$ROOT/checkpoints/initial/Qwen3-1.7B}"
BRIDGE_CHECKPOINT="${BRIDGE_CHECKPOINT:-$ROOT/checkpoints/initial/translator_bridge/10K/best_qformer.pt}"
BRIDGE_ARCH="${BRIDGE_ARCH:-}"
TRM_CHECKPOINT_DIR="${TRM_CHECKPOINT_DIR:-$ROOT/checkpoints/initial/trm-sudoku}"
TRM_CHECKPOINT_FILE="${TRM_CHECKPOINT_FILE:-$ROOT/checkpoints/initial/trm-sudoku/step_65100}"
TRM_REQUIRE_TEST_PATH="${TRM_REQUIRE_TEST_PATH:-/data/initial/sudoku_grid/test/}"
TRM_ROOT="${TRM_ROOT:-$ROOT/code/initial/TinyRecursiveModels}"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/test_pipeline}"
mkdir -p "$RESULTS_DIR"

BRIDGE_OUTPUT_JSON="${BRIDGE_OUTPUT_JSON:-$RESULTS_DIR/bridge_predictions_1000_v3.json}"
TRM_INPUT_JSON="${TRM_INPUT_JSON:-$ROOT/data/initial/sudoku_grid/test/test_pipeline_bridge_input_1000_v3.json}"
TRM_PREDICTIONS_JSON="${TRM_PREDICTIONS_JSON:-$RESULTS_DIR/trm_predictions_on_bridge_1000_v3.json}"
METRICS_JSON="${METRICS_JSON:-$RESULTS_DIR/pipeline_metrics_1000_v3.json}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-test-pipeline}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-test_pipe_${SLURM_JOB_ID}}"

echo "INPUT_JSON=$INPUT_JSON"
echo "NL_JSON=${NL_JSON:-<none>}"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "REQUIRE_NL_DESCRIPTION=$REQUIRE_NL_DESCRIPTION"
echo "QWEN_MODEL=$QWEN_MODEL"
echo "BRIDGE_CHECKPOINT=$BRIDGE_CHECKPOINT"
echo "BRIDGE_ARCH=${BRIDGE_ARCH:-<auto>}"
echo "TRM_CHECKPOINT_DIR=$TRM_CHECKPOINT_DIR"
echo "TRM_CHECKPOINT_FILE=${TRM_CHECKPOINT_FILE:-<auto-latest>}"
echo "TRM_REQUIRE_TEST_PATH=${TRM_REQUIRE_TEST_PATH:-<disabled>}"
echo "TRM_ROOT=$TRM_ROOT"
echo "BRIDGE_OUTPUT_JSON=$BRIDGE_OUTPUT_JSON"
echo "TRM_INPUT_JSON=$TRM_INPUT_JSON"
echo "TRM_PREDICTIONS_JSON=$TRM_PREDICTIONS_JSON"
echo "METRICS_JSON=$METRICS_JSON"

if [ ! -f "$PIPELINE_PY" ]; then
    echo "ERROR: pipeline script not found: $PIPELINE_PY"
    exit 1
fi
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: input JSON not found: $INPUT_JSON"
    exit 1
fi
if [ ! -d "$QWEN_MODEL" ]; then
    echo "ERROR: Qwen model dir not found: $QWEN_MODEL"
    exit 1
fi
if [ ! -f "$BRIDGE_CHECKPOINT" ]; then
    echo "ERROR: bridge checkpoint not found: $BRIDGE_CHECKPOINT"
    exit 1
fi
if [ ! -d "$TRM_ROOT" ]; then
    echo "ERROR: TRM root not found: $TRM_ROOT"
    exit 1
fi

WAND_ARGS=()
if [ "$WANDB" = "1" ]; then
    WAND_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
fi

echo "\n=== Stage 1/3: Qwen + bridge inference ==="
CMD_BRIDGE=(
    python "$PIPELINE_PY"
    --stage bridge
    --input-json "$INPUT_JSON"
    --qwen-model "$QWEN_MODEL"
    --bridge-checkpoint "$BRIDGE_CHECKPOINT"
    --max-samples "$MAX_SAMPLES"
    --bridge-output-json "$BRIDGE_OUTPUT_JSON"
    --trm-input-json "$TRM_INPUT_JSON"
    --metrics-json "$METRICS_JSON"
)
if [ "$REQUIRE_NL_DESCRIPTION" = "1" ]; then
    CMD_BRIDGE+=(--require-nl-description)
fi
if [ -n "$BRIDGE_ARCH" ]; then
    CMD_BRIDGE+=(--bridge-arch "$BRIDGE_ARCH")
fi
if [ -n "$NL_JSON" ]; then
    CMD_BRIDGE+=(--nl-json "$NL_JSON")
fi
"${CMD_BRIDGE[@]}"

echo "\n=== Stage 2/3: TRM inference on translated grids ==="
TRM_CMD=(
    conda run -n trm-env python "$ROOT/code/initial/run_trm_checkpoint_inference.py"
    --checkpoint-dir "$TRM_CHECKPOINT_DIR"
    --trm-root "$TRM_ROOT"
    --test-json "$TRM_INPUT_JSON"
    --output-json "$TRM_PREDICTIONS_JSON"
    --max-puzzles "$MAX_SAMPLES"
)
if [ -n "$TRM_CHECKPOINT_FILE" ]; then
    TRM_CMD+=(--checkpoint-file "$TRM_CHECKPOINT_FILE")
fi
if [ -n "$TRM_REQUIRE_TEST_PATH" ]; then
    TRM_CMD+=(--require-test-path "$TRM_REQUIRE_TEST_PATH")
fi
"${TRM_CMD[@]}"

echo "\n=== Stage 3/3: Finalize metrics ==="
CMD_FINAL=(
    python "$PIPELINE_PY"
    --stage finalize
    --bridge-output-json "$BRIDGE_OUTPUT_JSON"
    --trm-predictions-json "$TRM_PREDICTIONS_JSON"
    --metrics-json "$METRICS_JSON"
)
CMD_FINAL+=("${WAND_ARGS[@]}")
"${CMD_FINAL[@]}"

echo "\nDone. Final metrics: $METRICS_JSON"
echo "Job finished at $(date)"
