#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-openai-sudoku}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"
INPUT_JSON="${INPUT_JSON:-$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/results/openai_sudoku_code_interpreter/openai_predictions_corrected2.json}"
METRICS_JSON="${METRICS_JSON:-$ROOT/results/openai_sudoku_code_interpreter/openai_metrics_corrected2.json}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-$ROOT/results/test_pipeline_finetuned/pipeline_metrics_corrected2.json}"

MODELS_STRING="${MODELS:-gpt-5.4}"
read -r -a MODELS <<< "$MODELS_STRING"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
CALLS_PER_MINUTE="${CALLS_PER_MINUTE:-20}"
MAX_RETRIES="${MAX_RETRIES:-1}"
SAVE_EVERY="${SAVE_EVERY:-1}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-400}"
MEMORY_LIMIT="${MEMORY_LIMIT:-1g}"
TOOL_CHOICE="${TOOL_CHOICE:-required}"
CONTAINER_MODE="${CONTAINER_MODE:-shared}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-test-pipeline}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-openai-sudoku-code-interpreter}"

if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ROOT/.env"
    set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set and was not found in $ROOT/.env"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    if [ "$USE_SYSTEM_SITE_PACKAGES" = "1" ]; then
        "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$INSTALL_DEPS" = "1" ]; then
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "$ROOT/code/initial/requirements.txt"
fi

mkdir -p "$(dirname "$OUTPUT_JSON")" "$(dirname "$METRICS_JSON")"

CMD=(
    "$PYTHON_BIN" "$ROOT/code/initial/integration/benchmark_openai_code_interpreter_sudoku.py"
    --input-json "$INPUT_JSON"
    --output-json "$OUTPUT_JSON"
    --metrics-json "$METRICS_JSON"
    --baseline-metrics-json "$BASELINE_METRICS_JSON"
    --models "${MODELS[@]}"
    --max-samples "$MAX_SAMPLES"
    --concurrency "$CONCURRENCY"
    --calls-per-minute "$CALLS_PER_MINUTE"
    --max-retries "$MAX_RETRIES"
    --save-every "$SAVE_EVERY"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --memory-limit "$MEMORY_LIMIT"
    --tool-choice "$TOOL_CHOICE"
    --container-mode "$CONTAINER_MODE"
    --resume
    --retry-errors
)

if [ "$WANDB" = "1" ]; then
    CMD+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
