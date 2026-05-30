#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-together-sudoku}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"
INPUT_JSON="${INPUT_JSON:-$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/results/together_sudoku_code_interpreter/together_qwen3_1_7b_predictions_25.json}"
METRICS_JSON="${METRICS_JSON:-$ROOT/results/together_sudoku_code_interpreter/together_qwen3_1_7b_metrics_25.json}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-$ROOT/results/test_pipeline_finetuned/pipeline_metrics_corrected2.json}"

MODELS_STRING="${MODELS:-Qwen/Qwen3-1.7B}"
read -r -a MODELS <<< "$MODELS_STRING"

MAX_SAMPLES="${MAX_SAMPLES:-25}"
MAX_TURNS="${MAX_TURNS:-4}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
TEMPERATURE="${TEMPERATURE:-0}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"

if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ROOT/.env"
    set +a
fi

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY is not set and was not found in $ROOT/.env"
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
    "$PYTHON_BIN" "$ROOT/code/initial/integration/benchmark_together_code_interpreter_sudoku.py"
    --input-json "$INPUT_JSON"
    --output-json "$OUTPUT_JSON"
    --metrics-json "$METRICS_JSON"
    --baseline-metrics-json "$BASELINE_METRICS_JSON"
    --models "${MODELS[@]}"
    --max-samples "$MAX_SAMPLES"
    --max-turns "$MAX_TURNS"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --save-every "$SAVE_EVERY"
    --sleep-seconds "$SLEEP_SECONDS"
    --resume
    --retry-errors
)

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
