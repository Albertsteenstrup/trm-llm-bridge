#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-sudoku-frontier}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"
INPUT_JSON="${INPUT_JSON:-$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/results/frontier_sudoku_builder/frontier_predictions_corrected2.json}"
METRICS_JSON="${METRICS_JSON:-$ROOT/results/frontier_sudoku_builder/frontier_metrics_corrected2.json}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-$ROOT/results/test_pipeline_finetuned/pipeline_metrics_corrected2.json}"

MODELS_STRING="${MODELS:-deepseek-ai/deepseek-v4-pro z-ai/glm-5.1}"
read -r -a MODELS <<< "$MODELS_STRING"

MAX_SAMPLES="${MAX_SAMPLES:-25}"
CONCURRENCY="${CONCURRENCY:-1}"
CALLS_PER_MINUTE="${CALLS_PER_MINUTE:-15}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
MAX_RETRIES="${MAX_RETRIES:-1}"
SAVE_EVERY="${SAVE_EVERY:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
ALLOW_DIGIT_FALLBACK="${ALLOW_DIGIT_FALLBACK:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
REASONING_EFFORT_MODEL_REGEX="${REASONING_EFFORT_MODEL_REGEX:-deepseek-ai/deepseek-v4}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-trm-llm-test-pipeline}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-frontier-sudoku-deepseek-v4-pro-glm-5-1-thinking25}"

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "ERROR: NVIDIA_API_KEY is not set."
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

mkdir -p "$(dirname "$OUTPUT_JSON")"

CMD=(
    "$PYTHON_BIN" "$ROOT/code/initial/integration/benchmark_nvidia_frontier_sudoku.py"
    --input-json "$INPUT_JSON"
    --output-json "$OUTPUT_JSON"
    --metrics-json "$METRICS_JSON"
    --baseline-metrics-json "$BASELINE_METRICS_JSON"
    --models "${MODELS[@]}"
    --max-samples "$MAX_SAMPLES"
    --concurrency "$CONCURRENCY"
    --calls-per-minute "$CALLS_PER_MINUTE"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --max-retries "$MAX_RETRIES"
    --save-every "$SAVE_EVERY"
    --temperature "$TEMPERATURE"
    --max-tokens "$MAX_TOKENS"
    --resume
    --retry-errors
)

if [ "$ALLOW_DIGIT_FALLBACK" = "1" ]; then
    CMD+=(--allow-digit-fallback)
fi

if [ -n "$REASONING_EFFORT" ]; then
    CMD+=(--reasoning-effort "$REASONING_EFFORT" --reasoning-effort-model-regex "$REASONING_EFFORT_MODEL_REGEX")
fi

if [ -n "$EXTRA_BODY_JSON" ]; then
    CMD+=(--extra-body-json "$EXTRA_BODY_JSON")
fi

if [ "$WANDB" = "1" ]; then
    CMD+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
