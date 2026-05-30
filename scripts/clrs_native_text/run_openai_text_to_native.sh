#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ucloud/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-$ROOT/data/clrs_native_text/raw/paired_text_to_native_sorting_order_len64_25.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/results/clrs_native_text/openai_text_to_native_sorting_order_len64_25_gpt54.json}"
METRICS_JSON="${METRICS_JSON:-$ROOT/results/clrs_native_text/openai_text_to_native_sorting_order_len64_25_gpt54.metrics.json}"
MODEL="${MODEL:-gpt-5.4}"
TASK_FILTER="${TASK_FILTER:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-25}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-20000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MEMORY_LIMIT="${MEMORY_LIMIT:-1g}"
TOOL_CHOICE="${TOOL_CHOICE:-required}"
NUMERIC_ATOL="${NUMERIC_ATOL:-0.001}"

CMD=(
    "$PYTHON_BIN" "$ROOT/code/clrs_native_text/benchmark_openai_text_to_native.py"
    --input-jsonl "$INPUT_JSONL"
    --output-json "$OUTPUT_JSON"
    --metrics-json "$METRICS_JSON"
    --model "$MODEL"
    --task-filter "$TASK_FILTER"
    --max-samples "$MAX_SAMPLES"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --max-retries "$MAX_RETRIES"
    --memory-limit "$MEMORY_LIMIT"
    --tool-choice "$TOOL_CHOICE"
    --numeric-atol "$NUMERIC_ATOL"
)

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
