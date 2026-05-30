#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ucloud/bin/python}"
RUN_NAME="${RUN_NAME:-clrs30_hard_nl_native_v1}"
INPUT_JSONL="${INPUT_JSONL:-$ROOT/data/clrs_native_text/prepared/${RUN_NAME}_bridge/test_ood_size.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/results/clrs_native_text/openai_hard_nl_text_to_native_${RUN_NAME}_gpt54.json}"
METRICS_JSON="${METRICS_JSON:-$ROOT/results/clrs_native_text/openai_hard_nl_text_to_native_${RUN_NAME}_gpt54.metrics.json}"
MODEL="${MODEL:-gpt-5.4}"
MAX_SAMPLES="${MAX_SAMPLES:-25}"

"$PYTHON_BIN" "$ROOT/code/clrs_native_text/benchmark_openai_text_to_native.py" \
  --input-jsonl "$INPUT_JSONL" \
  --output-json "$OUTPUT_JSON" \
  --metrics-json "$METRICS_JSON" \
  --model "$MODEL" \
  --task-filter "${TASK_FILTER:-all}" \
  --max-samples "$MAX_SAMPLES" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-24000}" \
  --max-retries "${MAX_RETRIES:-1}" \
  --memory-limit "${MEMORY_LIMIT:-1g}" \
  --tool-choice "${TOOL_CHOICE:-required}" \
  --numeric-atol "${NUMERIC_ATOL:-0.001}" \
  ${HIDE_ALGORITHM:+--hide-algorithm} \
  ${HIDE_NATIVE_SCHEMA:+--hide-native-schema}
