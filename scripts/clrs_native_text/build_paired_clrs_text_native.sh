#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ucloud/bin/python}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$ROOT/data/clrs_native_text/raw/paired_clrs_text_native_clrs30_len64.jsonl}"
METADATA_JSON="${METADATA_JSON:-$ROOT/data/clrs_native_text/raw/paired_clrs_text_native_clrs30_len64.metadata.json}"
ALGORITHMS="${ALGORITHMS:-all}"
LENGTHS="${LENGTHS:-64}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-25}"
SEED="${SEED:-7}"
SPLIT="${SPLIT:-test}"
NUM_DECIMALS_IN_FLOAT="${NUM_DECIMALS_IN_FLOAT:-3}"
USE_HINTS="${USE_HINTS:-0}"

CMD=(
    "$PYTHON_BIN" "$ROOT/code/clrs_native_text/build_paired_clrs_text_native.py"
    --output-jsonl "$OUTPUT_JSONL"
    --metadata-json "$METADATA_JSON"
    --algorithms "$ALGORITHMS"
    --lengths "$LENGTHS"
    --samples-per-task "$SAMPLES_PER_TASK"
    --seed "$SEED"
    --split "$SPLIT"
    --num-decimals-in-float "$NUM_DECIMALS_IN_FLOAT"
)

if [ "$USE_HINTS" = "1" ]; then
    CMD+=(--use-hints)
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
