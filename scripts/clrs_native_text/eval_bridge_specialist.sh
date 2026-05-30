#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_hard_nl_native_v1}"
BRIDGE_DIR="${BRIDGE_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_bridge}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/clrs_native_text}"

BRIDGE_CHECKPOINT="${BRIDGE_CHECKPOINT:?set BRIDGE_CHECKPOINT to checkpoints/.../best.pt or last.pt}"
TRM_CHECKPOINT="${TRM_CHECKPOINT:?set TRM_CHECKPOINT to checkpoints/...pt from train_b200_specialist.sh}"
INPUT_JSONL="${INPUT_JSONL:-$BRIDGE_DIR/test_ood_size.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$RESULTS_ROOT/${RUN_NAME}_bridge_specialist_test_ood_predictions.jsonl}"

mkdir -p "$RESULTS_ROOT"

"$PYTHON_BIN" "$ROOT/code/clrs_native_text/bridge_native/run_clrs_text_bridge_to_trm_benchmark_native.py" \
  --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
  --trm-checkpoint "$TRM_CHECKPOINT" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --max-source-length "${MAX_SOURCE_LENGTH:-2048}" \
  --present-threshold "${PRESENT_THRESHOLD:-0.5}" \
  --edge-threshold "${EDGE_THRESHOLD:-0.5}" \
  --llm-dtype "${LLM_DTYPE:-auto}" \
  --device "${DEVICE:-cuda}" \
  --continuous-feature-clip "${CONTINUOUS_FEATURE_CLIP:-5.0}" \
  --eval-num-recurrences "${EVAL_NUM_RECURRENCES:-24}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --task-filter "${TASK_FILTER:-all}" \
  --progress-every "${PROGRESS_EVERY:-25}"
