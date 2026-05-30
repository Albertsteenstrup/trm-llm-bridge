#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_hard_nl_native_v1}"
NATIVE_DIR="${NATIVE_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_native}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/clrs_native_text}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/clrs_native_text}"
TAG="${TAG:-${RUN_NAME}_specialist_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT"

cmd=(
  "$PYTHON_BIN" "$ROOT/code/trm_llm/stage3_specialists/train_clrs_trm_specialist_benchmark_native.py"
  --train-jsonl "${TRAIN_JSONL:-$NATIVE_DIR/train.jsonl}"
  --val-jsonl "${VAL_JSONL:-$NATIVE_DIR/val.jsonl}"
  --test-jsonl "${TEST_JSONL:-$NATIVE_DIR/test_ood_size.jsonl}"
  --train-split train
  --val-split val
  --test-split test_ood_size
  --task-filter "${TASK_FILTER:-all}"
  --epochs "${EPOCHS:-80}"
  --batch-size "${BATCH_SIZE:-8}"
  --hidden-dim "${HIDDEN_DIM:-256}"
  --num-recurrences "${NUM_RECURRENCES:-16}"
  --inner-recurrences "${INNER_RECURRENCES:-2}"
  --recurrent-depth "${RECURRENT_DEPTH:-3}"
  --eval-num-recurrences "${EVAL_NUM_RECURRENCES:-24}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-4}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}"
  --forward-dtype "${FORWARD_DTYPE:-float32}"
  --raw-max-nodes "${RAW_MAX_NODES:-64}"
  --train-node-permute-prob "${TRAIN_NODE_PERMUTE_PROB:-0.5}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-12}"
  --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS:-20}"
  --checkpoint-path "${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/${TAG}.pt}"
  --checkpoint-every-epochs "${CHECKPOINT_EVERY_EPOCHS:-5}"
  --metrics-json "${METRICS_JSON:-$RESULTS_ROOT/${TAG}.metrics.json}"
  --amp "${AMP:-bf16}"
  --progress-every-batches "${PROGRESS_EVERY_BATCHES:-100}"
)

if [[ "${WANDB:-0}" == "1" ]]; then
  cmd+=(--wandb --wandb-project "${WANDB_PROJECT:-clrs-native-text-specialist}" --wandb-run-name "${WANDB_RUN_NAME:-$TAG}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    cmd+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
