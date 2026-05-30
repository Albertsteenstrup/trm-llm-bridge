#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_hard_nl_native_v1}"
BRIDGE_DIR="${BRIDGE_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_bridge}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/clrs_native_text/bridge}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/clrs_native_text}"
TAG="${TAG:-${RUN_NAME}_bridge_$(date +%Y%m%d_%H%M%S)}"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cmd=(
  "$PYTHON_BIN" "$ROOT/code/clrs_native_text/bridge_native/train_clrs_text_bridge_benchmark_native.py"
  --arch "${ARCH:-qformer}"
  --train-jsonl "${TRAIN_JSONL:-$BRIDGE_DIR/train.jsonl}"
  --val-jsonl "${VAL_JSONL:-$BRIDGE_DIR/val.jsonl}"
  --llm-model-name "${LLM_MODEL_NAME:-Qwen/Qwen3-1.7B}"
  --llm-layer-index "${LLM_LAYER_INDEX:--1}"
  --llm-dtype "${LLM_DTYPE:-auto}"
  --output-dir "${OUTPUT_DIR:-$CHECKPOINT_ROOT/$TAG}"
  --metrics-json "${METRICS_JSON:-$RESULTS_ROOT/${TAG}.metrics.json}"
  --epochs "${EPOCHS:-12}"
  --batch-size "${TRAIN_BATCH_SIZE:-2}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-2}"
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-8}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-2}"
  --warmup-ratio "${WARMUP_RATIO:-0.05}"
  --max-source-length "${MAX_SOURCE_LENGTH:-2048}"
  --max-nodes "${MAX_NODES:-64}"
  --bridge-dim "${BRIDGE_DIM:-512}"
  --bridge-heads "${BRIDGE_HEADS:-8}"
  --bridge-layers "${BRIDGE_LAYERS:-4}"
  --bridge-dropout "${BRIDGE_DROPOUT:-0.1}"
  --query-pos-dropout "${QUERY_POS_DROPOUT:-0.05}"
  --count-blend-alpha "${COUNT_BLEND_ALPHA:-0.5}"
  --count-decode-mode "${COUNT_DECODE_MODE:-blended}"
  --scalar-head-type "${SCALAR_HEAD_TYPE:-mlp}"
  --feature-quant-bins "${FEATURE_QUANT_BINS:-256}"
  --continuous-feature-clip "${CONTINUOUS_FEATURE_CLIP:-5.0}"
  --seed "${SEED:-42}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}"
  --present-threshold "${PRESENT_THRESHOLD:-0.5}"
  --scalar-exact-tolerance "${SCALAR_EXACT_TOLERANCE:-0.001}"
  --count-loss-weight-implicit "${COUNT_LOSS_WEIGHT_IMPLICIT:-0.10}"
  --count-loss-weight-explicit "${COUNT_LOSS_WEIGHT_EXPLICIT:-0.60}"
  --count-loss-weight-blended "${COUNT_LOSS_WEIGHT_BLENDED:-0.30}"
  --loss-weight-present "${LOSS_WEIGHT_PRESENT:-0.5}"
  --loss-weight-count "${LOSS_WEIGHT_COUNT:-0.3}"
  --loss-weight-node-scalar "${LOSS_WEIGHT_NODE_SCALAR:-1.0}"
  --loss-weight-node-discrete "${LOSS_WEIGHT_NODE_DISCRETE:-2.0}"
  --loss-weight-edge-scalar "${LOSS_WEIGHT_EDGE_SCALAR:-0.6}"
  --loss-weight-edge-discrete "${LOSS_WEIGHT_EDGE_DISCRETE:-0.8}"
  --loss-weight-graph-scalar "${LOSS_WEIGHT_GRAPH_SCALAR:-0.4}"
  --loss-weight-graph-discrete "${LOSS_WEIGHT_GRAPH_DISCRETE:-0.5}"
  --loss-weight-present-monotonic "${LOSS_WEIGHT_PRESENT_MONOTONIC:-0.05}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-8}"
  --checkpoint-every-epochs "${CHECKPOINT_EVERY_EPOCHS:-1}"
  --log-every-batches "${LOG_EVERY_BATCHES:-250}"
  --wandb-project "${WANDB_PROJECT:-clrs-native-text-bridge}"
  --wandb-run-name "${WANDB_RUN_NAME:-$TAG}"
  --wandb-group "${WANDB_GROUP:-$RUN_NAME}"
)

if [[ "${WANDB:-0}" != "1" ]]; then
  cmd+=(--no-wandb)
fi
if [[ "${WANDB:-0}" == "1" && -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
