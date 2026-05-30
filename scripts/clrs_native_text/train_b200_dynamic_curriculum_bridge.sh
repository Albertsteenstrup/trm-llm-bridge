#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_no_algo_curriculum_v1}"
CURRICULUM_DIR="${CURRICULUM_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_bridge_curriculum}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/clrs_native_text/bridge_curriculum}"
TAG="${TAG:-${RUN_NAME}_${ARCH:-qformer}_dynamic_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/$TAG}"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cmd=(
  "$PYTHON_BIN" "$ROOT/code/clrs_native_text/train_dynamic_bridge_curriculum.py"
  --python-bin "$PYTHON_BIN" \
  --curriculum-dir "$CURRICULUM_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --levels "${LEVELS:-0,1,2,3,4}" \
  --advance-metric "${ADVANCE_METRIC:-val.selection_score}" \
  --advance-thresholds "${ADVANCE_THRESHOLDS:-0:0.80,1:0.75,2:0.70,3:0.65,4:0.60}" \
  --max-epochs-per-level "${MAX_EPOCHS_PER_LEVEL:-12}" \
  --min-epochs-per-level "${MIN_EPOCHS_PER_LEVEL:-1}" \
  --arch "${ARCH:-qformer}" \
  --llm-model-name "${LLM_MODEL_NAME:-Qwen/Qwen3-1.7B}" \
  --llm-layer-index "${LLM_LAYER_INDEX:--1}" \
  --llm-dtype "${LLM_DTYPE:-auto}" \
  --batch-size "${TRAIN_BATCH_SIZE:-2}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-2}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-8}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-2}" \
  --warmup-ratio "${WARMUP_RATIO:-0.05}" \
  --max-source-length "${MAX_SOURCE_LENGTH:-2048}" \
  --max-nodes "${MAX_NODES:-64}" \
  --bridge-dim "${BRIDGE_DIM:-512}" \
  --bridge-heads "${BRIDGE_HEADS:-8}" \
  --bridge-layers "${BRIDGE_LAYERS:-4}" \
  --bridge-dropout "${BRIDGE_DROPOUT:-0.1}" \
  --query-pos-dropout "${QUERY_POS_DROPOUT:-0.05}" \
  --count-blend-alpha "${COUNT_BLEND_ALPHA:-0.5}" \
  --count-decode-mode "${COUNT_DECODE_MODE:-blended}" \
  --scalar-head-type "${SCALAR_HEAD_TYPE:-mlp}" \
  --scalar-head-hidden-mult "${SCALAR_HEAD_HIDDEN_MULT:-1.0}" \
  --edge-text-prior-logit-scale "${EDGE_TEXT_PRIOR_LOGIT_SCALE:-0.0}" \
  --transnar-recurrence-steps "${TRANSNAR_RECURRENCE_STEPS:-1}" \
  --feature-quant-bins "${FEATURE_QUANT_BINS:-256}" \
  --continuous-feature-clip "${CONTINUOUS_FEATURE_CLIP:-5.0}" \
  --seed "${SEED:-42}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --present-threshold "${PRESENT_THRESHOLD:-0.5}" \
  --scalar-exact-tolerance "${SCALAR_EXACT_TOLERANCE:-0.001}" \
  --count-loss-weight-implicit "${COUNT_LOSS_WEIGHT_IMPLICIT:-0.25}" \
  --count-loss-weight-explicit "${COUNT_LOSS_WEIGHT_EXPLICIT:-0.25}" \
  --count-loss-weight-blended "${COUNT_LOSS_WEIGHT_BLENDED:-0.50}" \
  --loss-weight-present-monotonic "${LOSS_WEIGHT_PRESENT_MONOTONIC:-0.05}" \
  --loss-weight-present "${LOSS_WEIGHT_PRESENT:-0.3}" \
  --loss-weight-count "${LOSS_WEIGHT_COUNT:-0.1}" \
  --loss-weight-node-scalar "${LOSS_WEIGHT_NODE_SCALAR:-0.5}" \
  --loss-weight-node-discrete "${LOSS_WEIGHT_NODE_DISCRETE:-2.0}" \
  --loss-weight-edge-scalar "${LOSS_WEIGHT_EDGE_SCALAR:-0.3}" \
  --loss-weight-edge-discrete "${LOSS_WEIGHT_EDGE_DISCRETE:-0.8}" \
  --edge-binary-pos-weight-cap "${EDGE_BINARY_POS_WEIGHT_CAP:-10.0}" \
  --edge-binary-neg-penalty "${EDGE_BINARY_NEG_PENALTY:-0.0}" \
  --loss-weight-graph-scalar "${LOSS_WEIGHT_GRAPH_SCALAR:-0.2}" \
  --loss-weight-graph-discrete "${LOSS_WEIGHT_GRAPH_DISCRETE:-0.5}" \
  --max-train-rows "${MAX_TRAIN_ROWS:-0}" \
  --max-val-rows "${MAX_VAL_ROWS:-0}" \
  --checkpoint-every-epochs "${CHECKPOINT_EVERY_EPOCHS:-1}" \
  --log-every-batches "${LOG_EVERY_BATCHES:-250}" \
  --wandb-project "${WANDB_PROJECT:-clrs-native-text-bridge-curriculum}" \
  --wandb-run-name "${WANDB_RUN_NAME:-$TAG}" \
  --wandb-group "${WANDB_GROUP:-${RUN_NAME}_${ARCH:-qformer}}"
)

if [[ "${QFORMER_EDGE_SELF_BIAS:-0}" == "1" ]]; then
  cmd+=(--qformer-edge-self-bias)
fi
if [[ "${QFORMER_REL_POS_SELF_BIAS:-0}" == "1" ]]; then
  cmd+=(--qformer-rel-pos-self-bias)
fi
if [[ "${WANDB:-0}" == "1" ]]; then
  cmd+=(--wandb)
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    cmd+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi
if [[ -n "${HIDDEN_CACHE_DIR:-}" ]]; then
  cmd+=(--hidden-cache-dir "$HIDDEN_CACHE_DIR")
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"

echo "Dynamic bridge curriculum output: $OUTPUT_DIR"
