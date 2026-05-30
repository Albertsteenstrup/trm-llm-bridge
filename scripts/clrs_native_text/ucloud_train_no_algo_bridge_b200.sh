#!/usr/bin/env bash
set -euo pipefail

# UCloud/B200 launcher for the no-algorithm CLRS native-text curriculum bridge.
# This mirrors the floorplan UCloud style: one small wrapper with UCloud-safe
# defaults, while the lower-level training script still accepts env overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export RUN_NAME="${RUN_NAME:-clrs30_no_algo_curriculum_v1}"
export WANDB="${WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-albst-it-universitetet-i-k-benhavn}"
export WANDB_PROJECT="${WANDB_PROJECT:-clrs-native-text-curriculum}"
export ARCH="${ARCH:-qformer}"
export WANDB_GROUP="${WANDB_GROUP:-no_algo_${ARCH}_b200}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb}"
export MAX_EPOCHS_PER_LEVEL="${MAX_EPOCHS_PER_LEVEL:-12}"
export MIN_EPOCHS_PER_LEVEL="${MIN_EPOCHS_PER_LEVEL:-1}"
export ADVANCE_METRIC="${ADVANCE_METRIC:-val.selection_score}"
export ADVANCE_THRESHOLDS="${ADVANCE_THRESHOLDS:-0:0.80,1:0.75,2:0.70,3:0.65,4:0.60}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-2048}"
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-Qwen/Qwen3-1.7B}"
export HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-}"

mkdir -p "$WANDB_DIR" logs

scripts/clrs_native_text/train_b200_dynamic_curriculum_bridge.sh "$@"
