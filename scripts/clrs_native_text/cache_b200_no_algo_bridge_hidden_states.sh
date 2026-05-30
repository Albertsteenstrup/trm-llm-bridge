#!/usr/bin/env bash
set -euo pipefail

# Cache frozen Qwen hidden states for the no-algorithm CLRS bridge curriculum.
# Cache one or more curriculum levels before training; pass HIDDEN_CACHE_DIR to
# train_b200_dynamic_curriculum_bridge.sh to use the generated caches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_no_algo_curriculum_v4}"
CURRICULUM_DIR="${CURRICULUM_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_bridge_curriculum}"
CACHE_DIR="${CACHE_DIR:-$DATA_ROOT/cache/${RUN_NAME}_qwen_hidden}"
LEVELS="${LEVELS:-0}"
SPLITS="${SPLITS:-train val}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-Qwen/Qwen3-1.7B}"
LLM_LAYER_INDEX="${LLM_LAYER_INDEX:--1}"
LLM_DTYPE="${LLM_DTYPE:-auto}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-2048}"
BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
SHARD_SIZE_ROWS="${SHARD_SIZE_ROWS:-2000}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

for level in $LEVELS; do
  level_dir="$CURRICULUM_DIR/level_${level}"
  out_dir="$CACHE_DIR/level_${level}"
  mkdir -p "$out_dir"
  for split in $SPLITS; do
    input_jsonl="$level_dir/${split}.jsonl"
    output_path="$out_dir/${split}.pt"
    if [[ ! -f "$input_jsonl" ]]; then
      echo "ERROR: missing input split: $input_jsonl" >&2
      exit 1
    fi
    if [[ -f "$output_path" && "${FORCE:-0}" != "1" ]]; then
      echo "[cache] skip existing level=$level split=$split path=$output_path"
      continue
    fi
    echo "[cache] level=$level split=$split input=$input_jsonl output=$output_path"
    "$PYTHON_BIN" -u "$ROOT/code/clrs_native_text/bridge_native/cache_clrs_text_bridge_hidden_states.py" \
      --input-jsonl "$input_jsonl" \
      --output-path "$output_path" \
      --llm-model-name "$LLM_MODEL_NAME" \
      --llm-layer-index "$LLM_LAYER_INDEX" \
      --llm-dtype "$LLM_DTYPE" \
      --cache-dtype "$CACHE_DTYPE" \
      --max-source-length "$MAX_SOURCE_LENGTH" \
      --batch-size "$BATCH_SIZE" \
      --shard-size-rows "$SHARD_SIZE_ROWS" \
      --device "$DEVICE" \
      --progress-every "${PROGRESS_EVERY:-500}"
  done
done

echo "Hidden cache directory: $CACHE_DIR"
