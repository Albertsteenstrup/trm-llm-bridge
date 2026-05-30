#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_no_algo_curriculum_v1}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw/$RUN_NAME}"
NATIVE_SPLIT_DIR="${NATIVE_SPLIT_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_native}"
CURRICULUM_DIR="${CURRICULUM_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_bridge_curriculum}"

CLRS30_ALGORITHMS="articulation_points,activity_selector,bellman_ford,bfs,binary_search,bridges,bubble_sort,dag_shortest_paths,dfs,dijkstra,find_maximum_subarray_kadane,floyd_warshall,graham_scan,heapsort,insertion_sort,jarvis_march,kmp_matcher,lcs_length,matrix_chain_order,minimum,mst_kruskal,mst_prim,naive_string_matcher,optimal_bst,quickselect,quicksort,segments_intersect,strongly_connected_components,task_scheduling,topological_sort"
ALGORITHMS="${ALGORITHMS:-$CLRS30_ALGORITHMS}"
TARGET_TOTAL="${TARGET_TOTAL:-60000}"
TRAIN_LENGTHS="${TRAIN_LENGTHS:-8,12,16}"
VAL_LENGTHS="${VAL_LENGTHS:-16}"
TEST_LENGTHS="${TEST_LENGTHS:-64}"
TEST_OOD_TOTAL="${TEST_OOD_TOTAL:-4500}"
TEST_OOD_LENGTHS="${TEST_OOD_LENGTHS:-64}"
SEED="${SEED:-42}"
VARIANTS_PER_ROW="${VARIANTS_PER_ROW:-1}"
MAX_ROWS_PER_SPLIT="${MAX_ROWS_PER_SPLIT:-0}"
PREP_THREADS="${PREP_THREADS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
STREAM_SUBPROCESS_LOGS="${STREAM_SUBPROCESS_LOGS:-0}"

RAW_JSONL="$RAW_DIR/clrs_native_raw.jsonl"
CHUNK_LOG_DIR="${CHUNK_LOG_DIR:-$RAW_DIR/chunk_logs}"

mkdir -p "$RAW_DIR" "$NATIVE_SPLIT_DIR" "$CURRICULUM_DIR"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PREP_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$PREP_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

staged_args=(
  "$PYTHON_BIN" -u "$ROOT/code/clrs_native_text/prepare_no_algo_curriculum_staged.py"
  --output-dir "$RAW_DIR"
  --output-jsonl "$(basename "$RAW_JSONL")"
  --algorithms "$ALGORITHMS"
  --target-total "$TARGET_TOTAL"
  --train-lengths "$TRAIN_LENGTHS"
  --val-lengths "$VAL_LENGTHS"
  --test-lengths "$TEST_LENGTHS"
  --test-ood-total "$TEST_OOD_TOTAL"
  --test-ood-lengths "$TEST_OOD_LENGTHS"
  --seed "$SEED"
  --generation-batch-size "$GENERATION_BATCH_SIZE"
  --python-bin "$PYTHON_BIN"
  --chunk-log-dir "$CHUNK_LOG_DIR"
)
if [[ "$STREAM_SUBPROCESS_LOGS" == "1" ]]; then
  staged_args+=(--stream-subprocess-logs)
fi

"${staged_args[@]}"

"$PYTHON_BIN" "$ROOT/code/clrs_native_text/split_native_raw_by_split.py" \
  --input-jsonl "$RAW_JSONL" \
  --output-dir "$NATIVE_SPLIT_DIR" \
  --splits train,val,test,test_ood_size

"$PYTHON_BIN" "$ROOT/code/clrs_native_text/build_no_algo_curriculum_bridge_dataset.py" \
  --input-jsonl "$RAW_JSONL" \
  --output-dir "$CURRICULUM_DIR" \
  --variants-per-row "$VARIANTS_PER_ROW" \
  --max-rows-per-split "$MAX_ROWS_PER_SPLIT" \
  --seed "$SEED"

cat > "$DATA_ROOT/prepared/${RUN_NAME}_manifest.json" <<JSON
{
  "run_name": "$RUN_NAME",
  "raw_jsonl": "$RAW_JSONL",
  "native_split_dir": "$NATIVE_SPLIT_DIR",
  "curriculum_dir": "$CURRICULUM_DIR",
  "algorithms": "$ALGORITHMS",
  "target_total": $TARGET_TOTAL,
  "train_lengths": "$TRAIN_LENGTHS",
  "val_lengths": "$VAL_LENGTHS",
  "test_lengths": "$TEST_LENGTHS",
  "test_ood_total": $TEST_OOD_TOTAL,
  "test_ood_lengths": "$TEST_OOD_LENGTHS",
  "variants_per_row": $VARIANTS_PER_ROW,
  "generation_batch_size": $GENERATION_BATCH_SIZE,
  "bridge_task_label": "unknown",
  "question_hides_algorithm": true,
  "question_includes_native_schema": false
}
JSON

echo "Prepared native specialist splits in: $NATIVE_SPLIT_DIR"
echo "Prepared no-algorithm bridge curriculum in: $CURRICULUM_DIR"
