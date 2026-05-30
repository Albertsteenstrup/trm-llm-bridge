#!/usr/bin/env bash
set -euo pipefail

# Build the CLRS native specialist data and the hard-NL bridge data in the
# isolated clrs_native_text folder. Defaults are sized for a real B200 run but
# can be overridden for smoke tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/clrs_native_text}"
RUN_NAME="${RUN_NAME:-clrs30_hard_nl_native_v1}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw/$RUN_NAME}"
NATIVE_SPLIT_DIR="${NATIVE_SPLIT_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_native}"
BRIDGE_DIR="${BRIDGE_DIR:-$DATA_ROOT/prepared/${RUN_NAME}_bridge}"

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

RAW_JSONL="$RAW_DIR/clrs_native_raw.jsonl"

mkdir -p "$RAW_DIR" "$NATIVE_SPLIT_DIR" "$BRIDGE_DIR"

"$PYTHON_BIN" -u "$ROOT/code/trm_llm/tools/download_clrs_curriculum.py" \
  --output-dir "$RAW_DIR" \
  --output-jsonl "$(basename "$RAW_JSONL")" \
  --algorithms "$ALGORITHMS" \
  --target-total "$TARGET_TOTAL" \
  --train-lengths "$TRAIN_LENGTHS" \
  --val-lengths "$VAL_LENGTHS" \
  --test-lengths "$TEST_LENGTHS" \
  --test-ood-total "$TEST_OOD_TOTAL" \
  --test-ood-lengths "$TEST_OOD_LENGTHS" \
  --seed "$SEED"

"$PYTHON_BIN" "$ROOT/code/clrs_native_text/split_native_raw_by_split.py" \
  --input-jsonl "$RAW_JSONL" \
  --output-dir "$NATIVE_SPLIT_DIR" \
  --splits train,val,test,test_ood_size

"$PYTHON_BIN" "$ROOT/code/clrs_native_text/build_hard_nl_native_bridge_dataset.py" \
  --input-jsonl "$RAW_JSONL" \
  --output-dir "$BRIDGE_DIR" \
  --variants-per-row "$VARIANTS_PER_ROW" \
  --max-rows-per-split "$MAX_ROWS_PER_SPLIT" \
  --seed "$SEED"

cat > "$DATA_ROOT/prepared/${RUN_NAME}_manifest.json" <<JSON
{
  "run_name": "$RUN_NAME",
  "raw_jsonl": "$RAW_JSONL",
  "native_split_dir": "$NATIVE_SPLIT_DIR",
  "bridge_dir": "$BRIDGE_DIR",
  "algorithms": "$ALGORITHMS",
  "target_total": $TARGET_TOTAL,
  "train_lengths": "$TRAIN_LENGTHS",
  "val_lengths": "$VAL_LENGTHS",
  "test_lengths": "$TEST_LENGTHS",
  "test_ood_total": $TEST_OOD_TOTAL,
  "test_ood_lengths": "$TEST_OOD_LENGTHS",
  "variants_per_row": $VARIANTS_PER_ROW
}
JSON

echo "Prepared native specialist splits in: $NATIVE_SPLIT_DIR"
echo "Prepared hard-NL bridge splits in: $BRIDGE_DIR"
