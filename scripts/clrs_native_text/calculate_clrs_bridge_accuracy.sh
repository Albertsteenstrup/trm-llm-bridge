#!/usr/bin/env bash
# Run accuracy calculation on all CLRS algorithms in the local workspace

ALGS="bfs dfs bellman_ford dijkstra dag_shortest_paths binary_search insertion_sort"
CLRS_DIR="clrs-native-text"
PYTHON_BIN=".venv-ucloud/bin/python"

if [[ ! -f "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

echo "Using python: $PYTHON_BIN"

for ALG in $ALGS; do
  PRED_JSONL="$CLRS_DIR/bridge_pred_native_${ALG}_level4_test.jsonl"
  CURR_JSONL="$CLRS_DIR/${ALG}_level4_test.jsonl"
  
  if [[ -f "$PRED_JSONL" && -f "$CURR_JSONL" ]]; then
    echo "========================================================================"
    echo "Processing $ALG..."
    echo "========================================================================"
    $PYTHON_BIN code/clrs_native_text/calculate_clrs_bridge_accuracy.py \
      --predicted-jsonl "$PRED_JSONL" \
      --curriculum-jsonl "$CURR_JSONL" \
      --tolerances "1e-5,1e-3,0.05,0.25"
  else
    echo "Skipping $ALG: prediction or curriculum file not found."
  fi
done
