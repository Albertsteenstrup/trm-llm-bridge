#!/usr/bin/env bash
set -euo pipefail

# This script loops over all selected algorithms, filters the bridge curriculum JSONL,
# exports the bridge predicted native inputs, finds the ReNAR checkpoint,
# and runs the evaluation.

ALGS="${ALGS:-bfs dfs bellman_ford dijkstra dag_shortest_paths binary_search minimum insertion_sort}"
TAG="${TAG:-clrs30_no_algo_curriculum_v4_subset10k_parserprior_transnar_lowthreshold_20260525_104410}"
RUN_DIR="/work/CLRS/thesis/checkpoints/clrs_native_text/bridge_curriculum/$TAG"
BRIDGE_CKPT="$RUN_DIR/last.pt"

# Defaults can be overridden by environment variables
SRC="${SRC:-/work/CLRS/thesis/data/clrs_native_text/prepared/clrs30_no_algo_curriculum_v4_bridge_curriculum_subset10k/level_4/test.jsonl}"
NATIVE_GOLD="${NATIVE_GOLD:-/work/CLRS/thesis/data/clrs_native_text/prepared/clrs30_no_algo_curriculum_v4_native/test.jsonl}"
SUFFIX="${SUFFIX:-level4_test}"

echo "Starting loop for algorithms: $ALGS"
echo "Bridge checkpoint: $BRIDGE_CKPT"
echo "Curriculum source: $SRC"
echo "Native gold source: $NATIVE_GOLD"
echo "Output suffix: $SUFFIX"

if [[ ! -f "$SRC" ]]; then
  echo "Error: curriculum source file not found: $SRC" >&2
  exit 1
fi
if [[ ! -f "$NATIVE_GOLD" ]]; then
  echo "Error: native gold file not found: $NATIVE_GOLD" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

# Resolve which python to use for bridge export (bypassing any active JAX venv)
if [[ -f "/usr/bin/python3" ]]; then
  EXPORT_PYTHON="/usr/bin/python3"
else
  EXPORT_PYTHON="python3"
fi

for ALG in $ALGS; do
  echo
  echo "=================================================="
  echo "ALG=$ALG: 1/3 filtering inputs... $(date)"
  echo "=================================================="
  
  ALG_JSONL="$RUN_DIR/${ALG}_${SUFFIX}.jsonl"
  PRED_JSONL="$RUN_DIR/bridge_pred_native_${ALG}_${SUFFIX}.jsonl"
  
  # Filter the curriculum file for this algorithm
  $EXPORT_PYTHON - <<PY
import json
with open("$SRC") as f, open("$ALG_JSONL", "w") as g:
    n = 0
    for line in f:
        row = json.loads(line)
        a = row.get("algorithm") or row.get("algo_name") or row.get("native_input_target", {}).get("algorithm")
        if a == "$ALG":
            g.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
print(f"Filtered {n} rows for $ALG -> $ALG_JSONL")
if n == 0:
    raise SystemExit(2)
PY

  echo
  echo "=================================================="
  echo "ALG=$ALG: 2/3 exporting bridge predictions... $(date)"
  echo "=================================================="
  
  # Run the PyTorch bridge to export native inputs (make sure we are deactivated from any other venv first)
  deactivate 2>/dev/null || true
  
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/work/CLRS/thesis/code \
  $EXPORT_PYTHON /work/CLRS/thesis/code/clrs_native_text/bridge_native/export_bridge_predicted_native_inputs.py \
    --bridge-checkpoint "$BRIDGE_CKPT" \
    --input-jsonl "$ALG_JSONL" \
    --output-jsonl "$PRED_JSONL" \
    --algorithms "$ALG" \
    --limit-samples 0 \
    --count-decode-mode prior \
    --edge-text-prior-logit-scale 4.0 \
    --llm-dtype auto

  echo
  echo "=================================================="
  echo "ALG=$ALG: 3/3 scoring with ReNAR... $(date)"
  echo "=================================================="

  MODEL_FILE=$(find /work/CLRS/renar_runs -path '*checkpoints*' -type f -name "single_${ALG}_*_best_MLM.pkl" | sort | tail -1)
  if [[ -z "$MODEL_FILE" ]]; then
    echo "Warning: No ReNAR checkpoint found for $ALG; skipping evaluation"
    continue
  fi
  
  CKPT_DIR=$(dirname "$MODEL_FILE")
  OUT_JSON="$RUN_DIR/bridge_to_renar_${ALG}_${SUFFIX}.json"
  
  echo "Found checkpoint folder: $CKPT_DIR"
  echo "Checkpoint file: $MODEL_FILE"
  echo "Output will go to: $OUT_JSON"
  
  # Activate the ReNAR virtual environment
  source /work/CLRS/.venv-renar/bin/activate
  
  # Run the JAX evaluator
  PYTHONPATH=/work/CLRS/renar_ucloud/ReNAR/ReNAR \
  CUDA_VISIBLE_DEVICES=0 python3 /work/CLRS/renar_ucloud/eval_predicted_native_inputs_with_renar.py \
    --predicted-jsonl "$PRED_JSONL" \
    --native-jsonl "$NATIVE_GOLD" \
    --algorithm "$ALG" \
    --checkpoint-path "$CKPT_DIR" \
    --output-json "$OUT_JSON"
    
  deactivate || true
  echo "ALG=$ALG completed."
  if [[ -f "$OUT_JSON" ]]; then
    cat "$OUT_JSON"
  fi
done

echo
echo "DONE all algorithms for suffix: $SUFFIX"
