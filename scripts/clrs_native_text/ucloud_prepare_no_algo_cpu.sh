#!/usr/bin/env bash
set -euo pipefail

# UCloud CPU launcher for CLRS native-text data preparation.
# Intended as the foreground command of a Terminal/PyTorch batch job.
# Do not wrap this script in "nohup ... &" when using UCloud batch mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export RUN_NAME="${RUN_NAME:-clrs30_no_algo_curriculum_v3}"
export TARGET_TOTAL="${TARGET_TOTAL:-60000}"
export TEST_OOD_TOTAL="${TEST_OOD_TOTAL:-4500}"
export TRAIN_LENGTHS="${TRAIN_LENGTHS:-8,12,16}"
export VAL_LENGTHS="${VAL_LENGTHS:-16}"
export TEST_LENGTHS="${TEST_LENGTHS:-64}"
export TEST_OOD_LENGTHS="${TEST_OOD_LENGTHS:-64}"
export PREP_THREADS="${PREP_THREADS:-8}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
export STREAM_SUBPROCESS_LOGS="${STREAM_SUBPROCESS_LOGS:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PREP_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$PREP_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

mkdir -p logs

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  bash scripts/clrs_native_text/install_ucloud_deps.sh
fi

log_path="logs/prepare_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[ucloud-prepare] log=$log_path"
echo "[ucloud-prepare] run=$RUN_NAME target=$TARGET_TOTAL ood=$TEST_OOD_TOTAL threads=$PREP_THREADS gen_batch=$GENERATION_BATCH_SIZE"
echo "[ucloud-prepare] subprocess logs under data/clrs_native_text/raw/$RUN_NAME/chunk_logs"

set +e
scripts/clrs_native_text/prepare_b200_no_algo_curriculum.sh >"$log_path" 2>&1
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
  echo "[ucloud-prepare] failed with exit code $status; tail of $log_path:"
  tail -160 "$log_path" || true
  echo "[ucloud-prepare] latest chunk logs:"
  find "data/clrs_native_text/raw/$RUN_NAME/chunk_logs" -type f -name '*.log' -print 2>/dev/null | sort | tail -5 || true
  latest_chunk="$(find "data/clrs_native_text/raw/$RUN_NAME/chunk_logs" -type f -name '*.log' -print 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$latest_chunk" ]]; then
    echo "[ucloud-prepare] tail of latest chunk log: $latest_chunk"
    tail -160 "$latest_chunk" || true
  fi
  exit "$status"
fi

echo "[ucloud-prepare] finished; tail of log:"
tail -80 "$log_path"
