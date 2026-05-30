#!/bin/bash
# =============================================================================
# run_trm_on_messy_gt.sbatch
#
# Runs TRM inference on the ground-truth grids from the messy pipeline test set
# (1,000 puzzles). This gives a TRM ceiling measured on the exact same puzzle
# sample as the pipeline end-to-end results, making the gap a precise subtraction
# rather than an estimate across different held-out sets.
#
# Usage:
#   sbatch scripts/initial/run_trm_on_messy_gt.sbatch
# =============================================================================

#SBATCH --job-name=trm_messy_gt
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_a100_80gb|gpu_a30|gpu_rtx8000|gpu_v100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/initial/trm_messy_gt_%j.out
#SBATCH --error=logs/initial/trm_messy_gt_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ---------- Modules & Environment ----------
unset -f module 2>/dev/null || true
if ! command -v module >/dev/null 2>&1; then
    set +u; [ -f /etc/profile ] && source /etc/profile || true; set -u
fi
if command -v module >/dev/null 2>&1; then
    module purge || true
    module use /opt/itu/easybuild/modules/all || true
    module load Anaconda3 || true
fi

if ! command -v conda >/dev/null 2>&1; then
    set +u
    for candidate in /opt/itu/easybuild/software/Anaconda3/*/bin/conda; do
        if [ -x "$candidate" ]; then ANACONDA_CONDA_BIN="$candidate"; fi
    done
    if [ -n "${ANACONDA_CONDA_BIN:-}" ]; then
        __conda_setup="$($ANACONDA_CONDA_BIN shell.bash hook 2>/dev/null || true)"
        if [ -n "$__conda_setup" ]; then eval "$__conda_setup"
        else
            CONDA_SH="${ANACONDA_CONDA_BIN%/bin/conda}/etc/profile.d/conda.sh"
            [ -f "$CONDA_SH" ] && . "$CONDA_SH" || export PATH="${ANACONDA_CONDA_BIN%/bin/conda}/bin:$PATH"
        fi
        unset __conda_setup
    fi
    set -u
fi

set +u
eval "$(conda shell.bash hook)"
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/run_trm_checkpoint_inference.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

TRM_CHECKPOINT_DIR="$ROOT/checkpoints/initial/trm-sudoku"
TRM_CHECKPOINT_FILE="$ROOT/checkpoints/initial/trm-sudoku/step_65100"
TRM_ROOT="$ROOT/code/initial/TinyRecursiveModels"
TEST_JSON="$ROOT/data/initial/sudoku_grid/test/messy_test_gt_n1000.json"
OUTPUT_JSON="$ROOT/results/trm_messy_gt/trm_predictions_messy_gt.json"

mkdir -p "$ROOT/results/trm_messy_gt"

echo "=== TRM inference on messy GT grids ==="
echo "TEST_JSON=$TEST_JSON"
echo "OUTPUT_JSON=$OUTPUT_JSON"

conda run -n trm-env python "$ROOT/code/initial/run_trm_checkpoint_inference.py" \
    --checkpoint-dir "$TRM_CHECKPOINT_DIR" \
    --checkpoint-file "$TRM_CHECKPOINT_FILE" \
    --trm-root "$TRM_ROOT" \
    --test-json "$TEST_JSON" \
    --output-json "$OUTPUT_JSON" \
    --require-test-path "" \
    --max-puzzles 1000

echo "Done. Output: $OUTPUT_JSON"
echo "Job finished at $(date)"
