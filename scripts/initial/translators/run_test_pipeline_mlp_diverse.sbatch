#!/bin/bash
# =============================================================================
# run_test_pipeline_mlp_diverse.sbatch
#
# Tests the MLP Projector bridge (trained on diverse 10K) on diverse and messy
# test sets, plus the LLM OOD set.
#
# Usage:
#   sbatch scripts/initial/translators/run_test_pipeline_mlp_diverse.sbatch
# =============================================================================

#SBATCH --job-name=test_pipe_mlp_diverse
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_a100_80gb|gpu_a30'
#SBATCH --nodelist=cn7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/initial/test_pipe_mlp_diverse_%j.out
#SBATCH --error=logs/initial/test_pipe_mlp_diverse_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ---------- Modules & Environment ----------
unset -f module 2>/dev/null || true
if ! command -v module >/dev/null 2>&1; then
    set +u; [ -f /etc/profile ] && source /etc/profile || true; set -u
fi
if ! command -v module >/dev/null 2>&1; then
    set +u; [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true; set -u
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

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found."; exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate llm-env
set -u

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$ROOT"

PIPELINE_PY="$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py"
QWEN_MODEL="$ROOT/checkpoints/initial/Qwen3-1.7B"
BRIDGE_CHECKPOINT="$ROOT/checkpoints/initial/translator_bridge/mlp-10k-diverse-fixed59/best_mlp.pt"
BRIDGE_ARCH="mlp"
TRM_CHECKPOINT_DIR="$ROOT/checkpoints/initial/trm-sudoku"
TRM_CHECKPOINT_FILE="$ROOT/checkpoints/initial/trm-sudoku/step_65100"
TRM_ROOT="$ROOT/code/initial/TinyRecursiveModels"
MAX_SAMPLES=1000
WANDB_PROJECT="trm-llm-test-pipeline"

run_pipeline() {
    local label="$1"
    local input_json="$2"
    local results_dir="$3"
    local use_wandb="${4:-0}"

    mkdir -p "$results_dir"
    local bridge_out="$results_dir/bridge_predictions.json"
    local trm_in="$results_dir/trm_input.json"
    local trm_out="$results_dir/trm_predictions.json"
    local metrics="$results_dir/pipeline_metrics.json"

    echo "=== [$label] Stage 1/3: bridge inference ==="
    python "$PIPELINE_PY" \
        --stage bridge \
        --input-json "$input_json" \
        --qwen-model "$QWEN_MODEL" \
        --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
        --bridge-arch "$BRIDGE_ARCH" \
        --max-samples "$MAX_SAMPLES" \
        --bridge-output-json "$bridge_out" \
        --trm-input-json "$trm_in" \
        --metrics-json "$metrics" \
        --require-nl-description

    echo "=== [$label] Stage 2/3: TRM inference ==="
    conda run -n trm-env python "$ROOT/code/initial/run_trm_checkpoint_inference.py" \
        --checkpoint-dir "$TRM_CHECKPOINT_DIR" \
        --checkpoint-file "$TRM_CHECKPOINT_FILE" \
        --trm-root "$TRM_ROOT" \
        --test-json "$trm_in" \
        --output-json "$trm_out" \
        --require-test-path "" \
        --max-puzzles "$MAX_SAMPLES"

    echo "=== [$label] Stage 3/3: finalize metrics ==="
    if [ "$use_wandb" = "1" ]; then
        python "$PIPELINE_PY" \
            --stage finalize \
            --bridge-output-json "$bridge_out" \
            --trm-predictions-json "$trm_out" \
            --metrics-json "$metrics" \
            --wandb \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-run-name "${label//\//_}"
    else
        python "$PIPELINE_PY" \
            --stage finalize \
            --bridge-output-json "$bridge_out" \
            --trm-predictions-json "$trm_out" \
            --metrics-json "$metrics"
    fi

    echo "[$label] Done. Metrics: $metrics"
}

run_pipeline "mlp_diverse/diverse" \
    "$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json" \
    "$ROOT/results/test_pipeline_mlp_diverse/diverse" \
    "1"

run_pipeline "mlp_diverse/messy" \
    "$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_messy_test.json" \
    "$ROOT/results/test_pipeline_mlp_diverse/messy" \
    "1"

run_pipeline "mlp_diverse/ood_llm" \
    "$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json" \
    "$ROOT/results/test_pipeline_mlp_diverse/ood_llm" \
    "1"

echo "All runs complete. Job finished at $(date)"
