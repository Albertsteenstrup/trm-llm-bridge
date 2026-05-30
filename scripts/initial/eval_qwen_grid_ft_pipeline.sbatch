#!/bin/bash
# =============================================================================
# eval_qwen_grid_ft_pipeline.sbatch
#
# Evaluates the Qwen-FT-as-bridge -> TRM pipeline on three test sets:
#   1. Varied (rule-based, 1000 examples)
#   2. Varied+ (rule-based, 1000 examples)
#   3. OOD / corrected2 (LLM-generated, 1000 examples)
#
# For each test set:
#   Step 1: Qwen FT eval  -> grid predictions + TRM input JSON
#   Step 2: TRM inference -> TRM predictions JSON
#   Step 3: Finalize      -> end-to-end metrics JSON
#
# Usage:
#   sbatch scripts/initial/eval_qwen_grid_ft_pipeline.sbatch
#
# Override checkpoints via env vars:
#   QWEN_VARIED_CKPT   (default: diverse10k best_model.pt)
#   QWEN_VARIEDPLUS_CKPT (default: diverse_then_messy10k best_model.pt)
#   TRM_CHECKPOINT_DIR
# =============================================================================
#SBATCH --job-name=qwen_grid_ft_pipeline_eval
#SBATCH --partition=acltr
#SBATCH --nodelist=cn7
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/initial/qwen_grid_ft_pipeline_eval_%j.out
#SBATCH --error=logs/initial/qwen_grid_ft_pipeline_eval_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ---------- Environment ----------
unset -f module 2>/dev/null || true
set +u
[ -f /etc/profile ] && source /etc/profile || true
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
if command -v module >/dev/null 2>&1; then
    module purge || true
    module use /opt/itu/easybuild/modules/all || true
    module load Anaconda3 || true
fi
if ! command -v conda >/dev/null 2>&1; then
    for candidate in /opt/itu/easybuild/software/Anaconda3/*/bin/conda; do
        if [ -x "$candidate" ]; then
            __conda_setup="$($candidate shell.bash hook 2>/dev/null || true)"
            [ -n "$__conda_setup" ] && eval "$__conda_setup" || . "${candidate%/bin/conda}/etc/profile.d/conda.sh"
            unset __conda_setup
            break
        fi
    done
fi
eval "$(conda shell.bash hook)"
conda activate llm-env
set -u
echo "Conda env: $CONDA_DEFAULT_ENV"
python --version
python -c "import wandb; print('wandb OK')"

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# ---------- Paths ----------
ROOT="$HOME/thesis-trm-llm"
CKPT_DIR="$ROOT/checkpoints/initial"
RESULTS_DIR="$ROOT/results/qwen_grid_ft_pipeline"
LOGS_DIR="$ROOT/logs/initial"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Qwen FT checkpoints
QWEN_VARIED_CKPT="${QWEN_VARIED_CKPT:-$CKPT_DIR/qwen_sudoku_grid_ft/diverse10k/best_model.pt}"
QWEN_VARIEDPLUS_CKPT="${QWEN_VARIEDPLUS_CKPT:-$CKPT_DIR/qwen_sudoku_grid_ft/diverse_then_messy10k/best_model.pt}"
QWEN_BASE="$CKPT_DIR/Qwen3-1.7B"

# TRM
TRM_CHECKPOINT_DIR="${TRM_CHECKPOINT_DIR:-$CKPT_DIR/trm-sudoku}"

# Test sets
VARIED_TEST="$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json"
VARIEDPLUS_TEST="$ROOT/data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_messy_test.json"
OOD_TEST="$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json"

# ---------- Helper: run one test set ----------
run_eval_set() {
    local LABEL="$1"          # e.g. "varied", "variedplus", "ood"
    local QWEN_CKPT="$2"      # path to best_model.pt
    local INPUT_JSON="$3"     # test set JSON

    echo ""
    echo "=== Evaluating: $LABEL ==="

    local OUT_DIR="$RESULTS_DIR/$LABEL"
    mkdir -p "$OUT_DIR"

    local BRIDGE_OUT="$OUT_DIR/bridge_predictions.json"
    local TRM_INPUT="$OUT_DIR/trm_input.json"
    local TRM_PREDS="$OUT_DIR/trm_predictions.json"
    local METRICS="$OUT_DIR/pipeline_metrics.json"

    # Step 1: Qwen FT grid eval -> bridge predictions + TRM input
    echo "Step 1: Qwen FT grid eval ($LABEL)"
    python "$ROOT/code/initial/integration/qwen_sudoku_grid_finetune.py" eval \
        --base-model "$QWEN_BASE" \
        --checkpoint "$QWEN_CKPT" \
        --input-json "$INPUT_JSON" \
        --bridge-output-json "$BRIDGE_OUT" \
        --trm-input-json "$TRM_INPUT" \
        --metrics-json "$OUT_DIR/grid_metrics.json" \
        --max-samples 1000 \
        --eval-batch-size 4 \
        --max-length 768 \
        --max-new-tokens 96 \
        --llm-dtype auto \
        --wandb \
        --wandb-project trm-llm-test-pipeline \
        --wandb-run-name "qwen-grid-ft-${LABEL}-$SLURM_JOB_ID"

    # Step 2: TRM inference on bridge output
    echo "Step 2: TRM inference ($LABEL)"
    conda run -n trm-env python "$ROOT/code/initial/run_trm_checkpoint_inference.py" \
        --checkpoint-dir "$TRM_CHECKPOINT_DIR" \
        --test-json "$TRM_INPUT" \
        --output-json "$TRM_PREDS" \
        --trm-root "$ROOT/code/initial/TinyRecursiveModels" \
        --require-test-path ""

    # Step 3: Finalize end-to-end metrics
    echo "Step 3: Finalize pipeline metrics ($LABEL)"
    python "$ROOT/code/initial/integration/test_pipeline_qwen_bridge_trm.py" \
        --stage finalize \
        --bridge-output-json "$BRIDGE_OUT" \
        --trm-predictions-json "$TRM_PREDS" \
        --metrics-json "$METRICS" \
        --wandb \
        --wandb-project trm-llm-test-pipeline \
        --wandb-run-name "qwen-grid-ft-${LABEL}-e2e-$SLURM_JOB_ID"

    echo "Done: $LABEL -> $METRICS"
}

# ---------- Run all three test sets ----------
# Varied test set uses the Varied-trained checkpoint
run_eval_set "varied"      "$QWEN_VARIED_CKPT"      "$VARIED_TEST"

# Varied+ test set uses the Varied+-trained checkpoint (continued from Varied)
run_eval_set "variedplus"  "$QWEN_VARIEDPLUS_CKPT"   "$VARIEDPLUS_TEST"

# OOD (LLM-generated corrected2) uses the Varied+-trained checkpoint
run_eval_set "ood"         "$QWEN_VARIEDPLUS_CKPT"   "$OOD_TEST"

# Varied FT checkpoint on Varied+ and OOD (cross-distribution evaluation)
run_eval_set "varied_on_variedplus"  "$QWEN_VARIED_CKPT"   "$VARIEDPLUS_TEST"
run_eval_set "varied_on_ood"         "$QWEN_VARIED_CKPT"   "$OOD_TEST"

echo ""
echo "=== All evaluations complete ==="
echo "Results in: $RESULTS_DIR"
echo "Job finished at $(date)"
