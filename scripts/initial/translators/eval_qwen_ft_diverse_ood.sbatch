#!/bin/bash
# =============================================================================
# eval_qwen_ft_diverse_ood.sbatch
#
# Evaluates Qwen3-1.7B fine-tuned on diverse 10K on the OOD LLM test set.
# Checkpoint: checkpoints/initial/qwen_sudoku_solver_ft/diverse10k_wandb_e10_stats/best_model.pt
#
# Usage:
#   sbatch scripts/initial/translators/eval_qwen_ft_diverse_ood.sbatch
# =============================================================================

#SBATCH --job-name=eval_qwen_ft_div_ood
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_a100_80gb|gpu_a30'
#SBATCH --nodelist=cn7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/initial/eval_qwen_ft_diverse_ood_%j.out
#SBATCH --error=logs/initial/eval_qwen_ft_diverse_ood_%j.err

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
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/initial/integration/qwen_sudoku_finetune.py" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$ROOT"

SCRIPT="$ROOT/code/initial/integration/qwen_sudoku_finetune.py"
RESULTS_DIR="$ROOT/results/qwen_ft_diverse_ood"
mkdir -p "$RESULTS_DIR"

python "$SCRIPT" eval \
    --base-model "$ROOT/checkpoints/initial/Qwen3-1.7B" \
    --checkpoint "$ROOT/checkpoints/initial/qwen_sudoku_solver_ft/diverse10k_wandb_e10_stats/best_model.pt" \
    --input-json "$ROOT/data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json" \
    --predictions-json "$RESULTS_DIR/predictions.json" \
    --metrics-json "$RESULTS_DIR/metrics.json" \
    --max-samples 1000 \
    --wandb \
    --wandb-project "trm-llm-test-pipeline" \
    --wandb-run-name "qwen_ft_diverse_ood_llm"

echo "Done. Metrics: $RESULTS_DIR/metrics.json"
echo "Job finished at $(date)"
