#!/bin/bash
# =============================================================================
# train_qwen_solve_direct.sbatch
#
# Fine-tunes Qwen3-1.7B to solve Sudoku end-to-end (NL -> solved grid).
# This is the direct-solve baseline: no TRM, no bridge.
# Target is the 81-digit solution string, not the puzzle grid.
#
# Compare results against:
#   - TransNAR bridge + TRM pipeline (86.6% exact solve)
#   - Qwen FT grid + TRM pipeline (this script's grid-parse equivalent)
#
# Usage:
#   sbatch scripts/initial/train_qwen_solve_direct.sbatch
# =============================================================================
#SBATCH --job-name=qwen_solve_direct
#SBATCH --partition=acltr
#SBATCH --nodelist=cn7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/initial/qwen_solve_direct_%j.out
#SBATCH --error=logs/initial/qwen_solve_direct_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
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
LOGS_DIR="$ROOT/logs/initial"
mkdir -p "$LOGS_DIR"

# Phase 1: train on Varied dataset
echo "=== Phase 1/2: Varied direct-solve fine-tuning ==="
python "$ROOT/code/initial/integration/qwen_sudoku_finetune.py" train \
    --base-model "$CKPT_DIR/Qwen3-1.7B" \
    --dataset "$ROOT/data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json" \
    --checkpoint-dir "$CKPT_DIR/qwen_sudoku_solve_ft/varied10k" \
    --log-dir "$LOGS_DIR" \
    --epochs 10 \
    --batch-size 2 \
    --eval-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --max-length 768 \
    --max-new-tokens 96 \
    --eval-split 0.1 \
    --seed 42 \
    --save-every-epoch 1 \
    --patience 4 \
    --llm-dtype auto \
    --wandb \
    --wandb-project trm-llm-qwen-ft \
    --wandb-run-name "qwen-solve-varied10k-$SLURM_JOB_ID"

VARIED_BEST="$CKPT_DIR/qwen_sudoku_solve_ft/varied10k/best_model.pt"

# Phase 2: continue on Varied+ dataset
echo "=== Phase 2/2: Varied+ direct-solve fine-tuning (continued) ==="
python "$ROOT/code/initial/integration/qwen_sudoku_finetune.py" train \
    --base-model "$CKPT_DIR/Qwen3-1.7B" \
    --dataset "$ROOT/data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json" \
    --checkpoint-dir "$CKPT_DIR/qwen_sudoku_solve_ft/variedplus10k" \
    --log-dir "$LOGS_DIR" \
    --epochs 5 \
    --batch-size 2 \
    --eval-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --max-length 768 \
    --max-new-tokens 96 \
    --eval-split 0.1 \
    --seed 42 \
    --save-every-epoch 1 \
    --patience 3 \
    --llm-dtype auto \
    --resume-from "$VARIED_BEST" \
    --no-auto-resume \
    --wandb \
    --wandb-project trm-llm-qwen-ft \
    --wandb-run-name "qwen-solve-variedplus10k-$SLURM_JOB_ID"

echo "Varied best: $VARIED_BEST"
echo "Varied+ best: $CKPT_DIR/qwen_sudoku_solve_ft/variedplus10k/best_model.pt"
echo "Job finished at $(date)"
