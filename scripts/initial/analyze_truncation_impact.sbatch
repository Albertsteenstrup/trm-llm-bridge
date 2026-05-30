#!/bin/bash
#SBATCH --job-name=truncation_analysis
#SBATCH --output=logs/initial/truncation_analysis_%j.out
#SBATCH --error=logs/initial/truncation_analysis_%j.err
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_a30|gpu_a100_40gb|gpu_a100_80gb|gpu_l40s|gpu_h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Exit on error
set -euo pipefail

echo "DEBUG: Script started"
echo "DEBUG: PWD=$PWD"
echo "DEBUG: BASH_SOURCE[0]=${BASH_SOURCE[0]}"

# Hardcode ROOT to avoid path resolution issues
ROOT="/home/albst/thesis-trm-llm"

echo "DEBUG: ROOT=$ROOT"

mkdir -p "$ROOT/logs/initial"

echo "DEBUG: logs/initial directory created"

echo "Starting Truncation Impact Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load modules and activate conda env
module purge
module use /opt/itu/easybuild/modules/all
module load Anaconda3

# Temporarily disable unbound variable check for conda activation
set +u
eval "$(conda shell.bash hook)"
conda activate llm-env
set -u

echo "Conda env active: ${CONDA_DEFAULT_ENV:-unknown}"
python --version

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# Ensure we are in the correct directory
cd "$ROOT"

# Set all cache/log directories explicitly to avoid system paths
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"

echo "Environment variables set:"
echo "  TMPDIR=$TMPDIR"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

# Create necessary directories
mkdir -p "$ROOT/logs/initial"

# Set checkpoint and dataset paths
CHECKPOINT="${CHECKPOINT:-checkpoints/initial/translator_bridge/10K-finetuned/last_qformer.pt}"
DATASET="${DATASET:-data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json}"
OUTPUT="${OUTPUT:-logs/initial/truncation_impact_analysis.json}"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT"
echo ""

# Run the analysis script
echo "Running analyze_truncation_impact.py..."
python code/initial/analyze_truncation_impact.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --max-seq-len 512 \
    --device cuda 2>&1

echo ""
echo "Job completed successfully at $(date)"
