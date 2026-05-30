#!/usr/bin/env bash
set -a
source .env
set +a

echo "=== Phase 1: OOD (LLM-generated corrected2) ==="
.venv-sudoku-frontier/bin/python3 code/initial/integration/benchmark_nvidia_frontier_sudoku.py \
  --task solve \
  --models "deepseek-ai/DeepSeek-V4-Pro" \
  --input-json "data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json" \
  --output-json "results/frontier_sudoku_builder/frontier_solve_together_v4pro_ood25.json" \
  --metrics-json "results/frontier_sudoku_builder/frontier_solve_together_v4pro_metrics_ood25.json" \
  --api-key "${TOGETHER_API_KEY}" \
  --base-url "https://api.together.xyz/v1" \
  --max-samples 25 \
  --max-tokens 1024 \
  --concurrency 2 \
  --calls-per-minute 10 \
  --save-every 1 \
  --resume \
  --retry-errors \
  --extra-body-json '{"reasoning":{"enabled":false}}' \
  --wandb \
  --wandb-project "trm-llm-test-pipeline" \
  --wandb-run-name "frontier-solve-together-v4pro-ood-25"

echo "=== Phase 2: Varied+ (messy rule-based test set) ==="
.venv-sudoku-frontier/bin/python3 code/initial/integration/benchmark_nvidia_frontier_sudoku.py \
  --task solve \
  --models "deepseek-ai/DeepSeek-V4-Pro" \
  --input-json "data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_messy_test.json" \
  --output-json "results/frontier_sudoku_builder/frontier_solve_together_v4pro_variedplus25.json" \
  --metrics-json "results/frontier_sudoku_builder/frontier_solve_together_v4pro_metrics_variedplus25.json" \
  --api-key "${TOGETHER_API_KEY}" \
  --base-url "https://api.together.xyz/v1" \
  --max-samples 25 \
  --max-tokens 1024 \
  --concurrency 2 \
  --calls-per-minute 10 \
  --save-every 1 \
  --resume \
  --extra-body-json '{"reasoning":{"enabled":false}}' \
  --wandb \
  --wandb-project "trm-llm-test-pipeline" \
  --wandb-run-name "frontier-solve-together-v4pro-variedplus-25"

echo "=== Done ==="
