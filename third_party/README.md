# Third-party dependencies

Upstream projects this repository builds on. Download and setup instructions are in
[`ARTIFACTS.md`](../ARTIFACTS.md).

## TinyRecursiveModels (TRM)
- **Repo:** https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **When needed:** retraining the Sudoku TRM specialist from scratch.
- **Already included:** frozen checkpoint + loader in `checkpoints/initial/trm-sudoku/`.

Clone into `code/initial/TinyRecursiveModels` before running `scripts/initial/setup_env.sh`
or `train_trm_sudoku.sbatch`.

## ReNAR / DeepMind CLRS
- **ReNAR source:** vendored at `code/clrs_native_text/external/ReNAR/` (CLRS preliminary study).
- **CLRS benchmark:** https://github.com/google-deepmind/clrs — install via `dm-clrs` (`requirements.txt`).

## Qwen3-1.7B
- **Model:** https://huggingface.co/Qwen/Qwen3-1.7B
- **Required** for any LLM→bridge pipeline. Download to `checkpoints/initial/Qwen3-1.7B/`
  (see `ARTIFACTS.md` § “What you must download”).
