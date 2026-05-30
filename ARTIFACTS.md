# Setup, downloads, and artifacts

This repository is the **main codebase** for the thesis *TRM-LLM: Bridging Language
Inputs to Tiny Recursive Reasoning Models*. It contains the report (`paper/`), the
experiment code, the datasets and checkpoints that fit in git, and the saved result
files behind every table in `paper/main.tex`.

**Do not read the old one-liner as “clone and run everything.”** A lot was left out
deliberately for size. What you need depends on what you are trying to do:

| Goal | What you need beyond `git clone` |
|------|----------------------------------|
| **Inspect reported numbers** (read JSONs, open checkpoints, trace code → table) | **Nothing.** `results/` + bundled checkpoints + `data/` are enough. |
| **Re-run the main bridge→TRM Sudoku pipeline** (headline 86.6% row) | **Download Qwen3-1.7B** (~3.4 GB). Bridge + TRM weights are included. |
| **Re-run every baseline row in the report** | Qwen3-1.7B **plus** several **excluded fine-tuned Qwen checkpoints** (see below). |
| **Retrain TRM or bridges from scratch** | Qwen3-1.7B, **TinyRecursiveModels**, large Sudoku/CLRS training corpora, GPU time — mostly **not** in this repo. |

The repo is ~1 GB on disk. The full working project on HPC was much larger (~12 GB
`data/`, ~7 GB `checkpoints/` before pruning). See
[Included in git](#included-in-git) and [Excluded for size](#excluded-for-size-not-in-git).

---

## Quick start

```bash
git clone <this-repo>
cd trm-llm-bridge

# 1. Python environment (local machine — see § Environment)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio   # pick a build matching your CUDA/CPU
pip install -r requirements.txt

# 2. Required download — frozen LLM front-end (~3.4 GB)
huggingface-cli download Qwen/Qwen3-1.7B \
  --local-dir checkpoints/initial/Qwen3-1.7B

# 3. Inspect a saved result (no download, no GPU)
python -c "import json; print(json.load(open('results/test_pipeline_transnar_messy/messy/pipeline_metrics.json')))"

# 4. Re-run the headline Sudoku pipeline (GPU + Qwen download from step 2)
python code/initial/integration/test_pipeline_qwen_bridge_trm.py \
  --bridge-checkpoint checkpoints/initial/translator_bridge_transnar_messy/last_transnar_trm.pt \
  --output-dir results/test_pipeline_transnar_messy/messy \
  --nl-json data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_messy_test.json
```

On **ITU HPC**, use conda instead of a venv and submit Slurm jobs — see
[Environment](#environment) and [Running on HPC](#running-on-hpc).

---

## Included in git

These paths are committed and sufficient to **verify every number in `paper/main.tex`**
without downloading anything.

| Path | Contents | Approx size |
|------|----------|-------------|
| `paper/` | Report source + compiled PDF | ~1 MB |
| `results/` | **Full per-run outputs** for every Sudoku table row: per-puzzle predictions *and* aggregate metrics (see below) | ~46 MB |
| `data/initial/sudoku_synthetic/` | Varied & Varied+ 10K sets, test sets, LLM-OOD set, n=25 subsets | ~28 MB |
| `data/initial/sudoku_grid/` | Raw Sudoku-Extreme puzzle JSON for generators | ~3 MB |
| `checkpoints/initial/trm-sudoku/` | Frozen TRM specialist used in all pipeline rows | ~19 MB |
| `checkpoints/initial/translator_bridge*/` | **Best/last only** for linear, MLP, Q-Former, TransNAR | ~300 MB |
| `clrs-native-text/` | CLRS Table 6: gold, predictions, routed scores, 7 specialists, bridge `last.pt` | ~650 MB |
| `code/`, `scripts/` | All runnable code for reported experiments | ~2 MB |

**Naming reminder:** report **Varied** = directory `diverse`; report **Varied+** = `messy`.

### What is inside `results/` (not just metrics)

For each bridge pipeline run (26 conditions × 3 test splits), you get **all four** files:

| File | Contents |
|------|----------|
| `bridge_predictions.json` | **1000 rows** — gold puzzle/solution, NL text, bridge-predicted grid per puzzle |
| `trm_predictions.json` | **1000 rows** — TRM solved grid per puzzle |
| `trm_input.json` | Grid strings passed to TRM |
| `pipeline_metrics.json` | Aggregated exact/cell accuracy (the numbers in the tables) |

Example path for the headline row:  
`results/test_pipeline_transnar_messy/messy/bridge_predictions.json`

Baselines likewise include per-puzzle files where applicable (`predictions.json` for Qwen zero-shot/FT, `openai_predictions_*.json` for GPT-5.4 CI, etc.).

CLRS Table 6 uses **`clrs-native-text/bridge_pred_native_<algo>_level4_test.jsonl`** — full per-example native tensor predictions (200 rows per algorithm), not just the summary in `bridge_to_renar_*.json`.

The only result JSONs **not** in this repo are six OpenAI-on-CLRS probe files under `results/clrs/` in the HPC project — those are not cited in `main.tex`.

---

## Excluded for size (not in git)

These existed in the full HPC project but were **not** bundled here. They matter if you
want to **re-run baselines** or **retrain**, not if you only want to **inspect** saved
results.

| Excluded | Approx size | Needed to inspect report? | Needed to re-run what? |
|----------|-------------|---------------------------|------------------------|
| **`Qwen3-1.7B` base weights** | ~3.4 GB | No | Any script that loads the frozen LLM (bridge eval, bridge training, Qwen baselines) |
| **`qwen_sudoku_grid_ft/`** | ~3–7 GB | No (grid-FT numbers are in `results/qwen_grid_ft_pipeline/`) | Re-run “Qwen FT (grid)” rows in Tables 1–3 |
| **`qwen_sudoku_solver_ft/`** | ~3.8 GB | No (solve-direct FT is 0.1%; results in `results/qwen_ft_*`) | Re-run direct-solve Qwen FT baseline |
| **Per-epoch bridge checkpoints** (`*_epoch*.pt`, ~180 files) | ~0.5–1 GB each run | No | Training-curve analysis; best/last are included |
| **Sudoku-Extreme augmented TRM training set** (`data/benchmarks/…`) | ~12 GB | No | Retrain TRM from scratch |
| **`TinyRecursiveModels` upstream repo** | small clone | No | Retrain TRM; frozen checkpoint + loader are included |
| **Full CLRS curriculum / prepared splits** (`data/clrs_native_text/prepared/…`) | tens of MB – GB | No for Table 6 | Extend CLRS study beyond the 7 reported tasks |
| **LLM hidden-state caches** (bridge training speed-up) | varies | No | Faster bridge retraining (can regenerate from Qwen + data) |
| **API keys** (OpenAI, Together, NVIDIA) | — | No (frontier/CI results saved under `results/`) | Re-run GPT-5.4 / DeepSeek frontier baselines |

---

## What you must download

### 1. Qwen3-1.7B (required to run any LLM→bridge pipeline)

All bridge evaluation and training scripts expect the base model at:

```text
checkpoints/initial/Qwen3-1.7B/
```

| | |
|---|---|
| **Source** | [huggingface.co/Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0) |
| **Size** | ~3.4 GB |
| **Why not in git** | Too large; standard Hugging Face distribution |

**Local download:**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-1.7B \
  --local-dir checkpoints/initial/Qwen3-1.7B
```

**HPC download** (CPU partition, no GPU needed):

```bash
sbatch scripts/initial/download_qwen3.sbatch
```

After this, you can train bridges and re-run the **main** Qwen→bridge→TRM pipeline.
You still **cannot** re-run Qwen fine-tuning baselines without the excluded FT
checkpoints listed above (or retraining them).

---

## Optional downloads (retrain / extend, not inspect)

| Item | When you need it | How to obtain |
|------|------------------|---------------|
| **TinyRecursiveModels** | Retrain TRM specialist | `git clone` → `code/initial/TinyRecursiveModels`; then `train_trm_sudoku.sbatch` |
| **Sudoku-Extreme augmented data** | TRM training | Built by TRM's `build_sudoku_dataset.py` or synced from HPC `data/benchmarks/` |
| **`qwen_sudoku_grid_ft/`** | Re-run grid-parser FT baseline | Train: `train_qwen_sudoku_grid_finetune_diverse_then_messy.sbatch`, or rsync from HPC |
| **`qwen_sudoku_solver_ft/`** | Re-run direct-solve FT baseline | Train: `train_qwen_sudoku_finetune.sbatch`, or rsync from HPC |
| **Per-epoch bridge checkpoints** | Training-curve analysis | `rsync` from HPC (see [Syncing with HPC](#syncing-with-hpc)) |
| **Full CLRS curriculum data** | Beyond Table 6 preliminary study | Build via `code/clrs_native_text/` scripts |
| **API keys** | Re-run frontier / code-interpreter calls | `.env` or environment variables |

The frozen TRM checkpoint is **included** — TinyRecursiveModels is only needed to
**retrain** the specialist, not to run the reported pipeline with the existing one.

---

## Environment

Scripts resolve paths from the **repository root** (e.g. `checkpoints/initial/…`,
`data/initial/…`). Run commands from the repo root unless a script says otherwise.

### Local (macOS / Linux)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel setuptools

# PyTorch — install a build for your platform first, then:
pip install -r requirements.txt
```

For the CLRS study you also need `dm-clrs` (pulled in via `requirements.txt`).

### ITU HPC (conda)

Inside a GPU or build allocation:

```bash
srun --partition=acltr --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash
bash scripts/initial/setup_env.sh
```

This creates two conda environments:

| Env | Purpose |
|-----|---------|
| `trm-env` | TRM training (needs `code/initial/TinyRecursiveModels` cloned first) |
| `llm-env` | Qwen3-1.7B, bridge training/eval, API baselines |

Activate before running jobs:

```bash
conda activate llm-env    # bridges, pipeline eval, Qwen FT
conda activate trm-env    # TRM specialist training
```

Optional: `wandb login` for experiment logging (`wandb/` is gitignored).

---

## Running experiments

### Inspect saved results (no GPU)

All thesis Sudoku numbers are in `results/`. Example — headline TransNAR Varied+ pipeline:

```bash
cat results/test_pipeline_transnar_messy/messy/pipeline_metrics.json
```

Appendix tables aggregate the same directories documented in `README.md` (§ “Where each
reported result comes from”).

### Sudoku pipeline eval (GPU + Qwen3-1.7B)

Direct Python (local or interactive GPU session):

```bash
python code/initial/integration/test_pipeline_qwen_bridge_trm.py \
  --bridge-checkpoint checkpoints/initial/translator_bridge_transnar_messy/last_transnar_trm.pt \
  --output-dir results/test_pipeline_transnar_messy/messy \
  --nl-json data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_messy_test.json
```

Slurm wrappers for every bridge × train-set combination:

```bash
sbatch scripts/initial/translators/run_test_pipeline_transnar_messy.sbatch
sbatch scripts/initial/translators/run_test_pipeline_qformer_messy.sbatch
sbatch scripts/initial/translators/run_test_pipeline_linear_messy.sbatch
# … see scripts/initial/translators/
```

### Train a new bridge (GPU + Qwen3-1.7B)

```bash
python code/initial/integration/train_translator_bridge.py \
  --arch transnar \
  --dataset-path data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json \
  --checkpoint-dir checkpoints/initial/translator_bridge_transnar_messy
```

Or: `sbatch scripts/initial/translators/train_bridge.sbatch transnar checkpoints/initial/translator_bridge_transnar_messy`

### Standalone TRM on native grids (GPU)

```bash
python code/initial/run_trm_checkpoint_inference.py \
  --checkpoint checkpoints/initial/trm-sudoku/step_65100
```

### CLRS preliminary study (Table 6)

Re-derive bridge input exact-match from saved artifacts (CPU):

```bash
python code/clrs_native_text/calculate_clrs_bridge_accuracy.py \
  --predicted-jsonl clrs-native-text/bridge_pred_native_insertion_sort_level4_test.jsonl \
  --curriculum-jsonl clrs-native-text/insertion_sort_level4_test.jsonl
```

End-to-end bridge→ReNAR scores are in `clrs-native-text/bridge_to_renar_*_level4_test.json`.
Full re-run loop: `scripts/clrs_native_text/eval_bridge_to_renar_loop.sh` (needs Qwen
weights, `clrs-native-text/last.pt`, and the per-task `single_<algo>_4_best_MLM.pkl` files).

### Regenerate synthetic Sudoku data (CPU)

```bash
# Varied (report name) / diverse (filename)
python code/initial/synthesize/rule/transform_sudoku_raw_to_diverse_nl.py \
  --input data/initial/sudoku_grid/sudoku_raw_1000_v3.json \
  --output data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json

# Varied+ / messy
python code/initial/synthesize/rule/transform_sudoku_raw_to_messy_nl.py \
  --input data/initial/sudoku_grid/sudoku_raw_1000_v3.json \
  --output data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json
```

---

## Running on HPC

Typical workflow on ITU HPC (`hpc3.itu.dk`, home directory `~/thesis-trm-llm` or a clone
of this repo):

```bash
# Clone / sync repo
git clone <this-repo> ~/thesis-trm-llm && cd ~/thesis-trm-llm

# One-time env
bash scripts/initial/setup_env.sh

# One-time model download
sbatch scripts/initial/download_qwen3.sbatch

# Submit an experiment
sbatch scripts/initial/translators/run_test_pipeline_transnar_messy.sbatch

# Monitor
squeue -u $USER
tail -f logs/initial/test_pipe_transnar_messy_*.out
```

Partition and GPU constraints are set in each `.sbatch` file (`acltr` partition for
bridge/pipeline jobs).

---

## Syncing with HPC

Use `rsync` to move checkpoints or data between HPC and a local machine.

**Pull from HPC → local** (example: extra bridge checkpoint history):

```bash
rsync -avz albst@hpc3.itu.dk:~/thesis-trm-llm/checkpoints/initial/translator_bridge/ \
  ./checkpoints/initial/translator_bridge/
```

**Push local → HPC:**

```bash
rsync -avz --exclude ".git" --exclude ".venv*" --exclude "wandb" \
  ./ albst@hpc3.itu.dk:~/thesis-trm-llm/
```

Large pulls (full `checkpoints/` or `data/`) can take several minutes.

---

## Checkpoint → thesis result mapping

Each row in the Sudoku tables corresponds to a specific trained bridge file:

| Checkpoint | Report usage |
|------------|----------------|
| `checkpoints/initial/trm-sudoku/step_65100` | Frozen TRM specialist — RQ2 ceiling (86.5% exact) |
| `…/linear-10k-diverse-fixed59/best_linear.pt` | Linear probe, Varied training |
| `…/linear-10k-stage2-messy-from-diverse/last_linear.pt` | Linear probe, Varied+ training |
| `…/mlp-10k-diverse-fixed59/best_mlp.pt` | MLP projector, Varied |
| `…/mlp-10k-stage2-messy-from-diverse/last_mlp.pt` | MLP projector, Varied+ |
| `…/translator_bridge/10K/best_qformer.pt` | Q-Former, Varied |
| `…/translator_bridge/10K-finetuned/last_qformer.pt` | Q-Former, Varied+ |
| `…/translator_bridge_transnar_diverse/best_transnar_trm.pt` | TransNAR, Varied |
| `…/translator_bridge_transnar_messy/last_transnar_trm.pt` | **TransNAR, Varied+ — 86.6% end-to-end** |

CLRS Table 6 (preliminary study):

| Artifact | Role |
|----------|------|
| `clrs-native-text/<algo>_level4_test.jsonl` | Gold native inputs (200 examples per algorithm) |
| `clrs-native-text/bridge_pred_native_<algo>_level4_test.jsonl` | Bridge-predicted inputs |
| `clrs-native-text/bridge_to_renar_<algo>_level4_test.json` | Routed end-to-end score |
| `clrs-native-text/single_<algo>_4_best_MLM.pkl` | Frozen M-ReNAR specialist (seed 4) |
| `clrs-native-text/last.pt` | Shared TransNAR CLRS bridge checkpoint |

Algorithms: `bfs`, `dfs`, `bellman_ford`, `dijkstra`, `dag_shortest_paths`,
`binary_search`, `insertion_sort`.

---

## Gitignored paths

These are expected on disk but not committed (see `.gitignore`):

| Path | Reason |
|------|--------|
| `checkpoints/initial/Qwen3-1.7B/` | Base LLM (~3.4 GB) — download separately |
| `checkpoints/initial/qwen_sudoku_grid_ft/` | Grid-parser FT checkpoints (~GB scale) |
| `checkpoints/initial/qwen_sudoku_solver_ft/` | Direct-solve FT checkpoints (~3.8 GB) |
| `.venv/`, `.venv-*/` | Local Python environments |
| `wandb/`, `logs/` | Run logs |
| `paper/*.aux`, `paper/*.log`, … | LaTeX build artifacts (PDF is committed) |

