# TRM-LLM: Bridging Language Inputs to Tiny Recursive Reasoning Models

Thesis repository for **TRM-LLM: Bridging Language Inputs to Tiny Recursive Reasoning
Models** (Albert Steenstrup). The report is in [`paper/main.pdf`](paper/main.pdf);
source in [`paper/main.tex`](paper/main.tex).

This repo holds the experiment code, datasets, saved results, and trained checkpoints
behind every table in the report. To set up a machine, download external weights, and
run pipelines, see **[`ARTIFACTS.md`](ARTIFACTS.md)**.

> Scripts resolve paths from the repository root (`checkpoints/initial/…`,
> `data/initial/…`). Run commands from the repo root.

---

## Naming key (important)

The report and the filenames use different words for the two rule-based datasets:

| Report term | Filename / directory term | Dataset file |
|-------------|---------------------------|--------------|
| **Varied**  | `diverse`                 | `data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json` |
| **Varied+** | `messy`                   | `data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json` |
| **LLM (OOD)** | `ood` / `ood_llm`       | LLM-generated held-out set (see `code/initial/synthesize/llm/`) |

---

## The pipeline

```
Natural-language Sudoku  →  frozen Qwen3-1.7B  →  trainable bridge  →  frozen TRM  →  solved grid
                                (hidden states)     (81×10 grid logits)   (5M specialist)
```

The contribution is the **trainable bridge**. Four bridge architectures are compared:
linear probe, MLP projector, Q-Former, and TransNAR. A preliminary study (CLRS) checks
whether the same bridge idea connects to a different frozen specialist family (M-ReNAR).

---

## Where each reported result comes from

### RQ1 — Bridge text-to-grid accuracy (Tables 1–2; App. A Table for full breakdown)
- **Bridge models / training:** `code/initial/integration/train_translator_bridge.py`
  (defines all four architectures via `ARCH_REGISTRY`).
- **Evaluation harness:** `code/initial/integration/test_pipeline_qwen_bridge_trm.py`.
- **Run scripts:** `scripts/initial/translators/run_test_pipeline_*_{diverse,messy}.sbatch`.
- **Saved outputs:** `results/test_pipeline_{linear,mlp,qformer,transnar}_{diverse,messy}/`
  (each with `diverse/`, `messy/`, `ood_llm/` subdirs containing `bridge_predictions.json`
  and `pipeline_metrics.json`).
- **Trained bridge weights:** `checkpoints/initial/translator_bridge/` (linear, MLP, Q-Former)
  and `checkpoints/initial/translator_bridge_transnar_{diverse,messy}/` (TransNAR). See
  [`ARTIFACTS.md`](ARTIFACTS.md) for the exact checkpoint-to-row mapping.
- **DeepSeek-V4-Pro zero-shot rows:** `code/initial/integration/benchmark_nvidia_frontier_sudoku.py`
  → `results/frontier_sudoku_builder/`.
- **Qwen3-1.7B FT (grid) rows:** `code/initial/integration/qwen_sudoku_grid_finetune.py`
  → `results/qwen_grid_ft_pipeline/`.

### RQ2 — Standalone frozen TRM ceiling (86.5% exact)
- **Code:** `code/initial/run_trm_checkpoint_inference.py`.
- **Run scripts:** `scripts/initial/run_trm_on_{diverse,messy,ood}_gt.sbatch`.
- **Checkpoint:** `checkpoints/initial/trm-sudoku/step_65100`.
- **Saved outputs:** `results/raw_baselines/trm_predictions_round1.json`,
  `results/raw_baselines/baselines_heldout_resume.json`.

### RQ3 — End-to-end exact solve accuracy (Table 3)
- **Pipeline metrics:** the `pipeline_metrics.json` / `trm_predictions.json` files inside
  each `results/test_pipeline_*` directory.
- **Direct LLM baselines:**
  - Qwen3-1.7B zero-shot → `results/qwen_zeroshot_ood/`.
  - Qwen3-1.7B FT (solve-direct) → `code/initial/integration/qwen_sudoku_finetune.py`,
    `results/qwen_ft_{diverse,messy}_ood/`.
  - GPT-5.4 with code interpreter (§ "Learned specialists vs tool use") →
    `code/initial/integration/benchmark_openai_code_interpreter_sudoku.py`,
    `results/openai_sudoku_code_interpreter/`.

### RQ4 — Adaptation efficiency (Tables 4–5)
- Parameter counts are computed from the model definitions in
  `train_translator_bridge.py` and the checkpoints. No separate result file; the
  per-architecture parameter counts are listed in the report appendix (App. C).

### Analysis (gates, truncation, recovery matrix)
- **Gate dynamics (App. Table):** read from the TransNAR checkpoint
  `checkpoints/initial/translator_bridge_transnar_messy/last_transnar_trm.pt`.
- **Sequence-length / truncation (App.):** `code/initial/analyze_truncation_impact.py`,
  `scripts/initial/analyze_truncation_impact.sbatch`.

### Synthetic data generation & validation (App. on data)
- **Rule-based generators:** `code/initial/synthesize/rule/transform_sudoku_raw_to_diverse_nl.py`
  (Varied) and `…_messy_nl.py` (Varied+).
- **LLM generation / model selection / correction:** `code/initial/synthesize/llm/`.
- **Raw puzzles:** `data/initial/sudoku_grid/`, downloaded via
  `code/initial/download_sudoku_raw.py`.

### Preliminary CLRS study (Table 6; § "Preliminary Study" and App. on CLRS)
- **Bridge + scoring code:** `code/clrs_native_text/bridge_native/`,
  `code/clrs_native_text/calculate_clrs_bridge_accuracy.py`.
- **Frozen specialist source (M-ReNAR):** `code/clrs_native_text/external/ReNAR/`.
- **Run scripts:** `scripts/clrs_native_text/eval_bridge_to_renar_loop.sh`,
  `calculate_clrs_bridge_accuracy.sh`, `eval_bridge_specialist.sh`.
- **Artifacts (`clrs-native-text/`), per algorithm `<algo>` ∈
  {bfs, dfs, bellman_ford, dijkstra, dag_shortest_paths, binary_search, insertion_sort}:**
  - `<algo>_level4_test.jsonl` — gold native inputs (200 length-64 examples each).
  - `bridge_pred_native_<algo>_level4_test.jsonl` — bridge-predicted native inputs.
  - `bridge_to_renar_<algo>_level4_test.json` — the **routed end-to-end score**
    (the "Bridge+routed ReNAR" column of Table 6).
  - `single_<algo>_4_best_MLM.pkl` — the frozen M-ReNAR specialist used for that row.
  - `last.pt` — the shared TransNAR CLRS bridge checkpoint.

---

## Repository layout

```text
.
├── paper/                     # main.tex, figures/, references.bib, compiled main.pdf
├── code/
│   ├── initial/               # Sudoku pipeline (main contribution)
│   │   ├── integration/       # bridge training, end-to-end eval, LLM baselines
│   │   ├── synthesize/        # Varied / Varied+ / LLM-OOD generators
│   │   ├── run_trm_checkpoint_inference.py
│   │   ├── evaluate_sudoku_outputs.py
│   │   └── analyze_truncation_impact.py
│   └── clrs_native_text/       # CLRS preliminary study (bridge + ReNAR source)
├── scripts/
│   ├── initial/               # Slurm/shell runners for the Sudoku experiments
│   └── clrs_native_text/       # runners for the CLRS study
├── data/
│   ├── initial/sudoku_*/      # Varied/Varied+ datasets, test sets, n=25 subsets, raw grids
│   └── clrs_native_text/       # CLRS test splits
├── results/                   # cited result JSONs (Sudoku pipeline + baselines + frontier)
├── clrs-native-text/          # CLRS Table-6 artifacts (gold, preds, scores, specialists, bridge)
├── checkpoints/initial/       # frozen TRM + trained bridges (linear, MLP, Q-Former, TransNAR)
├── requirements.txt
└── ARTIFACTS.md               # setup, downloads, how to run
```

## Quick start

See **[`ARTIFACTS.md`](ARTIFACTS.md)** for the full picture (what is in git vs excluded,
and what to download for inspect vs re-run vs retrain).

- **Inspect report numbers:** open `results/` — no downloads.
- **Re-run main bridge→TRM pipeline:** download Qwen3-1.7B → `checkpoints/initial/Qwen3-1.7B`, then run `test_pipeline_qwen_bridge_trm.py` or the matching `.sbatch`.
- **Re-run all baselines / retrain:** also needs excluded FT checkpoints and/or large training data — see `ARTIFACTS.md`.
