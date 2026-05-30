#!/bin/bash
# =============================================================================
# run_baselines_heldout.sbatch — Compare Qwen3-1.7B vs TRM on held-out Sudoku test set
#
# Required:
#   - Held-out test JSON with fields: puzzle, solution
#   - Qwen3-1.7B checkpoint directory
#   - TRM predictions JSON (or command that creates it)
#
# Usage:
#   sbatch scripts/initial/run_baselines_heldout.sbatch
#
# Optional exports:
#   TEST_JSON=$ROOT/data/initial/sudoku_grid/sudoku_raw_1000_v3.json
#   TRM_PREDICTIONS_JSON=$ROOT/results/raw_baselines/trm_predictions_round1.json
#   TRM_CHECKPOINT_DIR=$ROOT/checkpoints/initial/trm-sudoku
#   TRM_CHECKPOINT_FILE=$ROOT/checkpoints/initial/trm-sudoku/step_65100
#   TRM_EVAL_CMD='python code/initial/run_trm_checkpoint_inference.py --checkpoint-dir ... --checkpoint-file ... --test-json ... --output-json ...'
#   MAX_PUZZLES=200
# =============================================================================
#SBATCH --job-name=sudoku_baselines
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/initial/sudoku_baselines_%j.out

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)"
echo "Detected GPU memory: ${GPU_MEM_MB} MB"
if [ -z "${GPU_NAME}" ] || [ "${GPU_NAME}" = "N/A" ]; then
    echo "ERROR: No GPU detected for this job allocation."
    exit 1
fi

# ---------- Modules & Environment ----------
unset -f module 2>/dev/null || true

if ! command -v module >/dev/null 2>&1; then
    set +u
    [ -f /etc/profile ] && source /etc/profile || true
    set -u
fi
if ! command -v module >/dev/null 2>&1; then
    set +u
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
    set -u
fi
if command -v module >/dev/null 2>&1; then
    module purge || true
    module use /opt/itu/easybuild/modules/all || true
    module load Anaconda3 || true
fi

if ! command -v conda >/dev/null 2>&1; then
    set +u
    ANACONDA_CONDA_BIN=""
    for candidate in /opt/itu/easybuild/software/Anaconda3/*/bin/conda; do
        if [ -x "$candidate" ]; then
            ANACONDA_CONDA_BIN="$candidate"
        fi
    done

    if [ -n "$ANACONDA_CONDA_BIN" ]; then
        __conda_setup="$($ANACONDA_CONDA_BIN shell.bash hook 2>/dev/null || true)"
        if [ -n "$__conda_setup" ]; then
            eval "$__conda_setup"
        else
            CONDA_SH="${ANACONDA_CONDA_BIN%/bin/conda}/etc/profile.d/conda.sh"
            if [ -f "$CONDA_SH" ]; then
                . "$CONDA_SH"
            else
                export PATH="${ANACONDA_CONDA_BIN%/bin/conda}/bin:$PATH"
            fi
        fi
        unset __conda_setup
    fi
    set -u
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after module/direct fallback."
    module list || true
    exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate llm-env
set -u

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_JSON="${TEST_JSON:-$ROOT/data/initial/sudoku_grid/sudoku_raw_1000_v3.json}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$ROOT/checkpoints/initial/Qwen3-1.7B}"
RESULTS_DIR="$ROOT/results/raw_baselines"
TRM_PREDICTIONS_JSON="${TRM_PREDICTIONS_JSON:-$RESULTS_DIR/trm_predictions_round1.json}"
TRM_CHECKPOINT_DIR="${TRM_CHECKPOINT_DIR:-$ROOT/checkpoints/initial/trm-sudoku}"
TRM_CHECKPOINT_FILE="${TRM_CHECKPOINT_FILE:-}"
TRM_EVAL_CMD="${TRM_EVAL_CMD:-}"
TRM_EVAL_PYTHON="${TRM_EVAL_PYTHON:-conda run -n trm-env python}"
FORCE_TRM_REGEN="${FORCE_TRM_REGEN:-0}"
MAX_PUZZLES="${MAX_PUZZLES:-0}"
OUT_JSON="${OUT_JSON:-$RESULTS_DIR/baselines_heldout_resume.json}"
TRM_SCORED_JSON="${TRM_SCORED_JSON:-$TRM_PREDICTIONS_JSON}"
SAVE_EVERY="${SAVE_EVERY:-25}"

mkdir -p "$RESULTS_DIR"
echo "Keeping existing files in $RESULTS_DIR for resume support"

echo "TEST_JSON=$TEST_JSON"
echo "QWEN_MODEL_DIR=$QWEN_MODEL_DIR"
echo "TRM_PREDICTIONS_JSON=$TRM_PREDICTIONS_JSON"
echo "TRM_CHECKPOINT_DIR=$TRM_CHECKPOINT_DIR"
echo "TRM_CHECKPOINT_FILE=${TRM_CHECKPOINT_FILE:-<auto-latest>}"
echo "TRM_EVAL_PYTHON=$TRM_EVAL_PYTHON"
echo "FORCE_TRM_REGEN=$FORCE_TRM_REGEN"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "OUT_JSON=$OUT_JSON"
echo "TRM_SCORED_JSON=$TRM_SCORED_JSON"

if [ ! -f "$TEST_JSON" ]; then
    echo "ERROR: held-out test file not found: $TEST_JSON"
    exit 1
fi

if [ ! -d "$QWEN_MODEL_DIR" ]; then
    echo "ERROR: Qwen3-1.7B model directory not found: $QWEN_MODEL_DIR"
    exit 1
fi

if [ "$FORCE_TRM_REGEN" = "1" ] && [ -f "$TRM_PREDICTIONS_JSON" ]; then
    echo "FORCE_TRM_REGEN=1 -> removing existing TRM predictions: $TRM_PREDICTIONS_JSON"
    rm -f "$TRM_PREDICTIONS_JSON"
fi

if [ ! -f "$TRM_PREDICTIONS_JSON" ]; then
    if [ -z "$TRM_EVAL_CMD" ]; then
        TRM_EVAL_CMD="$TRM_EVAL_PYTHON $ROOT/code/initial/run_trm_checkpoint_inference.py --checkpoint-dir '$TRM_CHECKPOINT_DIR' --test-json '$TEST_JSON' --output-json '$TRM_PREDICTIONS_JSON'"
        if [ -n "$TRM_CHECKPOINT_FILE" ]; then
            TRM_EVAL_CMD="$TRM_EVAL_CMD --checkpoint-file '$TRM_CHECKPOINT_FILE'"
        fi
        if [ "$MAX_PUZZLES" -gt 0 ]; then
            TRM_EVAL_CMD="$TRM_EVAL_CMD --max-puzzles '$MAX_PUZZLES'"
        fi
    fi

    echo "TRM predictions missing - running TRM inference command..."
    echo "$TRM_EVAL_CMD"
    bash -lc "$TRM_EVAL_CMD"
fi

if [ ! -f "$TRM_PREDICTIONS_JSON" ]; then
    echo "ERROR: TRM predictions still not found: $TRM_PREDICTIONS_JSON"
    echo "Set TRM_PREDICTIONS_JSON to an existing file, or provide a valid TRM_EVAL_CMD."
    exit 1
fi

export TEST_JSON
export QWEN_MODEL_DIR
export TRM_PREDICTIONS_JSON
export OUT_JSON
export TRM_SCORED_JSON
export MAX_PUZZLES
export SAVE_EVERY

python - <<'PY'
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

use_tqdm = tqdm is not None and sys.stdout.isatty()

test_json = os.environ["TEST_JSON"]
qwen_model_dir = os.environ["QWEN_MODEL_DIR"]  # env var set from shell QWEN_MODEL_DIR (Qwen3-1.7B path)
trm_predictions_json = os.environ["TRM_PREDICTIONS_JSON"]
out_json = os.environ["OUT_JSON"]
trm_scored_json = os.environ["TRM_SCORED_JSON"]
max_puzzles = int(os.environ.get("MAX_PUZZLES", "0"))
qwen_progress_every = max(1, int(os.environ.get("QWEN_PROGRESS_EVERY", "1")))
trm_progress_every = max(1, int(os.environ.get("TRM_PROGRESS_EVERY", "10")))
save_every = max(1, int(os.environ.get("SAVE_EVERY", "25")))
trm_score_save_every = max(1, int(os.environ.get("TRM_SCORE_SAVE_EVERY", "100")))


def extract_solution_candidate(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    contiguous = re.findall(r"([1-9]{81})", text)
    if contiguous:
        return contiguous[-1]

    digits = re.findall(r"[1-9]", text)
    if len(digits) < 81:
        return None
    return "".join(digits[-81:])


def compute_all_accuracy(pred: str, solution: str) -> float:
    if not isinstance(pred, str) or not isinstance(solution, str):
        return 0.0
    if len(pred) != 81 or len(solution) != 81:
        return 0.0
    correct = sum(1 for p, s in zip(pred, solution) if p == s)
    return (correct / 81.0) * 100.0


def compute_trm_running_averages(rows_by_index):
    rows = ordered_rows(rows_by_index)
    if not rows:
        return 0.0, 0.0
    avg_all_accuracy = sum(float(row.get("all_accuracy", 0.0)) for row in rows) / len(rows)
    avg_exact_match = sum(int(row.get("exact_match", 0)) for row in rows) / len(rows)
    return avg_all_accuracy, avg_exact_match


def load_test_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        puzzle = item.get("puzzle")
        solution = item.get("solution")
        if isinstance(puzzle, str) and isinstance(solution, str) and len(puzzle) == 81 and len(solution) == 81:
            out.append({"index": i, "puzzle": puzzle, "solution": solution})
    return out


def load_trm_predictions(path: str, expected_examples: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    metadata_out = {}
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                mapping[i] = item
            elif isinstance(item, dict):
                idx = item.get("index", i)
                pred = (
                    item.get("predicted_solution")
                    or item.get("prediction")
                    or item.get("solution_pred")
                    or item.get("output")
                )
                if isinstance(idx, int):
                    mapping[idx] = pred
    elif isinstance(data, dict):
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            metadata_out = metadata
            metadata_examples = metadata.get("num_examples")
            print(
                "TRM metadata:",
                {
                    "num_examples": metadata_examples,
                    "dominant_extraction_key": metadata.get("dominant_extraction_key"),
                    "global_batch_size": metadata.get("global_batch_size"),
                },
            )
            if isinstance(metadata_examples, int) and metadata_examples != expected_examples:
                raise ValueError(
                    f"TRM predictions metadata.num_examples={metadata_examples} does not match expected {expected_examples}."
                )
        items = data.get("predictions", data)
        if isinstance(items, dict):
            for k, v in items.items():
                try:
                    mapping[int(k)] = v
                except Exception:
                    pass
        elif isinstance(items, list):
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    idx = item.get("index", i)
                    pred = (
                        item.get("predicted_solution")
                        or item.get("prediction")
                        or item.get("solution_pred")
                        or item.get("output")
                    )
                    if isinstance(idx, int):
                        mapping[idx] = pred
    return mapping, metadata_out


records = load_test_records(test_json)
if max_puzzles > 0:
    records = records[:max_puzzles]

if not records:
    raise ValueError("No valid test records found")

print(f"Loaded {len(records)} held-out puzzles")
record_indices = [rec["index"] for rec in records]
record_index_set = set(record_indices)
solution_by_index = {rec["index"]: rec["solution"] for rec in records}

qwen_correct = 0
qwen_valid = 0
qwen_parse_failures = 0
qwen_rows_by_index = {}
qwen_debug_rows = []
trm_correct = 0
trm_valid = 0
trm_rows_by_index = {}
trm_debug_rows = []


def ordered_rows(rows_by_index):
    return [rows_by_index[idx] for idx in record_indices if idx in rows_by_index]


def recompute_metrics(rows_by_index):
    rows = ordered_rows(rows_by_index)
    valid = sum(1 for row in rows if row.get("prediction") is not None)
    exact = sum(1 for row in rows if int(row.get("exact_match", 0)) == 1)
    return valid, exact


def load_resume_rows(snapshot_path: str):
    qwen_existing = {}
    trm_existing = {}

    if not os.path.exists(snapshot_path):
        return qwen_existing, trm_existing

    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"WARNING: could not parse existing snapshot '{snapshot_path}': {exc}", flush=True)
        return qwen_existing, trm_existing

    if not isinstance(data, dict):
        return qwen_existing, trm_existing

    snapshot_test_json = data.get("test_json")
    snapshot_num_examples = data.get("num_examples")
    if (snapshot_test_json and snapshot_test_json != test_json) or (
        isinstance(snapshot_num_examples, int) and snapshot_num_examples != len(records)
    ):
        print(
            "Existing snapshot metadata does not match current run inputs; ignoring old progress.",
            flush=True,
        )
        return qwen_existing, trm_existing

    details = data.get("details", {}) if isinstance(data.get("details", {}), dict) else {}

    for row in details.get("qwen", []):
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if isinstance(idx, int) and idx in record_index_set:
            pred = row.get("prediction")
            solution = solution_by_index[idx]
            row_all_accuracy = row.get("all_accuracy")
            if not isinstance(row_all_accuracy, (int, float)):
                row_all_accuracy = compute_all_accuracy(pred, solution)
            qwen_existing[idx] = {
                "index": idx,
                "prediction": pred,
                "all_accuracy": float(row_all_accuracy),
                "exact_match": 1 if bool(row.get("exact_match")) else 0,
            }

    for row in details.get("trm", []):
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if isinstance(idx, int) and idx in record_index_set:
            pred = row.get("prediction")
            solution = solution_by_index[idx]
            row_all_accuracy = row.get("all_accuracy")
            if not isinstance(row_all_accuracy, (int, float)):
                row_all_accuracy = compute_all_accuracy(pred, solution)
            trm_existing[idx] = {
                "index": idx,
                "prediction": pred,
                "all_accuracy": float(row_all_accuracy),
                "exact_match": 1 if bool(row.get("exact_match")) else 0,
            }

    return qwen_existing, trm_existing


def save_snapshot(stage: str):
    qwen_done = len(qwen_rows_by_index)
    trm_done = len(trm_rows_by_index)
    payload = {
        "test_json": test_json,
        "num_examples": len(records),
        "stage": stage,
        "completed": stage == "done",
        "progress": {
            "qwen_done": qwen_done,
            "trm_done": trm_done,
            "save_every": save_every,
        },
        "qwen": {
            "valid_predictions": qwen_valid,
            "parse_failures": qwen_parse_failures,
            "all_accurate_predictions": sum(1 for row in qwen_rows_by_index.values() if float(row.get("all_accuracy", 0.0)) == 100.0),
            "exact_accurate_predictions": qwen_correct,
            "exact_matches": qwen_correct,
            "exact_accuracy": (qwen_correct / qwen_done) if qwen_done > 0 else 0.0,
        },
        "trm": {
            "valid_predictions": trm_valid,
            "all_accurate_predictions": sum(1 for row in trm_rows_by_index.values() if float(row.get("all_accuracy", 0.0)) == 100.0),
            "exact_accurate_predictions": trm_correct,
            "exact_matches": trm_correct,
            "exact_accuracy": (trm_correct / trm_done) if trm_done > 0 else 0.0,
        },
        "details": {
            "qwen": ordered_rows(qwen_rows_by_index),
            "trm": ordered_rows(trm_rows_by_index),
        },
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Snapshot saved: {out_json} | stage={stage} | qwen={qwen_done}/{len(records)} | trm={trm_done}/{len(records)}",
        flush=True,
    )


def write_trm_scored_payload():
    avg_all_accuracy, avg_exact_match = compute_trm_running_averages(trm_rows_by_index)
    trm_scored_payload = {
        "metadata": {
            "source_file": trm_predictions_json,
            "num_examples": len(records),
            "scored_examples": len(trm_rows_by_index),
            "scored_fields": [
                "prediction",
                "all_accuracy",
                "exact_match",
                "avg_all_accuracy",
                "avg_exact_match",
            ],
            "averages": {
                "avg_all_accuracy": avg_all_accuracy,
                "avg_exact_match": avg_exact_match,
                "avg_exact_match_pct": avg_exact_match * 100.0,
            },
            "score_save_every": trm_score_save_every,
            "source_metadata": trm_source_metadata,
        },
        "predictions": ordered_rows(trm_rows_by_index),
    }
    Path(trm_scored_json).parent.mkdir(parents=True, exist_ok=True)
    with open(trm_scored_json, "w", encoding="utf-8") as f:
        json.dump(trm_scored_payload, f, indent=2)
    print(
        f"Saved scored TRM predictions: {trm_scored_json} | scored={len(trm_rows_by_index)}/{len(records)} | avg_all_accuracy={avg_all_accuracy:.2f} | avg_exact_match={avg_exact_match:.4f}",
        flush=True,
    )


qwen_rows_by_index, trm_rows_by_index = load_resume_rows(out_json)
qwen_valid, qwen_correct = recompute_metrics(qwen_rows_by_index)
trm_valid, trm_correct = recompute_metrics(trm_rows_by_index)
qwen_parse_failures = sum(1 for row in qwen_rows_by_index.values() if row.get("prediction") is None)
print(
    f"Resume state: qwen={len(qwen_rows_by_index)}/{len(records)}, trm={len(trm_rows_by_index)}/{len(records)} from {out_json}",
    flush=True,
)

trm_map, trm_source_metadata = load_trm_predictions(trm_predictions_json, expected_examples=len(records))
print(f"Loaded {len(trm_map)} TRM prediction entries", flush=True)

trm_pending_records = [rec for rec in records if rec["index"] not in trm_rows_by_index]
if trm_pending_records:
    trm_done_start = len(trm_rows_by_index)
    trm_t0 = time.time()
    for i, rec in enumerate(trm_pending_records, start=1):
        idx = rec["index"]
        solution = rec["solution"]
        raw_pred = trm_map.get(idx)
        pred = extract_solution_candidate(raw_pred) if isinstance(raw_pred, str) else None
        all_accuracy = compute_all_accuracy(pred, solution)
        exact = 1 if pred == solution else 0
        if pred is not None:
            trm_valid += 1
        if exact:
            trm_correct += 1

        trm_rows_by_index[idx] = {
            "index": idx,
            "prediction": pred,
            "all_accuracy": all_accuracy,
            "exact_match": exact,
        }

        if len(trm_debug_rows) < 3:
            trm_debug_rows.append(
                {
                    "index": idx,
                    "raw_prediction": raw_pred,
                    "parsed_prediction": pred,
                    "all_accuracy": all_accuracy,
                    "target": solution,
                    "exact_match": exact,
                }
            )

        done = trm_done_start + i
        if done == 1 or done % trm_progress_every == 0 or done == len(records):
            elapsed = time.time() - trm_t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(records) - done
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_minutes = eta_seconds / 60 if eta_seconds != float("inf") else None
            eta_text = f"{eta_minutes:.1f}m" if eta_minutes is not None else "?"
            print(
                f"TRM scoring progress: {done}/{len(records)} ({100.0*done/len(records):.1f}%) | {rate:.2f} it/s | ETA {eta_text}",
                flush=True,
            )

        if done % save_every == 0 or done == len(records):
            save_snapshot(stage="trm_running")

        if done % trm_score_save_every == 0 or done == len(records):
            write_trm_scored_payload()
else:
    print("TRM scoring already complete from snapshot; skipping.", flush=True)

write_trm_scored_payload()

qwen_pending_records = [rec for rec in records if rec["index"] not in qwen_rows_by_index]
if qwen_pending_records:
    print("Loading Qwen3-1.7B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    qwen_done_start = len(qwen_rows_by_index)
    qwen_iter = qwen_pending_records
    if use_tqdm:
        qwen_iter = tqdm(
            qwen_pending_records,
            desc="Qwen inference",
            total=len(records),
            initial=qwen_done_start,
            dynamic_ncols=True,
        )
    qwen_t0 = time.time()

    for i, rec in enumerate(qwen_iter, start=1):
        puzzle = rec["puzzle"].replace(".", "0")
        solution = rec["solution"]
        prompt = (
            "Return only the final solved Sudoku as exactly 81 digits (1-9). No reasoning. No tags. No spaces. No extra text.\n"
            f"Puzzle: {puzzle}"
        )

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_solution_candidate(decoded)
        all_accuracy = compute_all_accuracy(pred, solution)
        exact = 1 if pred == solution else 0
        if pred is not None:
            qwen_valid += 1
        else:
            qwen_parse_failures += 1
        if exact:
            qwen_correct += 1

        idx = rec["index"]
        qwen_row = {
            "index": idx,
            "prediction": pred,
            "all_accuracy": all_accuracy,
            "exact_match": exact,
        }
        if pred is None:
            qwen_row["parse_error"] = "no_81_digit_candidate"
            qwen_row["decoded_preview"] = decoded[:500]
        qwen_rows_by_index[idx] = qwen_row

        if len(qwen_debug_rows) < 3:
            qwen_debug_rows.append(
                {
                    "index": idx,
                    "decoded_preview": decoded[:200],
                    "parsed_prediction": pred,
                    "all_accuracy": all_accuracy,
                    "target": solution,
                    "exact_match": exact,
                }
            )

        done = qwen_done_start + i
        if done == 1 or done % qwen_progress_every == 0 or done == len(records):
            elapsed = time.time() - qwen_t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(records) - done
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_minutes = eta_seconds / 60 if eta_seconds != float("inf") else None
            eta_text = f"{eta_minutes:.1f}m" if eta_minutes is not None else "?"
            print(
                f"Qwen progress: {done}/{len(records)} ({100.0*done/len(records):.1f}%) | {rate:.2f} it/s | ETA {eta_text}",
                flush=True,
            )

        if done % save_every == 0 or done == len(records):
            save_snapshot(stage="qwen_running")
else:
    print("Qwen inference already complete from snapshot; skipping.", flush=True)

result = {
    "test_json": test_json,
    "num_examples": len(records),
    "stage": "done",
    "completed": True,
    "progress": {
        "qwen_done": len(records),
        "trm_done": len(records),
        "save_every": save_every,
    },
    "qwen": {
        "valid_predictions": qwen_valid,
        "parse_failures": qwen_parse_failures,
        "all_accurate_predictions": sum(1 for row in qwen_rows_by_index.values() if float(row.get("all_accuracy", 0.0)) == 100.0),
        "exact_accurate_predictions": qwen_correct,
        "exact_matches": qwen_correct,
        "exact_accuracy": qwen_correct / len(records),
    },
    "trm": {
        "valid_predictions": trm_valid,
        "all_accurate_predictions": sum(1 for row in trm_rows_by_index.values() if float(row.get("all_accuracy", 0.0)) == 100.0),
        "exact_accurate_predictions": trm_correct,
        "exact_matches": trm_correct,
        "exact_accuracy": trm_correct / len(records),
    },
    "details": {
        "qwen": ordered_rows(qwen_rows_by_index),
        "trm": ordered_rows(trm_rows_by_index),
    },
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("\\n=== Baseline comparison ===")
print(f"Qwen valid predictions: {qwen_valid}/{len(records)}")
print(f"Qwen parse failures: {qwen_parse_failures}/{len(records)}")
print(f"Qwen exact accuracy: {result['qwen']['exact_accuracy']:.4f} ({qwen_correct}/{len(records)})")
print(f"TRM  valid predictions: {trm_valid}/{len(records)}")
print(f"TRM  exact accuracy: {result['trm']['exact_accuracy']:.4f} ({trm_correct}/{len(records)})")
print("Qwen debug samples:")
for row in qwen_debug_rows:
    print(row)
print("TRM debug samples:")
for row in trm_debug_rows:
    print(row)
print(f"Saved: {out_json}")
PY

echo "Job finished at $(date)"
