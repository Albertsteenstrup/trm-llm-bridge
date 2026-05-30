#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def compare_arrays(gold_data: Any, pred_data: Any) -> tuple[float, float]:
    gold_arr = np.array(gold_data, dtype=np.float64)
    pred_arr = np.array(pred_data, dtype=np.float64)
    if gold_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: gold {gold_arr.shape} vs pred {pred_arr.shape}")
    abs_diff = np.abs(gold_arr - pred_arr)
    mae = float(np.mean(abs_diff))
    max_diff = float(np.max(abs_diff))
    return mae, max_diff

def main():
    parser = argparse.ArgumentParser(description="Calculate exact match accuracy of CLRS bridge-predicted inputs")
    parser.add_argument("--predicted-jsonl", required=True, help="Path to bridge_pred_native_<alg>.jsonl")
    parser.add_argument("--curriculum-jsonl", required=True, help="Path to filtered <alg>_level4_test.jsonl containing gold target")
    parser.add_argument("--tolerances", default="1e-5,1e-3,0.05,0.25", help="Comma-separated list of float tolerances for exact match")
    args = parser.parse_args()

    pred_path = Path(args.predicted_jsonl)
    curr_path = Path(args.curriculum_jsonl)
    
    if not pred_path.exists():
        print(f"Error: Prediction file {pred_path} does not exist.")
        return
    if not curr_path.exists():
        print(f"Error: Curriculum file {curr_path} does not exist.")
        return

    pred_rows = read_jsonl(pred_path)
    curr_rows = read_jsonl(curr_path)

    if len(pred_rows) != len(curr_rows):
        print(f"Warning: Row counts differ. Predictions: {len(pred_rows)}, Curriculum: {len(curr_rows)}")

    # Index curriculum rows by sample_id to be robust
    curr_by_id = {}
    for idx, row in enumerate(curr_rows):
        sid = row.get("sample_id") or f"row-{idx}"
        curr_by_id[sid] = row

    tolerances = [float(t) for t in args.tolerances.split(",")]

    # Statistics accumulators
    tensor_stats = {}  # tensor_name -> {tol: [matches], "mae": [values], "max_diff": [values]}
    row_exact_by_tol = {tol: 0 for tol in tolerances}
    total_compared = 0

    for pred_idx, pred_row in enumerate(pred_rows):
        sid = pred_row.get("sample_id") or f"row-{pred_idx}"
        if sid not in curr_by_id:
            # Fallback to positional matching if sample_id is not in curriculum
            if pred_idx < len(curr_rows):
                curr_row = curr_rows[pred_idx]
            else:
                print(f"Skipping pred index {pred_idx}: no matching curriculum row found.")
                continue
        else:
            curr_row = curr_by_id[sid]

        gold_target = curr_row.get("native_input_target")
        if not gold_target or "inputs" not in gold_target:
            print(f"Skipping row {sid}: no native_input_target/inputs in curriculum.")
            continue

        gold_inputs = {inp["name"]: inp for inp in gold_target["inputs"]}
        pred_inputs = {inp["name"]: inp for inp in pred_row.get("predicted_inputs", [])}

        row_ok_by_tol = {tol: True for tol in tolerances}
        all_tensors_matched = True

        for name, gold_inp in gold_inputs.items():
            if name not in pred_inputs:
                all_tensors_matched = False
                for tol in tolerances:
                    row_ok_by_tol[tol] = False
                continue

            pred_inp = pred_inputs[name]
            try:
                mae, max_diff = compare_arrays(gold_inp["data"], pred_inp["data"])
            except ValueError as e:
                print(f"Error comparing tensor {name} for row {sid}: {e}")
                all_tensors_matched = False
                for tol in tolerances:
                    row_ok_by_tol[tol] = False
                continue

            if name not in tensor_stats:
                tensor_stats[name] = {
                    "mae": [],
                    "max_diff": [],
                    "count": 0
                }
                for tol in tolerances:
                    tensor_stats[name][tol] = 0

            stats = tensor_stats[name]
            stats["mae"].append(mae)
            stats["max_diff"].append(max_diff)
            stats["count"] += 1

            for tol in tolerances:
                is_match = max_diff <= tol
                if is_match:
                    stats[tol] += 1
                else:
                    row_ok_by_tol[tol] = False

        if all_tensors_matched and len(gold_inputs) > 0:
            total_compared += 1
            for tol in tolerances:
                if row_ok_by_tol[tol]:
                    row_exact_by_tol[tol] += 1

    print(f"\n==========================================")
    print(f"Accuracy Report for {pred_path.name}")
    print(f"==========================================")
    print(f"Total rows compared: {total_compared}")
    print(f"Tolerances evaluated: {tolerances}\n")

    print(f"--- Individual Tensor Stats ---")
    for name, stats in sorted(tensor_stats.items()):
        count = stats["count"]
        mean_mae = np.mean(stats["mae"]) if count > 0 else 0.0
        mean_max = np.mean(stats["max_diff"]) if count > 0 else 0.0
        print(f"Tensor: {name:<10} | MAE: {mean_mae:.5f} | Mean MaxDiff: {mean_max:.5f}")
        for tol in tolerances:
            match_rate = (stats[tol] / count * 100.0) if count > 0 else 0.0
            print(f"  Exact Match @ tol={tol:<5g}: {match_rate:.2f}% ({stats[tol]}/{count})")
        print()

    print(f"--- End-to-End Row exact match (all inputs correct) ---")
    for tol in tolerances:
        row_rate = (row_exact_by_tol[tol] / total_compared * 100.0) if total_compared > 0 else 0.0
        print(f"Row Exact Match @ tol={tol:<5g}: {row_rate:.2f}% ({row_exact_by_tol[tol]}/{total_compared})")
    print(f"==========================================\n")

if __name__ == "__main__":
    main()
