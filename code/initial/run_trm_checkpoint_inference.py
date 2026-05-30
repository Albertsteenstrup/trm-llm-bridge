#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
import hashlib
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _load_test_records(path: str, max_puzzles: int) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list JSON at {path}")

    records: List[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        puzzle = item.get("puzzle")
        solution = item.get("solution")
        if isinstance(puzzle, str) and isinstance(solution, str) and len(puzzle) == 81 and len(solution) == 81:
            records.append({"index": i, "puzzle": puzzle, "solution": solution})

    if max_puzzles > 0:
        records = records[:max_puzzles]
    return records


def _resolve_checkpoint_file(checkpoint_dir: str, checkpoint_file: Optional[str]) -> str:
    if checkpoint_file:
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        return checkpoint_file

    candidates = []
    for p in glob.glob(os.path.join(checkpoint_dir, "step_*")):
        if os.path.isfile(p) and re.match(r".*?/step_\d+$", p):
            try:
                step = int(os.path.basename(p).split("_")[-1])
            except ValueError:
                continue
            candidates.append((step, p))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files like step_<N> found in {checkpoint_dir}. "
            f"Set --checkpoint-file explicitly."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _tensor_to_sudoku_strings(tensor, batch_size: int, batch_inputs: Optional[List[List[int]]] = None) -> Optional[List[Optional[str]]]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None

    t = tensor.detach().cpu()

    if t.ndim == 3:
        if t.shape[1] == 81:
            t = t.argmax(dim=-1)
        elif t.shape[2] == 81:
            t = t.argmax(dim=1)
        else:
            return None
    elif t.ndim == 2:
        if t.shape[1] == 81:
            pass
        elif t.shape[0] == 81 and batch_size == 1:
            t = t.unsqueeze(0)
        else:
            return None
    else:
        return None

    out: List[Optional[str]] = []
    for row in t.tolist():
        if len(row) != 81:
            out.append(None)
            continue

        try:
            row_int = [int(v) for v in row]
        except Exception:
            out.append(None)
            continue

        row_idx = len(out)
        input_row = None
        if batch_inputs is not None and row_idx < len(batch_inputs):
            input_row = batch_inputs[row_idx]

        digits: List[str] = []
        valid = True
        for pos, iv in enumerate(row_int):
            digit: Optional[int] = None
            if 2 <= iv <= 10:
                digit = iv - 1
            elif 1 <= iv <= 9:
                digit = iv
            elif input_row is not None and pos < len(input_row):
                input_token = int(input_row[pos])
                if 2 <= input_token <= 10:
                    digit = input_token - 1
                elif 1 <= input_token <= 9:
                    digit = input_token

            if digit is None or digit < 1 or digit > 9:
                valid = False
                break
            digits.append(str(digit))

        out.append("".join(digits) if valid and len(digits) == 81 else None)

    if len(out) != batch_size:
        if len(out) > batch_size:
            out = out[:batch_size]
        else:
            out.extend([None] * (batch_size - len(out)))

    return out


def _extract_predictions_from_preds(preds: Dict[str, object], batch_size: int, batch_inputs: Optional[List[List[int]]] = None) -> Tuple[List[Optional[str]], Optional[str], int]:
    best: Optional[List[Optional[str]]] = None
    best_key: Optional[str] = None
    best_valid = -1

    for key, value in preds.items():
        parsed = _tensor_to_sudoku_strings(value, batch_size, batch_inputs=batch_inputs)
        if parsed is None:
            continue
        valid_count = sum(1 for x in parsed if x is not None)
        if valid_count > best_valid:
            best_valid = valid_count
            best = parsed
            best_key = key

    if best is None:
        return [None] * batch_size, None, 0
    return best, best_key, best_valid


def _import_trm_pretrain(trm_root: str):
    if trm_root not in sys.path:
        sys.path.insert(0, trm_root)

    import pretrain  # type: ignore

    return pretrain


def _encode_sudoku_puzzle(puzzle: str) -> List[int]:
    out: List[int] = []
    for ch in puzzle:
        if ch in {".", "0"}:
            out.append(1)
        elif ch in "123456789":
            out.append(int(ch) + 1)
        else:
            raise ValueError(f"Unexpected puzzle character: {ch!r}")
    if len(out) != 81:
        raise ValueError("Puzzle length must be 81")
    return out


def _encode_sudoku_solution(solution: str) -> List[int]:
    out: List[int] = []
    for ch in solution:
        if ch in "123456789":
            out.append(int(ch) + 1)
        else:
            raise ValueError(f"Unexpected solution character: {ch!r}")
    if len(out) != 81:
        raise ValueError("Solution length must be 81")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run TRM checkpoint inference on held-out Sudoku JSON")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--trm-root", default="")
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-puzzles", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=0)
    parser.add_argument("--require-test-path", default="/data/initial/sudoku_grid/test/")
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    checkpoint_file = _resolve_checkpoint_file(checkpoint_dir, args.checkpoint_file or None)
    trm_root = os.path.abspath(args.trm_root) if args.trm_root else os.path.abspath("code/initial/TinyRecursiveModels")

    print(f"[TRM] checkpoint_dir={checkpoint_dir}")
    print(f"[TRM] checkpoint_file={checkpoint_file}")
    print(f"[TRM] trm_root={trm_root}")

    normalized_test_json = os.path.abspath(args.test_json)
    required_test_segment = args.require_test_path
    if required_test_segment and required_test_segment not in normalized_test_json:
        raise ValueError(
            f"test-json must come from '{required_test_segment}' for fair held-out evaluation. "
            f"Got: {normalized_test_json}"
        )

    records = _load_test_records(normalized_test_json, args.max_puzzles)
    if not records:
        raise ValueError("No valid held-out test records found")
    print(f"[TRM] loaded {len(records)} test records")

    test_hash = hashlib.sha256()
    for rec in records:
        test_hash.update(str(rec["index"]).encode("utf-8"))
        test_hash.update(rec["puzzle"].encode("utf-8"))
        test_hash.update(rec["solution"].encode("utf-8"))
    test_hash_hex = test_hash.hexdigest()

    if not os.path.isdir(trm_root):
        raise FileNotFoundError(f"TRM root not found: {trm_root}")

    pretrain = _import_trm_pretrain(trm_root)

    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Expected config file not found: {config_path}")

    import yaml
    import torch

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Invalid YAML config in {config_path}")

    config = pretrain.PretrainConfig(**cfg_dict)
    config.load_checkpoint = checkpoint_file
    config.checkpoint_path = None
    config.evaluators = []
    config.eval_save_outputs = []
    if int(args.global_batch_size) > 0:
        config.global_batch_size = int(args.global_batch_size)
    config.epochs = max(1, int(getattr(config, "epochs", 1)))
    config.eval_interval = config.epochs

    print(f"[TRM] global_batch_size={config.global_batch_size}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TRM checkpoint inference")

    inferred_metadata = SimpleNamespace(
        vocab_size=11,
        seq_len=81,
        num_puzzle_identifiers=1,
    )

    model, _, _ = pretrain.create_model(config, inferred_metadata, rank=0, world_size=1)
    model.eval()

    requested_keys = [
        "predictions",
        "prediction",
        "output",
        "outputs",
        "logits",
        "y",
        "y_pred",
        "solution",
        "solutions",
    ]

    rows: List[dict] = []
    seen = 0
    extraction_key_counts: Dict[str, int] = {}

    import torch

    batch_size = max(1, int(config.global_batch_size))
    starts = list(range(0, len(records), batch_size))
    batch_iter = starts
    if tqdm is not None:
        batch_iter = tqdm(starts, desc="TRM inference", total=len(starts), dynamic_ncols=True)

    with torch.inference_mode():
        for start in batch_iter:
            if seen >= len(records):
                break

            subset = records[start:start + batch_size]
            inputs = [_encode_sudoku_puzzle(rec["puzzle"]) for rec in subset]
            labels = [_encode_sudoku_solution(rec["solution"]) for rec in subset]

            batch = {
                "inputs": torch.tensor(inputs, dtype=torch.int32, device="cuda"),
                "labels": torch.tensor(labels, dtype=torch.int32, device="cuda"),
                "puzzle_identifiers": torch.zeros((len(subset),), dtype=torch.int32, device="cuda"),
            }

            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            first_pred_shape = None
            while True:
                carry, _, _, preds, all_finish = model(carry=carry, batch=batch, return_keys=requested_keys)
                if isinstance(preds, dict) and first_pred_shape is None:
                    for _k, _v in preds.items():
                        if hasattr(_v, "shape"):
                            first_pred_shape = tuple(_v.shape)
                            break
                if all_finish:
                    break

            local_batch_size = int(batch["inputs"].shape[0]) if "inputs" in batch else 0
            decoded, extraction_key, _ = _extract_predictions_from_preds(
                preds if isinstance(preds, dict) else {},
                local_batch_size,
                batch_inputs=inputs,
            )
            if extraction_key:
                extraction_key_counts[extraction_key] = extraction_key_counts.get(extraction_key, 0) + 1

            if start == 0:
                print(f"[TRM] first batch extraction_key={extraction_key}, first_pred_shape={first_pred_shape}")

            for row_i, pred in enumerate(decoded):
                if seen >= len(records):
                    break
                rows.append({"index": subset[row_i]["index"], "prediction": pred})
                seen += 1

            if tqdm is None and (seen % 100 == 0 or seen == len(records)):
                print(f"[TRM] progress {seen}/{len(records)}")

    if seen == 0:
        raise RuntimeError(
            "TRM inference produced no rows. "
            "If your environment differs from upstream TRM internals, provide TRM_EVAL_CMD with your custom inference command."
        )

    dominant_extraction_key = None
    if extraction_key_counts:
        dominant_extraction_key = sorted(extraction_key_counts.items(), key=lambda x: x[1], reverse=True)[0][0]

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "test_json": normalized_test_json,
                    "test_json_sha256": test_hash_hex,
                    "num_examples": len(records),
                    "checkpoint_file": checkpoint_file,
                    "checkpoint_dir": checkpoint_dir,
                    "test_data_source": "raw_test_json",
                    "global_batch_size": config.global_batch_size,
                    "dominant_extraction_key": dominant_extraction_key,
                    "extraction_key_counts": extraction_key_counts,
                },
                "predictions": rows,
            },
            f,
            indent=2,
        )
    print(f"[TRM] wrote predictions: {args.output_json}")


if __name__ == "__main__":
    main()
