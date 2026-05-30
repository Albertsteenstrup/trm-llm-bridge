#!/usr/bin/env python3
import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List

try:
    import wandb
except Exception:
    wandb = None


def load_json_list(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    return data


def load_raw_records(path: str, max_samples: int) -> List[dict]:
    rows = load_json_list(path)
    out: List[dict] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        puzzle = row.get("puzzle")
        solution = row.get("solution")
        if isinstance(puzzle, str) and isinstance(solution, str) and len(puzzle) == 81 and len(solution) == 81:
            # Prefer corrected_nl_description if it exists and is not empty
            nl_desc = row.get("corrected_nl_description")
            if not nl_desc or not isinstance(nl_desc, str) or not nl_desc.strip():
                nl_desc = row.get("nl_description")
                
            out.append(
                {
                    "index": int(row.get("index", idx)),
                    "puzzle": puzzle,
                    "solution": solution,
                    "rating": row.get("rating"),
                    "nl_description": nl_desc if isinstance(nl_desc, str) else "",
                }
            )
    if max_samples > 0:
        out = out[:max_samples]
    if not out:
        raise ValueError("No valid records found")
    return out


def simple_nl_from_puzzle(puzzle: str) -> str:
    lines = ["Given entries:"]
    for row in range(1, 10):
        clues = []
        for col in range(1, 10):
            ch = puzzle[(row - 1) * 9 + (col - 1)]
            if ch != ".":
                clues.append(f"c{col}={ch}")
        if clues:
            lines.append(f"Row {row}: {', '.join(clues)}")
    return "\n".join(lines)


def apply_nl(records: List[dict], nl_json: str, seed: int) -> None:
    if nl_json:
        nl_rows = load_json_list(nl_json)
        by_puzzle: Dict[str, str] = {}
        for row in nl_rows:
            if not isinstance(row, dict):
                continue
            puzzle = row.get("puzzle")
            description = row.get("nl_description")
            if isinstance(puzzle, str) and isinstance(description, str) and description.strip():
                by_puzzle[puzzle] = description
        for rec in records:
            if rec["puzzle"] in by_puzzle:
                rec["nl_description"] = by_puzzle[rec["puzzle"]]

    rng = random.Random(seed)
    for rec in records:
        if not rec["nl_description"].strip():
            rec["nl_description"] = simple_nl_from_puzzle(rec["puzzle"])
        if rng.random() < 0.0:
            rec["nl_description"] = rec["nl_description"]


def ensure_nl_descriptions(records: List[dict]) -> None:
    missing = [row.get("index") for row in records if not row.get("nl_description", "").strip()]
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        raise ValueError(
            "Missing nl_description for one or more records after NL assignment. "
            f"Missing count={len(missing)} (sample indices: {preview})."
        )


def constrained_decode(logits) -> List[str]:
    preds = logits.argmax(dim=-1)
    out: List[str] = []
    for row in preds.tolist():
        chars = []
        for value in row:
            if value == 0:
                chars.append(".")
            elif 1 <= value <= 9:
                chars.append(str(value))
            else:
                chars.append(".")
        out.append("".join(chars))
    return out


def load_bridge(bridge_ckpt: str, llm_hidden_size: int, device: str, bridge_arch: str = ""):
    import torch

    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "train_translator_bridge.py"
    spec = importlib.util.spec_from_file_location("train_translator_bridge_mod", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load bridge module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    BridgeConfig = module.BridgeConfig
    ARCH_REGISTRY = module.ARCH_REGISTRY
    extract_checkpoint_arch = module.extract_checkpoint_arch
    extract_bridge_config_from_checkpoint = module.extract_bridge_config_from_checkpoint
    extract_model_state_dict = module.extract_model_state_dict

    checkpoint = torch.load(bridge_ckpt, map_location=device)
    cfg = extract_bridge_config_from_checkpoint(checkpoint) or BridgeConfig()
    cfg.llm_hidden_size = int(llm_hidden_size)
    cfg.device = device

    arch = bridge_arch or extract_checkpoint_arch(checkpoint, default="qformer")
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unsupported bridge architecture '{arch}' in checkpoint: {bridge_ckpt}")

    bridge = ARCH_REGISTRY[arch](cfg).to(device)
    state = extract_model_state_dict(checkpoint)
    bridge.load_state_dict(state)
    bridge.eval()
    return bridge, cfg, arch


def compute_bridge_metrics(rows: List[dict]) -> dict:
    n = len(rows)
    exact = 0
    total_cell_acc = 0.0
    total_given_acc = 0.0
    total_empty_acc = 0.0
    valid = 0

    for row in rows:
        gold = row["puzzle"]
        pred = row.get("bridge_puzzle", "")
        if isinstance(pred, str) and len(pred) == 81 and all((c == "." or c in "123456789") for c in pred):
            valid += 1
        if not isinstance(pred, str) or len(pred) != 81:
            pred = "." * 81

        if pred == gold:
            exact += 1
        total_cell_acc += sum(1 for a, b in zip(pred, gold) if a == b) / 81.0

        given_idx = [i for i, ch in enumerate(gold) if ch != "."]
        empty_idx = [i for i, ch in enumerate(gold) if ch == "."]
        if given_idx:
            total_given_acc += sum(1 for i in given_idx if pred[i] == gold[i]) / len(given_idx)
        if empty_idx:
            total_empty_acc += sum(1 for i in empty_idx if pred[i] == gold[i]) / len(empty_idx)

    return {
        "num_examples": n,
        "exact_match": exact / max(n, 1),
        "cell_accuracy": total_cell_acc / max(n, 1),
        "given_accuracy": total_given_acc / max(n, 1),
        "empty_accuracy": total_empty_acc / max(n, 1),
        "format_valid_rate": valid / max(n, 1),
    }


def load_trm_prediction_map(path: str) -> Dict[int, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = data.get("predictions", []) if isinstance(data, dict) else []
    out: Dict[int, str] = {}
    for row in preds:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        pred = row.get("prediction")
        if isinstance(idx, int) and isinstance(pred, str):
            out[idx] = pred
    return out


def compute_trm_metrics(bridge_rows: List[dict], trm_map: Dict[int, str]) -> dict:
    n = len(bridge_rows)
    exact = 0
    cell = 0.0
    valid = 0
    consistent = 0

    for row in bridge_rows:
        idx = row["index"]
        solution = row["solution"]
        bridge_puzzle = row.get("bridge_puzzle", "." * 81)
        pred = trm_map.get(idx, "")
        if isinstance(pred, str) and len(pred) == 81 and pred.isdigit() and all(c in "123456789" for c in pred):
            valid += 1
        if not isinstance(pred, str) or len(pred) != 81:
            pred = "0" * 81

        if pred == solution:
            exact += 1
        cell += sum(1 for a, b in zip(pred, solution) if a == b) / 81.0

        ok = True
        for i, ch in enumerate(bridge_puzzle):
            if ch != "." and pred[i] != ch:
                ok = False
                break
        if ok:
            consistent += 1

    return {
        "num_examples": n,
        "exact_match": exact / max(n, 1),
        "cell_accuracy": cell / max(n, 1),
        "format_valid_rate": valid / max(n, 1),
        "consistent_with_bridge_givens_rate": consistent / max(n, 1),
    }


def run_bridge_stage(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_raw_records(args.input_json, args.max_samples)
    apply_nl(records, args.nl_json, args.seed)
    if args.require_nl_description:
        ensure_nl_descriptions(records)

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(args.qwen_model, torch_dtype=llm_dtype)
    llm.to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    bridge, bridge_cfg, bridge_arch = load_bridge(
        args.bridge_checkpoint,
        int(llm.config.hidden_size),
        device,
        bridge_arch=args.bridge_arch,
    )

    batch_size = max(1, int(args.batch_size))
    out_rows: List[dict] = []

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            subset = records[start:start + batch_size]
            texts = [row["nl_description"] for row in subset]

            enc = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[bridge_cfg.llm_layer_index]
            logits = bridge(hidden.float(), attention_mask.bool())
            preds = constrained_decode(logits)

            for rec, pred in zip(subset, preds):
                out_rows.append(
                    {
                        "index": rec["index"],
                        "puzzle": rec["puzzle"],
                        "solution": rec["solution"],
                        "rating": rec.get("rating"),
                        "nl_description": rec["nl_description"],
                        "bridge_puzzle": pred,
                    }
                )

    bridge_metrics = compute_bridge_metrics(out_rows)

    bridge_output_path = Path(args.bridge_output_json)
    bridge_output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_output_path.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")

    trm_rows = [{"puzzle": row["bridge_puzzle"], "solution": row["solution"]} for row in out_rows]
    trm_input_path = Path(args.trm_input_json)
    trm_input_path.parent.mkdir(parents=True, exist_ok=True)
    trm_input_path.write_text(json.dumps(trm_rows, indent=2), encoding="utf-8")

    metrics_payload = {
        "stage": "bridge",
        "input_json": args.input_json,
        "qwen_model": args.qwen_model,
        "bridge_checkpoint": args.bridge_checkpoint,
        "bridge_arch": bridge_arch,
        "bridge_metrics": bridge_metrics,
        "bridge_output_json": str(bridge_output_path),
        "trm_input_json": str(trm_input_path),
    }

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"[bridge] wrote bridge outputs: {bridge_output_path}")
    print(f"[bridge] wrote TRM input: {trm_input_path}")
    print(f"[bridge] metrics: {bridge_metrics}")
    print(f"[bridge] metrics JSON: {metrics_path}")

    if args.wandb:
        if wandb is None:
            raise ImportError("--wandb requested but wandb is not installed")
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        run.log({f"bridge/{k}": v for k, v in bridge_metrics.items()})
        run.summary["bridge_output_json"] = str(bridge_output_path)
        run.summary["trm_input_json"] = str(trm_input_path)
        run.finish()


def run_finalize_stage(args):
    bridge_rows = load_json_list(args.bridge_output_json)
    trm_map = load_trm_prediction_map(args.trm_predictions_json)
    bridge_metrics = compute_bridge_metrics(bridge_rows)
    trm_metrics = compute_trm_metrics(bridge_rows, trm_map)

    payload = {
        "stage": "finalize",
        "bridge_output_json": args.bridge_output_json,
        "trm_predictions_json": args.trm_predictions_json,
        "bridge_metrics": bridge_metrics,
        "trm_metrics": trm_metrics,
    }

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[finalize] bridge metrics: {bridge_metrics}")
    print(f"[finalize] trm metrics: {trm_metrics}")
    print(f"[finalize] metrics JSON: {metrics_path}")

    if args.wandb:
        if wandb is None:
            raise ImportError("--wandb requested but wandb is not installed")
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        run.log({f"bridge/{k}": v for k, v in bridge_metrics.items()})
        run.log({f"trm/{k}": v for k, v in trm_metrics.items()})
        run.summary["bridge_output_json"] = args.bridge_output_json
        run.summary["trm_predictions_json"] = args.trm_predictions_json
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 -> bridge -> TRM pipeline")
    parser.add_argument("--stage", choices=["bridge", "finalize"], required=True)

    parser.add_argument("--input-json", type=str, default="")
    parser.add_argument("--nl-json", type=str, default="")
    parser.add_argument("--qwen-model", type=str, default="checkpoints/initial/Qwen3-1.7B")
    parser.add_argument("--bridge-checkpoint", type=str, default="checkpoints/initial/translator_bridge/10K/best_qformer.pt")
    parser.add_argument("--bridge-arch", type=str, default="",
                        help="Optional override when checkpoint lacks saved architecture metadata")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-nl-description", action="store_true")

    parser.add_argument("--bridge-output-json", type=str, default="results/test_pipeline/bridge_predictions.json")
    parser.add_argument("--trm-input-json", type=str, default="data/initial/sudoku_grid/test/test_pipeline_bridge_input.json")
    parser.add_argument("--trm-predictions-json", type=str, default="results/test_pipeline/trm_predictions.json")
    parser.add_argument("--metrics-json", type=str, default="results/test_pipeline/pipeline_metrics.json")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="trm-llm-test-pipeline")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "bridge":
        if not args.input_json:
            raise ValueError("--input-json is required for --stage bridge")
        run_bridge_stage(args)
    else:
        run_finalize_stage(args)


if __name__ == "__main__":
    main()
