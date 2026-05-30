#!/usr/bin/env python3
"""Fine-tune Qwen to parse natural-language Sudoku clues into initial grids.

This baseline evaluates:
    Qwen / fine-tuned Qwen -> parsed givens grid -> TRM solver

The target is the 81-character puzzle grid, not the solved Sudoku. Empty cells
are represented with "." and givens with digits 1-9.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_sudoku_finetune import (  # noqa: E402
    extract_model_state_dict,
    load_checkpoint_any,
    load_model_and_tokenizer,
    load_records,
    masked_token_accuracy,
    resolve_path,
    resolve_torch_dtype,
    set_seed,
    split_train_val,
)

try:
    import wandb
except ImportError:
    wandb = None


PROMPT_TEMPLATE = """You translate natural-language Sudoku clues into the initial Sudoku grid.

Rules:
- Return only the initial grid as exactly 81 characters.
- Use digits 1-9 for given cells.
- Use . for empty cells.
- No explanation, no JSON, no markdown, no extra text.

Description:
{nl_description}

Initial grid:
"""


@dataclass
class GridFineTuneConfig:
    base_model_name: str = "checkpoints/initial/Qwen3-1.7B"
    train_dataset_path: str = "data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json"
    eval_dataset_path: str = "data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json"
    checkpoint_dir: str = "checkpoints/initial/qwen_sudoku_grid_ft"
    log_dir: str = "logs/initial"
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_fraction: float = 0.05
    max_length: int = 768
    max_new_tokens: int = 96
    eval_split: float = 0.1
    seed: int = 42
    save_every_epoch: int = 1
    patience: int = 3
    grad_clip_norm: float = 1.0
    llm_dtype: str = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_grid(grid: str) -> str:
    return "".join("." if ch == "0" else ch for ch in grid)


def build_prompt(nl_description: str) -> str:
    return PROMPT_TEMPLATE.format(nl_description=nl_description.strip())


def parse_grid_from_text(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", "empty"

    direct = re.search(r"(?<![0-9.])([0-9.]{81})(?![0-9.])", raw)
    if direct:
        grid = normalize_grid(direct.group(1))
        if all(ch == "." or ch in "123456789" for ch in grid):
            return grid, "regex81"

    compact = re.sub(r"[^0-9.]", "", raw)
    if len(compact) == 81:
        grid = normalize_grid(compact)
        if all(ch == "." or ch in "123456789" for ch in grid):
            return grid, "compact81"

    return "", "invalid_format"


def compute_grid_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    exact = 0
    valid = 0
    total_cell_acc = 0.0
    total_given_acc = 0.0
    total_empty_acc = 0.0

    for row in rows:
        gold = normalize_grid(row["puzzle"])
        pred = row.get("predicted_grid") or row.get("bridge_puzzle") or ""
        if isinstance(pred, str):
            pred = normalize_grid(pred)
        if isinstance(pred, str) and len(pred) == 81 and all(ch == "." or ch in "123456789" for ch in pred):
            valid += 1
        else:
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


class SudokuGridFineTuneDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        prompt = build_prompt(row["nl_description"])
        target = normalize_grid(row["puzzle"])

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        eos_ids = [self.eos_token_id] if self.eos_token_id is not None else []

        max_prompt_tokens = max(1, self.max_length - len(target_ids) - len(eos_ids))
        prompt_ids = prompt_ids[:max_prompt_tokens]

        input_ids = prompt_ids + target_ids + eos_ids
        labels = ([-100] * len(prompt_ids)) + target_ids + eos_ids

        return {
            "index": row["index"],
            "puzzle": normalize_grid(row["puzzle"]),
            "solution": row["solution"],
            "rating": row.get("rating"),
            "nl_description": row["nl_description"],
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": [1] * len(prompt_ids),
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_seq = max(len(item["input_ids"]) for item in batch)
        max_prompt = max(len(item["prompt_input_ids"]) for item in batch)

        return {
            "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
            "puzzle": [item["puzzle"] for item in batch],
            "solution": [item["solution"] for item in batch],
            "rating": [item.get("rating") for item in batch],
            "nl_description": [item["nl_description"] for item in batch],
            "input_ids": torch.tensor(
                [([pad_id] * (max_seq - len(item["input_ids"]))) + item["input_ids"] for item in batch],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [([0] * (max_seq - len(item["attention_mask"]))) + item["attention_mask"] for item in batch],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [([-100] * (max_seq - len(item["labels"]))) + item["labels"] for item in batch],
                dtype=torch.long,
            ),
            "prompt_input_ids": torch.tensor(
                [([pad_id] * (max_prompt - len(item["prompt_input_ids"]))) + item["prompt_input_ids"] for item in batch],
                dtype=torch.long,
            ),
            "prompt_attention_mask": torch.tensor(
                [([0] * (max_prompt - len(item["prompt_attention_mask"]))) + item["prompt_attention_mask"] for item in batch],
                dtype=torch.long,
            ),
        }

    return collate


def generate_grid_rows(model, tokenizer, loader: DataLoader, device: torch.device, max_new_tokens: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generate grids", leave=False):
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            generated = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = generated[:, prompt_input_ids.shape[1]:]
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            for i, text in enumerate(texts):
                predicted_grid, parse_status = parse_grid_from_text(text)
                rows.append(
                    {
                        "index": int(batch["index"][i].item()),
                        "puzzle": batch["puzzle"][i],
                        "solution": batch["solution"][i],
                        "rating": batch["rating"][i],
                        "nl_description": batch["nl_description"][i],
                        "predicted_grid": predicted_grid,
                        "bridge_puzzle": predicted_grid,
                        "response_text": text,
                        "parse_status": parse_status,
                    }
                )
    return rows


def evaluate_model(model, tokenizer, loader: DataLoader, device: torch.device, autocast_dtype: torch.dtype, max_new_tokens: int):
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval loss", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            total_loss += float(outputs.loss.item())
            total_token_acc += masked_token_accuracy(outputs.logits.detach().float(), labels)
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Validation loader is empty")

    prediction_rows = generate_grid_rows(model, tokenizer, loader, device, max_new_tokens)
    grid_metrics = compute_grid_metrics(prediction_rows)
    return {
        "loss": total_loss / total_batches,
        "target_token_accuracy": total_token_acc / total_batches,
        **grid_metrics,
    }, prediction_rows


def checkpoint_payload(model, optimizer, scheduler, config, epoch, best_exact, best_cell, no_improve, training_log, best_path):
    return {
        "config": asdict(config),
        "target_mode": "grid",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_exact_match": best_exact,
        "best_cell_accuracy": best_cell,
        "epochs_without_improvement": no_improve,
        "training_log": training_log,
        "best_checkpoint_path": best_path,
    }


def model_only_payload(model, config, epoch, metrics):
    return {
        "config": asdict(config),
        "target_mode": "grid",
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }


def run_train(args) -> None:
    config = GridFineTuneConfig()
    for attr, arg_name in [
        ("base_model_name", "base_model"),
        ("train_dataset_path", "dataset"),
        ("checkpoint_dir", "checkpoint_dir"),
        ("log_dir", "log_dir"),
        ("train_batch_size", "batch_size"),
        ("eval_batch_size", "eval_batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("learning_rate", "lr"),
        ("weight_decay", "weight_decay"),
        ("epochs", "epochs"),
        ("max_length", "max_length"),
        ("max_new_tokens", "max_new_tokens"),
        ("eval_split", "eval_split"),
        ("seed", "seed"),
        ("save_every_epoch", "save_every_epoch"),
        ("patience", "patience"),
        ("grad_clip_norm", "grad_clip_norm"),
        ("llm_dtype", "llm_dtype"),
    ]:
        value = getattr(args, arg_name)
        if value is not None and value != "":
            setattr(config, attr, value)

    config.base_model_name = resolve_path(config.base_model_name) if Path(resolve_path(config.base_model_name)).exists() else config.base_model_name
    config.train_dataset_path = resolve_path(config.train_dataset_path)
    config.checkpoint_dir = resolve_path(config.checkpoint_dir)
    config.log_dir = resolve_path(config.log_dir)

    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = torch.device(config.device)
    llm_dtype = resolve_torch_dtype(config.llm_dtype, config.device)
    model, tokenizer = load_model_and_tokenizer(config.base_model_name, llm_dtype, device)

    records = load_records(config.train_dataset_path)
    train_records, val_records = split_train_val(records, config.eval_split, config.seed)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(SudokuGridFineTuneDataset(train_records, tokenizer, config.max_length), batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SudokuGridFineTuneDataset(val_records, tokenizer, config.max_length), batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, config.gradient_accumulation_steps)))
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_fraction),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and llm_dtype == torch.float16)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = os.path.join(config.log_dir, f"qwen_sudoku_grid_ft_{int(time.time())}.json")
    latest_path = os.path.join(config.checkpoint_dir, "latest_full.pt")

    start_epoch = 0
    best_exact = -1.0
    best_cell = -1.0
    no_improve = 0
    best_path: str | None = None
    training_log: dict[str, Any] = {"config": asdict(config), "target_mode": "grid", "train_size": len(train_records), "val_size": len(val_records), "epochs": []}

    resume_path = resolve_path(args.resume_from) if args.resume_from else (latest_path if not args.no_auto_resume and os.path.isfile(latest_path) else "")
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = load_checkpoint_any(resume_path, config.device)
        model.load_state_dict(extract_model_state_dict(checkpoint))
        if isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict") is not None and not args.weights_only_resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            no_improve = int(checkpoint.get("epochs_without_improvement", 0))
            training_log = checkpoint.get("training_log", training_log)
            best_exact = float(checkpoint.get("best_exact_match", best_exact))
            best_cell = float(checkpoint.get("best_cell_accuracy", best_cell))
            best_path = checkpoint.get("best_checkpoint_path", best_path)
        elif args.weights_only_resume:
            print("Loaded model weights only; optimizer, scheduler, and validation-best counters are reset.")

    if args.wandb:
        if wandb is None:
            raise ImportError("--wandb was set but package 'wandb' is not installed")
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name or f"qwen_grid_ft_{int(time.time())}", config=asdict(config))

    print(f"Base model: {config.base_model_name}")
    print(f"Train dataset: {config.train_dataset_path}")
    print(f"Train/val sizes: {len(train_records)}/{len(val_records)}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Using dtype: {llm_dtype}")

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_token_acc = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch + 1}"), start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                loss = outputs.loss / float(max(1, config.gradient_accumulation_steps))

            token_acc = masked_token_accuracy(outputs.logits.detach().float(), labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % max(1, config.gradient_accumulation_steps) == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * float(max(1, config.gradient_accumulation_steps))
            epoch_token_acc += token_acc
            batch_count += 1

            if args.wandb and (step % 20 == 0 or step == len(train_loader)):
                wandb.log({"train/loss": epoch_loss / batch_count, "train/target_token_accuracy": epoch_token_acc / batch_count, "train/lr": float(scheduler.get_last_lr()[0])}, step=global_step + step)

        val_metrics, _ = evaluate_model(model, tokenizer, val_loader, device, llm_dtype, config.max_new_tokens)
        elapsed = time.time() - t0
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(1, batch_count),
            "train_target_token_accuracy": epoch_token_acc / max(1, batch_count),
            "val_loss": val_metrics["loss"],
            "val_target_token_accuracy": val_metrics["target_token_accuracy"],
            "val_grid_exact_match": val_metrics["exact_match"],
            "val_grid_cell_accuracy": val_metrics["cell_accuracy"],
            "val_given_accuracy": val_metrics["given_accuracy"],
            "val_empty_accuracy": val_metrics["empty_accuracy"],
            "val_format_valid_rate": val_metrics["format_valid_rate"],
            "elapsed_seconds": elapsed,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        training_log["epochs"].append(epoch_log)
        print(f"\nEpoch {epoch + 1}/{config.epochs} ({elapsed:.1f}s)")
        print(f"  Val: grid_exact={val_metrics['exact_match']:.4f}, cell={val_metrics['cell_accuracy']:.4f}, given={val_metrics['given_accuracy']:.4f}, empty={val_metrics['empty_accuracy']:.4f}, format={val_metrics['format_valid_rate']:.4f}")

        improved = val_metrics["exact_match"] > best_exact or (val_metrics["exact_match"] == best_exact and val_metrics["cell_accuracy"] > best_cell)
        if improved:
            best_exact = val_metrics["exact_match"]
            best_cell = val_metrics["cell_accuracy"]
            no_improve = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save(model_only_payload(model, config, epoch + 1, val_metrics), best_path)
            print(f"  New best checkpoint: {best_path}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{config.patience})")

        if (epoch + 1) % config.save_every_epoch == 0 or (epoch + 1) == config.epochs:
            payload = checkpoint_payload(model, optimizer, scheduler, config, epoch + 1, best_exact, best_cell, no_improve, training_log, best_path)
            torch.save(payload, os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch + 1:03d}.pt"))
            torch.save(payload, latest_path)

        Path(log_path).write_text(json.dumps(training_log, indent=2), encoding="utf-8")
        if args.wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step + len(train_loader))
        global_step += len(train_loader)

        if no_improve >= config.patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    training_log["best_checkpoint"] = best_path
    Path(log_path).write_text(json.dumps(training_log, indent=2), encoding="utf-8")
    print(f"Training log: {log_path}")
    print(f"Best checkpoint: {best_path}")
    if args.wandb:
        wandb.finish()


def run_eval(args) -> None:
    config = GridFineTuneConfig()
    if args.base_model:
        config.base_model_name = args.base_model
    if args.input_json:
        config.eval_dataset_path = args.input_json
    for attr in ("max_length", "max_new_tokens", "eval_batch_size", "seed", "llm_dtype"):
        value = getattr(args, attr)
        if value is not None and value != "":
            setattr(config, attr, value)

    config.base_model_name = resolve_path(config.base_model_name) if Path(resolve_path(config.base_model_name)).exists() else config.base_model_name
    config.eval_dataset_path = resolve_path(config.eval_dataset_path)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else ""

    set_seed(config.seed)
    device = torch.device(config.device)
    llm_dtype = resolve_torch_dtype(config.llm_dtype, config.device)
    model, tokenizer = load_model_and_tokenizer(config.base_model_name, llm_dtype, device)
    if checkpoint_path:
        checkpoint = load_checkpoint_any(checkpoint_path, config.device)
        model.load_state_dict(extract_model_state_dict(checkpoint))

    records = load_records(config.eval_dataset_path, args.max_samples)
    loader = DataLoader(SudokuGridFineTuneDataset(records, tokenizer, config.max_length), batch_size=config.eval_batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer))
    metrics, rows = evaluate_model(model, tokenizer, loader, device, llm_dtype, config.max_new_tokens)

    bridge_output_json = resolve_path(args.bridge_output_json or args.predictions_json)
    predictions_json = resolve_path(args.predictions_json or bridge_output_json)
    metrics_json = resolve_path(args.metrics_json)
    trm_input_json = resolve_path(args.trm_input_json) if args.trm_input_json else ""

    Path(predictions_json).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if bridge_output_json != predictions_json:
        Path(bridge_output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(bridge_output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if trm_input_json:
        trm_rows = [{"puzzle": row["bridge_puzzle"] if row["bridge_puzzle"] else "." * 81, "solution": row["solution"]} for row in rows]
        Path(trm_input_json).parent.mkdir(parents=True, exist_ok=True)
        Path(trm_input_json).write_text(json.dumps(trm_rows, indent=2), encoding="utf-8")

    payload = {
        "stage": "grid_parse",
        "input_json": config.eval_dataset_path,
        "base_model": config.base_model_name,
        "checkpoint": checkpoint_path or None,
        "target_mode": "grid",
        "predictions_json": predictions_json,
        "bridge_output_json": bridge_output_json,
        "trm_input_json": trm_input_json or None,
        "grid_metrics": metrics,
        "bridge_metrics": {k: metrics[k] for k in ("num_examples", "exact_match", "cell_accuracy", "given_accuracy", "empty_accuracy", "format_valid_rate")},
    }
    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[eval] wrote grid predictions: {predictions_json}")
    print(f"[eval] wrote bridge output: {bridge_output_json}")
    if trm_input_json:
        print(f"[eval] wrote TRM input: {trm_input_json}")
    print(f"[eval] grid metrics: {metrics}")
    print(f"[eval] metrics JSON: {metrics_json}")

    if args.wandb:
        if wandb is None:
            raise ImportError("--wandb requested but wandb is not installed")
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
        run.log({f"grid/{k}": v for k, v in metrics.items()})
        run.summary["predictions_json"] = predictions_json
        run.summary["checkpoint"] = checkpoint_path or "<base-model-only>"
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Qwen as a Sudoku text-to-grid parser")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Fine-tune Qwen to emit the initial Sudoku grid")
    train.add_argument("--base-model", type=str, default="checkpoints/initial/Qwen3-1.7B")
    train.add_argument("--dataset", type=str, default="data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse_10000.json")
    train.add_argument("--checkpoint-dir", type=str, default="checkpoints/initial/qwen_sudoku_grid_ft")
    train.add_argument("--log-dir", type=str, default="logs/initial")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--eval-batch-size", type=int, default=4)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--max-length", type=int, default=768)
    train.add_argument("--max-new-tokens", type=int, default=96)
    train.add_argument("--eval-split", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save-every-epoch", type=int, default=1)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--grad-clip-norm", type=float, default=1.0)
    train.add_argument("--llm-dtype", type=str, default="auto")
    train.add_argument("--resume-from", type=str, default="")
    train.add_argument("--weights-only-resume", action="store_true", help="Load model weights but start a fresh optimizer/schedule for continuation fine-tuning")
    train.add_argument("--no-auto-resume", action="store_true")
    train.add_argument("--wandb", action="store_true")
    train.add_argument("--wandb-project", type=str, default="trm-llm-qwen-grid-ft")
    train.add_argument("--wandb-run-name", type=str, default=None)
    train.add_argument("--wandb-entity", type=str, default=None)

    evaluate = subparsers.add_parser("eval", help="Evaluate Qwen grid parsing and optionally write TRM input JSON")
    evaluate.add_argument("--base-model", type=str, default="checkpoints/initial/Qwen3-1.7B")
    evaluate.add_argument("--checkpoint", type=str, default="")
    evaluate.add_argument("--input-json", type=str, default="data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json")
    evaluate.add_argument("--predictions-json", type=str, default="results/qwen_sudoku_grid_ft/grid_predictions_corrected2.json")
    evaluate.add_argument("--bridge-output-json", type=str, default="")
    evaluate.add_argument("--trm-input-json", type=str, default="")
    evaluate.add_argument("--metrics-json", type=str, default="results/qwen_sudoku_grid_ft/grid_metrics_corrected2.json")
    evaluate.add_argument("--max-samples", type=int, default=1000)
    evaluate.add_argument("--eval-batch-size", type=int, default=4)
    evaluate.add_argument("--max-length", type=int, default=768)
    evaluate.add_argument("--max-new-tokens", type=int, default=96)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--llm-dtype", type=str, default="auto")
    evaluate.add_argument("--wandb", action="store_true")
    evaluate.add_argument("--wandb-project", type=str, default="trm-llm-qwen-grid-ft-eval")
    evaluate.add_argument("--wandb-run-name", type=str, default=None)
    evaluate.add_argument("--wandb-entity", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
