#!/usr/bin/env python3
"""Fine-tune Qwen on direct Sudoku solving from natural-language clues."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb
except ImportError:
    wandb = None


PROMPT_TEMPLATE = """You solve Sudoku from natural-language clue descriptions.

Rules:
- Return only the final solved Sudoku as exactly 81 digits using 1-9.
- No explanation, no JSON, no markdown, no extra text.

Description:
{nl_description}

Solution:
"""


@dataclass
class FineTuneConfig:
    base_model_name: str = "checkpoints/initial/Qwen3-1.7B"
    train_dataset_path: str = "data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse.json"
    eval_dataset_path: str = "data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json"
    checkpoint_dir: str = "checkpoints/initial/qwen_sudoku_ft"
    log_dir: str = "logs/initial"
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_fraction: float = 0.05
    max_length: int = 768
    max_new_tokens: int = 128
    eval_split: float = 0.1
    seed: int = 42
    save_every_epoch: int = 10
    patience: int = 6
    grad_clip_norm: float = 1.0
    llm_dtype: str = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((repo_root() / path).resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "auto":
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def extract_nl_text(record: dict[str, Any]) -> str:
    corrected = record.get("corrected_nl_description")
    if isinstance(corrected, str) and corrected.strip():
        return corrected
    for key in ("nl_description", "description", "prompt", "text", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def build_prompt(nl_description: str) -> str:
    return PROMPT_TEMPLATE.format(nl_description=nl_description.strip())


def load_records(path: str, max_samples: int = 0) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON in {path}")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        puzzle = row.get("puzzle")
        solution = row.get("solution")
        nl_text = extract_nl_text(row)
        if not isinstance(puzzle, str) or len(puzzle) != 81:
            continue
        if solution is not None and (not isinstance(solution, str) or len(solution) != 81):
            continue
        if not nl_text:
            continue
        out.append(
            {
                "index": int(row.get("index", idx)),
                "puzzle": puzzle,
                "solution": solution if isinstance(solution, str) else "",
                "rating": row.get("rating"),
                "nl_description": nl_text,
            }
        )
    if max_samples > 0:
        out = out[:max_samples]
    if not out:
        raise ValueError(f"No valid records found in {path}")
    return out


def split_train_val(records: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    val_size = max(1, int(len(shuffled) * val_fraction))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[:-val_size], shuffled[-val_size:]


def parse_solution_from_text(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", "empty"
    direct = re.search(r"(?<![0-9])([1-9]{81})(?![0-9])", raw)
    if direct:
        return direct.group(1), "regex81"
    compact = re.sub(r"\s+", "", raw)
    if len(compact) == 81 and all(ch in "123456789" for ch in compact):
        return compact, "compact81"
    return "", "invalid_format"


def is_valid_sudoku_solution(solution: str) -> bool:
    if not isinstance(solution, str) or len(solution) != 81 or not solution.isdigit():
        return False

    def valid_group(chars: list[str]) -> bool:
        return sorted(chars) == list("123456789")

    rows = [list(solution[i * 9:(i + 1) * 9]) for i in range(9)]
    cols = [[solution[r * 9 + c] for r in range(9)] for c in range(9)]
    boxes = []
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            box = []
            for r in range(box_r, box_r + 3):
                for c in range(box_c, box_c + 3):
                    box.append(solution[r * 9 + c])
            boxes.append(box)
    return all(valid_group(group) for group in rows + cols + boxes)


def compute_solution_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    exact = 0
    total_cell_acc = 0.0
    valid = 0
    valid_sudoku = 0
    consistent_with_gold_givens = 0

    for row in rows:
        gold_solution = row["solution"]
        gold_puzzle = row["puzzle"]
        pred = row.get("predicted_solution", "")
        if isinstance(pred, str) and len(pred) == 81 and pred.isdigit() and all(c in "123456789" for c in pred):
            valid += 1
            if is_valid_sudoku_solution(pred):
                valid_sudoku += 1
        else:
            pred = "0" * 81

        if pred == gold_solution:
            exact += 1
        total_cell_acc += sum(1 for a, b in zip(pred, gold_solution) if a == b) / 81.0

        consistent = True
        for i, ch in enumerate(gold_puzzle):
            if ch != "." and pred[i] != ch:
                consistent = False
                break
        if consistent:
            consistent_with_gold_givens += 1

    return {
        "num_examples": n,
        "exact_match": exact / max(n, 1),
        "cell_accuracy": total_cell_acc / max(n, 1),
        "format_valid_rate": valid / max(n, 1),
        "valid_sudoku_rate": valid_sudoku / max(n, 1),
        "consistent_with_gold_givens_rate": consistent_with_gold_givens / max(n, 1),
    }


def extract_model_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format")


def load_checkpoint_any(path: str, device: str) -> Any:
    return torch.load(path, map_location=device)


def maybe_load_baseline_metrics(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def build_full_checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    config: FineTuneConfig,
    epoch: int,
    best_exact_match: float,
    best_cell_accuracy: float,
    epochs_without_improvement: int,
    training_log: dict[str, Any],
    best_checkpoint_path: str | None,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_exact_match": best_exact_match,
        "best_cell_accuracy": best_cell_accuracy,
        "epochs_without_improvement": epochs_without_improvement,
        "training_log": training_log,
        "best_checkpoint_path": best_checkpoint_path,
    }


def build_model_only_payload(
    *,
    model,
    config: FineTuneConfig,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }


class SudokuFineTuneDataset(Dataset):
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
        target = row["solution"]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        eos_ids = [self.eos_token_id] if self.eos_token_id is not None else []

        max_prompt_tokens = max(1, self.max_length - len(target_ids) - len(eos_ids))
        prompt_ids = prompt_ids[:max_prompt_tokens]

        input_ids = prompt_ids + target_ids + eos_ids
        labels = ([-100] * len(prompt_ids)) + target_ids + eos_ids
        attention_mask = [1] * len(input_ids)

        return {
            "index": row["index"],
            "puzzle": row["puzzle"],
            "solution": row["solution"],
            "rating": row.get("rating"),
            "nl_description": row["nl_description"],
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": [1] * len(prompt_ids),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_seq = max(len(item["input_ids"]) for item in batch)
        max_prompt = max(len(item["prompt_input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        prompt_input_ids = []
        prompt_attention_mask = []

        for item in batch:
            seq_pad = max_seq - len(item["input_ids"])
            prompt_pad = max_prompt - len(item["prompt_input_ids"])

            input_ids.append(([pad_id] * seq_pad) + item["input_ids"])
            attention_mask.append(([0] * seq_pad) + item["attention_mask"])
            labels.append(([-100] * seq_pad) + item["labels"])
            prompt_input_ids.append(([pad_id] * prompt_pad) + item["prompt_input_ids"])
            prompt_attention_mask.append(([0] * prompt_pad) + item["prompt_attention_mask"])

        return {
            "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
            "puzzle": [item["puzzle"] for item in batch],
            "solution": [item["solution"] for item in batch],
            "rating": [item.get("rating") for item in batch],
            "nl_description": [item["nl_description"] for item in batch],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_input_ids": torch.tensor(prompt_input_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt_attention_mask, dtype=torch.long),
        }

    return collate


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    valid = labels != -100
    if not bool(valid.any()):
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds[valid] == labels[valid]).float().mean().item())


def generate_solution_rows(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generate", leave=False):
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
                predicted_solution, parse_status = parse_solution_from_text(text)
                rows.append(
                    {
                        "index": int(batch["index"][i].item()),
                        "puzzle": batch["puzzle"][i],
                        "solution": batch["solution"][i],
                        "rating": batch["rating"][i],
                        "nl_description": batch["nl_description"][i],
                        "predicted_solution": predicted_solution,
                        "response_text": text,
                        "parse_status": parse_status,
                    }
                )
    return rows


def evaluate_model(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_new_tokens: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_token_acc = 0.0
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

    prediction_rows = generate_solution_rows(model, tokenizer, loader, device, max_new_tokens)
    solution_metrics = compute_solution_metrics(prediction_rows)
    loss_metrics = {
        "loss": total_loss / total_batches,
        "target_token_accuracy": total_token_acc / total_batches,
    }
    return {**loss_metrics, **solution_metrics}, prediction_rows


def load_model_and_tokenizer(base_model_name: str, torch_dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch_dtype)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    return model, tokenizer


def run_train(args) -> None:
    config = FineTuneConfig()
    if args.base_model:
        config.base_model_name = args.base_model
    if args.dataset:
        config.train_dataset_path = args.dataset
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.eval_batch_size:
        config.eval_batch_size = args.eval_batch_size
    if args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.lr:
        config.learning_rate = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.max_length:
        config.max_length = args.max_length
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens
    if args.eval_split is not None:
        config.eval_split = args.eval_split
    if args.seed is not None:
        config.seed = args.seed
    if args.save_every_epoch:
        config.save_every_epoch = args.save_every_epoch
    if args.patience is not None:
        config.patience = args.patience
    if args.grad_clip_norm is not None:
        config.grad_clip_norm = args.grad_clip_norm
    if args.llm_dtype:
        config.llm_dtype = args.llm_dtype

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

    all_records = load_records(config.train_dataset_path)
    train_records, val_records = split_train_val(all_records, config.eval_split, config.seed)
    train_dataset = SudokuFineTuneDataset(train_records, tokenizer, config.max_length)
    val_dataset = SudokuFineTuneDataset(val_records, tokenizer, config.max_length)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, config.gradient_accumulation_steps)))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_scaler = device.type == "cuda" and llm_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = os.path.join(config.log_dir, f"qwen_sudoku_ft_{int(time.time())}.json")
    latest_resume_path = os.path.join(config.checkpoint_dir, "latest_full.pt")

    start_epoch = 0
    best_exact_match = -1.0
    best_cell_accuracy = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path: str | None = None
    training_log: dict[str, Any] = {
        "config": asdict(config),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "epochs": [],
    }

    resume_path = None
    if args.resume_from:
        resume_path = resolve_path(args.resume_from)
    elif not args.no_auto_resume and os.path.isfile(latest_resume_path):
        resume_path = latest_resume_path

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = load_checkpoint_any(resume_path, config.device)
        model.load_state_dict(extract_model_state_dict(checkpoint))
        if isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_exact_match = float(checkpoint.get("best_exact_match", best_exact_match))
        best_cell_accuracy = float(checkpoint.get("best_cell_accuracy", best_cell_accuracy))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        training_log = checkpoint.get("training_log", training_log)
        best_checkpoint_path = checkpoint.get("best_checkpoint_path", best_checkpoint_path)

    use_wandb = args.wandb
    if use_wandb and wandb is None:
        raise ImportError("--wandb was set but package 'wandb' is not installed")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"qwen_sudoku_ft_{int(time.time())}",
            config={
                "base_model": config.base_model_name,
                "dataset": config.train_dataset_path,
                "train_batch_size": config.train_batch_size,
                "eval_batch_size": config.eval_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "epochs": config.epochs,
                "max_length": config.max_length,
                "max_new_tokens": config.max_new_tokens,
                "eval_split": config.eval_split,
                "save_every_epoch": config.save_every_epoch,
                "patience": config.patience,
            },
        )

    print(f"Base model: {config.base_model_name}")
    print(f"Train dataset: {config.train_dataset_path}")
    print(f"Train/val sizes: {len(train_records)}/{len(val_records)}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Using dtype: {llm_dtype}")
    print(f"Steps per epoch: {steps_per_epoch} | total steps: {total_steps}")

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_token_acc = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        progress = tqdm(train_loader, desc=f"Train epoch {epoch + 1}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                loss = outputs.loss / float(max(1, config.gradient_accumulation_steps))

            logits = outputs.logits.detach().float()
            token_acc = masked_token_accuracy(logits, labels)
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % max(1, config.gradient_accumulation_steps) == 0 or step == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * float(max(1, config.gradient_accumulation_steps))
            epoch_token_acc += token_acc
            batch_count += 1
            progress.set_postfix(loss=f"{epoch_loss / batch_count:.4f}", token_acc=f"{epoch_token_acc / batch_count:.4f}")

            if use_wandb and (step % 20 == 0 or step == len(train_loader)):
                current_step = global_step + step
                wandb.log(
                    {
                        "train/loss": epoch_loss / batch_count,
                        "train/target_token_accuracy": epoch_token_acc / batch_count,
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/step": current_step,
                        "epoch": epoch + 1,
                    },
                    step=current_step,
                )

        train_metrics = {
            "loss": epoch_loss / max(1, batch_count),
            "target_token_accuracy": epoch_token_acc / max(1, batch_count),
        }
        val_metrics, _ = evaluate_model(model, tokenizer, val_loader, device, llm_dtype, config.max_new_tokens)
        elapsed = time.time() - t0

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_target_token_accuracy": train_metrics["target_token_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_target_token_accuracy": val_metrics["target_token_accuracy"],
            "val_exact_match": val_metrics["exact_match"],
            "val_cell_accuracy": val_metrics["cell_accuracy"],
            "val_format_valid_rate": val_metrics["format_valid_rate"],
            "val_valid_sudoku_rate": val_metrics["valid_sudoku_rate"],
            "val_consistent_with_gold_givens_rate": val_metrics["consistent_with_gold_givens_rate"],
            "elapsed_seconds": elapsed,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        training_log["epochs"].append(epoch_log)

        print(f"\nEpoch {epoch + 1}/{config.epochs} ({elapsed:.1f}s)")
        print(
            f"  Train: loss={train_metrics['loss']:.4f}, target_token_acc={train_metrics['target_token_accuracy']:.4f}"
        )
        print(
            f"  Val:   loss={val_metrics['loss']:.4f}, exact_match={val_metrics['exact_match']:.4f}, "
            f"cell_acc={val_metrics['cell_accuracy']:.4f}, format_valid={val_metrics['format_valid_rate']:.4f}, "
            f"valid_sudoku={val_metrics['valid_sudoku_rate']:.4f}"
        )

        improved = False
        if val_metrics["exact_match"] > best_exact_match:
            improved = True
        elif val_metrics["exact_match"] == best_exact_match and val_metrics["cell_accuracy"] > best_cell_accuracy:
            improved = True

        if improved:
            best_exact_match = val_metrics["exact_match"]
            best_cell_accuracy = val_metrics["cell_accuracy"]
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save(
                build_model_only_payload(
                    model=model,
                    config=config,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                ),
                best_checkpoint_path,
            )
            print(f"  ↑ New best checkpoint: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  — No improvement ({epochs_without_improvement}/{config.patience})")

        if (epoch + 1) % config.save_every_epoch == 0 or (epoch + 1) == config.epochs:
            full_ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch + 1:03d}.pt")
            payload = build_full_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch + 1,
                best_exact_match=best_exact_match,
                best_cell_accuracy=best_cell_accuracy,
                epochs_without_improvement=epochs_without_improvement,
                training_log=training_log,
                best_checkpoint_path=best_checkpoint_path,
            )
            torch.save(payload, full_ckpt_path)
            torch.save(payload, latest_resume_path)
            print(f"  Saved full checkpoint: {full_ckpt_path}")

        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(training_log, handle, indent=2)

        epoch_end_step = global_step + len(train_loader)

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/target_token_accuracy": train_metrics["target_token_accuracy"],
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "train/epoch_seconds": elapsed,
                    "val/loss": val_metrics["loss"],
                    "val/target_token_accuracy": val_metrics["target_token_accuracy"],
                    "val/exact_match": val_metrics["exact_match"],
                    "val/cell_accuracy": val_metrics["cell_accuracy"],
                    "val/format_valid_rate": val_metrics["format_valid_rate"],
                    "val/valid_sudoku_rate": val_metrics["valid_sudoku_rate"],
                    "val/consistent_with_gold_givens_rate": val_metrics["consistent_with_gold_givens_rate"],
                },
                step=epoch_end_step,
            )

        global_step = epoch_end_step

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    if best_checkpoint_path is None:
        best_checkpoint_path = latest_resume_path if os.path.isfile(latest_resume_path) else ""

    training_log["best_checkpoint"] = best_checkpoint_path
    training_log["stopped_epoch"] = len(training_log["epochs"])
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(training_log, handle, indent=2)

    print(f"Training log: {log_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")

    if use_wandb:
        wandb.summary["best_checkpoint"] = best_checkpoint_path
        wandb.summary["training_log"] = log_path
        wandb.finish()


def run_eval(args) -> None:
    config = FineTuneConfig()
    if args.base_model:
        config.base_model_name = args.base_model
    if args.input_json:
        config.eval_dataset_path = args.input_json
    if args.max_length:
        config.max_length = args.max_length
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens
    if args.eval_batch_size:
        config.eval_batch_size = args.eval_batch_size
    if args.seed is not None:
        config.seed = args.seed
    if args.llm_dtype:
        config.llm_dtype = args.llm_dtype

    config.base_model_name = resolve_path(config.base_model_name) if Path(resolve_path(config.base_model_name)).exists() else config.base_model_name
    config.eval_dataset_path = resolve_path(config.eval_dataset_path)

    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else ""
    predictions_json = resolve_path(args.predictions_json)
    metrics_json = resolve_path(args.metrics_json)
    baseline_metrics = maybe_load_baseline_metrics(args.baseline_metrics_json)

    set_seed(config.seed)
    device = torch.device(config.device)
    llm_dtype = resolve_torch_dtype(config.llm_dtype, config.device)
    model, tokenizer = load_model_and_tokenizer(config.base_model_name, llm_dtype, device)

    if checkpoint_path:
        checkpoint = load_checkpoint_any(checkpoint_path, config.device)
        model.load_state_dict(extract_model_state_dict(checkpoint))

    records = load_records(config.eval_dataset_path, args.max_samples)
    dataset = SudokuFineTuneDataset(records, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer))

    metrics, prediction_rows = evaluate_model(model, tokenizer, loader, device, llm_dtype, config.max_new_tokens)

    predictions_path = Path(predictions_json)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")

    payload: dict[str, Any] = {
        "stage": "solve",
        "input_json": config.eval_dataset_path,
        "base_model": config.base_model_name,
        "checkpoint": checkpoint_path or None,
        "predictions_json": predictions_json,
        "solution_metrics": metrics,
    }

    if baseline_metrics:
        trm_baseline = baseline_metrics.get("trm_metrics", {})
        payload["comparison_to_pipeline"] = {
            "solution_exact_match_delta_vs_trm": metrics["exact_match"] - float(trm_baseline.get("exact_match", 0.0)),
            "solution_cell_accuracy_delta_vs_trm": metrics["cell_accuracy"] - float(trm_baseline.get("cell_accuracy", 0.0)),
        }
        payload["baseline_metrics_json"] = args.baseline_metrics_json
    else:
        payload["baseline_metrics_json"] = None

    metrics_path = Path(metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[eval] wrote predictions: {predictions_path}")
    print(f"[eval] solution metrics: {metrics}")
    print(f"[eval] metrics JSON: {metrics_path}")

    if args.wandb:
        if wandb is None:
            raise ImportError("--wandb requested but wandb is not installed")
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
        run.log({f"solution/{k}": v for k, v in metrics.items()})
        run.summary["predictions_json"] = predictions_json
        run.summary["checkpoint"] = checkpoint_path or "<base-model-only>"
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Qwen on direct Sudoku solving from NL")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Fine-tune Qwen to solve Sudoku from NL descriptions")
    train.add_argument("--base-model", type=str, default="checkpoints/initial/Qwen3-1.7B")
    train.add_argument("--dataset", type=str, default="data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_diverse.json")
    train.add_argument("--checkpoint-dir", type=str, default="checkpoints/initial/qwen_sudoku_ft")
    train.add_argument("--log-dir", type=str, default="logs/initial")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--eval-batch-size", type=int, default=4)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--max-length", type=int, default=768)
    train.add_argument("--max-new-tokens", type=int, default=128)
    train.add_argument("--eval-split", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save-every-epoch", type=int, default=10)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--grad-clip-norm", type=float, default=1.0)
    train.add_argument("--llm-dtype", type=str, default="auto")
    train.add_argument("--resume-from", type=str, default="")
    train.add_argument("--no-auto-resume", action="store_true")
    train.add_argument("--wandb", action="store_true")
    train.add_argument("--wandb-project", type=str, default="trm-llm-qwen-ft")
    train.add_argument("--wandb-run-name", type=str, default=None)
    train.add_argument("--wandb-entity", type=str, default=None)

    evaluate = subparsers.add_parser("eval", help="Evaluate frozen or fine-tuned Qwen on a Sudoku NL test set")
    evaluate.add_argument("--base-model", type=str, default="checkpoints/initial/Qwen3-1.7B")
    evaluate.add_argument("--checkpoint", type=str, default="")
    evaluate.add_argument("--input-json", type=str, default="data/initial/sudoku_synthetic/rule/test_pipeline/sudoku_nl_diverse_v3.json")
    evaluate.add_argument("--predictions-json", type=str, default="results/qwen_sudoku_ft/solver_predictions_diverse_v3.json")
    evaluate.add_argument("--metrics-json", type=str, default="results/qwen_sudoku_ft/solver_metrics_diverse_v3.json")
    evaluate.add_argument("--baseline-metrics-json", type=str, default="")
    evaluate.add_argument("--max-samples", type=int, default=1000)
    evaluate.add_argument("--eval-batch-size", type=int, default=4)
    evaluate.add_argument("--max-length", type=int, default=768)
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--llm-dtype", type=str, default="auto")
    evaluate.add_argument("--wandb", action="store_true")
    evaluate.add_argument("--wandb-project", type=str, default="trm-llm-qwen-ft-eval")
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
