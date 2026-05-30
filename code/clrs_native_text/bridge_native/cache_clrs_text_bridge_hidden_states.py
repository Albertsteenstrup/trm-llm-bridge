#!/usr/bin/env python3
"""Cache frozen-LM hidden states for CLRS-Text bridge training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parents[1]
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from clrs_native_text.bridge_native.bridge_hidden_state_cache import (
    BridgeHiddenStateCache,
    V1_FORMAT,
    save_hidden_state_cache,
    save_hidden_state_cache_sharded,
)
from clrs_native_text.bridge_native.train_clrs_text_bridge_benchmark_native import _resolve_torch_dtype


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QuestionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        row = self.rows[idx]
        return {
            "question": str(row.get("question", "")),
            "sample_id": str(row.get("sample_id", idx)),
        }


def make_question_collate(tokenizer, *, max_source_length: int):
    def collate(batch: list[dict[str, str]]) -> dict[str, Any]:
        enc = tokenizer(
            [item["question"] for item in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_source_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "sample_ids": [item["sample_id"] for item in batch],
        }

    return collate


def _resolve_cache_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    lowered = name.lower()
    if lowered not in mapping:
        raise ValueError(f"Unsupported cache dtype: {name}")
    return mapping[lowered]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen-LM hidden states for CLRS-Text bridge training")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--llm-layer-index", type=int, default=-1)
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--cache-dtype", default="float16")
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size-rows", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_path).resolve()
    rows = _read_jsonl(input_path)
    device = torch.device(args.device)
    llm_dtype = _resolve_torch_dtype(args.llm_dtype, device)
    cache_dtype = _resolve_cache_dtype(args.cache_dtype)

    from transformers import AutoModel, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The bridge only needs hidden states. Loading the base model avoids the
    # expensive LM head/logits path in AutoModelForCausalLM.
    llm = AutoModel.from_pretrained(args.llm_model_name, torch_dtype=llm_dtype).to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    ds = QuestionDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_question_collate(tokenizer, max_source_length=args.max_source_length),
    )

    sample_ids: list[str] = []
    lengths: list[int] = []
    sample_to_shard: list[int] = []
    sample_to_local: list[int] = []
    shard_files: list[str] = []
    shard_sample_ids: list[str] = []
    shard_lengths: list[int] = []
    shard_offsets: list[int] = [0]
    shard_hidden_chunks: list[torch.Tensor] = []
    shard_idx = 0
    processed = 0

    def flush_shard() -> None:
        nonlocal shard_sample_ids, shard_lengths, shard_offsets, shard_hidden_chunks, shard_idx
        if not shard_sample_ids:
            return
        shard_name = f"part_{shard_idx:05d}.pt"
        shard_dir = Path(f"{output_path}.shards")
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / shard_name
        hidden_states = torch.cat(shard_hidden_chunks, dim=0)
        shard_cache = BridgeHiddenStateCache(
            cache_format=V1_FORMAT,
            source_jsonl=str(input_path),
            llm_model_name=str(args.llm_model_name),
            llm_layer_index=int(args.llm_layer_index),
            llm_hidden_size=int(getattr(llm.config, "hidden_size")),
            max_source_length=int(args.max_source_length),
            cache_dtype=str(cache_dtype).replace("torch.", ""),
            sample_ids=list(shard_sample_ids),
            lengths=list(shard_lengths),
            offsets=list(shard_offsets),
            hidden_states=hidden_states,
        )
        save_hidden_state_cache(shard_path, shard_cache)
        shard_files.append(shard_name)
        shard_idx += 1
        shard_sample_ids = []
        shard_lengths = []
        shard_offsets = [0]
        shard_hidden_chunks = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                outputs = llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = outputs.hidden_states[int(args.llm_layer_index)].detach()
            for row_idx, sample_id in enumerate(batch["sample_ids"]):
                length = int(attention_mask[row_idx].sum().item())
                sample_hidden = hidden[row_idx, :length].to(dtype=cache_dtype).cpu().contiguous()
                sample_ids.append(str(sample_id))
                lengths.append(length)
                sample_to_shard.append(shard_idx)
                sample_to_local.append(len(shard_sample_ids))
                shard_sample_ids.append(str(sample_id))
                shard_lengths.append(length)
                shard_hidden_chunks.append(sample_hidden)
                shard_offsets.append(shard_offsets[-1] + length)
                processed += 1
                if args.progress_every > 0 and (processed % args.progress_every == 0 or processed == len(rows)):
                    print(f"[{processed}/{len(rows)}] cached hidden states", flush=True)
                if len(shard_sample_ids) >= int(max(1, args.shard_size_rows)):
                    flush_shard()

    flush_shard()

    save_hidden_state_cache_sharded(
        output_path,
        source_jsonl=str(input_path),
        llm_model_name=str(args.llm_model_name),
        llm_layer_index=int(args.llm_layer_index),
        llm_hidden_size=int(getattr(llm.config, "hidden_size")),
        max_source_length=int(args.max_source_length),
        cache_dtype=str(cache_dtype).replace("torch.", ""),
        sample_ids=sample_ids,
        lengths=lengths,
        shard_files=shard_files,
        sample_to_shard=sample_to_shard,
        sample_to_local=sample_to_local,
    )
    print(
        f"Saved sharded hidden-state cache: rows={len(sample_ids)} shards={len(shard_files)} "
        f"hidden_size={int(getattr(llm.config, 'hidden_size'))} path={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
