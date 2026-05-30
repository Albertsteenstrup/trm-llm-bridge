#!/usr/bin/env python3
"""Train a benchmark-native CLRS-Text bridge against TRM-facing slot tensors."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parents[1]
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

from clrs_native_text.bridge_native.bridge_hidden_state_cache import load_hidden_state_cache
from clrs_native_text.bridge_native.benchmark_native_slot_bridge import (
    CachedBenchmarkNativeBridgeDataset,
    CLRSTextBenchmarkNativeBridgeDataset,
    NativeTensorBridgeSchema,
    build_benchmark_native_bridge_model,
    build_native_tensor_bridge_schema,
    compute_benchmark_native_bridge_losses,
    compute_benchmark_native_bridge_metrics,
    make_cached_native_tensor_bridge_collate,
    make_native_tensor_bridge_collate,
    _move_native_batch,
)
from clrs_native_text.bridge_native.initial_style_bridge import DirectBridgeConfig


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = dtype_name.lower()
    if name == "auto":
        if device.type == "cuda":
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


def _run_epoch(
    *,
    llm,
    bridge,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    llm_layer_index: int,
    llm_autocast_dtype: torch.dtype,
    schema: NativeTensorBridgeSchema,
    present_threshold: float,
    scalar_exact_tolerance: float,
    grad_clip_norm: float,
    count_blend_alpha: float,
    count_decode_mode: str,
    count_loss_weight_implicit: float,
    count_loss_weight_explicit: float,
    count_loss_weight_blended: float,
    loss_weight_present: float,
    loss_weight_count: float,
    loss_weight_node_scalar: float,
    loss_weight_node_discrete: float,
    loss_weight_edge_scalar: float,
    loss_weight_edge_discrete: float,
    loss_weight_graph_scalar: float,
    loss_weight_graph_discrete: float,
    loss_weight_present_monotonic: float,
    edge_binary_pos_weight_cap: float,
    edge_binary_neg_penalty: float,
    gradient_accumulation_steps: int,
    epoch: int,
    split_name: str,
    log_every_batches: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    bridge.train(mode=train_mode)
    totals: dict[str, float] = {
        "loss/total": 0.0,
        "loss/present": 0.0,
        "loss/present_monotonic": 0.0,
        "loss/count": 0.0,
        "loss/node_scalar": 0.0,
        "loss/node_discrete": 0.0,
        "loss/edge_scalar": 0.0,
        "loss/edge_discrete": 0.0,
        "loss/graph_scalar": 0.0,
        "loss/graph_discrete": 0.0,
        "metric/present_count_acc": 0.0,
        "metric/present_count_mae": 0.0,
        "metric/present_count_acc_threshold": 0.0,
        "metric/present_count_mae_threshold": 0.0,
        "metric/scalar_mae": 0.0,
        "metric/discrete_acc": 0.0,
        "metric/class_acc": 0.0,
        "metric/binary_f1": 0.0,
        "metric/native_exact": 0.0,
        "selection_score": 0.0,
    }
    steps = 0
    total_batches = len(loader)
    grad_accum_steps = max(1, int(gradient_accumulation_steps))
    start_time = time.time()
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        batch = _move_native_batch(batch, device)
        if batch.llm_hidden_states is not None:
            llm_hidden = batch.llm_hidden_states
        else:
            if llm is None:
                raise RuntimeError("Frozen LLM is required when no hidden-state cache is provided")
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=llm_autocast_dtype, enabled=device.type == "cuda"):
                    llm_outputs = llm(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        output_hidden_states=llm_layer_index != -1,
                        use_cache=False,
                    )
                if llm_layer_index == -1:
                    llm_hidden = llm_outputs.last_hidden_state
                else:
                    llm_hidden = llm_outputs.hidden_states[llm_layer_index]

        outputs = bridge(
            llm_hidden.float(),
            batch.attention_mask.bool(),
            batch.task_ids,
            node_anchor_mask=batch.node_anchor_mask,
            edge_text_prior=batch.edge_text_prior,
            node_scalar_prior=batch.node_scalar_prior,
            node_scalar_prior_mask=batch.node_scalar_prior_mask,
            edge_scalar_prior=batch.edge_scalar_prior,
            edge_scalar_prior_mask=batch.edge_scalar_prior_mask,
            count_prior=batch.count_prior,
            count_prior_mask=batch.count_prior_mask,
        )
        total_loss, loss_parts = compute_benchmark_native_bridge_losses(
            outputs,
            batch,
            schema=schema,
            count_blend_alpha=count_blend_alpha,
            count_loss_weight_implicit=count_loss_weight_implicit,
            count_loss_weight_explicit=count_loss_weight_explicit,
            count_loss_weight_blended=count_loss_weight_blended,
            loss_weight_present=loss_weight_present,
            loss_weight_count=loss_weight_count,
            loss_weight_node_scalar=loss_weight_node_scalar,
            loss_weight_node_discrete=loss_weight_node_discrete,
            loss_weight_edge_scalar=loss_weight_edge_scalar,
            loss_weight_edge_discrete=loss_weight_edge_discrete,
            loss_weight_graph_scalar=loss_weight_graph_scalar,
            loss_weight_graph_discrete=loss_weight_graph_discrete,
            loss_weight_present_monotonic=loss_weight_present_monotonic,
            edge_binary_pos_weight_cap=edge_binary_pos_weight_cap,
            edge_binary_neg_penalty=edge_binary_neg_penalty,
        )
        metrics = compute_benchmark_native_bridge_metrics(
            outputs,
            batch,
            schema=schema,
            count_blend_alpha=count_blend_alpha,
            count_decode_mode=count_decode_mode,
            present_threshold=present_threshold,
            scalar_exact_tolerance=scalar_exact_tolerance,
        )

        if train_mode:
            (total_loss / float(grad_accum_steps)).backward()
            if (steps + 1) % grad_accum_steps == 0:
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        totals["loss/total"] += float(total_loss.item())
        for key, value in loss_parts.items():
            if key not in totals:
                totals[key] = 0.0
            totals[key] += float(value)
        for key, value in metrics.items():
            if key not in totals:
                totals[key] = 0.0
            totals[key] += float(value)
        steps += 1
        if log_every_batches > 0 and ((steps % log_every_batches) == 0 or steps == total_batches):
            elapsed = max(1e-6, time.time() - start_time)
            batches_per_sec = float(steps) / elapsed
            eta_batches = max(0, total_batches - steps)
            eta_seconds = eta_batches / max(1e-6, batches_per_sec)
            print(
                f"{split_name} epoch={epoch:03d} batch={steps}/{total_batches} "
                f"loss={float(total_loss.item()):.4f} "
                f"batches_per_sec={batches_per_sec:.2f} "
                f"eta_min={eta_seconds / 60.0:.1f}",
                flush=True,
            )

    if train_mode and steps > 0 and (steps % grad_accum_steps) != 0:
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if steps == 0:
        raise RuntimeError("No training/eval batches were produced")

    return {key: value / float(steps) for key, value in totals.items()}


class ShardAwareBatchSampler(BatchSampler):
    def __init__(
        self,
        sample_to_shard: list[int],
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, list[int]] = defaultdict(list)
        for sample_idx, shard_idx in enumerate(sample_to_shard):
            groups[int(shard_idx)].append(int(sample_idx))
        self.indices_by_shard = {shard_idx: indices for shard_idx, indices in sorted(groups.items())}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_ids = list(self.indices_by_shard.keys())
        if self.shuffle:
            rng.shuffle(shard_ids)
        for shard_idx in shard_ids:
            indices = list(self.indices_by_shard[shard_idx])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.indices_by_shard.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _checkpoint_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Prefer native exact first, then broader bridge quality, then scalar fidelity."""
    return (
        float(metrics["metric/native_exact"]),
        float(metrics["selection_score"]),
        -float(metrics["metric/scalar_mae"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a benchmark-native CLRS-Text bridge")
    parser.add_argument("--arch", choices=["qformer", "transnar"], default="qformer")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--llm-layer-index", type=int, default=-1)
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--train-hidden-cache", default="")
    parser.add_argument("--val-hidden-cache", default="")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--bridge-dim", type=int, default=512)
    parser.add_argument("--bridge-heads", type=int, default=8)
    parser.add_argument("--bridge-layers", type=int, default=4)
    parser.add_argument("--bridge-dropout", type=float, default=0.1)
    parser.add_argument("--query-pos-dropout", type=float, default=0.05)
    parser.add_argument("--count-blend-alpha", type=float, default=0.5)
    parser.add_argument("--count-decode-mode", choices=["blended", "head_only", "implicit", "prior"], default="blended")
    parser.add_argument("--scalar-head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--scalar-head-hidden-mult", type=float, default=1.0)
    parser.add_argument("--qformer-edge-self-bias", action="store_true")
    parser.add_argument("--qformer-rel-pos-self-bias", action="store_true")
    parser.add_argument("--edge-text-prior-logit-scale", type=float, default=0.0)
    parser.add_argument("--transnar-recurrence-steps", type=int, default=1)
    parser.add_argument("--feature-quant-bins", type=int, default=256)
    parser.add_argument("--continuous-feature-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--scalar-exact-tolerance", type=float, default=0.25)
    parser.add_argument("--count-loss-weight-implicit", type=float, default=0.25)
    parser.add_argument("--count-loss-weight-explicit", type=float, default=0.25)
    parser.add_argument("--count-loss-weight-blended", type=float, default=0.50)
    parser.add_argument("--loss-weight-present", type=float, default=0.3)
    parser.add_argument("--loss-weight-count", type=float, default=0.1)
    parser.add_argument("--loss-weight-node-scalar", type=float, default=0.5)
    parser.add_argument("--loss-weight-node-discrete", type=float, default=2.0)
    parser.add_argument("--loss-weight-edge-scalar", type=float, default=0.3)
    parser.add_argument("--loss-weight-edge-discrete", type=float, default=0.8)
    parser.add_argument("--edge-binary-pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--edge-binary-neg-penalty", type=float, default=0.0)
    parser.add_argument("--loss-weight-graph-scalar", type=float, default=0.2)
    parser.add_argument("--loss-weight-graph-discrete", type=float, default=0.5)
    parser.add_argument("--loss-weight-present-monotonic", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=1000)
    parser.add_argument("--resume-from", default="")
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load model weights from --resume-from but reset optimizer, scheduler, epoch counters, and metric history.",
    )
    parser.add_argument("--wandb-project", default="clrs-text-bridge-benchmark-native")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--allow-non-strict-native", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    strict_native_only = not bool(args.allow_non_strict_native)
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_dtype = _resolve_torch_dtype(args.llm_dtype, device)
    train_hidden_cache = load_hidden_state_cache(Path(args.train_hidden_cache).resolve()) if args.train_hidden_cache else None
    val_hidden_cache = load_hidden_state_cache(Path(args.val_hidden_cache).resolve()) if args.val_hidden_cache else None
    use_hidden_cache = train_hidden_cache is not None or val_hidden_cache is not None
    if use_hidden_cache and (train_hidden_cache is None or val_hidden_cache is None):
        raise ValueError("Both --train-hidden-cache and --val-hidden-cache are required when using hidden-state caching")

    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore

    train_rows = _read_jsonl(Path(args.train_jsonl).resolve())
    val_rows = _read_jsonl(Path(args.val_jsonl).resolve())
    if int(args.max_train_rows) > 0:
        train_rows = train_rows[: int(args.max_train_rows)]
    if int(args.max_val_rows) > 0:
        val_rows = val_rows[: int(args.max_val_rows)]
    if not train_rows:
        raise ValueError(f"Loaded 0 train rows from {args.train_jsonl}. Please verify the file is not empty or corrupted.")
    if not val_rows:
        raise ValueError(f"Loaded 0 val rows from {args.val_jsonl}. Please verify the file is not empty or corrupted.")
    schema, schema_stats = build_native_tensor_bridge_schema(
        train_rows + val_rows,
        strict_native_only=strict_native_only,
        feature_quant_bins=args.feature_quant_bins,
    )

    if use_hidden_cache:
        if train_hidden_cache.llm_hidden_size != val_hidden_cache.llm_hidden_size:
            raise ValueError("Train/val hidden-state caches disagree on llm_hidden_size")
        if train_hidden_cache.llm_model_name != val_hidden_cache.llm_model_name:
            raise ValueError("Train/val hidden-state caches disagree on llm_model_name")
        if train_hidden_cache.llm_layer_index != val_hidden_cache.llm_layer_index:
            raise ValueError("Train/val hidden-state caches disagree on llm_layer_index")
        if len(train_hidden_cache.sample_ids) != len(train_rows):
            raise ValueError("Train hidden-state cache row count does not match train JSONL")
        if len(val_hidden_cache.sample_ids) != len(val_rows):
            raise ValueError("Val hidden-state cache row count does not match val JSONL")
        llm_model_name = train_hidden_cache.llm_model_name
        llm_layer_index = int(train_hidden_cache.llm_layer_index)
        llm_hidden_size = int(train_hidden_cache.llm_hidden_size)
        tokenizer = None
        llm = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # The bridge only consumes hidden states. Loading the base model avoids
        # computing LM logits through the large lm_head on every frozen forward.
        llm = AutoModel.from_pretrained(args.llm_model_name, torch_dtype=llm_dtype)
        llm.to(device)
        llm.eval()
        for param in llm.parameters():
            param.requires_grad = False
        llm_model_name = str(args.llm_model_name)
        llm_layer_index = int(args.llm_layer_index)
        llm_hidden_size = int(getattr(llm.config, "hidden_size"))

    bridge_config = DirectBridgeConfig(
        llm_hidden_size=llm_hidden_size,
        max_nodes=args.max_nodes,
        bridge_dim=args.bridge_dim,
        bridge_heads=args.bridge_heads,
        bridge_layers=args.bridge_layers,
        bridge_dropout=args.bridge_dropout,
        query_pos_dropout=args.query_pos_dropout,
        count_blend_alpha=args.count_blend_alpha,
        count_decode_mode=args.count_decode_mode,
        scalar_head_type=args.scalar_head_type,
        scalar_head_hidden_mult=args.scalar_head_hidden_mult,
        qformer_edge_self_bias=bool(args.qformer_edge_self_bias),
        qformer_rel_pos_self_bias=bool(args.qformer_rel_pos_self_bias),
        edge_text_prior_logit_scale=float(args.edge_text_prior_logit_scale),
        transnar_recurrence_steps=args.transnar_recurrence_steps,
        arch=args.arch,
    )
    bridge = build_benchmark_native_bridge_model(args.arch, bridge_config, schema=schema).to(device)
    bridge_trainable_params = sum(param.numel() for param in bridge.parameters() if param.requires_grad)
    print(
        f"bridge arch={args.arch} trainable_params={bridge_trainable_params:,} "
        f"count_blend_alpha={args.count_blend_alpha:g} "
        f"count_decode_mode={args.count_decode_mode} "
        f"count_loss_mix=({args.count_loss_weight_implicit:g},{args.count_loss_weight_explicit:g},{args.count_loss_weight_blended:g}) "
        f"scalar_head_type={args.scalar_head_type} scalar_head_hidden_mult={args.scalar_head_hidden_mult:g} "
        f"qformer_edge_self_bias={bool(args.qformer_edge_self_bias)} "
        f"qformer_rel_pos_self_bias={bool(args.qformer_rel_pos_self_bias)} "
        f"edge_text_prior_logit_scale={args.edge_text_prior_logit_scale:g} "
        f"node_slots={schema.node_slot_dim} edge_slots={schema.edge_slot_dim} graph_slots={schema.graph_slot_dim} "
        f"strict_native_only={strict_native_only} max_nodes={args.max_nodes}"
    )
    print(
        "native schema filter stats: "
        f"kept={schema_stats.get('kept_rows', 0)} dropped={schema_stats.get('dropped_rows', 0)} "
        f"adapter_versions={schema_stats.get('adapter_versions', {})}"
    )

    train_base_ds = CLRSTextBenchmarkNativeBridgeDataset(
        train_rows,
        schema=schema,
        max_nodes=args.max_nodes,
        strict_native_only=strict_native_only,
        continuous_feature_clip=args.continuous_feature_clip,
        feature_quant_bins=args.feature_quant_bins,
    )
    val_base_ds = CLRSTextBenchmarkNativeBridgeDataset(
        val_rows,
        schema=schema,
        max_nodes=args.max_nodes,
        strict_native_only=strict_native_only,
        continuous_feature_clip=args.continuous_feature_clip,
        feature_quant_bins=args.feature_quant_bins,
    )
    print(
        "native dataset filter stats: "
        f"train={train_base_ds.filter_stats} val={val_base_ds.filter_stats}"
    )

    if use_hidden_cache:
        train_ds = CachedBenchmarkNativeBridgeDataset(train_base_ds, cache=train_hidden_cache)
        val_ds = CachedBenchmarkNativeBridgeDataset(val_base_ds, cache=val_hidden_cache)
        collate_fn = make_cached_native_tensor_bridge_collate(
            schema=schema,
            continuous_feature_clip=args.continuous_feature_clip,
        )
    else:
        train_ds = train_base_ds
        val_ds = val_base_ds
        collate_fn = make_native_tensor_bridge_collate(
            tokenizer,
            max_source_length=args.max_source_length,
            schema=schema,
            continuous_feature_clip=args.continuous_feature_clip,
        )
    train_batch_sampler = None
    if use_hidden_cache and train_hidden_cache.is_sharded():
        train_batch_sampler = ShardAwareBatchSampler(
            train_hidden_cache.sample_to_shard,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs // max(1, args.gradient_accumulation_steps))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(
        f"train_samples={len(train_ds):,} val_samples={len(val_ds):,} "
        f"train_batches={len(train_loader):,} val_batches={len(val_loader):,} "
        f"optimizer_steps_total={total_steps:,} grad_accum={args.gradient_accumulation_steps} "
        f"use_hidden_cache={use_hidden_cache} train_cache_sharded={bool(use_hidden_cache and train_hidden_cache.is_sharded())}",
        flush=True,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = Path(args.metrics_json).resolve() if args.metrics_json else output_dir / "metrics.json"
    best_checkpoint = output_dir / "best.pt"
    last_checkpoint = output_dir / "last.pt"

    best_selection_score = -1e9
    best_native_exact = -1e9
    best_scalar_mae = 1e9
    best_checkpoint_key = (-1e9, -1e9, -1e9)
    best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 1
    history: list[dict[str, Any]] = []

    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        checkpoint = torch.load(resume_path, map_location=device)
        bridge.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_weights_only:
            print(f"loaded model weights only from {resume_path}", flush=True)
        else:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_selection_score = float(checkpoint.get("best_selection_score", best_selection_score))
            best_native_exact = float(checkpoint.get("best_native_exact", best_native_exact))
            best_scalar_mae = float(checkpoint.get("best_scalar_mae", best_scalar_mae))
            if "best_checkpoint_key" in checkpoint:
                saved_key = checkpoint["best_checkpoint_key"]
                best_checkpoint_key = tuple(float(v) for v in saved_key)
            else:
                best_checkpoint_key = (
                    best_native_exact,
                    best_selection_score,
                    -best_scalar_mae,
                )
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            epochs_no_improve = int(checkpoint.get("epochs_no_improve", epochs_no_improve))
            history = list(checkpoint.get("history", history))

    run = None
    if not args.no_wandb:
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("wandb is required unless --no-wandb is set") from exc
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            job_type="bridge",
            config={
                "arch": args.arch,
                "llm_model_name": llm_model_name,
                "llm_layer_index": llm_layer_index,
                "use_hidden_cache": bool(use_hidden_cache),
                "bridge_trainable_params": bridge_trainable_params,
                "task_count": len(schema.task_names),
                "node_slot_dim": schema.node_slot_dim,
                "edge_slot_dim": schema.edge_slot_dim,
                "graph_slot_dim": schema.graph_slot_dim,
                "strict_native_only": strict_native_only,
                "feature_quant_bins": args.feature_quant_bins,
                "max_source_length": args.max_source_length,
                "max_nodes": args.max_nodes,
                "bridge_dim": args.bridge_dim,
                "bridge_heads": args.bridge_heads,
                "bridge_layers": args.bridge_layers,
                "bridge_dropout": args.bridge_dropout,
                "query_pos_dropout": args.query_pos_dropout,
                "count_blend_alpha": args.count_blend_alpha,
                "count_decode_mode": args.count_decode_mode,
                "count_loss_weight_implicit": args.count_loss_weight_implicit,
                "count_loss_weight_explicit": args.count_loss_weight_explicit,
                "count_loss_weight_blended": args.count_loss_weight_blended,
                "scalar_head_type": args.scalar_head_type,
                "scalar_head_hidden_mult": args.scalar_head_hidden_mult,
                "qformer_edge_self_bias": bool(args.qformer_edge_self_bias),
                "qformer_rel_pos_self_bias": bool(args.qformer_rel_pos_self_bias),
                "edge_text_prior_logit_scale": args.edge_text_prior_logit_scale,
                "edge_binary_pos_weight_cap": args.edge_binary_pos_weight_cap,
                "edge_binary_neg_penalty": args.edge_binary_neg_penalty,
                "max_train_rows": args.max_train_rows,
                "max_val_rows": args.max_val_rows,
                "transnar_recurrence_steps": args.transnar_recurrence_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            },
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            llm=llm,
            bridge=bridge,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            llm_layer_index=llm_layer_index,
            llm_autocast_dtype=llm_dtype,
            schema=schema,
            present_threshold=args.present_threshold,
            scalar_exact_tolerance=args.scalar_exact_tolerance,
            grad_clip_norm=args.grad_clip_norm,
            count_blend_alpha=args.count_blend_alpha,
            count_decode_mode=args.count_decode_mode,
            count_loss_weight_implicit=args.count_loss_weight_implicit,
            count_loss_weight_explicit=args.count_loss_weight_explicit,
            count_loss_weight_blended=args.count_loss_weight_blended,
            loss_weight_present=args.loss_weight_present,
            loss_weight_count=args.loss_weight_count,
            loss_weight_node_scalar=args.loss_weight_node_scalar,
            loss_weight_node_discrete=args.loss_weight_node_discrete,
            loss_weight_edge_scalar=args.loss_weight_edge_scalar,
            loss_weight_edge_discrete=args.loss_weight_edge_discrete,
            loss_weight_graph_scalar=args.loss_weight_graph_scalar,
            loss_weight_graph_discrete=args.loss_weight_graph_discrete,
            loss_weight_present_monotonic=args.loss_weight_present_monotonic,
            edge_binary_pos_weight_cap=args.edge_binary_pos_weight_cap,
            edge_binary_neg_penalty=args.edge_binary_neg_penalty,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epoch=epoch,
            split_name="train",
            log_every_batches=args.log_every_batches,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                llm=llm,
                bridge=bridge,
                loader=val_loader,
                optimizer=None,
                scheduler=None,
                device=device,
                llm_layer_index=llm_layer_index,
                llm_autocast_dtype=llm_dtype,
                schema=schema,
                present_threshold=args.present_threshold,
                scalar_exact_tolerance=args.scalar_exact_tolerance,
                grad_clip_norm=0.0,
                count_blend_alpha=args.count_blend_alpha,
                count_decode_mode=args.count_decode_mode,
                count_loss_weight_implicit=args.count_loss_weight_implicit,
                count_loss_weight_explicit=args.count_loss_weight_explicit,
                count_loss_weight_blended=args.count_loss_weight_blended,
                loss_weight_present=args.loss_weight_present,
                loss_weight_count=args.loss_weight_count,
                loss_weight_node_scalar=args.loss_weight_node_scalar,
                loss_weight_node_discrete=args.loss_weight_node_discrete,
                loss_weight_edge_scalar=args.loss_weight_edge_scalar,
                loss_weight_edge_discrete=args.loss_weight_edge_discrete,
                loss_weight_graph_scalar=args.loss_weight_graph_scalar,
                loss_weight_graph_discrete=args.loss_weight_graph_discrete,
                loss_weight_present_monotonic=args.loss_weight_present_monotonic,
                edge_binary_pos_weight_cap=args.edge_binary_pos_weight_cap,
                edge_binary_neg_penalty=args.edge_binary_neg_penalty,
                gradient_accumulation_steps=1,
                epoch=epoch,
                split_name="val",
                log_every_batches=args.log_every_batches,
            )

        history_row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(history_row)

        current_checkpoint_key = _checkpoint_key(val_metrics)
        improved = current_checkpoint_key > best_checkpoint_key
        if improved:
            best_checkpoint_key = current_checkpoint_key
            best_selection_score = float(val_metrics["selection_score"])
            best_native_exact = float(val_metrics["metric/native_exact"])
            best_scalar_mae = float(val_metrics["metric/scalar_mae"])
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "bridge_family": "benchmark_native_slot_bridge_v1",
                    "arch": args.arch,
                    "llm_model_name": llm_model_name,
                    "llm_layer_index": llm_layer_index,
                    "bridge_config": asdict(bridge_config),
                    "schema": schema.to_dict(),
                    "feature_quant_bins": int(args.feature_quant_bins),
                    "continuous_feature_clip": float(args.continuous_feature_clip),
                    "strict_native_only": bool(strict_native_only),
                    "epoch": epoch,
                    "best_checkpoint_key": list(best_checkpoint_key),
                    "best_selection_score": best_selection_score,
                    "best_native_exact": best_native_exact,
                    "best_scalar_mae": best_scalar_mae,
                    "best_epoch": best_epoch,
                    "model_state_dict": bridge.state_dict(),
                },
                best_checkpoint,
            )
        else:
            epochs_no_improve += 1

        if args.checkpoint_every_epochs > 0 and (epoch % args.checkpoint_every_epochs == 0):
            torch.save(
                {
                    "bridge_family": "benchmark_native_slot_bridge_v1",
                    "arch": args.arch,
                    "llm_model_name": llm_model_name,
                    "llm_layer_index": llm_layer_index,
                    "bridge_config": asdict(bridge_config),
                    "schema": schema.to_dict(),
                    "feature_quant_bins": int(args.feature_quant_bins),
                    "continuous_feature_clip": float(args.continuous_feature_clip),
                    "strict_native_only": bool(strict_native_only),
                    "epoch": epoch,
                    "model_state_dict": bridge.state_dict(),
                },
                output_dir / f"{args.arch}_epoch{epoch:03d}.pt",
            )

        torch.save(
            {
                "bridge_family": "benchmark_native_slot_bridge_v1",
                "arch": args.arch,
                "llm_model_name": llm_model_name,
                "llm_layer_index": llm_layer_index,
                "bridge_config": asdict(bridge_config),
                "schema": schema.to_dict(),
                "feature_quant_bins": int(args.feature_quant_bins),
                "continuous_feature_clip": float(args.continuous_feature_clip),
                "strict_native_only": bool(strict_native_only),
                "epoch": epoch,
                "best_checkpoint_key": list(best_checkpoint_key),
                "best_selection_score": best_selection_score,
                "best_native_exact": best_native_exact,
                "best_scalar_mae": best_scalar_mae,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "model_state_dict": bridge.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
            },
            last_checkpoint,
        )

        summary = {
            "bridge_family": "benchmark_native_slot_bridge_v1",
            "checkpoint_selection": "native_exact_then_selection_then_scalar_mae",
            "best_checkpoint_key": list(best_checkpoint_key),
            "best_selection_score": best_selection_score,
            "best_native_exact": best_native_exact,
            "best_scalar_mae": best_scalar_mae,
            "best_epoch": best_epoch,
            "history": history,
            "arch": args.arch,
            "llm_model_name": llm_model_name,
            "llm_layer_index": llm_layer_index,
            "use_hidden_cache": bool(use_hidden_cache),
            "bridge_config": asdict(bridge_config),
            "schema": schema.to_dict(),
            "feature_quant_bins": int(args.feature_quant_bins),
            "continuous_feature_clip": float(args.continuous_feature_clip),
            "strict_native_only": bool(strict_native_only),
            "schema_build_stats": schema_stats,
            "train_filter_stats": train_base_ds.filter_stats,
            "val_filter_stats": val_base_ds.filter_stats,
            "output_dir": str(output_dir),
        }
        _write_json(metrics_json, summary)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss/total']:.4f} "
            f"val_native_exact={val_metrics['metric/native_exact']:.4f} "
            f"val_scalar_mae={val_metrics['metric/scalar_mae']:.4f} "
            f"val_class_acc={val_metrics['metric/class_acc']:.4f} "
            f"val_binary_f1={val_metrics['metric/binary_f1']:.4f} "
            f"val_selection={val_metrics['selection_score']:.4f}"
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss/total"],
                    "train/loss_present_monotonic": train_metrics["loss/present_monotonic"],
                    "train/present_count_acc": train_metrics["metric/present_count_acc"],
                    "train/present_count_acc_implicit": train_metrics["metric/present_count_acc_implicit"],
                    "train/present_count_acc_head": train_metrics["metric/present_count_acc_head"],
                    "train/present_count_acc_threshold": train_metrics["metric/present_count_acc_threshold"],
                    "train/scalar_mae": train_metrics["metric/scalar_mae"],
                    "train/discrete_acc": train_metrics["metric/discrete_acc"],
                    "train/class_acc": train_metrics["metric/class_acc"],
                    "train/binary_f1": train_metrics["metric/binary_f1"],
                    "train/native_exact": train_metrics["metric/native_exact"],
                    "train/selection_score": train_metrics["selection_score"],
                    "val/loss": val_metrics["loss/total"],
                    "val/loss_present_monotonic": val_metrics["loss/present_monotonic"],
                    "val/present_count_acc": val_metrics["metric/present_count_acc"],
                    "val/present_count_acc_implicit": val_metrics["metric/present_count_acc_implicit"],
                    "val/present_count_acc_head": val_metrics["metric/present_count_acc_head"],
                    "val/present_count_acc_threshold": val_metrics["metric/present_count_acc_threshold"],
                    "val/scalar_mae": val_metrics["metric/scalar_mae"],
                    "val/discrete_acc": val_metrics["metric/discrete_acc"],
                    "val/class_acc": val_metrics["metric/class_acc"],
                    "val/binary_f1": val_metrics["metric/binary_f1"],
                    "val/native_exact": val_metrics["metric/native_exact"],
                    "val/selection_score": val_metrics["selection_score"],
                    "checkpoint/native_exact_priority_score": best_checkpoint_key[0] + (1e-3 * best_checkpoint_key[1]),
                }
            )
            run.summary["checkpoint_selection"] = "native_exact_then_selection_then_scalar_mae"
            run.summary["best_checkpoint_key"] = list(best_checkpoint_key)
            run.summary["best_selection_score"] = best_selection_score
            run.summary["best_native_exact"] = best_native_exact
            run.summary["best_scalar_mae"] = best_scalar_mae
            run.summary["best_epoch"] = best_epoch
            run.summary["best_checkpoint"] = str(best_checkpoint)
            run.summary["last_checkpoint"] = str(last_checkpoint)

        if epochs_no_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no improvement in native_exact/selection/scalar checkpoint key for "
                f"{epochs_no_improve} epochs."
            )
            break

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
