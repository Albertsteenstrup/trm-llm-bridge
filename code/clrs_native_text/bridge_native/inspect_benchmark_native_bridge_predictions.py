#!/usr/bin/env python3
"""Inspect benchmark-native bridge predictions from a saved checkpoint."""

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
from torch.utils.data import DataLoader

from clrs_native_text.bridge_native.benchmark_native_slot_bridge import (
    CLRSTextBenchmarkNativeBridgeDataset,
    NativeTensorBridgeSchema,
    _apply_node_mask_one_predictions,
    _binary_logits_from_discrete_logits,
    _count_class_predictions,
    _count_prior_predictions,
    _mask_f1,
    _mixed_discrete_predictions,
    _move_native_batch,
    _normalized_count_predictions,
    _schema_family_masks,
    _select_decoded_count_norm,
    build_benchmark_native_bridge_model,
    make_native_tensor_bridge_collate,
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


def _slot_index(schema_slots: list[Any], *, name: str, value_type: str | None = None) -> int | None:
    for idx, slot in enumerate(schema_slots):
        if slot.name == name and (value_type is None or slot.value_type == value_type):
            return idx
    return None


def _new_adj_totals() -> dict[str, float]:
    return {
        "adj_exact": 0.0,
        "adj_f1": 0.0,
        "adj_pred_present": 0.0,
        "adj_true_present": 0.0,
        "adj_total": 0.0,
    }


def _update_adj_totals(totals: dict[str, float], pred_adj: torch.Tensor, true_adj: torch.Tensor) -> None:
    pred_bool = pred_adj.bool()
    true_bool = true_adj.bool()
    totals["adj_f1"] += float(_mask_f1(pred_adj.reshape(-1), true_adj.reshape(-1)))
    totals["adj_exact"] += float(torch.equal(pred_adj, true_adj))
    totals["adj_pred_present"] += float(pred_bool.float().sum().item())
    totals["adj_true_present"] += float(true_bool.float().sum().item())
    totals["adj_total"] += float(max(1, pred_adj.numel()))


def _slot_active(active_by_task: list[list[float]], task_idx: int, slot_idx: int | None) -> bool:
    return (
        slot_idx is not None
        and 0 <= task_idx < len(active_by_task)
        and 0 <= slot_idx < len(active_by_task[task_idx])
        and float(active_by_task[task_idx][slot_idx]) > 0.5
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect native bridge checkpoint predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--edge-logit-threshold", type=float, default=0.0)
    parser.add_argument("--edge-logit-thresholds", default="")
    parser.add_argument("--edge-text-prior-logit-scale", type=float, default=None)
    parser.add_argument("--count-blend-alpha", type=float, default=0.5)
    parser.add_argument("--count-decode-mode", choices=["blended", "head_only", "implicit", "prior"], default="blended")
    parser.add_argument("--scalar-exact-tolerance", type=float, default=0.25)
    parser.add_argument("--llm-dtype", default="auto")
    args = parser.parse_args()
    sweep_thresholds = [
        float(item.strip())
        for item in str(args.edge_logit_thresholds).split(",")
        if item.strip()
    ]

    from transformers import AutoModel, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device)
    schema = NativeTensorBridgeSchema.from_dict(checkpoint["schema"])
    config = DirectBridgeConfig(**checkpoint["bridge_config"])
    if args.edge_text_prior_logit_scale is not None:
        config.edge_text_prior_logit_scale = float(args.edge_text_prior_logit_scale)
    arch = str(checkpoint.get("arch", config.arch))

    bridge = build_benchmark_native_bridge_model(arch, config, schema=schema).to(device)
    bridge.load_state_dict(checkpoint["model_state_dict"])
    bridge.eval()

    llm_model_name = str(checkpoint["llm_model_name"])
    llm_layer_index = int(checkpoint.get("llm_layer_index", -1))
    llm_dtype = _resolve_torch_dtype(args.llm_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModel.from_pretrained(llm_model_name, torch_dtype=llm_dtype).to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    rows = _read_jsonl(Path(args.jsonl).resolve())
    if int(args.limit_samples) > 0:
        rows = rows[: int(args.limit_samples)]
    dataset = CLRSTextBenchmarkNativeBridgeDataset(
        rows,
        schema=schema,
        max_nodes=int(config.max_nodes),
        strict_native_only=bool(checkpoint.get("strict_native_only", True)),
        continuous_feature_clip=float(checkpoint.get("continuous_feature_clip", 5.0)),
        feature_quant_bins=int(checkpoint.get("feature_quant_bins", 256)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=make_native_tensor_bridge_collate(
            tokenizer,
            max_source_length=int(args.max_source_length),
            schema=schema,
            continuous_feature_clip=float(checkpoint.get("continuous_feature_clip", 5.0)),
        ),
    )

    node_slots = schema.node_slots
    edge_slots = schema.edge_slots
    source_slot = _slot_index(node_slots, name="s", value_type="mask_one")
    adj_slot = _slot_index(edge_slots, name="adj", value_type="mask")
    edge_scalar_slot = _slot_index(edge_slots, name="A", value_type="scalar")
    pos_slot = _slot_index(node_slots, name="pos", value_type="scalar")

    totals = {
        "samples": 0,
        "count_ok": 0,
        "source_ok": 0,
        "source_count": 0,
        "adj_exact": 0,
        "adj_f1": 0.0,
        "adj_count": 0,
        "adj_pred_present": 0.0,
        "adj_true_present": 0.0,
        "adj_total": 0.0,
        "adj_logit_sum": 0.0,
        "adj_logit_count": 0,
        "adj_pos_logit_sum": 0.0,
        "adj_pos_logit_count": 0,
        "adj_neg_logit_sum": 0.0,
        "adj_neg_logit_count": 0,
        "adj_prior_present": 0.0,
        "edge_scalar_mae": 0.0,
        "edge_scalar_maxerr": 0.0,
        "pos_mae": 0.0,
        "pos_maxerr": 0.0,
        "scalar_exact": 0,
    }
    shown = 0
    sweep_totals = {threshold: _new_adj_totals() for threshold in sweep_thresholds}

    with torch.no_grad():
        for batch in loader:
            batch = _move_native_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                llm_outputs = llm(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_hidden_states=llm_layer_index != -1,
                    use_cache=False,
                )
            llm_hidden = llm_outputs.last_hidden_state if llm_layer_index == -1 else llm_outputs.hidden_states[llm_layer_index]
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

            masks = _schema_family_masks(schema, device)
            node_pred_bins = _mixed_discrete_predictions(outputs["node_discrete_logits"], masks["node_binary"])
            node_pred_bins = _apply_node_mask_one_predictions(
                node_pred_bins,
                outputs["node_discrete_logits"],
                batch.node_slot_mask,
                masks["node_mask_one"],
            )
            edge_pred_bins = _mixed_discrete_predictions(outputs["edge_discrete_logits"], masks["edge_binary"])
            edge_binary_logits = _binary_logits_from_discrete_logits(outputs["edge_discrete_logits"])
            if adj_slot is not None:
                edge_pred_bins[..., adj_slot] = (
                    edge_binary_logits[..., adj_slot] >= float(args.edge_logit_threshold)
                ).long()
            _, _, blended_count_norm = _normalized_count_predictions(
                outputs,
                max_nodes=int(batch.present_target.size(1)),
                count_blend_alpha=float(args.count_blend_alpha),
            )
            implicit_count_norm, explicit_count_norm, _ = _normalized_count_predictions(
                outputs,
                max_nodes=int(batch.present_target.size(1)),
                count_blend_alpha=float(args.count_blend_alpha),
            )
            count_prior_pred = _count_prior_predictions(outputs, max_nodes=int(batch.present_target.size(1)))
            decoded_count_norm = _select_decoded_count_norm(
                implicit_count_norm,
                explicit_count_norm,
                blended_count_norm,
                count_decode_mode=str(args.count_decode_mode),
                prior_norm=count_prior_pred[1] if count_prior_pred is not None else None,
            )
            pred_counts = torch.round(decoded_count_norm * float(batch.present_target.size(1))).long().clamp(1, int(batch.present_target.size(1)))
            class_count_pred = _count_class_predictions(outputs, max_nodes=int(batch.present_target.size(1)))
            if class_count_pred is not None and str(args.count_decode_mode).strip().lower() == "head_only":
                pred_counts = class_count_pred[0].long().clamp(1, int(batch.present_target.size(1)))
            if count_prior_pred is not None and str(args.count_decode_mode).strip().lower() == "prior":
                pred_counts = count_prior_pred[0].long().clamp(1, int(batch.present_target.size(1)))

            for sample_idx in range(int(batch.num_nodes.size(0))):
                n = int(batch.num_nodes[sample_idx].item())
                task_idx = int(batch.task_ids[sample_idx].item())
                totals["samples"] += 1
                count_ok = int(pred_counts[sample_idx].item()) == n
                totals["count_ok"] += int(count_ok)

                true_source = pred_source = None
                source_active = _slot_active(schema.task_node_slot_active, task_idx, source_slot)
                source_ok = True
                if source_active and source_slot is not None:
                    true_source = int(torch.argmax(batch.node_feature_bins[sample_idx, :n, source_slot]).item())
                    pred_source = int(torch.argmax(node_pred_bins[sample_idx, :n, source_slot]).item())
                    source_ok = true_source == pred_source
                    totals["source_ok"] += int(source_ok)
                    totals["source_count"] += 1

                adj_f1 = 0.0
                adj_exact = True
                adj_tp = adj_fp = adj_fn = 0
                adj_pred_density = 0.0
                adj_true_density = 0.0
                adj_logit_mean = 0.0
                adj_pos_logit_mean = 0.0
                adj_neg_logit_mean = 0.0
                adj_prior_density = 0.0
                adj_active = _slot_active(schema.task_edge_slot_active, task_idx, adj_slot)
                if adj_active and adj_slot is not None:
                    pred_adj = edge_pred_bins[sample_idx, :n, :n, adj_slot].clamp(0, 1)
                    true_adj = batch.edge_feature_bins[sample_idx, :n, :n, adj_slot].clamp(0, 1)
                    adj_logits = edge_binary_logits[sample_idx, :n, :n, adj_slot]
                    for threshold, threshold_totals in sweep_totals.items():
                        threshold_pred_adj = (adj_logits >= float(threshold)).long()
                        _update_adj_totals(threshold_totals, threshold_pred_adj, true_adj)
                    adj_f1 = _mask_f1(pred_adj.reshape(-1), true_adj.reshape(-1))
                    adj_exact = bool(torch.equal(pred_adj, true_adj))
                    pred_bool = pred_adj.bool()
                    true_bool = true_adj.bool()
                    adj_tp = int((pred_bool & true_bool).sum().item())
                    adj_fp = int((pred_bool & ~true_bool).sum().item())
                    adj_fn = int((~pred_bool & true_bool).sum().item())
                    adj_total = float(max(1, pred_adj.numel()))
                    adj_pred_density = float(pred_bool.float().mean().item())
                    adj_true_density = float(true_bool.float().mean().item())
                    adj_logit_mean = float(adj_logits.mean().item())
                    pos_logits = adj_logits[true_bool]
                    neg_logits = adj_logits[~true_bool]
                    if bool(pos_logits.numel() > 0):
                        adj_pos_logit_mean = float(pos_logits.mean().item())
                    if bool(neg_logits.numel() > 0):
                        adj_neg_logit_mean = float(neg_logits.mean().item())
                    if batch.edge_text_prior is not None:
                        adj_prior_density = float(batch.edge_text_prior[sample_idx, :n, :n].float().mean().item())
                    totals["adj_f1"] += float(adj_f1)
                    totals["adj_exact"] += int(adj_exact)
                    totals["adj_count"] += 1
                    totals["adj_pred_present"] += float(pred_bool.float().sum().item())
                    totals["adj_true_present"] += float(true_bool.float().sum().item())
                    totals["adj_total"] += adj_total
                    totals["adj_logit_sum"] += float(adj_logits.sum().item())
                    totals["adj_logit_count"] += int(adj_logits.numel())
                    totals["adj_pos_logit_sum"] += float(pos_logits.sum().item())
                    totals["adj_pos_logit_count"] += int(pos_logits.numel())
                    totals["adj_neg_logit_sum"] += float(neg_logits.sum().item())
                    totals["adj_neg_logit_count"] += int(neg_logits.numel())
                    totals["adj_prior_present"] += float(batch.edge_text_prior[sample_idx, :n, :n].float().sum().item()) if batch.edge_text_prior is not None else 0.0

                edge_scalar_mae = 0.0
                edge_scalar_maxerr = 0.0
                scalar_exact = True
                edge_scalar_active = _slot_active(schema.task_edge_slot_active, task_idx, edge_scalar_slot)
                pos_active = _slot_active(schema.task_node_slot_active, task_idx, pos_slot)
                if edge_scalar_active and edge_scalar_slot is not None:
                    pred_a = outputs["edge_scalar_values"][sample_idx, :n, :n, edge_scalar_slot]
                    true_a = batch.edge_feature_values[sample_idx, :n, :n, edge_scalar_slot]
                    edge_abs = torch.abs(pred_a - true_a)
                    edge_scalar_mae = float(edge_abs.mean().item())
                    edge_scalar_maxerr = float(edge_abs.max().item())
                    totals["edge_scalar_mae"] += edge_scalar_mae
                    totals["edge_scalar_maxerr"] += edge_scalar_maxerr
                    scalar_exact = scalar_exact and edge_scalar_maxerr <= float(args.scalar_exact_tolerance)

                pos_mae = 0.0
                pos_maxerr = 0.0
                if pos_active and pos_slot is not None:
                    pred_pos = outputs["node_scalar_values"][sample_idx, :n, pos_slot]
                    true_pos = batch.node_feature_values[sample_idx, :n, pos_slot]
                    pos_abs = torch.abs(pred_pos - true_pos)
                    pos_mae = float(pos_abs.mean().item())
                    pos_maxerr = float(pos_abs.max().item())
                    totals["pos_mae"] += pos_mae
                    totals["pos_maxerr"] += pos_maxerr
                    scalar_exact = scalar_exact and pos_maxerr <= float(args.scalar_exact_tolerance)
                totals["scalar_exact"] += int(scalar_exact)

                if shown < int(args.max_samples):
                    print(
                        f"sample={batch.sample_ids[sample_idx]} n={n} pred_n={int(pred_counts[sample_idx].item())} "
                        f"s={true_source}->{pred_source} source_ok={source_ok} "
                        f"adj_f1={adj_f1:.3f} adj_exact={adj_exact} tp={adj_tp} fp={adj_fp} fn={adj_fn} "
                        f"adj_density={adj_pred_density:.3f}/{adj_true_density:.3f} "
                        f"adj_logit={adj_logit_mean:.3f} pos={adj_pos_logit_mean:.3f} neg={adj_neg_logit_mean:.3f} "
                        f"edge_prior={adj_prior_density:.3f} "
                        f"A_mae={edge_scalar_mae:.4f} A_maxerr={edge_scalar_maxerr:.4f} "
                        f"pos_mae={pos_mae:.4f} pos_maxerr={pos_maxerr:.4f} scalar_exact={scalar_exact}"
                    )
                    shown += 1

    denom = max(1, int(totals["samples"]))
    source_denom = max(1, int(totals["source_count"]))
    adj_denom = max(1, int(totals["adj_count"]))
    adj_total = max(1.0, float(totals["adj_total"]))
    adj_logit_count = max(1, int(totals["adj_logit_count"]))
    adj_pos_logit_count = max(1, int(totals["adj_pos_logit_count"]))
    adj_neg_logit_count = max(1, int(totals["adj_neg_logit_count"]))
    print(
        "summary "
        f"samples={denom} "
        f"count_acc={totals['count_ok'] / denom:.3f} "
        f"source_acc={totals['source_ok'] / source_denom:.3f} "
        f"source_samples={int(totals['source_count'])} "
        f"adj_exact={totals['adj_exact'] / adj_denom:.3f} "
        f"adj_f1={totals['adj_f1'] / adj_denom:.3f} "
        f"adj_samples={int(totals['adj_count'])} "
        f"adj_density={totals['adj_pred_present'] / adj_total:.3f}/{totals['adj_true_present'] / adj_total:.3f} "
        f"adj_logit={totals['adj_logit_sum'] / adj_logit_count:.3f} "
        f"adj_pos_logit={totals['adj_pos_logit_sum'] / adj_pos_logit_count:.3f} "
        f"adj_neg_logit={totals['adj_neg_logit_sum'] / adj_neg_logit_count:.3f} "
        f"edge_prior_density={totals['adj_prior_present'] / adj_total:.3f} "
        f"A_mae={totals['edge_scalar_mae'] / denom:.4f} "
        f"A_maxerr={totals['edge_scalar_maxerr'] / denom:.4f} "
        f"pos_mae={totals['pos_mae'] / denom:.4f} "
        f"pos_maxerr={totals['pos_maxerr'] / denom:.4f} "
        f"scalar_exact@{float(args.scalar_exact_tolerance):g}={totals['scalar_exact'] / denom:.3f}"
    )
    for threshold in sweep_thresholds:
        threshold_totals = sweep_totals[threshold]
        threshold_adj_total = max(1.0, float(threshold_totals["adj_total"]))
        print(
            f"threshold_summary edge_logit_threshold={threshold:g} "
            f"samples={denom} "
            f"adj_exact={threshold_totals['adj_exact'] / denom:.3f} "
            f"adj_f1={threshold_totals['adj_f1'] / denom:.3f} "
            f"adj_density={threshold_totals['adj_pred_present'] / threshold_adj_total:.3f}/"
            f"{threshold_totals['adj_true_present'] / threshold_adj_total:.3f}"
        )


if __name__ == "__main__":
    main()
