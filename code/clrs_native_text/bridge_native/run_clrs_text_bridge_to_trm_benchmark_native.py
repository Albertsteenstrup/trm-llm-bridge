#!/usr/bin/env python3
"""Run CLRS-Text -> bridge -> benchmark-native TRM inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parents[1]
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch

from clrs_native_text.bridge_native.clrs_text_bridge_schema import graph_target_to_native_input_row
from clrs_native_text.bridge_native.initial_style_bridge import build_graph_target_from_prediction, make_direct_bridge_collate
from clrs_native_text.bridge_native.run_clrs_text_bridge_to_trm_v61 import (
    _decode_template_prediction,
    _load_bridge_components,
    _read_jsonl,
    _resolve_task,
    _resolve_torch_dtype,
    _shape_for_template,
    _task_output_templates,
    _write_jsonl_row,
)
from trm_llm.stage3_specialists.clrs_trm_benchmark_native import GraphTRMBenchmarkNative
from trm_llm.stage3_specialists.train_clrs_trm_specialist_benchmark_native import (
    InputSlotSpec,
    _infer_num_nodes_from_inputs,
    _normalize_edge_input,
    _normalize_graph_input,
    _normalize_node_input,
    _slot_key,
)


def _slot_specs_from_payload(payload: list[dict[str, Any]]) -> list[InputSlotSpec]:
    out: list[InputSlotSpec] = []
    for item in payload:
        out.append(
            InputSlotSpec(
                key=str(item["key"]),
                name=str(item["name"]),
                location=str(item["location"]),
                value_type=str(item["value_type"]),
                channel_idx=int(item["channel_idx"]),
                is_discrete=bool(item["is_discrete"]),
            )
        )
    return out


def _load_benchmark_native_trm_components(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    GraphTRMBenchmarkNative,
    dict[str, int],
    list[InputSlotSpec],
    list[InputSlotSpec],
    list[InputSlotSpec],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    node_slot_specs = _slot_specs_from_payload(list(checkpoint.get("node_slot_specs", [])))
    edge_slot_specs = _slot_specs_from_payload(list(checkpoint.get("edge_slot_specs", [])))
    graph_slot_specs = _slot_specs_from_payload(list(checkpoint.get("graph_slot_specs", [])))
    model = GraphTRMBenchmarkNative(
        num_node_slots=int(model_config["num_node_slots"]),
        num_edge_slots=int(model_config["num_edge_slots"]),
        num_graph_slots=int(model_config["num_graph_slots"]),
        max_input_discrete_value=int(model_config["max_input_discrete_value"]),
        node_slot_is_discrete=[slot.is_discrete for slot in node_slot_specs],
        edge_slot_is_discrete=[slot.is_discrete for slot in edge_slot_specs],
        graph_slot_is_discrete=[slot.is_discrete for slot in graph_slot_specs],
        hidden_dim=int(model_config["hidden_dim"]),
        num_tasks=int(model_config["num_tasks"]),
        num_discrete_classes=int(model_config["num_discrete_classes"]),
        num_mask_hints=int(model_config.get("num_mask_hints", 0)),
        num_scalar_hints=int(model_config.get("num_scalar_hints", 0)),
        num_mask_one_hints=int(model_config.get("num_mask_one_hints", 0)),
        num_categorical_hints=int(model_config.get("num_categorical_hints", 0)),
        num_graph_hints=int(model_config.get("num_graph_hints", 0)),
        num_recurrences=int(model_config["num_recurrences"]),
        inner_recurrences=int(model_config.get("inner_recurrences", 2)),
        recurrent_depth=int(model_config.get("recurrent_depth", 2)),
        expansion=float(model_config.get("expansion", 4.0)),
        forward_dtype=str(model_config.get("forward_dtype", "float32")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    task_vocab = [str(name) for name in checkpoint.get("task_vocab", [])]
    task_to_idx = {name: idx for idx, name in enumerate(task_vocab)}

    node_slot_mean = [float(v) for v in checkpoint.get("node_slot_mean", [0.0] * len(node_slot_specs))]
    node_slot_std = [max(1e-6, float(v)) for v in checkpoint.get("node_slot_std", [1.0] * len(node_slot_specs))]
    edge_slot_mean = [float(v) for v in checkpoint.get("edge_slot_mean", [0.0] * len(edge_slot_specs))]
    edge_slot_std = [max(1e-6, float(v)) for v in checkpoint.get("edge_slot_std", [1.0] * len(edge_slot_specs))]
    graph_slot_mean = [float(v) for v in checkpoint.get("graph_slot_mean", [0.0] * len(graph_slot_specs))]
    graph_slot_std = [max(1e-6, float(v)) for v in checkpoint.get("graph_slot_std", [1.0] * len(graph_slot_specs))]
    return (
        checkpoint,
        model,
        task_to_idx,
        node_slot_specs,
        edge_slot_specs,
        graph_slot_specs,
        node_slot_mean,
        node_slot_std,
        edge_slot_mean,
        edge_slot_std,
        graph_slot_mean,
        graph_slot_std,
    )


def _build_native_inference_tensors(
    raw_input_row: dict[str, Any],
    *,
    task: str,
    task_to_idx: dict[str, int],
    node_slot_specs: list[InputSlotSpec],
    edge_slot_specs: list[InputSlotSpec],
    graph_slot_specs: list[InputSlotSpec],
    node_slot_mean: list[float],
    node_slot_std: list[float],
    edge_slot_mean: list[float],
    edge_slot_std: list[float],
    graph_slot_mean: list[float],
    graph_slot_std: list[float],
    continuous_feature_clip: float,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    inputs = list(raw_input_row.get("inputs", []))
    num_nodes = _infer_num_nodes_from_inputs(inputs)
    if num_nodes <= 0:
        num_nodes = max(0, int(raw_input_row.get("num_nodes", 0)))
    if num_nodes <= 0:
        return {"num_nodes": 0}

    node_slot_to_idx = {slot.key: idx for idx, slot in enumerate(node_slot_specs)}
    edge_slot_to_idx = {slot.key: idx for idx, slot in enumerate(edge_slot_specs)}
    graph_slot_to_idx = {slot.key: idx for idx, slot in enumerate(graph_slot_specs)}

    node_values = [[0.0 for _ in range(len(node_slot_specs))] for _ in range(num_nodes)]
    node_bins = [[0 for _ in range(len(node_slot_specs))] for _ in range(num_nodes)]
    node_mask = [[0.0 for _ in range(len(node_slot_specs))] for _ in range(num_nodes)]

    edge_index: list[list[int]] = []
    edge_values = [[0.0 for _ in range(len(edge_slot_specs))] for _ in range(num_nodes * num_nodes)] if edge_slot_specs else []
    edge_bins = [[0 for _ in range(len(edge_slot_specs))] for _ in range(num_nodes * num_nodes)] if edge_slot_specs else []
    edge_mask = [[0.0 for _ in range(len(edge_slot_specs))] for _ in range(num_nodes * num_nodes)] if edge_slot_specs else []
    if edge_slot_specs:
        edge_index = [[src, dst] for src in range(num_nodes) for dst in range(num_nodes)]

    graph_values = [0.0 for _ in range(len(graph_slot_specs))]
    graph_bins = [0 for _ in range(len(graph_slot_specs))]
    graph_mask = [0.0 for _ in range(len(graph_slot_specs))]

    for dp in inputs:
        location = str(dp.get("location", "")).strip()
        value_type = str(dp.get("type", "")).strip()
        name = str(dp.get("name", "")).strip() or f"{location}_input"
        if location == "node":
            values = _normalize_node_input(dp, num_nodes=num_nodes)
            for channel_idx in range(values.shape[1]):
                slot_idx = node_slot_to_idx.get(_slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx))
                if slot_idx is None:
                    continue
                mean = float(node_slot_mean[slot_idx]) if slot_idx < len(node_slot_mean) else 0.0
                std = max(1e-6, float(node_slot_std[slot_idx]) if slot_idx < len(node_slot_std) else 1.0)
                for node_idx in range(num_nodes):
                    raw_value = float(values[node_idx, channel_idx])
                    node_mask[node_idx][slot_idx] = 1.0
                    if node_slot_specs[slot_idx].is_discrete:
                        node_values[node_idx][slot_idx] = raw_value
                        node_bins[node_idx][slot_idx] = max(0, int(round(raw_value)))
                    else:
                        normalized = (raw_value - mean) / std
                        if float(continuous_feature_clip) > 0.0:
                            normalized = max(-float(continuous_feature_clip), min(float(continuous_feature_clip), normalized))
                        node_values[node_idx][slot_idx] = float(normalized)
        elif location == "edge" and edge_slot_specs:
            values = _normalize_edge_input(dp, num_nodes=num_nodes).reshape(num_nodes * num_nodes, -1)
            for channel_idx in range(values.shape[1]):
                slot_idx = edge_slot_to_idx.get(_slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx))
                if slot_idx is None:
                    continue
                mean = float(edge_slot_mean[slot_idx]) if slot_idx < len(edge_slot_mean) else 0.0
                std = max(1e-6, float(edge_slot_std[slot_idx]) if slot_idx < len(edge_slot_std) else 1.0)
                for edge_row in range(values.shape[0]):
                    raw_value = float(values[edge_row, channel_idx])
                    edge_mask[edge_row][slot_idx] = 1.0
                    if edge_slot_specs[slot_idx].is_discrete:
                        edge_values[edge_row][slot_idx] = raw_value
                        edge_bins[edge_row][slot_idx] = max(0, int(round(raw_value)))
                    else:
                        normalized = (raw_value - mean) / std
                        if float(continuous_feature_clip) > 0.0:
                            normalized = max(-float(continuous_feature_clip), min(float(continuous_feature_clip), normalized))
                        edge_values[edge_row][slot_idx] = float(normalized)
        elif location == "graph":
            values = _normalize_graph_input(dp)
            for channel_idx in range(values.shape[0]):
                slot_idx = graph_slot_to_idx.get(_slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx))
                if slot_idx is None:
                    continue
                raw_value = float(values[channel_idx])
                graph_mask[slot_idx] = 1.0
                if graph_slot_specs[slot_idx].is_discrete:
                    graph_values[slot_idx] = raw_value
                    graph_bins[slot_idx] = max(0, int(round(raw_value)))
                else:
                    mean = float(graph_slot_mean[slot_idx]) if slot_idx < len(graph_slot_mean) else 0.0
                    std = max(1e-6, float(graph_slot_std[slot_idx]) if slot_idx < len(graph_slot_std) else 1.0)
                    normalized = (raw_value - mean) / std
                    if float(continuous_feature_clip) > 0.0:
                        normalized = max(-float(continuous_feature_clip), min(float(continuous_feature_clip), normalized))
                    graph_values[slot_idx] = float(normalized)

    edge_slot_dim = len(edge_slot_specs)
    graph_slot_dim = len(graph_slot_specs)
    return {
        "num_nodes": num_nodes,
        "node_feature_bins": torch.tensor(node_bins, dtype=torch.long, device=device),
        "node_feature_values": torch.tensor(node_values, dtype=torch.float32, device=device),
        "node_slot_mask": torch.tensor(node_mask, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(edge_index, dtype=torch.long, device=device) if edge_index else torch.zeros((0, 2), dtype=torch.long, device=device),
        "edge_feature_bins": torch.tensor(edge_bins, dtype=torch.long, device=device) if edge_slot_dim > 0 else torch.zeros((0, 0), dtype=torch.long, device=device),
        "edge_feature_values": torch.tensor(edge_values, dtype=torch.float32, device=device) if edge_slot_dim > 0 else torch.zeros((0, 0), dtype=torch.float32, device=device),
        "edge_slot_mask": torch.tensor(edge_mask, dtype=torch.float32, device=device) if edge_slot_dim > 0 else torch.zeros((0, 0), dtype=torch.float32, device=device),
        "graph_feature_bins": torch.tensor([graph_bins], dtype=torch.long, device=device) if graph_slot_dim > 0 else torch.zeros((1, 0), dtype=torch.long, device=device),
        "graph_feature_values": torch.tensor([graph_values], dtype=torch.float32, device=device) if graph_slot_dim > 0 else torch.zeros((1, 0), dtype=torch.float32, device=device),
        "graph_slot_mask": torch.tensor([graph_mask], dtype=torch.float32, device=device) if graph_slot_dim > 0 else torch.zeros((1, 0), dtype=torch.float32, device=device),
        "graph_index": torch.zeros((num_nodes,), dtype=torch.long, device=device),
        "graph_ptr": torch.tensor([0, num_nodes], dtype=torch.long, device=device),
        "task_ids": torch.tensor([task_to_idx[task]], dtype=torch.long, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLRS-Text bridge + benchmark-native TRM inference")
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--trm-checkpoint", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--continuous-feature-clip", type=float, default=5.0)
    parser.add_argument("--eval-num-recurrences", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--task-filter", default="all")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    llm_dtype = _resolve_torch_dtype(args.llm_dtype, device)

    bridge_checkpoint_path = Path(args.bridge_checkpoint).resolve()
    trm_checkpoint_path = Path(args.trm_checkpoint).resolve()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()

    (
        bridge_checkpoint,
        schema,
        bridge_config,
        bridge_arch,
        llm_model_name,
        llm_layer_index,
        tokenizer,
        llm,
        bridge,
    ) = _load_bridge_components(checkpoint_path=bridge_checkpoint_path, device=device, llm_dtype=llm_dtype)
    bridge_task_to_idx = schema.task_index()

    (
        trm_checkpoint,
        trm_model,
        task_to_idx,
        node_slot_specs,
        edge_slot_specs,
        graph_slot_specs,
        node_slot_mean,
        node_slot_std,
        edge_slot_mean,
        edge_slot_std,
        graph_slot_mean,
        graph_slot_std,
    ) = _load_benchmark_native_trm_components(
        checkpoint_path=trm_checkpoint_path,
        device=device,
    )
    eval_num_recurrences = int(args.eval_num_recurrences) if int(args.eval_num_recurrences) > 0 else int(trm_checkpoint["model_config"]["num_recurrences"])

    rows = _read_jsonl(input_path)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    task_filter = None if str(args.task_filter).strip().lower() == "all" else {part.strip() for part in str(args.task_filter).split(",") if part.strip()}
    if task_filter is not None:
        rows = [row for row in rows if _resolve_task(row) in task_filter]

    collate_fn = make_direct_bridge_collate(tokenizer, max_source_length=args.max_source_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = len(rows)
    progress_every = max(1, int(args.progress_every))
    processed = 0
    success_count = 0
    error_count = 0
    started_at = time.time()

    print(
        f"Starting bridge->benchmark-native-TRM inference on {total_rows} rows; writing incrementally to {output_path}",
        flush=True,
    )

    with output_path.open("w", encoding="utf-8") as output_handle:
        for row in rows:
            task = _resolve_task(row)
            if task not in task_to_idx:
                _write_jsonl_row(
                    output_handle,
                    {
                        "sample_id": row.get("sample_id", ""),
                        "algo_name": task,
                        "question": row.get("question", ""),
                        "error": f"Task '{task}' not present in benchmark-native TRM checkpoint vocab",
                    },
                )
                processed += 1
                error_count += 1
                continue

            templates = _task_output_templates(task)
            if not templates:
                _write_jsonl_row(
                    output_handle,
                    {
                        "sample_id": row.get("sample_id", ""),
                        "algo_name": task,
                        "question": row.get("question", ""),
                        "error": f"No TRM output templates registered for task '{task}'",
                    },
                )
                processed += 1
                error_count += 1
                continue

            batch = collate_fn(
                [
                    {
                        "question": str(row.get("question", "")),
                        "task_id": torch.tensor(int(bridge_task_to_idx.get(task, 0)), dtype=torch.long),
                        "present_target": torch.zeros(bridge_config.max_nodes, dtype=torch.float32),
                        "node_feature_values": torch.zeros((bridge_config.max_nodes, schema.feature_dim), dtype=torch.float32),
                        "node_feature_mask": torch.zeros((bridge_config.max_nodes, schema.feature_dim), dtype=torch.float32),
                        "edge_adj": torch.zeros((bridge_config.max_nodes, bridge_config.max_nodes), dtype=torch.float32),
                        "edge_values": torch.zeros((bridge_config.max_nodes, bridge_config.max_nodes), dtype=torch.float32),
                        "edge_value_mask": torch.zeros((bridge_config.max_nodes, bridge_config.max_nodes), dtype=torch.float32),
                        "graph_scalar_values": torch.zeros(schema.graph_scalar_dim, dtype=torch.float32),
                        "graph_scalar_mask": torch.zeros(schema.graph_scalar_dim, dtype=torch.float32),
                        "num_nodes": torch.tensor(0, dtype=torch.long),
                        "sample_id": str(row.get("sample_id", "")),
                        "task": task,
                    }
                ]
            )
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            bridge_task_ids = torch.tensor([int(bridge_task_to_idx.get(task, 0))], dtype=torch.long, device=device)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                    llm_outputs = llm(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                llm_hidden = llm_outputs.hidden_states[llm_layer_index]
                bridge_outputs = bridge(llm_hidden.float(), attention_mask.bool(), bridge_task_ids)
                pred_graph_target = build_graph_target_from_prediction(
                    row=row,
                    outputs=bridge_outputs,
                    schema=schema,
                    present_threshold=args.present_threshold,
                    edge_threshold=args.edge_threshold,
                )
                pred_native_input_row = graph_target_to_native_input_row(
                    pred_graph_target,
                    canonical_inputs=row.get("canonical_inputs"),
                    sample_id=str(row.get("sample_id", "")),
                    split="bridge",
                    sample_index=processed,
                )
                native_tensors = _build_native_inference_tensors(
                    pred_native_input_row,
                    task=task,
                    task_to_idx=task_to_idx,
                    node_slot_specs=node_slot_specs,
                    edge_slot_specs=edge_slot_specs,
                    graph_slot_specs=graph_slot_specs,
                    node_slot_mean=node_slot_mean,
                    node_slot_std=node_slot_std,
                    edge_slot_mean=edge_slot_mean,
                    edge_slot_std=edge_slot_std,
                    graph_slot_mean=graph_slot_mean,
                    graph_slot_std=graph_slot_std,
                    continuous_feature_clip=float(args.continuous_feature_clip),
                    device=device,
                )
                num_nodes = int(native_tensors["num_nodes"])
                if num_nodes <= 0:
                    _write_jsonl_row(
                        output_handle,
                        {
                            "sample_id": row.get("sample_id", ""),
                            "algo_name": task,
                            "question": row.get("question", ""),
                            "bridge_arch": bridge_arch,
                            "bridge_checkpoint": str(bridge_checkpoint_path),
                            "trm_checkpoint": str(trm_checkpoint_path),
                            "pred_graph_target": pred_graph_target,
                            "pred_native_input_row": pred_native_input_row,
                            "error": "Bridge adapter produced zero usable nodes for benchmark-native TRM",
                        },
                    )
                    processed += 1
                    error_count += 1
                    continue

                trm_outputs = trm_model(
                    node_feature_bins=native_tensors["node_feature_bins"],
                    node_feature_values=native_tensors["node_feature_values"],
                    node_slot_mask=native_tensors["node_slot_mask"],
                    edge_index=native_tensors["edge_index"],
                    edge_feature_bins=native_tensors["edge_feature_bins"],
                    edge_feature_values=native_tensors["edge_feature_values"],
                    edge_slot_mask=native_tensors["edge_slot_mask"],
                    graph_feature_bins=native_tensors["graph_feature_bins"],
                    graph_feature_values=native_tensors["graph_feature_values"],
                    graph_slot_mask=native_tensors["graph_slot_mask"],
                    graph_index=native_tensors["graph_index"],
                    graph_ptr=native_tensors["graph_ptr"],
                    task_ids=native_tensors["task_ids"],
                    num_recurrences=eval_num_recurrences,
                )
                node_hidden = trm_outputs["hidden_by_step"][-1]
                graph_hidden = trm_outputs["graph_hidden_by_step"][-1][0]

                decoded_outputs: list[dict[str, Any]] = []
                for template in templates:
                    preds, summary = _decode_template_prediction(
                        trm_model,
                        node_hidden=node_hidden,
                        graph_hidden=graph_hidden,
                        template=template,
                    )
                    decoded_outputs.append(
                        {
                            "name": template.name,
                            "location": template.location,
                            "type": template.value_type,
                            "shape": _shape_for_template(template, num_nodes=num_nodes),
                            "predictions": preds,
                            "summary": summary,
                        }
                    )

            _write_jsonl_row(
                output_handle,
                {
                    "sample_id": row.get("sample_id", ""),
                    "algo_name": task,
                    "question": row.get("question", ""),
                    "bridge_arch": bridge_arch,
                    "llm_model_name": llm_model_name,
                    "bridge_checkpoint": str(bridge_checkpoint_path),
                    "trm_checkpoint": str(trm_checkpoint_path),
                    "eval_num_recurrences": eval_num_recurrences,
                    "bridge_config": asdict(bridge_config),
                    "trm_model_config": trm_checkpoint["model_config"],
                    "pred_graph_target": pred_graph_target,
                    "pred_native_input_row": pred_native_input_row,
                    "trm_pred_final_outputs": decoded_outputs,
                },
            )
            processed += 1
            success_count += 1
            if processed % progress_every == 0 or processed == total_rows:
                elapsed = max(1e-6, time.time() - started_at)
                print(
                    f"[{processed}/{total_rows}] success={success_count} errors={error_count} "
                    f"elapsed={elapsed:.1f}s rows_per_sec={processed / elapsed:.2f}",
                    flush=True,
                )

    elapsed = max(1e-6, time.time() - started_at)
    print(
        f"Completed bridge->benchmark-native-TRM inference: rows={processed} success={success_count} errors={error_count} "
        f"elapsed={elapsed:.1f}s output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
