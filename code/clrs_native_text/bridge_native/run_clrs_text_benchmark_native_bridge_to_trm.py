#!/usr/bin/env python3
"""Run future benchmark-native CLRS-Text bridge -> benchmark-native TRM inference."""

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

from clrs_native_text.bridge_native.benchmark_native_slot_bridge import (
    NativeTensorBridgeSchema,
    build_benchmark_native_bridge_model,
    build_benchmark_native_trm_tensors_from_bridge_prediction,
)
from clrs_native_text.bridge_native.initial_style_bridge import DirectBridgeConfig
from clrs_native_text.bridge_native.run_clrs_text_bridge_to_trm_benchmark_native import (
    _load_benchmark_native_trm_components,
)
from clrs_native_text.bridge_native.run_clrs_text_bridge_to_trm_v61 import (
    _decode_template_prediction,
    _read_jsonl,
    _resolve_task,
    _resolve_torch_dtype,
    _shape_for_template,
    _task_output_templates,
    _write_jsonl_row,
)


def _load_benchmark_native_bridge_components(
    *,
    checkpoint_path: Path,
    device: torch.device,
    llm_dtype: torch.dtype,
) -> tuple[dict[str, Any], NativeTensorBridgeSchema, DirectBridgeConfig, str, str, int, Any, Any, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    bridge_family = str(checkpoint.get("bridge_family", ""))
    if bridge_family != "benchmark_native_slot_bridge_v1":
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a benchmark-native slot bridge "
            f"(bridge_family={bridge_family!r})"
        )
    schema = NativeTensorBridgeSchema.from_dict(checkpoint["schema"])
    bridge_config = DirectBridgeConfig(**checkpoint["bridge_config"])
    arch = str(checkpoint["arch"])
    llm_model_name = str(checkpoint["llm_model_name"])
    llm_layer_index = int(checkpoint.get("llm_layer_index", -1))

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=llm_dtype).to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    bridge = build_benchmark_native_bridge_model(arch, bridge_config, schema=schema).to(device)
    bridge.load_state_dict(checkpoint["model_state_dict"])
    bridge.eval()
    return checkpoint, schema, bridge_config, arch, llm_model_name, llm_layer_index, tokenizer, llm, bridge


def _serialize_native_tensor_summary(native_tensors: dict[str, torch.Tensor | int]) -> dict[str, Any]:
    num_nodes = int(native_tensors["num_nodes"])
    node_slot_mask = native_tensors["node_slot_mask"]
    edge_slot_mask = native_tensors["edge_slot_mask"]
    graph_slot_mask = native_tensors["graph_slot_mask"]
    return {
        "num_nodes": num_nodes,
        "node_slot_dim": int(node_slot_mask.shape[1]) if isinstance(node_slot_mask, torch.Tensor) and node_slot_mask.ndim == 2 else 0,
        "edge_slot_dim": int(edge_slot_mask.shape[1]) if isinstance(edge_slot_mask, torch.Tensor) and edge_slot_mask.ndim == 2 else 0,
        "graph_slot_dim": int(graph_slot_mask.shape[1]) if isinstance(graph_slot_mask, torch.Tensor) and graph_slot_mask.ndim == 2 else 0,
        "active_node_slot_cells": int(float(node_slot_mask.sum().item())) if isinstance(node_slot_mask, torch.Tensor) else 0,
        "active_edge_slot_cells": int(float(edge_slot_mask.sum().item())) if isinstance(edge_slot_mask, torch.Tensor) else 0,
        "active_graph_slot_cells": int(float(graph_slot_mask.sum().item())) if isinstance(graph_slot_mask, torch.Tensor) else 0,
    }


def _serialize_native_trm_input(native_tensors: dict[str, torch.Tensor | int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "num_nodes": int(native_tensors["num_nodes"]),
    }
    for key in (
        "node_feature_bins",
        "node_feature_values",
        "node_slot_mask",
        "edge_index",
        "edge_feature_bins",
        "edge_feature_values",
        "edge_slot_mask",
        "graph_feature_bins",
        "graph_feature_values",
        "graph_slot_mask",
        "graph_index",
        "graph_ptr",
        "task_ids",
    ):
        value = native_tensors.get(key)
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().tolist()
        else:
            payload[key] = value
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark-native CLRS-Text bridge + benchmark-native TRM inference")
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--trm-checkpoint", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-num-recurrences", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--task-filter", default="all")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-write-native-trm-input", action="store_true")
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
    ) = _load_benchmark_native_bridge_components(
        checkpoint_path=bridge_checkpoint_path,
        device=device,
        llm_dtype=llm_dtype,
    )
    bridge_task_to_idx = schema.task_index()

    (
        trm_checkpoint,
        trm_model,
        task_to_idx,
        _node_slot_specs,
        _edge_slot_specs,
        _graph_slot_specs,
        _node_slot_mean,
        _node_slot_std,
        _edge_slot_mean,
        _edge_slot_std,
        _graph_slot_mean,
        _graph_slot_std,
    ) = _load_benchmark_native_trm_components(
        checkpoint_path=trm_checkpoint_path,
        device=device,
    )
    eval_num_recurrences = (
        int(args.eval_num_recurrences)
        if int(args.eval_num_recurrences) > 0
        else int(trm_checkpoint["model_config"]["num_recurrences"])
    )

    rows = _read_jsonl(input_path)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    task_filter = None if str(args.task_filter).strip().lower() == "all" else {
        part.strip() for part in str(args.task_filter).split(",") if part.strip()
    }
    if task_filter is not None:
        rows = [row for row in rows if _resolve_task(row) in task_filter]

    total_rows = len(rows)
    progress_every = max(1, int(args.progress_every))
    processed = 0
    success_count = 0
    error_count = 0
    started_at = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting benchmark-native bridge->benchmark-native-TRM inference on {total_rows} rows; "
        f"writing incrementally to {output_path}",
        flush=True,
    )

    with output_path.open("w", encoding="utf-8") as output_handle:
        for row in rows:
            task = _resolve_task(row)
            if task not in bridge_task_to_idx:
                _write_jsonl_row(
                    output_handle,
                    {
                        "sample_id": row.get("sample_id", ""),
                        "algo_name": task,
                        "question": row.get("question", ""),
                        "error": f"Task '{task}' not present in bridge checkpoint vocab",
                    },
                )
                processed += 1
                error_count += 1
                continue
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

            enc = tokenizer(
                [str(row.get("question", ""))],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_source_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            bridge_task_id = int(bridge_task_to_idx[task])
            bridge_task_ids = torch.tensor([bridge_task_id], dtype=torch.long, device=device)

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
                native_tensors = build_benchmark_native_trm_tensors_from_bridge_prediction(
                    task=task,
                    schema=schema,
                    outputs=bridge_outputs,
                    task_to_idx=task_to_idx,
                    task_id=bridge_task_id,
                    present_threshold=args.present_threshold,
                    device=device,
                    count_blend_alpha=float(getattr(bridge_config, "count_blend_alpha", 0.5)),
                    count_decode_mode=str(getattr(bridge_config, "count_decode_mode", "blended")),
                )
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
                num_nodes = int(native_tensors["num_nodes"])

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
                    "bridge_family": bridge_checkpoint.get("bridge_family", "benchmark_native_slot_bridge_v1"),
                    "bridge_arch": bridge_arch,
                    "llm_model_name": llm_model_name,
                    "bridge_checkpoint": str(bridge_checkpoint_path),
                    "trm_checkpoint": str(trm_checkpoint_path),
                    "eval_num_recurrences": eval_num_recurrences,
                    "bridge_config": asdict(bridge_config),
                    "trm_model_config": trm_checkpoint["model_config"],
                    "pred_native_tensor_summary": _serialize_native_tensor_summary(native_tensors),
                    "pred_native_trm_input": None if args.no_write_native_trm_input else _serialize_native_trm_input(native_tensors),
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
        f"Completed benchmark-native bridge->benchmark-native-TRM inference: rows={processed} "
        f"success={success_count} errors={error_count} elapsed={elapsed:.1f}s output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
