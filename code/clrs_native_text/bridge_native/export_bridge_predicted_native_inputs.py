#!/usr/bin/env python3
"""Export CLRS-Text bridge predictions as CLRS native-input JSON rows.

This is the first half of text -> bridge -> ReNAR evaluation.  It runs the
PyTorch text bridge and writes predicted CLRS input probes in the same simple
JSON shape used by the native CLRS text data.  A separate ReNAR-side script can
then load these rows, splice the predicted inputs into gold CLRS Feedback
objects, and score a frozen specialist checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from clrs_native_text.bridge_native.benchmark_native_slot_bridge import (
    CLRSTextBenchmarkNativeBridgeDataset,
    NativeTensorBridgeSchema,
    build_benchmark_native_trm_tensors_from_bridge_prediction,
    make_native_tensor_bridge_collate,
    _move_native_batch,
)
from clrs_native_text.bridge_native.run_clrs_text_benchmark_native_bridge_to_trm import (
    _load_benchmark_native_bridge_components,
)
from clrs_native_text.bridge_native.train_clrs_text_bridge_benchmark_native import (
    _resolve_torch_dtype,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def _denorm(value: float, *, mean: float, std: float) -> float:
    return float(value) * max(1e-6, float(std)) + float(mean)


def _discrete_or_binary(value: int, value_type: str) -> int | float:
    if value_type in {"mask", "mask_one"}:
        return float(1 if int(value) > 0 else 0)
    return int(value)


def _native_inputs_from_tensors(
    *,
    schema: NativeTensorBridgeSchema,
    task_id: int,
    native_tensors: dict[str, torch.Tensor | int],
) -> list[dict[str, Any]]:
    num_nodes = int(native_tensors["num_nodes"])
    rows: list[dict[str, Any]] = []

    node_bins = native_tensors["node_feature_bins"].detach().cpu()
    node_values = native_tensors["node_feature_values"].detach().cpu()
    for slot_idx, slot in enumerate(schema.node_slots):
        if float(schema.task_node_slot_active[task_id][slot_idx]) <= 0.5:
            continue
        if slot.is_discrete:
            data = [
                _discrete_or_binary(int(node_bins[i, slot_idx].item()), slot.value_type)
                for i in range(num_nodes)
            ]
        else:
            mean = float(schema.node_slot_mean[slot_idx])
            std = float(schema.node_slot_std[slot_idx])
            data = [
                _denorm(float(node_values[i, slot_idx].item()), mean=mean, std=std)
                for i in range(num_nodes)
            ]
        rows.append(
            {
                "name": slot.name,
                "location": slot.location,
                "type": slot.value_type,
                "shape": [num_nodes],
                "data": data,
            }
        )

    edge_bins = native_tensors["edge_feature_bins"].detach().cpu()
    edge_values = native_tensors["edge_feature_values"].detach().cpu()
    edge_index = native_tensors["edge_index"].detach().cpu()
    for slot_idx, slot in enumerate(schema.edge_slots):
        if float(schema.task_edge_slot_active[task_id][slot_idx]) <= 0.5:
            continue
        matrix: list[list[int | float]] = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for edge_i in range(edge_index.shape[0]):
            src = int(edge_index[edge_i, 0].item())
            dst = int(edge_index[edge_i, 1].item())
            if slot.is_discrete:
                matrix[src][dst] = _discrete_or_binary(int(edge_bins[edge_i, slot_idx].item()), slot.value_type)
            else:
                mean = float(schema.edge_slot_mean[slot_idx])
                std = float(schema.edge_slot_std[slot_idx])
                matrix[src][dst] = _denorm(float(edge_values[edge_i, slot_idx].item()), mean=mean, std=std)
        rows.append(
            {
                "name": slot.name,
                "location": slot.location,
                "type": slot.value_type,
                "shape": [num_nodes, num_nodes],
                "data": matrix,
            }
        )

    graph_bins = native_tensors["graph_feature_bins"].detach().cpu()
    graph_values = native_tensors["graph_feature_values"].detach().cpu()
    for slot_idx, slot in enumerate(schema.graph_slots):
        if float(schema.task_graph_slot_active[task_id][slot_idx]) <= 0.5:
            continue
        if slot.is_discrete:
            data: int | float = _discrete_or_binary(int(graph_bins[0, slot_idx].item()), slot.value_type)
        else:
            mean = float(schema.graph_slot_mean[slot_idx])
            std = float(schema.graph_slot_std[slot_idx])
            data = _denorm(float(graph_values[0, slot_idx].item()), mean=mean, std=std)
        rows.append(
            {
                "name": slot.name,
                "location": slot.location,
                "type": slot.value_type,
                "shape": [],
                "data": data,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export bridge-predicted CLRS native inputs")
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--algorithms", default="", help="Optional comma-separated allow-list")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for compatibility; export runs row-wise")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--edge-text-prior-logit-scale", type=float, default=None)
    parser.add_argument("--count-decode-mode", default="prior")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_dtype = _resolve_torch_dtype(str(args.llm_dtype), device)
    checkpoint_path = Path(args.bridge_checkpoint).resolve()
    checkpoint, schema, config, _arch, _llm_model_name, llm_layer_index, tokenizer, llm, bridge = (
        _load_benchmark_native_bridge_components(
            checkpoint_path=checkpoint_path,
            device=device,
            llm_dtype=llm_dtype,
        )
    )
    if args.edge_text_prior_logit_scale is not None:
        config.edge_text_prior_logit_scale = float(args.edge_text_prior_logit_scale)

    task_to_idx = schema.task_index()
    allowed = {v.strip() for v in str(args.algorithms).split(",") if v.strip()}
    rows = _read_jsonl(Path(args.input_jsonl).resolve())
    if int(args.limit_samples) > 0:
        rows = rows[: int(args.limit_samples)]
    filtered_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        task = str(row.get("algorithm") or row.get("algo_name") or row.get("native_input_target", {}).get("algorithm") or "")
        if allowed and task not in allowed:
            skipped += 1
            continue
        if task not in task_to_idx:
            skipped += 1
            continue
        filtered_rows.append(row)

    dataset = CLRSTextBenchmarkNativeBridgeDataset(
        filtered_rows,
        schema=schema,
        max_nodes=int(config.max_nodes),
        strict_native_only=bool(checkpoint.get("strict_native_only", True)),
        continuous_feature_clip=float(checkpoint.get("continuous_feature_clip", 5.0)),
        feature_quant_bins=int(checkpoint.get("feature_quant_bins", 256)),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_native_tensor_bridge_collate(
            tokenizer,
            max_source_length=int(args.max_source_length),
            schema=schema,
            continuous_feature_clip=float(checkpoint.get("continuous_feature_clip", 5.0)),
        ),
    )

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row_idx, batch in enumerate(loader):
            row = filtered_rows[row_idx]
            batch = _move_native_batch(batch, device)
            task = str(batch.tasks[0])
            task_id = int(batch.task_ids[0].item())
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                    llm_outputs = llm(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
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
                native_tensors = build_benchmark_native_trm_tensors_from_bridge_prediction(
                    task=task,
                    schema=schema,
                    outputs=outputs,
                    task_to_idx=task_to_idx,
                    task_id=task_id,
                    present_threshold=0.5,
                    device=device,
                    count_blend_alpha=float(getattr(config, "count_blend_alpha", 0.0)),
                    count_decode_mode=str(args.count_decode_mode),
                )
            _write_jsonl_row(
                handle,
                {
                    "sample_id": row.get("sample_id", row_idx),
                    "algorithm": task,
                    "question": row.get("question", ""),
                    "bridge_checkpoint": str(checkpoint_path),
                    "bridge_family": checkpoint.get("bridge_family", "benchmark_native_slot_bridge_v1"),
                    "num_nodes": int(native_tensors["num_nodes"]),
                    "predicted_inputs": _native_inputs_from_tensors(
                        schema=schema,
                        task_id=task_id,
                        native_tensors=native_tensors,
                    ),
                },
            )
            written += 1
            if written % max(1, int(args.progress_every)) == 0:
                print(f"exported={written} skipped={skipped}", flush=True)

    print(f"done exported={written} skipped={skipped} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
