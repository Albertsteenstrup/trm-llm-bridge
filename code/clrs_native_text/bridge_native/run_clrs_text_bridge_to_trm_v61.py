#!/usr/bin/env python3
"""Run CLRS-Text -> bridge -> TRM V6.1 inference and emit decoded outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parents[1]
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch

from clrs_native_text.bridge_native.initial_style_bridge import (
    BridgeSchema,
    DirectBridgeConfig,
    build_bridge_model,
    build_graph_target_from_prediction,
    make_direct_bridge_collate,
)
from trm_llm.stage3_specialists.clrs_trm_v6 import GraphTRMSpecialistV6


POINTER_TYPES = {"pointer", "should_be_permutation"}
TASK_OUTPUT_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "articulation_points": [{"name": "is_cut", "location": "node", "type": "mask"}],
    "activity_selector": [{"name": "selected", "location": "node", "type": "mask"}],
    "bellman_ford": [{"name": "pi", "location": "node", "type": "pointer"}],
    "bfs": [{"name": "pi", "location": "node", "type": "pointer"}],
    "binary_search": [{"name": "return", "location": "node", "type": "mask_one"}],
    "bridges": [{"name": "is_bridge", "location": "edge", "type": "mask"}],
    "bubble_sort": [{"name": "pred", "location": "node", "type": "should_be_permutation"}],
    "dag_shortest_paths": [{"name": "pi", "location": "node", "type": "pointer"}],
    "dfs": [{"name": "pi", "location": "node", "type": "pointer"}],
    "dijkstra": [{"name": "pi", "location": "node", "type": "pointer"}],
    "find_maximum_subarray_kadane": [
        {"name": "start", "location": "node", "type": "mask_one"},
        {"name": "end", "location": "node", "type": "mask_one"},
    ],
    "floyd_warshall": [{"name": "Pi", "location": "edge", "type": "pointer"}],
    "graham_scan": [{"name": "in_hull", "location": "node", "type": "mask"}],
    "heapsort": [{"name": "pred", "location": "node", "type": "should_be_permutation"}],
    "insertion_sort": [{"name": "pred", "location": "node", "type": "should_be_permutation"}],
    "jarvis_march": [{"name": "in_hull", "location": "node", "type": "mask"}],
    "kmp_matcher": [{"name": "match", "location": "node", "type": "mask_one"}],
    "lcs_length": [{"name": "b", "location": "edge", "type": "categorical"}],
    "matrix_chain_order": [{"name": "s", "location": "edge", "type": "pointer"}],
    "minimum": [{"name": "min", "location": "node", "type": "mask_one"}],
    "mst_kruskal": [{"name": "in_mst", "location": "edge", "type": "mask"}],
    "mst_prim": [{"name": "pi", "location": "node", "type": "pointer"}],
    "naive_string_matcher": [{"name": "match", "location": "node", "type": "mask_one"}],
    "optimal_bst": [{"name": "root", "location": "edge", "type": "pointer"}],
    "quickselect": [{"name": "median", "location": "node", "type": "mask_one"}],
    "quicksort": [{"name": "pred", "location": "node", "type": "should_be_permutation"}],
    "segments_intersect": [{"name": "intersect", "location": "graph", "type": "mask"}],
    "strongly_connected_components": [{"name": "scc_id", "location": "node", "type": "pointer"}],
    "task_scheduling": [{"name": "selected", "location": "node", "type": "mask"}],
    "topological_sort": [
        {"name": "topo", "location": "node", "type": "pointer"},
        {"name": "topo_head", "location": "node", "type": "mask_one"},
    ],
}


@dataclass
class OutputTemplate:
    name: str
    location: str
    value_type: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    handle.flush()


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


def _resolve_task(row: dict[str, Any], pred_graph_target: dict[str, Any] | None = None) -> str:
    for key in ("algo_name", "algorithm", "task", "algo"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if pred_graph_target:
        value = pred_graph_target.get("algorithm")
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _load_bridge_components(
    *,
    checkpoint_path: Path,
    device: torch.device,
    llm_dtype: torch.dtype,
) -> tuple[dict[str, Any], BridgeSchema, DirectBridgeConfig, str, str, int, Any, Any, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    schema = BridgeSchema.from_dict(checkpoint["schema"])
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

    bridge = build_bridge_model(
        arch,
        bridge_config,
        feature_dim=schema.feature_dim,
        graph_scalar_dim=schema.graph_scalar_dim,
        num_tasks=max(1, len(schema.task_names)),
    ).to(device)
    bridge.load_state_dict(checkpoint["model_state_dict"])
    bridge.eval()

    return checkpoint, schema, bridge_config, arch, llm_model_name, llm_layer_index, tokenizer, llm, bridge


def _load_trm_components(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], GraphTRMSpecialistV6, dict[str, int], list[float], list[float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    model = GraphTRMSpecialistV6(
        feature_bins=int(model_config["feature_bins"]),
        feature_dim=int(model_config["feature_dim"]),
        continuous_feature_dim=int(model_config["continuous_feature_dim"]),
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
    feature_mean = [float(v) for v in checkpoint.get("continuous_feature_mean", [0.0] * int(model_config["feature_dim"]))]
    feature_std = [max(1e-6, float(v)) for v in checkpoint.get("continuous_feature_std", [1.0] * int(model_config["feature_dim"]))]
    return checkpoint, model, task_to_idx, feature_mean, feature_std


def _task_output_templates(task: str) -> list[OutputTemplate]:
    templates = TASK_OUTPUT_TEMPLATES.get(task, [])
    return [OutputTemplate(name=item["name"], location=item["location"], value_type=item["type"]) for item in templates]


def _shape_for_template(template: OutputTemplate, *, num_nodes: int) -> list[int]:
    if template.location == "node":
        return [int(num_nodes)]
    if template.location == "edge":
        return [int(num_nodes), int(num_nodes)]
    return [1]


def _normalize_graph_target(
    pred_graph_target: dict[str, Any],
    *,
    feature_dim: int,
    feature_mean: list[float],
    feature_std: list[float],
    continuous_feature_clip: float,
) -> tuple[list[list[int]], list[list[float]], list[list[int]], int]:
    raw_bins = pred_graph_target.get("node_feature_bins", [])
    raw_values = pred_graph_target.get("node_feature_values", [])

    node_feature_bins: list[list[int]] = []
    if isinstance(raw_bins, list):
        for feat in raw_bins:
            if isinstance(feat, list):
                values = [int(v) for v in feat[:feature_dim]]
            else:
                values = [int(feat)]
            if len(values) < feature_dim:
                values = values + [0] * (feature_dim - len(values))
            node_feature_bins.append(values)

    node_feature_values: list[list[float]] = []
    if isinstance(raw_values, list):
        for feat in raw_values[: len(node_feature_bins)]:
            if isinstance(feat, list):
                values = [float(v) for v in feat[:feature_dim]]
            else:
                values = [float(feat)]
            if len(values) < feature_dim:
                values = values + [0.0] * (feature_dim - len(values))
            normalized = [
                (float(value) - feature_mean[idx]) / feature_std[idx]
                for idx, value in enumerate(values)
            ]
            if continuous_feature_clip > 0.0:
                normalized = [
                    max(-continuous_feature_clip, min(continuous_feature_clip, float(value)))
                    for value in normalized
                ]
            node_feature_values.append(normalized)

    num_nodes = min(
        max(0, int(pred_graph_target.get("num_nodes", len(node_feature_bins)))),
        len(node_feature_bins),
        len(node_feature_values) if node_feature_values else len(node_feature_bins),
    )
    if num_nodes <= 0:
        return [], [], [], 0

    edge_index: list[list[int]] = []
    raw_edges = pred_graph_target.get("edge_index", [])
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, list) or len(edge) != 2:
                continue
            src = int(edge[0])
            dst = int(edge[1])
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                edge_index.append([src, dst])

    return node_feature_bins[:num_nodes], node_feature_values[:num_nodes], edge_index, num_nodes


def _decode_template_prediction(
    model: GraphTRMSpecialistV6,
    *,
    node_hidden: torch.Tensor,
    graph_hidden: torch.Tensor,
    template: OutputTemplate,
) -> tuple[list[Any], dict[str, Any]]:
    if template.location == "node" and template.value_type in POINTER_TYPES:
        logits = (
            model.decode_node_should_be_permutation_logits(graph_hidden_states=node_hidden)
            if template.value_type == "should_be_permutation"
            else model.decode_node_pointer_logits(graph_hidden_states=node_hidden)
        )
        preds = torch.argmax(logits, dim=-1).tolist()
        return [int(v) for v in preds], {"selected_indices": [int(v) for v in preds]}

    if template.location == "node" and template.value_type == "mask":
        logits = model.decode_node_mask_logits(graph_hidden_states=node_hidden)
        mask = (torch.sigmoid(logits) >= 0.5).to(torch.int64).tolist()
        selected = [idx for idx, value in enumerate(mask) if int(value) != 0]
        return [int(v) for v in mask], {"selected_indices": selected}

    if template.location == "node" and template.value_type == "mask_one":
        logits = model.decode_node_mask_one_logits(graph_hidden_states=node_hidden)
        pred_index = int(torch.argmax(logits).item())
        mask = [1.0 if idx == pred_index else 0.0 for idx in range(int(logits.numel()))]
        return mask, {"selected_index": pred_index}

    if template.location == "node" and template.value_type == "categorical":
        logits = model.decode_node_categorical_logits(graph_hidden_states=node_hidden)
        preds = torch.argmax(logits, dim=-1).tolist()
        return [int(v) for v in preds], {}

    if template.location == "node" and template.value_type == "scalar":
        preds = model.decode_node_scalar(graph_hidden_states=node_hidden).tolist()
        return [float(v) for v in preds], {}

    if template.location == "edge" and template.value_type == "mask":
        logits = model.decode_edge_mask_logits(graph_hidden_states=node_hidden)
        preds = (torch.sigmoid(logits) >= 0.5).to(torch.int64).tolist()
        num_nodes = int(node_hidden.shape[0])
        selected_edges = [[idx // num_nodes, idx % num_nodes] for idx, value in enumerate(preds) if int(value) != 0]
        return [int(v) for v in preds], {"selected_edges": selected_edges}

    if template.location == "edge" and template.value_type == "categorical":
        logits = model.decode_edge_categorical_logits(graph_hidden_states=node_hidden)
        preds = torch.argmax(logits, dim=-1).tolist()
        return [int(v) for v in preds], {}

    if template.location == "edge" and template.value_type == "pointer":
        logits = model.decode_edge_pointer_logits(graph_hidden_states=node_hidden)
        preds = torch.argmax(logits, dim=-1).tolist()
        return [int(v) for v in preds], {}

    if template.location == "graph" and template.value_type == "mask":
        logits = model.decode_graph_mask_logits(graph_hidden_states=graph_hidden.unsqueeze(0))
        pred = 1.0 if float(torch.sigmoid(logits)[0].item()) >= 0.5 else 0.0
        return [pred], {"value": bool(pred)}

    if template.location == "graph" and template.value_type == "categorical":
        logits = model.node_categorical_head(graph_hidden.unsqueeze(0)).to(torch.float32)
        preds = torch.argmax(logits, dim=-1).tolist()
        return [int(v) for v in preds], {}

    if template.location == "graph" and template.value_type == "scalar":
        preds = model.node_scalar_head(graph_hidden.unsqueeze(0)).squeeze(-1).to(torch.float32).tolist()
        return [float(v) for v in preds], {}

    raise ValueError(f"Unsupported template family: {template.location}/{template.value_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLRS-Text bridge + TRM V6.1 inference")
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

    trm_checkpoint, trm_model, task_to_idx, feature_mean, feature_std = _load_trm_components(
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
        f"Starting bridge->TRM inference on {total_rows} rows; writing incrementally to {output_path}",
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
                        "error": f"Task '{task}' not present in TRM checkpoint vocab",
                    },
                )
                processed += 1
                error_count += 1
                if processed % progress_every == 0 or processed == total_rows:
                    elapsed = max(1e-6, time.time() - started_at)
                    print(
                        f"[{processed}/{total_rows}] success={success_count} errors={error_count} "
                        f"elapsed={elapsed:.1f}s rows_per_sec={processed / elapsed:.2f}",
                        flush=True,
                    )
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
                if processed % progress_every == 0 or processed == total_rows:
                    elapsed = max(1e-6, time.time() - started_at)
                    print(
                        f"[{processed}/{total_rows}] success={success_count} errors={error_count} "
                        f"elapsed={elapsed:.1f}s rows_per_sec={processed / elapsed:.2f}",
                        flush=True,
                    )
                continue

            batch = collate_fn(
                [
                    {
                        "question": str(row.get("question", "")),
                        "task_id": torch.tensor(task_to_idx[task], dtype=torch.long),
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

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=llm_dtype, enabled=device.type == "cuda"):
                    llm_outputs = llm(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                llm_hidden = llm_outputs.hidden_states[llm_layer_index]
                bridge_outputs = bridge(llm_hidden.float(), attention_mask.bool(), batch.task_ids.to(device))

                pred_graph_target = build_graph_target_from_prediction(
                    row=row,
                    outputs=bridge_outputs,
                    schema=schema,
                    present_threshold=args.present_threshold,
                    edge_threshold=args.edge_threshold,
                )

                node_feature_bins, node_feature_values, edge_index, num_nodes = _normalize_graph_target(
                    pred_graph_target,
                    feature_dim=int(trm_checkpoint["model_config"]["feature_dim"]),
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    continuous_feature_clip=float(args.continuous_feature_clip),
                )
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
                            "error": "Bridge produced zero usable nodes",
                        },
                    )
                    processed += 1
                    error_count += 1
                    if processed % progress_every == 0 or processed == total_rows:
                        elapsed = max(1e-6, time.time() - started_at)
                        print(
                            f"[{processed}/{total_rows}] success={success_count} errors={error_count} "
                            f"elapsed={elapsed:.1f}s rows_per_sec={processed / elapsed:.2f}",
                            flush=True,
                        )
                    continue

                node_feature_bins_tensor = torch.tensor(node_feature_bins, dtype=torch.long, device=device)
                node_feature_values_tensor = torch.tensor(node_feature_values, dtype=torch.float32, device=device)
                edge_index_tensor = (
                    torch.tensor(edge_index, dtype=torch.long, device=device)
                    if edge_index
                    else torch.zeros((0, 2), dtype=torch.long, device=device)
                )
                graph_index_tensor = torch.zeros((num_nodes,), dtype=torch.long, device=device)
                graph_ptr_tensor = torch.tensor([0, num_nodes], dtype=torch.long, device=device)
                task_ids_tensor = torch.tensor([task_to_idx[task]], dtype=torch.long, device=device)

                trm_outputs = trm_model(
                    node_feature_bins=node_feature_bins_tensor,
                    node_feature_values=node_feature_values_tensor,
                    edge_index=edge_index_tensor,
                    graph_index=graph_index_tensor,
                    graph_ptr=graph_ptr_tensor,
                    task_ids=task_ids_tensor,
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
                    "pred_node_feature_bins": pred_graph_target.get("node_feature_bins", []),
                    "pred_node_features": pred_graph_target.get("node_feature_values", []),
                    "pred_edge_index": pred_graph_target.get("edge_index", []),
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
        f"Completed bridge->TRM inference: rows={processed} success={success_count} errors={error_count} "
        f"elapsed={elapsed:.1f}s output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
