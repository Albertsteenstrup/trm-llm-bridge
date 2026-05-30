#!/usr/bin/env python3
"""Schema helpers for official CLRS-Text bridge training.

The bridge target is a canonical JSON representation of the algorithm inputs
described in official CLRS-Text questions. A deterministic adapter then maps
that canonical representation into a graph-style IR that can be consumed by the
current TRM family.

This keeps the textual benchmark faithful to CLRS-Text while avoiding another
custom natural-language synthesis layer.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _split_top_level(text: str, separators: set[str]) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_square = 0
    depth_round = 0
    for ch in text:
        if ch == "[":
            depth_square += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth_square = max(0, depth_square - 1)
            buf.append(ch)
            continue
        if ch == "(":
            depth_round += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth_round = max(0, depth_round - 1)
            buf.append(ch)
            continue
        if ch in separators and depth_square == 0 and depth_round == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
            continue
        buf.append(ch)
    piece = "".join(buf).strip()
    if piece:
        parts.append(piece)
    return parts


def _parse_scalar(text: str) -> int | float | str:
    value = text.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return int(lowered == "true")
    try:
        if any(ch in value for ch in ".eE"):
            out = float(value)
            if math.isfinite(out):
                return out
            return 0.0
        return int(value)
    except Exception:
        return value


def _parse_sequence_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    depth_square = 0
    depth_round = 0
    for ch in text:
        if ch == "[":
            depth_square += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth_square = max(0, depth_square - 1)
            buf.append(ch)
            continue
        if ch == "(":
            depth_round += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth_round = max(0, depth_round - 1)
            buf.append(ch)
            continue
        if ch in {",", " ", "\t", "\n"} and depth_square == 0 and depth_round == 0:
            piece = "".join(buf).strip()
            if piece:
                tokens.append(piece)
            buf = []
            continue
        buf.append(ch)
    piece = "".join(buf).strip()
    if piece:
        tokens.append(piece)
    return tokens


def parse_text_value(text: str) -> Any:
    value = text.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        tokens = _parse_sequence_tokens(inner)
        return [parse_text_value(token) for token in tokens]
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_text_value(token) for token in _split_top_level(inner, {","})]
    return _parse_scalar(value)


def _is_scalar_sequence(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(not isinstance(item, list) for item in value)
        and all(isinstance(item, (int, float)) for item in value)
    )


def _is_matrix(value: Any) -> bool:
    return isinstance(value, list) and value and all(_is_scalar_sequence(row) for row in value)


def infer_num_nodes_from_inputs(inputs: dict[str, Any]) -> int:
    if "A" in inputs and _is_matrix(inputs["A"]):
        return len(inputs["A"])
    lengths = [len(v) for v in inputs.values() if _is_scalar_sequence(v)]
    if lengths:
        return int(max(lengths))
    return 1


def parse_question_to_canonical(question: str, *, answer: str = "", algo_name: str = "") -> dict[str, Any]:
    text = question.strip()
    if ":\n" in text:
        algo, remainder = text.split(":\n", 1)
    else:
        algo, _, remainder = text.partition(":")
    algorithm = algo_name.strip() or algo.strip()
    body = remainder.split("\ntrace |", 1)[0].strip()
    if ", initial_trace:" in body:
        body = body.split(", initial_trace:", 1)[0].strip()
    elif " initial_trace:" in body:
        body = body.split(" initial_trace:", 1)[0].strip()

    inputs: dict[str, Any] = {}
    for field in _split_top_level(body, {","}):
        if ":" not in field:
            continue
        key, raw_value = field.split(":", 1)
        key = key.strip()
        if not key:
            continue
        raw_value = raw_value.strip()
        marker_idx = raw_value.find("\n")
        if marker_idx >= 0:
            trailing = raw_value[marker_idx + 1 :].strip()
            if trailing.endswith(":") and trailing.replace("_", "").replace(":", "").isalnum():
                raw_value = raw_value[:marker_idx].strip()
        inputs[key] = parse_text_value(raw_value)

    num_nodes = infer_num_nodes_from_inputs(inputs)
    return {
        "algorithm": algorithm,
        "inputs": inputs,
        "num_nodes": num_nodes,
        "answer_preview": answer[:200] if answer else "",
    }


def _line_graph_edges(length: int, *, offset: int = 0) -> list[list[int]]:
    edges: list[list[int]] = []
    for idx in range(max(0, length - 1)):
        a = offset + idx
        b = offset + idx + 1
        edges.append([a, b])
        edges.append([b, a])
    if not edges and length == 1:
        edges.append([offset, offset])
    return edges


def _quantize_features(node_features: list[list[float]], bins: int) -> list[list[int]]:
    arr = np.asarray(node_features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("node_features must be rank-2")
    if bins <= 1:
        return np.zeros_like(arr, dtype=np.int64).tolist()
    mins = np.min(arr, axis=0, keepdims=True)
    maxs = np.max(arr, axis=0, keepdims=True)
    denom = np.where((maxs - mins) < 1e-8, 1.0, (maxs - mins))
    scaled = (arr - mins) / denom
    q = np.rint(scaled * float(bins - 1)).astype(np.int64)
    q = np.clip(q, 0, bins - 1)
    return q.tolist()


def canonical_to_graph_target(canonical: dict[str, Any], *, feature_quant_bins: int = 256) -> dict[str, Any]:
    algorithm = str(canonical.get("algorithm", "unknown")).strip() or "unknown"
    inputs = canonical.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("canonical inputs must be a dict")

    num_nodes = infer_num_nodes_from_inputs(inputs)
    node_feature_values: list[list[float]] = []
    edge_index: list[list[int]] = []
    edge_values: list[float] = []
    feature_names: list[str] = []
    graph_inputs: dict[str, Any] = {}

    matrix_A = inputs.get("A")
    scalar_vectors = {name: value for name, value in inputs.items() if _is_scalar_sequence(value)}
    scalar_inputs = {name: value for name, value in inputs.items() if isinstance(value, (int, float))}

    if _is_matrix(matrix_A):
        A = np.asarray(matrix_A, dtype=np.float32)
        num_nodes = int(A.shape[0])
        for src in range(num_nodes):
            for dst in range(int(A.shape[1])):
                weight = float(A[src, dst])
                if abs(weight) > 1e-9:
                    edge_index.append([src, dst])
                    edge_values.append(weight)

        columns: list[list[float]] = []
        for name in sorted(scalar_vectors):
            values = scalar_vectors[name]
            if len(values) != num_nodes:
                continue
            feature_names.append(name)
            columns.append([float(v) for v in values])

        for name in sorted(scalar_inputs):
            value = scalar_inputs[name]
            if isinstance(value, int) and 0 <= value < num_nodes:
                feature_names.append(f"{name}_one_hot")
                columns.append([1.0 if idx == int(value) else 0.0 for idx in range(num_nodes)])
            else:
                feature_names.append(f"{name}_repeat")
                columns.append([float(value)] * num_nodes)

        if columns:
            node_feature_values = [[float(column[node]) for column in columns] for node in range(num_nodes)]
        else:
            node_feature_values = [[0.0] for _ in range(num_nodes)]

        graph_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in scalar_vectors and key != "A" and key not in scalar_inputs
        }
    else:
        seq_fields = [(name, values) for name, values in sorted(scalar_vectors.items()) if len(values) > 0]
        if seq_fields and len({len(values) for _, values in seq_fields}) == 1:
            num_nodes = len(seq_fields[0][1])
            edge_index = _line_graph_edges(num_nodes)
            columns: list[list[float]] = []
            for name, values in seq_fields:
                feature_names.append(name)
                columns.append([float(v) for v in values])
            for name in sorted(scalar_inputs):
                value = scalar_inputs[name]
                if isinstance(value, int) and 0 <= value < num_nodes:
                    feature_names.append(f"{name}_one_hot")
                    columns.append([1.0 if idx == int(value) else 0.0 for idx in range(num_nodes)])
                else:
                    feature_names.append(f"{name}_repeat")
                    columns.append([float(value)] * num_nodes)
            node_feature_values = [[float(column[node]) for column in columns] for node in range(num_nodes)]
        elif seq_fields:
            rows: list[list[float]] = []
            edges: list[list[int]] = []
            segment_names = [name for name, _ in seq_fields]
            offset = 0
            for seg_idx, (name, values) in enumerate(seq_fields):
                seg_len = len(values)
                edges.extend(_line_graph_edges(seg_len, offset=offset))
                for pos, value in enumerate(values):
                    row = [float(value)]
                    row.extend(1.0 if seg_idx == candidate else 0.0 for candidate in range(len(segment_names)))
                    row.append(float(pos) / float(max(1, seg_len - 1)))
                    rows.append(row)
                offset += seg_len
            node_feature_values = rows or [[0.0]]
            edge_index = edges
            feature_names = ["value"] + [f"is_{name}" for name in segment_names] + ["segment_pos"]
            num_nodes = len(node_feature_values)
            graph_inputs = {key: value for key, value in scalar_inputs.items()}
        else:
            num_nodes = max(1, num_nodes)
            node_feature_values = [[float(scalar_inputs[name]) for name in sorted(scalar_inputs)]]
            feature_names = [f"{name}_repeat" for name in sorted(scalar_inputs)] or ["bias"]
            if not node_feature_values[0]:
                node_feature_values = [[0.0]]
                feature_names = ["bias"]
            edge_index = _line_graph_edges(num_nodes)

    if not edge_index and num_nodes > 0:
        edge_index = _line_graph_edges(num_nodes)
    node_feature_bins = _quantize_features(node_feature_values, bins=feature_quant_bins)
    return {
        "algorithm": algorithm,
        "num_nodes": int(num_nodes),
        "feature_names": feature_names,
        "node_feature_values": node_feature_values,
        "node_feature_bins": node_feature_bins,
        "edge_index": edge_index,
        "edge_values": edge_values,
        "graph_inputs": graph_inputs,
        "adapter_version": "bridge_v2_generic",
    }


def _dense_edge_mask(num_nodes: int, edge_index: Any) -> list[list[float]]:
    mask = [[0.0 for _ in range(max(0, num_nodes))] for _ in range(max(0, num_nodes))]
    if not isinstance(edge_index, list):
        return mask
    for raw_edge in edge_index:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            continue
        src = int(raw_edge[0])
        dst = int(raw_edge[1])
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            mask[src][dst] = 1.0
    return mask


def _is_binary_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) in {0.0, 1.0}


def _sequence_native_type(values: list[Any], *, num_nodes: int) -> str:
    if values and all(_is_binary_scalar(v) for v in values):
        return "mask"
    if values and all(isinstance(v, int) and 0 <= int(v) < max(1, num_nodes) for v in values):
        return "pointer"
    return "scalar"


def _matrix_native_type(matrix: list[list[Any]]) -> str:
    flat = [item for row in matrix for item in row]
    if flat and all(_is_binary_scalar(v) for v in flat):
        return "mask"
    return "scalar"


def _dense_edge_values(
    num_nodes: int,
    edge_index: Any,
    edge_values: Any,
) -> list[list[float]]:
    dense = [[0.0 for _ in range(max(0, num_nodes))] for _ in range(max(0, num_nodes))]
    if not isinstance(edge_index, list):
        return dense
    values_list = edge_values if isinstance(edge_values, list) else []
    for edge_idx, raw_edge in enumerate(edge_index):
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            continue
        src = int(raw_edge[0])
        dst = int(raw_edge[1])
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            if edge_idx < len(values_list):
                dense[src][dst] = float(values_list[edge_idx])
            else:
                dense[src][dst] = 1.0
    return dense


def _round_int_sequence(values: list[float], *, lower: int = 0, upper: int | None = None) -> list[int]:
    out: list[int] = []
    for value in values:
        rounded = int(round(float(value)))
        rounded = max(lower, rounded)
        if upper is not None:
            rounded = min(upper, rounded)
        out.append(rounded)
    return out


def _cast_sequence_for_native_type(values: list[float], *, native_type: str, num_nodes: int) -> list[Any]:
    if native_type == "mask":
        return [1.0 if float(v) >= 0.5 else 0.0 for v in values]
    if native_type == "pointer":
        return _round_int_sequence(values, lower=0, upper=max(0, num_nodes - 1))
    if native_type in {"categorical", "should_be_permutation"}:
        return _round_int_sequence(values, lower=0)
    return [float(v) for v in values]


def _normalize_compat_feature_name(name: str) -> str:
    if name.endswith("_one_hot") and not name.endswith("__one_hot"):
        return f"{name[:-8]}__one_hot"
    if name.endswith("_repeat") and not name.endswith("__repeat"):
        return f"{name[:-7]}__repeat"
    return name


def _generic_graph_target_to_native_input_row(
    graph_target: dict[str, Any],
    *,
    sample_id: str = "",
    split: str = "bridge",
    sample_index: int = -1,
) -> dict[str, Any]:
    algorithm = str(graph_target.get("algorithm", "unknown")).strip() or "unknown"
    num_nodes = max(0, int(graph_target.get("num_nodes", 0)))

    feature_names = [str(name) for name in graph_target.get("feature_names", []) if str(name).strip()]
    raw_values = graph_target.get("node_feature_values", [])
    node_feature_values: list[list[float]] = []
    if isinstance(raw_values, list):
        for raw_row in raw_values[:num_nodes]:
            if isinstance(raw_row, list):
                node_feature_values.append([float(v) for v in raw_row])
            else:
                node_feature_values.append([float(raw_row)])

    if not node_feature_values:
        raw_bins = graph_target.get("node_feature_bins", [])
        if isinstance(raw_bins, list):
            for raw_row in raw_bins[:num_nodes]:
                if isinstance(raw_row, list):
                    node_feature_values.append([float(v) for v in raw_row])
                else:
                    node_feature_values.append([float(raw_row)])

    feature_dim = max(
        1,
        len(feature_names),
        max((len(row) for row in node_feature_values), default=0),
    )
    if len(feature_names) < feature_dim:
        feature_names = feature_names + [f"bridge_feat_{idx}" for idx in range(len(feature_names), feature_dim)]

    if len(node_feature_values) < num_nodes:
        node_feature_values.extend([[0.0] * feature_dim for _ in range(num_nodes - len(node_feature_values))])

    normalized_node_feature_values = []
    for row in node_feature_values[:num_nodes]:
        values = row[:feature_dim]
        if len(values) < feature_dim:
            values = values + [0.0] * (feature_dim - len(values))
        normalized_node_feature_values.append([float(v) for v in values])

    edge_mask = _dense_edge_mask(num_nodes, graph_target.get("edge_index", []))
    graph_inputs = graph_target.get("graph_inputs", {})
    graph_input_items = []
    if isinstance(graph_inputs, dict):
        for key in sorted(graph_inputs):
            value = graph_inputs[key]
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                graph_input_items.append((str(key), float(value)))

    inputs: list[dict[str, Any]] = [
        {
            "name": "bridge_node_features",
            "location": "node",
            "type": "scalar",
            "shape": [int(num_nodes), int(feature_dim)],
            "data": normalized_node_feature_values,
        },
        {
            "name": "bridge_edge_mask",
            "location": "edge",
            "type": "mask",
            "shape": [int(num_nodes), int(num_nodes)],
            "data": edge_mask,
        },
    ]
    for key, value in graph_input_items:
        inputs.append(
            {
                "name": key,
                "location": "graph",
                "type": "scalar",
                "shape": [1],
                "data": [float(value)],
            }
        )

    resolved_sample_id = sample_id.strip() or f"{algorithm}:{split}:{sample_index}"
    return {
        "sample_id": resolved_sample_id,
        "algorithm": algorithm,
        "split": split,
        "sample_index": int(sample_index),
        "num_steps": 0,
        "source": "clrs-text-native-input-adapter",
        "spec": {
            "bridge_node_features": {"stage": "input", "location": "node", "type": "scalar"},
            "bridge_edge_mask": {"stage": "input", "location": "edge", "type": "mask"},
            **{
                key: {"stage": "input", "location": "graph", "type": "scalar"}
                for key, _ in graph_input_items
            },
        },
        "inputs": inputs,
        "hints": [],
        "outputs": [],
        "feature_names": feature_names,
        "graph_inputs": {key: value for key, value in graph_input_items},
        "adapter_version": "bridge_v2_native_input_v1",
    }


def graph_target_to_native_input_row(
    graph_target: dict[str, Any],
    *,
    canonical_inputs: dict[str, Any] | None = None,
    sample_id: str = "",
    split: str = "bridge",
    sample_index: int = -1,
) -> dict[str, Any]:
    algorithm = str(graph_target.get("algorithm", "unknown")).strip() or "unknown"
    num_nodes = max(0, int(graph_target.get("num_nodes", 0)))
    raw_canonical = canonical_inputs or {}
    if "inputs" in raw_canonical and isinstance(raw_canonical.get("inputs"), dict):
        raw_canonical = raw_canonical["inputs"]

    feature_names = [str(name) for name in graph_target.get("feature_names", []) if str(name).strip()]
    raw_values = graph_target.get("node_feature_values", [])
    node_feature_values: list[list[float]] = []
    if isinstance(raw_values, list):
        for raw_row in raw_values[:num_nodes]:
            if isinstance(raw_row, list):
                node_feature_values.append([float(v) for v in raw_row])
            else:
                node_feature_values.append([float(raw_row)])
    if len(node_feature_values) < num_nodes:
        raw_bins = graph_target.get("node_feature_bins", [])
        if isinstance(raw_bins, list):
            for raw_row in raw_bins[len(node_feature_values) : num_nodes]:
                if isinstance(raw_row, list):
                    node_feature_values.append([float(v) for v in raw_row])
                else:
                    node_feature_values.append([float(raw_row)])

    feature_dim = max(
        len(feature_names),
        max((len(row) for row in node_feature_values), default=0),
    )
    if feature_dim > len(feature_names):
        feature_names = feature_names + [f"bridge_feat_{idx}" for idx in range(len(feature_names), feature_dim)]
    if not feature_names or not isinstance(raw_canonical, dict) or not raw_canonical:
        return _generic_graph_target_to_native_input_row(
            graph_target,
            sample_id=sample_id,
            split=split,
            sample_index=sample_index,
        )

    if len(node_feature_values) < num_nodes:
        node_feature_values.extend([[0.0] * feature_dim for _ in range(num_nodes - len(node_feature_values))])
    node_feature_values = [
        (row[:feature_dim] + [0.0] * max(0, feature_dim - len(row)))[:feature_dim]
        for row in node_feature_values[:num_nodes]
    ]
    feature_columns = {
        feature_names[col_idx]: [float(node_feature_values[row_idx][col_idx]) for row_idx in range(num_nodes)]
        for col_idx in range(min(feature_dim, len(feature_names)))
    }
    graph_inputs = {
        str(key): float(value)
        for key, value in graph_target.get("graph_inputs", {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    edge_mask = _dense_edge_mask(num_nodes, graph_target.get("edge_index", []))
    edge_dense_values = _dense_edge_values(num_nodes, graph_target.get("edge_index", []), graph_target.get("edge_values", []))

    inputs: list[dict[str, Any]] = []
    spec: dict[str, dict[str, str]] = {}

    matrix_inputs = {
        str(name): value
        for name, value in raw_canonical.items()
        if _is_matrix(value)
    }
    vector_inputs = {
        str(name): value
        for name, value in raw_canonical.items()
        if _is_scalar_sequence(value)
    }
    scalar_inputs = {
        str(name): value
        for name, value in raw_canonical.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }

    for name in sorted(matrix_inputs):
        matrix = matrix_inputs[name]
        if len(matrix) != num_nodes:
            continue
        native_type = _matrix_native_type(matrix)
        data = edge_mask if native_type == "mask" else edge_dense_values
        inputs.append(
            {
                "name": name,
                "location": "edge",
                "type": native_type,
                "shape": [int(num_nodes), int(num_nodes)],
                "data": data,
            }
        )
        spec[name] = {"stage": "input", "location": "edge", "type": native_type}

    for name in sorted(vector_inputs):
        values = vector_inputs[name]
        if len(values) != num_nodes:
            continue
        pred_values = feature_columns.get(name)
        if pred_values is None:
            continue
        native_type = _sequence_native_type(values, num_nodes=num_nodes)
        typed_values = _cast_sequence_for_native_type(pred_values, native_type=native_type, num_nodes=num_nodes)
        inputs.append(
            {
                "name": name,
                "location": "node",
                "type": native_type,
                "shape": [int(num_nodes)],
                "data": typed_values,
            }
        )
        spec[name] = {"stage": "input", "location": "node", "type": native_type}

    for name in sorted(scalar_inputs):
        scalar_value: float | None = None
        one_hot_name = _normalize_compat_feature_name(f"{name}_one_hot")
        repeat_name = _normalize_compat_feature_name(f"{name}_repeat")
        if one_hot_name in feature_columns:
            values = feature_columns[one_hot_name]
            scalar_value = float(int(np.argmax(np.asarray(values, dtype=np.float32)))) if values else 0.0
            compat_values = [1.0 if idx == int(scalar_value) else 0.0 for idx in range(num_nodes)]
            inputs.append(
                {
                    "name": one_hot_name,
                    "location": "node",
                    "type": "scalar",
                    "shape": [int(num_nodes)],
                    "data": compat_values,
                }
            )
            spec[one_hot_name] = {"stage": "input", "location": "node", "type": "scalar"}
        elif repeat_name in feature_columns:
            values = feature_columns[repeat_name]
            scalar_value = float(sum(values) / max(1, len(values))) if values else 0.0
            compat_values = [float(scalar_value)] * num_nodes
            inputs.append(
                {
                    "name": repeat_name,
                    "location": "node",
                    "type": "scalar",
                    "shape": [int(num_nodes)],
                    "data": compat_values,
                }
            )
            spec[repeat_name] = {"stage": "input", "location": "node", "type": "scalar"}
        elif name in graph_inputs:
            scalar_value = float(graph_inputs[name])

        if scalar_value is None:
            continue
        inputs.append(
            {
                "name": name,
                "location": "graph",
                "type": "scalar",
                "shape": [1],
                "data": [float(scalar_value)],
            }
        )
        spec[name] = {"stage": "input", "location": "graph", "type": "scalar"}

    if not inputs:
        return _generic_graph_target_to_native_input_row(
            graph_target,
            sample_id=sample_id,
            split=split,
            sample_index=sample_index,
        )

    resolved_sample_id = sample_id.strip() or f"{algorithm}:{split}:{sample_index}"
    return {
        "sample_id": resolved_sample_id,
        "algorithm": algorithm,
        "split": split,
        "sample_index": int(sample_index),
        "num_steps": 0,
        "source": "clrs-text-native-input-adapter",
        "spec": spec,
        "inputs": inputs,
        "hints": [],
        "outputs": [],
        "feature_names": feature_names,
        "graph_inputs": graph_inputs,
        "adapter_version": "bridge_v2_native_input_v2",
    }


GRAPH_SOURCE_TASKS = {"bfs", "bellman_ford", "dag_shortest_paths", "dijkstra", "mst_prim"}
GRAPH_TASKS = {
    "articulation_points",
    "bellman_ford",
    "bfs",
    "bridges",
    "dag_shortest_paths",
    "dfs",
    "dijkstra",
    "floyd_warshall",
    "mst_kruskal",
    "mst_prim",
    "strongly_connected_components",
    "topological_sort",
}
STRING_TASKS = {"kmp_matcher", "lcs_length", "naive_string_matcher"}


def _native_pos(num_nodes: int) -> list[float]:
    n = max(0, int(num_nodes))
    denom = float(max(1, n))
    return [float(idx) / denom for idx in range(n)]


def _as_float_sequence(value: Any, *, name: str) -> list[float]:
    if isinstance(value, str):
        value = parse_text_value(value.split("\n", 1)[0].strip())
    if not _is_scalar_sequence(value):
        raise ValueError(f"Expected scalar sequence for {name}")
    return [float(v) for v in value]


def _as_int_sequence(value: Any, *, name: str) -> list[int]:
    if isinstance(value, str):
        value = parse_text_value(value.split("\n", 1)[0].strip())
    if not _is_scalar_sequence(value):
        raise ValueError(f"Expected integer sequence for {name}")
    return [int(round(float(v))) for v in value]


def _as_float_matrix(value: Any, *, name: str) -> list[list[float]]:
    if isinstance(value, str):
        value = parse_text_value(value.split("\n", 1)[0].strip())
    if not _is_matrix(value):
        raise ValueError(f"Expected matrix for {name}")
    return [[float(v) for v in row] for row in value]


def _node_input(name: str, value_type: str, data: Any, num_nodes: int) -> dict[str, Any]:
    return {
        "name": name,
        "location": "node",
        "type": value_type,
        "shape": list(np.asarray(data).shape) or [int(num_nodes)],
        "data": data,
    }


def _edge_input(name: str, value_type: str, data: Any, num_nodes: int) -> dict[str, Any]:
    return {
        "name": name,
        "location": "edge",
        "type": value_type,
        "shape": [int(num_nodes), int(num_nodes)],
        "data": data,
    }


def _graph_scalar_input(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "location": "graph",
        "type": "scalar",
        "shape": [],
        "data": float(value),
    }


def _one_hot(index: int, length: int) -> list[float]:
    idx = int(index)
    n = max(0, int(length))
    return [1.0 if i == idx else 0.0 for i in range(n)]


def _categorical_one_hot(values: list[int], *, num_classes: int = 4) -> list[list[float]]:
    classes = max(int(num_classes), max(values, default=0) + 1)
    out: list[list[float]] = []
    for value in values:
        idx = max(0, min(classes - 1, int(value)))
        out.append([1.0 if cls == idx else 0.0 for cls in range(classes)])
    return out


def _adjacency_from_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[1.0 if abs(float(value)) > 1e-12 else 0.0 for value in row] for row in matrix]


def _strict_benchmark_native_inputs(
    *,
    algorithm: str,
    raw_inputs: dict[str, Any],
    include_graph_scalar_compat_copies: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], int]:
    inputs: list[dict[str, Any]] = []
    spec: dict[str, dict[str, str]] = {}

    def add(dp: dict[str, Any]) -> None:
        inputs.append(dp)
        spec[str(dp["name"])] = {
            "stage": "input",
            "location": str(dp["location"]),
            "type": str(dp["type"]),
        }

    task = str(algorithm)

    if task in GRAPH_TASKS:
        A = _as_float_matrix(raw_inputs.get("A"), name="A")
        num_nodes = len(A)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        if task in GRAPH_SOURCE_TASKS:
            source = int(round(float(raw_inputs.get("s", 0))))
            add(_node_input("s", "mask_one", _one_hot(source, num_nodes), num_nodes))
            if include_graph_scalar_compat_copies:
                add(_graph_scalar_input("s", source))
        add(_edge_input("A", "scalar", A, num_nodes))
        add(_edge_input("adj", "mask", _adjacency_from_matrix(A), num_nodes))
        return inputs, spec, num_nodes

    if task in STRING_TASKS:
        key = _as_int_sequence(raw_inputs.get("key"), name="key")
        string = _as_float_sequence(raw_inputs.get("string"), name="string")
        num_nodes = len(key)
        if len(string) != num_nodes:
            raise ValueError(f"String task {task} has key/string length mismatch")
        add(_node_input("string", "mask", string, num_nodes))
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("key", "categorical", _categorical_one_hot(key, num_classes=4), num_nodes))
        return inputs, spec, num_nodes

    if task in {"segments_intersect", "graham_scan", "jarvis_march"}:
        x = _as_float_sequence(raw_inputs.get("x"), name="x")
        y = _as_float_sequence(raw_inputs.get("y"), name="y")
        if len(x) != len(y):
            raise ValueError(f"Geometry task {task} has x/y length mismatch")
        num_nodes = len(x)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("x", "scalar", x, num_nodes))
        add(_node_input("y", "scalar", y, num_nodes))
        return inputs, spec, num_nodes

    if task in {"activity_selector"}:
        s = _as_float_sequence(raw_inputs.get("s"), name="s")
        f = _as_float_sequence(raw_inputs.get("f"), name="f")
        if len(s) != len(f):
            raise ValueError("activity_selector has s/f length mismatch")
        num_nodes = len(s)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("s", "scalar", s, num_nodes))
        add(_node_input("f", "scalar", f, num_nodes))
        return inputs, spec, num_nodes

    if task in {"task_scheduling"}:
        d = _as_float_sequence(raw_inputs.get("d"), name="d")
        w = _as_float_sequence(raw_inputs.get("w"), name="w")
        if len(d) != len(w):
            raise ValueError("task_scheduling has d/w length mismatch")
        num_nodes = len(d)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("d", "scalar", d, num_nodes))
        add(_node_input("w", "scalar", w, num_nodes))
        return inputs, spec, num_nodes

    if task in {"binary_search"}:
        key = _as_float_sequence(raw_inputs.get("key"), name="key")
        num_nodes = len(key)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("key", "scalar", key, num_nodes))
        add(_graph_scalar_input("target", raw_inputs.get("target", 0.0)))
        return inputs, spec, num_nodes

    if task in {
        "bubble_sort",
        "find_maximum_subarray_kadane",
        "heapsort",
        "insertion_sort",
        "minimum",
        "quickselect",
        "quicksort",
    }:
        key = _as_float_sequence(raw_inputs.get("key"), name="key")
        num_nodes = len(key)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("key", "scalar", key, num_nodes))
        return inputs, spec, num_nodes

    if task in {"matrix_chain_order"}:
        p = _as_float_sequence(raw_inputs.get("p"), name="p")
        num_nodes = len(p)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("p", "scalar", p, num_nodes))
        return inputs, spec, num_nodes

    if task in {"optimal_bst"}:
        p = _as_float_sequence(raw_inputs.get("p"), name="p")
        q = _as_float_sequence(raw_inputs.get("q"), name="q")
        if len(p) != len(q):
            raise ValueError("optimal_bst has p/q length mismatch")
        num_nodes = len(p)
        add(_node_input("pos", "scalar", _native_pos(num_nodes), num_nodes))
        add(_node_input("p", "scalar", p, num_nodes))
        add(_node_input("q", "scalar", q, num_nodes))
        return inputs, spec, num_nodes

    return [], {}, 0


def canonical_to_native_input_row(
    canonical: dict[str, Any],
    *,
    feature_quant_bins: int = 256,
    sample_id: str = "",
    split: str = "bridge",
    sample_index: int = -1,
    include_graph_scalar_compat_copies: bool = False,
) -> dict[str, Any]:
    algorithm = str(canonical.get("algorithm", "unknown")).strip() or "unknown"
    raw_inputs = canonical.get("inputs", {})
    if not isinstance(raw_inputs, dict):
        graph_target = canonical_to_graph_target(canonical, feature_quant_bins=feature_quant_bins)
        return graph_target_to_native_input_row(
            graph_target,
            sample_id=sample_id,
            split=split,
            sample_index=sample_index,
        )

    num_nodes = infer_num_nodes_from_inputs(raw_inputs)
    strict_inputs, strict_spec, strict_num_nodes = _strict_benchmark_native_inputs(
        algorithm=algorithm,
        raw_inputs=raw_inputs,
        include_graph_scalar_compat_copies=include_graph_scalar_compat_copies,
    )
    if strict_inputs:
        resolved_sample_id = sample_id.strip() or f"{algorithm}:{split}:{sample_index}"
        return {
            "sample_id": resolved_sample_id,
            "algorithm": algorithm,
            "split": split,
            "sample_index": int(sample_index),
            "num_steps": 0,
            "source": "clrs-text-strict-benchmark-native-input-adapter",
            "spec": strict_spec,
            "inputs": strict_inputs,
            "hints": [],
            "outputs": [],
            "adapter_version": "bridge_v2_strict_benchmark_native_input_v1",
            "num_nodes": int(strict_num_nodes),
        }

    scalar_vectors = {
        str(name): values
        for name, values in raw_inputs.items()
        if _is_scalar_sequence(values)
    }
    matrix_inputs = {
        str(name): values
        for name, values in raw_inputs.items()
        if _is_matrix(values)
    }
    scalar_inputs = {
        str(name): value
        for name, value in raw_inputs.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }

    vector_lengths = {len(values) for values in scalar_vectors.values() if len(values) > 0}
    direct_native_supported = (
        ("A" in matrix_inputs and len(matrix_inputs["A"]) == num_nodes)
        or not scalar_vectors
        or len(vector_lengths) <= 1
    )
    if not direct_native_supported:
        graph_target = canonical_to_graph_target(canonical, feature_quant_bins=feature_quant_bins)
        return graph_target_to_native_input_row(
            graph_target,
            sample_id=sample_id,
            split=split,
            sample_index=sample_index,
        )

    inputs: list[dict[str, Any]] = []
    spec: dict[str, dict[str, str]] = {}

    for name in sorted(matrix_inputs):
        matrix = matrix_inputs[name]
        if len(matrix) != num_nodes:
            continue
        native_type = _matrix_native_type(matrix)
        inputs.append(
            {
                "name": name,
                "location": "edge",
                "type": native_type,
                "shape": [int(num_nodes), int(num_nodes)],
                "data": [[float(v) for v in row] for row in matrix],
            }
        )
        spec[name] = {"stage": "input", "location": "edge", "type": native_type}

    for name in sorted(scalar_vectors):
        values = scalar_vectors[name]
        if len(values) != num_nodes:
            continue
        native_type = _sequence_native_type(values, num_nodes=num_nodes)
        inputs.append(
            {
                "name": name,
                "location": "node",
                "type": native_type,
                "shape": [int(num_nodes)],
                "data": [float(v) if isinstance(v, float) else int(v) for v in values],
            }
        )
        spec[name] = {"stage": "input", "location": "node", "type": native_type}

    for name in sorted(scalar_inputs):
        value = scalar_inputs[name]
        inputs.append(
            {
                "name": name,
                "location": "graph",
                "type": "scalar",
                "shape": [1],
                "data": [float(value)],
            }
        )
        spec[name] = {"stage": "input", "location": "graph", "type": "scalar"}

        if include_graph_scalar_compat_copies:
            # Legacy V6 native-input conversion ignores graph-location inputs.
            if num_nodes > 0 and isinstance(value, int) and 0 <= int(value) < num_nodes:
                compat_name = f"{name}__one_hot"
                compat_data = [1.0 if idx == int(value) else 0.0 for idx in range(num_nodes)]
            else:
                compat_name = f"{name}__repeat"
                compat_data = [float(value)] * max(1, num_nodes)
            inputs.append(
                {
                    "name": compat_name,
                    "location": "node",
                    "type": "scalar",
                    "shape": [int(max(1, num_nodes))],
                    "data": compat_data,
                }
            )
            spec[compat_name] = {"stage": "input", "location": "node", "type": "scalar"}

    if not inputs:
        graph_target = canonical_to_graph_target(canonical, feature_quant_bins=feature_quant_bins)
        return graph_target_to_native_input_row(
            graph_target,
            sample_id=sample_id,
            split=split,
            sample_index=sample_index,
        )

    resolved_sample_id = sample_id.strip() or f"{algorithm}:{split}:{sample_index}"
    return {
        "sample_id": resolved_sample_id,
        "algorithm": algorithm,
        "split": split,
        "sample_index": int(sample_index),
        "num_steps": 0,
        "source": "clrs-text-canonical-native-input-adapter",
        "spec": spec,
        "inputs": inputs,
        "hints": [],
        "outputs": [],
        "adapter_version": "bridge_v2_canonical_native_input_v2",
    }


def canonical_to_target_text(canonical: dict[str, Any]) -> str:
    return stable_json_dumps(canonical)


def graph_target_to_text(graph_target: dict[str, Any]) -> str:
    return stable_json_dumps(graph_target)


def normalize_canonical_text(text: str) -> str:
    payload = json.loads(text)
    return canonical_to_target_text(payload)


def canonical_text_to_graph_target_text(text: str, *, feature_quant_bins: int = 256) -> str:
    payload = json.loads(text)
    graph_target = canonical_to_graph_target(payload, feature_quant_bins=feature_quant_bins)
    return graph_target_to_text(graph_target)
