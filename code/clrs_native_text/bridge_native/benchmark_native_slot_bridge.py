#!/usr/bin/env python3
"""Benchmark-native CLRS-Text bridge targets for the benchmark-native TRM.

This bridge family keeps the frozen LLM + QFormer/TransNAR trunk, but changes
the supervised target from projected graph IR to the benchmark-native slot
tensors consumed by the raw-probe TRM.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from clrs_native_text.bridge_native.bridge_hidden_state_cache import BridgeHiddenStateCache
from clrs_native_text.bridge_native.clrs_text_bridge_schema import canonical_to_native_input_row
from clrs_native_text.bridge_native.initial_style_bridge import (
    CrossAttentionLayer,
    DirectBridgeConfig,
    TransNARInterleaveLayer,
    _build_generic_positional_embeddings,
    _masked_mean_pool,
)
from trm_llm.stage3_specialists.train_clrs_trm_specialist_benchmark_native import (
    InputSlotSpec,
    _channel_count,
    _collect_slot_specs,
    _compute_slot_stats,
    _infer_max_input_discrete_value,
    _infer_num_nodes_from_inputs,
    _normalize_edge_input,
    _normalize_graph_input,
    _normalize_node_input,
    _slot_key,
)


STRICT_NATIVE_ADAPTER_VERSION = "bridge_v2_canonical_native_input_v2"
SCALAR_INPUT_TYPES = {"scalar"}
BINARY_INPUT_TYPES = {"mask", "mask_one"}
DISCRETE_INPUT_TYPES = {"pointer", "should_be_permutation", "categorical", "mask", "mask_one"}
COUNT_BLEND_ALPHA = 0.5


def _bridge_task_label(row: dict[str, Any], native_target: dict[str, Any] | None = None) -> str:
    for key in ("bridge_task_label", "algo_name", "algorithm", "task", "algo"):
        value = row.get(key)
        normalized = str(value).strip() if value is not None else ""
        if normalized and normalized.lower() != "unknown":
            return normalized
    if native_target:
        value = native_target.get("algorithm")
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _build_scalar_head(dim: int, out_dim: int, *, head_type: str, hidden_mult: float) -> nn.Module | None:
    if out_dim <= 0:
        return None
    normalized = head_type.strip().lower()
    if normalized == "linear":
        return nn.Linear(dim, out_dim)
    if normalized == "mlp":
        hidden_dim = max(out_dim, int(round(dim * max(0.25, float(hidden_mult)))))
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    raise ValueError(f"Unsupported scalar head type: {head_type}")


def _slot_is_scalar(slot: InputSlotSpec) -> bool:
    return str(slot.value_type) in SCALAR_INPUT_TYPES


def _slot_is_binary(slot: InputSlotSpec) -> bool:
    return str(slot.value_type) in BINARY_INPUT_TYPES


def _slot_is_mask_one(slot: InputSlotSpec) -> bool:
    return str(slot.value_type) == "mask_one"


def _slot_is_discrete(slot: InputSlotSpec) -> bool:
    return bool(slot.is_discrete)


def _slot_is_class(slot: InputSlotSpec) -> bool:
    return _slot_is_discrete(slot) and not _slot_is_binary(slot)


def _question_node_anchor_spans(question: str, *, max_nodes: int) -> list[list[tuple[int, int]]]:
    """Find text spans that explicitly bind a numeric node id to the instance."""
    spans: list[list[tuple[int, int]]] = [[] for _ in range(max_nodes)]

    def add_span(node_idx: int, span: tuple[int, int]) -> None:
        if 0 <= node_idx < max_nodes and span[1] > span[0]:
            spans[node_idx].append(span)

    for match in re.finditer(r"(?m)^id\s+(\d+)\b.*$", question):
        add_span(int(match.group(1)), match.span())

    for match in re.finditer(r"(?m)^(\d+)\s*->\s*(\d+)\b.*$", question):
        src = int(match.group(1))
        dst = int(match.group(2))
        add_span(src, match.span())
        add_span(dst, match.span())

    for match in re.finditer(r"(?mi)^Start/source index:\s*(\d+)\b.*$", question):
        add_span(int(match.group(1)), match.span())
    for match in re.finditer(r"(?mi)\bchosen\s+starting\s+index\s+is\s+(\d+)\b", question):
        add_span(int(match.group(1)), match.span())

    for src, dst, span in _question_edge_pair_spans(question, max_nodes=max_nodes):
        add_span(src, span)
        add_span(dst, span)

    return spans


def _build_node_anchor_mask(
    questions: list[str],
    offset_mapping: Any,
    *,
    max_nodes: int,
    seq_len: int,
) -> torch.Tensor:
    anchor_mask = torch.zeros((len(questions), max_nodes, seq_len), dtype=torch.float32)
    if offset_mapping is None:
        return anchor_mask
    offsets = offset_mapping.tolist() if hasattr(offset_mapping, "tolist") else offset_mapping
    for batch_idx, question in enumerate(questions):
        spans_by_node = _question_node_anchor_spans(question, max_nodes=max_nodes)
        if batch_idx >= len(offsets):
            continue
        for token_idx, token_offsets in enumerate(offsets[batch_idx][:seq_len]):
            if len(token_offsets) < 2:
                continue
            token_start = int(token_offsets[0])
            token_end = int(token_offsets[1])
            if token_end <= token_start:
                continue
            for node_idx, node_spans in enumerate(spans_by_node):
                if any(token_start < span_end and token_end > span_start for span_start, span_end in node_spans):
                    anchor_mask[batch_idx, node_idx, token_idx] = 1.0
    return anchor_mask


def _build_edge_text_prior(questions: list[str], *, max_nodes: int) -> torch.Tensor:
    edge_prior = torch.zeros((len(questions), max_nodes, max_nodes), dtype=torch.float32)
    for batch_idx, question in enumerate(questions):
        for src, dst, _span in _question_edge_pair_spans(question, max_nodes=max_nodes):
            edge_prior[batch_idx, src, dst] = 1.0
    return edge_prior


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_EDGE_HEADER_PATTERN = re.compile(r"(?i)\b(edge|edges|arc|arcs|link|links|connection|connections)\b")
_EDGE_LINE_PATTERN = re.compile(r"(?m)^\s*(\d+)\s*(?:->|>)\s*(\d+)\b")
_EDGE_TRIPLE_PATTERN = re.compile(rf"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*({_FLOAT_PATTERN})\s*\]")
_EDGE_PAIR_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
_NODE_SCALAR_LABEL_ALIASES = {
    "key": ("key", "keys", "value", "values", "array", "arrays", "number", "numbers"),
    "pos": ("pos", "position", "positions"),
}


def _question_edge_pair_spans(question: str, *, max_nodes: int) -> list[tuple[int, int, tuple[int, int]]]:
    pairs: dict[tuple[int, int], tuple[int, int]] = {}

    def add_pair(src: int, dst: int, span: tuple[int, int]) -> None:
        if 0 <= src < max_nodes and 0 <= dst < max_nodes:
            pairs.setdefault((src, dst), span)

    for match in _EDGE_LINE_PATTERN.finditer(question):
        add_pair(int(match.group(1)), int(match.group(2)), match.span())

    for match in _EDGE_TRIPLE_PATTERN.finditer(question):
        add_pair(int(match.group(1)), int(match.group(2)), match.span())

    offset = 0
    edge_context_lines = 0
    for line in question.splitlines(keepends=True):
        stripped = line.strip()
        if _EDGE_HEADER_PATTERN.search(stripped):
            edge_context_lines = 3
        if edge_context_lines > 0:
            for match in _EDGE_PAIR_PATTERN.finditer(line):
                add_pair(
                    int(match.group(1)),
                    int(match.group(2)),
                    (offset + match.start(), offset + match.end()),
                )
            if stripped:
                edge_context_lines -= 1
        offset += len(line)

    return [(src, dst, span) for (src, dst), span in sorted(pairs.items())]


def _normalized_scalar_value(raw_value: float, *, mean: float, std: float, clip: float) -> float:
    normalized = (float(raw_value) - float(mean)) / max(1e-6, float(std))
    if float(clip) > 0.0:
        normalized = max(-float(clip), min(float(clip), normalized))
    return float(normalized)


def _parse_json_prefix(text: str) -> Any | None:
    try:
        value, _end = json.JSONDecoder().raw_decode(text.strip())
        return value
    except json.JSONDecodeError:
        return None


def _normalized_text_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower().replace("_", " "))


def _slot_label_aliases(slot_name: str) -> set[str]:
    base = _normalized_text_label(slot_name)
    aliases = {base, base.rstrip("s"), f"{base}s"}
    for alias in _NODE_SCALAR_LABEL_ALIASES.get(str(slot_name), ()):
        normalized = _normalized_text_label(alias)
        aliases.add(normalized)
        aliases.add(normalized.rstrip("s"))
    return {alias for alias in aliases if alias}


def _coerce_scalar_channel(value: Any, channel_idx: int) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if channel_idx == 0 else None
    if isinstance(value, list):
        if 0 <= int(channel_idx) < len(value) and isinstance(value[int(channel_idx)], (int, float)):
            return float(value[int(channel_idx)])
    return None


def _question_count_value(question: str, *, max_nodes: int) -> int | None:
    patterns = [
        r"(?i)\bcount\s*(?:=|:)\s*(\d+)\b",
        r"(?i)\bn\s*=\s*(\d+)\b",
        r"(?i)\bthere\s+are\s+(\d+)\b",
        r"(?i)\bcontains\s+(\d+)\s+(?:zero[- ]indexed\s+)?(?:items?|nodes?|records?|vertices?)\b",
        r"(?i)\bwith\s+(\d+)\s+(?:numbered|indexed|zero[- ]indexed)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            count = int(match.group(1))
            if 1 <= count <= max_nodes:
                return count
    range_patterns = [
        r"(?i)\bids?\s+(?:are\s+)?(?:0|zero)\s*(?:through|to|\.\.)\s*(\d+)\b",
        r"(?i)\b(?:item|node|vertex)\s+labels\s+(?:are\s+)?(?:0|zero)\s*(?:through|to|\.\.)\s*(\d+)\b",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, question)
        if match:
            count = int(match.group(1)) + 1
            if 1 <= count <= max_nodes:
                return count
    return None


def _build_count_text_prior(questions: list[str], *, max_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    prior = torch.zeros((len(questions),), dtype=torch.float32)
    mask = torch.zeros((len(questions),), dtype=torch.float32)
    denom = float(max(1, max_nodes))
    for batch_idx, question in enumerate(questions):
        count = _question_count_value(question, max_nodes=max_nodes)
        if count is None:
            continue
        prior[batch_idx] = float(count) / denom
        mask[batch_idx] = 1.0
    return prior, mask


def _question_pos_values(question: str) -> dict[int, float]:
    values: dict[int, float] = {}
    pattern = re.compile(rf"(?m)^id\s+(\d+)\b[^\n]*?(?:^|[;,\s])pos\s*=\s*({_FLOAT_PATTERN})\b")
    for match in pattern.finditer(question):
        values[int(match.group(1))] = float(match.group(2))
    return values


def _question_node_scalar_values(
    question: str,
    *,
    slot_name: str,
    channel_idx: int,
    max_nodes: int,
) -> dict[int, float]:
    values: dict[int, float] = {}
    aliases = _slot_label_aliases(slot_name)

    for line in question.splitlines():
        by_index = re.match(r"(?i)^\s*([A-Za-z0-9_ /-]+?)\s+by\s+index\s*:\s*(.+?)\s*\.?\s*$", line)
        if by_index:
            label = _normalized_text_label(by_index.group(1))
            if label in aliases:
                parsed = _parse_json_prefix(by_index.group(2))
                if isinstance(parsed, list):
                    for node_idx, raw_value in enumerate(parsed[:max_nodes]):
                        scalar = _coerce_scalar_channel(raw_value, channel_idx)
                        if scalar is not None:
                            values[node_idx] = scalar
            continue

        record = re.match(r"(?i)^\s*id\s+(\d+)\b(.*)$", line)
        if not record:
            continue
        node_idx = int(record.group(1))
        if not (0 <= node_idx < max_nodes):
            continue
        for part in record.group(2).split(";"):
            if "=" not in part:
                continue
            key, raw = part.split("=", 1)
            if _normalized_text_label(key) not in aliases:
                continue
            parsed = _parse_json_prefix(raw)
            if parsed is None:
                parsed = raw.strip().rstrip(".")
            scalar = _coerce_scalar_channel(parsed, channel_idx)
            if scalar is not None:
                values[node_idx] = scalar

    return values


def _question_edge_values(question: str, *, max_nodes: int) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    line_pattern = re.compile(rf"(?m)^(\d+)\s*->\s*(\d+)\b[^\n]*?\(\s*({_FLOAT_PATTERN})\s*\)")
    for match in line_pattern.finditer(question):
        src = int(match.group(1))
        dst = int(match.group(2))
        if 0 <= src < max_nodes and 0 <= dst < max_nodes:
            values[(src, dst)] = float(match.group(3))
    triple_pattern = re.compile(rf"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*({_FLOAT_PATTERN})\s*\]")
    for match in triple_pattern.finditer(question):
        src = int(match.group(1))
        dst = int(match.group(2))
        if 0 <= src < max_nodes and 0 <= dst < max_nodes:
            values[(src, dst)] = float(match.group(3))
    return values


def _build_scalar_text_priors(
    questions: list[str],
    num_nodes: list[int],
    *,
    schema: "NativeTensorBridgeSchema",
    max_nodes: int,
    continuous_feature_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_prior = torch.zeros((len(questions), max_nodes, schema.node_slot_dim), dtype=torch.float32)
    node_prior_mask = torch.zeros_like(node_prior)
    edge_prior = torch.zeros((len(questions), max_nodes, max_nodes, schema.edge_slot_dim), dtype=torch.float32)
    edge_prior_mask = torch.zeros_like(edge_prior)
    node_slots = schema.node_slots
    edge_slots = schema.edge_slots

    node_scalar_slots = [
        (slot_idx, slot)
        for slot_idx, slot in enumerate(node_slots)
        if _slot_is_scalar(slot)
    ]
    pos_slots = [(slot_idx, slot) for slot_idx, slot in node_scalar_slots if slot.name == "pos"]
    edge_a_slots = [
        (slot_idx, slot)
        for slot_idx, slot in enumerate(edge_slots)
        if slot.name == "A" and _slot_is_scalar(slot)
    ]
    if not node_scalar_slots and not edge_a_slots:
        return node_prior, node_prior_mask, edge_prior, edge_prior_mask

    for batch_idx, question in enumerate(questions):
        n = max(0, min(max_nodes, int(num_nodes[batch_idx]) if batch_idx < len(num_nodes) else 0))
        parsed_pos = _question_pos_values(question)
        for slot_idx, _slot in pos_slots:
            mean = float(schema.node_slot_mean[slot_idx]) if slot_idx < len(schema.node_slot_mean) else 0.0
            std = max(1e-6, float(schema.node_slot_std[slot_idx]) if slot_idx < len(schema.node_slot_std) else 1.0)
            denom = float(max(1, n))
            for node_idx in range(n):
                raw_value = parsed_pos.get(node_idx, float(node_idx) / denom)
                node_prior[batch_idx, node_idx, slot_idx] = _normalized_scalar_value(
                    raw_value,
                    mean=mean,
                    std=std,
                    clip=continuous_feature_clip,
                )
                node_prior_mask[batch_idx, node_idx, slot_idx] = 1.0

        for slot_idx, slot in node_scalar_slots:
            if slot.name == "pos":
                continue
            parsed_values = _question_node_scalar_values(
                question,
                slot_name=slot.name,
                channel_idx=int(slot.channel_idx),
                max_nodes=max_nodes,
            )
            if not parsed_values:
                continue
            mean = float(schema.node_slot_mean[slot_idx]) if slot_idx < len(schema.node_slot_mean) else 0.0
            std = max(1e-6, float(schema.node_slot_std[slot_idx]) if slot_idx < len(schema.node_slot_std) else 1.0)
            for node_idx in range(n):
                if node_idx not in parsed_values:
                    continue
                node_prior[batch_idx, node_idx, slot_idx] = _normalized_scalar_value(
                    parsed_values[node_idx],
                    mean=mean,
                    std=std,
                    clip=continuous_feature_clip,
                )
                node_prior_mask[batch_idx, node_idx, slot_idx] = 1.0

        parsed_edges = _question_edge_values(question, max_nodes=max_nodes)
        if parsed_edges:
            for slot_idx, _slot in edge_a_slots:
                mean = float(schema.edge_slot_mean[slot_idx]) if slot_idx < len(schema.edge_slot_mean) else 0.0
                std = max(1e-6, float(schema.edge_slot_std[slot_idx]) if slot_idx < len(schema.edge_slot_std) else 1.0)
                for src in range(n):
                    for dst in range(n):
                        raw_value = parsed_edges.get((src, dst), 0.0)
                        edge_prior[batch_idx, src, dst, slot_idx] = _normalized_scalar_value(
                            raw_value,
                            mean=mean,
                            std=std,
                            clip=continuous_feature_clip,
                        )
                        edge_prior_mask[batch_idx, src, dst, slot_idx] = 1.0

    return node_prior, node_prior_mask, edge_prior, edge_prior_mask


def _slot_specs_to_payload(slots: list[InputSlotSpec]) -> list[dict[str, Any]]:
    return [slot.__dict__.copy() for slot in slots]


def _slot_specs_from_payload(payload: list[dict[str, Any]]) -> list[InputSlotSpec]:
    return [
        InputSlotSpec(
            key=str(item["key"]),
            name=str(item["name"]),
            location=str(item["location"]),
            value_type=str(item["value_type"]),
            channel_idx=int(item["channel_idx"]),
            is_discrete=bool(item["is_discrete"]),
        )
        for item in payload
    ]


@dataclass
class NativeTensorBridgeSchema:
    task_names: list[str]
    node_slot_specs: list[dict[str, Any]]
    edge_slot_specs: list[dict[str, Any]]
    graph_slot_specs: list[dict[str, Any]]
    node_slot_mean: list[float]
    node_slot_std: list[float]
    edge_slot_mean: list[float]
    edge_slot_std: list[float]
    graph_slot_mean: list[float]
    graph_slot_std: list[float]
    max_input_discrete_value: int
    task_node_slot_active: list[list[float]]
    task_edge_slot_active: list[list[float]]
    task_graph_slot_active: list[list[float]]
    strict_native_only: bool = True

    @property
    def node_slots(self) -> list[InputSlotSpec]:
        return _slot_specs_from_payload(self.node_slot_specs)

    @property
    def edge_slots(self) -> list[InputSlotSpec]:
        return _slot_specs_from_payload(self.edge_slot_specs)

    @property
    def graph_slots(self) -> list[InputSlotSpec]:
        return _slot_specs_from_payload(self.graph_slot_specs)

    @property
    def node_slot_dim(self) -> int:
        return len(self.node_slot_specs)

    @property
    def edge_slot_dim(self) -> int:
        return len(self.edge_slot_specs)

    @property
    def graph_slot_dim(self) -> int:
        return len(self.graph_slot_specs)

    def task_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.task_names)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NativeTensorBridgeSchema":
        return cls(
            task_names=[str(v) for v in payload.get("task_names", [])],
            node_slot_specs=[dict(v) for v in payload.get("node_slot_specs", [])],
            edge_slot_specs=[dict(v) for v in payload.get("edge_slot_specs", [])],
            graph_slot_specs=[dict(v) for v in payload.get("graph_slot_specs", [])],
            node_slot_mean=[float(v) for v in payload.get("node_slot_mean", [])],
            node_slot_std=[max(1e-6, float(v)) for v in payload.get("node_slot_std", [])],
            edge_slot_mean=[float(v) for v in payload.get("edge_slot_mean", [])],
            edge_slot_std=[max(1e-6, float(v)) for v in payload.get("edge_slot_std", [])],
            graph_slot_mean=[float(v) for v in payload.get("graph_slot_mean", [])],
            graph_slot_std=[max(1e-6, float(v)) for v in payload.get("graph_slot_std", [])],
            max_input_discrete_value=int(payload.get("max_input_discrete_value", 2)),
            task_node_slot_active=[
                [float(x) for x in row]
                for row in payload.get("task_node_slot_active", [])
            ],
            task_edge_slot_active=[
                [float(x) for x in row]
                for row in payload.get("task_edge_slot_active", [])
            ],
            task_graph_slot_active=[
                [float(x) for x in row]
                for row in payload.get("task_graph_slot_active", [])
            ],
            strict_native_only=bool(payload.get("strict_native_only", True)),
        )


@dataclass
class NativeTensorBridgeBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    task_ids: torch.Tensor
    present_target: torch.Tensor
    node_feature_values: torch.Tensor
    node_feature_bins: torch.Tensor
    node_slot_mask: torch.Tensor
    edge_feature_values: torch.Tensor
    edge_feature_bins: torch.Tensor
    edge_slot_mask: torch.Tensor
    graph_feature_values: torch.Tensor
    graph_feature_bins: torch.Tensor
    graph_slot_mask: torch.Tensor
    num_nodes: torch.Tensor
    sample_ids: list[str]
    tasks: list[str]
    adapter_versions: list[str]
    node_scalar_prior: torch.Tensor | None = None
    node_scalar_prior_mask: torch.Tensor | None = None
    edge_scalar_prior: torch.Tensor | None = None
    edge_scalar_prior_mask: torch.Tensor | None = None
    count_prior: torch.Tensor | None = None
    count_prior_mask: torch.Tensor | None = None
    node_anchor_mask: torch.Tensor | None = None
    edge_text_prior: torch.Tensor | None = None
    llm_hidden_states: torch.Tensor | None = None


def attach_native_targets(
    rows: list[dict[str, Any]],
    *,
    strict_native_only: bool,
    feature_quant_bins: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    version_counts: dict[str, int] = {}
    dropped = 0
    for sample_index, row in enumerate(rows):
        provided_target = row.get("native_input_target")
        if isinstance(provided_target, dict) and isinstance(provided_target.get("inputs"), list):
            native_target = dict(provided_target)
            native_target.setdefault("sample_id", str(row.get("sample_id", sample_index)))
            native_target.setdefault("split", str(row.get("source_split", "bridge")))
            native_target.setdefault("sample_index", sample_index)
        else:
            native_target = canonical_to_native_input_row(
                row.get("canonical_inputs", {}),
                feature_quant_bins=int(feature_quant_bins),
                sample_id=str(row.get("sample_id", sample_index)),
                split=str(row.get("source_split", "bridge")),
                sample_index=sample_index,
            )
        adapter_version = str(native_target.get("adapter_version", "unknown"))
        version_counts[adapter_version] = version_counts.get(adapter_version, 0) + 1
        if strict_native_only and adapter_version not in {STRICT_NATIVE_ADAPTER_VERSION, "bridge_v2_strict_benchmark_native_input_v1"}:
            dropped += 1
            continue
        enriched = dict(row)
        enriched["native_input_target"] = native_target
        attached.append(enriched)
    stats = {
        "total_rows": len(rows),
        "kept_rows": len(attached),
        "dropped_rows": dropped,
        "strict_native_only": bool(strict_native_only),
        "adapter_versions": version_counts,
    }
    return attached, stats


def _task_slot_active_masks(
    rows: list[dict[str, Any]],
    *,
    task_names: list[str],
    node_slot_specs: list[InputSlotSpec],
    edge_slot_specs: list[InputSlotSpec],
    graph_slot_specs: list[InputSlotSpec],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    task_to_idx = {task: idx for idx, task in enumerate(task_names)}
    node_slot_to_idx = {slot.key: idx for idx, slot in enumerate(node_slot_specs)}
    edge_slot_to_idx = {slot.key: idx for idx, slot in enumerate(edge_slot_specs)}
    graph_slot_to_idx = {slot.key: idx for idx, slot in enumerate(graph_slot_specs)}

    node_masks = [[0.0 for _ in range(len(node_slot_specs))] for _ in range(len(task_names))]
    edge_masks = [[0.0 for _ in range(len(edge_slot_specs))] for _ in range(len(task_names))]
    graph_masks = [[0.0 for _ in range(len(graph_slot_specs))] for _ in range(len(task_names))]

    for row in rows:
        task = _bridge_task_label(row, row.get("native_input_target", {}))
        task_idx = task_to_idx.get(task)
        if task_idx is None:
            continue
        native_target = row.get("native_input_target", {})
        inputs = native_target.get("inputs", [])
        if not isinstance(inputs, list):
            continue
        for dp in inputs:
            location = str(dp.get("location", "")).strip()
            value_type = str(dp.get("type", "")).strip()
            name = str(dp.get("name", "")).strip() or f"{location}_input"
            channels = _channel_count(dp)
            for channel_idx in range(max(0, channels)):
                key = _slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx)
                if location == "node":
                    slot_idx = node_slot_to_idx.get(key)
                    if slot_idx is not None:
                        node_masks[task_idx][slot_idx] = 1.0
                elif location == "edge":
                    slot_idx = edge_slot_to_idx.get(key)
                    if slot_idx is not None:
                        edge_masks[task_idx][slot_idx] = 1.0
                elif location == "graph":
                    slot_idx = graph_slot_to_idx.get(key)
                    if slot_idx is not None:
                        graph_masks[task_idx][slot_idx] = 1.0
    return node_masks, edge_masks, graph_masks


def build_native_tensor_bridge_schema(
    rows: list[dict[str, Any]],
    *,
    strict_native_only: bool,
    feature_quant_bins: int = 256,
) -> tuple[NativeTensorBridgeSchema, dict[str, Any]]:
    attached_rows, attach_stats = attach_native_targets(
        rows,
        strict_native_only=strict_native_only,
        feature_quant_bins=feature_quant_bins,
    )
    if not attached_rows:
        raise ValueError("No rows remain for native bridge training after native-target filtering")

    native_rows = [dict(row["native_input_target"]) for row in attached_rows]
    task_names = sorted({_bridge_task_label(row, row["native_input_target"]) for row in attached_rows})
    node_slot_specs, edge_slot_specs, graph_slot_specs = _collect_slot_specs(native_rows)
    node_slot_mean, node_slot_std = _compute_slot_stats(native_rows, node_slot_specs)
    edge_slot_mean, edge_slot_std = _compute_slot_stats(native_rows, edge_slot_specs)
    graph_slot_mean, graph_slot_std = _compute_slot_stats(native_rows, graph_slot_specs)
    max_input_discrete_value = _infer_max_input_discrete_value(native_rows)
    task_node_slot_active, task_edge_slot_active, task_graph_slot_active = _task_slot_active_masks(
        attached_rows,
        task_names=task_names,
        node_slot_specs=node_slot_specs,
        edge_slot_specs=edge_slot_specs,
        graph_slot_specs=graph_slot_specs,
    )
    schema = NativeTensorBridgeSchema(
        task_names=task_names,
        node_slot_specs=_slot_specs_to_payload(node_slot_specs),
        edge_slot_specs=_slot_specs_to_payload(edge_slot_specs),
        graph_slot_specs=_slot_specs_to_payload(graph_slot_specs),
        node_slot_mean=node_slot_mean,
        node_slot_std=node_slot_std,
        edge_slot_mean=edge_slot_mean,
        edge_slot_std=edge_slot_std,
        graph_slot_mean=graph_slot_mean,
        graph_slot_std=graph_slot_std,
        max_input_discrete_value=max_input_discrete_value,
        task_node_slot_active=task_node_slot_active,
        task_edge_slot_active=task_edge_slot_active,
        task_graph_slot_active=task_graph_slot_active,
        strict_native_only=bool(strict_native_only),
    )
    attach_stats["task_names"] = list(task_names)
    attach_stats["node_slot_dim"] = len(node_slot_specs)
    attach_stats["edge_slot_dim"] = len(edge_slot_specs)
    attach_stats["graph_slot_dim"] = len(graph_slot_specs)
    attach_stats["max_input_discrete_value"] = int(max_input_discrete_value)
    return schema, attach_stats


class CLRSTextBenchmarkNativeBridgeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        schema: NativeTensorBridgeSchema,
        max_nodes: int,
        strict_native_only: bool,
        continuous_feature_clip: float,
        feature_quant_bins: int = 256,
    ) -> None:
        self.max_nodes = int(max_nodes)
        self.schema = schema
        self.strict_native_only = bool(strict_native_only)
        self.continuous_feature_clip = float(continuous_feature_clip)
        attached_rows, stats = attach_native_targets(
            rows,
            strict_native_only=self.strict_native_only,
            feature_quant_bins=feature_quant_bins,
        )
        if not attached_rows:
            raise ValueError("No rows remain for native bridge dataset after filtering")
        self.rows = attached_rows
        self.filter_stats = stats
        self.task_to_idx = schema.task_index()
        self.node_slot_specs = schema.node_slots
        self.edge_slot_specs = schema.edge_slots
        self.graph_slot_specs = schema.graph_slots
        self.node_slot_to_idx = {slot.key: idx for idx, slot in enumerate(self.node_slot_specs)}
        self.edge_slot_to_idx = {slot.key: idx for idx, slot in enumerate(self.edge_slot_specs)}
        self.graph_slot_to_idx = {slot.key: idx for idx, slot in enumerate(self.graph_slot_specs)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        target = dict(row["native_input_target"])
        question = str(row.get("question", ""))
        sample_id = str(row.get("sample_id", idx))
        task = _bridge_task_label(row, target)
        task_id = int(self.task_to_idx.get(task, 0))
        adapter_version = str(target.get("adapter_version", "unknown"))

        inputs = list(target.get("inputs", []))
        num_nodes = _infer_num_nodes_from_inputs(inputs)
        if num_nodes <= 0:
            num_nodes = max(0, int(target.get("num_nodes", 0)))
        if num_nodes > self.max_nodes:
            raise ValueError(f"sample {sample_id} num_nodes={num_nodes} exceeds max_nodes={self.max_nodes}")

        present_target = torch.zeros(self.max_nodes, dtype=torch.float32)
        if num_nodes > 0:
            present_target[:num_nodes] = 1.0

        node_feature_values = torch.zeros((self.max_nodes, self.schema.node_slot_dim), dtype=torch.float32)
        node_feature_bins = torch.zeros((self.max_nodes, self.schema.node_slot_dim), dtype=torch.long)
        node_slot_mask = torch.zeros((self.max_nodes, self.schema.node_slot_dim), dtype=torch.float32)

        edge_feature_values = torch.zeros((self.max_nodes, self.max_nodes, self.schema.edge_slot_dim), dtype=torch.float32)
        edge_feature_bins = torch.zeros((self.max_nodes, self.max_nodes, self.schema.edge_slot_dim), dtype=torch.long)
        edge_slot_mask = torch.zeros((self.max_nodes, self.max_nodes, self.schema.edge_slot_dim), dtype=torch.float32)

        graph_feature_values = torch.zeros((self.schema.graph_slot_dim,), dtype=torch.float32)
        graph_feature_bins = torch.zeros((self.schema.graph_slot_dim,), dtype=torch.long)
        graph_slot_mask = torch.zeros((self.schema.graph_slot_dim,), dtype=torch.float32)

        for dp in inputs:
            location = str(dp.get("location", "")).strip()
            value_type = str(dp.get("type", "")).strip()
            name = str(dp.get("name", "")).strip() or f"{location}_input"
            if location == "node":
                values = _normalize_node_input(dp, num_nodes=num_nodes)
                for channel_idx in range(values.shape[1]):
                    slot_idx = self.node_slot_to_idx.get(_slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx))
                    if slot_idx is None:
                        continue
                    mean = float(self.schema.node_slot_mean[slot_idx]) if slot_idx < len(self.schema.node_slot_mean) else 0.0
                    std = max(1e-6, float(self.schema.node_slot_std[slot_idx]) if slot_idx < len(self.schema.node_slot_std) else 1.0)
                    for node_idx in range(num_nodes):
                        raw_value = float(values[node_idx, channel_idx])
                        node_slot_mask[node_idx, slot_idx] = 1.0
                        if _slot_is_discrete(self.node_slot_specs[slot_idx]):
                            node_feature_bins[node_idx, slot_idx] = max(0, int(round(raw_value)))
                        else:
                            normalized = (raw_value - mean) / std
                            if self.continuous_feature_clip > 0.0:
                                normalized = max(-self.continuous_feature_clip, min(self.continuous_feature_clip, normalized))
                            node_feature_values[node_idx, slot_idx] = float(normalized)
            elif location == "edge" and self.schema.edge_slot_dim > 0:
                values = _normalize_edge_input(dp, num_nodes=num_nodes)
                for channel_idx in range(values.shape[2]):
                    slot_idx = self.edge_slot_to_idx.get(_slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx))
                    if slot_idx is None:
                        continue
                    mean = float(self.schema.edge_slot_mean[slot_idx]) if slot_idx < len(self.schema.edge_slot_mean) else 0.0
                    std = max(1e-6, float(self.schema.edge_slot_std[slot_idx]) if slot_idx < len(self.schema.edge_slot_std) else 1.0)
                    for src in range(num_nodes):
                        for dst in range(num_nodes):
                            raw_value = float(values[src, dst, channel_idx])
                            edge_slot_mask[src, dst, slot_idx] = 1.0
                            if _slot_is_discrete(self.edge_slot_specs[slot_idx]):
                                edge_feature_bins[src, dst, slot_idx] = max(0, int(round(raw_value)))
                            else:
                                normalized = (raw_value - mean) / std
                                if self.continuous_feature_clip > 0.0:
                                    normalized = max(-self.continuous_feature_clip, min(self.continuous_feature_clip, normalized))
                                edge_feature_values[src, dst, slot_idx] = float(normalized)
            elif location == "graph":
                values = _normalize_graph_input(dp)
                for channel_idx in range(values.shape[0]):
                    slot_idx = self.graph_slot_to_idx.get(_slot_key(location=location, value_type=value_type, name=name, channel_idx=channel_idx))
                    if slot_idx is None:
                        continue
                    raw_value = float(values[channel_idx])
                    graph_slot_mask[slot_idx] = 1.0
                    if _slot_is_discrete(self.graph_slot_specs[slot_idx]):
                        graph_feature_bins[slot_idx] = max(0, int(round(raw_value)))
                    else:
                        mean = float(self.schema.graph_slot_mean[slot_idx]) if slot_idx < len(self.schema.graph_slot_mean) else 0.0
                        std = max(1e-6, float(self.schema.graph_slot_std[slot_idx]) if slot_idx < len(self.schema.graph_slot_std) else 1.0)
                        normalized = (raw_value - mean) / std
                        if self.continuous_feature_clip > 0.0:
                            normalized = max(-self.continuous_feature_clip, min(self.continuous_feature_clip, normalized))
                        graph_feature_values[slot_idx] = float(normalized)

        return {
            "question": question,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "present_target": present_target,
            "node_feature_values": node_feature_values,
            "node_feature_bins": node_feature_bins,
            "node_slot_mask": node_slot_mask,
            "edge_feature_values": edge_feature_values,
            "edge_feature_bins": edge_feature_bins,
            "edge_slot_mask": edge_slot_mask,
            "graph_feature_values": graph_feature_values,
            "graph_feature_bins": graph_feature_bins,
            "graph_slot_mask": graph_slot_mask,
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            "sample_id": sample_id,
            "task": task,
            "adapter_version": adapter_version,
        }


def make_native_tensor_bridge_collate(
    tokenizer,
    *,
    max_source_length: int,
    schema: NativeTensorBridgeSchema | None = None,
    continuous_feature_clip: float = 5.0,
):
    def collate(batch: list[dict[str, Any]]) -> NativeTensorBridgeBatch:
        questions = [item["question"] for item in batch]
        try:
            enc = tokenizer(
                questions,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_source_length,
                return_offsets_mapping=True,
            )
            offset_mapping = enc.pop("offset_mapping")
        except (NotImplementedError, TypeError, ValueError):
            enc = tokenizer(
                questions,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_source_length,
            )
            offset_mapping = None
        max_nodes = int(batch[0]["present_target"].shape[0]) if batch else 0
        seq_len = int(enc["input_ids"].shape[1])
        node_feature_values = torch.stack([item["node_feature_values"] for item in batch], dim=0)
        edge_feature_values = torch.stack([item["edge_feature_values"] for item in batch], dim=0)
        num_nodes = torch.stack([item["num_nodes"] for item in batch], dim=0)
        node_scalar_prior = torch.zeros_like(node_feature_values)
        node_scalar_prior_mask = torch.zeros_like(node_feature_values)
        edge_scalar_prior = torch.zeros_like(edge_feature_values)
        edge_scalar_prior_mask = torch.zeros_like(edge_feature_values)
        count_prior, count_prior_mask = _build_count_text_prior(questions, max_nodes=max_nodes)
        if schema is not None:
            priors = _build_scalar_text_priors(
                questions,
                [int(v.item()) for v in num_nodes],
                schema=schema,
                max_nodes=max_nodes,
                continuous_feature_clip=float(continuous_feature_clip),
            )
            node_scalar_prior, node_scalar_prior_mask, edge_scalar_prior, edge_scalar_prior_mask = priors
        return NativeTensorBridgeBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            task_ids=torch.stack([item["task_id"] for item in batch], dim=0),
            llm_hidden_states=None,
            present_target=torch.stack([item["present_target"] for item in batch], dim=0),
            node_feature_values=node_feature_values,
            node_feature_bins=torch.stack([item["node_feature_bins"] for item in batch], dim=0),
            node_slot_mask=torch.stack([item["node_slot_mask"] for item in batch], dim=0),
            edge_feature_values=edge_feature_values,
            edge_feature_bins=torch.stack([item["edge_feature_bins"] for item in batch], dim=0),
            edge_slot_mask=torch.stack([item["edge_slot_mask"] for item in batch], dim=0),
            graph_feature_values=torch.stack([item["graph_feature_values"] for item in batch], dim=0),
            graph_feature_bins=torch.stack([item["graph_feature_bins"] for item in batch], dim=0),
            graph_slot_mask=torch.stack([item["graph_slot_mask"] for item in batch], dim=0),
            num_nodes=num_nodes,
            sample_ids=[str(item["sample_id"]) for item in batch],
            tasks=[str(item["task"]) for item in batch],
            adapter_versions=[str(item["adapter_version"]) for item in batch],
            node_scalar_prior=node_scalar_prior,
            node_scalar_prior_mask=node_scalar_prior_mask,
            edge_scalar_prior=edge_scalar_prior,
            edge_scalar_prior_mask=edge_scalar_prior_mask,
            count_prior=count_prior,
            count_prior_mask=count_prior_mask,
            node_anchor_mask=_build_node_anchor_mask(
                questions,
                offset_mapping,
                max_nodes=max_nodes,
                seq_len=seq_len,
            ),
            edge_text_prior=_build_edge_text_prior(questions, max_nodes=max_nodes),
        )

    return collate


class CachedBenchmarkNativeBridgeDataset(Dataset):
    def __init__(self, base_dataset: CLRSTextBenchmarkNativeBridgeDataset, *, cache: BridgeHiddenStateCache) -> None:
        self.base_dataset = base_dataset
        self.cache = cache
        if len(base_dataset) != len(cache.sample_ids):
            raise ValueError(
                f"Cached hidden-state rows ({len(cache.sample_ids)}) do not match dataset rows ({len(base_dataset)})"
            )
        for idx, row in enumerate(base_dataset.rows):
            sample_id = str(row.get("sample_id", idx))
            cache_id = str(cache.sample_ids[idx])
            if sample_id != cache_id:
                raise ValueError(
                    f"Hidden-state cache sample_id mismatch at row {idx}: dataset={sample_id!r} cache={cache_id!r}"
                )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.base_dataset[idx])
        hidden_states = self.cache.get_hidden_states(idx).to(torch.float32)
        item["cached_hidden_states"] = hidden_states
        item["cached_attention_mask"] = torch.ones((hidden_states.shape[0],), dtype=torch.long)
        return item


def make_cached_native_tensor_bridge_collate(
    *,
    schema: NativeTensorBridgeSchema | None = None,
    continuous_feature_clip: float = 5.0,
):
    def collate(batch: list[dict[str, Any]]) -> NativeTensorBridgeBatch:
        batch_size = len(batch)
        max_seq_len = max(int(item["cached_hidden_states"].shape[0]) for item in batch)
        hidden_size = int(batch[0]["cached_hidden_states"].shape[1]) if batch else 0
        llm_hidden_states = torch.zeros((batch_size, max_seq_len, hidden_size), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        for idx, item in enumerate(batch):
            hidden = item["cached_hidden_states"]
            length = int(hidden.shape[0])
            llm_hidden_states[idx, :length] = hidden
            attention_mask[idx, :length] = 1
        max_nodes = int(batch[0]["present_target"].shape[0]) if batch else 0
        questions = [str(item["question"]) for item in batch]
        node_feature_values = torch.stack([item["node_feature_values"] for item in batch], dim=0)
        edge_feature_values = torch.stack([item["edge_feature_values"] for item in batch], dim=0)
        num_nodes = torch.stack([item["num_nodes"] for item in batch], dim=0)
        node_scalar_prior = torch.zeros_like(node_feature_values)
        node_scalar_prior_mask = torch.zeros_like(node_feature_values)
        edge_scalar_prior = torch.zeros_like(edge_feature_values)
        edge_scalar_prior_mask = torch.zeros_like(edge_feature_values)
        count_prior, count_prior_mask = _build_count_text_prior(questions, max_nodes=max_nodes)
        if schema is not None:
            priors = _build_scalar_text_priors(
                questions,
                [int(v.item()) for v in num_nodes],
                schema=schema,
                max_nodes=max_nodes,
                continuous_feature_clip=float(continuous_feature_clip),
            )
            node_scalar_prior, node_scalar_prior_mask, edge_scalar_prior, edge_scalar_prior_mask = priors
        return NativeTensorBridgeBatch(
            input_ids=torch.zeros((batch_size, 0), dtype=torch.long),
            attention_mask=attention_mask,
            task_ids=torch.stack([item["task_id"] for item in batch], dim=0),
            llm_hidden_states=llm_hidden_states,
            present_target=torch.stack([item["present_target"] for item in batch], dim=0),
            node_feature_values=node_feature_values,
            node_feature_bins=torch.stack([item["node_feature_bins"] for item in batch], dim=0),
            node_slot_mask=torch.stack([item["node_slot_mask"] for item in batch], dim=0),
            edge_feature_values=edge_feature_values,
            edge_feature_bins=torch.stack([item["edge_feature_bins"] for item in batch], dim=0),
            edge_slot_mask=torch.stack([item["edge_slot_mask"] for item in batch], dim=0),
            graph_feature_values=torch.stack([item["graph_feature_values"] for item in batch], dim=0),
            graph_feature_bins=torch.stack([item["graph_feature_bins"] for item in batch], dim=0),
            graph_slot_mask=torch.stack([item["graph_slot_mask"] for item in batch], dim=0),
            num_nodes=num_nodes,
            sample_ids=[str(item["sample_id"]) for item in batch],
            tasks=[str(item["task"]) for item in batch],
            adapter_versions=[str(item["adapter_version"]) for item in batch],
            node_scalar_prior=node_scalar_prior,
            node_scalar_prior_mask=node_scalar_prior_mask,
            edge_scalar_prior=edge_scalar_prior,
            edge_scalar_prior_mask=edge_scalar_prior_mask,
            count_prior=count_prior,
            count_prior_mask=count_prior_mask,
            node_anchor_mask=torch.zeros((batch_size, max_nodes, max_seq_len), dtype=torch.float32),
            edge_text_prior=_build_edge_text_prior(questions, max_nodes=max_nodes),
        )

    return collate


class NativeTensorBridgeOutputHeads(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        node_slot_dim: int,
        edge_slot_dim: int,
        graph_slot_dim: int,
        num_tasks: int,
        max_input_discrete_value: int,
        scalar_head_type: str = "linear",
        scalar_head_hidden_mult: float = 1.0,
        edge_prior_binary_slot_mask: torch.Tensor | None = None,
        edge_text_prior_logit_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.task_embed = nn.Embedding(max(1, num_tasks), dim)
        self.task_node_proj = nn.Linear(dim, dim)
        self.task_global_proj = nn.Linear(dim, dim)
        self.present_head = nn.Linear(dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.count_class_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, max_input_discrete_value + 1),
        )
        self.node_scalar_head = _build_scalar_head(
            dim,
            node_slot_dim,
            head_type=scalar_head_type,
            hidden_mult=scalar_head_hidden_mult,
        )
        self.node_discrete_head = nn.Linear(dim, node_slot_dim * max(1, max_input_discrete_value)) if node_slot_dim > 0 else None
        self.edge_pair_proj = nn.Linear(dim * 4, dim) if edge_slot_dim > 0 else None
        self.edge_text_prior_proj = nn.Linear(1, dim, bias=False) if edge_slot_dim > 0 else None
        self.edge_scalar_head = _build_scalar_head(
            dim,
            edge_slot_dim,
            head_type=scalar_head_type,
            hidden_mult=scalar_head_hidden_mult,
        )
        self.edge_discrete_head = nn.Linear(dim, edge_slot_dim * max(1, max_input_discrete_value)) if edge_slot_dim > 0 else None
        self.graph_scalar_head = _build_scalar_head(
            dim,
            graph_slot_dim,
            head_type=scalar_head_type,
            hidden_mult=scalar_head_hidden_mult,
        )
        self.graph_discrete_head = nn.Linear(dim, graph_slot_dim * max(1, max_input_discrete_value)) if graph_slot_dim > 0 else None
        self.node_slot_dim = int(node_slot_dim)
        self.edge_slot_dim = int(edge_slot_dim)
        self.graph_slot_dim = int(graph_slot_dim)
        self.max_input_discrete_value = int(max(1, max_input_discrete_value))
        self.scalar_head_type = scalar_head_type.strip().lower()
        self.scalar_head_hidden_mult = float(scalar_head_hidden_mult)
        self.edge_text_prior_logit_scale = float(edge_text_prior_logit_scale)
        if edge_prior_binary_slot_mask is None:
            edge_prior_binary_slot_mask = torch.zeros((self.edge_slot_dim,), dtype=torch.float32)
        self.register_buffer(
            "edge_prior_binary_slot_mask",
            edge_prior_binary_slot_mask.to(dtype=torch.float32).reshape(1, 1, 1, self.edge_slot_dim, 1),
            persistent=False,
        )

    def _edge_pair_hidden(self, conditioned_nodes: torch.Tensor) -> torch.Tensor:
        num_nodes = int(conditioned_nodes.shape[1])
        src = conditioned_nodes[:, :, None, :].expand(-1, num_nodes, num_nodes, -1)
        dst = conditioned_nodes[:, None, :, :].expand(-1, num_nodes, num_nodes, -1)
        pair = torch.cat([src, dst, src + dst, src - dst], dim=-1)
        return self.edge_pair_proj(pair)

    def forward(
        self,
        node_states: torch.Tensor,
        pooled_tokens: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        edge_text_prior: torch.Tensor | None = None,
        node_scalar_prior: torch.Tensor | None = None,
        node_scalar_prior_mask: torch.Tensor | None = None,
        edge_scalar_prior: torch.Tensor | None = None,
        edge_scalar_prior_mask: torch.Tensor | None = None,
        count_prior: torch.Tensor | None = None,
        count_prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        task_embed = self.task_embed(task_ids)
        conditioned_nodes = node_states + self.task_node_proj(task_embed).unsqueeze(1)
        conditioned_pooled = pooled_tokens + self.task_global_proj(task_embed)

        outputs: dict[str, torch.Tensor] = {
            "present_logits": self.present_head(conditioned_nodes).squeeze(-1),
            "count_logits": self.count_head(conditioned_pooled).squeeze(-1),
            "count_class_logits": self.count_class_head(conditioned_pooled),
            "node_scalar_values": conditioned_nodes.new_zeros((conditioned_nodes.size(0), conditioned_nodes.size(1), self.node_slot_dim)),
            "node_discrete_logits": conditioned_nodes.new_zeros(
                (conditioned_nodes.size(0), conditioned_nodes.size(1), self.node_slot_dim, self.max_input_discrete_value)
            ),
            "edge_scalar_values": conditioned_nodes.new_zeros(
                (conditioned_nodes.size(0), conditioned_nodes.size(1), conditioned_nodes.size(1), self.edge_slot_dim)
            ),
            "edge_discrete_logits": conditioned_nodes.new_zeros(
                (conditioned_nodes.size(0), conditioned_nodes.size(1), conditioned_nodes.size(1), self.edge_slot_dim, self.max_input_discrete_value)
            ),
            "graph_scalar_values": conditioned_pooled.new_zeros((conditioned_pooled.size(0), self.graph_slot_dim)),
            "graph_discrete_logits": conditioned_pooled.new_zeros((conditioned_pooled.size(0), self.graph_slot_dim, self.max_input_discrete_value)),
        }
        if count_prior is not None and count_prior_mask is not None:
            prior = count_prior.to(device=conditioned_pooled.device, dtype=conditioned_pooled.dtype).reshape(-1)
            prior_mask = count_prior_mask.to(device=conditioned_pooled.device, dtype=conditioned_pooled.dtype).reshape(-1)
            if prior.shape[0] == conditioned_pooled.shape[0] and prior_mask.shape[0] == conditioned_pooled.shape[0]:
                outputs["count_prior"] = prior.clamp(0.0, 1.0)
                outputs["count_prior_mask"] = prior_mask.clamp(0.0, 1.0)
        if self.node_slot_dim > 0 and self.node_scalar_head is not None and self.node_discrete_head is not None:
            outputs["node_scalar_values"] = self.node_scalar_head(conditioned_nodes)
            if node_scalar_prior is not None and node_scalar_prior_mask is not None:
                prior = node_scalar_prior.to(device=conditioned_nodes.device, dtype=conditioned_nodes.dtype)
                prior_mask = node_scalar_prior_mask.to(device=conditioned_nodes.device, dtype=conditioned_nodes.dtype)
                if prior.shape == outputs["node_scalar_values"].shape and prior_mask.shape == outputs["node_scalar_values"].shape:
                    outputs["node_scalar_values"] = torch.where(prior_mask > 0.5, prior, outputs["node_scalar_values"])
            node_discrete = self.node_discrete_head(conditioned_nodes)
            outputs["node_discrete_logits"] = node_discrete.view(
                conditioned_nodes.size(0),
                conditioned_nodes.size(1),
                self.node_slot_dim,
                self.max_input_discrete_value,
            )
        if self.edge_slot_dim > 0 and self.edge_pair_proj is not None and self.edge_scalar_head is not None and self.edge_discrete_head is not None:
            edge_hidden = self._edge_pair_hidden(conditioned_nodes)
            if self.edge_text_prior_proj is not None and edge_text_prior is not None:
                prior = edge_text_prior.to(device=edge_hidden.device, dtype=edge_hidden.dtype)
                if prior.shape[:3] == edge_hidden.shape[:3]:
                    edge_hidden = edge_hidden + self.edge_text_prior_proj(prior.unsqueeze(-1))
            outputs["edge_scalar_values"] = self.edge_scalar_head(edge_hidden)
            if edge_scalar_prior is not None and edge_scalar_prior_mask is not None:
                prior = edge_scalar_prior.to(device=edge_hidden.device, dtype=edge_hidden.dtype)
                prior_mask = edge_scalar_prior_mask.to(device=edge_hidden.device, dtype=edge_hidden.dtype)
                if prior.shape == outputs["edge_scalar_values"].shape and prior_mask.shape == outputs["edge_scalar_values"].shape:
                    outputs["edge_scalar_values"] = torch.where(prior_mask > 0.5, prior, outputs["edge_scalar_values"])
            edge_discrete = self.edge_discrete_head(edge_hidden)
            outputs["edge_discrete_logits"] = edge_discrete.view(
                conditioned_nodes.size(0),
                conditioned_nodes.size(1),
                conditioned_nodes.size(1),
                self.edge_slot_dim,
                self.max_input_discrete_value,
            )
            if (
                self.edge_text_prior_logit_scale != 0.0
                and edge_text_prior is not None
                and self.edge_slot_dim > 0
                and self.max_input_discrete_value >= 2
            ):
                prior = edge_text_prior.to(device=edge_hidden.device, dtype=edge_hidden.dtype).clamp(0.0, 1.0)
                if prior.shape[:3] == edge_hidden.shape[:3]:
                    bias = float(self.edge_text_prior_logit_scale) * (2.0 * prior - 1.0)
                    bias = bias[:, :, :, None, None] * self.edge_prior_binary_slot_mask.to(dtype=edge_hidden.dtype)
                    outputs["edge_discrete_logits"][..., 0:1] = outputs["edge_discrete_logits"][..., 0:1] - 0.5 * bias
                    outputs["edge_discrete_logits"][..., 1:2] = outputs["edge_discrete_logits"][..., 1:2] + 0.5 * bias
        if self.graph_slot_dim > 0 and self.graph_scalar_head is not None and self.graph_discrete_head is not None:
            outputs["graph_scalar_values"] = self.graph_scalar_head(conditioned_pooled)
            graph_discrete = self.graph_discrete_head(conditioned_pooled)
            outputs["graph_discrete_logits"] = graph_discrete.view(
                conditioned_pooled.size(0),
                self.graph_slot_dim,
                self.max_input_discrete_value,
            )
        return outputs


def _masked_node_anchor_pool(
    token_states: torch.Tensor,
    node_anchor_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, _, dim = token_states.shape
    if node_anchor_mask is None:
        return token_states.new_zeros((batch_size, 0, dim))
    anchor_mask = node_anchor_mask.to(device=token_states.device, dtype=token_states.dtype)
    if anchor_mask.size(0) != batch_size:
        return token_states.new_zeros((batch_size, 0, dim))
    if attention_mask is not None:
        anchor_mask = anchor_mask * attention_mask.to(device=token_states.device, dtype=token_states.dtype).unsqueeze(1)
    denom = anchor_mask.sum(dim=-1, keepdim=True)
    pooled = torch.bmm(anchor_mask, token_states) / denom.clamp_min(1.0)
    return pooled * (denom > 0.0).to(token_states.dtype)


class QFormerBenchmarkNativeBridge(nn.Module):
    def __init__(self, config: DirectBridgeConfig, *, schema: NativeTensorBridgeSchema) -> None:
        super().__init__()
        self.config = config
        self.bridge_heads = int(config.bridge_heads)
        self.input_proj = nn.Linear(config.llm_hidden_size, config.bridge_dim)
        self.node_queries = nn.Parameter(torch.randn(config.max_nodes, config.bridge_dim) * 0.02)
        pos = _build_generic_positional_embeddings(config.max_nodes, config.bridge_dim)
        self.pos_embed = nn.Parameter(pos)
        self.query_pos_gate = nn.Parameter(torch.tensor(2.0))
        self.query_pos_dropout = nn.Dropout(float(config.query_pos_dropout))
        self.anchor_proj = nn.Linear(config.bridge_dim, config.bridge_dim, bias=False)
        self.anchor_norm = nn.LayerNorm(config.bridge_dim)
        self.anchor_gate = nn.Parameter(torch.tensor(2.0))
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(config.bridge_dim, config.bridge_heads, config.bridge_dropout) for _ in range(config.bridge_layers)]
        )
        self.use_edge_self_bias = bool(getattr(config, "qformer_edge_self_bias", False))
        self.use_rel_pos_self_bias = bool(getattr(config, "qformer_rel_pos_self_bias", False))
        if self.use_edge_self_bias:
            self.edge_self_bias_gate = nn.Parameter(torch.tensor(1.0))
        if self.use_rel_pos_self_bias:
            self.rel_pos_self_bias_gate = nn.Parameter(torch.tensor(1.0))
            self.rel_pos_self_bias = nn.Parameter(torch.zeros(self.bridge_heads, 2 * config.max_nodes - 1))
            positions = torch.arange(config.max_nodes)
            rel_pos_index = positions[:, None] - positions[None, :] + config.max_nodes - 1
            self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)
        edge_prior_binary_slot_mask = torch.tensor(
            [1.0 if _slot_is_binary(slot) and str(slot.value_type) == "mask" else 0.0 for slot in schema.edge_slots],
            dtype=torch.float32,
        )
        self.heads = NativeTensorBridgeOutputHeads(
            config.bridge_dim,
            node_slot_dim=schema.node_slot_dim,
            edge_slot_dim=schema.edge_slot_dim,
            graph_slot_dim=schema.graph_slot_dim,
            num_tasks=max(1, len(schema.task_names)),
            max_input_discrete_value=int(schema.max_input_discrete_value),
            scalar_head_type=config.scalar_head_type,
            scalar_head_hidden_mult=config.scalar_head_hidden_mult,
            edge_prior_binary_slot_mask=edge_prior_binary_slot_mask,
            edge_text_prior_logit_scale=float(getattr(config, "edge_text_prior_logit_scale", 0.0)),
        )

    def _query_self_attn_mask(
        self,
        *,
        edge_text_prior: torch.Tensor | None,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        attn_mask: torch.Tensor | None = None
        if self.use_edge_self_bias and edge_text_prior is not None:
            prior = edge_text_prior[:, :num_nodes, :num_nodes].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if prior.shape == (batch_size, num_nodes, num_nodes):
                eye = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0)
                prior = prior.masked_fill(eye, 0.0)
                edge_bias = torch.tanh(self.edge_self_bias_gate).to(dtype=dtype) * prior
                attn_mask = edge_bias[:, None, :, :].expand(-1, self.bridge_heads, -1, -1).reshape(
                    batch_size * self.bridge_heads,
                    num_nodes,
                    num_nodes,
                )
        if self.use_rel_pos_self_bias:
            rel_index = self.rel_pos_index[:num_nodes, :num_nodes].to(device=device)
            rel_bias = self.rel_pos_self_bias[:, rel_index].to(device=device, dtype=dtype)
            rel_bias = torch.tanh(self.rel_pos_self_bias_gate).to(dtype=dtype) * rel_bias
            rel_bias = rel_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
                batch_size * self.bridge_heads,
                num_nodes,
                num_nodes,
            )
            attn_mask = rel_bias if attn_mask is None else attn_mask + rel_bias
        return attn_mask

    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        task_ids: torch.Tensor,
        *,
        node_anchor_mask: torch.Tensor | None = None,
        edge_text_prior: torch.Tensor | None = None,
        node_scalar_prior: torch.Tensor | None = None,
        node_scalar_prior_mask: torch.Tensor | None = None,
        edge_scalar_prior: torch.Tensor | None = None,
        edge_scalar_prior_mask: torch.Tensor | None = None,
        count_prior: torch.Tensor | None = None,
        count_prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_states = self.input_proj(llm_hidden_states)
        pooled_tokens = _masked_mean_pool(token_states, attention_mask)
        batch_size = llm_hidden_states.size(0)
        gated_pos = torch.sigmoid(self.query_pos_gate) * self.query_pos_dropout(self.pos_embed)
        node_states = (self.node_queries + gated_pos).unsqueeze(0).expand(batch_size, -1, -1)
        anchor_states = _masked_node_anchor_pool(token_states, node_anchor_mask, attention_mask)
        if anchor_states.size(1) == node_states.size(1):
            node_states = node_states + torch.sigmoid(self.anchor_gate) * self.anchor_norm(self.anchor_proj(anchor_states))
        query_self_attn_mask = self._query_self_attn_mask(
            edge_text_prior=edge_text_prior,
            batch_size=batch_size,
            num_nodes=node_states.size(1),
            device=node_states.device,
            dtype=node_states.dtype,
        )
        for layer in self.layers:
            node_states = layer(node_states, token_states, attention_mask, self_attn_mask=query_self_attn_mask)
        return self.heads(
            node_states,
            pooled_tokens,
            task_ids,
            edge_text_prior=edge_text_prior,
            node_scalar_prior=node_scalar_prior,
            node_scalar_prior_mask=node_scalar_prior_mask,
            edge_scalar_prior=edge_scalar_prior,
            edge_scalar_prior_mask=edge_scalar_prior_mask,
            count_prior=count_prior,
            count_prior_mask=count_prior_mask,
        )


class TransNARBenchmarkNativeBridge(nn.Module):
    def __init__(self, config: DirectBridgeConfig, *, schema: NativeTensorBridgeSchema) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.llm_hidden_size, config.bridge_dim)
        self.node_queries = nn.Parameter(torch.randn(config.max_nodes, config.bridge_dim) * 0.02)
        pos = _build_generic_positional_embeddings(config.max_nodes, config.bridge_dim)
        self.pos_embed = nn.Parameter(pos)
        self.query_pos_gate = nn.Parameter(torch.tensor(2.0))
        self.query_pos_dropout = nn.Dropout(float(config.query_pos_dropout))
        self.anchor_proj = nn.Linear(config.bridge_dim, config.bridge_dim, bias=False)
        self.anchor_norm = nn.LayerNorm(config.bridge_dim)
        self.anchor_gate = nn.Parameter(torch.tensor(2.0))
        self.layers = nn.ModuleList(
            [TransNARInterleaveLayer(config.bridge_dim, config.bridge_heads, config.bridge_dropout) for _ in range(config.bridge_layers)]
        )
        edge_prior_binary_slot_mask = torch.tensor(
            [1.0 if _slot_is_binary(slot) and str(slot.value_type) == "mask" else 0.0 for slot in schema.edge_slots],
            dtype=torch.float32,
        )
        self.heads = NativeTensorBridgeOutputHeads(
            config.bridge_dim,
            node_slot_dim=schema.node_slot_dim,
            edge_slot_dim=schema.edge_slot_dim,
            graph_slot_dim=schema.graph_slot_dim,
            num_tasks=max(1, len(schema.task_names)),
            max_input_discrete_value=int(schema.max_input_discrete_value),
            scalar_head_type=config.scalar_head_type,
            scalar_head_hidden_mult=config.scalar_head_hidden_mult,
            edge_prior_binary_slot_mask=edge_prior_binary_slot_mask,
            edge_text_prior_logit_scale=float(getattr(config, "edge_text_prior_logit_scale", 0.0)),
        )

    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        task_ids: torch.Tensor,
        *,
        node_anchor_mask: torch.Tensor | None = None,
        edge_text_prior: torch.Tensor | None = None,
        node_scalar_prior: torch.Tensor | None = None,
        node_scalar_prior_mask: torch.Tensor | None = None,
        edge_scalar_prior: torch.Tensor | None = None,
        edge_scalar_prior_mask: torch.Tensor | None = None,
        count_prior: torch.Tensor | None = None,
        count_prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_states = self.input_proj(llm_hidden_states)
        batch_size = llm_hidden_states.size(0)
        gated_pos = torch.sigmoid(self.query_pos_gate) * self.query_pos_dropout(self.pos_embed)
        node_states = (self.node_queries + gated_pos).unsqueeze(0).expand(batch_size, -1, -1)
        anchor_states = _masked_node_anchor_pool(token_states, node_anchor_mask, attention_mask)
        if anchor_states.size(1) == node_states.size(1):
            node_states = node_states + torch.sigmoid(self.anchor_gate) * self.anchor_norm(self.anchor_proj(anchor_states))
        recurrence_steps = max(1, int(self.config.transnar_recurrence_steps))
        for _ in range(recurrence_steps):
            for layer in self.layers:
                node_states, token_states = layer(node_states=node_states, token_states=token_states, token_mask=attention_mask)
        pooled_tokens = _masked_mean_pool(token_states, attention_mask)
        return self.heads(
            node_states,
            pooled_tokens,
            task_ids,
            edge_text_prior=edge_text_prior,
            node_scalar_prior=node_scalar_prior,
            node_scalar_prior_mask=node_scalar_prior_mask,
            edge_scalar_prior=edge_scalar_prior,
            edge_scalar_prior_mask=edge_scalar_prior_mask,
            count_prior=count_prior,
            count_prior_mask=count_prior_mask,
        )


def build_benchmark_native_bridge_model(arch: str, config: DirectBridgeConfig, *, schema: NativeTensorBridgeSchema) -> nn.Module:
    arch = arch.strip().lower()
    if arch == "qformer":
        return QFormerBenchmarkNativeBridge(config, schema=schema)
    if arch == "transnar":
        return TransNARBenchmarkNativeBridge(config, schema=schema)
    raise ValueError(f"Unsupported bridge arch: {arch}")


def _move_native_batch(batch: NativeTensorBridgeBatch, device: torch.device) -> NativeTensorBridgeBatch:
    return NativeTensorBridgeBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        task_ids=batch.task_ids.to(device),
        present_target=batch.present_target.to(device),
        node_feature_values=batch.node_feature_values.to(device),
        node_feature_bins=batch.node_feature_bins.to(device),
        node_slot_mask=batch.node_slot_mask.to(device),
        edge_feature_values=batch.edge_feature_values.to(device),
        edge_feature_bins=batch.edge_feature_bins.to(device),
        edge_slot_mask=batch.edge_slot_mask.to(device),
        graph_feature_values=batch.graph_feature_values.to(device),
        graph_feature_bins=batch.graph_feature_bins.to(device),
        graph_slot_mask=batch.graph_slot_mask.to(device),
        num_nodes=batch.num_nodes.to(device),
        sample_ids=batch.sample_ids,
        tasks=batch.tasks,
        adapter_versions=batch.adapter_versions,
        node_scalar_prior=None if batch.node_scalar_prior is None else batch.node_scalar_prior.to(device),
        node_scalar_prior_mask=None if batch.node_scalar_prior_mask is None else batch.node_scalar_prior_mask.to(device),
        edge_scalar_prior=None if batch.edge_scalar_prior is None else batch.edge_scalar_prior.to(device),
        edge_scalar_prior_mask=None if batch.edge_scalar_prior_mask is None else batch.edge_scalar_prior_mask.to(device),
        count_prior=None if batch.count_prior is None else batch.count_prior.to(device),
        count_prior_mask=None if batch.count_prior_mask is None else batch.count_prior_mask.to(device),
        node_anchor_mask=None if batch.node_anchor_mask is None else batch.node_anchor_mask.to(device),
        edge_text_prior=None if batch.edge_text_prior is None else batch.edge_text_prior.to(device),
        llm_hidden_states=None if batch.llm_hidden_states is None else batch.llm_hidden_states.to(device),
    )


def _schema_family_masks(schema: NativeTensorBridgeSchema, device: torch.device) -> dict[str, torch.Tensor]:
    node_slots = schema.node_slots
    edge_slots = schema.edge_slots
    graph_slots = schema.graph_slots
    return {
        "node_scalar": torch.tensor([1.0 if _slot_is_scalar(slot) else 0.0 for slot in node_slots], dtype=torch.float32, device=device).view(1, 1, -1),
        "node_binary": torch.tensor([1.0 if _slot_is_binary(slot) else 0.0 for slot in node_slots], dtype=torch.float32, device=device).view(1, 1, -1),
        "node_mask_one": torch.tensor([1.0 if _slot_is_mask_one(slot) else 0.0 for slot in node_slots], dtype=torch.float32, device=device).view(1, 1, -1),
        "node_class": torch.tensor([1.0 if _slot_is_class(slot) else 0.0 for slot in node_slots], dtype=torch.float32, device=device).view(1, 1, -1),
        "edge_scalar": torch.tensor([1.0 if _slot_is_scalar(slot) else 0.0 for slot in edge_slots], dtype=torch.float32, device=device).view(1, 1, 1, -1),
        "edge_binary": torch.tensor([1.0 if _slot_is_binary(slot) else 0.0 for slot in edge_slots], dtype=torch.float32, device=device).view(1, 1, 1, -1),
        "edge_mask_one": torch.tensor([1.0 if _slot_is_mask_one(slot) else 0.0 for slot in edge_slots], dtype=torch.float32, device=device).view(1, 1, 1, -1),
        "edge_class": torch.tensor([1.0 if _slot_is_class(slot) else 0.0 for slot in edge_slots], dtype=torch.float32, device=device).view(1, 1, 1, -1),
        "graph_scalar": torch.tensor([1.0 if _slot_is_scalar(slot) else 0.0 for slot in graph_slots], dtype=torch.float32, device=device).view(1, -1),
        "graph_binary": torch.tensor([1.0 if _slot_is_binary(slot) else 0.0 for slot in graph_slots], dtype=torch.float32, device=device).view(1, -1),
        "graph_mask_one": torch.tensor([1.0 if _slot_is_mask_one(slot) else 0.0 for slot in graph_slots], dtype=torch.float32, device=device).view(1, -1),
        "graph_class": torch.tensor([1.0 if _slot_is_class(slot) else 0.0 for slot in graph_slots], dtype=torch.float32, device=device).view(1, -1),
    }


def _masked_smooth_l1(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not bool(valid.any()):
        return preds.sum() * 0.0
    return F.smooth_l1_loss(preds[valid], targets[valid])


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


def _binary_logits_from_discrete_logits(logits: torch.Tensor) -> torch.Tensor:
    if int(logits.size(-1)) < 2:
        return logits[..., 0]
    return logits[..., 1] - logits[..., 0]


def _masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight_cap: float = 10.0,
    neg_penalty_weight: float = 0.0,
) -> torch.Tensor:
    valid = mask > 0.5
    if not bool(valid.any()):
        return logits.sum() * 0.0
    binary_logits = _binary_logits_from_discrete_logits(logits)
    valid_targets = targets[valid].float().clamp(0.0, 1.0)
    positives = valid_targets.sum()
    negatives = valid_targets.numel() - positives
    if bool((positives > 0).item()) and bool((negatives > 0).item()):
        pos_weight = (negatives / positives).clamp(min=1.0, max=max(1.0, float(pos_weight_cap)))
    else:
        pos_weight = torch.ones((), dtype=binary_logits.dtype, device=binary_logits.device)
    valid_logits = binary_logits[valid]
    bce = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=pos_weight)
    probs = torch.sigmoid(valid_logits)
    intersection = (probs * valid_targets).sum()
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probs.sum() + valid_targets.sum() + 1.0))
    if float(neg_penalty_weight) > 0.0:
        negative = valid_targets < 0.5
        if bool(negative.any()):
            dice = dice + float(neg_penalty_weight) * probs[negative].mean()
    return bce + dice


def _masked_node_mask_one_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    slot_mask = (mask > 0.5).any(dim=1)
    if not bool(slot_mask.any()):
        return logits.sum() * 0.0
    binary_logits = _binary_logits_from_discrete_logits(logits)
    losses: list[torch.Tensor] = []
    for slot_idx in torch.nonzero(slot_mask.any(dim=0), as_tuple=False).flatten().tolist():
        valid_samples = slot_mask[:, slot_idx]
        if not bool(valid_samples.any()):
            continue
        sample_logits = binary_logits[valid_samples, :, slot_idx]
        sample_valid = mask[valid_samples, :, slot_idx] > 0.5
        sample_targets = targets[valid_samples, :, slot_idx].float()
        target_has_positive = (sample_targets * sample_valid.float()).sum(dim=1) > 0.5
        if not bool(target_has_positive.any()):
            continue
        sample_logits = sample_logits[target_has_positive]
        sample_valid = sample_valid[target_has_positive]
        sample_targets = sample_targets[target_has_positive]
        target_idx = torch.argmax(sample_targets.masked_fill(~sample_valid, -1.0), dim=1)
        losses.append(F.cross_entropy(sample_logits.masked_fill(~sample_valid, -1e4), target_idx))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _mixed_discrete_predictions(logits: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
    pred_bins = _discrete_predictions(logits)
    if int(logits.size(-1)) < 2:
        return pred_bins
    binary_preds = (_binary_logits_from_discrete_logits(logits) >= 0.0).long()
    return torch.where((binary_mask > 0.5).expand_as(pred_bins), binary_preds, pred_bins)


def _apply_node_mask_one_predictions(pred_bins: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, mask_one_slots: torch.Tensor) -> torch.Tensor:
    slot_indices = torch.nonzero(mask_one_slots.reshape(-1) > 0.5, as_tuple=False).flatten().tolist()
    if not slot_indices:
        return pred_bins
    pred_bins = pred_bins.clone()
    binary_logits = _binary_logits_from_discrete_logits(logits)
    for slot_idx in slot_indices:
        valid = mask[:, :, slot_idx] > 0.5
        if not bool(valid.any()):
            continue
        slot_logits = binary_logits[:, :, slot_idx].masked_fill(~valid, -1e4)
        chosen = torch.argmax(slot_logits, dim=1)
        one_hot = torch.zeros_like(pred_bins[:, :, slot_idx])
        one_hot.scatter_(1, chosen.view(-1, 1), 1)
        pred_bins[:, :, slot_idx] = one_hot * valid.long()
    return pred_bins


def _normalized_count_predictions(
    outputs: dict[str, torch.Tensor],
    *,
    max_nodes: int,
    count_blend_alpha: float = COUNT_BLEND_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    implicit_norm = torch.sigmoid(outputs["present_logits"]).sum(dim=1) / float(max(1, max_nodes))
    count_logits = outputs.get("count_logits")
    if count_logits is None:
        explicit_norm = implicit_norm
    else:
        explicit_norm = torch.sigmoid(count_logits).reshape(-1)
    alpha = float(max(0.0, min(1.0, count_blend_alpha)))
    blended_norm = (alpha * explicit_norm) + ((1.0 - alpha) * implicit_norm)
    return implicit_norm.clamp(0.0, 1.0), explicit_norm.clamp(0.0, 1.0), blended_norm.clamp(0.0, 1.0)


def _count_class_logits_for_nodes(outputs: dict[str, torch.Tensor], *, max_nodes: int) -> torch.Tensor | None:
    logits = outputs.get("count_class_logits")
    if logits is None:
        return None
    count = min(int(max_nodes), int(logits.size(-1)) - 1)
    if count < 1:
        return None
    return logits[:, : count + 1]


def _count_class_predictions(
    outputs: dict[str, torch.Tensor],
    *,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    logits = _count_class_logits_for_nodes(outputs, max_nodes=max_nodes)
    if logits is None:
        return None
    pred_count = torch.argmax(logits[:, 1:], dim=-1) + 1
    pred_norm = pred_count.to(dtype=torch.float32) / float(max(1, max_nodes))
    return pred_count, pred_norm


def _count_prior_predictions(
    outputs: dict[str, torch.Tensor],
    *,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    prior = outputs.get("count_prior")
    prior_mask = outputs.get("count_prior_mask")
    if prior is None or prior_mask is None:
        return None
    prior = prior.reshape(-1).clamp(0.0, 1.0)
    prior_mask = prior_mask.reshape(-1)
    if not bool((prior_mask > 0.5).any()):
        return None
    pred_count = torch.round(prior * float(max(1, max_nodes))).long().clamp(min=1, max=max(1, max_nodes))
    return pred_count, prior


def _select_decoded_count_norm(
    implicit_norm: torch.Tensor,
    explicit_norm: torch.Tensor,
    blended_norm: torch.Tensor,
    *,
    count_decode_mode: str,
    prior_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = count_decode_mode.strip().lower()
    if mode == "blended":
        return blended_norm
    if mode == "head_only":
        return explicit_norm
    if mode == "implicit":
        return implicit_norm
    if mode == "prior":
        if prior_norm is None:
            return implicit_norm
        return prior_norm
    raise ValueError(f"Unsupported count decode mode: {count_decode_mode}")


def compute_benchmark_native_bridge_losses(
    outputs: dict[str, torch.Tensor],
    batch: NativeTensorBridgeBatch,
    *,
    schema: NativeTensorBridgeSchema,
    count_blend_alpha: float = COUNT_BLEND_ALPHA,
    count_loss_weight_implicit: float = 0.25,
    count_loss_weight_explicit: float = 0.25,
    count_loss_weight_blended: float = 0.50,
    loss_weight_present: float,
    loss_weight_count: float,
    loss_weight_node_scalar: float,
    loss_weight_node_discrete: float,
    loss_weight_edge_scalar: float,
    loss_weight_edge_discrete: float,
    loss_weight_graph_scalar: float,
    loss_weight_graph_discrete: float,
    loss_weight_present_monotonic: float,
    edge_binary_pos_weight_cap: float = 10.0,
    edge_binary_neg_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["present_logits"].device
    masks = _schema_family_masks(schema, device)
    present_loss = F.binary_cross_entropy_with_logits(outputs["present_logits"], batch.present_target)
    if outputs["present_logits"].size(1) > 1:
        present_monotonic_loss = F.relu(outputs["present_logits"][:, 1:] - outputs["present_logits"][:, :-1]).mean()
    else:
        present_monotonic_loss = outputs["present_logits"].sum() * 0.0
    max_nodes = float(max(1, int(batch.present_target.size(1))))
    true_node_count = batch.num_nodes.float()
    true_count_norm = true_node_count / max_nodes
    implicit_count_norm, explicit_count_norm, blended_count_norm = _normalized_count_predictions(
        outputs,
        max_nodes=int(max_nodes),
        count_blend_alpha=count_blend_alpha,
    )
    count_class_logits = _count_class_logits_for_nodes(outputs, max_nodes=int(max_nodes))
    implicit_count_loss = F.smooth_l1_loss(implicit_count_norm, true_count_norm)
    explicit_count_loss = F.smooth_l1_loss(explicit_count_norm, true_count_norm)
    blended_count_loss = F.smooth_l1_loss(blended_count_norm, true_count_norm)
    if count_class_logits is None:
        count_class_loss = explicit_count_loss.sum() * 0.0
    else:
        count_class_loss = F.cross_entropy(
            count_class_logits,
            batch.num_nodes.long().clamp(min=0, max=count_class_logits.size(-1) - 1),
        )
    implicit_w = max(0.0, float(count_loss_weight_implicit))
    explicit_w = max(0.0, float(count_loss_weight_explicit))
    blended_w = max(0.0, float(count_loss_weight_blended))
    count_weight_total = implicit_w + explicit_w + blended_w
    if count_weight_total <= 0.0:
        count_loss = implicit_count_loss.sum() * 0.0
    else:
        count_loss = (
            (implicit_w * implicit_count_loss)
            + (explicit_w * count_class_loss)
            + (blended_w * blended_count_loss)
        ) / count_weight_total

    node_scalar_loss = _masked_smooth_l1(
        outputs["node_scalar_values"],
        batch.node_feature_values,
        batch.node_slot_mask * masks["node_scalar"],
    )
    node_mask_one_loss = _masked_node_mask_one_cross_entropy(
        outputs["node_discrete_logits"],
        batch.node_feature_bins,
        batch.node_slot_mask * masks["node_mask_one"],
    )
    node_binary_loss = _masked_binary_cross_entropy(
        outputs["node_discrete_logits"],
        batch.node_feature_bins,
        batch.node_slot_mask * masks["node_binary"] * (1.0 - masks["node_mask_one"]),
    )
    node_class_loss = _masked_cross_entropy(
        outputs["node_discrete_logits"],
        batch.node_feature_bins,
        batch.node_slot_mask * masks["node_class"],
    )
    node_discrete_loss = node_mask_one_loss + node_binary_loss + node_class_loss
    edge_scalar_loss = _masked_smooth_l1(
        outputs["edge_scalar_values"],
        batch.edge_feature_values,
        batch.edge_slot_mask * masks["edge_scalar"],
    )
    edge_binary_loss = _masked_binary_cross_entropy(
        outputs["edge_discrete_logits"],
        batch.edge_feature_bins,
        batch.edge_slot_mask * masks["edge_binary"] * (1.0 - masks["edge_mask_one"]),
        pos_weight_cap=edge_binary_pos_weight_cap,
        neg_penalty_weight=edge_binary_neg_penalty,
    )
    edge_class_loss = _masked_cross_entropy(
        outputs["edge_discrete_logits"],
        batch.edge_feature_bins,
        batch.edge_slot_mask * masks["edge_class"],
    )
    edge_discrete_loss = edge_binary_loss + edge_class_loss
    graph_scalar_loss = _masked_smooth_l1(
        outputs["graph_scalar_values"],
        batch.graph_feature_values,
        batch.graph_slot_mask * masks["graph_scalar"],
    )
    graph_binary_loss = _masked_binary_cross_entropy(
        outputs["graph_discrete_logits"],
        batch.graph_feature_bins,
        batch.graph_slot_mask * masks["graph_binary"] * (1.0 - masks["graph_mask_one"]),
    )
    graph_class_loss = _masked_cross_entropy(
        outputs["graph_discrete_logits"],
        batch.graph_feature_bins,
        batch.graph_slot_mask * masks["graph_class"],
    )
    graph_discrete_loss = graph_binary_loss + graph_class_loss

    total_loss = (
        (float(loss_weight_present) * present_loss)
        + (float(loss_weight_count) * count_loss)
        + (float(loss_weight_node_scalar) * node_scalar_loss)
        + (float(loss_weight_node_discrete) * node_discrete_loss)
        + (float(loss_weight_edge_scalar) * edge_scalar_loss)
        + (float(loss_weight_edge_discrete) * edge_discrete_loss)
        + (float(loss_weight_graph_scalar) * graph_scalar_loss)
        + (float(loss_weight_graph_discrete) * graph_discrete_loss)
        + (float(loss_weight_present_monotonic) * present_monotonic_loss)
    )
    return total_loss, {
        "loss/present": float(present_loss.item()),
        "loss/present_monotonic": float(present_monotonic_loss.item()),
        "loss/count": float(count_loss.item()),
        "loss/count_implicit": float(implicit_count_loss.item()),
        "loss/count_head": float(explicit_count_loss.item()),
        "loss/count_class": float(count_class_loss.item()),
        "loss/count_blended": float(blended_count_loss.item()),
        "loss/node_scalar": float(node_scalar_loss.item()),
        "loss/node_discrete": float(node_discrete_loss.item()),
        "loss/node_mask_one": float(node_mask_one_loss.item()),
        "loss/node_binary": float(node_binary_loss.item()),
        "loss/node_class": float(node_class_loss.item()),
        "loss/edge_scalar": float(edge_scalar_loss.item()),
        "loss/edge_discrete": float(edge_discrete_loss.item()),
        "loss/edge_binary": float(edge_binary_loss.item()),
        "loss/edge_class": float(edge_class_loss.item()),
        "loss/graph_scalar": float(graph_scalar_loss.item()),
        "loss/graph_discrete": float(graph_discrete_loss.item()),
        "loss/graph_binary": float(graph_binary_loss.item()),
        "loss/graph_class": float(graph_class_loss.item()),
    }


def _discrete_predictions(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def _mask_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    pred_bool = preds.to(torch.bool)
    target_bool = targets.to(torch.bool)
    tp = int((pred_bool & target_bool).sum().item())
    fp = int((pred_bool & (~target_bool)).sum().item())
    fn = int(((~pred_bool) & target_bool).sum().item())
    if tp + fp > 0:
        precision = tp / float(tp + fp)
    else:
        precision = 1.0
    if tp + fn > 0:
        recall = tp / float(tp + fn)
    else:
        recall = 1.0
    if precision + recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def compute_benchmark_native_bridge_metrics(
    outputs: dict[str, torch.Tensor],
    batch: NativeTensorBridgeBatch,
    *,
    schema: NativeTensorBridgeSchema,
    count_blend_alpha: float = COUNT_BLEND_ALPHA,
    count_decode_mode: str = "blended",
    present_threshold: float,
    scalar_exact_tolerance: float,
) -> dict[str, float]:
    device = outputs["present_logits"].device
    masks = _schema_family_masks(schema, device)
    present_pred = (torch.sigmoid(outputs["present_logits"]) >= float(present_threshold)).float()
    pred_node_count_threshold = present_pred.sum(dim=1).to(dtype=batch.num_nodes.dtype)
    max_nodes = int(batch.present_target.size(1))
    implicit_count_norm, explicit_count_norm, blended_count_norm = _normalized_count_predictions(
        outputs,
        max_nodes=max_nodes,
        count_blend_alpha=count_blend_alpha,
    )
    count_prior_pred = _count_prior_predictions(outputs, max_nodes=max_nodes)
    prior_count_norm = count_prior_pred[1] if count_prior_pred is not None else None
    decoded_count_norm = _select_decoded_count_norm(
        implicit_count_norm,
        explicit_count_norm,
        blended_count_norm,
        count_decode_mode=count_decode_mode,
        prior_norm=prior_count_norm,
    )
    class_count_pred = _count_class_predictions(outputs, max_nodes=max_nodes)
    pred_node_count_implicit = torch.round(implicit_count_norm * float(max_nodes)).to(dtype=batch.num_nodes.dtype)
    pred_node_count_head = torch.round(explicit_count_norm * float(max_nodes)).to(dtype=batch.num_nodes.dtype)
    pred_node_count = torch.round(decoded_count_norm * float(max_nodes)).to(dtype=batch.num_nodes.dtype)
    if class_count_pred is not None:
        pred_node_count_head = class_count_pred[0].to(dtype=batch.num_nodes.dtype)
        if count_decode_mode.strip().lower() == "head_only":
            pred_node_count = pred_node_count_head
    if count_prior_pred is not None and count_decode_mode.strip().lower() == "prior":
        pred_node_count = count_prior_pred[0].to(dtype=batch.num_nodes.dtype)
    pred_node_count_implicit = pred_node_count_implicit.clamp(min=1, max=max(1, max_nodes))
    pred_node_count_head = pred_node_count_head.clamp(min=1, max=max(1, max_nodes))
    pred_node_count = pred_node_count.clamp(min=1, max=max(1, max_nodes))
    present_count_acc = (pred_node_count == batch.num_nodes).float().mean().item()
    present_count_mae = torch.abs(pred_node_count.float() - batch.num_nodes.float()).mean().item()
    present_count_acc_implicit = (pred_node_count_implicit == batch.num_nodes).float().mean().item()
    present_count_mae_implicit = torch.abs(pred_node_count_implicit.float() - batch.num_nodes.float()).mean().item()
    present_count_acc_head = (pred_node_count_head == batch.num_nodes).float().mean().item()
    present_count_mae_head = torch.abs(pred_node_count_head.float() - batch.num_nodes.float()).mean().item()
    present_count_acc_threshold = (pred_node_count_threshold == batch.num_nodes).float().mean().item()
    present_count_mae_threshold = torch.abs(pred_node_count_threshold.float() - batch.num_nodes.float()).mean().item()

    node_scalar_mask = batch.node_slot_mask * masks["node_scalar"]
    edge_scalar_mask = batch.edge_slot_mask * masks["edge_scalar"]
    graph_scalar_mask = batch.graph_slot_mask * masks["graph_scalar"]
    scalar_abs_errors: list[torch.Tensor] = []
    for preds, targets, mask in (
        (outputs["node_scalar_values"], batch.node_feature_values, node_scalar_mask),
        (outputs["edge_scalar_values"], batch.edge_feature_values, edge_scalar_mask),
        (outputs["graph_scalar_values"], batch.graph_feature_values, graph_scalar_mask),
    ):
        valid = mask > 0.5
        if bool(valid.any()):
            scalar_abs_errors.append(torch.abs(preds[valid] - targets[valid]))
    if scalar_abs_errors:
        scalar_mae = torch.cat(scalar_abs_errors).mean().item()
    else:
        scalar_mae = 0.0

    node_pred_bins = _mixed_discrete_predictions(outputs["node_discrete_logits"], masks["node_binary"])
    node_pred_bins = _apply_node_mask_one_predictions(
        node_pred_bins,
        outputs["node_discrete_logits"],
        batch.node_slot_mask,
        masks["node_mask_one"],
    )
    edge_pred_bins = _mixed_discrete_predictions(outputs["edge_discrete_logits"], masks["edge_binary"])
    graph_pred_bins = _mixed_discrete_predictions(outputs["graph_discrete_logits"], masks["graph_binary"])

    discrete_acc_parts: list[torch.Tensor] = []
    for preds, targets, mask in (
        (node_pred_bins, batch.node_feature_bins, batch.node_slot_mask * (masks["node_binary"] + masks["node_class"])),
        (edge_pred_bins, batch.edge_feature_bins, batch.edge_slot_mask * (masks["edge_binary"] + masks["edge_class"])),
        (graph_pred_bins, batch.graph_feature_bins, batch.graph_slot_mask * (masks["graph_binary"] + masks["graph_class"])),
    ):
        valid = mask > 0.5
        if bool(valid.any()):
            discrete_acc_parts.append((preds[valid] == targets[valid]).float())
    discrete_acc = torch.cat(discrete_acc_parts).mean().item() if discrete_acc_parts else 0.0

    class_acc_parts: list[torch.Tensor] = []
    for preds, targets, mask in (
        (node_pred_bins, batch.node_feature_bins, batch.node_slot_mask * masks["node_class"]),
        (edge_pred_bins, batch.edge_feature_bins, batch.edge_slot_mask * masks["edge_class"]),
        (graph_pred_bins, batch.graph_feature_bins, batch.graph_slot_mask * masks["graph_class"]),
    ):
        valid = mask > 0.5
        if bool(valid.any()):
            class_acc_parts.append((preds[valid] == targets[valid]).float())
    class_acc = torch.cat(class_acc_parts).mean().item() if class_acc_parts else 0.0

    binary_preds_parts: list[torch.Tensor] = []
    binary_targets_parts: list[torch.Tensor] = []
    for preds, targets, mask in (
        (node_pred_bins, batch.node_feature_bins, batch.node_slot_mask * masks["node_binary"]),
        (edge_pred_bins, batch.edge_feature_bins, batch.edge_slot_mask * masks["edge_binary"]),
        (graph_pred_bins, batch.graph_feature_bins, batch.graph_slot_mask * masks["graph_binary"]),
    ):
        valid = mask > 0.5
        if bool(valid.any()):
            binary_preds_parts.append(preds[valid].clamp(0, 1))
            binary_targets_parts.append(targets[valid].clamp(0, 1))
    if binary_preds_parts:
        binary_f1 = _mask_f1(torch.cat(binary_preds_parts), torch.cat(binary_targets_parts))
    else:
        binary_f1 = 0.0

    exact_hits = 0
    batch_size = int(batch.num_nodes.size(0))
    for sample_idx in range(batch_size):
        n_true = int(batch.num_nodes[sample_idx].item())
        n_pred = int(pred_node_count[sample_idx].item())
        if n_true != n_pred:
            continue
        node_disc_mask = (batch.node_slot_mask[sample_idx, :n_true] * (masks["node_binary"][0, 0] + masks["node_class"][0, 0])) > 0.5
        edge_disc_mask = (batch.edge_slot_mask[sample_idx, :n_true, :n_true] * (masks["edge_binary"][0, 0, 0] + masks["edge_class"][0, 0, 0])) > 0.5
        graph_disc_mask = (batch.graph_slot_mask[sample_idx] * (masks["graph_binary"][0] + masks["graph_class"][0])) > 0.5
        node_scalar_valid = (batch.node_slot_mask[sample_idx, :n_true] * masks["node_scalar"][0, 0]) > 0.5
        edge_scalar_valid = (batch.edge_slot_mask[sample_idx, :n_true, :n_true] * masks["edge_scalar"][0, 0, 0]) > 0.5
        graph_scalar_valid = (batch.graph_slot_mask[sample_idx] * masks["graph_scalar"][0]) > 0.5
        node_disc_ok = True if not bool(node_disc_mask.any()) else bool(torch.equal(node_pred_bins[sample_idx, :n_true][node_disc_mask], batch.node_feature_bins[sample_idx, :n_true][node_disc_mask]))
        edge_disc_ok = True if not bool(edge_disc_mask.any()) else bool(torch.equal(edge_pred_bins[sample_idx, :n_true, :n_true][edge_disc_mask], batch.edge_feature_bins[sample_idx, :n_true, :n_true][edge_disc_mask]))
        graph_disc_ok = True if not bool(graph_disc_mask.any()) else bool(torch.equal(graph_pred_bins[sample_idx][graph_disc_mask], batch.graph_feature_bins[sample_idx][graph_disc_mask]))

        scalar_ok = True
        if bool(node_scalar_valid.any()):
            scalar_ok = scalar_ok and bool(torch.all(torch.abs(outputs["node_scalar_values"][sample_idx, :n_true][node_scalar_valid] - batch.node_feature_values[sample_idx, :n_true][node_scalar_valid]) <= float(scalar_exact_tolerance)).item())
        if bool(edge_scalar_valid.any()):
            scalar_ok = scalar_ok and bool(torch.all(torch.abs(outputs["edge_scalar_values"][sample_idx, :n_true, :n_true][edge_scalar_valid] - batch.edge_feature_values[sample_idx, :n_true, :n_true][edge_scalar_valid]) <= float(scalar_exact_tolerance)).item())
        if bool(graph_scalar_valid.any()):
            scalar_ok = scalar_ok and bool(torch.all(torch.abs(outputs["graph_scalar_values"][sample_idx][graph_scalar_valid] - batch.graph_feature_values[sample_idx][graph_scalar_valid]) <= float(scalar_exact_tolerance)).item())
        if node_disc_ok and edge_disc_ok and graph_disc_ok and scalar_ok:
            exact_hits += 1

    native_exact = float(exact_hits) / float(max(1, batch_size))
    scalar_quality = 1.0 / (1.0 + max(0.0, scalar_mae))
    selection_score = (
        0.40 * native_exact
        + 0.25 * class_acc
        + 0.20 * binary_f1
        + 0.10 * present_count_acc
        + 0.05 * scalar_quality
    )
    return {
        "metric/present_count_acc": float(present_count_acc),
        "metric/present_count_mae": float(present_count_mae),
        "metric/present_count_acc_implicit": float(present_count_acc_implicit),
        "metric/present_count_mae_implicit": float(present_count_mae_implicit),
        "metric/present_count_acc_head": float(present_count_acc_head),
        "metric/present_count_mae_head": float(present_count_mae_head),
        "metric/present_count_acc_threshold": float(present_count_acc_threshold),
        "metric/present_count_mae_threshold": float(present_count_mae_threshold),
        "metric/scalar_mae": float(scalar_mae),
        "metric/discrete_acc": float(discrete_acc),
        "metric/class_acc": float(class_acc),
        "metric/binary_f1": float(binary_f1),
        "metric/native_exact": float(native_exact),
        "selection_score": float(selection_score),
    }


def build_benchmark_native_trm_tensors_from_bridge_prediction(
    *,
    task: str,
    schema: NativeTensorBridgeSchema,
    outputs: dict[str, torch.Tensor],
    task_to_idx: dict[str, int],
    task_id: int,
    present_threshold: float,
    device: torch.device,
    count_blend_alpha: float = COUNT_BLEND_ALPHA,
    count_decode_mode: str = "blended",
) -> dict[str, torch.Tensor | int]:
    task_idx = int(task_id)
    del present_threshold  # kept for CLI compatibility while decoding uses rounded expected count
    max_nodes = int(outputs["present_logits"].shape[1])
    implicit_count_norm, explicit_count_norm, blended_count_norm = _normalized_count_predictions(
        outputs,
        max_nodes=max_nodes,
        count_blend_alpha=count_blend_alpha,
    )
    count_prior_pred = _count_prior_predictions(outputs, max_nodes=max_nodes)
    prior_count_norm = count_prior_pred[1] if count_prior_pred is not None else None
    decoded_count_norm = _select_decoded_count_norm(
        implicit_count_norm,
        explicit_count_norm,
        blended_count_norm,
        count_decode_mode=count_decode_mode,
        prior_norm=prior_count_norm,
    )
    class_count_pred = _count_class_predictions(outputs, max_nodes=max_nodes)
    if count_prior_pred is not None and count_decode_mode.strip().lower() == "prior":
        num_nodes = int(count_prior_pred[0][0].item())
    elif class_count_pred is not None and count_decode_mode.strip().lower() == "head_only":
        num_nodes = int(class_count_pred[0][0].item())
    else:
        num_nodes = int(torch.round(decoded_count_norm[0] * float(max_nodes)).item())
    num_nodes = max(1, min(max(1, max_nodes), num_nodes))

    node_slots = schema.node_slots
    edge_slots = schema.edge_slots
    graph_slots = schema.graph_slots
    node_slot_dim = len(node_slots)
    edge_slot_dim = len(edge_slots)
    graph_slot_dim = len(graph_slots)

    node_feature_values = torch.zeros((num_nodes, node_slot_dim), dtype=torch.float32, device=device)
    node_feature_bins = torch.zeros((num_nodes, node_slot_dim), dtype=torch.long, device=device)
    node_slot_mask = torch.tensor(schema.task_node_slot_active[task_idx], dtype=torch.float32, device=device).view(1, -1).expand(num_nodes, -1).clone() if node_slot_dim > 0 else torch.zeros((num_nodes, 0), dtype=torch.float32, device=device)
    node_binary_mask = torch.tensor([1.0 if _slot_is_binary(slot) else 0.0 for slot in node_slots], dtype=torch.float32, device=device).view(1, -1) if node_slot_dim > 0 else torch.zeros((1, 0), dtype=torch.float32, device=device)
    node_pred_bins = _mixed_discrete_predictions(outputs["node_discrete_logits"][0, :num_nodes], node_binary_mask) if node_slot_dim > 0 else torch.zeros((num_nodes, 0), dtype=torch.long, device=device)
    if node_slot_dim > 0:
        node_mask_one_mask = torch.tensor([1.0 if _slot_is_mask_one(slot) else 0.0 for slot in node_slots], dtype=torch.float32, device=device).view(1, -1)
        node_pred_bins = _apply_node_mask_one_predictions(
            node_pred_bins.unsqueeze(0),
            outputs["node_discrete_logits"][0:1, :num_nodes],
            node_slot_mask.unsqueeze(0),
            node_mask_one_mask.view(1, 1, -1),
        ).squeeze(0)
    for slot_idx, slot in enumerate(node_slots):
        if not _slot_is_discrete(slot):
            node_feature_values[:, slot_idx] = outputs["node_scalar_values"][0, :num_nodes, slot_idx]
        else:
            node_feature_bins[:, slot_idx] = node_pred_bins[:, slot_idx].clamp(min=0, max=max(1, schema.max_input_discrete_value) - 1)

    edge_index = torch.zeros((0, 2), dtype=torch.long, device=device)
    edge_feature_values = torch.zeros((0, edge_slot_dim), dtype=torch.float32, device=device)
    edge_feature_bins = torch.zeros((0, edge_slot_dim), dtype=torch.long, device=device)
    edge_slot_mask = torch.zeros((0, edge_slot_dim), dtype=torch.float32, device=device)
    if edge_slot_dim > 0 and any(float(v) > 0.5 for v in schema.task_edge_slot_active[task_idx]):
        edge_index = torch.tensor([[src, dst] for src in range(num_nodes) for dst in range(num_nodes)], dtype=torch.long, device=device)
        edge_feature_values = torch.zeros((num_nodes * num_nodes, edge_slot_dim), dtype=torch.float32, device=device)
        edge_feature_bins = torch.zeros((num_nodes * num_nodes, edge_slot_dim), dtype=torch.long, device=device)
        active = torch.tensor(schema.task_edge_slot_active[task_idx], dtype=torch.float32, device=device).view(1, -1).expand(num_nodes * num_nodes, -1).clone()
        edge_slot_mask = active
        edge_binary_mask = torch.tensor([1.0 if _slot_is_binary(slot) else 0.0 for slot in edge_slots], dtype=torch.float32, device=device).view(1, 1, -1)
        edge_pred_bins = _mixed_discrete_predictions(outputs["edge_discrete_logits"][0, :num_nodes, :num_nodes], edge_binary_mask).reshape(num_nodes * num_nodes, edge_slot_dim)
        edge_pred_scalar = outputs["edge_scalar_values"][0, :num_nodes, :num_nodes].reshape(num_nodes * num_nodes, edge_slot_dim)
        for slot_idx, slot in enumerate(edge_slots):
            if not _slot_is_discrete(slot):
                edge_feature_values[:, slot_idx] = edge_pred_scalar[:, slot_idx]
            else:
                edge_feature_bins[:, slot_idx] = edge_pred_bins[:, slot_idx].clamp(min=0, max=max(1, schema.max_input_discrete_value) - 1)

    graph_feature_values = torch.zeros((1, graph_slot_dim), dtype=torch.float32, device=device)
    graph_feature_bins = torch.zeros((1, graph_slot_dim), dtype=torch.long, device=device)
    graph_slot_mask = torch.tensor([schema.task_graph_slot_active[task_idx]], dtype=torch.float32, device=device) if graph_slot_dim > 0 else torch.zeros((1, 0), dtype=torch.float32, device=device)
    graph_binary_mask = torch.tensor([1.0 if _slot_is_binary(slot) else 0.0 for slot in graph_slots], dtype=torch.float32, device=device) if graph_slot_dim > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
    graph_pred_bins = _mixed_discrete_predictions(outputs["graph_discrete_logits"][0], graph_binary_mask).view(1, graph_slot_dim) if graph_slot_dim > 0 else torch.zeros((1, 0), dtype=torch.long, device=device)
    for slot_idx, slot in enumerate(graph_slots):
        if not _slot_is_discrete(slot):
            graph_feature_values[0, slot_idx] = outputs["graph_scalar_values"][0, slot_idx]
        else:
            graph_feature_bins[0, slot_idx] = graph_pred_bins[0, slot_idx].clamp(min=0, max=max(1, schema.max_input_discrete_value) - 1)

    return {
        "num_nodes": num_nodes,
        "node_feature_bins": node_feature_bins,
        "node_feature_values": node_feature_values,
        "node_slot_mask": node_slot_mask,
        "edge_index": edge_index,
        "edge_feature_bins": edge_feature_bins,
        "edge_feature_values": edge_feature_values,
        "edge_slot_mask": edge_slot_mask,
        "graph_feature_bins": graph_feature_bins,
        "graph_feature_values": graph_feature_values,
        "graph_slot_mask": graph_slot_mask,
        "graph_index": torch.zeros((num_nodes,), dtype=torch.long, device=device),
        "graph_ptr": torch.tensor([0, num_nodes], dtype=torch.long, device=device),
        "task_ids": torch.tensor([task_to_idx[task]], dtype=torch.long, device=device),
    }
