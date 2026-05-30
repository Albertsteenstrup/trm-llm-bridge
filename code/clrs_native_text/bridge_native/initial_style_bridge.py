#!/usr/bin/env python3
"""Direct CLRS-Text bridge models adapted from the initial Sudoku bridges.

This module keeps the original design philosophy from the Sudoku path:
- freeze a text LM
- read one hidden-state stream from that LM
- train only a small bridge that predicts structured solver-facing inputs

Unlike the Sudoku version, the CLRS bridge targets variable-size graph inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class BridgeSchema:
    task_names: list[str]
    feature_names: list[str]
    feature_mins: list[float]
    feature_maxs: list[float]
    graph_scalar_names: list[str]
    graph_scalar_mins: list[float]
    graph_scalar_maxs: list[float]
    edge_value_min: float
    edge_value_max: float
    feature_quant_bins: int = 256

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def graph_scalar_dim(self) -> int:
        return len(self.graph_scalar_names)

    def feature_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.feature_names)}

    def graph_scalar_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.graph_scalar_names)}

    def task_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.task_names)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeSchema":
        return cls(
            task_names=[str(v) for v in payload.get("task_names", [])],
            feature_names=[str(v) for v in payload.get("feature_names", [])],
            feature_mins=[float(v) for v in payload.get("feature_mins", [])],
            feature_maxs=[float(v) for v in payload.get("feature_maxs", [])],
            graph_scalar_names=[str(v) for v in payload.get("graph_scalar_names", [])],
            graph_scalar_mins=[float(v) for v in payload.get("graph_scalar_mins", [])],
            graph_scalar_maxs=[float(v) for v in payload.get("graph_scalar_maxs", [])],
            edge_value_min=float(payload.get("edge_value_min", 0.0)),
            edge_value_max=float(payload.get("edge_value_max", 1.0)),
            feature_quant_bins=int(payload.get("feature_quant_bins", 256)),
        )


@dataclass
class DirectBridgeBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    task_ids: torch.Tensor
    present_target: torch.Tensor
    node_feature_values: torch.Tensor
    node_feature_mask: torch.Tensor
    edge_adj: torch.Tensor
    edge_values: torch.Tensor
    edge_value_mask: torch.Tensor
    graph_scalar_values: torch.Tensor
    graph_scalar_mask: torch.Tensor
    num_nodes: torch.Tensor
    sample_ids: list[str]
    tasks: list[str]


@dataclass
class DirectBridgeConfig:
    llm_hidden_size: int
    max_nodes: int
    bridge_dim: int = 512
    bridge_heads: int = 8
    bridge_layers: int = 4
    bridge_dropout: float = 0.1
    query_pos_dropout: float = 0.05
    transnar_recurrence_steps: int = 1
    count_blend_alpha: float = 0.5
    count_decode_mode: str = "blended"
    scalar_head_type: str = "linear"
    scalar_head_hidden_mult: float = 1.0
    qformer_edge_self_bias: bool = False
    qformer_rel_pos_self_bias: bool = False
    edge_text_prior_logit_scale: float = 0.0
    arch: str = "qformer"


def _numeric_graph_inputs(graph_inputs: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in graph_inputs.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[str(key)] = float(value)
    return out


def _normalize_scalar(value: float, lower: float, upper: float) -> float:
    denom = upper - lower
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip((value - lower) / denom, 0.0, 1.0))


def _denormalize_tensor(values: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    values = values.clamp(0.0, 1.0)
    return mins + (values * (maxs - mins))


def _unit_minmax_tensors(shape: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.zeros(shape, dtype=torch.float32, device=device)
    maxs = torch.ones(shape, dtype=torch.float32, device=device)
    return mins, maxs


def build_bridge_schema(rows: list[dict[str, Any]], *, feature_quant_bins: int = 256) -> BridgeSchema:
    task_names: set[str] = set()
    feature_names: set[str] = set()
    graph_scalar_names: set[str] = set()
    feature_mins: dict[str, float] = {}
    feature_maxs: dict[str, float] = {}
    graph_scalar_mins: dict[str, float] = {}
    graph_scalar_maxs: dict[str, float] = {}
    edge_value_min = 0.0
    edge_value_max = 1.0
    saw_edge_values = False

    def _update_min_max(store_min: dict[str, float], store_max: dict[str, float], key: str, value: float) -> None:
        store_min[key] = value if key not in store_min else min(store_min[key], value)
        store_max[key] = value if key not in store_max else max(store_max[key], value)

    for row in rows:
        task = str(row.get("algo_name", row.get("graph_target", {}).get("algorithm", "unknown"))).strip() or "unknown"
        task_names.add(task)
        graph_target = row.get("graph_target", {})
        names = [str(v) for v in graph_target.get("feature_names", [])]
        values = graph_target.get("node_feature_values", [])
        for local_idx, name in enumerate(names):
            feature_names.add(name)
            for node_values in values:
                if local_idx >= len(node_values):
                    continue
                value = float(node_values[local_idx])
                _update_min_max(feature_mins, feature_maxs, name, value)

        for key, value in _numeric_graph_inputs(graph_target.get("graph_inputs", {})).items():
            graph_scalar_names.add(key)
            _update_min_max(graph_scalar_mins, graph_scalar_maxs, key, float(value))

        edge_values = graph_target.get("edge_values", [])
        for raw in edge_values:
            value = float(raw)
            if not saw_edge_values:
                edge_value_min = value
                edge_value_max = value
                saw_edge_values = True
            else:
                edge_value_min = min(edge_value_min, value)
                edge_value_max = max(edge_value_max, value)

    ordered_feature_names = sorted(feature_names)
    ordered_graph_scalar_names = sorted(graph_scalar_names)

    def _ordered(store: dict[str, float], names: list[str], default: float) -> list[float]:
        return [float(store.get(name, default)) for name in names]

    return BridgeSchema(
        task_names=sorted(task_names),
        feature_names=ordered_feature_names,
        feature_mins=_ordered(feature_mins, ordered_feature_names, 0.0),
        feature_maxs=_ordered(feature_maxs, ordered_feature_names, 1.0),
        graph_scalar_names=ordered_graph_scalar_names,
        graph_scalar_mins=_ordered(graph_scalar_mins, ordered_graph_scalar_names, 0.0),
        graph_scalar_maxs=_ordered(graph_scalar_maxs, ordered_graph_scalar_names, 1.0),
        edge_value_min=float(edge_value_min),
        edge_value_max=float(edge_value_max),
        feature_quant_bins=int(feature_quant_bins),
    )


class CLRSTextDirectBridgeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        schema: BridgeSchema,
        max_nodes: int,
    ) -> None:
        self.rows = rows
        self.schema = schema
        self.max_nodes = int(max_nodes)
        self._task_index = schema.task_index()
        self._feature_index = schema.feature_index()
        self._graph_scalar_index = schema.graph_scalar_index()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        graph_target = row["graph_target"]
        question = str(row.get("question", ""))
        sample_id = str(row.get("sample_id", idx))
        task = str(row.get("algo_name", graph_target.get("algorithm", "unknown"))).strip() or "unknown"
        task_id = int(self._task_index.get(task, 0))

        num_nodes = max(0, min(self.max_nodes, int(graph_target.get("num_nodes", 0))))
        present_target = torch.zeros(self.max_nodes, dtype=torch.float32)
        if num_nodes > 0:
            present_target[:num_nodes] = 1.0

        node_feature_values = torch.zeros((self.max_nodes, self.schema.feature_dim), dtype=torch.float32)
        node_feature_mask = torch.zeros((self.max_nodes, self.schema.feature_dim), dtype=torch.float32)

        local_feature_names = [str(v) for v in graph_target.get("feature_names", [])]
        local_feature_values = graph_target.get("node_feature_values", [])
        for node_idx, per_node in enumerate(local_feature_values[:num_nodes]):
            for local_idx, name in enumerate(local_feature_names):
                if local_idx >= len(per_node):
                    continue
                global_idx = self._feature_index.get(name)
                if global_idx is None:
                    continue
                normalized = _normalize_scalar(
                    float(per_node[local_idx]),
                    float(self.schema.feature_mins[global_idx]),
                    float(self.schema.feature_maxs[global_idx]),
                )
                node_feature_values[node_idx, global_idx] = normalized
                node_feature_mask[node_idx, global_idx] = 1.0

        edge_adj = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        edge_values = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        edge_value_mask = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        raw_edge_index = graph_target.get("edge_index", [])
        raw_edge_values = graph_target.get("edge_values", [])
        for edge_idx, pair in enumerate(raw_edge_index):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            src = int(pair[0])
            dst = int(pair[1])
            if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
                continue
            edge_adj[src, dst] = 1.0
            if edge_idx < len(raw_edge_values):
                edge_values[src, dst] = _normalize_scalar(
                    float(raw_edge_values[edge_idx]),
                    float(self.schema.edge_value_min),
                    float(self.schema.edge_value_max),
                )
                edge_value_mask[src, dst] = 1.0

        graph_scalar_values = torch.zeros(self.schema.graph_scalar_dim, dtype=torch.float32)
        graph_scalar_mask = torch.zeros(self.schema.graph_scalar_dim, dtype=torch.float32)
        for key, value in _numeric_graph_inputs(graph_target.get("graph_inputs", {})).items():
            scalar_idx = self._graph_scalar_index.get(key)
            if scalar_idx is None:
                continue
            graph_scalar_values[scalar_idx] = _normalize_scalar(
                float(value),
                float(self.schema.graph_scalar_mins[scalar_idx]),
                float(self.schema.graph_scalar_maxs[scalar_idx]),
            )
            graph_scalar_mask[scalar_idx] = 1.0

        return {
            "question": question,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "present_target": present_target,
            "node_feature_values": node_feature_values,
            "node_feature_mask": node_feature_mask,
            "edge_adj": edge_adj,
            "edge_values": edge_values,
            "edge_value_mask": edge_value_mask,
            "graph_scalar_values": graph_scalar_values,
            "graph_scalar_mask": graph_scalar_mask,
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            "sample_id": sample_id,
            "task": task,
        }


def make_direct_bridge_collate(tokenizer, *, max_source_length: int):
    def collate(batch: list[dict[str, Any]]) -> DirectBridgeBatch:
        questions = [item["question"] for item in batch]
        enc = tokenizer(
            questions,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_source_length,
        )
        return DirectBridgeBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            task_ids=torch.stack([item["task_id"] for item in batch], dim=0),
            present_target=torch.stack([item["present_target"] for item in batch], dim=0),
            node_feature_values=torch.stack([item["node_feature_values"] for item in batch], dim=0),
            node_feature_mask=torch.stack([item["node_feature_mask"] for item in batch], dim=0),
            edge_adj=torch.stack([item["edge_adj"] for item in batch], dim=0),
            edge_values=torch.stack([item["edge_values"] for item in batch], dim=0),
            edge_value_mask=torch.stack([item["edge_value_mask"] for item in batch], dim=0),
            graph_scalar_values=torch.stack([item["graph_scalar_values"] for item in batch], dim=0),
            graph_scalar_mask=torch.stack([item["graph_scalar_mask"] for item in batch], dim=0),
            num_nodes=torch.stack([item["num_nodes"] for item in batch], dim=0),
            sample_ids=[str(item["sample_id"]) for item in batch],
            tasks=[str(item["task"]) for item in batch],
        )

    return collate


def _masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _build_generic_positional_embeddings(max_nodes: int, dim: int) -> torch.Tensor:
    pos_embed = torch.zeros(max_nodes, dim)
    half = max(1, dim // 2)
    with torch.no_grad():
        for node_idx in range(max_nodes):
            rel = float(node_idx) / float(max(1, max_nodes - 1))
            for j in range(half):
                freq = 1.0 / (10000.0 ** (2.0 * j / float(max(1, half))))
                pos_embed[node_idx, j] = math.sin(rel * freq)
                if half + j < dim:
                    pos_embed[node_idx, half + j] = math.cos(rel * freq)
    return pos_embed


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None,
        *,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = ~kv_mask.bool() if kv_mask is not None else None
        x = queries
        cross_out, _ = self.cross_attn(x, kv, kv, key_padding_mask=key_padding_mask)
        x = self.norm1(x + cross_out)
        self_out, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.norm2(x + self_out)
        x = self.norm3(x + self.ffn(x))
        return x


class TransNARInterleaveLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.node_cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.node_cross_norm_q = nn.LayerNorm(dim)
        self.node_cross_norm_kv = nn.LayerNorm(dim)
        self.node_self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.node_self_norm = nn.LayerNorm(dim)
        self.node_ffn_norm = nn.LayerNorm(dim)
        self.node_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.token_cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.token_cross_norm_q = nn.LayerNorm(dim)
        self.token_cross_norm_kv = nn.LayerNorm(dim)
        self.token_ffn_norm = nn.LayerNorm(dim)
        self.token_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        gate_init = torch.tensor(0.1)
        self.node_cross_gate = nn.Parameter(gate_init.clone())
        self.node_self_gate = nn.Parameter(gate_init.clone())
        self.node_ffn_gate = nn.Parameter(gate_init.clone())
        self.token_cross_gate = nn.Parameter(gate_init.clone())
        self.token_ffn_gate = nn.Parameter(gate_init.clone())

    def forward(
        self,
        *,
        node_states: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~token_mask.bool() if token_mask is not None else None

        node_q = self.node_cross_norm_q(node_states)
        token_kv = self.node_cross_norm_kv(token_states)
        node_cross_out, _ = self.node_cross_attn(node_q, token_kv, token_kv, key_padding_mask=key_padding_mask)
        node_states = node_states + torch.tanh(self.node_cross_gate) * node_cross_out

        node_self_in = self.node_self_norm(node_states)
        node_self_out, _ = self.node_self_attn(node_self_in, node_self_in, node_self_in)
        node_states = node_states + torch.tanh(self.node_self_gate) * node_self_out

        node_ffn_in = self.node_ffn_norm(node_states)
        node_states = node_states + torch.tanh(self.node_ffn_gate) * self.node_ffn(node_ffn_in)

        token_q = self.token_cross_norm_q(token_states)
        node_kv = self.token_cross_norm_kv(node_states)
        token_cross_out, _ = self.token_cross_attn(token_q, node_kv, node_kv)
        token_states = token_states + torch.tanh(self.token_cross_gate) * token_cross_out

        token_ffn_in = self.token_ffn_norm(token_states)
        token_states = token_states + torch.tanh(self.token_ffn_gate) * self.token_ffn(token_ffn_in)
        return node_states, token_states


class BridgeOutputHeads(nn.Module):
    def __init__(self, dim: int, feature_dim: int, graph_scalar_dim: int, num_tasks: int) -> None:
        super().__init__()
        self.task_embed = nn.Embedding(max(1, num_tasks), dim)
        self.task_node_proj = nn.Linear(dim, dim)
        self.task_global_proj = nn.Linear(dim, dim)
        self.task_feature_bias = nn.Linear(dim, feature_dim)
        self.task_graph_scalar_bias = nn.Linear(dim, graph_scalar_dim) if graph_scalar_dim > 0 else None
        self.present_head = nn.Linear(dim, 1)
        self.node_feature_head = nn.Linear(dim, feature_dim)
        self.edge_src = nn.Linear(dim, dim)
        self.edge_dst = nn.Linear(dim, dim)
        self.edge_val_src = nn.Linear(dim, dim)
        self.edge_val_dst = nn.Linear(dim, dim)
        self.graph_scalar_dim = graph_scalar_dim
        self.graph_scalar_head = nn.Linear(dim, graph_scalar_dim) if graph_scalar_dim > 0 else None

    def forward(self, node_states: torch.Tensor, pooled_tokens: torch.Tensor, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        task_embed = self.task_embed(task_ids)
        conditioned_nodes = node_states + self.task_node_proj(task_embed).unsqueeze(1)
        conditioned_pooled = pooled_tokens + self.task_global_proj(task_embed)

        present_logits = self.present_head(conditioned_nodes).squeeze(-1)
        node_feature_values = self.node_feature_head(conditioned_nodes) + self.task_feature_bias(task_embed).unsqueeze(1)
        src = self.edge_src(conditioned_nodes)
        dst = self.edge_dst(conditioned_nodes)
        edge_logits = torch.matmul(src, dst.transpose(1, 2)) / math.sqrt(max(1, src.size(-1)))

        val_src = self.edge_val_src(conditioned_nodes)
        val_dst = self.edge_val_dst(conditioned_nodes)
        edge_values = torch.matmul(val_src, val_dst.transpose(1, 2)) / math.sqrt(max(1, val_src.size(-1)))

        graph_scalar_values = (
            self.graph_scalar_head(conditioned_pooled) if self.graph_scalar_head is not None else conditioned_pooled.new_zeros((conditioned_pooled.size(0), 0))
        )
        if self.task_graph_scalar_bias is not None:
            graph_scalar_values = graph_scalar_values + self.task_graph_scalar_bias(task_embed)
        return {
            "present_logits": present_logits,
            "node_feature_values": node_feature_values,
            "edge_logits": edge_logits,
            "edge_values": edge_values,
            "graph_scalar_values": graph_scalar_values,
        }


class QFormerCLRSBridge(nn.Module):
    def __init__(self, config: DirectBridgeConfig, *, feature_dim: int, graph_scalar_dim: int, num_tasks: int) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.llm_hidden_size, config.bridge_dim)
        self.node_queries = nn.Parameter(torch.randn(config.max_nodes, config.bridge_dim) * 0.02)
        pos = _build_generic_positional_embeddings(config.max_nodes, config.bridge_dim)
        self.pos_embed = nn.Parameter(pos)
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(config.bridge_dim, config.bridge_heads, config.bridge_dropout) for _ in range(config.bridge_layers)]
        )
        self.heads = BridgeOutputHeads(
            config.bridge_dim,
            feature_dim=feature_dim,
            graph_scalar_dim=graph_scalar_dim,
            num_tasks=num_tasks,
        )

    def forward(self, llm_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        token_states = self.input_proj(llm_hidden_states)
        pooled_tokens = _masked_mean_pool(token_states, attention_mask)
        batch_size = llm_hidden_states.size(0)
        node_states = (self.node_queries + self.pos_embed).unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            node_states = layer(node_states, token_states, attention_mask)
        return self.heads(node_states, pooled_tokens, task_ids)


class TransNARCLRSBridge(nn.Module):
    def __init__(self, config: DirectBridgeConfig, *, feature_dim: int, graph_scalar_dim: int, num_tasks: int) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.llm_hidden_size, config.bridge_dim)
        self.node_queries = nn.Parameter(torch.randn(config.max_nodes, config.bridge_dim) * 0.02)
        pos = _build_generic_positional_embeddings(config.max_nodes, config.bridge_dim)
        self.pos_embed = nn.Parameter(pos)
        self.layers = nn.ModuleList(
            [TransNARInterleaveLayer(config.bridge_dim, config.bridge_heads, config.bridge_dropout) for _ in range(config.bridge_layers)]
        )
        self.heads = BridgeOutputHeads(
            config.bridge_dim,
            feature_dim=feature_dim,
            graph_scalar_dim=graph_scalar_dim,
            num_tasks=num_tasks,
        )

    def forward(self, llm_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        token_states = self.input_proj(llm_hidden_states)
        pooled_tokens = _masked_mean_pool(token_states, attention_mask)
        batch_size = llm_hidden_states.size(0)
        node_states = (self.node_queries + self.pos_embed).unsqueeze(0).expand(batch_size, -1, -1)
        recurrence_steps = max(1, int(self.config.transnar_recurrence_steps))
        for _ in range(recurrence_steps):
            for layer in self.layers:
                node_states, token_states = layer(node_states=node_states, token_states=token_states, token_mask=attention_mask)
        return self.heads(node_states, pooled_tokens, task_ids)


def build_bridge_model(
    arch: str,
    config: DirectBridgeConfig,
    *,
    feature_dim: int,
    graph_scalar_dim: int,
    num_tasks: int,
) -> nn.Module:
    arch = arch.strip().lower()
    if arch == "qformer":
        return QFormerCLRSBridge(config, feature_dim=feature_dim, graph_scalar_dim=graph_scalar_dim, num_tasks=num_tasks)
    if arch == "transnar":
        return TransNARCLRSBridge(config, feature_dim=feature_dim, graph_scalar_dim=graph_scalar_dim, num_tasks=num_tasks)
    raise ValueError(f"Unsupported bridge arch: {arch}")


def quantize_dense_values(
    values: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
    *,
    bins: int,
) -> torch.Tensor:
    if bins <= 1:
        return torch.zeros_like(values, dtype=torch.long)
    mins = mins.to(device=values.device, dtype=values.dtype)
    maxs = maxs.to(device=values.device, dtype=values.dtype)
    denom = torch.where((maxs - mins).abs() < 1e-8, torch.ones_like(maxs), maxs - mins)
    scaled = (values - mins) / denom
    q = torch.round(scaled * float(bins - 1))
    q = torch.clamp(q, 0.0, float(bins - 1))
    return q.to(dtype=torch.long)


def schema_feature_minmax_tensors(schema: BridgeSchema, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.tensor(schema.feature_mins, dtype=torch.float32, device=device)
    maxs = torch.tensor(schema.feature_maxs, dtype=torch.float32, device=device)
    return mins.view(1, 1, -1), maxs.view(1, 1, -1)


def schema_graph_scalar_minmax_tensors(schema: BridgeSchema, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.tensor(schema.graph_scalar_mins, dtype=torch.float32, device=device)
    maxs = torch.tensor(schema.graph_scalar_maxs, dtype=torch.float32, device=device)
    return mins.view(1, -1), maxs.view(1, -1)


def schema_edge_value_minmax_tensors(schema: BridgeSchema, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.tensor([schema.edge_value_min], dtype=torch.float32, device=device)
    maxs = torch.tensor([schema.edge_value_max], dtype=torch.float32, device=device)
    return mins.view(1, 1, 1), maxs.view(1, 1, 1)


def compute_bridge_losses(
    outputs: dict[str, torch.Tensor],
    batch: DirectBridgeBatch,
    *,
    loss_weight_present: float,
    loss_weight_feature: float,
    loss_weight_edge: float,
    loss_weight_edge_value: float,
    loss_weight_graph_scalar: float,
    loss_weight_count: float,
    edge_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    present_loss = F.binary_cross_entropy_with_logits(outputs["present_logits"], batch.present_target)
    max_nodes = float(max(1, int(batch.present_target.size(1))))
    pred_node_count = torch.sigmoid(outputs["present_logits"]).sum(dim=1)
    true_node_count = batch.num_nodes.float()
    count_loss = F.smooth_l1_loss(pred_node_count / max_nodes, true_node_count / max_nodes)

    feature_mask = batch.node_feature_mask > 0.5
    if feature_mask.any():
        feature_loss = F.smooth_l1_loss(outputs["node_feature_values"][feature_mask], batch.node_feature_values[feature_mask])
    else:
        feature_loss = outputs["node_feature_values"].sum() * 0.0

    valid_pair_mask = (batch.present_target > 0.5).unsqueeze(2) & (batch.present_target > 0.5).unsqueeze(1)
    edge_loss = F.binary_cross_entropy_with_logits(
        outputs["edge_logits"][valid_pair_mask],
        batch.edge_adj[valid_pair_mask],
        pos_weight=torch.tensor(float(max(1.0, edge_pos_weight)), device=outputs["edge_logits"].device),
    ) if valid_pair_mask.any() else outputs["edge_logits"].sum() * 0.0

    edge_value_mask = batch.edge_value_mask > 0.5
    if edge_value_mask.any():
        edge_value_loss = F.smooth_l1_loss(outputs["edge_values"][edge_value_mask], batch.edge_values[edge_value_mask])
    else:
        edge_value_loss = outputs["edge_values"].sum() * 0.0

    graph_scalar_mask = batch.graph_scalar_mask > 0.5
    if graph_scalar_mask.any():
        graph_scalar_loss = F.smooth_l1_loss(
            outputs["graph_scalar_values"][graph_scalar_mask],
            batch.graph_scalar_values[graph_scalar_mask],
        )
    else:
        graph_scalar_loss = outputs["graph_scalar_values"].sum() * 0.0

    total_loss = (
        (loss_weight_present * present_loss)
        + (loss_weight_feature * feature_loss)
        + (loss_weight_edge * edge_loss)
        + (loss_weight_edge_value * edge_value_loss)
        + (loss_weight_graph_scalar * graph_scalar_loss)
        + (loss_weight_count * count_loss)
    )
    return total_loss, {
        "loss/present": float(present_loss.item()),
        "loss/feature": float(feature_loss.item()),
        "loss/edge": float(edge_loss.item()),
        "loss/edge_value": float(edge_value_loss.item()),
        "loss/graph_scalar": float(graph_scalar_loss.item()),
        "loss/count": float(count_loss.item()),
    }


def compute_bridge_batch_metrics(
    outputs: dict[str, torch.Tensor],
    batch: DirectBridgeBatch,
    *,
    schema: BridgeSchema,
    present_threshold: float,
    edge_threshold: float,
) -> dict[str, float]:
    device = outputs["present_logits"].device
    present_pred = (torch.sigmoid(outputs["present_logits"]) >= present_threshold).float()
    pred_node_count = present_pred.sum(dim=1).to(dtype=batch.num_nodes.dtype)
    true_node_count = batch.num_nodes
    present_count_acc = (pred_node_count == true_node_count).float().mean().item()
    present_count_mae = torch.abs(pred_node_count.float() - true_node_count.float()).mean().item()

    feature_mask = batch.node_feature_mask > 0.5
    pred_feature_values = outputs["node_feature_values"].clamp(0.0, 1.0)
    if feature_mask.any():
        fmins, fmaxs = schema_feature_minmax_tensors(schema, device)
        pred_feature_raw = _denormalize_tensor(pred_feature_values, fmins, fmaxs)
        true_feature_raw = _denormalize_tensor(batch.node_feature_values, fmins, fmaxs)
        feature_mae = torch.abs(pred_feature_raw[feature_mask] - true_feature_raw[feature_mask]).mean().item()
        umins, umaxs = _unit_minmax_tensors((1, 1, schema.feature_dim), device)
        pred_bins = quantize_dense_values(pred_feature_values, umins, umaxs, bins=schema.feature_quant_bins)
        true_bins = quantize_dense_values(batch.node_feature_values.clamp(0.0, 1.0), umins, umaxs, bins=schema.feature_quant_bins)
        feature_bin_acc = (pred_bins[feature_mask] == true_bins[feature_mask]).float().mean().item()
    else:
        feature_mae = 0.0
        feature_bin_acc = 0.0
        pred_bins = torch.zeros_like(batch.node_feature_values, dtype=torch.long)
        true_bins = torch.zeros_like(batch.node_feature_values, dtype=torch.long)

    valid_pair_mask = (batch.present_target > 0.5).unsqueeze(2) & (batch.present_target > 0.5).unsqueeze(1)
    edge_pred = (torch.sigmoid(outputs["edge_logits"]) >= edge_threshold).float()
    if valid_pair_mask.any():
        pred_e = edge_pred[valid_pair_mask]
        true_e = batch.edge_adj[valid_pair_mask]
        tp = ((pred_e == 1) & (true_e == 1)).sum().item()
        fp = ((pred_e == 1) & (true_e == 0)).sum().item()
        fn = ((pred_e == 0) & (true_e == 1)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        edge_f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    else:
        edge_f1 = 0.0

    edge_value_mask = batch.edge_value_mask > 0.5
    pred_edge_values = outputs["edge_values"].clamp(0.0, 1.0)
    if edge_value_mask.any():
        emins, emaxs = schema_edge_value_minmax_tensors(schema, device)
        pred_edge_raw = _denormalize_tensor(pred_edge_values, emins, emaxs)
        true_edge_raw = _denormalize_tensor(batch.edge_values, emins, emaxs)
        edge_value_mae = torch.abs(pred_edge_raw[edge_value_mask] - true_edge_raw[edge_value_mask]).mean().item()
        umins, umaxs = _unit_minmax_tensors((1, 1, 1), device)
        pred_edge_bins = quantize_dense_values(pred_edge_values, umins, umaxs, bins=schema.feature_quant_bins)
        true_edge_bins = quantize_dense_values(batch.edge_values.clamp(0.0, 1.0), umins, umaxs, bins=schema.feature_quant_bins)
    else:
        edge_value_mae = 0.0
        pred_edge_bins = torch.zeros_like(batch.edge_values, dtype=torch.long)
        true_edge_bins = torch.zeros_like(batch.edge_values, dtype=torch.long)

    graph_scalar_mask = batch.graph_scalar_mask > 0.5
    pred_graph_scalar_values = outputs["graph_scalar_values"].clamp(0.0, 1.0)
    if graph_scalar_mask.any():
        gmins, gmaxs = schema_graph_scalar_minmax_tensors(schema, device)
        pred_graph_scalar_raw = _denormalize_tensor(pred_graph_scalar_values, gmins, gmaxs)
        true_graph_scalar_raw = _denormalize_tensor(batch.graph_scalar_values, gmins, gmaxs)
        graph_scalar_mae = torch.abs(
            pred_graph_scalar_raw[graph_scalar_mask] - true_graph_scalar_raw[graph_scalar_mask]
        ).mean().item()
        umins, umaxs = _unit_minmax_tensors((1, schema.graph_scalar_dim), device)
        pred_graph_bins = quantize_dense_values(
            pred_graph_scalar_values,
            umins,
            umaxs,
            bins=schema.feature_quant_bins,
        )
        true_graph_bins = quantize_dense_values(
            batch.graph_scalar_values.clamp(0.0, 1.0),
            umins,
            umaxs,
            bins=schema.feature_quant_bins,
        )
        graph_scalar_bin_acc = (pred_graph_bins[graph_scalar_mask] == true_graph_bins[graph_scalar_mask]).float().mean().item()
    else:
        graph_scalar_mae = 0.0
        graph_scalar_bin_acc = 1.0
        pred_graph_bins = torch.zeros_like(batch.graph_scalar_values, dtype=torch.long)
        true_graph_bins = torch.zeros_like(batch.graph_scalar_values, dtype=torch.long)

    graph_exact_hits = 0
    for sample_idx in range(batch.num_nodes.size(0)):
        n_true = int(batch.num_nodes[sample_idx].item())
        n_pred = int(present_pred[sample_idx].sum().item())
        if n_pred != n_true:
            continue
        feature_ok = True
        if n_true > 0:
            sample_feature_mask = feature_mask[sample_idx, :n_true]
            if sample_feature_mask.any():
                feature_ok = bool(torch.equal(pred_bins[sample_idx, :n_true][sample_feature_mask], true_bins[sample_idx, :n_true][sample_feature_mask]))
        edge_ok = bool(torch.equal(edge_pred[sample_idx, :n_true, :n_true], batch.edge_adj[sample_idx, :n_true, :n_true]))
        edge_value_ok = True
        sample_edge_mask = edge_value_mask[sample_idx, :n_true, :n_true]
        if sample_edge_mask.any():
            edge_value_ok = bool(
                torch.equal(
                    pred_edge_bins[sample_idx, :n_true, :n_true][sample_edge_mask],
                    true_edge_bins[sample_idx, :n_true, :n_true][sample_edge_mask],
                )
            )
        graph_scalar_ok = True
        sample_scalar_mask = graph_scalar_mask[sample_idx]
        if sample_scalar_mask.any():
            graph_scalar_ok = bool(
                torch.equal(
                    pred_graph_bins[sample_idx][sample_scalar_mask],
                    true_graph_bins[sample_idx][sample_scalar_mask],
                )
            )
        if feature_ok and edge_ok and edge_value_ok and graph_scalar_ok:
            graph_exact_hits += 1

    graph_exact = graph_exact_hits / max(1, batch.num_nodes.size(0))
    return {
        "metric/feature_mae": feature_mae,
        "metric/feature_bin_acc": feature_bin_acc,
        "metric/edge_f1": edge_f1,
        "metric/edge_value_mae": edge_value_mae,
        "metric/graph_scalar_mae": graph_scalar_mae,
        "metric/graph_scalar_bin_acc": graph_scalar_bin_acc,
        "metric/present_count_acc": present_count_acc,
        "metric/present_count_mae": present_count_mae,
        "metric/graph_exact": graph_exact,
    }


def build_graph_target_from_prediction(
    *,
    row: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    schema: BridgeSchema,
    present_threshold: float,
    edge_threshold: float,
) -> dict[str, Any]:
    present_logits = outputs["present_logits"][0]
    node_feature_values = outputs["node_feature_values"][0].clamp(0.0, 1.0)
    edge_logits = outputs["edge_logits"][0]
    edge_values = outputs["edge_values"][0].clamp(0.0, 1.0)
    graph_scalar_values = outputs["graph_scalar_values"][0].clamp(0.0, 1.0) if outputs["graph_scalar_values"].numel() > 0 else None

    num_nodes = int((torch.sigmoid(present_logits) >= present_threshold).sum().item())
    num_nodes = max(1, num_nodes)
    fmins, fmaxs = schema_feature_minmax_tensors(schema, node_feature_values.device)
    node_feature_values_trimmed = _denormalize_tensor(
        node_feature_values[:num_nodes].unsqueeze(0),
        fmins,
        fmaxs,
    )[0].detach().cpu()

    umins, umaxs = _unit_minmax_tensors((1, 1, schema.feature_dim), node_feature_values.device)
    node_feature_bins = quantize_dense_values(
        node_feature_values[:num_nodes].unsqueeze(0),
        umins,
        umaxs,
        bins=schema.feature_quant_bins,
    )[0].detach().cpu().tolist()

    edge_index: list[list[int]] = []
    edge_value_list: list[float] = []
    edge_probs = torch.sigmoid(edge_logits[:num_nodes, :num_nodes])
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if float(edge_probs[src, dst].item()) >= edge_threshold:
                edge_index.append([src, dst])
                emins, emaxs = schema_edge_value_minmax_tensors(schema, edge_values.device)
                edge_value_list.append(float(_denormalize_tensor(edge_values[src, dst].view(1, 1, 1), emins, emaxs).item()))

    graph_inputs: dict[str, float] = {}
    if graph_scalar_values is not None:
        gmins, gmaxs = schema_graph_scalar_minmax_tensors(schema, graph_scalar_values.device)
        graph_scalar_raw = _denormalize_tensor(graph_scalar_values.view(1, -1), gmins, gmaxs)[0]
        for idx, name in enumerate(schema.graph_scalar_names):
            graph_inputs[name] = float(graph_scalar_raw[idx].item())

    algorithm = str(row.get("algo_name", row.get("graph_target", {}).get("algorithm", "unknown"))).strip() or "unknown"
    return {
        "algorithm": algorithm,
        "num_nodes": num_nodes,
        "feature_names": list(schema.feature_names),
        "node_feature_values": node_feature_values_trimmed.tolist(),
        "node_feature_bins": node_feature_bins,
        "edge_index": edge_index,
        "edge_values": edge_value_list,
        "graph_inputs": graph_inputs,
        "adapter_version": "bridge_v2_initial_direct",
    }
