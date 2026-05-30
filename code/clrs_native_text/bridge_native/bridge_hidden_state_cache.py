#!/usr/bin/env python3
"""Utilities for caching frozen-LM hidden states for bridge training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


V1_FORMAT = "clrs_text_bridge_hidden_states_v1"
V2_SHARDED_FORMAT = "clrs_text_bridge_hidden_states_v2_sharded"


@dataclass
class BridgeHiddenStateCache:
    cache_format: str
    source_jsonl: str
    llm_model_name: str
    llm_layer_index: int
    llm_hidden_size: int
    max_source_length: int
    cache_dtype: str
    sample_ids: list[str]
    lengths: list[int]
    offsets: list[int] = field(default_factory=list)
    hidden_states: torch.Tensor | None = None
    shard_dir: str = ""
    shard_files: list[str] = field(default_factory=list)
    sample_to_shard: list[int] = field(default_factory=list)
    sample_to_local: list[int] = field(default_factory=list)
    _loaded_shards: dict[int, "BridgeHiddenStateCache"] = field(default_factory=dict, init=False, repr=False)

    def is_sharded(self) -> bool:
        return self.cache_format == V2_SHARDED_FORMAT

    def to_payload(self) -> dict[str, Any]:
        return {
            "cache_format": self.cache_format,
            "source_jsonl": self.source_jsonl,
            "llm_model_name": self.llm_model_name,
            "llm_layer_index": self.llm_layer_index,
            "llm_hidden_size": self.llm_hidden_size,
            "max_source_length": self.max_source_length,
            "cache_dtype": self.cache_dtype,
            "sample_ids": list(self.sample_ids),
            "lengths": list(self.lengths),
            "offsets": list(self.offsets),
            "hidden_states": self.hidden_states,
            "shard_dir": self.shard_dir,
            "shard_files": list(self.shard_files),
            "sample_to_shard": list(self.sample_to_shard),
            "sample_to_local": list(self.sample_to_local),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BridgeHiddenStateCache":
        hidden_states = payload.get("hidden_states")
        if hidden_states is not None and not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Hidden-state cache payload has non-tensor 'hidden_states'")
        return cls(
            cache_format=str(payload.get("cache_format", "")),
            source_jsonl=str(payload.get("source_jsonl", "")),
            llm_model_name=str(payload.get("llm_model_name", "")),
            llm_layer_index=int(payload.get("llm_layer_index", -1)),
            llm_hidden_size=int(payload.get("llm_hidden_size", 0)),
            max_source_length=int(payload.get("max_source_length", 0)),
            cache_dtype=str(payload.get("cache_dtype", str(getattr(hidden_states, "dtype", "float32")))),
            sample_ids=[str(v) for v in payload.get("sample_ids", [])],
            lengths=[int(v) for v in payload.get("lengths", [])],
            offsets=[int(v) for v in payload.get("offsets", [])],
            hidden_states=hidden_states,
            shard_dir=str(payload.get("shard_dir", "")),
            shard_files=[str(v) for v in payload.get("shard_files", [])],
            sample_to_shard=[int(v) for v in payload.get("sample_to_shard", [])],
            sample_to_local=[int(v) for v in payload.get("sample_to_local", [])],
        )

    def _load_shard(self, shard_idx: int) -> "BridgeHiddenStateCache":
        if shard_idx in self._loaded_shards:
            return self._loaded_shards[shard_idx]
        if not self.is_sharded():
            raise ValueError("Shard loading requested from non-sharded cache")
        shard_dir = Path(self.shard_dir)
        shard_path = shard_dir / self.shard_files[shard_idx]
        shard = load_hidden_state_cache(shard_path)
        self._loaded_shards = {shard_idx: shard}
        return shard

    def get_hidden_states(self, sample_idx: int) -> torch.Tensor:
        if self.hidden_states is not None:
            start = int(self.offsets[sample_idx])
            end = int(self.offsets[sample_idx + 1])
            return self.hidden_states[start:end]
        if not self.is_sharded():
            raise ValueError("Cache does not contain hidden states or shard metadata")
        shard_idx = int(self.sample_to_shard[sample_idx])
        local_idx = int(self.sample_to_local[sample_idx])
        shard = self._load_shard(shard_idx)
        return shard.get_hidden_states(local_idx)


def save_hidden_state_cache(path: Path, cache: BridgeHiddenStateCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache.to_payload(), path)


def save_hidden_state_cache_sharded(
    path: Path,
    *,
    source_jsonl: str,
    llm_model_name: str,
    llm_layer_index: int,
    llm_hidden_size: int,
    max_source_length: int,
    cache_dtype: str,
    sample_ids: list[str],
    lengths: list[int],
    shard_files: list[str],
    sample_to_shard: list[int],
    sample_to_local: list[int],
) -> None:
    shard_dir = Path(f"{path}.shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = BridgeHiddenStateCache(
        cache_format=V2_SHARDED_FORMAT,
        source_jsonl=str(source_jsonl),
        llm_model_name=str(llm_model_name),
        llm_layer_index=int(llm_layer_index),
        llm_hidden_size=int(llm_hidden_size),
        max_source_length=int(max_source_length),
        cache_dtype=str(cache_dtype),
        sample_ids=list(sample_ids),
        lengths=list(lengths),
        offsets=[],
        hidden_states=None,
        shard_dir=str(shard_dir),
        shard_files=list(shard_files),
        sample_to_shard=list(sample_to_shard),
        sample_to_local=list(sample_to_local),
    )
    save_hidden_state_cache(path, manifest)


def load_hidden_state_cache(path: Path) -> BridgeHiddenStateCache:
    payload = torch.load(path, map_location="cpu")
    cache = BridgeHiddenStateCache.from_payload(payload)
    if cache.cache_format == V1_FORMAT:
        if cache.hidden_states is None:
            raise ValueError("V1 hidden-state cache is missing in-memory hidden_states")
        if len(cache.sample_ids) != len(cache.lengths):
            raise ValueError("Hidden-state cache sample_ids and lengths have mismatched lengths")
        if len(cache.offsets) != len(cache.lengths) + 1:
            raise ValueError("Hidden-state cache offsets must have length len(lengths)+1")
        if cache.offsets[0] != 0:
            raise ValueError("Hidden-state cache offsets must start at 0")
        if cache.offsets[-1] != int(cache.hidden_states.shape[0]):
            raise ValueError("Hidden-state cache offsets do not match hidden state tensor length")
        if int(cache.hidden_states.shape[1]) != int(cache.llm_hidden_size):
            raise ValueError("Hidden-state cache hidden_size metadata does not match tensor shape")
        return cache
    if cache.cache_format == V2_SHARDED_FORMAT:
        if len(cache.sample_ids) != len(cache.lengths):
            raise ValueError("Sharded cache sample_ids and lengths have mismatched lengths")
        if len(cache.sample_to_shard) != len(cache.sample_ids):
            raise ValueError("Sharded cache sample_to_shard length mismatch")
        if len(cache.sample_to_local) != len(cache.sample_ids):
            raise ValueError("Sharded cache sample_to_local length mismatch")
        if not cache.shard_files:
            raise ValueError("Sharded cache manifest has no shard files")
        if not Path(cache.shard_dir).is_dir():
            raise ValueError(f"Sharded cache directory does not exist: {cache.shard_dir}")
        return cache
    raise ValueError(f"Unsupported hidden-state cache format: {cache.cache_format!r}")
