#!/usr/bin/env python3
"""Build a no-algorithm-name NL curriculum for CLRS native-input bridging.

The generated questions intentionally avoid explicit CLRS algorithm names and
native slot names. Rows keep the real ``algo_name`` for downstream specialist
evaluation, but set ``bridge_task_label=unknown`` so the bridge is not
task-conditioned during training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from build_hard_nl_native_bridge_dataset import (  # noqa: E402
    _active_indices,
    _active_pairs,
    _as_number,
    _categorical_ids,
    _edge_triples,
    _find_edge_mask,
    _infer_num_nodes,
    _native_target,
    _one_hot_index,
    _read_jsonl,
    _write_jsonl,
)


LEVEL_NAMES = {
    0: "clean_semantic",
    1: "clean_multi_template",
    2: "mixed_layout",
    3: "messy_notes",
    4: "hard_human_ood_like",
}


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _node_inputs(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dp for dp in inputs if str(dp.get("location")) == "node" and str(dp.get("name")) != "pos"]


def _edge_inputs(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dp for dp in inputs if str(dp.get("location")) == "edge"]


def _graph_inputs(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dp for dp in inputs if str(dp.get("location")) == "graph"]


def _looks_like_graph(inputs: list[dict[str, Any]]) -> bool:
    return any(str(dp.get("location")) == "edge" for dp in inputs)


def _input_by_name(inputs: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for dp in inputs:
        if str(dp.get("name")) == name:
            return dp
    return None


def _human_label(name: str) -> str:
    mapping = {
        "key": "values",
        "string": "text flags",
        "target": "target value",
        "s": "start/source",
        "x": "x coordinate",
        "y": "y coordinate",
        "p": "p sequence",
        "q": "q sequence",
        "d": "deadline",
        "w": "weight/reward",
        "f": "finish time",
    }
    return mapping.get(str(name), str(name).replace("_", " "))


def _describe_node_inputs(inputs: list[dict[str, Any]], *, level: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    node_inputs = _node_inputs(inputs)
    num_nodes = _infer_num_nodes(inputs)
    if not node_inputs:
        return out

    if level <= 1:
        out.append("Indexed records:")
        for idx in range(num_nodes):
            cells = [f"id {idx}"]
            for dp in node_inputs:
                name = str(dp.get("name"))
                arr = np.asarray(dp.get("data"))
                typ = str(dp.get("type"))
                if typ == "mask_one":
                    continue
                if typ == "categorical":
                    value = _categorical_ids(arr)[idx]
                elif typ == "mask":
                    value = int(float(arr.reshape(-1)[idx]) > 0.5)
                else:
                    raw = arr[idx]
                    value = raw.tolist() if getattr(raw, "ndim", 0) else _as_number(raw)
                cells.append(f"{_human_label(name)}={_j(value)}")
            out.append("; ".join(cells))
    else:
        for dp in node_inputs:
            name = str(dp.get("name"))
            arr = np.asarray(dp.get("data"))
            typ = str(dp.get("type"))
            if typ == "mask_one":
                out.append(f"The chosen starting index is {_one_hot_index(arr)}.")
            elif typ == "mask":
                out.append(f"Active item ids for {_human_label(name)}: {_j(_active_indices(arr))}.")
            elif typ == "categorical":
                out.append(f"Symbol/category ids by position: {_j(_categorical_ids(arr))}.")
            else:
                out.append(f"{_human_label(name).capitalize()} by index: {_j(arr.tolist())}.")

    source = next((dp for dp in node_inputs if str(dp.get("type")) == "mask_one"), None)
    if source is not None and level <= 1:
        out.append(f"Start/source index: {_one_hot_index(np.asarray(source.get('data')))}.")
    if level >= 3 and node_inputs:
        rng.shuffle(out)
    return out


def _describe_edges(inputs: list[dict[str, Any]], *, level: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    edges = _edge_inputs(inputs)
    if not edges:
        return out
    mask = _find_edge_mask(inputs)
    scalar_edges = [dp for dp in edges if str(dp.get("type")) == "scalar"]
    mask_edges = [dp for dp in edges if str(dp.get("type")) == "mask"]
    if scalar_edges:
        triples = _edge_triples(np.asarray(scalar_edges[0].get("data")), mask=mask)
        if level == 0:
            out.append("Directed connections as source -> destination (value):")
            out.extend(f"{src} -> {dst} ({val})" for src, dst, val in triples)
        elif level <= 2:
            out.append("Weighted/valued arcs, unmentioned ordered pairs are absent:")
            out.append(_j(triples))
        else:
            shuffled = list(triples)
            rng.shuffle(shuffled)
            out.append("link dump [from,to,val], unordered and noisy:")
            out.append(_j(shuffled))
    elif mask_edges:
        pairs = _active_pairs(np.asarray(mask_edges[0].get("data")))
        if level <= 1:
            out.append("Directed connections:")
            out.append(_j(pairs))
        else:
            rng.shuffle(pairs)
            out.append("links:")
            out.extend(f"{src}>{dst}" for src, dst in pairs)
    return out


def _describe_graph_scalars(inputs: list[dict[str, Any]]) -> list[str]:
    out = []
    for dp in _graph_inputs(inputs):
        arr = np.asarray(dp.get("data"))
        value = arr.tolist() if arr.ndim > 0 else _as_number(arr)
        out.append(f"{_human_label(str(dp.get('name')))}: {_j(value)}.")
    return out


def _preamble(inputs: list[dict[str, Any]], *, level: int, rng: random.Random, ood: bool) -> list[str]:
    num_nodes = _infer_num_nodes(inputs)
    graphish = _looks_like_graph(inputs)
    if ood:
        return [
            "Field note from a separate data-entry system.",
            f"There are {num_nodes} numbered items, using ids 0 through {num_nodes - 1}.",
            "Infer the native tensor inputs expected by the downstream CLRS component.",
        ]
    if level == 0:
        noun = "vertices" if graphish else "items"
        return [
            f"Clean instance description with {num_nodes} numbered {noun}, indexed from 0.",
            "Convert the described instance into the native input tensors used downstream.",
        ]
    if level == 1:
        return [
            f"The case contains {num_nodes} zero-indexed records.",
            "The downstream solver is already selected elsewhere; only reconstruct its inputs.",
        ]
    if level == 2:
        return [
            f"Transfer sheet: count={num_nodes}; ids are zero based.",
            "Sections may be in a different order from the target tensors.",
        ]
    if level == 3:
        return [
            f"scratch notes: n={num_nodes}; ids start at zero",
            "ignore outputs/traces; just recover the instance data",
        ]
    return [
        rng.choice(["ocr-ish handoff", "mixed human note", "copied support ticket"]),
        f"count maybe {num_nodes}; item labels are 0..{num_nodes - 1}",
        "downstream routine is not named here; infer only the input object",
    ]


def _build_question(row: dict[str, Any], *, level: int, rng: random.Random, ood: bool = False) -> str:
    inputs = [dict(dp) for dp in row.get("inputs", [])]
    parts = _preamble(inputs, level=level, rng=rng, ood=ood)
    node_lines = _describe_node_inputs(inputs, level=level, rng=rng)
    edge_lines = _describe_edges(inputs, level=level, rng=rng)
    graph_lines = _describe_graph_scalars(inputs)
    sections = [node_lines, edge_lines, graph_lines]
    if level >= 2 or ood:
        rng.shuffle(sections)
    for section in sections:
        parts.extend(section)
    if level >= 3:
        noise = [
            "note: repeated self-links, if listed, are real entries",
            "correction: use the latest start/source value if one is stated",
            "unmentioned ordered pairs should not be treated as links",
            "the index order is the order shown above",
        ]
        rng.shuffle(noise)
        parts.extend(noise[:2 if level == 3 else 3])
    if ood:
        parts.append("No algorithm label is available in this note.")
    return "\n".join(parts).strip() + "\n"


def _make_row(
    row: dict[str, Any],
    *,
    split: str,
    level: int,
    sample_index: int,
    variant_idx: int,
    rng: random.Random,
    ood: bool = False,
) -> dict[str, Any]:
    style = "heldout_ood_no_algo" if ood else LEVEL_NAMES[level]
    sample_id = (
        f"no-algo:{split}:level_{level}:{row.get('algorithm')}:{row.get('requested_length')}:"
        f"{row.get('sample_index', sample_index)}:{variant_idx}:{style}"
    )
    target = _native_target(row, sample_id=sample_id, split=split, sample_index=sample_index)
    num_nodes = int(target.get("num_nodes", 0))
    return {
        "sample_id": sample_id,
        "source_split": split,
        "algo_name": row.get("algorithm"),
        "algorithm": row.get("algorithm"),
        "bridge_task_label": "unknown",
        "length": int(row.get("requested_length", num_nodes)),
        "num_nodes": num_nodes,
        "question": _build_question(row, level=level, rng=rng, ood=ood),
        "answer": "",
        "nl_style": style,
        "curriculum_level": int(level),
        "question_hides_algorithm": True,
        "question_includes_native_schema": False,
        "native_input_target": target,
        "canonical_inputs": {
            "algorithm": row.get("algorithm"),
            "inputs": {},
            "num_nodes": num_nodes,
            "note": "No-algorithm-name curriculum row; target is native_input_target.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-algorithm-name bridge curriculum")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--levels", default="0,1,2,3,4")
    parser.add_argument("--variants-per-row", type=int, default=1)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL file not found: {input_jsonl}")
    levels = [int(x) for x in str(args.levels).split(",") if str(x).strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "levels": {str(k): v for k, v in LEVEL_NAMES.items()},
        "variants_per_row": int(args.variants_per_row),
        "bridge_task_label": "unknown",
        "question_hides_algorithm": True,
        "question_includes_native_schema": False,
        "splits": {},
    }

    for level in levels:
        level_dir = output_dir / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        manifest["splits"][f"level_{level}"] = {}
        for split in ("train", "val", "test"):
            manifest["splits"][f"level_{level}"][split] = {
                "path": str(level_dir / f"{split}.jsonl"),
                "rows": 0,
                "tasks": {},
            }

    ood_dir = output_dir / "ood"
    ood_dir.mkdir(parents=True, exist_ok=True)
    ood_path = ood_dir / "test_ood_size.jsonl"
    counters: dict[tuple[int, str], int] = {(level, split): 0 for level in levels for split in ("train", "val", "test")}
    tasks: dict[tuple[int, str], Counter[str]] = {(level, split): Counter() for level in levels for split in ("train", "val", "test")}
    seen_by_split: Counter[str] = Counter()
    ood_count = 0
    ood_tasks: Counter[str] = Counter()
    max_rows = int(args.max_rows_per_split)
    variants_per_row = max(1, int(args.variants_per_row))

    with ExitStack() as stack:
        level_handles = {
            (level, split): stack.enter_context((output_dir / f"level_{level}" / f"{split}.jsonl").open("w", encoding="utf-8"))
            for level in levels
            for split in ("train", "val", "test")
        }
        ood_handle = stack.enter_context(ood_path.open("w", encoding="utf-8"))
        with input_jsonl.open("r", encoding="utf-8") as input_handle:
            for line in input_handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                split = str(row.get("split", "train"))
                row_idx = seen_by_split[split]
                seen_by_split[split] += 1
                if max_rows > 0 and row_idx >= max_rows:
                    continue
                if split in {"train", "val", "test"}:
                    for level in levels:
                        handle = level_handles[(level, split)]
                        for variant_idx in range(variants_per_row):
                            rng = random.Random(int(args.seed) + level * 100_003 + row_idx * 1009 + variant_idx * 9176 + sum(ord(c) for c in split))
                            out_row = _make_row(
                                row,
                                split=split,
                                level=level,
                                sample_index=row_idx,
                                variant_idx=variant_idx,
                                rng=rng,
                            )
                            handle.write(json.dumps(out_row, ensure_ascii=True) + "\n")
                            counters[(level, split)] += 1
                            tasks[(level, split)][str(out_row.get("algo_name", ""))] += 1
                elif split == "test_ood_size":
                    rng = random.Random(int(args.seed) + 9_999_991 + row_idx * 1009)
                    out_row = _make_row(
                        row,
                        split="test_ood_size",
                        level=4,
                        sample_index=row_idx,
                        variant_idx=0,
                        rng=rng,
                        ood=True,
                    )
                    ood_handle.write(json.dumps(out_row, ensure_ascii=True) + "\n")
                    ood_count += 1
                    ood_tasks[str(out_row.get("algo_name", ""))] += 1

    for level in levels:
        for split in ("train", "val", "test"):
            count = counters[(level, split)]
            path = output_dir / f"level_{level}" / f"{split}.jsonl"
            manifest["splits"][f"level_{level}"][split]["rows"] = count
            manifest["splits"][f"level_{level}"][split]["tasks"] = dict(sorted(tasks[(level, split)].items()))
            print(f"wrote level={level} split={split} rows={count} path={path}", flush=True)

    manifest["ood"] = {
        "path": str(ood_path),
        "rows": ood_count,
        "tasks": dict(sorted(ood_tasks.items())),
        "style": "heldout_ood_no_algo",
    }
    print(f"wrote ood split=test_ood_size rows={ood_count} path={ood_path}", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
