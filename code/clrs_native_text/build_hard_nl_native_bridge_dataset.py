#!/usr/bin/env python3
"""Build hard natural-language bridge data from native CLRS rows.

Rows produced here train/evaluate a bridge from messy NL descriptions to the
exact native dm-clrs input tensors consumed by the specialist. The bridge target
is stored explicitly as ``native_input_target`` so the descriptions do not have
to be parsable by the older canonical CLRS-Text parser.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


STRICT_NATIVE_ADAPTER_VERSION = "bridge_v2_canonical_native_input_v2"
DEFAULT_STYLES = ["native_audit", "node_table", "edge_list", "noisy_notes"]
DEFAULT_OOD_STYLES = ["shuffled_report", "ocr_dump", "constraint_story", "edge_list"]


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


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _infer_num_nodes(inputs: list[dict[str, Any]]) -> int:
    for dp in inputs:
        arr = np.asarray(dp.get("data"))
        location = str(dp.get("location", ""))
        if location == "node" and arr.ndim >= 1:
            return int(arr.shape[0])
        if location == "edge" and arr.ndim >= 2:
            return int(arr.shape[0])
    return 0


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _as_number(value: Any) -> int | float:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _one_hot_index(values: np.ndarray) -> int:
    flat = values.astype(float).reshape(-1)
    if flat.size == 0:
        return 0
    return int(np.argmax(flat))


def _categorical_ids(values: np.ndarray) -> list[int]:
    arr = values.astype(float)
    if arr.ndim == 1:
        return [int(round(float(v))) for v in arr.tolist()]
    return [int(v) for v in np.argmax(arr, axis=-1).reshape(-1).tolist()]


def _active_indices(values: np.ndarray) -> list[int]:
    return [int(i) for i in np.nonzero(values.astype(float).reshape(-1) > 0.5)[0].tolist()]


def _active_pairs(values: np.ndarray) -> list[list[int]]:
    arr = values.astype(float)
    if arr.ndim < 2:
        return []
    src, dst = np.nonzero(arr[: arr.shape[0], : arr.shape[1]] > 0.5)
    return [[int(i), int(j)] for i, j in zip(src.tolist(), dst.tolist())]


def _edge_triples(values: np.ndarray, *, mask: np.ndarray | None = None) -> list[list[Any]]:
    arr = values.astype(float)
    if arr.ndim < 2:
        return []
    if arr.ndim == 2:
        matrix = arr
    else:
        matrix = arr.reshape(arr.shape[0], arr.shape[1], -1)[:, :, 0]
    if mask is None:
        active = np.abs(matrix) > 1e-12
    else:
        active = mask.astype(float) > 0.5
    src, dst = np.nonzero(active)
    return [[int(i), int(j), _as_number(matrix[i, j])] for i, j in zip(src.tolist(), dst.tolist())]


def _schema_text(inputs: list[dict[str, Any]]) -> str:
    parts = [
        f"{dp.get('name')}[{dp.get('location')}/{dp.get('type')}/shape={dp.get('shape')}]"
        for dp in inputs
    ]
    return "; ".join(parts)


def _find_edge_mask(inputs: list[dict[str, Any]]) -> np.ndarray | None:
    for dp in inputs:
        if str(dp.get("location")) == "edge" and str(dp.get("type")) == "mask" and str(dp.get("name")) == "adj":
            return np.asarray(dp.get("data"), dtype=float)
    for dp in inputs:
        if str(dp.get("location")) == "edge" and str(dp.get("type")) == "mask":
            return np.asarray(dp.get("data"), dtype=float)
    return None


def _node_table(inputs: list[dict[str, Any]], *, num_nodes: int) -> list[str]:
    node_fields = [dp for dp in inputs if str(dp.get("location")) == "node" and str(dp.get("name")) != "pos"]
    if not node_fields:
        return []
    lines = ["Node records, one record per item/node:"]
    for idx in range(num_nodes):
        cells = [f"id={idx}"]
        for dp in node_fields:
            name = str(dp.get("name"))
            typ = str(dp.get("type"))
            arr = np.asarray(dp.get("data"))
            if typ == "mask_one":
                continue
            if typ == "mask":
                value = int(float(arr[idx]) > 0.5) if arr.ndim == 1 else _json_value(arr[idx].tolist())
            elif typ == "categorical":
                value = int(np.argmax(arr[idx])) if arr.ndim > 1 else int(round(float(arr[idx])))
            else:
                value = arr[idx].tolist() if getattr(arr[idx], "ndim", 0) else _as_number(arr[idx])
            cells.append(f"{name}={_json_value(value)}")
        lines.append(" | ".join(cells))
    return lines


def _describe_input(dp: dict[str, Any], *, inputs: list[dict[str, Any]], num_nodes: int, style: str) -> list[str]:
    name = str(dp.get("name"))
    location = str(dp.get("location"))
    typ = str(dp.get("type"))
    arr = np.asarray(dp.get("data"))
    if name == "pos" and location == "node":
        return [f"{name}: implicit native position vector, length {num_nodes}, value at index i is i/{num_nodes}."]

    if location == "node":
        if typ == "mask_one":
            return [f"{name}: active node index {_one_hot_index(arr)}; native target is one-hot length {num_nodes}."]
        if typ == "mask":
            return [f"{name}: active node ids {_json_value(_active_indices(arr))}; other nodes are 0."]
        if typ == "categorical":
            classes = _categorical_ids(arr)
            return [f"{name}: categorical class id by node {_json_value(classes)}; native target keeps one-hot rows if present."]
        return [f"{name}: node values in index order {_json_value(arr.tolist())}."]

    if location == "edge":
        if typ == "mask":
            return [f"{name}: active directed edge pairs {_json_value(_active_pairs(arr))}; all unlisted pairs are 0."]
        if typ == "categorical":
            return [f"{name}: dense edge categorical tensor {_json_value(arr.tolist())}."]
        mask = _find_edge_mask(inputs)
        triples = _edge_triples(arr, mask=mask)
        if style in {"native_audit", "node_table"} and arr.ndim == 2 and arr.size <= 1024:
            return [
                f"{name}: dense edge matrix follows row-major; absent/unlisted graph edges are zeros.",
                _json_value(arr.tolist()),
            ]
        return [f"{name}: directed edge values as [src,dst,value] triples {_json_value(triples)}; all unlisted pairs are 0."]

    if location == "graph":
        value = arr.tolist() if arr.ndim > 0 else _as_number(arr)
        return [f"{name}: graph-level {typ} value {_json_value(value)}."]

    return [f"{name}: {_json_value(arr.tolist())}."]


def _native_target(row: dict[str, Any], *, sample_id: str, split: str, sample_index: int) -> dict[str, Any]:
    inputs = [dict(dp) for dp in row.get("inputs", [])]
    spec = {
        str(dp.get("name")): {
            "stage": "input",
            "location": str(dp.get("location")),
            "type": str(dp.get("type")),
        }
        for dp in inputs
    }
    return {
        "sample_id": sample_id,
        "algorithm": row.get("algorithm"),
        "split": split,
        "sample_index": int(sample_index),
        "num_steps": 0,
        "source": "clrs-native-text-hard-nl-direct-target",
        "spec": spec,
        "inputs": inputs,
        "hints": [],
        "outputs": [],
        "adapter_version": STRICT_NATIVE_ADAPTER_VERSION,
        "num_nodes": _infer_num_nodes(inputs),
    }


def _build_question(row: dict[str, Any], *, style: str, rng: random.Random) -> str:
    algorithm = str(row.get("algorithm", "unknown"))
    inputs = [dict(dp) for dp in row.get("inputs", [])]
    num_nodes = _infer_num_nodes(inputs)
    schema = _schema_text(inputs)
    body_lines: list[str] = []

    if style == "native_audit":
        body_lines.append(f"Native CLRS audit note. Task={algorithm}. Number of carriers/items={num_nodes}.")
        body_lines.append(f"The specialist expects these native input slots: {schema}.")
    elif style == "node_table":
        body_lines.append(f"Algorithm {algorithm}; reconstruct the specialist's native inputs from this table-style note.")
        body_lines.append(f"Required slots, not necessarily listed in the same order: {schema}.")
        body_lines.extend(_node_table(inputs, num_nodes=num_nodes))
    elif style == "edge_list":
        body_lines.append(f"Graph/sequence transfer sheet for {algorithm}. N={num_nodes}.")
        body_lines.append(f"Output native inputs for slots: {schema}.")
    elif style == "noisy_notes":
        body_lines.append(f"scratchpad // clrs native instance // algo -> {algorithm} // n ~= {num_nodes}")
        body_lines.append(f"schema-ish: {schema}")
    elif style == "shuffled_report":
        body_lines.append(f"Report from another system: algorithm name is {algorithm}.")
        body_lines.append(f"Do not solve it; reconstruct the native input tensors. Carrier count is {num_nodes}.")
        body_lines.append(f"Expected slots: {schema}.")
    elif style == "ocr_dump":
        body_lines.append(f"OCR dump: ALGO={algorithm}; COUNT={num_nodes}; fields may be reordered.")
        body_lines.append(f"native slots -> {schema}")
    elif style == "constraint_story":
        body_lines.append(f"A downstream CLRS specialist will receive the native tensors for {algorithm}.")
        body_lines.append(f"Translate the following constraints into the native slots ({schema}).")
    else:
        body_lines.append(f"Algorithm {algorithm}, N={num_nodes}. Native slots: {schema}.")

    input_order = list(inputs)
    if style in {"shuffled_report", "ocr_dump", "noisy_notes"}:
        rng.shuffle(input_order)
    for dp in input_order:
        body_lines.extend(_describe_input(dp, inputs=inputs, num_nodes=num_nodes, style=style))

    if row.get("clrs_text", {}).get("question") and style in {"native_audit", "shuffled_report"}:
        body_lines.append("Reference CLRS-Text prompt, included only as a second view:")
        body_lines.append(str(row["clrs_text"]["question"]).strip())

    if style in {"noisy_notes", "ocr_dump"}:
        noise = [
            "NOTE: values are zero-indexed even if the prose says item.",
            "Ignore final outputs, traces, and hints; this is input translation only.",
            "If a dense adjacency slot is requested, fill absent pairs with 0.",
        ]
        rng.shuffle(noise)
        body_lines.extend(noise[:2])

    return "\n".join(body_lines).strip() + "\n"


def _make_bridge_row(
    row: dict[str, Any],
    *,
    source_split: str,
    sample_id: str,
    style: str,
    rng: random.Random,
    sample_index: int,
) -> dict[str, Any]:
    target = _native_target(row, sample_id=sample_id, split=source_split, sample_index=sample_index)
    question = _build_question(row, style=style, rng=rng)
    num_nodes = int(target.get("num_nodes", 0))
    return {
        "sample_id": sample_id,
        "source_split": source_split,
        "algo_name": row.get("algorithm"),
        "algorithm": row.get("algorithm"),
        "length": int(row.get("requested_length", num_nodes)),
        "num_nodes": num_nodes,
        "question": question,
        "answer": "",
        "nl_style": style,
        "native_input_target": target,
        "canonical_inputs": {
            "algorithm": row.get("algorithm"),
            "inputs": {},
            "num_nodes": num_nodes,
            "note": "Target is stored in native_input_target; hard NL need not be canonical-parser compatible.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard NL-to-native bridge splits from native CLRS rows")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES))
    parser.add_argument("--ood-styles", default=",".join(DEFAULT_OOD_STYLES))
    parser.add_argument("--variants-per-row", type=int, default=1)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input_jsonl))
    styles = _parse_csv(args.styles) or list(DEFAULT_STYLES)
    ood_styles = _parse_csv(args.ood_styles) or list(DEFAULT_OOD_STYLES)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("split", "train")), []).append(row)

    manifest: dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "styles": styles,
        "ood_styles": ood_styles,
        "variants_per_row": int(args.variants_per_row),
        "splits": {},
        "notes": [
            "Rows train a bridge from hard natural language to exact native dm-clrs input tensors.",
            "native_input_target is explicit; canonical_inputs is metadata only for these rows.",
            "Specialist training should use the native JSONL splits, not these NL bridge rows.",
        ],
    }

    for split, split_rows in sorted(grouped.items()):
        selected = split_rows
        if int(args.max_rows_per_split) > 0:
            selected = selected[: int(args.max_rows_per_split)]
        split_styles = ood_styles if split.startswith("test_ood") else styles
        out_rows: list[dict[str, Any]] = []
        for row_idx, row in enumerate(selected):
            for variant_idx in range(max(1, int(args.variants_per_row))):
                style = split_styles[(row_idx + variant_idx) % len(split_styles)]
                rng = random.Random(int(args.seed) + (row_idx * 1009) + (variant_idx * 9176) + sum(ord(c) for c in split))
                sample_id = (
                    f"hard-nl:{split}:{row.get('algorithm')}:{row.get('requested_length')}:"
                    f"{row.get('sample_index', row_idx)}:{variant_idx}:{style}"
                )
                out_rows.append(
                    _make_bridge_row(
                        row,
                        source_split=split,
                        sample_id=sample_id,
                        style=style,
                        rng=rng,
                        sample_index=row_idx,
                    )
                )
        path = output_dir / f"{split}.jsonl"
        _write_jsonl(path, out_rows)
        manifest["splits"][split] = {
            "path": str(path),
            "native_rows": len(selected),
            "bridge_rows": len(out_rows),
            "tasks": dict(sorted(Counter(str(row.get("algo_name", "")) for row in out_rows).items())),
            "styles": dict(sorted(Counter(str(row.get("nl_style", "")) for row in out_rows).items())),
        }
        print(f"wrote split={split} bridge_rows={len(out_rows)} path={path}", flush=True)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
