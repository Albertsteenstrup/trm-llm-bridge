#!/usr/bin/env python3
"""Generate paired CLRS native rows and official CLRS-Text prompts.

This uses the official dm-clrs sampler and the official CLRS-Text formatter
(`clrs._src.clrs_text.clrs_utils.format_clrs_example`) on the same sampled
Feedback object. The output JSONL keeps both views:

* `inputs` / `hints` / `outputs`: native dm-clrs tensors in the local raw-row
  format used by the benchmark-native specialist pipeline.
* `clrs_text.question` / `clrs_text.answer`: official CLRS-Text strings.

This is the preferred way to create a paired dataset for comparing
GPT/bridge formalizers against the same downstream CLRS specialist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CLRS_30 = [
    "activity_selector",
    "articulation_points",
    "bellman_ford",
    "bfs",
    "binary_search",
    "bridges",
    "bubble_sort",
    "dag_shortest_paths",
    "dfs",
    "dijkstra",
    "find_maximum_subarray_kadane",
    "floyd_warshall",
    "graham_scan",
    "heapsort",
    "insertion_sort",
    "jarvis_march",
    "kmp_matcher",
    "lcs_length",
    "matrix_chain_order",
    "minimum",
    "mst_kruskal",
    "mst_prim",
    "naive_string_matcher",
    "optimal_bst",
    "quickselect",
    "quicksort",
    "segments_intersect",
    "strongly_connected_components",
    "task_scheduling",
    "topological_sort",
]


def _parse_csv(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return list(CLRS_30)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_lengths(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _serialize_datapoint(dp: Any, sample_idx: int, is_hint: bool, length: int) -> dict[str, Any]:
    data = np.asarray(dp.data)
    sample = data[:length, sample_idx] if is_hint else data[sample_idx]
    return {
        "name": str(dp.name),
        "location": str(dp.location),
        "type": str(dp.type_),
        "shape": list(sample.shape),
        "data": sample.tolist(),
    }


def _spec_to_json(spec: Any) -> dict[str, dict[str, str]]:
    return {
        key: {"stage": value[0], "location": value[1], "type": value[2]}
        for key, value in dict(spec).items()
    }


def _build_sampler(clrs_module: Any, *, algorithm: str, seed: int, length: int, num_decimals: int | None) -> tuple[Any, Any]:
    kwargs = {
        "name": algorithm,
        "seed": int(seed),
        "num_samples": -1,
        "length": int(length),
    }
    if num_decimals is not None:
        kwargs["truncate_decimals"] = int(num_decimals)
    try:
        return clrs_module.build_sampler(track_max_steps=False, **kwargs)
    except TypeError:
        kwargs.pop("truncate_decimals", None)
        return clrs_module.build_sampler(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired CLRS-Text and native CLRS rows")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--algorithms", default="all")
    parser.add_argument("--lengths", default="64")
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use-hints", action="store_true")
    parser.add_argument("--num-decimals-in-float", type=int, default=-1)
    args = parser.parse_args()

    import clrs  # type: ignore
    from clrs._src.clrs_text import clrs_utils  # type: ignore

    algorithms = _parse_csv(args.algorithms)
    lengths = _parse_lengths(args.lengths)
    if not lengths:
        raise ValueError("--lengths must contain at least one integer")
    num_decimals = int(args.num_decimals_in_float)
    truncate_decimals = None if num_decimals < 0 else num_decimals

    rows: list[dict[str, Any]] = []
    for alg_idx, algorithm in enumerate(algorithms):
        for length_idx, length in enumerate(lengths):
            sampler_seed = int(args.seed) + (10_000 * length_idx) + alg_idx
            sampler, spec = _build_sampler(
                clrs,
                algorithm=algorithm,
                seed=sampler_seed,
                length=int(length),
                num_decimals=truncate_decimals,
            )
            for sample_idx in range(int(args.samples_per_task)):
                feedback = sampler.next(batch_size=1)
                hint_len = int(np.asarray(feedback.features.lengths).astype(int).tolist()[0])
                question, answer = clrs_utils.format_clrs_example(
                    algorithm,
                    feedback,
                    use_hints=bool(args.use_hints),
                )
                rows.append(
                    {
                        "source": "dm-clrs+official-clrs-text",
                        "algorithm": algorithm,
                        "split": str(args.split),
                        "sample_index": sample_idx,
                        "requested_length": int(length),
                        "num_steps": hint_len,
                        "sampler_seed": sampler_seed,
                        "spec": _spec_to_json(spec),
                        "inputs": [
                            _serialize_datapoint(dp, 0, is_hint=False, length=hint_len)
                            for dp in feedback.features.inputs
                        ],
                        "hints": [
                            _serialize_datapoint(dp, 0, is_hint=True, length=hint_len)
                            for dp in feedback.features.hints
                        ],
                        "outputs": [
                            _serialize_datapoint(dp, 0, is_hint=False, length=hint_len)
                            for dp in feedback.outputs
                        ],
                        "clrs_text": {
                            "question": question,
                            "answer": answer,
                            "use_hints": bool(args.use_hints),
                        },
                    }
                )

    output_path = Path(args.output_jsonl)
    _write_jsonl(output_path, rows)
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": "dm-clrs+official-clrs-text",
                "output_jsonl": str(output_path),
                "num_rows": len(rows),
                "algorithms": algorithms,
                "lengths": lengths,
                "samples_per_task": int(args.samples_per_task),
                "split": str(args.split),
                "seed": int(args.seed),
                "use_hints": bool(args.use_hints),
                "num_decimals_in_float": truncate_decimals,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote rows={len(rows)} output={output_path} metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
