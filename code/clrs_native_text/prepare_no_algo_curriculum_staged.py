#!/usr/bin/env python3
"""Generate raw CLRS native rows in resumable per-algorithm chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CLRS30_ALGORITHMS = [
    "articulation_points",
    "activity_selector",
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_algorithms(raw: str) -> list[str]:
    if raw.strip().lower() in {"", "all", "clrs30"}:
        return list(CLRS30_ALGORITHMS)
    algorithms = [item.strip() for item in raw.split(",") if item.strip()]
    if not algorithms:
        raise ValueError("No algorithms selected")
    return algorithms


def _allocate_evenly(total: int, n: int) -> list[int]:
    if n <= 0:
        return []
    base = int(total) // n
    rem = int(total) % n
    return [base + (1 if idx < rem else 0) for idx in range(n)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare resumable per-algorithm CLRS raw chunks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-jsonl", default="clrs_native_raw.jsonl")
    parser.add_argument("--algorithms", default="clrs30")
    parser.add_argument("--target-total", type=int, default=60000)
    parser.add_argument("--test-ood-total", type=int, default=4500)
    parser.add_argument("--train-lengths", default="8,12,16")
    parser.add_argument("--val-lengths", default="16")
    parser.add_argument("--test-lengths", default="64")
    parser.add_argument("--test-ood-lengths", default="64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--chunk-log-dir", default="")
    parser.add_argument("--stream-subprocess-logs", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = output_dir / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_log_dir = Path(args.chunk_log_dir).resolve() if args.chunk_log_dir else output_dir / "chunk_logs"
    chunk_log_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / args.output_jsonl
    algorithms = _parse_algorithms(str(args.algorithms))
    target_counts = _allocate_evenly(int(args.target_total), len(algorithms))
    ood_counts = _allocate_evenly(int(args.test_ood_total), len(algorithms))

    download_script = root / "code/trm_llm/tools/download_clrs_curriculum.py"
    chunk_paths: list[Path] = []
    chunk_meta: list[dict[str, object]] = []
    for idx, alg in enumerate(algorithms):
        alg_dir = chunk_root / alg
        alg_jsonl = alg_dir / "clrs_native_raw.jsonl"
        alg_meta = alg_dir / "metadata.json"
        chunk_paths.append(alg_jsonl)
        if alg_jsonl.exists() and alg_meta.exists() and not args.force:
            print(f"[staged] skip existing alg={alg} path={alg_jsonl}", flush=True)
        else:
            alg_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python_bin,
                "-u",
                str(download_script),
                "--output-dir",
                str(alg_dir),
                "--output-jsonl",
                alg_jsonl.name,
                "--algorithms",
                alg,
                "--target-total",
                str(target_counts[idx]),
                "--train-lengths",
                args.train_lengths,
                "--val-lengths",
                args.val_lengths,
                "--test-lengths",
                args.test_lengths,
                "--test-ood-total",
                str(ood_counts[idx]),
                "--test-ood-lengths",
                args.test_ood_lengths,
                "--seed",
                str(int(args.seed) + idx * 10_000),
                "--generation-batch-size",
                str(int(args.generation_batch_size)),
            ]
            print(f"[staged] generate alg={alg} target={target_counts[idx]} ood={ood_counts[idx]}", flush=True)
            env = os.environ.copy()
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            env.setdefault("JAX_PLATFORMS", "cpu")
            env.setdefault("CUDA_VISIBLE_DEVICES", "")
            env.setdefault("OMP_NUM_THREADS", "8")
            env.setdefault("TF_NUM_INTRAOP_THREADS", "8")
            env.setdefault("TF_NUM_INTEROP_THREADS", "2")
            chunk_log = chunk_log_dir / f"{idx + 1:02d}_{alg}.log"
            if args.stream_subprocess_logs:
                with chunk_log.open("w", encoding="utf-8") as log_handle:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log_handle.write(line)
                    rc = proc.wait()
                    if rc != 0:
                        raise subprocess.CalledProcessError(rc, cmd)
            else:
                with chunk_log.open("w", encoding="utf-8") as log_handle:
                    subprocess.run(cmd, cwd=str(root), env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=True)
                print(f"[staged] completed alg={alg} log={chunk_log}", flush=True)
        if not alg_jsonl.exists() or not alg_meta.exists():
            raise FileNotFoundError(f"Chunk did not complete for {alg}: {alg_dir}")
        chunk_meta.append(json.loads(alg_meta.read_text(encoding="utf-8")))

    total_rows = 0
    with raw_path.open("w", encoding="utf-8") as output_handle:
        for alg, chunk_path in zip(algorithms, chunk_paths, strict=True):
            rows = 0
            with chunk_path.open("r", encoding="utf-8") as chunk_handle:
                for line in chunk_handle:
                    if line.strip():
                        output_handle.write(line)
                        rows += 1
            total_rows += rows
            print(f"[staged] combined alg={alg} rows={rows} total={total_rows}", flush=True)

    metadata = {
        "source": "dm-clrs",
        "mode": "staged_per_algorithm",
        "num_rows": total_rows,
        "algorithms": algorithms,
        "target_total": int(args.target_total),
        "test_ood_total": int(args.test_ood_total),
        "train_lengths": args.train_lengths,
        "val_lengths": args.val_lengths,
        "test_lengths": args.test_lengths,
        "test_ood_lengths": args.test_ood_lengths,
        "seed": int(args.seed),
        "output_jsonl": str(raw_path),
        "chunks": chunk_meta,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[staged] wrote {total_rows} rows to {raw_path}", flush=True)


if __name__ == "__main__":
    main()
