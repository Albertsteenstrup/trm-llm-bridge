"""
download_sudoku_raw.py — Download raw Sudoku puzzles from sapientinc/sudoku-extreme

This downloads the same dataset that TRM's build_sudoku_dataset.py uses.
Saves the raw puzzle/solution pairs as a simple JSON for inspection.

Usage:
    python code/initial/download_sudoku_raw.py --split test --num-puzzles 1000 --output data/initial/sudoku/test/sudoku_raw_test_n1000_seed42_round1.json
    python code/initial/download_sudoku_raw.py --split test --num-puzzles 1000 --round-id 1
    python code/initial/download_sudoku_raw.py --split train --num-puzzles 10000 --new-subset --exclude-path data/initial/sudoku_grid --exclude-path data/initial/sudoku_synthetic/rule --exclude-path data/initial/sudoku_synthetic/llm --exclude-by-index --output data/initial/sudoku_grid/sudoku_raw_10000_unused_v2.json
"""

import csv
import json
import argparse
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download


DEFAULT_OUTPUT_DIR = "data/initial/sudoku/test"
DEFAULT_NEW_SUBSET_DIR = "data/initial/sudoku_synthetic/rule/test_pipeline"


def _iter_json_files(root: Path):
    if root.is_file() and root.suffix.lower() == ".json":
        yield root
        return
    if root.is_dir():
        for path in sorted(root.rglob("*.json")):
            yield path


def _load_existing_puzzles_and_indices(exclude_paths: list[str]) -> tuple[set[str], set[int], list[str]]:
    existing_puzzles = set()
    existing_indices = set()
    used_paths = []

    for exclusion_path in exclude_paths:
        base = Path(exclusion_path)
        if not base.exists():
            continue
        used_paths.append(str(base))

        for path in _iter_json_files(base):
            if path.name.endswith(".meta.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    selected_indices = data.get("selected_indices")
                    if isinstance(selected_indices, list):
                        for idx in selected_indices:
                            if isinstance(idx, int):
                                existing_indices.add(idx)
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                puzzle = row.get("puzzle")
                if isinstance(puzzle, str) and len(puzzle) == 81:
                    existing_puzzles.add(puzzle)
                idx = row.get("index")
                if isinstance(idx, int):
                    existing_indices.add(idx)

    return existing_puzzles, existing_indices, used_paths


def _next_available_path(output_dir: str, base_name: str) -> str:
    output_path = Path(output_dir) / f"{base_name}.json"
    if not output_path.exists():
        return str(output_path)

    suffix = 2
    while True:
        candidate = Path(output_dir) / f"{base_name}_v{suffix}.json"
        if not candidate.exists():
            return str(candidate)
        suffix += 1


def main():
    parser = argparse.ArgumentParser(description="Download raw Sudoku-Extreme puzzles")
    parser.add_argument("--num-puzzles", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path. If omitted, uses --output-dir and round naming.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder used when --output is omitted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--round-id", type=int, default=1, help="1-based non-overlapping round number for deterministic sampling")
    parser.add_argument(
        "--new-subset",
        action="store_true",
        help=(
            "Select puzzles that are not already present in --existing-dir. "
            "If --output is omitted and --output-dir is left as default, writes to --existing-dir."
        ),
    )
    parser.add_argument(
        "--existing-dir",
        action="append",
        default=None,
        help=(
            "Directory used for de-duplication in --new-subset mode. "
            "Can be provided multiple times. Defaults to data/initial/sudoku_synthetic/rule/test_pipeline."
        ),
    )
    parser.add_argument(
        "--existing-file",
        action="append",
        default=[],
        help="Specific JSON file to include in de-duplication set. Can be provided multiple times.",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=[],
        help=(
            "Unified exclusion input (file or folder, recursively traversed). "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--exclude-by-index",
        action="store_true",
        help=(
            "Also exclude candidate rows by source index if indices are found in exclusion files "
            "(from record 'index' fields or *.meta.json selected_indices)."
        ),
    )
    args = parser.parse_args()

    if args.round_id < 1:
        raise ValueError("--round-id must be >= 1")

    rng = np.random.default_rng(args.seed)

    print(f"Downloading {args.split}.csv from sapientinc/sudoku-extreme...")
    csv_path = hf_hub_download("sapientinc/sudoku-extreme", f"{args.split}.csv", repo_type="dataset")

    puzzles = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header: source, puzzle, solution, rating
        for source, puzzle_str, solution_str, rating in reader:
            puzzles.append({
                "puzzle": puzzle_str,
                "solution": solution_str,
                "rating": int(rating),
            })

    print(f"Total puzzles in {args.split}: {len(puzzles)}")

    candidate_indices = list(range(len(puzzles)))
    existing_puzzles = set()
    existing_indices = set()
    existing_sources = []
    if args.new_subset:
        existing_dirs = args.existing_dir if args.existing_dir else [DEFAULT_NEW_SUBSET_DIR]
        exclusion_sources = list(existing_dirs) + list(args.existing_file) + list(args.exclude_path)
        existing_puzzles, existing_indices, used_sources = _load_existing_puzzles_and_indices(exclusion_sources)
        existing_sources = used_sources

        candidate_indices = []
        for idx, row in enumerate(puzzles):
            if row["puzzle"] in existing_puzzles:
                continue
            if args.exclude_by_index and idx in existing_indices:
                continue
            candidate_indices.append(idx)

        print(f"Found {len(existing_puzzles)} existing unique puzzles across {len(existing_sources)} source(s)")
        if args.exclude_by_index:
            print(f"Found {len(existing_indices)} existing source indices across exclusion sources")
        print(f"Available new puzzles in {args.split}: {len(candidate_indices)}")

    selected_indices = candidate_indices
    if args.num_puzzles < len(candidate_indices):
        permutation = rng.permutation(candidate_indices)
        start = (args.round_id - 1) * args.num_puzzles
        end = start + args.num_puzzles

        if end > len(candidate_indices):
            max_rounds = len(candidate_indices) // args.num_puzzles
            raise ValueError(
                f"Requested round {args.round_id} with {args.num_puzzles} puzzles exceeds available size {len(candidate_indices)}. "
                f"Max full rounds for this setup: {max_rounds}."
            )

        selected_indices = sorted(permutation[start:end].tolist())
    elif args.num_puzzles > len(candidate_indices):
        raise ValueError(
            f"Requested {args.num_puzzles} puzzles but only {len(candidate_indices)} are available "
            f"after filtering existing downloads."
        )

    puzzles = [puzzles[i] for i in selected_indices]

    if args.output:
        output_path = args.output
    else:
        output_dir = args.output_dir
        if args.new_subset and args.output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = (args.existing_dir[0] if args.existing_dir else DEFAULT_NEW_SUBSET_DIR)

        base_name = f"sudoku_raw_{args.split}_n{len(puzzles)}_seed{args.seed}_round{args.round_id}"
        if args.new_subset:
            base_name = f"{base_name}_newsubset"
        output_path = _next_available_path(output_dir.rstrip("/"), base_name)

    # Save
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(puzzles, f, indent=2)

    meta_path = output_path.replace(".json", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "repo": "sapientinc/sudoku-extreme",
                "split": args.split,
                "seed": args.seed,
                "round_id": args.round_id,
                "num_puzzles": len(puzzles),
                "selected_indices": selected_indices,
                "new_subset": args.new_subset,
                "existing_sources": existing_sources if args.new_subset else None,
                "existing_unique_puzzles": len(existing_puzzles) if args.new_subset else None,
                "exclude_by_index": args.exclude_by_index if args.new_subset else None,
                "existing_unique_indices": len(existing_indices) if (args.new_subset and args.exclude_by_index) else None,
            },
            f,
            indent=2,
        )

    print(f"Saved {len(puzzles)} puzzles to {output_path}")
    print(f"Saved metadata to {meta_path}")

    # Show sample
    sample = puzzles[0]
    print(f"\nSample puzzle: {sample['puzzle'][:27]}... (rating: {sample['rating']})")
    grid = sample["puzzle"].replace(".", "0")
    for r in range(9):
        print("  " + " ".join(grid[r*9:(r+1)*9]))


if __name__ == "__main__":
    main()
