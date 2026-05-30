"""
Generate Sudoku NL descriptions for the same puzzles across multiple models.

This runs models sequentially for each puzzle index (row):
for puzzle i -> model 1, model 2, ... model N.

Example:
        python code/initial/synthesize/llm/generate_sudoku_nl_multi_model.py \
      --num-puzzles 20 \
      --models qwen/qwen3-next-80b-a3b-instruct mistralai/mistral-large-3-675b-instruct-2512 qwen/qwen3-coder-480b-a35b-instruct \
            --output data/initial/sudoku_synthetic/llm/sudoku_nl_multi_model.json
"""

import argparse
import json
import os
import time
from datetime import datetime

from openai import OpenAI

from generate_sudoku_nl import (
    _print_env_debug,
    generate_nl_description,
    load_sudoku_puzzles,
)


def _log(message: str):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate same Sudoku tasks across multiple models")
    parser.add_argument("--num-puzzles", type=int, default=10, help="Number of puzzles to process")
    parser.add_argument("--input", type=str, default="data/initial/sudoku/test/sudoku_raw_test_n1000_seed42_round1.json", help="Path to raw Sudoku JSON")
    parser.add_argument("--output", type=str, default="data/initial/sudoku_synthetic/llm/sudoku_nl_multi_model.json", help="Output JSON path")
    parser.add_argument("--models", nargs="+", required=True, help="Space-separated model IDs")
    parser.add_argument("--api-key", type=str, default=None, help="NVIDIA API key (or set NVIDIA_API_KEY env)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for puzzle selection")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between model calls (seconds)")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per model/puzzle pair")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--debug-env", action="store_true", help="Print .env and API-key resolution debug info")
    args = parser.parse_args()

    if args.debug_env:
        _print_env_debug(args.api_key)

    _log("[startup] multi-model Sudoku generation")
    _log(f"[startup] time={datetime.now().isoformat(timespec='seconds')}")
    _log(f"[startup] num_puzzles={args.num_puzzles}")
    _log(f"[startup] models={args.models}")
    _log(f"[startup] output={args.output}")
    _log(f"[startup] resume={args.resume}, max_retries={args.max_retries}, delay={args.delay}")

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY env variable or pass --api-key")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=60.0,
    )

    puzzles = load_sudoku_puzzles(num_puzzles=args.num_puzzles, seed=args.seed, local_path=args.input)
    _log(f"Loaded {len(puzzles)} puzzles.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = []
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            results = json.load(f)
        for rec in results:
            done_keys.add((rec.get("index"), rec.get("model")))
        _log(f"Loaded {len(results)} existing records from {args.output}.")

    total_jobs = len(puzzles) * len(args.models)
    remaining_jobs = total_jobs - len(done_keys)
    _log(f"[plan] total_jobs={total_jobs}, remaining_jobs={remaining_jobs}, already_done={len(done_keys)}")

    for i, puzzle in enumerate(puzzles):
        _log(f"[puzzle] index={i} starting ({i + 1}/{len(puzzles)})")
        for model in args.models:
            key = (i, model)
            if key in done_keys:
                _log(f"[skip] index={i} model={model} already done")
                continue

            last_error = None
            nl_description = None
            for attempt in range(args.max_retries + 1):
                attempt_num = attempt + 1
                _log(f"[call] index={i} model={model} attempt={attempt_num}/{args.max_retries + 1} start")
                started = time.time()
                try:
                    nl_description = generate_nl_description(client, puzzle["puzzle"], model)
                    elapsed = time.time() - started
                    _log(f"[call] index={i} model={model} success in {elapsed:.2f}s")
                    break
                except Exception as e:
                    last_error = e
                    elapsed = time.time() - started
                    _log(f"[call] index={i} model={model} failed in {elapsed:.2f}s error={e}")
                    if attempt < args.max_retries:
                        backoff = 1.5 * (attempt + 1)
                        _log(f"[retry] sleeping {backoff:.1f}s before next attempt")
                        time.sleep(backoff)

            if nl_description is None:
                nl_description = f"ERROR: {str(last_error)}"

            results.append(
                {
                    "index": i,
                    "model": model,
                    "puzzle": puzzle["puzzle"],
                    "solution": puzzle["solution"],
                    "rating": puzzle["rating"],
                    "nl_description": nl_description,
                }
            )

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

            _log(f"[done] index={i} model={model} saved_records={len(results)}")
            time.sleep(args.delay)

    _log(f"[finish] Done. Saved {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
