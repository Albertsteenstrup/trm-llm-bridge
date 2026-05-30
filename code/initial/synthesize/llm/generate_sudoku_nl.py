"""
generate_sudoku_nl.py — Generate synthetic natural language Sudoku descriptions
using NVIDIA Cloud API .

This script:
1. Downloads 1000 Sudoku puzzles from sapientinc/sudoku-extreme (same dataset TRM trains on)
2. Sends each puzzle to NVIDIA's Qwen 3 coder to generate a natural language description
3. Saves the (puzzle_string, solution_string, nl_description) triples to JSON

Usage:
    export NVIDIA_API_KEY="nvapi-..."
    python code/initial/synthesize/llm/generate_sudoku_nl.py --num-puzzles 1000 --output data/initial/sudoku_synthetic/llm/sudoku_nl_dataset.json

Requires: openai, huggingface_hub, tqdm
The NVIDIA API uses OpenAI-compatible endpoints.

Evaluation mode:
    python code/initial/synthesize/llm/generate_sudoku_nl.py --evaluate-file data/initial/sudoku_synthetic/llm/sudoku_nl_dataset.json
"""

import os
import json
import csv
import time
import argparse
import re
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load NVIDIA_API_KEY from .env in project root
def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return start.parents[4]


PROJECT_ROOT = _find_repo_root(Path(__file__).resolve().parent)
DOTENV_PATH = PROJECT_ROOT / ".env"
DOTENV_LOADED = load_dotenv(dotenv_path=DOTENV_PATH, override=True)


def _mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def _print_env_debug(api_key_from_args: str | None = None):
    env_key = os.environ.get("NVIDIA_API_KEY")
    selected_key = api_key_from_args or env_key
    selected_source = "--api-key" if api_key_from_args else "NVIDIA_API_KEY env"

    print("\n[env-debug]")
    print(f"Python executable: {os.sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f".env path: {DOTENV_PATH}")
    print(f".env exists: {DOTENV_PATH.exists()}")
    print(f".env loaded via dotenv: {DOTENV_LOADED}")
    print(f"NVIDIA_API_KEY in environment: {bool(env_key)}")
    if env_key:
        print(f"NVIDIA_API_KEY (masked): {_mask_secret(env_key)}")
    print(f"Selected key source: {selected_source}")
    print(f"Selected key present: {bool(selected_key)}")
    if selected_key:
        print(f"Selected key (masked): {_mask_secret(selected_key)}")


def load_sudoku_puzzles(num_puzzles: int = 1000, seed: int = 42, local_path: str = "data/initial/sudoku/test/sudoku_raw_test_n1000_seed42_round1.json"):
    """Load raw Sudoku puzzles from local JSON file or sapientinc/sudoku-extreme."""
    import numpy as np
    np.random.seed(seed)

    if local_path and os.path.exists(local_path):
        print(f"Loading puzzles from local file: {local_path}")
        with open(local_path, "r") as f:
            puzzles = json.load(f)
    else:
        print("Downloading puzzles from sapientinc/sudoku-extreme...")
        csv_path = hf_hub_download("sapientinc/sudoku-extreme", "train.csv", repo_type="dataset")

        puzzles = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for source, puzzle_str, solution_str, rating in reader:
                puzzles.append({
                    "puzzle": puzzle_str,       # 81 chars, dots = empty
                    "solution": solution_str,   # 81 chars, digits
                    "rating": int(rating),
                })

    # Subsample
    if num_puzzles < len(puzzles):
        indices = np.random.choice(len(puzzles), size=num_puzzles, replace=False)
        puzzles = [puzzles[i] for i in indices]

    return puzzles


def format_grid(puzzle_str: str) -> str:
    """Format 81-char puzzle string into a readable 9x9 grid."""
    lines = []
    for row in range(9):
        cells = []
        for col in range(9):
            ch = puzzle_str[row * 9 + col]
            cells.append("_" if ch == "." else ch)
        lines.append(" ".join(cells))
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a Sudoku puzzle description assistant.

Goal: produce a human-readable description of Sudoku givens using conventions common in Sudoku literature:
- Row/Column phrasing (e.g., "Row 1, Column 3 contains 7")
- rNcM shorthand (e.g., "r1c3=7")

Output format (strict):
1) One short sentence introducing that these are givens.
2) A section title exactly: "Givens by row:"
3) One line per non-empty row, in ascending order:
   "Row <r>: r<r>c<c>=<v>, r<r>c<c>=<v>, ..."

Rules:
- Mention every pre-filled cell exactly once.
- Do not mention empty cells.
- Do NOT solve the puzzle or discuss strategies.
- Use only information present in the provided grid."""


ROW_COL_VALUE_PATTERN = re.compile(
    r"row\s*([1-9])\s*[,;]?\s*column\s*([1-9])\s*(?:contains|has|is|=|:)\s*([1-9])",
    flags=re.IGNORECASE,
)
RNCM_VALUE_PATTERN = re.compile(
    r"r\s*([1-9])\s*c\s*([1-9])\s*(?:=|is|contains|has|:)\s*([1-9])",
    flags=re.IGNORECASE,
)
ROW_LINE_PATTERN = re.compile(r"row\s*([1-9])\s*:\s*([^\n]+)", flags=re.IGNORECASE)
C_IN_ROW_PATTERN = re.compile(r"c\s*([1-9])\s*(?:=|is|contains|has|:)\s*([1-9])", flags=re.IGNORECASE)
ROW_MENTION_PATTERN = re.compile(r"row\s*([1-9])", flags=re.IGNORECASE)


def _is_valid_puzzle_string(puzzle_str: str) -> bool:
    return isinstance(puzzle_str, str) and len(puzzle_str) == 81 and all(ch == "." or ch.isdigit() for ch in puzzle_str)


def _is_valid_solution_string(solution_str: str) -> bool:
    return isinstance(solution_str, str) and len(solution_str) == 81 and all(ch in "123456789" for ch in solution_str)


def _given_triples_from_puzzle(puzzle_str: str):
    triples = set()
    for idx, ch in enumerate(puzzle_str):
        if ch != ".":
            row = idx // 9 + 1
            col = idx % 9 + 1
            triples.add((row, col, ch))
    return triples


def _extract_triples_from_description(text: str):
    triples = set()
    if not isinstance(text, str):
        return triples

    for row_s, col_s, val_s in ROW_COL_VALUE_PATTERN.findall(text):
        triples.add((int(row_s), int(col_s), val_s))

    for row_s, col_s, val_s in RNCM_VALUE_PATTERN.findall(text):
        triples.add((int(row_s), int(col_s), val_s))

    for row_s, row_body in ROW_LINE_PATTERN.findall(text):
        row = int(row_s)
        for col_s, val_s in C_IN_ROW_PATTERN.findall(row_body):
            triples.add((row, int(col_s), val_s))

    return triples


def _extract_row_mentions(text: str, triples):
    rows = {row for row, _, _ in triples}
    if isinstance(text, str):
        rows.update(int(x) for x in ROW_MENTION_PATTERN.findall(text))
    return rows


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _std(values):
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((x - mean_value) ** 2 for x in values) / float(len(values))
    return variance ** 0.5


def _normalized_hist(values, min_value, max_value):
    counter = Counter(values)
    total = sum(counter.values())
    if total == 0:
        return {x: 0.0 for x in range(min_value, max_value + 1)}
    return {x: counter.get(x, 0) / total for x in range(min_value, max_value + 1)}


def _hist_l1_distance(hist_a, hist_b):
    keys = set(hist_a.keys()) | set(hist_b.keys())
    return float(sum(abs(hist_a.get(k, 0.0) - hist_b.get(k, 0.0)) for k in keys))


def _append_eval_log_json(log_path: str, entry: dict):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    payload = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    payload = existing
        except (json.JSONDecodeError, ValueError, OSError):
            payload = []

    payload.append(entry)
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)


def evaluate_generated_dataset(records, reference_puzzles=None):
    total = len(records)
    schema_valid = 0
    generation_errors = 0

    aggregate_tp = 0
    aggregate_fp = 0
    aggregate_fn = 0
    exact_match_count = 0
    missing_entry_count = 0
    hallucination_entry_count = 0
    row_grouping_ok_count = 0

    clue_counts = []
    ratings = []

    for rec in records:
        puzzle = rec.get("puzzle")
        solution = rec.get("solution")
        rating = rec.get("rating")
        description = rec.get("nl_description", "")

        required_fields_ok = all(k in rec for k in ["puzzle", "solution", "rating", "nl_description"])
        puzzle_ok = _is_valid_puzzle_string(puzzle)
        solution_ok = _is_valid_solution_string(solution)
        rating_ok = isinstance(rating, int)

        if not (required_fields_ok and puzzle_ok and solution_ok and rating_ok):
            continue

        schema_valid += 1
        clue_counts.append(sum(1 for ch in puzzle if ch != "."))
        ratings.append(rating)

        if isinstance(description, str) and description.startswith("ERROR:"):
            generation_errors += 1
            continue

        gold = _given_triples_from_puzzle(puzzle)
        pred = _extract_triples_from_description(description)

        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)

        aggregate_tp += tp
        aggregate_fp += fp
        aggregate_fn += fn

        if fp > 0:
            hallucination_entry_count += 1
        if fn > 0:
            missing_entry_count += 1
        if pred == gold:
            exact_match_count += 1

        mentioned_rows = _extract_row_mentions(description, pred)
        gold_rows = {row for row, _, _ in gold}
        if gold_rows.issubset(mentioned_rows):
            row_grouping_ok_count += 1

    precision = aggregate_tp / (aggregate_tp + aggregate_fp) if (aggregate_tp + aggregate_fp) else 0.0
    recall = aggregate_tp / (aggregate_tp + aggregate_fn) if (aggregate_tp + aggregate_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    evaluated_entries = max(1, schema_valid - generation_errors)
    metrics = {
        "total_entries": total,
        "schema_valid_entries": schema_valid,
        "generation_error_entries": generation_errors,
        "schema_valid_rate": (schema_valid / total) if total else 0.0,
        "grounding_precision": precision,
        "grounding_recall": recall,
        "grounding_f1": f1,
        "exact_match_rate": exact_match_count / evaluated_entries,
        "missing_entry_rate": missing_entry_count / evaluated_entries,
        "hallucination_entry_rate": hallucination_entry_count / evaluated_entries,
        "row_grouping_adherence_rate": row_grouping_ok_count / evaluated_entries,
        "clue_count_mean": _mean(clue_counts),
        "clue_count_std": _std(clue_counts),
        "rating_mean": _mean(ratings),
        "rating_std": _std(ratings),
    }

    if reference_puzzles:
        ref_clue_counts = [sum(1 for ch in p["puzzle"] if ch != ".") for p in reference_puzzles]
        ref_ratings = [int(p["rating"]) for p in reference_puzzles]

        clue_hist_pred = _normalized_hist(clue_counts, 0, 81)
        clue_hist_ref = _normalized_hist(ref_clue_counts, 0, 81)

        pred_rating_buckets = [r // 100 for r in ratings]
        ref_rating_buckets = [r // 100 for r in ref_ratings]
        min_bucket = min(pred_rating_buckets + ref_rating_buckets) if (pred_rating_buckets or ref_rating_buckets) else 0
        max_bucket = max(pred_rating_buckets + ref_rating_buckets) if (pred_rating_buckets or ref_rating_buckets) else 0
        rating_hist_pred = _normalized_hist(pred_rating_buckets, min_bucket, max_bucket)
        rating_hist_ref = _normalized_hist(ref_rating_buckets, min_bucket, max_bucket)

        metrics["reference_alignment"] = {
            "reference_size": len(reference_puzzles),
            "reference_clue_count_mean": _mean(ref_clue_counts),
            "reference_clue_count_std": _std(ref_clue_counts),
            "reference_rating_mean": _mean(ref_ratings),
            "reference_rating_std": _std(ref_ratings),
            "clue_count_hist_l1_distance": _hist_l1_distance(clue_hist_pred, clue_hist_ref),
            "rating_bucket_hist_l1_distance": _hist_l1_distance(rating_hist_pred, rating_hist_ref),
        }

    return metrics


def generate_nl_description(client: OpenAI, puzzle_str: str, model: str) -> str:
    """Generate a natural language description for a single Sudoku puzzle."""
    grid_text = format_grid(puzzle_str)

    user_prompt = f"""Describe this Sudoku puzzle in natural language. The grid uses underscores for empty cells:

{grid_text}

Use human Sudoku notation and output exactly in this style:
- brief intro sentence
- "Givens by row:"
- one line per row with givens, e.g. "Row 1: r1c3=7, r1c8=2"

Provide a complete description of all pre-filled cells."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    message = response.choices[0].message
    content = message.content

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                piece = item.get("text")
                if isinstance(piece, str) and piece.strip():
                    text_parts.append(piece.strip())
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError(f"Empty/unsupported completion content for model={model}")


class GlobalRateLimiter:
    def __init__(self, calls_per_minute: float):
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be > 0")
        self.interval_seconds = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._next_slot = time.monotonic()

    def acquire_slot(self):
        with self._lock:
            now = time.monotonic()
            slot_time = max(now, self._next_slot)
            sleep_seconds = slot_time - now
            self._next_slot = slot_time + self.interval_seconds
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def run_generation(
    *,
    puzzles: list,
    existing_results: list,
    output_path: str,
    progress_label: str,
    client: OpenAI,
    model: str,
    generate_fn: Callable[[OpenAI, str, str], str],
    concurrency: int,
    calls_per_minute: float | None,
    delay: float,
    save_every: int = 10,
    max_retries: int = 2,
    log_path: str | None = None,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    results_by_index = {int(row["index"]): row for row in existing_results if "index" in row}
    
    pending_indices = []
    for i in range(len(puzzles)):
        if i not in results_by_index:
            pending_indices.append(i)
        else:
            desc = results_by_index[i].get("nl_description", "")
            if isinstance(desc, str) and desc.startswith("ERROR:"):
                pending_indices.append(i)

    if not pending_indices:
        print("No pending or failed indices to process.")
        ordered = [results_by_index[i] for i in sorted(results_by_index.keys())]
        with open(output_path, "w") as f:
            json.dump(ordered, f, indent=2)
        return ordered

    print(f"Indices to process: {len(pending_indices)} (out of total {len(puzzles)})")

    limiter = None
    if calls_per_minute and calls_per_minute > 0:
        limiter = GlobalRateLimiter(calls_per_minute)
    elif delay > 0:
        limiter = GlobalRateLimiter(60.0 / delay)

    def _build_record(i: int):
        puzzle = puzzles[i]
        thread_id = threading.get_ident()
        
        last_error = None
        for attempt in range(max_retries + 1):
            if limiter is not None:
                limiter.acquire_slot()

            try:
                # print(f"[debug] Worker {thread_id} calling API for puzzle {i} (attempt {attempt+1})...")
                nl_description = generate_fn(client, puzzle["puzzle"], model)
                return {
                    "index": i,
                    "puzzle": puzzle["puzzle"],
                    "solution": puzzle["solution"],
                    "rating": puzzle["rating"],
                    "nl_description": nl_description,
                }
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff

        return {
            "index": i,
            "puzzle": puzzle["puzzle"],
            "solution": puzzle["solution"],
            "rating": puzzle["rating"],
            "nl_description": f"ERROR: {str(last_error)}",
        }

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_build_record, i): i for i in pending_indices}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc=progress_label):
                record = future.result()
                results_by_index[record["index"]] = record
                completed_since_save += 1

                if completed_since_save >= save_every:
                    ordered = [results_by_index[i] for i in sorted(results_by_index.keys())]
                    with open(output_path, "w") as f:
                        json.dump(ordered, f, indent=2)
                    
                    if log_path:
                        current_report = evaluate_generated_dataset(ordered, reference_puzzles=puzzles)
                        log_entry = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "progress": f"{len(ordered)}/{len(puzzles)}",
                            "metrics": current_report
                        }
                        _append_eval_log_json(log_path, log_entry)

                    completed_since_save = 0
        except KeyboardInterrupt:
            ordered = [results_by_index[i] for i in sorted(results_by_index.keys())]
            with open(output_path, "w") as f:
                json.dump(ordered, f, indent=2)
            print(f"\nInterrupted. Saved {len(ordered)} records to {output_path}.")
            raise

    ordered = [results_by_index[i] for i in sorted(results_by_index.keys())]
    with open(output_path, "w") as f:
        json.dump(ordered, f, indent=2)

    if log_path:
        final_report = evaluate_generated_dataset(ordered, reference_puzzles=puzzles)
        final_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "progress": f"{len(ordered)}/{len(puzzles)}",
            "final": True,
            "metrics": final_report,
        }
        _append_eval_log_json(log_path, final_entry)

    return ordered


def main():
    parser = argparse.ArgumentParser(description="Generate NL Sudoku descriptions via NVIDIA API")
    parser.add_argument("--num-puzzles", type=int, default=1000, help="Number of puzzles to process")
    parser.add_argument("--input", type=str, default="data/initial/sudoku/test/sudoku_raw_test_n1000_seed42_round1.json", help="Path to raw Sudoku JSON")
    parser.add_argument("--output", type=str, default="data/initial/sudoku_synthetic/llm/sudoku_nl_dataset.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="qwen/qwen3-coder-480b-a35b-instruct", help="NVIDIA model ID")
    parser.add_argument("--api-key", type=str, default=None, help="NVIDIA API key (or set NVIDIA_API_KEY env)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for puzzle selection")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls (seconds)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel worker threads")
    parser.add_argument("--calls-per-minute", type=float, default=40.0, help="Global API call budget per minute across all workers")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint output and evaluate full set every N completed records")
    parser.add_argument("--log-file", type=str, default="logs/initial/generation_eval_log.json", help="Path to evaluation log JSON file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--evaluate-file", type=str, default=None, help="Evaluate an existing generated JSON dataset and exit")
    parser.add_argument("--eval-report", type=str, default=None, help="Optional path to write evaluation report JSON")
    parser.add_argument("--eval-no-reference", action="store_true", help="Skip reference alignment against sapientinc/sudoku-extreme")
    parser.add_argument("--debug-env", action="store_true", help="Print .env and API-key resolution debug info")
    args = parser.parse_args()

    if args.debug_env:
        _print_env_debug(args.api_key)

    if args.evaluate_file:
        with open(args.evaluate_file, "r") as f:
            records = json.load(f)

        reference = None
        if not args.eval_no_reference:
            reference = load_sudoku_puzzles(num_puzzles=len(records), seed=args.seed)

        report = evaluate_generated_dataset(records, reference_puzzles=reference)

        print("\nEvaluation summary")
        print("------------------")
        print(f"Total entries: {report['total_entries']}")
        print(f"Schema valid rate: {report['schema_valid_rate']:.4f}")
        print(f"Generation error entries: {report['generation_error_entries']}")
        print(f"Grounding precision: {report['grounding_precision']:.4f}")
        print(f"Grounding recall: {report['grounding_recall']:.4f}")
        print(f"Grounding F1: {report['grounding_f1']:.4f}")
        print(f"Exact match rate: {report['exact_match_rate']:.4f}")
        print(f"Missing entry rate: {report['missing_entry_rate']:.4f}")
        print(f"Hallucination entry rate: {report['hallucination_entry_rate']:.4f}")
        print(f"Row grouping adherence: {report['row_grouping_adherence_rate']:.4f}")

        if "reference_alignment" in report:
            ref = report["reference_alignment"]
            print(f"Clue count hist L1 distance: {ref['clue_count_hist_l1_distance']:.4f}")
            print(f"Rating bucket hist L1 distance: {ref['rating_bucket_hist_l1_distance']:.4f}")

        if args.eval_report:
            os.makedirs(os.path.dirname(args.eval_report) or ".", exist_ok=True)
            with open(args.eval_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Evaluation report written to {args.eval_report}")

        return

    # API setup
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY env variable or pass --api-key")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=60.0,
    )

    # Load puzzles
    print(f"Loading {args.num_puzzles} Sudoku puzzles...")
    puzzles = load_sudoku_puzzles(num_puzzles=args.num_puzzles, seed=args.seed, local_path=args.input)
    print(f"Loaded {len(puzzles)} puzzles.")

    # Resume support
    results = []
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing records from {args.output}.")
        except (json.JSONDecodeError, ValueError):
            print(f"Resume requested but output file is empty/invalid JSON: {args.output}. Starting from scratch.")
            results = []

    effective_rate = args.calls_per_minute if args.calls_per_minute > 0 else (60.0 / args.delay if args.delay > 0 else 0.0)
    if effective_rate > 0:
        print(f"Running with concurrency={args.concurrency}, global_rate={effective_rate:.4f} calls/min")
    else:
        print(f"Running with concurrency={args.concurrency}, no global rate cap")

    results = run_generation(
        puzzles=puzzles,
        existing_results=results,
        output_path=args.output,
        progress_label="Generating NL descriptions",
        client=client,
        model=args.model,
        generate_fn=generate_nl_description,
        concurrency=args.concurrency,
        calls_per_minute=args.calls_per_minute,
        delay=args.delay,
        save_every=args.save_every,
        log_path=args.log_file,
    )

    print(f"\nDone! {len(results)} descriptions saved to {args.output}")

    # Print stats
    errors = sum(1 for r in results if r["nl_description"].startswith("ERROR:"))
    print(f"Successful: {len(results) - errors}, Errors: {errors}")

    # Evaluation summary
    report = evaluate_generated_dataset(results, reference_puzzles=puzzles)
    print("\nQuality metrics")
    print("---------------")
    print(f"Grounding F1: {report['grounding_f1']:.4f}")
    print(f"Exact match rate: {report['exact_match_rate']:.4f}")
    print(f"Hallucination entry rate: {report['hallucination_entry_rate']:.4f}")
    print(f"Missing entry rate: {report['missing_entry_rate']:.4f}")

    ref = report.get("reference_alignment")
    if ref:
        print(f"Clue count hist L1 distance vs source: {ref['clue_count_hist_l1_distance']:.4f}")
        print(f"Rating bucket hist L1 distance vs source: {ref['rating_bucket_hist_l1_distance']:.4f}")


if __name__ == "__main__":
    main()
