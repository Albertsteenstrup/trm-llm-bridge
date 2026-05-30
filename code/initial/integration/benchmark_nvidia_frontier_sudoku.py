#!/usr/bin/env python3
"""
Benchmark NVIDIA Builder frontier models on the same Sudoku NL test set used by
the corrected2 initial pipeline run.

Two task modes:
  --task solve       (default) Ask the model to return the fully solved 81-digit solution.
  --task grid-parse  Ask the model to return only the initial puzzle grid (81 chars,
                     digits 1-9 for givens, '.' for empty cells). This measures
                     NL -> grid accuracy for inclusion in Tables 1 and 2.

Outputs:
- Per-example predictions JSON with resume support.
- Metrics JSON grouped by model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import wandb
except Exception:
    wandb = None


DEFAULT_INPUT_JSON = "data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json"
DEFAULT_OUTPUT_JSON = "results/frontier_sudoku_builder/frontier_predictions_corrected2.json"
DEFAULT_METRICS_JSON = "results/frontier_sudoku_builder/frontier_metrics_corrected2.json"
DEFAULT_BASELINE_METRICS_JSON = "results/test_pipeline_finetuned/pipeline_metrics_corrected2.json"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

SYSTEM_PROMPT = """You solve Sudoku from natural-language clue descriptions. /no_think

The final answer must contain exactly one JSON object with this schema:
{"solution":"<81 digits>"}

Rules:
- solution must be the fully solved Sudoku, 81 digits from 1-9.
- Put the JSON object at the end of the response.
- No markdown, no code fences, no extra keys in the JSON object.
- The string must be length 81.
- Do not show reasoning steps. Output only the JSON answer.
"""

GRID_PARSE_SYSTEM_PROMPT = """You extract the initial Sudoku grid from a natural-language description.

The description lists the given (pre-filled) cells of a Sudoku puzzle. Your task is to
reconstruct the initial puzzle grid, NOT to solve it.

The final answer must contain exactly one JSON object with this schema:
{"grid":"<81 characters>"}

Rules:
- grid must be exactly 81 characters.
- Use digits 1-9 for given cells exactly as described.
- Use '.' for every empty cell.
- Put the JSON object at the end of the response.
- No markdown, no code fences, no extra keys in the JSON object.
"""

GRID_PARSE_USER_TEMPLATE = (
    "Extract the initial Sudoku grid from the description below.\n\n"
    "Return only the given cells as digits 1-9 and empty cells as '.', "
    "exactly 81 characters total.\n"
    "End your response with exactly one JSON object: {{\"grid\":\"<81 chars>\"}}\n\n"
    "Description:\n{description}"
)

GRID_JSON_PATTERN = re.compile(r'\{[^{}]*"grid"\s*:\s*"[1-9.]{81}"[^{}]*\}', flags=re.IGNORECASE)
GRID_FIELD_PATTERN = re.compile(r'"grid"\s*:\s*"([1-9.]{81})"', flags=re.IGNORECASE)

SOLUTION_JSON_PATTERN = re.compile(r'\{[^{}]*"solution"\s*:\s*"[1-9]{81}"[^{}]*\}', flags=re.IGNORECASE)
SOLUTION_FIELD_PATTERN = re.compile(r'"solution"\s*:\s*"([1-9]{81})"', flags=re.IGNORECASE)
SOLUTION_TOKEN_PATTERN = re.compile(r"(?<![0-9])([1-9]{81})(?![0-9])")
SUCCESS_PARSE_STATUSES = {"json", "embedded_json", "regex_field", "token_fallback", "flat_digits"}


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return start


PROJECT_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _effective_nl_description(row: dict) -> str:
    corrected = row.get("corrected_nl_description")
    if isinstance(corrected, str) and corrected.strip():
        return corrected
    original = row.get("nl_description")
    if isinstance(original, str):
        return original
    return ""


def load_dataset(path: str, max_samples: int) -> List[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON in {path}")

    out: List[dict] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        puzzle = row.get("puzzle")
        solution = row.get("solution")
        description = _effective_nl_description(row)
        if (
            isinstance(puzzle, str)
            and isinstance(solution, str)
            and len(puzzle) == 81
            and len(solution) == 81
            and description.strip()
        ):
            out.append(
                {
                    "index": int(row.get("index", idx)),
                    "puzzle": puzzle,
                    "solution": solution,
                    "rating": row.get("rating"),
                    "nl_description": description,
                }
            )

    if max_samples > 0:
        out = out[:max_samples]
    if not out:
        raise ValueError(f"No valid examples loaded from {path}")
    return out


def _extract_message_text(message_content) -> str:
    content = getattr(message_content, "content", None)
    if content is not None:
        content_text = _extract_message_text(content)
        if content_text:
            return content_text
        reasoning_content = getattr(message_content, "reasoning_content", None)
        if isinstance(reasoning_content, str):
            return reasoning_content.strip()

    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict):
                piece = item.get("text")
                if isinstance(piece, str) and piece.strip():
                    parts.append(piece.strip())
        return "\n".join(parts).strip()
    reasoning_content = getattr(message_content, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content.strip()
    return ""


def _sanitize_candidate(value: str, allow_dots: bool) -> str:
    if not isinstance(value, str):
        return ""
    allowed = "123456789." if allow_dots else "123456789"
    return "".join(ch for ch in value if ch in allowed)


def parse_response_text(text: str, *, allow_digit_fallback: bool) -> Tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        return "", "empty_response"

    raw = text.strip()

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            solution = _sanitize_candidate(str(payload.get("solution", "")), allow_dots=False)
            if len(solution) == 81 and set(payload.keys()) == {"solution"}:
                return solution, "json"
    except Exception:
        pass

    json_matches = SOLUTION_JSON_PATTERN.findall(raw)
    for json_text in reversed(json_matches):
        try:
            payload = json.loads(json_text)
            if isinstance(payload, dict):
                solution = _sanitize_candidate(str(payload.get("solution", "")), allow_dots=False)
                if len(solution) == 81 and set(payload.keys()) == {"solution"}:
                    return solution, "embedded_json"
        except Exception:
            pass

    solution_matches = SOLUTION_FIELD_PATTERN.findall(raw)
    if solution_matches:
        return solution_matches[-1], "regex_field"

    if not allow_digit_fallback:
        return "", "missing_solution"

    solution_tokens = SOLUTION_TOKEN_PATTERN.findall(raw)
    if solution_tokens:
        return solution_tokens[-1], "token_fallback"

    flat_solution = _sanitize_candidate(raw, allow_dots=False)
    if len(flat_solution) >= 81:
        return flat_solution[-81:], "flat_digits"

    return "", "missing_solution"


def _is_valid_sudoku_solution(solution: str) -> bool:
    if not isinstance(solution, str) or len(solution) != 81 or not solution.isdigit():
        return False

    def _valid_group(chars: Iterable[str]) -> bool:
        return sorted(chars) == list("123456789")

    rows = [solution[i * 9:(i + 1) * 9] for i in range(9)]
    cols = ["".join(solution[r * 9 + c] for r in range(9)) for c in range(9)]
    boxes = []
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            chars = []
            for r in range(box_r, box_r + 3):
                for c in range(box_c, box_c + 3):
                    chars.append(solution[r * 9 + c])
            boxes.append("".join(chars))

    return all(_valid_group(group) for group in rows + cols + boxes)


def compute_solution_metrics(rows: List[dict]) -> dict:
    n = len(rows)
    exact = 0
    cell = 0.0
    valid = 0
    consistent_with_gold_givens = 0
    sudoku_valid = 0
    structure_valid = 0
    strict_json = 0

    for row in rows:
        gold_solution = row["solution"]
        gold_puzzle = row["puzzle"]
        pred = row.get("predicted_solution", "")
        parse_status = str(row.get("parse_status", ""))
        if parse_status in SUCCESS_PARSE_STATUSES:
            structure_valid += 1
        if parse_status == "json":
            strict_json += 1
        if isinstance(pred, str) and len(pred) == 81 and pred.isdigit() and all(c in "123456789" for c in pred):
            valid += 1
            if _is_valid_sudoku_solution(pred):
                sudoku_valid += 1
        else:
            pred = "0" * 81

        if pred == gold_solution:
            exact += 1
        cell += sum(1 for a, b in zip(pred, gold_solution) if a == b) / 81.0

        ok = True
        for i, ch in enumerate(gold_puzzle):
            if ch != "." and pred[i] != ch:
                ok = False
                break
        if ok:
            consistent_with_gold_givens += 1

    return {
        "num_examples": n,
        "structure_valid_rate": structure_valid / max(n, 1),
        "strict_json_rate": strict_json / max(n, 1),
        "exact_match": exact / max(n, 1),
        "cell_accuracy": cell / max(n, 1),
        "format_valid_rate": valid / max(n, 1),
        "valid_sudoku_rate": sudoku_valid / max(n, 1),
        "consistent_with_gold_givens_rate": consistent_with_gold_givens / max(n, 1),
    }


def parse_grid_response_text(text: str) -> Tuple[str, str]:
    """Parse a grid-parse response, returning (grid_81_chars, parse_status)."""
    if not isinstance(text, str) or not text.strip():
        return "", "empty_response"
    raw = text.strip()

    def _normalize_grid(g: str) -> str:
        """Pad with dots to 81 if short, truncate if long."""
        g = g.ljust(81, ".")[:81]
        return g if all(ch == "." or ch in "123456789" for ch in g) else ""

    # Try strict JSON
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "grid" in payload:
            grid = _normalize_grid(str(payload["grid"]))
            if grid:
                return grid, "json"
    except Exception:
        pass

    # Try embedded JSON — relax pattern to allow shorter grids
    for match in reversed(re.findall(r'\{[^{}]*"grid"\s*:\s*"[1-9.]{10,81}"[^{}]*\}', raw, flags=re.IGNORECASE)):
        try:
            payload = json.loads(match)
            if isinstance(payload, dict) and "grid" in payload:
                grid = _normalize_grid(str(payload["grid"]))
                if grid:
                    return grid, "embedded_json"
        except Exception:
            pass

    # Try regex field extraction — relax to 10+ chars
    field_matches = re.findall(r'"grid"\s*:\s*"([1-9.]{10,81})"', raw, flags=re.IGNORECASE)
    if field_matches:
        grid = _normalize_grid(field_matches[-1])
        if grid:
            return grid, "regex_field"

    # Try extracting any 10-81 char sequence of digits and dots
    token = re.search(r"(?<![0-9.])([1-9.]{10,81})(?![0-9.])", raw)
    if token:
        grid = _normalize_grid(token.group(1))
        if grid:
            return grid, "token_fallback"

    return "", "missing_grid"


GRID_PARSE_SUCCESS_STATUSES = {"json", "embedded_json", "regex_field", "token_fallback"}


def compute_grid_parse_metrics(rows: List[dict]) -> dict:
    """Compute bridge-equivalent metrics for grid-parse task (matches Tables 1/2)."""
    n = len(rows)
    exact = 0
    cell_acc = 0.0
    given_acc = 0.0
    empty_acc = 0.0
    format_valid = 0

    for row in rows:
        gold = row["puzzle"]  # the ground-truth initial grid
        pred = row.get("predicted_grid", "")
        if isinstance(pred, str) and len(pred) == 81 and all(ch == "." or ch in "123456789" for ch in pred):
            format_valid += 1
        else:
            pred = "." * 81

        if pred == gold:
            exact += 1
        cell_acc += sum(1 for a, b in zip(pred, gold) if a == b) / 81.0

        given_idx = [i for i, ch in enumerate(gold) if ch != "."]
        empty_idx = [i for i, ch in enumerate(gold) if ch == "."]
        if given_idx:
            given_acc += sum(1 for i in given_idx if pred[i] == gold[i]) / len(given_idx)
        if empty_idx:
            empty_acc += sum(1 for i in empty_idx if pred[i] == gold[i]) / len(empty_idx)

    return {
        "num_examples": n,
        "exact_match": exact / max(n, 1),
        "cell_accuracy": cell_acc / max(n, 1),
        "given_accuracy": given_acc / max(n, 1),
        "empty_accuracy": empty_acc / max(n, 1),
        "format_valid_rate": format_valid / max(n, 1),
    }


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


def build_user_prompt(description: str, retry_instruction: str = "") -> str:
    prompt = (
        "Solve the Sudoku described below. Output only the final JSON answer, no reasoning.\n\n"
        "End with exactly: {\"solution\":\"<81 digits>\"}\n"
    )
    if retry_instruction:
        prompt += f"\nRetry instruction:\n{retry_instruction}\n"
    prompt += f"\nClue description:\n{description}"
    return prompt


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


def call_model(
    *,
    client: OpenAI,
    model: str,
    description: str,
    temperature: float,
    max_tokens: int,
    allow_digit_fallback: bool,
    task: str = "solve",
    extra_body: Optional[Dict[str, Any]] = None,
    retry_instruction: str = "",
) -> Tuple[str, str, str, str]:
    if task == "grid-parse":
        system = GRID_PARSE_SYSTEM_PROMPT
        user = GRID_PARSE_USER_TEMPLATE.format(description=description)
        if retry_instruction:
            user += f"\n\nRetry instruction:\n{retry_instruction}"
    else:
        system = SYSTEM_PROMPT
        user = build_user_prompt(description, retry_instruction=retry_instruction)

    request_payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        request_payload["extra_body"] = extra_body
    response = client.chat.completions.create(**request_payload)
    message = response.choices[0].message
    text = _extract_message_text(message)

    if task == "grid-parse":
        predicted, parse_status = parse_grid_response_text(text)
    else:
        predicted, parse_status = parse_response_text(text, allow_digit_fallback=allow_digit_fallback)

    return text, "", predicted, parse_status


def load_existing_predictions(path: str) -> Dict[Tuple[int, str], dict]:
    p = Path(path)
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")

    out: Dict[Tuple[int, str], dict] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        model = row.get("model")
        if isinstance(idx, int) and isinstance(model, str):
            out[(idx, model)] = row
    return out


def should_rerun_existing(row: dict) -> bool:
    status = str(row.get("parse_status", ""))
    if not status:
        return True
    return status.startswith("api_error") or status not in SUCCESS_PARSE_STATUSES


def write_predictions(path: str, predictions_by_key: Dict[Tuple[int, str], dict]) -> None:
    ordered = [
        predictions_by_key[key]
        for key in sorted(predictions_by_key.keys(), key=lambda item: (item[0], item[1]))
    ]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def compute_metrics_by_model(
    predictions_by_key: Dict[Tuple[int, str], dict],
    models: List[str],
    baseline_metrics: Optional[dict],
    task: str = "solve",
) -> dict:
    metrics_by_model = {}
    for model in models:
        rows = [
            row
            for (_, row_model), row in predictions_by_key.items()
            if row_model == model
        ]
        rows.sort(key=lambda row: int(row["index"]))

        api_errors = sum(1 for row in rows if str(row.get("parse_status", "")).startswith("api_error"))
        success_statuses = GRID_PARSE_SUCCESS_STATUSES if task == "grid-parse" else SUCCESS_PARSE_STATUSES
        parse_failures = sum(
            1
            for row in rows
            if not str(row.get("parse_status", "")).startswith("api_error")
            and row.get("parse_status") not in success_statuses
        )

        if task == "grid-parse":
            task_metrics = compute_grid_parse_metrics(rows)
            model_metrics = {
                "num_examples": len(rows),
                "api_error_count": api_errors,
                "parse_failure_count": parse_failures,
                "grid_metrics": task_metrics,
            }
        else:
            task_metrics = compute_solution_metrics(rows)
            model_metrics = {
                "num_examples": len(rows),
                "api_error_count": api_errors,
                "parse_failure_count": parse_failures,
                "solution_metrics": task_metrics,
            }
            if baseline_metrics:
                trm_baseline = baseline_metrics.get("trm_metrics", {})
                model_metrics["comparison_to_pipeline"] = {
                    "solution_exact_match_delta_vs_trm": task_metrics["exact_match"] - float(trm_baseline.get("exact_match", 0.0)),
                    "solution_cell_accuracy_delta_vs_trm": task_metrics["cell_accuracy"] - float(trm_baseline.get("cell_accuracy", 0.0)),
                }

        metrics_by_model[model] = model_metrics

    return metrics_by_model


def maybe_load_baseline_metrics(path: str) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def log_to_wandb(args, payload: dict) -> None:
    if not args.wandb:
        return
    if wandb is None:
        raise ImportError("--wandb requested but wandb is not installed")

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    task = payload.get("task", "solve")
    for model, metrics in payload["models"].items():
        key = sanitize_model_name(model)
        if task == "grid-parse":
            grid_metrics = metrics.get("grid_metrics", {})
            run.log({f"{key}/grid/{name}": value for name, value in grid_metrics.items()})
        else:
            solution_metrics = metrics.get("solution_metrics", {})
            run.log({f"{key}/solution/{name}": value for name, value in solution_metrics.items()})
        run.summary[f"{key}/model"] = model
        run.summary[f"{key}/api_error_count"] = metrics["api_error_count"]
        run.summary[f"{key}/parse_failure_count"] = metrics["parse_failure_count"]
    run.summary["input_json"] = payload["input_json"]
    run.summary["predictions_json"] = payload["predictions_json"]
    run.summary["task"] = task
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA Builder frontier models on corrected2 Sudoku solving tasks")
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    parser.add_argument("--baseline-metrics-json", default=DEFAULT_BASELINE_METRICS_JSON)
    parser.add_argument("--models", nargs="+", required=True, help="NVIDIA Builder model IDs")
    parser.add_argument("--api-key", default=None, help="NVIDIA API key or set NVIDIA_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--allow-digit-fallback",
        action="store_true",
        help="Allow parser to accept a bare 81-digit token when no solution JSON is found. Disabled by default to avoid extracting digits from reasoning traces.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "high", "max"],
        default=None,
        help="NVIDIA DeepSeek V4 Pro reasoning mode. Applied only to models matching --reasoning-effort-model-regex.",
    )
    parser.add_argument(
        "--reasoning-effort-model-regex",
        default=r"deepseek-ai/deepseek-v4",
        help="Regex selecting which model IDs receive --reasoning-effort.",
    )
    parser.add_argument(
        "--extra-body-json",
        default="",
        help="Optional JSON object merged into OpenAI extra_body for every model request.",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--calls-per-minute", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="trm-llm-test-pipeline")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--task",
        choices=["solve", "grid-parse"],
        default="solve",
        help=(
            "solve: ask model to return the fully solved 81-digit solution (default). "
            "grid-parse: ask model to return only the initial puzzle grid with '.' for empty cells. "
            "Use grid-parse to generate results for Tables 1 and 2."
        ),
    )
    return parser.parse_args()


def build_extra_body_for_model(args, model: str) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if args.extra_body_json:
        parsed = json.loads(args.extra_body_json)
        if not isinstance(parsed, dict):
            raise ValueError("--extra-body-json must decode to a JSON object")
        extra_body.update(parsed)
    if args.reasoning_effort and re.search(args.reasoning_effort_model_regex, model):
        extra_body["reasoning_effort"] = args.reasoning_effort
    return extra_body


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY or pass --api-key")

    dataset = load_dataset(args.input_json, args.max_samples)
    baseline_metrics = maybe_load_baseline_metrics(args.baseline_metrics_json)
    request_extra_body_by_model = {
        model: build_extra_body_for_model(args, model)
        for model in args.models
    }

    existing = load_existing_predictions(args.output_json) if args.resume else {}
    predictions_by_key = dict(existing)

    pending: List[dict] = []
    for record in dataset:
        for model in args.models:
            key = (record["index"], model)
            existing_row = predictions_by_key.get(key)
            if existing_row is not None and not (args.retry_errors and should_rerun_existing(existing_row)):
                continue
            pending.append({"record": record, "model": model})

    print(f"[benchmark] input_json={args.input_json}")
    print(f"[benchmark] num_examples={len(dataset)}")
    print(f"[benchmark] models={args.models}")
    print(f"[benchmark] pending_calls={len(pending)}")
    print(f"[benchmark] output_json={args.output_json}")
    print(f"[benchmark] metrics_json={args.metrics_json}")
    print(f"[benchmark] max_tokens={args.max_tokens}")
    print(f"[benchmark] allow_digit_fallback={args.allow_digit_fallback}")
    print(f"[benchmark] extra_body_by_model={request_extra_body_by_model}")

    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout_seconds)
    limiter = GlobalRateLimiter(args.calls_per_minute) if args.calls_per_minute > 0 else None

    def _run_one(job: dict) -> dict:
        record = job["record"]
        model = job["model"]
        last_error: Optional[Exception] = None
        last_parse_status = "missing_solution"
        last_response_text = ""
        retry_instruction = ""
        success_statuses = GRID_PARSE_SUCCESS_STATUSES if args.task == "grid-parse" else SUCCESS_PARSE_STATUSES

        for attempt in range(args.max_retries + 1):
            if limiter is not None:
                limiter.acquire_slot()
            try:
                response_text, _, predicted, parse_status = call_model(
                    client=client,
                    model=model,
                    description=record["nl_description"],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    allow_digit_fallback=args.allow_digit_fallback,
                    task=args.task,
                    extra_body=request_extra_body_by_model.get(model) or None,
                    retry_instruction=retry_instruction,
                )
                last_response_text = response_text
                last_parse_status = parse_status
                if parse_status in success_statuses:
                    row = {
                        "index": record["index"],
                        "model": model,
                        "puzzle": record["puzzle"],
                        "solution": record["solution"],
                        "rating": record.get("rating"),
                        "nl_description": record["nl_description"],
                        "parse_status": parse_status,
                        "task": args.task,
                        "request_extra_body": request_extra_body_by_model.get(model, {}),
                        "response_text": response_text,
                    }
                    if args.task == "grid-parse":
                        row["predicted_grid"] = predicted
                    else:
                        row["predicted_solution"] = predicted
                    return row
                if attempt < args.max_retries:
                    if args.task == "grid-parse":
                        retry_instruction = (
                            f"Your previous answer had invalid format ({parse_status}). "
                            "Return only the initial grid (not the solution) as exactly 81 characters "
                            "using digits 1-9 for givens and '.' for empty cells, "
                            "ending with one JSON object: {\"grid\":\"<81 chars>\"}."
                        )
                    else:
                        retry_instruction = (
                            f"Your previous answer had invalid format ({parse_status}). "
                            "Keep solving, then end with one JSON object with exactly "
                            "one key named `solution` containing exactly 81 digits."
                        )
                    continue
            except Exception as exc:
                last_error = exc
                if attempt < args.max_retries:
                    retry_instruction = (
                        f"The previous attempt failed with {type(exc).__name__}. "
                        "Try again."
                    )
                    time.sleep(2.0 * (attempt + 1))
                    continue

        row = {
            "index": record["index"],
            "model": model,
            "puzzle": record["puzzle"],
            "solution": record["solution"],
            "rating": record.get("rating"),
            "nl_description": record["nl_description"],
            "predicted_grid" if args.task == "grid-parse" else "predicted_solution": "",
            "parse_status": last_parse_status if last_response_text else (f"api_error:{type(last_error).__name__}" if last_error else "api_error"),
            "task": args.task,
            "request_extra_body": request_extra_body_by_model.get(model, {}),
            "response_text": last_response_text or (str(last_error) if last_error else ""),
        }
        return row

    completed_since_save = 0
    progress = tqdm(total=len(pending), desc="Benchmarking models") if tqdm is not None and pending else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {executor.submit(_run_one, job): job for job in pending}
            for future in as_completed(futures):
                row = future.result()
                predictions_by_key[(row["index"], row["model"])] = row
                completed_since_save += 1
                if progress is not None:
                    progress.update(1)
                if completed_since_save >= max(1, args.save_every):
                    write_predictions(args.output_json, predictions_by_key)
                    completed_since_save = 0
    finally:
        if progress is not None:
            progress.close()

    write_predictions(args.output_json, predictions_by_key)

    payload = {
        "input_json": args.input_json,
        "task": args.task,
        "baseline_metrics_json": args.baseline_metrics_json if baseline_metrics else None,
        "baseline_pipeline_metrics": baseline_metrics,
        "predictions_json": args.output_json,
        "max_tokens": args.max_tokens,
        "allow_digit_fallback": args.allow_digit_fallback,
        "extra_body_by_model": request_extra_body_by_model,
        "models": compute_metrics_by_model(predictions_by_key, args.models, baseline_metrics, task=args.task),
    }

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log_to_wandb(args, payload)

    print(f"[benchmark] wrote predictions: {args.output_json}")
    print(f"[benchmark] wrote metrics: {args.metrics_json}")
    for model, metrics in payload["models"].items():
        if args.task == "grid-parse":
            gm = metrics.get("grid_metrics", {})
            print(
                "[summary]", model,
                f"grid_exact={gm.get('exact_match', 0):.4f}",
                f"cell_acc={gm.get('cell_accuracy', 0):.4f}",
                f"given_acc={gm.get('given_accuracy', 0):.4f}",
                f"format_valid={gm.get('format_valid_rate', 0):.4f}",
            )
        else:
            sm = metrics.get("solution_metrics", {})
            print(
                "[summary]", model,
                f"structure_valid={sm.get('structure_valid_rate', 0):.4f}",
                f"strict_json={sm.get('strict_json_rate', 0):.4f}",
                f"solution_exact={sm.get('exact_match', 0):.4f}",
                f"solution_cell={sm.get('cell_accuracy', 0):.4f}",
            )


if __name__ == "__main__":
    main()
