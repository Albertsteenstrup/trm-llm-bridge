#!/usr/bin/env python3
"""
Benchmark OpenAI models on the corrected2 Sudoku NL test set using the
Responses API with built-in Code Interpreter.

This mirrors the NVIDIA frontier benchmark structure so outputs and metrics are
easy to compare across providers.
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
from typing import Dict, Iterable, List, Optional, Tuple

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
DEFAULT_OUTPUT_JSON = "results/openai_sudoku_code_interpreter/openai_predictions_corrected2.json"
DEFAULT_METRICS_JSON = "results/openai_sudoku_code_interpreter/openai_metrics_corrected2.json"
DEFAULT_BASELINE_METRICS_JSON = "results/test_pipeline_finetuned/pipeline_metrics_corrected2.json"
DEFAULT_MODEL = "gpt-5.4"

SYSTEM_PROMPT = """You solve Sudoku from natural-language clue descriptions.

You may use the Code Interpreter tool to write and run Python code.

Return exactly one JSON object with this schema and nothing else:
{"solution":"<81 digits>"}

Rules:
- solution must be the fully solved Sudoku, 81 digits from 1-9.
- Use Code Interpreter if needed to parse or solve the puzzle.
- No markdown, no code fences, no explanations, no extra keys.
- The string must be length 81.
"""

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)
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


def _sanitize_candidate(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value if ch in "123456789")


def _response_dump(response) -> dict:
    if hasattr(response, "model_dump"):
        try:
            data = response.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _response_incomplete_reason(response) -> Optional[str]:
    data = _response_dump(response)
    incomplete = data.get("incomplete_details")
    if isinstance(incomplete, dict):
        reason = incomplete.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return None


def _extract_response_text_from_dump(response_dump: dict) -> str:
    pieces: List[str] = []
    for item in response_dump.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        pieces.append(text.strip())
    return "\n".join(pieces).strip()


def parse_response_text(text: str) -> Tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        return "", "empty_response"

    raw = text.strip()

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            solution = _sanitize_candidate(str(payload.get("solution", "")))
            if len(solution) == 81 and set(payload.keys()) == {"solution"}:
                return solution, "json"
    except Exception:
        pass

    json_match = JSON_OBJECT_PATTERN.search(raw)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                solution = _sanitize_candidate(str(payload.get("solution", "")))
                if len(solution) == 81 and set(payload.keys()) == {"solution"}:
                    return solution, "embedded_json"
        except Exception:
            pass

    solution_match = SOLUTION_FIELD_PATTERN.search(raw)
    if solution_match:
        return solution_match.group(1), "regex_field"

    solution_tokens = SOLUTION_TOKEN_PATTERN.findall(raw)
    if solution_tokens:
        return solution_tokens[-1], "token_fallback"

    flat_solution = _sanitize_candidate(raw)
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
    tool_used_count = 0

    for row in rows:
        gold_solution = row["solution"]
        gold_puzzle = row["puzzle"]
        pred = row.get("predicted_solution", "")
        parse_status = str(row.get("parse_status", ""))
        if parse_status in SUCCESS_PARSE_STATUSES:
            structure_valid += 1
        if parse_status == "json":
            strict_json += 1
        if row.get("tool_used"):
            tool_used_count += 1
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
        "tool_used_rate": tool_used_count / max(n, 1),
        "structure_valid_rate": structure_valid / max(n, 1),
        "strict_json_rate": strict_json / max(n, 1),
        "exact_match": exact / max(n, 1),
        "cell_accuracy": cell / max(n, 1),
        "format_valid_rate": valid / max(n, 1),
        "valid_sudoku_rate": sudoku_valid / max(n, 1),
        "consistent_with_gold_givens_rate": consistent_with_gold_givens / max(n, 1),
    }


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


def build_user_prompt(description: str, retry_instruction: str = "") -> str:
    prompt = (
        "Solve the Sudoku described below.\n\n"
        "Use Code Interpreter if needed.\n"
        "Return exactly one JSON object with key `solution`.\n"
        "- `solution`: 81 digits using 1-9 only\n"
        "- Output must be exactly like: {\"solution\":\"123...\"}\n"
        "- Do not include reasoning, markdown, code fences, or any extra text\n"
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


def _response_output_items(response) -> List[dict]:
    return _response_dump(response).get("output", []) or []


def _tool_used_from_response(response) -> bool:
    for item in _response_output_items(response):
        if item.get("type") == "code_interpreter_call":
            return True
    return False


def call_model(
    *,
    client: OpenAI,
    model: str,
    description: str,
    max_output_tokens: int,
    container_id: Optional[str],
    memory_limit: str,
    tool_choice: str,
    retry_instruction: str = "",
) -> Tuple[str, str, bool, Optional[str]]:
    tools = [
        {
            "type": "code_interpreter",
            "container": container_id if container_id else {"type": "auto", "memory_limit": memory_limit},
        }
    ]

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(description, retry_instruction=retry_instruction),
        tools=tools,
        tool_choice=tool_choice,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
    )

    response_dump = _response_dump(response)
    output_text = getattr(response, "output_text", "") or _extract_response_text_from_dump(response_dump)
    predicted_solution, parse_status = parse_response_text(output_text)
    incomplete_reason = _response_incomplete_reason(response)
    if not output_text and incomplete_reason:
        parse_status = f"incomplete:{incomplete_reason}"
    tool_used = _tool_used_from_response(response)
    response_id = getattr(response, "id", None)
    return output_text, predicted_solution, tool_used, response_id


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
        parse_failures = sum(
            1
            for row in rows
            if not str(row.get("parse_status", "")).startswith("api_error")
            and row.get("parse_status") not in SUCCESS_PARSE_STATUSES
        )

        model_metrics = {
            "num_examples": len(rows),
            "api_error_count": api_errors,
            "parse_failure_count": parse_failures,
            "solution_metrics": compute_solution_metrics(rows),
        }

        if baseline_metrics:
            trm_baseline = baseline_metrics.get("trm_metrics", {})
            model_metrics["comparison_to_pipeline"] = {
                "solution_exact_match_delta_vs_trm": model_metrics["solution_metrics"]["exact_match"] - float(trm_baseline.get("exact_match", 0.0)),
                "solution_cell_accuracy_delta_vs_trm": model_metrics["solution_metrics"]["cell_accuracy"] - float(trm_baseline.get("cell_accuracy", 0.0)),
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
    for model, metrics in payload["models"].items():
        key = sanitize_model_name(model)
        solution_metrics = metrics["solution_metrics"]
        run.log({f"{key}/solution/{name}": value for name, value in solution_metrics.items()})
        run.summary[f"{key}/model"] = model
        run.summary[f"{key}/api_error_count"] = metrics["api_error_count"]
        run.summary[f"{key}/parse_failure_count"] = metrics["parse_failure_count"]
    run.summary["input_json"] = payload["input_json"]
    run.summary["predictions_json"] = payload["predictions_json"]
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI models on corrected2 Sudoku tasks with Code Interpreter")
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    parser.add_argument("--baseline-metrics-json", default=DEFAULT_BASELINE_METRICS_JSON)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="OpenAI model IDs")
    parser.add_argument("--api-key", default=None, help="OpenAI API key or set OPENAI_API_KEY")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--calls-per-minute", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--memory-limit", choices=["1g", "4g", "16g", "64g"], default="1g")
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="auto")
    parser.add_argument("--container-mode", choices=["shared", "auto"], default="shared")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="trm-llm-test-pipeline")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key")
    if args.container_mode == "shared" and args.concurrency != 1:
        raise ValueError("--container-mode shared currently requires --concurrency 1")

    dataset = load_dataset(args.input_json, args.max_samples)
    baseline_metrics = maybe_load_baseline_metrics(args.baseline_metrics_json)

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

    client = OpenAI(api_key=api_key, timeout=300.0)
    limiter = GlobalRateLimiter(args.calls_per_minute) if args.calls_per_minute > 0 else None

    shared_container_id = None
    if args.container_mode == "shared":
        container = client.containers.create(name="sudoku-code-interpreter", memory_limit=args.memory_limit)
        shared_container_id = container.id
        print(f"[benchmark] shared_container_id={shared_container_id}")

    def _run_one(job: dict) -> dict:
        record = job["record"]
        model = job["model"]
        last_error: Optional[Exception] = None
        last_parse_status = "missing_solution"
        last_response_text = ""
        retry_instruction = ""
        last_tool_used = False
        last_response_id = None

        for attempt in range(args.max_retries + 1):
            if limiter is not None:
                limiter.acquire_slot()
            try:
                response_text, predicted_solution, tool_used, response_id = call_model(
                    client=client,
                    model=model,
                    description=record["nl_description"],
                    max_output_tokens=args.max_output_tokens,
                    container_id=shared_container_id if args.container_mode == "shared" else None,
                    memory_limit=args.memory_limit,
                    tool_choice=args.tool_choice,
                    retry_instruction=retry_instruction,
                )
                parse_status = parse_response_text(response_text)[1]
                last_response_text = response_text
                last_parse_status = parse_status
                last_tool_used = tool_used
                last_response_id = response_id
                if parse_status in SUCCESS_PARSE_STATUSES:
                    return {
                        "index": record["index"],
                        "model": model,
                        "puzzle": record["puzzle"],
                        "solution": record["solution"],
                        "rating": record.get("rating"),
                        "nl_description": record["nl_description"],
                        "predicted_solution": predicted_solution,
                        "parse_status": parse_status,
                        "tool_used": tool_used,
                        "response_id": response_id,
                        "response_text": response_text,
                    }
                if attempt < args.max_retries:
                    retry_instruction = (
                        f"Your previous answer had invalid format ({parse_status}). "
                        "Use Code Interpreter if useful, but return only one JSON object "
                        "with exactly one key named `solution` containing exactly 81 digits."
                    )
                    continue
            except Exception as exc:
                last_error = exc
                if attempt < args.max_retries:
                    retry_instruction = (
                        f"The previous attempt failed with {type(exc).__name__}. "
                        "Try again and return only one JSON object with exactly one "
                        "key named `solution` containing exactly 81 digits."
                    )
                    time.sleep(2.0 * (attempt + 1))
                    continue

        if last_response_text or last_parse_status != "missing_solution":
            return {
                "index": record["index"],
                "model": model,
                "puzzle": record["puzzle"],
                "solution": record["solution"],
                "rating": record.get("rating"),
                "nl_description": record["nl_description"],
                "predicted_solution": "",
                "parse_status": last_parse_status,
                "tool_used": last_tool_used,
                "response_id": last_response_id,
                "response_text": last_response_text,
            }

        return {
            "index": record["index"],
            "model": model,
            "puzzle": record["puzzle"],
            "solution": record["solution"],
            "rating": record.get("rating"),
            "nl_description": record["nl_description"],
            "predicted_solution": "",
            "parse_status": f"api_error:{type(last_error).__name__}" if last_error else "api_error",
            "tool_used": False,
            "response_id": None,
            "response_text": str(last_error) if last_error else "",
        }

    completed_since_save = 0
    progress = tqdm(total=len(pending), desc="Benchmarking OpenAI models") if tqdm is not None and pending else None
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
        "baseline_metrics_json": args.baseline_metrics_json if baseline_metrics else None,
        "baseline_pipeline_metrics": baseline_metrics,
        "predictions_json": args.output_json,
        "container_mode": args.container_mode,
        "shared_container_id": shared_container_id,
        "models": compute_metrics_by_model(predictions_by_key, args.models, baseline_metrics),
    }

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log_to_wandb(args, payload)

    print(f"[benchmark] wrote predictions: {args.output_json}")
    print(f"[benchmark] wrote metrics: {args.metrics_json}")
    for model, metrics in payload["models"].items():
        print(
            "[summary]",
            model,
            f"tool_used={metrics['solution_metrics']['tool_used_rate']:.4f}",
            f"structure_valid={metrics['solution_metrics']['structure_valid_rate']:.4f}",
            f"solution_exact={metrics['solution_metrics']['exact_match']:.4f}",
            f"solution_cell={metrics['solution_metrics']['cell_accuracy']:.4f}",
        )


if __name__ == "__main__":
    main()
