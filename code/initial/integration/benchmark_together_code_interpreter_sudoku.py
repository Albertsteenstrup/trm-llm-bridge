#!/usr/bin/env python3
"""
Benchmark Together-hosted open models on natural-language Sudoku with access to
Together Code Interpreter.

This is intentionally a small controlled baseline: the model either returns the
final JSON answer or asks the harness to execute Python. The harness keeps the
code-interpreter loop identical across future model sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_INPUT_JSON = "data/initial/sudoku_synthetic/llm/sudoku_nl_dataset_corrected2.json"
DEFAULT_OUTPUT_JSON = "results/together_sudoku_code_interpreter/together_qwen3_1_7b_predictions_25.json"
DEFAULT_METRICS_JSON = "results/together_sudoku_code_interpreter/together_qwen3_1_7b_metrics_25.json"
DEFAULT_BASELINE_METRICS_JSON = "results/test_pipeline_finetuned/pipeline_metrics_corrected2.json"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = """You solve Sudoku from natural-language clue descriptions.

You have access to a Python code interpreter through the harness.

Respond in exactly one of these two formats:

1. To run Python:
{"code":"<python code that prints the answer JSON>"}

2. To give the final answer:
{"solution":"<81 digits>"}

Rules:
- The final solution must be 81 digits from 1-9.
- If using code, write self-contained Python. It may parse the clue text and solve Sudoku.
- The code must print exactly one JSON object with key "solution" when it succeeds.
- No markdown, no code fences, no explanations, no extra keys.
"""

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)
SOLUTION_FIELD_PATTERN = re.compile(r'"solution"\s*:\s*"([1-9]{81})"', flags=re.IGNORECASE)
CODE_FIELD_PATTERN = re.compile(r'"code"\s*:\s*"', flags=re.IGNORECASE)
FENCED_CODE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
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
    return original if isinstance(original, str) else ""


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


def parse_solution_text(text: str) -> Tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        return "", "empty_response"
    raw = text.strip()

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            solution = _sanitize_candidate(str(payload.get("solution", "")))
            if len(solution) == 81:
                return solution, "json" if set(payload.keys()) == {"solution"} else "embedded_json"
    except Exception:
        pass

    json_match = JSON_OBJECT_PATTERN.search(raw)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                solution = _sanitize_candidate(str(payload.get("solution", "")))
                if len(solution) == 81:
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


def parse_code_request(text: str) -> Tuple[str, str]:
    """Return code and parse status from a model message."""
    if not isinstance(text, str) or not text.strip():
        return "", "missing_code"

    raw = text.strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("code"), str):
            return payload["code"], "json_code"
    except Exception:
        pass

    json_match = JSON_OBJECT_PATTERN.search(raw)
    if json_match and CODE_FIELD_PATTERN.search(json_match.group(0)):
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict) and isinstance(payload.get("code"), str):
                return payload["code"], "embedded_json_code"
        except Exception:
            pass

    fenced = FENCED_CODE_PATTERN.search(raw)
    if fenced:
        return fenced.group(1).strip(), "fenced_code"

    return "", "missing_code"


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
    format_valid = 0
    sudoku_valid = 0
    consistent_with_gold_givens = 0
    structure_valid = 0
    tool_used = 0

    for row in rows:
        gold_solution = row["solution"]
        gold_puzzle = row["puzzle"]
        pred = row.get("predicted_solution", "")
        parse_status = str(row.get("parse_status", ""))
        if parse_status in SUCCESS_PARSE_STATUSES or parse_status.startswith("tool_"):
            structure_valid += 1
        if row.get("tool_used"):
            tool_used += 1
        if isinstance(pred, str) and len(pred) == 81 and pred.isdigit():
            format_valid += 1
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
        "tool_used_rate": tool_used / max(n, 1),
        "structure_valid_rate": structure_valid / max(n, 1),
        "exact_match": exact / max(n, 1),
        "cell_accuracy": cell / max(n, 1),
        "format_valid_rate": format_valid / max(n, 1),
        "valid_sudoku_rate": sudoku_valid / max(n, 1),
        "consistent_with_gold_givens_rate": consistent_with_gold_givens / max(n, 1),
    }


def build_initial_user_prompt(description: str) -> str:
    return (
        "Solve the Sudoku described below. You may request Python execution by "
        "returning {\"code\":\"...\"}. If you already know the answer, return "
        "only {\"solution\":\"<81 digits>\"}.\n\n"
        f"Clue description:\n{description}"
    )


def build_tool_result_message(stdout: str, stderr: str, status: str) -> str:
    return (
        "The Python code was executed.\n"
        f"status: {status}\n"
        f"stdout:\n{stdout[-12000:]}\n"
        f"stderr:\n{stderr[-4000:]}\n\n"
        "Now return only {\"solution\":\"<81 digits>\"}. If the code failed, "
        "you may request one more Python execution with {\"code\":\"...\"}."
    )


def chat_completion(client, model: str, messages: List[dict], max_tokens: int, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def run_code_interpreter(client, code: str) -> Tuple[str, str, str, dict]:
    code_interpreter = client.code_interpreter
    if hasattr(code_interpreter, "execute"):
        response = code_interpreter.execute(code=code, language="python")
    else:
        response = code_interpreter.run(code=code, language="python")
    data = _field(response, "data", response)
    status = str(_field(data, "status", "unknown"))
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []

    for output in _field(data, "outputs", []) or []:
        output_type = str(_field(output, "type", ""))
        output_data = _field(output, "data", "")
        if output_type == "stderr":
            stderr_parts.append(str(output_data))
        else:
            stdout_parts.append(str(output_data))

    errors = _field(data, "errors", None) or _field(response, "errors", None)
    if errors:
        stderr_parts.append(str(errors))

    dump = response.model_dump() if hasattr(response, "model_dump") else {}
    return "\n".join(stdout_parts).strip(), "\n".join(stderr_parts).strip(), status, dump


def solve_one(client, record: dict, model: str, max_turns: int, max_tokens: int, temperature: float) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_user_prompt(record["nl_description"])},
    ]

    transcript: List[dict] = []
    tool_used = False
    last_text = ""
    last_code = ""
    last_code_status = ""
    last_code_stdout = ""
    last_code_stderr = ""

    for turn in range(max_turns):
        try:
            text = chat_completion(client, model, messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            transcript.append({"turn": turn, "role": "assistant", "error": error_text})
            return {
                "index": record["index"],
                "model": model,
                "puzzle": record["puzzle"],
                "solution": record["solution"],
                "rating": record.get("rating"),
                "nl_description": record["nl_description"],
                "predicted_solution": "",
                "parse_status": f"api_error:{type(exc).__name__}",
                "tool_used": tool_used,
                "turns": turn + 1,
                "last_code_status": last_code_status,
                "last_code_stdout": last_code_stdout,
                "last_code_stderr": last_code_stderr,
                "last_code": last_code,
                "response_text": error_text,
                "transcript": transcript,
            }
        last_text = text
        transcript.append({"turn": turn, "role": "assistant", "content": text})

        solution, parse_status = parse_solution_text(text)
        if parse_status in SUCCESS_PARSE_STATUSES:
            return {
                "index": record["index"],
                "model": model,
                "puzzle": record["puzzle"],
                "solution": record["solution"],
                "rating": record.get("rating"),
                "nl_description": record["nl_description"],
                "predicted_solution": solution,
                "parse_status": parse_status,
                "tool_used": tool_used,
                "turns": turn + 1,
                "last_code_status": last_code_status,
                "last_code_stdout": last_code_stdout,
                "last_code_stderr": last_code_stderr,
                "last_code": last_code,
                "response_text": text,
                "transcript": transcript,
            }

        code, code_status = parse_code_request(text)
        if code:
            tool_used = True
            last_code = code
            try:
                stdout, stderr, status, dump = run_code_interpreter(client, code)
            except Exception as exc:
                stdout, stderr, status, dump = "", str(exc), f"api_error:{type(exc).__name__}", {}
            last_code_stdout = stdout
            last_code_stderr = stderr
            last_code_status = status
            transcript.append(
                {
                    "turn": turn,
                    "role": "tool",
                    "code_parse_status": code_status,
                    "status": status,
                    "stdout": stdout,
                    "stderr": stderr,
                    "raw": dump,
                }
            )
            solution, tool_parse_status = parse_solution_text(stdout)
            if tool_parse_status in SUCCESS_PARSE_STATUSES:
                return {
                    "index": record["index"],
                    "model": model,
                    "puzzle": record["puzzle"],
                    "solution": record["solution"],
                    "rating": record.get("rating"),
                    "nl_description": record["nl_description"],
                    "predicted_solution": solution,
                    "parse_status": f"tool_{tool_parse_status}",
                    "tool_used": tool_used,
                    "turns": turn + 1,
                    "last_code_status": last_code_status,
                    "last_code_stdout": last_code_stdout,
                    "last_code_stderr": last_code_stderr,
                    "last_code": last_code,
                    "response_text": stdout,
                    "transcript": transcript,
                }
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": build_tool_result_message(stdout, stderr, status)})
            continue

        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Invalid format. Return either {\"code\":\"<python>\"} to run Python "
                    "or {\"solution\":\"<81 digits>\"} as the final answer."
                ),
            }
        )

    return {
        "index": record["index"],
        "model": model,
        "puzzle": record["puzzle"],
        "solution": record["solution"],
        "rating": record.get("rating"),
        "nl_description": record["nl_description"],
        "predicted_solution": "",
        "parse_status": "missing_solution",
        "tool_used": tool_used,
        "turns": max_turns,
        "last_code_status": last_code_status,
        "last_code_stdout": last_code_stdout,
        "last_code_stderr": last_code_stderr,
        "last_code": last_code,
        "response_text": last_text,
        "transcript": transcript,
    }


def load_existing_predictions(path: str) -> Dict[Tuple[int, str], dict]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    out: Dict[Tuple[int, str], dict] = {}
    for row in data:
        if isinstance(row, dict) and isinstance(row.get("index"), int) and isinstance(row.get("model"), str):
            out[(row["index"], row["model"])] = row
    return out


def write_predictions(path: str, predictions_by_key: Dict[Tuple[int, str], dict]) -> None:
    ordered = [
        predictions_by_key[key]
        for key in sorted(predictions_by_key.keys(), key=lambda item: (item[0], item[1]))
    ]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def maybe_load_baseline_metrics(path: str) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def compute_metrics_by_model(predictions_by_key: Dict[Tuple[int, str], dict], models: List[str], baseline_metrics: Optional[dict]) -> dict:
    metrics_by_model = {}
    for model in models:
        rows = [row for (_, row_model), row in predictions_by_key.items() if row_model == model]
        rows.sort(key=lambda row: int(row["index"]))
        api_errors = sum(1 for row in rows if str(row.get("parse_status", "")).startswith("api_error"))
        parse_failures = sum(
            1
            for row in rows
            if row.get("parse_status") not in SUCCESS_PARSE_STATUSES
            and not str(row.get("parse_status", "")).startswith(("tool_", "api_error"))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Together Qwen3-1.7B with Together Code Interpreter on 25 Sudoku tasks")
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    parser.add_argument("--baseline-metrics-json", default=DEFAULT_BASELINE_METRICS_JSON)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--api-key", default=None, help="Together API key or set TOGETHER_API_KEY")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def should_skip_existing(row: dict, retry_errors: bool) -> bool:
    if not retry_errors:
        return True
    status = str(row.get("parse_status", ""))
    return status in SUCCESS_PARSE_STATUSES or status.startswith("tool_")


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Set TOGETHER_API_KEY in .env or pass --api-key")

    try:
        from together import Together
    except Exception as exc:
        raise ImportError("Install the Together SDK with `python -m pip install together`") from exc

    client = Together(api_key=api_key)
    dataset = load_dataset(args.input_json, args.max_samples)
    baseline_metrics = maybe_load_baseline_metrics(args.baseline_metrics_json)
    predictions_by_key = load_existing_predictions(args.output_json) if args.resume else {}

    total_jobs = 0
    completed_since_save = 0
    progress_total = len(dataset) * len(args.models)
    progress = tqdm(total=progress_total, desc="Together code-interpreter Sudoku") if tqdm is not None else None

    try:
        for model in args.models:
            for record in dataset:
                key = (record["index"], model)
                existing = predictions_by_key.get(key)
                if existing is not None and should_skip_existing(existing, args.retry_errors):
                    if progress is not None:
                        progress.update(1)
                    continue

                row = solve_one(
                    client=client,
                    record=record,
                    model=model,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                predictions_by_key[key] = row
                total_jobs += 1
                completed_since_save += 1
                if progress is not None:
                    progress.update(1)
                if completed_since_save >= max(1, args.save_every):
                    write_predictions(args.output_json, predictions_by_key)
                    completed_since_save = 0
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
    finally:
        if progress is not None:
            progress.close()

    write_predictions(args.output_json, predictions_by_key)
    payload = {
        "input_json": args.input_json,
        "baseline_metrics_json": args.baseline_metrics_json if baseline_metrics else None,
        "baseline_pipeline_metrics": baseline_metrics,
        "predictions_json": args.output_json,
        "max_samples": args.max_samples,
        "max_turns": args.max_turns,
        "models": compute_metrics_by_model(predictions_by_key, args.models, baseline_metrics),
    }
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[benchmark] executed_jobs={total_jobs}")
    print(f"[benchmark] wrote predictions: {args.output_json}")
    print(f"[benchmark] wrote metrics: {args.metrics_json}")
    for model, metrics in payload["models"].items():
        solution_metrics = metrics["solution_metrics"]
        print(
            "[summary]",
            model,
            f"tool_used={solution_metrics['tool_used_rate']:.4f}",
            f"structure_valid={solution_metrics['structure_valid_rate']:.4f}",
            f"solution_exact={solution_metrics['exact_match']:.4f}",
            f"solution_cell={solution_metrics['cell_accuracy']:.4f}",
        )


if __name__ == "__main__":
    main()
