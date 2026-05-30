#!/usr/bin/env python3
"""Evaluate GPT translation from official CLRS-Text to native dm-clrs inputs.

The dataset must be produced by ``build_paired_clrs_text_native.py`` so each row
contains both the official CLRS-Text question and the native input tensors from
the same dm-clrs sampler draw. GPT sees only the text question and the requested
native input schema. Scoring compares the returned native input tensors against
the paired gold tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_INPUT_JSONL = "data/clrs_native_text/raw/paired_text_to_native_sorting_order_len64_25.jsonl"
DEFAULT_OUTPUT_JSON = "results/clrs_native_text/openai_text_to_native_sorting_order_len64_25_gpt54.json"
DEFAULT_METRICS_JSON = "results/clrs_native_text/openai_text_to_native_sorting_order_len64_25_gpt54.metrics.json"
DEFAULT_MODEL = "gpt-5.4"
SUCCESS_PARSE_STATUSES = {"json", "embedded_json"}
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


SYSTEM_PROMPT = """You translate official CLRS-Text benchmark prompts into native dm-clrs input tensors.

You may use Code Interpreter to parse the prompt and build arrays. Do not solve the algorithm and do not predict outputs or hints.

Return exactly one JSON object and nothing else:
{"inputs":{"<input_name>": <native_data>, "...": <native_data>}}

CRITICAL - COMPACT FORMAT & NO TRUNCATION:
- Never return truncated arrays (containing '...') in the JSON response.
- Always output the final JSON in a highly compact format without any unnecessary whitespace or newlines.
- For binary/boolean arrays and matrices (like `A`, `adj`, `s`), always use integer `0` and `1` instead of floats `0.0` and `1.0` to minimize output size.
- For floating point values, round to at most 4 decimal places (e.g. `0.0156` instead of `0.015625`).
- If you use NumPy in Code Interpreter, do not copy-paste default console outputs. Always convert arrays to nested Python lists using `.tolist()` or generate the JSON structure programmatically before outputting.

Native conventions:
- Use zero-based indexing.
- If an input named "pos" is requested, set pos[i] = i / N for N nodes/items.
- scalar node inputs are numeric lists with the requested shape.
- mask node inputs are 0/1 lists.
- mask_one node inputs are one-hot 0/1 lists.
- edge scalar or edge mask inputs are dense matrices with the requested shape.
- graph scalar inputs are a single number unless the schema explicitly says otherwise.
- Preserve numeric values from the CLRS-Text prompt exactly as numbers when they are shown.
"""



def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return start


PROJECT_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _parse_csv_filter(raw: str) -> set[str] | None:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def _select_rows(rows: list[dict[str, Any]], task_filter: set[str] | None, max_samples: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        task = _algorithm(row)
        if task_filter is not None and task not in task_filter:
            continue
        selected.append(row)
        if max_samples > 0 and len(selected) >= max_samples:
            break
    if not selected:
        raise ValueError("No rows selected")
    return selected


def _sample_id(row: dict[str, Any], row_index: int) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    return (
        f"clrs-text-native:{row.get('split', 'test')}:"
        f"{row.get('algorithm')}:{row.get('requested_length')}:{row.get('sampler_seed')}:{row.get('sample_index', row_index)}"
    )


def _algorithm(row: dict[str, Any]) -> str:
    return str(row.get("algorithm", row.get("algo_name", "")))


def _requested_length(row: dict[str, Any]) -> int | None:
    value = row.get("requested_length", row.get("length"))
    return int(value) if value is not None else None


def _gold_inputs(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("inputs"), list):
        return list(row.get("inputs", []))
    native_target = row.get("native_input_target", {})
    if isinstance(native_target, dict) and isinstance(native_target.get("inputs"), list):
        return list(native_target.get("inputs", []))
    return []


def _question(row: dict[str, Any]) -> str:
    clrs_text = row.get("clrs_text", {})
    if isinstance(clrs_text, dict) and str(clrs_text.get("question", "")).strip():
        return str(clrs_text.get("question", ""))
    return str(row.get("question", ""))


def _input_schema(row: dict[str, Any]) -> list[dict[str, Any]]:
    schema = []
    for dp in _gold_inputs(row):
        schema.append(
            {
                "name": dp.get("name"),
                "location": dp.get("location"),
                "type": dp.get("type"),
                "shape": dp.get("shape"),
            }
        )
    return schema


def build_user_prompt(
    row: dict[str, Any],
    retry_instruction: str = "",
    *,
    hide_algorithm: bool = False,
    hide_native_schema: bool = False,
) -> str:
    payload = {
        "algorithm": None if hide_algorithm else _algorithm(row),
        "requested_length": _requested_length(row),
        "natural_language_description": _question(row),
    }
    if hide_native_schema:
        payload["native_input_schema_to_return"] = "not provided; infer the dm-clrs native inputs from the description"
    else:
        payload["native_input_schema_to_return"] = _input_schema(row)
    prompt = "Translate this CLRS natural-language prompt into native dm-clrs inputs.\n"
    if retry_instruction:
        prompt += f"\nRetry instruction:\n{retry_instruction}\n"
    prompt += "\nInstance JSON:\n" + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return prompt


def _response_dump(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        if isinstance(data, dict):
            return data
    return {}


def _extract_response_text_from_dump(response_dump: dict[str, Any]) -> str:
    pieces: list[str] = []
    for item in response_dump.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    pieces.append(text.strip())
    return "\n".join(pieces).strip()


def _tool_used_from_response(response: Any) -> bool:
    for item in _response_dump(response).get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "code_interpreter_call":
            return True
    return False


def parse_response_text(text: str) -> tuple[dict[str, Any], str]:
    raw = str(text or "").strip()
    if not raw:
        return {}, "empty_response"
    candidates = [raw]
    match = JSON_OBJECT_PATTERN.search(raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            inputs = payload.get("inputs", payload)
            if isinstance(inputs, dict):
                return inputs, "json" if candidate == raw else "embedded_json"
            if isinstance(inputs, list):
                by_name = {
                    str(item.get("name")): item.get("data")
                    for item in inputs
                    if isinstance(item, dict) and item.get("name") is not None
                }
                return by_name, "json" if candidate == raw else "embedded_json"
    return {}, "missing_inputs"


def _numeric_array(value: Any, shape: list[int]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if list(arr.shape) == list(shape):
        return arr
    if shape == [] and arr.size == 1:
        return np.asarray(float(arr.reshape(-1)[0]))
    target_size = int(np.prod(shape)) if shape else 1
    if arr.size == target_size:
        return arr.reshape(shape)
    raise ValueError(f"shape mismatch predicted={list(arr.shape)} expected={shape}")


def _score_input(gold_dp: dict[str, Any], pred_value: Any, *, atol: float) -> dict[str, Any]:
    shape = list(gold_dp.get("shape", []))
    gold = np.asarray(gold_dp.get("data"), dtype=float)
    pred = _numeric_array(pred_value, shape)
    if list(gold.shape) != list(pred.shape):
        pred = pred.reshape(gold.shape)

    is_integer_like = str(gold_dp.get("type")) in {"mask", "mask_one", "pointer", "categorical", "should_be_permutation"}
    if is_integer_like:
        gold_cmp = np.rint(gold).astype(int)
        pred_cmp = np.rint(pred).astype(int)
        correct_arr = pred_cmp == gold_cmp
        strict_arr = np.asarray(pred, dtype=float) == np.asarray(gold, dtype=float)
    else:
        correct_arr = np.isclose(pred, gold, atol=atol, rtol=0.0)
        strict_arr = np.asarray(pred, dtype=float) == np.asarray(gold, dtype=float)

    total = int(gold.size) if gold.shape else 1
    correct = int(np.sum(correct_arr)) if gold.shape else int(bool(correct_arr))
    strict_correct = int(np.sum(strict_arr)) if gold.shape else int(bool(strict_arr))
    max_abs_error = float(np.max(np.abs(np.asarray(pred, dtype=float) - np.asarray(gold, dtype=float)))) if total else 0.0
    return {
        "name": gold_dp.get("name"),
        "shape_ok": True,
        "correct": correct,
        "strict_correct": strict_correct,
        "total": total,
        "accuracy": correct / max(1, total),
        "strict_accuracy": strict_correct / max(1, total),
        "exact": bool(correct == total),
        "strict_exact": bool(strict_correct == total),
        "max_abs_error": max_abs_error if math.isfinite(max_abs_error) else None,
    }


def score_inputs(row: dict[str, Any], predicted_inputs: dict[str, Any], *, atol: float) -> tuple[dict[str, Any], str]:
    input_scores: dict[str, Any] = {}
    correct = 0
    strict_correct = 0
    total = 0
    exact_all = True
    strict_exact_all = True
    missing = []
    errors = []

    for gold_dp in _gold_inputs(row):
        name = str(gold_dp.get("name"))
        if name not in predicted_inputs:
            missing.append(name)
            expected_total = int(np.asarray(gold_dp.get("data")).size)
            total += expected_total
            exact_all = False
            strict_exact_all = False
            continue
        try:
            score = _score_input(gold_dp, predicted_inputs[name], atol=atol)
        except Exception as exc:
            expected_total = int(np.asarray(gold_dp.get("data")).size)
            total += expected_total
            errors.append(f"{name}:{type(exc).__name__}:{exc}")
            input_scores[name] = {"name": name, "shape_ok": False, "correct": 0, "strict_correct": 0, "total": expected_total}
            exact_all = False
            strict_exact_all = False
            continue
        input_scores[name] = score
        correct += int(score["correct"])
        strict_correct += int(score["strict_correct"])
        total += int(score["total"])
        exact_all = exact_all and bool(score["exact"])
        strict_exact_all = strict_exact_all and bool(score["strict_exact"])

    if missing:
        status = "missing_inputs:" + ",".join(missing)
    elif errors:
        status = "decode_errors"
    else:
        status = "scored"

    return (
        {
            "scored": status == "scored",
            "algorithm": _algorithm(row),
            "requested_length": row.get("requested_length"),
            "correct": correct,
            "strict_correct": strict_correct,
            "total": total,
            "element_accuracy": correct / max(1, total),
            "strict_element_accuracy": strict_correct / max(1, total),
            "row_exact": exact_all and status == "scored",
            "strict_row_exact": strict_exact_all and status == "scored",
            "inputs": input_scores,
            "missing": missing,
            "errors": errors,
        },
        status,
    )


def call_model(
    *,
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    max_output_tokens: int,
    container_id: str | None,
    memory_limit: str,
    tool_choice: str,
    retry_instruction: str = "",
    hide_algorithm: bool = False,
    hide_native_schema: bool = False,
) -> tuple[str, dict[str, Any], str, bool, str | None]:
    tools = [{"type": "code_interpreter", "container": container_id if container_id else {"type": "auto", "memory_limit": memory_limit}}]
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(
            row,
            retry_instruction=retry_instruction,
            hide_algorithm=hide_algorithm,
            hide_native_schema=hide_native_schema,
        ),
        tools=tools,
        tool_choice=tool_choice,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
    )
    response_dump = _response_dump(response)
    for item in response_dump.get("output", []) or []:
        if isinstance(item, dict):
            if item.get("type") == "code_interpreter_call":
                call_info = item.get("code_interpreter_call", {})
                if isinstance(call_info, dict) and call_info.get("input"):
                    print("\n--- GPT Code Interpreter Call ---")
                    print(call_info.get("input").strip())
                    print("---------------------------------")
            elif item.get("type") == "code_interpreter_output":
                out_info = item.get("code_interpreter_output", {})
                if isinstance(out_info, dict) and out_info.get("logs"):
                    print("\n--- GPT Code Interpreter Logs ---")
                    print(out_info.get("logs").strip())
                    print("---------------------------------")
    output_text = getattr(response, "output_text", "") or _extract_response_text_from_dump(response_dump)
    parsed, parse_status = parse_response_text(output_text)
    return output_text, parsed, parse_status, _tool_used_from_response(response), getattr(response, "id", None)


def _counter_finalize(counter: dict[str, Any]) -> dict[str, Any]:
    samples = int(counter.get("samples", 0))
    return {
        "samples": samples,
        "element_accuracy": counter.get("correct", 0) / max(1, counter.get("total", 0)),
        "strict_element_accuracy": counter.get("strict_correct", 0) / max(1, counter.get("total", 0)),
        "row_exact": counter.get("row_exact", 0) / max(1, samples),
        "strict_row_exact": counter.get("strict_row_exact", 0) / max(1, samples),
    }


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    parse_counts = Counter(str(row.get("parse_status", "")) for row in predictions)
    score_counts = Counter(str(row.get("score_status", "")) for row in predictions)
    by_task: dict[str, dict[str, Any]] = {}
    overall = {"samples": 0, "correct": 0, "strict_correct": 0, "total": 0, "row_exact": 0, "strict_row_exact": 0}
    tool_used = 0
    scored = 0
    for pred in predictions:
        if pred.get("tool_used"):
            tool_used += 1
        score = pred.get("score")
        if not isinstance(score, dict) or not score:
            continue
        if score.get("scored"):
            scored += 1
        task = str(pred.get("algorithm", score.get("algorithm", "")))
        for bucket in (overall, by_task.setdefault(task, {"samples": 0, "correct": 0, "strict_correct": 0, "total": 0, "row_exact": 0, "strict_row_exact": 0})):
            bucket["samples"] += 1
            bucket["correct"] += int(score.get("correct", 0))
            bucket["strict_correct"] += int(score.get("strict_correct", 0))
            bucket["total"] += int(score.get("total", 0))
            bucket["row_exact"] += int(bool(score.get("row_exact", False)))
            bucket["strict_row_exact"] += int(bool(score.get("strict_row_exact", False)))
    return {
        "num_examples": len(predictions),
        "num_scored": scored,
        "tool_used_rate": tool_used / max(1, len(predictions)),
        "parse_status_counts": dict(parse_counts),
        "score_status_counts": dict(score_counts),
        "samples_by_task": dict(sorted(Counter(str(row.get("algorithm", "")) for row in predictions).items())),
        "overall": _counter_finalize(overall),
        "by_task": {task: _counter_finalize(counter) for task, counter in sorted(by_task.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPT CLRS-Text to native dm-clrs input translation")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--task-filter", default="all")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--max-output-tokens", type=int, default=20000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--memory-limit", choices=["1g", "4g", "16g", "64g"], default="1g")
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    parser.add_argument("--numeric-atol", type=float, default=1e-3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--hide-algorithm", action="store_true")
    parser.add_argument("--hide-native-schema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key")

    rows = _select_rows(_read_jsonl(Path(args.input_jsonl)), _parse_csv_filter(args.task_filter), int(args.max_samples))
    print(f"[text-to-native] input_jsonl={args.input_jsonl}")
    print(f"[text-to-native] num_examples={len(rows)} model={args.model}")
    print(f"[text-to-native] output_json={args.output_json}")
    print(f"[text-to-native] metrics_json={args.metrics_json}")

    client = OpenAI(api_key=api_key, timeout=300.0)
    try:
        container = client.containers.create(name="clrs-text-to-native-code-interpreter", memory_limit=args.memory_limit)
    except TypeError:
        container = client.containers.create(name="clrs-text-to-native-code-interpreter")
    container_id = container.id
    print(f"[text-to-native] shared_container_id={container_id}")

    predictions: list[dict[str, Any]] = []
    iterator = enumerate(rows)
    if tqdm is not None:
        iterator = tqdm(list(iterator), desc="GPT CLRS-Text -> native")

    for row_index, row in iterator:
        retry_instruction = ""
        last: dict[str, Any] = {
            "response_text": "",
            "predicted_inputs": {},
            "parse_status": "missing_inputs",
            "score": {},
            "score_status": "missing_inputs",
            "tool_used": False,
            "response_id": None,
            "error": None,
        }
        for attempt in range(int(args.max_retries) + 1):
            try:
                response_text, predicted_inputs, parse_status, tool_used, response_id = call_model(
                    client=client,
                    model=str(args.model),
                    row=row,
                    max_output_tokens=int(args.max_output_tokens),
                    container_id=container_id,
                    memory_limit=str(args.memory_limit),
                    tool_choice=str(args.tool_choice),
                    retry_instruction=retry_instruction,
                    hide_algorithm=bool(args.hide_algorithm),
                    hide_native_schema=bool(args.hide_native_schema),
                )
                score, score_status = (
                    score_inputs(row, predicted_inputs, atol=float(args.numeric_atol))
                    if parse_status in SUCCESS_PARSE_STATUSES
                    else ({}, parse_status)
                )
                last = {
                    "response_text": response_text,
                    "predicted_inputs": predicted_inputs,
                    "parse_status": parse_status,
                    "score": score,
                    "score_status": score_status,
                    "tool_used": tool_used,
                    "response_id": response_id,
                    "error": None,
                }
                if parse_status in SUCCESS_PARSE_STATUSES and score_status == "scored":
                    break
                retry_instruction = f"Previous answer failed scoring ({score_status}). Return every requested native input name with data only."
            except Exception as exc:
                last["error"] = f"{type(exc).__name__}:{exc}"
                last["parse_status"] = f"api_error:{type(exc).__name__}"
                last["score_status"] = "api_error"
                if attempt < int(args.max_retries):
                    retry_instruction = f"Previous attempt failed with {type(exc).__name__}. Return only the required JSON object."
                    time.sleep(2.0 * (attempt + 1))
        predictions.append(
            {
                "sample_id": _sample_id(row, row_index),
                "row_index": row_index,
                "model": args.model,
                "algorithm": _algorithm(row),
                "requested_length": _requested_length(row),
                "sample_index": row.get("sample_index"),
                "sampler_seed": row.get("sampler_seed"),
                "input_schema": _input_schema(row),
                "clrs_text_question": _question(row),
                "predicted_inputs": last.get("predicted_inputs", {}),
                "parse_status": last.get("parse_status", ""),
                "score_status": last.get("score_status", ""),
                "score": last.get("score", {}),
                "tool_used": bool(last.get("tool_used", False)),
                "response_id": last.get("response_id"),
                "response_text": last.get("response_text") or last.get("error") or "",
            }
        )
        _write_json(Path(args.output_json), predictions)
        if float(args.sleep_seconds) > 0:
            time.sleep(float(args.sleep_seconds))

    metrics = {
        "input_jsonl": args.input_jsonl,
        "predictions_json": args.output_json,
        "model": args.model,
        "max_samples": int(args.max_samples),
        "numeric_atol": float(args.numeric_atol),
        "tool": "OpenAI Responses API Code Interpreter",
        "hide_algorithm": bool(args.hide_algorithm),
        "hide_native_schema": bool(args.hide_native_schema),
        "notes": [
            "GPT sees the CLRS-Text question and the native input schema only.",
            "Gold native inputs come from the paired dm-clrs sampler row.",
            "element_accuracy uses numeric tolerance; strict_element_accuracy requires exact numeric equality after JSON parsing.",
        ],
        "metrics": compute_metrics(predictions),
    }
    _write_json(Path(args.metrics_json), metrics)

    overall = metrics["metrics"]["overall"]
    print(
        "[summary]",
        f"scored={metrics['metrics']['num_scored']}/{metrics['metrics']['num_examples']}",
        f"tool_used={metrics['metrics']['tool_used_rate']:.4f}",
        f"element_accuracy={overall['element_accuracy']:.4f}",
        f"strict_element_accuracy={overall['strict_element_accuracy']:.4f}",
        f"row_exact={overall['row_exact']:.4f}",
        f"strict_row_exact={overall['strict_row_exact']:.4f}",
    )


if __name__ == "__main__":
    main()
