#!/usr/bin/env python3
"""Benchmark GPT on no-algorithm CLRS native curriculum rows.

Two conditions are evaluated:

* translate: recover native dm-clrs input tensors from the natural language row.
* solve: solve the named algorithm and emit native dm-clrs output tensors.

The curriculum rows store gold native inputs only. For solve scoring, pass the
raw native JSONL used to build the curriculum so gold outputs can be joined back
by algorithm, split, length, and raw sample index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from benchmark_openai_text_to_native import (  # noqa: E402
    _input_schema,
    _question,
    _response_dump,
    _score_input,
    _tool_used_from_response,
    parse_response_text,
)


DEFAULT_CURRICULUM_DIR = "data/clrs_native_text/prepared/clrs30_no_algo_curriculum_v4_bridge_curriculum_subset10k"
DEFAULT_RAW_JSONL = "data/clrs_native_text/raw/clrs30_no_algo_curriculum_v4/clrs_native_raw.jsonl"
DEFAULT_OUTPUT_JSON = "results/clrs_native_text/openai_curriculum_native_gpt54_predictions.json"
DEFAULT_METRICS_JSON = "results/clrs_native_text/openai_curriculum_native_gpt54_metrics.json"
DEFAULT_MODEL = "gpt-5.4"
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


TRANSLATE_SYSTEM = """You translate CLRS natural-language instance descriptions into native dm-clrs input tensors.

You may use Code Interpreter to parse the prompt and construct arrays. Do not solve the algorithm and do not predict outputs or hints.

Return exactly one JSON object and nothing else:
{"inputs":{"<input_name>": <native_data>, "...": <native_data>}}

Native conventions:
- Use zero-based indexing.
- If an input named "pos" is requested, set pos[i] = i / N for N nodes/items.
- mask and mask_one values are 0/1.
- edge inputs are dense matrices with the requested shape.
- Preserve numeric values from the prompt exactly as numbers when they are shown.
"""


SOLVE_SYSTEM = """You solve CLRS algorithm instances and return native dm-clrs output tensors.

You may use Code Interpreter to parse the prompt, reconstruct the input object, run the algorithm carefully, and build arrays.

Return exactly one JSON object and nothing else:
{"outputs":{"<output_name>": <native_data>, "...": <native_data>}}

Native conventions:
- Use zero-based indexing.
- pointer outputs are integer node/item indices.
- mask and mask_one outputs are 0/1.
- scalar outputs are numbers.
- categorical outputs are integer class ids unless the requested schema explicitly asks for one-hot rows.
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


def _algorithm(row: dict[str, Any]) -> str:
    return str(row.get("algorithm", row.get("algo_name", "")))


def _level_path(curriculum_dir: Path, level: int, split: str) -> Path:
    return curriculum_dir / f"level_{level}" / f"{split}.jsonl"


def _select_rows(rows: list[dict[str, Any]], *, samples_per_task: int, max_tasks: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = _algorithm(row)
        if len(by_task[task]) < samples_per_task:
            by_task[task].append(row)
    tasks = sorted(by_task)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    selected: list[dict[str, Any]] = []
    for task in tasks:
        selected.extend(by_task[task])
    return selected


def _schema_for(datapoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": dp.get("name"),
            "location": dp.get("location"),
            "type": dp.get("type"),
            "shape": dp.get("shape"),
        }
        for dp in datapoints
    ]


def _gold_inputs(row: dict[str, Any]) -> list[dict[str, Any]]:
    target = row.get("native_input_target", {})
    if isinstance(target, dict) and isinstance(target.get("inputs"), list):
        return list(target["inputs"])
    return []


def _parse_sample_id(sample_id: str) -> dict[str, Any]:
    parts = str(sample_id).split(":")
    # no-algo:val:level_0:bfs:8:0:0:clean_semantic
    if len(parts) < 8:
        return {}
    return {
        "source_split": parts[1],
        "level": parts[2],
        "algorithm": parts[3],
        "requested_length": int(parts[4]),
        "raw_sample_index": int(parts[5]),
    }


def _raw_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("algorithm")),
        str(row.get("split")),
        int(row.get("requested_length", row.get("length", -1))),
        int(row.get("sample_index", -1)),
    )


def _curriculum_raw_key(row: dict[str, Any]) -> tuple[str, str, int, int] | None:
    parsed = _parse_sample_id(str(row.get("sample_id", "")))
    if not parsed:
        return None
    return (
        str(parsed["algorithm"]),
        str(parsed["source_split"]),
        int(parsed["requested_length"]),
        int(parsed["raw_sample_index"]),
    )


def _load_raw_outputs(path: Path | None) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    mapping: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        outputs = row.get("outputs")
        if isinstance(outputs, list) and outputs:
            mapping[_raw_key(row)] = list(outputs)
    return mapping


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return str(text)
    pieces: list[str] = []
    dump = _response_dump(response)
    for item in dump.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "output_text" and content.get("text"):
                pieces.append(str(content["text"]))
    return "\n".join(pieces).strip()


def _parse_mode_response(text: str, mode: str) -> tuple[dict[str, Any], str]:
    parsed, status = parse_response_text(text)
    key = "outputs" if mode == "solve" else "inputs"
    if status in {"json", "embedded_json"}:
        if isinstance(parsed, dict) and isinstance(parsed.get(key), dict):
            return dict(parsed[key]), status
        return parsed, status
    raw = str(text or "").strip()
    candidates = [raw]
    match = JSON_OBJECT_PATTERN.search(raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return dict(payload[key]), "json" if candidate == raw else "embedded_json"
    return {}, status


def _score_datapoints(gold: list[dict[str, Any]], predicted: dict[str, Any], *, atol: float) -> tuple[dict[str, Any], str]:
    input_scores: dict[str, Any] = {}
    correct = strict_correct = total = 0
    exact_all = strict_exact_all = True
    missing: list[str] = []
    errors: list[str] = []
    for dp in gold:
        name = str(dp.get("name"))
        expected_total = int(__import__("numpy").asarray(dp.get("data")).size)
        if name not in predicted:
            missing.append(name)
            total += expected_total
            exact_all = strict_exact_all = False
            continue
        try:
            score = _score_input(dp, predicted[name], atol=atol)
        except Exception as exc:
            errors.append(f"{name}:{type(exc).__name__}:{exc}")
            score = {"name": name, "shape_ok": False, "correct": 0, "strict_correct": 0, "total": expected_total, "exact": False, "strict_exact": False}
        input_scores[name] = score
        correct += int(score.get("correct", 0))
        strict_correct += int(score.get("strict_correct", 0))
        total += int(score.get("total", expected_total))
        exact_all = exact_all and bool(score.get("exact"))
        strict_exact_all = strict_exact_all and bool(score.get("strict_exact"))
    if missing:
        status = "missing:" + ",".join(missing)
    elif errors:
        status = "decode_errors"
    else:
        status = "scored"
    return {
        "scored": status == "scored",
        "correct": correct,
        "strict_correct": strict_correct,
        "total": total,
        "element_accuracy": correct / max(1, total),
        "strict_element_accuracy": strict_correct / max(1, total),
        "row_exact": status == "scored" and exact_all,
        "strict_row_exact": status == "scored" and strict_exact_all,
        "fields": input_scores,
        "missing": missing,
        "errors": errors,
    }, status


def _build_prompt(row: dict[str, Any], *, mode: str, gold_outputs: list[dict[str, Any]] | None, retry: str = "") -> str:
    if mode == "translate":
        payload = {
            "natural_language_description": _question(row),
            "native_input_schema_to_return": _input_schema(row),
        }
        task = "Translate this instance into native dm-clrs inputs."
    else:
        payload = {
            "algorithm": _algorithm(row),
            "natural_language_description": _question(row),
            "native_output_schema_to_return": _schema_for(gold_outputs or []),
        }
        task = "Solve this algorithm instance and return native dm-clrs outputs."
    prompt = task
    if retry:
        prompt += "\n\nRetry instruction:\n" + retry
    prompt += "\n\nInstance JSON:\n" + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return prompt


def _call_model(
    client: OpenAI,
    *,
    model: str,
    row: dict[str, Any],
    mode: str,
    gold_outputs: list[dict[str, Any]] | None,
    container_id: str | None,
    memory_limit: str,
    max_output_tokens: int,
    tool_choice: str,
    retry: str,
) -> tuple[str, dict[str, Any], str, bool, str | None]:
    tools = [{"type": "code_interpreter", "container": container_id if container_id else {"type": "auto", "memory_limit": memory_limit}}]
    response = client.responses.create(
        model=model,
        instructions=SOLVE_SYSTEM if mode == "solve" else TRANSLATE_SYSTEM,
        input=_build_prompt(row, mode=mode, gold_outputs=gold_outputs, retry=retry),
        tools=tools,
        tool_choice=tool_choice,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
    )
    text = _extract_response_text(response)
    parsed, status = _parse_mode_response(text, mode)
    return text, parsed, status, _tool_used_from_response(response), getattr(response, "id", None)


def _finalize(counter: dict[str, int]) -> dict[str, float | int]:
    samples = int(counter.get("samples", 0))
    total = int(counter.get("total", 0))
    return {
        "samples": samples,
        "element_accuracy": counter.get("correct", 0) / max(1, total),
        "strict_element_accuracy": counter.get("strict_correct", 0) / max(1, total),
        "row_exact": counter.get("row_exact", 0) / max(1, samples),
        "strict_row_exact": counter.get("strict_row_exact", 0) / max(1, samples),
    }


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    parse_counts = Counter()
    score_counts = Counter()
    for pred in predictions:
        key = f"level_{pred['level']}/{pred['mode']}"
        task_key = f"{key}/{pred['algorithm']}"
        parse_counts[f"{key}:{pred.get('parse_status')}"] += 1
        score_counts[f"{key}:{pred.get('score_status')}"] += 1
        score = pred.get("score") if isinstance(pred.get("score"), dict) else {}
        for bucket_name in (key, task_key):
            bucket = by_task[bucket_name] if bucket_name == task_key else buckets[bucket_name]
            bucket["samples"] += 1
            bucket["correct"] += int(score.get("correct", 0))
            bucket["strict_correct"] += int(score.get("strict_correct", 0))
            bucket["total"] += int(score.get("total", 0))
            bucket["row_exact"] += int(bool(score.get("row_exact", False)))
            bucket["strict_row_exact"] += int(bool(score.get("strict_row_exact", False)))
    return {
        "num_examples": len(predictions),
        "parse_status_counts": dict(parse_counts),
        "score_status_counts": dict(score_counts),
        "by_level_mode": {k: _finalize(v) for k, v in sorted(buckets.items())},
        "by_level_mode_task": {k: _finalize(v) for k, v in sorted(by_task.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-dir", default=DEFAULT_CURRICULUM_DIR)
    parser.add_argument("--raw-jsonl", default=DEFAULT_RAW_JSONL)
    parser.add_argument("--levels", default="0,4")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    parser.add_argument("--numeric-atol", type=float, default=1e-3)
    parser.add_argument("--max-output-tokens", type=int, default=20000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=0, help="Stop after this many new API calls; 0 means no cap.")
    parser.add_argument("--resume-existing", action="store_true", help="Load output-json, repair/rescore completed entries, and skip them.")
    parser.add_argument("--memory-limit", choices=["1g", "4g", "16g", "64g"], default="1g")
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def _gold_for_prediction(
    pred: dict[str, Any],
    row_lookup: dict[tuple[int, str], dict[str, Any]],
    raw_outputs: dict[tuple[str, str, int, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    row = row_lookup.get((int(pred.get("level", -1)), str(pred.get("sample_id", ""))))
    if row is None:
        return []
    if pred.get("mode") == "translate":
        return _gold_inputs(row)
    raw_key = _curriculum_raw_key(row)
    return raw_outputs.get(raw_key, []) if raw_key is not None else []


def _repair_existing_predictions(
    predictions: list[dict[str, Any]],
    row_lookup: dict[tuple[int, str], dict[str, Any]],
    raw_outputs: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    *,
    atol: float,
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        mode = str(pred.get("mode", ""))
        if mode not in {"translate", "solve"}:
            repaired.append(pred)
            continue
        gold = _gold_for_prediction(pred, row_lookup, raw_outputs)
        if not gold:
            repaired.append(pred)
            continue
        response_text = str(pred.get("response_text", ""))
        parsed, parse_status = _parse_mode_response(response_text, mode)
        if parse_status in {"json", "embedded_json"}:
            score, score_status = _score_datapoints(gold, parsed, atol=atol)
        else:
            score, score_status = {}, parse_status
        updated = dict(pred)
        updated.update({
            "parse_status": parse_status,
            "score_status": score_status,
            "score": score,
            "predicted": parsed,
        })
        repaired.append(updated)
    return repaired


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env or environment")

    curriculum_dir = Path(args.curriculum_dir)
    raw_outputs = _load_raw_outputs(Path(args.raw_jsonl) if args.raw_jsonl else None)
    selected: list[tuple[int, dict[str, Any]]] = []
    for level in [int(x) for x in str(args.levels).split(",") if x.strip()]:
        rows = _read_jsonl(_level_path(curriculum_dir, level, str(args.split)))
        for row in _select_rows(rows, samples_per_task=int(args.samples_per_task), max_tasks=int(args.max_tasks)):
            selected.append((level, row))
    row_lookup = {(level, str(row.get("sample_id", ""))): row for level, row in selected}

    planned_calls = len(selected) * 2
    print(f"[openai-curriculum] selected_rows={len(selected)} planned_calls={planned_calls} model={args.model}")
    print(f"[openai-curriculum] output_json={args.output_json}")
    print(f"[openai-curriculum] metrics_json={args.metrics_json}")

    predictions: list[dict[str, Any]] = []
    completed: set[tuple[int, str, str]] = set()
    if args.resume_existing and Path(args.output_json).is_file():
        loaded = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            predictions = _repair_existing_predictions(loaded, row_lookup, raw_outputs, atol=float(args.numeric_atol))
            for pred in predictions:
                completed.add((int(pred.get("level", -1)), str(pred.get("sample_id", "")), str(pred.get("mode", ""))))
            _write_json(Path(args.output_json), predictions)
            _write_json(Path(args.metrics_json), {"metrics": compute_metrics(predictions)})
            print(f"[openai-curriculum] resumed_existing={len(predictions)} repaired_and_skipped={len(completed)}")

    client = OpenAI(api_key=api_key, timeout=300.0)
    try:
        container = client.containers.create(name="clrs-curriculum-native-code-interpreter", memory_limit=str(args.memory_limit))
    except TypeError:
        container = client.containers.create(name="clrs-curriculum-native-code-interpreter")
    container_id = container.id
    print(f"[openai-curriculum] shared_container_id={container_id}")

    new_calls = 0
    iterator = selected
    if tqdm is not None:
        iterator = tqdm(selected, desc="GPT curriculum native")

    for level, row in iterator:
        raw_key = _curriculum_raw_key(row)
        gold_outputs = raw_outputs.get(raw_key) if raw_key is not None else None
        for mode in ("translate", "solve"):
            pred_key = (level, str(row.get("sample_id", "")), mode)
            if pred_key in completed:
                continue
            if int(args.max_calls) > 0 and new_calls >= int(args.max_calls):
                payload = {
                    "model": args.model,
                    "curriculum_dir": str(curriculum_dir),
                    "raw_jsonl": str(args.raw_jsonl),
                    "levels": args.levels,
                    "split": args.split,
                    "samples_per_task": int(args.samples_per_task),
                    "numeric_atol": float(args.numeric_atol),
                    "predictions_json": args.output_json,
                    "new_calls": new_calls,
                    "metrics": compute_metrics(predictions),
                }
                _write_json(Path(args.metrics_json), payload)
                print(f"[openai-curriculum] max_calls reached new_calls={new_calls}")
                print(json.dumps(payload["metrics"]["by_level_mode"], indent=2))
                return
            gold = _gold_inputs(row) if mode == "translate" else (gold_outputs or [])
            if mode == "solve" and not gold:
                predictions.append({
                    "level": level,
                    "mode": mode,
                    "sample_id": row.get("sample_id"),
                    "algorithm": _algorithm(row),
                    "parse_status": "skipped",
                    "score_status": "no_gold_outputs",
                    "score": {},
                    "response_text": "",
                })
                completed.add(pred_key)
                continue
            last: dict[str, Any] = {"parse_status": "", "score_status": "", "score": {}, "response_text": "", "predicted": {}, "tool_used": False, "response_id": None}
            retry = ""
            for attempt in range(int(args.max_retries) + 1):
                try:
                    response_text, predicted, parse_status, tool_used, response_id = _call_model(
                        client,
                        model=str(args.model),
                        row=row,
                        mode=mode,
                        gold_outputs=gold_outputs,
                        container_id=container_id,
                        memory_limit=str(args.memory_limit),
                        max_output_tokens=int(args.max_output_tokens),
                        tool_choice=str(args.tool_choice),
                        retry=retry,
                    )
                    score, score_status = _score_datapoints(gold, predicted, atol=float(args.numeric_atol)) if parse_status in {"json", "embedded_json"} else ({}, parse_status)
                    last = {
                        "parse_status": parse_status,
                        "score_status": score_status,
                        "score": score,
                        "response_text": response_text,
                        "predicted": predicted,
                        "tool_used": tool_used,
                        "response_id": response_id,
                    }
                    if score_status == "scored":
                        break
                    retry = f"Previous answer failed scoring ({score_status}). Return every requested {'output' if mode == 'solve' else 'input'} name with data only."
                except Exception as exc:
                    last = {"parse_status": f"api_error:{type(exc).__name__}", "score_status": "api_error", "score": {}, "response_text": f"{type(exc).__name__}:{exc}", "predicted": {}, "tool_used": False, "response_id": None}
                    retry = f"Previous attempt failed with {type(exc).__name__}. Return only the required JSON object."
                    if attempt < int(args.max_retries):
                        time.sleep(2.0 * (attempt + 1))
            new_calls += 1
            predictions.append({
                "level": level,
                "mode": mode,
                "sample_id": row.get("sample_id"),
                "algorithm": _algorithm(row),
                "nl_style": row.get("nl_style"),
                "num_nodes": row.get("num_nodes"),
                "schema": _schema_for(gold),
                "parse_status": last["parse_status"],
                "score_status": last["score_status"],
                "score": last["score"],
                "tool_used": last["tool_used"],
                "response_id": last["response_id"],
                "predicted": last["predicted"],
                "response_text": last["response_text"],
            })
            completed.add(pred_key)
            _write_json(Path(args.output_json), predictions)
            _write_json(Path(args.metrics_json), {"metrics": compute_metrics(predictions)})
            if float(args.sleep_seconds) > 0:
                time.sleep(float(args.sleep_seconds))

    payload = {
        "model": args.model,
        "curriculum_dir": str(curriculum_dir),
        "raw_jsonl": str(args.raw_jsonl),
        "levels": args.levels,
        "split": args.split,
        "samples_per_task": int(args.samples_per_task),
        "numeric_atol": float(args.numeric_atol),
        "predictions_json": args.output_json,
        "metrics": compute_metrics(predictions),
    }
    _write_json(Path(args.metrics_json), payload)
    print(json.dumps(payload["metrics"]["by_level_mode"], indent=2))


if __name__ == "__main__":
    main()
