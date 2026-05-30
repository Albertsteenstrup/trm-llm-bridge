"""
Correct bad Sudoku NL rows using an LLM and an evaluation report.

Expected evaluation input: output from code/initial/evaluate_sudoku_outputs.py

Example:
        python code/initial/synthesize/llm/correct_sudoku_nl_with_llm.py \
            --input-json data/initial/sudoku_synthetic/llm/sudoku_nl_multi_model.json \
            --evaluation-json logs/initial/eval_report_multi.json \
      --eval-file-index 0 \
            --output-json data/initial/sudoku_synthetic/llm/sudoku_nl_multi_model_corrected.json \
      --corrector-model qwen/qwen3-coder-480b-a35b-instruct
"""

import argparse
import json
import os
import time

from openai import APITimeoutError, OpenAI

from generate_sudoku_nl import _print_env_debug, format_grid


CORRECTOR_SYSTEM_PROMPT = """You correct Sudoku givens descriptions.

Return ONLY:
1) One short intro sentence.
2) A section title exactly: "Givens by row:"
3) One line per non-empty row, ascending order:
   "Row <r>: r<r>c<c>=<v>, r<r>c<c>=<v>, ..."

Rules:
- Mention every pre-filled cell exactly once.
- Do not mention empty cells.
- Do not add strategy or solution text.
"""


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _build_issue_lookup(eval_report: dict, eval_file_index: int):
    file_report = eval_report["files"][eval_file_index]
    lookup = {}
    for issue in file_report.get("issues", []):
        key = (issue.get("index"), issue.get("model"))
        lookup[key] = issue
    return lookup


def _record_key(record: dict):
    return record.get("index"), record.get("model")


def _hydrate_existing_corrections(records: list[dict], output_json_path: str):
    if not os.path.exists(output_json_path):
        return set()

    try:
        existing_records = _load_json(output_json_path)
    except Exception:
        return set()

    if not isinstance(existing_records, list):
        return set()

    existing_lookup = {
        _record_key(rec): rec.get("corrected_nl_description", "")
        for rec in existing_records
        if isinstance(rec, dict)
    }

    resumed_keys = set()

    for rec in records:
        key = _record_key(rec)
        existing_value = existing_lookup.get(key)
        if existing_value:
            rec["corrected_nl_description"] = existing_value
            resumed_keys.add(key)

    return resumed_keys


def _correct_one(client: OpenAI, record: dict, issue: dict, corrector_model: str) -> str:
    grid = format_grid(record["puzzle"])
    corrected = record.get("corrected_nl_description", "")
    original = corrected if isinstance(corrected, str) and corrected.strip() else record.get("nl_description", "")

    missing = issue.get("missing", [])
    hallucinated = issue.get("hallucinated", [])

    user_prompt = f"""Correct this Sudoku NL description.

Grid (underscores are empty):
{grid}

Original description:
{original}

Evaluation errors:
- missing entries: {missing}
- hallucinated entries: {hallucinated}

Produce a corrected description in strict row-wise format."""

    response = client.chat.completions.create(
        model=corrector_model,
        messages=[
            {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Correct Sudoku NL rows using LLM + evaluation report")
    parser.add_argument("--input-json", required=True, help="Generated dataset JSON")
    parser.add_argument("--evaluation-json", required=True, help="Evaluation report JSON from evaluate_sudoku_outputs.py")
    parser.add_argument("--eval-file-index", type=int, default=0, help="Index into evaluation report files[]")
    parser.add_argument("--output-json", required=True, help="Corrected output JSON")
    parser.add_argument("--corrector-model", default="qwen/qwen3-coder-480b-a35b-instruct", help="Model used for correction")
    parser.add_argument("--api-key", type=str, default=None, help="NVIDIA API key (or set NVIDIA_API_KEY env)")
    parser.add_argument("--target-model", type=str, default=None, help="Optional: only correct rows for this generated model")
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on number of rows to correct (0 means all)")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per row on API timeout")
    parser.add_argument("--debug-env", action="store_true", help="Print .env and API-key resolution debug info")
    args = parser.parse_args()

    if args.debug_env:
        _print_env_debug(args.api_key)

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY env variable or pass --api-key")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=args.request_timeout,
    )

    records = _load_json(args.input_json)
    eval_report = _load_json(args.evaluation_json)

    issue_lookup = _build_issue_lookup(eval_report, args.eval_file_index)

    corrected = 0
    processed = 0

    for rec in records:
        rec.setdefault("corrected_nl_description", "")

    resumed_keys = _hydrate_existing_corrections(records, args.output_json)
    _write_json(args.output_json, records)

    for rec in records:
        processed += 1

        key = _record_key(rec)
        issue = issue_lookup.get(key)

        should_correct = (
            issue is not None
            and (not args.target_model or rec.get("model") == args.target_model)
            and issue.get("status") in {"generation_error", "mismatch"}
        )

        if key in resumed_keys and rec.get("corrected_nl_description"):
            continue

        if should_correct:
            for attempt in range(1, args.max_retries + 1):
                try:
                    new_text = _correct_one(client, rec, issue, args.corrector_model)
                    rec["corrected_nl_description"] = new_text
                    corrected += 1
                    print(f"corrected index={rec.get('index')} model={rec.get('model')} ({processed}/{len(records)})")
                    break
                except APITimeoutError:
                    if attempt >= args.max_retries:
                        print(
                            f"timeout index={rec.get('index')} model={rec.get('model')} "
                            f"after {args.max_retries} attempts; keeping empty corrected_nl_description"
                        )
                    else:
                        backoff = 2 * attempt
                        print(
                            f"timeout index={rec.get('index')} model={rec.get('model')} "
                            f"attempt {attempt}/{args.max_retries}; retrying in {backoff}s"
                        )
                        time.sleep(backoff)

            time.sleep(0.2)

        _write_json(args.output_json, records)

        if args.max_items > 0 and corrected >= args.max_items:
            break

    print(f"Done. Processed {processed} rows, corrected {corrected}. Saved to {args.output_json}")


if __name__ == "__main__":
    main()
