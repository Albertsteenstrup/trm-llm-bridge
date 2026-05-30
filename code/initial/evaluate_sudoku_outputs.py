"""
Evaluate one or more generated Sudoku NL JSON files.

Examples:
    python code/initial/evaluate_sudoku_outputs.py --files data/initial/sudoku_synthetic/llm/sudoku_nl_dataset.json
    python code/initial/evaluate_sudoku_outputs.py --files data/initial/sudoku_synthetic/rule/sudoku_nl_diverse.json data/initial/sudoku_synthetic/llm/sudoku_nl_multi_model.json --output logs/initial/eval_report_multi.json
"""

import argparse
import json
import os
import re
from pathlib import Path

try:
    from synthesize.llm.generate_sudoku_nl import (
        _extract_triples_from_description,
        _given_triples_from_puzzle,
        evaluate_generated_dataset,
    )
except ImportError:
    from generate_sudoku_nl import (  # type: ignore
        _extract_triples_from_description,
        _given_triples_from_puzzle,
        evaluate_generated_dataset,
    )


def _load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    return data


def _effective_description(rec: dict) -> str:
    corrected = rec.get("corrected_nl_description")
    if isinstance(corrected, str) and corrected.strip():
        return corrected
    return rec.get("nl_description", "")


def _records_with_effective_description(records):
    effective_records = []
    for rec in records:
        rec_copy = dict(rec)
        rec_copy["nl_description"] = _effective_description(rec)
        effective_records.append(rec_copy)
    return effective_records


ROW_COL_PATTERN = re.compile(
    r"row\s*([1-9])\s*[,;:\-\s]*col(?:umn)?\s*([1-9])\s*(?:=|is|has|contains|:|->)\s*([1-9])",
    flags=re.IGNORECASE,
)
RNCM_PATTERN = re.compile(
    r"r\s*([1-9])\s*c\s*([1-9])\s*(?:=|is|has|contains|:|->)\s*([1-9])",
    flags=re.IGNORECASE,
)
BRACKET_PATTERN = re.compile(
    r"\[\s*r\s*([1-9])\s*[,/]\s*c\s*([1-9])\s*\]\s*(?:=|:|->)\s*([1-9])",
    flags=re.IGNORECASE,
)
CELL_FUNC_PATTERN = re.compile(
    r"cell\s*\(\s*r\s*([1-9])\s*[,/]\s*c\s*([1-9])\s*\)\s*(?:=|:|->)\s*([1-9])",
    flags=re.IGNORECASE,
)
ROW_LINE_PATTERN = re.compile(r"(?:row|r)\s*([1-9])\s*[:\-\s>]+([^\n]+)", flags=re.IGNORECASE)
C_IN_ROW_PATTERN = re.compile(r"c\s*([1-9])\s*(?:=|is|contains|has|:|->)\s*([1-9])", flags=re.IGNORECASE)


def _extract_triples_robust(text: str):
    triples = set()
    if not isinstance(text, str):
        return triples

    for row_s, col_s, val_s in ROW_COL_PATTERN.findall(text):
        triples.add((int(row_s), int(col_s), val_s))
    for row_s, col_s, val_s in RNCM_PATTERN.findall(text):
        triples.add((int(row_s), int(col_s), val_s))
    for row_s, col_s, val_s in BRACKET_PATTERN.findall(text):
        triples.add((int(row_s), int(col_s), val_s))
    for row_s, col_s, val_s in CELL_FUNC_PATTERN.findall(text):
        triples.add((int(row_s), int(col_s), val_s))
    for row_s, row_body in ROW_LINE_PATTERN.findall(text):
        row = int(row_s)
        for col_s, val_s in C_IN_ROW_PATTERN.findall(row_body):
            triples.add((row, int(col_s), val_s))

    return triples


def _canonical_description_from_triples(triples):
    by_row = {}
    for row, col, val in triples:
        by_row.setdefault(row, []).append((col, val))

    lines = ["Given entries:"]
    for row in sorted(by_row.keys()):
        clues = sorted(by_row[row], key=lambda x: x[0])
        clue_str = ", ".join([f"c{col}={val}" for col, val in clues])
        lines.append(f"Row {row}: {clue_str}")
    return "\n".join(lines)


def _records_with_parser_ready_descriptions(records, parser_mode: str):
    base = _records_with_effective_description(records)
    if parser_mode != "robust":
        return base

    out = []
    for rec in base:
        rec_copy = dict(rec)
        desc = rec_copy.get("nl_description", "")
        triples = _extract_triples_robust(desc)
        if triples:
            rec_copy["nl_description"] = _canonical_description_from_triples(triples)
        out.append(rec_copy)
    return out


def _entry_level_issues(records, extract_fn):
    issues = []
    for rec in records:
        idx = rec.get("index")
        puzzle = rec.get("puzzle")
        description = _effective_description(rec)
        model = rec.get("model")

        if not isinstance(puzzle, str) or len(puzzle) != 81:
            continue

        if isinstance(description, str) and description.startswith("ERROR:"):
            issues.append(
                {
                    "index": idx,
                    "model": model,
                    "status": "generation_error",
                    "error": description,
                }
            )
            continue

        gold = _given_triples_from_puzzle(puzzle)
        pred = extract_fn(description)

        missing = sorted(list(gold - pred))
        hallucinated = sorted(list(pred - gold))

        if missing or hallucinated:
            issues.append(
                {
                    "index": idx,
                    "model": model,
                    "status": "mismatch",
                    "missing": [{"row": r, "col": c, "val": v} for (r, c, v) in missing],
                    "hallucinated": [{"row": r, "col": c, "val": v} for (r, c, v) in hallucinated],
                }
            )

    return issues


def _split_by_model(records):
    buckets = {}
    for rec in records:
        model = rec.get("model", "__single_model__")
        buckets.setdefault(model, []).append(rec)
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more Sudoku NL output JSON files")
    parser.add_argument("--files", nargs="+", required=True, help="One or more output JSON files")
    parser.add_argument("--output", type=str, default="logs/initial/eval_report_multi.json", help="Combined evaluation report JSON")
    parser.add_argument("--parser", choices=["standard", "robust"], default="standard", help="Parser mode for extracting givens from NL")
    args = parser.parse_args()

    extract_fn = _extract_triples_robust if args.parser == "robust" else _extract_triples_from_description

    report = {
        "files": [],
    }

    for path in args.files:
        records = _load_json(path)
        effective_records = _records_with_parser_ready_descriptions(records, parser_mode=args.parser)
        file_entry = {
            "file": path,
            "total_records": len(records),
            "overall": evaluate_generated_dataset(effective_records, reference_puzzles=records),
            "issues": _entry_level_issues(records, extract_fn=extract_fn),
            "by_model": {},
            "parser": args.parser,
        }

        by_model = _split_by_model(effective_records)
        for model, model_records in by_model.items():
            file_entry["by_model"][model] = {
                "count": len(model_records),
                "metrics": evaluate_generated_dataset(model_records, reference_puzzles=model_records),
                "issues": _entry_level_issues(model_records, extract_fn=extract_fn),
            }

        report["files"].append(file_entry)

        overall = file_entry["overall"]
        print(f"\nFile: {path}")
        print(f"Grounding F1: {overall['grounding_f1']:.4f}")
        print(f"Exact match rate: {overall['exact_match_rate']:.4f}")
        print(f"Hallucination entry rate: {overall['hallucination_entry_rate']:.4f}")
        print(f"Missing entry rate: {overall['missing_entry_rate']:.4f}")

        if len(file_entry["by_model"]) > 1:
            print("Per model:")
            for model_name, model_summary in file_entry["by_model"].items():
                mm = model_summary["metrics"]
                print(f"  - {model_name}: F1={mm['grounding_f1']:.4f}, exact={mm['exact_match_rate']:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
