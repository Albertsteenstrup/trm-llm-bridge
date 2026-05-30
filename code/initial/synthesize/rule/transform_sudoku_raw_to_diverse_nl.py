"""
Create diverse Sudoku natural-language descriptions with exact semantic preservation.

Method (no LLM in the loop):
1) Read each raw puzzle string.
2) Convert exact givens into varied, parser-compatible NL phrasings.
3) Round-trip validate NL -> extracted triples == puzzle givens.

This keeps task identity unchanged (same puzzle/solution) while reducing repetitive wording.

Example:
    python code/initial/synthesize/rule/transform_sudoku_raw_to_diverse_nl.py \
      --input-json data/initial/sudoku/test/sudoku_raw_test_n1000_seed42_round1.json \
      --output-json data/initial/sudoku_synthetic/rule/sudoku_nl_diverse.json \
      --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path


ROW_LINE_PATTERN = re.compile(r"row\s*([1-9])\s*:\s*([^\n]+)", flags=re.IGNORECASE)
C_IN_ROW_PATTERN = re.compile(r"c\s*([1-9])\s*(?:=|is|contains|has|:)\s*([1-9])", flags=re.IGNORECASE)


def load_records(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def given_triples_from_puzzle(puzzle: str):
    if not isinstance(puzzle, str) or len(puzzle) != 81:
        raise ValueError("Invalid puzzle string; expected 81-char string")

    triples = []
    for idx, ch in enumerate(puzzle):
        if ch != ".":
            row = idx // 9 + 1
            col = idx % 9 + 1
            triples.append((row, col, ch))
    return triples


def extract_triples_from_description(text: str):
    triples = set()
    if not isinstance(text, str):
        return triples

    for row_s, row_body in ROW_LINE_PATTERN.findall(text):
        row = int(row_s)
        for col_s, val_s in C_IN_ROW_PATTERN.findall(row_body):
            triples.add((row, int(col_s), val_s))
    return triples


def group_by_row(triples):
    rows = {}
    for row, col, val in triples:
        rows.setdefault(row, []).append((row, col, val))
    return rows


def choose_row_order(rows: dict, rng: random.Random):
    mode = rng.choice(["asc", "desc", "dense", "sparse", "random"])
    row_ids = list(rows.keys())

    if mode == "asc":
        row_ids.sort()
    elif mode == "desc":
        row_ids.sort(reverse=True)
    elif mode == "dense":
        row_ids.sort(key=lambda r: (-len(rows[r]), r))
    elif mode == "sparse":
        row_ids.sort(key=lambda r: (len(rows[r]), r))
    else:
        rng.shuffle(row_ids)

    return row_ids


def clue_phrase(col: int, val: str, rng: random.Random):
    templates = [
        f"c{col}={val}",
        f"c{col} is {val}",
        f"c{col} contains {val}",
        f"c{col} has {val}",
        f"c{col}: {val}",
    ]
    return rng.choice(templates)


def build_description(puzzle: str, rng: random.Random):
    triples = given_triples_from_puzzle(puzzle)
    rows = group_by_row(triples)

    intro_options = [
        "Here are the fixed clues for this Sudoku instance. They define the exact starting state and should be read as the authoritative givens.",
        "This Sudoku starts with the following givens. Every listed entry is already present in the grid before any solving begins.",
        "Use these preset entries as the puzzle's starting point. Together they specify the same initial board configuration as the raw puzzle string.",
        "Below is the complete list of pre-filled cells. No additional givens are implied beyond what is explicitly written.",
        "These are the clues already present in the grid. They form the fixed anchors from which the remaining cells must be deduced.",
        "The puzzle is defined by the givens listed below. Treat each row line as part of one consistent initial Sudoku state.",
        "Only the following cells are pre-filled. All other positions are intentionally left open for reasoning during solving.",
        "The initial Sudoku clues are listed here. Each clue corresponds directly to a fixed value in the starting board.",
    ]

    section_options = [
        "Given entries:",
        "Preset cells:",
        "Clues by position:",
        "Initial filled cells:",
        "Starting clues:",
    ]

    middle_options = [
        "These fixed values continue to constrain the same starting grid and do not introduce a different puzzle state.",
        "The next lines add more givens that belong to the same initial board configuration.",
        "What follows is still part of the exact same set of preset clues, simply presented in continuing row blocks.",
        "The clue list continues below with additional fixed entries from the same starting position.",
        "Continuing the inventory of givens, the remaining rows preserve the original puzzle definition.",
        "More pre-filled cells are listed next; they are consistent with the same underlying Sudoku instance.",
    ]

    ending_options = [
        "Taken together, these lines fully describe the givens of the original puzzle and nothing else.",
        "This completes the clue inventory for the initial board; all unlisted cells remain empty at the start.",
        "These entries close the full set of starting givens for this Sudoku instance.",
        "The list above finishes the fixed clues that define the puzzle's initial state.",
        "No further preset values are assumed beyond the givens recorded above.",
        "This is the complete pre-filled configuration from which solving should proceed.",
    ]

    separator_options = [", ", "; ", " | ", " • "]
    suffix_options = ["", "", "."]

    row_ids = choose_row_order(rows, rng)

    intro_prob = 0.85
    middle_prob = 0.45
    ending_prob = 0.4

    lines = []
    if rng.random() < intro_prob:
        lines.append(rng.choice(intro_options))
        lines.append("")

    lines.append(rng.choice(section_options))

    insertion_points = set()
    if len(row_ids) > 1:
        if rng.random() < middle_prob:
            mid = len(row_ids) // 2 - 1
            insertion_points.add(max(0, mid))
        if len(row_ids) >= 6 and rng.random() < middle_prob * 0.5:
            second = rng.randint(0, len(row_ids) - 2)
            insertion_points.add(second)

    for pos, row in enumerate(row_ids):
        clues = list(rows[row])
        if rng.random() < 0.5:
            clues.sort(key=lambda t: t[1])
        else:
            rng.shuffle(clues)

        clue_texts = [clue_phrase(c, v, rng) for (_, c, v) in clues]
        sep = rng.choice(separator_options)
        suffix = rng.choice(suffix_options)
        lines.append(f"Row {row}: {sep.join(clue_texts)}{suffix}")

        if pos in insertion_points and pos < len(row_ids) - 1:
            lines.append(rng.choice(middle_options))

    if rng.random() < ending_prob:
        lines.extend(["", rng.choice(ending_options)])

    return "\n".join(lines)


def verify_description(puzzle: str, description: str):
    gold = set(given_triples_from_puzzle(puzzle))
    pred = extract_triples_from_description(description)
    if pred != gold:
        missing = sorted(gold - pred)
        extra = sorted(pred - gold)
        raise ValueError(
            "Description failed round-trip validation "
            f"(missing={len(missing)}, extra={len(extra)})."
        )


def transform(records, seed: int, strict_verify: bool):
    output = []

    for idx, rec in enumerate(records):
        puzzle = rec.get("puzzle")
        solution = rec.get("solution")
        rating = rec.get("rating")

        if not isinstance(puzzle, str) or len(puzzle) != 81:
            raise ValueError(f"Invalid puzzle at index {idx}")
        if not isinstance(solution, str) or len(solution) != 81:
            raise ValueError(f"Invalid solution at index {idx}")

        rng = random.Random((seed * 1_000_003) + idx)
        nl_description = build_description(puzzle, rng)

        if strict_verify:
            verify_description(puzzle, nl_description)

        output.append(
            {
                "index": idx,
                "puzzle": puzzle,
                "solution": solution,
                "rating": rating,
                "nl_description": nl_description,
                "corrected_nl_description": "",
            }
        )

    return output


def main():
    parser = argparse.ArgumentParser(description="Transform raw Sudoku to diverse NL with semantic guarantees")
    parser.add_argument("--input-json", required=True, help="Raw Sudoku JSON")
    parser.add_argument("--output-json", required=True, help="Output NL JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Round-trip validate each description against puzzle givens",
    )
    args = parser.parse_args()

    records = load_records(args.input_json)
    transformed = transform(records, args.seed, strict_verify=args.strict_verify)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(transformed, f, indent=2)

    print(f"Wrote {len(transformed)} records to {args.output_json}")


if __name__ == "__main__":
    main()
