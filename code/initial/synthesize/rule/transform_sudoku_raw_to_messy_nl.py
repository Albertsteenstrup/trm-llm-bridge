"""
Create intentionally messy but semantically correct Sudoku NL descriptions.

Goal:
- Stress translator generalization with noisy, heterogeneous writing styles.
- Preserve exact puzzle semantics (same givens) via strict round-trip verification.

Usage:
  python code/initial/synthesize/rule/transform_sudoku_raw_to_messy_nl.py \
    --input-json data/initial/sudoku_grid/sudoku_raw_1000_v3.json \
    --output-json data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_1000.json \
    --seed 42 --strict-verify
"""

import argparse
import json
import random
import re
from pathlib import Path


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
C_IN_ROW_PATTERN = re.compile(r"c\s*([1-9])\s*(?:=|is|has|contains|:|->)\s*([1-9])", flags=re.IGNORECASE)


def load_records(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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

    for r, c, v in ROW_COL_PATTERN.findall(text):
        triples.add((int(r), int(c), v))
    for r, c, v in RNCM_PATTERN.findall(text):
        triples.add((int(r), int(c), v))
    for r, c, v in BRACKET_PATTERN.findall(text):
        triples.add((int(r), int(c), v))
    for r, c, v in CELL_FUNC_PATTERN.findall(text):
        triples.add((int(r), int(c), v))
    for row_s, row_body in ROW_LINE_PATTERN.findall(text):
        row = int(row_s)
        for col_s, val_s in C_IN_ROW_PATTERN.findall(row_body):
            triples.add((row, int(col_s), val_s))

    return triples


def verify_description(puzzle: str, description: str):
    gold = set(given_triples_from_puzzle(puzzle))
    pred = extract_triples_from_description(description)
    if pred != gold:
        missing = sorted(gold - pred)
        extra = sorted(pred - gold)
        print(f"Missing: {missing}")
        print(f"Extra: {extra}")
        print(f"Description:\n{description}")
        raise ValueError(f"Round-trip mismatch (missing={len(missing)}, extra={len(extra)})")


def group_by_row(triples):
    rows = {}
    for row, col, val in triples:
        rows.setdefault(row, []).append((row, col, val))
    return rows


def group_by_col(triples):
    cols = {}
    for row, col, val in triples:
        cols.setdefault(col, []).append((row, col, val))
    return cols


def group_by_box(triples):
    boxes = {}
    for row, col, val in triples:
        box = ((row - 1) // 3) * 3 + ((col - 1) // 3) + 1
        boxes.setdefault(box, []).append((row, col, val))
    return boxes


def messy_clue(row: int, col: int, val: str, rng: random.Random):
    templates = [
        f"r{row}c{col}={val}",
        f"r{row}c{col} -> {val}",
        f"Row {row}, Col {col}: {val}",
        f"row{row} col{col} is {val}",
        f"[r{row},c{col}]={val}",
        f"cell(r{row},c{col})={val}",
        f"R{row}C{col}:{val}",
        f"row {row} col {col} = {val}",
        f"r{row} c{col} has {val}",
        f"row {row}, column {col} contains {val}",
        f"cell(r{row}/c{col})={val}",
        f"[r{row}/c{col}]:{val}",
    ]
    out = rng.choice(templates)
    if rng.random() < 0.2:
        out = out + rng.choice([" !", " ;", "  ", " #ok", " (checked)", " ?? no wait, yes", " -> confirmed"])
    return out


def row_style_clues(clues, rng: random.Random):
    style = rng.choice(["compact", "is", "arrow", "mixed", "has"])
    rendered = []
    for (_, c, v) in clues:
        if style == "compact":
            rendered.append(f"c{c}={v}")
        elif style == "is":
            rendered.append(f"c{c} is {v}")
        elif style == "arrow":
            rendered.append(f"c{c} -> {v}")
        elif style == "has":
            rendered.append(f"c{c} has {v}")
        else:
            rendered.append(rng.choice([f"c{c}={v}", f"c{c}:{v}", f"c{c} is {v}", f"c{c} contains {v}"]))
    return rendered


def render_row_block(rows: dict, row_ids: list[int], lines: list[str], rng: random.Random):
    for row in row_ids:
        clues = rows[row][:]
        if rng.random() < 0.5:
            clues.sort(key=lambda t: t[1])
        else:
            rng.shuffle(clues)
        clue_text = row_style_clues(clues, rng)
        joiner = rng.choice([", ", " ; ", " | ", "  ", " / ", " - "])
        row_prefix = rng.choice([f"Row {row}:", f"row{row} -", f"ROW {row}:", f"r{row}:", f"r{row} ->", f"R{row} "])
        lines.append(f"{row_prefix} {joiner.join(clue_text)}")
        if rng.random() < 0.12:
            lines.append(rng.choice(["(double-check this row)", "...", "ok next", "scribble", "looks right", "wait, is that right? yes", "moving on"] ))


def render_column_block(cols: dict, col_ids: list[int], lines: list[str], rng: random.Random):
    for col in col_ids:
        clues = cols[col][:]
        if rng.random() < 0.5:
            clues.sort(key=lambda t: t[0])
        else:
            rng.shuffle(clues)
        style = rng.choice(["rnc", "rowcol", "cell", "bracket"])
        clue_text = []
        for (r, c, v) in clues:
            if style == "rnc":
                clue_text.append(f"r{r}c{c}={v}")
            elif style == "rowcol":
                clue_text.append(f"Row {r}, Col {c}: {v}")
            elif style == "bracket":
                clue_text.append(f"[r{r},c{c}]={v}")
            else:
                clue_text.append(f"cell(r{r},c{c})={v}")
        joiner = rng.choice([", ", " ; ", " | ", "  ", " / ", " - "])
        col_prefix = rng.choice([f"Column {col}:", f"col{col}:", f"C{col} ->", f"c{col}:", f"COL {col} "]) 
        lines.append(f"{col_prefix} {joiner.join(clue_text)}")
        if rng.random() < 0.1:
            lines.append(rng.choice(["(col done)", "next col", "...", "check later"]))


def render_box_block(boxes: dict, box_ids: list[int], lines: list[str], rng: random.Random):
    for box in box_ids:
        clues = boxes[box][:]
        if rng.random() < 0.5:
            clues.sort(key=lambda t: (t[0], t[1]))
        else:
            rng.shuffle(clues)
        style = rng.choice(["rnc", "rowcol", "cell", "bracket"])
        clue_text = []
        for (r, c, v) in clues:
            if style == "rnc":
                clue_text.append(f"r{r}c{c}={v}")
            elif style == "rowcol":
                clue_text.append(f"Row {r}, Col {c}: {v}")
            elif style == "bracket":
                clue_text.append(f"[r{r},c{c}]={v}")
            else:
                clue_text.append(f"cell(r{r},c{c})={v}")
        joiner = rng.choice([", ", " ; ", " | ", "  ", " / ", " - "])
        box_prefix = rng.choice([f"Box {box}:", f"b{box}:", f"Subgrid {box} ->", f"BOX {box} "]) 
        lines.append(f"{box_prefix} {joiner.join(clue_text)}")
        if rng.random() < 0.1:
            lines.append(rng.choice(["(box done)", "next box", "...", "looks ok"]))


def render_flat_block(triples, lines: list[str], rng: random.Random):
    flat = triples[:]
    rng.shuffle(flat)
    for r, c, v in flat:
        lines.append(messy_clue(r, c, v, rng))
        if rng.random() < 0.08:
            lines.append(rng.choice(["typo? no", "keep going", "n/a", "line break", "wait, let me check", "looks good", "moving on", "almost done", "need coffee"]))


def render_band_notes(triples, lines: list[str], rng: random.Random):
    bands = {1: [], 2: [], 3: []}
    for r, c, v in triples:
        band = ((r - 1) // 3) + 1
        bands[band].append((r, c, v))
    
    band_ids = [1, 2, 3]
    rng.shuffle(band_ids)
    picked = band_ids[: rng.randint(1, 2)]
    lines.append(rng.choice(["band notes:", "horizontal bands:", "row-groups:"]))
    for band in picked:
        clues = bands[band][:]
        rng.shuffle(clues)
        use = clues[: rng.randint(2, min(8, len(clues)))]
        clue_text = [f"r{r}c{c}={v}" for (r, c, v) in use]
        lines.append(f"Band {band}: {', '.join(clue_text)}")


def render_stack_notes(triples, lines: list[str], rng: random.Random):
    stacks = {1: [], 2: [], 3: []}
    for r, c, v in triples:
        stack = ((c - 1) // 3) + 1
        stacks[stack].append((r, c, v))
    
    stack_ids = [1, 2, 3]
    rng.shuffle(stack_ids)
    picked = stack_ids[: rng.randint(1, 2)]
    lines.append(rng.choice(["stack notes:", "vertical stacks:", "col-groups:"]))
    for stack in picked:
        clues = stacks[stack][:]
        rng.shuffle(clues)
        use = clues[: rng.randint(2, min(8, len(clues)))]
        clue_text = [f"r{r}c{c}={v}" for (r, c, v) in use]
        lines.append(f"Stack {stack}: {', '.join(clue_text)}")


def render_subgrid_notes(triples, lines: list[str], rng: random.Random):
    boxes = group_by_box(triples)
    box_ids = sorted(boxes.keys())
    rng.shuffle(box_ids)
    picked = box_ids[: rng.randint(2, min(4, len(box_ids)))]
    lines.append(rng.choice(["subgrid spot-check:", "box notes:", "3x3 snapshot:"]))
    for box in picked:
        clues = boxes[box][:]
        rng.shuffle(clues)
        use = clues[: rng.randint(1, min(6, len(clues)))]
        clue_text = [f"r{r}c{c}={v}" for (r, c, v) in use]
        lines.append(f"Box {box}: {', '.join(clue_text)}")


def render_empty_row_notes(puzzle: str, lines: list[str], rng: random.Random):
    empties = {}
    for row in range(1, 10):
        cols = []
        for col in range(1, 10):
            ch = puzzle[(row - 1) * 9 + (col - 1)]
            if ch == ".":
                cols.append(col)
        if cols:
            empties[row] = cols

    if not empties:
        return

    lines.append(rng.choice(["empty-cell notes by row:", "row empties:", "blank slots by row:"]))
    row_ids = sorted(empties.keys())
    rng.shuffle(row_ids)
    for row in row_ids[: rng.randint(2, min(5, len(row_ids)))]:
        col_list = empties[row][:]
        col_list.sort()
        col_text = ",".join([f"c{c}" for c in col_list])
        lines.append(rng.choice([f"Row {row} empty -> {col_text}", f"Row {row}: blanks at {col_text}"]))


def render_empty_column_notes(puzzle: str, lines: list[str], rng: random.Random):
    empties = {}
    for col in range(1, 10):
        rows = []
        for row in range(1, 10):
            ch = puzzle[(row - 1) * 9 + (col - 1)]
            if ch == ".":
                rows.append(row)
        if rows:
            empties[col] = rows

    if not empties:
        return

    lines.append(rng.choice(["empty-cell notes by column:", "column empties:", "blank slots by column:"]))
    col_ids = sorted(empties.keys())
    rng.shuffle(col_ids)
    for col in col_ids[: rng.randint(2, min(5, len(col_ids)))]:
        row_list = empties[col][:]
        row_list.sort()
        row_text = ",".join([f"r{r}" for r in row_list])
        lines.append(rng.choice([f"Col {col} empty -> {row_text}", f"Column {col}: blanks at {row_text}"]))


VARIANTS = [
    ("rows_then_flat", ["rows", "flat"]),
    ("rows_then_columns", ["rows", "columns"]),
    ("columns_then_rows", ["columns", "rows"]),
    ("columns_then_flat", ["columns", "flat"]),
    ("flat_only", ["flat"]),
    ("rows_only", ["rows"]),
    ("columns_only", ["columns"]),
    ("boxes_only", ["boxes"]),
    ("boxes_then_flat", ["boxes", "flat"]),
    ("flat_then_boxes", ["flat", "boxes"]),
    ("rows_columns_flat", ["rows", "columns", "flat"]),
    ("columns_rows_flat", ["columns", "rows", "flat"]),
    ("flat_rows_columns", ["flat", "rows", "columns"]),
    ("flat_columns_rows", ["flat", "columns", "rows"]),
    ("rows_flat_rows", ["rows", "flat", "rows"]),
    ("columns_flat_columns", ["columns", "flat", "columns"]),
    ("rows_columns_rows", ["rows", "columns", "rows"]),
    ("columns_rows_columns", ["columns", "rows", "columns"]),
    ("flat_rows", ["flat", "rows"]),
    ("flat_columns", ["flat", "columns"]),
    ("rows_flat", ["rows", "flat"]),
    ("columns_flat", ["columns", "flat"]),
    ("rows_columns", ["rows", "columns"]),
    ("columns_rows", ["columns", "rows"]),
    ("rows_columns_flat_rows", ["rows", "columns", "flat", "rows"]),
    ("columns_rows_flat_columns", ["columns", "rows", "flat", "columns"]),
    ("flat_rows_columns_flat", ["flat", "rows", "columns", "flat"]),
    ("boxes_rows", ["boxes", "rows"]),
    ("rows_boxes", ["rows", "boxes"]),
    ("boxes_columns", ["boxes", "columns"]),
    ("columns_boxes", ["columns", "boxes"]),
    ("boxes_rows_flat", ["boxes", "rows", "flat"]),
    ("flat_boxes_columns", ["flat", "boxes", "columns"]),
    ("rows_boxes_columns", ["rows", "boxes", "columns"]),
    ("boxes_flat_boxes", ["boxes", "flat", "boxes"]),
    ("boxes_columns_rows", ["boxes", "columns", "rows"]),
    ("columns_boxes_rows", ["columns", "boxes", "rows"]),
    ("rows_columns_boxes", ["rows", "columns", "boxes"]),
    ("flat_boxes_rows_columns", ["flat", "boxes", "rows", "columns"]),
    ("boxes_flat_rows_columns", ["boxes", "flat", "rows", "columns"]),
    ("rows_boxes_flat_columns", ["rows", "boxes", "flat", "columns"]),
    ("columns_rows_boxes_flat", ["columns", "rows", "boxes", "flat"]),
    ("flat_flat", ["flat", "flat"]),
    ("rows_rows", ["rows", "rows"]),
    ("columns_columns", ["columns", "columns"]),
    ("boxes_boxes", ["boxes", "boxes"]),
]


def build_description(puzzle: str, rng: random.Random):
    triples = given_triples_from_puzzle(puzzle)
    rows = group_by_row(triples)
    cols = group_by_col(triples)
    boxes = group_by_box(triples)
    row_ids = list(rows.keys())
    col_ids = list(cols.keys())
    box_ids = list(boxes.keys())

    if rng.random() < 0.55:
        row_ids.sort()
    else:
        rng.shuffle(row_ids)
    if rng.random() < 0.55:
        col_ids.sort()
    else:
        rng.shuffle(col_ids)
    if rng.random() < 0.55:
        box_ids.sort()
    else:
        rng.shuffle(box_ids)

    header = rng.choice(
        [
            "notes dump / givens follow",
            "raw clue transcript (noisy):",
            "sudoku clues copied from sheet:",
            "messy annotations below, same puzzle",
            "working copy of givens (messy order):",
            "transcribed clue sheet // unstructured",
            "field notes from initial board:",
            "clues copied by hand, order may vary:",
            "brain dump of sudoku grid:",
            "initial state (unverified):",
            "puzzle givens (scrawled):",
            "SUDOKU GIVENS:",
            "--- start of clues ---",
            "board state:",
        ]
    )
    if rng.random() < 0.3:
        header = header.upper()
    lines = [header]

    variant_name, sections = rng.choice(VARIANTS)
    if rng.random() < 0.2:
        lines.append(f"variant={variant_name}")

    # To make it even messier, we can split the ids so that if a section appears multiple times, it gets different parts
    # But for now, we just render all of them. If they appear multiple times, they are duplicated.
    # Let's actually split them if they appear multiple times to avoid exact duplication, or just let them duplicate.
    # Duplication is fine for "messy".

    for section in sections:
        if section == "rows":
            prefix = rng.choice(["by rows:", "row pass:", "rowwise log:", "horizontal scan:"])
            if rng.random() < 0.3: prefix = prefix.upper()
            lines.append(prefix)
            render_row_block(rows, row_ids, lines, rng)
        elif section == "columns":
            prefix = rng.choice(["by columns:", "column pass:", "column-wise notes:", "vertical scan:"])
            if rng.random() < 0.3: prefix = prefix.upper()
            lines.append(prefix)
            render_column_block(cols, col_ids, lines, rng)
        elif section == "boxes":
            prefix = rng.choice(["by boxes:", "subgrid pass:", "3x3 blocks:", "box scan:"])
            if rng.random() < 0.3: prefix = prefix.upper()
            lines.append(prefix)
            render_box_block(boxes, box_ids, lines, rng)
        elif section == "flat":
            prefix = rng.choice(["flat clue list:", "unordered snippets:", "misc clue lines:", "random givens:"])
            if rng.random() < 0.3: prefix = prefix.upper()
            lines.append(prefix)
            render_flat_block(triples, lines, rng)

    if rng.random() < 0.4:
        render_empty_row_notes(puzzle, lines, rng)
    if rng.random() < 0.4:
        render_empty_column_notes(puzzle, lines, rng)
    if rng.random() < 0.4:
        render_subgrid_notes(triples, lines, rng)
    if rng.random() < 0.3:
        render_band_notes(triples, lines, rng)
    if rng.random() < 0.3:
        render_stack_notes(triples, lines, rng)

    if rng.random() < 0.4:
        lines.append(rng.choice(["done", "end of notes", "that's all givens", "EOF", "ready to solve"]))

    # Add some random blank lines for messiness
    final_lines = []
    for line in lines:
        final_lines.append(line)
        if rng.random() < 0.05:
            final_lines.append("")

    return "\n".join(final_lines)


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
    parser = argparse.ArgumentParser(description="Transform raw Sudoku to messy NL with semantic guarantees")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-verify", action="store_true")
    args = parser.parse_args()

    records = load_records(args.input_json)
    transformed = transform(records, args.seed, strict_verify=args.strict_verify)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(transformed, indent=2), encoding="utf-8")
    print(f"Wrote {len(transformed)} records to {args.output_json}")


if __name__ == "__main__":
    main()
