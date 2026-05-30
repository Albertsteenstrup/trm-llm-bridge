#!/usr/bin/env python3
"""Split a combined native CLRS JSONL into train/val/test files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split combined native CLRS raw rows by row['split']")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train,val,test,test_ood_size")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    split_counts: dict[str, int] = {split: 0 for split in requested}
    split_tasks: dict[str, Counter[str]] = {split: Counter() for split in requested}
    extra_counts: Counter[str] = Counter()

    with ExitStack() as stack:
        handles = {
            split: stack.enter_context((output_dir / f"{split}.jsonl").open("w", encoding="utf-8"))
            for split in requested
        }
        with input_jsonl.open("r", encoding="utf-8") as input_handle:
            for line in input_handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                split = str(row.get("split", "")).strip()
                if split in handles:
                    handles[split].write(json.dumps(row, ensure_ascii=True) + "\n")
                    split_counts[split] += 1
                    split_tasks[split][str(row.get("algorithm", ""))] += 1
                else:
                    extra_counts[split] += 1

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "splits": {},
        "ignored_splits": dict(extra_counts),
    }
    for split in requested:
        path = output_dir / f"{split}.jsonl"
        manifest["splits"][split] = {
            "path": str(path),
            "rows": split_counts[split],
            "tasks": dict(sorted(split_tasks[split].items())),
        }
        print(f"wrote split={split} rows={split_counts[split]} path={path}", flush=True)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
