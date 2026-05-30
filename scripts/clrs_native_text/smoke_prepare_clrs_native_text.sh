#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ucloud/bin/python}"

RUN_NAME=smoke_clrs_native_text \
TARGET_TOTAL=60 \
TEST_OOD_TOTAL=30 \
TRAIN_LENGTHS=8 \
VAL_LENGTHS=8 \
TEST_LENGTHS=16 \
TEST_OOD_LENGTHS=16 \
ALGORITHMS=bfs,dijkstra,insertion_sort,kmp_matcher,graham_scan \
MAX_ROWS_PER_SPLIT=20 \
VARIANTS_PER_ROW=1 \
"$ROOT/scripts/clrs_native_text/prepare_b200_clrs_bridge_specialist.sh"

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path("data/clrs_native_text/prepared")
for path in sorted((root / "smoke_clrs_native_text_native").glob("*.jsonl")):
    print(path, sum(1 for _ in path.open()))
for path in sorted((root / "smoke_clrs_native_text_bridge").glob("*.jsonl")):
    first = next(path.open()).strip() if path.stat().st_size else ""
    print(path, sum(1 for _ in path.open()), first[:180])
PY
