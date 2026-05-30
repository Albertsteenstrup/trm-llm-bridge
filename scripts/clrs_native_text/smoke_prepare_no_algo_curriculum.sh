#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ucloud/bin/python}"

RUN_NAME=smoke_no_algo_curriculum \
TARGET_TOTAL=60 \
TEST_OOD_TOTAL=30 \
TRAIN_LENGTHS=8 \
VAL_LENGTHS=8 \
TEST_LENGTHS=16 \
TEST_OOD_LENGTHS=16 \
ALGORITHMS=bfs,dijkstra,insertion_sort,kmp_matcher,graham_scan \
MAX_ROWS_PER_SPLIT=20 \
VARIANTS_PER_ROW=1 \
"$ROOT/scripts/clrs_native_text/prepare_b200_no_algo_curriculum.sh"

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
base = Path("data/clrs_native_text/prepared/smoke_no_algo_curriculum_bridge_curriculum")
for path in sorted(base.glob("level_*/*.jsonl")):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    first = rows[0] if rows else {}
    print(path, len(rows), "bridge_task_label=", first.get("bridge_task_label"), "algo_name=", first.get("algo_name"))
    if rows:
        print(first["question"][:260].replace("\n", " | "))
ood = base / "ood/test_ood_size.jsonl"
rows = [json.loads(line) for line in ood.open() if line.strip()]
print(ood, len(rows), rows[0]["question"][:260].replace("\n", " | "))
PY
