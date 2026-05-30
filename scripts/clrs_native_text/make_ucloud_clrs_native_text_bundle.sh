#!/usr/bin/env bash
set -euo pipefail

# Create a small upload bundle for UCloud. It contains the isolated CLRS
# native-text bridge code plus the non-stage2 trm_llm modules it depends on,
# but not large generated datasets/checkpoints.

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_DIR="${BUNDLE_DIR:-/private/tmp/clrs_native_text_ucloud_minimal_$$}"
BUNDLE_TAR="${BUNDLE_TAR:-clrs_native_text_ucloud_minimal.tar.gz}"

if [ -e "$BUNDLE_DIR" ]; then
  echo "Bundle dir already exists: $BUNDLE_DIR" >&2
  exit 1
fi
mkdir -p "$BUNDLE_DIR/thesis/code" "$BUNDLE_DIR/thesis/scripts" "$BUNDLE_DIR/thesis/docs"
mkdir -p "$BUNDLE_DIR/thesis/data/clrs_native_text/raw"
mkdir -p "$BUNDLE_DIR/thesis/data/clrs_native_text/prepared"
mkdir -p "$BUNDLE_DIR/thesis/data/clrs_native_text/test_splits"
mkdir -p "$BUNDLE_DIR/thesis/results/clrs_native_text"
mkdir -p "$BUNDLE_DIR/thesis/checkpoints/clrs_native_text"

rsync -a --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' code/clrs_native_text "$BUNDLE_DIR/thesis/code/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'stage2_bridge_v2/' code/trm_llm "$BUNDLE_DIR/thesis/code/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' scripts/clrs_native_text "$BUNDLE_DIR/thesis/scripts/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' docs/clrs_native_text "$BUNDLE_DIR/thesis/docs/"

# Include only tiny local examples/test prompts, not generated full datasets.
if [ -d data/clrs_native_text/test_splits ]; then
  rsync -a data/clrs_native_text/test_splits/ "$BUNDLE_DIR/thesis/data/clrs_native_text/test_splits/"
fi
if [ -f data/clrs_native_text/raw/paired_smoke_sorting.jsonl ]; then
  cp data/clrs_native_text/raw/paired_smoke_sorting.jsonl "$BUNDLE_DIR/thesis/data/clrs_native_text/raw/"
  cp data/clrs_native_text/raw/paired_smoke_sorting.metadata.json "$BUNDLE_DIR/thesis/data/clrs_native_text/raw/" 2>/dev/null || true
fi

tar -C "$BUNDLE_DIR" -czf "$BUNDLE_TAR" thesis
du -h "$BUNDLE_TAR"
