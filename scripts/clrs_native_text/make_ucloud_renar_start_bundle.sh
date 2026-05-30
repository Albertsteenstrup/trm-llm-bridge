#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${OUT:-$ROOT/renar_ucloud_start.tar.gz}"
PKG_ROOT="$ROOT/scripts/clrs_native_text/renar_ucloud_pkg"
RENAR_SRC="$ROOT/code/clrs_native_text/external/ReNAR"
RENAR_DST="$PKG_ROOT/renar_ucloud/ReNAR"

if [[ -d "$RENAR_SRC/ReNAR" ]]; then
  rm -rf "$RENAR_DST"
  mkdir -p "$RENAR_DST"
  tar --no-xattrs --disable-copyfile \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='*.ipynb' \
    -cf - -C "$RENAR_SRC" . | tar -xf - -C "$RENAR_DST"
else
  echo "Warning: ReNAR source not found at $RENAR_SRC; packaging bootstrap scripts only." >&2
fi

tar --no-xattrs --disable-copyfile \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='*.ipynb' \
  -czf "$OUT" -C "$PKG_ROOT" renar_ucloud
du -h "$OUT"
