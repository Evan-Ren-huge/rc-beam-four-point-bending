#!/usr/bin/env ksh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
: "${TRAIN_ROOT:?Set TRAIN_ROOT to the training dataset directory}"
CASE_ID=${CASE_ID:-Job-new0a0b0}
FIELD=${FIELD:-stress}
FPS=${FPS:-5}
DPI=${DPI:-120}

exec "$PYTHON_BIN" "$BENCHMARK_ROOT/viz/compare_models.py" \
  --train-root "$TRAIN_ROOT" \
  --case "$CASE_ID" \
  --field "$FIELD" \
  --fps "$FPS" \
  --dpi "$DPI"

