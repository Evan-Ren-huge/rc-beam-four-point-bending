#!/usr/bin/env ksh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

: "${PYTHON_BIN:=python3}"
: "${TRAIN_ROOT:?Set TRAIN_ROOT to the directory containing the three rollout folders}"
: "${CASE_ID:=Job-new0a0b0}"
: "${FIELD:=stress}"
: "${OUT_DIR:=${BENCHMARK_ROOT}/outputs}"

"${PYTHON_BIN}" "${BENCHMARK_ROOT}/viz/compare_models_3d.py" \
  --train-root "${TRAIN_ROOT}" \
  --case "${CASE_ID}" \
  --field "${FIELD}" \
  --gif "${OUT_DIR}/${CASE_ID}-three-model-${FIELD}-3d.gif" \
  --png "${OUT_DIR}/${CASE_ID}-three-model-${FIELD}-3d.png"

print "Wrote 3D GIF and PNG to ${OUT_DIR}"
