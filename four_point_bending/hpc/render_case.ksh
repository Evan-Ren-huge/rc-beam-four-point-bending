#!/usr/bin/env ksh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

: "${PYTHON_BIN:=python3}"
: "${H5_ROOT:=/mnt/d/obd2npz3_canonical_h5}"
: "${CASE_ID:=Job-new0a0b0}"
: "${FIELD:=stress}"
: "${OUT_DIR:=${BENCHMARK_ROOT}/outputs}"

"${PYTHON_BIN}" "${BENCHMARK_ROOT}/viz/animate_case.py" \
  --h5 "${H5_ROOT}/${CASE_ID}.h5" \
  --field "${FIELD}" \
  --gif "${OUT_DIR}/${CASE_ID}-${FIELD}.gif" \
  --png "${OUT_DIR}/${CASE_ID}-${FIELD}-final.png"

print "Wrote ${OUT_DIR}/${CASE_ID}-${FIELD}.gif"


