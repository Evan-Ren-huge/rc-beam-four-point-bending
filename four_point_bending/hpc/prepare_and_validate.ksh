#!/usr/bin/env ksh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

: "${PYTHON_BIN:=python3}"
: "${NPZ_ROOT:=/mnt/d/obd2npz3}"
: "${H5_ROOT:=/mnt/d/obd2npz3_canonical_h5}"
: "${MANIFEST:=${BENCHMARK_ROOT}/data/cases.csv}"

"${PYTHON_BIN}" "${BENCHMARK_ROOT}/tools/build_case_manifest.py" \
  --npz-root "${NPZ_ROOT}" \
  --h5-root "${H5_ROOT}" \
  --output "${MANIFEST}"

"${PYTHON_BIN}" "${BENCHMARK_ROOT}/tools/validate_dataset.py" \
  --npz-root "${NPZ_ROOT}" \
  --h5-root "${H5_ROOT}" \
  --manifest "${MANIFEST}"

print "Benchmark inventory and canonical schema validation passed."


