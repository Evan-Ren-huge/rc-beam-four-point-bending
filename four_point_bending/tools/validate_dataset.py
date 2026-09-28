#!/usr/bin/env python3
"""Validate the candidate RC four-point-bending NPZ/HDF5 archive."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit("h5py is required; install requirements.txt") from exc


CASE_RE = re.compile(
    r"^job-new(?P<index>\d+)a(?P<a>-?\d+)b(?P<b>-?\d+)$", re.IGNORECASE
)
EXPECTED_NPZ_COUNT = 195
EXPECTED_H5_COUNT = 196
EXPECTED_H5_ONLY = {"job-new0a0b0"}
EXPECTED_NPZ_KEYS = {
    "disp",
    "rf2",
    "head_u2",
    "connectivity",
    "node_coords",
    "SURF1_NODE_LABELS",
    "s_elem",
    "frame_times",
    "node_labels",
    "peeq_elem",
}
EXPECTED_H5_SHAPES = {
    "nodes/coords": (8393, 3),
    "nodes/reference_coords": (8393, 3),
    "nodes/node_id": (8393,),
    "nodes/node_type": (8393,),
    "elements/solid/connectivity": (6480, 8),
    "elements/solid/element_id": (6480,),
    "elements/solid/part_id": (6480,),
    "response/time/t": (21,),
    "response/node/displacement": (21, 8393, 3),
    "response/element/solid/stress_scalar": (21, 6480),
    "response/element/solid/peeq": (21, 6480),
    "response/global/rf2": (21,),
    "response/global/head_u2": (21,),
}


def parse(stem: str) -> tuple[int, int, int]:
    match = CASE_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"invalid case id: {stem}")
    return tuple(int(match.group(name)) for name in ("index", "a", "b"))


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["case_id"].casefold(): row for row in rows}


def split_loading_centers(coords_m: np.ndarray, node_type: np.ndarray) -> tuple[float, float]:
    z_mm = np.unique(np.round(coords_m[node_type == 1, 2] * 1000.0, 6))
    if z_mm.size != 10:
        raise ValueError(f"expected 10 unique flagged z positions, got {z_mm.size}")
    gaps = np.diff(z_mm)
    cut = int(np.argmax(gaps)) + 1
    if cut != 5:
        raise ValueError(f"expected two five-plane loading patches, cut={cut}")
    return float(z_mm[:cut].mean()), float(z_mm[cut:].mean())


def validate_npz(path: Path, deep: bool) -> list[str]:
    errors: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_NPZ_KEYS:
                errors.append(f"{path.name}: NPZ keys differ")
                return errors
            expected = {
                "disp": (21, 8393, 3),
                "rf2": (21,),
                "head_u2": (21,),
                "connectivity": (6480, 8),
                "node_coords": (8393, 3),
                "s_elem": (21, 6480),
                "frame_times": (21,),
                "node_labels": (8393,),
                "peeq_elem": (21, 6480),
            }
            for key, shape in expected.items():
                if archive[key].shape != shape:
                    errors.append(f"{path.name}: {key} shape={archive[key].shape}, expected={shape}")
                if deep and not np.isfinite(archive[key]).all():
                    errors.append(f"{path.name}: {key} contains non-finite values")
            conn = archive["connectivity"]
            if int(conn.min()) != 1 or int(conn.max()) != 8393:
                errors.append(f"{path.name}: NPZ connectivity is not 1-based labels")
    return errors


def validate_h5(path: Path, deep: bool) -> list[str]:
    errors: list[str] = []
    _, a_mm, b_mm = parse(path.stem)
    with h5py.File(path, "r") as handle:
        metadata = handle["metadata"].attrs
        expected_attrs = {
            "case_id": path.stem,
            "dataset_id": "RC-Beam-4Point",
            "schema_version": "0.2.0",
            "source_units": "mm-MPa-s-N",
            "units_convention": "SI",
        }
        for key, value in expected_attrs.items():
            if metadata.get(key) != value:
                errors.append(f"{path.name}: metadata[{key}]={metadata.get(key)!r}")
        for key, shape in EXPECTED_H5_SHAPES.items():
            if key not in handle:
                errors.append(f"{path.name}: missing {key}")
                continue
            dataset = handle[key]
            if dataset.shape != shape:
                errors.append(f"{path.name}: {key} shape={dataset.shape}, expected={shape}")
            if deep and not np.isfinite(dataset[...]).all():
                errors.append(f"{path.name}: {key} contains non-finite values")

        coords = handle["nodes/coords"][...]
        node_type = handle["nodes/node_type"][...]
        left, right = split_loading_centers(coords, node_type)
        if not math.isclose(left, 950 + b_mm, abs_tol=1e-6):
            errors.append(f"{path.name}: left loading center={left}, expected={950 + b_mm}")
        if not math.isclose(right, 1750 + a_mm, abs_tol=1e-6):
            errors.append(f"{path.name}: right loading center={right}, expected={1750 + a_mm}")

        connectivity = handle["elements/solid/connectivity"][...]
        if int(connectivity.min()) != 0 or int(connectivity.max()) != 8392:
            errors.append(f"{path.name}: H5 connectivity is not zero-based")
        times = handle["response/time/t"][...]
        if not np.allclose(times, np.linspace(0.0, 1.0, 21), atol=3e-8):
            errors.append(f"{path.name}: loading-progress grid differs")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deep", action="store_true", help="scan every numeric value for finiteness")
    args = parser.parse_args()

    npz_paths = sorted(args.npz_root.glob("*.npz"), key=lambda path: path.name.casefold())
    h5_paths = sorted(args.h5_root.glob("*.h5"), key=lambda path: path.name.casefold())
    manifest = load_manifest(args.manifest)
    errors: list[str] = []

    if len(npz_paths) != EXPECTED_NPZ_COUNT:
        errors.append(f"NPZ count={len(npz_paths)}, expected={EXPECTED_NPZ_COUNT}")
    if len(h5_paths) != EXPECTED_H5_COUNT:
        errors.append(f"H5 count={len(h5_paths)}, expected={EXPECTED_H5_COUNT}")

    npz_ids = {path.stem.casefold() for path in npz_paths}
    h5_ids = {path.stem.casefold() for path in h5_paths}
    if h5_ids - npz_ids != EXPECTED_H5_ONLY:
        errors.append(f"unexpected H5-only cases: {sorted(h5_ids - npz_ids)}")
    if npz_ids - h5_ids:
        errors.append(f"NPZ cases without H5: {sorted(npz_ids - h5_ids)}")
    if set(manifest) != npz_ids | h5_ids:
        errors.append("manifest case ids do not match the file inventory")

    for path in npz_paths:
        try:
            parse(path.stem)
            errors.extend(validate_npz(path, args.deep))
        except Exception as exc:  # keep scanning and report all broken cases
            errors.append(f"{path.name}: {exc}")
    for path in h5_paths:
        try:
            parse(path.stem)
            errors.extend(validate_h5(path, args.deep))
        except Exception as exc:  # keep scanning and report all broken cases
            errors.append(f"{path.name}: {exc}")

    print(f"NPZ files: {len(npz_paths)}")
    print(f"H5 files: {len(h5_paths)}")
    print(f"Manifest cases: {len(manifest)}")
    print(f"Deep value scan: {'yes' if args.deep else 'no'}")
    if errors:
        print(f"FAIL: {len(errors)} issue(s)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("PASS: inventory, schema, shapes, units/index conventions, and load encoding agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


