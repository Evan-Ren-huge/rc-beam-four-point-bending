#!/usr/bin/env python3
"""Build the RC four-point-bending case manifest from files on disk."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


CASE_RE = re.compile(
    r"^job-new(?P<index>\d+)a(?P<a>-?\d+)b(?P<b>-?\d+)$", re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedCase:
    case_id: str
    index: int
    a_mm: int
    b_mm: int


def parse_case_id(case_id: str) -> ParsedCase:
    match = CASE_RE.fullmatch(case_id)
    if match is None:
        raise ValueError(f"unrecognized case id: {case_id!r}")
    return ParsedCase(
        case_id=case_id,
        index=int(match.group("index")),
        a_mm=int(match.group("a")),
        b_mm=int(match.group("b")),
    )


def files_by_stem(root: Path, suffix: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"data directory not found: {root}")
    output: dict[str, Path] = {}
    for path in root.glob(f"*{suffix}"):
        key = path.stem.casefold()
        if key in output:
            raise ValueError(f"duplicate case id ignoring case: {path.stem}")
        parse_case_id(path.stem)
        output[key] = path
    return output


def build_rows(npz_root: Path, h5_root: Path) -> list[dict[str, object]]:
    npz = files_by_stem(npz_root, ".npz")
    h5 = files_by_stem(h5_root, ".h5")
    rows: list[dict[str, object]] = []
    for key in npz.keys() | h5.keys():
        path = h5.get(key) or npz[key]
        case = parse_case_id(path.stem)
        left_center = 950 + case.b_mm
        right_center = 1750 + case.a_mm
        rows.append(
            {
                "case_id": path.stem,
                "case_index": case.index,
                "a_right_shift_mm": case.a_mm,
                "b_left_shift_mm": case.b_mm,
                "left_load_center_mm": left_center,
                "right_load_center_mm": right_center,
                "load_center_spacing_mm": right_center - left_center,
                "npz_available": "yes" if key in npz else "no",
                "h5_available": "yes" if key in h5 else "no",
                "npz_bytes": npz[key].stat().st_size if key in npz else "",
                "h5_bytes": h5[key].stat().st_size if key in h5 else "",
                "split": "unassigned",
            }
        )
    return sorted(rows, key=lambda row: (int(row["case_index"]), str(row["case_id"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.npz_root, args.h5_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    npz_count = sum(row["npz_available"] == "yes" for row in rows)
    h5_count = sum(row["h5_available"] == "yes" for row in rows)
    print(f"wrote {len(rows)} cases to {args.output}")
    print(f"NPZ={npz_count} H5={h5_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


