#!/usr/bin/env python3
"""Validate converted fixed-tree .dat files for schema and basic correction invariants."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import FIXED_TREE_DIR, TARGET_MANIFEST_CSV, ensure_dirs, write_json


EXPECTED_NCOL = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Validate only the first N files from the manifest. 0 means all.",
    )
    return parser.parse_args()


def parse_fixed_tree(path: Path) -> tuple[str, list[list[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        rows = [line.split() for line in handle if line.strip()]
    return header, rows


def main() -> None:
    args = parse_args()
    ensure_dirs()

    with TARGET_MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if args.limit > 0:
        manifest_rows = manifest_rows[: args.limit]

    errors: list[str] = []
    branch_count_hist = Counter()
    per_file: list[dict[str, object]] = []

    for row in manifest_rows:
        path = FIXED_TREE_DIR / row["fixed_tree_basename"]
        if not path.exists():
            errors.append(f"Missing fixed tree file: {path}")
            continue

        header, raw_rows = parse_fixed_tree(path)
        if not header:
            errors.append(f"{path.name}: missing header line")
            continue

        parsed_rows = []
        for lineno, cols in enumerate(raw_rows, start=2):
            if len(cols) != EXPECTED_NCOL:
                errors.append(f"{path.name}:{lineno} expected {EXPECTED_NCOL} columns, found {len(cols)}")
                continue
            try:
                parsed_rows.append(
                    {
                        "logmh": float(cols[0]),
                        "fp": int(cols[1]),
                        "subhalo_id": int(cols[2]),
                        "main_leaf": int(cols[3]),
                        "desc": int(cols[4]),
                        "z": float(cols[5]),
                        "spin_x": float(cols[6]),
                        "spin_y": float(cols[7]),
                        "spin_z": float(cols[8]),
                    }
                )
            except Exception as exc:
                errors.append(f"{path.name}:{lineno} parse failure: {exc}")

        if not parsed_rows:
            per_file.append(
                {
                    "fixed_tree_basename": row["fixed_tree_basename"],
                    "rows": 0,
                    "unique_main_leaf_ids": 0,
                }
            )
            continue

        finite_cols = ("logmh", "z", "spin_x", "spin_y", "spin_z")
        for item in parsed_rows:
            for key in finite_cols:
                if not math.isfinite(float(item[key])):
                    errors.append(f"{path.name}: non-finite {key} value")
                    break

        grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
        for item in parsed_rows:
            grouped[int(item["main_leaf"])].append(item)

        for main_leaf_id, branch in grouped.items():
            branch.sort(key=lambda item: float(item["z"]), reverse=True)
            logmh = np.array([float(item["logmh"]) for item in branch], dtype=float)
            if np.any(np.diff(logmh) < -1e-10):
                errors.append(
                    f"{path.name}: corrected branch for main_leaf {main_leaf_id} still has mass dips"
                )

        unique_branch_count = len(grouped)
        branch_count_hist[unique_branch_count] += 1
        per_file.append(
            {
                "fixed_tree_basename": row["fixed_tree_basename"],
                "rows": len(parsed_rows),
                "unique_main_leaf_ids": unique_branch_count,
            }
        )

    summary = {
        "validated_files": len(manifest_rows),
        "error_count": len(errors),
        "branch_count_histogram": dict(sorted(branch_count_hist.items())),
        "per_file": per_file,
    }
    write_json(FIXED_TREE_DIR / "validation_report.json", summary)
    (FIXED_TREE_DIR / "validation_errors.txt").write_text(
        ("\n".join(errors) + "\n") if errors else "No validation errors.\n",
        encoding="utf-8",
    )

    print(f"Validated files: {len(manifest_rows)}")
    print(f"Validation errors: {len(errors)}")
    print(f"Branch-count histogram: {dict(sorted(branch_count_hist.items()))}")
    print(f"Validation report: {FIXED_TREE_DIR / 'validation_report.json'}")

    if errors:
        raise RuntimeError(f"Validation found {len(errors)} issues. See validation_errors.txt.")


if __name__ == "__main__":
    main()
