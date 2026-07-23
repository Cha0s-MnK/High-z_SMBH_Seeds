#!/usr/bin/env python3
"""Convert raw dark-matter full trees to corrected fixed-tree .dat files."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from _common import (
    FIXED_TREE_DIR,
    H100,
    RAW_TREE_DIR,
    TARGET_MANIFEST_CSV,
    ensure_dirs,
    snap_to_z_path,
    write_json,
)


HEADER = (
    "log10_mhalo_msun first_progenitor_id subhalo_id main_leaf_progenitor_id "
    "descendant_id redshift subhalo_spin_x subhalo_spin_y subhalo_spin_z"
)


@dataclass(frozen=True)
class HaloNode:
    mass_msun: float
    subhalo_id: int
    descendant_id: int
    first_progenitor_id: int
    main_leaf_id: int
    redshift: float
    spin_x: float
    spin_y: float
    spin_z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Convert only the first N manifest rows. 0 means all available rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted .dat files.",
    )
    parser.add_argument(
        "--min-mass-msun",
        type=float,
        default=1e9,
        help="Minimum halo mass retained before correction, matching Alex treefix.py.",
    )
    return parser.parse_args()


def load_snap_to_redshift(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot table: {path}")
    table = np.loadtxt(path, dtype=float, comments="#")
    if table.ndim != 1:
        raise RuntimeError(f"Snapshot table must be one-dimensional: {path}")
    return table


def load_raw_nodes(path: Path, snap_to_z: np.ndarray, min_mass_msun: float) -> list[HaloNode]:
    with h5py.File(path, "r") as handle:
        mass = np.asarray(handle["SubhaloMass"][()], dtype=np.float64) * 1e10 / H100
        first_progenitor = np.asarray(handle["FirstProgenitorID"][()], dtype=np.int64)
        subhalo_id = np.asarray(handle["SubhaloID"][()], dtype=np.int64)
        descendant_id = np.asarray(handle["DescendantID"][()], dtype=np.int64)
        main_leaf = np.asarray(handle["MainLeafProgenitorID"][()], dtype=np.int64)
        snap_num = np.asarray(handle["SnapNum"][()], dtype=np.int64)
        spin = np.asarray(handle["SubhaloSpin"][()], dtype=np.float64)

    mask = (mass > float(min_mass_msun)) & (first_progenitor != -1)
    if not np.any(mask):
        return []

    mass = mass[mask][::-1]
    first_progenitor = first_progenitor[mask][::-1]
    subhalo_id = subhalo_id[mask][::-1]
    descendant_id = descendant_id[mask][::-1]
    main_leaf = main_leaf[mask][::-1]
    snap_num = snap_num[mask][::-1]
    spin = spin[mask][::-1]

    if np.any((snap_num < 0) | (snap_num >= len(snap_to_z))):
        raise RuntimeError(f"{path.name} contains snapshot indices outside the lookup table range.")
    redshift = snap_to_z[snap_num]
    if np.any(~np.isfinite(redshift)):
        raise RuntimeError(f"{path.name} references snapshot numbers with undefined redshifts.")

    out: list[HaloNode] = []
    for idx in range(len(mass)):
        out.append(
            HaloNode(
                mass_msun=float(mass[idx]),
                subhalo_id=int(subhalo_id[idx]),
                descendant_id=int(descendant_id[idx]),
                first_progenitor_id=int(first_progenitor[idx]),
                main_leaf_id=int(main_leaf[idx]),
                redshift=float(redshift[idx]),
                spin_x=float(spin[idx, 0]),
                spin_y=float(spin[idx, 1]),
                spin_z=float(spin[idx, 2]),
            )
        )
    return out


def correct_branch(branch: list[HaloNode]) -> list[HaloNode]:
    if not branch:
        return []
    branch = sorted(branch, key=lambda node: node.redshift, reverse=True)
    corrected = [branch[0]]
    i = 0
    while i <= len(branch) - 1:
        current = branch[i]
        j = i + 1
        found = False
        while (not found) and j < len(branch):
            candidate = branch[j]
            if candidate.mass_msun < current.mass_msun:
                j += 1
                continue
            corrected.append(
                HaloNode(
                    mass_msun=candidate.mass_msun,
                    subhalo_id=candidate.subhalo_id,
                    descendant_id=candidate.descendant_id,
                    first_progenitor_id=current.subhalo_id,
                    main_leaf_id=candidate.main_leaf_id,
                    redshift=candidate.redshift,
                    spin_x=candidate.spin_x,
                    spin_y=candidate.spin_y,
                    spin_z=candidate.spin_z,
                )
            )
            found = True
        i = j
    return corrected


def write_fixed_tree(path: Path, nodes: list[HaloNode]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(HEADER + "\n")
        for node in nodes:
            handle.write(
                f"{math.log10(node.mass_msun):.8f} "
                f"{node.first_progenitor_id:d} "
                f"{node.subhalo_id:d} "
                f"{node.main_leaf_id:d} "
                f"{node.descendant_id:d} "
                f"{node.redshift:.11f} "
                f"{node.spin_x:.8f} "
                f"{node.spin_y:.8f} "
                f"{node.spin_z:.8f}\n"
            )


def main() -> None:
    args = parse_args()
    ensure_dirs()

    with TARGET_MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[: args.limit]

    snap_tables = {
        sim_key: load_snap_to_redshift(snap_to_z_path(sim_key))
        for sim_key in sorted({row["simulation_key"] for row in rows})
    }

    converted = 0
    skipped = 0
    conversion_rows: list[dict[str, object]] = []

    id_lookup_csv = FIXED_TREE_DIR / "id_lookup_large_dark.csv"
    id_lookup_txt = FIXED_TREE_DIR / "id_lookup_large_dark.txt"

    for row in rows:
        sim_key = row["simulation_key"]
        raw_path = RAW_TREE_DIR / row["raw_tree_basename"]
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw tree file: {raw_path}")

        out_path = FIXED_TREE_DIR / row["fixed_tree_basename"]
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        raw_nodes = load_raw_nodes(raw_path, snap_tables[sim_key], args.min_mass_msun)
        corrected_nodes: list[HaloNode] = []
        for main_leaf_id in sorted({node.main_leaf_id for node in raw_nodes}):
            branch = [node for node in raw_nodes if node.main_leaf_id == main_leaf_id]
            corrected_nodes.extend(correct_branch(branch))

        write_fixed_tree(out_path, corrected_nodes)
        converted += 1
        conversion_rows.append(
            {
                "file_index": int(row["file_index"]),
                "simulation": row["simulation"],
                "simulation_key": sim_key,
                "halo_id_z0": int(row["halo_id_z0"]),
                "subhalo_id_z0": int(row["subhalo_id_z0"]),
                "label": row["label"],
                "raw_tree_basename": row["raw_tree_basename"],
                "fixed_tree_basename": row["fixed_tree_basename"],
                "raw_rows_after_prefilter": len(raw_nodes),
                "corrected_rows_written": len(corrected_nodes),
                "unique_main_leaf_ids": len({node.main_leaf_id for node in raw_nodes}),
            }
        )

    with id_lookup_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "file_index",
            "simulation",
            "simulation_key",
            "halo_id_z0",
            "subhalo_id_z0",
            "label",
            "raw_tree_basename",
            "fixed_tree_basename",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    with id_lookup_txt.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row['file_index']},{row['simulation_key']},{row['raw_tree_basename']}\n"
            )

    summary = {
        "requested_rows": len(rows),
        "converted": converted,
        "skipped_existing": skipped,
        "fixed_tree_dir": str(FIXED_TREE_DIR),
        "per_file": conversion_rows,
    }
    write_json(FIXED_TREE_DIR / "conversion_summary.json", summary)

    print(f"Requested rows: {len(rows)}")
    print(f"Converted: {converted}")
    print(f"Skipped existing: {skipped}")
    print(f"Fixed-tree directory: {FIXED_TREE_DIR}")


if __name__ == "__main__":
    main()
