#!/usr/bin/env python3
"""Convert, correct, and validate the current mixed TNG fixed-tree manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import _common


HEADER = (
    "log10_mhalo_msun first_progenitor_id subhalo_id main_leaf_progenitor_id "
    "descendant_id redshift subhalo_spin_x subhalo_spin_y subhalo_spin_z"
)
EXPECTED_NCOL = 9
LOOKUP_FIELDS = [
    "file_index",
    "simulation",
    "simulation_key",
    "halo_id_z0",
    "subhalo_id_z0",
    "label",
    "raw_tree_basename",
    "fixed_tree_basename",
]
TNG100_HALO_ID_OFFSET = 1_000_000
REFERENCE_LOOKUP_DIR = Path(
    "/lingshan/disk3/subonan/TNG50+100-1-Dark_New"
) / "fixed_trees_large_spin_dark_runid"
REFERENCE_ORIGINAL_LOOKUP = REFERENCE_LOOKUP_DIR / "id_lookup_original.csv"
REFERENCE_SHIFTED_LOOKUP = REFERENCE_LOOKUP_DIR / "id_lookup_large_dark.csv"

REQUIRED_MANIFEST_FIELDS = {
    "file_index",
    "simulation",
    "simulation_key",
    "halo_id_z0",
    "subhalo_id_z0",
    "label",
    "selection_rule",
    "raw_tree_basename",
    "fixed_tree_basename",
}


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
        "--data_dir",
        default=str(_common.DEFAULT_DATA_DIR),
        help=(
            "Absolute directory containing the manifest, raw trees, and fixed trees. "
            "The default is /lingshan/disk3/subonan/TNG50+100-1-Dark."
        ),
    )
    args = parser.parse_args()
    if not Path(args.data_dir).expanduser().is_absolute():
        parser.error("--data_dir must be an absolute path.")
    return args


def load_snap_to_redshift(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot table: {path}")
    try:
        table = np.loadtxt(path, dtype=float, comments="#", ndmin=1)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read snapshot table: {path}") from error
    table = np.asarray(table, dtype=float)
    if table.ndim != 1 or table.size == 0:
        raise RuntimeError(f"Snapshot table must be a non-empty one-dimensional array: {path}")
    if not np.any(np.isfinite(table)):
        raise RuntimeError(f"Snapshot table contains no defined redshifts: {path}")
    return table


def _read_hdf5_array(handle: h5py.File, name: str, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(handle[name][()], dtype=dtype)
    except KeyError as error:
        raise RuntimeError(f"Raw tree is missing required dataset {name!r}.") from error
    except (TypeError, ValueError, OSError) as error:
        raise RuntimeError(f"Could not read raw-tree dataset {name!r}.") from error


def _read_integer_hdf5_array(handle: h5py.File, name: str) -> np.ndarray:
    try:
        dataset = handle[name]
        values = np.asarray(dataset[()])
    except KeyError as error:
        raise RuntimeError(f"Raw tree is missing required dataset {name!r}.") from error
    except (TypeError, ValueError, OSError) as error:
        raise RuntimeError(f"Could not read raw-tree dataset {name!r}.") from error
    if values.dtype.kind not in "iu":
        raise RuntimeError(f"Raw-tree dataset {name!r} must contain integer values.")
    try:
        return values.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"Raw-tree dataset {name!r} cannot be represented as int64.") from error


def load_raw_nodes(
    path: Path,
    snap_to_z: np.ndarray,
    min_mass_msun: float,
    h: float,
) -> tuple[list[HaloNode], set[int]]:
    """Load one raw tree and return filtered nodes plus all source node IDs."""

    if not np.isfinite(h) or h <= 0.0:
        raise RuntimeError(f"Invalid Hubble parameter for {path.name}: {h!r}")
    try:
        with h5py.File(path, "r") as handle:
            mass_native = _read_hdf5_array(handle, "SubhaloMass", np.float64)
            first_progenitor = _read_integer_hdf5_array(handle, "FirstProgenitorID")
            subhalo_id = _read_integer_hdf5_array(handle, "SubhaloID")
            descendant_id = _read_integer_hdf5_array(handle, "DescendantID")
            main_leaf = _read_integer_hdf5_array(handle, "MainLeafProgenitorID")
            snap_num = _read_integer_hdf5_array(handle, "SnapNum")
            spin = _read_hdf5_array(handle, "SubhaloSpin", np.float64)
    except OSError as error:
        raise RuntimeError(f"Could not open raw tree: {path}") from error

    arrays = {
        "SubhaloMass": mass_native,
        "FirstProgenitorID": first_progenitor,
        "SubhaloID": subhalo_id,
        "DescendantID": descendant_id,
        "MainLeafProgenitorID": main_leaf,
        "SnapNum": snap_num,
    }
    if mass_native.ndim != 1:
        raise RuntimeError(f"{path.name}: SubhaloMass must be one-dimensional.")
    n_rows = mass_native.size
    for name, values in arrays.items():
        if values.shape != (n_rows,):
            raise RuntimeError(
                f"{path.name}: {name} has shape {values.shape}, expected {(n_rows,)}."
            )
    if spin.shape != (n_rows, 3):
        raise RuntimeError(
            f"{path.name}: SubhaloSpin has shape {spin.shape}, expected {(n_rows, 3)}."
        )
    for name, values in arrays.items():
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"{path.name}: {name} contains non-finite values.")
    if not np.all(np.isfinite(spin)):
        raise RuntimeError(f"{path.name}: SubhaloSpin contains non-finite values.")
    if np.any(mass_native < 0.0):
        raise RuntimeError(f"{path.name}: SubhaloMass contains negative values.")

    mass_msun = mass_native * 1.0e10 / float(h)
    if not np.all(np.isfinite(mass_msun)):
        raise RuntimeError(f"{path.name}: converted SubhaloMass contains non-finite values.")
    if np.any((snap_num < 0) | (snap_num >= len(snap_to_z))):
        raise RuntimeError(f"{path.name} contains snapshot indices outside its lookup table.")
    redshift_all = snap_to_z[snap_num]
    if np.any(~np.isfinite(redshift_all)):
        raise RuntimeError(f"{path.name} references snapshots with undefined redshifts.")
    if np.any(redshift_all < 0.0):
        raise RuntimeError(f"{path.name} references negative redshifts.")

    source_ids: set[int] = set()
    for values in (subhalo_id, first_progenitor, descendant_id, main_leaf):
        source_ids.update(int(value) for value in values if int(value) >= 0)

    if not np.isfinite(min_mass_msun) or min_mass_msun <= 0.0:
        raise RuntimeError(f"Invalid node mass threshold: {min_mass_msun!r}")
    mask = (mass_msun >= float(min_mass_msun)) & (first_progenitor != -1)
    if not np.any(mask):
        return [], source_ids

    mass_msun = mass_msun[mask][::-1]
    first_progenitor = first_progenitor[mask][::-1]
    subhalo_id = subhalo_id[mask][::-1]
    descendant_id = descendant_id[mask][::-1]
    main_leaf = main_leaf[mask][::-1]
    redshift = redshift_all[mask][::-1]
    spin = spin[mask][::-1]

    nodes: list[HaloNode] = []
    for idx in range(len(mass_msun)):
        nodes.append(
            HaloNode(
                mass_msun=float(mass_msun[idx]),
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
    return nodes, source_ids


def correct_branch(branch: list[HaloNode]) -> list[HaloNode]:
    """Apply the existing monotonic main-branch mass correction."""

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


def convert_nodes(nodes: list[HaloNode]) -> list[HaloNode]:
    grouped: dict[int, list[HaloNode]] = defaultdict(list)
    for node in nodes:
        grouped[node.main_leaf_id].append(node)
    corrected: list[HaloNode] = []
    for main_leaf_id in sorted(grouped):
        corrected.extend(correct_branch(grouped[main_leaf_id]))
    return corrected


def write_fixed_tree(path: Path, nodes: list[HaloNode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing target manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_MANIFEST_FIELDS - columns
        if missing:
            raise RuntimeError(f"Target manifest is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    seen_file_indices: set[int] = set()
    seen_fixed: set[str] = set()
    seen_raw: set[str] = set()
    for row in rows:
        try:
            file_index = int(row["file_index"])
            halo_id = int(row["halo_id_z0"])
            subhalo_id = int(row["subhalo_id_z0"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Malformed numeric manifest row: {row}") from error
        sim_key = row.get("simulation_key", "")
        if sim_key not in _common.SIMULATIONS:
            raise RuntimeError(f"Unsupported simulation key in manifest: {sim_key!r}")
        if file_index < 0 or file_index in seen_file_indices:
            raise RuntimeError(f"Manifest file_index is missing, negative, or duplicated: {file_index}")
        if halo_id < 0 or subhalo_id < 0:
            raise RuntimeError(f"Manifest IDs must be non-negative: {row}")
        for key, seen in (("fixed_tree_basename", seen_fixed), ("raw_tree_basename", seen_raw)):
            name = row.get(key, "").strip()
            if not name or Path(name).name != name or name in seen:
                raise RuntimeError(f"Manifest contains an invalid or duplicated {key}: {name!r}")
            seen.add(name)
        seen_file_indices.add(file_index)
    return rows


def _lookup_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Missing lookup file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != LOOKUP_FIELDS:
            raise RuntimeError(f"Lookup {path} has unexpected columns: {reader.fieldnames!r}")
        return [dict(row) for row in reader]


def write_lookup_files(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "id_lookup_original.csv"
    shifted_path = output_dir / "id_lookup_large_dark.csv"
    for path, shift_tng100 in ((original_path, False), (shifted_path, True)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOOKUP_FIELDS)
            writer.writeheader()
            for row in rows:
                output = {key: row[key] for key in LOOKUP_FIELDS}
                if shift_tng100 and row["simulation_key"] == "tng100_1_dark":
                    output["halo_id_z0"] = str(
                        int(row["halo_id_z0"]) + TNG100_HALO_ID_OFFSET
                    )
                writer.writerow(output)

    with (output_dir / "id_lookup_large_dark.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row['file_index']},{row['simulation_key']},{row['raw_tree_basename']}\n"
            )


def validate_lookup_files(rows: list[dict[str, str]], directory: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        original = _lookup_rows(directory / "id_lookup_original.csv")
        shifted = _lookup_rows(directory / "id_lookup_large_dark.csv")
    except RuntimeError as error:
        return [str(error)], {"original_rows": 0, "shifted_rows": 0}

    expected_by_fixed = {row["fixed_tree_basename"]: row for row in rows}
    if len(original) != len(rows):
        errors.append(f"id_lookup_original.csv has {len(original)} rows; expected {len(rows)}.")
    if len(shifted) != len(rows):
        errors.append(f"id_lookup_large_dark.csv has {len(shifted)} rows; expected {len(rows)}.")
    if len({row.get("fixed_tree_basename", "") for row in original}) != len(original):
        errors.append("id_lookup_original.csv contains duplicate fixed-tree basenames.")
    if len({row.get("fixed_tree_basename", "") for row in shifted}) != len(shifted):
        errors.append("id_lookup_large_dark.csv contains duplicate fixed-tree basenames.")

    original_by_fixed = {row.get("fixed_tree_basename", ""): row for row in original}
    shifted_by_fixed = {row.get("fixed_tree_basename", ""): row for row in shifted}
    final_ids: list[int] = []
    for fixed_name, expected in expected_by_fixed.items():
        for label, lookup_by_fixed, shifted_lookup in (
            ("original", original_by_fixed, False),
            ("shifted", shifted_by_fixed, True),
        ):
            actual = lookup_by_fixed.get(fixed_name)
            if actual is None:
                errors.append(f"id_lookup_{label}: missing {fixed_name}.")
                continue
            for key in ("file_index", "simulation", "simulation_key", "subhalo_id_z0", "label", "raw_tree_basename", "fixed_tree_basename"):
                if actual.get(key) != expected.get(key):
                    errors.append(
                        f"id_lookup_{label}: {fixed_name} has mismatched {key} "
                        f"({actual.get(key)!r} != {expected.get(key)!r})."
                    )
            try:
                expected_halo = int(expected["halo_id_z0"])
                actual_halo = int(actual["halo_id_z0"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"id_lookup_{label}: {fixed_name} has a non-integer halo ID.")
                continue
            required_halo = expected_halo
            if shifted_lookup and expected["simulation_key"] == "tng100_1_dark":
                required_halo += TNG100_HALO_ID_OFFSET
            if actual_halo != required_halo:
                errors.append(
                    f"id_lookup_{label}: {fixed_name} has halo_id_z0={actual_halo}; "
                    f"expected {required_halo}."
                )
            if shifted_lookup:
                final_ids.append(actual_halo)
    if len(final_ids) != len(set(final_ids)):
        errors.append("id_lookup_large_dark.csv contains duplicate final halo IDs.")
    for row in shifted:
        try:
            final_id = int(row["halo_id_z0"])
        except (KeyError, TypeError, ValueError):
            errors.append("id_lookup_large_dark.csv contains a non-integer halo ID.")
            continue
        if final_id < 0:
            errors.append("id_lookup_large_dark.csv contains a negative halo ID.")
    return errors, {"original_rows": len(original), "shifted_rows": len(shifted), "unique_final_ids": len(set(final_ids))}


def validate_fixed_tree(
    path: Path,
    source_ids: set[int],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing fixed tree file: {path}"], {"rows": 0, "unique_main_leaf_ids": 0}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        rows = [(line_no, line.split()) for line_no, line in enumerate(handle, start=2) if line.strip()]
    if not header:
        errors.append(f"{path.name}: missing header line.")
    elif header != HEADER:
        errors.append(f"{path.name}: unexpected fixed-tree header.")

    parsed: list[dict[str, float | int]] = []
    for line_no, columns in rows:
        if len(columns) != EXPECTED_NCOL:
            errors.append(f"{path.name}:{line_no} expected {EXPECTED_NCOL} columns, found {len(columns)}.")
            continue
        try:
            item = {
                "logmh": float(columns[0]),
                "fp": int(columns[1]),
                "subhalo_id": int(columns[2]),
                "main_leaf": int(columns[3]),
                "desc": int(columns[4]),
                "z": float(columns[5]),
                "spin_x": float(columns[6]),
                "spin_y": float(columns[7]),
                "spin_z": float(columns[8]),
            }
        except (TypeError, ValueError) as error:
            errors.append(f"{path.name}:{line_no} parse failure: {error}.")
            continue
        parsed.append(item)

    finite_columns = ("logmh", "z", "spin_x", "spin_y", "spin_z")
    identifier_columns = ("fp", "subhalo_id", "main_leaf", "desc")
    for item in parsed:
        for key in finite_columns:
            if not math.isfinite(float(item[key])):
                errors.append(f"{path.name}: non-finite {key} value.")
        if float(item["z"]) < 0.0:
            errors.append(f"{path.name}: negative redshift value.")
        for key in identifier_columns:
            value = int(item[key])
            if value < -1:
                errors.append(f"{path.name}: invalid {key} sentinel/value {value}.")
            if value >= 0 and value not in source_ids:
                errors.append(
                    f"{path.name}: {key}={value} is not a source raw-tree ID; "
                    "fixed-tree node IDs must not be shifted."
                )

    subhalo_ids = [int(item["subhalo_id"]) for item in parsed]
    if len(subhalo_ids) != len(set(subhalo_ids)):
        errors.append(f"{path.name}: duplicate fixed-tree subhalo IDs.")
    grouped: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for item in parsed:
        grouped[int(item["main_leaf"])].append(item)
    for main_leaf_id, branch in grouped.items():
        branch.sort(key=lambda item: float(item["z"]), reverse=True)
        logmh = np.asarray([float(item["logmh"]) for item in branch], dtype=float)
        if np.any(np.diff(logmh) < -1.0e-10):
            errors.append(f"{path.name}: corrected branch for main leaf {main_leaf_id} has a mass dip.")
    return errors, {"rows": len(parsed), "unique_main_leaf_ids": len(grouped)}


def verify_reference_offset() -> dict[str, Any]:
    """Audit the supplied reference lookup pair when it is available."""

    available = REFERENCE_ORIGINAL_LOOKUP.exists() or REFERENCE_SHIFTED_LOOKUP.exists()
    if not available:
        return {"available": False, "verified": False, "path": str(REFERENCE_LOOKUP_DIR)}
    if not REFERENCE_ORIGINAL_LOOKUP.exists() or not REFERENCE_SHIFTED_LOOKUP.exists():
        raise RuntimeError("The reference lookup directory contains only one of the two lookup files.")
    original = _lookup_rows(REFERENCE_ORIGINAL_LOOKUP)
    shifted = _lookup_rows(REFERENCE_SHIFTED_LOOKUP)
    if len(original) != len(shifted):
        raise RuntimeError("Reference original and shifted lookup files have different row counts.")
    for old, new in zip(original, shifted):
        for key in ("file_index", "simulation", "simulation_key", "subhalo_id_z0", "label", "raw_tree_basename", "fixed_tree_basename"):
            if old[key] != new[key]:
                raise RuntimeError(f"Reference lookup provenance differs for {key}: {old[key]!r} != {new[key]!r}")
        difference = int(new["halo_id_z0"]) - int(old["halo_id_z0"])
        expected = TNG100_HALO_ID_OFFSET if old["simulation_key"] == "tng100_1_dark" else 0
        if difference != expected:
            raise RuntimeError(
                f"Reference lookup shift is {difference} for {old['fixed_tree_basename']}; "
                f"expected {expected}."
            )
    return {
        "available": True,
        "verified": True,
        "path": str(REFERENCE_LOOKUP_DIR),
        "rows": len(original),
        "tng100_halo_id_offset": TNG100_HALO_ID_OFFSET,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    import json

    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def write_reports(
    summary: dict[str, Any],
    validation_report: dict[str, Any],
    errors: list[str],
) -> None:
    _atomic_write_json(_common.FIXED_TREE_DIR / "conversion_summary.json", summary)
    _atomic_write_json(_common.FIXED_TREE_DIR / "validation_report.json", validation_report)
    _atomic_write_text(
        _common.FIXED_TREE_DIR / "validation_errors.txt",
        ("\n".join(errors) + "\n") if errors else "No validation errors.\n",
    )


def build_manifest_identity(rows: list[dict[str, str]]) -> dict[str, Any]:
    digest = hashlib.sha256(_common.TARGET_MANIFEST_CSV.read_bytes()).hexdigest()
    return {
        "path": str(_common.TARGET_MANIFEST_CSV),
        "sha256": digest,
        "rows": len(rows),
    }


def main() -> None:
    args = parse_args()
    _common.configure_data_dir(args.data_dir)
    _common.ensure_dirs()

    rows: list[dict[str, str]] = []
    conversion_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    stage_dir: Path | None = None
    diagnostics_written = False
    reference_info: dict[str, Any] = {"available": False, "verified": False}
    per_suite = {
        sim_key: {"requested": 0, "converted": 0, "replaced_existing": 0}
        for sim_key in ("tng50_1_dark", "tng100_1_dark")
    }

    try:
        rows = read_manifest(_common.TARGET_MANIFEST_CSV)
        reference_info = verify_reference_offset()
        snap_tables = {
            sim_key: load_snap_to_redshift(_common.snap_to_z_path(sim_key))
            for sim_key in ("tng50_1_dark", "tng100_1_dark")
        }
        stage_dir = Path(tempfile.mkdtemp(prefix=".combined_fixed_trees_", dir=_common.FIXED_TREE_DIR))

        for row in rows:
            sim_key = row["simulation_key"]
            per_suite[sim_key]["requested"] += 1
            spec = _common.get_simulation_spec(sim_key)
            raw_path = _common.RAW_TREE_DIR / row["raw_tree_basename"]
            if not raw_path.exists():
                raise FileNotFoundError(f"Missing raw tree file: {raw_path}")
            nodes, source_ids = load_raw_nodes(
                raw_path,
                snap_tables[sim_key],
                float(_common.TNG_TREE_MIN_MASS_MSUN[sim_key]),
                float(spec["h"]),
            )
            corrected_nodes = convert_nodes(nodes)
            staged_path = stage_dir / row["fixed_tree_basename"]
            write_fixed_tree(staged_path, corrected_nodes)
            per_suite[sim_key]["converted"] += 1
            per_suite[sim_key]["replaced_existing"] += int(
                (_common.FIXED_TREE_DIR / row["fixed_tree_basename"]).exists()
            )
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
                    "raw_rows_after_prefilter": len(nodes),
                    "corrected_rows_written": len(corrected_nodes),
                    "unique_main_leaf_ids": len({node.main_leaf_id for node in nodes}),
                }
            )
            fixed_errors, fixed_info = validate_fixed_tree(staged_path, source_ids)
            errors.extend(f"{row['fixed_tree_basename']}: {error}" for error in fixed_errors)
            validation_rows.append(
                {
                    "fixed_tree_basename": row["fixed_tree_basename"],
                    "simulation_key": sim_key,
                    **fixed_info,
                }
            )

        write_lookup_files(rows, stage_dir)
        lookup_errors, lookup_info = validate_lookup_files(rows, stage_dir)
        errors.extend(lookup_errors)

        manifest_identity = build_manifest_identity(rows)
        summary = {
            "status": "validation_failed" if errors else "success",
            "committed": not bool(errors),
            "requested_rows": len(rows),
            "converted": len(conversion_rows),
            "skipped_existing": 0,
            "replaced_current_manifest_files": sum(item["replaced_existing"] for item in per_suite.values()),
            "node_filter": {
                "min_particles": _common.TNG_TREE_MIN_PARTICLES,
                "comparison": "inclusive",
                "condition": "SubhaloMass_msun >= min_mass_msun",
                "min_mass_msun_by_simulation": {
                    sim_key: float(_common.TNG_TREE_MIN_MASS_MSUN[sim_key])
                    for sim_key in ("tng50_1_dark", "tng100_1_dark")
                },
            },
            "h_by_simulation": {
                sim_key: float(_common.get_simulation_spec(sim_key)["h"])
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "tng100_halo_id_offset": TNG100_HALO_ID_OFFSET,
            "reference_lookup_shift": reference_info,
            "manifest_identity": manifest_identity,
            "fixed_tree_dir": str(_common.FIXED_TREE_DIR),
            "per_suite": per_suite,
            "per_file": conversion_rows,
        }
        validation_report = {
            "status": "failed" if errors else "passed",
            "committed": not bool(errors),
            "validated_files": len(validation_rows),
            "error_count": len(errors),
            "branch_count_histogram": dict(
                sorted(Counter(int(item["unique_main_leaf_ids"]) for item in validation_rows).items())
            ),
            "lookup_validation": lookup_info,
            "fixed_tree_dir": str(_common.FIXED_TREE_DIR),
            "per_file": validation_rows,
        }
        if errors:
            write_reports(summary, validation_report, errors)
            diagnostics_written = True
            raise RuntimeError(
                f"Conversion/validation found {len(errors)} issue(s). See "
                f"{_common.FIXED_TREE_DIR / 'validation_errors.txt'}."
            )

        commit_names = [row["fixed_tree_basename"] for row in rows]
        commit_names.extend(["id_lookup_original.csv", "id_lookup_large_dark.csv", "id_lookup_large_dark.txt"])
        for name in commit_names:
            (stage_dir / name).replace(_common.FIXED_TREE_DIR / name)
        summary["committed"] = True
        validation_report["committed"] = True
        write_reports(summary, validation_report, [])
        diagnostics_written = True
        print(f"Requested rows: {len(rows)}")
        print(f"Converted and replaced: {len(conversion_rows)}")
        print(f"TNG100 halo-ID offset in model-facing lookup: +{TNG100_HALO_ID_OFFSET}")
        print(f"Fixed-tree directory: {_common.FIXED_TREE_DIR}")
        print(f"Validation report: {_common.FIXED_TREE_DIR / 'validation_report.json'}")
    except Exception as error:
        if not diagnostics_written:
            failure_text = str(error)
            failure_summary = {
                "status": "conversion_failed",
                "committed": False,
                "requested_rows": len(rows),
                "converted": len(conversion_rows),
                "skipped_existing": 0,
                "node_filter": {
                    "min_particles": _common.TNG_TREE_MIN_PARTICLES,
                    "comparison": "inclusive",
                    "min_mass_msun_by_simulation": {
                        sim_key: float(_common.TNG_TREE_MIN_MASS_MSUN[sim_key])
                        for sim_key in ("tng50_1_dark", "tng100_1_dark")
                    },
                },
                "tng100_halo_id_offset": TNG100_HALO_ID_OFFSET,
                "reference_lookup_shift": reference_info,
                "manifest_identity": (
                    build_manifest_identity(rows)
                    if _common.TARGET_MANIFEST_CSV.exists()
                    else None
                ),
                "fixed_tree_dir": str(_common.FIXED_TREE_DIR),
                "per_suite": per_suite,
                "per_file": conversion_rows,
                "error": failure_text,
            }
            failure_report = {
                "status": "not_run",
                "committed": False,
                "validated_files": len(validation_rows),
                "error_count": 1,
                "lookup_validation": {},
                "fixed_tree_dir": str(_common.FIXED_TREE_DIR),
                "per_file": validation_rows,
            }
            write_reports(failure_summary, failure_report, errors + [failure_text])
        raise
    finally:
        if stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir)


if __name__ == "__main__":
    main()
