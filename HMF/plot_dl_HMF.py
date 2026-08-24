#!/usr/bin/env python3
"""Plot the z=0 TNG50-Dark and TNG100-Dark target halo mass functions.

The script reads the downloaded Group_M_Mean200 fields and target manifest,
then overlays a Tinker (2008) Lambda-CDM prediction computed with the
IllustrisTNG cosmological parameters.

Examples
--------
python plot_dl_HMF.py \
    --data-dir '/lingshan/disk3/subonan/TNG50+100-1-Dark' \
    --output tng_hmf_z0

`Group_M_Mean200` uses the native TNG mass unit, 1e10 Msun/h. The target
manifest selects the complete TNG50 and TNG100 z=0 catalogues with separate
full-box physical volumes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


TNG_COSMOLOGY = {
    "flat": True,
    "H0": 67.74,
    "Om0": 0.3089,
    "Ob0": 0.0486,
    "sigma8": 0.8159,
    "ns": 0.9667,
}
TNG_H = TNG_COSMOLOGY["H0"] / 100.0
TNG50_1_DARK_DM_PARTICLE_MASS_MSUN = 5.3843825e5
TNG100_1_DARK_DM_PARTICLE_MASS_MSUN = 8.8565106e6

DEFAULT_DATA_DIR = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark")
TARGETS_METADATA_FILENAME = "targets_z0_dark.json"
TARGET_MANIFEST_FILENAME = "target_manifest_dark.csv"
MASS_FIELD = "Group_M_Mean200"
FIRST_SUB_FIELD = "GroupFirstSub"
TNG50_KEY = "tng50_1_dark"
TNG100_KEY = "tng100_1_dark"
SUPPORTED_SIMULATION_KEYS = (TNG50_KEY, TNG100_KEY)
SNAPSHOT_Z0 = 99
REDSHIFT_TOLERANCE = 1.0e-5
MANIFEST_MASS_RTOL = 1.0e-10
MANIFEST_MASS_ATOL_MSUN = 1.0e-3
TNG50_TARGET_MIN_MASS_MSUN = 1.0e10
TNG50_TARGET_MAX_MASS_MSUN = 1.0e13
TNG100_TARGET_MIN_MASS_MSUN = 1.0e13
EXPECTED_BOX_SIZES_CKPC_H = {TNG50_KEY: 35000.0, TNG100_KEY: 75000.0}
EXPECTED_SIMULATION_NAMES = {
    TNG50_KEY: "TNG50-1-Dark",
    TNG100_KEY: "TNG100-1-Dark",
}
EXPECTED_SELECTION_RULES = {
    TNG50_KEY: "full_box_Group_M_Mean200_gt_1e10_msun_and_le_1e13_msun",
    TNG100_KEY: "full_box_Group_M_Mean200_gt_1e13_msun",
}


@dataclass(frozen=True)
class GroupCatalogue:
    """A single TNG group catalogue reduced to what is needed for an HMF."""

    label: str
    masses_msun: np.ndarray
    volume_cmpc3: float
    redshift: float
    h: float
    dm_particle_mass_msun: float | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure target-selected M200m abundances from the downloaded "
            "TNG50-Dark and TNG100-Dark group fields."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        default=str(DEFAULT_DATA_DIR),
        help=(
            "Downloaded TNG50+100-1-Dark data directory containing the cached "
            "group fields, metadata, and target manifest."
        ),
    )
    parser.add_argument(
        "--output",
        default="tng_hmf",
        help="Output path without a file extension.",
    )
    parser.add_argument(
        "--mmin",
        type=float,
        default=1.0e9,
        help=(
            "Minimum halo mass shown and included in the binned target abundance "
            "[Msun]; it does not change manifest selection thresholds."
        ),
    )
    parser.add_argument(
        "--mmax",
        type=float,
        default=1.0e15,
        help="Maximum halo mass shown and included in the binned HMF [Msun].",
    )
    parser.add_argument(
        "--dlogm",
        type=float,
        default=0.20,
        help="Bin width in log10(M200m/Msun) [dex].",
    )
    parser.add_argument(
        "--redshift",
        type=float,
        default=None,
        help=(
            "Optional redshift check; the downloaded field product is snapshot "
            "99 at z=0."
        ),
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=500,
        help=(
            "Particle count used only to mark conservative resolution limits. "
            "It does not discard catalogue entries."
        ),
    )
    parser.add_argument(
        "--tng50-dm-particle-mass",
        type=float,
        default=TNG50_1_DARK_DM_PARTICLE_MASS_MSUN,
        metavar="MSUN",
        help=(
            "TNG50 DM-particle mass [Msun] for the resolution-limit line; "
            "default is TNG50-1-Dark."
        ),
    )
    parser.add_argument(
        "--tng100-dm-particle-mass",
        type=float,
        default=TNG100_1_DARK_DM_PARTICLE_MASS_MSUN,
        metavar="MSUN",
        help=(
            "TNG100 DM-particle mass [Msun] for the resolution-limit line; "
            "default is TNG100-1-Dark."
        ),
    )
    parser.add_argument(
        "--no-resolution-lines",
        action="store_true",
        help="Do not draw particle-number resolution-limit lines.",
    )
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir).expanduser()
    if not args.data_dir.is_absolute():
        parser.error("--data-dir must be an absolute path.")
    if args.redshift is not None and not np.isfinite(args.redshift):
        parser.error("--redshift must be finite when supplied.")
    return args


def load_targets_metadata(data_dir: Path) -> dict[str, Any]:
    """Load and validate full-box suite geometry and selection metadata."""

    path = data_dir / TARGETS_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing target metadata: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read target metadata: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Target metadata must be a JSON object: {path}")

    simulations = metadata.get("simulations")
    if not isinstance(simulations, dict) or set(simulations) != set(SUPPORTED_SIMULATION_KEYS):
        raise ValueError(
            "Target metadata must contain exactly the supported TNG50 and TNG100 suites."
        )

    for simulation_key in SUPPORTED_SIMULATION_KEYS:
        spec = simulations.get(simulation_key)
        if not isinstance(spec, dict):
            raise ValueError(f"Malformed metadata for {simulation_key}.")
        if spec.get("name") != EXPECTED_SIMULATION_NAMES[simulation_key]:
            raise ValueError(f"Unexpected simulation name for {simulation_key}.")
        try:
            h = float(spec["h"])
            box_size_ckpc_h = float(spec["box_size_ckpc_h"])
            snap_z0 = spec["snap_z0"]
            snapnum = int(snap_z0["number"])
            redshift = float(snap_z0["redshift"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed z=0 metadata for {simulation_key}.") from error
        expected_box = EXPECTED_BOX_SIZES_CKPC_H[simulation_key]
        if not np.isclose(h, TNG_H, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{simulation_key}: metadata h={h} does not match {TNG_H}.")
        if not np.isclose(box_size_ckpc_h, expected_box, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"{simulation_key}: metadata box side {box_size_ckpc_h} ckpc/h "
                f"does not match {expected_box} ckpc/h."
            )
        if snapnum != SNAPSHOT_Z0 or not np.isfinite(redshift):
            raise ValueError(f"{simulation_key}: metadata is not a finite snapshot-99 entry.")
        if not np.isclose(redshift, 0.0, rtol=0.0, atol=REDSHIFT_TOLERANCE):
            raise ValueError(f"{simulation_key}: snapshot 99 metadata is not z=0.")

    full_box = metadata.get("full_box_selection")
    if not isinstance(full_box, dict):
        raise ValueError("Target metadata is missing full_box_selection.")
    if full_box.get("geometry") != "native_full_simulation_box":
        raise ValueError("Target metadata does not describe native full-box selection.")
    if bool(full_box.get("periodic_wrapping")) or bool(full_box.get("coordinate_filter_applied")):
        raise ValueError("Full-box selection must not apply wrapping or a coordinate filter.")
    for field in ("box_size_ckpc_h", "side_native_cmpc_h", "side_physical_cmpc", "volume_native_cmpc_h3", "volume_physical_cmpc3"):
        values = full_box.get(field)
        if not isinstance(values, dict) or set(values) != set(SUPPORTED_SIMULATION_KEYS):
            raise ValueError(f"full_box_selection is missing suite values for {field}.")
        for simulation_key in SUPPORTED_SIMULATION_KEYS:
            try:
                value = float(values[simulation_key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Malformed full-box metadata value {field}/{simulation_key}.") from error
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"full_box_selection has an invalid {field}/{simulation_key}.")
    for simulation_key in SUPPORTED_SIMULATION_KEYS:
        h = float(simulations[simulation_key]["h"])
        box_size_ckpc_h = float(simulations[simulation_key]["box_size_ckpc_h"])
        expected_side_native = box_size_ckpc_h / 1000.0
        expected_side_physical = expected_side_native / h
        if not np.isclose(float(full_box["box_size_ckpc_h"][simulation_key]), box_size_ckpc_h, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{simulation_key}: full-box native side disagrees with simulation metadata.")
        if not np.isclose(float(full_box["side_native_cmpc_h"][simulation_key]), expected_side_native, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{simulation_key}: full-box native side is inconsistent.")
        if not np.isclose(float(full_box["side_physical_cmpc"][simulation_key]), expected_side_physical, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"{simulation_key}: full-box physical side is inconsistent.")
        if not np.isclose(float(full_box["volume_native_cmpc_h3"][simulation_key]), expected_side_native**3, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{simulation_key}: full-box native volume is inconsistent.")
        if not np.isclose(float(full_box["volume_physical_cmpc3"][simulation_key]), expected_side_physical**3, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{simulation_key}: full-box physical volume is inconsistent.")

    criteria = metadata.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("Target metadata is missing selection criteria.")
    if criteria.get("mass_field") != MASS_FIELD:
        raise ValueError(f"Target metadata mass field is not {MASS_FIELD}.")
    try:
        criteria_h = float(criteria["h"])
        target_rules = criteria["target_mass_rules"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Target metadata contains malformed selection criteria.") from error
    if not np.isclose(criteria_h, TNG_H, rtol=0.0, atol=1.0e-10) or not isinstance(target_rules, dict):
        raise ValueError("Target metadata h or target-mass rules do not match the validated product.")
    expected_rules = {
        TNG50_KEY: (TNG50_TARGET_MIN_MASS_MSUN, False, TNG50_TARGET_MAX_MASS_MSUN, True),
        TNG100_KEY: (TNG100_TARGET_MIN_MASS_MSUN, False, None, False),
    }
    for simulation_key, (lower, lower_inclusive, upper, upper_inclusive) in expected_rules.items():
        rule = target_rules.get(simulation_key)
        if not isinstance(rule, dict):
            raise ValueError(f"Target metadata is missing target rule for {simulation_key}.")
        try:
            actual_lower = float(rule["lower_msun"])
            actual_lower_inclusive = bool(rule["lower_inclusive"])
            actual_upper = rule["upper_msun"]
            actual_upper = None if actual_upper is None else float(actual_upper)
            actual_upper_inclusive = bool(rule["upper_inclusive"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed target rule for {simulation_key}.") from error
        if (
            not np.isclose(actual_lower, lower, rtol=0.0, atol=0.0)
            or actual_lower_inclusive != lower_inclusive
            or (actual_upper is None) != (upper is None)
            or (actual_upper is not None and not np.isclose(actual_upper, upper, rtol=0.0, atol=0.0))
            or actual_upper_inclusive != upper_inclusive
            or rule.get("selection_rule") != EXPECTED_SELECTION_RULES[simulation_key]
        ):
            raise ValueError(f"Target metadata selection rule changed for {simulation_key}.")

    selected_by_simulation = metadata.get("counts", {}).get("selected_by_simulation")
    if not isinstance(selected_by_simulation, dict):
        raise ValueError("Target metadata is missing selected_by_simulation counts.")
    for simulation_key in SUPPORTED_SIMULATION_KEYS:
        try:
            count = int(selected_by_simulation[simulation_key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Target metadata has no valid count for {simulation_key}.") from error
        if count < 0:
            raise ValueError(f"Target metadata has a negative count for {simulation_key}.")
    if int(metadata.get("counts", {}).get("selected_total", -1)) != sum(int(selected_by_simulation[key]) for key in SUPPORTED_SIMULATION_KEYS):
        raise ValueError("Target metadata selected_total does not match suite counts.")
    return metadata


def load_target_manifest(
    data_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Read and validate the exact target rows used by the HMF plot."""

    path = data_dir / TARGET_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing target manifest: {path}")
    required_columns = {
        "simulation",
        "simulation_key",
        "snapnum_z0",
        "halo_id_z0",
        "selection_rule",
        "mass_field",
        "mass_msun",
        "fixed_tree_basename",
        "raw_tree_basename",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = required_columns - columns
            if missing:
                raise ValueError(f"Target manifest is missing columns: {sorted(missing)}")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Could not read target manifest: {path}") from error
    by_simulation = {simulation_key: [] for simulation_key in SUPPORTED_SIMULATION_KEYS}
    seen_halo_ids: set[tuple[str, int]] = set()
    seen_fixed_tree_names: set[str] = set()
    seen_raw_tree_names: set[str] = set()
    for row in rows:
        simulation_key = row.get("simulation_key", "").strip()
        if simulation_key not in SUPPORTED_SIMULATION_KEYS:
            raise ValueError(f"Target manifest contains unsupported suite: {simulation_key!r}")
        if row.get("simulation", "").strip() != EXPECTED_SIMULATION_NAMES[simulation_key]:
            raise ValueError(f"Target manifest has an unexpected name for {simulation_key}.")
        if row.get("selection_rule", "").strip() != EXPECTED_SELECTION_RULES[simulation_key]:
            raise ValueError(f"Target manifest has an unexpected selection rule for {simulation_key}.")
        if row.get("mass_field", "").strip() != MASS_FIELD:
            raise ValueError(f"Target manifest has an unexpected mass field for {simulation_key}.")
        try:
            snapnum = int(row["snapnum_z0"])
            halo_id = int(row["halo_id_z0"])
            mass_msun = float(row["mass_msun"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Target manifest has malformed numeric values for {simulation_key}.") from error
        valid_mass = (
            (mass_msun > TNG50_TARGET_MIN_MASS_MSUN)
            and (mass_msun <= TNG50_TARGET_MAX_MASS_MSUN)
            if simulation_key == TNG50_KEY
            else (mass_msun > TNG100_TARGET_MIN_MASS_MSUN)
        )
        if snapnum != SNAPSHOT_Z0 or halo_id < 0 or not np.isfinite(mass_msun) or not valid_mass:
            raise ValueError(f"Target manifest has an invalid z=0 target row for {simulation_key}.")
        halo_key = (simulation_key, halo_id)
        if halo_key in seen_halo_ids:
            raise ValueError(f"Target manifest contains duplicate halo ID: {halo_key}")
        seen_halo_ids.add(halo_key)
        fixed_tree_name = row.get("fixed_tree_basename", "").strip()
        raw_tree_name = row.get("raw_tree_basename", "").strip()
        if not fixed_tree_name or fixed_tree_name in seen_fixed_tree_names:
            raise ValueError("Target manifest contains a missing or duplicate fixed-tree name.")
        if not raw_tree_name or raw_tree_name in seen_raw_tree_names:
            raise ValueError("Target manifest contains a missing or duplicate raw-tree name.")
        seen_fixed_tree_names.add(fixed_tree_name)
        seen_raw_tree_names.add(raw_tree_name)
        by_simulation[simulation_key].append(row)

    if metadata is not None:
        metadata_counts = metadata["counts"]["selected_by_simulation"]
        for simulation_key in SUPPORTED_SIMULATION_KEYS:
            if len(by_simulation[simulation_key]) != int(metadata_counts[simulation_key]):
                raise ValueError(
                    f"Target manifest has {len(by_simulation[simulation_key])} rows for {simulation_key}; "
                    f"metadata records {metadata_counts[simulation_key]}."
                )
        records = metadata.get("records")
        if not isinstance(records, list) or len(records) != len(rows):
            raise ValueError("Target metadata records do not have one row per manifest target.")
        manifest_keys = {
            (row["simulation_key"], row["fixed_tree_basename"], int(row["halo_id_z0"]))
            for row in rows
        }
        record_keys = {
            (str(record.get("simulation_key")), str(record.get("fixed_tree_basename")), int(record.get("halo_id_z0")))
            for record in records
        }
        if manifest_keys != record_keys:
            raise ValueError("Target metadata records do not match the manifest provenance.")
    return by_simulation


def read_hdf5_field(
    path: Path,
    field: str,
    *,
    expected_ndim: int,
    allowed_dtype_kinds: set[str],
    output_dtype: Any,
) -> np.ndarray:
    """Read one validated Group field from a cached HDF5 product."""

    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading downloaded group fields requires h5py. Install it in the "
            "environment used to run this script."
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing group field: {path}")
    try:
        with h5py.File(path, "r") as handle:
            dataset = handle["Group"][field]
            if dataset.ndim != expected_ndim:
                raise ValueError(
                    f"{path}: Group/{field} must have {expected_ndim} dimensions, "
                    f"not {dataset.ndim}."
                )
            if dataset.dtype.kind not in allowed_dtype_kinds:
                raise ValueError(f"{path}: Group/{field} has an invalid numeric dtype.")
            values = np.asarray(dataset[()], dtype=output_dtype)
    except KeyError as error:
        raise KeyError(f"{path}: Group/{field} was not found.") from error
    except OSError as error:
        raise OSError(f"Could not read group field: {path}") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: Group/{field} contains non-finite values.")
    return values


def read_group_catalogue(
    data_dir: Path,
    simulation_key: str,
    metadata: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    dm_particle_mass_override_msun: float | None,
) -> GroupCatalogue:
    """Read one manifest-selected suite from the downloaded group fields."""

    if simulation_key not in SUPPORTED_SIMULATION_KEYS:
        raise ValueError(f"Unsupported simulation key: {simulation_key}")
    spec = metadata["simulations"][simulation_key]
    field_dir = data_dir / f"groupcat_fields_{simulation_key}"
    mass_native = read_hdf5_field(
        field_dir / f"{MASS_FIELD}.hdf5",
        MASS_FIELD,
        expected_ndim=1,
        allowed_dtype_kinds={"i", "u", "f"},
        output_dtype=np.float64,
    )
    first_sub = read_hdf5_field(
        field_dir / f"{FIRST_SUB_FIELD}.hdf5",
        FIRST_SUB_FIELD,
        expected_ndim=1,
        allowed_dtype_kinds={"i", "u"},
        output_dtype=np.int64,
    )
    if first_sub.shape != mass_native.shape:
        raise ValueError(f"{simulation_key}: cached Group field shapes do not match.")

    try:
        halo_ids = np.asarray([int(row["halo_id_z0"]) for row in manifest_rows], dtype=np.int64)
        manifest_masses = np.asarray([float(row["mass_msun"]) for row in manifest_rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{simulation_key}: malformed manifest target values.") from error
    if np.any(halo_ids >= mass_native.size):
        raise ValueError(f"{simulation_key}: manifest halo ID exceeds cached field length.")

    h = float(spec["h"])
    masses_msun = mass_native[halo_ids] * 1.0e10 / h
    if not np.allclose(
        masses_msun,
        manifest_masses,
        rtol=MANIFEST_MASS_RTOL,
        atol=MANIFEST_MASS_ATOL_MSUN,
    ):
        difference = np.abs(masses_msun - manifest_masses)
        raise ValueError(
            f"{simulation_key}: cached masses disagree with the target manifest; "
            f"maximum absolute difference is {np.max(difference):.6e} Msun."
        )
    selected_first_sub = first_sub[halo_ids]
    if np.any(selected_first_sub < 0) or np.any(~np.isfinite(masses_msun)) or np.any(masses_msun <= 0.0):
        raise ValueError(f"{simulation_key}: selected targets contain invalid mass or GroupFirstSub values.")
    if simulation_key == TNG50_KEY:
        valid_selection = (masses_msun > TNG50_TARGET_MIN_MASS_MSUN) & (masses_msun <= TNG50_TARGET_MAX_MASS_MSUN)
        label = "TNG50-1-Dark full box"
    else:
        valid_selection = masses_msun > TNG100_TARGET_MIN_MASS_MSUN
        label = "TNG100-1-Dark full box"
    if not np.all(valid_selection):
        raise ValueError(f"{simulation_key}: selected targets violate their full-box mass rule.")

    full_box = metadata["full_box_selection"]
    volume_cmpc3 = float(full_box["volume_physical_cmpc3"][simulation_key])
    expected_volume_cmpc3 = (float(spec["box_size_ckpc_h"]) / (1000.0 * h)) ** 3
    if not np.isclose(volume_cmpc3, expected_volume_cmpc3, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{simulation_key}: physical full-box volume is inconsistent with h and the box side.")

    dm_particle_mass_msun = dm_particle_mass_override_msun
    if dm_particle_mass_msun is None:
        dm_particle_mass_msun = (
            TNG50_1_DARK_DM_PARTICLE_MASS_MSUN
            if simulation_key == TNG50_KEY
            else TNG100_1_DARK_DM_PARTICLE_MASS_MSUN
        )
    if not np.isfinite(dm_particle_mass_msun) or dm_particle_mass_msun <= 0.0:
        raise ValueError(f"{simulation_key}: DM-particle mass must be finite and positive.")

    return GroupCatalogue(
        label=label,
        masses_msun=masses_msun,
        volume_cmpc3=volume_cmpc3,
        redshift=float(spec["snap_z0"]["redshift"]),
        h=h,
        dm_particle_mass_msun=dm_particle_mass_msun,
    )


def make_log_mass_bins(mmin: float, mmax: float, dlogm: float) -> np.ndarray:
    if not (mmin > 0.0 and mmax > mmin and dlogm > 0.0):
        raise ValueError("Require 0 < mmin < mmax and dlogm > 0.")
    logmin = np.log10(mmin)
    logmax = np.log10(mmax)
    return np.arange(logmin, logmax + 1.01 * dlogm, dlogm)


def measure_differential_hmf(
    catalogue: GroupCatalogue, log_mass_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centres, d n/d log10 M, and 68% Poisson bounds."""

    log_masses = np.log10(catalogue.masses_msun)
    counts, _ = np.histogram(log_masses, bins=log_mass_edges)
    dlogm = np.diff(log_mass_edges)
    phi = counts / (catalogue.volume_cmpc3 * dlogm)

    # Exact central 68% Poisson confidence interval for each non-empty bin.
    alpha = 1.0 - 0.682689492137086
    count_lo = np.where(
        counts > 0,
        0.5 * chi2.ppf(alpha / 2.0, 2.0 * counts),
        0.0,
    )
    count_hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (counts + 1))
    phi_lo = count_lo / (catalogue.volume_cmpc3 * dlogm)
    phi_hi = count_hi / (catalogue.volume_cmpc3 * dlogm)
    log_centres = 0.5 * (log_mass_edges[1:] + log_mass_edges[:-1])
    return log_centres, phi, phi_lo, phi_hi


def tinker08_dndlog10m(masses_msun: np.ndarray, redshift: float, h: float) -> np.ndarray:
    """Evaluate the Tinker08 M200m mass function in cMpc^-3 dex^-1.

    Colossus uses masses in Msun/h and returns dn/dlnM in h^3 Mpc^-3.
    The conversion to the physical units plotted here is applied explicitly.
    """

    try:
        from colossus.cosmology import cosmology
        from colossus.lss import mass_function
    except ImportError as error:
        raise ImportError(
            "The theoretical curve requires Colossus. Install it in the runtime "
            "environment, for example with `python -m pip install colossus`."
        ) from error

    cosmology_name = "IllustrisTNG_HMF"
    cosmology.addCosmology(cosmology_name, TNG_COSMOLOGY)
    cosmology.setCosmology(cosmology_name)

    masses_msun_over_h = np.asarray(masses_msun, dtype=float) * h
    dndlnm_h3_mpc3 = mass_function.massFunction(
        masses_msun_over_h,
        redshift,
        mdef="200m",
        model="tinker08",
        q_out="dndlnM",
    )
    return np.asarray(dndlnm_h3_mpc3) * h**3 * np.log(10.0)


def plot_catalogue_hmf(
    ax: plt.Axes,
    catalogue: GroupCatalogue,
    log_mass_edges: np.ndarray,
    colour: str,
    marker: str,
) -> None:
    logm, phi, phi_lo, phi_hi = measure_differential_hmf(catalogue, log_mass_edges)
    valid = phi > 0.0
    log_phi = np.log10(phi[valid])
    log_phi_lo = np.log10(phi_lo[valid])
    log_phi_hi = np.log10(phi_hi[valid])
    yerr = np.vstack((log_phi - log_phi_lo, log_phi_hi - log_phi))
    ax.errorbar(
        logm[valid],
        log_phi,
        yerr=yerr,
        fmt=marker,
        ms=5.5,
        mew=0.7,
        lw=1.0,
        capsize=2.3,
        color=colour,
        mec="white",
        label=catalogue.label,
        zorder=3,
    )


def add_resolution_limit(
    ax: plt.Axes,
    catalogue: GroupCatalogue,
    min_particles: int,
    colour: str,
) -> None:
    if catalogue.dm_particle_mass_msun is None:
        print(
            f"Warning: {catalogue.label}: no DM-particle mass is available; "
            "no resolution-limit line was drawn."
        )
        return
    limit_msun = min_particles * catalogue.dm_particle_mass_msun
    log_limit_msun = np.log10(limit_msun)
    xmin, xmax = ax.get_xlim()
    if xmin <= log_limit_msun <= xmax:
        ax.axvline(log_limit_msun, color=colour, ls=":", lw=1.3, alpha=0.9)


def resolve_redshift(
    catalogues: tuple[GroupCatalogue, ...], requested_redshift: float | None
) -> float:
    redshifts = np.array([catalogue.redshift for catalogue in catalogues])
    if requested_redshift is not None:
        if not np.allclose(redshifts, requested_redshift, rtol=0.0, atol=REDSHIFT_TOLERANCE):
            raise ValueError(
                "The supplied --redshift does not match the downloaded snapshot-99 metadata."
            )
        return float(requested_redshift)
    if not np.allclose(redshifts, redshifts[0], rtol=0.0, atol=REDSHIFT_TOLERANCE):
        raise ValueError("TNG50 and TNG100 target catalogues must come from the same redshift.")
    redshift = float(redshifts[0])
    return 0.0 if np.isclose(redshift, 0.0, rtol=0.0, atol=REDSHIFT_TOLERANCE) else redshift


def main() -> None:
    args = parse_arguments()
    metadata = load_targets_metadata(args.data_dir)
    manifest = load_target_manifest(args.data_dir, metadata)
    tng50 = read_group_catalogue(
        args.data_dir,
        TNG50_KEY,
        metadata,
        manifest[TNG50_KEY],
        args.tng50_dm_particle_mass,
    )
    tng100 = read_group_catalogue(
        args.data_dir,
        TNG100_KEY,
        metadata,
        manifest[TNG100_KEY],
        args.tng100_dm_particle_mass,
    )
    redshift = resolve_redshift((tng50, tng100), args.redshift)
    print(
        f"{tng50.label}: selected={tng50.masses_msun.size}, "
        f"threshold>{TNG50_TARGET_MIN_MASS_MSUN:.6e} Msun, "
        f"volume={tng50.volume_cmpc3:.12g} cMpc^3"
    )
    print(
        f"{tng100.label}: selected={tng100.masses_msun.size}, "
        f"threshold>{TNG100_TARGET_MIN_MASS_MSUN:.6e} Msun, "
        f"volume={tng100.volume_cmpc3:.12g} cMpc^3"
    )
    if redshift > 2.5:
        print(
            "Warning: Tinker08 is being evaluated beyond its z <= 2.5 calibration "
            "range; treat the theoretical line as an extrapolation."
        )
    if not np.isclose(tng50.h, tng100.h, rtol=0.0, atol=1.0e-8):
        raise ValueError("TNG50 and TNG100 catalogues have different Hubble parameters.")

    log_mass_edges = make_log_mass_bins(args.mmin, args.mmax, args.dlogm)
    theory_masses = np.logspace(np.log10(args.mmin), np.log10(args.mmax), 500)
    theory_phi = tinker08_dndlog10m(theory_masses, redshift, tng50.h)

    fig, ax = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
    ax.set_xlim(np.log10(args.mmin), np.log10(args.mmax))
    ax.plot(
        np.log10(theory_masses),
        np.log10(theory_phi),
        color="black",
        lw=2.0,
        label=rf"Tinker08 $\Lambda$CDM ($z={redshift:.3g}$)",
        zorder=2,
    )
    plot_catalogue_hmf(ax, tng50, log_mass_edges, "#2171b5", "o")
    plot_catalogue_hmf(ax, tng100, log_mass_edges, "#cb181d", "s")

    if not args.no_resolution_lines:
        add_resolution_limit(ax, tng50, args.min_particles, "#2171b5")
        add_resolution_limit(ax, tng100, args.min_particles, "#cb181d")

    ax.set_xlabel(r"$\log_{10}(M_{200\mathrm{m}}/M_\odot)$")
    ax.set_ylabel(
        r"$\log_{10}\!\left[\frac{\mathrm{d}n}{\mathrm{d}\log_{10}M_{200\mathrm{m}}}"
        r"\,/\,(\mathrm{cMpc}^{-3}\,\mathrm{dex}^{-1})\right]$"
    )
    ax.grid(True, alpha=0.22, linestyle=":", which="major")
    ax.legend(frameon=False, loc="best")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Saved {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
