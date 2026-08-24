#!/usr/bin/env python3
"""Select full-box TNG50 and TNG100 dark-matter target haloes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

import _common


MASS_FIELD = "Group_M_Mean200"
FIRST_SUB_FIELD = "GroupFirstSub"
TNG50_TARGET_MIN_MASS_MSUN = 1.0e10
TNG50_TARGET_MAX_MASS_MSUN = 1.0e13
TNG100_TARGET_MIN_MASS_MSUN = 1.0e13
TNG50_FULL_BOX_LABEL = "tng50_1_dark_full_box_gt_1e10_msun_and_le_1e13_msun"
TNG100_FULL_BOX_LABEL = "tng100_1_dark_full_box_gt_1e13_msun"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default=str(_common.DEFAULT_DATA_DIR),
        help=(
            "Absolute directory for all caches and outputs. The default is "
            "/lingshan/disk3/subonan/TNG50+100-1-Dark."
        ),
    )
    parser.add_argument(
        "--overwrite-groupcat-cache",
        action="store_true",
        help="Redownload cached z=0 group-catalogue fields.",
    )
    args = parser.parse_args()
    if not Path(args.data_dir).expanduser().is_absolute():
        parser.error("--data_dir must be an absolute path.")
    return args


def find_sim_and_snapshots(
    session,
    sim_key: str,
    sims_by_name: dict[str, dict],
) -> tuple[dict, dict, list[dict]]:
    spec = _common.get_simulation_spec(sim_key)
    sim = sims_by_name.get(spec["name"])
    if sim is None:
        raise RuntimeError(f"{spec['name']} was not found in the TNG API simulation list.")
    snapshots_url = sim.get("snapshots") or (str(sim["url"]).rstrip("/") + "/snapshots/")
    snapshots = _common.request_json(session, snapshots_url)
    if not snapshots:
        raise RuntimeError(f"{spec['name']} returned an empty snapshot list.")
    snap_z0 = max(snapshots, key=lambda item: int(item["number"]))
    snap_z0 = _common.request_json(session, snap_z0["url"])
    return sim, snap_z0, snapshots


def write_snap_to_redshift_table(sim_key: str, snapshots: list[dict]) -> Path:
    path = _common.snap_to_z_path(sim_key)
    pairs = sorted((int(item["number"]), float(item["redshift"])) for item in snapshots)
    max_snap = max(number for number, _ in pairs)
    redshifts = np.full(max_snap + 1, np.nan, dtype=np.float64)
    for number, redshift in pairs:
        redshifts[number] = redshift
    header = f"# redshift lookup indexed by SnapNum for {_common.get_simulation_spec(sim_key)['name']}"
    lines = [header]
    lines.extend(f"{redshift:.11f}" for redshift in redshifts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def download_group_field(
    session,
    sim_key: str,
    snapnum: int,
    field: str,
    *,
    overwrite: bool,
) -> Path:
    out = _common.groupcat_field_path(sim_key, field)
    if out.exists() and not overwrite:
        with h5py.File(out, "r") as handle:
            _ = handle["Group"][field].shape
        return out

    spec = _common.get_simulation_spec(sim_key)
    url = _common.normalize_url(
        f"{_common.BASE_URL}{spec['name']}/files/groupcat-{snapnum}/?Group={field}"
    )
    _common.download_binary(session, url, out)
    with h5py.File(out, "r") as handle:
        _ = handle["Group"][field].shape
    return out


def load_suite_catalog(
    session,
    sim_key: str,
    *,
    overwrite: bool,
    sims_by_name: dict[str, dict],
) -> dict[str, object]:
    spec = _common.get_simulation_spec(sim_key)
    sim, snap_z0, snapshots = find_sim_and_snapshots(session, sim_key, sims_by_name)
    write_snap_to_redshift_table(sim_key, snapshots)

    snapnum = int(snap_z0["number"])
    p_mass = download_group_field(session, sim_key, snapnum, MASS_FIELD, overwrite=overwrite)
    p_first = download_group_field(session, sim_key, snapnum, FIRST_SUB_FIELD, overwrite=overwrite)

    with h5py.File(p_mass, "r") as handle:
        mass_api_all = np.asarray(handle["Group"][MASS_FIELD][()], dtype=np.float64)
    with h5py.File(p_first, "r") as handle:
        first_sub_all = np.asarray(handle["Group"][FIRST_SUB_FIELD][()], dtype=np.int64)

    if mass_api_all.shape != first_sub_all.shape:
        raise RuntimeError(f"{spec['name']} group-catalogue fields have mismatched shapes.")

    mass_msun_all = mass_api_all * 1.0e10 / float(spec["h"])
    eligible = (
        (first_sub_all >= 0)
        & np.isfinite(mass_msun_all)
        & (mass_msun_all > 0.0)
    )
    halo_ids = np.flatnonzero(eligible).astype(np.int64)
    subhalo_ids = first_sub_all[eligible].astype(np.int64)
    mass_api = mass_api_all[eligible]
    mass_msun = mass_msun_all[eligible]

    return {
        "simulation": sim["name"],
        "simulation_key": sim_key,
        "h": float(spec["h"]),
        "box_size_ckpc_h": float(spec["box_size_ckpc_h"]),
        "snap_z0": {
            "number": snapnum,
            "redshift": float(snap_z0["redshift"]),
            "url": _common.normalize_url(str(snap_z0["url"])),
        },
        "snapshots_count": len(snapshots),
        "total_groups": int(mass_api_all.size),
        "eligible_total": int(np.count_nonzero(eligible)),
        "eligible_tng50_target": int(
            np.count_nonzero(
                (mass_msun > TNG50_TARGET_MIN_MASS_MSUN)
                & (mass_msun <= TNG50_TARGET_MAX_MASS_MSUN)
            )
        ),
        "eligible_tng100_target": int(
            np.count_nonzero(mass_msun > TNG100_TARGET_MIN_MASS_MSUN)
        ),
        "halo_id_z0": halo_ids,
        "subhalo_id_z0": subhalo_ids,
        "mass_api_1e10msun_over_h": mass_api,
        "mass_msun": mass_msun,
        "log_mass_msun": np.log10(mass_msun),
    }


def select_records(suite_tables: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sim_key in ("tng50_1_dark", "tng100_1_dark"):
        suite = suite_tables[sim_key]
        halo_ids = np.asarray(suite["halo_id_z0"], dtype=np.int64)
        subhalo_ids = np.asarray(suite["subhalo_id_z0"], dtype=np.int64)
        mass_api = np.asarray(suite["mass_api_1e10msun_over_h"], dtype=np.float64)
        mass_msun = np.asarray(suite["mass_msun"], dtype=np.float64)
        log_mass_msun = np.asarray(suite["log_mass_msun"], dtype=np.float64)
        snap_url = str(suite["snap_z0"]["url"]).rstrip("/")

        if sim_key == "tng50_1_dark":
            selection_mask = (
                (mass_msun > TNG50_TARGET_MIN_MASS_MSUN)
                & (mass_msun <= TNG50_TARGET_MAX_MASS_MSUN)
            )
            label = TNG50_FULL_BOX_LABEL
            selection_rule = "full_box_Group_M_Mean200_gt_1e10_msun_and_le_1e13_msun"
            threshold = TNG50_TARGET_MIN_MASS_MSUN
        else:
            selection_mask = mass_msun > TNG100_TARGET_MIN_MASS_MSUN
            label = TNG100_FULL_BOX_LABEL
            selection_rule = "full_box_Group_M_Mean200_gt_1e13_msun"
            threshold = TNG100_TARGET_MIN_MASS_MSUN

        indices = np.flatnonzero(selection_mask)
        for choice in indices:
            choice = int(choice)
            records.append(
                {
                    "simulation": str(suite["simulation"]),
                    "simulation_key": sim_key,
                    "snapnum_z0": int(suite["snap_z0"]["number"]),
                    "halo_id_z0": int(halo_ids[choice]),
                    "subhalo_id_z0": int(subhalo_ids[choice]),
                    "label": label,
                    "selection_rule": selection_rule,
                    "mass_field": MASS_FIELD,
                    "mass_api_1e10msun_over_h": float(mass_api[choice]),
                    "mass_msun": float(mass_msun[choice]),
                    "log_mass_msun": float(log_mass_msun[choice]),
                    "z0_mass_threshold_msun": float(threshold),
                    "tree_node_min_particles": _common.TNG_TREE_MIN_PARTICLES,
                    "tree_node_min_mass_msun": float(
                        _common.TNG_TREE_MIN_MASS_MSUN[sim_key]
                    ),
                    "subhalo_url_z0": _common.normalize_url(
                        f"{snap_url}/subhalos/{int(subhalo_ids[choice])}/"
                    ),
                }
            )

    for file_index, row in enumerate(records):
        sim_key = str(row["simulation_key"])
        row["file_index"] = file_index
        row["raw_tree_basename"] = _common.suite_prefixed_raw_name(
            sim_key, int(row["subhalo_id_z0"])
        )
        row["fixed_tree_basename"] = _common.suite_prefixed_fixed_name(sim_key, file_index)
    return records


def write_selection_outputs(
    records: list[dict[str, object]],
    suite_tables: dict[str, dict[str, object]],
) -> None:
    manifest_fields = [
        "file_index",
        "simulation",
        "simulation_key",
        "snapnum_z0",
        "halo_id_z0",
        "subhalo_id_z0",
        "label",
        "selection_rule",
        "mass_field",
        "mass_api_1e10msun_over_h",
        "mass_msun",
        "log_mass_msun",
        "z0_mass_threshold_msun",
        "tree_node_min_particles",
        "tree_node_min_mass_msun",
        "subhalo_url_z0",
        "raw_tree_basename",
        "fixed_tree_basename",
    ]
    with _common.TARGET_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in manifest_fields})

    label_fields = [
        "file_index",
        "simulation",
        "simulation_key",
        "halo_id_z0",
        "subhalo_id_z0",
        "label",
        "selection_rule",
        "mass_field",
        "mass_msun",
        "z0_mass_threshold_msun",
        "tree_node_min_particles",
        "tree_node_min_mass_msun",
    ]
    with _common.SELECTION_LABELS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in label_fields})

    _common.SELECTED_SUBHALO_IDS_TXT.write_text(
        "\n".join(f"{row['simulation_key']},{row['subhalo_id_z0']}" for row in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    _common.SELECTED_HALO_IDS_TXT.write_text(
        "\n".join(f"{row['simulation_key']},{row['halo_id_z0']}" for row in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )

    selected_by_suite = {
        sim_key: sum(1 for row in records if row["simulation_key"] == sim_key)
        for sim_key in ("tng50_1_dark", "tng100_1_dark")
    }
    selected_masses = [float(row["mass_msun"]) for row in records]
    tng50_records = [row for row in records if row["simulation_key"] == "tng50_1_dark"]
    tng100_records = [row for row in records if row["simulation_key"] == "tng100_1_dark"]
    payload = {
        "simulations": {
            sim_key: {
                "name": str(suite_tables[sim_key]["simulation"]),
                "h": float(suite_tables[sim_key]["h"]),
                "box_size_ckpc_h": float(suite_tables[sim_key]["box_size_ckpc_h"]),
                "snap_z0": suite_tables[sim_key]["snap_z0"],
                "snapshots_count": int(suite_tables[sim_key]["snapshots_count"]),
                "snap_to_z_path": str(_common.snap_to_z_path(sim_key)),
                "total_groups": int(suite_tables[sim_key]["total_groups"]),
                "eligible_total": int(suite_tables[sim_key]["eligible_total"]),
                "eligible_tng50_target": int(
                    suite_tables[sim_key]["eligible_tng50_target"]
                ),
                "eligible_tng100_target": int(
                    suite_tables[sim_key]["eligible_tng100_target"]
                ),
                "box_size_native_cmpc_h": float(
                    suite_tables[sim_key]["box_size_ckpc_h"]
                )
                / 1000.0,
                "box_size_physical_cmpc": float(
                    suite_tables[sim_key]["box_size_ckpc_h"]
                )
                / (1000.0 * float(suite_tables[sim_key]["h"])),
                "volume_native_cmpc_h3": (
                    float(suite_tables[sim_key]["box_size_ckpc_h"]) / 1000.0
                )
                ** 3,
                "volume_physical_cmpc3": (
                    float(suite_tables[sim_key]["box_size_ckpc_h"])
                    / (1000.0 * float(suite_tables[sim_key]["h"]))
                )
                ** 3,
            }
            for sim_key in ("tng50_1_dark", "tng100_1_dark")
        },
        "full_box_selection": {
            "simulation_keys": ["tng50_1_dark", "tng100_1_dark"],
            "geometry": "native_full_simulation_box",
            "box_size_ckpc_h": {
                sim_key: float(suite_tables[sim_key]["box_size_ckpc_h"])
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "side_native_cmpc_h": {
                sim_key: float(suite_tables[sim_key]["box_size_ckpc_h"]) / 1000.0
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "side_physical_cmpc": {
                sim_key: float(suite_tables[sim_key]["box_size_ckpc_h"])
                / (1000.0 * float(suite_tables[sim_key]["h"]))
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "volume_native_cmpc_h3": {
                sim_key: (float(suite_tables[sim_key]["box_size_ckpc_h"]) / 1000.0)
                ** 3
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "volume_physical_cmpc3": {
                sim_key: (
                    float(suite_tables[sim_key]["box_size_ckpc_h"])
                    / (1000.0 * float(suite_tables[sim_key]["h"]))
                )
                ** 3
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "periodic_wrapping": False,
            "coordinate_filter_applied": False,
            "catalogue_scope": "complete_z0_group_catalogue",
        },
        "criteria": {
            "mass_field": MASS_FIELD,
            "mass_field_unit": "1e10 Msun / h",
            "h": _common.TNG_H,
            "target_mass_rules": {
                "tng50_1_dark": {
                    "selection_rule": "full_box_Group_M_Mean200_gt_1e10_msun_and_le_1e13_msun",
                    "lower_msun": TNG50_TARGET_MIN_MASS_MSUN,
                    "lower_inclusive": False,
                    "upper_msun": TNG50_TARGET_MAX_MASS_MSUN,
                    "upper_inclusive": True,
                },
                "tng100_1_dark": {
                    "selection_rule": "full_box_Group_M_Mean200_gt_1e13_msun",
                    "lower_msun": TNG100_TARGET_MIN_MASS_MSUN,
                    "lower_inclusive": False,
                    "upper_msun": None,
                    "upper_inclusive": False,
                },
            },
            "tree_node_min_particles": _common.TNG_TREE_MIN_PARTICLES,
            "tree_node_min_mass_msun_by_simulation": {
                sim_key: float(_common.TNG_TREE_MIN_MASS_MSUN[sim_key])
                for sim_key in ("tng50_1_dark", "tng100_1_dark")
            },
            "require_groupfirstsub_nonnegative": True,
            "require_finite_positive_mass": True,
            "coordinate_filter_applied": False,
            "combined_order": ["tng50_1_dark_ascending_halo_id", "tng100_1_dark_ascending_halo_id"],
        },
        "counts": {
            "selected_total": len(records),
            "selected_by_simulation": selected_by_suite,
            "tng50_full_box_selected_total": len(tng50_records),
            "tng100_full_box_selected_total": len(tng100_records),
            "selected_mass_min_msun": min(selected_masses, default=None),
            "selected_mass_max_msun": max(selected_masses, default=None),
        },
        "manifest_csv": str(_common.TARGET_MANIFEST_CSV),
        "records": records,
    }
    _common.write_json(_common.TARGETS_JSON, payload)


def main() -> None:
    args = parse_args()
    _common.configure_data_dir(args.data_dir)
    _common.ensure_dirs()

    required_sim_keys = ("tng50_1_dark", "tng100_1_dark")
    session = _common.build_session()
    try:
        simulations_payload = _common.request_json(session, _common.BASE_URL)
        sims_by_name = {item["name"]: item for item in simulations_payload["simulations"]}
        suite_tables = {
            sim_key: load_suite_catalog(
                session,
                sim_key,
                overwrite=args.overwrite_groupcat_cache,
                sims_by_name=sims_by_name,
            )
            for sim_key in required_sim_keys
        }
    finally:
        session.close()

    records = select_records(suite_tables)
    write_selection_outputs(records, suite_tables)

    selected_masses = [float(row["mass_msun"]) for row in records]
    print(f"Saved target manifest -> {_common.TARGET_MANIFEST_CSV}")
    print(f"Saved target metadata -> {_common.TARGETS_JSON}")
    print(
        "TNG50 full box: Group_M_Mean200 > 1e10 Msun and "
        "Group_M_Mean200 <= 1e13 Msun, selected="
        f"{sum(row['simulation_key'] == 'tng50_1_dark' for row in records)}"
    )
    print(
        "TNG100 full box: Group_M_Mean200 > 1e13 Msun, selected="
        f"{sum(row['simulation_key'] == 'tng100_1_dark' for row in records)}"
    )
    print(f"Selected halos: {len(records)}")
    if selected_masses:
        print(
            "Selected mass range [Msun]: "
            f"{min(selected_masses):.6e} - {max(selected_masses):.6e}"
        )
    else:
        print("Selected mass range: no halos selected")


if __name__ == "__main__":
    main()
