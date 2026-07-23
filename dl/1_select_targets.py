#!/usr/bin/env python3
"""Select z=0 dark-matter halos by mass-bin sampling or from an Illustris cube."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from _common import (
    BASE_URL,
    DATA_DIR,
    H100,
    SELECTION_LABELS_CSV,
    SIMULATIONS,
    TARGETS_JSON,
    TARGET_MANIFEST_CSV,
    build_session,
    download_binary,
    ensure_dirs,
    get_simulation_spec,
    groupcat_field_path,
    normalize_url,
    request_json,
    snap_to_z_path,
    suite_prefixed_fixed_name,
    suite_prefixed_raw_name,
    write_json,
)

MASS_FIELD = "Group_M_Mean200"
FIRST_SUB_FIELD = "GroupFirstSub"
POSITION_FIELD = "GroupPos"
LOW_MASS_SUITE_SPLIT_LOG10_MSUN = 11.0
ILLUSTRIS_BOX_SIZE_CKPC_H = 75000.0
RNG_SEED = 1
TARGET_LABEL = "random_log_mass_bin"
CUBE_TARGET_LABEL = "illustris1_dark_cube"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max_num_halo",
        type=int,
        default=256,
        help=("Maximum number of log-mass bins, and therefore the maximum number of selected halos; "
              "ignored in cube mode."))
    parser.add_argument(
        "--min_halo_mass",
        type=float,
        default=11.0,
        help="Lower halo-mass bound in log10(Msun).")
    parser.add_argument(
        "--max_halo_mass",
        type=float,
        default=14.0,
        help="Upper halo-mass bound in log10(Msun).")
    parser.add_argument(
        "--overwrite-groupcat-cache",
        action="store_true",
        help="Redownload cached group catalog cutouts even if they already exist.")
    parser.add_argument(
        "--cube-origin-ckpc-h",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help=("Lower cube corner in Illustris GroupPos coordinates [ckpc/h]. "
              "When supplied, select every eligible Illustris-1-Dark halo in the cube."))
    parser.add_argument(
        "--cube-side-cmpc",
        type=float,
        default=0.0,
        help="Cube side length [cMpc]; required together with --cube-origin-ckpc-h.")
    args = parser.parse_args()
    if args.max_num_halo < 1:
        raise ValueError("--max_num_halo must be >= 1.")
    if not args.min_halo_mass < args.max_halo_mass:
        raise ValueError("--min_halo_mass must be smaller than --max_halo_mass.")
    if args.cube_origin_ckpc_h is None:
        if args.cube_side_cmpc != 0.0:
            raise ValueError("--cube-side-cmpc requires --cube-origin-ckpc-h.")
    else:
        if args.cube_side_cmpc <= 0.0:
            raise ValueError("--cube-origin-ckpc-h requires a positive --cube-side-cmpc.")
        origin = np.asarray(args.cube_origin_ckpc_h, dtype=np.float64)
        side_ckpc_h = float(args.cube_side_cmpc) * 1000.0 * H100
        if np.any(origin < 0.0) or np.any(origin + side_ckpc_h > ILLUSTRIS_BOX_SIZE_CKPC_H):
            raise ValueError("Cube must lie wholly inside the Illustris-1-Dark box; periodic wrapping is not supported.")
    return args


def find_sim_and_snapshots(session, sim_key: str, sims_by_name: dict[str, dict]) -> tuple[dict, dict, list[dict]]:
    spec = get_simulation_spec(sim_key)
    sim = sims_by_name.get(spec["name"])
    if sim is None:
        raise RuntimeError(f"{spec['name']} not found in the Illustris API simulation list.")
    snapshots_url = sim.get("snapshots") or (str(sim["url"]).rstrip("/") + "/snapshots/")
    snapshots = request_json(session, snapshots_url)
    if not snapshots:
        raise RuntimeError(f"{spec['name']} returned an empty snapshot list.")
    snap_z0 = request_json(session, snapshots[-1]["url"])
    return sim, snap_z0, snapshots


def write_snap_to_redshift_table(sim_key: str, snapshots: list[dict]) -> Path:
    path = snap_to_z_path(sim_key)
    pairs = sorted((int(item["number"]), float(item["redshift"])) for item in snapshots)
    max_snap = max(number for number, _ in pairs)
    redshifts = np.full(max_snap + 1, np.nan, dtype=np.float64)
    for number, redshift in pairs:
        redshifts[number] = redshift
    header = f"# redshift lookup indexed by SnapNum for {get_simulation_spec(sim_key)['name']}"
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
    out = groupcat_field_path(sim_key, field)
    if out.exists() and not overwrite:
        with h5py.File(out, "r") as handle:
            _ = handle["Group"][field].shape
        return out

    spec = get_simulation_spec(sim_key)
    url = normalize_url(f"{BASE_URL}{spec['name']}/files/groupcat-{snapnum}/?Group={field}")
    download_binary(session, url, out)
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
    sim, snap_z0, snapshots = find_sim_and_snapshots(session, sim_key, sims_by_name)
    write_snap_to_redshift_table(sim_key, snapshots)

    snapnum = int(snap_z0["number"])
    p_mass = download_group_field(session, sim_key, snapnum, MASS_FIELD, overwrite=overwrite)
    p_first = download_group_field(session, sim_key, snapnum, FIRST_SUB_FIELD, overwrite=overwrite)
    p_pos = download_group_field(session, sim_key, snapnum, POSITION_FIELD, overwrite=overwrite)

    with h5py.File(p_mass, "r") as handle:
        mass_api = np.asarray(handle["Group"][MASS_FIELD][()], dtype=np.float64)
    with h5py.File(p_first, "r") as handle:
        first_sub = np.asarray(handle["Group"][FIRST_SUB_FIELD][()], dtype=np.int64)
    with h5py.File(p_pos, "r") as handle:
        group_pos = np.asarray(handle["Group"][POSITION_FIELD][()], dtype=np.float64)

    if mass_api.shape != first_sub.shape or group_pos.shape != (mass_api.size, 3):
        raise RuntimeError(f"{spec['name']} group catalog fields have mismatched shapes.")

    mass_msun = mass_api * 1e10 / H100
    eligible = (first_sub >= 0) & (mass_msun > 0.0)
    halo_ids = np.flatnonzero(eligible).astype(np.int64)
    subhalo_ids = first_sub[eligible].astype(np.int64)
    mass_api = mass_api[eligible]
    mass_msun = mass_msun[eligible]
    log_mass_msun = np.log10(mass_msun)
    group_pos = group_pos[eligible]

    return {
        "simulation": sim["name"],
        "simulation_key": sim_key,
        "snap_z0": {
            "number": snapnum,
            "redshift": float(snap_z0["redshift"]),
            "url": normalize_url(str(snap_z0["url"])),
        },
        "snapshots_count": len(snapshots),
        "eligible_total_with_firstsub": int(np.count_nonzero(first_sub >= 0)),
        "halo_id_z0": halo_ids,
        "subhalo_id_z0": subhalo_ids,
        "mass_api_1e10msun_over_h": mass_api,
        "mass_msun": mass_msun,
        "log_mass_msun": log_mass_msun,
        "position_ckpc_h": group_pos,
    }


def allocate_split_bins(total_bins: int, low_width: float, high_width: float) -> tuple[int, int]:
    if total_bins < 1:
        raise ValueError("total_bins must be >= 1")
    if low_width <= 0.0:
        return 0, total_bins
    if high_width <= 0.0:
        return total_bins, 0
    if total_bins == 1:
        return (1, 0) if low_width >= high_width else (0, 1)
    low_bins = int(round(total_bins * low_width / (low_width + high_width)))
    low_bins = max(1, min(total_bins - 1, low_bins))
    return low_bins, total_bins - low_bins


def build_selection_segments(min_log: float, max_log: float, max_num_halo: int) -> list[dict[str, object]]:
    if max_log <= LOW_MASS_SUITE_SPLIT_LOG10_MSUN:
        return [
            {
                "simulation_key": "tng50_1_dark",
                "left": float(min_log),
                "right": float(max_log),
                "num_bins": int(max_num_halo),
            }
        ]
    if min_log >= LOW_MASS_SUITE_SPLIT_LOG10_MSUN:
        return [
            {
                "simulation_key": "illustris1_dark",
                "left": float(min_log),
                "right": float(max_log),
                "num_bins": int(max_num_halo),
            }
        ]

    low_width = LOW_MASS_SUITE_SPLIT_LOG10_MSUN - min_log
    high_width = max_log - LOW_MASS_SUITE_SPLIT_LOG10_MSUN
    low_bins, high_bins = allocate_split_bins(max_num_halo, low_width, high_width)
    segments: list[dict[str, object]] = []
    if low_bins > 0:
        segments.append(
            {
                "simulation_key": "tng50_1_dark",
                "left": float(min_log),
                "right": float(LOW_MASS_SUITE_SPLIT_LOG10_MSUN),
                "num_bins": int(low_bins),
            }
        )
    if high_bins > 0:
        segments.append(
            {
                "simulation_key": "illustris1_dark",
                "left": float(LOW_MASS_SUITE_SPLIT_LOG10_MSUN),
                "right": float(max_log),
                "num_bins": int(high_bins),
            }
        )
    return segments


def sample_records(
    suite_tables: dict[str, dict[str, object]],
    segments: list[dict[str, object]],
    *,
    cube_origin_ckpc_h: np.ndarray | None = None,
    cube_side_ckpc_h: float = 0.0,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    records: list[dict[str, object]] = []
    cube_mode = cube_origin_ckpc_h is not None
    if cube_mode and cube_side_ckpc_h <= 0.0:
        raise ValueError("Cube selection requires a positive cube side.")

    for segment in segments:
        sim_key = str(segment["simulation_key"])
        if cube_mode and sim_key != "illustris1_dark":
            raise RuntimeError("Cube selection currently supports Illustris-1-Dark only.")
        suite = suite_tables[sim_key]
        left = float(segment["left"])
        right = float(segment["right"])
        num_bins = int(segment["num_bins"])
        edges = np.linspace(left, right, num_bins + 1, dtype=np.float64)

        log_mass = np.asarray(suite["log_mass_msun"], dtype=np.float64)
        halo_ids = np.asarray(suite["halo_id_z0"], dtype=np.int64)
        subhalo_ids = np.asarray(suite["subhalo_id_z0"], dtype=np.int64)
        mass_api = np.asarray(suite["mass_api_1e10msun_over_h"], dtype=np.float64)
        mass_msun = np.asarray(suite["mass_msun"], dtype=np.float64)
        position = np.asarray(suite["position_ckpc_h"], dtype=np.float64)
        snap_url = str(suite["snap_z0"]["url"]).rstrip("/")
        if position.shape != (log_mass.size, 3):
            raise RuntimeError(f"{suite['simulation']} position array has an unexpected shape.")

        if cube_mode:
            cube_upper = cube_origin_ckpc_h + cube_side_ckpc_h
            spatial_mask = np.all((position >= cube_origin_ckpc_h) & (position < cube_upper), axis=1)
        else:
            spatial_mask = np.ones(log_mass.size, dtype=bool)

        for ibin in range(num_bins):
            bin_left = float(edges[ibin])
            bin_right = float(edges[ibin + 1])
            if ibin == num_bins - 1:
                mask = (log_mass >= bin_left) & (log_mass <= bin_right)
            else:
                mask = (log_mass >= bin_left) & (log_mass < bin_right)
            indices = np.flatnonzero(mask & spatial_mask)
            if indices.size == 0:
                continue
            choices = indices if cube_mode else np.asarray([indices[rng.integers(0, indices.size)]])
            for choice in choices:
                choice = int(choice)
                subhalo_id = int(subhalo_ids[choice])
                records.append(
                    {
                    "simulation": str(suite["simulation"]),
                    "simulation_key": sim_key,
                    "snapnum_z0": int(suite["snap_z0"]["number"]),
                    "halo_id_z0": int(halo_ids[choice]),
                    "subhalo_id_z0": subhalo_id,
                    "label": CUBE_TARGET_LABEL if cube_mode else TARGET_LABEL,
                    "mass_field": MASS_FIELD,
                    "mass_api_1e10msun_over_h": float(mass_api[choice]),
                    "mass_msun": float(mass_msun[choice]),
                    "log_mass_msun": float(log_mass[choice]),
                    "selection_bin_left_log_msun": bin_left,
                    "selection_bin_right_log_msun": bin_right,
                    "subhalo_url_z0": normalize_url(f"{snap_url}/subhalos/{subhalo_id}/"),
                    }
                )

    for file_index, row in enumerate(records):
        sim_key = str(row["simulation_key"])
        row["file_index"] = file_index
        row["raw_tree_basename"] = suite_prefixed_raw_name(sim_key, int(row["subhalo_id_z0"]))
        row["fixed_tree_basename"] = suite_prefixed_fixed_name(sim_key, file_index)
    return records


def main() -> None:
    args = parse_args()
    ensure_dirs()

    cube_mode = args.cube_origin_ckpc_h is not None
    if cube_mode:
        segments = [{
            "simulation_key": "illustris1_dark",
            "left": float(args.min_halo_mass),
            "right": float(args.max_halo_mass),
            "num_bins": 1,
        }]
    else:
        segments = build_selection_segments(args.min_halo_mass, args.max_halo_mass, args.max_num_halo)
    required_sim_keys = sorted({str(segment["simulation_key"]) for segment in segments})

    session = build_session()
    try:
        sims = request_json(session, BASE_URL)["simulations"]
        sims_by_name = {item["name"]: item for item in sims}
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

    cube_side_ckpc_h = float(args.cube_side_cmpc) * 1000.0 * H100 if cube_mode else 0.0
    cube_origin_ckpc_h = np.asarray(args.cube_origin_ckpc_h, dtype=np.float64) if cube_mode else None
    records = sample_records(
        suite_tables,
        segments,
        cube_origin_ckpc_h=cube_origin_ckpc_h,
        cube_side_ckpc_h=cube_side_ckpc_h,
    )

    manifest_fields = [
        "file_index",
        "simulation",
        "simulation_key",
        "snapnum_z0",
        "halo_id_z0",
        "subhalo_id_z0",
        "label",
        "mass_field",
        "mass_api_1e10msun_over_h",
        "mass_msun",
        "log_mass_msun",
        "selection_bin_left_log_msun",
        "selection_bin_right_log_msun",
        "subhalo_url_z0",
        "raw_tree_basename",
        "fixed_tree_basename",
    ]
    with TARGET_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
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
        "mass_field",
        "mass_api_1e10msun_over_h",
        "mass_msun",
        "log_mass_msun",
        "selection_bin_left_log_msun",
        "selection_bin_right_log_msun",
    ]
    with SELECTION_LABELS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in label_fields})

    stale_names = (
        "mw_subhalo_ids_z0_dark.txt",
        "m31_extra_subhalo_ids_z0_dark.txt",
        "mw_halo_ids_z0_dark.txt",
        "m31_extra_halo_ids_z0_dark.txt",
        "mass_gt_1e9_subhalo_ids_z0_dark.txt",
        "mass_gt_1e9_halo_ids_z0_dark.txt",
        "selected_subhalos_z0_dark.txt",
        "selected_halos_z0_dark.txt",
    )
    for stale_name in stale_names:
        stale_path = DATA_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    (DATA_DIR / "selected_subhalos_z0_dark.txt").write_text(
        "\n".join(f"{row['simulation_key']},{row['subhalo_id_z0']}" for row in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    (DATA_DIR / "selected_halos_z0_dark.txt").write_text(
        "\n".join(f"{row['simulation_key']},{row['halo_id_z0']}" for row in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    selected_mass_min_msun = min((float(row["mass_msun"]) for row in records), default=None)
    selected_mass_max_msun = max((float(row["mass_msun"]) for row in records), default=None)

    payload = {
        "simulations": {
            sim_key: {
                "name": str(suite_tables[sim_key]["simulation"]),
                "snap_z0": suite_tables[sim_key]["snap_z0"],
                "snapshots_count": int(suite_tables[sim_key]["snapshots_count"]),
                "snap_to_z_path": str(snap_to_z_path(sim_key)),
                "eligible_total_with_firstsub": int(suite_tables[sim_key]["eligible_total_with_firstsub"]),
            }
            for sim_key in required_sim_keys
        },
        "criteria": {
            "mass_field": MASS_FIELD,
            "mass_field_unit": "1e10 Msun / h",
            "h100": H100,
            "selection_mode": "all_halos_in_illustris_cube" if cube_mode else "random_one_per_log_mass_bin",
            "max_num_halo": None if cube_mode else int(args.max_num_halo),
            "min_halo_mass_log10_msun": float(args.min_halo_mass),
            "max_halo_mass_log10_msun": float(args.max_halo_mass),
            "suite_split_log10_msun": LOW_MASS_SUITE_SPLIT_LOG10_MSUN,
            "selection_seed": RNG_SEED,
            "require_groupfirstsub_nonnegative": True,
            "cube_origin_ckpc_h": cube_origin_ckpc_h.tolist() if cube_mode else None,
            "cube_side_cmpc": float(args.cube_side_cmpc) if cube_mode else None,
            "cube_side_ckpc_h": cube_side_ckpc_h if cube_mode else None,
            "segments": segments,
        },
        "counts": {
            "selected_total": len(records),
            "nonempty_bins": None if cube_mode else len(records),
            "selected_mass_min_msun": selected_mass_min_msun,
            "selected_mass_max_msun": selected_mass_max_msun,
        },
        "manifest_csv": str(TARGET_MANIFEST_CSV),
        "records": records,
    }
    write_json(TARGETS_JSON, payload)

    print(f"Saved target manifest -> {TARGET_MANIFEST_CSV}")
    print(f"Saved target metadata -> {TARGETS_JSON}")
    if cube_mode:
        print(
            "Selection: all Illustris-1-Dark halos in cube "
            f"origin={cube_origin_ckpc_h.tolist()} ckpc/h, side={cube_side_ckpc_h:.3f} ckpc/h"
        )
    else:
        print(
            "Selection: one random halo per non-empty log-mass bin "
            f"across log10(M/Msun) = [{args.min_halo_mass:.3f}, {args.max_halo_mass:.3f}]"
        )
        print(f"Requested bins / max halos: {args.max_num_halo}")
    print(f"Selected halos: {len(records)}")
    if selected_mass_min_msun is None:
        print("Selected mass range: no halos selected")
    else:
        print(
            "Selected mass range [Msun]: "
            f"{selected_mass_min_msun:.6e} - {selected_mass_max_msun:.6e}"
        )
    for sim_key in required_sim_keys:
        suite = suite_tables[sim_key]
        count = sum(1 for row in records if row["simulation_key"] == sim_key)
        print(
            f"{get_simulation_spec(sim_key)['name']}: "
            f"eligible={suite['eligible_total_with_firstsub']} selected={count}"
        )


if __name__ == "__main__":
    main()
