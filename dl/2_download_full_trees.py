#!/usr/bin/env python3
"""Download full TNG SubLink trees for the selected dark-matter haloes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
from tqdm.auto import tqdm

import _common


REQUIRED_DATASETS = (
    "SubhaloID",
    "FirstProgenitorID",
    "DescendantID",
    "MainLeafProgenitorID",
    "SnapNum",
    "SubhaloSpin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default=str(_common.DEFAULT_DATA_DIR),
        help=(
            "Absolute directory containing the manifest and all downloaded products. "
            "The default is /lingshan/disk3/subonan/TNG50+100-1-Dark."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload trees even if valid raw HDF5 files already exist.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.1,
        help="Sleep duration between successful downloads.",
    )
    args = parser.parse_args()
    if not Path(args.data_dir).expanduser().is_absolute():
        parser.error("--data_dir must be an absolute path.")
    return args


def validate_tree_file(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise RuntimeError(f"{path.name} is missing required datasets: {missing}")


def write_download_summary(
    rows: list[dict[str, str]],
    downloaded: int,
    skipped: int,
    failures: list[dict[str, object]],
) -> None:
    summary = {
        "requested_rows": len(rows),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": len(failures),
        "raw_tree_dir": str(_common.RAW_TREE_DIR),
        "simulations": sorted({row["simulation_key"] for row in rows}),
        "stopped_on_failure": bool(failures),
    }
    _common.write_json(_common.DOWNLOAD_FAILURES_JSON, failures)
    _common.write_json(_common.DOWNLOAD_SUMMARY_JSON, summary)


def main() -> None:
    args = parse_args()
    _common.configure_data_dir(args.data_dir)
    _common.ensure_dirs()

    if not _common.TARGETS_JSON.exists():
        raise FileNotFoundError(f"Missing {_common.TARGETS_JSON}. Run 1_select_targets.py first.")
    if not _common.TARGET_MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Missing {_common.TARGET_MANIFEST_CSV}. Run 1_select_targets.py first."
        )

    _common.read_json()
    rows = _common.read_manifest()
    downloaded = 0
    skipped = 0
    failures: list[dict[str, object]] = []

    session = _common.build_session()
    try:
        with tqdm(rows, total=len(rows), desc="Full TNG trees", unit="halo") as progress:
            for row in progress:
                progress.set_postfix(
                    downloaded=downloaded,
                    skipped=skipped,
                    failed=len(failures),
                )
                sim_key = row["simulation_key"]
                subhalo_id = int(row["subhalo_id_z0"])
                outpath = _common.RAW_TREE_DIR / row["raw_tree_basename"]
                progress.set_description(f"Full trees {sim_key}:{subhalo_id}")

                try:
                    if outpath.exists() and not args.overwrite:
                        validate_tree_file(outpath)
                        skipped += 1
                        continue

                    subhalo_url = row.get("subhalo_url_z0", "").strip()
                    if not subhalo_url:
                        raise RuntimeError("Manifest row is missing subhalo_url_z0.")
                    sub_json = _common.request_json(session, subhalo_url)
                    tree_url = sub_json["trees"]["sublink"]
                    _common.download_binary(session, tree_url, outpath)
                    validate_tree_file(outpath)
                    downloaded += 1
                    time_to_sleep = max(0.0, args.sleep_sec)
                    if time_to_sleep:
                        time.sleep(time_to_sleep)
                except Exception as exc:
                    failure = {
                        "file_index": int(row["file_index"]),
                        "simulation": row["simulation"],
                        "simulation_key": sim_key,
                        "halo_id_z0": int(row["halo_id_z0"]),
                        "subhalo_id_z0": subhalo_id,
                        "raw_tree_basename": row["raw_tree_basename"],
                        "error": str(exc),
                    }
                    failures.append(failure)
                    write_download_summary(rows, downloaded, skipped, failures)
                    tqdm.write(f"[download failed; stopping] {sim_key}:{subhalo_id}: {exc}")
                    raise RuntimeError(
                        f"TNG tree download failed for {sim_key}:{subhalo_id}; "
                        f"see {_common.DOWNLOAD_FAILURES_JSON}."
                    ) from exc
    finally:
        session.close()

    write_download_summary(rows, downloaded, skipped, failures)
    print(f"Requested rows: {len(rows)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Raw tree directory: {_common.RAW_TREE_DIR}")
    print(f"Download summary: {_common.DOWNLOAD_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
