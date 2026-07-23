#!/usr/bin/env python3
"""Download full SubLink subtree HDF5 files for the selected dark-matter halos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
from tqdm.auto import tqdm

from _common import (
    DOWNLOAD_FAILURES_JSON,
    DOWNLOAD_SUMMARY_JSON,
    RAW_TREE_DIR,
    TARGETS_JSON,
    TARGET_MANIFEST_CSV,
    build_session,
    download_binary,
    ensure_dirs,
    read_json,
    read_manifest,
    request_json,
)


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
        "--limit",
        type=int,
        default=0,
        help="Download only the first N targets from the manifest. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload trees even if the raw HDF5 file already exists.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.1,
        help="Sleep duration between successful downloads.",
    )
    return parser.parse_args()


def validate_tree_file(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise RuntimeError(f"{path.name} is missing required datasets: {missing}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not TARGETS_JSON.exists():
        raise FileNotFoundError(f"Missing {TARGETS_JSON}. Run 1_select_targets.py first.")
    if not TARGET_MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGET_MANIFEST_CSV}. Run 1_select_targets.py first.")

    _ = read_json(TARGETS_JSON)
    rows = read_manifest()
    if args.limit > 0:
        rows = rows[: args.limit]

    downloaded = 0
    skipped = 0
    failures: list[dict[str, object]] = []

    session = build_session()
    try:
        with tqdm(rows, total=len(rows), desc="Full trees", unit="halo") as progress:
            for row in progress:
                progress.set_postfix(
                    downloaded=downloaded,
                    skipped=skipped,
                    failed=len(failures),
                )
                sim_key = row["simulation_key"]
                subhalo_id = int(row["subhalo_id_z0"])
                outpath = RAW_TREE_DIR / row["raw_tree_basename"]
                progress.set_description(f"Full trees {sim_key}:{subhalo_id}")
                if outpath.exists() and not args.overwrite:
                    validate_tree_file(outpath)
                    skipped += 1
                    continue

                try:
                    subhalo_url = row.get("subhalo_url_z0", "").strip()
                    if not subhalo_url:
                        raise RuntimeError("Manifest row is missing subhalo_url_z0.")
                    sub_json = request_json(session, subhalo_url)
                    tree_url = sub_json["trees"]["sublink"]
                    download_binary(session, tree_url, outpath)
                    validate_tree_file(outpath)
                    downloaded += 1
                    time.sleep(max(0.0, args.sleep_sec))
                except Exception as exc:
                    failures.append(
                        {
                            "file_index": int(row["file_index"]),
                            "simulation": row["simulation"],
                            "simulation_key": sim_key,
                            "halo_id_z0": int(row["halo_id_z0"]),
                            "subhalo_id_z0": subhalo_id,
                            "raw_tree_basename": row["raw_tree_basename"],
                            "error": str(exc),
                        }
                    )
                    tqdm.write(f"[download failed] {sim_key}:{subhalo_id}: {exc}")
    finally:
        session.close()

    summary = {
        "requested_rows": len(rows),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": len(failures),
        "raw_tree_dir": str(RAW_TREE_DIR),
        "simulations": sorted({row["simulation_key"] for row in rows}),
    }
    DOWNLOAD_FAILURES_JSON.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    DOWNLOAD_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Requested rows: {len(rows)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Raw tree directory: {RAW_TREE_DIR}")
    print(f"Download summary: {DOWNLOAD_SUMMARY_JSON}")

    if failures:
        raise RuntimeError(f"{len(failures)} full-tree downloads failed. See {DOWNLOAD_FAILURES_JSON}.")


if __name__ == "__main__":
    main()
