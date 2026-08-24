#!/usr/bin/env python3
"""Download full TNG SubLink trees for the selected dark-matter haloes."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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

_WORKER_SESSION = None


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
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel download worker processes (default: 1).",
    )
    args = parser.parse_args()
    if not Path(args.data_dir).expanduser().is_absolute():
        parser.error("--data_dir must be an absolute path.")
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer.")
    return args


def validate_tree_file(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise RuntimeError(f"{path.name} is missing required datasets: {missing}")


def _initialise_worker(data_dir: str) -> None:
    global _WORKER_SESSION
    _common.configure_data_dir(data_dir)
    _WORKER_SESSION = _common.build_session()


def _download_one(
    row: dict[str, str],
    data_dir: str,
    overwrite: bool,
) -> str:
    _common.configure_data_dir(data_dir)
    if _WORKER_SESSION is None:
        raise RuntimeError("Download worker session was not initialised.")

    sim_key = row["simulation_key"]
    subhalo_id = int(row["subhalo_id_z0"])
    outpath = _common.RAW_TREE_DIR / row["raw_tree_basename"]

    if outpath.exists() and not overwrite:
        validate_tree_file(outpath)
        return "skipped"

    subhalo_url = row.get("subhalo_url_z0", "").strip()
    if not subhalo_url:
        raise RuntimeError("Manifest row is missing subhalo_url_z0.")
    sub_json = _common.request_json(_WORKER_SESSION, subhalo_url)
    tree_url = sub_json["trees"]["sublink"]
    _common.download_binary(_WORKER_SESSION, tree_url, outpath)
    validate_tree_file(outpath)
    return "downloaded"


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


def _failure_record(row: dict[str, str], exc: Exception) -> dict[str, object]:
    return {
        "file_index": int(row["file_index"]),
        "simulation": row["simulation"],
        "simulation_key": row["simulation_key"],
        "halo_id_z0": int(row["halo_id_z0"]),
        "subhalo_id_z0": int(row["subhalo_id_z0"]),
        "raw_tree_basename": row["raw_tree_basename"],
        "error": str(exc),
    }


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

    _common.require_api_key()
    if rows:
        data_dir = str(_common.DATA_DIR)
        max_workers = min(args.jobs, len(rows))
        row_iter = iter(rows)
        active = {}

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialise_worker,
            initargs=(data_dir,),
        ) as executor:
            with tqdm(total=len(rows), desc="Full TNG trees", unit="halo") as progress:
                def submit_next() -> bool:
                    try:
                        row = next(row_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(_download_one, row, data_dir, args.overwrite)
                    active[future] = row
                    return True

                for _ in range(max_workers):
                    submit_next()

                while active:
                    completed, _ = wait(active, return_when=FIRST_COMPLETED)
                    batch_failed = False
                    for future in completed:
                        row = active.pop(future)
                        try:
                            status = future.result()
                        except Exception as exc:
                            failures.append(_failure_record(row, exc))
                            batch_failed = True
                        else:
                            if status == "downloaded":
                                downloaded += 1
                            elif status == "skipped":
                                skipped += 1
                            else:
                                raise RuntimeError(f"Unknown download result: {status}")

                        sim_key = row["simulation_key"]
                        subhalo_id = int(row["subhalo_id_z0"])
                        progress.set_description(f"Full trees {sim_key}:{subhalo_id}")
                        progress.update(1)
                        progress.set_postfix(
                            downloaded=downloaded,
                            skipped=skipped,
                            failed=len(failures),
                        )

                    if batch_failed:
                        for future in active:
                            future.cancel()

                        for future, row in list(active.items()):
                            if future.cancelled():
                                active.pop(future)
                                continue
                            try:
                                status = future.result()
                            except Exception as exc:
                                failures.append(_failure_record(row, exc))
                            else:
                                if status == "downloaded":
                                    downloaded += 1
                                elif status == "skipped":
                                    skipped += 1
                                else:
                                    raise RuntimeError(f"Unknown download result: {status}")
                            sim_key = row["simulation_key"]
                            subhalo_id = int(row["subhalo_id_z0"])
                            progress.set_description(f"Full trees {sim_key}:{subhalo_id}")
                            progress.update(1)
                            progress.set_postfix(
                                downloaded=downloaded,
                                skipped=skipped,
                                failed=len(failures),
                            )
                            active.pop(future)
                        break

                    for _ in range(max_workers - len(active)):
                        if not submit_next():
                            break

    failures.sort(key=lambda item: int(item["file_index"]))
    write_download_summary(rows, downloaded, skipped, failures)

    if failures:
        failure = failures[0]
        sim_key = str(failure["simulation_key"])
        subhalo_id = int(failure["subhalo_id_z0"])
        tqdm.write(f"[download failed; stopping] {sim_key}:{subhalo_id}: {failure['error']}")
        raise RuntimeError(
            f"TNG tree download failed for {sim_key}:{subhalo_id}; "
            f"see {_common.DOWNLOAD_FAILURES_JSON}."
        )

    print(f"Requested rows: {len(rows)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Raw tree directory: {_common.RAW_TREE_DIR}")
    print(f"Download summary: {_common.DOWNLOAD_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
