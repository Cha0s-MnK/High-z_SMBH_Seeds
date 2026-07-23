#!/usr/bin/env python3
"""Shared helpers for the Illustris-1-Dark + TNG50-1-Dark fixed-tree pipeline."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_TREE_DIR = DATA_DIR / "sublink_full_dark"
FIXED_TREE_DIR = DATA_DIR / "fixed_trees_large_spin_dark"

TARGETS_JSON = DATA_DIR / "targets_z0_dark.json"
TARGET_MANIFEST_CSV = DATA_DIR / "target_manifest_dark.csv"
SELECTION_LABELS_CSV = DATA_DIR / "halo_selection_labels_dark.csv"
FULL_PHYSICS_COUNTERPARTS_CSV = DATA_DIR / "full_physics_counterparts_z0.csv"
FULL_PHYSICS_COUNTERPARTS_SUMMARY_JSON = DATA_DIR / "full_physics_counterparts_summary.json"
EFF_RADIUS_CATALOGUE_CSV = DATA_DIR / "eff_radius_catalogue.csv"
NEUMAYER_FIG3_DIVIDER_JSON = DATA_DIR / "neumayer2020_fig3_divider.json"

SUBHALO_MATCHING_TO_DARK_FILES = {
    "illustris1": DATA_DIR / "subhalo_matching_to_dark_illustris1.hdf5",
    "tng50_1": DATA_DIR / "subhalo_matching_to_dark_tng50_1.hdf5",
}

DOWNLOAD_FAILURES_JSON = ROOT / "full_tree_download_failures.json"
DOWNLOAD_SUMMARY_JSON = ROOT / "full_tree_download_summary.json"

BASE_URL = "https://www.illustris-project.org/api/"
H100 = 0.704

SIMULATIONS: dict[str, dict[str, Any]] = {
    "illustris1_dark": {
        "key": "illustris1_dark",
        "name": "Illustris-1-Dark",
        "groupcat_dir": DATA_DIR / "groupcat_fields_illustris1_dark",
        "snap_to_z_path": DATA_DIR / "snaps2redshifts_illustris1_dark.txt",
        "full_physics_key": "illustris1",
        "is_dark": True,
    },
    "tng50_1_dark": {
        "key": "tng50_1_dark",
        "name": "TNG50-1-Dark",
        "groupcat_dir": DATA_DIR / "groupcat_fields_tng50_1_dark",
        "snap_to_z_path": DATA_DIR / "snaps2redshifts_tng50_1_dark.txt",
        "full_physics_key": "tng50_1",
        "is_dark": True,
    },
    "illustris1": {
        "key": "illustris1",
        "name": "Illustris-1",
        "groupcat_dir": DATA_DIR / "groupcat_fields_illustris1",
        "snap_to_z_path": DATA_DIR / "snaps2redshifts_illustris1.txt",
        "matching_to_dark_path": SUBHALO_MATCHING_TO_DARK_FILES["illustris1"],
        "dark_key": "illustris1_dark",
        "is_dark": False,
    },
    "tng50_1": {
        "key": "tng50_1",
        "name": "TNG50-1",
        "groupcat_dir": DATA_DIR / "groupcat_fields_tng50_1",
        "snap_to_z_path": DATA_DIR / "snaps2redshifts_tng50_1.txt",
        "matching_to_dark_path": SUBHALO_MATCHING_TO_DARK_FILES["tng50_1"],
        "dark_key": "tng50_1_dark",
        "is_dark": False,
    },
}


def ensure_dirs() -> None:
    paths = [DATA_DIR, RAW_TREE_DIR, FIXED_TREE_DIR]
    paths.extend(spec["groupcat_dir"] for spec in SIMULATIONS.values())
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_simulation_spec(key_or_name: str) -> dict[str, Any]:
    key_or_name = str(key_or_name).strip()
    if key_or_name in SIMULATIONS:
        return SIMULATIONS[key_or_name]
    for spec in SIMULATIONS.values():
        if spec["name"] == key_or_name:
            return spec
    raise KeyError(f"Unknown simulation key/name: {key_or_name}")


def suite_safe_name(sim_name: str) -> str:
    return get_simulation_spec(sim_name)["key"]


def groupcat_field_path(sim_key: str, field: str) -> Path:
    spec = get_simulation_spec(sim_key)
    return spec["groupcat_dir"] / f"{field}.hdf5"


def groupcat_snapshot_field_path(sim_key: str, snapnum: int, field: str) -> Path:
    spec = get_simulation_spec(sim_key)
    return spec["groupcat_dir"] / f"snap_{int(snapnum):03d}" / f"{field}.hdf5"


def full_physics_key_for_dark(sim_key: str) -> str:
    spec = get_simulation_spec(sim_key)
    if not spec.get("is_dark", False):
        return spec["key"]
    fp_key = spec.get("full_physics_key")
    if not fp_key:
        raise KeyError(f"No full-physics counterpart is registered for simulation {sim_key}")
    return str(fp_key)


def subhalo_matching_to_dark_path(sim_key_or_fp_key: str) -> Path:
    fp_key = full_physics_key_for_dark(sim_key_or_fp_key)
    spec = get_simulation_spec(fp_key)
    try:
        return Path(spec["matching_to_dark_path"])
    except KeyError as exc:
        raise KeyError(f"No subhalo_matching_to_dark cache path is registered for simulation {sim_key_or_fp_key}") from exc


def full_physics_groupcat_field_path(sim_key_or_fp_key: str, field: str) -> Path:
    fp_key = full_physics_key_for_dark(sim_key_or_fp_key)
    return groupcat_field_path(fp_key, field)


def full_physics_groupcat_snapshot_field_path(sim_key_or_fp_key: str, snapnum: int, field: str) -> Path:
    fp_key = full_physics_key_for_dark(sim_key_or_fp_key)
    return groupcat_snapshot_field_path(fp_key, snapnum, field)


def snap_to_z_path(sim_key: str) -> Path:
    spec = get_simulation_spec(sim_key)
    return spec["snap_to_z_path"]


def suite_prefixed_raw_name(sim_key: str, subhalo_id: int) -> str:
    sim_key = get_simulation_spec(sim_key)["key"]
    return f"{sim_key}_sublink_full_subhalo_{int(subhalo_id)}.hdf5"


def suite_prefixed_fixed_name(sim_key: str, file_index: int) -> str:
    sim_key = get_simulation_spec(sim_key)["key"]
    return f"{sim_key}_{int(file_index):04d}.dat"


def require_api_key() -> str:
    api_key = os.environ.get("ILLUSTRIS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ILLUSTRIS_API_KEY is not set. Export it before running the pipeline."
        )
    return api_key


def normalize_url(url: str) -> str:
    return url.replace("http://www.illustris-project.org/", "https://www.illustris-project.org/")


def build_session() -> requests.Session:
    api_key = require_api_key()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=0,
        backoff_factor=0.5,
        status_forcelist=(),
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"api-key": api_key, "Accept": "application/json"})
    session.trust_env = False
    return session


def request_json(session: requests.Session, url: str, retries: int = 4) -> dict[str, Any]:
    url = normalize_url(url)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=(12, 60))
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network retry
            last_err = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)
            else:
                raise
    assert last_err is not None
    raise last_err


def download_binary(
    session: requests.Session,
    url: str,
    outpath: Path,
    retries: int = 10,
    chunk_size: int = 1024 * 1024,
) -> None:
    del session, chunk_size
    url = normalize_url(url)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".part")
    last_err: Exception | None = None
    api_key = require_api_key()

    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            env = os.environ.copy()
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
                env.pop(key, None)
            cmd = [
                "wget",
                "--no-proxy",
                "--tries=1",
                "--timeout=180",
                "--header",
                f"api-key: {api_key}",
                "-O",
                str(tmp),
                url,
            ]
            subprocess.run(
                cmd,
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            tmp.replace(outpath)
            return
        except Exception as exc:  # pragma: no cover - network retry
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(min(10.0, 1.0 * attempt))
            else:
                if isinstance(exc, subprocess.CalledProcessError):
                    stderr = (exc.stderr or "").strip()
                    raise RuntimeError(
                        f"wget failed after {retries} direct no-proxy attempts for {url}: {stderr}"
                    ) from exc
                raise
    assert last_err is not None
    raise last_err


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path = TARGETS_JSON) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path = TARGET_MANIFEST_CSV) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]
