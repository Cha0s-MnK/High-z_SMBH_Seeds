#!/usr/bin/env python3
"""Shared helpers for the TNG50-1-Dark and TNG100-1-Dark tree pipeline."""

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


DEFAULT_DATA_DIR = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark")
TNG_API_ENV = "TNG_API_KEY"
BASE_URL = "https://www.tng-project.org/api/"
TNG_H = 0.6774
TNG_BOX_SIZES_CKPC_H = {
    "tng50_1_dark": 35000.0,
    "tng100_1_dark": 75000.0,
}
TNG_DM_PARTICLE_MASS_MSUN = {
    "tng50_1_dark": 5.3843825e5,
    "tng100_1_dark": 8.8565106e6,
}
TNG_TREE_MIN_PARTICLES = 500
TNG_TREE_MIN_MASS_MSUN = {
    sim_key: TNG_TREE_MIN_PARTICLES * particle_mass
    for sim_key, particle_mass in TNG_DM_PARTICLE_MASS_MSUN.items()
}


ROOT = DEFAULT_DATA_DIR
DATA_DIR = DEFAULT_DATA_DIR
RAW_TREE_DIR = DATA_DIR / "sublink_full_dark"
FIXED_TREE_DIR = DATA_DIR / "fixed_trees_large_spin_dark"

TARGETS_JSON = DATA_DIR / "targets_z0_dark.json"
TARGET_MANIFEST_CSV = DATA_DIR / "target_manifest_dark.csv"
SELECTION_LABELS_CSV = DATA_DIR / "halo_selection_labels_dark.csv"
SELECTED_HALO_IDS_TXT = DATA_DIR / "selected_halos_z0_dark.txt"
SELECTED_SUBHALO_IDS_TXT = DATA_DIR / "selected_subhalos_z0_dark.txt"

DOWNLOAD_FAILURES_JSON = DATA_DIR / "full_tree_download_failures.json"
DOWNLOAD_SUMMARY_JSON = DATA_DIR / "full_tree_download_summary.json"


def _build_simulations(data_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "tng50_1_dark": {
            "key": "tng50_1_dark",
            "name": "TNG50-1-Dark",
            "h": TNG_H,
            "box_size_ckpc_h": TNG_BOX_SIZES_CKPC_H["tng50_1_dark"],
            "groupcat_dir": data_dir / "groupcat_fields_tng50_1_dark",
            "snap_to_z_path": data_dir / "snaps2redshifts_tng50_1_dark.txt",
            "is_dark": True,
        },
        "tng100_1_dark": {
            "key": "tng100_1_dark",
            "name": "TNG100-1-Dark",
            "h": TNG_H,
            "box_size_ckpc_h": TNG_BOX_SIZES_CKPC_H["tng100_1_dark"],
            "groupcat_dir": data_dir / "groupcat_fields_tng100_1_dark",
            "snap_to_z_path": data_dir / "snaps2redshifts_tng100_1_dark.txt",
            "is_dark": True,
        },
    }


SIMULATIONS: dict[str, dict[str, Any]] = _build_simulations(DATA_DIR)


def configure_data_dir(data_dir: str | os.PathLike[str] | Path) -> Path:
    """Configure every pipeline path from one absolute data directory."""

    candidate = Path(data_dir).expanduser()
    if not candidate.is_absolute():
        raise ValueError(
            f"--data_dir must be an absolute path, received: {data_dir}"
        )
    configured = Path(os.path.abspath(candidate))

    global ROOT, DATA_DIR, RAW_TREE_DIR, FIXED_TREE_DIR
    global TARGETS_JSON, TARGET_MANIFEST_CSV, SELECTION_LABELS_CSV
    global SELECTED_HALO_IDS_TXT, SELECTED_SUBHALO_IDS_TXT
    global DOWNLOAD_FAILURES_JSON, DOWNLOAD_SUMMARY_JSON, SIMULATIONS

    ROOT = configured
    DATA_DIR = configured
    RAW_TREE_DIR = DATA_DIR / "sublink_full_dark"
    FIXED_TREE_DIR = DATA_DIR / "fixed_trees_large_spin_dark"
    TARGETS_JSON = DATA_DIR / "targets_z0_dark.json"
    TARGET_MANIFEST_CSV = DATA_DIR / "target_manifest_dark.csv"
    SELECTION_LABELS_CSV = DATA_DIR / "halo_selection_labels_dark.csv"
    SELECTED_HALO_IDS_TXT = DATA_DIR / "selected_halos_z0_dark.txt"
    SELECTED_SUBHALO_IDS_TXT = DATA_DIR / "selected_subhalos_z0_dark.txt"
    DOWNLOAD_FAILURES_JSON = DATA_DIR / "full_tree_download_failures.json"
    DOWNLOAD_SUMMARY_JSON = DATA_DIR / "full_tree_download_summary.json"
    SIMULATIONS = _build_simulations(DATA_DIR)
    return DATA_DIR


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


def snap_to_z_path(sim_key: str) -> Path:
    return get_simulation_spec(sim_key)["snap_to_z_path"]


def suite_prefixed_raw_name(sim_key: str, subhalo_id: int) -> str:
    key = get_simulation_spec(sim_key)["key"]
    return f"{key}_sublink_full_subhalo_{int(subhalo_id)}.hdf5"


def suite_prefixed_fixed_name(sim_key: str, file_index: int) -> str:
    key = get_simulation_spec(sim_key)["key"]
    return f"{key}_{int(file_index):04d}.dat"


def require_api_key() -> str:
    api_key = os.environ.get(TNG_API_ENV)
    if not api_key:
        raise RuntimeError(
            f"{TNG_API_ENV} is not set. Export it before running the TNG pipeline."
        )
    return api_key


def normalize_url(url: str) -> str:
    return (
        url.replace("http://www.tng-project.org/", "https://www.tng-project.org/")
        .replace("http://tng-project.org/", "https://www.tng-project.org/")
    )


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
            for key in (
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "all_proxy",
            ):
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


def read_json(path: Path | None = None) -> Any:
    target = TARGETS_JSON if path is None else path
    return json.loads(target.read_text(encoding="utf-8"))


def read_manifest(path: Path | None = None) -> list[dict[str, str]]:
    target = TARGET_MANIFEST_CSV if path is None else path
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]
