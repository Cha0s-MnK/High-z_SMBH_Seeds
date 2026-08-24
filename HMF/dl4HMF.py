#!/usr/bin/env python3
"""Download the minimal z=0 TNG group data required for the HMF plot."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_BASE_URL = "https://www.tng-project.org/api/"
TNG_API_ENV = "TNG_API_KEY"
DEFAULT_DATA_DIR = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark_HMF")
TNG_H = 0.6774
SNAPSHOT = 99
REDSHIFT_TOLERANCE = 1.0e-5
MASS_FIELD = "Group_M_Crit200"
GROUP_LEN_FIELD = "GroupLen"
MIN_GROUP_LEN = 500
SCHEMA_VERSION = 1
API_RETRIES = 4
DOWNLOAD_RETRIES = 10
JSON_TIMEOUT = (12, 60)
WGET_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class SimulationSpec:
    name: str
    box_side_cmpc_h: float
    dm_particle_mass_msun: float


SIMULATIONS = (
    SimulationSpec(
        name="TNG50-1-Dark",
        box_side_cmpc_h=35.0,
        dm_particle_mass_msun=5.3843825e5,
    ),
    SimulationSpec(
        name="TNG100-1-Dark",
        box_side_cmpc_h=75.0,
        dm_particle_mass_msun=8.8565106e6,
    ),
)


class ProductError(RuntimeError):
    """Raised when an API product or derived HMF file is not trustworthy."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download z=0 TNG50/TNG100 Group_M_Crit200 and GroupLen fields, "
            "apply GroupLen >= 500, and write plot-compatible HDF5 files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default=str(DEFAULT_DATA_DIR),
        help=(
            "Absolute directory for field caches, filtered catalogues, and the "
            "manifest."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refresh all field caches and replace derived aggregate files.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        parser.error("--data_dir must be an absolute path.")
    args.data_dir = data_dir.resolve()
    return args


def require_api_key() -> str:
    api_key = os.environ.get(TNG_API_ENV)
    if not api_key:
        raise RuntimeError(
            f"{TNG_API_ENV} is not set; export it before downloading TNG data."
        )
    return api_key


def normalise_url(url: str) -> str:
    return (
        str(url)
        .replace("http://www.tng-project.org/", "https://www.tng-project.org/")
        .replace("http://tng-project.org/", "https://tng-project.org/")
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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"api-key": api_key, "Accept": "application/json"})
    session.trust_env = False
    return session


def request_json(session: requests.Session, url: str) -> Any:
    url = normalise_url(url)
    last_error: Exception | None = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            response = session.get(url, timeout=JSON_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # pragma: no cover - network retry path
            last_error = error
            if attempt < API_RETRIES:
                time.sleep(0.5 * attempt)
    assert last_error is not None
    raise RuntimeError(f"TNG API request failed after {API_RETRIES} attempts: {url}") from last_error


def simulation_paths(data_dir: Path, spec: SimulationSpec) -> dict[str, Path]:
    cache_dir = data_dir / "field_cache" / spec.name
    output = (
        data_dir
        / spec.name
        / "output"
        / f"groups_{SNAPSHOT:03d}"
        / f"fof_subhalo_tab_{SNAPSHOT:03d}.0.hdf5"
    )
    return {
        "cache_dir": cache_dir,
        "mass_cache": cache_dir / f"groupcat_{SNAPSHOT:03d}_{MASS_FIELD}.hdf5",
        "group_len_cache": cache_dir / f"groupcat_{SNAPSHOT:03d}_{GROUP_LEN_FIELD}.hdf5",
        "output": output,
    }


def ensure_directories(data_dir: Path) -> None:
    (data_dir / "field_cache").mkdir(parents=True, exist_ok=True)
    for spec in SIMULATIONS:
        paths = simulation_paths(data_dir, spec)
        paths["cache_dir"].mkdir(parents=True, exist_ok=True)
        paths["output"].parent.mkdir(parents=True, exist_ok=True)


def resolve_snapshot(
    session: requests.Session,
    spec: SimulationSpec,
    simulations_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    simulation = simulations_by_name.get(spec.name)
    if simulation is None:
        raise ProductError(f"{spec.name} was not found in the TNG API simulation list.")

    snapshots_url = simulation.get("snapshots")
    if not snapshots_url:
        simulation_url = simulation.get("url")
        if not simulation_url:
            raise ProductError(f"{spec.name} has no snapshots URL in the API response.")
        snapshots_url = f"{str(simulation_url).rstrip('/')}/snapshots/"
    snapshots_payload = request_json(session, str(snapshots_url))
    snapshots = snapshots_payload.get("snapshots") if isinstance(snapshots_payload, dict) else snapshots_payload
    if not isinstance(snapshots, list):
        raise ProductError(f"{spec.name} returned a malformed snapshot list.")

    snapshot_item = None
    for item in snapshots:
        try:
            number = int(item["number"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProductError(f"{spec.name} returned a malformed snapshot entry.") from error
        if number == SNAPSHOT:
            snapshot_item = item
            break
    if snapshot_item is None or not snapshot_item.get("url"):
        raise ProductError(f"{spec.name} does not expose snapshot {SNAPSHOT}.")

    detail = request_json(session, str(snapshot_item["url"]))
    if not isinstance(detail, dict):
        raise ProductError(f"{spec.name} snapshot {SNAPSHOT} detail is not an object.")
    try:
        number = int(detail["number"])
        redshift = float(detail["redshift"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProductError(f"{spec.name} snapshot {SNAPSHOT} has invalid metadata.") from error
    if number != SNAPSHOT or not np.isfinite(redshift):
        raise ProductError(f"{spec.name} returned invalid snapshot-99 metadata.")
    if not np.isclose(redshift, 0.0, rtol=0.0, atol=REDSHIFT_TOLERANCE):
        raise ProductError(
            f"{spec.name} snapshot {SNAPSHOT} has redshift {redshift}, not z=0."
        )
    return {
        "number": number,
        "redshift": redshift,
        "url": normalise_url(str(detail.get("url", snapshot_item["url"]))),
    }


def field_api_url(spec: SimulationSpec, field: str) -> str:
    encoded_name = quote(spec.name, safe="-")
    return normalise_url(
        f"{API_BASE_URL}{encoded_name}/files/groupcat-{SNAPSHOT}/?Group={field}"
    )


def remove_path(path: Path) -> None:
    if path.exists():
        path.unlink()


def download_binary(url: str, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    temporary = outpath.with_suffix(outpath.suffix + ".part")
    api_key = require_api_key()
    config_fd, config_name = tempfile.mkstemp(
        prefix=".dl4HMF-wget-", dir=str(outpath.parent), text=True
    )
    config_path = Path(config_name)
    try:
        with os.fdopen(config_fd, "w", encoding="utf-8") as config_handle:
            config_handle.write(f"header = api-key: {api_key}\n")
        os.chmod(config_path, 0o600)
    except Exception:
        os.close(config_fd)
        remove_path(config_path)
        raise
    last_error: Exception | None = None
    try:
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            remove_path(temporary)
            environment = os.environ.copy()
            for key in (
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "all_proxy",
            ):
                environment.pop(key, None)
            command = [
                "wget",
                f"--config={config_path}",
                "--no-proxy",
                "--tries=1",
                f"--timeout={WGET_TIMEOUT_SECONDS}",
                "-O",
                str(temporary),
                normalise_url(url),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return
            except Exception as error:  # pragma: no cover - network retry path
                last_error = error
                remove_path(temporary)
                if attempt < DOWNLOAD_RETRIES:
                    time.sleep(min(10.0, float(attempt)))
        assert last_error is not None
        raise RuntimeError(
            f"wget failed after {DOWNLOAD_RETRIES} attempts for {url}"
        ) from last_error
    except BaseException:
        remove_path(temporary)
        raise
    finally:
        remove_path(config_path)


def dataset_is_numeric(dataset: h5py.Dataset, *, integer_only: bool = False) -> bool:
    allowed = {"i", "u"} if integer_only else {"i", "u", "f"}
    return dataset.dtype.kind in allowed


def validate_downloaded_field(path: Path, field: str) -> int:
    try:
        with h5py.File(path, "r") as handle:
            if "Group" not in handle or field not in handle["Group"]:
                raise ProductError(f"{path} is missing Group/{field}.")
            dataset = handle["Group"][field]
            if dataset.ndim != 1:
                raise ProductError(f"{path}: Group/{field} must be one-dimensional.")
            if not dataset_is_numeric(dataset):
                raise ProductError(f"{path}: Group/{field} has an invalid dtype.")
            if dataset.shape[0] == 0:
                raise ProductError(f"{path}: Group/{field} is empty.")
            return int(dataset.shape[0])
    except OSError as error:
        raise ProductError(f"Cannot read downloaded HDF5 file {path}.") from error


def set_cache_metadata(path: Path, spec: SimulationSpec, field: str) -> None:
    with h5py.File(path, "r+") as handle:
        handle.attrs["dl4HMF_schema_version"] = SCHEMA_VERSION
        handle.attrs["dl4HMF_simulation"] = spec.name
        handle.attrs["dl4HMF_snapshot"] = SNAPSHOT
        handle.attrs["dl4HMF_field"] = field


def read_string_attribute(attributes: h5py.AttributeManager, key: str) -> str:
    if key not in attributes:
        raise ProductError(f"Missing HDF5 provenance attribute {key!r}.")
    value = attributes[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def validate_cache(path: Path, spec: SimulationSpec, field: str) -> int:
    try:
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("dl4HMF_schema_version", -1)) != SCHEMA_VERSION:
                raise ProductError(f"{path} is not a dl4HMF cache with the expected schema.")
            if read_string_attribute(handle.attrs, "dl4HMF_simulation") != spec.name:
                raise ProductError(f"{path} belongs to a different simulation.")
            if int(handle.attrs.get("dl4HMF_snapshot", -1)) != SNAPSHOT:
                raise ProductError(f"{path} belongs to a different snapshot.")
            if read_string_attribute(handle.attrs, "dl4HMF_field") != field:
                raise ProductError(f"{path} contains a different field.")
    except OSError as error:
        raise ProductError(f"Cannot read cached HDF5 file {path}.") from error
    return validate_downloaded_field(path, field)


def ensure_field_cache(
    spec: SimulationSpec,
    field: str,
    path: Path,
    *,
    overwrite: bool,
) -> tuple[Path, int, str]:
    url = field_api_url(spec, field)
    if path.exists() and not overwrite:
        length = validate_cache(path, spec, field)
        return path, length, url

    temporary = path.with_suffix(path.suffix + ".part")
    download_binary(url, path)
    try:
        length = validate_downloaded_field(temporary, field)
        set_cache_metadata(temporary, spec, field)
        length = validate_cache(temporary, spec, field)
        temporary.replace(path)
    except Exception:
        remove_path(temporary)
        raise
    return path, length, url


def load_mass_and_group_length(
    mass_path: Path,
    group_len_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with h5py.File(mass_path, "r") as mass_file, h5py.File(group_len_path, "r") as length_file:
            mass_dataset = mass_file["Group"][MASS_FIELD]
            length_dataset = length_file["Group"][GROUP_LEN_FIELD]
            if not dataset_is_numeric(mass_dataset):
                raise ProductError(f"{mass_path}: invalid {MASS_FIELD} dtype.")
            if not dataset_is_numeric(length_dataset):
                raise ProductError(f"{group_len_path}: invalid {GROUP_LEN_FIELD} dtype.")
            if mass_dataset.ndim != 1 or length_dataset.ndim != 1:
                raise ProductError("Mass and GroupLen datasets must both be one-dimensional.")
            masses = np.asarray(mass_dataset[()], dtype=np.float64)
            raw_lengths = np.asarray(length_dataset[()])
    except (KeyError, OSError) as error:
        raise ProductError(
            f"Could not read {MASS_FIELD} and {GROUP_LEN_FIELD} from the caches."
        ) from error

    if masses.shape != raw_lengths.shape:
        raise ProductError(
            f"Mass and GroupLen lengths differ: {masses.size} versus {raw_lengths.size}."
        )
    if raw_lengths.dtype.kind not in {"i", "u", "f"}:
        raise ProductError(f"{GROUP_LEN_FIELD} is not integer-compatible.")
    try:
        lengths_float = np.asarray(raw_lengths, dtype=np.float64)
    except (OverflowError, ValueError) as error:
        raise ProductError(f"{GROUP_LEN_FIELD} cannot be converted to numbers.") from error
    if np.any(~np.isfinite(lengths_float)):
        raise ProductError(f"{GROUP_LEN_FIELD} contains non-finite values.")
    if np.any(lengths_float < 0.0):
        raise ProductError(f"{GROUP_LEN_FIELD} contains negative values.")
    if np.any(lengths_float != np.floor(lengths_float)):
        raise ProductError(f"{GROUP_LEN_FIELD} contains non-integral values.")
    if np.any(lengths_float > np.iinfo(np.int64).max):
        raise ProductError(f"{GROUP_LEN_FIELD} exceeds the supported integer range.")
    lengths = lengths_float.astype(np.int64)
    if masses.size == 0:
        raise ProductError("The cached group catalogue is empty.")
    return masses, lengths


def select_resolved_masses(
    masses_native: np.ndarray,
    group_lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    resolved_mask = group_lengths >= MIN_GROUP_LEN
    selected_masses = masses_native[resolved_mask]
    masses_msun = selected_masses * 1.0e10 / TNG_H
    finite_positive = np.isfinite(masses_msun) & (masses_msun > 0.0)
    counts = {
        "input_group_count": int(masses_native.size),
        "selected_group_count": int(np.count_nonzero(resolved_mask)),
        "rejected_group_count": int(np.count_nonzero(~resolved_mask)),
        "selected_finite_positive_mass_count": int(np.count_nonzero(finite_positive)),
    }
    if counts["selected_group_count"] == 0:
        raise ProductError(f"No groups satisfy {GROUP_LEN_FIELD} >= {MIN_GROUP_LEN}.")
    if counts["selected_finite_positive_mass_count"] == 0:
        raise ProductError(
            f"No selected groups have a finite positive {MASS_FIELD} mass."
        )
    return selected_masses, counts


def set_final_metadata(
    header: h5py.Group,
    spec: SimulationSpec,
    redshift: float,
    counts: dict[str, int],
) -> None:
    header.attrs["Redshift"] = float(redshift)
    header.attrs["dl4HMF_schema_version"] = SCHEMA_VERSION
    header.attrs["dl4HMF_product"] = "filtered_group_catalogue_mass"
    header.attrs["dl4HMF_simulation"] = spec.name
    header.attrs["dl4HMF_snapshot"] = SNAPSHOT
    header.attrs["dl4HMF_mass_field"] = MASS_FIELD
    header.attrs["dl4HMF_group_len_field"] = GROUP_LEN_FIELD
    header.attrs["dl4HMF_min_group_len"] = MIN_GROUP_LEN
    header.attrs["dl4HMF_selection"] = f"{GROUP_LEN_FIELD} >= {MIN_GROUP_LEN}"
    header.attrs["dl4HMF_input_group_count"] = counts["input_group_count"]
    header.attrs["dl4HMF_selected_group_count"] = counts["selected_group_count"]


def validate_final_file(
    path: Path,
    spec: SimulationSpec,
    redshift: float,
    expected_counts: dict[str, int],
) -> dict[str, int]:
    try:
        with h5py.File(path, "r") as handle:
            if "Header" not in handle or "Group" not in handle:
                raise ProductError(f"{path} lacks the required Header or Group group.")
            header = handle["Header"]
            group = handle["Group"]
            if MASS_FIELD not in group:
                raise ProductError(f"{path} is missing Group/{MASS_FIELD}.")
            if GROUP_LEN_FIELD in group:
                raise ProductError(f"{path} is not minimal: it contains Group/{GROUP_LEN_FIELD}.")
            dataset = group[MASS_FIELD]
            if dataset.ndim != 1 or not dataset_is_numeric(dataset):
                raise ProductError(f"{path}: final mass dataset has an invalid shape or dtype.")
            if int(dataset.shape[0]) != expected_counts["selected_group_count"]:
                raise ProductError(
                    f"{path}: row count {dataset.shape[0]} does not match "
                    f"the GroupLen selection count {expected_counts['selected_group_count']}."
                )
            actual_redshift = float(header.attrs["Redshift"])
            if not np.isclose(actual_redshift, redshift, rtol=0.0, atol=REDSHIFT_TOLERANCE):
                raise ProductError(f"{path}: Header redshift does not match the API.")
            required_attributes = {
                "dl4HMF_schema_version": SCHEMA_VERSION,
                "dl4HMF_product": "filtered_group_catalogue_mass",
                "dl4HMF_simulation": spec.name,
                "dl4HMF_snapshot": SNAPSHOT,
                "dl4HMF_mass_field": MASS_FIELD,
                "dl4HMF_group_len_field": GROUP_LEN_FIELD,
                "dl4HMF_min_group_len": MIN_GROUP_LEN,
                "dl4HMF_selection": f"{GROUP_LEN_FIELD} >= {MIN_GROUP_LEN}",
                "dl4HMF_input_group_count": expected_counts["input_group_count"],
                "dl4HMF_selected_group_count": expected_counts["selected_group_count"],
            }
            for key, expected in required_attributes.items():
                if key not in header.attrs:
                    raise ProductError(f"{path}: missing provenance attribute {key!r}.")
                actual = header.attrs[key]
                if isinstance(actual, bytes):
                    actual = actual.decode("utf-8")
                if isinstance(expected, str):
                    if str(actual) != expected:
                        raise ProductError(f"{path}: provenance attribute {key!r} differs.")
                elif int(actual) != expected:
                    raise ProductError(f"{path}: provenance attribute {key!r} differs.")
            masses_msun = np.asarray(dataset[()], dtype=np.float64) * 1.0e10 / TNG_H
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, ProductError):
            raise
        raise ProductError(f"Could not validate final HDF5 file {path}.") from error

    finite_positive_count = int(np.count_nonzero(np.isfinite(masses_msun) & (masses_msun > 0.0)))
    if finite_positive_count == 0:
        raise ProductError(f"{path}: final file contains no finite positive masses.")
    return {"final_group_count": int(masses_msun.size), "selected_finite_positive_mass_count": finite_positive_count}


def write_final_file(
    path: Path,
    spec: SimulationSpec,
    redshift: float,
    selected_masses: np.ndarray,
    counts: dict[str, int],
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    remove_path(temporary)
    try:
        with h5py.File(temporary, "w") as handle:
            group = handle.create_group("Group")
            group.create_dataset(MASS_FIELD, data=selected_masses, dtype=np.float64)
            header = handle.create_group("Header")
            set_final_metadata(header, spec, redshift, counts)
        validate_final_file(path=temporary, spec=spec, redshift=redshift, expected_counts=counts)
        temporary.replace(path)
    except Exception:
        remove_path(temporary)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    remove_path(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        remove_path(temporary)
        raise


def box_metadata(spec: SimulationSpec) -> dict[str, float]:
    side_cmpc = spec.box_side_cmpc_h / TNG_H
    return {
        "side_cmpc_h": spec.box_side_cmpc_h,
        "side_cmpc": side_cmpc,
        "volume_cmpc3": side_cmpc**3,
    }


def shell_command_text(command: list[str]) -> str:
    return (" " + chr(92) + "\n  ").join(shlex.quote(item) for item in command)


def main() -> None:
    args = parse_arguments()
    data_dir: Path = args.data_dir
    ensure_directories(data_dir)

    session = build_session()
    try:
        simulations_payload = request_json(session, API_BASE_URL)
        if not isinstance(simulations_payload, dict) or not isinstance(
            simulations_payload.get("simulations"), list
        ):
            raise ProductError("The TNG API root did not contain a simulations list.")
        simulations_by_name = {
            str(item["name"]): item
            for item in simulations_payload["simulations"]
            if isinstance(item, dict) and "name" in item
        }
        snapshots = {
            spec.name: resolve_snapshot(session, spec, simulations_by_name)
            for spec in SIMULATIONS
        }
        redshifts = [snapshots[spec.name]["redshift"] for spec in SIMULATIONS]
        if not np.allclose(redshifts, redshifts[0], rtol=0.0, atol=REDSHIFT_TOLERANCE):
            raise ProductError("TNG50 and TNG100 snapshot-99 redshifts differ.")
        redshift = float(redshifts[0])

        records: list[dict[str, Any]] = []
        for spec in SIMULATIONS:
            paths = simulation_paths(data_dir, spec)
            mass_path, mass_length, mass_url = ensure_field_cache(
                spec,
                MASS_FIELD,
                paths["mass_cache"],
                overwrite=args.overwrite,
            )
            group_len_path, group_len_length, group_len_url = ensure_field_cache(
                spec,
                GROUP_LEN_FIELD,
                paths["group_len_cache"],
                overwrite=args.overwrite,
            )
            if mass_length != group_len_length:
                raise ProductError(
                    f"{spec.name}: cached field lengths differ: "
                    f"{mass_length} versus {group_len_length}."
                )
            masses_native, group_lengths = load_mass_and_group_length(
                mass_path, group_len_path
            )
            selected_masses, counts = select_resolved_masses(masses_native, group_lengths)
            output_path = paths["output"]
            if output_path.exists() and not args.overwrite:
                final_counts = validate_final_file(
                    output_path, spec, redshift, counts
                )
            else:
                write_final_file(
                    output_path,
                    spec,
                    redshift,
                    selected_masses,
                    counts,
                )
                final_counts = validate_final_file(
                    output_path, spec, redshift, counts
                )
            records.append(
                {
                    "simulation": spec.name,
                    "snapshot": SNAPSHOT,
                    "redshift": redshift,
                    "h": TNG_H,
                    "box": box_metadata(spec),
                    "dm_particle_mass_msun": spec.dm_particle_mass_msun,
                    "fields": {
                        MASS_FIELD: {
                            "path": str(mass_path.resolve()),
                            "api_url": mass_url,
                            "unit": "1e10 M_sun / h",
                            "count": mass_length,
                        },
                        GROUP_LEN_FIELD: {
                            "path": str(group_len_path.resolve()),
                            "api_url": group_len_url,
                            "unit": "count",
                            "count": group_len_length,
                        },
                    },
                    "selection": {
                        "field": GROUP_LEN_FIELD,
                        "operator": ">=",
                        "threshold": MIN_GROUP_LEN,
                        "scope": "z=0 FoF group catalogue",
                        "meaning": "dark-matter particle count in the FoF group",
                    },
                    "counts": {**counts, **final_counts},
                    "output": {
                        "path": str(output_path.resolve()),
                        "plot_glob": str(output_path.parent / "fof_subhalo_tab_099.*.hdf5"),
                        "datasets": [f"Group/{MASS_FIELD}"],
                        "header_attributes": ["Redshift"],
                    },
                }
            )
            print(
                f"{spec.name}: {counts['input_group_count']} input groups, "
                f"{counts['selected_group_count']} with {GROUP_LEN_FIELD} >= "
                f"{MIN_GROUP_LEN}, {counts['rejected_group_count']} rejected; "
                f"final file {output_path}"
            )
    finally:
        session.close()

    data_dir = data_dir.resolve()
    plot_output = data_dir / "tng50_tng100_hmf_z0"
    plot_command = [
        "python",
        "/home/subonan/GitHub/HMF/plot_TNG_HMF.py",
        "--tng50",
        str(data_dir / "TNG50-1-Dark/output/groups_099/fof_subhalo_tab_099.*.hdf5"),
        "--tng100",
        str(data_dir / "TNG100-1-Dark/output/groups_099/fof_subhalo_tab_099.*.hdf5"),
        "--min-particles",
        str(MIN_GROUP_LEN),
        "--output",
        str(plot_output),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "product": "z=0 TNG HMF filtered by GroupLen >= 500",
        "snapshot": SNAPSHOT,
        "redshift": records[0]["redshift"],
        "h": TNG_H,
        "mass_field": MASS_FIELD,
        "mass_field_unit": "1e10 M_sun / h",
        "mass_conversion_to_msun": "Group_M_Crit200 * 1e10 / h",
        "group_len_field": GROUP_LEN_FIELD,
        "group_len_unit": "count",
        "selection": {
            "field": GROUP_LEN_FIELD,
            "operator": ">=",
            "threshold": MIN_GROUP_LEN,
            "applies_to": [spec.name for spec in SIMULATIONS],
            "scope": "z=0 FoF group catalogue",
            "meaning": "dark-matter particle count in the FoF group",
        },
        "full_box_volumes": {
            spec.name: box_metadata(spec) for spec in SIMULATIONS
        },
        "simulations": records,
        "final_files_contain": [f"Group/{MASS_FIELD}", "Header.attrs[Redshift]"],
        "final_files_do_not_contain": [f"Group/{GROUP_LEN_FIELD}"],
        "plot_command": plot_command,
        "plot_command_shell": shell_command_text(plot_command),
        "plot_output": str(plot_output.resolve()),
    }
    manifest_path = data_dir / "dl4HMF_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")
    print("Run the plotter with:")
    print(shell_command_text(plot_command))


if __name__ == "__main__":
    main()
