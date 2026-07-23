#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Self-contained Kong & Li 2026 plots for MBH-Mstar, UV aperture, QSO1 rotation, and BHMF."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlparse
import warnings

THREAD_CAP_DEFAULT = str(min(64, max(1, os.cpu_count() or 1)))
for env_name in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(env_name, THREAD_CAP_DEFAULT)

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     #"font.size": 10,
                     "mathtext.default": "regular",
                     "xtick.direction": "in",
                     "ytick.direction": "in",
                     #"xtick.top": True,
                     #"ytick.right": True,
                     #"axes.grid": False,
                     "text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"})
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CosmicAge2Redshift,
    G_Arepo,
    Mstar_SMHM,
    Redshift2CosmicAge,
    STD_DPI,
    check_finite,
    check_finite_non_negative,
)


# input params
DATA_ROOT = PROJECT_ROOT / "data"
MBH_MSTAR_DATA_PATH = DATA_ROOT / "Mbh-Mstar.csv"
BHMF_DATA_PATH = DATA_ROOT / "BHMF.csv"
MBH_MSTAR_MARKER_STYLES = {
    "Carnall+2023": {"marker": "h", "marker_size": 7.0, "edgecolor": "#4682b4", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Ding+2023": {"marker": "X", "marker_size": 7.2, "edgecolor": "#ffd400", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Goulding+2023": {"marker": "P", "marker_size": 7.0, "edgecolor": "#0b84c9", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Harikane+2023": {"marker": "D", "marker_size": 6.4, "edgecolor": "#4b0082", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Ivey+2026": {"marker": "o", "marker_size": 6.5, "edgecolor": "white", "edgewidth": 0.7, "alpha": 1.0, "zorder": 7},
    "Juodzbalis+2025": {"marker": "o", "marker_size": 6.6, "edgecolor": "#ff9900", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Juodzbalis+2026": {"marker": "*", "marker_size": 11.0, "edgecolor": "black", "edgewidth": 0.7, "alpha": 1.0, "zorder": 9},
    "Kokorev+2023": {"marker": "^", "marker_size": 7.0, "edgecolor": "#191970", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Maiolino+2024": {"marker": "*", "marker_size": 8.0, "edgecolor": "#40d8cf", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Stone+2024": {"marker": "*", "marker_size": 7.6, "edgecolor": "#cd853f", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Ubler+2023": {"marker": "D", "marker_size": 6.4, "edgecolor": "#4169e1", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Yue+2024": {"marker": "^", "marker_size": 7.4, "edgecolor": "#dca51d", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
}
NS_VALUE_DEFAULT = 2.0
# Figure 01 central-BH threshold in M_sun; the qualified selection is strict: M_SMBH_final > 100.0.
FIG01_DASHED_BH_THRESHOLD_MSUN = 100.0
# The qualified scatter band is deliberately lighter than the unchanged solid-band alpha=0.18.
FIG01_DASHED_SCATTER_ALPHA = 0.10

STATUS_SUNK_GC = -3
STATUS_WANDERER = -4
STATUS_SUNK_WANDERER = -5
STATUS_ALIVE = 1
STATUS_EXHAUSTED = -1
STATUS_TORN = -2
SATELLITE_BH_STATUSES = (STATUS_ALIVE, STATUS_WANDERER)

FIG08_TARGET_REDSHIFT = 7.04
FIG08_REDSHIFT_ATOL = 0.1
FIG08_MATCH_RADIUS_RANGE_PC = (12.5, 150.0)
FIG08_SCATTER_PERCENTILES = (16.0, 84.0)
FIG08_VELOCITY_SIN_I = 1.0
FIG08_RADIUS_MAX_PC = 170.0

FIGURE_04_FILENAME = "Fig.04_BHmasses.pdf"
FIGURE_05_FILENAME = "Fig.05_UVmag.pdf"
QSO1_REDSHIFT = 7.04
UV_AGE_REDSHIFT = 7.0
QSO1_MUV_AB = -15.60
QSO1_MUV_TOL_MAG = 0.5
QSO1_MOKA3D_LOGMBH = 7.7
QSO1_MOKA3D_LOGMBH_ERR = 0.3
QSO1_DIRECT_LOWER_LIMIT_LOGM = 6.94
QSO1_NSC_APERTURE_PC = 6.0
QSO1_SCORE_WEIGHT_MUV = 1.0
QSO1_SCORE_WEIGHT_KEPLERIAN = 1.0
UV_APERTURES_PC = np.arange(1.0, 8.0, 1.0)
# Total weights for the four absolute-radius velocity-point groups, inner to outer.
QSO1_VELOCITY_GROUP_WEIGHTS = (0.40, 0.30, 0.15, 0.15)
FIG04_XLIM_LOGM = (5.3, 8.4)
UV_CALIBRATION_PATH = DATA_ROOT / "UV" / "fsps_mist_chabrier_m1500_grid.csv"
UV_MODE_LABEL = "FSPS-MIST/Chabrier pure-stellar M1500(age,[Fe/H]) table"
UV_MIN_TABLE_AGE_GYR = 1.0e-4
UV_MIN_TABLE_FEH = -2.50
UV_MAX_TABLE_FEH = 0.50
UV_CALIBRATION_COLUMNS = ("age_gyr", "log10_age_gyr", "feh", "z_ratio", "M1500_AB_per_Msun")

FIG09_BHMF_SIDE_CMPC = 16.0
FIG09_BHMF_VOLUME_CMPC3 = FIG09_BHMF_SIDE_CMPC**3
FIG09_BIN_EDGES = np.logspace(2.0, 9.0, 32)
FIG06_BIN_EDGES = np.arange(4.0, 9.1, 0.25)

BH_TO_STELLAR_MASS_RATIOS = (0.01, 0.1, 1.0)
REINES_VOLONTERI_2015_NORM = 7.45
REINES_VOLONTERI_2015_SLOPE = 1.05
REINES_VOLONTERI_2015_SCATTER_DEX = 0.55
HALO_MASS_UNIT_LABEL = r"M_{\odot}"

def _stellar_mass_from_halo_mass_at_redshift(halo_mass, redshift):
    mh = np.asarray(halo_mass, dtype=float)
    z = np.asarray(redshift, dtype=float)
    mh_b, z_b = np.broadcast_arrays(mh, z)
    out = np.full(mh_b.shape, np.nan, dtype=float)
    for i, (mass, z_val) in enumerate(zip(mh_b.ravel(), z_b.ravel())):
        if np.isfinite(mass) and mass > 0.0 and np.isfinite(z_val) and z_val >= 0.0:
            out.ravel()[i] = Mstar_SMHM(float(mass), float(z_val), scatter=False)
    if np.isscalar(halo_mass) and np.isscalar(redshift):
        return float(out.reshape(-1)[0])
    return out


def _regular_log_bin_edges(values, step_dex):
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0, float(step_dex)], dtype=float)
    lo = float(step_dex) * math.floor(float(vals.min()) / float(step_dex))
    hi = float(step_dex) * math.ceil(float(vals.max()) / float(step_dex))
    if hi <= lo:
        hi = lo + float(step_dex)
    return np.arange(lo, hi + 0.5 * float(step_dex), float(step_dex), dtype=float)


def _bin_track(track, edges, x_log_col):
    x_log = track[x_log_col].to_numpy(dtype=float)
    y = track["M_SMBH_final"].to_numpy(dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = np.isfinite(x_log) & np.isfinite(y) & (x_log >= left)
        mask &= x_log <= right if idx == len(edges) - 2 else x_log < right
        if np.any(mask):
            y_sel = y[mask]
            rows.append({"logx_center": 0.5 * (left + right), "mean_mass": float(np.mean(y_sel)), "std_mass": float(np.std(y_sel))})
    return pd.DataFrame(rows)


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _row_float(row, name, default=np.nan):
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(value) else value


def _row_text(row, name, default=""):
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return str(value).strip()


def _log_error_to_linear(log_value, err_lo, err_hi):
    if not np.isfinite(log_value):
        return None
    value = 10.0**log_value
    lo = float(err_lo) if np.isfinite(err_lo) and err_lo > 0.0 else 0.0
    hi = float(err_hi) if np.isfinite(err_hi) and err_hi > 0.0 else 0.0
    if lo <= 0.0 and hi <= 0.0:
        return None
    return np.array([[value - 10.0 ** (log_value - lo)], [10.0 ** (log_value + hi) - value]], dtype=float)


def _reines_volonteri_2015_mbh(mstar_msun):
    return np.power(10.0, REINES_VOLONTERI_2015_NORM + REINES_VOLONTERI_2015_SLOPE * np.log10(np.asarray(mstar_msun) / 1.0e11))


def _ns_tag(ns_value):
    return f"{float(ns_value):.1f}".replace(".", "p")


def _read_comment_columns(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                text = line[1:].strip()
                if text:
                    return text.split()
    raise ValueError(f"Cannot find header columns in {path}")


def _read_headered_whitespace_table(path):
    columns = _read_comment_columns(path)
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    raw = raw.iloc[:, : len(columns)].copy()
    raw.columns = columns[: raw.shape[1]]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw

def _final_gcs_path(out_dir, ns_value):
    path = Path(out_dir).resolve() / f"ns{_ns_tag(ns_value)}" / f"finalGCs_ns{_ns_tag(ns_value)}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing final-GC catalogue: {path}")
    return path


def _halo_summary_by_z_path(out_dir, ns_value):
    path = Path(out_dir).resolve() / f"ns{_ns_tag(ns_value)}" / f"haloSummaryByZ_ns{_ns_tag(ns_value)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing redshift-resolved halo summary: {path}")
    return path


def _deposit_path(out_dir, ns_value):
    path = Path(out_dir).resolve() / f"ns{_ns_tag(ns_value)}" / f"depos_ns{_ns_tag(ns_value)}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing deposit profile: {path}")
    return path


def _rename_existing_columns(table, mapping):
    return table.rename(columns={old: new for old, new in mapping.items() if old in table.columns and new not in table.columns})


def _add_aliases(table, aliases):
    out = table.copy()
    for alias, source in aliases.items():
        if source in out.columns and alias not in out.columns:
            out[alias] = out[source]
    return out


def load_run_metadata(out_dir):
    path = Path(out_dir).resolve() / "run_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_halo_summary_by_z(out_dir, ns_value):
    mapping = {
        "hid_z0": "halo_id_z0",
        "z_out": "redshift",
        "logMh_z_msun": "log10_halo_mass_at_redshift",
        "M_NSC": "nsc_mass_msun",
        "M_SMBH_init": "central_bh_mass_init_msun",
        "M_SMBH_final": "central_bh_mass_final_msun",
        "z_depos_sampled": "deposit_sample_redshift",
        "lookback_depos_sampled_gyr": "deposit_sample_lookback_gyr",
        "depos_time_match_delta_gyr": "deposit_sample_time_delta_gyr",
    }
    table = _rename_existing_columns(pd.read_csv(_halo_summary_by_z_path(out_dir, ns_value)), mapping)
    required = ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift", "central_bh_mass_final_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing required columns after normalisation: {missing}")
    for col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if table[["halo_id_z0", "redshift", "central_bh_mass_final_msun"]].isna().any().any():
        raise ValueError("haloSummaryByZ contains non-finite halo IDs, redshifts, or central BH masses.")
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    bh = table["central_bh_mass_final_msun"].to_numpy(dtype=float)
    if np.any(bh < 0.0):
        raise ValueError("haloSummaryByZ contains negative central BH masses.")
    available = table["halo_mass_available"].to_numpy(dtype=float) == 1.0
    logmh = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    table["halo_mass_at_redshift_msun"] = np.nan
    valid = available & np.isfinite(logmh)
    table.loc[valid, "halo_mass_at_redshift_msun"] = np.power(10.0, logmh[valid])
    mstar = _stellar_mass_from_halo_mass_at_redshift(table["halo_mass_at_redshift_msun"].to_numpy(dtype=float), table["redshift"].to_numpy(dtype=float))
    table["mstar_z_smhm_msun"] = mstar
    table["logMstar_z_smhm_msun"] = np.where(np.isfinite(mstar) & (mstar > 0.0), np.log10(mstar), np.nan)
    return _add_aliases(
        table,
        {
            "hid_z0": "halo_id_z0",
            "z_out": "redshift",
            "logMh_z_msun": "log10_halo_mass_at_redshift",
            "M_NSC": "nsc_mass_msun",
            "M_SMBH_init": "central_bh_mass_init_msun",
            "M_SMBH_final": "central_bh_mass_final_msun",
            "z_depos_sampled": "deposit_sample_redshift",
            "lookback_depos_sampled_gyr": "deposit_sample_lookback_gyr",
            "depos_time_match_delta_gyr": "deposit_sample_time_delta_gyr",
        },
    )


def load_final_gc(out_dir, ns_value):
    mapping = {
        "M_IMBH_final": "imbh_mass_final_msun",
        "halo_id_z0": "halo_id_z0",
        "status": "status",
    }
    table = _rename_existing_columns(_read_headered_whitespace_table(_final_gcs_path(out_dir, ns_value)), mapping)
    table = _add_aliases(table, {"M_IMBH_final": "imbh_mass_final_msun"})
    missing = [name for name in ["status", "M_IMBH_final"] if name not in table.columns]
    if missing:
        raise ValueError(f"Final-GC table is missing required columns: {missing}")
    status_raw = pd.to_numeric(table["status"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(status_raw)):
        raise ValueError("Final-GC table contains non-finite status values.")
    status = status_raw.astype(int)
    if np.any(np.abs(status_raw - status.astype(float)) > 1.0e-8):
        raise ValueError("Final-GC table contains non-integer status values.")
    table["status"] = status
    table["M_IMBH_final"] = pd.to_numeric(table["M_IMBH_final"], errors="coerce")
    if table["M_IMBH_final"].isna().any() or (table["M_IMBH_final"] < 0.0).any():
        raise ValueError("Final-GC table contains non-finite or negative M_IMBH_final values.")
    return table


def _read_csv_required(path, required, numeric=()):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required cached data file: {path}")
    table = pd.read_csv(path)
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in numeric:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if numeric and table[list(numeric)].isna().any().any():
        raise ValueError(f"{path} contains non-finite numeric values in required columns.")
    return table


def load_uv_calibration(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing FSPS UV calibration table: {path}")
    table = pd.read_csv(path, comment="#")
    missing = [name for name in UV_CALIBRATION_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required FSPS UV columns: {missing}")
    table = table.loc[:, list(UV_CALIBRATION_COLUMNS)].copy()
    for col in UV_CALIBRATION_COLUMNS:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if table[list(UV_CALIBRATION_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains non-finite FSPS UV calibration values.")
    if np.any(table["age_gyr"].to_numpy(dtype=float) <= 0.0) or np.any(table["z_ratio"].to_numpy(dtype=float) <= 0.0):
        raise ValueError(f"{path} contains non-positive age_gyr or z_ratio values.")

    log_age_unique = np.sort(table["log10_age_gyr"].unique().astype(float))
    age_unique = np.sort(table["age_gyr"].unique().astype(float))
    feh_unique = np.sort(table["feh"].unique().astype(float))
    if len(log_age_unique) != len(age_unique):
        raise ValueError(f"{path} has inconsistent age_gyr and log10_age_gyr axes.")
    if len(table) != len(log_age_unique) * len(feh_unique):
        raise ValueError(f"{path} does not form a complete rectangular age-[Fe/H] grid.")
    if table.duplicated(subset=["log10_age_gyr", "feh"]).any():
        raise ValueError(f"{path} contains duplicate age-[Fe/H] grid nodes.")

    age_by_log = table.sort_values("log10_age_gyr").drop_duplicates("log10_age_gyr")
    age_axis = age_by_log["age_gyr"].to_numpy(dtype=float)
    log_age_axis = age_by_log["log10_age_gyr"].to_numpy(dtype=float)
    if not np.all(np.diff(log_age_axis) > 0.0) or not np.all(np.diff(age_axis) > 0.0):
        raise ValueError(f"{path} age axis is not strictly increasing.")
    if not np.allclose(np.log10(age_axis), log_age_axis, rtol=0.0, atol=2.0e-10):
        raise ValueError(f"{path} has inconsistent log10_age_gyr values.")

    sorted_table = table.sort_values(["log10_age_gyr", "feh"])
    expected_log_age = np.repeat(log_age_axis, len(feh_unique))
    expected_feh = np.tile(feh_unique, len(log_age_axis))
    if not np.allclose(sorted_table["log10_age_gyr"].to_numpy(dtype=float), expected_log_age, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{path} age rows are not rectangular after sorting.")
    if not np.allclose(sorted_table["feh"].to_numpy(dtype=float), expected_feh, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{path} [Fe/H] rows are not rectangular after sorting.")
    if not np.allclose(sorted_table["z_ratio"].to_numpy(dtype=float), np.power(10.0, expected_feh), rtol=2.0e-10, atol=1.0e-12):
        raise ValueError(f"{path} z_ratio values are inconsistent with feh = log10(Z/Zsun).")

    m1500_grid = sorted_table["M1500_AB_per_Msun"].to_numpy(dtype=float).reshape(len(log_age_axis), len(feh_unique))
    if np.any(~np.isfinite(m1500_grid)):
        raise ValueError(f"{path} contains non-finite M1500_AB_per_Msun values.")
    if not np.isclose(float(feh_unique[0]), UV_MIN_TABLE_FEH, rtol=0.0, atol=1.0e-8) or not np.isclose(float(feh_unique[-1]), UV_MAX_TABLE_FEH, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"{path} metallicity bounds must be {UV_MIN_TABLE_FEH:.2f} <= feh <= {UV_MAX_TABLE_FEH:.2f}.")
    if float(age_axis[0]) > UV_MIN_TABLE_AGE_GYR * (1.0 + 1.0e-8):
        raise ValueError(f"{path} minimum age {float(age_axis[0]):.6g} Gyr exceeds the planned minimum {UV_MIN_TABLE_AGE_GYR:.6g} Gyr.")

    return {
        "path": path,
        "age_gyr": age_axis,
        "log10_age_gyr": log_age_axis,
        "feh": feh_unique,
        "m1500_grid": m1500_grid,
        "age_min_gyr": float(age_axis[0]),
        "age_max_gyr": float(age_axis[-1]),
        "feh_min": float(feh_unique[0]),
        "feh_max": float(feh_unique[-1]),
        "uv_mode": UV_MODE_LABEL,
    }


def _mbh_mstar_base_label(label):
    return str(label).split("(", 1)[0].strip()


def load_mbh_mstar_observations():
    required = ["Mbh [M☉]", "Mstar [M☉]", "z", "label", "ADSABS"]
    table = _read_csv_required(MBH_MSTAR_DATA_PATH, required, numeric=["Mbh [M☉]", "Mstar [M☉]", "z"])
    if list(table.columns) != required:
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} must contain exactly these columns in this order: {required}.")
    table["label"] = table["label"].fillna("").astype(str).str.strip()
    table["ADSABS"] = table["ADSABS"].fillna("").astype(str).str.strip()
    if table["label"].eq("").any():
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains blank observational labels.")
    table["base_label"] = table["label"].map(_mbh_mstar_base_label)
    unknown = sorted(set(table["base_label"]) - set(MBH_MSTAR_MARKER_STYLES))
    if unknown:
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains labels without marker styles: {unknown}")
    urls = table["ADSABS"].to_numpy(dtype=str)
    invalid_urls = [url for url in urls if url and not (url.startswith("https://ui.adsabs.harvard.edu/abs/") or url.startswith("https://arxiv.org/abs/"))]
    if invalid_urls:
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains malformed ADSABS/arXiv URLs: {invalid_urls}")
    mass_columns = ["Mbh [M☉]", "Mstar [M☉]"]
    for column in mass_columns:
        values = table[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains non-finite or non-positive values in {column}.")
    redshifts = table["z"].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshifts)) or np.any(redshifts < 0.0):
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains non-finite or negative redshifts.")
    table["logMstar"] = np.log10(table["Mstar [M☉]"].to_numpy(dtype=float))
    table["logMBH"] = np.log10(table["Mbh [M☉]"].to_numpy(dtype=float))
    for name in ["marker", "marker_size", "edgecolor", "edgewidth", "alpha", "zorder"]:
        table[name] = table["base_label"].map(lambda label: MBH_MSTAR_MARKER_STYLES[label][name])
    return table


def load_juodzbalis2026_fig2():
    point_path = DATA_ROOT / "Juodzbalis+2026Fig2" / "juodzbalis2026_fig2_points.csv"
    curve_path = DATA_ROOT / "Juodzbalis+2026Fig2" / "juodzbalis2026_fig2_curves.csv"
    point_cols = ["component", "r_pc", "r_err_low_pc", "r_err_high_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s", "source_kind", "source_note"]
    curve_cols = ["curve", "r_pc", "v_km_s", "log10_mass_reference", "chi2_reduced", "source_kind", "source_note"]
    points = _read_csv_required(point_path, point_cols, numeric=["r_pc", "r_err_low_pc", "r_err_high_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"])
    curves = _read_csv_required(curve_path, curve_cols, numeric=["r_pc", "v_km_s", "log10_mass_reference", "chi2_reduced"])
    if sorted(set(["resolved_kinematics", "spectroastrometry", "spectroastrometry_fine"]) - set(points["component"])):
        raise ValueError("Juodzbalis Fig. 2 point table is missing one or more expected components.")
    if sorted(set(["point_mass_keplerian", "mw_nsc"]) - set(curves["curve"])):
        raise ValueError("Juodzbalis Fig. 2 curve table is missing one or more expected curves.")
    if np.any(curves["r_pc"].to_numpy(dtype=float) == 0.0):
        raise ValueError("Juodzbalis Fig. 2 curve table must not contain r_pc == 0 rows.")
    err_cols = ["r_err_low_pc", "r_err_high_pc", "v_err_low_km_s", "v_err_high_km_s"]
    if np.any(points[err_cols].dropna().to_numpy(dtype=float) < 0.0):
        raise ValueError("Juodzbalis Fig. 2 point table contains negative error values.")
    return points, curves


# Juodzbalis+2026 Fig. 3 values are vector-extracted/approximated from
# Figures/Mbh_comparison_new.pdf in the source bundle. They are intended for
# a compact comparison plot, not as a pixel-perfect reproduction.
JUODZBALIS2026_FIG3_BH_MASS_ROWS = [
    {
        "method": r"Furtak+24 virial H$\beta$",
        "group": "virial",
        "log10_mass": 7.60,
        "err_low": 0.25,
        "err_high": 0.25,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": r"Ji+25 virial H$\beta$",
        "group": "virial",
        "log10_mass": 7.59,
        "err_low": 0.20,
        "err_high": 0.20,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": r"D'Eugenio/Maiolino+25 virial H$\alpha$",
        "group": "virial",
        "log10_mass": 7.30,
        "err_low": 0.30,
        "err_high": 0.30,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": "Scattering-cocoon scenario",
        "group": "scattering",
        "log10_mass": 5.90,
        "err_low": 0.10,
        "err_high": 0.10,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": r"Bolometric luminosity, $L/L_{\rm Edd}=1$",
        "group": "bolometric",
        "log10_mass": 5.90,
        "err_low": 0.10,
        "err_high": 0.10,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": "This work, spectroastrometry lower limit",
        "group": "direct",
        "log10_mass": QSO1_DIRECT_LOWER_LIMIT_LOGM,
        "err_low": 0.0,
        "err_high": 0.0,
        "is_lower_limit": True,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": "This work, MOKA3D direct measurement",
        "group": "direct",
        "log10_mass": QSO1_MOKA3D_LOGMBH,
        "err_low": QSO1_MOKA3D_LOGMBH_ERR,
        "err_high": QSO1_MOKA3D_LOGMBH_ERR,
        "is_lower_limit": False,
        "show_moka_band": True,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
]


def load_juodzbalis2026_fig3_bh_masses():
    table = pd.DataFrame(JUODZBALIS2026_FIG3_BH_MASS_ROWS)
    required = ["method", "group", "log10_mass", "err_low", "err_high", "is_lower_limit", "show_moka_band", "source_note"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"Juodzbalis+2026 Fig. 3 table is missing columns: {missing}")
    table["log10_mass"] = [check_finite(value, name="Juodzbalis Fig. 3 log10 mass") for value in table["log10_mass"]]
    for name in ["err_low", "err_high"]:
        table[name] = [check_finite_non_negative(value, name=f"Juodzbalis Fig. 3 {name}") for value in table[name]]
    table["is_lower_limit"] = table["is_lower_limit"].map(_as_bool)
    table["show_moka_band"] = table["show_moka_band"].map(_as_bool)
    if len(table) != 7:
        raise ValueError("Juodzbalis+2026 Fig. 3 table should contain exactly seven reference rows.")
    return table


def _lookback_to_z0_gyr(redshift):
    return Redshift2CosmicAge(0.0) - Redshift2CosmicAge(float(redshift))


def interpolate_formed_mass_inside_aperture(deposit_profile, halo_id, aperture_pc=QSO1_NSC_APERTURE_PC):
    if "cumulative_formed_mass_msun" not in deposit_profile:
        return np.nan
    halo_ids = np.asarray(deposit_profile["halo_ids"], dtype=int)
    matches = np.flatnonzero(halo_ids == int(halo_id))
    if len(matches) != 1:
        return np.nan
    index = int(matches[0])
    rout_pc = np.asarray(deposit_profile["r_outer_kpc"][index], dtype=float) * 1.0e3
    cumulative = np.asarray(deposit_profile["cumulative_formed_mass_msun"][index], dtype=float)
    if (
        len(rout_pc) == 0
        or len(rout_pc) != len(cumulative)
        or np.any(~np.isfinite(rout_pc))
        or np.any(~np.isfinite(cumulative))
        or np.any(cumulative < 0.0)
        or np.any(np.diff(rout_pc) <= 0.0)
        or float(aperture_pc) < rout_pc[0]
        or float(aperture_pc) > rout_pc[-1]
    ):
        return np.nan
    return float(np.interp(float(aperture_pc), rout_pc, cumulative))


def select_qso1_gc_contributors(final_gc, halo_id, lookback_qso1_gyr):
    required = ["halo_id_z0", "status", "M_GC_final", "m_init_msun", "lookback_time_final_gyr", "lookback_time_init_gyr", "feh"]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        return final_gc.iloc[0:0].copy(), f"final-GC catalogue missing {missing}"
    rows = final_gc.loc[pd.to_numeric(final_gc["halo_id_z0"], errors="coerce").eq(int(halo_id))].copy()
    if len(rows) == 0:
        return rows, "no final-GC rows for halo"
    status = pd.to_numeric(rows["status"], errors="coerce").to_numpy(dtype=float)
    lookback_final = pd.to_numeric(rows["lookback_time_final_gyr"], errors="coerce").to_numpy(dtype=float)
    lookback_init = pd.to_numeric(rows["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(status)
        & (status.astype(int) == STATUS_SUNK_GC)
        & np.isfinite(lookback_final)
        & np.isfinite(lookback_init)
        & (lookback_final >= float(lookback_qso1_gyr))
        & (lookback_init >= float(lookback_qso1_gyr))
    )
    contributors = rows.loc[mask].copy()
    if len(contributors) == 0:
        return contributors, "no clean STATUS_SUNK_GC contributors before QSO1"
    return contributors, ""


def _blank_uv_result(formed_mass_msun, contributors, uv_calibration, missing_reason=""):
    formed_mass = float(formed_mass_msun) if np.isfinite(formed_mass_msun) else np.nan
    return {
        "formed_mass_msun": formed_mass,
        "weighted_age_gyr": np.nan,
        "weighted_feh": np.nan,
        "m1500_per_msun": np.nan,
        "M_UV": np.nan,
        "n_contributors": int(len(contributors)),
        "n_uv_valid_contributors": 0,
        "n_uv_age_nearest_grid": 0,
        "n_uv_feh_nearest_grid": 0,
        "n_uv_any_nearest_grid": 0,
        "min_raw_age_gyr": np.nan,
        "max_raw_age_gyr": np.nan,
        "min_raw_feh": np.nan,
        "max_raw_feh": np.nan,
        "min_eval_age_gyr": np.nan,
        "max_eval_age_gyr": np.nan,
        "min_eval_feh": np.nan,
        "max_eval_feh": np.nan,
        "uv_table_path": str(uv_calibration["path"]),
        "uv_mode": str(uv_calibration["uv_mode"]),
        "missing_reason": missing_reason,
    }


def _interpolate_uv_m1500(uv_calibration, age_gyr, feh):
    age_raw = np.asarray(age_gyr, dtype=float)
    feh_raw = np.asarray(feh, dtype=float)
    if age_raw.shape != feh_raw.shape:
        raise ValueError("UV interpolation age and [Fe/H] arrays must have the same shape.")

    log_age_grid = np.asarray(uv_calibration["log10_age_gyr"], dtype=float)
    age_grid = np.asarray(uv_calibration["age_gyr"], dtype=float)
    feh_grid = np.asarray(uv_calibration["feh"], dtype=float)
    m1500_grid = np.asarray(uv_calibration["m1500_grid"], dtype=float)
    if len(log_age_grid) < 2 or len(feh_grid) < 2:
        raise ValueError("FSPS UV calibration must have at least two age and [Fe/H] grid points.")

    m1500 = np.full(age_raw.shape, np.nan, dtype=float)
    eval_age = np.full(age_raw.shape, np.nan, dtype=float)
    eval_feh = np.full(age_raw.shape, np.nan, dtype=float)
    valid = np.isfinite(age_raw) & (age_raw > 0.0) & np.isfinite(feh_raw)
    age_used_nearest = valid & ((age_raw < age_grid[0]) | (age_raw > age_grid[-1]))
    feh_used_nearest = valid & ((feh_raw < feh_grid[0]) | (feh_raw > feh_grid[-1]))
    if np.any(feh_used_nearest):
        warnings.warn(
            f"FSPS UV metallicity outside native MIST grid clipped to {feh_grid[0]:.2f} <= feh <= {feh_grid[-1]:.2f}.",
            RuntimeWarning,
            stacklevel=2,
        )

    eval_age[valid] = np.clip(age_raw[valid], age_grid[0], age_grid[-1])
    eval_feh[valid] = np.clip(feh_raw[valid], feh_grid[0], feh_grid[-1])
    log_age_eval = np.full(age_raw.shape, np.nan, dtype=float)
    log_age_eval[valid] = np.log10(eval_age[valid])

    flat_valid = np.flatnonzero(valid.ravel())
    for flat_index in flat_valid:
        idx = np.unravel_index(int(flat_index), age_raw.shape)
        x = float(log_age_eval[idx])
        y = float(eval_feh[idx])
        i = int(np.clip(np.searchsorted(log_age_grid, x, side="right") - 1, 0, len(log_age_grid) - 2))
        j = int(np.clip(np.searchsorted(feh_grid, y, side="right") - 1, 0, len(feh_grid) - 2))
        x0, x1 = float(log_age_grid[i]), float(log_age_grid[i + 1])
        y0, y1 = float(feh_grid[j]), float(feh_grid[j + 1])
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        f00 = float(m1500_grid[i, j])
        f10 = float(m1500_grid[i + 1, j])
        f01 = float(m1500_grid[i, j + 1])
        f11 = float(m1500_grid[i + 1, j + 1])
        m1500[idx] = (1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10 + (1.0 - tx) * ty * f01 + tx * ty * f11

    return {
        "M1500_AB_per_Msun": m1500,
        "age_eval_gyr": eval_age,
        "feh_eval": eval_feh,
        "valid": valid,
        "age_used_nearest_grid": age_used_nearest,
        "feh_used_nearest_grid": feh_used_nearest,
        "any_used_nearest_grid": age_used_nearest | feh_used_nearest,
    }


def estimate_old_nsc_uv_mag(formed_mass_msun, contributors, lookback_eval_gyr, uv_calibration):
    result = _blank_uv_result(formed_mass_msun, contributors, uv_calibration)
    if not (np.isfinite(formed_mass_msun) and float(formed_mass_msun) > 0.0):
        result["missing_reason"] = "formed aperture mass unavailable"
        return result
    if len(contributors) == 0:
        result["missing_reason"] = "no usable GC-origin age weights"
        return result

    lookback_init = pd.to_numeric(contributors["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    age_gyr = lookback_init - float(lookback_eval_gyr)
    weights = pd.to_numeric(contributors["M_GC_final"], errors="coerce").to_numpy(dtype=float)
    fallback = pd.to_numeric(contributors["m_init_msun"], errors="coerce").to_numpy(dtype=float)
    use_fallback = ~np.isfinite(weights) | (weights <= 0.0)
    weights = weights.copy()
    weights[use_fallback] = fallback[use_fallback]
    feh = pd.to_numeric(contributors["feh"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(age_gyr) & (age_gyr > 0.0) & np.isfinite(feh) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        result["missing_reason"] = "no finite positive GC-origin UV age, [Fe/H], and weight tuples"
        return result

    age_sel = age_gyr[valid]
    feh_sel = feh[valid]
    weight_sel = weights[valid]
    interp = _interpolate_uv_m1500(uv_calibration, age_sel, feh_sel)
    m1500 = np.asarray(interp["M1500_AB_per_Msun"], dtype=float)
    interp_valid = np.isfinite(m1500)
    if not np.any(interp_valid):
        result["missing_reason"] = "non-finite FSPS UV interpolation"
        return result

    age_sel = age_sel[interp_valid]
    feh_sel = feh_sel[interp_valid]
    weight_sel = weight_sel[interp_valid]
    m1500 = m1500[interp_valid]
    eval_age = np.asarray(interp["age_eval_gyr"], dtype=float)[interp_valid]
    eval_feh = np.asarray(interp["feh_eval"], dtype=float)[interp_valid]
    age_nearest = np.asarray(interp["age_used_nearest_grid"], dtype=bool)[interp_valid]
    feh_nearest = np.asarray(interp["feh_used_nearest_grid"], dtype=bool)[interp_valid]
    any_nearest = np.asarray(interp["any_used_nearest_grid"], dtype=bool)[interp_valid]

    luminosity_per_msun = np.power(10.0, -0.4 * m1500)
    weighted_luminosity = float(np.average(luminosity_per_msun, weights=weight_sel))
    if not (np.isfinite(weighted_luminosity) and weighted_luminosity > 0.0):
        result["missing_reason"] = "non-finite composite stellar UV luminosity"
        return result

    m1500_per_msun = float(-2.5 * np.log10(weighted_luminosity))
    m_uv = float(m1500_per_msun - 2.5 * np.log10(float(formed_mass_msun)))

    result.update(
        {
            "weighted_age_gyr": float(np.average(age_sel, weights=weight_sel)),
            "weighted_feh": float(np.average(feh_sel, weights=weight_sel)),
            "m1500_per_msun": m1500_per_msun,
            "M_UV": m_uv,
            "n_uv_valid_contributors": int(len(age_sel)),
            "n_uv_age_nearest_grid": int(np.sum(age_nearest)),
            "n_uv_feh_nearest_grid": int(np.sum(feh_nearest)),
            "n_uv_any_nearest_grid": int(np.sum(any_nearest)),
            "min_raw_age_gyr": float(np.min(age_sel)),
            "max_raw_age_gyr": float(np.max(age_sel)),
            "min_raw_feh": float(np.min(feh_sel)),
            "max_raw_feh": float(np.max(feh_sel)),
            "min_eval_age_gyr": float(np.min(eval_age)),
            "max_eval_age_gyr": float(np.max(eval_age)),
            "min_eval_feh": float(np.min(eval_feh)),
            "max_eval_feh": float(np.max(eval_feh)),
            "missing_reason": "",
        }
    )
    return result


def estimate_uv_magnitude_apertures(deposit_profile, final_gc, halo_id, uv_calibration):
    selection_lookback_gyr = _lookback_to_z0_gyr(QSO1_REDSHIFT)
    age_lookback_gyr = _lookback_to_z0_gyr(UV_AGE_REDSHIFT)
    contributors, contributor_reason = select_qso1_gc_contributors(final_gc, halo_id, selection_lookback_gyr)
    if contributor_reason:
        raise ValueError(f"Fig. 05 selected halo {int(halo_id)} has no usable 7.04-selected UV contributors: {contributor_reason}.")

    rows = []
    for aperture_pc in UV_APERTURES_PC:
        formed_mass = interpolate_formed_mass_inside_aperture(deposit_profile, halo_id, float(aperture_pc))
        uv = estimate_old_nsc_uv_mag(formed_mass, contributors, age_lookback_gyr, uv_calibration)
        if not (np.isfinite(formed_mass) and float(formed_mass) > 0.0):
            raise ValueError(f"Fig. 05 selected halo {int(halo_id)} has no positive initially formed mass at R_UV={float(aperture_pc):.1f} pc.")
        if uv["missing_reason"] or not np.isfinite(float(uv["M_UV"])):
            reason = uv["missing_reason"] or "non-finite UV magnitude"
            raise ValueError(f"Fig. 05 selected halo {int(halo_id)} cannot evaluate R_UV={float(aperture_pc):.1f} pc: {reason}.")
        row = {"aperture_pc": float(aperture_pc)}
        row.update({key: value for key, value in uv.items() if key != "formed_mass_msun"})
        row["formed_mass_msun"] = float(formed_mass)
        rows.append(row)

    aperture_table = pd.DataFrame(rows)
    expected_apertures = np.arange(1.0, 8.0, 1.0)
    actual_apertures = aperture_table["aperture_pc"].to_numpy(dtype=float)
    if len(aperture_table) != len(expected_apertures) or not np.array_equal(actual_apertures, expected_apertures):
        raise ValueError(f"Fig. 05 selected halo {int(halo_id)} aperture series must contain exactly 1--7 pc, got {actual_apertures.tolist()}.")
    formed_mass_values = aperture_table["formed_mass_msun"].to_numpy(dtype=float)
    uv_values = aperture_table["M_UV"].to_numpy(dtype=float)
    if np.any(~np.isfinite(formed_mass_values)) or np.any(~np.isfinite(uv_values)):
        raise ValueError(f"Fig. 05 selected halo {int(halo_id)} aperture series contains non-finite values.")
    mass_tolerance = 1.0e-12 * max(1.0, float(np.max(formed_mass_values)))
    if np.any(np.diff(formed_mass_values) < -mass_tolerance):
        raise ValueError(f"Fig. 05 selected halo {int(halo_id)} initially formed mass is not cumulative over 1--7 pc.")
    if np.any(np.diff(uv_values) > 1.0e-10):
        raise ValueError(f"Fig. 05 selected halo {int(halo_id)} UV magnitude becomes fainter at larger aperture.")
    return aperture_table


def _fig02_weighted_velocity_points(points):
    required = ["component", "r_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"]
    missing = [name for name in required if name not in points.columns]
    if missing:
        raise ValueError(f"Juodzbalis Fig. 2 point table is missing velocity-score columns: {missing}")
    weights = np.asarray(QSO1_VELOCITY_GROUP_WEIGHTS, dtype=float)
    if len(weights) != 4 or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0) or not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("QSO1_VELOCITY_GROUP_WEIGHTS must contain four positive finite values that sum to unity.")

    table = points.copy()
    for col in ["r_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    numeric = table[["r_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Juodzbalis Fig. 2 point table contains non-finite velocity-score values.")
    if np.any(table["r_pc"].to_numpy(dtype=float) == 0.0):
        raise ValueError("Juodzbalis Fig. 2 velocity-score points must not have r_pc == 0.")
    if np.any(table[["v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Juodzbalis Fig. 2 velocity-score points must have positive velocity uncertainties.")
    table["abs_r_pc"] = np.abs(table["r_pc"].to_numpy(dtype=float))

    def _component_rows(component, expected_count):
        rows = table.loc[table["component"].eq(component)].copy()
        if len(rows) != expected_count:
            raise ValueError(f"Juodzbalis Fig. 2 velocity-score component {component!r} has {len(rows)} rows, expected {expected_count}.")
        return rows.sort_values("abs_r_pc").reset_index(drop=True)

    def _validate_signed_pair(rows, label):
        signs = sorted(np.sign(rows["r_pc"].to_numpy(dtype=float)).astype(int).tolist())
        if len(rows) != 2 or signs != [-1, 1]:
            raise ValueError(f"Juodzbalis Fig. 2 velocity-score group {label!r} must contain one negative- and one positive-radius point.")

    spectro = _component_rows("spectroastrometry", 2)
    spectro_fine = _component_rows("spectroastrometry_fine", 2)
    resolved = _component_rows("resolved_kinematics", 4)
    groups = [
        ("spectroastrometry_inner", spectro, float(weights[0])),
        ("spectroastrometry_fine", spectro_fine, float(weights[1])),
        ("resolved_inner", resolved.iloc[:2].copy(), float(weights[2])),
        ("resolved_outer", resolved.iloc[2:].copy(), float(weights[3])),
    ]

    weighted = []
    for name, rows, group_weight in groups:
        _validate_signed_pair(rows, name)
        rows = rows.sort_values("r_pc").copy()
        rows["velocity_group"] = name
        rows["velocity_group_weight"] = group_weight
        rows["point_weight"] = group_weight / float(len(rows))
        weighted.append(rows)
    out = pd.concat(weighted, ignore_index=True)
    if not np.isclose(float(out["point_weight"].sum()), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Juodzbalis Fig. 2 velocity-score point weights do not sum to unity.")
    return out


def _fig02_observed_velocity_score_by_halo(points, deposit_profile, z_rows):
    observed = _fig02_weighted_velocity_points(points)
    halo_ids, radius_pc, _stellar_cumulative, velocity_profiles = _fig08_total_velocity_profiles(deposit_profile, z_rows)
    obs_abs_r = np.abs(observed["r_pc"].to_numpy(dtype=float))
    obs_sign = np.sign(observed["r_pc"].to_numpy(dtype=float))
    obs_velocity = observed["v_km_s"].to_numpy(dtype=float)
    point_weight = observed["point_weight"].to_numpy(dtype=float)
    err_low = observed["v_err_low_km_s"].to_numpy(dtype=float)
    err_high = observed["v_err_high_km_s"].to_numpy(dtype=float)

    if np.any(obs_abs_r < radius_pc[0]) or np.any(obs_abs_r > radius_pc[-1]):
        reason = f"observed velocity radii {float(np.min(obs_abs_r)):.6g}-{float(np.max(obs_abs_r)):.6g} pc exceed model velocity grid {float(radius_pc[0]):.6g}-{float(radius_pc[-1]):.6g} pc"
        return {
            int(hid): {
                "keplerian_term": np.nan,
                "keplerian_chi2_weighted": np.nan,
                "keplerian_n_points": 0,
                "missing_reason": reason,
            }
            for hid in halo_ids
        }

    scores = {}
    for hid, velocity in zip(halo_ids, velocity_profiles):
        model_velocity = np.interp(obs_abs_r, radius_pc, velocity)
        signed_model_velocity = obs_sign * model_velocity
        residual = signed_model_velocity - obs_velocity
        sigma = np.where(residual >= 0.0, err_high, err_low)
        bad_sigma = ~np.isfinite(sigma) | (sigma <= 0.0)
        if np.any(bad_sigma):
            for idx in np.flatnonzero(bad_sigma):
                alternatives = np.asarray([err_low[idx], err_high[idx]], dtype=float)
                alternatives = alternatives[np.isfinite(alternatives) & (alternatives > 0.0)]
                if len(alternatives) > 0:
                    sigma[idx] = float(np.mean(alternatives))
        valid = np.isfinite(model_velocity) & np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0.0)
        if not np.all(valid):
            scores[int(hid)] = {
                "keplerian_term": np.nan,
                "keplerian_chi2_weighted": np.nan,
                "keplerian_n_points": int(np.sum(valid)),
                "missing_reason": "non-finite observed-velocity residual or uncertainty",
            }
            continue
        normalised_residual = residual / sigma
        chi2_weighted = float(np.sum(point_weight * normalised_residual**2))
        scores[int(hid)] = {
            "keplerian_term": float(np.sqrt(chi2_weighted)),
            "keplerian_chi2_weighted": chi2_weighted,
            "keplerian_n_points": int(len(observed)),
            "missing_reason": "",
        }
    return scores


def _candidate_no_score_error(out_dir, ns_value, score_table):
    n_candidates = int(len(score_table))
    finite_score = int(np.isfinite(score_table["score_keplerian_uv"]).sum()) if "score_keplerian_uv" in score_table else 0
    finite_keplerian = int(np.isfinite(score_table["keplerian_term"]).sum()) if "keplerian_term" in score_table else 0
    finite_uv = int(np.isfinite(score_table["M_UV"]).sum()) if "M_UV" in score_table else 0
    finite_formed = int((np.isfinite(score_table["formed_mass_6pc_msun"]) & (score_table["formed_mass_6pc_msun"] > 0.0)).sum()) if "formed_mass_6pc_msun" in score_table else 0
    usable_age = int(np.isfinite(score_table["weighted_age_gyr"]).sum()) if "weighted_age_gyr" in score_table else 0
    reasons = []
    if "missing_reason" in score_table:
        for row in score_table[["halo_id_z0", "missing_reason"]].itertuples(index=False):
            reason = str(row.missing_reason).strip()
            if reason:
                reasons.append(f"halo {int(row.halo_id_z0)}: {reason}")
            if len(reasons) >= 5:
                break
    reason_text = "; ".join(reasons) if reasons else "no candidate-specific missing-producer reason recorded"
    return (
        "No finite Keplerian+UV score is available for Fig. 02/Fig. 04 selection. "
        f"out_dir={Path(out_dir).resolve()}, ns-value={float(ns_value):.3g}, "
        f"redshift-selected candidates={n_candidates}, finite Keplerian terms={finite_keplerian}, "
        f"finite UV values={finite_uv}, finite Keplerian+UV scores={finite_score}, "
        f"finite formed 6 pc mass={finite_formed}, usable GC-origin age weights={usable_age}, "
        f"first missing producers: {reason_text}."
    )


def score_fig02_candidate_haloes(out_dir, ns_value, points, z_rows, deposit_profile, final_gc, uv_calibration):
    lookback_qso1 = _lookback_to_z0_gyr(QSO1_REDSHIFT)
    deposit_halo_ids = np.asarray(deposit_profile["halo_ids"], dtype=int)
    velocity_scores = _fig02_observed_velocity_score_by_halo(points, deposit_profile, z_rows)
    rows = []
    for row in z_rows.sort_values("halo_id_z0").itertuples(index=False):
        hid = int(getattr(row, "halo_id_z0"))
        redshift = float(getattr(row, "z_out", getattr(row, "redshift", np.nan)))
        nsc_mass = float(getattr(row, "M_NSC", getattr(row, "nsc_mass_msun", np.nan)))
        central_bh = float(getattr(row, "M_SMBH_final", getattr(row, "central_bh_mass_final_msun", np.nan)))
        log_nsc = float(np.log10(nsc_mass)) if np.isfinite(nsc_mass) and nsc_mass > 0.0 else np.nan
        log_bh = float(np.log10(central_bh)) if np.isfinite(central_bh) and central_bh > 0.0 else np.nan
        formed_mass = interpolate_formed_mass_inside_aperture(deposit_profile, hid, QSO1_NSC_APERTURE_PC)
        contributors, contributor_reason = select_qso1_gc_contributors(final_gc, hid, lookback_qso1)
        uv = estimate_old_nsc_uv_mag(formed_mass, contributors, lookback_qso1, uv_calibration)

        missing = []
        if not (np.isfinite(formed_mass) and formed_mass > 0.0):
            missing.append("formed 6 pc mass unavailable")
        if contributor_reason:
            missing.append(contributor_reason)
        if uv["missing_reason"]:
            missing.append(uv["missing_reason"])
        m_uv = float(uv["M_UV"])
        uv_term = (m_uv - QSO1_MUV_AB) / QSO1_MUV_TOL_MAG if np.isfinite(m_uv) else np.nan
        velocity_score = velocity_scores.get(
            hid,
            {
                "keplerian_term": np.nan,
                "keplerian_chi2_weighted": np.nan,
                "keplerian_n_points": 0,
                "missing_reason": "observed-velocity score unavailable",
            },
        )
        if velocity_score["missing_reason"]:
            missing.append(str(velocity_score["missing_reason"]))
        keplerian_term = float(velocity_score["keplerian_term"])
        keplerian_chi2_weighted = float(velocity_score["keplerian_chi2_weighted"])
        keplerian_n_points = int(velocity_score["keplerian_n_points"])
        if np.isfinite(keplerian_term) and np.isfinite(uv_term):
            score_keplerian_uv = float(np.sqrt(QSO1_SCORE_WEIGHT_KEPLERIAN * keplerian_term**2 + QSO1_SCORE_WEIGHT_MUV * uv_term**2))
        else:
            score_keplerian_uv = np.nan

        if hid in set(deposit_halo_ids.tolist()):
            velocity_index = int(np.flatnonzero(deposit_halo_ids == hid)[0])
        else:
            velocity_index = -1
            missing.append("deposit-profile halo index unavailable")

        rows.append(
            {
                "index": velocity_index,
                "halo_id_z0": hid,
                "redshift": redshift,
                "nsc_mass_msun": nsc_mass,
                "log10_nsc_mass": log_nsc,
                "central_bh_mass_msun": central_bh,
                "log10_central_bh_mass": log_bh,
                "formed_mass_6pc_msun": float(uv["formed_mass_msun"]) if np.isfinite(uv["formed_mass_msun"]) else np.nan,
                "weighted_age_gyr": float(uv["weighted_age_gyr"]),
                "weighted_feh": float(uv["weighted_feh"]) if np.isfinite(uv["weighted_feh"]) else np.nan,
                "m1500_per_msun": float(uv["m1500_per_msun"]) if np.isfinite(uv["m1500_per_msun"]) else np.nan,
                "M_UV": m_uv,
                "uv_term": float(uv_term) if np.isfinite(uv_term) else np.nan,
                "keplerian_term": keplerian_term,
                "keplerian_chi2_weighted": keplerian_chi2_weighted,
                "keplerian_n_points": keplerian_n_points,
                "score_keplerian_uv": score_keplerian_uv,
                "n_gc_contributors": int(uv["n_contributors"]),
                "n_uv_valid_contributors": int(uv["n_uv_valid_contributors"]),
                "n_uv_age_nearest_grid": int(uv["n_uv_age_nearest_grid"]),
                "n_uv_feh_nearest_grid": int(uv["n_uv_feh_nearest_grid"]),
                "n_uv_any_nearest_grid": int(uv["n_uv_any_nearest_grid"]),
                "min_raw_age_gyr": float(uv["min_raw_age_gyr"]) if np.isfinite(uv["min_raw_age_gyr"]) else np.nan,
                "max_raw_age_gyr": float(uv["max_raw_age_gyr"]) if np.isfinite(uv["max_raw_age_gyr"]) else np.nan,
                "min_raw_feh": float(uv["min_raw_feh"]) if np.isfinite(uv["min_raw_feh"]) else np.nan,
                "max_raw_feh": float(uv["max_raw_feh"]) if np.isfinite(uv["max_raw_feh"]) else np.nan,
                "min_eval_age_gyr": float(uv["min_eval_age_gyr"]) if np.isfinite(uv["min_eval_age_gyr"]) else np.nan,
                "max_eval_age_gyr": float(uv["max_eval_age_gyr"]) if np.isfinite(uv["max_eval_age_gyr"]) else np.nan,
                "min_eval_feh": float(uv["min_eval_feh"]) if np.isfinite(uv["min_eval_feh"]) else np.nan,
                "max_eval_feh": float(uv["max_eval_feh"]) if np.isfinite(uv["max_eval_feh"]) else np.nan,
                "uv_table_path": str(uv["uv_table_path"]),
                "uv_mode": str(uv["uv_mode"]),
                "missing_reason": "; ".join(dict.fromkeys(reason for reason in missing if reason)),
            }
        )

    score_table = pd.DataFrame(rows)
    finite = np.isfinite(score_table["score_keplerian_uv"].to_numpy(dtype=float))
    if not np.any(finite):
        raise ValueError(_candidate_no_score_error(out_dir, ns_value, score_table))
    best = score_table.loc[finite].sort_values(["score_keplerian_uv", "halo_id_z0"], ascending=[True, True]).iloc[0].to_dict()
    return score_table, best


def load_kritos2025_fig9():
    path = DATA_ROOT / "Kritos+2025Fig9" / "kritos2025_fig9_greene2020_mass_functions.csv"
    numeric = ["log10_mbh_msun", "mbh_msun", "linear_mpc3", "linear_low_mpc3", "linear_high_mpc3", "nsc_mpc3", "nsc_low_mpc3", "nsc_high_mpc3"]
    table = _read_csv_required(path, numeric + ["source_kind", "source_note"], numeric=numeric)
    if np.any(table[[name for name in numeric if name != "log10_mbh_msun"]].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Kritos Fig. 9 reference table contains non-positive plotted values.")
    return table


def load_bhmf_data():
    required = [
        "Phi [lgM☉⁻¹Mpc⁻³]", "sigma_Phi [lgM☉⁻¹Mpc⁻³]", "Mbh [M☉]", "sigma_Mbh",
        "colour", "face colour", "shape", "label", "ADSABS", "data",
    ]
    if not BHMF_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing BHMF catalogue: {BHMF_DATA_PATH}")
    table = pd.read_csv(BHMF_DATA_PATH, dtype=str, keep_default_na=False)
    if list(table.columns) != required:
        raise ValueError(f"BHMF catalogue columns must be exactly {required}, got {list(table.columns)}")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    numeric_columns = [required[0], required[1], required[2], required[3]]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column].replace("", np.nan), errors="coerce")
    if table.empty:
        raise ValueError("BHMF catalogue is empty.")
    if table[required[0]].isna().any() or ~np.isfinite(table[required[0]].to_numpy(dtype=float)).all():
        raise ValueError("BHMF catalogue contains non-finite Phi values.")
    if table[required[2]].isna().any() or ~np.isfinite(table[required[2]].to_numpy(dtype=float)).all() or np.any(table[required[2]] <= 0.0):
        raise ValueError("BHMF catalogue contains non-positive or non-finite Mbh values.")
    allowed_shapes = {"h", "s", "o", "^", "line"}
    if not set(table["shape"]).issubset(allowed_shapes):
        raise ValueError(f"BHMF catalogue contains unsupported shapes: {sorted(set(table['shape']) - allowed_shapes)}")
    if table["label"].eq("").any():
        raise ValueError("BHMF catalogue contains an empty label.")
    for column in ["colour", "face colour"]:
        invalid = [value for value in table[column].unique() if value.lower() != "none" and not mpl.colors.is_color_like(value)]
        if invalid:
            raise ValueError(f"BHMF catalogue contains invalid {column} values: {invalid}")
    for column in ["ADSABS", "data"]:
        invalid = []
        for value in table[column]:
            if value == "":
                continue
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                invalid.append(value)
        if invalid:
            raise ValueError(f"BHMF catalogue contains invalid {column} URLs: {invalid[:5]}")
    marker = table["shape"].ne("line")
    line = ~marker
    if table.loc[marker, required[1]].isna().any() or np.any(table.loc[marker, required[1]] < 0.0):
        raise ValueError("BHMF marker rows must have finite non-negative sigma_Phi values.")
    if table.loc[marker, required[3]].isna().any() or np.any(table.loc[marker, required[3]] <= 0.0):
        raise ValueError("BHMF marker rows must have positive sigma_Mbh values in dex.")
    if table.loc[line, required[1]].notna().any() or table.loc[line, required[3]].notna().any():
        raise ValueError("BHMF line rows must leave sigma_Phi and sigma_Mbh blank.")
    if table.loc[marker, "ADSABS"].eq("").any() or table.loc[line, "ADSABS"].eq("").any():
        raise ValueError("Every BHMF row must have an ADSABS or arXiv provenance URL.")
    expected_marker_counts = {"Fei+2026": 4, "Fei+2026 (w/o correction)": 4, "Matthee+2024": 3, "Taylor+2025": 5, "He+2024": 5}
    marker_counts = table.loc[marker, "label"].value_counts().to_dict()
    if marker_counts != expected_marker_counts:
        raise ValueError(f"BHMF marker counts must be {expected_marker_counts}, got {marker_counts}")
    line_labels = {
        "Heavy Eddington-limited", "Heavy Eddington-limited (lower envelope)", "Heavy Eddington-limited (upper envelope)",
        "Heavy Super-Eddington", "Heavy Super-Eddington (lower envelope)", "Heavy Super-Eddington (upper envelope)",
        "Light Eddington-limited", "Light Eddington-limited (lower envelope)", "Light Eddington-limited (upper envelope)",
    }
    if set(table.loc[line, "label"]) != line_labels:
        raise ValueError(f"BHMF line labels are incomplete or unexpected: {sorted(set(table.loc[line, 'label']))}")
    for label in line_labels:
        rows = table.loc[line & table["label"].eq(label)].sort_values(required[2])
        if len(rows) < 1000 or np.any(np.diff(rows[required[2]].to_numpy(dtype=float)) <= 0.0):
            raise ValueError(f"BHMF line group {label!r} is not a dense strictly increasing curve.")
    return table


def load_deposit_profile_for_redshift_summary(deposit_path, summary_rows, final_redshift):
    table = _read_headered_whitespace_table(deposit_path)
    required = ["halo_id_z0", "lookback_time_gyr", "bin_index", "r_inner_kpc", "r_outer_kpc", "m_star_no_evo_msun", "m_star_with_evo_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{deposit_path} is missing required deposit columns: {missing}")
    for col in required:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if table[required].isna().any().any():
        raise ValueError(f"{deposit_path} contains non-finite values in required deposit columns.")
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    table["bin_index"] = table["bin_index"].astype(int)

    summary = summary_rows.copy()
    if "redshift" not in summary.columns and "z_out" in summary.columns:
        summary["redshift"] = summary["z_out"]
    for col in summary.columns:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary["halo_id_z0"] = summary["halo_id_z0"].astype(int)
    if summary["halo_id_z0"].duplicated().any():
        dupes = summary.loc[summary["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].astype(int).unique().tolist()
        raise ValueError(f"Selected halo summary has duplicated halo_id_z0 values: {dupes[:10]}")

    final_age_gyr = Redshift2CosmicAge(float(final_redshift))
    grouped = {int(hid): group for hid, group in table.groupby("halo_id_z0", sort=True)}
    halo_ids, r_outer, cumulative, cumulative_formed = [], [], [], []
    for row in summary.sort_values("halo_id_z0").itertuples(index=False):
        hid = int(getattr(row, "halo_id_z0"))
        if hid not in grouped:
            raise ValueError(f"Deposit profile has no rows for selected halo_id_z0={hid}.")
        group = grouped[hid]
        unique_lookbacks = np.unique(np.sort(group["lookback_time_gyr"].to_numpy(dtype=float)))
        target_lookback = getattr(row, "lookback_depos_sampled_gyr", np.nan)
        if not np.isfinite(target_lookback):
            target_lookback = getattr(row, "deposit_sample_lookback_gyr", np.nan)
        if np.isfinite(target_lookback):
            block_lookback = float(unique_lookbacks[np.argmin(np.abs(unique_lookbacks - float(target_lookback)))])
            if abs(block_lookback - float(target_lookback)) > 1.0e-6:
                raise ValueError(f"Deposit profile for halo_id_z0={hid} does not contain the requested lookback.")
        else:
            target_redshift = getattr(row, "z_depos_sampled", np.nan)
            if not np.isfinite(target_redshift):
                target_redshift = getattr(row, "deposit_sample_redshift", np.nan)
            if not np.isfinite(target_redshift):
                target_redshift = getattr(row, "z_out", getattr(row, "redshift"))
            block_redshifts = np.array([CosmicAge2Redshift(final_age_gyr - float(lb)) for lb in unique_lookbacks], dtype=float)
            block_lookback = float(unique_lookbacks[np.argmin(np.abs(block_redshifts - float(target_redshift)))])

        block = group[np.isclose(group["lookback_time_gyr"].to_numpy(dtype=float), block_lookback, rtol=0.0, atol=1.0e-8)]
        ordered = block.sort_values("bin_index")
        bin_index = ordered["bin_index"].to_numpy(dtype=int)
        if len(bin_index) == 0 or not np.array_equal(bin_index, np.arange(1, len(bin_index) + 1, dtype=int)):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has non-contiguous bin_index values.")
        rin = ordered["r_inner_kpc"].to_numpy(dtype=float)
        rout = ordered["r_outer_kpc"].to_numpy(dtype=float)
        shell_formed = ordered["m_star_no_evo_msun"].to_numpy(dtype=float)
        shell = ordered["m_star_with_evo_msun"].to_numpy(dtype=float)
        if not np.isclose(rin[0], 0.0, rtol=0.0, atol=1.0e-10) or not np.isclose(rout[0], 1.0e-3, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has an unexpected innermost radial bin.")
        if np.any(~np.isfinite(rin)) or np.any(~np.isfinite(rout)) or np.any(~np.isfinite(shell)) or np.any(~np.isfinite(shell_formed)) or np.any(shell < 0.0) or np.any(shell_formed < 0.0):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} contains invalid radial or mass values.")
        if np.any(np.diff(rout) <= 0.0) or np.any(rout <= rin):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has invalid radial bin edges.")
        halo_ids.append(hid)
        r_outer.append(rout)
        cumulative.append(np.cumsum(shell))
        cumulative_formed.append(np.cumsum(shell_formed))
    return {
        "halo_ids": np.asarray(halo_ids, dtype=int),
        "r_outer_kpc": r_outer,
        "cumulative_mass_msun": cumulative,
        "cumulative_formed_mass_msun": cumulative_formed,
    }


def _plot_mbh_mstar_observations(ax, observations, norm, cmap):
    seen_labels = set()
    for _, row in observations.iterrows():
        log_mstar = float(row["logMstar"])
        log_mbh = float(row["logMBH"])
        colour = cmap(norm(float(row["z"])))
        label_base = _row_text(row, "label")
        label = None if label_base in seen_labels else label_base
        seen_labels.add(label_base)
        x = 10.0**log_mstar
        y = 10.0**log_mbh
        ax.plot(x, y, marker=row["marker"], linestyle="None", ms=float(row["marker_size"]), mfc=colour, mec=row["edgecolor"], mew=float(row["edgewidth"]), color=colour, alpha=float(row["alpha"]), label=label, zorder=int(row["zorder"]))


def plot_fig01_mbh_mstar(summary_by_z, observations, mass_bin_width_dex):
    plot_rows = summary_by_z[np.isfinite(summary_by_z["logMstar_z_smhm_msun"].to_numpy(dtype=float)) & np.isfinite(summary_by_z["M_SMBH_final"].to_numpy(dtype=float))].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite rows are available for Fig. 01.")
    z_values = np.sort(plot_rows["z_out"].unique())
    edges = _regular_log_bin_edges(plot_rows["logMstar_z_smhm_msun"], mass_bin_width_dex)
    x_limit_values = plot_rows["logMstar_z_smhm_msun"].to_numpy(dtype=float)
    if observations is not None and len(observations) > 0:
        x_limit_values = np.concatenate([x_limit_values, observations["logMstar"].to_numpy(dtype=float)])
    x_limit_edges = _regular_log_bin_edges(x_limit_values, mass_bin_width_dex)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=10.0, clip=True)
    #cmap = mpl.cm.viridis
    cmap = mpl.cm.jet

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        track = plot_rows[plot_rows["z_out"] == float(z_out)].copy()
        # Solid curves include every finite central-BH row; dashed curves and bands use the same statistic after strict M_SMBH_final > 100 M_sun selection.
        qualified_track = track.loc[track["M_SMBH_final"] > FIG01_DASHED_BH_THRESHOLD_MSUN].copy()
        if len(qualified_track) == 0:
            print(f"Fig. 01 qualified track omitted at z={float(z_out):.6g}: no central BH with M_SMBH_final > {FIG01_DASHED_BH_THRESHOLD_MSUN:g} M_sun.")
        binned = _bin_track(track, edges, "logMstar_z_smhm_msun")
        if len(binned) == 0:
            continue
        mean_mass = binned["mean_mass"].to_numpy(dtype=float)
        valid = np.isfinite(mean_mass) & (mean_mass > 0.0)
        if not np.any(valid):
            continue
        x = np.power(10.0, binned.loc[valid, "logx_center"].to_numpy(dtype=float))
        mean_mass = mean_mass[valid]
        std_mass = binned.loc[valid, "std_mass"].to_numpy(dtype=float)
        colour = cmap(norm(float(z_out)))
        ax.fill_between(x, np.maximum(mean_mass - std_mass, mean_mass * 1.0e-3), np.maximum(mean_mass + std_mass, mean_mass * 1.0e-3), color=colour, alpha=0.18, edgecolor="none")
        ax.plot(x, mean_mass, c=colour, lw=2.0)
        n_tracks += 1
        if len(qualified_track) == 0:
            continue
        qualified_binned = _bin_track(qualified_track, edges, "logMstar_z_smhm_msun")
        if len(qualified_binned) == 0:
            print(f"Fig. 01 qualified track omitted at z={float(z_out):.6g}: no finite qualified stellar-mass bins.")
            continue
        qualified_bin_index = np.searchsorted(edges, qualified_binned["logx_center"].to_numpy(dtype=float), side="right") - 1
        qualified_mean = np.full(len(edges) - 1, np.nan, dtype=float)
        qualified_std = np.full(len(edges) - 1, np.nan, dtype=float)
        qualified_mean[qualified_bin_index] = qualified_binned["mean_mass"].to_numpy(dtype=float)
        qualified_std[qualified_bin_index] = qualified_binned["std_mass"].to_numpy(dtype=float)
        qualified_valid = np.isfinite(qualified_mean) & (qualified_mean > 0.0)
        if not np.any(qualified_valid):
            continue
        qualified_x = np.power(10.0, 0.5 * (edges[:-1] + edges[1:]))
        qualified_lower = np.full(len(qualified_x), np.nan, dtype=float)
        qualified_upper = np.full(len(qualified_x), np.nan, dtype=float)
        qualified_lower[qualified_valid] = np.maximum(qualified_mean[qualified_valid] - qualified_std[qualified_valid], qualified_mean[qualified_valid] * 1.0e-3)
        qualified_upper[qualified_valid] = np.maximum(qualified_mean[qualified_valid] + qualified_std[qualified_valid], qualified_mean[qualified_valid] * 1.0e-3)
        ax.fill_between(qualified_x, qualified_lower, qualified_upper, color=colour, alpha=FIG01_DASHED_SCATTER_ALPHA, edgecolor="none", zorder=1)
        ax.plot(qualified_x, np.where(qualified_valid, qualified_mean, np.nan), c=colour, ls="--", lw=1.4, zorder=2)
    if n_tracks == 0:
        raise ValueError("All binned central-BH tracks are empty or non-positive for Fig. 01.")
    ax.plot([], [], c="black", ls="--", lw=1.4, label="Model: central BH > 100 M☉")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    colour_bar.set_ticks(np.arange(0.0, 10.1, 2.0))
    x_line = np.logspace(x_limit_edges[0], x_limit_edges[-1], 256)
    rv15 = _reines_volonteri_2015_mbh(x_line)
    rv15_scatter = 10.0**REINES_VOLONTERI_2015_SCATTER_DEX
    ax.fill_between(x_line, rv15 / rv15_scatter, rv15 * rv15_scatter, color="#2ca25f", alpha=0.16, edgecolor="none", label="Reines+Volonteri 2015", zorder=0)
    ax.plot(x_line, rv15, c="#238b45", lw=1.8, zorder=1)
    for ratio in BH_TO_STELLAR_MASS_RATIOS:
        ax.plot(x_line, ratio * x_line, c="#31a354", ls="--", lw=1.0, alpha=0.75, zorder=2)
        label_x = 10.0**max(min(6.0, float(x_limit_edges[-1]) - 0.35), float(x_limit_edges[0]) + 0.45)
        label_y = ratio * label_x
        if 1.0e2 < label_y < 1.0e11:
            ax.text(label_x, label_y, rf"$M_{{\rm BH}}/M_\ast={ratio:g}$", color="#31a354", fontsize=8.5, rotation=33.0, ha="center", va="bottom", clip_on=True, zorder=3)
    if observations is not None and len(observations) > 0:
        _plot_mbh_mstar_observations(ax, observations, norm, cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"Stellar mass at redshift $z$ from MPB $M_h(z)$ and SMHM [${HALO_MASS_UNIT_LABEL}$]")
    ax.set_ylabel(r"Central BH mass [$M_{\odot}$]")
    ax.set_xlim(left=10.0**x_limit_edges[0], right=10.0**x_limit_edges[-1])
    ax.set_ylim(bottom=1.0e2, top=1.0e11)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    legend = ax.legend(loc="lower right", fontsize=6.2, frameon=True, framealpha=0.85, ncol=2)
    for legend_text in legend.get_texts():
        if legend_text.get_text() == "Model: central BH > 100 M☉":
            legend_text.set_usetex(False)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def _fig08_bh_column(summary):
    for name in ["M_SMBH_final", "central_bh_mass_final_msun", "M_BH"]:
        if name in summary.columns:
            return name
    raise ValueError("haloSummaryByZ is missing a central-BH mass column required for Fig. 02.")


def _select_fig02_z_rows(summary_by_z):
    redshift = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(redshift) & (np.abs(redshift - FIG08_TARGET_REDSHIFT) < FIG08_REDSHIFT_ATOL)
    if not np.any(mask):
        raise ValueError(f"No haloSummaryByZ rows satisfy |z - {FIG08_TARGET_REDSHIFT:.2f}| < {FIG08_REDSHIFT_ATOL:.2f}.")
    rows = summary_by_z.loc[mask].copy()
    for col in rows.columns:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.sort_values("halo_id_z0").reset_index(drop=True)
    rows["halo_id_z0"] = rows["halo_id_z0"].astype(int)
    if rows["halo_id_z0"].duplicated().any():
        dupes = rows.loc[rows["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].astype(int).unique().tolist()
        raise ValueError(f"Fig. 02 z selection produced duplicate halo rows: {dupes[:10]}")
    bh = rows[_fig08_bh_column(rows)].to_numpy(dtype=float)
    if np.any(~np.isfinite(bh)) or np.any(bh < 0.0):
        raise ValueError("Fig. 02 selected rows contain invalid central BH masses.")
    return rows


def _fig08_error_arrays(rows):
    xerr = rows[["r_err_low_pc", "r_err_high_pc"]].to_numpy(dtype=float).T
    yerr = rows[["v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float).T
    return np.where(np.isfinite(xerr), xerr, 0.0), np.where(np.isfinite(yerr), yerr, 0.0)


def _plot_fig08_observed_curve(ax, curves, curve_name, label, **kwargs):
    used_label = False
    for sign in [-1.0, 1.0]:
        rows = curves.loc[curves["curve"].eq(curve_name) & (np.sign(curves["r_pc"].to_numpy(dtype=float)) == sign)]
        if len(rows) == 0:
            continue
        ordered = rows.sort_values("r_pc")
        ax.plot(ordered["r_pc"].to_numpy(dtype=float), ordered["v_km_s"].to_numpy(dtype=float), label=label if not used_label else None, **kwargs)
        used_label = True


def _fig08_signed_profile(radius_pc, velocity_km_s):
    radius = np.asarray(radius_pc, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    return np.concatenate([-radius[::-1], radius]), np.concatenate([-velocity[::-1], velocity])


def _fig08_total_velocity_profiles(deposit_profile, z_rows):
    bh_column = _fig08_bh_column(z_rows)
    bh_by_halo = {int(row.halo_id_z0): float(getattr(row, bh_column)) for row in z_rows[["halo_id_z0", bh_column]].itertuples(index=False)}
    summary_halos = set(bh_by_halo)
    profile_halos = set(int(value) for value in np.asarray(deposit_profile["halo_ids"], dtype=int))
    if summary_halos != profile_halos:
        raise ValueError("Fig. 02 deposit-profile halo IDs do not match selected redshift rows.")
    first_radii_pc = [float(rout[0]) * 1.0e3 for rout in deposit_profile["r_outer_kpc"]]
    last_radii_pc = [float(rout[-1]) * 1.0e3 for rout in deposit_profile["r_outer_kpc"]]
    r_min = max(1.0, max(first_radii_pc))
    r_max = min(FIG08_RADIUS_MAX_PC, min(last_radii_pc))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max < FIG08_MATCH_RADIUS_RANGE_PC[1]:
        raise ValueError(f"Fig. 02 deposit radial coverage is insufficient: {r_min:.6g}-{r_max:.6g} pc.")
    radius_pc = np.unique(np.concatenate([np.geomspace(r_min, r_max, 256), np.asarray(FIG08_MATCH_RADIUS_RANGE_PC, dtype=float)]))
    stellar_cumulative, velocity_profiles = [], []
    for hid, rout_kpc, cumulative in zip(deposit_profile["halo_ids"], deposit_profile["r_outer_kpc"], deposit_profile["cumulative_mass_msun"]):
        rout_pc = np.asarray(rout_kpc, dtype=float) * 1.0e3
        cum_mass = np.asarray(cumulative, dtype=float)
        if radius_pc[0] < rout_pc[0] or radius_pc[-1] > rout_pc[-1]:
            raise ValueError(f"Fig. 02 common radius grid exceeds deposit coverage for halo_id_z0={int(hid)}.")
        stellar = np.interp(radius_pc, rout_pc, cum_mass)
        velocity = FIG08_VELOCITY_SIN_I * np.sqrt(G_Arepo * (stellar + bh_by_halo[int(hid)]) / radius_pc)
        if np.any(~np.isfinite(velocity)):
            raise ValueError(f"Fig. 02 velocity profile is non-finite for halo_id_z0={int(hid)}.")
        stellar_cumulative.append(stellar)
        velocity_profiles.append(velocity)
    return np.asarray(deposit_profile["halo_ids"], dtype=int), radius_pc, np.asarray(stellar_cumulative, dtype=float), np.asarray(velocity_profiles, dtype=float)


def plot_fig02_rotation_curve(points, curves, z_rows, deposit_profile, best):
    halo_ids, radius_pc, stellar_cumulative, velocity_profiles = _fig08_total_velocity_profiles(deposit_profile, z_rows)
    matches = np.flatnonzero(halo_ids == int(best["halo_id_z0"]))
    if len(matches) != 1:
        raise ValueError(f"Fig. 02 Keplerian+UV best halo {int(best['halo_id_z0'])} is not present in the velocity-profile grid.")
    best_index = int(matches[0])
    median_velocity = np.median(velocity_profiles, axis=0)
    mean_velocity = np.mean(velocity_profiles, axis=0)
    low_velocity, high_velocity = np.percentile(velocity_profiles, FIG08_SCATTER_PERCENTILES, axis=0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    _plot_fig08_observed_curve(ax, curves, "point_mass_keplerian", r"QSO1 point mass, $\log M_{\rm BH}=6.75$", color="black", linewidth=1.7, zorder=4)
    _plot_fig08_observed_curve(ax, curves, "mw_nsc", "MW-like NSC model", color="0.45", linestyle="dashdot", linewidth=1.5, zorder=3)
    for component, colour, marker, size, label in [
        ("resolved_kinematics", "tab:blue", "o", 5.0, "Resolved kinematics"),
        ("spectroastrometry", "magenta", "X", 6.0, "Spectroastrometry"),
        ("spectroastrometry_fine", "orchid", "P", 5.5, "Fine spectroastrometry"),
    ]:
        rows = points.loc[points["component"].eq(component)]
        if len(rows) == 0:
            continue
        xerr, yerr = _fig08_error_arrays(rows)
        ax.errorbar(rows["r_pc"].to_numpy(dtype=float), rows["v_km_s"].to_numpy(dtype=float), xerr=xerr, yerr=yerr, fmt=marker, ms=size, color=colour, ecolor=colour, elinewidth=1.0, markeredgecolor=colour, markerfacecolor=colour, capsize=0.0, linestyle="none", label=label, zorder=6)
    ax.fill_between(radius_pc, low_velocity, high_velocity, color="tab:green", alpha=0.16, linewidth=0.0, label="z~7 stack 16-84%")
    ax.fill_between(-radius_pc[::-1], -high_velocity[::-1], -low_velocity[::-1], color="tab:green", alpha=0.16, linewidth=0.0)
    signed_r, signed_median = _fig08_signed_profile(radius_pc, median_velocity)
    _, signed_mean = _fig08_signed_profile(radius_pc, mean_velocity)
    _, signed_best = _fig08_signed_profile(radius_pc, velocity_profiles[best_index])
    ax.plot(signed_r, signed_median, color="tab:green", linewidth=1.8, label="z~7 median simulation")
    ax.plot(signed_r, signed_mean, color="tab:green", linewidth=1.2, linestyle="--", label="z~7 mean simulation")
    ax.plot(signed_r, signed_best, color="tab:red", linewidth=1.5, linestyle="-", label=f"Best Keplerian+UV halo {int(best['halo_id_z0'])}")
    ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.set_xlim(-FIG08_RADIUS_MAX_PC, FIG08_RADIUS_MAX_PC)
    ax.set_ylim(-72.0, 72.0)
    ax.set_xlabel(r"Projected radius $r$ [pc]")
    ax.set_ylabel(r"Line-of-sight velocity $v$ [km s$^{-1}$]")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2, ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def _plot_fig04_logmass_marker(ax, x_value, y_value, *, marker, colour, label, size=7.0, zorder=6):
    xmin, xmax = FIG04_XLIM_LOGM
    if np.isfinite(x_value) and xmin <= float(x_value) <= xmax:
        ax.plot(float(x_value), y_value, marker=marker, ms=size, mfc=colour, mec=colour, color=colour, linestyle="none", label=label, zorder=zorder)
    elif np.isfinite(x_value):
        if float(x_value) > xmax:
            edge_x = xmax - 0.035
            edge_marker = ">"
            text_x = xmax - 0.72
            ha = "right"
        else:
            edge_x = xmin + 0.035
            edge_marker = "<"
            text_x = xmin + 0.72
            ha = "left"
        ax.plot(edge_x, y_value, marker=edge_marker, ms=size, mfc=colour, mec=colour, color=colour, linestyle="none", label=label, clip_on=False, zorder=zorder)
        ax.annotate(
            rf"$\log M={float(x_value):.2f}$",
            xy=(edge_x, y_value),
            xytext=(text_x, y_value + 0.20),
            ha=ha,
            va="bottom",
            fontsize=7.0,
            color=colour,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": colour},
        )


def plot_fig04_bh_masses(fig3_reference, best, uv_estimate):
    labels = fig3_reference["method"].tolist() + [
        r"Best Keplerian+UV halo NSC stellar mass ($<6$ pc)",
        "Best Keplerian+UV halo central BH",
    ]
    y_positions = np.arange(len(labels) - 1, -1, -1, dtype=float)
    y_by_label_index = {index: float(y_positions[index]) for index in range(len(labels))}

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.7))
    ax.axvspan(QSO1_MOKA3D_LOGMBH - QSO1_MOKA3D_LOGMBH_ERR, QSO1_MOKA3D_LOGMBH + QSO1_MOKA3D_LOGMBH_ERR, color="#7b3294", alpha=0.14, lw=0.0, label="MOKA3D 1 sigma")

    styles = {
        "virial": {"colour": "black", "marker": "o", "label": "Virial estimates", "size": 5.3},
        "scattering": {"colour": "#756bb1", "marker": "D", "label": "Scattering scenario", "size": 5.2},
        "bolometric": {"colour": "#d7301f", "marker": "s", "label": r"$L_{\rm bol}$ estimate", "size": 5.4},
        "direct": {"colour": "#238b45", "marker": "*", "label": "Direct estimates", "size": 8.0},
    }
    used_labels = set()
    for index, row in fig3_reference.iterrows():
        style = styles[str(row["group"])]
        label = None if style["label"] in used_labels else style["label"]
        used_labels.add(style["label"])
        x_value = float(row["log10_mass"])
        y_value = y_by_label_index[int(index)]
        colour = style["colour"]
        if _as_bool(row["is_lower_limit"]):
            ax.plot(x_value, y_value, marker=style["marker"], ms=style["size"], mfc=colour, mec=colour, color=colour, linestyle="none", label=label, zorder=6)
            ax.annotate("", xy=(min(FIG04_XLIM_LOGM[1] - 0.08, x_value + 0.42), y_value), xytext=(x_value, y_value), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": colour}, zorder=5)
        else:
            xerr = np.array([[float(row["err_low"])], [float(row["err_high"])]], dtype=float)
            xerr = None if np.all(xerr <= 0.0) else xerr
            ax.errorbar(x_value, y_value, xerr=xerr, fmt=style["marker"], ms=style["size"], color=colour, mfc=colour, mec=colour, ecolor=colour, elinewidth=1.0, capsize=2.2, linestyle="none", label=label, zorder=6)

    nsc_y = y_by_label_index[len(fig3_reference)]
    bh_y = y_by_label_index[len(fig3_reference) + 1]
    _plot_fig04_logmass_marker(ax, float(best["log10_nsc_mass"]), nsc_y, marker="P", colour="#1f9e89", label="Model NSC mass", size=7.0)
    _plot_fig04_logmass_marker(ax, float(best["log10_central_bh_mass"]), bh_y, marker="X", colour="#e6550d", label="Model central BH", size=7.0)

    m_uv = float(uv_estimate.get("M_UV", np.nan)) if isinstance(uv_estimate, dict) else np.nan
    if np.isfinite(m_uv):
        uv_text = rf"stellar old-NSC $M_{{UV}}={m_uv:.2f}$; QSO1 $M_{{UV}}={QSO1_MUV_AB:.2f}$"
        x_anchor = min(max(float(best["log10_nsc_mass"]), FIG04_XLIM_LOGM[0] + 0.1), FIG04_XLIM_LOGM[1] - 0.1)
        ax.annotate(
            uv_text,
            xy=(x_anchor, nsc_y),
            xytext=(FIG04_XLIM_LOGM[0] + 0.10, nsc_y + 0.70),
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="0.20",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.35"},
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(*FIG04_XLIM_LOGM)
    ax.set_ylim(-0.65, y_positions[0] + 0.65)
    ax.set_xlabel(r"$\log_{10}(M/M_{\odot})$")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    ax.legend(frameon=False, loc="lower right", fontsize=7.0, ncol=2)
    return fig


def plot_fig05_uvmag(aperture_table):
    required = ["aperture_pc", "M_UV"]
    missing = [name for name in required if name not in aperture_table.columns]
    if missing:
        raise ValueError(f"Fig. 05 aperture table is missing required columns: {missing}")
    aperture_pc = aperture_table["aperture_pc"].to_numpy(dtype=float)
    m_uv = aperture_table["M_UV"].to_numpy(dtype=float)
    if len(aperture_pc) != 7 or np.any(~np.isfinite(aperture_pc)) or np.any(~np.isfinite(m_uv)):
        raise ValueError("Fig. 05 aperture table must contain seven finite aperture and UV-magnitude values.")

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.7))
    ax.plot(aperture_pc, m_uv, marker="o", ms=5.5, color="#1f77b4", label="Selected best halo")
    ax.axhline(QSO1_MUV_AB, c="black", ls=":", label=rf"QSO1 $M_{{\rm UV}}={QSO1_MUV_AB:.2f}$")
    ax.set_xlabel(r"UV aperture $R_{\rm UV}$ [pc]")
    ax.set_ylabel(r"Rest-frame $1500\,\AA$ absolute AB magnitude $M_{\rm UV}$")
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(np.arange(1.0, 8.0, 1.0))
    y_min = min(float(np.min(m_uv)), float(QSO1_MUV_AB))
    y_max = max(float(np.max(m_uv)), float(QSO1_MUV_AB))
    y_span = max(y_max - y_min, 0.5)
    y_margin = 0.08 * y_span
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def _bhmf_density(masses):
    masses = np.asarray(masses, dtype=float)
    counts, _ = np.histogram(masses[np.isfinite(masses) & (masses > 0.0)], bins=FIG09_BIN_EDGES)
    return counts.astype(float) / FIG09_BHMF_VOLUME_CMPC3


def _bhmf_density_per_dex(masses):
    masses = np.asarray(masses, dtype=float)
    counts, _ = np.histogram(masses[np.isfinite(masses) & (masses > 0.0)], bins=10.0 ** FIG06_BIN_EDGES)
    bin_width_dex = np.diff(FIG06_BIN_EDGES)
    return counts.astype(float) / (FIG09_BHMF_VOLUME_CMPC3 * bin_width_dex)


def _fig06_project_densities(summary_by_z, final_gc):
    z_out = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(z_out)):
        raise ValueError("Fig. 06 requires finite z_out values in haloSummaryByZ.")
    z_values = np.sort(np.unique(z_out))
    nuclear_by_z = {}
    inventory = {}
    for z_out_value in z_values:
        rows = summary_by_z.loc[z_out == float(z_out_value)]
        masses = pd.to_numeric(rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
        masses = masses[np.isfinite(masses) & (masses > 0.0)]
        nuclear_by_z[float(z_out_value)] = _bhmf_density_per_dex(masses)
        inventory[float(z_out_value)] = int(len(masses))

    if not np.any(z_out == 0.0):
        raise ValueError("Fig. 06 requires an exact z_out=0 row; nearest-redshift fallback is disabled.")
    nuclear_z0 = nuclear_by_z[0.0]

    status = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=int)
    imbh_mass = pd.to_numeric(final_gc["M_IMBH_final"], errors="coerce").to_numpy(dtype=float)
    bad = np.isin(status, np.asarray([STATUS_EXHAUSTED, STATUS_TORN], dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)
    if np.any(bad):
        raise ValueError(f"Fig. 06 found positive IMBH masses in exhausted/torn statuses: {sorted(set(status[bad].tolist()))}")
    satellite_mass = imbh_mass[np.isin(status, np.asarray(SATELLITE_BH_STATUSES, dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)]
    satellite = _bhmf_density_per_dex(satellite_mass)
    total = nuclear_z0 + satellite
    if not np.allclose(total, nuclear_z0 + satellite, rtol=0.0, atol=1.0e-15):
        raise ValueError("Fig. 06 total BHMF failed the total=nuclear+satellite identity.")
    return 0.5 * (FIG06_BIN_EDGES[:-1] + FIG06_BIN_EDGES[1:]), nuclear_by_z, satellite, total, inventory, int(len(satellite_mass))


def _plot_values(values):
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out) | (out <= 0.0)] = np.nan
    return out


def _log10_plot_values(values):
    out = _plot_values(values)
    valid = np.isfinite(out)
    out[valid] = np.log10(out[valid])
    return out


def plot_fig03_bhmf(summary_by_z, final_gc, reference):
    bins = np.asarray(FIG09_BIN_EDGES, dtype=float)
    x = bins[:-1]
    z_values = np.sort(pd.to_numeric(summary_by_z["z_out"], errors="coerce").dropna().unique())
    norm = mpl.colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5) if len(z_values) == 1 else mpl.colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    #cmap = mpl.cm.viridis
    cmap = mpl.cm.jet

    # Nuclear is redshift-resolved in haloSummaryByZ; satellites are only a final inventory.
    nuclear_by_z, inventory = {}, {}
    for z_out in z_values:
        rows = summary_by_z[np.isclose(summary_by_z["z_out"].to_numpy(dtype=float), float(z_out), rtol=0.0, atol=1.0e-8)]
        masses = pd.to_numeric(rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
        masses = masses[np.isfinite(masses) & (masses > 0.0)]
        inventory[float(z_out)] = int(len(masses))
        if len(masses) > 0:
            nuclear_by_z[float(z_out)] = _bhmf_density(masses)
    if not nuclear_by_z:
        raise ValueError("No positive nuclear BH masses are available for Fig. 03.")

    status = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=int)
    imbh_mass = pd.to_numeric(final_gc["M_IMBH_final"], errors="coerce").to_numpy(dtype=float)
    bad = np.isin(status, np.asarray([STATUS_EXHAUSTED, STATUS_TORN], dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)
    if np.any(bad):
        raise ValueError(f"Fig. 03 found positive IMBH masses in exhausted/torn statuses: {sorted(set(status[bad].tolist()))}")
    satellite_mass = imbh_mass[np.isin(status, np.asarray(SATELLITE_BH_STATUSES, dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)]
    satellite_density = _bhmf_density(satellite_mass)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    first_nuclear = True
    for z_out in z_values:
        if float(z_out) not in nuclear_by_z:
            continue
        ax.plot(x, _plot_values(nuclear_by_z[float(z_out)]), c=cmap(norm(float(z_out))), lw=1.8, alpha=0.95, label="Nuclear" if first_nuclear else None)
        first_nuclear = False
    ax.plot(x, _plot_values(satellite_density), color="0.35", lw=2.0, ls="dashdot", label="Satellite")
    positive_values = [arr[arr > 0.0] for arr in nuclear_by_z.values()]
    positive_values.append(satellite_density[satellite_density > 0.0])
    if reference is not None:
        ref_x = reference["mbh_msun"].to_numpy(dtype=float)
        ax.plot(ref_x, reference["linear_mpc3"].to_numpy(dtype=float), color="red", lw=1.0, label="Linear")
        ax.fill_between(ref_x, reference["linear_low_mpc3"].to_numpy(dtype=float), reference["linear_high_mpc3"].to_numpy(dtype=float), color="red", alpha=0.20, linewidth=0.0)
        ax.plot(ref_x, reference["nsc_mpc3"].to_numpy(dtype=float), color="blue", lw=1.0, label="NSC")
        ax.fill_between(ref_x, reference["nsc_low_mpc3"].to_numpy(dtype=float), reference["nsc_high_mpc3"].to_numpy(dtype=float), color="blue", alpha=0.20, linewidth=0.0)
        positive_values.append(reference[["linear_mpc3", "linear_low_mpc3", "linear_high_mpc3", "nsc_mpc3", "nsc_low_mpc3", "nsc_high_mpc3"]].to_numpy(dtype=float).ravel())
    positive = np.concatenate([arr[np.isfinite(arr) & (arr > 0.0)] for arr in positive_values if len(arr) > 0])
    y_min = 1.0e-5
    y_max = 1.0e-1
    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e2, 1.0e11)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$M_{\rm BH}/M_{\odot}$")
    ax.set_ylabel(r"$n_{\rm BH}/{\rm cMpc}^{-3}$")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig, {"nuclear_by_z": inventory, "satellite": int(len(satellite_mass))}


def plot_fig06_bhmf2(bhmf_data, summary_by_z, final_gc):
    required = ["Phi [lgM☉⁻¹Mpc⁻³]", "sigma_Phi [lgM☉⁻¹Mpc⁻³]", "Mbh [M☉]", "sigma_Mbh", "colour", "face colour", "shape", "label"]
    missing = [name for name in required if name not in bhmf_data.columns]
    if missing:
        raise ValueError(f"Fig. 06 BHMF table is missing required columns: {missing}")
    x_project, nuclear_by_z, satellite, total, inventory, satellite_count = _fig06_project_densities(summary_by_z, final_gc)
    marker_rows = bhmf_data.loc[bhmf_data["shape"].ne("line")].copy()
    line_rows = bhmf_data.loc[bhmf_data["shape"].eq("line")].copy()

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 5.0))
    plotted_labels = set()
    for _, row in marker_rows.iterrows():
        label = str(row["label"])
        marker_label = label if label not in plotted_labels else None
        plotted_labels.add(label)
        face_colour = str(row["face colour"])
        ax.errorbar(np.log10(float(row["Mbh [M☉]"])), float(row["Phi [lgM☉⁻¹Mpc⁻³]"]), xerr=float(row["sigma_Mbh"]), yerr=float(row["sigma_Phi [lgM☉⁻¹Mpc⁻³]"]), fmt=str(row["shape"]), ms=6.0, color=str(row["colour"]), ecolor=str(row["colour"]), markerfacecolor=face_colour, markeredgecolor=str(row["colour"]), markeredgewidth=0.8, elinewidth=0.8, capsize=0.0, label=marker_label, zorder=7)

    base_line_labels = sorted(label for label in line_rows["label"].unique() if not label.endswith(" (lower envelope)") and not label.endswith(" (upper envelope)"))
    for base_label in base_line_labels:
        central = line_rows.loc[line_rows["label"].eq(base_label)].sort_values("Mbh [M☉]")
        lower = line_rows.loc[line_rows["label"].eq(base_label + " (lower envelope)")].sort_values("Mbh [M☉]")
        upper = line_rows.loc[line_rows["label"].eq(base_label + " (upper envelope)")].sort_values("Mbh [M☉]")
        if len(central) == 0 or len(lower) != len(central) or len(upper) != len(central):
            raise ValueError(f"Fig. 06 line group {base_label!r} does not contain matching central and envelope rows.")
        x_line = np.log10(central["Mbh [M☉]"].to_numpy(dtype=float))
        lower_x = np.log10(lower["Mbh [M☉]"].to_numpy(dtype=float))
        upper_x = np.log10(upper["Mbh [M☉]"].to_numpy(dtype=float))
        if not np.allclose(x_line, lower_x, rtol=0.0, atol=1.0e-12) or not np.allclose(x_line, upper_x, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Fig. 06 line group {base_label!r} has mismatched mass grids.")
        colour = str(central["colour"].iloc[0])
        face_colour = str(lower["face colour"].iloc[0])
        ax.fill_between(x_line, lower["Phi [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float), upper["Phi [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float), color=face_colour, alpha=0.35, linewidth=0.0, zorder=1)
        ax.plot(x_line, central["Phi [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float), c=colour, lw=1.8, label=base_label, zorder=3)

    z_values = np.asarray(sorted(nuclear_by_z), dtype=float)
    #cmap = mpl.cm.viridis
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5) if len(z_values) == 1 else mpl.colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    nuclear_label_used = False
    for z_out, density in sorted(nuclear_by_z.items()):
        if not np.any(density > 0.0):
            continue
        ax.plot(x_project, _log10_plot_values(density), c=cmap(norm(float(z_out))), lw=1.5, alpha=0.9, label="Nuclear" if not nuclear_label_used else None, zorder=2)
        nuclear_label_used = True
    ax.plot(x_project, _log10_plot_values(satellite), color="0.35", lw=1.8, ls="dashdot", label="Satellite", zorder=2)
    ax.plot(x_project, _log10_plot_values(total), color="black", lw=2.0, ls="-", label="Total (z=0)", zorder=4)

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])
    ax.set_xlim(4.2, 8.9)
    ax.set_ylim(-6.2, 0.2)
    ax.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_{\odot})$")
    ax.set_ylabel(r"$\log \Phi\ [M_{\odot}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="lower left", fontsize=7.0, ncol=2)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig, {"nuclear_by_z": inventory, "satellite": satellite_count, "total_identity": bool(np.allclose(total, nuclear_by_z[0.0] + satellite, rtol=0.0, atol=1.0e-15))}


# MAIN FUNCTION
def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot the Kong & Li 2026 figures, including the UV-aperture diagnostic, from one High-z SMBH Seeds output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument("--ns-value", type=float, default=NS_VALUE_DEFAULT, help="Single N_s value used for this plot set.")
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.5, help="Log10 stellar-mass bin width for Fig. 01.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Default: <out_dir>/_plots_Kong&Li2026.")
    parser.add_argument("--uv-table", type=Path, default=UV_CALIBRATION_PATH, help="FSPS-MIST/Chabrier pure-stellar 1500 Angstrom UV table per initially formed stellar mass; feh is log10(Z/Zsun).")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else out_dir / "_plots_Kong&Li2026"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_by_z = load_halo_summary_by_z(out_dir, args.ns_value)
    final_gc = load_final_gc(out_dir, args.ns_value)
    metadata = load_run_metadata(out_dir)
    final_redshift = float(metadata.get("final_redshift", 0.0))
    if not np.isfinite(final_redshift) or final_redshift < 0.0:
        raise ValueError(f"run_metadata final_redshift must be finite and non-negative, got {final_redshift!r}.")

    observations = load_mbh_mstar_observations()
    fig01 = plot_fig01_mbh_mstar(summary_by_z, observations, float(args.mass_bin_width_dex))
    _save_figure(fig01, plot_dir / "Fig.01_Mbh-Mstar.pdf")

    points, curves = load_juodzbalis2026_fig2()
    fig3_reference = load_juodzbalis2026_fig3_bh_masses()
    uv_calibration = load_uv_calibration(args.uv_table)
    all_z_rows = _select_fig02_z_rows(summary_by_z)
    all_deposit_profile = load_deposit_profile_for_redshift_summary(_deposit_path(out_dir, args.ns_value), all_z_rows, final_redshift)
    profile_max_radius_pc = np.asarray([float(np.asarray(rout, dtype=float)[-1]) * 1.0e3 for rout in all_deposit_profile["r_outer_kpc"]], dtype=float)
    keep_profile = np.isfinite(profile_max_radius_pc) & (profile_max_radius_pc >= FIG08_MATCH_RADIUS_RANGE_PC[1])
    all_halo_ids = np.asarray(all_deposit_profile["halo_ids"], dtype=int)
    eligible_halo_ids = all_halo_ids[keep_profile]
    excluded_halo_ids = all_halo_ids[~keep_profile]
    eligible_z_rows = all_z_rows.loc[all_z_rows["halo_id_z0"].isin(eligible_halo_ids)].reset_index(drop=True)
    eligible_deposit_profile = {
        "halo_ids": eligible_halo_ids,
        "r_outer_kpc": [value for value, keep in zip(all_deposit_profile["r_outer_kpc"], keep_profile) if keep],
        "cumulative_mass_msun": [value for value, keep in zip(all_deposit_profile["cumulative_mass_msun"], keep_profile) if keep],
        "cumulative_formed_mass_msun": [value for value, keep in zip(all_deposit_profile["cumulative_formed_mass_msun"], keep_profile) if keep],
    }
    print(
        f"Fig. 02/Fig. 04 radial-coverage filter: excluded {len(excluded_halo_ids)} halo(s) "
        f"with profile coverage <{FIG08_MATCH_RADIUS_RANGE_PC[1]:.0f} pc; "
        f"IDs={excluded_halo_ids.tolist()}; eligible profiles={len(eligible_z_rows)}."
    )
    score_table, fig02_best = score_fig02_candidate_haloes(out_dir, args.ns_value, points, eligible_z_rows, eligible_deposit_profile, final_gc, uv_calibration)
    if len(excluded_halo_ids) > 0:
        excluded_summary = all_z_rows.set_index("halo_id_z0").loc[excluded_halo_ids]
        excluded_table = pd.DataFrame(np.nan, index=np.arange(len(excluded_halo_ids)), columns=score_table.columns)
        excluded_table["index"] = -1
        excluded_table["halo_id_z0"] = excluded_halo_ids
        excluded_table["redshift"] = excluded_summary["redshift"].to_numpy(dtype=float)
        excluded_table["nsc_mass_msun"] = excluded_summary["nsc_mass_msun"].to_numpy(dtype=float)
        excluded_nsc = excluded_table["nsc_mass_msun"].to_numpy(dtype=float)
        excluded_table["log10_nsc_mass"] = np.nan
        valid_excluded_nsc = np.isfinite(excluded_nsc) & (excluded_nsc > 0.0)
        excluded_table.loc[valid_excluded_nsc, "log10_nsc_mass"] = np.log10(excluded_nsc[valid_excluded_nsc])
        excluded_bh = excluded_summary["central_bh_mass_final_msun"].to_numpy(dtype=float)
        excluded_table["central_bh_mass_msun"] = excluded_bh
        excluded_table["log10_central_bh_mass"] = np.nan
        valid_excluded_bh = np.isfinite(excluded_bh) & (excluded_bh > 0.0)
        excluded_table.loc[valid_excluded_bh, "log10_central_bh_mass"] = np.log10(excluded_bh[valid_excluded_bh])
        excluded_table["missing_reason"] = [
            f"insufficient deposit radial coverage: {radius:.6g} pc < {FIG08_MATCH_RADIUS_RANGE_PC[1]:.6g} pc"
            for radius in profile_max_radius_pc[~keep_profile]
        ]
        score_table = pd.concat([score_table, excluded_table], ignore_index=True).sort_values("halo_id_z0").reset_index(drop=True)
    score_path = plot_dir / "Fig.02_Fig04_candidate_scores.csv"
    score_table.to_csv(score_path, index=False)
    print(f"Saved {score_path}")
    fig02 = plot_fig02_rotation_curve(points, curves, eligible_z_rows, eligible_deposit_profile, fig02_best)
    z_vals = eligible_z_rows["z_out"].to_numpy(dtype=float)
    print(f"Fig. 02 z selection: N={len(eligible_z_rows)}, z range={float(np.min(z_vals)):.3f}-{float(np.max(z_vals)):.3f}.")
    print(f"Fig. 02/Fig. 04 candidate pool: out_dir={out_dir}, ns-value={float(args.ns_value):.3g}, eligible profiles={len(eligible_z_rows)}, CSV rows={len(score_table)}, finite Keplerian+UV scores={int(np.isfinite(score_table['score_keplerian_uv'].to_numpy(dtype=float)).sum())}.")
    print(f"UV mode: {uv_calibration['uv_mode']}.")
    print(f"UV table: {uv_calibration['path']} (age={uv_calibration['age_min_gyr']:.6g}-{uv_calibration['age_max_gyr']:.6g} Gyr, [Fe/H]={uv_calibration['feh_min']:.2f}-{uv_calibration['feh_max']:.2f}).")
    print(
        "Selected Keplerian+UV halo: "
        f"halo_id_z0={int(fig02_best['halo_id_z0'])}, z={float(fig02_best['redshift']):.3f}, "
        f"Keplerian term={float(fig02_best['keplerian_term']):.4f}, "
        f"weighted velocity chi2={float(fig02_best['keplerian_chi2_weighted']):.4f}, "
        f"velocity points={int(fig02_best['keplerian_n_points'])}, "
        f"M_UV={float(fig02_best['M_UV']):.2f}, target M_UV={QSO1_MUV_AB:.2f}, "
        f"UV term={float(fig02_best['uv_term']):.4f}, "
        f"score_keplerian_uv={float(fig02_best['score_keplerian_uv']):.4f}, "
        f"formed mass <{QSO1_NSC_APERTURE_PC:.1f} pc={float(fig02_best['formed_mass_6pc_msun']):.6g} Msun, "
        f"weighted age={float(fig02_best['weighted_age_gyr']):.3f} Gyr, "
        f"weighted [Fe/H]={float(fig02_best['weighted_feh']):.3f}, "
        f"nearest-grid counts(age, [Fe/H], any)=({int(fig02_best['n_uv_age_nearest_grid'])}, {int(fig02_best['n_uv_feh_nearest_grid'])}, {int(fig02_best['n_uv_any_nearest_grid'])}), "
        f"log10(M_NSC/Msun)={float(fig02_best['log10_nsc_mass']):.3f}, "
        f"log10(M_SMBH_final/Msun)={float(fig02_best['log10_central_bh_mass']):.3f}."
    )
    _save_figure(fig02, plot_dir / "Fig.02_RotationCurve.pdf")

    fig04 = plot_fig04_bh_masses(fig3_reference, fig02_best, fig02_best)
    _save_figure(fig04, plot_dir / FIGURE_04_FILENAME)

    fig05_best = fig02_best
    if int(fig05_best["halo_id_z0"]) in set(excluded_halo_ids.tolist()):
        raise ValueError(f"Fig. 05 fig02_best halo_id_z0={int(fig05_best['halo_id_z0'])} is excluded for insufficient radial coverage.")
    aperture_table = estimate_uv_magnitude_apertures(all_deposit_profile, final_gc, int(fig05_best["halo_id_z0"]), uv_calibration)
    fig05 = plot_fig05_uvmag(aperture_table)
    _save_figure(fig05, plot_dir / FIGURE_05_FILENAME)

    reference = load_kritos2025_fig9()
    fig03, inventory = plot_fig03_bhmf(summary_by_z, final_gc, reference)
    counts = ", ".join(f"z={z:.6g}: {n}" for z, n in sorted(inventory["nuclear_by_z"].items()))
    print(f"Fig. 03 nuclear positive-BH inventory: {counts}.")
    print(f"Fig. 03 satellite positive-BH inventory: {inventory['satellite']} (status 1 and -4).")
    _save_figure(fig03, plot_dir / "Fig.03_BHMF.pdf")

    bhmf_data = load_bhmf_data()
    fig06, inventory06 = plot_fig06_bhmf2(bhmf_data, summary_by_z, final_gc)
    counts06 = ", ".join(f"z={z:.6g}: {n}" for z, n in sorted(inventory06["nuclear_by_z"].items()))
    print(f"Fig. 06 nuclear positive-BH inventory: {counts06}.")
    print(f"Fig. 06 satellite positive-BH inventory: {inventory06['satellite']} (status 1 and -4).")
    print(f"Fig. 06 exact total identity at z=0: {inventory06['total_identity']}.")
    _save_figure(fig06, plot_dir / "Fig.06_BHMF2.pdf")


if __name__ == "__main__":
    main()
