#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Self-contained Kong & Li 2026 plots for MBH-Mstar, Mbh-Mhalo abundance matching, UV aperture, QSO1 rotation, and BHMF."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
from evo import read_haloevo_mpb  # noqa: E402
from run import (  # noqa: E402
    VALID_EVOLUTION_STATUS,
    _interpolate_mpb_logmh_at_redshift,
    _mpb_branch_id,
    _read_full_tree_numeric,
)


# input params
DATA_ROOT = PROJECT_ROOT / "data"
MBH_MSTAR_DATA_PATH = DATA_ROOT / "Mbh-Mstar.csv"
BHMF_DATA_PATH = DATA_ROOT / "BHMF.csv"
FIGURE_10_FILENAME = "Fig.10_BHSMF.pdf"
CHEN2026_FIG05A_DATA_PATH = DATA_ROOT / "Chen+2026" / "chen2026_fig05a_seed_mass_functions.csv"
CHEN2026_FIG05A_REDSHIFT = 20.0
FIGURE_12_FILENAME = "Fig.12_BHseed_hist.pdf"
CHEN2026_FIG06_DATA_PATH = DATA_ROOT / "Chen+2026" / "chen2026_fig06_seeding_history.csv"
FIG12_CHEN_CURVE_ROLES = ("all_seeds", "popiii_subedd", "popiii_edd", "popii")
FIG12_CHEN_CURVE_LABELS = {
    "all_seeds": "All seeds",
    "popiii_subedd": "Pop-III (sub-Eddington)",
    "popiii_edd": "Pop-III (Eddington)",
    "popii": "Pop-II",
}
FIG12_DISPLAY_REDSHIFT_RANGE = (0.0, 50.0)
FIG12_XLIM_LOG1PZ = tuple(np.log10(1.0 + np.asarray(FIG12_DISPLAY_REDSHIFT_RANGE, dtype=float)))
FIG12_RATE_LOG1PZ_BIN_WIDTH = 0.02
FIG12_CHEN_REDSHIFT_ATOL = 1.0e-6
FIG12_MODEL_COLOUR = "#6a3d9a"
FIG12_MODEL_LINEWIDTH = 2.2
FIG12_REFERENCE_LINEWIDTH = 1.35
FIG12_FIGSIZE = (7.0, 7.0)
FIG10_SEED_LOGM_BIN_EDGES = np.round(np.arange(0.0, 5.0 + 0.05, 0.1), decimals=10)
if len(FIG10_SEED_LOGM_BIN_EDGES) != 51 or not np.isclose(FIG10_SEED_LOGM_BIN_EDGES[-1], 5.0, rtol=0.0, atol=1.0e-12):
    raise RuntimeError("Fig. 10 seed-mass grid must contain the explicit 0.0--5.0 dex sequence in 0.1 dex steps.")
FIG10_SEED_LOGM_BIN_WIDTH_DEX = 0.1
FIG10_CENTRAL_LOGM_BIN_EDGES = np.round(np.arange(0.0, 8.0 + 0.05, 0.1), decimals=10)
if (
    len(FIG10_CENTRAL_LOGM_BIN_EDGES) != 81
    or not np.isclose(FIG10_CENTRAL_LOGM_BIN_EDGES[0], 0.0, rtol=0.0, atol=1.0e-12)
    or not np.isclose(FIG10_CENTRAL_LOGM_BIN_EDGES[-1], 8.0, rtol=0.0, atol=1.0e-12)
    or np.any(~np.isfinite(FIG10_CENTRAL_LOGM_BIN_EDGES))
    or np.any(np.diff(FIG10_CENTRAL_LOGM_BIN_EDGES) <= 0.0)
    or not np.allclose(np.diff(FIG10_CENTRAL_LOGM_BIN_EDGES), 0.1, rtol=0.0, atol=1.0e-12)
):
    raise RuntimeError("Fig. 10 central-BH grid must contain the explicit 0.0--8.0 dex sequence in 0.1 dex steps.")
FIG10_CENTRAL_LOGM_BIN_WIDTH_DEX = 0.1
FIG10_CHEN_CURVE_ROLES = (
    "all_seeds_central",
    "all_seeds_lower_envelope",
    "all_seeds_upper_envelope",
    "popiii_subedd",
    "popiii_edd",
    "fast_halo",
    "lw_halo",
    "popii",
)
FIG10_CHEN_CURVE_LABELS = {
    "all_seeds_central": "All seeds",
    "all_seeds_lower_envelope": "All seeds (lower envelope)",
    "all_seeds_upper_envelope": "All seeds (upper envelope)",
    "popiii_subedd": "Pop-III (sub-Eddington)",
    "popiii_edd": "Pop-III (Eddington)",
    "fast_halo": "Fast halo (gamma_v >= 3)",
    "lw_halo": "LW halo (J_LW,21 >= 7.5)",
    "popii": "Pop-II",
}
FIG10_CHEN_VISIBLE_CURVE_ROLES = (
    "popiii_subedd",
    "popiii_edd",
    "popii",
)
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
FIG08_MATCH_RADIUS_RANGE_PC = (10.0, 160.0)
FIG08_SCATTER_PERCENTILES = (16.0, 84.0)
FIG08_VELOCITY_SIN_I = 1.0
FIG08_RADIUS_MAX_PC = 160.0
FIG08_HALO_MASS_WINDOW_DEX = 0.1

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
# Total weights for the five absolute-radius velocity-point groups, inner to outer.
QSO1_VELOCITY_GROUP_WEIGHTS = (0.30, 0.20, 0.20, 0.15, 0.15)
FIG04_XLIM_LOGM = (5.3, 8.4)
UV_CALIBRATION_PATH = DATA_ROOT / "UV" / "fsps_mist_chabrier_m1500_grid.csv"
UV_MODE_LABEL = "FSPS-MIST/Chabrier pure-stellar M1500(age,[Fe/H]) table"
UV_MIN_TABLE_AGE_GYR = 1.0e-4
UV_MIN_TABLE_FEH = -2.50
UV_MAX_TABLE_FEH = 0.50
UV_CALIBRATION_COLUMNS = ("age_gyr", "log10_age_gyr", "feh", "z_ratio", "M1500_AB_per_Msun")

TNG_H = 0.6774
TNG50_NATIVE_SIDE_CMPC_H = 35.0
TNG100_NATIVE_SIDE_CMPC_H = 75.0
TNG50_FULL_BOX_SIDE_CMPC = TNG50_NATIVE_SIDE_CMPC_H / TNG_H
TNG100_FULL_BOX_SIDE_CMPC = TNG100_NATIVE_SIDE_CMPC_H / TNG_H
FIG09_BHMF_SIDE_CMPC = TNG50_FULL_BOX_SIDE_CMPC
FIG09_BHMF_VOLUME_CMPC3 = FIG09_BHMF_SIDE_CMPC**3
FIG09_BIN_EDGES = np.logspace(2.0, 9.0, 32)
FIG06_BIN_EDGES = np.arange(4.0, 9.1, 0.25)
FIG06_MIN_MODEL_REDSHIFT_EXCLUSIVE = 3.0
# Fixed common halo-mass binning selected from the available project outputs;
# this preserves comparable resolution and count semantics between output runs.
FIGURE_09_DISTR_FILENAME = "Fig.09_distr.pdf"
FIG09_DISTR_BIN_WIDTH_DEX = 0.5
FIGURE_11_FILENAME = "Fig.11_assembly.pdf"
FIG11_TARGET_REDSHIFT = 7.0
FIG11_REDSHIFT_ROW_ATOL = 1.0e-10
FIG11_COMPARISON_COUNT = 8
FIG11_MAX_PANELS = 9
FIG11_SATELLITE_CMAP = "jet"
FIG11_SCORE_RTOL = 1.0e-10
FIG11_SCORE_ATOL = 1.0e-10

# The abundance-matching figures use the high-redshift snapshots most relevant
# to the seed problem.  The physical TNG50 full box is the reference volume for
# converting the mixed-suite counts to cumulative number densities.
ABUNDANCE_MATCHING_REDSHIFTS = (5.0, 7.0, 9.0)
ABUNDANCE_MATCHING_REDSHIFT_ATOL = 0.11
ABUNDANCE_MATCHING_BIN_WIDTH_DEX = 0.5
ABUNDANCE_MATCHING_VOLUME_CMPC3 = FIG09_BHMF_VOLUME_CMPC3
FIGURE_07_FILENAME = "Fig.07_Mbh-Mhalo_AbundanceMatching.pdf"
FIGURE_08_FILENAME = "Fig.08_Mbh-Mhalo_CumulativeAbundance.pdf"

TNG_CATALOGUE_ROOT = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark")
TNG_TARGET_MANIFEST_FILENAME = "target_manifest_dark.csv"
TNG_TARGET_METADATA_FILENAME = "targets_z0_dark.json"
TNG_TREE_LOOKUP_FILENAME = "halo_tree_lookup.csv"
TNG_FIXED_TREE_DIRNAME = "fixed_trees_large_spin_dark"
TNG_ORIGINAL_LOOKUP_FILENAME = "id_lookup_original.csv"
TNG_SHIFTED_LOOKUP_FILENAME = "id_lookup_large_dark.csv"
TNG_SUITE_KEYS = ("tng50_1_dark", "tng100_1_dark")
TNG100_HALO_ID_OFFSET = 1_000_000

BH_TO_STELLAR_MASS_RATIOS = (0.01, 0.1, 1.0)
REINES_VOLONTERI_2015_NORM = 7.45
REINES_VOLONTERI_2015_SLOPE = 1.05
REINES_VOLONTERI_2015_SCATTER_DEX = 0.55

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

def _final_gcs_path(out_dir):
    path = Path(out_dir).resolve() / "finalGCs.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing final-GC catalogue: {path}")
    return path


def _halo_summary_by_z_path(out_dir):
    path = Path(out_dir).resolve() / "haloSummaryByZ.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing redshift-resolved halo summary: {path}")
    return path


def _deposit_path(out_dir):
    path = Path(out_dir).resolve() / "depos.dat"
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


def load_halo_summary_by_z(out_dir):
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
    table = _rename_existing_columns(pd.read_csv(_halo_summary_by_z_path(out_dir)), mapping)
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


def _build_fig09_halo_distribution(summary_by_z, best):
    required = ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing Fig. 09 distribution columns: {missing}")

    table = summary_by_z.loc[:, required].copy()
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id_raw = table["halo_id_z0"].to_numpy(dtype=float)
    redshift = table["redshift"].to_numpy(dtype=float)
    available_raw = table["halo_mass_available"].to_numpy(dtype=float)
    log10_halo_mass = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    if np.any(~np.isfinite(halo_id_raw)) or np.any(np.abs(halo_id_raw - np.rint(halo_id_raw)) > 1.0e-8):
        raise ValueError("Fig. 09 haloSummaryByZ contains non-finite or non-integer halo IDs.")
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 09 haloSummaryByZ contains non-finite or negative redshifts.")
    if np.any(~np.isfinite(available_raw)) or np.any(~np.isin(available_raw, np.asarray([0.0, 1.0]))):
        raise ValueError("Fig. 09 haloSummaryByZ halo_mass_available must contain only finite 0/1 values.")

    halo_id = np.rint(halo_id_raw).astype(np.int64)
    available = available_raw == 1.0
    key_table = pd.DataFrame({"halo_id_z0": halo_id, "redshift": redshift})
    if key_table.duplicated().any():
        duplicate_keys = key_table.loc[key_table.duplicated(keep=False)].drop_duplicates().to_dict("records")
        raise ValueError(f"Fig. 09 requires one haloSummaryByZ row per (halo_id_z0, redshift); duplicates={duplicate_keys[:10]}.")

    invalid_available_mass = available & ~np.isfinite(log10_halo_mass)
    if np.any(invalid_available_mass):
        bad_keys = key_table.loc[invalid_available_mass].to_dict("records")
        raise ValueError(f"Fig. 09 has unavailable numeric halo masses for rows marked available: {bad_keys[:10]}.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        halo_mass_msun = np.power(10.0, log10_halo_mass)
    valid_mass = available & np.isfinite(log10_halo_mass) & np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
    invalid_mass = available & ~valid_mass
    if np.any(invalid_mass):
        bad_keys = key_table.loc[invalid_mass].to_dict("records")
        raise ValueError(f"Fig. 09 has non-positive or non-finite available halo masses: {bad_keys[:10]}.")
    if not np.any(valid_mass):
        raise ValueError("Fig. 09 cannot plot a halo distribution because no valid MPB halo masses remain.")

    log_edges = _regular_log_bin_edges(log10_halo_mass[valid_mass], FIG09_DISTR_BIN_WIDTH_DEX)
    if len(log_edges) < 2 or np.any(~np.isfinite(log_edges)) or np.any(np.diff(log_edges) <= 0.0):
        raise ValueError("Fig. 09 generated invalid common logarithmic halo-mass bin edges.")
    all_redshifts = np.sort(np.unique(redshift))
    distributions = []
    empty_redshifts = []
    for z_value in all_redshifts:
        z_mask = redshift == float(z_value)
        valid_z = z_mask & valid_mass
        if not np.any(valid_z):
            empty_redshifts.append(float(z_value))
            continue
        samples_log = log10_halo_mass[valid_z]
        counts, _ = np.histogram(samples_log, bins=log_edges)
        if int(np.sum(counts)) != int(len(samples_log)):
            raise ValueError(f"Fig. 09 histogram dropped halo masses at z={float(z_value):.6g}.")
        distributions.append(
            {
                "redshift": float(z_value),
                "log10_halo_mass": samples_log.copy(),
                "halo_mass_msun": halo_mass_msun[valid_z].copy(),
                "counts": counts.astype(int, copy=False),
            }
        )
    if len(distributions) == 0:
        raise ValueError("Fig. 09 has no redshift snapshot with valid halo masses.")

    try:
        best_id_value = float(best["halo_id_z0"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fig. 09 best-halo selection is missing a finite halo_id_z0.") from exc
    if not np.isfinite(best_id_value) or abs(best_id_value - np.rint(best_id_value)) > 1.0e-8:
        raise ValueError(f"Fig. 09 best-halo selection has an invalid halo_id_z0={best_id_value!r}.")
    best_id = int(np.rint(best_id_value))
    best_rows = halo_id == best_id
    if not np.any(best_rows):
        raise ValueError(f"Fig. 09 best halo_id_z0={best_id} is absent from haloSummaryByZ.")
    best_track = []
    best_missing_redshifts = []
    for z_value in all_redshifts:
        z_mask = best_rows & (redshift == float(z_value))
        if not np.any(z_mask):
            best_missing_redshifts.append(float(z_value))
            continue
        index = int(np.flatnonzero(z_mask)[0])
        if valid_mass[index]:
            best_track.append(
                {
                    "redshift": float(z_value),
                    "log10_halo_mass": float(log10_halo_mass[index]),
                    "halo_mass_msun": float(halo_mass_msun[index]),
                }
            )
        else:
            best_missing_redshifts.append(float(z_value))
    if len(best_track) == 0:
        raise ValueError(f"Fig. 09 best halo_id_z0={best_id} has no valid MPB halo mass at any output redshift.")

    return {
        "redshifts": np.asarray([item["redshift"] for item in distributions], dtype=float),
        "all_redshifts": all_redshifts.astype(float),
        "empty_redshifts": np.asarray(empty_redshifts, dtype=float),
        "log_bin_edges": log_edges,
        "distributions": distributions,
        "excluded_unavailable_rows": int(np.count_nonzero(~available)),
        "best_halo_id_z0": best_id,
        "best_halo_track": best_track,
        "best_halo_missing_redshifts": np.asarray(best_missing_redshifts, dtype=float),
    }


def load_final_gc(out_dir):
    mapping = {
        "M_IMBH_final": "imbh_mass_final_msun",
        "halo_id_z0": "halo_id_z0",
        "status": "status",
    }
    table = _rename_existing_columns(_read_headered_whitespace_table(_final_gcs_path(out_dir)), mapping)
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


def _fig11_allcat_path(out_dir):
    paths = sorted(Path(out_dir).resolve().glob("allcat_*.txt"))
    if len(paths) != 1:
        raise ValueError(
            f"Fig. 11 requires exactly one allcat_*.txt catalogue in {Path(out_dir).resolve()}, "
            f"found {len(paths)}: {[path.name for path in paths[:10]]}."
        )
    path = paths[0]
    if not path.is_file():
        raise ValueError(f"Fig. 11 allcat catalogue is not a regular file: {path}")
    return path


def _fig11_read_allcat_header(path):
    header = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            fields = stripped.lstrip("#").strip().split()
            if "hid_z0" in fields and "isMPB" in fields and "subfind_form" in fields:
                header = fields
                break
    if header is None or len(header) != len(set(header)):
        raise ValueError(f"Could not identify a unique allcat header in {path}.")
    required = ("hid_z0", "logMh_form", "zform", "isMPB", "subfind_form")
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"Fig. 11 allcat catalogue {path} is missing required fields: {missing}")
    return header


def _fig11_integer_values(values, name):
    raw = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(np.abs(raw - np.rint(raw)) > 1.0e-8):
        raise ValueError(f"Fig. 11 {name} contains non-finite or non-integer values.")
    return np.rint(raw).astype(np.int64)


def _load_fig11_formation_catalogue(out_dir, final_gc):
    """Read only the allcat fields needed for branch membership and auditing."""

    path = _fig11_allcat_path(out_dir)
    header = _fig11_read_allcat_header(path)
    required = ["hid_z0", "logMh_form", "zform", "isMPB", "subfind_form"]
    dtype = {name: np.float64 for name in required}
    try:
        table = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=header,
            usecols=required,
            dtype=dtype,
            engine="c",
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not read the required Fig. 11 allcat fields from {path}") from exc
    if table.empty:
        raise ValueError(f"Fig. 11 allcat catalogue is empty: {path}")

    halo_id = _fig11_integer_values(table["hid_z0"].to_numpy(dtype=float), "allcat halo IDs")
    subfind_form = _fig11_integer_values(table["subfind_form"].to_numpy(dtype=float), "allcat subfind_form")
    is_mpb = _fig11_integer_values(table["isMPB"].to_numpy(dtype=float), "allcat isMPB flags")
    if not np.all(np.isin(is_mpb, np.asarray([0, 1], dtype=np.int64))):
        raise ValueError("Fig. 11 allcat isMPB flags must be exactly 0 or 1.")
    logmh_form = table["logMh_form"].to_numpy(dtype=float)
    zform = table["zform"].to_numpy(dtype=float)
    if np.any(~np.isfinite(logmh_form)) or np.any(~np.isfinite(zform)) or np.any(zform < 0.0):
        raise ValueError("Fig. 11 allcat formation log masses and redshifts must be finite, with zform >= 0.")

    final_required = ["halo_id_z0", "gc_index_halo", "status"]
    missing = [name for name in final_required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"finalGCs.dat is missing the Fig. 11 alignment fields: {missing}")
    final_halo_id = _fig11_integer_values(final_gc["halo_id_z0"].to_numpy(dtype=float), "finalGCs halo IDs")
    final_gc_index = _fig11_integer_values(final_gc["gc_index_halo"].to_numpy(dtype=float), "finalGCs GC indices")
    final_status = _fig11_integer_values(final_gc["status"].to_numpy(dtype=float), "finalGCs statuses")
    if np.any(final_gc_index < 1):
        raise ValueError("Fig. 11 finalGCs GC indices must be positive and one-based.")
    if len(final_halo_id) != len(halo_id):
        raise ValueError(
            f"Fig. 11 allcat/finalGCs row counts disagree: allcat={len(halo_id)}, finalGCs={len(final_halo_id)}."
        )
    if not np.array_equal(final_halo_id, halo_id):
        mismatch = np.flatnonzero(final_halo_id != halo_id)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            "Fig. 11 allcat/finalGCs parent halo IDs are not aligned "
            f"(first mismatch row={first}, allcat={int(halo_id[first]) if first >= 0 else 'n/a'}, "
            f"finalGCs={int(final_halo_id[first]) if first >= 0 else 'n/a'})."
        )

    expected_gc_index = np.empty(len(halo_id), dtype=np.int64)
    for hid in np.unique(halo_id):
        rows = np.flatnonzero(halo_id == int(hid))
        expected_gc_index[rows] = np.arange(1, len(rows) + 1, dtype=np.int64)
    if not np.array_equal(final_gc_index, expected_gc_index):
        mismatch = np.flatnonzero(final_gc_index != expected_gc_index)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            "Fig. 11 allcat/finalGCs GC indices are not aligned "
            f"(first mismatch row={first}, expected={int(expected_gc_index[first]) if first >= 0 else 'n/a'}, "
            f"finalGCs={int(final_gc_index[first]) if first >= 0 else 'n/a'})."
        )

    return {
        "path": path,
        "halo_id_z0": halo_id,
        "logMh_form": logmh_form,
        "zform": zform,
        "isMPB": is_mpb,
        "subfind_form": subfind_form,
        "status": final_status,
    }


def _fig11_fixed_tree_path(fixed_tree_basename, tree_root=None):
    root = (Path(tree_root) if tree_root is not None else TNG_CATALOGUE_ROOT / TNG_FIXED_TREE_DIRNAME).resolve()
    basename = str(fixed_tree_basename).strip()
    if not basename or Path(basename).is_absolute():
        raise ValueError(f"Fig. 11 fixed-tree basename must be a non-empty relative path: {fixed_tree_basename!r}")
    path = (root / basename).resolve()
    try:
        inside_root = os.path.commonpath([str(root), str(path)]) == str(root)
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Fig. 11 fixed tree escapes the declared tree directory: {fixed_tree_basename!r}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Fig. 11 fixed tree is missing or not a regular file: {path}")
    return path


def _fig11_read_full_tree_numeric(path):
    """Validate raw fixed-tree rows before using the shared project reader."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first = stripped.split()[0].lower()
            if first.startswith("logmh") or first == "log10_mhalo_msun":
                continue
            parts = stripped.split()
            if len(parts) < 9:
                raise ValueError(f"Fig. 11 raw fixed tree has fewer than nine columns at line {line_no}: {path}")
            try:
                float(parts[0])
                int(parts[1])
                int(parts[2])
                int(parts[3])
                int(parts[4])
                float(parts[5])
                float(parts[6])
                float(parts[7])
                float(parts[8])
            except ValueError as exc:
                raise ValueError(f"Fig. 11 raw fixed tree has a malformed numeric row at line {line_no}: {path}") from exc
    with warnings.catch_warnings():
        # The shared reader recognises the legacy logMh header but the
        # current fixed-tree files use log10_mhalo_msun for the same header.
        warnings.filterwarnings(
            "ignore",
            message="Malformed fixed-tree row .*row=log10_mhalo_msun first_progenitor_id.*",
            category=RuntimeWarning,
        )
        rows = _read_full_tree_numeric(path)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError(f"Fig. 11 raw fixed tree contains no numeric rows: {path}")
    return rows


def _fig11_tree_logmass_and_redshift(tree_rows, context):
    rows = np.asarray(tree_rows, dtype=object)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 6:
        raise ValueError(f"Fig. 11 {context} fixed-tree rows are empty or malformed: shape={rows.shape}")
    logmh = np.asarray(rows[:, 0], dtype=float)
    redshift = np.asarray(rows[:, 5], dtype=float)
    if np.any(~np.isfinite(logmh)) or np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError(f"Fig. 11 {context} fixed-tree log masses and redshifts must be finite, with z >= 0.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mass = np.power(10.0, logmh)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError(f"Fig. 11 {context} fixed-tree halo masses must be finite and positive.")
    return logmh, redshift, mass


def _fig11_main_track(path, mpb_rows=None):
    if mpb_rows is None:
        mpb_rows = read_haloevo_mpb(path)
    rows = np.asarray(mpb_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 9:
        raise ValueError(f"Fig. 11 raw MPB is empty or malformed: {path}")
    logmh = np.asarray(rows[:, 0], dtype=float)
    redshift = np.asarray(rows[:, 5], dtype=float)
    if np.any(~np.isfinite(logmh)) or np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError(f"Fig. 11 raw MPB has non-finite log mass/redshift values: {path}")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mass = np.power(10.0, logmh)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError(f"Fig. 11 raw MPB masses must be finite and positive: {path}")
    if float(np.max(redshift)) < FIG11_TARGET_REDSHIFT - FIG11_REDSHIFT_ROW_ATOL or float(np.min(redshift)) > FIG11_TARGET_REDSHIFT + FIG11_REDSHIFT_ROW_ATOL:
        raise ValueError(f"Fig. 11 raw MPB does not bracket z={FIG11_TARGET_REDSHIFT:g}: {path}")

    exact = np.flatnonzero(np.abs(redshift - FIG11_TARGET_REDSHIFT) <= FIG11_REDSHIFT_ROW_ATOL)
    if len(exact) > 1:
        raise ValueError(f"Fig. 11 raw MPB contains duplicate z={FIG11_TARGET_REDSHIFT:g} rows: {path}")
    if len(exact) == 1:
        endpoint_logmh = float(logmh[int(exact[0])])
    else:
        endpoint_logmh, available = _interpolate_mpb_logmh_at_redshift(rows[:, :9], FIG11_TARGET_REDSHIFT)
        if int(available) != 1 or not np.isfinite(endpoint_logmh):
            raise ValueError(f"Fig. 11 raw MPB cannot be interpolated to z={FIG11_TARGET_REDSHIFT:g}: {path}")

    visible = redshift >= FIG11_TARGET_REDSHIFT
    if len(exact) == 1:
        visible[int(exact[0])] = False
    if not np.any(visible) and len(exact) == 0:
        raise ValueError(f"Fig. 11 raw MPB has no raw rows above z={FIG11_TARGET_REDSHIFT:g}: {path}")
    visible_logmh = logmh[visible]
    visible_redshift = redshift[visible]
    visible_time = np.asarray([Redshift2CosmicAge(float(z), time_unit="Gyr") for z in visible_redshift], dtype=float)
    order = np.argsort(visible_time, kind="mergesort")
    visible_logmh = visible_logmh[order]
    visible_redshift = visible_redshift[order]
    visible_time = visible_time[order]
    visible_x = float(Redshift2CosmicAge(0.0, time_unit="Gyr")) - visible_time
    endpoint_time = float(Redshift2CosmicAge(FIG11_TARGET_REDSHIFT, time_unit="Gyr"))
    endpoint_x = float(Redshift2CosmicAge(0.0, time_unit="Gyr")) - endpoint_time
    track_logmh = np.concatenate([visible_logmh, np.asarray([endpoint_logmh], dtype=float)])
    track_redshift = np.concatenate([visible_redshift, np.asarray([FIG11_TARGET_REDSHIFT], dtype=float)])
    track_x = np.concatenate([visible_x, np.asarray([endpoint_x], dtype=float)])
    track_mass = np.power(10.0, track_logmh)
    if np.any(~np.isfinite(track_mass)) or np.any(track_mass <= 0.0) or np.any(track_redshift < FIG11_TARGET_REDSHIFT):
        raise ValueError(f"Fig. 11 raw MPB produced an invalid z >= 7 track: {path}")
    return {
        "x_gyr": track_x,
        "redshift": track_redshift,
        "log10_halo_mass": track_logmh,
        "halo_mass_msun": track_mass,
        "endpoint_log10_halo_mass": float(endpoint_logmh),
    }


def _fig11_map_formation_rows_to_branches(logmh_form, zform, subfind_form, tree_rows):
    logmh_form = np.asarray(logmh_form, dtype=float)
    zform = np.asarray(zform, dtype=float)
    subfind_form = _fig11_integer_values(subfind_form, "formation subfind IDs")
    if len(logmh_form) != len(zform) or len(logmh_form) != len(subfind_form):
        raise ValueError("Fig. 11 formation fields have inconsistent lengths.")
    if np.any(~np.isfinite(logmh_form)) or np.any(~np.isfinite(zform)) or np.any(zform < 0.0):
        raise ValueError("Fig. 11 formation fields must be finite, with zform >= 0.")
    rows = np.asarray(tree_rows, dtype=object)
    candidates_by_subfind = {}
    for row in rows:
        branch = int(row[3])
        subfind = int(row[2])
        if branch < 0:
            raise ValueError(f"Fig. 11 fixed-tree branch IDs must be non-negative; got {branch}")
        candidates_by_subfind.setdefault(subfind, []).append((branch, float(row[5]), float(row[0])))
    branch_ids = np.empty(len(logmh_form), dtype=np.int64)
    for index, (logmh, z_value, subfind) in enumerate(zip(logmh_form, zform, subfind_form)):
        candidates = candidates_by_subfind.get(int(subfind), ())
        if not candidates:
            raise ValueError(
                f"Fig. 11 cannot map formation row {index}: subfind_form={int(subfind)} is absent from the raw tree."
            )
        scored = [
            (abs(float(z_tree) - float(z_value)) + abs(float(logmh_tree) - float(logmh)), int(branch), float(z_tree), float(logmh_tree))
            for branch, z_tree, logmh_tree in candidates
        ]
        scored.sort(key=lambda item: item)
        best_score, best_branch, _, _ = scored[0]
        if best_score > 1.0e-3:
            raise ValueError(
                f"Fig. 11 cannot robustly map formation row {index} to a raw-tree branch; "
                f"nearest score={best_score:.6g}, branch={best_branch}."
            )
        branch_ids[index] = int(best_branch)
    return branch_ids


def _fig11_satellite_tracks(tree_rows, mpb_branch, branch_ids, formation):
    branch_ids = np.asarray(branch_ids, dtype=np.int64)
    statuses = np.asarray(formation["status"], dtype=np.int64)
    zform = np.asarray(formation["zform"], dtype=float)
    valid_status = np.isin(statuses, np.asarray(sorted(VALID_EVOLUTION_STATUS), dtype=np.int64))
    high_z_gc = valid_status & np.isfinite(zform) & (zform >= FIG11_TARGET_REDSHIFT)
    tree_branch = np.asarray(tree_rows[:, 3], dtype=np.int64)
    tree_logmh, tree_redshift, tree_mass = _fig11_tree_logmass_and_redshift(tree_rows, "satellite")
    tracks = []
    for branch in sorted(set(int(value) for value in tree_branch if int(value) != int(mpb_branch))):
        branch_gc_count = int(np.count_nonzero(high_z_gc & (branch_ids == int(branch))))
        visible = (tree_branch == int(branch)) & (tree_redshift >= FIG11_TARGET_REDSHIFT)
        if not np.any(visible):
            # Omit a branch with no visible raw-tree row from both the plot and
            # the reported satellite count.
            continue
        visible_logmh = tree_logmh[visible]
        visible_redshift = tree_redshift[visible]
        visible_mass = tree_mass[visible]
        visible_time = np.asarray([Redshift2CosmicAge(float(z), time_unit="Gyr") for z in visible_redshift], dtype=float)
        order = np.argsort(visible_time, kind="mergesort")
        visible_logmh = visible_logmh[order]
        visible_redshift = visible_redshift[order]
        visible_mass = visible_mass[order]
        visible_time = visible_time[order]
        x_gyr = float(Redshift2CosmicAge(0.0, time_unit="Gyr")) - visible_time
        maximum_logmh = float(np.max(visible_logmh))
        maximum_indices = np.flatnonzero(visible_logmh == maximum_logmh)
        marker_index = int(maximum_indices[-1])
        tracks.append(
            {
                "branch_id": int(branch),
                "n_gc_high_z": branch_gc_count,
                "x_gyr": x_gyr,
                "redshift": visible_redshift,
                "log10_halo_mass": visible_logmh,
                "halo_mass_msun": visible_mass,
                "marker_x_gyr": float(x_gyr[marker_index]),
                "marker_log10_halo_mass": maximum_logmh,
                "marker_halo_mass_msun": float(visible_mass[marker_index]),
            }
        )
    return tracks


def _fig11_load_history(candidate, formation):
    tree_path = _fig11_fixed_tree_path(candidate["fixed_tree_basename"])
    tree_rows = _fig11_read_full_tree_numeric(tree_path)
    _fig11_tree_logmass_and_redshift(tree_rows, "full")
    mpb_branch = _mpb_branch_id(tree_rows)
    mpb_rows = read_haloevo_mpb(tree_path)
    main = _fig11_main_track(tree_path, mpb_rows=mpb_rows)
    row_indices = np.flatnonzero(formation["halo_id_z0"] == int(candidate["halo_id_z0"]))
    if len(row_indices):
        branch_ids = _fig11_map_formation_rows_to_branches(
            formation["logMh_form"][row_indices],
            formation["zform"][row_indices],
            formation["subfind_form"][row_indices],
            tree_rows,
        )
        allcat_is_mpb = formation["isMPB"][row_indices].astype(bool)
        expected_is_mpb = branch_ids == int(mpb_branch)
        if not np.array_equal(allcat_is_mpb, expected_is_mpb):
            mismatch = np.flatnonzero(allcat_is_mpb != expected_is_mpb)
            first = int(mismatch[0]) if len(mismatch) else -1
            raise ValueError(
                f"Fig. 11 allcat isMPB disagrees with raw-tree branch membership for halo {int(candidate['halo_id_z0'])}; "
                f"first local mismatch={first}."
            )
        satellite_tracks = _fig11_satellite_tracks(
            tree_rows,
            mpb_branch,
            branch_ids,
            {
                "status": formation["status"][row_indices],
                "zform": formation["zform"][row_indices],
            },
        )
    else:
        satellite_tracks = _fig11_satellite_tracks(
            tree_rows,
            mpb_branch,
            np.asarray([], dtype=np.int64),
            {
                "status": np.asarray([], dtype=np.int64),
                "zform": np.asarray([], dtype=float),
            },
        )
    history = dict(candidate)
    history.update(
        {
            "tree_path": tree_path,
            "mpb_branch_id": int(mpb_branch),
            "main": main,
            "satellites": satellite_tracks,
            "n_satellites": len(satellite_tracks),
        }
    )
    return history


def _fig11_lookup_table(tng_volume_context):
    lookup = tng_volume_context.get("lookup") if isinstance(tng_volume_context, dict) else None
    if not isinstance(lookup, pd.DataFrame):
        raise ValueError("Fig. 11 requires the validated TNG lookup in tng_volume_context.")
    required = ["halo_id_z0", "simulation_key", "fixed_tree_basename"]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"Fig. 11 TNG lookup is missing required provenance fields: {missing}")
    out = lookup.loc[:, [name for name in lookup.columns if name in required + ["simulation"]]].copy()
    out["halo_id_z0"] = _fig11_integer_values(out["halo_id_z0"].to_numpy(dtype=float), "TNG lookup halo IDs")
    out["simulation_key"] = out["simulation_key"].fillna("").astype(str).str.strip()
    out["fixed_tree_basename"] = out["fixed_tree_basename"].fillna("").astype(str).str.strip()
    if "simulation" in out.columns:
        out["simulation"] = out["simulation"].fillna("").astype(str).str.strip()
    if out["simulation_key"].eq("").any() or out["fixed_tree_basename"].eq("").any():
        raise ValueError("Fig. 11 TNG lookup contains blank suite or fixed-tree provenance values.")
    if not set(out["simulation_key"]).issubset(set(TNG_SUITE_KEYS)):
        raise ValueError(f"Fig. 11 TNG lookup contains unsupported suites: {sorted(set(out['simulation_key']) - set(TNG_SUITE_KEYS))}")
    if out["halo_id_z0"].duplicated().any():
        raise ValueError("Fig. 11 TNG lookup contains duplicate model-facing halo IDs.")
    return out


def _fig11_exact_catalogue_rows(summary_by_z, best_id):
    column_aliases = {
        "halo_id_z0": "halo_id_z0",
        "redshift": "redshift" if "redshift" in summary_by_z.columns else "z_out",
        "halo_mass_available": "halo_mass_available",
        "log10_halo_mass_at_redshift": "log10_halo_mass_at_redshift" if "log10_halo_mass_at_redshift" in summary_by_z.columns else "logMh_z_msun",
    }
    missing = [name for name, source in column_aliases.items() if source not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing Fig. 11 catalogue fields: {missing}")
    table = summary_by_z.loc[:, list(column_aliases.values())].copy()
    table.columns = list(column_aliases.keys())
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id = _fig11_integer_values(table["halo_id_z0"].to_numpy(dtype=float), "haloSummaryByZ halo IDs")
    redshift = table["redshift"].to_numpy(dtype=float)
    available = table["halo_mass_available"].to_numpy(dtype=float)
    logmh = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("haloSummaryByZ contains invalid redshifts for Fig. 11.")
    exact = np.abs(redshift - FIG11_TARGET_REDSHIFT) <= FIG11_REDSHIFT_ROW_ATOL
    if not np.any(exact):
        raise ValueError(f"haloSummaryByZ contains no exact z={FIG11_TARGET_REDSHIFT:g} catalogue row.")
    exact_table = pd.DataFrame(
        {
            "halo_id_z0": halo_id[exact],
            "catalogue_redshift": redshift[exact],
            "halo_mass_available": available[exact],
            "catalogue_log10_halo_mass": logmh[exact],
        }
    )
    if exact_table["halo_id_z0"].duplicated().any():
        duplicate_ids = exact_table.loc[exact_table["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].drop_duplicates().tolist()
        raise ValueError(f"haloSummaryByZ contains duplicate exact z={FIG11_TARGET_REDSHIFT:g} rows: {duplicate_ids[:10]}")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exact_mass = np.power(10.0, exact_table["catalogue_log10_halo_mass"].to_numpy(dtype=float))
    exact_table["catalogue_halo_mass_msun"] = exact_mass
    best_rows = exact_table.loc[exact_table["halo_id_z0"] == int(best_id)]
    if len(best_rows) != 1:
        raise ValueError(f"Fig. 11 Fig. 02 best halo {int(best_id)} lacks one unique exact z=7 catalogue row.")
    best_row = best_rows.iloc[0]
    if float(best_row["halo_mass_available"]) != 1.0 or not np.isfinite(float(best_row["catalogue_log10_halo_mass"])) or not np.isfinite(float(best_row["catalogue_halo_mass_msun"])) or float(best_row["catalogue_halo_mass_msun"]) <= 0.0:
        raise ValueError(f"Fig. 11 Fig. 02 best halo {int(best_id)} has an invalid exact z=7 catalogue mass gate row.")
    exact_table["catalogue_mass_valid"] = (
        (exact_table["halo_mass_available"] == 1.0)
        & np.isfinite(exact_table["catalogue_log10_halo_mass"])
        & np.isfinite(exact_table["catalogue_halo_mass_msun"])
        & (exact_table["catalogue_halo_mass_msun"] > 0.0)
    )
    return exact_table, best_row


def _fig11_candidate_row(row, suite_label=None):
    simulation_key = str(row["simulation_key"])
    if suite_label is None:
        suite_label = {"tng50_1_dark": "TNG50", "tng100_1_dark": "TNG100"}.get(simulation_key, simulation_key)
    candidate = row.to_dict()
    candidate["halo_id_z0"] = int(row["halo_id_z0"])
    candidate["catalogue_halo_mass_msun"] = float(row["catalogue_halo_mass_msun"])
    candidate["catalogue_log10_halo_mass"] = float(row["catalogue_log10_halo_mass"])
    candidate["suite_label"] = str(suite_label)
    return candidate


def _fig11_unavailable_score_fields():
    return {
        "score_keplerian": np.nan,
        "score_uv": np.nan,
        "score_keplerian_uv": np.nan,
        "score_keplerian_available": False,
        "score_uv_available": False,
        "score_keplerian_uv_available": False,
    }


def _fig11_score_lookup(score_table, fig02_best):
    """Join validated Fig. 02 scores to Fig. 11 histories by integer halo ID."""

    if not isinstance(score_table, pd.DataFrame):
        raise ValueError("Fig. 11 score propagation requires the Fig. 02 score table as a pandas DataFrame.")
    redshift_column = "redshift" if "redshift" in score_table.columns else "z_out" if "z_out" in score_table.columns else None
    required = ["halo_id_z0", "keplerian_term", "uv_term", "score_keplerian_uv"]
    if redshift_column is None:
        required.append("redshift")
    else:
        required.append(redshift_column)
    missing = [name for name in required if name not in score_table.columns]
    if missing:
        raise ValueError(f"Fig. 11 score table is missing required columns: {missing}")

    table = score_table.loc[:, ["halo_id_z0", redshift_column, "keplerian_term", "uv_term", "score_keplerian_uv"]].copy()
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id_raw = table["halo_id_z0"].to_numpy(dtype=float)
    redshift = table[redshift_column].to_numpy(dtype=float)
    if np.any(~np.isfinite(halo_id_raw)) or np.any(np.abs(halo_id_raw - np.rint(halo_id_raw)) > 1.0e-8):
        raise ValueError("Fig. 11 target score rows contain non-finite or non-integer halo IDs.")
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 11 score rows contain non-finite or negative redshifts.")
    table["halo_id_z0"] = np.rint(halo_id_raw).astype(np.int64)
    keplerian = table["keplerian_term"].to_numpy(dtype=float)
    uv = table["uv_term"].to_numpy(dtype=float)
    combined = table["score_keplerian_uv"].to_numpy(dtype=float)
    if np.any(np.isfinite(keplerian) & (keplerian < 0.0)):
        raise ValueError("Fig. 11 score table contains a finite negative keplerian_term.")
    if np.any(np.isfinite(combined) & (combined < 0.0)):
        raise ValueError("Fig. 11 score table contains a finite negative score_keplerian_uv.")

    target = np.abs(redshift - FIG11_TARGET_REDSHIFT) <= FIG11_REDSHIFT_ROW_ATOL
    target_table = table.loc[target].copy()
    if len(target_table) == 0:
        raise ValueError(f"Fig. 11 score table contains no row at target z={FIG11_TARGET_REDSHIFT:g}.")
    if target_table["halo_id_z0"].duplicated().any():
        duplicate_ids = target_table.loc[target_table["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].drop_duplicates().tolist()
        raise ValueError(f"Fig. 11 score table contains duplicate target-redshift rows for halo IDs: {duplicate_ids[:10]}")

    target_keplerian = target_table["keplerian_term"].to_numpy(dtype=float)
    target_uv = target_table["uv_term"].to_numpy(dtype=float)
    target_combined = target_table["score_keplerian_uv"].to_numpy(dtype=float)
    finite_combined_missing_component = np.isfinite(target_combined) & (~np.isfinite(target_keplerian) | ~np.isfinite(target_uv))
    if np.any(finite_combined_missing_component):
        bad_ids = target_table.loc[finite_combined_missing_component, "halo_id_z0"].astype(int).tolist()
        raise ValueError(f"Fig. 11 finite combined scores require finite keplerian and UV components: halo IDs={bad_ids[:10]}")
    finite_all = np.isfinite(target_keplerian) & np.isfinite(target_uv) & np.isfinite(target_combined)
    if np.any(finite_all):
        expected_combined = np.sqrt(
            QSO1_SCORE_WEIGHT_KEPLERIAN * target_keplerian[finite_all] ** 2
            + QSO1_SCORE_WEIGHT_MUV * target_uv[finite_all] ** 2
        )
        if not np.allclose(target_combined[finite_all], expected_combined, rtol=FIG11_SCORE_RTOL, atol=FIG11_SCORE_ATOL):
            bad_ids = target_table.loc[finite_all, "halo_id_z0"].astype(int).to_numpy()
            mismatch = ~np.isclose(target_combined[finite_all], expected_combined, rtol=FIG11_SCORE_RTOL, atol=FIG11_SCORE_ATOL)
            raise ValueError(f"Fig. 11 score table violates the weighted score formula for halo IDs={bad_ids[mismatch][:10].tolist()}")

    try:
        best_id_raw = float(fig02_best["halo_id_z0"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fig. 11 cannot validate the Fig. 02 best-halo score without halo_id_z0.") from exc
    if not np.isfinite(best_id_raw) or abs(best_id_raw - np.rint(best_id_raw)) > 1.0e-8:
        raise ValueError(f"Fig. 11 Fig. 02 best halo ID is not a finite integer: {best_id_raw!r}")
    best_id = int(np.rint(best_id_raw))
    best_rows = target_table.loc[target_table["halo_id_z0"] == best_id]
    if len(best_rows) != 1:
        raise ValueError(f"Fig. 11 Fig. 02 best halo {best_id} lacks one unique target-redshift score row.")
    best_row = best_rows.iloc[0]
    for column in ("keplerian_term", "uv_term", "score_keplerian_uv"):
        try:
            best_value = float(fig02_best[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Fig. 11 Fig. 02 best-halo score is missing {column}.") from exc
        joined_value = float(best_row[column])
        if not np.isfinite(best_value) or not np.isfinite(joined_value) or not np.isclose(joined_value, best_value, rtol=FIG11_SCORE_RTOL, atol=FIG11_SCORE_ATOL):
            raise ValueError(
                f"Fig. 11 best-halo {column} mismatch for halo_id_z0={best_id}: "
                f"joined={joined_value!r}, Fig. 02={best_value!r}."
            )

    by_halo_id = {}
    for row in target_table.itertuples(index=False):
        row_keplerian = float(row.keplerian_term)
        row_uv = float(row.uv_term)
        row_combined = float(row.score_keplerian_uv)
        keplerian_available = bool(np.isfinite(row_keplerian))
        uv_available = bool(np.isfinite(row_uv))
        combined_available = bool(np.isfinite(row_combined) and keplerian_available and uv_available)
        by_halo_id[int(row.halo_id_z0)] = {
            "score_keplerian": row_keplerian if keplerian_available else np.nan,
            "score_uv": row_uv if uv_available else np.nan,
            "score_keplerian_uv": row_combined if combined_available else np.nan,
            "score_keplerian_available": keplerian_available,
            "score_uv_available": uv_available,
            "score_keplerian_uv_available": combined_available,
        }
    return {"by_halo_id": by_halo_id, "target_rows": target_table, "target_redshift": FIG11_TARGET_REDSHIFT}


def _fig11_score_text(value, available):
    return f"{float(value):.2f}" if bool(available) and np.isfinite(float(value)) else "n/a"


def _fig11_score_annotation(history):
    keplerian = _fig11_score_text(history["score_keplerian"], history["score_keplerian_available"])
    uv = _fig11_score_text(history["score_uv"], history["score_uv_available"])
    combined_available = (
        history["score_keplerian_available"]
        and history["score_uv_available"]
        and history["score_keplerian_uv_available"]
    )
    combined = _fig11_score_text(history["score_keplerian_uv"], combined_available)
    keplerian_text = rf"$S_{{\rm K}}={keplerian}$" if keplerian != "n/a" else r"$S_{\rm K}$=n/a"
    uv_text = rf"$S_{{\rm UV}}={uv}$" if uv != "n/a" else r"$S_{\rm UV}$=n/a"
    combined_text = rf"$S_{{\rm K+UV}}={combined}$" if combined != "n/a" else r"$S_{\rm K+UV}$=n/a"
    return f"{keplerian_text}, {uv_text}\n{combined_text}"


def _fig11_score_diagnostic(history):
    keplerian = _fig11_score_text(history["score_keplerian"], history["score_keplerian_available"])
    uv = _fig11_score_text(history["score_uv"], history["score_uv_available"])
    combined_available = (
        history["score_keplerian_available"]
        and history["score_uv_available"]
        and history["score_keplerian_uv_available"]
    )
    combined = _fig11_score_text(history["score_keplerian_uv"], combined_available)
    return f"S_K={keplerian}, S_UV={uv}, S_K+UV={combined}"


def select_fig11_assembly_histories(out_dir, summary_by_z, final_gc, metadata, tng_volume_context, score_table, fig02_best):
    """Select Fig. 11 panels and build their raw-tree assembly histories."""

    if not isinstance(metadata, dict):
        raise ValueError("Fig. 11 requires the validated run metadata mapping.")
    try:
        best_id_value = float(fig02_best["halo_id_z0"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fig. 11 cannot read the Fig. 02 best halo ID.") from exc
    if not np.isfinite(best_id_value) or abs(best_id_value - np.rint(best_id_value)) > 1.0e-8:
        raise ValueError(f"Fig. 11 Fig. 02 best halo ID is not a finite integer: {best_id_value!r}")
    best_id = int(np.rint(best_id_value))
    score_lookup = _fig11_score_lookup(score_table, fig02_best)
    scores_by_halo_id = score_lookup["by_halo_id"]
    formation = _load_fig11_formation_catalogue(out_dir, final_gc)
    exact_table, best_row = _fig11_exact_catalogue_rows(summary_by_z, best_id)
    lookup = _fig11_lookup_table(tng_volume_context)
    missing_provenance = sorted(set(exact_table["halo_id_z0"].tolist()) - set(lookup["halo_id_z0"].tolist()))
    if missing_provenance:
        raise ValueError(f"Fig. 11 exact z=7 catalogue rows lack strict TNG provenance: {missing_provenance[:10]}")
    candidate_table = exact_table.merge(lookup, on="halo_id_z0", how="left", validate="one_to_one")
    best_lookup = candidate_table.loc[candidate_table["halo_id_z0"] == best_id]
    if len(best_lookup) != 1:
        raise ValueError(f"Fig. 11 Fig. 02 best halo {best_id} lacks one unique strict TNG provenance row.")
    best_candidate = _fig11_candidate_row(best_lookup.iloc[0])
    # The best halo is a fatal gate: the first panel cannot be replaced by a
    # nearby halo if its raw MPB is incomplete at z=7.
    best_history = _fig11_load_history(best_candidate, formation)
    best_history.update(scores_by_halo_id.get(best_id, _fig11_unavailable_score_fields()))

    best_mass = float(best_row["catalogue_halo_mass_msun"])
    comparisons = candidate_table.loc[
        candidate_table["catalogue_mass_valid"] & (candidate_table["halo_id_z0"] != best_id)
    ].copy()
    comparisons["mass_delta_msun"] = np.abs(comparisons["catalogue_halo_mass_msun"].to_numpy(dtype=float) - best_mass)
    comparisons = comparisons.sort_values(
        ["mass_delta_msun", "halo_id_z0", "fixed_tree_basename"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    histories = [best_history]
    rejected = []
    for _, row in comparisons.iterrows():
        if len(histories) >= FIG11_MAX_PANELS:
            break
        candidate = _fig11_candidate_row(row)
        try:
            history = _fig11_load_history(candidate, formation)
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
            rejected.append({"halo_id_z0": int(candidate["halo_id_z0"]), "reason": str(exc)})
            continue
        history.update(scores_by_halo_id.get(int(candidate["halo_id_z0"]), _fig11_unavailable_score_fields()))
        histories.append(history)
    return {
        "histories": histories,
        "rejected": rejected,
        "formation_catalogue_path": formation["path"],
        "best_halo_id_z0": best_id,
        "target_redshift": FIG11_TARGET_REDSHIFT,
    }


def _read_tng_lookup(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing TNG lookup: {path}")
    lookup = pd.read_csv(path)
    required = [
        "file_index",
        "simulation",
        "simulation_key",
        "halo_id_z0",
        "subhalo_id_z0",
        "label",
        "raw_tree_basename",
        "fixed_tree_basename",
    ]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"{path} is missing TNG lookup columns: {missing}")
    for column in ("file_index", "halo_id_z0", "subhalo_id_z0"):
        values = pd.to_numeric(lookup[column], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(np.abs(values - np.rint(values)) > 1.0e-8):
            raise ValueError(f"{path} contains non-finite or non-integer {column} values.")
        lookup[column] = np.rint(values).astype(np.int64)
    for column in ("simulation", "simulation_key", "label", "raw_tree_basename", "fixed_tree_basename"):
        lookup[column] = lookup[column].fillna("").astype(str).str.strip()
        if lookup[column].eq("").any():
            raise ValueError(f"{path} contains empty {column} values.")
    if not set(lookup["simulation_key"]).issubset(set(TNG_SUITE_KEYS)):
        raise ValueError(f"{path} contains an unsupported TNG suite.")
    if lookup.duplicated(["simulation_key", "fixed_tree_basename"]).any():
        raise ValueError(f"{path} contains duplicate suite/fixed-tree provenance keys.")
    return lookup


def _load_tng_catalogue_provenance():
    manifest_path = TNG_CATALOGUE_ROOT / TNG_TARGET_MANIFEST_FILENAME
    metadata_path = TNG_CATALOGUE_ROOT / TNG_TARGET_METADATA_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing TNG target manifest: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing TNG target metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    simulations = metadata.get("simulations")
    full_box = metadata.get("full_box_selection")
    if not isinstance(simulations, dict) or not isinstance(full_box, dict):
        raise ValueError("TNG metadata must contain simulations and full_box_selection.")
    if full_box.get("geometry") != "native_full_simulation_box" or bool(full_box.get("coordinate_filter_applied")) or bool(full_box.get("periodic_wrapping")):
        raise ValueError("TNG metadata does not describe unfiltered native full-box selection.")

    volume_by_suite = {}
    for simulation_key in TNG_SUITE_KEYS:
        spec = simulations.get(simulation_key)
        if not isinstance(spec, dict):
            raise ValueError(f"TNG metadata is missing {simulation_key}.")
        h = float(spec.get("h", np.nan))
        box_size_ckpc_h = float(spec.get("box_size_ckpc_h", np.nan))
        if not np.isfinite(h) or h <= 0.0 or not np.isfinite(box_size_ckpc_h) or box_size_ckpc_h <= 0.0:
            raise ValueError(f"TNG metadata has invalid h or box size for {simulation_key}.")
        side_native = box_size_ckpc_h / 1000.0
        side_physical = side_native / h
        if not np.isclose(float(full_box["side_native_cmpc_h"][simulation_key]), side_native, rtol=0.0, atol=1.0e-10) or not np.isclose(float(full_box["side_physical_cmpc"][simulation_key]), side_physical, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"TNG full-box side metadata is inconsistent for {simulation_key}.")
        if not np.isclose(float(full_box["volume_physical_cmpc3"][simulation_key]), side_physical**3, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"TNG full-box physical volume metadata is inconsistent for {simulation_key}.")
        volume_by_suite[simulation_key] = float(full_box["volume_physical_cmpc3"][simulation_key])

    rules = {
        "tng50_1_dark": "full_box_Group_M_Mean200_gt_1e10_msun_and_le_1e13_msun",
        "tng100_1_dark": "full_box_Group_M_Mean200_gt_1e13_msun",
    }
    try:
        manifest = pd.read_csv(manifest_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read TNG target manifest: {manifest_path}") from error
    required_manifest = ["simulation", "simulation_key", "halo_id_z0", "subhalo_id_z0", "selection_rule", "raw_tree_basename", "fixed_tree_basename"]
    missing = [name for name in required_manifest if name not in manifest.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing TNG manifest columns: {missing}")
    manifest["simulation_key"] = manifest["simulation_key"].fillna("").astype(str).str.strip()
    manifest["selection_rule"] = manifest["selection_rule"].fillna("").astype(str).str.strip()
    manifest["fixed_tree_basename"] = manifest["fixed_tree_basename"].fillna("").astype(str).str.strip()
    manifest["raw_tree_basename"] = manifest["raw_tree_basename"].fillna("").astype(str).str.strip()
    if not set(manifest["simulation_key"]).issubset(set(TNG_SUITE_KEYS)):
        raise ValueError(f"{manifest_path} contains an unsupported TNG suite.")
    if manifest.duplicated("fixed_tree_basename").any():
        raise ValueError(f"{manifest_path} contains duplicate fixed-tree basenames.")
    for simulation_key, selection_rule in rules.items():
        rows = manifest.loc[manifest["simulation_key"].eq(simulation_key)]
        if len(rows) and set(rows["selection_rule"]) != {selection_rule}:
            raise ValueError(f"{manifest_path} contains an unexpected selection rule for {simulation_key}.")
    metadata_counts = metadata.get("counts", {}).get("selected_by_simulation", {})
    manifest_counts = {key: int((manifest["simulation_key"] == key).sum()) for key in TNG_SUITE_KEYS}
    for key in TNG_SUITE_KEYS:
        if int(metadata_counts.get(key, -1)) != manifest_counts[key]:
            raise ValueError(f"TNG metadata and manifest counts disagree for {key}.")

    fixed_tree_dir = TNG_CATALOGUE_ROOT / TNG_FIXED_TREE_DIRNAME
    original = _read_tng_lookup(fixed_tree_dir / TNG_ORIGINAL_LOOKUP_FILENAME)
    shifted = _read_tng_lookup(fixed_tree_dir / TNG_SHIFTED_LOOKUP_FILENAME)
    expected = manifest.set_index(["simulation_key", "fixed_tree_basename"], drop=False)
    original = original.set_index(["simulation_key", "fixed_tree_basename"], drop=False)
    shifted = shifted.set_index(["simulation_key", "fixed_tree_basename"], drop=False)
    if len(original) != len(manifest) or len(shifted) != len(manifest) or set(original.index) != set(expected.index) or set(shifted.index) != set(expected.index):
        raise ValueError("TNG lookup files do not contain exactly one row per current manifest target.")
    final_ids = []
    for key in expected.index:
        manifest_row = expected.loc[key]
        original_row = original.loc[key]
        shifted_row = shifted.loc[key]
        if int(original_row["halo_id_z0"]) != int(manifest_row["halo_id_z0"]):
            raise ValueError(f"Original lookup halo ID disagrees with manifest for {key}.")
        required_shifted = int(manifest_row["halo_id_z0"]) + (TNG100_HALO_ID_OFFSET if key[0] == "tng100_1_dark" else 0)
        if int(shifted_row["halo_id_z0"]) != required_shifted:
            raise ValueError(f"Shifted lookup halo ID disagrees with the required offset for {key}.")
        final_ids.append(required_shifted)
    if len(final_ids) != len(set(final_ids)):
        raise ValueError("The model-facing TNG lookup contains duplicate final halo IDs.")
    return {
        "manifest": manifest.reset_index(drop=True),
        "metadata": metadata,
        "lookup_original": original.reset_index(drop=True),
        "lookup_final": shifted.reset_index(drop=True),
        "volume_tng50_cmpc3": volume_by_suite["tng50_1_dark"],
        "volume_tng100_cmpc3": volume_by_suite["tng100_1_dark"],
        "tng100_weight": float(volume_by_suite["tng50_1_dark"] / volume_by_suite["tng100_1_dark"]),
        "manifest_counts": manifest_counts,
    }


def _load_tng_tree_lookup(out_dir, catalogue):
    path = Path(out_dir).resolve() / TNG_TREE_LOOKUP_FILENAME
    lookup = pd.read_csv(path) if path.exists() else None
    if lookup is None:
        raise FileNotFoundError(f"Missing required TNG halo-tree lookup: {path}")
    if "halo_id_z0" not in lookup.columns and "hid_z0" in lookup.columns:
        lookup = lookup.rename(columns={"hid_z0": "halo_id_z0"})
    required = ["halo_id_z0", "simulation_key", "fixed_tree_basename"]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"{path} is missing TNG halo-tree lookup columns: {missing}")
    halo_id_raw = pd.to_numeric(lookup["halo_id_z0"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(halo_id_raw)) or np.any(np.abs(halo_id_raw - np.rint(halo_id_raw)) > 1.0e-8):
        raise ValueError(f"{path} contains non-finite or non-integer halo IDs.")
    lookup["halo_id_z0"] = np.rint(halo_id_raw).astype(np.int64)
    lookup["simulation_key"] = lookup["simulation_key"].fillna("").astype(str).str.strip()
    lookup["fixed_tree_basename"] = lookup["fixed_tree_basename"].fillna("").astype(str).str.strip()
    if lookup["simulation_key"].eq("").any() or lookup["fixed_tree_basename"].eq("").any() or lookup.duplicated(["simulation_key", "fixed_tree_basename"]).any():
        raise ValueError(f"{path} contains invalid or duplicate suite/fixed-tree provenance.")
    catalogue_lookup = catalogue["lookup_final"][["simulation_key", "fixed_tree_basename", "halo_id_z0"]].rename(columns={"halo_id_z0": "catalogue_halo_id_z0"})
    joined = lookup.merge(catalogue_lookup, on=["simulation_key", "fixed_tree_basename"], how="left", validate="one_to_one")
    if joined["catalogue_halo_id_z0"].isna().any():
        missing_names = joined.loc[joined["catalogue_halo_id_z0"].isna(), "fixed_tree_basename"].tolist()
        raise ValueError(f"{path} contains fixed-tree names absent from the shifted catalogue lookup: {missing_names[:10]}")
    mismatch = joined["halo_id_z0"].to_numpy(dtype=np.int64) != joined["catalogue_halo_id_z0"].to_numpy(dtype=np.int64)
    if np.any(mismatch):
        raise ValueError("Model halo_tree_lookup.csv does not use the shifted catalogue halo-ID namespace.")
    return joined


def attach_tng_volume_weights(out_dir, summary_by_z, final_gc):
    catalogue = _load_tng_catalogue_provenance()
    lookup = _load_tng_tree_lookup(out_dir, catalogue)
    summary_ids = pd.to_numeric(summary_by_z["halo_id_z0"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(summary_ids)) or np.any(np.abs(summary_ids - np.rint(summary_ids)) > 1.0e-8):
        raise ValueError("haloSummaryByZ contains non-finite or non-integer halo IDs before TNG weighting.")
    summary_ids_int = np.rint(summary_ids).astype(np.int64)
    lookup_id_set = set(lookup["halo_id_z0"].tolist())
    missing_summary_ids = sorted(set(summary_ids_int.tolist()) - lookup_id_set)
    if missing_summary_ids:
        raise ValueError(f"TNG halo-tree lookup is missing haloSummaryByZ IDs: {missing_summary_ids[:10]}")

    weight_by_halo = dict(zip(lookup["halo_id_z0"].tolist(), np.where(lookup["simulation_key"].eq("tng100_1_dark"), catalogue["tng100_weight"], 1.0)))
    weighted_summary = summary_by_z.copy()
    weighted_summary["volume_weight_tng50"] = [float(weight_by_halo[int(halo_id)]) for halo_id in summary_ids_int]

    if "halo_id_z0" not in final_gc.columns:
        raise ValueError("finalGCs.dat must contain halo_id_z0 for strict TNG Satellite volume weighting.")
    final_ids = pd.to_numeric(final_gc["halo_id_z0"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(final_ids)) or np.any(np.abs(final_ids - np.rint(final_ids)) > 1.0e-8):
        raise ValueError("finalGCs.dat contains non-finite or non-integer parent halo IDs.")
    final_ids_int = np.rint(final_ids).astype(np.int64)
    missing_final_ids = sorted(set(final_ids_int.tolist()) - lookup_id_set)
    if missing_final_ids:
        raise ValueError(f"TNG halo-tree lookup is missing finalGCs.dat parent IDs: {missing_final_ids[:10]}")
    weighted_final_gc = final_gc.copy()
    weighted_final_gc["volume_weight_tng50"] = [float(weight_by_halo[int(halo_id)]) for halo_id in final_ids_int]

    output_counts = {key: int(value) for key, value in lookup["simulation_key"].value_counts().to_dict().items()}
    return weighted_summary, weighted_final_gc, {
        "lookup": lookup,
        "manifest_counts": catalogue["manifest_counts"],
        "output_counts": output_counts,
        "volume_tng50_cmpc3": float(catalogue["volume_tng50_cmpc3"]),
        "volume_tng100_cmpc3": float(catalogue["volume_tng100_cmpc3"]),
        "tng100_weight": float(catalogue["tng100_weight"]),
    }


def _set_tng_volume_context(context):
    """Use the current full-box metadata for all physical-density figures."""

    global FIG09_BHMF_SIDE_CMPC, FIG09_BHMF_VOLUME_CMPC3, ABUNDANCE_MATCHING_VOLUME_CMPC3
    volume_tng50 = float(context["volume_tng50_cmpc3"])
    if not np.isfinite(volume_tng50) or volume_tng50 <= 0.0:
        raise ValueError("TNG50 full-box volume must be finite and positive.")
    FIG09_BHMF_VOLUME_CMPC3 = volume_tng50
    FIG09_BHMF_SIDE_CMPC = float(volume_tng50 ** (1.0 / 3.0))
    ABUNDANCE_MATCHING_VOLUME_CMPC3 = volume_tng50


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
    #if len(weights) != 4 or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0) or not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12):
    #    raise ValueError("QSO1_VELOCITY_GROUP_WEIGHTS must contain four positive finite values that sum to unity.")

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

    spectro = _component_rows("spectroastrometry", 2)
    spectro_fine = _component_rows("spectroastrometry_fine", 4)
    resolved = _component_rows("resolved_kinematics", 4)
    groups = [
        ("spectroastrometry", spectro, float(weights[0])),
        ("spectroastrometry_inner", spectro_fine.iloc[:2].copy(), float(weights[1])),
        ("spectroastrometry_outer", spectro_fine.iloc[2:].copy(), float(weights[2])),
        ("resolved_inner", resolved.iloc[:2].copy(), float(weights[3])),
        ("resolved_outer", resolved.iloc[2:].copy(), float(weights[4])),
    ]

    weighted = []
    for name, rows, group_weight in groups:
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


def _candidate_no_score_error(out_dir, score_table):
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
        f"out_dir={Path(out_dir).resolve()}, "
        f"redshift-selected candidates={n_candidates}, finite Keplerian terms={finite_keplerian}, "
        f"finite UV values={finite_uv}, finite Keplerian+UV scores={finite_score}, "
        f"finite formed 6 pc mass={finite_formed}, usable GC-origin age weights={usable_age}, "
        f"first missing producers: {reason_text}."
    )


def score_fig02_candidate_haloes(out_dir, points, z_rows, deposit_profile, final_gc, uv_calibration):
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
        raise ValueError(_candidate_no_score_error(out_dir, score_table))
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


def load_chen2026_fig05a_seed_mass_functions():
    """Load and validate the digitised Chen+2026 panel-5a curves."""

    required = [
        "curve_id", "curve_label", "curve_role", "redshift",
        "log10_mbh_seed_msun", "phi_mpc3_dex1", "colour", "linestyle", "source",
    ]
    if not CHEN2026_FIG05A_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Chen+2026 Fig. 5a reference table: {CHEN2026_FIG05A_DATA_PATH}")
    table = pd.read_csv(CHEN2026_FIG05A_DATA_PATH, dtype=str, keep_default_na=False)
    if list(table.columns) != required:
        raise ValueError(f"Chen+2026 Fig. 5a columns must be exactly {required}, got {list(table.columns)}")
    if table.empty:
        raise ValueError("Chen+2026 Fig. 5a reference table is empty.")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    for column in ["curve_id", "curve_label", "curve_role", "colour", "linestyle", "source"]:
        if table[column].eq("").any():
            raise ValueError(f"Chen+2026 Fig. 5a contains empty {column} values.")

    numeric = ["redshift", "log10_mbh_seed_msun", "phi_mpc3_dex1"]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[numeric].isna().any().any() or not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Chen+2026 Fig. 5a contains non-finite numeric values.")
    if not np.allclose(table["redshift"].to_numpy(dtype=float), CHEN2026_FIG05A_REDSHIFT, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Chen+2026 Fig. 5a must contain only the reference redshift z={CHEN2026_FIG05A_REDSHIFT:g}.")
    log_mass = table["log10_mbh_seed_msun"].to_numpy(dtype=float)
    phi = table["phi_mpc3_dex1"].to_numpy(dtype=float)
    if np.any(log_mass < FIG10_SEED_LOGM_BIN_EDGES[0] - 1.0e-10) or np.any(log_mass > FIG10_SEED_LOGM_BIN_EDGES[-1] + 1.0e-10):
        raise ValueError("Chen+2026 Fig. 5a mass coordinates lie outside the calibrated 0--5 dex range.")
    if np.any(phi <= 0.0):
        raise ValueError("Chen+2026 Fig. 5a Phi values must be finite and positive.")
    invalid_colours = [value for value in table["colour"].unique() if not mpl.colors.is_color_like(value)]
    if invalid_colours:
        raise ValueError(f"Chen+2026 Fig. 5a contains invalid colours: {invalid_colours}")
    valid_linestyles = {"-", "--", "-.", ":", "None", "none", "solid", "dashed", "dashdot", "dotted"}
    invalid_linestyles = sorted(set(table["linestyle"]) - valid_linestyles)
    if invalid_linestyles:
        raise ValueError(f"Chen+2026 Fig. 5a contains invalid line styles: {invalid_linestyles}")
    if table.duplicated(["curve_role", "log10_mbh_seed_msun"]).any():
        raise ValueError("Chen+2026 Fig. 5a contains duplicate curve-coordinate pairs.")

    roles = set(table["curve_role"])
    if roles != set(FIG10_CHEN_CURVE_ROLES):
        raise ValueError(f"Chen+2026 Fig. 5a curve roles must be exactly {FIG10_CHEN_CURVE_ROLES}, got {sorted(roles)}")
    if table["curve_id"].nunique() != len(FIG10_CHEN_CURVE_ROLES):
        raise ValueError("Chen+2026 Fig. 5a must contain one unique curve identifier per curve role.")

    by_role = {}
    for role in FIG10_CHEN_CURVE_ROLES:
        rows = table.loc[table["curve_role"].eq(role)].sort_values("log10_mbh_seed_msun").reset_index(drop=True)
        if rows["curve_label"].nunique() != 1 or rows["curve_label"].iloc[0] != FIG10_CHEN_CURVE_LABELS[role]:
            raise ValueError(f"Chen+2026 Fig. 5a label for {role!r} is missing or unexpected.")
        if rows["curve_id"].nunique() != 1 or rows["curve_id"].iloc[0] == "":
            raise ValueError(f"Chen+2026 Fig. 5a role {role!r} must have one non-empty curve identifier.")
        if len(rows) < 2 or np.any(np.diff(rows["log10_mbh_seed_msun"].to_numpy(dtype=float)) <= 0.0):
            raise ValueError(f"Chen+2026 Fig. 5a role {role!r} is not a strictly increasing curve.")
        if rows["colour"].nunique() != 1 or rows["linestyle"].nunique() != 1 or rows["source"].nunique() != 1:
            raise ValueError(f"Chen+2026 Fig. 5a role {role!r} has inconsistent plotting metadata.")
        by_role[role] = rows

    central = by_role["all_seeds_central"]
    lower = by_role["all_seeds_lower_envelope"]
    upper = by_role["all_seeds_upper_envelope"]
    central_x = central["log10_mbh_seed_msun"].to_numpy(dtype=float)
    lower_x = lower["log10_mbh_seed_msun"].to_numpy(dtype=float)
    upper_x = upper["log10_mbh_seed_msun"].to_numpy(dtype=float)
    if not np.array_equal(central_x, lower_x) or not np.array_equal(central_x, upper_x):
        raise ValueError("Chen+2026 All-seeds envelope curves must share the central mass grid.")
    central_phi = central["phi_mpc3_dex1"].to_numpy(dtype=float)
    lower_phi = lower["phi_mpc3_dex1"].to_numpy(dtype=float)
    upper_phi = upper["phi_mpc3_dex1"].to_numpy(dtype=float)
    if np.any(lower_phi > central_phi + 1.0e-15) or np.any(central_phi > upper_phi + 1.0e-15):
        raise ValueError("Chen+2026 All-seeds envelope ordering must be lower <= central <= upper.")

    sorted_table = table.sort_values(["curve_role", "log10_mbh_seed_msun"]).reset_index(drop=True)
    return {
        "table": sorted_table,
        "by_role": by_role,
        "roles": FIG10_CHEN_CURVE_ROLES,
        "redshift": CHEN2026_FIG05A_REDSHIFT,
        "log10_mass_range": (float(log_mass.min()), float(log_mass.max())),
        "phi_range": (float(phi.min()), float(phi.max())),
    }


def load_chen2026_fig06_seed_history():
    """Load and validate the four retained Chen+2026 Fig. 6a/6d curves."""

    required = [
        "curve_id", "curve_label", "curve_role", "panel", "quantity",
        "x_log10_1pz", "redshift", "value_mpc3", "colour", "linestyle", "source",
    ]
    if not CHEN2026_FIG06_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Chen+2026 Fig. 6 reference table: {CHEN2026_FIG06_DATA_PATH}")
    table = pd.read_csv(CHEN2026_FIG06_DATA_PATH, dtype=str, keep_default_na=False)
    if list(table.columns) != required:
        raise ValueError(f"Chen+2026 Fig. 6 columns must be exactly {required}, got {list(table.columns)}")
    if table.empty:
        raise ValueError("Chen+2026 Fig. 6 reference table is empty.")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    for column in ["curve_id", "curve_label", "curve_role", "panel", "quantity", "colour", "linestyle", "source"]:
        if table[column].eq("").any():
            raise ValueError(f"Chen+2026 Fig. 6 contains empty {column} values.")

    numeric = ["x_log10_1pz", "redshift", "value_mpc3"]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    numeric_values = table[numeric].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Chen+2026 Fig. 6 contains non-finite numeric values.")
    if np.any(table["value_mpc3"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Chen+2026 Fig. 6 reference values must be finite and positive.")
    if set(table["panel"]) != {"a", "d"}:
        raise ValueError(f"Chen+2026 Fig. 6 panels must be exactly ['a', 'd'], got {sorted(set(table['panel']))}")
    if set(table["curve_role"]) != set(FIG12_CHEN_CURVE_ROLES):
        raise ValueError(
            f"Chen+2026 Fig. 6 visible curve roles must be exactly {FIG12_CHEN_CURVE_ROLES}, "
            f"got {sorted(set(table['curve_role']))}. Fast/LW rows cannot be plotted."
        )
    if table.duplicated(["curve_id", "panel", "x_log10_1pz"]).any():
        raise ValueError("Chen+2026 Fig. 6 contains duplicate curve-coordinate rows.")
    valid_linestyles = {"-", "--", "-.", ":", "None", "none", "solid", "dashed", "dashdot", "dotted"}
    invalid_colours = [value for value in table["colour"].unique() if not mpl.colors.is_color_like(value)]
    invalid_linestyles = sorted(set(table["linestyle"]) - valid_linestyles)
    if invalid_colours:
        raise ValueError(f"Chen+2026 Fig. 6 contains invalid colours: {invalid_colours}")
    if invalid_linestyles:
        raise ValueError(f"Chen+2026 Fig. 6 contains invalid line styles: {invalid_linestyles}")

    by_panel = {panel: {} for panel in ("a", "d")}
    expected_quantity = {"a": "rate", "d": "cumulative"}
    for panel in ("a", "d"):
        panel_rows = table.loc[table["panel"].eq(panel)]
        if set(panel_rows["curve_role"]) != set(FIG12_CHEN_CURVE_ROLES):
            raise ValueError(f"Chen+2026 Fig. 6 panel {panel} does not contain exactly the four visible roles.")
        if set(panel_rows["quantity"]) != {expected_quantity[panel]}:
            raise ValueError(f"Chen+2026 Fig. 6 panel {panel} must have quantity={expected_quantity[panel]!r}.")
        for role in FIG12_CHEN_CURVE_ROLES:
            rows = panel_rows.loc[panel_rows["curve_role"].eq(role)].reset_index(drop=True)
            if len(rows) < 2:
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} has fewer than two vertices.")
            if rows["curve_label"].nunique() != 1 or rows["curve_label"].iloc[0] != FIG12_CHEN_CURVE_LABELS[role]:
                raise ValueError(f"Chen+2026 Fig. 6 label for {panel}/{role} is missing or unexpected.")
            if rows["curve_id"].nunique() != 1:
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} must have one curve identifier.")
            x_values = rows["x_log10_1pz"].to_numpy(dtype=float)
            if np.any(np.diff(x_values) <= 0.0):
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} is not strictly increasing in x.")
            if rows["colour"].nunique() != 1 or rows["linestyle"].nunique() != 1 or rows["source"].nunique() != 1:
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} has inconsistent plotting metadata.")
            expected_redshift = np.power(10.0, x_values) - 1.0
            if not np.allclose(rows["redshift"].to_numpy(dtype=float), expected_redshift, rtol=0.0, atol=FIG12_CHEN_REDSHIFT_ATOL):
                raise ValueError(f"Chen+2026 Fig. 6 redshift calibration disagrees with x for {panel}/{role}.")
            by_panel[panel][role] = rows

    sorted_table = table.sort_values(["panel", "curve_role", "x_log10_1pz"]).reset_index(drop=True)
    return {
        "table": sorted_table,
        "by_panel": by_panel,
        "roles": FIG12_CHEN_CURVE_ROLES,
        "panels": ("a", "d"),
        "redshift_atol": FIG12_CHEN_REDSHIFT_ATOL,
    }


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
    norm = mpl.colors.Normalize(vmin=3.0, vmax=10.0, clip=True)
    cmap = mpl.cm.jet

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        if z_out < 3.0:
            continue
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
        ax.plot(x, mean_mass, c=colour, ls="--", lw=1.5, zorder=2)
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
        ax.plot(qualified_x, np.where(qualified_valid, qualified_mean, np.nan), c=colour, ls="-", lw=2.0, zorder=2)
    if n_tracks == 0:
        raise ValueError("All binned central-BH tracks are empty or non-positive for Fig. 01.")
    ax.plot([], [], c="black", ls="--", lw=1.5, label="Model")
    ax.plot([], [], c="black", ls="-", lw=2.0, label=r"Model ($M_\bullet > 100~M_\odot$)")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label(r"Redshift $z$")
    colour_bar.set_ticks(np.arange(3.0, 10.1, 1.0))
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
    ax.set_xlabel(r"Stellar mass $M_\star (z)$ [$M_\odot$]")
    ax.set_ylabel(r"Nuclear BH mass $M_\bullet$ [$M_{\odot}$]")
    ax.set_xlim(left=10.0**5, right=10.0**x_limit_edges[-1])
    ax.set_ylim(bottom=1.0e2, top=1.0e10)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    legend = ax.legend(loc="upper left", fontsize=6.2, frameon=False, framealpha=0.85, ncol=2)
    for legend_text in legend.get_texts():
        if legend_text.get_text() == r"Model ($M_\bullet > 100~M_\odot$)":
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


def _fig08_total_velocity_profiles(deposit_profile, z_rows, best_halo_id=None):
    profile_halo_ids = np.asarray(deposit_profile["halo_ids"], dtype=int)
    if best_halo_id is not None:
        mass_column = next(
            (name for name in ("log10_halo_mass_at_redshift", "logMh_z_msun") if name in z_rows.columns),
            None,
        )
        if mass_column is None:
            raise ValueError("Fig. 02 mass filter requires a redshift-resolved halo-mass column.")
        halo_id_values = pd.to_numeric(z_rows["halo_id_z0"], errors="coerce").to_numpy(dtype=float)
        log_halo_mass = pd.to_numeric(z_rows[mass_column], errors="coerce").to_numpy(dtype=float)
        best_matches = np.flatnonzero(halo_id_values == float(best_halo_id))
        if len(best_matches) != 1:
            raise ValueError(f"Fig. 02 best halo_id_z0={int(best_halo_id)} is not unique in z_rows.")
        best_index = int(best_matches[0])
        available = np.ones(len(z_rows), dtype=bool)
        if "halo_mass_available" in z_rows.columns:
            available = pd.to_numeric(z_rows["halo_mass_available"], errors="coerce").to_numpy(dtype=float) == 1.0
        if not available[best_index] or not np.isfinite(log_halo_mass[best_index]):
            raise ValueError(f"Fig. 02 best halo_id_z0={int(best_halo_id)} has no valid redshift-resolved halo mass.")
        keep = available & np.isfinite(log_halo_mass) & (np.abs(log_halo_mass - log_halo_mass[best_index]) <= FIG08_HALO_MASS_WINDOW_DEX)
        z_rows = z_rows.loc[keep].copy()
        profile_keep = np.isin(profile_halo_ids, z_rows["halo_id_z0"].to_numpy(dtype=int))
        if not np.any(profile_keep):
            raise ValueError("Fig. 02 halo-mass filter removed every deposit profile.")
        deposit_profile = dict(deposit_profile)
        deposit_profile["halo_ids"] = profile_halo_ids[profile_keep]
        for name in ("r_outer_kpc", "cumulative_mass_msun", "cumulative_formed_mass_msun"):
            if name in deposit_profile:
                deposit_profile[name] = [value for value, keep_value in zip(deposit_profile[name], profile_keep) if keep_value]

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
    halo_ids, radius_pc, stellar_cumulative, velocity_profiles = _fig08_total_velocity_profiles(deposit_profile, z_rows, best_halo_id=int(best["halo_id_z0"]))
    matches = np.flatnonzero(halo_ids == int(best["halo_id_z0"]))
    if len(matches) != 1:
        raise ValueError(f"Fig. 02 Keplerian+UV best halo {int(best['halo_id_z0'])} is not present in the velocity-profile grid.")
    best_index = int(matches[0])
    median_velocity = np.median(velocity_profiles, axis=0)
    mean_velocity = np.mean(velocity_profiles, axis=0)
    low_velocity, high_velocity = np.percentile(velocity_profiles, FIG08_SCATTER_PERCENTILES, axis=0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    _plot_fig08_observed_curve(ax, curves, "point_mass_keplerian", r"Keplerian point mass $\log M_\bullet = 6.75$", color="black", linewidth=1.7, zorder=4)
    _plot_fig08_observed_curve(ax, curves, "mw_nsc", "MW-like NSC model", color="0.45", linestyle="dashdot", linewidth=1.5, zorder=3)
    for component, colour, marker, size, label in [
        ("resolved_kinematics", "tab:blue", "o", 5.0, "Resolved kinematics"),
        ("spectroastrometry", "magenta", "X", 6.0, "Spectroastrometry"),
        ("spectroastrometry_fine", "orchid", "P", 5.5, "Spectroastrometry, fine split"),
    ]:
        rows = points.loc[points["component"].eq(component)]
        if len(rows) == 0:
            continue
        xerr, yerr = _fig08_error_arrays(rows)
        ax.errorbar(rows["r_pc"].to_numpy(dtype=float), rows["v_km_s"].to_numpy(dtype=float), xerr=xerr, yerr=yerr, fmt=marker, ms=size, color=colour, ecolor=colour, elinewidth=1.0, markeredgecolor=colour, markerfacecolor=colour, capsize=0.0, linestyle="none", label=label, zorder=6)
    ax.fill_between(radius_pc, low_velocity, high_velocity, color="tab:green", alpha=0.16, linewidth=0.0, label=r"$z \simeq 7$ stack 16-84\%")
    ax.fill_between(-radius_pc[::-1], -high_velocity[::-1], -low_velocity[::-1], color="tab:green", alpha=0.16, linewidth=0.0)
    signed_r, signed_median = _fig08_signed_profile(radius_pc, median_velocity)
    _, signed_mean = _fig08_signed_profile(radius_pc, mean_velocity)
    _, signed_best = _fig08_signed_profile(radius_pc, velocity_profiles[best_index])
    #ax.plot(signed_r, signed_median, color="tab:green", linewidth=1.8, label=r"$z \simeq 7$ median simulation")
    #ax.plot(signed_r, signed_mean, color="tab:green", linewidth=1.2, linestyle="--", label="z~7 mean simulation")
    ax.plot(signed_r, signed_best, color="tab:red", linewidth=1.5, linestyle="-", label="Best Keplerian+UV halo")
    ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.set_xlim(-FIG08_RADIUS_MAX_PC, FIG08_RADIUS_MAX_PC)
    ax.set_ylim(-72.0, 72.0)
    ax.set_xlabel(r"Projected radius $r$ [pc]")
    ax.set_ylabel(r"Line-of-sight velocity $v$ [km/s]")
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


def _weighted_density_inputs(masses, weights):
    masses = np.asarray(masses, dtype=float)
    if weights is None:
        weights = np.ones(len(masses), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    if masses.ndim != 1 or weights.ndim != 1 or len(masses) != len(weights):
        raise ValueError("BH density masses and volume weights must be one-dimensional arrays with equal lengths.")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("BH density masses must be finite and positive after population selection.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("BH density volume weights must be finite and non-negative.")
    return masses, weights


def _bhmf_density(masses, weights=None):
    masses, weights = _weighted_density_inputs(masses, weights)
    counts, _ = np.histogram(masses, bins=FIG09_BIN_EDGES, weights=weights)
    return counts.astype(float) / FIG09_BHMF_VOLUME_CMPC3


def _bhmf_density_per_dex(masses, weights=None):
    masses, weights = _weighted_density_inputs(masses, weights)
    counts, _ = np.histogram(masses, bins=10.0 ** FIG06_BIN_EDGES, weights=weights)
    bin_width_dex = np.diff(FIG06_BIN_EDGES)
    return counts.astype(float) / (FIG09_BHMF_VOLUME_CMPC3 * bin_width_dex)


def _fig06_project_densities(summary_by_z):
    required = ["z_out", "M_SMBH_final", "volume_weight_tng50"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"Fig. 06 haloSummaryByZ is missing required columns: {missing}")
    z_out = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(z_out)):
        raise ValueError("Fig. 06 requires finite z_out values in haloSummaryByZ.")
    retained = z_out > FIG06_MIN_MODEL_REDSHIFT_EXCLUSIVE
    z_values = np.sort(np.unique(z_out[retained]))
    if len(z_values) == 0:
        raise ValueError("Fig. 06 has no model output redshifts strictly above z=3.")

    this_work_by_z = {}
    positive_raw_by_z = {}
    positive_effective_by_z = {}
    invalid_mass_by_z = {}
    out_of_range_raw_by_z = {}
    out_of_range_effective_by_z = {}
    omitted_no_positive_redshifts = []
    omitted_no_visible_density_redshifts = []
    lower_log_mass = float(FIG06_BIN_EDGES[0])
    upper_log_mass = float(FIG06_BIN_EDGES[-1])
    for z_out_value in z_values:
        rows = summary_by_z.loc[z_out == float(z_out_value)]
        masses_all = pd.to_numeric(rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
        weights_all = pd.to_numeric(rows["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(weights_all)) or np.any(weights_all <= 0.0):
            raise ValueError(f"Fig. 06 volume weights must be finite and positive at z={float(z_out_value):.6g}.")
        positive = np.isfinite(masses_all) & (masses_all > 0.0)
        masses = masses_all[positive]
        weights = weights_all[positive]
        positive_raw_by_z[float(z_out_value)] = int(len(masses))
        positive_effective_by_z[float(z_out_value)] = float(np.sum(weights))
        invalid_mass_by_z[float(z_out_value)] = int(np.count_nonzero(~positive))
        if len(masses) == 0:
            omitted_no_positive_redshifts.append(float(z_out_value))
            continue

        log_masses = np.log10(masses)
        in_range = (log_masses >= lower_log_mass) & (log_masses <= upper_log_mass)
        out_of_range = ~in_range
        out_of_range_raw_by_z[float(z_out_value)] = int(np.count_nonzero(out_of_range))
        out_of_range_effective_by_z[float(z_out_value)] = float(np.sum(weights[out_of_range]))
        if not np.any(in_range):
            omitted_no_visible_density_redshifts.append(float(z_out_value))
            continue

        density = _bhmf_density_per_dex(masses[in_range], weights[in_range])
        if not np.any(density > 0.0):
            omitted_no_visible_density_redshifts.append(float(z_out_value))
            continue
        this_work_by_z[float(z_out_value)] = density

    inventory = {
        "this_work_by_z": positive_raw_by_z,
        "this_work_effective_by_z": positive_effective_by_z,
        "invalid_mass_by_z": invalid_mass_by_z,
        "out_of_range_raw_by_z": out_of_range_raw_by_z,
        "out_of_range_effective_by_z": out_of_range_effective_by_z,
        "omitted_no_positive_redshifts": np.asarray(omitted_no_positive_redshifts, dtype=float),
        "omitted_no_visible_density_redshifts": np.asarray(omitted_no_visible_density_redshifts, dtype=float),
    }
    return 0.5 * (FIG06_BIN_EDGES[:-1] + FIG06_BIN_EDGES[1:]), this_work_by_z, inventory


def _fig12_project_bhseed_events(final_gc, volume_cmpc3):
    """Return every final-GC row with a positive initial IMBH seed mass."""

    if not isinstance(final_gc, pd.DataFrame):
        raise ValueError("Fig. 12 seed-event projection requires finalGCs.dat as a pandas DataFrame.")
    required = ["M_IMBH_init", "lookback_time_init_gyr", "volume_weight_tng50", "halo_id_z0", "status"]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"Fig. 12 finalGCs.dat is missing required columns: {missing}")
    if len(final_gc) == 0:
        raise ValueError("Fig. 12 cannot project an empty finalGCs.dat catalogue.")

    initial_mass = pd.to_numeric(final_gc["M_IMBH_init"], errors="coerce").to_numpy(dtype=float)
    lookback_init = pd.to_numeric(final_gc["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(final_gc["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)
    status_raw = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=float)
    halo_id_raw = pd.to_numeric(final_gc["halo_id_z0"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(initial_mass)) or np.any(initial_mass < 0.0):
        raise ValueError("Fig. 12 M_IMBH_init values must be finite and non-negative.")
    if np.any(~np.isfinite(lookback_init)) or np.any(lookback_init < 0.0):
        raise ValueError("Fig. 12 initial lookback times must be finite and non-negative.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Fig. 12 inherited volume weights must be finite and strictly positive.")
    if np.any(~np.isfinite(status_raw)) or np.any(np.abs(status_raw - np.rint(status_raw)) > 1.0e-8):
        raise ValueError("Fig. 12 finalGCs.dat statuses must be finite integers.")
    if np.any(~np.isfinite(halo_id_raw)) or np.any(np.abs(halo_id_raw - np.rint(halo_id_raw)) > 1.0e-8):
        raise ValueError("Fig. 12 finalGCs.dat parent halo IDs must be finite integers.")
    status = np.rint(status_raw).astype(np.int64)
    halo_id = np.rint(halo_id_raw).astype(np.int64)

    volume_cmpc3 = float(volume_cmpc3)
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError("Fig. 12 reference volume must be finite and strictly positive.")
    t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    formation_time = t0 - lookback_init
    if np.any(~np.isfinite(formation_time)) or np.any(formation_time <= 0.0) or np.any(formation_time > t0):
        raise ValueError("Fig. 12 contains a cosmic formation time outside 0 < t_form <= t_0.")
    formation_redshift = np.asarray(
        [CosmicAge2Redshift(float(value), time_unit="Gyr") for value in formation_time],
        dtype=float,
    )
    if np.any(~np.isfinite(formation_redshift)) or np.any(formation_redshift < 0.0):
        raise ValueError("Fig. 12 formation-time conversion produced an invalid formation redshift.")
    formation_x = np.log10(1.0 + formation_redshift)
    if np.any(~np.isfinite(formation_x)) or np.any(formation_x < 0.0):
        raise ValueError("Fig. 12 formation redshift conversion produced an invalid log10(1+z).")

    positive = initial_mass > 0.0
    if not np.any(positive):
        raise ValueError("Fig. 12 found no positive M_IMBH_init seed masses.")
    positive_indices = np.flatnonzero(positive)
    positive_status = status[positive]
    positive_weights = weights[positive]
    positive_x = formation_x[positive]
    positive_redshift = formation_redshift[positive]
    status_raw_counts = {
        int(code): int(np.count_nonzero(positive_status == code))
        for code in np.unique(positive_status)
    }
    status_effective_counts = {
        int(code): float(np.sum(positive_weights[positive_status == code]))
        for code in np.unique(positive_status)
    }
    display_mask = (positive_x >= FIG12_XLIM_LOG1PZ[0]) & (positive_x <= FIG12_XLIM_LOG1PZ[1])
    outside_low_mask = positive_x < FIG12_XLIM_LOG1PZ[0]
    outside_high_mask = positive_x > FIG12_XLIM_LOG1PZ[1]
    if np.any(display_mask & (outside_low_mask | outside_high_mask)):
        raise ValueError("Fig. 12 display-window event masks overlap unexpectedly.")
    return {
        "positive_indices": positive_indices,
        "M_IMBH_init": initial_mass[positive],
        "formation_time_gyr": formation_time[positive],
        "formation_redshift": positive_redshift,
        "formation_x": positive_x,
        "weights": positive_weights,
        "status": positive_status,
        "halo_id_z0": halo_id[positive],
        "volume_cmpc3": volume_cmpc3,
        "raw_positive_count": int(np.count_nonzero(positive)),
        "effective_positive_count": float(np.sum(positive_weights)),
        "status_raw_counts": status_raw_counts,
        "status_effective_counts": status_effective_counts,
        "display_mask": display_mask,
        "outside_low_mask": outside_low_mask,
        "outside_high_mask": outside_high_mask,
        "display_raw_count": int(np.count_nonzero(display_mask)),
        "display_effective_count": float(np.sum(positive_weights[display_mask])),
        "outside_low_raw_count": int(np.count_nonzero(outside_low_mask)),
        "outside_low_effective_count": float(np.sum(positive_weights[outside_low_mask])),
        "outside_high_raw_count": int(np.count_nonzero(outside_high_mask)),
        "outside_high_effective_count": float(np.sum(positive_weights[outside_high_mask])),
    }


def _fig10_project_seed_densities(summary_by_z, final_gc, volume_cmpc3):
    """Project weighted, cumulative initial IMBH seed mass functions."""

    if "z_out" in summary_by_z.columns:
        z_column = "z_out"
    elif "redshift" in summary_by_z.columns:
        z_column = "redshift"
    else:
        raise ValueError("Fig. 10 requires z_out or redshift in haloSummaryByZ.")
    summary_redshift = pd.to_numeric(summary_by_z[z_column], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(summary_redshift)) or np.any(summary_redshift < 0.0):
        raise ValueError("Fig. 10 requires finite non-negative output redshifts in haloSummaryByZ.")
    redshift_values = np.sort(np.unique(summary_redshift))
    if len(redshift_values) == 0:
        raise ValueError("Fig. 10 requires at least one output redshift in haloSummaryByZ.")

    required = [
        "M_IMBH_init", "lookback_time_init_gyr", "lookback_time_final_gyr",
        "volume_weight_tng50", "halo_id_z0", "status",
    ]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"Fig. 10 finalGCs.dat is missing required columns: {missing}")
    if len(final_gc) == 0:
        raise ValueError("Fig. 10 cannot project an empty finalGCs.dat catalogue.")

    initial_mass = pd.to_numeric(final_gc["M_IMBH_init"], errors="coerce").to_numpy(dtype=float)
    lookback_init = pd.to_numeric(final_gc["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    lookback_final = pd.to_numeric(final_gc["lookback_time_final_gyr"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(final_gc["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)
    status_raw = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=float)
    halo_id_raw = pd.to_numeric(final_gc["halo_id_z0"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(initial_mass)) or np.any(initial_mass < 0.0):
        raise ValueError("Fig. 10 initial IMBH seed masses must be finite and non-negative.")
    if np.any(~np.isfinite(lookback_init)) or np.any(lookback_init < 0.0):
        raise ValueError("Fig. 10 initial lookback times must be finite and non-negative.")
    if np.any(~np.isfinite(lookback_final)) or np.any(lookback_final < 0.0):
        raise ValueError("Fig. 10 final lookback times must be finite and non-negative.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Fig. 10 inherited volume weights must be finite and strictly positive.")
    if np.any(~np.isfinite(status_raw)) or np.any(np.abs(status_raw - np.rint(status_raw)) > 1.0e-8):
        raise ValueError("Fig. 10 finalGCs.dat statuses must be finite integers.")
    status = np.rint(status_raw).astype(np.int64)
    allowed_statuses = np.asarray([STATUS_EXHAUSTED, STATUS_TORN, STATUS_SUNK_GC, STATUS_WANDERER, STATUS_ALIVE, STATUS_SUNK_WANDERER], dtype=int)
    if not np.all(np.isin(status, allowed_statuses)):
        raise ValueError(f"Fig. 10 found unsupported finalGC status codes: {sorted(set(status.tolist()) - set(allowed_statuses.tolist()))}")
    if np.any(~np.isfinite(halo_id_raw)) or np.any(np.abs(halo_id_raw - np.rint(halo_id_raw)) > 1.0e-8):
        raise ValueError("Fig. 10 finalGCs.dat parent halo IDs must be finite integers.")

    volume_cmpc3 = float(volume_cmpc3)
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError("Fig. 10 reference volume must be finite and strictly positive.")
    t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    # finalGCs.dat writes lookback times at 1e-10 Gyr precision; this small
    # comparison tolerance also absorbs the round-off of an event at formation.
    time_tolerance = 1.0e-8
    formation_time = t0 - lookback_init
    if np.any(~np.isfinite(formation_time)) or np.any(formation_time <= 0.0) or np.any(formation_time > t0 + time_tolerance):
        raise ValueError("Fig. 10 contains an invalid cosmic formation time after lookback conversion.")
    formation_redshift = np.asarray([CosmicAge2Redshift(float(value), time_unit="Gyr") for value in formation_time], dtype=float)
    if np.any(~np.isfinite(formation_redshift)) or np.any(formation_redshift < 0.0):
        raise ValueError("Fig. 10 formation-time conversion produced an invalid formation redshift.")

    sink_status = np.isin(status, np.asarray([STATUS_SUNK_GC, STATUS_SUNK_WANDERER], dtype=int))
    sink_time = np.full(len(final_gc), np.nan, dtype=float)
    sink_time[sink_status] = t0 - lookback_final[sink_status]
    if np.any(sink_status & (~np.isfinite(sink_time) | (sink_time < 0.0) | (sink_time > t0 + time_tolerance))):
        raise ValueError("Fig. 10 central-sink event times are outside the cosmic-time interval.")
    if np.any(sink_status & (sink_time < formation_time - time_tolerance)):
        raise ValueError("Fig. 10 contains a central-sink event earlier than seed formation.")

    positive = initial_mass > 0.0
    if not np.any(positive):
        raise ValueError("Fig. 10 found no positive M_IMBH_init seed masses.")
    log_mass = np.full(len(final_gc), np.nan, dtype=float)
    log_mass[positive] = np.log10(initial_mass[positive])
    if np.any(~np.isfinite(log_mass[positive])):
        raise ValueError("Fig. 10 positive initial IMBH masses have non-finite logarithms.")
    if np.any(log_mass[positive] < FIG10_SEED_LOGM_BIN_EDGES[0] - 1.0e-12) or np.any(log_mass[positive] > FIG10_SEED_LOGM_BIN_EDGES[-1] + 1.0e-12):
        raise ValueError("Fig. 10 positive seed masses lie outside the fixed 0--5 dex mass grid.")

    bin_edges = np.asarray(FIG10_SEED_LOGM_BIN_EDGES, dtype=float)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = np.diff(bin_edges)
    densities = {group: [] for group in ("total", "nuclear", "satellite")}
    raw_counts = {group: [] for group in ("total", "nuclear", "satellite")}
    effective_counts = {group: [] for group in ("total", "nuclear", "satellite")}
    plotted_redshifts = []
    omitted_redshifts = []
    missing_groups = {}
    identity_tolerance = 1.0e-14

    for redshift in redshift_values:
        cosmic_age = float(Redshift2CosmicAge(float(redshift), time_unit="Gyr"))
        formed = positive & (formation_time <= cosmic_age + time_tolerance)
        if not np.any(formed):
            omitted_redshifts.append(float(redshift))
            continue
        nuclear = formed & sink_status & (sink_time <= cosmic_age + time_tolerance)
        satellite = formed & ~nuclear
        groups = {"total": formed, "nuclear": nuclear, "satellite": satellite}
        if not np.array_equal(formed, nuclear | satellite) or np.any(nuclear & satellite):
            raise ValueError(f"Fig. 10 group partition failed at z={float(redshift):.6g}.")

        density_by_group = {}
        for group, mask in groups.items():
            selected_log_mass = log_mass[mask]
            selected_weights = weights[mask]
            weighted_counts, _ = np.histogram(selected_log_mass, bins=bin_edges, weights=selected_weights)
            integer_counts, _ = np.histogram(selected_log_mass, bins=bin_edges)
            if int(np.sum(integer_counts)) != int(np.count_nonzero(mask)):
                raise ValueError(f"Fig. 10 histogram dropped selected {group} seeds at z={float(redshift):.6g}.")
            density_by_group[group] = weighted_counts.astype(float) / (volume_cmpc3 * bin_width)
            densities[group].append(density_by_group[group])
            raw_counts[group].append(int(np.count_nonzero(mask)))
            effective_counts[group].append(float(np.sum(selected_weights)))
            if not np.any(mask):
                missing_groups.setdefault(float(redshift), []).append(group)
        if not np.allclose(density_by_group["total"], density_by_group["nuclear"] + density_by_group["satellite"], rtol=1.0e-10, atol=identity_tolerance):
            raise ValueError(f"Fig. 10 total=nuclear+satellite identity failed at z={float(redshift):.6g}.")
        plotted_redshifts.append(float(redshift))

    if not plotted_redshifts:
        raise ValueError("Fig. 10 has no output redshift with a formed positive seed.")
    effective_total = np.asarray(effective_counts["total"], dtype=float)
    effective_tolerance = 1.0e-10 * max(1.0, float(np.max(effective_total)))
    if np.any(np.diff(effective_total) > effective_tolerance):
        raise ValueError("Fig. 10 total effective seed inventory is not non-increasing with output redshift.")
    densities = {group: np.asarray(values, dtype=float) for group, values in densities.items()}
    raw_counts = {group: np.asarray(values, dtype=int) for group, values in raw_counts.items()}
    effective_counts = {group: np.asarray(values, dtype=float) for group, values in effective_counts.items()}
    return {
        "redshifts": np.asarray(plotted_redshifts, dtype=float),
        "omitted_redshifts": np.asarray(omitted_redshifts, dtype=float),
        "missing_groups": {float(z): tuple(groups) for z, groups in missing_groups.items()},
        "log10_mass_bin_edges": bin_edges,
        "log10_mass_bin_centres": bin_centres,
        "mass_bin_centres_msun": np.power(10.0, bin_centres),
        "densities": densities,
        "raw_counts": raw_counts,
        "effective_counts": effective_counts,
        "formation_redshift": formation_redshift[positive],
        "volume_cmpc3": volume_cmpc3,
    }


def _fig10_project_central_bh_densities(summary_by_z, volume_cmpc3, mass_column="M_SMBH_init"):
    """Project one weighted central M_SMBH_init state per halo and snapshot."""

    if not isinstance(summary_by_z, pd.DataFrame):
        raise ValueError("Fig. 10 central-state projection requires haloSummaryByZ as a pandas DataFrame.")
    mass_aliases = {
        "M_SMBH_init": "central_bh_mass_init_msun",
        "central_bh_mass_init_msun": "M_SMBH_init",
    }
    source_mass_column = mass_column if mass_column in summary_by_z.columns else mass_aliases.get(mass_column)
    if source_mass_column not in summary_by_z.columns:
        raise ValueError(f"haloSummaryByZ is missing the requested central mass column {mass_column!r}.")
    if "redshift" in summary_by_z.columns:
        redshift_column = "redshift"
    elif "z_out" in summary_by_z.columns:
        redshift_column = "z_out"
    else:
        raise ValueError("Fig. 10 central-state projection requires redshift or z_out in haloSummaryByZ.")
    required = ["halo_id_z0", redshift_column, source_mass_column, "volume_weight_tng50"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing Fig. 10 central-state columns: {missing}")
    if len(summary_by_z) == 0:
        raise ValueError("Fig. 10 central-state projection cannot use an empty haloSummaryByZ.")

    table = summary_by_z.loc[:, required].copy()
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id_raw = table["halo_id_z0"].to_numpy(dtype=float)
    redshift = table[redshift_column].to_numpy(dtype=float)
    central_mass = table[source_mass_column].to_numpy(dtype=float)
    weights = table["volume_weight_tng50"].to_numpy(dtype=float)
    if np.any(~np.isfinite(halo_id_raw)) or np.any(np.abs(halo_id_raw - np.rint(halo_id_raw)) > 1.0e-8):
        raise ValueError("Fig. 10 central states contain non-finite or non-integer halo IDs.")
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 10 central states require finite non-negative output redshifts.")
    if np.any(~np.isfinite(central_mass)) or np.any(central_mass < 0.0):
        raise ValueError("Fig. 10 central M_SMBH_init states must be finite and non-negative.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Fig. 10 central-state inherited volume weights must be finite and strictly positive.")
    volume_cmpc3 = float(volume_cmpc3)
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError("Fig. 10 central-state reference volume must be finite and strictly positive.")

    halo_id = np.rint(halo_id_raw).astype(np.int64)
    key_table = pd.DataFrame({"halo_id_z0": halo_id, "redshift": redshift})
    if key_table.duplicated().any():
        duplicate_keys = key_table.loc[key_table.duplicated(keep=False)].drop_duplicates().to_dict("records")
        raise ValueError(f"Fig. 10 central projection requires one state per (halo_id_z0, redshift); duplicates={duplicate_keys[:10]}.")

    bin_edges = np.asarray(FIG10_CENTRAL_LOGM_BIN_EDGES, dtype=float)
    if (
        len(bin_edges) != 81
        or not np.isclose(bin_edges[0], 0.0, rtol=0.0, atol=1.0e-12)
        or not np.isclose(bin_edges[-1], 8.0, rtol=0.0, atol=1.0e-12)
        or np.any(~np.isfinite(bin_edges))
        or np.any(np.diff(bin_edges) <= 0.0)
        or not np.allclose(np.diff(bin_edges), 0.1, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError("Fig. 10 central-state mass grid is not the validated 0.0--8.0 dex sequence.")
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = np.diff(bin_edges)
    positive = central_mass > 0.0
    if not np.any(positive):
        raise ValueError("Fig. 10 found no positive M_SMBH_init central states.")
    positive_log_mass = np.log10(central_mass[positive])
    if np.any(~np.isfinite(positive_log_mass)):
        raise ValueError("Fig. 10 positive central M_SMBH_init states have non-finite logarithms.")
    if np.any(positive_log_mass < bin_edges[0] - 1.0e-12) or np.any(positive_log_mass > bin_edges[-1] + 1.0e-12):
        offending = central_mass[positive][(positive_log_mass < bin_edges[0] - 1.0e-12) | (positive_log_mass > bin_edges[-1] + 1.0e-12)]
        raise ValueError(f"Fig. 10 positive central M_SMBH_init states lie outside the fixed 0--8 dex mass grid: {offending[:10].tolist()}")

    redshift_values = np.sort(np.unique(redshift))
    densities = []
    raw_counts = []
    effective_counts = []
    positive_raw_counts = []
    positive_effective_counts = []
    zero_raw_counts = []
    zero_effective_counts = []
    identity_tolerance = 1.0e-10
    for redshift_value in redshift_values:
        snapshot = redshift == float(redshift_value)
        positive_snapshot = snapshot & positive
        zero_snapshot = snapshot & (central_mass == 0.0)
        snapshot_log_mass = np.log10(central_mass[positive_snapshot]) if np.any(positive_snapshot) else np.asarray([], dtype=float)
        snapshot_weights = weights[positive_snapshot]
        snapshot_raw, _ = np.histogram(snapshot_log_mass, bins=bin_edges)
        snapshot_effective, _ = np.histogram(snapshot_log_mass, bins=bin_edges, weights=snapshot_weights)
        if int(np.sum(snapshot_raw)) != int(np.count_nonzero(positive_snapshot)):
            raise ValueError(f"Fig. 10 central histogram dropped positive states at z={float(redshift_value):.6g}.")
        positive_effective = float(np.sum(snapshot_weights))
        if not np.isclose(float(np.sum(snapshot_effective)), positive_effective, rtol=identity_tolerance, atol=identity_tolerance):
            raise ValueError(f"Fig. 10 central histogram weights failed at z={float(redshift_value):.6g}.")
        densities.append(snapshot_effective.astype(float) / (volume_cmpc3 * bin_width))
        raw_counts.append(snapshot_raw.astype(int, copy=False))
        effective_counts.append(snapshot_effective.astype(float, copy=False))
        positive_raw_counts.append(int(np.count_nonzero(positive_snapshot)))
        positive_effective_counts.append(positive_effective)
        zero_raw_counts.append(int(np.count_nonzero(zero_snapshot)))
        zero_effective_counts.append(float(np.sum(weights[zero_snapshot])))

    positive_raw_counts = np.asarray(positive_raw_counts, dtype=int)
    positive_effective_counts = np.asarray(positive_effective_counts, dtype=float)
    zero_raw_counts = np.asarray(zero_raw_counts, dtype=int)
    zero_effective_counts = np.asarray(zero_effective_counts, dtype=float)
    densities = np.asarray(densities, dtype=float)
    raw_counts = np.asarray(raw_counts, dtype=int)
    effective_counts = np.asarray(effective_counts, dtype=float)
    if int(np.sum(raw_counts)) != int(np.sum(positive_raw_counts)):
        raise ValueError("Fig. 10 central per-bin raw counts do not equal the positive-state inventory.")
    if not np.isclose(float(np.sum(effective_counts)), float(np.sum(positive_effective_counts)), rtol=identity_tolerance, atol=identity_tolerance):
        raise ValueError("Fig. 10 central per-bin effective counts do not equal the positive-state inventory.")
    positive_redshift_indices = np.flatnonzero(positive_raw_counts > 0)
    if len(positive_redshift_indices) == 0:
        raise ValueError("Fig. 10 found no output redshift with a positive M_SMBH_init central state.")
    return {
        "redshifts": redshift_values.astype(float),
        "plotted_redshifts": redshift_values[positive_redshift_indices].astype(float),
        "positive_redshift_indices": positive_redshift_indices.astype(int),
        "omitted_redshifts": redshift_values[positive_raw_counts == 0].astype(float),
        "log10_mass_bin_edges": bin_edges,
        "log10_mass_bin_centres": bin_centres,
        "mass_bin_centres_msun": np.power(10.0, bin_centres),
        "densities": densities,
        "raw_counts": raw_counts,
        "effective_counts": effective_counts,
        "positive_raw_counts": positive_raw_counts,
        "positive_effective_counts": positive_effective_counts,
        "zero_raw_counts": zero_raw_counts,
        "zero_effective_counts": zero_effective_counts,
        "volume_cmpc3": volume_cmpc3,
        "mass_column": str(mass_column),
    }


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
    nuclear_by_z, inventory, effective_inventory = {}, {}, {}
    for z_out in z_values:
        rows = summary_by_z[np.isclose(summary_by_z["z_out"].to_numpy(dtype=float), float(z_out), rtol=0.0, atol=1.0e-8)]
        masses = pd.to_numeric(rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
        weights = pd.to_numeric(rows["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)
        positive = np.isfinite(masses) & (masses > 0.0)
        masses = masses[positive]
        weights = weights[positive]
        inventory[float(z_out)] = int(len(masses))
        effective_inventory[float(z_out)] = float(np.sum(weights))
        if len(masses) > 0:
            nuclear_by_z[float(z_out)] = _bhmf_density(masses, weights)
    if not nuclear_by_z:
        raise ValueError("No positive nuclear BH masses are available for Fig. 03.")

    status = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=int)
    imbh_mass = pd.to_numeric(final_gc["M_IMBH_final"], errors="coerce").to_numpy(dtype=float)
    bad = np.isin(status, np.asarray([STATUS_EXHAUSTED, STATUS_TORN], dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)
    if np.any(bad):
        raise ValueError(f"Fig. 03 found positive IMBH masses in exhausted/torn statuses: {sorted(set(status[bad].tolist()))}")
    satellite_mask = np.isin(status, np.asarray(SATELLITE_BH_STATUSES, dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)
    satellite_mass = imbh_mass[satellite_mask]
    satellite_weights = pd.to_numeric(final_gc["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)[satellite_mask]
    satellite_density = _bhmf_density(satellite_mass, satellite_weights)

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
    ax.text(0.03, 0.04, r"TNG100 full-box objects weighted by $V_{\rm TNG50}/V_{\rm TNG100}$", transform=ax.transAxes, fontsize=7.0, color="0.25")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig, {"nuclear_by_z": inventory, "nuclear_effective_by_z": effective_inventory, "satellite": int(len(satellite_mass)), "satellite_effective": float(np.sum(satellite_weights))}


def plot_fig06_bhmf2(bhmf_data, summary_by_z):
    required = ["Phi [lgM☉⁻¹Mpc⁻³]", "sigma_Phi [lgM☉⁻¹Mpc⁻³]", "Mbh [M☉]", "sigma_Mbh", "colour", "face colour", "shape", "label"]
    missing = [name for name in required if name not in bhmf_data.columns]
    if missing:
        raise ValueError(f"Fig. 06 BHMF table is missing required columns: {missing}")
    x_project, this_work_by_z, inventory = _fig06_project_densities(summary_by_z)
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
        ax.plot(x_line, central["Phi [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float), alpha=0.3, c=colour, lw=1.0, label=base_label, zorder=3)

    z_values = np.asarray(sorted(this_work_by_z), dtype=float)
    if len(z_values) > 0:
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5) if len(z_values) == 1 else mpl.colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
        this_work_label_used = False
        for z_out, density in sorted(this_work_by_z.items()):
            ax.plot(x_project, _log10_plot_values(density), c=cmap(norm(float(z_out))), lw=1.5, alpha=0.9, label="This work" if not this_work_label_used else None, zorder=2)
            this_work_label_used = True
        colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
        colour_bar.set_label(r"Redshift $z$")
        if len(z_values) == 1:
            colour_bar.set_ticks([float(z_values[0])])
    ax.set_xlim(4.2, 8.9)
    ax.set_ylim(-6.2, 0.2)
    ax.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_{\odot})$")
    ax.set_ylabel(r"$\log \Phi\ [M_{\odot}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="lower left", fontsize=7.0, ncol=2)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig, inventory


def plot_fig10_bhsmf(chen_data, summary_by_z, volume_cmpc3=FIG09_BHMF_VOLUME_CMPC3):
    """Plot the Chen+2026 reference curves and the central This work curve."""

    if not isinstance(chen_data, dict) or "by_role" not in chen_data:
        raise ValueError("Fig. 10 requires the structured output of load_chen2026_fig05a_seed_mass_functions().")
    chen_curves = chen_data["by_role"]
    missing_roles = [role for role in FIG10_CHEN_CURVE_ROLES if role not in chen_curves]
    if missing_roles:
        raise ValueError(f"Fig. 10 Chen+2026 data is missing curve roles: {missing_roles}")
    central_projection = _fig10_project_central_bh_densities(summary_by_z, volume_cmpc3=volume_cmpc3, mass_column="M_SMBH_init")
    central_x = central_projection["mass_bin_centres_msun"]
    central_z_values = central_projection["plotted_redshifts"]
    colour_redshifts = central_z_values
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=float(colour_redshifts[0]) - 0.5, vmax=float(colour_redshifts[0]) + 0.5) if len(colour_redshifts) == 1 else mpl.colors.Normalize(vmin=float(colour_redshifts.min()), vmax=float(colour_redshifts.max()))

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 5.0))
    central = chen_curves["all_seeds_central"]
    lower = chen_curves["all_seeds_lower_envelope"]
    upper = chen_curves["all_seeds_upper_envelope"]
    x_chen = np.power(10.0, central["log10_mbh_seed_msun"].to_numpy(dtype=float))
    y_central = central["phi_mpc3_dex1"].to_numpy(dtype=float)
    y_lower = lower["phi_mpc3_dex1"].to_numpy(dtype=float)
    y_upper = upper["phi_mpc3_dex1"].to_numpy(dtype=float)
    ax.fill_between(x_chen, y_lower, y_upper, color="#9e9e9e", alpha=0.35, linewidth=0.0, zorder=1)
    ax.plot(x_chen, y_lower, c="#9e9e9e", lw=0.65, zorder=2)
    ax.plot(x_chen, y_upper, c="#9e9e9e", lw=0.65, zorder=2)
    ax.plot(x_chen, y_central, c="#000000", lw=1.6, zorder=3)
    for role in FIG10_CHEN_VISIBLE_CURVE_ROLES:
        curve = chen_curves[role]
        x_curve = np.power(10.0, curve["log10_mbh_seed_msun"].to_numpy(dtype=float))
        y_curve = curve["phi_mpc3_dex1"].to_numpy(dtype=float)
        ax.plot(x_curve, y_curve, c=str(curve["colour"].iloc[0]), ls=str(curve["linestyle"].iloc[0]), lw=1.25, zorder=3)

    for redshift, row_index in zip(central_z_values, central_projection["positive_redshift_indices"]):
        density = central_projection["densities"][int(row_index)]
        if not np.any(density > 0.0):
            continue
        ax.plot(central_x, _plot_values(density), c=cmap(norm(float(redshift))), ls="-", lw=1.8, alpha=0.95, zorder=4)

    positive_x = [x_chen]
    positive_y = [y_central, y_lower, y_upper]
    for role in FIG10_CHEN_VISIBLE_CURVE_ROLES:
        curve = chen_curves[role]
        positive_x.append(np.power(10.0, curve["log10_mbh_seed_msun"].to_numpy(dtype=float)))
        positive_y.append(curve["phi_mpc3_dex1"].to_numpy(dtype=float))
    for row_index in central_projection["positive_redshift_indices"]:
        central_density = central_projection["densities"][int(row_index)]
        positive_mask = central_density > 0.0
        if np.any(positive_mask):
            positive_x.append(central_x[positive_mask])
            positive_y.append(central_density[positive_mask])
    x_positive = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in positive_x])
    y_positive = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in positive_y])
    if len(x_positive) == 0 or len(y_positive) == 0:
        raise ValueError("Fig. 10 cannot determine finite positive plot limits.")
    ax.set_xlim(float(np.min(x_positive) / 1.35), float(np.max(x_positive) * 1.35))
    y_min = 1.0e-6
    y_max = float(np.max(y_positive) * 1.6)
    if not np.isfinite(y_max) or y_max <= y_min:
        raise ValueError(f"Fig. 10 visible curves do not extend above the requested y-axis floor {y_min:.1e}.")
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Seed mass $M_{\rm BH,seed}\ [M_{\odot}]$")
    ax.set_ylabel(r"$\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.text(0.03, 0.97, "Central initial seed states\nTNG100 objects use approved parent-halo volume weights\nThis work: one central $M_{\\rm SMBH,init}$ state per halo per output snapshot\n(not an additional $M_{\\rm IMBH,init}$ event)", transform=ax.transAxes, fontsize=6.2, color="0.25", va="top")

    source_handles = [
        Line2D([], [], color="#000000", lw=1.6, label="All seeds"),
        Patch(facecolor="#9e9e9e", edgecolor="none", alpha=0.35, label="All seeds envelope"),
    ]
    for role in FIG10_CHEN_VISIBLE_CURVE_ROLES:
        curve = chen_curves[role]
        source_handles.append(Line2D([], [], color=str(curve["colour"].iloc[0]), ls=str(curve["linestyle"].iloc[0]), lw=1.25, label=str(curve["curve_label"].iloc[0])))
    source_legend = ax.legend(handles=source_handles, title="Chen+2026", frameon=False, fontsize=6.7, title_fontsize=7.4, loc="upper right", ncol=1, borderaxespad=0.3)
    ax.add_artist(source_legend)
    ax.legend(handles=[Line2D([], [], color="black", ls="-", lw=1.8, label="This work")], frameon=False, fontsize=7.0, loc="lower left", ncol=1, borderaxespad=0.3)

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(colour_redshifts) == 1:
        colour_bar.set_ticks([float(colour_redshifts[0])])
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig, {"central": central_projection, "colour_redshifts": colour_redshifts}


def plot_fig12_bhseed_history(chen_data, final_gc, volume_cmpc3=FIG09_BHMF_VOLUME_CMPC3):
    """Plot the unpartitioned final-GC IMBH-seed formation history."""

    if not isinstance(chen_data, dict) or "by_panel" not in chen_data:
        raise ValueError("Fig. 12 requires the structured output of load_chen2026_fig06_seed_history().")
    for panel in ("a", "d"):
        if panel not in chen_data["by_panel"]:
            raise ValueError(f"Fig. 12 Chen+2026 data is missing panel {panel!r}.")
        missing_roles = [role for role in FIG12_CHEN_CURVE_ROLES if role not in chen_data["by_panel"][panel]]
        if missing_roles:
            raise ValueError(f"Fig. 12 Chen+2026 panel {panel} is missing curve roles: {missing_roles}")

    events = _fig12_project_bhseed_events(final_gc, volume_cmpc3)
    volume_cmpc3 = float(events["volume_cmpc3"])
    display_x = events["formation_x"][events["display_mask"]]
    display_weights = events["weights"][events["display_mask"]]
    if len(display_x) == 0:
        raise ValueError("Fig. 12 has no positive IMBH seeds in the requested 0 <= z <= 50 display range.")

    x_left, x_right = map(float, FIG12_XLIM_LOG1PZ)
    rate_edges = np.arange(x_left, x_right, FIG12_RATE_LOG1PZ_BIN_WIDTH, dtype=float)
    if len(rate_edges) == 0 or not np.isclose(rate_edges[0], x_left, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Fig. 12 rate grid failed to start at the requested lower x limit.")
    if rate_edges[-1] < x_right - 1.0e-12:
        rate_edges = np.append(rate_edges, x_right)
    else:
        rate_edges[-1] = x_right
    if np.any(np.diff(rate_edges) <= 0.0):
        raise RuntimeError("Fig. 12 rate grid is not strictly increasing.")
    rate_bin_widths = np.diff(rate_edges)
    rate_x = 0.5 * (rate_edges[:-1] + rate_edges[1:])
    weighted_counts, _ = np.histogram(display_x, bins=rate_edges, weights=display_weights)
    raw_counts, _ = np.histogram(display_x, bins=rate_edges)
    if int(np.sum(raw_counts)) != int(events["display_raw_count"]):
        raise ValueError("Fig. 12 rate histogram dropped positive seed events in the display range.")
    rate_density = weighted_counts.astype(float) / (volume_cmpc3 * rate_bin_widths)
    rate_integral = float(np.sum(rate_density * rate_bin_widths))
    expected_rate_integral = float(events["display_effective_count"] / volume_cmpc3)
    if not np.isclose(rate_integral, expected_rate_integral, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("Fig. 12 weighted rate integral does not equal the displayed effective seed density.")

    # The survival count uses all positive events before the display cut and the
    # strict convention z_form > z, implemented equivalently in x.
    order = np.argsort(events["formation_x"], kind="mergesort")
    sorted_x = events["formation_x"][order]
    sorted_weights = events["weights"][order]
    suffix_weights = np.cumsum(sorted_weights[::-1], dtype=float)[::-1]
    survival_index = np.searchsorted(sorted_x, rate_x, side="right")
    cumulative_density = np.zeros_like(rate_x, dtype=float)
    has_survivors = survival_index < len(sorted_x)
    cumulative_density[has_survivors] = suffix_weights[survival_index[has_survivors]] / volume_cmpc3
    if np.any(~np.isfinite(cumulative_density)) or np.any(cumulative_density < 0.0):
        raise ValueError("Fig. 12 cumulative seed density is not finite and non-negative.")
    if np.any(np.diff(cumulative_density) > 1.0e-12 * max(1.0, float(np.max(cumulative_density)))):
        raise ValueError("Fig. 12 cumulative seed density is not non-increasing with redshift.")

    reference_values = {"a": [], "d": []}
    for panel in ("a", "d"):
        for role in FIG12_CHEN_CURVE_ROLES:
            reference_values[panel].append(chen_data["by_panel"][panel][role]["value_mpc3"].to_numpy(dtype=float))
    top_values = np.concatenate(reference_values["a"] + [rate_density[rate_density > 0.0]])
    bottom_values = np.concatenate(reference_values["d"] + [cumulative_density[cumulative_density > 0.0]])
    if len(top_values) == 0 or len(bottom_values) == 0:
        raise ValueError("Fig. 12 cannot determine finite positive y-axis limits.")
    top_values = top_values[np.isfinite(top_values) & (top_values > 0.0)]
    bottom_values = bottom_values[np.isfinite(bottom_values) & (bottom_values > 0.0)]
    if len(top_values) == 0 or len(bottom_values) == 0:
        raise ValueError("Fig. 12 has no finite positive values for logarithmic axes.")
    top_y_min = 10.0 ** (math.floor(math.log10(float(np.min(top_values)))) - 0.25)
    top_y_max = 10.0 ** (math.ceil(math.log10(float(np.max(top_values)))) + 0.25)
    bottom_y_min = 10.0 ** (math.floor(math.log10(float(np.min(bottom_values)))) - 0.25)
    bottom_y_max = 10.0 ** (math.ceil(math.log10(float(np.max(bottom_values)))) + 0.25)

    fig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True, dpi=STD_DPI, figsize=FIG12_FIGSIZE)
    axes = np.asarray(axes).reshape(-1)
    for panel, ax, quantity, model_x, model_values in (
        ("a", axes[0], "rate", rate_x, rate_density),
        ("d", axes[1], "cumulative", rate_x, cumulative_density),
    ):
        for role in FIG12_CHEN_CURVE_ROLES:
            curve = chen_data["by_panel"][panel][role]
            x_curve = curve["x_log10_1pz"].to_numpy(dtype=float)
            y_curve = curve["value_mpc3"].to_numpy(dtype=float)
            # Extend only the panel-d plotting arrays with the native endpoint value.
            if panel == "d" and x_curve[0] > x_left:
                x_curve = np.concatenate(([x_left], x_curve))
                y_curve = np.concatenate(([y_curve[0]], y_curve))
            colour = str(curve["colour"].iloc[0])
            linestyle = str(curve["linestyle"].iloc[0])
            if role == "all_seeds":
                fill_floor = float(np.min(y_curve) * 0.75)
                ax.fill_between(x_curve, fill_floor, y_curve, color=colour, alpha=0.20, linewidth=0.0, zorder=1)
            elif role == "popii":
                fill_floor = float(np.min(y_curve) * 0.75)
                ax.fill_between(x_curve, fill_floor, y_curve, color=colour, alpha=0.50, linewidth=0.0, zorder=1)
            ax.plot(x_curve, y_curve, c=colour, ls=linestyle, lw=FIG12_REFERENCE_LINEWIDTH, zorder=3)
        model_plot_values = np.where(model_values > 0.0, model_values, np.nan)
        ax.plot(
            model_x,
            model_plot_values,
            c=FIG12_MODEL_COLOUR,
            lw=FIG12_MODEL_LINEWIDTH,
            drawstyle="steps-mid" if panel == "a" else "default",
            label="_nolegend_",
            zorder=5,
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.tick_params(direction="in", right=True, top=True, which="both")
        ax.text(0.965, 0.965, panel, transform=ax.transAxes, ha="right", va="top", fontsize=10.0, fontweight="bold")

    axes[0].set_ylim(top_y_min, top_y_max)
    axes[1].set_ylim(bottom_y_min, bottom_y_max)
    axes[0].set_ylabel(r"$d n_{\rm seed}/d\log_{10}(1+z)\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    axes[1].set_ylabel(r"$n_{\rm seed}(>z)\ [\mathrm{Mpc}^{-3}]$")
    axes[1].set_xlabel(r"$\log_{10}(1+z)$")
    axes[0].set_xlim(x_left, x_right)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticks(np.arange(0.0, 1.61, 0.2))

    source_handles = {}
    for role in FIG12_CHEN_CURVE_ROLES:
        curve = chen_data["by_panel"]["a"][role]
        source_handles[role] = Line2D(
            [], [], color=str(curve["colour"].iloc[0]), ls=str(curve["linestyle"].iloc[0]),
            lw=FIG12_REFERENCE_LINEWIDTH, label=FIG12_CHEN_CURVE_LABELS[role],
        )
    axes[0].legend(
        handles=[source_handles[role] for role in FIG12_CHEN_CURVE_ROLES[:3]],
        title="Chen+2026", frameon=False, fontsize=6.9, title_fontsize=7.4,
        loc="best", ncol=1, borderaxespad=0.25,
    )
    axes[1].legend(
        handles=[source_handles["popii"], Line2D([], [], color=FIG12_MODEL_COLOUR, lw=FIG12_MODEL_LINEWIDTH, label="This work")],
        title="Chen+2026 / model", frameon=False, fontsize=6.9, title_fontsize=7.4,
        loc="best", ncol=1, borderaxespad=0.25,
    )

    secondary_axis = axes[0].secondary_xaxis(
        "top",
        functions=(lambda x: np.power(10.0, x) - 1.0, lambda z: np.log10(1.0 + z)),
    )
    secondary_axis.set_xticks([0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    secondary_axis.set_xticklabels(["0", "5", "10", "20", "30", "40", "50"])
    secondary_axis.set_xlabel(r"Redshift $z$")
    secondary_axis.tick_params(direction="in", which="both")

    return fig, {
        **events,
        "rate_x": rate_x,
        "rate_bin_edges": rate_edges,
        "rate_bin_widths": rate_bin_widths,
        "rate_density": rate_density,
        "cumulative_x": rate_x.copy(),
        "cumulative_density": cumulative_density,
        "rate_integral": rate_integral,
        "expected_rate_integral": expected_rate_integral,
        "rate_bin_width": float(FIG12_RATE_LOG1PZ_BIN_WIDTH),
        "cumulative_boundary": "strict z_form > z",
    }


def _select_abundance_matching_snapshot(summary_by_z, target_redshift):
    required = ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift", "central_bh_mass_final_msun", "volume_weight_tng50"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing abundance-matching columns: {missing}")
    target_redshift = float(target_redshift)
    if not np.isfinite(target_redshift) or target_redshift < 0.0:
        raise ValueError(f"Abundance-matching target redshift must be finite and non-negative, got {target_redshift!r}.")

    redshift = pd.to_numeric(summary_by_z["redshift"], errors="coerce").to_numpy(dtype=float)
    available_redshifts = np.unique(redshift[np.isfinite(redshift)])
    if len(available_redshifts) == 0:
        raise ValueError("haloSummaryByZ contains no finite redshifts for abundance matching.")
    selected_redshift = float(available_redshifts[np.argmin(np.abs(available_redshifts - target_redshift))])
    if abs(selected_redshift - target_redshift) >= ABUNDANCE_MATCHING_REDSHIFT_ATOL:
        raise ValueError(
            f"No output snapshot is within {ABUNDANCE_MATCHING_REDSHIFT_ATOL:.2f} of z={target_redshift:.3g}; "
            f"nearest available redshift is z={selected_redshift:.6g}."
        )

    rows = summary_by_z.loc[np.isclose(redshift, selected_redshift, rtol=0.0, atol=1.0e-8)].copy()
    for column in ["halo_id_z0", "halo_mass_available", "log10_halo_mass_at_redshift", "central_bh_mass_final_msun", "volume_weight_tng50"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    valid = (
        np.isfinite(rows["halo_id_z0"].to_numpy(dtype=float))
        & (rows["halo_mass_available"].to_numpy(dtype=float) == 1.0)
        & np.isfinite(rows["log10_halo_mass_at_redshift"].to_numpy(dtype=float))
        & np.isfinite(rows["central_bh_mass_final_msun"].to_numpy(dtype=float))
        & (rows["central_bh_mass_final_msun"].to_numpy(dtype=float) >= 0.0)
        & np.isfinite(rows["volume_weight_tng50"].to_numpy(dtype=float))
        & (rows["volume_weight_tng50"].to_numpy(dtype=float) > 0.0)
    )
    rows = rows.loc[valid].copy()
    if len(rows) == 0:
        raise ValueError(f"No valid halo--central-BH rows are available for abundance matching at z={selected_redshift:.6g}.")
    rows["halo_id_z0"] = rows["halo_id_z0"].astype(int)
    if rows["halo_id_z0"].duplicated().any():
        duplicate_ids = rows.loc[rows["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].tolist()
        raise ValueError(f"Abundance matching requires one row per halo at z={selected_redshift:.6g}; duplicates={duplicate_ids[:10]}.")
    rows["log10_halo_mass"] = rows["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    rows["central_bh_mass_msun"] = rows["central_bh_mass_final_msun"].to_numpy(dtype=float)
    rows["log10_central_bh_mass"] = np.where(
        rows["central_bh_mass_msun"].to_numpy(dtype=float) > 0.0,
        np.log10(np.maximum(rows["central_bh_mass_msun"].to_numpy(dtype=float), np.finfo(float).tiny)),
        np.nan,
    )
    return rows.sort_values("halo_id_z0").reset_index(drop=True), selected_redshift


def _cumulative_abundance_curve(log10_masses, volume_cmpc3, weights=None):
    values = np.asarray(log10_masses, dtype=float)
    if values.ndim != 1:
        raise ValueError("Cumulative-abundance masses must be one-dimensional.")
    if weights is None:
        weight_values = np.ones(len(values), dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float)
    if weight_values.ndim != 1 or len(weight_values) != len(values):
        raise ValueError("Cumulative-abundance masses and weights must have equal one-dimensional lengths.")
    if np.any(~np.isfinite(weight_values)) or np.any(weight_values < 0.0):
        raise ValueError("Cumulative-abundance weights must be finite and non-negative.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Cumulative-abundance masses must be finite after population selection.")
    if len(values) == 0:
        raise ValueError("Cannot construct a cumulative abundance curve from an empty mass sample.")
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError(f"Cumulative-abundance volume must be finite and positive, got {volume_cmpc3!r}.")
    order = np.argsort(values, kind="mergesort")
    ascending = values[order]
    ascending_weights = weight_values[order]
    count_above = np.cumsum(ascending_weights[::-1], dtype=float)[::-1]
    density = count_above / float(volume_cmpc3)
    if np.any(~np.isfinite(density)) or np.any(np.diff(density) > 1.0e-15):
        raise ValueError("Cumulative-abundance density is non-finite or increases with mass threshold.")
    return np.power(10.0, ascending), density


def build_mbh_mhalo_abundance_matching(summary_by_z, target_redshifts=ABUNDANCE_MATCHING_REDSHIFTS, volume_cmpc3=ABUNDANCE_MATCHING_VOLUME_CMPC3):
    target_redshifts = [float(value) for value in target_redshifts]
    if len(target_redshifts) == 0:
        raise ValueError("At least one target redshift is required for abundance matching.")
    if not np.isfinite(volume_cmpc3) or float(volume_cmpc3) <= 0.0:
        raise ValueError(f"Abundance-matching volume must be finite and positive, got {volume_cmpc3!r}.")

    table_rows = []
    snapshots = []
    selected_redshifts = []
    for target_redshift in target_redshifts:
        rows, selected_redshift = _select_abundance_matching_snapshot(summary_by_z, target_redshift)
        if any(np.isclose(selected_redshift, value, rtol=0.0, atol=1.0e-8) for value in selected_redshifts):
            raise ValueError(f"Target redshifts select the same output snapshot more than once: z={selected_redshift:.6g}.")
        selected_redshifts.append(selected_redshift)

        log10_halo_mass = rows["log10_halo_mass"].to_numpy(dtype=float)
        central_bh_mass = rows["central_bh_mass_msun"].to_numpy(dtype=float)
        log10_central_bh_mass = rows["log10_central_bh_mass"].to_numpy(dtype=float)
        volume_weights = rows["volume_weight_tng50"].to_numpy(dtype=float)
        halo_order = np.argsort(-log10_halo_mass, kind="mergesort")
        positive = central_bh_mass > 0.0
        positive_indices = np.flatnonzero(positive)
        if len(positive_indices) == 0:
            raise ValueError(f"No positive central BH masses are available for abundance matching at z={selected_redshift:.6g}.")
        positive_order = positive_indices[np.argsort(-log10_central_bh_mass[positive_indices], kind="mergesort")]
        n_halo = len(rows)
        n_positive_bh = len(positive_indices)
        if n_positive_bh > n_halo:
            raise ValueError(f"Positive central-BH count exceeds halo count at z={selected_redshift:.6g}.")

        matched_halo_indices = halo_order[:n_positive_bh]
        matched_bh_indices = positive_order[:n_positive_bh]
        matched_halo_log_mass = log10_halo_mass[matched_halo_indices]
        matched_bh_log_mass = log10_central_bh_mass[matched_bh_indices]
        direct_bh_at_matched_halo_log_mass = log10_central_bh_mass[matched_halo_indices]
        rank = np.arange(1, n_positive_bh + 1, dtype=int)
        cumulative_number_density = rank.astype(float) / float(volume_cmpc3)
        cumulative_fraction = rank.astype(float) / float(n_halo)
        halo_rank = np.empty(n_halo, dtype=int)
        halo_rank[halo_order] = np.arange(1, n_halo + 1, dtype=int)

        table_rows.append(
            pd.DataFrame(
                {
                    "target_redshift": float(target_redshift),
                    "redshift": float(selected_redshift),
                    "rank_desc": rank,
                    "halo_id_z0": rows.iloc[matched_halo_indices]["halo_id_z0"].to_numpy(dtype=int),
                    "halo_rank_desc": rank,
                    "log10_halo_mass_ranked": matched_halo_log_mass,
                    "log10_central_bh_mass_abundance_matched": matched_bh_log_mass,
                    "log10_central_bh_mass_model_at_ranked_halo": direct_bh_at_matched_halo_log_mass,
                    "model_bh_positive_at_ranked_halo": np.isfinite(direct_bh_at_matched_halo_log_mass),
                    "cumulative_fraction_of_halo_sample": cumulative_fraction,
                    "cumulative_number_density_cmpc3": cumulative_number_density,
                    "occupation_fraction": float(n_positive_bh) / float(n_halo),
                }
            )
        )
        halo_curve_mass, halo_curve_density = _cumulative_abundance_curve(log10_halo_mass, volume_cmpc3, volume_weights)
        bh_curve_mass, bh_curve_density = _cumulative_abundance_curve(log10_central_bh_mass[positive], volume_cmpc3, volume_weights[positive])
        snapshots.append(
            {
                "target_redshift": float(target_redshift),
                "redshift": float(selected_redshift),
                "n_halo": int(n_halo),
                "n_positive_bh": int(n_positive_bh),
                "n_halo_effective": float(np.sum(volume_weights)),
                "n_positive_bh_effective": float(np.sum(volume_weights[positive])),
                "occupation_fraction": float(n_positive_bh) / float(n_halo),
                "log10_halo_mass": log10_halo_mass,
                "log10_halo_mass_positive_bh": log10_halo_mass[positive],
                "log10_central_bh_mass": log10_central_bh_mass[positive],
                "matched_halo_log_mass": matched_halo_log_mass,
                "matched_bh_log_mass": matched_bh_log_mass,
                "matched_halo_threshold_log_mass": float(matched_halo_log_mass[-1]),
                "halo_curve_mass": halo_curve_mass,
                "halo_curve_density": halo_curve_density,
                "bh_curve_mass": bh_curve_mass,
                "bh_curve_density": bh_curve_density,
                "halo_rank": halo_rank,
            }
        )
    return pd.concat(table_rows, ignore_index=True), snapshots


def _binned_log_relation_percentiles(x_log, y_log, bin_width_dex):
    x_log = np.asarray(x_log, dtype=float)
    y_log = np.asarray(y_log, dtype=float)
    valid = np.isfinite(x_log) & np.isfinite(y_log)
    if not np.any(valid):
        return pd.DataFrame(columns=["logx_center", "median", "lower", "upper"])
    edges = _regular_log_bin_edges(x_log[valid], bin_width_dex)
    records = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = valid & (x_log >= left)
        mask &= x_log <= right if index == len(edges) - 2 else x_log < right
        if np.any(mask):
            records.append(
                {
                    "logx_center": 0.5 * (left + right),
                    "median": float(np.percentile(y_log[mask], 50.0)),
                    "lower": float(np.percentile(y_log[mask], 16.0)),
                    "upper": float(np.percentile(y_log[mask], 84.0)),
                }
            )
    return pd.DataFrame(records)


def _abundance_matching_colour_setup(snapshots):
    redshifts = np.asarray([float(snapshot["redshift"]) for snapshot in snapshots], dtype=float)
    if len(redshifts) == 1:
        norm = mpl.colors.Normalize(vmin=float(redshifts[0]) - 0.5, vmax=float(redshifts[0]) + 0.5)
    else:
        norm = mpl.colors.Normalize(vmin=float(redshifts.min()), vmax=float(redshifts.max()))
    return redshifts, norm, mpl.cm.viridis


def plot_fig07_mbh_mhalo_abundance_matching(snapshots):
    if len(snapshots) == 0:
        raise ValueError("Fig. 07 requires at least one abundance-matching snapshot.")
    redshifts, norm, cmap = _abundance_matching_colour_setup(snapshots)
    all_halo_log_mass = np.concatenate([snapshot["log10_halo_mass"] for snapshot in snapshots])
    all_bh_log_mass = np.concatenate([snapshot["log10_central_bh_mass"] for snapshot in snapshots])
    x_limits = (float(np.min(all_halo_log_mass)) - 0.15, float(np.max(all_halo_log_mass)) + 0.15)
    y_limits = (max(0.0, float(np.min(all_bh_log_mass)) - 0.35), float(np.max(all_bh_log_mass)) + 0.35)

    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=STD_DPI, figsize=(10.0, 4.4), sharex=True, sharey=True)
    direct_ax, matched_ax = axes
    for index, snapshot in enumerate(snapshots):
        colour = cmap(norm(float(snapshot["redshift"])))
        direct_ax.scatter(
            np.power(10.0, snapshot["log10_halo_mass_positive_bh"]),
            np.power(10.0, snapshot["log10_central_bh_mass"]),
            s=12.0,
            color=colour,
            alpha=0.24,
            edgecolors="none",
            label="Positive central BHs" if index == 0 else None,
            rasterized=True,
        )
        binned = _binned_log_relation_percentiles(snapshot["log10_halo_mass_positive_bh"], snapshot["log10_central_bh_mass"], ABUNDANCE_MATCHING_BIN_WIDTH_DEX)
        if len(binned) > 0:
            x_binned = np.power(10.0, binned["logx_center"].to_numpy(dtype=float))
            direct_ax.fill_between(
                x_binned,
                np.power(10.0, binned["lower"].to_numpy(dtype=float)),
                np.power(10.0, binned["upper"].to_numpy(dtype=float)),
                color=colour,
                alpha=0.12,
                linewidth=0.0,
            )
            direct_ax.plot(x_binned, np.power(10.0, binned["median"].to_numpy(dtype=float)), color=colour, lw=1.8)

        matched_x = np.power(10.0, snapshot["matched_halo_log_mass"][::-1])
        matched_y = np.power(10.0, snapshot["matched_bh_log_mass"][::-1])
        matched_ax.plot(
            matched_x,
            matched_y,
            color=colour,
            lw=2.0,
            label=rf"$z={float(snapshot['redshift']):.1f}$, $f_{{\mathrm{{occ}}}}={float(snapshot['occupation_fraction']):.2f}$",
        )
        matched_ax.axvline(
            np.power(10.0, snapshot["matched_halo_threshold_log_mass"]),
            color=colour,
            ls=":",
            lw=0.9,
            alpha=0.75,
        )

    direct_ax.plot([], [], color="0.35", lw=1.8, label="Median; shaded 16--84%")
    matched_ax.plot([], [], color="0.35", ls=":", lw=0.9, label="BH occupation threshold")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(np.power(10.0, x_limits[0]), np.power(10.0, x_limits[1]))
        ax.set_ylim(np.power(10.0, y_limits[0]), np.power(10.0, y_limits[1]))
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.tick_params(direction="in", right=True, top=True, which="both")
    direct_ax.set_title("Direct model pairing")
    matched_ax.set_title("Rank-ordered abundance matching")
    direct_ax.set_xlabel(r"Halo mass $M_{\mathrm{h}}(z)$ [$M_{\odot}$]")
    matched_ax.set_xlabel(r"Halo mass $M_{\mathrm{h}}(z)$ [$M_{\odot}$]")
    direct_ax.set_ylabel(r"Nuclear BH mass $M_{\bullet}$ [$M_{\odot}$]")
    direct_ax.legend(frameon=False, fontsize=7.3, loc="lower right")
    matched_ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, aspect=30, pad=0.01)
    colour_bar.set_label("Redshift z")
    if len(redshifts) == 1:
        colour_bar.set_ticks([float(redshifts[0])])
    return fig


def plot_fig08_mbh_mhalo_cumulative_abundance(snapshots):
    if len(snapshots) == 0:
        raise ValueError("Fig. 08 requires at least one abundance-matching snapshot.")
    redshifts, norm, cmap = _abundance_matching_colour_setup(snapshots)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    for index, snapshot in enumerate(snapshots):
        colour = cmap(norm(float(snapshot["redshift"])))
        ax.step(
            snapshot["halo_curve_mass"],
            snapshot["halo_curve_density"],
            where="post",
            color=colour,
            lw=1.8,
            label="Halo population" if index == 0 else None,
        )
        ax.step(
            snapshot["bh_curve_mass"],
            snapshot["bh_curve_density"],
            where="post",
            color=colour,
            ls="--",
            lw=1.6,
            label="Positive central-BH population" if index == 0 else None,
        )

    all_curve_mass = np.concatenate([np.concatenate([snapshot["halo_curve_mass"], snapshot["bh_curve_mass"]]) for snapshot in snapshots])
    all_curve_density = np.concatenate([np.concatenate([snapshot["halo_curve_density"], snapshot["bh_curve_density"]]) for snapshot in snapshots])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(all_curve_mass)) * 0.9, float(np.max(all_curve_mass)) * 1.1)
    ax.set_ylim(float(np.min(all_curve_density)) * 0.7, float(np.max(all_curve_density)) * 1.5)
    ax.set_xlabel(r"Mass threshold $M$ [$M_{\odot}$]")
    ax.set_ylabel(r"Cumulative abundance $n(>M)$ [cMpc$^{-3}$]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    ax.text(
        0.03,
        0.04,
        rf"Reference volume $V=({FIG09_BHMF_SIDE_CMPC:.2f}\,\mathrm{{cMpc}})^3$; parent-halo weights; TNG100: $V_{{\rm TNG50}}/V_{{\rm TNG100}}$",
        transform=ax.transAxes,
        fontsize=7.4,
        color="0.25",
    )
    ax.legend(frameon=False, fontsize=8.0, loc="lower left", bbox_to_anchor=(0.0, 0.10))
    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(redshifts) == 1:
        colour_bar.set_ticks([float(redshifts[0])])
    return fig


def plot_fig09_halo_distribution(summary_by_z, best):
    distribution = _build_fig09_halo_distribution(summary_by_z, best)
    redshifts = distribution["redshifts"]
    log_edges = distribution["log_bin_edges"]
    mass_edges = np.power(10.0, log_edges)
    if np.any(~np.isfinite(mass_edges)) or np.any(mass_edges <= 0.0):
        raise ValueError("Fig. 09 generated non-positive or non-finite linear halo-mass bin edges.")
    if len(redshifts) == 1:
        norm = mpl.colors.Normalize(vmin=float(redshifts[0]) - 0.5, vmax=float(redshifts[0]) + 0.5)
    else:
        norm = mpl.colors.Normalize(vmin=float(redshifts.min()), vmax=float(redshifts.max()))
    cmap = mpl.cm.jet

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    maximum_count = 0
    for item in distribution["distributions"]:
        z_value = float(item["redshift"])
        counts = np.asarray(item["counts"], dtype=int)
        if len(counts) != len(mass_edges) - 1 or np.any(counts < 0):
            raise ValueError(f"Fig. 09 contains invalid integer histogram counts at z={z_value:.6g}.")
        maximum_count = max(maximum_count, int(np.max(counts)))
        ax.stairs(counts, mass_edges, baseline=0.0, fill=False, color=cmap(norm(z_value)), lw=1.35, alpha=0.92, zorder=2)

    for track in distribution["best_halo_track"]:
        ax.axvline(float(track["halo_mass_msun"]), color=cmap(norm(float(track["redshift"]))), ls="--", lw=1.0, alpha=0.45, zorder=5)

    ax.plot([], [], color="0.35", lw=1.5, label="Tracked halo distribution")
    ax.plot([], [], color="0.20", ls="--", lw=1.0, alpha=0.45, label=r"Best halo $M_{\rm h}(z)$")
    ax.set_xscale("log")
    ax.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
    ax.set_ylim(0.0, max(1.0, 1.15 * float(maximum_count)))
    ax.set_xlabel(r"Halo mass $M_{\rm h}(z)$ [$M_{\odot}$]")
    ax.set_ylabel("Halo number per mass bin")
    ax.text(
        0.03,
        0.96,
        rf"Tracked MPB final-halo sample; best $h_{{z=0}}$={int(distribution['best_halo_id_z0'])}" + "\n"
        r"coloured dashed lines: selected halo $M_{\rm h}(z)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.20",
    )
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="upper right", fontsize=7.4, ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(redshifts) == 1:
        colour_bar.set_ticks([float(redshifts[0])])
    return fig, distribution


def _fig11_redshift_tick_values(maximum_redshift):
    standard = np.asarray([7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0], dtype=float)
    ticks = standard[standard <= float(maximum_redshift) + FIG11_REDSHIFT_ROW_ATOL]
    if len(ticks) == 0:
        ticks = np.asarray([FIG11_TARGET_REDSHIFT], dtype=float)
    return ticks


def plot_fig11_assembly(selection):
    histories = list(selection.get("histories", ()))
    if len(histories) == 0 or len(histories) > FIG11_MAX_PANELS:
        raise ValueError(f"Fig. 11 requires between one and {FIG11_MAX_PANELS} selected histories.")
    if len({int(history["halo_id_z0"]) for history in histories}) != len(histories):
        raise ValueError("Fig. 11 selected histories contain repeated halo IDs.")

    t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    x_right = float(t0 - Redshift2CosmicAge(FIG11_TARGET_REDSHIFT, time_unit="Gyr"))
    x_left = max(float(np.max(history["main"]["x_gyr"])) for history in histories)
    if not np.isfinite(x_left) or not np.isfinite(x_right) or x_left < x_right:
        raise ValueError("Fig. 11 selected histories do not share a valid high-z to z=7 time interval.")
    if x_left == x_right:
        x_left = float(np.nextafter(x_right, np.inf))

    mass_values = []
    maximum_redshift = FIG11_TARGET_REDSHIFT
    for history in histories:
        main = history["main"]
        mass_values.append(np.asarray(main["halo_mass_msun"], dtype=float))
        maximum_redshift = max(maximum_redshift, float(np.max(main["redshift"])))
        for satellite in history["satellites"]:
            mass_values.append(np.asarray(satellite["halo_mass_msun"], dtype=float))
    all_mass = np.concatenate(mass_values)
    if np.any(~np.isfinite(all_mass)) or np.any(all_mass <= 0.0):
        raise ValueError("Fig. 11 contains non-finite or non-positive plotted halo masses.")
    mass_min = float(np.min(all_mass))
    mass_max = float(np.max(all_mass))
    if mass_min == mass_max:
        mass_min = float(np.nextafter(mass_min, 0.0))
        mass_max = float(np.nextafter(mass_max, np.inf))
    redshift_ticks = _fig11_redshift_tick_values(maximum_redshift)
    redshift_x = np.asarray([t0 - Redshift2CosmicAge(float(z), time_unit="Gyr") for z in redshift_ticks], dtype=float)
    in_range = (redshift_x >= x_right - FIG11_REDSHIFT_ROW_ATOL) & (redshift_x <= x_left + FIG11_REDSHIFT_ROW_ATOL)
    redshift_ticks = redshift_ticks[in_range]
    redshift_x = redshift_x[in_range]
    if len(redshift_ticks) == 0:
        redshift_ticks = np.asarray([FIG11_TARGET_REDSHIFT], dtype=float)
        redshift_x = np.asarray([x_right], dtype=float)

    satellite_gc_counts = [
        int(satellite["n_gc_high_z"])
        for history in histories
        for satellite in history["satellites"]
    ]
    if any(count < 0 for count in satellite_gc_counts):
        raise ValueError("Fig. 11 satellite GC counts must be non-negative.")
    maximum_satellite_gc_count = max(satellite_gc_counts, default=0)
    satellite_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0, float(maximum_satellite_gc_count)))
    satellite_cmap = plt.get_cmap(FIG11_SATELLITE_CMAP)
    colour_mappable = mpl.cm.ScalarMappable(norm=satellite_norm, cmap=satellite_cmap)
    colour_mappable.set_array(np.asarray(satellite_gc_counts, dtype=float))

    fig, axes = plt.subplots(3, 3, constrained_layout=True, dpi=STD_DPI, figsize=(14.2, 11.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(3, 3)
    legend_handles = [Line2D([], [], color="black", lw=1.35, label="Main progenitor")]
    if any(history["satellites"] for history in histories):
        legend_handles.extend([
            Line2D([], [], color="0.35", lw=0.9, label="Satellite branch"),
            Line2D([], [], marker="o", color="0.35", markerfacecolor="0.35", markeredgecolor="none", lw=0.0, label="Satellite maximum"),
        ])

    visible_index = 0
    for panel_index, ax in enumerate(axes.flat):
        row_index, column_index = divmod(panel_index, 3)
        if visible_index >= len(histories):
            ax.set_visible(False)
            continue
        history = histories[visible_index]
        visible_index += 1
        main = history["main"]
        ax.plot(main["x_gyr"], main["halo_mass_msun"], color="black", lw=1.35, zorder=4)
        for satellite in history["satellites"]:
            colour = satellite_cmap(satellite_norm(float(satellite["n_gc_high_z"])))
            ax.plot(satellite["x_gyr"], satellite["halo_mass_msun"], color=colour, lw=0.9, zorder=3)
            ax.plot(
                satellite["marker_x_gyr"],
                satellite["marker_halo_mass_msun"],
                marker="o",
                color=colour,
                markerfacecolor=colour,
                markeredgecolor="none",
                ms=3.8,
                linestyle="none",
                zorder=5,
            )
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(mass_min, mass_max)
        ax.set_xticks(redshift_x)
        ax.set_xticklabels([f"{float(z):g}" for z in redshift_ticks])
        ax.tick_params(direction="in", right=True, top=False, which="both", labelbottom=(row_index == 2), bottom=True)
        if row_index == 0:
            ax.tick_params(labeltop=False)
            lookback_axis = ax.twiny()
            lookback_axis.set_xlim(x_left, x_right)
            lookback_axis.set_xticks(redshift_x)
            lookback_axis.set_xticklabels([f"{float(x):.2f}" for x in redshift_x])
            lookback_axis.set_xlabel(r"Lookback time $t_{\rm lookback}$ [Gyr]")
            lookback_axis.tick_params(direction="in", top=True, bottom=False, which="both")
        if row_index == 2:
            ax.set_xlabel(r"Redshift $z$")
        if column_index == 0:
            ax.set_ylabel(r"Halo mass $M_{\rm h}$ [$M_{\odot}$]")
        score_annotation = _fig11_score_annotation(history)
        ax.text(
            0.04,
            0.96,
            f"{history['suite_label']}; $h_{{z=0}}$={int(history['halo_id_z0'])}\n"
            rf"$\log_{{10}}[M_{{\rm h,cat}}(z=7)/M_{{\odot}}]={float(history['catalogue_log10_halo_mass']):.2f}$; "
            f"$N_{{\\rm sat}}={int(history['n_satellites'])}$\n"
            f"{score_annotation}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            linespacing=1.0,
            color="0.15",
        )
        if visible_index == 1:
            ax.legend(handles=legend_handles, frameon=False, fontsize=7.0, loc="lower right", ncol=1)
        ax.grid(True, alpha=0.3, linestyle=":", which="both")

    colour_bar = fig.colorbar(colour_mappable, ax=axes, orientation="vertical", fraction=0.05, pad=0.025, aspect=25)
    colour_bar.set_label(r"GC count $N_{\rm GC}(z_{\rm form}\geq 7)$")
    colour_bar.locator = mpl.ticker.MaxNLocator(integer=True, nbins=6)
    colour_bar.update_ticks()
    return fig


# MAIN FUNCTION
def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot the Kong & Li 2026 figures, including the UV-aperture diagnostic, from one High-z SMBH Seeds output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.5, help="Log10 stellar-mass bin width for Fig. 01.")
    parser.add_argument("--abundance-matching-redshifts", type=float, nargs="+", default=ABUNDANCE_MATCHING_REDSHIFTS, metavar="Z", help="Output redshifts for the M_bh-M_h abundance-matching figures; each requested value is matched to the nearest available snapshot.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Default: <out_dir>/_plots_Kong&Li2026.")
    parser.add_argument("--uv-table", type=Path, default=UV_CALIBRATION_PATH, help="FSPS-MIST/Chabrier pure-stellar 1500 Angstrom UV table per initially formed stellar mass; feh is log10(Z/Zsun).")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else out_dir / "_plots_Kong&Li2026"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_run_metadata(out_dir)
    if "N_S" not in metadata:
        raise ValueError(f"run_metadata.json is missing required N_S: {out_dir / 'run_metadata.json'}")
    n_s = float(metadata["N_S"])
    if not np.isfinite(n_s) or n_s <= 0.0:
        raise ValueError(f"run_metadata N_S must be finite and positive, got {metadata['N_S']!r}.")
    summary_by_z = load_halo_summary_by_z(out_dir)
    final_gc = load_final_gc(out_dir)
    summary_by_z, final_gc, tng_volume_context = attach_tng_volume_weights(out_dir, summary_by_z, final_gc)
    _set_tng_volume_context(tng_volume_context)
    print(
        f"TNG volume normalisation: catalogue={TNG_CATALOGUE_ROOT}, "
        f"manifest_counts={tng_volume_context['manifest_counts']}, "
        f"output_counts={tng_volume_context['output_counts']}, "
        f"V_TNG50={tng_volume_context['volume_tng50_cmpc3']:.12g} cMpc^3, "
        f"V_TNG100={tng_volume_context['volume_tng100_cmpc3']:.12g} cMpc^3, "
        f"w100={tng_volume_context['tng100_weight']:.12g}."
    )
    final_redshift = float(metadata.get("final_redshift", 0.0))
    if not np.isfinite(final_redshift) or final_redshift < 0.0:
        raise ValueError(f"run_metadata final_redshift must be finite and non-negative, got {final_redshift!r}.")
    chen_fig12_data = load_chen2026_fig06_seed_history()
    fig12, inventory12 = plot_fig12_bhseed_history(
        chen_fig12_data,
        final_gc,
        volume_cmpc3=tng_volume_context["volume_tng50_cmpc3"],
    )
    status_inventory12 = ", ".join(
        f"status {int(status)}: raw={int(inventory12['status_raw_counts'][status])}, "
        f"effective={float(inventory12['status_effective_counts'][status]):.6g}"
        for status in sorted(inventory12["status_raw_counts"])
    )
    print(
        f"Fig. 12 This work positive IMBH-seed inventory: raw={int(inventory12['raw_positive_count'])}, "
        f"effective={float(inventory12['effective_positive_count']):.12g}; {status_inventory12}."
    )
    print(
        f"Fig. 12 display z range={FIG12_DISPLAY_REDSHIFT_RANGE[0]:g}--{FIG12_DISPLAY_REDSHIFT_RANGE[1]:g}: "
        f"raw={int(inventory12['display_raw_count'])}, effective={float(inventory12['display_effective_count']):.12g}; "
        f"outside low-z raw/effective={int(inventory12['outside_low_raw_count'])}/{float(inventory12['outside_low_effective_count']):.12g}, "
        f"outside high-z raw/effective={int(inventory12['outside_high_raw_count'])}/{float(inventory12['outside_high_effective_count']):.12g}; "
        f"unsmoothed Δlog10(1+z)={FIG12_RATE_LOG1PZ_BIN_WIDTH:g}."
    )
    native_first_d = ", ".join(
        f"{role}: x={float(chen_fig12_data['by_panel']['d'][role]['x_log10_1pz'].iloc[0]):.12g}, "
        f"y={float(chen_fig12_data['by_panel']['d'][role]['value_mpc3'].iloc[0]):.12g}"
        for role in FIG12_CHEN_CURVE_ROLES
    )
    print(
        f"Fig. 12 cumulative model grid/value summary: N={len(inventory12['cumulative_x'])}, "
        f"x={float(inventory12['cumulative_x'][0]):.12g}--{float(inventory12['cumulative_x'][-1]):.12g}, "
        f"n={float(inventory12['cumulative_density'][0]):.12g}--{float(inventory12['cumulative_density'][-1]):.12g}, "
        f"boundary={inventory12['cumulative_boundary']}; "
        f"Chen+2026 panel d native first points [{native_first_d}], "
        "endpoint-constant plotting extension to x=0; native CSV values unchanged."
    )
    _save_figure(fig12, plot_dir / FIGURE_12_FILENAME)
    bhmf_data = load_bhmf_data()
    fig06, inventory06 = plot_fig06_bhmf2(bhmf_data, summary_by_z)
    counts06 = ", ".join(f"z={z:.6g}: {n}" for z, n in sorted(inventory06["this_work_by_z"].items()))
    effective_counts06 = ", ".join(f"z={z:.6g}: {n:.6g}" for z, n in sorted(inventory06["this_work_effective_by_z"].items()))
    invalid_counts06 = ", ".join(f"z={z:.6g}: {n}" for z, n in sorted(inventory06["invalid_mass_by_z"].items()) if n > 0)
    out_of_range_counts06 = ", ".join(
        f"z={z:.6g}: raw={inventory06['out_of_range_raw_by_z'][z]}, effective={inventory06['out_of_range_effective_by_z'][z]:.6g}"
        for z in sorted(inventory06["out_of_range_raw_by_z"])
        if inventory06["out_of_range_raw_by_z"][z] > 0
    )
    summary_z_values06 = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    low_redshifts06 = np.sort(np.unique(summary_z_values06[summary_z_values06 <= FIG06_MIN_MODEL_REDSHIFT_EXCLUSIVE]))
    omitted_no_positive06 = ", ".join(f"{float(z):.6g}" for z in inventory06["omitted_no_positive_redshifts"])
    omitted_no_visible06 = ", ".join(f"{float(z):.6g}" for z in inventory06["omitted_no_visible_density_redshifts"])
    print(f"Fig. 06 excluded model output redshifts z<=3: [{', '.join(f'{float(z):.6g}' for z in low_redshifts06)}].")
    print(f"Fig. 06 This work positive-BH inventory: {counts06}.")
    print(f"Fig. 06 This work effective inventory: {effective_counts06}.")
    print(f"Fig. 06 invalid M_SMBH_final rows omitted: [{invalid_counts06}].")
    print(f"Fig. 06 positive masses outside plotted mass range: [{out_of_range_counts06}].")
    print(f"Fig. 06 omitted redshifts with no positive mass: [{omitted_no_positive06}].")
    print(f"Fig. 06 omitted redshifts with no visible density: [{omitted_no_visible06}].")
    _save_figure(fig06, plot_dir / "Fig.06_BHMF2.pdf")
    """
    abundance_table, abundance_snapshots = build_mbh_mhalo_abundance_matching(
        summary_by_z,
        args.abundance_matching_redshifts,
        volume_cmpc3=tng_volume_context["volume_tng50_cmpc3"],
    )
    abundance_table_path = plot_dir / "Fig.07_Mbh-Mhalo_AbundanceMatching.csv"
    abundance_table.to_csv(abundance_table_path, index=False)
    print(f"Saved {abundance_table_path}")
    for snapshot in abundance_snapshots:
        print(
            f"Abundance matching z={float(snapshot['redshift']):.6g}: "
            f"N_halo={int(snapshot['n_halo'])}, N_positive_central_BH={int(snapshot['n_positive_bh'])}, "
            f"N_halo,eff={float(snapshot['n_halo_effective']):.6g}, N_positive_central_BH,eff={float(snapshot['n_positive_bh_effective']):.6g}, "
            f"f_occ={float(snapshot['occupation_fraction']):.4f}, "
            f"M_halo,occ={10.0**float(snapshot['matched_halo_threshold_log_mass']):.6g} M_sun."
        )
    fig07 = plot_fig07_mbh_mhalo_abundance_matching(abundance_snapshots)
    _save_figure(fig07, plot_dir / FIGURE_07_FILENAME)
    fig08 = plot_fig08_mbh_mhalo_cumulative_abundance(abundance_snapshots)
    _save_figure(fig08, plot_dir / FIGURE_08_FILENAME)

    observations = load_mbh_mstar_observations()
    fig01 = plot_fig01_mbh_mstar(summary_by_z, observations, float(args.mass_bin_width_dex))
    _save_figure(fig01, plot_dir / "Fig.01_Mbh-Mstar.pdf")

    points, curves = load_juodzbalis2026_fig2()
    fig3_reference = load_juodzbalis2026_fig3_bh_masses()
    uv_calibration = load_uv_calibration(args.uv_table)
    all_z_rows = _select_fig02_z_rows(summary_by_z)
    all_deposit_profile = load_deposit_profile_for_redshift_summary(_deposit_path(out_dir), all_z_rows, final_redshift)
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
    score_table, fig02_best = score_fig02_candidate_haloes(out_dir, points, eligible_z_rows, eligible_deposit_profile, final_gc, uv_calibration)
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
    print(f"Fig. 02/Fig. 04 candidate pool: out_dir={out_dir}, N_S={n_s:.3g}, eligible profiles={len(eligible_z_rows)}, CSV rows={len(score_table)}, finite Keplerian+UV scores={int(np.isfinite(score_table['score_keplerian_uv'].to_numpy(dtype=float)).sum())}.")
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

    fig11_selection = select_fig11_assembly_histories(out_dir, summary_by_z, final_gc, metadata, tng_volume_context, score_table, fig02_best)
    fig11 = plot_fig11_assembly(fig11_selection)
    _save_figure(fig11, plot_dir / FIGURE_11_FILENAME)
    print(
        "Fig. 11 assembly panels: "
        + ", ".join(
            f"{history['suite_label']} halo_id_z0={int(history['halo_id_z0'])} "
            f"log10M_h,cat(z=7)={float(history['catalogue_log10_halo_mass']):.4f} "
            f"raw-MPB-log10M_h(z=7)={float(history['main']['endpoint_log10_halo_mass']):.4f} "
            f"N_sat={int(history['n_satellites'])}; {_fig11_score_diagnostic(history)}"
            for history in fig11_selection["histories"]
        )
    )
    if fig11_selection["rejected"]:
        print(
            "Fig. 11 discarded comparison candidates: "
            + ", ".join(str(item["halo_id_z0"]) for item in fig11_selection["rejected"])
        )

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
    effective_counts = ", ".join(f"z={z:.6g}: {n:.6g}" for z, n in sorted(inventory["nuclear_effective_by_z"].items()))
    print(f"Fig. 03 nuclear positive-BH inventory: {counts}.")
    print(f"Fig. 03 nuclear effective inventory: {effective_counts}.")
    print(f"Fig. 03 satellite positive-BH inventory: {inventory['satellite']} (status 1 and -4), effective={inventory['satellite_effective']:.6g}.")
    _save_figure(fig03, plot_dir / "Fig.03_BHMF.pdf")

    chen_data = load_chen2026_fig05a_seed_mass_functions()
    fig10, inventory10 = plot_fig10_bhsmf(chen_data, summary_by_z, volume_cmpc3=tng_volume_context["volume_tng50_cmpc3"])
    central_inventory = inventory10["central"]
    plotted_central_indices = set(int(value) for value in central_inventory["positive_redshift_indices"])
    for index, redshift in enumerate(central_inventory["redshifts"]):
        print(
            f"Fig. 10 This work inventory z={float(redshift):.6g}: "
            f"positive raw={int(central_inventory['positive_raw_counts'][index])}, "
            f"positive effective={float(central_inventory['positive_effective_counts'][index]):.6g}, "
            f"zero raw={int(central_inventory['zero_raw_counts'][index])}, "
            f"zero effective={float(central_inventory['zero_effective_counts'][index]):.6g}, "
            f"plotted positive curve={index in plotted_central_indices}."
        )
    #omitted = ", ".join(f"{float(redshift):.6g}" for redshift in inventory10["omitted_redshifts"])
    #print(f"Fig. 10 omitted empty This work redshift thresholds: [{omitted}].")
    print(f"Fig. 10 reference volume: {inventory10['central']['volume_cmpc3']:.12g} cMpc^3; positive central M_SMBH_init states use inherited TNG parent-halo weights.")
    _save_figure(fig10, plot_dir / FIGURE_10_FILENAME)

    fig09, inventory09 = plot_fig09_halo_distribution(summary_by_z, fig02_best)
    _save_figure(fig09, plot_dir / FIGURE_09_DISTR_FILENAME)
    empty_redshifts = ", ".join(f"{float(z):.6g}" for z in inventory09["empty_redshifts"])
    missing_best_redshifts = ", ".join(f"{float(z):.6g}" for z in inventory09["best_halo_missing_redshifts"])
    print(
        f"Fig. 09 halo distribution: plotted redshifts={len(inventory09['redshifts'])}, "
        f"excluded unavailable rows={int(inventory09['excluded_unavailable_rows'])}, "
        f"empty redshifts=[{empty_redshifts}], bin width={FIG09_DISTR_BIN_WIDTH_DEX:.2f} dex, "
        f"best halo_id_z0={int(inventory09['best_halo_id_z0'])}, "
        f"best-halo track lines={len(inventory09['best_halo_track'])}, "
        f"missing best-halo redshifts=[{missing_best_redshifts}]."
    )
    """

if __name__ == "__main__":
    main()
