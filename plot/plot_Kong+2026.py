#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Plot IMBH seed diagnostics and stored central-BH summaries from one finished run.

This script reads one per-``N_s`` formation catalogue for Fig.01 and Fig.02:
initial cluster mass versus radius, and initial surface density versus
metallicity with IMBH-mass threshold contours. It also reads the
redshift-resolved halo summary and root ``allcat`` table written by
``my/run.py`` so Fig.03 plots sunk BH mass against descendant z=0 halo mass.
Fig.04 writes ``Fig.04_Mbh~Mstar.pdf`` after converting the run-output
same-redshift MPB halo masses to stellar masses with the project SMHM helper
for comparison to observed ``M_BH-M_*`` points. Fig.05 writes
``Fig.05_Mbh~Mnsc.pdf`` and plots sunk BH mass against the redshift-resolved
NSC mass when that column is present in the run output. Fig.06 compares the
individual seed BH mass distribution for all BH seeds and the subset that sank
into the centre. Fig.07 plots individual sunk IMBH mass against sunk GC stellar
mass, including sunk wanderers, coloured by sunk redshift. Fig.08 reproduces
the Juodzbalis et al. (2026) QSO1 rotation curve and overlays the z~7 radial
mass-deposition velocity profiles. Fig.09 plots the local massive-BH mass
function, decomposed into nuclear and satellite components and compared to the
Greene et al. (2020) lower-limit mass-function estimates used by Kritos et al.
(2025).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List, Tuple

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

from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CosmicAge2Redshift,
    Mstar_SMHM,
    Redshift2CosmicAge,
    STD_DPI,
    imbh_mass_from_sigma_metallicity,
)
from load_obs import (  # noqa: E402
    load_cliff_fig14_observations,
    load_juodzbalis2026_fig2_rotation_curve,
    load_juodzbalis2026_fig4_observations,
    load_kritos2025_fig9_mass_functions,
    load_kritos2025_fig10_mbh_mnsc_observations,
)
from load_output import build_kong_model, load_deposit_profile_for_redshift_summary, load_run_metadata  # noqa: E402
from plot_common import plot_dir as default_plot_dir  # noqa: E402


class IMBHModelConfig:
    def __init__(self, *, enabled: bool = True, fh: float = 0.125, metallicity_kind: str = "feh") -> None:
        self.enabled = bool(enabled)
        self.fh = float(fh)
        self.metallicity_kind = str(metallicity_kind)


class IMBHModel:
    def __init__(self, config: IMBHModelConfig) -> None:
        self.config = config

    def radius_eq7(self, cluster_mass_msun):
        mass = np.asarray(cluster_mass_msun, dtype=float)
        radius = self.config.fh * 2.365 / 1.3 * (np.clip(mass, 1.0e-30, None) / 1.0e4) ** 0.180
        return float(radius) if radius.ndim == 0 else radius

    def imbh_mass_from_sigma_metallicity(self, sigma_h_msun_pc2, metallicity):
        if self.config.metallicity_kind == "z_ratio":
            z_ratio = metallicity
        else:
            z_ratio = np.power(10.0, np.asarray(metallicity, dtype=float))
        return imbh_mass_from_sigma_metallicity(sigma_h_msun_pc2, z_ratio)

NS_VALUE_DEFAULT = 2.0
FIGURE_01_FILENAME = "Fig.01_initial_cluster_mass_radius_imbh_seeds.pdf"
FIGURE_02_FILENAME = "Fig.02_initial_surface_density_metallicity_imbh_thresholds.pdf"
FIGURE_03_FILENAME = "Fig.03_sunk_bh_mass_vs_halo_mass.pdf"
FIGURE_04_FILENAME = "Fig.04_Mbh~Mstar.pdf"
FIGURE_05_FILENAME = "Fig.05_Mbh~Mnsc.pdf"
FIGURE_06_FILENAME = "Fig.06_sunk_bh_mass_histogram.pdf"
FIGURE_07_FILENAME = "Fig.07_sunkMimbh~sunkMgc.pdf"
FIGURE_08_FILENAME = "Fig.08_QSO1_rotation_curve_mass_profile.pdf"
FIGURE_09_FILENAME = "Fig.09_local_bh_mass_function.pdf"
HALO_MASS_UNIT_LABEL = r"M_{\odot}"
SMHM_TOP_AXIS_DEFAULT = True
BH_TO_STELLAR_MASS_RATIOS = (0.01, 0.1, 1.0)
REINES_VOLONTERI_2015_NORM = 7.45
REINES_VOLONTERI_2015_SLOPE = 1.05
REINES_VOLONTERI_2015_SCATTER_DEX = 0.55
FH_VALUES = (0.125, 0.184, 0.269, 0.395, 0.580)
IMBH_CONTOUR_LEVELS = (100.0, 300.0, 1000.0, 3000.0)
IMBH_SIZE_REFERENCE_MASSES = (100.0, 300.0, 1000.0, 3000.0)
MIN_IMBH_PLOT_MASS_MSUN = 0.0
SYMLINTHRESH_MGC_MSUN = 1.0e3
STATUS_SUNK_GC = -3
STATUS_WANDERER = -4
STATUS_SUNK_WANDERER = -5
STATUS_ALIVE = 1
STATUS_EXHAUSTED = -1
STATUS_TORN = -2
SUNK_BH_STATUSES = (STATUS_SUNK_GC, STATUS_SUNK_WANDERER)
SATELLITE_BH_STATUSES = (STATUS_ALIVE, STATUS_WANDERER)
FIG08_TARGET_REDSHIFT = 7.04
FIG08_REDSHIFT_ATOL = 0.1
FIG08_QSO1_LOGMBH_TARGET = 7.7
FIG08_MATCH_MASS_WEIGHT = 0.10
FIG08_MATCH_RADIUS_RANGE_PC = (12.5, 150.0)
FIG08_SCATTER_PERCENTILES = (16.0, 84.0)
FIG08_VELOCITY_SIN_I = 1.0
FIG08_G_PC_MSUN_KMS2 = 4.30091e-3
FIG08_RADIUS_MAX_PC = 170.0
FIG09_ILLUSTRIS1_DARK_H = 0.704
FIG09_ILLUSTRIS1_DARK_SIDE_MPC = 75.0 / FIG09_ILLUSTRIS1_DARK_H
FIG09_ILLUSTRIS1_DARK_VOLUME_MPC3 = FIG09_ILLUSTRIS1_DARK_SIDE_MPC**3
FIG09_BIN_EDGES = np.logspace(2.0, 11.0, 30)
FIG05_BH_TO_NSC_RATIOS = (10.0, 1.0, 0.1, 0.01)
TIMES_COMPATIBLE_FONT_PATHS = (
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf"),
)

def _ns_tag(ns_value: float) -> str:
    return f"{float(ns_value):.1f}".replace(".", "p")


def _available_times_serif_fonts() -> List[str]:
    for font_path in TIMES_COMPATIBLE_FONT_PATHS:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))

    serif_fonts: List[str] = []
    for family in ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Times"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        serif_fonts.append(family)
    return serif_fonts or ["DejaVu Serif"]


def _apply_plot_style() -> None:
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _available_times_serif_fonts(),
            "font.size": 10,
            "mathtext.default": "regular",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "text.usetex": use_tex,
        }
    )
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{bm}"


def _read_comment_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                text = line[1:].strip()
                if text:
                    return text.split()
    raise ValueError(f"Cannot find header columns in {path}")


def _imbh_marker_sizes(imbh_mass: np.ndarray) -> np.ndarray:
    mass = np.asarray(imbh_mass, dtype=float)
    sizes = np.full(mass.shape, 18.0, dtype=float)
    valid = np.isfinite(mass) & (mass > 0.0)
    if not np.any(valid):
        return sizes

    log_mass = np.log10(mass[valid])
    lo = float(np.nanmin(log_mass))
    hi = float(np.nanmax(log_mass))
    if hi <= lo:
        sizes[valid] = 55.0
    else:
        sizes[valid] = np.interp(log_mass, (lo, hi), (12.0, 95.0))
    return sizes


def _mass_hist_label(label: str, masses: np.ndarray) -> str:
    mass = np.asarray(masses, dtype=float)
    mass = mass[np.isfinite(mass) & (mass > 0.0)]
    if len(mass) == 0:
        return f"{label} (N=0)"
    return f"{label} (N={len(mass)}, median={np.median(mass):.2g} $M_\\odot$)"


def _redshift_from_final_lookback(lookback_time_gyr: np.ndarray, final_redshift: float) -> np.ndarray:
    lookback = np.asarray(lookback_time_gyr, dtype=float)
    if np.any(~np.isfinite(lookback)):
        raise ValueError("Cannot convert non-finite final-GC lookback times to sunk redshifts.")
    if np.any(lookback < 0.0):
        raise ValueError("Cannot convert negative final-GC lookback times to sunk redshifts.")

    final_age_gyr = float(Redshift2CosmicAge(float(final_redshift), time_unit="Gyr"))
    if np.any(lookback > final_age_gyr):
        max_lookback = float(np.nanmax(lookback))
        raise ValueError(
            "Final-GC lookback time is older than the final cosmic age: "
            f"max_lookback={max_lookback:.6g} Gyr, final_age={final_age_gyr:.6g} Gyr."
        )

    event_age_gyr = final_age_gyr - lookback
    if np.any(event_age_gyr <= 0.0):
        min_event_age = float(np.nanmin(event_age_gyr))
        raise ValueError(f"Final-GC sunk event has non-positive cosmic age: {min_event_age:.6g} Gyr.")

    return np.asarray(
        [CosmicAge2Redshift(float(age), time_unit="Gyr") for age in event_age_gyr],
        dtype=float,
    )


def _stellar_mass_from_halo_mass_at_redshift(
    halo_mass: np.ndarray | float,
    redshift: np.ndarray | float,
) -> np.ndarray | float:
    mh = np.asarray(halo_mass, dtype=float)
    z = np.asarray(redshift, dtype=float)
    mh_broadcast, z_broadcast = np.broadcast_arrays(mh, z)
    out = np.full(mh_broadcast.shape, np.nan, dtype=float)
    flat_mh = mh_broadcast.ravel()
    flat_z = z_broadcast.ravel()
    flat_out = out.ravel()
    for i, (mass, z_val) in enumerate(zip(flat_mh, flat_z)):
        if np.isfinite(mass) and mass > 0.0 and np.isfinite(z_val):
            flat_out[i] = Mstar_SMHM(Mhalo=float(mass), z=float(z_val), scatter=False)
    if np.isscalar(halo_mass) and np.isscalar(redshift):
        return float(out.reshape(-1)[0])
    return out


def _attach_plot_masses(summary: pd.DataFrame, z0_halo_mass_lookup: Dict[int, float]) -> pd.DataFrame:
    out = summary.copy()
    out["hid_z0"] = pd.to_numeric(out["hid_z0"], errors="coerce").astype(int)
    out["z_out"] = pd.to_numeric(out["z_out"], errors="coerce")
    out["logMh_z0_msun"] = out["hid_z0"].map(z0_halo_mass_lookup)
    missing_z0 = out.loc[~np.isfinite(out["logMh_z0_msun"].to_numpy(dtype=float)), "hid_z0"].unique()
    if len(missing_z0) > 0:
        missing_text = ", ".join(str(int(hid)) for hid in sorted(missing_z0)[:8])
        raise KeyError(f"Root allcat is missing descendant z=0 halo mass for halo(s): {missing_text}")

    out["mhalo_z0_msun"] = np.power(10.0, out["logMh_z0_msun"].to_numpy(dtype=float))
    out["halo_mass_available"] = pd.to_numeric(out["halo_mass_available"], errors="coerce").fillna(0).astype(int)
    out["logMh_z_msun"] = pd.to_numeric(out["logMh_z_msun"], errors="coerce")
    valid_halo_z = (
        (out["halo_mass_available"].to_numpy(dtype=int) == 1)
        & np.isfinite(out["logMh_z_msun"].to_numpy(dtype=float))
    )
    out["mhalo_z_msun"] = np.nan
    out.loc[valid_halo_z, "mhalo_z_msun"] = np.power(
        10.0,
        out.loc[valid_halo_z, "logMh_z_msun"].to_numpy(dtype=float),
    )
    out["mstar_z_smhm_msun"] = _stellar_mass_from_halo_mass_at_redshift(
        out["mhalo_z_msun"].to_numpy(dtype=float),
        out["z_out"].to_numpy(dtype=float),
    )
    mstar = out["mstar_z_smhm_msun"].to_numpy(dtype=float)
    out["logMstar_z_smhm_msun"] = np.where(
        np.isfinite(mstar) & (mstar > 0.0),
        np.log10(mstar),
        np.nan,
    )
    out["M_NSC"] = pd.to_numeric(out["M_NSC"], errors="coerce")
    out["M_SMBH_final"] = pd.to_numeric(out["M_SMBH_final"], errors="coerce")
    if (out["M_NSC"].dropna() < 0.0).any() or (out["M_SMBH_final"].dropna() < 0.0).any():
        raise ValueError("haloSummaryByZ contains negative M_NSC or M_SMBH_final values.")
    return out


def _regular_log_bin_edges(values: Iterable[float], step_dex: float) -> np.ndarray:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0, float(step_dex)], dtype=float)
    lo = float(step_dex) * math.floor(float(vals.min()) / float(step_dex))
    hi = float(step_dex) * math.ceil(float(vals.max()) / float(step_dex))
    if hi <= lo:
        hi = lo + float(step_dex)
    edges = np.arange(lo, hi + 0.5 * float(step_dex), float(step_dex), dtype=float)
    if len(edges) < 2:
        edges = np.array([lo, lo + float(step_dex)], dtype=float)
    return edges


def _bin_track(track: pd.DataFrame, edges: np.ndarray, x_log_col: str) -> pd.DataFrame:
    x_log = track[x_log_col].to_numpy(dtype=float)
    y = track["M_SMBH_final"].to_numpy(dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        include_right = idx == len(edges) - 2
        mask = np.isfinite(x_log) & np.isfinite(y) & (x_log >= left)
        mask &= (x_log <= right) if include_right else (x_log < right)
        if int(mask.sum()) == 0:
            continue
        y_sel = y[mask]
        rows.append(
            {
                "logx_center": 0.5 * (left + right),
                "count": int(mask.sum()),
                "mean_mass": float(np.mean(y_sel)),
                "std_mass": float(np.std(y_sel)),
            }
        )
    return pd.DataFrame(rows)


def _present_day_stellar_mass_from_halo_mass(halo_mass: np.ndarray | float) -> np.ndarray | float:
    return _stellar_mass_from_halo_mass_at_redshift(halo_mass, 0.0)


_PRESENT_DAY_SMHM_AXIS_CACHE: Tuple[np.ndarray, np.ndarray] | None = None


def _present_day_halo_mass_from_stellar_mass(stellar_mass: np.ndarray | float) -> np.ndarray | float:
    global _PRESENT_DAY_SMHM_AXIS_CACHE
    if _PRESENT_DAY_SMHM_AXIS_CACHE is None:
        log_mh_grid = np.linspace(8.0, 16.0, 4096)
        mh_grid = np.power(10.0, log_mh_grid)
        mstar_grid = _present_day_stellar_mass_from_halo_mass(mh_grid)
        valid_grid = np.isfinite(mstar_grid) & (mstar_grid > 0.0)
        log_mstar_grid = np.log10(mstar_grid[valid_grid])
        log_mh_grid = log_mh_grid[valid_grid]
        order = np.argsort(log_mstar_grid)
        log_mstar_sorted = log_mstar_grid[order]
        log_mh_sorted = log_mh_grid[order]
        log_mstar_unique, unique_idx = np.unique(log_mstar_sorted, return_index=True)
        _PRESENT_DAY_SMHM_AXIS_CACHE = (log_mstar_unique, log_mh_sorted[unique_idx])

    log_mstar_grid, log_mh_grid = _PRESENT_DAY_SMHM_AXIS_CACHE
    sm = np.asarray(stellar_mass, dtype=float)
    out = np.full(sm.shape, np.nan, dtype=float)
    valid = np.isfinite(sm) & (sm > 0.0)
    out[valid] = np.power(
        10.0,
        np.interp(np.log10(sm[valid]), log_mstar_grid, log_mh_grid, left=np.nan, right=np.nan),
    )
    if np.isscalar(stellar_mass):
        return float(out.reshape(-1)[0])
    return out


def _add_smhm_top_axis(ax: plt.Axes) -> None:
    secax = ax.secondary_xaxis(
        "top",
        functions=(_present_day_stellar_mass_from_halo_mass, _present_day_halo_mass_from_stellar_mass),
    )
    secax.set_xlabel(r"Corresponding $z=0$ stellar mass from SMHM [$M_{\odot}$]")
    secax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    secax.tick_params(direction="in", top=True, which="both")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _log_error_to_linear(log_value: float, err_lo: float, err_hi: float) -> np.ndarray | None:
    if not np.isfinite(log_value):
        return None
    value = 10.0**log_value
    lo = float(err_lo) if np.isfinite(err_lo) and err_lo > 0.0 else 0.0
    hi = float(err_hi) if np.isfinite(err_hi) and err_hi > 0.0 else 0.0
    if lo <= 0.0 and hi <= 0.0:
        return None
    return np.array([[value - 10.0 ** (log_value - lo)], [10.0 ** (log_value + hi) - value]], dtype=float)


def _reines_volonteri_2015_mbh(mstar_msun: np.ndarray) -> np.ndarray:
    return np.power(10.0, REINES_VOLONTERI_2015_NORM + REINES_VOLONTERI_2015_SLOPE * np.log10(mstar_msun / 1.0e11))


def _plot_mbh_mstar_observations(ax: plt.Axes, observations: pd.DataFrame) -> None:
    seen_labels: set[str] = set()
    for _, row in observations.iterrows():
        log_mstar = float(row["logMstar"])
        log_mbh = float(row["logMBH"])
        x = 10.0**log_mstar
        y = 10.0**log_mbh
        colour = str(row["color"]).strip() or "0.45"
        marker = str(row["marker"]).strip() or "o"
        group = str(row["plot_group"]).strip().lower()
        label_base = str(row.get("legend_label", "")).strip()
        if not label_base:
            if group == "cliff":
                label_base = str(row["name"]).strip() or "The Cliff"
            else:
                label_base = str(row.get("reference", "")).strip() or "Observed comparison points"
        default_zorder = 7 if group == "cliff" else 5
        default_alpha = 1.0 if group == "cliff" else 0.75
        default_marker_size = 6.5 if group == "cliff" else 5.5
        zorder_value = float(row.get("zorder", np.nan))
        alpha_value = float(row.get("alpha", np.nan))
        marker_size_value = float(row.get("marker_size", np.nan))
        edgewidth_value = float(row.get("marker_edgewidth", np.nan))
        zorder = int(zorder_value) if np.isfinite(zorder_value) else default_zorder
        alpha = alpha_value if np.isfinite(alpha_value) else default_alpha
        marker_size = marker_size_value if np.isfinite(marker_size_value) and marker_size_value > 0.0 else default_marker_size
        marker_edgecolor = str(row.get("marker_edgecolor", "")).strip()
        if not marker_edgecolor:
            marker_edgecolor = "white" if group == "cliff" else colour
        marker_edgewidth = edgewidth_value if np.isfinite(edgewidth_value) else (0.7 if group == "cliff" else 0.0)
        label = None if label_base in seen_labels else label_base
        seen_labels.add(label_base)

        yerr = None
        if not _as_bool(row["logMBH_upper_limit"]):
            yerr = _log_error_to_linear(log_mbh, float(row["logMBH_err_lo"]), float(row["logMBH_err_hi"]))

        xerr = None
        if not _as_bool(row["logMstar_upper_limit"]):
            xerr = _log_error_to_linear(log_mstar, float(row["logMstar_err_lo"]), float(row["logMstar_err_hi"]))

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=marker_size,
            mfc=colour,
            mec=marker_edgecolor,
            mew=marker_edgewidth,
            ecolor=colour,
            elinewidth=1.0,
            capsize=2.5,
            color=colour,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
        if _as_bool(row["logMstar_upper_limit"]):
            ax.annotate(
                "",
                xy=(x * 0.55, y),
                xytext=(x * 0.95, y),
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.1, "alpha": alpha},
                zorder=zorder,
            )
        if _as_bool(row["logMBH_upper_limit"]):
            ax.annotate(
                "",
                xy=(x, y * 0.55),
                xytext=(x, y * 0.95),
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.1, "alpha": alpha},
                zorder=zorder,
            )


def _plot_fig05_ratio_guides(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 256)
    styles = {
        10.0: "dashdot",
        1.0: "solid",
        0.1: "dashed",
        0.01: "dotted",
    }
    for ratio in FIG05_BH_TO_NSC_RATIOS:
        ax.plot(x_line, ratio * x_line, color="lime", lw=0.55, ls=styles[ratio], zorder=1)
        label_x = x_min * 2.0
        label_y = ratio * label_x
        if y_min < label_y < y_max:
            exponent = int(round(np.log10(ratio)))
            ax.text(label_x, label_y, rf"$10^{{{exponent}}}$", color="lime", fontsize=8.5, ha="left", va="center", zorder=2)


def _fig05_observation_yerr(row: pd.Series) -> np.ndarray | None:
    if _as_bool(row["mbh_upper_limit"]):
        return None
    return _log_error_to_linear(
        float(row["log10_mbh_msun"]),
        float(row["log10_mbh_err_lo"]),
        float(row["log10_mbh_err_hi"]),
    )


def _plot_fig05_kritos_observations(ax: plt.Axes, observations: pd.DataFrame) -> None:
    seen_labels: set[str] = set()
    order = [
        ("galaxy", "upper_limit"),
        ("ucd", "upper_limit"),
        ("galaxy", "detection"),
        ("ucd", "detection"),
    ]
    for sample, measurement in order:
        rows = observations.loc[observations["sample"].eq(sample) & observations["measurement"].eq(measurement)]
        for _, row in rows.iterrows():
            x = 10.0 ** float(row["log10_mnsc_msun"])
            y = 10.0 ** float(row["log10_mbh_msun"])
            colour = str(row["colour"]).strip() or ("red" if sample == "galaxy" else "blue")
            marker = str(row["marker"]).strip() or ("v" if measurement == "upper_limit" else ".")
            label_base = str(row.get("legend_label", "")).strip()
            if not label_base:
                label_base = f"{'UCD' if sample == 'ucd' else 'galaxy'}, {'limits' if measurement == 'upper_limit' else 'detections'}"
            label = None if label_base in seen_labels else label_base
            seen_labels.add(label_base)

            if measurement == "upper_limit":
                ax.scatter(
                    [x],
                    [y],
                    marker=marker,
                    s=36,
                    color=colour,
                    edgecolors=colour,
                    linewidths=0.8,
                    label=label,
                    zorder=8,
                )
                continue

            ax.errorbar(
                x,
                y,
                yerr=_fig05_observation_yerr(row),
                fmt=marker,
                ms=6.0,
                color=colour,
                ecolor=colour,
                elinewidth=1.1,
                capsize=2.0,
                markeredgecolor=colour,
                markerfacecolor=colour,
                label=label,
                zorder=9,
            )

    ngc1023 = observations.loc[observations["point_id"].astype(str).eq("NGC1023")]
    if len(ngc1023) > 0:
        ax.text(5.5e5, 3.0e7, "NGC 1023", color="red", fontsize=9.5, ha="left", va="center", zorder=10)


def plot_fig01(formation: pd.DataFrame) -> plt.Figure:
    log_mass = formation["logM_form"].to_numpy(dtype=float)
    cluster_mass = np.power(10.0, log_mass)
    radius_pc = formation["gc_radius_pc"].to_numpy(dtype=float)
    z_form = formation["zform"].to_numpy(dtype=float)
    imbh_mass = formation["M_IMBH_init"].to_numpy(dtype=float)
    valid = (
        np.isfinite(cluster_mass)
        & (cluster_mass > 0.0)
        & np.isfinite(radius_pc)
        & (radius_pc > 0.0)
        & np.isfinite(z_form)
        & np.isfinite(imbh_mass)
        & (imbh_mass > MIN_IMBH_PLOT_MASS_MSUN)
    )
    if not np.any(valid):
        raise ValueError("No finite IMBH-seeded formation rows are available for Fig.01.")

    cluster_mass = cluster_mass[valid]
    radius_pc = radius_pc[valid]
    z_form = z_form[valid]
    imbh_mass = imbh_mass[valid]
    reference_masses = np.asarray(IMBH_SIZE_REFERENCE_MASSES, dtype=float)
    combined_sizes = _imbh_marker_sizes(np.concatenate([imbh_mass, reference_masses]))
    marker_sizes = combined_sizes[: len(imbh_mass)]
    reference_sizes = combined_sizes[len(imbh_mass) :]

    x_min = cluster_mass.min() / 1.25
    x_max = cluster_mass.max() * 1.25
    mass_grid = np.logspace(np.log10(x_min), np.log10(x_max), 512)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))

    fh_colours = mpl.cm.cividis(np.linspace(0.15, 0.9, len(FH_VALUES)))
    curve_radii = []
    for fh, colour in zip(FH_VALUES, fh_colours):
        model = IMBHModel(IMBHModelConfig(enabled=True, fh=fh))
        radius_grid = np.asarray(model.radius_eq7(mass_grid), dtype=float)
        curve_radii.append(radius_grid)
        ax.plot(mass_grid, radius_grid, c=colour, lw=1.6, alpha=0.95, label=rf"$f_h={fh:.3f}$")

    if len(np.unique(z_form)) == 1:
        norm = colors.Normalize(vmin=float(z_form[0]) - 0.5, vmax=float(z_form[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(np.nanmin(z_form)), vmax=float(np.nanmax(z_form)))
    scatter = ax.scatter(cluster_mass, radius_pc, c=z_form, s=marker_sizes, cmap=mpl.cm.viridis, norm=norm, alpha=0.35, edgecolors="none", rasterized=True, zorder=5)
    colour_bar = fig.colorbar(scatter, ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Formation redshift z")

    curve_legend = ax.legend(loc="lower right", fontsize=7.0, frameon=False, title=r"Eq. (7)")
    ax.add_artist(curve_legend)
    size_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="0.45", markeredgecolor="none", markersize=float(np.sqrt(size)), lw=0, label=rf"${mass:g}\ M_\odot$")
        for mass, size in zip(reference_masses, reference_sizes)
    ]
    ax.legend(handles=size_handles, loc="upper left", fontsize=7.0, frameon=False, title=r"IMBH seed mass")

    y_values = np.concatenate([radius_pc, *curve_radii])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(float(np.nanmin(y_values)) / 1.25, float(np.nanmax(y_values)) * 1.25)
    ax.set_xlabel(r"Initial cluster mass $M_{\rm cl}$ [$M_{\odot}$]")
    ax.set_ylabel(r"Initial 3D half-mass radius $r_h$ [pc]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig02(formation: pd.DataFrame) -> plt.Figure:
    sigma_h = formation["sigma_h_msun_pc2"].to_numpy(dtype=float)
    feh = formation["feh"].to_numpy(dtype=float)
    z_form = formation["zform"].to_numpy(dtype=float)
    imbh_mass = formation["M_IMBH_init"].to_numpy(dtype=float)
    valid = (
        np.isfinite(sigma_h)
        & (sigma_h > 0.0)
        & np.isfinite(feh)
        & np.isfinite(z_form)
        & np.isfinite(imbh_mass)
        & (imbh_mass > MIN_IMBH_PLOT_MASS_MSUN)
    )
    if not np.any(valid):
        raise ValueError("No finite IMBH-seeded formation rows are available for Fig.02.")

    sigma_h = sigma_h[valid]
    feh = feh[valid]
    z_form = z_form[valid]
    imbh_mass = imbh_mass[valid]
    reference_masses = np.asarray(IMBH_SIZE_REFERENCE_MASSES, dtype=float)
    combined_sizes = _imbh_marker_sizes(np.concatenate([imbh_mass, reference_masses]))
    marker_sizes = combined_sizes[: len(imbh_mass)]
    reference_sizes = combined_sizes[len(imbh_mass) :]

    log_sigma = np.log10(sigma_h)
    log_sigma_span = max(float(log_sigma.max() - log_sigma.min()), 0.3)
    log_sigma_min = float(log_sigma.min()) - 0.05 * log_sigma_span
    log_sigma_max = float(log_sigma.max()) + 0.05 * log_sigma_span
    feh_span = max(float(feh.max() - feh.min()), 0.3)
    feh_min = float(feh.min()) - 0.05 * feh_span
    feh_max = float(feh.max()) + 0.05 * feh_span
    sigma_grid_1d = np.logspace(log_sigma_min, log_sigma_max, 480)
    feh_grid_1d = np.linspace(feh_min, feh_max, 360)
    sigma_grid, feh_grid = np.meshgrid(sigma_grid_1d, feh_grid_1d)
    contour_model = IMBHModel(IMBHModelConfig(enabled=True, metallicity_kind="z_ratio"))
    imbh_grid = contour_model.imbh_mass_from_sigma_metallicity(sigma_grid, np.power(10.0, feh_grid))

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    if len(np.unique(z_form)) == 1:
        norm = colors.Normalize(vmin=float(z_form[0]) - 0.5, vmax=float(z_form[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(np.nanmin(z_form)), vmax=float(np.nanmax(z_form)))
    scatter = ax.scatter(sigma_h, feh, c=z_form, s=marker_sizes, cmap=mpl.cm.viridis, norm=norm, alpha=0.35, edgecolors="none", rasterized=True, zorder=5)
    finite_grid = imbh_grid[np.isfinite(imbh_grid)]
    if finite_grid.size > 0:
        levels = [level for level in IMBH_CONTOUR_LEVELS if float(finite_grid.min()) <= level <= float(finite_grid.max())]
        if levels:
            contours = ax.contour(sigma_grid, feh_grid, imbh_grid, levels=levels, colors="black", linewidths=1.2, zorder=7)
            ax.clabel(contours, inline=True, fmt=rf"%d $M_\odot$", fontsize=7.0)
    colour_bar = fig.colorbar(scatter, ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Formation redshift z")

    size_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="0.45", markeredgecolor="none", markersize=float(np.sqrt(size)), lw=0, label=rf"${mass:g}\ M_\odot$")
        for mass, size in zip(reference_masses, reference_sizes)
    ]
    ax.legend(handles=size_handles, loc="upper right", fontsize=7.0, frameon=False, title=r"IMBH seed mass")

    # Formation outputs store metallicity as [Fe/H]; the IMBH contour model uses Z/Zsun internally.
    ax.set_xscale("log")
    ax.set_xlim(10.0**log_sigma_min, 10.0**log_sigma_max)
    ax.set_ylim(feh_min, feh_max)
    ax.set_xlabel(r"Initial half-mass surface density $\Sigma_h$ [$M_{\odot}\,{\rm pc}^{-2}$]")
    ax.set_ylabel(r"Initial metallicity [Fe/H]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig03(joined: pd.DataFrame, mass_bin_width_dex: float, add_stellar_mass_axis: bool) -> plt.Figure:
    x_log_col = "logMh_z0_msun"
    plot_rows = joined[
        np.isfinite(joined[x_log_col].to_numpy(dtype=float))
        & np.isfinite(joined["M_SMBH_final"].to_numpy(dtype=float))
    ].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite rows are available for plotting.")

    z_values = np.sort(plot_rows["z_out"].unique())
    edges = _regular_log_bin_edges(plot_rows[x_log_col], step_dex=mass_bin_width_dex)

    if len(z_values) == 1:
        norm = colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        track = plot_rows[plot_rows["z_out"] == float(z_out)].copy()
        binned = _bin_track(track, edges, x_log_col=x_log_col)
        if len(binned) == 0:
            continue
        mean_mass = binned["mean_mass"].to_numpy(dtype=float)
        valid = np.isfinite(mean_mass) & (mean_mass > 0.0)
        if not np.any(valid):
            continue
        x = np.power(10.0, binned.loc[valid, "logx_center"].to_numpy(dtype=float))
        mean_mass = mean_mass[valid]
        std_mass = binned.loc[valid, "std_mass"].to_numpy(dtype=float)
        lower = np.maximum(mean_mass - std_mass, mean_mass * 1.0e-3)
        upper = np.maximum(mean_mass + std_mass, mean_mass * 1.0e-3)
        colour = cmap(norm(float(z_out)))
        ax.fill_between(x, lower, upper, color=colour, alpha=0.18, edgecolor="none")
        ax.plot(x, mean_mass, c=colour, lw=2.0)
        n_tracks += 1

    if n_tracks == 0:
        raise ValueError("All binned central-BH tracks are empty or non-positive.")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"Descendant $z=0$ halo mass [${HALO_MASS_UNIT_LABEL}$]")
    ax.set_ylabel(r"Central BH mass within 1 pc [$M_{\odot}$]")
    ax.set_xlim(left=10.0**edges[0], right=10.0**edges[-1])
    ax.set_ylim(bottom=1.0e2, top=1.0e8)
    if add_stellar_mass_axis:
        _add_smhm_top_axis(ax)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    return fig


def plot_fig04(joined: pd.DataFrame, mass_bin_width_dex: float, observations: pd.DataFrame | None) -> plt.Figure:
    x_log_col = "logMstar_z_smhm_msun"
    plot_rows = joined[
        np.isfinite(joined[x_log_col].to_numpy(dtype=float))
        & np.isfinite(joined["M_SMBH_final"].to_numpy(dtype=float))
    ].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite rows are available for plotting.")

    z_values = np.sort(plot_rows["z_out"].unique())
    edges = _regular_log_bin_edges(plot_rows[x_log_col], step_dex=mass_bin_width_dex)
    x_limit_values = plot_rows[x_log_col].to_numpy(dtype=float)
    if observations is not None and len(observations) > 0:
        x_limit_values = np.concatenate([x_limit_values, observations["logMstar"].to_numpy(dtype=float)])
    x_limit_edges = _regular_log_bin_edges(x_limit_values, step_dex=mass_bin_width_dex)

    if len(z_values) == 1:
        norm = colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        track = plot_rows[plot_rows["z_out"] == float(z_out)].copy()
        binned = _bin_track(track, edges, x_log_col=x_log_col)
        if len(binned) == 0:
            continue
        mean_mass = binned["mean_mass"].to_numpy(dtype=float)
        valid = np.isfinite(mean_mass) & (mean_mass > 0.0)
        if not np.any(valid):
            continue
        x = np.power(10.0, binned.loc[valid, "logx_center"].to_numpy(dtype=float))
        mean_mass = mean_mass[valid]
        std_mass = binned.loc[valid, "std_mass"].to_numpy(dtype=float)
        lower = np.maximum(mean_mass - std_mass, mean_mass * 1.0e-3)
        upper = np.maximum(mean_mass + std_mass, mean_mass * 1.0e-3)
        colour = cmap(norm(float(z_out)))
        ax.fill_between(x, lower, upper, color=colour, alpha=0.18, edgecolor="none")
        ax.plot(x, mean_mass, c=colour, lw=2.0)
        n_tracks += 1

    if n_tracks == 0:
        raise ValueError("All binned central-BH tracks are empty or non-positive.")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"Stellar mass at redshift $z$ from MPB $M_h(z)$ and SMHM [${HALO_MASS_UNIT_LABEL}$]")
    ax.set_ylabel(r"Central BH mass [$M_{\odot}$]")
    ax.set_xlim(left=10.0**x_limit_edges[0], right=10.0**x_limit_edges[-1])

    x_line = np.logspace(x_limit_edges[0], x_limit_edges[-1], 256)
    rv15 = _reines_volonteri_2015_mbh(x_line)
    rv15_scatter = 10.0**REINES_VOLONTERI_2015_SCATTER_DEX
    y_limit_values = [plot_rows["M_SMBH_final"].to_numpy(dtype=float), rv15 * rv15_scatter]
    if observations is not None and len(observations) > 0:
        obs_log_mbh = observations["logMBH"].to_numpy(dtype=float)
        obs_err_hi = observations["logMBH_err_hi"].to_numpy(dtype=float)
        obs_limit = observations["logMBH_upper_limit"].map(_as_bool).to_numpy(dtype=bool)
        obs_log_top = np.where(obs_limit, obs_log_mbh, obs_log_mbh + np.where(np.isfinite(obs_err_hi), np.maximum(obs_err_hi, 0.0), 0.0))
        y_limit_values.append(np.power(10.0, obs_log_top[np.isfinite(obs_log_top)]))
    y_limit_array = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in y_limit_values])
    y_top = 1.0e11
    ax.set_ylim(bottom=1.0e2, top=1.0e11)

    ax.fill_between(x_line, rv15 / rv15_scatter, rv15 * rv15_scatter, color="#2ca25f", alpha=0.16, edgecolor="none", label="Reines+Volonteri 2015", zorder=0)
    ax.plot(x_line, rv15, c="#238b45", lw=1.8, zorder=1)
    for ratio in BH_TO_STELLAR_MASS_RATIOS:
        ratio_label = f"{ratio:g}"
        ax.plot(x_line, ratio * x_line, c="#31a354", ls="--", lw=1.0, alpha=0.75, zorder=2)
        label_x_log = min(float(x_limit_edges[-1]) - 0.35, math.log10(y_top) - math.log10(ratio) - 1.05)
        label_x_log = max(label_x_log, float(x_limit_edges[0]) + 0.45)
        label_x_log = 6.0
        label_x = 10.0**label_x_log
        label_y = ratio * label_x
        if 1.0e2 < label_y < y_top:
            ax.text(label_x, label_y, rf"$M_{{\rm BH}}/M_\ast={ratio_label}$", color="#31a354", fontsize=8.5, rotation=33.0, ha="center", va="bottom", clip_on=True, zorder=3)
    if observations is not None and len(observations) > 0:
        _plot_mbh_mstar_observations(ax, observations)
    ax.legend(loc="lower right", fontsize=6.2, frameon=True, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    return fig


def plot_fig05(joined: pd.DataFrame, observations: pd.DataFrame | None = None) -> plt.Figure:
    if "M_NSC" not in joined.columns:
        raise ValueError(
            "Run output is missing the expected NSC mass column 'M_NSC'. "
            "Regenerate the haloSummaryByZ output with redshift-resolved NSC masses."
        )

    plot_rows = joined[
        np.isfinite(joined["M_NSC"].to_numpy(dtype=float))
        & (joined["M_NSC"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(joined["M_SMBH_final"].to_numpy(dtype=float))
        & (joined["M_SMBH_final"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(joined["z_out"].to_numpy(dtype=float))
    ].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite positive NSC and central-BH masses are available for Fig.05.")

    z_values = np.sort(plot_rows["z_out"].unique())
    if len(z_values) == 1:
        norm = colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    cmap = mpl.cm.viridis
    x_limit_values = [plot_rows["M_NSC"].to_numpy(dtype=float)]
    y_limit_values = [plot_rows["M_SMBH_final"].to_numpy(dtype=float)]
    if observations is not None and len(observations) > 0:
        obs_x = np.power(10.0, observations["log10_mnsc_msun"].to_numpy(dtype=float))
        obs_log_y = observations["log10_mbh_msun"].to_numpy(dtype=float)
        obs_err_hi = observations["log10_mbh_err_hi"].to_numpy(dtype=float)
        obs_limit = observations["mbh_upper_limit"].map(_as_bool).to_numpy(dtype=bool)
        obs_log_y_top = np.where(
            obs_limit,
            obs_log_y,
            obs_log_y + np.where(np.isfinite(obs_err_hi), np.maximum(obs_err_hi, 0.0), 0.0),
        )
        x_limit_values.append(obs_x)
        y_limit_values.append(np.power(10.0, obs_log_y_top[np.isfinite(obs_log_y_top)]))
    x_limit_array = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in x_limit_values])
    y_limit_array = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in y_limit_values])
    x_min = 10.0 ** math.floor(min(5.0, float(np.log10(np.nanmin(x_limit_array)))))
    x_max = 10.0 ** math.ceil(max(9.0, float(np.log10(np.nanmax(x_limit_array)))))
    y_min = 1.0e2
    y_max = 1.0e11

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    _plot_fig05_ratio_guides(ax, x_min, x_max, y_min, y_max)
    scatter = ax.scatter(
        plot_rows["M_NSC"].to_numpy(dtype=float),
        plot_rows["M_SMBH_final"].to_numpy(dtype=float),
        c=plot_rows["z_out"].to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        s=20,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )
    colour_bar = fig.colorbar(scatter, ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(1.0e2, 1.0e11)
    if observations is not None and len(observations) > 0:
        _plot_fig05_kritos_observations(ax, observations)
        ax.legend(frameon=False, loc="upper left", fontsize=7.2, ncol=1)
    ax.set_xlabel(rf"NSC mass at redshift $z$ [${HALO_MASS_UNIT_LABEL}$]")
    ax.set_ylabel(r"Central BH mass [$M_{\odot}$]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig06(final_gc: pd.DataFrame) -> plt.Figure:
    imbh_mass = final_gc["M_IMBH_final"].to_numpy(dtype=float)
    status = final_gc["status"].to_numpy(dtype=int)
    if np.any(np.isfinite(imbh_mass) & (imbh_mass < 0.0)):
        raise ValueError("Final-GC table contains negative M_IMBH_final values.")

    all_bh_mass = imbh_mass[np.isfinite(imbh_mass) & (imbh_mass > 0.0)]
    sunk_mask = np.isin(status, np.asarray(SUNK_BH_STATUSES, dtype=int))
    sunk_bh_mass = imbh_mass[sunk_mask & np.isfinite(imbh_mass) & (imbh_mass > 0.0)]
    if len(all_bh_mass) == 0:
        raise ValueError("No positive final IMBH masses are available for Fig.06.")

    log_mass = np.log10(all_bh_mass)
    log_lo = math.floor(float(np.nanmin(log_mass)) * 4.0) / 4.0
    log_hi = math.ceil(float(np.nanmax(log_mass)) * 4.0) / 4.0
    if log_hi <= log_lo:
        log_hi = log_lo + 0.25
    bins = np.logspace(log_lo, log_hi, int(round((log_hi - log_lo) / 0.25)) + 1)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    counts_all, _, _ = ax.hist(
        all_bh_mass,
        bins=bins,
        histtype="stepfilled",
        color="0.72",
        alpha=0.55,
        edgecolor="0.45",
        linewidth=1.0,
        label=_mass_hist_label("All IMBH rows", all_bh_mass),
    )
    counts_sunk = np.array([], dtype=float)
    if len(sunk_bh_mass) > 0:
        counts_sunk, _, _ = ax.hist(
            sunk_bh_mass,
            bins=bins,
            histtype="step",
            color="black",
            linewidth=1.8,
            label=_mass_hist_label("Sunk IMBH rows", sunk_bh_mass),
        )
    else:
        ax.plot([], [], c="black", lw=1.8, label=_mass_hist_label("Sunk IMBH rows", sunk_bh_mass))

    counts_nonzero = np.concatenate([counts_all[counts_all > 0.0], counts_sunk[counts_sunk > 0.0]])
    if len(counts_nonzero) > 0 and float(np.nanmax(counts_nonzero)) / max(float(np.nanmin(counts_nonzero)), 1.0) >= 20.0:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"Final GC-line IMBH mass [$M_{\odot}$]")
    ax.set_ylabel("Number of BHs per bin")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig07(final_gc: pd.DataFrame, final_redshift: float = 0.0) -> plt.Figure:
    required = ["status", "M_GC_final", "M_IMBH_final", "lookback_time_final_gyr"]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"Final-GC table is missing required Fig.07 column(s): {missing}")

    status_raw = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(status_raw)):
        raise ValueError("Final-GC table contains non-finite status values.")
    status = status_raw.astype(int)
    if np.any(np.abs(status_raw - status.astype(float)) > 1.0e-8):
        raise ValueError("Final-GC table contains non-integer status values.")

    gc_mass = pd.to_numeric(final_gc["M_GC_final"], errors="coerce").to_numpy(dtype=float)
    imbh_mass = pd.to_numeric(final_gc["M_IMBH_final"], errors="coerce").to_numpy(dtype=float)
    lookback = pd.to_numeric(final_gc["lookback_time_final_gyr"], errors="coerce").to_numpy(dtype=float)

    sunk_mask = np.isin(status, np.asarray(SUNK_BH_STATUSES, dtype=int))
    sunk_mask &= status != STATUS_WANDERER
    if np.any(sunk_mask & np.isfinite(gc_mass) & (gc_mass < 0.0)):
        raise ValueError("Final-GC table contains negative M_GC_final for sunk rows.")
    if np.any(sunk_mask & np.isfinite(imbh_mass) & (imbh_mass < 0.0)):
        raise ValueError("Final-GC table contains negative M_IMBH_final for sunk rows.")

    positive_sunk_bh = (
        sunk_mask
        & np.isfinite(gc_mass)
        & (gc_mass >= 0.0)
        & np.isfinite(imbh_mass)
        & (imbh_mass > 0.0)
    )
    if np.any(positive_sunk_bh & ~np.isfinite(lookback)):
        raise ValueError("Final-GC table contains non-finite lookback times for positive sunk IMBH rows.")

    plot_mask = positive_sunk_bh & np.isfinite(lookback)
    if not np.any(plot_mask):
        raise ValueError("No finite sunk rows with positive final IMBH masses are available for Fig.07.")

    sunk_redshift = _redshift_from_final_lookback(lookback[plot_mask], final_redshift=final_redshift)
    plot_rows = pd.DataFrame(
        {
            "status": status[plot_mask],
            "M_GC_final": gc_mass[plot_mask],
            "M_IMBH_final": imbh_mass[plot_mask],
            "sunk_redshift": sunk_redshift,
        }
    )

    z_values = plot_rows["sunk_redshift"].to_numpy(dtype=float)
    if len(np.unique(z_values)) == 1:
        norm = colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(np.nanmin(z_values)), vmax=float(np.nanmax(z_values)))
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    first_scatter = None
    for status_value, marker, label in [
        (STATUS_SUNK_GC, "o", "Sunk GC"),
        (STATUS_SUNK_WANDERER, "s", "Sunk IMBH wanderer"),
    ]:
        rows = plot_rows.loc[plot_rows["status"].to_numpy(dtype=int) == status_value]
        if len(rows) == 0:
            continue
        scatter = ax.scatter(
            rows["M_GC_final"].to_numpy(dtype=float),
            rows["M_IMBH_final"].to_numpy(dtype=float),
            c=rows["sunk_redshift"].to_numpy(dtype=float),
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=20,
            alpha=0.6,
            edgecolors="none",
            rasterized=True,
            label=label,
        )
        if first_scatter is None:
            first_scatter = scatter

    if first_scatter is None:
        raise ValueError("No finite sunk rows with positive final IMBH masses are available for Fig.07.")

    colour_bar = fig.colorbar(first_scatter, ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Sunk redshift z")
    if len(np.unique(z_values)) == 1:
        colour_bar.set_ticks([float(z_values[0])])

    ax.set_xscale("symlog", linthresh=SYMLINTHRESH_MGC_MSUN)
    ax.set_yscale("log")
    ax.set_xlabel(r"Sunk GC stellar mass $M_{\rm GC,sunk}$ [$M_{\odot}$]")
    ax.set_ylabel(r"Sunk IMBH mass $M_{\rm IMBH,sunk}$ [$M_{\odot}$]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def _select_fig09_local_rows(summary_by_z: pd.DataFrame, final_redshift: float) -> pd.DataFrame:
    if "z_out" not in summary_by_z.columns:
        raise ValueError("haloSummaryByZ is missing z_out for Fig.09 local-row selection.")
    redshift = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(redshift) & np.isclose(redshift, float(final_redshift), rtol=0.0, atol=1.0e-8)
    if not np.any(mask):
        raise ValueError(f"No haloSummaryByZ rows match final_redshift={final_redshift:.8g} for Fig.09.")
    rows = summary_by_z.loc[mask].copy()
    bh = pd.to_numeric(rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(bh)) or np.any(bh < 0.0):
        raise ValueError("Fig.09 local rows contain non-finite or negative M_SMBH_final values.")
    return rows


def _fig09_histogram_density(masses: np.ndarray, bins: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    valid = np.isfinite(masses) & (masses > 0.0)
    counts, _ = np.histogram(masses[valid], bins=bins)
    return counts.astype(float) / FIG09_ILLUSTRIS1_DARK_VOLUME_MPC3


def _fig09_plot_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr) | (arr <= 0.0)] = np.nan
    return arr


def plot_fig09(summary_by_z: pd.DataFrame, final_gc: pd.DataFrame, reference: pd.DataFrame, final_redshift: float) -> plt.Figure:
    local_rows = _select_fig09_local_rows(summary_by_z, final_redshift)
    nuclear_mass = pd.to_numeric(local_rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
    nuclear_mass = nuclear_mass[np.isfinite(nuclear_mass) & (nuclear_mass > 0.0)]

    required = ["status", "M_IMBH_final"]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"Final-GC table is missing required Fig.09 column(s): {missing}")
    status_raw = pd.to_numeric(final_gc["status"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(status_raw)):
        raise ValueError("Final-GC table contains non-finite status values for Fig.09.")
    status = status_raw.astype(int)
    if np.any(np.abs(status_raw - status.astype(float)) > 1.0e-8):
        raise ValueError("Final-GC table contains non-integer status values for Fig.09.")
    imbh_mass = pd.to_numeric(final_gc["M_IMBH_final"], errors="coerce").to_numpy(dtype=float)
    if np.any(np.isfinite(imbh_mass) & (imbh_mass < 0.0)):
        raise ValueError("Final-GC table contains negative M_IMBH_final values for Fig.09.")

    ignored_positive = np.isin(status, np.asarray([STATUS_EXHAUSTED, STATUS_TORN], dtype=int)) & np.isfinite(imbh_mass) & (imbh_mass > 0.0)
    if np.any(ignored_positive):
        bad_statuses = sorted(set(status[ignored_positive].astype(int).tolist()))
        raise ValueError(f"Fig.09 found positive IMBH masses in exhausted/torn statuses: {bad_statuses}")

    satellite_mask = np.isin(status, np.asarray(SATELLITE_BH_STATUSES, dtype=int))
    satellite_mass = imbh_mass[satellite_mask & np.isfinite(imbh_mass) & (imbh_mass > 0.0)]
    if len(nuclear_mass) == 0 and len(satellite_mass) == 0:
        raise ValueError("No positive nuclear or satellite BH masses are available for Fig.09.")

    bins = np.asarray(FIG09_BIN_EDGES, dtype=float)
    x = bins[:-1]
    n_nuclear = _fig09_histogram_density(nuclear_mass, bins)
    n_satellite = _fig09_histogram_density(satellite_mass, bins)
    n_total = n_nuclear + n_satellite

    print(
        "Fig.09 inventory: "
        f"z={float(final_redshift):.6g}, nuclear={len(nuclear_mass)}, satellite={len(satellite_mass)}, "
        f"volume={FIG09_ILLUSTRIS1_DARK_VOLUME_MPC3:.6g} Mpc^3."
    )

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    ax.plot(x, _fig09_plot_values(n_total), color="0.55", lw=4.0, alpha=0.75, label="Total", zorder=5)

    ref_x = reference["mbh_msun"].to_numpy(dtype=float)
    ax.plot(ref_x, reference["linear_mpc3"].to_numpy(dtype=float), color="red", lw=1.0, label="Linear", zorder=3)
    ax.fill_between(
        ref_x,
        reference["linear_low_mpc3"].to_numpy(dtype=float),
        reference["linear_high_mpc3"].to_numpy(dtype=float),
        color="red",
        alpha=0.28,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(ref_x, reference["nsc_mpc3"].to_numpy(dtype=float), color="blue", lw=1.0, label="NSC", zorder=3)
    ax.fill_between(
        ref_x,
        reference["nsc_low_mpc3"].to_numpy(dtype=float),
        reference["nsc_high_mpc3"].to_numpy(dtype=float),
        color="blue",
        alpha=0.28,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(x, _fig09_plot_values(n_satellite), color="0.45", lw=2.0, ls="dashdot", label="Satellite", zorder=4)
    ax.plot(x, _fig09_plot_values(n_nuclear), color="0.45", lw=2.0, ls="--", label="Nuclear", zorder=4)

    positive_values = np.concatenate(
        [
            n_total[n_total > 0.0],
            n_satellite[n_satellite > 0.0],
            n_nuclear[n_nuclear > 0.0],
            reference[["linear_mpc3", "linear_low_mpc3", "linear_high_mpc3", "nsc_mpc3", "nsc_low_mpc3", "nsc_high_mpc3"]].to_numpy(dtype=float).ravel(),
        ]
    )
    positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0.0)]
    y_min = min(1.0e-5, 10.0 ** math.floor(float(np.log10(np.nanmin(positive_values)))))
    y_max = 10.0 ** math.ceil(max(-1.0, float(np.log10(np.nanmax(positive_values)))))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e2, 1.0e11)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$M_{\rm BH}/M_{\odot}$")
    ax.set_ylabel(r"$n_{\rm BH}/{\rm Mpc}^{-3}$")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2)
    return fig


def _fig08_redshift_column(summary: pd.DataFrame) -> str:
    for name in ["z_out", "redshift"]:
        if name in summary.columns:
            return name
    raise ValueError("haloSummaryByZ is missing both z_out and redshift columns required for Fig.08.")


def _fig08_bh_column(summary: pd.DataFrame) -> str:
    for name in ["M_SMBH_final", "central_bh_mass_final_msun", "M_BH"]:
        if name in summary.columns:
            return name
    raise ValueError("haloSummaryByZ is missing a central-BH mass column required for Fig.08.")


def _select_fig08_z_rows(summary_by_z: pd.DataFrame) -> pd.DataFrame:
    redshift_column = _fig08_redshift_column(summary_by_z)
    bh_column = _fig08_bh_column(summary_by_z)
    redshift = pd.to_numeric(summary_by_z[redshift_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(redshift) & (np.abs(redshift - FIG08_TARGET_REDSHIFT) < FIG08_REDSHIFT_ATOL)
    if not np.any(mask):
        raise ValueError(
            f"No haloSummaryByZ rows satisfy |z - {FIG08_TARGET_REDSHIFT:.2f}| < {FIG08_REDSHIFT_ATOL:.2f}; "
            "Fig.08 requires the z~7 output."
        )

    rows = summary_by_z.loc[mask].copy()
    if "halo_id_z0" not in rows.columns:
        raise ValueError("haloSummaryByZ is missing halo_id_z0 for Fig.08.")
    for column in rows.columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.sort_values("halo_id_z0").reset_index(drop=True)
    rows["halo_id_z0"] = rows["halo_id_z0"].astype(int)
    if rows["halo_id_z0"].duplicated().any():
        dupes = sorted(rows.loc[rows["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].unique().tolist())
        raise ValueError(f"Fig.08 z selection produced duplicate halo rows: {dupes[:10]}")

    bh = rows[bh_column].to_numpy(dtype=float)
    if np.any(~np.isfinite(bh)) or np.any(bh < 0.0):
        raise ValueError(f"Fig.08 selected rows contain non-finite or negative {bh_column} values.")
    return rows


def _fig08_error_arrays(rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    xerr = rows[["r_err_low_pc", "r_err_high_pc"]].to_numpy(dtype=float).T
    yerr = rows[["v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float).T
    xerr = np.where(np.isfinite(xerr), xerr, 0.0)
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    return xerr, yerr


def _plot_fig08_observed_curve(ax: plt.Axes, curves: pd.DataFrame, curve_name: str, label: str, **kwargs: object) -> None:
    used_label = False
    for sign in [-1.0, 1.0]:
        sign_rows = curves.loc[curves["curve"].eq(curve_name) & (np.sign(curves["r_pc"].to_numpy(dtype=float)) == sign)]
        if len(sign_rows) == 0:
            continue
        ordered = sign_rows.sort_values("r_pc")
        ax.plot(
            ordered["r_pc"].to_numpy(dtype=float),
            ordered["v_km_s"].to_numpy(dtype=float),
            label=label if not used_label else None,
            **kwargs,
        )
        used_label = True


def _fig08_signed_profile(radius_pc: np.ndarray, velocity_km_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(radius_pc, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    return np.concatenate([-radius[::-1], radius]), np.concatenate([-velocity[::-1], velocity])


def _fig08_total_velocity_profiles(deposit_profile, z_rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bh_column = _fig08_bh_column(z_rows)
    bh_by_halo = {
        int(row.halo_id_z0): float(getattr(row, bh_column))
        for row in z_rows[["halo_id_z0", bh_column]].itertuples(index=False)
    }
    summary_halos = set(bh_by_halo)
    profile_halos = set(int(value) for value in np.asarray(deposit_profile.halo_ids, dtype=int))
    missing_profiles = sorted(summary_halos - profile_halos)
    extra_profiles = sorted(profile_halos - summary_halos)
    if missing_profiles:
        raise ValueError(f"Fig.08 deposit profile is missing selected halo_id_z0 values: {missing_profiles[:10]}")
    if extra_profiles:
        raise ValueError(f"Fig.08 deposit profile contains halo_id_z0 values outside the selected z rows: {extra_profiles[:10]}")
    if deposit_profile.cumulative_mass_msun is None:
        raise ValueError("Fig.08 requires cumulative deposited stellar masses.")

    first_radii_pc = [float(rout[0]) * 1.0e3 for rout in deposit_profile.r_outer_kpc]
    last_radii_pc = [float(rout[-1]) * 1.0e3 for rout in deposit_profile.r_outer_kpc]
    r_min = max(1.0, max(first_radii_pc))
    r_max = min(FIG08_RADIUS_MAX_PC, min(last_radii_pc))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max < FIG08_MATCH_RADIUS_RANGE_PC[1]:
        raise ValueError(
            f"Fig.08 deposit radial coverage is insufficient: common range is {r_min:.6g}-{r_max:.6g} pc, "
            f"but {FIG08_MATCH_RADIUS_RANGE_PC[1]:.1f} pc is required."
        )

    radius_pc = np.unique(
        np.concatenate(
            [
                np.geomspace(r_min, r_max, 256),
                np.asarray(FIG08_MATCH_RADIUS_RANGE_PC, dtype=float),
            ]
        )
    )
    stellar_cumulative: list[np.ndarray] = []
    velocity_profiles: list[np.ndarray] = []
    ordered_halo_ids = np.asarray(deposit_profile.halo_ids, dtype=int)
    for hid, rout_kpc, cumulative in zip(ordered_halo_ids, deposit_profile.r_outer_kpc, deposit_profile.cumulative_mass_msun):
        rout_pc = np.asarray(rout_kpc, dtype=float) * 1.0e3
        cum_mass = np.asarray(cumulative, dtype=float)
        if np.any(~np.isfinite(cum_mass)) or np.any(cum_mass < 0.0):
            raise ValueError(f"Fig.08 cumulative deposited stellar mass is non-finite or negative for halo_id_z0={int(hid)}.")
        if radius_pc[0] < rout_pc[0] or radius_pc[-1] > rout_pc[-1]:
            raise ValueError(f"Fig.08 common radius grid exceeds deposit coverage for halo_id_z0={int(hid)}.")
        stellar = np.interp(radius_pc, rout_pc, cum_mass)
        total_mass = stellar + bh_by_halo[int(hid)]
        if np.any(~np.isfinite(total_mass)) or np.any(total_mass < 0.0):
            raise ValueError(f"Fig.08 total enclosed mass is non-finite or negative for halo_id_z0={int(hid)}.")
        velocity = FIG08_VELOCITY_SIN_I * np.sqrt(FIG08_G_PC_MSUN_KMS2 * total_mass / radius_pc)
        if np.any(~np.isfinite(velocity)):
            raise ValueError(f"Fig.08 velocity profile is non-finite for halo_id_z0={int(hid)}.")
        stellar_cumulative.append(stellar)
        velocity_profiles.append(velocity)

    return ordered_halo_ids, radius_pc, np.asarray(stellar_cumulative, dtype=float), np.asarray(velocity_profiles, dtype=float)


def _fig08_best_halo(
    halo_ids: np.ndarray,
    radius_pc: np.ndarray,
    velocity_profiles: np.ndarray,
    stellar_cumulative: np.ndarray,
    z_rows: pd.DataFrame,
    curves: pd.DataFrame,
) -> dict[str, float]:
    bh_column = _fig08_bh_column(z_rows)
    redshift_column = _fig08_redshift_column(z_rows)
    target_rows = curves.loc[curves["curve"].eq("point_mass_keplerian") & (curves["r_pc"].to_numpy(dtype=float) > 0.0)].sort_values("r_pc")
    if len(target_rows) == 0:
        raise ValueError("Juodzbalis+2026 Fig.2 point-mass curve has no positive-r rows for Fig.08 matching.")
    target_r = target_rows["r_pc"].to_numpy(dtype=float)
    target_v = target_rows["v_km_s"].to_numpy(dtype=float)
    lo, hi = FIG08_MATCH_RADIUS_RANGE_PC
    match_mask = (radius_pc >= lo) & (radius_pc <= hi)
    if not np.any(match_mask):
        raise ValueError(f"Fig.08 has no simulation radii inside the match range {lo}-{hi} pc.")
    match_r = radius_pc[match_mask]
    target_interp = np.interp(match_r, target_r, target_v)
    if np.any(target_interp <= 0.0) or np.any(~np.isfinite(target_interp)):
        raise ValueError("Fig.08 target point-mass curve is non-positive or non-finite over the match range.")

    info_by_halo = z_rows.set_index("halo_id_z0")
    scores: list[dict[str, float]] = []
    for index, hid in enumerate(halo_ids):
        profile = velocity_profiles[index, match_mask]
        if np.any(profile <= 0.0) or np.any(~np.isfinite(profile)):
            continue
        curve_rms = float(np.sqrt(np.mean((np.log10(profile) - np.log10(target_interp)) ** 2)))
        bh_mass = float(info_by_halo.loc[int(hid), bh_column])
        if np.isfinite(bh_mass) and bh_mass > 0.0:
            log_bh = float(np.log10(bh_mass))
            mass_penalty = abs(log_bh - FIG08_QSO1_LOGMBH_TARGET)
            score = curve_rms + FIG08_MATCH_MASS_WEIGHT * mass_penalty
        else:
            log_bh = np.nan
            mass_penalty = np.inf
            score = np.inf
        stellar_150 = float(np.interp(150.0, radius_pc, stellar_cumulative[index]))
        scores.append(
            {
                "index": float(index),
                "halo_id_z0": float(hid),
                "score": float(score),
                "curve_rms_dex": curve_rms,
                "mass_penalty_dex": float(mass_penalty),
                "central_bh_mass_msun": bh_mass,
                "log10_central_bh_mass": log_bh,
                "deposited_stellar_mass_150pc_msun": stellar_150,
                "redshift": float(info_by_halo.loc[int(hid), redshift_column]),
            }
        )
    if not scores:
        raise ValueError("No finite Fig.08 halo velocity profiles were available for best-halo matching.")
    finite_scores = [row for row in scores if np.isfinite(row["score"])]
    if not finite_scores:
        raise ValueError("No Fig.08 candidate has a finite score; check central-BH masses and velocity profiles.")
    return min(finite_scores, key=lambda row: row["score"])


def plot_fig08(fig2_obs, z_rows: pd.DataFrame, deposit_profile) -> plt.Figure:
    halo_ids, radius_pc, stellar_cumulative, velocity_profiles = _fig08_total_velocity_profiles(deposit_profile, z_rows)
    best = _fig08_best_halo(halo_ids, radius_pc, velocity_profiles, stellar_cumulative, z_rows, fig2_obs.curves)
    best_index = int(best["index"])

    median_velocity = np.median(velocity_profiles, axis=0)
    mean_velocity = np.mean(velocity_profiles, axis=0)
    low_velocity, high_velocity = np.percentile(velocity_profiles, FIG08_SCATTER_PERCENTILES, axis=0)
    max_bh = float(np.max(z_rows[_fig08_bh_column(z_rows)].to_numpy(dtype=float)))
    target_bh = 10.0**FIG08_QSO1_LOGMBH_TARGET
    z_values = z_rows[_fig08_redshift_column(z_rows)].to_numpy(dtype=float)
    print(
        "Fig.08 z selection: "
        f"N={len(z_rows)}, |z - {FIG08_TARGET_REDSHIFT:.2f}| < {FIG08_REDSHIFT_ATOL:.2f}, "
        f"z range={float(np.min(z_values)):.3f}-{float(np.max(z_values)):.3f}, "
        f"max central BH={max_bh:.6g} Msun, QSO1 target={target_bh:.6g} Msun."
    )
    print(
        "Fig.08 best halo: "
        f"halo_id_z0={int(best['halo_id_z0'])}, z={best['redshift']:.3f}, "
        f"central BH={best['central_bh_mass_msun']:.6g} Msun "
        f"(log10={best['log10_central_bh_mass']:.3f}), "
        f"deposited stellar mass within 150 pc={best['deposited_stellar_mass_150pc_msun']:.6g} Msun, "
        f"curve RMS={best['curve_rms_dex']:.4f} dex, score={best['score']:.4f}."
    )

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    curves = fig2_obs.curves
    _plot_fig08_observed_curve(
        ax,
        curves,
        "point_mass_keplerian",
        r"QSO1 point mass, $\log M_{\rm BH}=6.75$",
        color="black",
        linewidth=1.7,
        zorder=4,
    )
    _plot_fig08_observed_curve(
        ax,
        curves,
        "mw_nsc",
        "MW-like NSC model",
        color="0.45",
        linestyle="dashdot",
        linewidth=1.5,
        zorder=3,
    )

    points = fig2_obs.points
    point_styles = [
        ("resolved_kinematics", "tab:blue", "o", 5.0, "Resolved kinematics"),
        ("spectroastrometry", "magenta", "X", 6.0, "Spectroastrometry"),
        ("spectroastrometry_fine", "orchid", "P", 5.5, "Fine spectroastrometry"),
    ]
    for component, colour, marker, size, label in point_styles:
        rows = points.loc[points["component"].eq(component)]
        if len(rows) == 0:
            continue
        xerr, yerr = _fig08_error_arrays(rows)
        ax.errorbar(
            rows["r_pc"].to_numpy(dtype=float),
            rows["v_km_s"].to_numpy(dtype=float),
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=size,
            color=colour,
            ecolor=colour,
            elinewidth=1.0,
            markeredgecolor=colour,
            markerfacecolor=colour,
            capsize=0.0,
            linestyle="none",
            label=label,
            zorder=6,
        )

    ax.fill_between(radius_pc, low_velocity, high_velocity, color="tab:green", alpha=0.16, linewidth=0.0, label="z~7 stack 16-84%")
    ax.fill_between(-radius_pc[::-1], -high_velocity[::-1], -low_velocity[::-1], color="tab:green", alpha=0.16, linewidth=0.0)
    signed_r, signed_median = _fig08_signed_profile(radius_pc, median_velocity)
    _, signed_mean = _fig08_signed_profile(radius_pc, mean_velocity)
    _, signed_best = _fig08_signed_profile(radius_pc, velocity_profiles[best_index])
    ax.plot(signed_r, signed_median, color="tab:green", linewidth=1.8, label="z~7 median simulation")
    ax.plot(signed_r, signed_mean, color="tab:green", linewidth=1.2, linestyle="--", label="z~7 mean simulation")
    ax.plot(
        signed_r,
        signed_best,
        color="tab:red",
        linewidth=1.5,
        linestyle="-",
        label=f"Best halo {int(best['halo_id_z0'])}",
    )

    ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.set_xlim(-FIG08_RADIUS_MAX_PC, FIG08_RADIUS_MAX_PC)
    ax.set_ylim(-72.0, 72.0)
    ax.set_xlabel(r"Projected radius $r$ [pc]")
    ax.set_ylabel(r"Line-of-sight velocity $v$ [km s$^{-1}$]")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2, ncol=1)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IMBH seed diagnostics, central-BH halo tracks, SMHM comparison points, NSC-BH mass points, the QSO1 z~7 rotation-curve comparison, and the local BH mass function from one local High-z SMBHs output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory containing ns*/allcat_ns*.txt, allcat_s-*.txt, and ns*/haloSummaryByZ_ns*.csv.")
    parser.add_argument("--ns-value", type=float, default=NS_VALUE_DEFAULT, help="Single N_s value used for all figures; Fig.01 and Fig.02 intentionally use only this per-N_s formation catalogue.")
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.5, help="Log10 halo-mass bin width.")
    parser.add_argument("--no-smhm-top-axis", action="store_true", help="Do not add the SMHM stellar-mass top x-axis to Fig.03.")
    parser.add_argument("--no-cliff-observations", action="store_true", help="Do not overlay Cliff Fig.14 observational points on Fig.04.")
    parser.add_argument("--no-juodzbalis2026-observations", action="store_true", help="Do not overlay Juodzbalis et al. 2026 Fig.4 observational points on Fig.04.")
    parser.add_argument("--no-kritos2025-fig10-observations", action="store_true", help="Do not overlay Kritos et al. 2025 Fig.10 observational points on Fig.05.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = default_plot_dir(out_dir, "Kong+2026")
    plot_dir.mkdir(parents=True, exist_ok=True)

    _apply_plot_style()
    model = build_kong_model(out_dir, args.ns_value)
    formation = model.formation
    final_gc = model.final_gc
    joined = model.summary_by_z
    metadata = load_run_metadata(out_dir)
    final_redshift = float(metadata.get("final_redshift", 0.0))
    if not np.isfinite(final_redshift) or final_redshift < 0.0:
        raise ValueError(f"run_metadata final_redshift must be finite and non-negative, got {final_redshift!r}.")
    fig04_observation_tables = []
    if not args.no_cliff_observations:
        fig04_observation_tables.append(load_cliff_fig14_observations().table)
    if not args.no_juodzbalis2026_observations:
        fig04_observation_tables.append(load_juodzbalis2026_fig4_observations().table)
    fig04_observations = pd.concat(fig04_observation_tables, ignore_index=True) if fig04_observation_tables else None
    fig05_observations = None if args.no_kritos2025_fig10_observations else load_kritos2025_fig10_mbh_mnsc_observations().table
    fig09_reference = load_kritos2025_fig9_mass_functions().table

    fig01 = plot_fig01(formation)
    path01 = plot_dir / FIGURE_01_FILENAME
    fig01.savefig(path01, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig01)
    print(f"Saved {path01}")

    fig02 = plot_fig02(formation)
    path02 = plot_dir / FIGURE_02_FILENAME
    fig02.savefig(path02, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig02)
    print(f"Saved {path02}")

    fig03 = plot_fig03(
        joined,
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        add_stellar_mass_axis=SMHM_TOP_AXIS_DEFAULT and not args.no_smhm_top_axis,
    )
    path03 = plot_dir / FIGURE_03_FILENAME
    fig03.savefig(path03, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig03)
    print(f"Saved {path03}")

    # Fig.04 uses same-redshift MPB halo masses stored by my/run.py.
    fig04 = plot_fig04(
        joined,
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        observations=fig04_observations,
    )
    path04 = plot_dir / FIGURE_04_FILENAME
    fig04.savefig(path04, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig04)
    print(f"Saved {path04}")

    fig05 = plot_fig05(joined, observations=fig05_observations)
    path05 = plot_dir / FIGURE_05_FILENAME
    fig05.savefig(path05, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig05)
    print(f"Saved {path05}")

    fig06 = plot_fig06(final_gc)
    path06 = plot_dir / FIGURE_06_FILENAME
    fig06.savefig(path06, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig06)
    print(f"Saved {path06}")

    fig07 = plot_fig07(final_gc, final_redshift=final_redshift)
    path07 = plot_dir / FIGURE_07_FILENAME
    fig07.savefig(path07, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig07)
    print(f"Saved {path07}")

    if model.paths.deposit is None:
        raise FileNotFoundError("Fig.08 requires the per-N_s deposit profile, but no deposit path was configured.")
    fig2_obs = load_juodzbalis2026_fig2_rotation_curve()
    fig08_z_rows = _select_fig08_z_rows(joined)
    fig08_deposit_profile = load_deposit_profile_for_redshift_summary(
        model.paths.deposit,
        fig08_z_rows,
        final_redshift=final_redshift,
    )
    fig08 = plot_fig08(fig2_obs, fig08_z_rows, fig08_deposit_profile)
    path08 = plot_dir / FIGURE_08_FILENAME
    fig08.savefig(path08, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig08)
    print(f"Saved {path08}")

    fig09 = plot_fig09(joined, final_gc, fig09_reference, final_redshift=final_redshift)
    path09 = plot_dir / FIGURE_09_FILENAME
    fig09.savefig(path09, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig09)
    print(f"Saved {path09}")


if __name__ == "__main__":
    main()
