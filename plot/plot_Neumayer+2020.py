#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Reproduce Neumayer et al. (2020) Figures 3, 12, and 13 from local outputs.

This script is intentionally standalone. It reads one local model output directory
and cached observational source tables under
``/home/subonan/High-z_SMBH_Seeds/data/Neumayer+2020``. It does not modify the
existing run or plotting pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List, Tuple
import warnings

import matplotlib as mpl
from matplotlib import font_manager

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

mpl.use("Agg")

from astropy.table import Table, join, vstack
from astropy.units import UnitsWarning
from astropy.utils.metadata import MergeConflictWarning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", category=MergeConflictWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import NSC_RAD_PC, STD_DPI  # noqa: E402
from load_obs import (  # noqa: E402
    load_neumayer_fig03_observations,
    load_neumayer_fig13_observations,
    load_neumayer_observations,
)
from load_output import build_neumayer_model  # noqa: E402
from plot_common import plot_dir as default_plot_dir  # noqa: E402

DEFAULT_OBS_CACHE_DIR = PROJECT_ROOT / "data" / "Neumayer+2020"
if not DEFAULT_OBS_CACHE_DIR.is_dir():
    DEFAULT_OBS_CACHE_DIR = PROJECT_ROOT.parent / "data" / "Neumayer+2020"

NS_VALUE_DEFAULT = 2.0
FIGURE_03_FILENAME = "Fig.03_galaxy_demographics.pdf"
FIGURE_12_FILENAME = "Fig.12_nsc_scaling.pdf"
FIGURE_13_FILENAME = "Fig.13_bh_nsc_mass_ratio.pdf"
RAW_SUBDIR = "raw"
ORIGINAL_REVIEW_SUBDIR = "original_nsc_review"
FIG13_SOURCE_FILENAME = "bh_nsc_galmass.csv"
COMPILED_FIG03_CSV = "neumayer2020_fig03_demographics.csv"
COMPILED_FIG03_META_JSON = "neumayer2020_fig03_demographics_meta.json"
COMPILED_OBS_CSV = "neumayer2020_fig12_compilation.csv"
COMPILED_OBS_META_JSON = "neumayer2020_fig12_compilation_meta.json"
RUN_METADATA_NAME = "run_metadata.json"
HALO_TREE_LOOKUP_NAME = "halo_tree_lookup.csv"
FULL_PHYSICS_COUNTERPARTS_NAME = "full_physics_counterparts_z0.csv"
NEUMAYER_DIVIDER_NAME = "neumayer2020_fig3_divider.json"
OBS_HOST_TYPE_COLOURS = {"late": "tab:blue", "early": "tab:red"}
MODEL_HOST_TYPE_COLOURS = {"late": "dodgerblue", "early": "firebrick"}
FIG13_OBS_COLOURS = {"late": "b", "early": "r", "ucd": "m"}
FIG13_MARKER_SIZE = 100
FIG13_AXIS_MARGIN = 0.1

PAPER_FULL_FIT = (0.48, 6.51)
PAPER_GOOD_FIT = (0.92, 6.13)
TIMES_COMPATIBLE_FONT_PATHS = (
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf"),
)

REQUIRED_RAW_FILES = [
    "README.md",
    "sanchez-janssen19_tab4.tex",
    "sanchez-janssen19_tab5.tex",
    "ordenes-briceno18_tab1.tex",
    "eigenthaler18_tab1.fits",
    "georgiev16.fits",
    "georgiev14_tab1plus.fits",
    "georgiev14_tab2.fits",
    "spengler17_tab8.dat",
    "erwin12_tab2.tex",
    "additional_goodmass.dat",
    "lauer05_alltab.fits",
]


@dataclass
class DepositProfile:
    halo_ids: np.ndarray
    r_outer_kpc: List[np.ndarray]
    cumulative_mass_msun: List[np.ndarray]


@dataclass
class ObsCatalog:
    table: pd.DataFrame
    cache_dir: Path
    metadata: Dict[str, object]


@dataclass
class Fig03Catalog:
    table: pd.DataFrame
    cache_dir: Path
    metadata: Dict[str, object]


@dataclass
class Fig13Catalog:
    table: pd.DataFrame
    source_path: Path
    duplicate_names: List[str]
    missing_host_mass_count: int
    nonfinite_mass_count: int
    unknown_galtype_count: int
    ucd_upper_limit_count: int


@dataclass
class ModelSummary:
    table: pd.DataFrame
    ns_value: float
    nsc_radius_pc: float
    fit_slope: float
    fit_intercept: float
    divider: Dict[str, object] | None = None
    mixed_suite: bool = False


def _apply_plot_style() -> None:
    plt.style.use("default")
    serif_fonts = _available_times_serif_fonts()
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
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


def _safe_log10(arr: np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    return np.log10(np.clip(np.asarray(arr, dtype=float), floor, None))


def _ns_tag(ns_value: float) -> str:
    return f"{float(ns_value):.1f}".replace(".", "p")


def _regular_log_bin_edges(values: Iterable[float], step_dex: float = 0.5) -> np.ndarray:
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


def _expanded_linear_limits(base: Tuple[float, float], values: Iterable[float], margin: float) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(base[0]), float(base[1])
    lo = min(float(base[0]), float(vals.min()) - float(margin))
    hi = max(float(base[1]), float(vals.max()) + float(margin))
    if hi <= lo:
        hi = lo + 2.0 * float(margin)
    return lo, hi


def _binned_percentiles(x_log: pd.Series | np.ndarray, y: pd.Series | np.ndarray, bins: np.ndarray, min_count: int = 5) -> pd.DataFrame:
    x_arr = np.asarray(x_log, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include_right = idx == len(bins) - 2
        mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr >= left)
        mask &= (x_arr <= right) if include_right else (x_arr < right)
        if int(mask.sum()) < int(min_count):
            continue
        y_sel = y_arr[mask]
        rows.append(
            {
                "x": 0.5 * (left + right),
                "count": int(mask.sum()),
                "q25": float(np.percentile(y_sel, 25.0)),
                "median": float(np.percentile(y_sel, 50.0)),
                "q75": float(np.percentile(y_sel, 75.0)),
            }
        )
    return pd.DataFrame(rows)


def _occupation_fraction_summary(x_log: pd.Series | np.ndarray, has_nsc: pd.Series | np.ndarray, bins: np.ndarray, min_count: int = 1) -> pd.DataFrame:
    x_arr = np.asarray(x_log, dtype=float)
    nsc_arr = np.asarray(has_nsc, dtype=bool)
    rows = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include_right = idx == len(bins) - 2
        mask = np.isfinite(x_arr) & (x_arr >= left)
        mask &= (x_arr <= right) if include_right else (x_arr < right)
        total = int(mask.sum())
        if total < int(min_count):
            continue
        n_nsc = int(nsc_arr[mask].sum())
        fraction = float(n_nsc / total)
        err = float(math.sqrt(fraction * (1.0 - fraction) / total)) if total > 0 else np.nan
        rows.append(
            {
                "x": 0.5 * (left + right),
                "count": total,
                "n_nsc": n_nsc,
                "fraction": fraction,
                "err": err,
            }
        )
    return pd.DataFrame(rows)


def _model_output_root_from_allcat_path(allcat_path: Path) -> Path:
    parent = allcat_path.parent
    if parent.name.startswith("ns"):
        return parent.parent
    return parent


def _deposit_mass_within_radius(profile: DepositProfile, radius_kpc: float, halo_ids: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if halo_ids is None:
        use = np.ones(len(profile.halo_ids), dtype=bool)
    else:
        use = np.isin(profile.halo_ids, np.asarray(halo_ids, dtype=int))
    halo_use = profile.halo_ids[use]
    if len(halo_use) == 0:
        return halo_use, np.array([], dtype=float)
    vals = np.zeros(len(halo_use), dtype=float)
    use_idx = np.where(use)[0]
    for jj, ii in enumerate(use_idx):
        rout = np.asarray(profile.r_outer_kpc[ii], dtype=float)
        cum = np.asarray(profile.cumulative_mass_msun[ii], dtype=float)
        vals[jj] = float(np.interp(float(radius_kpc), rout, cum, left=0.0, right=cum[-1]))
    return halo_use, vals


def build_figure_03(model: ModelSummary, fig03_obs: Fig03Catalog) -> Tuple[plt.Figure, Dict[str, object]]:
    _apply_plot_style()

    obs_table = fig03_obs.table.dropna(subset=["logMstar_gal", "g_minus_i"]).copy()
    obs_table["has_nsc"] = obs_table["has_nsc"].astype(bool)
    bins = np.asarray(fig03_obs.metadata.get("occupation_bins", np.arange(5.5, 12.5, 0.7)), dtype=float)
    slope = float(fig03_obs.metadata.get("divider_slope", 0.12))
    intercept = float(fig03_obs.metadata.get("divider_intercept", -0.32))
    split_host_types = model.mixed_suite

    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=STD_DPI, figsize=(10.8, 4.8))
    ax_left, ax_right = axes

    x_line = np.linspace(5.5, 12.0, 400, dtype=float)
    y_line = slope * x_line + intercept
    ax_left.fill_between(x_line, y_line, np.zeros_like(x_line) + 1.55, color="tab:red", alpha=0.14, linewidth=0.0)
    ax_left.fill_between(x_line, np.zeros_like(x_line) - 1.4, y_line, color="tab:blue", alpha=0.14, linewidth=0.0)
    non_nucleated = obs_table.loc[~obs_table["has_nsc"]].copy()
    nucleated = obs_table.loc[obs_table["has_nsc"]].copy()
    ax_left.scatter(non_nucleated["logMstar_gal"].to_numpy(dtype=float), non_nucleated["g_minus_i"].to_numpy(dtype=float), facecolors="white", edgecolors="black", s=22, alpha=0.75, linewidths=0.5, label="No NSC detected")
    ax_left.scatter(nucleated["logMstar_gal"].to_numpy(dtype=float), nucleated["g_minus_i"].to_numpy(dtype=float), facecolors="black", edgecolors="black", s=20, alpha=0.55, linewidths=0.0, label="With NSC")
    ax_left.plot(x_line, y_line, c="black", lw=1.8, label="Blue/red divider")
    ax_left.text(6.0, 1.28, "red sequence", color="tab:red", fontsize=9)
    ax_left.text(6.0, -1.08, "blue cloud", color="tab:blue", fontsize=9)

    for host_type, colour, label in [("early", OBS_HOST_TYPE_COLOURS["early"], "Obs early-type"), ("late", OBS_HOST_TYPE_COLOURS["late"], "Obs late-type")]:
        subset = obs_table.loc[obs_table["host_type_fig3"] == host_type].copy()
        summary = _occupation_fraction_summary(subset["logMstar_gal"], subset["has_nsc"], bins, min_count=1)
        if summary.empty:
            continue
        x = summary["x"].to_numpy(dtype=float)
        fraction = summary["fraction"].to_numpy(dtype=float)
        err = summary["err"].to_numpy(dtype=float)
        ax_right.fill_between(x, np.clip(fraction - err, 0.0, 1.0), np.clip(fraction + err, 0.0, 1.0), color=colour, alpha=0.18, linewidth=0.0)
        ax_right.plot(x, fraction, c=colour, lw=2.2, marker="o", markersize=4, label=label)

    model_table = model.table.copy()
    model_mask = np.isfinite(model_table["logMstar_plot"].to_numpy(dtype=float))
    model_rows = model_table.loc[model_mask].copy()
    model_rows["has_nsc_model"] = model_rows["M_NSC"].to_numpy(dtype=float) > 0.0

    if split_host_types:
        unmatched = model_rows.loc[~model_rows["host_type_fig3"].isin(["early", "late"])].copy()
        unmatched_summary = _occupation_fraction_summary(unmatched["logMstar_plot"], unmatched["has_nsc_model"], bins, min_count=1)
        if not unmatched_summary.empty:
            ax_right.plot(unmatched_summary["x"].to_numpy(dtype=float), unmatched_summary["fraction"].to_numpy(dtype=float), c="0.45", lw=1.6, ls=":", marker="s", markersize=4, label="Model unmatched")
        for host_type, colour, label in [("early", MODEL_HOST_TYPE_COLOURS["early"], "Model early-type"), ("late", MODEL_HOST_TYPE_COLOURS["late"], "Model late-type")]:
            subset = model_rows.loc[model_rows["host_type_fig3"] == host_type].copy()
            summary = _occupation_fraction_summary(subset["logMstar_plot"], subset["has_nsc_model"], bins, min_count=1)
            if summary.empty:
                continue
            ax_right.plot(summary["x"].to_numpy(dtype=float), summary["fraction"].to_numpy(dtype=float), c=colour, lw=2.0, ls="--", marker="s", markersize=4, label=label)
    else:
        model_summary = _occupation_fraction_summary(model_rows["logMstar_plot"], model_rows["has_nsc_model"], bins, min_count=1)
        if not model_summary.empty:
            ax_right.plot(model_summary["x"].to_numpy(dtype=float), model_summary["fraction"].to_numpy(dtype=float), c="black", lw=2.0, ls="--", marker="s", markersize=4, label="Model")

    x_limits = _expanded_linear_limits((5.5, 12.0), list(obs_table["logMstar_gal"].to_numpy(dtype=float)) + list(model_rows["logMstar_plot"].to_numpy(dtype=float)), 0.1)
    y_limits = _expanded_linear_limits((-1.4, 1.55), obs_table["g_minus_i"].to_numpy(dtype=float), 0.05)
    ax_left.set_xlim(*x_limits)
    ax_left.set_ylim(*y_limits)
    ax_left.set_xlabel(r"$\log_{10}(M_{\star}/M_{\odot})$")
    ax_left.set_ylabel(r"$(g-i)_0$")
    ax_left.grid(True, alpha=0.3, linestyle=":", which="both")
    ax_left.legend(frameon=False, loc="best", ncol=1, fontsize=8)

    ax_right.set_xlim(*x_limits)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xlabel(r"$\log_{10}(M_{\star}/M_{\odot})$")
    ax_right.set_ylabel("Fraction of galaxies with NSC")
    ax_right.grid(True, alpha=0.3, linestyle=":", which="both")
    ax_right.legend(frameon=False, loc="best", ncol=1, fontsize=8)

    summary = {
        "n_obs_total": int(len(obs_table)),
        "n_obs_nucleated": int(obs_table["has_nsc"].sum()),
        "n_obs_non_nucleated": int((~obs_table["has_nsc"]).sum()),
        "n_obs_late": int((obs_table["host_type_fig3"] == "late").sum()),
        "n_obs_early": int((obs_table["host_type_fig3"] == "early").sum()),
        "n_model_total": int(len(model_rows)),
        "n_model_nucleated": int(model_rows["has_nsc_model"].sum()),
        "split_host_types": int(split_host_types),
        "n_model_late": int((model_rows["host_type_fig3"] == "late").sum()),
        "n_model_early": int((model_rows["host_type_fig3"] == "early").sum()),
        "n_model_unmatched": int((~model_rows["host_type_fig3"].isin(["late", "early"])).sum()),
        "divider_slope": slope,
        "divider_intercept": intercept,
    }
    return fig, summary


def build_figure_12(model: ModelSummary, obs: ObsCatalog) -> Tuple[plt.Figure, Dict[str, object]]:
    _apply_plot_style()

    obs_table = obs.table.dropna(subset=["logMstar_gal", "logM_nsc"]).copy()
    obs_bins = _regular_log_bin_edges(obs_table["logMstar_gal"], 0.5)

    model_table = model.table.copy()
    model_fit_mask = np.isfinite(model_table["logMstar_plot"]) & np.isfinite(model_table["logM_NSC"]) & (model_table["M_NSC"] > 0.0)
    model_points = model_table.loc[model_fit_mask].copy()
    model_bins = _regular_log_bin_edges(model_points["logMstar_plot"], 0.5)
    model_ratio_summary = _binned_percentiles(model_points["logMstar_plot"], model_points["f_NSC_plot"], model_bins, min_count=5)
    split_host_types = model.mixed_suite

    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=STD_DPI, figsize=(10.8, 4.8))
    ax_left, ax_right = axes

    late_obs = obs_table.loc[obs_table["host_type"] == "late"].copy()
    early_obs = obs_table.loc[obs_table["host_type"] == "early"].copy()
    late_good = late_obs.loc[late_obs["is_high_quality"]].copy()
    early_good = early_obs.loc[early_obs["is_high_quality"]].copy()
    ax_left.scatter(np.power(10.0, late_obs["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, late_obs["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["late"], s=18, alpha=0.35, linewidths=0.0)
    ax_left.scatter(np.power(10.0, early_obs["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, early_obs["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["early"], s=18, alpha=0.35, linewidths=0.0)
    ax_left.scatter(np.power(10.0, late_good["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, late_good["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["late"], s=130, alpha=0.95, marker="*", edgecolors="black", linewidths=0.35)
    ax_left.scatter(np.power(10.0, early_good["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, early_good["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["early"], s=130, alpha=0.95, marker="*", edgecolors="black", linewidths=0.35)

    if split_host_types:
        late_model = model_points.loc[model_points["host_type_fig3"] == "late"].copy()
        early_model = model_points.loc[model_points["host_type_fig3"] == "early"].copy()
        unmatched_model = model_points.loc[~model_points["host_type_fig3"].isin(["late", "early"])].copy()
        if len(unmatched_model) > 0:
            ax_left.scatter(unmatched_model["M_star_plot"].to_numpy(dtype=float), unmatched_model["M_NSC"].to_numpy(dtype=float), c="0.65", s=18, alpha=0.25, marker="s", linewidths=0.0)
        if len(late_model) > 0:
            ax_left.scatter(late_model["M_star_plot"].to_numpy(dtype=float), late_model["M_NSC"].to_numpy(dtype=float), c=MODEL_HOST_TYPE_COLOURS["late"], s=18, alpha=0.45, marker="s", linewidths=0.0)
        if len(early_model) > 0:
            ax_left.scatter(early_model["M_star_plot"].to_numpy(dtype=float), early_model["M_NSC"].to_numpy(dtype=float), c=MODEL_HOST_TYPE_COLOURS["early"], s=18, alpha=0.45, marker="s", linewidths=0.0)
    else:
        ax_left.scatter(model_points["M_star_plot"].to_numpy(dtype=float), model_points["M_NSC"].to_numpy(dtype=float), c="0.25", s=18, alpha=0.45, marker="s", linewidths=0.0)

    left_x_log_limits = _expanded_linear_limits(
        (5.5, 11.2),
        list(obs_table["logMstar_gal"].to_numpy(dtype=float)) + list(model_points["logMstar_plot"].to_numpy(dtype=float)),
        0.1,
    )
    left_y_log_limits = _expanded_linear_limits(
        (4.5, 9.1),
        list(obs_table["logM_nsc"].to_numpy(dtype=float)) + list(model_points["logM_NSC"].to_numpy(dtype=float)),
        0.1,
    )
    model_log_fraction = np.log10(model_points.loc[model_points["f_NSC_plot"] > 0.0, "f_NSC_plot"].to_numpy(dtype=float))
    right_y_log_limits = _expanded_linear_limits(
        (-4.0, 0.0),
        list(obs_table["log_fraction"].to_numpy(dtype=float)) + list(model_log_fraction),
        0.1,
    )
    x_line = np.logspace(left_x_log_limits[0], left_x_log_limits[1], 300)
    y_paper = np.power(10.0, PAPER_FULL_FIT[0] * (np.log10(x_line) - 9.0) + PAPER_FULL_FIT[1])
    y_good = np.power(10.0, PAPER_GOOD_FIT[0] * (np.log10(x_line) - 9.0) + PAPER_GOOD_FIT[1])
    y_model = np.power(10.0, model.fit_slope * (np.log10(x_line) - 9.0) + model.fit_intercept)
    ax_left.plot(x_line, y_paper, c="black", ls="--", lw=1.8)
    ax_left.plot(x_line, y_good, c="black", ls="-.", lw=1.8)
    ax_left.plot(x_line, y_model, c="black", ls="-", lw=2.0)

    for host_type, colour in [("late", OBS_HOST_TYPE_COLOURS["late"]), ("early", OBS_HOST_TYPE_COLOURS["early"])]:
        subset = obs_table.loc[obs_table["host_type"] == host_type].copy()
        summary = _binned_percentiles(subset["logMstar_gal"], np.power(10.0, subset["log_fraction"]), obs_bins, min_count=5)
        if summary.empty:
            continue
        ax_right.fill_between(np.power(10.0, summary["x"]), summary["q25"], summary["q75"], facecolor=colour, edgecolor="none", linewidth=0.0, alpha=0.2)
        ax_right.plot(np.power(10.0, summary["x"]), summary["median"], c=colour, lw=2.0)

    if split_host_types:
        for host_type, colour in [("late", MODEL_HOST_TYPE_COLOURS["late"]), ("early", MODEL_HOST_TYPE_COLOURS["early"])]:
            subset = model_points.loc[model_points["host_type_fig3"] == host_type].copy()
            summary = _binned_percentiles(subset["logMstar_plot"], subset["f_NSC_plot"], model_bins, min_count=5)
            if summary.empty:
                continue
            ax_right.fill_between(np.power(10.0, summary["x"]), summary["q25"], summary["q75"], facecolor=colour, edgecolor="none", linewidth=0.0, alpha=0.15)
            ax_right.plot(np.power(10.0, summary["x"]), summary["median"], c=colour, lw=2.0)
    else:
        if not model_ratio_summary.empty:
            ax_right.fill_between(np.power(10.0, model_ratio_summary["x"]), model_ratio_summary["q25"], model_ratio_summary["q75"], facecolor="0.55", edgecolor="none", linewidth=0.0, alpha=0.25)
            ax_right.plot(np.power(10.0, model_ratio_summary["x"]), model_ratio_summary["median"], c="black", lw=2.0)
    ax_right.plot(x_line, y_paper / x_line, c="black", ls="--", lw=1.8)

    left_handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=OBS_HOST_TYPE_COLOURS["late"], markersize=6, alpha=0.7, label="Obs late-type"),
        mpl.lines.Line2D([], [], marker="o", ls="", color=OBS_HOST_TYPE_COLOURS["early"], markersize=6, alpha=0.7, label="Obs early-type"),
        mpl.lines.Line2D([], [], marker="*", ls="", color="black", markersize=10, label="Obs dyn/spec subset"),
    ]
    if split_host_types:
        left_handles.extend(
            [
                mpl.lines.Line2D([], [], marker="s", ls="", color=MODEL_HOST_TYPE_COLOURS["late"], markersize=6, alpha=0.7, label="Model late (Fig.3 colour cut)"),
                mpl.lines.Line2D([], [], marker="s", ls="", color=MODEL_HOST_TYPE_COLOURS["early"], markersize=6, alpha=0.7, label="Model early (Fig.3 colour cut)"),
            ]
        )
        if len(unmatched_model) > 0:
            left_handles.append(mpl.lines.Line2D([], [], marker="s", ls="", color="0.65", markersize=6, alpha=0.45, label="Model unmatched"))
    else:
        left_handles.append(mpl.lines.Line2D([], [], marker="s", ls="", color="0.25", markersize=6, alpha=0.7, label="Model"))
    left_handles.extend(
        [
            mpl.lines.Line2D([], [], c="black", ls="--", lw=1.8, label="Paper fit"),
            mpl.lines.Line2D([], [], c="black", ls="-.", lw=1.8, label="Paper dyn/spec fit"),
            mpl.lines.Line2D([], [], c="black", ls="-", lw=2.0, label=f"Model fit ({int(model.nsc_radius_pc)} pc proxy)"),
        ]
    )
    ax_left.legend(handles=left_handles, frameon=False, loc="upper left", ncol=1, fontsize=8)

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_left.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_left.set_xlabel(r"$M_{\star,\mathrm{gal}}\,[M_{\odot}]$")
    ax_left.set_ylabel(r"$M_{\mathrm{NSC}}\,[M_{\odot}]$")
    ax_left.set_xlim(10.0 ** left_x_log_limits[0], 10.0 ** left_x_log_limits[1])
    ax_left.set_ylim(10.0 ** left_y_log_limits[0], 10.0 ** left_y_log_limits[1])
    ax_left.grid(True, alpha=0.3, linestyle=":", which="both")

    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_right.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_right.set_xlabel(r"$M_{\star,\mathrm{gal}}\,[M_{\odot}]$")
    ax_right.set_ylabel(r"$M_{\mathrm{NSC}}/M_{\star,\mathrm{gal}}$")
    ax_right.set_xlim(10.0 ** left_x_log_limits[0], 10.0 ** left_x_log_limits[1])
    ax_right.set_ylim(10.0 ** right_y_log_limits[0], 10.0 ** right_y_log_limits[1])
    ax_right.grid(True, alpha=0.3, linestyle=":", which="both")
    right_handles = [
        mpl.lines.Line2D([], [], c=OBS_HOST_TYPE_COLOURS["late"], lw=2.0, label="Obs late-type median"),
        mpl.lines.Line2D([], [], c=OBS_HOST_TYPE_COLOURS["early"], lw=2.0, label="Obs early-type median"),
    ]
    if split_host_types:
        right_handles.extend(
            [
                mpl.patches.Patch(facecolor=MODEL_HOST_TYPE_COLOURS["late"], edgecolor="none", alpha=0.15, label="Model late IQR"),
                mpl.lines.Line2D([], [], c=MODEL_HOST_TYPE_COLOURS["late"], lw=2.0, label="Model late median"),
                mpl.patches.Patch(facecolor=MODEL_HOST_TYPE_COLOURS["early"], edgecolor="none", alpha=0.15, label="Model early IQR"),
                mpl.lines.Line2D([], [], c=MODEL_HOST_TYPE_COLOURS["early"], lw=2.0, label="Model early median"),
            ]
        )
    else:
        right_handles.extend(
            [
                mpl.patches.Patch(facecolor="0.55", edgecolor="none", alpha=0.25, label="Model IQR"),
                mpl.lines.Line2D([], [], c="black", lw=2.0, label="Model median"),
            ]
        )
    right_handles.append(mpl.lines.Line2D([], [], c="black", ls="--", lw=1.8, label=r"Paper fit / $M_{\star}$"))
    ax_right.legend(handles=right_handles, frameon=False, loc="lower left", ncol=1, fontsize=8)

    summary = {
        "obs_full_fit_slope_from_cache": float(obs.metadata["fit_full_slope"]),
        "obs_full_fit_intercept_from_cache": float(obs.metadata["fit_full_intercept"]),
        "obs_high_quality_fit_slope_from_cache": float(obs.metadata["fit_high_quality_slope"]),
        "obs_high_quality_fit_intercept_from_cache": float(obs.metadata["fit_high_quality_intercept"]),
        "model_fit_slope": float(model.fit_slope),
        "model_fit_intercept": float(model.fit_intercept),
        "n_obs_fit_rows": int(obs.metadata["n_fit_rows"]),
        "n_model_halos": int(len(model_points)),
        "nsc_proxy_radius_pc": float(model.nsc_radius_pc),
        "mixed_suite": int(model.mixed_suite),
        "split_host_types": int(split_host_types),
        "n_model_late": int((model_points["host_type_fig3"] == "late").sum()),
        "n_model_early": int((model_points["host_type_fig3"] == "early").sum()),
        "n_model_unmatched": int((~model_points["host_type_fig3"].isin(["late", "early"])).sum()),
    }
    return fig, summary


def build_figure_13(model: ModelSummary, obs: Fig13Catalog) -> Tuple[plt.Figure, Dict[str, object]]:
    _apply_plot_style()

    obs_table = obs.table.loc[obs.table["plot_keep"].astype(bool)].copy()
    model_table = model.table.copy()
    model_valid = (
        np.isfinite(model_table["logMstar_z0"].to_numpy(dtype=float))
        & (model_table["M_BH"].to_numpy(dtype=float) > 0.0)
        & (model_table["M_NSC"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(model_table["logM_NSC"].to_numpy(dtype=float))
        & np.isfinite(model_table["log_bh_to_nsc"].to_numpy(dtype=float))
    )
    model_points = model_table.loc[model_valid].copy()
    if len(model_points) == 0:
        raise ValueError("No finite Figure 13 model rows remain after excluding zero-BH, zero-NSC, or missing-host rows.")

    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=STD_DPI, figsize=(10.8, 4.8))
    ax_left, ax_right = axes

    detections = (~obs_table["bh_is_upper_limit"]) & (~obs_table["nsc_is_upper_limit"])
    late = obs_table["host_type"] == "late"
    early = obs_table["host_type"] == "early"
    ucd = obs_table["host_type"] == "ucd"
    bh_upper = obs_table["bh_is_upper_limit"].astype(bool)
    nsc_upper = obs_table["nsc_is_upper_limit"].astype(bool)
    finite_host = np.isfinite(obs_table["logMstar_gal"].to_numpy(dtype=float))

    ax_left.axhline(0.0, c="black", ls="--", alpha=0.3)
    ax_right.axhline(0.0, c="black", ls="--", alpha=0.3)

    late_bh_left = late & bh_upper & finite_host
    early_bh_left = early & bh_upper & finite_host
    ax_left.quiver(obs_table.loc[late_bh_left, "logMstar_gal"], obs_table.loc[late_bh_left, "log_bh_to_nsc"], np.zeros(int(late_bh_left.sum())), -np.ones(int(late_bh_left.sum())), color=FIG13_OBS_COLOURS["late"], alpha=0.4)
    ax_left.quiver(obs_table.loc[early_bh_left, "logMstar_gal"], obs_table.loc[early_bh_left, "log_bh_to_nsc"], np.zeros(int(early_bh_left.sum())), -np.ones(int(early_bh_left.sum())), color=FIG13_OBS_COLOURS["early"], alpha=0.4)
    late_nsc_left = late & nsc_upper & finite_host
    early_nsc_left = early & nsc_upper & finite_host
    ax_left.quiver(obs_table.loc[late_nsc_left, "logMstar_gal"], obs_table.loc[late_nsc_left, "log_bh_to_nsc"], np.zeros(int(late_nsc_left.sum())), np.ones(int(late_nsc_left.sum())), color=FIG13_OBS_COLOURS["late"], alpha=0.4)
    ax_left.quiver(obs_table.loc[early_nsc_left, "logMstar_gal"], obs_table.loc[early_nsc_left, "log_bh_to_nsc"], np.zeros(int(early_nsc_left.sum())), np.ones(int(early_nsc_left.sum())), color=FIG13_OBS_COLOURS["early"], alpha=0.4)

    late_bh_right = late & bh_upper
    early_bh_right = early & bh_upper
    ax_right.quiver(obs_table.loc[late_bh_right, "logM_nsc"], obs_table.loc[late_bh_right, "log_bh_to_nsc"], np.zeros(int(late_bh_right.sum())), -np.ones(int(late_bh_right.sum())), color=FIG13_OBS_COLOURS["late"], alpha=0.4)
    ax_right.quiver(obs_table.loc[early_bh_right, "logM_nsc"], obs_table.loc[early_bh_right, "log_bh_to_nsc"], np.zeros(int(early_bh_right.sum())), -np.ones(int(early_bh_right.sum())), color=FIG13_OBS_COLOURS["early"], alpha=0.4)
    late_nsc_right = late & nsc_upper
    early_nsc_right = early & nsc_upper
    ax_right.quiver(obs_table.loc[late_nsc_right, "logM_nsc"], obs_table.loc[late_nsc_right, "log_bh_to_nsc"], -np.ones(int(late_nsc_right.sum())), np.ones(int(late_nsc_right.sum())), color=FIG13_OBS_COLOURS["late"], alpha=0.4)
    ax_right.quiver(obs_table.loc[early_nsc_right, "logM_nsc"], obs_table.loc[early_nsc_right, "log_bh_to_nsc"], -np.ones(int(early_nsc_right.sum())), np.ones(int(early_nsc_right.sum())), color=FIG13_OBS_COLOURS["early"], alpha=0.4)

    for host_type, colour in [("late", FIG13_OBS_COLOURS["late"]), ("early", FIG13_OBS_COLOURS["early"])]:
        left_mask = (obs_table["host_type"] == host_type) & detections & finite_host
        right_mask = (obs_table["host_type"] == host_type) & detections
        ax_left.scatter(obs_table.loc[left_mask, "logMstar_gal"], obs_table.loc[left_mask, "log_bh_to_nsc"], color=colour, s=FIG13_MARKER_SIZE, alpha=0.7, linewidths=0.0)
        ax_right.scatter(obs_table.loc[right_mask, "logM_nsc"], obs_table.loc[right_mask, "log_bh_to_nsc"], color=colour, s=FIG13_MARKER_SIZE, alpha=0.7, linewidths=0.0)
    ucd_det = ucd & detections
    ax_right.scatter(obs_table.loc[ucd_det, "logM_nsc"], obs_table.loc[ucd_det, "log_bh_to_nsc"], color=FIG13_OBS_COLOURS["ucd"], s=FIG13_MARKER_SIZE, alpha=0.7, linewidths=0.0)

    unmatched_model = model_points.loc[~model_points["host_type_fig3"].isin(["late", "early"])].copy()
    if len(unmatched_model) > 0:
        ax_left.scatter(unmatched_model["logMstar_z0"], unmatched_model["log_bh_to_nsc"], c="0.55", s=FIG13_MARKER_SIZE, alpha=1.0, marker="s", edgecolors="none", linewidths=0.0)
        ax_right.scatter(unmatched_model["logM_NSC"], unmatched_model["log_bh_to_nsc"], c="0.55", s=FIG13_MARKER_SIZE, alpha=1.0, marker="s", edgecolors="none", linewidths=0.0)
    for host_type, colour in [("late", MODEL_HOST_TYPE_COLOURS["late"]), ("early", MODEL_HOST_TYPE_COLOURS["early"])]:
        subset = model_points.loc[model_points["host_type_fig3"] == host_type].copy()
        if len(subset) == 0:
            continue
        ax_left.scatter(subset["logMstar_z0"], subset["log_bh_to_nsc"], c=colour, s=FIG13_MARKER_SIZE, alpha=1.0, marker="s", edgecolors="none", linewidths=0.0)
        ax_right.scatter(subset["logM_NSC"], subset["log_bh_to_nsc"], c=colour, s=FIG13_MARKER_SIZE, alpha=1.0, marker="s", edgecolors="none", linewidths=0.0)

    left_obs_mask = ((late | early) & finite_host)
    right_obs_mask = late | early | ucd
    left_x_limits = _expanded_linear_limits((8.6, 11.9), list(obs_table.loc[left_obs_mask, "logMstar_gal"]) + list(model_points["logMstar_z0"]), FIG13_AXIS_MARGIN)
    left_y_limits = _expanded_linear_limits((-4.0, 4.0), list(obs_table.loc[left_obs_mask, "log_bh_to_nsc"]) + list(model_points["log_bh_to_nsc"]), FIG13_AXIS_MARGIN)
    right_x_limits = _expanded_linear_limits((5.5, 8.5), list(obs_table.loc[right_obs_mask, "logM_nsc"]) + list(model_points["logM_NSC"]), FIG13_AXIS_MARGIN)
    right_y_limits = _expanded_linear_limits((-4.0, 4.0), list(obs_table.loc[right_obs_mask, "log_bh_to_nsc"]) + list(model_points["log_bh_to_nsc"]), FIG13_AXIS_MARGIN)

    ax_left.set_xlim(*left_x_limits)
    ax_left.set_ylim(*left_y_limits)
    ax_right.set_xlim(*right_x_limits)
    ax_right.set_ylim(*right_y_limits)
    ax_left.set_xlabel(r"$\log_{10}(M_{\star,\mathrm{gal}}/M_{\odot})$")
    ax_right.set_xlabel(r"$\log_{10}(M_{\mathrm{NSC}}/M_{\odot})$")
    ax_left.set_ylabel(r"$\log_{10}(M_{\mathrm{BH}}/M_{\mathrm{NSC}})$")
    ax_right.set_ylabel(r"$\log_{10}(M_{\mathrm{BH}}/M_{\mathrm{NSC}})$")
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.tick_params(direction="in", right=True, top=True, which="both")

    bh_limit_handle = mpl.lines.Line2D([], [], marker=r"$\downarrow$", ls="", color="0.25", markersize=10, label="BH upper limit")
    nsc_limit_handle = mpl.lines.Line2D([], [], marker=r"$\uparrow$", ls="", color="0.45", markersize=10, label="NSC upper limit")
    left_handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=FIG13_OBS_COLOURS["late"], markersize=7, alpha=0.7, label="Obs late-type"),
        mpl.lines.Line2D([], [], marker="o", ls="", color=FIG13_OBS_COLOURS["early"], markersize=7, alpha=0.7, label="Obs early-type"),
        bh_limit_handle,
        nsc_limit_handle,
        mpl.lines.Line2D([], [], marker="s", ls="", color=MODEL_HOST_TYPE_COLOURS["late"], markersize=7, label="Model late-type"),
        mpl.lines.Line2D([], [], marker="s", ls="", color=MODEL_HOST_TYPE_COLOURS["early"], markersize=7, label="Model early-type"),
    ]
    if len(unmatched_model) > 0:
        left_handles.append(mpl.lines.Line2D([], [], marker="s", ls="", color="0.55", markersize=7, label="Model unmatched"))
    right_handles = list(left_handles)
    if int(ucd_det.sum()) > 0:
        right_handles.insert(2, mpl.lines.Line2D([], [], marker="o", ls="", color=FIG13_OBS_COLOURS["ucd"], markersize=7, alpha=0.7, label="Obs UCDs"))
    ax_left.legend(handles=left_handles, frameon=False, loc="best", ncol=1, fontsize=8)
    ax_right.legend(handles=right_handles, frameon=False, loc="best", ncol=1, fontsize=8)

    zero_bh = model_table["M_BH"].to_numpy(dtype=float) <= 0.0
    zero_nsc = model_table["M_NSC"].to_numpy(dtype=float) <= 0.0
    missing_host = ~np.isfinite(model_table["logMstar_z0"].to_numpy(dtype=float))
    summary = {
        "n_obs_rows": int(len(obs.table)),
        "duplicate_names": obs.duplicate_names,
        "missing_host_mass_count": int(obs.missing_host_mass_count),
        "nonfinite_mass_count": int(obs.nonfinite_mass_count),
        "unknown_galtype_count": int(obs.unknown_galtype_count),
        "ucd_upper_limit_count": int(obs.ucd_upper_limit_count),
        "n_model_plotted": int(len(model_points)),
        "n_model_zero_bh": int(zero_bh.sum()),
        "n_model_zero_nsc": int(zero_nsc.sum()),
        "n_model_missing_host": int(missing_host.sum()),
        "bh_mass_column": "M_SMBH_final",
    }
    return fig, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Neumayer+2020 Figures 3, 12, and 13 from one local High-z SMBHs output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory containing allcat/ns*/deposit products.")
    parser.add_argument("--ns-value", type=float, default=NS_VALUE_DEFAULT, help="Sersic N_s value to load from the ns* subdirectory.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = default_plot_dir(out_dir, "Neumayer+2020")
    plot_dir.mkdir(parents=True, exist_ok=True)

    obs = load_neumayer_observations()
    fig03_obs = load_neumayer_fig03_observations()
    fig13_obs = load_neumayer_fig13_observations()
    model = build_neumayer_model(out_dir, args.ns_value, NSC_RAD_PC)

    fig03, summary03 = build_figure_03(model, fig03_obs)
    figure03_path = plot_dir / FIGURE_03_FILENAME
    fig03.savefig(figure03_path, bbox_inches="tight")
    plt.close(fig03)
    print(
        f"Saved {figure03_path} | "
        f"n_obs={summary03['n_obs_total']} | "
        f"n_obs_nsc={summary03['n_obs_nucleated']} | "
        f"n_model={summary03['n_model_total']}"
    )

    fig12, summary12 = build_figure_12(model, obs)
    figure12_path = plot_dir / FIGURE_12_FILENAME
    fig12.savefig(figure12_path, bbox_inches="tight")
    plt.close(fig12)
    print(
        f"Saved {figure12_path} | "
        f"n_obs_fit={summary12['n_obs_fit_rows']} | "
        f"n_model={summary12['n_model_halos']}"
    )

    fig13, summary13 = build_figure_13(model, fig13_obs)
    figure13_path = plot_dir / FIGURE_13_FILENAME
    fig13.savefig(figure13_path, bbox_inches="tight")
    plt.close(fig13)
    summary13["saved_pdf_path"] = str(figure13_path)
    duplicates = ", ".join(summary13["duplicate_names"]) if summary13["duplicate_names"] else "none"
    print(
        f"Saved {figure13_path} | "
        f"n_obs={summary13['n_obs_rows']} | "
        f"duplicates={duplicates} | "
        f"missing_host_obs={summary13['missing_host_mass_count']} | "
        f"skipped_nonfinite_obs={summary13['nonfinite_mass_count']} | "
        f"skipped_unknown_galtype={summary13['unknown_galtype_count']} | "
        f"skipped_ucd_upper={summary13['ucd_upper_limit_count']} | "
        f"n_model={summary13['n_model_plotted']} | "
        f"excluded_zero_bh={summary13['n_model_zero_bh']} | "
        f"excluded_zero_nsc={summary13['n_model_zero_nsc']} | "
        f"excluded_missing_host={summary13['n_model_missing_host']} | "
        f"bh_mass_column={summary13['bh_mass_column']}"
    )


if __name__ == "__main__":
    main()
