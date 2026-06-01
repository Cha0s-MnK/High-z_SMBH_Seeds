#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Small plotting-only helpers shared by the paper figure scripts."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THREAD_CAP_DEFAULT = str(min(64, max(1, os.cpu_count() or 1)))
TIMES_COMPATIBLE_FONT_PATHS = (
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf"),
)


def set_thread_env() -> None:
    for env_name in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(env_name, THREAD_CAP_DEFAULT)


def use_agg_backend() -> None:
    mpl.use("Agg")


def times_serif_fonts() -> list[str]:
    for font_path in TIMES_COMPATIBLE_FONT_PATHS:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
    serif_fonts: list[str] = []
    for family in ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Times"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        serif_fonts.append(family)
    return serif_fonts or ["DejaVu Serif"]


def apply_style(font: str = "serif", tex: bool = True, grid: bool = False) -> None:
    use_tex = bool(tex) and shutil.which("latex") is not None
    if font == "serif":
        font_family: str | list[str] = "serif"
        font_serif = times_serif_fonts()
    else:
        font_family = font
        font_serif = plt.rcParams.get("font.serif", ["DejaVu Serif"])
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": font_serif,
            "font.size": 10,
            "mathtext.default": "regular",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": bool(grid),
            "text.usetex": use_tex,
        }
    )
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{bm}"


def plot_dir(out_dir: Path, suite_name: str) -> Path:
    return Path(out_dir).resolve() / f"_plots_{suite_name}"


def test_plot_dir(out_dir: Path, suite_name: str) -> Path:
    return Path(out_dir).resolve() / f"_test_plots_{suite_name}"


def save_pdf(fig: plt.Figure, path: Path, dpi: int, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def finish_axis(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    grid: bool = True,
    legend: bool = False,
    legend_kwargs: dict | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
    if legend:
        kwargs = {"frameon": False, "loc": "best", "ncol": 1}
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)
    ax.tick_params(direction="in", right=True, top=True, which="both")


def finish_log_axis(ax: plt.Axes, **kwargs) -> None:
    kwargs.setdefault("xscale", "log")
    kwargs.setdefault("yscale", "log")
    finish_axis(ax, **kwargs)


def hide_unused_axes(axes: Iterable[plt.Axes]) -> None:
    for ax in axes:
        ax.axis("off")


def plot_band(ax: plt.Axes, x, y, ylo, yhi, colour, label: str | None = None, alpha: float = 0.18, **kwargs) -> None:
    ax.fill_between(x, ylo, yhi, color=colour, alpha=alpha, edgecolor="none")
    ax.plot(x, y, c=colour, label=label, **kwargs)


def plot_binned_track(
    ax: plt.Axes,
    table: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "median",
    lo_col: str = "q25",
    hi_col: str = "q75",
    colour: str = "black",
    label: str | None = None,
    alpha: float = 0.18,
    **kwargs,
) -> None:
    x = table[x_col].to_numpy(dtype=float)
    y = table[y_col].to_numpy(dtype=float)
    ylo = table[lo_col].to_numpy(dtype=float)
    yhi = table[hi_col].to_numpy(dtype=float)
    plot_band(ax, x, y, ylo, yhi, colour, label=label, alpha=alpha, **kwargs)


def plot_error_points(ax: plt.Axes, x, y, xerr=None, yerr=None, **kwargs) -> None:
    defaults = {"fmt": "o", "capsize": 3}
    defaults.update(kwargs)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, **defaults)


def plot_reference_lines(ax: plt.Axes, refs: Sequence[dict], scale: float = 1.0) -> None:
    for ref in refs:
        if "x" in ref:
            ax.axvline(scale * float(ref["x"]), **{k: v for k, v in ref.items() if k != "x"})
        elif "y" in ref:
            ax.axhline(scale * float(ref["y"]), **{k: v for k, v in ref.items() if k != "y"})


def regular_log_bin_edges(values: Iterable[float], step_dex: float) -> np.ndarray:
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


def binned_mean_std(x, y, bins, min_count: int = 1) -> pd.DataFrame:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include_right = idx == len(bins) - 2
        mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr >= left)
        mask &= (x_arr <= right) if include_right else (x_arr < right)
        if int(mask.sum()) < int(min_count):
            continue
        rows.append({"x": 0.5 * (left + right), "count": int(mask.sum()), "mean": float(np.mean(y_arr[mask])), "std": float(np.std(y_arr[mask]))})
    return pd.DataFrame(rows)


def binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count: int = 1) -> pd.DataFrame:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rows = []
    labels = [f"q{int(p)}" for p in percentiles]
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include_right = idx == len(bins) - 2
        mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr >= left)
        mask &= (x_arr <= right) if include_right else (x_arr < right)
        if int(mask.sum()) < int(min_count):
            continue
        row = {"x": 0.5 * (left + right), "count": int(mask.sum())}
        values = np.percentile(y_arr[mask], percentiles)
        row.update({label: float(value) for label, value in zip(labels, values)})
        rows.append(row)
    return pd.DataFrame(rows)


def log_error_to_linear(log_value: float, err_lo: float, err_hi: float) -> np.ndarray | None:
    if not np.isfinite(log_value):
        return None
    value = 10.0**log_value
    lo = float(err_lo) if np.isfinite(err_lo) and err_lo > 0.0 else 0.0
    hi = float(err_hi) if np.isfinite(err_hi) and err_hi > 0.0 else 0.0
    if lo <= 0.0 and hi <= 0.0:
        return None
    return np.array([[value - 10.0 ** (log_value - lo)], [10.0 ** (log_value + hi) - value]], dtype=float)


def finite_positive(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr > 0.0)
