#!/usr/bin/env python3

"""
Batch driver for the Python rewrite of Kong+2026 GC evolution.

This workflow uses the bundled project ``data/`` directory, with an optional
override for the fixed-tree input directory:

- ``fixed_trees_large_spin`` (halo trees)
- ``mass_loss.txt`` (stellar-evolution mass-loss table)
- ``snaps2redshifts.txt`` (snapshot-redshift table)

The script performs three major steps:
1. Run ``src/main.py`` per Sersic index ``N_s`` to build fresh GC formation catalogs from raw trees.
2. Evolve each catalog halo-by-halo with ``src/evo.py`` physics.
3. Write plotting-ready outputs consumed by the paper-specific plot scripts in ``plot/``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = THIS_FILE.parent
MAIN_SPATIAL_PATH = SRC_DIR / "main.py"
EVO_PATH = SRC_DIR / "evo.py"
PLOT_GAO2023_PATH = PROJECT_ROOT / "plot" / "plot_Gao+2024.py"
PLOT_CHOKSI2018_PATH = PROJECT_ROOT / "plot" / "plot_Choksi+2018.py"
PLOT_NEUMAYER2020_PATH = PROJECT_ROOT / "plot" / "plot_Neumayer+2020.py"
PLOT_KONG2026_PATH = PROJECT_ROOT / "plot" / "plot_Kong+2026.py"
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_ROOT.parent / "data"
SNAPS_PATH = DATA_DIR / "snaps2redshifts.txt"
MASS_LOSS_PATH = DATA_DIR / "mass_loss.txt"
DEFAULT_TREE_DIR = DATA_DIR / "fixed_trees_large_spin"
TREE_LOOKUP_BASENAME = "id_lookup_large_dark.csv"
for _required_path, _label, _kind in (
    (SRC_DIR, "source directory", "dir"),
    (MAIN_SPATIAL_PATH, "formation script", "file"),
    (EVO_PATH, "evolution module", "file"),
):
    if _kind == "dir" and (not _required_path.is_dir()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing {_label}: {_required_path}"
        )
    if _kind == "file" and (not _required_path.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing {_label}: {_required_path}"
        )
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evo import (  # noqa: E402
    DEPOS_HEADER,
    STAT_ALIVE,
    STAT_EXHAUSTED,
    STAT_SUNK,
    STAT_TORN,
    STAT_WANDERER,
    STAT_WANDERER_SUNK,
    evolve_single_halo,
    read_haloevo_mpb,
)
from config import *  # noqa: E402

NS_VALUES_DEFAULT = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
OUT_Z_DEFAULT = "1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0"

FINAL_GC_HEADER = "\n".join([
    ("hid_z0 logMh_z0 subfind_form logMh_form logMstar_form logMgas_form "
     "logM_form zform feh r_galaxy_kpc gc_radius_pc sigma_h_msun_pc2 M_IMBH_init"),
    "rows: one formed GC per row; this is the per-halo format evolution input table",])

ALLCAT_HEADER = "\n".join([
    ("hid_z0 logMh_z0 logMstar_z0 logMh_form logMstar_form logM_form "
     "zform feh isMPB subfind_form snap_form r_galaxy_kpc "
     "gc_radius_pc sigma_h_msun_pc2 M_IMBH_init"),
    "rows: one formed GC per row; companion finalGCs_ns files use the same row ordering",])
ALLCAT_FMT = [
    "%.0f",    # hid_z0
    "%.10e",  # logMh_z0
    "%.10e",  # logMstar_z0
    "%.10e",  # logMh_form
    "%.10e",  # logMstar_form
    "%.10e",  # logM_form
    "%.10f",  # zform
    "%.10e",  # feh
    "%.0f",   # isMPB
    "%.0f",   # subfind_form
    "%.0f",   # snap_form
    "%.10e",  # r_galaxy_kpc
    "%.10e",  # gc_radius_pc
    "%.10e",  # sigma_h_msun_pc2
    "%.10e",  # M_IMBH_init
]

COMBINED_FINAL_GC_HEADER = "\n".join(
    [("halo_id_z0 gc_index_halo status M_GC_final "
      "m_init_msun lookback_time_final_gyr lookback_time_init_gyr "
      "r_final_kpc r_init_kpc gc_radius_pc sigma_h_msun_pc2 feh "
      "M_IMBH_init M_IMBH_final"),
     ("rows: one GC row per allcat_ns row for this N_s; feh and "
      "the GC/IMBH columns are fixed at formation."),])

COMBINED_DEPOS_HEADER = "\n".join([
    "halo_id_z0 lookback_time_gyr bin_index r_inner_kpc r_outer_kpc m_depo_total_msun m_star_no_evo_msun m_star_with_evo_msun",
    "rows: one deposited radial-bin row from the per-halo Depos files for this N_s; halo_id_z0 identifies the source halo",])

GLOBAL_FINAL_GC_HEADER = "\n".join([
    ("ns halo_id_z0 gc_index_halo status M_GC_final "
     "m_init_msun lookback_time_final_gyr lookback_time_init_gyr "
     "r_final_kpc r_init_kpc gc_radius_pc sigma_h_msun_pc2 feh "
     "M_IMBH_init M_IMBH_final"),
    "rows: one GC row from the per-N_s finalGCs files; ns and halo_id_z0 identify the source run and halo",])

GLOBAL_DEPOS_HEADER = "\n".join([
    "ns halo_id_z0 lookback_time_gyr bin_index r_inner_kpc r_outer_kpc m_depo_total_msun m_star_no_evo_msun m_star_with_evo_msun",
    "rows: one deposited radial-bin row from the per-N_s combined Depos files; ns and halo_id_z0 identify the source run and halo",])

HALO_SUMMARY_COLUMNS = [
    "hid_z0",
    "logMh_z0",
    "n_gc_total",
    "n_alive",
    "n_wanderer",
    "n_exhausted",
    "n_torn",
    "n_sunk_gc",
    "n_sunk_wanderer",
    "n_sunk",
    "m_gc_init_total_msun",
    "m_gc_final_total_msun",
    "M_IMBH_init_tot",
    "M_IMBH_final_tot",
    "M_NSC",
    "M_SMBH_init",
    "M_SMBH_final",
]
HALO_SUMMARY_BY_Z_COLUMNS = [
    "hid_z0",
    "z_out",
    "lookback_to_z0_gyr",
    "halo_mass_available",
    "logMh_z_msun",
    "M_NSC",
    "M_SMBH_init",
    "M_SMBH_final",
]

RUN_METADATA_NAME = "run_metadata.json"
HALO_TREE_LOOKUP_NAME = "halo_tree_lookup.csv"
SCRATCH_DIR_DEFAULT = Path("/lingshan/disk3/subonan/_scratch")
EX_SITU_GAO_ANALYTIC = 0
EX_SITU_BRANCH_NO_IMPORT = 1
EX_SITU_BRANCH_IMPORT = 2
EX_SITU_MODES = (EX_SITU_GAO_ANALYTIC, EX_SITU_BRANCH_NO_IMPORT, EX_SITU_BRANCH_IMPORT)
VALID_EVOLUTION_STATUS = {
    STAT_ALIVE,
    STAT_EXHAUSTED,
    STAT_TORN,
    STAT_SUNK,
    STAT_WANDERER,
    STAT_WANDERER_SUNK,
}
TIME_ROUNDOFF_TOL_GYR = 1.0e-5


def _ns_tag(ns: float) -> str:
    """Convert one Sersic index into the filename-safe `0p5` style tag."""

    return f"{check_finite_positive(ns, name='Sersic index N_s'):.1f}".replace(".", "p")


def _fmt_param_tag(value: float) -> str:
    """Compact float formatting for output filenames."""

    return f"{check_finite(value, name='filename parameter'):g}"


def _ns_output_dir(base_output_dir: Path, ns_value: float) -> Path:
    """Return the per-N_s output directory and create it if needed."""

    path = base_output_dir / f"ns{_ns_tag(ns_value)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ns_product_name(kind: str, ns_value: float) -> str:
    tag = _ns_tag(ns_value)
    templates = {
        "final_gcs": f"finalGCs_ns{tag}.dat",
        "depos": f"depos_ns{tag}.dat",
        "halo_summary_by_z": f"haloSummaryByZ_ns{tag}.csv",
    }
    try:
        return templates[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown per-N_s product kind: {kind}") from exc


def _tmp_product_path(work_dir: Path, kind: str, hz0: int, ns_tag: str, branch_id: int | None = None) -> Path:
    hid = int(hz0)
    if hid < 0:
        raise ValueError(f"halo_id_z0 must be non-negative; got {hid}")
    if branch_id is None:
        templates = {
            "final_gcs_halo": f"finalGCs_halo{hid}_ns{ns_tag}.tmp.dat",
            "depos_halo": f"depos_halo{hid}_ns{ns_tag}.tmp.dat",
        }
    else:
        branch = int(branch_id)
        if branch < 0:
            raise ValueError(f"branch_id must be non-negative; got {branch}")
        templates = {
            "final_gcs_branch": f"finalGCs_halo{hid}_branch{branch}_ns{ns_tag}.tmp.dat",
            "depos_branch": f"depos_halo{hid}_branch{branch}_ns{ns_tag}.tmp.dat",
            "tree_branch": f"tree_halo{hid}_branch{branch}_ns{ns_tag}.txt",
        }
    try:
        return Path(work_dir) / templates[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown temporary product kind: {kind}") from exc


def _parse_ns_values(text: str) -> List[float]:
    out: List[float] = []
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(check_finite_positive(float(tok), name="Sersic index N_s"))
    if not out:
        raise ValueError("No valid N_s values were provided.")
    return out


def _parse_out_z(text: str) -> List[float]:
    out: List[float] = []
    seen: set[float] = set()
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        value = check_finite_non_negative(float(tok), name="Output redshift z")
        if abs(value) < 1.0e-12:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    out.sort()
    return out

def _clear_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _prepare_scratch_root(path: Path, min_free_gb: float = 0.0) -> Path:
    """Create and validate the shared transient-work root."""

    scratch_root = Path(path).expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    probe_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".nsc_scratch_write_test_",
            dir=scratch_root,
            delete=False,
        ) as f:
            probe_path = Path(f.name)
            f.write("ok\n")
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink()
            except FileNotFoundError:
                pass
    usage = shutil.disk_usage(scratch_root)
    free_gb = usage.free / 1024.0**3
    print(f"SCRATCH_ROOT {scratch_root} free_gb={free_gb:.2f}")
    if float(min_free_gb) > 0.0 and free_gb < float(min_free_gb):
        raise OSError(
            f"Scratch root {scratch_root} has only {free_gb:.2f} GB free, "
            f"below requested --min-scratch-free-gb={float(min_free_gb):.2f}."
        )
    return scratch_root


def _make_run_scratch_dir(scratch_root: Path) -> Path:
    """Create one unique scratch directory for one top-level run.py invocation."""

    run_scratch_dir = Path(tempfile.mkdtemp(prefix="nsc_run_", dir=scratch_root))
    print(f"RUN_SCRATCH {run_scratch_dir}")
    return run_scratch_dir


def _remove_run_scratch_dir(path: Path) -> None:
    """Remove only the scratch directory owned by the current run."""

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _check_project_layout(
    *,
    plot_gao2023_requested: bool,
    plot_choksi2018_requested: bool,
    plot_neumayer2020_requested: bool,
    plot_kong2026_requested: bool,
    tree_dir: Path | None,
) -> tuple[Path, Path]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing data directory: {DATA_DIR}"
        )
    if not SNAPS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing snapshot-redshift table: {SNAPS_PATH}"
        )
    if not MASS_LOSS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing stellar-evolution mass-loss table: {MASS_LOSS_PATH}"
        )
    if plot_gao2023_requested and (not PLOT_GAO2023_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Gao+2024 plot script: {PLOT_GAO2023_PATH}"
        )
    if plot_choksi2018_requested and (not PLOT_CHOKSI2018_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Choksi+2018 plot script: {PLOT_CHOKSI2018_PATH}"
        )
    if plot_neumayer2020_requested and (not PLOT_NEUMAYER2020_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Neumayer+2020 plot script: {PLOT_NEUMAYER2020_PATH}"
        )
    if plot_kong2026_requested and (not PLOT_KONG2026_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Kong+2026 plot script: {PLOT_KONG2026_PATH}"
        )
    effective_tree_dir = tree_dir.resolve() if tree_dir is not None else DEFAULT_TREE_DIR
    if not effective_tree_dir.is_dir():
        raise FileNotFoundError(
            f"Expected fixed-tree input directory not found: {effective_tree_dir}. "
            f"The runner uses the bundled repository layout under {PROJECT_ROOT}."
        )
    _tree_file_map(str(effective_tree_dir.resolve()))
    return DATA_DIR, effective_tree_dir


def _confirm_clear_output(path: Path) -> None:
    """Confirm clearing only when the output directory already has contents."""

    path.mkdir(parents=True, exist_ok=True)
    try:
        next(path.iterdir())
    except StopIteration:
        return

    prompt = (
        f"--clear-output will remove all existing contents under:\n"
        f"{path}\n"
        "Continue? [y/N]: "
    )
    try:
        reply = input(prompt).strip().lower()
    except EOFError as exc:
        raise SystemExit("Aborted: no confirmation received for --clear-output.") from exc
    if reply not in {"y", "yes"}:
        raise SystemExit("Aborted: output directory was not cleared.")


def _warn_if_output_nonempty(path: Path) -> None:
    """Warn when keeping existing output files."""

    path.mkdir(parents=True, exist_ok=True)
    try:
        next(path.iterdir())
    except StopIteration:
        return
    print(f"Warning: output directory is non-empty and will be kept: {path}")


def _iter_numeric_text_lines(path: Path) -> Sequence[str]:
    """Yield non-comment, non-empty lines from a text table."""

    out: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            out.append(s)
    return out


def _coerce_tree_id(value: float | str) -> int:
    """Convert fixed-tree IDs that may arrive as float-like text."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    value_float = check_finite(float(value), name="tree identifier")
    out = int(round(value_float))
    if abs(value_float - float(out)) > 1.0e-6:
        raise ValueError(f"Tree identifier is not integer-like: {value}")
    return out


def _check_array(values: np.ndarray, name: str, *, positive: bool = False, non_negative: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if positive:
        bad = arr <= 0.0
        if np.any(bad):
            raise ValueError(f"{name} contains {int(np.sum(bad))} non-positive values.")
    if non_negative:
        bad = arr < 0.0
        if np.any(bad):
            raise ValueError(f"{name} contains {int(np.sum(bad))} negative values.")
    return arr


def _checked_non_negative_time(value: float, name: str) -> float:
    time_value = check_finite(value, name=name)
    if time_value < -TIME_ROUNDOFF_TOL_GYR:
        raise ValueError(f"{name} must be non-negative; got {time_value}")
    return 0.0 if time_value < 0.0 else float(time_value)


def _check_gcfin_array(gcfin_arr: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(gcfin_arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= 8:
        raise ValueError(f"{context} final-GC array must have at least 9 columns; got shape={arr.shape}")
    status = arr[:, 1].astype(int)
    if np.any(np.abs(arr[:, 1] - status.astype(float)) > 1.0e-8):
        raise ValueError(f"{context} final-GC array has non-integer status codes.")
    invalid_status = ~np.isin(status, np.asarray(sorted(VALID_EVOLUTION_STATUS), dtype=int))
    if np.any(invalid_status):
        raise ValueError(f"{context} final-GC array has invalid status codes: {sorted(set(status[invalid_status]))}")
    _check_array(arr[:, 2], f"{context} final GC stellar mass", non_negative=True)
    _check_array(arr[:, 3], f"{context} initial GC mass", positive=True)
    arr[:, 4] = np.asarray(
        [_checked_non_negative_time(value, f"{context} final lookback time") for value in arr[:, 4]],
        dtype=float,
    )
    arr[:, 5] = np.asarray(
        [_checked_non_negative_time(value, f"{context} initial lookback time") for value in arr[:, 5]],
        dtype=float,
    )
    _check_array(arr[:, 6], f"{context} final radius", non_negative=True)
    _check_array(arr[:, 7], f"{context} initial radius", positive=True)
    _check_array(arr[:, 8], f"{context} final IMBH mass", non_negative=True)
    return arr


def _format_combined_gcfin_row(hid: int, row: str, formation_row: np.ndarray | None = None, gc_index_halo_override: int | None = None) -> str:
    """Reformat one temporary per-halo GC row into the published finalGCs schema."""

    parts = row.split()
    if len(parts) <= 8:
        raise ValueError(f"Per-halo gcfin row is missing required M_IMBH_final column: {row}")

    gc_index_float = check_finite(float(parts[0]), name="GC index")
    gc_index_halo = int(round(gc_index_float))
    if abs(gc_index_float - float(gc_index_halo)) > 1.0e-8 or gc_index_halo < 1:
        raise ValueError(f"GC index must be a positive integer-like value in row: {row}")
    if gc_index_halo_override is not None:
        gc_index_halo = int(gc_index_halo_override)
    status_float = check_finite(float(parts[1]), name="GC evolution status")
    status = int(round(status_float))
    if abs(status_float - float(status)) > 1.0e-8:
        raise ValueError(f"GC evolution status is not integer-like in row: {row}")
    if status not in VALID_EVOLUTION_STATUS:
        raise ValueError(f"Invalid GC evolution status code {status} in row: {row}")
    m_gc_final = check_finite_non_negative(float(parts[2]), name="Final GC stellar mass M_GC_final")
    m_init_msun = check_finite_positive(float(parts[3]), name="Initial GC mass m_init_msun")
    lookback_time_final_gyr = _checked_non_negative_time(float(parts[4]), "Final lookback time")
    lookback_time_init_gyr = _checked_non_negative_time(float(parts[5]), "Initial lookback time")
    r_final_kpc = check_finite_non_negative(float(parts[6]), name="Final GC radius r_final_kpc")
    r_init_kpc = check_finite_positive(float(parts[7]), name="Initial GC radius r_init_kpc")
    feh = 0.0
    gc_radius_pc = 0.0
    sigma_h_msun_pc2 = 0.0
    M_IMBH_init = 0.0

    if formation_row is not None:
        # The evolution code only knows about the compact GCini columns. The
        # merged public table restores birth-time GC properties from allcat.
        feh = check_finite(float(formation_row[8]), name="GC metallicity [Fe/H]")
        if len(formation_row) > 10:
            gc_radius_pc = check_finite_positive(float(formation_row[10]), name="GC half-mass radius")
        if len(formation_row) > 11:
            sigma_h_msun_pc2 = check_finite_positive(float(formation_row[11]), name="GC half-mass surface density")
        if len(formation_row) > 12:
            M_IMBH_init = check_finite_non_negative(float(formation_row[12]), name="Initial IMBH mass")
    M_IMBH_final = check_finite_non_negative(float(parts[8]), name="Final IMBH mass")

    return (
        f"{hid:d} {gc_index_halo:d} {status:d} "
        f"{m_gc_final:.10e} {m_init_msun:.10e} "
        f"{lookback_time_final_gyr:.10e} {lookback_time_init_gyr:.10e} "
        f"{r_final_kpc:.10e} {r_init_kpc:.10e} "
        f"{gc_radius_pc:.10e} {sigma_h_msun_pc2:.10e} {feh:.10e} "
        f"{M_IMBH_init:.10e} {M_IMBH_final:.10e}"
    )


def _shift_gcfin_row_lookbacks(row: str, lookback_shift_gyr: float) -> str:
    parts = row.split()
    if len(parts) < 6:
        raise ValueError(f"Expected at least 6 final-GC columns, got {len(parts)} in row: {row}")
    shift = check_finite(lookback_shift_gyr, name="lookback shift")
    parts[4] = f"{_checked_non_negative_time(float(parts[4]) + shift, 'Shifted final lookback time'):.10e}"
    parts[5] = f"{_checked_non_negative_time(float(parts[5]) + shift, 'Shifted initial lookback time'):.10e}"
    return " ".join(parts)


def _shift_depos_row_lookback(row: str, lookback_shift_gyr: float) -> str:
    parts = row.split()
    if len(parts) < 7:
        raise ValueError(f"Expected at least 7 deposit columns, got {len(parts)} in row: {row}")
    shift = check_finite(lookback_shift_gyr, name="lookback shift")
    parts[0] = f"{_checked_non_negative_time(float(parts[0]) + shift, 'Shifted deposit lookback time'):.10e}"
    return " ".join(parts)


def _combine_per_halo_outputs(
    per_halo_dir: Path,
    ns_output_dir: Path,
    ns_value: float,
    halo_ids: Sequence[int],
    all_rows: np.ndarray,
) -> None:
    """Merge temporary per-halo outputs for one N_s into the published files."""

    ns_tag = _ns_tag(ns_value)
    all_rows_arr = np.asarray(all_rows, dtype=float)
    if all_rows_arr.ndim != 2 or all_rows_arr.shape[1] <= 12:
        raise ValueError(f"all_rows must have the 13-column formation schema; got shape={all_rows_arr.shape}")
    halo_ids_sorted = sorted({int(hid) for hid in halo_ids})
    hid_all = np.asarray(all_rows_arr[:, 0], dtype=int)
    formation_rows_by_halo = {
        int(hid): np.asarray(all_rows_arr[hid_all == int(hid)], dtype=float)
        for hid in halo_ids_sorted
    }

    gcfin_out = ns_output_dir / _ns_product_name("final_gcs", ns_value)
    depos_out = ns_output_dir / _ns_product_name("depos", ns_value)

    with gcfin_out.open("w", encoding="utf-8") as f_gcfin:
        f_gcfin.write("# " + COMBINED_FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for hid in halo_ids_sorted:
            src = _tmp_product_path(per_halo_dir, "final_gcs_halo", hid, ns_tag)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-halo GCfin file: {src}")
            halo_rows = formation_rows_by_halo.get(int(hid))
            if halo_rows is None:
                raise ValueError(f"Missing formation rows for halo {hid}.")
            for row in _iter_numeric_text_lines(src):
                parts = row.split()
                if len(parts) < 1:
                    raise ValueError(f"Malformed per-halo GCfin row in {src}: {row}")
                gc_index_halo = int(float(parts[0]))
                if gc_index_halo < 1 or gc_index_halo > len(halo_rows):
                    raise ValueError(
                        f"GC index {gc_index_halo} is out of bounds for halo {hid} "
                        f"with {len(halo_rows)} formation rows."
                    )
                # GC indices inside each temporary halo file are 1-based and
                # follow that halo's local allcat ordering.
                formation_row = halo_rows[gc_index_halo - 1]
                f_gcfin.write(_format_combined_gcfin_row(hid, row, formation_row=formation_row) + "\n")

    with depos_out.open("w", encoding="utf-8") as f_depos:
        f_depos.write("# " + COMBINED_DEPOS_HEADER.replace("\n", "\n# ") + "\n")
        for hid in halo_ids_sorted:
            src = _tmp_product_path(per_halo_dir, "depos_halo", hid, ns_tag)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-halo Depos file: {src}")
            for row in _iter_numeric_text_lines(src):
                f_depos.write(f"{hid:d} {row}\n")


def _combine_all_ns_outputs(output_dir: Path, ns_values: Sequence[float]) -> None:
    """Merge per-N_s combined GCfin/Depos files into one top-level file each."""

    gcfin_out = output_dir / "finalGCs_all.dat"
    depos_out = output_dir / "depos_all.dat"

    with gcfin_out.open("w", encoding="utf-8") as f_gcfin:
        f_gcfin.write("# " + GLOBAL_FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for ns in ns_values:
            ns_tag = _ns_tag(ns)
            src = output_dir / f"ns{ns_tag}" / _ns_product_name("final_gcs", ns)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-N_s combined GCfin file: {src}")
            for row in _iter_numeric_text_lines(src):
                f_gcfin.write(f"{float(ns):.1f} {row}\n")

    with depos_out.open("w", encoding="utf-8") as f_depos:
        f_depos.write("# " + GLOBAL_DEPOS_HEADER.replace("\n", "\n# ") + "\n")
        for ns in ns_values:
            ns_tag = _ns_tag(ns)
            src = output_dir / f"ns{ns_tag}" / _ns_product_name("depos", ns)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-N_s combined Depos file: {src}")
            for row in _iter_numeric_text_lines(src):
                f_depos.write(f"{float(ns):.1f} {row}\n")


def _build_halo_summary_table(
    all_rows: np.ndarray,
    status: np.ndarray,
    m_final: np.ndarray,
    M_IMBH_final: np.ndarray,
    central_history_by_halo: Dict[int, Sequence[dict]] | None = None,
    eddington_ratio: float = 0.0,
) -> pd.DataFrame:
    """Build one halo-level summary table, including the SMBH estimate."""

    all_rows_arr = np.asarray(all_rows, dtype=float)
    if all_rows_arr.ndim != 2 or all_rows_arr.shape[1] <= 12:
        raise ValueError(f"all_rows must have the 13-column formation schema; got shape={all_rows_arr.shape}")
    n_rows = len(all_rows_arr)
    if len(status) != n_rows or len(m_final) != n_rows or len(M_IMBH_final) != n_rows:
        raise ValueError(
            "Halo summary inputs must have compatible lengths: "
            f"all_rows={n_rows}, status={len(status)}, m_final={len(m_final)}, M_IMBH_final={len(M_IMBH_final)}"
        )

    hid = np.asarray(all_rows_arr[:, 0], dtype=int)
    logmh_z0 = _check_array(all_rows_arr[:, 1], "z=0 halo log mass")
    m_init = np.power(10.0, _check_array(all_rows_arr[:, 6], "initial GC log mass"))
    _check_array(m_init, "initial GC mass", positive=True)
    M_IMBH_init = _check_array(all_rows_arr[:, 12], "initial IMBH mass", non_negative=True)
    M_IMBH_final = np.asarray(M_IMBH_final, dtype=float)
    status = np.asarray(status, dtype=int)
    m_final = np.asarray(m_final, dtype=float)
    if np.any(~np.isin(status, np.asarray(sorted(VALID_EVOLUTION_STATUS), dtype=int))):
        raise ValueError(f"Halo summary received invalid GC status code(s): {sorted(set(status))}")
    _check_array(m_final, "final GC stellar mass", non_negative=True)
    _check_array(M_IMBH_final, "final IMBH mass", non_negative=True)

    rows: List[Dict[str, float | int]] = []
    for hid0 in np.unique(hid):
        idx = hid == int(hid0)
        s = status[idx]
        m_final_halo = m_final[idx]
        survivor_mask = s == STAT_ALIVE
        imbh_init = M_IMBH_init[idx]
        imbh_final = M_IMBH_final[idx]
        n_sunk_gc = int(np.sum(s == STAT_SUNK))
        n_sunk_wanderer = int(np.sum(s == STAT_WANDERER_SUNK))
        central_events = list((central_history_by_halo or {}).get(int(hid0), []))
        m_nsc, m_smbh_init, m_smbh_final = _central_masses_at_redshift(
            central_events,
            0.0,
            eddington_ratio,
        )
        _warn_if_central_bh_high(m_smbh_final, context=f"halo {int(hid0)} at z=0")
        sunk_mask = np.isin(s, np.asarray([STAT_SUNK, STAT_WANDERER_SUNK], dtype=int))
        m_imbh_final_tot = float(m_smbh_final + np.sum(imbh_final[(imbh_init > 0.0) & (~sunk_mask)]))
        rows.append(
            {
                "hid_z0": int(hid0),
                "logMh_z0": float(logmh_z0[idx][0]),
                "n_gc_total": int(np.sum(idx)),
                "n_alive": int(np.sum(s == STAT_ALIVE)),
                "n_wanderer": int(np.sum(s == STAT_WANDERER)),
                "n_exhausted": int(np.sum(s == STAT_EXHAUSTED)),
                "n_torn": int(np.sum(s == STAT_TORN)),
                "n_sunk_gc": n_sunk_gc,
                "n_sunk_wanderer": n_sunk_wanderer,
                "n_sunk": n_sunk_gc + n_sunk_wanderer,
                "m_gc_init_total_msun": float(np.sum(m_init[idx])),
                "m_gc_final_total_msun": float(np.sum(m_final_halo[survivor_mask])),
                "M_IMBH_init_tot": float(np.sum(imbh_init)),
                "M_IMBH_final_tot": m_imbh_final_tot,
                "M_NSC": m_nsc,
                "M_SMBH_init": float(m_smbh_init),
                "M_SMBH_final": float(m_smbh_final),
            }
        )

    out = pd.DataFrame(rows, columns=HALO_SUMMARY_COLUMNS)
    if len(out) == 0:
        return out
    out.sort_values("hid_z0", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _interpolate_mpb_logmh_at_redshift(mpb_rows: np.ndarray, z_out: float) -> tuple[float, int]:
    """Interpolate MPB halo mass at the requested redshift in linear mass."""

    z_value = check_finite_non_negative(z_out, name="Output redshift z_out")
    rows = np.asarray(mpb_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 6:
        return np.nan, 0

    redshift = rows[:, 5]
    logmh = rows[:, 0]
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("MPB rows contain non-finite or negative redshifts.")
    if np.any(~np.isfinite(logmh)):
        raise ValueError("MPB rows contain non-finite halo log masses.")
    z_min = float(np.min(redshift))
    z_max = float(np.max(redshift))
    tol = 1.0e-10
    if z_value < z_min - tol or z_value > z_max + tol:
        return np.nan, 0

    cosmic_time = np.array([Redshift2CosmicAge(float(z), time_unit="Gyr") for z in redshift], dtype=float)
    mass = np.power(10.0, logmh)
    _check_array(cosmic_time, "MPB cosmic time", non_negative=True)
    _check_array(mass, "MPB halo mass", positive=True)
    order = np.argsort(cosmic_time, kind="mergesort")
    cosmic_time = cosmic_time[order]
    mass = mass[order]
    keep_last_duplicate = np.r_[cosmic_time[1:] != cosmic_time[:-1], True]
    unique_time = cosmic_time[keep_last_duplicate]
    mass = mass[keep_last_duplicate]
    if len(unique_time) == 0:
        return np.nan, 0

    target_time = float(Redshift2CosmicAge(z_value, time_unit="Gyr"))
    if target_time < float(unique_time[0]) - tol or target_time > float(unique_time[-1]) + tol:
        return np.nan, 0

    interp_mass = float(np.interp(target_time, unique_time, mass))
    if not np.isfinite(interp_mass) or interp_mass <= 0.0:
        raise ValueError(f"Interpolated MPB halo mass is non-positive at z={z_value}: {interp_mass}")
    return float(np.log10(interp_mass)), 1

def _warn_if_central_bh_high(m_bh_msun: float, *, context: str) -> None:
    if central_bh_mass_warning_needed(m_bh_msun):
        warnings.warn(
            f"Stored central BH mass exceeds {CENTRAL_BH_WARNING_MASS_MSUN:.3e} Msun "
            f"for {context}: M_SMBH_final={float(m_bh_msun):.6e} Msun",
            RuntimeWarning,
        )


def _central_masses_at_redshift(events: Sequence[dict], z_out: float, eddington_ratio: float) -> tuple[float, float, float]:
    """Sample stored central stellar and BH masses at one output redshift."""

    z_value = check_finite_non_negative(z_out, name="Output redshift z_out")
    eddington_ratio = check_eddington_ratio(eddington_ratio)
    if not events:
        return 0.0, 0.0, 0.0
    target_time = float(Redshift2CosmicAge(z_value, time_unit="Gyr"))
    eligible = [
        event for event in events
        if float(event.get("t_cosmic_gyr", -np.inf)) <= target_time + 1.0e-10
    ]
    if not eligible:
        return 0.0, 0.0, 0.0
    latest = max(eligible, key=lambda item: float(item.get("t_cosmic_gyr", 0.0)))
    latest_time = _checked_non_negative_time(float(latest.get("t_cosmic_gyr", target_time)), "Central-event cosmic time")
    dt_gyr = _checked_non_negative_time(target_time - latest_time, "Central BH growth timestep")
    m_nsc = check_finite_non_negative(float(latest.get("M_NSC", 0.0)), name="Central NSC mass")
    m_smbh_init = check_finite_non_negative(float(latest.get("M_SMBH_init", 0.0)), name="Initial central BH mass")
    m_smbh_final = float(grow_eddington_mass_msun(
        check_finite_non_negative(float(latest.get("M_SMBH_current", 0.0)), name="Current central BH mass"),
        dt_gyr=dt_gyr,
        f_edd=eddington_ratio,
        overflow_policy="warn_inf",
    ))
    return (
        m_nsc,
        m_smbh_init,
        m_smbh_final,
    )

def _build_halo_summary_by_z_table(
       all_rows: np.ndarray,
       status: np.ndarray,
       lookback_time_final_gyr: np.ndarray,
       out_redshifts: Sequence[float],
       tree_dir: Path,
       per_halo_dir: Path,
       ns_tag: str,
       central_history_by_halo: Dict[int, Sequence[dict]] | None = None,
       eddington_ratio: float = 0.0) -> pd.DataFrame:
    """Build one long-format halo summary table across requested output redshifts."""

    all_rows_arr = np.asarray(all_rows, dtype=float)
    if all_rows_arr.ndim != 2 or all_rows_arr.shape[1] <= 12:
        raise ValueError(f"all_rows must have the 13-column formation schema; got shape={all_rows_arr.shape}")
    if len(status) != len(all_rows_arr) or len(lookback_time_final_gyr) != len(all_rows_arr):
        raise ValueError(
            "Halo-by-redshift summary inputs must have compatible lengths: "
            f"all_rows={len(all_rows_arr)}, status={len(status)}, lookback={len(lookback_time_final_gyr)}"
        )
    _check_array(np.asarray(lookback_time_final_gyr, dtype=float), "final lookback time", non_negative=True)

    hid = np.asarray(all_rows_arr[:, 0], dtype=int)
    t_z0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    output_redshifts = [0.0] + [check_finite_non_negative(float(z), name="Output redshift z_out") for z in out_redshifts]
    unique_hids = np.unique(hid)
    mpb_by_halo = {
        int(hid0): read_haloevo_mpb(_tree_file_for_halo(tree_dir, int(hid0)))
        for hid0 in unique_hids
    }

    rows: List[Dict[str, float | int]] = []
    for z_out in output_redshifts:
        lookback_to_z0_gyr = _checked_non_negative_time(
            t_z0 - float(Redshift2CosmicAge(float(z_out), time_unit="Gyr")),
            "Lookback time from z=0 to output redshift",
        )
        for hid0 in unique_hids:
            logmh_z, halo_mass_available = _interpolate_mpb_logmh_at_redshift(
                mpb_by_halo[int(hid0)],
                float(z_out),
            )
            m_nsc_z, m_smbh_init_z, m_smbh_final_z = _central_masses_at_redshift(
                list((central_history_by_halo or {}).get(int(hid0), [])),
                float(z_out),
                eddington_ratio,
            )
            _warn_if_central_bh_high(m_smbh_final_z, context=f"halo {int(hid0)} at z={float(z_out):g}")
            m_nsc_z = check_finite_non_negative(m_nsc_z, name="NSC mass")
            m_smbh_init_z = check_finite_non_negative(m_smbh_init_z, name="Initial central BH mass")
            m_smbh_final_z = check_finite_non_negative(m_smbh_final_z, name="Final central BH mass")
            rows.append(
                {
                    "hid_z0": int(hid0),
                    "z_out": float(z_out),
                    "lookback_to_z0_gyr": float(lookback_to_z0_gyr),
                    "halo_mass_available": int(halo_mass_available),
                    "logMh_z_msun": float(logmh_z),
                    "M_NSC": float(m_nsc_z),
                    "M_SMBH_init": float(m_smbh_init_z),
                    "M_SMBH_final": float(m_smbh_final_z),
                }
            )

    out = pd.DataFrame(rows, columns=HALO_SUMMARY_BY_Z_COLUMNS)
    if len(out) == 0:
        return out
    out.sort_values(["hid_z0", "z_out"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _build_snap_map(snaps2redshifts_path: Path) -> np.ndarray:
    """Load the snapshot->redshift lookup used across the workflow."""

    z = np.loadtxt(snaps2redshifts_path, comments="#", ndmin=1)
    return np.asarray(z, dtype=float).reshape(-1)


def _nearest_snap(z_form: np.ndarray, z_snap: np.ndarray) -> np.ndarray:
    """Map formation redshifts onto the nearest discrete simulation snapshot."""

    out = np.empty(len(z_form), dtype=int)
    for i, z in enumerate(z_form):
        out[i] = int(np.argmin(np.abs(z_snap - z)))
    return out


def _legacy_tree_file_map(tree_dir: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for path in sorted(tree_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".txt", ".dat"):
            continue
        try:
            hid = int(path.stem)
        except ValueError:
            continue
        mapping[int(hid)] = str(path)
    return mapping


@lru_cache(maxsize=None)
def _tree_file_map(tree_dir_str: str) -> Dict[int, str]:
    tree_dir = Path(tree_dir_str)
    lookup_path = tree_dir / TREE_LOOKUP_BASENAME
    if lookup_path.is_file():
        mapping: Dict[int, str] = {}
        with lookup_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    hid = int(row["halo_id_z0"])
                    basename = row["fixed_tree_basename"].strip()
                except (KeyError, ValueError) as exc:
                    raise RuntimeError(f"Malformed tree lookup row in {lookup_path}: {row}") from exc
                path = tree_dir / basename
                if not path.is_file():
                    raise FileNotFoundError(f"Tree lookup references missing fixed tree: {path}")
                if hid in mapping:
                    raise RuntimeError(f"Duplicate halo_id_z0 {hid} in tree lookup {lookup_path}")
                mapping[hid] = str(path)
        if not mapping:
            raise RuntimeError(f"Tree lookup exists but contains no usable rows: {lookup_path}")
        return mapping

    mapping = _legacy_tree_file_map(tree_dir)
    if not mapping:
        raise RuntimeError(f"No usable fixed-tree files were found under {tree_dir}")
    return mapping


def _tree_file_for_halo(tree_dir: Path, halo_id: int) -> Path:
    hid = int(halo_id)
    mapping = _tree_file_map(str(tree_dir.resolve()))
    try:
        return Path(mapping[hid])
    except KeyError as exc:
        raise FileNotFoundError(f"Missing tree file for halo {hid} under {tree_dir}") from exc


def _read_full_tree_numeric(tree_path: Path) -> np.ndarray:
    """Read the fixed-tree columns needed by formation/evolution mapping."""

    rows: List[List[object]] = []
    skipped = 0
    with Path(tree_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if (not stripped) or stripped.startswith("#") or stripped.lower().startswith("logmh"):
                continue
            parts = stripped.split()
            if len(parts) < 9:
                skipped += 1
                continue
            try:
                rows.append(
                    [
                        float(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                        float(parts[8]),
                    ]
                )
            except ValueError:
                skipped += 1
                continue
    if skipped:
        warnings.warn(
            f"Skipped {skipped} malformed fixed-tree row(s) while reading {Path(tree_path)}.",
            RuntimeWarning,
        )
    if not rows:
        return np.zeros((0, 9), dtype=object)
    arr = np.asarray(rows, dtype=object)
    _check_array(np.asarray(arr[:, 0], dtype=float), f"{tree_path} halo log mass")
    _check_array(np.asarray(arr[:, 5], dtype=float), f"{tree_path} redshift", non_negative=True)
    _check_array(np.asarray(arr[:, 6:9], dtype=float), f"{tree_path} spin components")
    return arr


def _mpb_branch_id(tree_rows: np.ndarray) -> int:
    """Return the shared project MPB branch ID for fixed-tree rows."""

    rows = np.asarray(tree_rows, dtype=object)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 4:
        raise ValueError("Cannot identify the MPB branch from an empty or malformed fixed tree.")
    return fixed_tree_mpb_branch_id(rows[:, 0], rows[:, 3])


@dataclass(frozen=True)
class BranchMergerEvent:
    source_branch_id: int
    recipient_branch_id: int
    z_merge: float
    t_merge_gyr: float
    source_logmh: float
    recipient_logmh: float
    recipient_mhalo_msun: float
    recipient_rvir_kpc: float
    r_accretion_kpc: float


def _branch_merger_events_by_source(tree_rows: np.ndarray, required_branches: set[int] | None = None) -> Dict[int, BranchMergerEvent]:
    """Build branch-merger events from fixed-tree descendant IDs."""

    rows = np.asarray(tree_rows, dtype=object)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 9:
        raise ValueError("Cannot build branch-merger events from an empty or malformed fixed tree.")
    _check_array(np.asarray(rows[:, 0], dtype=float), "fixed-tree halo log mass")
    _check_array(np.asarray(rows[:, 5], dtype=float), "fixed-tree redshift", non_negative=True)
    _check_array(np.asarray(rows[:, 6:9], dtype=float), "fixed-tree spin components")
    mpb_branch = _mpb_branch_id(rows)
    required = {int(v) for v in required_branches} if required_branches is not None else None
    by_subhalo: Dict[int, List[np.ndarray]] = {}
    for row in rows:
        by_subhalo.setdefault(_coerce_tree_id(row[2]), []).append(row)

    events: Dict[int, BranchMergerEvent] = {}
    for branch_id in sorted({_coerce_tree_id(value) for value in rows[:, 3]}):
        if branch_id == int(mpb_branch):
            continue
        if required is not None and branch_id not in required:
            continue
        if branch_id < 0:
            raise ValueError(f"Branch ID must be non-negative; got {branch_id}")
        branch_rows = rows[np.array([_coerce_tree_id(value) == branch_id for value in rows[:, 3]], dtype=bool)]
        if len(branch_rows) == 0:
            continue
        terminal = branch_rows[int(np.argmin(np.asarray(branch_rows[:, 5], dtype=float)))]
        desc_id = _coerce_tree_id(terminal[4])
        candidates = [np.asarray(candidate, dtype=object) for candidate in by_subhalo.get(desc_id, [])]
        candidates = [candidate for candidate in candidates if _coerce_tree_id(candidate[3]) != branch_id]
        if len(candidates) > 1:
            terminal_z = float(terminal[5])
            before_or_at = [
                candidate
                for candidate in candidates
                if float(candidate[5]) <= terminal_z + 1.0e-10
            ]
            if before_or_at:
                candidates = sorted(before_or_at, key=lambda candidate: float(candidate[5]), reverse=True)[:1]
            else:
                candidates = sorted(candidates, key=lambda candidate: abs(float(candidate[5]) - terminal_z))[:1]
        if len(candidates) == 1:
            recipient = np.asarray(candidates[0], dtype=object)
            recipient_branch = _coerce_tree_id(recipient[3])
            z_merge = float(recipient[5])
            recipient_logmh = float(recipient[0])
        else:
            # Some fixed trees omit the direct descendant row when a satellite
            # falls below the retained-tree mass threshold.  The only retained
            # descendant frame available for continuing its survivors is then
            # the MPB descendant at, or nearest to, the terminal branch redshift.
            recipient_branch = int(mpb_branch)
            z_merge = float(terminal[5])
            mpb_rows = rows[np.array([_coerce_tree_id(value) == int(mpb_branch) for value in rows[:, 3]], dtype=bool)]
            recipient_logmh, available = _interpolate_mpb_logmh_at_redshift(mpb_rows, z_merge)
            if not available:
                mpb_redshift = np.asarray(mpb_rows[:, 5], dtype=float)
                if len(mpb_redshift) == 0:
                    raise ValueError(
                        f"Branch {branch_id} has no retained descendant row for subhalo {desc_id}, "
                        "and the MPB branch has no usable rows."
                    )
                z_clamped = float(np.clip(z_merge, float(np.min(mpb_redshift)), float(np.max(mpb_redshift))))
                recipient_logmh, available = _interpolate_mpb_logmh_at_redshift(mpb_rows, z_clamped)
                if not available:
                    nearest = int(np.argmin(np.abs(mpb_redshift - z_clamped)))
                    recipient_logmh = float(mpb_rows[nearest, 0])
                    z_clamped = float(mpb_rows[nearest, 5])
                z_merge = z_clamped
        z_merge = check_finite_non_negative(z_merge, name="Branch merger redshift")
        recipient_logmh = check_finite(recipient_logmh, name="Recipient halo log mass")
        recipient_mhalo_msun = check_finite_positive(10.0 ** recipient_logmh, name="Recipient halo mass")
        recipient_rvir_kpc = check_finite_positive(Rv(Mh=recipient_mhalo_msun, z=z_merge), name="Recipient virial radius")
        events[branch_id] = BranchMergerEvent(
            source_branch_id=int(branch_id),
            recipient_branch_id=int(recipient_branch),
            z_merge=z_merge,
            t_merge_gyr=float(Redshift2CosmicAge(z_merge, time_unit="Gyr")),
            source_logmh=check_finite(float(terminal[0]), name="Source halo log mass"),
            recipient_logmh=recipient_logmh,
            recipient_mhalo_msun=recipient_mhalo_msun,
            recipient_rvir_kpc=recipient_rvir_kpc,
            r_accretion_kpc=0.5 * recipient_rvir_kpc,
        )
    return events


def _write_branch_tree(path: Path, tree_rows: np.ndarray, branch_id: int) -> None:
    """Write a branch-only fixed-tree table in the nine-column tree format."""

    branch = int(branch_id)
    if branch < 0:
        raise ValueError(f"Branch ID must be non-negative; got {branch}")
    rows = np.asarray(tree_rows, dtype=object)
    if rows.ndim != 2 or rows.shape[1] < 9:
        raise ValueError(f"Cannot write branch tree for malformed tree rows with shape={rows.shape}.")
    branch_rows = rows[np.array([_coerce_tree_id(value) == branch for value in rows[:, 3]], dtype=bool)]
    if branch_rows.size == 0:
        raise ValueError(f"Cannot write branch tree; branch {branch} has no rows.")
    _check_array(np.asarray(branch_rows[:, 0], dtype=float), f"branch {branch} halo log mass")
    _check_array(np.asarray(branch_rows[:, 5], dtype=float), f"branch {branch} redshift", non_negative=True)
    _check_array(np.asarray(branch_rows[:, 6:9], dtype=float), f"branch {branch} spin components")
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("logMh | fpID | subhaloID | main leaf ID | descID |  z\n")
        for row in branch_rows:
            handle.write(
                f"{float(row[0]):.12g} "
                f"{_coerce_tree_id(row[1])} {_coerce_tree_id(row[2])} "
                f"{_coerce_tree_id(row[3])} {_coerce_tree_id(row[4])} "
                f"{float(row[5]):.12g} {float(row[6]):.12g} {float(row[7]):.12g} {float(row[8]):.12g}\n"
            )


def _branch_ids_for_rows(all_rows: np.ndarray, tree_dir: Path) -> np.ndarray:
    """Map every formation row to the fixed-tree branch where it formed."""

    all_rows_arr = np.asarray(all_rows, dtype=float)
    if all_rows_arr.ndim != 2 or all_rows_arr.shape[1] <= 7:
        raise ValueError(f"Formation rows are malformed; got shape={all_rows_arr.shape}")
    hid = np.asarray(all_rows_arr[:, 0], dtype=int)
    _check_array(all_rows_arr[:, 3], "formation halo log mass")
    _check_array(all_rows_arr[:, 7], "formation redshift", non_negative=True)
    branch_ids = np.empty(len(all_rows), dtype=np.int64)
    for hz0 in np.unique(hid):
        tree_path = _tree_file_for_halo(tree_dir, int(hz0))
        tree_rows = _read_full_tree_numeric(tree_path)
        if tree_rows.shape[0] == 0:
            raise ValueError(f"No usable fixed-tree rows found for halo {int(hz0)}: {tree_path}")

        candidates_by_subfind: Dict[int, List[tuple[int, float, float]]] = {}
        for row in tree_rows:
            candidate = (_coerce_tree_id(row[3]), float(row[5]), float(row[0]))
            for subfind in {_coerce_tree_id(row[2]), _coerce_tree_id(float(row[2]))}:
                candidates_by_subfind.setdefault(subfind, []).append(candidate)

        for row_index in np.where(hid == int(hz0))[0]:
            subfind = _coerce_tree_id(all_rows_arr[row_index, 2])
            candidates = candidates_by_subfind.get(subfind)
            if not candidates:
                raise ValueError(
                    f"Cannot map formation row {row_index} in halo {int(hz0)} to a tree branch; "
                    f"subfind_form={all_rows_arr[row_index, 2]:.10e} is absent from {tree_path}."
                )
            zform = check_finite_non_negative(all_rows_arr[row_index, 7], name="formation redshift")
            logmh_form = check_finite(all_rows_arr[row_index, 3], name="formation halo log mass")
            scored = [
                (abs(z_tree - zform) + abs(logmh_tree - logmh_form), branch, z_tree, logmh_tree)
                for branch, z_tree, logmh_tree in candidates
            ]
            scored.sort(key=lambda item: item[0])
            best_score, best_branch, best_z, best_logmh = scored[0]
            if best_score > 1.0e-3:
                raise ValueError(
                    f"Cannot robustly map formation row {row_index} in halo {int(hz0)} to a tree branch; "
                    f"nearest candidate branch={best_branch}, z={best_z}, logMh={best_logmh}, "
                    f"row z={zform}, row logMh={logmh_form}."
                )
            branch_ids[row_index] = int(best_branch)

    return branch_ids


def _build_ismpb_flags(all_rows: np.ndarray, tree_dir: Path) -> np.ndarray:
    """Map each formed GC to MPB/non-MPB using its formation subhalo ID."""

    all_rows_arr = np.asarray(all_rows, dtype=float)
    hid = np.asarray(all_rows_arr[:, 0], dtype=int)
    branch_ids = _branch_ids_for_rows(all_rows_arr, tree_dir)
    flags = np.zeros(len(all_rows), dtype=int)

    for hz0 in np.unique(hid):
        tree_rows = _read_full_tree_numeric(_tree_file_for_halo(tree_dir, int(hz0)))
        mpb_branch = _mpb_branch_id(tree_rows)
        idx = np.where(hid == hz0)[0]
        flags[idx] = np.asarray(branch_ids[idx] == int(mpb_branch), dtype=int)

    return flags


def _stable_unique_halo_ids(halo_ids: np.ndarray) -> List[int]:
    ordered: List[int] = []
    seen: set[int] = set()
    for hid in np.asarray(halo_ids, dtype=int):
        hid_int = int(hid)
        if hid_int in seen:
            continue
        seen.add(hid_int)
        ordered.append(hid_int)
    return ordered


def _write_halo_tree_lookup(output_dir: Path, tree_dir: Path, halo_ids: np.ndarray) -> None:
    lookup_path = tree_dir / TREE_LOOKUP_BASENAME
    if not lookup_path.is_file():
        return

    rows_by_halo: Dict[int, Dict[str, str]] = {}
    with lookup_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                hid = int(row["halo_id_z0"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Malformed tree lookup row in {lookup_path}: {row}") from exc
            if hid in rows_by_halo:
                raise RuntimeError(f"Duplicate halo_id_z0 {hid} in tree lookup {lookup_path}")
            rows_by_halo[hid] = dict(row)

    ordered_halo_ids = _stable_unique_halo_ids(halo_ids)
    out_rows = []
    for hid in ordered_halo_ids:
        try:
            row = rows_by_halo[int(hid)]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Tree lookup {lookup_path} is missing halo_id_z0={hid}, which entered the simulation output."
            ) from exc
        out_rows.append(
            {
                "hid_z0": int(hid),
                "simulation_key": row["simulation_key"].strip(),
                "simulation": row["simulation"].strip(),
                "subhalo_id_z0": int(row["subhalo_id_z0"]),
                "fixed_tree_basename": row["fixed_tree_basename"].strip(),
                "file_index": int(row["file_index"]),
            }
        )

    pd.DataFrame(out_rows, columns=[
        "hid_z0",
        "simulation_key",
        "simulation",
        "subhalo_id_z0",
        "fixed_tree_basename",
        "file_index",
    ]).to_csv(output_dir / HALO_TREE_LOOKUP_NAME, index=False)


def _build_mpb_csv_from_trees(tree_dir: Path, halo_ids: np.ndarray, z_snap: np.ndarray, out_csv: Path) -> None:
    """Flatten the fixed trees into the compact MPB table used by plotting.

    The plotting script only needs the host id, snapshot number, halo mass, and
    spin vector, so this CSV is much smaller than re-reading the full trees
    every time figures are generated.
    """

    rows: List[Dict[str, float]] = []
    for hid in np.unique(halo_ids.astype(int)):
        try:
            tfile = _tree_file_for_halo(tree_dir, int(hid))
        except FileNotFoundError:
            continue
        for vals in _read_full_tree_numeric(tfile):
            z = check_finite_non_negative(float(vals[5]), name="fixed-tree redshift")
            snap = int(np.argmin(np.abs(z_snap - z)))
            rows.append(
                {
                    "subhalo_id_z0": int(hid),
                    "SnapNum": int(snap),
                    "Redshift": float(z),
                    "logMh_msun_h": check_finite(float(vals[0]), name="fixed-tree halo log mass"),
                    "SubhaloSpin_x": check_finite(float(vals[6]), name="fixed-tree spin x"),
                    "SubhaloSpin_y": check_finite(float(vals[7]), name="fixed-tree spin y"),
                    "SubhaloSpin_z": check_finite(float(vals[8]), name="fixed-tree spin z"),
                }
            )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(f"No MPB rows were built from tree directory: {tree_dir}")
    df.sort_values(["subhalo_id_z0", "SnapNum"], ascending=[True, False], inplace=True)
    df.to_csv(out_csv, index=False)


def _read_main_spatial_all(path: Path) -> np.ndarray:
    """Read ``all_<Ns>.txt`` generated by ``main_spatial.py``.

    The maintained modern schema is exactly 13 columns:
    the legacy 10-column formation catalog plus fixed formation-time GC radius,
    surface density, and IMBH mass.
    """

    if len(_iter_numeric_text_lines(path)) == 0:
        raise ValueError(
            f"{path} contains no GC rows. main_spatial.py likely selected no halos in the requested "
            "descendant z=0 mass window, found no usable tree entries in the configured tree directory, "
            "or formed no GCs in the selected run."
        )
    arr = np.loadtxt(path, comments="#", ndmin=2)
    n_expected = 13
    if arr.ndim != 2 or arr.shape[1] != n_expected:
        raise ValueError(f"{path} must have exactly {n_expected} columns; got shape={arr.shape}")
    arr = arr.astype(float, copy=False)
    _check_array(arr[:, 1], f"{path} z=0 halo log mass")
    _check_array(arr[:, 3], f"{path} formation halo log mass")
    _check_array(arr[:, 4], f"{path} formation stellar log mass")
    _check_array(arr[:, 5], f"{path} formation gas log mass")
    _check_array(arr[:, 6], f"{path} formation GC log mass")
    _check_array(arr[:, 7], f"{path} formation redshift", non_negative=True)
    _check_array(arr[:, 9], f"{path} initial GC radius", positive=True)
    _check_array(arr[:, 10], f"{path} GC half-mass radius", positive=True)
    _check_array(arr[:, 11], f"{path} GC half-mass surface density", positive=True)
    _check_array(arr[:, 12], f"{path} initial IMBH mass", non_negative=True)
    return arr


def _read_analytic_survival(path: Path) -> np.ndarray:
    """Read row-aligned analytic survival output from ``src/main.py``."""

    if len(_iter_numeric_text_lines(path)) == 0:
        raise ValueError(f"{path} contains no analytic survival rows.")
    arr = np.loadtxt(path, comments="#", ndmin=2)
    n_expected = 8
    if arr.ndim != 2 or arr.shape[1] != n_expected:
        raise ValueError(f"{path} must have exactly {n_expected} columns; got shape={arr.shape}")
    arr = arr.astype(float, copy=False)
    hid = arr[:, 0].astype(int)
    gc_uid = arr[:, 1].astype(int)
    if np.any(np.abs(arr[:, 1] - gc_uid.astype(float)) > 1.0e-8) or np.any(gc_uid < 1):
        raise ValueError(f"{path} has non-positive or non-integer gc_uid values.")
    for hid0 in np.unique(hid):
        uid_halo = gc_uid[hid == int(hid0)]
        if len(np.unique(uid_halo)) != len(uid_halo):
            raise ValueError(f"{path} has duplicate gc_uid values for halo {int(hid0)}.")
    _check_array(arr[:, 4], f"{path} analytic final GC mass", non_negative=True)
    survives = arr[:, 5]
    survives_i = survives.astype(int)
    if np.any(np.abs(survives - survives_i.astype(float)) > 1.0e-8) or np.any(~np.isin(survives_i, [0, 1])):
        raise ValueError(f"{path} has non-binary survives_analytic values.")
    if np.any(survives_i != (arr[:, 4] > 0.0).astype(int)):
        raise ValueError(f"{path} has survives_analytic values inconsistent with M_GC_analytic_final.")
    _check_array(arr[:, 6], f"{path} initial IMBH mass", non_negative=True)
    _check_array(arr[:, 7], f"{path} analytic initial radius", positive=True)
    return arr


def _stable_row_order(all_rows: np.ndarray) -> np.ndarray:
    """Build a deterministic sort order that is independent of filesystem order."""

    n = len(all_rows)
    df = pd.DataFrame(
        {
            "row": np.arange(n, dtype=int),
            "hid_z0": all_rows[:, 0].astype(int),
            "subfind_form": all_rows[:, 2].astype(np.int64),
            "logMh_form": np.round(all_rows[:, 3], 8),
            "logMstar_form": np.round(all_rows[:, 4], 8),
            "logMgas_form": np.round(all_rows[:, 5], 8),
            "logM_form": np.round(all_rows[:, 6], 8),
            "zform": np.round(all_rows[:, 7], 8),
            "feh": np.round(all_rows[:, 8], 8),
        }
    )
    sort_cols = [
        "hid_z0",
        "subfind_form",
        "zform",
        "logMh_form",
        "logMstar_form",
        "logMgas_form",
        "logM_form",
        "feh",
        "row",
    ]
    # mergesort keeps ordering stable for any exact key ties.
    return df.sort_values(sort_cols, kind="mergesort")["row"].to_numpy(dtype=int)


def _run_main_spatial_for_ns(
    stage_dir: Path,
    data_dir: Path,
    tree_dir: Path,
    ns_value: float,
    *,
    p2: float,
    p3: float,
    lg_cut_off_mass: float,
    ex_situ_mode: int,
    run_all: int,
    log_mh_min: float,
    log_mh_max: float,
    n_halos: int) -> Path:
    ns_str = f"{float(ns_value):.1f}"
    log_path = stage_dir / f"main_spatial_ns{_ns_tag(ns_value)}.log"
    cmd = [
        sys.executable,
        str(MAIN_SPATIAL_PATH),
        ns_str,
        "--data-dir",
        str(data_dir),
        "--tree-dir",
        str(tree_dir),
        "--output-dir",
        str(stage_dir),
        "--p2",
        f"{float(p2):g}",
        "--p3",
        f"{float(p3):g}",
        "--lg_cut-off_mass",
        f"{float(lg_cut_off_mass):g}",
        "--run-all",
        str(int(run_all)),
        "--log-mh-min",
        f"{float(log_mh_min):g}",
        "--log-mh-max",
        f"{float(log_mh_max):g}",
        "--n-halos",
        str(int(n_halos)),
    ]
    cmd.extend(["--ex-situ", str(int(ex_situ_mode))])
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=logf, stderr=subprocess.STDOUT)
    print(f"main_spatial finished for N_s={ns_str}. log={log_path}")
    all_path = stage_dir / f"all_{ns_str}.txt"
    if not all_path.exists():
        raise FileNotFoundError(f"Expected formation catalog not found: {all_path}")
    return all_path


def _default_plot_ns_value(ns_values: Sequence[float]) -> float:
    if any(abs(float(ns) - 2.0) < 1.0e-8 for ns in ns_values):
        return 2.0
    return float(ns_values[0])


PLOT_RUNNERS = {
    "gao2023": (PLOT_GAO2023_PATH, "_plots_Gao+2024", "plot_Gao+2024.py", "ns-values"),
    "choksi2018": (PLOT_CHOKSI2018_PATH, "_plots_Choksi+2018", "plot_Choksi+2018.py", "ns-value"),
    "neumayer2020": (PLOT_NEUMAYER2020_PATH, "_plots_Neumayer+2020", "plot_Neumayer+2020.py", "ns-value"),
    "kong2026": (PLOT_KONG2026_PATH, "_plots_Kong+2026", "plot_Kong+2026.py", "ns-value"),
}


def _run_plot_product(kind: str, *, output_dir: Path, ns_values: Sequence[float]) -> Path:
    try:
        script_path, suffix, script_name, ns_mode = PLOT_RUNNERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown plot product kind: {kind}") from exc
    plot_output_dir = output_dir / suffix
    cmd = [sys.executable, str(script_path), "--out_dir", str(output_dir)]
    if ns_mode == "ns-values":
        cmd.extend(["--ns-values", ",".join(f"{float(ns):.1f}" for ns in ns_values)])
    else:
        cmd.extend(["--ns-value", f"{_default_plot_ns_value(ns_values):.1f}"])
    print(f"{script_name} starting. output={plot_output_dir}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"{script_name} finished. output={plot_output_dir}")
    return plot_output_dir


def _build_allcat_table(
    all_rows: np.ndarray,
    *,
    tree_dir: Path,
    z_snap: np.ndarray,
) -> np.ndarray:
    """Assemble the plotting-facing allcat schema from main_spatial output."""

    all_rows_arr = np.asarray(all_rows, dtype=float)
    if all_rows_arr.ndim != 2 or all_rows_arr.shape[1] <= 12:
        raise ValueError(f"all_rows must have the 13-column formation schema; got shape={all_rows_arr.shape}")
    hid_z0 = all_rows_arr[:, 0].astype(int)
    logmh_z0 = _check_array(all_rows_arr[:, 1], "z=0 halo log mass")
    subfind_form = all_rows_arr[:, 2].astype(np.int64)
    logmh_form = _check_array(all_rows_arr[:, 3], "formation halo log mass")
    logmstar_form = _check_array(all_rows_arr[:, 4], "formation stellar log mass")
    _check_array(all_rows_arr[:, 5], "formation gas log mass")
    logm_form = _check_array(all_rows_arr[:, 6], "formation GC log mass")
    z_form = _check_array(all_rows_arr[:, 7], "formation redshift", non_negative=True)
    feh = _check_array(all_rows_arr[:, 8], "formation metallicity")
    r_init = _check_array(all_rows_arr[:, 9], "initial GC radius", positive=True)
    gc_radius_pc = _check_array(all_rows_arr[:, 10], "GC half-mass radius", positive=True)
    sigma_h_msun_pc2 = _check_array(all_rows_arr[:, 11], "GC half-mass surface density", positive=True)
    M_IMBH_init = _check_array(all_rows_arr[:, 12], "initial IMBH mass", non_negative=True)

    mstar_z0 = np.asarray([Mstar_SMHM(Mhalo=10.0 ** m, z=0.0, scatter=False) for m in logmh_z0], dtype=float)
    _check_array(mstar_z0, "z=0 stellar mass from SMHM", positive=True)
    logmstar_z0 = np.log10(mstar_z0)
    snap_form = _nearest_snap(z_form, z_snap)
    is_mpb = _build_ismpb_flags(all_rows_arr, tree_dir)

    return np.column_stack([
        hid_z0.astype(float),
        logmh_z0,
        logmstar_z0,
        logmh_form,
        logmstar_form,
        logm_form,
        z_form,
        feh,
        is_mpb.astype(float),
        subfind_form.astype(float),
        snap_form.astype(float),
        r_init,
        gc_radius_pc,
        sigma_h_msun_pc2,
        M_IMBH_init,])


def _evolve_one_halo_task(
    *,
    hz0: int,
    halo_rows: np.ndarray,
    ns: float,
    ns_tag: str,
    tmp_work_dir: str,
    tree_halo: str,
    ts_m: float,
    ts_r: float,
    eddington_ratio: float,
    out_redshifts: Sequence[float]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[float, float]]:
    """Worker for one halo evolution.

    The per-halo GC evolution is embarrassingly parallel once the formation
    catalog has already been built. Each worker writes its own temporary GCini
    file and returns only the columns needed to assemble the final vectors.
    """

    tmp_work_dir_p = Path(tmp_work_dir)
    tree_halo_p = Path(tree_halo)
    halo_rows_arr = np.asarray(halo_rows, dtype=float)
    if halo_rows_arr.ndim != 2 or halo_rows_arr.shape[1] <= 12:
        raise ValueError(f"Halo {int(hz0)} formation rows are malformed; got shape={halo_rows_arr.shape}")

    gcini_halo = tmp_work_dir_p / f"gcini_halo{hz0}_ns{ns_tag}.txt"
    # The fast evolution code now reads the modern per-GC formation rows,
    # including the fixed IMBH seed mass used by the wanderer branch.
    np.savetxt(gcini_halo, halo_rows_arr, fmt="%.10e", header=FINAL_GC_HEADER)

    depos_halo = _tmp_product_path(tmp_work_dir_p, "depos_halo", hz0, ns_tag)
    gcfin_halo = _tmp_product_path(tmp_work_dir_p, "final_gcs_halo", hz0, ns_tag)
    gcfin_arr, _, central_history, imbh_inventory_by_z = evolve_single_halo(
        ts_m=ts_m,
        ts_r=ts_r,
        gcini_path=gcini_halo,
        depos_path=depos_halo,
        gcfin_path=gcfin_halo,
        haloevo_path=tree_halo_p,
        sersic_n=float(ns),
        eddington_ratio=float(eddington_ratio),
        inventory_redshifts=[0.0] + [float(z) for z in out_redshifts])

    gcfin_arr = _check_gcfin_array(gcfin_arr, f"halo {int(hz0)}")
    return (
        int(hz0),
        gcfin_arr[:, 1].astype(int),
        np.asarray(gcfin_arr[:, 2], dtype=float),
        np.asarray(gcfin_arr[:, 4], dtype=float),
        np.asarray(gcfin_arr[:, 6], dtype=float),
        np.asarray(gcfin_arr[:, 8], dtype=float),
        central_history,
        {float(k): float(v) for k, v in imbh_inventory_by_z.items()},)


def _evolve_one_gao_analytic_halo_task(
    *,
    hz0: int,
    halo_global_indices: Sequence[int],
    halo_rows: np.ndarray,
    is_mpb: Sequence[int],
    analytic_rows: np.ndarray,
    ns: float,
    ns_tag: str,
    tmp_work_dir: str,
    tree_halo: str,
    ts_m: float,
    ts_r: float,
    eddington_ratio: float,
    out_redshifts: Sequence[float],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[float, float]]:
    """Evolve MPB rows dynamically and finalise non-MPB rows analytically."""

    tmp_work_dir_p = Path(tmp_work_dir)
    tree_halo_p = Path(tree_halo)
    halo_rows_arr = np.asarray(halo_rows, dtype=float)
    halo_global_indices_arr = np.asarray(halo_global_indices, dtype=int)
    is_mpb_arr = np.asarray(is_mpb, dtype=int)
    analytic_arr = np.asarray(analytic_rows, dtype=float)
    if (
        len(halo_rows_arr) != len(halo_global_indices_arr)
        or len(halo_rows_arr) != len(is_mpb_arr)
        or len(halo_rows_arr) != len(analytic_arr)
    ):
        raise ValueError(f"Halo {int(hz0)} has inconsistent Gao-analytic input lengths.")
    if halo_rows_arr.ndim != 2 or halo_rows_arr.shape[1] <= 12:
        raise ValueError(f"Halo {int(hz0)} formation rows are malformed; got shape={halo_rows_arr.shape}")
    if analytic_arr.ndim != 2 or analytic_arr.shape[1] != 8:
        raise ValueError(f"Halo {int(hz0)} analytic survival rows are malformed; got shape={analytic_arr.shape}")
    if np.any(~np.isin(is_mpb_arr, [0, 1])):
        raise ValueError(f"Halo {int(hz0)} has non-binary MPB flags.")

    analytic_is_mpb = analytic_arr[:, 2].astype(int)
    if np.any(analytic_is_mpb != is_mpb_arr):
        raise ValueError(f"Halo {int(hz0)} analytic survival MPB flags do not match fixed-tree mapping.")

    n_halo = len(halo_rows_arr)
    status = np.zeros(n_halo, dtype=int)
    m_final = np.zeros(n_halo, dtype=float)
    lookback_time_final = np.zeros(n_halo, dtype=float)
    r_final = np.zeros(n_halo, dtype=float)
    M_IMBH_final = np.asarray(halo_rows_arr[:, 12], dtype=float).copy()
    central_history: List[dict] = []

    mpb_positions = np.where(is_mpb_arr == 1)[0]
    if len(mpb_positions) > 0:
        gcini_mpb = tmp_work_dir_p / f"gcini_halo{int(hz0)}_mpb_ns{ns_tag}.txt"
        np.savetxt(gcini_mpb, halo_rows_arr[mpb_positions, :], fmt="%.10e", header=FINAL_GC_HEADER)
        depos_mpb = tmp_work_dir_p / f"depos_halo{int(hz0)}_mpb_ns{ns_tag}.tmp.dat"
        gcfin_mpb = tmp_work_dir_p / f"finalGCs_halo{int(hz0)}_mpb_ns{ns_tag}.tmp.dat"
        gcfin_arr, _, central_history, _ = evolve_single_halo(
            ts_m=ts_m,
            ts_r=ts_r,
            gcini_path=gcini_mpb,
            depos_path=depos_mpb,
            gcfin_path=gcfin_mpb,
            haloevo_path=tree_halo_p,
            sersic_n=float(ns),
            eddington_ratio=float(eddington_ratio),
            inventory_redshifts=[0.0] + [float(z) for z in out_redshifts],
        )
        gcfin_arr = _check_gcfin_array(gcfin_arr, f"halo {int(hz0)} MPB")
        for local_pos, out_row in zip(mpb_positions, gcfin_arr):
            status[local_pos] = int(out_row[1])
            m_final[local_pos] = check_finite_non_negative(float(out_row[2]), name="MPB final GC stellar mass")
            lookback_time_final[local_pos] = _checked_non_negative_time(float(out_row[4]), "MPB final lookback time")
            r_final[local_pos] = check_finite_non_negative(float(out_row[6]), name="MPB final radius")
            M_IMBH_final[local_pos] = check_finite_non_negative(float(out_row[8]), name="MPB final IMBH mass")
    else:
        depos_mpb = None

    t_z0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    for local_pos in np.where(is_mpb_arr == 0)[0]:
        row = halo_rows_arr[local_pos]
        analytic = analytic_arr[local_pos]
        analytic_final_raw = check_finite_non_negative(
            float(analytic[4]),
            name="Gao-style analytic final GC mass",
        )
        M_IMBH_init = check_finite_non_negative(float(analytic[6]), name="analytic initial IMBH mass")
        r_init = check_finite_positive(float(row[9]), name="analytic initial GC radius")
        if analytic_final_raw > 0.0:
            stellar_final = max(float(analytic_final_raw - M_IMBH_init), 0.0)
            if stellar_final > 0.0:
                status_i = STAT_ALIVE
            elif M_IMBH_init > 0.0:
                status_i = STAT_WANDERER
            else:
                status_i = STAT_EXHAUSTED
        else:
            stellar_final = 0.0
            status_i = STAT_WANDERER if M_IMBH_init > 0.0 else STAT_EXHAUSTED
        status[local_pos] = int(status_i)
        m_final[local_pos] = float(stellar_final)
        lookback_time_final[local_pos] = 0.0
        r_final[local_pos] = r_init
        M_IMBH_final[local_pos] = float(M_IMBH_init)

    if np.any(status == 0):
        missing = [int(halo_global_indices_arr[pos]) for pos in np.where(status == 0)[0]]
        raise ValueError(f"Halo {int(hz0)} Gao-analytic evolution did not finalise {len(missing)} GC rows.")

    gcfin_halo = _tmp_product_path(tmp_work_dir_p, "final_gcs_halo", int(hz0), ns_tag)
    with gcfin_halo.open("w", encoding="utf-8") as fgc:
        fgc.write("# " + FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for local_pos, row in enumerate(halo_rows_arr):
            lookback_init_i = _checked_non_negative_time(
                t_z0 - float(Redshift2CosmicAge(check_finite_non_negative(float(row[7]), name="formation redshift"), time_unit="Gyr")),
                "initial lookback time",
            )
            fgc.write(
                f"{local_pos + 1:d} {int(status[local_pos]):d} {float(m_final[local_pos]):.10e} "
                f"{check_finite_positive(10.0 ** float(row[6]), name='initial GC mass'):.10e} "
                f"{float(lookback_time_final[local_pos]):.10e} {float(lookback_init_i):.10e} "
                f"{float(r_final[local_pos]):.10e} {check_finite_positive(float(row[9]), name='initial GC radius'):.10e} "
                f"{float(M_IMBH_final[local_pos]):.10e}\n"
            )

    depos_halo = _tmp_product_path(tmp_work_dir_p, "depos_halo", int(hz0), ns_tag)
    with depos_halo.open("w", encoding="utf-8") as fdep:
        fdep.write("# " + DEPOS_HEADER.replace("\n", "\n# ") + "\n")
        if depos_mpb is not None:
            for row in _iter_numeric_text_lines(depos_mpb):
                fdep.write(row + "\n")

    return (
        int(hz0),
        status,
        m_final,
        lookback_time_final,
        r_final,
        M_IMBH_final,
        list(central_history),
        {},
    )


def _state_from_formation_row(global_index: int, row: np.ndarray) -> dict:
    row_arr = np.asarray(row, dtype=float)
    if row_arr.ndim != 1 or len(row_arr) <= 12:
        raise ValueError(f"Formation row {int(global_index)} is malformed; got shape={row_arr.shape}")
    m_init = check_finite_positive(10.0 ** check_finite(row_arr[6], name="initial GC log mass"), name="initial GC mass")
    current_z = check_finite_non_negative(row_arr[7], name="formation redshift")
    current_r = check_finite_positive(row_arr[9], name="initial GC radius")
    gc_radius_pc = check_finite_positive(row_arr[10], name="GC half-mass radius")
    sigma_h_msun_pc2 = check_finite_positive(row_arr[11], name="GC half-mass surface density")
    M_IMBH_init = check_finite_non_negative(row_arr[12], name="initial IMBH mass")
    return {
        "global_index": int(global_index),
        "hid_z0": int(row_arr[0]),
        "track_id": int(row_arr[2]),
        "logMh_context": check_finite(row_arr[3], name="formation halo log mass"),
        "logMstar_context": check_finite(row_arr[4], name="formation stellar log mass"),
        "logMgas_context": check_finite(row_arr[5], name="formation gas log mass"),
        "current_mass_msun": m_init,
        "current_z": current_z,
        "M_GC_init": m_init,
        "z_GC_init": current_z,
        "feh": check_finite(row_arr[8], name="formation metallicity"),
        "current_r_kpc": current_r,
        "r_init_kpc": current_r,
        "gc_radius_pc": gc_radius_pc,
        "sigma_h_msun_pc2": sigma_h_msun_pc2,
        "M_IMBH_init": M_IMBH_init,
        "M_IMBH_current": M_IMBH_init,
    }


def _extended_gcini_rows_from_states(states: Sequence[dict]) -> np.ndarray:
    rows: List[List[float]] = []
    for state in states:
        current_mass = check_finite_positive(float(state["current_mass_msun"]), name="live continuation current mass")
        current_z = check_finite_non_negative(float(state["current_z"]), name="live continuation redshift")
        current_r = check_finite_positive(float(state["current_r_kpc"]), name="live continuation radius")
        gc_radius_pc = check_finite_positive(float(state["gc_radius_pc"]), name="live continuation GC half-mass radius")
        sigma_h_msun_pc2 = check_finite_positive(float(state["sigma_h_msun_pc2"]), name="live continuation GC half-mass surface density")
        M_IMBH_init = check_finite_non_negative(float(state["M_IMBH_init"]), name="live continuation initial IMBH mass")
        M_IMBH_current = check_finite_non_negative(float(state["M_IMBH_current"]), name="live continuation current IMBH mass")
        rows.append(
            [
                float(state["hid_z0"]),
                float(state["track_id"]),
                float(state["logMh_context"]),
                float(state["logMstar_context"]),
                float(state["logMgas_context"]),
                math.log10(current_mass),
                current_z,
                float(state["M_GC_init"]),
                float(state["z_GC_init"]),
                float(state["feh"]),
                current_r,
                gc_radius_pc,
                sigma_h_msun_pc2,
                M_IMBH_init,
                M_IMBH_current,
                float(state["global_index"]),
            ]
        )
    return np.asarray(rows, dtype=float)


def _required_branch_events(tree_rows: np.ndarray, initial_branches: set[int], mpb_branch: int) -> Dict[int, BranchMergerEvent]:
    mpb_branch = int(mpb_branch)
    if mpb_branch < 0:
        raise ValueError(f"MPB branch ID must be non-negative; got {mpb_branch}")
    required = {int(branch) for branch in initial_branches if int(branch) != int(mpb_branch)}
    if any(branch < 0 for branch in required):
        raise ValueError(f"Branch IDs must be non-negative; got {sorted(required)}")
    events: Dict[int, BranchMergerEvent] = {}
    while True:
        missing = required.difference(events.keys()).difference({int(mpb_branch)})
        if not missing:
            return events
        new_events = _branch_merger_events_by_source(tree_rows, missing)
        unresolved = missing.difference(new_events.keys())
        if unresolved:
            unresolved_preview = ", ".join(str(branch) for branch in sorted(unresolved)[:10])
            raise RuntimeError(
                "Cannot resolve required ex-situ branch merger events for "
                f"{len(unresolved)} branch(es): {unresolved_preview}"
            )
        events.update(new_events)
        for event in new_events.values():
            if int(event.recipient_branch_id) != int(mpb_branch):
                required.add(int(event.recipient_branch_id))


def _cumulative_central_events(events: Sequence[dict], eddington_ratio: float = 0.0) -> List[dict]:
    running_nsc = 0.0
    running_smbh_init = 0.0
    running_smbh_entry = 0.0
    running_smbh_current = 0.0
    t_current = 0.0
    out: List[dict] = []
    ordered = sorted(
        events,
        key=lambda item: (
            float(item.get("t_cosmic_gyr", 0.0)),
            int(item.get("event_order", 0)),
            int(item.get("gc_index", 0)),
        ),
    )
    for event in ordered:
        t_event = _checked_non_negative_time(float(event.get("t_cosmic_gyr", t_current)), "Central-event cosmic time")
        if t_event < t_current - TIME_ROUNDOFF_TOL_GYR:
            raise ValueError(f"Central events are not time-ordered: {t_event} < {t_current}")
        dt_gyr = _checked_non_negative_time(t_event - t_current, "Central-event timestep")
        running_smbh_current = float(grow_eddington_mass_msun(
            running_smbh_current,
            dt_gyr=dt_gyr,
            f_edd=eddington_ratio,
            overflow_policy="warn_inf",
        ))
        t_current = t_event
        delta_nsc = check_finite_non_negative(float(event.get("delta_M_NSC", 0.0)), name="central NSC mass increment")
        delta_smbh_init = check_finite_non_negative(float(event.get("delta_M_SMBH_init", 0.0)), name="central initial BH increment")
        delta_smbh_entry = check_finite_non_negative(float(event.get("delta_M_SMBH_entry", 0.0)), name="central entry BH increment")
        delta_smbh_current = check_finite_non_negative(
            float(event.get("delta_M_SMBH_current", event.get("delta_M_SMBH_entry", 0.0))),
            name="central current BH increment",
        )
        running_nsc += delta_nsc
        running_smbh_init += delta_smbh_init
        running_smbh_entry += delta_smbh_entry
        running_smbh_current += delta_smbh_current
        new_event = dict(event)
        new_event["M_NSC"] = float(running_nsc)
        new_event["M_SMBH_init"] = float(running_smbh_init)
        new_event["M_SMBH_entry"] = float(running_smbh_entry)
        new_event["M_SMBH_current"] = float(running_smbh_current)
        out.append(new_event)
    return out


def _evolve_one_segmented_halo_task(
    *,
    hz0: int,
    halo_global_indices: Sequence[int],
    halo_rows: np.ndarray,
    branch_ids: Sequence[int],
    tree_rows: np.ndarray,
    ns: float,
    ns_tag: str,
    tmp_work_dir: str,
    ts_m: float,
    ts_r: float,
    eddington_ratio: float,
    import_branch_central_masses: bool,
    out_redshifts: Sequence[float],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[float, float]]:
    """Evolve one halo with recursive ex-situ branch continuation."""

    tmp_work_dir_p = Path(tmp_work_dir)
    halo_rows_arr = np.asarray(halo_rows, dtype=float)
    halo_global_indices_arr = np.asarray(halo_global_indices, dtype=int)
    branch_ids_arr = np.asarray(branch_ids, dtype=int)
    tree_rows_arr = np.asarray(tree_rows, dtype=object)
    if len(halo_rows_arr) != len(halo_global_indices_arr) or len(halo_rows_arr) != len(branch_ids_arr):
        raise ValueError(f"Halo {int(hz0)} has inconsistent segmented-evolution input lengths.")
    if halo_rows_arr.ndim != 2 or halo_rows_arr.shape[1] <= 12:
        raise ValueError(f"Halo {int(hz0)} segmented formation rows are malformed; got shape={halo_rows_arr.shape}")
    if np.any(branch_ids_arr < 0):
        raise ValueError(f"Halo {int(hz0)} has negative branch IDs.")
    _check_array(halo_rows_arr[:, 7], f"halo {int(hz0)} formation redshifts", non_negative=True)
    _check_array(halo_rows_arr[:, 9], f"halo {int(hz0)} initial GC radii", positive=True)

    mpb_branch = _mpb_branch_id(tree_rows_arr)
    initial_by_branch: Dict[int, List[dict]] = {}
    local_position_by_global = {int(global_index): pos for pos, global_index in enumerate(halo_global_indices_arr)}
    for global_index, row, branch_id in zip(halo_global_indices_arr, halo_rows_arr, branch_ids_arr):
        initial_by_branch.setdefault(int(branch_id), []).append(_state_from_formation_row(int(global_index), row))

    events_by_source = _required_branch_events(tree_rows_arr, set(initial_by_branch.keys()), mpb_branch)
    children_by_recipient: Dict[int, List[int]] = {}
    for event in events_by_source.values():
        children_by_recipient.setdefault(int(event.recipient_branch_id), []).append(int(event.source_branch_id))

    t_z0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    final_records: Dict[int, tuple[int, float, float, float, float]] = {}
    depos_rows: List[tuple[float, int, str]] = []
    imbh_inventory_by_z: Dict[float, float] = {float(z): 0.0 for z in ([0.0] + [float(v) for v in out_redshifts])}
    memo: Dict[int, dict] = {}
    event_order = 0

    def add_depos_rows(path: Path, shift_gyr: float) -> None:
        for row in _iter_numeric_text_lines(path):
            shifted = _shift_depos_row_lookback(row, shift_gyr)
            parts = shifted.split()
            depos_rows.append((float(parts[0]), int(float(parts[1])), shifted))

    def finalise_state(state: dict, out_row: np.ndarray, segment_final_redshift: float) -> None:
        status_float = check_finite(float(out_row[1]), name="GC evolution status")
        status_i = int(round(status_float))
        if abs(status_float - float(status_i)) > 1.0e-8 or status_i not in VALID_EVOLUTION_STATUS:
            raise ValueError(f"Halo {int(hz0)} has invalid segmented-evolution status {out_row[1]}")
        m_final_i = check_finite_non_negative(float(out_row[2]), name="segmented final GC stellar mass")
        r_final_i = check_finite_non_negative(float(out_row[6]), name="segmented final GC radius")
        m_imbh_final_i = check_finite_non_negative(float(out_row[8]), name="segmented final IMBH mass")
        segment_z = check_finite_non_negative(segment_final_redshift, name="segment final redshift")
        t_segment_end = float(Redshift2CosmicAge(segment_z, time_unit="Gyr"))
        segment_lookback = _checked_non_negative_time(float(out_row[4]), "segment final lookback time")
        lookback_final_z0 = _checked_non_negative_time(
            t_z0 - (t_segment_end - segment_lookback),
            "z=0 final lookback time",
        )
        final_records[int(state["global_index"])] = (
            status_i,
            m_final_i,
            lookback_final_z0,
            r_final_i,
            m_imbh_final_i,
        )

    def process_branch(branch_id: int) -> dict:
        nonlocal event_order
        branch = int(branch_id)
        if branch in memo:
            return memo[branch]

        live_states: List[dict] = [dict(state) for state in initial_by_branch.get(branch, [])]
        central_deltas: List[dict] = []

        for child_branch in sorted(children_by_recipient.get(branch, [])):
            child_result = process_branch(int(child_branch))
            event = events_by_source[int(child_branch)]
            child_smbh_current = float(grow_eddington_mass_msun(
                check_finite_non_negative(float(child_result["M_SMBH_current"]), name="child-branch current BH mass"),
                dt_gyr=_checked_non_negative_time(
                    float(event.t_merge_gyr) - float(child_result["t_smbh_current_gyr"]),
                    "child-branch central BH growth timestep",
                ),
                f_edd=eddington_ratio,
                overflow_policy="warn_inf",
            ))
            child_nsc = check_finite_non_negative(float(child_result["M_NSC"]), name="child-branch NSC mass")
            if import_branch_central_masses and (child_nsc > 0.0 or child_smbh_current > 0.0):
                event_order += 1
                central_deltas.append(
                    {
                        "gc_index": -1,
                        "status": 0,
                        "t_cosmic_gyr": float(event.t_merge_gyr),
                        "redshift": float(event.z_merge),
                        "delta_M_NSC": child_nsc,
                        "delta_M_SMBH_init": check_finite_non_negative(float(child_result["M_SMBH_init"]), name="child-branch initial BH mass"),
                        "delta_M_SMBH_entry": check_finite_non_negative(float(child_result["M_SMBH_entry"]), name="child-branch entry BH mass"),
                        "delta_M_SMBH_current": float(child_smbh_current),
                        "event_order": event_order,
                        "source_branch_id": int(child_branch),
                        "recipient_branch_id": int(branch),
                        "event_type": "branch_import",
                    }
                )
            for survivor in child_result["survivors"]:
                continued = dict(survivor)
                continued["current_r_kpc"] = check_finite_positive(event.r_accretion_kpc, name="branch accretion radius")
                continued["current_z"] = check_finite_non_negative(event.z_merge, name="branch merger redshift")
                live_states.append(continued)

        final_redshift = 0.0 if branch == int(mpb_branch) else check_finite_non_negative(events_by_source[branch].z_merge, name="branch final redshift")
        survivors: List[dict] = []

        if live_states:
            for state in live_states:
                current_z = check_finite_non_negative(float(state["current_z"]), name="live GC redshift")
                if current_z < final_redshift - 1.0e-3:
                    raise ValueError(
                        f"Halo {int(hz0)} branch {branch} has a live GC/import at z={state['current_z']} "
                        f"after the branch final redshift z={final_redshift}."
                    )
                if current_z < final_redshift:
                    state["current_z"] = final_redshift
                    z_gc_init = check_finite_non_negative(float(state["z_GC_init"]), name="live GC initial redshift")
                    if z_gc_init < final_redshift:
                        state["z_GC_init"] = final_redshift

            gcini_segment = tmp_work_dir_p / f"gcini_halo{int(hz0)}_branch{branch}_seg_ns{ns_tag}.txt"
            np.savetxt(gcini_segment, _extended_gcini_rows_from_states(live_states), fmt="%.10e", header=FINAL_GC_HEADER)
            tree_segment = _tmp_product_path(tmp_work_dir_p, "tree_branch", int(hz0), ns_tag, branch_id=branch)
            _write_branch_tree(tree_segment, tree_rows_arr, branch)
            depos_segment = _tmp_product_path(tmp_work_dir_p, "depos_branch", int(hz0), ns_tag, branch_id=branch)
            gcfin_segment = _tmp_product_path(tmp_work_dir_p, "final_gcs_branch", int(hz0), ns_tag, branch_id=branch)
            gcfin_arr, _, local_central, local_imbh_inventory = evolve_single_halo(
                ts_m=ts_m,
                ts_r=ts_r,
                gcini_path=gcini_segment,
                depos_path=depos_segment,
                gcfin_path=gcfin_segment,
                haloevo_path=tree_segment,
                sersic_n=float(ns),
                final_redshift=float(final_redshift),
                eddington_ratio=float(eddington_ratio),
                inventory_redshifts=[0.0] + [float(z) for z in out_redshifts],
            )
            gcfin_arr = _check_gcfin_array(gcfin_arr, f"halo {int(hz0)} branch {branch}")
            for z_value, inventory in local_imbh_inventory.items():
                z_key = check_finite_non_negative(float(z_value), name="IMBH inventory redshift")
                imbh_inventory_by_z[z_key] = imbh_inventory_by_z.get(z_key, 0.0) + check_finite_non_negative(float(inventory), name="IMBH inventory mass")
            shift_gyr = _checked_non_negative_time(
                t_z0 - float(Redshift2CosmicAge(float(final_redshift), time_unit="Gyr")),
                "branch deposit lookback shift",
            )
            add_depos_rows(depos_segment, shift_gyr)
            for event in local_central:
                event_order += 1
                central_deltas.append({**event, "event_order": event_order, "branch_id": branch, "event_type": "local_sink"})

            for state, out_row in zip(live_states, gcfin_arr):
                status_i = int(out_row[1])
                m_stellar = check_finite_non_negative(float(out_row[2]), name="segmented survivor stellar mass")
                M_IMBH_current = check_finite_non_negative(float(out_row[8]), name="segmented survivor IMBH mass")
                if status_i in (1, -4) and branch != int(mpb_branch):
                    if status_i == -4:
                        current_mass = M_IMBH_current
                    else:
                        current_mass = m_stellar + M_IMBH_current
                    if current_mass > 0.0:
                        continued = dict(state)
                        continued["current_mass_msun"] = float(current_mass)
                        continued["current_z"] = float(final_redshift)
                        continued["current_r_kpc"] = float(out_row[6])
                        continued["M_IMBH_current"] = float(M_IMBH_current)
                        survivors.append(continued)
                    else:
                        finalise_state(state, out_row, final_redshift)
                else:
                    finalise_state(state, out_row, final_redshift)

        cumulative = _cumulative_central_events(central_deltas, eddington_ratio=eddington_ratio)
        t_branch_final = float(Redshift2CosmicAge(float(final_redshift), time_unit="Gyr"))
        if cumulative:
            last_time = _checked_non_negative_time(float(cumulative[-1].get("t_cosmic_gyr", t_branch_final)), "last central-event time")
            M_SMBH_current = float(grow_eddington_mass_msun(
                check_finite_non_negative(float(cumulative[-1]["M_SMBH_current"]), name="branch current BH mass"),
                dt_gyr=_checked_non_negative_time(t_branch_final - last_time, "branch central BH growth timestep"),
                f_edd=eddington_ratio,
                overflow_policy="warn_inf",
            ))
            M_NSC_branch = check_finite_non_negative(float(cumulative[-1]["M_NSC"]), name="branch NSC mass")
            M_SMBH_init_branch = check_finite_non_negative(float(cumulative[-1]["M_SMBH_init"]), name="branch initial BH mass")
            M_SMBH_entry_branch = check_finite_non_negative(float(cumulative[-1]["M_SMBH_entry"]), name="branch entry BH mass")
        else:
            M_SMBH_current = 0.0
            M_NSC_branch = 0.0
            M_SMBH_init_branch = 0.0
            M_SMBH_entry_branch = 0.0
        result = {
            "survivors": survivors,
            "central_history": cumulative,
            "M_NSC": M_NSC_branch,
            "M_SMBH_init": M_SMBH_init_branch,
            "M_SMBH_entry": M_SMBH_entry_branch,
            "M_SMBH_current": M_SMBH_current,
            "t_smbh_current_gyr": t_branch_final,
        }
        memo[branch] = result
        return result

    mpb_result = process_branch(int(mpb_branch))
    missing = [int(idx) for idx in halo_global_indices_arr if int(idx) not in final_records]
    if missing:
        raise ValueError(f"Halo {int(hz0)} segmented evolution did not finalise {len(missing)} GC rows.")

    n_halo = len(halo_rows_arr)
    status = np.zeros(n_halo, dtype=int)
    m_final = np.zeros(n_halo, dtype=float)
    lookback_time_final = np.zeros(n_halo, dtype=float)
    r_final = np.zeros(n_halo, dtype=float)
    M_IMBH_final = np.zeros(n_halo, dtype=float)
    ns_tag = str(ns_tag)
    gcfin_halo = _tmp_product_path(tmp_work_dir_p, "final_gcs_halo", int(hz0), ns_tag)
    with gcfin_halo.open("w", encoding="utf-8") as fgc:
        fgc.write("# " + FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for global_index in halo_global_indices_arr:
            local_pos = int(local_position_by_global[int(global_index)])
            row = halo_rows_arr[local_pos]
            status_i, m_final_i, lookback_i, r_final_i, M_IMBH_final_i = final_records[int(global_index)]
            lookback_init_i = _checked_non_negative_time(
                t_z0 - float(Redshift2CosmicAge(check_finite_non_negative(float(row[7]), name="formation redshift"), time_unit="Gyr")),
                "initial lookback time",
            )
            fgc.write(
                f"{local_pos + 1:d} {int(status_i):d} {float(m_final_i):.10e} {check_finite_positive(10.0 ** float(row[6]), name='initial GC mass'):.10e} "
                f"{float(lookback_i):.10e} {float(lookback_init_i):.10e} "
                f"{float(r_final_i):.10e} {check_finite_positive(float(row[9]), name='initial GC radius'):.10e} {float(M_IMBH_final_i):.10e}\n"
            )
            status[local_pos] = int(status_i)
            m_final[local_pos] = float(m_final_i)
            lookback_time_final[local_pos] = float(lookback_i)
            r_final[local_pos] = float(r_final_i)
            M_IMBH_final[local_pos] = float(M_IMBH_final_i)

    depos_halo = _tmp_product_path(tmp_work_dir_p, "depos_halo", int(hz0), ns_tag)
    with depos_halo.open("w", encoding="utf-8") as fdep:
        fdep.write("# " + DEPOS_HEADER.replace("\n", "\n# ") + "\n")
        for _, _, row in sorted(depos_rows, key=lambda item: (-item[0], item[1])):
            fdep.write(row + "\n")

    return (
        int(hz0),
        status,
        m_final,
        lookback_time_final,
        r_final,
        M_IMBH_final,
        list(mpb_result["central_history"]),
        {float(k): float(v) for k, v in imbh_inventory_by_z.items()},
    )


def _run_single_ns_pipeline(
    *,
    ns: float,
    data_dir: Path,
    tree_dir: Path,
    output_dir: Path,
    stage_root: Path,
    tmp_gcini_root: Path,
    z_snap: np.ndarray,
    p2: float,
    p3: float,
    lg_cut_off_mass: float,
    ex_situ_mode: int,
    run_all: int,
    log_mh_min: float,
    log_mh_max: float,
    n_halos: int,
    ts_m: float,
    ts_r: float,
    eddington_ratio: float,
    out_redshifts: Sequence[float],
    jobs: int) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run formation + evolution for one Sersic index value.

    Each ``N_s`` gets its own isolated temporary output directory for
    ``main_spatial.py`` and its own temporary GCini directory. That avoids file
    collisions when multiple ``N_s`` values are processed concurrently, while
    still reading the raw data directly from the bundled project ``data/``.
    """

    ns_tag = _ns_tag(ns)
    p2_tag = _fmt_param_tag(p2)
    p3_tag = _fmt_param_tag(p3)
    ns_output_dir = _ns_output_dir(output_dir, ns)
    ex_situ_mode = int(ex_situ_mode)
    if ex_situ_mode not in EX_SITU_MODES:
        raise ValueError(f"Unsupported ex-situ mode {ex_situ_mode}; expected one of {EX_SITU_MODES}")

    stage_dir = stage_root / f"ns{ns_tag}"
    tmp_gcini_dir = tmp_gcini_root / f"ns{ns_tag}"
    _clear_dir_contents(stage_dir)
    _clear_dir_contents(tmp_gcini_dir)

    all_path = _run_main_spatial_for_ns(
        stage_dir,
        data_dir,
        tree_dir,
        ns,
        p2=p2,
        p3=p3,
        lg_cut_off_mass=lg_cut_off_mass,
        ex_situ_mode=ex_situ_mode,
        run_all=run_all,
        log_mh_min=log_mh_min,
        log_mh_max=log_mh_max,
        n_halos=n_halos)
    eff_summary_src = stage_dir / f"eff_radius_summary_{float(ns):.1f}.csv"
    if eff_summary_src.exists():
        shutil.copy2(eff_summary_src, ns_output_dir / eff_summary_src.name)
    all_rows_raw = _read_main_spatial_all(all_path)
    row_order = _stable_row_order(all_rows_raw)
    # The raw all_<Ns>.txt order depends on legacy tree traversal and can vary
    # with filesystem order. Sorting once here makes later ns-to-ns comparisons
    # and merged output tables deterministic.
    all_rows = np.array(all_rows_raw[row_order], dtype=float, copy=True)
    analytic_rows = None
    if ex_situ_mode == EX_SITU_GAO_ANALYTIC:
        analytic_path = stage_dir / f"analytic_survival_{float(ns):.1f}.txt"
        analytic_raw = _read_analytic_survival(analytic_path)
        if len(analytic_raw) != len(all_rows_raw):
            raise ValueError(
                f"{analytic_path} row count ({len(analytic_raw)}) does not match {all_path} ({len(all_rows_raw)})."
            )
        analytic_rows = np.array(analytic_raw[row_order], dtype=float, copy=True)
        if np.any(analytic_rows[:, 0].astype(int) != all_rows[:, 0].astype(int)):
            raise ValueError(f"{analytic_path} halo IDs do not match sorted formation rows.")
        if not np.allclose(analytic_rows[:, 6], all_rows[:, 12], rtol=1.0e-8, atol=1.0e-4):
            raise ValueError(f"{analytic_path} M_IMBH_init values do not match sorted formation rows.")
        if not np.allclose(analytic_rows[:, 7], all_rows[:, 9], rtol=1.0e-8, atol=1.0e-4):
            raise ValueError(f"{analytic_path} rGalaxy values do not match sorted formation rows.")
    invalid_initial_r = (~np.isfinite(all_rows[:, 9])) | (all_rows[:, 9] <= 0.0)
    if np.any(invalid_initial_r):
        raise ValueError(
            f"{all_path} contains {int(np.sum(invalid_initial_r))} invalid initial GC radii "
            "after formation-time validation."
        )

    hid_z0 = all_rows[:, 0].astype(int)
    m_final = np.zeros(len(all_rows), dtype=float)
    M_IMBH_final = np.asarray(all_rows[:, 12], dtype=float).copy()
    lookback_time_final = np.zeros(len(all_rows), dtype=float)
    r_final = -1.0 * np.ones(len(all_rows), dtype=float)
    status = np.zeros(len(all_rows), dtype=int)
    unique_halos = np.unique(hid_z0)
    halo_index_map = {int(hz0): np.where(hid_z0 == hz0)[0] for hz0 in unique_halos}
    jobs = max(1, int(jobs))
    central_history_by_halo: Dict[int, List[dict]] = {}
    branch_mode = ex_situ_mode in (EX_SITU_BRANCH_NO_IMPORT, EX_SITU_BRANCH_IMPORT)
    import_branch_central_masses = ex_situ_mode == EX_SITU_BRANCH_IMPORT

    if branch_mode:
        branch_ids = _branch_ids_for_rows(all_rows, tree_dir)
        if jobs == 1:
            for hz0 in unique_halos:
                idx = halo_index_map[int(hz0)]
                print(f"N_s={ns_tag}: segmented ex-situ evolution for halo {hz0} ({len(idx)} GCs)")
                hz0_ret, status_h, m_final_h, lookback_time_final_h, r_final_h, M_IMBH_final_h, central_history, _imbh_inventory = _evolve_one_segmented_halo_task(
                    hz0=int(hz0),
                    halo_global_indices=idx.astype(int),
                    halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                    branch_ids=np.array(branch_ids[idx], dtype=int, copy=True),
                    tree_rows=_read_full_tree_numeric(_tree_file_for_halo(tree_dir, int(hz0))),
                    ns=float(ns),
                    ns_tag=ns_tag,
                    tmp_work_dir=str(tmp_gcini_dir),
                    ts_m=ts_m,
                    ts_r=ts_r,
                    eddington_ratio=eddington_ratio,
                    import_branch_central_masses=import_branch_central_masses,
                    out_redshifts=out_redshifts,
                )
                status[idx] = status_h
                m_final[idx] = m_final_h
                lookback_time_final[idx] = lookback_time_final_h
                r_final[idx] = r_final_h
                M_IMBH_final[idx] = M_IMBH_final_h
                central_history_by_halo[int(hz0_ret)] = list(central_history)
        else:
            max_workers = min(jobs, len(unique_halos))
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for hz0 in unique_halos:
                    idx = halo_index_map[int(hz0)]
                    fut = ex.submit(
                        _evolve_one_segmented_halo_task,
                        hz0=int(hz0),
                        halo_global_indices=idx.astype(int),
                        halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                        branch_ids=np.array(branch_ids[idx], dtype=int, copy=True),
                        tree_rows=_read_full_tree_numeric(_tree_file_for_halo(tree_dir, int(hz0))),
                        ns=float(ns),
                        ns_tag=ns_tag,
                        tmp_work_dir=str(tmp_gcini_dir),
                        ts_m=ts_m,
                        ts_r=ts_r,
                        eddington_ratio=eddington_ratio,
                        import_branch_central_masses=import_branch_central_masses,
                        out_redshifts=out_redshifts,
                    )
                    futures[fut] = int(hz0)

                completed = 0
                for fut in as_completed(futures):
                    hz0_ret, status_h, m_final_h, lookback_time_final_h, r_final_h, M_IMBH_final_h, central_history, _imbh_inventory = fut.result()
                    idx = halo_index_map[hz0_ret]
                    status[idx] = status_h
                    m_final[idx] = m_final_h
                    lookback_time_final[idx] = lookback_time_final_h
                    r_final[idx] = r_final_h
                    M_IMBH_final[idx] = M_IMBH_final_h
                    central_history_by_halo[int(hz0_ret)] = list(central_history)
                    completed += 1
                    if (completed == 1 or completed % 10 == 0 or completed == len(unique_halos)):
                        print(f"N_s={ns_tag}: completed {completed}/{len(unique_halos)} segmented halos")
    elif ex_situ_mode == EX_SITU_GAO_ANALYTIC:
        if analytic_rows is None:
            raise RuntimeError("Gao-style analytic mode requires analytic_survival rows.")
        is_mpb_flags = _build_ismpb_flags(all_rows, tree_dir)
        if jobs == 1:
            for hz0 in unique_halos:
                idx = halo_index_map[int(hz0)]
                tree_halo = _tree_file_for_halo(tree_dir, int(hz0))
                print(f"N_s={ns_tag}: Gao-style analytic ex-situ evolution for halo {hz0} ({len(idx)} GCs)")
                hz0_ret, status_h, m_final_h, lookback_time_final_h, r_final_h, M_IMBH_final_h, central_history, _imbh_inventory = _evolve_one_gao_analytic_halo_task(
                    hz0=int(hz0),
                    halo_global_indices=idx.astype(int),
                    halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                    is_mpb=np.array(is_mpb_flags[idx], dtype=int, copy=True),
                    analytic_rows=np.array(analytic_rows[idx, :], dtype=float, copy=True),
                    ns=float(ns),
                    ns_tag=ns_tag,
                    tmp_work_dir=str(tmp_gcini_dir),
                    tree_halo=str(tree_halo),
                    ts_m=ts_m,
                    ts_r=ts_r,
                    eddington_ratio=eddington_ratio,
                    out_redshifts=out_redshifts,
                )
                status[idx] = status_h
                m_final[idx] = m_final_h
                lookback_time_final[idx] = lookback_time_final_h
                r_final[idx] = r_final_h
                M_IMBH_final[idx] = M_IMBH_final_h
                central_history_by_halo[int(hz0_ret)] = list(central_history)
        else:
            max_workers = min(jobs, len(unique_halos))
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for hz0 in unique_halos:
                    idx = halo_index_map[int(hz0)]
                    tree_halo = _tree_file_for_halo(tree_dir, int(hz0))
                    fut = ex.submit(
                        _evolve_one_gao_analytic_halo_task,
                        hz0=int(hz0),
                        halo_global_indices=idx.astype(int),
                        halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                        is_mpb=np.array(is_mpb_flags[idx], dtype=int, copy=True),
                        analytic_rows=np.array(analytic_rows[idx, :], dtype=float, copy=True),
                        ns=float(ns),
                        ns_tag=ns_tag,
                        tmp_work_dir=str(tmp_gcini_dir),
                        tree_halo=str(tree_halo),
                        ts_m=ts_m,
                        ts_r=ts_r,
                        eddington_ratio=eddington_ratio,
                        out_redshifts=out_redshifts,
                    )
                    futures[fut] = int(hz0)

                completed = 0
                for fut in as_completed(futures):
                    hz0_ret, status_h, m_final_h, lookback_time_final_h, r_final_h, M_IMBH_final_h, central_history, _imbh_inventory = fut.result()
                    idx = halo_index_map[hz0_ret]
                    status[idx] = status_h
                    m_final[idx] = m_final_h
                    lookback_time_final[idx] = lookback_time_final_h
                    r_final[idx] = r_final_h
                    M_IMBH_final[idx] = M_IMBH_final_h
                    central_history_by_halo[int(hz0_ret)] = list(central_history)
                    completed += 1
                    if (completed == 1 or completed % 10 == 0 or completed == len(unique_halos)):
                        print(f"N_s={ns_tag}: completed {completed}/{len(unique_halos)} Gao-analytic halos")
    else:
        raise RuntimeError(f"Unreachable ex-situ mode: {ex_situ_mode}")

    allcat = _build_allcat_table(
        all_rows,
        tree_dir=tree_dir,
        z_snap=z_snap,
    )
    allcat_ns_path = ns_output_dir / f"allcat_ns{ns_tag}_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
    np.savetxt(allcat_ns_path, allcat, fmt=ALLCAT_FMT, header=ALLCAT_HEADER)

    _combine_per_halo_outputs(
        per_halo_dir=tmp_gcini_dir,
        ns_output_dir=ns_output_dir,
        ns_value=ns,
        halo_ids=unique_halos,
        all_rows=all_rows,
    )

    summary_df = pd.DataFrame(
        {
            "ns": np.full(len(all_rows), float(ns)),
            "hid_z0": hid_z0.astype(int),
            "status": status.astype(int),
            "M_GC_final": m_final,
            "M_IMBH_init": np.asarray(all_rows[:, 12], dtype=float),
            "M_IMBH_final": M_IMBH_final,
            "r_final_kpc": r_final,
        }
    )
    halo_summary_df = _build_halo_summary_table(
        all_rows=all_rows,
        status=status,
        m_final=m_final,
        M_IMBH_final=M_IMBH_final,
        central_history_by_halo=central_history_by_halo,
        eddington_ratio=eddington_ratio,
    )
    halo_summary_df.to_csv(ns_output_dir / f"haloSummary_ns{ns_tag}.csv", index=False)
    halo_summary_by_z_df = _build_halo_summary_by_z_table(
        all_rows=all_rows,
        status=status,
        lookback_time_final_gyr=lookback_time_final,
        out_redshifts=out_redshifts,
        tree_dir=tree_dir,
        per_halo_dir=tmp_gcini_dir,
        ns_tag=ns_tag,
        central_history_by_halo=central_history_by_halo,
        eddington_ratio=eddington_ratio,
    )
    halo_summary_by_z_df.to_csv(ns_output_dir / _ns_product_name("halo_summary_by_z", ns), index=False)
    return float(ns), allcat[:, 0].astype(int), summary_df, halo_summary_df, halo_summary_by_z_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run the High-z SMBHs Python GC pipeline using the copied new/src and data layout, with an optional fixed-tree directory override."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,)
    parser.add_argument("--output", type=Path, default=Path("/lingshan/disk3/subonan/_outputs/Gao+2024"), help="Output directory.")
    parser.add_argument("--tree-dir", type=Path, default=None, help="Optional fixed-tree input directory. Defaults to the bundled data/fixed_trees_large_spin in this repository.")
    parser.add_argument(
        "--clear-output",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Output clearing mode: 0 keep existing output with a warning if non-empty, 1 ask before clearing, 2 clear without asking.",
    )
    parser.add_argument(
        "--ns-values",
        type=str,
        default=",".join(str(v) for v in NS_VALUES_DEFAULT),
        help="Comma-separated N_s values to run.",
    )

    # Physics/evolution controls used by the active Python GCevo rewrite.
    parser.add_argument("--ts-m", type=float, default=0.2, help="adaptive mass-loss timestep factor for evo")
    parser.add_argument("--ts-r", type=float, default=0.2, help="adaptive orbital-decay timestep factor for evo")
    parser.add_argument("--out_z", "--extra_out_z_list", dest="out_z", type=str, default=OUT_Z_DEFAULT, help=("comma-separated output redshifts for halo-level "
                        "sunk-BH and NSC-mass summaries; z=0 is always included automatically"))
    parser.add_argument(
        "--Eddington",
        type=float,
        default=0.0,
        help="dimensionless Eddington ratio for uncapped central BH growth; GC-hosted and non-central wandering IMBHs do not accrete; 0 disables central growth",
    )

    # Formation-model parameters passed directly to main_spatial.py.
    parser.add_argument("--p2", type=float, default=6.75, help="GC formation-efficiency normalization in M_GC = 3e-5 * p2 * M_gas / f_b")
    parser.add_argument("--p3", type=float, default=0.5, help="threshold in ((Delta M_h / M_h) / Delta t) above which a GC formation event is triggered")
    parser.add_argument("--lg_cut-off_mass", dest="lg_cut_off_mass", type=float, default=12.0, help="log10 Schechter cutoff mass Mc in Msun for the GC initial mass function")
    parser.add_argument(
        "--ex-situ",
        dest="ex_situ",
        type=int,
        choices=list(EX_SITU_MODES),
        default=EX_SITU_GAO_ANALYTIC,
        help=(
            "ex-situ GC treatment: 0 Gao+2024-style analytic survival for non-MPB GCs; "
            "1 branch evolution without importing satellite central NSC/BH masses; "
            "2 branch evolution with satellite central NSC/BH import"
        ),
    )
    parser.add_argument("--run-all", type=int, default=1, help="if 1, process all halos in the tree set; if 0, apply the mass window and halo count below")
    parser.add_argument("--log-mh-min", type=float, default=11.5, help="minimum descendant z=0 host-halo log mass when --run-all=0")
    parser.add_argument("--log-mh-max", type=float, default=12.5, help="maximum descendant z=0 host-halo log mass when --run-all=0")
    parser.add_argument("--n-halos", type=int, default=10, help="maximum number of halos to run when --run-all=0")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel halo-evolution workers per N_s run.")
    parser.add_argument("--ns-jobs", type=int, default=1, help="Concurrent N_s pipelines.")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=SCRATCH_DIR_DEFAULT,
        help="Shared root directory under which one unique transient-work directory is created per run.",
    )
    parser.add_argument(
        "--min-scratch-free-gb",
        type=float,
        default=0.0,
        help="Abort before running if the selected scratch filesystem has less than this many GB free; 0 disables the check.",
    )
    parser.add_argument(
        "--plot_Gao+2024",
        dest="plot_gao2023",
        action="store_true",
        help="Run plot/plot_Gao+2024.py automatically after the simulation and write figures to <output>/_plots_Gao+2024.",
    )
    parser.add_argument(
        "--plot_Choksi+2018",
        dest="plot_choksi2018",
        action="store_true",
        help="Run plot/plot_Choksi+2018.py automatically after the simulation and write figures to <output>/_plots_Choksi+2018.",
    )
    parser.add_argument(
        "--plot_Neumayer+2020",
        dest="plot_neumayer2020",
        action="store_true",
        help="Run plot/plot_Neumayer+2020.py automatically after the simulation and write figures to <output>/_plots_Neumayer+2020.",
    )
    parser.add_argument(
        "--plot_Kong+2026",
        dest="plot_kong2026",
        action="store_true",
        help="Run plot/plot_Kong+2026.py automatically after the simulation and write figures to <output>/_plots_Kong+2026.",
    )
    old_ex_situ_flag = "--ex-situ" + "NSC"
    if any(arg == old_ex_situ_flag or arg.startswith(old_ex_situ_flag + "=") for arg in sys.argv[1:]):
        parser.error(f"{old_ex_situ_flag} has been removed; use --ex-situ")
    args = parser.parse_args()

    data_dir, tree_dir = _check_project_layout(
        plot_gao2023_requested=bool(args.plot_gao2023),
        plot_choksi2018_requested=bool(args.plot_choksi2018),
        plot_neumayer2020_requested=bool(args.plot_neumayer2020),
        plot_kong2026_requested=bool(args.plot_kong2026),
        tree_dir=args.tree_dir,
    )

    output_dir = args.output.resolve()
    if args.clear_output == 0:
        _warn_if_output_nonempty(output_dir)
    elif args.clear_output == 1:
        _confirm_clear_output(output_dir)
        _clear_dir_contents(output_dir)
    elif args.clear_output == 2:
        _clear_dir_contents(output_dir)

    ns_values = _parse_ns_values(args.ns_values)
    out_redshifts = _parse_out_z(args.out_z)
    eddington_ratio = check_eddington_ratio(args.Eddington)
    z_snap = _build_snap_map(SNAPS_PATH)

    run_scratch_dir = None
    run_succeeded = False

    try:
        # These directories are only transient working areas for the formation
        # and per-halo evolution stages. They no longer contain any copied or
        # linked copies of the raw Gao+2024 input data.
        scratch_root = _prepare_scratch_root(args.scratch_dir, args.min_scratch_free_gb)
        run_scratch_dir = _make_run_scratch_dir(scratch_root)
        stage_root = run_scratch_dir / "main_spatial"
        tmp_gcini_root = run_scratch_dir / "gcini"
        stage_root.mkdir(parents=True, exist_ok=False)
        tmp_gcini_root.mkdir(parents=True, exist_ok=False)
        print(f"TEMP_STAGE {stage_root}")
        print(f"TEMP_GCINI {tmp_gcini_root}")

        p2_tag = _fmt_param_tag(args.p2)
        p3_tag = _fmt_param_tag(args.p3)

        t0 = time.time()
        summary_parts: List[pd.DataFrame] = []
        halo_summary_parts: List[pd.DataFrame] = []
        halo_summary_by_z_parts: List[pd.DataFrame] = []
        template_halo_ids: np.ndarray | None = None
        ns_jobs = max(1, int(args.ns_jobs))
        ns_results: Dict[float, tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

        if ns_jobs == 1 or len(ns_values) == 1:
            for ns in ns_values:
                ns_ret, halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret = _run_single_ns_pipeline(
                    ns=ns,
                    data_dir=data_dir,
                    tree_dir=tree_dir,
                    output_dir=output_dir,
                    stage_root=stage_root,
                    tmp_gcini_root=tmp_gcini_root,
                    z_snap=z_snap,
                    p2=args.p2,
                    p3=args.p3,
                    lg_cut_off_mass=args.lg_cut_off_mass,
                    ex_situ_mode=args.ex_situ,
                    run_all=args.run_all,
                    log_mh_min=args.log_mh_min,
                    log_mh_max=args.log_mh_max,
                    n_halos=args.n_halos,
                    ts_m=args.ts_m,
                    ts_r=args.ts_r,
                    eddington_ratio=eddington_ratio,
                    out_redshifts=out_redshifts,
                    jobs=args.jobs)
                ns_results[ns_ret] = (halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret)
        else:
            max_ns_workers = min(ns_jobs, len(ns_values))
            if args.jobs > 1:
                print(
                    "Running concurrent N_s pipelines with nested halo workers: "
                    f"ns_jobs={max_ns_workers}, halo_jobs={max(1, int(args.jobs))}, "
                    f"max_processes~{max_ns_workers * max(1, int(args.jobs))}"
                )
            futures = {}
            with ThreadPoolExecutor(max_workers=max_ns_workers) as ex:
                for ns in ns_values:
                    fut = ex.submit(
                        _run_single_ns_pipeline,
                        ns=ns,
                        data_dir=data_dir,
                        tree_dir=tree_dir,
                        output_dir=output_dir,
                        stage_root=stage_root,
                        tmp_gcini_root=tmp_gcini_root,
                        z_snap=z_snap,
                        p2=args.p2,
                        p3=args.p3,
                        lg_cut_off_mass=args.lg_cut_off_mass,
                        ex_situ_mode=args.ex_situ,
                        run_all=args.run_all,
                        log_mh_min=args.log_mh_min,
                        log_mh_max=args.log_mh_max,
                        n_halos=args.n_halos,
                        ts_m=args.ts_m,
                        ts_r=args.ts_r,
                        eddington_ratio=eddington_ratio,
                        out_redshifts=out_redshifts,
                        jobs=args.jobs)
                    futures[fut] = float(ns)

                completed = 0
                for fut in as_completed(futures):
                    ns_ret, halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret = fut.result()
                    ns_results[ns_ret] = (halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret)
                    completed += 1
                    print(f"N_s batch: completed {completed}/{len(ns_values)} N_s runs")

        for ns in ns_values:
            halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret = ns_results[float(ns)]
            summary_parts.append(summary_df_ret)
            halo_summary_parts.append(halo_summary_df_ret.assign(ns=float(ns)))
            halo_summary_by_z_parts.append(halo_summary_by_z_df_ret.assign(ns=float(ns)))
            if template_halo_ids is None:
                template_halo_ids = halo_ids_ret
                template_ns_tag = _ns_tag(ns)
                template_ns_path = (
                    output_dir
                    / f"ns{template_ns_tag}"
                    / f"allcat_ns{template_ns_tag}_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
                )
                template_allcat = np.loadtxt(template_ns_path, ndmin=2)
                # Keep one top-level allcat template for downstream tools that
                # accept the historical single-file entry point and then infer
                # the per-N_s directories from it.
                template_path = output_dir / f"allcat_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
                np.savetxt(template_path, template_allcat, fmt=ALLCAT_FMT, header=ALLCAT_HEADER)

        if template_halo_ids is None:
            raise RuntimeError("No catalogs were produced; check input trees and model parameters.")

        _build_mpb_csv_from_trees(
            tree_dir=tree_dir,
            halo_ids=template_halo_ids,
            z_snap=z_snap,
            out_csv=output_dir / "mpb_from_fixed_trees.csv",
        )
        _write_halo_tree_lookup(
            output_dir=output_dir,
            tree_dir=tree_dir,
            halo_ids=template_halo_ids,
        )

        _combine_all_ns_outputs(output_dir=output_dir, ns_values=ns_values)

        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(output_dir / "python_evo_summary.csv", index=False)
        halo_summary = pd.concat(halo_summary_parts, ignore_index=True)
        halo_summary.to_csv(output_dir / "haloSummary_all.csv", index=False)
        halo_summary_by_z = pd.concat(halo_summary_by_z_parts, ignore_index=True)
        halo_summary_by_z.to_csv(output_dir / "haloSummaryByZ_all.csv", index=False)
        central_warning_values = pd.concat(
            [
                halo_summary[["M_SMBH_final"]],
                halo_summary_by_z[["M_SMBH_final"]],
            ],
            ignore_index=True,
        )["M_SMBH_final"].to_numpy(dtype=float)
        finite_central_warning_values = central_warning_values[np.isfinite(central_warning_values)]
        central_bh_warning_count = int(np.sum(finite_central_warning_values > CENTRAL_BH_WARNING_MASS_MSUN))
        central_bh_warning_max = (
            float(np.max(finite_central_warning_values)) if len(finite_central_warning_values) else 0.0
        )
        metadata = {
            "metadata_schema": "nsc_ex_situ_modes_v1",
            "tree_dir": str(tree_dir.resolve()),
            "final_redshift": 0.0,
            "out_z": [float(z) for z in out_redshifts],
            "output_redshifts": [0.0] + [float(z) for z in out_redshifts],
            "ts_m": float(args.ts_m),
            "ts_r": float(args.ts_r),
            "Eddington": float(eddington_ratio),
            "Eddington_accretion_scope": (
                "central BH state only; IMBHs inside GCs and non-central wandering IMBHs do not accrete"
            ),
            "EDDINGTON_EPSILON": float(EDDINGTON_EPSILON),
            "central_bh_warning_mass_msun": float(CENTRAL_BH_WARNING_MASS_MSUN),
            "central_bh_warning_count": central_bh_warning_count,
            "central_bh_warning_max_msun": central_bh_warning_max,
            "p2": float(args.p2),
            "p3": float(args.p3),
            "lg_cut_off_mass": float(args.lg_cut_off_mass),
            "ex-situ": int(args.ex_situ),
            "ex_situ_mode": int(args.ex_situ),
            "ex_situ_mode_definition": {
                "0": "Gao+2024-style analytic survival/disruption for non-MPB GCs; MPB GCs use active dynamical NSC evolution",
                "1": "branch evolution with surviving non-central GCs/wanderers released at 0.5 Rvir; satellite central NSC/BH masses are not imported",
                "2": "branch evolution with surviving non-central GCs/wanderers released at 0.5 Rvir; satellite central NSC/BH masses are imported",
            },
            "eff_rad_catalogue_fallback_policy": "catalogue rows with missing matches, zero SFR, invalid radii/fractions, unresolved stellar components, or inconsistent aperture estimates fall back to empirical",
            "run_all": int(args.run_all),
            "log_mh_min": float(args.log_mh_min),
            "log_mh_max": float(args.log_mh_max),
            "n_halos": int(args.n_halos),
            "ns_values": [float(v) for v in ns_values],
            "nsc_radius_pc": float(NSC_RADIUS_PC),
            "nsc_radius_kpc": float(NSC_RADIUS_PC) * 1.0e-3,
            "initial_imbh_mass_column": "M_IMBH_init",
            "final_imbh_mass_column": "M_IMBH_final",
            "initial_imbh_total_column": "M_IMBH_init_tot",
            "final_imbh_total_column": "M_IMBH_final_tot",
            "initial_smbh_mass_column": "M_SMBH_init",
            "final_smbh_mass_column": "M_SMBH_final",
            "final_smbh_mass_source": "stored halo central BH state",
            "central_stellar_mass_column": "M_NSC",
            "central_bh_mass_column": "M_SMBH_final",
            "final_gc_stellar_mass_column": "M_GC_final",
            "m_gc_final_total_msun_definition": (
                "Halo-level surviving GC-system mass, equal to the sum of M_GC_final "
                "over rows with status == 1 only."
            ),
            "M_NSC_definition": (
                "Cumulative stellar mass carried into r <= NSC_RADIUS_PC by intact GCs; "
                "terminal NSC transfers are stored in M_NSC and are not added to deposited radial bookkeeping."
            ),
            "M_IMBH_init_definition": "Initial IMBH seed mass assigned at GC formation.",
            "M_IMBH_final_definition": "Final GC-line BH mass without non-central Eddington growth; for sunk rows this is the central drop-in mass.",
            "M_IMBH_init_tot_definition": "Sum of initial IMBH seed masses over all seeded GC rows.",
            "M_IMBH_final_tot_definition": "z=0 total BH inventory, equal to stored M_SMBH_final plus M_IMBH_final for seeded rows not sunk to the centre.",
            "M_SMBH_init_definition": "Sum of initial seed masses for BHs that enter or are born in the central region.",
            "M_SMBH_final_definition": "Stored halo central BH mass at the relevant output epoch.",
            "M_GC_final_definition": (
                "Final bound stellar mass of each GC outside the central NSC sink; zero is valid and negative values are disallowed."
            ),
            "deposited_mass_bookkeeping": (
                "depos_ns*.dat and depos_all.dat retain radial stellar mass-loss, stripping, exhaustion, "
                "and tidal-disruption profiles outside the terminal NSC-transfer channel. "
                "For non-IMBH GCs that are torn after ending inside the 6 pc aperture, the final residual "
                "stellar mass is terminal but is not added to depos or M_NSC."
            ),
            "haloSummaryByZ_definition": (
                "Redshift-resolved central NSC and central BH state. "
                "Non-central IMBH inventories are not included in haloSummaryByZ; "
                "z=0 total BH inventory is stored in M_IMBH_final_tot in haloSummary."
            ),
            "plot_schema": {
                "finalGCs": list(COMBINED_FINAL_GC_HEADER.splitlines()[0].split()),
                "haloSummary": list(HALO_SUMMARY_COLUMNS),
                "haloSummaryByZ": list(HALO_SUMMARY_BY_Z_COLUMNS),
                "old_column_fallbacks": False,
            },
        }
        if int(args.ex_situ) == EX_SITU_GAO_ANALYTIC:
            metadata["ex_situ_model"] = {
                "non_mpb_gc_evolution": "analytic Gao+2024-style survival/disruption to z=0 using src/config.py cosmology",
                "non_mpb_dynamical_inspiral": False,
                "non_mpb_deposited_mass": "none",
                "non_mpb_final_radius": "r_final_kpc = r_init_kpc = 0.5 Rvir placement radius",
                "non_mpb_wanderer_status": int(STAT_WANDERER),
                "redshift_resolved_noncentral_imbh_inventory": False,
            }
        elif int(args.ex_situ) == EX_SITU_BRANCH_NO_IMPORT:
            metadata["ex_situ_model"] = {
                "branch_continuation": True,
                "central_masses_imported_at_branch_merger": [],
                "surviving_branch_GCs_released_to_recipient": True,
                "release_radius_fraction_of_recipient_Rv": 0.5,
                "redshift_resolved_noncentral_imbh_inventory": False,
            }
        elif int(args.ex_situ) == EX_SITU_BRANCH_IMPORT:
            metadata["ex_situ_model"] = {
                "branch_continuation": True,
                "central_masses_imported_at_branch_merger": ["M_NSC", "M_SMBH_current"],
                "surviving_branch_GCs_released_to_recipient": True,
                "release_radius_fraction_of_recipient_Rv": 0.5,
                "redshift_resolved_noncentral_imbh_inventory": False,
            }
        with (output_dir / RUN_METADATA_NAME).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        plot_outputs: List[Path] = []
        if args.plot_gao2023:
            plot_outputs.append(_run_plot_product(
                "gao2023",
                output_dir=output_dir,
                ns_values=ns_values))
        if args.plot_choksi2018:
            plot_outputs.append(_run_plot_product(
                "choksi2018",
                output_dir=output_dir,
                ns_values=ns_values))
        if args.plot_neumayer2020:
            plot_outputs.append(_run_plot_product(
                "neumayer2020",
                output_dir=output_dir,
                ns_values=ns_values))
        if args.plot_kong2026:
            plot_outputs.append(_run_plot_product(
                "kong2026",
                output_dir=output_dir,
                ns_values=ns_values))

        elapsed = time.time() - t0
        print(
            "DONE "
            f"ns={len(ns_values)} "
            f"halos={len(np.unique(template_halo_ids))} "
            f"rows_per_ns={len(template_halo_ids)} "
            f"elapsed_s={elapsed:.2f}"
        )
        print(f"OUTPUT {output_dir}")
        for plot_output_dir in plot_outputs:
            print(f"PLOTS {plot_output_dir}")
        run_succeeded = True
    finally:
        if run_succeeded and run_scratch_dir is not None:
            _remove_run_scratch_dir(run_scratch_dir)
        elif run_scratch_dir is not None:
            print(f"FAILED_SCRATCH {run_scratch_dir}")


if __name__ == "__main__":
    main()
