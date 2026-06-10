#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Model-output readers and derived tables used by the plotting scripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Ez, H100, Mstar_SMHM, NSC_RAD_PC, CosmicAge2Redshift, Redshift2CosmicAge  # noqa: E402


RUN_METADATA_NAME = "run_metadata.json"
HALO_TREE_LOOKUP_NAME = "halo_tree_lookup.csv"
FULL_PHYSICS_COUNTERPARTS_NAME = "full_physics_counterparts_z0.csv"
NEUMAYER_DIVIDER_NAME = "neumayer2020_fig3_divider.json"

FEH_MIN = -2.3
FEH_MAX = 0.3
GLOBAL_SPLIT_DEFAULT = -0.88
MIN_GMM_COUNT = 20
FB = 0.167
MMR_SLOPE = 0.35
MMR_TURNOVER = 10.5
MAX_FEH = 0.3
TDEP = 0.3
MMR_EVOLUTION = 0.9
T_UNIVERSE_GYR = float(Redshift2CosmicAge(0.0))

ALLCAT_COLUMN_MAP = {
    "hid_z0": "halo_id_z0",
    "logMh_z0": "log10_halo_mass_z0",
    "logMstar_z0": "log10_stellar_mass_z0",
    "logMh_form": "log10_halo_mass_form",
    "logMstar_form": "log10_stellar_mass_form",
    "logM_form": "log10_gc_mass_init",
    "zform": "redshift_form",
    "feh": "metallicity_feh",
    "isMPB": "is_mpb",
    "subfind_form": "subhalo_id_form",
    "snap_form": "snapshot_form",
    "r_galaxy_kpc": "galaxy_radius_form_kpc",
    "gc_radius_pc": "gc_radius_form_pc",
    "sigma_h_msun_pc2": "gc_surface_density_msun_pc2",
    "M_IMBH_init": "imbh_mass_init_msun",
    "imbh_mass_msun": "imbh_mass_init_msun",
}
FINAL_GC_COLUMN_MAP = {
    "halo_id_z0": "halo_id_z0",
    "gc_index_halo": "gc_index_halo",
    "status": "status",
    "M_GC_final": "gc_mass_final_msun",
    "log10_m_final_msun": "log10_gc_mass_final",
    "m_init_msun": "gc_mass_init_msun",
    "lookback_time_final_gyr": "lookback_time_final_gyr",
    "lookback_time_init_gyr": "lookback_time_init_gyr",
    "r_final_kpc": "radius_final_kpc",
    "r_init_kpc": "radius_init_kpc",
    "gc_radius_pc": "gc_radius_form_pc",
    "sigma_h_msun_pc2": "gc_surface_density_msun_pc2",
    "feh": "metallicity_feh",
    "M_IMBH_init": "imbh_mass_init_msun",
    "M_IMBH_final": "imbh_mass_final_msun",
}
HALO_SUMMARY_COLUMN_MAP = {
    "hid_z0": "halo_id_z0",
    "logMh_z0": "log10_halo_mass_z0",
    "n_gc_total": "n_gc_total",
    "n_alive": "n_alive",
    "n_wanderer": "n_wanderer",
    "n_exhausted": "n_exhausted",
    "n_torn": "n_torn",
    "n_sunk_gc": "n_sunk_gc",
    "n_sunk_wanderer": "n_sunk_wanderer",
    "n_sunk": "n_sunk",
    "m_gc_init_total_msun": "gc_mass_init_total_msun",
    "M_GC_init_tot": "gc_mass_init_total_msun",
    "m_gc_final_total_msun": "gc_mass_final_total_msun",
    "M_GC_final_tot": "gc_mass_final_total_msun",
    "M_IMBH_init_tot": "imbh_mass_init_total_msun",
    "M_IMBH_final_tot": "imbh_mass_final_total_msun",
    "M_NSC": "nsc_mass_msun",
    "M_SMBH_init": "central_bh_mass_init_msun",
    "M_SMBH_final": "central_bh_mass_final_msun",
    "ns": "sersic_n",
}
HALO_SUMMARY_BY_Z_COLUMN_MAP = {
    "hid_z0": "halo_id_z0",
    "z_out": "redshift",
    "lookback_to_z0_gyr": "lookback_to_z0_gyr",
    "halo_mass_available": "halo_mass_available",
    "logMh_z_msun": "log10_halo_mass_at_redshift",
    "M_NSC": "nsc_mass_msun",
    "M_SMBH_init": "central_bh_mass_init_msun",
    "M_SMBH_final": "central_bh_mass_final_msun",
    "z_depos_sampled": "deposit_sample_redshift",
    "lookback_depos_sampled_gyr": "deposit_sample_lookback_gyr",
    "depos_time_match_delta_gyr": "deposit_sample_time_delta_gyr",
    "ns": "sersic_n",
}


@dataclass(frozen=True)
class OutputPaths:
    out_dir: Path
    root_allcat: Path
    mpb: Path
    run_metadata: Path
    ns_value: float | None = None
    ns_dir: Path | None = None
    ns_allcat: Path | None = None
    final_gcs: Path | None = None
    deposit: Path | None = None
    halo_summary: Path | None = None
    halo_summary_by_z: Path | None = None


@dataclass
class DepositProfile:
    halo_ids: np.ndarray
    r_inner_kpc: List[np.ndarray]
    r_outer_kpc: List[np.ndarray]
    shell_mass_msun: List[np.ndarray]
    cumulative_mass_msun: List[np.ndarray] | None = None


@dataclass
class GaoModel:
    formed: pd.DataFrame
    mpb: pd.DataFrame
    paths: OutputPaths


@dataclass
class ChoksiModel:
    formed: pd.DataFrame
    catalog: pd.DataFrame
    survivors: pd.DataFrame
    halo_summary: pd.DataFrame
    mpb: pd.DataFrame
    allcat_path: Path
    final_gcs_path: Path
    run_metadata: Dict[str, object]
    split_threshold: float


@dataclass
class ChoksiPaperModel:
    survivors: pd.DataFrame
    split_threshold: float


@dataclass
class NeumayerModel:
    table: pd.DataFrame
    ns_value: float
    nsc_radius_pc: float
    fit_slope: float
    fit_intercept: float
    divider: Dict[str, object] | None = None
    mixed_suite: bool = False


@dataclass
class KongModel:
    formation: pd.DataFrame
    summary_by_z: pd.DataFrame
    final_gc: pd.DataFrame
    paths: OutputPaths


def ns_tag(ns_value: float) -> str:
    return f"{float(ns_value):.1f}".replace(".", "p")


def root_allcat_path(out_dir: Path) -> Path:
    candidates = sorted(Path(out_dir).resolve().glob("allcat_s-*.txt"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"Missing root allcat file in {out_dir}. Expected exactly one file matching allcat_s-*.txt.")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"Found multiple root allcat files in {out_dir}; expected exactly one: {names}")
    return candidates[0].resolve()


def mpb_path(out_dir: Path) -> Path:
    path = Path(out_dir).resolve() / "mpb_from_fixed_trees.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MPB catalog in {out_dir}: {path}")
    return path


def run_metadata_path(out_dir: Path) -> Path:
    return Path(out_dir).resolve() / RUN_METADATA_NAME


def ns_dir(out_dir: Path, ns_value: float) -> Path:
    return Path(out_dir).resolve() / f"ns{ns_tag(ns_value)}"


def ns_allcat_path(out_dir: Path, ns_value: float, root_allcat: Path | None = None) -> Path:
    root = Path(root_allcat) if root_allcat is not None else root_allcat_path(out_dir)
    match = re.match(r"^(?P<prefix>.+?)(?P<suffix>_s-.*\.txt)$", root.name)
    if match is None:
        raise ValueError(f"Cannot infer per-N_s allcat path from template name. Expected '*_s-...txt', got {root.name}")
    tag = ns_tag(ns_value)
    prefix = re.sub(r"_ns[0-9p]+$", "", match.group("prefix"))
    return ns_dir(out_dir, ns_value) / f"{prefix}_ns{tag}{match.group('suffix')}"


def final_gcs_path(out_dir: Path, ns_value: float) -> Path:
    path = ns_dir(out_dir, ns_value) / f"finalGCs_ns{ns_tag(ns_value)}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-N_s final-GC catalogue: {path}")
    return path


def deposit_path(out_dir: Path, ns_value: float) -> Path:
    return ns_dir(out_dir, ns_value) / f"depos_ns{ns_tag(ns_value)}.dat"


def halo_summary_path(out_dir: Path, ns_value: float) -> Path:
    path = ns_dir(out_dir, ns_value) / f"haloSummary_ns{ns_tag(ns_value)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing haloSummary file: {path}")
    return path


def halo_summary_by_z_path(out_dir: Path, ns_value: float) -> Path:
    path = ns_dir(out_dir, ns_value) / f"haloSummaryByZ_ns{ns_tag(ns_value)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing redshift-resolved halo summary: {path}")
    return path


def output_paths(out_dir: Path, ns_value: float | None = None) -> OutputPaths:
    out_dir = Path(out_dir).resolve()
    root = root_allcat_path(out_dir)
    mpb = mpb_path(out_dir)
    if ns_value is None:
        return OutputPaths(out_dir=out_dir, root_allcat=root, mpb=mpb, run_metadata=run_metadata_path(out_dir))
    ns_allcat = ns_allcat_path(out_dir, ns_value, root)
    return OutputPaths(
        out_dir=out_dir,
        root_allcat=root,
        mpb=mpb,
        run_metadata=run_metadata_path(out_dir),
        ns_value=float(ns_value),
        ns_dir=ns_dir(out_dir, ns_value),
        ns_allcat=ns_allcat,
        final_gcs=final_gcs_path(out_dir, ns_value),
        deposit=deposit_path(out_dir, ns_value),
        halo_summary=halo_summary_path(out_dir, ns_value),
        halo_summary_by_z=halo_summary_by_z_path(out_dir, ns_value),
    )


def read_comment_columns(path: Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                text = line[1:].strip()
                if text:
                    return text.split()
    raise ValueError(f"Cannot find header columns in {path}")


def read_headered_whitespace_table(path: Path) -> pd.DataFrame:
    columns = read_comment_columns(path)
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    raw = raw.iloc[:, : len(columns)].copy()
    raw.columns = columns[: raw.shape[1]]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def load_run_metadata(out_dir: Path) -> Dict[str, object]:
    path = run_metadata_path(out_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _rename_existing_columns(table: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rename = {old: new for old, new in mapping.items() if old in table.columns and new not in table.columns}
    return table.rename(columns=rename)


def _add_aliases(table: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    out = table.copy()
    for alias, source in aliases.items():
        if source in out.columns and alias not in out.columns:
            out[alias] = out[source]
    return out


def load_allcat(path: Path) -> pd.DataFrame:
    raw = read_headered_whitespace_table(path)
    table = _rename_existing_columns(raw, ALLCAT_COLUMN_MAP)
    required = ["halo_id_z0", "log10_halo_mass_z0", "log10_halo_mass_form", "log10_stellar_mass_form", "log10_gc_mass_init", "redshift_form", "metallicity_feh"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required allcat columns after normalisation: {missing}")
    table = table.dropna(subset=required).copy()
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    for int_col in ["is_mpb", "subhalo_id_form", "snapshot_form"]:
        if int_col in table.columns:
            table[int_col] = table[int_col].astype(int)
    table["gc_mass_init_msun"] = np.power(10.0, table["log10_gc_mass_init"].to_numpy(dtype=float))
    table["halo_mass_z0_msun"] = np.power(10.0, table["log10_halo_mass_z0"].to_numpy(dtype=float))
    table["halo_mass_form_msun"] = np.power(10.0, table["log10_halo_mass_form"].to_numpy(dtype=float))
    if "log10_stellar_mass_z0" in table.columns:
        table["stellar_mass_z0_msun"] = np.power(10.0, table["log10_stellar_mass_z0"].to_numpy(dtype=float))
    table["stellar_mass_form_msun"] = np.power(10.0, table["log10_stellar_mass_form"].to_numpy(dtype=float))
    return _add_allcat_legacy_aliases(table).reset_index(drop=True)


def _add_allcat_legacy_aliases(table: pd.DataFrame) -> pd.DataFrame:
    return _add_aliases(
        table,
        {
            "hid_z0": "halo_id_z0",
            "logMh_z0": "log10_halo_mass_z0",
            "logMstar_z0": "log10_stellar_mass_z0",
            "logMh_form": "log10_halo_mass_form",
            "logMstar_form": "log10_stellar_mass_form",
            "logM_form": "log10_gc_mass_init",
            "zform": "redshift_form",
            "feh": "metallicity_feh",
            "isMPB": "is_mpb",
            "subfind_form": "subhalo_id_form",
            "snap_form": "snapshot_form",
            "r_galaxy_kpc": "galaxy_radius_form_kpc",
            "gc_radius_pc": "gc_radius_form_pc",
            "sigma_h_msun_pc2": "gc_surface_density_msun_pc2",
            "M_IMBH_init": "imbh_mass_init_msun",
            "M_form": "gc_mass_init_msun",
            "M_halo_z0": "halo_mass_z0_msun",
            "M_halo_form": "halo_mass_form_msun",
            "M_star_z0": "stellar_mass_z0_msun",
            "M_star_form": "stellar_mass_form_msun",
        },
    )


def load_mpb(path: Path) -> pd.DataFrame:
    mpb = pd.read_csv(path)
    for col in ["subhalo_id_z0", "SnapNum"]:
        if col not in mpb.columns:
            raise ValueError(f"MPB table is missing required column '{col}': {path}")
    mpb["subhalo_id_z0"] = pd.to_numeric(mpb["subhalo_id_z0"], errors="coerce").astype(int)
    mpb["SnapNum"] = pd.to_numeric(mpb["SnapNum"], errors="coerce").astype(int)
    if {"SubhaloSpin_x", "SubhaloSpin_y", "SubhaloSpin_z"}.issubset(mpb.columns):
        mpb["spin_mag"] = np.sqrt(
            np.square(pd.to_numeric(mpb["SubhaloSpin_x"], errors="coerce"))
            + np.square(pd.to_numeric(mpb["SubhaloSpin_y"], errors="coerce"))
            + np.square(pd.to_numeric(mpb["SubhaloSpin_z"], errors="coerce"))
        )
        mpb["spin_mag"] = np.where(np.isfinite(mpb["spin_mag"]), mpb["spin_mag"], 500.0)
    else:
        mpb["spin_mag"] = 500.0
    for col in ["logMh_msun_h", "Redshift"]:
        if col in mpb.columns:
            mpb[col] = pd.to_numeric(mpb[col], errors="coerce")
    return mpb


def load_final_gcs(path: Path, expected_halo_ids: np.ndarray | None = None) -> pd.DataFrame:
    table = _rename_existing_columns(read_headered_whitespace_table(path), FINAL_GC_COLUMN_MAP)
    required = ["halo_id_z0", "gc_index_halo", "status", "gc_mass_final_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required final-GC columns after normalisation: {missing}")
    if expected_halo_ids is not None:
        expected_halo_ids = np.asarray(expected_halo_ids, dtype=int)
        if len(table) != len(expected_halo_ids):
            raise ValueError(f"{path} has {len(table)} rows, expected {len(expected_halo_ids)}")
        halo_ids = table["halo_id_z0"].to_numpy(dtype=int)
        if not np.array_equal(halo_ids, expected_halo_ids):
            raise ValueError(f"Row-order mismatch between {path} and the matching allcat_ns file")
        expected_gc_index = np.empty(len(expected_halo_ids), dtype=int)
        for hid in np.unique(expected_halo_ids):
            idx = np.where(expected_halo_ids == int(hid))[0]
            expected_gc_index[idx] = np.arange(1, len(idx) + 1, dtype=int)
        if not np.array_equal(table["gc_index_halo"].to_numpy(dtype=int), expected_gc_index):
            raise ValueError(f"GC index ordering mismatch between {path} and the matching allcat_ns file")
    table["status"] = table["status"].astype(int)
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    table["gc_index_halo"] = table["gc_index_halo"].astype(int)
    if (table["gc_mass_final_msun"].dropna() < 0.0).any():
        raise ValueError(f"{path} contains negative final GC masses")
    table["gc_mass_final_msun"] = np.where(
        np.isfinite(table["gc_mass_final_msun"]) & (table["gc_mass_final_msun"] > 0.0),
        table["gc_mass_final_msun"],
        0.0,
    )
    if "log10_gc_mass_final" not in table.columns:
        m_final = table["gc_mass_final_msun"].to_numpy(dtype=float)
        log_m_final = np.full(len(m_final), np.nan, dtype=float)
        positive = m_final > 0.0
        log_m_final[positive] = np.log10(m_final[positive])
        table["log10_gc_mass_final"] = log_m_final
    table["is_survivor"] = table["status"] == 1
    table["is_sunk"] = table["status"].isin([-3, -5])
    table["is_wanderer"] = table["status"] == -4
    return _add_final_gcs_legacy_aliases(table).reset_index(drop=True)


def _add_final_gcs_legacy_aliases(table: pd.DataFrame) -> pd.DataFrame:
    return _add_aliases(
        table,
        {
            "M_GC_final": "gc_mass_final_msun",
            "log10_M_GC_final": "log10_gc_mass_final",
            "m_init_msun": "gc_mass_init_msun",
            "r_final_kpc": "radius_final_kpc",
            "r_init_kpc": "radius_init_kpc",
            "gc_radius_pc": "gc_radius_form_pc",
            "sigma_h_msun_pc2": "gc_surface_density_msun_pc2",
            "feh": "metallicity_feh",
            "M_IMBH_init": "imbh_mass_init_msun",
            "M_IMBH_final": "imbh_mass_final_msun",
        },
    )


def load_deposit_profile(path: Path) -> DepositProfile:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing deposit profile: {path}")
    table = read_headered_whitespace_table(path)
    required = ["halo_id_z0", "lookback_time_gyr", "r_inner_kpc", "r_outer_kpc", "m_star_with_evo_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required deposit columns: {missing}")
    halo_ids: list[int] = []
    r_inner: list[np.ndarray] = []
    r_outer: list[np.ndarray] = []
    shell_mass: list[np.ndarray] = []
    cumulative: list[np.ndarray] = []
    for hid, group in table.groupby("halo_id_z0", sort=True):
        final_lookback = float(group["lookback_time_gyr"].min())
        final_block = group[np.isclose(group["lookback_time_gyr"].to_numpy(dtype=float), final_lookback)]
        ordered = final_block.sort_values("r_outer_kpc")
        halo_ids.append(int(hid))
        r_inner.append(ordered["r_inner_kpc"].to_numpy(dtype=float))
        r_outer.append(ordered["r_outer_kpc"].to_numpy(dtype=float))
        shell = ordered["m_star_with_evo_msun"].to_numpy(dtype=float)
        shell_mass.append(shell)
        cumulative.append(np.cumsum(shell))
    return DepositProfile(np.asarray(halo_ids, dtype=int), r_inner, r_outer, shell_mass, cumulative)


def load_deposit_profile_for_redshift_summary(deposit_path: Path, summary_rows: pd.DataFrame, final_redshift: float = 0.0) -> DepositProfile:
    """Load the deposit-profile block matched to each selected redshift-summary row."""

    path = Path(deposit_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing deposit profile: {path}")

    summary = _add_halo_summary_by_z_legacy_aliases(summary_rows).copy()
    required_summary = ["halo_id_z0", "redshift"]
    missing_summary = [name for name in required_summary if name not in summary.columns]
    if missing_summary:
        raise ValueError(f"Selected halo summary is missing required deposit-match columns: {missing_summary}")
    for col in summary.columns:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    if summary.empty:
        raise ValueError("Selected halo summary is empty; cannot match deposit-profile blocks.")
    if summary["halo_id_z0"].isna().any():
        raise ValueError("Selected halo summary contains non-finite halo_id_z0 values.")
    summary["halo_id_z0"] = summary["halo_id_z0"].astype(int)
    duplicated = summary["halo_id_z0"].duplicated(keep=False)
    if duplicated.any():
        dupes = sorted(summary.loc[duplicated, "halo_id_z0"].astype(int).unique().tolist())
        raise ValueError(f"Selected halo summary must contain one row per halo; duplicated halo_id_z0 values: {dupes[:10]}")

    table = read_headered_whitespace_table(path)
    required_deposit = [
        "halo_id_z0",
        "lookback_time_gyr",
        "bin_index",
        "r_inner_kpc",
        "r_outer_kpc",
        "m_star_with_evo_msun",
    ]
    missing_deposit = [name for name in required_deposit if name not in table.columns]
    if missing_deposit:
        raise ValueError(f"{path} is missing required deposit columns: {missing_deposit}")
    for col in required_deposit:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if table[required_deposit].isna().any().any():
        raise ValueError(f"{path} contains non-finite values in required deposit columns.")
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    table["bin_index"] = table["bin_index"].astype(int)

    final_age_gyr = float(Redshift2CosmicAge(float(final_redshift)))
    if not np.isfinite(final_age_gyr):
        raise ValueError(f"final_redshift={final_redshift!r} gives a non-finite cosmic age.")

    halo_ids: list[int] = []
    r_inner: list[np.ndarray] = []
    r_outer: list[np.ndarray] = []
    shell_mass: list[np.ndarray] = []
    cumulative: list[np.ndarray] = []
    grouped = {int(hid): group for hid, group in table.groupby("halo_id_z0", sort=True)}

    for row in summary.sort_values("halo_id_z0").itertuples(index=False):
        hid = int(getattr(row, "halo_id_z0"))
        if hid not in grouped:
            raise ValueError(f"Deposit profile {path} has no rows for selected halo_id_z0={hid}.")
        group = grouped[hid]
        lookbacks = np.sort(group["lookback_time_gyr"].to_numpy(dtype=float))
        unique_lookbacks = np.unique(lookbacks)

        target_lookback = getattr(row, "deposit_sample_lookback_gyr", np.nan)
        if np.isfinite(target_lookback):
            block_lookback = float(unique_lookbacks[np.argmin(np.abs(unique_lookbacks - float(target_lookback)))])
            delta = abs(block_lookback - float(target_lookback))
            if delta > 1.0e-6:
                raise ValueError(
                    f"Deposit profile for halo_id_z0={hid} does not contain the requested "
                    f"lookback {float(target_lookback):.9g} Gyr; nearest is {block_lookback:.9g} Gyr."
                )
        else:
            target_redshift = getattr(row, "deposit_sample_redshift", np.nan)
            if not np.isfinite(target_redshift):
                target_redshift = getattr(row, "redshift")
            block_redshifts = np.array([CosmicAge2Redshift(final_age_gyr - float(lb)) for lb in unique_lookbacks], dtype=float)
            if not np.all(np.isfinite(block_redshifts)):
                raise ValueError(f"Cannot infer finite deposit-block redshifts for halo_id_z0={hid}.")
            block_lookback = float(unique_lookbacks[np.argmin(np.abs(block_redshifts - float(target_redshift)))])

        block = group[np.isclose(group["lookback_time_gyr"].to_numpy(dtype=float), block_lookback, rtol=0.0, atol=1.0e-8)]
        ordered = block.sort_values("bin_index")
        bin_index = ordered["bin_index"].to_numpy(dtype=int)
        expected = np.arange(1, len(bin_index) + 1, dtype=int)
        if len(bin_index) == 0 or not np.array_equal(bin_index, expected):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has non-contiguous bin_index values at lookback {block_lookback:.9g} Gyr.")

        rin = ordered["r_inner_kpc"].to_numpy(dtype=float)
        rout = ordered["r_outer_kpc"].to_numpy(dtype=float)
        shell = ordered["m_star_with_evo_msun"].to_numpy(dtype=float)
        if not np.isclose(rin[0], 0.0, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} does not start at r_inner_kpc=0.")
        if not np.isclose(rout[0], 1.0e-3, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} does not start with r_outer_kpc=0.001.")
        if np.any(~np.isfinite(rin)) or np.any(~np.isfinite(rout)) or np.any(~np.isfinite(shell)):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} contains non-finite radial or mass values.")
        if np.any(shell < 0.0):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} contains negative deposited stellar shell masses.")
        if np.any(np.diff(rout) <= 0.0):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has non-increasing r_outer_kpc values.")
        if np.any(rout <= rin):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has non-positive-width radial bins.")

        halo_ids.append(hid)
        r_inner.append(rin)
        r_outer.append(rout)
        shell_mass.append(shell)
        cumulative.append(np.cumsum(shell))

    return DepositProfile(np.asarray(halo_ids, dtype=int), r_inner, r_outer, shell_mass, cumulative)


def load_halo_summary(path: Path) -> pd.DataFrame:
    table = _rename_existing_columns(pd.read_csv(path), HALO_SUMMARY_COLUMN_MAP)
    if "halo_id_z0" not in table.columns:
        raise ValueError(f"{path} is missing required halo identifier column")
    for col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    if "nsc_mass_msun" in table.columns:
        nsc = table["nsc_mass_msun"].to_numpy(dtype=float)
        table["log10_nsc_mass"] = np.where(np.isfinite(nsc) & (nsc > 0.0), np.log10(nsc), np.nan)
    if "central_bh_mass_final_msun" in table.columns:
        bh = table["central_bh_mass_final_msun"].to_numpy(dtype=float)
        table["log10_central_bh_mass"] = np.where(np.isfinite(bh) & (bh > 0.0), np.log10(bh), np.nan)
    if {"central_bh_mass_final_msun", "nsc_mass_msun"}.issubset(table.columns):
        bh = table["central_bh_mass_final_msun"].to_numpy(dtype=float)
        nsc = table["nsc_mass_msun"].to_numpy(dtype=float)
        table["log10_bh_to_nsc_mass_ratio"] = np.where(np.isfinite(bh) & np.isfinite(nsc) & (bh > 0.0) & (nsc > 0.0), np.log10(bh / nsc), np.nan)
    return _add_halo_summary_legacy_aliases(table).sort_values("halo_id_z0").reset_index(drop=True)


def _add_halo_summary_legacy_aliases(table: pd.DataFrame) -> pd.DataFrame:
    return _add_aliases(
        table,
        {
            "hid_z0": "halo_id_z0",
            "logMh_z0": "log10_halo_mass_z0",
            "m_gc_init_total_msun": "gc_mass_init_total_msun",
            "m_gc_final_total_msun": "gc_mass_final_total_msun",
            "M_IMBH_init_tot": "imbh_mass_init_total_msun",
            "M_IMBH_final_tot": "imbh_mass_final_total_msun",
            "M_NSC": "nsc_mass_msun",
            "M_SMBH_init": "central_bh_mass_init_msun",
            "M_SMBH_final": "central_bh_mass_final_msun",
            "M_BH": "central_bh_mass_final_msun",
        },
    )


def load_halo_summary_by_z(path: Path) -> pd.DataFrame:
    table = _rename_existing_columns(pd.read_csv(path), HALO_SUMMARY_BY_Z_COLUMN_MAP)
    required = ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift", "nsc_mass_msun", "central_bh_mass_final_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        if "halo_mass_available" in missing or "log10_halo_mass_at_redshift" in missing:
            raise ValueError(
                f"{path} is missing required same-redshift halo-mass columns: {missing}. "
                "Regenerate the run output with the updated src/run.py."
            )
        raise ValueError(f"{path} is missing required columns after normalisation: {missing}")
    for col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table["halo_id_z0"] = table["halo_id_z0"].astype(int)
    return _add_halo_summary_by_z_legacy_aliases(table)


def _add_halo_summary_by_z_legacy_aliases(table: pd.DataFrame) -> pd.DataFrame:
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


def present_day_halo_mass_from_observed_stellar_mass(stellar_mass: np.ndarray | float) -> np.ndarray | float:
    log_mh_grid = np.linspace(8.0, 16.0, 4096)
    mh_grid = np.power(10.0, log_mh_grid)
    mstar_grid = np.array([Mstar_SMHM(Mhalo=float(mh), z=0.0, scatter=False) for mh in mh_grid], dtype=float)
    log_mstar_grid = np.log10(np.clip(mstar_grid, 1.0e-30, None))
    order = np.argsort(log_mstar_grid)
    sm = np.asarray(stellar_mass, dtype=float)
    valid = np.isfinite(sm) & (sm > 0.0)
    out = np.full(sm.shape, np.nan, dtype=float)
    out[valid] = np.power(10.0, np.interp(np.log10(sm[valid]), log_mstar_grid[order], log_mh_grid[order], left=np.nan, right=np.nan))
    if np.isscalar(stellar_mass):
        return float(out)
    return out


def metallicity_mmr(log_mstar: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
    log_sm = np.asarray(log_mstar, dtype=float)
    zz = np.broadcast_to(np.asarray(z, dtype=float), log_sm.shape)
    feh = MMR_SLOPE * (log_sm - MMR_TURNOVER) - MMR_EVOLUTION * np.log10(1.0 + zz)
    feh = np.minimum(feh, MAX_FEH)
    if np.isscalar(log_mstar):
        return float(feh)
    return feh


def gas_mass_from_stellar_halo(stellar_mass: np.ndarray | float, halo_mass: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
    sm = np.asarray(stellar_mass, dtype=float)
    mh = np.broadcast_to(np.asarray(halo_mass, dtype=float), sm.shape)
    zz = np.broadcast_to(np.asarray(z, dtype=float), sm.shape)
    slope = np.where(sm < 1.0e9, 0.19, 0.33)
    log_ratio = 0.05 - 0.5 - slope * (np.log10(np.clip(sm, 1.0e-30, None)) - 9.0)
    mask_low = zz < 2.0
    mask_mid = (zz >= 2.0) & (zz < 3.0)
    mask_high = zz >= 3.0
    log_ratio = log_ratio.copy()
    log_ratio[mask_low] += (3.0 - TDEP) * np.log10((1.0 + zz[mask_low]) / 3.0) + (3.0 - TDEP) * np.log10(3.0)
    log_ratio[mask_mid] += (1.7 - TDEP) * np.log10((1.0 + zz[mask_mid]) / 3.0) + (3.0 - TDEP) * np.log10(3.0)
    log_ratio[mask_high] += (1.7 - TDEP) * np.log10(4.0 / 3.0) + (3.0 - TDEP) * np.log10(3.0)
    mg = sm * np.power(10.0, log_ratio)
    fstar = sm / np.clip(FB * mh, 1.0e-30, None)
    fgas = mg / np.clip(FB * mh, 1.0e-30, None)
    e_of_z = np.array([Ez(float(redshift)) for redshift in zz.ravel()], dtype=float).reshape(zz.shape)
    mc = 3.6e9 * np.exp(-0.6 * (1.0 + zz)) / H100
    mc_min = 1.5e10 * np.power(180.0, -0.5) / np.clip(e_of_z * H100, 1.0e-30, None)
    mc = np.maximum(mc, mc_min)
    fin = 1.0 / np.power(1.0 + mc / np.clip(mh, 1.0e-30, None), 3.0)
    overflow = fstar + fgas > fin
    fgas = np.where(overflow, np.maximum(fin - fstar, 0.0), fgas)
    mg = fgas * FB * mh
    if np.isscalar(stellar_mass):
        return float(mg)
    return mg


def _solve_gaussian_crossing(mu1: float, sig1: float, w1: float, mu2: float, sig2: float, w2: float) -> float:
    a = 0.5 / (sig2 * sig2) - 0.5 / (sig1 * sig1)
    b = mu1 / (sig1 * sig1) - mu2 / (sig2 * sig2)
    c = 0.5 * mu2 * mu2 / (sig2 * sig2) - 0.5 * mu1 * mu1 / (sig1 * sig1) + np.log(np.clip((w2 * sig1) / np.clip(w1 * sig2, 1.0e-30, None), 1.0e-30, None))
    if abs(a) < 1.0e-12:
        if abs(b) < 1.0e-12:
            return 0.5 * (mu1 + mu2)
        return -c / b
    roots = np.roots([a, b, c])
    real_roots = [float(root.real) for root in roots if abs(root.imag) < 1.0e-8]
    interior = [root for root in real_roots if min(mu1, mu2) <= root <= max(mu1, mu2)]
    if interior:
        return interior[0]
    if real_roots:
        return min(real_roots, key=lambda value: abs(value - 0.5 * (mu1 + mu2)))
    return 0.5 * (mu1 + mu2)


def fit_metallicity_split(values: Iterable[float], min_count: int = MIN_GMM_COUNT) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= FEH_MIN) & (arr <= FEH_MAX)]
    if len(arr) < min_count:
        return GLOBAL_SPLIT_DEFAULT, np.nan, np.nan
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gm.fit(arr.reshape(-1, 1))
    means = gm.means_.ravel()
    variances = gm.covariances_.reshape(-1)
    weights = gm.weights_.ravel()
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    sigmas = np.sqrt(np.clip(variances, 1.0e-8, None))
    split = _solve_gaussian_crossing(means[0], sigmas[0], weights[0], means[1], sigmas[1], weights[1])
    split = float(np.clip(split, FEH_MIN, FEH_MAX))
    blue_peak = float(means[0])
    red_peak = float(means[1]) if means[1] >= -1.0 else np.nan
    return split, blue_peak, red_peak


def _population_from_threshold(feh: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(np.where(feh.to_numpy(dtype=float) <= threshold, "blue", "red"), index=feh.index)


def build_gao_model(out_dir: Path, ns_value: float | None = None) -> GaoModel:
    paths = output_paths(out_dir, ns_value)
    formed = load_allcat(paths.ns_allcat if paths.ns_allcat is not None else paths.root_allcat)
    mpb = load_mpb(paths.mpb)
    return GaoModel(formed=formed, mpb=mpb, paths=paths)


def build_choksi_model(out_dir: Path, ns_value: float) -> ChoksiModel:
    paths = output_paths(out_dir, ns_value)
    assert paths.ns_allcat is not None
    assert paths.final_gcs is not None
    assert paths.halo_summary is not None
    formed = load_allcat(paths.ns_allcat)
    final_gcs = load_final_gcs(paths.final_gcs, expected_halo_ids=formed["halo_id_z0"].to_numpy(dtype=int))
    keep_cols = [col for col in ["status", "gc_mass_final_msun", "log10_gc_mass_final", "radius_final_kpc"] if col in final_gcs.columns]
    catalog = formed.join(final_gcs[keep_cols])
    catalog = _add_final_gcs_legacy_aliases(catalog)
    catalog["status"] = catalog["status"].fillna(0).astype(int)
    survivors = catalog.loc[catalog["status"] == 1].copy().reset_index(drop=True)
    split_threshold, _, _ = fit_metallicity_split(survivors["metallicity_feh"].to_numpy(dtype=float))
    survivors["population"] = _population_from_threshold(survivors["metallicity_feh"], split_threshold)
    survivors = survivors.loc[survivors["gc_mass_final_msun"].to_numpy(dtype=float) > 0.0].copy().reset_index(drop=True)
    survivors["logM_final"] = np.log10(survivors["gc_mass_final_msun"].to_numpy(dtype=float))
    survivors["log10_gc_mass_final"] = survivors["logM_final"]
    survivors["t_form_gyr"] = np.array([Redshift2CosmicAge(float(value)) for value in survivors["redshift_form"].to_numpy(dtype=float)], dtype=float)
    survivors["M_gas_form"] = gas_mass_from_stellar_halo(
        survivors["stellar_mass_form_msun"].to_numpy(dtype=float),
        survivors["halo_mass_form_msun"].to_numpy(dtype=float),
        survivors["redshift_form"].to_numpy(dtype=float),
    )
    survivors["logMgas_form"] = np.log10(np.clip(survivors["M_gas_form"].to_numpy(dtype=float), 1.0e-30, None))
    survivors = _add_allcat_legacy_aliases(_add_final_gcs_legacy_aliases(survivors))
    halo_summary = load_halo_summary(paths.halo_summary)
    mpb = load_mpb(paths.mpb)
    return ChoksiModel(
        formed=formed,
        catalog=catalog,
        survivors=survivors,
        halo_summary=halo_summary,
        mpb=mpb,
        allcat_path=paths.ns_allcat,
        final_gcs_path=paths.final_gcs,
        run_metadata=load_run_metadata(paths.out_dir),
        split_threshold=split_threshold,
    )


def load_choksi_paper_model(path: Path | None = None) -> ChoksiPaperModel:
    if path is None:
        path = PROJECT_ROOT / "data" / "Choksi+2018" / "choksi_supplement" / "model.txt"
        if not path.exists():
            path = PROJECT_ROOT.parent / "data" / "Choksi+2018" / "choksi_supplement" / "model.txt"
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing Choksi+2018 model catalogue: {path}")
    columns = ["halo_id_z0", "log10_halo_mass_z0", "log10_stellar_mass_z0", "log10_halo_mass_form", "log10_stellar_mass_form", "logM_final", "log10_gc_mass_init", "redshift_form", "cluster_age_gyr", "metallicity_feh", "is_mpb"]
    survivors = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=columns, engine="python")
    for col in columns:
        survivors[col] = pd.to_numeric(survivors[col], errors="coerce")
    survivors = survivors.dropna(subset=columns).copy()
    survivors["halo_id_z0"] = survivors["halo_id_z0"].astype(int)
    survivors["is_mpb"] = survivors["is_mpb"].astype(int)
    survivors["gc_mass_final_msun"] = np.power(10.0, survivors["logM_final"].to_numpy(dtype=float))
    survivors["M_gas_form"] = gas_mass_from_stellar_halo(
        np.power(10.0, survivors["log10_stellar_mass_form"].to_numpy(dtype=float)),
        np.power(10.0, survivors["log10_halo_mass_form"].to_numpy(dtype=float)),
        survivors["redshift_form"].to_numpy(dtype=float),
    )
    survivors["logMgas_form"] = np.log10(np.clip(survivors["M_gas_form"].to_numpy(dtype=float), 1.0e-30, None))
    survivors["t_form_gyr"] = T_UNIVERSE_GYR - survivors["cluster_age_gyr"].to_numpy(dtype=float)
    split_threshold, _, _ = fit_metallicity_split(survivors["metallicity_feh"].to_numpy(dtype=float))
    survivors["population"] = _population_from_threshold(survivors["metallicity_feh"], split_threshold)
    survivors = _add_allcat_legacy_aliases(survivors)
    survivors["M_GC_final"] = survivors["gc_mass_final_msun"]
    survivors["feh"] = survivors["metallicity_feh"]
    survivors["zform"] = survivors["redshift_form"]
    survivors["logM_form"] = survivors["log10_gc_mass_init"]
    return ChoksiPaperModel(survivors=survivors.reset_index(drop=True), split_threshold=split_threshold)


def build_neumayer_model(out_dir: Path, ns_value: float, nsc_radius_pc: float = NSC_RAD_PC) -> NeumayerModel:
    paths = output_paths(out_dir, ns_value)
    assert paths.ns_allcat is not None
    assert paths.halo_summary is not None
    formed = load_allcat(paths.ns_allcat)
    halo_summary = load_halo_summary(paths.halo_summary)
    halo = (
        formed.groupby("halo_id_z0", sort=True)
        .agg(log10_halo_mass_z0=("log10_halo_mass_z0", "first"), log10_stellar_mass_z0=("log10_stellar_mass_z0", "first"))
        .reset_index()
    )
    halo["halo_mass_z0_msun"] = np.power(10.0, halo["log10_halo_mass_z0"].to_numpy(dtype=float))
    halo["stellar_mass_z0_msun"] = np.power(10.0, halo["log10_stellar_mass_z0"].to_numpy(dtype=float))
    halo = halo.merge(halo_summary[["halo_id_z0", "nsc_mass_msun", "central_bh_mass_final_msun"]], on="halo_id_z0", how="left", validate="one_to_one")
    if halo["nsc_mass_msun"].isna().any():
        missing = halo.loc[halo["nsc_mass_msun"].isna(), "halo_id_z0"].astype(int).tolist()
        raise ValueError("haloSummary is missing one or more halos present in the model output: " + ", ".join(str(item) for item in missing[:12]))

    mixed_suite, halo_lookup, counterparts, divider = _load_mixed_suite_inputs(Path(out_dir), require_counterparts=True)
    if not mixed_suite:
        raise FileNotFoundError(
            f"{out_dir} is missing required split-style full-physics counterpart data. "
            f"Expected {RUN_METADATA_NAME}, {HALO_TREE_LOOKUP_NAME}, {FULL_PHYSICS_COUNTERPARTS_NAME}, and {NEUMAYER_DIVIDER_NAME}."
        )
    assert halo_lookup is not None
    assert counterparts is not None
    halo_lookup = halo_lookup.copy()
    counterparts = counterparts.copy()
    for col in ["hid_z0", "subhalo_id_z0", "file_index"]:
        halo_lookup[col] = pd.to_numeric(halo_lookup[col], errors="raise")
    for col in [
        "halo_id_z0",
        "subhalo_id_z0_dark",
        "fp_subhalo_id_z0",
        "matched",
        "ambiguous_match",
        "n_fp_matches",
        "stellar_mass_fp_msun",
        "logMstar_fp_msun",
        "g_mag_fp",
        "i_mag_fp",
        "g_minus_i",
    ]:
        if col in counterparts.columns:
            counterparts[col] = pd.to_numeric(counterparts[col], errors="coerce")
    halo = halo.merge(
        halo_lookup[["hid_z0", "simulation_key", "simulation", "subhalo_id_z0", "fixed_tree_basename", "file_index"]].rename(columns={"hid_z0": "halo_id_z0"}),
        on="halo_id_z0",
        how="left",
        validate="one_to_one",
    )
    if halo["simulation_key"].isna().any():
        missing = halo.loc[halo["simulation_key"].isna(), "halo_id_z0"].astype(int).tolist()
        raise ValueError("halo_tree_lookup.csv is missing one or more halos present in the model output: " + ", ".join(str(item) for item in missing[:12]))
    halo = halo.merge(
        counterparts[
            [
                "simulation_key_dark",
                "subhalo_id_z0_dark",
                "fp_subhalo_id_z0",
                "logMstar_fp_msun",
                "g_minus_i",
                "host_type_fig3",
                "colour_class_fig3",
                "matched",
                "ambiguous_match",
                "n_fp_matches",
                "stellar_mass_fp_msun",
                "g_mag_fp",
                "i_mag_fp",
            ]
        ],
        left_on=["simulation_key", "subhalo_id_z0"],
        right_on=["simulation_key_dark", "subhalo_id_z0_dark"],
        how="left",
        validate="one_to_one",
    )
    if halo["matched"].isna().any():
        missing = halo.loc[halo["matched"].isna(), ["halo_id_z0", "simulation_key", "subhalo_id_z0"]]
        preview = ", ".join(f"({int(row.halo_id_z0)}, {row.simulation_key}, {int(row.subhalo_id_z0)})" for row in missing.head(12).itertuples())
        raise ValueError("The cached full-physics counterpart table is missing one or more selected halos: " + preview)
    matched = pd.to_numeric(halo["matched"], errors="coerce").fillna(0).astype(int) == 1
    if int(matched.sum()) == 0:
        raise ValueError("All selected model rows are unmatched to the full-physics counterpart table.")
    halo.loc[~matched, "host_type_fig3"] = "unmatched"
    halo.loc[~matched, "colour_class_fig3"] = "unmatched"
    halo["log10_stellar_mass_plot"] = halo["log10_stellar_mass_z0"].to_numpy(dtype=float)
    use_fp = np.isfinite(pd.to_numeric(halo["logMstar_fp_msun"], errors="coerce").to_numpy(dtype=float))
    halo.loc[use_fp, "log10_stellar_mass_plot"] = halo.loc[use_fp, "logMstar_fp_msun"].to_numpy(dtype=float)
    halo["stellar_mass_plot_msun"] = np.power(10.0, halo["log10_stellar_mass_plot"].to_numpy(dtype=float))
    halo["nsc_to_stellar_mass_fraction"] = halo["nsc_mass_msun"] / halo["stellar_mass_z0_msun"]
    halo["nsc_to_stellar_mass_plot_fraction"] = halo["nsc_mass_msun"] / halo["stellar_mass_plot_msun"]
    nsc = halo["nsc_mass_msun"].to_numpy(dtype=float)
    bh = halo["central_bh_mass_final_msun"].to_numpy(dtype=float)
    positive_nsc = nsc > 0.0
    positive_bh = bh > 0.0
    halo["log10_nsc_mass"] = np.where(positive_nsc, np.log10(nsc), np.nan)
    halo["log10_bh_mass"] = np.where(positive_bh, np.log10(bh), np.nan)
    halo["log10_bh_to_nsc_mass_ratio"] = np.where(positive_bh & positive_nsc, np.log10(bh / nsc), np.nan)
    halo = _add_neumayer_model_aliases(halo)
    fit_mask = np.isfinite(halo["log10_stellar_mass_plot"]) & np.isfinite(halo["log10_nsc_mass"]) & positive_nsc
    if int(fit_mask.sum()) < 2:
        raise ValueError("Need at least two model halos with finite host stellar mass and non-zero M_NSC to fit Fig.12.")
    fit = np.polyfit(halo.loc[fit_mask, "log10_stellar_mass_plot"].to_numpy(dtype=float) - 9.0, halo.loc[fit_mask, "log10_nsc_mass"].to_numpy(dtype=float), 1)
    return NeumayerModel(halo, float(ns_value), float(nsc_radius_pc), float(fit[0]), float(fit[1]), divider=divider, mixed_suite=bool(mixed_suite))


def _load_mixed_suite_inputs(out_dir: Path, require_counterparts: bool) -> tuple[bool, pd.DataFrame | None, pd.DataFrame | None, Dict[str, object] | None]:
    if not require_counterparts:
        return False, None, None, None
    out_dir = Path(out_dir)
    metadata_path = out_dir / RUN_METADATA_NAME
    halo_lookup_path = out_dir / HALO_TREE_LOOKUP_NAME
    if not metadata_path.exists():
        if halo_lookup_path.exists():
            raise FileNotFoundError(f"Mixed-suite output {out_dir} is missing {RUN_METADATA_NAME}. Re-run src/run.py after the provenance update.")
        return False, None, None, None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tree_dir_raw = metadata.get("tree_dir")
    if not tree_dir_raw:
        if halo_lookup_path.exists():
            raise FileNotFoundError(f"Mixed-suite output {out_dir} is missing the resolved tree_dir in {RUN_METADATA_NAME}.")
        return False, None, None, None
    tree_dir = Path(str(tree_dir_raw)).resolve()
    mixed_suite = (tree_dir / "id_lookup_large_dark.csv").is_file() or halo_lookup_path.is_file()
    if not mixed_suite:
        return False, None, None, None
    if not halo_lookup_path.exists():
        raise FileNotFoundError(f"Mixed-suite output {out_dir} is missing {HALO_TREE_LOOKUP_NAME}.")
    data_root = tree_dir.parent
    counterparts_path = data_root / FULL_PHYSICS_COUNTERPARTS_NAME
    divider_path = data_root / NEUMAYER_DIVIDER_NAME
    missing = [str(path) for path in [counterparts_path, divider_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Mixed-suite output requires the cached full-physics counterpart products, "
            f"but these files are missing: {', '.join(missing)}. Run scripts/5_build_full_physics_counterparts.py first."
        )
    return True, pd.read_csv(halo_lookup_path), pd.read_csv(counterparts_path), json.loads(divider_path.read_text(encoding="utf-8"))


def _add_neumayer_model_aliases(table: pd.DataFrame) -> pd.DataFrame:
    return _add_aliases(
        table,
        {
            "hid_z0": "halo_id_z0",
            "logMh_z0": "log10_halo_mass_z0",
            "logMstar_z0": "log10_stellar_mass_z0",
            "M_halo_z0": "halo_mass_z0_msun",
            "M_star_z0": "stellar_mass_z0_msun",
            "M_NSC": "nsc_mass_msun",
            "M_BH": "central_bh_mass_final_msun",
            "logMstar_plot": "log10_stellar_mass_plot",
            "M_star_plot": "stellar_mass_plot_msun",
            "f_NSC": "nsc_to_stellar_mass_fraction",
            "f_NSC_plot": "nsc_to_stellar_mass_plot_fraction",
            "logM_NSC": "log10_nsc_mass",
            "logM_BH": "log10_bh_mass",
            "log_bh_to_nsc": "log10_bh_to_nsc_mass_ratio",
        },
    )


def _stellar_mass_from_halo_mass_at_redshift(halo_mass: np.ndarray | float, redshift: np.ndarray | float) -> np.ndarray | float:
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


def build_kong_model(out_dir: Path, ns_value: float) -> KongModel:
    paths = output_paths(out_dir, ns_value)
    assert paths.ns_allcat is not None
    assert paths.final_gcs is not None
    assert paths.halo_summary_by_z is not None
    formation = load_allcat(paths.ns_allcat)
    summary = load_halo_summary_by_z(paths.halo_summary_by_z)
    final_gc = load_final_gcs(paths.final_gcs)
    z0_lookup = _z0_halo_mass_lookup(paths.root_allcat)
    summary["log10_halo_mass_z0"] = summary["halo_id_z0"].map(z0_lookup)
    missing_z0 = summary.loc[~np.isfinite(summary["log10_halo_mass_z0"].to_numpy(dtype=float)), "halo_id_z0"].unique()
    if len(missing_z0) > 0:
        missing_text = ", ".join(str(int(hid)) for hid in sorted(missing_z0)[:8])
        raise KeyError(f"Root allcat is missing descendant z=0 halo mass for halo(s): {missing_text}")
    summary["halo_mass_z0_msun"] = np.power(10.0, summary["log10_halo_mass_z0"].to_numpy(dtype=float))
    valid_halo_z = (
        (summary["halo_mass_available"].to_numpy(dtype=int) == 1)
        & np.isfinite(summary["log10_halo_mass_at_redshift"].to_numpy(dtype=float))
    )
    summary["halo_mass_at_redshift_msun"] = np.nan
    summary.loc[valid_halo_z, "halo_mass_at_redshift_msun"] = np.power(
        10.0,
        summary.loc[valid_halo_z, "log10_halo_mass_at_redshift"].to_numpy(dtype=float),
    )
    summary["stellar_mass_at_redshift_smhm_msun"] = _stellar_mass_from_halo_mass_at_redshift(
        summary["halo_mass_at_redshift_msun"].to_numpy(dtype=float),
        summary["redshift"].to_numpy(dtype=float),
    )
    mstar = summary["stellar_mass_at_redshift_smhm_msun"].to_numpy(dtype=float)
    summary["log10_stellar_mass_at_redshift_smhm"] = np.where(np.isfinite(mstar) & (mstar > 0.0), np.log10(mstar), np.nan)
    if (summary["nsc_mass_msun"].dropna() < 0.0).any() or (summary["central_bh_mass_final_msun"].dropna() < 0.0).any():
        raise ValueError("haloSummaryByZ contains negative NSC or central-BH masses.")
    summary = _add_kong_aliases(summary)
    return KongModel(formation=formation, summary_by_z=summary, final_gc=final_gc, paths=paths)


def _z0_halo_mass_lookup(root_allcat: Path) -> Dict[int, float]:
    table = load_allcat(root_allcat)
    grouped = table.groupby("halo_id_z0", sort=True)["log10_halo_mass_z0"].first()
    return {int(hid): float(logmh) for hid, logmh in grouped.items()}


def _add_kong_aliases(table: pd.DataFrame) -> pd.DataFrame:
    return _add_aliases(
        table,
        {
            "logMh_z0_msun": "log10_halo_mass_z0",
            "mhalo_z0_msun": "halo_mass_z0_msun",
            "mhalo_z_msun": "halo_mass_at_redshift_msun",
            "mstar_z_smhm_msun": "stellar_mass_at_redshift_smhm_msun",
            "logMstar_z_smhm_msun": "log10_stellar_mass_at_redshift_smhm",
        },
    )
