#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Observational cache readers for the paper plotting scripts.

This module deliberately does not download or rebuild observational data. Missing
cache products raise an explicit error listing the required files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Mstar_SMHM  # noqa: E402


CHOKSI_CACHE_DIR = PROJECT_ROOT / "data" / "Choksi+2018"
NEUMAYER_CACHE_DIR = PROJECT_ROOT / "data" / "Neumayer+2020"
CLIFF_CACHE_DIR = PROJECT_ROOT / "data" / "TheCliff+2026"
if not CHOKSI_CACHE_DIR.is_dir():
    CHOKSI_CACHE_DIR = PROJECT_ROOT.parent / "data" / "Choksi+2018"
if not NEUMAYER_CACHE_DIR.is_dir():
    NEUMAYER_CACHE_DIR = PROJECT_ROOT.parent / "data" / "Neumayer+2020"
if not CLIFF_CACHE_DIR.is_dir():
    CLIFF_CACHE_DIR = PROJECT_ROOT.parent / "data" / "TheCliff+2026"

FEH_MIN = -2.3
FEH_MAX = 0.3
VIRGO_DISTANCE_MODULUS = 31.09
ACS_SOLAR_MAG_Z = 4.51
GC_ML_Z = 1.45

CHOKSI_REQUIRED_CACHE_FILES = [
    CHOKSI_CACHE_DIR / "choksi_supplement" / "data.txt",
    CHOKSI_CACHE_DIR / "choksi_supplement" / "model.txt",
    CHOKSI_CACHE_DIR / "acsvcs" / "hosts_J_ApJS_164_334_acsvcs.tsv",
    CHOKSI_CACHE_DIR / "acsvcs" / "gc_catalog_J_ApJS_180_54_table4.tsv",
    CHOKSI_CACHE_DIR / "vandenberg2013" / "table2_gc_ages.csv",
    CHOKSI_CACHE_DIR / "wagner_kaiser2017" / "lmc_gc_age_metallicity.csv",
    CHOKSI_CACHE_DIR / "lamers2017" / "table1_mdf_summary.csv",
]
NEUMAYER_REQUIRED_CACHE_FILES = [
    NEUMAYER_CACHE_DIR / "neumayer2020_fig03_demographics.csv",
    NEUMAYER_CACHE_DIR / "neumayer2020_fig03_demographics_meta.json",
    NEUMAYER_CACHE_DIR / "neumayer2020_fig12_compilation.csv",
    NEUMAYER_CACHE_DIR / "neumayer2020_fig12_compilation_meta.json",
    NEUMAYER_CACHE_DIR / "original_nsc_review" / "bh_nsc_galmass.csv",
]
CLIFF_OBS_PATH = CLIFF_CACHE_DIR / "cliff_fig14_mbh_mstar_points.csv"


@dataclass(frozen=True)
class GalaxyObs:
    name: str
    m_smbh: float
    m_smbh_err: float
    m_nsc: float
    m_nsc_err: float
    r_nsc_pc: float
    r_nsc_err_pc: float
    color: str


@dataclass(frozen=True)
class GaoObs:
    mw: GalaxyObs
    m31: GalaxyObs


@dataclass
class ChoksiObs:
    systems: pd.DataFrame
    vcs_systems: pd.DataFrame
    acsvcs_hosts: pd.DataFrame
    acsvcs_gc: pd.DataFrame
    mw_age_metallicity: pd.DataFrame
    lmc_age_metallicity: pd.DataFrame
    lamers_summary: pd.DataFrame
    obs_cache_dir: Path


@dataclass
class NeumayerObs:
    table: pd.DataFrame
    cache_dir: Path
    metadata: Dict[str, object]


@dataclass
class NeumayerFig03Obs:
    table: pd.DataFrame
    cache_dir: Path
    metadata: Dict[str, object]


@dataclass
class NeumayerFig13Obs:
    table: pd.DataFrame
    source_path: Path
    duplicate_names: List[str]
    missing_host_mass_count: int
    nonfinite_mass_count: int
    unknown_galtype_count: int
    ucd_upper_limit_count: int


@dataclass
class KongObs:
    table: pd.DataFrame
    source_path: Path


def _require_paths(paths: list[Path], context: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing cached {context} observation files. Plotting no longer downloads or rebuilds caches automatically:\n{missing_text}"
        )


def load_gao_observations() -> GaoObs:
    mw = GalaxyObs("MW", 4.297e6, 0.012e6, 3.15e7, 2.15e7, 5.7, 3.5, "tab:red")
    m31 = GalaxyObs("M31", 1.7e8, 0.6e8, 5.0e7, 0.0, 8.0, 4.0, "tab:purple")
    return GaoObs(mw=mw, m31=m31)


_PRESENT_DAY_SMHM_INVERSE_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def present_day_halo_mass_from_observed_stellar_mass(stellar_mass: np.ndarray | float) -> np.ndarray | float:
    global _PRESENT_DAY_SMHM_INVERSE_CACHE
    if _PRESENT_DAY_SMHM_INVERSE_CACHE is None:
        log_mh_grid = np.linspace(8.0, 16.0, 4096)
        mh_grid = np.power(10.0, log_mh_grid)
        mstar_grid = np.array([Mstar_SMHM(Mhalo=float(mh), z=0.0, scatter=False) for mh in mh_grid], dtype=float)
        log_mstar_grid = np.log10(np.clip(mstar_grid, 1.0e-30, None))
        order = np.argsort(log_mstar_grid)
        _PRESENT_DAY_SMHM_INVERSE_CACHE = (log_mstar_grid[order], log_mh_grid[order])
    log_mstar_grid, log_mh_grid = _PRESENT_DAY_SMHM_INVERSE_CACHE
    sm = np.asarray(stellar_mass, dtype=float)
    valid = np.isfinite(sm) & (sm > 0.0)
    out = np.full(sm.shape, np.nan, dtype=float)
    out[valid] = np.power(10.0, np.interp(np.log10(sm[valid]), log_mstar_grid, log_mh_grid, left=np.nan, right=np.nan))
    if np.isscalar(stellar_mass):
        return float(out)
    return out


def require_choksi_cache(obs_cache_dir: Path = CHOKSI_CACHE_DIR) -> dict[str, Path]:
    obs_cache_dir = Path(obs_cache_dir)
    paths = {
        "choksi_data": obs_cache_dir / "choksi_supplement" / "data.txt",
        "choksi_model": obs_cache_dir / "choksi_supplement" / "model.txt",
        "acsvcs_hosts": obs_cache_dir / "acsvcs" / "hosts_J_ApJS_164_334_acsvcs.tsv",
        "acsvcs_gc_catalog": obs_cache_dir / "acsvcs" / "gc_catalog_J_ApJS_180_54_table4.tsv",
        "vandenberg2013_table2_csv": obs_cache_dir / "vandenberg2013" / "table2_gc_ages.csv",
        "wagner2017_lmc_csv": obs_cache_dir / "wagner_kaiser2017" / "lmc_gc_age_metallicity.csv",
        "lamers2017_summary_csv": obs_cache_dir / "lamers2017" / "table1_mdf_summary.csv",
    }
    _require_paths(list(paths.values()), "Choksi+2018")
    return paths


def load_choksi_system_table(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            galaxy_id, log_sm, mean_feh, err_mean, sigma_feh, err_sigma, blue_peak, red_peak = line.split()[:8]
            rows.append(
                {
                    "galaxy_id": galaxy_id,
                    "log10_stellar_mass": float(log_sm),
                    "mean_metallicity_feh": float(mean_feh),
                    "err_mean": float(err_mean),
                    "metallicity_dispersion_feh": float(sigma_feh),
                    "err_sigma": float(err_sigma),
                    "blue_peak": float(blue_peak),
                    "red_peak": np.nan if float(red_peak) > 1000.0 else float(red_peak),
                }
            )
    systems = pd.DataFrame(rows)
    systems["dataset"] = np.select(
        [
            systems["galaxy_id"].str.startswith("VCS"),
            systems["galaxy_id"].str.startswith("HST_BCG"),
            systems["galaxy_id"].isin(["MW", "M31"]),
        ],
        ["VCS", "HST_BCG", "LG"],
        default="other",
    )
    systems["vcc_id"] = np.nan
    vcs_mask = systems["dataset"] == "VCS"
    systems.loc[vcs_mask, "vcc_id"] = (
        systems.loc[vcs_mask, "galaxy_id"].str.replace("VCS", "", regex=False).str.replace(".0", "", regex=False).astype(int)
    )
    systems["stellar_mass_msun"] = np.power(10.0, systems["log10_stellar_mass"].to_numpy(dtype=float))
    systems["halo_mass_plot_msun"] = present_day_halo_mass_from_observed_stellar_mass(systems["stellar_mass_msun"].to_numpy(dtype=float))
    systems["log10_halo_mass_plot"] = np.log10(np.clip(systems["halo_mass_plot_msun"].to_numpy(dtype=float), 1.0e-30, None))
    return _with_choksi_system_aliases(systems)


def _with_choksi_system_aliases(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    aliases = {
        "galaxyID": "galaxy_id",
        "logSM": "log10_stellar_mass",
        "mean_feh": "mean_metallicity_feh",
        "sigma_feh": "metallicity_dispersion_feh",
        "M_star_msun": "stellar_mass_msun",
        "M_halo_plot_msun": "halo_mass_plot_msun",
        "logMh_plot": "log10_halo_mass_plot",
        "VCC": "vcc_id",
    }
    for alias, source in aliases.items():
        out[alias] = out[source]
    return out


def _read_vizier_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#")


def load_acsvcs_hosts(path: Path) -> pd.DataFrame:
    hosts = _read_vizier_tsv(path)
    for col in ["VCC", "BTmag", "E(B-V)", "Vsys"]:
        if col in hosts.columns:
            hosts[col] = pd.to_numeric(hosts[col], errors="coerce")
    hosts = hosts.dropna(subset=["VCC"]).copy()
    hosts["VCC"] = hosts["VCC"].astype(int)
    hosts["vcc_id"] = hosts["VCC"]
    return hosts


def _vcs_color_to_feh(g_minus_z: np.ndarray) -> np.ndarray:
    colour = np.asarray(g_minus_z, dtype=float)
    disc = 0.481 * 0.481 - 4.0 * 0.051 * (1.513 - colour)
    disc = np.clip(disc, 0.0, None)
    return (-0.481 + np.sqrt(disc)) / (2.0 * 0.051)


def _zmag_to_mass_proxy(zmag: np.ndarray) -> np.ndarray:
    abs_mag = np.asarray(zmag, dtype=float) - VIRGO_DISTANCE_MODULUS
    lum = np.power(10.0, -0.4 * (abs_mag - ACS_SOLAR_MAG_Z))
    return GC_ML_Z * lum


def load_acsvcs_gc_catalog(path: Path) -> pd.DataFrame:
    gc = _read_vizier_tsv(path)
    numeric_cols = ["VCC", "RAJ2000", "DEJ2000", "GDist", "zmag", "zamag", "gmag", "gamag", "rhz", "rhg", "pGC", "E(B-V)"]
    for col in numeric_cols:
        if col in gc.columns:
            gc[col] = pd.to_numeric(gc[col], errors="coerce")
    gc = gc.dropna(subset=["VCC"]).copy()
    gc["VCC"] = gc["VCC"].astype(int)
    gc["vcc_id"] = gc["VCC"]
    g_use = np.where(np.isfinite(gc["gamag"]), gc["gamag"], gc["gmag"])
    z_use = np.where(np.isfinite(gc["zamag"]), gc["zamag"], gc["zmag"])
    gc["g_minus_z"] = g_use - z_use
    gc["metallicity_feh"] = _vcs_color_to_feh(gc["g_minus_z"].to_numpy(dtype=float))
    gc["metallicity_feh"] = np.where((gc["metallicity_feh"] >= FEH_MIN) & (gc["metallicity_feh"] <= FEH_MAX), gc["metallicity_feh"], np.nan)
    gc["gc_mass_proxy_msun"] = _zmag_to_mass_proxy(z_use)
    gc["feh"] = gc["metallicity_feh"]
    gc["m_gc_proxy_msun"] = gc["gc_mass_proxy_msun"]
    return gc


def load_choksi_observations(obs_cache_dir: Path = CHOKSI_CACHE_DIR) -> ChoksiObs:
    paths = require_choksi_cache(obs_cache_dir)
    systems = load_choksi_system_table(paths["choksi_data"])
    hosts = load_acsvcs_hosts(paths["acsvcs_hosts"])
    gc = load_acsvcs_gc_catalog(paths["acsvcs_gc_catalog"])
    vcs_systems = systems.loc[systems["dataset"] == "VCS"].copy()
    vcs_systems["VCC"] = vcs_systems["VCC"].astype(int)
    return ChoksiObs(
        systems=systems,
        vcs_systems=vcs_systems,
        acsvcs_hosts=hosts,
        acsvcs_gc=gc,
        mw_age_metallicity=pd.read_csv(paths["vandenberg2013_table2_csv"]),
        lmc_age_metallicity=pd.read_csv(paths["wagner2017_lmc_csv"]),
        lamers_summary=pd.read_csv(paths["lamers2017_summary_csv"]),
        obs_cache_dir=Path(obs_cache_dir),
    )


def load_neumayer_observations(cache_dir: Path = NEUMAYER_CACHE_DIR) -> NeumayerObs:
    cache_dir = Path(cache_dir)
    compiled_csv = cache_dir / "neumayer2020_fig12_compilation.csv"
    compiled_meta = cache_dir / "neumayer2020_fig12_compilation_meta.json"
    _require_paths([compiled_csv, compiled_meta], "Neumayer+2020 Fig.12")
    table = pd.read_csv(compiled_csv)
    table = table.rename(
        columns={
            "logMstar_gal": "log10_galaxy_stellar_mass",
            "logM_nsc": "log10_nsc_mass",
            "log_fraction": "log10_nsc_to_galaxy_mass_fraction",
        }
    )
    table["logMstar_gal"] = table["log10_galaxy_stellar_mass"]
    table["logM_nsc"] = table["log10_nsc_mass"]
    table["log_fraction"] = table["log10_nsc_to_galaxy_mass_fraction"]
    metadata = json.loads(compiled_meta.read_text(encoding="utf-8"))
    return NeumayerObs(table=table, cache_dir=cache_dir, metadata=metadata)


def load_neumayer_fig03_observations(cache_dir: Path = NEUMAYER_CACHE_DIR) -> NeumayerFig03Obs:
    cache_dir = Path(cache_dir)
    compiled_csv = cache_dir / "neumayer2020_fig03_demographics.csv"
    compiled_meta = cache_dir / "neumayer2020_fig03_demographics_meta.json"
    _require_paths([compiled_csv, compiled_meta], "Neumayer+2020 Fig.03")
    table = pd.read_csv(compiled_csv)
    required = ["name", "source", "logMstar_gal", "g_minus_i", "has_nsc", "colour_class_fig3", "host_type_fig3"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{compiled_csv} is missing required columns: {missing}")
    if table["has_nsc"].dtype != bool:
        table["has_nsc"] = table["has_nsc"].map(lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"})
    table["log10_galaxy_stellar_mass"] = table["logMstar_gal"]
    table["host_type"] = table["host_type_fig3"]
    metadata = json.loads(compiled_meta.read_text(encoding="utf-8"))
    return NeumayerFig03Obs(table=table, cache_dir=cache_dir, metadata=metadata)


def load_neumayer_fig13_observations(cache_dir: Path = NEUMAYER_CACHE_DIR) -> NeumayerFig13Obs:
    source_path = Path(cache_dir) / "original_nsc_review" / "bh_nsc_galmass.csv"
    _require_paths([source_path], "Neumayer+2020 Fig.13")
    table = pd.read_csv(source_path, skipinitialspace=True)
    required = ["object", "logbhmass", "bhulimit", "lognscmass", "nsculimit", "logmstar", "galtype", "sources"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")
    table = table.rename(
        columns={
            "object": "name",
            "logbhmass": "log10_bh_mass",
            "lognscmass": "log10_nsc_mass",
            "logmstar": "log10_galaxy_stellar_mass",
            "bhulimit": "bh_is_upper_limit",
            "nsculimit": "nsc_is_upper_limit",
            "galtype": "galtype_code",
        }
    )
    for col in ["log10_bh_mass", "bh_is_upper_limit", "log10_nsc_mass", "nsc_is_upper_limit", "log10_galaxy_stellar_mass", "galtype_code"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table["name"] = table["name"].astype(str).str.strip()
    table["sources"] = table["sources"].astype(str)
    duplicate_names = sorted(table.loc[table["name"].duplicated(keep=False), "name"].dropna().astype(str).unique().tolist())
    missing_host_mass_count = int((~np.isfinite(table["log10_galaxy_stellar_mass"].to_numpy(dtype=float))).sum())
    host_type_map = {1: "late", 2: "early", 0: "ucd"}
    table["host_type"] = table["galtype_code"].map(host_type_map)
    table["bh_is_upper_limit"] = table["bh_is_upper_limit"] == 1
    table["nsc_is_upper_limit"] = table["nsc_is_upper_limit"] == 1
    table["log10_bh_to_nsc_mass_ratio"] = table["log10_bh_mass"] - table["log10_nsc_mass"]
    table["logM_bh"] = table["log10_bh_mass"]
    table["logM_nsc"] = table["log10_nsc_mass"]
    table["logMstar_gal"] = table["log10_galaxy_stellar_mass"]
    table["log_bh_to_nsc"] = table["log10_bh_to_nsc_mass_ratio"]
    nonfinite_mass_mask = (
        ~np.isfinite(table["log10_bh_mass"].to_numpy(dtype=float))
        | ~np.isfinite(table["log10_nsc_mass"].to_numpy(dtype=float))
        | ~np.isfinite(table["log10_bh_to_nsc_mass_ratio"].to_numpy(dtype=float))
    )
    unknown_galtype_mask = table["host_type"].isna() & ~nonfinite_mass_mask
    ucd_upper_limit_mask = (
        (table["host_type"] == "ucd")
        & (table["bh_is_upper_limit"] | table["nsc_is_upper_limit"])
        & ~nonfinite_mass_mask
        & ~unknown_galtype_mask
    )
    table["plot_keep"] = ~(nonfinite_mass_mask | unknown_galtype_mask | ucd_upper_limit_mask)
    return NeumayerFig13Obs(
        table=table,
        source_path=source_path,
        duplicate_names=duplicate_names,
        missing_host_mass_count=missing_host_mass_count,
        nonfinite_mass_count=int(nonfinite_mass_mask.sum()),
        unknown_galtype_count=int(unknown_galtype_mask.sum()),
        ucd_upper_limit_count=int(ucd_upper_limit_mask.sum()),
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_cliff_fig14_observations() -> KongObs:
    _require_paths([CLIFF_OBS_PATH], "The Cliff Fig.14")
    table = pd.read_csv(CLIFF_OBS_PATH)
    required = [
        "name",
        "sample",
        "reference",
        "z",
        "logMstar",
        "logMstar_err_lo",
        "logMstar_err_hi",
        "logMstar_upper_limit",
        "logMBH",
        "logMBH_err_lo",
        "logMBH_err_hi",
        "logMBH_upper_limit",
        "plot_group",
        "marker",
        "color",
        "source_note",
    ]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{CLIFF_OBS_PATH} is missing required columns: {missing}")
    numeric_columns = ["z", "logMstar", "logMstar_err_lo", "logMstar_err_hi", "logMBH", "logMBH_err_lo", "logMBH_err_hi"]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    for column in ["logMstar_upper_limit", "logMBH_upper_limit"]:
        table[column] = table[column].map(_as_bool)
    for column in ["name", "sample", "reference", "plot_group", "marker", "color", "source_note"]:
        table[column] = table[column].fillna("").astype(str)
    table["log10_stellar_mass"] = table["logMstar"]
    table["log10_bh_mass"] = table["logMBH"]
    table["stellar_mass_is_upper_limit"] = table["logMstar_upper_limit"]
    table["bh_mass_is_upper_limit"] = table["logMBH_upper_limit"]
    valid = np.isfinite(table["log10_stellar_mass"].to_numpy(dtype=float)) & np.isfinite(table["log10_bh_mass"].to_numpy(dtype=float))
    return KongObs(table=table.loc[valid].copy(), source_path=CLIFF_OBS_PATH)
