"""Effective-radius models for NSC GC birth and background profiles.

All public radius helpers return physical kpc. Halo masses are in Msun for
formation-stage calls and in 1e9 Msun for evolution-stage calls. Catalogue
stellar half-mass radii are expected to have been converted from ckpc/h to
physical kpc by the sidecar builder; SFR values are in Msun/yr.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import special

import NSC.old.src.smhm as smhm

EFF_RAD_GAO2023 = "Gao+2024"
EFF_RAD_EMPIRICAL = "empirical"
EFF_RAD_CATALOGUE = "catalogue"
EFF_RAD_CHOICES = (EFF_RAD_GAO2023, EFF_RAD_EMPIRICAL, EFF_RAD_CATALOGUE)
EFF_RADIUS_CATALOGUE_BUILD_COMMAND = (
    "python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/"
    "6_build_eff_radius_catalogue.py"
)

H100 = 0.704
KPC_IN_M = 3.0856775814913673e19
PI = math.pi
OMEGA_M0 = 0.307
H_0 = 67.97
OMEGA_LAMBDA0 = 1.0 - OMEGA_M0
MPC_IN_KM = 3.0856775814913673e19
GYR_IN_S = 365.25 * 24.0 * 3600.0 * 1.0e9
LCDM_H0_GYR_INV = (H_0 / MPC_IN_KM) * GYR_IN_S
LCDM_TIME_PREFAC_GYR = 2.0 / (3.0 * LCDM_H0_GYR_INV * math.sqrt(OMEGA_LAMBDA0))
LCDM_ASINH_RATIO = math.sqrt(OMEGA_LAMBDA0 / OMEGA_M0)

EMPIRICAL_ZBINS = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75], dtype=float)
EMPIRICAL_LOG10_A = np.array([0.86, 0.78, 0.70, 0.65, 0.55, 0.51], dtype=float)
EMPIRICAL_ALPHA = np.array([0.25, 0.22, 0.22, 0.23, 0.22, 0.18], dtype=float)
EMPIRICAL_SIGMA_LOGR = np.array([0.16, 0.16, 0.17, 0.18, 0.19, 0.19], dtype=float)
FGC_DEFAULT = 1.0


@dataclass(frozen=True)
class EffectiveRadiusResult:
    re_kpc: float
    source: str
    fallback_reason: str = ""


def validate_eff_rad_mode(mode: str) -> str:
    """Validate the effective-radius mode used by CLI parsers and callers."""

    if mode not in EFF_RAD_CHOICES:
        allowed = ", ".join(EFF_RAD_CHOICES)
        raise ValueError(f"eff_rad must be one of: {allowed}")
    return mode


def nearest_snap_from_redshift(redshift: float, snap_redshifts: np.ndarray) -> int:
    """Return the nearest snapshot number for one redshift table."""

    z_table = np.asarray(snap_redshifts, dtype=float).reshape(-1)
    if z_table.size == 0:
        raise ValueError("snapshot-redshift table is empty")
    z = float(redshift)
    if not math.isfinite(z):
        raise ValueError(f"redshift must be finite, got {redshift!r}")
    return int(np.argmin(np.abs(z_table - z)))


def sersic_shape_coeffs(n: float) -> tuple[float, float, float]:
    """Return ``p``, ``b``, and ``a_n`` for the deprojected Sersic profile."""

    ns = float(n)
    if not math.isfinite(ns) or ns <= 0.0:
        raise ValueError(f"Sersic index must be positive, got {n!r}")
    p = 1.0 - 0.6097 / ns + 0.05563 / (ns * ns)
    b = 2.0 * ns - 1.0 / 3.0 + 0.009876 / ns
    a_n = ns * (3.0 - p)
    return p, b, a_n


def sersic_re_from_aperture_fractions(
    rstar_half_phys_kpc: float,
    f1: float,
    f2: float,
    sersic_n: float,
) -> tuple[float, float, float]:
    """Infer SFR Sersic ``R_e`` from fractions inside 1 and 2 stellar half-radii.

    Parameters
    ----------
    rstar_half_phys_kpc:
        Stellar half-mass radius in physical kpc.
    f1, f2:
        SFR fractions inside ``Rstar_half`` and ``2*Rstar_half``.
    sersic_n:
        Deprojected Sersic index used for the GC birth profile.
    """

    radius = float(rstar_half_phys_kpc)
    frac1 = float(f1)
    frac2 = float(f2)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("stellar half-mass radius must be positive")
    if (
        (not math.isfinite(frac1))
        or (not math.isfinite(frac2))
        or frac1 <= 0.0
        or frac2 <= 0.0
        or frac1 > frac2
        or frac2 > 1.0
    ):
        raise ValueError("SFR aperture fractions must satisfy 0 < f1 <= f2 <= 1")

    _, b, a_n = sersic_shape_coeffs(sersic_n)
    inv1 = float(special.gammaincinv(a_n, frac1))
    inv2 = float(special.gammaincinv(a_n, frac2))
    if (not math.isfinite(inv1)) or (not math.isfinite(inv2)) or inv1 <= 0.0 or inv2 <= 0.0:
        raise ValueError("invalid Sersic aperture inversion")

    ns = float(sersic_n)
    re1 = radius * (b / inv1) ** ns
    re2 = 2.0 * radius * (b / inv2) ** ns
    if (not math.isfinite(re1)) or (not math.isfinite(re2)) or re1 <= 0.0 or re2 <= 0.0:
        raise ValueError("invalid catalogue SFR effective radius")
    re_sfr = 10.0 ** (0.5 * (math.log10(re1) + math.log10(re2)))
    return re1, re2, re_sfr


def gao2023_birth_re_kpc(jsp: float, halomass_msun: float, redshift: float) -> float:
    """Gao+2024 birth-radius scale in kpc.

    ``jsp`` is the fixed-tree spin norm in ``(kpc/h) km/s`` and ``halomass_msun``
    is the host halo mass in Msun.
    """

    rvir_pc = smhm.virialRadius(float(halomass_msun), float(redshift))
    hz = smhm.H0 * smhm.E(float(redshift))
    return float(jsp) * H100 / 20.0 / hz / rvir_pc


def _omega_m(redshift: float) -> float:
    z = max(float(redshift), 0.0)
    zp1_cubed = (1.0 + z) ** 3
    return OMEGA_M0 * zp1_cubed / (1.0 - OMEGA_M0 + OMEGA_M0 * zp1_cubed)

def CosmicTimeGyr2Redshift(t_Gyr: float) -> float:
    """flat LambdaCDM cosmic age to redshift conversion without radiation"""

    sinh_val = math.sinh(t_Gyr / LCDM_TIME_PREFAC_GYR)
    if sinh_val <= 0.0:
        return 0.0
    return max((LCDM_ASINH_RATIO / sinh_val) ** (2.0 / 3.0) - 1.0, 0.0)

def _evolution_rvir_kpc(mhalo_1e9msun: float, t_l_gyr: float, tun) -> float:
    z = CosmicTimeGyr2Redshift(t_l_gyr)
    omega_m_z = _omega_m(z)
    return 163.0 / tun.h * (float(mhalo_1e9msun) / 1.0e3 * tun.h) ** (1.0 / 3.0) / (
        (OMEGA_M0 * (18.0 * PI * PI + 82.0 * (omega_m_z - 1.0) - 39.0 * (omega_m_z - 1.0) ** 2) / omega_m_z / 200.0) ** (1.0 / 3.0) * (1.0 + z)
    )


def gao2023_evolution_re_kpc(spin_norm: float, mhalo_1e9msun: float, t_l_gyr: float, tun) -> float:
    """Gao+2024 analytical-background radius scale in physical kpc.

    ``spin_norm`` is the fixed-tree spin norm after the legacy evolution-unit
    conversion used by ``src/evo.py``. ``mhalo_1e9msun`` is in 1e9 Msun.
    """

    rvir = _evolution_rvir_kpc(float(mhalo_1e9msun), float(t_l_gyr), tun)
    spinprm = float(spin_norm) / math.sqrt(
        2.0 * 6.67 * rvir * KPC_IN_M * float(mhalo_1e9msun) * 2.0e28
    )
    return spinprm * rvir / math.sqrt(2.0)


def _f_x_smhm(x: float, z: float) -> float:
    a = 1.0 / (1.0 + z)
    nu = math.exp(-4.0 * a * a)
    alpha = -1.412 + 0.731 * (a - 1.0) * nu
    delta = 3.508 + (2.608 * (a - 1.0) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu
    return -math.log10(10.0 ** (alpha * x) + 1.0) + delta * (math.log10(1.0 + math.exp(x))) ** gamma / (1.0 + math.exp(0.1**x))


def evolution_mstar_msun_smhm(mhalo_1e9msun: float, t_l_gyr: float) -> float:
    """SMHM stellar mass in Msun, matching the evolution solver helper."""

    z = CosmicTimeGyr2Redshift(t_l_gyr)
    a = 1.0 / (1.0 + z)
    nu = math.exp(-4.0 * a * a)
    epsilon = 10.0 ** (-1.777 - 0.006 * (a - 1.0) * nu - 0.119 * (a - 1.0))
    m1 = 10.0 ** (11.514 - (1.793 * (a - 1.0) + 0.251 * z) * nu)
    mstar_1e9 = epsilon * m1 * 10.0 ** (_f_x_smhm(math.log10(float(mhalo_1e9msun) * 1.0e9 / m1), z) - _f_x_smhm(0.0, z)) / 1.0e9
    return mstar_1e9 * 1.0e9


def empirical_re_kpc(mstar_msun: float, redshift: float, *, scatter: bool, rng=None) -> float:
    """Empirical star-forming galaxy size relation converted to GC ``R_e``.

    ``mstar_msun`` is in Msun. The returned radius is physical kpc. The fixed
    GC compactness factor is ``f_GC = 1``; the PDF suggests ``0.3, 1, 3`` for
    later systematic tests, but this implementation intentionally exposes no
    separate compactness CLI option.
    """

    mstar = float(mstar_msun)
    z = float(redshift)
    if not math.isfinite(mstar) or mstar <= 0.0:
        raise ValueError(f"mstar_msun must be positive, got {mstar_msun!r}")
    if not math.isfinite(z) or z < 0.0:
        raise ValueError(f"redshift must be finite and non-negative, got {redshift!r}")

    if z > 3.0:
        re_sf = 0.83 * (mstar / (10.0 ** 8.5)) ** 0.25 * ((1.0 + z) / 5.0) ** (-1.24)
        sigma_logr = 0.0
        log_re = math.log10(re_sf)
    else:
        x_table = np.log1p(EMPIRICAL_ZBINS)
        x = float(np.clip(math.log1p(z), x_table[0], x_table[-1]))
        log10_a = float(np.interp(x, x_table, EMPIRICAL_LOG10_A))
        alpha = float(np.interp(x, x_table, EMPIRICAL_ALPHA))
        sigma_logr = float(np.interp(x, x_table, EMPIRICAL_SIGMA_LOGR))
        log_re = log10_a + alpha * math.log10(mstar / 5.0e10)

    if scatter and sigma_logr > 0.0:
        normal = np.random.normal if rng is None else rng.normal
        log_re += float(normal(0.0, sigma_logr))
    re_gc = FGC_DEFAULT * (10.0 ** log_re)
    if not math.isfinite(re_gc) or re_gc <= 0.0:
        raise ValueError("empirical effective radius is outside the physical domain")
    return re_gc


def load_eff_radius_catalogue(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    """Read the sidecar effective-radius catalogue keyed by tree, dark subhalo, snapshot."""

    catalogue_path = Path(path)
    lookup: dict[tuple[str, int, int], dict[str, str]] = {}
    with catalogue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                key = (
                    row["fixed_tree_basename"].strip(),
                    int(float(row["dark_subhalo_id"])),
                    int(float(row["snapnum"])),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Malformed effective-radius catalogue row in {catalogue_path}: {row}") from exc
            if key in lookup:
                raise RuntimeError(f"Duplicate effective-radius catalogue key {key} in {catalogue_path}")
            lookup[key] = dict(row)
    return lookup


def _record_float(record: Mapping[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = record[key]
    except KeyError:
        return default
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _record_int(record: Mapping[str, str], key: str, default: int = 0) -> int:
    value = _record_float(record, key, float(default))
    if not math.isfinite(value):
        return default
    return int(value)


def catalogue_re_kpc(record: Mapping[str, str], sersic_n: float) -> EffectiveRadiusResult:
    """Infer a catalogue SFR-concentration radius or return a visible fallback reason."""

    matched = _record_int(record, "matched", 1)
    if matched != 1:
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "missing_hydro_match")

    sfr = _record_float(record, "SubhaloSFR")
    sfr_half = _record_float(record, "SubhaloSFRinHalfRad")
    sfr_rad = _record_float(record, "SubhaloSFRinRad")
    rhalf = _record_float(record, "stellar_halfmass_radius_phys_kpc")
    stellar_particles = _record_int(record, "stellar_particle_count")
    if not math.isfinite(sfr) or sfr <= 0.0:
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "zero_sfr")
    if not math.isfinite(rhalf) or rhalf <= 0.0:
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "invalid_stellar_halfmass_radius")
    if stellar_particles < 100:
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "unresolved_stellar_component")

    f1 = _record_float(record, "f1", sfr_half / sfr if math.isfinite(sfr_half) else math.nan)
    f2 = _record_float(record, "f2", sfr_rad / sfr if math.isfinite(sfr_rad) else math.nan)
    if (
        (not math.isfinite(f1))
        or (not math.isfinite(f2))
        or f1 <= 0.0
        or f2 <= 0.0
        or f1 > f2
        or f2 > 1.0
    ):
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "invalid_sfr_fractions")

    try:
        re1, re2, re_sfr = sersic_re_from_aperture_fractions(rhalf, f1, f2, sersic_n)
    except ValueError:
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "invalid_sersic_inversion")

    if abs(math.log10(re1 / re2)) > 0.3:
        return EffectiveRadiusResult(math.nan, EFF_RAD_EMPIRICAL, "inconsistent_aperture_estimates")
    return EffectiveRadiusResult(FGC_DEFAULT * re_sfr, EFF_RAD_CATALOGUE, "")


def _empirical_result(mstar_msun: float, redshift: float, *, scatter: bool, rng=None, reason: str = "") -> EffectiveRadiusResult:
    return EffectiveRadiusResult(
        empirical_re_kpc(mstar_msun, redshift, scatter=scatter, rng=rng),
        EFF_RAD_EMPIRICAL,
        reason,
    )


def resolve_birth_re_kpc(
    *,
    mode: str,
    halomass_msun: float,
    redshift: float,
    mstar_msun: float,
    jsp: float,
    sersic_n: float,
    fixed_tree_basename: str,
    dark_subhalo_id: int,
    snapnum: int,
    catalogue: Mapping[tuple[str, int, int], Mapping[str, str]] | None = None,
    rng=None,
) -> EffectiveRadiusResult:
    """Resolve the GC birth-radius scale in kpc for one formation event."""

    eff_mode = validate_eff_rad_mode(mode)
    if eff_mode == EFF_RAD_GAO2023:
        return EffectiveRadiusResult(gao2023_birth_re_kpc(jsp, halomass_msun, redshift), EFF_RAD_GAO2023, "")
    if eff_mode == EFF_RAD_EMPIRICAL:
        return _empirical_result(mstar_msun, redshift, scatter=True, rng=rng)

    key = (str(fixed_tree_basename), int(dark_subhalo_id), int(snapnum))
    record = None if catalogue is None else catalogue.get(key)
    if record is None:
        return _empirical_result(mstar_msun, redshift, scatter=True, rng=rng, reason="missing_catalogue_row")
    result = catalogue_re_kpc(record, sersic_n)
    if result.source == EFF_RAD_CATALOGUE:
        return result
    return _empirical_result(mstar_msun, redshift, scatter=True, rng=rng, reason=result.fallback_reason)


def resolve_background_re_kpc(
    *,
    mode: str,
    mhalo_1e9msun: float,
    redshift: float,
    t_l_gyr: float,
    spin_norm: float,
    sersic_n: float,
    tun,
    fixed_tree_basename: str,
    dark_subhalo_id: int,
    snapnum: int,
    catalogue: Mapping[tuple[str, int, int], Mapping[str, str]] | None = None,
) -> EffectiveRadiusResult:
    """Resolve the analytical stellar-background radius scale in kpc."""

    eff_mode = validate_eff_rad_mode(mode)
    if eff_mode == EFF_RAD_GAO2023:
        return EffectiveRadiusResult(
            gao2023_evolution_re_kpc(spin_norm, mhalo_1e9msun, t_l_gyr, tun),
            EFF_RAD_GAO2023,
            "",
        )

    mstar_msun = evolution_mstar_msun_smhm(mhalo_1e9msun, t_l_gyr)
    if eff_mode == EFF_RAD_EMPIRICAL:
        return _empirical_result(mstar_msun, redshift, scatter=False)

    key = (str(fixed_tree_basename), int(dark_subhalo_id), int(snapnum))
    record = None if catalogue is None else catalogue.get(key)
    if record is None:
        return _empirical_result(mstar_msun, redshift, scatter=False, reason="missing_catalogue_row")
    result = catalogue_re_kpc(record, sersic_n)
    if result.source == EFF_RAD_CATALOGUE:
        return result
    return _empirical_result(mstar_msun, redshift, scatter=False, reason=result.fallback_reason)
