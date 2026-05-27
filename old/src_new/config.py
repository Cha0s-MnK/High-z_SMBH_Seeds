# ===================== #
# CONFIGURE ENVIRONMENT #
# ===================== #

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import scipy
from scipy import interpolate, special

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
AU        = 1.495978707e11        # astronomical unit [m] (reference: https://en.wikipedia.org/wiki/Astronomical_unit)
c         = 2.99792458e8          # speed of light [m·s⁻¹]
day       = 24 * 3600             # day [s]
e         = 1.602176634e-19       # elementary charge [C]
epsilon_0 = 8.854187817e-12       # vacuum permittivity [F·m⁻¹]
G         = 6.6743015e-11         # gravitational constant [m³·kg⁻¹·s⁻²]
G_Arepo   = 4.300931494278067e-3  # gravitational constant [pc·(km/s)²·M☉⁻¹]
G_kpc     = 4.300931494278067e-6  # gravitational constant [kpc·(km/s)²·M☉⁻¹]
G_astro   = 0.004498517029175462  # gravitational constant [pc³·M☉⁻¹·Myr⁻²]
h         = 6.62607015e-34        # Planck constant [J·s]
k_B       = 1.380649e-23          # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31    # electron mass [kg] (reference: https://en.wikipedia.org/wiki/Electron_mass)
m_e_c2    = 0.5109989506916       # electron mass [MeV] (reference: https://en.wikipedia.org/wiki/Electron_mass)
m_p       = 1.6726219259552e-27   # proton mass [kg]
m_u       = 1.6605390689252e-27   # unified atomic mass unit [kg] (reference: https://en.wikipedia.org/wiki/Dalton_(unit))
M_sun     = 1.988416e30           # solar mass [kg] (reference: https://en.wikipedia.org/wiki/Solar_mass)
N_A       = 6.02214076e23         # Avogadro constant [mol⁻¹] (reference: https://en.wikipedia.org/wiki/Avogadro_constant)
pc        = 3.0856775814913673e16 # parsec [m] (reference: https://en.wikipedia.org/wiki/Parsec)
PI        = np.pi                 # π
yr        = 365.25 * 24 * 3600    # Julian year [s]

kpc       = 1.0e3 * pc            # kiloparsec [m]
Mpc       = 1.0e6 * pc            # megaparsec [m]
Myr       = 1.0e6 * yr            # megayear [s]
Gyr       = 1.0e9 * yr            # gigayear [s]

# DESI 2024 + CMB
Omega_m0      = 0.307 # present-day matter density parameter
Omega_Lambda0 = 1 - Omega_m0 # present-day dark-energy density parameter
H0            = 67.97 # Hubble constant [(km/s)/Mpc]
ReducedH0     = H0 / 100.0 # reduced Hubble constant h; H_0 = 100 h (km/s)/Mpc
t_Lambda_Gyr  = 2.0 / (3.0 * H0 * math.sqrt(Omega_Lambda0)) * Mpc / 1.0e3 / Gyr
t_universe    = 13.780 # age of the universe [Gyr]
SqrtOmega_Lambda0OverOmega_m0 = math.sqrt(Omega_Lambda0 / Omega_m0)

H100 = 0.704

NUM_PROC = 16
OUT_DIR  = "/lingshan/disk3/subonan/_output"
STD_DPI  = 512

# ================== #
# HELPER FUNCTION(S) #
# ================== #

# checking utilities for input parameters

def check_finite(val, name="val"):
    val = float(val)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite, but got {val}!")
    return val
    
def check_finite_non_negative(val, name="val"):
    val = float(val)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, but got {val}!")
    return val

def check_finite_positive(val, name="val"):
    val = float(val)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and positive, but got {val}!")
    return val

# linear interpolation on a uniformly spaced grid

def lininterp_uniform(xq, x_grid, y_grid, dx_inv=None, *, allow_extrapolate=False):
    """
    Linear interpolation on a uniformly spaced grid.

    Parameters
    ----------
    xq : float
        Query coordinate.
    x_grid : array-like
        Uniformly spaced grid coordinates.
    y_grid : array-like
        Function values tabulated on x_grid.
    dx_inv : float, optional
        Inverse grid spacing, 1 / dx. If None, it is computed from x_grid.
    allow_extrapolate : bool
        If False, raise an error outside the grid.
        If True, linearly extrapolate using the edge interval.

    Returns
    -------
    yq : float
        Interpolated value at xq.
    """

    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be 1D arrays.")

    if len(x_grid) != len(y_grid):
        raise ValueError("x_grid and y_grid must have the same length.")

    if len(x_grid) < 2:
        raise ValueError("Need at least two grid points for interpolation.")

    if dx_inv is None:
        dx = x_grid[1] - x_grid[0]
        if dx == 0.0:
            raise ValueError("x_grid spacing cannot be zero.")
        dx_inv = 1.0 / dx

    u = (xq - x_grid[0]) * dx_inv

    if not allow_extrapolate:
        if u < 0.0 or u > len(x_grid) - 1:
            raise ValueError(f"xq={xq} is outside the interpolation grid.")

    # Exact upper boundary: return the final value directly.
    if u == len(x_grid) - 1:
        return float(y_grid[-1])

    i = math.floor(u)

    if allow_extrapolate:
        i = max(0, min(i, len(x_grid) - 2))
    else:
        i = max(0, min(i, len(x_grid) - 2))

    f = u - i

    return float(y_grid[i] + f * (y_grid[i + 1] - y_grid[i]))

# cosmology utilities

def Ez(z: float) -> float:
    """
    dimensionless Hubble parameter E(z) = H(z) / H0 for flat ΛCDM without radiation
    """
    check_finite_non_negative(z, name="Redshift z")

    return np.sqrt(Omega_m0 * (1.0 + z)**3 + Omega_Lambda0)

def H(z: float) -> float:
    """
    Hubble parameter H(z) in (km/s)/Mpc for flat ΛCDM without radiation
    """
    return H0 * Ez(z)

def Omega_m(z: float) -> float:
    """matter density parameter Ω_m(z) for flat ΛCDM without radiation"""
    check_finite_non_negative(z, name="Redshift z")

    zPlus1Cubed = (1.0 + z) ** 3
    return Omega_m0 * zPlus1Cubed / (1.0 - Omega_m0 + Omega_m0 * zPlus1Cubed)

def Redshift2CosmicAge(z: float, time_unit: str = "Gyr") -> float:
    """Flat LCDM cosmic age in Gyr.

    For a spatially flat matter+Lambda cosmology, the age at redshift z has the analytic form

        t(z) = 2 / (3 H0 sqrt(Omega_L)) * asinh(sqrt(Omega_L / Omega_M) / (1 + z)^(3/2))

    which is exact under the flat-LCDM assumption.
    """

    check_finite_non_negative(z, name="Redshift z")
    if time_unit == "Gyr":
        t_Lambda = t_Lambda_Gyr
    elif time_unit == "Myr":
        t_Lambda = t_Lambda_Gyr * 1.0e3
    elif time_unit == "yr":
        t_Lambda = t_Lambda_Gyr * 1.0e9
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")

    return t_Lambda * math.asinh(SqrtOmega_Lambda0OverOmega_m0 / ((1.0 + z) ** 1.5))

def Rv(Mh: float, z: float) -> float:
    """
    Virial radius in kpc for halo mass Mh in Msun for flat ΛCDM without radiation

    Uses the Bryan & Norman virial overdensity relative to the critical density.
    """
    check_finite_positive(Mh, name="Halo mass Mh")
    check_finite_non_negative(z, name="Redshift z")

    # critical density
    Hz_kpc = H(z=z) * 1.0e-3 # [(km/s)/Mpc] --> [(km/s)/kpc]
    rho_crit = 3.0 * (Hz_kpc ** 2) / (8.0 * PI * G_kpc) # [M☉/kpc³]

    # Bryan & Norman virial overdensity relative to critical density
    x = Omega_m(z=z) - 1.0
    Delta_v = 18.0 * (PI ** 2) + 82.0 * x - 39.0 * (x ** 2)

    # virial radius in kpc
    return np.cbrt(3.0 * Mh / (4.0 * PI * Delta_v * rho_crit))

def Rv_kpc(Mhalo_1e9Msun: float, t_Gyr: float, tun: Tunables) -> float:
    check_finite_positive(Mhalo_1e9Msun, name="Halo mass Mhalo_1e9Msun")
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")

    z = CosmicAge2Redshift(t_Gyr, time_unit="Gyr")
    Omega_m_z = Omega_m(z)
    Delta_v = (18.0 * PI * PI + 82.0 * (Omega_m_z - 1.0) - 39.0 * (Omega_m_z - 1.0) ** 2) / Omega_m_z
    check_finite_positive(Delta_v, name="Average halo over-density at Rv Delta_v")
    Rv_kpc = 163.0 / ((1.0 + z) * tun.h) * (Mhalo_1e9Msun * tun.h * 200.0 / (1.0e3 * Omega_m0 * Delta_v)) ** (1.0 / 3.0)
    check_finite_positive(Rv_kpc, name="Halo virial radius in kpc Rv_kpc")
    return Rv_kpc

def CosmicAge2Redshift(t: float, time_unit: str = "Gyr") -> float:
    """cosmic age to redshift conversion for flat ΛCDM without radiation"""
    check_finite_positive(t, name="Cosmic age t")
    if time_unit == "Gyr":
        t = t
    elif time_unit == "Myr":
        t = t * 1.0e3
    elif time_unit == "yr":
        t = t * 1.0e9
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")

    z = (SqrtOmega_Lambda0OverOmega_m0 / math.sinh(t / t_Lambda_Gyr)) ** (2.0 / 3.0) - 1.0
    check_finite_non_negative(z, name="Redshift z")
    return z

def Vv(Mh, z):
    return np.sqrt(G_kpc * Mh / Rv(Mh=Mh, z=z))

# Behroozi+2013 stellar mass-halo-mass(SMHM) relation

def f_x_SMHM(x: float, z: float) -> float:
    check_finite(x, name="lg(M_h/M_1) x")
    check_finite_non_negative(z, name="Redshift z")

    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    alpha = - 1.412 + 0.731 * (a - 1.0) * nu
    delta = 3.508 + (2.608 * (a - 1.0) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu
    low_mass_arg = 10.0 ** (-x)
    low_mass_weight = 0.0 if low_mass_arg > 700.0 else 1.0 / (1.0 + math.exp(low_mass_arg))
    return - math.log10(10.0 ** (alpha * x) + 1.0) + delta * (math.log10(1.0 + math.exp(x))) ** gamma * low_mass_weight

def Mstar_SMHM(Mhalo: float, z: float, scatter: bool = False) -> float:
    check_finite_positive(Mhalo, name="Halo mass Mhalo")
    check_finite_non_negative(z, name="Redshift z")

    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    epsilon = 10.0 ** (- 1.777 - 0.006 * (a - 1.0) * nu - 0.119 * (a - 1.0))
    M1 = 10.0 ** (11.514 - (1.793 * (a - 1.0) + 0.251 * z) * nu)
    lg_Mstar = math.log10(epsilon * M1) + f_x_SMHM(math.log10(Mhalo / M1), z) - f_x_SMHM(0.0, z)
    if scatter:
        xi = np.random.normal(0.0, 0.218 + 0.023 * z / (1.0 + z))
        lg_Mstar += xi
    Mstar = 10 ** lg_Mstar
    check_finite_positive(Mstar, name="Stellar mass in 1e9 Msun Mstar")
    return Mstar

def Mstar_1e9Msun_SMHM(Mhalo_1e9Msun: float, t_Gyr: float, scatter: bool = False) -> float:
    check_finite_positive(Mhalo_1e9Msun, name="Halo mass in 1e9 Msun Mhalo_1e9Msun")
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")

    z = CosmicAge2Redshift(t_Gyr, time_unit="Gyr")
    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    epsilon = 10.0 ** (- 1.777 - 0.006 * (a - 1.0) * nu - 0.119 * (a - 1.0))
    M1 = 10.0 ** (11.514 - (1.793 * (a - 1.0) + 0.251 * z) * nu)
    lg_Mstar = math.log10(epsilon * M1) + f_x_SMHM(math.log10(Mhalo_1e9Msun * 1.0e9 / M1), z) - f_x_SMHM(0.0, z)
    if scatter:
        xi = np.random.normal(0.0, 0.218 + 0.023 * z / (1.0 + z))
        lg_Mstar += xi
    Mstar_1e9Msun = 10 ** lg_Mstar / 1.0e9
    check_finite_positive(Mstar_1e9Msun, name="Stellar mass in 1e9 Msun Mstar_1e9Msun")
    return Mstar_1e9Msun

# Schechter star cluster initial mass function

def upperIncompleteGamma0(x):
    """
    upper incomplete gamma function Gamma(0, x) = exponential integral E_1(x) for x > 0
    """
    check_finite_positive(x, name="x for upper incomplete gamma function Gamma(0, x)")

    return scipy.special.exp1(x)

def upperIncompleteGammaMinus1(x):
    """
    upper incomplete gamma function Gamma(-1, x) = exp(-x) / x - Gamma(0, x) for x > 0
    """
    check_finite_positive(x, name="x for upper incomplete gamma function Gamma(-1, x)")

    return np.exp(-x) / x - scipy.special.exp1(x)

def makeLogMgcToLogMmaxInterpolator(Mc: float, Mmin: float = 1.0e5, dlog_mmax: float = 0.02):
    """
    Build an interpolator from log10(total GC mass) to log10(Mmax)
    for a Schechter cluster initial mass function with alpha = -2.

    The CIMF is

        dN/dM = A M^-2 exp(-M / Mc)

    The normalization A is set by requiring one expected cluster above Mmax:

        1 = int_{Mmax}^{inf} dN/dM dM

    Then the total GC mass formed in the event is

        M_GC = int_{Mmin}^{Mmax} M dN/dM dM

    For alpha = -2, this gives

        M_GC =
            Mc * [Gamma(0, Mmin/Mc) - Gamma(0, Mmax/Mc)]
               / Gamma(-1, Mmax/Mc)

    Parameters
    ----------
    mc : float
        Schechter cutoff mass Mc in Msun.

    mmin : float
        Minimum cluster mass in Msun.

    dlog_mmax : float
        Grid spacing in log10(Mmax).

    Returns
    -------
    scipy.interpolate.interp1d
        Interpolator with usage:

            log_mmax = interp(log_mgc)

        where log_mgc = log10(total GC mass formed in one event).
    """
    check_finite_positive(Mc, name="Schechter cutoff mass Mc")
    check_finite_positive(Mmin, name="Minimum cluster mass Mmin")
    check_finite_positive(dlog_mmax, name="log Mmax grid spacing dlog_mmax")
    if Mmin >= 1.0e6:
        raise ValueError(f"Minimum cluster mass Mmin must be less than 1e6 Msun, but got Mmin = {Mmin}!")

    log_mmin = np.log10(Mmin)

    # Mmax must be larger than Mmin, so start one grid step above Mmin.
    log_mmax_grid = np.arange(log_mmin + dlog_mmax, 8.6, dlog_mmax, dtype=float)

    mmax_grid = 10.0 ** log_mmax_grid

    # Dimensionless mass ratios x = M / Mc.
    x_min = Mmin / Mc
    x_max_grid = mmax_grid / Mc

    # Gamma(0, Mmin / Mc), scalar.
    gamma0_min = upperIncompleteGamma0(x_min)

    # Use the updated gamma functions on each grid point.
    gamma0_max_grid = np.array([
        upperIncompleteGamma0(x) for x in x_max_grid
    ])

    gamma_minus1_max_grid = np.array([
        upperIncompleteGammaMinus1(x) for x in x_max_grid
    ])

    # Total GC mass corresponding to each Mmax.
    mgc_grid = Mc * (gamma0_min - gamma0_max_grid) / gamma_minus1_max_grid

    if np.any(~np.isfinite(mgc_grid)) or np.any(mgc_grid <= 0.0):
        raise RuntimeError("Generated invalid M_GC values while building the interpolator.")

    log_mgc_grid = np.log10(mgc_grid)

    if not np.all(np.diff(log_mgc_grid) > 0.0):
        raise RuntimeError(
            "log10(M_GC) is not strictly increasing with log10(Mmax). "
            "Cannot build a safe inverse interpolator."
        )

    return interpolate.interp1d(
        log_mgc_grid,
        log_mmax_grid,
        bounds_error=True,
        assume_sorted=True,
    )

def upper_gamma2_log_mass(log_m: float, Mc: float) -> float:
    """Return Gamma(-1, M/Mc) for a base-10 log mass."""

    check_finite(log_m, name="log10 cluster mass")
    check_finite_positive(Mc, name="Schechter cutoff mass Mc")
    return float(upperIncompleteGammaMinus1((10.0 ** float(log_m)) / float(Mc)))
    
# Sersic profile utilities

def Sersic_coefs(N_S: float) -> Tuple[float, float]:
    check_finite_positive(N_S, name="Sersic index N_S")
    p = 1.0 - 0.6097 / N_S + 0.05563 / (N_S * N_S)
    b = 2.0 * N_S - 1.0 / 3.0 + 0.009876 / N_S
    return p, b

# Gao-only effective-radius helpers used by the formation and evolution stages.

def gao2023_birth_re_kpc(jsp: float, halomass_msun: float, redshift: float) -> float:
    """Gao+2024 birth-radius scale in physical kpc."""

    j_kpc_kms = float(jsp) * H100
    rvir_kpc = Rv(Mh=float(halomass_msun), z=float(redshift))
    hz_km_s_kpc = H(float(redshift)) * 1.0e-3
    re_kpc = j_kpc_kms / (20.0 * hz_km_s_kpc * rvir_kpc)
    return check_finite_positive(re_kpc, name="Gao+2024 birth effective radius in kpc")

def gao2023_evolution_re_kpc(spin_norm: float, mhalo_1e9msun: float, t_l_gyr: float, tun) -> float:
    """Gao+2024 analytical-background radius scale in physical kpc."""

    rvir_kpc = Rv_kpc(float(mhalo_1e9msun), float(t_l_gyr), tun)
    halo_mass_kg = float(mhalo_1e9msun) * 1.0e9 * M_sun
    rvir_m = rvir_kpc * kpc
    spin_parameter = float(spin_norm) / math.sqrt(2.0 * G * halo_mass_kg * rvir_m)
    re_kpc = spin_parameter * rvir_kpc / math.sqrt(2.0)
    return check_finite_positive(re_kpc, name="Gao+2024 evolution effective radius in kpc")

def resolve_birth_re_kpc(
    *,
    halomass_msun: float,
    redshift: float,
    jsp: float,
) -> float:
    return gao2023_birth_re_kpc(jsp, halomass_msun, redshift)

def resolve_background_re_kpc(
    *,
    mhalo_1e9msun: float,
    t_l_gyr: float,
    spin_norm: float,
    tun,
) -> float:
    return gao2023_evolution_re_kpc(spin_norm, mhalo_1e9msun, t_l_gyr, tun)

"""
Function-only IMBH seeding estimator for GC formation outputs.

Metallicity inputs are assumed to be Z/Zsun, not [Fe/H].  All coefficients are
hard-coded in the equations below.  The public API is intentionally small and
works with ``from IMBH import *`` because imported modules are underscore-named.
"""

def radius_eq7(cluster_mass_msun):
    """Eq. (7): GC 3D half-mass radius in pc."""

    mass = np.asarray(cluster_mass_msun, dtype=float)
    radius = 0.125 * 2.365 / 1.3 * (np.clip(mass, 1.0e-30, None) / 1.0e4) ** 0.180
    return float(radius) if radius.ndim == 0 else radius


def metallicity_to_z_ratio(metallicity):
    """Return metallicity unchanged; input is assumed to be Z/Zsun."""

    z_ratio = np.asarray(metallicity, dtype=float)
    return float(z_ratio) if z_ratio.ndim == 0 else z_ratio


def projected_half_mass_radius_plummer(r_h_pc):
    """Convert Plummer 3D half-mass radius to projected half-mass radius."""

    radius = np.asarray(r_h_pc, dtype=float) / 1.305
    return float(radius) if radius.ndim == 0 else radius


def sigma_h_from_mass_radius(cluster_mass_msun, r_h_pc):
    """Projected half-mass surface density in Msun pc^-2.

    The input radius is the 3D half-mass radius.  For a Plummer profile,
    R_h,proj = r_h / 1.305 and Sigma_h = M / (2 pi R_h,proj^2).
    """

    mass = np.asarray(cluster_mass_msun, dtype=float)
    radius = np.asarray(r_h_pc, dtype=float)
    scalar_output = mass.ndim == 0 and radius.ndim == 0
    mass, radius = np.broadcast_arrays(mass, radius)

    projected_radius = radius / 1.305
    sigma_h = np.where(
        projected_radius > 0.0,
        mass / (2.0 * np.pi * projected_radius**2),
        0.0,
    )
    return float(sigma_h) if scalar_output else sigma_h


def eq9_coeffs(z_ratio):
    """Piecewise coefficients for Eq. (9), using Z/Zsun."""

    z = np.asarray(z_ratio, dtype=float)
    log_z = np.log10(np.clip(z, 1.0e-30, None))

    A = np.where(z < 0.126, -1790.07 * log_z - 7392.65,
                  np.where(z < 0.398, 9707.84 * log_z + 2627.71,
                            1147.68 * log_z - 721.18))
    B = np.where(z < 0.126, 162.46 * log_z + 829.42,
                  np.where(z < 0.398, -1015.72 * log_z - 211.38,
                            -166.92 * log_z + 126.15))
    C = np.where(z < 0.126, 4734.11 * log_z + 16556.82,
                  np.where(z < 0.398, -23585.20 * log_z - 7814.04,
                            -2002.25 * log_z + 471.59))
    log_sigma_crit = np.where(z < 0.126, 0.0386 * log_z + 4.53,
                               np.where(z < 0.398, 0.91 * log_z + 5.22,
                                         0.91 * log_z + 5.22))
    return A, B, C, log_sigma_crit


def eq10_coeffs(z_ratio):
    """Piecewise coefficients for Eq. (10), using Z/Zsun."""

    z = np.asarray(z_ratio, dtype=float)
    log_z = np.log10(np.clip(z, 1.0e-30, None))

    D = np.where(z < 0.079, -37.37 * log_z + 1452.33,
                  np.where(z < 0.316, -922.15 * log_z + 466.71,
                            -611.27 * log_z + 628.40))
    E = np.where(z < 0.079, 81.66 * log_z - 6892.57,
                  np.where(z < 0.316, 4242.58 * log_z - 2280.02,
                            2620.25 * log_z - 3137.93))
    return D, E


def imbh_mass_eq9(sigma_h_msun_pc2, z_ratio):
    """Eq. (9): IMBH mass fit within the calibrated surface-density range."""

    sigma_h = np.asarray(sigma_h_msun_pc2, dtype=float)
    z = np.asarray(z_ratio, dtype=float)
    scalar_output = sigma_h.ndim == 0 and z.ndim == 0
    sigma_h, z = np.broadcast_arrays(sigma_h, z)

    A, B, C, log_sigma_crit = eq9_coeffs(z)
    log_sigma_h = np.log10(np.clip(sigma_h, 1.0e-30, None))
    mass = A * log_sigma_h + B * log_sigma_h**2 + C
    mass = np.where(sigma_h >= 10.0**log_sigma_crit, mass, 0.0)
    mass = np.where(np.isfinite(mass), mass, 0.0)
    mass = np.clip(mass, 0.0, None)
    return float(mass) if scalar_output else mass


def imbh_mass_eq10(sigma_h_msun_pc2, z_ratio):
    """Eq. (10): high-surface-density extrapolation."""

    sigma_h = np.asarray(sigma_h_msun_pc2, dtype=float)
    z = np.asarray(z_ratio, dtype=float)
    scalar_output = sigma_h.ndim == 0 and z.ndim == 0
    sigma_h, z = np.broadcast_arrays(sigma_h, z)

    D, E = eq10_coeffs(z)
    mass = D * np.log10(np.clip(sigma_h, 1.0e-30, None)) + E
    mass = np.where(np.isfinite(mass), mass, 0.0)
    mass = np.clip(mass, 0.0, None)
    return float(mass) if scalar_output else mass


def imbh_mass_from_sigma_metallicity(sigma_h_msun_pc2, z_ratio):
    """Estimate IMBH mass from Sigma_h and metallicity Z/Zsun."""

    sigma_h = np.asarray(sigma_h_msun_pc2, dtype=float)
    z = np.asarray(z_ratio, dtype=float)
    scalar_output = sigma_h.ndim == 0 and z.ndim == 0
    sigma_h, z = np.broadcast_arrays(sigma_h, z)

    z = np.clip(z, 1.0e-2, 1.0)
    mass = imbh_mass_eq9(sigma_h, z)
    use_eq10 = np.log10(np.clip(sigma_h, 1.0e-30, None)) >= 5.22
    mass = np.where(use_eq10, imbh_mass_eq10(sigma_h, z), mass)
    mass = np.where(mass >= 100.0, mass, 0.0)
    return float(mass) if scalar_output else mass


def estimate_for_gc(cluster_mass_msun, metallicity):
    """Full GC-level IMBH estimate.

    Parameters
    ----------
    cluster_mass_msun : float or array-like
        GC mass in Msun.
    metallicity : float or array-like
        Metallicity ratio Z/Zsun.

    Returns
    -------
    dict
        Same output keys as the original class-based implementation.
    """

    mass = np.asarray(cluster_mass_msun, dtype=float)
    z_ratio_raw = np.asarray(metallicity, dtype=float)
    scalar_output = mass.ndim == 0 and z_ratio_raw.ndim == 0
    mass, z_ratio_raw = np.broadcast_arrays(mass, z_ratio_raw)

    r_h_3d_pc = 0.125 * 2.365 / 1.3 * (np.clip(mass, 1.0e-30, None) / 1.0e4) ** 0.180
    sigma_h = sigma_h_from_mass_radius(mass, r_h_3d_pc)
    z_ratio_used = np.clip(z_ratio_raw, 1.0e-2, 1.0)
    imbh_mass = imbh_mass_from_sigma_metallicity(sigma_h, z_ratio_used)

    output = {
        "r_h_pc": r_h_3d_pc,
        "sigma_h_msun_pc2": sigma_h,
        "z_ratio_input": z_ratio_raw,
        "z_ratio_used": z_ratio_used,
        "imbh_mass_msun": imbh_mass,
    }
    if scalar_output:
        return {key: float(np.asarray(value)) for key, value in output.items()}
    return output