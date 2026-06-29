# ===================== #
# CONFIGURE ENVIRONMENT #
# ===================== #

from __future__ import annotations # Annotations are not evaluated immediately when the file is imported.
import math # to be optimized
import numpy as np
import scipy
from scipy import interpolate
from typing import Tuple
import warnings

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
AU        = 1.495978707e11        # astronomical unit [m] (reference: https://en.wikipedia.org/wiki/Astronomical_unit)
c         = 2.99792458e8          # speed of light [m·s⁻¹] (reference: https://en.wikipedia.org/wiki/Speed_of_light)
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
m_p       = 1.6726219259552e-27   # proton mass [kg] (reference: https://en.wikipedia.org/wiki/Proton)
m_u       = 1.6605390689252e-27   # unified atomic mass unit [kg] (reference: https://en.wikipedia.org/wiki/Dalton_(unit))
M_sun     = 1.988416e30           # solar mass [kg] (reference: https://en.wikipedia.org/wiki/Solar_mass)
N_A       = 6.02214076e23         # Avogadro constant [mol⁻¹] (reference: https://en.wikipedia.org/wiki/Avogadro_constant)
pc        = 3.0856775814913673e16 # parsec [m] (reference: https://en.wikipedia.org/wiki/Parsec)
PI        = np.pi                 # π
sigma_T   = 6.652458705162e-29    # Thomson cross section [m^2] (reference: https://en.wikipedia.org/wiki/Thomson_scattering)
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

MIN_RAD_PC = 1.0 # inner aperture/bin edge
NSC_RAD_PC = 6.0 # public stellar NSC aperture
Eddington_varepsilon = 0.1 # Eddington radiative efficiency
Eddington_time_Gyr = Eddington_varepsilon * sigma_T * c / (4.0 * PI * G * m_p * (1.0 - Eddington_varepsilon)) / Gyr # Eddington time [Gyr]
M_BH_warning = 1.0e12 # BH mass threshold for warnings about excessive Eddington growth [Msun]
STD_DPI = 512

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

def fixed_tree_mpb_branch_id(log_mh, branch_id): # to be checked
    """Return the project MPB branch ID from fixed-tree rows.

    This intentionally follows the Gao+2024 formation scripts and the
    previous ``src/main.py`` behaviour: choose the branch ID at the first row
    with maximum log10(M_h/Msun).
    """

    log_mh_arr = np.asarray(log_mh, dtype=float)
    branch_raw = np.asarray(branch_id)
    if log_mh_arr.ndim != 1 or branch_raw.ndim != 1:
        raise ValueError("Fixed-tree MPB inputs must be one-dimensional arrays.")
    if len(log_mh_arr) == 0:
        raise ValueError("Cannot identify the MPB branch from empty fixed-tree arrays.")
    if len(log_mh_arr) != len(branch_raw):
        raise ValueError("Fixed-tree log_mh and branch_id arrays must have matching lengths.")
    if np.any(~np.isfinite(log_mh_arr)):
        raise ValueError("Fixed-tree log_mh values must be finite.")

    branch_values = []
    for value in branch_raw:
        if isinstance(value, (int, np.integer)):
            branch = int(value)
        elif isinstance(value, str):
            try:
                branch = int(value)
            except ValueError:
                value_float = check_finite(float(value), name="fixed-tree branch ID")
                if abs(value_float) > 2**53:
                    raise ValueError("Large fixed-tree branch IDs must be read as integers, not float-like text.")
                branch = int(round(value_float))
                if abs(value_float - float(branch)) > 1.0e-6:
                    raise ValueError(f"Fixed-tree branch ID is not integer-like: {value}")
        else:
            value_float = check_finite(float(value), name="fixed-tree branch ID")
            if abs(value_float) > 2**53:
                raise ValueError("Large fixed-tree branch IDs must be read as integers, not floats.")
            branch = int(round(value_float))
            if abs(value_float - float(branch)) > 1.0e-6:
                raise ValueError(f"Fixed-tree branch ID is not integer-like: {value}")
        if branch < 0:
            raise ValueError(f"Fixed-tree branch ID must be non-negative; got {branch}")
        branch_values.append(branch)

    return int(branch_values[int(np.argmax(log_mh_arr))])

# Eddington-limited BH growth utilities

def grow_eddington_mass_msun(M_BH: float, dt_Gyr: float, f_Eddington: float):
    """
    Grow BH mass by simplified Eddington-limited accretion.

    M_BH (t + dt) = M_BH (t) * exp(f_Eddington * dt / t_Eddington)
    """
    M_BH        = check_finite_non_negative(M_BH, name="BH mass M_BH before Eddington accretion")
    dt_Gyr      = check_finite_non_negative(dt_Gyr, name="Eddington accretion timestep dt_Gyr")
    f_Eddington = check_finite_non_negative(f_Eddington, name="Eddington ratio f_Eddington")

    if M_BH > 0.0 and dt_Gyr > 0.0 and f_Eddington > 0.0:
        M_BH *= math.exp(f_Eddington * dt_Gyr / Eddington_time_Gyr)
        check_finite_non_negative(M_BH, name="BH mass M_BH after Eddington accretion")

    if M_BH > M_BH_warning:
        warnings.warn(f"BH mass M_BH after Eddington accretion exceeds {M_BH_warning:.0e} Msun.", RuntimeWarning)

    return M_BH

# linear interpolation on a uniformly spaced grid

def lininterp_uniform(xq, x_grid, y_grid, dx_inv=None, *, allow_extrapolate=False): # to be checked
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

# cosmology

def E(z: float) -> float:
    """
    dimensionless Hubble parameter E(z) = H(z) / H0 for flat ΛCDM without radiation
    """
    check_finite_non_negative(z, name="Redshift z")

    return np.sqrt(Omega_m0 * (1.0 + z)**3 + Omega_Lambda0)

def H(z: float) -> float:
    """
    Hubble parameter H(z) in (km/s)/Mpc for flat ΛCDM without radiation
    """
    return H0 * E(z)

def Omega_m(z: float) -> float:
    """matter density parameter Ω_m(z) for flat ΛCDM without radiation"""
    check_finite_non_negative(z, name="Redshift z")

    zPlus1Cubed = (1.0 + z) ** 3
    return Omega_m0 * zPlus1Cubed / (1.0 - Omega_m0 + Omega_m0 * zPlus1Cubed)

def Redshift2CosmicAge(z: float, time_unit: str = "Gyr") -> float:
    """
    For flat ΛCDM without radiation, the age at redshift z has the analytic form:
        t(z) = 2 / (3 H0 sqrt(Omega_L)) * asinh(sqrt(Omega_L / Omega_M) / (1 + z)^(3/2))
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

def Rv(Mhalo: float, z: float) -> float:
    """
    Virial radius in kpc for halo mass Mhalo in Msun for flat ΛCDM without radiation

    Uses the Bryan & Norman virial overdensity relative to the critical density.
    """
    check_finite_positive(Mhalo, name="Halo mass in M☉ Mhalo")
    check_finite_non_negative(z, name="Redshift z")

    # critical density
    H_kpc = H(z=z) * 1.0e-3 # [(km/s)/Mpc] --> [(km/s)/kpc]
    rho_crit = 3.0 * (H_kpc ** 2) / (8.0 * PI * G_kpc) # [M☉/kpc³]

    # Bryan & Norman virial overdensity relative to critical density
    x = Omega_m(z=z) - 1.0
    Delta_c = 18.0 * (PI ** 2) + 82.0 * x - 39.0 * (x ** 2)

    # virial radius in kpc
    return np.cbrt(3.0 * Mhalo / (4.0 * PI * Delta_c * rho_crit))

"""
def Rv_kpc(Mhalo_1e9Msun: float, t_Gyr: float, tun: Tunables) -> float:
    check_finite_positive(Mhalo_1e9Msun, name="Halo mass Mhalo_1e9Msun")
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")

    z = CosmicAge2Redshift(t_Gyr, time_unit="Gyr")
    Omega_m_z = Omega_m(z)
    Delta_c = (18.0 * PI * PI + 82.0 * (Omega_m_z - 1.0) - 39.0 * (Omega_m_z - 1.0) ** 2) / Omega_m_z
    check_finite_positive(Delta_c, name="Average halo over-density at Rv Delta_c")
    Rv_kpc = 163.0 / ((1.0 + z) * tun.h) * (Mhalo_1e9Msun * tun.h * 200.0 / (1.0e3 * Omega_m0 * Delta_c)) ** (1.0 / 3.0)
    check_finite_positive(Rv_kpc, name="Halo virial radius in kpc Rv_kpc")
    return Rv_kpc
"""

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
    return check_finite_non_negative(z, name="Redshift z")

def v_v(Mhalo: float, z: float) -> float:
    """virial velocity in km/s for halo mass Mhalo in M☉ at redshift z for flat ΛCDM without radiation"""
    return np.sqrt(G_kpc * Mhalo / Rv(Mhalo=Mhalo, z=z))

# Behroozi+2013 stellar mass-halo mass(SMHM) relation

def f_x_SMHM(x: float, z: float) -> float:
    check_finite(x, name="lg(Mhalo/M1) x")
    check_finite_non_negative(z, name="Redshift z")

    a     = 1.0 / (1.0 + z)
    nu    = math.exp(- 4.0 * a * a)
    alpha = - 1.412 + 0.731 * (a - 1.0) * nu
    delta = 3.508 + (2.608 * (a - 1.0) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu
    exp   = 10.0 ** (-x)
    coef  = 0.0 if exp > 700.0 else 1.0 / (1.0 + math.exp(exp))
    return - math.log10(10.0 ** (alpha * x) + 1.0) + delta * (math.log10(1.0 + math.exp(x))) ** gamma * coef

def Mstar_SMHM(Mhalo: float, z: float, scatter: bool = False) -> float:
    check_finite_positive(Mhalo, name="Halo mass in M☉ Mhalo")
    check_finite_non_negative(z, name="Redshift z")

    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    epsilon = 10.0 ** (- 1.777 - 0.006 * (a - 1.0) * nu - 0.119 * (a - 1.0))
    M1 = 10.0 ** (11.514 - (1.793 * (a - 1.0) + 0.251 * z) * nu)
    lg_Mstar = math.log10(epsilon * M1) + f_x_SMHM(math.log10(Mhalo / M1), z) - f_x_SMHM(0.0, z)
    if scatter:
        xi = np.random.normal(0.0, 0.218 + 0.023 * z / (1.0 + z))
        lg_Mstar += xi
    return check_finite_positive(10 ** lg_Mstar, name="Stellar mass in M☉ Mstar")

"""
def Mstar_1e9Msun_SMHM(Mhalo_1e9Msun: float, t_Gyr: float, scatter: bool = False) -> float:
    check_finite_positive(Mhalo_1e9Msun, name="Halo mass in 1e9 M☉ Mhalo_1e9Msun")
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
    check_finite_positive(Mstar_1e9Msun, name="Stellar mass in 1e9 M☉ Mstar_1e9Msun")
    return Mstar_1e9Msun
"""

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

def makeLogMgcToLogMmaxInterpolator(Mc: float, Mmin: float = 1.0e5, dlog_mmax: float = 0.02): # to be checked
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

def upper_gamma2_log_mass(log_m: float, Mc: float) -> float: # to be checked
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

"""
def resolve_birth_re_kpc(halomass_msun: float, redshift: float, jsp: float) -> float:
    j_kpc_kms = float(jsp) * ReducedH0
    Rv_kpc = Rv(Mhalo=halomass_msun, z=redshift)
    hz_km_s_kpc = H(float(redshift)) * 1.0e-3
    Re = j_kpc_kms / (20.0 * hz_km_s_kpc * Rv_kpc)
    return check_finite_positive(Re, name="Gao+2024 birth effective radius in kpc")
"""

def calcRe(mhalo_1e9msun: float, t_Gyr: float, j: float) -> float:
    """compute the effective radius of the galactic disc in kpc"""

    Mhalo = float(mhalo_1e9msun) * 1.0e9
    Rv_kpc = Rv(Mhalo=Mhalo, z=CosmicAge2Redshift(t_Gyr, time_unit="Gyr"))
    lambdaB = j / math.sqrt(2.0 * G_kpc * Mhalo * Rv_kpc)
    Re = lambdaB * Rv_kpc / math.sqrt(2.0)
    return check_finite_positive(Re, name="Effective radius of the galactic disc in kpc Re")

"""
Function-only scalar IMBH seeding estimator for GC formation outputs.

Metallicity inputs are the literal ratio Z/Zsun, not [Fe/H].  The Rantala+2026
fit is evaluated for finite positive floats and warns outside the simulated
metallicity range 0.01 <= Z/Zsun <= 1.0.
"""

def initMRRofSCs(Mcl: float, f_h: float = 0.125) -> float:
    """Eq.7: initial mass-radius relation of star clusters,
    returning the star cluster 3D half-mass radius in pc."""

    check_finite_positive(Mcl, name="Star cluster mass Mcl in M☉")
    r_h = f_h * 2.365 / 1.3 * ((Mcl / 1.0e4) ** 0.18)
    return check_finite_positive(r_h, name="Star cluster 3D half-mass radius r_h in pc")

def calcSigma_h(Mcl: float, r_h: float) -> float:
    """Projected half-mass surface density in M☉/pc².

    The input radius is the 3D half-mass radius.  For a Plummer profile,
    r_h_2D = r_h / 1.305 and Sigma_h = M / (2 pi r_h_2D^2).
    """
    check_finite_positive(Mcl, name="Star cluster mass Mcl in M☉")
    check_finite_positive(r_h, name="Star cluster 3D half-mass radius r_h in pc")

    r_h_2D  = r_h / 1.305
    Sigma_h = Mcl / (2.0 * PI * r_h_2D**2)
    return check_finite_positive(Sigma_h, name="Projected half-mass surface density Sigma_h in M☉/pc²")

def calcMimbhEq9(Sigma_h: float, Z: float) -> float:
    """Eq.9: IMBH mass fit within the calibrated surface-density range."""
    Sigma_h = check_finite_positive(Sigma_h, name="Projected 2D half-mass surface density Sigma_h in M☉/pc²")
    lgZ     = math.log10(check_finite_non_negative(Z, name="Metallicity Z in Z☉"))

    if Z < 0.126:
        A, B, C, lgSigma_crit = - 1790.07 * lgZ - 7392.65, 162.46 * lgZ + 829.42, 4734.11 * lgZ + 16556.82, 0.0386 * lgZ + 4.53
    elif Z < 0.398:
        A, B, C, lgSigma_crit = 9707.84 * lgZ + 2627.71, - 1015.72 * lgZ - 211.38, - 23585.20 * lgZ - 7814.04, 0.91 * lgZ + 5.22
    else:
        A, B, C, lgSigma_crit = 1147.68 * lgZ - 721.18, - 166.92 * lgZ + 126.15, - 2002.25 * lgZ + 471.59, 0.91 * lgZ + 5.22
    lgSigma_h = math.log10(Sigma_h)
    Mimbh = 0.0 if Sigma_h <= 10.0**lgSigma_crit else A * lgSigma_h + B * lgSigma_h**2 + C
    Mimbh = check_finite(Mimbh, name="Eq.9 IMBH mass Mimbh in M☉")
    Mimbh = Mimbh if Mimbh >= 100.0 else 0.0
    return Mimbh

def calcMimbhEq10(Sigma_h: float, Z: float) -> float:
    """Eq.10: high-surface-density extrapolation."""
    Sigma_h = check_finite_positive(Sigma_h, name="Projected 2D half-mass surface density Sigma_h in M☉/pc²")
    lgZ     = math.log10(check_finite_non_negative(Z, name="Metallicity Z in Z☉"))

    if Z < 0.079:
        D, E = - 37.37 * lgZ + 1452.33, 81.66 * lgZ - 6892.57
    elif Z < 0.316:
        D, E = - 922.15 * lgZ + 466.71, 4242.58 * lgZ - 2280.02
    else:
        D, E = - 611.27 * lgZ + 628.40, 2620.25 * lgZ - 3137.93

    Mimbh = check_finite(D * math.log10(Sigma_h) + E, name="Eq.10 IMBH mass Mimbh in M☉")
    Mimbh = Mimbh if Mimbh >= 100.0 else 0.0
    return Mimbh

"""
def eq9_coeffs(Z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #Piecewise coefficients for Eq.9, using literal Z/Z☉

    lgZ = np.log10(check_finite_non_negative(Z, name="Metallicity Z in Z☉"))

    A = np.where(Z < 0.126, -1790.07 * lgZ - 7392.65,
                  np.where(Z < 0.398, 9707.84 * lgZ + 2627.71,
                            1147.68 * lgZ - 721.18))
    B = np.where(Z < 0.126, 162.46 * lgZ + 829.42,
                  np.where(Z < 0.398, -1015.72 * lgZ - 211.38,
                            -166.92 * lgZ + 126.15))
    C = np.where(Z < 0.126, 4734.11 * lgZ + 16556.82,
                  np.where(Z < 0.398, -23585.20 * lgZ - 7814.04,
                            -2002.25 * lgZ + 471.59))
    lgSigma_crit = np.where(Z < 0.126, 0.0386 * lgZ + 4.53,
                               np.where(Z < 0.398, 0.91 * lgZ + 5.22,
                                         0.91 * lgZ + 5.22))
    return A, B, C, lgSigma_crit

def eq10_coeffs(Z: float) -> Tuple[np.ndarray, np.ndarray]:
    #Piecewise coefficients for Eq.10, using literal Z/Z☉

    lgZ = np.log10(check_finite_non_negative(Z, name="Metallicity Z in Z☉"))

    D = np.where(Z < 0.079, -37.37 * lgZ + 1452.33,
                  np.where(Z < 0.316, -922.15 * lgZ + 466.71,
                            -611.27 * lgZ + 628.40))
    E = np.where(Z < 0.079, 81.66 * lgZ - 6892.57,
                  np.where(Z < 0.316, 4242.58 * lgZ - 2280.02,
                            2620.25 * lgZ - 3137.93))
    return D, E

def imbh_mass_eq9(sigma_h_msun_pc2, z_ratio):
    #Eq.9: IMBH mass fit within the calibrated surface-density range.

    sigma_h = np.asarray(sigma_h_msun_pc2, dtype=float)
    z = np.asarray(z_ratio, dtype=float)
    scalar_output = sigma_h.ndim == 0 and z.ndim == 0
    sigma_h, z = np.broadcast_arrays(sigma_h, z)

    A, B, C, lgSigma_crit = eq9_coeffs(z)
    lgSigma_h = np.log10(np.clip(sigma_h, 1.0e-30, None))
    mass = A * lgSigma_h + B * lgSigma_h**2 + C
    mass = np.where(sigma_h >= 10.0**lgSigma_crit, mass, 0.0)
    mass = np.where(np.isfinite(mass), mass, 0.0)
    mass = np.clip(mass, 0.0, None)
    return float(mass) if scalar_output else mass

def imbh_mass_eq10(sigma_h_msun_pc2, z_ratio):
    #Eq.10: high-surface-density extrapolation.

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
    #Estimate IMBH mass from Sigma_h and metallicity Z/Zsun.

    sigma_h = np.asarray(sigma_h_msun_pc2, dtype=float)
    z = np.asarray(z_ratio, dtype=float)
    scalar_output = sigma_h.ndim == 0 and z.ndim == 0
    sigma_h, z = np.broadcast_arrays(sigma_h, z)

    mass = imbh_mass_eq9(sigma_h, z)
    use_eq10 = np.log10(np.clip(sigma_h, 1.0e-30, None)) >= 5.22
    mass = np.where(use_eq10, imbh_mass_eq10(sigma_h, z), mass)
    mass = np.where(mass >= 100.0, mass, 0.0)
    return float(mass) if scalar_output else mass
"""

def imbh_mass_from_sigma_metallicity(sigma_h_msun_pc2: float, z_ratio: float) -> float:
    """Estimate IMBH mass from Sigma_h and metallicity Z/Zsun."""
    Sigma_h = check_finite_positive(sigma_h_msun_pc2, name="Projected 2D half-mass surface density Sigma_h in M☉/pc²")
    Z       = check_finite_non_negative(z_ratio, name="Metallicity Z in Z☉")
    if Z < 0.01 or Z > 1.0:
        warnings.warn(f"Rantala+2026 IMBH fit evaluated outside 0.01 <= Z/Z☉ <= 1.0: Z/Z☉ = {Z:.6g}.",
            RuntimeWarning, stacklevel=2)

    Mimbh = calcMimbhEq10(Sigma_h, Z) if math.log10(Sigma_h) >= 5.22 else calcMimbhEq9(Sigma_h, Z)
    return Mimbh if Mimbh >= 100.0 else 0.0

def estimate_for_gc(Mcl: float, Z: float) -> dict:
    """Full GC-level IMBH estimate.

    Parameters
    ----------
    Mcl : float
        GC mass in Msun.
    Z : float
        Metallicity ratio Z/Zsun.

    Returns
    -------
    dict
        Same output keys as the original class-based implementation.
    """
    Mcl = check_finite_positive(Mcl, name="Star cluster mass Mcl in M☉")
    Z   = check_finite_non_negative(Z, name="Metallicity Z in Z☉")

    r_h     = initMRRofSCs(Mcl=Mcl, f_h=0.125)
    Sigma_h = calcSigma_h(Mcl=Mcl, r_h=r_h)
    Mimbh   = imbh_mass_from_sigma_metallicity(sigma_h_msun_pc2=Sigma_h, z_ratio=Z)

    return {
        "r_h_pc": r_h,
        "sigma_h_msun_pc2": Sigma_h,
        "Z": Z,
        "imbh_mass_msun": Mimbh,
    }