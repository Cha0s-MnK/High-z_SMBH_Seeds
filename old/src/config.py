# ===================== #
# CONFIGURE ENVIRONMENT #
# ===================== #

import math
import numpy as np
from scipy import interpolate

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
PI        = np.pi               # π
yr        = 365.25 * 24 * 3600    # Julian year [s]

kpc       = 1.0e3 * pc            # kiloparsec [m]
Mpc       = 1.0e6 * pc            # megaparsec [m]
Myr       = 1.0e6 * yr            # megayear [s]

# DESI 2024 + CMB
Omega_m0           = 0.307 # present-day matter density parameter
Omega_Lambda0      = 1 - Omega_m0 # present-day dark-energy density parameter
ReducedHubbleConst = 0.6797 # reduced Hubble constant h; H_0 = 100 h (km/s)/Mpc
t_universe = 13.780 # age of the universe [Gyr]

NUM_PROC = 16
OUT_DIR  = "/lingshan/disk3/subonan/_output"
STD_DPI  = 512
STD_FPS  = 24.0

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

def Omega_m(z: float) -> float:
    """matter density parameter Ω_m(z) for flat ΛCDM without radiation"""
    check_finite_non_negative(z, name="Redshift z")

    z_plus_1_cubed = (1.0 + z) ** 3
    return Omega_m0 * z_plus_1_cubed / (1.0 - Omega_m0 + Omega_m0 * z_plus_1_cubed)

def Rv(m, z):
    """
    Virial radius in pc for halo mass m in Msun.

    Uses flat ΛCDM without radiation and the Bryan & Norman
    virial overdensity relative to the critical density.

    Requirements
    ------------
    H0 must be in units compatible with G.
    For output in pc:
        H0 should be in km s^-1 pc^-1
        G  should be in pc (km/s)^2 Msun^-1
        m  should be in Msun
    """

    m = float(m)
    z = float(z)

    if not np.isfinite(m) or m <= 0.0:
        raise ValueError(f"Halo mass m must be finite and positive. Got m={m}")

    if not np.isfinite(z) or z < 0.0:
        raise ValueError(f"Redshift z must be finite and non-negative. Got z={z}")

    # Dimensionless Hubble parameter
    ez2 = Omega_m0 * (1.0 + z)**3 + Omega_Lambda0

    # Critical density at redshift z
    rhocrit = 3.0 * H0**2 * ez2 / (8.0 * np.pi * G)

    # Matter density parameter at redshift z
    Omega_m_at_z = Omega_m0 * (1.0 + z)**3 / ez2

    # Bryan & Norman virial overdensity relative to critical density
    x = Omega_m_at_z - 1.0
    Delta_vir = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    # Virial radius
    return (3.0 * m / (4.0 * np.pi * Delta_vir * rhocrit))**(1.0 / 3.0)

def CosmicAgeGyr2Redshift(t_Gyr: float) -> float:
    """cosmic age to redshift conversion for flat ΛCDM without radiation"""
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")

    z = (SqrtOmega_Lambda0OverOmega_m0 / math.sinh(t_Gyr / t_Lambda_Gyr)) ** (2.0 / 3.0) - 1.0
    check_finite_non_negative(z, name="Redshift z")
    return z

# Behroozi+2013 stellar mass-halo-mass(SMHM) relation

def f_x_SMHM(x: float, z: float) -> float:
    check_finite(x, name="lg(M_h/M_1) x")
    check_finite_non_negative(z, name="Redshift z")

    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    alpha = - 1.412 + 0.731 * (a - 1.0) * nu
    delta = 3.508 + (2.608 * (a - 1.0) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu
    return - math.log10(10.0 ** (alpha * x) + 1.0) + delta * (math.log10(1.0 + math.exp(x))) ** gamma / (1.0 + math.exp(0.1**x))

def Mstar_1e9Msun_SMHM(Mhalo_1e9Msun: float, t_Gyr: float, scatter: bool = False) -> float:
    check_finite_positive(Mhalo_1e9Msun, name="Halo mass in 1e9 Msun Mhalo_1e9Msun")
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")

    z = CosmicAgeGyr2Redshift(t_Gyr)
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