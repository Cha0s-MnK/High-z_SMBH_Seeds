import math

def check_finite(val, name="val"):
    val = float(val)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, but got {val}!")
    return val
    
def check_finite_non_negative(val, name="val"):
    val = float(val)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, but got {val}!")
    return val

def check_finite_positive(val, name="val"):
    val = float(val)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and positive, but got {val}!")
    return val

def Omega_m(z: float) -> float:
    """flat LambdaCDM matter density parameter"""
    check_finite_non_negative(z, name="Redshift z")

    z_plus_1_cubed = (1.0 + z) ** 3
    return Omega_m0 * z_plus_1_cubed / (1.0 - Omega_m0 + Omega_m0 * z_plus_1_cubed)

def CosmicAgeGyr2Redshift(t_Gyr: float) -> float:
    """flat LambdaCDM cosmic age to redshift conversion without radiation"""
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")

    z = (SqrtOmega_Lambda0OverOmega_m0 / math.sinh(t_Gyr / t_Lambda_Gyr)) ** (2.0 / 3.0) - 1.0
    check_finite_non_negative(z, name="Redshift z")
    return z