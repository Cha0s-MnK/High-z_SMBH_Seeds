#!/usr/bin/env python3
"""Plot the z-matched TNG50-Dark and TNG100-Dark halo mass functions.

The script measures the differential HMF from IllustrisTNG group catalogues
using Group_M_Crit200, then overlays a Tinker (2008) Lambda-CDM prediction
computed with the IllustrisTNG cosmological parameters.

Examples
--------
python plot_tng_hmf.py \
    --tng50 '/data/TNG50-1-Dark/output/groups_099/fof_subhalo_tab_099.*.hdf5' \
    --tng100 '/data/TNG100-1-Dark/output/groups_099/fof_subhalo_tab_099.*.hdf5' \
    --output tng_hmf_z0

`Group_M_Crit200` uses the native TNG mass unit, 1e10 Msun/h. The group
catalogue header provides the redshift but not the box size or Hubble parameter,
so the full-box TNG50 (35 cMpc/h) and TNG100 (75 cMpc/h) values are defaults.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


TNG_COSMOLOGY = {
    "flat": True,
    "H0": 67.74,
    "Om0": 0.3089,
    "Ob0": 0.0486,
    "sigma8": 0.8159,
    "ns": 0.9667,
}
TNG_H = TNG_COSMOLOGY["H0"] / 100.0
TNG50_1_DARK_DM_PARTICLE_MASS_MSUN = 5.3843825e5
TNG100_1_DARK_DM_PARTICLE_MASS_MSUN = 8.8565106e6


@dataclass(frozen=True)
class GroupCatalogue:
    """A single TNG group catalogue reduced to what is needed for an HMF."""

    label: str
    masses_msun: np.ndarray
    volume_cmpc3: float
    redshift: float
    h: float
    dm_particle_mass_msun: float | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure M200c HMFs from TNG50-Dark and TNG100-Dark group "
            "catalogues and compare them with a Tinker08 prediction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tng50",
        nargs="+",
        required=True,
        metavar="PATH_OR_GLOB",
        help="TNG50-1-Dark group-catalogue file(s), or quoted glob pattern(s).",
    )
    parser.add_argument(
        "--tng100",
        nargs="+",
        required=True,
        metavar="PATH_OR_GLOB",
        help="TNG100-1-Dark group-catalogue file(s), or quoted glob pattern(s).",
    )
    parser.add_argument(
        "--output",
        default="tng_hmf",
        help="Output path without a file extension.",
    )
    parser.add_argument(
        "--mass-field",
        default="Group_M_Crit200",
        help="Dataset in the Group HDF5 group used as halo mass.",
    )
    parser.add_argument(
        "--mmin",
        type=float,
        default=1.0e9,
        help="Minimum halo mass shown and included in the binned HMF [Msun].",
    )
    parser.add_argument(
        "--mmax",
        type=float,
        default=1.0e15,
        help="Maximum halo mass shown and included in the binned HMF [Msun].",
    )
    parser.add_argument(
        "--dlogm",
        type=float,
        default=0.20,
        help="Bin width in log10(M200c/Msun) [dex].",
    )
    parser.add_argument(
        "--redshift",
        type=float,
        default=None,
        help="Redshift for the theory curve; default reads it from both catalogues.",
    )
    parser.add_argument(
        "--hubble-param",
        type=float,
        default=TNG_H,
        help="Dimensionless Hubble parameter used for TNG native-unit conversion.",
    )
    parser.add_argument(
        "--tng50-box-size-cmpc",
        type=float,
        default=35.0 / TNG_H,
        metavar="CMPC",
        help="TNG50 box side length [cMpc].",
    )
    parser.add_argument(
        "--tng100-box-size-cmpc",
        type=float,
        default=75.0 / TNG_H,
        metavar="CMPC",
        help="TNG100 box side length [cMpc].",
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=300,
        help=(
            "Particle count used only to mark conservative resolution limits. "
            "It does not discard catalogue entries."
        ),
    )
    parser.add_argument(
        "--tng50-dm-particle-mass",
        type=float,
        default=TNG50_1_DARK_DM_PARTICLE_MASS_MSUN,
        metavar="MSUN",
        help=(
            "TNG50 DM-particle mass [Msun] for the resolution-limit line; "
            "default is TNG50-1-Dark."
        ),
    )
    parser.add_argument(
        "--tng100-dm-particle-mass",
        type=float,
        default=TNG100_1_DARK_DM_PARTICLE_MASS_MSUN,
        metavar="MSUN",
        help=(
            "TNG100 DM-particle mass [Msun] for the resolution-limit line; "
            "default is TNG100-1-Dark."
        ),
    )
    parser.add_argument(
        "--no-resolution-lines",
        action="store_true",
        help="Do not draw particle-number resolution-limit lines.",
    )
    return parser.parse_args()


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    """Resolve quoted glob patterns and ordinary paths into HDF5 files."""

    files: list[Path] = []
    for pattern in patterns:
        matched = [Path(path) for path in glob.glob(pattern)]
        files.extend(matched if matched else [Path(pattern)])

    unique_files = sorted(set(files))
    missing = [str(path) for path in unique_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "No group-catalogue file was found for: " + ", ".join(missing)
        )
    return unique_files


def read_group_catalogue(
    paths: Iterable[Path],
    label: str,
    mass_field: str,
    h: float,
    box_size_cmpc: float,
    dm_particle_mass_override_msun: float | None,
) -> GroupCatalogue:
    """Read all shards of one TNG group catalogue in physical units."""

    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading raw TNG group catalogues requires h5py. Install it in the "
            "environment used to run this script."
        ) from error

    paths = list(paths)
    if not (h > 0.0 and box_size_cmpc > 0.0):
        raise ValueError(f"{label}: h and box size must both be positive.")
    mass_chunks: list[np.ndarray] = []
    with h5py.File(paths[0], "r") as first_file:
        header = first_file["Header"].attrs
        redshift = float(header["Redshift"])
        mass_table = np.asarray(header["MassTable"], dtype=float) if "MassTable" in header else None

    for path in paths:
        with h5py.File(path, "r") as catalogue:
            header = catalogue["Header"].attrs
            if not np.isclose(float(header["Redshift"]), redshift):
                raise ValueError(f"{label}: inconsistent Redshift across catalogue shards.")
            try:
                masses_native = np.asarray(catalogue["Group"][mass_field], dtype=float)
            except KeyError as error:
                raise KeyError(
                    f"{label}: Group/{mass_field} was not found in {path}."
                ) from error
            mass_chunks.append(masses_native)

    # TNG group-catalogue mass unit: 1e10 Msun/h.
    masses_msun = np.concatenate(mass_chunks) * 1.0e10 / h
    masses_msun = masses_msun[np.isfinite(masses_msun) & (masses_msun > 0.0)]

    volume_cmpc3 = box_size_cmpc**3

    dm_particle_mass_msun = dm_particle_mass_override_msun
    if (
        dm_particle_mass_msun is None
        and mass_table is not None
        and mass_table.size > 1
        and mass_table[1] > 0.0
    ):
        dm_particle_mass_msun = mass_table[1] * 1.0e10 / h
    if dm_particle_mass_msun is not None and dm_particle_mass_msun <= 0.0:
        raise ValueError(f"{label}: DM-particle mass must be positive.")

    return GroupCatalogue(
        label=label,
        masses_msun=masses_msun,
        volume_cmpc3=volume_cmpc3,
        redshift=redshift,
        h=h,
        dm_particle_mass_msun=dm_particle_mass_msun,
    )


def make_log_mass_bins(mmin: float, mmax: float, dlogm: float) -> np.ndarray:
    if not (mmin > 0.0 and mmax > mmin and dlogm > 0.0):
        raise ValueError("Require 0 < mmin < mmax and dlogm > 0.")
    logmin = np.log10(mmin)
    logmax = np.log10(mmax)
    return np.arange(logmin, logmax + 1.01 * dlogm, dlogm)


def measure_differential_hmf(
    catalogue: GroupCatalogue, log_mass_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centres, d n/d log10 M, and 68% Poisson bounds."""

    log_masses = np.log10(catalogue.masses_msun)
    counts, _ = np.histogram(log_masses, bins=log_mass_edges)
    dlogm = np.diff(log_mass_edges)
    phi = counts / (catalogue.volume_cmpc3 * dlogm)

    # Exact central 68% Poisson confidence interval for each non-empty bin.
    alpha = 1.0 - 0.682689492137086
    count_lo = np.where(
        counts > 0,
        0.5 * chi2.ppf(alpha / 2.0, 2.0 * counts),
        0.0,
    )
    count_hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (counts + 1))
    phi_lo = count_lo / (catalogue.volume_cmpc3 * dlogm)
    phi_hi = count_hi / (catalogue.volume_cmpc3 * dlogm)
    log_centres = 0.5 * (log_mass_edges[1:] + log_mass_edges[:-1])
    return log_centres, phi, phi_lo, phi_hi


def tinker08_dndlog10m(masses_msun: np.ndarray, redshift: float, h: float) -> np.ndarray:
    """Evaluate the Tinker08 M200c mass function in cMpc^-3 dex^-1.

    Colossus uses masses in Msun/h and returns dn/dlnM in h^3 Mpc^-3.
    The conversion to the physical units plotted here is applied explicitly.
    """

    try:
        from colossus.cosmology import cosmology
        from colossus.lss import mass_function
    except ImportError as error:
        raise ImportError(
            "The theoretical curve requires Colossus. Install it in the runtime "
            "environment, for example with `python -m pip install colossus`."
        ) from error

    cosmology_name = "IllustrisTNG_HMF"
    cosmology.addCosmology(cosmology_name, TNG_COSMOLOGY)
    cosmology.setCosmology(cosmology_name)

    masses_msun_over_h = np.asarray(masses_msun, dtype=float) * h
    dndlnm_h3_mpc3 = mass_function.massFunction(
        masses_msun_over_h,
        redshift,
        mdef="200c",
        model="tinker08",
        q_out="dndlnM",
    )
    return np.asarray(dndlnm_h3_mpc3) * h**3 * np.log(10.0)


def plot_catalogue_hmf(
    ax: plt.Axes,
    catalogue: GroupCatalogue,
    log_mass_edges: np.ndarray,
    colour: str,
    marker: str,
) -> None:
    logm, phi, phi_lo, phi_hi = measure_differential_hmf(catalogue, log_mass_edges)
    valid = phi > 0.0
    log_phi = np.log10(phi[valid])
    log_phi_lo = np.log10(phi_lo[valid])
    log_phi_hi = np.log10(phi_hi[valid])
    yerr = np.vstack((log_phi - log_phi_lo, log_phi_hi - log_phi))
    ax.errorbar(
        logm[valid],
        log_phi,
        yerr=yerr,
        fmt=marker,
        ms=5.5,
        mew=0.7,
        lw=1.0,
        capsize=2.3,
        color=colour,
        mec="white",
        label=catalogue.label,
        zorder=3,
    )


def add_resolution_limit(
    ax: plt.Axes,
    catalogue: GroupCatalogue,
    min_particles: int,
    colour: str,
) -> None:
    if catalogue.dm_particle_mass_msun is None:
        print(
            f"Warning: {catalogue.label}: no DM-particle mass is available; "
            "no resolution-limit line was drawn."
        )
        return
    limit_msun = min_particles * catalogue.dm_particle_mass_msun
    log_limit_msun = np.log10(limit_msun)
    xmin, xmax = ax.get_xlim()
    if xmin <= log_limit_msun <= xmax:
        ax.axvline(log_limit_msun, color=colour, ls=":", lw=1.3, alpha=0.9)


def resolve_redshift(catalogues: Iterable[GroupCatalogue], requested_redshift: float | None) -> float:
    redshifts = np.array([catalogue.redshift for catalogue in catalogues])
    if requested_redshift is not None:
        if not np.allclose(redshifts, requested_redshift, rtol=0.0, atol=1.0e-5):
            raise ValueError(
                "The supplied --redshift does not match the group-catalogue header. "
                "Use two catalogues from the same snapshot."
            )
        return requested_redshift
    if not np.allclose(redshifts, redshifts[0], rtol=0.0, atol=1.0e-5):
        raise ValueError("TNG50 and TNG100 catalogues must come from the same redshift.")
    return float(redshifts[0])


def main() -> None:
    args = parse_arguments()
    tng50 = read_group_catalogue(
        expand_paths(args.tng50),
        "TNG50-1-Dark",
        args.mass_field,
        args.hubble_param,
        args.tng50_box_size_cmpc,
        args.tng50_dm_particle_mass,
    )
    tng100 = read_group_catalogue(
        expand_paths(args.tng100),
        "TNG100-1-Dark",
        args.mass_field,
        args.hubble_param,
        args.tng100_box_size_cmpc,
        args.tng100_dm_particle_mass,
    )
    redshift = resolve_redshift((tng50, tng100), args.redshift)
    if redshift > 2.5:
        print(
            "Warning: Tinker08 is being evaluated beyond its z <= 2.5 calibration "
            "range; treat the theoretical line as an extrapolation."
        )
    if not np.isclose(tng50.h, tng100.h, rtol=0.0, atol=1.0e-8):
        raise ValueError("TNG50 and TNG100 catalogues have different Hubble parameters.")

    log_mass_edges = make_log_mass_bins(args.mmin, args.mmax, args.dlogm)
    theory_masses = np.logspace(np.log10(args.mmin), np.log10(args.mmax), 500)
    theory_phi = tinker08_dndlog10m(theory_masses, redshift, tng50.h)

    fig, ax = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
    ax.set_xlim(np.log10(args.mmin), np.log10(args.mmax))
    ax.plot(
        np.log10(theory_masses),
        np.log10(theory_phi),
        color="black",
        lw=2.0,
        label=rf"Tinker08 $\Lambda$CDM ($z={redshift:.3g}$)",
        zorder=2,
    )
    plot_catalogue_hmf(ax, tng50, log_mass_edges, "#2171b5", "o")
    plot_catalogue_hmf(ax, tng100, log_mass_edges, "#cb181d", "s")

    if not args.no_resolution_lines:
        add_resolution_limit(ax, tng50, args.min_particles, "#2171b5")
        add_resolution_limit(ax, tng100, args.min_particles, "#cb181d")

    ax.set_xlabel(r"$\log_{10}(M_{200\mathrm{c}}/M_\odot)$")
    ax.set_ylabel(
        r"$\log_{10}\!\left[\frac{\mathrm{d}n}{\mathrm{d}\log_{10}M_{200\mathrm{c}}}"
        r"\,/\,(\mathrm{cMpc}^{-3}\,\mathrm{dex}^{-1})\right]$"
    )
    ax.grid(True, alpha=0.22, linestyle=":", which="major")
    ax.legend(frameon=False, loc="best")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Saved {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
