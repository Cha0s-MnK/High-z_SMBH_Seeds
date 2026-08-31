"""Gao+2024 GC formation-and-placement stage.

This script reads one fixed merger tree per target halo, identifies GC-forming
events from halo growth along each branch, samples a cluster initial mass
function for each event, assigns galactocentric radii, and writes:

- `all_<Ns>.txt`: every formed GC, whether it survives to the configured final
  redshift or not

Although the implementation is legacy-style and fairly stateful, the rough
flow is:
1. load one corrected tree from the configured fixed-tree directory
2. walk every retained branch node and identify rapid-growth events
3. form GCs and assign their birth radii
"""

import argparse
import csv
from dataclasses import dataclass
import numpy as np
from scipy import interpolate
import time
import os
import sys
import warnings
from pathlib import Path
from config import *

#use same cosmology as Illustris
fb = 0.167 #cosmic baryon fraction

# model parameters, as defined in CGL18
TREE_LOOKUP_BASENAME = "id_lookup_large_dark.csv"
ZFORM_FORMAT = "{:.10f}"

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Gao+2024 GC formation-and-placement stage.", allow_abbrev=False)
    parser.add_argument("ns", type=float, help="Sersic index N_s")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Path to the raw Gao+2024 data directory.",
    )
    parser.add_argument(
        "--tree-dir",
        type=Path,
        default=None,
        help="Optional fixed-tree directory. Defaults to <data-dir>/fixed_trees_large_spin.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where all_<Ns>.txt is written.",
    )
    parser.add_argument("--p2", type=float, default=6.75, help="GC formation-efficiency normalization")
    parser.add_argument("--p3", type=float, default=0.5, help="halo growth-rate threshold for triggering GC formation")
    parser.add_argument("--lg_cut-off_mass", dest="lg_cut_off_mass", type=float, default=12.0, help="log10 Schechter cutoff mass Mc in Msun")
    parser.add_argument(
        "--Mmin",
        type=float,
        default=1.0e5,
        help=(
            "minimum initial GC mass Mmin in linear Msun (default: 1e5); "
            "finite and positive and less than 1e6 Msun; controls the CIMF "
            "lower endpoint and event-budget eligibility"
        ),
    )
    parser.add_argument(
        "--IMBH",
        type=float,
        default=1.0,
        help=(
            "dimensionless IMBH seed coefficient (default: 1.0); finite and "
            "non-negative; applied once to the formation-time estimator result"
        ),
    )
    parser.add_argument(
        "--fit",
        choices=IMBH_FIT_CHOICES,
        default=DEFAULT_IMBH_FIT,
        help=(
            "formation-time IMBH mass prescription; choose Rantala+2026, "
            "Vergara+2026conservative, or Vergara+2026optimistic "
            "(default: Rantala+2026)"
        ),
    )
    parser.add_argument("--run-all", type=int, default=1, help="run all halos if 1, otherwise use the halo count and mass window below")
    parser.add_argument("--log-mh-min", type=float, default=11.5, help="minimum descendant z=0 host halo log mass for selection")
    parser.add_argument("--log-mh-max", type=float, default=12.5, help="maximum descendant z=0 host halo log mass for selection")
    parser.add_argument(
        "--n-halos",
        type=int,
        default=10,
        help="number of logarithmic descendant-mass bins and requested distinct halo trees when --run-all=0",
    )
    return parser


_parser = _build_arg_parser()
args = _parser.parse_args()

try:
    Mmin = check_finite_positive(args.Mmin, name="Minimum cluster mass Mmin")
    if Mmin >= 1.0e6:
        raise ValueError(f"Minimum cluster mass Mmin must be less than 1e6 Msun, but got Mmin = {Mmin}!")
    IMBH = check_finite_non_negative(args.IMBH, name="IMBH coefficient")
    fit = validate_imbh_fit(args.fit)
except ValueError as exc:
    _parser.error(str(exc))

log_Mmin = np.log10(Mmin)

# Sersic index
# ns = 2.2
ns = float(args.ns)
nsStr = f"{ns:.1f}"

data_dir = args.data_dir.resolve()
output_dir = args.output_dir.resolve()
output_dir.mkdir(parents = True, exist_ok = True)

treedir = args.tree_dir.resolve() if(args.tree_dir is not None) else (data_dir / "fixed_trees_large_spin")
if(not treedir.exists()):
    raise FileNotFoundError("Required Gao+2024 input path not found: " + str(treedir))

allcat = open(output_dir / ('all_'+nsStr+'.txt'), 'w') #full catalog of all GCs

p2 = float(args.p2)
p3 = float(args.p3)
lg_cut_off_mass = float(args.lg_cut_off_mass)
run_all = bool(args.run_all)
log_mh_min = float(args.log_mh_min)
log_mh_max = float(args.log_mh_max)
N = int(args.n_halos)

if(not run_all):
    if(N < 1):
        raise ValueError("--n-halos must be at least 1 when --run-all=0")
    if(not np.isfinite(log_mh_min)) or (not np.isfinite(log_mh_max)):
        raise ValueError("--log-mh-min and --log-mh-max must be finite when --run-all=0")
    if(log_mh_max <= log_mh_min):
        raise ValueError("--log-mh-max must be greater than --log-mh-min when --run-all=0")

allcat.write(
    '#model parameters: p2, p3, lg_cut_off_mass, Mmin, IMBH, fit = '
    + str(p2) + " " + str(p3) + " " + str(lg_cut_off_mass) + " " + str(Mmin) + " " + str(IMBH) + " " + fit + "\n"
)
allcat.write(
    "#haloID | logMh(z=0) | haloID @ form | logMh(tform) | logM*(tform) | "
    "logMgas(tform) | logMcl(tform) | zform | [Fe/H] | rGalaxy (kpc) | "
    "GC radius (pc) | Sigma_h (Msun/pc^2) | M_IMBH_init\n"
)

#initialize all the interpolation tables for use with Schechter function
mc = 10**lg_cut_off_mass
mgc_to_mmax = makeLogMgcToLogMmaxInterpolator(mc, Mmin=Mmin)
alpha = -2.0
ug52 = upper_gamma2_log_mass(log_Mmin, mc)

# First-step GC IMBH seeding: seed exactly once at GC formation, using the
# Eq. (7) cluster radius relation and the GC metallicity after intrinsic
# cluster-to-cluster scatter has been applied.

# Define globular cluster class to store data about each cluster.
class GC :
    def __init__(self, mass, originHaloMass, origin_redshift, metallicity, osm, omgas, is_mpb, idform,
                 gc_radius_pc = 0.0, gc_sigma_h_msun_pc2 = 0.0, imbh_mass_msun = 0.0,
                 branch_id = -1, formation_tree_index = -1, gc_uid = -1) :
        self.mass = mass
        self.originHaloMass = originHaloMass
        self.origin_redshift = origin_redshift
        self.metallicity = metallicity
        self.origin_sm = osm
        self.origin_mgas = omgas
        self.is_mpb = is_mpb
        self.idform = idform
        self.rGalaxy = 0.0
        self.local_rGalaxy = 0.0
        self.gc_radius_pc = gc_radius_pc
        self.gc_sigma_h_msun_pc2 = gc_sigma_h_msun_pc2
        self.imbh_mass_msun = imbh_mass_msun
        self.branch_id = int(branch_id)
        self.formation_tree_index = int(formation_tree_index)
        self.gc_uid = int(gc_uid)
    def assign_rGalaxy(self, radius):
        self.rGalaxy = radius
    def assign_local_rGalaxy(self, radius):
        self.local_rGalaxy = radius


def seed_imbh_properties(cluster_mass, metallicity, fit=DEFAULT_IMBH_FIT):
    Z = 10.0**metallicity
    estimate = estimate_for_gc(cluster_mass, Z, fit=fit)
    gc_radius_pc = float(estimate["r_h_pc"])
    sigma_h_msun_pc2 = float(estimate["sigma_h_msun_pc2"])
    imbh_mass_msun = float(estimate["imbh_mass_msun"])

    check_finite_positive(gc_radius_pc, "IMBH model GC half-mass radius")
    check_finite_positive(sigma_h_msun_pc2, "IMBH model GC half-mass surface density")
    check_finite_non_negative(imbh_mass_msun, "IMBH model seed mass")
    imbh_mass_msun *= IMBH
    check_finite_non_negative(imbh_mass_msun, "Scaled IMBH model seed mass")

    return gc_radius_pc, sigma_h_msun_pc2, imbh_mass_msun

# the accreted baryon fraction normalization as described in CGL18, which is used to cap the total baryonic mass (stars + gas) that can be formed in a halo at a given redshift.
def accreted_baryon_fin_norm(Mh, z):
    beta = 6.0 * (np.log(1.82e3 * np.exp(-3.78)) - 1.0) ** (-1.0 / 15.0) # 5.61
    return (1.0 + (2.0 ** (2.0 / 3.0) - 1.0) * (1.69e10 * np.exp(- 0.63 * z) / Mh / (1 + np.exp((z / beta) ** 15.0))) ** 2.0) ** (- 1.5)

# Calculate gas mass given stellar mass, halo mass, redshift using scaling relations. Double power law for SM-Mg relation, then scale with redshift. Revise if gas fraction exceeds accreted baryon fraction. As described in Choksi, Gnedin, and Li (2018).
def gasMass(SM, Mh, z) :
    check_finite_positive(SM, "stellar mass in gasMass")
    check_finite_positive(Mh, "halo mass in gasMass")
    n_M = 0.19 if SM < 1e9 else 0.33
    log_ratio = np.log10(0.35 * 3 ** 2.7) -  n_M * (np.log10(SM) - 9.0) #log10(Mg/M*)
    if (z <= 2):
        log_ratio += 2.7 * np.log10((1.0 + z) / 3.0) # strong ssfr evolution at z < 2
    elif (z <= 3):
        log_ratio += 1.4 * np.log10((1.0 + z) / 3.0) # weak ssfr evolution at z > 2 (Lilly+)
    else: #fg saturates at z > 3
        log_ratio += 1.4 * np.log10(4.0 / 3.0) # weak ssfr evolution at z > 2
    log_ratio += np.random.normal(0, 0.3)
    Mg = SM * (10 ** log_ratio)
    check_finite_positive(Mg, "unlimited gas mass in gasMass")
    fstar = SM/(fb*Mh)
    fgas = Mg/(fb*Mh)
    fin = accreted_baryon_fin_norm(Mh, z)
    check_finite_positive(fstar, "stellar baryon fraction in gasMass")
    check_finite_positive(fgas, "gas baryon fraction in gasMass")
    check_finite_positive(fin, "accreted baryon fraction in gasMass")

    if(fstar+fgas > fin):
        fgas = fin - fstar if fstar < fin else 0.0
        Mg = fgas*fb*Mh
        check_finite_non_negative(fgas, "limited gas baryon fraction in gasMass")
        check_finite_non_negative(Mg, "limited gas mass in gasMass")
    return Mg

# galaxy stellar mass-metallicity relation (SMMR) as described in Chen&Gnedin2024
def gSMMR(SM, z):
    FeH = 0.3 * np.log10(SM / 1.0e9) - np.log10(1 + z) - 0.5
    return 0.3 if FeH > 0.3 else FeH

import scipy.special as special

def Mr_frac_sersic_inverse(fm, ns):
    # give the mass fraction of total enclosed mass
    # return radius of the exactly location in unit of re
    p = 1.0 - 0.6097/ns + 0.05563/ns/ns
    b = 2.*ns - 1/3. + 0.009876/ns
    ZZ = special.gammaincinv(ns*(3.-p), fm)
    return (ZZ/b)**ns

def gc_sersic_sampling(gc_list, mass_sum, halomass, redshift, re_kpc, re_source, ns):
    """
    sample GC spatial distribution within galactic disk with a Sersic profile
    """
    rgal_min_kpc = 1.0e-3
    rgal_max_kpc = 1.0e4
    rVir = Rv(Mhalo=halomass, z=redshift) # [kpc]

    fallback_outer_kpc = 0.5*rVir
    #fallback_outer_kpc = 0.2*rVir
    if((not np.isfinite(fallback_outer_kpc)) or (fallback_outer_kpc <= 0.0)):
        print(
            "[gc_sersic_sampling] invalid fallback outer radius "
            + f"(rVir={rVir}, halomass={halomass}, z={redshift}); using 1.0 kpc",
            file = sys.stderr,
        )
        fallback_outer_kpc = 1.0
    fallback_outer_kpc = float(np.clip(fallback_outer_kpc, rgal_min_kpc, rgal_max_kpc))

    Re = float(re_kpc)
    if((not np.isfinite(Re)) or (Re <= 0.0)):
        print(
            "[gc_sersic_sampling] invalid Sersic scale radius "
            + f"(source={re_source}, Re={Re}, rVir={rVir}, "
            + f"halomass={halomass}, z={redshift}); "
            + f"using fallback Re={fallback_outer_kpc} kpc",
            file = sys.stderr,
        )
        Re = fallback_outer_kpc
    Re = float(np.clip(Re, rgal_min_kpc, rgal_max_kpc))

    # Sersic profile based on inverse incomplete Gamma function, see eq. (A2) in Terzic & Graham 2005
    if((not np.isfinite(mass_sum)) or (mass_sum <= 0.0)):
        print(
            "[gc_sersic_sampling] invalid total GC mass budget "
            + f"(mass_sum={mass_sum}, halomass={halomass}, z={redshift}); "
            + "placing all GCs at the fallback outer radius",
            file = sys.stderr,
        )
        for gc in gc_list:
            gc.assign_rGalaxy(fallback_outer_kpc)
        return

    m_tot = -0.5*gc_list[0].mass
    for gc in gc_list:
        m_tot += gc.mass
        enclosed_mass_fraction_raw = m_tot/mass_sum
        if((not np.isfinite(enclosed_mass_fraction_raw)) or (enclosed_mass_fraction_raw <= 0.0) or (enclosed_mass_fraction_raw >= 1.0)):
            enclosed_mass_fraction = np.clip(enclosed_mass_fraction_raw, 1.0e-12, 1.0-1.0e-12) if np.isfinite(enclosed_mass_fraction_raw) else 1.0e-12
            print(
                "[gc_sersic_sampling] invalid enclosed Sersic mass fraction "
                + f"(fm={enclosed_mass_fraction_raw}, m_tot={m_tot}, mass_sum={mass_sum}, "
                + f"halomass={halomass}, z={redshift}); clamping to {enclosed_mass_fraction}",
                file = sys.stderr,
            )
        else:
            enclosed_mass_fraction = enclosed_mass_fraction_raw

        rGalaxy = Mr_frac_sersic_inverse(enclosed_mass_fraction, ns) * Re
        if((not np.isfinite(rGalaxy)) or (rGalaxy <= 0.0)):
            print(
                "[gc_sersic_sampling] invalid Sersic-sampled GC radius "
                + f"(rGalaxy={rGalaxy}, fm={enclosed_mass_fraction}, Re={Re}, "
                + f"halomass={halomass}, z={redshift}); using fallback outer radius",
                file = sys.stderr,
            )
            rGalaxy = fallback_outer_kpc
        rGalaxy = float(np.clip(rGalaxy, rgal_min_kpc, rgal_max_kpc))
        gc.assign_local_rGalaxy(rGalaxy)
        gc.assign_rGalaxy(rGalaxy)

    return


def clusterFormation(Mg, halomass, redshift, metallicity, SM, is_mpb, hid, jj, branch_id, formation_tree_index, re_kpc, re_source, fit=DEFAULT_IMBH_FIT) :
    gc_list = []
    if(Mg == 0.0):
        return gc_list
    check_finite_positive(Mg, "gas mass in clusterFormation")
    Mgc = 1.8e-4 * p2 * Mg #total mass of all GCs formed in cluster formation event
    check_finite_positive(Mgc, "GC mass budget in clusterFormation")
    log_Mgc = np.log10(Mgc)
    if(log_Mgc < log_Mmin): #not enough mass to form a single cluster of mass Mmin
        return gc_list
    # First reserve the most massive cluster explicitly, then sample the rest
    # from the Schechter CIMF until the event budget Mgc is exhausted.
    # This mirrors the historical Fortran/IDL logic used in the original model.
    # calculate the cumulative distribution r(<M), and invert it numerically
    log_Mmax = mgc_to_mmax(log_Mgc)
    if(log_Mmax > log_Mgc):
        log_Mmax = log_Mgc
    Mmax = 10**log_Mmax
    mt = np.logspace(log_Mmin, log_Mmax, num = 500)

    maxGC_metallicity = metallicity + np.random.normal(0, 0.3)
    gc_radius_pc, sigma_h_msun_pc2, imbh_mass_msun = seed_imbh_properties(Mmax, maxGC_metallicity, fit=fit)
    maxGC = GC(
        Mmax,
        halomass,
        redshift,
        maxGC_metallicity,
        SM,
        Mg,
        is_mpb,
        hid,
        gc_radius_pc = gc_radius_pc,
        gc_sigma_h_msun_pc2 = sigma_h_msun_pc2,
        imbh_mass_msun = imbh_mass_msun,
        branch_id = branch_id,
        formation_tree_index = formation_tree_index,
    )
    gc_list.append(maxGC)
    mass_sum = Mmax

    ntot = ug52 - upper_gamma2_log_mass(log_Mmax, mc)
    cum = np.array([(ug52 - upper_gamma2_log_mass(np.log10(mv), mc))/ntot for mv in mt])

    r_to_m = interpolate.interp1d(cum, mt)
    mass_sum2 = Mmax
    while(mass_sum < Mgc):
        r = np.random.random()
        M = r_to_m(r)
        if(mass_sum+M > Mgc): #make sure the final cluster drawn doesn't exceed the total mass to be formed. it may produce some clusters below Mmin, but shouldn't really matter (will disrupt)
            M = Mgc-mass_sum
        mass_sum += M
        cluster_metallicity = metallicity + np.random.normal(0, 0.3)
        gc_radius_pc, sigma_h_msun_pc2, imbh_mass_msun = seed_imbh_properties(M, cluster_metallicity, fit=fit)
        cluster = GC(
            M,
            halomass,
            redshift,
            cluster_metallicity,
            SM,
            Mg,
            is_mpb,
            hid,
            gc_radius_pc = gc_radius_pc,
            gc_sigma_h_msun_pc2 = sigma_h_msun_pc2,
            imbh_mass_msun = imbh_mass_msun,
            branch_id = branch_id,
            formation_tree_index = formation_tree_index,
        )
        gc_list.append(cluster)

    # Shuffle before assigning radii so the maxGC does not always
    # inherit the smallest radius purely because it was appended first.
    np.random.shuffle(gc_list)
    # sample spatial distribution of GCs within a Sersic disk
    gc_sersic_sampling(gc_list, mass_sum, halomass, redshift, re_kpc, re_source, ns)
    return gc_list


@dataclass(frozen = True)
class TreeEntry:
    halo_id_z0: int
    path: Path


@dataclass(frozen = True)
class HaloCandidate:
    """Metadata for one non-empty tree eligible for mass-bin selection."""

    tree_entry: TreeEntry
    log_msub_z0: float
    traversal_index: int


def _legacy_tree_entries(tree_dir):
    tree_entries = []
    for path in sorted(tree_dir.iterdir()):
        if(not path.is_file()):
            continue
        if(path.suffix.lower() not in (".txt", ".dat")):
            continue
        try:
            hid = int(path.stem)
        except ValueError:
            continue
        tree_entries.append(TreeEntry(halo_id_z0 = hid, path = path))
    return tree_entries


def _iter_tree_files(tree_dir):
    lookup_path = tree_dir / TREE_LOOKUP_BASENAME
    if(lookup_path.is_file()):
        tree_entries = []
        seen_halo_ids = set()
        with lookup_path.open("r", encoding = "utf-8", newline = "") as handle:
            for row in csv.DictReader(handle):
                try:
                    hid = int(row["halo_id_z0"])
                    basename = row["fixed_tree_basename"].strip()
                except (KeyError, ValueError) as exc:
                    raise RuntimeError("Malformed tree lookup row in " + str(lookup_path) + ": " + str(row)) from exc
                tree_path = tree_dir / basename
                if(not tree_path.is_file()):
                    raise FileNotFoundError("Tree lookup references missing fixed tree: " + str(tree_path))
                if(hid in seen_halo_ids):
                    raise RuntimeError("Duplicate halo_id_z0 " + str(hid) + " in tree lookup " + str(lookup_path))
                seen_halo_ids.add(hid)
                tree_entries.append(TreeEntry(halo_id_z0 = hid, path = tree_path))
        if(len(tree_entries) == 0):
            raise RuntimeError("Tree lookup exists but contains no usable rows: " + str(lookup_path))
        return tree_entries

    tree_entries = _legacy_tree_entries(tree_dir)
    if(len(tree_entries) == 0):
        raise RuntimeError("No usable fixed-tree files were found under " + str(tree_dir))
    return tree_entries


def _select_halos_by_mass_bins(candidates, n_bins, log_mh_min, log_mh_max):
    """Select distinct halo trees using direct matches followed by bin-centre fallback."""

    bin_width = (log_mh_max - log_mh_min) / n_bins
    bin_edges = np.linspace(log_mh_min, log_mh_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    in_range_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.log_msub_z0)
        and log_mh_min <= candidate.log_msub_z0 <= log_mh_max
    ]
    if(len(in_range_candidates) == 0):
        raise RuntimeError(
            "No non-empty fixed-tree halo lies within descendant log-mass interval "
            + f"[{log_mh_min}, {log_mh_max}] for --run-all=0."
        )

    selected_by_bin = [None] * n_bins
    selected_indices = set()
    for candidate in in_range_candidates:
        bin_index = int(np.searchsorted(bin_edges, candidate.log_msub_z0, side="right") - 1)
        if(bin_index == n_bins):
            bin_index = n_bins - 1
        if(selected_by_bin[bin_index] is None):
            selected_by_bin[bin_index] = candidate
            selected_indices.add(candidate.traversal_index)

    fallback_bins = 0
    for bin_index in range(n_bins):
        if(selected_by_bin[bin_index] is not None):
            continue
        available = [
            candidate
            for candidate in in_range_candidates
            if candidate.traversal_index not in selected_indices
        ]
        if(len(available) == 0):
            break
        selected = min(
            available,
            key=lambda candidate: (
                abs(candidate.log_msub_z0 - bin_centres[bin_index]),
                candidate.traversal_index,
            ),
        )
        selected_by_bin[bin_index] = selected
        selected_indices.add(selected.traversal_index)
        fallback_bins += 1

    selected_candidates = sorted(
        [
            candidate
            for candidate in in_range_candidates
            if candidate.traversal_index in selected_indices
        ],
        key=lambda candidate: candidate.traversal_index,
    )
    missing_bins = sum(candidate is None for candidate in selected_by_bin)
    selected_log_masses = [candidate.log_msub_z0 for candidate in selected_candidates]
    return selected_candidates, fallback_bins, missing_bins, selected_log_masses


def loadTree(tree_path):
    log_mh = []
    first_prog_id = []
    subhalo_id = []
    branch_id = []
    redshift = []
    spin_norm = []
    schema = "float, int, int, int, int, float, float, float, float"

    with Path(tree_path).open("r", encoding = "utf-8") as handle:
        for line_no, line in enumerate(handle, start = 1):
            if(line_no == 1):
                continue
            row_text = line.rstrip("\n")
            stripped = row_text.strip()
            if((not stripped) or stripped.startswith("#") or stripped.lower().startswith("logmh")):
                continue
            cols = stripped.split()
            if(len(cols) < 9):
                warnings.warn(
                    "Malformed fixed-tree row in "
                    + str(tree_path)
                    + " at physical line "
                    + str(line_no)
                    + ": found "
                    + str(len(cols))
                    + " column(s), expected at least 9; row="
                    + row_text,
                    RuntimeWarning,
                    stacklevel = 2,
                )
                continue
            if(len(cols) > 9):
                warnings.warn(
                    "Fixed-tree row in "
                    + str(tree_path)
                    + " at physical line "
                    + str(line_no)
                    + " has "
                    + str(len(cols))
                    + " column(s); using the first 9 columns; row="
                    + row_text,
                    RuntimeWarning,
                    stacklevel = 2,
                )
            try:
                parsed = [
                    float(cols[0]),
                    int(cols[1]),
                    int(cols[2]),
                    int(cols[3]),
                    int(cols[4]),
                    float(cols[5]),
                    float(cols[6]),
                    float(cols[7]),
                    float(cols[8]),
                ]
            except ValueError as exc:
                warnings.warn(
                    "Malformed fixed-tree row in "
                    + str(tree_path)
                    + " at physical line "
                    + str(line_no)
                    + ": expected first 9 columns as "
                    + schema
                    + "; parser error="
                    + str(exc)
                    + "; row="
                    + row_text,
                    RuntimeWarning,
                    stacklevel = 2,
                )
                continue
            log_mh.append(parsed[0])
            first_prog_id.append(parsed[1])
            subhalo_id.append(parsed[2])
            branch_id.append(parsed[3])
            redshift.append(parsed[5])
            sx = parsed[6]
            sy = parsed[7]
            sz = parsed[8]
            spin_norm.append(float(np.sqrt(sx * sx + sy * sy + sz * sz)))

    if(len(log_mh) == 0):
        empty_float = np.array([], dtype = float)
        empty_int = np.array([], dtype = int)
        return (
            empty_float,
            empty_int,
            empty_int,
            empty_float,
            empty_float,
            empty_int,
            0.0,
            -1,
        )

    log_mh = np.asarray(log_mh, dtype = float)
    mass_msun = np.power(10.0, log_mh)
    first_prog_id = np.asarray(first_prog_id, dtype = int)
    subhalo_id = np.asarray(subhalo_id, dtype = int)
    branch_id = np.asarray(branch_id, dtype = int)
    redshift = np.asarray(redshift, dtype = float)
    spin_norm = np.asarray(spin_norm, dtype = float)

    mpb_branch_id = fixed_tree_mpb_branch_id(log_mh, branch_id)
    main_mask = branch_id == mpb_branch_id
    if(np.any(main_mask)):
        msub_z0_msun = float(np.max(mass_msun[main_mask]))
    else:
        msub_z0_msun = float(np.max(mass_msun))

    keep = redshift >= 0.0
    return (
        mass_msun[keep],
        first_prog_id[keep],
        subhalo_id[keep],
        redshift[keep],
        spin_norm[keep],
        branch_id[keep],
        msub_z0_msun,
        mpb_branch_id,
    )


num = -1
num_run = 0
tree_entries = _iter_tree_files(treedir)
if(run_all):
    formation_tree_entries = tree_entries
else:
    candidates = []
    for traversal_index, tree_entry in enumerate(tree_entries):
        tree_data = loadTree(tree_entry.path)
        if(len(tree_data[0]) == 0):
            del tree_data
            continue
        log_msub_z0 = float(np.log10(tree_data[6]))
        del tree_data
        if(not np.isfinite(log_msub_z0)) or (log_msub_z0 < log_mh_min) or (log_msub_z0 > log_mh_max):
            continue
        candidates.append(
            HaloCandidate(
                tree_entry = tree_entry,
                log_msub_z0 = log_msub_z0,
                traversal_index = traversal_index,
            )
        )

    selected_candidates, fallback_bins, missing_bins, selected_log_masses = _select_halos_by_mass_bins(
        candidates,
        N,
        log_mh_min,
        log_mh_max,
    )
    selected_log_mh_min = float(np.min(selected_log_masses))
    selected_log_mh_max = float(np.max(selected_log_masses))
    bin_width = (log_mh_max - log_mh_min) / N
    print(
        "HALO_SELECTION_SUMMARY "
        + f"requested_logMh_min={log_mh_min:.10g} "
        + f"requested_logMh_max={log_mh_max:.10g} "
        + f"n_bins={N} "
        + f"bin_width_dex={bin_width:.10g} "
        + f"candidate_halos={len(candidates)} "
        + f"selected_halos={len(selected_candidates)} "
        + f"fallback_bins={fallback_bins} "
        + f"missing_bins={missing_bins} "
        + f"selected_logMh_min={selected_log_mh_min:.10g} "
        + f"selected_logMh_max={selected_log_mh_max:.10g}"
    )
    formation_tree_entries = [candidate.tree_entry for candidate in selected_candidates]

for tree_entry in formation_tree_entries:
    hid_num = int(tree_entry.halo_id_z0)
    tree_path = tree_entry.path
    m, fp, subid, redshifts, jsp, mpi, msub_z0, mpbi = loadTree(tree_path)
    if(len(m) == 0):
        continue

    num_run += 1

    # Go through each halo along the tree and look for events satisfying Rm > p3.
    sm_arr = np.zeros(len(redshifts))
    clusters = []
    for i in range(0, len(m)) : #for each halo in the merger tree
        mass = m[i] #mass of this halo
        fpID = fp[i] #ID of the main progenitor
        jj = jsp[i]

        if(fpID == -1 or len(subid[subid == fpID]) == 0): #then we've reached the first point along this track of the tree
            sm_arr[i] = Mstar_SMHM(Mhalo=mass, z=redshifts[i], scatter=True) #assign a "seed" stellar mass which we will grow self-consistently
            continue

        progIdx = np.where(subid == fpID)[0][0] #identify index of first progenitor in data
        progMass = m[progIdx] #get mass of fprogenitor

        ratio = mass/progMass - 1 #calculate merger ratio, Rm = dMh/Mh
        znow, zbefore = redshifts[i], redshifts[progIdx]
        dt = Redshift2CosmicAge(znow, time_unit="Gyr") - Redshift2CosmicAge(zbefore, time_unit="Gyr")
        ratio = ratio/dt #(dMh/Mh)/dt

        #evolve stellar mass self-consistently as described in CGL18
        sm1 = Mstar_SMHM(Mhalo=mass, z=znow, scatter=False); sm2 = Mstar_SMHM(Mhalo=progMass, z=zbefore, scatter=False)
        dsm = sm1-sm2
        scatter = np.random.normal(0, 0.218 + 0.023 * znow / (1.0 + znow))
        SM = sm_arr[progIdx] + dsm*10**scatter
        if(SM < 0): #only happens in very weird cases at very high redshift
            SM = sm_arr[progIdx]
        sm_arr[i] = SM
        Mg = gasMass(SM, mass, znow)
        if(ratio > p3):  #if merger criterion satisfied
            metallicity = gSMMR(SM, znow)
            is_mpb = mpi[i] == mpbi
            #Re = resolve_birth_re_kpc(halomass_msun = mass, redshift = znow, jsp = jj) # [kpc]
            Re = calcRe(mhalo_1e9msun=mass/1.0e9, t_Gyr=Redshift2CosmicAge(znow, time_unit="Gyr"), j=jj) # [kpc]
            # `subid[i]` records the halo hosting the formation event; later
            # stages use it to mark MPB vs accreted GCs in merged catalogs.
            clusters.extend(clusterFormation(Mg, mass, znow, metallicity, SM, is_mpb, subid[i], jj, mpi[i], i, Re, "Gao+2024", fit=fit))
            continue

    for gc_uid, cluster in enumerate(clusters, start=1):
        cluster.gc_uid = gc_uid

    # All formed GCs are passed to the dynamic evolution stage for survival and
    # deposition decisions.
    GC_mets = np.array([cluster.metallicity for cluster in clusters]); GC_masses = np.array([cluster.mass for cluster in clusters]); GC_log_masses = np.log10(GC_masses); GC_redshifts = np.array([cluster.origin_redshift for cluster in clusters])
    GC_idform = np.array([cluster.idform for cluster in clusters]); GC_mhost_tform = np.array([cluster.originHaloMass for cluster in clusters])
    GC_log_mhost_tform = np.log10(GC_mhost_tform); GC_log_mstar_tform = np.round(np.log10(np.array([cluster.origin_sm for cluster in clusters])), 3)
    GC_log_mgas_tform = np.round(np.log10(np.array([cluster.origin_mgas for cluster in clusters])), 3)
    GC_radius = np.array([cluster.rGalaxy for cluster in clusters])
    GC_gc_radius_pc = np.array([cluster.gc_radius_pc for cluster in clusters])
    GC_sigma_h = np.array([cluster.gc_sigma_h_msun_pc2 for cluster in clusters])
    GC_imbh_mass = np.array([cluster.imbh_mass_msun for cluster in clusters])

    logmsub = np.log10(msub_z0)

    # The host column refers to the descendant z=0 halo.
    for i in range(len(GC_masses)): #all clusters
        allcat.write(
            str(hid_num)
            + " "
            + str(np.round(logmsub,5))
            + " "
            + str(GC_idform[i])
            + " "
            + str(np.round(GC_log_mhost_tform[i],5))
            + " "
            + str(np.round(GC_log_mstar_tform[i],5))
            + " "
            + str(np.round(GC_log_mgas_tform[i],5))
            + " "
            + str(np.round(GC_log_masses[i],5))
            + " "
            + ZFORM_FORMAT.format(float(GC_redshifts[i]))
            + " "
            + str(np.round(GC_mets[i],5))
            + " "
            + str(np.round(GC_radius[i],5))
            + " "
            + str(np.round(GC_gc_radius_pc[i],5))
            + " "
            + str(np.round(GC_sigma_h[i],5))
            + " "
            + str(np.round(GC_imbh_mass[i],5))
            + "\n"
        )
allcat.close()
print("all done!")
