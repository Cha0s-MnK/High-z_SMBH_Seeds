# High-z SMBH Seeds

This repository is the current working branch derived from `/home/subonan/Gao+2024`. It extends the Gao+2024 globular cluster (GC) model toward a "GC to IMBH to high-$z$ SMBHs" workflow and now provides the active Python implementation for GC formation, GC evolution, IMBH seeding, one-`N_S` execution, and figure reproduction.

## New Features

### Python GC-evolution workflow

The original Python-plus-Fortran split has been replaced by an active Python evolution path centred on `src/evo.py`. Relative to `/home/subonan/Gao+2024`, the current workflow always uses the evolving-host background with analytical background-density evaluation and exposes timestep controls, one selected Sérsic index through `--N_S`, the `--DF` dynamical-friction switch, the `--tidal_stripping` continuous-stripping switch, and a redshift-list interface for extra central NSC/BH summary outputs directly through `src/run.py`, while keeping the formation stage tied to the Gao-style tree and GC catalogue logic. All branches use one dynamic branch-import treatment: live child GCs are released at `0.5 Rvir`, and child central BH states plus fixed-bin-1 deposited stellar components are imported at branch merger. The physical simulation itself now always runs to `z=0`, and optional extra redshifts are reconstructed afterwards from the `z=0` evolution outputs. The current pipeline is also easier to inspect and compare because one command now rebuilds formation catalogues, runs halo-by-halo evolution, and writes one flat set of outputs.

### IMBH extension

The main scientific extension beyond Gao+2024 is the IMBH path. `src/config.py` and `src/main.py` add formation-time IMBH seeding, and the formation catalogues now store GC radius, surface density, metallicity, and IMBH seed mass for downstream use. Halo-level summaries also track SMBH-proxy quantities from sunk GC and IMBH channels. When `--Eddington` is positive, it applies only to the stored central BH state after central entry or branch import; IMBHs inside GCs and non-central wandering IMBHs do not accrete. This is still a first bridge from GC evolution to SMBH-oriented diagnostics rather than a full black-hole growth model with accretion and merger physics.

The formation-time IMBH prescription is selected with `--fit`. The default `Rantala+2026` keeps the existing metallicity- and surface-density-dependent estimator, including its current $100\,M_\odot$ IMBH convention. `Vergara+2026conservative` and `Vergara+2026optimistic` use the birth stellar mass of each cluster, namely the `cluster.mass` value drawn from the CIMF before evolution, expressed in $M_\odot$; they do not use metallicity, radius, or surface density in the BH-mass formula. The two fits are:

$$
\log_{10}\left(\frac{M_{\rm BH}}{M_\odot}\right) = -2 + 0.88\log_{10}\left(\frac{M_{\rm cl,\star,birth}}{M_\odot}\right)
$$

for `Vergara+2026conservative`, and

$$
\log_{10}\left(\frac{M_{\rm BH}}{M_\odot}\right) = -0.76 + 0.76\log_{10}\left(\frac{M_{\rm cl,\star,birth}}{M_\odot}\right)
$$

for `Vergara+2026optimistic`. Every GC that is formed by the existing event and CIMF-budget logic receives the selected Vergara estimate. The relation is directly extrapolated for finite positive birth masses: there is no implicit calibration-range gate, clipping, warning, or range-based rejection. GC radius and surface-density columns remain available for diagnostics in every fit, but post-formation stellar mass loss does not recompute or reduce the stored IMBH mass. The finite, non-negative `--IMBH` coefficient is applied once to the selected fit result at formation; its scaled value is then carried into `M_IMBH_init`, evolution, and aggregation without a second multiplication.
The selected fit is recorded in the formation-catalogue comment and `run_metadata.json`; it does not add a numeric output column or change output filenames.

### Improved outputs and analysis support

The output layout uses one selected `N_S` per run: there are no `N_S` subdirectories and no cross-`N_S` aggregation step. The run writes one `finalGCs.dat`, one `depos.dat`, halo-summary tables, a redshift-resolved central NSC/BH summary, and machine-readable run metadata. `src/run.py` can optionally trigger the maintained Choksi+2018 and Kong&Li2026 plot suites through `--plot_Choksi+2018` and `--plot_KongLi2026`. Different Sérsic indices are compared by running separate output directories.

#### `plot/plot_Choksi+2018.py`

This script reproduces the Choksi, Gnedin & Li (2018) figure suite from one finished model output directory. It reads the root-level model products from `--out_dir`, uses the cached observational and supplemental comparison data under `data/Choksi+2018`, and writes its figures to `<out_dir>/_plots_Choksi+2018`. In addition to the local model, it overlays the published `Choksi+2018` supplemental survivor catalogue where that comparison is directly available. It does not accept an `N_S` selector because the output directory already represents one selected run.

#### `plot/plot_Kong&Li2026.py`

This script combines the IMBH seed diagnostics and the redshift-resolved central NSC/BH summaries. It reads the root-level `allcat_s-*.txt`, `finalGCs.dat`, `haloSummaryByZ.csv`, and `depos.dat` products, plus the single `N_S` value recorded in `run_metadata.json` for diagnostics. Fig. 4 uses the same-redshift MPB halo mass stored in that summary, converts it to stellar mass with the project SMHM helper, and does not reconstruct halo mass from the flattened `mpb_from_fixed_trees.csv` table. Fig. 7 adds an individual sunk IMBH versus sunk GC stellar mass diagnostic, including sunk IMBH wanderers at `M_GC_final = 0`. Fig. 8 reproduces the Juodzbalis+2026 QSO1 rotation curve and overlays the z~7 deposited-stellar-plus-central-BH mass profiles. Fig. 9 compares the local nuclear and satellite BH mass function with the Kritos+2025/Greene+2020 reference curves. It writes figures to `<out_dir>/_plots_Kong&Li2026` and does not accept an `N_S` selector.

The plotting helpers are split by responsibility: `plot/load_output.py` handles model-output paths, readers, validation, and derived model tables; `plot/load_obs.py` handles observational cache readers and explicit missing-cache errors; `plot/plot_common.py` contains small plotting-only utilities. Plotting does not download or rebuild observational caches automatically.

## Repository Layout

- `data/`: reference tables used by the model, plus the bundled fixed-tree sample. External corrected tree directories can also be supplied at runtime through `--tree-dir`.
- `data/fixed_trees_large_spin/`: bundled Gao-compatible fixed-tree input set.
- `src/main.py`: GC formation stage based on the Gao/Choksi-style model.
- `src/evo.py`: active Python GC evolution solver.
- `src/config.py`: IMBH fitting functions and shared model configuration used at GC formation.
- `src/schechter_interp.py`: Schechter-sampling support for GC initial masses.
- `src/smhm.py`: stellar-mass-halo-mass helper functions.
- `src/run.py`: end-to-end runner for one formation/evolution pass and optional paper-style plotting.
- `plot/plot_Choksi+2018.py`: Choksi+2018 figure reproduction and comparison script.
- `plot/plot_Kong&Li2026.py`: IMBH seed diagnostics plus redshift-resolved central NSC/BH plot script.
- `plot/load_output.py`: shared model-output path discovery, table readers, validation, and derived plotting tables.
- `plot/load_obs.py`: shared observational cache readers with no automatic downloads or cache rebuilding.
- `plot/plot_common.py`: shared plotting-only style, output-directory, figure-IO, binning, and axis helpers.
- `papers/`: method papers and reference PDFs used for the project.
- `plots/`: project figures and plotting artifacts kept in the repository.
- `tex/`: manuscript and note material.

## Typical Run

Use a separate output directory for each `--fit` choice; the three example runs below therefore write to three different directories.

```bash
python ~/GitHub/src/run.py --help
nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark/fixed_trees_large_spin_dark \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50_Eddington0_IMBH1_Min1e5_p2-7.0_p3-0.5_Rantala+2026 \
  --Eddington 0 --fit "Rantala+2026" --Mmin 1.0e5 --IMBH 1.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 7.0 --p3 0.5 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 128 --log-mh-min 10.0 --log-mh-max 13.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 32 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50_Eddington0_IMBH1_Min1e5_p2-7.0_p3-0.5_Rantala+2026.log 2>&1 &

nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark_Small/fixed_trees \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50_Eddington0_IMBH1_Min3e4_p2-14.0_p3-0.7_Rantala+2026 \
  --Eddington 0 --fit "Rantala+2026" --Mmin 3.0e4 --IMBH 1.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 14.0 --p3 0.7 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 128 --log-mh-min 10.0 --log-mh-max 13.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 32 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50_Eddington0_IMBH1_Min3e4_p2-14.0_p3-0.7_Rantala+2026.log 2>&1 &

nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark_Small/fixed_trees \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50_Eddington0.25_IMBH1_Min3e4_p2-14.0_p3-0.7_Rantala+2026 \
  --Eddington 0.25 --fit "Rantala+2026" --Mmin 3.0e4 --IMBH 1.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 14.0 --p3 0.7 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 128 --log-mh-min 10.0 --log-mh-max 13.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 32 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50_Eddington0.25_IMBH1_Min3e4_p2-14.0_p3-0.7_Rantala+2026.log 2>&1 &

nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark_Small/fixed_trees \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50+100_Eddington0.25_IMBH1_Min3e4_p2-14.0_p3-0.7_Rantala+2026 \
  --Eddington 0.25 --fit "Rantala+2026" --Mmin 3.0e4 --IMBH 1.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 14.0 --p3 0.7 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 256 --log-mh-min 10.0 --log-mh-max 14.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 64 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50+100_Eddington0.25_IMBH1_Min3e4_p2-14.0_p3-0.7_Rantala+2026.log 2>&1 &

nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark/fixed_trees_large_spin_dark \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50_Eddington0.25_IMBH3_Min1e5_p2-7.0_p3-0.5_Rantala+2026 \
  --Eddington 0.25 --fit "Rantala+2026" --Mmin 1.0e5 --IMBH 10.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 7.0 --p3 0.5 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 128 --log-mh-min 10.0 --log-mh-max 13.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 32 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50_Eddington0.25_IMBH3_Min1e5_p2-7.0_p3-0.5_Rantala+2026.log 2>&1 &

nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark/fixed_trees_large_spin_dark \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50+100_Eddington0_IMBH3_Min1e5_p2-7.0_p3-0.5_Rantala+2026 \
  --Eddington 0 --fit "Rantala+2026" --Mmin 1.0e5 --IMBH 1.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 7.0 --p3 0.5 --ts-m 0.2 --ts-r 0.2 \
  --run-all 1 --n-halos 32768 --log-mh-min 10.0 --log-mh-max 15.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 80 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50+100_Eddington0_IMBH3_Min1e5_p2-7.0_p3-0.5_Rantala+2026.log 2>&1 &

nohup python3 /home/subonan/GitHub/src/run.py \
  --tree-dir /lingshan/disk3/subonan/TNG50+100-1-Dark/fixed_trees_large_spin_dark \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/TNG50+100_Eddington0.25_IMBH3_Min1e5_p2-7.0_p3-0.5_Rantala+2026 \
  --Eddington 0.25 --fit "Rantala+2026" --Mmin 1.0e5 --IMBH 1.0 --lg_cut-off_mass 7.0 --N_S 2.0 --p2 7.0 --p3 0.5 --ts-m 0.2 --ts-r 0.2 \
  --run-all 1 --n-halos 32768 --log-mh-min 10.0 --log-mh-max 15.0 \
  --out_z '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0'\
  --main_jobs 80 --satellite_jobs 1 --plot_Choksi+2018 --plot_KongLi2026 \
  > ~/TNG50+100_Eddington0.25_IMBH3_Min1e5_p2-7.0_p3-0.5_Rantala+2026.log 2>&1 &
```

Prefer running from the repository root because the project path contains spaces and the `src/run.py` entry point is the least error-prone form.
`src/run.py` now always uses the bundled repository `src/` and `data/` layout and checks those paths automatically at startup.

- `--run-all 1` processes the full tree set, while `--run-all 0` activates the mass window and `--n-halos` selection.
- `--N_S` selects the one positive, dimensionless Sérsic index used by this run; the default is `2.0`.
- `--main_jobs` controls parallel evolution of different descendant halos within this one `N_S` run. There is no parallel `N_S` job layer.
- `--satellite_jobs` controls independent ready satellite-branch evolution within one descendant halo; child branches must finish before their recipient branch runs. Neither worker setting parallelises formation or individual GCs.
- The two worker limits are independent. The runtime message reports `main_jobs = M`, `satellite_jobs = S`, and `M * S = P` possible inner satellite-worker slots, together with the possible process footprint including the outer halo workers and coordinator. No automatic global cap is applied.
- GC evolution now always uses the evolving host-halo background.
- The physical simulation now always runs to `z=0`; `--out_z` only controls extra halo-level central NSC/BH summaries reconstructed at earlier redshifts.
- `z=0` is always included automatically in the redshift-resolved central NSC/BH outputs.
- `--plot_Choksi+2018` writes Choksi-style figures under `<output>/_plots_Choksi+2018/`.
- `--plot_KongLi2026` writes the IMBH seed diagnostics and redshift-resolved central NSC/BH figures under `<output>/_plots_Kong&Li2026/`.
- If an explicitly requested plotter fails, the simulation keeps its data products and emits a warning.
- Temporary work directories are created under the system temp area and removed automatically at the end of the run.
- The output directory is flat: it contains one `allcat_s-0_p2-..._p3-....txt`, `finalGCs.dat`, `depos.dat`, `haloSummary.csv`, `haloSummaryByZ.csv`, `python_evo_summary.csv`, and `run_metadata.json`. No `ns*/` directories are created.
- `--clear-output 0` refuses to run in a non-empty output directory; modes `1` and `2` retain their confirmation and clearing behaviour.

```bash
python3 ~/GitHub/plot/plot_Choksi+2018.py --out_dir /lingshan/disk3/subonan/_outputs/TNG50+100_Eddington0.3_ll_Mc7
python3 -u ~/GitHub/plot/plot_Kong\&Li2026.py --out_dir /lingshan/disk3/subonan/_outputs/TNG50+100_Eddington0.3_Mc7 2>&1 | tee /tmp/plot_KongLi2026.log
```

New style:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name ".DS_Store" -delete
rsync -av "/Users/kcz0324/Documents/THU_Mac/High-z SMBH Seeds/GitHub" lingshan-subonan:/home/subonan/
```

## Main Run Parameters

The active workflow no longer uses the legacy Gao `input.txt` interface. The main controls now live in `src/run.py`.

### Path and output control

- `--tree-dir`: optional fixed-tree input directory; if omitted, the runner uses the bundled `data/fixed_trees_large_spin` inside this repository.
- Output-directory flag: destination directory for the whole run.
- `--clear-output`: `0` requires an empty output directory; `1` asks before clearing; `2` clears without asking before writing fresh results.

### Formation-model parameters

- `--p2`: GC formation-efficiency normalization in `M_GC = 3e-5 * p2 * M_gas / f_b`.
- `--p3`: threshold in `((Delta M_h / M_h) / Delta t)` above which a formation event is triggered.
- MPB-only switch: if `1`, form GCs only on the main progenitor branch; if `0`, include all retained branches in the fixed tree.
- `--lg_cut-off_mass`: `log10(M_c / Msun)` for the Schechter cutoff mass in the GC initial-mass function.
- `--fit`: formation-time IMBH mass prescription. The default is `Rantala+2026`; the other allowed values are `Vergara+2026conservative` and `Vergara+2026optimistic`. The Rantala choice retains the existing metallicity/surface-density calculation, while both Vergara choices use only the birth stellar cluster mass in the relations above.
- `--Mmin`: minimum initial GC mass in linear $M_\odot$ (not `log10`), default $10^5\,M_\odot$; it must be finite, positive, and less than $10^6\,M_\odot$. The same value sets the lower endpoint of the CIMF and the event-eligibility check. Ordinary CIMF draws and the reserved maximum cluster use this lower endpoint, but the final budget-correction residual may be below `Mmin` so that the event budget remains conserved.
- `--metal`: stellar mass-metallicity relation used at GC formation; choices are `Choksi+2018` and `Chen&Gnedin2024`.
- `--accreted_baryon`: accreted-baryon fraction limiter used for the cold-gas mass; choices are `Muratov&Gnedin2010` and `Chen&Gnedin2023`.
- `--eff_rad`: effective-radius model used for both GC birth radii and the analytical stellar-background radius. `Gao+2024` keeps the current spin-based `R_e` control. `empirical` uses the star-forming galaxy size-mass-redshift relation from `papers/gc_birth_radius_methods.pdf`, with stellar mass supplied by the existing SMHM relation. `catalogue` uses the matched full-physics SFR-concentration sidecar and falls back to the empirical relation for missing, unresolved, zero-SFR, or out-of-domain rows.
- `--eff_rad_catalogue`: optional sidecar CSV used by `--eff_rad catalogue`. Build the default catalogue before production catalogue-mode runs with:
  ```bash
  python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/6_build_eff_radius_catalogue.py
  ```
- `--run-all`: if `1`, process all halos in the selected tree directory.
- `--log-mh-min`: lower bound on descendant `z=0` host-halo `log10(M_h)` when `--run-all 0`.
- `--log-mh-max`: upper bound on descendant `z=0` host-halo `log10(M_h)` when `--run-all 0`.
- `--n-halos`: maximum number of halos to keep when `--run-all 0`.

### Evolution and scan parameters

The evolution solver now always uses the evolving-host background implementation in `src/evo.py`, with analytical background-density evaluation and no lookup-table mode.

- `--ts-m`: adaptive mass-loss timestep factor.
- `--ts-r`: adaptive orbital-decay timestep factor.
- `--DF`: if `1`, enable dynamical-friction orbital decay; if `0`, disable the radial-inspiral term while leaving stellar evolution, tidal stripping, and tidal tearing active.
- `--tidal_stripping`: continuous tidal-stripping prescription. `Fragione+2019` keeps the current local-orbit rate; `Choksi+2018` uses a fixed `P = 0.5` Choksi-style disruption/stripping rate. Direct tidal tearing and stellar evolution are unchanged.
- `--out_z`: comma-separated extra redshifts for halo-level central NSC/BH summaries. The simulation itself still runs to `z=0`, `z=0` is always included automatically, and halo selection remains tied to the descendant `z=0` host.
- `--IMBH`: finite, non-negative, dimensionless coefficient for the IMBH seed mass, default `1.0`. The coefficient is applied once in `src/main.py` to the formation-time estimator result: `1` preserves the estimate, `0` sets every formation seed to zero while retaining GC formation and its GC radius and surface density, and values above `1` amplify the seed. It does not change the GC radius or surface density and is not applied again during evolution or aggregation.
- `--Eddington`: dimensionless Eddington ratio for uncapped growth of the stored central BH state only; IMBHs inside GCs and non-central wandering IMBHs remain non-accreting.
- Branch-import GC treatment is unconditional: every formed GC receives its local Sérsic placement, every branch uses dynamic evolution, live child GCs are released at `0.5 Rvir`, and child central BH states plus fixed-bin-1 deposited stellar components are imported at branch merger.
- `--N_S`: one positive, dimensionless Sérsic index for the run; default `2.0`. Compare different values with separate output directories.
- `--main_jobs`: positive integer number of concurrent descendant-halo evolution workers; default `1`.
- `--satellite_jobs`: positive integer number of concurrent independent satellite-branch evolution workers within one descendant halo; default `1`. Child branches finish before their recipient branch runs.
- `--plot_Choksi+2018`: run `plot/plot_Choksi+2018.py` automatically after the simulation.
- `--plot_KongLi2026`: run `plot/plot_Kong&Li2026.py` automatically after the simulation.
- Automatic plotting is opt-in. If a requested plotter fails, the simulation retains its data products and emits a warning.
- `--quiet`: reduce progress logging.

### Internal `evo.py` tunables

These are not exposed as `src/run.py` flags, but they still define the evolution grid and deposited-mass bookkeeping:

- `T_UNIVERSE_GYR = 13.799`: Universe-age constant used by the approximate cosmic-time and redshift conversions.
- `dt_max = 0.01` and `t_div = 100`: cap the adaptive step size and define the coarse cosmic-time blocks.
- `binnub = 100`, `MIN_RAD_PC = 1 pc`, and `NSC_RAD_PC = 6 pc`: set the deposited-profile radial binning, the fixed 0-1 pc sink/inner bin edge, and the public stellar NSC aperture sampled from the deposit profile.
- `t_limit = 1.0e-2`: sets the minimum adaptive timescale floor.

## Figure Reproduction

### `plot/plot_Choksi+2018.py`

```bash
python3 plot/plot_Choksi+2018.py \
  --out_dir <out_dir>
```

- `--out_dir`: one finished model output directory containing the root allcat template, `finalGCs.dat`, `haloSummary.csv`, `mpb_from_fixed_trees.csv`, and `run_metadata.json`.
- `--figures`: optional comma-separated subset, for example `1,3,6`.
- `--final-z`: optional override for the final redshift if you want the age-based panels to ignore `run_metadata.json`.

This script writes the Choksi-style figure PDFs under `<out_dir>/_plots_Choksi+2018/`.

### `plot/plot_Kong&Li2026.py`

```bash
python3 plot/plot_Kong\&Li2026.py \
  --out_dir <out_dir>
```

- `--out_dir`: one finished model output directory containing the root `allcat_s-*.txt`, `finalGCs.dat`, `haloSummaryByZ.csv`, `depos.dat`, and `run_metadata.json` products.
- `--mass-bin-width-dex`: optional log halo-mass bin width for the mean and standard-deviation tracks.

This script writes the Kong&Li2026 figures under `<out_dir>/_plots_Kong&Li2026/`. It reads the one selected `N_S` from `run_metadata.json` only for diagnostics; it does not accept an `N_S` selector.

For both maintained plotting scripts, figure products always go to `<out_dir>/_plots_<suite>/`. Required observational cache files must already exist under `data/`; missing cache files now raise an explicit error instead of triggering a download or rebuild during plotting.

## Output Schema

Each simulation output directory represents exactly one positive `N_S` value and is flat. The runner does not create `ns*/` directories and does not merge outputs from multiple `N_S` values. Temporary formation/evolution workspaces are created outside the output directory and removed after a successful run.

### Persistent files

#### `allcat_s-0_p2-..._p3-....txt`

Formation catalogue for this run. Each row is one formed GC, and the row order matches `finalGCs.dat`.

Columns:
- `hid_z0`, `logMh_z0`, `logMstar_z0`
- `logMh_form`, `logMstar_form`, `logM_form`
- `zform`, `feh`, `isMPB`
- `subfind_form`, `snap_form`
- `r_galaxy_kpc`, `gc_radius_pc`, `sigma_h_msun_pc2`, `imbh_mass_msun`

#### `mpb_from_fixed_trees.csv`

Compact halo-history table rebuilt from the selected fixed-tree directory. It supports halo-history diagnostics and redshift-matched halo masses.

Columns:
- `subhalo_id_z0`
- `SnapNum`
- `Redshift`
- `logMh_msun_h`
- `SubhaloSpin_x`, `SubhaloSpin_y`, `SubhaloSpin_z`

#### `python_evo_summary.csv`

Compact per-GC summary for this one run, useful for quick QA without rereading `finalGCs.dat`.

Columns:
- `hid_z0`
- `status`
- `M_GC_final`
- `M_IMBH_init`
- `M_IMBH_final`
- `r_final_kpc`

Status codes:
- `1`: alive at the final simulated epoch (`z=0` for runs produced by `src/run.py`)
- `-1`: exhausted to zero mass
- `-2`: tidally torn apart
- `-3`: sunk into the galaxy center
- `-4`: non-central IMBH wanderer at the final simulated epoch
- `-5`: IMBH wanderer sunk into the galaxy center

#### `finalGCs.dat`

Final-GC table for this run. Each row corresponds to one GC from one halo.

Columns:
- `halo_id_z0`
- `gc_index_halo`
- `status`
- `M_GC_final`
- `m_init_msun`
- `lookback_time_final_gyr`
- `lookback_time_init_gyr`
- `r_final_kpc`
- `r_init_kpc`
- `gc_radius_pc`
- `sigma_h_msun_pc2`
- `feh`
- `M_IMBH_init`
- `M_IMBH_final`

`M_GC_final` is the final bound stellar mass outside the BH.

#### `depos.dat`

Deposited-mass profile table for this run.
`depos` records mass lost through external GC evolution channels and terminal stellar mass deposited when an object reaches the fixed 1 pc sink. The first radial bin is always `[0, 1.0e-3] kpc`, and public `M_NSC` is sampled from `m_star_with_evo_msun` inside `NSC_RAD_PC = 6 pc`.

Columns:
- `halo_id_z0`
- `lookback_time_gyr`
- `bin_index`
- `r_inner_kpc`
- `r_outer_kpc`
- `m_depo_total_msun`
- `m_star_no_evo_msun`
- `m_star_with_evo_msun`

#### `haloSummary.csv`

Halo-level summary for this run, including status counts, total GC masses, and SMBH-proxy quantities built from sunk GC and IMBH channels. It contains no `N_S` column; the selected value is stored only in `run_metadata.json`.

Columns:
- `hid_z0`
- `logMh_z0`
- `n_gc_total`
- `n_alive`
- `n_wanderer`
- `n_exhausted`
- `n_torn`
- `n_sunk_gc`
- `n_sunk_wanderer`
- `n_sunk`
- `m_gc_init_total_msun`
- `m_gc_final_total_msun`
- `M_IMBH_init_tot`
- `M_IMBH_final_tot`
- `M_NSC`
- `M_SMBH_init`
- `M_SMBH_final`

`M_IMBH_final_tot` is the `z = 0` total BH inventory: stored central BH mass plus non-sunk non-central IMBH masses.
`M_NSC` is the evolved deposited stellar mass sampled inside 6 pc, not a separate 1 pc sunk-stellar column.

#### `haloSummaryByZ.csv`

Long-format halo-level central NSC/BH summary for this run and all requested output redshifts. Each row corresponds to one `(hid_z0, z_out)` combination. It contains no `N_S` column.

Columns:
- `hid_z0`
- `z_out`
- `lookback_to_z0_gyr`
- `halo_mass_available`
- `logMh_z_msun`
- `M_NSC`
- `M_SMBH_init`
- `M_SMBH_final`
- `z_depos_sampled`
- `lookback_depos_sampled_gyr`
- `depos_time_match_delta_gyr`

`logMh_z_msun` is the MPB halo mass at `z_out`, interpolated in linear halo mass versus cosmic time using the same monotonic MPB block convention as `src/evo.py`. `halo_mass_available` is `0` and `logMh_z_msun` is `NaN` when the requested redshift lies outside the available MPB history for that halo.
`haloSummaryByZ` samples `M_NSC` from the closest deposited-profile time block, preferring the earlier cosmic time on ties; the three deposit diagnostics record that sampled block. Non-central IMBH inventories are not redshift-resolved in this table; use `M_IMBH_final_tot` in `haloSummary` for the `z = 0` total BH inventory.

#### `run_metadata.json`

Machine-readable record of the main run configuration used to build the output directory.

Keys surfaced in the README:
- `final_redshift`
- `out_z`
- `output_redshifts`
- `ts_m`
- `ts_r`
- `DF`
- `tidal_stripping`
- `p2`
- `p3`
- `lg_cut_off_mass`
- `Mmin`: linear minimum initial GC mass in $M_\odot$ used for the CIMF lower endpoint and the event-budget eligibility check; the default is $10^5\,M_\odot$ and the accepted range is finite, positive, and below $10^6\,M_\odot$.
- `fit`: exact formation-time IMBH prescription selected by `--fit`; the default is `Rantala+2026`, with `Vergara+2026conservative` and `Vergara+2026optimistic` as the alternatives.
- `IMBH`: finite, non-negative, dimensionless coefficient applied once to the formation-time IMBH estimator result; the default is `1.0`.
- `metal`
- `accreted_baryon`
- `eff_rad`
- `eff_rad_catalogue`
- `eff_rad_catalogue_fallback_policy`
- `mpb_only`
- `run_all`
- `log_mh_min`
- `log_mh_max`
- `n_halos`
- `N_S`: the one positive, dimensionless Sérsic index used by this output directory. This is the only published record of the selected `N_S`; no Summary CSV repeats it.
- `main_jobs`: validated descendant-halo worker count used for this run.
- `satellite_jobs`: validated satellite-branch worker count used for this run.

### Plot outputs

When the maintained plot scripts are run, they write:

- `_plots_Choksi+2018/Fig.XX_*.pdf`: Choksi+2018 suite from `plot/plot_Choksi+2018.py`.
- `_plots_Kong&Li2026/Fig.XX_*.pdf`: IMBH seed diagnostics, redshift-resolved central NSC/BH summaries, the QSO1 z~7 rotation-curve comparison, and the local BH mass-function comparison from `plot/plot_Kong&Li2026.py`.

## Install McLuster

https://github.com/lwang-astro/mcluster

### Copy the repository and Build the basic version

On a Linux/HPC machine with GCC:

```bash
cd McLuster_Wang+2019/
make mcluster
./mcluster -h
```

### Case A: cored NSC

Use an EFF/Nuker-like profile with inner slope 0:

```bash
cd _ic/
mcluster -N 100000 -P 3 -r 1.0 -c 10.0 -g 4.0 -g 0.0 -g 2.0 -Q 0.5 -C 3 -o NSCcore -f 0 -u 1
python add2BHs.py NSCcore.txt NSCcoreM1e3M1e2r1.txt --m1 1.0e3 --m2 1.0e2 --r2 1.0
python add2BHs.py NSCcore.txt NSCcoreM1e3M1e3r1.txt --m1 1.0e3 --m2 1.0e3 --r2 1.0
python add2BHs.py NSCcore.txt NSCcoreM1e3M1e2r2.txt --m1 1.0e3 --m2 1.0e2 --r2 2.0
python add2BHs.py NSCcore.txt NSCcoreM1e4M1e2r1.txt --m1 1.0e4 --m2 1.0e2 --r2 1.0
```

### Case B: Cuspy NSC

```bash
mcluster -N 100000 -P 3 -r 1.0 -c 10.0 -g 4.0 -g 1.0 -g 2.0 -Q 0.5 -C 3 -o NSCcusp -f 0 -u 1
python add2BHs.py NSCcusp.txt NSCcuspM1e3M1e2r1.txt --m1 1.0e3 --m2 1.0e2 --r2 1.0
```

### Add 2 BHs

`_ic/add2BHs.py`
