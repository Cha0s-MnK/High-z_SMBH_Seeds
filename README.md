# High-z SMBH Seeds

This repository is the current working branch derived from `/home/subonan/Gao+2024`. It extends the Gao+2024 globular cluster (GC) model toward a "GC to IMBH to high-$z$ SMBHs" workflow and now provides the active Python implementation for GC formation, GC evolution, IMBH seeding, batch execution, and figure reproduction.

## New Features

### External Illustris-1-Dark tree workflow

Compared with the original Gao+2024 repository, this project now has an explicit external tree pipeline under `/lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark` for building Gao-compatible fixed trees before any GC physics is run. The maintained workflow is now a mixed-suite pipeline: `1_select_targets.py` samples one halo from each non-empty log halo-mass bin, controlled by `--max_num_halo`, `--min_halo_mass`, and `--max_halo_mass`. If the requested lower bound is below `10^11 Msun`, bins below `10^11 Msun` are sampled from `TNG50-1-Dark`, while bins at and above `10^11 Msun` are sampled from `Illustris-1-Dark`. The selector uses a fixed internal random seed, so repeated runs with the same cached group catalogs are deterministic. In practice, this means the model no longer depends on one bundled fixed-tree sample: `my/run.py` can ingest any corrected external tree directory through `--tree-dir`, for example `/lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/data/fixed_trees_large_spin_dark`.

The maintained external workflow is:

```bash
python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/1_select_targets.py \
  --max_num_halo 512 --min_halo_mass 8.0 --max_halo_mass 14.65
python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/2_download_full_trees.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/3_convert_full_trees_to_fixed_dat.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/4_validate_fixed_trees.py
```

Current storage layout under `/lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark`:

- `data/groupcat_fields_illustris1_dark/`: cached `Illustris-1-Dark` z=0 `Group_M_Mean200` and `GroupFirstSub` cutouts used by the selector.
- `data/groupcat_fields_tng50_1_dark/`: cached `TNG50-1-Dark` z=0 `Group_M_Mean200` and `GroupFirstSub` cutouts used by the selector.
- `data/sublink_full_dark/`: raw downloaded full SubLink subtree HDF5 files, with suite-prefixed basenames such as `illustris1_dark_sublink_full_subhalo_*.hdf5` and `tng50_1_dark_sublink_full_subhalo_*.hdf5`.
- `data/fixed_trees_large_spin_dark/`: corrected Gao-compatible fixed-tree `.dat` files plus conversion and validation metadata.
- `full_tree_download_summary.json` and `full_tree_download_failures.json`: downloader summary logs written directly in the parent directory, not in a separate `logs/` subdirectory.

Script roles:

- `1_select_targets.py`: resolves the required suite(s), caches the z=0 group-catalog fields, writes per-suite `SnapNum -> redshift` lookup tables, and builds the suite-aware manifest.
- `2_download_full_trees.py`: reads one manifest row at a time, uses the saved per-row `subhalo_url_z0`, and downloads one raw full subtree HDF5 file per selected halo.
- `3_convert_full_trees_to_fixed_dat.py`: reads the suite-specific `snaps2redshifts_*.txt` files, applies the existing branch-correction logic, and writes corrected fixed-tree `.dat` files.
- `4_validate_fixed_trees.py`: checks the converted `.dat` files for schema consistency and basic correction invariants before they are used by the GC model.

Main workflow outputs and their meaning:

- `data/target_manifest_dark.csv`: the main suite-aware manifest consumed by steps 2-4; each row records `simulation`, `simulation_key`, `subhalo_url_z0`, and suite-prefixed raw/fixed basenames.
- `data/halo_selection_labels_dark.csv`: a lighter selection table carrying the chosen halo IDs, suite keys, and the mass-bin boundaries that produced each selection.
- `data/targets_z0_dark.json`: machine-readable selector metadata, including the suite list, selection criteria, counts, and the saved records.
- `data/selected_halos_z0_dark.txt`: one `(simulation_key, halo_id_z0)` pair per selected row.
- `data/selected_subhalos_z0_dark.txt`: one `(simulation_key, subhalo_id_z0)` pair per selected row.
- `data/snaps2redshifts_illustris1_dark.txt` and `data/snaps2redshifts_tng50_1_dark.txt`: per-suite snapshot-to-redshift lookup tables used by the converter.
- `data/fixed_trees_large_spin_dark/id_lookup_large_dark.csv` and `data/fixed_trees_large_spin_dark/id_lookup_large_dark.txt`: lookup files that map the manifest ordering to the raw and converted filenames.
- `data/fixed_trees_large_spin_dark/conversion_summary.json`: machine-readable per-file conversion summary, including row counts after prefiltering and after branch correction.
- `data/fixed_trees_large_spin_dark/validation_report.json` and `data/fixed_trees_large_spin_dark/validation_errors.txt`: validation summary plus any detected schema or branch-invariant failures.

`--limit N` in steps 2-4 now means the first `N` rows of the saved manifest in bin order, not the top `N` most massive halos. When the requested range crosses `10^11 Msun`, the low-mass `TNG50-1-Dark` bins appear first, followed by the higher-mass `Illustris-1-Dark` bins.

### Python GC-evolution workflow

The original Python-plus-Fortran split has been replaced by an active Python evolution path centred on `src/evo.py`. Relative to `/home/subonan/Gao+2024`, the current workflow always uses the evolving-host background with analytical background-density evaluation and exposes timestep controls, the Sersic-index scan, the `--DF` dynamical-friction switch, the `--tidal_stripping` continuous-stripping switch, and a redshift-list interface for extra sunk-BH summary outputs directly through `my/run.py`, while keeping the formation stage tied to the Gao-style tree and GC catalogue logic. The physical simulation itself now always runs to `z=0`, and optional extra redshifts are reconstructed afterwards from the `z=0` evolution outputs. The current pipeline is also easier to inspect and compare because one command now rebuilds formation catalogues, runs halo-by-halo evolution, and merges outputs.

### IMBH extension

The main scientific extension beyond Gao+2024 is the IMBH path. `src/IMBH.py` adds formation-time IMBH seeding tied to GC structural properties, and the formation catalogs now store GC radius, surface density, metallicity, and IMBH seed mass for downstream use. Halo-level summaries also track SMBH-proxy quantities from sunk GC and IMBH channels. When `--Eddington` is positive, it applies only to the stored central BH state after central entry or branch import; IMBHs inside GCs and non-central wandering IMBHs do not accrete. This is still a first bridge from GC evolution to SMBH-oriented diagnostics rather than a full black-hole growth model with accretion and merger physics.

### Improved outputs and analysis support

The output layout is now organized by `N_s`, with merged `finalGCs` and `depos` products, halo-summary tables, a new redshift-resolved halo-level sunk-BH summary, machine-readable run metadata, and four separate paper-style plotting entry points. `my/run.py` can now trigger these plot suites directly after the simulation through `--plot_Gao+2024`, `--plot_Choksi+2018`, `--plot_Neumayer+2020`, and `--plot_Kong+2026`. Compared with the original Gao layout, the emphasis here is on a cleaner batch workflow and outputs that are easier to audit, compare, and reuse in later SMBH-focused analysis.

#### `plot/plot_Gao+2024.py`

This is the maintained Gao+2024 reproduction script. It reads the top-level `allcat_s-...txt` template, resolves the per-`N_s` `ns*/` catalogs automatically, uses `mpb_from_fixed_trees.csv` for halo-history diagnostics, and writes its figures to `<output>/_plots_Gao+2024`. When `my/run.py` is given `--plot_Gao+2024`, it forwards the full `N_s` list, the top-level allcat template, and the output directory to this script.

#### `plot/plot_Choksi+2018.py`

This script reproduces the Choksi, Gnedin & Li (2018) figure suite from one finished model output directory. It reads the local run products from `--out_dir`, uses the cached observational and supplemental comparison data under `data/Choksi+2018`, and writes its figures to `<output>/_plots_Choksi+2018`. In addition to the local model, it now overlays the published `Choksi+2018` supplemental survivor catalog where that comparison is directly available. When `my/run.py` is given `--plot_Choksi+2018`, it automatically uses `N_s = 2.0` if that value is present in the run, otherwise it uses the first requested `N_s`.

#### `plot/plot_Neumayer+2020.py`

This script builds the Neumayer et al. (2020) NSC comparison from one finished model output directory. It reads the deposited-mass profiles, final GC products, halo summaries, and cached observational compilations under `data/Neumayer+2020`, and writes its figures to `<output>/_plots_Neumayer+2020`. It requires split-style full-physics counterpart data for the early/late model split: `<out_dir>/run_metadata.json` identifies the tree-data parent, `<out_dir>/halo_tree_lookup.csv` connects selected output haloes to that tree set, `<tree_data_parent>/full_physics_counterparts_z0.csv` supplies counterpart properties and matching, and `<tree_data_parent>/neumayer2020_fig3_divider.json` supplies the late/early colour divider. When `my/run.py` is given `--plot_Neumayer+2020`, it uses the same automatic `N_s` choice as the Choksi plot suite.

#### `plot/plot_Kong+2026.py`

This script now combines the IMBH seed diagnostics and the redshift-resolved sunk-BH summaries. It reads one per-`N_s` `allcat_ns*.txt` formation catalogue for the initial cluster mass-radius and surface-density-metallicity IMBH plots, then reads `haloSummaryByZ_ns*.csv` for the sunk-BH tracks. Fig. 4 uses the same-redshift MPB halo mass stored in that summary, converts it to stellar mass with the project SMHM helper, and does not reconstruct halo mass from the flattened `mpb_from_fixed_trees.csv` table. It writes six figures to `<output>/_plots_Kong+2026`. When `my/run.py` is given `--plot_Kong+2026`, it uses the same automatic `N_s` choice as the Choksi and Neumayer plot suites.

## Repository Layout

- `data/`: reference tables used by the model, plus the bundled fixed-tree sample. External corrected tree directories can also be supplied at runtime through `--tree-dir`.
- `data/fixed_trees_large_spin/`: bundled Gao-compatible fixed-tree input set.
- `src/main_spatial.py`: GC formation stage based on the Gao/Choksi-style model.
- `src/evo.py`: active Python GC evolution solver.
- `src/IMBH.py`: IMBH seeding module used at GC formation.
- `src/schechter_interp.py`: Schechter-sampling support for GC initial masses.
- `src/smhm.py`: stellar-mass-halo-mass helper functions.
- `my/run.py`: end-to-end batch runner for formation, evolution, merging, and optional paper-style plotting.
- `plot/plot_Gao+2024.py`: Gao+2024 figure reproduction script for the current output layout.
- `plot/plot_Choksi+2018.py`: Choksi+2018 figure reproduction and comparison script.
- `plot/plot_Neumayer+2020.py`: Neumayer+2020 NSC-scaling comparison script.
- `plot/plot_Kong+2026.py`: IMBH seed diagnostics plus redshift-resolved sunk-BH plot script.
- `papers/`: method papers and reference PDFs used for the project.
- `plots/`: project figures and plotting artifacts kept in the repository.
- `tex/`: manuscript and note material.

## Typical Run

```bash
python ~/High-z_SMBH_Seeds/src/run.py --help
nohup python3 ~/High-z_SMBH_Seeds/src/run.py \
  --tree-dir /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/data/fixed_trees_large_spin_dark \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/High-z_SMBH_Seeds_Eddington0.0_ex-situNSC1_Mc7 \
  --Eddington 0.01 --ex-situNSC 1 --lg_cut-off_mass 7.0 --p2 6.75 --p3 0.5 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 512 --log-mh-min 9.0 --log-mh-max 15.0 \
  --extra_out_z_list '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0' --ns-values 2.0 \
  --jobs 32 --ns-jobs 1 \
  --plot_Choksi+2018 --plot_Neumayer+2020 --plot_Gao+2024 --plot_Kong+2026 \
  > ~/run3.log 2>&1 &
nohup python3 ~/High-z_SMBH_Seeds/src/run.py \
  --tree-dir /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/data/fixed_trees_large_spin_dark \
  --clear-output 2 --output /lingshan/disk3/subonan/_outputs/High-z_SMBH_Seeds_Eddington0_ex-situNSC1_M31_Mc7 \
  --Eddington 0.0 --ex-situNSC 1 --lg_cut-off_mass 7.0 --p2 6.75 --p3 0.5 --ts-m 0.2 --ts-r 0.2 \
  --run-all 0 --n-halos 64 --log-mh-min 11.845 --log-mh-max 12.398 \
  --extra_out_z_list '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0' --ns-values 2.0 \
  --jobs 32 --ns-jobs 1 \
  --plot_Choksi+2018 --plot_Neumayer+2020 --plot_Gao+2024 --plot_Kong+2026 \
  > ~/run1.log 2>&1 &
```

Prefer running from the repository root because the project path contains spaces and the relative `my/run.py` entry point is the least error-prone form.
`my/run.py` now always uses the bundled repository `src/` and `data/` layout and checks those paths automatically at startup.

- `--run-all 1` processes the full tree set, while `--run-all 0` activates the mass window and `--n-halos` selection.
- `--jobs` controls halo-level parallelism inside one `N_s` run, and `--ns-jobs` controls how many `N_s` pipelines are run concurrently.
- GC evolution now always uses the evolving host-halo background.
- The physical simulation now always runs to `z=0`; `--extra_out_z_list` only controls extra halo-level sunk-BH summaries reconstructed at earlier redshifts.
- `z=0` is always included automatically in the redshift-resolved sunk-BH outputs.
- `--plot_Gao+2024` writes Gao-style figures under `<output>/_plots_Gao+2024/`.
- `--plot_Choksi+2018` writes Choksi-style figures under `<output>/_plots_Choksi+2018/`.
- `--plot_Neumayer+2020` writes the NSC scaling figures under `<output>/_plots_Neumayer+2020/`.
- `--plot_Kong+2026` writes the IMBH seed diagnostics and redshift-resolved sunk-BH figures under `<output>/_plots_Kong+2026/`.
- Temporary work directories are created under the system temp area and removed automatically at the end of the run.
- Each Sersic index writes to its own `ns*/` directory, while merged products stay at the output top level.
- The main merged outputs are `finalGCs_all.dat`, `depos_all.dat`, `haloSummary_all.csv`, `haloSummaryByZ_all.csv`, `python_evo_summary.csv`, and `run_metadata.json`.

```bash
python3 ~/High-z_SMBH_Seeds/plot/plot_Choksi+2018.py \
  --ns-value 2.0 --out_dir /lingshan/disk3/subonan/_outputs/NSC_Mix_IMBH
python3 ~/High-z_SMBH_Seeds/plot/plot_Neumayer+2020.py \
  --ns-value 2.0 --out_dir /lingshan/disk3/subonan/_outputs/NSC_Mix_IMBH
python3 ~/High-z_SMBH_Seeds/plot/plot_Gao+2024.py --out_dir /lingshan/disk3/subonan/_outputs/NSC_Mix_R0.5
python3 ~/High-z_SMBH_Seeds/plot/plot_Kong+2026.py \
  --ns-value 2.0 --out_dir /lingshan/disk3/subonan/_outputs/NSC_DF1_IMBH1
```

New style:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name ".DS_Store" -delete
```

## Main Run Parameters

The active workflow no longer uses the legacy Gao `input.txt` interface. The main controls now live in `my/run.py`.

### Path and output control

- `--tree-dir`: optional fixed-tree input directory; if omitted, the runner uses the bundled `data/fixed_trees_large_spin` inside this repository.
- `--output`: output directory for the whole run.
- `--clear-output`: remove existing files in the output directory before writing fresh results.

### Formation-model parameters

- `--p2`: GC formation-efficiency normalization in `M_GC = 3e-5 * p2 * M_gas / f_b`.
- `--p3`: threshold in `((Delta M_h / M_h) / Delta t)` above which a formation event is triggered.
- `--mpb-only`: if `1`, form GCs only on the main progenitor branch; if `0`, include all retained branches in the fixed tree.
- `--lg_cut-off_mass`: `log10(M_c / Msun)` for the Schechter cutoff mass in the GC initial-mass function.
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
- `--extra_out_z_list`: comma-separated extra redshifts for halo-level sunk-BH summaries. The simulation itself still runs to `z=0`, `z=0` is always included automatically, and halo selection remains tied to the descendant `z=0` host.
- `--IMBH`: if `1`, enable IMBH seeding in `src/main_spatial.py`; if `0`, write zero IMBH-related columns.
- `--Eddington`: dimensionless Eddington ratio for uncapped growth of the stored central BH state only; IMBHs inside GCs and non-central wandering IMBHs remain non-accreting.
- `--ns-values`: comma-separated list of Sersic indices to run.
- `--jobs`: parallel halo-evolution workers per `N_s`.
- `--ns-jobs`: concurrent `N_s` pipelines.
- `--plot_Gao+2024`: run `plot/plot_Gao+2024.py` automatically after the simulation.
- `--plot_Choksi+2018`: run `plot/plot_Choksi+2018.py` automatically after the simulation.
- `--plot_Neumayer+2020`: run `plot/plot_Neumayer+2020.py` automatically after the simulation.
- `--plot_Kong+2026`: run `plot/plot_Kong+2026.py` automatically after the simulation.
- `--quiet`: reduce progress logging.

### Internal `evo.py` tunables

These are not exposed as `my/run.py` flags, but they still define the evolution grid and deposited-mass bookkeeping:

- `T_UNIVERSE_GYR = 13.799`: Universe-age constant used by the approximate cosmic-time and redshift conversions.
- `dt_max = 0.01` and `t_div = 100`: cap the adaptive step size and define the coarse cosmic-time blocks.
- `binnub = 100`, `r_min = 1.0e-3 kpc`, and `r_sink = NSC_RADIUS_PC * 1.0e-3 = 6.0e-3 kpc`: set the deposited-profile radial binning and the 6 pc NSC/BH sink radius.
- `t_limit = 1.0e-2`: sets the minimum adaptive timescale floor.

## Figure Reproduction

### `plot/plot_Gao+2024.py`

```bash
cd ~/High-z_SMBH_Seeds
python3 plot/plot_Gao+2024.py \
  --allcat <out_dir>/allcat_s-0_p2-6.75_p3-0.5.txt \
  --mpb <out_dir>/mpb_from_fixed_trees.csv \
  --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --output <out_dir>/_plots \
  --gao-fig2-dir <gao_fig2_dir>
```

- `--allcat`: root allcat template path, or one per-`N_s` allcat file from which the script resolves the other `ns*/` tables.
- `--mpb`: path to `mpb_from_fixed_trees.csv`.
- `--ns-values`: set of Sersic indices to include in the figure suite.
- `--output`: destination directory for figure PDFs and the manifest.
- `--gao-fig2-dir`: optional Gao+2024 merged-output directory used for the Figure 2 comparison overlay.
- `--no-observables`: optional switch to disable observational overlays.

This script writes the Gao-style figure PDFs plus `figure_manifest.csv` in the requested output directory.

### `plot/plot_Choksi+2018.py`

```bash
python3 plot/plot_Choksi+2018.py \
  --out_dir <out_dir> \
  --ns-value 2.0
```

- `--out_dir`: one finished model output directory containing the root allcat template, `mpb_from_fixed_trees.csv`, and the `ns*/` products.
- `--plot_dir`: optional override for the output directory. By default it writes to `<out_dir>/_plots_Choksi+2018`.
- `--ns-value`: the single `N_s` value to compare against Choksi+2018 in this plot suite.
- `--figures`: optional comma-separated subset, for example `1,3,6`.
- `--final-z`: optional override for the final redshift if you want the age-based panels to ignore `run_metadata.json`.

This script writes the Choksi-style figure PDFs under `_plots_Choksi+2018/`.

### `plot/plot_Neumayer+2020.py`

```bash
python3 plot/plot_Neumayer+2020.py \
  --out_dir <out_dir> \
  --ns-value 2.0
```

- `--out_dir`: one finished model output directory containing the allcat template, `ns*/` final GC tables, deposited-mass profiles, halo summaries, `run_metadata.json`, and `halo_tree_lookup.csv`.
- `--plot_dir`: optional override for the output directory. By default it writes to `<out_dir>/_plots_Neumayer+2020`.
- `--ns-value`: the single `N_s` value used to build the NSC proxy comparison.
- Required split-style counterpart inputs are `<out_dir>/run_metadata.json`, `<out_dir>/halo_tree_lookup.csv`, `<tree_data_parent>/full_physics_counterparts_z0.csv`, and `<tree_data_parent>/neumayer2020_fig3_divider.json`, where `<tree_data_parent>` is inferred from `run_metadata.json`.

This script writes `Fig.03_galaxy_demographics.pdf`, `Fig.12_nsc_scaling.pdf`, and `Fig.13_bh_nsc_mass_ratio.pdf` under `_plots_Neumayer+2020/`.

### `plot/plot_Kong+2026.py`

```bash
python3 plot/plot_Kong+2026.py \
  --out_dir <out_dir> \
  --ns-value 2.0
```

- `--out_dir`: one finished model output directory containing `ns*/allcat_ns*.txt` for Fig. 1/2, plus `allcat_s-*.txt` and `ns*/haloSummaryByZ_ns*.csv` for the redshift-resolved figures.
- `--plot_dir`: optional override for the output directory. By default it writes to `<out_dir>/_plots_Kong+2026`.
- `--ns-value`: the single `N_s` value used for all six figures; Fig. 1/2 intentionally use only this per-`N_s` formation catalogue.
- `--mass-bin-width-dex`: optional log halo-mass bin width for the mean and standard-deviation tracks.

This script writes `Fig.01_initial_cluster_mass_radius_imbh_seeds.pdf`, `Fig.02_initial_surface_density_metallicity_imbh_thresholds.pdf`, `Fig.03_sunk_bh_mass_vs_halo_mass.pdf`, `Fig.04_sunk_bh_mass_vs_stellar_mass_at_redshift.pdf`, `Fig.05_bh_mass_vs_nsc_mass_at_redshift.pdf`, and `Fig.06_sunk_bh_mass_histogram.pdf` under `_plots_Kong+2026/`.

## Output Schema

The output directory has two persistent layers:

- top level: merged summaries shared across all `N_s`
- `ns*/`: per-Sersic-index outputs such as `ns0p5/`, `ns1p0/`, and `ns4p0/`

Temporary work directories are transient, created under the system temp area, and removed automatically. They are not part of the published output schema.

### Top-level outputs

#### `allcat_s-0_p2-..._p3-....txt`

Convenience copy of one per-`N_s` formation catalog. It is the historical single-file entry point for `plot/plot_Gao+2024.py`, and the paper-specific plot scripts use the containing output directory to resolve the corresponding per-`N_s` tables. Each row is one formed GC, and the row order matches the corresponding `finalGCs_ns*.dat` tables.

Columns:
- `hid_z0`, `logMh_z0`, `logMstar_z0`
- `logMh_form`, `logMstar_form`, `logM_form`
- `zform`, `feh`, `isMPB`
- `subfind_form`, `snap_form`
- `r_galaxy_kpc`, `gc_radius_pc`, `sigma_h_msun_pc2`, `imbh_mass_msun`

#### `mpb_from_fixed_trees.csv`

Compact halo-history table rebuilt from the selected fixed-tree directory. It is used mainly by `plot/plot_Gao+2024.py`, `plot/plot_Choksi+2018.py`, and `plot/plot_Kong+2026.py` for halo-history diagnostics and redshift-matched halo masses.

Columns:
- `subhalo_id_z0`
- `SnapNum`
- `Redshift`
- `logMh_msun_h`
- `SubhaloSpin_x`, `SubhaloSpin_y`, `SubhaloSpin_z`

#### `python_evo_summary.csv`

Compact per-GC summary across all `N_s` values, useful for quick QA without rereading the merged `finalGCs` tables.

Columns:
- `ns`
- `hid_z0`
- `status`
- `m_final_msun`
- `r_final_kpc`

Status codes:
- `1`: alive at the final simulated epoch (`z=0` for runs produced by `my/run.py`)
- `-1`: exhausted to zero mass
- `-2`: tidally torn apart
- `-3`: sunk into the galaxy center
- `-4`: surviving IMBH wanderer at the final simulated epoch
- `-5`: IMBH wanderer sunk into the galaxy center

#### `finalGCs_all.dat`

Merged final-GC table across all `N_s` runs. Each row corresponds to one GC from one halo and one Sersic-index run.

Columns:
- `ns`
- `halo_id_z0`
- `gc_index_halo`
- `status`
- `m_final_msun`
- `log10_m_final_msun`
- `m_init_msun`
- `lookback_time_final_gyr`
- `lookback_time_init_gyr`
- `r_final_kpc`
- `r_init_kpc`
- `gc_radius_pc`
- `sigma_h_msun_pc2`
- `feh`
- `imbh_mass_msun`

#### `depos_all.dat`

Merged deposited-mass profile table across all `N_s` runs.
`depos` records mass lost through external GC evolution channels; terminal stellar mass transferred into `M_NSC` at the 6 pc sink is not added to `depos`.
For non-IMBH GCs that are tidally torn after ending inside the 6 pc aperture, the final residual mass is not added to either `depos` or `M_NSC`.

Columns:
- `ns`
- `halo_id_z0`
- `lookback_time_gyr`
- `bin_index`
- `r_inner_kpc`
- `r_outer_kpc`
- `m_depo_total_msun`
- `m_star_no_evo_msun`
- `m_star_with_evo_msun`

#### `haloSummary_all.csv`

Halo-level summary across all `N_s` runs, including status counts, total GC masses, and SMBH-proxy quantities built from sunk GC and IMBH channels.

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
- `m_imbh_seed_total_msun`
- `m_smbh_gc_sunk_msun`
- `m_smbh_wanderer_sunk_msun`
- `m_smbh_est_msun`
- `ns`

#### `haloSummaryByZ_all.csv`

Long-format halo-level sunk-BH summary across all `N_s` runs and all requested output redshifts. Each row corresponds to one `(ns, hid_z0, z_out)` combination.

Columns:
- `hid_z0`
- `z_out`
- `lookback_to_z0_gyr`
- `halo_mass_available`
- `logMh_z_msun`
- `m_smbh_gc_sunk_msun`
- `m_smbh_wanderer_sunk_msun`
- `m_smbh_est_msun`
- `ns`

`logMh_z_msun` is the MPB halo mass at `z_out`, interpolated in linear halo mass versus cosmic time using the same monotonic MPB block convention as `src/evo.py`. `halo_mass_available` is `0` and `logMh_z_msun` is `NaN` when the requested redshift lies outside the available MPB history for that halo.

#### `run_metadata.json`

Machine-readable record of the main run configuration used to build the output directory.

Keys surfaced in the README:
- `final_redshift`
- `extra_out_z_list`
- `output_redshifts`
- `ts_m`
- `ts_r`
- `DF`
- `tidal_stripping`
- `p2`
- `p3`
- `lg_cut_off_mass`
- `metal`
- `accreted_baryon`
- `eff_rad`
- `eff_rad_catalogue`
- `eff_rad_catalogue_fallback_policy`
- `IMBH`
- `mpb_only`
- `run_all`
- `log_mh_min`
- `log_mh_max`
- `n_halos`
- `ns_values`

### Per-`N_s` outputs

Each `N_s` writes to its own directory such as `ns0p5/`, `ns1p0/`, `ns1p5/`, `ns2p0/`, `ns2p5/`, `ns3p0/`, `ns3p5/`, and `ns4p0/`.

#### `allcat_nsXpY_s-0_p2-..._p3-....txt`

Formation catalog for one `N_s`. It uses the same columns as the top-level `allcat_s-...txt` file.

#### `finalGCs_nsXpY.dat`

Published final-GC table for one `N_s`. It uses the same columns as `finalGCs_all.dat` except for the leading `ns` column.

#### `depos_nsXpY.dat`

Published deposited-mass profile table for one `N_s`. It uses the same columns as `depos_all.dat` except for the leading `ns` column.

#### `haloSummary_nsXpY.csv`

Halo-level summary for one `N_s`. It uses the same columns as `haloSummary_all.csv` except for the trailing `ns` column.

#### `haloSummaryByZ_nsXpY.csv`

Long-format halo-level sunk-BH summary for one `N_s`. It uses the same columns as `haloSummaryByZ_all.csv` except for the trailing `ns` column.

### Plot outputs

When the plot scripts are run, they write:

- `_plots_Gao+2024/Fig.XX_*.pdf` and `_plots_Gao+2024/figure_manifest.csv`: Gao+2024 suite from `plot/plot_Gao+2024.py`.
- `_plots_Choksi+2018/Fig.XX_*.pdf`: Choksi+2018 suite from `plot/plot_Choksi+2018.py`.
- `_plots_Neumayer+2020/Fig.03_galaxy_demographics.pdf`, `_plots_Neumayer+2020/Fig.12_nsc_scaling.pdf`, and `_plots_Neumayer+2020/Fig.13_bh_nsc_mass_ratio.pdf`: Neumayer+2020 comparison from `plot/plot_Neumayer+2020.py`.
- `_plots_Kong+2026/Fig.01_*.pdf` through `_plots_Kong+2026/Fig.06_*.pdf`: IMBH seed diagnostics and redshift-resolved sunk-BH summaries from `plot/plot_Kong+2026.py`.
