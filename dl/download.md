# Data Download Details

External Illustris-1-Dark tree workflow

Compared with the original Gao+2024 repository, this project now has an explicit external tree pipeline under `/lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark` for building Gao-compatible fixed trees before any GC physics is run. The maintained workflow is now a mixed-suite pipeline: `1_select_targets.py` samples one halo from each non-empty log halo-mass bin, controlled by `--max_num_halo`, `--min_halo_mass`, and `--max_halo_mass`. If the requested lower bound is below `10^11 Msun`, bins below `10^11 Msun` are sampled from `TNG50-1-Dark`, while bins at and above `10^11 Msun` are sampled from `Illustris-1-Dark`. The selector uses a fixed internal random seed, so repeated runs with the same cached group catalogs are deterministic. In practice, this means the model no longer depends on one bundled fixed-tree sample: `my/run.py` can ingest any corrected external tree directory through `--tree-dir`, for example `/lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/data/fixed_trees_large_spin_dark`.

##

The maintained external workflow is:

```bash
python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/1_select_targets.py \
  --max_num_halo 512 --min_halo_mass 8.0 --max_halo_mass 14.65
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Cube/scripts/1_select_targets.py \
  --cube-origin-ckpc-h 53239.5 23993.7 11946.8 --cube-side-cmpc 16.0 \
  --min_halo_mass 10.0 --max_halo_mass 14.65
python /lingshan/disk3/subonan/Illustris-1-Dark_Min13/scripts/1_select_targets.py \
  --all-illustris-halos --min_halo_mass 13.0
python /lingshan/disk3/subonan/TNG50-1-Dark_Min13/scripts/1_select_targets.py \
  --all-tng50-halos --min_halo_mass 13.0

python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/2_download_full_trees.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Cube/scripts/2_download_full_trees.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Min13/scripts/2_download_full_trees.py
python3 /lingshan/disk3/subonan/TNG50-1-Dark_Min13/scripts/2_download_full_trees.py

python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/3_convert_full_trees_to_fixed_dat.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Cube/scripts/3_convert_full_trees_to_fixed_dat.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Min13/scripts/3_convert_full_trees_to_fixed_dat.py
python3 /lingshan/disk3/subonan/TNG50-1-Dark_Min13/scripts/3_convert_full_trees_to_fixed_dat.py

python3 /lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark/scripts/4_validate_fixed_trees.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Cube/scripts/4_validate_fixed_trees.py
python3 /lingshan/disk3/subonan/Illustris-1-Dark_Min13/scripts/4_validate_fixed_trees.py
python3 /lingshan/disk3/subonan/TNG50-1-Dark_Min13/scripts/4_validate_fixed_trees.py
```

##

Current storage layout under `/lingshan/disk3/subonan/Illustris-1-Dark+TNG50-1-Dark`:

- `data/groupcat_fields_illustris1_dark/`: cached `Illustris-1-Dark` z=0 `Group_M_Mean200` and `GroupFirstSub` cutouts used by the selector.
- `data/groupcat_fields_tng50_1_dark/`: cached `TNG50-1-Dark` z=0 `Group_M_Mean200` and `GroupFirstSub` cutouts used by the selector.
- `data/sublink_full_dark/`: raw downloaded full SubLink subtree HDF5 files, with suite-prefixed basenames such as `illustris1_dark_sublink_full_subhalo_*.hdf5` and `tng50_1_dark_sublink_full_subhalo_*.hdf5`.
- `data/fixed_trees_large_spin_dark/`: corrected Gao-compatible fixed-tree `.dat` files plus conversion and validation metadata.
- `full_tree_download_summary.json` and `full_tree_download_failures.json`: downloader summary logs written directly in the parent directory, not in a separate `logs/` subdirectory.

##

Script roles:

- `1_select_targets.py`: resolves the required suite(s), caches the z=0 group-catalog fields, writes per-suite `SnapNum -> redshift` lookup tables, and builds the suite-aware manifest.
- `2_download_full_trees.py`: reads one manifest row at a time, uses the saved per-row `subhalo_url_z0`, and downloads one raw full subtree HDF5 file per selected halo.
- `3_convert_full_trees_to_fixed_dat.py`: reads the suite-specific `snaps2redshifts_*.txt` files, applies the existing branch-correction logic, and writes corrected fixed-tree `.dat` files.
- `4_validate_fixed_trees.py`: checks the converted `.dat` files for schema consistency and basic correction invariants before they are used by the GC model.

##

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