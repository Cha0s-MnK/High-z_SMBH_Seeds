# TNG50 and TNG100 dark-matter tree workflow

This directory contains the external tree workflow for the High-z SMBH Seeds project. It prepares corrected fixed merger trees from TNG50-1-Dark and TNG100-1-Dark before the GC, NSC, dynamical-friction, and IMBH calculations are run.

The default data directory is `/lingshan/disk3/subonan/TNG50+100-1-Dark`. Every script accepts the absolute `--data_dir` option, and the same directory must be supplied to all stages in one run. Relative paths are rejected so that catalogue caches, raw trees, fixed trees, and metadata cannot silently be split between working directories.

The target collection is now full-box for both simulations. TNG50-1-Dark contributes every eligible z = 0 group satisfying $M_{\rm halo} > 10^{10}\,M_\odot$ and $M_{\rm halo} \le 10^{13}\,M_\odot$ in its complete z = 0 group catalogue. TNG100-1-Dark contributes every eligible z = 0 group satisfying $M_{\rm halo} > 10^{13}\,M_\odot$ in its complete z = 0 group catalogue. The lower bounds are strict; the TNG50 upper bound is inclusive.

Both selections use `Group_M_Mean200`, whose native API unit is $10^{10}\,M_\odot/h$. The selector converts it with $h=0.6774$ before applying the target rules. A group is eligible only if `GroupFirstSub >= 0` and its converted mass is finite and positive. `GroupPos` is not downloaded, read, or used: the complete catalogue is interpreted as the native full simulation box, with no coordinate filter and no periodic wrapping.

The merger-tree resolution rule is separate from the z = 0 target rules. The combined converter retains nodes with at least 500 dark-matter particles, using an inclusive mass comparison:

$$
M_{\rm node} \ge 500\,m_{\rm DM}.
$$

The resulting suite-specific limits are $2.69219125\times10^8\,M_\odot$ for TNG50 and $4.4282553\times10^9\,M_\odot$ for TNG100. This filter is applied to `SubhaloMass` during conversion, not while downloading the raw SubLink tree.

## Workflow order

Run the three normal stages in order:

| Stage | Python file | Main input | Main output |
| --- | --- | --- | --- |
| Target selection | `1_select_targets.py` | TNG API snapshot information and z = 0 group-catalogue fields | Full-box manifest, selection metadata, redshift tables, and catalogue caches |
| Raw-tree download | `2_download_full_trees.py` | The complete manifest and TNG primary-subhalo URLs | One validated raw SubLink HDF5 tree per selected target |
| Conversion and validation | `3_convert_and_validate_fixed_trees.py` | The manifest, raw HDF5 trees, and both suite redshift tables | Regenerated nine-column fixed trees, lookup files, conversion summary, and validation reports |

Example:

```bash
python3 /home/subonan/GitHub/dl/1_select_targets.py --data_dir /lingshan/disk3/subonan/TNG50+100-1-Dark
python3 /home/subonan/GitHub/dl/2_download_full_trees_parallel.py --data_dir /lingshan/disk3/subonan/TNG50+100-1-Dark --jobs 64
nohup bash -c '
  while :; do
    python3 /home/subonan/GitHub/dl/2_download_full_trees_parallel.py \
      --data_dir /lingshan/disk3/subonan/TNG50+100-1-Dark \
      --jobs 96
    rc=$?
    [ "$rc" -eq 0 ] && break
    echo "Download failed with exit code $rc; retrying in 30 seconds..."
    sleep 30
  done
  ' > /lingshan/disk3/subonan/TNG50+100-1-Dark/full_tree_download_loop.log 2>&1 < /dev/null & echo "PID: $!"
python3 /home/subonan/GitHub/dl/3_convert_and_validate_fixed_trees.py --data_dir /lingshan/disk3/subonan/TNG50+100-1-Dark --min_particle 500 --min_mass 7.17e7 --out_dir /lingshan/disk3/subonan/TNG50+100-1-Dark_Small
```

The old `3_convert_full_trees_to_fixed_dat.py` and `4_validate_fixed_trees.py` remain unchanged legacy/reference entry points. They retain their historical interfaces and mass-filter behaviour. They are not part of the normal three-stage workflow. The new combined command has no `--overwrite` and no `--min-mass-msun` option: every fixed `.dat` file named by the current manifest is converted again and replaces the current-manifest product after all conversions and validations succeed.

## Selector outputs and full-box provenance

`1_select_targets.py` writes the complete mixed-suite manifest in deterministic order: TNG50 rows sorted by ascending z = 0 halo ID, followed by TNG100 rows sorted by ascending z = 0 halo ID. The manifest preserves the original suite key, z = 0 halo and subhalo IDs, selection rule, API URL, raw-tree basename, fixed-tree basename, and the 500-particle tree-node policy. The later model-facing halo-ID offset is not inserted into any selector output or filename.

`targets_z0_dark.json` contains a `full_box_selection` object. It records the native side lengths $35\,\mathrm{cMpc}/h$ and $75\,\mathrm{cMpc}/h$, their physical equivalents $35/h=51.668\,\mathrm{cMpc}$ and $75/h=110.717\,\mathrm{cMpc}$, both native and physical volumes, `geometry: native_full_simulation_box`, `periodic_wrapping: false`, and `coordinate_filter_applied: false`. It also records the complete catalogue scope, eligibility rules, strict/inclusive target inequalities, tree-node thresholds, dynamic suite counts, manifest path, combined order, and all manifest records. The old cube fields such as `cube_origin_ckpc_h`, `cube_side_cmpc`, and `cube_side_ckpc_h` are not written.

The two simulations are not one homogeneous cosmological volume. They are a mixed collection with separate box sizes and separate target populations. Any abundance or mass-function calculation must preserve the suite key and use the appropriate volume.

## Raw-data and fixed-tree consequences

Removing a z = 0 mass threshold or changing the 500-particle tree-node policy does not require a wholesale re-download of valid existing `sublink_full_dark` files. SubLink downloads contain the full raw tree; the 500-particle floor is applied later by the combined converter. Rerun the selector to create the new full-box manifest, then run `2_download_full_trees.py`. Valid existing raw files are reused by default, while newly selected or missing target trees are downloaded. Use the downloader's overwrite option only when a raw file itself must be refreshed.

Existing fixed `.dat` files made with the old strict $10^9\,M_\odot$ policy are not equivalent to the new products. Once the new manifest and raw files are available, `3_convert_and_validate_fixed_trees.py` regenerates the expected fixed files automatically. Files belonging to older manifests that are not named by the current manifest are retained for auditability and are not included in the new lookups.

The combined command stages all current-manifest fixed trees and lookup products in a temporary directory. It commits them only after conversion, fixed-tree validation, and lookup validation have succeeded. A missing or malformed raw tree, redshift table, or required dataset therefore leaves the previous current-manifest fixed products untouched; the command writes diagnostics and exits non-zero.

## Mixed-suite halo-ID correction

The raw TNG halo namespace is preserved in the raw trees, fixed-tree node IDs, manifest, URLs, filenames, selected-ID text files, and `id_lookup_original.csv`. The model-facing `id_lookup_large_dark.csv` applies only this z = 0 host-ID correction:

$$
H_{\rm model} =
\begin{cases}
H_{\rm original}, & \text{TNG50},\\
H_{\rm original}+1{,}000{,}000, & \text{TNG100}.
\end{cases}
$$

The offset is a fixed constant in the combined script and is audited against `/lingshan/disk3/subonan/TNG50+100-1-Dark_New/fixed_trees_large_spin_dark_runid`. It changes the model-facing z = 0 join key only; it does not shift any fixed-tree topology or node identifier. `id_lookup_large_dark.txt` retains its original meaning as `file_index,simulation_key,raw_tree_basename` and contains no shifted halo-ID field.

The model reads `id_lookup_large_dark.csv`, so model output tables use the shifted TNG100 namespace. The model's copied `halo_tree_lookup.csv` should be joined to catalogue provenance by `simulation_key` and `fixed_tree_basename`, not by an unqualified original halo number.

## Data layout

Products are direct children of the configured data directory or of its fixed-tree directory:

| Path | Contents |
| --- | --- |
| `groupcat_fields_tng50_1_dark/` | Cached TNG50 z = 0 `Group_M_Mean200` and `GroupFirstSub` fields |
| `groupcat_fields_tng100_1_dark/` | Cached TNG100 z = 0 `Group_M_Mean200` and `GroupFirstSub` fields |
| `snaps2redshifts_tng50_1_dark.txt` | TNG50 snapshot-number to redshift lookup |
| `snaps2redshifts_tng100_1_dark.txt` | TNG100 snapshot-number to redshift lookup |
| `target_manifest_dark.csv` | Complete suite-aware target manifest |
| `halo_selection_labels_dark.csv` | Selection rule and target-property table |
| `targets_z0_dark.json` | Full-box geometry, criteria, counts, and complete records |
| `selected_halos_z0_dark.txt` | Simulation key and original z = 0 group ID |
| `selected_subhalos_z0_dark.txt` | Simulation key and z = 0 primary-subhalo ID |
| `sublink_full_dark/` | Suite-prefixed raw SubLink HDF5 trees |
| `fixed_trees_large_spin_dark/` | Suite-prefixed corrected `.dat` trees and conversion/validation products |
| `fixed_trees_large_spin_dark/id_lookup_original.csv` | Original manifest IDs for auditability |
| `fixed_trees_large_spin_dark/id_lookup_large_dark.csv` | Model-facing IDs with the TNG100 offset |
| `fixed_trees_large_spin_dark/id_lookup_large_dark.txt` | File index, suite key, and raw basename lookup |
| `fixed_trees_large_spin_dark/conversion_summary.json` | Conversion policy, manifest identity, counts, and per-file details |
| `fixed_trees_large_spin_dark/validation_report.json` | Fixed-tree and lookup validation results |
| `fixed_trees_large_spin_dark/validation_errors.txt` | Human-readable validation diagnostics |

The API key is read from `TNG_API_KEY`. The workflow requires Python 3 with `numpy`, `h5py`, `requests`, `urllib3`, and `tqdm`; raw downloads use `wget`.

## Scientific bookkeeping

For physical number densities, use the physical full-box volumes:

$$
V_{50}=(51.668\,\mathrm{cMpc})^3,\qquad
V_{100}=(110.717\,\mathrm{cMpc})^3,
\qquad
w_{100}=V_{50}/V_{100}.
$$

The native volumes $35^3$ and $75^3$ in $({\rm cMpc}/h)^3$ may also be recorded, but they must not be mixed with a density reported in $\mathrm{cMpc}^{-3}$. The HMF and Kong&Li plotting readers derive both volumes from the full-box metadata and use physical $\mathrm{cMpc}^3$ for their final density normalisation.

This workflow changes the target population, tree-node resolution floor, mixed-suite ID lookup, and volume provenance. It does not change GC formation, GC disruption, NSC growth, dynamical-friction treatment, or the downstream Rantala-based IMBH prescription.
