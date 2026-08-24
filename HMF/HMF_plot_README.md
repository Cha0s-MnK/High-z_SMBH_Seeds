# TNG50/TNG100 halo mass-function plot

This script produces one overlaid HMF figure:

- x-axis: \(\log_{10}(M_{200\mathrm{c}}/M_\odot)\), beginning at \(10^9\,M_\odot\);
- y-axis: \(\log_{10}[\mathrm{d}n/\mathrm{d}\log_{10}M_{200\mathrm{c}}\,/\,(\mathrm{cMpc}^{-3}\,\mathrm{dex}^{-1})]\);
- black solid line: Tinker (2008) \(M_{200\mathrm{c}}\) \(\Lambda\)CDM HMF with the IllustrisTNG cosmology;
- blue circles: TNG50-1-Dark;
- red squares: TNG100-1-Dark;
- dotted vertical lines: a 300-DM-particle reference limit when particle masses are supplied;
- error bars: central 68% Poisson intervals for the binned number density.

It uses `Group/Group_M_Crit200` so that the simulation data and the theoretical line share the same halo-mass definition. It reads the catalogue redshift from the group-catalogue headers and requires TNG50 and TNG100 to be from the same snapshot. The standard full-box volume defaults are 35 cMpc/h for TNG50 and 75 cMpc/h for TNG100, converted with \(h=0.6774\). Override them only for a non-standard catalogue or subvolume.

## Requirements

The environment that runs the script needs `numpy`, `scipy`, `matplotlib`, `h5py`, and `colossus`.

## z=0 command

```bash
python3 /home/subonan/GitHub/HMF/dl4HMF.py --help
python3 /home/subonan/GitHub/HMF/dl4HMF.py --data_dir /lingshan/disk3/subonan/TNG50+100-1-Dark_HMF
python /home/subonan/GitHub/HMF/plot_TNG_HMF.py \
  --tng50 '/lingshan/disk3/subonan/TNG50+100-1-Dark_HMF/TNG50-1-Dark/output/groups_099/fof_subhalo_tab_099.*.hdf5' \
  --tng100 '/lingshan/disk3/subonan/TNG50+100-1-Dark_HMF/TNG100-1-Dark/output/groups_099/fof_subhalo_tab_099.*.hdf5' \
  --min-particles 500 \
  --output /lingshan/disk3/subonan/TNG50+100-1-Dark_HMF/tng50_tng100_hmf_z0
python3 /home/subonan/GitHub/HMF/plot_dl_HMF.py --data-dir /lingshan/disk3/subonan/TNG50+100-1-Dark --min-particles 500
python3 /home/subonan/GitHub/HMF/plot_dl_HMF.py --data-dir /lingshan/disk3/subonan/TNG50+100-1-Dark
```

The file patterns should include all group-catalogue shards. The script saves both `tng50_tng100_hmf_z0.pdf` and `tng50_tng100_hmf_z0.png`.

## Deliberate defaults

- `--mmin 1e9`: fixes the left edge requested for the plot.
- `--mmax 1e15`: can be raised if the selected TNG100 snapshot contains more massive groups.
- `--dlogm 0.20`: a balance between detail and Poisson noise.
- `--min-particles 300`: a conservative visual reference only. It does not delete lower-mass catalogue points. Use `--min-particles 32` to show the Friends-of-Friends catalogue threshold instead.
- The default DM-particle masses are \(5.384\times10^5\,M_\odot\) for TNG50-1-Dark and \(8.857\times10^6\,M_\odot\) for TNG100-1-Dark. Override `--tng50-dm-particle-mass` and `--tng100-dm-particle-mass` for another resolution level; do not reuse these values across levels.
- `--hubble-param`, `--tng50-box-size-cmpc`, and `--tng100-box-size-cmpc`: override the unit conversion or volume only when the input is not a standard full TNG50/TNG100 box.
