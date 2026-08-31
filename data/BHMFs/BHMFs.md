# High-redshift black-hole mass-function catalogue

`BHMFs.csv` is a point-only catalogue for `Fig.06_BHMFs.pdf`. It contains 38
observational points from six samples. The exact CSV header is:

```text
Phi [lgM☉⁻¹Mpc⁻³],sigma_Phi_low [lgM☉⁻¹Mpc⁻³],sigma_Phi_high [lgM☉⁻¹Mpc⁻³],Mbh [M☉],sigma_Mbh_low,sigma_Mbh_high,shape,label,z_low,z_high,ADSABS,data
```

`Phi [lgM☉⁻¹Mpc⁻³]` stores $\Phi=\log_{10}(P)$, where $P$ is the positive
number density per black-hole-mass dex. Thus the stored logarithmic ordinate
can be negative even though the physical density is positive. `Mbh [M☉]` is
the linear mass corresponding to the quoted logarithmic mass-bin centre.
`sigma_Mbh_low` and `sigma_Mbh_high` are lower and upper mass uncertainties in
dex. `shape` contains only Matplotlib marker symbols; the plot colour is
determined by redshift, not by a catalogue colour column.

The six retained samples and their redshift limits are:

| label | points | redshift range | selection |
| --- | ---: | --- | --- |
| `Fei+2026` | 4 | $4.5<z<7.0$ | solid completeness-corrected points |
| `Matthee+2024` | 2 | $4.2<z<5.5$ | complete points only |
| `He+2024` | 11 | $3.5<z<4.25$ | combined HSC+SDSS points through $\log_{10}(M_{\rm BH}/M_\odot)=10.5$ |
| `Taylor+2025` | 4 | $3.5<z<6.0$ | complete points only |
| `Wu+2022` | 11 | $5.7<z<6.5$ | seven SDSS M plus four SDSS O points under one label |
| `Lai+2024` | 6 | $4.5<z<5.3$ | XQz5+ points |

The plotted sample colour is evaluated at the midpoint
$z_{\rm mid}=(z_{\rm low}+z_{\rm high})/2$. The redshift limits are metadata
for this colour assignment; they are not mass or ordinate error bars.

## Asymmetric uncertainties

For a linear density $P^{+u}_{-l}$, after applying the paper's density-unit
factor, the catalogue stores the logarithmic central value and the two
logarithmic error distances as

$$
\Phi=\log_{10}(P),\qquad
\sigma_{\Phi,\mathrm{low}}=\log_{10}(P)-\log_{10}(P-l),\qquad
\sigma_{\Phi,\mathrm{high}}=\log_{10}(P+u)-\log_{10}(P).
$$

Quoted logarithmic errors are retained directly. No ordinate or mass error is
symmetrised. If the quoted lower density reaches zero, the lower error
distance is stored as the positive value `inf`. This means that the lower
logarithmic ordinate is $-\infty$; it is not a finite upper limit disguised by
a numerical floor. Matplotlib receives a finite lower error of zero for these
rows, and `Fig.06_BHMFs.pdf` draws an uncapped vertical extension from the
point to the visible lower boundary of the logarithmic ordinate axis.

## Source selection and provenance

The `ADSABS` column contains the abstract URL for each paper. The `data`
column contains the direct public machine-readable table URL when one was
available, or the public DOI article URL when the table had to be read from
the downloaded article. The downloaded files and any digitisation material
remain outside the repositories.

- **Fei+2026:** The four solid, completeness-corrected points in Figure 7
  (left) and Table 4 are retained. The four uncorrected points are omitted.
  Table 4 was downloaded from the AAS machine-readable supplement. The first
  mass bin is 1 dex wide and has a 0.5 dex half-width; the remaining bins have
  0.25 dex half-widths.
- **Matthee+2024:** The two complete points from Figure 19 and Table 6 are
  retained; the incomplete $\log_{10}(M_{\rm BH}/M_\odot)=7.1$ point is omitted.
  The quoted logarithmic ordinate errors are stored separately as lower and
  upper errors. Table 6 was downloaded from the AAS machine-readable
  supplement.
- **He+2024:** The right-hand-column combined HSC+SDSS values from Figure 12
  and Table 8 are retained at mass-bin centres 7.5 through 10.5 in 0.3 dex
  steps. The 10.8 bin is excluded because its combined density is zero and it
  cannot be plotted on a logarithmic ordinate. Table 8 was downloaded from the
  AAS machine-readable supplement; the combined column, rather than the HSC
  column alone, is used.
- **Taylor+2025:** The four complete points from the right-hand column of
  Figure 12 and Table 3 are retained at centres 6.75, 7.25, 7.75, and 8.25.
  The incomplete 6.25 bin is omitted. Table 3 was downloaded from the AAS
  machine-readable supplement.
- **Wu+2022:** The 11 points in the left-hand panel of Figure 5 and Table B3
  are retained: seven SDSS M points and four SDSS O points. They share the
  label `Wu+2022`, and repeated mass positions remain separate rows rather
  than being averaged. Circles and squares preserve the two source marker
  shapes, while the plot assigns both the same redshift-resolved colour.
  The values were read from the downloaded public MNRAS article because no
  machine-readable Table B3 download was available.
- **Lai+2024:** All six XQz5+ points from Figure 3 and Table 3 are retained.
  The six bins are equally spaced in logarithmic mass; their plotted
  half-width is represented by `sigma_Mbh_low = sigma_Mbh_high = 0.19` dex.
  The values were read from the downloaded public MNRAS article because no
  machine-readable Table 3 download was available.

The public source URLs recorded in `BHMFs.csv` are the AAS supplements for
Fei+2026, Matthee+2024, He+2024, and Taylor+2025, and
`https://doi.org/10.1093/mnras/stac2833` and
`https://doi.org/10.1093/mnras/stae1301` for Wu+2022 and Lai+2024,
respectively.

No `shape=line` rows, Schechter fits, theoretical envelopes, or other
theoretical catalogue rows are included. The only theoretical curves in
`Fig.06_BHMFs.pdf` are the redshift-resolved `This work` model curves produced
directly from the project output by `plot_Kong&Li2026.py`.
