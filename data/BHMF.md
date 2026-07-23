# BHMF catalogue provenance

`BHMF.csv` stores the ordinate used by Figure 7 of Fei et al. (2026), namely
`log10(Phi / (Mpc^-3 dex^-1))`, in the column named `Phi [lgM☉⁻¹Mpc⁻³]`.
The plotting code therefore plots that column directly; it does not exponentiate
it.  `Mbh [M☉]` is a linear mass, while `sigma_Mbh` is a symmetric uncertainty
in `log10(M_BH/M☉)` (dex) and is used directly as the horizontal error bar.
The `sigma_Phi` values are symmetric errors in the logarithmic ordinate.

No Schechter fit is included in the catalogue or in `Fig.06_BHMF2.pdf`.
The `data` field is blank for every row because none of the sources below
provided a direct machine-readable download for the plotted BHMF values.
The `ADSABS` field contains the abstract landing page for the corresponding
reference.

## Fei+2026

The four corrected and four uncorrected points are taken from Section 4.4,
Figure 7 (left), and Table 4 of Fei et al. (2026).  Table 4 gives the central
logarithmic mass-bin positions `5.50, 6.25, 6.75, 7.25` and densities in
`10^-3 Mpc^-3 dex^-1`:

```text
corrected:   12.70 +/- 6.73, 0.64 +/- 0.29, 0.50 +/- 0.25, 0.18 +/- 0.11
uncorrected:  4.45 +/- 2.36, 0.22 +/- 0.10, 0.17 +/- 0.09, 0.06 +/- 0.04
```

The first mass bin is 5.0--6.0, so its symmetric log-mass error is 0.50 dex;
the remaining 0.5-dex bins have 0.25 dex errors.  The filled red hexagons are
the completeness-corrected GLIMPSE BHs and the open red hexagons are the same
measurements without the completeness correction.  For a linear density
`P^{+u}_{-l}`, the catalogue uses

```text
log10(P)                           -> Phi
0.5 * [log10((P+u)/P) + log10(P/(P-l))] -> sigma_Phi
```

with the `10^-3` unit factor applied before taking the logarithm.  The source
paper is available through
[ADSABS](https://ui.adsabs.harvard.edu/abs/2026ApJ...1003..244F/abstract).
The exact CSV labels are `Fei+2026` and `Fei+2026 (w/o correction)`.

## Matthee+2024

The three orange square points are from the BHMF table associated with
Figure 7 of Matthee et al. (2024), as reproduced in the comparison used by
Fei et al. (2026).  The paper's logarithmic values are

```text
log10(M_BH/M☉)   log10(Phi)                 redshift
7.1 +/- 0.2      -4.86 +0.20/-0.37          4.2 < z < 5.5
7.5 +/- 0.2      -4.27 +0.11/-0.15          4.2 < z < 5.5
8.1 +/- 0.4      -5.05 +0.18/-0.30          4.2 < z < 5.5
```

The ordinate errors in `BHMF.csv` are the direct averages of the two quoted
logarithmic errors: 0.285, 0.130, and 0.240 dex.  The lowest-mass square is
open because the source flags that point as incomplete; the two higher-mass
squares are filled orange.  The reference abstract is at
[ADSABS](https://ui.adsabs.harvard.edu/abs/2024ApJ...963..129M/abstract).

## Taylor+2025

The five blue circles are from Section 6.2 and Table 3 of Taylor et al.
(2025), the completeness-corrected BHMF for `3.5 < z < 6`.  Table 3 defines
0.5-dex bins centred at `log10(M_BH/M☉) = 6.25, 6.75, 7.25, 7.75, 8.25` and
quotes densities in `10^-6 Mpc^-3 dex^-1`:

```text
258 +125/-113, 276 +83.7/-59.8, 113 +32.5/-24.3,
36.1 +14.8/-10.5, 7.67 +7.51/-4.17
```

Each bin therefore has `sigma_Mbh = 0.25` dex.  The linear-density errors
were converted using the same logarithmic rule stated in the Fei section;
the resulting symmetric ordinate errors are approximately
`0.2109, 0.1105, 0.1075, 0.1492, 0.3186` dex.  The lowest-mass point is the
open circle because its line-detection completeness is below 20%; the other
four points are filled blue.  The reference abstract is at
[ADSABS](https://ui.adsabs.harvard.edu/abs/2025ApJ...986..165T/abstract).

## He+2024

The five grey triangles are the HSC-only entries in Table 8 of He et al.
(2024), for the `3.5 < z < 4.25` sample.  The table uses 0.3-dex mass bins,
so `sigma_Mbh = 0.15` dex.  The included centres and densities are

```text
log10(M_BH/M☉)       Phi (10^-7 Mpc^-3 dex^-1)
7.50                  12.14 +/- 8.70
7.80                   9.92 +/- 6.24
8.10                  11.30 +/- 4.34
8.40                  13.90 +/- 3.68
8.70                   7.99 +/- 2.87
```

The catalogue includes only the five HSC points visible within the Figure 7
x-range; the 9.00 bin and higher-mass bins from Table 8 are intentionally
omitted.  The symmetric linear errors were converted to logarithmic errors
before storage.  All five triangles are filled grey.  The reference abstract
is at [ADSABS](https://ui.adsabs.harvard.edu/abs/2024ApJ...962..152H/abstract).

## Jeon+2025

The line rows are a digitisation of the top panel (`z=5--6`) of Figure 1 in
Jeon et al. (2025), which is the theoretical source of the shaded model
regions compared by Fei et al. (2026).  The official arXiv source package
was checked and contains the figure image but no machine-readable BHMF array,
so the `data` field remains blank.  The source figure used for the
digitisation was `bhmf_z5.png` from arXiv:2503.14703.  The reference abstract
is at [ADSABS](https://ui.adsabs.harvard.edu/abs/2025ApJ...988..110J/abstract).

The pixel calibration was

```text
log10(M_BH/M☉) = 2 + (x_pixel - 159) / 133
log10(Phi)      = 1.70457 - 0.0152616 * y_pixel
```

where the first relation uses the labelled `log(M_BH)` ticks and the second
was fitted to the labelled `log(Phi)` ticks from 1 to -6.  The coloured curve
pixels were sampled over the Figure 7 comparison range, retaining 628 source
x-pixels for the heavy Eddington-limited morphology, 629 for the heavy
super-Eddington morphology, and 595 for the light-seed morphology.  Each
central/lower/upper role was then linearly interpolated to 1200 log-mass
samples.  A 19-pixel rolling median was used only to remove anti-aliased and
dashed-line pixel excursions; this is a digitisation cleanup, not an analytic
fit.  The resulting three dense groups per model use `shape=line`, blank error
columns, and labels ending in `(lower envelope)` or `(upper envelope)` for
the envelope rows.  Their coordinates are stored in this same CSV rather
than in a second auxiliary file.
The exact line-row labels are `Heavy Eddington-limited`, `Heavy Eddington-limited (lower envelope)`, `Heavy Eddington-limited (upper envelope)`, `Heavy Super-Eddington`, `Heavy Super-Eddington (lower envelope)`, `Heavy Super-Eddington (upper envelope)`, `Light Eddington-limited`, `Light Eddington-limited (lower envelope)`, and `Light Eddington-limited (upper envelope)`.

The visible Fei Figure 7 model labels and their source-curve mappings are:

| Fei Figure 7 label | Jeon source morphology | CSV colour |
| --- | --- | --- |
| Heavy Eddington-limited | `Heavy seeds Edd-limited` (Bondi, `f_Edd=1`, `f_duty=0.5`) | red |
| Heavy Super-Eddington | `Heavy seeds super-Edd-limited` (Bondi, `f_Edd=1.5`, `f_duty=0.5`) | blue |
| Light Eddington-limited | `Light seeds super-Edd-limited` (Bondi, `f_Edd=1.5`, `f_duty=0.5`) | green |

The last row is the requested morphology check: Fei calls the green region
“Light Eddington-limited”, but its low-mass peak followed by the declining
high-mass tail matches Jeon's `Light seeds super-Edd-limited` curve, not the
long high-abundance `Light seeds forced super-Edd` curve.  The catalogue
therefore retains Fei's visible label while documenting this source-label
mismatch; it does not use the forced-super-Edd curve.

For reference, Jeon's model prescription uses light seeds of order
`100 M☉` and heavy seeds of order `10^5 M☉`, with

```text
Delta M_BH = min(f_duty * dot(M)_acc * Delta t, M_cold)
dot(M)_acc = f_Edd * dot(M)_Edd                       (Eddington model)
dot(M)_acc = min(dot(M)_Bondi, f_Edd * dot(M)_Edd)     (Bondi model)
```

and `dot(M)_Edd = 2.7e-3 (M_BH/10^5 M☉) (epsilon_r/0.1)^-1 M☉ yr^-1`.
These equations document the original model source; the CSV contains the
digitised Figure 1 values rather than a newly evaluated analytic model.
