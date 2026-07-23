# Provenance for `Mbh-Mstar.csv`

The CSV contains the 52 current observational entries, plus four additional method-specific entries for estimates explicitly reported by the original papers. Masses are linear values in `M☉`. Where a paper reports a logarithmic mass, the CSV value is `10**log10(M/M☉)`. Errors are not plotted and are recorded here only. A numerical upper-limit value remains in the CSV as the plotted position and is identified as an upper limit below.

The current Juodzbalis+2026 comparison points do not have object identifiers in its machine-readable source package. For those rows, the existing marker centres from `juodzbalis2026_fig4_mbh_mstar_points.csv` were retained as explicitly labelled fallback coordinates and converted from its `logMstar` and `logMBH` columns. The original reference paper was still used for the source label and redshift rule wherever possible.

## Carnall+2023

- CSV rows 2--3 are the two black-hole estimates for GS-9209. The source is [Carnall+2023](https://arxiv.org/abs/2301.11413). The galaxy redshift is `z = 4.6582 ± 0.0002` in the main text. The stellar mass is `log10(M*/M☉) = 10.58 ± 0.02` in the main text. Row 2 uses the broad-Hα estimate `log10(MBH/M☉) = 8.7 ± 0.1`; row 3 uses the velocity-dispersion estimate `log10(MBH/M☉) = 8.9 ± 0.1`. These two rows are direct source-paper values, not the old digitised marker centre. The original stellar-mass uncertainty and both black-hole uncertainties are recorded here but are not plotted.

## Ding+2023

- CSV rows 4--7 are the two estimates for each of the two quasars in [Ding+2023](https://arxiv.org/abs/2211.14329), at `z ≃ 6.4`. The host masses come from the main-text discussion of the SED fits and Extended Data Table 1: `Mstar = 1.3^{+2.0}_{-0.6} × 10^11 M☉` for J2236+0032 and `3.4^{+7.6}_{-1.9} × 10^10 M☉` for J2255+0251. The uncorrected virial estimates in the black-hole-mass section are `1.54 ± 0.27 × 10^9 M☉` and `2.02 ± 0.17 × 10^8 M☉` (rows 5 and 7). After subtracting the host contribution, the values adopted for Figure 4 and listed in Extended Data Table 2 are `1.36 ± 0.15 × 10^9 M☉` and `1.97 ± 0.17 × 10^8 M☉` (rows 4 and 6). Each estimate is one CSV entry; the unchanged host mass is repeated for the two methods of each quasar. The source also quotes an intrinsic single-epoch uncertainty of about 0.4 dex, not included in the formal errors.

## Goulding+2023

- CSV row 8 represents UHZ-1 from [Goulding+2023](https://arxiv.org/abs/2308.02750). The spectroscopic redshift is `z = 10.073 ± 0.002` in the abstract and Section 3.1. Table 2 gives `log10(Mstar/M☉) = 8.14` at the posterior median; this is converted to `1.380384264 × 10^8 M☉`. The paper gives a conservative black-hole range of approximately `10^7--10^8 M☉` rather than one preferred value. Following the agreed rule, the CSV uses the ordinary linear midpoint `5.5 × 10^7 M☉` and records the range here. The previous plotted coordinate was checked against this source value; no separate object-specific point identity was available beyond UHZ-1.

## Harikane+2023

- CSV rows 9--17 correspond to the nine Harikane comparison markers currently digitised in Juodzbalis+2026 Figure 4. The reference is [Harikane+2023](https://arxiv.org/abs/2303.11946), whose final broad-line sample spans `4.015 ≤ z ≤ 6.936` (abstract and Table 1). Because the digitised marker centres cannot be matched to the individual Table 1 objects, the CSV uses the middle of that quoted range, `z = 5.4755`, for each row, as agreed. The exact `Mbh` and `Mstar` coordinates are fallback values converted from the current Juodzbalis+2026 digitisation; they are not claimed to be direct table values. The source paper’s object-level errors remain in its Table 3 but cannot be assigned safely to these unidentified fallback rows, so they are not copied into the CSV or plotted.

## Ivey+2026

- CSV rows 18--19 are the two The Cliff estimates from [Ivey+2026](https://arxiv.org/abs/2604.09177), both at `z = 3.55`. Table 4 and the black-hole-mass section give the fiducial Hβ value `log10(MBH/M☉) = 7.35 ± 0.24` and the scattering-scenario value `log10(MBH/M☉) = 6.32 ± 0.22`. The same table gives the dynamical stellar-mass upper limit `log10(Mstar/M☉) = 8.41`; the CSV stores `2.570395783 × 10^8 M☉` as the numerical upper-limit position for both rows. The two rows deliberately use the same `Ivey+2026` label and therefore share one legend entry. The stellar-mass upper-limit status and both black-hole errors are recorded here only.

## Juodzbalis+2025

- CSV rows 20--44 are the 25 JADES AGN comparison markers currently digitised in Juodzbalis+2026 Figure 4. The reference is [Juodzbalis+2025](https://arxiv.org/abs/2504.03551). Section 5.2 and Table 5 describe the source stellar-mass measurements; the paper states that the ForcePho subsample of 14 objects is mostly at `4 < z < 7` with median `z = 5`. Because the current marker centres cannot be associated with individual source names or Table 5 rows, `z = 5.0` is used as the paper’s representative redshift for every fallback row. The `Mbh` and `Mstar` values are the current digitised marker centres converted from logarithmic masses, not direct original-table values. The paper’s object-level uncertainties cannot be assigned to these unidentified rows and are therefore not plotted.

## Juodzbalis+2026

- CSV row 45 is QSO1 from [Juodzbalis+2026](https://arxiv.org/abs/2508.21748), at `z = 7.04`. The main text and Methods around Figure 4 give the conservative host stellar-mass upper limit `Mstar < 2 × 10^7 M☉` and the direct MOKA3D black-hole estimate `log10(MBH/M☉) = 7.7 ± 0.3`. The CSV stores `2 × 10^7 M☉` as the numerical upper-limit position and `10^7.7 M☉` for the black hole. The upper-limit meaning and black-hole error are recorded here only.

## Kokorev+2023

- CSV row 46 is the UNCOVER broad-line AGN from [Kokorev+2023](https://arxiv.org/abs/2308.11610). Section 3.1 gives `z = 8.502 ± 0.003`; Section 4.3 gives `log10(MBH/M☉) = 8.17 ± 0.42`; and the abstract/stellar-mass discussion gives `log10(Mstar/M☉) < 8.3` at the 95th percentile. The CSV stores `10^8.17 M☉` and `10^8.3 M☉`, with the latter retained as a numerical upper-limit position. These are direct source-paper values rather than the old Juodzbalis digitised coordinates.

## Maiolino+2024

- CSV rows 47--48 are two line-width estimates for GN-z11 from [Maiolino+2024](https://arxiv.org/abs/2305.12492), at the spectroscopic `z = 10.603`. The Figure 4 discussion uses the extended disk stellar mass `Mstar = 8 × 10^8 M☉`. The black-hole-mass Methods section gives two estimates, approximately `1.4 × 10^6 M☉` and `1.6 × 10^6 M☉`; each is retained as a separate row. The paper quotes about 0.3 dex uncertainty for the black-hole estimate. These are direct source-paper values.

## Stone+2024

- CSV rows 49--53 are the five Stone comparison markers currently digitised in Juodzbalis+2026 Figure 4. The reference is [Stone+2024](https://arxiv.org/abs/2310.18395), which describes the five-quasar sample as spanning approximately `z = 5--7`. The current marker centres cannot be matched unambiguously to the five named quasars and their Table 1/Table 3 values, so the CSV uses the middle of that quoted range, `z = 6.0`, for every row. The `Mbh` and `Mstar` values are explicitly marked fallback coordinates converted from the Juodzbalis+2026 digitisation. The source paper’s individual mass errors and upper limits remain in its tables but are not assigned to unidentified fallback rows.

## Ubler+2023

- CSV row 54 is GS_3073 from [Ubler+2023](https://arxiv.org/abs/2302.06647), at `z = 5.55`. Section 4.1/Table 2 gives `log10(MBH/M☉) = 8.2 ± 0.4`; Section 4.2 gives the fiducial host stellar mass `log10(Mstar/M☉) = 9.52 ± 0.13`. The CSV stores the corresponding linear central values. These are direct source-paper values, and the quoted errors are retained here only.

## Yue+2024

- CSV rows 55--57 are the three Yue comparison markers currently digitised in Juodzbalis+2026 Figure 4. The reference is [Yue+2024](https://arxiv.org/abs/2309.04614), whose quasar sample spans `5.9 < z < 7.1` (abstract and Table 1). Because the current marker centres cannot be matched to individual Table 1/Table 3 quasars, the CSV uses the middle of that range, `z = 6.5`, for every row. The `Mbh` and `Mstar` values are explicitly marked fallback coordinates converted from the Juodzbalis+2026 digitisation; the source-paper uncertainties and host upper-limit flags cannot be assigned safely to the unidentified fallback rows.
