# Chen+2026 Fig. 5a seed-mass-function data

This directory contains the persistent reference data used for the Chen+2026 comparison in `Fig.10_BHSMF.pdf`. The source is panel 5a of

`/home/subonan/_papers/A two-phase model of galaxy formation- IV. Seeding and growing SMBHs in DM halos_Chen+2026.pdf`

and is identified as arXiv:2509.03283v3. A search of the local arXiv source package, its vector `seed-mass-func.pdf`, and the public two-phase-model repository found no usable machine-readable Paper-IV seed table. The CSV is therefore a vector-path digitisation of panel 5a, with the extraction audit recorded in `chen2026_fig05a_digitisation.json`.

Panel 5a shows the cumulative $z=20$ seed mass function. The plotted quantity follows Chen+2026 Eq. (7),

$$
\Phi(M_{\rm BH,seed},z)=\frac{1}{V_{\rm u}}\frac{dN_{\rm seed}(>z)}{d\log_{10}M_{\rm BH,seed}},
$$

in units of $\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}$, where $N_{\rm seed}(>z)$ counts seeds formed at redshift at least $z$. The CSV stores the horizontal coordinate as $\log_{10}(M_{\rm BH,seed}/M_{\odot})$ and stores $\Phi$ as a positive linear density; the plotting code converts the former to solar masses for its logarithmic mass axis.

The eight curve roles are the central All seeds curve, its lower and upper grey support boundaries, Pop-III sub-Eddington, Pop-III Eddington, Fast halo ($\gamma_v\geq3$), LW halo ($J_{\rm LW,21}\geq7.5$), and Pop-II. The Fast-halo and LW-halo curves retain their source dashed and solid orange styles, respectively.

The grey All-seeds region in the source figure extends to the positive plotting floor at $\Phi=10^{-6}\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}$; it is plotted support on the log axis rather than a separately reported statistical confidence interval. Consequently, the lower-envelope rows in the CSV describe that digitised support floor, while the upper boundary follows the black All-seeds trace. This source-plot limitation is distinct from the digitisation/calibration uncertainty recorded in the JSON audit. No analytic fit or smoothing is applied.

# Chen+2026 Fig. 6 seeding-history data

`chen2026_fig06_seeding_history.csv` stores the vector-path digitisation of the first-column panels 6a and 6d on PDF page 10 of the same Chen+2026 paper. Panel 6a contains the seeding-rate density $d n_{\rm seed}/d\log_{10}(1+z)$ in $\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}$; panel 6d contains the cumulative seeding density $n_{\rm seed}(>z)$ in $\mathrm{Mpc}^{-3}$. The CSV uses $x=\log_{10}(1+z)$ and retains the four Fig. 12 comparison roles: All seeds, Pop-III (sub-Eddington), Pop-III (Eddington), and Pop-II.

Fast halo ($\gamma_v\geq3$) and LW halo ($J_{\rm LW,21}\geq7.5$) were deliberately excluded from the visible Fig. 12 comparison. Their inspected vector paths and the omission decision are recorded in `chen2026_fig06_digitisation.json`. The reference traces are unsmoothed, use no analytic fit or extrapolation below their native low-redshift end near $z=5$, and are source-figure digitisation rather than uncertainty bands.
