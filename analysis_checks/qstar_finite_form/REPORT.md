# Scientific audit of the finite coordinate and Appendix slope self-consistency

## Bottom line

The current factored coordinate is strongly supported as an accurate **low-dimensional continuation** of the local ridge, but it is not mathematically unique. Along the complete pooled raw structural ridge, its worst coordinate drift is 1.803 per cent, compared with 0.0011 per cent for direct numerical integration of the local projection slope and 99.9 per cent for the first-order tangent. Two equal-coefficient alternatives that keep the logarithmic contact term but change the finite Hashin--Shtrikman continuation are almost as accurate (2.168 and 1.739 per cent). Replacing the logarithmic contact continuation by a linear one produces 97.4 per cent drift. Thus the data identify the need for a strongly curved/log-like contact term; they discriminate much less strongly among plausible finite forms for the smaller HS-path correction.

In the genuinely held-out level test, the worst absolute error is 3.29 per cent for the current factorization and 2.43 per cent for integrated local slopes, versus 542.7 per cent for the local tangent. The log-contact/linear-HS and log-contact/reciprocal-HS alternatives have worst errors of 3.54 and 3.03 per cent. All power-family variants use the same two local coefficients, the same operating points, the same weighting, and have the same first derivative at the training point (maximum finite-difference mismatch 8.03e-09). No coefficient was re-fitted to the held-out trajectory.

The endpoint calculation supplies a model-internal analytic decomposition and an implementation self-consistency check; it is not an independent validation of Scheme 1. At the pooled point, the exact Scheme-1 endpoint exponents give beta_K=24.515 and beta_G=24.002. Their observable-weighted contact projection is 24.120, so the endpoint physics accounts for the **contact component**, not 20.3 directly. The moving endpoint/HS path contributes -3.849, yielding 20.272; nuisance projection changes these to 24.138 and -4.176, yielding 19.962. Because `A` and `Gamma` are defined from this same projected derivative, agreement with the numerically stored coefficients verifies analytic--numerical consistency rather than supplying external physical confirmation.

## What is exact

1. In Scheme 1, `a_c` obeys `d ln(a_c) = (d ln(Vcem)-d ln(Cn))/4`.
2. For an endpoint modulus with exponent `m=d ln(S)/d ln(a_c)`, the constant-endpoint direction is `d ln(Cn) + [m/(4-m)] d ln(Vcem)=0`; therefore `beta=m/[V(4-m)]`.
3. At the pooled point, the analytic normal and effective shear endpoint exponents are represented by the sample medians `mK=0.940869` and `mG=0.925754`.
4. Propagating those endpoint derivatives through the modified HS path, Gassmann substitution, `Vp`/`Vs` weighting and (when requested) the Schur nuisance projection gives the contact and HS contributions listed above. Recomputing the entire decomposition for all 400 saved E2 bootstrap samples recovers the stored `A` and `Gamma` values with maximum absolute differences 2.219e-10. This near-identity is expected because both routes use the same projected local derivatives.
5. The density block has zero target-direction weight in the raw projection; the raw denominator weights are 42.8 per cent for `Vp` and 57.2 per cent for `Vs`. Density may still constrain nuisances in the adjusted calculation.

## What is a controlled approximation

1. `q_star` is obtained by freezing the two local chain-rule coefficients and integrating them with log factors. It is a two-coefficient finite ansatz, not a unique microscopic state variable.
2. The power-family alternatives use `[(x/x0)^r-1]/r`, whose log limit is `r=0`. They have identical value and derivative at the reference point and differ only at second and higher order. This makes the comparison a direct test of finite continuation rather than local fit quality.
3. The numerical integrated-projection curve recalculates the local metric along the path. It is the closest differential reconstruction of the local ridge, but it is more complex than a two-coefficient coordinate and is therefore a reference, not a same-complexity competitor.
4. The adjusted numerical integration follows the local Schur projection. It is not a fully non-linear Bayesian marginalization over nuisances; the held-out adjusted profile itself uses the E2 non-linear nuisance-MAP profile under the fixed priors.
5. Medians of endpoint `mK` and `mG` summarize small mineralogical variation. The exact observable-weighted projection is calculated from every sample and observable, rather than from the two medians alone.

## What is empirical

1. At a fixed pooled `V0`, the 20-m block bootstrap gives adjusted `A` median 0.3041 (conditional 2.5--97.5 percentile range 0.2901--0.3114) and adjusted `Gamma` median 1.6207 (1.5528--1.6680). Their combined slope has median 20.063 (19.136--20.517). With only about six block-length units per trajectory, these are conditional resampling diagnostics with uncertain finite-sample coverage, not between-well population uncertainty.
2. The held-out comparisons use the two related Hugin trajectories. They demonstrate internal transport, not external geological universality.
3. The current log-log form is not uniquely selected: keeping a log contact term while using linear or reciprocal finite HS terms changes held-out errors only slightly. Manuscript language should therefore say **a stable physics-factored coordinate** or **the selected low-order coordinate**, not imply uniqueness.

## Recommended manuscript changes for criticisms 4 and 6

1. Add a compact robustness paragraph/table reporting the matched-tangent power-family comparison and the integrated-slope reference. The defensible claim is that the log-contact structure is robust, while the exact finite HS factor is weakly resolved over this range.
2. Replace Appendix A with the executed endpoint calculation: report `mK`, `mG`, endpoint beta values, observable-weighted contact beta, HS correction and total, both raw and adjusted, explicitly as a model-internal decomposition.
3. Add the bootstrap self-consistency audit: endpoint/contact terms remain near 24, while the HS correction moves the total to the range containing 20.3. Do not present recovery of `(A,Gamma)` as an independent prediction; the same projected local derivatives define both routes. The endpoint exponents constrain the contact contribution underlying `A`, whereas `Gamma` comes from the moving-porosity/HS derivative.
4. Describe numerical slope integration as a differential reconstruction, not as another two-parameter model. Its near-exact in-sample result is expected because a profiled least-squares ridge has tangent `-G12/G22`; its useful evidence is the held-out transport result.

## Machine-readable outputs

- `results/tables/Q1_pooled_coordinate_curves.csv`
- `results/tables/Q1_pooled_coordinate_summary.csv`
- `results/tables/Q1_loto_level.csv`
- `results/tables/Q1_loto_shape_curves.csv`
- `results/tables/Q1_loto_shape_summary.csv`
- `results/tables/Q2_endpoint_point_closure.csv`
- `results/tables/Q2_observable_projection_weights.csv`
- `results/tables/Q2_bootstrap_endpoint_replicates.csv`
- `results/tables/Q2_bootstrap_endpoint_summary.csv`
- `results/figures/Fig_Q1_finite_coordinate_robustness.*`
- `results/figures/Fig_Q2_endpoint_slope_closure.*`
