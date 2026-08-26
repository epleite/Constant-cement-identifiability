# Methods

## Frozen model and metric

The constant-cement Scheme 1 forward model is the exact vendored RPIA/E1 implementation. The E1 transfer-aware standard deviations are reconstructed once from the two original cross-trajectory residuals and then frozen. The target scales are `0.015` for cement volume and `0.20` for `ln Cn`; nuisance priors remain independent unit Gaussians in standardized coordinates.

The bootstrap operating point is recalibrated with the original RPIA `Vp`/`Vs` objective, matching the E1 `pooled_RPIA` definition. At every replicate, the target and nuisance Jacobians and the Schur-adjusted geometry are recomputed under the frozen metric.

## Bootstrap

Rows are resampled jointly, preserving `(phi, vsh, sw, Vp, Vs, rho)`. Sampling is stratified by trajectory and restores the original sample counts. The primary non-circular moving-block bootstrap uses five consecutive samples (20 m). Each resample draws six blocks per trajectory (with the last block truncated for 19A) from 25 and 26 overlapping starting positions. The ratios `N/L` are only 5.8 and 6.0, a rough count of the block-length units supported by each trajectory. Sensitivity analyses use IID sampling and 3- and 10-sample blocks (12 and 40 m). Blocks never wrap from the bottom to the top of a trajectory or cross trajectories.

The Monte Carlo replicate count refines the empirical resampling distribution but does not increase its effective data support. Reported 2.5--97.5 percentile ranges are conditional on the selected dependence scale and the two Hugin trajectories; finite-sample coverage is uncertain, and dependence beyond the tested block lengths may be omitted. Boundary solutions and failures are retained and reported rather than silently removed.

## Leave-one-trajectory-out transport

The unrecentered level test is

\[
\Delta q_{i\rightarrow j}=(\ell_j-\ell_i)
+A_i\ln(v_j/v_i)
+\Gamma_i\ln[(\phi_c-v_j)/(\phi_c-v_i)].
\]

The shape-only test recenters the held-out numerical ridge at its own operating point and applies the training coefficients:

\[
\ell_{\rm pred}(v)=\ell_{0,j}
-A_i\ln(v/v_{0,j})
-\Gamma_i\ln[(\phi_c-v)/(\phi_c-v_{0,j})].
\]

Because this recentering uses the held-out point, it tests finite ridge geometry rather than blind level prediction. Raw predictions are compared with the fixed-nuisance structural profile; adjusted predictions are compared with a nonlinear nuisance-profiled structural profile using the same prior.

## Hierarchical models

Four models are fitted jointly to the two trajectories:

- `separate`: four target parameters `(v1, ell1, v2, ell2)`;
- `shared_qstar`: three target parameters `(v1, v2, qstar)`;
- `shared_Cn`: three target parameters `(v1, v2, ell_shared)`;
- `pooled_theta`: two target parameters `(v, ell)`.

`shared_qstar` and `shared_Cn` have equal target dimension. `A` and `Gamma` are fixed during their comparison. The primary data objective uses all `Vp`, `Vs`, and density observations with frozen transfer-aware scales. The adjusted fit adds one shared nine-dimensional standardized nuisance vector and counts its prior once.

The nonlinear observed-data comparison is supplemented by local quadratic comparisons based on each trajectory's raw Gram matrix and Schur-adjusted Gram matrix. These local costs preserve the E1 definition of nuisance adjustment.

No likelihood-ratio p-values, Bayes factors, AIC, or BIC are interpreted: the transfer-aware covariance is empirical, depth samples are correlated, boundaries can be active, and only two trajectories are available.

## Interpretation guardrail

Stability of `A` and `Gamma` means that the geometry of the weak ridge is stable under the specified resampling. It does not imply separate identifiability of cement volume and coordination number. Likewise, a shared `Cn` constraint can improve conditional precision by deleting a ridge direction without creating new sensitivity information.
