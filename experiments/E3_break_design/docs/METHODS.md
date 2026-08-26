# Methods

## 1. Frozen-model audit

The static baseline is the exact constant-cement Scheme 1 implementation used
in RPIA E1–E2. Its targets are

\[
\theta=(V_{\rm cem},\ln C_n),
\]

scaled by `(0.015, 0.20)`. The pooled operating point is
`Vcem = 0.0125460` and `Cn = 5.4360`. The frozen code computes an effective
pressure but does not use it in the constant-cement branch. Consequently,

\[
\partial f_{\rm cc}/\partial P_{\rm eff}=0.
\]

Identical-state replication was retained as a precision control. Its raw
target direction rotates by exactly zero; changes after nuisance adjustment
come from replication and shared-prior geometry, not a new constitutive
direction.

## 2. Prospective pressure scenario

The pressure experiment is represented by an Avseth–Skjei-inspired bounding
average anchored at `P0 = 39 MPa`, the effective pressure of the frozen RPIA
environment. For bulk modulus,

\[
K(P)=K_{\rm cc}+(1-W_K)\{K_{\rm soft}(P)-K_{\rm soft}(P_0)\},
\]

\[
W_K=\frac{K_{\rm cc}-K_{\rm soft}(P_0)}
{K_{\rm stiff}-K_{\rm soft}(P_0)},
\]

with an analogous construction for shear modulus. `Ksoft,Gsoft` use the
friable-sand/Hertz–Mindlin branch; `Kstiff,Gstiff` use a pressure-insensitive
10%-cement constant-cement endpoint. The extension recovers the frozen model
exactly at `P0`.

This is a heuristic scenario generator. It is not claimed to reproduce a
specific published pressure implementation or observed Hugin pressure data.

## 3. Fabric-link ablation

Four configurations locate the source of the apparent new direction:

1. `shared`: compliant-contact `Cn` equals the nominal target `Cn`.
2. `fixed`: compliant-contact `Cn` is fixed at its operating-point value; this
   is a local oracle control, not an externally measured quantity.
3. `nuisance`: compliant-contact `Cn` has an independent log-offset nuisance.
4. `expanded_nuisance`: compliant-contact `Cn`, stiff-bound `Cn`, and the
   compliant critical porosity are independently nuisance-adjusted.

The fourth case is primary. It does not claim complete fabric independence;
mineral properties and the bounding-average construction remain shared.

## 4. Nuisances and discrepancy

The frozen RPIA nuisances are retained. Additional one-sigma pressure-model
scales are:

| Nuisance | Scale |
|---|---:|
| Log pressure calibration | 0.10 |
| Log bulk stress-response scale | 0.25 |
| Log shear stress-response scale | 0.25 |
| Stiff-end cement-volume shift | 0.015 |
| Compliant-contact `ln Cn` offset | 0.20 |
| Stiff-bound `ln Cn` offset | 0.10 |
| Compliant critical-porosity shift | 0.02 |

The primary discrepancy has 1% total RMS amplitude. For every added pressure,
elastic property, and trajectory it spans an intercept plus orthonormalized
porosity and clay-fraction trends. A separate control adds a discrepancy
exactly aligned with the static weak target direction.

Bounding weights are convex for the pooled operating point and its one-sigma
local nuisance perturbations. Wider E2 bootstrap states can leave `[0,1]`;
operating-point OED sensitivity is therefore restricted to states in the
convex validity domain.

## 5. Differential observations and covariance

For every added state, the observables are differential log velocities,

\[
\Delta\ln V_p(P),\qquad\Delta\ln V_s(P),
\]

relative to the 39 MPa state. Each absolute velocity has primary log standard
deviation `0.005`. Differences sharing a reference have covariance

\[
C_\Delta=\sigma^2(I+\mathbf{1}\mathbf{1}^{T}),
\]

so two differences have correlation 0.5. Whitening is performed before
information matrices are formed. The implementation is verified against
direct generalized least squares to machine precision.

## 6. Nuisance-adjusted information

For target and nuisance Jacobians `Jtheta` and `Jeta`, the efficient local
information is

\[
G_{\rm adj}=G_{\theta\theta}-G_{\theta\eta}
(G_{\eta\eta}+C_\eta^{-1})^{-1}G_{\eta\theta}.
\]

E-optimality maximizes `lambda_min(Gadj)`. Pressure-prior ablations vary only
the physical pressure-nuisance block; frozen nuisances and discrepancy priors
remain fixed. Schur and joint-covariance/Woodbury calculations agree in the
verification suite.

Candidate added pressures are 5, 7.5, 10, 12.5, 15, 17.5, 22.5, 25, 27.5,
30, 35, 40, 45, 50, 55, and 60 MPa. The primary pair is the best within this
finite candidate set, not a continuous or cost-optimal experiment.

## 7. Robustness and controls

- 400 trajectory-stratified, non-circular 20 m moving-block replicates evaluate
  the selected design conditionally at the E1 operating point.
- The pair is re-optimized in 100 conditional replicates at block lengths of
  12, 20, and 40 m.
- Nine representative, non-bound E2 bootstrap operating points inside the
  convex weight domain receive fully recomputed static and pressure Jacobians.
- Each Hugin trajectory is analysed separately with its own calibrated target.
- Pressure anchors of 10, 20, and 39 MPa are compared.
- Single-state, multi-fluid, target-aligned discrepancy, finite-difference, and
  bounding-weight controls are included.

The finite target grid evaluates the target forward model nonlinearly. Nuisance
effects are locally linearized at the pooled operating point and analytically
adjusted. It is therefore a nonlinear-target/local-linear-nuisance diagnostic,
not a fully nonlinear nuisance profile. Censored intervals are flagged.
