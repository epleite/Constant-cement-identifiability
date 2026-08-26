# Quantitative audit of non-linearity along the constant-cement ridge

## Question

The original values 633.776, 3.351, and 1.207 are local eigenvalue gains at
the pooled operating point.  Because the static `Vcem`--`Cn` ridge is long and
curved, a single linearization cannot establish that these values describe
the global inverse geometry.  This audit asks two different questions:

1. Does the fully non-linear, nuisance-profiled ridge contract?
2. How much does the *local* efficient-information gain vary when it is
   recalculated at MAP points along that non-linear ridge?

The answers must not be conflated.  A profile objective is a global
likelihood-geometry diagnostic; a pointwise Schur complement is a local
Gauss--Newton metric.

## Definitions

Let `theta=(Vcem,lnCn)`, let `z_s` denote the nine standardized static
nuisances, and let `z_p` denote the seven additional standardized pressure
nuisances.  The static structural reference is the frozen forward prediction
at the pooled point `theta0=(0.0125460,ln 5.4360)` with zero nuisances.

For every fixed `Vcem=v`, the static profile is

```text
Phi_s(v) = min_{lnCn,z_s} ||r_s(v,lnCn,z_s)||^2 + ||z_s||^2.
```

For the pressure experiment at 5 and 7.5 MPa, the combined profile is

```text
Phi_c(v) = min_{lnCn,z_s,z_p}
           ||(I + D D^T)^(-1/2) [r_s ; r_P]||^2
           + ||z_s||^2 + ||z_p||^2.
```

`D` contains the 1%-RMS trajectory/state/property discrepancy basis and, in
the aligned stress test, one extra 1%-RMS column fixed to the pooled static
weak target direction.  The low-rank square-root expression is algebraically
identical to profiling the Gaussian linear discrepancy coefficients with
unit-normal priors.  All physical nuisance effects are evaluated through the
full non-linear forward model and optimized numerically.

At each combined-profile MAP point, the local efficient Gauss--Newton metric
is recalculated as

```text
G_eff = Jt^T Jt - Jt^T Jn (Jn^T Jn + I)^(-1) Jn^T Jt.
```

The pointwise gain is the ratio between the combined and static minimum
eigenvalues evaluated at that same target state and static-nuisance centre.
No matrices from different ridge points are summed or averaged.

## Numerical design

- 32 `Vcem` values from 0.001 to 0.060, including the exact pooled point;
- continuous optimization of `lnCn` and 9 or 16 physical nuisances;
- target scales `(0.015,0.20)` and the frozen nuisance scales/priors;
- added differential observations at 5 and 7.5 MPa;
- state log standard deviation 0.005 and shared-reference covariance;
- generic model discrepancy of 1% total RMS;
- exact frozen E1 static model and frozen E3 pressure scenario.

## Results

### Exact recovery at the pooled point

The independent implementation reproduces all three E3 values to relative
error below `5e-11`:

| Scenario | E3 pooled gain | Recomputed |
|---|---:|---:|
| Shared nominal/compliant fabric | 633.776 | 633.776 |
| Expanded fabric nuisances | 3.351 | 3.351 |
| Expanded + 1% target-aligned discrepancy | 1.207 | 1.207 |

This eliminates an implementation mismatch as the source of any differences
away from the pooled point.

### Fully non-linear profile contraction

At `Delta Phi=2.30`, the static profile reaches both grid boundaries.  Its
`Vcem` width is therefore at least 5.90 percentage points, and the MAP ridge
spans `Cn=4.20--11.36`.

The shared-fabric experiment is qualitatively different.  Adaptive non-linear
root finding places its `Delta Phi=2.30` crossings at
`Vcem=0.8386--1.7878` percentage points, a width of **0.9492 percentage
points**; the endpoint MAP values are `Cn=5.943` and `5.158`.  The large local
gain therefore survives as genuine profile
contraction **when the shared-fabric link is enforced**.

At `Delta Phi=5.99`, the corresponding adaptive interval is
`Vcem=0.6354--2.1676` percentage points, with width 1.5322 percentage points.
The former coarse linear interpolation gave 0.9309 and 1.5172 percentage
points, respectively; it is retained in the machine-readable output but is
not used for headline reporting.

After the fabric variables are separated, neither pressure scenario closes
the ridge.  Both profiles remain boundary-censored over the full 0.1--6.0
percentage-point grid:

| Scenario | `Vcem` support at `Delta Phi=2.30` | MAP `Cn` span |
|---|---:|---:|
| Static | at least 5.90 percentage points | 7.15 |
| Shared fabric | 0.949 percentage points | 0.785 between adaptive crossing MAPs |
| Expanded fabric | at least 5.90 percentage points | 7.20 |
| Expanded + aligned discrepancy | at least 5.90 percentage points | 7.15 |

The maximum expanded-fabric objective over the entire grid is only 0.400;
with target-aligned discrepancy it is only 0.182.  Both are far below 2.30.
Thus the 3.35-fold local increase is real but is not large enough to produce
globally resolved microstructural separation.

### Pointwise local information along each MAP ridge

On points satisfying both `Delta Phi <= 2.30` and the convex bounding-weight
domain:

| Scenario | Minimum | Reference value | Maximum | Weak-direction rotation |
|---|---:|---:|---:|---:|
| Shared fabric | 394 | 634 pooled | 1,159 | 15.7--41.2 deg |
| Expanded fabric | 2.50 | 3.35 pooled | 4.82 | 0.002--0.765 deg |
| Expanded + aligned discrepancy | 1.19 | 1.21 pooled | 2.53 | 0.001--0.168 deg |

The shared range was recomputed on 41 uniformly spaced `Vcem` values between
the adaptive crossings, plus the exact pooled point.  Because medians and
quantiles depend on the measure chosen along a curved ridge, the recommended
reporting is the robust range **394--1,159**, with **634 at the pooled point**,
rather than a grid-dependent median.  The pooled 3.35-fold number is
also not a numerical accident, but it varies materially and mostly rescales
the existing weak direction rather than rotating it.  The target-aligned
case is generally close to unity.  Its upper value occurs at the lower
`Vcem=0.001` boundary, where contact-radius derivatives become steep; it
should not be read as broad recovery of the ridge.

## Interpretation for the paper

The non-linear audit does **not** overturn the main scientific conclusion; it
makes it sharper:

- shared fabric can create a strong and globally visible contraction, but the
  result is conditional on that structural link;
- after fabric freedom and model discrepancy are admitted, the local gain
  remains finite yet the full non-linear ridge stays unresolved;
- therefore `lambda_min` gain at one operating point must be reported as a
  local design diagnostic, not as proof of global parameter separability.

The manuscript should add this experiment and replace any implication that
3.35-fold means the ambiguity was broken.  A defensible statement is:

> The expanded-nuisance design increases the pointwise weak information by
> 2.50--4.82 along the nuisance-MAP ridge (3.35 at the pooled point), but its
> fully non-linear profile remains boundary-censored; local information gain
> does not translate into resolved global separability.

This result directly supports the proposed criterion that a remediation must
create sensitivity diversity that survives nuisance adjustment *and* remains
effective over the non-linear region supported by the inverse problem.

## Multi-start and reverse-path stability

The relatively large `least_squares` optimality diagnostics in a few original
fits were traced to finite-difference noise near a very shallow profile, not
to competing minima that alter the interpretation.  A systematic audit used:

- controlled refits from every stored MAP;
- continuation from high to low and low to high `Vcem` at every static,
  expanded-fabric, and aligned-discrepancy point, and at every shared-fabric
  point inside its convex validity domain;
- four independent starts at profile extremes, largest-optimality points,
  threshold brackets, and representative interior points;
- 698 stored optimization runs in total.

Every new optimization reported success.  Relative to the original paths,
the largest objective improvements and absolute `Cn` shifts were:

| Profile | Maximum objective improvement | Maximum `|Delta Cn|` |
|---|---:|---:|
| Static | 2.07e-4 | 0.0101 |
| Shared, convex domain | 3.97e-8 | 1.72e-5 |
| Expanded fabric | 7.97e-4 | 0.0101 |
| Expanded + aligned discrepancy | 3.10e-4 | 0.0080 |

These small changes leave every supported/censored classification unchanged.
Recomputing the pointwise metrics at the best multi-start MAPs gives
2.502--4.820 for expanded fabric and 1.186--2.531 for the aligned case, so the
rounded ranges **2.50--4.82** and **1.19--2.53** remain valid.

## Limitations

- These are structural synthetic-reference profiles, not fits to observed
  multi-pressure Hugin data.
- The pressure extension remains the E3 heuristic scenario generator.
- A MAP/profile calculation conditions on the stated nuisance priors; it does
  not marginalize posterior mass and is not a Bayes factor.
- `G_eff` is a pointwise Gauss--Newton metric, not the exact observed Hessian.
- Censored expanded/static widths remain lower bounds, not finite confidence
  intervals.  The finite shared widths use adaptive roots, while the refined
  pointwise shared range uses 41 uniform `Vcem` values plus the pooled point.
- The aligned discrepancy direction is fixed at the pooled weak direction; it
  is not adversarially reoriented at every point.
- Shared-fabric extrapolations outside the convex bounding-weight domain are
  retained in the CSV and drawn dashed, but excluded from supported pointwise
  summaries.

## Verification

`verify.py` includes the original checks plus multi-start stability, adaptive
crossing, refined-domain, and post-refinement pointwise checks.  These cover
exact reproduction of the three
pooled gains, recovery of the frozen E1 non-linear static profile at common
anchors, the analytical linear-discrepancy profiling identity, convex-domain
checks, finite-difference stability (`maximum relative change 6.38e-6`), and
complete PNG decoding.
