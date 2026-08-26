# E3 results: can pressure break the constant-cement ridge?

## Main result

The frozen constant-cement model is exactly pressure independent. Repetition can increase precision, but pressure labels alone cannot create a new constitutive sensitivity direction. A legitimate multi-state experiment therefore requires an explicit pressure-sensitive branch.

Under the prospective Avseth–Skjei-inspired extension, the E-optimal pair among the tested candidates used **5 + 7.5 MPa** in addition to the 39 MPa reference state. The primary analysis uses expanded fabric nuisances, 0.50% per-state log-velocity uncertainty, the shared-reference covariance, and 1.0%-RMS trajectory-state discrepancy bases spanning intercept, porosity, and clay-fraction trends.

- Nuisance-adjusted lambda-min gain: **3.35x**.
- Reduction in the worst local standard deviation: **1.83x**.
- Adjusted spectral ratio after the experiment: **0.0007**.
- The unconstrained local-Gaussian marginal SDs remain 7.65 percentage points in Vcem and 1.541 in ln Cn; their correlation is -0.998.
- The best single added state was 5 MPa; the second state increased lambda-min by only 29.9%.
- The selected pair touches the lower boundary of the tested pressure set; pressures below 5 MPa were not ruled out.

## Why the fabric ablation matters

- Shared nominal and compliant-contact Cn: lambda-min gain 633.78x.
- Locally fixed compliant-contact Cn (an oracle control): gain 402.22x.
- Independent compliant-contact Cn with a prior: gain 368.43x.
- Expanded fabric nuisances, also separating stiff-bound Cn and compliant critical porosity: gain 3.35x.

The expanded control is the conservative primary result. It shows that most of the apparently new direction can be reabsorbed when the fabric of the compliant and stiff branches is not assumed known. Pressure alone therefore does not robustly repair the nominal Vcem-Cn ambiguity in this scenario model.

## Robustness

In the 400-replicate, trajectory-stratified 20 m moving-block bootstrap, the expanded-fabric best-pair lambda-min gain had median 3.39x and 95% interval [3.19, 3.55]x. Conditional re-optimization selected the full-sample pair in 100.0% of 20 m replicates; this does not include full recalibration of the pressure model.

The nonlinear target grid uses a nuisance tangent space frozen at the operating point. Its static Delta-Phi=2.30 support is boundary-censored: the Vcem width is at least 3.300 percentage points and the Cn width is at least 6.826. The combined support is also boundary-censored: its widths are at least 3.300 percentage points and 6.460, respectively. No Vcem contraction is resolved on the grid, and the apparent Cn contraction cannot be quantified precisely. These are local-linear nuisance diagnostics, not fully nonlinear profiles.

An additional target-aligned discrepancy of 1% RMS leaves a gain of 1.21x and produces a spectral ratio of 0.000248, below the static baseline of 0.000425. Experimental design must therefore account for model-error directions, not only generic smooth discrepancy.

The maximum fraction of samples outside the [0,1] weight interval is 0.0% for the pooled operating point and its one-sigma local nuisance scenarios, but reaches 100.0% across the wider E2 bootstrap stress test. Operating-point design sensitivity is therefore restricted to the convex validity domain; extrapolations beyond it are not interpreted.

## Interpretation

The scientifically defensible conclusion is negative but useful: the bounding-average pressure response can appear to strongly contract the ridge, yet that result is conditional on fabric-sharing assumptions. With expanded fabric adjustment, the ridge is not eliminated and may barely contract. A successful real experiment must either constrain the fabric variables independently or use a stress-response mechanism whose sensitivity remains distinct after those adjustments.

This is a prospective experimental-design result. It identifies a candidate pressure configuration and the fabric constraints required for a laboratory validation experiment; it does not prescribe a definitive acquisition and is not validation with observed pressure-dependent Hugin data.
