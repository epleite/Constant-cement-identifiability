# Non-linear ridge robustness audit

This isolated review test addresses whether the local nuisance-adjusted
information gains reported by E3 survive the strong curvature of the
constant-cement `Vcem`--`Cn` ridge.

It does **not** edit the manuscript or any frozen E1/E3 package. Those
packages are read as dependencies from this repository.

## Two deliberately separate calculations

1. `nonlinear_MAP_profiles.csv` contains fully non-linear penalized profile
   objectives.  At each fixed `Vcem`, `lnCn` and all applicable physical
   nuisances are optimized.  The 1%-RMS linear discrepancy coefficients are
   profiled analytically with their unit-normal priors.
2. `pointwise_efficient_information.csv` contains the Gauss--Newton efficient
   information recalculated at each nuisance-MAP point.  These matrices are
   local.  They are neither added nor averaged along the ridge and are not
   described as global information.

The three scenarios exactly reproduce, at the pooled point, the original E3
gains of approximately 634, 3.35, and 1.21.

## Run

From the repository root:

```bash
python experiments/E4_nonlinear_ridge_audit/run_nonlinear_ridge.py
python experiments/E4_nonlinear_ridge_audit/multistart_audit.py
python experiments/E4_nonlinear_ridge_audit/refine_shared_supported.py
python experiments/E4_nonlinear_ridge_audit/recompute_multistart_pointwise.py
python experiments/E4_nonlinear_ridge_audit/synchronize_outputs.py
python experiments/E4_nonlinear_ridge_audit/verify.py
```

The full run takes several minutes because every profile point optimizes a
non-linear forward model with up to 16 physical nuisance variables.

## Outputs

- `results/nonlinear_MAP_profiles.csv`: full non-linear target profiles and
  nuisance MAP values;
- `results/pointwise_efficient_information.csv`: local Schur metrics and
  convex-domain flags;
- `results/finite_difference_stability.csv`: derivative-step audit;
- `results/multistart_runs.csv`: stored starts and results for refits,
  bidirectional continuation, and independent starts;
- `results/multistart_best_profiles.csv`: best MAP at every audited point;
- `results/shared_dense_crossings.csv`: adaptive roots at `Delta Phi=2.30`
  and `5.99`;
- `results/shared_refined_supported_profile.csv` and
  `shared_refined_pointwise_information.csv`: 41-point uniform refinement
  between the shared `Delta Phi=2.30` crossings, plus the pooled point;
- `results/multistart_pointwise_information.csv`: expanded/aligned local
  metrics recomputed at their best multi-start MAPs;
- `results/summary.json`: machine-readable headline results;
- `results/figures/Fig_nonlinear_ridge_robustness.{png,pdf}`: four-panel
  scientific figure;
- `results/verification.json`: nineteen automated checks;
- `REPORT.md`: definitions, interpretation, and manuscript implications.

## Dependencies

Python 3.11 or newer with NumPy, pandas, SciPy, and Matplotlib. The exact
constant-cement and pressure-scenario implementations are imported from the
frozen E1 and E3 packages in this repository.
