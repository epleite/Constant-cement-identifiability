# Constant-cement experiment E2: Stability and hierarchy

This package tests whether the factored constant-cement coordinate

\[
q_\star=\ln(C_n/C_{n0})+A\ln(V_{\rm cem}/V_0)
+\Gamma\ln[(\phi_c-V_{\rm cem})/(\phi_c-V_0)]
\]

is stable under within-trajectory uncertainty, transports between the two related Hugin trajectories, and offers a better hierarchical sharing assumption than a common coordination number.

The package vendors the exact E1/RPIA forward-model snapshot. The transfer-aware observable scales, target scales, nuisance scales, priors, constant-cement Scheme 1, and nominal `phi_c=0.40` are frozen.

## Run

```bash
python scripts/run_e2.py
```

For a reduced deterministic smoke run:

```bash
python scripts/run_e2.py --quick
```

The reference run uses 400 primary 20 m moving-block replicates, 120 replicates for each block-length sensitivity, and 300 train-only LOTO replicates in each direction. Increasing the Monte Carlo count would refine empirical quantiles but would not increase the roughly six block-length units supported by either trajectory. Publication text therefore treats the reported percentile ranges as conditional resampling diagnostics with uncertain finite-sample coverage, not as population-level confidence intervals.

## Outputs

- `results/tables/`: complete bootstrap replicates, summaries, LOTO profiles, and hierarchical fits.
- `results/figures/`: publication-ready PNG/PDF figures.
- `results/summary.json`: machine-readable headline results.
- `results/RESULTS.md`: scientific interpretation and limitations.
- `results/verification/`: automated checks.

The bootstrap intervals are conditional on the two available Hugin trajectories. With only two trajectory clusters, they do not estimate between-trajectory population uncertainty.
