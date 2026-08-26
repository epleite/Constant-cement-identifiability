# E3 prior-scale sensitivity audit

This directory addresses the manuscript-review question of whether the
pressure-design headline result is carried by the assumed Gaussian prior
scales for either prospective pressure-model nuisances or the nine static
nuisances inherited from RPIA.

The audit preserves the frozen E3 expanded-fabric model, static baseline,
measurement/discrepancy model, candidate pressure set, and primary
5 + 7.5 MPa design. It varies only prior standard deviations. It also
re-optimizes every candidate pair under each grouped scale scenario.

## Scientific separation of priors

- **Fabric state:** compliant/stiff `ln Cn` offsets and compliant critical
  porosity.
- **Stress/calibration:** pressure calibration and bulk/shear stress-response
  amplitudes.
- **Scenario endpoint:** the assumed stiff-end cement-volume shift.
- **All pressure nuisances:** all seven prospective nuisance scales together.

The static sweep separately treats state-variable biases, solid composition and
moduli, fluid properties, the packing reference, and all nine static
nuisances. For these scenarios the static Schur-complement baseline is
recomputed under the same altered prior; the gain denominator is not held
fixed artificially.

The 0.5--2 multiplier interval is the principal factor-of-two sensitivity
band. The 0.25--4 endpoints are broader mathematical stress tests and should
not be interpreted as equally plausible physical priors.

## Run

From the repository root:

```bash
python3 experiments/E5_prior_scale_audit/scripts/run_prior_scale_sweep.py
python3 experiments/E5_prior_scale_audit/scripts/verify_prior_scale_sweep.py
```

The script locates the frozen E3 package within this repository. An alternative
path may be supplied with `E3_ROOT`.

## Outputs

- `results/REPORT.md`: scientific interpretation and manuscript implications.
- `results/summary.json`: machine-readable headline values and ranges.
- `results/tables/prior_scale_definitions.csv`: physical meaning and baseline
  scales.
- `results/tables/prior_scale_group_sweep.csv`: grouped fixed-design sweep.
- `results/tables/prior_scale_one_at_a_time.csv`: individual nuisance sweep.
- `results/tables/prior_scale_design_reoptimization.csv`: best pair for every
  grouped scenario.
- `results/tables/prior_scale_structural_context.csv`: scale sensitivity placed
  beside the four E3 fabric-link models.
- `results/tables/static_prior_scale_definitions.csv`: physical definitions of
  the nine static nuisance scales.
- `results/tables/static_prior_scale_group_sweep.csv`: grouped static-prior
  sensitivity with recomputed static baselines.
- `results/tables/static_prior_scale_one_at_a_time.csv`: individual static
  nuisance sweep.
- `results/tables/static_prior_scale_design_reoptimization.csv`: pair
  reoptimization for each grouped static-prior scenario.
- `results/figures/Fig_prior_scale_sensitivity.{png,pdf}`: four-panel summary.
- `results/figures/Fig_static_prior_scale_sensitivity.{png,pdf}`: four-panel
  static-prior summary.
- `results/figures/Fig_combined_prior_scale_audit.{png,pdf}`: compact 2x2
  manuscript figure containing pressure-prior gain, static-prior gain,
  target-aligned-discrepancy controls, and the all-static denominator audit.
- `results/verification/`: independent consistency checks.

## Interpretation boundary

This test quantifies sensitivity to prior scales; it does not calibrate those
scales externally. It also remains within the local Gaussian
Schur-complement framework of E3. Its direct conclusion is that the exact
3.35-fold value is conditional. Factor-of-two pressure-prior changes give
2.91--3.57, whereas grouped static-prior changes give 2.39--5.54 because the
static denominator also moves. Persistent high target correlation, limited
gain under target-aligned discrepancy, and selection of the 5 + 7.5 MPa pair
remain stable across the tested scale range.
