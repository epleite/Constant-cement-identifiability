# Scientific-content revision matrix

This file maps the nine substantive review points to tests and manuscript
changes. The quantitative analyses are isolated under `experiments/` and
their headline values are used in the manuscript only after automated and
independent checks.

| Review point | Required evidence | Manuscript consequence | Status |
|---|---|---|---|
| Local Schur geometry versus curved ridge | Fully non-linear physical-forward nuisance-MAP profiles, adaptive contour refinement, 698-run multistart/reverse-continuation audit, and pointwise efficient information | Replace a single-point headline by a pathwise/non-linear statement; distinguish finite contour geometry from local Gauss--Newton diagnostics | Complete: `nonlinear_ridge/` (19/19 checks) |
| Dependence on nuisance-prior scales | One-at-a-time and grouped scale sweeps for all 7 pressure and all 9 static nuisance priors | Report 2.39--5.54 as the factor-two grouped range, expose the moving denominator, and label 3.35 as conditional | Complete: `prior_scales/` (44/44 checks) |
| Nine static nuisances undefined | Complete names, baselines, scales, clipping intervals and forward-model roles | Static-nuisance table and pathway paragraph make the Schur projection reproducible | Complete in Methods |
| Finite form of q-star is an ansatz | Equal-complexity matched-tangent alternatives and held-out/transport comparison | Call q-star an adopted physics-factored model coordinate; identify robust contact curvature and weakly discriminated HS continuation | Complete: `qstar_appendix/` (13/13 checks) |
| Frozen pressure no-go overstated | Separate constitutive physics from code-path audit | Abstract, Methods, Results and Conclusions now label it an implementation-level negative control | Complete in text |
| Appendix A does not execute the promised test | Numerical bulk/shear contact-slope calculation and comparison with the empirical tangent | Worked decomposition accounts for approximately 24 contact minus 4 HS to recover the coefficient near 20.3; agreement with the stored bootstrap coefficients is labelled model-internal analytic--numerical self-consistency, not independent validation | Complete: Appendix A |
| Point estimate versus bootstrap median | Explicitly distinguish pooled estimate from the resampling distribution | Results reconcile pooled 3.35 with bootstrap median 3.39 | Complete in text |
| Excess numerical precision | Three-significant-figure reporting except for reproducibility anchors and near-unity transport comparisons | Headline numbers and correlations are rounded; retained extra digits are explicitly motivated | Complete in text |
| Gaussian nuisance priors plus frequentist bootstrap | Explicit statement that they represent different uncertainty layers | Methods distinguishes conditional nuisance absorption from sampling stability and states that bootstrap intervals are not posterior intervals | Complete in text |

## Second-round interpretive safeguards

| Review point | Required correction | Manuscript consequence | Status |
|---|---|---|---|
| Appendix agreement is partly definitional | Distinguish a chain-rule decomposition from an independent physical prediction | Section 5.1, Appendix title, table caption and numerical comparison now call the result analytic--numerical self-consistency and state that external contact observations are still required | Complete in text |
| Only about six block-length units per trajectory | Separate Monte Carlo replicate count from effective data support | Methods reports six block draws and $N/L=5.8,6.0$; all reported bootstrap bounds are conditional percentile ranges; Results, Limitations and Conclusions no longer imply nominal coverage | Complete in text |
| Rotation diagnostics use different reference frames | Define same-state pressure rotation separately from across-path static eigendirection change | Methods defines both angles; Results adds an explicit bridge before juxtaposing $0.77^\circ$ and $55.5^\circ$; Discussion and Conclusions use same-state wording | Complete in text |

## Statistical guardrail

An arithmetic integral or average of local Schur matrices along a ridge is not
treated as global information: points on that ridge are mutually exclusive
parameter states, not repeated observations. Finite contour geometry is
assessed with the fully nonlinear profiled objective. Local efficient information is audited
pointwise along the corresponding low-objective path and reported as a range.

## Frozen interpretive statement

- The raw local target geometry gives the empirical slope 20.3; nuisance
  adjustment rotates it slightly to approximately 20.0. These are no longer
  conflated in the Introduction or Appendix A.
- Under shared fabric, the synthetic profile closes at the fixed
  `DeltaPhi=2.30` two-target benchmark to approximately 0.95 percentage
  points in cement volume.
- With expanded fabric nuisances, the local gain is 3.35 at the pooled point
  and 2.50--4.82 along the tested MAP path, but no finite contour closure is
  resolved across the 5.90-percentage-point grid.
- The 3.35 value is prior-conditional: complete-block factor-two sweeps give
  2.91--3.57 for pressure-prior scales and 2.39--5.54 for static-prior scales.
  Extreme correlation and spectral ratios below `1e-3` persist.
