# Prior-scale sensitivity of the pressure-design result

## Question

Does the reported 3.35-fold gain in nuisance-adjusted minimum information depend materially on the assumed one-standard-deviation scales of either the prospective pressure-model nuisances or the nine static nuisances?

## Design of the audit

The frozen E3 expanded-fabric Jacobians and the 5 + 7.5 MPa design were retained. Prior standard deviations were multiplied by 0.25--4, with 0.5--2 treated as the main factor-of-two interpretation band. The sweep separates fabric-state priors (soft/stiff ln Cn and soft critical porosity), stress/calibration priors, and the assumed stiff-end cement-volume prior. All seven pressure nuisances were also varied together. Every scenario was then re-optimized over all candidate pressure pairs. A second calculation added the pre-existing 1% RMS discrepancy aligned with the static weak target direction. The same sweep was applied to the nine static nuisances, individually and in five physical groups. For those cases, both the static reference Schur complement and the combined pressure Schur complement were recomputed under the altered prior; the aligned discrepancy followed the corresponding static weak direction.

Changing a physical prior scale by a factor m was implemented as a prior precision factor 1/m^2 on the frozen standardized nuisance column. Direct Jacobian rescaling agrees to maximum absolute matrix errors of 6.82e-13 for a pressure nuisance and 5.68e-14 for a static nuisance.

## Main result

At the declared baseline, the gain is 3.35, the spectral ratio is 0.00067, and the local target correlation is -0.9984. With 1% target-aligned discrepancy the gain is 1.21.

Within the factor-of-two band:

| Prior group varied | Gain range | Spectral-ratio range | Correlation range | Gain with 1% aligned discrepancy |
|---|---:|---:|---:|---:|
| Fabric state | 3.23--3.38 | 0.000435--0.000997 | [-0.9990, -0.9977] | 1.2060--1.2071 |
| Stress/calibration | 3.08--3.49 | 0.000634--0.000741 | [-0.9985, -0.9983] | 1.2053--1.2076 |
| Scenario endpoint | 3.24--3.38 | 0.000668--0.000677 | [-0.9984, -0.9984] | 1.2059--1.2072 |
| All pressure nuisances | 2.91--3.57 | 0.000322--0.000938 | [-0.9992, -0.9978] | 1.2033--1.2081 |

The original 5 + 7.5 MPa pair remained optimal in 32 of 32 grouped scale scenarios. Thus the selected design is stable over this sweep even though the exact gain is prior-conditional.

## Which individual priors matter most?

The following ranking uses the absolute gain difference between a half-scale and a double-scale prior while all other priors remain at their baselines:

| Nuisance | Gain at 0.5x | Gain at 2x | Span |
|---|---:|---:|---:|
| `log_stress_shear_scale` | 3.49 | 3.09 | 0.4 |
| `stiff_cement_volume_shift` | 3.38 | 3.24 | 0.146 |
| `ln_soft_cn_offset` | 3.38 | 3.25 | 0.134 |
| `logP_calibration` | 3.35 | 3.34 | 0.00964 |
| `ln_stiff_cn_offset` | 3.35 | 3.35 | 0.00524 |
| `log_stress_bulk_scale` | 3.35 | 3.35 | 0.00491 |
| `soft_phic_shift` | 3.35 | 3.35 | 6.46e-05 |

No individual scale restores nominal separability. The local target correlation remains close to -1, and the aligned-discrepancy gain remains near 1.2 throughout the scientifically interpretable band.

The spectral ratio is not monotonic in every sweep because widening a prior can reduce the strong eigenvalue as well as the weak eigenvalue; it must therefore be read jointly with absolute lambda-min gain and target correlation, not as a stand-alone improvement score.

## Static-nuisance prior sensitivity

For static-prior changes, the gain denominator is not frozen at its original value: the static adjusted information is recomputed with the same altered prior used in the combined experiment. Within the factor-of-two band:

| Static prior group varied | Gain range | Spectral-ratio range | Correlation range | Gain with 1% aligned discrepancy |
|---|---:|---:|---:|---:|
| State-variable biases | 3.25--3.56 | 0.000645--0.000684 | [-0.9985, -0.9984] | 1.1828--1.2433 |
| Solid composition/moduli | 2.54--5.01 | 0.000545--0.000794 | [-0.9987, -0.9981] | 1.0946--1.6780 |
| Fluid properties | 3.33--3.4 | 0.000666--0.000672 | [-0.9984, -0.9984] | 1.2027--1.2146 |
| Packing reference | 3.31--3.52 | 0.00064--0.000748 | [-0.9985, -0.9982] | 1.1493--1.2261 |
| All static nuisances | 2.39--5.54 | 0.000418--0.000816 | [-0.9990, -0.9981] | 1.0782--1.5970 |

The 5 + 7.5 MPa pair remained optimal in 40 of 40 grouped static-prior scenarios.

When all static priors are varied together, the factor-of-two gain range is 2.39--5.54, but the ratio must be interpreted with care: the static denominator changes from 0.00185 to 0.00587, while the combined absolute minimum eigenvalue spans only 0.0102--0.014. The larger gain under looser static priors therefore does not mean more absolute information; it largely reflects a smaller static baseline.

One-at-a-time static-prior influence, ranked by the gain span between 0.5x and 2x:

| Static nuisance | Gain at 0.5x | Gain at 2x | Span |
|---|---:|---:|---:|
| `log_clay_mod_scale` | 2.56 | 4.36 | 1.8 |
| `f_kf_shift` | 3.24 | 3.56 | 0.316 |
| `phic_pack_shift` | 3.31 | 3.52 | 0.203 |
| `sw_bias` | 3.29 | 3.49 | 0.202 |
| `vsh_bias` | 3.3 | 3.41 | 0.105 |
| `brine_salinity_shift` | 3.33 | 3.38 | 0.052 |
| `GOR_shift` | 3.35 | 3.36 | 0.00775 |
| `phi_bias` | 3.35 | 3.35 | 0.00309 |
| `log_cement_mod_scale` | 3.35 | 3.35 | 0.000226 |

## Scale dependence versus structural dependence

| Fabric-link model | Gain | Spectral ratio | Correlation | 1% aligned-discrepancy gain |
|---|---:|---:|---:|---:|
| `shared` | 634 | 0.0822 | -0.3634 | 301 |
| `fixed` | 402 | 0.0503 | -0.6816 | 189 |
| `nuisance` | 368 | 0.0577 | -0.7007 | 181 |
| `expanded_nuisance` | 3.35 | 0.00067 | -0.9984 | 1.21 |

This comparison is decisive for interpretation. The drop from the shared-link result to the expanded-fabric result cannot be reproduced by merely widening or tightening the expanded-model priors. The model modes also change which fabric quantities inherit derivatives with respect to the nominal target Cn. The large collapse is therefore primarily structural; prior scale controls the conditional value within that structural assumption.

## What can be claimed

- Robust: plausible factor-of-two scale perturbations do not turn the pressure experiment into a well-separated inversion; target correlation remains extreme and target-aligned discrepancy leaves no more than a 1.68-fold gain across the grouped pressure and static sweeps.
- Robust: the 5 + 7.5 MPa pair is the selected pair throughout the grouped 0.25--4 scale sweeps for both pressure and static nuisances over the tested candidate set.
- Conditional: 3.35 should be reported as approximately 3.4 under the declared priors. Pressure-prior perturbations give 2.9--3.6, whereas grouped static-prior perturbations broaden the factor-two range to 2.4--5.5, partly through movement of the static denominator.
- Not established: this audit does not provide external empirical calibration of the prior scales. It tests consequence, not provenance.

## Statistical scope

The calculation remains a local Gaussian Schur-complement audit. Very wide multipliers, especially for critical porosity, are mathematical stress tests and may extend beyond the locally meaningful physical support. That is why the factor-of-two band is the primary interpretation.
