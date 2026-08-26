# Results index

This index maps the principal manuscript claims to frozen machine-readable
outputs and verification files.

| Claim | Primary output | Verification |
|---|---|---|
| Local weak direction and coefficient near 20.3 | `experiments/E1_discover_explain/results/summary.json` | `experiments/E1_discover_explain/results/verification/verification.json` |
| Contact and moving-endpoint decomposition | `experiments/E1_discover_explain/results/tables/E1_coordinate_coefficients.csv` | E1 verification plus `analysis_checks/qstar_finite_form/verification.json` |
| Finite-coordinate robustness | `analysis_checks/qstar_finite_form/results/tables/Q1_pooled_coordinate_summary.csv` | `analysis_checks/qstar_finite_form/verification.json` |
| 400-replicate moving-block stability | `experiments/E2_stability_hierarchy/results/tables/E2_bootstrap_summary.csv` | `experiments/E2_stability_hierarchy/results/verification/verification.json` |
| Leave-one-trajectory-out transport | `experiments/E2_stability_hierarchy/results/tables/E2_loto_level.csv` | E2 verification |
| Shared coordinate versus shared `Cn` | `experiments/E2_stability_hierarchy/results/tables/E2_hierarchical_comparison.csv` | E2 verification |
| Frozen pressure no-go | `experiments/E3_break_design/results/tables/E3_no_go_repetition.csv` | `experiments/E3_break_design/results/verification/E3_verification.json` |
| Fabric-link ablation, 634 to 3.35 | `experiments/E3_break_design/results/tables/E3_pressure_ablation.csv` | E3 verification |
| Target-aligned discrepancy, gain near 1.21 | `experiments/E3_break_design/results/tables/E3_target_aligned_discrepancy.csv` | E3 verification |
| Fully non-linear profile closure/censoring | `experiments/E4_nonlinear_ridge_audit/results/shared_dense_profile_widths.csv` and `nonlinear_MAP_profiles.csv` | `experiments/E4_nonlinear_ridge_audit/results/verification.json` |
| Pathwise gain and weak-direction rotation | `experiments/E4_nonlinear_ridge_audit/results/multistart_pointwise_information.csv` | E4 verification |
| 698-run initialization audit | `experiments/E4_nonlinear_ridge_audit/results/multistart_summary.json` | E4 verification |
| Pressure- and static-prior scale ranges | `experiments/E5_prior_scale_audit/results/tables/prior_scale_group_sweep.csv` and `static_prior_scale_group_sweep.csv` | `experiments/E5_prior_scale_audit/results/verification/verification.json` |

The PDF figures used in the manuscript are derived from these result trees.
CSV and JSON files are the authoritative numeric records; PDFs and PNGs are
presentation products.
