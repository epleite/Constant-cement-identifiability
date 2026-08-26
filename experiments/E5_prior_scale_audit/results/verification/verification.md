# Verification

Result: **44/44 checks passed**.

| Check | Status | Detail |
|---|---|---|
| group_row_count | PASS | rows=32 |
| individual_row_count | PASS | rows=35 |
| design_row_count | PASS | rows=32 |
| scale_definition_count | PASS | rows=7 |
| structural_mode_count | PASS | rows=4 |
| static_group_row_count | PASS | rows=40 |
| static_individual_row_count | PASS | rows=45 |
| static_design_row_count | PASS | rows=40 |
| static_scale_definition_count | PASS | rows=9 |
| group_metrics_finite | PASS | columns=['lambda_min', 'lambda_max', 'spectral_ratio', 'lambda_min_gain', 'Vcem_lnCn_correlation', 'aligned_1pct_lambda_min_gain'] |
| positive_information | PASS | lambda_min range=0.00809171--0.0127717 |
| valid_correlations | PASS | rho range=-0.999639---0.997411 |
| aligned_discrepancy_never_increases_gain | PASS | all grouped scenarios |
| static_group_metrics_finite | PASS | static and combined geometry columns |
| static_positive_information | PASS | static lambda-min range=0.00103563--0.00782493 |
| static_aligned_discrepancy_never_increases_gain | PASS | all static grouped scenarios |
| static_weak_directions_normalized | PASS | max norm error=2.220e-16 |
| aligned_discrepancy_recomputed_with_static_direction | PASS | weak-direction Vcem-component span=1.983e-02 |
| baseline_gain_reproduced | PASS | max error=0.000e+00 |
| baseline_spectral_ratio_reproduced | PASS | max error=2.635e-17 |
| baseline_target_correlation_reproduced | PASS | max error=0.000e+00 |
| baseline_aligned_gain_reproduced | PASS | baseline E3 target-aligned control |
| static_sweep_baseline_reproduced | PASS | baseline rows=5 |
| design_pair_stable | PASS | best pair=5+7.5 MPa in all scenarios |
| static_design_pair_stable | PASS | best pair=5+7.5 MPa in all static-prior scenarios |
| best_exceeds_second_best | PASS | minimum margin=3.329e-04 |
| static_best_exceeds_second_best | PASS | minimum margin=7.855e-04 |
| factor_two_gain_bounded | PASS | range=2.90604--3.57377 |
| factor_two_correlation_extreme | PASS | minimum |rho|=0.997652 |
| factor_two_aligned_gain_near_unity | PASS | range=1.20334--1.20811 |
| static_factor_two_gain_bounded | PASS | range=2.38854--5.54318 |
| static_factor_two_correlation_extreme | PASS | minimum |rho|=0.998075 |
| static_factor_two_aligned_gain_limited | PASS | range=1.07824--1.67796 |
| structural_contrast_reproduced | PASS | shared=633.776165, expanded=3.3511375 |
| prior_precision_scaling_equivalence | PASS | max absolute matrix error=6.821e-13 |
| static_prior_precision_scaling_equivalence | PASS | max absolute matrix error=5.684e-14 |
| figure_exists_Fig_prior_scale_sensitivity_png | PASS | bytes=415423 |
| figure_exists_Fig_prior_scale_sensitivity_pdf | PASS | bytes=35361 |
| figure_exists_Fig_static_prior_scale_sensitivity_png | PASS | bytes=490065 |
| figure_exists_Fig_static_prior_scale_sensitivity_pdf | PASS | bytes=37662 |
| figure_exists_Fig_combined_prior_scale_audit_png | PASS | bytes=483920 |
| figure_exists_Fig_combined_prior_scale_audit_pdf | PASS | bytes=33212 |
| combined_pdf_single_page | PASS | pages=1 |
| combined_pdf_contains_both_prior_blocks | PASS | four expected panel titles present |
