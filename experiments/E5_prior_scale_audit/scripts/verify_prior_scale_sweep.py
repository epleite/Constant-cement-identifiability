#!/usr/bin/env python3
"""Independent consistency checks for the prior-scale sensitivity audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
VERIFY = RESULTS / "verification"


def main() -> None:
    grouped = pd.read_csv(TABLES / "prior_scale_group_sweep.csv")
    individual = pd.read_csv(TABLES / "prior_scale_one_at_a_time.csv")
    designs = pd.read_csv(TABLES / "prior_scale_design_reoptimization.csv")
    scales = pd.read_csv(TABLES / "prior_scale_definitions.csv")
    structural = pd.read_csv(TABLES / "prior_scale_structural_context.csv")
    static_grouped = pd.read_csv(TABLES / "static_prior_scale_group_sweep.csv")
    static_individual = pd.read_csv(TABLES / "static_prior_scale_one_at_a_time.csv")
    static_designs = pd.read_csv(
        TABLES / "static_prior_scale_design_reoptimization.csv"
    )
    static_scales = pd.read_csv(TABLES / "static_prior_scale_definitions.csv")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    check("group_row_count", len(grouped) == 32, f"rows={len(grouped)}")
    check("individual_row_count", len(individual) == 35, f"rows={len(individual)}")
    check("design_row_count", len(designs) == 32, f"rows={len(designs)}")
    check("scale_definition_count", len(scales) == 7, f"rows={len(scales)}")
    check("structural_mode_count", len(structural) == 4, f"rows={len(structural)}")
    check("static_group_row_count", len(static_grouped) == 40, f"rows={len(static_grouped)}")
    check("static_individual_row_count", len(static_individual) == 45, f"rows={len(static_individual)}")
    check("static_design_row_count", len(static_designs) == 40, f"rows={len(static_designs)}")
    check("static_scale_definition_count", len(static_scales) == 9, f"rows={len(static_scales)}")

    numeric_columns = [
        "lambda_min",
        "lambda_max",
        "spectral_ratio",
        "lambda_min_gain",
        "Vcem_lnCn_correlation",
        "aligned_1pct_lambda_min_gain",
    ]
    finite = np.isfinite(grouped[numeric_columns].to_numpy(dtype=float)).all()
    check("group_metrics_finite", bool(finite), f"columns={numeric_columns}")
    check(
        "positive_information",
        bool((grouped.lambda_min > 0).all() and (grouped.lambda_max > 0).all()),
        f"lambda_min range={grouped.lambda_min.min():.6g}--{grouped.lambda_min.max():.6g}",
    )
    check(
        "valid_correlations",
        bool((grouped.Vcem_lnCn_correlation.abs() <= 1.0 + 1e-12).all()),
        f"rho range={grouped.Vcem_lnCn_correlation.min():.6g}--{grouped.Vcem_lnCn_correlation.max():.6g}",
    )
    check(
        "aligned_discrepancy_never_increases_gain",
        bool(
            (
                grouped.aligned_1pct_lambda_min_gain
                <= grouped.lambda_min_gain + 1e-10
            ).all()
        ),
        "all grouped scenarios",
    )
    static_finite = np.isfinite(
        static_grouped[
            numeric_columns + ["static_lambda_min", "static_lambda_max"]
        ].to_numpy(dtype=float)
    ).all()
    check(
        "static_group_metrics_finite",
        bool(static_finite),
        "static and combined geometry columns",
    )
    check(
        "static_positive_information",
        bool(
            (static_grouped.static_lambda_min > 0).all()
            and (static_grouped.lambda_min > 0).all()
        ),
        f"static lambda-min range={static_grouped.static_lambda_min.min():.6g}--{static_grouped.static_lambda_min.max():.6g}",
    )
    check(
        "static_aligned_discrepancy_never_increases_gain",
        bool(
            (
                static_grouped.aligned_1pct_lambda_min_gain
                <= static_grouped.lambda_min_gain + 1e-10
            ).all()
        ),
        "all static grouped scenarios",
    )
    static_weak_norm = np.sqrt(
        static_grouped.static_weak_direction_scaled_Vcem**2
        + static_grouped.static_weak_direction_scaled_lnCn**2
    )
    weak_direction_span = float(
        static_grouped.static_weak_direction_scaled_Vcem.max()
        - static_grouped.static_weak_direction_scaled_Vcem.min()
    )
    check(
        "static_weak_directions_normalized",
        bool(np.allclose(static_weak_norm, 1.0, atol=1e-10)),
        f"max norm error={np.max(np.abs(static_weak_norm - 1.0)):.3e}",
    )
    check(
        "aligned_discrepancy_recomputed_with_static_direction",
        bool(
            weak_direction_span > 1e-5
            and (
                static_grouped.aligned_1pct_raw_alignment_rms_per_scaled_parameter
                > 0
            ).all()
        ),
        f"weak-direction Vcem-component span={weak_direction_span:.3e}",
    )

    baseline_rows = grouped[np.isclose(grouped.prior_sd_multiplier, 1.0)]
    expected_gain = 3.3511375010382625
    expected_ratio = 0.0006703996686359264
    expected_rho = -0.9984188232644072
    expected_aligned = 1.2068996463319068
    check(
        "baseline_gain_reproduced",
        bool(np.allclose(baseline_rows.lambda_min_gain, expected_gain, atol=2e-10)),
        f"max error={np.max(np.abs(baseline_rows.lambda_min_gain - expected_gain)):.3e}",
    )
    check(
        "baseline_spectral_ratio_reproduced",
        bool(np.allclose(baseline_rows.spectral_ratio, expected_ratio, atol=2e-12)),
        f"max error={np.max(np.abs(baseline_rows.spectral_ratio - expected_ratio)):.3e}",
    )
    check(
        "baseline_target_correlation_reproduced",
        bool(np.allclose(baseline_rows.Vcem_lnCn_correlation, expected_rho, atol=2e-10)),
        f"max error={np.max(np.abs(baseline_rows.Vcem_lnCn_correlation - expected_rho)):.3e}",
    )
    check(
        "baseline_aligned_gain_reproduced",
        bool(
            np.allclose(
                baseline_rows.aligned_1pct_lambda_min_gain,
                expected_aligned,
                atol=2e-10,
            )
        ),
        "baseline E3 target-aligned control",
    )
    static_baseline_rows = static_grouped[
        np.isclose(static_grouped.prior_sd_multiplier, 1.0)
    ]
    check(
        "static_sweep_baseline_reproduced",
        bool(
            np.allclose(
                static_baseline_rows.lambda_min_gain, expected_gain, atol=2e-10
            )
            and np.allclose(
                static_baseline_rows.aligned_1pct_lambda_min_gain,
                expected_aligned,
                atol=2e-10,
            )
        ),
        f"baseline rows={len(static_baseline_rows)}",
    )

    all_primary = bool(
        (designs.best_pressure_1_mpa == 5.0).all()
        and (designs.best_pressure_2_mpa == 7.5).all()
    )
    check("design_pair_stable", all_primary, "best pair=5+7.5 MPa in all scenarios")
    all_static_primary = bool(
        (static_designs.best_pressure_1_mpa == 5.0).all()
        and (static_designs.best_pressure_2_mpa == 7.5).all()
    )
    check(
        "static_design_pair_stable",
        all_static_primary,
        "best pair=5+7.5 MPa in all static-prior scenarios",
    )
    check(
        "best_exceeds_second_best",
        bool((designs.best_minus_second_best_lambda_min > 0).all()),
        f"minimum margin={designs.best_minus_second_best_lambda_min.min():.3e}",
    )
    check(
        "static_best_exceeds_second_best",
        bool((static_designs.best_minus_second_best_lambda_min > 0).all()),
        f"minimum margin={static_designs.best_minus_second_best_lambda_min.min():.3e}",
    )

    factor_two = grouped[grouped.prior_sd_multiplier.between(0.5, 2.0)]
    check(
        "factor_two_gain_bounded",
        bool(factor_two.lambda_min_gain.between(2.9, 3.6).all()),
        f"range={factor_two.lambda_min_gain.min():.6g}--{factor_two.lambda_min_gain.max():.6g}",
    )
    check(
        "factor_two_correlation_extreme",
        bool((factor_two.Vcem_lnCn_correlation.abs() > 0.997).all()),
        f"minimum |rho|={factor_two.Vcem_lnCn_correlation.abs().min():.6g}",
    )
    check(
        "factor_two_aligned_gain_near_unity",
        bool(factor_two.aligned_1pct_lambda_min_gain.between(1.19, 1.22).all()),
        f"range={factor_two.aligned_1pct_lambda_min_gain.min():.6g}--{factor_two.aligned_1pct_lambda_min_gain.max():.6g}",
    )
    static_factor_two = static_grouped[
        static_grouped.prior_sd_multiplier.between(0.5, 2.0)
    ]
    check(
        "static_factor_two_gain_bounded",
        bool(static_factor_two.lambda_min_gain.between(2.38, 5.55).all()),
        f"range={static_factor_two.lambda_min_gain.min():.6g}--{static_factor_two.lambda_min_gain.max():.6g}",
    )
    check(
        "static_factor_two_correlation_extreme",
        bool((static_factor_two.Vcem_lnCn_correlation.abs() > 0.998).all()),
        f"minimum |rho|={static_factor_two.Vcem_lnCn_correlation.abs().min():.6g}",
    )
    check(
        "static_factor_two_aligned_gain_limited",
        bool(
            static_factor_two.aligned_1pct_lambda_min_gain.between(1.07, 1.68).all()
        ),
        f"range={static_factor_two.aligned_1pct_lambda_min_gain.min():.6g}--{static_factor_two.aligned_1pct_lambda_min_gain.max():.6g}",
    )

    shared = structural[structural.fabric_mode == "shared"].iloc[0]
    expanded = structural[structural.fabric_mode == "expanded_nuisance"].iloc[0]
    check(
        "structural_contrast_reproduced",
        bool(
            math.isclose(shared.lambda_min_gain, 633.7761651279503, abs_tol=2e-8)
            and math.isclose(expanded.lambda_min_gain, expected_gain, abs_tol=2e-10)
        ),
        f"shared={shared.lambda_min_gain:.9g}, expanded={expanded.lambda_min_gain:.9g}",
    )

    equivalence_error = float(
        summary["scale_parameterization_equivalence_max_abs_error"]
    )
    check(
        "prior_precision_scaling_equivalence",
        equivalence_error < 1e-10,
        f"max absolute matrix error={equivalence_error:.3e}",
    )
    static_equivalence_error = float(
        summary["static_scale_parameterization_equivalence_max_abs_error"]
    )
    check(
        "static_prior_precision_scaling_equivalence",
        static_equivalence_error < 1e-10,
        f"max absolute matrix error={static_equivalence_error:.3e}",
    )

    for filename in (
        "Fig_prior_scale_sensitivity.png",
        "Fig_prior_scale_sensitivity.pdf",
        "Fig_static_prior_scale_sensitivity.png",
        "Fig_static_prior_scale_sensitivity.pdf",
        "Fig_combined_prior_scale_audit.png",
        "Fig_combined_prior_scale_audit.pdf",
    ):
        path = RESULTS / "figures" / filename
        check(
            f"figure_exists_{path.stem}_{path.suffix[1:]}",
            path.is_file() and path.stat().st_size > 10_000,
            f"bytes={path.stat().st_size if path.exists() else 0}",
        )

    combined_pdf = RESULTS / "figures" / "Fig_combined_prior_scale_audit.pdf"
    combined_reader = PdfReader(combined_pdf)
    combined_text = "\n".join(page.extract_text() or "" for page in combined_reader.pages)
    expected_panel_titles = (
        "Pressure-model priors",
        "Static-model priors",
        "aligned model discrepancy",
        "All-static denominator audit",
    )
    check(
        "combined_pdf_single_page",
        len(combined_reader.pages) == 1,
        f"pages={len(combined_reader.pages)}",
    )
    check(
        "combined_pdf_contains_both_prior_blocks",
        all(title in combined_text for title in expected_panel_titles),
        "four expected panel titles present",
    )

    passed = sum(bool(item["passed"]) for item in checks)
    report = {
        "passed": passed == len(checks),
        "n_passed": passed,
        "n_checks": len(checks),
        "checks": checks,
    }
    VERIFY.mkdir(parents=True, exist_ok=True)
    (VERIFY / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Verification",
        "",
        f"Result: **{passed}/{len(checks)} checks passed**.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        markdown.append(
            f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |"
        )
    (VERIFY / "verification.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f"{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
