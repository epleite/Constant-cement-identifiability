from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import e3_analysis as a  # noqa: E402
import e3_model as pm  # noqa: E402


checks: list[dict[str, object]] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": str(detail)})


def close(actual: float, expected: float, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    return bool(np.isclose(actual, expected, atol=atol, rtol=rtol))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    summary_text = (ROOT / "results" / "summary.json").read_text()
    summary = json.loads(summary_text)
    e1_expected = json.loads(
        (ROOT / "vendor" / "e1_v1" / "reference" / "expected_summary.json").read_text()
    )
    baseline = a.load_baseline()
    tables = ROOT / "results" / "tables"

    rng = np.random.default_rng(17)
    n_pressures, n_samples = 2, 3
    synthetic_target = rng.normal(size=(n_pressures * 2 * n_samples, 2))
    synthetic_nuisance = rng.normal(size=(n_pressures * 2 * n_samples, 4))
    sigma = 0.007
    whitened_target, whitened_nuisance = a.whiten_pressure_blocks(
        synthetic_target,
        synthetic_nuisance,
        n_pressures,
        n_samples,
        sigma,
        shared_reference=True,
    )
    covariance = sigma**2 * (
        np.eye(n_pressures) + np.ones((n_pressures, n_pressures))
    )
    inverse_covariance = np.linalg.inv(covariance)
    target_blocks = synthetic_target.reshape(n_pressures, 2, n_samples, 2)
    nuisance_blocks = synthetic_nuisance.reshape(n_pressures, 2, n_samples, 4)
    direct_tt = np.zeros((2, 2))
    direct_tn = np.zeros((2, 4))
    direct_nn = np.zeros((4, 4))
    for prop in range(2):
        for sample in range(n_samples):
            jt = target_blocks[:, prop, sample, :]
            jn = nuisance_blocks[:, prop, sample, :]
            direct_tt += jt.T @ inverse_covariance @ jt
            direct_tn += jt.T @ inverse_covariance @ jn
            direct_nn += jn.T @ inverse_covariance @ jn
    check(
        "analytic whitening target Gram",
        np.allclose(whitened_target.T @ whitened_target, direct_tt, rtol=1e-12),
    )
    check(
        "analytic whitening target-nuisance cross product",
        np.allclose(whitened_target.T @ whitened_nuisance, direct_tn, rtol=1e-12),
    )
    check(
        "analytic whitening nuisance Gram",
        np.allclose(whitened_nuisance.T @ whitened_nuisance, direct_nn, rtol=1e-12),
    )
    _, schur = a.schur_geometry(whitened_target, whitened_nuisance)
    joint = np.block(
        [
            [direct_tt, direct_tn],
            [direct_tn.T, direct_nn + np.eye(4)],
        ]
    )
    covariance_theta = np.linalg.inv(joint)[:2, :2]
    check(
        "Schur-Woodbury equivalence",
        np.allclose(np.linalg.inv(covariance_theta), schur, rtol=1e-11),
    )

    check("sample count 19A", len(baseline.wells["19A"]) == 29)
    check("sample count BT2", len(baseline.wells["BT2"]) == 30)
    check("pooled sample count", len(baseline.pooled) == 59)
    check(
        "E1 Vcem anchor",
        close(float(baseline.theta[0]), e1_expected["pooled_operating_point"]["Vcem_fraction"]),
        baseline.theta[0],
    )
    check(
        "E1 Cn anchor",
        close(float(np.exp(baseline.theta[1])), e1_expected["pooled_operating_point"]["Cn"]),
        np.exp(baseline.theta[1]),
    )
    check(
        "E1 adjusted lambda-min anchor",
        close(
            float(np.linalg.eigvalsh(baseline.gram_adjusted)[0]),
            e1_expected["pooled_operating_point"]["adjusted_lambda_min"],
        ),
    )
    check(
        "E1 A anchor",
        close(
            baseline.coordinate["A_adjusted"],
            e1_expected["physics_factored_coordinate"]["A_adjusted"],
        ),
    )
    check(
        "E1 Gamma anchor",
        close(
            baseline.coordinate["Gamma_adjusted"],
            e1_expected["physics_factored_coordinate"]["Gamma_adjusted"],
        ),
    )
    for key, relative in [
        ("E1_expected_summary_sha256", Path("vendor/e1_v1/reference/expected_summary.json")),
        ("E2_summary_sha256", Path("reference/E2_summary.json")),
        (
            "E2_bootstrap_replicates_sha256",
            Path("reference/E2_bootstrap_replicates.csv"),
        ),
        ("rpia_core_sha256", Path("vendor/e1_v1/vendor/rpia_v1/rpia_core.py")),
    ]:
        actual = sha256(ROOT / relative)
        check(f"provenance hash {key}", actual == summary["provenance"][key], actual)

    frozen = pm.rc.forward(baseline.pooled, a.MODEL, baseline.theta)
    extended_reference = pm.pressure_extended_forward(
        baseline.pooled,
        baseline.theta,
        pm.REFERENCE_PRESSURE_MPA,
        **a.FABRIC_CONFIGS[a.PRIMARY_FABRIC_MODE],
        soft_lncn_reference=float(baseline.theta[1]),
    )
    reference_error = float(np.max(np.abs(frozen - extended_reference)))
    check("pressure extension recovers frozen model at P0", reference_error < 1e-10, reference_error)

    audit = pd.read_csv(tables / "E3_pressure_independence_audit.csv")
    audit_change = float(
        audit[["max_abs_delta_Vp_mps", "max_abs_delta_Vs_mps", "max_abs_delta_rho_gcc"]]
        .to_numpy()
        .max()
    )
    check("frozen pressure audit is exactly zero", audit_change == 0.0, audit_change)
    check(
        "summary frozen no-go matches audit",
        summary["exact_no_go"]["max_pressure_induced_change_standard_model"] == audit_change,
    )
    no_go = pd.read_csv(tables / "E3_no_go_repetition.csv")
    check("no-go adds no raw direction", float(no_go.raw_strong_direction_rotation_deg.max()) < 1e-12)
    check("no-go target rank stays two", set(no_go.target_rank) == {2})
    check("no-go flag stays zero", set(no_go.new_sensitivity_direction) == {0})

    curves = pd.read_csv(tables / "E3_pressure_curves.csv")
    frozen_curve = curves[curves.model == "frozen_constant_cement"]
    check("frozen Vp curve is flat", float(np.abs(frozen_curve.Vp_change_percent).max()) == 0.0)
    check("frozen Vs curve is flat", float(np.abs(frozen_curve.Vs_change_percent).max()) == 0.0)
    pressure_curve = curves[curves.model == "pressure_extension_shared_Cn"]
    at_reference = pressure_curve[
        np.isclose(pressure_curve.pressure_mpa, pm.REFERENCE_PRESSURE_MPA)
    ]
    check(
        "pressure curve is anchored at the RPIA pressure",
        len(at_reference) == 1
        and float(np.abs(at_reference.Vp_change_percent).max()) < 1e-10,
    )
    check("pressure extension is non-flat", float(np.abs(pressure_curve.Vs_change_percent).max()) > 0.5)

    weights = pd.read_csv(tables / "E3_patchy_weights.csv")
    check("bulk bounding weights are physical", bool(weights.W_K.between(0.0, 1.0).all()))
    check("shear bounding weights are physical", bool(weights.W_G.between(0.0, 1.0).all()))
    weight_stress = pd.read_csv(tables / "E3_bounding_weight_stress.csv")
    check(
        "bounding-weight stress metrics are finite",
        bool(np.isfinite(weight_stress.select_dtypes("number")).all().all()),
    )

    design_grid = pd.read_csv(tables / "E3_pressure_design_grid.csv")
    metric_columns = [
        "lambda_min",
        "lambda_max",
        "spectral_ratio",
        "condition_number",
        "lambda_min_gain",
        "worst_sd_reduction",
    ]
    check(
        "design grid has finite metrics",
        bool(np.isfinite(design_grid[metric_columns]).all().all()),
    )
    check("all adjusted eigenvalues are positive", bool((design_grid.lambda_min > 0).all()))
    pairs = design_grid[design_grid.n_additional_pressures == 2]
    check("pressure pairs are strictly ordered", bool((pairs.pressure_1_mpa < pairs.pressure_2_mpa).all()))
    primary_rows = design_grid[
        (design_grid.fabric_mode == a.PRIMARY_FABRIC_MODE)
        & np.isclose(design_grid.state_log_sigma, a.PRIMARY_STATE_LOG_SIGMA)
        & (design_grid.n_additional_pressures == 2)
    ]
    primary_best = primary_rows.loc[primary_rows.lambda_min.idxmax()]
    selected = tuple(summary["primary_design"]["pressures_mpa"])
    check("E-optimal primary pair is 5 and 7.5 MPa", selected == (5.0, 7.5), selected)
    check(
        "summary pair matches direct grid optimum",
        selected == (float(primary_best.pressure_1_mpa), float(primary_best.pressure_2_mpa)),
    )
    check(
        "summary lambda-min matches design grid",
        close(summary["primary_design"]["lambda_min"], float(primary_best.lambda_min)),
    )
    check(
        "summary gain matches design grid",
        close(summary["primary_design"]["lambda_min_gain"], float(primary_best.lambda_min_gain)),
    )
    check("primary gain exceeds one", summary["primary_design"]["lambda_min_gain"] > 1.0)
    check("primary spectral ratio remains below one", 0.0 < summary["primary_design"]["spectral_ratio"] < 1.0)
    check("summary JSON contains no non-standard NaN tokens", "NaN" not in summary_text)
    check(
        "selected pair touches candidate boundary",
        summary["primary_design"]["touches_candidate_boundary"] is True,
    )
    check(
        "selected pair does not span full candidate range",
        summary["primary_design"]["spans_full_candidate_range"] is False,
    )

    discrepancy = pd.read_csv(tables / "E3_model_discrepancy_sensitivity.csv")
    full_basis = discrepancy[discrepancy.basis == "intercept_plus_porosity_plus_clay"].sort_values(
        "trajectory_discrepancy_percent"
    )
    check("full discrepancy gain decreases monotonically", bool((np.diff(full_basis.lambda_min_gain) <= 1e-12).all()))
    primary_discrepancy = discrepancy[np.isclose(discrepancy.trajectory_discrepancy_percent, 1.0)]
    by_basis = primary_discrepancy.set_index("basis").lambda_min_gain
    check("porosity discrepancy does not increase gain", by_basis["intercept_plus_porosity"] <= by_basis["intercept"])
    check(
        "clay discrepancy does not increase gain",
        by_basis["intercept_plus_porosity_plus_clay"] <= by_basis["intercept_plus_porosity"],
    )
    check("primary gain remains positive at 5% discrepancy", float(full_basis.iloc[-1].lambda_min_gain) > 0.99)

    modes = summary["fabric_link_ablation"]
    check("independent soft fabric is more conservative than shared", modes["nuisance"]["lambda_min_gain"] < modes["shared"]["lambda_min_gain"])
    check("independent soft fabric is more conservative than fixed", modes["nuisance"]["lambda_min_gain"] < modes["fixed"]["lambda_min_gain"])
    check(
        "expanded fabric adjustment is most conservative",
        modes["expanded_nuisance"]["lambda_min_gain"]
        < modes["nuisance"]["lambda_min_gain"],
    )
    check(
        "expanded fabric gain remains modest",
        modes["expanded_nuisance"]["lambda_min_gain"] < 10.0,
    )

    reference_table = pd.read_csv(tables / "E3_reference_pressure_sensitivity.csv")
    check("reference-pressure control has 12 rows", len(reference_table) == 12)
    check(
        "reference-pressure gains are finite",
        bool(np.isfinite(reference_table.lambda_min_gain).all()),
    )
    aligned_table = pd.read_csv(tables / "E3_target_aligned_discrepancy.csv")
    primary_aligned = aligned_table[
        aligned_table.fabric_mode == a.PRIMARY_FABRIC_MODE
    ].set_index("target_aligned_discrepancy_percent_rms")
    check(
        "target-aligned discrepancy reduces apparent gain",
        primary_aligned.loc[1.0, "lambda_min_gain"]
        < primary_aligned.loc[0.0, "lambda_min_gain"],
    )
    check(
        "one-percent target-aligned spectral ratio is below static baseline",
        summary["target_aligned_discrepancy_primary"]["1_percent"]["spectral_ratio"]
        < summary["operating_point"]["baseline_adjusted_spectral_ratio"],
    )
    trajectory_table = pd.read_csv(tables / "E3_trajectory_specific_designs.csv")
    check("trajectory-specific control has four rows", len(trajectory_table) == 4)
    check(
        "trajectory-specific primary gains are finite",
        bool(
            np.isfinite(
                trajectory_table[
                    trajectory_table.fabric_mode == a.PRIMARY_FABRIC_MODE
                ].lambda_min_gain
            ).all()
        ),
    )

    fluids = pd.read_csv(tables / "E3_multi_fluid_control.csv")
    check("multi-fluid control is weak", float(fluids.lambda_min_gain.max()) < 3.0)
    candidates = pd.read_csv(tables / "E3_candidate_observations.csv")
    qstar_gain = float(candidates.loc[candidates.candidate == "q_star", "lambda_min_gain"].iloc[0])
    check("q-star proxy adds essentially no weak direction", qstar_gain < 1.01, qstar_gain)
    response_gain = float(
        candidates.loc[candidates.candidate == "pressure_response_proxy", "lambda_min_gain"].iloc[0]
    )
    check("pressure-response proxy exceeds q-star proxy", response_gain > qstar_gain)

    finite = pd.read_csv(tables / "E3_finite_difference_stability.csv")
    check(
        "target finite differences are stable",
        float(finite["target_relative_error_to_1e-4"].max()) < 1e-4,
    )
    check(
        "nuisance finite differences are stable",
        float(finite["nuisance_relative_error_to_1e-4"].max()) < 1e-4,
    )

    bootstrap = pd.read_csv(tables / "E3_bootstrap_replicates.csv")
    expected_bootstrap_rows = 400 * 3 * 2
    check("bootstrap row count", len(bootstrap) == expected_bootstrap_rows, len(bootstrap))
    check("bootstrap has 400 unique replicates", bootstrap.replicate.nunique() == 400)
    check("bootstrap metrics are finite", bool(np.isfinite(bootstrap.select_dtypes("number")).all().all()))
    primary_boot = bootstrap[
        (bootstrap.fabric_mode == a.PRIMARY_FABRIC_MODE)
        & (bootstrap.design == "best_pair")
    ]
    median_gain = float(np.median(primary_boot.lambda_min_gain))
    check(
        "bootstrap median matches summary",
        close(
            median_gain,
            summary["bootstrap_primary"][a.PRIMARY_FABRIC_MODE][
                "lambda_min_gain"
            ]["median"],
        ),
        median_gain,
    )
    check("all primary bootstrap gains exceed one", bool((primary_boot.lambda_min_gain > 1.0).all()))

    selection_table = pd.read_csv(tables / "E3_conditional_design_selection.csv")
    check("conditional design selection row count", len(selection_table) == 300)
    check(
        "conditional design regrets are non-negative",
        bool((selection_table.full_sample_design_regret_fraction >= -1e-12).all()),
    )
    operating_table = pd.read_csv(
        tables / "E3_operating_point_design_sensitivity.csv"
    )
    check("nine operating-point controls", len(operating_table) == 9)
    check(
        "operating-point design gains are finite",
        bool(np.isfinite(operating_table.lambda_min_gain).all()),
    )

    profiles = pd.read_csv(tables / "E3_profile_widths.csv")
    profile_230 = profiles[np.isclose(profiles.threshold, 2.30)].set_index("objective")
    check(
        "pressure does not widen Vcem profile",
        profile_230.loc["combined_adjusted_objective", "Vcem_width_percentage_points"]
        <= profile_230.loc["static_adjusted_objective", "Vcem_width_percentage_points"]
        + 1e-12,
    )
    check(
        "pressure does not widen Cn profile",
        profile_230.loc["combined_adjusted_objective", "Cn_width"]
        <= profile_230.loc["static_adjusted_objective", "Cn_width"] + 1e-12,
    )
    check(
        "combined Vcem profile is boundary-censored",
        bool(profile_230.loc["combined_adjusted_objective", "Vcem_lower_censored"])
        and bool(profile_230.loc["combined_adjusted_objective", "Vcem_upper_censored"]),
    )
    check(
        "combined Cn profile is lower-bound censored",
        bool(profile_230.loc["combined_adjusted_objective", "Cn_lower_censored"]),
    )

    figures = sorted((ROOT / "results" / "figures").glob("*.png"))
    check("four PNG figures exist", len(figures) == 4, len(figures))
    for figure in figures:
        try:
            with Image.open(figure) as image:
                image.verify()
            check(f"valid PNG {figure.name}", True, figure.stat().st_size)
        except Exception as error:  # pragma: no cover - verification path
            check(f"valid PNG {figure.name}", False, repr(error))
    pdfs = sorted((ROOT / "results" / "figures").glob("*.pdf"))
    check("four PDF figures exist", len(pdfs) == 4, len(pdfs))
    for pdf in pdfs:
        header = pdf.read_bytes()[:5]
        check(f"valid PDF header {pdf.name}", header == b"%PDF-", header)

    csv_files = sorted(tables.glob("*.csv"))
    check("all expected CSV tables exist", len(csv_files) >= 16, len(csv_files))
    for csv_file in csv_files:
        table = pd.read_csv(csv_file)
        numeric = table.select_dtypes("number")
        check(
            f"no infinities in {csv_file.name}",
            bool((~np.isinf(numeric.to_numpy())).all()) if numeric.size else True,
        )

    failed = [item for item in checks if not item["passed"]]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "n_checks": len(checks),
        "n_passed": len(checks) - len(failed),
        "n_failed": len(failed),
        "checks": checks,
    }
    output_dir = ROOT / "results" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "E3_verification.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# E3 verification",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Passed {report['n_passed']} of {report['n_checks']} checks.",
        "",
    ]
    if failed:
        lines.extend(["## Failed checks", ""])
        lines.extend(f"- {item['name']}: {item['detail']}" for item in failed)
    else:
        lines.append("No failed checks.")
    (output_dir / "E3_verification.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "n_checks", "n_passed", "n_failed"]}))
    if failed:
        for item in failed:
            print("FAIL", item["name"], item["detail"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
