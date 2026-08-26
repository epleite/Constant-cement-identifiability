from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nonlinear_ridge as audit  # noqa: E402


def check(name: str, condition: bool, details: dict, checks: list[dict]) -> None:
    checks.append({"name": name, "passed": bool(condition), **details})


def main() -> None:
    baseline = audit.e3.load_baseline()
    profiles = pd.read_csv(HERE / "results" / "nonlinear_MAP_profiles.csv")
    information = pd.read_csv(
        HERE / "results" / "pointwise_efficient_information.csv"
    )
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    checks: list[dict] = []

    # Exact recovery of the three E3 headline calculations at the pooled point.
    for scenario in audit.SCENARIOS:
        result = summary["scenarios"][scenario.name]
        error = abs(float(result["relative_reproduction_error"]))
        check(
            f"pooled headline gain reproduced: {scenario.name}",
            error < 1.0e-8,
            {"absolute_relative_error": error},
            checks,
        )

    # The static fully non-linear profile must recover the independently frozen
    # E1 nuisance profile at the common lower endpoint, truth, and upper endpoint.
    frozen = pd.read_csv(
        audit.E1_ROOT / "results" / "tables" / "E1_ridge_validation.csv"
    )
    computed = profiles[profiles.profile == "static"]
    anchors = np.array(
        [
            float(frozen.Vcem_fraction.min()),
            float(baseline.theta[0]),
            float(frozen.Vcem_fraction.max()),
        ]
    )
    common = frozen[
        np.any(
            np.isclose(frozen.Vcem_fraction.to_numpy()[:, None], anchors[None, :]),
            axis=1,
        )
    ]
    cn_errors = []
    objective_errors = []
    for row in common.itertuples(index=False):
        match = computed.iloc[
            int(np.argmin(np.abs(computed.Vcem_fraction - row.Vcem_fraction)))
        ]
        cn_errors.append(abs(match.Cn_MAP - row.Cn_nuisance_profiled_structural))
        objective_errors.append(
            abs(match.objective_MAP - row.objective_nuisance_profiled_structural)
        )
    check(
        "static nonlinear profile reproduces E1",
        max(cn_errors) < 5.0e-3 and max(objective_errors) < 2.0e-4,
        {
            "maximum_Cn_absolute_error": float(max(cn_errors)),
            "maximum_objective_absolute_error": float(max(objective_errors)),
            "common_points": int(len(common)),
        },
        checks,
    )

    # Verify the exact Gaussian profiling identity used for linear discrepancy.
    generic = audit.generic_discrepancy_matrix(baseline)
    pooled_weak = audit.weak_vector(baseline.gram_adjusted)
    scenario = audit.SCENARIOS[-1]
    discrepancy, _ = audit.discrepancy_basis(
        baseline, scenario, generic, pooled_weak
    )
    rng = np.random.default_rng(20260824)
    residual = rng.normal(size=discrepancy.shape[0])
    apply = audit.low_rank_sqrt_precision(discrepancy)
    transformed = float(apply(residual) @ apply(residual))
    coefficient = np.linalg.solve(
        discrepancy.T @ discrepancy + np.eye(discrepancy.shape[1]),
        discrepancy.T @ residual,
    )
    direct = float(
        np.sum((residual - discrepancy @ coefficient) ** 2)
        + coefficient @ coefficient
    )
    identity_error = abs(transformed - direct) / max(abs(direct), 1.0)
    check(
        "linear discrepancy profiling identity",
        identity_error < 1.0e-12,
        {"relative_error": identity_error},
        checks,
    )

    # Numerical and domain checks on all retained results.
    check(
        "all profile optimizers report success",
        bool(profiles.optimizer_success.all()),
        {"failed_count": int(np.count_nonzero(~profiles.optimizer_success))},
        checks,
    )
    check(
        "all stored results finite",
        bool(
            np.isfinite(profiles[["Cn_MAP", "objective_MAP"]].to_numpy()).all()
            and np.isfinite(
                information[
                    [
                        "lambda_min_gain",
                        "weak_rotation_combined_vs_static_deg",
                        "bounding_weight_min",
                        "bounding_weight_max",
                    ]
                ].to_numpy()
            ).all()
        ),
        {},
        checks,
    )
    expanded = information[information.scenario.str.startswith("expanded")]
    check(
        "expanded-fabric MAP path remains in convex bounding domain",
        bool(expanded.convex_weight_valid.all()),
        {
            "invalid_count": int(np.count_nonzero(~expanded.convex_weight_valid)),
            "minimum_weight": float(expanded.bounding_weight_min.min()),
            "maximum_weight": float(expanded.bounding_weight_max.max()),
        },
        checks,
    )

    # Weak central finite-difference audit at the exact pooled point.
    pooled_rows = {}
    for scenario in audit.SCENARIOS:
        profile = profiles[profiles.profile == scenario.name]
        pooled_rows[scenario.name] = profile.iloc[
            int(np.argmin(np.abs(profile.Vcem_fraction - baseline.theta[0])))
        ]
    fd_rows = []
    original_step = audit.FD_STEP
    for step in (1.0e-3, 1.0e-4, 1.0e-5):
        audit.FD_STEP = step
        for scenario in audit.SCENARIOS:
            row = pooled_rows[scenario.name]
            one = pd.DataFrame([row])
            local = audit.pointwise_information(
                baseline,
                scenario,
                one,
                audit.discrepancy_basis(
                    baseline, scenario, generic, pooled_weak
                )[0],
                pooled_weak,
            ).iloc[0]
            fd_rows.append(
                {
                    "step": step,
                    "scenario": scenario.name,
                    "lambda_min_gain": float(local.lambda_min_gain),
                }
            )
    audit.FD_STEP = original_step
    fd = pd.DataFrame(fd_rows)
    reference = fd[np.isclose(fd.step, 1.0e-4)].set_index("scenario").lambda_min_gain
    fd["relative_error_to_1e-4"] = [
        abs(row.lambda_min_gain / reference[row.scenario] - 1.0)
        for row in fd.itertuples(index=False)
    ]
    fd.to_csv(HERE / "results" / "finite_difference_stability.csv", index=False)
    check(
        "pointwise gain finite-difference stability",
        float(fd["relative_error_to_1e-4"].max()) < 1.0e-5,
        {
            "maximum_relative_error_to_1e-4": float(
                fd["relative_error_to_1e-4"].max()
            )
        },
        checks,
    )

    # Multi-start, reverse-continuation, and adaptive-crossing checks.
    multistart = json.loads(
        (HERE / "results" / "multistart_summary.json").read_text()
    )
    all_multistart_success = all(
        item["all_new_runs_success"]
        for item in multistart["profiles"].values()
    )
    check(
        "all multi-start and reverse-path optimizations report success",
        all_multistart_success,
        {"run_count": int(multistart["run_count"])},
        checks,
    )
    max_objective_improvement = max(
        float(item["maximum_objective_improvement"])
        for item in multistart["profiles"].values()
    )
    max_cn_shift = max(
        float(item["maximum_absolute_Cn_shift"])
        for item in multistart["profiles"].values()
    )
    check(
        "multi-start changes remain below interpretation tolerances",
        max_objective_improvement < 1.0e-3 and max_cn_shift < 1.1e-2,
        {
            "maximum_objective_improvement": max_objective_improvement,
            "maximum_absolute_Cn_shift": max_cn_shift,
        },
        checks,
    )
    shared_stability = multistart["profiles"]["shared_generic"]
    check(
        "shared profile is start-path invariant in convex domain",
        float(shared_stability["maximum_objective_improvement"]) < 1.0e-6
        and float(shared_stability["maximum_absolute_Cn_shift"]) < 1.0e-3,
        {
            "maximum_objective_improvement": float(
                shared_stability["maximum_objective_improvement"]
            ),
            "maximum_absolute_Cn_shift": float(
                shared_stability["maximum_absolute_Cn_shift"]
            ),
        },
        checks,
    )

    crossings = pd.read_csv(HERE / "results" / "shared_dense_crossings.csv")
    widths = pd.read_csv(
        HERE / "results" / "shared_dense_profile_widths.csv"
    ).set_index("threshold")
    max_root_residual = float(np.abs(crossings.root_residual).max())
    width_230 = float(widths.loc[2.30, "Vcem_width_percentage_points"])
    width_599 = float(widths.loc[5.99, "Vcem_width_percentage_points"])
    check(
        "adaptive shared crossings satisfy profile thresholds",
        max_root_residual < 2.0e-5,
        {
            "maximum_absolute_root_residual": max_root_residual,
            "DeltaPhi_2p30_width_percentage_points": width_230,
            "DeltaPhi_5p99_width_percentage_points": width_599,
        },
        checks,
    )
    check(
        "adaptive shared widths reproduce dense audit",
        abs(width_230 - 0.94919) < 5.0e-4
        and abs(width_599 - 1.53216) < 5.0e-4,
        {
            "DeltaPhi_2p30_width_percentage_points": width_230,
            "DeltaPhi_5p99_width_percentage_points": width_599,
        },
        checks,
    )

    refined_shared = json.loads(
        (HERE / "results" / "shared_refined_summary.json").read_text()
    )
    check(
        "refined shared supported path is converged and convex",
        bool(refined_shared["all_optimizer_success"])
        and bool(refined_shared["all_convex_weight_valid"]),
        {
            "points": int(refined_shared["grid"]["total_points"]),
            "minimum_gain": float(refined_shared["gain"]["minimum"]),
            "maximum_gain": float(refined_shared["gain"]["maximum"]),
        },
        checks,
    )

    pointwise_multistart = json.loads(
        (HERE / "results" / "multistart_pointwise_summary.json").read_text()
    )
    expanded_range = pointwise_multistart["expanded_generic"]
    aligned_range = pointwise_multistart["expanded_target_aligned"]
    check(
        "post-multistart pointwise ranges preserve rounded conclusions",
        2.49 < float(expanded_range["minimum_gain"]) < 2.51
        and 4.81 < float(expanded_range["maximum_gain"]) < 4.83
        and 1.18 < float(aligned_range["minimum_gain"]) < 1.20
        and 2.52 < float(aligned_range["maximum_gain"]) < 2.54,
        {
            "expanded_minimum": float(expanded_range["minimum_gain"]),
            "expanded_maximum": float(expanded_range["maximum_gain"]),
            "aligned_minimum": float(aligned_range["minimum_gain"]),
            "aligned_maximum": float(aligned_range["maximum_gain"]),
        },
        checks,
    )

    for figure in [
        HERE / "results" / "figures" / "Fig_nonlinear_ridge_robustness.png",
        HERE / "results" / "figures" / "Fig_nonlinear_ridge_robustness.pdf",
    ]:
        check(
            f"figure exists: {figure.name}",
            figure.exists() and figure.stat().st_size > 1000,
            {"size_bytes": int(figure.stat().st_size) if figure.exists() else 0},
            checks,
        )

    png = HERE / "results" / "figures" / "Fig_nonlinear_ridge_robustness.png"
    image_ok = False
    image_size: tuple[int, int] | None = None
    try:
        with Image.open(png) as opened:
            opened.verify()
        with Image.open(png) as opened:
            opened.load()
            image_size = tuple(map(int, opened.size))
        image_ok = True
    except Exception:
        image_ok = False
    check(
        "PNG decodes completely",
        image_ok,
        {"pixel_dimensions": list(image_size) if image_size else None},
        checks,
    )

    output = {
        "all_passed": bool(all(item["passed"] for item in checks)),
        "n_checks": len(checks),
        "checks": checks,
    }
    (HERE / "results" / "verification.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
