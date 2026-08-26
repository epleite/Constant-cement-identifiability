from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multistart_audit as multi  # noqa: E402
import run_nonlinear_ridge as core  # noqa: E402


def main() -> None:
    baseline = core.e3.load_baseline()
    scenario = next(item for item in core.SCENARIOS if item.name == "shared_generic")
    generic = core.generic_discrepancy_matrix(baseline)
    pooled_weak = core.weak_vector(baseline.gram_adjusted)
    discrepancy = core.discrepancy_basis(
        baseline, scenario, generic, pooled_weak
    )[0]
    problem = multi.ProfileProblem(baseline, scenario, discrepancy)

    crossings = pd.read_csv(HERE / "results" / "shared_dense_crossings.csv")
    roots = crossings[np.isclose(crossings.threshold, 2.30)].set_index("side")
    lower = float(roots.loc["lower", "Vcem_fraction"])
    upper = float(roots.loc["upper", "Vcem_fraction"])
    v_values = np.unique(
        np.r_[np.linspace(lower, upper, 41), float(baseline.theta[0])]
    )
    coarse = pd.read_csv(HERE / "results" / "multistart_best_profiles.csv")
    coarse = coarse[coarse.profile == scenario.name].sort_values("Vcem_fraction")

    def nearest_start(vcem: float) -> np.ndarray:
        row = coarse.iloc[
            int(np.argmin(np.abs(coarse.Vcem_fraction.to_numpy() - vcem)))
        ]
        return np.r_[
            float(row.result_lnCn),
            [float(row[f"result_{name}_sigma"]) for name in problem.names],
        ]

    rows = []
    previous: np.ndarray | None = None
    for vcem in v_values:
        starts = [("nearest_multistart_MAP", nearest_start(float(vcem)))]
        if previous is not None:
            starts.append(("forward_continuation", previous.copy()))
        candidates = []
        for label, start in starts:
            fit = problem.fit(float(vcem), start)
            candidates.append((label, fit))
        label, best = min(candidates, key=lambda pair: pair[1].objective)
        previous = best.x.copy()
        row = {
            "profile": scenario.name,
            "Vcem_fraction": float(vcem),
            "Vcem_percent": float(100.0 * vcem),
            "lnCn_MAP": float(best.x[0]),
            "Cn_MAP": float(np.exp(best.x[0])),
            "objective_MAP": float(best.objective),
            "nuisance_prior_norm": float(np.linalg.norm(best.x[1:])),
            "optimizer_success": bool(best.success),
            "optimizer_nfev": int(best.nfev),
            "optimizer_optimality": float(best.optimality),
            "active_bounds": int(best.active_bounds),
            "selected_start_strategy": label,
            "candidate_count": len(candidates),
        }
        for name, value in zip(problem.names, best.x[1:]):
            row[f"MAP_{name}_sigma"] = float(value)
        rows.append(row)
    profile = pd.DataFrame(rows)
    information = core.pointwise_information(
        baseline,
        scenario,
        profile,
        discrepancy,
        pooled_weak,
    )
    profile.to_csv(
        HERE / "results" / "shared_refined_supported_profile.csv", index=False
    )
    information.to_csv(
        HERE / "results" / "shared_refined_pointwise_information.csv", index=False
    )

    gain = information.lambda_min_gain.to_numpy(dtype=float)
    rotation = information.weak_rotation_combined_vs_static_deg.to_numpy(dtype=float)

    def anchor(label: str, value: float) -> dict:
        index = int(np.argmin(np.abs(information.Vcem_fraction - value)))
        row = information.iloc[index]
        return {
            "label": label,
            "Vcem_fraction": float(row.Vcem_fraction),
            "Vcem_percent": float(row.Vcem_percent),
            "Cn_MAP": float(row.Cn_MAP),
            "objective_MAP": float(row.objective_MAP),
            "lambda_min_gain": float(row.lambda_min_gain),
            "weak_rotation_deg": float(
                row.weak_rotation_combined_vs_static_deg
            ),
        }

    summary = {
        "grid": {
            "uniform_Vcem_points_between_DeltaPhi_2p30_crossings": 41,
            "exact_pooled_point_added": True,
            "total_points": int(len(profile)),
            "lower_crossing_fraction": lower,
            "upper_crossing_fraction": upper,
        },
        "all_optimizer_success": bool(profile.optimizer_success.all()),
        "all_convex_weight_valid": bool(information.convex_weight_valid.all()),
        "gain": {
            "minimum": float(np.min(gain)),
            "q25_uniform_Vcem_grid": float(np.quantile(gain, 0.25)),
            "median_uniform_Vcem_grid": float(np.median(gain)),
            "q75_uniform_Vcem_grid": float(np.quantile(gain, 0.75)),
            "maximum": float(np.max(gain)),
            "recommended_reporting": "report minimum--maximum and pooled anchor; quantiles depend on the chosen Vcem measure",
        },
        "weak_rotation_deg": {
            "minimum": float(np.min(rotation)),
            "maximum": float(np.max(rotation)),
        },
        "anchors": [
            anchor("lower_DeltaPhi_2p30_crossing", lower),
            anchor("pooled", float(baseline.theta[0])),
            anchor("upper_DeltaPhi_2p30_crossing", upper),
        ],
    }
    (HERE / "results" / "shared_refined_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
