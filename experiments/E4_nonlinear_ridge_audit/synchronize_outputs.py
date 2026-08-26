from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nonlinear_ridge as core  # noqa: E402


def strict(value):
    if isinstance(value, dict):
        return {key: strict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    summary_path = HERE / "results" / "summary.json"
    summary = json.loads(summary_path.read_text())
    multistart = json.loads(
        (HERE / "results" / "multistart_summary.json").read_text()
    )
    shared_refined = json.loads(
        (HERE / "results" / "shared_refined_summary.json").read_text()
    )
    multistart_pointwise = json.loads(
        (HERE / "results" / "multistart_pointwise_summary.json").read_text()
    )
    dense_widths = pd.read_csv(
        HERE / "results" / "shared_dense_profile_widths.csv"
    )

    shared = summary["scenarios"]["shared_generic"]
    if "coarse_profile_widths" not in shared:
        shared["coarse_profile_widths"] = shared["profile_widths"]
    shared["profile_widths"] = {
        str(float(row.threshold)): {
            "threshold": float(row.threshold),
            "Vcem_lower_fraction": float(row.Vcem_lower_fraction),
            "Vcem_upper_fraction": float(row.Vcem_upper_fraction),
            "Vcem_width_percentage_points": float(
                row.Vcem_width_percentage_points
            ),
            "lower_censored": bool(row.lower_censored),
            "upper_censored": bool(row.upper_censored),
            "estimation": "adaptive nonlinear nuisance profile root",
        }
        for row in dense_widths.itertuples(index=False)
    }
    if "coarse_grid_pointwise_distribution" not in shared:
        shared["coarse_grid_pointwise_distribution"] = shared[
            "pointwise_gain_on_convex_DeltaPhi_le_2p30_profile"
        ]
    shared["pointwise_gain_on_refined_DeltaPhi_le_2p30_profile"] = {
        "minimum": shared_refined["gain"]["minimum"],
        "maximum": shared_refined["gain"]["maximum"],
        "pooled": next(
            item["lambda_min_gain"]
            for item in shared_refined["anchors"]
            if item["label"] == "pooled"
        ),
        "n_uniform_Vcem_plus_pooled": shared_refined["grid"]["total_points"],
        "reporting_note": shared_refined["gain"]["recommended_reporting"],
        "lower_crossing_anchor": shared_refined["anchors"][0],
        "upper_crossing_anchor": shared_refined["anchors"][-1],
    }
    shared["weak_rotation_refined_supported_deg"] = shared_refined[
        "weak_rotation_deg"
    ]

    for name, result in multistart_pointwise.items():
        summary["scenarios"][name]["multistart_confirmed_pointwise_range"] = result

    summary["multistart_reverse_path_audit"] = multistart
    summary["shared_refined_supported_audit"] = shared_refined
    summary_path.write_text(
        json.dumps(strict(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    baseline = core.e3.load_baseline()
    profiles = pd.read_csv(HERE / "results" / "nonlinear_MAP_profiles.csv")
    information = pd.read_csv(
        HERE / "results" / "pointwise_efficient_information.csv"
    )
    core.plot_results(baseline, profiles, information)
    print("summary and figures synchronized")


if __name__ == "__main__":
    main()
