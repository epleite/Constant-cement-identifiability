from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nonlinear_ridge as core  # noqa: E402


def compatible_profile(table: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "Vcem_fraction": table.Vcem_fraction,
            "Vcem_percent": table.Vcem_percent,
            "lnCn_MAP": table.result_lnCn,
            "Cn_MAP": table.result_Cn,
            "objective_MAP": table.objective,
        }
    )
    for name in names:
        output[f"MAP_{name}_sigma"] = table[f"result_{name}_sigma"]
    return output


def main() -> None:
    baseline = core.e3.load_baseline()
    best = pd.read_csv(HERE / "results" / "multistart_best_profiles.csv")
    generic = core.generic_discrepancy_matrix(baseline)
    pooled_weak = core.weak_vector(baseline.gram_adjusted)
    parts = []
    summary = {}
    for scenario in core.SCENARIOS:
        if scenario.name == "shared_generic":
            continue
        table = best[best.profile == scenario.name].copy()
        profile = compatible_profile(table, core.ALL_NAMES)
        discrepancy = core.discrepancy_basis(
            baseline, scenario, generic, pooled_weak
        )[0]
        information = core.pointwise_information(
            baseline,
            scenario,
            profile,
            discrepancy,
            pooled_weak,
        )
        parts.append(information)
        gains = information.lambda_min_gain.to_numpy(dtype=float)
        rotations = information.weak_rotation_combined_vs_static_deg.to_numpy(
            dtype=float
        )
        summary[scenario.name] = {
            "minimum_gain": float(np.min(gains)),
            "maximum_gain": float(np.max(gains)),
            "minimum_rotation_deg": float(np.min(rotations)),
            "maximum_rotation_deg": float(np.max(rotations)),
            "all_convex_weight_valid": bool(
                information.convex_weight_valid.all()
            ),
            "n": int(len(information)),
        }
    output = pd.concat(parts, ignore_index=True)
    output.to_csv(
        HERE / "results" / "multistart_pointwise_information.csv", index=False
    )
    (HERE / "results" / "multistart_pointwise_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
