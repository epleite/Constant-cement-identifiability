from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"


def check(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: list[dict] = []
    pooled = pd.read_csv(TABLES / "Q1_pooled_coordinate_curves.csv")
    level = pd.read_csv(TABLES / "Q1_loto_level.csv")
    endpoint = pd.read_csv(TABLES / "Q2_endpoint_point_closure.csv").set_index(
        "component"
    )
    bootstrap = pd.read_csv(TABLES / "Q2_bootstrap_endpoint_replicates.csv")

    check(len(bootstrap) == 400, "400 primary bootstrap replicates", checks)
    check(
        float(np.max(np.abs(level.tangent_difference))) < 1e-6,
        "all finite alternatives have the matched training-point tangent",
        checks,
    )
    for adjustment in ("raw", "adjusted"):
        total = float(
            endpoint.loc[
                f"complete_observable_coordinate_{adjustment}",
                "beta_contribution_per_fraction",
            ]
        )
        contact = float(
            endpoint.loc[
                f"observable_weighted_contact_{adjustment}",
                "beta_contribution_per_fraction",
            ]
        )
        hs = float(
            endpoint.loc[
                f"HS_porosity_path_{adjustment}",
                "beta_contribution_per_fraction",
            ]
        )
        check(
            abs(total - contact - hs) < 1e-10,
            f"{adjustment} analytic contact/HS decomposition identity",
            checks,
        )
    for key in ("A_raw", "Gamma_raw", "A_adjusted", "Gamma_adjusted"):
        difference = float(
            np.max(
                np.abs(
                    bootstrap[f"{key}_saved"]
                    - bootstrap[f"{key}_recomputed"]
                )
            )
        )
        check(difference < 1e-8, f"bootstrap {key} recomputation", checks)
    reference_rows = pooled[
        np.isclose(
            pooled.Vcem_fraction,
            0.012546045825196089,
            rtol=0,
            atol=1e-12,
        )
    ]
    check(
        float(np.max(np.abs(reference_rows.log_coordinate_error))) < 2e-7,
        "all pooled coordinates pass through the reference point",
        checks,
    )
    for stem in (
        "Fig_Q1_finite_coordinate_robustness",
        "Fig_Q2_endpoint_slope_closure",
    ):
        for suffix in ("png", "pdf"):
            path = ROOT / "results" / "figures" / f"{stem}.{suffix}"
            check(path.exists() and path.stat().st_size > 10_000, f"{path.name} exists", checks)

    output = {"n_checks": len(checks), "all_passed": True, "checks": checks}
    (ROOT / "results" / "verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
