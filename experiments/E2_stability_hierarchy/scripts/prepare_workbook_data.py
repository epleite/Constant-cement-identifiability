#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def records(path: Path, columns: list[str] | None = None) -> list[dict]:
    frame = pd.read_csv(path)
    if columns is not None:
        frame = frame[columns]
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    tables = root / "results" / "tables"
    summary = json.loads((root / "results" / "summary.json").read_text())

    data = {
        "summary": summary,
        "key_results": [
            {
                "metric": metric,
                "value": values["median"],
                "ci95_low": values["ci95"][0],
                "ci95_high": values["ci95"][1],
                "point_estimate": values["point"],
                "unit": "dimensionless",
            }
            for metric, values in summary["coordinate_stability"].items()
        ]
        + [
            {
                "metric": "hierarchy_qstar_preferred_fraction",
                "value": summary["hierarchical_bootstrap"]["qstar_preferred_fraction"],
                "ci95_low": None,
                "ci95_high": None,
                "point_estimate": None,
                "unit": "fraction",
            },
            {
                "metric": "hierarchy_median_delta_Cn_minus_qstar",
                "value": summary["hierarchical_bootstrap"]["median_delta_Cn_minus_qstar"],
                "ci95_low": summary["hierarchical_bootstrap"]["ci95_delta_Cn_minus_qstar"][0],
                "ci95_high": summary["hierarchical_bootstrap"]["ci95_delta_Cn_minus_qstar"][1],
                "point_estimate": None,
                "unit": "objective units",
            },
        ],
        "bootstrap_summary": records(tables / "E2_bootstrap_summary.csv"),
        "bootstrap_diagnostics": records(tables / "E2_bootstrap_diagnostics.csv"),
        "bootstrap_replicates": records(
            tables / "E2_bootstrap_replicates.csv",
            [
                "scheme",
                "block_length_m",
                "replicate",
                "unique_fraction_19A",
                "unique_fraction_BT2",
                "bootstrap_success",
                "pooled_fit_bound_hit",
                "Vcem_fraction",
                "Cn",
                "A_raw",
                "Gamma_raw",
                "A_adjusted",
                "Gamma_adjusted",
                "beta_raw",
                "beta_adjusted",
                "hierarchy_qstar_delta",
                "hierarchy_Cn_delta",
                "hierarchy_delta_Cn_minus_qstar",
                "hierarchy_qstar_bound_hit",
                "hierarchy_Cn_bound_hit",
            ],
        ),
        "loto_level": records(tables / "E2_loto_level.csv"),
        "loto_bootstrap_summary": records(tables / "E2_loto_bootstrap_summary.csv"),
        "loto_bootstrap_replicates": records(
            tables / "E2_loto_bootstrap_replicates.csv",
            [
                "train",
                "test",
                "replicate",
                "unique_fraction_train",
                "success",
                "bound_hit",
                "Vcem_train_bootstrap",
                "Cn_train_bootstrap",
                "A_raw",
                "Gamma_raw",
                "A_adjusted",
                "Gamma_adjusted",
                "qstar_ratio_raw",
                "qstar_ratio_adjusted",
                "local_exponential_ratio_raw",
                "local_exponential_ratio_adjusted",
            ],
        ),
        "loto_shape": records(tables / "E2_loto_shape_summary.csv"),
        "hierarchy": records(tables / "E2_hierarchical_comparison.csv"),
        "acf": records(tables / "E2_residual_acf.csv"),
        "verification": records(root / "results" / "verification" / "verification.csv"),
        "sources": [
            {
                "artifact": "Frozen E1 forward model",
                "path": "vendor/e1_v1/vendor/rpia_v1/rpia_core.py",
                "note": "Exact RPIA v1 constant-cement Scheme 1 snapshot.",
            },
            {
                "artifact": "19A compact data",
                "path": "vendor/e1_v1/data/compact/19A_training_window.csv",
                "note": "Selected Hugin interval is derived by the frozen E1 selector.",
            },
            {
                "artifact": "BT2 compact data",
                "path": "vendor/e1_v1/data/compact/BT2_training_window.csv",
                "note": "Selected Hugin interval is derived by the frozen E1 selector.",
            },
            {
                "artifact": "Methods",
                "path": "docs/METHODS.md",
                "note": "Definitions, resampling design, hierarchy, and limitations.",
            },
            {
                "artifact": "Machine-readable result",
                "path": "results/summary.json",
                "note": "Headline outputs used by this workbook.",
            },
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, allow_nan=False))


if __name__ == "__main__":
    main()

