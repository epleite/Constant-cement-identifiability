#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import e2_analysis as e2


EXPECTED_HASHES = {
    "vendor/e1_v1/src/e1_analysis.py": "c5ada1583b454d05cd38ff28d1dfce73078e27ae2394ce761e9c86b0fc2d331b",
    "vendor/e1_v1/vendor/rpia_v1/rpia_core.py": "b41a6ec9471ffccb03d586cb8c8c1231a937a909e72f3aeb9434b03b2d1ab80a",
    "vendor/e1_v1/data/compact/19A_training_window.csv": "33dfc5cb97a369542a7856c99f093688276d19b4ddb8d439270786818692d91a",
    "vendor/e1_v1/data/compact/BT2_training_window.csv": "db72d040b8bdef03f0c6cf0345038fc722cebc36b6600c35275e800ebc7e6d46",
}

# Bounded least-squares optima can differ by a few ulps across BLAS/libm,
# CPU, and Python patch releases even when the model, data, seeds, and pinned
# Python packages are identical.  These tolerances remain orders of magnitude
# below the precision used for scientific interpretation and still detect any
# material change to the frozen analysis.
ANCHOR_VCEM_ATOL = 2e-7
ANCHOR_CN_ATOL = 2e-5
BOOTSTRAP_COORD_ATOL = 2e-6


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append({"check": name, "passed": bool(passed), "detail": detail})

    def close(self) -> dict:
        failures = [row for row in self.rows if not row["passed"]]
        return {
            "passed": not failures,
            "n_checks": len(self.rows),
            "n_failed": len(failures),
            "checks": self.rows,
        }


def near(a: float, b: float, atol: float = 1e-8, rtol: float = 1e-8) -> bool:
    return bool(np.isclose(a, b, atol=atol, rtol=rtol))


def main() -> None:
    checks = Checks()
    for relative, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        checks.add(f"frozen hash: {relative}", actual == expected, actual)

    baseline = e2.load_baseline()
    checks.add("sample count 19A", len(baseline["wells"]["19A"]) == 29, str(len(baseline["wells"]["19A"])))
    checks.add("sample count BT2", len(baseline["wells"]["BT2"]) == 30, str(len(baseline["wells"]["BT2"])))
    checks.add("sample count pooled", len(baseline["pooled"]) == 59, str(len(baseline["pooled"])))

    expected_points = {
        "19A": (0.024592733574879642, 4.7029523059115945),
        "BT2": (0.002709648210837285, 8.292355473009664),
        "pooled": (0.012546045825196089, 5.436001877454284),
    }
    for well in ("19A", "BT2"):
        theta = baseline["rp_fits"][well].theta
        checks.add(
            f"E1 Vcem anchor {well}",
            near(theta[0], expected_points[well][0], ANCHOR_VCEM_ATOL),
            str(theta[0]),
        )
        checks.add(
            f"E1 Cn anchor {well}",
            near(math.exp(theta[1]), expected_points[well][1], ANCHOR_CN_ATOL),
            str(math.exp(theta[1])),
        )
    theta = baseline["rp_pooled"].theta
    checks.add(
        "E1 Vcem anchor pooled",
        near(theta[0], expected_points["pooled"][0], ANCHOR_VCEM_ATOL),
        str(theta[0]),
    )
    checks.add(
        "E1 Cn anchor pooled",
        near(math.exp(theta[1]), expected_points["pooled"][1], ANCHOR_CN_ATOL),
        str(math.exp(theta[1])),
    )

    for suffix in ("raw", "adjusted"):
        coordinate = baseline["coordinate_rpia"]
        beta = e2.beta_from_factored_coordinate(
            float(theta[0]),
            float(coordinate[f"A_{suffix}"]),
            float(coordinate[f"Gamma_{suffix}"]),
        )
        target = float(coordinate[f"beta_total_{suffix}_projection"])
        checks.add(
            f"factored derivative identity {suffix}",
            near(beta, target, 2e-8),
            f"factored={beta:.12g}, target={target:.12g}",
        )

    table_dir = ROOT / "results" / "tables"
    bootstrap = pd.read_csv(table_dir / "E2_bootstrap_replicates.csv")
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    hierarchy = pd.read_csv(table_dir / "E2_hierarchical_comparison.csv")
    level = pd.read_csv(table_dir / "E2_loto_level.csv")
    shape = pd.read_csv(table_dir / "E2_loto_shape_summary.csv")
    loto_bs = pd.read_csv(table_dir / "E2_loto_bootstrap_replicates.csv")
    profiles = pd.read_csv(table_dir / "E2_loto_profiles.csv")

    expected_counts = {"IID": 120, "MBB_12m": 120, "MBB_20m_primary": 400, "MBB_40m": 120}
    actual_counts = bootstrap.groupby("scheme").size().to_dict()
    checks.add("bootstrap requested counts", actual_counts == expected_counts, str(actual_counts))
    checks.add("bootstrap all successful", bool(bootstrap.bootstrap_success.all()), str(bootstrap.bootstrap_success.value_counts().to_dict()))
    checks.add("LOTO bootstrap count", len(loto_bs) == 600, str(len(loto_bs)))
    checks.add("LOTO bootstrap all successful", bool(loto_bs.success.all()), str(loto_bs.success.value_counts().to_dict()))
    checks.add("full mode summary", summary.get("quick_mode") is False, str(summary.get("quick_mode")))

    critical = ["A_raw", "Gamma_raw", "A_adjusted", "Gamma_adjusted", "beta_raw", "beta_adjusted"]
    checks.add(
        "bootstrap critical values finite",
        bool(np.isfinite(bootstrap[critical].to_numpy()).all()),
        f"nonfinite={int((~np.isfinite(bootstrap[critical].to_numpy())).sum())}",
    )
    primary = bootstrap[bootstrap.scheme == "MBB_20m_primary"]
    checks.add(
        "hierarchical bootstrap constraints nonnegative",
        bool((primary[["hierarchy_qstar_delta", "hierarchy_Cn_delta"]] >= -1e-6).all().all()),
        f"min={primary[['hierarchy_qstar_delta','hierarchy_Cn_delta']].min().to_dict()}",
    )

    for family, adjustment in hierarchy[["comparison_family", "adjustment"]].drop_duplicates().itertuples(index=False):
        group = hierarchy[(hierarchy.comparison_family == family) & (hierarchy.adjustment == adjustment)].set_index("model")
        separate = group.loc["separate", "objective_total"]
        constrained_min = group.loc[["shared_qstar", "shared_Cn", "pooled_theta"], "objective_total"].min()
        checks.add(
            f"separate lower bound {family}/{adjustment}",
            separate <= constrained_min + 2e-6,
            f"separate={separate:.12g}, constrained_min={constrained_min:.12g}",
        )
        qdelta = abs(group.loc["shared_qstar", "delta_q_BT2_minus_19A"])
        cdelta = abs(group.loc["shared_Cn", "delta_lnCn_BT2_minus_19A"])
        checks.add(f"shared q equality {family}/{adjustment}", qdelta < 2e-7, str(qdelta))
        checks.add(f"shared Cn equality {family}/{adjustment}", cdelta < 2e-7, str(cdelta))

    rp = level[level.operating_definition == "RPIA"]
    checks.add(
        "LOTO point qstar closer than local exponential",
        bool((np.abs(rp.qstar_level_ratio - 1) < np.abs(rp.local_exponential_level_ratio - 1)).all()),
        rp[["train", "test", "adjustment", "qstar_level_ratio", "local_exponential_level_ratio"]].to_json(orient="records"),
    )
    local = shape[shape.window == "local_0p015"]
    pivot = local.pivot_table(index=["train", "test", "adjustment"], columns="method", values="rms_lnCn_error")
    checks.add(
        "LOTO shape qstar beats local tangent",
        bool((pivot.train_qstar < pivot.train_local_exponential).all()),
        (pivot.train_qstar / pivot.train_local_exponential).to_json(),
    )
    checks.add(
        "all adjusted profiles converged",
        bool(profiles[profiles.adjustment == "adjusted"].profile_objective.notna().all()),
        f"rows={len(profiles[profiles.adjustment == 'adjusted'])}",
    )

    # Reproduce the first primary bootstrap replicate from its independent child seed.
    rng = np.random.default_rng(np.random.SeedSequence(20260823).spawn(4)[2])
    sampled = {
        well: e2.noncircular_moving_block_sample(df, 5, rng)[0]
        for well, df in baseline["wells"].items()
    }
    pooled = pd.concat([sampled["19A"], sampled["BT2"]], ignore_index=True)
    fit = e2.rp_grid_fit(pooled, baseline["rp_pooled"].theta)
    coordinate = e2.e1.metric_contact_hs_decomposition(pooled, fit.theta, baseline["scale"])
    stored = primary.set_index("replicate").loc[0]
    checks.add(
        "deterministic first primary A_raw",
        near(coordinate["A_raw"], stored.A_raw, BOOTSTRAP_COORD_ATOL),
        f"recomputed={coordinate['A_raw']:.15g}, stored={stored.A_raw:.15g}",
    )
    checks.add(
        "deterministic first primary Gamma_adjusted",
        near(
            coordinate["Gamma_adjusted"],
            stored.Gamma_adjusted,
            BOOTSTRAP_COORD_ATOL,
        ),
        f"recomputed={coordinate['Gamma_adjusted']:.15g}, stored={stored.Gamma_adjusted:.15g}",
    )

    for stem in [
        "Fig_E2_bootstrap_stability",
        "Fig_E2_loto_transport",
        "Fig_E2_hierarchical_comparison",
    ]:
        for suffix in ("png", "pdf"):
            path = ROOT / "results" / "figures" / f"{stem}.{suffix}"
            checks.add(f"figure exists {stem}.{suffix}", path.is_file() and path.stat().st_size > 5000, str(path.stat().st_size if path.exists() else 0))

    report = checks.close()
    output = ROOT / "results" / "verification" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    pd.DataFrame(report["checks"]).to_csv(
        ROOT / "results" / "verification" / "verification.csv", index=False
    )
    e2.write_manifest()
    print(json.dumps({key: report[key] for key in ("passed", "n_checks", "n_failed")}, indent=2))
    if not report["passed"]:
        for failure in [row for row in report["checks"] if not row["passed"]]:
            print(f"FAILED: {failure['check']}: {failure['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
