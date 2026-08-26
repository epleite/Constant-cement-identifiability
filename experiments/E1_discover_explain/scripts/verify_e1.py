from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "rpia_v1"))

import e1_analysis as e1
import rpia_core as rc


EXPECTED_CORE_SHA256 = "b41a6ec9471ffccb03d586cb8c8c1231a937a909e72f3aeb9434b03b2d1ab80a"


def check(condition: bool, label: str, diagnostics: dict, value=None) -> None:
    diagnostics[label] = {"passed": bool(condition), "value": value}
    if not condition:
        raise AssertionError(f"verification failed: {label}; value={value}")


def target_jacobian_step(df, theta, scale, factor):
    sig = e1.stacked_sigma(df, scale)
    J = np.zeros((len(sig), 2))
    for j, parameter_scale in enumerate(rc.PARAM_SCALES[e1.MODEL]):
        h = 1e-4 * factor
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += h * parameter_scale
        minus[j] -= h * parameter_scale
        J[:, j] = (
            rc.stack(df, e1.MODEL, plus, e1.NAMES)
            - rc.stack(df, e1.MODEL, minus, e1.NAMES)
        ) / (2 * h) / sig
    return J


def main() -> None:
    diagnostics = {}
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    operating = pd.read_csv(ROOT / "results" / "tables" / "E1_operating_points.csv")
    staircase = pd.read_csv(ROOT / "results" / "tables" / "E1_observable_staircase.csv")
    grid = pd.read_csv(ROOT / "results" / "tables" / "E1_vcem_cn_map.csv")
    phi_grid = pd.read_csv(ROOT / "results" / "tables" / "E1_vcem_phi_map.csv")
    ridge = pd.read_csv(ROOT / "results" / "tables" / "E1_ridge_validation.csv")
    candidate = pd.read_csv(ROOT / "results" / "tables" / "E1_candidate_state_invariance.csv")
    coordinate = pd.read_csv(ROOT / "results" / "tables" / "E1_coordinate_coefficients.csv").iloc[0]
    cross = pd.read_csv(ROOT / "results" / "tables" / "E1_cross_trajectory_coordinate.csv").set_index("trajectory")
    wells, pooled, _ = e1.load_data()

    core = ROOT / "vendor" / "rpia_v1" / "rpia_core.py"
    digest = hashlib.sha256(core.read_bytes()).hexdigest()
    check(digest == EXPECTED_CORE_SHA256, "frozen_core_sha256", diagnostics, digest)
    check(len(wells["19A"]) == 29 and len(wells["BT2"]) == 30, "sample_counts", diagnostics, [len(wells["19A"]), len(wells["BT2"])])
    check(float(pooled.phi.max()) < 0.999 * (rc.PHIC_PACK - 0.060), "domain_forward_validity", diagnostics, float(pooled.phi.max()))

    base = operating.set_index("operating_point").loc["pooled_RPIA"]
    anchors = {
        "Vcem_fraction": 0.012546045825196089,
        "Cn": 5.436001877454284,
        "beta_raw_projection": 20.271560409627071,
        "beta_raw_eigen": 20.274117849462762,
        "beta_adjusted_projection": 19.961765636121356,
        "beta_adjusted_eigen": 19.98935150814289,
        "raw_lambda_min": 0.009035935138662410,
        "adjusted_lambda_min": 0.003498495204923069,
    }
    for name, expected in anchors.items():
        value = float(base[name])
        tolerance = 2e-8 * max(1.0, abs(expected))
        check(abs(value - expected) <= tolerance, f"anchor_{name}", diagnostics, value)

    check(
        np.max(np.abs(phi_grid.beta_contact_K + phi_grid.beta_HS_path_K - phi_grid.beta_total_K)) < 2e-10,
        "bulk_decomposition_identity",
        diagnostics,
        float(np.max(np.abs(phi_grid.beta_contact_K + phi_grid.beta_HS_path_K - phi_grid.beta_total_K))),
    )
    check(
        np.max(np.abs(phi_grid.beta_contact_G + phi_grid.beta_HS_path_G - phi_grid.beta_total_G)) < 2e-10,
        "shear_decomposition_identity",
        diagnostics,
        float(np.max(np.abs(phi_grid.beta_contact_G + phi_grid.beta_HS_path_G - phi_grid.beta_total_G))),
    )
    check(abs(base.beta_Vs_median - base.median_beta_total_G) < 5e-6, "Vs_equals_Gd_slope", diagnostics, float(base.beta_Vs_median - base.median_beta_total_G))

    by_set = staircase.set_index("observable_set")
    check(by_set.loc["Vp+Vs+rho", "density_target_jacobian_norm"] < 1e-13, "density_has_zero_target_sensitivity", diagnostics, float(by_set.loc["Vp+Vs+rho", "density_target_jacobian_norm"]))
    check(abs(by_set.loc["Vp+Vs", "raw_lambda_min"] - by_set.loc["Vp+Vs+rho", "raw_lambda_min"]) < 1e-12, "density_leaves_raw_information_unchanged", diagnostics, float(by_set.loc["Vp+Vs+rho", "raw_lambda_min"] - by_set.loc["Vp+Vs", "raw_lambda_min"]))
    check(by_set.loc["Vp+Vs+rho", "adjusted_lambda_min"] >= by_set.loc["Vp+Vs", "adjusted_lambda_min"], "density_can_constrain_nuisances", diagnostics, float(by_set.loc["Vp+Vs+rho", "adjusted_lambda_min"] - by_set.loc["Vp+Vs", "adjusted_lambda_min"]))

    theta = np.array([base.Vcem_fraction, base.lnCn], dtype=float)
    discrepancy = np.array([summary["transfer_discrepancy"][name] for name in e1.NAMES])
    scale = e1.transfer_aware_scale(discrepancy)
    betas = []
    for factor in (0.5, 1.0, 2.0):
        J = target_jacobian_step(pooled, theta, scale, factor)
        betas.append(e1.beta_from_eigenvector(J.T @ J))
    check(max(abs(np.asarray(betas) - betas[1])) < 2e-5, "target_step_stability", diagnostics, betas)

    metric, Jt, Jn, G, Gadj, _ = e1.metric_summary(pooled, theta, scale)
    check(np.linalg.eigvalsh(Gadj).min() > -1e-10, "Gadj_positive_semidefinite", diagnostics, np.linalg.eigvalsh(Gadj).tolist())
    check(np.linalg.eigvalsh(G - Gadj).min() > -1e-10, "schur_information_loss_positive_semidefinite", diagnostics, np.linalg.eigvalsh(G - Gadj).tolist())

    K0, G0 = rc.matrix(float(pooled.vsh.median()))[:2]
    Cn = float(np.exp(theta[1]))
    endpoint = e1.endpoint_diagnostics(K0, G0, float(theta[0]), Cn)
    hv, hl = 1e-6, 1e-5

    def endpoint_log(vcem, ell):
        return np.log(rc.contact(K0, G0, rc.KCEM, rc.GCEM, rc.PHIC_PACK - vcem, rc.PHIC_PACK, np.exp(ell), 1))

    jv = (endpoint_log(theta[0] + hv, theta[1]) - endpoint_log(theta[0] - hv, theta[1])) / (2 * hv)
    jl = (endpoint_log(theta[0], theta[1] + hl) - endpoint_log(theta[0], theta[1] - hl)) / (2 * hl)
    beta_numeric = jv / jl
    check(abs(beta_numeric[0] - endpoint["beta_endpoint_K"]) < 2e-5, "endpoint_beta_K_analytic", diagnostics, [float(beta_numeric[0]), endpoint["beta_endpoint_K"]])
    check(abs(beta_numeric[1] - endpoint["beta_endpoint_G"]) < 2e-5, "endpoint_beta_G_analytic", diagnostics, [float(beta_numeric[1]), endpoint["beta_endpoint_G"]])

    v0 = float(theta[0])
    l0 = float(theta[1])

    def loga(vcem, ell):
        return np.log(2 * (vcem / (3 * np.exp(ell) * (1 - rc.PHIC_PACK))) ** 0.25)

    d_loga_d_lnv = (loga(v0 * np.exp(hl), l0) - loga(v0 * np.exp(-hl), l0)) / (2 * hl)
    d_loga_d_ell = (loga(v0, l0 + hl) - loga(v0, l0 - hl)) / (2 * hl)
    check(abs(d_loga_d_lnv - 0.25) < 1e-9, "scheme1_dloga_dlnVcem", diagnostics, float(d_loga_d_lnv))
    check(abs(d_loga_d_ell + 0.25) < 1e-9, "scheme1_dloga_dlnCn", diagnostics, float(d_loga_d_ell))

    check(np.isfinite(grid.select_dtypes(include=[np.number]).to_numpy()).all(), "finite_parameter_grid", diagnostics, list(grid.shape))
    check(np.isfinite(phi_grid.select_dtypes(include=[np.number]).to_numpy()).all(), "finite_phi_grid", diagnostics, list(phi_grid.shape))
    check(ridge.nuisance_profile_success.astype(bool).all(), "nonlinear_profiles_converged", diagnostics, int(ridge.nuisance_profile_success.astype(bool).sum()))
    nearest = candidate.iloc[(candidate.Vcem_fraction - base.Vcem_fraction).abs().argsort()[:1]].iloc[0]
    candidate_fields = [
        "local_empirical_coordinate_ratio",
        "physics_factored_raw_coordinate_ratio",
        "physics_factored_adjusted_coordinate_ratio",
        "physics_linearized_raw_coordinate_ratio",
        "physics_linearized_adjusted_coordinate_ratio",
        "a_c_ratio",
        "median_Kb_ratio",
        "median_Gb_ratio",
        "T_ratio",
        "median_Kd_ratio",
        "median_Gd_ratio",
        "median_Vp_ratio",
        "median_Vs_ratio",
    ]
    check(
        np.max(np.abs(nearest[candidate_fields].to_numpy(dtype=float) - 1.0)) < 1e-8,
        "candidate_ratios_center_at_unity",
        diagnostics,
        nearest[candidate_fields].to_dict(),
    )
    check(
        abs(
            coordinate.beta_contact_raw_projection
            + coordinate.beta_HS_path_raw_projection
            - coordinate.beta_total_raw_projection
        )
        < 2e-10,
        "raw_metric_contact_HS_identity",
        diagnostics,
        float(coordinate.beta_total_raw_projection),
    )
    check(
        abs(
            coordinate.beta_contact_adjusted_projection
            + coordinate.beta_HS_path_adjusted_projection
            - coordinate.beta_total_adjusted_projection
        )
        < 2e-10,
        "adjusted_metric_contact_HS_identity",
        diagnostics,
        float(coordinate.beta_total_adjusted_projection),
    )
    check(
        abs(coordinate.A_raw / base.Vcem_fraction - coordinate.B_raw - base.beta_raw_projection)
        < 2e-7,
        "physics_factored_raw_local_slope",
        diagnostics,
        float(coordinate.A_raw / base.Vcem_fraction - coordinate.B_raw),
    )
    check(
        abs(
            coordinate.A_adjusted / base.Vcem_fraction
            - coordinate.B_adjusted
            - base.beta_adjusted_projection
        )
        < 2e-7,
        "physics_factored_adjusted_local_slope",
        diagnostics,
        float(coordinate.A_adjusted / base.Vcem_fraction - coordinate.B_adjusted),
    )
    phib0 = rc.PHIC_PACK - base.Vcem_fraction
    check(
        abs(
            coordinate.A_raw / base.Vcem_fraction
            - coordinate.Gamma_raw / phib0
            - base.beta_raw_projection
        )
        < 2e-7,
        "physics_factored_phib_raw_local_slope",
        diagnostics,
        float(coordinate.A_raw / base.Vcem_fraction - coordinate.Gamma_raw / phib0),
    )
    check(
        abs(
            coordinate.A_adjusted / base.Vcem_fraction
            - coordinate.Gamma_adjusted / phib0
            - base.beta_adjusted_projection
        )
        < 2e-7,
        "physics_factored_phib_adjusted_local_slope",
        diagnostics,
        float(
            coordinate.A_adjusted / base.Vcem_fraction
            - coordinate.Gamma_adjusted / phib0
        ),
    )
    check(
        candidate.physics_factored_raw_coordinate_ratio.max()
        / candidate.physics_factored_raw_coordinate_ratio.min()
        < 1.03,
        "physics_factored_raw_global_invariance",
        diagnostics,
        [
            float(candidate.physics_factored_raw_coordinate_ratio.min()),
            float(candidate.physics_factored_raw_coordinate_ratio.max()),
        ],
    )
    check(
        candidate.local_empirical_coordinate_ratio.max()
        / candidate.local_empirical_coordinate_ratio.min()
        > 1.5,
        "local_exponential_is_not_global_invariant",
        diagnostics,
        [
            float(candidate.local_empirical_coordinate_ratio.min()),
            float(candidate.local_empirical_coordinate_ratio.max()),
        ],
    )
    check(
        max(
            abs(cross.loc["19A", "pooled_physics_coordinate_raw_ratio"] - 1.0),
            abs(cross.loc["BT2", "pooled_physics_coordinate_raw_ratio"] - 1.0),
        )
        < 0.02,
        "pooled_coordinate_cross_trajectory_transfer",
        diagnostics,
        cross.pooled_physics_coordinate_raw_ratio.to_dict(),
    )
    check(
        max(cross.A_raw_local) / min(cross.A_raw_local) < 1.06,
        "A_raw_cross_trajectory_stability",
        diagnostics,
        cross.A_raw_local.to_dict(),
    )
    check(
        max(cross.Gamma_raw_local) / min(cross.Gamma_raw_local) < 1.06,
        "Gamma_raw_cross_trajectory_stability",
        diagnostics,
        cross.Gamma_raw_local.to_dict(),
    )

    for stem in [
        "Fig_E1_parameter_geometry",
        "Fig_E1_physics_decomposition",
        "Fig_E1_ridge_validation",
        "Fig_E1_candidate_invariance",
    ]:
        png = ROOT / "results" / "figures" / f"{stem}.png"
        pdf = ROOT / "results" / "figures" / f"{stem}.pdf"
        with Image.open(png) as image:
            image.verify()
        check(png.stat().st_size > 50_000 and pdf.stat().st_size > 10_000, f"figure_integrity_{stem}", diagnostics, [png.stat().st_size, pdf.stat().st_size])

    result = {
        "status": "PASS",
        "checks_passed": len(diagnostics),
        "diagnostics": diagnostics,
    }
    out = ROOT / "results" / "verification" / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    e1.write_manifest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
