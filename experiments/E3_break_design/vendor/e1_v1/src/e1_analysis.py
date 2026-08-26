from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0, str(ROOT / "vendor" / "rpia_v1"))
import rpia_core as rc


MODEL = "constant_cement"
NAMES = ("Vp", "Vs", "rho")
S_V = float(rc.PARAM_SCALES[MODEL][0])
S_L = float(rc.PARAM_SCALES[MODEL][1])
BETA_REFERENCE = 20.3
TARGET_BOUNDS = (np.array([0.001, np.log(3.0)]), np.array([0.060, np.log(18.0)]))

COLORS = {
    "navy": "#17324D",
    "blue": "#3478A6",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C94C4C",
    "gray": "#6B7280",
    "light": "#EEF2F5",
}


def _mkdirs() -> None:
    for path in [
        ROOT / "results" / "tables",
        ROOT / "results" / "figures",
        ROOT / "results" / "verification",
        ROOT / ".mplconfig",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    meta = json.loads((ROOT / "data" / "compact" / "rpia_metadata.json").read_text())
    wells: dict[str, pd.DataFrame] = {}
    for well, filename in [
        ("19A", "19A_training_window.csv"),
        ("BT2", "BT2_training_window.csv"),
    ]:
        raw = pd.read_csv(ROOT / "data" / "compact" / filename)
        wells[well] = rc.select_hugin_sand(
            raw,
            meta["hugin_intervals_m"][well],
            meta["selection"]["phi_min"],
            meta["selection"]["phi_max"],
            meta["selection"]["vsh_max"],
        )
    pooled = pd.concat([wells["19A"], wells["BT2"]], ignore_index=True)
    return wells, pooled, meta


def transfer_discrepancy(
    wells: dict[str, pd.DataFrame], calibrations: dict[str, np.ndarray]
) -> np.ndarray:
    errors = []
    for train, target in [("19A", "BT2"), ("BT2", "19A")]:
        errors.append(
            rc.forward(wells[target], MODEL, calibrations[train])
            - rc.observed(wells[target])
        )
    return np.sqrt(np.nanmean(np.vstack(errors) ** 2, axis=0))


def transfer_aware_scale(discrepancy: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.sqrt(rc.MEAS[name] ** 2 + discrepancy[i] ** 2))
        for i, name in enumerate(NAMES)
    }


def stacked_observed(df: pd.DataFrame, names: tuple[str, ...] = NAMES) -> np.ndarray:
    observed = rc.observed(df)
    index = {"Vp": 0, "Vs": 1, "rho": 2}
    return np.concatenate([observed[:, index[name]] for name in names])


def stacked_sigma(df: pd.DataFrame, scale: dict[str, float], names: tuple[str, ...] = NAMES) -> np.ndarray:
    return np.concatenate([np.full(len(df), scale[name]) for name in names])


def weighted_target_optimum(
    df: pd.DataFrame,
    scale: dict[str, float],
    start: np.ndarray,
    names: tuple[str, ...] = NAMES,
) -> np.ndarray:
    obs = stacked_observed(df, names)
    sig = stacked_sigma(df, scale, names)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = rc.stack(df, MODEL, theta, names)
        if not np.all(np.isfinite(prediction)):
            return np.full_like(obs, 1e6)
        return (prediction - obs) / sig

    return least_squares(
        residual,
        start,
        bounds=TARGET_BOUNDS,
        max_nfev=3000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    ).x


def schur_geometry(Jt: np.ndarray, Jn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    G = Jt.T @ Jt
    C = Jt.T @ Jn
    N = Jn.T @ Jn
    Gadj = G - C @ np.linalg.solve(N + np.eye(N.shape[0]), C.T)
    return (G + G.T) / 2.0, (Gadj + Gadj.T) / 2.0


def beta_from_eigenvector(G: np.ndarray) -> float:
    values, vectors = np.linalg.eigh(G)
    weak = vectors[:, np.argmin(values)]
    if abs(weak[0]) < 1e-14:
        return np.nan
    return float(-(S_L / S_V) * weak[1] / weak[0])


def beta_from_projection(G: np.ndarray) -> float:
    if abs(G[1, 1]) < 1e-20:
        return np.nan
    return float((S_L / S_V) * G[0, 1] / G[1, 1])


def strong_coordinate_angle(beta: float, reference: float = BETA_REFERENCE) -> float:
    a = np.array([beta * S_V, S_L], dtype=float)
    b = np.array([reference * S_V, S_L], dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine = float(np.clip(abs(a @ b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def metric_summary(
    df: pd.DataFrame,
    theta: np.ndarray,
    scale: dict[str, float],
    names: tuple[str, ...] = NAMES,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    Jt, Jn, nuisance_names = rc.jac(df, MODEL, theta, names, {n: scale[n] for n in names})
    G, Gadj = schur_geometry(Jt, Jn)
    eval_raw = np.linalg.eigvalsh(G)
    eval_adj = np.linalg.eigvalsh(Gadj)

    def correlation(M: np.ndarray) -> float:
        denominator = math.sqrt(max(M[0, 0] * M[1, 1], 0.0))
        return float(M[0, 1] / denominator) if denominator > 0 else np.nan

    raw_rho = correlation(G)
    adj_rho = correlation(Gadj)
    _, _, retention, _ = rc.geom(Jt, Jn)
    output: dict[str, float] = {
        "beta_raw_projection": beta_from_projection(G),
        "beta_raw_eigen": beta_from_eigenvector(G),
        "beta_adjusted_projection": beta_from_projection(Gadj),
        "beta_adjusted_eigen": beta_from_eigenvector(Gadj),
        "raw_lambda_min": float(eval_raw[0]),
        "raw_lambda_max": float(eval_raw[-1]),
        "adjusted_lambda_min": float(eval_adj[0]),
        "adjusted_lambda_max": float(eval_adj[-1]),
        "raw_spectral_ratio": float(eval_raw[0] / eval_raw[-1]),
        "adjusted_spectral_ratio": float(eval_adj[0] / eval_adj[-1]),
        "raw_rho": raw_rho,
        "adjusted_rho": adj_rho,
        "raw_angular_independence": float(max(0.0, 1.0 - raw_rho**2)),
        "adjusted_angular_independence": float(max(0.0, 1.0 - adj_rho**2)),
        "raw_condition": float(math.sqrt(eval_raw[-1] / eval_raw[0])),
        "adjusted_condition": float(math.sqrt(eval_adj[-1] / eval_adj[0])),
        "retention_min": float(retention[-1]),
        "retention_max": float(retention[0]),
    }

    n = len(df)
    block = {name: i for i, name in enumerate(names)}
    for prop in ("Vp", "Vs"):
        if prop not in block:
            continue
        rows = slice(block[prop] * n, (block[prop] + 1) * n)
        denominator = Jt[rows, 1]
        valid = np.abs(denominator) > 1e-12
        beta = (S_L / S_V) * Jt[rows, 0][valid] / denominator[valid]
        output[f"beta_{prop}_median"] = float(np.median(beta))
        output[f"beta_{prop}_q25"] = float(np.quantile(beta, 0.25))
        output[f"beta_{prop}_q75"] = float(np.quantile(beta, 0.75))

    output["strong_coordinate_angle_to_20p3_deg"] = strong_coordinate_angle(
        output["beta_raw_eigen"]
    )
    return output, Jt, Jn, G, Gadj, nuisance_names


def contact_endpoint_from_a(
    K0: float,
    G0: float,
    Kc: float,
    Gc: float,
    phic: float,
    Cn: float,
    a: float,
) -> tuple[float, float]:
    nu0 = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0)
    nuc = (3 * Kc - 2 * Gc) / (6 * Kc + 2 * Gc)
    LN = 2 * Gc * (1 - nu0) * (1 - nuc) / (np.pi * G0 * (1 - 2 * nuc))
    Sn = (
        (-0.024153 * LN ** -1.3646) * a * a
        + (0.20405 * LN ** -0.89008) * a
        + 0.00024649 * LN ** -1.9864
    )
    LT = Gc / (np.pi * G0)
    T1 = -1e-2 * (2.26 * nu0**2 + 2.07 * nu0 + 2.3) * LT ** (
        0.079 * nu0**2 + 0.1754 * nu0 - 1.342
    )
    T2 = (0.0573 * nu0**2 + 0.0937 * nu0 + 0.202) * LT ** (
        0.0274 * nu0**2 + 0.0529 * nu0 - 0.8765
    )
    T3 = 1e-4 * (9.654 * nu0**2 + 4.945 * nu0 + 3.1) * LT ** (
        0.01867 * nu0**2 + 0.4011 * nu0 - 1.8186
    )
    St = T1 * a * a + T2 * a + T3
    Kb = (1.0 / 6.0) * Cn * (1 - phic) * (Kc + 4 * Gc / 3) * Sn
    Gb = 3 * Kb / 5 + (3.0 / 20.0) * Cn * (1 - phic) * Gc * St
    return float(Kb), float(Gb)


def endpoint_diagnostics(K0: float, G0: float, vcem: float, Cn: float) -> dict[str, float]:
    phic = rc.PHIC_PACK
    Kc, Gc = rc.KCEM, rc.GCEM
    a = 2 * (vcem / (3 * Cn * (1 - phic))) ** 0.25
    nu0 = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0)
    nuc = (3 * Kc - 2 * Gc) / (6 * Kc + 2 * Gc)
    LN = 2 * Gc * (1 - nu0) * (1 - nuc) / (np.pi * G0 * (1 - 2 * nuc))
    n2 = -0.024153 * LN ** -1.3646
    n1 = 0.20405 * LN ** -0.89008
    n0 = 0.00024649 * LN ** -1.9864
    Sn = n2 * a**2 + n1 * a + n0
    mN = (2 * n2 * a**2 + n1 * a) / Sn

    LT = Gc / (np.pi * G0)
    t2 = -1e-2 * (2.26 * nu0**2 + 2.07 * nu0 + 2.3) * LT ** (
        0.079 * nu0**2 + 0.1754 * nu0 - 1.342
    )
    t1 = (0.0573 * nu0**2 + 0.0937 * nu0 + 0.202) * LT ** (
        0.0274 * nu0**2 + 0.0529 * nu0 - 0.8765
    )
    t0 = 1e-4 * (9.654 * nu0**2 + 4.945 * nu0 + 3.1) * LT ** (
        0.01867 * nu0**2 + 0.4011 * nu0 - 1.8186
    )
    St = t2 * a**2 + t1 * a + t0
    mT = (2 * t2 * a**2 + t1 * a) / St

    Kb = (1.0 / 6.0) * Cn * (1 - phic) * (Kc + 4 * Gc / 3) * Sn
    tangential = (3.0 / 20.0) * Cn * (1 - phic) * Gc * St
    Gb = 3 * Kb / 5 + tangential
    mG = ((3 * Kb / 5) * mN + tangential * mT) / Gb
    betaK = mN / (vcem * (4 - mN))
    betaG = mG / (vcem * (4 - mG))
    return {
        "a_c": float(a),
        "m_N": float(mN),
        "m_T": float(mT),
        "m_G": float(mG),
        "m_N_minus_m_G": float(mN - mG),
        "beta_endpoint_K": float(betaK),
        "beta_endpoint_G": float(betaG),
        "endpoint_log_jacobian_det": float((mN - mG) / (4 * vcem)),
    }


def dry_from_latent(
    K0: float,
    G0: float,
    phi: float,
    loga: float,
    ell: float,
    phib: float,
) -> np.ndarray:
    Kb, Gb = contact_endpoint_from_a(
        K0, G0, rc.KCEM, rc.GCEM, rc.PHIC_PACK, float(np.exp(ell)), float(np.exp(loga))
    )
    T = phi / phib
    Z = Gb / 6 * (9 * Kb + 8 * Gb) / (Kb + 2 * Gb)
    Kd = (T / (Kb + 4 * Gb / 3) + (1 - T) / (K0 + 4 * Gb / 3)) ** -1 - 4 * Gb / 3
    Gd = (T / (Gb + Z) + (1 - T) / (G0 + Z)) ** -1 - Z
    return np.array([Kd, Gd], dtype=float)


def dry_sensitivity_decomposition(
    K0: float,
    G0: float,
    phi: float,
    vcem: float,
    Cn: float,
) -> dict[str, float]:
    ell = float(np.log(Cn))
    phib = rc.PHIC_PACK - vcem
    a = 2 * (vcem / (3 * Cn * (1 - rc.PHIC_PACK))) ** 0.25
    loga = float(np.log(a))
    h = 1e-5
    hp = 1e-6

    def log_moduli(la: float, le: float, pb: float) -> np.ndarray:
        return np.log(dry_from_latent(K0, G0, phi, la, le, pb))

    A = (log_moduli(loga + h, ell, phib) - log_moduli(loga - h, ell, phib)) / (2 * h)
    B = (log_moduli(loga, ell, phib + hp) - log_moduli(loga, ell, phib - hp)) / (2 * hp)
    C = (log_moduli(loga, ell + h, phib) - log_moduli(loga, ell - h, phib)) / (2 * h)
    denominator = C - A / 4
    contact = (A / (4 * vcem)) / denominator
    hs_path = (-B) / denominator
    total = contact + hs_path
    return {
        "beta_contact_K": float(contact[0]),
        "beta_HS_path_K": float(hs_path[0]),
        "beta_total_K": float(total[0]),
        "beta_contact_G": float(contact[1]),
        "beta_HS_path_G": float(hs_path[1]),
        "beta_total_G": float(total[1]),
        "dlnK_dln_a": float(A[0]),
        "dlnG_dln_a": float(A[1]),
        "dlnK_dphib": float(B[0]),
        "dlnG_dphib": float(B[1]),
        "dlnK_dlnCn_fixed": float(C[0]),
        "dlnG_dlnCn_fixed": float(C[1]),
    }


def latent_stacked_outputs(
    df: pd.DataFrame,
    loga: float,
    ell: float,
    phib: float,
    names: tuple[str, ...] = NAMES,
) -> np.ndarray:
    output = []
    for row in df.itertuples(index=False):
        Km, Gm, rhom = rc.matrix(float(row.vsh))
        Kd, Gd = dry_from_latent(
            Km, Gm, float(row.phi), loga, ell, phib
        )
        output.append(
            rc.elastic(
                Kd,
                Gd,
                Km,
                rhom,
                float(row.phi),
                float(row.sw),
            )
        )
    output = np.asarray(output)
    index = {"Vp": 0, "Vs": 1, "rho": 2}
    return np.concatenate([output[:, index[name]] for name in names])


def metric_contact_hs_decomposition(
    df: pd.DataFrame,
    theta: np.ndarray,
    scale: dict[str, float],
) -> dict[str, float]:
    vcem, ell = map(float, theta)
    Cn = float(np.exp(ell))
    phib = rc.PHIC_PACK - vcem
    a = 2 * (vcem / (3 * Cn * (1 - rc.PHIC_PACK))) ** 0.25
    loga = float(np.log(a))
    h = 1e-5
    hp = 1e-6
    A = (
        latent_stacked_outputs(df, loga + h, ell, phib)
        - latent_stacked_outputs(df, loga - h, ell, phib)
    ) / (2 * h)
    B = (
        latent_stacked_outputs(df, loga, ell, phib + hp)
        - latent_stacked_outputs(df, loga, ell, phib - hp)
    ) / (2 * hp)
    C = (
        latent_stacked_outputs(df, loga, ell + h, phib)
        - latent_stacked_outputs(df, loga, ell - h, phib)
    ) / (2 * h)
    sigma = stacked_sigma(df, scale)
    j_contact = A / (4 * vcem) / sigma
    j_hs = -B / sigma
    j_ell = (C - A / 4) / sigma
    _, Jn, _ = rc.jac(df, MODEL, theta, NAMES, scale)
    projection = np.eye(len(j_ell)) - Jn @ np.linalg.solve(
        Jn.T @ Jn + np.eye(Jn.shape[1]), Jn.T
    )

    output = {}
    for label, P in [("raw", np.eye(len(j_ell))), ("adjusted", projection)]:
        denominator = float(j_ell @ P @ j_ell)
        beta_contact = float(j_ell @ P @ j_contact / denominator)
        beta_hs = float(j_ell @ P @ j_hs / denominator)
        output[f"beta_contact_{label}_projection"] = beta_contact
        output[f"beta_HS_path_{label}_projection"] = beta_hs
        output[f"beta_total_{label}_projection"] = beta_contact + beta_hs
        output[f"A_{label}"] = vcem * beta_contact
        output[f"B_{label}"] = -beta_hs
        output[f"Gamma_{label}"] = phib * (-beta_hs)
    output["Vcem_reference"] = vcem
    output["Cn_reference"] = Cn
    return output


def baseline_matrices(df: pd.DataFrame) -> list[tuple[float, float]]:
    return [rc.matrix(float(vsh))[:2] for vsh in df.vsh.to_numpy()]


def median_physical_diagnostics(df: pd.DataFrame, theta: np.ndarray) -> dict[str, float]:
    vcem, ell = map(float, theta)
    Cn = float(np.exp(ell))
    endpoint_rows = []
    dry_rows = []
    for (K0, G0), phi in zip(baseline_matrices(df), df.phi.to_numpy()):
        endpoint_rows.append(endpoint_diagnostics(K0, G0, vcem, Cn))
        dry_rows.append(dry_sensitivity_decomposition(K0, G0, float(phi), vcem, Cn))
    out: dict[str, float] = {}
    for key in endpoint_rows[0]:
        out[f"median_{key}"] = float(np.median([row[key] for row in endpoint_rows]))
    for key in dry_rows[0]:
        out[f"median_{key}"] = float(np.median([row[key] for row in dry_rows]))
    return out


def operating_points_table(
    wells: dict[str, pd.DataFrame],
    pooled: pd.DataFrame,
    calibrations: dict[str, np.ndarray],
    pooled_theta: np.ndarray,
    weighted_theta: np.ndarray,
    scale: dict[str, float],
) -> pd.DataFrame:
    specifications = [
        ("19A", wells["19A"], calibrations["19A"]),
        ("BT2", wells["BT2"], calibrations["BT2"]),
        ("pooled_RPIA", pooled, pooled_theta),
        ("pooled_transfer_weighted", pooled, weighted_theta),
    ]
    rows = []
    for label, df, theta in specifications:
        metric, *_ = metric_summary(df, theta, scale)
        physics = median_physical_diagnostics(df, theta)
        rows.append(
            {
                "operating_point": label,
                "N": len(df),
                "Vcem_fraction": float(theta[0]),
                "Vcem_percent": float(100 * theta[0]),
                "lnCn": float(theta[1]),
                "Cn": float(np.exp(theta[1])),
                "phi_b": float(rc.PHIC_PACK - theta[0]),
                **metric,
                **physics,
            }
        )
    return pd.DataFrame(rows)


def observable_staircase(
    pooled: pd.DataFrame, theta: np.ndarray, scale: dict[str, float]
) -> pd.DataFrame:
    sets = [
        ("Vp", ("Vp",)),
        ("Vs", ("Vs",)),
        ("Vp+Vs", ("Vp", "Vs")),
        ("Vp+Vs+rho", NAMES),
    ]
    rows = []
    for label, names in sets:
        metric, Jt, *_ = metric_summary(pooled, theta, scale, names)
        rows.append(
            {
                "observable_set": label,
                "target_jacobian_norm": float(np.linalg.norm(Jt)),
                "density_target_jacobian_norm": (
                    float(np.linalg.norm(Jt[-len(pooled) :, :])) if "rho" in names else np.nan
                ),
                **metric,
            }
        )
    return pd.DataFrame(rows)


def parameter_grid(
    pooled: pd.DataFrame,
    scale: dict[str, float],
    data_optimum: np.ndarray,
    quick: bool,
) -> pd.DataFrame:
    nv, nc = (11, 9) if quick else (31, 29)
    v_values = np.linspace(0.001, 0.060, nv)
    cn_values = np.geomspace(3.0, 18.0, nc)
    obs = stacked_observed(pooled)
    sig = stacked_sigma(pooled, scale)

    def objective(theta: np.ndarray) -> float:
        residual = (rc.stack(pooled, MODEL, theta, NAMES) - obs) / sig
        return float(residual @ residual)

    objective_min = objective(data_optimum)
    Kmed = float(np.median([x[0] for x in baseline_matrices(pooled)]))
    Gmed = float(np.median([x[1] for x in baseline_matrices(pooled)]))
    rows: list[dict[str, float]] = []
    for iv, vcem in enumerate(v_values):
        for Cn in cn_values:
            theta = np.array([vcem, np.log(Cn)])
            metric, *_ = metric_summary(pooled, theta, scale)
            endpoint = endpoint_diagnostics(Kmed, Gmed, float(vcem), float(Cn))
            rows.append(
                {
                    "Vcem_fraction": float(vcem),
                    "Vcem_percent": float(100 * vcem),
                    "lnCn": float(np.log(Cn)),
                    "Cn": float(Cn),
                    "delta_data_objective": float(objective(theta) - objective_min),
                    **metric,
                    **endpoint,
                }
            )
        if iv % max(1, nv // 6) == 0 or iv == nv - 1:
            print(f"parameter grid: {iv + 1}/{nv} Vcem rows", flush=True)
    return pd.DataFrame(rows)


def phi_decomposition_grid(
    pooled: pd.DataFrame,
    pooled_theta: np.ndarray,
    quick: bool,
) -> pd.DataFrame:
    nv, np_ = (15, 15) if quick else (60, 57)
    v_values = np.linspace(0.001, 0.060, nv)
    phi_values = np.linspace(0.08, 0.30, np_)
    Cn = float(np.exp(pooled_theta[1]))
    K0, G0 = rc.matrix(float(pooled.vsh.median()))[:2]
    rows = []
    for vcem in v_values:
        endpoint = endpoint_diagnostics(K0, G0, float(vcem), Cn)
        for phi in phi_values:
            decomposition = dry_sensitivity_decomposition(
                K0, G0, float(phi), float(vcem), Cn
            )
            rows.append(
                {
                    "Vcem_fraction": float(vcem),
                    "Vcem_percent": float(100 * vcem),
                    "phi": float(phi),
                    "Cn_fixed": Cn,
                    "beta_endpoint_K": endpoint["beta_endpoint_K"],
                    "beta_endpoint_G": endpoint["beta_endpoint_G"],
                    **decomposition,
                }
            )
    return pd.DataFrame(rows)


def interpolated_ridge(
    grid: pd.DataFrame,
    beta_column: str,
    v_eval: np.ndarray,
    theta0: np.ndarray,
) -> np.ndarray:
    v_axis = np.sort(grid.Vcem_fraction.unique())
    l_axis = np.sort(grid.lnCn.unique())
    field = (
        grid.pivot(index="Vcem_fraction", columns="lnCn", values=beta_column)
        .reindex(index=v_axis, columns=l_axis)
        .to_numpy()
    )
    interp = RegularGridInterpolator(
        (v_axis, l_axis), field, bounds_error=False, fill_value=np.nan
    )
    v0, l0 = map(float, theta0)

    def rhs(v: float, y: np.ndarray) -> np.ndarray:
        vc = float(np.clip(v, v_axis[0], v_axis[-1]))
        lc = float(np.clip(y[0], l_axis[0], l_axis[-1]))
        beta = float(interp([[vc, lc]])[0])
        if not np.isfinite(beta):
            beta = 0.0
        return np.array([-beta])

    result = np.full_like(v_eval, np.nan, dtype=float)
    lower = np.sort(v_eval[v_eval <= v0])[::-1]
    upper = np.sort(v_eval[v_eval >= v0])
    if len(lower):
        sol = solve_ivp(rhs, (v0, float(lower[-1])), [l0], t_eval=lower, rtol=1e-7, atol=1e-9)
        for v, ell in zip(sol.t, sol.y[0]):
            result[np.argmin(np.abs(v_eval - v))] = ell
    if len(upper):
        sol = solve_ivp(rhs, (v0, float(upper[-1])), [l0], t_eval=upper, rtol=1e-7, atol=1e-9)
        for v, ell in zip(sol.t, sol.y[0]):
            result[np.argmin(np.abs(v_eval - v))] = ell
    outside = (result < np.log(3.0)) | (result > np.log(18.0))
    result[outside] = np.nan
    return result


def ridge_profiles(
    pooled: pd.DataFrame,
    scale: dict[str, float],
    pooled_theta: np.ndarray,
    data_optimum: np.ndarray,
    parameter_map: pd.DataFrame,
    quick: bool,
) -> pd.DataFrame:
    npoints = 17 if quick else 41
    v_eval = np.unique(np.r_[np.linspace(0.001, 0.060, npoints), pooled_theta[0]])
    v_eval.sort()
    obs = stacked_observed(pooled)
    sig = stacked_sigma(pooled, scale)
    pred0 = rc.stack(pooled, MODEL, pooled_theta, NAMES)
    base_metric, _, _, _, _, nuisance_names = metric_summary(pooled, pooled_theta, scale)

    def raw_structural_objective(vcem: float, ell: float) -> float:
        pred = rc.stack(pooled, MODEL, np.array([vcem, ell]), NAMES)
        residual = (pred - pred0) / sig
        return float(residual @ residual)

    def data_objective(vcem: float, ell: float) -> float:
        pred = rc.stack(pooled, MODEL, np.array([vcem, ell]), NAMES)
        residual = (pred - obs) / sig
        return float(residual @ residual)

    data_min = data_objective(float(data_optimum[0]), float(data_optimum[1]))
    base_nuisance_scales = np.array([rc.NUI_SCALES[n] for n in nuisance_names])

    def profiled_structural_residual(vcem: float, x: np.ndarray) -> np.ndarray:
        ell = float(x[0])
        z = np.asarray(x[1:], dtype=float)
        nuisance = {
            name: float(scale_i * zi)
            for name, scale_i, zi in zip(nuisance_names, base_nuisance_scales, z)
        }
        pred = rc.stack(pooled, MODEL, np.array([vcem, ell]), NAMES, nuisance)
        if not np.all(np.isfinite(pred)):
            return np.full(len(pred0) + len(z), 1e6)
        return np.r_[(pred - pred0) / sig, z]

    raw_profile: dict[float, tuple[float, float]] = {}
    data_profile: dict[float, tuple[float, float]] = {}
    for vcem in v_eval:
        raw = minimize_scalar(
            lambda ell: raw_structural_objective(float(vcem), float(ell)),
            bounds=(np.log(3.0), np.log(18.0)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        dat = minimize_scalar(
            lambda ell: data_objective(float(vcem), float(ell)),
            bounds=(np.log(3.0), np.log(18.0)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        raw_profile[float(vcem)] = (float(raw.x), float(raw.fun))
        data_profile[float(vcem)] = (float(dat.x), float(dat.fun - data_min))

    profiled: dict[float, tuple[np.ndarray, float, bool]] = {}
    x_center = np.r_[pooled_theta[1], np.zeros(len(nuisance_names))]
    profiled[float(pooled_theta[0])] = (x_center.copy(), 0.0, True)
    low = sorted([float(v) for v in v_eval if v < pooled_theta[0]], reverse=True)
    high = sorted([float(v) for v in v_eval if v > pooled_theta[0]])
    lower_bounds = np.r_[np.log(3.0), np.full(len(nuisance_names), -4.0)]
    upper_bounds = np.r_[np.log(18.0), np.full(len(nuisance_names), 4.0)]
    for direction in (low, high):
        x_start = x_center.copy()
        for vcem in direction:
            fit = least_squares(
                lambda x: profiled_structural_residual(vcem, x),
                x_start,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=350,
                xtol=2e-9,
                ftol=2e-9,
                gtol=2e-9,
            )
            profiled[vcem] = (fit.x.copy(), float(fit.fun @ fit.fun), bool(fit.success))
            x_start = fit.x.copy()
        print(f"completed nonlinear nuisance profile: {len(direction)} points", flush=True)

    ode_raw = interpolated_ridge(
        parameter_map, "beta_raw_eigen", v_eval, pooled_theta
    )
    ode_adjusted = interpolated_ridge(
        parameter_map, "beta_adjusted_eigen", v_eval, pooled_theta
    )
    local_ref = pooled_theta[1] - BETA_REFERENCE * (v_eval - pooled_theta[0])
    local_raw = pooled_theta[1] - base_metric["beta_raw_eigen"] * (v_eval - pooled_theta[0])
    local_adj = pooled_theta[1] - base_metric["beta_adjusted_eigen"] * (v_eval - pooled_theta[0])

    rows = []
    for i, vcem in enumerate(v_eval):
        key = float(vcem)
        xprof, prof_objective, success = profiled[key]
        row = {
            "Vcem_fraction": key,
            "Vcem_percent": 100 * key,
            "lnCn_raw_structural_profile": raw_profile[key][0],
            "Cn_raw_structural_profile": float(np.exp(raw_profile[key][0])),
            "objective_raw_structural_profile": raw_profile[key][1],
            "lnCn_nuisance_profiled_structural": float(xprof[0]),
            "Cn_nuisance_profiled_structural": float(np.exp(xprof[0])),
            "objective_nuisance_profiled_structural": prof_objective,
            "nuisance_prior_norm": float(np.linalg.norm(xprof[1:])),
            "nuisance_profile_success": success,
            "lnCn_observed_data_profile": data_profile[key][0],
            "Cn_observed_data_profile": float(np.exp(data_profile[key][0])),
            "delta_objective_observed_data": data_profile[key][1],
            "lnCn_local_20p3": float(local_ref[i]),
            "lnCn_local_raw": float(local_raw[i]),
            "lnCn_local_adjusted": float(local_adj[i]),
            "lnCn_integrated_raw": float(ode_raw[i]) if np.isfinite(ode_raw[i]) else np.nan,
            "lnCn_integrated_adjusted": float(ode_adjusted[i]) if np.isfinite(ode_adjusted[i]) else np.nan,
        }
        for name, value in zip(nuisance_names, xprof[1:]):
            row[f"profiled_{name}_sigma"] = float(value)
        rows.append(row)
    out = pd.DataFrame(rows)
    for source, destination in [
        ("lnCn_raw_structural_profile", "beta_tangent_raw_structural"),
        ("lnCn_nuisance_profiled_structural", "beta_tangent_nuisance_profiled"),
        ("lnCn_observed_data_profile", "beta_tangent_observed_data"),
    ]:
        out[destination] = -np.gradient(out[source].to_numpy(), out.Vcem_fraction.to_numpy())
    return out


def candidate_state_invariance(
    pooled: pd.DataFrame,
    pooled_theta: np.ndarray,
    ridge: pd.DataFrame,
    coordinate: dict[str, float],
) -> pd.DataFrame:
    v0, ell0 = map(float, pooled_theta)
    Cn0 = float(np.exp(ell0))
    matrices = baseline_matrices(pooled)
    phi = pooled.phi.to_numpy(dtype=float)
    endpoint0 = []
    dry0 = []
    a0 = endpoint_diagnostics(matrices[0][0], matrices[0][1], v0, Cn0)["a_c"]
    for (K0, G0), phii in zip(matrices, phi):
        Kb, Gb = contact_endpoint_from_a(
            K0, G0, rc.KCEM, rc.GCEM, rc.PHIC_PACK, Cn0, a0
        )
        Kd, Gd = dry_from_latent(K0, G0, float(phii), np.log(a0), ell0, rc.PHIC_PACK - v0)
        endpoint0.append((Kb, Gb))
        dry0.append((Kd, Gd))
    endpoint0 = np.asarray(endpoint0)
    dry0 = np.asarray(dry0)
    prediction0 = rc.forward(pooled, MODEL, pooled_theta)
    T_ratio_denominator = 1.0 / (rc.PHIC_PACK - v0)

    rows = []
    for _, profile in ridge.iterrows():
        vcem = float(profile.Vcem_fraction)
        ell = float(profile.lnCn_raw_structural_profile)
        Cn = float(np.exp(ell))
        a = endpoint_diagnostics(matrices[0][0], matrices[0][1], vcem, Cn)["a_c"]
        endpoint = []
        dry = []
        for (K0, G0), phii in zip(matrices, phi):
            Kb, Gb = contact_endpoint_from_a(
                K0, G0, rc.KCEM, rc.GCEM, rc.PHIC_PACK, Cn, a
            )
            Kd, Gd = dry_from_latent(
                K0, G0, float(phii), np.log(a), ell, rc.PHIC_PACK - vcem
            )
            endpoint.append((Kb, Gb))
            dry.append((Kd, Gd))
        endpoint = np.asarray(endpoint)
        dry = np.asarray(dry)
        prediction = rc.forward(pooled, MODEL, np.array([vcem, ell]))
        rows.append(
            {
                "Vcem_fraction": vcem,
                "Vcem_percent": 100 * vcem,
                "Cn_on_raw_structural_ridge": Cn,
                "local_empirical_coordinate_ratio": float(
                    (Cn / Cn0) * np.exp(BETA_REFERENCE * (vcem - v0))
                ),
                "physics_factored_raw_coordinate_ratio": float(
                    (Cn / Cn0)
                    * (vcem / v0) ** coordinate["A_raw"]
                    * (
                        (rc.PHIC_PACK - vcem) / (rc.PHIC_PACK - v0)
                    )
                    ** coordinate["Gamma_raw"]
                ),
                "physics_factored_adjusted_coordinate_ratio": float(
                    (float(profile.Cn_nuisance_profiled_structural) / Cn0)
                    * (vcem / v0) ** coordinate["A_adjusted"]
                    * (
                        (rc.PHIC_PACK - vcem) / (rc.PHIC_PACK - v0)
                    )
                    ** coordinate["Gamma_adjusted"]
                ),
                "physics_linearized_raw_coordinate_ratio": float(
                    (Cn / Cn0)
                    * (vcem / v0) ** coordinate["A_raw"]
                    * np.exp(-coordinate["B_raw"] * (vcem - v0))
                ),
                "physics_linearized_adjusted_coordinate_ratio": float(
                    (float(profile.Cn_nuisance_profiled_structural) / Cn0)
                    * (vcem / v0) ** coordinate["A_adjusted"]
                    * np.exp(-coordinate["B_adjusted"] * (vcem - v0))
                ),
                "a_c_ratio": float(a / a0),
                "median_Kb_ratio": float(np.median(endpoint[:, 0] / endpoint0[:, 0])),
                "median_Gb_ratio": float(np.median(endpoint[:, 1] / endpoint0[:, 1])),
                "T_ratio": float((1.0 / (rc.PHIC_PACK - vcem)) / T_ratio_denominator),
                "median_Kd_ratio": float(np.median(dry[:, 0] / dry0[:, 0])),
                "median_Gd_ratio": float(np.median(dry[:, 1] / dry0[:, 1])),
                "median_Vp_ratio": float(np.median(prediction[:, 0] / prediction0[:, 0])),
                "median_Vs_ratio": float(np.median(prediction[:, 1] / prediction0[:, 1])),
                "rms_Vp_change_mps": float(np.sqrt(np.mean((prediction[:, 0] - prediction0[:, 0]) ** 2))),
                "rms_Vs_change_mps": float(np.sqrt(np.mean((prediction[:, 1] - prediction0[:, 1]) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def cross_trajectory_coordinates(
    wells: dict[str, pd.DataFrame],
    pooled: pd.DataFrame,
    operating: pd.DataFrame,
    scale: dict[str, float],
    pooled_coordinate: dict[str, float],
) -> pd.DataFrame:
    points = operating.set_index("operating_point")
    reference = points.loc["pooled_RPIA"]
    v0 = float(reference.Vcem_fraction)
    Cn0 = float(reference.Cn)
    phib0 = rc.PHIC_PACK - v0
    specifications = [
        ("19A", wells["19A"]),
        ("BT2", wells["BT2"]),
        ("pooled_RPIA", pooled),
    ]
    rows = []
    for label, df in specifications:
        point = points.loc[label]
        vcem = float(point.Vcem_fraction)
        Cn = float(point.Cn)
        theta = np.array([vcem, float(point.lnCn)])
        local_coordinate = metric_contact_hs_decomposition(df, theta, scale)
        rows.append(
            {
                "trajectory": label,
                "Vcem_fraction": vcem,
                "Vcem_percent": 100 * vcem,
                "Cn": Cn,
                "beta_raw_projection": local_coordinate["beta_total_raw_projection"],
                "beta_adjusted_projection": local_coordinate[
                    "beta_total_adjusted_projection"
                ],
                "A_raw_local": local_coordinate["A_raw"],
                "Gamma_raw_local": local_coordinate["Gamma_raw"],
                "A_adjusted_local": local_coordinate["A_adjusted"],
                "Gamma_adjusted_local": local_coordinate["Gamma_adjusted"],
                "pooled_physics_coordinate_raw_ratio": float(
                    (Cn / Cn0)
                    * (vcem / v0) ** pooled_coordinate["A_raw"]
                    * ((rc.PHIC_PACK - vcem) / phib0)
                    ** pooled_coordinate["Gamma_raw"]
                ),
                "pooled_physics_coordinate_adjusted_ratio": float(
                    (Cn / Cn0)
                    * (vcem / v0) ** pooled_coordinate["A_adjusted"]
                    * ((rc.PHIC_PACK - vcem) / phib0)
                    ** pooled_coordinate["Gamma_adjusted"]
                ),
                "pooled_local_20p3_coordinate_ratio": float(
                    (Cn / Cn0) * np.exp(BETA_REFERENCE * (vcem - v0))
                ),
            }
        )
    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, value: str, y: str = "Cn") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.sort(df.Vcem_percent.unique())
    yy = np.sort(df[y].unique())
    z = df.pivot(index=y, columns="Vcem_percent", values=value).reindex(index=yy, columns=x).to_numpy()
    return x, yy, z


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "figure.titlesize": 13,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )


def plot_parameter_geometry(
    grid: pd.DataFrame,
    operating: pd.DataFrame,
    weighted_theta: np.ndarray,
) -> None:
    configure_plotting()
    fields = [
        ("beta_raw_eigen", r"Raw $\beta_{\rm eig}$", "beta"),
        ("beta_adjusted_eigen", r"Nuisance-adjusted $\beta_{\rm eig}$", "beta"),
        ("adjusted_spectral_ratio", r"$\log_{10}(\lambda_{\min}/\lambda_{\max})$", "log"),
        ("adjusted_angular_independence", r"$\log_{10}(1-\rho_{\rm adj}^2)$", "log"),
    ]
    all_beta = np.r_[grid.beta_raw_eigen.to_numpy(), grid.beta_adjusted_eigen.to_numpy()]
    beta_limit = float(np.nanquantile(np.abs(all_beta), 0.985))
    beta_limit = max(beta_limit, 25.0)
    beta_norm = SymLogNorm(linthresh=1.0, linscale=0.7, vmin=-beta_limit, vmax=beta_limit, base=10)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1), constrained_layout=True)
    points = operating[operating.operating_point.isin(["19A", "BT2", "pooled_RPIA"])]
    labels = {"19A": "19A", "BT2": "BT2", "pooled_RPIA": "pooled"}
    for ax, (field, title, mode) in zip(axes.flat, fields):
        x, y, z = _pivot(grid, field)
        if mode == "beta":
            image = ax.pcolormesh(x, y, z, shading="auto", cmap="RdBu_r", norm=beta_norm)
            if np.nanmin(z) <= BETA_REFERENCE <= np.nanmax(z):
                contour = ax.contour(x, y, z, levels=[BETA_REFERENCE], colors="black", linewidths=1.0)
                ax.clabel(contour, fmt={BETA_REFERENCE: "20.3"}, fontsize=7.5)
            cbar = fig.colorbar(image, ax=ax, pad=0.015)
            cbar.set_label(r"slope $\beta$ (per Vcem fraction)")
        else:
            logz = np.log10(np.maximum(z, 1e-16))
            image = ax.pcolormesh(x, y, logz, shading="auto", cmap="viridis")
            cbar = fig.colorbar(image, ax=ax, pad=0.015)
            cbar.set_label(title)

        _, _, delta = _pivot(grid, "delta_data_objective")
        if np.nanmin(delta) <= 2.30 <= np.nanmax(delta):
            ax.contour(x, y, delta, levels=[2.30], colors="white", linewidths=1.3)
        if np.nanmin(delta) <= 6.18 <= np.nanmax(delta):
            ax.contour(x, y, delta, levels=[6.18], colors="white", linestyles="--", linewidths=1.1)
        for _, point in points.iterrows():
            ax.scatter(point.Vcem_percent, point.Cn, s=42, c="white", edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(labels[point.operating_point], (point.Vcem_percent, point.Cn), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color="black")
        ax.scatter(100 * weighted_theta[0], np.exp(weighted_theta[1]), marker="*", s=90, c=COLORS["gold"], edgecolors="black", linewidths=0.7, zorder=6)
        ax.set_yscale("log")
        ax.set_yticks([3, 4, 6, 9, 12, 18])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
        ax.set_ylabel(r"Coordination number $C_n$")
        ax.set_title(title)
    fig.suptitle(
        "Local identifiable direction and residual separability\n"
        + r"white contours: transfer-weighted fit distance $\Delta\Phi=2.30$ (solid), 6.18 (dashed); star: weighted optimum"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "figures" / f"Fig_E1_parameter_geometry.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_physics_decomposition(
    phi_grid: pd.DataFrame,
    pooled_theta: np.ndarray,
    observed_phi: tuple[float, float],
    median_phi: float,
) -> None:
    configure_plotting()
    fields = [
        ("beta_contact_K", r"$K_d$: contact-radius term", "positive"),
        ("beta_HS_path_K", r"$K_d$: HS-path term", "path"),
        ("beta_total_K", r"$K_d$: total $\beta_K$", "total"),
        ("beta_contact_G", r"$G_d$: contact-radius term", "positive"),
        ("beta_HS_path_G", r"$G_d$: HS-path term", "path"),
        ("beta_total_G", r"$G_d$: total $\beta_G$", "total"),
    ]
    positive = np.r_[
        phi_grid.beta_contact_K,
        phi_grid.beta_total_K,
        phi_grid.beta_contact_G,
        phi_grid.beta_total_G,
    ]
    pos_norm = LogNorm(vmin=max(1.0, float(np.nanquantile(positive, 0.01))), vmax=float(np.nanquantile(positive, 0.99)))
    signed = np.r_[phi_grid.beta_HS_path_K, phi_grid.beta_HS_path_G]
    max_abs = max(1.0, float(np.nanquantile(np.abs(signed), 0.99)))
    signed_norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    total = np.r_[phi_grid.beta_total_K, phi_grid.beta_total_G]
    total_limit = max(25.0, float(np.nanquantile(np.abs(total), 0.99)))
    total_norm = SymLogNorm(
        linthresh=1.0, linscale=0.7, vmin=-total_limit, vmax=total_limit, base=10
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.1), constrained_layout=True, sharex=True, sharey=True)
    for ax, (field, title, mode) in zip(axes.flat, fields):
        x, y, z = _pivot(phi_grid, field, y="phi")
        if mode == "positive":
            image = ax.pcolormesh(x, y, z, shading="auto", cmap="magma", norm=pos_norm)
        elif mode == "path":
            image = ax.pcolormesh(x, y, z, shading="auto", cmap="RdBu_r", norm=signed_norm)
            if np.nanmin(z) < 0 < np.nanmax(z):
                ax.contour(x, y, z, levels=[0], colors="black", linewidths=0.7)
        else:
            image = ax.pcolormesh(x, y, z, shading="auto", cmap="RdBu_r", norm=total_norm)
            if np.nanmin(z) <= BETA_REFERENCE <= np.nanmax(z):
                contour = ax.contour(x, y, z, levels=[BETA_REFERENCE], colors="cyan", linewidths=1.1)
                ax.clabel(contour, fmt={BETA_REFERENCE: "20.3"}, fontsize=7.5)
            if np.nanmin(z) < 0 < np.nanmax(z):
                ax.contour(x, y, z, levels=[0], colors="black", linewidths=0.8)
        ax.axhspan(0.08, observed_phi[0], color="white", alpha=0.28, lw=0)
        ax.axhspan(observed_phi[1], 0.30, color="white", alpha=0.28, lw=0)
        ax.axhline(observed_phi[0], color="white", lw=0.7, ls="--")
        ax.axhline(observed_phi[1], color="white", lw=0.7, ls="--")
        ax.scatter(100 * pooled_theta[0], median_phi, marker="*", s=70, c=COLORS["teal"], edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_title(title)
        ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
        ax.set_ylabel(r"Porosity $\phi$")
        cbar = fig.colorbar(image, ax=ax, pad=0.012)
        cbar.set_label(r"contribution to $\beta$ (per fraction)")
    fig.suptitle(
        r"Why the full constant-cement slope differs from the contact endpoint ($C_n$ fixed at pooled value)"
        + "\n"
        + r"shaded bands: extrapolation beyond selected Hugin porosity range; cyan contour: $\beta=20.3$"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "figures" / f"Fig_E1_physics_decomposition.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_validation(ridge: pd.DataFrame, pooled_theta: np.ndarray) -> None:
    configure_plotting()
    v = ridge.Vcem_percent.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
    ax = axes[0]
    ax.plot(v, ridge.lnCn_raw_structural_profile, color=COLORS["navy"], lw=2.2, label="structural profile, nuisances fixed")
    ax.plot(v, ridge.lnCn_nuisance_profiled_structural, color=COLORS["red"], lw=2.2, label="structural profile, nuisances profiled")
    ax.plot(v, ridge.lnCn_integrated_raw, color=COLORS["blue"], lw=1.4, ls="--", label=r"integrated $\beta_{\rm eig,raw}$")
    ax.plot(v, ridge.lnCn_integrated_adjusted, color=COLORS["orange"], lw=1.4, ls="--", label=r"integrated $\beta_{\rm eig,adj}$")
    ax.plot(v, ridge.lnCn_local_20p3, color=COLORS["gray"], lw=1.1, ls=":", label=r"local $\beta=20.3$")
    ax.scatter([100 * pooled_theta[0]], [pooled_theta[1]], marker="*", s=100, c=COLORS["gold"], edgecolors="black", zorder=5)
    ax.axhline(np.log(3), color="0.65", lw=0.7)
    ax.axhline(np.log(18), color="0.65", lw=0.7)
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel(r"$\ln C_n$")
    ax.set_title("Finite ridge geometry")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    ax.plot(v, ridge.objective_raw_structural_profile, color=COLORS["navy"], lw=2, label="fixed nuisances")
    ax.plot(v, ridge.objective_nuisance_profiled_structural, color=COLORS["red"], lw=2, label="profiled nuisances")
    ax.plot(v, np.maximum(ridge.delta_objective_observed_data, 0), color=COLORS["teal"], lw=1.6, label="observed-data target profile")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel(r"profile objective increment $\Delta\Phi$")
    ax.set_title("Ridge flatness")
    ax.legend(frameon=False)

    ax = axes[2]
    for field, label, color in [
        ("beta_tangent_raw_structural", "fixed-nuisance profile tangent", COLORS["navy"]),
        ("beta_tangent_nuisance_profiled", "nuisance-profiled tangent", COLORS["red"]),
        ("beta_tangent_observed_data", "observed-data tangent", COLORS["teal"]),
    ]:
        values = ridge[field].to_numpy().copy()
        values[(values <= 0) | (values > 1000)] = np.nan
        ax.plot(v, values, lw=1.8, label=label, color=color)
    ax.axhline(BETA_REFERENCE, color=COLORS["gray"], lw=1.1, ls=":", label="20.3")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel(r"local tangent $\beta=-d\ln C_n/dV_{\rm cem}$")
    ax.set_title("The slope is local, not universal")
    ax.legend(frameon=False)
    fig.suptitle("Nonlinear validation of the empirical identifiable coordinate")
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "figures" / f"Fig_E1_ridge_validation.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_invariance(
    candidate: pd.DataFrame,
    pooled_theta: np.ndarray,
    cross_trajectory: pd.DataFrame,
) -> None:
    configure_plotting()
    x = candidate.Vcem_percent.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.3), constrained_layout=True)
    ax = axes[0]
    for field, label, color, style in [
        ("local_empirical_coordinate_ratio", r"local $C_ne^{20.3(V-V_0)}$", COLORS["gray"], ":"),
        ("physics_factored_raw_coordinate_ratio", r"physics-factored, raw", COLORS["navy"], "-"),
        ("physics_factored_adjusted_coordinate_ratio", r"physics-factored, adjusted", COLORS["red"], "--"),
    ]:
        ax.plot(x, candidate[field], label=label, color=color, ls=style, lw=2.2)
    ax.axhline(1.0, color="0.75", lw=0.8)
    ax.axvline(100 * pooled_theta[0], color="0.75", lw=0.8)
    for _, point in cross_trajectory[cross_trajectory.trajectory != "pooled_RPIA"].iterrows():
        ax.plot(
            [point.Vcem_percent, point.Vcem_percent],
            [
                point.pooled_physics_coordinate_raw_ratio,
                point.pooled_local_20p3_coordinate_ratio,
            ],
            color="0.65",
            lw=0.8,
            zorder=3,
        )
        ax.scatter(
            point.Vcem_percent,
            point.pooled_physics_coordinate_raw_ratio,
            s=42,
            marker="o",
            facecolor="white",
            edgecolor=COLORS["navy"],
            linewidth=1.3,
            zorder=5,
        )
        ax.scatter(
            point.Vcem_percent,
            point.pooled_local_20p3_coordinate_ratio,
            s=40,
            marker="x",
            color=COLORS["gray"],
            linewidth=1.3,
            zorder=5,
        )
        ax.annotate(
            point.trajectory,
            (point.Vcem_percent, point.pooled_physics_coordinate_raw_ratio),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel("coordinate ratio to pooled point")
    ax.set_title("Local exponential versus factored coordinate")
    ax.legend(frameon=False)

    ax = axes[1]
    for field, label, color, style in [
        ("a_c_ratio", r"contact radius $a_c$", COLORS["orange"], "-"),
        ("median_Kb_ratio", r"contact endpoint $K_b$", COLORS["navy"], "-"),
        ("median_Gb_ratio", r"contact endpoint $G_b$", COLORS["blue"], "--"),
        ("T_ratio", r"HS coordinate $T=\phi/\phi_b$", COLORS["teal"], "-."),
    ]:
        ax.plot(x, candidate[field], label=label, color=color, ls=style, lw=2.0)
    ax.axhline(1.0, color="0.75", lw=0.8)
    ax.axvline(100 * pooled_theta[0], color="0.75", lw=0.8)
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel("ratio to pooled operating point")
    ax.set_title("Upstream states are not invariant")
    ax.legend(frameon=False)

    ax = axes[2]
    for field, label, color, style in [
        ("median_Kd_ratio", r"dry bulk modulus $K_d$", COLORS["navy"], "-"),
        ("median_Gd_ratio", r"dry shear modulus $G_d$", COLORS["blue"], "--"),
        ("median_Vp_ratio", r"saturated $V_P$", COLORS["red"], "-"),
        ("median_Vs_ratio", r"saturated $V_S$", COLORS["teal"], "--"),
    ]:
        ax.plot(x, candidate[field], label=label, color=color, ls=style, lw=2.0)
    ax.axhline(1.0, color="0.65", lw=0.8)
    ax.axvline(100 * pooled_theta[0], color="0.75", lw=0.8)
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel("ratio to pooled operating point")
    ax.set_title("Downstream elastic states remain nearly invariant")
    ax.legend(frameon=False)
    fig.suptitle("What is actually conserved along the nonlinear constant-cement ridge?")
    for suffix in ("png", "pdf"):
        fig.savefig(
            ROOT / "results" / "figures" / f"Fig_E1_candidate_invariance.{suffix}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)


def summarize_results(
    wells: dict[str, pd.DataFrame],
    pooled: pd.DataFrame,
    discrepancy: np.ndarray,
    scale: dict[str, float],
    operating: pd.DataFrame,
    staircase: pd.DataFrame,
    grid: pd.DataFrame,
    phi_grid: pd.DataFrame,
    ridge: pd.DataFrame,
    candidate: pd.DataFrame,
    coordinate: dict[str, float],
    cross_trajectory: pd.DataFrame,
) -> dict:
    base = operating.set_index("operating_point").loc["pooled_RPIA"]
    practical = grid[
        grid.Vcem_fraction.between(0.002, 0.035)
        & grid.Cn.between(4.0, 12.0)
        & (grid.delta_data_objective <= 6.18)
    ]
    if practical.empty:
        practical = grid[grid.Vcem_fraction.between(0.002, 0.035) & grid.Cn.between(4.0, 12.0)]
    phi_near = phi_grid.iloc[
        ((phi_grid.Vcem_fraction - base.Vcem_fraction).abs() + (phi_grid.phi - pooled.phi.median()).abs()).argsort()[:1]
    ].iloc[0]
    local = ridge[(ridge.Vcem_fraction - base.Vcem_fraction).abs() <= 0.006]

    def rms(a: pd.Series, b: pd.Series) -> float:
        valid = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))

    vpvs = staircase.set_index("observable_set")
    summary = {
        "experiment": "E1 Discover-Explain constant-cement geometry",
        "model": "constant_cement Scheme 1, frozen RPIA v1 forward model",
        "sample_counts": {"19A": len(wells["19A"]), "BT2": len(wells["BT2"]), "pooled": len(pooled)},
        "selected_phi_range": [float(pooled.phi.min()), float(pooled.phi.max())],
        "transfer_discrepancy": {name: float(discrepancy[i]) for i, name in enumerate(NAMES)},
        "transfer_aware_sigma": scale,
        "pooled_operating_point": {
            key: float(base[key])
            for key in [
                "Vcem_fraction",
                "Cn",
                "beta_raw_projection",
                "beta_raw_eigen",
                "beta_adjusted_projection",
                "beta_adjusted_eigen",
                "beta_Vp_median",
                "beta_Vs_median",
                "median_beta_endpoint_K",
                "median_beta_endpoint_G",
                "median_beta_total_K",
                "median_beta_total_G",
                "median_beta_contact_K",
                "median_beta_HS_path_K",
                "median_beta_contact_G",
                "median_beta_HS_path_G",
                "raw_lambda_min",
                "adjusted_lambda_min",
                "adjusted_spectral_ratio",
                "adjusted_angular_independence",
                "retention_min",
            ]
        },
        "practical_supported_grid": {
            "n_cells": int(len(practical)),
            "beta_raw_eigen_min": float(practical.beta_raw_eigen.min()),
            "beta_raw_eigen_median": float(practical.beta_raw_eigen.median()),
            "beta_raw_eigen_max": float(practical.beta_raw_eigen.max()),
            "beta_adjusted_eigen_min": float(practical.beta_adjusted_eigen.min()),
            "beta_adjusted_eigen_median": float(practical.beta_adjusted_eigen.median()),
            "beta_adjusted_eigen_max": float(practical.beta_adjusted_eigen.max()),
            "adjusted_spectral_ratio_max": float(practical.adjusted_spectral_ratio.max()),
            "adjusted_angular_independence_max": float(practical.adjusted_angular_independence.max()),
        },
        "full_grid": {
            "beta_raw_eigen_min": float(grid.beta_raw_eigen.min()),
            "beta_raw_eigen_max": float(grid.beta_raw_eigen.max()),
            "beta_adjusted_eigen_min": float(grid.beta_adjusted_eigen.min()),
            "beta_adjusted_eigen_max": float(grid.beta_adjusted_eigen.max()),
            "negative_adjusted_beta_cells": int((grid.beta_adjusted_eigen < 0).sum()),
        },
        "physics_at_nearest_phi_grid_cell": {
            key: float(phi_near[key])
            for key in [
                "Vcem_fraction",
                "phi",
                "beta_endpoint_K",
                "beta_endpoint_G",
                "beta_contact_K",
                "beta_HS_path_K",
                "beta_total_K",
                "beta_contact_G",
                "beta_HS_path_G",
                "beta_total_G",
            ]
        },
        "density_control": {
            "raw_lambda_min_VpVs": float(vpvs.loc["Vp+Vs", "raw_lambda_min"]),
            "raw_lambda_min_VpVsrho": float(vpvs.loc["Vp+Vs+rho", "raw_lambda_min"]),
            "adjusted_lambda_min_VpVs": float(vpvs.loc["Vp+Vs", "adjusted_lambda_min"]),
            "adjusted_lambda_min_VpVsrho": float(vpvs.loc["Vp+Vs+rho", "adjusted_lambda_min"]),
            "density_target_jacobian_norm": float(vpvs.loc["Vp+Vs+rho", "density_target_jacobian_norm"]),
        },
        "finite_ridge": {
            "local_lnCn_rms_raw_profile_vs_20p3": rms(local.lnCn_raw_structural_profile, local.lnCn_local_20p3),
            "local_lnCn_rms_raw_profile_vs_integrated_raw": rms(local.lnCn_raw_structural_profile, local.lnCn_integrated_raw),
            "local_lnCn_rms_profiled_vs_integrated_adjusted": rms(local.lnCn_nuisance_profiled_structural, local.lnCn_integrated_adjusted),
            "max_profiled_nuisance_prior_norm": float(ridge.nuisance_prior_norm.max()),
            "all_nuisance_profiles_converged": bool(ridge.nuisance_profile_success.all()),
        },
        "candidate_state_invariance": {
            field: {
                "min": float(candidate[field].min()),
                "max": float(candidate[field].max()),
            }
            for field in [
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
        },
        "physics_factored_coordinate": coordinate,
        "cross_trajectory_coordinate": {
            row.trajectory: {
                key: float(getattr(row, key))
                for key in [
                    "beta_raw_projection",
                    "beta_adjusted_projection",
                    "A_raw_local",
                    "Gamma_raw_local",
                    "A_adjusted_local",
                    "Gamma_adjusted_local",
                    "pooled_physics_coordinate_raw_ratio",
                    "pooled_physics_coordinate_adjusted_ratio",
                    "pooled_local_20p3_coordinate_ratio",
                ]
            }
            for row in cross_trajectory.itertuples()
        },
    }
    return summary


def write_results_markdown(summary: dict, operating: pd.DataFrame) -> None:
    p = summary["pooled_operating_point"]
    grid = summary["full_grid"]
    supported = summary["practical_supported_grid"]
    ridge = summary["finite_ridge"]
    invariant = summary["candidate_state_invariance"]
    coordinate = summary["physics_factored_coordinate"]
    cross = summary["cross_trajectory_coordinate"]
    text = f"""# E1 result: Discover–Explain

This experiment uses the frozen RPIA constant-cement Scheme 1 forward model, the 59 selected Hugin samples, fixed target scales `diag(0.015, 0.20)`, transfer-aware data covariance, and the original unit-Gaussian nuisance priors.

## Main result

At the pooled RPIA operating point (`Vcem={100*p['Vcem_fraction']:.3f}%`, `Cn={p['Cn']:.3f}`), the empirical identifiable-coordinate slope is reproduced by the complete physics:

- raw eigen-direction: **{p['beta_raw_eigen']:.3f}**;
- nuisance-adjusted eigen-direction: **{p['beta_adjusted_eigen']:.3f}**;
- median `Vp` sensitivity ratio: **{p['beta_Vp_median']:.3f}**;
- median `Vs` sensitivity ratio: **{p['beta_Vs_median']:.3f}**.

The contact endpoint alone is steeper (`beta_Kb={p['median_beta_endpoint_K']:.2f}`, `beta_Gb={p['median_beta_endpoint_G']:.2f}`). Propagation through the constant-cement HS path lowers the dry-modulus slopes to `beta_Kd={p['median_beta_total_K']:.2f}` and `beta_Gd={p['median_beta_total_G']:.2f}`. The median decomposition is `{p['median_beta_contact_K']:.2f} + ({p['median_beta_HS_path_K']:.2f}) = {p['median_beta_total_K']:.2f}` for bulk modulus and `{p['median_beta_contact_G']:.2f} + ({p['median_beta_HS_path_G']:.2f}) = {p['median_beta_total_G']:.2f}` for shear modulus.

Therefore, **20.3 is physically explainable but not universal**. Across the complete calibration box, the raw eigen-slope spans `{grid['beta_raw_eigen_min']:.2f}` to `{grid['beta_raw_eigen_max']:.2f}`, and the adjusted slope spans `{grid['beta_adjusted_eigen_min']:.2f}` to `{grid['beta_adjusted_eigen_max']:.2f}`. Within the transfer-supported practical cells, the raw range is narrower: `{supported['beta_raw_eigen_min']:.2f}` to `{supported['beta_raw_eigen_max']:.2f}`.

## Separability result

The pooled adjusted minimum information is `{p['adjusted_lambda_min']:.4g}`, the adjusted spectral ratio is `{p['adjusted_spectral_ratio']:.4g}`, and the angular-independence factor is `{p['adjusted_angular_independence']:.4g}`. Thus the full model is formally rank two, but the second direction remains extremely weak after nuisance adjustment.

Density has zero direct target sensitivity in this forward model. It can change adjusted information only by constraining nuisances; this is verified in `E1_observable_staircase.csv`.

## Finite-ridge result

Near the pooled point, the RMS discrepancy in `ln Cn` between the raw structural profile and the local 20.3 exponential is `{ridge['local_lnCn_rms_raw_profile_vs_20p3']:.4g}`. Integrating the spatially varying raw slope reduces this to `{ridge['local_lnCn_rms_raw_profile_vs_integrated_raw']:.4g}`. The nuisance-profiled structural ridge versus the integrated adjusted slope has local RMS `{ridge['local_lnCn_rms_profiled_vs_integrated_adjusted']:.4g}`.

The sensitivity decomposition gives `beta_contact={coordinate['beta_contact_raw_projection']:.3f}` and `beta_HS={coordinate['beta_HS_path_raw_projection']:.3f}` in the raw metric. Approximating these terms as `A/Vcem` and `-Gamma/phi_b` gives `A={coordinate['A_raw']:.5f}` and `Gamma={coordinate['Gamma_raw']:.5f}`, and therefore the physics-factored coordinate

`q_phys = ln(Cn/Cn0) + A ln(Vcem/V0) + Gamma ln(phi_b/phi_b0)`.

Its exponential remains between `{invariant['physics_factored_raw_coordinate_ratio']['min']:.4f}` and `{invariant['physics_factored_raw_coordinate_ratio']['max']:.4f}` along the complete raw structural ridge. The nuisance-adjusted version remains between `{invariant['physics_factored_adjusted_coordinate_ratio']['min']:.4f}` and `{invariant['physics_factored_adjusted_coordinate_ratio']['max']:.4f}` along the nuisance-profiled ridge. By contrast, the local 20.3 exponential varies from `{invariant['local_empirical_coordinate_ratio']['min']:.3f}` to `{invariant['local_empirical_coordinate_ratio']['max']:.3f}`. The simpler `-B(Vcem-V0)` term is retained in the tables as the local linearization of `Gamma ln(phi_b/phi_b0)`.

As an internal cross-trajectory check, the independently calibrated 19A and BT2 optima give pooled physics-coordinate ratios `{cross['19A']['pooled_physics_coordinate_raw_ratio']:.4f}` and `{cross['BT2']['pooled_physics_coordinate_raw_ratio']:.4f}`, respectively. Their local 20.3 ratios are `{cross['19A']['pooled_local_20p3_coordinate_ratio']:.4f}` and `{cross['BT2']['pooled_local_20p3_coordinate_ratio']:.4f}`. Independently recomputed raw coefficients are `A={cross['19A']['A_raw_local']:.4f}, Gamma={cross['19A']['Gamma_raw_local']:.4f}` for 19A and `A={cross['BT2']['A_raw_local']:.4f}, Gamma={cross['BT2']['Gamma_raw_local']:.4f}` for BT2. This supports Hugin-level transferability, but remains an internal validation within the same constitutive model.

The strongest interpretation supported by E1 is therefore: static elastic data identify a **nonlinear constant-cement trajectory coordinate**, jointly generated by cemented-contact stiffness and the HS porosity path. The physics-factored expression is a strong approximate invariant for this experiment, but its coefficients remain operating-point, weighting, and model dependent; it is not yet a universal material state. The residual second direction is too weak to treat `Vcem` and `Cn` as independently recovered microstructural parameters.

Along the full raw structural ridge, the centered local empirical coordinate varies from `{invariant['local_empirical_coordinate_ratio']['min']:.3f}` to `{invariant['local_empirical_coordinate_ratio']['max']:.3f}` and the contact radius from `{invariant['a_c_ratio']['min']:.3f}` to `{invariant['a_c_ratio']['max']:.3f}` of its pooled value. In contrast, the median dry shear modulus remains between `{invariant['median_Gd_ratio']['min']:.5f}` and `{invariant['median_Gd_ratio']['max']:.5f}`. This is direct evidence that the finite invariant is a compensated trajectory through contact and HS state, not either nominal parameter or the contact endpoint alone.

## Files

- `results/tables/E1_operating_points.csv`
- `results/tables/E1_observable_staircase.csv`
- `results/tables/E1_vcem_cn_map.csv`
- `results/tables/E1_vcem_phi_map.csv`
- `results/tables/E1_ridge_validation.csv`
- `results/tables/E1_candidate_state_invariance.csv`
- `results/tables/E1_coordinate_coefficients.csv`
- `results/tables/E1_cross_trajectory_coordinate.csv`
- `results/figures/Fig_E1_parameter_geometry.*`
- `results/figures/Fig_E1_physics_decomposition.*`
- `results/figures/Fig_E1_ridge_validation.*`
- `results/figures/Fig_E1_candidate_invariance.*`
"""
    (ROOT / "results" / "RESULTS.md").write_text(text)


def write_manifest() -> None:
    rows = []
    excluded = {"SHA256SUMS.txt", "MANIFEST.csv"}
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name in excluded
            or ".mplconfig" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(ROOT / "MANIFEST.csv", index=False)
    lines = [f"{row.sha256}  {row.path}" for row in manifest.itertuples()]
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def run(quick: bool = False) -> dict:
    _mkdirs()
    wells, pooled, _ = load_data()
    calibrations = {name: rc.calibrate(df, MODEL) for name, df in wells.items()}
    pooled_theta = rc.calibrate(pooled, MODEL)
    discrepancy = transfer_discrepancy(wells, calibrations)
    scale = transfer_aware_scale(discrepancy)
    weighted_theta = weighted_target_optimum(pooled, scale, pooled_theta)

    operating = operating_points_table(
        wells, pooled, calibrations, pooled_theta, weighted_theta, scale
    )
    operating.to_csv(ROOT / "results" / "tables" / "E1_operating_points.csv", index=False)
    staircase = observable_staircase(pooled, pooled_theta, scale)
    staircase.to_csv(ROOT / "results" / "tables" / "E1_observable_staircase.csv", index=False)

    grid = parameter_grid(pooled, scale, weighted_theta, quick)
    grid.to_csv(ROOT / "results" / "tables" / "E1_vcem_cn_map.csv", index=False)
    phi_grid = phi_decomposition_grid(pooled, pooled_theta, quick)
    phi_grid.to_csv(ROOT / "results" / "tables" / "E1_vcem_phi_map.csv", index=False)
    ridge = ridge_profiles(pooled, scale, pooled_theta, weighted_theta, grid, quick)
    ridge.to_csv(ROOT / "results" / "tables" / "E1_ridge_validation.csv", index=False)
    coordinate = metric_contact_hs_decomposition(pooled, pooled_theta, scale)
    pd.DataFrame([coordinate]).to_csv(
        ROOT / "results" / "tables" / "E1_coordinate_coefficients.csv", index=False
    )
    candidate = candidate_state_invariance(pooled, pooled_theta, ridge, coordinate)
    candidate.to_csv(
        ROOT / "results" / "tables" / "E1_candidate_state_invariance.csv", index=False
    )
    cross_trajectory = cross_trajectory_coordinates(
        wells, pooled, operating, scale, coordinate
    )
    cross_trajectory.to_csv(
        ROOT / "results" / "tables" / "E1_cross_trajectory_coordinate.csv",
        index=False,
    )

    plot_parameter_geometry(grid, operating, weighted_theta)
    plot_physics_decomposition(
        phi_grid,
        pooled_theta,
        (float(pooled.phi.min()), float(pooled.phi.max())),
        float(pooled.phi.median()),
    )
    plot_ridge_validation(ridge, pooled_theta)
    plot_candidate_invariance(candidate, pooled_theta, cross_trajectory)

    summary = summarize_results(
        wells,
        pooled,
        discrepancy,
        scale,
        operating,
        staircase,
        grid,
        phi_grid,
        ridge,
        candidate,
        coordinate,
        cross_trajectory,
    )
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    write_results_markdown(summary, operating)
    write_manifest()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
