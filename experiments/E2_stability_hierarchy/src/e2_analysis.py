from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = ROOT / "vendor" / "e1_v1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(E1_ROOT / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

import e1_analysis as e1

rc = e1.rc
MODEL = e1.MODEL
NAMES = e1.NAMES
TARGET_BOUNDS = e1.TARGET_BOUNDS
SCALES = np.array([e1.S_V, e1.S_L], dtype=float)
PHIC = float(rc.PHIC_PACK)

COLORS = {
    "navy": "#17324D",
    "blue": "#3478A6",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C94C4C",
    "purple": "#7C5AA6",
    "gray": "#6B7280",
    "light": "#EEF2F5",
}


@dataclass
class FitResult:
    theta: np.ndarray
    objective: float
    success: bool
    nfev: int
    bound_hit: bool


def _mkdirs() -> None:
    for path in [
        ROOT / "results" / "tables",
        ROOT / "results" / "figures",
        ROOT / "results" / "verification",
        ROOT / ".mplconfig",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _bound_hit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float = 2e-5) -> bool:
    span = np.maximum(hi - lo, 1.0)
    return bool(np.any((x - lo) <= tol * span) or np.any((hi - x) <= tol * span))


def rp_grid_fit(df: pd.DataFrame, start: np.ndarray) -> FitResult:
    lo, hi = TARGET_BOUNDS
    fit = least_squares(
        lambda theta: rc.residual(theta, df, MODEL),
        np.clip(np.asarray(start, dtype=float), lo + 1e-8, hi - 1e-8),
        bounds=(lo, hi),
        max_nfev=3000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    residual = rc.residual(fit.x, df, MODEL)
    return FitResult(
        fit.x.copy(),
        float(residual @ residual),
        bool(fit.success),
        int(fit.nfev),
        _bound_hit(fit.x, lo, hi),
    )


def weighted_fit(
    df: pd.DataFrame,
    scale: dict[str, float],
    start: np.ndarray,
    names: tuple[str, ...] = NAMES,
) -> FitResult:
    observed = e1.stacked_observed(df, names)
    sigma = e1.stacked_sigma(df, scale, names)
    lo, hi = TARGET_BOUNDS

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = rc.stack(df, MODEL, theta, names)
        if not np.all(np.isfinite(prediction)):
            return np.full_like(observed, 1e6)
        return (prediction - observed) / sigma

    fit = least_squares(
        residual,
        np.clip(np.asarray(start, dtype=float), lo + 1e-8, hi - 1e-8),
        bounds=(lo, hi),
        max_nfev=3000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    r = residual(fit.x)
    return FitResult(
        fit.x.copy(),
        float(r @ r),
        bool(fit.success),
        int(fit.nfev),
        _bound_hit(fit.x, lo, hi),
    )


def load_baseline() -> dict:
    wells, pooled, metadata = e1.load_data()
    # Preserve the E1 operating points bit-for-bit: do not refine rc.calibrate's
    # returned solution with a second optimizer call.
    rp_fits = {}
    for name, df in wells.items():
        theta = rc.calibrate(df, MODEL)
        residual = rc.residual(theta, df, MODEL)
        rp_fits[name] = FitResult(
            theta.copy(),
            float(residual @ residual),
            True,
            -1,
            _bound_hit(theta, *TARGET_BOUNDS),
        )
    theta_pooled = rc.calibrate(pooled, MODEL)
    residual_pooled = rc.residual(theta_pooled, pooled, MODEL)
    rp_pooled = FitResult(
        theta_pooled.copy(),
        float(residual_pooled @ residual_pooled),
        True,
        -1,
        _bound_hit(theta_pooled, *TARGET_BOUNDS),
    )
    discrepancy = e1.transfer_discrepancy(
        wells, {name: fit.theta for name, fit in rp_fits.items()}
    )
    scale = e1.transfer_aware_scale(discrepancy)
    weighted = {
        name: weighted_fit(df, scale, rp_fits[name].theta)
        for name, df in wells.items()
    }
    weighted_pooled = weighted_fit(pooled, scale, rp_pooled.theta)
    coordinate_rpia = e1.metric_contact_hs_decomposition(
        pooled, rp_pooled.theta, scale
    )
    coordinate_weighted = e1.metric_contact_hs_decomposition(
        pooled, weighted_pooled.theta, scale
    )
    return {
        "wells": wells,
        "pooled": pooled,
        "metadata": metadata,
        "rp_fits": rp_fits,
        "rp_pooled": rp_pooled,
        "discrepancy": discrepancy,
        "scale": scale,
        "weighted": weighted,
        "weighted_pooled": weighted_pooled,
        "coordinate_rpia": coordinate_rpia,
        "coordinate_weighted": coordinate_weighted,
    }


def noncircular_moving_block_sample(
    df: pd.DataFrame, block_length: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, float]:
    n = len(df)
    length = int(max(1, min(block_length, n)))
    if length == 1:
        indices = rng.integers(0, n, size=n)
    else:
        starts = rng.integers(0, n - length + 1, size=int(math.ceil(n / length)))
        indices = np.concatenate([np.arange(s, s + length) for s in starts])[:n]
    sampled = df.iloc[indices].reset_index(drop=True)
    return sampled, float(len(np.unique(indices)) / n)


def residual_acf_table(
    wells: dict[str, pd.DataFrame], fits: dict[str, FitResult], max_lag: int = 10
) -> pd.DataFrame:
    rows: list[dict] = []
    for well, df in wells.items():
        residual = rc.forward(df, MODEL, fits[well].theta) - rc.observed(df)
        for ip, prop in enumerate(NAMES):
            x = residual[:, ip] - np.mean(residual[:, ip])
            denominator = float(x @ x)
            for lag in range(1, min(max_lag, len(x) - 1) + 1):
                acf = float(x[:-lag] @ x[lag:] / denominator) if denominator > 0 else np.nan
                rows.append(
                    {
                        "trajectory": well,
                        "property": prop,
                        "lag_samples": lag,
                        "lag_m": 4 * lag,
                        "residual_acf": acf,
                    }
                )
    return pd.DataFrame(rows)


def coordinate_ratio(
    theta: np.ndarray,
    reference: np.ndarray,
    A: float,
    Gamma: float,
) -> float:
    v, ell = map(float, theta)
    v0, ell0 = map(float, reference)
    dq = (
        ell
        - ell0
        + A * math.log(v / v0)
        + Gamma * math.log((PHIC - v) / (PHIC - v0))
    )
    return float(math.exp(dq))


def beta_from_factored_coordinate(vcem: float, A: float, Gamma: float) -> float:
    return float(A / vcem - Gamma / (PHIC - vcem))


def _hierarchical_parameterization(
    kind: str,
    x: np.ndarray,
    A: float,
    Gamma: float,
    reference: np.ndarray,
) -> dict[str, np.ndarray]:
    if kind == "separate":
        return {"19A": x[:2], "BT2": x[2:4]}
    if kind == "shared_Cn":
        return {
            "19A": np.array([x[0], x[2]], dtype=float),
            "BT2": np.array([x[1], x[2]], dtype=float),
        }
    if kind == "shared_qstar":
        vref = float(reference[0])
        phibref = PHIC - vref
        out: dict[str, np.ndarray] = {}
        for i, well in enumerate(("19A", "BT2")):
            v = float(x[i])
            ell = (
                float(x[2])
                - A * math.log(v / vref)
                - Gamma * math.log((PHIC - v) / phibref)
            )
            out[well] = np.array([v, ell], dtype=float)
        return out
    if kind == "pooled_theta":
        return {"19A": x[:2], "BT2": x[:2]}
    raise ValueError(f"Unknown hierarchical model: {kind}")


def _hierarchical_start_bounds(
    kind: str,
    separate_points: dict[str, np.ndarray],
    A: float,
    Gamma: float,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = np.asarray(separate_points["19A"], dtype=float)
    t2 = np.asarray(separate_points["BT2"], dtype=float)
    if kind == "separate":
        return (
            np.r_[t1, t2],
            np.r_[TARGET_BOUNDS[0], TARGET_BOUNDS[0]],
            np.r_[TARGET_BOUNDS[1], TARGET_BOUNDS[1]],
        )
    if kind == "shared_Cn":
        return (
            np.array([t1[0], t2[0], np.mean([t1[1], t2[1]])]),
            np.array([0.001, 0.001, math.log(3.0)]),
            np.array([0.060, 0.060, math.log(18.0)]),
        )
    if kind == "shared_qstar":
        q = [
            math.log(coordinate_ratio(t, reference, A, Gamma)) + float(reference[1])
            for t in (t1, t2)
        ]
        return (
            np.array([t1[0], t2[0], np.mean(q)]),
            np.array([0.001, 0.001, math.log(3.0)]),
            np.array([0.060, 0.060, math.log(18.0)]),
        )
    if kind == "pooled_theta":
        return (
            np.asarray(reference, dtype=float).copy(),
            TARGET_BOUNDS[0].copy(),
            TARGET_BOUNDS[1].copy(),
        )
    raise ValueError(kind)


def fit_hierarchical_data_model(
    wells: dict[str, pd.DataFrame],
    scale: dict[str, float],
    kind: str,
    separate_points: dict[str, np.ndarray],
    A: float,
    Gamma: float,
    reference: np.ndarray,
    adjusted: bool,
    max_nfev: int = 3500,
    multistart: bool = True,
) -> dict:
    start, lo, hi = _hierarchical_start_bounds(
        kind, separate_points, A, Gamma, reference
    )
    target_size = len(start)
    nuisance_names = rc.nuisance_names(MODEL)
    nuisance_scales = np.array([rc.NUI_SCALES[name] for name in nuisance_names])
    observed = {name: e1.stacked_observed(df) for name, df in wells.items()}
    sigma = {name: e1.stacked_sigma(df, scale) for name, df in wells.items()}
    data_size = int(sum(len(values) for values in observed.values()))

    if adjusted:
        start = np.r_[start, np.zeros(len(nuisance_names))]
        lo = np.r_[lo, np.full(len(nuisance_names), -4.0)]
        hi = np.r_[hi, np.full(len(nuisance_names), 4.0)]

    def residual(x: np.ndarray) -> np.ndarray:
        target_x = x[:target_size]
        theta = _hierarchical_parameterization(
            kind, target_x, A, Gamma, reference
        )
        z = x[target_size:] if adjusted else np.zeros(0)
        nuisance = (
            {
                name: float(z[i] * nuisance_scales[i])
                for i, name in enumerate(nuisance_names)
            }
            if adjusted
            else None
        )
        output = []
        for well in ("19A", "BT2"):
            if not (
                TARGET_BOUNDS[0][1]
                <= theta[well][1]
                <= TARGET_BOUNDS[1][1]
            ):
                return np.full(data_size + len(z), 1e6)
            prediction = rc.stack(wells[well], MODEL, theta[well], NAMES, nuisance)
            if not np.all(np.isfinite(prediction)):
                return np.full(data_size + len(z), 1e6)
            output.append((prediction - observed[well]) / sigma[well])
        if adjusted:
            output.append(z)
        return np.concatenate(output)

    candidates = [start]
    if multistart and kind != "separate":
        alternate = start.copy()
        alternate[: min(2, target_size)] = float(reference[0])
        candidates.append(np.clip(alternate, lo + 1e-7, hi - 1e-7))
    if multistart and adjusted:
        alternate = start.copy()
        alternate[: min(2, target_size)] = np.maximum(
            0.003, np.minimum(0.02, alternate[: min(2, target_size)])
        )
        candidates.append(np.clip(alternate, lo + 1e-7, hi - 1e-7))

    fits = []
    for candidate in candidates:
        fit = least_squares(
            residual,
            candidate,
            bounds=(lo, hi),
            max_nfev=max_nfev,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        r = residual(fit.x)
        fits.append((float(r @ r), fit, r))
    objective, fit, r = min(fits, key=lambda item: item[0])
    theta = _hierarchical_parameterization(
        kind, fit.x[:target_size], A, Gamma, reference
    )
    n_nuisance = len(nuisance_names) if adjusted else 0
    data_r = r[:-n_nuisance] if n_nuisance else r
    z = fit.x[target_size:] if adjusted else np.zeros(0)
    return {
        "model": kind,
        "adjustment": "nonlinear_MAP" if adjusted else "fixed_nuisance",
        "objective_total": objective,
        "objective_data": float(data_r @ data_r),
        "objective_nuisance_prior": float(z @ z),
        "success": bool(fit.success),
        "nfev": int(fit.nfev),
        "target_parameter_count": target_size,
        "nuisance_parameter_count": n_nuisance,
        "bound_hit": _bound_hit(fit.x, lo, hi),
        "theta_19A": theta["19A"].copy(),
        "theta_BT2": theta["BT2"].copy(),
        "target_x": fit.x[:target_size].copy(),
        "nuisance_z": z.copy(),
        "nuisance_names": nuisance_names,
    }


def fit_local_geometry_model(
    kind: str,
    independent_points: dict[str, np.ndarray],
    geometries: dict[str, np.ndarray],
    A: float,
    Gamma: float,
    reference: np.ndarray,
    adjustment: str,
) -> dict:
    start, lo, hi = _hierarchical_start_bounds(
        kind, independent_points, A, Gamma, reference
    )

    square_roots = {}
    for well, matrix in geometries.items():
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        square_roots[well] = np.diag(np.sqrt(np.maximum(values, 0.0))) @ vectors.T

    def residual(x: np.ndarray) -> np.ndarray:
        theta = _hierarchical_parameterization(kind, x, A, Gamma, reference)
        output = []
        for well in ("19A", "BT2"):
            if not TARGET_BOUNDS[0][1] <= theta[well][1] <= TARGET_BOUNDS[1][1]:
                return np.full(4, 1e6)
            delta = (theta[well] - independent_points[well]) / SCALES
            output.append(square_roots[well] @ delta)
        return np.concatenate(output)

    fit = least_squares(
        residual,
        start,
        bounds=(lo, hi),
        max_nfev=3000,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    r = residual(fit.x)
    theta = _hierarchical_parameterization(kind, fit.x, A, Gamma, reference)
    return {
        "model": kind,
        "adjustment": adjustment,
        "objective_total": float(r @ r),
        "objective_data": float(r @ r),
        "objective_nuisance_prior": 0.0,
        "success": bool(fit.success),
        "nfev": int(fit.nfev),
        "target_parameter_count": len(start),
        "nuisance_parameter_count": 0,
        "bound_hit": _bound_hit(fit.x, lo, hi),
        "theta_19A": theta["19A"].copy(),
        "theta_BT2": theta["BT2"].copy(),
        "target_x": fit.x.copy(),
        "nuisance_z": np.zeros(0),
        "nuisance_names": [],
    }


def _hierarchy_flat_row(result: dict) -> dict:
    t1 = result["theta_19A"]
    t2 = result["theta_BT2"]
    row = {
        key: result[key]
        for key in [
            "model",
            "adjustment",
            "objective_total",
            "objective_data",
            "objective_nuisance_prior",
            "success",
            "nfev",
            "target_parameter_count",
            "nuisance_parameter_count",
            "bound_hit",
        ]
    }
    row.update(
        {
            "Vcem_19A": float(t1[0]),
            "Cn_19A": float(math.exp(t1[1])),
            "lnCn_19A": float(t1[1]),
            "Vcem_BT2": float(t2[0]),
            "Cn_BT2": float(math.exp(t2[1])),
            "lnCn_BT2": float(t2[1]),
            "delta_Vcem_BT2_minus_19A": float(t2[0] - t1[0]),
            "delta_lnCn_BT2_minus_19A": float(t2[1] - t1[1]),
            "nuisance_prior_norm": float(np.linalg.norm(result["nuisance_z"])),
            "max_abs_nuisance_sigma": float(np.max(np.abs(result["nuisance_z"])))
            if len(result["nuisance_z"])
            else 0.0,
        }
    )
    for name, value in zip(result["nuisance_names"], result["nuisance_z"]):
        row[f"nuisance_{name}_sigma"] = float(value)
    return row


def hierarchical_comparison(baseline: dict) -> pd.DataFrame:
    wells = baseline["wells"]
    scale = baseline["scale"]
    points = {name: fit.theta for name, fit in baseline["weighted"].items()}
    reference = baseline["weighted_pooled"].theta
    rows: list[dict] = []

    for adjusted in (False, True):
        suffix = "adjusted" if adjusted else "raw"
        coordinate = baseline["coordinate_weighted"]
        A = float(coordinate[f"A_{suffix}"])
        Gamma = float(coordinate[f"Gamma_{suffix}"])
        for kind in ("separate", "shared_qstar", "shared_Cn", "pooled_theta"):
            result = fit_hierarchical_data_model(
                wells,
                scale,
                kind,
                points,
                A,
                Gamma,
                reference,
                adjusted=adjusted,
            )
            row = _hierarchy_flat_row(result)
            row["comparison_family"] = "observed_data"
            row["coordinate_A"] = A
            row["coordinate_Gamma"] = Gamma
            t1 = np.array([row["Vcem_19A"], row["lnCn_19A"]])
            t2 = np.array([row["Vcem_BT2"], row["lnCn_BT2"]])
            row["q_19A"] = float(math.log(coordinate_ratio(t1, reference, A, Gamma)))
            row["q_BT2"] = float(math.log(coordinate_ratio(t2, reference, A, Gamma)))
            row["delta_q_BT2_minus_19A"] = row["q_BT2"] - row["q_19A"]
            rows.append(row)

    raw_geom: dict[str, np.ndarray] = {}
    adjusted_geom: dict[str, np.ndarray] = {}
    for well, df in wells.items():
        _, _, _, G, Gadj, _ = e1.metric_summary(
            df, points[well], scale
        )
        raw_geom[well] = G
        adjusted_geom[well] = Gadj

    for adjustment, geometries, suffix in [
        ("local_raw_geometry", raw_geom, "raw"),
        ("local_Schur_geometry", adjusted_geom, "adjusted"),
    ]:
        A = float(baseline["coordinate_weighted"][f"A_{suffix}"])
        Gamma = float(baseline["coordinate_weighted"][f"Gamma_{suffix}"])
        for kind in ("separate", "shared_qstar", "shared_Cn", "pooled_theta"):
            result = fit_local_geometry_model(
                kind,
                points,
                geometries,
                A,
                Gamma,
                reference,
                adjustment,
            )
            row = _hierarchy_flat_row(result)
            row["comparison_family"] = "local_geometry"
            row["coordinate_A"] = A
            row["coordinate_Gamma"] = Gamma
            t1 = np.array([row["Vcem_19A"], row["lnCn_19A"]])
            t2 = np.array([row["Vcem_BT2"], row["lnCn_BT2"]])
            row["q_19A"] = float(math.log(coordinate_ratio(t1, reference, A, Gamma)))
            row["q_BT2"] = float(math.log(coordinate_ratio(t2, reference, A, Gamma)))
            row["delta_q_BT2_minus_19A"] = row["q_BT2"] - row["q_19A"]
            rows.append(row)

    table = pd.DataFrame(rows)
    table["delta_objective_from_separate"] = np.nan
    table["delta_Cn_minus_qstar"] = np.nan
    for (family, adjustment), group in table.groupby(
        ["comparison_family", "adjustment"], dropna=False
    ):
        index = group.index
        separate = float(group.loc[group.model == "separate", "objective_total"].iloc[0])
        q = float(group.loc[group.model == "shared_qstar", "objective_total"].iloc[0])
        cn = float(group.loc[group.model == "shared_Cn", "objective_total"].iloc[0])
        table.loc[index, "delta_objective_from_separate"] = (
            table.loc[index, "objective_total"] - separate
        )
        table.loc[index, "delta_Cn_minus_qstar"] = cn - q
    return table


def bootstrap_coordinate_and_hierarchy(
    baseline: dict,
    quick: bool,
    seed: int = 20260823,
) -> pd.DataFrame:
    schemes = [
        ("IID", 1, 24 if quick else 120, False),
        ("MBB_12m", 3, 24 if quick else 120, False),
        ("MBB_20m_primary", 5, 40 if quick else 400, True),
        ("MBB_40m", 10, 24 if quick else 120, False),
    ]
    wells = baseline["wells"]
    scale = baseline["scale"]
    pooled_start = baseline["rp_pooled"].theta
    weighted_starts = {
        name: fit.theta for name, fit in baseline["weighted"].items()
    }
    reference = baseline["weighted_pooled"].theta
    A_hier = float(baseline["coordinate_weighted"]["A_raw"])
    Gamma_hier = float(baseline["coordinate_weighted"]["Gamma_raw"])
    seed_sequence = np.random.SeedSequence(seed)
    scheme_seeds = seed_sequence.spawn(len(schemes))
    rows: list[dict] = []

    for (scheme, block_length, repetitions, do_hierarchy), child_seed in zip(
        schemes, scheme_seeds
    ):
        rng = np.random.default_rng(child_seed)
        started = time.time()
        for replicate in range(repetitions):
            sampled: dict[str, pd.DataFrame] = {}
            unique: dict[str, float] = {}
            for well, df in wells.items():
                sampled[well], unique[well] = noncircular_moving_block_sample(
                    df, block_length, rng
                )
            pooled = pd.concat([sampled["19A"], sampled["BT2"]], ignore_index=True)
            row: dict = {
                "scheme": scheme,
                "block_length_samples": block_length,
                "block_length_m": 4 * block_length,
                "replicate": replicate,
                "unique_fraction_19A": unique["19A"],
                "unique_fraction_BT2": unique["BT2"],
                "bootstrap_success": False,
                "hierarchy_evaluated": do_hierarchy,
            }
            try:
                pooled_fit = rp_grid_fit(pooled, pooled_start)
                coordinate = e1.metric_contact_hs_decomposition(
                    pooled, pooled_fit.theta, scale
                )
                v = float(pooled_fit.theta[0])
                row.update(
                    {
                        "bootstrap_success": bool(pooled_fit.success),
                        "pooled_fit_nfev": pooled_fit.nfev,
                        "pooled_fit_bound_hit": pooled_fit.bound_hit,
                        "Vcem_fraction": v,
                        "Cn": float(math.exp(pooled_fit.theta[1])),
                        "A_raw": float(coordinate["A_raw"]),
                        "Gamma_raw": float(coordinate["Gamma_raw"]),
                        "A_adjusted": float(coordinate["A_adjusted"]),
                        "Gamma_adjusted": float(coordinate["Gamma_adjusted"]),
                        "beta_raw": beta_from_factored_coordinate(
                            v, coordinate["A_raw"], coordinate["Gamma_raw"]
                        ),
                        "beta_adjusted": beta_from_factored_coordinate(
                            v,
                            coordinate["A_adjusted"],
                            coordinate["Gamma_adjusted"],
                        ),
                    }
                )
                for label, v_eval in [
                    ("BT2", baseline["rp_fits"]["BT2"].theta[0]),
                    ("pooled", baseline["rp_pooled"].theta[0]),
                    ("19A", baseline["rp_fits"]["19A"].theta[0]),
                ]:
                    row[f"beta_raw_at_{label}"] = beta_from_factored_coordinate(
                        float(v_eval), coordinate["A_raw"], coordinate["Gamma_raw"]
                    )
                    row[f"beta_adjusted_at_{label}"] = beta_from_factored_coordinate(
                        float(v_eval),
                        coordinate["A_adjusted"],
                        coordinate["Gamma_adjusted"],
                    )

                if do_hierarchy:
                    well_fits = {
                        name: weighted_fit(sampled[name], scale, weighted_starts[name])
                        for name in ("19A", "BT2")
                    }
                    separate_points = {
                        name: fit.theta for name, fit in well_fits.items()
                    }
                    separate_objective = float(
                        sum(fit.objective for fit in well_fits.values())
                    )
                    qfit = fit_hierarchical_data_model(
                        sampled,
                        scale,
                        "shared_qstar",
                        separate_points,
                        A_hier,
                        Gamma_hier,
                        reference,
                        adjusted=False,
                        max_nfev=1600,
                        multistart=False,
                    )
                    cnfit = fit_hierarchical_data_model(
                        sampled,
                        scale,
                        "shared_Cn",
                        separate_points,
                        A_hier,
                        Gamma_hier,
                        reference,
                        adjusted=False,
                        max_nfev=1600,
                        multistart=False,
                    )
                    row.update(
                        {
                            "hierarchy_separate_objective": separate_objective,
                            "hierarchy_qstar_objective": qfit["objective_total"],
                            "hierarchy_Cn_objective": cnfit["objective_total"],
                            "hierarchy_qstar_delta": qfit["objective_total"]
                            - separate_objective,
                            "hierarchy_Cn_delta": cnfit["objective_total"]
                            - separate_objective,
                            "hierarchy_delta_Cn_minus_qstar": cnfit["objective_total"]
                            - qfit["objective_total"],
                            "hierarchy_qstar_success": qfit["success"],
                            "hierarchy_Cn_success": cnfit["success"],
                            "hierarchy_qstar_bound_hit": qfit["bound_hit"],
                            "hierarchy_Cn_bound_hit": cnfit["bound_hit"],
                        }
                    )
            except Exception as exc:  # recorded, never silently discarded
                row["bootstrap_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if (replicate + 1) % max(10, repetitions // 8) == 0 or replicate + 1 == repetitions:
                elapsed = time.time() - started
                print(
                    f"bootstrap {scheme}: {replicate + 1}/{repetitions} "
                    f"({elapsed:.1f} s)",
                    flush=True,
                )
    return pd.DataFrame(rows)


def bootstrap_summary(
    replicates: pd.DataFrame, baseline: dict
) -> pd.DataFrame:
    metrics = [
        "A_raw",
        "Gamma_raw",
        "A_adjusted",
        "Gamma_adjusted",
        "beta_raw",
        "beta_adjusted",
        "beta_raw_at_BT2",
        "beta_raw_at_pooled",
        "beta_raw_at_19A",
        "beta_adjusted_at_BT2",
        "beta_adjusted_at_pooled",
        "beta_adjusted_at_19A",
        "Vcem_fraction",
        "Cn",
        "hierarchy_qstar_delta",
        "hierarchy_Cn_delta",
        "hierarchy_delta_Cn_minus_qstar",
    ]
    point = {
        "A_raw": baseline["coordinate_rpia"]["A_raw"],
        "Gamma_raw": baseline["coordinate_rpia"]["Gamma_raw"],
        "A_adjusted": baseline["coordinate_rpia"]["A_adjusted"],
        "Gamma_adjusted": baseline["coordinate_rpia"]["Gamma_adjusted"],
        "beta_raw": baseline["coordinate_rpia"]["beta_total_raw_projection"],
        "beta_adjusted": baseline["coordinate_rpia"]["beta_total_adjusted_projection"],
        "Vcem_fraction": baseline["rp_pooled"].theta[0],
        "Cn": math.exp(baseline["rp_pooled"].theta[1]),
    }
    for suffix in ("raw", "adjusted"):
        A = baseline["coordinate_rpia"][f"A_{suffix}"]
        Gamma = baseline["coordinate_rpia"][f"Gamma_{suffix}"]
        for label, v in [
            ("BT2", baseline["rp_fits"]["BT2"].theta[0]),
            ("pooled", baseline["rp_pooled"].theta[0]),
            ("19A", baseline["rp_fits"]["19A"].theta[0]),
        ]:
            point[f"beta_{suffix}_at_{label}"] = beta_from_factored_coordinate(
                float(v), float(A), float(Gamma)
            )

    rows = []
    for scheme, group in replicates.groupby("scheme", sort=False):
        successful = group[group.bootstrap_success.fillna(False)]
        subsets = [
            ("all_success", successful),
            ("interior_only", successful[~successful.pooled_fit_bound_hit.fillna(False)]),
        ]
        for sample_subset, valid in subsets:
            for metric in metrics:
                values = pd.to_numeric(valid.get(metric), errors="coerce").dropna().to_numpy()
                if not len(values):
                    continue
                med = float(np.median(values))
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                mad = float(np.median(np.abs(values - med)))
                rows.append(
                    {
                        "scheme": scheme,
                        "sample_subset": sample_subset,
                        "metric": metric,
                        "n_valid": len(values),
                        "point_estimate": point.get(metric, np.nan),
                        "mean": mean,
                        "median": med,
                        "mad": mad,
                        "std": std,
                        "cv_abs": float(std / abs(mean)) if mean != 0 and np.isfinite(std) else np.nan,
                        "q02p5": float(np.quantile(values, 0.025)),
                        "q16": float(np.quantile(values, 0.16)),
                        "q84": float(np.quantile(values, 0.84)),
                        "q97p5": float(np.quantile(values, 0.975)),
                        "median_bias": float(med - point[metric]) if metric in point else np.nan,
                        "probability_positive": float(np.mean(values > 0)),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_diagnostics(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, group in replicates.groupby("scheme", sort=False):
        valid = group[group.bootstrap_success.fillna(False)]
        row = {
            "scheme": scheme,
            "n_requested": len(group),
            "n_success": len(valid),
            "failure_rate": float(1 - len(valid) / len(group)),
            "pooled_bound_hit_rate": float(valid.pooled_fit_bound_hit.mean()) if len(valid) else np.nan,
            "median_unique_fraction_19A": float(valid.unique_fraction_19A.median()) if len(valid) else np.nan,
            "median_unique_fraction_BT2": float(valid.unique_fraction_BT2.median()) if len(valid) else np.nan,
            "corr_A_Gamma_raw": float(valid[["A_raw", "Gamma_raw"]].corr().iloc[0, 1]) if len(valid) > 2 else np.nan,
            "corr_A_Gamma_adjusted": float(valid[["A_adjusted", "Gamma_adjusted"]].corr().iloc[0, 1]) if len(valid) > 2 else np.nan,
        }
        hierarchy = valid[valid.hierarchy_evaluated.fillna(False)]
        if len(hierarchy):
            row.update(
                {
                    "hierarchy_n": len(hierarchy),
                    "qstar_preferred_fraction": float(
                        np.mean(hierarchy.hierarchy_delta_Cn_minus_qstar > 0)
                    ),
                    "qstar_bound_hit_rate": float(hierarchy.hierarchy_qstar_bound_hit.mean()),
                    "Cn_bound_hit_rate": float(hierarchy.hierarchy_Cn_bound_hit.mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def loto_level_table(baseline: dict) -> pd.DataFrame:
    rows = []
    operating_sets = {
        "RPIA": {
            well: fit.theta for well, fit in baseline["rp_fits"].items()
        },
        "transfer_weighted": {
            well: fit.theta for well, fit in baseline["weighted"].items()
        },
    }
    for operating_definition, points in operating_sets.items():
        for train, test in (("19A", "BT2"), ("BT2", "19A")):
            local = e1.metric_contact_hs_decomposition(
                baseline["wells"][train], points[train], baseline["scale"]
            )
            for suffix in ("raw", "adjusted"):
                A = float(local[f"A_{suffix}"])
                Gamma = float(local[f"Gamma_{suffix}"])
                beta = float(local[f"beta_total_{suffix}_projection"])
                q_ratio = coordinate_ratio(points[test], points[train], A, Gamma)
                dv = float(points[test][0] - points[train][0])
                local_ratio = float(
                    math.exp(points[test][1] - points[train][1] + beta * dv)
                )
                predicted_ell = (
                    points[train][1]
                    - A * math.log(points[test][0] / points[train][0])
                    - Gamma
                    * math.log(
                        (PHIC - points[test][0]) / (PHIC - points[train][0])
                    )
                )
                rows.append(
                    {
                        "operating_definition": operating_definition,
                        "train": train,
                        "test": test,
                        "adjustment": suffix,
                        "Vcem_train": float(points[train][0]),
                        "Cn_train": float(math.exp(points[train][1])),
                        "Vcem_test": float(points[test][0]),
                        "Cn_test": float(math.exp(points[test][1])),
                        "A_train": A,
                        "Gamma_train": Gamma,
                        "beta_local_train": beta,
                        "delta_q_train_to_test": float(math.log(q_ratio)),
                        "qstar_level_ratio": q_ratio,
                        "qstar_level_percent_error": float(100 * (q_ratio - 1)),
                        "Cn_predicted_from_train_qstar": float(math.exp(predicted_ell)),
                        "Cn_prediction_percent_error": float(
                            100 * (math.exp(predicted_ell - points[test][1]) - 1)
                        ),
                        "local_exponential_level_ratio": local_ratio,
                        "local_exponential_percent_error": float(100 * (local_ratio - 1)),
                    }
                )
    return pd.DataFrame(rows)


def loto_train_bootstrap(
    baseline: dict,
    quick: bool,
    seed: int = 20260824,
) -> pd.DataFrame:
    repetitions = 40 if quick else 300
    rng = np.random.default_rng(seed)
    points = {well: fit.theta for well, fit in baseline["rp_fits"].items()}
    rows = []
    for train, test in (("19A", "BT2"), ("BT2", "19A")):
        started = time.time()
        for replicate in range(repetitions):
            sampled, unique = noncircular_moving_block_sample(
                baseline["wells"][train], 5, rng
            )
            row = {
                "train": train,
                "test": test,
                "replicate": replicate,
                "block_length_samples": 5,
                "block_length_m": 20,
                "unique_fraction_train": unique,
                "success": False,
            }
            try:
                fit = rp_grid_fit(sampled, points[train])
                coordinate = e1.metric_contact_hs_decomposition(
                    sampled, fit.theta, baseline["scale"]
                )
                row.update(
                    {
                        "success": bool(fit.success),
                        "bound_hit": fit.bound_hit,
                        "Vcem_train_bootstrap": float(fit.theta[0]),
                        "Cn_train_bootstrap": float(math.exp(fit.theta[1])),
                    }
                )
                for suffix in ("raw", "adjusted"):
                    A = float(coordinate[f"A_{suffix}"])
                    Gamma = float(coordinate[f"Gamma_{suffix}"])
                    beta = float(coordinate[f"beta_total_{suffix}_projection"])
                    q_ratio = coordinate_ratio(points[test], fit.theta, A, Gamma)
                    local_ratio = math.exp(
                        points[test][1]
                        - fit.theta[1]
                        + beta * (points[test][0] - fit.theta[0])
                    )
                    row.update(
                        {
                            f"A_{suffix}": A,
                            f"Gamma_{suffix}": Gamma,
                            f"qstar_ratio_{suffix}": float(q_ratio),
                            f"local_exponential_ratio_{suffix}": float(local_ratio),
                        }
                    )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if (replicate + 1) % max(10, repetitions // 6) == 0 or replicate + 1 == repetitions:
                print(
                    f"LOTO bootstrap {train}->{test}: {replicate + 1}/{repetitions} "
                    f"({time.time() - started:.1f} s)",
                    flush=True,
                )
    return pd.DataFrame(rows)


def loto_bootstrap_summary(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (train, test), group in replicates.groupby(["train", "test"], sort=False):
        valid = group[group.success.fillna(False)]
        for metric in [
            "qstar_ratio_raw",
            "qstar_ratio_adjusted",
            "local_exponential_ratio_raw",
            "local_exponential_ratio_adjusted",
        ]:
            values = valid[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "train": train,
                    "test": test,
                    "metric": metric,
                    "n_valid": len(values),
                    "median": float(np.median(values)),
                    "q02p5": float(np.quantile(values, 0.025)),
                    "q16": float(np.quantile(values, 0.16)),
                    "q84": float(np.quantile(values, 0.84)),
                    "q97p5": float(np.quantile(values, 0.975)),
                    "probability_within_5_percent": float(
                        np.mean(np.abs(values - 1.0) <= 0.05)
                    ),
                    "probability_within_10_percent": float(
                        np.mean(np.abs(values - 1.0) <= 0.10)
                    ),
                }
            )
    return pd.DataFrame(rows)


def structural_profile(
    df: pd.DataFrame,
    theta0: np.ndarray,
    scale: dict[str, float],
    quick: bool,
) -> pd.DataFrame:
    npoints = 17 if quick else 35
    v_eval = np.unique(np.r_[np.linspace(0.001, 0.060, npoints), theta0[0]])
    v_eval.sort()
    prediction0 = rc.stack(df, MODEL, theta0, NAMES)
    sigma = e1.stacked_sigma(df, scale)
    nuisance_names = rc.nuisance_names(MODEL)
    nuisance_scales = np.array([rc.NUI_SCALES[n] for n in nuisance_names])

    def raw_objective(v: float, ell: float) -> float:
        residual = (rc.stack(df, MODEL, np.array([v, ell]), NAMES) - prediction0) / sigma
        return float(residual @ residual)

    def adjusted_residual(v: float, x: np.ndarray) -> np.ndarray:
        ell = float(x[0])
        z = x[1:]
        nuisance = {
            name: float(z[i] * nuisance_scales[i])
            for i, name in enumerate(nuisance_names)
        }
        prediction = rc.stack(df, MODEL, np.array([v, ell]), NAMES, nuisance)
        if not np.all(np.isfinite(prediction)):
            return np.full(len(prediction0) + len(z), 1e6)
        return np.r_[(prediction - prediction0) / sigma, z]

    raw = {}
    for v in v_eval:
        fit = minimize_scalar(
            lambda ell: raw_objective(float(v), float(ell)),
            bounds=(math.log(3.0), math.log(18.0)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        raw[float(v)] = (float(fit.x), float(fit.fun))

    adjusted = {
        float(theta0[0]): (
            np.r_[theta0[1], np.zeros(len(nuisance_names))],
            0.0,
            True,
        )
    }
    center = np.r_[theta0[1], np.zeros(len(nuisance_names))]
    lower = np.r_[math.log(3.0), np.full(len(nuisance_names), -4.0)]
    upper = np.r_[math.log(18.0), np.full(len(nuisance_names), 4.0)]
    directions = [
        sorted([float(v) for v in v_eval if v < theta0[0]], reverse=True),
        sorted([float(v) for v in v_eval if v > theta0[0]]),
    ]
    for direction in directions:
        x_start = center.copy()
        for v in direction:
            fit = least_squares(
                lambda x: adjusted_residual(v, x),
                x_start,
                bounds=(lower, upper),
                max_nfev=450,
                xtol=2e-9,
                ftol=2e-9,
                gtol=2e-9,
            )
            r = adjusted_residual(v, fit.x)
            adjusted[v] = (fit.x.copy(), float(r @ r), bool(fit.success))
            x_start = fit.x.copy()

    rows = []
    for v in v_eval:
        x, objective, success = adjusted[float(v)]
        rows.append(
            {
                "Vcem_fraction": float(v),
                "lnCn_raw_profile": raw[float(v)][0],
                "objective_raw_profile": raw[float(v)][1],
                "lnCn_adjusted_profile": float(x[0]),
                "objective_adjusted_profile": objective,
                "adjusted_success": success,
                "nuisance_prior_norm": float(np.linalg.norm(x[1:])),
            }
        )
    return pd.DataFrame(rows)


def loto_profile_tables(baseline: dict, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = {well: fit.theta for well, fit in baseline["rp_fits"].items()}
    local_coordinates = {
        well: e1.metric_contact_hs_decomposition(
            baseline["wells"][well], points[well], baseline["scale"]
        )
        for well in ("19A", "BT2")
    }
    pooled = baseline["coordinate_rpia"]
    rows = []
    summaries = []
    v_low = min(points["19A"][0], points["BT2"][0])
    v_high = max(points["19A"][0], points["BT2"][0])

    for train, test in (("19A", "BT2"), ("BT2", "19A")):
        profile = structural_profile(
            baseline["wells"][test], points[test], baseline["scale"], quick
        )
        v0, ell0 = map(float, points[test])
        for suffix in ("raw", "adjusted"):
            target_field = f"lnCn_{suffix}_profile"
            A_train = float(local_coordinates[train][f"A_{suffix}"])
            Gamma_train = float(local_coordinates[train][f"Gamma_{suffix}"])
            beta_train = float(
                local_coordinates[train][f"beta_total_{suffix}_projection"]
            )
            A_test = float(local_coordinates[test][f"A_{suffix}"])
            Gamma_test = float(local_coordinates[test][f"Gamma_{suffix}"])
            A_pool = float(pooled[f"A_{suffix}"])
            Gamma_pool = float(pooled[f"Gamma_{suffix}"])
            v = profile.Vcem_fraction.to_numpy(dtype=float)
            curves = {
                "train_qstar": ell0
                - A_train * np.log(v / v0)
                - Gamma_train * np.log((PHIC - v) / (PHIC - v0)),
                "test_oracle_qstar": ell0
                - A_test * np.log(v / v0)
                - Gamma_test * np.log((PHIC - v) / (PHIC - v0)),
                "pooled_qstar": ell0
                - A_pool * np.log(v / v0)
                - Gamma_pool * np.log((PHIC - v) / (PHIC - v0)),
                "train_local_exponential": ell0 - beta_train * (v - v0),
                "constant_Cn": np.full_like(v, ell0),
            }
            observed = profile[target_field].to_numpy(dtype=float)
            for i, vi in enumerate(v):
                base = {
                    "train": train,
                    "test": test,
                    "adjustment": suffix,
                    "Vcem_fraction": float(vi),
                    "Vcem_percent": float(100 * vi),
                    "lnCn_profile": float(observed[i]),
                    "Cn_profile": float(math.exp(observed[i])),
                    "profile_objective": float(
                        profile[f"objective_{suffix}_profile"].iloc[i]
                    ),
                }
                for method, values in curves.items():
                    base[f"lnCn_{method}"] = float(values[i])
                    base[f"error_{method}"] = float(values[i] - observed[i])
                rows.append(base)

            masks = {
                "local_0p015": np.abs(v - v0) <= 0.015 + 1e-12,
                "between_operating_points": (v >= v_low) & (v <= v_high),
                "full_admissible_Vcem": np.ones(len(v), dtype=bool),
            }
            for window, mask in masks.items():
                for method, values in curves.items():
                    error = values[mask] - observed[mask]
                    outside = (values[mask] < math.log(3.0)) | (
                        values[mask] > math.log(18.0)
                    )
                    summaries.append(
                        {
                            "train": train,
                            "test": test,
                            "adjustment": suffix,
                            "window": window,
                            "method": method,
                            "n_points": int(mask.sum()),
                            "rms_lnCn_error": float(np.sqrt(np.mean(error**2))),
                            "max_abs_lnCn_error": float(np.max(np.abs(error))),
                            "median_abs_percent_Cn_error": float(
                                100 * np.median(np.abs(np.exp(error) - 1))
                            ),
                            "fraction_outside_Cn_bounds": float(np.mean(outside)),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "figure.titlesize": 13,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(
            ROOT / "results" / "figures" / f"{stem}.{suffix}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_bootstrap_stability(replicates: pd.DataFrame, baseline: dict) -> None:
    configure_plotting()
    valid = replicates[replicates.bootstrap_success.fillna(False)].copy()
    order = ["IID", "MBB_12m", "MBB_20m_primary", "MBB_40m"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["navy"], COLORS["orange"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)

    for ax, suffix in zip(axes[0], ("raw", "adjusted")):
        for scheme, color in zip(order, colors):
            part = valid[valid.scheme == scheme]
            ax.scatter(
                part[f"A_{suffix}"],
                part[f"Gamma_{suffix}"],
                s=11,
                alpha=0.24,
                color=color,
                linewidths=0,
                label=scheme.replace("_primary", ""),
            )
        ax.scatter(
            [baseline["coordinate_rpia"][f"A_{suffix}"]],
            [baseline["coordinate_rpia"][f"Gamma_{suffix}"]],
            marker="*",
            s=120,
            color=COLORS["gold"],
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="E1 point" if suffix == "raw" else None,
        )
        ax.set_xlabel(fr"$A_{{\rm {suffix}}}$")
        ax.set_ylabel(fr"$\Gamma_{{\rm {suffix}}}$")
        ax.set_title(f"Joint stability, {suffix}")
    axes[0, 0].legend(frameon=False, ncol=2)

    metrics = ["A_raw", "Gamma_raw", "A_adjusted", "Gamma_adjusted"]
    x = np.arange(len(metrics))
    width = 0.18
    ax = axes[1, 0]
    point_values = np.array(
        [
            baseline["coordinate_rpia"]["A_raw"],
            baseline["coordinate_rpia"]["Gamma_raw"],
            baseline["coordinate_rpia"]["A_adjusted"],
            baseline["coordinate_rpia"]["Gamma_adjusted"],
        ],
        dtype=float,
    )
    for j, (scheme, color) in enumerate(zip(order, colors)):
        part = valid[valid.scheme == scheme]
        medians = np.array([part[m].median() for m in metrics]) / point_values
        low = np.array([part[m].quantile(0.025) for m in metrics]) / point_values
        high = np.array([part[m].quantile(0.975) for m in metrics]) / point_values
        pos = x + (j - 1.5) * width
        ax.errorbar(
            pos,
            medians,
            yerr=[np.array(medians) - np.array(low), np.array(high) - np.array(medians)],
            fmt="o",
            color=color,
            capsize=2.5,
            ms=4.5,
            label=scheme.replace("_primary", ""),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([r"$A_r$", r"$\Gamma_r$", r"$A_a$", r"$\Gamma_a$"])
    ax.axhline(1.0, color="0.72", lw=0.9)
    ax.set_ylabel("bootstrap coefficient / E1 coefficient")
    ax.set_title("Block-length sensitivity")

    ax = axes[1, 1]
    primary = valid[valid.scheme == "MBB_20m_primary"]
    for field, label, color, ls in [
        ("beta_raw_at_BT2", "raw at BT2 point", COLORS["teal"], "-"),
        ("beta_raw_at_pooled", "raw at pooled point", COLORS["navy"], "-"),
        ("beta_raw_at_19A", "raw at 19A point", COLORS["orange"], "-"),
        ("beta_adjusted_at_pooled", "adjusted at pooled point", COLORS["red"], "--"),
    ]:
        values = primary[field].dropna().to_numpy()
        ax.hist(values, bins=28, density=True, histtype="step", lw=1.8, color=color, ls=ls, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"implied $\beta_\star(v)=A/v-\Gamma/(\phi_c-v)$")
    ax.set_ylabel("density")
    ax.set_title("Stable factors do not imply a universal local slope")
    ax.legend(frameon=False)

    fig.suptitle(
        r"Bootstrap stability of the factored contact-state coordinate"
        + "\npaired within-trajectory resampling; 20 m moving blocks are primary"
    )
    _save_figure(fig, "Fig_E2_bootstrap_stability")


def plot_loto(level: pd.DataFrame, profiles: pd.DataFrame, summary: pd.DataFrame) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), constrained_layout=True)
    for column, adjustment in enumerate(("raw", "adjusted")):
        ax = axes[0, column]
        part = level[
            (level.operating_definition == "RPIA") & (level.adjustment == adjustment)
        ]
        labels = [f"{r.train}→{r.test}" for r in part.itertuples()]
        xpos = np.arange(len(part))
        ax.scatter(xpos - 0.08, part.qstar_level_ratio, s=55, color=COLORS["navy"], label=r"factored $q_\star$")
        ax.scatter(xpos + 0.08, part.local_exponential_level_ratio, s=55, marker="x", color=COLORS["red"], label="train-local exponential")
        ax.axhline(1.0, color="0.7", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("transported state ratio (log scale)")
        ax.set_title(f"Level transport, {adjustment}")
        if column == 0:
            ax.legend(frameon=False)

    for column, test in enumerate(("19A", "BT2")):
        ax = axes[1, column]
        part = profiles[(profiles.test == test) & (profiles.adjustment == "adjusted")]
        if part.empty:
            continue
        train = part.train.iloc[0]
        ax.plot(part.Vcem_percent, part.lnCn_profile, color="black", lw=2.2, label="target numerical ridge")
        ax.plot(part.Vcem_percent, part.lnCn_train_qstar, color=COLORS["navy"], lw=2.0, ls="--", label=f"{train}-trained q★")
        ax.plot(part.Vcem_percent, part.lnCn_pooled_qstar, color=COLORS["teal"], lw=1.5, ls="-.", label="pooled q★")
        ax.plot(part.Vcem_percent, part.lnCn_train_local_exponential, color=COLORS["red"], lw=1.4, ls=":", label="train-local exponential")
        ax.plot(part.Vcem_percent, part.lnCn_constant_Cn, color=COLORS["gray"], lw=1.1, label=r"constant $C_n$")
        ax.set_ylim(math.log(2.6), math.log(22.0))
        ax.set_xlabel(r"cement volume $V_{\rm cem}$ (%)")
        ax.set_ylabel(r"$\ln C_n$")
        ax.set_title(f"Shape transfer to {test}, nuisance-adjusted")
        if np.nanmax(part.lnCn_train_local_exponential) > math.log(22.0):
            ax.text(
                0.03,
                0.96,
                "local tangent continues off-scale",
                transform=ax.transAxes,
                color=COLORS["red"],
                va="top",
                fontsize=7.8,
            )
        if column == 0:
            ax.legend(frameon=False, fontsize=7.6)
    fig.suptitle(
        "Leave-one-trajectory-out transport\n"
        "top: unrecentered level test; bottom: recentered shape test"
    )
    _save_figure(fig, "Fig_E2_loto_transport")


def plot_hierarchy(hierarchy: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    models = ["shared_qstar", "shared_Cn", "pooled_theta"]
    labels = [r"shared $q_\star$", r"shared $C_n$", r"pooled $\theta$"]
    x = np.arange(len(models))

    ax = axes[0]
    for j, adjustment in enumerate(("fixed_nuisance", "nonlinear_MAP")):
        part = hierarchy[
            (hierarchy.comparison_family == "observed_data")
            & (hierarchy.adjustment == adjustment)
        ].set_index("model")
        values = [part.loc[m, "delta_objective_from_separate"] for m in models]
        ax.bar(
            x + (j - 0.5) * 0.32,
            values,
            width=0.32,
            color=COLORS["navy"] if j == 0 else COLORS["red"],
            label="fixed nuisances" if j == 0 else "joint nuisance MAP",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"constraint cost $\Delta\Phi$")
    ax.set_title("Observed-data hierarchy")
    ax.legend(frameon=False)

    ax = axes[1]
    for j, adjustment in enumerate(("local_raw_geometry", "local_Schur_geometry")):
        part = hierarchy[
            (hierarchy.comparison_family == "local_geometry")
            & (hierarchy.adjustment == adjustment)
        ].set_index("model")
        values = [part.loc[m, "delta_objective_from_separate"] for m in models]
        ax.bar(
            x + (j - 0.5) * 0.32,
            values,
            width=0.32,
            color=COLORS["blue"] if j == 0 else COLORS["orange"],
            label="raw geometry" if j == 0 else "Schur-adjusted geometry",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("local quadratic constraint cost")
    ax.set_title("Geometry after nuisance adjustment")
    ax.legend(frameon=False)

    ax = axes[2]
    primary = bootstrap[
        (bootstrap.scheme == "MBB_20m_primary")
        & bootstrap.bootstrap_success.fillna(False)
        & bootstrap.hierarchy_evaluated.fillna(False)
    ]
    delta = primary.hierarchy_delta_Cn_minus_qstar.dropna().to_numpy()
    ax.hist(delta, bins=30, color=COLORS["teal"], alpha=0.78, edgecolor="white")
    ax.axvline(0.0, color="black", lw=1.0)
    if len(delta):
        ax.axvline(np.median(delta), color=COLORS["red"], lw=1.6, ls="--", label=f"median = {np.median(delta):.3g}")
    ax.set_xlabel(r"$\Phi_{\rm shared\ C_n}-\Phi_{\rm shared\ q_\star}$")
    ax.set_ylabel("20 m block-bootstrap count")
    ax.set_title("Can static data choose the hierarchy?")
    ax.legend(frameon=False)
    fig.suptitle(
        "Hierarchical comparison of shared state assumptions\n"
        "positive difference favors shared q★; near zero means the ridge absorbs both constraints"
    )
    _save_figure(fig, "Fig_E2_hierarchical_comparison")


def summarize(
    baseline: dict,
    bootstrap: pd.DataFrame,
    bootstrap_summary_table: pd.DataFrame,
    bootstrap_diag: pd.DataFrame,
    loto_level: pd.DataFrame,
    loto_bs_summary: pd.DataFrame,
    loto_shape_summary: pd.DataFrame,
    hierarchy: pd.DataFrame,
) -> dict:
    primary_summary = bootstrap_summary_table[
        (bootstrap_summary_table.scheme == "MBB_20m_primary")
        & (bootstrap_summary_table.sample_subset == "all_success")
    ].set_index("metric")
    level_rpia = loto_level[loto_level.operating_definition == "RPIA"]
    hierarchy_rows = hierarchy.set_index(
        ["comparison_family", "adjustment", "model"]
    )
    primary_bootstrap = bootstrap[
        (bootstrap.scheme == "MBB_20m_primary")
        & bootstrap.bootstrap_success.fillna(False)
        & bootstrap.hierarchy_evaluated.fillna(False)
    ]

    coordinate_stability = {}
    for metric in ("A_raw", "Gamma_raw", "A_adjusted", "Gamma_adjusted"):
        row = primary_summary.loc[metric]
        coordinate_stability[metric] = {
            "point": float(row.point_estimate),
            "median": float(row["median"]),
            "ci95": [float(row.q02p5), float(row.q97p5)],
            "cv_abs": float(row.cv_abs),
        }

    level = {}
    for row in level_rpia.itertuples():
        key = f"{row.train}_to_{row.test}_{row.adjustment}"
        level[key] = {
            "qstar_ratio": float(row.qstar_level_ratio),
            "local_exponential_ratio": float(row.local_exponential_level_ratio),
        }

    shape = {}
    selected_shape = loto_shape_summary[
        (loto_shape_summary.window == "local_0p015")
        & loto_shape_summary.method.isin(
            ["train_qstar", "train_local_exponential", "constant_Cn"]
        )
    ]
    for row in selected_shape.itertuples():
        key = f"{row.train}_to_{row.test}_{row.adjustment}_{row.method}"
        shape[key] = {
            "rms_lnCn_error": float(row.rms_lnCn_error),
            "median_abs_percent_Cn_error": float(row.median_abs_percent_Cn_error),
        }

    hierarchy_out = {}
    for family, adjustment in [
        ("observed_data", "fixed_nuisance"),
        ("observed_data", "nonlinear_MAP"),
        ("local_geometry", "local_raw_geometry"),
        ("local_geometry", "local_Schur_geometry"),
    ]:
        q = hierarchy_rows.loc[(family, adjustment, "shared_qstar")]
        cn = hierarchy_rows.loc[(family, adjustment, "shared_Cn")]
        hierarchy_out[f"{family}_{adjustment}"] = {
            "qstar_constraint_cost": float(q.delta_objective_from_separate),
            "Cn_constraint_cost": float(cn.delta_objective_from_separate),
            "delta_Cn_minus_qstar": float(cn.delta_Cn_minus_qstar),
        }

    delta = primary_bootstrap.hierarchy_delta_Cn_minus_qstar.dropna().to_numpy()
    primary_diag = bootstrap_diag.set_index("scheme").loc["MBB_20m_primary"]
    return {
        "experiment": "E2 stability, LOTO transport, and hierarchical comparison",
        "model": "constant_cement Scheme 1, frozen RPIA v1 forward model",
        "bootstrap_interpretation": "conditional within-trajectory uncertainty for two Hugin trajectories; not between-trajectory population uncertainty",
        "bootstrap_primary": {
            "scheme": "non-circular moving-block bootstrap, stratified by trajectory",
            "block_length_samples": 5,
            "block_length_m": 20,
            "n_requested": int(primary_diag.n_requested),
            "n_success": int(primary_diag.n_success),
            "bound_hit_rate": float(primary_diag.pooled_bound_hit_rate),
            "corr_A_Gamma_raw": float(primary_diag.corr_A_Gamma_raw),
            "corr_A_Gamma_adjusted": float(primary_diag.corr_A_Gamma_adjusted),
        },
        "coordinate_stability": coordinate_stability,
        "loto_level_transport_RPIA": level,
        "loto_shape_local_window": shape,
        "hierarchical_comparison": hierarchy_out,
        "hierarchical_bootstrap": {
            "n": int(len(delta)),
            "median_delta_Cn_minus_qstar": float(np.median(delta)),
            "ci95_delta_Cn_minus_qstar": [
                float(np.quantile(delta, 0.025)),
                float(np.quantile(delta, 0.975)),
            ],
            "qstar_preferred_fraction": float(np.mean(delta > 0)),
        },
        "loto_bootstrap_summary": loto_bs_summary.to_dict(orient="records"),
        "frozen_scale": baseline["scale"],
        "operating_points": {
            "19A": {
                "Vcem": float(baseline["rp_fits"]["19A"].theta[0]),
                "Cn": float(math.exp(baseline["rp_fits"]["19A"].theta[1])),
            },
            "BT2": {
                "Vcem": float(baseline["rp_fits"]["BT2"].theta[0]),
                "Cn": float(math.exp(baseline["rp_fits"]["BT2"].theta[1])),
            },
            "pooled": {
                "Vcem": float(baseline["rp_pooled"].theta[0]),
                "Cn": float(math.exp(baseline["rp_pooled"].theta[1])),
            },
        },
    }


def write_results_markdown(summary: dict) -> None:
    stability = summary["coordinate_stability"]
    level = summary["loto_level_transport_RPIA"]
    hierarchy = summary["hierarchical_comparison"]
    boot_h = summary["hierarchical_bootstrap"]
    text = f"""# E2 results: stability, LOTO transport, and hierarchy

## Main result

The factorized coordinate is stable under paired within-trajectory resampling and transports much better between the two Hugin trajectories than a local exponential tangent. This does **not** make the nominal parameters separately identifiable: the hierarchical comparison shows that static elastic data can absorb both a shared-`q_star` and a shared-`C_n` restriction along the same weak ridge.

## Bootstrap stability

The primary analysis used {summary['bootstrap_primary']['n_requested']} non-circular moving-block replicates, stratified by trajectory, with a 5-sample (20 m) block. Successful replicates: {summary['bootstrap_primary']['n_success']}.

- `A_raw`: median {stability['A_raw']['median']:.4f}, 95% interval [{stability['A_raw']['ci95'][0]:.4f}, {stability['A_raw']['ci95'][1]:.4f}].
- `Gamma_raw`: median {stability['Gamma_raw']['median']:.4f}, 95% interval [{stability['Gamma_raw']['ci95'][0]:.4f}, {stability['Gamma_raw']['ci95'][1]:.4f}].
- `A_adjusted`: median {stability['A_adjusted']['median']:.4f}, 95% interval [{stability['A_adjusted']['ci95'][0]:.4f}, {stability['A_adjusted']['ci95'][1]:.4f}].
- `Gamma_adjusted`: median {stability['Gamma_adjusted']['median']:.4f}, 95% interval [{stability['Gamma_adjusted']['ci95'][0]:.4f}, {stability['Gamma_adjusted']['ci95'][1]:.4f}].

These intervals are conditional on the two available Hugin trajectories. With only two trajectory clusters, they are not estimates of between-trajectory population uncertainty.

## Leave-one-trajectory-out level transport

- 19A to BT2, raw: factored ratio {level['19A_to_BT2_raw']['qstar_ratio']:.4f}; train-local exponential {level['19A_to_BT2_raw']['local_exponential_ratio']:.4f}.
- 19A to BT2, adjusted: factored ratio {level['19A_to_BT2_adjusted']['qstar_ratio']:.4f}; train-local exponential {level['19A_to_BT2_adjusted']['local_exponential_ratio']:.4f}.
- BT2 to 19A, raw: factored ratio {level['BT2_to_19A_raw']['qstar_ratio']:.4f}; train-local exponential {level['BT2_to_19A_raw']['local_exponential_ratio']:.4f}.
- BT2 to 19A, adjusted: factored ratio {level['BT2_to_19A_adjusted']['qstar_ratio']:.4f}; train-local exponential {level['BT2_to_19A_adjusted']['local_exponential_ratio']:.4f}.

This is an internal transport test under the frozen E1 metric, not external validation: the transfer-aware scale was estimated from both trajectories.

## Hierarchical comparison

For equal-dimensional shared-state models, positive `delta_Cn_minus_qstar` favors shared `q_star`.

- Fixed-nuisance observed-data difference: {hierarchy['observed_data_fixed_nuisance']['delta_Cn_minus_qstar']:.4g}.
- Joint nuisance-MAP difference: {hierarchy['observed_data_nonlinear_MAP']['delta_Cn_minus_qstar']:.4g}.
- Local Schur-adjusted difference: {hierarchy['local_geometry_local_Schur_geometry']['delta_Cn_minus_qstar']:.4g}.
- In the 20 m block bootstrap, shared `q_star` had the lower raw objective in {100*boot_h['qstar_preferred_fraction']:.1f}% of replicates; the 95% interval for the objective difference was [{boot_h['ci95_delta_Cn_minus_qstar'][0]:.4g}, {boot_h['ci95_delta_Cn_minus_qstar'][1]:.4g}].

The scientifically conservative interpretation is that the data support the **transportability of the factorized ridge coordinate**, but do not yet discriminate decisively between geological sharing assumptions. Sharing `C_n` can still slide along the local ridge at negligible cost. A hierarchy is therefore not, by itself, a source of new sensitivity diversity.
"""
    (ROOT / "results" / "RESULTS.md").write_text(text)


def write_manifest() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.csv", "SHA256SUMS.txt"}:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "MANIFEST.csv", index=False)
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(f"{row['sha256']}  {row['relative_path']}" for row in rows) + "\n"
    )


def run(quick: bool = False) -> dict:
    _mkdirs()
    baseline = load_baseline()
    print("baseline loaded; frozen transfer-aware metric reconstructed", flush=True)
    acf = residual_acf_table(baseline["wells"], baseline["rp_fits"])
    acf.to_csv(ROOT / "results" / "tables" / "E2_residual_acf.csv", index=False)

    bootstrap = bootstrap_coordinate_and_hierarchy(baseline, quick)
    bootstrap.to_csv(
        ROOT / "results" / "tables" / "E2_bootstrap_replicates.csv", index=False
    )
    boot_summary = bootstrap_summary(bootstrap, baseline)
    boot_summary.to_csv(
        ROOT / "results" / "tables" / "E2_bootstrap_summary.csv", index=False
    )
    boot_diag = bootstrap_diagnostics(bootstrap)
    boot_diag.to_csv(
        ROOT / "results" / "tables" / "E2_bootstrap_diagnostics.csv", index=False
    )

    loto_level = loto_level_table(baseline)
    loto_level.to_csv(
        ROOT / "results" / "tables" / "E2_loto_level.csv", index=False
    )
    loto_bootstrap = loto_train_bootstrap(baseline, quick)
    loto_bootstrap.to_csv(
        ROOT / "results" / "tables" / "E2_loto_bootstrap_replicates.csv", index=False
    )
    loto_boot_summary = loto_bootstrap_summary(loto_bootstrap)
    loto_boot_summary.to_csv(
        ROOT / "results" / "tables" / "E2_loto_bootstrap_summary.csv", index=False
    )
    loto_profiles, loto_shape_summary = loto_profile_tables(baseline, quick)
    loto_profiles.to_csv(
        ROOT / "results" / "tables" / "E2_loto_profiles.csv", index=False
    )
    loto_shape_summary.to_csv(
        ROOT / "results" / "tables" / "E2_loto_shape_summary.csv", index=False
    )

    hierarchy = hierarchical_comparison(baseline)
    hierarchy.to_csv(
        ROOT / "results" / "tables" / "E2_hierarchical_comparison.csv", index=False
    )

    plot_bootstrap_stability(bootstrap, baseline)
    plot_loto(loto_level, loto_profiles, loto_shape_summary)
    plot_hierarchy(hierarchy, bootstrap)

    summary = summarize(
        baseline,
        bootstrap,
        boot_summary,
        boot_diag,
        loto_level,
        loto_boot_summary,
        loto_shape_summary,
        hierarchy,
    )
    summary["quick_mode"] = bool(quick)
    (ROOT / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    write_results_markdown(summary)
    write_manifest()
    return summary
