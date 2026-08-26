from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
E1_ROOT = REPOSITORY_ROOT / "experiments" / "E1_discover_explain"
E2_ROOT = REPOSITORY_ROOT / "experiments" / "E2_stability_hierarchy"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(E1_ROOT / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import e1_analysis as e1


PHIC = float(e1.rc.PHIC_PACK)
NAMES = e1.NAMES
COLORS = {
    "navy": "#17324D",
    "blue": "#3478A6",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C94C4C",
    "purple": "#7C5AA6",
    "gray": "#6B7280",
}


def mkdirs() -> None:
    for path in [
        ROOT / "results" / "tables",
        ROOT / "results" / "figures",
        ROOT / ".mplconfig",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_baseline() -> dict:
    wells, pooled, metadata = e1.load_data()
    points = {name: e1.rc.calibrate(df, e1.MODEL) for name, df in wells.items()}
    pooled_theta = e1.rc.calibrate(pooled, e1.MODEL)
    discrepancy = e1.transfer_discrepancy(wells, points)
    scale = e1.transfer_aware_scale(discrepancy)
    coordinates = {
        name: e1.metric_contact_hs_decomposition(df, points[name], scale)
        for name, df in wells.items()
    }
    coordinates["pooled"] = e1.metric_contact_hs_decomposition(
        pooled, pooled_theta, scale
    )
    return {
        "wells": wells,
        "pooled": pooled,
        "metadata": metadata,
        "points": points,
        "pooled_theta": pooled_theta,
        "discrepancy": discrepancy,
        "scale": scale,
        "coordinates": coordinates,
    }


def power_increment(x: np.ndarray | float, x0: float, power: float) -> np.ndarray:
    """Finite increment with d/dx=1/x0 at x=x0; power=0 is the log limit."""
    xx = np.asarray(x, dtype=float) / float(x0)
    if abs(power) < 1e-14:
        return np.log(xx)
    return (xx**power - 1.0) / power


FORM_META = {
    "factored_log_log": {
        "label": "log contact + log HS (current)",
        "family": "two-coefficient power continuation",
        "n_local_coefficients": 2,
        "contact_power": 0.0,
        "hs_power": 0.0,
    },
    "log_linear_hs": {
        "label": "log contact + linear HS",
        "family": "two-coefficient power continuation",
        "n_local_coefficients": 2,
        "contact_power": 0.0,
        "hs_power": 1.0,
    },
    "log_reciprocal_hs": {
        "label": "log contact + reciprocal HS",
        "family": "two-coefficient power continuation",
        "n_local_coefficients": 2,
        "contact_power": 0.0,
        "hs_power": -1.0,
    },
    "linear_contact_log_hs": {
        "label": "linear contact + log HS",
        "family": "two-coefficient power continuation",
        "n_local_coefficients": 2,
        "contact_power": 1.0,
        "hs_power": 0.0,
    },
    "reciprocal_contact_log_hs": {
        "label": "reciprocal contact + log HS",
        "family": "two-coefficient power continuation",
        "n_local_coefficients": 2,
        "contact_power": -1.0,
        "hs_power": 0.0,
    },
    "quadratic_tangent": {
        "label": "second-order local tangent",
        "family": "local Taylor continuation",
        "n_local_coefficients": 2,
    },
    "local_linear_tangent": {
        "label": "first-order local tangent",
        "family": "local Taylor continuation",
        "n_local_coefficients": 1,
    },
    "integrated_local_projection": {
        "label": "integrated local projection slope",
        "family": "numerical differential continuation",
        "n_local_coefficients": 0,
    },
}


def coordinate_increment(
    method: str,
    v: np.ndarray | float,
    v0: float,
    A: float,
    Gamma: float,
) -> np.ndarray:
    vv = np.asarray(v, dtype=float)
    phib0 = PHIC - v0
    beta0 = A / v0 - Gamma / phib0
    if method in (
        "factored_log_log",
        "log_linear_hs",
        "log_reciprocal_hs",
        "linear_contact_log_hs",
        "reciprocal_contact_log_hs",
    ):
        meta = FORM_META[method]
        return A * power_increment(vv, v0, meta["contact_power"]) + Gamma * power_increment(
            PHIC - vv, phib0, meta["hs_power"]
        )
    if method == "local_linear_tangent":
        return beta0 * (vv - v0)
    if method == "quadratic_tangent":
        curvature0 = -A / v0**2 - Gamma / phib0**2
        dv = vv - v0
        return beta0 * dv + 0.5 * curvature0 * dv**2
    raise ValueError(method)


def local_tangent(method: str, v0: float, A: float, Gamma: float) -> float:
    h = max(1e-8, 1e-5 * v0)
    return float(
        (
            coordinate_increment(method, v0 + h, v0, A, Gamma)
            - coordinate_increment(method, v0 - h, v0, A, Gamma)
        )
        / (2 * h)
    )


def integrate_projection_curve(
    df: pd.DataFrame,
    theta0: np.ndarray,
    scale: dict[str, float],
    v_values: np.ndarray,
    adjusted: bool,
) -> np.ndarray:
    """Integrate d ln(Cn)/dV=-beta_projection(V,lnCn) from theta0."""
    values = np.asarray(v_values, dtype=float)
    v0, ell0 = map(float, theta0)
    key = "beta_adjusted_projection" if adjusted else "beta_raw_projection"

    def rhs(v: float, y: np.ndarray) -> np.ndarray:
        metric, *_ = e1.metric_summary(
            df, np.array([float(v), float(y[0])]), scale
        )
        return np.array([-float(metric[key])])

    requested = np.unique(np.r_[values, v0])
    lookup: dict[float, float] = {v0: ell0}
    lower = np.sort(requested[requested < v0])[::-1]
    upper = np.sort(requested[requested > v0])
    for side in (lower, upper):
        if not len(side):
            continue
        sol = solve_ivp(
            rhs,
            (v0, float(side[-1])),
            [ell0],
            t_eval=side,
            rtol=2e-7,
            atol=2e-9,
            max_step=0.0015,
        )
        if not sol.success or len(sol.t) != len(side):
            raise RuntimeError(f"Projection-slope integration failed: {sol.message}")
        for v, ell in zip(sol.t, sol.y[0]):
            lookup[float(v)] = float(ell)
    return np.array([lookup[float(v)] for v in values], dtype=float)


def summarize_errors(
    errors: np.ndarray,
    method: str,
    **metadata: object,
) -> dict:
    error = np.asarray(errors, dtype=float)
    ratio = np.exp(error)
    return {
        **metadata,
        "method": method,
        "method_label": FORM_META[method]["label"],
        "family": FORM_META[method]["family"],
        "n_local_coefficients": FORM_META[method]["n_local_coefficients"],
        "n_points": int(len(error)),
        "rms_log_coordinate_error": float(np.sqrt(np.mean(error**2))),
        "max_abs_log_coordinate_error": float(np.max(np.abs(error))),
        "median_abs_percent_ratio_error": float(
            100 * np.median(np.abs(ratio - 1.0))
        ),
        "max_abs_percent_ratio_error": float(100 * np.max(np.abs(ratio - 1.0))),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
    }


def pooled_form_test(baseline: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ridge = pd.read_csv(E1_ROOT / "results" / "tables" / "E1_ridge_validation.csv")
    v = ridge.Vcem_fraction.to_numpy(dtype=float)
    v0, ell0 = map(float, baseline["pooled_theta"])
    v_low = min(float(x[0]) for x in baseline["points"].values())
    v_high = max(float(x[0]) for x in baseline["points"].values())
    rows: list[dict] = []
    summaries: list[dict] = []

    for adjustment in ("raw", "adjusted"):
        adjusted = adjustment == "adjusted"
        coordinate = baseline["coordinates"]["pooled"]
        A = float(coordinate[f"A_{adjustment}"])
        Gamma = float(coordinate[f"Gamma_{adjustment}"])
        observed = ridge[
            "lnCn_nuisance_profiled_structural"
            if adjusted
            else "lnCn_raw_structural_profile"
        ].to_numpy(dtype=float)
        integrated = integrate_projection_curve(
            baseline["pooled"], baseline["pooled_theta"], baseline["scale"], v, adjusted
        )
        predictions: dict[str, np.ndarray] = {
            method: ell0 - coordinate_increment(method, v, v0, A, Gamma)
            for method in FORM_META
            if method != "integrated_local_projection"
        }
        predictions["integrated_local_projection"] = integrated

        common_beta = A / v0 - Gamma / (PHIC - v0)
        for method, predicted in predictions.items():
            q_error = observed - predicted
            if method == "integrated_local_projection":
                tangent = common_beta
            else:
                tangent = local_tangent(method, v0, A, Gamma)
            for i, vi in enumerate(v):
                rows.append(
                    {
                        "adjustment": adjustment,
                        "method": method,
                        "method_label": FORM_META[method]["label"],
                        "Vcem_fraction": float(vi),
                        "Vcem_percent": float(100 * vi),
                        "lnCn_profile": float(observed[i]),
                        "lnCn_predicted": float(predicted[i]),
                        "log_coordinate_error": float(q_error[i]),
                        "coordinate_ratio": float(math.exp(q_error[i])),
                        "local_tangent_at_reference": tangent,
                        "reference_beta": common_beta,
                        "tangent_difference": tangent - common_beta,
                    }
                )
            masks = {
                "local_pm_0p006": np.abs(v - v0) <= 0.006 + 1e-12,
                "between_operating_points": (v >= v_low) & (v <= v_high),
                "full_admissible_grid": np.ones(len(v), dtype=bool),
            }
            for window, mask in masks.items():
                summaries.append(
                    summarize_errors(
                        q_error[mask],
                        method,
                        test="pooled_profile_invariance",
                        adjustment=adjustment,
                        window=window,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def loto_tests(baseline: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    e2_profiles = pd.read_csv(
        E2_ROOT / "results" / "tables" / "E2_loto_profiles.csv"
    )
    level_rows: list[dict] = []
    shape_rows: list[dict] = []
    shape_summary: list[dict] = []
    points = baseline["points"]
    v_low = min(float(x[0]) for x in points.values())
    v_high = max(float(x[0]) for x in points.values())

    for train, test in (("19A", "BT2"), ("BT2", "19A")):
        train_theta = points[train]
        test_theta = points[test]
        v0, ell0 = map(float, train_theta)
        vt, ellt = map(float, test_theta)
        profile_base = e2_profiles[
            (e2_profiles.train == train) & (e2_profiles.test == test)
        ]
        for adjustment in ("raw", "adjusted"):
            adjusted = adjustment == "adjusted"
            coordinate = baseline["coordinates"][train]
            A = float(coordinate[f"A_{adjustment}"])
            Gamma = float(coordinate[f"Gamma_{adjustment}"])
            beta = A / v0 - Gamma / (PHIC - v0)
            profile = profile_base[
                profile_base.adjustment == adjustment
            ].sort_values("Vcem_fraction")
            vshape = profile.Vcem_fraction.to_numpy(dtype=float)
            observed = profile.lnCn_profile.to_numpy(dtype=float)

            integrated_values = np.unique(np.r_[vshape, vt])
            integrated_curve = integrate_projection_curve(
                baseline["wells"][train],
                train_theta,
                baseline["scale"],
                integrated_values,
                adjusted,
            )
            integrated_lookup = {
                float(v): float(ell)
                for v, ell in zip(integrated_values, integrated_curve)
            }
            ell_integrated_test = integrated_lookup[vt]

            for method in FORM_META:
                if method == "integrated_local_projection":
                    predicted_test = ell_integrated_test
                    q_error = ellt - predicted_test
                    predicted_shape = ellt + np.array(
                        [integrated_lookup[float(v)] for v in vshape]
                    ) - ell_integrated_test
                    tangent = beta
                else:
                    h_test = float(coordinate_increment(method, vt, v0, A, Gamma))
                    predicted_test = ell0 - h_test
                    q_error = ellt - predicted_test
                    h_shape = coordinate_increment(method, vshape, v0, A, Gamma)
                    predicted_shape = ellt - (h_shape - h_test)
                    tangent = local_tangent(method, v0, A, Gamma)

                level_rows.append(
                    {
                        "train": train,
                        "test": test,
                        "adjustment": adjustment,
                        "method": method,
                        "method_label": FORM_META[method]["label"],
                        "family": FORM_META[method]["family"],
                        "n_local_coefficients": FORM_META[method][
                            "n_local_coefficients"
                        ],
                        "Vcem_train": v0,
                        "Cn_train": math.exp(ell0),
                        "Vcem_test": vt,
                        "Cn_test": math.exp(ellt),
                        "A_train": A,
                        "Gamma_train": Gamma,
                        "beta_train": beta,
                        "method_tangent_at_train": tangent,
                        "tangent_difference": tangent - beta,
                        "lnCn_test_predicted": predicted_test,
                        "log_coordinate_error": q_error,
                        "coordinate_level_ratio": math.exp(q_error),
                        "absolute_percent_level_error": 100
                        * abs(math.exp(q_error) - 1.0),
                    }
                )

                errors = predicted_shape - observed
                for i, vi in enumerate(vshape):
                    shape_rows.append(
                        {
                            "train": train,
                            "test": test,
                            "adjustment": adjustment,
                            "method": method,
                            "method_label": FORM_META[method]["label"],
                            "Vcem_fraction": float(vi),
                            "Vcem_percent": float(100 * vi),
                            "lnCn_profile": float(observed[i]),
                            "lnCn_predicted": float(predicted_shape[i]),
                            "lnCn_error": float(errors[i]),
                        }
                    )
                masks = {
                    "local_pm_0p015": np.abs(vshape - vt) <= 0.015 + 1e-12,
                    "between_operating_points": (vshape >= v_low)
                    & (vshape <= v_high),
                    "full_admissible_grid": np.ones(len(vshape), dtype=bool),
                }
                for window, mask in masks.items():
                    shape_summary.append(
                        summarize_errors(
                            errors[mask],
                            method,
                            test="held_out_profile_shape",
                            train=train,
                            held_out=test,
                            adjustment=adjustment,
                            window=window,
                        )
                    )
    return (
        pd.DataFrame(level_rows),
        pd.DataFrame(shape_rows),
        pd.DataFrame(shape_summary),
    )


def endpoint_point_closure(baseline: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = baseline["pooled"]
    v, ell = map(float, baseline["pooled_theta"])
    Cn = math.exp(ell)
    phib = PHIC - v
    endpoints = []
    for row in df.itertuples(index=False):
        K0, G0, _ = e1.rc.matrix(float(row.vsh))
        endpoints.append(e1.endpoint_diagnostics(K0, G0, v, Cn))
    endpoint = pd.DataFrame(endpoints)
    mK = float(endpoint.m_N.median())
    mG = float(endpoint.m_G.median())
    alphaK = mK / (4 - mK)
    alphaG = mG / (4 - mG)
    coordinate = baseline["coordinates"]["pooled"]

    rows = [
        {
            "component": "bulk_endpoint_mK",
            "status": "exact endpoint derivative; median across samples",
            "m": mK,
            "dimensionless_contact_coefficient": alphaK,
            "beta_contribution_per_fraction": alphaK / v,
        },
        {
            "component": "shear_endpoint_mG",
            "status": "exact endpoint derivative; median across samples",
            "m": mG,
            "dimensionless_contact_coefficient": alphaG,
            "beta_contribution_per_fraction": alphaG / v,
        },
    ]
    for adjustment in ("raw", "adjusted"):
        A = float(coordinate[f"A_{adjustment}"])
        Gamma = float(coordinate[f"Gamma_{adjustment}"])
        contact = A / v
        hs = -Gamma / phib
        rows.extend(
            [
                {
                    "component": f"observable_weighted_contact_{adjustment}",
                    "status": "exact local chain rule in stated data metric",
                    "m": np.nan,
                    "dimensionless_contact_coefficient": A,
                    "beta_contribution_per_fraction": contact,
                },
                {
                    "component": f"HS_porosity_path_{adjustment}",
                    "status": "exact local chain rule in stated data metric",
                    "m": np.nan,
                    "dimensionless_contact_coefficient": Gamma,
                    "beta_contribution_per_fraction": hs,
                },
                {
                    "component": f"complete_observable_coordinate_{adjustment}",
                    "status": "sum of contact and HS-path projections",
                    "m": np.nan,
                    "dimensionless_contact_coefficient": np.nan,
                    "beta_contribution_per_fraction": contact + hs,
                },
            ]
        )

    # Raw property weights in the denominator of the projection coefficient.
    loga = math.log(float(endpoint.a_c.iloc[0]))
    h = 1e-5
    hp = 1e-6
    Aout = (
        e1.latent_stacked_outputs(df, loga + h, ell, phib)
        - e1.latent_stacked_outputs(df, loga - h, ell, phib)
    ) / (2 * h)
    Bout = (
        e1.latent_stacked_outputs(df, loga, ell, phib + hp)
        - e1.latent_stacked_outputs(df, loga, ell, phib - hp)
    ) / (2 * hp)
    Cout = (
        e1.latent_stacked_outputs(df, loga, ell + h, phib)
        - e1.latent_stacked_outputs(df, loga, ell - h, phib)
    ) / (2 * h)
    sigma = e1.stacked_sigma(df, baseline["scale"])
    jell = (Cout - Aout / 4) / sigma
    jcontact = Aout / 4 / sigma
    jphib = Bout / sigma
    denominator = float(jell @ jell)
    contact_numerator = float(jell @ jcontact)
    hs_numerator = float(jell @ jphib)
    weights = []
    n = len(df)
    for iprop, prop in enumerate(NAMES):
        block = slice(iprop * n, (iprop + 1) * n)
        weights.append(
            {
                "observable": prop,
                "raw_projection_denominator_weight": float(
                    (jell[block] @ jell[block]) / denominator
                ),
                "raw_contact_numerator_weight": float(
                    (jell[block] @ jcontact[block]) / contact_numerator
                )
                if contact_numerator
                else np.nan,
                "raw_HS_numerator_weight": float(
                    (jell[block] @ jphib[block]) / hs_numerator
                )
                if hs_numerator
                else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(weights)


def moving_block_sample(
    df: pd.DataFrame, block_length: int, rng: np.random.Generator
) -> pd.DataFrame:
    n = len(df)
    starts = rng.integers(0, n - block_length + 1, size=int(math.ceil(n / block_length)))
    indices = np.concatenate(
        [np.arange(start, start + block_length) for start in starts]
    )[:n]
    return df.iloc[indices].reset_index(drop=True)


def bootstrap_endpoint_link(baseline: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    saved = pd.read_csv(
        E2_ROOT / "results" / "tables" / "E2_bootstrap_replicates.csv"
    )
    saved = saved[
        (saved.scheme == "MBB_20m_primary")
        & saved.bootstrap_success.fillna(False)
    ].sort_values("replicate")
    if len(saved) != 400:
        raise RuntimeError(f"Expected 400 primary replicates, found {len(saved)}")

    seed_children = np.random.SeedSequence(20260823).spawn(4)
    rng = np.random.default_rng(seed_children[2])
    vref = float(baseline["pooled_theta"][0])
    phibref = PHIC - vref
    rows = []
    for record in saved.itertuples(index=False):
        sampled = {
            name: moving_block_sample(baseline["wells"][name], 5, rng)
            for name in ("19A", "BT2")
        }
        pooled = pd.concat([sampled["19A"], sampled["BT2"]], ignore_index=True)
        theta = np.array([float(record.Vcem_fraction), math.log(float(record.Cn))])
        v, ell = map(float, theta)
        Cn = math.exp(ell)
        endpoint = []
        for sample in pooled.itertuples(index=False):
            K0, G0, _ = e1.rc.matrix(float(sample.vsh))
            endpoint.append(e1.endpoint_diagnostics(K0, G0, v, Cn))
        endpoint_df = pd.DataFrame(endpoint)
        mK = float(endpoint_df.m_N.median())
        mG = float(endpoint_df.m_G.median())
        alphaK = mK / (4 - mK)
        alphaG = mG / (4 - mG)

        # Recompute the full observable-weighted decomposition from the sampled data.
        # This is an audit of the saved E2 distribution, not a new fit.
        recomputed = e1.metric_contact_hs_decomposition(
            pooled, theta, baseline["scale"]
        )
        rows.append(
            {
                "replicate": int(record.replicate),
                "Vcem_fraction": v,
                "Cn": Cn,
                "median_mK": mK,
                "median_mG": mG,
                "endpoint_AK": alphaK,
                "endpoint_AG": alphaG,
                "endpoint_betaK_at_pooled_V": alphaK / vref,
                "endpoint_betaG_at_pooled_V": alphaG / vref,
                "contact_beta_raw_at_pooled_V": float(record.A_raw) / vref,
                "HS_beta_raw_at_pooled_V": -float(record.Gamma_raw) / phibref,
                "total_beta_raw_at_pooled_V": float(record.beta_raw_at_pooled),
                "contact_beta_adjusted_at_pooled_V": float(record.A_adjusted)
                / vref,
                "HS_beta_adjusted_at_pooled_V": -float(record.Gamma_adjusted)
                / phibref,
                "total_beta_adjusted_at_pooled_V": float(
                    record.beta_adjusted_at_pooled
                ),
                "A_raw_saved": float(record.A_raw),
                "Gamma_raw_saved": float(record.Gamma_raw),
                "A_adjusted_saved": float(record.A_adjusted),
                "Gamma_adjusted_saved": float(record.Gamma_adjusted),
                "A_raw_recomputed": float(recomputed["A_raw"]),
                "Gamma_raw_recomputed": float(recomputed["Gamma_raw"]),
                "A_adjusted_recomputed": float(recomputed["A_adjusted"]),
                "Gamma_adjusted_recomputed": float(recomputed["Gamma_adjusted"]),
            }
        )
        if (len(rows) % 50) == 0:
            print(f"endpoint/bootstrap audit: {len(rows)}/400", flush=True)
    replicates = pd.DataFrame(rows)

    metrics = [
        "median_mK",
        "median_mG",
        "endpoint_AK",
        "endpoint_AG",
        "endpoint_betaK_at_pooled_V",
        "endpoint_betaG_at_pooled_V",
        "contact_beta_raw_at_pooled_V",
        "HS_beta_raw_at_pooled_V",
        "total_beta_raw_at_pooled_V",
        "contact_beta_adjusted_at_pooled_V",
        "HS_beta_adjusted_at_pooled_V",
        "total_beta_adjusted_at_pooled_V",
        "A_raw_saved",
        "Gamma_raw_saved",
        "A_adjusted_saved",
        "Gamma_adjusted_saved",
    ]
    summary_rows = []
    for metric in metrics:
        x = replicates[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "n": len(x),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "q02p5": float(np.quantile(x, 0.025)),
                "q16": float(np.quantile(x, 0.16)),
                "q84": float(np.quantile(x, 0.84)),
                "q97p5": float(np.quantile(x, 0.975)),
                "std": float(np.std(x, ddof=1)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    closure = {
        key: float(np.max(np.abs(replicates[f"{key}_saved"] - replicates[f"{key}_recomputed"])))
        for key in ("A_raw", "Gamma_raw", "A_adjusted", "Gamma_adjusted")
    }
    closure["fraction_A_raw_between_endpoint_AK_AG"] = float(
        np.mean(
            (replicates.A_raw_saved >= replicates[["endpoint_AK", "endpoint_AG"]].min(axis=1))
            & (replicates.A_raw_saved <= replicates[["endpoint_AK", "endpoint_AG"]].max(axis=1))
        )
    )
    closure["corr_A_raw_endpoint_midpoint"] = float(
        np.corrcoef(
            replicates.A_raw_saved,
            0.5 * (replicates.endpoint_AK + replicates.endpoint_AG),
        )[0, 1]
    )
    closure["rmse_A_raw_endpoint_midpoint"] = float(
        np.sqrt(
            np.mean(
                (
                    replicates.A_raw_saved
                    - 0.5 * (replicates.endpoint_AK + replicates.endpoint_AG)
                )
                ** 2
            )
        )
    )
    return replicates, summary, closure


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(
            ROOT / "results" / "figures" / f"{stem}.{suffix}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_coordinate_forms(
    pooled_curves: pd.DataFrame,
    loto_level: pd.DataFrame,
    loto_shape_summary: pd.DataFrame,
) -> None:
    configure_plotting()
    methods = [
        "factored_log_log",
        "log_linear_hs",
        "log_reciprocal_hs",
        "linear_contact_log_hs",
        "local_linear_tangent",
        "integrated_local_projection",
    ]
    colors = {
        "factored_log_log": COLORS["navy"],
        "log_linear_hs": COLORS["blue"],
        "log_reciprocal_hs": COLORS["teal"],
        "linear_contact_log_hs": COLORS["orange"],
        "local_linear_tangent": COLORS["gray"],
        "integrated_local_projection": COLORS["red"],
    }
    styles = {
        "factored_log_log": "-",
        "log_linear_hs": "--",
        "log_reciprocal_hs": ":",
        "linear_contact_log_hs": "-.",
        "local_linear_tangent": (0, (1, 2)),
        "integrated_local_projection": "-",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for ax, adjustment in zip(axes[0], ("raw", "adjusted")):
        subset = pooled_curves[pooled_curves.adjustment == adjustment]
        for method in methods:
            d = subset[subset.method == method]
            ax.plot(
                d.Vcem_percent,
                100 * (d.coordinate_ratio - 1),
                label=FORM_META[method]["label"],
                color=colors[method],
                ls=styles[method],
                lw=2.0 if method in ("factored_log_log", "integrated_local_projection") else 1.5,
            )
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.05, linscale=1.15)
        ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
        ax.set_ylabel("coordinate drift (%)")
        ax.set_title(f"Pooled {adjustment} ridge")
        ax.set_ylim(-1.5, 120)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 0]
    level = loto_level[loto_level.method.isin(methods)].copy()
    level["case"] = (
        level.train + r"$\rightarrow$" + level.test + " " + level.adjustment
    )
    cases = level.case.drop_duplicates().tolist()
    x = np.arange(len(cases), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    for off, method in zip(offsets, methods):
        d = level[level.method == method].set_index("case").reindex(cases)
        ax.scatter(
            x + off,
            d.coordinate_level_ratio,
            s=35,
            color=colors[method],
            marker="o" if method != "integrated_local_projection" else "D",
            label=FORM_META[method]["label"],
        )
    ax.axhline(1.0, color="0.65", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=18, ha="right")
    ax.set_ylabel("held-out level ratio (ideal = 1)")
    ax.set_title("Leave-one-trajectory-out level transport")

    ax = axes[1, 1]
    shape = loto_shape_summary[
        (loto_shape_summary.window == "between_operating_points")
        & loto_shape_summary.method.isin(methods)
    ].copy()
    aggregate = (
        shape.groupby("method", as_index=False)
        .agg(
            mean_rms=("rms_log_coordinate_error", "mean"),
            worst_rms=("rms_log_coordinate_error", "max"),
        )
        .set_index("method")
        .reindex(methods)
    )
    y = np.arange(len(methods))
    ax.barh(
        y,
        aggregate.mean_rms,
        color=[colors[m] for m in methods],
        alpha=0.85,
        label="mean across 4 held-out cases",
    )
    ax.scatter(
        aggregate.worst_rms,
        y,
        facecolors="white",
        edgecolors="black",
        s=28,
        label="worst case",
        zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([FORM_META[m]["label"] for m in methods])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(r"held-out ridge-shape RMS in $\ln C_n$")
    ax.set_title("Shape transport between operating points")
    ax.legend(frameon=False)
    fig.suptitle(
        "Finite-coordinate alternatives with a matched local tangent\n"
        "Power-family variants use the same A and Gamma; the integrated curve is a numerical reference"
    )
    save_figure(fig, "Fig_Q1_finite_coordinate_robustness")


def plot_endpoint_link(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    ax = axes[0]
    names = [
        "bulk_endpoint_mK",
        "shear_endpoint_mG",
        "observable_weighted_contact_raw",
        "HS_porosity_path_raw",
        "complete_observable_coordinate_raw",
        "observable_weighted_contact_adjusted",
        "HS_porosity_path_adjusted",
        "complete_observable_coordinate_adjusted",
    ]
    labels = [
        r"endpoint $K_b$",
        r"endpoint $G_b$",
        "contact, raw",
        "HS path, raw",
        "total, raw",
        "contact, adjusted",
        "HS path, adjusted",
        "total, adjusted",
    ]
    p = point.set_index("component").reindex(names)
    values = p.beta_contribution_per_fraction.to_numpy()
    colors = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["navy"],
        COLORS["orange"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["gold"],
        COLORS["red"],
    ]
    x = np.arange(len(names))
    ax.bar(x, values, color=colors)
    ax.axhline(20.3, color="black", ls=":", lw=1.3, label="empirical 20.3")
    ax.axhline(0, color="0.65", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"slope contribution $\beta$ (per fraction)")
    ax.set_title("Endpoint benchmark and HS-path contribution")
    ax.legend(frameon=False)

    ax = axes[1]
    metrics = [
        "endpoint_betaK_at_pooled_V",
        "endpoint_betaG_at_pooled_V",
        "contact_beta_raw_at_pooled_V",
        "total_beta_raw_at_pooled_V",
        "contact_beta_adjusted_at_pooled_V",
        "total_beta_adjusted_at_pooled_V",
    ]
    labels = [
        r"endpoint $K_b$",
        r"endpoint $G_b$",
        "contact raw",
        "total raw",
        "contact adjusted",
        "total adjusted",
    ]
    data = [bootstrap[m].to_numpy(dtype=float) for m in metrics]
    box = ax.boxplot(data, patch_artist=True, widths=0.65, showfliers=False)
    for patch, color in zip(
        box["boxes"],
        [COLORS["blue"], COLORS["teal"], COLORS["navy"], COLORS["red"], COLORS["purple"], COLORS["orange"]],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.axhline(20.3, color="black", ls=":", lw=1.3)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylabel(r"$\beta$ at the fixed pooled $V_0$")
    ax.set_title("20-m moving-block bootstrap (400 replicates)")
    fig.suptitle(
        "Model-internal decomposition: contact term plus HS-path contribution"
    )
    save_figure(fig, "Fig_Q2_endpoint_slope_closure")


def write_report(
    pooled_summary: pd.DataFrame,
    loto_level: pd.DataFrame,
    loto_shape_summary: pd.DataFrame,
    endpoint_point: pd.DataFrame,
    property_weights: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    closure: dict,
) -> dict:
    def pooled(method: str, adjustment: str, window: str) -> pd.Series:
        return pooled_summary[
            (pooled_summary.method == method)
            & (pooled_summary.adjustment == adjustment)
            & (pooled_summary.window == window)
        ].iloc[0]

    def bsum(metric: str) -> pd.Series:
        return bootstrap_summary.set_index("metric").loc[metric]

    point = endpoint_point.set_index("component")
    full_raw = {
        method: pooled(method, "raw", "full_admissible_grid")
        for method in FORM_META
    }
    between_raw = {
        method: pooled(method, "raw", "between_operating_points")
        for method in FORM_META
    }
    level_primary = loto_level[
        loto_level.method.isin(
            [
                "factored_log_log",
                "log_linear_hs",
                "log_reciprocal_hs",
                "linear_contact_log_hs",
                "local_linear_tangent",
                "integrated_local_projection",
            ]
        )
    ]
    level_worst = (
        level_primary.groupby("method").absolute_percent_level_error.max().to_dict()
    )
    shape_between = loto_shape_summary[
        loto_shape_summary.window == "between_operating_points"
    ]
    shape_mean = (
        shape_between.groupby("method").rms_log_coordinate_error.mean().to_dict()
    )
    tangent_max = float(
        np.max(np.abs(loto_level.tangent_difference.to_numpy(dtype=float)))
    )

    numeric = {
        "pooled_full_raw_max_abs_percent_drift": {
            method: float(row.max_abs_percent_ratio_error)
            for method, row in full_raw.items()
        },
        "pooled_between_points_raw_rms_log_error": {
            method: float(row.rms_log_coordinate_error)
            for method, row in between_raw.items()
        },
        "loto_worst_absolute_level_error_percent": {
            key: float(value) for key, value in level_worst.items()
        },
        "loto_mean_between_points_shape_rms_lnCn": {
            key: float(value) for key, value in shape_mean.items()
        },
        "maximum_numerical_tangent_mismatch": tangent_max,
        "endpoint_point": {
            key: float(point.loc[key, "beta_contribution_per_fraction"])
            for key in point.index
        },
        "bootstrap_fixed_pooled_V": {
            metric: {
                "median": float(bsum(metric)["median"]),
                "ci95": [float(bsum(metric).q02p5), float(bsum(metric).q97p5)],
            }
            for metric in [
                "endpoint_betaK_at_pooled_V",
                "endpoint_betaG_at_pooled_V",
                "contact_beta_raw_at_pooled_V",
                "HS_beta_raw_at_pooled_V",
                "total_beta_raw_at_pooled_V",
                "contact_beta_adjusted_at_pooled_V",
                "HS_beta_adjusted_at_pooled_V",
                "total_beta_adjusted_at_pooled_V",
                "A_raw_saved",
                "Gamma_raw_saved",
                "A_adjusted_saved",
                "Gamma_adjusted_saved",
            ]
        },
        "bootstrap_recomputation_closure": closure,
        "raw_observable_projection_weights": property_weights.to_dict(
            orient="records"
        ),
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(numeric, indent=2))

    fact = full_raw["factored_log_log"]
    integrated = full_raw["integrated_local_projection"]
    tangent = full_raw["local_linear_tangent"]
    loglin = full_raw["log_linear_hs"]
    logrec = full_raw["log_reciprocal_hs"]
    lincontact = full_raw["linear_contact_log_hs"]
    bK = point.loc["bulk_endpoint_mK", "beta_contribution_per_fraction"]
    bG = point.loc["shear_endpoint_mG", "beta_contribution_per_fraction"]
    cRaw = point.loc[
        "observable_weighted_contact_raw", "beta_contribution_per_fraction"
    ]
    hsRaw = point.loc["HS_porosity_path_raw", "beta_contribution_per_fraction"]
    totalRaw = point.loc[
        "complete_observable_coordinate_raw", "beta_contribution_per_fraction"
    ]
    cAdj = point.loc[
        "observable_weighted_contact_adjusted", "beta_contribution_per_fraction"
    ]
    hsAdj = point.loc[
        "HS_porosity_path_adjusted", "beta_contribution_per_fraction"
    ]
    totalAdj = point.loc[
        "complete_observable_coordinate_adjusted", "beta_contribution_per_fraction"
    ]
    total_adj_bs = bsum("total_beta_adjusted_at_pooled_V")
    A_adj_bs = bsum("A_adjusted_saved")
    G_adj_bs = bsum("Gamma_adjusted_saved")

    report = f"""# Scientific audit of the finite coordinate and Appendix slope self-consistency

## Bottom line

The current factored coordinate is strongly supported as an accurate **low-dimensional continuation** of the local ridge, but it is not mathematically unique. Along the complete pooled raw structural ridge, its worst coordinate drift is {fact.max_abs_percent_ratio_error:.3f} per cent, compared with {integrated.max_abs_percent_ratio_error:.4f} per cent for direct numerical integration of the local projection slope and {tangent.max_abs_percent_ratio_error:.1f} per cent for the first-order tangent. Two equal-coefficient alternatives that keep the logarithmic contact term but change the finite Hashin--Shtrikman continuation are almost as accurate ({loglin.max_abs_percent_ratio_error:.3f} and {logrec.max_abs_percent_ratio_error:.3f} per cent). Replacing the logarithmic contact continuation by a linear one produces {lincontact.max_abs_percent_ratio_error:.1f} per cent drift. Thus the data identify the need for a strongly curved/log-like contact term; they discriminate much less strongly among plausible finite forms for the smaller HS-path correction.

In the genuinely held-out level test, the worst absolute error is {level_worst['factored_log_log']:.2f} per cent for the current factorization and {level_worst['integrated_local_projection']:.2f} per cent for integrated local slopes, versus {level_worst['local_linear_tangent']:.1f} per cent for the local tangent. The log-contact/linear-HS and log-contact/reciprocal-HS alternatives have worst errors of {level_worst['log_linear_hs']:.2f} and {level_worst['log_reciprocal_hs']:.2f} per cent. All power-family variants use the same two local coefficients, the same operating points, the same weighting, and have the same first derivative at the training point (maximum finite-difference mismatch {tangent_max:.2e}). No coefficient was re-fitted to the held-out trajectory.

The endpoint calculation supplies a model-internal analytic decomposition and an implementation self-consistency check; it is not an independent validation of Scheme 1. At the pooled point, the exact Scheme-1 endpoint exponents give beta_K={bK:.3f} and beta_G={bG:.3f}. Their observable-weighted contact projection is {cRaw:.3f}, so the endpoint physics accounts for the **contact component**, not 20.3 directly. The moving endpoint/HS path contributes {hsRaw:.3f}, yielding {totalRaw:.3f}; nuisance projection changes these to {cAdj:.3f} and {hsAdj:.3f}, yielding {totalAdj:.3f}. Because `A` and `Gamma` are defined from this same projected derivative, agreement with the numerically stored coefficients verifies analytic--numerical consistency rather than supplying external physical confirmation.

## What is exact

1. In Scheme 1, `a_c` obeys `d ln(a_c) = (d ln(Vcem)-d ln(Cn))/4`.
2. For an endpoint modulus with exponent `m=d ln(S)/d ln(a_c)`, the constant-endpoint direction is `d ln(Cn) + [m/(4-m)] d ln(Vcem)=0`; therefore `beta=m/[V(4-m)]`.
3. At the pooled point, the analytic normal and effective shear endpoint exponents are represented by the sample medians `mK={float(point.loc['bulk_endpoint_mK','m']):.6f}` and `mG={float(point.loc['shear_endpoint_mG','m']):.6f}`.
4. Propagating those endpoint derivatives through the modified HS path, Gassmann substitution, `Vp`/`Vs` weighting and (when requested) the Schur nuisance projection gives the contact and HS contributions listed above. Recomputing the entire decomposition for all 400 saved E2 bootstrap samples recovers the stored `A` and `Gamma` values with maximum absolute differences {max(closure[k] for k in ['A_raw','Gamma_raw','A_adjusted','Gamma_adjusted']):.3e}. This near-identity is expected because both routes use the same projected local derivatives.
5. The density block has zero target-direction weight in the raw projection; the raw denominator weights are {100*property_weights.set_index('observable').loc['Vp','raw_projection_denominator_weight']:.1f} per cent for `Vp` and {100*property_weights.set_index('observable').loc['Vs','raw_projection_denominator_weight']:.1f} per cent for `Vs`. Density may still constrain nuisances in the adjusted calculation.

## What is a controlled approximation

1. `q_star` is obtained by freezing the two local chain-rule coefficients and integrating them with log factors. It is a two-coefficient finite ansatz, not a unique microscopic state variable.
2. The power-family alternatives use `[(x/x0)^r-1]/r`, whose log limit is `r=0`. They have identical value and derivative at the reference point and differ only at second and higher order. This makes the comparison a direct test of finite continuation rather than local fit quality.
3. The numerical integrated-projection curve recalculates the local metric along the path. It is the closest differential reconstruction of the local ridge, but it is more complex than a two-coefficient coordinate and is therefore a reference, not a same-complexity competitor.
4. The adjusted numerical integration follows the local Schur projection. It is not a fully non-linear Bayesian marginalization over nuisances; the held-out adjusted profile itself uses the E2 non-linear nuisance-MAP profile under the fixed priors.
5. Medians of endpoint `mK` and `mG` summarize small mineralogical variation. The exact observable-weighted projection is calculated from every sample and observable, rather than from the two medians alone.

## What is empirical

1. At a fixed pooled `V0`, the 20-m block bootstrap gives adjusted `A` median {A_adj_bs['median']:.4f} (conditional 2.5--97.5 percentile range {A_adj_bs.q02p5:.4f}--{A_adj_bs.q97p5:.4f}) and adjusted `Gamma` median {G_adj_bs['median']:.4f} ({G_adj_bs.q02p5:.4f}--{G_adj_bs.q97p5:.4f}). Their combined slope has median {total_adj_bs['median']:.3f} ({total_adj_bs.q02p5:.3f}--{total_adj_bs.q97p5:.3f}). With only about six block-length units per trajectory, these are conditional resampling diagnostics with uncertain finite-sample coverage, not between-well population uncertainty.
2. The held-out comparisons use the two related Hugin trajectories. They demonstrate internal transport, not external geological universality.
3. The current log-log form is not uniquely selected: keeping a log contact term while using linear or reciprocal finite HS terms changes held-out errors only slightly. Manuscript language should therefore say **a stable physics-factored coordinate** or **the selected low-order coordinate**, not imply uniqueness.

## Recommended manuscript changes for criticisms 4 and 6

1. Add a compact robustness paragraph/table reporting the matched-tangent power-family comparison and the integrated-slope reference. The defensible claim is that the log-contact structure is robust, while the exact finite HS factor is weakly resolved over this range.
2. Replace Appendix A with the executed endpoint calculation: report `mK`, `mG`, endpoint beta values, observable-weighted contact beta, HS correction and total, both raw and adjusted, explicitly as a model-internal decomposition.
3. Add the bootstrap self-consistency audit: endpoint/contact terms remain near 24, while the HS correction moves the total to the range containing 20.3. Do not present recovery of `(A,Gamma)` as an independent prediction; the same projected local derivatives define both routes. The endpoint exponents constrain the contact contribution underlying `A`, whereas `Gamma` comes from the moving-porosity/HS derivative.
4. Describe numerical slope integration as a differential reconstruction, not as another two-parameter model. Its near-exact in-sample result is expected because a profiled least-squares ridge has tangent `-G12/G22`; its useful evidence is the held-out transport result.

## Machine-readable outputs

- `results/tables/Q1_pooled_coordinate_curves.csv`
- `results/tables/Q1_pooled_coordinate_summary.csv`
- `results/tables/Q1_loto_level.csv`
- `results/tables/Q1_loto_shape_curves.csv`
- `results/tables/Q1_loto_shape_summary.csv`
- `results/tables/Q2_endpoint_point_closure.csv`
- `results/tables/Q2_observable_projection_weights.csv`
- `results/tables/Q2_bootstrap_endpoint_replicates.csv`
- `results/tables/Q2_bootstrap_endpoint_summary.csv`
- `results/figures/Fig_Q1_finite_coordinate_robustness.*`
- `results/figures/Fig_Q2_endpoint_slope_closure.*`
"""
    (ROOT / "REPORT.md").write_text(report)
    return numeric


def run() -> dict:
    mkdirs()
    baseline = load_baseline()
    pooled_curves, pooled_summary = pooled_form_test(baseline)
    loto_level, loto_shape, loto_shape_summary = loto_tests(baseline)
    endpoint_point, property_weights = endpoint_point_closure(baseline)
    bootstrap, bootstrap_summary, closure = bootstrap_endpoint_link(baseline)

    tables = {
        "Q1_pooled_coordinate_curves.csv": pooled_curves,
        "Q1_pooled_coordinate_summary.csv": pooled_summary,
        "Q1_loto_level.csv": loto_level,
        "Q1_loto_shape_curves.csv": loto_shape,
        "Q1_loto_shape_summary.csv": loto_shape_summary,
        "Q2_endpoint_point_closure.csv": endpoint_point,
        "Q2_observable_projection_weights.csv": property_weights,
        "Q2_bootstrap_endpoint_replicates.csv": bootstrap,
        "Q2_bootstrap_endpoint_summary.csv": bootstrap_summary,
    }
    for filename, table in tables.items():
        table.to_csv(ROOT / "results" / "tables" / filename, index=False)

    plot_coordinate_forms(pooled_curves, loto_level, loto_shape_summary)
    plot_endpoint_link(endpoint_point, bootstrap)
    summary = write_report(
        pooled_summary,
        loto_level,
        loto_shape_summary,
        endpoint_point,
        property_weights,
        bootstrap_summary,
        closure,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run()
