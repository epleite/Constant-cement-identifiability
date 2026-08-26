from __future__ import annotations

"""Non-linear robustness audit for the constant-cement pressure-design gains.

This script intentionally keeps two objects separate:

1. a fully non-linear nuisance-MAP profile objective, evaluated at fixed
   cement volume while jointly optimizing ln(Cn) and all physical nuisances;
2. the pointwise Gauss--Newton efficient information (Schur complement),
   recalculated at every nuisance-MAP point on each non-linear profile.

The pointwise matrices are local metrics.  They are not integrated or averaged
and are not presented as a global likelihood or as independent information.
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
E1_ROOT = REPOSITORY_ROOT / "experiments" / "E1_discover_explain"
E3_ROOT = REPOSITORY_ROOT / "experiments" / "E3_break_design"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
sys.path.insert(0, str(E3_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import e3_analysis as e3  # noqa: E402
import e3_model as pm  # noqa: E402


rc = pm.rc
e1 = pm.e1

DESIGN = (5.0, 7.5)
TARGET_SCALES = np.asarray(rc.PARAM_SCALES[e1.MODEL], dtype=float)
STATIC_NAMES = rc.nuisance_names(e1.MODEL)
ALL_NAMES = pm.all_nuisance_names()
ALL_SCALES = pm.all_nuisance_scales()
FD_STEP = 1.0e-4


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    fabric_mode: str
    aligned_sigma: float
    local_reference_gain: float


SCENARIOS = (
    Scenario(
        "shared_generic",
        "Shared fabric + generic discrepancy",
        "shared",
        0.0,
        633.7761651279503,
    ),
    Scenario(
        "expanded_generic",
        "Expanded fabric nuisances",
        "expanded_nuisance",
        0.0,
        3.3511375010382625,
    ),
    Scenario(
        "expanded_target_aligned",
        "Expanded + target-aligned discrepancy",
        "expanded_nuisance",
        0.010,
        1.2068996463319068,
    ),
)


def ensure_directories() -> None:
    for path in [
        HERE / ".mplconfig",
        HERE / "results",
        HERE / "results" / "figures",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def nuisance_dict(names: list[str], z: np.ndarray) -> dict[str, float]:
    return {
        name: float(ALL_SCALES[name] * value)
        for name, value in zip(names, np.asarray(z, dtype=float))
    }


def static_vector(
    baseline: e3.Baseline,
    theta: np.ndarray,
    z_static: np.ndarray,
) -> np.ndarray:
    return rc.stack(
        baseline.pooled,
        e1.MODEL,
        np.asarray(theta, dtype=float),
        e1.NAMES,
        nuisance_dict(STATIC_NAMES, z_static),
    )


def pressure_vector(
    baseline: e3.Baseline,
    theta: np.ndarray,
    z_all: np.ndarray,
    fabric_mode: str,
) -> np.ndarray:
    config = e3.FABRIC_CONFIGS[fabric_mode]
    return pm.pressure_differential_vector(
        baseline.pooled,
        np.asarray(theta, dtype=float),
        list(DESIGN),
        nuisance_dict(ALL_NAMES, z_all),
        soft_lncn_reference=float(baseline.theta[1]),
        stiff_lncn_reference=float(baseline.theta[1]),
        soft_phic_reference=float(rc.PHIC_PACK),
        **config,
    )


def whiten_pressure_vector(vector: np.ndarray, n_samples: int) -> np.ndarray:
    target = np.asarray(vector, dtype=float).reshape(-1, 1)
    nuisance = np.zeros((len(target), 1), dtype=float)
    out, _ = e3.whiten_pressure_blocks(
        target,
        nuisance,
        len(DESIGN),
        n_samples,
        e3.PRIMARY_STATE_LOG_SIGMA,
        shared_reference=True,
    )
    return out[:, 0]


def whiten_pressure_matrices(
    target: np.ndarray,
    nuisance: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    return e3.whiten_pressure_blocks(
        target,
        nuisance,
        len(DESIGN),
        n_samples,
        e3.PRIMARY_STATE_LOG_SIGMA,
        shared_reference=True,
    )


def generic_discrepancy_matrix(baseline: e3.Baseline) -> np.ndarray:
    """Whitened 1%-RMS trajectory/state/property discrepancy basis."""

    n_samples = len(baseline.pooled)
    features = e3.model_discrepancy_features(baseline)
    trajectory_lengths = tuple(len(baseline.wells[name]) for name in baseline.wells)
    n_columns = len(DESIGN) * 2 * len(trajectory_lengths) * features.shape[1]
    output = np.zeros((len(DESIGN) * 2 * n_samples, n_columns), dtype=float)
    offsets = np.cumsum((0,) + trajectory_lengths)
    column = 0
    for state in range(len(DESIGN)):
        for prop in range(2):
            block_start = (2 * state + prop) * n_samples
            for trajectory in range(len(trajectory_lengths)):
                sample_start = int(offsets[trajectory])
                sample_stop = int(offsets[trajectory + 1])
                start = block_start + sample_start
                stop = block_start + sample_stop
                for feature in range(features.shape[1]):
                    output[start:stop, column] = (
                        e3.PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA
                        * features[sample_start:sample_stop, feature]
                    )
                    column += 1
    _, whitened = whiten_pressure_matrices(
        np.zeros((len(output), 2), dtype=float), output, n_samples
    )
    return whitened


def static_jacobians(
    baseline: e3.Baseline,
    theta: np.ndarray,
    z_static: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = e1.stacked_sigma(baseline.pooled, baseline.scale)
    target = np.zeros((len(sigma), 2), dtype=float)
    for j in range(2):
        plus = np.asarray(theta, dtype=float).copy()
        minus = np.asarray(theta, dtype=float).copy()
        plus[j] += FD_STEP * TARGET_SCALES[j]
        minus[j] -= FD_STEP * TARGET_SCALES[j]
        target[:, j] = (
            static_vector(baseline, plus, z_static)
            - static_vector(baseline, minus, z_static)
        ) / (2.0 * FD_STEP * sigma)

    nuisance = np.zeros((len(sigma), len(STATIC_NAMES)), dtype=float)
    for j in range(len(STATIC_NAMES)):
        plus = np.asarray(z_static, dtype=float).copy()
        minus = np.asarray(z_static, dtype=float).copy()
        plus[j] += FD_STEP
        minus[j] -= FD_STEP
        nuisance[:, j] = (
            static_vector(baseline, theta, plus)
            - static_vector(baseline, theta, minus)
        ) / (2.0 * FD_STEP * sigma)
    return target, nuisance


def pressure_jacobians(
    baseline: e3.Baseline,
    theta: np.ndarray,
    z_all: np.ndarray,
    fabric_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = 2 * len(DESIGN) * len(baseline.pooled)
    target_raw = np.zeros((n_rows, 2), dtype=float)
    for j in range(2):
        plus = np.asarray(theta, dtype=float).copy()
        minus = np.asarray(theta, dtype=float).copy()
        plus[j] += FD_STEP * TARGET_SCALES[j]
        minus[j] -= FD_STEP * TARGET_SCALES[j]
        target_raw[:, j] = (
            pressure_vector(baseline, plus, z_all, fabric_mode)
            - pressure_vector(baseline, minus, z_all, fabric_mode)
        ) / (2.0 * FD_STEP)

    nuisance_raw = np.zeros((n_rows, len(ALL_NAMES)), dtype=float)
    for j in range(len(ALL_NAMES)):
        plus = np.asarray(z_all, dtype=float).copy()
        minus = np.asarray(z_all, dtype=float).copy()
        plus[j] += FD_STEP
        minus[j] -= FD_STEP
        nuisance_raw[:, j] = (
            pressure_vector(baseline, theta, plus, fabric_mode)
            - pressure_vector(baseline, theta, minus, fabric_mode)
        ) / (2.0 * FD_STEP)
    target, nuisance = whiten_pressure_matrices(
        target_raw, nuisance_raw, len(baseline.pooled)
    )
    return target, nuisance, target_raw


def schur(target: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    gram = target.T @ target
    if nuisance.shape[1] == 0:
        return (gram + gram.T) / 2.0
    cross = target.T @ nuisance
    adjusted = gram - cross @ np.linalg.solve(
        nuisance.T @ nuisance + np.eye(nuisance.shape[1]), cross.T
    )
    return (adjusted + adjusted.T) / 2.0


def weak_vector(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return vectors[:, int(np.argmin(values))]


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float) / np.linalg.norm(a)
    bb = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(
        np.degrees(np.arccos(np.clip(abs(float(aa @ bb)), -1.0, 1.0)))
    )


def discrepancy_basis(
    baseline: e3.Baseline,
    scenario: Scenario,
    generic: np.ndarray,
    pooled_weak: np.ndarray,
) -> tuple[np.ndarray, float]:
    columns = [generic]
    raw_rms = float("nan")
    if scenario.aligned_sigma > 0.0:
        zeros = np.zeros(len(ALL_NAMES), dtype=float)
        target, _, target_raw = pressure_jacobians(
            baseline, baseline.theta, zeros, scenario.fabric_mode
        )
        raw_alignment = target_raw @ pooled_weak
        raw_rms = float(np.sqrt(np.mean(raw_alignment**2)))
        aligned = scenario.aligned_sigma * (target @ pooled_weak) / raw_rms
        columns.append(aligned[:, None])
    pressure = np.column_stack(columns)
    return np.vstack(
        [np.zeros((len(baseline.target_jacobian), pressure.shape[1])), pressure]
    ), raw_rms


def low_rank_sqrt_precision(discrepancy: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return r -> (I + D D^T)^(-1/2) r using the thin SVD of D.

    This is algebraically identical to profiling Gaussian linear discrepancy
    coefficients with independent unit-normal priors, up to constants that do
    not depend on the target parameters.
    """

    if discrepancy.shape[1] == 0:
        return lambda residual: np.asarray(residual, dtype=float)
    u, singular, _ = np.linalg.svd(discrepancy, full_matrices=False)
    coefficient = 1.0 - 1.0 / np.sqrt(1.0 + singular**2)

    def apply(residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=float)
        return residual - u @ (coefficient * (u.T @ residual))

    return apply


def map_profile(
    baseline: e3.Baseline,
    scenario: Scenario | None,
    v_values: np.ndarray,
    discrepancy: np.ndarray | None,
) -> pd.DataFrame:
    """Profile ln(Cn) and nuisances at each fixed Vcem.

    For the static profile, nine physical nuisances are optimized.  For a
    combined profile, all 16 physical nuisances are optimized non-linearly;
    generic and target-aligned discrepancy coefficients are profiled exactly
    through the low-rank Gaussian identity above.
    """

    static_reference = static_vector(
        baseline, baseline.theta, np.zeros(len(STATIC_NAMES))
    )
    static_sigma = e1.stacked_sigma(baseline.pooled, baseline.scale)
    if scenario is None:
        n_z = len(STATIC_NAMES)
        pressure_reference = None
        apply_precision = lambda residual: residual
        profile_name = "static"
    else:
        n_z = len(ALL_NAMES)
        pressure_reference = pressure_vector(
            baseline,
            baseline.theta,
            np.zeros(n_z),
            scenario.fabric_mode,
        )
        if discrepancy is None:
            raise ValueError("combined profile requires discrepancy matrix")
        apply_precision = low_rank_sqrt_precision(discrepancy)
        profile_name = scenario.name

    def residual(vcem: float, x: np.ndarray) -> np.ndarray:
        theta = np.array([float(vcem), float(x[0])])
        z = np.asarray(x[1:], dtype=float)
        static_residual = (
            static_vector(baseline, theta, z[: len(STATIC_NAMES)])
            - static_reference
        ) / static_sigma
        if not np.all(np.isfinite(static_residual)):
            n_data = len(static_residual)
            if scenario is not None:
                n_data += len(pressure_reference)
            return np.r_[np.full(n_data, 1.0e6), z]
        if scenario is None:
            data_residual = static_residual
        else:
            pressure_prediction = pressure_vector(
                baseline, theta, z, scenario.fabric_mode
            )
            if not np.all(np.isfinite(pressure_prediction)):
                return np.r_[
                    np.full(len(static_residual) + len(pressure_reference), 1.0e6),
                    z,
                ]
            pressure_residual = whiten_pressure_vector(
                pressure_prediction - pressure_reference, len(baseline.pooled)
            )
            data_residual = apply_precision(
                np.r_[static_residual, pressure_residual]
            )
        return np.r_[data_residual, z]

    v_values = np.sort(np.unique(np.r_[v_values, float(baseline.theta[0])]))
    truth_index = int(np.argmin(np.abs(v_values - baseline.theta[0])))
    x_truth = np.r_[baseline.theta[1], np.zeros(n_z)]
    results: dict[float, tuple[np.ndarray, float, bool, int, float, int]] = {
        float(v_values[truth_index]): (x_truth, 0.0, True, 0, 0.0, 0)
    }
    lower = list(v_values[:truth_index][::-1])
    upper = list(v_values[truth_index + 1 :])
    lower_bound = np.r_[np.log(3.0), np.full(n_z, -4.0)]
    upper_bound = np.r_[np.log(18.0), np.full(n_z, 4.0)]
    for direction in (lower, upper):
        x_start = x_truth.copy()
        for vcem in direction:
            fit = least_squares(
                lambda x: residual(float(vcem), x),
                x_start,
                bounds=(lower_bound, upper_bound),
                max_nfev=900,
                xtol=2.0e-9,
                ftol=2.0e-9,
                gtol=2.0e-9,
            )
            value = float(fit.fun @ fit.fun)
            active = int(np.count_nonzero(fit.active_mask))
            results[float(vcem)] = (
                fit.x.copy(),
                value,
                bool(fit.success),
                int(fit.nfev),
                float(fit.optimality),
                active,
            )
            x_start = fit.x.copy()

    rows: list[dict[str, float | str | bool | int]] = []
    for vcem in v_values:
        x, objective, success, nfev, optimality, active = results[float(vcem)]
        row: dict[str, float | str | bool | int] = {
            "profile": profile_name,
            "Vcem_fraction": float(vcem),
            "Vcem_percent": float(100.0 * vcem),
            "lnCn_MAP": float(x[0]),
            "Cn_MAP": float(np.exp(x[0])),
            "objective_MAP": objective,
            "nuisance_prior_norm": float(np.linalg.norm(x[1:])),
            "optimizer_success": success,
            "optimizer_nfev": nfev,
            "optimizer_optimality": optimality,
            "active_bounds": active,
        }
        for name, value in zip(
            STATIC_NAMES if scenario is None else ALL_NAMES, x[1:]
        ):
            row[f"MAP_{name}_sigma"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def convex_weight_diagnostics(
    baseline: e3.Baseline,
    theta: np.ndarray,
    z_all: np.ndarray,
    fabric_mode: str,
) -> dict[str, float | bool]:
    weights = pm.generalized_bounding_weights(
        baseline.pooled,
        theta,
        nuisance=nuisance_dict(ALL_NAMES, z_all),
        soft_lncn_reference=float(baseline.theta[1]),
        stiff_lncn_reference=float(baseline.theta[1]),
        soft_phic_reference=float(rc.PHIC_PACK),
        **e3.FABRIC_CONFIGS[fabric_mode],
    )
    values = weights[["W_K", "W_G"]].to_numpy(dtype=float)
    return {
        "bounding_weight_min": float(np.nanmin(values)),
        "bounding_weight_max": float(np.nanmax(values)),
        "convex_weight_valid": bool(
            np.all(np.isfinite(values))
            and np.all(values >= 0.0)
            and np.all(values <= 1.0)
        ),
    }


def pointwise_information(
    baseline: e3.Baseline,
    scenario: Scenario,
    profile: pd.DataFrame,
    discrepancy: np.ndarray,
    pooled_static_weak: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for item in profile.itertuples(index=False):
        theta = np.array([item.Vcem_fraction, item.lnCn_MAP], dtype=float)
        z_all = np.array(
            [getattr(item, f"MAP_{name}_sigma") for name in ALL_NAMES],
            dtype=float,
        )
        static_target, static_nuisance = static_jacobians(
            baseline, theta, z_all[: len(STATIC_NAMES)]
        )
        pressure_target, pressure_nuisance, _ = pressure_jacobians(
            baseline, theta, z_all, scenario.fabric_mode
        )
        static_adjusted = schur(static_target, static_nuisance)

        base_nuisance = np.column_stack(
            [
                static_nuisance,
                np.zeros(
                    (
                        len(static_nuisance),
                        len(ALL_NAMES) - len(STATIC_NAMES),
                    )
                ),
            ]
        )
        target = np.vstack([static_target, pressure_target])
        nuisance_physical = np.vstack([base_nuisance, pressure_nuisance])
        nuisance = np.column_stack([nuisance_physical, discrepancy])
        combined_adjusted = schur(target, nuisance)

        static_values = np.linalg.eigvalsh(static_adjusted)
        combined_values = np.linalg.eigvalsh(combined_adjusted)
        static_weak = weak_vector(static_adjusted)
        combined_weak = weak_vector(combined_adjusted)
        diagnostics = convex_weight_diagnostics(
            baseline, theta, z_all, scenario.fabric_mode
        )
        rows.append(
            {
                "scenario": scenario.name,
                "Vcem_fraction": float(item.Vcem_fraction),
                "Vcem_percent": float(item.Vcem_percent),
                "lnCn_MAP": float(item.lnCn_MAP),
                "Cn_MAP": float(item.Cn_MAP),
                "objective_MAP": float(item.objective_MAP),
                "static_lambda_min": float(static_values[0]),
                "static_lambda_max": float(static_values[-1]),
                "combined_lambda_min": float(combined_values[0]),
                "combined_lambda_max": float(combined_values[-1]),
                "lambda_min_gain": float(combined_values[0] / static_values[0]),
                "static_spectral_ratio": float(static_values[0] / static_values[-1]),
                "combined_spectral_ratio": float(
                    combined_values[0] / combined_values[-1]
                ),
                "weak_rotation_combined_vs_static_deg": angle_degrees(
                    combined_weak, static_weak
                ),
                "static_weak_rotation_vs_pooled_deg": angle_degrees(
                    static_weak, pooled_static_weak
                ),
                "combined_weak_rotation_vs_pooled_deg": angle_degrees(
                    combined_weak, pooled_static_weak
                ),
                **diagnostics,
            }
        )
    return pd.DataFrame(rows)


def interpolated_width(
    profile: pd.DataFrame, threshold: float
) -> dict[str, float | bool]:
    """Estimate the supported Vcem interval using linear crossing interpolation."""

    table = profile.sort_values("Vcem_fraction")
    x = table.Vcem_fraction.to_numpy(dtype=float)
    y = table.objective_MAP.to_numpy(dtype=float)
    truth = int(np.argmin(y))

    def crossing(indices: list[int], boundary: float) -> tuple[float, bool]:
        previous = truth
        for current in indices:
            if y[current] > threshold and y[previous] <= threshold:
                fraction = (threshold - y[previous]) / (y[current] - y[previous])
                return float(x[previous] + fraction * (x[current] - x[previous])), False
            previous = current
        return boundary, True

    lower, lower_censored = crossing(list(range(truth - 1, -1, -1)), float(x[0]))
    upper, upper_censored = crossing(
        list(range(truth + 1, len(x))), float(x[-1])
    )
    return {
        "threshold": float(threshold),
        "Vcem_lower_fraction": lower,
        "Vcem_upper_fraction": upper,
        "Vcem_width_percentage_points": float(100.0 * (upper - lower)),
        "lower_censored": lower_censored,
        "upper_censored": upper_censored,
    }


def supported_map_span(
    profile: pd.DataFrame, threshold: float
) -> dict[str, float | int]:
    supported = profile[profile.objective_MAP <= threshold]
    return {
        "MAP_Cn_min": float(supported.Cn_MAP.min()),
        "MAP_Cn_max": float(supported.Cn_MAP.max()),
        "MAP_Cn_span": float(supported.Cn_MAP.max() - supported.Cn_MAP.min()),
        "grid_points": int(len(supported)),
    }


def summarize_information(
    info: pd.DataFrame,
    scenario: Scenario,
    truth_vcem: float,
) -> dict:
    subset = info[info.scenario == scenario.name].copy()
    valid = subset[subset.convex_weight_valid].copy()
    supported = valid[valid.objective_MAP <= 2.30].copy()
    if supported.empty:
        supported = valid
    truth_row = subset.iloc[int(np.argmin(np.abs(subset.Vcem_fraction - truth_vcem)))]

    def distribution(table: pd.DataFrame, field: str) -> dict[str, float | int]:
        values = table[field].to_numpy(dtype=float)
        return {
            "minimum": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(np.max(values)),
            "n": int(len(values)),
        }

    return {
        "label": scenario.label,
        "headline_gain_from_E3": scenario.local_reference_gain,
        "recomputed_gain_at_pooled_point": float(truth_row.lambda_min_gain),
        "relative_reproduction_error": float(
            truth_row.lambda_min_gain / scenario.local_reference_gain - 1.0
        ),
        "pointwise_gain_on_convex_DeltaPhi_le_2p30_profile": distribution(
            supported, "lambda_min_gain"
        ),
        "weak_rotation_combined_vs_static_deg": distribution(
            supported, "weak_rotation_combined_vs_static_deg"
        ),
        "valid_convex_points": int(len(valid)),
        "supported_convex_points": int(len(supported)),
    }


def plot_results(
    baseline: e3.Baseline,
    profiles: pd.DataFrame,
    information: pd.DataFrame,
) -> None:
    colors = {
        "static": "#252525",
        "shared_generic": "#C94C4C",
        "expanded_generic": "#2A9D8F",
        "expanded_target_aligned": "#3478A6",
    }
    labels = {"static": "Static only", **{s.name: s.label for s in SCENARIOS}}
    refined_profile_path = HERE / "results" / "shared_refined_supported_profile.csv"
    refined_information_path = HERE / "results" / "shared_refined_pointwise_information.csv"
    crossings_path = HERE / "results" / "shared_dense_crossings.csv"
    refined_profile = (
        pd.read_csv(refined_profile_path) if refined_profile_path.exists() else None
    )
    refined_information = (
        pd.read_csv(refined_information_path)
        if refined_information_path.exists()
        else None
    )
    dense_crossings = pd.read_csv(crossings_path) if crossings_path.exists() else None
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)

    ax = axes[0, 0]
    for name, group in profiles.groupby("profile", sort=False):
        if name == "static":
            ax.plot(
                group.Vcem_percent,
                group.Cn_MAP,
                lw=2.0,
                color=colors[name],
                label=labels[name],
            )
        else:
            validity = information[information.scenario == name].set_index(
                "Vcem_fraction"
            ).convex_weight_valid
            valid = group.Vcem_fraction.map(validity).to_numpy(dtype=bool)
            ax.plot(
                group.Vcem_percent[valid],
                group.Cn_MAP[valid],
                lw=2.0,
                color=colors[name],
                label=labels[name],
            )
            if np.any(~valid):
                ax.plot(
                    group.Vcem_percent[~valid],
                    group.Cn_MAP[~valid],
                    lw=1.1,
                    ls="--",
                    alpha=0.45,
                    color=colors[name],
                )
    ax.scatter(
        [100.0 * baseline.theta[0]],
        [np.exp(baseline.theta[1])],
        marker="*",
        s=110,
        c="#E9C46A",
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
        label="Pooled point",
    )
    if refined_profile is not None:
        ax.plot(
            refined_profile.Vcem_percent,
            refined_profile.Cn_MAP,
            lw=2.8,
            color=colors["shared_generic"],
            zorder=4,
        )
    ax.set_xlabel(r"$V_{\rm cem}$ (percentage points)")
    ax.set_ylabel(r"MAP $C_n$")
    ax.set_title("(a) Fully non-linear nuisance-MAP ridges", loc="left")
    ax.legend(fontsize=7.7, frameon=False)

    ax = axes[0, 1]
    for name, group in profiles.groupby("profile", sort=False):
        if name == "static":
            valid = np.ones(len(group), dtype=bool)
        else:
            validity = information[information.scenario == name].set_index(
                "Vcem_fraction"
            ).convex_weight_valid
            valid = group.Vcem_fraction.map(validity).to_numpy(dtype=bool)
        ax.semilogy(
            group.Vcem_percent[valid],
            np.maximum(group.objective_MAP[valid], 1.0e-8),
            lw=2.0,
            color=colors[name],
            label=labels[name],
        )
        if np.any(~valid):
            ax.semilogy(
                group.Vcem_percent[~valid],
                np.maximum(group.objective_MAP[~valid], 1.0e-8),
                lw=1.1,
                ls="--",
                alpha=0.45,
                color=colors[name],
            )
    ax.axhline(2.30, color="#777777", ls="--", lw=1.0, label=r"$\Delta\Phi=2.30$")
    ax.axhline(5.99, color="#AAAAAA", ls=":", lw=1.0, label=r"$\Delta\Phi=5.99$")
    if refined_profile is not None:
        ax.semilogy(
            refined_profile.Vcem_percent,
            np.maximum(refined_profile.objective_MAP, 1.0e-8),
            lw=2.8,
            color=colors["shared_generic"],
            zorder=4,
        )
    if dense_crossings is not None:
        ax.scatter(
            dense_crossings.Vcem_percent,
            dense_crossings.threshold,
            s=34,
            marker="o",
            facecolor="white",
            edgecolor=colors["shared_generic"],
            linewidth=1.3,
            zorder=5,
        )
        width = dense_crossings[np.isclose(dense_crossings.threshold, 2.30)]
        if len(width) == 2:
            value = float(width.Vcem_percent.max() - width.Vcem_percent.min())
            ax.text(
                0.98,
                0.96,
                rf"shared $\Delta\Phi=2.30$: {value:.3f} pp",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.2,
                color=colors["shared_generic"],
            )
    ax.set_xlabel(r"$V_{\rm cem}$ (percentage points)")
    ax.set_ylabel(r"Profile objective $\Delta\Phi$")
    ax.set_title("(b) Global profile shape (not Fisher information)", loc="left")
    ax.set_ylim(1.0e-5, max(20.0, float(profiles.objective_MAP.max()) * 1.15))

    ax = axes[1, 0]
    for scenario in SCENARIOS:
        group = information[information.scenario == scenario.name]
        valid = group.convex_weight_valid.to_numpy(dtype=bool)
        if scenario.name == "shared_generic" and refined_information is not None:
            lower = float(refined_information.Vcem_fraction.min())
            upper = float(refined_information.Vcem_fraction.max())
            outside = valid & (
                (group.Vcem_fraction < lower) | (group.Vcem_fraction > upper)
            )
            ax.semilogy(
                group.Vcem_percent[outside],
                group.lambda_min_gain[outside],
                lw=1.0,
                ls="--",
                alpha=0.35,
                color=colors[scenario.name],
            )
            ax.semilogy(
                refined_information.Vcem_percent,
                refined_information.lambda_min_gain,
                lw=2.2,
                color=colors[scenario.name],
                label=scenario.label,
            )
            continue
        ax.semilogy(
            group.Vcem_percent[valid],
            group.lambda_min_gain[valid],
            lw=2.0,
            color=colors[scenario.name],
            label=scenario.label,
        )
        if np.any(~valid):
            ax.semilogy(
                group.Vcem_percent[~valid],
                group.lambda_min_gain[~valid],
                lw=1.1,
                ls="--",
                alpha=0.45,
                color=colors[scenario.name],
            )
    ax.axhline(1.0, color="#777777", lw=0.9, ls=":")
    ax.set_xlabel(r"$V_{\rm cem}$ (percentage points)")
    ax.set_ylabel(r"Pointwise $\lambda_{\min}$ gain")
    ax.set_title("(c) Efficient-information gain at each MAP point", loc="left")
    ax.legend(fontsize=7.4, frameon=False)

    ax = axes[1, 1]
    for scenario in SCENARIOS:
        group = information[information.scenario == scenario.name]
        valid = group.convex_weight_valid.to_numpy(dtype=bool)
        if scenario.name == "shared_generic" and refined_information is not None:
            ax.plot(
                refined_information.Vcem_percent,
                refined_information.weak_rotation_combined_vs_static_deg,
                lw=2.2,
                color=colors[scenario.name],
                label=scenario.label,
            )
            continue
        ax.plot(
            group.Vcem_percent[valid],
            group.weak_rotation_combined_vs_static_deg[valid],
            lw=2.0,
            color=colors[scenario.name],
            label=scenario.label,
        )
    ax.set_xlabel(r"$V_{\rm cem}$ (percentage points)")
    ax.set_ylabel("Weak-direction rotation (degrees)")
    ax.set_title("(d) Local rotation after nuisance adjustment", loc="left")

    fig.suptitle(
        "Non-linear robustness of constant-cement pressure-design gains",
        fontsize=14,
    )
    for ax in axes.flat:
        ax.grid(True, alpha=0.18)
    figure_root = HERE / "results" / "figures"
    png_path = figure_root / "Fig_nonlinear_ridge_robustness.png"
    pdf_path = figure_root / "Fig_nonlinear_ridge_robustness.pdf"
    png_temporary = figure_root / ".Fig_nonlinear_ridge_robustness.tmp.png"
    pdf_temporary = figure_root / ".Fig_nonlinear_ridge_robustness.tmp.pdf"
    fig.savefig(png_temporary, dpi=260)
    fig.savefig(pdf_temporary)
    plt.close(fig)
    os.replace(png_temporary, png_path)
    os.replace(pdf_temporary, pdf_path)


def main() -> None:
    ensure_directories()
    baseline = e3.load_baseline()
    v_values = np.unique(
        np.r_[np.linspace(0.001, 0.060, 31), float(baseline.theta[0])]
    )

    # Establish the pooled static weak direction and reproduce the fixed E3
    # discrepancy construction exactly before moving away from the pooled point.
    pooled_static_weak = weak_vector(baseline.gram_adjusted)
    generic_pressure = generic_discrepancy_matrix(baseline)
    discrepancy_by_scenario: dict[str, np.ndarray] = {}
    aligned_raw_rms: dict[str, float] = {}
    for scenario in SCENARIOS:
        matrix, raw_rms = discrepancy_basis(
            baseline, scenario, generic_pressure, pooled_static_weak
        )
        discrepancy_by_scenario[scenario.name] = matrix
        aligned_raw_rms[scenario.name] = raw_rms

    print("Profiling static model", flush=True)
    static_profile = map_profile(baseline, None, v_values, None)
    profiles = [static_profile]
    information_parts = []
    for scenario in SCENARIOS:
        print(f"Profiling {scenario.name}", flush=True)
        profile = map_profile(
            baseline,
            scenario,
            v_values,
            discrepancy_by_scenario[scenario.name],
        )
        profiles.append(profile)
        print(f"Recomputing pointwise information for {scenario.name}", flush=True)
        information_parts.append(
            pointwise_information(
                baseline,
                scenario,
                profile,
                discrepancy_by_scenario[scenario.name],
                pooled_static_weak,
            )
        )

    profile_table = pd.concat(profiles, ignore_index=True)
    information = pd.concat(information_parts, ignore_index=True)
    profile_table.to_csv(HERE / "results" / "nonlinear_MAP_profiles.csv", index=False)
    information.to_csv(HERE / "results" / "pointwise_efficient_information.csv", index=False)

    summary = {
        "method": {
            "target_profile": (
                "fully nonlinear forward model; lnCn and physical nuisances optimized "
                "at each fixed Vcem; linear discrepancy coefficients profiled exactly"
            ),
            "pointwise_information": (
                "Gauss-Newton efficient information (Schur complement) recalculated "
                "at each nuisance-MAP point; no averaging or integration of matrices"
            ),
            "design_pressures_mpa": list(DESIGN),
            "profile_grid_points": int(len(v_values)),
            "Vcem_grid_fraction": [float(np.min(v_values)), float(np.max(v_values))],
            "finite_difference_step_standardized": FD_STEP,
        },
        "pooled_operating_point": {
            "Vcem_fraction": float(baseline.theta[0]),
            "Cn": float(np.exp(baseline.theta[1])),
        },
        "static_profile_widths": {
            str(threshold): interpolated_width(static_profile, threshold)
            for threshold in (2.30, 5.99)
        },
        "static_MAP_Cn_spans": {
            str(threshold): supported_map_span(static_profile, threshold)
            for threshold in (2.30, 5.99)
        },
        "scenarios": {},
        "target_aligned_raw_rms": {
            key: (float(value) if np.isfinite(value) else None)
            for key, value in aligned_raw_rms.items()
        },
    }
    for scenario in SCENARIOS:
        profile = profile_table[profile_table.profile == scenario.name]
        summary["scenarios"][scenario.name] = {
            **summarize_information(
                information, scenario, float(baseline.theta[0])
            ),
            "profile_widths": {
                str(threshold): interpolated_width(profile, threshold)
                for threshold in (2.30, 5.99)
            },
            "MAP_Cn_spans": {
                str(threshold): supported_map_span(profile, threshold)
                for threshold in (2.30, 5.99)
            },
            "optimizer": {
                "all_success": bool(profile.optimizer_success.all()),
                "maximum_optimality": float(profile.optimizer_optimality.max()),
                "maximum_nfev": int(profile.optimizer_nfev.max()),
                "points_with_active_bounds": int(np.count_nonzero(profile.active_bounds)),
            },
        }

    (HERE / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(baseline, profile_table, information)
    print(json.dumps(summary["scenarios"], indent=2), flush=True)


if __name__ == "__main__":
    main()
