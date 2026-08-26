from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(ROOT / "src"))
import e3_model as pm


e1 = pm.e1
rc = pm.rc

MODEL = e1.MODEL
NAMES = e1.NAMES
PRIMARY_STATE_LOG_SIGMA = 0.005
PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA = 0.010
PRIMARY_TRAJECTORY_DISCREPANCY_BASIS = (
    "intercept",
    "orthonormalized_porosity_trend",
    "orthonormalized_clay_fraction_trend",
)
PRIMARY_FABRIC_MODE = "expanded_nuisance"
FABRIC_CONFIGS = {
    "shared": {
        "soft_cn_mode": "shared",
        "stiff_cn_mode": "shared",
        "soft_phic_mode": "shared",
    },
    "fixed": {
        "soft_cn_mode": "fixed",
        "stiff_cn_mode": "shared",
        "soft_phic_mode": "shared",
    },
    "nuisance": {
        "soft_cn_mode": "nuisance",
        "stiff_cn_mode": "shared",
        "soft_phic_mode": "shared",
    },
    "expanded_nuisance": {
        "soft_cn_mode": "nuisance",
        "stiff_cn_mode": "nuisance",
        "soft_phic_mode": "nuisance",
    },
}
PRESSURE_CANDIDATES = [
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    17.5,
    22.5,
    25.0,
    27.5,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
]

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


@dataclass
class Baseline:
    wells: dict[str, pd.DataFrame]
    pooled: pd.DataFrame
    theta: np.ndarray
    scale: dict[str, float]
    target_jacobian: np.ndarray
    nuisance_jacobian: np.ndarray
    nuisance_names: list[str]
    gram_raw: np.ndarray
    gram_adjusted: np.ndarray
    metrics: dict[str, float]
    coordinate: dict[str, float]
    e2_summary: dict


def _mkdirs() -> None:
    for path in [
        ROOT / "results" / "tables",
        ROOT / "results" / "figures",
        ROOT / "results" / "verification",
        ROOT / ".mplconfig",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_baseline() -> Baseline:
    wells, pooled, _ = e1.load_data()
    calibrations = {name: rc.calibrate(df, MODEL) for name, df in wells.items()}
    scale = e1.transfer_aware_scale(e1.transfer_discrepancy(wells, calibrations))
    theta = rc.calibrate(pooled, MODEL)
    metrics, jt, jn, gram, gram_adjusted, nuisance_names = e1.metric_summary(
        pooled, theta, scale
    )
    coordinate = e1.metric_contact_hs_decomposition(pooled, theta, scale)
    e2_summary = json.loads((ROOT / "reference" / "E2_summary.json").read_text())
    return Baseline(
        wells=wells,
        pooled=pooled,
        theta=theta,
        scale=scale,
        target_jacobian=jt,
        nuisance_jacobian=jn,
        nuisance_names=nuisance_names,
        gram_raw=gram,
        gram_adjusted=gram_adjusted,
        metrics=metrics,
        coordinate=coordinate,
        e2_summary=e2_summary,
    )


def schur_geometry(
    target: np.ndarray,
    nuisance: np.ndarray | None,
    prior_precision_factor: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    gram = target.T @ target
    if nuisance is None or nuisance.shape[1] == 0:
        return (gram + gram.T) / 2.0, (gram + gram.T) / 2.0
    cross = target.T @ nuisance
    nuisance_gram = nuisance.T @ nuisance
    if np.isscalar(prior_precision_factor):
        prior_precision = float(prior_precision_factor) * np.eye(nuisance.shape[1])
    else:
        factors = np.asarray(prior_precision_factor, dtype=float)
        if factors.shape != (nuisance.shape[1],):
            raise ValueError("prior precision vector has incompatible shape")
        prior_precision = np.diag(factors)
    adjusted = gram - cross @ np.linalg.solve(
        nuisance_gram + prior_precision,
        cross.T,
    )
    return (gram + gram.T) / 2.0, (adjusted + adjusted.T) / 2.0


def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    return values, vectors


def geometry_metrics(
    matrix: np.ndarray,
    baseline_matrix: np.ndarray,
) -> dict[str, float]:
    values, vectors = eigensystem(matrix)
    baseline_values, baseline_vectors = eigensystem(baseline_matrix)
    strong_angle = pm.angle_degrees(vectors[:, -1], baseline_vectors[:, -1])
    weak_angle = pm.angle_degrees(vectors[:, 0], baseline_vectors[:, 0])
    return {
        "lambda_min": float(values[0]),
        "lambda_max": float(values[-1]),
        "spectral_ratio": float(values[0] / values[-1]),
        "condition_number": float(math.sqrt(values[-1] / values[0])),
        "lambda_min_gain": float(values[0] / baseline_values[0]),
        "worst_sd_reduction": float(math.sqrt(values[0] / baseline_values[0])),
        "determinant_gain": float(np.linalg.det(matrix) / np.linalg.det(baseline_matrix)),
        "strong_direction_rotation_deg": strong_angle,
        "weak_direction_rotation_deg": weak_angle,
    }


def parameter_uncertainty_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Local Cramer--Rao diagnostics in nominal parameter units."""

    covariance_scaled = np.linalg.inv((matrix + matrix.T) / 2.0)
    scales = np.asarray(rc.PARAM_SCALES[MODEL], dtype=float)
    covariance = np.diag(scales) @ covariance_scaled @ np.diag(scales)
    sd_vcem = float(math.sqrt(covariance[0, 0]))
    sd_lncn = float(math.sqrt(covariance[1, 1]))
    correlation = float(
        covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1])
    )
    return {
        "sd_Vcem_fraction": sd_vcem,
        "sd_Vcem_percentage_points": 100.0 * sd_vcem,
        "sd_lnCn": sd_lncn,
        "multiplicative_Cn_one_sigma": float(math.exp(sd_lncn)),
        "Vcem_lnCn_correlation": correlation,
    }


def selected_design_geometry(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    design: tuple[float, ...],
    fabric_mode: str = PRIMARY_FABRIC_MODE,
) -> np.ndarray:
    target_all, nuisance_all, _ = jacobians[fabric_mode]
    target, nuisance = pressure_subset(
        target_all,
        nuisance_all,
        PRESSURE_CANDIDATES,
        design,
        len(baseline.pooled),
        PRIMARY_STATE_LOG_SIGMA,
        shared_reference=True,
        trajectory_lengths=(len(baseline.wells["19A"]), len(baseline.wells["BT2"])),
        trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
        trajectory_discrepancy_features=model_discrepancy_features(baseline),
    )
    _, adjusted = combined_geometry(
        baseline, target, nuisance, "full_pressure_nuisance"
    )
    return adjusted


def expanded_baseline_nuisance(
    baseline: Baseline,
    include_pressure_nuisances: bool,
) -> np.ndarray:
    if not include_pressure_nuisances:
        return baseline.nuisance_jacobian.copy()
    extra = len(pm.pressure_nuisance_names())
    return np.c_[
        baseline.nuisance_jacobian,
        np.zeros((len(baseline.nuisance_jacobian), extra)),
    ]


def whiten_pressure_blocks(
    target: np.ndarray,
    nuisance: np.ndarray,
    n_pressures: int,
    n_samples: int,
    state_log_sigma: float,
    shared_reference: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten pressure differences with their shared-reference covariance.

    If every log-velocity state has independent error sigma, differences to a
    common reference have covariance sigma^2(I + 11^T), hence correlation 0.5.
    """

    if n_pressures < 1:
        return target.copy(), nuisance.copy()
    if shared_reference:
        covariance = state_log_sigma**2 * (
            np.eye(n_pressures) + np.ones((n_pressures, n_pressures))
        )
    else:
        covariance = 2.0 * state_log_sigma**2 * np.eye(n_pressures)
    chol = np.linalg.cholesky(covariance)

    def transform(matrix: np.ndarray) -> np.ndarray:
        reshaped = matrix.reshape(n_pressures, 2, n_samples, matrix.shape[1])
        output = np.empty_like(reshaped)
        for prop in range(2):
            for sample in range(n_samples):
                output[:, prop, sample, :] = np.linalg.solve(
                    chol, reshaped[:, prop, sample, :]
                )
        return output.reshape(matrix.shape)

    return transform(target), transform(nuisance)


def model_discrepancy_features(
    baseline: Baseline, n_features: int = 3
) -> np.ndarray:
    """Low-order basis for state/trajectory-correlated model discrepancy.

    A separate coefficient is used for every pressure, elastic observable, and
    trajectory.  Porosity and clay-fraction trends prevent the prospective
    gain from relying only on smooth sample-to-sample model mismatch.
    """

    if n_features not in (1, 2, 3):
        raise ValueError("n_features must be 1, 2, or 3")
    output = np.zeros((len(baseline.pooled), n_features))
    offset = 0
    for name in baseline.wells:
        part = baseline.wells[name]
        phi = part.phi.to_numpy(dtype=float)
        vsh = part.vsh.to_numpy(dtype=float)
        raw = np.column_stack(
            [
                np.ones(len(part)),
                phi - np.mean(phi),
                vsh - np.mean(vsh),
            ]
        )[:, :n_features]
        q, _ = np.linalg.qr(raw)
        # The sum of the basis-induced pointwise variances has unit mean, so
        # ``trajectory_discrepancy_sigma`` is a total RMS amplitude rather than
        # an independent amplitude assigned to every basis coefficient.
        normalized = q * math.sqrt(len(part) / n_features)
        output[offset : offset + len(part)] = normalized
        offset += len(part)
    return output


def pressure_subset(
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    pressures_all: list[float],
    selected: Iterable[float],
    n_samples: int,
    state_log_sigma: float,
    shared_reference: bool = True,
    trajectory_lengths: tuple[int, ...] | None = None,
    trajectory_discrepancy_sigma: float = 0.0,
    trajectory_discrepancy_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = list(map(float, selected))
    rows = pm.pressure_row_indices(pressures_all, selected, n_samples)
    target = target_all[rows]
    nuisance = nuisance_all[rows]
    if trajectory_discrepancy_sigma > 0.0:
        if trajectory_lengths is None or sum(trajectory_lengths) != n_samples:
            raise ValueError("valid trajectory_lengths are required for discrepancy terms")
        if trajectory_discrepancy_features is None:
            trajectory_discrepancy_features = np.ones((n_samples, 1))
        trajectory_discrepancy_features = np.asarray(
            trajectory_discrepancy_features, dtype=float
        )
        if trajectory_discrepancy_features.shape[0] != n_samples:
            raise ValueError("discrepancy features must have one row per sample")
        n_states = len(selected)
        n_features = trajectory_discrepancy_features.shape[1]
        n_columns = n_states * 2 * len(trajectory_lengths) * n_features
        discrepancy = np.zeros((len(target), n_columns))
        offsets = np.cumsum((0,) + trajectory_lengths)
        column = 0
        for state in range(n_states):
            for prop in range(2):
                block_start = (2 * state + prop) * n_samples
                for trajectory in range(len(trajectory_lengths)):
                    sample_start = offsets[trajectory]
                    sample_stop = offsets[trajectory + 1]
                    start = block_start + sample_start
                    stop = block_start + sample_stop
                    for feature in range(n_features):
                        discrepancy[start:stop, column] = (
                            trajectory_discrepancy_sigma
                            * trajectory_discrepancy_features[
                                sample_start:sample_stop, feature
                            ]
                        )
                        column += 1
        nuisance = np.c_[nuisance, discrepancy]
    return whiten_pressure_blocks(
        target,
        nuisance,
        len(selected),
        n_samples,
        state_log_sigma,
        shared_reference,
    )


def combined_geometry(
    baseline: Baseline,
    pressure_target: np.ndarray,
    pressure_nuisance: np.ndarray,
    adjustment: str,
    prior_precision_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.vstack([baseline.target_jacobian, pressure_target])
    if adjustment == "raw":
        return schur_geometry(target, None)
    if adjustment == "shared_static_nuisance":
        base_nuisance = expanded_baseline_nuisance(baseline, False)
        pressure_nuisance_use = pressure_nuisance[:, : len(baseline.nuisance_names)]
    elif adjustment == "full_pressure_nuisance":
        base_nuisance = expanded_baseline_nuisance(baseline, True)
        if pressure_nuisance.shape[1] > base_nuisance.shape[1]:
            base_nuisance = np.c_[
                base_nuisance,
                np.zeros(
                    (
                        len(base_nuisance),
                        pressure_nuisance.shape[1] - base_nuisance.shape[1],
                    )
                ),
            ]
        pressure_nuisance_use = pressure_nuisance
    else:
        raise ValueError(adjustment)
    nuisance = np.vstack([base_nuisance, pressure_nuisance_use])
    if adjustment == "full_pressure_nuisance" and prior_precision_factor != 1.0:
        prior_precision = np.ones(nuisance.shape[1])
        start = len(baseline.nuisance_names)
        stop = start + len(pm.pressure_nuisance_names())
        prior_precision[start:stop] = prior_precision_factor
        return schur_geometry(target, nuisance, prior_precision)
    return schur_geometry(target, nuisance)


def no_go_repetition_table(baseline: Baseline) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    base_values, base_vectors = eigensystem(baseline.gram_adjusted)
    _, base_raw_vectors = eigensystem(baseline.gram_raw)
    for n_states in [1, 2, 3, 5, 10, 20]:
        target = np.vstack([baseline.target_jacobian] * n_states)
        nuisance = np.vstack([baseline.nuisance_jacobian] * n_states)
        raw, adjusted = schur_geometry(target, nuisance)
        values, vectors = eigensystem(adjusted)
        _, raw_vectors = eigensystem(raw)
        rows.append(
            {
                "n_identical_static_states": n_states,
                "lambda_min": float(values[0]),
                "lambda_max": float(values[-1]),
                "lambda_min_gain": float(values[0] / base_values[0]),
                "spectral_ratio": float(values[0] / values[-1]),
                "strong_direction_rotation_deg": pm.angle_degrees(
                    vectors[:, -1], base_vectors[:, -1]
                ),
                "raw_strong_direction_rotation_deg": pm.angle_degrees(
                    raw_vectors[:, -1], base_raw_vectors[:, -1]
                ),
                "target_rank": int(np.linalg.matrix_rank(target)),
                "new_sensitivity_direction": 0,
            }
        )
    return pd.DataFrame(rows)


def multi_fluid_control_table(baseline: Baseline) -> pd.DataFrame:
    specifications = [
        ("baseline_only", []),
        ("plus_brine", [1.0]),
        ("plus_hydrocarbon", [0.20]),
        ("plus_brine_and_hydrocarbon", [1.0, 0.20]),
        ("plus_three_fluid_states", [1.0, 0.60, 0.20]),
    ]
    base_values, _ = eigensystem(baseline.gram_adjusted)
    rows = []
    for label, saturations in specifications:
        target_parts = [baseline.target_jacobian]
        nuisance_parts = [baseline.nuisance_jacobian]
        for saturation in saturations:
            state = baseline.pooled.copy()
            state["sw"] = saturation
            _, jt, jn, _, _, _ = e1.metric_summary(
                state, baseline.theta, baseline.scale
            )
            target_parts.append(jt)
            nuisance_parts.append(jn)
        _, adjusted = schur_geometry(
            np.vstack(target_parts), np.vstack(nuisance_parts)
        )
        metrics = geometry_metrics(adjusted, baseline.gram_adjusted)
        rows.append(
            {
                "design": label,
                "n_additional_states": len(saturations),
                "lambda_min": metrics["lambda_min"],
                "lambda_min_gain": metrics["lambda_min"] / base_values[0],
                "spectral_ratio": metrics["spectral_ratio"],
                "strong_direction_rotation_deg": metrics[
                    "strong_direction_rotation_deg"
                ],
            }
        )
    return pd.DataFrame(rows)


def pressure_curve_table(baseline: Baseline) -> pd.DataFrame:
    pressures = np.unique(
        np.r_[np.linspace(5.0, 60.0, 23), pm.REFERENCE_PRESSURE_MPA]
    )
    representative = baseline.pooled.iloc[[
        int(np.argmin(np.abs(baseline.pooled.phi.to_numpy() - baseline.pooled.phi.median())))
    ]].copy()
    frozen = rc.forward(representative, MODEL, baseline.theta)[0]
    rows: list[dict[str, float | str]] = []
    for mode in ["frozen_constant_cement", "pressure_extension_shared_Cn", "pressure_extension_free_soft_Cn"]:
        for pressure in pressures:
            if mode == "frozen_constant_cement":
                state = frozen
            else:
                soft_mode = "shared" if mode.endswith("shared_Cn") else "nuisance"
                state = pm.pressure_extended_forward(
                    representative,
                    baseline.theta,
                    float(pressure),
                    soft_cn_mode=soft_mode,
                    soft_lncn_reference=float(baseline.theta[1]),
                )[0]
            rows.append(
                {
                    "model": mode,
                    "pressure_mpa": float(pressure),
                    "Vp_mps": float(state[0]),
                    "Vs_mps": float(state[1]),
                    "Vp_change_percent": float(100.0 * (state[0] / frozen[0] - 1.0)),
                    "Vs_change_percent": float(100.0 * (state[1] / frozen[1] - 1.0)),
                    "phi_representative": float(representative.phi.iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def precompute_pressure_jacobians(
    baseline: Baseline,
    pressures: list[float] = PRESSURE_CANDIDATES,
    reference_pressure_mpa: float = pm.REFERENCE_PRESSURE_MPA,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    output = {}
    for mode, configuration in FABRIC_CONFIGS.items():
        print(f"pressure Jacobian: fabric mode={mode}", flush=True)
        output[mode] = pm.pressure_differential_jacobian(
            baseline.pooled,
            baseline.theta,
            pressures,
            reference_pressure_mpa=reference_pressure_mpa,
            **configuration,
        )
    return output


def pressure_design_grid(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    pressures: list[float] = PRESSURE_CANDIDATES,
    state_sigmas: tuple[float, ...] = (0.0025, 0.005, 0.010, 0.020),
) -> pd.DataFrame:
    designs: list[tuple[float, ...]] = [(value,) for value in pressures]
    designs.extend(itertools.combinations(pressures, 2))
    rows: list[dict[str, float | str | int]] = []
    n_samples = len(baseline.pooled)
    trajectory_lengths = (len(baseline.wells["19A"]), len(baseline.wells["BT2"]))
    discrepancy_features = model_discrepancy_features(baseline)
    for mode, (target_all, nuisance_all, _) in jacobians.items():
        for sigma in state_sigmas:
            for design in designs:
                target, nuisance = pressure_subset(
                    target_all,
                    nuisance_all,
                    pressures,
                    design,
                    n_samples,
                    sigma,
                    shared_reference=True,
                    trajectory_lengths=trajectory_lengths,
                    trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
                    trajectory_discrepancy_features=discrepancy_features,
                )
                _, adjusted = combined_geometry(
                    baseline,
                    target,
                    nuisance,
                    "full_pressure_nuisance",
                )
                metrics = geometry_metrics(adjusted, baseline.gram_adjusted)
                rows.append(
                    {
                        "fabric_mode": mode,
                        "state_log_sigma": sigma,
                        "state_precision_percent": 100.0 * sigma,
                        "n_additional_pressures": len(design),
                        "pressure_1_mpa": design[0],
                        "pressure_2_mpa": design[1] if len(design) == 2 else np.nan,
                        "pressure_span_mpa": max(design + (pm.REFERENCE_PRESSURE_MPA,))
                        - min(design + (pm.REFERENCE_PRESSURE_MPA,)),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def best_designs(design_grid: pd.DataFrame) -> pd.DataFrame:
    index = (
        design_grid.groupby(
            ["fabric_mode", "state_log_sigma", "n_additional_pressures"],
            sort=True,
        )["lambda_min"]
        .idxmax()
        .to_numpy()
    )
    return design_grid.loc[index].sort_values(
        ["fabric_mode", "state_log_sigma", "n_additional_pressures"]
    ).reset_index(drop=True)


def design_from_row(row: pd.Series) -> tuple[float, ...]:
    if int(row.n_additional_pressures) == 1:
        return (float(row.pressure_1_mpa),)
    return (float(row.pressure_1_mpa), float(row.pressure_2_mpa))


def primary_design(best: pd.DataFrame) -> tuple[float, ...]:
    row = best[
        (best.fabric_mode == PRIMARY_FABRIC_MODE)
        & np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (best.n_additional_pressures == 2)
    ].iloc[0]
    return design_from_row(row)


def pressure_ablation_table(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    best: pd.DataFrame,
    pressures: list[float] = PRESSURE_CANDIDATES,
) -> pd.DataFrame:
    primary_pair = primary_design(best)
    primary_single_row = best[
        (best.fabric_mode == PRIMARY_FABRIC_MODE)
        & np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (best.n_additional_pressures == 1)
    ].iloc[0]
    primary_single = design_from_row(primary_single_row)
    design_specs = [
        ("best_single", primary_single),
        ("symmetric_10_40", (10.0, 40.0)),
        ("best_pair", primary_pair),
    ]
    rows: list[dict[str, float | str | int]] = []
    trajectory_lengths = (len(baseline.wells["19A"]), len(baseline.wells["BT2"]))
    discrepancy_features = model_discrepancy_features(baseline)
    for mode, (target_all, nuisance_all, _) in jacobians.items():
        for design_label, design in design_specs:
            for sigma in [0.0025, 0.005, 0.010, 0.020, 0.050]:
                target, nuisance = pressure_subset(
                    target_all,
                    nuisance_all,
                    pressures,
                    design,
                    len(baseline.pooled),
                    sigma,
                    shared_reference=True,
                    trajectory_lengths=trajectory_lengths,
                    trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
                    trajectory_discrepancy_features=discrepancy_features,
                )
                for adjustment in [
                    "raw",
                    "shared_static_nuisance",
                    "full_pressure_nuisance",
                ]:
                    prior_factors = (
                        [0.25, 1.0, 4.0]
                        if adjustment == "full_pressure_nuisance"
                        else [1.0]
                    )
                    for prior_precision in prior_factors:
                        _, matrix = combined_geometry(
                            baseline,
                            target,
                            nuisance,
                            adjustment,
                            prior_precision,
                        )
                        reference = (
                            baseline.gram_raw
                            if adjustment == "raw"
                            else baseline.gram_adjusted
                        )
                        metrics = geometry_metrics(matrix, reference)
                        rows.append(
                            {
                                "design": design_label,
                                "pressures_mpa": "+".join(f"{x:g}" for x in design),
                                "fabric_mode": mode,
                                "state_log_sigma": sigma,
                                "state_precision_percent": 100.0 * sigma,
                                "adjustment": adjustment,
                                "pressure_nuisance_prior_precision_factor": prior_precision,
                                "pressure_nuisance_prior_sd_factor": 1.0
                                / math.sqrt(prior_precision),
                                **metrics,
                            }
                        )
    return pd.DataFrame(rows)


def discrepancy_sensitivity_table(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    design: tuple[float, ...],
) -> pd.DataFrame:
    target_all, nuisance_all, _ = jacobians[PRIMARY_FABRIC_MODE]
    rows = []
    basis_specs = [
        ("intercept", 1),
        ("intercept_plus_porosity", 2),
        ("intercept_plus_porosity_plus_clay", 3),
    ]
    for basis_name, n_features in basis_specs:
        for discrepancy in [0.0, 0.005, 0.010, 0.020, 0.050]:
            target, nuisance = pressure_subset(
                target_all,
                nuisance_all,
                PRESSURE_CANDIDATES,
                design,
                len(baseline.pooled),
                PRIMARY_STATE_LOG_SIGMA,
                shared_reference=True,
                trajectory_lengths=(
                    len(baseline.wells["19A"]),
                    len(baseline.wells["BT2"]),
                ),
                trajectory_discrepancy_sigma=discrepancy,
                trajectory_discrepancy_features=model_discrepancy_features(
                    baseline, n_features
                ),
            )
            _, adjusted = combined_geometry(
                baseline, target, nuisance, "full_pressure_nuisance"
            )
            rows.append(
                {
                    "basis": basis_name,
                    "n_basis_terms": n_features,
                    "trajectory_discrepancy_log_sigma": discrepancy,
                    "trajectory_discrepancy_percent": 100.0 * discrepancy,
                    **geometry_metrics(adjusted, baseline.gram_adjusted),
                }
            )
    return pd.DataFrame(rows)


def reference_pressure_sensitivity_table(baseline: Baseline) -> pd.DataFrame:
    """Re-optimize the candidate design for alternative pressure anchors."""

    rows: list[dict[str, float | str | int]] = []
    n_samples = len(baseline.pooled)
    trajectory_lengths = (len(baseline.wells["19A"]), len(baseline.wells["BT2"]))
    discrepancy_features = model_discrepancy_features(baseline)
    for reference_pressure in [10.0, 20.0, float(rc.P_EFF_MPA)]:
        candidates = [
            value
            for value in PRESSURE_CANDIDATES
            if not np.isclose(value, reference_pressure)
        ]
        designs: list[tuple[float, ...]] = [(value,) for value in candidates]
        designs.extend(itertools.combinations(candidates, 2))
        for mode in ["nuisance", "expanded_nuisance"]:
            target_all, nuisance_all, _ = pm.pressure_differential_jacobian(
                baseline.pooled,
                baseline.theta,
                candidates,
                reference_pressure_mpa=reference_pressure,
                **FABRIC_CONFIGS[mode],
            )
            for size in [1, 2]:
                best_row: dict[str, float | str | int] | None = None
                for design in (item for item in designs if len(item) == size):
                    target, nuisance = pressure_subset(
                        target_all,
                        nuisance_all,
                        candidates,
                        design,
                        n_samples,
                        PRIMARY_STATE_LOG_SIGMA,
                        shared_reference=True,
                        trajectory_lengths=trajectory_lengths,
                        trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
                        trajectory_discrepancy_features=discrepancy_features,
                    )
                    _, adjusted = combined_geometry(
                        baseline, target, nuisance, "full_pressure_nuisance"
                    )
                    metrics = geometry_metrics(adjusted, baseline.gram_adjusted)
                    candidate_row: dict[str, float | str | int] = {
                        "reference_pressure_mpa": reference_pressure,
                        "fabric_mode": mode,
                        "n_additional_pressures": size,
                        "pressure_1_mpa": design[0],
                        "pressure_2_mpa": design[1] if size == 2 else np.nan,
                        "absolute_state_count": size + 1,
                        "pressure_span_mpa": max(design + (reference_pressure,))
                        - min(design + (reference_pressure,)),
                        **metrics,
                    }
                    if best_row is None or float(candidate_row["lambda_min"]) > float(
                        best_row["lambda_min"]
                    ):
                        best_row = candidate_row
                if best_row is None:
                    raise RuntimeError("no reference-pressure design candidates")
                rows.append(best_row)
    return pd.DataFrame(rows)


def target_aligned_discrepancy_table(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    design: tuple[float, ...],
) -> pd.DataFrame:
    """Stress-test model error aligned with the static weak target direction."""

    _, static_vectors = eigensystem(baseline.gram_adjusted)
    weak_direction = static_vectors[:, 0]
    rows = []
    for mode in ["nuisance", "expanded_nuisance"]:
        target_all, nuisance_all, _ = jacobians[mode]
        selected_rows = pm.pressure_row_indices(
            PRESSURE_CANDIDATES, design, len(baseline.pooled)
        )
        raw_alignment = target_all[selected_rows] @ weak_direction
        raw_rms = float(np.sqrt(np.mean(raw_alignment**2)))
        target, nuisance = pressure_subset(
            target_all,
            nuisance_all,
            PRESSURE_CANDIDATES,
            design,
            len(baseline.pooled),
            PRIMARY_STATE_LOG_SIGMA,
            shared_reference=True,
            trajectory_lengths=(len(baseline.wells["19A"]), len(baseline.wells["BT2"])),
            trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
            trajectory_discrepancy_features=model_discrepancy_features(baseline),
        )
        whitened_alignment_unit_rms = (target @ weak_direction) / max(raw_rms, 1e-30)
        for sigma in [0.0, 0.005, 0.010, 0.020, 0.050]:
            nuisance_use = nuisance
            if sigma > 0.0:
                nuisance_use = np.c_[
                    nuisance, sigma * whitened_alignment_unit_rms
                ]
            _, adjusted = combined_geometry(
                baseline, target, nuisance_use, "full_pressure_nuisance"
            )
            rows.append(
                {
                    "fabric_mode": mode,
                    "target_aligned_discrepancy_log_rms": sigma,
                    "target_aligned_discrepancy_percent_rms": 100.0 * sigma,
                    "raw_alignment_rms_per_scaled_parameter": raw_rms,
                    **geometry_metrics(adjusted, baseline.gram_adjusted),
                }
            )
    return pd.DataFrame(rows)


def trajectory_specific_design_table(baseline: Baseline) -> pd.DataFrame:
    """Repeat OED within each trajectory instead of pooling a common target."""

    rows: list[dict[str, float | str | int]] = []
    pair_designs = list(itertools.combinations(PRESSURE_CANDIDATES, 2))
    for trajectory, dataframe in baseline.wells.items():
        theta = rc.calibrate(dataframe, MODEL)
        metrics, jt, jn, gram, adjusted, nuisance_names = e1.metric_summary(
            dataframe, theta, baseline.scale
        )
        local = Baseline(
            wells={trajectory: dataframe},
            pooled=dataframe,
            theta=theta,
            scale=baseline.scale,
            target_jacobian=jt,
            nuisance_jacobian=jn,
            nuisance_names=nuisance_names,
            gram_raw=gram,
            gram_adjusted=adjusted,
            metrics=metrics,
            coordinate=e1.metric_contact_hs_decomposition(
                dataframe, theta, baseline.scale
            ),
            e2_summary=baseline.e2_summary,
        )
        for mode in ["nuisance", "expanded_nuisance"]:
            target_all, nuisance_all, _ = pm.pressure_differential_jacobian(
                dataframe,
                theta,
                PRESSURE_CANDIDATES,
                **FABRIC_CONFIGS[mode],
            )
            best_row: dict[str, float | str | int] | None = None
            for design in pair_designs:
                target, nuisance = pressure_subset(
                    target_all,
                    nuisance_all,
                    PRESSURE_CANDIDATES,
                    design,
                    len(dataframe),
                    PRIMARY_STATE_LOG_SIGMA,
                    shared_reference=True,
                    trajectory_lengths=(len(dataframe),),
                    trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
                    trajectory_discrepancy_features=model_discrepancy_features(local),
                )
                _, combined = combined_geometry(
                    local, target, nuisance, "full_pressure_nuisance"
                )
                combined_metrics = geometry_metrics(combined, adjusted)
                candidate: dict[str, float | str | int] = {
                    "trajectory": trajectory,
                    "n_samples": len(dataframe),
                    "fabric_mode": mode,
                    "Vcem_fraction": float(theta[0]),
                    "Cn": float(np.exp(theta[1])),
                    "pressure_1_mpa": design[0],
                    "pressure_2_mpa": design[1],
                    "baseline_lambda_min": float(np.linalg.eigvalsh(adjusted)[0]),
                    **combined_metrics,
                }
                if best_row is None or float(candidate["lambda_min"]) > float(
                    best_row["lambda_min"]
                ):
                    best_row = candidate
            if best_row is None:
                raise RuntimeError("no trajectory-specific pressure design")
            rows.append(best_row)
    return pd.DataFrame(rows)


def operating_point_design_sensitivity_table(
    baseline: Baseline,
    quick: bool = False,
) -> pd.DataFrame:
    """Recompute the primary OED at representative E2 bootstrap optima."""

    source = ROOT / "reference" / "E2_bootstrap_replicates.csv"
    replicates = pd.read_csv(source)
    primary = replicates[
        (replicates.scheme == "MBB_20m_primary")
        & replicates.bootstrap_success.astype(bool)
        & ~replicates.pooled_fit_bound_hit.astype(bool)
    ].copy()
    convex = []
    for row in primary.itertuples(index=False):
        theta = np.array([row.Vcem_fraction, math.log(row.Cn)])
        weights = pm.generalized_bounding_weights(
            baseline.pooled,
            theta,
            **FABRIC_CONFIGS[PRIMARY_FABRIC_MODE],
            soft_lncn_reference=float(theta[1]),
            stiff_lncn_reference=float(theta[1]),
            soft_phic_reference=float(rc.PHIC_PACK),
        )
        convex.append(
            bool(weights.W_K.between(0.0, 1.0).all())
            and bool(weights.W_G.between(0.0, 1.0).all())
        )
    primary = primary.loc[np.asarray(convex)].sort_values("Vcem_fraction")
    quantiles = np.linspace(0.0, 1.0, 3 if quick else 9)
    indices = np.unique(
        np.round(quantiles * (len(primary) - 1)).astype(int)
    )
    selected = primary.iloc[indices]
    designs = list(itertools.combinations(PRESSURE_CANDIDATES, 2))
    rows: list[dict[str, float | int]] = []
    for quantile, (_, replicate) in zip(quantiles, selected.iterrows()):
        theta = np.array(
            [float(replicate.Vcem_fraction), math.log(float(replicate.Cn))]
        )
        metrics, jt, jn, gram, adjusted, nuisance_names = e1.metric_summary(
            baseline.pooled, theta, baseline.scale
        )
        local = Baseline(
            wells=baseline.wells,
            pooled=baseline.pooled,
            theta=theta,
            scale=baseline.scale,
            target_jacobian=jt,
            nuisance_jacobian=jn,
            nuisance_names=nuisance_names,
            gram_raw=gram,
            gram_adjusted=adjusted,
            metrics=metrics,
            coordinate=e1.metric_contact_hs_decomposition(
                baseline.pooled, theta, baseline.scale
            ),
            e2_summary=baseline.e2_summary,
        )
        target_all, nuisance_all, _ = pm.pressure_differential_jacobian(
            baseline.pooled,
            theta,
            PRESSURE_CANDIDATES,
            **FABRIC_CONFIGS[PRIMARY_FABRIC_MODE],
        )
        best_row: dict[str, float | int] | None = None
        for design in designs:
            target, nuisance = pressure_subset(
                target_all,
                nuisance_all,
                PRESSURE_CANDIDATES,
                design,
                len(baseline.pooled),
                PRIMARY_STATE_LOG_SIGMA,
                shared_reference=True,
                trajectory_lengths=(len(baseline.wells["19A"]), len(baseline.wells["BT2"])),
                trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
                trajectory_discrepancy_features=model_discrepancy_features(local),
            )
            _, combined = combined_geometry(
                local, target, nuisance, "full_pressure_nuisance"
            )
            design_metrics = geometry_metrics(combined, adjusted)
            candidate: dict[str, float | int] = {
                "E2_Vcem_quantile": float(quantile),
                "E2_replicate": int(replicate.replicate),
                "Vcem_fraction": float(theta[0]),
                "Cn": float(np.exp(theta[1])),
                "pressure_1_mpa": design[0],
                "pressure_2_mpa": design[1],
                "baseline_lambda_min": float(np.linalg.eigvalsh(adjusted)[0]),
                **design_metrics,
            }
            if best_row is None or float(candidate["lambda_min"]) > float(
                best_row["lambda_min"]
            ):
                best_row = candidate
        if best_row is None:
            raise RuntimeError("no operating-point design candidates")
        rows.append(best_row)
        print(
            "operating-point design sensitivity: "
            f"quantile={quantile:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def bounding_weight_stress_table(baseline: Baseline) -> pd.DataFrame:
    """Audit convexity of the bounding average over plausible local states."""

    replicates = pd.read_csv(ROOT / "reference" / "E2_bootstrap_replicates.csv")
    primary = replicates[
        (replicates.scheme == "MBB_20m_primary")
        & replicates.bootstrap_success.astype(bool)
    ].sort_values("Vcem_fraction")
    quantiles = np.linspace(0.0, 1.0, 9)
    selected = primary.iloc[
        np.round(quantiles * (len(primary) - 1)).astype(int)
    ]
    theta_states = [
        ("pooled", baseline.theta),
        *[
            (
                f"E2_q{quantile:g}",
                np.array([row.Vcem_fraction, math.log(row.Cn)]),
            )
            for quantile, (_, row) in zip(quantiles, selected.iterrows())
        ],
    ]
    scenarios: list[tuple[str, dict[str, float]]] = [("central", {})]
    for name in [
        "ln_soft_cn_offset",
        "ln_stiff_cn_offset",
        "soft_phic_shift",
        "stiff_cement_volume_shift",
        "phic_pack_shift",
    ]:
        scale = pm.all_nuisance_scales()[name]
        scenarios.extend(
            [
                (f"{name}_minus_1sd", {name: -scale}),
                (f"{name}_plus_1sd", {name: scale}),
            ]
        )
    rows = []
    for theta_label, theta in theta_states:
        for scenario, nuisance in scenarios:
            weights = pm.generalized_bounding_weights(
                baseline.pooled,
                theta,
                nuisance=nuisance,
                **FABRIC_CONFIGS[PRIMARY_FABRIC_MODE],
                soft_lncn_reference=float(theta[1]),
                stiff_lncn_reference=float(theta[1]),
                soft_phic_reference=float(rc.PHIC_PACK),
            )
            outside = (~weights.W_K.between(0.0, 1.0)) | (
                ~weights.W_G.between(0.0, 1.0)
            )
            rows.append(
                {
                    "theta_state": theta_label,
                    "Vcem_fraction": float(theta[0]),
                    "Cn": float(np.exp(theta[1])),
                    "nuisance_scenario": scenario,
                    "W_K_min": float(weights.W_K.min()),
                    "W_K_max": float(weights.W_K.max()),
                    "W_G_min": float(weights.W_G.min()),
                    "W_G_max": float(weights.W_G.max()),
                    "fraction_outside_unit_interval": float(outside.mean()),
                }
            )
    return pd.DataFrame(rows)


def rank_one_design_map(baseline: Baseline) -> pd.DataFrame:
    baseline_values, baseline_vectors = eigensystem(baseline.gram_adjusted)
    strong = baseline_vectors[:, -1]
    weak = baseline_vectors[:, 0]
    rows: list[dict[str, float]] = []
    for angle in np.linspace(0.0, 90.0, 181):
        radians = math.radians(angle)
        direction = math.cos(radians) * strong + math.sin(radians) * weak
        for ratio in np.logspace(-4, 2, 121):
            information = ratio * baseline_values[-1]
            matrix = baseline.gram_adjusted + information * np.outer(
                direction, direction
            )
            metrics = geometry_metrics(matrix, baseline.gram_adjusted)
            rows.append(
                {
                    "angle_to_static_strong_deg": angle,
                    "information_over_baseline_lambda_max": ratio,
                    "lambda_min_gain": metrics["lambda_min_gain"],
                    "spectral_ratio": metrics["spectral_ratio"],
                    "worst_sd_reduction": metrics["worst_sd_reduction"],
                }
            )
    return pd.DataFrame(rows)


def candidate_observation_table(baseline: Baseline) -> pd.DataFrame:
    gradients = pm.candidate_gradients(
        baseline.theta,
        baseline.coordinate["A_adjusted"],
        baseline.coordinate["Gamma_adjusted"],
    )
    assumptions = {
        "q_star": (0.10, "10% log-scale precision; duplicates the static strong coordinate"),
        "Vcem_proxy": (0.005, "0.5 percentage-point absolute cement-volume precision"),
        "lnCn_proxy": (0.20, "20% log coordination-number precision"),
        "ln_contact_radius": (0.10, "10% relative contact-radius precision"),
        "pressure_response_proxy": (
            0.10,
            "10% log response precision; p=2/3, s=1 sensitivity family",
        ),
    }
    _, baseline_vectors = eigensystem(baseline.gram_adjusted)
    strong = baseline_vectors[:, -1]
    rows = []
    for name, gradient in gradients.items():
        sigma, description = assumptions[name]
        matrix = baseline.gram_adjusted + np.outer(gradient, gradient) / sigma**2
        metrics = geometry_metrics(matrix, baseline.gram_adjusted)
        rows.append(
            {
                "candidate": name,
                "assumed_sigma": sigma,
                "assumption": description,
                "scaled_gradient_Vcem": float(gradient[0]),
                "scaled_gradient_lnCn": float(gradient[1]),
                "gradient_norm": float(np.linalg.norm(gradient)),
                "angle_to_static_strong_deg": pm.angle_degrees(gradient, strong),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def moving_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    upper = max(n - block_length + 1, 1)
    while sum(len(block) for block in blocks) < n:
        start = int(rng.integers(0, upper))
        blocks.append(np.arange(start, min(start + block_length, n), dtype=int))
    return np.concatenate(blocks)[:n]


def stratified_block_indices(
    baseline: Baseline,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output: list[np.ndarray] = []
    offset = 0
    for name in ["19A", "BT2"]:
        n = len(baseline.wells[name])
        output.append(offset + moving_block_indices(n, block_length, rng))
        offset += n
    return np.concatenate(output)


def bootstrap_pressure_design(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    best: pd.DataFrame,
    repetitions: int = 400,
    block_length: int = 5,
    seed: int = 20260824,
) -> pd.DataFrame:
    specifications: list[tuple[str, tuple[float, ...]]] = []
    for size in [1, 2]:
        row = best[
            (best.fabric_mode == PRIMARY_FABRIC_MODE)
            & np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
            & (best.n_additional_pressures == size)
        ].iloc[0]
        specifications.append((f"best_{'single' if size == 1 else 'pair'}", design_from_row(row)))

    n_samples = len(baseline.pooled)
    trajectory_lengths = (len(baseline.wells["19A"]), len(baseline.wells["BT2"]))
    discrepancy_features = model_discrepancy_features(baseline)
    prepared: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for mode in ["shared", "nuisance", "expanded_nuisance"]:
        target_all, nuisance_all, _ = jacobians[mode]
        for label, design in specifications:
            prepared[(mode, label)] = pressure_subset(
                target_all,
                nuisance_all,
                PRESSURE_CANDIDATES,
                design,
                n_samples,
                PRIMARY_STATE_LOG_SIGMA,
                shared_reference=True,
                trajectory_lengths=trajectory_lengths,
                trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
                trajectory_discrepancy_features=discrepancy_features,
            )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []
    for replicate in range(repetitions):
        sample_index = stratified_block_indices(baseline, block_length, rng)
        base_rows = pm.sample_row_indices(sample_index, n_samples, len(NAMES))
        base_target = baseline.target_jacobian[base_rows]
        base_nuisance = expanded_baseline_nuisance(baseline, True)[base_rows]
        _, static_adjusted = schur_geometry(base_target, base_nuisance)
        static_values, _ = eigensystem(static_adjusted)

        for mode in ["shared", "nuisance", "expanded_nuisance"]:
            for label, design in specifications:
                pressure_target, pressure_nuisance = prepared[(mode, label)]
                if base_nuisance.shape[1] < pressure_nuisance.shape[1]:
                    base_nuisance_use = np.c_[
                        base_nuisance,
                        np.zeros(
                            (
                                len(base_nuisance),
                                pressure_nuisance.shape[1] - base_nuisance.shape[1],
                            )
                        ),
                    ]
                else:
                    base_nuisance_use = base_nuisance
                pressure_rows = pm.pressure_sample_row_indices(
                    sample_index,
                    list(design),
                    list(design),
                    n_samples,
                )
                target = np.vstack([base_target, pressure_target[pressure_rows]])
                nuisance = np.vstack(
                    [base_nuisance_use, pressure_nuisance[pressure_rows]]
                )
                _, adjusted = schur_geometry(target, nuisance)
                values, _ = eigensystem(adjusted)
                rows.append(
                    {
                        "replicate": replicate,
                        "block_length_samples": block_length,
                        "fabric_mode": mode,
                        "design": label,
                        "pressures_mpa": "+".join(f"{x:g}" for x in design),
                        "static_lambda_min": float(static_values[0]),
                        "combined_lambda_min": float(values[0]),
                        "lambda_min_gain": float(values[0] / static_values[0]),
                        "worst_sd_reduction": float(
                            math.sqrt(values[0] / static_values[0])
                        ),
                        "spectral_ratio": float(values[0] / values[-1]),
                    }
                )
        if (replicate + 1) % 50 == 0 or replicate + 1 == repetitions:
            print(f"pressure-design bootstrap: {replicate + 1}/{repetitions}", flush=True)
    return pd.DataFrame(rows)


def bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in bootstrap.groupby(["fabric_mode", "design", "pressures_mpa"]):
        mode, design, pressures = keys
        for metric in ["lambda_min_gain", "worst_sd_reduction", "spectral_ratio"]:
            values = group[metric].to_numpy()
            rows.append(
                {
                    "fabric_mode": mode,
                    "design": design,
                    "pressures_mpa": pressures,
                    "metric": metric,
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def conditional_bootstrap_design_selection(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    full_sample_design: tuple[float, ...],
    repetitions_per_block: int = 100,
    seed: int = 20260825,
) -> pd.DataFrame:
    """Re-optimize the pressure pair under conditional row resampling.

    The operating point and pressure-model Jacobians remain fixed.  This tests
    selection stability conditional on E1, rather than a population bootstrap.
    """

    mode = PRIMARY_FABRIC_MODE
    target_all, nuisance_all, _ = jacobians[mode]
    n_samples = len(baseline.pooled)
    trajectory_lengths = (len(baseline.wells["19A"]), len(baseline.wells["BT2"]))
    designs = list(itertools.combinations(PRESSURE_CANDIDATES, 2))
    prepared = {
        design: pressure_subset(
            target_all,
            nuisance_all,
            PRESSURE_CANDIDATES,
            design,
            n_samples,
            PRIMARY_STATE_LOG_SIGMA,
            shared_reference=True,
            trajectory_lengths=trajectory_lengths,
            trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
            trajectory_discrepancy_features=model_discrepancy_features(baseline),
        )
        for design in designs
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | bool]] = []
    for block_length in [3, 5, 10]:
        for replicate in range(repetitions_per_block):
            sample_index = stratified_block_indices(
                baseline, block_length, rng
            )
            base_rows = pm.sample_row_indices(sample_index, n_samples, len(NAMES))
            base_target = baseline.target_jacobian[base_rows]
            base_nuisance = expanded_baseline_nuisance(baseline, True)[base_rows]
            best_design: tuple[float, ...] | None = None
            best_lambda = -np.inf
            full_design_lambda = np.nan
            for design, (pressure_target, pressure_nuisance) in prepared.items():
                pressure_rows = pm.pressure_sample_row_indices(
                    sample_index, list(design), list(design), n_samples
                )
                if base_nuisance.shape[1] < pressure_nuisance.shape[1]:
                    base_nuisance_use = np.c_[
                        base_nuisance,
                        np.zeros(
                            (
                                len(base_nuisance),
                                pressure_nuisance.shape[1]
                                - base_nuisance.shape[1],
                            )
                        ),
                    ]
                else:
                    base_nuisance_use = base_nuisance
                target = np.vstack([base_target, pressure_target[pressure_rows]])
                nuisance = np.vstack(
                    [base_nuisance_use, pressure_nuisance[pressure_rows]]
                )
                _, adjusted = schur_geometry(target, nuisance)
                lambda_min = float(np.linalg.eigvalsh(adjusted)[0])
                if design == full_sample_design:
                    full_design_lambda = lambda_min
                if lambda_min > best_lambda:
                    best_lambda = lambda_min
                    best_design = design
            if best_design is None or not np.isfinite(full_design_lambda):
                raise RuntimeError("conditional design selection failed")
            rows.append(
                {
                    "block_length_samples": block_length,
                    "block_length_m": 4 * block_length,
                    "replicate": replicate,
                    "best_pressure_1_mpa": best_design[0],
                    "best_pressure_2_mpa": best_design[1],
                    "best_lambda_min": best_lambda,
                    "full_sample_design_lambda_min": full_design_lambda,
                    "full_sample_design_selected": best_design
                    == full_sample_design,
                    "full_sample_design_regret_fraction": best_lambda
                    / full_design_lambda
                    - 1.0,
                }
            )
        print(
            "conditional design selection: "
            f"block={block_length}, n={repetitions_per_block}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _profiled_objective(residual: np.ndarray, nuisance: np.ndarray) -> float:
    right = nuisance.T @ residual
    value = residual @ residual - right @ np.linalg.solve(
        nuisance.T @ nuisance + np.eye(nuisance.shape[1]), right
    )
    return float(max(value, 0.0))


def _whiten_pressure_vector(
    vector: np.ndarray,
    n_pressures: int,
    n_samples: int,
    state_log_sigma: float,
) -> np.ndarray:
    target = vector.reshape(-1, 1)
    nuisance = np.zeros((len(vector), 1))
    whitened, _ = whiten_pressure_blocks(
        target,
        nuisance,
        n_pressures,
        n_samples,
        state_log_sigma,
        shared_reference=True,
    )
    return whitened[:, 0]


def nonlinear_ridge_profiles(
    baseline: Baseline,
    jacobians: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    design: tuple[float, ...],
    quick: bool = False,
) -> pd.DataFrame:
    nv, nc = (17, 19) if quick else (31, 33)
    v_values = np.unique(
        np.r_[np.linspace(0.002, 0.035, nv), float(baseline.theta[0])]
    )
    cn_values = np.unique(
        np.r_[np.geomspace(4.0, 12.0, nc), float(np.exp(baseline.theta[1]))]
    )
    n_samples = len(baseline.pooled)
    static_reference = rc.stack(baseline.pooled, MODEL, baseline.theta, NAMES)
    static_sigma = e1.stacked_sigma(baseline.pooled, baseline.scale)
    static_nuisance = expanded_baseline_nuisance(baseline, True)

    target_all, nuisance_all, _ = jacobians[PRIMARY_FABRIC_MODE]
    pressure_target, pressure_nuisance = pressure_subset(
        target_all,
        nuisance_all,
        PRESSURE_CANDIDATES,
        design,
        n_samples,
        PRIMARY_STATE_LOG_SIGMA,
        shared_reference=True,
        trajectory_lengths=(len(baseline.wells["19A"]), len(baseline.wells["BT2"])),
        trajectory_discrepancy_sigma=PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
        trajectory_discrepancy_features=model_discrepancy_features(baseline),
    )
    del pressure_target
    pressure_reference = pm.pressure_differential_vector(
        baseline.pooled,
        baseline.theta,
        design,
        **FABRIC_CONFIGS[PRIMARY_FABRIC_MODE],
        soft_lncn_reference=float(baseline.theta[1]),
        stiff_lncn_reference=float(baseline.theta[1]),
        soft_phic_reference=float(rc.PHIC_PACK),
    )
    if static_nuisance.shape[1] < pressure_nuisance.shape[1]:
        static_nuisance = np.c_[
            static_nuisance,
            np.zeros(
                (
                    len(static_nuisance),
                    pressure_nuisance.shape[1] - static_nuisance.shape[1],
                )
            ),
        ]
    combined_nuisance = np.vstack([static_nuisance, pressure_nuisance])

    rows: list[dict[str, float]] = []
    for iv, vcem in enumerate(v_values):
        for cn in cn_values:
            theta = np.array([vcem, math.log(cn)])
            static_prediction = rc.stack(baseline.pooled, MODEL, theta, NAMES)
            static_residual = (static_prediction - static_reference) / static_sigma
            pressure_prediction = pm.pressure_differential_vector(
                baseline.pooled,
                theta,
                design,
                **FABRIC_CONFIGS[PRIMARY_FABRIC_MODE],
                soft_lncn_reference=float(baseline.theta[1]),
                stiff_lncn_reference=float(baseline.theta[1]),
                soft_phic_reference=float(rc.PHIC_PACK),
            )
            pressure_residual = _whiten_pressure_vector(
                pressure_prediction - pressure_reference,
                len(design),
                n_samples,
                PRIMARY_STATE_LOG_SIGMA,
            )
            combined_residual = np.r_[static_residual, pressure_residual]
            rows.append(
                {
                    "Vcem_fraction": float(vcem),
                    "Vcem_percent": float(100.0 * vcem),
                    "Cn": float(cn),
                    "lnCn": float(math.log(cn)),
                    "static_raw_objective": float(static_residual @ static_residual),
                    "static_adjusted_objective": _profiled_objective(
                        static_residual, static_nuisance
                    ),
                    "combined_raw_objective": float(
                        combined_residual @ combined_residual
                    ),
                    "combined_adjusted_objective": _profiled_objective(
                        combined_residual, combined_nuisance
                    ),
                }
            )
        if (iv + 1) % 5 == 0 or iv + 1 == len(v_values):
            print(f"nonlinear design grid: {iv + 1}/{len(v_values)} Vcem rows", flush=True)
    return pd.DataFrame(rows)


def profile_width_summary(
    profiles: pd.DataFrame,
    baseline: Baseline,
) -> pd.DataFrame:
    rows = []
    for objective in [
        "static_raw_objective",
        "static_adjusted_objective",
        "combined_raw_objective",
        "combined_adjusted_objective",
    ]:
        table = profiles.pivot(index="Vcem_fraction", columns="Cn", values=objective)
        minimum = float(np.nanmin(table.to_numpy()))
        prof_v = table.min(axis=1) - minimum
        prof_cn = table.min(axis=0) - minimum
        for threshold in [2.30, 5.99]:
            supported_v = prof_v.index[prof_v <= threshold].to_numpy(dtype=float)
            supported_cn = prof_cn.index[prof_cn <= threshold].to_numpy(dtype=float)
            rows.append(
                {
                    "objective": objective,
                    "threshold": threshold,
                    "Vcem_min_fraction": float(np.min(supported_v)),
                    "Vcem_max_fraction": float(np.max(supported_v)),
                    "Vcem_width_percentage_points": float(
                        100.0 * (np.max(supported_v) - np.min(supported_v))
                    ),
                    "Cn_min": float(np.min(supported_cn)),
                    "Cn_max": float(np.max(supported_cn)),
                    "Cn_width": float(np.max(supported_cn) - np.min(supported_cn)),
                    "Vcem_lower_censored": bool(
                        np.isclose(np.min(supported_v), table.index.min())
                    ),
                    "Vcem_upper_censored": bool(
                        np.isclose(np.max(supported_v), table.index.max())
                    ),
                    "Cn_lower_censored": bool(
                        np.isclose(np.min(supported_cn), table.columns.min())
                    ),
                    "Cn_upper_censored": bool(
                        np.isclose(np.max(supported_cn), table.columns.max())
                    ),
                    "truth_Vcem_fraction": float(baseline.theta[0]),
                    "truth_Cn": float(np.exp(baseline.theta[1])),
                }
            )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "figure.titlesize": 13,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix in ["png", "pdf"]:
        fig.savefig(
            ROOT / "results" / "figures" / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_no_go_and_controls(
    pressure_curves: pd.DataFrame,
    no_go: pd.DataFrame,
    multi_fluid: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    styles = {
        "frozen_constant_cement": (COLORS["gray"], "--", "frozen constant-cement"),
        "pressure_extension_shared_Cn": (COLORS["navy"], "-", "pressure extension"),
    }
    for model, (color, linestyle, label) in styles.items():
        part = pressure_curves[pressure_curves.model == model]
        ax.plot(
            part.pressure_mpa,
            part.Vp_change_percent,
            color=color,
            ls=linestyle,
            lw=2.1,
            label=f"$V_p$: {label}",
        )
        ax.plot(
            part.pressure_mpa,
            part.Vs_change_percent,
            color=color,
            ls=":" if linestyle == "-" else "-.",
            lw=1.8,
            label=f"$V_s$: {label}",
        )
    ax.axvline(pm.REFERENCE_PRESSURE_MPA, color="black", lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.4)
    ax.set_xlabel("Effective pressure (MPa)")
    ax.set_ylabel(f"Change from the {pm.REFERENCE_PRESSURE_MPA:g} MPa state (%)")
    ax.set_title("A. The standard model is pressure-blind")
    ax.legend(ncol=2, frameon=False)

    ax = axes[0, 1]
    ax.plot(
        no_go.n_identical_static_states,
        no_go.lambda_min_gain,
        marker="o",
        color=COLORS["navy"],
        lw=2,
        label=r"$\lambda_{\min}$ gain",
    )
    ax2 = ax.twinx()
    ax2.plot(
        no_go.n_identical_static_states,
        no_go.raw_strong_direction_rotation_deg,
        marker="s",
        color=COLORS["red"],
        lw=1.7,
        label="raw new angle",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of identical static states")
    ax.set_ylabel(r"Gain in $\lambda_{\min}$")
    ax2.set_ylabel("Rotation of strong direction (degrees)", color=COLORS["red"])
    ax2.set_ylim(-0.02, 0.25)
    ax.set_title("B. Replication reduces noise but adds no direction")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False)

    ax = axes[1, 0]
    labels = multi_fluid.design.str.replace("plus_", "").str.replace("_", " ")
    ax.barh(labels, multi_fluid.lambda_min_gain, color=COLORS["blue"], alpha=0.82)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xlim(0.92, 2.50)
    ax.set_xlabel(r"Gain in nuisance-adjusted $\lambda_{\min}$")
    ax.set_title("C. Additional fluid states mostly repeat static elasticity")
    for y, value in enumerate(multi_fluid.lambda_min_gain):
        ax.text(value * 1.02, y, f"{value:.2f}×", va="center", fontsize=8)

    ax = axes[1, 1]
    primary = best[
        np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (best.n_additional_pressures == 2)
    ].copy()
    order = ["shared", "fixed", "nuisance", "expanded_nuisance"]
    primary["fabric_mode"] = pd.Categorical(primary.fabric_mode, order, ordered=True)
    primary = primary.sort_values("fabric_mode")
    labels = [
        "shared\n" + r"$C_n^{\mathrm{soft}}$",
        "fixed\n" + r"$C_n^{\mathrm{soft}}$",
        "free\n" + r"$C_n^{\mathrm{soft}}$",
        "expanded\nfabric",
    ]
    bars = ax.bar(
        labels,
        primary.lambda_min_gain,
        color=[COLORS["teal"], COLORS["gold"], COLORS["red"], COLORS["navy"]],
        alpha=0.88,
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(0.8, 0.75 * primary.lambda_min_gain.min()), top=1.75 * primary.lambda_min_gain.max())
    ax.set_ylabel(r"Gain in nuisance-adjusted $\lambda_{\min}$")
    ax.set_title("D. Pressure gain depends on the fabric-link hypothesis")
    for bar, (_, row) in zip(bars, primary.iterrows()):
        design = design_from_row(row)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{bar.get_height():.0f}×\n({'+'.join(f'{p:g}' for p in design)} MPa)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.suptitle(
        "Sensitivity diversity requires new pressure-dependent physics",
        fontweight="semibold",
    )
    save_figure(fig, "Fig_E3_no_go_and_controls")


def plot_pressure_design(
    design_grid: pd.DataFrame,
    best: pd.DataFrame,
    ablation: pd.DataFrame,
    discrepancy: pd.DataFrame,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    primary = design_grid[
        (design_grid.fabric_mode == PRIMARY_FABRIC_MODE)
        & np.isclose(design_grid.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (design_grid.n_additional_pressures == 2)
    ]
    matrix = primary.pivot(
        index="pressure_1_mpa", columns="pressure_2_mpa", values="lambda_min_gain"
    )
    ax = axes[0, 0]
    image = ax.imshow(
        np.log10(matrix.to_numpy()),
        origin="lower",
        aspect="auto",
        extent=[matrix.columns.min(), matrix.columns.max(), matrix.index.min(), matrix.index.max()],
        cmap="viridis",
    )
    best_row = primary.loc[primary.lambda_min.idxmax()]
    ax.scatter(best_row.pressure_2_mpa, best_row.pressure_1_mpa, marker="*", s=120, c="white", edgecolor="black")
    ax.set_xlim(matrix.columns.min() - 1.5, matrix.columns.max() + 1.5)
    ax.set_ylim(matrix.index.min() - 1.5, matrix.index.max() + 1.5)
    ax.set_xlabel("Second pressure (MPa)")
    ax.set_ylabel("First pressure (MPa)")
    ax.set_title("A. E-optimal two-state design, expanded fabric nuisances")
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(r"$\log_{10}$ gain in $\lambda_{\min}$")

    ax = axes[0, 1]
    for mode, color, label in [
        ("shared", COLORS["teal"], "shared compliant $C_n$"),
        ("fixed", COLORS["gold"], "fixed compliant $C_n$"),
        ("nuisance", COLORS["red"], "independent compliant $C_n$"),
        ("expanded_nuisance", COLORS["navy"], "expanded fabric nuisances"),
    ]:
        part = design_grid[
            (design_grid.fabric_mode == mode)
            & np.isclose(design_grid.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
            & (design_grid.n_additional_pressures == 1)
        ].sort_values("pressure_1_mpa")
        ax.plot(part.pressure_1_mpa, part.lambda_min_gain, marker="o", ms=3.5, lw=1.8, color=color, label=label)
    ax.axvline(pm.REFERENCE_PRESSURE_MPA, color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Additional pressure (MPa)")
    ax.set_ylabel(r"Gain in $\lambda_{\min}$")
    ax.set_title("B. Pressure contrast and fabric assumptions")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for mode, color, label in [
        ("shared", COLORS["teal"], "shared compliant $C_n$"),
        ("fixed", COLORS["gold"], "fixed compliant $C_n$"),
        ("nuisance", COLORS["red"], "independent compliant $C_n$"),
        ("expanded_nuisance", COLORS["navy"], "expanded fabric nuisances"),
    ]:
        part = best[
            (best.fabric_mode == mode) & (best.n_additional_pressures == 2)
        ].sort_values("state_log_sigma")
        ax.plot(
            100 * part.state_log_sigma,
            part.lambda_min_gain,
            marker="o",
            color=color,
            lw=2,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Per-state log-velocity uncertainty (%)")
    ax.set_ylabel(r"Best-pair gain in $\lambda_{\min}$")
    ax.set_title("C. Measurement-precision requirement")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    discrepancy_styles = [
        ("intercept", COLORS["gold"], "intercept"),
        ("intercept_plus_porosity", COLORS["teal"], "intercept + porosity"),
        (
            "intercept_plus_porosity_plus_clay",
            COLORS["navy"],
            "intercept + porosity + clay",
        ),
    ]
    for basis, color, label in discrepancy_styles:
        part = discrepancy[discrepancy.basis == basis]
        ax.plot(
            part.trajectory_discrepancy_percent,
            part.lambda_min_gain,
            marker="o",
            lw=1.8,
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Discrepancy-basis coefficient uncertainty (%)")
    ax.set_ylabel(r"Gain in $\lambda_{\min}$")
    ax.set_title("D. Survival under correlated model discrepancy")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Prospective multi-pressure design under an Avseth–Skjei-inspired extension",
        fontweight="semibold",
    )
    save_figure(fig, "Fig_E3_pressure_design")


def _profile_grid(
    profiles: pd.DataFrame,
    value: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.sort(profiles.Vcem_percent.unique())
    y = np.sort(profiles.Cn.unique())
    z = (
        profiles.pivot(index="Cn", columns="Vcem_percent", values=value)
        .reindex(index=y, columns=x)
        .to_numpy()
    )
    z -= np.nanmin(z)
    return x, y, z


def plot_ridge_breaking(
    profiles: pd.DataFrame,
    baseline: Baseline,
    design: tuple[float, ...],
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    objectives = [
        (
            "static_adjusted_objective",
            "A. Static data: locally adjusted ridge (boundary-censored)",
        ),
        (
            "combined_adjusted_objective",
            "B. Static + pressure: locally adjusted ridge (boundary-censored)",
        ),
    ]
    for ax, (field, title) in zip(axes[0], objectives):
        x, y, z = _profile_grid(profiles, field)
        image = ax.pcolormesh(
            x,
            y,
            np.log10(np.maximum(z, 1e-4)),
            shading="auto",
            cmap="magma_r",
            vmin=-4,
            vmax=3,
        )
        contour = ax.contour(x, y, z, levels=[2.30, 5.99, 20.0], colors=["white", "cyan", "black"], linewidths=[1.4, 1.2, 0.9])
        ax.clabel(contour, fmt={2.30: "2.30", 5.99: "5.99", 20.0: "20"}, fontsize=7)
        ax.scatter(
            100 * baseline.theta[0],
            np.exp(baseline.theta[1]),
            marker="*",
            s=120,
            c=COLORS["gold"],
            edgecolors="black",
            zorder=5,
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
        ax.set_ylabel(r"Coordination number $C_n$")
        ax.set_title(title)
    fig.colorbar(image, ax=axes[0].tolist(), pad=0.015, label=r"$\log_{10}\Delta\Phi$")

    ax = axes[1, 0]
    for field, color, label in [
        ("static_adjusted_objective", COLORS["gray"], "static"),
        ("combined_adjusted_objective", COLORS["red"], "static + pressure"),
    ]:
        table = profiles.pivot(index="Vcem_percent", columns="Cn", values=field)
        curve = table.min(axis=1) - table.to_numpy().min()
        ax.plot(curve.index, curve.to_numpy(), color=color, lw=2, label=label)
    ax.axhline(2.30, color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cement volume $V_{\rm cem}$ (%)")
    ax.set_ylabel(r"Profile objective $\Delta\Phi$")
    ax.set_title("C. Profile over coordination number")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for field, color, label in [
        ("static_adjusted_objective", COLORS["gray"], "static"),
        ("combined_adjusted_objective", COLORS["red"], "static + pressure"),
    ]:
        table = profiles.pivot(index="Vcem_fraction", columns="Cn", values=field)
        curve = table.min(axis=0) - table.to_numpy().min()
        ax.plot(curve.index, curve.to_numpy(), color=color, lw=2, label=label)
    ax.axhline(2.30, color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Coordination number $C_n$")
    ax.set_ylabel(r"Profile objective $\Delta\Phi$")
    ax.set_title("D. Profile over cement volume")
    ax.legend(frameon=False)
    fig.suptitle(
        "How much does the selected pressure experiment contract the ridge?\n"
        f"Design: {' + '.join(f'{p:g}' for p in design)} MPa relative to {pm.REFERENCE_PRESSURE_MPA:g} MPa",
        fontweight="semibold",
    )
    save_figure(fig, "Fig_E3_ridge_breaking")


def plot_robustness_and_direction(
    bootstrap: pd.DataFrame,
    design_map: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
    ax = axes[0]
    combinations = [
        ("shared", "best_pair", COLORS["teal"], "shared compliant $C_n$"),
        ("nuisance", "best_pair", COLORS["red"], "independent compliant $C_n$"),
        (
            "expanded_nuisance",
            "best_pair",
            COLORS["navy"],
            "expanded fabric nuisances",
        ),
    ]
    for mode, design, color, label in combinations:
        values = bootstrap[
            (bootstrap.fabric_mode == mode) & (bootstrap.design == design)
        ].lambda_min_gain
        ax.hist(
            np.log10(values),
            bins=28,
            alpha=0.48,
            color=color,
            label=label,
        )
    ax.set_xlabel(r"$\log_{10}$ bootstrap gain in $\lambda_{\min}$")
    ax.set_ylabel("Replicates")
    ax.set_title("A. Block-bootstrap robustness")
    ax.legend(frameon=False)

    ax = axes[1]
    x = np.sort(design_map.angle_to_static_strong_deg.unique())
    y = np.sort(design_map.information_over_baseline_lambda_max.unique())
    z = (
        design_map.pivot(
            index="information_over_baseline_lambda_max",
            columns="angle_to_static_strong_deg",
            values="lambda_min_gain",
        )
        .reindex(index=y, columns=x)
        .to_numpy()
    )
    image = ax.pcolormesh(x, y, np.log10(z), shading="auto", cmap="viridis")
    ax.set_yscale("log")
    ax.set_xlabel("Angle to static strong direction (degrees)")
    ax.set_ylabel(r"Added information / static $\lambda_{\max}$")
    ax.set_title("B. One-observable design law")
    fig.colorbar(image, ax=ax, pad=0.015, label=r"$\log_{10}$ gain in $\lambda_{\min}$")

    ax = axes[2]
    ordered = candidates.sort_values("lambda_min_gain")
    display_labels = {
        "q_star": r"$q_\star$",
        "Vcem_proxy": r"$V_{\rm cem}$ proxy",
        "lnCn_proxy": r"$\ln C_n$ proxy",
        "ln_contact_radius": r"$\ln a_c$",
        "pressure_response_proxy": "pressure-response proxy",
    }
    labels = ordered.candidate.map(display_labels)
    colors = [COLORS["gray"] if name == "q_star" else COLORS["blue"] for name in ordered.candidate]
    ax.barh(labels, ordered.lambda_min_gain, color=colors, alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xlabel(r"Illustrative gain in $\lambda_{\min}$")
    ax.set_title("C. Direction matters more than observable count")
    for yindex, value in enumerate(ordered.lambda_min_gain):
        ax.text(value * 1.03, yindex, f"{value:.1f}×", va="center", fontsize=7.5)
    fig.suptitle(
        "New information must project onto the weak microstructural direction",
        fontweight="semibold",
    )
    save_figure(fig, "Fig_E3_robustness_and_direction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_to_dict(
    row: pd.Series, fields: list[str]
) -> dict[str, float | int | str | bool | None]:
    output: dict[str, float | int | str | bool | None] = {}
    for field in fields:
        value = row[field]
        if pd.isna(value):
            output[field] = None
        elif isinstance(value, (np.bool_, bool)):
            output[field] = bool(value)
        elif isinstance(value, (np.floating, float)):
            output[field] = float(value)
        elif isinstance(value, (np.integer, int)):
            output[field] = int(value)
        else:
            output[field] = str(value)
    return output


def build_summary(
    baseline: Baseline,
    pressure_audit: pd.DataFrame,
    no_go: pd.DataFrame,
    fluid_control: pd.DataFrame,
    weights: pd.DataFrame,
    weight_stress: pd.DataFrame,
    best: pd.DataFrame,
    ablation: pd.DataFrame,
    bootstrap_summary_table: pd.DataFrame,
    profile_widths: pd.DataFrame,
    finite_difference: pd.DataFrame,
    discrepancy_sensitivity: pd.DataFrame,
    reference_pressure_sensitivity: pd.DataFrame,
    target_aligned_discrepancy: pd.DataFrame,
    trajectory_specific: pd.DataFrame,
    operating_point_sensitivity: pd.DataFrame,
    conditional_design_selection: pd.DataFrame,
    design: tuple[float, ...],
    primary_matrix: np.ndarray,
) -> dict:
    best_primary = best[
        (best.fabric_mode == PRIMARY_FABRIC_MODE)
        & np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (best.n_additional_pressures == 2)
    ].iloc[0]
    best_primary_single = best[
        (best.fabric_mode == PRIMARY_FABRIC_MODE)
        & np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (best.n_additional_pressures == 1)
    ].iloc[0]
    best_modes = {}
    for mode in ["shared", "fixed", "nuisance", "expanded_nuisance"]:
        row = best[
            (best.fabric_mode == mode)
            & np.isclose(best.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
            & (best.n_additional_pressures == 2)
        ].iloc[0]
        best_modes[mode] = _row_to_dict(
            row,
            [
                "pressure_1_mpa",
                "pressure_2_mpa",
                "lambda_min",
                "lambda_min_gain",
                "worst_sd_reduction",
                "spectral_ratio",
                "condition_number",
            ],
        )

    bootstrap_primary = {}
    for mode in ["shared", "nuisance", "expanded_nuisance"]:
        rows = bootstrap_summary_table[
            (bootstrap_summary_table.fabric_mode == mode)
            & (bootstrap_summary_table.design == "best_pair")
        ].set_index("metric")
        bootstrap_primary[mode] = {
            metric: {
                "median": float(rows.loc[metric, "median"]),
                "ci95": [
                    float(rows.loc[metric, "q025"]),
                    float(rows.loc[metric, "q975"]),
                ],
            }
            for metric in ["lambda_min_gain", "worst_sd_reduction", "spectral_ratio"]
        }
        bootstrap_primary[mode]["n_replicates"] = int(
            rows.loc["lambda_min_gain", "n"]
        )

    full_discrepancy_basis = discrepancy_sensitivity[
        discrepancy_sensitivity.basis == "intercept_plus_porosity_plus_clay"
    ]
    discrepancy_basis_at_primary_sigma = discrepancy_sensitivity[
        np.isclose(
            discrepancy_sensitivity.trajectory_discrepancy_log_sigma,
            PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
        )
    ]

    profile_summary = {}
    for objective in ["static_adjusted_objective", "combined_adjusted_objective"]:
        row = profile_widths[
            (profile_widths.objective == objective)
            & np.isclose(profile_widths.threshold, 2.30)
        ].iloc[0]
        profile_summary[objective] = _row_to_dict(
            row,
            [
                "Vcem_min_fraction",
                "Vcem_max_fraction",
                "Vcem_width_percentage_points",
                "Cn_min",
                "Cn_max",
                "Cn_width",
                "Vcem_lower_censored",
                "Vcem_upper_censored",
                "Cn_lower_censored",
                "Cn_upper_censored",
            ],
        )

    primary_ablation = ablation[
        (ablation.design == "best_pair")
        & (ablation.fabric_mode == PRIMARY_FABRIC_MODE)
        & np.isclose(ablation.state_log_sigma, PRIMARY_STATE_LOG_SIGMA)
        & (ablation.adjustment == "full_pressure_nuisance")
        & np.isclose(ablation.pressure_nuisance_prior_precision_factor, 1.0)
    ].iloc[0]

    conditional_selection_summary = {}
    for block_length, group in conditional_design_selection.groupby(
        "block_length_samples"
    ):
        conditional_selection_summary[f"{int(block_length)}_samples"] = {
            "n": int(len(group)),
            "full_sample_pair_selection_frequency": float(
                group.full_sample_design_selected.mean()
            ),
            "median_regret_fraction": float(
                group.full_sample_design_regret_fraction.median()
            ),
            "max_regret_fraction": float(
                group.full_sample_design_regret_fraction.max()
            ),
        }

    operating_pair_match = (
        np.isclose(operating_point_sensitivity.pressure_1_mpa, design[0])
        & np.isclose(operating_point_sensitivity.pressure_2_mpa, design[1])
    )
    target_aligned_primary = target_aligned_discrepancy[
        target_aligned_discrepancy.fabric_mode == PRIMARY_FABRIC_MODE
    ]

    return {
        "experiment": "E3 Break: nuisance-adjusted multi-state design",
        "model": {
            "static": "frozen RPIA constant-cement Scheme 1",
            "pressure_extension": "Avseth-Skjei-inspired bounding average",
            "status": "prospective heuristic experimental-design model",
            "reference_pressure_mpa": pm.REFERENCE_PRESSURE_MPA,
            "stiff_end_member_cement_volume": pm.STIFF_CEMENT_VOLUME,
            "primary_fabric_mode": PRIMARY_FABRIC_MODE,
        },
        "sample_counts": {
            "19A": len(baseline.wells["19A"]),
            "BT2": len(baseline.wells["BT2"]),
            "pooled": len(baseline.pooled),
        },
        "operating_point": {
            "Vcem_fraction": float(baseline.theta[0]),
            "Cn": float(np.exp(baseline.theta[1])),
            "baseline_adjusted_lambda_min": float(
                np.linalg.eigvalsh(baseline.gram_adjusted)[0]
            ),
            "static_local_uncertainty": parameter_uncertainty_metrics(
                baseline.gram_adjusted
            ),
            "baseline_adjusted_spectral_ratio": float(
                baseline.metrics["adjusted_spectral_ratio"]
            ),
        },
        "exact_no_go": {
            "max_pressure_induced_change_standard_model": float(
                pressure_audit[
                    [
                        "max_abs_delta_Vp_mps",
                        "max_abs_delta_Vs_mps",
                        "max_abs_delta_rho_gcc",
                    ]
                ].to_numpy().max()
            ),
            "maximum_raw_direction_rotation_deg_under_replication": float(
                no_go.raw_strong_direction_rotation_deg.max()
            ),
            "interpretation": "The standard constant-cement equations contain no pressure variable; repeated pressure labels do not create sensitivity diversity.",
        },
        "controls": {
            "best_multi_fluid_lambda_min_gain": float(
                fluid_control.lambda_min_gain.max()
            ),
            "patchy_weight_ranges": {
                "W_K": [float(weights.W_K.min()), float(weights.W_K.max())],
                "W_G": [float(weights.W_G.min()), float(weights.W_G.max())],
            },
            "bounding_weight_stress": {
                "overall_W_K_range": [
                    float(weight_stress.W_K_min.min()),
                    float(weight_stress.W_K_max.max()),
                ],
                "overall_W_G_range": [
                    float(weight_stress.W_G_min.min()),
                    float(weight_stress.W_G_max.max()),
                ],
                "maximum_fraction_outside_unit_interval": float(
                    weight_stress.fraction_outside_unit_interval.max()
                ),
                "pooled_state_maximum_fraction_outside_unit_interval": float(
                    weight_stress.loc[
                        weight_stress.theta_state == "pooled",
                        "fraction_outside_unit_interval",
                    ].max()
                ),
            },
        },
        "primary_design": {
            "pressures_mpa": list(map(float, design)),
            "per_state_log_velocity_sigma": PRIMARY_STATE_LOG_SIGMA,
            "shared_reference_error_correlation": 0.5,
            "absolute_state_count": len(design) + 1,
            "candidate_pressure_range_mpa": [
                float(min(PRESSURE_CANDIDATES)),
                float(max(PRESSURE_CANDIDATES)),
            ],
            "touches_candidate_boundary": bool(
                min(design) == min(PRESSURE_CANDIDATES)
                or max(design) == max(PRESSURE_CANDIDATES)
            ),
            "spans_full_candidate_range": bool(
                min(design) == min(PRESSURE_CANDIDATES)
                and max(design) == max(PRESSURE_CANDIDATES)
            ),
            "trajectory_level_model_discrepancy_sigma": PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
            "trajectory_level_model_discrepancy_basis": list(
                PRIMARY_TRAJECTORY_DISCREPANCY_BASIS
            ),
            "post_design_local_uncertainty": parameter_uncertainty_metrics(
                primary_matrix
            ),
            **_row_to_dict(
                best_primary,
                [
                    "lambda_min",
                    "lambda_min_gain",
                    "worst_sd_reduction",
                    "spectral_ratio",
                    "condition_number",
                ],
            ),
            "full_nuisance_ablation_check": _row_to_dict(
                primary_ablation,
                ["lambda_min_gain", "worst_sd_reduction", "spectral_ratio"],
            ),
            "best_single_state_comparator": {
                **_row_to_dict(
                    best_primary_single,
                    [
                        "pressure_1_mpa",
                        "lambda_min",
                        "lambda_min_gain",
                        "worst_sd_reduction",
                        "spectral_ratio",
                    ],
                ),
                "pair_over_single_lambda_min_ratio": float(
                    best_primary.lambda_min / best_primary_single.lambda_min
                ),
                "pair_over_single_worst_sd_reduction_ratio": float(
                    best_primary.worst_sd_reduction
                    / best_primary_single.worst_sd_reduction
                ),
            },
        },
        "fabric_link_ablation": best_modes,
        "reference_pressure_sensitivity": [
            _row_to_dict(
                row,
                [
                    "reference_pressure_mpa",
                    "fabric_mode",
                    "n_additional_pressures",
                    "pressure_1_mpa",
                    "pressure_2_mpa",
                    "lambda_min_gain",
                    "worst_sd_reduction",
                    "spectral_ratio",
                ],
            )
            for _, row in reference_pressure_sensitivity.iterrows()
        ],
        "target_aligned_discrepancy_primary": {
            f"{row.target_aligned_discrepancy_percent_rms:g}_percent": {
                "lambda_min_gain": float(row.lambda_min_gain),
                "worst_sd_reduction": float(row.worst_sd_reduction),
                "spectral_ratio": float(row.spectral_ratio),
            }
            for row in target_aligned_primary.itertuples(index=False)
        },
        "trajectory_specific_primary": [
            _row_to_dict(
                row,
                [
                    "trajectory",
                    "n_samples",
                    "Vcem_fraction",
                    "Cn",
                    "pressure_1_mpa",
                    "pressure_2_mpa",
                    "lambda_min_gain",
                    "worst_sd_reduction",
                    "spectral_ratio",
                ],
            )
            for _, row in trajectory_specific[
                trajectory_specific.fabric_mode == PRIMARY_FABRIC_MODE
            ].iterrows()
        ],
        "operating_point_design_sensitivity": {
            "n_representative_E2_bootstrap_points": int(
                len(operating_point_sensitivity)
            ),
            "full_sample_pair_selection_frequency": float(
                operating_pair_match.mean()
            ),
            "lambda_min_gain_range": [
                float(operating_point_sensitivity.lambda_min_gain.min()),
                float(operating_point_sensitivity.lambda_min_gain.max()),
            ],
        },
        "conditional_design_selection": conditional_selection_summary,
        "model_discrepancy_sensitivity": {
            f"{row.trajectory_discrepancy_percent:g}_percent": {
                "lambda_min_gain": float(row.lambda_min_gain),
                "worst_sd_reduction": float(row.worst_sd_reduction),
                "spectral_ratio": float(row.spectral_ratio),
            }
            for row in full_discrepancy_basis.itertuples(index=False)
        },
        "model_discrepancy_basis_ablation_at_1_percent": {
            row.basis: {
                "n_basis_terms": int(row.n_basis_terms),
                "lambda_min_gain": float(row.lambda_min_gain),
                "worst_sd_reduction": float(row.worst_sd_reduction),
                "spectral_ratio": float(row.spectral_ratio),
            }
            for row in discrepancy_basis_at_primary_sigma.itertuples(index=False)
        },
        "bootstrap_primary": bootstrap_primary,
        "nonlinear_profile_widths_at_delta_phi_2p30": profile_summary,
        "finite_difference": {
            "max_target_relative_error": float(
                finite_difference["target_relative_error_to_1e-4"].max()
            ),
            "max_nuisance_relative_error": float(
                finite_difference["nuisance_relative_error_to_1e-4"].max()
            ),
        },
        "scientific_conclusion": (
            "Pressure labels cannot create a constitutively new direction in the frozen constant-cement model. "
            "The prospective stress-sensitive extension appears to contract the ridge only while important compliant and stiff fabric variables remain linked to the nominal targets. "
            "When compliant-contact Cn, stiff-bound Cn, and compliant critical porosity are all nuisance-adjusted, most of the apparent remediation is absorbed."
        ),
        "limitations": [
            "The pressure extension is heuristic and prospective; no pressure-dependent observations were available in E1-E2.",
            "The moving-block bootstrap is conditional on two Hugin trajectories; a separate sensitivity analysis samples representative E2 operating points but is not a full population bootstrap.",
            "Loading-unloading hysteresis, Biot uncertainty beyond the pressure-calibration nuisance, and higher-order model-form discrepancy require laboratory validation.",
            "The bounding average is convex at the pooled operating point and its one-sigma local nuisance scenarios, but it leaves the unit-weight interval for part of the wider E2 bootstrap operating-point range; operating-point OED controls are restricted to the convex validity domain.",
        ],
        "provenance": {
            "E1_expected_summary_sha256": sha256(
                ROOT / "vendor" / "e1_v1" / "reference" / "expected_summary.json"
            ),
            "E2_summary_sha256": sha256(ROOT / "reference" / "E2_summary.json"),
            "E2_bootstrap_replicates_sha256": sha256(
                ROOT / "reference" / "E2_bootstrap_replicates.csv"
            ),
            "rpia_core_sha256": sha256(
                ROOT
                / "vendor"
                / "e1_v1"
                / "vendor"
                / "rpia_v1"
                / "rpia_core.py"
            ),
        },
    }


def write_results_markdown(summary: dict) -> None:
    design = summary["primary_design"]
    modes = summary["fabric_link_ablation"]
    boot = summary["bootstrap_primary"][PRIMARY_FABRIC_MODE]["lambda_min_gain"]
    widths = summary["nonlinear_profile_widths_at_delta_phi_2p30"]
    uncertainty = design["post_design_local_uncertainty"]
    single = design["best_single_state_comparator"]
    selection = summary["conditional_design_selection"]["5_samples"]
    aligned = summary["target_aligned_discrepancy_primary"]
    weight_control = summary["controls"]["bounding_weight_stress"]
    text = f"""# E3 results: can pressure break the constant-cement ridge?

## Main result

The frozen constant-cement model is exactly pressure independent. Repetition can increase precision, but pressure labels alone cannot create a new constitutive sensitivity direction. A legitimate multi-state experiment therefore requires an explicit pressure-sensitive branch.

Under the prospective Avseth–Skjei-inspired extension, the E-optimal pair among the tested candidates used **{' + '.join(f'{p:g}' for p in design['pressures_mpa'])} MPa** in addition to the {summary['model']['reference_pressure_mpa']:g} MPa reference state. The primary analysis uses expanded fabric nuisances, {100*design['per_state_log_velocity_sigma']:.2f}% per-state log-velocity uncertainty, the shared-reference covariance, and {100*design['trajectory_level_model_discrepancy_sigma']:.1f}%-RMS trajectory-state discrepancy bases spanning intercept, porosity, and clay-fraction trends.

- Nuisance-adjusted lambda-min gain: **{design['lambda_min_gain']:.2f}x**.
- Reduction in the worst local standard deviation: **{design['worst_sd_reduction']:.2f}x**.
- Adjusted spectral ratio after the experiment: **{design['spectral_ratio']:.4f}**.
- The unconstrained local-Gaussian marginal SDs remain {uncertainty['sd_Vcem_percentage_points']:.2f} percentage points in Vcem and {uncertainty['sd_lnCn']:.3f} in ln Cn; their correlation is {uncertainty['Vcem_lnCn_correlation']:.3f}.
- The best single added state was {single['pressure_1_mpa']:g} MPa; the second state increased lambda-min by only {100*(single['pair_over_single_lambda_min_ratio']-1):.1f}%.
- The selected pair touches the lower boundary of the tested pressure set; pressures below {design['candidate_pressure_range_mpa'][0]:g} MPa were not ruled out.

## Why the fabric ablation matters

- Shared nominal and compliant-contact Cn: lambda-min gain {modes['shared']['lambda_min_gain']:.2f}x.
- Locally fixed compliant-contact Cn (an oracle control): gain {modes['fixed']['lambda_min_gain']:.2f}x.
- Independent compliant-contact Cn with a prior: gain {modes['nuisance']['lambda_min_gain']:.2f}x.
- Expanded fabric nuisances, also separating stiff-bound Cn and compliant critical porosity: gain {modes['expanded_nuisance']['lambda_min_gain']:.2f}x.

The expanded control is the conservative primary result. It shows that most of the apparently new direction can be reabsorbed when the fabric of the compliant and stiff branches is not assumed known. Pressure alone therefore does not robustly repair the nominal Vcem-Cn ambiguity in this scenario model.

## Robustness

In the {summary['bootstrap_primary'][PRIMARY_FABRIC_MODE]['n_replicates']}-replicate, trajectory-stratified 20 m moving-block bootstrap, the expanded-fabric best-pair lambda-min gain had median {boot['median']:.2f}x and 95% interval [{boot['ci95'][0]:.2f}, {boot['ci95'][1]:.2f}]x. Conditional re-optimization selected the full-sample pair in {100*selection['full_sample_pair_selection_frequency']:.1f}% of 20 m replicates; this does not include full recalibration of the pressure model.

The nonlinear target grid uses a nuisance tangent space frozen at the operating point. Its static Delta-Phi=2.30 support is boundary-censored: the Vcem width is at least {widths['static_adjusted_objective']['Vcem_width_percentage_points']:.3f} percentage points and the Cn width is at least {widths['static_adjusted_objective']['Cn_width']:.3f}. The combined support is also boundary-censored: its widths are at least {widths['combined_adjusted_objective']['Vcem_width_percentage_points']:.3f} percentage points and {widths['combined_adjusted_objective']['Cn_width']:.3f}, respectively. No Vcem contraction is resolved on the grid, and the apparent Cn contraction cannot be quantified precisely. These are local-linear nuisance diagnostics, not fully nonlinear profiles.

An additional target-aligned discrepancy of 1% RMS leaves a gain of {aligned['1_percent']['lambda_min_gain']:.2f}x and produces a spectral ratio of {aligned['1_percent']['spectral_ratio']:.6f}, below the static baseline of {summary['operating_point']['baseline_adjusted_spectral_ratio']:.6f}. Experimental design must therefore account for model-error directions, not only generic smooth discrepancy.

The maximum fraction of samples outside the [0,1] weight interval is {100*weight_control['pooled_state_maximum_fraction_outside_unit_interval']:.1f}% for the pooled operating point and its one-sigma local nuisance scenarios, but reaches {100*weight_control['maximum_fraction_outside_unit_interval']:.1f}% across the wider E2 bootstrap stress test. Operating-point design sensitivity is therefore restricted to the convex validity domain; extrapolations beyond it are not interpreted.

## Interpretation

The scientifically defensible conclusion is negative but useful: the bounding-average pressure response can appear to strongly contract the ridge, yet that result is conditional on fabric-sharing assumptions. With expanded fabric adjustment, the ridge is not eliminated and may barely contract. A successful real experiment must either constrain the fabric variables independently or use a stress-response mechanism whose sensitivity remains distinct after those adjustments.

This is a prospective experimental-design result. It identifies a candidate pressure configuration and the fabric constraints required for a laboratory validation experiment; it does not prescribe a definitive acquisition and is not validation with observed pressure-dependent Hugin data.
"""
    (ROOT / "results" / "RESULTS.md").write_text(text)


def run_analysis(quick: bool = False) -> dict:
    _mkdirs()
    baseline = load_baseline()
    pressure_audit = pm.pressure_independence_audit(
        baseline.pooled,
        baseline.theta,
        [5.0, 10.0, 20.0, pm.REFERENCE_PRESSURE_MPA, 40.0, 60.0],
    )
    no_go = no_go_repetition_table(baseline)
    fluid_control = multi_fluid_control_table(baseline)
    pressure_curves = pressure_curve_table(baseline)
    weights = pm.patchy_weights(baseline.pooled, baseline.theta)
    weight_stress = bounding_weight_stress_table(baseline)
    jacobians = precompute_pressure_jacobians(baseline)
    design_grid = pressure_design_grid(baseline, jacobians)
    best = best_designs(design_grid)
    design = primary_design(best)
    ablation = pressure_ablation_table(baseline, jacobians, best)
    discrepancy_sensitivity = discrepancy_sensitivity_table(
        baseline, jacobians, design
    )
    reference_pressure_sensitivity = reference_pressure_sensitivity_table(baseline)
    target_aligned_discrepancy = target_aligned_discrepancy_table(
        baseline, jacobians, design
    )
    trajectory_specific = trajectory_specific_design_table(baseline)
    operating_point_sensitivity = operating_point_design_sensitivity_table(
        baseline, quick=quick
    )
    design_map = rank_one_design_map(baseline)
    candidates = candidate_observation_table(baseline)
    finite_difference = pm.finite_difference_stability(
        baseline.pooled,
        baseline.theta,
        list(design),
        **FABRIC_CONFIGS[PRIMARY_FABRIC_MODE],
    )
    bootstrap = bootstrap_pressure_design(
        baseline,
        jacobians,
        best,
        repetitions=60 if quick else 400,
    )
    bootstrap_stats = bootstrap_summary(bootstrap)
    conditional_design_selection = conditional_bootstrap_design_selection(
        baseline,
        jacobians,
        design,
        repetitions_per_block=10 if quick else 100,
    )
    profiles = nonlinear_ridge_profiles(baseline, jacobians, design, quick=quick)
    profile_widths = profile_width_summary(profiles, baseline)
    primary_matrix = selected_design_geometry(baseline, jacobians, design)

    tables = {
        "E3_pressure_independence_audit.csv": pressure_audit,
        "E3_no_go_repetition.csv": no_go,
        "E3_multi_fluid_control.csv": fluid_control,
        "E3_pressure_curves.csv": pressure_curves,
        "E3_patchy_weights.csv": weights,
        "E3_bounding_weight_stress.csv": weight_stress,
        "E3_pressure_design_grid.csv": design_grid,
        "E3_best_designs.csv": best,
        "E3_pressure_ablation.csv": ablation,
        "E3_model_discrepancy_sensitivity.csv": discrepancy_sensitivity,
        "E3_reference_pressure_sensitivity.csv": reference_pressure_sensitivity,
        "E3_target_aligned_discrepancy.csv": target_aligned_discrepancy,
        "E3_trajectory_specific_designs.csv": trajectory_specific,
        "E3_operating_point_design_sensitivity.csv": operating_point_sensitivity,
        "E3_rank_one_design_map.csv": design_map,
        "E3_candidate_observations.csv": candidates,
        "E3_finite_difference_stability.csv": finite_difference,
        "E3_bootstrap_replicates.csv": bootstrap,
        "E3_bootstrap_summary.csv": bootstrap_stats,
        "E3_conditional_design_selection.csv": conditional_design_selection,
        "E3_nonlinear_profiles.csv": profiles,
        "E3_profile_widths.csv": profile_widths,
    }
    for filename, table in tables.items():
        table.to_csv(ROOT / "results" / "tables" / filename, index=False)

    plot_no_go_and_controls(pressure_curves, no_go, fluid_control, best)
    plot_pressure_design(design_grid, best, ablation, discrepancy_sensitivity)
    plot_ridge_breaking(profiles, baseline, design)
    plot_robustness_and_direction(bootstrap, design_map, candidates)

    summary = build_summary(
        baseline,
        pressure_audit,
        no_go,
        fluid_control,
        weights,
        weight_stress,
        best,
        ablation,
        bootstrap_stats,
        profile_widths,
        finite_difference,
        discrepancy_sensitivity,
        reference_pressure_sensitivity,
        target_aligned_discrepancy,
        trajectory_specific,
        operating_point_sensitivity,
        conditional_design_selection,
        design,
        primary_matrix,
    )
    (ROOT / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    )
    write_results_markdown(summary)
    return summary


if __name__ == "__main__":
    run_analysis(quick="--quick" in sys.argv)
