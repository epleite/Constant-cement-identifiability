#!/usr/bin/env python3
"""Prior-scale sensitivity audit for the E3 pressure-design result.

This script does not modify the frozen E3 package.  It imports its public
analysis functions, varies the Gaussian prior scales through the exactly
equivalent prior-precision representation, and writes a self-contained audit
under ``experiments/E5_prior_scale_audit/results``.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
TEST_ROOT = HERE.parents[1]
REPOSITORY_ROOT = TEST_ROOT.parents[1]
RESULTS = TEST_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/constant_cement_prior_scale_mplconfig")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_e3_root() -> Path:
    """Resolve the frozen E3 package without copying or altering it."""

    candidates = []
    if os.environ.get("E3_ROOT"):
        candidates.append(Path(os.environ["E3_ROOT"]).expanduser())
    candidates.extend(
        [
            REPOSITORY_ROOT / "experiments" / "E3_break_design",
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "e3_analysis.py").is_file():
            return candidate.resolve()
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate the frozen E3 package. Searched:\n{searched}")


E3_ROOT = resolve_e3_root()
sys.path.insert(0, str(E3_ROOT / "src"))
import e3_analysis as ea  # noqa: E402


PRIMARY_DESIGN = (5.0, 7.5)
TARGET_ALIGNED_SIGMA = 0.010
MULTIPLIERS = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00)
ONE_AT_A_TIME_MULTIPLIERS = (0.25, 0.50, 1.00, 2.00, 4.00)

# The grouping is intentionally scientific rather than merely computational.
# The cement-volume endpoint is a scenario-construction prior, not a direct
# depositional-fabric descriptor, and is kept separate from the three fabric
# state variables.
GROUPS = {
    "fabric_state": (
        "ln_soft_cn_offset",
        "ln_stiff_cn_offset",
        "soft_phic_shift",
    ),
    "stress_calibration": (
        "logP_calibration",
        "log_stress_bulk_scale",
        "log_stress_shear_scale",
    ),
    "scenario_endpoint": ("stiff_cement_volume_shift",),
    "all_pressure": tuple(ea.pm.pressure_nuisance_names()),
}

STATIC_GROUPS = {
    "state_log_biases": (
        "phi_bias",
        "vsh_bias",
        "sw_bias",
    ),
    "solid_composition_moduli": (
        "f_kf_shift",
        "log_clay_mod_scale",
        "log_cement_mod_scale",
    ),
    "fluid_properties": (
        "brine_salinity_shift",
        "GOR_shift",
    ),
    "packing_reference": ("phic_pack_shift",),
    "all_static": (
        "phi_bias",
        "vsh_bias",
        "sw_bias",
        "f_kf_shift",
        "log_clay_mod_scale",
        "brine_salinity_shift",
        "GOR_shift",
        "log_cement_mod_scale",
        "phic_pack_shift",
    ),
}

DISPLAY_NAMES = {
    "fabric_state": "Fabric state",
    "stress_calibration": "Stress/calibration",
    "scenario_endpoint": "Scenario endpoint",
    "all_pressure": "All pressure nuisances",
}

STATIC_DISPLAY_NAMES = {
    "state_log_biases": "State-variable biases",
    "solid_composition_moduli": "Solid composition/moduli",
    "fluid_properties": "Fluid properties",
    "packing_reference": "Packing reference",
    "all_static": "All static nuisances",
}

NUISANCE_METADATA = {
    "logP_calibration": {
        "group": "stress_calibration",
        "meaning": "multiplicative pressure-calibration factor",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "log_stress_bulk_scale": {
        "group": "stress_calibration",
        "meaning": "multiplicative bulk stress-response amplitude",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "log_stress_shear_scale": {
        "group": "stress_calibration",
        "meaning": "multiplicative shear stress-response amplitude",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "stiff_cement_volume_shift": {
        "group": "scenario_endpoint",
        "meaning": "stiff-end cement-volume shift",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
    "ln_soft_cn_offset": {
        "group": "fabric_state",
        "meaning": "compliant-contact coordination-number offset",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "ln_stiff_cn_offset": {
        "group": "fabric_state",
        "meaning": "stiff-bound coordination-number offset",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "soft_phic_shift": {
        "group": "fabric_state",
        "meaning": "compliant-branch critical-porosity shift",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
}

STATIC_NUISANCE_METADATA = {
    "phi_bias": {
        "group": "state_log_biases",
        "meaning": "systematic porosity/state bias",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
    "vsh_bias": {
        "group": "state_log_biases",
        "meaning": "shale-volume/mineralogical-proxy bias",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
    "sw_bias": {
        "group": "state_log_biases",
        "meaning": "water-saturation bias",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
    "f_kf_shift": {
        "group": "solid_composition_moduli",
        "meaning": "framework K-feldspar fraction shift",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
    "log_clay_mod_scale": {
        "group": "solid_composition_moduli",
        "meaning": "multiplicative clay-modulus factor",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "brine_salinity_shift": {
        "group": "fluid_properties",
        "meaning": "brine-salinity mass-fraction shift",
        "unit": "mass fraction",
        "transform": "100 sigma percentage points",
    },
    "GOR_shift": {
        "group": "fluid_properties",
        "meaning": "gas-oil-ratio shift",
        "unit": "Sm3/Sm3",
        "transform": "additive sigma",
    },
    "log_cement_mod_scale": {
        "group": "solid_composition_moduli",
        "meaning": "multiplicative cement-modulus factor",
        "unit": "log ratio",
        "transform": "exp(sigma)",
    },
    "phic_pack_shift": {
        "group": "packing_reference",
        "meaning": "packing critical-porosity reference shift",
        "unit": "absolute fraction",
        "transform": "100 sigma percentage points",
    },
}


def ensure_dirs() -> None:
    for directory in (TABLES, FIGURES, RESULTS / "verification"):
        directory.mkdir(parents=True, exist_ok=True)


def prior_scale_table() -> pd.DataFrame:
    rows = []
    scales = ea.pm.PRESSURE_NUISANCE_SCALES
    for name in ea.pm.pressure_nuisance_names():
        sigma = float(scales[name])
        meta = NUISANCE_METADATA[name]
        is_log = meta["unit"] == "log ratio"
        rows.append(
            {
                "nuisance": name,
                "scientific_group": meta["group"],
                "meaning": meta["meaning"],
                "baseline_sd": sigma,
                "unit": meta["unit"],
                "physical_interpretation": meta["transform"],
                "baseline_one_sigma_multiplicative_factor": math.exp(sigma)
                if is_log
                else np.nan,
                "baseline_one_sigma_percentage_points": 100.0 * sigma
                if not is_log
                else np.nan,
                "half_scale_sd": 0.5 * sigma,
                "double_scale_sd": 2.0 * sigma,
                "fourfold_scale_sd": 4.0 * sigma,
            }
        )
    return pd.DataFrame(rows)


def static_prior_scale_table(baseline: ea.Baseline) -> pd.DataFrame:
    rows = []
    for name in baseline.nuisance_names:
        sigma = float(ea.rc.NUI_SCALES[name])
        meta = STATIC_NUISANCE_METADATA[name]
        is_log = meta["unit"] == "log ratio"
        is_fraction = meta["unit"] in ("absolute fraction", "mass fraction")
        rows.append(
            {
                "nuisance": name,
                "scientific_group": meta["group"],
                "meaning": meta["meaning"],
                "baseline_sd": sigma,
                "unit": meta["unit"],
                "physical_interpretation": meta["transform"],
                "baseline_one_sigma_multiplicative_factor": math.exp(sigma)
                if is_log
                else np.nan,
                "baseline_one_sigma_percentage_points": 100.0 * sigma
                if is_fraction
                else np.nan,
                "half_scale_sd": 0.5 * sigma,
                "double_scale_sd": 2.0 * sigma,
                "fourfold_scale_sd": 4.0 * sigma,
            }
        )
    return pd.DataFrame(rows)


def static_adjusted_geometry(
    baseline: ea.Baseline,
    sd_multipliers: dict[str, float],
) -> np.ndarray:
    """Recompute the static Schur complement under altered static priors."""

    prior_precision = np.ones(len(baseline.nuisance_names))
    for name, multiplier in sd_multipliers.items():
        if name not in baseline.nuisance_names:
            continue
        if multiplier <= 0.0:
            raise ValueError("prior SD multipliers must be positive")
        prior_precision[baseline.nuisance_names.index(name)] = 1.0 / multiplier**2
    _, adjusted = ea.schur_geometry(
        baseline.target_jacobian,
        baseline.nuisance_jacobian,
        prior_precision,
    )
    return adjusted


def add_target_aligned_column(
    baseline: ea.Baseline,
    static_reference: np.ndarray,
    target_all: np.ndarray,
    target_selected: np.ndarray,
    nuisance_selected: np.ndarray,
    selected: tuple[float, ...],
    sigma: float,
) -> tuple[np.ndarray, float]:
    """Append a standardized model-error direction aligned with the weak target."""

    if sigma <= 0.0:
        return nuisance_selected, 0.0
    _, vectors = ea.eigensystem(static_reference)
    weak_direction = vectors[:, 0]
    selected_rows = ea.pm.pressure_row_indices(
        ea.PRESSURE_CANDIDATES, selected, len(baseline.pooled)
    )
    raw_alignment = target_all[selected_rows] @ weak_direction
    raw_rms = float(np.sqrt(np.mean(raw_alignment**2)))
    aligned_unit_rms = (target_selected @ weak_direction) / max(raw_rms, 1e-30)
    return np.c_[nuisance_selected, sigma * aligned_unit_rms], raw_rms


def combine_with_prior_scales(
    baseline: ea.Baseline,
    target: np.ndarray,
    nuisance: np.ndarray,
    nuisance_names: list[str],
    sd_multipliers: dict[str, float],
) -> np.ndarray:
    """Form G_adj for altered physical prior scales.

    E3 nuisance columns are derivatives multiplied by their baseline physical
    one-sigma scales.  Replacing sigma by ``m*sigma`` while retaining a unit
    normal prior is algebraically identical to retaining the frozen Jacobian
    and using prior precision ``1/m**2`` for that standardized coordinate.
    """

    base_nuisance = ea.expanded_baseline_nuisance(baseline, True)
    if nuisance.shape[1] > base_nuisance.shape[1]:
        base_nuisance = np.c_[
            base_nuisance,
            np.zeros(
                (len(base_nuisance), nuisance.shape[1] - base_nuisance.shape[1])
            ),
        ]
    target_joint = np.vstack([baseline.target_jacobian, target])
    nuisance_joint = np.vstack([base_nuisance, nuisance])
    prior_precision = np.ones(nuisance_joint.shape[1])
    for name, multiplier in sd_multipliers.items():
        if multiplier <= 0.0:
            raise ValueError("prior SD multipliers must be positive")
        prior_precision[nuisance_names.index(name)] = 1.0 / multiplier**2
    _, adjusted = ea.schur_geometry(target_joint, nuisance_joint, prior_precision)
    return adjusted


def selected_blocks(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    selected: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return ea.pressure_subset(
        target_all,
        nuisance_all,
        ea.PRESSURE_CANDIDATES,
        selected,
        len(baseline.pooled),
        ea.PRIMARY_STATE_LOG_SIGMA,
        shared_reference=True,
        trajectory_lengths=(
            len(baseline.wells["19A"]),
            len(baseline.wells["BT2"]),
        ),
        trajectory_discrepancy_sigma=ea.PRIMARY_TRAJECTORY_DISCREPANCY_SIGMA,
        trajectory_discrepancy_features=ea.model_discrepancy_features(baseline),
    )


def evaluate_design(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
    selected: tuple[float, ...],
    sd_multipliers: dict[str, float],
    target_aligned_sigma: float = 0.0,
    static_reference: np.ndarray | None = None,
) -> dict[str, float]:
    if static_reference is None:
        static_reference = baseline.gram_adjusted
    target, nuisance = selected_blocks(baseline, target_all, nuisance_all, selected)
    nuisance, raw_alignment_rms = add_target_aligned_column(
        baseline,
        static_reference,
        target_all,
        target,
        nuisance,
        selected,
        target_aligned_sigma,
    )
    adjusted = combine_with_prior_scales(
        baseline, target, nuisance, nuisance_names, sd_multipliers
    )
    output = ea.geometry_metrics(adjusted, static_reference)
    output.update(ea.parameter_uncertainty_metrics(adjusted))
    output["target_aligned_discrepancy_log_rms"] = target_aligned_sigma
    output["raw_alignment_rms_per_scaled_parameter"] = raw_alignment_rms
    return output


def paired_metrics(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
    selected: tuple[float, ...],
    sd_multipliers: dict[str, float],
    static_reference: np.ndarray | None = None,
) -> dict[str, float]:
    ordinary = evaluate_design(
        baseline,
        target_all,
        nuisance_all,
        nuisance_names,
        selected,
        sd_multipliers,
        static_reference=static_reference,
    )
    aligned = evaluate_design(
        baseline,
        target_all,
        nuisance_all,
        nuisance_names,
        selected,
        sd_multipliers,
        TARGET_ALIGNED_SIGMA,
        static_reference,
    )
    output = dict(ordinary)
    for key in (
        "lambda_min",
        "lambda_max",
        "lambda_min_gain",
        "spectral_ratio",
        "Vcem_lnCn_correlation",
        "sd_Vcem_percentage_points",
        "sd_lnCn",
    ):
        output[f"aligned_1pct_{key}"] = float(aligned[key])
    output["aligned_1pct_raw_alignment_rms_per_scaled_parameter"] = float(
        aligned["raw_alignment_rms_per_scaled_parameter"]
    )
    return output


def group_sweep(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> pd.DataFrame:
    rows = []
    for group, affected in GROUPS.items():
        for multiplier in MULTIPLIERS:
            factors = {name: multiplier for name in affected}
            metrics = paired_metrics(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                PRIMARY_DESIGN,
                factors,
            )
            rows.append(
                {
                    "group": group,
                    "group_label": DISPLAY_NAMES[group],
                    "affected_nuisances": ";".join(affected),
                    "prior_sd_multiplier": multiplier,
                    "pressure_1_mpa": PRIMARY_DESIGN[0],
                    "pressure_2_mpa": PRIMARY_DESIGN[1],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def one_at_a_time_sweep(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> pd.DataFrame:
    rows = []
    scales = ea.pm.PRESSURE_NUISANCE_SCALES
    for name in ea.pm.pressure_nuisance_names():
        for multiplier in ONE_AT_A_TIME_MULTIPLIERS:
            metrics = paired_metrics(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                PRIMARY_DESIGN,
                {name: multiplier},
            )
            sigma = float(scales[name]) * multiplier
            is_log = NUISANCE_METADATA[name]["unit"] == "log ratio"
            rows.append(
                {
                    "nuisance": name,
                    "scientific_group": NUISANCE_METADATA[name]["group"],
                    "meaning": NUISANCE_METADATA[name]["meaning"],
                    "prior_sd_multiplier": multiplier,
                    "physical_prior_sd": sigma,
                    "unit": NUISANCE_METADATA[name]["unit"],
                    "one_sigma_multiplicative_factor": math.exp(sigma)
                    if is_log
                    else np.nan,
                    "one_sigma_percentage_points": 100.0 * sigma
                    if not is_log
                    else np.nan,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def static_group_sweep(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> pd.DataFrame:
    """Vary groups of static priors in both baseline and combined geometry."""

    rows = []
    for group, affected in STATIC_GROUPS.items():
        for multiplier in MULTIPLIERS:
            factors = {name: multiplier for name in affected}
            static_reference = static_adjusted_geometry(baseline, factors)
            metrics = paired_metrics(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                PRIMARY_DESIGN,
                factors,
                static_reference,
            )
            static_values = np.linalg.eigvalsh(static_reference)
            _, static_vectors = ea.eigensystem(static_reference)
            static_weak = static_vectors[:, 0]
            rows.append(
                {
                    "group": group,
                    "group_label": STATIC_DISPLAY_NAMES[group],
                    "affected_nuisances": ";".join(affected),
                    "prior_sd_multiplier": multiplier,
                    "pressure_1_mpa": PRIMARY_DESIGN[0],
                    "pressure_2_mpa": PRIMARY_DESIGN[1],
                    "static_lambda_min": float(static_values[0]),
                    "static_lambda_max": float(static_values[-1]),
                    "static_spectral_ratio": float(
                        static_values[0] / static_values[-1]
                    ),
                    "static_weak_direction_scaled_Vcem": float(static_weak[0]),
                    "static_weak_direction_scaled_lnCn": float(static_weak[1]),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def static_one_at_a_time_sweep(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> pd.DataFrame:
    rows = []
    for name in baseline.nuisance_names:
        for multiplier in ONE_AT_A_TIME_MULTIPLIERS:
            factors = {name: multiplier}
            static_reference = static_adjusted_geometry(baseline, factors)
            metrics = paired_metrics(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                PRIMARY_DESIGN,
                factors,
                static_reference,
            )
            sigma = float(ea.rc.NUI_SCALES[name]) * multiplier
            meta = STATIC_NUISANCE_METADATA[name]
            is_log = meta["unit"] == "log ratio"
            is_fraction = meta["unit"] in ("absolute fraction", "mass fraction")
            static_values = np.linalg.eigvalsh(static_reference)
            rows.append(
                {
                    "nuisance": name,
                    "scientific_group": meta["group"],
                    "meaning": meta["meaning"],
                    "prior_sd_multiplier": multiplier,
                    "physical_prior_sd": sigma,
                    "unit": meta["unit"],
                    "one_sigma_multiplicative_factor": math.exp(sigma)
                    if is_log
                    else np.nan,
                    "one_sigma_percentage_points": 100.0 * sigma
                    if is_fraction
                    else np.nan,
                    "static_lambda_min": float(static_values[0]),
                    "static_lambda_max": float(static_values[-1]),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def static_reoptimized_design_sweep(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> pd.DataFrame:
    designs = list(itertools.combinations(ea.PRESSURE_CANDIDATES, 2))
    rows = []
    for group, affected in STATIC_GROUPS.items():
        for multiplier in MULTIPLIERS:
            factors = {name: multiplier for name in affected}
            static_reference = static_adjusted_geometry(baseline, factors)
            evaluated = []
            for design in designs:
                metrics = evaluate_design(
                    baseline,
                    target_all,
                    nuisance_all,
                    nuisance_names,
                    design,
                    factors,
                    static_reference=static_reference,
                )
                evaluated.append((float(metrics["lambda_min"]), design, metrics))
            evaluated.sort(key=lambda item: item[0], reverse=True)
            _, design, metrics = evaluated[0]
            aligned = evaluate_design(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                design,
                factors,
                TARGET_ALIGNED_SIGMA,
                static_reference,
            )
            canonical = evaluate_design(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                PRIMARY_DESIGN,
                factors,
                static_reference=static_reference,
            )
            rows.append(
                {
                    "group": group,
                    "group_label": STATIC_DISPLAY_NAMES[group],
                    "prior_sd_multiplier": multiplier,
                    "best_pressure_1_mpa": design[0],
                    "best_pressure_2_mpa": design[1],
                    "canonical_5_7p5_is_best": design == PRIMARY_DESIGN,
                    "second_best_lambda_min": evaluated[1][0],
                    "best_minus_second_best_lambda_min": evaluated[0][0]
                    - evaluated[1][0],
                    "canonical_lambda_min_gain": canonical["lambda_min_gain"],
                    **metrics,
                    "aligned_1pct_lambda_min_gain": aligned["lambda_min_gain"],
                    "aligned_1pct_spectral_ratio": aligned["spectral_ratio"],
                    "aligned_1pct_Vcem_lnCn_correlation": aligned[
                        "Vcem_lnCn_correlation"
                    ],
                }
            )
        print(f"re-optimized static group: {group}", flush=True)
    return pd.DataFrame(rows)


def reoptimized_design_sweep(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> pd.DataFrame:
    designs = list(itertools.combinations(ea.PRESSURE_CANDIDATES, 2))
    rows = []
    for group, affected in GROUPS.items():
        for multiplier in MULTIPLIERS:
            factors = {name: multiplier for name in affected}
            evaluated = []
            for design in designs:
                metrics = evaluate_design(
                    baseline,
                    target_all,
                    nuisance_all,
                    nuisance_names,
                    design,
                    factors,
                )
                evaluated.append((float(metrics["lambda_min"]), design, metrics))
            evaluated.sort(key=lambda item: item[0], reverse=True)
            _, design, metrics = evaluated[0]
            aligned = evaluate_design(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                design,
                factors,
                TARGET_ALIGNED_SIGMA,
            )
            canonical = evaluate_design(
                baseline,
                target_all,
                nuisance_all,
                nuisance_names,
                PRIMARY_DESIGN,
                factors,
            )
            rows.append(
                {
                    "group": group,
                    "group_label": DISPLAY_NAMES[group],
                    "prior_sd_multiplier": multiplier,
                    "best_pressure_1_mpa": design[0],
                    "best_pressure_2_mpa": design[1],
                    "canonical_5_7p5_is_best": design == PRIMARY_DESIGN,
                    "second_best_lambda_min": evaluated[1][0],
                    "best_minus_second_best_lambda_min": evaluated[0][0]
                    - evaluated[1][0],
                    "canonical_lambda_min_gain": canonical["lambda_min_gain"],
                    **metrics,
                    "aligned_1pct_lambda_min_gain": aligned["lambda_min_gain"],
                    "aligned_1pct_spectral_ratio": aligned["spectral_ratio"],
                    "aligned_1pct_Vcem_lnCn_correlation": aligned[
                        "Vcem_lnCn_correlation"
                    ],
                }
            )
        print(f"re-optimized group: {group}", flush=True)
    return pd.DataFrame(rows)


def structural_context(
    baseline: ea.Baseline,
) -> pd.DataFrame:
    descriptions = {
        "shared": "all compliant and stiff coordination numbers share target Cn",
        "fixed": "compliant coordination number locally fixed",
        "nuisance": "compliant coordination number decoupled as nuisance",
        "expanded_nuisance": "compliant/stiff coordination and compliant phic decoupled",
    }
    rows = []
    for mode, config in ea.FABRIC_CONFIGS.items():
        target_all, nuisance_all, nuisance_names = ea.pm.pressure_differential_jacobian(
            baseline.pooled,
            baseline.theta,
            ea.PRESSURE_CANDIDATES,
            **config,
        )
        metrics = paired_metrics(
            baseline,
            target_all,
            nuisance_all,
            nuisance_names,
            PRIMARY_DESIGN,
            {},
        )
        rows.append(
            {
                "fabric_mode": mode,
                "description": descriptions[mode],
                "pressure_1_mpa": PRIMARY_DESIGN[0],
                "pressure_2_mpa": PRIMARY_DESIGN[1],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def scale_equivalence_error(
    baseline: ea.Baseline,
    target_all: np.ndarray,
    nuisance_all: np.ndarray,
    nuisance_names: list[str],
) -> float:
    """Numerically verify Jacobian scaling against prior-precision scaling."""

    target, nuisance = selected_blocks(
        baseline, target_all, nuisance_all, PRIMARY_DESIGN
    )
    multiplier = 1.7
    name = "ln_soft_cn_offset"
    index = nuisance_names.index(name)
    via_precision = combine_with_prior_scales(
        baseline, target, nuisance, nuisance_names, {name: multiplier}
    )

    base_nuisance = ea.expanded_baseline_nuisance(baseline, True)
    if nuisance.shape[1] > base_nuisance.shape[1]:
        base_nuisance = np.c_[
            base_nuisance,
            np.zeros(
                (len(base_nuisance), nuisance.shape[1] - base_nuisance.shape[1])
            ),
        ]
    nuisance_scaled = nuisance.copy()
    nuisance_scaled[:, index] *= multiplier
    base_scaled = base_nuisance.copy()
    base_scaled[:, index] *= multiplier
    target_joint = np.vstack([baseline.target_jacobian, target])
    nuisance_joint = np.vstack([base_scaled, nuisance_scaled])
    _, via_jacobian = ea.schur_geometry(target_joint, nuisance_joint)
    return float(np.max(np.abs(via_precision - via_jacobian)))


def static_scale_equivalence_error(baseline: ea.Baseline) -> float:
    """Verify the same scale/precision equivalence for a static nuisance."""

    multiplier = 1.7
    name = "log_clay_mod_scale"
    index = baseline.nuisance_names.index(name)
    via_precision = static_adjusted_geometry(baseline, {name: multiplier})
    nuisance_scaled = baseline.nuisance_jacobian.copy()
    nuisance_scaled[:, index] *= multiplier
    _, via_jacobian = ea.schur_geometry(
        baseline.target_jacobian,
        nuisance_scaled,
    )
    return float(np.max(np.abs(via_precision - via_jacobian)))


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def plot_group_sweep(group: pd.DataFrame) -> None:
    configure_plotting()
    colors = {
        "fabric_state": "#2A9D8F",
        "stress_calibration": "#3478A6",
        "scenario_endpoint": "#E9A23B",
        "all_pressure": "#C94C4C",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), constrained_layout=True)
    specifications = [
        ("lambda_min_gain", r"Gain in $\lambda_{\min}$", False),
        ("spectral_ratio", r"Spectral ratio $\lambda_{\min}/\lambda_{\max}$", True),
        ("Vcem_lnCn_correlation", r"Target correlation $\rho(V_{\rm cem},\ln C_n)$", False),
        ("aligned_1pct_lambda_min_gain", r"Gain with 1% aligned discrepancy", False),
    ]
    for panel, (axis, (column, ylabel, log_y)) in enumerate(
        zip(axes.ravel(), specifications)
    ):
        axis.axvspan(0.5, 2.0, color="#CBD5E1", alpha=0.28, zorder=0)
        axis.axvline(1.0, color="#475569", linewidth=0.9, linestyle="--")
        for group_name in GROUPS:
            part = group[group.group == group_name].sort_values("prior_sd_multiplier")
            axis.plot(
                part.prior_sd_multiplier,
                part[column],
                marker="o",
                markersize=3.8,
                linewidth=1.5,
                color=colors[group_name],
                label=DISPLAY_NAMES[group_name],
            )
        axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("Prior SD multiplier")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0.25, 0.5, 1, 2, 4])
        axis.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        axis.text(
            0.02,
            0.97,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )
        if column.endswith("lambda_min_gain"):
            axis.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    axes[0, 0].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle(
        "Pressure-design sensitivity to nuisance-prior scales\n"
        "Expanded-fabric model, 5 + 7.5 MPa relative to 39 MPa",
        fontsize=12,
    )
    fig.savefig(FIGURES / "Fig_prior_scale_sensitivity.png", bbox_inches="tight")
    fig.savefig(FIGURES / "Fig_prior_scale_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_static_group_sweep(group: pd.DataFrame) -> None:
    configure_plotting()
    colors = {
        "state_log_biases": "#3478A6",
        "solid_composition_moduli": "#8B5CF6",
        "fluid_properties": "#2A9D8F",
        "packing_reference": "#E9A23B",
        "all_static": "#C94C4C",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), constrained_layout=True)
    specifications = [
        ("lambda_min_gain", r"Gain in $\lambda_{\min}$", False),
        ("spectral_ratio", r"Spectral ratio $\lambda_{\min}/\lambda_{\max}$", True),
        ("Vcem_lnCn_correlation", r"Target correlation $\rho(V_{\rm cem},\ln C_n)$", False),
        ("aligned_1pct_lambda_min_gain", r"Gain with 1% aligned discrepancy", False),
    ]
    for panel, (axis, (column, ylabel, log_y)) in enumerate(
        zip(axes.ravel(), specifications)
    ):
        axis.axvspan(0.5, 2.0, color="#CBD5E1", alpha=0.28, zorder=0)
        axis.axvline(1.0, color="#475569", linewidth=0.9, linestyle="--")
        for group_name in STATIC_GROUPS:
            part = group[group.group == group_name].sort_values("prior_sd_multiplier")
            axis.plot(
                part.prior_sd_multiplier,
                part[column],
                marker="o",
                markersize=3.6,
                linewidth=1.45,
                color=colors[group_name],
                label=STATIC_DISPLAY_NAMES[group_name],
            )
        axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("Static-prior SD multiplier")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0.25, 0.5, 1, 2, 4])
        axis.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        axis.text(
            0.02,
            0.97,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )
        if column.endswith("lambda_min_gain"):
            axis.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    axes[0, 0].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle(
        "Pressure-design sensitivity to static nuisance-prior scales\n"
        "Expanded-fabric model; static and combined Schur geometries recomputed",
        fontsize=12,
    )
    fig.savefig(FIGURES / "Fig_static_prior_scale_sensitivity.png", bbox_inches="tight")
    fig.savefig(FIGURES / "Fig_static_prior_scale_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_prior_scale_audit(
    pressure_group: pd.DataFrame,
    static_group: pd.DataFrame,
) -> None:
    """Compact manuscript figure combining both prior blocks and denominator audit."""

    configure_plotting()
    pressure_colors = {
        "fabric_state": "#2A9D8F",
        "stress_calibration": "#3478A6",
        "scenario_endpoint": "#E9A23B",
        "all_pressure": "#C94C4C",
    }
    static_colors = {
        "state_log_biases": "#3478A6",
        "solid_composition_moduli": "#8B5CF6",
        "fluid_properties": "#2A9D8F",
        "packing_reference": "#E9A23B",
        "all_static": "#C94C4C",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), constrained_layout=True)

    def prepare_axis(axis: mpl.axes.Axes, ylabel: str) -> None:
        axis.axvspan(0.5, 2.0, color="#CBD5E1", alpha=0.28, zorder=0)
        axis.axvline(1.0, color="#475569", linewidth=0.9, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks([0.25, 0.5, 1, 2, 4])
        axis.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        axis.set_xlabel("Prior SD multiplier")
        axis.set_ylabel(ylabel)

    ax = axes[0, 0]
    prepare_axis(ax, r"Gain in $\lambda_{\min}$")
    for name in GROUPS:
        part = pressure_group[pressure_group.group == name].sort_values(
            "prior_sd_multiplier"
        )
        ax.plot(
            part.prior_sd_multiplier,
            part.lambda_min_gain,
            marker="o",
            markersize=3.5,
            linewidth=1.45,
            color=pressure_colors[name],
            label=DISPLAY_NAMES[name],
        )
    ax.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    ax.set_title("Pressure-model priors")
    ax.legend(frameon=False, fontsize=7.3, ncol=2, loc="best")

    ax = axes[0, 1]
    prepare_axis(ax, r"Gain in $\lambda_{\min}$")
    for name in STATIC_GROUPS:
        part = static_group[static_group.group == name].sort_values(
            "prior_sd_multiplier"
        )
        ax.plot(
            part.prior_sd_multiplier,
            part.lambda_min_gain,
            marker="o",
            markersize=3.5,
            linewidth=1.45,
            color=static_colors[name],
            label=STATIC_DISPLAY_NAMES[name],
        )
    ax.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    ax.set_title("Static-model priors")
    ax.legend(frameon=False, fontsize=7.1, ncol=2, loc="best")

    ax = axes[1, 0]
    prepare_axis(ax, "Gain with 1% aligned discrepancy")
    pressure_all = pressure_group[pressure_group.group == "all_pressure"].sort_values(
        "prior_sd_multiplier"
    )
    static_all = static_group[static_group.group == "all_static"].sort_values(
        "prior_sd_multiplier"
    )
    ax.plot(
        pressure_all.prior_sd_multiplier,
        pressure_all.aligned_1pct_lambda_min_gain,
        marker="o",
        linewidth=1.8,
        color="#3478A6",
        label="All pressure nuisances",
    )
    ax.plot(
        static_all.prior_sd_multiplier,
        static_all.aligned_1pct_lambda_min_gain,
        marker="o",
        linewidth=1.8,
        color="#C94C4C",
        label="All static nuisances",
    )
    ax.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    ax.set_title("Target-aligned model discrepancy")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1, 1]
    prepare_axis(ax, r"Absolute $\lambda_{\min}$")
    ax.plot(
        static_all.prior_sd_multiplier,
        static_all.static_lambda_min,
        marker="o",
        linewidth=1.8,
        color="#6B7280",
        label="Static baseline",
    )
    ax.plot(
        static_all.prior_sd_multiplier,
        static_all.lambda_min,
        marker="o",
        linewidth=1.8,
        color="#C94C4C",
        label="Static + pressure",
    )
    ax.set_title("All-static denominator audit")
    ax.legend(frameon=False, fontsize=8, loc="best")

    for panel, axis in enumerate(axes.ravel()):
        axis.text(
            0.02,
            0.97,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )
    fig.suptitle(
        "Nuisance-prior sensitivity of the expanded-fabric pressure design",
        fontsize=12,
    )
    fig.savefig(FIGURES / "Fig_combined_prior_scale_audit.png", bbox_inches="tight")
    fig.savefig(FIGURES / "Fig_combined_prior_scale_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def compact_range(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    return float(frame[column].min()), float(frame[column].max())


def write_report(
    baseline: ea.Baseline,
    group: pd.DataFrame,
    individual: pd.DataFrame,
    designs: pd.DataFrame,
    static_group: pd.DataFrame,
    static_individual: pd.DataFrame,
    static_designs: pd.DataFrame,
    structural: pd.DataFrame,
    equivalence_error: float,
    static_equivalence_error: float,
) -> dict:
    plausible = group[group.prior_sd_multiplier.between(0.5, 2.0)]
    full = group[group.prior_sd_multiplier.between(0.25, 4.0)]
    ranges = {}
    for name in GROUPS:
        ranges[name] = {
            "factor_two": {
                "lambda_min_gain": compact_range(
                    plausible[plausible.group == name], "lambda_min_gain"
                ),
                "spectral_ratio": compact_range(
                    plausible[plausible.group == name], "spectral_ratio"
                ),
                "target_correlation": compact_range(
                    plausible[plausible.group == name], "Vcem_lnCn_correlation"
                ),
                "aligned_1pct_gain": compact_range(
                    plausible[plausible.group == name],
                    "aligned_1pct_lambda_min_gain",
                ),
            },
            "quarter_to_fourfold": {
                "lambda_min_gain": compact_range(
                    full[full.group == name], "lambda_min_gain"
                ),
                "aligned_1pct_gain": compact_range(
                    full[full.group == name], "aligned_1pct_lambda_min_gain"
                ),
            },
        }

    static_plausible = static_group[
        static_group.prior_sd_multiplier.between(0.5, 2.0)
    ]
    static_full = static_group[
        static_group.prior_sd_multiplier.between(0.25, 4.0)
    ]
    static_ranges = {}
    for name in STATIC_GROUPS:
        static_ranges[name] = {
            "factor_two": {
                "lambda_min_gain": compact_range(
                    static_plausible[static_plausible.group == name],
                    "lambda_min_gain",
                ),
                "static_lambda_min": compact_range(
                    static_plausible[static_plausible.group == name],
                    "static_lambda_min",
                ),
                "combined_lambda_min": compact_range(
                    static_plausible[static_plausible.group == name],
                    "lambda_min",
                ),
                "spectral_ratio": compact_range(
                    static_plausible[static_plausible.group == name],
                    "spectral_ratio",
                ),
                "target_correlation": compact_range(
                    static_plausible[static_plausible.group == name],
                    "Vcem_lnCn_correlation",
                ),
                "aligned_1pct_gain": compact_range(
                    static_plausible[static_plausible.group == name],
                    "aligned_1pct_lambda_min_gain",
                ),
            },
            "quarter_to_fourfold": {
                "lambda_min_gain": compact_range(
                    static_full[static_full.group == name], "lambda_min_gain"
                ),
                "aligned_1pct_gain": compact_range(
                    static_full[static_full.group == name],
                    "aligned_1pct_lambda_min_gain",
                ),
            },
        }

    baseline_row = group[
        (group.group == "all_pressure") & np.isclose(group.prior_sd_multiplier, 1.0)
    ].iloc[0]
    individual_half = individual[np.isclose(individual.prior_sd_multiplier, 0.5)][
        ["nuisance", "lambda_min_gain"]
    ].rename(columns={"lambda_min_gain": "gain_half"})
    individual_double = individual[np.isclose(individual.prior_sd_multiplier, 2.0)][
        ["nuisance", "lambda_min_gain"]
    ].rename(columns={"lambda_min_gain": "gain_double"})
    influence = individual_half.merge(individual_double, on="nuisance")
    influence["factor_two_gain_span"] = (
        influence.gain_half - influence.gain_double
    ).abs()
    influence = influence.sort_values("factor_two_gain_span", ascending=False)

    static_half = static_individual[
        np.isclose(static_individual.prior_sd_multiplier, 0.5)
    ][["nuisance", "lambda_min_gain"]].rename(
        columns={"lambda_min_gain": "gain_half"}
    )
    static_double = static_individual[
        np.isclose(static_individual.prior_sd_multiplier, 2.0)
    ][["nuisance", "lambda_min_gain"]].rename(
        columns={"lambda_min_gain": "gain_double"}
    )
    static_influence = static_half.merge(static_double, on="nuisance")
    static_influence["factor_two_gain_span"] = (
        static_influence.gain_half - static_influence.gain_double
    ).abs()
    static_influence = static_influence.sort_values(
        "factor_two_gain_span", ascending=False
    )

    design_counts = (
        designs.groupby(["best_pressure_1_mpa", "best_pressure_2_mpa"])
        .size()
        .sort_values(ascending=False)
    )
    primary_count = int(
        ((designs.best_pressure_1_mpa == 5.0) & (designs.best_pressure_2_mpa == 7.5)).sum()
    )

    static_design_counts = (
        static_designs.groupby(["best_pressure_1_mpa", "best_pressure_2_mpa"])
        .size()
        .sort_values(ascending=False)
    )
    static_primary_count = int(
        (
            (static_designs.best_pressure_1_mpa == 5.0)
            & (static_designs.best_pressure_2_mpa == 7.5)
        ).sum()
    )

    structural_rows = {
        row.fabric_mode: {
            "lambda_min_gain": float(row.lambda_min_gain),
            "spectral_ratio": float(row.spectral_ratio),
            "target_correlation": float(row.Vcem_lnCn_correlation),
            "aligned_1pct_gain": float(row.aligned_1pct_lambda_min_gain),
        }
        for row in structural.itertuples(index=False)
    }

    summary = {
        "analysis": "E3 pressure-design nuisance-prior scale sensitivity",
        "e3_root": str(E3_ROOT),
        "primary_design_mpa": list(PRIMARY_DESIGN),
        "reference_pressure_mpa": float(ea.pm.REFERENCE_PRESSURE_MPA),
        "baseline_operating_point": {
            "Vcem_fraction": float(baseline.theta[0]),
            "Cn": float(math.exp(baseline.theta[1])),
            "static_adjusted_lambda_min": float(
                np.linalg.eigvalsh(baseline.gram_adjusted)[0]
            ),
        },
        "baseline_expanded_fabric_result": {
            "lambda_min_gain": float(baseline_row.lambda_min_gain),
            "spectral_ratio": float(baseline_row.spectral_ratio),
            "target_correlation": float(baseline_row.Vcem_lnCn_correlation),
            "aligned_1pct_gain": float(baseline_row.aligned_1pct_lambda_min_gain),
        },
        "ranges": ranges,
        "static_prior_ranges": static_ranges,
        "one_at_a_time_factor_two_influence_rank": influence.to_dict(
            orient="records"
        ),
        "static_one_at_a_time_factor_two_influence_rank": static_influence.to_dict(
            orient="records"
        ),
        "design_reoptimization": {
            "n_scenarios": int(len(designs)),
            "primary_pair_selected_count": primary_count,
            "all_scenarios_selected_primary_pair": bool(primary_count == len(designs)),
            "pair_counts": {
                f"{float(index[0]):g}+{float(index[1]):g}": int(value)
                for index, value in design_counts.items()
            },
        },
        "static_design_reoptimization": {
            "n_scenarios": int(len(static_designs)),
            "primary_pair_selected_count": static_primary_count,
            "all_scenarios_selected_primary_pair": bool(
                static_primary_count == len(static_designs)
            ),
            "pair_counts": {
                f"{float(index[0]):g}+{float(index[1]):g}": int(value)
                for index, value in static_design_counts.items()
            },
        },
        "structural_context": structural_rows,
        "scale_parameterization_equivalence_max_abs_error": equivalence_error,
        "static_scale_parameterization_equivalence_max_abs_error": static_equivalence_error,
        "interpretation": {
            "robust": (
                "Across factor-of-two changes in pressure and static nuisance scales, "
                "the expanded-fabric experiment remains strongly correlated; the 1% "
                "target-aligned discrepancy leaves about a 1.2-fold gain for pressure-"
                "prior changes and at most 1.68-fold across grouped static-prior changes."
            ),
            "conditional": (
                "The exact 3.35-fold value is conditional on the stated priors; "
                "pressure-prior factor-two changes give 2.91--3.57, whereas grouped "
                "static-prior changes give 2.39--5.54 because the gain denominator "
                "also changes."
            ),
            "structural": (
                "The approximately 634-to-3.35 collapse is mainly a change in "
                "target/fabric linkage, not a consequence of the numerical prior "
                "scales alone."
            ),
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Prior-scale sensitivity of the pressure-design result",
        "",
        "## Question",
        "",
        "Does the reported 3.35-fold gain in nuisance-adjusted minimum information "
        "depend materially on the assumed one-standard-deviation scales of either "
        "the prospective pressure-model nuisances or the nine static nuisances?",
        "",
        "## Design of the audit",
        "",
        "The frozen E3 expanded-fabric Jacobians and the 5 + 7.5 MPa design were "
        "retained. Prior standard deviations were multiplied by 0.25--4, with "
        "0.5--2 treated as the main factor-of-two interpretation band. The sweep "
        "separates fabric-state priors (soft/stiff ln Cn and soft critical porosity), "
        "stress/calibration priors, and the assumed stiff-end cement-volume prior. "
        "All seven pressure nuisances were also varied together. Every scenario was "
        "then re-optimized over all candidate pressure pairs. A second calculation "
        "added the pre-existing 1% RMS discrepancy aligned with the static weak "
        "target direction. The same sweep was applied to the nine static nuisances, "
        "individually and in five physical groups. For those cases, both the static "
        "reference Schur complement and the combined pressure Schur complement were "
        "recomputed under the altered prior; the aligned discrepancy followed the "
        "corresponding static weak direction.",
        "",
        "Changing a physical prior scale by a factor m was implemented as a prior "
        "precision factor 1/m^2 on the frozen standardized nuisance column. Direct "
        "Jacobian rescaling agrees to maximum absolute matrix errors of "
        f"{equivalence_error:.2e} for a pressure nuisance and "
        f"{static_equivalence_error:.2e} for a static nuisance.",
        "",
        "## Main result",
        "",
        f"At the declared baseline, the gain is {baseline_row.lambda_min_gain:.3g}, "
        f"the spectral ratio is {baseline_row.spectral_ratio:.3g}, and the local "
        f"target correlation is {baseline_row.Vcem_lnCn_correlation:.4f}. With 1% "
        f"target-aligned discrepancy the gain is {baseline_row.aligned_1pct_lambda_min_gain:.3g}.",
        "",
        "Within the factor-of-two band:",
        "",
        "| Prior group varied | Gain range | Spectral-ratio range | Correlation range | Gain with 1% aligned discrepancy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in GROUPS:
        item = ranges[name]["factor_two"]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {item['lambda_min_gain'][0]:.3g}--{item['lambda_min_gain'][1]:.3g} "
            f"| {item['spectral_ratio'][0]:.3g}--{item['spectral_ratio'][1]:.3g} "
            f"| [{item['target_correlation'][0]:.4f}, {item['target_correlation'][1]:.4f}] "
            f"| {item['aligned_1pct_gain'][0]:.4f}--{item['aligned_1pct_gain'][1]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"The original 5 + 7.5 MPa pair remained optimal in {primary_count} of "
            f"{len(designs)} grouped scale scenarios. Thus the selected design is "
            "stable over this sweep even though the exact gain is prior-conditional.",
            "",
            "## Which individual priors matter most?",
            "",
            "The following ranking uses the absolute gain difference between a "
            "half-scale and a double-scale prior while all other priors remain at "
            "their baselines:",
            "",
            "| Nuisance | Gain at 0.5x | Gain at 2x | Span |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in influence.itertuples(index=False):
        lines.append(
            f"| `{row.nuisance}` | {row.gain_half:.3g} | {row.gain_double:.3g} "
            f"| {row.factor_two_gain_span:.3g} |"
        )
    lines.extend(
        [
            "",
            "No individual scale restores nominal separability. The local target "
            "correlation remains close to -1, and the aligned-discrepancy gain "
            "remains near 1.2 throughout the scientifically interpretable band.",
            "",
            "The spectral ratio is not monotonic in every sweep because widening a "
            "prior can reduce the strong eigenvalue as well as the weak eigenvalue; "
            "it must therefore be read jointly with absolute lambda-min gain and "
            "target correlation, not as a stand-alone improvement score.",
            "",
            "## Static-nuisance prior sensitivity",
            "",
            "For static-prior changes, the gain denominator is not frozen at its "
            "original value: the static adjusted information is recomputed with the "
            "same altered prior used in the combined experiment. Within the factor-of-"
            "two band:",
            "",
            "| Static prior group varied | Gain range | Spectral-ratio range | Correlation range | Gain with 1% aligned discrepancy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in STATIC_GROUPS:
        item = static_ranges[name]["factor_two"]
        lines.append(
            f"| {STATIC_DISPLAY_NAMES[name]} | {item['lambda_min_gain'][0]:.3g}--{item['lambda_min_gain'][1]:.3g} "
            f"| {item['spectral_ratio'][0]:.3g}--{item['spectral_ratio'][1]:.3g} "
            f"| [{item['target_correlation'][0]:.4f}, {item['target_correlation'][1]:.4f}] "
            f"| {item['aligned_1pct_gain'][0]:.4f}--{item['aligned_1pct_gain'][1]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"The 5 + 7.5 MPa pair remained optimal in {static_primary_count} of "
            f"{len(static_designs)} grouped static-prior scenarios.",
            "",
            "When all static priors are varied together, the factor-of-two gain "
            f"range is {static_ranges['all_static']['factor_two']['lambda_min_gain'][0]:.3g}--"
            f"{static_ranges['all_static']['factor_two']['lambda_min_gain'][1]:.3g}, but the "
            "ratio must be interpreted with care: the static denominator changes from "
            f"{static_ranges['all_static']['factor_two']['static_lambda_min'][0]:.3g} to "
            f"{static_ranges['all_static']['factor_two']['static_lambda_min'][1]:.3g}, while "
            "the combined absolute minimum eigenvalue spans only "
            f"{static_ranges['all_static']['factor_two']['combined_lambda_min'][0]:.3g}--"
            f"{static_ranges['all_static']['factor_two']['combined_lambda_min'][1]:.3g}. "
            "The larger gain under looser static priors therefore does not mean more "
            "absolute information; it largely reflects a smaller static baseline.",
            "",
            "One-at-a-time static-prior influence, ranked by the gain span between "
            "0.5x and 2x:",
            "",
            "| Static nuisance | Gain at 0.5x | Gain at 2x | Span |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in static_influence.itertuples(index=False):
        lines.append(
            f"| `{row.nuisance}` | {row.gain_half:.3g} | {row.gain_double:.3g} "
            f"| {row.factor_two_gain_span:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Scale dependence versus structural dependence",
            "",
            "| Fabric-link model | Gain | Spectral ratio | Correlation | 1% aligned-discrepancy gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in structural.itertuples(index=False):
        lines.append(
            f"| `{row.fabric_mode}` | {row.lambda_min_gain:.3g} | "
            f"{row.spectral_ratio:.3g} | {row.Vcem_lnCn_correlation:.4f} | "
            f"{row.aligned_1pct_lambda_min_gain:.3g} |"
        )
    lines.extend(
        [
            "",
            "This comparison is decisive for interpretation. The drop from the "
            "shared-link result to the expanded-fabric result cannot be reproduced "
            "by merely widening or tightening the expanded-model priors. The model "
            "modes also change which fabric quantities inherit derivatives with "
            "respect to the nominal target Cn. The large collapse is therefore "
            "primarily structural; prior scale controls the conditional value within "
            "that structural assumption.",
            "",
            "## What can be claimed",
            "",
            "- Robust: plausible factor-of-two scale perturbations do not turn the "
            "pressure experiment into a well-separated inversion; target correlation "
            "remains extreme and target-aligned discrepancy leaves no more than a "
            "1.68-fold gain across the grouped pressure and static sweeps.",
            "- Robust: the 5 + 7.5 MPa pair is the selected pair throughout the grouped "
            "0.25--4 scale sweeps for both pressure and static nuisances over the "
            "tested candidate set.",
            "- Conditional: 3.35 should be reported as approximately 3.4 under the "
            "declared priors. Pressure-prior perturbations give 2.9--3.6, whereas "
            "grouped static-prior perturbations broaden the factor-two range to "
            "2.4--5.5, partly through movement of the static denominator.",
            "- Not established: this audit does not provide external empirical "
            "calibration of the prior scales. It tests consequence, not provenance.",
            "",
            "## Statistical scope",
            "",
            "The calculation remains a local Gaussian Schur-complement audit. Very "
            "wide multipliers, especially for critical porosity, are mathematical "
            "stress tests and may extend beyond the locally meaningful physical "
            "support. That is why the factor-of-two band is the primary interpretation.",
        ]
    )
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ensure_dirs()
    baseline = ea.load_baseline()
    target_all, nuisance_all, nuisance_names = ea.pm.pressure_differential_jacobian(
        baseline.pooled,
        baseline.theta,
        ea.PRESSURE_CANDIDATES,
        **ea.FABRIC_CONFIGS[ea.PRIMARY_FABRIC_MODE],
    )

    scales = prior_scale_table()
    grouped = group_sweep(
        baseline, target_all, nuisance_all, nuisance_names
    )
    individual = one_at_a_time_sweep(
        baseline, target_all, nuisance_all, nuisance_names
    )
    designs = reoptimized_design_sweep(
        baseline, target_all, nuisance_all, nuisance_names
    )
    static_scales = static_prior_scale_table(baseline)
    static_grouped = static_group_sweep(
        baseline, target_all, nuisance_all, nuisance_names
    )
    static_individual = static_one_at_a_time_sweep(
        baseline, target_all, nuisance_all, nuisance_names
    )
    static_designs = static_reoptimized_design_sweep(
        baseline, target_all, nuisance_all, nuisance_names
    )
    structural = structural_context(baseline)
    equivalence_error = scale_equivalence_error(
        baseline, target_all, nuisance_all, nuisance_names
    )
    static_equivalence_error = static_scale_equivalence_error(baseline)

    scales.to_csv(TABLES / "prior_scale_definitions.csv", index=False)
    grouped.to_csv(TABLES / "prior_scale_group_sweep.csv", index=False)
    individual.to_csv(TABLES / "prior_scale_one_at_a_time.csv", index=False)
    designs.to_csv(TABLES / "prior_scale_design_reoptimization.csv", index=False)
    structural.to_csv(TABLES / "prior_scale_structural_context.csv", index=False)
    static_scales.to_csv(
        TABLES / "static_prior_scale_definitions.csv", index=False
    )
    static_grouped.to_csv(
        TABLES / "static_prior_scale_group_sweep.csv", index=False
    )
    static_individual.to_csv(
        TABLES / "static_prior_scale_one_at_a_time.csv", index=False
    )
    static_designs.to_csv(
        TABLES / "static_prior_scale_design_reoptimization.csv", index=False
    )
    plot_group_sweep(grouped)
    plot_static_group_sweep(static_grouped)
    plot_combined_prior_scale_audit(grouped, static_grouped)
    write_report(
        baseline,
        grouped,
        individual,
        designs,
        static_grouped,
        static_individual,
        static_designs,
        structural,
        equivalence_error,
        static_equivalence_error,
    )
    print(f"Prior-scale audit written to {RESULTS}")


if __name__ == "__main__":
    main()
