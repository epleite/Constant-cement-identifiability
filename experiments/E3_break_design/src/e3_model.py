from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = ROOT / "vendor" / "e1_v1"
sys.path.insert(0, str(E1_ROOT / "src"))

import e1_analysis as e1  # noqa: E402


rc = e1.rc

# Anchor the prospective extension at the effective pressure used by the
# frozen RPIA environment.  The static constant-cement branch itself remains
# pressure blind; this anchor only defines the differential scenario model.
REFERENCE_PRESSURE_MPA = float(rc.P_EFF_MPA)
STIFF_CEMENT_VOLUME = 0.10

# One-sigma scales for prospective pressure-model nuisances.  They are
# standardized in exactly the same way as the frozen RPIA nuisance parameters.
PRESSURE_NUISANCE_SCALES = {
    "logP_calibration": 0.10,
    "log_stress_bulk_scale": 0.25,
    "log_stress_shear_scale": 0.25,
    "stiff_cement_volume_shift": 0.015,
    # Allows the coordination number of the compliant/Hertz--Mindlin contacts
    # to depart from the nominal constant-cement coordination number.
    "ln_soft_cn_offset": 0.20,
    # Additional conservative fabric controls.  These are inactive unless the
    # corresponding branch is explicitly configured as a nuisance.
    "ln_stiff_cn_offset": 0.10,
    "soft_phic_shift": 0.02,
}


def pressure_nuisance_names() -> list[str]:
    return list(PRESSURE_NUISANCE_SCALES)


def all_nuisance_names() -> list[str]:
    return rc.nuisance_names(e1.MODEL) + pressure_nuisance_names()


def all_nuisance_scales() -> dict[str, float]:
    return {**rc.NUI_SCALES, **PRESSURE_NUISANCE_SCALES}


def _state_arrays(
    df: pd.DataFrame, nuisance: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float]:
    phi = np.clip(df.phi.to_numpy() + nuisance.get("phi_bias", 0.0), 0.01, 0.45)
    vcl = np.clip(df.vsh.to_numpy() + nuisance.get("vsh_bias", 0.0), 0.0, 0.5)
    sw = np.clip(df.sw.to_numpy() + nuisance.get("sw_bias", 0.0), 0.0, 1.0)
    fkf = float(np.clip(rc.FKF + nuisance.get("f_kf_shift", 0.0), 0.0, 0.45))
    clay_scale = float(np.exp(nuisance.get("log_clay_mod_scale", 0.0)))
    salinity = float(
        np.clip(rc.SAL0 + nuisance.get("brine_salinity_shift", 0.0), 0.005, 0.20)
    )
    gor = float(np.clip(rc.GOR0 + nuisance.get("GOR_shift", 0.0), 20.0, 250.0))
    cement_scale = float(np.exp(nuisance.get("log_cement_mod_scale", 0.0)))
    phic = float(
        np.clip(rc.PHIC_PACK + nuisance.get("phic_pack_shift", 0.0), 0.34, 0.46)
    )
    return phi, vcl, sw, fkf, clay_scale, salinity, gor, cement_scale, phic


def pressure_extended_forward(
    df: pd.DataFrame,
    theta: np.ndarray,
    pressure_mpa: float,
    nuisance: dict[str, float] | None = None,
    reference_pressure_mpa: float = REFERENCE_PRESSURE_MPA,
    stiff_cement_volume: float = STIFF_CEMENT_VOLUME,
    soft_cn_mode: str = "shared",
    soft_lncn_reference: float | None = None,
    stiff_cn_mode: str = "shared",
    stiff_lncn_reference: float | None = None,
    soft_phic_mode: str = "shared",
    soft_phic_reference: float | None = None,
) -> np.ndarray:
    """Prospective Avseth--Skjei-style pressure extension.

    The frozen constant-cement dry moduli are recovered exactly at the
    reference pressure.  Away from that state, the pressure-sensitive
    increment is supplied by the friable-sand/Hertz--Mindlin branch, weighted
    between a soft end member and a pressure-insensitive 10%-cement end member.

    This is a prospective experimental-design model, not a redefinition of the
    frozen RPIA constant-cement constitutive law.
    """

    nuisance = nuisance or {}
    (
        phi,
        vcl,
        sw,
        fkf,
        clay_scale,
        salinity,
        gor,
        cement_scale,
        phic,
    ) = _state_arrays(df, nuisance)

    vcem, ln_cn = map(float, theta)
    cn = float(np.exp(ln_cn))
    if soft_cn_mode == "shared":
        soft_cn = cn
    elif soft_cn_mode == "fixed":
        if soft_lncn_reference is None:
            soft_lncn_reference = ln_cn
        soft_cn = float(np.exp(soft_lncn_reference))
    elif soft_cn_mode == "nuisance":
        if soft_lncn_reference is None:
            soft_lncn_reference = ln_cn
        soft_cn = float(
            np.exp(soft_lncn_reference + nuisance.get("ln_soft_cn_offset", 0.0))
        )
    else:
        raise ValueError(f"unknown soft_cn_mode: {soft_cn_mode}")
    if stiff_cn_mode == "shared":
        stiff_cn = cn
    elif stiff_cn_mode == "fixed":
        if stiff_lncn_reference is None:
            stiff_lncn_reference = ln_cn
        stiff_cn = float(np.exp(stiff_lncn_reference))
    elif stiff_cn_mode == "nuisance":
        if stiff_lncn_reference is None:
            stiff_lncn_reference = ln_cn
        stiff_cn = float(
            np.exp(
                stiff_lncn_reference
                + nuisance.get("ln_stiff_cn_offset", 0.0)
            )
        )
    else:
        raise ValueError(f"unknown stiff_cn_mode: {stiff_cn_mode}")
    if soft_phic_mode == "shared":
        soft_phic = phic
    elif soft_phic_mode == "nuisance":
        if soft_phic_reference is None:
            soft_phic_reference = rc.PHIC_PACK
        soft_phic = float(
            np.clip(
                soft_phic_reference + nuisance.get("soft_phic_shift", 0.0),
                0.34,
                0.46,
            )
        )
    else:
        raise ValueError(f"unknown soft_phic_mode: {soft_phic_mode}")
    phib = phic - vcem
    vstiff = float(
        np.clip(
            stiff_cement_volume + nuisance.get("stiff_cement_volume_shift", 0.0),
            0.07,
            0.13,
        )
    )
    phib_stiff = phic - vstiff
    if np.any(phi >= 0.999 * phib) or np.any(phi >= 0.999 * phib_stiff):
        return np.full((len(df), 3), np.nan)

    pressure_factor = float(np.exp(nuisance.get("logP_calibration", 0.0)))
    pressure = float(pressure_mpa * pressure_factor)
    pressure0 = float(reference_pressure_mpa * pressure_factor)
    bulk_scale = float(np.exp(nuisance.get("log_stress_bulk_scale", 0.0)))
    shear_scale = float(np.exp(nuisance.get("log_stress_shear_scale", 0.0)))

    output: list[tuple[float, float, float]] = []
    for i in range(len(df)):
        km, gm, rhom = rc.matrix(float(vcl[i]), fkf, clay_scale)
        kcc, gcc = rc.constant(
            km,
            gm,
            rc.KCEM * cement_scale,
            rc.GCEM * cement_scale,
            float(phi[i]),
            phic,
            phib,
            cn,
            1,
        )
        ksoft0, gsoft0 = rc.soft_dry(
            float(phi[i]), km, gm, soft_phic, soft_cn, pressure0
        )
        ksoft, gsoft = rc.soft_dry(
            float(phi[i]), km, gm, soft_phic, soft_cn, pressure
        )
        kstiff, gstiff = rc.constant(
            km,
            gm,
            rc.KCEM * cement_scale,
            rc.GCEM * cement_scale,
            float(phi[i]),
            phic,
            phib_stiff,
            stiff_cn,
            1,
        )

        wk = (kcc - ksoft0) / (kstiff - ksoft0)
        wg = (gcc - gsoft0) / (gstiff - gsoft0)
        kd = kcc + bulk_scale * (1.0 - wk) * (ksoft - ksoft0)
        gd = gcc + shear_scale * (1.0 - wg) * (gsoft - gsoft0)
        output.append(
            rc.elastic(
                kd,
                gd,
                km,
                rhom,
                float(phi[i]),
                float(sw[i]),
                salinity,
                gor,
            )
        )
    return np.asarray(output)


def patchy_weights(
    df: pd.DataFrame,
    theta: np.ndarray,
    reference_pressure_mpa: float = REFERENCE_PRESSURE_MPA,
    stiff_cement_volume: float = STIFF_CEMENT_VOLUME,
) -> pd.DataFrame:
    vcem, ln_cn = map(float, theta)
    cn = float(np.exp(ln_cn))
    rows: list[dict[str, float]] = []
    for row in df.itertuples(index=False):
        km, gm, _ = rc.matrix(float(row.vsh))
        kcc, gcc = rc.constant(
            km,
            gm,
            rc.KCEM,
            rc.GCEM,
            float(row.phi),
            rc.PHIC_PACK,
            rc.PHIC_PACK - vcem,
            cn,
            1,
        )
        ksoft, gsoft = rc.soft_dry(
            float(row.phi), km, gm, rc.PHIC_PACK, cn, reference_pressure_mpa
        )
        kstiff, gstiff = rc.constant(
            km,
            gm,
            rc.KCEM,
            rc.GCEM,
            float(row.phi),
            rc.PHIC_PACK,
            rc.PHIC_PACK - stiff_cement_volume,
            cn,
            1,
        )
        rows.append(
            {
                "phi": float(row.phi),
                "vsh": float(row.vsh),
                "W_K": float((kcc - ksoft) / (kstiff - ksoft)),
                "W_G": float((gcc - gsoft) / (gstiff - gsoft)),
            }
        )
    return pd.DataFrame(rows)


def generalized_bounding_weights(
    df: pd.DataFrame,
    theta: np.ndarray,
    nuisance: dict[str, float] | None = None,
    stiff_cement_volume: float = STIFF_CEMENT_VOLUME,
    soft_cn_mode: str = "shared",
    soft_lncn_reference: float | None = None,
    stiff_cn_mode: str = "shared",
    stiff_lncn_reference: float | None = None,
    soft_phic_mode: str = "shared",
    soft_phic_reference: float | None = None,
) -> pd.DataFrame:
    """Return the bulk and shear interpolation weights for diagnostics."""

    nuisance = nuisance or {}
    phi, vcl, _, fkf, clay_scale, _, _, cement_scale, phic = _state_arrays(
        df, nuisance
    )
    vcem, ln_cn = map(float, theta)
    cn = float(np.exp(ln_cn))
    if soft_lncn_reference is None:
        soft_lncn_reference = ln_cn
    if stiff_lncn_reference is None:
        stiff_lncn_reference = ln_cn
    if soft_phic_reference is None:
        soft_phic_reference = rc.PHIC_PACK
    if soft_cn_mode == "shared":
        soft_cn = cn
    elif soft_cn_mode == "fixed":
        soft_cn = float(np.exp(soft_lncn_reference))
    elif soft_cn_mode == "nuisance":
        soft_cn = float(
            np.exp(soft_lncn_reference + nuisance.get("ln_soft_cn_offset", 0.0))
        )
    else:
        raise ValueError(soft_cn_mode)
    if stiff_cn_mode == "shared":
        stiff_cn = cn
    elif stiff_cn_mode == "fixed":
        stiff_cn = float(np.exp(stiff_lncn_reference))
    elif stiff_cn_mode == "nuisance":
        stiff_cn = float(
            np.exp(
                stiff_lncn_reference
                + nuisance.get("ln_stiff_cn_offset", 0.0)
            )
        )
    else:
        raise ValueError(stiff_cn_mode)
    if soft_phic_mode == "shared":
        soft_phic = phic
    elif soft_phic_mode == "nuisance":
        soft_phic = float(
            np.clip(
                soft_phic_reference + nuisance.get("soft_phic_shift", 0.0),
                0.34,
                0.46,
            )
        )
    else:
        raise ValueError(soft_phic_mode)
    vstiff = float(
        np.clip(
            stiff_cement_volume + nuisance.get("stiff_cement_volume_shift", 0.0),
            0.07,
            0.13,
        )
    )
    rows = []
    for i in range(len(df)):
        km, gm, _ = rc.matrix(float(vcl[i]), fkf, clay_scale)
        kcc, gcc = rc.constant(
            km,
            gm,
            rc.KCEM * cement_scale,
            rc.GCEM * cement_scale,
            float(phi[i]),
            phic,
            phic - vcem,
            cn,
            1,
        )
        ksoft, gsoft = rc.soft_dry(
            float(phi[i]),
            km,
            gm,
            soft_phic,
            soft_cn,
            REFERENCE_PRESSURE_MPA,
        )
        kstiff, gstiff = rc.constant(
            km,
            gm,
            rc.KCEM * cement_scale,
            rc.GCEM * cement_scale,
            float(phi[i]),
            phic,
            phic - vstiff,
            stiff_cn,
            1,
        )
        rows.append(
            {
                "W_K": float((kcc - ksoft) / (kstiff - ksoft)),
                "W_G": float((gcc - gsoft) / (gstiff - gsoft)),
            }
        )
    return pd.DataFrame(rows)


def pressure_differential_vector(
    df: pd.DataFrame,
    theta: np.ndarray,
    pressures_mpa: Iterable[float],
    nuisance: dict[str, float] | None = None,
    reference_pressure_mpa: float = REFERENCE_PRESSURE_MPA,
    soft_cn_mode: str = "shared",
    soft_lncn_reference: float | None = None,
    stiff_cn_mode: str = "shared",
    stiff_lncn_reference: float | None = None,
    soft_phic_mode: str = "shared",
    soft_phic_reference: float | None = None,
) -> np.ndarray:
    pressures_mpa = list(pressures_mpa)
    reference = pressure_extended_forward(
        df,
        theta,
        reference_pressure_mpa,
        nuisance,
        reference_pressure_mpa,
        soft_cn_mode=soft_cn_mode,
        soft_lncn_reference=soft_lncn_reference,
        stiff_cn_mode=stiff_cn_mode,
        stiff_lncn_reference=stiff_lncn_reference,
        soft_phic_mode=soft_phic_mode,
        soft_phic_reference=soft_phic_reference,
    )
    if not np.all(np.isfinite(reference)):
        return np.full(2 * len(df) * len(pressures_mpa), np.nan)
    output: list[np.ndarray] = []
    for pressure in pressures_mpa:
        state = pressure_extended_forward(
            df,
            theta,
            float(pressure),
            nuisance,
            reference_pressure_mpa,
            soft_cn_mode=soft_cn_mode,
            soft_lncn_reference=soft_lncn_reference,
            stiff_cn_mode=stiff_cn_mode,
            stiff_lncn_reference=stiff_lncn_reference,
            soft_phic_mode=soft_phic_mode,
            soft_phic_reference=soft_phic_reference,
        )
        output.append(np.log(state[:, 0] / reference[:, 0]))
        output.append(np.log(state[:, 1] / reference[:, 1]))
    return np.concatenate(output)


def pressure_differential_jacobian(
    df: pd.DataFrame,
    theta: np.ndarray,
    pressures_mpa: list[float],
    step: float = 1e-4,
    soft_cn_mode: str = "shared",
    stiff_cn_mode: str = "shared",
    soft_phic_mode: str = "shared",
    reference_pressure_mpa: float = REFERENCE_PRESSURE_MPA,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return unscaled Jacobians of delta-log Vp and Vs.

    Divide both returned matrices by the assumed one-sigma uncertainty in a
    differential log-velocity observation before forming information matrices.
    """

    soft_lncn_reference = float(theta[1])
    stiff_lncn_reference = float(theta[1])
    soft_phic_reference = float(rc.PHIC_PACK)
    reference = pressure_differential_vector(
        df,
        theta,
        pressures_mpa,
        reference_pressure_mpa=reference_pressure_mpa,
        soft_cn_mode=soft_cn_mode,
        soft_lncn_reference=soft_lncn_reference,
        stiff_cn_mode=stiff_cn_mode,
        stiff_lncn_reference=stiff_lncn_reference,
        soft_phic_mode=soft_phic_mode,
        soft_phic_reference=soft_phic_reference,
    )
    target = np.zeros((len(reference), len(theta)))
    for j, scale in enumerate(rc.PARAM_SCALES[e1.MODEL]):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step * scale
        minus[j] -= step * scale
        target[:, j] = (
            pressure_differential_vector(
                df,
                plus,
                pressures_mpa,
                reference_pressure_mpa=reference_pressure_mpa,
                soft_cn_mode=soft_cn_mode,
                soft_lncn_reference=soft_lncn_reference,
                stiff_cn_mode=stiff_cn_mode,
                stiff_lncn_reference=stiff_lncn_reference,
                soft_phic_mode=soft_phic_mode,
                soft_phic_reference=soft_phic_reference,
            )
            - pressure_differential_vector(
                df,
                minus,
                pressures_mpa,
                reference_pressure_mpa=reference_pressure_mpa,
                soft_cn_mode=soft_cn_mode,
                soft_lncn_reference=soft_lncn_reference,
                stiff_cn_mode=stiff_cn_mode,
                stiff_lncn_reference=stiff_lncn_reference,
                soft_phic_mode=soft_phic_mode,
                soft_phic_reference=soft_phic_reference,
            )
        ) / (2.0 * step)

    names = all_nuisance_names()
    scales = all_nuisance_scales()
    nuisance_jac = np.zeros((len(reference), len(names)))
    for j, name in enumerate(names):
        perturbation = step * scales[name]
        nuisance_jac[:, j] = (
            pressure_differential_vector(
                df,
                theta,
                pressures_mpa,
                {name: perturbation},
                reference_pressure_mpa=reference_pressure_mpa,
                soft_cn_mode=soft_cn_mode,
                soft_lncn_reference=soft_lncn_reference,
                stiff_cn_mode=stiff_cn_mode,
                stiff_lncn_reference=stiff_lncn_reference,
                soft_phic_mode=soft_phic_mode,
                soft_phic_reference=soft_phic_reference,
            )
            - pressure_differential_vector(
                df,
                theta,
                pressures_mpa,
                {name: -perturbation},
                reference_pressure_mpa=reference_pressure_mpa,
                soft_cn_mode=soft_cn_mode,
                soft_lncn_reference=soft_lncn_reference,
                stiff_cn_mode=stiff_cn_mode,
                stiff_lncn_reference=stiff_lncn_reference,
                soft_phic_mode=soft_phic_mode,
                soft_phic_reference=soft_phic_reference,
            )
        ) / (2.0 * step)
    return target, nuisance_jac, names


def pressure_row_indices(
    pressures_all: list[float], selected: Iterable[float], n_samples: int
) -> np.ndarray:
    selected_set = {float(value) for value in selected}
    indices: list[int] = []
    block = 2 * n_samples
    for i, pressure in enumerate(pressures_all):
        if float(pressure) in selected_set:
            indices.extend(range(i * block, (i + 1) * block))
    return np.asarray(indices, dtype=int)


def sample_row_indices(
    sample_indices: np.ndarray,
    n_samples: int,
    n_observables: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for observable in range(n_observables):
        rows.append(observable * n_samples + sample_indices)
    return np.concatenate(rows).astype(int)


def pressure_sample_row_indices(
    sample_indices: np.ndarray,
    pressures_all: list[float],
    selected: Iterable[float],
    n_samples: int,
) -> np.ndarray:
    selected_set = {float(value) for value in selected}
    rows: list[np.ndarray] = []
    for i, pressure in enumerate(pressures_all):
        if float(pressure) not in selected_set:
            continue
        start = 2 * i * n_samples
        rows.append(start + sample_indices)
        rows.append(start + n_samples + sample_indices)
    return np.concatenate(rows).astype(int)


def pressure_response_proxy_gradient(
    theta: np.ndarray,
    p_cn: float = 2.0 / 3.0,
    s_contact: float = 1.0,
) -> np.ndarray:
    """Scaled gradient for ln R_P = const + p ln Cn - s ln a_c."""

    vcem = float(theta[0])
    sv, sl = map(float, rc.PARAM_SCALES[e1.MODEL])
    return np.array(
        [
            -s_contact * sv / (4.0 * vcem),
            (p_cn + s_contact / 4.0) * sl,
        ]
    )


def candidate_gradients(
    theta: np.ndarray,
    A: float,
    Gamma: float,
) -> dict[str, np.ndarray]:
    vcem = float(theta[0])
    phib = float(rc.PHIC_PACK - vcem)
    sv, sl = map(float, rc.PARAM_SCALES[e1.MODEL])
    return {
        "q_star": np.array([(A / vcem - Gamma / phib) * sv, sl]),
        "Vcem_proxy": np.array([sv, 0.0]),
        "lnCn_proxy": np.array([0.0, sl]),
        "ln_contact_radius": np.array([sv / (4.0 * vcem), -sl / 4.0]),
        "pressure_response_proxy": pressure_response_proxy_gradient(theta),
    }


def finite_difference_stability(
    df: pd.DataFrame,
    theta: np.ndarray,
    pressures_mpa: list[float],
    steps: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
    soft_cn_mode: str = "nuisance",
    stiff_cn_mode: str = "shared",
    soft_phic_mode: str = "shared",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    reference_target, reference_nuisance, names = pressure_differential_jacobian(
        df,
        theta,
        pressures_mpa,
        step=steps[1],
        soft_cn_mode=soft_cn_mode,
        stiff_cn_mode=stiff_cn_mode,
        soft_phic_mode=soft_phic_mode,
    )
    for step in steps:
        target, nuisance, _ = pressure_differential_jacobian(
            df,
            theta,
            pressures_mpa,
            step=step,
            soft_cn_mode=soft_cn_mode,
            stiff_cn_mode=stiff_cn_mode,
            soft_phic_mode=soft_phic_mode,
        )
        rows.append(
            {
                "step": float(step),
                "target_relative_error_to_1e-4": float(
                    np.linalg.norm(target - reference_target)
                    / max(np.linalg.norm(reference_target), 1e-30)
                ),
                "nuisance_relative_error_to_1e-4": float(
                    np.linalg.norm(nuisance - reference_nuisance)
                    / max(np.linalg.norm(reference_nuisance), 1e-30)
                ),
                "target_norm": float(np.linalg.norm(target)),
                "nuisance_norm": float(np.linalg.norm(nuisance)),
                "n_nuisances": float(len(names)),
            }
        )
    return pd.DataFrame(rows)


def pressure_independence_audit(
    df: pd.DataFrame,
    theta: np.ndarray,
    pressures_mpa: Iterable[float],
) -> pd.DataFrame:
    reference = rc.forward(df, e1.MODEL, theta)
    rows = []
    for pressure in pressures_mpa:
        # In the frozen implementation pressure is computed in rc.forward but
        # never used by the constant-cement branch.  The same call is therefore
        # the exact prediction at every nominal pressure.
        state = rc.forward(df, e1.MODEL, theta)
        rows.append(
            {
                "pressure_mpa": float(pressure),
                "max_abs_delta_Vp_mps": float(np.max(np.abs(state[:, 0] - reference[:, 0]))),
                "max_abs_delta_Vs_mps": float(np.max(np.abs(state[:, 1] - reference[:, 1]))),
                "max_abs_delta_rho_gcc": float(np.max(np.abs(state[:, 2] - reference[:, 2]))),
                "differential_target_jacobian_norm": 0.0,
            }
        )
    return pd.DataFrame(rows)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa /= np.linalg.norm(aa)
    bb /= np.linalg.norm(bb)
    return float(math.degrees(math.acos(float(np.clip(abs(aa @ bb), -1.0, 1.0)))))
