from __future__ import annotations

"""Systematic multi-start and reverse-continuation audit of nonlinear profiles."""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nonlinear_ridge as core  # noqa: E402


@dataclass
class Fit:
    x: np.ndarray
    objective: float
    success: bool
    nfev: int
    optimality: float
    active_bounds: int


class ProfileProblem:
    def __init__(
        self,
        baseline: core.e3.Baseline,
        scenario: core.Scenario | None,
        discrepancy: np.ndarray | None,
    ) -> None:
        self.baseline = baseline
        self.scenario = scenario
        self.static_reference = core.static_vector(
            baseline, baseline.theta, np.zeros(len(core.STATIC_NAMES))
        )
        self.static_sigma = core.e1.stacked_sigma(
            baseline.pooled, baseline.scale
        )
        if scenario is None:
            self.name = "static"
            self.names = core.STATIC_NAMES
            self.pressure_reference = None
            self.apply_precision = lambda residual: residual
        else:
            self.name = scenario.name
            self.names = core.ALL_NAMES
            self.pressure_reference = core.pressure_vector(
                baseline,
                baseline.theta,
                np.zeros(len(self.names)),
                scenario.fabric_mode,
            )
            if discrepancy is None:
                raise ValueError("combined problem requires discrepancy")
            self.apply_precision = core.low_rank_sqrt_precision(discrepancy)
        self.lower = np.r_[np.log(3.0), np.full(len(self.names), -4.0)]
        self.upper = np.r_[np.log(18.0), np.full(len(self.names), 4.0)]

    def residual(self, vcem: float, x: np.ndarray) -> np.ndarray:
        theta = np.array([float(vcem), float(x[0])], dtype=float)
        z = np.asarray(x[1:], dtype=float)
        static_residual = (
            core.static_vector(
                self.baseline, theta, z[: len(core.STATIC_NAMES)]
            )
            - self.static_reference
        ) / self.static_sigma
        if not np.all(np.isfinite(static_residual)):
            n_data = len(static_residual)
            if self.scenario is not None:
                n_data += len(self.pressure_reference)
            return np.r_[np.full(n_data, 1.0e6), z]
        if self.scenario is None:
            data_residual = static_residual
        else:
            pressure = core.pressure_vector(
                self.baseline, theta, z, self.scenario.fabric_mode
            )
            if not np.all(np.isfinite(pressure)):
                return np.r_[
                    np.full(
                        len(static_residual) + len(self.pressure_reference),
                        1.0e6,
                    ),
                    z,
                ]
            pressure_residual = core.whiten_pressure_vector(
                pressure - self.pressure_reference,
                len(self.baseline.pooled),
            )
            data_residual = self.apply_precision(
                np.r_[static_residual, pressure_residual]
            )
        return np.r_[data_residual, z]

    def fit(self, vcem: float, start: np.ndarray) -> Fit:
        start = np.minimum(np.maximum(np.asarray(start, dtype=float), self.lower + 1e-9), self.upper - 1e-9)
        result = least_squares(
            lambda x: self.residual(vcem, x),
            start,
            bounds=(self.lower, self.upper),
            jac="2-point",
            diff_step=1.0e-4,
            max_nfev=250,
            xtol=2.0e-10,
            ftol=2.0e-10,
            gtol=2.0e-9,
        )
        return Fit(
            x=result.x.copy(),
            objective=float(result.fun @ result.fun),
            success=bool(result.success),
            nfev=int(result.nfev),
            optimality=float(result.optimality),
            active_bounds=int(np.count_nonzero(result.active_mask)),
        )


def stored_x(row: pd.Series, problem: ProfileProblem) -> np.ndarray:
    return np.r_[
        float(row.lnCn_MAP),
        [float(row[f"MAP_{name}_sigma"]) for name in problem.names],
    ]


def row_from_fit(
    problem: ProfileProblem,
    vcem: float,
    strategy: str,
    start: np.ndarray,
    fit: Fit,
) -> dict[str, float | str | bool | int]:
    output: dict[str, float | str | bool | int] = {
        "profile": problem.name,
        "Vcem_fraction": float(vcem),
        "Vcem_percent": float(100.0 * vcem),
        "start_strategy": strategy,
        "start_lnCn": float(start[0]),
        "start_Cn": float(np.exp(start[0])),
        "result_lnCn": float(fit.x[0]),
        "result_Cn": float(np.exp(fit.x[0])),
        "objective": fit.objective,
        "success": fit.success,
        "nfev": fit.nfev,
        "optimality": fit.optimality,
        "active_bounds": fit.active_bounds,
    }
    for name, value in zip(problem.names, start[1:]):
        output[f"start_{name}_sigma"] = float(value)
    for name, value in zip(problem.names, fit.x[1:]):
        output[f"result_{name}_sigma"] = float(value)
    return output


def deterministic_perturbation(n: int) -> np.ndarray:
    index = np.arange(1, n + 1, dtype=float)
    return 0.18 * np.sin(1.61803398875 * index)


def risk_indices(table: pd.DataFrame, profile_name: str) -> list[int]:
    n = len(table)
    indices = {0, n - 1, n // 4, n // 2, 3 * n // 4}
    truth = int(np.argmin(table.objective_MAP.to_numpy(dtype=float)))
    indices.update({max(0, truth - 1), truth, min(n - 1, truth + 1)})
    top_optimality = np.argsort(table.optimizer_optimality.to_numpy(dtype=float))[-3:]
    indices.update(map(int, top_optimality))
    if profile_name == "shared_generic":
        objective = table.objective_MAP.to_numpy(dtype=float)
        for threshold in (2.30, 5.99):
            nearest = np.argsort(np.abs(objective - threshold))[:4]
            for value in nearest:
                indices.update(
                    {
                        max(0, int(value) - 1),
                        int(value),
                        min(n - 1, int(value) + 1),
                    }
                )
        # Include the high-V branch transition even though it lies outside the
        # convex validity domain; this tests whether the stored discontinuity is
        # an optimizer artifact.
        transition = int(np.argmin(np.abs(table.Vcem_fraction - 0.052)))
        indices.update({max(0, transition - 1), transition, min(n - 1, transition + 1)})
    return sorted(indices)


def audit_problem(
    problem: ProfileProblem,
    stored: pd.DataFrame,
) -> pd.DataFrame:
    stored = stored.sort_values("Vcem_fraction").reset_index(drop=True)
    output: list[dict[str, float | str | bool | int]] = []

    # Preserve the original solution as an explicit candidate.
    for row in stored.itertuples(index=False):
        series = pd.Series(row._asdict())
        x = stored_x(series, problem)
        fit = Fit(
            x=x,
            objective=float(row.objective_MAP),
            success=bool(row.optimizer_success),
            nfev=int(row.optimizer_nfev),
            optimality=float(row.optimizer_optimality),
            active_bounds=int(row.active_bounds),
        )
        output.append(
            row_from_fit(
                problem, float(row.Vcem_fraction), "stored_original", x, fit
            )
        )

    # Refit every stored MAP with a controlled finite-difference step.
    for _, row in stored.iterrows():
        start = stored_x(row, problem)
        fit = problem.fit(float(row.Vcem_fraction), start)
        output.append(
            row_from_fit(
                problem,
                float(row.Vcem_fraction),
                "stored_refined",
                start,
                fit,
            )
        )
    print(f"  {problem.name}: stored refinements complete", flush=True)

    # Independent reverse continuations from both target bounds.
    for label, indices in [
        ("reverse_high_to_low", list(range(len(stored) - 1, -1, -1))),
        ("reverse_low_to_high", list(range(len(stored)))),
    ]:
        first = stored.iloc[indices[0]]
        previous = np.r_[
            float(first.lnCn_MAP), np.zeros(len(problem.names), dtype=float)
        ]
        for index in indices:
            row = stored.iloc[index]
            start = previous.copy()
            fit = problem.fit(float(row.Vcem_fraction), start)
            output.append(
                row_from_fit(
                    problem, float(row.Vcem_fraction), label, start, fit
                )
            )
            previous = fit.x.copy()
        print(f"  {problem.name}: {label} complete", flush=True)

    # Four independent starts on a comprehensive high-risk subset.
    perturb = deterministic_perturbation(len(problem.names))
    for index in risk_indices(stored, problem.name):
        row = stored.iloc[index]
        original = stored_x(row, problem)
        starts = {
            "zero_nuisance_local_lnCn": np.r_[
                float(row.lnCn_MAP), np.zeros(len(problem.names))
            ],
            "pooled_zero": np.r_[
                float(problem.baseline.theta[1]), np.zeros(len(problem.names))
            ],
            "deterministic_plus": np.r_[
                float(row.lnCn_MAP) + 0.08, original[1:] + perturb
            ],
            "deterministic_minus": np.r_[
                float(row.lnCn_MAP) - 0.08, original[1:] - perturb
            ],
        }
        for strategy, start in starts.items():
            fit = problem.fit(float(row.Vcem_fraction), start)
            output.append(
                row_from_fit(
                    problem,
                    float(row.Vcem_fraction),
                    strategy,
                    start,
                    fit,
                )
            )
    print(f"  {problem.name}: independent starts complete", flush=True)
    return pd.DataFrame(output)


def best_profiles(
    runs: pd.DataFrame,
    original_profiles: pd.DataFrame,
) -> pd.DataFrame:
    index = runs.groupby(["profile", "Vcem_fraction"])["objective"].idxmin()
    best = runs.loc[index].copy().sort_values(["profile", "Vcem_fraction"])
    old = original_profiles[
        ["profile", "Vcem_fraction", "objective_MAP", "Cn_MAP"]
    ].rename(
        columns={
            "objective_MAP": "stored_objective",
            "Cn_MAP": "stored_Cn",
        }
    )
    best = best.merge(old, on=["profile", "Vcem_fraction"], how="left")
    best["objective_improvement"] = best.stored_objective - best.objective
    best["Cn_shift_from_stored"] = best.result_Cn - best.stored_Cn
    return best.reset_index(drop=True)


def profile_width(table: pd.DataFrame, threshold: float) -> dict:
    adapted = table.rename(
        columns={
            "objective": "objective_MAP",
            "result_Cn": "Cn_MAP",
        }
    )
    return {
        **core.interpolated_width(adapted, threshold),
        **core.supported_map_span(adapted, threshold),
    }


def crossing_bracket(table: pd.DataFrame, threshold: float, side: str) -> tuple[int, int]:
    table = table.sort_values("Vcem_fraction").reset_index(drop=True)
    objective = table.objective.to_numpy(dtype=float)
    truth = int(np.argmin(objective))
    if side == "lower":
        for inner in range(truth, 0, -1):
            outer = inner - 1
            if objective[inner] <= threshold < objective[outer]:
                return outer, inner
    elif side == "upper":
        for inner in range(truth, len(table) - 1):
            outer = inner + 1
            if objective[inner] <= threshold < objective[outer]:
                return inner, outer
    raise RuntimeError(f"no {side} crossing bracket for threshold {threshold}")


def dense_shared_crossings(
    problem: ProfileProblem,
    shared_best: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared_best = shared_best.sort_values("Vcem_fraction").reset_index(drop=True)
    cache: dict[float, Fit] = {}
    audit_rows: list[dict] = []

    def best_start_near(vcem: float) -> np.ndarray:
        distance = np.abs(shared_best.Vcem_fraction.to_numpy(dtype=float) - vcem)
        row = shared_best.iloc[int(np.argmin(distance))]
        return np.r_[
            float(row.result_lnCn),
            [float(row[f"result_{name}_sigma"]) for name in problem.names],
        ]

    def evaluate(vcem: float) -> Fit:
        key = round(float(vcem), 12)
        if key in cache:
            return cache[key]
        local = best_start_near(vcem)
        zero = np.r_[local[0], np.zeros(len(problem.names))]
        candidates = []
        for strategy, start in [
            ("dense_nearest_MAP", local),
            ("dense_zero_nuisance", zero),
        ]:
            fit = problem.fit(float(vcem), start)
            candidates.append(fit)
            audit_rows.append(row_from_fit(problem, vcem, strategy, start, fit))
        best = min(candidates, key=lambda item: item.objective)
        cache[key] = best
        return best

    crossing_rows = []
    width_rows = []
    for threshold in (2.30, 5.99):
        roots = {}
        for side in ("lower", "upper"):
            first, second = crossing_bracket(shared_best, threshold, side)
            a = float(shared_best.iloc[first].Vcem_fraction)
            b = float(shared_best.iloc[second].Vcem_fraction)
            root = brentq(
                lambda value: evaluate(float(value)).objective - threshold,
                a,
                b,
                xtol=5.0e-8,
                rtol=1.0e-10,
                maxiter=50,
            )
            fitted = evaluate(root)
            roots[side] = root
            crossing_rows.append(
                {
                    "threshold": threshold,
                    "side": side,
                    "Vcem_fraction": float(root),
                    "Vcem_percent": float(100.0 * root),
                    "objective": fitted.objective,
                    "lnCn_MAP": float(fitted.x[0]),
                    "Cn_MAP": float(np.exp(fitted.x[0])),
                    "root_residual": float(fitted.objective - threshold),
                    "coarse_bracket_min_fraction": min(a, b),
                    "coarse_bracket_max_fraction": max(a, b),
                }
            )
        width_rows.append(
            {
                "threshold": threshold,
                "Vcem_lower_fraction": roots["lower"],
                "Vcem_upper_fraction": roots["upper"],
                "Vcem_width_percentage_points": 100.0
                * (roots["upper"] - roots["lower"]),
                "lower_censored": False,
                "upper_censored": False,
            }
        )
    return (
        pd.DataFrame(crossing_rows),
        pd.DataFrame(width_rows),
        pd.DataFrame(audit_rows),
    )


def main() -> None:
    baseline = core.e3.load_baseline()
    original = pd.read_csv(HERE / "results" / "nonlinear_MAP_profiles.csv")
    generic = core.generic_discrepancy_matrix(baseline)
    pooled_weak = core.weak_vector(baseline.gram_adjusted)

    scenario_lookup: dict[str, core.Scenario | None] = {"static": None}
    scenario_lookup.update({scenario.name: scenario for scenario in core.SCENARIOS})
    problems: dict[str, ProfileProblem] = {}
    run_parts = []
    for name, scenario in scenario_lookup.items():
        discrepancy = None
        if scenario is not None:
            discrepancy = core.discrepancy_basis(
                baseline, scenario, generic, pooled_weak
            )[0]
        problem = ProfileProblem(baseline, scenario, discrepancy)
        problems[name] = problem
        print(f"multi-start audit: {name}", flush=True)
        stored_subset = original[original.profile == name]
        if name == "shared_generic":
            # The primary pressure construction is convex only through the
            # 0.0344 grid point on the shared-fabric path.  Audit every point
            # in that domain; retain the original extrapolated points in the
            # main profile CSV but do not spend optimizer effort on them.
            stored_subset = stored_subset[
                stored_subset.Vcem_fraction <= 0.0345
            ]
        run_parts.append(
            audit_problem(problem, stored_subset)
        )

    runs = pd.concat(run_parts, ignore_index=True)
    best = best_profiles(runs, original)
    shared_best = best[best.profile == "shared_generic"]
    print("adaptive shared-profile crossings", flush=True)
    crossings, dense_widths, dense_runs = dense_shared_crossings(
        problems["shared_generic"], shared_best
    )
    runs = pd.concat([runs, dense_runs], ignore_index=True)

    runs.to_csv(HERE / "results" / "multistart_runs.csv", index=False)
    best.to_csv(HERE / "results" / "multistart_best_profiles.csv", index=False)
    crossings.to_csv(
        HERE / "results" / "shared_dense_crossings.csv", index=False
    )
    dense_widths.to_csv(
        HERE / "results" / "shared_dense_profile_widths.csv", index=False
    )

    summary: dict = {
        "method": {
            "stored_refinement_all_points_in_audited_domain": True,
            "reverse_continuation_both_directions_all_points_in_audited_domain": True,
            "audited_domain": (
                "all 32 points for static/expanded/aligned; all 19 shared-fabric "
                "points through Vcem=0.034433 inside the convex validity domain"
            ),
            "independent_starts_on_high_risk_subset": [
                "zero_nuisance_local_lnCn",
                "pooled_zero",
                "deterministic_plus",
                "deterministic_minus",
            ],
            "controlled_optimizer_diff_step": 1.0e-4,
            "dense_shared_crossings": "Brent root with two-start nonlinear nuisance profiling at every evaluation",
        },
        "profiles": {},
        "shared_dense_widths": dense_widths.to_dict(orient="records"),
        "shared_dense_crossings": crossings.to_dict(orient="records"),
        "run_count": int(len(runs)),
    }
    for name in scenario_lookup:
        old = original[original.profile == name].sort_values("Vcem_fraction")
        new = best[best.profile == name].sort_values("Vcem_fraction")
        risk = runs[
            (runs.profile == name)
            & ~runs.start_strategy.isin(["stored_original", "stored_refined"])
        ]
        width_comparison = {}
        for threshold in (2.30, 5.99):
            old_adapted = old.copy()
            old_width = core.interpolated_width(old_adapted, threshold)
            new_width = profile_width(new, threshold)
            width_comparison[str(threshold)] = {
                "stored_coarse": old_width,
                "multistart_coarse": new_width,
            }
        summary["profiles"][name] = {
            "points": int(len(new)),
            "total_runs": int(len(runs[runs.profile == name])),
            "independent_or_reverse_runs": int(len(risk)),
            "all_new_runs_success": bool(
                runs[
                    (runs.profile == name)
                    & (runs.start_strategy != "stored_original")
                ].success.all()
            ),
            "maximum_objective_improvement": float(
                new.objective_improvement.max()
            ),
            "median_objective_improvement": float(
                new.objective_improvement.median()
            ),
            "maximum_absolute_Cn_shift": float(
                np.abs(new.Cn_shift_from_stored).max()
            ),
            "best_strategy_counts": {
                str(key): int(value)
                for key, value in new.start_strategy.value_counts().items()
            },
            "width_comparison": width_comparison,
        }

    (HERE / "results" / "multistart_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
