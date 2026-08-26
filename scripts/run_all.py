#!/usr/bin/env python3
"""Run, smoke-test, or verify the frozen manuscript analyses."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    relative = cwd.relative_to(ROOT) if cwd.is_relative_to(ROOT) else cwd
    print(f"[{relative}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def verify_frozen() -> None:
    run([sys.executable, "scripts/verify_frozen.py"], ROOT)


def quick_smoke_test() -> None:
    """Execute reduced E1--E3 runs in temporary copies.

    The publication results under the repository root are never modified.
    """

    experiments = [
        ("E1_discover_explain", "run_e1.py"),
        ("E2_stability_hierarchy", "run_e2.py"),
        ("E3_break_design", "run_e3.py"),
    ]
    with tempfile.TemporaryDirectory(prefix="constant-cement-smoke-") as tmp:
        temporary_root = Path(tmp)
        for directory, runner in experiments:
            source = ROOT / "experiments" / directory
            target = temporary_root / directory
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", ".mplconfig"),
            )
            run([sys.executable, f"scripts/{runner}", "--quick"], target)
    print("Quick smoke test completed without modifying frozen publication outputs.")


def full_regeneration() -> None:
    primary = [
        ("E1_discover_explain", "run_e1.py", "verify_e1.py"),
        ("E2_stability_hierarchy", "run_e2.py", "verify_e2.py"),
        ("E3_break_design", "run_e3.py", "verify_e3.py"),
    ]
    for directory, runner, verifier in primary:
        cwd = ROOT / "experiments" / directory
        run([sys.executable, f"scripts/{runner}"], cwd)
        run([sys.executable, f"scripts/{verifier}"], cwd)

    qstar = ROOT / "analysis_checks" / "qstar_finite_form"
    run([sys.executable, "run_analysis.py"], qstar)
    run([sys.executable, "verify.py"], qstar)

    nonlinear = ROOT / "experiments" / "E4_nonlinear_ridge_audit"
    for script in (
        "run_nonlinear_ridge.py",
        "multistart_audit.py",
        "refine_shared_supported.py",
        "recompute_multistart_pointwise.py",
        "synchronize_outputs.py",
        "verify.py",
    ):
        run([sys.executable, script], nonlinear)

    priors = ROOT / "experiments" / "E5_prior_scale_audit"
    environment = os.environ.copy()
    environment["E3_ROOT"] = str(ROOT / "experiments" / "E3_break_design")
    run([sys.executable, "scripts/run_prior_scale_sweep.py"], priors, environment)
    run([sys.executable, "scripts/verify_prior_scale_sweep.py"], priors, environment)

    run([sys.executable, "scripts/freeze_package.py"], ROOT)
    verify_frozen()


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="run reduced E1--E3 calculations in temporary copies",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify frozen outputs without regenerating them",
    )
    args = parser.parse_args()

    if args.quick:
        quick_smoke_test()
    elif args.verify_only:
        verify_frozen()
    else:
        full_regeneration()


if __name__ == "__main__":
    main()
