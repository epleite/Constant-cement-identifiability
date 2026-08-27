#!/usr/bin/env python3
"""Verify every frozen result group and the root integrity manifest."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> None:
    print(f"VERIFY {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_checksums(root: Path) -> None:
    checksum_file = root / "SHA256SUMS.txt"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = digest(path)
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")
    label = root.relative_to(ROOT) if root != ROOT else Path(".")
    print(f"SHA256 manifest {label}: OK")


def main() -> None:
    # Several scientific verifiers intentionally emit machine-readable audit
    # reports.  Run them in an isolated copy so --verify-only is genuinely
    # read-only and platform-specific last digits cannot alter frozen files.
    with tempfile.TemporaryDirectory(prefix="constant-cement-verify-") as tmp:
        isolated_root = Path(tmp) / "repository"
        shutil.copytree(
            ROOT,
            isolated_root,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".mplconfig"),
        )
        checks = [
            (
                isolated_root / "experiments" / "E1_discover_explain",
                [sys.executable, "scripts/verify_e1.py"],
            ),
            (
                isolated_root / "experiments" / "E2_stability_hierarchy",
                [sys.executable, "scripts/verify_e2.py"],
            ),
            (
                isolated_root / "experiments" / "E3_break_design",
                [sys.executable, "scripts/verify_e3.py"],
            ),
            (
                isolated_root / "analysis_checks" / "qstar_finite_form",
                [sys.executable, "verify.py"],
            ),
            (
                isolated_root / "experiments" / "E4_nonlinear_ridge_audit",
                [sys.executable, "verify.py"],
            ),
            (
                isolated_root / "experiments" / "E5_prior_scale_audit",
                [sys.executable, "scripts/verify_prior_scale_sweep.py"],
            ),
        ]
        for cwd, command in checks:
            run(command, cwd)
    for name in ("E1_discover_explain", "E2_stability_hierarchy", "E3_break_design"):
        verify_checksums(ROOT / "experiments" / name)
    verify_checksums(ROOT)
    print("All frozen publication outputs verified.")


if __name__ == "__main__":
    main()
