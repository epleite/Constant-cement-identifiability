from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e3_analysis import run_analysis  # noqa: E402


if __name__ == "__main__":
    summary = run_analysis(quick="--quick" in sys.argv)
    print(
        "E3 complete:",
        summary["primary_design"]["pressures_mpa"],
        "lambda-min gain=",
        summary["primary_design"]["lambda_min_gain"],
    )
