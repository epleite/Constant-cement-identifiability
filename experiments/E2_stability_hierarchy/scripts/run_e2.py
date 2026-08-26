#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2_analysis import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run constant-cement E2 bootstrap, LOTO, and hierarchy experiments."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced deterministic smoke experiment.",
    )
    args = parser.parse_args()
    print(json.dumps(run(quick=args.quick), indent=2))


if __name__ == "__main__":
    main()

