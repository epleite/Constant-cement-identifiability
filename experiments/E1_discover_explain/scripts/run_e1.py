from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e1_analysis import run


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
