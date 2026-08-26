#!/usr/bin/env python3
"""Generate nested and root inventories and SHA256 files deterministically."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_NAMES = {"MANIFEST.csv", "SHA256SUMS.txt"}


def payload_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and "__pycache__" not in path.parts
        and ".mplconfig" not in path.parts
        and ".git" not in path.parts
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def role(relative: str) -> str:
    if relative.startswith("data/"):
        return "input data"
    if "/results/" in relative:
        return "reported result"
    if relative.startswith("experiments/"):
        return "experiment code or documentation"
    if relative.startswith("docs/") or relative.endswith(".md"):
        return "documentation"
    return "repository metadata or driver"


def freeze_tree(root: Path) -> None:
    files = payload_files(root)
    rows = []
    checksum_lines = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        sha256 = digest(path)
        rows.append((relative, path.stat().st_size, sha256, role(relative)))
        checksum_lines.append(f"{sha256}  {relative}\n")

    with (root / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "sha256", "role"])
        writer.writerows(rows)
    (root / "SHA256SUMS.txt").write_text("".join(checksum_lines), encoding="utf-8")
    print(f"Frozen {len(rows)} payload files under {root.relative_to(ROOT) or Path('.')}" )


def main() -> None:
    for name in ("E1_discover_explain", "E2_stability_hierarchy", "E3_break_design"):
        freeze_tree(ROOT / "experiments" / name)
    freeze_tree(ROOT)


if __name__ == "__main__":
    main()
