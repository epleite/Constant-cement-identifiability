from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
VERSION = (ROOT / "VERSION").read_text().strip()
ARCHIVE = WORKSPACE / f"{ROOT.name}_v{VERSION}.zip"
MANIFEST = ROOT / "MANIFEST.csv"
CHECKSUMS = ROOT / "SHA256SUMS.txt"
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(*, include_metadata: bool) -> list[Path]:
    excluded_directories = {"__pycache__", ".mplconfig", "workbook_previews"}
    excluded_suffixes = {".pyc", ".pyo"}
    metadata = {MANIFEST, CHECKSUMS}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded_directories for part in relative.parts):
            continue
        if path.suffix in excluded_suffixes:
            continue
        if path.name.endswith(".inspect.ndjson"):
            continue
        if not include_metadata and path in metadata:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def write_metadata() -> list[Path]:
    payload_files = included_files(include_metadata=False)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in payload_files:
            writer.writerow(
                [
                    path.relative_to(ROOT).as_posix(),
                    path.stat().st_size,
                    sha256(path),
                ]
            )
    CHECKSUMS.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in payload_files
        ),
        encoding="utf-8",
    )
    return payload_files


def build_archive() -> None:
    files = included_files(include_metadata=True)
    with zipfile.ZipFile(
        ARCHIVE, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def verify_archive(payload_files: list[Path]) -> dict[str, object]:
    expected_manifest = {
        path.relative_to(ROOT).as_posix(): sha256(path) for path in payload_files
    }
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        if any(Path(member).is_absolute() or ".." in Path(member).parts for member in members):
            raise RuntimeError("Archive contains an unsafe member path")
        with tempfile.TemporaryDirectory(prefix="e3_package_verify_") as temp:
            archive.extractall(temp)
            extracted_root = Path(temp) / ROOT.name
            for relative, expected in expected_manifest.items():
                actual = sha256(extracted_root / relative)
                if actual != expected:
                    raise RuntimeError(f"Checksum mismatch after extraction: {relative}")
            workbook_path = (
                extracted_root / "results" / "constant_cement_E3_results.xlsx"
            )
            with zipfile.ZipFile(workbook_path) as workbook_archive:
                bad_workbook_member = workbook_archive.testzip()
                if bad_workbook_member is not None:
                    raise RuntimeError(
                        f"Workbook ZIP CRC failure: {bad_workbook_member}"
                    )
                workbook_xml = workbook_archive.read("xl/workbook.xml")
                for sheet_name in [
                    "Summary",
                    "Fabric ablation",
                    "Design",
                    "Robustness",
                    "Discrepancy controls",
                    "Profiles",
                    "Verification",
                    "Source index",
                ]:
                    if f'name="{sheet_name}"'.encode() not in workbook_xml:
                        raise RuntimeError(
                            f"Workbook is missing required sheet: {sheet_name}"
                        )
            completed = subprocess.run(
                [sys.executable, "scripts/verify_e3.py"],
                cwd=extracted_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Extracted verification failed:\n"
                    + completed.stdout
                    + completed.stderr
                )
    return {
        "archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "payload_file_count": len(payload_files),
        "archive_member_count": len(members),
        "verification": "PASS",
    }


def main() -> int:
    payload_files = write_metadata()
    build_archive()
    report = verify_archive(payload_files)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
