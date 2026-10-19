"""Package current code and verified offline assets; never installs or connects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from scripts.lab_session import ROOT, load


def package(bundle, initialization, output):
    bundle = Path(bundle).resolve()
    if not bundle.is_relative_to(ROOT):
        raise ValueError("bundle must be inside repository")
    initialized = load(initialization)
    if initialized.get("status") != "ALGORITHM_INITIALIZED" or initialized.get("source_hashes_checked") is not True:
        raise ValueError("actual frozen initialization evidence required")
    source = initialized["public_prior"]
    files = set()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    for name in tracked:
        if not name:
            continue
        p = ROOT / name
        if p.suffix in (".py", ".ps1", ".ini") and p.is_file():
            files.add(p.resolve())
    for folder in ("scripts", "tests"):
        files.update(p.resolve() for p in (ROOT / folder).glob("*.py"))
        files.update(p.resolve() for p in (ROOT / folder).glob("*.ps1"))
    files.add(ROOT / "config/lab_next_visit.json")
    files.add(ROOT / "pytest-core.ini")
    files.update(p.resolve() for p in bundle.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.update((ROOT / "site_evidence" / name).resolve() for name in (
        "COLLISION_EVENT_INTERFACE_2026-10-19.md", "OFFLINE_EMPTY_LOAD_REVIEW_2026-10-19.md"))
    for name in source["asset_files"]:
        asset = (Path(source["asset_root"]) / name).resolve(strict=True)
        if not asset.is_relative_to(ROOT):
            raise ValueError("model assets must be in this repository's verified cache")
        files.add(asset)
    metadata = dict(base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        purpose="CODE_AND_FROZEN_ASSET_DELIVERY_NOT_APPROVAL", installation_performed=False,
        note="Contains current source snapshot, not a Git merge. Reconcile lab changes before copying. Existing site safety configs and raw lab data are not included.",
        asset_provenance=source, files={})
    for p in sorted(files):
        if not p.is_relative_to(ROOT):
            raise ValueError("source escapes repository")
        metadata["files"][p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    output = Path(output)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr("lab_delivery_manifest.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        for name, digest in metadata["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("archive checksum mismatch: " + name)
        if archive.testzip() is not None:
            raise ValueError("archive CRC failure")
    return dict(output=str(output.resolve()), files=len(files), bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(), zip_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package(args.bundle, args.initialization, args.output), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
