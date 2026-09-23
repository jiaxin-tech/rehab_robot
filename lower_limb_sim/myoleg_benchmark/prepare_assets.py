"""Fetch only the pinned official MyoSuite model assets; install no dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from .portable_model import ASSET_SOURCE_URL, ASSET_VERSION, DEFAULT_ASSET_ROOT, DEFAULT_MODEL_PATH, PROJECT_ROOT

WHEEL_SHA256 = "bec0133d9d4ba0249aeccdd7de632eead4749436ffb4e5a3c44a8f9cedafd9ea"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=PROJECT_ROOT / ".cache/myosuite-assets/myosuite-2.12.2-py3-none-any.whl")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve()
    wheel.parent.mkdir(parents=True, exist_ok=True)
    if not wheel.exists():
        temporary = wheel.with_suffix(".download")
        print(f"Downloading pinned MyoSuite {ASSET_VERSION} assets (about 91 MB)", flush=True)
        with urllib.request.urlopen(ASSET_SOURCE_URL, timeout=60) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != WHEEL_SHA256:
            raise ValueError("MYOSUITE_WHEEL_SHA256_MISMATCH")
        temporary.replace(wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("MYOSUITE_WHEEL_SHA256_MISMATCH")
    prefix = "myosuite/simhive/myo_sim/"
    needed = sorted({element.get("file").replace("\\", "/").split(prefix, 1)[1]
                     for element in ET.parse(DEFAULT_MODEL_PATH).iter()
                     if element.get("file") and prefix in element.get("file").replace("\\", "/")})
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for relative in needed:
            path = (DEFAULT_ASSET_ROOT / relative).resolve()
            if not path.is_relative_to(DEFAULT_ASSET_ROOT.resolve()):
                raise ValueError("ASSET_PATH_ESCAPES_CACHE")
            payload = archive.read(prefix + relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != payload:
                path.write_bytes(payload)
            hashes[relative] = hashlib.sha256(payload).hexdigest()
        for name in archive.namelist():
            if ".dist-info/" in name and name.rsplit("/", 1)[-1].upper() in {"LICENSE", "METADATA"}:
                (DEFAULT_ASSET_ROOT.parents[2] / name.rsplit("/", 1)[-1]).write_bytes(archive.read(name))
    manifest = dict(source_url=ASSET_SOURCE_URL, wheel_sha256=WHEEL_SHA256,
                    myosuite_version=ASSET_VERSION, asset_sha256=hashes)
    (DEFAULT_ASSET_ROOT.parents[2] / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(hashes)} assets: {DEFAULT_ASSET_ROOT}")


if __name__ == "__main__":
    main()
