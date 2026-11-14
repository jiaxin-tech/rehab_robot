"""Bind the current version to a reviewed source lock, without live config edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name("v2") / "SOURCE_LOCK.json"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def current_binding():
    """Verify canonical source content, then bind actual local bytes for a run.

    The lock explicitly normalizes CRLF to LF for source-text portability.
    Content changes still fail. Returned raw hashes are checked again by the
    existing request/trainer boundaries, without changing historical bindings.
    """
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("version") != "REHAB_OFFLINE_V2" or lock.get("identity") != "LF_NORMALIZED_SOURCE_CONTENT":
        raise ValueError("unsupported algorithm source lock")
    hashes = {}
    for relative, expected in lock["source_sha256"].items():
        path = ROOT / relative
        payload = path.read_bytes()
        if _sha(payload.replace(b"\r\n", b"\n")) != expected:
            raise ValueError("algorithm source lock mismatch: " + relative)
        hashes[relative] = _sha(payload)
    algorithm_root = ROOT / "experiment_algorithm/v2"
    primary = {name: hashes[f"experiment_algorithm/v2/{name}"] for name in ("config.py", "surrogate.py", "policy.py")}
    dependencies = {str(ROOT / relative): digest for relative, digest in hashes.items()
                    if relative not in {f"experiment_algorithm/v2/{name}" for name in primary}}
    return dict(algorithm_version=lock["version"], algorithm_root=str(algorithm_root),
                algorithm_sha256=primary, algorithm_dependencies=dependencies,
                algorithm_source_lock_sha256=_sha(LOCK.read_bytes()))
