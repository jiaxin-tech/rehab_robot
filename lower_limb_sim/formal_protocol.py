"""Strict read-only access to the formal paper/active-pipeline protocol.

The JSON manifest is the single editable source of truth.  This module exposes
validated immutable values for existing Python call sites; it does not contain
robot safety limits and does not authorize motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_EXPERIMENT_MANIFEST_PATH = (
    PROJECT_ROOT / "config" / "formal_experiment_manifest.json"
)
SOURCE_ACTIVE_REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_candidates"
    / "reference_measured_asymmetric_closed_slow.csv"
)


def _load_manifest() -> dict[str, Any]:
    payload = json.loads(
        FORMAL_EXPERIMENT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("formal experiment manifest root must be an object")
    expected_keys = {
        "schema_version",
        "rom_protocol_version",
        "hip_rom_deg",
        "knee_rom_deg",
        "theta_shank_definition",
        "active_reference_id",
        "active_reference_parent_cycle_frames",
        "active_reference_sha256",
        "reference_release_manifest",
        "scope",
        "real_robot_safety_thresholds_reviewed",
        "notes",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "formal experiment manifest keys do not match the strict schema"
        )
    return payload


def _finite_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"{name} must be a two-value JSON array")
    if any(isinstance(item, bool) for item in value):
        raise RuntimeError(f"{name} must contain numbers, not booleans")
    pair = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in pair) or not pair[0] < pair[1]:
        raise RuntimeError(f"{name} must contain increasing finite values")
    return pair


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FORMAL_EXPERIMENT_MANIFEST = _load_manifest()
ROM_PROTOCOL_VERSION = str(FORMAL_EXPERIMENT_MANIFEST["rom_protocol_version"])
FORMAL_HIP_ROM_DEG = _finite_pair(
    FORMAL_EXPERIMENT_MANIFEST["hip_rom_deg"], "hip_rom_deg"
)
FORMAL_KNEE_ROM_DEG = _finite_pair(
    FORMAL_EXPERIMENT_MANIFEST["knee_rom_deg"], "knee_rom_deg"
)
THETA_SHANK_DEFINITION = str(
    FORMAL_EXPERIMENT_MANIFEST["theta_shank_definition"]
)
ACTIVE_REFERENCE_ID = str(FORMAL_EXPERIMENT_MANIFEST["active_reference_id"])
ACTIVE_REFERENCE_SHA256 = str(
    FORMAL_EXPERIMENT_MANIFEST["active_reference_sha256"]
)
ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES = tuple(
    int(value)
    for value in FORMAL_EXPERIMENT_MANIFEST[
        "active_reference_parent_cycle_frames"
    ]
)
REFERENCE_RELEASE_MANIFEST_PATH = PROJECT_ROOT / str(
    FORMAL_EXPERIMENT_MANIFEST["reference_release_manifest"]
)
ACTIVE_REFERENCE_PATH = (
    REFERENCE_RELEASE_MANIFEST_PATH.parent
    / "reference_measured_asymmetric_closed_slow.csv"
)

if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
    raise RuntimeError("formal ROM protocol must remain ROM_PROTOCOL_V2")
if FORMAL_HIP_ROM_DEG != (0.0, 120.0):
    raise RuntimeError("formal hip ROM must remain 0--120 deg")
if FORMAL_KNEE_ROM_DEG != (5.0, 145.0):
    raise RuntimeError("formal knee ROM must remain 5--145 deg")
if THETA_SHANK_DEFINITION != "q_hip - q_knee":
    raise RuntimeError("theta_shank must remain q_hip - q_knee")
if ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES != (5844, 5895, 5934):
    raise RuntimeError("active reference parent cycle does not match approval")
if FORMAL_EXPERIMENT_MANIFEST["real_robot_safety_thresholds_reviewed"] is not False:
    raise RuntimeError("ROM migration cannot approve real-robot safety thresholds")


def validate_active_reference_file(path: str | Path = ACTIVE_REFERENCE_PATH) -> None:
    """Fail if the pinned active reference is absent or has changed."""

    reference_path = Path(path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"active reference is missing: {reference_path}")
    actual = sha256_file(reference_path)
    if actual != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError(
            "active reference SHA-256 mismatch: "
            f"expected {ACTIVE_REFERENCE_SHA256}, got {actual}"
        )


__all__ = [
    "ACTIVE_REFERENCE_ID",
    "ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES",
    "ACTIVE_REFERENCE_PATH",
    "ACTIVE_REFERENCE_SHA256",
    "FORMAL_EXPERIMENT_MANIFEST",
    "FORMAL_EXPERIMENT_MANIFEST_PATH",
    "FORMAL_HIP_ROM_DEG",
    "FORMAL_KNEE_ROM_DEG",
    "ROM_PROTOCOL_VERSION",
    "REFERENCE_RELEASE_MANIFEST_PATH",
    "SOURCE_ACTIVE_REFERENCE_PATH",
    "THETA_SHANK_DEFINITION",
    "sha256_file",
    "validate_active_reference_file",
]
