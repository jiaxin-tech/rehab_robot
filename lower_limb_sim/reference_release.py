"""Fail-closed access to the frozen measured-asymmetric reference release.

This module audits existing bytes only.  It never selects a cycle, fits a
spline, smooths samples, retimes a path, or writes reference artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import L1, L2
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES,
    SOURCE_ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    REFERENCE_RELEASE_MANIFEST_PATH,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from .kinematics import forward_kinematics


RELEASE_DIRECTORY = REFERENCE_RELEASE_MANIFEST_PATH.parent
RELEASE_ACTIVE_REFERENCE_PATH = (
    RELEASE_DIRECTORY / "reference_measured_asymmetric_closed_slow.csv"
)
RELEASE_METADATA_PATH = (
    RELEASE_DIRECTORY / "reference_measured_asymmetric_metadata.json"
)
RELEASE_VERSION_MANIFEST_PATH = RELEASE_DIRECTORY / "reference_version_manifest.csv"
RELEASE_CLOSURE_AUDIT_PATH = RELEASE_DIRECTORY / "reference_cycle_closure_audit.csv"
RELEASE_SHA256SUMS_PATH = RELEASE_DIRECTORY / "SHA256SUMS.txt"
REFERENCE_RELEASE_VERSION = "reference_measured_asymmetric_closed_v1_slow"

_REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "reference_id",
    "reference_version",
    "sha256",
    "release_file",
    "source_file",
    "source_file_sha256",
    "source_bone_csv",
    "source_bone_csv_sha256",
    "selected_cycle_start_frame",
    "selected_cycle_peak_frame",
    "selected_cycle_end_frame",
    "hip_rom_deg",
    "knee_rom_deg",
    "rom_protocol_version",
    "L1_m",
    "L2_m",
    "L2_definition",
    "theta_shank_definition",
    "closure_method",
    "trajectory_continuity",
    "flexion_duration_s",
    "extension_duration_s",
    "total_duration_s",
    "measured_extension_is_reversed_flexion",
    "hip_flexion_extension_asymmetry_rmse_deg",
    "knee_flexion_extension_asymmetry_rmse_deg",
    "pull_path_asymmetry_rmse_mm",
    "asymmetry_preservation_ratio",
    "active",
    "legacy",
    "approved_for_offline_personalization",
    "approved_for_first_robot_trial",
    "robot_execution_status",
    "superseded_source_first_trial_flags_authoritative",
    "freeze_policy",
    "frozen_reference_content_must_not_change",
}
_REFERENCE_REQUIRED_COLUMNS = {
    "reference_version",
    "trajectory_id",
    "profile",
    "time_s",
    "cycle_phase",
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
    "theta_shank_rad",
    "x_pull_m",
    "z_pull_m",
    "extension_source_is_measured",
    "measured_extension_is_reversed_flexion",
    "spline_degree",
    "continuity_order",
    "L1_m",
    "L2_m",
    "approved_hip_min_deg",
    "approved_hip_max_deg",
    "approved_knee_min_deg",
    "approved_knee_max_deg",
    "joint_limit_valid",
    "trajectory_sample_valid",
}


@dataclass(frozen=True)
class ReferenceFreezeAudit:
    reference_id: str
    sha256: str
    sample_count: int
    all_finite: bool
    joint_closure_valid: bool
    pull_closure_valid: bool
    theta_shank_valid: bool
    rom_valid: bool
    c2_continuity_valid: bool
    asymmetry_valid: bool
    duration_valid: bool
    source_bytes_match_release: bool

    @property
    def valid(self) -> bool:
        return all(
            (
                self.all_finite,
                self.joint_closure_valid,
                self.pull_closure_valid,
                self.theta_shank_valid,
                self.rom_valid,
                self.c2_continuity_valid,
                self.asymmetry_valid,
                self.duration_valid,
                self.source_bytes_match_release,
            )
        )


@dataclass(frozen=True)
class FrozenReferenceBundle:
    trajectory: pd.DataFrame
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    audit: ReferenceFreezeAudit


def _strict_bool(series: pd.Series, field_name: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise RuntimeError(f"{field_name} contains a non-boolean value")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def _finite_pair(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"{field_name} must be a two-number array")
    if any(isinstance(item, bool) for item in value):
        raise RuntimeError(f"{field_name} cannot contain booleans")
    pair = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in pair):
        raise RuntimeError(f"{field_name} must contain finite numbers")
    return pair


def load_reference_release_manifest(
    path: str | Path = REFERENCE_RELEASE_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_MANIFEST_KEYS:
        raise RuntimeError("reference release manifest does not match strict schema")
    expected = {
        "reference_id": ACTIVE_REFERENCE_ID,
        "reference_version": REFERENCE_RELEASE_VERSION,
        "sha256": ACTIVE_REFERENCE_SHA256,
        "source_file_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "active": True,
        "legacy": False,
        "approved_for_offline_personalization": True,
        "approved_for_first_robot_trial": False,
        "robot_execution_status": "NO_GO",
        "measured_extension_is_reversed_flexion": False,
        "frozen_reference_content_must_not_change": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"reference release manifest has invalid {key}")
    if _finite_pair(payload["hip_rom_deg"], "hip_rom_deg") != FORMAL_HIP_ROM_DEG:
        raise RuntimeError("release hip ROM differs from formal manifest")
    if _finite_pair(payload["knee_rom_deg"], "knee_rom_deg") != FORMAL_KNEE_ROM_DEG:
        raise RuntimeError("release knee ROM differs from formal manifest")
    cycle = (
        int(payload["selected_cycle_start_frame"]),
        int(payload["selected_cycle_peak_frame"]),
        int(payload["selected_cycle_end_frame"]),
    )
    if cycle != ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES:
        raise RuntimeError("reference release cycle differs from formal manifest")
    if not math.isclose(float(payload["L1_m"]), L1, abs_tol=1e-15):
        raise RuntimeError("reference release L1 differs from runtime")
    if not math.isclose(float(payload["L2_m"]), L2, abs_tol=1e-15):
        raise RuntimeError("reference release L2 differs from runtime")
    return payload


def _resolve_manifest_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{field_name} must be a non-empty path")
    return (REFERENCE_RELEASE_MANIFEST_PATH.parent / value).resolve()


def _validate_asymmetry_metadata(
    manifest: Mapping[str, Any], metadata: Mapping[str, Any]
) -> bool:
    stored = metadata.get("flexion_extension_asymmetry")
    if not isinstance(stored, Mapping):
        return False
    expected = (
        ("closed_hip_flexion_extension_asymmetry_rmse_deg", "hip_flexion_extension_asymmetry_rmse_deg"),
        ("closed_knee_flexion_extension_asymmetry_rmse_deg", "knee_flexion_extension_asymmetry_rmse_deg"),
        ("closed_pull_path_asymmetry_rmse_mm", "pull_path_asymmetry_rmse_mm"),
    )
    if any(
        not math.isclose(
            float(stored[source_key]),
            float(manifest[release_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for source_key, release_key in expected
    ):
        return False
    ratios = manifest.get("asymmetry_preservation_ratio")
    ratio_keys = {
        "hip": "hip_asymmetry_retention_ratio",
        "knee": "knee_asymmetry_retention_ratio",
        "pull_path": "pull_asymmetry_retention_ratio",
    }
    return isinstance(ratios, Mapping) and all(
        math.isclose(
            float(ratios[key]),
            float(stored[source_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key, source_key in ratio_keys.items()
    ) and stored.get("asymmetry_preserved") is True


def _closed_asymmetry_from_timed_csv(
    trajectory: pd.DataFrame,
) -> tuple[float, float, float]:
    """Audit branch difference; reversing extension is comparison-only."""

    if "segment_phase" not in trajectory:
        raise RuntimeError("frozen reference has no segment_phase for asymmetry audit")
    flexion = trajectory.loc[trajectory["cycle_phase"].eq("flexion")]
    extension = trajectory.loc[trajectory["cycle_phase"].eq("extension")]
    if len(flexion) < 3 or len(extension) < 3:
        raise RuntimeError("frozen reference must retain both measured branches")
    local_phase = np.linspace(0.0, 1.0, 2001)
    extension_phase = np.concatenate(
        ([0.0], extension["segment_phase"].to_numpy(dtype=float))
    )

    def branch_values(column: str) -> tuple[np.ndarray, np.ndarray]:
        flexion_values = PchipInterpolator(
            flexion["segment_phase"].to_numpy(dtype=float),
            flexion[column].to_numpy(dtype=float),
        )(local_phase)
        extension_values = PchipInterpolator(
            extension_phase,
            np.concatenate(
                ([float(flexion[column].iloc[-1])], extension[column].to_numpy(dtype=float))
            ),
        )(1.0 - local_phase)
        return np.asarray(flexion_values), np.asarray(extension_values)

    flexion_hip, extension_hip = branch_values("q_hip_rad")
    flexion_knee, extension_knee = branch_values("q_knee_rad")
    flexion_x, extension_x = branch_values("x_pull_m")
    flexion_z, extension_z = branch_values("z_pull_m")
    return (
        float(np.sqrt(np.mean(np.rad2deg(flexion_hip - extension_hip) ** 2))),
        float(np.sqrt(np.mean(np.rad2deg(flexion_knee - extension_knee) ** 2))),
        float(
            np.sqrt(
                np.mean(
                    ((flexion_x - extension_x) * 1000.0) ** 2
                    + ((flexion_z - extension_z) * 1000.0) ** 2
                )
            )
        ),
    )


def verify_reference_sha256(
    path: str | Path,
    expected_sha256: str = ACTIVE_REFERENCE_SHA256,
) -> str:
    """Return the exact digest or raise the required fail-closed marker."""

    reference_path = Path(path)
    actual = sha256_file(reference_path)
    if actual != expected_sha256:
        raise RuntimeError(f"REFERENCE_HASH_MISMATCH: {reference_path}")
    return actual


def _verify_release_checksums() -> None:
    lines = RELEASE_SHA256SUMS_PATH.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for line in lines:
        digest, separator, filename = line.partition("  ")
        if not separator or len(digest) != 64 or not filename:
            raise RuntimeError("release SHA256SUMS.txt has invalid syntax")
        if filename in seen or Path(filename).name != filename:
            raise RuntimeError("release SHA256SUMS.txt has unsafe or duplicate path")
        seen.add(filename)
        path = RELEASE_DIRECTORY / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"release bundle checksum mismatch: {path}")
    expected = {
        "reference_cycle_closure_audit.csv",
        "reference_measured_asymmetric_closed_slow.csv",
        "reference_measured_asymmetric_metadata.json",
        "reference_release_manifest.json",
        "reference_version_manifest.csv",
        "source_reference_information.json",
    }
    if seen != expected:
        raise RuntimeError("release SHA256SUMS.txt file set is incomplete")


def audit_frozen_reference(
    trajectory: pd.DataFrame,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    source_bytes_match_release: bool,
    closure_tolerance: float = 1e-12,
) -> ReferenceFreezeAudit:
    missing = _REFERENCE_REQUIRED_COLUMNS.difference(trajectory.columns)
    if missing:
        raise RuntimeError(f"frozen reference is missing columns: {sorted(missing)}")
    if trajectory.empty:
        raise RuntimeError("frozen reference is empty")
    if set(trajectory["trajectory_id"].astype(str)) != {ACTIVE_REFERENCE_ID}:
        raise RuntimeError("frozen reference contains a non-active trajectory ID")
    if set(trajectory["profile"].astype(str)) != {"slow"}:
        raise RuntimeError("frozen active reference must contain only slow profile")

    numerical_columns = sorted(
        _REFERENCE_REQUIRED_COLUMNS.intersection(trajectory.select_dtypes(include=[np.number]).columns)
        - {"invalid_reason"}
    )
    all_finite = bool(
        np.isfinite(trajectory[numerical_columns].to_numpy(dtype=float)).all()
    )
    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
    pull_stored = trajectory[["x_pull_m", "z_pull_m"]].to_numpy(dtype=float)
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    pull_fk = np.column_stack((x_pull, z_pull))
    joint_closure = bool(
        np.allclose(
            trajectory[["q_hip_rad", "q_knee_rad"]].iloc[0],
            trajectory[["q_hip_rad", "q_knee_rad"]].iloc[-1],
            atol=closure_tolerance,
            rtol=0.0,
        )
    )
    pull_closure = bool(
        np.allclose(pull_stored[0], pull_stored[-1], atol=closure_tolerance, rtol=0.0)
        and np.allclose(pull_stored, pull_fk, atol=closure_tolerance, rtol=0.0)
    )
    theta_valid = bool(
        np.allclose(
            trajectory["theta_shank_rad"].to_numpy(dtype=float),
            q_hip - q_knee,
            atol=1e-14,
            rtol=0.0,
        )
    )
    hip_deg = np.rad2deg(q_hip)
    knee_deg = np.rad2deg(q_knee)
    rom_valid = bool(
        ((hip_deg >= FORMAL_HIP_ROM_DEG[0]) & (hip_deg <= FORMAL_HIP_ROM_DEG[1])).all()
        and ((knee_deg >= FORMAL_KNEE_ROM_DEG[0]) & (knee_deg <= FORMAL_KNEE_ROM_DEG[1])).all()
        and _strict_bool(trajectory["joint_limit_valid"], "joint_limit_valid").all()
        and _strict_bool(
            trajectory["trajectory_sample_valid"], "trajectory_sample_valid"
        ).all()
    )
    continuity = metadata.get("continuity_audit")
    c2_valid = bool(
        isinstance(continuity, Mapping)
        and continuity.get("passed") is True
        and int(continuity.get("continuity_order", -1)) == 2
        and set(trajectory["continuity_order"].astype(int)) == {2}
        and set(trajectory["spline_degree"].astype(int)) == {3}
        and np.allclose(
            trajectory[
                [
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                ]
            ].iloc[0],
            trajectory[
                [
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                ]
            ].iloc[-1],
            atol=closure_tolerance,
            rtol=0.0,
        )
    )
    extension_measured = _strict_bool(
        trajectory["extension_source_is_measured"], "extension_source_is_measured"
    )
    extension_reversed = _strict_bool(
        trajectory["measured_extension_is_reversed_flexion"],
        "measured_extension_is_reversed_flexion",
    )
    audited_asymmetry = _closed_asymmetry_from_timed_csv(trajectory)
    released_asymmetry = (
        float(manifest["hip_flexion_extension_asymmetry_rmse_deg"]),
        float(manifest["knee_flexion_extension_asymmetry_rmse_deg"]),
        float(manifest["pull_path_asymmetry_rmse_mm"]),
    )
    asymmetry_valid = bool(
        extension_measured.all()
        and not extension_reversed.any()
        and metadata.get("measured_extension_is_reversed_flexion") is False
        and _validate_asymmetry_metadata(manifest, metadata)
        and all(value > 1.0 for value in audited_asymmetry)
        and np.allclose(
            audited_asymmetry,
            released_asymmetry,
            atol=1e-4,
            rtol=0.0,
        )
    )
    time_s = trajectory["time_s"].to_numpy(dtype=float)
    flexion = trajectory.loc[trajectory["cycle_phase"].eq("flexion"), "time_s"]
    extension = trajectory.loc[trajectory["cycle_phase"].eq("extension"), "time_s"]
    duration_valid = bool(
        len(flexion) == 201
        and len(extension) == 200
        and np.all(np.diff(time_s) > 0.0)
        and math.isclose(float(flexion.iloc[-1]), float(manifest["flexion_duration_s"]), abs_tol=1e-12)
        and math.isclose(
            float(time_s[-1] - flexion.iloc[-1]),
            float(manifest["extension_duration_s"]),
            abs_tol=1e-12,
        )
        and math.isclose(float(time_s[-1]), float(manifest["total_duration_s"]), abs_tol=1e-12)
    )
    return ReferenceFreezeAudit(
        reference_id=ACTIVE_REFERENCE_ID,
        sha256=ACTIVE_REFERENCE_SHA256,
        sample_count=int(len(trajectory)),
        all_finite=all_finite,
        joint_closure_valid=joint_closure,
        pull_closure_valid=pull_closure,
        theta_shank_valid=theta_valid,
        rom_valid=rom_valid,
        c2_continuity_valid=c2_valid,
        asymmetry_valid=asymmetry_valid,
        duration_valid=duration_valid,
        source_bytes_match_release=source_bytes_match_release,
    )


def load_frozen_active_reference(
    path: str | Path = RELEASE_ACTIVE_REFERENCE_PATH,
) -> FrozenReferenceBundle:
    """Load only the canonical release CSV and validate all freeze invariants."""

    requested = Path(path).expanduser().resolve()
    canonical = RELEASE_ACTIVE_REFERENCE_PATH.resolve()
    if requested != canonical:
        raise PermissionError("final active loader accepts only the canonical release CSV")
    manifest = load_reference_release_manifest()
    release_path = _resolve_manifest_path(manifest["release_file"], "release_file")
    source_path = _resolve_manifest_path(manifest["source_file"], "source_file")
    if release_path != canonical:
        raise RuntimeError("reference release manifest points outside canonical CSV")
    for required_path in (
        release_path,
        source_path,
        RELEASE_METADATA_PATH,
        RELEASE_VERSION_MANIFEST_PATH,
        RELEASE_CLOSURE_AUDIT_PATH,
        RELEASE_SHA256SUMS_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"required frozen-reference artifact missing: {required_path}")
    _verify_release_checksums()
    release_sha = verify_reference_sha256(release_path)
    source_sha = verify_reference_sha256(source_path)
    if release_path.read_bytes() != source_path.read_bytes():
        raise RuntimeError("REFERENCE_HASH_MISMATCH: source and release bytes differ")
    if source_path != SOURCE_ACTIVE_REFERENCE_PATH.resolve():
        raise RuntimeError("reference release source path is not the formal source CSV")
    source_bone = _resolve_manifest_path(manifest["source_bone_csv"], "source_bone_csv")
    if not source_bone.is_file():
        raise FileNotFoundError(f"reference source bone CSV missing: {source_bone}")
    verify_reference_sha256(
        source_bone,
        str(manifest["source_bone_csv_sha256"]),
    )
    source_information_path = RELEASE_DIRECTORY / "source_reference_information.json"
    source_information = json.loads(
        source_information_path.read_text(encoding="utf-8")
    )
    if not isinstance(source_information, dict):
        raise RuntimeError("source reference information root must be an object")
    source_fields = (
        ("source_bone_csv", "source_bone_csv_sha256"),
        ("stage5a_full_angles", "stage5a_full_angles_sha256"),
        ("stage5a_detected_cycles", "stage5a_detected_cycles_sha256"),
        ("stage5a_metadata", "stage5a_metadata_sha256"),
    )
    for path_key, digest_key in source_fields:
        source_artifact = _resolve_manifest_path(
            source_information[path_key], path_key
        )
        if not source_artifact.is_file():
            raise FileNotFoundError(
                f"reference source artifact missing: {source_artifact}"
            )
        verify_reference_sha256(
            source_artifact,
            str(source_information[digest_key]),
        )

    metadata = json.loads(RELEASE_METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("reference release metadata root must be an object")
    if metadata.get("active_reference_trajectory") != ACTIVE_REFERENCE_ID:
        raise RuntimeError("reference release metadata contains another active ID")
    if metadata.get("active_reference_sha256") != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("reference release metadata contains another active SHA")
    require_parent_reference(metadata)
    if tuple(
        int(metadata["selected_cycle_candidate"][key])
        for key in ("start_frame", "peak_frame", "end_frame")
    ) != ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES:
        raise RuntimeError("reference release metadata contains another source cycle")

    versions = pd.read_csv(RELEASE_VERSION_MANIFEST_PATH)
    required_version_columns = {
        "trajectory_id",
        "active_reference",
        "legacy_software_comparison",
        "allowed_for_first_robot_trial",
        "legacy",
        "not_used_for_final_personalization",
        "not_used_for_robot_execution",
    }
    if required_version_columns.difference(versions.columns):
        raise RuntimeError("reference version manifest schema is incomplete")
    active_values = _strict_bool(versions["active_reference"], "active_reference")
    active_rows = versions.loc[active_values]
    if len(active_rows) != 1 or str(active_rows.iloc[0]["trajectory_id"]) != ACTIVE_REFERENCE_ID:
        raise RuntimeError("reference version manifest must contain one active ID")
    legacy_rows = versions["trajectory_id"].astype(str).str.contains(
        "reference_closed_symmetric|reference_closed_c2", regex=True
    )
    if bool((_strict_bool(versions["active_reference"], "active_reference") & legacy_rows).any()):
        raise RuntimeError("legacy symmetric reference cannot be active")
    if not _strict_bool(
        versions.loc[legacy_rows, "legacy_software_comparison"],
        "legacy_software_comparison",
    ).all():
        raise RuntimeError("legacy reference rows must be marked legacy")
    if not _strict_bool(versions.loc[legacy_rows, "legacy"], "legacy").all():
        raise RuntimeError("legacy reference release rows must set legacy=true")
    for field in (
        "not_used_for_final_personalization",
        "not_used_for_robot_execution",
    ):
        if not _strict_bool(versions.loc[legacy_rows, field], field).all():
            raise RuntimeError(f"legacy reference release rows must set {field}=true")
    if _strict_bool(
        versions["allowed_for_first_robot_trial"],
        "allowed_for_first_robot_trial",
    ).any():
        raise RuntimeError("reference freeze cannot approve a first robot trial")

    trajectory = pd.read_csv(release_path)
    audit = audit_frozen_reference(
        trajectory,
        manifest,
        metadata,
        source_bytes_match_release=True,
    )
    if not audit.valid:
        raise RuntimeError(f"frozen reference invariant failure: {audit}")
    return FrozenReferenceBundle(
        trajectory=trajectory,
        manifest=dict(manifest),
        metadata=dict(metadata),
        audit=audit,
    )


def require_parent_reference(metadata: Mapping[str, object]) -> None:
    """Reject a final-result metadata payload without the frozen parent binding."""

    if metadata.get("parent_reference_id") != ACTIVE_REFERENCE_ID:
        raise RuntimeError("final result parent_reference_id is missing or invalid")
    if metadata.get("parent_reference_sha256") != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("final result parent_reference_sha256 is missing or invalid")


def load_final_result_metadata(path: str | Path) -> dict[str, Any]:
    """Strict final-result JSON loader with mandatory frozen-parent binding."""

    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("final result metadata root must be an object")
    require_parent_reference(payload)
    return payload


__all__ = [
    "FrozenReferenceBundle",
    "REFERENCE_RELEASE_VERSION",
    "RELEASE_ACTIVE_REFERENCE_PATH",
    "RELEASE_CLOSURE_AUDIT_PATH",
    "RELEASE_DIRECTORY",
    "RELEASE_METADATA_PATH",
    "RELEASE_SHA256SUMS_PATH",
    "RELEASE_VERSION_MANIFEST_PATH",
    "ReferenceFreezeAudit",
    "audit_frozen_reference",
    "load_frozen_active_reference",
    "load_final_result_metadata",
    "load_reference_release_manifest",
    "require_parent_reference",
    "verify_reference_sha256",
]
