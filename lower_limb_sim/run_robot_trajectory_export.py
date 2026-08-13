"""Stage 6A offline H-to-B trajectory export and dry-run validation.

This module creates a *file* for future review.  It contains no robot SDK
imports and has no code path that connects, powers, servos or moves a robot.
Experiment-specific H/B/T calibration is mandatory; the module deliberately
has no default laboratory transform.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence, TextIO

import numpy as np
import pandas as pd

from .config import L1, L2
from .kinematics import forward_kinematics
from .formal_protocol import REFERENCE_RELEASE_MANIFEST_PATH
from .reference_execution_trajectory import (
    CLOSED_REFERENCE,
    retime_closed_reference,
)
from .reference_closed_c2 import C2_REFERENCE
from .reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
)
from .reference_release import (
    RELEASE_ACTIVE_REFERENCE_PATH,
    load_frozen_active_reference,
)
from .robot_coordinate_transform import (
    MODEL_ANGLE_DEFINITION,
    RobotFrameCalibration,
    human_pull_points_to_base,
    load_calibration_json,
    pull_points_base_to_tcp_origins,
)
from .robot_trajectory_audit import (
    RobotTrajectoryAudit,
    audit_robot_trajectory,
    validate_dry_run_command_file,
)


STAGE6A_SOFTWARE_VERSION = "lower_limb_sim_stage6a_v2"
STAGE5C_PCHIP_REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_candidates"
    / "reference_execution_versions.csv"
)
DEFAULT_REFERENCE_PATH = RELEASE_ACTIVE_REFERENCE_PATH
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "data" / "robot_trajectories"
)
COMMAND_FILENAME = "reference_robot_trajectory.csv"
METADATA_FILENAME = "metadata.json"
FIGURE_FILENAMES = (
    "robot_trajectory_preview.png",
    "robot_workspace_preview.png",
    "human_vs_robot_coordinate_preview.png",
)
REQUIRED_CALIBRATION_FIELDS = (
    "hip_center_in_base_m",
    "human_x_axis_in_base",
    "human_z_axis_in_base",
    "tool_offset_m",
    "tcp_orientation",
    "approved_hip_rom_deg",
    "approved_knee_rom_deg",
    "reviewed",
    "notes",
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTECTED_OUTPUT_DIRECTORIES = tuple(
    PROJECT_ROOT / name
    for name in ("hardware", "control", "collection", "safety", "config", "scripts")
)

SOURCE_APPROVED_ROM_ALIASES = {
    "source_approved_hip_min_deg": (
        "source_approved_hip_min_deg",
        "approved_hip_min_deg",
        "q_hip_approved_min_deg",
    ),
    "source_approved_hip_max_deg": (
        "source_approved_hip_max_deg",
        "approved_hip_max_deg",
        "q_hip_approved_max_deg",
    ),
    "source_approved_knee_min_deg": (
        "source_approved_knee_min_deg",
        "approved_knee_min_deg",
        "q_knee_approved_min_deg",
    ),
    "source_approved_knee_max_deg": (
        "source_approved_knee_max_deg",
        "approved_knee_max_deg",
        "q_knee_approved_max_deg",
    ),
}
ROM_COMPARISON_TOLERANCE_DEG = 1e-12


@dataclass(frozen=True)
class Stage6AExportResult:
    """In-memory result and every persisted Stage-6A artifact."""

    trajectory: pd.DataFrame
    audit: RobotTrajectoryAudit | None
    metadata: dict[str, object]
    output_paths: dict[str, Path]
    visualization_paths: dict[str, Path]
    skipped_visualizations: dict[str, str]
    blocked: bool
    block_reasons: tuple[str, ...]


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibration_sha256(calibration: RobotFrameCalibration) -> str:
    encoded = json.dumps(
        calibration.as_metadata_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_output_directory(value: str | Path) -> Path:
    destination = Path(value).expanduser().resolve()
    for protected in PROTECTED_OUTPUT_DIRECTORIES:
        protected_resolved = protected.resolve()
        if destination == protected_resolved or destination.is_relative_to(
            protected_resolved
        ):
            raise ValueError(
                "Stage 6A refuses to write inside protected robot/configuration "
                f"directory: {protected_resolved}"
            )
    return destination


def _append_reason(values: np.ndarray, mask: np.ndarray, token: str) -> None:
    for index in np.flatnonzero(mask):
        current = str(values[index])
        values[index] = f"{current};{token}" if current else token


def _strict_bool_values(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse bool/True/False/1/0 and reject every other encoding."""

    if pd.api.types.is_bool_dtype(values.dtype):
        recognized = values.notna().to_numpy(dtype=bool)
        parsed = values.fillna(False).to_numpy(dtype=bool)
        return parsed, recognized
    normalized = values.astype("string").str.strip().str.lower()
    tokens = normalized.fillna("").to_numpy(dtype=str)
    recognized = np.isin(tokens, ("true", "false", "1", "0"))
    parsed = np.isin(tokens, ("true", "1"))
    return parsed, recognized


def _normalize_source_approved_rom(
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[float, float], tuple[float, float]]:
    """Normalize run-local ROM approval fields without inventing defaults.

    Stage 5C and the C2 artifact use different persisted column names.  Every
    accepted source must nevertheless carry one finite, constant approval for
    each joint.  Conflicting aliases are rejected rather than resolved by
    column precedence.
    """

    normalized = reference.copy(deep=True)
    if normalized.empty:
        raise ValueError("reference trajectory must contain at least one sample.")
    resolved: dict[str, float] = {}
    for target, aliases in SOURCE_APPROVED_ROM_ALIASES.items():
        present = [column for column in aliases if column in normalized.columns]
        if not present:
            raise ValueError(
                "reference trajectory is missing explicit source ROM approval "
                f"field for {target}."
            )
        alias_values: list[float] = []
        for column in present:
            values = pd.to_numeric(normalized[column], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"source ROM approval field {column} must contain finite values."
                )
            first = float(values[0])
            if not np.allclose(
                values,
                first,
                atol=ROM_COMPARISON_TOLERANCE_DEG,
                rtol=0.0,
            ):
                raise ValueError(
                    f"source ROM approval field {column} must be constant."
                )
            alias_values.append(first)
        if not np.allclose(
            alias_values,
            alias_values[0],
            atol=ROM_COMPARISON_TOLERANCE_DEG,
            rtol=0.0,
        ):
            raise ValueError(
                f"conflicting source ROM approval aliases for {target}: {present}."
            )
        resolved[target] = alias_values[0]
        normalized[target] = alias_values[0]

    hip_rom = (
        resolved["source_approved_hip_min_deg"],
        resolved["source_approved_hip_max_deg"],
    )
    knee_rom = (
        resolved["source_approved_knee_min_deg"],
        resolved["source_approved_knee_max_deg"],
    )
    if not hip_rom[0] < hip_rom[1]:
        raise ValueError("source approved hip ROM must be strictly increasing.")
    if not knee_rom[0] < knee_rom[1]:
        raise ValueError("source approved knee ROM must be strictly increasing.")
    return normalized, hip_rom, knee_rom


def _source_validity(reference: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(reference)
    validity_field_missing = False
    if "trajectory_sample_valid" in reference:
        sample_valid, sample_encoding_valid = _strict_bool_values(
            reference["trajectory_sample_valid"]
        )
    elif "source_angle_valid" in reference:
        sample_valid, sample_encoding_valid = _strict_bool_values(
            reference["source_angle_valid"]
        )
    else:
        sample_valid = np.zeros(count, dtype=bool)
        sample_encoding_valid = np.zeros(count, dtype=bool)
        validity_field_missing = True

    if "invalid_reason" in reference:
        reasons = reference["invalid_reason"].fillna("").astype(str).to_numpy(object)
    else:
        reasons = np.full(count, "", dtype=object)
    if validity_field_missing:
        _append_reason(
            reasons,
            np.ones(count, dtype=bool),
            "source_sample_validity_missing",
        )
    else:
        _append_reason(
            reasons,
            ~sample_encoding_valid,
            "source_sample_validity_encoding_invalid",
        )
        sample_valid &= sample_encoding_valid

    # A missing upstream formal gate is not interpreted as approval.
    if "formal_execution_allowed" in reference:
        formal_allowed, formal_encoding_valid = _strict_bool_values(
            reference["formal_execution_allowed"]
        )
        _append_reason(
            reasons,
            ~formal_encoding_valid,
            "source_formal_gate_encoding_invalid",
        )
        formal_allowed &= formal_encoding_valid
    else:
        formal_allowed = np.zeros(count, dtype=bool)
    return sample_valid, reasons.copy(), formal_allowed


def load_closed_reference_trajectory(
    input_reference: str | Path = DEFAULT_REFERENCE_PATH,
    *,
    samples_per_segment: int = 201,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a timed closed reference or reuse Stage 5C's slow retimer.

    The persistent Stage-5C version is phase-only.  In that case this function
    reuses the existing minimum-jerk implementation with the already-defined
    slow schedule (12 s flexion + 12 s extension); it does not reconstruct the
    measured asymmetric cycle and does not invent the original skeleton FPS.
    """

    path = Path(input_reference).expanduser().resolve()
    frozen_bundle = None
    if path == DEFAULT_REFERENCE_PATH.resolve():
        frozen_bundle = load_frozen_active_reference(path)
    if not path.is_file():
        raise FileNotFoundError(f"closed reference file does not exist: {path}")
    reference = (
        frozen_bundle.trajectory.copy(deep=True)
        if frozen_bundle is not None
        else pd.read_csv(path)
    )
    reference_version_tag_present = "reference_version" in reference
    selected_reference_version: str | None = None
    if reference_version_tag_present:
        versions = set(reference["reference_version"].astype(str))
        if MEASURED_ASYMMETRIC_CLOSED_REFERENCE in versions:
            selected_reference_version = MEASURED_ASYMMETRIC_CLOSED_REFERENCE
        elif C2_REFERENCE in versions:
            selected_reference_version = C2_REFERENCE
        elif CLOSED_REFERENCE in versions:
            selected_reference_version = CLOSED_REFERENCE
        else:
            raise ValueError(
                "reference file contains no supported closed reference version; "
                f"expected {MEASURED_ASYMMETRIC_CLOSED_REFERENCE!r}, "
                f"{C2_REFERENCE!r}, or {CLOSED_REFERENCE!r}."
            )
        reference = reference.loc[
            reference["reference_version"].astype(str).eq(
                selected_reference_version
            )
        ].copy(deep=True)

    reference, source_approved_hip_rom, source_approved_knee_rom = (
        _normalize_source_approved_rom(reference)
    )

    required_angles = {"q_hip_rad", "q_knee_rad"}
    missing_angles = required_angles.difference(reference.columns)
    if missing_angles:
        raise ValueError(f"closed reference is missing angles: {sorted(missing_angles)}")

    if "formal_execution_allowed" in reference:
        source_formal_values, source_formal_encoding = _strict_bool_values(
            reference["formal_execution_allowed"]
        )
        source_formal_all = bool(
            source_formal_encoding.all() and source_formal_values.all()
        )
    else:
        source_formal_all = False
    stage6a_retimed = "time_s" not in reference.columns
    if stage6a_retimed:
        required_phase = {
            "cycle_phase",
            "segment_phase",
            "q_hip_reference_rad",
            "q_knee_reference_rad",
            "formal_execution_allowed",
        }
        missing_phase = required_phase.difference(reference.columns)
        if missing_phase:
            raise ValueError(
                "phase-only closed reference cannot be retimed; missing: "
                f"{sorted(missing_phase)}"
            )
        reference = retime_closed_reference(
            reference.reset_index(drop=True),
            profile="stage6a_closed_slow",
            flexion_duration_s=12.0,
            extension_duration_s=12.0,
            samples_per_segment=samples_per_segment,
        )
        # The Stage-5C helper historically copied the first row.  Stage 6A is
        # fail-closed: a mixed upstream gate can never become globally true.
        reference["formal_execution_allowed"] = source_formal_all
    else:
        reference = reference.reset_index(drop=True)

    reference["source_approved_hip_min_deg"] = source_approved_hip_rom[0]
    reference["source_approved_hip_max_deg"] = source_approved_hip_rom[1]
    reference["source_approved_knee_min_deg"] = source_approved_knee_rom[0]
    reference["source_approved_knee_max_deg"] = source_approved_knee_rom[1]

    time_s = reference["time_s"].to_numpy(dtype=float)
    if len(reference) < 3 or not np.isfinite(time_s).all():
        raise ValueError("closed reference needs at least three finite timed samples.")
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError("closed reference time_s must be strictly increasing.")

    q_hip = reference["q_hip_rad"].to_numpy(dtype=float)
    q_knee = reference["q_knee_rad"].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack((q_hip, q_knee))).all():
        raise ValueError("closed reference joint angles must be finite.")
    expected_theta = q_hip - q_knee
    if "theta_shank_rad" in reference and not np.allclose(
        reference["theta_shank_rad"].to_numpy(dtype=float),
        expected_theta,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("closed reference violates theta_shank = q_hip - q_knee.")
    reference["theta_shank_rad"] = expected_theta

    x_knee_fk, z_knee_fk, x_pull_fk, z_pull_fk = forward_kinematics(
        q_hip, q_knee, L1, L2
    )
    if "x_pull_human_m" not in reference:
        if "x_pull_m" in reference:
            reference["x_pull_human_m"] = reference["x_pull_m"].to_numpy(float)
        else:
            reference["x_pull_human_m"] = x_pull_fk
    if "z_pull_human_m" not in reference:
        if "z_pull_m" in reference:
            reference["z_pull_human_m"] = reference["z_pull_m"].to_numpy(float)
        else:
            reference["z_pull_human_m"] = z_pull_fk
    reference["x_knee_fk_m"] = x_knee_fk
    reference["z_knee_fk_m"] = z_knee_fk
    reference["x_pull_fk_m"] = x_pull_fk
    reference["z_pull_fk_m"] = z_pull_fk

    source_metadata: dict[str, object] = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "reference_version": (
            selected_reference_version if reference_version_tag_present else None
        ),
        "supported_reference_versions": [
            MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
            C2_REFERENCE,
            CLOSED_REFERENCE,
        ],
        "reference_version_verified": bool(
            reference_version_tag_present
            and selected_reference_version
            in (
                MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
                C2_REFERENCE,
                CLOSED_REFERENCE,
            )
        ),
        "source_approved_hip_rom_deg": list(source_approved_hip_rom),
        "source_approved_knee_rom_deg": list(source_approved_knee_rom),
        "source_approved_rom_normalized": True,
        "input_had_time_axis": not stage6a_retimed,
        "stage6a_retimed_from_phase": stage6a_retimed,
        "retiming_profile": "slow_minimum_jerk_12s_flexion_12s_extension"
        if stage6a_retimed
        else "provided_by_input",
        "retimed_timing_is_original": False
        if stage6a_retimed
        else bool(reference.get("retimed_timing_is_original", False).iloc[0])
        if "retimed_timing_is_original" in reference
        else None,
        "sample_count": int(len(reference)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "source_formal_execution_allowed_all": source_formal_all,
    }
    if frozen_bundle is not None:
        source_metadata.update(
            {
                "parent_reference_id": frozen_bundle.manifest["reference_id"],
                "parent_reference_sha256": frozen_bundle.manifest["sha256"],
                "reference_release_version": frozen_bundle.manifest[
                    "reference_version"
                ],
                "reference_release_manifest": str(
                    REFERENCE_RELEASE_MANIFEST_PATH.resolve()
                ),
                "approved_for_first_robot_trial": frozen_bundle.manifest[
                    "approved_for_first_robot_trial"
                ],
                "robot_execution_status": frozen_bundle.manifest[
                    "robot_execution_status"
                ],
            }
        )
    return reference, source_metadata


def build_robot_trajectory(
    reference: pd.DataFrame,
    calibration: RobotFrameCalibration,
    *,
    forward_kinematics_tolerance_m: float = 1e-10,
) -> tuple[pd.DataFrame, RobotTrajectoryAudit, dict[str, object]]:
    """Transform a closed H-frame reference and attach the offline audit."""

    required = {
        "time_s",
        "q_hip_rad",
        "q_knee_rad",
        "x_pull_human_m",
        "z_pull_human_m",
    }
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"reference trajectory missing fields: {sorted(missing)}")
    tolerance = float(forward_kinematics_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("forward_kinematics_tolerance_m must be finite and positive.")
    if calibration.tcp_orientation.representation != "euler_xyz_rad":
        raise ValueError(
            "Stage 6A command export requires tcp_orientation representation "
            "euler_xyz_rad so tcp_rx/tcp_ry/tcp_rz remain unambiguous active "
            "XYZ-Euler/RPY components."
        )

    source, source_approved_hip_rom, source_approved_knee_rom = (
        _normalize_source_approved_rom(reference.reset_index(drop=True))
    )
    q_hip = source["q_hip_rad"].to_numpy(dtype=float)
    q_knee = source["q_knee_rad"].to_numpy(dtype=float)
    x_h = source["x_pull_human_m"].to_numpy(dtype=float)
    z_h = source["z_pull_human_m"].to_numpy(dtype=float)
    _, _, x_pull_fk, z_pull_fk = forward_kinematics(q_hip, q_knee, L1, L2)
    fk_error = np.hypot(x_h - x_pull_fk, z_h - z_pull_fk)
    fk_consistent = np.isfinite(fk_error) & (fk_error <= tolerance)

    source_valid, source_reasons, formal_allowed = _source_validity(source)
    calibration_hip_rom = tuple(map(float, calibration.approved_hip_rom_deg))
    calibration_knee_rom = tuple(map(float, calibration.approved_knee_rom_deg))
    hip_approval_matches = bool(
        np.allclose(
            source_approved_hip_rom,
            calibration_hip_rom,
            atol=ROM_COMPARISON_TOLERANCE_DEG,
            rtol=0.0,
        )
    )
    knee_approval_matches = bool(
        np.allclose(
            source_approved_knee_rom,
            calibration_knee_rom,
            atol=ROM_COMPARISON_TOLERANCE_DEG,
            rtol=0.0,
        )
    )
    if not hip_approval_matches:
        _append_reason(
            source_reasons,
            np.ones(len(source), dtype=bool),
            "source_approved_hip_rom_mismatch_with_calibration",
        )
        source_valid[:] = False
    if not knee_approval_matches:
        _append_reason(
            source_reasons,
            np.ones(len(source), dtype=bool),
            "source_approved_knee_rom_mismatch_with_calibration",
        )
        source_valid[:] = False

    hip_lower, hip_upper = np.deg2rad(calibration_hip_rom)
    knee_lower, knee_upper = np.deg2rad(calibration_knee_rom)
    hip_rom_valid = np.isfinite(q_hip) & (q_hip >= hip_lower - 1e-12) & (
        q_hip <= hip_upper + 1e-12
    )
    knee_rom_valid = np.isfinite(q_knee) & (q_knee >= knee_lower - 1e-12) & (
        q_knee <= knee_upper + 1e-12
    )
    _append_reason(
        source_reasons,
        ~hip_rom_valid,
        "hip_outside_calibration_approved_rom",
    )
    _append_reason(
        source_reasons,
        ~knee_rom_valid,
        "knee_outside_calibration_approved_rom",
    )
    source_valid &= hip_rom_valid & knee_rom_valid
    _append_reason(
        source_reasons,
        ~fk_consistent,
        "human_pull_point_inconsistent_with_existing_forward_kinematics",
    )
    source_valid &= fk_consistent

    pull_base = human_pull_points_to_base(x_h, z_h, calibration)
    tcp_base = pull_points_base_to_tcp_origins(pull_base, calibration)
    orientation = np.asarray(calibration.tcp_orientation.values_rad, dtype=float)

    command = pd.DataFrame(
        {
            "time_s": source["time_s"].to_numpy(dtype=float),
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "theta_shank_rad": q_hip - q_knee,
            "x_pull_human_m": x_h,
            "z_pull_human_m": z_h,
            "pull_x_base_m": pull_base[:, 0],
            "pull_y_base_m": pull_base[:, 1],
            "pull_z_base_m": pull_base[:, 2],
            "tcp_x_base_m": tcp_base[:, 0],
            "tcp_y_base_m": tcp_base[:, 1],
            "tcp_z_base_m": tcp_base[:, 2],
            "tcp_rx_rad": np.full(len(source), orientation[0]),
            "tcp_ry_rad": np.full(len(source), orientation[1]),
            "tcp_rz_rad": np.full(len(source), orientation[2]),
            "tcp_orientation_representation": calibration.tcp_orientation.representation,
            "source_trajectory_valid": source_valid,
            "source_reference_formal_execution_allowed": formal_allowed,
            "source_approved_hip_min_deg": source_approved_hip_rom[0],
            "source_approved_hip_max_deg": source_approved_hip_rom[1],
            "source_approved_knee_min_deg": source_approved_knee_rom[0],
            "source_approved_knee_max_deg": source_approved_knee_rom[1],
            "calibration_approved_hip_min_deg": calibration_hip_rom[0],
            "calibration_approved_hip_max_deg": calibration_hip_rom[1],
            "calibration_approved_knee_min_deg": calibration_knee_rom[0],
            "calibration_approved_knee_max_deg": calibration_knee_rom[1],
            "robot_execution_approved": np.zeros(len(source), dtype=bool),
            "trajectory_generated_offline_only": np.ones(len(source), dtype=bool),
            "source_invalid_reason": source_reasons.astype(str),
            "forward_kinematics_consistency_error_m": fk_error,
            "model_angle_definition": MODEL_ANGLE_DEFINITION,
        }
    )
    audited, audit = audit_robot_trajectory(command, calibration)
    transform_details = {
        "forward_kinematics_tolerance_m": tolerance,
        "maximum_forward_kinematics_consistency_error_m": float(np.max(fk_error)),
        "all_pull_points_consistent_with_forward_kinematics": bool(
            fk_consistent.all()
        ),
        "source_formal_execution_allowed_all": bool(formal_allowed.all()),
        "source_approved_hip_rom_deg": list(source_approved_hip_rom),
        "source_approved_knee_rom_deg": list(source_approved_knee_rom),
        "calibration_approved_hip_rom_deg": list(calibration_hip_rom),
        "calibration_approved_knee_rom_deg": list(calibration_knee_rom),
        "source_calibration_hip_rom_exact_match": hip_approval_matches,
        "source_calibration_knee_rom_exact_match": knee_approval_matches,
        "hip_rom_violation_sample_count": int((~hip_rom_valid).sum()),
        "knee_rom_violation_sample_count": int((~knee_rom_valid).sum()),
        "source_sample_validity_field_present": bool(
            "trajectory_sample_valid" in source or "source_angle_valid" in source
        ),
        "source_valid_sample_count_before_transform": int(source_valid.sum()),
        "source_invalid_sample_count_before_transform": int((~source_valid).sum()),
    }
    return audited, audit, transform_details


def _base_metadata(
    *,
    source_metadata: Mapping[str, object] | None,
    calibration: RobotFrameCalibration | None,
    calibration_source: str | None,
    blocked: bool,
    block_reasons: Sequence[str],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "stage": "6A_human_to_robot_base_offline_transform_and_preexecution_audit",
        "software_version": STAGE6A_SOFTWARE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_status": "blocked" if blocked else "generated_for_offline_audit",
        "block_reasons": list(block_reasons),
        "model_angle_definition": MODEL_ANGLE_DEFINITION,
        "coordinate_frames": {
            "H": "human_local_sagittal_frame_with_pull_point_[x,0,z]",
            "B": "ROKAE_Base_frame_from_explicit_offline_calibration",
            "T": "TCP_tool_frame_with_strap_connection_offset",
        },
        "human_to_base_formula": (
            "p_pull_B = hip_center_B + R_base_from_human @ [x_pull_H, 0, z_pull_H]"
        ),
        "human_axis_mapping_semantics": (
            "positive_H_x_and_H_z_displacements_map_in_the_same_direction_as_"
            "human_x_axis_in_base_and_human_z_axis_in_base"
        ),
        "tcp_formula": "p_tcp_B = p_pull_B - R_base_from_tcp @ tool_offset_T",
        "tool_offset_definition": (
            "vector_from_tcp_origin_to_actual_strap_connection_pull_point_"
            "expressed_in_T"
        ),
        "robot_execution_approved": False,
        "trajectory_generated_offline_only": True,
        "real_robot_connected": False,
        "robot_sdk_imported": False,
        "robot_connection_attempted": False,
        "robot_servo_power_or_motion_command_sent": False,
        "real_robot_safety_thresholds_configured": False,
        "real_robot_velocity_limit_m_s": None,
        "real_robot_acceleration_limit_m_s2": None,
        "real_robot_workspace_limits_base_m": None,
        "reported_extrema_are_safety_limits": False,
        "source_reference": dict(source_metadata or {}),
    }
    if calibration is None:
        metadata["calibration"] = None
        metadata["calibration_reviewed"] = False
        metadata["calibration_required_fields"] = list(REQUIRED_CALIBRATION_FIELDS)
        metadata["laboratory_coordinates_hardcoded"] = False
    else:
        calibration_metadata = calibration.as_metadata_dict()
        metadata["calibration"] = calibration_metadata
        metadata["calibration_reviewed"] = calibration.reviewed
        metadata["calibration_source"] = calibration_source
        metadata["calibration_parameter_sha256"] = _calibration_sha256(calibration)
        if calibration_source is not None:
            source_path = Path(calibration_source).expanduser()
            if source_path.is_file():
                metadata["calibration_source_file_sha256"] = _sha256_file(source_path)
        metadata["tcp_orientation_representation"] = (
            calibration.tcp_orientation.representation
        )
        metadata["tcp_orientation_vendor_pose_semantics_verified"] = False
    return metadata


def run_robot_trajectory_export(
    *,
    input_reference: str | Path = DEFAULT_REFERENCE_PATH,
    calibration: RobotFrameCalibration | None = None,
    calibration_source: str | None = None,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    samples_per_segment: int = 201,
    save_outputs: bool = True,
    generate_plots: bool = True,
) -> Stage6AExportResult:
    """Generate the offline file, or save a fail-closed calibration audit."""

    destination = (
        _validated_output_directory(output_directory)
        if save_outputs
        else Path(output_directory).expanduser()
    )
    reference, source_metadata = load_closed_reference_trajectory(
        input_reference, samples_per_segment=samples_per_segment
    )
    output_paths: dict[str, Path] = {}
    visual_paths: dict[str, Path] = {}
    skipped_visualizations: dict[str, str] = {}

    if calibration is None:
        reasons = ("explicit_H_B_T_calibration_missing",)
        metadata = _base_metadata(
            source_metadata=source_metadata,
            calibration=None,
            calibration_source=None,
            blocked=True,
            block_reasons=reasons,
        )
        metadata["preexecution_audit_passed"] = False
        metadata["reference_robot_trajectory_csv_generated"] = False
        metadata["visualizations_generated"] = False
        metadata["visualization_skip_reasons"] = {
            filename: "explicit calibration is missing" for filename in FIGURE_FILENAMES
        }
        if save_outputs:
            destination.mkdir(parents=True, exist_ok=True)
            metadata_path = destination / METADATA_FILENAME
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(_json_ready(metadata), handle, indent=2, ensure_ascii=False)
            output_paths["metadata"] = metadata_path
        return Stage6AExportResult(
            trajectory=pd.DataFrame(),
            audit=None,
            metadata=metadata,
            output_paths=output_paths,
            visualization_paths=visual_paths,
            skipped_visualizations=dict(metadata["visualization_skip_reasons"]),
            blocked=True,
            block_reasons=reasons,
        )

    trajectory, audit, transform_details = build_robot_trajectory(
        reference, calibration
    )
    block_reasons: list[str] = []
    if not bool(transform_details["source_calibration_hip_rom_exact_match"]):
        block_reasons.append("source_calibration_hip_rom_approval_mismatch")
    if not bool(transform_details["source_calibration_knee_rom_exact_match"]):
        block_reasons.append("source_calibration_knee_rom_approval_mismatch")
    if int(transform_details["hip_rom_violation_sample_count"]) > 0:
        block_reasons.append("hip_outside_calibration_approved_rom")
    if int(transform_details["knee_rom_violation_sample_count"]) > 0:
        block_reasons.append("knee_outside_calibration_approved_rom")
    if not bool(transform_details["source_formal_execution_allowed_all"]):
        block_reasons.append("source_reference_formal_gate_not_approved")
    if int(transform_details["source_invalid_sample_count_before_transform"]) > 0:
        block_reasons.append("source_reference_contains_invalid_samples")
    if not audit.all_samples_finite:
        block_reasons.append("non_finite_trajectory_samples")
    if not audit.position_continuous:
        block_reasons.append("tcp_position_continuity_audit_failed")
    if not audit.velocity_continuous:
        block_reasons.append("tcp_velocity_continuity_audit_failed")
    if not audit.acceleration_continuous:
        block_reasons.append("tcp_acceleration_continuity_audit_failed")
    if not audit.start_end_closed:
        block_reasons.append("tcp_start_end_closure_audit_failed")
    if not audit.transform_is_orthogonal:
        block_reasons.append("coordinate_transform_orthogonality_audit_failed")
    if not audit.tool_offset_correctly_applied:
        block_reasons.append("tool_offset_application_audit_failed")
    if not audit.theta_shank_definition_valid:
        block_reasons.append("theta_shank_definition_audit_failed")
    if not audit.trajectory_all_samples_valid and not block_reasons:
        block_reasons.append("trajectory_contains_invalid_samples")
    preexecution_blocked = bool(block_reasons)

    metadata = _base_metadata(
        source_metadata=source_metadata,
        calibration=calibration,
        calibration_source=calibration_source,
        blocked=preexecution_blocked,
        block_reasons=block_reasons,
    )
    metadata["generation_status"] = (
        "generated_but_preexecution_audit_blocked"
        if preexecution_blocked
        else "generated_for_offline_audit"
    )
    metadata["preexecution_audit_passed"] = not preexecution_blocked
    metadata["transform_audit"] = transform_details
    metadata["trajectory_audit"] = audit.as_dict()
    metadata["reference_robot_trajectory_csv_generated"] = bool(save_outputs)

    if save_outputs:
        destination.mkdir(parents=True, exist_ok=True)
        command_path = destination / COMMAND_FILENAME
        trajectory.to_csv(command_path, index=False)
        output_paths["reference_robot_trajectory"] = command_path
    if generate_plots and save_outputs:
        # Deliberately lazy: dry-run and blocked-without-calibration paths do
        # not import matplotlib or create its cache directory.
        from .visualize_robot_trajectory import (
            generate_robot_trajectory_visualizations,
        )

        visual_result = generate_robot_trajectory_visualizations(
            trajectory, metadata, destination
        )
        visual_paths = dict(visual_result.paths)
        skipped_visualizations = dict(visual_result.skipped)
    elif generate_plots:
        skipped_visualizations = {
            filename: "save_outputs is false; no visualization files were written"
            for filename in FIGURE_FILENAMES
        }
    else:
        skipped_visualizations = {
            filename: "plot generation disabled by caller" for filename in FIGURE_FILENAMES
        }
    metadata["visualization_paths"] = {
        key: str(value) for key, value in visual_paths.items()
    }
    metadata["visualization_skip_reasons"] = skipped_visualizations
    metadata["all_requested_visualizations_generated"] = not skipped_visualizations

    if save_outputs:
        metadata_path = destination / METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_ready(metadata), handle, indent=2, ensure_ascii=False)
        output_paths["metadata"] = metadata_path
    return Stage6AExportResult(
        trajectory=trajectory,
        audit=audit,
        metadata=metadata,
        output_paths=output_paths,
        visualization_paths=visual_paths,
        skipped_visualizations=skipped_visualizations,
        blocked=preexecution_blocked,
        block_reasons=tuple(block_reasons),
    )


def dry_run_robot_trajectory(
    input_trajectory: str | Path,
    *,
    print_samples: int = 0,
    stream: TextIO | None = None,
) -> dict[str, object]:
    """Read and validate a command file without timing playback or robot I/O."""

    path = Path(input_trajectory).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dry-run trajectory file does not exist: {path}")
    dataframe = pd.read_csv(path)
    result = validate_dry_run_command_file(dataframe)
    result["input_path"] = str(path)
    result["input_sha256"] = _sha256_file(path)
    result["rows_printed"] = 0
    destination = sys.stdout if stream is None else stream
    count = int(print_samples)
    if count < -1:
        raise ValueError("print_samples must be -1 (all), zero, or a positive count.")
    if count != 0:
        selected = dataframe if count == -1 else dataframe.head(count)
        columns = [
            "time_s",
            "tcp_x_base_m",
            "tcp_y_base_m",
            "tcp_z_base_m",
            "tcp_rx_rad",
            "tcp_ry_rad",
            "tcp_rz_rad",
            "trajectory_valid",
            "invalid_reason",
        ]
        columns = [name for name in columns if name in selected]
        print(selected.loc[:, columns].to_string(index=False), file=destination)
        result["rows_printed"] = int(len(selected))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 6A offline human-to-Base trajectory export. This command never "
            "imports a robot SDK or sends a robot command."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--calibration-json")
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--samples-per-segment", type=int, default=201)
    parser.add_argument(
        "--dry-run",
        metavar="REFERENCE_ROBOT_TRAJECTORY_CSV",
        help="read/print/validate an existing command CSV; performs no timed playback",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=0,
        help="dry-run rows to print; -1 prints all, 0 only validates",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        if args.calibration_json:
            raise SystemExit("--calibration-json is not used in read-only dry-run mode.")
        result = dry_run_robot_trajectory(
            args.dry_run, print_samples=args.print_samples
        )
        print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False))
        return 0 if bool(result["dry_run_valid"]) else 2

    calibration = None
    calibration_source = None
    if args.calibration_json:
        calibration_path = Path(args.calibration_json).expanduser().resolve()
        calibration = load_calibration_json(calibration_path)
        calibration_source = str(calibration_path)
    result = run_robot_trajectory_export(
        input_reference=args.input,
        calibration=calibration,
        calibration_source=calibration_source,
        output_directory=args.output_directory,
        samples_per_segment=args.samples_per_segment,
        generate_plots=not args.no_plots,
    )
    if result.blocked and result.audit is None:
        print(
            "Stage 6A blocked: explicit experiment calibration was not supplied; "
            "metadata was saved but no robot trajectory was fabricated."
        )
        return 2
    assert result.audit is not None
    print(json.dumps(_json_ready(result.audit.as_dict()), indent=2, ensure_ascii=False))
    print("Offline file only: robot_execution_approved = false")
    if result.blocked:
        print(
            "The audit CSV was saved, but pre-execution audit is blocked: "
            + "; ".join(result.block_reasons)
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMAND_FILENAME",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_REFERENCE_PATH",
    "STAGE5C_PCHIP_REFERENCE_PATH",
    "METADATA_FILENAME",
    "FIGURE_FILENAMES",
    "PROTECTED_OUTPUT_DIRECTORIES",
    "REQUIRED_CALIBRATION_FIELDS",
    "STAGE6A_SOFTWARE_VERSION",
    "SOURCE_APPROVED_ROM_ALIASES",
    "Stage6AExportResult",
    "build_robot_trajectory",
    "dry_run_robot_trajectory",
    "load_closed_reference_trajectory",
    "main",
    "run_robot_trajectory_export",
]
