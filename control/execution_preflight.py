"""Single fail-closed gate for the first supervised real robot trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from collection.real_robot_acquisition import AcquisitionHealth
from control.start_anchor import StartAnchor
from control.start_anchored_relative_trajectory import (
    ALLOWED_FIRST_ROBOT_TRIAL_TRAJECTORIES,
    APPROVED_FIRST_ROBOT_TRIAL_L1_M,
    APPROVED_FIRST_ROBOT_TRIAL_L2_M,
    APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256,
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    RehabFrameConfig,
    RelativeTrajectoryAudit,
)
from lower_limb_sim.reference_closed_c2 import (
    APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG,
)
from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)
from safety.experiment_safety import ExperimentSafetyConfig


EXECUTE_MODE = "execute"
OPERATOR_CONFIRMATION = "I CONFIRM SUPERVISED SLOW ROBOT MOTION"
_LIVE_PREFLIGHT_TOKEN = object()


def trajectory_execution_digest(trajectory: pd.DataFrame) -> str:
    """Bind a preflight result to one exact, ordered execution table."""

    if not isinstance(trajectory, pd.DataFrame):
        raise TypeError("trajectory must be a pandas DataFrame")
    descriptor = json.dumps(
        {
            "columns": [str(column) for column in trajectory.columns],
            "dtypes": [str(dtype) for dtype in trajectory.dtypes],
            "shape": list(trajectory.shape),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        row_hashes = pd.util.hash_pandas_object(
            trajectory,
            index=True,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory contains values that cannot be bound") from exc
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def experiment_safety_digest(safety: ExperimentSafetyConfig) -> str:
    """Bind a live approval to the exact reviewed safety snapshot."""

    if not isinstance(safety, ExperimentSafetyConfig):
        raise TypeError("safety must be ExperimentSafetyConfig")
    encoded = json.dumps(
        safety.to_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExecutionPreflight:
    allowed: bool
    reasons: tuple[str, ...]
    trajectory_id: str
    anchor_id: str
    maximum_tcp_speed_m_s: float | None
    maximum_tcp_acceleration_m_s2: float | None
    runtime_robot_summary: dict[str, object] | None = None
    evaluation_phase: str = "unbound"
    trajectory_sha256: str | None = None
    experiment_safety_sha256: str | None = None
    _live_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def require_allowed(self) -> None:
        if not self.allowed:
            raise PermissionError("real robot execution blocked: " + ";".join(self.reasons))

    def require_live_trajectory_binding(self, trajectory: pd.DataFrame) -> None:
        """Reject offline, manually constructed, or subsequently changed input."""

        if self.evaluation_phase != "live" or self._live_token is not _LIVE_PREFLIGHT_TOKEN:
            raise PermissionError("real robot execution blocked: preflight_is_not_live_bound")
        if not self.trajectory_sha256:
            raise PermissionError("real robot execution blocked: trajectory_binding_missing")
        if trajectory_execution_digest(trajectory) != self.trajectory_sha256:
            raise PermissionError("real robot execution blocked: trajectory_binding_mismatch")

    def require_safety_binding(self, safety: ExperimentSafetyConfig) -> None:
        if not self.experiment_safety_sha256:
            raise PermissionError("real robot execution blocked: safety_binding_missing")
        if experiment_safety_digest(safety) != self.experiment_safety_sha256:
            raise PermissionError("real robot execution blocked: safety_binding_mismatch")

    def as_metadata(self) -> dict[str, object]:
        """JSON-safe evidence without exposing the in-process validation token."""

        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "trajectory_id": self.trajectory_id,
            "anchor_id": self.anchor_id,
            "maximum_tcp_speed_m_s": self.maximum_tcp_speed_m_s,
            "maximum_tcp_acceleration_m_s2": self.maximum_tcp_acceleration_m_s2,
            "runtime_robot_summary": self.runtime_robot_summary,
            "evaluation_phase": self.evaluation_phase,
            "trajectory_sha256": self.trajectory_sha256,
            "experiment_safety_sha256": self.experiment_safety_sha256,
        }


def _connected(adapter: Any) -> bool:
    value = getattr(adapter, "is_connected", False)
    return bool(value() if callable(value) else value)


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _finite_vector(value: object, size: int) -> tuple[float, ...] | None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != size
    ):
        return None
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        parsed.append(number)
    return tuple(parsed)


def _soft_limits(value: object) -> tuple[tuple[float, float], ...] | None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 6
    ):
        return None
    parsed: list[tuple[float, float]] = []
    for pair in value:
        limits = _finite_vector(pair, 2)
        if limits is None or limits[0] >= limits[1]:
            return None
        parsed.append((limits[0], limits[1]))
    return tuple(parsed)


def _same_numbers(left: object, right: object, size: int) -> bool:
    parsed_left = _finite_vector(left, size)
    parsed_right = _finite_vector(right, size)
    if parsed_left is None or parsed_right is None:
        return False
    return bool(np.allclose(parsed_left, parsed_right, rtol=0.0, atol=1e-9))


def _same_soft_limits(left: object, right: object) -> bool:
    parsed_left = _soft_limits(left)
    parsed_right = _soft_limits(right)
    if parsed_left is None or parsed_right is None:
        return False
    return bool(
        np.allclose(
            np.asarray(parsed_left),
            np.asarray(parsed_right),
            rtol=0.0,
            atol=1e-9,
        )
    )


def audit_execution_trajectory_content(
    trajectory: pd.DataFrame,
    safety: ExperimentSafetyConfig,
    *,
    expected_trajectory_id: str | None = None,
    frame: RehabFrameConfig | None = None,
    anchor: StartAnchor | None = None,
    require_official_reference: bool = False,
) -> tuple[tuple[str, ...], float | None, float | None]:
    """Independently audit the exact table that may reach the scheduler."""

    reasons: list[str] = []

    def block(condition: bool, reason: str) -> None:
        if condition and reason not in reasons:
            reasons.append(reason)

    required = {
        "time_s",
        "trajectory_id",
        "delta_x_R",
        "delta_y_R",
        "delta_z_R",
        "tcp_x_base",
        "tcp_y_base",
        "tcp_z_base",
        "tcp_rx",
        "tcp_ry",
        "tcp_rz",
        "tcp_vx_base",
        "tcp_vy_base",
        "tcp_vz_base",
        "tcp_ax_base",
        "tcp_ay_base",
        "tcp_az_base",
        "q_hip_ref",
        "q_knee_ref",
        "theta_shank_ref",
        "trajectory_valid",
        "experiment_mode",
        "tcp_orientation_strategy",
    }
    missing = required.difference(trajectory.columns)
    block(bool(missing), "trajectory_execution_columns_missing")
    block(len(trajectory) < 3, "trajectory_has_too_few_samples")
    if missing or len(trajectory) < 3:
        return tuple(reasons), None, None

    trajectory_ids = trajectory["trajectory_id"].tolist()
    id_valid = all(
        isinstance(value, str) and bool(value.strip()) for value in trajectory_ids
    )
    unique_ids = set(trajectory_ids) if id_valid else set()
    block(not id_valid or len(unique_ids) != 1, "trajectory_id_column_invalid")
    table_trajectory_id = next(iter(unique_ids)) if len(unique_ids) == 1 else None
    if expected_trajectory_id is not None:
        block(
            table_trajectory_id != expected_trajectory_id,
            "trajectory_id_binding_mismatch",
        )
    block(
        table_trajectory_id not in ALLOWED_FIRST_ROBOT_TRIAL_TRAJECTORIES,
        "trajectory_not_whitelisted_for_first_robot_trial",
    )
    block(
        not math.isclose(
            L1,
            APPROVED_FIRST_ROBOT_TRIAL_L1_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            L2,
            APPROVED_FIRST_ROBOT_TRIAL_L2_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "equivalent_pull_point_geometry_not_approved",
    )

    valid_values = trajectory["trajectory_valid"].tolist()
    strict_bools = all(
        type(value) is bool or isinstance(value, np.bool_) for value in valid_values
    )
    block(not strict_bools, "trajectory_valid_column_not_strict_boolean")
    block(
        not strict_bools or not all(bool(value) for value in valid_values),
        "trajectory_rows_invalid",
    )
    block(
        not all(value == "start_anchored_relative" for value in trajectory["experiment_mode"]),
        "trajectory_experiment_mode_invalid",
    )
    block(
        not all(
            value == "fixed_at_start_anchor"
            for value in trajectory["tcp_orientation_strategy"]
        ),
        "trajectory_orientation_strategy_invalid",
    )

    numeric_columns = [
        "time_s",
        "delta_x_R",
        "delta_y_R",
        "delta_z_R",
        "tcp_x_base",
        "tcp_y_base",
        "tcp_z_base",
        "tcp_rx",
        "tcp_ry",
        "tcp_rz",
        "tcp_vx_base",
        "tcp_vy_base",
        "tcp_vz_base",
        "tcp_ax_base",
        "tcp_ay_base",
        "tcp_az_base",
        "q_hip_ref",
        "q_knee_ref",
        "theta_shank_ref",
    ]
    try:
        numeric = trajectory[numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError):
        block(True, "trajectory_execution_values_not_numeric")
        return tuple(reasons), None, None
    block(not bool(np.isfinite(numeric).all()), "trajectory_contains_non_finite_values")
    if not np.isfinite(numeric).all():
        return tuple(reasons), None, None

    time_s = trajectory["time_s"].to_numpy(dtype=float)
    time_valid = bool(np.all(np.diff(time_s) > 0.0))
    block(not time_valid, "trajectory_time_invalid")
    if not time_valid:
        return tuple(reasons), None, None
    q_hip = trajectory["q_hip_ref"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_ref"].to_numpy(dtype=float)
    theta = trajectory["theta_shank_ref"].to_numpy(dtype=float)
    hip_min, hip_max = np.deg2rad(APPROVED_HIP_ROM_DEG)
    knee_min, knee_max = np.deg2rad(APPROVED_KNEE_ROM_DEG)
    tolerance = 1e-10
    block(
        not bool(
            ((q_hip >= hip_min - tolerance) & (q_hip <= hip_max + tolerance)).all()
            and ((q_knee >= knee_min - tolerance) & (q_knee <= knee_max + tolerance)).all()
        ),
        "approved_rom_invalid",
    )
    block(
        not bool(np.allclose(theta, q_hip - q_knee, rtol=0.0, atol=tolerance)),
        "theta_shank_definition_invalid",
    )

    delta = trajectory[["delta_x_R", "delta_y_R", "delta_z_R"]].to_numpy(float)
    block(not bool(np.allclose(delta[0], 0.0, rtol=0.0, atol=tolerance)), "relative_start_not_zero")
    block(not bool(np.allclose(delta[-1], 0.0, rtol=0.0, atol=tolerance)), "trajectory_not_closed")
    _, _, pull_x, pull_z = forward_kinematics(q_hip, q_knee, L1, L2)
    expected_delta = np.column_stack(
        (
            np.asarray(pull_x, dtype=float) - float(pull_x[0]),
            np.zeros(len(trajectory)),
            np.asarray(pull_z, dtype=float) - float(pull_z[0]),
        )
    )
    if np.linalg.norm(expected_delta[-1]) <= tolerance:
        expected_delta[-1] = 0.0
    block(
        not bool(np.allclose(delta, expected_delta, rtol=0.0, atol=tolerance)),
        "trajectory_pull_point_fk_mismatch",
    )

    xyz = trajectory[["tcp_x_base", "tcp_y_base", "tcp_z_base"]].to_numpy(float)
    orientation = trajectory[["tcp_rx", "tcp_ry", "tcp_rz"]].to_numpy(float)
    block(
        not bool(np.allclose(xyz[0], xyz[-1], rtol=0.0, atol=tolerance)),
        "trajectory_end_not_anchor",
    )
    block(
        not bool(np.allclose(orientation, orientation[0], rtol=0.0, atol=tolerance)),
        "fixed_orientation_violation",
    )
    if frame is not None and anchor is not None:
        anchor_pose = np.asarray(anchor.tcp_pose_base, dtype=float)
        expected_xyz = anchor_pose[:3] + delta @ frame.rotation_base_from_rehab.T
        block(
            not bool(np.allclose(xyz, expected_xyz, rtol=0.0, atol=tolerance)),
            "trajectory_frame_transform_mismatch",
        )
        block(
            not bool(np.allclose(orientation, anchor_pose[3:], rtol=0.0, atol=tolerance)),
            "trajectory_orientation_not_anchor",
        )

    declared_velocity = trajectory[
        ["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]
    ].to_numpy(float)
    declared_acceleration = trajectory[
        ["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]
    ].to_numpy(float)
    derived_velocity = np.gradient(xyz, time_s, axis=0, edge_order=2)
    derived_acceleration = np.gradient(
        derived_velocity,
        time_s,
        axis=0,
        edge_order=2,
    )
    block(
        not bool(
            np.allclose(
                declared_velocity,
                derived_velocity,
                rtol=0.0,
                atol=1e-10,
            )
        ),
        "trajectory_velocity_columns_mismatch",
    )
    block(
        not bool(
            np.allclose(
                declared_acceleration,
                derived_acceleration,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "trajectory_acceleration_columns_mismatch",
    )
    max_speed = float(np.linalg.norm(derived_velocity, axis=1).max())
    max_acceleration = float(np.linalg.norm(derived_acceleration, axis=1).max())
    if safety.max_tcp_speed_m_s is not None:
        block(max_speed > safety.max_tcp_speed_m_s, "trajectory_speed_limit_exceeded")
    if safety.max_tcp_acceleration_m_s2 is not None:
        block(
            max_acceleration > safety.max_tcp_acceleration_m_s2,
            "trajectory_acceleration_limit_exceeded",
        )
    if safety.workspace_min_base_m is not None and safety.workspace_max_base_m is not None:
        lower = np.asarray(safety.workspace_min_base_m, dtype=float)
        upper = np.asarray(safety.workspace_max_base_m, dtype=float)
        block(
            not bool(((xyz >= lower) & (xyz <= upper)).all()),
            "trajectory_outside_reviewed_workspace",
        )

    if require_official_reference:
        official_match = False
        if table_trajectory_id == FIRST_ROBOT_TRIAL_TRAJECTORY_ID:
            try:
                official, official_metadata = load_closed_reference_trajectory(
                    DEFAULT_REFERENCE_PATH
                )
                official_ids = official["trajectory_id"].astype(str).unique().tolist()
                official_match = bool(
                    official_metadata.get("sha256")
                    == APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
                    and
                    official_ids == [FIRST_ROBOT_TRIAL_TRAJECTORY_ID]
                    and len(official) == len(trajectory)
                    and np.allclose(
                        official["time_s"].to_numpy(float),
                        time_s,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    and np.allclose(
                        official["q_hip_rad"].to_numpy(float),
                        q_hip,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    and np.allclose(
                        official["q_knee_rad"].to_numpy(float),
                        q_knee,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError):
                official_match = False
        block(not official_match, "trajectory_not_exact_official_slow_reference")

    return tuple(reasons), max_speed, max_acceleration


def evaluate_execution_preflight(
    *,
    mode: str,
    enable_motion: bool,
    operator_confirmation: str,
    requested_anchor_id: str,
    frame: RehabFrameConfig,
    anchor: StartAnchor,
    safety: ExperimentSafetyConfig,
    trajectory: pd.DataFrame,
    audit: RelativeTrajectoryAudit,
    acquisition_health: AcquisitionHealth,
    logger: Any,
    robot_adapter: Any,
    current_tcp_pose_base: Any | None = None,
    _evaluation_phase: str = "live",
) -> ExecutionPreflight:
    """Evaluate every offline, runtime, logging, and human-review gate."""
    if _evaluation_phase not in {"live", "offline"}:
        raise ValueError("execution preflight phase must be live or offline")
    reasons: list[str] = []

    def block(condition: bool, reason: str) -> None:
        if condition and reason not in reasons:
            reasons.append(reason)

    block(mode != EXECUTE_MODE, "mode_is_not_execute")
    block(type(enable_motion) is not bool or not enable_motion, "enable_motion_flag_missing")
    block(operator_confirmation != OPERATOR_CONFIRMATION, "operator_confirmation_missing")
    block(not frame.reviewed, "rehab_frame_not_reviewed")
    block(not anchor.reviewed, "start_anchor_not_reviewed")
    block(requested_anchor_id != anchor.anchor_id, "requested_anchor_id_mismatch")
    for reason in safety.execution_block_reasons():
        block(True, reason)
    adapter_connected = _connected(robot_adapter)
    block(not adapter_connected, "sdk_not_connected")
    block(not bool(getattr(logger, "ready", False)), "logger_not_ready")
    block(not bool(getattr(logger, "healthy", False)), "logger_not_healthy")

    runtime_summary: Mapping[str, object] | None = None
    if adapter_connected:
        summary_reader = getattr(robot_adapter, "get_robot_state_summary", None)
        if callable(summary_reader):
            try:
                candidate = summary_reader()
                if isinstance(candidate, Mapping):
                    runtime_summary = candidate
            except Exception:
                runtime_summary = None
    block(runtime_summary is None, "runtime_robot_summary_unavailable")
    metadata_value = (
        runtime_summary.get("robot_metadata") if runtime_summary is not None else None
    )
    metadata = metadata_value if isinstance(metadata_value, Mapping) else None
    block(metadata is None, "runtime_robot_metadata_unavailable")

    identity_bindings = (
        ("robot_model", "expected_robot_model"),
        ("robot_serial_number", "expected_robot_serial_number"),
        ("controller_version", "expected_controller_version"),
    )
    for metadata_name, expected_name in identity_bindings:
        anchor_value = _optional_text(getattr(anchor, metadata_name, None))
        expected_value = _optional_text(getattr(safety, expected_name, None))
        runtime_value = (
            _optional_text(metadata.get(metadata_name)) if metadata is not None else None
        )
        block(anchor_value is None, f"anchor_{metadata_name}_unavailable")
        block(runtime_value is None, f"runtime_{metadata_name}_unavailable")
        if expected_value is not None:
            block(anchor_value != expected_value, f"anchor_{metadata_name}_mismatch")
            block(runtime_value != expected_value, f"runtime_{metadata_name}_mismatch")

    expected_tool = _optional_text(safety.reviewed_tool_name)
    expected_workpiece = _optional_text(safety.reviewed_workpiece_name)
    anchor_tool = _optional_text(getattr(anchor, "tool_name", None))
    anchor_workpiece = _optional_text(getattr(anchor, "workpiece_name", None))
    block(anchor_tool is None, "anchor_tool_name_unavailable")
    block(anchor_workpiece is None, "anchor_workpiece_name_unavailable")
    if expected_tool is not None:
        block(anchor_tool != expected_tool, "anchor_tool_name_mismatch")
    if expected_workpiece is not None:
        block(anchor_workpiece != expected_workpiece, "anchor_workpiece_name_mismatch")

    payload_value = metadata.get("sdk_tool_payload") if metadata is not None else None
    payload = payload_value if isinstance(payload_value, Mapping) else None
    block(payload is None, "runtime_sdk_tool_payload_unavailable")
    available_tools = payload.get("sdk_available_tool_names") if payload is not None else None
    available_workpieces = (
        payload.get("sdk_available_workobject_names") if payload is not None else None
    )
    tool_names = (
        tuple(value for item in available_tools if (value := _optional_text(item)) is not None)
        if isinstance(available_tools, Sequence)
        and not isinstance(available_tools, (str, bytes))
        else None
    )
    workpiece_names = (
        tuple(
            value
            for item in available_workpieces
            if (value := _optional_text(item)) is not None
        )
        if isinstance(available_workpieces, Sequence)
        and not isinstance(available_workpieces, (str, bytes))
        else None
    )
    block(tool_names is None, "runtime_sdk_tool_names_unavailable")
    block(workpiece_names is None, "runtime_sdk_workpiece_names_unavailable")
    if expected_tool is not None and tool_names is not None:
        block(expected_tool not in tool_names, "reviewed_tool_name_not_reported_by_sdk")
    if expected_workpiece is not None and workpiece_names is not None:
        block(
            expected_workpiece not in workpiece_names,
            "reviewed_workpiece_name_not_reported_by_sdk",
        )

    runtime_mass = payload.get("toolset_load_mass_kg") if payload is not None else None
    runtime_cog = payload.get("toolset_load_cog_m") if payload is not None else None
    runtime_inertia = (
        payload.get("toolset_load_inertia_kg_m2") if payload is not None else None
    )
    mass_valid = (
        not isinstance(runtime_mass, bool)
        and isinstance(runtime_mass, (int, float))
        and math.isfinite(float(runtime_mass))
        and float(runtime_mass) >= 0.0
    )
    block(not mass_valid, "runtime_payload_mass_unavailable")
    block(_finite_vector(runtime_cog, 3) is None, "runtime_payload_cog_unavailable")
    parsed_runtime_inertia = _finite_vector(runtime_inertia, 3)
    block(
        parsed_runtime_inertia is None
        or any(value < 0.0 for value in parsed_runtime_inertia),
        "runtime_payload_inertia_unavailable",
    )
    if safety.reviewed_payload_mass_kg is not None and mass_valid:
        block(
            not math.isclose(
                float(runtime_mass),
                safety.reviewed_payload_mass_kg,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "runtime_payload_mass_mismatch",
        )
    if safety.reviewed_payload_cog_m is not None:
        block(
            not _same_numbers(runtime_cog, safety.reviewed_payload_cog_m, 3),
            "runtime_payload_cog_mismatch",
        )
    if safety.reviewed_payload_inertia_kg_m2 is not None:
        block(
            not _same_numbers(
                runtime_inertia,
                safety.reviewed_payload_inertia_kg_m2,
                3,
            ),
            "runtime_payload_inertia_mismatch",
        )

    runtime_limits_value = (
        metadata.get("joint_soft_limits_rad") if metadata is not None else None
    )
    runtime_limits = _soft_limits(runtime_limits_value)
    block(runtime_limits is None, "runtime_joint_soft_limits_unavailable")
    if safety.reviewed_joint_soft_limits_rad is not None:
        block(
            not _same_soft_limits(
                runtime_limits_value,
                safety.reviewed_joint_soft_limits_rad,
            ),
            "runtime_joint_soft_limits_mismatch",
        )
        anchor_joints = _finite_vector(anchor.robot_joint_positions, 6)
        if anchor_joints is not None:
            block(
                any(
                    joint < lower or joint > upper
                    for joint, (lower, upper) in zip(
                        anchor_joints,
                        safety.reviewed_joint_soft_limits_rad,
                    )
                ),
                "anchor_joint_position_outside_reviewed_soft_limits",
            )

    runtime_joints_value = (
        runtime_summary.get("joint_position_rad") if runtime_summary is not None else None
    )
    runtime_joints = _finite_vector(runtime_joints_value, 6)
    block(runtime_joints is None, "runtime_joint_positions_unavailable")
    if runtime_joints is not None and runtime_limits is not None:
        block(
            any(
                joint < lower or joint > upper
                for joint, (lower, upper) in zip(runtime_joints, runtime_limits)
            ),
            "runtime_joint_position_outside_soft_limits",
        )

    collision_state = (
        runtime_summary.get("collision_state") if runtime_summary is not None else None
    )
    collision_query_valid = (
        runtime_summary.get("collision_state_query_valid")
        if runtime_summary is not None
        else None
    )
    block(
        collision_query_valid is not True or type(collision_state) is not bool,
        "runtime_collision_state_unavailable",
    )
    block(collision_state is True, "runtime_collision_detected")

    current_pose = np.asarray(current_tcp_pose_base, dtype=float) if current_tcp_pose_base is not None else None
    if current_pose is None or current_pose.shape != (6,) or not np.isfinite(current_pose).all():
        block(True, "current_tcp_pose_unavailable")
    else:
        anchor_pose = np.asarray(anchor.tcp_pose_base, dtype=float)
        position_error = float(np.linalg.norm(current_pose[:3] - anchor_pose[:3]))
        orientation_delta = np.arctan2(
            np.sin(current_pose[3:] - anchor_pose[3:]),
            np.cos(current_pose[3:] - anchor_pose[3:]),
        )
        orientation_error = float(np.linalg.norm(orientation_delta))
        if safety.max_start_anchor_position_error_m is not None:
            block(
                position_error > safety.max_start_anchor_position_error_m,
                "robot_not_at_reviewed_start_anchor_position",
            )
        if safety.max_start_anchor_orientation_error_rad is not None:
            block(
                orientation_error > safety.max_start_anchor_orientation_error_rad,
                "robot_not_at_reviewed_start_anchor_orientation",
            )

    trajectory_id = audit.trajectory_id
    block(
        trajectory_id not in ALLOWED_FIRST_ROBOT_TRIAL_TRAJECTORIES,
        "trajectory_not_whitelisted_for_first_robot_trial",
    )
    block(anchor.trajectory_id != trajectory_id, "anchor_trajectory_id_mismatch")
    if "q_hip_ref" in trajectory and "q_knee_ref" in trajectory and len(trajectory):
        try:
            first_q_hip = float(trajectory.iloc[0]["q_hip_ref"])
            first_q_knee = float(trajectory.iloc[0]["q_knee_ref"])
        except (TypeError, ValueError):
            first_q_hip = math.nan
            first_q_knee = math.nan
        block(
            not math.isclose(
                anchor.reference_start_q_hip,
                first_q_hip,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            or not math.isclose(
                anchor.reference_start_q_knee,
                first_q_knee,
                rel_tol=0.0,
                abs_tol=1e-10,
            ),
            "anchor_reference_start_joint_mismatch",
        )
    else:
        block(True, "anchor_reference_start_joint_mismatch")
    block(not audit.trajectory_valid, "trajectory_audit_invalid")
    block(not audit.first_target_equals_anchor, "trajectory_start_not_anchor")
    block(not audit.final_target_equals_anchor, "trajectory_end_not_anchor")
    block(not audit.first_relative_displacement_zero, "relative_start_not_zero")
    block(not audit.final_relative_displacement_zero, "trajectory_not_closed")
    block(not audit.approved_rom_valid, "approved_rom_invalid")
    block(not audit.theta_shank_definition_valid, "theta_shank_definition_invalid")
    block(not audit.all_samples_finite, "trajectory_contains_non_finite_values")
    block(not audit.time_strictly_increasing, "trajectory_time_invalid")
    block(audit.sample_count != len(trajectory), "trajectory_audit_sample_count_mismatch")
    content_reasons, max_speed, max_acceleration = audit_execution_trajectory_content(
        trajectory,
        safety,
        expected_trajectory_id=trajectory_id,
        frame=frame,
        anchor=anchor,
        require_official_reference=True,
    )
    for content_reason in content_reasons:
        block(True, content_reason)

    health = acquisition_health
    block(not health.valid, "acquisition_streams_unhealthy")
    block(not health.state_thread_alive, "state_thread_not_alive")
    block(not health.wrench_thread_alive, "wrench_thread_not_alive")
    block(not health.alignment_thread_alive, "alignment_thread_not_alive")
    if safety.max_state_age_s is not None:
        block(
            health.state_age_s is None or health.state_age_s > safety.max_state_age_s,
            "state_stale",
        )
    if safety.max_wrench_age_s is not None:
        block(
            health.wrench_age_s is None or health.wrench_age_s > safety.max_wrench_age_s,
            "wrench_stale",
        )
    if safety.max_state_wrench_skew_s is not None:
        block(
            health.state_wrench_skew_s is None
            or health.state_wrench_skew_s > safety.max_state_wrench_skew_s,
            "state_wrench_skew_limit_exceeded",
        )
    if safety.max_force_n is not None:
        block(
            health.force_magnitude_n is None or health.force_magnitude_n > safety.max_force_n,
            "force_limit_exceeded_or_unavailable",
        )
    if safety.max_torque_nm is not None:
        block(
            health.torque_magnitude_nm is None
            or health.torque_magnitude_nm > safety.max_torque_nm,
            "torque_limit_exceeded_or_unavailable",
        )
    block(not math.isfinite(max_speed) if max_speed is not None else False, "invalid_speed_audit")
    block(
        not math.isfinite(max_acceleration) if max_acceleration is not None else False,
        "invalid_acceleration_audit",
    )
    return ExecutionPreflight(
        allowed=not reasons,
        reasons=tuple(reasons),
        trajectory_id=trajectory_id,
        anchor_id=anchor.anchor_id,
        maximum_tcp_speed_m_s=max_speed,
        maximum_tcp_acceleration_m_s2=max_acceleration,
        runtime_robot_summary=(
            dict(runtime_summary) if runtime_summary is not None else None
        ),
        evaluation_phase=_evaluation_phase,
        trajectory_sha256=trajectory_execution_digest(trajectory),
        experiment_safety_sha256=experiment_safety_digest(safety),
        _live_token=(
            _LIVE_PREFLIGHT_TOKEN if _evaluation_phase == "live" else None
        ),
    )


def evaluate_offline_execution_request(
    *,
    mode: str,
    enable_motion: bool,
    operator_confirmation: str,
    requested_anchor_id: str,
    frame: RehabFrameConfig,
    anchor: StartAnchor,
    safety: ExperimentSafetyConfig,
    trajectory: pd.DataFrame,
    audit: RelativeTrajectoryAudit,
) -> ExecutionPreflight:
    """Run all static gates before a robot connection is attempted.

    Runtime state/wrench/logger/connection values are intentionally neutral in
    this pass and are evaluated again from real caches immediately before
    motion.  The returned object must never be passed to the executor.
    """

    class _ReadyLogger:
        ready = True
        healthy = True

    class _Connected:
        @staticmethod
        def is_connected() -> bool:
            return True

        @staticmethod
        def get_robot_state_summary() -> dict[str, object]:
            return {
                "connected": True,
                "joint_position_rad": list(anchor.robot_joint_positions),
                "collision_state": False,
                "collision_state_query_valid": True,
                "robot_metadata": {
                    "robot_model": anchor.robot_model,
                    "robot_serial_number": anchor.robot_serial_number,
                    "controller_version": anchor.controller_version,
                    "joint_soft_limits_rad": safety.reviewed_joint_soft_limits_rad,
                    "sdk_tool_payload": {
                        "toolset_load_mass_kg": safety.reviewed_payload_mass_kg,
                        "toolset_load_cog_m": safety.reviewed_payload_cog_m,
                        "toolset_load_inertia_kg_m2": (
                            safety.reviewed_payload_inertia_kg_m2
                        ),
                        "sdk_available_tool_names": [safety.reviewed_tool_name],
                        "sdk_available_workobject_names": [
                            safety.reviewed_workpiece_name
                        ],
                        "active_hmi_tool_workobject_verified": False,
                    },
                },
            }

    neutral_health = AcquisitionHealth(
        host_time_s=0.0,
        state_time_s=0.0,
        wrench_time_s=0.0,
        state_age_s=0.0,
        wrench_age_s=0.0,
        state_wrench_skew_s=0.0,
        state_valid=True,
        wrench_valid=True,
        state_thread_alive=True,
        wrench_thread_alive=True,
        alignment_thread_alive=True,
        query_duration_ms=0.0,
        force_magnitude_n=0.0,
        torque_magnitude_nm=0.0,
        valid=True,
        invalid_reason="",
    )
    return evaluate_execution_preflight(
        mode=mode,
        enable_motion=enable_motion,
        operator_confirmation=operator_confirmation,
        requested_anchor_id=requested_anchor_id,
        frame=frame,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
        audit=audit,
        acquisition_health=neutral_health,
        logger=_ReadyLogger(),
        robot_adapter=_Connected(),
        current_tcp_pose_base=anchor.tcp_pose_base,
        _evaluation_phase="offline",
    )


__all__ = [
    "EXECUTE_MODE",
    "ExecutionPreflight",
    "OPERATOR_CONFIRMATION",
    "audit_execution_trajectory_content",
    "evaluate_execution_preflight",
    "experiment_safety_digest",
    "evaluate_offline_execution_request",
    "trajectory_execution_digest",
]
