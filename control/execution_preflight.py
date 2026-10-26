"""Single fail-closed gate for the first supervised real robot trajectory."""

from __future__ import annotations

from dataclasses import dataclass
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
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    RehabFrameConfig,
    RelativeTrajectoryAudit,
)
from lower_limb_sim.formal_protocol import (
    FORMAL_HIP_ROM_DEG as APPROVED_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG as APPROVED_KNEE_ROM_DEG,
)
from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)
from safety.experiment_safety import ExperimentSafetyConfig


OPERATOR_CONFIRMATION = "I CONFIRM SUPERVISED SLOW ROBOT MOTION"


@dataclass(frozen=True)
class ExecutionPreflight:
    allowed: bool
    reasons: tuple
    trajectory_id: str
    anchor_id: str
    maximum_tcp_speed_m_s: float | None
    maximum_tcp_acceleration_m_s2: float | None
    runtime_robot_summary: dict | None = None

    def require_allowed(self):
        if not self.allowed:
            raise PermissionError("execution blocked: " + "; ".join(self.reasons))

    def as_metadata(self):
        from dataclasses import asdict
        return asdict(self)


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
    numerical_limit_tolerance = 1e-12
    if safety.max_tcp_speed_m_s is not None:
        block(
            max_speed > safety.max_tcp_speed_m_s + numerical_limit_tolerance,
            "trajectory_speed_limit_exceeded",
        )
    if safety.max_tcp_acceleration_m_s2 is not None:
        block(
            max_acceleration
            > safety.max_tcp_acceleration_m_s2 + numerical_limit_tolerance,
            "trajectory_acceleration_limit_exceeded",
        )
    if safety.workspace_min_base_m is not None and safety.workspace_max_base_m is not None:
        lower = np.asarray(safety.workspace_min_base_m, dtype=float)
        upper = np.asarray(safety.workspace_max_base_m, dtype=float)
        block(
            not bool(((xyz >= lower) & (xyz <= upper)).all()),
            "trajectory_outside_reviewed_workspace",
        )

    return tuple(reasons), max_speed, max_acceleration


def evaluate_offline_execution_request(*, mode, enable_motion, operator_confirmation,
        requested_anchor_id, frame, anchor, safety, trajectory, audit):
    reasons = list(safety.execution_block_reasons())
    if mode != "execute" or enable_motion is not True or operator_confirmation != OPERATOR_CONFIRMATION:
        reasons.append("explicit motion confirmation required")
    if requested_anchor_id != anchor.anchor_id or anchor.trajectory_id != audit.trajectory_id:
        reasons.append("start anchor does not match this trajectory")
    content, speed, acceleration = audit_execution_trajectory_content(
        trajectory, safety, expected_trajectory_id=audit.trajectory_id, frame=frame, anchor=anchor,
        require_official_reference=False)
    reasons.extend(content)
    if not audit.trajectory_valid:
        reasons.append("invalid trajectory")
    return ExecutionPreflight(not reasons, tuple(reasons), audit.trajectory_id, anchor.anchor_id, speed, acceleration)


def evaluate_execution_preflight(*, acquisition_health, logger, robot_adapter, current_tcp_pose_base, **request):
    from hardware.safety_events import safety_event_block_reason, collision_feedback_allows_motion
    result = evaluate_offline_execution_request(**request)
    reasons = list(result.reasons)
    safety, anchor = request["safety"], request["anchor"]
    if not _connected(robot_adapter) or not logger.ready or not logger.healthy:
        reasons.append("robot or logger not ready")
    summary = robot_adapter.get_robot_state_summary()
    if not collision_feedback_allows_motion(summary):
        reasons.append("collision state is unavailable or active")
    event_reason = safety_event_block_reason(robot_adapter)
    if event_reason:
        reasons.append(event_reason)
    pose = np.asarray(current_tcp_pose_base, float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        reasons.append("invalid current TCP pose")
    else:
        expected = np.asarray(anchor.tcp_pose_base, float)
        delta = np.arctan2(np.sin(pose[3:] - expected[3:]), np.cos(pose[3:] - expected[3:]))
        if np.linalg.norm(pose[:3] - expected[:3]) > safety.max_start_anchor_position_error_m or np.linalg.norm(delta) > safety.max_start_anchor_orientation_error_rad:
            reasons.append("robot moved away from previewed start")
    health = acquisition_health
    if not health.valid or not all((health.state_thread_alive, health.wrench_thread_alive, health.alignment_thread_alive)):
        reasons.append("acquisition unhealthy")
    for value, limit in ((health.state_age_s, safety.max_state_age_s),
                         (health.wrench_age_s, safety.max_wrench_age_s),
                         (health.state_wrench_skew_s, safety.max_state_wrench_skew_s),
                         (health.force_magnitude_n, safety.max_force_n),
                         (health.torque_magnitude_nm, safety.max_torque_nm)):
        if value is None or not math.isfinite(value) or value < 0 or limit is None or value > limit:
            reasons.append("measurement missing, stale or over limit")
    return ExecutionPreflight(not reasons, tuple(reasons), result.trajectory_id, result.anchor_id,
                              result.maximum_tcp_speed_m_s, result.maximum_tcp_acceleration_m_s2, summary)
