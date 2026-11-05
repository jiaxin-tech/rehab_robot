"""Strict millimetre-scale trajectory and feedback checks for RT commissioning.

This module is intentionally separate from the approved rehabilitation
trajectory path.  It accepts only a closed translation along base +Z and is
not a general-purpose Cartesian jog interface.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import pandas as pd

from collection.real_robot_acquisition import AcquisitionHealth
from control.execution_preflight import ExecutionPreflight, OPERATOR_CONFIRMATION
from control.start_anchor import StartAnchor
from hardware.safety_events import (
    collision_feedback_allows_motion,
    safety_event_block_reason,
)
from safety.experiment_safety import ExperimentSafetyConfig


REALTIME_LINK_TRAJECTORY_ID = "realtime_link_smoke_v1"
REALTIME_LINK_PROFILE = "realtime_link_diagnostic"
MIN_STEP_M = 0.0005
MAX_STEP_M = 0.002
SAMPLE_PERIOD_S = 0.25
SETTLE_END_S = 3.0
OUTBOUND_END_S = 5.0
FAR_CHECK_S = 6.0
RETURN_START_S = 7.0
RETURN_END_S = 9.0
RETURN_CHECK_S = 10.0
TRAJECTORY_END_S = 11.0
MAX_DIAGNOSTIC_SPEED_M_S = 0.002
MAX_DIAGNOSTIC_ACCELERATION_M_S2 = 0.005


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def build_realtime_link_trajectory(
    start_pose: tuple[float, float, float, float, float, float],
    *,
    step_m: float = 0.001,
    q_hip_ref: float = 0.0,
    q_knee_ref: float = 0.0,
) -> pd.DataFrame:
    """Build the sole accepted diagnostic path: +Z, hold, return, hold."""

    pose = np.asarray(start_pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("start_pose must contain six finite values")
    if isinstance(step_m, bool) or not isinstance(step_m, (int, float)):
        raise ValueError("step_m must be a finite number")
    step = float(step_m)
    if not math.isfinite(step) or not MIN_STEP_M <= step <= MAX_STEP_M:
        raise ValueError(
            f"step_m must be between {MIN_STEP_M:g} and {MAX_STEP_M:g} m"
        )

    times = np.arange(
        0.0,
        TRAJECTORY_END_S + SAMPLE_PERIOD_S / 2.0,
        SAMPLE_PERIOD_S,
        dtype=float,
    )
    delta_z = np.zeros_like(times)
    outbound = (times >= SETTLE_END_S) & (times <= OUTBOUND_END_S)
    outbound_u = (times[outbound] - SETTLE_END_S) / (
        OUTBOUND_END_S - SETTLE_END_S
    )
    delta_z[outbound] = step * _minimum_jerk(outbound_u)
    far_hold = (times > OUTBOUND_END_S) & (times < RETURN_START_S)
    delta_z[far_hold] = step
    returning = (times >= RETURN_START_S) & (times <= RETURN_END_S)
    return_u = (times[returning] - RETURN_START_S) / (
        RETURN_END_S - RETURN_START_S
    )
    delta_z[returning] = step * (1.0 - _minimum_jerk(return_u))

    xyz = np.repeat(pose[None, :3], len(times), axis=0)
    xyz[:, 2] += delta_z
    velocity = np.gradient(xyz, times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    phases = np.select(
        (
            times < SETTLE_END_S,
            times <= OUTBOUND_END_S,
            times < RETURN_START_S,
            times <= RETURN_END_S,
        ),
        ("settle_hold", "outbound", "far_hold", "return"),
        default="home_hold",
    )
    return pd.DataFrame(
        {
            "time_s": times,
            "trajectory_id": REALTIME_LINK_TRAJECTORY_ID,
            "trajectory_profile": REALTIME_LINK_PROFILE,
            "trajectory_phase": phases,
            "delta_x_R": np.zeros(len(times)),
            "delta_y_R": np.zeros(len(times)),
            "delta_z_R": delta_z,
            "tcp_x_base": xyz[:, 0],
            "tcp_y_base": xyz[:, 1],
            "tcp_z_base": xyz[:, 2],
            "tcp_rx": np.full(len(times), pose[3]),
            "tcp_ry": np.full(len(times), pose[4]),
            "tcp_rz": np.full(len(times), pose[5]),
            "tcp_vx_base": velocity[:, 0],
            "tcp_vy_base": velocity[:, 1],
            "tcp_vz_base": velocity[:, 2],
            "tcp_ax_base": acceleration[:, 0],
            "tcp_ay_base": acceleration[:, 1],
            "tcp_az_base": acceleration[:, 2],
            "q_hip_ref": np.full(len(times), float(q_hip_ref)),
            "q_knee_ref": np.full(len(times), float(q_knee_ref)),
            "trajectory_valid": np.ones(len(times), dtype=bool),
        }
    )


def audit_realtime_link_trajectory(
    trajectory: pd.DataFrame,
    safety: ExperimentSafetyConfig,
    *,
    expected_trajectory_id: str | None = None,
) -> tuple[tuple[str, ...], float | None, float | None]:
    """Fail closed unless *trajectory* is the constrained +Z smoke test."""

    reasons: list[str] = []

    def block(condition: bool, reason: str) -> None:
        if condition and reason not in reasons:
            reasons.append(reason)

    required = {
        "time_s", "trajectory_id", "trajectory_profile", "trajectory_phase",
        "delta_x_R", "delta_y_R", "delta_z_R", "tcp_x_base", "tcp_y_base",
        "tcp_z_base", "tcp_rx", "tcp_ry", "tcp_rz", "tcp_vx_base",
        "tcp_vy_base", "tcp_vz_base", "tcp_ax_base", "tcp_ay_base",
        "tcp_az_base", "q_hip_ref", "q_knee_ref", "trajectory_valid",
    }
    missing = required.difference(trajectory.columns)
    block(bool(missing), "diagnostic_trajectory_columns_missing")
    block(len(trajectory) < 30, "diagnostic_trajectory_has_too_few_samples")
    if missing or len(trajectory) < 3:
        return tuple(reasons), None, None

    ids = trajectory["trajectory_id"].tolist()
    unique_ids = set(ids) if all(isinstance(item, str) for item in ids) else set()
    table_id = next(iter(unique_ids)) if len(unique_ids) == 1 else None
    block(table_id != REALTIME_LINK_TRAJECTORY_ID, "diagnostic_trajectory_id_invalid")
    if expected_trajectory_id is not None:
        block(table_id != expected_trajectory_id, "trajectory_id_binding_mismatch")
    block(
        not all(value == REALTIME_LINK_PROFILE for value in trajectory["trajectory_profile"]),
        "diagnostic_trajectory_profile_invalid",
    )
    valid = trajectory["trajectory_valid"].tolist()
    strict_bool = all(type(value) is bool or isinstance(value, np.bool_) for value in valid)
    block(not strict_bool or not all(bool(value) for value in valid), "diagnostic_rows_invalid")

    numeric_columns = [
        "time_s", "delta_x_R", "delta_y_R", "delta_z_R", "tcp_x_base",
        "tcp_y_base", "tcp_z_base", "tcp_rx", "tcp_ry", "tcp_rz",
        "tcp_vx_base", "tcp_vy_base", "tcp_vz_base", "tcp_ax_base",
        "tcp_ay_base", "tcp_az_base", "q_hip_ref", "q_knee_ref",
    ]
    try:
        numeric = trajectory[numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError):
        block(True, "diagnostic_trajectory_values_not_numeric")
        return tuple(reasons), None, None
    block(not bool(np.isfinite(numeric).all()), "diagnostic_trajectory_non_finite")
    if not np.isfinite(numeric).all():
        return tuple(reasons), None, None

    times = trajectory["time_s"].to_numpy(float)
    time_deltas = np.diff(times)
    block(
        not math.isclose(float(times[0]), 0.0, abs_tol=1e-12)
        or not math.isclose(float(times[-1]), TRAJECTORY_END_S, abs_tol=1e-12)
        or not bool((time_deltas > 0.0).all())
        or float(time_deltas.max()) > SAMPLE_PERIOD_S + 1e-12,
        "diagnostic_trajectory_time_invalid",
    )
    xyz = trajectory[["tcp_x_base", "tcp_y_base", "tcp_z_base"]].to_numpy(float)
    relative = xyz - xyz[0]
    delta = trajectory[["delta_x_R", "delta_y_R", "delta_z_R"]].to_numpy(float)
    tolerance = 1e-10
    block(
        not bool(np.allclose(delta, relative, rtol=0.0, atol=tolerance)),
        "diagnostic_delta_columns_mismatch",
    )
    block(
        not bool(np.allclose(relative[:, :2], 0.0, rtol=0.0, atol=tolerance)),
        "diagnostic_lateral_motion_forbidden",
    )
    step = float(relative[:, 2].max())
    block(
        step < MIN_STEP_M - tolerance or step > MAX_STEP_M + tolerance,
        "diagnostic_step_out_of_range",
    )
    block(
        float(relative[:, 2].min()) < -tolerance,
        "diagnostic_negative_z_motion_forbidden",
    )
    block(
        not bool(np.allclose(relative[0], 0.0, atol=tolerance, rtol=0.0))
        or not bool(np.allclose(relative[-1], 0.0, atol=tolerance, rtol=0.0)),
        "diagnostic_trajectory_not_closed",
    )
    outbound = relative[
        (times >= SETTLE_END_S) & (times <= OUTBOUND_END_S), 2
    ]
    returning = relative[(times >= RETURN_START_S) & (times <= RETURN_END_S), 2]
    block(bool((np.diff(outbound) < -tolerance).any()), "diagnostic_outbound_not_monotonic")
    block(bool((np.diff(returning) > tolerance).any()), "diagnostic_return_not_monotonic")
    block(
        not bool(
            np.allclose(
                relative[times < SETTLE_END_S, 2],
                0.0,
                atol=tolerance,
                rtol=0.0,
            )
        ),
        "diagnostic_settle_hold_invalid",
    )
    block(
        not bool(np.allclose(relative[(times > OUTBOUND_END_S) & (times < RETURN_START_S), 2], step, atol=tolerance, rtol=0.0)),
        "diagnostic_far_hold_invalid",
    )
    block(
        not bool(np.allclose(relative[times > RETURN_END_S, 2], 0.0, atol=tolerance, rtol=0.0)),
        "diagnostic_home_hold_invalid",
    )
    orientation = trajectory[["tcp_rx", "tcp_ry", "tcp_rz"]].to_numpy(float)
    block(
        not bool(np.allclose(orientation, orientation[0], atol=tolerance, rtol=0.0)),
        "diagnostic_orientation_change_forbidden",
    )

    derived_velocity = np.gradient(xyz, times, axis=0, edge_order=2)
    derived_acceleration = np.gradient(derived_velocity, times, axis=0, edge_order=2)
    declared_velocity = trajectory[["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]].to_numpy(float)
    declared_acceleration = trajectory[["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]].to_numpy(float)
    block(
        not bool(np.allclose(declared_velocity, derived_velocity, atol=1e-10, rtol=0.0)),
        "diagnostic_velocity_columns_mismatch",
    )
    block(
        not bool(np.allclose(declared_acceleration, derived_acceleration, atol=1e-9, rtol=0.0)),
        "diagnostic_acceleration_columns_mismatch",
    )
    maximum_speed = float(np.linalg.norm(derived_velocity, axis=1).max())
    maximum_acceleration = float(np.linalg.norm(derived_acceleration, axis=1).max())
    speed_limit = min(
        MAX_DIAGNOSTIC_SPEED_M_S,
        safety.max_tcp_speed_m_s if safety.max_tcp_speed_m_s is not None else math.inf,
    )
    acceleration_limit = min(
        MAX_DIAGNOSTIC_ACCELERATION_M_S2,
        safety.max_tcp_acceleration_m_s2
        if safety.max_tcp_acceleration_m_s2 is not None else math.inf,
    )
    block(maximum_speed > speed_limit + 1e-12, "diagnostic_speed_limit_exceeded")
    block(
        maximum_acceleration > acceleration_limit + 1e-12,
        "diagnostic_acceleration_limit_exceeded",
    )
    if safety.workspace_min_base_m is None or safety.workspace_max_base_m is None:
        block(True, "diagnostic_workspace_unavailable")
    else:
        lower = np.asarray(safety.workspace_min_base_m, dtype=float)
        upper = np.asarray(safety.workspace_max_base_m, dtype=float)
        block(
            not bool(((xyz >= lower) & (xyz <= upper)).all()),
            "diagnostic_trajectory_outside_workspace",
        )
    return tuple(reasons), maximum_speed, maximum_acceleration


def evaluate_realtime_link_offline(
    *,
    enable_motion: bool,
    confirm_unloaded: bool,
    operator_confirmation: str,
    requested_anchor_id: str,
    anchor: StartAnchor,
    safety: ExperimentSafetyConfig,
    trajectory: pd.DataFrame,
) -> ExecutionPreflight:
    reasons = list(safety.execution_block_reasons())
    if enable_motion is not True:
        reasons.append("--live is required")
    if confirm_unloaded is not True:
        reasons.append("--confirm-unloaded is required")
    if operator_confirmation != OPERATOR_CONFIRMATION:
        reasons.append("exact operator confirmation is required")
    if requested_anchor_id != anchor.anchor_id:
        reasons.append("start anchor id mismatch")
    try:
        first_pose = trajectory.iloc[0][
            [
                "tcp_x_base", "tcp_y_base", "tcp_z_base",
                "tcp_rx", "tcp_ry", "tcp_rz",
            ]
        ].to_numpy(dtype=float)
    except (IndexError, KeyError, TypeError, ValueError):
        first_pose = np.asarray([], dtype=float)
    if (
        first_pose.shape != (6,)
        or not np.isfinite(first_pose).all()
        or not np.allclose(
            first_pose,
            np.asarray(anchor.tcp_pose_base, dtype=float),
            rtol=0.0,
            atol=1e-10,
        )
    ):
        reasons.append("diagnostic trajectory is not bound to the prepared anchor")
    content, speed, acceleration = audit_realtime_link_trajectory(
        trajectory,
        safety,
        expected_trajectory_id=REALTIME_LINK_TRAJECTORY_ID,
    )
    reasons.extend(content)
    return ExecutionPreflight(
        not reasons,
        tuple(reasons),
        REALTIME_LINK_TRAJECTORY_ID,
        anchor.anchor_id,
        speed,
        acceleration,
    )


def evaluate_realtime_link_live(
    *,
    offline: ExecutionPreflight,
    acquisition_health: AcquisitionHealth,
    logger: Any,
    robot_adapter: Any,
    current_tcp_pose_base: Any,
    anchor: StartAnchor,
    safety: ExperimentSafetyConfig,
) -> ExecutionPreflight:
    reasons = list(offline.reasons)
    connected = getattr(robot_adapter, "is_connected", False)
    connected = connected() if callable(connected) else connected
    if not bool(connected) or not logger.ready or not logger.healthy:
        reasons.append("robot or logger not ready")
    summary = robot_adapter.get_robot_state_summary()
    if not collision_feedback_allows_motion(summary):
        reasons.append("collision state is unavailable or active")
    event_reason = safety_event_block_reason(robot_adapter)
    if event_reason:
        reasons.append(event_reason)
    pose = np.asarray(current_tcp_pose_base, dtype=float)
    expected = np.asarray(anchor.tcp_pose_base, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        reasons.append("invalid current TCP pose")
    else:
        angular_delta = np.arctan2(
            np.sin(pose[3:] - expected[3:]),
            np.cos(pose[3:] - expected[3:]),
        )
        position_limit = min(
            safety.max_start_anchor_position_error_m
            if safety.max_start_anchor_position_error_m is not None
            else math.inf,
            0.00025,
        )
        orientation_limit = min(
            safety.max_start_anchor_orientation_error_rad
            if safety.max_start_anchor_orientation_error_rad is not None
            else math.inf,
            0.01,
        )
        if (
            np.linalg.norm(pose[:3] - expected[:3]) > position_limit
            or np.linalg.norm(angular_delta) > orientation_limit
        ):
            reasons.append("robot moved away from prepared diagnostic start")
    health = acquisition_health
    if not health.valid or not all(
        (
            health.state_thread_alive,
            health.wrench_thread_alive,
            health.alignment_thread_alive,
        )
    ):
        reasons.append("acquisition unhealthy")
    for value, limit in (
        (health.state_age_s, safety.max_state_age_s),
        (health.wrench_age_s, safety.max_wrench_age_s),
        (health.state_wrench_skew_s, safety.max_state_wrench_skew_s),
        (health.force_magnitude_n, safety.max_force_n),
        (health.torque_magnitude_nm, safety.max_torque_nm),
    ):
        if (
            value is None
            or not math.isfinite(value)
            or value < 0.0
            or limit is None
            or value > limit
        ):
            reasons.append("measurement missing, stale or over limit")
    return ExecutionPreflight(
        not reasons,
        tuple(reasons),
        offline.trajectory_id,
        offline.anchor_id,
        offline.maximum_tcp_speed_m_s,
        offline.maximum_tcp_acceleration_m_s2,
        summary,
    )


class RealtimeLinkTrackingGuard:
    """Require motion relative to a post-mode-switch settled measurement."""

    def __init__(self, adapter: Any, start_pose: Any, step_m: float) -> None:
        self._adapter = adapter
        self._absolute_start = np.asarray(start_pose, dtype=float)
        self._tracking_start: np.ndarray | None = None
        self._step_m = float(step_m)
        self._lock = threading.Lock()
        self._latest_trajectory_time_s = 0.0
        self._max_displacement_m = 0.0
        self._max_absolute_displacement_m = 0.0
        self._max_positive_z_m = 0.0
        self._max_lateral_m = 0.0
        self._last_displacement_m = 0.0
        self._settle_displacement_m: float | None = None
        self._far_verified = False
        self._return_verified = False
        self._samples = 0

    def observe_dispatch(self, trajectory_time_s: float, *_: Any) -> None:
        with self._lock:
            self._latest_trajectory_time_s = max(
                self._latest_trajectory_time_s,
                float(trajectory_time_s),
            )

    def check(self) -> str | None:
        try:
            pose = np.asarray(self._adapter.read_tcp_pose(), dtype=float)
        except Exception as exc:
            return f"diagnostic_tracking_pose_unavailable:{exc}"
        if pose.shape != (6,) or not np.isfinite(pose).all():
            return "diagnostic_tracking_pose_invalid"
        absolute_delta = pose[:3] - self._absolute_start[:3]
        absolute_displacement = float(np.linalg.norm(absolute_delta))
        absolute_orientation_delta = np.arctan2(
            np.sin(pose[3:] - self._absolute_start[3:]),
            np.cos(pose[3:] - self._absolute_start[3:]),
        )
        absolute_orientation_error = float(np.linalg.norm(absolute_orientation_delta))
        with self._lock:
            self._samples += 1
            self._max_absolute_displacement_m = max(
                self._max_absolute_displacement_m,
                absolute_displacement,
            )
            trajectory_time = self._latest_trajectory_time_s
            if absolute_displacement > 0.005:
                return (
                    "diagnostic_absolute_displacement_limit_exceeded_m="
                    f"{absolute_displacement:.9f}"
                )
            if absolute_orientation_error > 0.02:
                return (
                    "diagnostic_absolute_orientation_limit_exceeded_rad="
                    f"{absolute_orientation_error:.9f}"
                )
            if trajectory_time < SETTLE_END_S:
                return None
            if self._tracking_start is None:
                self._tracking_start = pose.copy()
                self._settle_displacement_m = absolute_displacement
                return None

            delta = pose[:3] - self._tracking_start[:3]
            displacement = float(np.linalg.norm(delta))
            lateral = float(np.linalg.norm(delta[:2]))
            positive_z = float(delta[2])
            orientation_delta = np.arctan2(
                np.sin(pose[3:] - self._tracking_start[3:]),
                np.cos(pose[3:] - self._tracking_start[3:]),
            )
            orientation_error = float(np.linalg.norm(orientation_delta))
            self._last_displacement_m = displacement
            self._max_displacement_m = max(self._max_displacement_m, displacement)
            self._max_positive_z_m = max(self._max_positive_z_m, positive_z)
            self._max_lateral_m = max(self._max_lateral_m, lateral)
            if trajectory_time >= FAR_CHECK_S and not self._far_verified:
                minimum_observed = max(0.0002, 0.4 * self._step_m)
                maximum_far_error = max(0.00035, 0.5 * self._step_m)
                if self._max_positive_z_m < minimum_observed:
                    return (
                        "diagnostic_no_observed_motion:"
                        f"max_positive_z_m={self._max_positive_z_m:.9f}"
                    )
                if abs(positive_z - self._step_m) > maximum_far_error:
                    return (
                        "diagnostic_far_target_not_reached:"
                        f"positive_z_m={positive_z:.9f}"
                    )
                if lateral > 0.0003:
                    return f"diagnostic_lateral_tracking_error_m={lateral:.9f}"
                if orientation_error > 0.01:
                    return (
                        "diagnostic_orientation_tracking_error_rad="
                        f"{orientation_error:.9f}"
                    )
                self._far_verified = True
            if trajectory_time >= RETURN_CHECK_S:
                maximum_return_error = max(0.00035, 0.4 * self._step_m)
                if displacement > maximum_return_error:
                    return (
                        "diagnostic_return_not_observed:"
                        f"displacement_m={displacement:.9f}"
                    )
                self._return_verified = True
        return None

    def require_complete(self) -> None:
        reason = self.check()
        if reason:
            raise RuntimeError(reason)
        with self._lock:
            if not self._far_verified or not self._return_verified:
                raise RuntimeError("diagnostic tracking milestones were not verified")

    def as_metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "samples": self._samples,
                "latest_trajectory_time_s": self._latest_trajectory_time_s,
                "max_displacement_m": self._max_displacement_m,
                "max_absolute_displacement_m": self._max_absolute_displacement_m,
                "max_positive_z_m": self._max_positive_z_m,
                "max_lateral_m": self._max_lateral_m,
                "last_displacement_m": self._last_displacement_m,
                "settle_displacement_m": self._settle_displacement_m,
                "settled_baseline_captured": self._tracking_start is not None,
                "far_verified": self._far_verified,
                "return_verified": self._return_verified,
                "step_m": self._step_m,
            }


__all__ = [
    "MAX_STEP_M",
    "MIN_STEP_M",
    "OPERATOR_CONFIRMATION",
    "REALTIME_LINK_PROFILE",
    "REALTIME_LINK_TRAJECTORY_ID",
    "RealtimeLinkTrackingGuard",
    "audit_realtime_link_trajectory",
    "build_realtime_link_trajectory",
    "evaluate_realtime_link_live",
    "evaluate_realtime_link_offline",
]
