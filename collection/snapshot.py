"""Compose time-qualified kinematic and internal-wrench frames safely."""

from __future__ import annotations

import math
import time
from typing import Any

from collection.state import (
    InternalWrenchFrame,
    KinematicStateFrame,
    RobotStateSample,
    as_vec3,
    internal_skew_ms,
    merge_invalid_reasons,
    state_age_ms,
    utc_now_iso,
)
from config import settings


def _acceleration_estimate(
    current: KinematicStateFrame,
    previous: RobotStateSample | None,
) -> tuple[tuple[float, float, float] | None, str]:
    """Estimate linear acceleration from stored, raw numerical velocity only."""
    current_velocity = current.tcp_linear_velocity_mps
    if current_velocity is None:
        return None, "unavailable_no_velocity"
    if previous is None or previous.tcp_linear_velocity_mps is None:
        return None, "unavailable_no_previous_velocity"
    if current.velocity_time_s is None or previous.velocity_time_s is None:
        return None, "unavailable_missing_velocity_time"
    dt_s = current.velocity_time_s - previous.velocity_time_s
    if not math.isfinite(dt_s) or dt_s <= 0.0 or dt_s > 1.0:
        return None, "unavailable_invalid_velocity_dt"
    values = tuple(
        (current_velocity[index] - previous.tcp_linear_velocity_mps[index]) / dt_s
        for index in range(3)
    )
    result = as_vec3(values)
    return (
        result,
        "numerical_difference" if result is not None else "unavailable_nonfinite_difference",
    )


def compose_robot_state_sample(
    kinematic: KinematicStateFrame,
    wrench: InternalWrenchFrame,
    *,
    sample_time_s: float | None = None,
    wall_time_iso: str | None = None,
    previous_sample: RobotStateSample | None = None,
    max_robot_state_age_s: float = settings.MAX_ROBOT_STATE_AGE_S,
    max_force_sample_age_s: float = settings.MAX_FORCE_SAMPLE_AGE_S,
    max_internal_skew_s: float = settings.MAX_INTERNAL_STATE_SKEW_S,
    require_verified_base_rotation: bool | None = None,
) -> RobotStateSample:
    """Merge two same-robot sources without asserting they are synchronous.

    ``pose`` and ``joint`` originate from one xCore realtime reception packet;
    all wrench values originate from one separate ``getEndTorque`` query.  The
    resulting sample carries both host times and is invalidated when age/skew
    exceeds policy.  Missing values remain ``None`` for CSV blank output.
    """
    now_s = time.monotonic() if sample_time_s is None else float(sample_time_s)
    if not math.isfinite(now_s):
        raise ValueError("sample_time_s must be finite")
    state_age = state_age_ms(now_s, kinematic.host_monotonic_time_s)
    force_age = state_age_ms(now_s, wrench.host_monotonic_time_s)
    skew = internal_skew_ms(
        (
            kinematic.pose_time_s,
            kinematic.joint_time_s,
            wrench.force_time_s,
            wrench.torque_time_s,
        )
    )
    if require_verified_base_rotation is None:
        require_verified_base_rotation = not settings.BASE_WRENCH_ROTATION_VERIFIED
    reasons: list[str | None] = [
        kinematic.invalid_reason if not kinematic.valid else None,
        wrench.invalid_reason if not wrench.valid else None,
    ]
    if state_age is None:
        reasons.append(
            "robot_state_time_in_future"
            if kinematic.host_monotonic_time_s is not None
            and kinematic.host_monotonic_time_s > now_s
            else "robot_state_stale"
        )
    elif state_age > max_robot_state_age_s * 1000.0:
        reasons.append("robot_state_stale")
    if force_age is None:
        reasons.append(
            "robot_wrench_time_in_future"
            if wrench.host_monotonic_time_s is not None
            and wrench.host_monotonic_time_s > now_s
            else "robot_wrench_stale"
        )
    elif force_age > max_force_sample_age_s * 1000.0:
        reasons.append("robot_wrench_stale")
    if skew is None or skew > max_internal_skew_s * 1000.0:
        reasons.append("robot_state_internal_skew")
    if wrench.cartesian_force_corrected_n is None or wrench.cartesian_torque_corrected_nm is None:
        reasons.append("software_force_bias_not_ready")
    if wrench.cartesian_force_base_n is None or wrench.cartesian_torque_base_nm is None:
        reasons.append("base_wrench_unavailable")
    if (
        require_verified_base_rotation
        and wrench.base_transform_kind == "rotation_only_pending_robot_validation"
    ):
        reasons.append("base_wrench_rotation_requires_robot_validation")
    if kinematic.collision_state is True:
        reasons.append("robot_collision")
    if kinematic.controller_error:
        reasons.append("controller_error")
    if kinematic.operation_state in {"DISCONNECTED", "UNKNOWN"}:
        reasons.append(f"robot_operation_state:{kinematic.operation_state}")

    invalid_reason = merge_invalid_reasons(*reasons)
    acceleration, acceleration_source = _acceleration_estimate(kinematic, previous_sample)
    force_valid = bool(wrench.valid and wrench.cartesian_force_corrected_n is not None)
    query_duration_ms: float | None = None
    if (
        wrench.force_query_started_s is not None
        and wrench.force_query_finished_s is not None
        and math.isfinite(wrench.force_query_started_s)
        and math.isfinite(wrench.force_query_finished_s)
        and wrench.force_query_finished_s >= wrench.force_query_started_s
    ):
        query_duration_ms = (
            wrench.force_query_finished_s - wrench.force_query_started_s
        ) * 1000.0
    return RobotStateSample(
        sequence_id=kinematic.sequence_id,
        host_monotonic_time_s=now_s,
        wall_time_iso=wall_time_iso or utc_now_iso(),
        robot_device_time_s=kinematic.robot_device_time_s,
        valid=not bool(invalid_reason),
        invalid_reason=invalid_reason,
        sample_time_s=now_s,
        robot_state_time_s=kinematic.host_monotonic_time_s,
        pose_time_s=kinematic.pose_time_s,
        joint_time_s=kinematic.joint_time_s,
        velocity_time_s=kinematic.velocity_time_s,
        torque_time_s=wrench.torque_time_s,
        force_time_s=wrench.force_time_s,
        robot_state_age_ms=state_age,
        force_sample_age_ms=force_age,
        state_internal_skew_ms=skew,
        tcp_position_m=kinematic.tcp_position_m,
        tcp_orientation_rad=kinematic.tcp_orientation_rad,
        tcp_linear_velocity_mps=kinematic.tcp_linear_velocity_mps,
        tcp_angular_velocity_radps=kinematic.tcp_angular_velocity_radps,
        tcp_linear_acceleration_est_mps2=acceleration,
        velocity_source=kinematic.velocity_source,
        acceleration_source=acceleration_source,
        joint_position_rad=kinematic.joint_position_rad,
        joint_velocity_radps=kinematic.joint_velocity_radps,
        joint_measured_torque_nm=wrench.joint_measured_torque_nm,
        joint_external_torque_nm=wrench.joint_external_torque_nm,
        cartesian_force_raw_n=wrench.cartesian_force_raw_n,
        cartesian_torque_raw_nm=wrench.cartesian_torque_raw_nm,
        raw_force_frame=wrench.raw_force_frame,
        cartesian_force_bias_n=wrench.cartesian_force_bias_n,
        cartesian_torque_bias_nm=wrench.cartesian_torque_bias_nm,
        cartesian_force_corrected_n=wrench.cartesian_force_corrected_n,
        cartesian_torque_corrected_nm=wrench.cartesian_torque_corrected_nm,
        cartesian_force_base_n=wrench.cartesian_force_base_n,
        cartesian_torque_base_nm=wrench.cartesian_torque_base_nm,
        base_transform_kind=wrench.base_transform_kind,
        operation_state=kinematic.operation_state,
        collision_state=kinematic.collision_state,
        controller_error=kinematic.controller_error,
        force_estimate_valid=force_valid,
        force_query_started_s=wrench.force_query_started_s,
        force_query_finished_s=wrench.force_query_finished_s,
        force_query_duration_ms=query_duration_ms,
    )


def read_live_robot_state_sample(
    robot: Any,
    wrench_source: Any,
    *,
    previous_sample: RobotStateSample | None = None,
    sample_time_s: float | None = None,
) -> RobotStateSample:
    """Read only cached snapshots; never perform sequential state API queries."""
    if not hasattr(robot, "get_state_frame"):
        raise TypeError("Robot must expose get_state_frame() for synchronized collection")
    if not hasattr(wrench_source, "snapshot"):
        raise TypeError("Wrench source must expose snapshot() for synchronized collection")
    # Take cached frames before assigning a sample timestamp.  Capturing ``now``
    # first can make a concurrently published state appear to come from the
    # future and was previously hidden by age clamping.
    kinematic = robot.get_state_frame()
    snapshot_time_s = time.monotonic() if sample_time_s is None else float(sample_time_s)
    wrench = wrench_source.snapshot(snapshot_time_s)
    # Mark the sample only after both cached frames have been obtained.  This
    # avoids treating a producer that published during the read as a
    # future-dated zero-age frame.
    now_s = time.monotonic() if sample_time_s is None else snapshot_time_s
    return compose_robot_state_sample(
        kinematic,
        wrench,
        sample_time_s=now_s,
        previous_sample=previous_sample,
    )
