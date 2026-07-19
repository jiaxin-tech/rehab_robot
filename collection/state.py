"""Typed, time-qualified robot state primitives used by collection and control.

The xCoreSDK 0.7.0 realtime stream does not expose a controller timestamp,
joint velocity, or wrench.  These types make that limitation explicit instead
of silently filling unavailable measurements with zeroes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Sequence


Vec3 = tuple[float, float, float]
Vec6 = tuple[float, float, float, float, float, float]


def utc_now_iso() -> str:
    """Return an unambiguous wall-clock timestamp for metadata/CSV output."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def finite_vector(values: Sequence[float] | None, size: int) -> bool:
    return (
        values is not None
        and len(values) == size
        and all(math.isfinite(float(value)) for value in values)
    )


def as_vec3(values: Sequence[float] | None) -> Vec3 | None:
    if not finite_vector(values, 3):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def as_vec6(values: Sequence[float] | None) -> Vec6 | None:
    if not finite_vector(values, 6):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def as_float_tuple(values: Sequence[float] | None, size: int = 6) -> tuple[float, ...] | None:
    if not finite_vector(values, size):
        return None
    return tuple(float(value) for value in values)


def merge_invalid_reasons(*reasons: str | None) -> str:
    """Join unique diagnostics while retaining the first-occurrence order."""
    result: list[str] = []
    for reason in reasons:
        if not reason:
            continue
        for item in str(reason).split(";"):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return ";".join(result)


@dataclass(frozen=True)
class KinematicStateFrame:
    """Latest frame from ``startReceiveRobotState``.

    ``host_monotonic_time_s`` is the host receipt time.  xCoreSDK v0.7.0 does
    not expose a device/controller timestamp in ``RtSupportedFields``; the
    corresponding field is consequently ``None`` rather than a fabricated
    value.
    """

    sequence_id: int
    host_monotonic_time_s: float | None
    wall_time_iso: str | None
    robot_device_time_s: float | None
    valid: bool
    invalid_reason: str

    tcp_position_m: Vec3 | None
    tcp_orientation_rad: Vec3 | None
    tcp_linear_velocity_mps: Vec3 | None
    tcp_angular_velocity_radps: Vec3 | None
    velocity_source: str

    joint_position_rad: tuple[float, ...] | None
    joint_velocity_radps: tuple[float, ...] | None
    pose_time_s: float | None
    joint_time_s: float | None
    velocity_time_s: float | None

    operation_state: str
    collision_state: bool | None
    controller_error: str | None
    keypad_state: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class InternalWrenchFrame:
    """One ``forceControl().getEndTorque`` result from the robot controller."""

    sequence_id: int
    host_monotonic_time_s: float | None
    wall_time_iso: str | None
    valid: bool
    invalid_reason: str
    source: str

    joint_measured_torque_nm: tuple[float, ...] | None
    joint_external_torque_nm: tuple[float, ...] | None

    cartesian_force_raw_n: Vec3 | None
    cartesian_torque_raw_nm: Vec3 | None
    raw_force_frame: str

    cartesian_force_bias_n: Vec3 | None
    cartesian_torque_bias_nm: Vec3 | None
    cartesian_force_corrected_n: Vec3 | None
    cartesian_torque_corrected_nm: Vec3 | None

    # These values are only coordinate-expression rotations.  They are not a
    # point-of-application wrench transform unless a verified lever arm was
    # supplied separately.
    cartesian_force_base_n: Vec3 | None
    cartesian_torque_base_nm: Vec3 | None
    base_transform_kind: str
    force_time_s: float | None
    torque_time_s: float | None
    # Host timing bounds of the single getEndTorque query.  ``force_time_s`` /
    # ``torque_time_s`` are its midpoint; these bounds preserve query latency.
    force_query_started_s: float | None = None
    force_query_finished_s: float | None = None


@dataclass(frozen=True)
class RobotStateSample:
    """Collection-ready state snapshot with explicit timing and provenance."""

    sequence_id: int
    host_monotonic_time_s: float
    wall_time_iso: str
    robot_device_time_s: float | None
    valid: bool
    invalid_reason: str

    sample_time_s: float
    robot_state_time_s: float | None
    pose_time_s: float | None
    joint_time_s: float | None
    velocity_time_s: float | None
    torque_time_s: float | None
    force_time_s: float | None
    robot_state_age_ms: float | None
    force_sample_age_ms: float | None
    state_internal_skew_ms: float | None

    tcp_position_m: Vec3 | None
    tcp_orientation_rad: Vec3 | None
    tcp_linear_velocity_mps: Vec3 | None
    tcp_angular_velocity_radps: Vec3 | None
    tcp_linear_acceleration_est_mps2: Vec3 | None
    velocity_source: str
    acceleration_source: str

    joint_position_rad: tuple[float, ...] | None
    joint_velocity_radps: tuple[float, ...] | None
    joint_measured_torque_nm: tuple[float, ...] | None
    joint_external_torque_nm: tuple[float, ...] | None

    cartesian_force_raw_n: Vec3 | None
    cartesian_torque_raw_nm: Vec3 | None
    raw_force_frame: str
    cartesian_force_bias_n: Vec3 | None
    cartesian_torque_bias_nm: Vec3 | None
    cartesian_force_corrected_n: Vec3 | None
    cartesian_torque_corrected_nm: Vec3 | None
    cartesian_force_base_n: Vec3 | None
    cartesian_torque_base_nm: Vec3 | None
    base_transform_kind: str

    operation_state: str
    collision_state: bool | None
    controller_error: str | None
    force_estimate_valid: bool

    trajectory_s: float | None = None
    trajectory_arc_length_m: float | None = None
    trajectory_tangent: Vec3 | None = None
    force_tangent_n: float | None = None
    velocity_tangent_mps: float | None = None
    acceleration_tangent_mps2: float | None = None
    force_query_started_s: float | None = None
    force_query_finished_s: float | None = None
    force_query_duration_ms: float | None = None


def state_age_ms(now_s: float, frame_time_s: float | None) -> float | None:
    if frame_time_s is None or not math.isfinite(frame_time_s):
        return None
    age_s = now_s - frame_time_s
    # A frame timestamp after the claimed sample time is a timing race or a
    # clock/provenance fault—not a fresh zero-age measurement.  Returning None
    # lets callers invalidate it rather than silently clamping the sign away.
    if age_s < 0.0:
        return None
    return age_s * 1000.0


def internal_skew_ms(times_s: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in times_s if value is not None and math.isfinite(value)]
    if not usable:
        return None
    return (max(usable) - min(usable)) * 1000.0


def dot3(left: Vec3 | None, right: Vec3 | None) -> float | None:
    if left is None or right is None:
        return None
    value = sum(a * b for a, b in zip(left, right))
    return float(value) if math.isfinite(value) else None


def calculate_software_bias(samples: Sequence[Sequence[float]]) -> Vec6:
    """Mean software reference offset; input must contain finite 6D wrenches."""
    if not samples:
        raise ValueError("At least one wrench sample is required for software bias")
    if not all(finite_vector(sample, 6) for sample in samples):
        raise ValueError("Software-bias samples must be finite six-axis wrenches")
    count = float(len(samples))
    return tuple(
        sum(float(sample[index]) for sample in samples) / count for index in range(6)
    )  # type: ignore[return-value]


def subtract_bias(values: Sequence[float] | None, bias: Sequence[float] | None) -> Vec6 | None:
    if not finite_vector(values, 6) or not finite_vector(bias, 6):
        return None
    corrected = tuple(float(values[i]) - float(bias[i]) for i in range(6))
    return corrected if all(math.isfinite(value) for value in corrected) else None


def rpy_euler_xyz_rotation_matrix(rpy_rad: Sequence[float]) -> tuple[Vec3, Vec3, Vec3]:
    """Return the active XYZ-Euler rotation matrix used by the xCore Frame API."""
    rpy = as_vec3(rpy_rad)
    if rpy is None:
        raise ValueError("RPY rotation must contain three finite radians")
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rz(yaw) @ Ry(pitch) @ Rx(roll): the standard active XYZ Euler form.
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def transpose_rotation(rotation: Sequence[Sequence[float]]) -> tuple[Vec3, Vec3, Vec3]:
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("Rotation matrix must be 3x3")
    result = tuple(
        tuple(float(rotation[column][row]) for column in range(3))
        for row in range(3)
    )
    if not all(math.isfinite(value) for row in result for value in row):
        raise ValueError("Rotation matrix must be finite")
    return result  # type: ignore[return-value]


def rotate_vector(rotation: Sequence[Sequence[float]], vector: Sequence[float] | None) -> Vec3 | None:
    value = as_vec3(vector)
    if value is None:
        return None
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("Rotation matrix must be 3x3")
    result = tuple(
        sum(float(rotation[row][column]) * value[column] for column in range(3))
        for row in range(3)
    )
    return as_vec3(result)


def cross3(left: Sequence[float], right: Sequence[float]) -> Vec3:
    a = as_vec3(left)
    b = as_vec3(right)
    if a is None or b is None:
        raise ValueError("Cross-product operands must be finite 3D vectors")
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def transform_wrench(
    force_a_n: Sequence[float],
    torque_a_nm: Sequence[float],
    rotation_b_from_a: Sequence[Sequence[float]],
    origin_b_to_a_m: Sequence[float],
) -> tuple[Vec3, Vec3]:
    """Full wrench transform ``F_b=R F_a, tau_b=R tau_a+p×(R F_a)``.

    This mathematical helper intentionally does not decide whether an SDK
    frame request changed the wrench reference point.  Callers must supply a
    verified ``origin_b_to_a_m`` before using it for physical data.
    """
    force_b = rotate_vector(rotation_b_from_a, force_a_n)
    torque_rotated = rotate_vector(rotation_b_from_a, torque_a_nm)
    if force_b is None or torque_rotated is None:
        raise ValueError("Wrench must contain finite 3D force and torque")
    shift = cross3(origin_b_to_a_m, force_b)
    torque_b = tuple(torque_rotated[i] + shift[i] for i in range(3))
    if as_vec3(torque_b) is None:
        raise ValueError("Full wrench transform produced non-finite torque")
    return force_b, torque_b  # type: ignore[return-value]
