"""Observation-only project adapter for ROKAE xCoreSDK state and wrench.

The public class in this module intentionally exposes no power, servo, alarm
reset, drag, calibration, or motion method.  Motion lives behind a separate
executor and an explicit experiment preflight gate.  The label describes
project calls only: vendor initialization/connect/disconnect side effects must
still be assessed under supervision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

from collection.state import KinematicStateFrame, finite_vector, utc_now_iso
from config import settings


def _clock_s() -> float:
    return time.perf_counter_ns() / 1_000_000_000.0


@dataclass(frozen=True)
class RobotWrenchFrame:
    """One raw xCoreSDK ``getEndTorque`` query with explicit host timing."""

    sequence_id: int
    host_query_start_s: float | None
    host_query_end_s: float | None
    host_publish_s: float
    host_monotonic_time_s: float | None
    wall_time_iso: str
    timestamp_source: str
    valid: bool
    invalid_reason: str
    raw_force_frame: str
    cartesian_force_raw_n: tuple[float, float, float] | None
    cartesian_torque_raw_nm: tuple[float, float, float] | None
    joint_measured_torque_nm: tuple[float, ...] | None
    joint_external_torque_nm: tuple[float, ...] | None

    @property
    def query_duration_ms(self) -> float | None:
        if self.host_query_start_s is None or self.host_query_end_s is None:
            return None
        duration = self.host_query_end_s - self.host_query_start_s
        return duration * 1000.0 if duration >= 0.0 else None


class RokaeRobotAdapter:
    """Narrow observation facade around :class:`RokaeRobot`.

    ``connect`` starts the native receive-only state cache for compatibility
    with existing diagnostics.  Calling ``start_state_stream`` again is safe.
    xCoreSDK supplies no device timestamp in the fields used here, so all
    timestamps are explicitly labelled host receive/query times.
    """

    def __init__(
        self,
        ip_address: str = settings.ROBOT_IP,
        *,
        local_ip: str = settings.ROBOT_LOCAL_IP,
        robot_class: str = settings.ROBOT_CLASS,
        state_interval_ms: int = settings.ROBOT_STATE_MS,
        native_robot: Any | None = None,
    ) -> None:
        if native_robot is None:
            # Importing this module is cross-platform; the native extension is
            # loaded only when the wrapped object's connect() is called.
            from hardware.windows.rokae_xcore import RokaeRobot

            native_robot = RokaeRobot(
                ip_address=ip_address,
                local_ip=local_ip,
                robot_class=robot_class,
                state_interval_ms=state_interval_ms,
                max_linear_speed_m_s=settings.ROBOT_MAX_LINEAR_SPEED_M_S,
                command_cache_size=settings.ROBOT_CMD_CACHE,
                rt_network_tolerance_percent=settings.ROBOT_RT_NETWORK_TOLERANCE,
                rt_filter_hz=settings.ROBOT_RT_FILTER_HZ,
            )
        self._robot = native_robot
        self._wrench_sequence_id = 0
        # A safety event observed by an earlier summary in the same adapter
        # lifetime must not disappear if a later event query is edge-triggered.
        self._collision_latched = False

    @property
    def native_robot(self) -> Any:
        """Internal integration hook for the existing independent wrench source."""
        return self._robot

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def start_state_stream(self) -> None:
        self._robot.start_state_stream()

    def stop_state_stream(self) -> None:
        self._robot.stop_state_stream()

    def is_connected(self) -> bool:
        return bool(getattr(self._robot, "is_connected", False))

    @property
    def state_thread_alive(self) -> bool:
        thread = getattr(self._robot, "_state_thread", None)
        return bool(thread is not None and thread.is_alive())

    def read_tcp_pose(self) -> tuple[float, float, float, float, float, float]:
        pose = self._robot.get_cartesian_pose()
        if not finite_vector(pose, 6):
            raise RuntimeError("ROKAE TCP pose is unavailable or non-finite")
        return tuple(float(value) for value in pose)  # type: ignore[return-value]

    def read_joint_positions(self) -> tuple[float, ...]:
        joints = self._robot.get_joint_angles()
        if not finite_vector(joints, 6):
            raise RuntimeError("ROKAE joint positions are unavailable or non-finite")
        return tuple(float(value) for value in joints)

    def read_state_frame(self) -> KinematicStateFrame:
        return self._robot.get_state_frame()

    def read_robot_metadata(self) -> dict[str, Any]:
        """Read SDK identity, tool/payload, and soft-limit metadata unchanged."""

        soft_limit_reader = getattr(self._robot, "get_joint_soft_limits_rad", None)
        if callable(soft_limit_reader):
            # Refresh is a normal read-only xCoreSDK getSoftLimit query.  A
            # failed refresh is preserved by the native metadata error fields.
            soft_limit_reader(refresh=True)
        metadata = self._robot.get_robot_metadata()
        if not isinstance(metadata, Mapping):
            raise RuntimeError("ROKAE robot metadata is not a mapping")
        return dict(metadata)

    def read_internal_wrench(self, reference_frame: str = "world") -> RobotWrenchFrame:
        """Perform one unmodified controller wrench query.

        This does not calibrate, bias, compensate, or claim a base-frame point
        transform.  The existing background wrench source may be used by the
        acquisition layer when independent fixed-rate updates are required.
        """
        raw = self._robot.get_end_wrench(reference_frame)
        publish_s = _clock_s()
        self._wrench_sequence_id += 1

        def tuple_if_finite(value: Any, size: int) -> tuple[float, ...] | None:
            if not finite_vector(value, size):
                return None
            return tuple(float(item) for item in value)

        force = tuple_if_finite(raw.get("cartesian_force_raw_n"), 3)
        torque = tuple_if_finite(raw.get("cartesian_torque_raw_nm"), 3)
        measured = tuple_if_finite(raw.get("joint_measured_torque_nm"), 6)
        external = tuple_if_finite(raw.get("joint_external_torque_nm"), 6)

        def optional_finite(name: str) -> float | None:
            value = raw.get(name)
            if value is None:
                return None
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None

        query_start = optional_finite("force_query_started_s")
        query_end = optional_finite("force_query_finished_s")
        midpoint = optional_finite("host_monotonic_time_s")
        valid = (
            force is not None
            and torque is not None
            and measured is not None
            and external is not None
            and query_start is not None
            and query_end is not None
            and query_end >= query_start
            and midpoint is not None
        )
        reason = "" if valid else "incomplete_or_invalid_get_end_torque_result"
        return RobotWrenchFrame(
            sequence_id=self._wrench_sequence_id,
            host_query_start_s=query_start,
            host_query_end_s=query_end,
            host_publish_s=publish_s,
            host_monotonic_time_s=midpoint,
            wall_time_iso=str(raw.get("wall_time_iso") or utc_now_iso()),
            timestamp_source="host_query_time_no_robot_device_timestamp",
            valid=valid,
            invalid_reason=reason,
            raw_force_frame=str(raw.get("raw_force_frame", reference_frame)),
            cartesian_force_raw_n=force,  # type: ignore[arg-type]
            cartesian_torque_raw_nm=torque,  # type: ignore[arg-type]
            joint_measured_torque_nm=measured,
            joint_external_torque_nm=external,
        )

    def get_robot_state_summary(self) -> dict[str, Any]:
        frame = self.read_state_frame()
        now_s = _clock_s()
        age_ms = None
        if frame.host_monotonic_time_s is not None:
            age_s = now_s - frame.host_monotonic_time_s
            age_ms = age_s * 1000.0 if age_s >= 0.0 else None
        try:
            metadata = self.read_robot_metadata()
        except Exception as exc:
            metadata = {
                "metadata_read_error": f"{type(exc).__name__}:{exc}",
            }
        collision_state: bool | None = None
        collision_state_query_valid = False
        collision_state_invalid_reason = "collision_state_query_unavailable"
        collision_reader = getattr(self._robot, "get_collision_state", None)
        if callable(collision_reader):
            try:
                collision_value = collision_reader()
                if type(collision_value) is bool:
                    if collision_value:
                        self._collision_latched = True
                    collision_state = self._collision_latched or collision_value
                    collision_state_query_valid = True
                    collision_state_invalid_reason = ""
                else:
                    collision_state_invalid_reason = str(
                        getattr(self._robot, "_collision_error", None)
                        or "collision_state_not_returned_by_sdk"
                    )
            except Exception as exc:
                collision_state_invalid_reason = (
                    f"collision_state_query_error:{type(exc).__name__}:{exc}"
                )

        soft_limits = metadata.get("joint_soft_limits_rad")
        joint_soft_limits_valid = False
        if (
            isinstance(soft_limits, Sequence)
            and not isinstance(soft_limits, (str, bytes))
            and len(soft_limits) == 6
        ):
            try:
                joint_soft_limits_valid = all(
                    isinstance(pair, Sequence)
                    and not isinstance(pair, (str, bytes))
                    and len(pair) == 2
                    and all(math.isfinite(float(value)) for value in pair)
                    and float(pair[0]) < float(pair[1])
                    for pair in soft_limits
                )
            except (TypeError, ValueError):
                joint_soft_limits_valid = False

        tool_payload = metadata.get("sdk_tool_payload")
        sdk_tool_payload_read_valid = (
            isinstance(tool_payload, Mapping)
            and "read_error" not in tool_payload
            and "toolset_load_mass_kg" in tool_payload
            and "toolset_load_cog_m" in tool_payload
            and "toolset_load_inertia_kg_m2" in tool_payload
        )
        return {
            "connected": self.is_connected(),
            "state_stream_thread_alive": self.state_thread_alive,
            "timestamp_source": "host_receive_time_no_robot_device_timestamp",
            "state_valid": frame.valid,
            "state_invalid_reason": frame.invalid_reason,
            "state_sequence_id": frame.sequence_id,
            "state_host_monotonic_time_s": frame.host_monotonic_time_s,
            "state_age_ms": age_ms,
            "tcp_pose_base_m_rad": list(frame.tcp_position_m or ())
            + list(frame.tcp_orientation_rad or ()),
            "joint_position_rad": list(frame.joint_position_rad or ()),
            "operation_state": frame.operation_state,
            "collision_state": collision_state,
            "collision_state_query_valid": collision_state_query_valid,
            "collision_state_invalid_reason": collision_state_invalid_reason,
            "collision_state_timestamp_source": (
                "host_query_unsynchronized_xcoresdk_safety_event"
            ),
            "joint_soft_limits_valid": joint_soft_limits_valid,
            "sdk_tool_payload_read_valid": sdk_tool_payload_read_valid,
            "robot_metadata": metadata,
        }

    def state_frame_dict(self) -> dict[str, Any]:
        """Serialization helper for diagnostics and episode metadata."""
        return asdict(self.read_state_frame())


__all__ = ["RobotWrenchFrame", "RokaeRobotAdapter"]
