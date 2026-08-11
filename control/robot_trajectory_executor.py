"""Bounded scheduler for an already-approved slow Cartesian reference.

This module does not connect, power, select robot mode, or move to a start
pose.  It accepts only a successful :class:`ExecutionPreflight`, begins with a
hold at the already captured anchor, logs each dispatch before sending it, and
routes every completion/fault/operator request through ``request_stop(reason)``.
Software stop is not a substitute for the robot emergency stop, safety
controller, or an attending operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np

from control.execution_preflight import (
    ExecutionPreflight,
    audit_execution_trajectory_content,
)
from safety.experiment_safety import ExperimentSafetyConfig
from utils.clock import MonotonicClock, SYSTEM_CLOCK


@dataclass(frozen=True)
class ExecutionResult:
    completed: bool
    commands_dispatched: int
    stop_reason: str
    started_host_time_s: float
    finished_host_time_s: float


class RokaeMotionExecutor:
    """Send validated TCP targets through the confirmed xCoreSDK RT interface."""

    def __init__(
        self,
        motion_adapter: Any,
        acquisition: Any,
        logger: Any,
        safety: ExperimentSafetyConfig,
        *,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        self.motion_adapter = motion_adapter
        self.acquisition = acquisition
        self.logger = logger
        self.safety = safety
        self.clock = clock
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._native_stop_lock = threading.Lock()
        self._stop_reason: str | None = None
        self._stop_confirmed = False
        self._executing = False
        self._used = False

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def request_stop(self, reason: str) -> None:
        """Idempotent after success and retryable after a native stop failure."""
        reason = str(reason).strip()
        if not reason:
            raise ValueError("stop reason must not be empty")
        # Stop intent and each target/start transition share one lifecycle
        # lock.  Whichever acquires it first defines an unambiguous ordering;
        # once stop intent is published, no later target can pass the lock.
        with self._stop_lock:
            self._stop_event.set()
            if self._stop_reason is None:
                self._stop_reason = reason
            if self._stop_confirmed:
                return
            stop_reason = self._stop_reason
        # Motion stop is deliberately not followed by synchronous metadata I/O
        # here.  A disk flush may be the fault that triggered this path; stop
        # must remain able to reach the SDK without waiting for the logger.
        with self._native_stop_lock:
            with self._stop_lock:
                if self._stop_confirmed:
                    return
                stop_reason = self._stop_reason or stop_reason
            self.motion_adapter.request_stop(stop_reason)
            with self._stop_lock:
                self._stop_confirmed = True

    def _runtime_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        logger_health = getattr(self.logger, "healthy_signal", None)
        if logger_health is None:
            logger_health = getattr(self.logger, "healthy", False)
        if not bool(logger_health):
            reasons.append("logger_unhealthy")
        health = self.acquisition.latest_health()
        if not health.valid:
            reasons.append(health.invalid_reason or "acquisition_unhealthy")
        if not health.state_thread_alive:
            reasons.append("state_thread_not_alive")
        if not health.wrench_thread_alive:
            reasons.append("wrench_thread_not_alive")
        if not health.alignment_thread_alive:
            reasons.append("alignment_thread_not_alive")
        limits = self.safety
        if limits.max_state_age_s is None or health.state_age_s is None:
            reasons.append("state_age_limit_or_value_unavailable")
        elif health.state_age_s > limits.max_state_age_s:
            reasons.append("state_stale")
        if limits.max_wrench_age_s is None or health.wrench_age_s is None:
            reasons.append("wrench_age_limit_or_value_unavailable")
        elif health.wrench_age_s > limits.max_wrench_age_s:
            reasons.append("wrench_stale")
        if limits.max_state_wrench_skew_s is None or health.state_wrench_skew_s is None:
            reasons.append("state_wrench_skew_limit_or_value_unavailable")
        elif health.state_wrench_skew_s > limits.max_state_wrench_skew_s:
            reasons.append("state_wrench_skew_limit_exceeded")
        if limits.max_force_n is None or health.force_magnitude_n is None:
            reasons.append("force_limit_or_value_unavailable")
        elif health.force_magnitude_n > limits.max_force_n:
            reasons.append("force_limit_exceeded")
        if limits.max_torque_nm is None or health.torque_magnitude_nm is None:
            reasons.append("torque_limit_or_value_unavailable")
        elif health.torque_magnitude_nm > limits.max_torque_nm:
            reasons.append("torque_limit_exceeded")
        return tuple(dict.fromkeys(reasons))

    def _wait_until(self, deadline_s: float) -> None:
        while not self._stop_event.is_set():
            remaining = deadline_s - self.clock.now_s()
            if remaining <= 0.0:
                return
            runtime_reasons = self._runtime_reasons()
            if runtime_reasons:
                raise RuntimeError("runtime safety gate: " + ";".join(runtime_reasons))
            self._stop_event.wait(min(remaining, 0.01))
        raise RuntimeError(f"execution stop requested: {self._stop_reason}")

    def _require_live_start_anchor(self, initial_pose: tuple[float, ...]) -> None:
        """Recheck the cached robot pose immediately before starting the hold."""
        state = self.acquisition.latest_state_frame()
        if state is None or not state.valid:
            raise RuntimeError("live start-anchor state is unavailable")
        current = np.asarray(
            [*(state.tcp_position_m or ()), *(state.tcp_orientation_rad or ())],
            dtype=float,
        )
        expected = np.asarray(initial_pose, dtype=float)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise RuntimeError("live start-anchor TCP pose is unavailable")
        position_error = float(np.linalg.norm(current[:3] - expected[:3]))
        orientation_delta = np.arctan2(
            np.sin(current[3:] - expected[3:]),
            np.cos(current[3:] - expected[3:]),
        )
        orientation_error = float(np.linalg.norm(orientation_delta))
        position_limit = self.safety.max_start_anchor_position_error_m
        orientation_limit = self.safety.max_start_anchor_orientation_error_rad
        if position_limit is None or position_error > position_limit:
            raise RuntimeError("robot moved away from reviewed start-anchor position")
        if orientation_limit is None or orientation_error > orientation_limit:
            raise RuntimeError("robot moved away from reviewed start-anchor orientation")

    @staticmethod
    def _pose_from_row(row: Any) -> tuple[float, ...]:
        pose = tuple(
            float(row[name])
            for name in (
                "tcp_x_base",
                "tcp_y_base",
                "tcp_z_base",
                "tcp_rx",
                "tcp_ry",
                "tcp_rz",
            )
        )
        if not all(math.isfinite(value) for value in pose):
            raise RuntimeError("non-finite Cartesian command")
        return pose

    def _persist_command_intent(
        self,
        row: Any,
        pose: tuple[float, ...],
        *,
        latest_finish_s: float,
    ) -> None:
        """Persist one target before dispatch and enforce its absolute deadline."""

        remaining_log_budget_s = latest_finish_s - self.clock.now_s()
        if remaining_log_budget_s <= 0.0:
            raise RuntimeError("command deadline exhausted before intent logging")
        command_values = dict(
            host_time_s=self.clock.now_s(),
            trajectory_time_s=float(row["time_s"]),
            trajectory_phase=row.get("trajectory_phase", None),
            delta_x_R=row.get("delta_x_R", None),
            delta_y_R=row.get("delta_y_R", None),
            delta_z_R=row.get("delta_z_R", None),
            tcp_target_x=pose[0],
            tcp_target_y=pose[1],
            tcp_target_z=pose[2],
            tcp_target_rx=pose[3],
            tcp_target_ry=pose[4],
            tcp_target_rz=pose[5],
            q_hip_ref=float(row["q_hip_ref"]),
            q_knee_ref=float(row["q_knee_ref"]),
            command_valid=True,
            invalid_reason="",
        )
        bounded_append = getattr(
            self.logger,
            "append_trajectory_command_bounded",
            None,
        )
        if callable(bounded_append):
            bounded_append(
                timeout_s=remaining_log_budget_s,
                **command_values,
            )
        else:
            self.logger.append_trajectory_command(**command_values)
        if self.clock.now_s() > latest_finish_s:
            raise RuntimeError(
                "command deadline missed after intent logging; refusing dispatch"
            )

    def _require_post_attach_robot_summary(self) -> dict[str, Any]:
        """Re-query cached state and the SDK safety event after RT attachment."""

        adapter = getattr(self.acquisition, "adapter", None)
        reader = getattr(adapter, "get_robot_state_summary", None)
        if not callable(reader):
            raise RuntimeError("post-attach robot summary is unavailable")
        summary = reader()
        if not isinstance(summary, dict):
            raise RuntimeError("post-attach robot summary is invalid")
        reasons: list[str] = []
        if summary.get("connected") is not True:
            reasons.append("sdk_not_connected")
        if summary.get("state_valid") is not True:
            reasons.append("robot_state_invalid")
        if summary.get("state_stream_thread_alive") is not True:
            reasons.append("state_thread_not_alive")
        if summary.get("operation_state") != "IDLE":
            reasons.append("robot_not_idle_after_attachment")
        if summary.get("collision_state_query_valid") is not True:
            reasons.append("collision_state_unavailable_after_attachment")
        if summary.get("collision_state") is not False:
            reasons.append("collision_detected_after_attachment")
        metadata_value = summary.get("robot_metadata")
        metadata = metadata_value if isinstance(metadata_value, Mapping) else None
        if metadata is None:
            reasons.append("robot_metadata_unavailable_after_attachment")
        else:
            for runtime_name, expected_name in (
                ("robot_model", "expected_robot_model"),
                ("robot_serial_number", "expected_robot_serial_number"),
                ("controller_version", "expected_controller_version"),
            ):
                runtime_value = metadata.get(runtime_name)
                expected_value = getattr(self.safety, expected_name)
                if runtime_value != expected_value:
                    reasons.append(f"{runtime_name}_changed_after_preflight")

            payload_value = metadata.get("sdk_tool_payload")
            payload = payload_value if isinstance(payload_value, Mapping) else None
            if payload is None:
                reasons.append("payload_unavailable_after_attachment")
            else:
                scalar_payload = payload.get("toolset_load_mass_kg")
                expected_mass = self.safety.reviewed_payload_mass_kg
                if (
                    isinstance(scalar_payload, bool)
                    or not isinstance(scalar_payload, (int, float))
                    or expected_mass is None
                    or not math.isfinite(float(scalar_payload))
                    or not math.isclose(
                        float(scalar_payload),
                        expected_mass,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    reasons.append("payload_mass_changed_after_preflight")
                for runtime_name, expected_name in (
                    ("toolset_load_cog_m", "reviewed_payload_cog_m"),
                    (
                        "toolset_load_inertia_kg_m2",
                        "reviewed_payload_inertia_kg_m2",
                    ),
                ):
                    runtime_vector = payload.get(runtime_name)
                    expected_vector = getattr(self.safety, expected_name)
                    if not self._same_finite_vector(
                        runtime_vector,
                        expected_vector,
                        3,
                    ):
                        reasons.append(f"{runtime_name}_changed_after_preflight")
                tool_names = payload.get("sdk_available_tool_names")
                workpiece_names = payload.get("sdk_available_workobject_names")
                if not self._contains_exact_text(
                    tool_names,
                    self.safety.reviewed_tool_name,
                ):
                    reasons.append("reviewed_tool_disappeared_after_preflight")
                if not self._contains_exact_text(
                    workpiece_names,
                    self.safety.reviewed_workpiece_name,
                ):
                    reasons.append("reviewed_workpiece_disappeared_after_preflight")

            runtime_limits = metadata.get("joint_soft_limits_rad")
            reviewed_limits = self.safety.reviewed_joint_soft_limits_rad
            if not self._same_finite_matrix(runtime_limits, reviewed_limits, 6, 2):
                reasons.append("joint_soft_limits_changed_after_preflight")
            runtime_joints = summary.get("joint_position_rad")
            if not self._joints_inside_reviewed_limits(
                runtime_joints,
                reviewed_limits,
            ):
                reasons.append("joint_position_invalid_after_attachment")
        if reasons:
            raise RuntimeError(
                "post-attach robot safety summary: " + ";".join(reasons)
            )
        return summary

    @staticmethod
    def _same_finite_vector(
        runtime_value: object,
        expected_value: object,
        size: int,
    ) -> bool:
        try:
            runtime = np.asarray(runtime_value, dtype=float)
            expected = np.asarray(expected_value, dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(
            runtime.shape == (size,)
            and expected.shape == (size,)
            and np.isfinite(runtime).all()
            and np.isfinite(expected).all()
            and np.allclose(runtime, expected, rtol=0.0, atol=1e-9)
        )

    @staticmethod
    def _same_finite_matrix(
        runtime_value: object,
        expected_value: object,
        rows: int,
        columns: int,
    ) -> bool:
        try:
            runtime = np.asarray(runtime_value, dtype=float)
            expected = np.asarray(expected_value, dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(
            runtime.shape == (rows, columns)
            and expected.shape == (rows, columns)
            and np.isfinite(runtime).all()
            and np.isfinite(expected).all()
            and np.allclose(runtime, expected, rtol=0.0, atol=1e-9)
        )

    @staticmethod
    def _contains_exact_text(values: object, expected: str | None) -> bool:
        return bool(
            expected is not None
            and isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and expected in values
        )

    @staticmethod
    def _joints_inside_reviewed_limits(
        joints_value: object,
        limits_value: object,
    ) -> bool:
        try:
            joints = np.asarray(joints_value, dtype=float)
            limits = np.asarray(limits_value, dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(
            joints.shape == (6,)
            and limits.shape == (6, 2)
            and np.isfinite(joints).all()
            and np.isfinite(limits).all()
            and ((joints >= limits[:, 0]) & (joints <= limits[:, 1])).all()
        )

    def execute(
        self,
        trajectory: pd.DataFrame,
        preflight: ExecutionPreflight,
    ) -> ExecutionResult:
        """Execute one approved trajectory synchronously; never move to start."""
        with self._stop_lock:
            if self._executing:
                raise RuntimeError("trajectory execution is already active")
            if self._used:
                raise RuntimeError("RokaeMotionExecutor is single-use")
            if self._stop_reason is not None or self._stop_event.is_set():
                raise RuntimeError(
                    f"execution was stopped before start: {self._stop_reason}"
                )
            self._used = True
        preflight.require_allowed()
        preflight.require_live_trajectory_binding(trajectory)
        preflight.require_safety_binding(self.safety)
        self.safety.require_execute_allowed()
        self.logger.assert_healthy()
        if len(trajectory) < 2:
            raise ValueError("trajectory must contain at least two samples")
        required = {
            "time_s",
            "tcp_x_base",
            "tcp_y_base",
            "tcp_z_base",
            "tcp_rx",
            "tcp_ry",
            "tcp_rz",
            "q_hip_ref",
            "q_knee_ref",
            "trajectory_valid",
        }
        missing = required.difference(trajectory.columns)
        if missing:
            raise ValueError(f"trajectory missing execution columns: {sorted(missing)}")
        times = trajectory["time_s"].to_numpy(float)
        if not all(math.isfinite(value) for value in times) or not all(
            later > earlier for earlier, later in zip(times, times[1:])
        ):
            raise ValueError("trajectory time must be finite and strictly increasing")
        content_reasons, maximum_speed, maximum_acceleration = (
            audit_execution_trajectory_content(
                trajectory,
                self.safety,
                expected_trajectory_id=preflight.trajectory_id,
            )
        )
        if content_reasons:
            raise PermissionError(
                "real robot execution blocked by scheduler trajectory audit: "
                + ";".join(content_reasons)
            )
        if (
            preflight.maximum_tcp_speed_m_s is None
            or maximum_speed is None
            or not math.isclose(
                preflight.maximum_tcp_speed_m_s,
                maximum_speed,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise PermissionError(
                "real robot execution blocked: preflight_speed_audit_mismatch"
            )
        if (
            preflight.maximum_tcp_acceleration_m_s2 is None
            or maximum_acceleration is None
            or not math.isclose(
                preflight.maximum_tcp_acceleration_m_s2,
                maximum_acceleration,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise PermissionError(
                "real robot execution blocked: preflight_acceleration_audit_mismatch"
            )

        with self._stop_lock:
            if self._stop_reason is not None or self._stop_event.is_set():
                raise RuntimeError(
                    f"execution was stopped before start: {self._stop_reason}"
                )
            self._executing = True

        dispatched = 0
        started_s = self.clock.now_s()
        first = trajectory.iloc[0]
        initial_pose = self._pose_from_row(first)
        lateness_limit_s = self.safety.max_command_lateness_s
        if lateness_limit_s is None:
            raise RuntimeError("reviewed command-lateness limit is unavailable")
        try:
            initial_reasons = self._runtime_reasons()
            if initial_reasons:
                raise RuntimeError("runtime safety gate: " + ";".join(initial_reasons))
            reviewed_filter_hz = self.safety.reviewed_rt_filter_hz
            if reviewed_filter_hz is None:
                raise RuntimeError("reviewed realtime command filter is unavailable")
            self.motion_adapter.attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            post_attach_summary = self._require_post_attach_robot_summary()
            post_attach_reasons = self._runtime_reasons()
            if post_attach_reasons:
                raise RuntimeError(
                    "post-attach runtime safety gate: "
                    + ";".join(post_attach_reasons)
                )
            self._require_live_start_anchor(initial_pose)
            self.logger.update_metadata(
                {
                    "motion_attachment_confirmed": True,
                    "motion_attachment_host_time_s": self.clock.now_s(),
                    "post_attach_runtime_gate_passed": True,
                    "robot_state_summary_after_motion_attachment": (
                        post_attach_summary
                    ),
                    "robot_execution_approved": False,
                }
            )
            initial_start_deadline_s = self.clock.now_s() + lateness_limit_s
            # Starting the RT hold already publishes the first TCP target.
            # Persist that first command before any callback can become active.
            self._persist_command_intent(
                first,
                initial_pose,
                latest_finish_s=initial_start_deadline_s,
            )
            pre_start_reasons = self._runtime_reasons()
            if pre_start_reasons:
                raise RuntimeError(
                    "pre-start runtime safety gate: "
                    + ";".join(pre_start_reasons)
                )
            self.logger.update_metadata(
                {
                    "initial_hold_intent_persisted": True,
                    "robot_execution_approved": True,
                }
            )
            # Metadata persistence may have taken long enough for state/wrench,
            # the anchor, operation state, or collision state to change.  Make
            # this the last potentially blocking SDK gate before the atomic
            # stop-check/start transition.
            self._require_post_attach_robot_summary()
            final_start_reasons = self._runtime_reasons()
            if final_start_reasons:
                raise RuntimeError(
                    "final pre-start runtime safety gate: "
                    + ";".join(final_start_reasons)
                )
            self._require_live_start_anchor(initial_pose)
            with self._stop_lock:
                if self._stop_reason is not None or self._stop_event.is_set():
                    raise RuntimeError(
                        f"execution stopped before realtime start: {self._stop_reason}"
                    )
                if self.clock.now_s() > initial_start_deadline_s:
                    raise RuntimeError(
                        "initial hold deadline expired before realtime start"
                    )
                self.motion_adapter.start_cartesian_hold(initial_pose)
                dispatched = 1
            schedule_origin_s = self.clock.now_s() - float(times[0])
            for _, row in trajectory.iloc[1:].iterrows():
                command_deadline_s = schedule_origin_s + float(row["time_s"])
                self._wait_until(command_deadline_s)
                lateness_s = self.clock.now_s() - command_deadline_s
                if lateness_s > lateness_limit_s:
                    raise RuntimeError(
                        "command deadline missed; refusing late catch-up dispatch: "
                        f"lateness={lateness_s:.6f}s"
                    )
                runtime_reasons = self._runtime_reasons()
                if runtime_reasons:
                    raise RuntimeError("runtime safety gate: " + ";".join(runtime_reasons))
                pose = self._pose_from_row(row)
                self._persist_command_intent(
                    row,
                    pose,
                    latest_finish_s=command_deadline_s + lateness_limit_s,
                )
                post_log_reasons = self._runtime_reasons()
                if post_log_reasons:
                    raise RuntimeError(
                        "post-log runtime safety gate: "
                        + ";".join(post_log_reasons)
                    )
                with self._stop_lock:
                    if self._stop_reason is not None or self._stop_event.is_set():
                        raise RuntimeError(
                            "execution stopped before target dispatch: "
                            f"{self._stop_reason}"
                        )
                    if self.clock.now_s() > command_deadline_s + lateness_limit_s:
                        raise RuntimeError(
                            "command deadline expired during final safety check"
                        )
                    self.motion_adapter.send_cartesian_target(pose)
                    dispatched += 1
            self.request_stop("trajectory_completed")
            with self._stop_lock:
                if (
                    not self._stop_confirmed
                    or self._stop_reason != "trajectory_completed"
                ):
                    raise RuntimeError(
                        "trajectory did not reach confirmed normal completion: "
                        f"{self._stop_reason}"
                    )
            finished_s = self.clock.now_s()
            return ExecutionResult(
                completed=True,
                commands_dispatched=dispatched,
                stop_reason=self._stop_reason or "trajectory_completed",
                started_host_time_s=started_s,
                finished_host_time_s=finished_s,
            )
        except BaseException as exc:
            self.request_stop(f"execution_exception:{type(exc).__name__}:{exc}")
            raise
        finally:
            with self._stop_lock:
                self._executing = False


__all__ = ["ExecutionResult", "RokaeMotionExecutor"]
