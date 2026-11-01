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
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np

from control.execution_preflight import (
    ExecutionPreflight,
    audit_execution_trajectory_content,
)
from safety.experiment_safety import ExperimentSafetyConfig
from utils.clock import MonotonicClock, SYSTEM_CLOCK
from hardware.safety_events import safety_event_block_reason, collision_feedback_allows_motion


# The first target is persisted before the realtime loop exists, so it has no
# trajectory schedule to be late against.  Give Windows enough time to run the
# bounded logging worker, then recheck every live gate immediately before the
# hold starts.  Scheduled targets continue to use max_command_lateness_s.
INITIAL_HOLD_INTENT_TIMEOUT_S = 0.25
# Blocking metadata persistence is complete before motion begins.  During that
# bounded pause the isolated 50 Hz wrench worker can keep producing while the
# parent has not yet admitted its newest sample.  Wait for an actually fresh,
# aligned sample afterwards; this timeout does not relax any safety limit.
PRESTART_ACQUISITION_RECOVERY_TIMEOUT_S = 0.5
# Starting the vendor realtime callback can hold the Python caller for tens of
# milliseconds while the target remains the already-validated anchor.  Do not
# start trajectory time, or permanently arm motion-time freshness latches,
# until both independent streams have recovered from that bounded transition.
POST_HOLD_ACQUISITION_RECOVERY_TIMEOUT_S = 0.5
# The 50 Hz wrench stream and 125 Hz state stream are independently supervised.
# A dedicated watchdog owns continuous live health checks.  The deadline
# scheduler can therefore make one interruptible wait and reserve only a short
# final interval for a GIL-cooperative spin.  A fresh full gate immediately
# after the deadline remains mandatory; no target is sent from cached approval.
DEADLINE_FINE_WAIT_WINDOW_S = 0.030
DEADLINE_FINE_SLEEP_S = 0.0005
DEADLINE_BUSY_WAIT_WINDOW_S = 0.010
RUNTIME_WATCHDOG_PERIOD_S = 0.005
RUNTIME_WATCHDOG_READY_TIMEOUT_S = 0.25
RUNTIME_WATCHDOG_JOIN_TIMEOUT_S = 0.5


class _WindowsSchedulerPriority:
    """Bounded Windows scheduling boost with mandatory restoration.

    ``HIGH_PRIORITY_CLASS`` is intentionally used instead of Windows' realtime
    class.  The calling scheduler thread is raised to ``HIGHEST`` only while a
    real trajectory executor owns the system clock.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled and sys.platform == "win32")
        self._kernel = None
        self._process = None
        self._thread = None
        self._old_process_class = None
        self._old_thread_priority = None
        self._old_switch_interval = None
        self._avrt = None
        self._mmcss_handle = None
        self._active = False

    def enter(self) -> None:
        if not self.enabled:
            return
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetCurrentThread.restype = wintypes.HANDLE
        kernel.SwitchToThread.argtypes = []
        kernel.SwitchToThread.restype = wintypes.BOOL
        kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel.GetPriorityClass.restype = wintypes.DWORD
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.GetThreadPriority.argtypes = [wintypes.HANDLE]
        kernel.GetThreadPriority.restype = ctypes.c_int
        kernel.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        kernel.SetThreadPriority.restype = wintypes.BOOL

        process = kernel.GetCurrentProcess()
        thread = kernel.GetCurrentThread()
        old_process_class = kernel.GetPriorityClass(process)
        old_thread_priority = kernel.GetThreadPriority(thread)
        if not old_process_class or old_thread_priority == 0x7FFFFFFF:
            raise RuntimeError("unable to read Windows scheduler priority")
        if not kernel.SetPriorityClass(process, 0x00000080):
            raise RuntimeError(
                f"unable to set Windows HIGH_PRIORITY_CLASS: {ctypes.get_last_error()}"
            )
        if not kernel.SetThreadPriority(thread, 2):
            error = ctypes.get_last_error()
            kernel.SetPriorityClass(process, old_process_class)
            raise RuntimeError(
                f"unable to set Windows THREAD_PRIORITY_HIGHEST: {error}"
            )
        # xCoreSDK owns a time-sensitive native callback thread.  Register only
        # this Python deadline scheduler with MMCSS so it cannot be suspended
        # for tens of milliseconds by an equal-priority callback.  HIGH is
        # intentionally used instead of AVRT_PRIORITY_CRITICAL.
        avrt = ctypes.WinDLL("avrt", use_last_error=True)
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        avrt.AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        avrt.AvSetMmThreadPriority.restype = wintypes.BOOL
        avrt.AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
        avrt.AvRevertMmThreadCharacteristics.restype = wintypes.BOOL
        task_index = wintypes.DWORD(0)
        mmcss_handle = avrt.AvSetMmThreadCharacteristicsW(
            "Pro Audio", ctypes.byref(task_index)
        )
        if not mmcss_handle:
            error = ctypes.get_last_error()
            kernel.SetThreadPriority(thread, old_thread_priority)
            kernel.SetPriorityClass(process, old_process_class)
            raise RuntimeError(
                f"unable to register Windows MMCSS Pro Audio task: {error}"
            )
        if not avrt.AvSetMmThreadPriority(mmcss_handle, 1):
            error = ctypes.get_last_error()
            avrt.AvRevertMmThreadCharacteristics(mmcss_handle)
            kernel.SetThreadPriority(thread, old_thread_priority)
            kernel.SetPriorityClass(process, old_process_class)
            raise RuntimeError(
                f"unable to set Windows MMCSS HIGH priority: {error}"
            )
        self._kernel = kernel
        self._process = process
        self._thread = thread
        self._old_process_class = old_process_class
        self._old_thread_priority = old_thread_priority
        self._avrt = avrt
        self._mmcss_handle = mmcss_handle
        # The acquisition, audit, and SDK callback threads all share CPython's
        # GIL.  Its default 5 ms switching quantum can compound across several
        # ready threads even when this OS thread has elevated priority.  A
        # 1 ms quantum keeps the 1 kHz SDK callback and every producer live
        # while bounding scheduler-thread starvation.
        self._old_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        self._active = True

    def restore(self, active_exception: BaseException | None) -> None:
        if not self._active:
            return
        errors: list[str] = []
        if self._mmcss_handle is not None and not self._avrt.AvRevertMmThreadCharacteristics(
            self._mmcss_handle
        ):
            errors.append("mmcss_priority_restore_failed")
        self._mmcss_handle = None
        if not self._kernel.SetThreadPriority(
            self._thread, self._old_thread_priority
        ):
            errors.append("thread_priority_restore_failed")
        if not self._kernel.SetPriorityClass(
            self._process, self._old_process_class
        ):
            errors.append("process_priority_restore_failed")
        if self._old_switch_interval is not None:
            sys.setswitchinterval(self._old_switch_interval)
        self._active = False
        if not errors:
            return
        message = ";".join(errors)
        if active_exception is not None and hasattr(active_exception, "add_note"):
            active_exception.add_note(message)
            return
        raise RuntimeError(message)

    def yield_once(self) -> None:
        if not self._active or self._kernel is None:
            raise RuntimeError("Windows scheduler priority is not active")
        self._kernel.SwitchToThread()


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
        execution_guard=None,
        dispatch_observer=None,
        use_windows_scheduler_priority: bool = False,
    ) -> None:
        self.motion_adapter = motion_adapter
        self.acquisition = acquisition
        self.logger = logger
        self.safety = safety
        self.clock = clock
        self.execution_guard = execution_guard
        self.dispatch_observer = dispatch_observer
        if type(use_windows_scheduler_priority) is not bool:
            raise TypeError("use_windows_scheduler_priority must be bool")
        self._use_windows_scheduler_priority = bool(
            use_windows_scheduler_priority and clock is SYSTEM_CLOCK
        )
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._native_stop_lock = threading.Lock()
        self._stop_reason: str | None = None
        self._stop_confirmed = False
        self._executing = False
        self._used = False
        self._runtime_gate_count = 0
        self._max_runtime_gate_ns = 0
        self._max_health_snapshot_ns = 0
        self._last_wait_diagnostics: dict[str, Any] = {}
        self._deadline_yield_once = None
        self._runtime_watchdog_stop = threading.Event()
        self._runtime_watchdog_ready = threading.Event()
        self._runtime_watchdog_thread: threading.Thread | None = None
        self._runtime_watchdog_fault: str | None = None

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
        gate_started_ns = time.perf_counter_ns()
        reasons: list[str] = []
        execution_guard = getattr(self, "execution_guard", None)
        if execution_guard is not None:
            try:
                reason = execution_guard()
                if reason:
                    reasons.append(str(reason))
            except Exception as exc:
                reasons.append("external_acquisition_guard_failed:" + str(exc))
        event_reason = safety_event_block_reason(getattr(self.acquisition, "adapter", None))
        if event_reason:
            reasons.append(event_reason)
        logger_health = getattr(self.logger, "healthy_signal", None)
        if logger_health is None:
            logger_health = getattr(self.logger, "healthy", False)
        if not bool(logger_health):
            reasons.append("logger_unhealthy")
        # The wrench provider is already serviced by two independent producer
        # threads.  A command-deadline gate must consume their immutable health
        # snapshot rather than becoming the third audit-draining poller; the
        # latter can monopolize the scheduler around a disk flush.
        health_reader = getattr(
            self.acquisition,
            "latest_health_snapshot",
            self.acquisition.latest_health,
        )
        health_started_ns = time.perf_counter_ns()
        health = health_reader()
        health_duration_ns = time.perf_counter_ns() - health_started_ns
        self._max_health_snapshot_ns = max(
            self._max_health_snapshot_ns, health_duration_ns
        )
        tolerated_stale = {
            "wrench_stale_or_unavailable",
            "state_wrench_skew_unavailable_or_exceeded",
        }
        health_reasons = set(filter(None, health.invalid_reason.split(";")))
        grace_only = bool(
            getattr(health, "wrench_stale_grace_active", False)
            and health_reasons
            and health_reasons.issubset(tolerated_stale)
        )
        if not health.valid and not grace_only:
            reasons.append(health.invalid_reason or "acquisition_unhealthy")
        if not health.state_thread_alive:
            reasons.append("state_thread_not_alive")
        if not health.wrench_thread_alive:
            reasons.append("wrench_thread_not_alive")
        if not health.alignment_thread_alive:
            reasons.append("alignment_thread_not_alive")
        limits = self.safety
        for name in ("state_age_s", "wrench_age_s", "state_wrench_skew_s", "force_magnitude_n", "torque_magnitude_nm"):
            value = getattr(health, name)
            if grace_only and name in ("force_magnitude_n", "torque_magnitude_nm") and value is None:
                continue
            if value is None or not math.isfinite(value) or value < 0:
                reasons.append(name + "_invalid")
        if limits.max_state_age_s is None or health.state_age_s is None:
            reasons.append("state_age_limit_or_value_unavailable")
        elif health.state_age_s > limits.max_state_age_s:
            reasons.append("state_stale")
        if limits.max_wrench_age_s is None or health.wrench_age_s is None:
            reasons.append("wrench_age_limit_or_value_unavailable")
        elif health.wrench_age_s > limits.max_wrench_age_s and not grace_only:
            reasons.append("wrench_stale")
        if limits.max_state_wrench_skew_s is None or health.state_wrench_skew_s is None:
            reasons.append("state_wrench_skew_limit_or_value_unavailable")
        elif health.state_wrench_skew_s > limits.max_state_wrench_skew_s and not grace_only:
            reasons.append("state_wrench_skew_limit_exceeded")
        if limits.max_force_n is None or (health.force_magnitude_n is None and not grace_only):
            reasons.append("force_limit_or_value_unavailable")
        elif health.force_magnitude_n is not None and health.force_magnitude_n > limits.max_force_n:
            reasons.append("force_limit_exceeded")
        if limits.max_torque_nm is None or (health.torque_magnitude_nm is None and not grace_only):
            reasons.append("torque_limit_or_value_unavailable")
        elif health.torque_magnitude_nm is not None and health.torque_magnitude_nm > limits.max_torque_nm:
            reasons.append("torque_limit_exceeded")
        result = tuple(dict.fromkeys(reasons))
        gate_duration_ns = time.perf_counter_ns() - gate_started_ns
        self._runtime_gate_count += 1
        self._max_runtime_gate_ns = max(
            self._max_runtime_gate_ns, gate_duration_ns
        )
        return result

    def _runtime_watchdog_loop(self) -> None:
        """Continuously supervise live gates and stop without scheduler help."""
        try:
            while (
                not self._runtime_watchdog_stop.is_set()
                and not self._stop_event.is_set()
            ):
                reasons = self._runtime_reasons()
                if reasons:
                    fault = "runtime safety gate: " + ";".join(reasons)
                    self._runtime_watchdog_fault = fault
                    self.request_stop("runtime_watchdog:" + fault)
                    return
                self._runtime_watchdog_ready.set()
                self._runtime_watchdog_stop.wait(RUNTIME_WATCHDOG_PERIOD_S)
        except BaseException as exc:
            fault = f"runtime watchdog failure: {type(exc).__name__}:{exc}"
            self._runtime_watchdog_fault = fault
            try:
                self.request_stop("runtime_watchdog:" + fault)
            except BaseException as stop_exc:
                self._runtime_watchdog_fault = (
                    fault
                    + f"; stop failure: {type(stop_exc).__name__}:{stop_exc}"
                )

    def _start_runtime_watchdog(self) -> None:
        """Start only for a real-clock execution after the anchor hold is live."""
        if self.clock is not SYSTEM_CLOCK:
            return
        if self._runtime_watchdog_thread is not None:
            raise RuntimeError("runtime watchdog already started")
        self._runtime_watchdog_stop.clear()
        self._runtime_watchdog_ready.clear()
        self._runtime_watchdog_fault = None
        thread = threading.Thread(
            target=self._runtime_watchdog_loop,
            name="robot-runtime-safety-watchdog",
            daemon=True,
        )
        self._runtime_watchdog_thread = thread
        thread.start()
        if not self._runtime_watchdog_ready.wait(RUNTIME_WATCHDOG_READY_TIMEOUT_S):
            fault = self._runtime_watchdog_fault
            if fault is not None:
                raise RuntimeError(fault)
            raise RuntimeError("runtime safety watchdog did not become ready")

    def _raise_runtime_watchdog_fault(self) -> None:
        fault = self._runtime_watchdog_fault
        if fault is not None:
            raise RuntimeError(fault)

    def _stop_runtime_watchdog(self, *, require_clean: bool) -> None:
        thread = self._runtime_watchdog_thread
        if thread is None:
            return
        self._runtime_watchdog_stop.set()
        if thread is not threading.current_thread():
            thread.join(RUNTIME_WATCHDOG_JOIN_TIMEOUT_S)
        if thread.is_alive():
            fault = "runtime safety watchdog did not stop"
            if self._runtime_watchdog_fault is None:
                self._runtime_watchdog_fault = fault
        self._runtime_watchdog_thread = None
        if require_clean:
            self._raise_runtime_watchdog_fault()

    def _wait_until(self, deadline_s: float) -> None:
        # The live watchdog independently checks all safety inputs and calls
        # the native stop path on any fault.  One high-resolution Windows sleep
        # avoids both the coarse Event.wait timeout and the repeated scheduling
        # losses observed with Sleep(0)/SwitchToThread.  CPython's 1 ms switch
        # interval keeps the watchdog,
        # acquisition, audit, and SDK callback threads runnable during the
        # bounded 10 ms final spin.
        if self._runtime_watchdog_thread is not None:
            fine_entry_remaining_s = None
            while True:
                self._raise_runtime_watchdog_fault()
                if self._stop_event.is_set():
                    raise RuntimeError(
                        f"execution stop requested: {self._stop_reason}"
                    )
                remaining = deadline_s - self.clock.now_s()
                if remaining <= 0.0:
                    self._last_wait_diagnostics = {
                        "fine_entry_remaining_s": fine_entry_remaining_s,
                        "fine_yields": 0,
                        "wake_lateness_s": self.clock.now_s() - deadline_s,
                    }
                    return
                if remaining > DEADLINE_BUSY_WAIT_WINDOW_S:
                    # Python 3.11+ implements time.sleep() on Windows with a
                    # high-resolution waitable timer.  It is intentionally not
                    # the stop event: the independent watchdog already invokes
                    # native stop immediately, and this scheduler observes the
                    # published fault as soon as the timer returns.
                    time.sleep(remaining - DEADLINE_BUSY_WAIT_WINDOW_S)
                    continue
                if fine_entry_remaining_s is None:
                    fine_entry_remaining_s = remaining
                # Intentionally no OS sleep/yield here.  The interpreter still
                # hands off the GIL every 1 ms while this executor owns the
                # Windows priority scope.
                continue

        fine_entry_remaining_s = None
        fine_yields = 0
        while not self._stop_event.is_set():
            remaining = deadline_s - self.clock.now_s()
            if remaining <= 0.0:
                self._last_wait_diagnostics = {
                    "fine_entry_remaining_s": fine_entry_remaining_s,
                    "fine_yields": fine_yields,
                    "wake_lateness_s": self.clock.now_s() - deadline_s,
                }
                return
            if remaining <= DEADLINE_FINE_WAIT_WINDOW_S:
                if fine_entry_remaining_s is None:
                    fine_entry_remaining_s = remaining
                fine_yields += 1
                # Windows Event.wait() can oversleep a short final interval by
                # more than the reviewed dispatch budget.  On the real Windows
                # clock, Sleep(0) keeps this elevated scheduler runnable while
                # releasing the GIL to the SDK's 1 kHz Python callback and the
                # acquisition producers. SwitchToThread yields to at most one
                # ready thread on the current processor, avoiding Sleep(0)'s
                # repeated loss of the remaining time slice. Deterministic/fake
                # clocks retain a positive sleep so tests can advance time.
                if sys.platform == "win32" and self.clock is SYSTEM_CLOCK:
                    self._deadline_yield_once()
                else:
                    time.sleep(min(remaining, DEADLINE_FINE_SLEEP_S))
                continue
            runtime_reasons = self._runtime_reasons()
            if runtime_reasons:
                raise RuntimeError("runtime safety gate: " + ";".join(runtime_reasons))
            self._stop_event.wait(
                min(remaining - DEADLINE_FINE_WAIT_WINDOW_S, 0.005)
            )
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

    def _require_post_attach_robot_summary(self):
        adapter = self.acquisition.adapter
        summary = adapter.get_robot_state_summary()
        if not isinstance(summary, dict) or summary.get("connected") is not True or summary.get("state_valid") is not True or summary.get("operation_state") != "RTCONTROLLING":
            raise RuntimeError("robot is not ready for realtime commands")
        if not collision_feedback_allows_motion(summary):
            raise RuntimeError("collision protection state unavailable or active")
        limits = summary.get("robot_metadata", {}).get("joint_soft_limits_rad")
        if not self._joints_inside_reviewed_limits(summary.get("joint_position_rad"), limits):
            raise RuntimeError("joint state or controller soft limits invalid")
        return summary




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
        scheduled_poses = tuple(
            self._pose_from_row(row) for _, row in trajectory.iterrows()
        )
        callback_scheduled = all(
            callable(getattr(self.motion_adapter, name, None))
            for name in (
                "configure_cartesian_schedule",
                "admit_scheduled_target",
                "wait_scheduled_target",
                "scheduled_origin_s",
                "start_scheduled_timing",
            )
        )
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
        scheduler_priority = _WindowsSchedulerPriority(
            self._use_windows_scheduler_priority and not callback_scheduled
        )
        callback_old_switch_interval: float | None = None
        try:
            scheduler_priority.enter()
            if callback_scheduled and sys.platform == "win32":
                # The MMCSS SDK callback still has to acquire CPython's GIL.
                # Keep Python producer critical sections below the reviewed
                # command-lateness budget without elevating the main thread.
                callback_old_switch_interval = sys.getswitchinterval()
                sys.setswitchinterval(0.001)
            self._deadline_yield_once = scheduler_priority.yield_once
            initial_reasons = self._runtime_reasons()
            if initial_reasons:
                raise RuntimeError("runtime safety gate: " + ";".join(initial_reasons))
            reviewed_filter_hz = self.safety.rt_filter_hz
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
            # Arm the provider's bounded in-flight-query grace before metadata
            # I/O can delay parent-side IPC admission.  No motion exists yet,
            # and a strict fresh-health gate below must still pass before the
            # realtime callback is allowed to start.
            arm_grace = getattr(self.acquisition, "arm_runtime_grace", None)
            if arm_grace is not None:
                arm_grace()
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
            initial_intent_deadline_s = (
                self.clock.now_s() + INITIAL_HOLD_INTENT_TIMEOUT_S
            )
            # Starting the RT hold already publishes the first TCP target.
            # Persist that first command before any callback can become active.
            self._persist_command_intent(
                first,
                initial_pose,
                latest_finish_s=initial_intent_deadline_s,
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
            wait_healthy = getattr(self.acquisition, "wait_until_healthy", None)
            if callable(wait_healthy) and not wait_healthy(
                timeout_s=PRESTART_ACQUISITION_RECOVERY_TIMEOUT_S
            ):
                raise RuntimeError(
                    "pre-start acquisition did not recover after metadata persistence"
                )
            # Metadata persistence may have taken long enough for state/wrench,
            # the anchor, operation state, or collision state to change.  Make
            # this the last potentially blocking SDK gate before the atomic
            # stop-check/start transition.
            final_summary = self._require_post_attach_robot_summary()
            final_start_reasons = self._runtime_reasons()
            if final_start_reasons:
                raise RuntimeError(
                    "final pre-start runtime safety gate: "
                    + ";".join(final_start_reasons)
                )
            self._require_live_start_anchor(initial_pose)
            if callback_scheduled:
                self.motion_adapter.configure_cartesian_schedule(
                    scheduled_poses,
                    times,
                    max_lateness_s=lateness_limit_s,
                )
                # Target zero was durably persisted above.  Admission is the
                # only operation that makes a preloaded target visible to the
                # realtime callback.
                self.motion_adapter.admit_scheduled_target(0)
            with self._stop_lock:
                if self._stop_reason is not None or self._stop_event.is_set():
                    raise RuntimeError(
                        f"execution stopped before realtime start: {self._stop_reason}"
                    )
                dispatch_start = time.perf_counter_ns()
                self.motion_adapter.start_cartesian_hold(initial_pose)
            if callback_scheduled:
                _, dispatch_start, dispatch_end = (
                    self.motion_adapter.wait_scheduled_target(
                        0, timeout_s=INITIAL_HOLD_INTENT_TIMEOUT_S
                    )
                )
            else:
                dispatch_end = time.perf_counter_ns()
            if self.dispatch_observer is not None:
                self.dispatch_observer(
                    float(times[0]), dispatch_start, dispatch_end
                )
            dispatched = 1
            if callable(wait_healthy) and not wait_healthy(
                timeout_s=POST_HOLD_ACQUISITION_RECOVERY_TIMEOUT_S
            ):
                raise RuntimeError(
                    "post-hold acquisition did not recover before trajectory timing"
                )
            arm_freshness = getattr(
                self.acquisition, "arm_runtime_freshness", None
            )
            if callable(arm_freshness):
                arm_freshness()
            post_hold_reasons = self._runtime_reasons()
            if post_hold_reasons:
                raise RuntimeError(
                    "post-hold runtime safety gate: "
                    + ";".join(post_hold_reasons)
                )
            self._start_runtime_watchdog()
            if callback_scheduled:
                self.motion_adapter.start_scheduled_timing(
                    timeout_s=INITIAL_HOLD_INTENT_TIMEOUT_S
                )
            schedule_origin_s = (
                self.motion_adapter.scheduled_origin_s()
                if callback_scheduled
                else self.clock.now_s() - float(times[0])
            )
            for schedule_index, (_, row) in enumerate(
                trajectory.iloc[1:].iterrows(), start=1
            ):
                command_deadline_s = schedule_origin_s + float(row["time_s"])
                pose = scheduled_poses[schedule_index]
                # Persist the next intent during the interval after the
                # previous dispatch, rather than beginning disk I/O at the
                # target instant.  The dispatch deadline itself is unchanged;
                # every live gate is checked again after the wait below.
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
                if callback_scheduled:
                    # The callback owns the reviewed deadline.  Main-thread
                    # admission occurs only after durable intent logging and a
                    # fresh safety gate; the independent watchdog remains live
                    # until callback dispatch or stop.
                    self.motion_adapter.admit_scheduled_target(schedule_index)
                    remaining_s = (
                        command_deadline_s
                        + lateness_limit_s
                        + 0.25
                        - self.clock.now_s()
                    )
                    _, dispatch_start, dispatch_end = (
                        self.motion_adapter.wait_scheduled_target(
                            schedule_index,
                            timeout_s=max(0.0, remaining_s),
                        )
                    )
                    actual_dispatch_s = dispatch_start / 1e9
                    lateness_s = actual_dispatch_s - command_deadline_s
                    if lateness_s > lateness_limit_s:
                        raise RuntimeError(
                            "realtime callback reported a late dispatch: "
                            f"lateness={lateness_s:.6f}s"
                        )
                    if self.dispatch_observer is not None:
                        self.dispatch_observer(
                            float(row["time_s"]),
                            dispatch_start,
                            dispatch_end,
                        )
                    dispatched += 1
                    continue
                self._wait_until(command_deadline_s)
                lateness_s = self.clock.now_s() - command_deadline_s
                if lateness_s > lateness_limit_s:
                    wait_diag = self._last_wait_diagnostics
                    raise RuntimeError(
                        "command deadline missed; refusing late catch-up dispatch: "
                        f"lateness={lateness_s:.6f}s;"
                        f"max_runtime_gate_ms={self._max_runtime_gate_ns / 1e6:.3f};"
                        f"max_health_snapshot_ms={self._max_health_snapshot_ns / 1e6:.3f};"
                        "fine_entry_remaining_ms="
                        f"{(wait_diag.get('fine_entry_remaining_s') or 0.0) * 1e3:.3f};"
                        f"fine_yields={wait_diag.get('fine_yields', 0)};"
                        f"wake_lateness_ms={wait_diag.get('wake_lateness_s', 0.0) * 1e3:.3f}"
                    )
                runtime_reasons = self._runtime_reasons()
                if runtime_reasons:
                    raise RuntimeError("runtime safety gate: " + ";".join(runtime_reasons))
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
                    dispatch_start = time.perf_counter_ns()
                    self.motion_adapter.send_cartesian_target(pose)
                    if self.dispatch_observer is not None:
                        self.dispatch_observer(float(row["time_s"]), dispatch_start, time.perf_counter_ns())
                    dispatched += 1
            # End continuous supervision before the intentional controller
            # transition out of realtime mode can look like a safety fault.
            self._stop_runtime_watchdog(require_clean=True)
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
            try:
                self._stop_runtime_watchdog(require_clean=False)
            finally:
                try:
                    scheduler_priority.restore(sys.exc_info()[1])
                finally:
                    if callback_old_switch_interval is not None:
                        sys.setswitchinterval(callback_old_switch_interval)
                    self._deadline_yield_once = None
                    with self._stop_lock:
                        self._executing = False


__all__ = ["ExecutionResult", "RokaeMotionExecutor"]
