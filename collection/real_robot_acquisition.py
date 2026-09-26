"""State/alignment and wrench IPC consumers for real episodes.

Native wrench queries belong exclusively to an explicitly supplied spawned
provider. No parent native-wrench fallback is permitted. State/RT remains in
the parent; its native scheduling and controller isolation remain unvalidated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any

from collection.episode_logger import EpisodeLogger, EpisodeLoggerError
from collection.state import KinematicStateFrame
from hardware.rokae_adapter import RobotWrenchFrame
from utils.clock import MonotonicClock, SYSTEM_CLOCK


def _component(values: tuple[float, ...] | None, index: int) -> float | None:
    return None if values is None or len(values) <= index else float(values[index])


def _age_s(now_s: float, sample_s: float | None) -> float | None:
    if sample_s is None or not math.isfinite(sample_s):
        return None
    age = now_s - sample_s
    return age if age >= 0.0 and math.isfinite(age) else None


@dataclass(frozen=True)
class AcquisitionHealth:
    host_time_s: float
    state_time_s: float | None
    wrench_time_s: float | None
    state_age_s: float | None
    wrench_age_s: float | None
    state_wrench_skew_s: float | None
    state_valid: bool
    wrench_valid: bool
    state_thread_alive: bool
    wrench_thread_alive: bool
    alignment_thread_alive: bool
    query_duration_ms: float | None
    force_magnitude_n: float | None
    torque_magnitude_nm: float | None
    valid: bool
    invalid_reason: str
    wrench_process_alive: bool = False
    wrench_stream_healthy: bool = False
    wrench_query_state: str = "NOT_STARTED"
    wrench_failure_latch: tuple[str, ...] = ()


class RealRobotAcquisition:
    """Lifecycle and latest-snapshot cache for one connected episode."""

    def __init__(
        self,
        adapter: Any,
        logger: EpisodeLogger,
        *,
        state_poll_hz: float = 250.0,
        wrench_hz: float = 50.0,
        alignment_hz: float = 50.0,
        join_timeout_s: float = 1.0,
        clock: MonotonicClock = SYSTEM_CLOCK,
        thread_factory: Any | None = None,
        wrench_provider: Any | None = None,
        diagnostic_state_only: bool = False,
    ) -> None:
        for name, value in (
            ("state_poll_hz", state_poll_hz),
            ("wrench_hz", wrench_hz),
            ("alignment_hz", alignment_hz),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.adapter = adapter
        self.wrench_provider = wrench_provider
        self.diagnostic_state_only = diagnostic_state_only
        if wrench_provider is not None and wrench_provider.config["rate_hz"] != wrench_hz:
            raise ValueError("provider rate must match explicit acquisition wrench_hz")
        self.logger = logger
        self.state_poll_hz = float(state_poll_hz)
        self.wrench_hz = float(wrench_hz)
        self.alignment_hz = float(alignment_hz)
        if not math.isfinite(join_timeout_s) or join_timeout_s <= 0.0:
            raise ValueError("join_timeout_s must be finite and positive")
        self.join_timeout_s = float(join_timeout_s)
        self.clock = clock
        self._thread_factory = thread_factory or threading.Thread
        if not callable(self._thread_factory):
            raise TypeError("thread_factory must be callable")
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._wrench_lock = threading.Lock()
        self._latest_state: KinematicStateFrame | None = None
        self._latest_wrench: RobotWrenchFrame | None = None
        self._state_error: str | None = None
        self._wrench_error: str | None = None
        self._background_error: str | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._started_threads: dict[str, threading.Thread] = {}
        self._started = False
        self._manage_connection = True

    @property
    def running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    @property
    def background_error(self) -> str | None:
        return self._background_error

    @property
    def live_producer_names(self) -> tuple[str, ...]:
        """Name producer threads that still own episode/SDK resources."""

        return tuple(
            sorted(name for name, thread in self._threads.items() if thread.is_alive())
        )

    def _thread_alive(self, name: str) -> bool:
        thread = self._threads.get(name)
        return bool(thread is not None and thread.is_alive())

    def start(self, *, manage_connection: bool = True) -> "RealRobotAcquisition":
        """Start observation producers after the five-file logger is ready."""
        if self._started:
            raise RuntimeError("real robot acquisition is already started")
        if self.wrench_provider is None and not self.diagnostic_state_only:
            raise PermissionError("explicit isolated wrench provider and reviewed budgets required before connection")
        if self.wrench_provider is not None and self.wrench_provider.config["mode"] == "live":
            native = self.adapter.native_robot
            for child_key, parent_key in (("robot_ip", "ip_address"), ("local_ip", "local_ip"), ("robot_class", "robot_class")):
                if self.wrench_provider.config[child_key] != getattr(native, parent_key, None):
                    raise PermissionError("parent_child_connection_binding_mismatch:" + child_key)
        self.logger.assert_healthy()
        self._manage_connection = bool(manage_connection)
        self._started_threads = {}
        try:
            if manage_connection:
                self.adapter.connect()
            connected = self.adapter.is_connected()
            if not connected:
                raise ConnectionError("ROKAE adapter connection was not confirmed")
            if self.wrench_provider is not None and self.wrench_provider.config["mode"] == "live":
                metadata = self.adapter.read_robot_metadata()
                expected = self.wrench_provider.config["expected_identity"]
                for key, parent_key in (("robot_model", "robot_model"), ("robot_serial", "robot_serial_number"),
                                        ("controller_version", "controller_version"), ("sdk_version", "sdk_version")):
                    if str(metadata.get(parent_key)) != expected[key]:
                        raise PermissionError("parent_child_identity_binding_mismatch:" + key)
            self.adapter.start_state_stream()
            if self.wrench_provider is not None:
                self.wrench_provider.start()
            self._stop_event.clear()
            self._threads = {
                "state": self._thread_factory(
                    target=self._state_loop,
                    name="real-episode-state",
                    daemon=True,
                ),
                "wrench": self._thread_factory(
                    target=self._wrench_loop,
                    name="real-episode-wrench",
                    daemon=True,
                ),
                "alignment": self._thread_factory(
                    target=self._alignment_loop,
                    name="real-episode-alignment",
                    daemon=True,
                ),
            }
            self._started = True
            for name, thread in self._threads.items():
                try:
                    thread.start()
                except BaseException:
                    # A custom/native thread wrapper may report a start error
                    # after spawning.  Track it whenever it is in fact alive.
                    if thread.is_alive():
                        self._started_threads[name] = thread
                    raise
                self._started_threads[name] = thread
            return self
        except Exception as exc:
            self._stop_event.set()
            for thread in self._started_threads.values():
                if thread is not threading.current_thread():
                    thread.join(timeout=self.join_timeout_s)
            still_alive = sorted(
                name
                for name, thread in self._started_threads.items()
                if thread.is_alive()
            )
            start_reason = f"acquisition_start:{type(exc).__name__}:{exc}"
            self._started = False
            if still_alive:
                if self.wrench_provider is not None and self.wrench_provider.process is not None:
                    self.wrench_provider.stop()
                reason = (
                    start_reason
                    + ";acquisition_threads_did_not_stop:"
                    + ",".join(still_alive)
                )
                self._background_error = reason
                self.logger.signal_failure(reason, stream="acquisition_start")
                raise RuntimeError(
                    reason
                    + "; refusing SDK cleanup/disconnect while a producer "
                    "may still own native resources"
                ) from exc
            self._background_error = start_reason
            self.logger.signal_failure(start_reason, stream="acquisition_start")
            try:
                self.logger.mark_failed(start_reason)
            except EpisodeLoggerError:
                pass
            if self.wrench_provider is not None and self.wrench_provider.process is not None:
                self.wrench_provider.stop()
            self._cleanup_adapter()
            raise

    def stop(self) -> None:
        """Stop producers without racing disconnect against a blocked SDK call.

        xCoreSDK exposes no timeout/cancellation argument for ``getEndTorque``.
        If any producer fails to join, this method deliberately leaves the SDK
        session untouched and raises for supervised/manual recovery.
        """
        self._stop_event.set()
        for thread in self._started_threads.values():
            if thread is not threading.current_thread():
                thread.join(timeout=self.join_timeout_s)
        still_alive = list(self.live_producer_names)
        if still_alive:
            if self.wrench_provider is not None and self.wrench_provider.process is not None:
                self.wrench_provider.stop()
            reason = "acquisition_threads_did_not_stop:" + ",".join(sorted(still_alive))
            self._background_error = reason
            # Do not inspect logger.healthy or call mark_failed here: either can
            # wait on the logger I/O lock held by the producer we just failed to
            # join.  This path must publish only the lock-free signal and leave
            # both SDK cleanup and metadata untouched.
            self.logger.signal_failure(reason, stream="acquisition_stop")
            self._started = False
            raise RuntimeError(
                reason
                + "; refusing SDK disconnect while a native query may still be active"
            )
        try:
            if self.wrench_provider is not None:
                self.wrench_provider.stop()
            self._cleanup_adapter()
        except Exception as exc:
            reason = f"acquisition_cleanup:{type(exc).__name__}:{exc}"
            self._background_error = reason
            self.logger.signal_failure(reason, stream="acquisition_cleanup")
            try:
                self.logger.mark_failed(reason)
            except EpisodeLoggerError:
                pass
            raise
        finally:
            self._started = False

    def _cleanup_adapter(self) -> None:
        try:
            self.adapter.stop_state_stream()
        finally:
            if self._manage_connection:
                self.adapter.disconnect()

    def _sleep_until(self, deadline_s: float) -> float:
        remaining = deadline_s - self.clock.now_s()
        if remaining > 0.0:
            self._stop_event.wait(remaining)
        return self.clock.now_s()

    def _background_failure(self, producer: str, exc: BaseException) -> None:
        reason = f"{producer}:{type(exc).__name__}:{exc}"
        self._background_error = reason
        self._stop_event.set()
        self.logger.signal_failure(reason, stream=producer)

    def _state_loop(self) -> None:
        period = 1.0 / self.state_poll_hz
        next_tick = self.clock.now_s()
        previous_sequence: int | None = None
        while not self._stop_event.is_set():
            try:
                frame = self.adapter.read_state_frame()
                if not isinstance(frame, KinematicStateFrame):
                    raise TypeError("read_state_frame must return KinematicStateFrame")
                with self._state_lock:
                    self._latest_state = frame
                    self._state_error = None
                if frame.sequence_id != previous_sequence:
                    self.logger.append_robot_state(
                        host_time_s=frame.host_monotonic_time_s,
                        **{
                            f"q{index + 1}": _component(frame.joint_position_rad, index)
                            for index in range(6)
                        },
                        tcp_x=_component(frame.tcp_position_m, 0),
                        tcp_y=_component(frame.tcp_position_m, 1),
                        tcp_z=_component(frame.tcp_position_m, 2),
                        tcp_rx=_component(frame.tcp_orientation_rad, 0),
                        tcp_ry=_component(frame.tcp_orientation_rad, 1),
                        tcp_rz=_component(frame.tcp_orientation_rad, 2),
                        valid=frame.valid,
                        invalid_reason=frame.invalid_reason,
                    )
                    previous_sequence = frame.sequence_id
            except EpisodeLoggerError as exc:
                self._background_failure("state_logger", exc)
                return
            except Exception as exc:
                with self._state_lock:
                    self._state_error = f"state_read:{type(exc).__name__}:{exc}"
            next_tick += period
            now_s = self._sleep_until(next_tick)
            if now_s - next_tick > period:
                next_tick = now_s

    def _wrench_loop(self) -> None:
        # IPC service frequency is independent of the native query rate.
        period = min(0.002, 1.0 / self.wrench_hz)
        next_tick = self.clock.now_s()
        previous_sequence = None
        while not self._stop_event.is_set():
            try:
                if self.wrench_provider is None:
                    self._stop_event.wait(period)
                    continue
                process_health = self.wrench_provider.poll()
                frame = self.wrench_provider.last_good
                with self._wrench_lock:
                    self._wrench_error = ";".join(process_health["FAILURE_LATCH"]) or None
                if frame is None or frame.sequence_id == previous_sequence:
                    self._stop_event.wait(period)
                    continue
                if not isinstance(frame, RobotWrenchFrame):
                    raise TypeError("read_internal_wrench must return RobotWrenchFrame")
                with self._wrench_lock:
                    self._latest_wrench = frame
                previous_sequence = frame.sequence_id
                self.logger.append_robot_wrench(
                    query_start_s=frame.host_query_start_s,
                    query_end_s=frame.host_query_end_s,
                    publish_time_s=frame.host_publish_s,
                    **{
                        f"joint_measured_torque_{index + 1}": _component(
                            frame.joint_measured_torque_nm, index
                        )
                        for index in range(6)
                    },
                    **{
                        f"joint_external_torque_{index + 1}": _component(
                            frame.joint_external_torque_nm, index
                        )
                        for index in range(6)
                    },
                    fx=_component(frame.cartesian_force_raw_n, 0),
                    fy=_component(frame.cartesian_force_raw_n, 1),
                    fz=_component(frame.cartesian_force_raw_n, 2),
                    tx=_component(frame.cartesian_torque_raw_nm, 0),
                    ty=_component(frame.cartesian_torque_raw_nm, 1),
                    tz=_component(frame.cartesian_torque_raw_nm, 2),
                    frame_type=frame.raw_force_frame,
                    query_duration_ms=frame.query_duration_ms,
                    valid=frame.valid,
                    invalid_reason=frame.invalid_reason,
                )
            except EpisodeLoggerError as exc:
                self._background_failure("wrench_logger", exc)
                return
            except Exception as exc:
                with self._wrench_lock:
                    self._wrench_error = f"wrench_read:{type(exc).__name__}:{exc}"
                if self.wrench_provider is not None:
                    self.wrench_provider.fail(self._wrench_error)
            next_tick += period
            now_s = self._sleep_until(next_tick)
            if now_s - next_tick > period:
                next_tick = now_s

    def latest_health(self) -> AcquisitionHealth:
        now_s = self.clock.now_s()
        with self._state_lock:
            state = self._latest_state
            state_error = self._state_error
        with self._wrench_lock:
            wrench = self._latest_wrench
            wrench_error = self._wrench_error
        state_time = state.host_monotonic_time_s if state is not None else None
        wrench_time = wrench.host_monotonic_time_s if wrench is not None else None
        state_age = _age_s(now_s, state_time)
        wrench_age = _age_s(now_s, wrench_time)
        skew = (
            abs(state_time - wrench_time)
            if state_time is not None and wrench_time is not None
            else None
        )
        state_valid = bool(state is not None and state.valid and state_age is not None)
        wrench_valid = bool(wrench is not None and wrench.valid and wrench_age is not None)
        state_alive = self._thread_alive("state")
        wrench_alive = self._thread_alive("wrench")
        alignment_alive = self._thread_alive("alignment")
        reasons = []
        # This second supervisor entry can detect failure even if the raw-CSV
        # consumer is held up in logger I/O. poll never waits on child locks/I/O.
        process_health = self.wrench_provider.poll() if self.wrench_provider is not None else {}
        stream_healthy = bool(process_health.get("STREAM_HEALTH"))
        if not stream_healthy:
            reasons.append(wrench_error or ";".join(process_health.get("FAILURE_LATCH", ())) or
                           ("DISABLED_FOR_STATIONARY_A" if self.diagnostic_state_only else "wrench_process_not_ready"))
        if self.wrench_provider is not None:
            limits = self.wrench_provider.budgets
            if state_age is None or state_age > limits.state_age_s:
                reasons.append("state_stale_or_unavailable")
                if state_age is not None:
                    self.wrench_provider.fail("state_freshness_failure")
            if wrench_age is None or wrench_age > limits.sample_age_s:
                stream_healthy = False
                reasons.append("wrench_stale_or_unavailable")
            if skew is None or skew > limits.skew_s:
                reasons.append("state_wrench_skew_unavailable_or_exceeded")
                if skew is not None:
                    self.wrench_provider.fail("state_wrench_skew_failure")
        if not state_valid:
            reasons.append(state_error or (state.invalid_reason if state else "state_not_ready"))
        if not wrench_valid:
            reasons.append(wrench_error or (wrench.invalid_reason if wrench else "wrench_not_ready"))
        if not state_alive:
            reasons.append("state_thread_not_alive")
        if not wrench_alive:
            reasons.append("wrench_thread_not_alive")
        if not alignment_alive:
            reasons.append("alignment_thread_not_alive")
        unique_reasons = tuple(dict.fromkeys(reason for reason in reasons if reason))
        force_magnitude = (
            math.sqrt(sum(value * value for value in wrench.cartesian_force_raw_n))
            if wrench is not None and wrench.cartesian_force_raw_n is not None
            else None
        )
        torque_magnitude = (
            math.sqrt(sum(value * value for value in wrench.cartesian_torque_raw_nm))
            if wrench is not None and wrench.cartesian_torque_raw_nm is not None
            else None
        )
        return AcquisitionHealth(
            host_time_s=now_s,
            state_time_s=state_time,
            wrench_time_s=wrench_time,
            state_age_s=state_age,
            wrench_age_s=wrench_age,
            state_wrench_skew_s=skew,
            state_valid=state_valid,
            wrench_valid=wrench_valid,
            state_thread_alive=state_alive,
            wrench_thread_alive=wrench_alive,
            alignment_thread_alive=alignment_alive,
            query_duration_ms=wrench.query_duration_ms if wrench else None,
            force_magnitude_n=force_magnitude,
            torque_magnitude_nm=torque_magnitude,
            valid=bool(state_valid and wrench_valid and state_alive and wrench_alive and alignment_alive and stream_healthy and not reasons),
            invalid_reason=";".join(unique_reasons),
            wrench_process_alive=bool(process_health.get("PROCESS_ALIVE")),
            wrench_stream_healthy=stream_healthy,
            wrench_query_state=process_health.get("CURRENT_QUERY_STATE", "NOT_STARTED"),
            wrench_failure_latch=tuple(process_health.get("FAILURE_LATCH", ())),
        )

    def latest_state_frame(self) -> KinematicStateFrame | None:
        """Return the immutable latest cached state without an SDK query."""
        with self._state_lock:
            return self._latest_state

    def _alignment_loop(self) -> None:
        period = 1.0 / self.alignment_hz
        next_tick = self.clock.now_s()
        while not self._stop_event.is_set():
            try:
                health = self.latest_health()
                self.logger.append_aligned_snapshot(
                    host_time_s=health.host_time_s,
                    state_time_s=health.state_time_s,
                    wrench_time_s=health.wrench_time_s,
                    state_age_s=health.state_age_s,
                    wrench_age_s=health.wrench_age_s,
                    state_wrench_skew_s=health.state_wrench_skew_s,
                    state_valid=health.state_valid,
                    wrench_valid=health.wrench_valid,
                    state_thread_alive=health.state_thread_alive,
                    wrench_thread_alive=health.wrench_thread_alive,
                    query_duration_ms=health.query_duration_ms,
                    valid=health.valid,
                    invalid_reason=health.invalid_reason,
                )
            except EpisodeLoggerError as exc:
                self._background_failure("alignment_logger", exc)
                return
            next_tick += period
            now_s = self._sleep_until(next_tick)
            if now_s - next_tick > period:
                next_tick = now_s

    def wait_until_healthy(self, timeout_s: float = 3.0) -> bool:
        deadline = self.clock.now_s() + timeout_s
        while self.clock.now_s() < deadline and not self._stop_event.is_set():
            if self.latest_health().valid:
                return True
            time.sleep(0.005)
        return False


__all__ = ["AcquisitionHealth", "RealRobotAcquisition"]
