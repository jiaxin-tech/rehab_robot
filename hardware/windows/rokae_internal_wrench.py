"""Robot-internal wrench source backed solely by xCoreSDK ``getEndTorque``.

This module intentionally owns no UDP socket, external sensor timestamp, ATI
command, hardware calibration, or external-device watchdog.  Its optional
background thread is only a scheduler for independent robot-controller wrench
queries; collection records the resulting time skew against the realtime state
packet.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
import threading
import time
from typing import Any

from collection.state import (
    InternalWrenchFrame,
    as_float_tuple,
    as_vec3,
    calculate_software_bias,
    merge_invalid_reasons,
    rotate_vector,
    subtract_bias,
    utc_now_iso,
)
from config import settings


class RokaeInternalWrenchSource:
    """Time-stamped xCore internal wrench estimates with software reference bias.

    ``getEndTorque`` is queried in ``world`` by default because xCoreSDK v0.7.0
    does not document ``base`` as a supported input.  When the adapter can read
    ``baseFrame()``, corrected world values are additionally *rotated* to base
    expression.  That is not a full wrench reference-point transform and is
    deliberately labelled ``rotation_only_pending_robot_validation``.
    """

    source_name = "rokae_force_control_get_end_torque"

    def __init__(
        self,
        robot: Any,
        raw_force_frame: str = settings.ROBOT_FORCE_RAW_FRAME,
        sample_hz: int = settings.ROBOT_FORCE_HZ,
        bias_samples: int = settings.ROBOT_FORCE_BIAS_SAMPLES,
        stale_timeout_s: float = settings.ROBOT_FORCE_STALE_S,
    ) -> None:
        if raw_force_frame not in {"world", "flange", "tool"}:
            raise ValueError("raw_force_frame must be world, flange, or tool")
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        if bias_samples <= 0:
            raise ValueError("bias_samples must be positive")
        if stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be positive")

        self.robot = robot
        self.raw_force_frame = raw_force_frame
        self.sample_hz = int(sample_hz)
        self.bias_samples = int(bias_samples)
        self.stale_timeout_s = float(stale_timeout_s)

        self._condition = threading.Condition()
        self._connected = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._sequence_id = 0
        self._latest: InternalWrenchFrame | None = None
        self._last_error: str | None = None
        self._bias: tuple[float, ...] | None = None
        self._bias_history: deque[tuple[float, ...]] = deque(maxlen=self.bias_samples)
        self._world_to_base_rotation: tuple[tuple[float, float, float], ...] | None = None
        self._base_transform_error: str | None = None

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    @property
    def bias(self) -> tuple[float, ...] | None:
        with self._condition:
            return tuple(self._bias) if self._bias is not None else None

    def connect(self) -> None:
        """Attach to an already-connected robot; no second device is opened."""
        if not bool(getattr(self.robot, "is_connected", False)):
            raise ConnectionError("Connect the Rokae robot before starting wrench queries")
        self._connected = True
        self._world_to_base_rotation = None
        self._base_transform_error = None

        if self.raw_force_frame == "world":
            try:
                rotation = self.robot.get_world_to_base_rotation()
                if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
                    raise ValueError("world-to-base rotation is not 3x3")
                self._world_to_base_rotation = tuple(
                    tuple(float(value) for value in row) for row in rotation
                )
            except Exception as exc:
                # Raw world data remain useful for diagnostics, but must not be
                # silently sent to base-frame algorithms.
                self._base_transform_error = (
                    f"base_rotation_unavailable:{type(exc).__name__}:{exc}"
                )

    def disconnect(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._connected = False

    def start_streaming(self) -> None:
        if not self._connected:
            raise ConnectionError("Connect the robot before starting wrench updates")
        if self._running:
            return
        with self._condition:
            self._last_error = None
            self._latest = None
            # A reference bias belongs to one connected collection session.
            # Never carry it across stop/reconnect and accidentally call a new
            # patient's raw wrench "corrected".
            self._bias = None
            self._bias_history.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="rokae-internal-wrench",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 3.0
        with self._condition:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            latest = self._latest
            error = self._last_error
        if latest is None or not latest.valid:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            if error:
                raise RuntimeError(f"Unable to read Rokae internal wrench: {error}")
            raise TimeoutError("Timed out waiting for the first Rokae wrench sample")

    def _sample_loop(self) -> None:
        period_s = 1.0 / self.sample_hz
        next_tick = time.monotonic()
        while self._running:
            try:
                raw = self.robot.get_end_wrench(self.raw_force_frame)
                frame = self._frame_from_raw(raw)
                with self._condition:
                    self._latest = frame
                    self._last_error = None
                    raw_wrench = (
                        *(frame.cartesian_force_raw_n or ()),
                        *(frame.cartesian_torque_raw_nm or ()),
                    )
                    if len(raw_wrench) == 6:
                        self._bias_history.append(tuple(raw_wrench))
                    self._condition.notify_all()
            except Exception as exc:
                self._publish_failure(exc)

            next_tick += period_s
            delay_s = next_tick - time.monotonic()
            if delay_s > 0:
                time.sleep(delay_s)
            elif -delay_s > period_s:
                # Do not try to catch up with a query burst; preserve a bounded
                # polling interval and let the CSV timing show actual skew.
                next_tick = time.monotonic()

    def _publish_failure(self, exc: BaseException) -> None:
        now_s = time.monotonic()
        error = f"robot_wrench_query_error:{type(exc).__name__}:{exc}"
        with self._condition:
            self._sequence_id += 1
            self._last_error = error
            self._latest = InternalWrenchFrame(
                sequence_id=self._sequence_id,
                host_monotonic_time_s=now_s,
                wall_time_iso=utc_now_iso(),
                valid=False,
                invalid_reason=error,
                source=self.source_name,
                joint_measured_torque_nm=None,
                joint_external_torque_nm=None,
                cartesian_force_raw_n=None,
                cartesian_torque_raw_nm=None,
                raw_force_frame=self.raw_force_frame,
                cartesian_force_bias_n=None,
                cartesian_torque_bias_nm=None,
                cartesian_force_corrected_n=None,
                cartesian_torque_corrected_nm=None,
                cartesian_force_base_n=None,
                cartesian_torque_base_nm=None,
                base_transform_kind="unavailable",
                force_time_s=None,
                torque_time_s=None,
            )
            self._condition.notify_all()

    def _frame_from_raw(self, raw: dict[str, Any]) -> InternalWrenchFrame:
        force = raw.get("cartesian_force_raw_n", raw.get("force"))
        torque = raw.get("cartesian_torque_raw_nm", raw.get("torque"))
        measured = raw.get("joint_measured_torque_nm", raw.get("joint_torque_measured"))
        external = raw.get("joint_external_torque_nm", raw.get("joint_torque_external"))
        raw_frame = str(raw.get("raw_force_frame", raw.get("reference_frame", self.raw_force_frame))).lower()
        now_s = float(raw.get("host_monotonic_time_s", raw.get("ts", time.monotonic())))
        if not math.isfinite(now_s):
            raise ValueError("Wrench query did not provide a finite host timestamp")

        def optional_time(name: str) -> float | None:
            value = raw.get(name)
            if value is None:
                return None
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None

        query_started_s = optional_time("force_query_started_s")
        query_finished_s = optional_time("force_query_finished_s")
        if (
            query_started_s is not None
            and query_finished_s is not None
            and query_finished_s < query_started_s
        ):
            raise ValueError("getEndTorque query finished before it started")

        force_v = as_vec3(force)
        torque_v = as_vec3(torque)
        measured_v = as_float_tuple(measured, 6)
        external_v = as_float_tuple(external, 6)
        valid = (
            raw_frame == self.raw_force_frame
            and force_v is not None
            and torque_v is not None
            and measured_v is not None
            and external_v is not None
        )
        reason = "" if valid else "incomplete_or_invalid_get_end_torque_result"

        with self._condition:
            bias = tuple(self._bias) if self._bias is not None else None
        raw_wrench = (*force_v, *torque_v) if force_v is not None and torque_v is not None else None
        corrected = subtract_bias(raw_wrench, bias)
        force_bias = as_vec3(bias[:3]) if bias is not None else None
        torque_bias = as_vec3(bias[3:]) if bias is not None else None
        corrected_force = as_vec3(corrected[:3]) if corrected is not None else None
        corrected_torque = as_vec3(corrected[3:]) if corrected is not None else None

        base_force = None
        base_torque = None
        transform_kind = "unavailable"
        if corrected_force is not None and corrected_torque is not None:
            if raw_frame == "base":
                base_force, base_torque = corrected_force, corrected_torque
                transform_kind = "sdk_base"
            elif raw_frame == "world" and self._world_to_base_rotation is not None:
                base_force = rotate_vector(self._world_to_base_rotation, corrected_force)
                base_torque = rotate_vector(self._world_to_base_rotation, corrected_torque)
                transform_kind = (
                    "rotation_only_verified_by_project_procedure"
                    if settings.BASE_WRENCH_ROTATION_VERIFIED
                    else "rotation_only_pending_robot_validation"
                )
            elif self._base_transform_error:
                reason = merge_invalid_reasons(reason, self._base_transform_error)

        with self._condition:
            self._sequence_id += 1
            sequence_id = self._sequence_id
        return InternalWrenchFrame(
            sequence_id=sequence_id,
            host_monotonic_time_s=now_s,
            wall_time_iso=str(raw.get("wall_time_iso", utc_now_iso())),
            valid=valid,
            invalid_reason=reason,
            source=self.source_name,
            joint_measured_torque_nm=measured_v,
            joint_external_torque_nm=external_v,
            cartesian_force_raw_n=force_v,
            cartesian_torque_raw_nm=torque_v,
            raw_force_frame=raw_frame,
            cartesian_force_bias_n=force_bias,
            cartesian_torque_bias_nm=torque_bias,
            cartesian_force_corrected_n=corrected_force,
            cartesian_torque_corrected_nm=corrected_torque,
            cartesian_force_base_n=base_force,
            cartesian_torque_base_nm=base_torque,
            base_transform_kind=transform_kind,
            force_time_s=now_s,
            torque_time_s=now_s,
            force_query_started_s=query_started_s,
            force_query_finished_s=query_finished_s,
        )

    def set_bias(self, duration_s: float | None = None) -> tuple[float, ...]:
        """Compute a session-local software reference bias from fresh raw samples.

        This does not call ``calibrateForceSensor`` and never changes controller
        calibration.  The caller must first ensure the configured tool/load,
        idle state, low TCP speed, and no human contact.
        """
        if not self._running:
            raise RuntimeError("Start wrench streaming before software bias")
        required = max(
            self.bias_samples,
            int(math.ceil((duration_s or 0.0) * self.sample_hz)),
        )
        deadline = time.monotonic() + max(3.0, 3.0 * required / self.sample_hz)
        with self._condition:
            self._bias_history = deque(maxlen=required)
            while len(self._bias_history) < required:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Not enough fresh internal wrench samples for bias")
                self._condition.wait(timeout=remaining)
            bias = calculate_software_bias(list(self._bias_history))
            previous_sequence_id = self._latest.sequence_id if self._latest is not None else -1
            self._bias = bias
            # Wait for one post-bias controller query so callers never receive
            # an older frame whose corrected fields are still absent.
            refresh_deadline = time.monotonic() + max(1.0, 3.0 / self.sample_hz)
            while (
                self._latest is None
                or self._latest.sequence_id <= previous_sequence_id
                or self._latest.cartesian_force_corrected_n is None
            ):
                remaining = refresh_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No post-bias internal wrench sample arrived")
                self._condition.wait(timeout=remaining)
        return bias

    def snapshot(self, now_s: float | None = None) -> InternalWrenchFrame:
        """Return the latest controller wrench, invalidating stale data safely."""
        current_s = time.monotonic() if now_s is None else float(now_s)
        with self._condition:
            latest = self._latest
            last_error = self._last_error
        if latest is None:
            return InternalWrenchFrame(
                sequence_id=0,
                host_monotonic_time_s=None,
                wall_time_iso=None,
                valid=False,
                invalid_reason=last_error or "robot_wrench_not_ready",
                source=self.source_name,
                joint_measured_torque_nm=None,
                joint_external_torque_nm=None,
                cartesian_force_raw_n=None,
                cartesian_torque_raw_nm=None,
                raw_force_frame=self.raw_force_frame,
                cartesian_force_bias_n=None,
                cartesian_torque_bias_nm=None,
                cartesian_force_corrected_n=None,
                cartesian_torque_corrected_nm=None,
                cartesian_force_base_n=None,
                cartesian_torque_base_nm=None,
                base_transform_kind="unavailable",
                force_time_s=None,
                torque_time_s=None,
            )
        if latest.host_monotonic_time_s is None:
            return replace(latest, valid=False, invalid_reason=merge_invalid_reasons(latest.invalid_reason, "robot_wrench_missing_time"))
        age_s = current_s - latest.host_monotonic_time_s
        if age_s > self.stale_timeout_s:
            return replace(
                latest,
                valid=False,
                invalid_reason=merge_invalid_reasons(
                    latest.invalid_reason,
                    f"robot_wrench_stale:{age_s:.6f}s",
                ),
            )
        return latest

    def get(self) -> dict[str, Any]:
        """Compatibility view; new code should use :meth:`snapshot` directly."""
        frame = self.snapshot()
        if not frame.valid:
            raise RuntimeError(f"Rokae internal wrench unavailable: {frame.invalid_reason}")
        if frame.cartesian_force_corrected_n is None or frame.cartesian_torque_corrected_nm is None:
            raise RuntimeError("Software bias has not been established for this episode")
        return {
            "fx": frame.cartesian_force_corrected_n[0],
            "fy": frame.cartesian_force_corrected_n[1],
            "fz": frame.cartesian_force_corrected_n[2],
            "tx": frame.cartesian_torque_corrected_nm[0],
            "ty": frame.cartesian_torque_corrected_nm[1],
            "tz": frame.cartesian_torque_corrected_nm[2],
            "ts": frame.host_monotonic_time_s,
            "frame": frame.raw_force_frame,
            "snapshot": frame,
        }

    def get_force_magnitude(self) -> float:
        frame = self.snapshot()
        if not frame.valid or frame.cartesian_force_base_n is None:
            raise RuntimeError(
                "A valid base-frame corrected wrench is required for force safety: "
                f"{frame.invalid_reason or frame.base_transform_kind}"
            )
        return math.sqrt(sum(value * value for value in frame.cartesian_force_base_n))

    def get_array(self):
        import numpy as np

        frame = self.snapshot()
        if not frame.valid or frame.cartesian_force_base_n is None or frame.cartesian_torque_base_nm is None:
            raise RuntimeError("No valid base-frame corrected wrench is available")
        return np.asarray(
            [*frame.cartesian_force_base_n, *frame.cartesian_torque_base_nm],
            dtype=float,
        )
