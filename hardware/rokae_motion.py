"""Explicit real-motion boundary for an externally prepared ROKAE robot.

The adapter contains only the SDK methods statically confirmed in xCoreSDK
0.7.0's bundled ``.pyi`` and examples: realtime Cartesian callback control via
``getRtMotionController``, ``setControlLoopCar``, ``startMove``, ``startLoop``,
and callback-command ``setFinished``.  It never powers on or selects automatic
mode. Physical suitability remains unverified until a supervised Windows robot
test.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence


class RokaeCartesianMotionAdapter:
    """Motion-only facade around a connected observation adapter's wrapper."""

    def __init__(self, read_adapter: Any) -> None:
        native = getattr(read_adapter, "native_robot", None)
        if native is None:
            raise TypeError(
                "observation adapter must expose its native_robot integration hook"
            )
        self._robot = native
        self._attached = False
        self._active = False
        self._stop_reason: str | None = None
        self._stop_confirmed = False
        self._stop_lock = threading.Lock()

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stop_reason(self) -> str | None:
        with self._stop_lock:
            return self._stop_reason

    @property
    def stop_confirmed(self) -> bool:
        with self._stop_lock:
            return self._stop_confirmed

    def attach_externally_prepared(self, *, reviewed_filter_hz: float) -> None:
        with self._stop_lock:
            if self._stop_reason is not None:
                raise RuntimeError(
                    f"motion adapter was permanently stopped: {self._stop_reason}"
                )
            if self._attached:
                raise RuntimeError("motion adapter is single-use and already attached")
            self._robot.attach_externally_prepared_realtime(
                reviewed_filter_hz=reviewed_filter_hz
            )
            self._attached = True

    def start_cartesian_hold(self, initial_pose: Sequence[float]) -> None:
        with self._stop_lock:
            if self._stop_reason is not None:
                raise RuntimeError(
                    f"motion adapter was permanently stopped: {self._stop_reason}"
                )
            if not self._attached:
                raise RuntimeError("motion adapter is not attached to external preparation")
            if self._active:
                raise RuntimeError("realtime Cartesian motion is already active")
            self._robot.start_realtime_cartesian(initial_pose)
            self._active = True

    def configure_cartesian_schedule(
        self,
        poses: Sequence[Sequence[float]],
        times_s: Sequence[float],
        *,
        max_lateness_s: float,
    ) -> None:
        with self._stop_lock:
            if self._stop_reason is not None:
                raise RuntimeError(
                    f"motion adapter was permanently stopped: {self._stop_reason}"
                )
            if not self._attached or self._active:
                raise RuntimeError(
                    "motion adapter must be attached and inactive before schedule setup"
                )
            self._robot.configure_realtime_cartesian_schedule(
                poses,
                times_s,
                max_lateness_s=max_lateness_s,
                # Keep the SDK callback and state producer responsive under
                # Windows scheduling jitter. A failed MMCSS registration is
                # treated as a realtime schedule fault rather than silently
                # running without the requested scheduling guarantee.
                use_windows_callback_priority=True,
            )

    def admit_scheduled_target(self, index: int) -> None:
        self._robot.admit_realtime_cartesian_schedule_target(index)

    def wait_scheduled_target(
        self, index: int, *, timeout_s: float
    ) -> tuple[int, int, int]:
        return self._robot.wait_realtime_cartesian_schedule_target(
            index, timeout_s=timeout_s
        )

    def scheduled_origin_s(self) -> float:
        return self._robot.realtime_cartesian_schedule_origin_s()

    def start_scheduled_timing(self, *, timeout_s: float) -> float:
        return self._robot.start_realtime_cartesian_schedule_timing(
            timeout_s=timeout_s
        )

    def send_cartesian_target(self, pose: Sequence[float]) -> None:
        with self._stop_lock:
            if self._stop_reason is not None:
                raise RuntimeError(
                    f"motion adapter was permanently stopped: {self._stop_reason}"
                )
            if not self._active:
                raise RuntimeError("realtime Cartesian motion is not active")
            self._robot.set_realtime_cartesian_target(pose)

    def has_motion_error(self) -> bool:
        return bool(self._robot.realtime_motion_error())

    def request_stop(self, reason: str) -> None:
        """The only upper-layer route to the SDK callback-finish handshake."""
        reason = str(reason).strip()
        if not reason:
            raise ValueError("stop reason must not be empty")
        with self._stop_lock:
            if self._stop_reason is None:
                self._stop_reason = reason
            if self._stop_confirmed:
                return
            # Keep the lock over the native lifecycle call: concurrent callers
            # wait, then either observe success or retry after a failure.
            cancel_schedule = getattr(
                self._robot, "cancel_realtime_cartesian_schedule", None
            )
            if callable(cancel_schedule):
                cancel_schedule(self._stop_reason)
            self._robot.stop_realtime(switch_to_nrt=False)
            self._active = False
            self._stop_confirmed = True


__all__ = ["RokaeCartesianMotionAdapter"]
