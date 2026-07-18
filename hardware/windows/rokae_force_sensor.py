"""Force-sensor contract backed by Rokae joint-torque sensing."""

from __future__ import annotations

from collections import deque
import math
import threading
import time

from config import settings


class RokaeForceSensor:
    """Expose xCoreSDK's estimated Cartesian wrench to the project.

    The robot controller derives the wrench from built-in joint torque sensors
    and the configured dynamics/tool model. This is not an external flange
    force/torque sensor. ``set_bias`` therefore applies a session-local software
    offset and never recalibrates the robot hardware.
    """

    def __init__(
        self,
        robot,
        reference_frame: str = settings.ROBOT_FORCE_FRAME,
        sample_hz: int = settings.ROBOT_FORCE_HZ,
        bias_samples: int = settings.ROBOT_FORCE_BIAS_SAMPLES,
        stale_timeout: float = settings.ROBOT_FORCE_STALE_S,
    ):
        if reference_frame not in {"world", "flange", "tool"}:
            raise ValueError("reference_frame must be world, flange, or tool")
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        if bias_samples <= 0:
            raise ValueError("bias_samples must be positive")
        if stale_timeout <= 0:
            raise ValueError("stale_timeout must be positive")

        self.robot = robot
        self.reference_frame = reference_frame
        self.sample_hz = int(sample_hz)
        self.bias_samples = int(bias_samples)
        self.stale_timeout = float(stale_timeout)

        self._condition = threading.Condition()
        self._latest_raw = [0.0] * 6
        self._latest_ts = 0.0
        self._bias = [0.0] * 6
        self._bias_buffer = deque(maxlen=self.bias_samples)
        self._last_error: Exception | None = None
        self._connected = False
        self._running = False
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        """Bind to the already-connected robot; no second network link is made."""
        if not self.robot.is_connected:
            raise ConnectionError("Connect the Rokae robot before its force sensor")
        self._connected = True

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._connected = False

    def start_streaming(self) -> None:
        if not self._connected:
            raise ConnectionError("Rokae force sensor is not connected")
        if self._running:
            return

        with self._condition:
            self._latest_ts = 0.0
            self._last_error = None
            self._bias_buffer.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="rokae-internal-force",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 3.0
        failure: Exception | None = None
        with self._condition:
            while self._latest_ts == 0.0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            if self._latest_ts == 0.0:
                if self._last_error is not None:
                    failure = RuntimeError(
                        f"Unable to read Rokae internal force: {self._last_error}"
                    )
                else:
                    failure = TimeoutError(
                        "Timed out waiting for Rokae internal force data"
                    )
        if failure is not None:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            raise failure

    def _sample_loop(self) -> None:
        period = 1.0 / self.sample_hz
        next_tick = time.perf_counter()
        while self._running:
            try:
                wrench = self.robot.get_end_wrench(self.reference_frame)
                raw = [*wrench["force"], *wrench["torque"]]
                if len(raw) != 6 or not all(math.isfinite(value) for value in raw):
                    raise ValueError("xCoreSDK returned a non-finite wrench")
                with self._condition:
                    self._latest_raw = [float(value) for value in raw]
                    self._latest_ts = float(wrench["ts"])
                    self._bias_buffer.append(list(self._latest_raw))
                    self._last_error = None
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._last_error = exc
                    self._condition.notify_all()

            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            if time.perf_counter() - next_tick > period:
                next_tick = time.perf_counter()

    def set_bias(self) -> None:
        """Average fresh samples and subtract them as a session software bias."""
        if not self._running:
            raise RuntimeError("Start force streaming before setting bias")

        timeout = max(3.0, 3.0 * self.bias_samples / self.sample_hz)
        deadline = time.monotonic() + timeout
        with self._condition:
            self._bias_buffer.clear()
            while len(self._bias_buffer) < self.bias_samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Not enough fresh force samples to set bias")
                self._condition.wait(timeout=remaining)
            samples = list(self._bias_buffer)
            self._bias = [
                sum(sample[index] for sample in samples) / len(samples)
                for index in range(6)
            ]

    def get(self) -> dict[str, float]:
        """Return the latest bias-corrected wrench in N and Nm."""
        with self._condition:
            if self._latest_ts == 0.0:
                if self._last_error is not None:
                    raise RuntimeError(
                        f"Rokae internal force unavailable: {self._last_error}"
                    ) from self._last_error
                raise RuntimeError("No Rokae internal force sample is available")
            age = time.time() - self._latest_ts
            if age > self.stale_timeout:
                detail = f": {self._last_error}" if self._last_error else ""
                raise RuntimeError(
                    f"Rokae internal force data is stale ({age:.3f}s){detail}"
                )
            values = [
                self._latest_raw[index] - self._bias[index] for index in range(6)
            ]
            ts = self._latest_ts

        return {
            "fx": values[0],
            "fy": values[1],
            "fz": values[2],
            "tx": values[3],
            "ty": values[4],
            "tz": values[5],
            "ts": ts,
        }

    def get_force_magnitude(self) -> float:
        data = self.get()
        return math.sqrt(data["fx"] ** 2 + data["fy"] ** 2 + data["fz"] ** 2)

    def get_array(self):
        import numpy as np

        data = self.get()
        return np.array(
            [
                data["fx"],
                data["fy"],
                data["fz"],
                data["tx"],
                data["ty"],
                data["tz"],
            ],
            dtype=float,
        )
