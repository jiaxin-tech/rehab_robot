"""Diagnostic-only non-blocking latest-wrench cache and worker.

This is deliberately not part of ``hardware`` or ``collection``.  The main
loop only calls :meth:`snapshot`; the worker is the sole caller of the supplied
wrench query function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class WrenchCacheSnapshot:
    value: Any
    host_timestamp_ns: int | None
    sequence_id: int
    age_ms: float | None
    valid: bool
    stale: bool
    source_alive: bool
    query_in_flight: bool
    in_flight_age_ms: float | None
    last_success_timestamp_ns: int | None
    last_error: str | None
    last_error_code: int | None
    last_error_timestamp_ns: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LatestWrenchCache:
    def __init__(self, *, stale_threshold_ms: float) -> None:
        if stale_threshold_ms <= 0:
            raise ValueError("stale_threshold_ms must be positive")
        self.stale_threshold_ms = float(stale_threshold_ms)
        self._lock = threading.Lock()
        self._snapshot = WrenchCacheSnapshot(
            value=None,
            host_timestamp_ns=None,
            sequence_id=0,
            age_ms=None,
            valid=False,
            stale=True,
            source_alive=False,
            query_in_flight=False,
            in_flight_age_ms=None,
            last_success_timestamp_ns=None,
            last_error=None,
            last_error_code=None,
            last_error_timestamp_ns=None,
        )
        self._in_flight_started_ns: int | None = None

    def set_source_alive(self, alive: bool) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, source_alive=bool(alive))

    def begin_query(self, started_ns: int) -> None:
        with self._lock:
            self._in_flight_started_ns = int(started_ns)
            self._snapshot = replace(self._snapshot, query_in_flight=True)

    def publish_success(self, value: Any, timestamp_ns: int) -> int:
        with self._lock:
            sequence_id = self._snapshot.sequence_id + 1
            self._in_flight_started_ns = None
            self._snapshot = replace(
                self._snapshot,
                value=value,
                host_timestamp_ns=int(timestamp_ns),
                sequence_id=sequence_id,
                valid=True,
                stale=False,
                query_in_flight=False,
                last_success_timestamp_ns=int(timestamp_ns),
            )
            return sequence_id

    def publish_error(self, error: BaseException, timestamp_ns: int) -> int:
        message = f"{type(error).__name__}:{error}"
        match = re.search(r"\((\d+)\)", str(error))
        code = int(match.group(1)) if match else None
        with self._lock:
            sequence_id = self._snapshot.sequence_id + 1
            self._in_flight_started_ns = None
            self._snapshot = replace(
                self._snapshot,
                sequence_id=sequence_id,
                valid=False,
                query_in_flight=False,
                last_error=message,
                last_error_code=code,
                last_error_timestamp_ns=int(timestamp_ns),
            )
            return sequence_id

    def snapshot(self, now_ns: int | None = None) -> WrenchCacheSnapshot:
        current_ns = time.perf_counter_ns() if now_ns is None else int(now_ns)
        with self._lock:
            base = self._snapshot
            in_flight_started = self._in_flight_started_ns
        age_ms = (
            None
            if base.host_timestamp_ns is None
            else max(0.0, (current_ns - base.host_timestamp_ns) / 1e6)
        )
        in_flight_age_ms = (
            None
            if in_flight_started is None
            else max(0.0, (current_ns - in_flight_started) / 1e6)
        )
        stale = age_ms is None or age_ms > self.stale_threshold_ms
        return replace(
            base,
            age_ms=age_ms,
            stale=stale,
            valid=bool(base.valid and not stale),
            in_flight_age_ms=in_flight_age_ms,
        )


class NonBlockingWrenchWorker:
    """Single-owner query worker; no cancellation claim is made for native calls."""

    def __init__(
        self,
        query: Callable[[], Any],
        *,
        target_hz: float,
        stale_threshold_ms: float,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        state_snapshot: Callable[[], dict[str, Any]] | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if target_hz <= 0:
            raise ValueError("target_hz must be positive")
        self.query = query
        self.target_hz = float(target_hz)
        self.period_ns = int(1e9 / self.target_hz)
        self.cache = LatestWrenchCache(stale_threshold_ms=stale_threshold_ms)
        self.on_result = on_result
        self.state_snapshot = state_snapshot
        self.clock_ns = clock_ns
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._call_sequence = 0

    @property
    def alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        if self.alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="diagnostic-wrench-worker", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> bool:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return not self.alive

    def _state(self) -> dict[str, Any]:
        if self.state_snapshot is None:
            return {}
        try:
            return dict(self.state_snapshot())
        except Exception as exc:
            return {"state_snapshot_error": f"{type(exc).__name__}:{exc}"}

    def _run(self) -> None:
        self.cache.set_source_alive(True)
        next_tick_ns = self.clock_ns()
        previous_success_latency_ms: float | None = None
        last_success_ns: int | None = None
        try:
            while not self._stop.is_set():
                scheduled_ns = next_tick_ns
                started_ns = self.clock_ns()
                state_before = self._state()
                self.cache.begin_query(started_ns)
                self._call_sequence += 1
                success = False
                value: Any = None
                error: BaseException | None = None
                try:
                    value = self.query()
                    success = True
                except BaseException as exc:
                    error = exc
                finished_ns = self.clock_ns()
                latency_ms = (finished_ns - started_ns) / 1e6
                state_after = self._state()
                if success:
                    cache_sequence = self.cache.publish_success(value, finished_ns)
                    time_since_last_success_ms = (
                        None if last_success_ns is None else (finished_ns - last_success_ns) / 1e6
                    )
                    last_success_ns = finished_ns
                else:
                    assert error is not None
                    cache_sequence = self.cache.publish_error(error, finished_ns)
                    time_since_last_success_ms = (
                        None if last_success_ns is None else (finished_ns - last_success_ns) / 1e6
                    )
                row = {
                    "sequence_id": self._call_sequence,
                    "cache_sequence_id": cache_sequence,
                    "scheduled_time_ns": scheduled_ns,
                    "call_start_ns": started_ns,
                    "call_end_ns": finished_ns,
                    "query_latency_ms": latency_ms,
                    "host_call_start_monotonic_ns": started_ns,
                    "host_call_end_monotonic_ns": finished_ns,
                    "latency_ms": latency_ms,
                    "success": success,
                    "error_code": None if success else self.cache.snapshot(finished_ns).last_error_code,
                    "error_message": "" if success else f"{type(error).__name__}:{error}",
                    "robot_state_at_call": state_before.get("operation_state"),
                    "robot_operation_state": state_before.get("operation_state"),
                    "latest_rt_sequence": state_before.get("sequence_id"),
                    "latest_rt_timestamp": state_before.get("timestamp_s"),
                    "latest_rt_timestamp_ns": state_before.get("timestamp_ns"),
                    "latest_rt_age_ms": state_before.get("age_ms"),
                    "rt_sequence_after_call": state_after.get("sequence_id"),
                    "rt_updates_during_call": (
                        int(state_after["sequence_id"]) - int(state_before["sequence_id"])
                        if state_before.get("sequence_id") is not None and state_after.get("sequence_id") is not None
                        else None
                    ),
                    "previous_success_latency_ms": previous_success_latency_ms,
                    "time_since_last_success_ms": time_since_last_success_ms,
                    "value": value,
                }
                if success:
                    previous_success_latency_ms = latency_ms
                if self.on_result is not None:
                    self.on_result(row)
                next_tick_ns += self.period_ns
                delay_ns = next_tick_ns - self.clock_ns()
                if delay_ns > 0:
                    self._stop.wait(delay_ns / 1e9)
                elif -delay_ns > self.period_ns:
                    next_tick_ns = self.clock_ns()
        finally:
            self.cache.set_source_alive(False)
