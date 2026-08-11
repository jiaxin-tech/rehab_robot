"""Fail-closed, hardware-independent logging for one real experiment episode.

The logger owns four independent CSV streams plus one atomically replaced
``metadata.json``.  It does not import a robot adapter, start acquisition, or
issue motion commands.  Callers must wait for :meth:`wait_until_ready` before
allowing any trajectory execution and must stop if :attr:`healthy` becomes
false.

Rows are deliberately sparse: missing keys and explicit ``None`` values are
written as empty CSV cells.  The logger never substitutes a numeric zero for an
unavailable measurement.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping, TextIO

from utils.clock import MonotonicClock, SYSTEM_CLOCK, TIMESTAMP_SOURCE


ROBOT_STATE_FIELDS = (
    "host_time_s",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "valid",
    "invalid_reason",
)

ROBOT_WRENCH_FIELDS = (
    "query_start_s",
    "query_end_s",
    "publish_time_s",
    "joint_measured_torque_1",
    "joint_measured_torque_2",
    "joint_measured_torque_3",
    "joint_measured_torque_4",
    "joint_measured_torque_5",
    "joint_measured_torque_6",
    "joint_external_torque_1",
    "joint_external_torque_2",
    "joint_external_torque_3",
    "joint_external_torque_4",
    "joint_external_torque_5",
    "joint_external_torque_6",
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "frame_type",
    "query_duration_ms",
    "valid",
    "invalid_reason",
)

TRAJECTORY_COMMAND_FIELDS = (
    "host_time_s",
    "trajectory_time_s",
    "trajectory_phase",
    "delta_x_R",
    "delta_y_R",
    "delta_z_R",
    "tcp_target_x",
    "tcp_target_y",
    "tcp_target_z",
    "tcp_target_rx",
    "tcp_target_ry",
    "tcp_target_rz",
    "q_hip_ref",
    "q_knee_ref",
    "command_valid",
    "invalid_reason",
)

ALIGNED_SNAPSHOT_FIELDS = (
    "host_time_s",
    "state_time_s",
    "wrench_time_s",
    "state_age_s",
    "wrench_age_s",
    "state_wrench_skew_s",
    "state_valid",
    "wrench_valid",
    "state_thread_alive",
    "wrench_thread_alive",
    "query_duration_ms",
    "valid",
    "invalid_reason",
)

STREAM_FIELDS = {
    "robot_state": ROBOT_STATE_FIELDS,
    "robot_wrench": ROBOT_WRENCH_FIELDS,
    "trajectory_command": TRAJECTORY_COMMAND_FIELDS,
    "aligned_snapshot": ALIGNED_SNAPSHOT_FIELDS,
}

STREAM_FILENAMES = {
    "robot_state": "robot_state.csv",
    "robot_wrench": "robot_wrench.csv",
    "trajectory_command": "trajectory_command.csv",
    "aligned_snapshot": "aligned_snapshot.csv",
}


class EpisodeLoggerError(RuntimeError):
    """Raised when an episode logger is unavailable or enters failure state."""


def _row_mapping(row: Mapping[str, Any] | Any | None, values: Mapping[str, Any]) -> dict[str, Any]:
    if row is None:
        result: dict[str, Any] = {}
    elif isinstance(row, Mapping):
        result = dict(row)
    elif is_dataclass(row) and not isinstance(row, type):
        result = asdict(row)
    else:
        raise TypeError("row must be a mapping, dataclass instance, or None")
    overlap = set(result).intersection(values)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"duplicate row fields supplied twice: {names}")
    result.update(values)
    return result


def _json_value(value: Any, *, path: str = "metadata") -> Any:
    """Convert common metadata values without inventing unavailable numbers."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value), path=path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # NumPy scalar metadata is common in offline preparation, but importing
    # NumPy here would unnecessarily couple the pure logging layer to it.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        scalar = item_method()
        if scalar is not value:
            return _json_value(scalar, path=path)
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


class EpisodeLogger:
    """Own and monitor all persistent streams for one episode.

    Construction is side-effect free.  :meth:`start` creates the files and sets
    the ready barrier only after every CSV header and the initial metadata file
    have been flushed successfully.  A failure in any append marks the entire
    logger failed, atomically records the reason when possible, and prevents
    later writes.
    """

    METADATA_FILENAME = "metadata.json"

    def __init__(
        self,
        episode_directory: str | os.PathLike[str],
        metadata: Mapping[str, Any] | None = None,
        *,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        if not isinstance(clock, MonotonicClock):
            raise TypeError("clock must provide now_ns() and now_s()")
        self.episode_directory = Path(episode_directory)
        self._metadata = deepcopy(dict(metadata or {}))
        self._clock = clock
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._ready_event = threading.Event()
        self._failure_event = threading.Event()
        self._write_timeout_event = threading.Event()
        self._closed_event = threading.Event()
        self._pending_write_lock = threading.Lock()
        self._pending_write_threads: set[threading.Thread] = set()
        self._state = "new"
        # These slots are intentionally published without ``_lock``.  They are
        # the emergency stop-facing view used when a producer is stuck inside
        # flush/fsync while holding the logger I/O lock.  The next safe
        # lock-owning lifecycle operation reconciles them into metadata.
        self._signaled_failure_reason: str | None = None
        self._signaled_failed_stream: str | None = None
        self._failure_reason: str | None = None
        self._failed_stream: str | None = None
        self._metadata_write_error: str | None = None
        self._owns_episode_outputs = False
        self._started_ns: int | None = None
        self._finished_ns: int | None = None
        self._handles: dict[str, TextIO] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._row_counts = {name: 0 for name in STREAM_FIELDS}

    @property
    def paths(self) -> dict[str, Path]:
        result = {
            name: self.episode_directory / filename
            for name, filename in STREAM_FILENAMES.items()
        }
        result["metadata"] = self.episode_directory / self.METADATA_FILENAME
        return result

    @property
    def ready_event(self) -> threading.Event:
        """One-shot event set only after every output is ready for appends."""

        return self._ready_event

    @property
    def failure_event(self) -> threading.Event:
        return self._failure_event

    @property
    def write_timeout_event(self) -> threading.Event:
        """Signal that a durability acknowledgement missed its deadline."""

        return self._write_timeout_event

    @property
    def healthy_signal(self) -> bool:
        """Non-blocking health view for a real-time scheduler.

        A stream may be stuck inside an OS flush while holding the logger lock;
        the scheduler must still be able to observe timeout/failure signals and
        request motion stop without waiting for that lock.
        """

        return bool(
            self._ready_event.is_set()
            and not self._failure_event.is_set()
            and not self._write_timeout_event.is_set()
            and not self._closed_event.is_set()
        )

    @property
    def pending_write_count(self) -> int:
        with self._pending_write_lock:
            return len(self._pending_write_threads)

    @property
    def ready(self) -> bool:
        if self._failure_event.is_set():
            return False
        with self._lock:
            return self._state == "ready"

    @property
    def healthy(self) -> bool:
        if self._failure_event.is_set():
            return False
        with self._lock:
            return bool(
                self._state == "ready"
                and self._failure_reason is None
                and not self._failure_event.is_set()
                and not self._write_timeout_event.is_set()
            )

    @property
    def failed(self) -> bool:
        # Every transition to ``failed`` publishes this event first.  Reading
        # it avoids waiting behind a producer stuck in OS durability I/O.
        return self._failure_event.is_set()

    @property
    def status(self) -> str:
        if self._failure_event.is_set():
            return "failed"
        with self._lock:
            return self._state

    @property
    def failure_reason(self) -> str | None:
        if self._failure_event.is_set() and self._signaled_failure_reason is not None:
            return self._signaled_failure_reason
        with self._lock:
            return self._failure_reason

    @property
    def row_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._row_counts)

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait for successful initialization; return early on startup failure."""

        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._failure_event.is_set()
                    or self._state not in {"new", "starting"}
                ),
                timeout=timeout,
            )
            return self._state == "ready" and not self._failure_event.is_set()

    def assert_healthy(self) -> None:
        if self._failure_event.is_set():
            detail = (
                self._signaled_failure_reason
                or "logger failure was signaled"
            )
            raise EpisodeLoggerError(detail)
        with self._lock:
            if (
                self._state != "ready"
                or self._failure_reason is not None
                or self._failure_event.is_set()
                or self._write_timeout_event.is_set()
            ):
                detail = self._failure_reason or f"logger status is {self._state}"
                if self._write_timeout_event.is_set() and self._failure_reason is None:
                    detail = "logger durability acknowledgement timed out"
                raise EpisodeLoggerError(detail)

    def start(self) -> "EpisodeLogger":
        """Create all outputs without overwriting an existing episode."""

        with self._condition:
            if self._state != "new":
                raise EpisodeLoggerError(f"cannot start logger from state {self._state}")
            self._state = "starting"
            self._started_ns = self._clock.now_ns()
            self._condition.notify_all()

        try:
            self.episode_directory.mkdir(parents=True, exist_ok=True)
            collisions = [path for path in self.paths.values() if path.exists()]
            if collisions:
                names = ", ".join(path.name for path in collisions)
                raise FileExistsError(f"episode outputs already exist: {names}")

            with self._lock:
                for stream_name, filename in STREAM_FILENAMES.items():
                    path = self.episode_directory / filename
                    handle = path.open("x", newline="", encoding="utf-8")
                    self._owns_episode_outputs = True
                    self._handles[stream_name] = handle
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=STREAM_FIELDS[stream_name],
                        extrasaction="raise",
                    )
                    self._writers[stream_name] = writer
                    writer.writeheader()
                    handle.flush()
                self._state = "ready"
                self._write_metadata_locked()
                self._ready_event.set()
                self._condition.notify_all()
            return self
        except BaseException as exc:
            with self._condition:
                self._record_failure_locked("startup", exc)
                self._condition.notify_all()
            raise EpisodeLoggerError(
                f"episode logger startup failed: {type(exc).__name__}: {exc}"
            ) from exc

    # ``open`` is a readable alias for callers that model logger lifecycle as a
    # resource rather than a background service.
    open = start

    def append_robot_state(
        self, row: Mapping[str, Any] | Any | None = None, **values: Any
    ) -> None:
        self._append("robot_state", _row_mapping(row, values))

    def append_robot_wrench(
        self, row: Mapping[str, Any] | Any | None = None, **values: Any
    ) -> None:
        self._append("robot_wrench", _row_mapping(row, values))

    def append_trajectory_command(
        self, row: Mapping[str, Any] | Any | None = None, **values: Any
    ) -> None:
        self._append("trajectory_command", _row_mapping(row, values))

    def append_trajectory_command_bounded(
        self,
        row: Mapping[str, Any] | Any | None = None,
        *,
        timeout_s: float,
        **values: Any,
    ) -> None:
        """Wait at most ``timeout_s`` for durable command-intent logging.

        The write runs outside the trajectory scheduler.  A missed deadline is
        permanently fail-closed even if the OS write later returns: the caller
        must not dispatch that target or continue the episode.
        """

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError("timeout_s must be a finite positive number")
        mapped = _row_mapping(row, values)
        if self._failure_event.is_set():
            detail = self._signaled_failure_reason or "logger failure was signaled"
            raise EpisodeLoggerError(
                f"cannot append trajectory_command: {detail}"
            )
        done = threading.Event()
        decision_lock = threading.Lock()
        decision = "pending"
        outcome: BaseException | None = None

        def write_intent() -> None:
            nonlocal decision, outcome
            caught: BaseException | None = None
            try:
                self._append("trajectory_command", mapped)
            except BaseException as exc:
                caught = exc
            finally:
                finalize_timeout = False
                # Completion is acknowledged before any auxiliary cleanup can
                # block.  The caller and writer serialize on this tiny state
                # machine, so exactly one of completion or timeout wins.
                with decision_lock:
                    outcome = caught
                    if decision == "pending":
                        decision = "completed"
                    elif decision == "timed_out":
                        finalize_timeout = True
                    done.set()
                if finalize_timeout:
                    with self._condition:
                        if self._state == "ready":
                            self._record_failure_locked(
                                "trajectory_command",
                                TimeoutError(
                                    "trajectory command durability deadline missed"
                                ),
                            )
                            self._condition.notify_all()
                with self._pending_write_lock:
                    self._pending_write_threads.discard(threading.current_thread())

        writer = threading.Thread(
            target=write_intent,
            name="episode-command-durable-write",
            daemon=True,
        )
        with self._pending_write_lock:
            self._pending_write_threads.add(writer)
        try:
            writer.start()
        except BaseException as exc:
            with self._pending_write_lock:
                self._pending_write_threads.discard(writer)
            self.signal_failure(
                f"trajectory command durability writer failed to start: "
                f"{type(exc).__name__}: {exc}",
                stream="trajectory_command",
            )
            raise EpisodeLoggerError(
                "trajectory_command durability writer failed to start"
            ) from exc
        if not done.wait(float(timeout_s)):
            timeout_won = False
            with decision_lock:
                if decision == "pending":
                    decision = "timed_out"
                    # This method never acquires the logger I/O lock or writes
                    # metadata, so a blocked fsync cannot delay the stop-facing
                    # failure signal.
                    self.signal_failure(
                        "trajectory command durability deadline missed",
                        stream="trajectory_command",
                        write_timeout=True,
                    )
                    timeout_won = True
            if timeout_won:
                raise EpisodeLoggerError(
                    "trajectory_command durability acknowledgement exceeded "
                    f"{float(timeout_s):.6f}s"
                )
        with decision_lock:
            completed_outcome = outcome
        if completed_outcome is not None:
            raise completed_outcome

    def wait_for_pending_writes(self, timeout_s: float) -> bool:
        """Testing/cleanup helper; never used to delay a motion stop request."""

        if timeout_s < 0.0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be finite and non-negative")
        deadline = self._clock.now_s() + timeout_s
        while self.pending_write_count:
            remaining = deadline - self._clock.now_s()
            if remaining <= 0.0:
                return False
            threading.Event().wait(min(remaining, 0.005))
        return True

    def append_aligned_snapshot(
        self, row: Mapping[str, Any] | Any | None = None, **values: Any
    ) -> None:
        self._append("aligned_snapshot", _row_mapping(row, values))

    def append(
        self,
        stream_name: str,
        row: Mapping[str, Any] | Any | None = None,
        **values: Any,
    ) -> None:
        """Generic append for integrations that route records by stream name."""

        if stream_name not in STREAM_FIELDS:
            raise ValueError(f"unknown episode stream: {stream_name}")
        self._append(stream_name, _row_mapping(row, values))

    def _append(self, stream_name: str, row: Mapping[str, Any]) -> None:
        if self._failure_event.is_set():
            detail = self._signaled_failure_reason or "logger failure was signaled"
            raise EpisodeLoggerError(f"cannot append {stream_name}: {detail}")
        with self._condition:
            if self._failure_event.is_set() or self._state != "ready":
                detail = (
                    self._signaled_failure_reason
                    or self._failure_reason
                    or f"logger status is {self._state}"
                )
                raise EpisodeLoggerError(f"cannot append {stream_name}: {detail}")
            allowed = set(STREAM_FIELDS[stream_name])
            unexpected = sorted(set(row) - allowed)
            try:
                if unexpected:
                    raise ValueError(
                        f"unexpected {stream_name} fields: {', '.join(unexpected)}"
                    )
                # Supplying every field explicitly guarantees that missing values
                # use csv's empty-cell representation rather than a numeric fill.
                sparse_row = {
                    field: row[field] if field in row else None
                    for field in STREAM_FIELDS[stream_name]
                }
                self._writers[stream_name].writerow(sparse_row)
                handle = self._handles[stream_name]
                handle.flush()
                if stream_name == "trajectory_command":
                    # Each motion target is acknowledged only after the host OS
                    # has accepted an explicit file synchronization request.
                    os.fsync(handle.fileno())
                self._row_counts[stream_name] += 1
            except BaseException as exc:
                self._record_failure_locked(stream_name, exc)
                self._condition.notify_all()
                raise EpisodeLoggerError(
                    f"append to {stream_name} failed: {type(exc).__name__}: {exc}"
                ) from exc

    def update_metadata(self, updates: Mapping[str, Any]) -> None:
        """Merge caller metadata and atomically publish the complete JSON file."""

        if not isinstance(updates, Mapping):
            raise TypeError("metadata updates must be a mapping")
        if self._failure_event.is_set():
            detail = self._signaled_failure_reason or "logger failure was signaled"
            raise EpisodeLoggerError(f"cannot update metadata: {detail}")
        with self._condition:
            if self._failure_event.is_set() or self._state != "ready":
                raise EpisodeLoggerError(
                    f"cannot update metadata while logger status is {self._state}"
                )
            previous = deepcopy(self._metadata)
            self._metadata.update(deepcopy(dict(updates)))
            try:
                self._write_metadata_locked()
            except BaseException as exc:
                self._metadata = previous
                self._record_failure_locked("metadata", exc)
                self._condition.notify_all()
                raise EpisodeLoggerError(
                    f"metadata update failed: {type(exc).__name__}: {exc}"
                ) from exc

    def signal_failure(
        self,
        reason: str,
        *,
        stream: str = "external",
        write_timeout: bool = False,
    ) -> None:
        """Publish a stop-facing failure without logger-lock or file I/O.

        This is safe to call while another producer may be stuck inside
        ``flush``/``fsync``.  It intentionally does not mutate durable metadata;
        :meth:`mark_failed`, a completed bounded writer, or :meth:`close`
        reconciles the signal under the normal logger lock later.
        """

        reason = str(reason).strip()
        if not reason:
            raise ValueError("failure reason must not be empty")
        stream = str(stream).strip()
        if not stream:
            raise ValueError("failure stream must not be empty")
        if self._signaled_failure_reason is None:
            self._signaled_failure_reason = reason
        if self._signaled_failed_stream is None:
            self._signaled_failed_stream = stream
        if write_timeout:
            self._write_timeout_event.set()
        self._failure_event.set()

    def mark_failed(self, reason: str) -> None:
        """Public durable fail-closed transition for an acquisition fault."""

        reason = str(reason).strip()
        if not reason:
            raise ValueError("failure reason must not be empty")
        self.signal_failure(reason, stream="external")
        with self._condition:
            if self._state in {"completed", "aborted", "closed"}:
                raise EpisodeLoggerError(f"cannot fail logger from state {self._state}")
            if self._state == "failed":
                return
            self._record_failure_locked("external", RuntimeError(reason))
            self._condition.notify_all()

    def close(
        self,
        *,
        completed: bool = True,
        stop_reason: str | None = None,
    ) -> None:
        """Flush streams and atomically record the terminal episode state."""

        with self._condition:
            if self._state in {"completed", "aborted", "closed"}:
                return
            if self._state == "new":
                self._state = "closed"
                self._finished_ns = self._clock.now_ns()
                self._closed_event.set()
                self._condition.notify_all()
                return
            if self._state == "starting":
                raise EpisodeLoggerError("cannot close logger while startup is in progress")
            self._finished_ns = self._clock.now_ns()
            self._apply_signaled_failure_locked()
            if self._state != "failed":
                self._state = "completed" if completed else "aborted"
                if stop_reason:
                    self._failure_reason = str(stop_reason)
            # A collision failure owns none of the pre-existing episode.  Close
            # must therefore be side-effect free for that directory as well.
            if not self._owns_episode_outputs:
                self._close_streams_locked()
                self._closed_event.set()
                self._condition.notify_all()
                return
            try:
                self._write_metadata_locked()
            except BaseException as exc:
                self._state = "failed"
                self._failure_reason = (
                    f"metadata_close:{type(exc).__name__}:{exc}"
                )
                self._failed_stream = "metadata"
                self._failure_event.set()
                self._metadata_write_error = self._failure_reason
                self._condition.notify_all()
                self._close_streams_locked()
                raise EpisodeLoggerError(self._failure_reason) from exc
            self._close_streams_locked()
            self._closed_event.set()
            self._condition.notify_all()

    def _record_failure_locked(self, stream_name: str, exc: BaseException) -> None:
        reason = f"{stream_name}:{type(exc).__name__}:{exc}"
        if self._failure_reason is None:
            self._failure_reason = reason
            self._failed_stream = stream_name
        self._state = "failed"
        self._finished_ns = self._finished_ns or self._clock.now_ns()
        self.signal_failure(reason, stream=stream_name)
        # Never write metadata unless this instance created at least one of the
        # episode outputs.  In particular, a startup collision remains foreign
        # even when close() is called later by a finally block.
        if self._owns_episode_outputs:
            try:
                self._write_metadata_locked()
            except BaseException as metadata_exc:
                self._metadata_write_error = (
                    f"{type(metadata_exc).__name__}:{metadata_exc}"
                )
        self._close_streams_locked()

    def _apply_signaled_failure_locked(self) -> None:
        if not self._failure_event.is_set():
            return
        if self._failure_reason is None:
            self._failure_reason = (
                self._signaled_failure_reason or "external failure was signaled"
            )
        if self._failed_stream is None:
            self._failed_stream = self._signaled_failed_stream or "external"
        self._state = "failed"
        self._finished_ns = self._finished_ns or self._clock.now_ns()

    def _close_streams_locked(self) -> None:
        for handle in self._handles.values():
            try:
                handle.flush()
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass
        self._handles.clear()
        self._writers.clear()

    def _metadata_payload_locked(self) -> dict[str, Any]:
        updated_ns = self._clock.now_ns()
        duration_s = None
        if self._started_ns is not None and self._finished_ns is not None:
            elapsed_ns = self._finished_ns - self._started_ns
            if elapsed_ns >= 0:
                duration_s = elapsed_ns / 1_000_000_000
        average_rates_hz = {
            name: (
                count / duration_s
                if duration_s is not None and duration_s > 0.0
                else None
            )
            for name, count in self._row_counts.items()
        }
        payload = deepcopy(self._metadata)
        payload["logger"] = {
            "status": self._state,
            "ready": self._state == "ready",
            "healthy": self._state == "ready" and self._failure_reason is None,
            "timestamp_source": TIMESTAMP_SOURCE,
            "started_host_time_ns": self._started_ns,
            "started_host_time_s": (
                self._started_ns / 1_000_000_000
                if self._started_ns is not None
                else None
            ),
            "metadata_updated_host_time_ns": updated_ns,
            "metadata_updated_host_time_s": updated_ns / 1_000_000_000,
            "finished_host_time_ns": self._finished_ns,
            "finished_host_time_s": (
                self._finished_ns / 1_000_000_000
                if self._finished_ns is not None
                else None
            ),
            "episode_duration_s": duration_s,
            # Host-observed publication averages; these are not controller
            # device rates and do not imply SDK/device synchronization.
            "average_stream_publish_rate_hz": average_rates_hz,
            "failure_reason": self._failure_reason,
            "failed_stream": self._failed_stream,
            "metadata_write_error": self._metadata_write_error,
            "row_counts": dict(self._row_counts),
            "files": dict(STREAM_FILENAMES),
        }
        return _json_value(payload)

    def _write_metadata_locked(self) -> None:
        self.episode_directory.mkdir(parents=True, exist_ok=True)
        payload = self._metadata_payload_locked()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".metadata.",
            suffix=".tmp",
            dir=self.episode_directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.paths["metadata"])
        except BaseException:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def __enter__(self) -> "EpisodeLogger":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is None:
            self.close(completed=True)
        else:
            if self._state not in {"failed", "completed", "aborted", "closed"}:
                self.mark_failed(f"context_exception:{exc_type.__name__}:{exc}")
            self.close(completed=False, stop_reason=str(exc))
        return False


__all__ = [
    "ALIGNED_SNAPSHOT_FIELDS",
    "EpisodeLogger",
    "EpisodeLoggerError",
    "ROBOT_STATE_FIELDS",
    "ROBOT_WRENCH_FIELDS",
    "STREAM_FIELDS",
    "STREAM_FILENAMES",
    "TRAJECTORY_COMMAND_FIELDS",
]
