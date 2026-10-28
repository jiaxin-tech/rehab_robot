"""Offline regression tests for acquisition, mechanics and personal experiments."""
from __future__ import annotations
from config.bundle import section_path
from utils.assets import artifact_path

# ============================================================================
# Source: tests/test_acquisition_timing_and_tail.py
# ============================================================================

"""Offline regression coverage for interrupted waits and shutdown handoff.

Unit doubles control the shutdown phases. The integration test uses a real
spawned WrenchProcessProvider and holds a success event before durable ACK.
No native SDK, robot connection, or motion is used.
"""


import csv as acquisition_timing_and_tail__csv
import json as acquisition_timing_and_tail__json
from pathlib import Path as acquisition_timing_and_tail__Path
import threading as acquisition_timing_and_tail__threading
import time as acquisition_timing_and_tail__time
from types import SimpleNamespace as acquisition_timing_and_tail__SimpleNamespace

import pytest as acquisition_timing_and_tail__pytest

from collection import real_robot_acquisition as acquisition_timing_and_tail__acquisition_module
from collection.episode_logger import EpisodeLogger as acquisition_timing_and_tail__EpisodeLogger, EpisodeLoggerError as acquisition_timing_and_tail__EpisodeLoggerError
from collection.real_robot_acquisition import RealRobotAcquisition as acquisition_timing_and_tail__RealRobotAcquisition
from collection.state import KinematicStateFrame as acquisition_timing_and_tail__KinematicStateFrame
from collection.wrench_process import DurableAudit as acquisition_timing_and_tail__DurableAudit, WrenchBudgets as acquisition_timing_and_tail__WrenchBudgets, WrenchProcessProvider as acquisition_timing_and_tail__WrenchProcessProvider
from hardware.rokae_adapter import RobotWrenchFrame as acquisition_timing_and_tail__RobotWrenchFrame


def acquisition_timing_and_tail__frame(sequence):
    stamp = acquisition_timing_and_tail__time.perf_counter() + sequence * 1e-6
    return acquisition_timing_and_tail__RobotWrenchFrame(
        sequence, stamp, stamp, stamp, stamp, "", "OFFLINE_TEST", True, "", "world",
        (float(sequence), 2., 3.), (.1, .2, .3), (1.,) * 6, (.5,) * 6,
    )


class acquisition_timing_and_tail__OfflineAdapter:
    def __init__(self):
        self.connected = False
        self.calls = []
        self.sequence = 0

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def is_connected(self):
        return self.connected

    def start_state_stream(self):
        self.calls.append("start_state_stream")

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False

    def read_state_frame(self):
        self.sequence += 1
        stamp = acquisition_timing_and_tail__time.perf_counter()
        return acquisition_timing_and_tail__KinematicStateFrame(
            sequence_id=self.sequence, host_monotonic_time_s=stamp,
            wall_time_iso=None, robot_device_time_s=None, valid=True,
            invalid_reason="", tcp_position_m=(0., 0., 0.),
            tcp_orientation_rad=(0., 0., 0.), tcp_linear_velocity_mps=None,
            tcp_angular_velocity_radps=None, velocity_source="unavailable",
            joint_position_rad=(0.,) * 6, joint_velocity_radps=None,
            pose_time_s=stamp, joint_time_s=stamp, velocity_time_s=None,
            operation_state="IDLE", collision_state=None, controller_error=None,
        )


class acquisition_timing_and_tail__RecordingLogger:
    def __init__(self, *, fail_append=False):
        self.rows = []
        self.append_attempts = 0
        self.closed = False
        self.fail_append = fail_append
        self.failures = []

    @property
    def healthy_signal(self):
        return not self.closed and not self.failures

    def assert_healthy(self):
        assert self.healthy_signal

    def append_robot_wrench(self, **row):
        self.append_attempts += 1
        if self.closed:
            raise AssertionError("a duplicate stop tried to append to a closed logger")
        if self.fail_append:
            raise acquisition_timing_and_tail__EpisodeLoggerError("injected_tail_write_failure")
        self.rows.append(row)

    def signal_failure(self, reason, **_kwargs):
        self.failures.append(reason)

    def mark_failed(self, reason):
        self.failures.append(reason)


class acquisition_timing_and_tail__ShutdownProvider:
    """A cached result before shutdown and one durable result during shutdown."""
    def __init__(self, before, after, *, stop_error=None):
        self.config = {"mode": "offline", "rate_hz": 50.}
        self.budgets = acquisition_timing_and_tail__SimpleNamespace(state_age_s=3., sample_age_s=3., skew_s=3.)
        self.last_good = before
        self.after = after
        self.stop_error = stop_error
        self.process = None
        self.cleanup = None
        self.stop_calls = 0
        self.failures = []

    def start(self):
        self.process = object()
        return self

    def poll(self):
        return self.health()

    def health(self):
        return dict(FAILURE_LATCH=tuple(self.failures), STREAM_HEALTH=not self.failures,
                    PROCESS_ALIVE=self.cleanup is None, CURRENT_QUERY_STATE="SUCCEEDED")

    def fail(self, reason):
        self.failures.append(reason)

    def stop(self):
        self.stop_calls += 1
        if self.stop_error:
            raise self.stop_error
        if self.cleanup is None:
            self.last_good = self.after
            self.cleanup = dict(NORMAL_CLEANUP=True, process_exited=True)
        return dict(self.cleanup)


class acquisition_timing_and_tail__ControlledThread:
    def __init__(self, *, name, stuck=False, **_kwargs):
        self.name = name
        self.stuck = stuck
        self.alive = False
        self.join_calls = 0

    def start(self):
        self.alive = True

    def join(self, timeout=None):
        self.join_calls += 1
        if not self.stuck:
            self.alive = False

    def is_alive(self):
        return self.alive


def acquisition_timing_and_tail__controlled_acquisition(before, after, *, fail_append=False, stuck_name=None, stop_error=None):
    adapter = acquisition_timing_and_tail__OfflineAdapter()
    logger = acquisition_timing_and_tail__RecordingLogger(fail_append=fail_append)
    provider = acquisition_timing_and_tail__ShutdownProvider(before, after, stop_error=stop_error)
    threads = {}
    def make_thread(**kwargs):
        item = acquisition_timing_and_tail__ControlledThread(stuck=kwargs["name"] == stuck_name, **kwargs)
        threads[kwargs["name"]] = item
        return item
    acquisition = acquisition_timing_and_tail__RealRobotAcquisition(adapter, logger, wrench_provider=provider, thread_factory=make_thread)
    acquisition.start()
    return acquisition, adapter, logger, provider, threads


@acquisition_timing_and_tail__pytest.mark.parametrize("before_id,after_id,expected", [
    (1, 2, [1., 2.]),
    (1, 1, [1.]),
    (None, 1, [1.]),
    (None, None, []),
])
def test_acquisition_timing_and_tail__shutdown_keeps_snapshot_and_final_success_once(before_id, after_id, expected):
    before = acquisition_timing_and_tail__frame(before_id) if before_id is not None else None
    # Use the same immutable frame for the duplicate case, as the real provider does.
    after = before if before_id == after_id else acquisition_timing_and_tail__frame(after_id) if after_id is not None else None
    acquisition, adapter, logger, provider, threads = acquisition_timing_and_tail__controlled_acquisition(before, after)
    acquisition.stop()
    assert [row["fx"] for row in logger.rows] == expected
    assert logger.append_attempts == len(expected)
    assert all(not item.is_alive() and item.join_calls == 1 for item in threads.values())
    assert provider.cleanup["NORMAL_CLEANUP"]
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]


def test_acquisition_timing_and_tail__repeated_stop_does_not_duplicate_tail_or_touch_closed_logger():
    acquisition, _, logger, _, _ = acquisition_timing_and_tail__controlled_acquisition(acquisition_timing_and_tail__frame(1), acquisition_timing_and_tail__frame(2))
    acquisition.stop()
    assert [row["fx"] for row in logger.rows] == [1., 2.]
    logger.closed = True
    acquisition.stop()
    assert [row["fx"] for row in logger.rows] == [1., 2.]
    assert logger.append_attempts == 2


def test_acquisition_timing_and_tail__tail_write_failure_still_stops_provider_and_cleans_adapter_then_raises():
    acquisition, adapter, logger, provider, _ = acquisition_timing_and_tail__controlled_acquisition(acquisition_timing_and_tail__frame(1), acquisition_timing_and_tail__frame(2), fail_append=True)
    with acquisition_timing_and_tail__pytest.raises(Exception, match="injected_tail_write_failure"):
        acquisition.stop()
    assert provider.stop_calls == 1
    assert provider.cleanup["NORMAL_CLEANUP"]
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]
    assert not adapter.connected
    assert logger.rows == []
    assert logger.append_attempts == 1, "do not try a second write after an uncertain first write failed"
    assert logger.failures


def test_acquisition_timing_and_tail__provider_stop_failure_does_not_append_tail_or_change_existing_adapter_error_path():
    acquisition, adapter, logger, provider, _ = acquisition_timing_and_tail__controlled_acquisition(
        acquisition_timing_and_tail__frame(1), acquisition_timing_and_tail__frame(2), stop_error=RuntimeError("injected_provider_stop_failure"),
    )
    with acquisition_timing_and_tail__pytest.raises(RuntimeError, match="injected_provider_stop_failure"):
        acquisition.stop()
    assert provider.stop_calls == 1
    assert logger.rows == []
    assert logger.append_attempts == 0
    assert adapter.calls == ["connect", "start_state_stream"]
    assert logger.failures


def test_acquisition_timing_and_tail__join_failure_never_appends_tail_or_disconnects_parent():
    acquisition, adapter, logger, provider, _ = acquisition_timing_and_tail__controlled_acquisition(
        acquisition_timing_and_tail__frame(1), acquisition_timing_and_tail__frame(2), stuck_name="real-episode-state",
    )
    with acquisition_timing_and_tail__pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        acquisition.stop()
    assert provider.stop_calls == 1  # isolated child still receives its existing cleanup.
    assert logger.append_attempts == 0
    assert adapter.calls == ["connect", "start_state_stream"]
    assert logger.failures


class acquisition_timing_and_tail__FakeClock:
    def __init__(self):
        self.value = 10.

    def now_s(self):
        return self.value

    def now_ns(self):
        return int(self.value * 1e9)


class acquisition_timing_and_tail__NoBlockingEvent:
    def __init__(self):
        self.stopped = False
        self.wait_calls = 0

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, timeout=None):
        # Stop old implementations on first bad wait so regressions never hang.
        self.wait_calls += 1
        self.stopped = True
        return True


class acquisition_timing_and_tail__AlwaysAliveThread:
    def is_alive(self):
        return True


class acquisition_timing_and_tail__StartupHealthProvider:
    def __init__(self, stamp):
        self.config = {"mode": "offline", "rate_hz": 50.0}
        self.budgets = acquisition_timing_and_tail__SimpleNamespace(
            state_age_s=.025,
            sample_age_s=.05,
            skew_s=.04,
        )
        self.last_good = acquisition_timing_and_tail__frame(1)
        object.__setattr__(self.last_good, "host_monotonic_time_s", stamp)
        self.failures = []

    def poll(self):
        return dict(
            STREAM_HEALTH=not self.failures,
            LATEST_GOOD_SAMPLE_VALID=True,
            PROCESS_ALIVE=True,
            CURRENT_QUERY_STATE="SUCCEEDED",
            FAILURE_LATCH=tuple(self.failures),
        )

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)


def acquisition_timing_and_tail__state_frame(sequence, stamp):
    return acquisition_timing_and_tail__KinematicStateFrame(
        sequence_id=sequence,
        host_monotonic_time_s=stamp,
        wall_time_iso=None,
        robot_device_time_s=None,
        valid=True,
        invalid_reason="",
        tcp_position_m=(0., 0., 0.),
        tcp_orientation_rad=(0., 0., 0.),
        tcp_linear_velocity_mps=None,
        tcp_angular_velocity_radps=None,
        velocity_source="unavailable",
        joint_position_rad=(0.,) * 6,
        joint_velocity_radps=None,
        pose_time_s=stamp,
        joint_time_s=stamp,
        velocity_time_s=None,
        operation_state="IDLE",
        collision_state=None,
        controller_error=None,
    )


def test_acquisition_timing_and_tail__startup_misalignment_does_not_kill_wrench_before_first_healthy_pair():
    clock = acquisition_timing_and_tail__FakeClock()
    provider = acquisition_timing_and_tail__StartupHealthProvider(clock.now_s())
    acquisition = acquisition_timing_and_tail__RealRobotAcquisition(
        acquisition_timing_and_tail__SimpleNamespace(), acquisition_timing_and_tail__RecordingLogger(), wrench_provider=provider, clock=clock,
    )
    acquisition._threads = {
        name: acquisition_timing_and_tail__AlwaysAliveThread() for name in ("state", "wrench", "alignment")
    }
    acquisition._latest_wrench = provider.last_good
    acquisition._latest_state = acquisition_timing_and_tail__state_frame(1, clock.now_s() - .2)

    startup = acquisition.latest_health()
    assert not startup.valid
    assert "state_stale_or_unavailable" in startup.invalid_reason
    assert "state_wrench_skew_unavailable_or_exceeded" in startup.invalid_reason
    assert provider.failures == []

    acquisition._latest_state = acquisition_timing_and_tail__state_frame(2, clock.now_s())
    assert acquisition.latest_health().valid
    assert acquisition._runtime_freshness_armed is True

    acquisition._latest_state = acquisition_timing_and_tail__state_frame(3, clock.now_s() - .2)
    runtime = acquisition.latest_health()
    assert not runtime.valid
    assert provider.failures == [
        "state_freshness_failure",
        "state_wrench_skew_failure",
    ]


@acquisition_timing_and_tail__pytest.mark.parametrize("interrupt", [False, True])
def test_acquisition_timing_and_tail__deadline_wait_uses_short_sleep_slices_and_observes_stop(monkeypatch, interrupt):
    clock = acquisition_timing_and_tail__FakeClock()
    acquisition = acquisition_timing_and_tail__RealRobotAcquisition(acquisition_timing_and_tail__OfflineAdapter(), acquisition_timing_and_tail__RecordingLogger(), diagnostic_state_only=True, clock=clock)
    event = acquisition_timing_and_tail__NoBlockingEvent()
    acquisition._stop_event = event
    slices = []
    def sleep(seconds):
        slices.append(seconds)
        assert 0 < seconds <= .002
        clock.value += seconds
        if interrupt:
            event.set()
    monkeypatch.setattr(acquisition_timing_and_tail__acquisition_module, "time", acquisition_timing_and_tail__SimpleNamespace(sleep=sleep))
    deadline = clock.now_s() + .009
    returned = acquisition._sleep_until(deadline)
    assert event.wait_calls == 0
    assert slices
    if interrupt:
        assert len(slices) == 1
        assert returned < deadline
    else:
        assert returned == acquisition_timing_and_tail__pytest.approx(deadline)
        assert len(slices) >= 5


def test_acquisition_timing_and_tail__wrench_no_new_frame_still_uses_deadline_scheduler():
    clock = acquisition_timing_and_tail__FakeClock()
    provider = acquisition_timing_and_tail__ShutdownProvider(None, None)
    acquisition = acquisition_timing_and_tail__RealRobotAcquisition(acquisition_timing_and_tail__OfflineAdapter(), acquisition_timing_and_tail__RecordingLogger(), wrench_provider=provider, clock=clock)
    event = acquisition_timing_and_tail__NoBlockingEvent()
    acquisition._stop_event = event
    deadlines = []
    def scheduled_wait(deadline):
        deadlines.append(deadline)
        clock.value = deadline
        if len(deadlines) == 3:
            event.set()
        return clock.now_s()
    acquisition._sleep_until = scheduled_wait
    acquisition._wrench_loop()
    assert event.wait_calls == 0
    assert len(deadlines) == 3
    assert all(b > a for a, b in zip(deadlines, deadlines[1:]))


def acquisition_timing_and_tail__wait_for(predicate, timeout=5.):
    deadline = acquisition_timing_and_tail__time.perf_counter() + timeout
    while acquisition_timing_and_tail__time.perf_counter() < deadline:
        if predicate():
            return
        acquisition_timing_and_tail__time.sleep(.001)
    raise AssertionError("offline test condition was not observed before its deadline")


def test_acquisition_timing_and_tail__real_provider_publishes_before_fsync_and_settles_audit_during_stop(tmp_path):
    success_waiting = acquisition_timing_and_tail__threading.Event()
    release_write = acquisition_timing_and_tail__threading.Event()
    real_stop_observed = acquisition_timing_and_tail__threading.Event()
    release_errors = []

    def hold_second_success(event):
        if event.get("type") == "QUERY_SUCCESS" and event.get("query_id") == 2:
            success_waiting.set()
            if not release_write.wait(5.):
                raise TimeoutError("test did not release second success")

    budgets = acquisition_timing_and_tail__WrenchBudgets(startup_s=5., query_s=3., progress_s=4., sample_age_s=4.,
        ack_s=5., log_s=4., shutdown_s=1.5, terminate_s=.5, state_age_s=4., skew_s=4.,
        provenance="NOT_SAFETY_THRESHOLDS")
    audit_path = tmp_path / "query_events.jsonl"
    provider = acquisition_timing_and_tail__WrenchProcessProvider(dict(mode="offline", rate_hz=50., behavior="normal"), budgets,
        audit_path, audit_factory=lambda path: acquisition_timing_and_tail__DurableAudit(path, write_hook=hold_second_success))
    logger = acquisition_timing_and_tail__EpisodeLogger(tmp_path / "episode").start()
    adapter = acquisition_timing_and_tail__OfflineAdapter()
    acquisition = acquisition_timing_and_tail__RealRobotAcquisition(adapter, logger, wrench_provider=provider)
    released = None

    def release_only_after_actual_provider_stop():
        try:
            acquisition_timing_and_tail__wait_for(lambda: provider.stop_flag.value == 1, timeout=2.)
            real_stop_observed.set()
        except Exception as exc:
            release_errors.append(str(exc))
        finally:
            release_write.set()

    try:
        acquisition.start()
        assert success_waiting.wait(5.), "second success did not reach the audit writer"
        # QUERY_SUCCESS 2 is usable after bounded admission even while its
        # fsync is deliberately blocked.
        acquisition_timing_and_tail__wait_for(lambda: logger.row_counts["robot_wrench"] >= 2)
        assert provider.last_good.sequence_id >= 2
        assert not release_write.is_set()
        released = acquisition_timing_and_tail__threading.Thread(target=release_only_after_actual_provider_stop, daemon=True)
        released.start()
        acquisition.stop()
        released.join(2.)
        assert not released.is_alive()
        assert not release_errors
        assert real_stop_observed.is_set()
        assert provider.cleanup["NORMAL_CLEANUP"], provider.cleanup
        assert provider.cleanup["process_exited"]
        assert not provider.cleanup["FORCED_WORKER_TERMINATION"]
        assert provider.last_good.sequence_id >= 2
        assert logger.row_counts["robot_wrench"] >= 2
        logger.close(completed=False, stop_reason="OFFLINE_REGRESSION_TEST")
        with (tmp_path / "episode/robot_wrench.csv").open(newline="", encoding="utf-8") as stream:
            recorded = list(acquisition_timing_and_tail__csv.DictReader(stream))
        events = [acquisition_timing_and_tail__json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        successful = [row for row in events if row["type"] == "QUERY_SUCCESS"]
        assert [row["query_id"] for row in successful[:2]] == [1, 2]
        audited_starts = {row["frame"]["host_query_start_s"] for row in successful}
        assert {float(row["query_start_s"]) for row in recorded}.issubset(audited_starts)
        # A second stop after logger closure must not append the final row again.
        recorded_count = len(recorded)
        acquisition.stop()
        with (tmp_path / "episode/robot_wrench.csv").open(newline="", encoding="utf-8") as stream:
            assert len(list(acquisition_timing_and_tail__csv.DictReader(stream))) == recorded_count
    finally:
        release_write.set()
        if released is not None:
            released.join(2.)
        try:
            if acquisition.live_producer_names:
                acquisition.stop()
        finally:
            try:
                if provider.process is not None and provider.cleanup is None:
                    provider.stop()
            finally:
                logger.close(completed=False, stop_reason="OFFLINE_REGRESSION_TEST_CLEANUP")
    assert provider.process is not None and not provider.process.is_alive()



# ============================================================================
# Source: tests/test_diagnose_collision_events.py
# ============================================================================

import json as diagnose_collision_events__json

import pytest as diagnose_collision_events__pytest

from hardware.safety_events import SafetyEventMonitor as diagnose_collision_events__SafetyEventMonitor
from scripts.diagnose_collision_events import diagnose as diagnose_collision_events__diagnose


@diagnose_collision_events__pytest.mark.parametrize("behavior", ["silent", "clear", "periodic", "collision", "disconnect", "registration_failure"])
def test_diagnose_collision_events__callback_diagnostic_preserves_unknown_faults_and_cleanup(tmp_path, behavior):
    class Clock:
        now = 10.

        def __call__(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds
            if behavior == "periodic":
                monitor.receive(robot.generation, dict(collided=False))
            if behavior == "disconnect":
                robot.is_connected = False
                monitor.invalidate("safety_event_disconnected")

    clock = Clock()
    monitor = diagnose_collision_events__SafetyEventMonitor(.05, clock=clock)
    calls = []

    class CallbackRobot:
        is_connected = False

        def connect(self):
            calls.append("connect")
            self.is_connected = True
            generation = monitor.begin()
            self.generation = generation
            if behavior == "registration_failure":
                monitor.invalidate("safety_event_registration_failed")
                raise RuntimeError("registration failure")
            monitor.registered(generation)
            if behavior in ("clear", "periodic"):
                monitor.receive(generation, dict(collided=False))
            if behavior == "collision":
                monitor.receive(generation, dict(collided=True))
                monitor.receive(generation, dict(collided=False))

        def get_safety_event_status(self):
            calls.append("cached_status")
            return monitor.snapshot().to_dict()

        def disconnect(self):
            calls.append("disconnect")
            self.is_connected = False
            monitor.invalidate("safety_event_disconnected")

    robot = CallbackRobot()

    def factory(ip, **kwargs):
        assert ip == "192.0.2.1"
        assert kwargs == dict(local_ip="192.0.2.2", collision_source="events", collision_event_max_age_s=.05)
        return robot

    path = tmp_path / "events.json"
    report = diagnose_collision_events__diagnose(robot_ip="192.0.2.1", local_ip="192.0.2.2", output=path,
                      duration_s=.1, max_age_s=.05, robot_factory=factory,
                      clock=clock, sleep=clock.sleep)
    assert diagnose_collision_events__json.loads(path.read_text(encoding="utf-8")) == report
    assert report["cleanup"]["disconnected_confirmed"]
    assert report["cleanup"]["status_after_disconnect"]["valid"] is False
    assert report["execution_authorized"] is False
    assert report["assessment"]["final_status_valid"] is (behavior == "periodic")
    assert calls[-2:] == ["disconnect", "cached_status"]
    if behavior == "silent":
        assert report["assessment"]["observed_event_count"] == 0
        assert report["assessment"]["unresolved_reasons"] == ["safety_event_initial_state_unknown"]
    if behavior == "clear":
        assert report["assessment"]["observed_clear"]
        assert "safety_event_expired_or_clock_invalid" in report["assessment"]["unresolved_reasons"]
    if behavior == "collision":
        assert report["assessment"]["observed_collision"]
    if behavior == "registration_failure":
        assert report["errors"]
    if behavior == "periodic":
        assert report["assessment"]["observed_clear"]
        assert report["assessment"]["observed_event_count"] > 1
        assert not report["assessment"]["unresolved_reasons"]


@diagnose_collision_events__pytest.mark.parametrize("age,duration", [(None, 1), (0, 1), (float("nan"), 1), (.1, -1)])
def test_diagnose_collision_events__invalid_parameters_fail_before_connection_and_output(tmp_path, age, duration):
    path = tmp_path / "events.json"
    with diagnose_collision_events__pytest.raises(ValueError):
        diagnose_collision_events__diagnose(robot_ip="192.0.2.1", local_ip="192.0.2.2", output=path,
                 max_age_s=age, duration_s=duration,
                 robot_factory=lambda *_args, **_kwargs: diagnose_collision_events__pytest.fail("connection forbidden"))
    assert not path.exists()


def test_diagnose_collision_events__existing_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "events.json"
    path.write_text("original", encoding="utf-8")
    with diagnose_collision_events__pytest.raises(FileExistsError):
        diagnose_collision_events__diagnose(robot_ip="192.0.2.1", local_ip="192.0.2.2", output=path,
                 max_age_s=.1, duration_s=1,
                 robot_factory=lambda *_args, **_kwargs: diagnose_collision_events__pytest.fail("connection forbidden"))
    assert path.read_text() == "original"



# ============================================================================
# Source: tests/test_episode_logger.py
# ============================================================================

"""Offline tests for the hardware-independent real-episode data logger."""



import csv as episode_logger__csv
import json as episode_logger__json
from pathlib import Path as episode_logger__Path
import tempfile as episode_logger__tempfile
import threading as episode_logger__threading
import unittest as episode_logger__unittest
from unittest.mock import patch as episode_logger__patch

from collection.episode_logger import (
    EpisodeLogger as episode_logger__EpisodeLogger,
    EpisodeLoggerError as episode_logger__EpisodeLoggerError,
    STREAM_FILENAMES as episode_logger__STREAM_FILENAMES,
)
from utils.clock import PerfCounterClock as episode_logger__PerfCounterClock, TIMESTAMP_SOURCE as episode_logger__TIMESTAMP_SOURCE


class episode_logger___DeterministicClock:
    def __init__(self, first_ns: int = 10_000_000_000) -> None:
        self.value_ns = first_ns

    def now_ns(self) -> int:
        result = self.value_ns
        self.value_ns += 1_000_000
        return result

    def now_s(self) -> float:
        return self.now_ns() / 1_000_000_000


def episode_logger___read_one(path: episode_logger__Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(episode_logger__csv.DictReader(stream))
    if len(rows) != 1:
        raise AssertionError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


class episode_logger___WriterCleanupGate:
    """Block only the durable writer's pending-set cleanup."""

    def __init__(self) -> None:
        self._lock = episode_logger__threading.Lock()
        self.writer_entered = episode_logger__threading.Event()
        self.release_writer = episode_logger__threading.Event()

    def __enter__(self):
        if episode_logger__threading.current_thread().name == "episode-command-durable-write":
            self.writer_entered.set()
            if not self.release_writer.wait(timeout=2.0):
                raise TimeoutError("test did not release writer cleanup gate")
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lock.release()
        return False


class episode_logger__ClockTests(episode_logger__unittest.TestCase):
    def test_clock_uses_perf_counter_ns_as_the_only_source(self) -> None:
        with episode_logger__patch("utils.clock.time.perf_counter_ns", return_value=1_234_567_890):
            clock = episode_logger__PerfCounterClock()
            self.assertEqual(clock.now_ns(), 1_234_567_890)
            self.assertAlmostEqual(clock.now_s(), 1.23456789)
        self.assertEqual(clock.timestamp_source, episode_logger__TIMESTAMP_SOURCE)


class episode_logger__EpisodeLoggerTests(episode_logger__unittest.TestCase):
    def test_ready_barrier_independent_streams_and_sparse_values(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_0001"
            logger = episode_logger__EpisodeLogger(
                episode,
                {
                    "git_commit": None,
                    "trajectory_id": "reference_measured_asymmetric_closed_slow",
                },
                clock=episode_logger___DeterministicClock(),
            )
            self.assertFalse(episode.exists())
            self.assertFalse(logger.wait_until_ready(timeout=0.0))

            logger.start()
            self.assertTrue(logger.ready_event.is_set())
            self.assertTrue(logger.wait_until_ready(timeout=0.0))
            self.assertTrue(logger.healthy)
            for filename in (*episode_logger__STREAM_FILENAMES.values(), "metadata.json"):
                self.assertTrue((episode / filename).is_file(), filename)

            logger.append_robot_state(
                host_time_s=1.0,
                q1=None,
                tcp_x=0.31,
                valid=False,
                invalid_reason="joint_positions_unavailable",
            )
            logger.append_robot_wrench(
                query_start_s=1.001,
                query_end_s=1.004,
                publish_time_s=1.004,
                fx=None,
                frame_type="world",
                query_duration_ms=3.0,
                valid=False,
                invalid_reason="cartesian_force_unavailable",
            )
            logger.append_trajectory_command(
                host_time_s=1.005,
                trajectory_time_s=0.0,
                trajectory_phase=0.0,
                delta_x_R=0.0,
                delta_y_R=0.0,
                delta_z_R=0.0,
                tcp_target_x=0.31,
                tcp_target_y=None,
                tcp_target_z=0.42,
                command_valid=True,
                invalid_reason="",
            )
            logger.append_aligned_snapshot(
                host_time_s=1.006,
                state_time_s=1.0,
                wrench_time_s=1.004,
                state_age_s=0.006,
                wrench_age_s=0.002,
                state_wrench_skew_s=0.004,
                state_valid=False,
                wrench_valid=False,
                state_thread_alive=True,
                wrench_thread_alive=True,
                query_duration_ms=3.0,
                valid=False,
                invalid_reason="joint_positions_unavailable;cartesian_force_unavailable",
            )
            logger.update_metadata({"operator_confirmation": None})
            logger.close(completed=True)

            state = episode_logger___read_one(episode / "robot_state.csv")
            self.assertEqual(state["q1"], "")
            self.assertEqual(state["q2"], "")
            self.assertNotEqual(state["valid"], "")
            self.assertEqual(state["invalid_reason"], "joint_positions_unavailable")

            wrench = episode_logger___read_one(episode / "robot_wrench.csv")
            self.assertEqual(wrench["fx"], "")
            self.assertEqual(wrench["fy"], "")
            self.assertEqual(wrench["query_duration_ms"], "3.0")

            command = episode_logger___read_one(episode / "trajectory_command.csv")
            self.assertEqual(command["tcp_target_y"], "")
            self.assertEqual(command["delta_x_R"], "0.0")

            snapshot = episode_logger___read_one(episode / "aligned_snapshot.csv")
            self.assertEqual(snapshot["state_wrench_skew_s"], "0.004")
            self.assertEqual(snapshot["invalid_reason"], (
                "joint_positions_unavailable;cartesian_force_unavailable"
            ))

            metadata = episode_logger__json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertIsNone(metadata["git_commit"])
            self.assertIsNone(metadata["operator_confirmation"])
            self.assertEqual(metadata["logger"]["status"], "completed")
            self.assertEqual(metadata["logger"]["timestamp_source"], episode_logger__TIMESTAMP_SOURCE)
            self.assertEqual(
                metadata["logger"]["row_counts"],
                {
                    "aligned_snapshot": 1,
                    "robot_state": 1,
                    "robot_wrench": 1,
                    "trajectory_command": 1,
                },
            )
            self.assertGreater(metadata["logger"]["episode_duration_s"], 0.0)
            self.assertGreater(
                metadata["logger"]["average_stream_publish_rate_hz"]["robot_state"],
                0.0,
            )
            self.assertEqual(list(episode.glob(".metadata.*.tmp")), [])

    def test_append_error_marks_logger_failed_and_blocks_future_writes(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_0002"
            logger = episode_logger__EpisodeLogger(episode, clock=episode_logger___DeterministicClock()).start()

            with self.assertRaisesRegex(episode_logger__EpisodeLoggerError, "unexpected_field"):
                logger.append_robot_state(
                    host_time_s=2.0,
                    valid=True,
                    invalid_reason="",
                    unexpected_field=123,
                )

            self.assertEqual(logger.status, "failed")
            self.assertFalse(logger.healthy)
            self.assertTrue(logger.failure_event.is_set())
            self.assertFalse(logger.wait_until_ready(timeout=0.0))
            self.assertIn("unexpected_field", logger.failure_reason or "")
            with self.assertRaises(episode_logger__EpisodeLoggerError):
                logger.append_robot_wrench(valid=False, invalid_reason="blocked")

            metadata = episode_logger__json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failed_stream"], "robot_state")
            self.assertEqual(metadata["logger"]["row_counts"]["robot_state"], 0)
            self.assertEqual(list(episode.glob(".metadata.*.tmp")), [])

    def test_startup_collision_fails_without_overwriting_existing_data(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_existing"
            episode.mkdir()
            state_path = episode / "robot_state.csv"
            metadata_path = episode / "metadata.json"
            state_path.write_text("do-not-overwrite\n", encoding="utf-8")
            metadata_path.write_text('{"owner":"existing"}\n', encoding="utf-8")
            logger = episode_logger__EpisodeLogger(episode, clock=episode_logger___DeterministicClock())

            with self.assertRaisesRegex(episode_logger__EpisodeLoggerError, "already exist"):
                logger.start()

            self.assertEqual(state_path.read_text(encoding="utf-8"), "do-not-overwrite\n")
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"owner":"existing"}\n',
            )
            self.assertEqual(logger.status, "failed")
            self.assertTrue(logger.failed)
            self.assertFalse(logger.wait_until_ready(timeout=0.0))

            logger.close(completed=False, stop_reason="startup cleanup")
            self.assertEqual(state_path.read_text(encoding="utf-8"), "do-not-overwrite\n")
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"owner":"existing"}\n',
            )
            self.assertEqual(list(episode.glob(".metadata.*.tmp")), [])

    def test_csv_only_collision_close_does_not_create_metadata(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_csv_collision"
            episode.mkdir()
            state_path = episode / "robot_state.csv"
            state_path.write_text("foreign-state\n", encoding="utf-8")
            logger = episode_logger__EpisodeLogger(episode, clock=episode_logger___DeterministicClock())

            with self.assertRaisesRegex(episode_logger__EpisodeLoggerError, "already exist"):
                logger.start()
            logger.close(completed=False)

            self.assertEqual(state_path.read_text(encoding="utf-8"), "foreign-state\n")
            self.assertFalse((episode / "metadata.json").exists())

    def test_completed_writer_ack_is_not_delayed_by_pending_cleanup(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            logger = episode_logger__EpisodeLogger(
                episode_logger__Path(root) / "episode_cleanup_gate",
                clock=episode_logger___DeterministicClock(),
            ).start()
            cleanup_gate = episode_logger___WriterCleanupGate()
            logger._pending_write_lock = cleanup_gate
            caller_done = episode_logger__threading.Event()
            errors: list[BaseException] = []

            def append_bounded() -> None:
                try:
                    logger.append_trajectory_command_bounded(
                        timeout_s=1.0,
                        host_time_s=1.0,
                        command_valid=True,
                        invalid_reason="",
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    caller_done.set()

            caller = episode_logger__threading.Thread(target=append_bounded, name="bounded-caller")
            caller.start()
            try:
                self.assertTrue(cleanup_gate.writer_entered.wait(timeout=1.0))
                # Durable completion wins the handshake before the worker can
                # block in pending-set bookkeeping.
                self.assertTrue(caller_done.wait(timeout=0.2))
                self.assertEqual(errors, [])
                self.assertFalse(logger.failure_event.is_set())
            finally:
                cleanup_gate.release_writer.set()
                caller.join(timeout=1.0)
            self.assertFalse(caller.is_alive())
            self.assertTrue(logger.wait_for_pending_writes(1.0))
            logger.close(completed=True)

    def test_bounded_timeout_signals_without_logger_lock_and_blocks_later_append(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_bounded_timeout"
            logger = episode_logger__EpisodeLogger(episode, clock=episode_logger___DeterministicClock()).start()
            flush_entered = episode_logger__threading.Event()
            release_flush = episode_logger__threading.Event()
            caller_done = episode_logger__threading.Event()
            errors: list[BaseException] = []

            def blocking_flush() -> None:
                flush_entered.set()
                if not release_flush.wait(timeout=2.0):
                    raise TimeoutError("test did not release flush")

            def append_bounded() -> None:
                try:
                    logger.append_trajectory_command_bounded(
                        timeout_s=0.05,
                        host_time_s=1.0,
                        command_valid=True,
                        invalid_reason="",
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    caller_done.set()

            with episode_logger__patch.object(logger._handles["trajectory_command"], "flush", side_effect=blocking_flush):
                caller = episode_logger__threading.Thread(target=append_bounded, name="bounded-caller")
                caller.start()
                self.assertTrue(flush_entered.wait(timeout=1.0))
                self.assertTrue(caller_done.wait(timeout=1.0))
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], episode_logger__EpisodeLoggerError)
                self.assertTrue(logger.write_timeout_event.is_set())
                self.assertTrue(logger.failure_event.is_set())
                self.assertFalse(logger.healthy_signal)

                later_done = episode_logger__threading.Event()
                later_errors: list[BaseException] = []

                def append_after_timeout() -> None:
                    try:
                        logger.append_robot_state(valid=True, invalid_reason="")
                    except BaseException as exc:
                        later_errors.append(exc)
                    finally:
                        later_done.set()

                later = episode_logger__threading.Thread(target=append_after_timeout)
                later.start()
                # This must finish while flush still owns the logger I/O lock.
                self.assertTrue(later_done.wait(timeout=0.2))
                self.assertEqual(len(later_errors), 1)
                self.assertIsInstance(later_errors[0], episode_logger__EpisodeLoggerError)

                release_flush.set()
                caller.join(timeout=1.0)
                later.join(timeout=1.0)
                self.assertTrue(logger.wait_for_pending_writes(1.0))

            self.assertEqual(logger.status, "failed")
            logger.close(completed=False)
            metadata = episode_logger__json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failed_stream"], "trajectory_command")

    def test_signal_failure_returns_while_flush_holds_logger_lock(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_signal_failure"
            logger = episode_logger__EpisodeLogger(episode, clock=episode_logger___DeterministicClock()).start()
            flush_entered = episode_logger__threading.Event()
            release_flush = episode_logger__threading.Event()
            append_done = episode_logger__threading.Event()
            signal_done = episode_logger__threading.Event()

            def blocking_flush() -> None:
                flush_entered.set()
                if not release_flush.wait(timeout=2.0):
                    raise TimeoutError("test did not release flush")

            def append_command() -> None:
                try:
                    logger.append_trajectory_command(
                        host_time_s=1.0,
                        command_valid=True,
                        invalid_reason="",
                    )
                finally:
                    append_done.set()

            def publish_failure() -> None:
                logger.signal_failure("sensor_dead", stream="wrench")
                signal_done.set()

            with episode_logger__patch.object(logger._handles["trajectory_command"], "flush", side_effect=blocking_flush):
                writer = episode_logger__threading.Thread(target=append_command)
                writer.start()
                self.assertTrue(flush_entered.wait(timeout=1.0))
                signaler = episode_logger__threading.Thread(target=publish_failure)
                signaler.start()
                self.assertTrue(signal_done.wait(timeout=0.2))
                self.assertTrue(logger.failure_event.is_set())
                self.assertFalse(logger.healthy_signal)
                release_flush.set()
                writer.join(timeout=1.0)
                signaler.join(timeout=1.0)
                self.assertTrue(append_done.is_set())

            logger.close(completed=False)
            metadata = episode_logger__json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failure_reason"], "sensor_dead")
            self.assertEqual(metadata["logger"]["failed_stream"], "wrench")


if __name__ == "__main__":
    episode_logger__unittest.main()



# ============================================================================
# Source: tests/test_leg_episode_import.py
# ============================================================================

"""Regression tests for the TwinCAT dummy-leg episode importer.

The fixtures mirror the 2026-10-07 acquisition exactly enough to keep the two
real traps visible: axis-sequential reads at ~2.6 Hz, and a static torque
offset that moves with pose.  No fixture is presented as identification data.
"""



import json as leg_episode_import__json
from pathlib import Path as leg_episode_import__Path

import numpy as leg_episode_import__np
import pytest as leg_episode_import__pytest

from measurement_validation.leg_episode import (
    CalibrationError as leg_episode_import__CalibrationError,
    JointCalibration as leg_episode_import__JointCalibration,
    LegImportError as leg_episode_import__LegImportError,
    MOTOR_SIDE_DEG_PER_COUNT as leg_episode_import__MOTOR_SIDE_DEG_PER_COUNT,
    quality_report as leg_episode_import__quality_report,
    read_leg_episode as leg_episode_import__read_leg_episode,
)
from measurement_validation.leg_episode import main as leg_episode_import__import_main


def leg_episode_import___write_episode(path: leg_episode_import__Path, rows: int, rate_hz: float, *, atomic: bool = False) -> leg_episode_import__Path:
    """One CSV with the column names the handover record described."""

    times = leg_episode_import__np.arange(rows, dtype=float) / rate_hz
    # Hip flexion decreases counts; knee flexion increases counts.
    axis1_counts = 100_000.0 - leg_episode_import__np.linspace(0.0, 5_000.0, rows)
    axis4_counts = 200_000.0 + leg_episode_import__np.linspace(0.0, 9_000.0, rows)
    axis1_torque = leg_episode_import__np.full(rows, -34.5872)
    axis4_torque = leg_episode_import__np.full(rows, 2.6403)
    sync = "atomic_task_buffer" if atomic else "not_atomic_axes_read_sequentially"
    lines = ["host_time_s,sync_quality,axis1_position_raw,axis4_position_raw,"
             "axis1_velocity_606c,axis4_velocity_606c,axis1_torque_3c1a,axis4_torque_3c1a,"
             "axis1_torque_6077,axis4_torque_6077,axis1_statusword_6041,axis4_statusword_6041,"
             "axis1_error_603f,axis4_error_603f,axis1_brake_override_381e,axis4_brake_override_381e"]
    for index in range(rows):
        lines.append(
            f"{times[index]:.6f},{sync},{axis1_counts[index]:.0f},{axis4_counts[index]:.0f},"
            f"0,0,{axis1_torque[index]:.4f},{axis4_torque[index]:.4f},"
            f"0,0,561,561,0,0,{128 if not atomic else 0},{128 if not atomic else 0}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def leg_episode_import___calibration() -> dict[str, leg_episode_import__JointCalibration]:
    # Counts-per-degree and zero are illustrative only, never presented as the
    # real mechanism calibration.
    return {
        "hip": leg_episode_import__JointCalibration(counts_per_degree=1000.0, zero_counts=100_000.0,
                                flexion_sign=-1, safe_min_deg=-20.0, safe_max_deg=90.0),
        "knee": leg_episode_import__JointCalibration(counts_per_degree=1000.0, zero_counts=200_000.0,
                                 flexion_sign=+1, safe_min_deg=0.0, safe_max_deg=120.0),
    }


def test_leg_episode_import__axis_binding_does_not_cross_between_axis_one_and_four(tmp_path: leg_episode_import__Path):
    """A bare ``1`` inside ``3c1a`` must not steal axis 4's torque column."""

    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 12, 2.6))
    assert episode.joints["hip"].signals["torque_nm"] == "axis1_torque_3c1a"
    assert episode.joints["knee"].signals["torque_nm"] == "axis4_torque_3c1a"
    assert episode.joints["hip"].signals["position"] == "axis1_position_raw"
    assert episode.joints["knee"].signals["position"] == "axis4_position_raw"


def test_leg_episode_import__underscore_alias_names_still_resolve(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 5, 2.6))
    assert leg_episode_import__np.allclose(episode.host_time_s, leg_episode_import__np.arange(5) / 2.6)
    assert episode.sync_quality == "not_atomic_axes_read_sequentially"


def test_leg_episode_import__joint_angle_needs_a_real_calibration(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 5, 2.6))
    assert episode.joints["hip"].flexion_deg is None
    # Motor-side degrees are recorded but never promoted to a joint angle.
    assert episode.joints["hip"].motor_position_deg is not None


def test_leg_episode_import__calibration_converts_counts_to_signed_flexion(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 3, 2.6),
                               calibration=leg_episode_import___calibration())
    hip = episode.joints["hip"].flexion_deg
    knee = episode.joints["knee"].flexion_deg
    assert hip[0] == leg_episode_import__pytest.approx(0.0)
    assert hip[-1] > 0.0          # hip counts fall during flexion, sign -1
    assert knee[0] == leg_episode_import__pytest.approx(0.0)
    assert knee[-1] > 0.0         # knee counts rise during flexion, sign +1


def test_leg_episode_import__missing_calibration_entry_is_rejected(tmp_path: leg_episode_import__Path):
    calibration = leg_episode_import___calibration()
    del calibration["knee"]
    with leg_episode_import__pytest.raises(leg_episode_import__CalibrationError):
        leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 3, 2.6), calibration=calibration)


def test_leg_episode_import__low_rate_and_non_atomic_reads_are_reported(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 27, 2.6))
    report = leg_episode_import__quality_report(episode, minimum_rate_hz=100.0)
    assert report["sample_rate_hz"] == leg_episode_import__pytest.approx(2.6, rel=1e-6)
    assert report["usable_for_dynamics"] is False
    assert report["non_atomic_axis_reads"] is True
    assert any(finding.startswith("sample_rate_below_100hz") for finding in report["findings"])
    assert "axis_channels_not_a_single_instant" in report["findings"]


def test_leg_episode_import__high_rate_alone_does_not_establish_dynamics_readiness(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 101, 200.0, atomic=True))
    report = leg_episode_import__quality_report(episode, minimum_rate_hz=100.0)
    assert report["usable_for_dynamics"] is False
    calibrated = leg_episode_import__read_leg_episode(episode.path, calibration=leg_episode_import___calibration())
    assert leg_episode_import__quality_report(calibrated)["usable_for_dynamics"] is True
    assert report["non_atomic_axis_reads"] is False


def test_leg_episode_import__static_offset_is_descriptive_and_never_tared(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 27, 2.6))
    report = leg_episode_import__quality_report(episode)
    hip = report["joints"]["hip"]["raw_torque_nm"]
    assert hip["mean"] == leg_episode_import__pytest.approx(-34.5872, abs=1e-6)
    assert "never subtracted by default" in report["joints"]["hip"]["torque_offset_note"]


def test_leg_episode_import__time_column_absent_is_rejected(tmp_path: leg_episode_import__Path):
    path = tmp_path / "no_time.csv"
    path.write_text("axis1_position_motor_deg,axis1_torque_3c1a\n1,2\n3,4\n", encoding="utf-8")
    with leg_episode_import__pytest.raises(leg_episode_import__LegImportError):
        leg_episode_import__read_leg_episode(path)


def test_leg_episode_import__column_map_override_is_honoured(tmp_path: leg_episode_import__Path):
    path = tmp_path / "custom.csv"
    path.write_text("clock,hip_counts,knee_counts,hip_torque,knee_torque\n"
                    "0,100,200,1,2\n0.5,110,210,2,3\n", encoding="utf-8")
    column_map = {
        "time": "clock",
        "axis1_position": "hip_counts",
        "axis4_position": "knee_counts",
        "axis1_torque_nm": "hip_torque",
        "axis4_torque_nm": "knee_torque",
    }
    episode = leg_episode_import__read_leg_episode(path, column_map=column_map)
    assert episode.joints["hip"].signals["position"] == "hip_counts"
    assert episode.joints["knee"].signals["torque_nm"] == "knee_torque"


def test_leg_episode_import__motor_side_scale_matches_handover_record(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 3, 2.6))
    counts = episode.joints["hip"].position_counts
    assert leg_episode_import__np.allclose(episode.joints["hip"].motor_position_deg,
                       counts * leg_episode_import__MOTOR_SIDE_DEG_PER_COUNT)


def test_leg_episode_import__json_report_is_serialisable(tmp_path: leg_episode_import__Path):
    episode = leg_episode_import__read_leg_episode(leg_episode_import___write_episode(tmp_path / "ep.csv", 12, 2.6))
    leg_episode_import__json.dumps(leg_episode_import__quality_report(episode), ensure_ascii=False)


def test_leg_episode_import__cli_writes_a_report_without_touching_hardware(tmp_path: leg_episode_import__Path, capsys):
    csv_path = leg_episode_import___write_episode(tmp_path / "ep.csv", 12, 2.6)
    output = tmp_path / "report.json"
    assert leg_episode_import__import_main([str(csv_path), "--minimum-rate-hz", "100", "--output", str(output)]) == 0
    report = leg_episode_import__json.loads(output.read_text(encoding="utf-8"))
    assert report["file"] == "ep.csv"
    assert report["usable_for_dynamics"] is False


def test_leg_episode_import__cli_promotes_joint_angles_only_with_calibration(tmp_path: leg_episode_import__Path):
    csv_path = leg_episode_import___write_episode(tmp_path / "ep.csv", 3, 2.6)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(leg_episode_import__json.dumps({
        "hip": {"counts_per_degree": 1000.0, "zero_counts": 100000.0,
                "flexion_sign": -1, "safe_min_deg": -20.0, "safe_max_deg": 90.0},
        "knee": {"counts_per_degree": 1000.0, "zero_counts": 200000.0,
                 "flexion_sign": 1, "safe_min_deg": 0.0, "safe_max_deg": 120.0},
    }), encoding="utf-8")
    output = tmp_path / "report.json"
    assert leg_episode_import__import_main([str(csv_path), "--calibration", str(calibration_path),
                        "--output", str(output)]) == 0
    report = leg_episode_import__json.loads(output.read_text(encoding="utf-8"))
    assert report["joints"]["hip"]["joint_angle_available"] is True



# ============================================================================
# Source: tests/test_preview_rehab_trajectory.py
# ============================================================================

"""Pure-offline preview regressions."""



import inspect as preview_rehab_trajectory__inspect
import json as preview_rehab_trajectory__json
from pathlib import Path as preview_rehab_trajectory__Path

import pandas as preview_rehab_trajectory__pd
import pytest as preview_rehab_trajectory__pytest

from control.start_anchor import FixedTcpOrientation as preview_rehab_trajectory__FixedTcpOrientation, StartAnchor as preview_rehab_trajectory__StartAnchor
from control.start_anchored_relative_trajectory import (
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID as preview_rehab_trajectory__FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_NOMINAL_ID as preview_rehab_trajectory__MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH as preview_rehab_trajectory__DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory as preview_rehab_trajectory__load_closed_reference_trajectory,
)
from scripts.preview_rehab_trajectory import preview_trajectory as preview_rehab_trajectory__preview_trajectory


def preview_rehab_trajectory___anchor() -> preview_rehab_trajectory__StartAnchor:
    reference, _ = preview_rehab_trajectory__load_closed_reference_trajectory(preview_rehab_trajectory__DEFAULT_REFERENCE_PATH)
    first = reference.iloc[0]
    pose = (0.41, -0.12, 0.36, 0.1, -0.2, 0.3)
    return preview_rehab_trajectory__StartAnchor(
        capture_host_time_s=123.0,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=preview_rehab_trajectory__FixedTcpOrientation(pose[3:]),
        robot_joint_positions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        trajectory_id=preview_rehab_trajectory__FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
        reference_start_q_hip=float(first["q_hip_rad"]),
        reference_start_q_knee=float(first["q_knee_rad"]),
        anchor_id="anchor_preview_test",
        reviewed=False,
        notes="offline test",
    )


def preview_rehab_trajectory___frame_path(tmp_path: preview_rehab_trajectory__Path) -> preview_rehab_trajectory__Path:
    path = tmp_path / "frame.json"
    path.write_text(
        preview_rehab_trajectory__json.dumps(
            {
                "rehab_x_axis_in_base": [1.0, 0.0, 0.0],
                "rehab_z_axis_in_base": [0.0, 0.0, 1.0],
                "reviewed": False,
                "notes": "offline preview draft",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_preview_rehab_trajectory__preview_writes_csv_json_and_four_nonempty_plots(tmp_path: preview_rehab_trajectory__Path) -> None:
    output = tmp_path / "preview"
    summary = preview_rehab_trajectory__preview_trajectory(
        anchor=preview_rehab_trajectory___anchor(),
        frame_config=preview_rehab_trajectory___frame_path(tmp_path),
        output_dir=output,
    )
    assert summary["trajectory_valid"] is True
    assert summary["execution_approved"] is False
    assert summary["frame_reviewed"] is False
    assert summary["anchor_reviewed"] is False
    assert preview_rehab_trajectory__Path(summary["trajectory_csv"]).is_file()
    assert preview_rehab_trajectory__Path(summary["metadata_json"]).is_file()
    assert len(summary["plots"]) == 4
    assert all(preview_rehab_trajectory__Path(path).stat().st_size > 0 for path in summary["plots"])
    trajectory = preview_rehab_trajectory__pd.read_csv(summary["trajectory_csv"])
    assert len(trajectory) == 401
    assert trajectory["trajectory_valid"].all()
    metadata = preview_rehab_trajectory__json.loads(preview_rehab_trajectory__Path(summary["metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["git_commit"]
    assert metadata["preview_only"] is True
    assert metadata["hardware_accessed"] is False
    assert metadata["robot_execution_approved"] is False


def test_preview_rehab_trajectory__preview_module_has_no_hardware_or_motion_dependency() -> None:
    import scripts.preview_rehab_trajectory as module

    source = preview_rehab_trajectory__inspect.getsource(module)
    assert "from hardware" not in source
    assert "import hardware" not in source
    for forbidden in ("enable_realtime", "move_l(", "move_j(", "setPowerState"):
        assert forbidden not in source


def test_preview_rehab_trajectory__preview_rejects_anchor_reference_mismatch(tmp_path: preview_rehab_trajectory__Path) -> None:
    anchor = preview_rehab_trajectory___anchor()
    mismatched = preview_rehab_trajectory__StartAnchor(
        **{**anchor.__dict__, "trajectory_id": preview_rehab_trajectory__MEASURED_ASYMMETRIC_NOMINAL_ID}
    )
    with preview_rehab_trajectory__pytest.raises(ValueError, match="trajectory_id does not match"):
        preview_rehab_trajectory__preview_trajectory(
            anchor=mismatched,
            frame_config=preview_rehab_trajectory___frame_path(tmp_path),
            output_dir=tmp_path / "preview",
        )



# ============================================================================
# Source: tests/test_real_measurement_validation_analysis_v1.py
# ============================================================================



import ast as real_measurement_validation_analysis_v1__ast
import json as real_measurement_validation_analysis_v1__json
from pathlib import Path as real_measurement_validation_analysis_v1__Path

import pytest as real_measurement_validation_analysis_v1__pytest

from measurement_validation.analysis import (
    ANALYSIS_ID as real_measurement_validation_analysis_v1__ANALYSIS_ID,
    analyze_same_trajectory_repeatability as real_measurement_validation_analysis_v1__analyze_same_trajectory_repeatability,
    analyze_static_load as real_measurement_validation_analysis_v1__analyze_static_load,
    analyze_trajectory_sensitivity as real_measurement_validation_analysis_v1__analyze_trajectory_sensitivity,
    build_demo_input as real_measurement_validation_analysis_v1__build_demo_input,
    extract_episode_features as real_measurement_validation_analysis_v1__extract_episode_features,
    run_validation_analysis as real_measurement_validation_analysis_v1__run_validation_analysis,
)
from scripts.run_real_measurement_validation_analysis import main as real_measurement_validation_analysis_v1__main


real_measurement_validation_analysis_v1__ROOT = real_measurement_validation_analysis_v1__Path(__file__).resolve().parents[1]


def real_measurement_validation_analysis_v1___static_cell(repeat_id: int = 1):
    phase_vectors = {
        "PRE": ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        "LOAD": ([5.0, 8.0, 11.0], [1.0, 2.0, 3.0]),
        "POST": ([3.0, 4.0, 5.0], [0.2, 0.2, 0.2]),
    }
    rows = []
    for phase, (force, torque) in phase_vectors.items():
        rows.append(
            {
                "session_id": "SESSION_A",
                "pose_id": "POSE_A",
                "direction_id": "KNOWN_DIRECTION",
                "load_level_id": "LOAD_A",
                "repeat_id": repeat_id,
                "cell_id": f"CELL_{repeat_id}",
                "phase": phase,
                "fx_raw_n": force[0],
                "fy_raw_n": force[1],
                "fz_raw_n": force[2],
                "mx": torque[0],
                "my": torque[1],
                "mz": torque[2],
                "reference_load_vector_n": [3.0, 5.0, 7.0],
                "query_start_s": 1.0,
                "query_end_s": 1.004,
                "state_host_time_s": 1.0,
                "tcp_x_m": 0.3,
                "tcp_y_m": 0.0,
                "tcp_z_m": 0.4,
                "q1_rad": 0.1,
                "valid": True,
                "robot_state_valid": True,
                "invalid_reason": "",
            }
        )
    return rows


def real_measurement_validation_analysis_v1___episode(beta=(0.0, 0.0), repeat=0, *, tool="TOOL_A"):
    samples = []
    for index, phase in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        samples.append(
            {
                "time_s": index * 0.1,
                "phase": phase,
                "branch": "FLEXION" if phase < 0.5 else "EXTENSION",
                "fx": 1.0 + repeat * 0.01,
                "fy": 2.0,
                "fz": 3.0 + beta[0] * 10.0 + phase,
                "mx": 0.1,
                "my": 0.2,
                "mz": 0.3,
                "hip_response": 4.0 + phase,
                "knee_response": 5.0 - phase,
                "joint_measured_torque_1": 0.4 + phase,
                "joint_external_torque_1": 0.3 + phase,
                "valid": True,
                "state_valid": True,
                "invalid_reason": "",
            }
        )
    return {
        "episode_id": f"episode_{beta[0]}_{beta[1]}_{repeat}",
        "entity_id": "DUMMY_A",
        "entity_kind": "DUMMY_LEG",
        "beta_flex": beta[0],
        "beta_extend": beta[1],
        "rom_profile_id": "ROM_A",
        "rom_profile_version": "V1",
        "hardware_setup_id": "SETUP_A",
        "measurement_source": "RECORDED_LOG",
        "robot_model": "R",
        "robot_serial_number": "S",
        "controller_version": "C",
        "active_tool_name": tool,
        "active_workobject_name": "W",
        "tcp_id": "TCP_A",
        "payload_id": "PAYLOAD_A",
        "samples": samples,
    }


def test_real_measurement_validation_analysis_v1__pre_load_post_keeps_raw_values_and_calculates_force_and_torque_contrast():
    result = real_measurement_validation_analysis_v1__analyze_static_load(real_measurement_validation_analysis_v1___static_cell())
    cell = result["cells"][0]
    assert cell["pre_mean_force_n"] == real_measurement_validation_analysis_v1__pytest.approx([1.0, 2.0, 3.0])
    assert cell["load_mean_force_n"] == real_measurement_validation_analysis_v1__pytest.approx([5.0, 8.0, 11.0])
    assert cell["post_mean_force_n"] == real_measurement_validation_analysis_v1__pytest.approx([3.0, 4.0, 5.0])
    assert cell["baseline_bias_force_n"] == real_measurement_validation_analysis_v1__pytest.approx([2.0, 3.0, 4.0])
    assert cell["drift_post_minus_pre_force_n"] == real_measurement_validation_analysis_v1__pytest.approx([2.0, 2.0, 2.0])
    assert cell["measured_load_vector_n"] == real_measurement_validation_analysis_v1__pytest.approx([3.0, 5.0, 7.0])
    assert cell["load_delta_torque_nm"] == real_measurement_validation_analysis_v1__pytest.approx([0.9, 1.9, 2.9])
    assert cell["sign_agreement_by_component"] == {"fx": True, "fy": True, "fz": True}
    assert cell["direction_cosine_agreement"] == real_measurement_validation_analysis_v1__pytest.approx(1.0)
    assert cell["raw_phase_samples"]["PRE"][0]["force_n"] == [1.0, 2.0, 3.0]
    assert result["criterion_status"] == "INSUFFICIENT"


def test_real_measurement_validation_analysis_v1__missing_and_invalid_samples_remain_missing_and_are_counted():
    rows = real_measurement_validation_analysis_v1___static_cell()
    rows[0]["fx_raw_n"] = None
    rows[0]["invalid_reason"] = "force_missing"
    rows[2]["valid"] = False
    rows[2]["invalid_reason"] = "wrench_invalid"
    result = real_measurement_validation_analysis_v1__analyze_static_load(rows)
    cell = result["cells"][0]
    assert cell["raw_phase_samples"]["PRE"][0]["force_n"][0] is None
    assert cell["raw_phase_samples"]["PRE"][0]["valid"] is False
    assert result["usable_record_count"] == 1
    assert result["missing_or_invalid_record_count"] == 2
    assert result["invalid_reason_counts"] == {"force_missing": 1, "wrench_invalid": 1}
    assert cell["complete_pre_load_post"] is False
    assert cell["measured_load_vector_n"] is None


def test_real_measurement_validation_analysis_v1__static_repeatability_groups_repeats_without_hardcoded_pass_thresholds():
    rows = real_measurement_validation_analysis_v1___static_cell(1) + real_measurement_validation_analysis_v1___static_cell(2)
    rows[-2]["fz_raw_n"] += 0.2
    result = real_measurement_validation_analysis_v1__analyze_static_load(rows)
    repeatability = result["known_load_contrast_repeatability"]
    assert len(repeatability) == 1
    assert repeatability[0]["repeat_count"] == 2
    assert repeatability[0]["component_repeatability_standard_deviation_n"] is not None
    assert result["criteria_source"] == "NOT_SUPPLIED"
    assert result["criterion_checks"] == []


def test_real_measurement_validation_analysis_v1__episode_features_preserve_branches_time_local_data_and_provenance():
    raw = real_measurement_validation_analysis_v1___episode()
    result = real_measurement_validation_analysis_v1__extract_episode_features(raw)
    assert result["features"]["fx"]["full_cycle_rms"] == real_measurement_validation_analysis_v1__pytest.approx(1.0)
    assert result["features"]["fz"]["flexion_rms"] != result["features"]["fz"]["extension_rms"]
    assert result["features"]["fz"]["peak_abs"] == real_measurement_validation_analysis_v1__pytest.approx(4.0)
    assert result["time_local_curves"]["fz"] is not None
    assert len(result["time_local_curves"]["fz"]) == 51
    assert result["channel_provenance"]["fx"] == "ROBOT_REPORTED_RAW_CHANNEL"
    assert result["channel_provenance"]["hip_response"] == "MODEL_DERIVED"
    assert result["channel_provenance"]["joint_measured_torque_1"] == (
        "ROBOT_REPORTED_RAW_CHANNEL"
    )
    assert result["channel_provenance"]["joint_external_torque_1"] == "MODEL_DERIVED"
    assert "task_direction_projection" not in result["features"]
    assert result["task_direction_projection"]["available"] is False

    projected = real_measurement_validation_analysis_v1__extract_episode_features(
        raw,
        task_direction={
            "vector": [0.0, 1.0, 0.0],
            "frame": "world",
            "source": "reviewed_geometry",
            "geometry_validated": False,
        },
    )
    assert projected["features"]["task_direction_projection"]["full_cycle_rms"] == real_measurement_validation_analysis_v1__pytest.approx(2.0)
    assert projected["task_direction_projection"]["decision_eligible"] is False


def test_real_measurement_validation_analysis_v1__same_beta_grouping_and_metadata_consistency_are_explicit():
    features = [real_measurement_validation_analysis_v1__extract_episode_features(real_measurement_validation_analysis_v1___episode(repeat=index)) for index in range(3)]
    criteria = {
        "minimum_repeats_per_group": 3,
        "minimum_valid_episode_fraction": 1.0,
        "require_metadata_consistency": True,
        "decision_feature_ids": ["fz.full_cycle_rms"],
        "maximum_feature_cv": 0.01,
        "minimum_time_series_correlation": 0.99,
    }
    result = real_measurement_validation_analysis_v1__analyze_same_trajectory_repeatability(features, criteria)
    assert result["criterion_status"] == "SUPPORTED"
    assert result["same_condition_group_count"] == 1
    assert result["groups"][0]["repeat_count"] == 3
    assert result["groups"][0]["metadata_consistent"] is True
    assert result["groups"][0]["time_series_similarity"]["fz"]["pair_count"] == 3

    inconsistent = features[:2] + [real_measurement_validation_analysis_v1__extract_episode_features(real_measurement_validation_analysis_v1___episode(repeat=2, tool="TOOL_B"))]
    failed = real_measurement_validation_analysis_v1__analyze_same_trajectory_repeatability(inconsistent, criteria)
    assert failed["criterion_status"] == "FAILED"
    assert failed["groups"][0]["metadata_consistent"] is False
    assert failed["groups"][0]["inconsistent_setup_metadata_fields"] == [
        "active_tool_name"
    ]


def test_real_measurement_validation_analysis_v1__different_beta_effect_uses_within_variation_and_retains_time_local_response():
    demo = real_measurement_validation_analysis_v1__build_demo_input()
    features = [real_measurement_validation_analysis_v1__extract_episode_features(episode) for episode in demo["episodes"]]
    result = real_measurement_validation_analysis_v1__analyze_trajectory_sensitivity(features, demo["criteria"]["level_3"])
    assert result["criterion_status"] == "SUPPORTED"
    assert result["analyzable_context_count"] == 1
    context = result["contexts"][0]
    assert context["beta_condition_count"] == 3
    by_feature = {row["feature_id"]: row for row in context["feature_effects"]}
    assert by_feature["fz.full_cycle_rms"]["pooled_within_trajectory_standard_deviation"] > 0.0
    assert by_feature["fz.full_cycle_rms"]["trajectory_snr"] > 2.0
    assert "fz.flexion_rms" in by_feature
    assert "fz.extension_rms" in by_feature
    local = {row["channel"]: row for row in context["time_local_effects"]}["fz"]
    assert local["maximum_time_local_effect"] > 0.6
    assert 0.2 < local["phase_at_maximum_effect"] < 0.5
    assert local["time_local_snr"] > 2.0


def test_real_measurement_validation_analysis_v1__hierarchy_and_evidence_classification_prevent_automatic_personalization():
    demo = real_measurement_validation_analysis_v1__build_demo_input()
    result = real_measurement_validation_analysis_v1__run_validation_analysis(demo)
    assert result[real_measurement_validation_analysis_v1__ANALYSIS_ID] == "IMPLEMENTED_WITH_LIMITATIONS"
    assert result["criterion_statuses_before_evidence_classification"] == {
        "MEASUREMENT_VALIDITY": "SUPPORTED",
        "SAME_TRAJECTORY_REPEATABILITY": "SUPPORTED",
        "TRAJECTORY_SENSITIVITY": "SUPPORTED",
    }
    assert result["MEASUREMENT_VALIDITY"] == "INSUFFICIENT"
    assert result["SAME_TRAJECTORY_REPEATABILITY"] == "INSUFFICIENT"
    assert result["TRAJECTORY_SENSITIVITY"] == "INSUFFICIENT"
    assert result["FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY"] is False
    assert result["personalization_conclusion_emitted"] is False
    assert "PERSONALIZATION_SUPPORTED" not in result
    assert result["robot_connection_attempted"] is False
    assert result["robot_action_count"] == 0

    no_static = {**demo, "evidence_classification": "REAL_MEASUREMENT", "static_records": []}
    blocked = real_measurement_validation_analysis_v1__run_validation_analysis(no_static)
    assert blocked["MEASUREMENT_VALIDITY"] == "INSUFFICIENT"
    assert blocked["level_2"]["blocked_by"] == "MEASUREMENT_VALIDITY"
    assert blocked["level_3"]["blocked_by"] == "MEASUREMENT_VALIDITY"


def test_real_measurement_validation_analysis_v1__real_classification_can_open_only_the_future_evaluation_gate():
    payload = real_measurement_validation_analysis_v1__build_demo_input()
    payload["evidence_classification"] = "REAL_MEASUREMENT"
    result = real_measurement_validation_analysis_v1__run_validation_analysis(payload)
    assert result["MEASUREMENT_VALIDITY"] == "SUPPORTED"
    assert result["SAME_TRAJECTORY_REPEATABILITY"] == "SUPPORTED"
    assert result["TRAJECTORY_SENSITIVITY"] == "SUPPORTED"
    assert result["FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY"] is True
    assert result["personalization_conclusion_emitted"] is False


def test_real_measurement_validation_analysis_v1__demo_runner_writes_reports_and_new_code_has_no_robot_runtime_import(tmp_path):
    output = tmp_path / "demo"
    assert real_measurement_validation_analysis_v1__main(["--demo", "--output-dir", str(output)]) == 0
    summary = real_measurement_validation_analysis_v1__json.loads((output / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_classification"] == "OFFLINE_SYNTHETIC_DEMO_ONLY"
    assert summary["not_real_measurement_evidence"] is True
    required = {
        "validation_metrics.csv",
    }
    assert all((output / filename).stat().st_size > 0 for filename in required)

    forbidden_import_roots = {
        "hardware",
        "control",
        "collection",
        "personalization",
        "safety",
    }
    for path in (
        real_measurement_validation_analysis_v1__ROOT / "measurement_validation" / "analysis.py",
        real_measurement_validation_analysis_v1__ROOT / "scripts" / "run_real_measurement_validation_analysis.py",
    ):
        imported = set()
        for node in real_measurement_validation_analysis_v1__ast.walk(real_measurement_validation_analysis_v1__ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, real_measurement_validation_analysis_v1__ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, real_measurement_validation_analysis_v1__ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])
        assert not imported & forbidden_import_roots



# ============================================================================
# Source: tests/test_real_robot_acquisition.py
# ============================================================================

"""Offline concurrency tests for independent real-episode acquisition."""



import threading as real_robot_acquisition__threading
import time as real_robot_acquisition__time
from types import SimpleNamespace as real_robot_acquisition__SimpleNamespace

import pytest as real_robot_acquisition__pytest

from collection.episode_logger import EpisodeLogger as real_robot_acquisition__EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition as real_robot_acquisition__RealRobotAcquisition
from collection.state import KinematicStateFrame as real_robot_acquisition__KinematicStateFrame
from hardware.rokae_adapter import RobotWrenchFrame as real_robot_acquisition__RobotWrenchFrame
from utils.clock import host_time_s as real_robot_acquisition__host_time_s


class real_robot_acquisition__FakeIPCProvider:
    """Consumer/lifecycle unit double; native isolation tested with OS children
    in test_wrench_production_process.py. No SDK or shared parent adapter.
    """
    def __init__(self, adapter):
        self.adapter = adapter
        self.config = {'mode':'offline','rate_hz':50.0}
        self.budgets = real_robot_acquisition__SimpleNamespace(state_age_s=1.,sample_age_s=1.,skew_s=1.)
        self.process = None
        self.last_good = None
        self.failures = []
        self.sequence = 0

    def start(self):
        return self

    def poll(self):
        self.adapter.wrench_entered.set()
        if self.adapter.block_wrench:
            self.adapter.release_wrench.wait(timeout=1.)
        self.sequence += 1
        now=real_robot_acquisition__host_time_s()
        self.last_good=real_robot_acquisition__RobotWrenchFrame(self.sequence,now,now,now,now,'','offline',True,'','world',
                                        (1.,2.,3.),(.1,.2,.3),(1.,)*6,(.5,)*6)
        return self.health()

    def health(self):
        return dict(STREAM_HEALTH=not self.failures,PROCESS_ALIVE=True,
                    FAILURE_LATCH=tuple(self.failures),CURRENT_QUERY_STATE='SUCCEEDED')

    def fail(self,reason):
        self.failures.append(reason)

    def stop(self):
        return {}


@real_robot_acquisition__pytest.fixture(autouse=False)
def real_robot_acquisition__inject_offline_consumer_provider(monkeypatch):
    original=real_robot_acquisition__RealRobotAcquisition.__init__
    def init(self,adapter,logger,**kwargs):
        kwargs.setdefault('wrench_provider',real_robot_acquisition__FakeIPCProvider(adapter))
        return original(self,adapter,logger,**kwargs)
    monkeypatch.setattr(real_robot_acquisition__RealRobotAcquisition,'__init__',init)


class real_robot_acquisition__FakeAdapter:
    def __init__(self, *, block_wrench: bool = False) -> None:
        self.connected = False
        self.sequence = 0
        self.calls: list[str] = []
        self.block_wrench = block_wrench
        self.wrench_entered = real_robot_acquisition__threading.Event()
        self.release_wrench = real_robot_acquisition__threading.Event()

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False

    def is_connected(self):
        return self.connected

    def start_state_stream(self):
        self.calls.append("start_state_stream")

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")

    def read_state_frame(self):
        self.sequence += 1
        now = real_robot_acquisition__host_time_s()
        return real_robot_acquisition__KinematicStateFrame(
            sequence_id=self.sequence,
            host_monotonic_time_s=now,
            wall_time_iso=None,
            robot_device_time_s=None,
            valid=True,
            invalid_reason="",
            tcp_position_m=(0.4, 0.0, 0.3),
            tcp_orientation_rad=(0.1, 0.2, 0.3),
            tcp_linear_velocity_mps=None,
            tcp_angular_velocity_radps=None,
            velocity_source="not_available",
            joint_position_rad=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            joint_velocity_radps=None,
            pose_time_s=now,
            joint_time_s=now,
            velocity_time_s=None,
            operation_state="IDLE",
            collision_state=None,
            controller_error=None,
        )

    def read_internal_wrench(self):
        self.wrench_entered.set()
        if self.block_wrench:
            self.release_wrench.wait(timeout=1.0)
        start = real_robot_acquisition__host_time_s()
        end = real_robot_acquisition__host_time_s()
        return real_robot_acquisition__RobotWrenchFrame(
            sequence_id=1,
            host_query_start_s=start,
            host_query_end_s=end,
            host_publish_s=end,
            host_monotonic_time_s=(start + end) / 2.0,
            wall_time_iso="",
            timestamp_source="host_query_time_no_robot_device_timestamp",
            valid=True,
            invalid_reason="",
            raw_force_frame="world",
            cartesian_force_raw_n=(1.0, 2.0, 3.0),
            cartesian_torque_raw_nm=(0.1, 0.2, 0.3),
            joint_measured_torque_nm=(1.0,) * 6,
            joint_external_torque_nm=(0.5,) * 6,
        )


class real_robot_acquisition___SignalOnlyLogger:
    """Logger double that rejects every lock-taking failure path."""

    def __init__(self) -> None:
        self.failure_event = real_robot_acquisition__threading.Event()
        self.signals: list[tuple[str, str]] = []

    def assert_healthy(self) -> None:
        return None

    @property
    def healthy(self):
        raise AssertionError("stop must not inspect logger.healthy")

    def mark_failed(self, _reason: str) -> None:
        raise AssertionError("stop must not call lock-taking mark_failed")

    def signal_failure(self, reason: str, *, stream: str = "external", **_values) -> None:
        self.signals.append((reason, stream))
        self.failure_event.set()

    def append_robot_state(self, **_values) -> None:
        return None

    def append_robot_wrench(self, **_values) -> None:
        return None

    def append_aligned_snapshot(self, **_values) -> None:
        return None


class real_robot_acquisition___LifecycleLogger(real_robot_acquisition___SignalOnlyLogger):
    def __init__(self) -> None:
        super().__init__()
        self.durable_failures: list[str] = []

    @property
    def healthy(self):
        return not self.failure_event.is_set()

    def mark_failed(self, reason: str) -> None:
        self.durable_failures.append(reason)


class real_robot_acquisition___ControlledThread:
    def __init__(
        self,
        *,
        name: str,
        fail_start: bool,
        remain_alive_after_join: bool,
        stop_event: real_robot_acquisition__threading.Event,
    ) -> None:
        self.name = name
        self.fail_start = fail_start
        self.remain_alive_after_join = remain_alive_after_join
        self.stop_event = stop_event
        self.start_called = False
        self.join_called = False
        self.stop_was_set_at_join = False
        self._alive = False

    def start(self) -> None:
        self.start_called = True
        if self.fail_start:
            raise RuntimeError("synthetic start failure")
        self._alive = True

    def join(self, timeout=None) -> None:
        self.join_called = True
        self.stop_was_set_at_join = self.stop_event.is_set()
        if not self.remain_alive_after_join:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def release(self) -> None:
        self._alive = False


class real_robot_acquisition___ControlledThreadFactory:
    def __init__(self, *, stuck_state: bool) -> None:
        self.stuck_state = stuck_state
        self.stop_event: real_robot_acquisition__threading.Event | None = None
        self.created: dict[str, real_robot_acquisition___ControlledThread] = {}

    def __call__(self, *, target, name: str, daemon: bool):
        assert self.stop_event is not None
        thread = real_robot_acquisition___ControlledThread(
            name=name,
            fail_start=name == "real-episode-wrench",
            remain_alive_after_join=(
                self.stuck_state and name == "real-episode-state"
            ),
            stop_event=self.stop_event,
        )
        self.created[name] = thread
        return thread


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__independent_streams_publish_and_health_is_visible(tmp_path):
    logger = real_robot_acquisition__EpisodeLogger(tmp_path / "episode").start()
    adapter = real_robot_acquisition__FakeAdapter()
    acquisition = real_robot_acquisition__RealRobotAcquisition(adapter, logger).start()
    try:
        assert acquisition.wait_until_healthy(1.0)
        health = acquisition.latest_health()
        assert health.valid
        assert health.state_age_s is not None
        assert health.wrench_age_s is not None
        assert health.state_wrench_skew_s is not None
        assert logger.row_counts["robot_state"] > 0
        assert logger.row_counts["robot_wrench"] > 0
        deadline = real_robot_acquisition__time.perf_counter() + 1.0
        while logger.row_counts["aligned_snapshot"] == 0 and real_robot_acquisition__time.perf_counter() < deadline:
            real_robot_acquisition__time.sleep(.001)
        assert logger.row_counts["aligned_snapshot"] > 0
    finally:
        acquisition.stop()
        logger.close(completed=True)
    assert adapter.calls == [
        "connect",
        "start_state_stream",
        "stop_state_stream",
        "disconnect",
    ]


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__blocking_wrench_does_not_block_state_or_command_logging(tmp_path):
    logger = real_robot_acquisition__EpisodeLogger(tmp_path / "episode").start()
    adapter = real_robot_acquisition__FakeAdapter(block_wrench=True)
    acquisition = real_robot_acquisition__RealRobotAcquisition(adapter, logger).start()
    assert adapter.wrench_entered.wait(timeout=0.5)
    try:
        real_robot_acquisition__time.sleep(0.04)
        assert logger.row_counts["robot_state"] >= 2
        logger.append_trajectory_command(
            host_time_s=real_robot_acquisition__host_time_s(),
            trajectory_time_s=0.0,
            trajectory_phase="test",
            command_valid=True,
            invalid_reason="",
        )
        assert logger.row_counts["trajectory_command"] == 1
    finally:
        adapter.release_wrench.set()
        acquisition.stop()
        logger.close(completed=False, stop_reason="offline test")


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__acquisition_never_calls_motion_methods(tmp_path):
    logger = real_robot_acquisition__EpisodeLogger(tmp_path / "episode").start()
    adapter = real_robot_acquisition__FakeAdapter()
    acquisition = real_robot_acquisition__RealRobotAcquisition(adapter, logger).start()
    acquisition.stop()
    logger.close(completed=True)
    assert not any(
        token in call
        for call in adapter.calls
        for token in ("enable", "power", "move", "drag", "calibrate", "clear_error")
    )


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__event_source_health_is_required_in_addition_to_state_and_wrench(tmp_path):
    logger = real_robot_acquisition__EpisodeLogger(tmp_path / "episode").start()
    adapter = real_robot_acquisition__FakeAdapter()
    event = {"valid": True, "collision_state": False, "invalid_reason": ""}
    adapter.get_safety_event_status = lambda: dict(event)
    acquisition = real_robot_acquisition__RealRobotAcquisition(adapter, logger).start()
    try:
        assert acquisition.wait_until_healthy(1.)
        event.update(valid=False, collision_state=None, invalid_reason="safety_event_initial_state_unknown")
        assert not acquisition.latest_health().valid
        assert "safety_event_initial_state_unknown" in acquisition.latest_health().invalid_reason
        event.update(valid=True, collision_state=True, invalid_reason="")
        assert "robot_collision" in acquisition.latest_health().invalid_reason
    finally:
        acquisition.stop()
        logger.close(completed=False)


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__stuck_wrench_refuses_concurrent_disconnect(tmp_path):
    logger = real_robot_acquisition__EpisodeLogger(tmp_path / "episode").start()
    adapter = real_robot_acquisition__FakeAdapter(block_wrench=True)
    acquisition = real_robot_acquisition__RealRobotAcquisition(
        adapter,
        logger,
        join_timeout_s=0.02,
    ).start()
    assert adapter.wrench_entered.wait(timeout=0.5)
    with real_robot_acquisition__pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        acquisition.stop()
    assert "stop_state_stream" not in adapter.calls
    assert "disconnect" not in adapter.calls
    assert logger.failed
    adapter.release_wrench.set()
    logger.close(completed=False)


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__stuck_producer_stop_uses_only_nonblocking_failure_signal():
    logger = real_robot_acquisition___SignalOnlyLogger()
    adapter = real_robot_acquisition__FakeAdapter(block_wrench=True)
    acquisition = real_robot_acquisition__RealRobotAcquisition(
        adapter,
        logger,
        join_timeout_s=0.02,
    ).start()
    assert adapter.wrench_entered.wait(timeout=0.5)

    with real_robot_acquisition__pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        acquisition.stop()

    assert logger.failure_event.is_set()
    assert logger.signals
    assert logger.signals[-1][1] == "acquisition_stop"
    assert "stop_state_stream" not in adapter.calls
    assert "disconnect" not in adapter.calls

    adapter.release_wrench.set()
    acquisition.stop()
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__partial_start_failure_joins_started_thread_before_cleanup():
    logger = real_robot_acquisition___LifecycleLogger()
    adapter = real_robot_acquisition__FakeAdapter()
    factory = real_robot_acquisition___ControlledThreadFactory(stuck_state=False)
    acquisition = real_robot_acquisition__RealRobotAcquisition(
        adapter,
        logger,
        thread_factory=factory,
    )
    factory.stop_event = acquisition._stop_event

    with real_robot_acquisition__pytest.raises(RuntimeError, match="synthetic start failure"):
        acquisition.start()

    state_thread = factory.created["real-episode-state"]
    wrench_thread = factory.created["real-episode-wrench"]
    alignment_thread = factory.created["real-episode-alignment"]
    assert state_thread.start_called
    assert state_thread.join_called
    assert state_thread.stop_was_set_at_join
    assert not state_thread.is_alive()
    assert wrench_thread.start_called
    assert not alignment_thread.start_called
    assert adapter.calls == [
        "connect",
        "start_state_stream",
        "stop_state_stream",
        "disconnect",
    ]
    assert logger.failure_event.is_set()
    assert logger.durable_failures
    assert not acquisition.running


@real_robot_acquisition__pytest.mark.usefixtures('real_robot_acquisition__inject_offline_consumer_provider')
def test_real_robot_acquisition__partial_start_failure_refuses_cleanup_while_started_thread_is_alive():
    logger = real_robot_acquisition___LifecycleLogger()
    adapter = real_robot_acquisition__FakeAdapter()
    factory = real_robot_acquisition___ControlledThreadFactory(stuck_state=True)
    acquisition = real_robot_acquisition__RealRobotAcquisition(
        adapter,
        logger,
        join_timeout_s=0.01,
        thread_factory=factory,
    )
    factory.stop_event = acquisition._stop_event

    with real_robot_acquisition__pytest.raises(RuntimeError, match="refusing SDK cleanup/disconnect"):
        acquisition.start()

    state_thread = factory.created["real-episode-state"]
    assert state_thread.join_called
    assert state_thread.stop_was_set_at_join
    assert state_thread.is_alive()
    assert adapter.calls == ["connect", "start_state_stream"]
    assert logger.failure_event.is_set()
    assert logger.signals[-1][1] == "acquisition_start"
    assert logger.durable_failures == []
    assert not acquisition.running

    state_thread.release()
    acquisition.stop()
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]



# ============================================================================
# Source: tests/test_rokae_adapter.py
# ============================================================================

"""Offline contract tests for the read-only ROKAE adapter."""



import threading as rokae_adapter__threading
from types import SimpleNamespace as rokae_adapter__SimpleNamespace
from unittest.mock import patch as rokae_adapter__patch

import pytest as rokae_adapter__pytest

from collection.state import KinematicStateFrame as rokae_adapter__KinematicStateFrame
from hardware.rokae_adapter import RokaeRobotAdapter as rokae_adapter__RokaeRobotAdapter
from hardware.windows.rokae_xcore import RokaeRobot as rokae_adapter__RokaeRobot


class rokae_adapter__FakeReadOnlyNative:
    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[str] = []
        self._state_thread = rokae_adapter__threading.Thread()

    def connect(self) -> None:
        self.calls.append("connect")
        self.is_connected = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.is_connected = False

    def start_state_stream(self) -> None:
        self.calls.append("start_state_stream")

    def stop_state_stream(self) -> None:
        self.calls.append("stop_state_stream")

    def get_cartesian_pose(self):
        self.calls.append("get_cartesian_pose")
        return [0.4, -0.1, 0.3, 0.1, 0.2, 0.3]

    def get_joint_angles(self):
        self.calls.append("get_joint_angles")
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def get_state_frame(self):
        self.calls.append("get_state_frame")
        return rokae_adapter__KinematicStateFrame(
            sequence_id=3,
            host_monotonic_time_s=1.0,
            wall_time_iso="2026-08-09T00:00:00+00:00",
            robot_device_time_s=None,
            valid=True,
            invalid_reason="",
            tcp_position_m=(0.4, -0.1, 0.3),
            tcp_orientation_rad=(0.1, 0.2, 0.3),
            tcp_linear_velocity_mps=None,
            tcp_angular_velocity_radps=None,
            velocity_source="unavailable_first_realtime_frame",
            joint_position_rad=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            joint_velocity_radps=None,
            pose_time_s=1.0,
            joint_time_s=1.0,
            velocity_time_s=None,
            operation_state="IDLE",
            collision_state=None,
            controller_error=None,
        )

    def get_end_wrench(self, frame):
        self.calls.append(f"get_end_wrench:{frame}")
        return {
            "cartesian_force_raw_n": [1.0, 2.0, 3.0],
            "cartesian_torque_raw_nm": [0.1, 0.2, 0.3],
            "joint_measured_torque_nm": [1.0] * 6,
            "joint_external_torque_nm": [0.5] * 6,
            "raw_force_frame": frame,
            "force_query_started_s": 10.0,
            "force_query_finished_s": 10.004,
            "host_monotonic_time_s": 10.002,
            "wall_time_iso": "2026-08-09T00:00:00+00:00",
        }

    def get_robot_metadata(self):
        self.calls.append("get_robot_metadata")
        return {
            "robot_model": "xMate6-Pro",
            "robot_serial_number": "SN-ADAPTER",
            "controller_version": "controller-test",
            "xcore_sdk_version": "0.7.0",
            "joint_soft_limits_rad": [[-2.0, 2.0]] * 6,
            "sdk_tool_payload": {
                "toolset_load_mass_kg": 1.25,
                "toolset_load_cog_m": [0.0, 0.0, 0.08],
                "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
                "sdk_available_tool_names": ["rehab_cuff"],
                "sdk_available_workobject_names": ["rehab_fixture"],
                "active_hmi_tool_workobject_verified": False,
            },
        }

    def get_collision_state(self):
        self.calls.append("get_collision_state")
        return False


def test_rokae_adapter__adapter_exposes_only_read_contract_and_preserves_units():
    native = rokae_adapter__FakeReadOnlyNative()
    adapter = rokae_adapter__RokaeRobotAdapter(native_robot=native)
    adapter.connect()
    adapter.start_state_stream()
    assert adapter.is_connected()
    assert adapter.read_tcp_pose() == (0.4, -0.1, 0.3, 0.1, 0.2, 0.3)
    assert adapter.read_joint_positions() == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    adapter.stop_state_stream()
    adapter.disconnect()
    assert not adapter.is_connected()
    assert native.calls == [
        "connect",
        "start_state_stream",
        "get_cartesian_pose",
        "get_joint_angles",
        "stop_state_stream",
        "disconnect",
    ]
    for forbidden in ("enable", "power", "move", "drag", "calibrate", "clear_error"):
        assert not hasattr(adapter, forbidden)


def test_rokae_adapter__wrench_frame_keeps_query_and_publish_times():
    native = rokae_adapter__FakeReadOnlyNative()
    adapter = rokae_adapter__RokaeRobotAdapter(native_robot=native)
    frame = adapter.read_internal_wrench("world")
    assert frame.valid
    assert frame.timestamp_source == "host_query_time_no_robot_device_timestamp"
    assert frame.host_query_start_s == 10.0
    assert frame.host_query_end_s == 10.004
    assert abs(frame.query_duration_ms - 4.0) < 1e-9
    assert frame.host_publish_s > 0.0
    assert frame.cartesian_force_raw_n == (1.0, 2.0, 3.0)


def test_rokae_adapter__summary_labels_host_time_and_does_not_invent_device_time():
    native = rokae_adapter__FakeReadOnlyNative()
    native.is_connected = True
    adapter = rokae_adapter__RokaeRobotAdapter(native_robot=native)
    summary = adapter.get_robot_state_summary()
    assert summary["connected"] is True
    assert summary["timestamp_source"] == "host_receive_time_no_robot_device_timestamp"
    assert summary["robot_metadata"]["xcore_sdk_version"] == "0.7.0"
    assert summary["collision_state"] is False
    assert summary["collision_state_query_valid"] is True
    assert summary["joint_soft_limits_valid"] is True
    assert summary["sdk_tool_payload_read_valid"] is True
    assert "robot_device_time_s" not in summary


def test_rokae_adapter__summary_latches_observed_collision_across_later_false_query():
    native = rokae_adapter__FakeReadOnlyNative()
    values = iter((True, False))
    native.get_collision_state = lambda: next(values)
    adapter = rokae_adapter__RokaeRobotAdapter(native_robot=native)
    assert adapter.get_robot_state_summary()["collision_state"] is True
    assert adapter.get_robot_state_summary()["collision_state"] is True


def test_rokae_adapter__native_connect_does_not_change_motion_mode_or_command_cache():
    class Native:
        def __init__(self, *_args):
            self.calls = []

        def connectToRobot(self, *args):
            self.calls.append(("connectToRobot", *args))

        def robotInfo(self, ec):
            self.calls.append("robotInfo")
            return rokae_adapter__SimpleNamespace(joint_num=6, type="fake", version="fake")

        def forceControl(self):
            self.calls.append("forceControl")
            return object()

        def setMotionControlMode(self, *_args):
            raise AssertionError("read-only connect changed motion mode")

        def setMaxCacheSize(self, *_args):
            raise AssertionError("read-only connect changed command cache")

    native_holder = {}

    def make_native(*args):
        native_holder["robot"] = Native(*args)
        return native_holder["robot"]

    sdk = rokae_adapter__SimpleNamespace(
        BaseRobot=rokae_adapter__SimpleNamespace(sdkVersion=lambda: "0.7.0"),
        xMateRobot=make_native,
    )
    robot = rokae_adapter__RokaeRobot("192.0.2.1")
    with (
        rokae_adapter__patch("hardware.windows.rokae_xcore._load_sdk", return_value=sdk),
        rokae_adapter__patch.object(robot, "_read_joint_soft_limits_rad", return_value=None),
        rokae_adapter__patch.object(robot, "start_state_stream") as start_stream,
        rokae_adapter__patch.object(robot, "_refresh_operation_state"),
    ):
        robot.connect()

    assert robot.is_connected
    assert native_holder["robot"].calls == [
        ("connectToRobot", "192.0.2.1"),
        "robotInfo",
        "forceControl",
    ]
    start_stream.assert_called_once_with()


def test_rokae_adapter__command_target_update_does_not_take_wrench_query_lock():
    robot = rokae_adapter__RokaeRobot("192.0.2.1")
    robot._rt_active = True
    robot._rt_controller = object()
    with robot._wrench_sdk_lock:
        robot.set_realtime_cartesian_target([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
    assert robot._rt_target_native == (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)


def test_rokae_adapter__external_realtime_attach_checks_but_never_sets_power_or_mode():
    automatic = object()
    powered_on = object()
    idle = rokae_adapter__SimpleNamespace(name="idle")
    rt_controlling = rokae_adapter__SimpleNamespace(name="rtControlling")

    class Controller:
        def __init__(self, calls):
            self.calls = calls

        def setFilterFrequency(self, joint, cartesian, torque, ec):
            self.calls.append(("setFilterFrequency", joint, cartesian, torque))

    class Native:
        def __init__(self):
            self.calls = []
            self.controller = Controller(self.calls)
            self.operation_reads = 0

        def operateMode(self, ec):
            self.calls.append("operateMode")
            return automatic

        def powerState(self, ec):
            self.calls.append("powerState")
            return powered_on

        def getRtMotionController(self):
            self.calls.append("getRtMotionController")
            return self.controller

        def operationState(self, ec):
            self.calls.append("operationState")
            self.operation_reads += 1
            return idle if self.operation_reads == 1 else rt_controlling

        def setOperateMode(self, *_args):
            raise AssertionError("attach set operate mode")

        def setPowerState(self, *_args):
            raise AssertionError("attach set power state")

        def setMotionControlMode(self, *_args):
            raise AssertionError("attach set motion mode")

    robot = rokae_adapter__RokaeRobot("192.0.2.1", local_ip="192.0.2.2")
    robot._sdk = rokae_adapter__SimpleNamespace(
        OperateMode=rokae_adapter__SimpleNamespace(automatic=automatic),
        PowerState=rokae_adapter__SimpleNamespace(on=powered_on),
        OperationState=rokae_adapter__SimpleNamespace(
            idle=idle,
            rtControlling=rt_controlling,
        ),
    )
    robot._robot = Native()
    robot.is_connected = True
    robot.attach_externally_prepared_realtime(reviewed_filter_hz=25.0)
    assert robot._robot.calls == [
        "operationState",
        "operateMode",
        "powerState",
        "getRtMotionController",
        ("setFilterFrequency", 25.0, 25.0, 25.0),
        "operationState",
    ]


def test_rokae_adapter__native_stop_failure_calls_both_stops_and_preserves_hold_target():
    class Controller:
        def __init__(self):
            self.calls = []

        def stopLoop(self):
            self.calls.append("stopLoop")
            raise RuntimeError("loop did not confirm stop")

        def stopMove(self):
            self.calls.append("stopMove")

    robot = rokae_adapter__RokaeRobot("192.0.2.1")
    controller = Controller()
    robot._robot = object()
    robot._rt_controller = controller
    robot._rt_active = True
    robot._rt_callback = lambda: None
    robot._rt_target_native = (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)
    robot.is_connected = True
    with rokae_adapter__pytest.raises(RuntimeError, match="stopLoop"):
        robot.stop_realtime(switch_to_nrt=False)
    assert controller.calls == ["stopLoop", "stopMove"]
    assert robot._rt_active is True
    assert robot._rt_callback is not None
    assert robot._rt_target_native == (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)


def test_rokae_adapter__state_stream_stop_retains_live_thread_and_refuses_native_stop():
    class StuckThread:
        def __init__(self):
            self.join_calls = []

        def join(self, timeout):
            self.join_calls.append(timeout)

        def is_alive(self):
            return True

    class Native:
        def __init__(self):
            self.stop_receive_calls = 0

        def stopReceiveRobotState(self):
            self.stop_receive_calls += 1

    robot = rokae_adapter__RokaeRobot("192.0.2.1")
    thread = StuckThread()
    native = Native()
    robot._robot = native
    robot.is_connected = True
    robot._state_thread = thread
    robot._state_running = True
    robot._state_streaming = True
    with rokae_adapter__pytest.raises(RuntimeError, match="refusing.*disconnect"):
        robot.stop_state_stream()
    assert thread.join_calls == [1.0]
    assert robot._state_thread is thread
    assert robot._state_streaming is True
    assert native.stop_receive_calls == 0


def test_rokae_adapter__disconnect_error_preserves_native_handle_and_connection_state():
    class Native:
        def __init__(self):
            self.fail = True
            self.calls = 0

        def disconnectFromRobot(self, ec):
            self.calls += 1
            if self.fail:
                ec.update(ec=42, message="controller did not confirm disconnect")

    robot = rokae_adapter__RokaeRobot("192.0.2.1")
    native = Native()
    robot._robot = native
    robot._robot_info = object()
    robot._force_control = object()
    robot.is_connected = True
    with rokae_adapter__pytest.raises(RuntimeError, match="disconnectFromRobot failed"):
        robot.disconnect()
    assert native.calls == 1
    assert robot._robot is native
    assert robot._robot_info is not None
    assert robot._force_control is not None
    assert robot.is_connected is True

    native.fail = False
    robot.disconnect()
    assert native.calls == 2
    assert robot._robot is None
    assert robot.is_connected is False
    assert robot.robot_mode == "DISCONNECTED"


def test_rokae_adapter__start_and_stop_share_one_atomic_native_lifecycle_lock():
    start_entered = rokae_adapter__threading.Event()
    allow_start_return = rokae_adapter__threading.Event()

    class Controller:
        def __init__(self):
            self.calls = []

        def setControlLoopCar(self, callback, priority, use_state):
            self.calls.append("setControlLoopCar")

        def startMove(self, mode):
            self.calls.append("startMove")

        def startLoop(self, blocking):
            self.calls.append("startLoop")
            start_entered.set()
            assert allow_start_return.wait(timeout=1.0)

        def stopLoop(self):
            self.calls.append("stopLoop")

        def stopMove(self):
            self.calls.append("stopMove")

    robot = rokae_adapter__RokaeRobot("192.0.2.1")
    controller = Controller()
    robot._sdk = rokae_adapter__SimpleNamespace(
        CartesianPosition=lambda target: tuple(target),
        RtControllerMode=rokae_adapter__SimpleNamespace(cartesianPosition="cartesianPosition"),
    )
    robot._robot = object()
    robot._rt_controller = controller
    robot.is_connected = True
    errors = []

    starter = rokae_adapter__threading.Thread(
        target=lambda: rokae_adapter___capture_thread_error(
            errors,
            robot.start_realtime_cartesian,
            (0.3, 0.0, 0.4, 0.0, 0.0, 0.0),
        )
    )
    stopper = rokae_adapter__threading.Thread(
        target=lambda: rokae_adapter___capture_thread_error(
            errors,
            robot.stop_realtime,
            False,
        )
    )
    starter.start()
    assert start_entered.wait(timeout=1.0)
    stopper.start()
    allow_start_return.set()
    starter.join(timeout=1.0)
    stopper.join(timeout=1.0)
    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert errors == []
    assert controller.calls == [
        "setControlLoopCar",
        "startMove",
        "startLoop",
        "stopLoop",
        "stopMove",
    ]
    assert robot._rt_active is False


def rokae_adapter___capture_thread_error(errors, function, *args):
    try:
        function(*args)
    except BaseException as exc:
        errors.append(exc)



# ============================================================================
# Source: tests/test_rokae_commissioning.py
# ============================================================================

"""Offline tests for the supervised ROKAE commissioning layer."""



import json as rokae_commissioning__json
import math as rokae_commissioning__math
from pathlib import Path as rokae_commissioning__Path
from types import SimpleNamespace as rokae_commissioning__SimpleNamespace

import pytest as rokae_commissioning__pytest

from hardware.rokae_commissioning import (
    COMMISSIONING_CONFIRMATION as rokae_commissioning__COMMISSIONING_CONFIRMATION,
    CommissioningError as rokae_commissioning__CommissioningError,
    CommissioningLimits as rokae_commissioning__CommissioningLimits,
    RokaeCommissioningConsole as rokae_commissioning__RokaeCommissioningConsole,
)

rokae_commissioning__SOFT_LIMITS = (
    (-rokae_commissioning__math.pi, rokae_commissioning__math.pi),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
)


class rokae_commissioning__FakeNative:
    """Records every call so the tests can prove which commands were issued."""

    def __init__(self, *, collision=False, soft_limits=rokae_commissioning__SOFT_LIMITS, state_ok=True):
        self.calls = []
        self.ip_address = "192.0.2.10"
        self.local_ip = "192.0.2.20"
        self.rt_network_tolerance_percent = 20
        self._collision = collision
        self._soft_limits = soft_limits
        self._state_ok = state_ok
        self._joints = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self._pose = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]
        self.powered = False

    def get_robot_metadata(self):
        self.calls.append("get_robot_metadata")
        return {
            "robot_model": "xMate3",
            "xcore_sdk_version": "0.7.0",
            "joint_soft_limits_rad": self._soft_limits,
        }

    def get_cartesian_pose(self):
        return list(self._pose)

    def get_joint_angles(self):
        return list(self._joints)

    def enable(self, load=0.0):
        self.calls.append(("enable", load))
        self.powered = True

    def disable(self):
        self.calls.append("disable")
        self.powered = False

    def set_speed(self, ratio):
        self.calls.append(("set_speed", ratio))

    def get_robot_mode(self):
        return "IDLE" if self.powered else "DISCONNECTED"

    def clear_error(self):
        self.calls.append("clear_error")

    def reset(self):
        self.calls.append("reset")

    def move_j(self, joints):
        self.calls.append(("move_j", [float(v) for v in joints]))
        self._joints = [float(v) for v in joints]

    def move_l(self, pose):
        self.calls.append(("move_l", [float(v) for v in pose]))
        self._pose = [float(v) for v in pose]

    def wait_idle(self, timeout=30.0):
        self.calls.append(("wait_idle", timeout))
        return True

    def stop(self):
        self.calls.append("stop")

    def enable_realtime(self, load=0.0):
        self.calls.append(("enable_realtime", load))
        self.powered = True

    def stop_realtime(self, switch_to_nrt=True):
        self.calls.append(("stop_realtime", switch_to_nrt))
        self.powered = False

    def _realtime_prepared_state(self):
        return {
            "operate_mode": "automatic",
            "power_state": "on" if self.powered else "off",
            "rt_controller_obtained": any(
                isinstance(call, tuple) and call[0] == "enable_realtime"
                for call in self.calls
            ),
        }


class rokae_commissioning__FakeAdapter:
    def __init__(self, native):
        self._native = native
        self.calls = []
        self.connected = False
        self.streaming = False

    @property
    def native_robot(self):
        return self._native

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False

    def start_state_stream(self):
        self.calls.append("start_state_stream")
        self.streaming = True

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")
        self.streaming = False

    def is_connected(self):
        return self.connected

    @property
    def state_thread_alive(self):
        return self.streaming

    def read_tcp_pose(self):
        return tuple(self._native.get_cartesian_pose())

    def read_joint_positions(self):
        return tuple(self._native.get_joint_angles())

    def read_internal_wrench(self):
        return rokae_commissioning__SimpleNamespace(valid=True)

    def get_robot_state_summary(self):
        return {
            "connected": self.connected,
            "state_stream_thread_alive": self.streaming,
            "state_valid": True,
            "state_invalid_reason": "",
            "state_age_ms": 1.0,
            "tcp_pose_base_m_rad": list(self._native.get_cartesian_pose()),
            "joint_position_rad": list(self._native.get_joint_angles()),
            "operation_state": self._native.get_robot_mode(),
            "collision_state": self._native._collision,
            "collision_state_query_valid": True,
            "collision_state_invalid_reason": "",
            "joint_soft_limits_valid": True,
            "sdk_tool_payload_read_valid": True,
            "robot_metadata": self._native.get_robot_metadata(),
        }


def rokae_commissioning__make_console(**native_kwargs):
    native = rokae_commissioning__FakeNative(**native_kwargs)
    adapter = rokae_commissioning__FakeAdapter(native)
    console = rokae_commissioning__RokaeCommissioningConsole(adapter, limits=rokae_commissioning__CommissioningLimits())
    return console, adapter, native


def test_rokae_commissioning__power_on_requires_exact_confirmation_and_connect():
    console, adapter, native = rokae_commissioning__make_console()
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="connect before power_on"):
        console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    console.connect()
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="requires --confirm"):
        console.power_on(confirmation="yes")
    assert native.calls.count("enable") == 0
    result = console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION, speed_ratio=5)
    assert result["powered"] is True
    assert ("enable", 0.0) in native.calls
    assert ("set_speed", 5) in native.calls


def test_rokae_commissioning__joint_jog_is_bounded_by_step_and_soft_limits():
    console, adapter, native = rokae_commissioning__make_console()
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="exceeds per-command limit"):
        console.jog_joint(1, rokae_commissioning__math.radians(30.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in native.calls)

    console.jog_joint(1, rokae_commissioning__math.radians(1.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    issued = [call for call in native.calls if isinstance(call, tuple) and call[0] == "move_j"]
    assert len(issued) == 1
    assert issued[0][1][1] == rokae_commissioning__pytest.approx(0.1 + rokae_commissioning__math.radians(1.0))

    native._joints = [0.1, 1.99, 0.1, 0.1, 0.1, 0.1]
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="outside"):
        console.jog_joint(1, rokae_commissioning__math.radians(1.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)


def test_rokae_commissioning__collision_state_blocks_motion_before_any_command():
    console, adapter, native = rokae_commissioning__make_console(collision=True)
    console.connect()
    # A collision latch is a stop condition; it blocks power-on as well as jog.
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="collision"):
        console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    assert "enable" not in native.calls

    clean, clean_adapter, clean_native = rokae_commissioning__make_console()
    clean.connect()
    clean.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    clean_native._collision = True
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="collision"):
        clean.jog_joint(1, rokae_commissioning__math.radians(1.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in clean_native.calls)


def test_rokae_commissioning__missing_soft_limits_block_motion():
    console, adapter, native = rokae_commissioning__make_console(soft_limits=None)
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="soft limits"):
        console.jog_joint(1, rokae_commissioning__math.radians(1.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)


def test_rokae_commissioning__cartesian_jog_respects_step_limit_and_workspace():
    limits = rokae_commissioning__CommissioningLimits(
        max_cartesian_step_m=0.01,
        workspace_min_base_m=(0.2, -0.5, 0.0),
        workspace_max_base_m=(0.8, 0.5, 0.6),
    )
    native = rokae_commissioning__FakeNative()
    adapter = rokae_commissioning__FakeAdapter(native)
    console = rokae_commissioning__RokaeCommissioningConsole(adapter, limits=limits)
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="must not exceed"):
        console.jog_cartesian((0.05, 0.0, 0.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)

    console.jog_cartesian((0.005, 0.0, 0.01), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    issued = [call for call in native.calls if isinstance(call, tuple) and call[0] == "move_l"]
    assert issued[-1][1][0] == rokae_commissioning__pytest.approx(0.405)
    assert issued[-1][1][2] == rokae_commissioning__pytest.approx(0.31)

    # Raise the step ceiling so this case exercises the workspace gate, not the step gate.
    wide = rokae_commissioning__CommissioningLimits(
        max_cartesian_step_m=1.0,
        workspace_min_base_m=(0.2, -0.5, 0.0),
        workspace_max_base_m=(0.8, 0.5, 0.6),
    )
    wide_native = rokae_commissioning__FakeNative()
    wide_console = rokae_commissioning__RokaeCommissioningConsole(rokae_commissioning__FakeAdapter(wide_native), limits=wide)
    wide_console.connect()
    wide_console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="outside the reviewed workspace"):
        wide_console.jog_cartesian((0.0, 0.0, 0.6), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)


def test_rokae_commissioning__stop_motion_needs_no_confirmation_and_is_audited(tmp_path: rokae_commissioning__Path):
    audit = tmp_path / "commission.jsonl"
    native = rokae_commissioning__FakeNative()
    adapter = rokae_commissioning__FakeAdapter(native)
    console = rokae_commissioning__RokaeCommissioningConsole(adapter, audit_path=audit)
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    result = console.stop_motion("operator_pressed_estop")
    assert result["stopped"] is True
    assert "stop" in native.calls
    assert ("stop_realtime", True) in native.calls
    records = [rokae_commissioning__json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert any(record["action"] == "stop_motion" for record in records)
    assert any(record["action"] == "power_on" for record in records)


def test_rokae_commissioning__self_test_connects_runs_read_only_checks_then_disconnects():
    native = rokae_commissioning__FakeNative()
    adapter = rokae_commissioning__FakeAdapter(native)
    console = rokae_commissioning__RokaeCommissioningConsole(adapter)
    result = console.self_test()
    assert result["motion_commanded"] is False
    assert result["power_commanded"] is False
    assert result["cleanup"]["opened_by_self_test"] is True
    assert adapter.calls[:2] == ["connect", "start_state_stream"]
    assert "disconnect" in adapter.calls
    assert "stop_state_stream" in adapter.calls
    assert "enable" not in native.calls
    assert result["success"] is True


def test_rokae_commissioning__disconnect_powers_down_and_stops_stream():
    native = rokae_commissioning__FakeNative()
    adapter = rokae_commissioning__FakeAdapter(native)
    console = rokae_commissioning__RokaeCommissioningConsole(adapter)
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    result = console.disconnect()
    assert result["disconnected"] is True
    assert "disable" in native.calls
    assert "stop_state_stream" in adapter.calls
    assert "disconnect" in adapter.calls


def test_rokae_commissioning__cli_requires_confirmation_for_motion(monkeypatch, capsys):
    from scripts import rokae_commission as cli

    code = cli.main(
        ["--ip", "192.0.2.10", "power-on"],
        adapter_factory=lambda ip: rokae_commissioning__FakeAdapter(rokae_commissioning__FakeNative()),
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "requires --confirm" in captured.err


def test_rokae_commissioning__cli_status_runs_and_reports_connected_state(capsys):
    from scripts import rokae_commission as cli

    code = cli.main(
        ["--ip", "192.0.2.10", "status"],
        adapter_factory=lambda ip: rokae_commissioning__FakeAdapter(rokae_commissioning__FakeNative()),
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = rokae_commissioning__json.loads(captured.out)
    assert payload["status"]["connected"] is True

def test_rokae_commissioning__prepare_realtime_selects_mode_without_starting_motion():
    console, adapter, native = rokae_commissioning__make_console()
    console.connect()
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="requires --confirm"):
        console.prepare_realtime(confirmation="nope")
    assert not any(isinstance(call, tuple) and call[0] == "enable_realtime" for call in native.calls)

    result = console.prepare_realtime(
        confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION,
        network_tolerance_percent=10,
    )
    assert result["realtime_prepared"] is True
    assert result["motion_loop_started"] is False
    assert ("enable_realtime", 0.0) in native.calls
    assert native.rt_network_tolerance_percent == 10
    # Preparing realtime never issues a target or starts the RT loop.
    assert not any(isinstance(call, tuple) and call[0] in ("move_j", "move_l") for call in native.calls)
    assert "start_realtime_cartesian" not in native.calls


def test_rokae_commissioning__prepare_realtime_refuses_power_already_on_and_bad_tolerance():
    console, adapter, native = rokae_commissioning__make_console()
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="power_off before"):
        console.prepare_realtime(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)

    clean, clean_adapter, clean_native = rokae_commissioning__make_console()
    clean.connect()
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="network_tolerance_percent"):
        clean.prepare_realtime(
            confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION,
            network_tolerance_percent=999,
        )
    assert "enable_realtime" not in clean_native.calls


def test_rokae_commissioning__end_realtime_returns_to_non_realtime_mode():
    console, adapter, native = rokae_commissioning__make_console()
    console.connect()
    console.prepare_realtime(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    assert console.realtime_prepared is True
    result = console.end_realtime(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)
    assert result["realtime_prepared"] is False
    assert console.powered is False
    assert ("stop_realtime", True) in native.calls
    assert "disable" in native.calls


def test_rokae_commissioning__prepare_realtime_requires_a_local_interface():
    console, adapter, native = rokae_commissioning__make_console()
    native.local_ip = ""
    console.connect()
    with rokae_commissioning__pytest.raises(rokae_commissioning__CommissioningError, match="local_ip|subnet"):
        console.prepare_realtime(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)


def test_rokae_commissioning__audit_records_a_failed_motion_wait(tmp_path: rokae_commissioning__Path):
    audit = tmp_path / "commission.jsonl"
    native = rokae_commissioning__FakeNative()
    adapter = rokae_commissioning__FakeAdapter(native)
    console = rokae_commissioning__RokaeCommissioningConsole(adapter, audit_path=audit)
    console.connect()
    console.power_on(confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)

    def failing_wait(timeout=30.0):
        raise TimeoutError("robot did not become idle")

    native.wait_idle = failing_wait
    with rokae_commissioning__pytest.raises(TimeoutError):
        console.jog_joint(1, rokae_commissioning__math.radians(1.0), confirmation=rokae_commissioning__COMMISSIONING_CONFIRMATION)

    records = [rokae_commissioning__json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    entries = [r for r in records if r["action"] == "jog_joint"]
    assert len(entries) == 1
    assert entries[0]["command_issued"] is True
    assert entries[0]["ok"] is False
    assert "TimeoutError" in entries[0]["failure"]


def test_rokae_commissioning__network_probe_validates_and_separates_network_from_sdk(monkeypatch):
    from hardware import network_probe

    assert network_probe.DEFAULT_PROBE_PORTS == (6666, 7777)
    assert network_probe.validate_ipv4("192.168.50.103") == "192.168.50.103"
    with rokae_commissioning__pytest.raises(ValueError, match="not an IPv4"):
        network_probe.validate_ipv4("192.168.50")
    with rokae_commissioning__pytest.raises(ValueError, match="not an IPv4"):
        network_probe.validate_ipv4("192.168.50.300")

    monkeypatch.setattr(network_probe, "icmp_ping", lambda ip, timeout_s=1.0: {"attempted": True, "reachable": False})
    monkeypatch.setattr(network_probe, "tcp_reachable", lambda ip, port, timeout_s=0.5: port == 8055)
    result = network_probe.probe_controller("192.0.2.10", ports=(8055,), timeout_s=0.1).to_dict()
    assert result["reachable"] is True
    assert result["robot_commands_sent"] is False


def test_rokae_commissioning__network_probe_reports_inconclusive_when_ping_blocked(monkeypatch):
    from hardware import network_probe

    monkeypatch.setattr(network_probe, "icmp_ping", lambda ip, timeout_s=1.0: {"attempted": True, "reachable": False})
    monkeypatch.setattr(network_probe, "tcp_reachable", lambda ip, port, timeout_s=0.5: False)
    result = network_probe.probe_controller("192.0.2.10", ports=(8055,), timeout_s=0.1).to_dict()
    assert result["reachable"] is False
    assert "inconclusive" in result["operator_hint"] or "subnet" in result["operator_hint"]


def test_rokae_commissioning__cli_network_command_sends_no_robot_commands(monkeypatch, capsys):
    from scripts import rokae_commission as cli
    from hardware import network_probe

    monkeypatch.setattr(network_probe, "icmp_ping", lambda ip, timeout_s=1.0: {"attempted": True, "reachable": True})
    monkeypatch.setattr(network_probe, "tcp_reachable", lambda ip, port, timeout_s=0.5: True)
    monkeypatch.setattr(cli, "probe_controller", network_probe.probe_controller)

    code = cli.main(["--ip", "192.0.2.10", "network"])
    captured = capsys.readouterr()
    assert code == 0
    payload = rokae_commissioning__json.loads(captured.out)
    assert payload["robot_commands_sent"] is False


def test_rokae_commissioning__cli_rejects_bad_confirmation_before_opening_a_session():
    from scripts import rokae_commission as cli

    opened = []

    def factory(ip):
        opened.append(ip)
        return rokae_commissioning__FakeAdapter(rokae_commissioning__FakeNative())

    code = cli.main(["--ip", "192.0.2.10", "--confirm", "wrong", "power-on"], adapter_factory=factory)
    assert code == 2
    assert opened == []


def test_rokae_commissioning__cli_accepts_shared_option_after_subcommand():
    from scripts import rokae_commission as cli

    captured = {}

    def factory(ip):
        captured["ip"] = ip
        return rokae_commissioning__FakeAdapter(rokae_commissioning__FakeNative())

    code = cli.main(
        ["--ip", "192.0.2.10", "power-on", "--confirm", rokae_commissioning__COMMISSIONING_CONFIRMATION],
        adapter_factory=factory,
    )
    assert code == 0
    assert captured["ip"] == "192.0.2.10"



# ============================================================================
# Source: tests/test_rokae_diagnostics.py
# ============================================================================

"""Cross-platform tests for read-only xCore diagnostic tooling."""



import time as rokae_diagnostics__time

from collection.snapshot import read_live_robot_state_sample as rokae_diagnostics__read_live_robot_state_sample
from collection.state import InternalWrenchFrame as rokae_diagnostics__InternalWrenchFrame, KinematicStateFrame as rokae_diagnostics__KinematicStateFrame
from scripts.diagnose_robot import measure_state as rokae_diagnostics__measure_rt
from scripts.diagnose_robot import measure_alignment as rokae_diagnostics__measure_alignment
from scripts.diagnose_robot import measure_wrench as rokae_diagnostics__measure_query
from scripts.diagnose_robot import (
    invalid_reason_counts as rokae_diagnostics__invalid_reason_counts,
    pose_dependence_analysis as rokae_diagnostics__pose_dependence_analysis,
    rotation_push_analysis as rokae_diagnostics__rotation_push_analysis,
    sequence_drops as rokae_diagnostics__sequence_drops,
    write_report as rokae_diagnostics__write_report,
)


def rokae_diagnostics___state(sequence_id: int) -> rokae_diagnostics__KinematicStateFrame:
    now = rokae_diagnostics__time.monotonic()
    return rokae_diagnostics__KinematicStateFrame(
        sequence_id=sequence_id, host_monotonic_time_s=now, wall_time_iso="now",
        robot_device_time_s=None, valid=True, invalid_reason="", tcp_position_m=(0.0, 0.0, 0.0),
        tcp_orientation_rad=(0.0, 0.0, 0.0), tcp_linear_velocity_mps=(0.0, 0.0, 0.0),
        tcp_angular_velocity_radps=(0.0, 0.0, 0.0), velocity_source="fake", joint_position_rad=(0.0,) * 6,
        joint_velocity_radps=(0.0,) * 6, pose_time_s=now, joint_time_s=now, velocity_time_s=now,
        operation_state="IDLE", collision_state=False, controller_error=None,
    )


def rokae_diagnostics___wrench(sequence_id: int) -> rokae_diagnostics__InternalWrenchFrame:
    now = rokae_diagnostics__time.monotonic()
    return rokae_diagnostics__InternalWrenchFrame(
        sequence_id=sequence_id, host_monotonic_time_s=now, wall_time_iso="now", valid=True,
        invalid_reason="", source="fake", joint_measured_torque_nm=(0.0,) * 6,
        joint_external_torque_nm=(0.0,) * 6, cartesian_force_raw_n=(1.0, 0.0, 0.0),
        cartesian_torque_raw_nm=(0.0, 0.0, 0.0), raw_force_frame="world",
        cartesian_force_bias_n=(0.0, 0.0, 0.0), cartesian_torque_bias_nm=(0.0, 0.0, 0.0),
        cartesian_force_corrected_n=(1.0, 0.0, 0.0), cartesian_torque_corrected_nm=(0.0, 0.0, 0.0),
        cartesian_force_base_n=(1.0, 0.0, 0.0), cartesian_torque_base_nm=(0.0, 0.0, 0.0),
        base_transform_kind="rotation_only_pending_robot_validation", force_time_s=now, torque_time_s=now,
        force_query_started_s=now - 0.001, force_query_finished_s=now,
    )


class rokae_diagnostics__FakeAdapter:
    def __init__(self) -> None:
        self.sequence = 0

    def get_state_frame(self) -> rokae_diagnostics__KinematicStateFrame:
        self.sequence += 2  # deliberately demonstrates detectable source gaps
        return rokae_diagnostics___state(self.sequence)

    def get_end_wrench(self, frame: str) -> dict[str, object]:
        now = rokae_diagnostics__time.monotonic()
        return {"raw_force_frame": frame, "force_query_started_s": now - 0.0001, "force_query_finished_s": now}


class rokae_diagnostics__FakeSource:
    def __init__(self) -> None:
        self.sequence = 0

    def snapshot(self, now_s: float | None = None) -> rokae_diagnostics__InternalWrenchFrame:
        self.sequence += 1
        return rokae_diagnostics___wrench(self.sequence)


def test_rokae_diagnostics__fake_adapter_timing_diagnostics_collect_rows() -> None:
    robot = rokae_diagnostics__FakeAdapter()
    rows, summary = rokae_diagnostics__measure_rt(robot, duration_s=0.004, poll_hz=2_000.0)
    assert rows and summary["dropped_frames"] >= 1
    query_rows, query_summary = rokae_diagnostics__measure_query(robot, duration_s=0.004, target_hz=2_000.0, reference_frame="world")
    assert query_rows and query_summary["query_duration_ms"]["count"] == len(query_rows)


def test_rokae_diagnostics__fake_adapter_snapshot_alignment_uses_project_snapshot(monkeypatch) -> None:
    monkeypatch.setattr("config.settings.BASE_WRENCH_ROTATION_VERIFIED", False)
    rows, summary = rokae_diagnostics__measure_alignment(rokae_diagnostics__FakeAdapter(), rokae_diagnostics__FakeSource(), duration_s=0.004, sample_hz=1_000.0)
    assert rows
    assert summary["force_query_duration_ms"]["count"] == len(rows)
    assert "base_wrench_rotation_requires_robot_validation" in summary["invalid_reasons"]


def test_rokae_diagnostics__rotation_analysis_reports_axis_sign_and_cross_axis() -> None:
    result = rokae_diagnostics__rotation_push_analysis([(0.0, 0.0, 0.0)], [(0.2, 4.0, 0.4)], "Y")
    assert result["principal_axis"] == "Y"
    assert result["principal_sign"] == "+"
    assert result["expected_axis_positive"] is True
    assert result["cross_axis_ratio"] is not None


def test_rokae_diagnostics__pose_dependence_is_not_gravity_compensation_claim() -> None:
    result = rokae_diagnostics__pose_dependence_analysis(
        [
            {"corrected_force_mean_n": (0.0, 0.0, 0.0), "corrected_torque_mean_nm": (0.0, 0.0, 0.0)},
            {"corrected_force_mean_n": (1.2, 0.0, 0.0), "corrected_torque_mean_nm": (0.0, 0.3, 0.0)},
        ], force_threshold_n=1.0, torque_threshold_nm=0.2,
    )
    assert result["software_bias_pose_dependence_observed"] is True
    assert "does not establish" in result["interpretation"]


def test_rokae_diagnostics__report_writes_csv_and_json(tmp_path) -> None:
    csv_path, json_path = rokae_diagnostics__write_report(tmp_path, "fake", [{"sample_index": 0, "invalid_reason": "a;b"}], {"ok": True})
    assert csv_path.exists() and json_path.exists()
    assert rokae_diagnostics__invalid_reason_counts([{"invalid_reason": "a;b"}]) == {"a": 1, "b": 1}
    assert rokae_diagnostics__sequence_drops([1, 4, 5]) == 2



# ============================================================================
# Source: tests/test_rokae_motion_adapter.py
# ============================================================================

"""Offline tests for the externally prepared motion boundary."""



from types import SimpleNamespace as rokae_motion_adapter__SimpleNamespace

import pytest as rokae_motion_adapter__pytest

from hardware.rokae_motion import RokaeCartesianMotionAdapter as rokae_motion_adapter__RokaeCartesianMotionAdapter


class rokae_motion_adapter__FakeNativeMotion:
    def __init__(self):
        self.calls = []

    def attach_externally_prepared_realtime(self, *, reviewed_filter_hz):
        self.calls.append(("attach_externally_prepared_realtime", reviewed_filter_hz))

    def start_realtime_cartesian(self, pose):
        self.calls.append(("start_realtime_cartesian", tuple(pose)))

    def set_realtime_cartesian_target(self, pose):
        self.calls.append(("set_realtime_cartesian_target", tuple(pose)))

    def realtime_motion_error(self):
        self.calls.append("realtime_motion_error")
        return False

    def stop_realtime(self, *, switch_to_nrt):
        self.calls.append(("stop_realtime", switch_to_nrt))


def test_rokae_motion_adapter__motion_adapter_requires_explicit_attach_and_has_one_stop_path():
    native = rokae_motion_adapter__FakeNativeMotion()
    adapter = rokae_motion_adapter__RokaeCartesianMotionAdapter(rokae_motion_adapter__SimpleNamespace(native_robot=native))
    pose = (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)
    with rokae_motion_adapter__pytest.raises(RuntimeError, match="not attached"):
        adapter.start_cartesian_hold(pose)
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.start_cartesian_hold(pose)
    adapter.send_cartesian_target(pose)
    assert not adapter.has_motion_error()
    adapter.request_stop("offline test")
    adapter.request_stop("duplicate ignored")
    assert native.calls == [
        ("attach_externally_prepared_realtime", 25.0),
        ("start_realtime_cartesian", pose),
        ("set_realtime_cartesian_target", pose),
        "realtime_motion_error",
        ("stop_realtime", False),
    ]
    for forbidden in ("enable", "power_on", "clear_error", "move_l", "move_j"):
        assert not hasattr(adapter, forbidden)


def test_rokae_motion_adapter__stop_failure_is_not_reported_as_success():
    class FailingStop(rokae_motion_adapter__FakeNativeMotion):
        def stop_realtime(self, *, switch_to_nrt):
            self.calls.append(("stop_realtime", switch_to_nrt))
            raise RuntimeError("simulated stopLoop failure")

    native = FailingStop()
    adapter = rokae_motion_adapter__RokaeCartesianMotionAdapter(rokae_motion_adapter__SimpleNamespace(native_robot=native))
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.start_cartesian_hold((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    with rokae_motion_adapter__pytest.raises(RuntimeError, match="stopLoop failure"):
        adapter.request_stop("fault")
    assert adapter.stop_reason == "fault"
    assert adapter.stop_confirmed is False
    assert adapter.active is True
    with rokae_motion_adapter__pytest.raises(RuntimeError, match="stopLoop failure"):
        adapter.request_stop("retry keeps original reason")
    assert native.calls.count(("stop_realtime", False)) == 2


def test_rokae_motion_adapter__stop_failure_can_be_retried_and_then_becomes_idempotent():
    class FailOnce(rokae_motion_adapter__FakeNativeMotion):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def stop_realtime(self, *, switch_to_nrt):
            self.calls.append(("stop_realtime", switch_to_nrt))
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("first stop failed")

    native = FailOnce()
    adapter = rokae_motion_adapter__RokaeCartesianMotionAdapter(rokae_motion_adapter__SimpleNamespace(native_robot=native))
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.start_cartesian_hold((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    with rokae_motion_adapter__pytest.raises(RuntimeError, match="first stop failed"):
        adapter.request_stop("operator")
    adapter.request_stop("retry")
    adapter.request_stop("idempotent")
    assert native.calls.count(("stop_realtime", False)) == 2
    assert adapter.stop_reason == "operator"
    assert adapter.stop_confirmed is True
    assert adapter.active is False


def test_rokae_motion_adapter__stop_intent_permanently_blocks_later_attach_or_start():
    native = rokae_motion_adapter__FakeNativeMotion()
    adapter = rokae_motion_adapter__RokaeCartesianMotionAdapter(rokae_motion_adapter__SimpleNamespace(native_robot=native))
    adapter.request_stop("prestart operator stop")
    with rokae_motion_adapter__pytest.raises(RuntimeError, match="permanently stopped"):
        adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    assert native.calls == [("stop_realtime", False)]

    native = rokae_motion_adapter__FakeNativeMotion()
    adapter = rokae_motion_adapter__RokaeCartesianMotionAdapter(rokae_motion_adapter__SimpleNamespace(native_robot=native))
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.request_stop("stop after attach")
    with rokae_motion_adapter__pytest.raises(RuntimeError, match="permanently stopped"):
        adapter.start_cartesian_hold((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    assert native.calls == [
        ("attach_externally_prepared_realtime", 25.0),
        ("stop_realtime", False),
    ]



# ============================================================================
# Source: tests/test_rokae_probe.py
# ============================================================================

"""Offline safety coverage for the read-only ROKAE probe."""



import json as rokae_probe__json
import math as rokae_probe__math

import pytest as rokae_probe__pytest

from scripts.rokae_probe import (
    PROJECT_ACTION_STATEMENT as rokae_probe__PROJECT_ACTION_STATEMENT,
    VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE as rokae_probe__VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE,
    main as rokae_probe__main,
    probe_adapter as rokae_probe__probe_adapter,
    run_probe as rokae_probe__run_probe,
)


class rokae_probe__FakeReadOnlyAdapter:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.connected = False
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.state_summary: object = {
            "operation_state": "IDLE",
            "sdk_version": "fake",
            "state_valid": True,
            "state_stream_thread_alive": True,
        }
        self.tcp_pose: object = {
            "position_base_m": [0.3, -0.2, 0.4],
            "orientation_rad": [0.0, 0.0, 0.0],
        }
        self.joint_positions: object = [0.0] * 6
        self.internal_wrench: object = {
            "cartesian_force_n": [1.0, 2.0, 3.0],
            "cartesian_torque_nm": [0.1, 0.2, 0.3],
            "host_query_start_s": 10.0,
            "host_query_end_s": 10.01,
            "host_monotonic_time_s": 10.005,
            "valid": True,
        }

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"simulated {name} failure")

    def connect(self) -> None:
        self._call("connect")
        self.connected = True

    def disconnect(self) -> None:
        self._call("disconnect")
        self.connected = False

    def is_connected(self) -> bool:
        self._call("is_connected")
        return self.connected

    def start_state_stream(self) -> None:
        self._call("start_state_stream")

    def stop_state_stream(self) -> None:
        self._call("stop_state_stream")

    def read_tcp_pose(self) -> dict[str, object]:
        self._call("read_tcp_pose")
        return self.tcp_pose  # type: ignore[return-value]

    def read_joint_positions(self) -> list[float]:
        self._call("read_joint_positions")
        return self.joint_positions  # type: ignore[return-value]

    def read_internal_wrench(self) -> dict[str, object]:
        self._call("read_internal_wrench")
        return self.internal_wrench  # type: ignore[return-value]

    def get_robot_state_summary(self) -> dict[str, object]:
        self._call("get_robot_state_summary")
        return self.state_summary  # type: ignore[return-value]

    # If the probe ever expands into a motion/power workflow these methods make
    # the regression fail immediately, in addition to the exact call-list check.
    def clear_error(self) -> None:
        raise AssertionError("probe called clear_error")

    def enable(self) -> None:
        raise AssertionError("probe called enable")

    def set_power_state(self, _enabled: bool) -> None:
        raise AssertionError("probe called set_power_state")

    def move_l(self, _target: object) -> None:
        raise AssertionError("probe called move_l")

    def move_j(self, _target: object) -> None:
        raise AssertionError("probe called move_j")

    def enable_realtime(self) -> None:
        raise AssertionError("probe called enable_realtime")

    def start_realtime_cartesian(self, _target: object) -> None:
        raise AssertionError("probe called start_realtime_cartesian")


rokae_probe__EXPECTED_SUCCESS_CALLS = [
    "connect",
    "is_connected",
    "start_state_stream",
    "get_robot_state_summary",
    "read_tcp_pose",
    "read_joint_positions",
    "read_internal_wrench",
    "stop_state_stream",
    "disconnect",
    "is_connected",
]


def test_rokae_probe__probe_uses_only_read_only_contract_and_cleans_up() -> None:
    adapter = rokae_probe__FakeReadOnlyAdapter()

    result = rokae_probe__probe_adapter(adapter, robot_ip="192.0.2.10")

    assert result["success"] is True
    assert "declared_read_only" not in result
    assert result["project_action_statement"] == rokae_probe__PROJECT_ACTION_STATEMENT
    assert result["vendor_session_side_effects_possible"] is True
    assert (
        result["vendor_session_side_effect_disclosure"]
        == rokae_probe__VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE
    )
    assert result["checks"]["tcp_pose"]["value"]["position_base_m"] == [0.3, -0.2, 0.4]
    assert result["checks"]["tcp_pose"]["semantic_valid"] is True
    assert result["checks"]["internal_wrench"]["value"]["valid"] is True
    assert result["cleanup"]["is_disconnected"]["value"] is True
    assert adapter.connected is False
    assert adapter.calls == rokae_probe__EXPECTED_SUCCESS_CALLS


@rokae_probe__pytest.mark.parametrize(
    ("field_name", "invalid_value", "check_name", "message"),
    [
        (
            "state_summary",
            {"state_valid": False, "state_stream_thread_alive": True},
            "robot_state_summary",
            "state_valid",
        ),
        (
            "state_summary",
            {"state_valid": True, "state_stream_thread_alive": False},
            "robot_state_summary",
            "thread_alive",
        ),
        ("tcp_pose", [0.0] * 5, "tcp_pose", "exactly 6"),
        ("tcp_pose", [0.0, 0.0, 0.0, 0.0, rokae_probe__math.nan, 0.0], "tcp_pose", "finite"),
        ("joint_positions", [0.0] * 5, "joint_positions", "exactly 6"),
        (
            "internal_wrench",
            {
                "valid": False,
                "host_query_start_s": 1.0,
                "host_query_end_s": 2.0,
            },
            "internal_wrench",
            "valid",
        ),
        (
            "internal_wrench",
            {
                "valid": True,
                "host_query_start_s": 2.0,
                "host_query_end_s": 1.0,
            },
            "internal_wrench",
            "must not precede",
        ),
        (
            "internal_wrench",
            {
                "valid": True,
                "host_query_start_s": rokae_probe__math.nan,
                "host_query_end_s": 2.0,
            },
            "internal_wrench",
            "must be finite",
        ),
    ],
)
def test_rokae_probe__success_requires_semantically_valid_measurements(
    field_name: str,
    invalid_value: object,
    check_name: str,
    message: str,
) -> None:
    adapter = rokae_probe__FakeReadOnlyAdapter()
    setattr(adapter, field_name, invalid_value)

    result = rokae_probe__probe_adapter(adapter)

    check = result["checks"][check_name]
    assert result["success"] is False
    assert check["transport_ok"] is True
    assert check["semantic_valid"] is False
    assert check["error"]["type"] == "ProbeSemanticError"
    assert message in check["error"]["message"]
    assert adapter.connected is False


def test_rokae_probe__probe_read_failure_is_reported_and_cleanup_still_runs() -> None:
    adapter = rokae_probe__FakeReadOnlyAdapter(fail_on="read_internal_wrench")

    result = rokae_probe__probe_adapter(adapter)

    assert result["success"] is False
    assert result["checks"]["internal_wrench"]["error"] == {
        "type": "RuntimeError",
        "message": "simulated read_internal_wrench failure",
    }
    assert adapter.calls[-3:] == ["stop_state_stream", "disconnect", "is_connected"]
    assert adapter.connected is False


def test_rokae_probe__partial_stream_start_failure_is_cleaned_up() -> None:
    adapter = rokae_probe__FakeReadOnlyAdapter(fail_on="start_state_stream")

    result = rokae_probe__probe_adapter(adapter)

    assert result["success"] is False
    assert result["checks"]["start_state_stream"]["ok"] is False
    assert "stop_state_stream" in adapter.calls
    assert adapter.calls[-2:] == ["disconnect", "is_connected"]
    assert adapter.connected is False


def test_rokae_probe__disconnect_must_be_semantically_confirmed() -> None:
    class StickyConnection(rokae_probe__FakeReadOnlyAdapter):
        def disconnect(self) -> None:
            self._call("disconnect")
            # Simulate a transport call that returned without actually changing
            # the adapter's observable connection state.

    adapter = StickyConnection()
    result = rokae_probe__probe_adapter(adapter)
    assert result["success"] is False
    assert result["cleanup"]["disconnect"]["transport_ok"] is True
    assert result["cleanup"]["is_disconnected"]["semantic_valid"] is False
    assert "must be true" in result["cleanup"]["is_disconnected"]["error"]["message"]


def test_rokae_probe__cli_prints_machine_readable_json_with_injected_factory(capsys) -> None:
    adapter = rokae_probe__FakeReadOnlyAdapter()

    exit_code = rokae_probe__main(
        ["--ip", "192.0.2.20"],
        adapter_factory=lambda robot_ip: adapter,
    )

    payload = rokae_probe__json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["robot_ip"] == "192.0.2.20"
    assert adapter.calls == rokae_probe__EXPECTED_SUCCESS_CALLS


def test_rokae_probe__factory_failure_is_json_reportable_without_importing_sdk() -> None:
    def unavailable(_robot_ip: str) -> object:
        raise ImportError("simulated missing xCoreSDK adapter")

    payload = rokae_probe__run_probe("192.0.2.30", adapter_factory=unavailable)

    assert payload["success"] is False
    assert payload["checks"]["adapter_factory"]["error"]["type"] == "ImportError"
    assert payload["cleanup"] == {}



# ============================================================================
# Source: tests/test_rokae_safety_events.py
# ============================================================================

"""Safety-event lifecycle regression with a fake SDK; never opens a socket."""

from types import SimpleNamespace as rokae_safety_events__SimpleNamespace
import threading as rokae_safety_events__threading

import pytest as rokae_safety_events__pytest

from hardware.safety_events import SafetyEventMonitor as rokae_safety_events__SafetyEventMonitor, validate_collision_source as rokae_safety_events__validate_collision_source
from hardware.rokae_adapter import RokaeRobotAdapter as rokae_safety_events__RokaeRobotAdapter
from hardware.windows import rokae_xcore as rokae_safety_events__rokae_xcore


class rokae_safety_events__Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class rokae_safety_events__Native:
    def __init__(self):
        self.callbacks = []
        self.fail_registration = False
        self.fail_disconnect = False
        self.initial_event = None
        self.initial_query = None
        self.initial_query_error = None
        self.calls = []

    def connectToRobot(self, *args):
        self.calls.append("connect")

    def robotInfo(self, ec):
        return rokae_safety_events__SimpleNamespace(joint_num=6, type="synthetic", version="test", id="fake")

    def forceControl(self):
        return object()

    def setEventWatcher(self, event, callback, ec):
        assert event == "safety"
        self.calls.append("subscribe")
        self.callbacks.append(callback)
        if self.initial_event is not None:
            callback(self.initial_event)
        if self.fail_registration:
            ec.update(ec=259, message="synthetic registration failure")

    def setNoneEventWatcher(self, event, ec):
        assert event == "safety"
        self.calls.append("unsubscribe")

    def disconnectFromRobot(self, ec):
        self.calls.append("disconnect")
        if self.fail_disconnect:
            ec.update(ec=42, message="synthetic disconnect failure")

    def queryEventInfo(self, event, ec):
        assert event == "safety"
        self.calls.append("query")
        if self.initial_query_error is not None:
            ec.update(ec=259, message=self.initial_query_error)
            return {}
        return self.initial_query if self.initial_query is not None else {}


@rokae_safety_events__pytest.fixture
def rokae_safety_events__rig(monkeypatch):
    clock = rokae_safety_events__Clock()
    native = rokae_safety_events__Native()
    robot = rokae_safety_events__rokae_xcore.RokaeRobot("192.0.2.1", collision_source="events",
                                  collision_event_max_age_s=0.5)
    robot._safety_events = rokae_safety_events__SafetyEventMonitor(0.5, clock=clock)
    sdk = rokae_safety_events__SimpleNamespace(BaseRobot=rokae_safety_events__SimpleNamespace(sdkVersion=lambda: "0.7.0"),
                          xMateRobot=lambda: native, Event=rokae_safety_events__SimpleNamespace(safety="safety"))
    monkeypatch.setattr(rokae_safety_events__rokae_xcore, "_load_sdk", lambda: sdk)
    monkeypatch.setattr(robot, "start_state_stream", lambda: None)
    monkeypatch.setattr(robot, "_refresh_operation_state", lambda: None)
    monkeypatch.setattr(robot, "_read_joint_soft_limits_rad", lambda: ((-2., 2.),) * 6)
    return robot, native, clock


def test_rokae_safety_events__subscription_without_initial_event_is_unknown(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    assert native.calls == ["connect", "subscribe", "query"]
    assert robot.get_collision_state() is None
    status = robot.get_safety_event_status()
    assert status["registered"]
    assert not status["valid"]
    assert status["invalid_reason"] == "safety_event_initial_state_unknown"
    assert status["initial_query_attempted"]
    assert not status["initial_query_succeeded"]
    assert status["initial_query_error"] == "safety_initial_query_invalid_collided_boolean"


def test_rokae_safety_events__registered_active_query_can_supply_truthful_initial_clear(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    native.initial_query = {"collided": False}
    robot.connect()
    status = robot.get_safety_event_status()
    assert status["valid"]
    assert status["collision_state"] is False
    assert status["sequence"] == 0
    assert status["initial_query_succeeded"]
    assert status["timestamp_source"].startswith("host_active_query")


def test_rokae_safety_events__failed_initial_query_keeps_unknown_but_callback_can_recover(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    native.initial_query_error = "unsupported"
    robot.connect()
    status = robot.get_safety_event_status()
    assert not status["valid"]
    assert "259" in status["initial_query_error"]
    native.callbacks[-1]({"collided": False})
    status = robot.get_safety_event_status()
    assert status["valid"] and status["collision_state"] is False


def test_rokae_safety_events__actual_boolean_false_makes_event_source_valid(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is False
    status = robot.get_safety_event_status()
    assert status["valid"] and status["sequence"] == 1
    assert status["received_host_time_s"] == 10.0
    adapter = rokae_safety_events__RokaeRobotAdapter(native_robot=robot)
    summary = adapter.get_robot_state_summary()
    assert summary["collision_state_query_valid"]
    assert summary["collision_state_source"] == "events"
    assert summary["collision_state_timestamp_source"].startswith("host_callback")


@rokae_safety_events__pytest.mark.parametrize("payload", [{}, {"collided": "false"}, {"collided": 0},
                                     {"collided": None}, None, {"not_collided": False}])
def test_rokae_safety_events__bad_payload_cannot_become_clearance_and_fault_stays_latched(rokae_safety_events__rig, payload):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    native.callbacks[-1](payload)
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    assert "invalid_collided_boolean" in robot.get_safety_event_status()["invalid_reason"]


def test_rokae_safety_events__collision_survives_false_event_and_reconnect(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    native.callbacks[-1]({"collided": True})
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is True
    robot.disconnect()
    robot.connect()
    assert robot.get_collision_state() is None
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is True


def test_rokae_safety_events__expiry_and_late_event_do_not_automatically_restore_clearance(rokae_safety_events__rig):
    robot, native, clock = rokae_safety_events__rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    clock.now += 0.6
    # The receive path must detect the gap even without an intervening reader.
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    assert "expired" in robot.get_safety_event_status()["invalid_reason"]


def test_rokae_safety_events__reconnect_needs_new_initial_event_and_ignores_old_callback(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    old_callback = native.callbacks[-1]
    old_callback({"collided": False})
    robot.disconnect()
    old_callback({"collided": False})
    assert robot.get_collision_state() is None
    robot.connect()
    old_callback({"collided": False})
    assert robot.get_collision_state() is None
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is False


def test_rokae_safety_events__disconnect_cancels_native_callback_before_native_disconnect(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    robot.disconnect()
    assert native.calls[-2:] == ["unsubscribe", "disconnect"]
    assert robot._safety_event_callback is None


def test_rokae_safety_events__failed_disconnect_invalidates_events_and_prevents_resubscription(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    native.fail_disconnect = True
    with rokae_safety_events__pytest.raises(RuntimeError, match="disconnectFromRobot"):
        robot.disconnect()
    assert robot.is_connected
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    with rokae_safety_events__pytest.raises(RuntimeError, match="disconnect is pending"):
        robot.start_safety_events()


def test_rokae_safety_events__registration_failure_discards_even_synchronous_initial_false(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    native.initial_event = {"collided": False}
    native.fail_registration = True
    with rokae_safety_events__pytest.raises(RuntimeError, match="setEventWatcher"):
        robot.connect()
    assert not robot.is_connected
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None


def test_rokae_safety_events__synchronous_initial_event_waits_for_registration_confirmation():
    monitor = rokae_safety_events__SafetyEventMonitor(1.)
    generation = monitor.begin()
    monitor.receive(generation, {"collided": False})
    assert not monitor.snapshot().valid
    monitor.registered(generation)
    assert monitor.snapshot().valid


def test_rokae_safety_events__stream_fault_does_not_recover_from_next_callback(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    robot._mark_state_error(RuntimeError("lost state transport"))
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    assert "stream_error" in robot.get_safety_event_status()["invalid_reason"]


def test_rokae_safety_events__explicit_resubscription_starts_unknown_and_rejects_old_callbacks(rokae_safety_events__rig):
    robot, native, clock = rokae_safety_events__rig
    robot.connect()
    old_callback = native.callbacks[-1]
    old_callback({"collided": False})
    clock.now += 1.
    assert robot.get_collision_state() is None
    robot.start_safety_events()
    old_callback({"collided": False})
    assert robot.get_collision_state() is None
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is False
    robot._rt_active = True
    with rokae_safety_events__pytest.raises(RuntimeError, match="during realtime motion"):
        robot.start_safety_events()


def test_rokae_safety_events__callback_never_takes_sdk_or_state_lock(rokae_safety_events__rig):
    robot, native, _ = rokae_safety_events__rig
    robot.connect()
    finished = rokae_safety_events__threading.Event()
    with robot._sdk_lock, robot._state_lock:
        thread = rokae_safety_events__threading.Thread(target=lambda: (native.callbacks[-1]({"collided": False}), finished.set()))
        thread.start()
        assert finished.wait(1.0)
    thread.join(1.0)
    assert robot.get_collision_state() is False


@rokae_safety_events__pytest.mark.parametrize("age", [None, 0., -1., float("inf"), float("nan"), True])
def test_rokae_safety_events__event_mode_requires_explicit_finite_freshness_budget(age):
    with rokae_safety_events__pytest.raises(ValueError, match="explicit finite positive"):
        rokae_safety_events__validate_collision_source("events", age)


def test_rokae_safety_events__query_remains_default_and_event_only_budget_is_rejected():
    robot = rokae_safety_events__rokae_xcore.RokaeRobot("192.0.2.1")
    assert robot.get_safety_event_status() is None
    with rokae_safety_events__pytest.raises(ValueError, match="requires collision_source=events"):
        rokae_safety_events__validate_collision_source("query", .5)


def test_rokae_safety_events__disconnect_during_registration_cannot_confirm_clearance(rokae_safety_events__rig, monkeypatch):
    robot, native, _ = rokae_safety_events__rig
    original = native.setEventWatcher
    native.initial_event = {"collided": False}

    def disconnect_pending(*args):
        original(*args)
        robot._disconnect_requested = True

    monkeypatch.setattr(native, "setEventWatcher", disconnect_pending)
    with rokae_safety_events__pytest.raises(RuntimeError, match="disconnect requested during"):
        robot.connect()
    assert robot.get_collision_state() is None


def test_rokae_safety_events__events_mode_is_forwarded_by_both_default_adapter_factories(monkeypatch):
    import hardware.rokae_adapter as adapter_module
    from scripts.acquire_robot_data import _configured_adapter_factory
    from scripts.run_rehab_experiment import _default_adapter_factory

    calls = []
    monkeypatch.setattr(adapter_module, "RokaeRobotAdapter",
                        lambda ip, **kwargs: calls.append((ip, kwargs)))
    _configured_adapter_factory(local_ip="192.0.2.2", robot_class=None,
                                state_interval_ms=None, collision_source="events",
                                collision_event_max_age_s=.2)("192.0.2.1")
    _default_adapter_factory("192.0.2.1", local_ip="192.0.2.2",
                             collision_source="events", collision_event_max_age_s=.2)
    for ip, kwargs in calls:
        assert ip == "192.0.2.1"
        assert kwargs["collision_source"] == "events"
        assert kwargs["collision_event_max_age_s"] == .2
    assert len(calls) == 2




def test_rokae_safety_events__state_only_cli_can_subscribe_with_explicit_local_ip(monkeypatch, capsys):
    import scripts.acquire_robot_data as module
    import hardware.rokae_adapter as adapter_module

    calls = []
    monkeypatch.setattr(adapter_module, "RokaeRobotAdapter",
                        lambda ip, **kwargs: calls.append((ip, kwargs)))

    def acquire(**kwargs):
        assert kwargs["mode"] == "state-only"
        kwargs["adapter_factory"](kwargs["robot_ip"])
        return {}

    monkeypatch.setattr(module, "run_acquisition", acquire)
    assert module.main(["--ip", "192.0.2.1", "--local-ip", "192.0.2.2",
                        "--episode-dir", "unused", "--duration-s", "1",
                        "--collision-source", "events", "--collision-event-max-age-s", "0.2"]) == 0
    assert calls[0][1]["collision_source"] == "events"
    assert calls[0][1]["local_ip"] == "192.0.2.2"


def test_rokae_safety_events__missing_event_budget_fails_before_sdk_or_files(monkeypatch, tmp_path):
    import scripts.acquire_robot_data as module

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid event configuration reached acquisition")

    monkeypatch.setattr(module, "run_acquisition", forbidden)
    episode = tmp_path / "episode"
    with rokae_safety_events__pytest.raises(ValueError, match="explicit finite positive"):
        module.main(["--ip", "192.0.2.1", "--episode-dir", str(episode),
                     "--duration-s", "1", "--collision-source", "events"])
    assert not episode.exists()



# ============================================================================
# Source: tests/test_rokae_xcore.py
# ============================================================================

import unittest as rokae_xcore__unittest
import sys as rokae_xcore__sys

from hardware.windows.rokae_xcore import RokaeRobot as rokae_xcore__RokaeRobot, _load_sdk as rokae_xcore___load_sdk


class rokae_xcore___FakeRealtimeController:
    def __init__(self):
        self.callback = None
        self.calls = []

    def setFilterFrequency(self, joint_hz, cartesian_hz, torque_hz, ec):
        self.calls.append(
            ("setFilterFrequency", joint_hz, cartesian_hz, torque_hz)
        )

    def setControlLoopCar(self, callback, priority, use_state):
        self.callback = callback
        self.calls.append(("setControlLoopCar", priority, use_state))

    def startMove(self, mode):
        self.calls.append(("startMove", mode))

    def startLoop(self, blocking):
        self.calls.append(("startLoop", blocking))

    def hasMotionError(self):
        return False

    def stopLoop(self):
        self.calls.append(("stopLoop",))

    def stopMove(self):
        self.calls.append(("stopMove",))


class rokae_xcore___FakeNativeRobot:
    def __init__(self, sdk):
        self.sdk = sdk
        self.rt = rokae_xcore___FakeRealtimeController()
        self.calls = []

    def setOperateMode(self, mode, ec):
        self.calls.append(("setOperateMode", mode))

    def setRtNetworkTolerance(self, percent, ec):
        self.calls.append(("setRtNetworkTolerance", percent))

    def setMotionControlMode(self, mode, ec):
        self.calls.append(("setMotionControlMode", mode))

    def setPowerState(self, powered, ec):
        self.calls.append(("setPowerState", powered))

    def getRtMotionController(self):
        self.calls.append(("getRtMotionController",))
        return self.rt

    def operationState(self, ec):
        return self.sdk.OperationState.idle


class rokae_xcore__RokaeXCoreTests(rokae_xcore__unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if rokae_xcore__sys.platform != "win32":
            raise rokae_xcore__unittest.SkipTest("bundled xCoreSDK tests require Windows")
        previous_sdk = rokae_xcore__sys.modules.get("xCoreSDK_python")
        previous_path = list(rokae_xcore__sys.path)
        def restore_sdk_import():
            # Keep SDK imports local to these fake-native tests.
            if previous_sdk is None:
                rokae_xcore__sys.modules.pop("xCoreSDK_python", None)
            else:
                rokae_xcore__sys.modules["xCoreSDK_python"] = previous_sdk
            rokae_xcore__sys.path[:] = previous_path
        cls.addClassCleanup(restore_sdk_import)
        cls.sdk = rokae_xcore___load_sdk()

    def make_robot(self, local_ip="192.168.50.10"):
        robot = rokae_xcore__RokaeRobot("192.168.50.102", local_ip=local_ip)
        native = rokae_xcore___FakeNativeRobot(self.sdk)
        robot._sdk = self.sdk
        robot._robot = native
        robot.is_connected = True
        robot.cartesian_pose = [0.3, -0.2, 0.35, 0.0, 1.5707963267948966, 0.0]
        robot._state_ready.set()
        return robot, native

    def test_bundled_sdk_is_0_7_0(self):
        self.assertEqual(self.sdk.BaseRobot.sdkVersion(), "0.7.0")
        self.assertTrue(hasattr(self.sdk.xMateRobot, "getRtMotionController"))

    def test_pose_unit_round_trip(self):
        pose = [0.1234, -0.0567, 0.8901, 0.1, -0.2, 0.3]
        restored = rokae_xcore__RokaeRobot._pose_from_sdk(rokae_xcore__RokaeRobot._pose_to_sdk(pose))
        for actual, expected in zip(restored, pose):
            self.assertAlmostEqual(actual, expected)

    def test_realtime_callback_holds_latest_target(self):
        robot, native = self.make_robot()
        robot.enable_realtime()
        robot.start_realtime_cartesian()

        first = native.rt.callback()
        self.assertEqual(first.trans, [0.3, -0.2, 0.35])
        self.assertAlmostEqual(first.rpy[1], 1.5707963267948966)

        robot.set_realtime_cartesian_target(
            [0.310, -0.190, 0.340, 0.1, 0.2, 0.3]
        )
        updated = native.rt.callback()
        self.assertEqual(updated.trans, [0.31, -0.19, 0.34])
        self.assertEqual(updated.rpy, [0.1, 0.2, 0.3])
        self.assertIn(("startLoop", False), native.rt.calls)

        robot.stop_realtime(switch_to_nrt=False)
        self.assertIn(("stopLoop",), native.rt.calls)
        self.assertIn(("stopMove",), native.rt.calls)

    def test_realtime_requires_local_ip(self):
        robot, _ = self.make_robot(local_ip="")
        with self.assertRaisesRegex(ValueError, "ROBOT_LOCAL_IP"):
            robot.enable_realtime()


if __name__ == "__main__":
    rokae_xcore__unittest.main()



# ============================================================================
# Source: tests/test_state_frame_capture.py
# ============================================================================

"""Offline state-capture contracts: accepted frames, not native wire timing.

The controlled source publishes independently of reads. Consumer pauses and
overflow are event-controlled, so a latest-only implementation fails reliably.
No SDK load, network connection, real robot, or wall-time throughput claim.
"""
from dataclasses import replace as state_frame_capture__replace
import csv as state_frame_capture__csv
import threading as state_frame_capture__threading
import time as state_frame_capture__time

import pytest as state_frame_capture__pytest

from collection.episode_logger import EpisodeLogger as state_frame_capture__EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition as state_frame_capture__RealRobotAcquisition
from collection.state import KinematicStateFrame as state_frame_capture__KinematicStateFrame
from collection.state_buffer import StateBufferOverflow as state_frame_capture__StateBufferOverflow, StateFrameBuffer as state_frame_capture__StateFrameBuffer
from hardware.rokae_adapter import RokaeRobotAdapter as state_frame_capture__RokaeRobotAdapter
from hardware.windows.rokae_xcore import RokaeRobot as state_frame_capture__RokaeRobot


def state_frame_capture__state_frame(sequence, stamp=None):
    stamp = state_frame_capture__time.perf_counter() if stamp is None else stamp
    return state_frame_capture__KinematicStateFrame(
        sequence_id=sequence, host_monotonic_time_s=stamp, wall_time_iso=None,
        robot_device_time_s=None, valid=True, invalid_reason="",
        tcp_position_m=(float(sequence), 0., 0.), tcp_orientation_rad=(0., 0., 0.),
        tcp_linear_velocity_mps=None, tcp_angular_velocity_radps=None,
        velocity_source="unavailable", joint_position_rad=(float(sequence),) * 6,
        joint_velocity_radps=None, pose_time_s=stamp, joint_time_s=stamp,
        velocity_time_s=None, operation_state="IDLE", collision_state=None,
        controller_error=None,
    )


def state_frame_capture__wait_for(predicate, timeout=3.):
    deadline = state_frame_capture__time.perf_counter() + timeout
    while state_frame_capture__time.perf_counter() < deadline:
        if predicate():
            return
        state_frame_capture__time.sleep(.001)
    raise AssertionError("offline state-capture condition did not arrive")


def state_frame_capture__read_state_rows(directory):
    with (directory / "robot_state.csv").open(newline="", encoding="utf-8") as stream:
        return list(state_frame_capture__csv.DictReader(stream))


class state_frame_capture__ControlledSource:
    """Independent publication with a deliberate, bounded consumer pause."""
    def __init__(self, capacity=32, tail_sequence=None):
        self.capacity = capacity
        self.tail_sequence = tail_sequence
        self.connected = False
        self.latest = None
        self.buffer = None
        self.source_error = None
        self.generated = []
        self.calls = []
        self.pause_requested = state_frame_capture__threading.Event()
        self.drain_entered = state_frame_capture__threading.Event()
        self.resume_drain = state_frame_capture__threading.Event()
        self.lock = state_frame_capture__threading.Lock()

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def is_connected(self):
        return self.connected

    def start_state_stream(self):
        self.calls.append("start_state_stream")
        self.publish(1)

    def begin_state_capture(self):
        self.calls.append("begin_state_capture")
        with self.lock:
            self.buffer = state_frame_capture__StateFrameBuffer(capacity=self.capacity)
            if self.latest is not None:
                self.buffer.publish(self.latest)
        return True

    def publish(self, sequence):
        item = state_frame_capture__state_frame(sequence)
        with self.lock:
            self.latest = item
            self.generated.append(item)
            if self.buffer is not None:
                self.buffer.publish(item)
        return item

    def read_state_frame(self):
        with self.lock:
            item = self.latest
            error = self.source_error
        if item is None:
            raise RuntimeError("controlled_source_not_ready")
        return state_frame_capture__replace(item, valid=False, invalid_reason=error, controller_error=error) if error else item

    def drain_state_frames(self):
        if self.pause_requested.is_set():
            self.drain_entered.set()
            if not self.resume_drain.wait(3.):
                raise TimeoutError("test did not release state consumer")
        with self.lock:
            buffer = self.buffer
        return () if buffer is None else buffer.drain()

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")
        if self.tail_sequence is not None:
            sequence, self.tail_sequence = self.tail_sequence, None
            self.publish(sequence)

    def end_state_capture(self):
        self.calls.append("end_state_capture")
        with self.lock:
            self.buffer = None

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False


def state_frame_capture__acquisition_for(source, tmp_path):
    logger = state_frame_capture__EpisodeLogger(tmp_path / "episode").start()
    acquisition = state_frame_capture__RealRobotAcquisition(source, logger, diagnostic_state_only=True)
    acquisition.start()
    return acquisition, logger


def state_frame_capture__close_fixture(acquisition, logger, source):
    source.resume_drain.set()
    try:
        acquisition.stop()
    except state_frame_capture__StateBufferOverflow:
        # Overflow is expected in its failure test; it must never become a pass.
        pass
    finally:
        logger.close(completed=False, stop_reason="OFFLINE_STATE_CAPTURE_REGRESSION")


def test_state_frame_capture__buffer_drains_every_accepted_frame_in_order_once():
    buffer = state_frame_capture__StateFrameBuffer(capacity=4)
    frames = tuple(state_frame_capture__state_frame(sequence) for sequence in range(1, 5))
    for item in frames:
        buffer.publish(item)
    assert buffer.drain() == frames
    assert buffer.drain() == ()
    following = state_frame_capture__state_frame(5)
    buffer.publish(following)
    assert buffer.drain() == (following,)


def test_state_frame_capture__overflow_is_explicit_and_stays_latched_on_publish_and_drain():
    buffer = state_frame_capture__StateFrameBuffer(capacity=2)
    buffer.publish(state_frame_capture__state_frame(1))
    buffer.publish(state_frame_capture__state_frame(2))
    with state_frame_capture__pytest.raises(state_frame_capture__StateBufferOverflow):
        buffer.publish(state_frame_capture__state_frame(3))
    with state_frame_capture__pytest.raises(state_frame_capture__StateBufferOverflow):
        buffer.drain()
    with state_frame_capture__pytest.raises(state_frame_capture__StateBufferOverflow):
        buffer.publish(state_frame_capture__state_frame(4))
    # The failed buffer never silently becomes healthy because somebody drained it.
    with state_frame_capture__pytest.raises(state_frame_capture__StateBufferOverflow):
        buffer.drain()


@state_frame_capture__pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_state_frame_capture__buffer_rejects_nonpositive_or_nonintegral_capacity(capacity):
    with state_frame_capture__pytest.raises((TypeError, ValueError)):
        state_frame_capture__StateFrameBuffer(capacity=capacity)


def test_state_frame_capture__consumer_pause_preserves_all_published_frames_and_latest_reads_do_not_consume(tmp_path):
    source = state_frame_capture__ControlledSource(capacity=32)
    acquisition, logger = state_frame_capture__acquisition_for(source, tmp_path)
    try:
        state_frame_capture__wait_for(lambda: logger.row_counts["robot_state"] == 1)
        source.pause_requested.set()
        assert source.drain_entered.wait(2.)
        for sequence in range(2, 13):
            source.publish(sequence)
        assert source.read_state_frame().sequence_id == 12
        assert source.read_state_frame().sequence_id == 12
        # Live health must advance while the history consumer is still paused.
        assert acquisition.latest_state_frame().sequence_id == 12
        assert acquisition.latest_health().state_time_s == source.latest.host_monotonic_time_s
        source.source_error = "controller_stream_fault"
        assert not acquisition.latest_health().state_valid
        source.source_error = None
        source.resume_drain.set()
        state_frame_capture__wait_for(lambda: logger.row_counts["robot_state"] == 12)
        assert acquisition.latest_state_frame().sequence_id == 12
        assert acquisition.latest_health().state_valid
    finally:
        state_frame_capture__close_fixture(acquisition, logger, source)
    rows = state_frame_capture__read_state_rows(tmp_path / "episode")
    assert [int(float(row["q1"])) for row in rows] == list(range(1, 13))
    assert [float(row["host_time_s"]) for row in rows] == [item.host_monotonic_time_s for item in source.generated]


def test_state_frame_capture__capture_overflow_latches_background_failure_instead_of_using_old_latest_as_success(tmp_path):
    source = state_frame_capture__ControlledSource(capacity=2)
    acquisition, logger = state_frame_capture__acquisition_for(source, tmp_path)
    try:
        state_frame_capture__wait_for(lambda: logger.row_counts["robot_state"] == 1)
        source.pause_requested.set()
        assert source.drain_entered.wait(2.)
        source.publish(2)
        source.publish(3)
        with state_frame_capture__pytest.raises(state_frame_capture__StateBufferOverflow):
            source.publish(4)
        source.resume_drain.set()
        state_frame_capture__wait_for(lambda: acquisition.background_error is not None)
        assert "overflow" in acquisition.background_error.lower()
        assert logger.failed
        assert not acquisition.running
    finally:
        state_frame_capture__close_fixture(acquisition, logger, source)


def test_state_frame_capture__empty_batch_still_refreshes_latest_source_error_without_rewriting_same_frame(tmp_path):
    source = state_frame_capture__ControlledSource()
    acquisition, logger = state_frame_capture__acquisition_for(source, tmp_path)
    try:
        state_frame_capture__wait_for(lambda: logger.row_counts["robot_state"] == 1)
        assert source.buffer.drain() == ()
        source.source_error = "injected_source_error_without_new_frame"
        state_frame_capture__wait_for(lambda: "injected_source_error_without_new_frame" in acquisition.latest_health().invalid_reason
                 and not acquisition.latest_health().state_valid)
        assert "injected_source_error_without_new_frame" in acquisition.latest_health().invalid_reason
        assert logger.row_counts["robot_state"] == 1
    finally:
        state_frame_capture__close_fixture(acquisition, logger, source)
    assert len(state_frame_capture__read_state_rows(tmp_path / "episode")) == 1


def test_state_frame_capture__source_is_stopped_then_tail_is_drained_before_capture_end_and_disconnect(tmp_path):
    source = state_frame_capture__ControlledSource(tail_sequence=3)
    acquisition, logger = state_frame_capture__acquisition_for(source, tmp_path)
    try:
        state_frame_capture__wait_for(lambda: logger.row_counts["robot_state"] == 1)
        source.publish(2)
        state_frame_capture__wait_for(lambda: logger.row_counts["robot_state"] == 2)
    finally:
        state_frame_capture__close_fixture(acquisition, logger, source)
    assert [int(float(row["q1"])) for row in state_frame_capture__read_state_rows(tmp_path / "episode")] == [1, 2, 3]
    assert source.calls.index("stop_state_stream") < source.calls.index("end_state_capture") < source.calls.index("disconnect")
    assert not source.connected


def state_frame_capture__make_native():
    native = state_frame_capture__RokaeRobot("192.0.2.1")
    native.is_connected = True
    native.robot_mode = "IDLE"
    return native


def state_frame_capture__accept_native(native, sequence):
    native._accept_state([float(sequence), 0., .4, 0., 0., 0.], [float(sequence)] * 6,
                         1000. + sequence * .008, (False, False))


def test_state_frame_capture__native_latest_only_remains_nonconsuming_without_optional_capture():
    native = state_frame_capture__make_native()
    for sequence in range(1, 1101):
        state_frame_capture__accept_native(native, sequence)
    latest = native.get_state_frame()
    assert latest.valid
    assert latest.sequence_id == 1100
    assert native.get_state_frame() == latest


def test_state_frame_capture__native_capture_preserves_accepted_frames_and_latest_health_error():
    native = state_frame_capture__make_native()
    assert native.begin_state_capture() is True
    for sequence in (1, 2, 3):
        state_frame_capture__accept_native(native, sequence)
    assert native.get_state_frame().sequence_id == 3
    assert native.get_state_frame().sequence_id == 3
    batch = native.drain_state_frames()
    assert tuple(item.sequence_id for item in batch) == (1, 2, 3)
    assert all(item.valid for item in batch)
    assert native.drain_state_frames() == ()
    native._mark_state_error(RuntimeError("injected_native_state_error"))
    invalid_latest = native.get_state_frame()
    assert not invalid_latest.valid
    assert "injected_native_state_error" in invalid_latest.invalid_reason
    assert invalid_latest.sequence_id == 3
    assert native.drain_state_frames() == ()
    assert all(item.valid for item in batch), "historical accepted frames must remain immutable"
    native.end_state_capture()
    assert native.get_state_frame() == invalid_latest


class state_frame_capture__LegacyNative:
    def __init__(self):
        self.latest = state_frame_capture__state_frame(7)

    def get_state_frame(self):
        return self.latest


def test_state_frame_capture__adapter_keeps_legacy_latest_native_compatible_without_claiming_capture():
    native = state_frame_capture__LegacyNative()
    adapter = state_frame_capture__RokaeRobotAdapter(native_robot=native)
    assert adapter.begin_state_capture() is False
    assert adapter.read_state_frame() is native.latest
    adapter.end_state_capture()
    assert adapter.read_state_frame() is native.latest


def test_state_frame_capture__adapter_delegates_native_capture_and_preserves_nondestructive_latest():
    native = state_frame_capture__make_native()
    adapter = state_frame_capture__RokaeRobotAdapter(native_robot=native)
    assert adapter.begin_state_capture() is True
    state_frame_capture__accept_native(native, 1)
    state_frame_capture__accept_native(native, 2)
    assert adapter.read_state_frame().sequence_id == 2
    assert tuple(item.sequence_id for item in adapter.drain_state_frames()) == (1, 2)
    assert adapter.read_state_frame().sequence_id == 2
    adapter.end_state_capture()
    assert adapter.read_state_frame().sequence_id == 2





# ============================================================================
# Source: tests/test_units_and_frames.py
# ============================================================================

"""Offline coverage for the xCoreSDK-only snapshot collection contract."""



import csv as units_and_frames__csv
import json as units_and_frames__json
import math as units_and_frames__math
import tempfile as units_and_frames__tempfile
import time as units_and_frames__time
import unittest as units_and_frames__unittest
from dataclasses import replace as units_and_frames__replace
from unittest.mock import patch as units_and_frames__patch

import numpy as units_and_frames__np

from collection.snapshot import compose_robot_state_sample as units_and_frames__compose_robot_state_sample
from collection.state import (
    InternalWrenchFrame as units_and_frames__InternalWrenchFrame,
    KinematicStateFrame as units_and_frames__KinematicStateFrame,
    calculate_software_bias as units_and_frames__calculate_software_bias,
    rpy_euler_xyz_rotation_matrix as units_and_frames__rpy_euler_xyz_rotation_matrix,
    rotate_vector as units_and_frames__rotate_vector,
    transform_wrench as units_and_frames__transform_wrench,
)
from config import settings as units_and_frames__settings
from hardware.windows.rokae_xcore import RokaeRobot as units_and_frames__RokaeRobot
from utils.clock import host_time_s as units_and_frames__host_time_s


def units_and_frames___state_frame(now_s: float, *, sequence_id: int = 1, position=(0.2, 0.0, 0.0),
                 velocity=(0.1, 0.0, 0.0), valid: bool = True,
                 reason: str = "", collision: bool | None = None) -> units_and_frames__KinematicStateFrame:
    return units_and_frames__KinematicStateFrame(
        sequence_id=sequence_id,
        host_monotonic_time_s=now_s,
        wall_time_iso="2026-01-01T00:00:00.000+00:00",
        robot_device_time_s=None,
        valid=valid,
        invalid_reason=reason,
        tcp_position_m=tuple(float(value) for value in position),
        tcp_orientation_rad=(0.0, 0.0, 0.0),
        tcp_linear_velocity_mps=tuple(float(value) for value in velocity),
        tcp_angular_velocity_radps=(0.0, 0.0, 0.0),
        velocity_source="numerical_difference_realtime_frame_rpy",
        joint_position_rad=(0.0,) * 6,
        joint_velocity_radps=(0.0,) * 6,
        pose_time_s=now_s,
        joint_time_s=now_s,
        velocity_time_s=now_s,
        operation_state="IDLE",
        collision_state=collision,
        controller_error=None,
        keypad_state=(),
    )


def units_and_frames___wrench_frame(now_s: float, *, sequence_id: int = 1, valid: bool = True,
                  reason: str = "", force=(1.0, 2.0, 3.0),
                  torque=(0.1, 0.2, 0.3),
                  transform_kind: str = "sdk_base") -> units_and_frames__InternalWrenchFrame:
    return units_and_frames__InternalWrenchFrame(
        sequence_id=sequence_id,
        host_monotonic_time_s=now_s,
        wall_time_iso="2026-01-01T00:00:00.000+00:00",
        valid=valid,
        invalid_reason=reason,
        source=units_and_frames__settings.ROBOT_FORCE_SOURCE,
        joint_measured_torque_nm=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        joint_external_torque_nm=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        cartesian_force_raw_n=force,
        cartesian_torque_raw_nm=torque,
        raw_force_frame="world",
        cartesian_force_bias_n=(0.1, 0.2, 0.3),
        cartesian_torque_bias_nm=(0.01, 0.02, 0.03),
        cartesian_force_corrected_n=(force[0] - 0.1, force[1] - 0.2, force[2] - 0.3),
        cartesian_torque_corrected_nm=(torque[0] - 0.01, torque[1] - 0.02, torque[2] - 0.03),
        cartesian_force_base_n=(force[0] - 0.1, force[1] - 0.2, force[2] - 0.3),
        cartesian_torque_base_nm=(torque[0] - 0.01, torque[1] - 0.02, torque[2] - 0.03),
        base_transform_kind=transform_kind,
        force_time_s=now_s,
        torque_time_s=now_s,
        force_query_started_s=now_s - 0.001,
        force_query_finished_s=now_s + 0.001,
    )




class units_and_frames___FakeNativeSpeedRobot:
    def __init__(self) -> None:
        self.speed_mm_s = None

    def setDefaultSpeed(self, speed_mm_s, ec):
        self.speed_mm_s = speed_mm_s


class units_and_frames__UnitAndFrameTests(units_and_frames__unittest.TestCase):
    def test_sdk_boundary_uses_si_and_only_converts_vendor_speed(self):
        pose = [0.3, -0.2, 0.35, 0.0, units_and_frames__np.pi / 2, 0.0]
        self.assertEqual(units_and_frames__RokaeRobot._pose_to_sdk(pose), pose)
        self.assertEqual(units_and_frames__RokaeRobot._pose_from_sdk(pose), pose)
        robot = units_and_frames__RokaeRobot("192.0.2.1", max_linear_speed_m_s=0.8)
        native = units_and_frames___FakeNativeSpeedRobot()
        robot._robot = native
        robot.is_connected = True
        robot.set_speed(25)
        self.assertAlmostEqual(native.speed_mm_s, 200.0)

    def test_snapshot_keeps_sources_times_and_distinct_joint_torques(self):
        sample = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(10.0), units_and_frames___wrench_frame(10.0), sample_time_s=10.01
        )
        self.assertTrue(sample.valid)
        self.assertEqual(sample.joint_measured_torque_nm[0], 1.0)
        self.assertEqual(sample.joint_external_torque_nm[0], 10.0)
        self.assertEqual(sample.cartesian_force_raw_n, (1.0, 2.0, 3.0))
        self.assertEqual(sample.cartesian_force_corrected_n, (0.9, 1.8, 2.7))
        self.assertAlmostEqual(sample.force_query_duration_ms, 2.0)
        self.assertLess(sample.state_internal_skew_ms, 0.001)
        pending = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(11.0),
            units_and_frames___wrench_frame(11.0, transform_kind="rotation_only_pending_robot_validation"),
            sample_time_s=11.01,
            require_verified_base_rotation=True,
        )
        self.assertFalse(pending.valid)
        self.assertIn("base_wrench_rotation_requires_robot_validation", pending.invalid_reason)

        previous = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(12.0, velocity=(0.1, 0.0, 0.0)),
            units_and_frames___wrench_frame(12.0), sample_time_s=12.001,
        )
        accelerated = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(12.02, velocity=(0.2, 0.0, 0.0)),
            units_and_frames___wrench_frame(12.02), sample_time_s=12.021,
            previous_sample=previous,
        )
        self.assertEqual(accelerated.acceleration_source, "numerical_difference")
        self.assertAlmostEqual(accelerated.tcp_linear_acceleration_est_mps2[0], 5.0)

    def test_stale_skew_and_future_timestamps_are_invalid(self):
        stale = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(1.0), units_and_frames___wrench_frame(1.0), sample_time_s=1.2
        )
        self.assertFalse(stale.valid)
        self.assertIn("robot_state_stale", stale.invalid_reason)

        skewed = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(10.0), units_and_frames___wrench_frame(10.03), sample_time_s=10.035,
            max_robot_state_age_s=1.0, max_force_sample_age_s=1.0,
            max_internal_skew_s=0.02,
        )
        self.assertFalse(skewed.valid)
        self.assertIn("robot_state_internal_skew", skewed.invalid_reason)

        future = units_and_frames__compose_robot_state_sample(
            units_and_frames___state_frame(20.1), units_and_frames___wrench_frame(20.1), sample_time_s=20.0
        )
        self.assertFalse(future.valid)
        self.assertIn("robot_state_time_in_future", future.invalid_reason)

    def test_wrench_rotation_and_full_point_transform(self):
        rotation = units_and_frames__rpy_euler_xyz_rotation_matrix((0.0, 0.0, units_and_frames__math.pi / 2.0))
        self.assertTrue(units_and_frames__np.allclose(units_and_frames__rotate_vector(rotation, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0)))
        force_b, torque_b = units_and_frames__transform_wrench(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), rotation, (0.0, 0.0, 1.0)
        )
        self.assertTrue(units_and_frames__np.allclose(force_b, (0.0, 1.0, 0.0)))
        self.assertTrue(units_and_frames__np.allclose(torque_b, (-1.0, 0.0, 0.0)))







if __name__ == "__main__":
    units_and_frames__unittest.main()



# ============================================================================
# Source: tests/test_wrench_deadline_regressions.py
# ============================================================================

"""Deterministic offline protocol clocks; no SDK or hardware connection."""
from dataclasses import asdict as wrench_deadline_regressions__asdict
import json as wrench_deadline_regressions__json
import threading as wrench_deadline_regressions__threading
import time as wrench_deadline_regressions__time
from types import SimpleNamespace as wrench_deadline_regressions__SimpleNamespace

import pytest as wrench_deadline_regressions__pytest

from collection.real_robot_acquisition import RealRobotAcquisition as wrench_deadline_regressions__RealRobotAcquisition
from collection.wrench_process import DurableAudit as wrench_deadline_regressions__DurableAudit, WrenchBudgets as wrench_deadline_regressions__WrenchBudgets, WrenchProcessProvider as wrench_deadline_regressions__WrenchProcessProvider
from hardware.rokae_adapter import RobotWrenchFrame as wrench_deadline_regressions__RobotWrenchFrame


class wrench_deadline_regressions__AuditDouble:
    failure = None

    def __init__(self, delay_ns=0, closed_ok=True):
        self.delay_ns = delay_ns
        self.closed_ok = closed_ok

    def submit(self, event):
        done = wrench_deadline_regressions__threading.Event()
        done.set()
        return dict(done=done, error=None,
                    write_complete_ns=event.get("parent_receive_ns", event["timestamp_ns"]) + self.delay_ns)

    def close(self, budget):
        return self.closed_ok


def wrench_deadline_regressions__provider(tmp_path):
    limits = wrench_deadline_regressions__WrenchBudgets(startup_s=4, query_s=.05, progress_s=2,
        sample_age_s=1, ack_s=.1, log_s=.1, shutdown_s=.2,
        terminate_s=.5, state_age_s=1, skew_s=1,
        provenance="NOT_SAFETY_THRESHOLDS")
    p = wrench_deadline_regressions__WrenchProcessProvider(dict(mode="offline", rate_hz=50), limits, tmp_path / "audit.jsonl")
    p.process = wrench_deadline_regressions__SimpleNamespace(pid=123, is_alive=lambda: True, exitcode=0)
    p.started_ns = 1_000_000_000
    p.writer = wrench_deadline_regressions__AuditDouble()
    return p


@wrench_deadline_regressions__pytest.mark.parametrize("kind", ["QUERY_SUCCESS", "QUERY_ERROR"])
@wrench_deadline_regressions__pytest.mark.parametrize("duration_ns,exceeded", [(40_000_000, False), (60_000_000, True)])
def test_wrench_deadline_regressions__completed_query_cannot_hide_overrun_between_polls(tmp_path, kind, duration_ns, exceeded):
    p = wrench_deadline_regressions__provider(tmp_path)
    p.query_id = 1
    p.query_state = "IN_PROGRESS"
    p.query_start_ns = 1_100_000_000
    start = 1_200_000_000
    end = start + duration_ns
    frame = wrench_deadline_regressions__RobotWrenchFrame(1, start / 1e9, end / 1e9, end / 1e9,
        (start + end) / 2e9, "", "OFFLINE_HOST_QUERY", True, "", "world",
        (1., 2., 3.), (.1, .2, .3), (1.,) * 6, (.5,) * 6)
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
        event_id=1, query_id=1, type=kind, timestamp_ns=end,
        native_start_ns=start, completion_ns=end,
        frame=wrench_deadline_regressions__asdict(frame) if kind == "QUERY_SUCCESS" else None,
        error={"code": 263, "message": "offline"} if kind == "QUERY_ERROR" else None)
    p._accept(event, end + 1_000_000)
    assert ("query_pending_too_long" in p.failure_latch) is exceeded
    assert p.events[-1]["completion_ns"] == end
    assert p.pending_audits[0][0]["completion_ns"] == end
    if kind == "QUERY_ERROR":
        assert "sdk_error" in p.failure_latch


@wrench_deadline_regressions__pytest.mark.parametrize("delay_ns,exceeded", [(30_000_000, False), (110_000_000, True)])
def test_wrench_deadline_regressions__finished_audit_write_uses_actual_completion_deadline(tmp_path, delay_ns, exceeded):
    p = wrench_deadline_regressions__provider(tmp_path)
    p.writer = wrench_deadline_regressions__AuditDouble(delay_ns)
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
        event_id=1, query_id=0, type="PROCESS_START", timestamp_ns=1_100_000_000)
    p._accept(event, 1_100_000_000)
    # A delayed poll must neither hide a late fsync nor penalize a timely one.
    p.poll(now_ns=1_300_000_000)
    assert ("logger_backpressure" in p.failure_latch) is exceeded


@wrench_deadline_regressions__pytest.mark.parametrize("native_age_ns,exceeded", [(10_000_000, False), (60_000_000, True)])
def test_wrench_deadline_regressions__native_deadline_excludes_boundary_ack_wait(tmp_path, native_age_ns, exceeded):
    p = wrench_deadline_regressions__provider(tmp_path)
    now = 1_400_000_000
    p.progress_ns = 1_100_000_000
    p.query_id = 1
    p.query_state = "IN_PROGRESS"
    p.query_start_ns = 1_100_000_000
    p.native_boundary_ns = 1_200_000_000
    p.latest_slot.put(dict(run_id=p.run_id, pid=123, query_id=1,
        query_in_progress=True, native_start_ns=now-native_age_ns))
    p.poll(now_ns=now)
    assert ("query_pending_too_long" in p.failure_latch) is exceeded


@wrench_deadline_regressions__pytest.mark.parametrize("disconnected,fault,closed_ok,normal", [
    (True, False, True, True), (True, True, True, False),
    (False, False, True, False), (True, False, False, False),
])
def test_wrench_deadline_regressions__cleanup_requires_confirmed_disconnect_and_reports_all_faults(
        tmp_path, disconnected, fault, closed_ok, normal):
    p = wrench_deadline_regressions__provider(tmp_path)
    p.started_ns = wrench_deadline_regressions__time.perf_counter_ns()
    p.process.is_alive = lambda: False
    p.offline_disconnect_completed = disconnected
    p.writer = wrench_deadline_regressions__AuditDouble(closed_ok=closed_ok)
    if fault:
        p.fail("offline_test_fault")
    result = p.stop()
    assert result["NORMAL_CLEANUP"] is normal
    assert result["failure_latch"] == p.failure_latch
    logged = next(e["outcome"] for e in p.events if e["type"] == "CLEANUP_OUTCOME")
    if fault or not disconnected:
        assert logged["NORMAL_CLEANUP"] is False
    if not closed_ok:
        assert "audit_cleanup_unconfirmed" in result["failure_latch"]
    assert p.stop() == result


def test_wrench_deadline_regressions__startup_wait_returns_immediately_after_latched_failure():
    acquisition = object.__new__(wrench_deadline_regressions__RealRobotAcquisition)
    acquisition._stop_event = wrench_deadline_regressions__threading.Event()
    acquisition._background_error = None
    ticks = iter([0., 0., 5.])
    acquisition.clock = wrench_deadline_regressions__SimpleNamespace(now_s=lambda: next(ticks))
    calls = []

    def health():
        calls.append(True)
        return wrench_deadline_regressions__SimpleNamespace(valid=False, wrench_failure_latch=("logger_backpressure",))

    acquisition.latest_health = health
    assert acquisition.wait_until_healthy(timeout_s=10) is False
    assert calls == [True]


@wrench_deadline_regressions__pytest.mark.parametrize("write_fails", [False, True])
def test_wrench_deadline_regressions__queued_audit_batch_is_ordered_and_finally_fsynced(tmp_path, monkeypatch, write_fails):
    import collection.wrench_process as module
    entered = wrench_deadline_regressions__threading.Event()
    release = wrench_deadline_regressions__threading.Event()
    fsync_calls = []
    original_fsync = module.os.fsync

    def hook(event):
        if event["index"] == 0:
            entered.set()
            assert release.wait(2)
        if write_fails and event["index"] == 3:
            raise OSError("offline batch write failure")

    def fsync(fd):
        fsync_calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", fsync)
    path = tmp_path / "batch.jsonl"
    audit = wrench_deadline_regressions__DurableAudit(path, write_hook=hook)
    tickets = []
    try:
        tickets.append(audit.submit(dict(index=0)))
        assert entered.wait(2)
        tickets.extend(audit.submit(dict(index=i)) for i in range(1, 6))
        assert all(not t["done"].is_set() for t in tickets)
        release.set()
        assert all(t["done"].wait(2) for t in tickets)
    finally:
        release.set()
        closed = audit.close(2)
    assert closed is (not write_fails)
    assert tickets[0]["error"] is None
    if write_fails:
        assert all(t["error"] and t["write_complete_ns"] is None for t in tickets[1:])
        assert len(fsync_calls) == 0
    else:
        assert [e["index"] for e in map(wrench_deadline_regressions__json.loads, path.read_text().splitlines())] == list(range(6))
        assert len(fsync_calls) == 1
        assert len({t["write_complete_ns"] for t in tickets[1:]}) == 1



# ============================================================================
# Source: tests/test_wrench_production_process.py
# ============================================================================

"""Offline OS-process tests. All budgets are NOT_SAFETY_THRESHOLDS."""
from dataclasses import asdict as wrench_production_process__asdict, replace as wrench_production_process__replace
import json as wrench_production_process__json
import os as wrench_production_process__os
from pathlib import Path as wrench_production_process__Path
import queue as wrench_production_process__queue
import threading as wrench_production_process__threading
import time as wrench_production_process__time
from types import SimpleNamespace as wrench_production_process__SimpleNamespace

import pytest as wrench_production_process__pytest

from collection.wrench_process import DurableAudit as wrench_production_process__DurableAudit, Slot as wrench_production_process__Slot, WrenchBudgets as wrench_production_process__WrenchBudgets, WrenchProcessProvider as wrench_production_process__WrenchProcessProvider


@wrench_production_process__pytest.fixture
def wrench_production_process__budgets():
    return wrench_production_process__WrenchBudgets(startup_s=4, query_s=.25, progress_s=2,
        sample_age_s=1, ack_s=2, log_s=.25, shutdown_s=.2,
        terminate_s=.5, state_age_s=1, skew_s=1,
        provenance="NOT_SAFETY_THRESHOLDS")


@wrench_production_process__pytest.fixture
def wrench_production_process__make_provider(tmp_path, request, wrench_production_process__budgets):
    providers = []
    root = wrench_production_process__Path(wrench_production_process__os.environ.get("WRENCH_TEST_OUTPUT", str(tmp_path))) / request.node.name.replace("/", "_")
    root.mkdir(parents=True, exist_ok=True)

    def make(*, overrides=None, writer=None, limits=None, **config):
        directory = root / str(len(providers))
        directory.mkdir(exist_ok=False)
        args = dict(mode="offline", rate_hz=50, **config)
        provider = wrench_production_process__WrenchProcessProvider(args, limits or wrench_production_process__budgets, directory / "query_events.jsonl",
                                        **({"audit_factory": writer} if writer else {}))
        provider.trace = []
        provider.output = directory
        provider.start()
        providers.append(provider)
        return provider

    yield make
    for provider in providers:
        outcome = provider.cleanup or provider.stop()
        (provider.output / "cleanup_outcome.json").write_text(wrench_production_process__json.dumps(outcome, indent=2))
        (provider.output / "health_timeline.json").write_text(wrench_production_process__json.dumps(provider.trace, indent=2))
        (provider.output / "gap_drop_diagnostics.json").write_text(wrench_production_process__json.dumps({
            "gaps": provider.gaps, "failures": provider.failure_latch,
            "budget_provenance": provider.budgets.provenance}, indent=2))
        events = [e for e in provider.events if e["type"].startswith(("PROCESS", "DISCONNECT", "FORCED", "CLEANUP"))]
        (provider.output / "process_lifecycle.json").write_text(wrench_production_process__json.dumps(events, indent=2))
        assert not provider.process.is_alive(), "offline child leaked"


def wrench_production_process__until(provider, predicate, seconds=6):
    end = wrench_production_process__time.perf_counter() + seconds
    while wrench_production_process__time.perf_counter() < end:
        state = provider.poll()
        provider.trace.append(state)
        if predicate(state):
            return state
        wrench_production_process__time.sleep(.002)
    raise AssertionError(f"condition not reached: {provider.health()}")


def test_wrench_production_process__normal_bounded_async_durable_protocol(wrench_production_process__make_provider):
    p = wrench_production_process__make_provider()
    wrench_production_process__until(p, lambda h: p.query_id >= 3 and h["STREAM_HEALTH"])
    assert p.last_good.valid
    assert p.last_good.host_monotonic_time_s < wrench_production_process__time.perf_counter()
    outcome = p.stop()
    assert outcome["NORMAL_CLEANUP"]
    assert outcome["OFFLINE_DISCONNECT_COMPLETED"]
    assert outcome["GRACEFUL_SDK_DISCONNECT"] is None  # no SDK used
    events = [wrench_production_process__json.loads(x) for x in p.audit_path.read_text().splitlines()]
    summaries = [row for row in events if row["type"] == "AUDIT_WRITE_SUMMARY"]
    assert len(summaries) == 1
    child_events = [row for row in events if row.get("origin") != "supervisor"]
    assert summaries[0]["pending_child_event_count"] == 0
    assert summaries[0]["written_child_event_count"] == len(child_events)
    for event in events:
        if event["type"] == "QUERY_SUCCESS":
            assert event["intent_ns"] <= event["native_start_ns"]
            assert event["native_start_ns"] <= event["completion_ns"]


def test_wrench_production_process__50hz_publication_is_not_serialized_by_each_fsync(wrench_production_process__make_provider, monkeypatch):
    real_fsync = wrench_production_process__os.fsync

    def slow_fsync(fd):
        wrench_production_process__time.sleep(.03)
        return real_fsync(fd)

    monkeypatch.setattr(wrench_production_process__os, "fsync", slow_fsync)
    p = wrench_production_process__make_provider()
    wrench_production_process__until(p, lambda h: p.query_id >= 25 and h["STREAM_HEALTH"])
    successes = [e for e in p.events if e["type"] == "QUERY_SUCCESS"]
    assert len(successes) >= 20
    elapsed_s = (successes[-1]["native_start_ns"] - successes[0]["native_start_ns"]) / 1e9
    observed_hz = (len(successes) - 1) / elapsed_s
    assert observed_hz >= 45
    assert not p.failure_latch
    # Batching depends on scheduler timing; the invariant is that fsync count
    # remains far below event count while publication holds 50 Hz.
    assert p.writer.fsync_count < len(successes)


@wrench_production_process__pytest.mark.parametrize("behavior", ["block", "never", "gil_block"])
def test_wrench_production_process__blocking_parent_progress_and_forced_cleanup(wrench_production_process__make_provider, behavior):
    p = wrench_production_process__make_provider(behavior=behavior, delay_s=10)
    wrench_production_process__until(p, lambda h: h["CURRENT_QUERY_STATE"] == "IN_PROGRESS")
    ticks = 0
    simulated_rt = 0
    deadline = wrench_production_process__time.perf_counter() + 1
    while wrench_production_process__time.perf_counter() < deadline and not p.failure_latch:
        simulated_rt += 1  # parent RT surrogate, no native SDK
        p.trace.append(p.poll())
        ticks += 1
        wrench_production_process__time.sleep(.002)
    assert ticks > 10 and simulated_rt == ticks
    assert "query_pending_too_long" in p.failure_latch
    assert p.health()["completion"] is None
    assert not any(e["type"] in ("QUERY_SUCCESS", "QUERY_ERROR") for e in p.events)
    result = p.stop()
    assert result["FORCED_WORKER_TERMINATION"]
    assert result["GRACEFUL_SDK_DISCONNECT"] is False
    assert result["CONTROLLER_SESSION_RECOVERY"] is None


def test_wrench_production_process__delayed_263_preserves_last_good(wrench_production_process__make_provider):
    p = wrench_production_process__make_provider(behavior="error263", delay_s=.04, good_queries=1)
    wrench_production_process__until(p, lambda h: h["LAST_ERROR"] is not None)
    h = p.health()
    assert h["LATEST_GOOD_SAMPLE_VALID"]
    assert not h["STREAM_HEALTH"]
    assert h["LAST_ERROR"]["code"] == 263
    assert p.last_good.sequence_id == 1
    assert len([e for e in p.events if e["type"] == "QUERY_ERROR"]) == 1
    assert p.query_id == 2  # no retry


def test_wrench_production_process__crash_pending_no_fabricated_completion(wrench_production_process__make_provider):
    p = wrench_production_process__make_provider(behavior="crash")
    wrench_production_process__until(p, lambda h: "worker_exited" in h["FAILURE_LATCH"])
    assert p.process.exitcode == 73
    assert p.health()["completion"] is None


def test_wrench_production_process__stale_last_good_and_failure_stays_latched(wrench_production_process__make_provider, wrench_production_process__budgets):
    p = wrench_production_process__make_provider(behavior="block", good_queries=1, delay_s=.3,
                      limits=wrench_production_process__replace(wrench_production_process__budgets, query_s=1, sample_age_s=.08, shutdown_s=1))
    wrench_production_process__until(p, lambda h: "wrench_sample_stale_or_clock_invalid" in h["FAILURE_LATCH"])
    stamp = p.last_good.host_monotonic_time_s
    assert p.health()["LATEST_GOOD_SAMPLE_VALID"] and not p.health()["STREAM_HEALTH"]
    wrench_production_process__until(p, lambda h: p.last_completed_ns is not None and p.query_state == "SUCCEEDED" and p.last_good.sequence_id == 2)
    assert p.last_good.host_monotonic_time_s > stamp
    assert not p.health()["STREAM_HEALTH"]
    assert p.failure_latch


@wrench_production_process__pytest.mark.parametrize("phase", ["startup", "ack", "query", "completion", "disconnect"])
def test_wrench_production_process__shutdown_phases(wrench_production_process__make_provider, phase):
    config = {}
    if phase == "startup": config["startup_delay_s"] = 10
    if phase == "query": config.update(behavior="never")
    if phase == "disconnect": config["disconnect_delay_s"] = 10
    p = wrench_production_process__make_provider(**config)
    if phase == "startup": wrench_production_process__until(p, lambda h: h["CURRENT_QUERY_STATE"] == "STARTING")
    elif phase == "query":
        wrench_production_process__until(p, lambda h: bool((p.latest_slot.read() or {}).get("query_in_progress")))
    elif phase == "completion": wrench_production_process__until(p, lambda h: h["LAST_COMPLETED_QUERY_TIME"] is not None)
    elif phase == "disconnect": wrench_production_process__until(p, lambda h: h["STREAM_HEALTH"])
    # ack case stops before polling, while child awaits its first durable ACK.
    result = p.stop()
    assert result["process_exited"]
    if phase in ("startup", "query", "disconnect"):
        assert result["FORCED_WORKER_TERMINATION"]
        assert result["GRACEFUL_SDK_DISCONNECT"] is False


def test_wrench_production_process__missing_first_heartbeat(wrench_production_process__make_provider, wrench_production_process__budgets):
    p = wrench_production_process__make_provider(pre_start_delay_s=10, limits=wrench_production_process__replace(wrench_production_process__budgets, startup_s=.15))
    wrench_production_process__until(p, lambda h: "missing_first_heartbeat" in h["FAILURE_LATCH"])


def test_wrench_production_process__independent_supervisor_prevents_ack_starvation(wrench_production_process__make_provider, wrench_production_process__budgets):
    p = wrench_production_process__make_provider(limits=wrench_production_process__replace(wrench_production_process__budgets, ack_s=.1))
    deadline=wrench_production_process__time.perf_counter()+4
    while p.audit_slot.read() is None and wrench_production_process__time.perf_counter()<deadline:
        wrench_production_process__time.sleep(.002)
    wrench_production_process__time.sleep(.2)  # caller does not poll; supervisor must continue ACK service
    state = p.poll()
    assert state["STREAM_HEALTH"]
    assert not any("ack_timeout" in x for x in state["FAILURE_LATCH"])
    assert p.query_id >= 3
    assert p.last_good is not None


def test_wrench_production_process__query_budget_starts_only_after_native_boundary(tmp_path, wrench_production_process__budgets):
    p = wrench_production_process__WrenchProcessProvider(
        dict(mode="offline", rate_hz=50),
        wrench_production_process__replace(wrench_production_process__budgets, query_s=.05, progress_s=1.0),
        tmp_path / "audit.jsonl",
    )
    now = wrench_production_process__time.perf_counter_ns()
    p.process = wrench_production_process__SimpleNamespace(pid=123, is_alive=lambda: True)
    p.started_ns = now - 200_000_000
    p.progress_ns = now - 60_000_000
    p.query_state = "START_INTENT"
    p.query_start_ns = now - 60_000_000
    p.lifecycle_state = "READY"
    health = p.poll(now_ns=now)
    assert "query_pending_too_long" not in health["FAILURE_LATCH"]


@wrench_production_process__pytest.mark.parametrize("mode", ["backpressure", "failure"])
def test_wrench_production_process__logger_faults(wrench_production_process__make_provider, mode, wrench_production_process__budgets):
    release = wrench_production_process__threading.Event()
    def hook(event):
        if mode == "failure": raise OSError("offline disk failure")
        release.wait(2)
    p = wrench_production_process__make_provider(writer=lambda path: wrench_production_process__DurableAudit(path, write_hook=hook))
    try:
        wrench_production_process__until(p, lambda h: any("logger" in x for x in h["FAILURE_LATCH"]))
        assert not p.health()["STREAM_HEALTH"]
        # Admission may allow bounded progress before the asynchronous writer
        # failure is observed, but the fault must remain latched.
        assert p.failure_latch
    finally:
        release.set()


def test_wrench_production_process__periodic_fsync_failure_is_latched(wrench_production_process__make_provider, monkeypatch):
    import collection.wrench_process as module

    def fail_fsync(_fd):
        raise OSError("offline fsync failure")

    monkeypatch.setattr(module.os, "fsync", fail_fsync)
    p = wrench_production_process__make_provider(
        writer=lambda path: wrench_production_process__DurableAudit(path, sync_interval_s=.01)
    )
    health = wrench_production_process__until(
        p,
        lambda h: any("logger_failure:audit_fsync:OSError" in x for x in h["FAILURE_LATCH"]),
    )
    assert not health["STREAM_HEALTH"]
    assert p.stop_flag.value == 1


@wrench_production_process__pytest.mark.parametrize("fault", ["gap", "duplicate", "order", "pid", "run", "clock", "overflow", "disconnect_order", "stopped_order"])
def test_wrench_production_process__protocol_faults_are_latched(tmp_path, wrench_production_process__budgets, fault):
    p = wrench_production_process__WrenchProcessProvider(dict(mode="offline", rate_hz=50), wrench_production_process__budgets, tmp_path/'audit.jsonl')
    # Protocol unit test, no process/native provider required.
    p.process = wrench_production_process__SimpleNamespace(pid=123, is_alive=lambda: True)
    p.started_ns = wrench_production_process__time.perf_counter_ns()
    p.writer = wrench_production_process__DurableAudit(p.audit_path)
    now = wrench_production_process__time.perf_counter_ns()
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
                 event_id=1, query_id=0, type="PROCESS_START", timestamp_ns=now)
    if fault == "gap": event['event_id'] = 3
    if fault == "duplicate": p.expected_event = 2
    if fault == "order": event.update(type="QUERY_SUCCESS", query_id=1)
    if fault == "pid": event['pid'] = 999
    if fault == "run": event['run_id'] = 'wrong'
    if fault == "clock": event['timestamp_ns'] = -1
    if fault == "disconnect_order": event['type'] = 'DISCONNECT_COMPLETE'
    if fault == "stopped_order": event['type'] = 'PROCESS_STOPPED'
    if fault == "overflow":
        with wrench_production_process__pytest.raises(ValueError, match="overflow"):
            p.audit_slot.put(dict(event, padding='x'*70000))
        p.fatal_slot.put({'reason':'audit_overflow'})
    else:
        p.audit_slot.put(event)
    h = p.poll()
    assert h['FAILURE_LATCH'] and not h['STREAM_HEALTH']
    p.writer.close(1)


def test_wrench_production_process__invalid_supervisor_clock(wrench_production_process__make_provider):
    p = wrench_production_process__make_provider()
    wrench_production_process__until(p, lambda h: h['STREAM_HEALTH'])
    p.poll(-1)
    assert 'invalid_supervisor_clock' in p.failure_latch


def test_wrench_production_process__audit_queue_overflow_is_explicit(tmp_path):
    release = wrench_production_process__threading.Event()
    writer = wrench_production_process__DurableAudit(tmp_path/'overflow.jsonl', write_hook=lambda _:release.wait(2))
    try:
        with wrench_production_process__pytest.raises(wrench_production_process__queue.Full):
            for i in range(1000): writer.submit({'event_id':i})
    finally:
        release.set()
        writer.close(2)


def test_wrench_production_process__republished_acknowledged_event_is_rejected(wrench_production_process__make_provider):
    p=wrench_production_process__make_provider(behavior='never')
    wrench_production_process__until(p,lambda h:h['CURRENT_QUERY_STATE']=='IN_PROGRESS')
    # Offline protocol injection only; a conforming child is now in the stub.
    event=p.audit_slot.read()
    p.audit_slot.put(event)
    p.poll()
    assert any('sequence' in x for x in p.failure_latch)


def test_wrench_production_process__runtime_state_freshness_failure_is_latched(wrench_production_process__make_provider,tmp_path):
    FakeAdapter = real_robot_acquisition__FakeAdapter
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    p=wrench_production_process__make_provider()
    wrench_production_process__until(p,lambda h:h['STREAM_HEALTH'])
    p.arm_runtime_grace()
    acquisition=RealRobotAcquisition(FakeAdapter(),EpisodeLogger(tmp_path/'unused'),wrench_provider=p)
    acquisition._runtime_freshness_armed=True
    stale=FakeAdapter().read_state_frame()
    acquisition._latest_state=wrench_production_process__replace(stale,host_monotonic_time_s=wrench_production_process__time.perf_counter()-10)
    acquisition._latest_wrench=p.last_good
    h=acquisition.latest_health()
    assert not h.valid
    assert 'state_freshness_failure' in p.failure_latch






def test_wrench_production_process__acquisition_process_integration(wrench_production_process__make_provider, tmp_path):
    FakeAdapter = real_robot_acquisition__FakeAdapter
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    p = wrench_production_process__make_provider(behavior='error263', good_queries=2, delay_s=.05)
    # Integration owns start; use a fresh provider with the same protocol config.
    p.stop()
    provider = wrench_production_process__WrenchProcessProvider(dict(mode='offline',rate_hz=50,behavior='error263',good_queries=2,delay_s=.05),
                                    p.budgets,tmp_path/'integration_events.jsonl')
    logger = EpisodeLogger(tmp_path/'episode').start()
    adapter = FakeAdapter()
    def forbidden(): raise AssertionError('parent native wrench called')
    adapter.read_internal_wrench = forbidden
    acquisition = RealRobotAcquisition(adapter, logger, wrench_provider=provider)
    acquisition.start()
    try:
        deadline=wrench_production_process__time.perf_counter()+6
        while wrench_production_process__time.perf_counter()<deadline and 'sdk_error' not in provider.failure_latch: wrench_production_process__time.sleep(.005)
        h=acquisition.latest_health()
        assert 'sdk_error' in provider.failure_latch
        assert h.wrench_valid and not h.valid
        assert h.wrench_thread_alive  # actual consumer
        assert h.wrench_failure_latch
        assert logger.row_counts['robot_wrench'] == 2
    finally:
        acquisition.stop()
        logger.close(completed=False)
    assert not provider.process.is_alive()


def test_wrench_production_process__missing_provider_refuses_before_connect(tmp_path):
    FakeAdapter = real_robot_acquisition__FakeAdapter
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    adapter=FakeAdapter()
    acquisition=RealRobotAcquisition(adapter,EpisodeLogger(tmp_path/'not_started'))
    with wrench_production_process__pytest.raises(PermissionError): acquisition.start()
    assert adapter.calls == []


# Personal experiment workflow regressions. All device interfaces below are fakes.
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from config.bundle import BUNDLE_PATH, configuration
from control.execution_preflight import OPERATOR_CONFIRMATION, evaluate_offline_execution_request, evaluate_execution_preflight
from control.robot_trajectory_executor import RokaeMotionExecutor
from control.start_anchor import StartAnchor, FixedTcpOrientation
from control.start_anchored_relative_trajectory import RehabFrameConfig, build_start_anchored_relative_trajectory
from safety.experiment_safety import ExperimentSafetyConfig
from scripts import lab_session as personal_round


@pytest.fixture
def personal_context():
    frame = RehabFrameConfig((1, 0, 0), (0, 0, 1))
    trajectory, audit, _ = build_start_anchored_relative_trajectory(current_tcp_start_pose=(.5, 0, .5, 0, 0, 0), rehab_frame=frame)
    first = trajectory.iloc[0]
    anchor = StartAnchor(capture_host_time_s=1., tcp_pose_base=(.5, 0, .5, 0, 0, 0),
        tcp_position_base_m=(.5, 0, .5), tcp_orientation=FixedTcpOrientation((0, 0, 0)),
        robot_joint_positions=(0,) * 6, trajectory_id=audit.trajectory_id,
        reference_start_q_hip=float(first.q_hip_ref), reference_start_q_knee=float(first.q_knee_ref), anchor_id="test")
    limits = dict(configuration()["motion"])
    limits.update(max_start_anchor_position_error_m=.005, max_start_anchor_orientation_error_rad=.05,
                  max_command_lateness_s=.05, rt_filter_hz=50, rt_network_tolerance_percent=20,
                  workspace_min_base_m=(-2., -2., -2.), workspace_max_base_m=(2., 2., 2.))
    safety = ExperimentSafetyConfig(**limits)
    request = dict(mode="execute", enable_motion=True, operator_confirmation=OPERATOR_CONFIRMATION,
                   requested_anchor_id="test", frame=frame, anchor=anchor, safety=safety, trajectory=trajectory, audit=audit)
    health = SimpleNamespace(valid=True, invalid_reason="", state_thread_alive=True, wrench_thread_alive=True,
        alignment_thread_alive=True, state_age_s=.001, wrench_age_s=.001, state_wrench_skew_s=.001,
        force_magnitude_n=1., torque_magnitude_nm=1.)
    summary = dict(connected=True, state_valid=True, operation_state="RT_CONTROLLING",
                   collision_state_query_valid=True, collision_state=False, joint_position_rad=[0.] * 6,
                   robot_metadata={"joint_soft_limits_rad": [[-3., 3.]] * 6})
    adapter = SimpleNamespace(is_connected=True, get_robot_state_summary=lambda: summary)
    events = []
    logger = SimpleNamespace(ready=True, healthy=True, healthy_signal=True, assert_healthy=lambda: None,
        update_metadata=lambda v: None, append_trajectory_command=lambda **v: events.append("log"))
    acquisition = SimpleNamespace(adapter=adapter, latest_health=lambda: health,
        latest_state_frame=lambda: SimpleNamespace(valid=True, tcp_position_m=(.5, 0, .5), tcp_orientation_rad=(0, 0, 0)))
    motion = SimpleNamespace(request_stop=lambda reason: events.append("stop"),
        attach_externally_prepared=lambda **kwargs: events.append("attach"),
        start_cartesian_hold=lambda pose: events.append("hold"), send_cartesian_target=lambda pose: events.append("target"))
    return SimpleNamespace(**request, request=request, health=health, summary=summary, logger=logger,
                           acquisition=acquisition, motion=motion, events=events, adapter=adapter)


def test_personal_unreviewed_frame_and_anchor_need_no_signoff(personal_context):
    c = personal_context
    assert not c.anchor.reviewed and not c.frame.reviewed
    evaluate_offline_execution_request(**c.request).require_allowed()


@pytest.mark.parametrize("changes", [{"enable_motion": False}, {"operator_confirmation": ""}, {"requested_anchor_id": "other"}])
def test_personal_explicit_motion_and_anchor_required(personal_context, changes):
    assert not evaluate_offline_execution_request(**(personal_context.request | changes)).allowed


@pytest.mark.parametrize("field,value", [("max_tcp_speed_m_s", 1e-6), ("max_tcp_acceleration_m_s2", 1e-8), ("workspace_max_base_m", (.1, .1, .1))])
def test_personal_physical_trajectory_limits_still_apply(personal_context, field, value):
    c = personal_context
    safety = replace(c.safety, **{field: value})
    assert not evaluate_offline_execution_request(**(c.request | {"safety": safety})).allowed


def test_personal_tampered_derivative_rejected(personal_context):
    c = personal_context
    trajectory = c.trajectory.copy()
    trajectory.loc[50, "tcp_vx_base"] = 99.
    assert not evaluate_offline_execution_request(**(c.request | {"trajectory": trajectory})).allowed


@pytest.mark.parametrize("field,value", [("valid", False), ("state_age_s", 10.), ("wrench_age_s", 10.), ("force_magnitude_n", 1000.), ("torque_magnitude_nm", float("nan"))])
def test_personal_live_measurements_required(personal_context, field, value):
    c = personal_context
    setattr(c.health, field, value)
    result = evaluate_execution_preflight(**c.request, acquisition_health=c.health, logger=c.logger,
        robot_adapter=c.adapter, current_tcp_pose_base=c.anchor.tcp_pose_base)
    assert not result.allowed


def test_personal_collision_feedback_remains_required(personal_context):
    c = personal_context
    c.summary["collision_state_query_valid"] = False
    result = evaluate_execution_preflight(**c.request, acquisition_health=c.health, logger=c.logger,
        robot_adapter=c.adapter, current_tcp_pose_base=c.anchor.tcp_pose_base)
    assert not result.allowed


def personal_executor(c):
    clock = SimpleNamespace(value=1.)
    clock.now_s = lambda: clock.value
    executor = RokaeMotionExecutor(c.motion, c.acquisition, c.logger, c.safety, clock=clock)
    executor._wait_until = lambda deadline: setattr(clock, "value", deadline)
    return executor, clock


def test_personal_executor_completes_and_stops(personal_context):
    c = personal_context
    executor, _ = personal_executor(c)
    result = executor.execute(c.trajectory, evaluate_offline_execution_request(**c.request))
    assert result.completed and result.commands_dispatched == len(c.trajectory)
    assert c.events.count("log") == len(c.trajectory)
    assert c.events[-1] == "stop"
    assert c.events.index("log") < c.events.index("hold")
    executor.request_stop("again")
    assert c.events.count("stop") == 1


@pytest.mark.parametrize("source,allowed", [("controller", True), ("query", False), ("events", False)])
def test_unknown_collision_feedback_through_preflight_and_executor(personal_context, source, allowed):
    c = personal_context
    c.summary.update(collision_state_source=source, collision_state_query_valid=False,
                     collision_state=None)
    preflight = evaluate_execution_preflight(**c.request, acquisition_health=c.health,
        logger=c.logger, robot_adapter=c.adapter, current_tcp_pose_base=c.anchor.tcp_pose_base)
    assert preflight.allowed is allowed
    executor, _ = personal_executor(c)
    if allowed:
        result = executor.execute(c.trajectory, preflight)
        assert result.completed and result.commands_dispatched == len(c.trajectory)
        assert "hold" in c.events and c.events[-1] == "stop"
    else:
        with pytest.raises(RuntimeError, match="collision protection"):
            executor.execute(c.trajectory, evaluate_offline_execution_request(**c.request))
        assert "hold" not in c.events and c.events[-1] == "stop"


@pytest.mark.parametrize("field,value,reason", [
    ("state_age_s", .026, "state_stale"),
    ("wrench_age_s", .051, "wrench_stale"),
    ("state_wrench_skew_s", .041, "state_wrench_skew_limit_exceeded"),
    ("force_magnitude_n", 101., "force_limit_exceeded"),
    ("torque_magnitude_nm", 201., "torque_limit_exceeded"),
    ("valid", False, "controller_realtime_motion_error"),
])
def test_controller_mode_stops_after_motion_begins_on_measurement_fault(personal_context, field, value, reason):
    c = personal_context
    c.summary.update(collision_state_source="controller", collision_state_query_valid=False,
                     collision_state=None)
    preflight = evaluate_execution_preflight(**c.request, acquisition_health=c.health,
        logger=c.logger, robot_adapter=c.adapter, current_tcp_pose_base=c.anchor.tcp_pose_base)
    preflight.require_allowed()
    def target(pose):
        c.events.append("target")
        setattr(c.health, field, value)
        if field == "valid":
            c.health.invalid_reason = reason
    c.motion.send_cartesian_target = target
    executor, _ = personal_executor(c)
    with pytest.raises(RuntimeError, match=reason):
        executor.execute(c.trajectory, preflight)
    assert "hold" in c.events
    assert c.events.count("target") == 1
    assert c.events.count("stop") == 1 and c.events[-1] == "stop"


@pytest.mark.parametrize("remaining,allowed", [(0.20, True), (0.0, False)])
def test_wrench_stale_grace_allows_only_short_runtime_gap(personal_context, remaining, allowed):
    c = personal_context
    c.summary.update(collision_state_source="controller", collision_state_query_valid=False,
                     collision_state=None)
    c.health.valid = False
    c.health.invalid_reason = "wrench_stale_or_unavailable;state_wrench_skew_unavailable_or_exceeded"
    c.health.wrench_stale_grace_active = remaining > 0
    c.health.wrench_stale_grace_remaining_s = remaining
    c.health.wrench_age_s = .20 if remaining > 0 else .40
    c.health.state_wrench_skew_s = .10
    c.health.force_magnitude_n = None
    c.health.torque_magnitude_nm = None
    executor, _ = personal_executor(c)
    reasons = executor._runtime_reasons()
    assert (not reasons) is allowed


@pytest.mark.parametrize("delay", [.240, .312])
def test_runtime_wrench_block_recovers_across_provider_health_and_executor(
        wrench_production_process__make_provider, wrench_production_process__budgets,
        personal_context, tmp_path, delay):
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    limits = wrench_production_process__replace(wrench_production_process__budgets,
        query_s=.05, sample_age_s=.05, runtime_stale_grace_s=.30, progress_s=.4,
        state_age_s=.025, skew_s=.04, shutdown_s=1)
    p = wrench_production_process__make_provider(limits=limits, behavior="block",
        delay_s=delay, good_queries=12, block_once=True)
    wrench_production_process__until(p, lambda h: p.last_good is not None)
    c = personal_context
    acquisition = RealRobotAcquisition(c.adapter, EpisodeLogger(tmp_path / "unused"), wrench_provider=p)
    acquisition.latest_state_frame = lambda: wrench_production_process__replace(
        real_robot_acquisition__FakeAdapter().read_state_frame(),
        host_monotonic_time_s=wrench_production_process__time.perf_counter())
    acquisition._thread_alive = lambda name: True
    acquisition._latest_wrench = p.last_good
    acquisition.arm_runtime_grace()
    c.acquisition = acquisition
    executor, _ = personal_executor(c)
    saw_gap = False
    baseline = None
    deadline = wrench_production_process__time.perf_counter() + 4
    while wrench_production_process__time.perf_counter() < deadline:
        acquisition._latest_wrench = p.last_good
        h = acquisition.latest_health()
        if p.query_id == 13 and h.wrench_age_s > .05:
            saw_gap = True
            if baseline is None:
                baseline = acquisition._latest_wrench
            assert acquisition._latest_wrench == baseline
            assert not h.valid and not h.wrench_valid
            assert h.wrench_stale_grace_active
            assert h.force_magnitude_n is None and h.torque_magnitude_nm is None
            assert not executor._runtime_reasons()
        if acquisition._latest_wrench.sequence_id >= 14:
            assert h.valid
            break
        assert not p.failure_latch
        wrench_production_process__time.sleep(.002)
    else:
        pytest.fail(str(p.health()))
    assert saw_gap and baseline.sequence_id == 12
    assert p.native_query_overrun_count == 1
    slow = next(e for e in p.events if e["type"] == "QUERY_SUCCESS" and e["query_id"] == 13)
    assert slow["frame"]["sequence_id"] == 13  # raw measurement retained
    assert not p.failure_latch


@pytest.mark.parametrize("armed,delay,once", [(False, .240, True), (True, .5, True), (True, .12, False)])
def test_runtime_wrench_gap_deadline_is_latched(
        wrench_production_process__make_provider, wrench_production_process__budgets, armed, delay, once):
    limits = wrench_production_process__replace(wrench_production_process__budgets,
        query_s=.05, sample_age_s=.05, runtime_stale_grace_s=.30, progress_s=.4, shutdown_s=1)
    p = wrench_production_process__make_provider(limits=limits, behavior="block",
        delay_s=delay, good_queries=10, block_once=once)
    wrench_production_process__until(p, lambda h: p.last_good is not None)
    if armed:
        p.arm_runtime_grace()
    wrench_production_process__until(p, lambda h: bool(h["FAILURE_LATCH"]))
    assert p.last_good.sequence_id == 10
    assert set(p.failure_latch) & {"query_pending_too_long", "wrench_sample_stale_or_clock_invalid"}
    failures = tuple(p.failure_latch)
    p.stop()
    assert not p.health()["STREAM_HEALTH"]
    assert all(reason in p.failure_latch for reason in failures)


@pytest.mark.parametrize("behavior,reason", [("error263", "sdk_error"), ("crash", "worker_exited")])
def test_runtime_wrench_grace_never_hides_worker_fault(
        wrench_production_process__make_provider, behavior, reason):
    p = wrench_production_process__make_provider(behavior=behavior, good_queries=10, delay_s=.08)
    wrench_production_process__until(p, lambda h: p.last_good is not None)
    p.arm_runtime_grace()
    wrench_production_process__until(p, lambda h: reason in h["FAILURE_LATCH"])
    assert not p.health()["STREAM_HEALTH"]


@pytest.mark.parametrize("elapsed_ns,expired", [(349_000_000, False), (351_000_000, True)])
def test_runtime_wrench_completed_between_polls_cannot_erase_gap(tmp_path, elapsed_ns, expired):
    p = wrench_deadline_regressions__provider(tmp_path)
    p.budgets = wrench_production_process__replace(p.budgets, sample_age_s=.05)
    old_ns = 1_100_000_000
    end = old_ns + elapsed_ns
    start = end - 1_000_000
    p.last_good = wrench_deadline_regressions__RobotWrenchFrame(0, 1.1, 1.1, 1.1,
        1.1, "", "OFFLINE_HOST_QUERY", True, "", "world",
        (1., 2., 3.), (.1, .2, .3), (1.,) * 6, (.5,) * 6)
    p.runtime_grace_armed = True
    p.query_state = "IDLE"
    frame = wrench_production_process__replace(p.last_good, sequence_id=1,
        host_monotonic_time_s=end / 1e9)
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
        event_id=1, query_id=1, type="QUERY_SUCCESS", timestamp_ns=end,
        intent_ns=start, native_start_ns=start, completion_ns=end,
        frame=wrench_deadline_regressions__asdict(frame), error=None)
    p._accept(event, end)
    assert ("wrench_sample_stale_or_clock_invalid" in p.failure_latch) is expired


def test_runtime_grace_is_armed_after_final_gate_before_hold(personal_context):
    c = personal_context
    c.acquisition.arm_runtime_grace = lambda: c.events.append("arm")
    executor, _ = personal_executor(c)
    executor.execute(c.trajectory, evaluate_offline_execution_request(**c.request))
    assert c.events.index("arm") + 1 == c.events.index("hold")


def test_personal_executor_stops_on_late_logging(personal_context):
    c = personal_context
    executor, clock = personal_executor(c)
    c.logger.append_trajectory_command = lambda **kw: setattr(clock, "value", clock.value + 1.)
    with pytest.raises(RuntimeError, match="deadline"):
        executor.execute(c.trajectory, evaluate_offline_execution_request(**c.request))
    assert "hold" not in c.events and c.events[-1] == "stop"


def test_personal_executor_stops_on_external_leg_fault(personal_context):
    c = personal_context
    executor, _ = personal_executor(c)
    executor.execution_guard = lambda: "ADS fault"
    with pytest.raises(RuntimeError, match="ADS fault"):
        executor.execute(c.trajectory, evaluate_offline_execution_request(**c.request))
    assert "hold" not in c.events and c.events[-1] == "stop"


def test_personal_run_requires_live_before_io(tmp_path):
    with pytest.raises(PermissionError, match="--live"):
        personal_round.run(tmp_path)


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("collision_source", ["events", "controller"])
def test_personal_round_capture_and_cleanup(tmp_path, monkeypatch, personal_context, fails, collision_source):
    c = personal_context
    config = personal_round.configuration()
    config["acquisition"].update(pre_roll_s=.0001, post_roll_s=.0001, max_leg_start_age_s=1., max_start_angle_error_deg=1.)
    config["training"]["enabled"] = False
    config["robot"].update(collision_source=collision_source,
        collision_event_max_age_s=.2 if collision_source == "events" else None)
    state = dict(status="PREPARED", config_path=str(BUNDLE_PATH))
    personal_round.report(tmp_path, state)
    c.anchor.save_json(tmp_path / "anchor.json")
    monkeypatch.setattr(personal_round, "configuration", lambda p: config)
    monkeypatch.setattr(personal_round, "check_config", lambda p: [])
    monkeypatch.setattr(personal_round, "inspect_file", lambda p, **kwargs: {"rows": 3})
    stopped = []
    class Capture:
        def __init__(self, *args): pass
        def start(self): pass
        def stop(self): stopped.append(True)
        def health_reason(self): return None
        def snapshot(self):
            import time
            latest = {}
            for name, axis in (("hip", 1), ("knee", 4)):
                values = config["calibration"][name]
                q = getattr(c.anchor, "reference_start_q_" + name)
                latest[f"axis{axis}_position_raw"] = values["zero_counts"] + np.rad2deg(q) * values["counts_per_degree"] / values["flexion_sign"]
            return {"last_sample_ns": time.perf_counter_ns(), "latest": latest}
    def execute(**kwargs):
        kwargs["dispatch_observer"](0., 1, 2)
        if fails:
            raise RuntimeError("synthetic motion fault")
        return {"completed": True, "same_session_realtime_cleanup_completed": True}
    if fails:
        with pytest.raises(RuntimeError, match="synthetic motion fault"):
            personal_round.run(tmp_path, live=True, capture_factory=Capture, execute_fn=execute)
    else:
        assert personal_round.run(tmp_path, live=True, capture_factory=Capture, execute_fn=execute)["status"] == "CAPTURE_COMPLETE"
    assert stopped == [True]
    assert (tmp_path / "dispatch.csv").is_file()
    saved = json.loads((tmp_path / "round_status.json").read_text(encoding="utf8"))
    assert saved["status"] == ("FAILED" if fails else "CAPTURE_COMPLETE")


def test_personal_algorithm_runs_without_hash_manifest(tmp_path):
    from scripts.train_robot_leg_round import train
    from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
    frame = pd.read_csv(DEFAULT_REFERENCE_PATH)
    frame["tau_hip_nm"] = 2. + np.sin(frame.q_hip_rad)
    frame["tau_knee_nm"] = 2. + np.sin(frame.q_knee_rad)
    measured = tmp_path / "synthetic.csv"
    frame.to_csv(measured, index=False)
    reference_id = str(frame.trajectory_id.iloc[0])
    request = dict(candidates=[dict(path=str(DEFAULT_REFERENCE_PATH), candidate_id=reference_id,
        beta_flex=0., beta_extend=0., share_shift=0.)], reference_candidate_id=reference_id,
        history=[dict(episode_id="SYNTHETIC", candidate_id=reference_id, measured_path=str(measured),
                      endpoints=dict(E3=1., E2=1., peak_ratio=1.))])
    result = train(request)
    assert result["status"] == "MODEL_UPDATED"
    assert result["action"]["stop_reason"] == "DOMAIN_EXHAUSTED"


def test_personal_archive_completed_session(tmp_path, monkeypatch):
    import zipfile
    from utils import assets
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w"): pass
    monkeypatch.setattr(assets, "ARCHIVE", archive)
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/experiment_summary.csv").write_text("source,metric,value\n")
    session = tmp_path / "round_test"
    session.mkdir()
    (session / "round_status.json").write_text(json.dumps(dict(status="CAPTURE_COMPLETE", metrics={"rows": 3})))
    (session / "raw.csv").write_text("time,value\n0,1\n")
    assets.archive_session(session)
    with zipfile.ZipFile(archive) as z:
        assert z.read("sessions/round_test/raw.csv") == (session / "raw.csv").read_bytes()
    with pytest.raises(FileExistsError):
        assets.archive_session(session)


@pytest.mark.parametrize("fails", [False, True])
def test_personal_execute_runner_cleanup(tmp_path, monkeypatch, personal_context, fails):
    from scripts import run_rehab_experiment as runner
    c = personal_context
    anchor_path = tmp_path / "anchor.json"
    c.anchor.save_json(anchor_path)
    safety_path = tmp_path / "limits.json"
    safety_path.write_text(json.dumps(c.safety.to_dict()))
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(json.dumps({"rehab_x_axis_in_base": [1, 0, 0], "rehab_z_axis_in_base": [0, 0, 1]}))
    cleanup = []
    c.adapter.read_tcp_pose = lambda: c.anchor.tcp_pose_base
    c.adapter.disconnect = lambda: cleanup.append("disconnect")
    class Acquisition:
        def __init__(self, adapter, logger, **kwargs):
            self.adapter = adapter
            self.live_producer_names = []
        def start(self, **kwargs): pass
        def stop(self): cleanup.append("acquisition_stop")
        def latest_health(self): return c.health
        def wait_until_healthy(self, **kwargs): return True
        def latest_state_frame(self): return c.acquisition.latest_state_frame()
    monkeypatch.setattr(runner, "RealRobotAcquisition", Acquisition)
    monkeypatch.setattr(runner, "build_live_wrench_provider", lambda *a, **kw: object())
    monkeypatch.setattr(runner, "_prepare_realtime_session", lambda *a, **kw: None)
    def executor_factory(motion, acquisition, logger, safety, **kwargs):
        clock = SimpleNamespace(value=1.)
        clock.now_s = lambda: clock.value
        executor = RokaeMotionExecutor(motion, acquisition, logger, safety, clock=clock, **kwargs)
        executor._wait_until = lambda deadline: setattr(clock, "value", deadline)
        return executor
    monkeypatch.setattr(runner, "RokaeMotionExecutor", executor_factory)
    if fails:
        def fail(pose): raise RuntimeError("synthetic target failure")
        c.motion.send_cartesian_target = fail
    kwargs = dict(robot_ip="192.0.2.1", local_ip="192.0.2.2", episode_dir=tmp_path / "robot",
        anchor_path=anchor_path, requested_anchor_id=c.anchor.anchor_id, frame_config_path=frame_path,
        safety_config_path=safety_path, trajectory_id=c.audit.trajectory_id, enable_motion=True,
        operator_confirmation=OPERATOR_CONFIRMATION, adapter_factory=lambda ip: c.adapter,
        motion_factory=lambda adapter: c.motion, prepare_realtime=True, wrench_hz=50,
        live_wrench_config={"connection": {"robot_ip": "192.0.2.1", "local_ip": "192.0.2.2"},
                            "budgets": {"startup_s": 1}, "allowed_operation_states": ["rtcontrolling"]})
    if fails:
        with pytest.raises(RuntimeError, match="synthetic target failure"):
            runner.run_execute(**kwargs)
    else:
        assert runner.run_execute(**kwargs)["completed"]
    assert cleanup == ["acquisition_stop", "disconnect"]
    assert c.events[-1] == "stop"
    metadata = json.loads((tmp_path / "robot/metadata.json").read_text(encoding="utf8"))
    assert metadata["logger"]["status"] == ("failed" if fails else "completed")


def test_personal_ads_compact_configuration_decodes():
    import struct
    from config.bundle import load_section
    from collection.twincat_leg import decode_image, validate_config
    config = validate_config(load_section("twincat_leg_capture"))
    image = bytearray(config["image_bytes"])
    for offset in config["axis_offsets"]:
        struct.pack_into("<HiihiHbf", image, offset, 0, 123, 0, 0, 0, 0, 8, 2.5)
    decoded = decode_image(image, config)
    assert decoded["axis1_position_raw"] == 123
    assert decoded["axis4_torque_3c1a_nm"] == 2.5


@pytest.mark.parametrize("value", [float("nan"), -1., None])
def test_personal_runtime_invalid_force_stops(personal_context, value):
    c = personal_context
    c.health.force_magnitude_n = value
    executor, _ = personal_executor(c)
    assert executor._runtime_reasons()


def test_unified_config_readers_need_no_generated_sections(tmp_path, monkeypatch):
    from config import bundle
    from control.start_anchored_relative_trajectory import load_rehab_frame_config
    from collection.twincat_leg import read_json
    from measurement_validation.leg_episode import _calibration_from_json
    from scripts.acquire_robot_data import load_live_wrench_config
    config = bundle.configuration()
    config["robot"]["ip"] = "192.0.2.33"
    path = tmp_path / "custom_experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    def no_cache(*args, **kwargs):
        raise AssertionError("configuration must not generate section files")
    monkeypatch.setattr(bundle, "section_path", no_cache)
    assert load_rehab_frame_config(path).rehab_x_axis_in_base == (1., 0., 0.)
    assert ExperimentSafetyConfig.load_json(path).max_force_n == config["motion"]["max_force_n"]
    assert read_json(path)["image_bytes"] == 1366
    assert Path(read_json(path)["ads_dll_path"]).is_absolute()
    assert load_live_wrench_config(path)["connection"]["robot_ip"] == "192.0.2.33"
    assert set(_calibration_from_json(path)) == {"hip", "knee"}
    assert personal_round.check_config(path)  # Actual missing physical parameters remain visible.
    assert list(tmp_path.iterdir()) == [path]


def test_training_failure_updates_same_round_report(tmp_path, monkeypatch):
    status = dict(status="CAPTURE_COMPLETE", config_path=str(personal_round.BUNDLE_PATH))
    (tmp_path / "round_status.json").write_text(json.dumps(status), encoding="utf-8")
    def invalid_alignment(*args, **kwargs):
        raise ValueError("recording has a time gap")
    monkeypatch.setattr(personal_round, "aligned_measurement", invalid_alignment)
    with pytest.raises(ValueError, match="time gap"):
        personal_round.analyze(tmp_path)
    state = json.loads((tmp_path / "round_status.json").read_text(encoding="utf-8"))
    assert state["status"] == "CAPTURE_COMPLETE"
    assert state["training_status"] == "FAILED"
    assert "time gap" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["state", "wrench", "alignment"])
def test_unified_diagnostic_cli_uses_one_report_pair(tmp_path, monkeypatch, kind):
    from contextlib import contextmanager
    from scripts import diagnose_robot as diagnostic
    @contextmanager
    def connected(*args, **kwargs):
        yield object(), object()
    monkeypatch.setattr(diagnostic, "make_robot", lambda *args: object())
    monkeypatch.setattr(diagnostic, "readonly_connection", connected)
    measure_name = {"state": "measure_state", "wrench": "measure_wrench", "alignment": "measure_alignment"}[kind]
    monkeypatch.setattr(diagnostic, measure_name,
        lambda *args, **kwargs: ([dict(sample_index=0, valid=True)], dict(sample_count=1, valid_count=1)))
    assert diagnostic.main([kind, "--output-dir", str(tmp_path)]) == 0
    files = list(tmp_path.iterdir())
    assert len(files) == 2
    result = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert "rows" not in result
    assert (tmp_path / result["samples_csv"]).is_file()


@pytest.mark.parametrize("arguments", [
    ["state", "--duration", "nan"], ["wrench", "--hz", "inf"],
    ["alignment", "--software-bias"], ["state", "--software-bias", "--confirm-unloaded"],
])
def test_unified_diagnostic_rejects_invalid_request_before_connect(monkeypatch, arguments):
    from scripts import diagnose_robot as diagnostic
    def no_connect(*args):
        raise AssertionError("invalid request must not connect")
    monkeypatch.setattr(diagnostic, "make_robot", no_connect)
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(arguments)
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", [None, "robot", "leg", "cleanup"])
def test_observe_records_without_motion_parameters_and_always_cleans_up(tmp_path, monkeypatch, failure):
    config = personal_round.configuration()
    dll = tmp_path / "TcAdsDll.dll"
    dll.write_bytes(b"offline double")
    config["leg"]["ads_dll_path"] = str(dll)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []
    class Capture:
        def __init__(self, source, directory):
            self.directory = directory
        def start(self):
            calls.append("start")
            self.directory.mkdir()
            (self.directory / "leg.csv").write_text("raw data", encoding="ascii")
        def health_reason(self):
            return "leg stalled" if failure == "leg" else None
        def stop(self):
            calls.append("stop")
            if failure == "cleanup":
                raise RuntimeError("cleanup failed")
    def acquire(**kwargs):
        calls.append("acquire")
        assert kwargs["mode"] == "live"
        assert kwargs["observation_guard"] is not None
        if failure == "robot":
            raise RuntimeError("robot query stalled")
        reason = kwargs["observation_guard"]()
        if reason:
            raise RuntimeError(reason)
        return dict(completed=True, motion_commanded=False)
    monkeypatch.setattr(personal_round, "inspect_file", lambda *args, **kwargs: dict(rows=42))
    directory = tmp_path / "observation"
    if failure:
        with pytest.raises(RuntimeError):
            personal_round.observe(config_path, directory, capture_factory=Capture, acquire_fn=acquire)
    else:
        result = personal_round.observe(config_path, directory, capture_factory=Capture, acquire_fn=acquire)
        assert result["status"] == "OBSERVATION_COMPLETE"
    assert calls == ["start", "acquire", "stop"]
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    assert state["motion_commanded"] is False
    assert state["status"] == ("FAILED" if failure else "OBSERVATION_COMPLETE")
    assert not (directory / "anchor.json").exists()
    assert not (directory / "dispatch.csv").exists()
    with pytest.raises(ValueError, match="completed recording"):
        personal_round.analyze(directory)


def test_observation_guard_stops_robot_acquisition_and_releases_resources(tmp_path):
    from scripts.acquire_robot_data import run_acquisition
    adapter = real_robot_acquisition__FakeAdapter()
    adapter.get_robot_state_summary = lambda: dict(connected=True)
    with pytest.raises(RuntimeError, match="leg stopped"):
        run_acquisition(robot_ip="192.0.2.1", episode_dir=tmp_path / "robot",
            duration_s=.1, adapter_factory=lambda ip: adapter,
            observation_guard=lambda: "leg stopped")
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]


@pytest.mark.parametrize("duration", [0., -1., float("nan"), float("inf")])
def test_observe_invalid_duration_does_not_create_session(tmp_path, duration):
    with pytest.raises(ValueError, match="duration_s"):
        personal_round.observe(personal_round.BUNDLE_PATH, tmp_path / "unused", duration_s=duration)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("source,valid,state,allowed", [
    ("controller", False, None, True), ("controller", False, True, False),
    ("query", False, None, False), ("events", False, None, False),
    ("query", True, False, True), ("events", True, True, False),
])
def test_collision_mode_keeps_unknown_explicit_and_never_ignores_collision(source, valid, state, allowed):
    from hardware.safety_events import collision_feedback_allows_motion
    assert collision_feedback_allows_motion(dict(collision_state_source=source,
        collision_state_query_valid=valid, collision_state=state)) is allowed


def test_controller_mode_summary_never_queries_or_invents_collision_clearance():
    from hardware.safety_events import collision_feedback_allows_motion
    native = rokae_adapter__FakeReadOnlyNative()
    def forbidden():
        raise AssertionError("controller mode must not call queryEventInfo")
    native.get_collision_state = forbidden
    adapter = rokae_adapter__RokaeRobotAdapter(native_robot=native, collision_source="controller")
    summary = adapter.get_robot_state_summary()
    assert summary["collision_state_source"] == "controller"
    assert summary["collision_state"] is None
    assert summary["collision_state_query_valid"] is False
    assert collision_feedback_allows_motion(summary)
    adapter._collision_latched = True
    assert not collision_feedback_allows_motion(adapter.get_robot_state_summary())


def test_controller_motion_fault_invalidates_stream_and_stays_latched(monkeypatch):
    import sys
    from types import SimpleNamespace
    from hardware.windows.rokae_xcore import RokaeRobot
    native = RokaeRobot("192.0.2.1", collision_source="controller")
    native.is_connected = True
    native._rt_active = True
    native._state_running = True
    class Vector:
        def content(self): return [0.] * 6
    native._sdk = SimpleNamespace(PyTypeVectorDouble=Vector, PyTypeVectorBool=Vector)
    updates = iter((True, False))
    native._robot = SimpleNamespace(updateRobotState=lambda timeout: next(updates),
        getStateData=lambda *args: None)
    native.realtime_motion_error = lambda: True
    monkeypatch.setitem(sys.modules, "xCoreSDK_python", SimpleNamespace(
        RtSupportedFields=SimpleNamespace(tcpPoseAbc_m=1, jointPos_m=2, keypads=3)))
    import hardware.windows.rokae_xcore as module
    monkeypatch.setattr(module.time, "sleep", lambda seconds: setattr(native, "_state_running", False))
    native._state_loop()
    assert native._controller_motion_fault
    assert "controller_realtime_motion_error" in native.get_state_frame().invalid_reason
    assert not native.get_state_frame().valid


def test_slow_controller_fault_poll_does_not_refresh_old_state_timestamp(monkeypatch):
    import sys
    from types import SimpleNamespace
    import hardware.windows.rokae_xcore as module
    native = module.RokaeRobot("192.0.2.1", collision_source="controller")
    native.is_connected = True
    native._rt_active = True
    native._state_running = True
    class Vector:
        def content(self): return [0.] * 6
    native._sdk = SimpleNamespace(PyTypeVectorDouble=Vector, PyTypeVectorBool=Vector)
    updates = iter((True, False))
    native._robot = SimpleNamespace(updateRobotState=lambda timeout: next(updates),
        getStateData=lambda *args: None)
    clock = [1_000_000_000]
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: clock[0])
    def poll():
        clock[0] += 300_000_000
        return False
    native.realtime_motion_error = poll
    monkeypatch.setitem(sys.modules, "xCoreSDK_python", SimpleNamespace(
        RtSupportedFields=SimpleNamespace(tcpPoseAbc_m=1, jointPos_m=2, keypads=3)))
    accept = native._accept_state
    def accept_once(*args):
        accept(*args)
        native._state_running = False
    native._accept_state = accept_once
    native._state_loop()
    assert native.get_state_frame().host_monotonic_time_s == 1.
    assert clock[0] == 1_300_000_000


@pytest.mark.parametrize("source", ["controller", "query", "events"])
def test_realtime_preparation_unknown_feedback_only_allows_controller_mode(source):
    from scripts.run_rehab_experiment import _prepare_realtime_session
    calls = []
    native = SimpleNamespace(get_robot_mode=lambda: "IDLE",
        enable_realtime=lambda **kw: calls.append(kw))
    adapter = SimpleNamespace(connect=lambda: None, native_robot=native,
        get_robot_state_summary=lambda: dict(collision_state_source=source,
            collision_state_query_valid=False, collision_state=None))
    if source == "controller":
        _prepare_realtime_session(adapter, filter_hz=50., network_tolerance_percent=20.)
        assert calls == [{"load": 0.}]
    else:
        with pytest.raises(PermissionError, match="collision state"):
            _prepare_realtime_session(adapter, filter_hz=50., network_tolerance_percent=20.)
        assert not calls


@pytest.mark.parametrize("states", [[False, False], [False, True, True], [False, True, False]])
def test_recording_report_retains_startup_and_failed_samples(tmp_path, states):
    import csv
    from scripts.diagnose_robot import summarize_recording
    directory = tmp_path / "robot"
    directory.mkdir()
    path = directory / "aligned_snapshot.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["host_time_s", "query_duration_ms", "state_age_s",
            "wrench_age_s", "state_wrench_skew_s", "valid", "invalid_reason"])
        writer.writeheader()
        for i, valid in enumerate(states):
            writer.writerow(dict(host_time_s=1.+i*.02, query_duration_ms=312. if i == 0 else 1.,
                state_age_s=.36 if i == 0 else .01, wrench_age_s=.06, state_wrench_skew_s=.03,
                valid=valid, invalid_reason="" if valid else "state_stale;wrench_stale"))
    result = summarize_recording(tmp_path)
    assert result["samples"] == len(states)
    assert result["valid_samples"] == sum(states)
    assert result["ever_ready"] == any(states)
    assert result["invalid_after_ready"] == (None if not any(states) else states[states.index(True):].count(False))
    assert result["metrics"]["query_duration_ms"]["max"] == 312.
    assert result["metrics"]["state_age_ms"]["max"] == 360.
    assert result["invalid_reasons"]["state_stale"] == states.count(False)
    state = dict(status="FAILED", reason="original acquisition failure")
    personal_round.report(tmp_path, state)
    saved = json.loads((tmp_path / "round_status.json").read_text(encoding="utf-8"))
    assert saved["reason"] == "original acquisition failure"
    assert saved["robot_timing"] == result
    assert "312.000" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("content", [None, "not,a,snapshot\n", "host_time_s,valid,invalid_reason,query_duration_ms,state_age_s,wrench_age_s,state_wrench_skew_s\nnan,True,,1,.01,.02,.01\n"])
def test_unreadable_timing_never_masks_session_failure(tmp_path, content):
    from scripts.diagnose_robot import summarize_recording
    if content is not None:
        (tmp_path / "robot").mkdir()
        (tmp_path / "robot/aligned_snapshot.csv").write_text(content, encoding="utf-8")
    assert not summarize_recording(tmp_path)["available"]
    state = dict(status="FAILED", reason="original failure")
    personal_round.report(tmp_path, state)
    assert json.loads((tmp_path / "round_status.json").read_text(encoding="utf-8"))["reason"] == "original failure"


def test_current_origin_translates_path_and_preserves_captured_orientation(personal_context):
    c = personal_context
    pose = (.6, .1, .7, .1, -.2, .3)
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        current_tcp_start_pose=pose, rehab_frame=c.frame)
    columns = ["tcp_x_base", "tcp_y_base", "tcp_z_base"]
    assert audit.trajectory_valid
    assert np.allclose(trajectory[columns].to_numpy() - c.trajectory[columns].to_numpy(),
                       np.array(pose[:3]) - c.anchor.tcp_position_base_m)
    assert np.allclose(trajectory[columns].iloc[0], pose[:3])
    assert np.allclose(trajectory[columns].iloc[-1], pose[:3])
    orientations = ["tcp_rx", "tcp_ry", "tcp_rz"]
    assert np.allclose(trajectory[orientations].to_numpy(), pose[3:])
    assert np.allclose(trajectory[["q_hip_ref", "q_knee_ref"]], c.trajectory[["q_hip_ref", "q_knee_ref"]])


def test_from_current_captures_fresh_pose_before_running(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(personal_round, "check_config", lambda config: [])
    monkeypatch.setattr(personal_round, "prepare", lambda config, directory: calls.append(("prepare", config, directory)))
    def run(directory, *, live):
        calls.append(("run", directory, live))
        return dict(status="CAPTURE_COMPLETE")
    monkeypatch.setattr(personal_round, "run", run)
    config, directory = tmp_path / "config.json", tmp_path / "new_session"
    assert personal_round.run_from_current(config, directory, live=True)["status"] == "CAPTURE_COMPLETE"
    assert calls == [("prepare", config, directory), ("run", directory, True)]


@pytest.mark.parametrize("live", [False, True])
def test_from_current_checks_confirmation_and_config_before_device_io(tmp_path, monkeypatch, live):
    calls = []
    monkeypatch.setattr(personal_round, "check_config", lambda config: ["missing physical limits"])
    monkeypatch.setattr(personal_round, "prepare", lambda *args: calls.append("prepare"))
    monkeypatch.setattr(personal_round, "run", lambda *args, **kwargs: calls.append("run"))
    with pytest.raises(ValueError if live else PermissionError):
        personal_round.run_from_current(BUNDLE_PATH, tmp_path / "new_session", live=live)
    assert not calls and not (tmp_path / "new_session").exists()


def test_from_current_does_not_reuse_existing_session(tmp_path, monkeypatch):
    monkeypatch.setattr(personal_round, "check_config", lambda config: [])
    saved = tmp_path / "raw.csv"
    saved.write_text("original raw data", encoding="ascii")
    with pytest.raises(FileExistsError):
        personal_round.run_from_current(BUNDLE_PATH, tmp_path, live=True)
    assert saved.read_text(encoding="ascii") == "original raw data"


def test_from_current_cli_routes_config_and_explicit_live(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(personal_round, "run_from_current", lambda config, directory, **kwargs:
        calls.append((config, directory, kwargs)) or dict(status="CAPTURE_COMPLETE"))
    config, session = tmp_path / "config.json", tmp_path / "new_session"
    assert personal_round.main(["run", "--config", str(config), "--session", str(session),
        "--from-current", "--live"]) == 0
    assert calls == [(config, session, dict(live=True))]
    with pytest.raises(SystemExit):
        personal_round.main(["observe", "--session", str(session), "--from-current"])


def test_command_logging_flushes_before_dispatch_and_syncs_only_on_close(tmp_path):
    from collection.episode_logger import EpisodeLogger
    from unittest.mock import patch, call
    logger = EpisodeLogger(tmp_path / "episode").start()
    command_fd = logger._handles["trajectory_command"].fileno()
    with patch("collection.episode_logger.os.fsync") as sync:
        logger.append_trajectory_command_bounded(timeout_s=.5, host_time_s=1.,
            command_valid=True, invalid_reason="")
        sync.assert_not_called()
        with (tmp_path / "episode/trajectory_command.csv").open(encoding="utf-8") as stream:
            assert len(stream.readlines()) == 2
        logger.close(completed=True)
        assert call(command_fd) in sync.call_args_list
