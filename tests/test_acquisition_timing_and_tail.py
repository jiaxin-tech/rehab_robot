"""Offline regression coverage for interrupted waits and shutdown handoff.

Unit doubles control the shutdown phases. The integration test uses a real
spawned WrenchProcessProvider and holds a success event before durable ACK.
No native SDK, robot connection, or motion is used.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from collection import real_robot_acquisition as acquisition_module
from collection.episode_logger import EpisodeLogger, EpisodeLoggerError
from collection.real_robot_acquisition import RealRobotAcquisition
from collection.state import KinematicStateFrame
from collection.wrench_process import DurableAudit, WrenchBudgets, WrenchProcessProvider
from hardware.rokae_adapter import RobotWrenchFrame


def frame(sequence):
    stamp = time.perf_counter() + sequence * 1e-6
    return RobotWrenchFrame(
        sequence, stamp, stamp, stamp, stamp, "", "OFFLINE_TEST", True, "", "world",
        (float(sequence), 2., 3.), (.1, .2, .3), (1.,) * 6, (.5,) * 6,
    )


class OfflineAdapter:
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
        stamp = time.perf_counter()
        return KinematicStateFrame(
            sequence_id=self.sequence, host_monotonic_time_s=stamp,
            wall_time_iso=None, robot_device_time_s=None, valid=True,
            invalid_reason="", tcp_position_m=(0., 0., 0.),
            tcp_orientation_rad=(0., 0., 0.), tcp_linear_velocity_mps=None,
            tcp_angular_velocity_radps=None, velocity_source="unavailable",
            joint_position_rad=(0.,) * 6, joint_velocity_radps=None,
            pose_time_s=stamp, joint_time_s=stamp, velocity_time_s=None,
            operation_state="IDLE", collision_state=None, controller_error=None,
        )


class RecordingLogger:
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
            raise EpisodeLoggerError("injected_tail_write_failure")
        self.rows.append(row)

    def signal_failure(self, reason, **_kwargs):
        self.failures.append(reason)

    def mark_failed(self, reason):
        self.failures.append(reason)


class ShutdownProvider:
    """A cached result before shutdown and one durable result during shutdown."""
    def __init__(self, before, after, *, stop_error=None):
        self.config = {"mode": "offline", "rate_hz": 50.}
        self.budgets = SimpleNamespace(state_age_s=3., sample_age_s=3., skew_s=3.)
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


class ControlledThread:
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


def controlled_acquisition(before, after, *, fail_append=False, stuck_name=None, stop_error=None):
    adapter = OfflineAdapter()
    logger = RecordingLogger(fail_append=fail_append)
    provider = ShutdownProvider(before, after, stop_error=stop_error)
    threads = {}
    def make_thread(**kwargs):
        item = ControlledThread(stuck=kwargs["name"] == stuck_name, **kwargs)
        threads[kwargs["name"]] = item
        return item
    acquisition = RealRobotAcquisition(adapter, logger, wrench_provider=provider, thread_factory=make_thread)
    acquisition.start()
    return acquisition, adapter, logger, provider, threads


@pytest.mark.parametrize("before_id,after_id,expected", [
    (1, 2, [1., 2.]),
    (1, 1, [1.]),
    (None, 1, [1.]),
    (None, None, []),
])
def test_shutdown_keeps_snapshot_and_final_success_once(before_id, after_id, expected):
    before = frame(before_id) if before_id is not None else None
    # Use the same immutable frame for the duplicate case, as the real provider does.
    after = before if before_id == after_id else frame(after_id) if after_id is not None else None
    acquisition, adapter, logger, provider, threads = controlled_acquisition(before, after)
    acquisition.stop()
    assert [row["fx"] for row in logger.rows] == expected
    assert logger.append_attempts == len(expected)
    assert all(not item.is_alive() and item.join_calls == 1 for item in threads.values())
    assert provider.cleanup["NORMAL_CLEANUP"]
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]


def test_repeated_stop_does_not_duplicate_tail_or_touch_closed_logger():
    acquisition, _, logger, _, _ = controlled_acquisition(frame(1), frame(2))
    acquisition.stop()
    assert [row["fx"] for row in logger.rows] == [1., 2.]
    logger.closed = True
    acquisition.stop()
    assert [row["fx"] for row in logger.rows] == [1., 2.]
    assert logger.append_attempts == 2


def test_tail_write_failure_still_stops_provider_and_cleans_adapter_then_raises():
    acquisition, adapter, logger, provider, _ = controlled_acquisition(frame(1), frame(2), fail_append=True)
    with pytest.raises(Exception, match="injected_tail_write_failure"):
        acquisition.stop()
    assert provider.stop_calls == 1
    assert provider.cleanup["NORMAL_CLEANUP"]
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]
    assert not adapter.connected
    assert logger.rows == []
    assert logger.append_attempts == 1, "do not try a second write after an uncertain first write failed"
    assert logger.failures


def test_provider_stop_failure_does_not_append_tail_or_change_existing_adapter_error_path():
    acquisition, adapter, logger, provider, _ = controlled_acquisition(
        frame(1), frame(2), stop_error=RuntimeError("injected_provider_stop_failure"),
    )
    with pytest.raises(RuntimeError, match="injected_provider_stop_failure"):
        acquisition.stop()
    assert provider.stop_calls == 1
    assert logger.rows == []
    assert logger.append_attempts == 0
    assert adapter.calls == ["connect", "start_state_stream"]
    assert logger.failures


def test_join_failure_never_appends_tail_or_disconnects_parent():
    acquisition, adapter, logger, provider, _ = controlled_acquisition(
        frame(1), frame(2), stuck_name="real-episode-state",
    )
    with pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        acquisition.stop()
    assert provider.stop_calls == 1  # isolated child still receives its existing cleanup.
    assert logger.append_attempts == 0
    assert adapter.calls == ["connect", "start_state_stream"]
    assert logger.failures


class FakeClock:
    def __init__(self):
        self.value = 10.

    def now_s(self):
        return self.value

    def now_ns(self):
        return int(self.value * 1e9)


class NoBlockingEvent:
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


@pytest.mark.parametrize("interrupt", [False, True])
def test_deadline_wait_uses_short_sleep_slices_and_observes_stop(monkeypatch, interrupt):
    clock = FakeClock()
    acquisition = RealRobotAcquisition(OfflineAdapter(), RecordingLogger(), diagnostic_state_only=True, clock=clock)
    event = NoBlockingEvent()
    acquisition._stop_event = event
    slices = []
    def sleep(seconds):
        slices.append(seconds)
        assert 0 < seconds <= .002
        clock.value += seconds
        if interrupt:
            event.set()
    monkeypatch.setattr(acquisition_module, "time", SimpleNamespace(sleep=sleep))
    deadline = clock.now_s() + .009
    returned = acquisition._sleep_until(deadline)
    assert event.wait_calls == 0
    assert slices
    if interrupt:
        assert len(slices) == 1
        assert returned < deadline
    else:
        assert returned == pytest.approx(deadline)
        assert len(slices) >= 5


def test_wrench_no_new_frame_still_uses_deadline_scheduler():
    clock = FakeClock()
    provider = ShutdownProvider(None, None)
    acquisition = RealRobotAcquisition(OfflineAdapter(), RecordingLogger(), wrench_provider=provider, clock=clock)
    event = NoBlockingEvent()
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


def wait_for(predicate, timeout=5.):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(.001)
    raise AssertionError("offline test condition was not observed before its deadline")


def test_real_provider_preserves_success_whose_durable_ack_finishes_during_stop(tmp_path):
    success_waiting = threading.Event()
    release_write = threading.Event()
    real_stop_observed = threading.Event()
    release_errors = []

    def hold_second_success(event):
        if event.get("type") == "QUERY_SUCCESS" and event.get("query_id") == 2:
            success_waiting.set()
            if not release_write.wait(5.):
                raise TimeoutError("test did not release second success")

    budgets = WrenchBudgets(startup_s=5., query_s=3., progress_s=4., sample_age_s=4.,
        ack_s=5., log_s=4., shutdown_s=1.5, terminate_s=.5, state_age_s=4., skew_s=4.,
        provenance="NOT_SAFETY_THRESHOLDS")
    audit_path = tmp_path / "query_events.jsonl"
    provider = WrenchProcessProvider(dict(mode="offline", rate_hz=50., behavior="normal"), budgets,
        audit_path, audit_factory=lambda path: DurableAudit(path, write_hook=hold_second_success))
    logger = EpisodeLogger(tmp_path / "episode").start()
    adapter = OfflineAdapter()
    acquisition = RealRobotAcquisition(adapter, logger, wrench_provider=provider)
    released = None

    def release_only_after_actual_provider_stop():
        try:
            wait_for(lambda: provider.stop_flag.value == 1, timeout=2.)
            real_stop_observed.set()
        except Exception as exc:
            release_errors.append(str(exc))
        finally:
            release_write.set()

    try:
        acquisition.start()
        assert success_waiting.wait(5.), "second success did not reach the audit writer"
        wait_for(lambda: logger.row_counts["robot_wrench"] == 1)
        assert provider.last_good.sequence_id == 1
        assert not release_write.is_set()
        released = threading.Thread(target=release_only_after_actual_provider_stop, daemon=True)
        released.start()
        acquisition.stop()
        released.join(2.)
        assert not released.is_alive()
        assert not release_errors
        assert real_stop_observed.is_set()
        assert provider.cleanup["NORMAL_CLEANUP"], provider.cleanup
        assert provider.cleanup["process_exited"]
        assert not provider.cleanup["FORCED_WORKER_TERMINATION"]
        assert provider.last_good.sequence_id == 2
        assert logger.row_counts["robot_wrench"] == 2
        logger.close(completed=False, stop_reason="OFFLINE_REGRESSION_TEST")
        with (tmp_path / "episode/robot_wrench.csv").open(newline="", encoding="utf-8") as stream:
            recorded = list(csv.DictReader(stream))
        events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        successful = [row for row in events if row["type"] == "QUERY_SUCCESS"]
        assert [row["query_id"] for row in successful] == [1, 2]
        assert [float(row["query_start_s"]) for row in recorded] == [row["frame"]["host_query_start_s"] for row in successful]
        # A second stop after logger closure must not append the final row again.
        acquisition.stop()
        assert len(recorded) == 2
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
