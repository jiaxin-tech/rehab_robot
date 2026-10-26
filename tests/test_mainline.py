"""Offline prosthetic-leg mainline regression suite.
Retained source sections preserve original comments, assertions and failure cases.
Integrity regressions cover archive and unified-configuration boundaries.
Section prefixes isolate imports, helpers and fixtures. Former module autouse
fixtures apply explicitly to their original tests. Hardware is simulated.
"""
from __future__ import annotations

from config.bundle import export_sections, section_path
from utils.evidence import artifact_path


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
# Source: tests/test_capture_and_acquire_cli.py
# ============================================================================

"""No-motion regressions for capture and acquire entry points."""



import json as capture_and_acquire_cli__json
import threading as capture_and_acquire_cli__threading
import time as capture_and_acquire_cli__time
from types import SimpleNamespace as capture_and_acquire_cli__SimpleNamespace

import pytest as capture_and_acquire_cli__pytest

from collection.state import KinematicStateFrame as capture_and_acquire_cli__KinematicStateFrame
from collection.wrench_process import WrenchBudgets as capture_and_acquire_cli__WrenchBudgets
from hardware.rokae_adapter import RobotWrenchFrame as capture_and_acquire_cli__RobotWrenchFrame
import scripts.acquire_robot_data as capture_and_acquire_cli__acquire_cli
from scripts.acquire_robot_data import (
    load_live_wrench_config as capture_and_acquire_cli__load_live_wrench_config,
    run_acquisition as capture_and_acquire_cli__run_acquisition,
)
from scripts.capture_start_anchor import run_capture as capture_and_acquire_cli__run_capture
from utils.clock import host_time_s as capture_and_acquire_cli__host_time_s


class capture_and_acquire_cli__FakeAdapter:
    def __init__(self, *, fail_on=()):
        self.connected = False
        self.calls = []
        self.sequence = 0
        self._state_thread = capture_and_acquire_cli__threading.Thread()
        self.fail_on = {fail_on} if isinstance(fail_on, str) else set(fail_on)

    def _raise_if_requested(self, name):
        if name in self.fail_on:
            raise RuntimeError(f"simulated {name} failure")

    def connect(self):
        self.calls.append("connect")
        self._raise_if_requested("connect")
        self.connected = True

    def disconnect(self):
        self.calls.append("disconnect")
        self._raise_if_requested("disconnect")
        self.connected = False

    def is_connected(self):
        return self.connected

    def start_state_stream(self):
        self.calls.append("start_state_stream")

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")
        self._raise_if_requested("stop_state_stream")

    def read_state_frame(self):
        self.calls.append("read_state_frame")
        self.sequence += 1
        now = capture_and_acquire_cli__host_time_s()
        return capture_and_acquire_cli__KinematicStateFrame(
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
            velocity_source="unavailable",
            joint_position_rad=(0.0,) * 6,
            joint_velocity_radps=None,
            pose_time_s=now,
            joint_time_s=now,
            velocity_time_s=None,
            operation_state="IDLE",
            collision_state=None,
            controller_error=None,
        )

    def read_tcp_pose(self):
        return (0.4, 0.0, 0.3, 0.1, 0.2, 0.3)

    def read_joint_positions(self):
        return (0.0,) * 6

    def read_robot_metadata(self):
        self.calls.append("read_robot_metadata")
        return {
            "robot_model": "xMate6-Pro",
            "robot_serial_number": "SN-FAKE",
            "controller_version": "controller-fake",
        }

    def read_internal_wrench(self):
        self.calls.append("read_internal_wrench")
        now = capture_and_acquire_cli__host_time_s()
        return capture_and_acquire_cli__RobotWrenchFrame(
            sequence_id=self.sequence,
            host_query_start_s=now,
            host_query_end_s=now,
            host_publish_s=now,
            host_monotonic_time_s=now,
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

    def get_robot_state_summary(self):
        return {"connected": self.connected, "xcore_sdk_version": "fake"}

    def __getattr__(self, name):
        if any(token in name.lower() for token in ("enable", "power", "move", "drag", "calibrate", "clear")):
            raise AssertionError(f"forbidden method accessed: {name}")
        raise AttributeError(name)


def capture_and_acquire_cli___track_episode_logger_close(monkeypatch):
    original = capture_and_acquire_cli__acquire_cli.EpisodeLogger
    instances = []

    class TrackingEpisodeLogger(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            instances.append(self)

        def close(self, *args, **kwargs):
            self.close_calls += 1
            return super().close(*args, **kwargs)

    monkeypatch.setattr(capture_and_acquire_cli__acquire_cli, "EpisodeLogger", TrackingEpisodeLogger)
    return instances


class capture_and_acquire_cli__StopFailureAcquisition:
    stop_message = "simulated acquisition.stop failure"

    def __init__(self, adapter, logger, **kwargs):
        self.adapter = adapter
        self.logger = logger
        self.wrench_provider = kwargs.get("wrench_provider")
        self.background_error = None

    def start(self, *, manage_connection):
        assert manage_connection is False
        return self

    def wait_until_healthy(self, timeout_s):
        return True

    def latest_health(self):
        return capture_and_acquire_cli__SimpleNamespace(invalid_reason="")

    def stop(self):
        self.adapter.calls.append("acquisition.stop")
        raise RuntimeError(self.stop_message)


class capture_and_acquire_cli__RefusingStopAcquisition(capture_and_acquire_cli__StopFailureAcquisition):
    stop_message = (
        "acquisition_threads_did_not_stop:wrench; refusing SDK disconnect "
        "while a native query may still be active"
    )


class capture_and_acquire_cli__FakeWrenchProvider:
    """Explicit provider double for wiring regressions.

    It reproduces the parts of ``WrenchProcessProvider`` that the acquisition
    lifecycle actually touches, so the CLI wiring is exercised without an OS
    process or a native SDK import.
    """

    def __init__(self, *, rate_hz=50.0):
        self.config = {"mode": "offline", "rate_hz": rate_hz}
        self.budgets = capture_and_acquire_cli__WrenchBudgets(
            startup_s=5.0,
            query_s=0.5,
            progress_s=2.0,
            sample_age_s=2.0,
            ack_s=2.0,
            log_s=2.0,
            shutdown_s=0.5,
            terminate_s=0.5,
            state_age_s=2.0,
            skew_s=2.0,
            provenance="NOT_SAFETY_THRESHOLDS",
        )
        self.host_monotonic_time_s = None
        self.last_good = None
        self.cleanup = None
        self.process = None
        self.failure_latch = []
        self.started = False
        self.stopped = False
        self.failed_reasons = []

    def start(self):
        self.started = True
        self._publish()
        return self

    def _publish(self):
        now = capture_and_acquire_cli__host_time_s()
        self.host_monotonic_time_s = now
        self.last_good = capture_and_acquire_cli__RobotWrenchFrame(
            sequence_id=1,
            host_query_start_s=now,
            host_query_end_s=now,
            host_publish_s=now,
            host_monotonic_time_s=now,
            wall_time_iso="",
            timestamp_source="FAKE_PROVIDER",
            valid=True,
            invalid_reason="",
            raw_force_frame="world",
            cartesian_force_raw_n=(1.0, 2.0, 3.0),
            cartesian_torque_raw_nm=(0.1, 0.2, 0.3),
            joint_measured_torque_nm=(1.0,) * 6,
            joint_external_torque_nm=(0.5,) * 6,
        )

    def poll(self, now_ns=None):
        self._publish()
        return {
            "STREAM_HEALTH": True,
            "PROCESS_ALIVE": not self.stopped,
            "CURRENT_QUERY_STATE": "SUCCEEDED",
            "FAILURE_LATCH": tuple(self.failure_latch),
        }

    def health(self):
        return self.poll()

    def fail(self, reason):
        self.failure_latch.append(reason)
        self.failed_reasons.append(reason)

    def stop(self):
        self.stopped = True
        self.cleanup = {"NORMAL_CLEANUP": True, "FORCED_WORKER_TERMINATION": False}
        return dict(self.cleanup)


def test_capture_and_acquire_cli__capture_saves_unreviewed_anchor_and_never_moves(tmp_path):
    adapter = capture_and_acquire_cli__FakeAdapter()
    path = tmp_path / "anchor.json"
    result = capture_and_acquire_cli__run_capture(
        robot_ip="192.0.2.1",
        output_path=path,
        anchor_id="anchor_cli_test",
        tool_name="rehab_cuff",
        workpiece_name="rehab_fixture",
        adapter_factory=lambda _ip: adapter,
    )
    payload = capture_and_acquire_cli__json.loads(path.read_text(encoding="utf-8"))
    assert result["motion_commanded"] is False
    assert payload["reviewed"] is False
    assert payload["robot_model"] == "xMate6-Pro"
    assert payload["robot_serial_number"] == "SN-FAKE"
    assert payload["controller_version"] == "controller-fake"
    assert payload["tool_name"] == "rehab_cuff"
    assert payload["workpiece_name"] == "rehab_fixture"
    assert payload["created_at"]
    assert adapter.calls == [
        "connect",
        "start_state_stream",
        "read_state_frame",
        "read_robot_metadata",
        "stop_state_stream",
        "disconnect",
    ]


def test_capture_and_acquire_cli__capture_partial_connect_failure_still_attempts_disconnect(tmp_path):
    adapter = capture_and_acquire_cli__FakeAdapter(fail_on="connect")
    with capture_and_acquire_cli__pytest.raises(RuntimeError, match="simulated connect failure"):
        capture_and_acquire_cli__run_capture(
            robot_ip="192.0.2.1",
            output_path=tmp_path / "anchor.json",
            adapter_factory=lambda _ip: adapter,
        )
    assert adapter.calls == ["connect", "disconnect"]


def test_capture_and_acquire_cli__capture_stream_cleanup_failure_does_not_skip_disconnect(tmp_path):
    adapter = capture_and_acquire_cli__FakeAdapter(fail_on="stop_state_stream")
    with capture_and_acquire_cli__pytest.raises(RuntimeError, match="anchor capture cleanup failure"):
        capture_and_acquire_cli__run_capture(
            robot_ip="192.0.2.1",
            output_path=tmp_path / "anchor.json",
            tool_name="rehab_cuff",
            workpiece_name="rehab_fixture",
            adapter_factory=lambda _ip: adapter,
        )
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]
    assert adapter.connected is False


def test_capture_and_acquire_cli__acquire_writes_five_files_without_motion(tmp_path):
    """End-to-end wiring regression for the reviewed observation path.

    The provider is injected explicitly, exactly as the live path does, so this
    test fails if the CLI silently drops back to the removed parent-native
    wrench fallback.
    """

    adapter = capture_and_acquire_cli__FakeAdapter()
    provider = capture_and_acquire_cli__FakeWrenchProvider()
    episode = tmp_path / "episode"

    def provider_factory(_episode_dir, rate_hz):
        assert rate_hz == 50.0
        return provider

    result = capture_and_acquire_cli__run_acquisition(
        robot_ip="192.0.2.1",
        episode_dir=episode,
        duration_s=0.03,
        adapter_factory=lambda _ip: adapter,
        mode="offline",
        wrench_provider_factory=provider_factory,
    )
    assert result["completed"]
    assert result["motion_commanded"] is False
    assert result["wrench_mode"] == "offline"
    assert provider.started is True
    assert provider.stopped is True
    for filename in (
        "robot_state.csv",
        "robot_wrench.csv",
        "trajectory_command.csv",
        "aligned_snapshot.csv",
        "metadata.json",
    ):
        assert (episode / filename).is_file()
    # The explicit provider is the only wrench source that may exist.
    assert "read_internal_wrench" not in adapter.calls
    assert "connect" in adapter.calls
    assert "disconnect" in adapter.calls
    assert not any(
        token in call
        for call in adapter.calls
        for token in ("enable", "power", "move", "drag", "calibrate", "clear_error")
    )


def test_capture_and_acquire_cli__acquire_defaults_to_state_only_without_wrench_provider(tmp_path):
    """The default must not silently construct an unreviewed wrench source."""

    adapter = capture_and_acquire_cli__FakeAdapter()
    episode = tmp_path / "state_only_episode"
    result = capture_and_acquire_cli__run_acquisition(
        robot_ip="192.0.2.1",
        episode_dir=episode,
        duration_s=0.01,
        adapter_factory=lambda _ip: adapter,
    )
    assert result["completed"]
    assert result["wrench_mode"] == "state-only"
    assert "read_internal_wrench" not in adapter.calls
    metadata = capture_and_acquire_cli__json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["wrench_mode"] == "state-only"
    assert metadata["reviewed_observation_acquisition"] is False


def test_capture_and_acquire_cli__acquire_offline_mode_builds_explicit_isolated_provider(monkeypatch, tmp_path):
    """``mode='offline'`` must construct the process-isolated provider."""

    adapter = capture_and_acquire_cli__FakeAdapter()
    provider = capture_and_acquire_cli__FakeWrenchProvider()
    captured = {}

    def build(episode_dir, *, wrench_hz):
        captured["episode_dir"] = episode_dir
        captured["wrench_hz"] = wrench_hz
        return provider

    monkeypatch.setattr(capture_and_acquire_cli__acquire_cli, "build_offline_wrench_provider", build)
    result = capture_and_acquire_cli__run_acquisition(
        robot_ip="192.0.2.1",
        episode_dir=tmp_path / "offline_episode",
        duration_s=0.01,
        adapter_factory=lambda _ip: adapter,
        mode="offline",
    )
    assert result["wrench_mode"] == "offline"
    assert captured["wrench_hz"] == 50.0
    assert provider.started and provider.stopped


def test_capture_and_acquire_cli__acquire_rejects_conflicting_mode_and_provider(tmp_path):
    adapter = capture_and_acquire_cli__FakeAdapter()
    with capture_and_acquire_cli__pytest.raises(ValueError, match="incompatible with state-only"):
        capture_and_acquire_cli__run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "conflict",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
            mode="state-only",
            wrench_provider_factory=lambda _dir, _hz: capture_and_acquire_cli__FakeWrenchProvider(),
        )


def test_capture_and_acquire_cli__acquire_live_mode_requires_reviewed_config(tmp_path):
    adapter = capture_and_acquire_cli__FakeAdapter()
    with capture_and_acquire_cli__pytest.raises(ValueError, match="reviewed live_wrench_config"):
        capture_and_acquire_cli__run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "unreviewed_live",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
            mode="live",
        )


def test_capture_and_acquire_cli__acquire_factory_failure_still_closes_logger(monkeypatch, tmp_path):
    loggers = capture_and_acquire_cli___track_episode_logger_close(monkeypatch)

    def fail_factory(_robot_ip):
        raise RuntimeError("simulated adapter factory failure")

    with capture_and_acquire_cli__pytest.raises(RuntimeError, match="adapter factory failure"):
        capture_and_acquire_cli__run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "factory_failure",
            duration_s=0.01,
            adapter_factory=fail_factory,
        )

    assert len(loggers) == 1
    assert loggers[0].close_calls == 1
    assert loggers[0].status == "failed"


def test_capture_and_acquire_cli__acquire_connect_failure_attempts_stop_disconnect_and_close(
    monkeypatch,
    tmp_path,
):
    loggers = capture_and_acquire_cli___track_episode_logger_close(monkeypatch)
    adapter = capture_and_acquire_cli__FakeAdapter(fail_on="connect")

    with capture_and_acquire_cli__pytest.raises(RuntimeError, match="simulated connect failure"):
        capture_and_acquire_cli__run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "connect_failure",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
        )

    assert adapter.calls == ["connect", "stop_state_stream", "disconnect"]
    assert loggers[0].close_calls == 1
    assert loggers[0].status == "failed"


def test_capture_and_acquire_cli__acquire_stop_failure_retries_cleanup_stages_independently(
    monkeypatch,
    tmp_path,
):
    loggers = capture_and_acquire_cli___track_episode_logger_close(monkeypatch)
    monkeypatch.setattr(capture_and_acquire_cli__acquire_cli, "RealRobotAcquisition", capture_and_acquire_cli__StopFailureAcquisition)
    adapter = capture_and_acquire_cli__FakeAdapter(fail_on="stop_state_stream")

    with capture_and_acquire_cli__pytest.raises(RuntimeError, match="acquisition.stop failure"):
        capture_and_acquire_cli__run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "stop_failure",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
        )

    assert adapter.calls[-3:] == [
        "acquisition.stop",
        "stop_state_stream",
        "disconnect",
    ]
    assert adapter.connected is False
    assert loggers[0].close_calls == 1
    assert loggers[0].status == "failed"


def test_capture_and_acquire_cli__acquire_does_not_bypass_live_thread_disconnect_refusal(
    monkeypatch,
    tmp_path,
):
    loggers = capture_and_acquire_cli___track_episode_logger_close(monkeypatch)
    monkeypatch.setattr(
        capture_and_acquire_cli__acquire_cli,
        "RealRobotAcquisition",
        capture_and_acquire_cli__RefusingStopAcquisition,
    )
    adapter = capture_and_acquire_cli__FakeAdapter()

    with capture_and_acquire_cli__pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        capture_and_acquire_cli__run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "refused_disconnect",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
        )

    assert adapter.calls == ["connect", "acquisition.stop"]
    assert adapter.connected is True
    assert loggers[0].close_calls == 0
    assert loggers[0].status == "failed"


# ============================================================================
# Source: tests/test_run_rehab_experiment.py
# ============================================================================

"""The unified runner must fail offline before any forbidden connection."""



from dataclasses import replace as run_rehab_experiment__replace
import json as run_rehab_experiment__json
from pathlib import Path as run_rehab_experiment__Path

import pytest as run_rehab_experiment__pytest

from config import settings as run_rehab_experiment__settings
from control.execution_preflight import OPERATOR_CONFIRMATION as run_rehab_experiment__OPERATOR_CONFIRMATION
from control.start_anchor import FixedTcpOrientation as run_rehab_experiment__FixedTcpOrientation, StartAnchor as run_rehab_experiment__StartAnchor
from control.start_anchored_relative_trajectory import (
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID as run_rehab_experiment__FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    build_start_anchored_relative_trajectory as run_rehab_experiment__build_start_anchored_relative_trajectory,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_NOMINAL_ID as run_rehab_experiment__MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH as run_rehab_experiment__DEFAULT_REFERENCE_PATH
from safety.experiment_safety import ExperimentSafetyConfig as run_rehab_experiment__ExperimentSafetyConfig
from scripts.run_rehab_experiment import run_execute as run_rehab_experiment__run_execute
import scripts.run_rehab_experiment as run_rehab_experiment__runner_module


run_rehab_experiment__SOURCE_CANDIDATE_DIRECTORY = (
    artifact_path("lower_limb_sim/data/reference_candidates")
)


def run_rehab_experiment___reviewed_files(
    tmp_path, *, trajectory_id=run_rehab_experiment__FIRST_ROBOT_TRIAL_TRAJECTORY_ID
):
    pose = (0.4, 0.0, 0.3, 0.1, 0.2, 0.3)
    reference = (
        run_rehab_experiment__DEFAULT_REFERENCE_PATH
        if trajectory_id.endswith("slow")
        else run_rehab_experiment__SOURCE_CANDIDATE_DIRECTORY
        / "reference_measured_asymmetric_closed_nominal.csv"
    )
    trajectory, audit, _ = run_rehab_experiment__build_start_anchored_relative_trajectory(
        reference,
        current_tcp_start_pose=pose,
        rehab_frame=__import__(
            "control.start_anchored_relative_trajectory", fromlist=["RehabFrameConfig"]
        ).RehabFrameConfig((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), True),
    )
    anchor = run_rehab_experiment__StartAnchor(
        capture_host_time_s=1.0,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=run_rehab_experiment__FixedTcpOrientation(pose[3:]),
        robot_joint_positions=(0.0,) * 6,
        trajectory_id=trajectory_id,
        reference_start_q_hip=float(trajectory.iloc[0]["q_hip_ref"]),
        reference_start_q_knee=float(trajectory.iloc[0]["q_knee_ref"]),
        anchor_id="anchor_runner_test",
        robot_model="xMate6-Pro",
        robot_serial_number="SN-RUNNER",
        controller_version="controller-test",
        tool_name="rehab_cuff",
        workpiece_name="rehab_fixture",
        reviewed=True,
    )
    anchor_path = tmp_path / "anchor.json"
    anchor.save_json(anchor_path)
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(
        run_rehab_experiment__json.dumps(
            {
                "rehab_x_axis_in_base": [1.0, 0.0, 0.0],
                "rehab_z_axis_in_base": [0.0, 0.0, 1.0],
                "reviewed": True,
                "notes": "synthetic test",
            }
        ),
        encoding="utf-8",
    )
    safety = run_rehab_experiment__ExperimentSafetyConfig(
        max_tcp_speed_m_s=10.0,
        max_tcp_acceleration_m_s2=10.0,
        max_start_anchor_position_error_m=0.01,
        max_start_anchor_orientation_error_rad=0.01,
        max_command_lateness_s=0.1,
        max_force_n=100.0,
        max_torque_nm=100.0,
        max_state_age_s=1.0,
        max_wrench_age_s=1.0,
        max_state_wrench_skew_s=1.0,
        workspace_min_base_m=(-2.0, -2.0, -2.0),
        workspace_max_base_m=(2.0, 2.0, 2.0),
        expected_robot_model="xMate6-Pro",
        expected_robot_serial_number="SN-RUNNER",
        expected_controller_version="controller-test",
        reviewed_tool_name="rehab_cuff",
        reviewed_workpiece_name="rehab_fixture",
        reviewed_payload_mass_kg=1.25,
        reviewed_payload_cog_m=(0.0, 0.0, 0.08),
        reviewed_payload_inertia_kg_m2=(0.01, 0.02, 0.03),
        reviewed_joint_soft_limits_rad=tuple((-2.0, 2.0) for _ in range(6)),
        reviewed_rt_filter_hz=25.0,
        reviewed_rt_network_tolerance_percent=20.0,
        robot_identity_reviewed=True,
        tool_workpiece_reviewed=True,
        payload_configuration_reviewed=True,
        collision_configuration_reviewed=True,
        joint_soft_limits_reviewed=True,
        realtime_configuration_reviewed=True,
        reviewed=True,
        notes="synthetic test values",
    )
    safety_path = tmp_path / "safety.json"
    safety.save_json(safety_path)
    return anchor_path, frame_path, safety_path


def run_rehab_experiment___call(tmp_path, factory, **overrides):
    trajectory_id = overrides.pop(
        "trajectory_id", run_rehab_experiment__FIRST_ROBOT_TRIAL_TRAJECTORY_ID
    )
    anchor, frame, safety = run_rehab_experiment___reviewed_files(tmp_path, trajectory_id=trajectory_id)
    values = dict(
        robot_ip="192.0.2.1",
        episode_dir=tmp_path / "episode",
        anchor_path=anchor,
        requested_anchor_id="anchor_runner_test",
        frame_config_path=frame,
        safety_config_path=safety,
        trajectory_id=trajectory_id,
        enable_motion=True,
        operator_confirmation=run_rehab_experiment__OPERATOR_CONFIRMATION,
        adapter_factory=factory,
    )
    values.update(overrides)
    return run_rehab_experiment__run_execute(**values)


def test_run_rehab_experiment__missing_enable_motion_fails_before_adapter_construction(tmp_path):
    calls = []
    with run_rehab_experiment__pytest.raises(PermissionError, match="enable_motion_flag_missing"):
        run_rehab_experiment___call(
            tmp_path,
            lambda ip: calls.append(ip),
            enable_motion=False,
        )
    assert calls == []
    assert not (tmp_path / "episode").exists()


def test_run_rehab_experiment__same_session_without_release_blocks_before_adapter_construction(tmp_path):
    calls = []
    with run_rehab_experiment__pytest.raises(PermissionError, match="requires a reviewed release"):
        run_rehab_experiment___call(tmp_path, lambda ip: calls.append(ip), prepare_realtime=True)
    assert calls == []


def test_run_rehab_experiment__wrong_operator_confirmation_fails_before_connection(tmp_path):
    calls = []
    with run_rehab_experiment__pytest.raises(PermissionError, match="operator_confirmation_missing"):
        run_rehab_experiment___call(tmp_path, lambda ip: calls.append(ip), operator_confirmation="yes")
    assert calls == []


def test_run_rehab_experiment__nominal_is_refused_before_adapter_construction(tmp_path):
    calls = []
    with run_rehab_experiment__pytest.raises(PermissionError, match="not_whitelisted"):
        run_rehab_experiment___call(
            tmp_path,
            lambda ip: calls.append(ip),
            trajectory_id=run_rehab_experiment__MEASURED_ASYMMETRIC_NOMINAL_ID,
        )
    assert calls == []


def test_run_rehab_experiment__real_default_factory_requires_explicit_local_interface_before_connection(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(run_rehab_experiment__settings, "ROBOT_LOCAL_IP", "")
    with run_rehab_experiment__pytest.raises(
        PermissionError,
        match="reference_release_not_approved_for_first_robot_trial",
    ):
        run_rehab_experiment___call(tmp_path, None)
    assert not (tmp_path / "episode").exists()


class run_rehab_experiment___AllowedOfflinePreflight:
    """Stand-in for a preflight that passed every static gate."""

    def require_allowed(self) -> None:
        return None


def test_run_rehab_experiment__execute_requires_reviewed_wrench_config_before_connection(
    monkeypatch,
    tmp_path,
):
    # Isolate the new wiring guard from the frozen release gate, which would
    # otherwise be the first failure.  The point under test is that execution
    # cannot reach the adapter without an explicit isolated wrench provider.
    monkeypatch.setattr(
        run_rehab_experiment__runner_module,
        "evaluate_offline_execution_request",
        lambda **kwargs: run_rehab_experiment___AllowedOfflinePreflight(),
    )
    calls = []
    with run_rehab_experiment__pytest.raises(PermissionError, match="reviewed live wrench configuration"):
        run_rehab_experiment___call(
            tmp_path,
            lambda ip: calls.append(ip),
            local_ip="192.0.2.10",
        )
    assert calls == []


def test_run_rehab_experiment__execute_rejects_wrench_config_with_mismatched_connection(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        run_rehab_experiment__runner_module,
        "evaluate_offline_execution_request",
        lambda **kwargs: run_rehab_experiment___AllowedOfflinePreflight(),
    )
    calls = []
    with run_rehab_experiment__pytest.raises(PermissionError, match="wrench connection robot_ip must match"):
        run_rehab_experiment___call(
            tmp_path,
            lambda ip: calls.append(ip),
            local_ip="192.0.2.10",
            wrench_hz=50.0,
            live_wrench_config={
                "connection": {"robot_ip": "192.0.2.99", "local_ip": "192.0.2.10",
                               "robot_class": "xMateRobot"}
            },
        )
    assert calls == []


def test_run_rehab_experiment__empty_load_realtime_session_prepares_and_cleans_up_in_order():
    calls = []

    class Native:
        rt_network_tolerance_percent = 0
        rt_filter_hz = 0.0

        def enable_realtime(self, *, load):
            calls.append(("enable_realtime", load))

        def stop_realtime(self, *, switch_to_nrt):
            calls.append(("stop_realtime", switch_to_nrt))

        def disable(self):
            calls.append(("disable",))

    class Adapter:
        native_robot = Native()

        def connect(self):
            calls.append(("connect",))

        def disconnect(self):
            calls.append(("disconnect",))

    adapter = Adapter()
    run_rehab_experiment__runner_module._prepare_empty_load_realtime_session(
        adapter,
        filter_hz=50.0,
        network_tolerance_percent=20.0,
    )
    assert adapter.native_robot.rt_filter_hz == 50.0
    assert adapter.native_robot.rt_network_tolerance_percent == 20
    assert run_rehab_experiment__runner_module._cleanup_empty_load_realtime_session(adapter) == []
    assert calls == [
        ("connect",),
        ("enable_realtime", 0.0),
        ("stop_realtime", True),
        ("disable",),
        ("disconnect",),
    ]


def test_run_rehab_experiment__empty_load_realtime_session_rejects_fractional_network_tolerance():
    class Adapter:
        def connect(self):
            raise AssertionError("must reject before connection")

    with run_rehab_experiment__pytest.raises(ValueError, match="integer percent"):
        run_rehab_experiment__runner_module._prepare_empty_load_realtime_session(
            Adapter(),
            filter_hz=50.0,
            network_tolerance_percent=20.5,
        )


# ============================================================================
# Source: tests/test_robot_trial_release.py
# ============================================================================

"""Hash-bound trial release tests; no robot connection or motion."""



from datetime import datetime as robot_trial_release__datetime, timedelta as robot_trial_release__timedelta, timezone as robot_trial_release__timezone
import json as robot_trial_release__json
from pathlib import Path as robot_trial_release__Path

import pytest as robot_trial_release__pytest

from safety.robot_trial_release import (
    build_robot_trial_release_approval as robot_trial_release__build_robot_trial_release_approval,
    load_and_validate_robot_trial_release as robot_trial_release__load_and_validate_robot_trial_release,
    sha256_file as robot_trial_release__sha256_file,
    write_robot_trial_release as robot_trial_release__write_robot_trial_release,
)
robot_trial_release___reviewed_files = run_rehab_experiment___reviewed_files
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH as robot_trial_release__DEFAULT_REFERENCE_PATH


def robot_trial_release___wrench_config(path: robot_trial_release__Path, *, allowed_operation_states=None) -> robot_trial_release__Path:
    payload = {
        "schema_version": 1,
        "scope": "LIVE_OBSERVATION_ACQUISITION",
        "reviewed": True,
        "connection": {"robot_ip": "192.0.2.1", "local_ip": "192.0.2.2", "robot_class": "xMateRobot"},
        "expected_identity": {
            "robot_model": "xMate6-Pro", "robot_serial": "SN-RUNNER",
            "controller_version": "controller-test", "sdk_version": "0.7.0",
        },
        "allowed_power_states": ["off", "on"],
        "budgets": {
            "startup_s": 5.0, "query_s": 0.5, "progress_s": 2.0, "sample_age_s": 1.0,
            "ack_s": 1.0, "log_s": 1.0, "shutdown_s": 1.0, "terminate_s": 1.0,
            "state_age_s": 1.0, "skew_s": 1.0, "provenance": "REVIEWED_SITE_BUDGET",
        },
    }
    if allowed_operation_states is not None:
        payload["allowed_operation_states"] = allowed_operation_states
    path.write_text(robot_trial_release__json.dumps(payload), encoding="utf-8")
    return path


def robot_trial_release___draft(tmp_path: robot_trial_release__Path):
    anchor, frame, safety = robot_trial_release___reviewed_files(tmp_path)
    wrench = robot_trial_release___wrench_config(tmp_path / "wrench.json")
    expires = (robot_trial_release__datetime.now(robot_trial_release__timezone.utc) + robot_trial_release__timedelta(hours=2)).isoformat()
    payload = robot_trial_release__build_robot_trial_release_approval(
        reviewer="safety", operator="operator", expires_at=expires,
        robot_identity={"robot_model": "xMate6-Pro", "robot_serial_number": "SN-RUNNER",
                        "controller_version": "controller-test", "sdk_version": "0.7.0"},
        reference_path=robot_trial_release__DEFAULT_REFERENCE_PATH, safety_config_path=safety,
        frame_config_path=frame, anchor_path=anchor, wrench_config_path=wrench,
        collision_source="events", collision_event_max_age_s=0.5, wrench_hz=50.0,
        notes="synthetic reviewed release test",
    )
    return payload, anchor, frame, safety, wrench


def test_robot_trial_release__draft_is_explicitly_unapproved_and_written_once(tmp_path):
    payload, *_ = robot_trial_release___draft(tmp_path)
    assert payload["approved"] is False
    output = tmp_path / "release.json"
    robot_trial_release__write_robot_trial_release(output, payload)
    with robot_trial_release__pytest.raises(FileExistsError):
        robot_trial_release__write_robot_trial_release(output, payload)


def test_robot_trial_release__approval_requires_bound_site_review_and_all_evidence(tmp_path):
    payload, anchor, frame, safety, wrench = robot_trial_release___draft(tmp_path)
    draft = tmp_path / "draft.json"
    robot_trial_release__write_robot_trial_release(draft, payload)
    evidence = {}
    for name in ("controller.json", "acquisition.json", "wrench.jsonl", "stop.json"):
        path = tmp_path / name
        path.write_text("validated\n", encoding="utf-8")
        evidence[name] = {"passed": True, "path": name, "sha256": robot_trial_release__sha256_file(path)}
    site = tmp_path / "site_review.json"
    site.write_text(robot_trial_release__json.dumps({
        "scope": payload["scope"], "decision": "approved", "reviewer": payload["reviewer"],
        "reviewed_at": robot_trial_release__datetime.now(robot_trial_release__timezone.utc).isoformat(),
        "bundle_sha256": payload["bundle_sha256"], "records": {
            "controller_safety": evidence["controller.json"],
            "acquisition_stability": evidence["acquisition.json"],
            "wrench_semantics": evidence["wrench.jsonl"],
            "setup_and_stop": evidence["stop.json"],
        },
    }), encoding="utf-8")
    approved = dict(payload, approved=True, site_review={"path": site.name, "sha256": robot_trial_release__sha256_file(site)})
    approved_path = tmp_path / "approved.json"
    robot_trial_release__write_robot_trial_release(approved_path, approved)
    validated = robot_trial_release__load_and_validate_robot_trial_release(
        approved_path, reference_path=robot_trial_release__DEFAULT_REFERENCE_PATH, safety_config_path=safety,
        frame_config_path=frame, anchor_path=anchor, wrench_config_path=wrench,
    )
    assert validated.payload["approved"] is True
    assert validated.payload["collision_source"] == "events"
    validated.require_current(check_files=True)


def test_robot_trial_release__input_change_invalidates_approved_release(tmp_path):
    payload, anchor, frame, safety, wrench = robot_trial_release___draft(tmp_path)
    draft = tmp_path / "draft.json"
    robot_trial_release__write_robot_trial_release(draft, payload)
    evidence = {}
    for name in ("controller.json", "acquisition.json", "wrench.jsonl", "stop.json"):
        evidence_path = tmp_path / name
        evidence_path.write_text("validated\n", encoding="utf-8")
        evidence[name] = {"passed": True, "path": name, "sha256": robot_trial_release__sha256_file(evidence_path)}
    site = tmp_path / "site.json"
    site.write_text(robot_trial_release__json.dumps({
        "scope": payload["scope"], "decision": "approved", "reviewer": payload["reviewer"],
        "reviewed_at": robot_trial_release__datetime.now(robot_trial_release__timezone.utc).isoformat(),
        "bundle_sha256": payload["bundle_sha256"], "records": {
            "controller_safety": evidence["controller.json"],
            "acquisition_stability": evidence["acquisition.json"],
            "wrench_semantics": evidence["wrench.jsonl"],
            "setup_and_stop": evidence["stop.json"],
        },
    }), encoding="utf-8")
    approved = dict(payload, approved=True, site_review={"path": site.name, "sha256": robot_trial_release__sha256_file(site)})
    approved_path = tmp_path / "approved.json"
    robot_trial_release__write_robot_trial_release(approved_path, approved)
    safety.write_text(safety.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with robot_trial_release__pytest.raises(PermissionError, match="input binding mismatch"):
        robot_trial_release__load_and_validate_robot_trial_release(
            approved_path, reference_path=robot_trial_release__DEFAULT_REFERENCE_PATH, safety_config_path=safety,
            frame_config_path=frame, anchor_path=anchor, wrench_config_path=wrench,
        )


# ============================================================================
# Source: tests/test_robot_trajectory_executor.py
# ============================================================================

"""Offline scheduler and unified-stop tests."""



from dataclasses import replace as robot_trajectory_executor__replace
from types import SimpleNamespace as robot_trajectory_executor__SimpleNamespace
import threading as robot_trajectory_executor__threading
import time as robot_trajectory_executor__time

import numpy as robot_trajectory_executor__np
import pandas as robot_trajectory_executor__pd
import pytest as robot_trajectory_executor__pytest

from collection.episode_logger import EpisodeLogger as robot_trajectory_executor__EpisodeLogger, EpisodeLoggerError as robot_trajectory_executor__EpisodeLoggerError
from collection.real_robot_acquisition import AcquisitionHealth as robot_trajectory_executor__AcquisitionHealth
from control.execution_preflight import (
    ExecutionPreflight as robot_trajectory_executor__ExecutionPreflight,
    _LIVE_PREFLIGHT_TOKEN as robot_trajectory_executor___LIVE_PREFLIGHT_TOKEN,
    audit_execution_trajectory_content as robot_trajectory_executor__audit_execution_trajectory_content,
    experiment_safety_digest as robot_trajectory_executor__experiment_safety_digest,
    trajectory_execution_digest as robot_trajectory_executor__trajectory_execution_digest,
)
from control.robot_trajectory_executor import RokaeMotionExecutor as robot_trajectory_executor__RokaeMotionExecutor
from lower_limb_sim.config import L1 as robot_trajectory_executor__L1, L2 as robot_trajectory_executor__L2
from lower_limb_sim.kinematics import forward_kinematics as robot_trajectory_executor__forward_kinematics
from safety.experiment_safety import ExperimentSafetyConfig as robot_trajectory_executor__ExperimentSafetyConfig


def robot_trajectory_executor___safety():
    return robot_trajectory_executor__ExperimentSafetyConfig(
        max_tcp_speed_m_s=1_000_000.0,
        max_tcp_acceleration_m_s2=1_000_000_000.0,
        max_start_anchor_position_error_m=0.01,
        max_start_anchor_orientation_error_rad=0.01,
        max_command_lateness_s=0.1,
        max_force_n=20.0,
        max_torque_nm=5.0,
        max_state_age_s=1.0,
        max_wrench_age_s=1.0,
        max_state_wrench_skew_s=1.0,
        workspace_min_base_m=(-1.0, -1.0, -1.0),
        workspace_max_base_m=(1.0, 1.0, 1.0),
        expected_robot_model="xMate6-Pro",
        expected_robot_serial_number="SN-EXECUTOR",
        expected_controller_version="controller-test",
        reviewed_tool_name="rehab_cuff",
        reviewed_workpiece_name="rehab_fixture",
        reviewed_payload_mass_kg=1.25,
        reviewed_payload_cog_m=(0.0, 0.0, 0.08),
        reviewed_payload_inertia_kg_m2=(0.01, 0.02, 0.03),
        reviewed_joint_soft_limits_rad=tuple((-2.0, 2.0) for _ in range(6)),
        reviewed_rt_filter_hz=25.0,
        reviewed_rt_network_tolerance_percent=20.0,
        robot_identity_reviewed=True,
        tool_workpiece_reviewed=True,
        payload_configuration_reviewed=True,
        collision_configuration_reviewed=True,
        joint_soft_limits_reviewed=True,
        realtime_configuration_reviewed=True,
        reviewed=True,
        notes="synthetic test values",
    )


def robot_trajectory_executor___health():
    return robot_trajectory_executor__AcquisitionHealth(
        host_time_s=1.0,
        state_time_s=1.0,
        wrench_time_s=1.0,
        state_age_s=0.0,
        wrench_age_s=0.0,
        state_wrench_skew_s=0.0,
        state_valid=True,
        wrench_valid=True,
        state_thread_alive=True,
        wrench_thread_alive=True,
        alignment_thread_alive=True,
        query_duration_ms=1.0,
        force_magnitude_n=1.0,
        torque_magnitude_nm=0.1,
        valid=True,
        invalid_reason="",
    )


class robot_trajectory_executor__Acquisition:
    def __init__(self):
        self.health = robot_trajectory_executor___health()
        self.summary = {
            "connected": True,
            "state_valid": True,
            "state_stream_thread_alive": True,
            "operation_state": "RT_CONTROLLING",
            "collision_state_query_valid": True,
            "collision_state": False,
            "joint_position_rad": [0.0] * 6,
            "robot_metadata": {
                "robot_model": "xMate6-Pro",
                "robot_serial_number": "SN-EXECUTOR",
                "controller_version": "controller-test",
                "joint_soft_limits_rad": [[-2.0, 2.0]] * 6,
                "sdk_tool_payload": {
                    "toolset_load_mass_kg": 1.25,
                    "toolset_load_cog_m": [0.0, 0.0, 0.08],
                    "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
                    "sdk_available_tool_names": ["rehab_cuff"],
                    "sdk_available_workobject_names": ["rehab_fixture"],
                    "active_hmi_tool_workobject_verified": False,
                },
            },
        }
        self.adapter = robot_trajectory_executor__SimpleNamespace(
            get_robot_state_summary=lambda: dict(self.summary)
        )

    def latest_health(self):
        return self.health

    def latest_state_frame(self):
        return robot_trajectory_executor__SimpleNamespace(
            valid=True,
            tcp_position_m=(0.3, 0.0, 0.4),
            tcp_orientation_rad=(0.0, 0.0, 0.0),
        )


class robot_trajectory_executor__Motion:
    def __init__(self):
        self.calls = []

    def attach_externally_prepared(self, *, reviewed_filter_hz):
        self.calls.append(("attach", reviewed_filter_hz))

    def start_cartesian_hold(self, pose):
        self.calls.append(("start", tuple(pose)))

    def send_cartesian_target(self, pose):
        self.calls.append(("send", tuple(pose)))

    def has_motion_error(self):
        return False

    def request_stop(self, reason):
        self.calls.append(("stop", reason))


def robot_trajectory_executor___trajectory():
    time_s = robot_trajectory_executor__np.asarray([0.0, 0.001, 0.002])
    q_hip = robot_trajectory_executor__np.asarray([0.1, 0.1001, 0.1])
    q_knee = robot_trajectory_executor__np.asarray([0.2, 0.2001, 0.2])
    _, _, pull_x, pull_z = robot_trajectory_executor__forward_kinematics(q_hip, q_knee, robot_trajectory_executor__L1, robot_trajectory_executor__L2)
    delta = robot_trajectory_executor__np.column_stack(
        (
            pull_x - pull_x[0],
            robot_trajectory_executor__np.zeros(3),
            pull_z - pull_z[0],
        )
    )
    delta[-1] = 0.0
    xyz = robot_trajectory_executor__np.asarray((0.3, 0.0, 0.4)) + delta
    velocity = robot_trajectory_executor__np.gradient(xyz, time_s, axis=0, edge_order=2)
    acceleration = robot_trajectory_executor__np.gradient(velocity, time_s, axis=0, edge_order=2)
    return robot_trajectory_executor__pd.DataFrame(
        {
            "time_s": time_s,
            "trajectory_id": ["reference_measured_asymmetric_closed_slow"] * 3,
            "trajectory_phase": ["start", "middle", "end"],
            "delta_x_R": delta[:, 0],
            "delta_y_R": delta[:, 1],
            "delta_z_R": delta[:, 2],
            "tcp_x_base": xyz[:, 0],
            "tcp_y_base": xyz[:, 1],
            "tcp_z_base": xyz[:, 2],
            "tcp_rx": [0.0, 0.0, 0.0],
            "tcp_ry": [0.0, 0.0, 0.0],
            "tcp_rz": [0.0, 0.0, 0.0],
            "tcp_vx_base": velocity[:, 0],
            "tcp_vy_base": velocity[:, 1],
            "tcp_vz_base": velocity[:, 2],
            "tcp_ax_base": acceleration[:, 0],
            "tcp_ay_base": acceleration[:, 1],
            "tcp_az_base": acceleration[:, 2],
            "q_hip_ref": q_hip,
            "q_knee_ref": q_knee,
            "theta_shank_ref": q_hip - q_knee,
            "trajectory_valid": [True, True, True],
            "experiment_mode": ["start_anchored_relative"] * 3,
            "tcp_orientation_strategy": ["fixed_at_start_anchor"] * 3,
        }
    )


def robot_trajectory_executor___preflight(allowed=True, trajectory=None, safety=None):
    bound_trajectory = robot_trajectory_executor___trajectory() if trajectory is None else trajectory
    bound_safety = robot_trajectory_executor___safety() if safety is None else safety
    content_reasons, max_speed, max_acceleration = robot_trajectory_executor__audit_execution_trajectory_content(
        bound_trajectory,
        bound_safety,
        expected_trajectory_id="reference_measured_asymmetric_closed_slow",
    )
    assert content_reasons == ()
    return robot_trajectory_executor__ExecutionPreflight(
        allowed=allowed,
        reasons=() if allowed else ("blocked_test",),
        trajectory_id="reference_measured_asymmetric_closed_slow",
        anchor_id="anchor",
        maximum_tcp_speed_m_s=max_speed,
        maximum_tcp_acceleration_m_s2=max_acceleration,
        runtime_robot_summary={"synthetic_test": True},
        evaluation_phase="live",
        trajectory_sha256=robot_trajectory_executor__trajectory_execution_digest(bound_trajectory),
        experiment_safety_sha256=robot_trajectory_executor__experiment_safety_digest(bound_safety),
        _live_token=robot_trajectory_executor___LIVE_PREFLIGHT_TOKEN,
    )


def test_robot_trajectory_executor__executor_logs_before_each_send_and_stops_once(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    result = executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=True)
    assert result.completed
    assert result.commands_dispatched == 3
    assert logger.row_counts["trajectory_command"] == 3
    assert motion.calls[0] == ("attach", 25.0)
    assert motion.calls[1][0] == "start"
    assert [call[0] for call in motion.calls if isinstance(call, tuple)].count("send") == 2
    assert [call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"] == [
        ("stop", "trajectory_completed")
    ]


def test_robot_trajectory_executor__preflight_failure_sends_no_motion(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(PermissionError, match="blocked_test"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight(False))
    logger.close(completed=False)
    assert motion.calls == []


def test_robot_trajectory_executor__safety_event_failure_after_start_stops_before_next_target(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()
    event = {"valid": True, "collision_state": False, "invalid_reason": ""}
    acquisition.adapter.get_safety_event_status = lambda: dict(event)

    class EventLostMotion(robot_trajectory_executor__Motion):
        def start_cartesian_hold(self, pose):
            super().start_cartesian_hold(pose)
            event.update(valid=False, collision_state=None, invalid_reason="safety_event_expired")

    motion = EventLostMotion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    try:
        with robot_trajectory_executor__pytest.raises(RuntimeError, match="safety_event_expired"):
            executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
        assert not any(call[0] == "send" for call in motion.calls)
        assert any(call[0] == "stop" for call in motion.calls)
        event.update(valid=True, collision_state=False, invalid_reason="")
        with robot_trajectory_executor__pytest.raises(RuntimeError):
            executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    finally:
        logger.close(completed=False)


def test_robot_trajectory_executor__runtime_stale_state_enters_unified_stop(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()
    acquisition.health = robot_trajectory_executor__replace(acquisition.health, state_age_s=2.0)
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="state_stale"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert len([call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"]) == 1


def test_robot_trajectory_executor__logging_failure_prevents_target_dispatch(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    original_append = logger._append

    def fail_append(stream_name, values):
        if stream_name == "trajectory_command":
            raise robot_trajectory_executor__EpisodeLoggerError("simulated disk failure")
        original_append(stream_name, values)

    logger._append = fail_append
    with robot_trajectory_executor__pytest.raises(robot_trajectory_executor__EpisodeLoggerError, match="disk failure"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger._append = original_append
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] in {"start", "send"}
        for call in motion.calls
    )
    assert len([call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"]) == 1


def test_robot_trajectory_executor__operator_stop_is_idempotent(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    executor.request_stop("operator stop")
    executor.request_stop("second stop")
    logger.close(completed=False)
    assert motion.calls == [("stop", "operator stop")]


def test_robot_trajectory_executor__prestart_stop_cannot_be_cleared_by_execute(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    executor.request_stop("operator prestart stop")
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="stopped before start"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert motion.calls == [("stop", "operator prestart stop")]


def test_robot_trajectory_executor__executor_cannot_be_reused_after_completion(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="single-use"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=True)


def test_robot_trajectory_executor__attach_is_followed_by_fresh_health_gate_before_start(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()

    class HealthChangingMotion(robot_trajectory_executor__Motion):
        def attach_externally_prepared(self, *, reviewed_filter_hz):
            super().attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            acquisition.health = robot_trajectory_executor__replace(acquisition.health, state_age_s=2.0)

    motion = HealthChangingMotion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="post-attach.*state_stale"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert motion.calls[0] == ("attach", 25.0)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_robot_trajectory_executor__collision_change_during_attach_blocks_realtime_start(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()

    class CollisionChangingMotion(robot_trajectory_executor__Motion):
        def attach_externally_prepared(self, *, reviewed_filter_hz):
            super().attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            acquisition.summary["collision_state"] = True

    motion = CollisionChangingMotion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="collision_detected_after_attachment"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_robot_trajectory_executor__payload_change_during_attach_blocks_realtime_start(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()

    class PayloadChangingMotion(robot_trajectory_executor__Motion):
        def attach_externally_prepared(self, *, reviewed_filter_hz):
            super().attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            acquisition.summary["robot_metadata"]["sdk_tool_payload"][
                "toolset_load_mass_kg"
            ] = 2.0

    motion = PayloadChangingMotion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="payload_mass_changed_after_preflight"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_robot_trajectory_executor__slow_intent_logging_misses_deadline_without_dispatch(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    original_append = logger._append

    def slow_append(stream_name, values):
        if stream_name == "trajectory_command":
            robot_trajectory_executor__time.sleep(0.02)
        original_append(stream_name, values)

    logger._append = slow_append
    motion = robot_trajectory_executor__Motion()
    safety = robot_trajectory_executor__replace(robot_trajectory_executor___safety(), max_command_lateness_s=0.005)
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, safety)
    with robot_trajectory_executor__pytest.raises((RuntimeError, robot_trajectory_executor__EpisodeLoggerError), match="durability|deadline"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight(safety=safety))
    assert logger.wait_for_pending_writes(1.0)
    logger._append = original_append
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] in {"start", "send"}
        for call in motion.calls
    )


def test_robot_trajectory_executor__stop_failure_never_returns_completed_result(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()

    class StopFailingMotion(robot_trajectory_executor__Motion):
        def __init__(self):
            super().__init__()
            self.stop_attempts = 0

        def request_stop(self, reason):
            super().request_stop(reason)
            self.stop_attempts += 1
            if self.stop_attempts == 1:
                raise RuntimeError("simulated stopLoop failure")

    motion = StopFailingMotion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="stopLoop failure"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert len(
        [call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"]
    ) == 2


def test_robot_trajectory_executor__unbound_preflight_never_reaches_motion(tmp_path):
    trajectory = robot_trajectory_executor___trajectory()
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    manual = robot_trajectory_executor__replace(
        robot_trajectory_executor___preflight(trajectory=trajectory),
        evaluation_phase="offline",
        _live_token=None,
    )
    with robot_trajectory_executor__pytest.raises(PermissionError, match="not_live_bound"):
        executor.execute(trajectory, manual)
    logger.close(completed=False)
    assert motion.calls == []


def test_robot_trajectory_executor__preflight_cannot_be_reused_with_a_different_safety_snapshot(tmp_path):
    trajectory = robot_trajectory_executor___trajectory()
    approved = robot_trajectory_executor___preflight(trajectory=trajectory)
    changed_safety = robot_trajectory_executor__replace(robot_trajectory_executor___safety(), reviewed_rt_filter_hz=30.0)
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(
        motion,
        robot_trajectory_executor__Acquisition(),
        logger,
        changed_safety,
    )
    with robot_trajectory_executor__pytest.raises(PermissionError, match="safety_binding_mismatch"):
        executor.execute(trajectory, approved)
    logger.close(completed=False)
    assert motion.calls == []


def test_robot_trajectory_executor__mutated_trajectory_never_reaches_motion(tmp_path):
    trajectory = robot_trajectory_executor___trajectory()
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode-mutated").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    bound = robot_trajectory_executor___preflight(trajectory=trajectory)
    trajectory.loc[1, "tcp_x_base"] += 1e-6
    with robot_trajectory_executor__pytest.raises(PermissionError, match="binding_mismatch"):
        executor.execute(trajectory, bound)
    logger.close(completed=False)
    assert motion.calls == []


def test_robot_trajectory_executor__string_false_is_not_accepted_as_trajectory_valid(tmp_path):
    trajectory = robot_trajectory_executor___trajectory()
    bound = robot_trajectory_executor___preflight(trajectory=trajectory)
    trajectory["trajectory_valid"] = ["True", "False", "True"]
    # Rebind only the digest to exercise the scheduler's independent content
    # audit instead of stopping at mutation detection.
    bound = robot_trajectory_executor__replace(
        bound,
        trajectory_sha256=robot_trajectory_executor__trajectory_execution_digest(trajectory),
    )
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(PermissionError, match="not_strict_boolean"):
        executor.execute(trajectory, bound)
    logger.close(completed=False)
    assert motion.calls == []


def test_robot_trajectory_executor__final_metadata_persistence_is_followed_by_fresh_collision_gate(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()
    motion = robot_trajectory_executor__Motion()
    original_update = logger.update_metadata

    def collision_after_approval_metadata(updates):
        original_update(updates)
        if updates.get("initial_hold_intent_persisted") is True:
            acquisition.summary["collision_state"] = True

    logger.update_metadata = collision_after_approval_metadata
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="collision_detected_after_attachment"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.update_metadata = original_update
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_robot_trajectory_executor__metadata_delay_expires_initial_hold_deadline_before_start(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    original_update = logger.update_metadata

    def slow_final_metadata(updates):
        original_update(updates)
        if updates.get("initial_hold_intent_persisted") is True:
            robot_trajectory_executor__time.sleep(0.01)

    logger.update_metadata = slow_final_metadata
    safety = robot_trajectory_executor__replace(robot_trajectory_executor___safety(), max_command_lateness_s=0.005)
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, safety)
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="initial hold deadline expired"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight(safety=safety))
    logger.update_metadata = original_update
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_robot_trajectory_executor__health_change_during_command_logging_blocks_that_target(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    acquisition = robot_trajectory_executor__Acquisition()
    motion = robot_trajectory_executor__Motion()
    original_bounded = logger.append_trajectory_command_bounded
    calls = 0

    def change_health_after_logging(*args, **kwargs):
        nonlocal calls
        original_bounded(*args, **kwargs)
        calls += 1
        if calls == 2:
            acquisition.health = robot_trajectory_executor__replace(acquisition.health, wrench_age_s=2.0)

    logger.append_trajectory_command_bounded = change_health_after_logging
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, acquisition, logger, robot_trajectory_executor___safety())
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="post-log.*wrench_stale"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.append_trajectory_command_bounded = original_bounded
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "send" for call in motion.calls
    )


def test_robot_trajectory_executor__operator_stop_published_during_logging_prevents_later_send(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()
    motion = robot_trajectory_executor__Motion()
    executor = robot_trajectory_executor__RokaeMotionExecutor(motion, robot_trajectory_executor__Acquisition(), logger, robot_trajectory_executor___safety())
    original_bounded = logger.append_trajectory_command_bounded
    second_log_entered = robot_trajectory_executor__threading.Event()
    release_second_log = robot_trajectory_executor__threading.Event()
    calls = 0

    def block_second_log(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_log_entered.set()
            assert release_second_log.wait(timeout=1.0)
        return original_bounded(*args, **kwargs)

    logger.append_trajectory_command_bounded = block_second_log
    errors: list[BaseException] = []

    def run_execution():
        try:
            executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
        except BaseException as exc:
            errors.append(exc)

    worker = robot_trajectory_executor__threading.Thread(target=run_execution)
    worker.start()
    assert second_log_entered.wait(timeout=1.0)
    executor.request_stop("operator_stop_during_logging")
    release_second_log.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert errors
    logger.append_trajectory_command_bounded = original_bounded
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "send" for call in motion.calls
    )
    assert executor.stop_reason == "operator_stop_during_logging"


def test_robot_trajectory_executor__operator_reason_can_never_be_reported_as_normal_completion(tmp_path):
    logger = robot_trajectory_executor__EpisodeLogger(tmp_path / "episode").start()

    class OperatorWinsAtCompletion(robot_trajectory_executor__RokaeMotionExecutor):
        def request_stop(self, reason):
            if reason == "trajectory_completed" and self.stop_reason is None:
                super().request_stop("operator_stop_at_completion_boundary")
            return super().request_stop(reason)

    motion = robot_trajectory_executor__Motion()
    executor = OperatorWinsAtCompletion(
        motion,
        robot_trajectory_executor__Acquisition(),
        logger,
        robot_trajectory_executor___safety(),
    )
    with robot_trajectory_executor__pytest.raises(RuntimeError, match="did not reach confirmed normal completion"):
        executor.execute(robot_trajectory_executor___trajectory(), robot_trajectory_executor___preflight())
    logger.close(completed=False)
    assert executor.stop_reason == "operator_stop_at_completion_boundary"


# ============================================================================
# Source: tests/test_robot_leg_round.py
# ============================================================================

"""Offline integration: synthetic hardware data, real validation and frozen learner."""
from copy import deepcopy as robot_leg_round__deepcopy
from datetime import datetime as robot_leg_round__datetime, timedelta as robot_leg_round__timedelta, timezone as robot_leg_round__timezone
import ctypes as robot_leg_round__ctypes
import json as robot_leg_round__json
import os as robot_leg_round__os
from pathlib import Path as robot_leg_round__Path
import struct as robot_leg_round__struct
import subprocess as robot_leg_round__subprocess
import sys as robot_leg_round__sys
import threading as robot_leg_round__threading
import time as robot_leg_round__time
from types import SimpleNamespace as robot_leg_round__SimpleNamespace

import numpy as robot_leg_round__np
import pandas as robot_leg_round__pd
import pytest as robot_leg_round__pytest

from collection.twincat_leg import (AdsInputImage as robot_leg_round__AdsInputImage, AmsAddr as robot_leg_round__AmsAddr, AmsNetId as robot_leg_round__AmsNetId, capture as robot_leg_round__capture, decode_image as robot_leg_round__decode_image,
    LegCaptureProcess as robot_leg_round__LegCaptureProcess, validate_config as robot_leg_round__validate_config, atomic_json as robot_leg_round__atomic_json)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH as robot_leg_round__DEFAULT_REFERENCE_PATH
from measurement_validation.robot_leg_round import aligned_measurement as robot_leg_round__aligned_measurement, endpoints as robot_leg_round__endpoints, validate_analysis as robot_leg_round__validate_analysis
from safety.robot_trial_release import read_json as robot_leg_round__read_json, sha256_file as robot_leg_round__sha256_file, EVIDENCE_KINDS as robot_leg_round__EVIDENCE_KINDS, LOADED_EVIDENCE as robot_leg_round__LOADED_EVIDENCE
import scripts.run_robot_leg_round as robot_leg_round__rounds

robot_leg_round__ROOT = robot_leg_round__Path(__file__).resolve().parents[1]


def robot_leg_round__portable_test_config():
    """Relocate checkout files only, always retaining frozen expected hashes."""
    c = robot_leg_round__rounds.paths(section_path("robot_leg_round"))
    if robot_leg_round__os.environ.get("REHAB_TEST_ROUND_CONFIG"):
        portable = robot_leg_round__rounds.paths(robot_leg_round__os.environ["REHAB_TEST_ROUND_CONFIG"])
        assert portable["algorithm_sha256"] == c["algorithm_sha256"]
        assert sorted(portable["algorithm_dependencies"].values()) == sorted(c["algorithm_dependencies"].values())
        c = portable
    def relocate(name):
        parts = name.replace("\\", "/").split("/rehab_robot/", 1)
        return str(robot_leg_round__ROOT / parts[1]) if len(parts) == 2 else name
    c["algorithm_dependencies"] = {relocate(p): h for p, h in c["algorithm_dependencies"].items()}
    for item in c["candidates"]:
        item["path"] = relocate(item["path"])
    return c


def robot_leg_round__capture_config():
    c = robot_leg_round__read_json(section_path("twincat_leg_capture"))
    c.update(mapping_reviewed=True, ads_dll_path=__file__, ads_timeout_ms=100,
        startup_timeout_s=.2, max_progress_age_s=.2, shutdown_timeout_s=.2,
        heartbeat_s=.005, no_new_sample_timeout_s=.02, poll_interval_s=.0001)
    c["guards"] = {f"axis{axis}": dict(max_abs_torque_nm=50., expected_operation_enabled=False) for axis in (1, 4)}
    return c


def robot_leg_round__image(toggle=0, status=0, torque=2.):
    data = bytearray(1366)
    for axis, offset in enumerate((0, 23, 46, 69), 1):
        robot_leg_round__struct.pack_into("<HiihiHbf", data, offset, status, -1000*axis, -20, -12, 9, 0, 8, torque)
    data[1365] = toggle << 4
    return data


def robot_leg_round__analysis_config():
    return dict(reviewed=True, alignment_reviewed=True, sensor_receive_delay_s=0.,
        command_application_delay_s=0., max_gap_s=.01, max_read_duration_s=.01,
        max_dispatch_time_error_s=.02, derivative_rate_hz=500., derivative_window_s=.1,
        max_start_angle_error_deg=1., max_tracking_error_deg=2.)


def robot_leg_round__calibration_config():
    return dict(reviewed=True, independent_validation_passed=True,
        joints={name: dict(counts_per_degree=1000., zero_counts=0., flexion_sign=1,
            safe_min_deg=-10., safe_max_deg=140., torque_sign_to_inverse_dynamics=1) for name in ("hip", "knee")})


def robot_leg_round__measured_fixture(directory, reference=robot_leg_round__DEFAULT_REFERENCE_PATH):
    """Explicit synthetic data for software tests only; no hardware is contacted."""
    ref = robot_leg_round__pd.read_csv(reference)
    t = robot_leg_round__np.arange(-.1, float(ref.time_s.iloc[-1])+.102, .002)
    origin = 123456789000000
    ticks = origin + robot_leg_round__np.rint(t*1e9).astype(robot_leg_round__np.int64)
    data = dict(sample_index=robot_leg_round__np.arange(len(t)), receive_elapsed_ms=(t-t[0])*1000,
        host_monotonic_ns=ticks, query_start_ns=ticks-100000, query_end_ns=ticks)
    for name, axis in (("hip", 1), ("knee", 4)):
        q = robot_leg_round__np.interp(t, ref.time_s, ref[f"q_{name}_rad"])
        data.update({f"axis{axis}_position_raw": robot_leg_round__np.rint(robot_leg_round__np.rad2deg(q)*1000),
            f"axis{axis}_torque_3c1a_nm": 3.+axis+robot_leg_round__np.sin(t), f"axis{axis}_statusword_hex": "0x0000",
            f"axis{axis}_error_code_hex": "0x0000", f"axis{axis}_fault": False,
            f"axis{axis}_operation_enabled": False})
    csv = directory / "leg.csv"
    robot_leg_round__pd.DataFrame(data).to_csv(csv, index=False)
    dispatch = directory / "dispatch.csv"
    d = origin + robot_leg_round__np.rint(ref.time_s.to_numpy()*1e9).astype(robot_leg_round__np.int64)
    robot_leg_round__pd.DataFrame(dict(trajectory_time_s=ref.time_s, dispatch_start_ns=d-100000, dispatch_end_ns=d)).to_csv(dispatch, index=False)
    return csv, dispatch


def robot_leg_round__training_request(directory, measured):
    c = robot_leg_round__portable_test_config()
    return dict(algorithm_root=c["algorithm_root"], algorithm_sha256=c["algorithm_sha256"],
        algorithm_dependencies=c["algorithm_dependencies"], geometry_m=c["geometry_m"],
        candidates=c["candidates"], reference_candidate_id=c["reference_candidate_id"],
        setup_id="SYNTHETIC_SOFTWARE_TEST_ONLY", setup_sha256="0"*64,
        reference_sha256=robot_leg_round__sha256_file(robot_leg_round__DEFAULT_REFERENCE_PATH), history=[dict(episode_id="synthetic-test",
            candidate_id=c["reference_candidate_id"], measured_path=str(measured),
            measured_sha256=robot_leg_round__sha256_file(measured), endpoints=dict(E3=1., E2=1., peak_ratio=1.))])


def test_robot_leg_round__decoder_axis_offsets_raw_signedness_and_native_structs():
    c = robot_leg_round__capture_config()
    row = robot_leg_round__decode_image(robot_leg_round__image(1), c)
    assert robot_leg_round__ctypes.sizeof(robot_leg_round__AmsNetId) == 6 and robot_leg_round__ctypes.sizeof(robot_leg_round__AmsAddr) == 8
    assert row["axis4_position_raw"] == -4000
    assert row["axis1_torque_6077_raw_bits"] == 65524
    assert row["axis1_torque_6077_pct_rated"] is None
    assert row["axis4_torque_3c1a_nm"] == 2. and row["input_toggle"] == 1
    c["torque_6077_signedness_verified"] = True
    assert robot_leg_round__decode_image(robot_leg_round__image(), c)["axis1_torque_6077_pct_rated"] == robot_leg_round__pytest.approx(-1.2)
    with robot_leg_round__pytest.raises(ValueError, match="incomplete"):
        robot_leg_round__decode_image(robot_leg_round__image()[:-1], c)


def test_robot_leg_round__polling_cannot_exceed_source_cycle():
    c = robot_leg_round__capture_config()
    c["poll_interval_s"] = .002
    with robot_leg_round__pytest.raises(ValueError, match="shorter than"):
        robot_leg_round__validate_config(c)


def test_robot_leg_round__capture_ready_and_completed_only_after_close(tmp_path):
    stop = robot_leg_round__threading.Event()
    stop.wait = lambda *a: robot_leg_round__pytest.fail("Event.wait is too coarse for Windows 2 ms capture")
    messages = []
    class Source:
        closed = False
        n = 0
        def __init__(self, config): pass
        def read(self):
            self.n += 1
            return robot_leg_round__image(self.n % 2)
        def close(self): self.closed = True
    source = Source(None)
    def notify(record):
        messages.append(record)
        if record["state"] == "recording":
            assert not source.closed
            stop.set()
        else:
            assert source.closed
            assert len(robot_leg_round__pd.read_csv(tmp_path / "leg.csv")) == 2
    assert robot_leg_round__capture(robot_leg_round__capture_config(), tmp_path / "leg.csv", stop_event=stop,
        notify=notify, source_factory=lambda c: source) == 2
    assert [m["state"] for m in messages] == ["recording", "completed"]


@robot_leg_round__pytest.mark.parametrize("kind", ["fault", "torque", "stuck", "short"])
def test_robot_leg_round__capture_failure_never_acknowledges_complete(tmp_path, kind):
    closed, messages = [], []
    class Source:
        def __init__(self, config): pass
        def read(self):
            if kind == "short": return b""
            return robot_leg_round__image(status=8 if kind == "fault" else 0, torque=51. if kind == "torque" else 2.)
        def close(self): closed.append(True)
    with robot_leg_round__pytest.raises((RuntimeError, TimeoutError, ValueError)):
        robot_leg_round__capture(robot_leg_round__capture_config(), tmp_path / "leg.csv", stop_event=robot_leg_round__threading.Event(),
            notify=messages.append, source_factory=Source)
    assert closed and not any(m["state"] == "completed" for m in messages)


def test_robot_leg_round__supervisor_rejects_dead_stale_and_bad_protocol(tmp_path):
    p = tmp_path / "capture.json"
    robot_leg_round__atomic_json(p, robot_leg_round__capture_config())
    supervisor = robot_leg_round__LegCaptureProcess(p, tmp_path)
    assert supervisor.health_reason() == "leg_capture_process_not_running"
    supervisor.process = robot_leg_round__SimpleNamespace(poll=lambda: None)
    supervisor.status = dict(state="recording", last_sample_ns=robot_leg_round__time.perf_counter_ns())
    assert supervisor.health_reason() is None
    snapshot = supervisor.snapshot()
    snapshot["state"] = "completed"
    assert supervisor.snapshot()["state"] == "recording"
    supervisor.status["last_sample_ns"] = robot_leg_round__time.perf_counter_ns()-1_000_000_000
    assert supervisor.health_reason() == "leg_capture_stale"
    supervisor._error = "bad json"
    assert "protocol_error" in supervisor.health_reason()


@robot_leg_round__pytest.mark.parametrize("code, length", [(0, 1365), (1797, 1366)])
def test_robot_leg_round__native_ads_short_or_failed_read_is_rejected(code, length):
    reader = robot_leg_round__AdsInputImage.__new__(robot_leg_round__AdsInputImage)
    reader.config, reader.port, reader.address = robot_leg_round__capture_config(), 1, robot_leg_round__AmsAddr()
    def read(port, address, group, offset, size, buffer, actual):
        robot_leg_round__ctypes.cast(actual, robot_leg_round__ctypes.POINTER(robot_leg_round__ctypes.c_uint32))[0] = length
        return code
    reader.dll = robot_leg_round__SimpleNamespace(AdsSyncReadReqEx2=read)
    with robot_leg_round__pytest.raises(OSError, match="ADS input read failed"):
        reader.read()


@robot_leg_round__pytest.mark.parametrize("hung", [True, False])
def test_robot_leg_round__supervisor_requires_close_ack_and_bounds_shutdown(tmp_path, hung):
    config = tmp_path / "capture.json"
    robot_leg_round__atomic_json(config, robot_leg_round__capture_config())
    supervisor = robot_leg_round__LegCaptureProcess(config, tmp_path)
    calls = []
    class Process:
        returncode = None if hung else 0
        stdin = robot_leg_round__SimpleNamespace(write=lambda x: None, flush=lambda: None)
        def poll(self): return self.returncode
        def wait(self, timeout):
            calls.append(timeout)
            if self.returncode is None: raise robot_leg_round__subprocess.TimeoutExpired("test", timeout)
        def kill(self): self.returncode = -1
    supervisor.process = Process()
    supervisor._reader = robot_leg_round__SimpleNamespace(join=lambda timeout: None)
    supervisor._stderr = robot_leg_round__SimpleNamespace(close=lambda: calls.append("closed"))
    supervisor.status = dict(state="recording")
    with robot_leg_round__pytest.raises(TimeoutError if hung else RuntimeError): supervisor.stop()
    assert calls[-1] == "closed"
    if hung: assert calls[:-1] == [.2, 2]


def test_robot_leg_round__measured_alignment_and_reference_normalization(tmp_path):
    csv, dispatch = robot_leg_round__measured_fixture(tmp_path)
    measured, metrics = robot_leg_round__aligned_measurement(csv, dispatch, robot_leg_round__DEFAULT_REFERENCE_PATH, robot_leg_round__analysis_config(), robot_leg_round__calibration_config())
    ref = robot_leg_round__pd.read_csv(robot_leg_round__DEFAULT_REFERENCE_PATH)
    assert robot_leg_round__np.max(robot_leg_round__np.abs(measured.q_hip_rad-ref.q_hip_rad)) < .001
    assert robot_leg_round__np.isfinite(measured.select_dtypes(include="number")).all().all()
    assert robot_leg_round__endpoints(metrics, metrics) == dict(E3=1., E2=1., peak_ratio=1.)
    assert not metrics["automatically_tared"]


@robot_leg_round__pytest.mark.parametrize("kind", ["gap", "duplicate", "fault", "missing", "tracking", "dispatch", "short_read"])
def test_robot_leg_round__invalid_data_cannot_enter_training(tmp_path, kind):
    csv, dispatch = robot_leg_round__measured_fixture(tmp_path)
    f = robot_leg_round__pd.read_csv(csv)
    if kind == "gap": f = f.drop(index=range(200, 230))
    if kind == "duplicate": f.loc[200, "host_monotonic_ns"] = f.loc[199, "host_monotonic_ns"]
    if kind == "fault": f.loc[200, "axis4_fault"] = True
    if kind == "missing": f.loc[200, "axis4_torque_3c1a_nm"] = robot_leg_round__np.nan
    if kind == "tracking": f["axis4_position_raw"] += 10000
    if kind == "short_read": f.loc[200, "query_start_ns"] -= 100_000_000
    if kind == "dispatch":
        d = robot_leg_round__pd.read_csv(dispatch).iloc[:-1]
        d.to_csv(dispatch, index=False)
    f.to_csv(csv, index=False)
    with robot_leg_round__pytest.raises(ValueError):
        robot_leg_round__aligned_measurement(csv, dispatch, robot_leg_round__DEFAULT_REFERENCE_PATH, robot_leg_round__analysis_config(), robot_leg_round__calibration_config())


def test_robot_leg_round__missing_knee_is_not_substituted():
    c = robot_leg_round__calibration_config()
    c["joints"]["knee"] = None
    with robot_leg_round__pytest.raises(ValueError, match="missing knee"):
        robot_leg_round__validate_analysis(robot_leg_round__analysis_config(), c)


def test_robot_leg_round__real_frozen_training_worker(tmp_path):
    csv, dispatch = robot_leg_round__measured_fixture(tmp_path)
    measured, _ = robot_leg_round__aligned_measurement(csv, dispatch, robot_leg_round__DEFAULT_REFERENCE_PATH, robot_leg_round__analysis_config(), robot_leg_round__calibration_config())
    target = tmp_path / "measured.csv"
    measured.to_csv(target, index=False)
    request = tmp_path / "request.json"
    robot_leg_round__atomic_json(request, robot_leg_round__training_request(tmp_path, target))
    output = tmp_path / "update.json"
    result = robot_leg_round__subprocess.run([robot_leg_round__sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round", str(request),
        "--output", str(output)], cwd=robot_leg_round__ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    update = robot_leg_round__read_json(output)
    assert update["method"] == "FROZEN_RESIDUAL_GP_MEAN_GREEDY"
    assert len(update["theta"]) == 6
    assert update["action"]["role"] == "STOP"
    assert update["action"]["stop_reason"] == "DOMAIN_EXHAUSTED"


def test_robot_leg_round__unreviewed_setup_blocks_before_hardware(tmp_path):
    def forbidden(**kwargs): robot_leg_round__pytest.fail("must not connect")
    result = robot_leg_round__rounds.prepare(section_path("robot_leg_round"), tmp_path / "round", capture_fn=forbidden)
    assert result["status"] == "BLOCKED_BEFORE_CONNECTION"
    assert result["blockers"] and (tmp_path / "round/report.md").is_file()


def test_robot_leg_round__malformed_setup_still_produces_report(tmp_path):
    p = tmp_path / "bad.json"
    robot_leg_round__atomic_json(p, {})
    result = robot_leg_round__rounds.prepare(p, tmp_path / "round")
    assert result["status"] == "BLOCKED_BEFORE_CONNECTION"


def test_robot_leg_round__round_attempt_is_exclusive_and_persistent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(robot_leg_round__rounds, "_run_once", lambda *a, **k: calls.append(True))
    robot_leg_round__rounds.run(tmp_path, confirmation=robot_leg_round__rounds.CONFIRMATION)
    with robot_leg_round__pytest.raises(FileExistsError): robot_leg_round__rounds.run(tmp_path, confirmation=robot_leg_round__rounds.CONFIRMATION)
    assert calls == [True]
    with robot_leg_round__pytest.raises(PermissionError): robot_leg_round__rounds.run(tmp_path, confirmation="no")


def robot_leg_round__prepared_round(tmp_path, *, empty=False, capture_only=False):
    _reviewed_files = run_rehab_experiment___reviewed_files
    _wrench_config = robot_trial_release___wrench_config
    anchor, frame, safety = _reviewed_files(tmp_path)
    wrench = _wrench_config(tmp_path / "wrench.json", allowed_operation_states=["idle", "rtcontrolling"])
    configs = {}
    for key, value in (("leg_config", robot_leg_round__capture_config()), ("calibration", robot_leg_round__calibration_config()), ("analysis", robot_leg_round__analysis_config())):
        p = tmp_path / (key+".json")
        robot_leg_round__atomic_json(p, value)
        configs[key] = str(p)
    c = robot_leg_round__portable_test_config()
    c.update(configs, reviewed=True, setup_id="SYNTHETIC_TEST_ONLY", reviewer="test", operator="test",
        safety_config=str(safety), frame_config=str(frame), wrench_config=str(wrench),
        robot_ip="192.0.2.1", local_ip="192.0.2.2", robot_class="xMateRobot",
        tool_name="rehab_cuff", workpiece_name="rehab_fixture", wrench_hz=50.,
        collision_event_max_age_s=.5, expires_at=(robot_leg_round__datetime.now(robot_leg_round__timezone.utc)+robot_leg_round__timedelta(hours=1)).isoformat(),
        pre_roll_s=.001, post_roll_s=.001, max_leg_start_age_s=.5, realtime_preparation="same_session",
        robot_identity=dict(robot_model="xMate6-Pro", robot_serial_number="SN-RUNNER",
            controller_version="controller-test", sdk_version="0.7.0"), site_evidence={})
    if empty and capture_only:
        raise ValueError("test fixture cannot be both empty and loaded capture-only")
    c["trial_kind"] = ("empty_load" if empty else
        "dummy_leg_capture_only" if capture_only else "dummy_leg_reference")
    for kind in (*robot_leg_round__EVIDENCE_KINDS, *(() if empty else robot_leg_round__LOADED_EVIDENCE)):
        p = tmp_path / (kind+".txt")
        p.write_text("SYNTHETIC TEST EVIDENCE ONLY", encoding="ascii")
        c["site_evidence"][kind] = dict(passed=True, path=str(p), sha256=robot_leg_round__sha256_file(p))
    p = tmp_path / "round.json"
    robot_leg_round__atomic_json(p, c)
    assert robot_leg_round__rounds.check_config(p) == []
    def capture_anchor(**kwargs):
        robot_leg_round__Path(kwargs["output_path"]).write_bytes(anchor.read_bytes())
    directory = tmp_path / "session"
    assert robot_leg_round__rounds.prepare(p, directory, capture_fn=capture_anchor)["status"] == "AWAITING_ONSITE_CONFIRMATION"
    return directory


class robot_leg_round__SyntheticCapture:
    """Test-only source; never exposed by the production CLI."""
    def __init__(self, config, directory): self.directory = directory
    def start(self):
        self.directory.mkdir()
        robot_leg_round__measured_fixture(self.directory)
    def health_reason(self): return None
    def snapshot(self):
        ref = robot_leg_round__pd.read_csv(robot_leg_round__DEFAULT_REFERENCE_PATH).iloc[0]
        return dict(last_sample_ns=robot_leg_round__time.perf_counter_ns(), latest={f"axis{axis}_position_raw":
            robot_leg_round__np.rad2deg(ref[f"q_{name}_rad"])*1000 for name, axis in (("hip", 1), ("knee", 4))})
    def stop(self): pass


def robot_leg_round__synthetic_execute(**kwargs):
    assert kwargs["execution_guard"]() is None
    d = robot_leg_round__pd.read_csv(kwargs["episode_dir"].parent / "leg/dispatch.csv")
    for r in d.itertuples():
        kwargs["dispatch_observer"](r.trajectory_time_s, r.dispatch_start_ns, r.dispatch_end_ns)
    return dict(completed=True, commands_dispatched=len(d), same_session_realtime_cleanup_completed=True)


def test_robot_leg_round__full_round_with_real_release_analysis_and_training(tmp_path):
    directory = robot_leg_round__prepared_round(tmp_path)
    state = robot_leg_round__rounds.run(directory, confirmation=robot_leg_round__rounds.CONFIRMATION,
        capture_factory=robot_leg_round__SyntheticCapture, execute_fn=robot_leg_round__synthetic_execute)
    assert state["status"] == "ROUND_COMPLETE"
    assert state["training_status"] == "MODEL_UPDATED"
    assert state["endpoints"]["E3"] == 1.
    assert robot_leg_round__read_json(directory / "algorithm_ready.json")["status"] == "ALGORITHM_INITIALIZED"
    assert (directory / "report.md").is_file() and (directory / "quality.json").is_file()
    assert robot_leg_round__read_json(directory / "release_approval.json")["scope"] == robot_leg_round__rounds.LOADED_SCOPE
    # Changing a bound calibration invalidates the existing release.
    from safety.robot_trial_release import load_and_validate_robot_trial_release
    c = robot_leg_round__rounds.paths(tmp_path / "round.json")
    robot_leg_round__Path(c["calibration"]).write_text("{}", encoding="ascii")
    with robot_leg_round__pytest.raises(PermissionError):
        load_and_validate_robot_trial_release(directory / "release_approval.json", reference_path=robot_leg_round__DEFAULT_REFERENCE_PATH,
            anchor_path=directory / "reviewed_anchor.json", frame_config_path=c["frame_config"],
            safety_config_path=c["safety_config"], wrench_config_path=c["wrench_config"])


def test_robot_leg_round__empty_load_is_never_trained(tmp_path):
    directory = robot_leg_round__prepared_round(tmp_path, empty=True)
    state = robot_leg_round__rounds.run(directory, confirmation=robot_leg_round__rounds.CONFIRMATION,
        capture_factory=robot_leg_round__SyntheticCapture, execute_fn=robot_leg_round__synthetic_execute)
    assert state["training_status"] == "EXCLUDED_EMPTY_LOAD"
    assert not (directory / "training_request.json").exists()


def test_robot_leg_round__loaded_capture_only_keeps_guards_and_never_trains(tmp_path):
    directory = robot_leg_round__prepared_round(tmp_path, capture_only=True)
    state = robot_leg_round__rounds.run(directory, confirmation=robot_leg_round__rounds.CONFIRMATION,
        capture_factory=robot_leg_round__SyntheticCapture, execute_fn=robot_leg_round__synthetic_execute)
    assert state["status"] == "CAPTURE_ONLY_COMPLETE"
    assert state["training_status"] == "SKIPPED_CAPTURE_ONLY"
    assert (directory / "leg/leg.csv").is_file()
    assert (directory / "quality.json").is_file()
    assert (directory / "measured.csv").is_file()
    assert (directory / "measured_metrics.json").is_file()
    assert not (directory / "algorithm_initialization_request.json").exists()
    assert not (directory / "algorithm_ready.json").exists()
    assert not (directory / "training_request.json").exists()
    assert not (directory / "model_update.json").exists()


@robot_leg_round__pytest.mark.parametrize("kind", ["unready", "incomplete", "unclosed"])
def test_robot_leg_round__failed_round_keeps_partial_data_without_training(tmp_path, kind):
    directory = robot_leg_round__prepared_round(tmp_path)
    class Capture(robot_leg_round__SyntheticCapture):
        def health_reason(self): return "stale" if kind == "unready" else None
        def stop(self):
            if kind == "unclosed": raise RuntimeError("CSV not closed")
    def execute(**kwargs):
        assert kind != "unready", "robot must not execute before capture is ready"
        if kind == "incomplete": return dict(completed=False)
        return robot_leg_round__synthetic_execute(**kwargs)
    with robot_leg_round__pytest.raises(RuntimeError):
        robot_leg_round__rounds.run(directory, confirmation=robot_leg_round__rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
    state = robot_leg_round__read_json(directory / "round_status.json")
    assert state["status"] == "ROUND_FAILED" and state["training_status"] == "NOT_UPDATED"
    assert (directory / "leg/leg.csv").is_file()
    assert not (directory / "model_update.json").exists()


def test_robot_leg_round__external_capture_failure_stops_real_executor(tmp_path):
    from collection.episode_logger import EpisodeLogger
    from control.robot_trajectory_executor import RokaeMotionExecutor
    Motion = robot_trajectory_executor__Motion
    Acquisition = robot_trajectory_executor__Acquisition
    _trajectory = robot_trajectory_executor___trajectory
    _preflight = robot_trajectory_executor___preflight
    _safety = robot_trajectory_executor___safety
    motion = Motion()
    observed = []
    def observer(*args): observed.append(args)
    logger = EpisodeLogger(tmp_path / "episode").start()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety(),
        execution_guard=lambda: "leg_capture_stale" if observed else None, dispatch_observer=observer)
    try:
        with robot_leg_round__pytest.raises(RuntimeError, match="leg_capture_stale"):
            executor.execute(_trajectory(), _preflight())
    finally:
        logger.close(completed=False)
    assert len(observed) == 1
    assert any(c[0] == "stop" for c in motion.calls)
    assert not any(c[0] == "send" for c in motion.calls)


def test_robot_leg_round__unavailable_algorithm_blocks_before_capture_and_motion(tmp_path, monkeypatch):
    directory = robot_leg_round__prepared_round(tmp_path)
    def failed_worker(*args, **kwargs): raise RuntimeError("missing model resources")
    monkeypatch.setattr(robot_leg_round__rounds, "run_algorithm_worker", failed_worker)
    def forbidden(*a, **k): robot_leg_round__pytest.fail("hardware must not be used")
    with robot_leg_round__pytest.raises(RuntimeError, match="missing model resources"):
        robot_leg_round__rounds.run(directory, confirmation=robot_leg_round__rounds.CONFIRMATION, capture_factory=forbidden, execute_fn=forbidden)
    assert robot_leg_round__read_json(directory / "round_status.json")["status"] == "ROUND_FAILED"


def test_robot_leg_round__connection_class_mismatch_blocks_readonly_preparation(tmp_path):
    robot_leg_round__prepared_round(tmp_path)
    p = tmp_path / "round.json"
    c = robot_leg_round__read_json(p)
    c["robot_class"] = "WRONG_TEST_CLASS"
    robot_leg_round__atomic_json(p, c)
    assert any("connection/class must match" in x for x in robot_leg_round__rounds.check_config(p))


# ============================================================================
# Source: tests/test_capture_leg_segment.py
# ============================================================================

"""Bounded segment lifecycle, with test-only hardware substitute."""
import pytest as capture_leg_segment__pytest

from collection.twincat_leg import atomic_json as capture_leg_segment__atomic_json
from safety.robot_trial_release import read_json as capture_leg_segment__read_json
from scripts.capture_leg_segment import record as capture_leg_segment__record
capture_leg_segment__measured_fixture = robot_leg_round__measured_fixture


@capture_leg_segment__pytest.mark.parametrize("failure", [None, "start", "health", "close"])
def test_capture_leg_segment__segment_closes_and_retains_failure_report(tmp_path, failure):
    config = tmp_path / "config.json"
    capture_leg_segment__atomic_json(config, {})
    calls = []
    class Worker:
        def __init__(self, config, directory): self.directory = directory
        def start(self):
            calls.append("start")
            if failure == "start": raise RuntimeError("startup failed")
            capture_leg_segment__measured_fixture(self.directory)
        def health_reason(self): return "stale" if failure == "health" else None
        def stop(self):
            calls.append("stop")
            if failure == "close": raise RuntimeError("unclosed CSV")
    directory = tmp_path / "segment"
    if failure:
        with capture_leg_segment__pytest.raises(RuntimeError): capture_leg_segment__record(config, directory, .01, factory=Worker)
        assert capture_leg_segment__read_json(directory / "segment_status.json")["status"] == "CAPTURE_FAILED"
        assert not (directory / "quality.json").exists()
    else:
        result = capture_leg_segment__record(config, directory, .01, factory=Worker)
        assert result["status"] == "CAPTURE_COMPLETE" and not result["motion_commanded"]
        assert (directory / "quality.json").exists()
        with capture_leg_segment__pytest.raises(FileExistsError): capture_leg_segment__record(config, directory, .01, factory=Worker)
    assert calls[0] == "start" and "stop" in calls


@capture_leg_segment__pytest.mark.parametrize("duration", [0, -1, float('nan'), 3601])
def test_capture_leg_segment__bad_duration_never_creates_source(tmp_path, duration):
    def forbidden(*a): capture_leg_segment__pytest.fail("must validate before constructing source")
    with capture_leg_segment__pytest.raises(ValueError): capture_leg_segment__record(tmp_path / "config", tmp_path / "segment", duration, factory=forbidden)


# ============================================================================
# Source: tests/test_characterize_wrench_longrun.py
# ============================================================================

import pytest as characterize_wrench_longrun__pytest

from scripts.characterize_wrench_longrun import concurrency_summary as characterize_wrench_longrun__concurrency_summary, event_windows as characterize_wrench_longrun__event_windows, latency_summary as characterize_wrench_longrun__latency_summary


def characterize_wrench_longrun___row(index: int, latency_ms: float, *, success: bool = True, code=None):
    return {
        "latency_ms": latency_ms,
        "success": success,
        "error_code": code,
        "host_call_start_monotonic_ns": index * 20_000_000,
        "rt_updates_during_call": int(latency_ms / 8),
    }


def test_characterize_wrench_longrun__latency_buckets_deadlines_and_recovery_are_distinct() -> None:
    rows = [
        characterize_wrench_longrun___row(0, 1.0),
        characterize_wrench_longrun___row(1, 7.0),
        characterize_wrench_longrun___row(2, 15.0),
        characterize_wrench_longrun___row(3, 36.0),
        characterize_wrench_longrun___row(4, 75.0),
        characterize_wrench_longrun___row(5, 250.0),
        characterize_wrench_longrun___row(6, 750.0),
        characterize_wrench_longrun___row(7, 10_000.0, success=False, code=263),
        characterize_wrench_longrun___row(8, 1.0),
    ]
    result = characterize_wrench_longrun__latency_summary(rows, 50.0)
    assert result["latency_buckets"] == {
        "le_5_ms": 2,
        "gt_5_le_10_ms": 1,
        "gt_10_le_20_ms": 1,
        "gt_20_le_50_ms": 1,
        "gt_50_le_100_ms": 1,
        "gt_100_le_500_ms": 1,
        "gt_500_le_1000_ms": 1,
        "gt_1000_ms": 1,
    }
    assert result["deadline_miss_count"] == 5
    assert result["error_263_count"] == 1
    assert result["error_263_recovered_on_next_query_count"] == 1
    assert result["query_start_period_ms"]["mean"] == 20.0


def test_characterize_wrench_longrun__concurrency_summary_divides_timestamp_gap_by_sequence_delta() -> None:
    ticks = [
        {
            "main_loop_timestamp_ns": 0,
            "loop_period_ms": None,
            "rt_sequence": 10,
            "rt_timestamp_s": 1.000,
            "rt_age_ms": 2.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_age_ms": 5.0,
            "wrench_stale": False,
            "wrench_valid": True,
            "wrench_worker_alive": True,
            "wrench_query_in_flight": False,
        },
        {
            "main_loop_timestamp_ns": 10_000_000,
            "loop_period_ms": 10.0,
            "rt_sequence": 12,
            "rt_timestamp_s": 1.016,
            "rt_age_ms": 4.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_age_ms": 7.0,
            "wrench_stale": False,
            "wrench_valid": True,
            "wrench_worker_alive": True,
            "wrench_query_in_flight": True,
        },
    ]
    result = characterize_wrench_longrun__concurrency_summary(ticks)
    assert result["rt_source_period_ms"]["mean"] == characterize_wrench_longrun__pytest.approx(8.0)
    assert result["consumer_skipped_rt_sequence_count"] == 1
    assert result["rt_sequence_advance"] == 2


def test_characterize_wrench_longrun__event_windows_keep_previous_current_and_next_calls() -> None:
    rows = []
    for index in range(25):
        row = characterize_wrench_longrun___row(index, 1.0)
        row.update({
            "sequence_id": index + 1,
            "scheduled_time_ns": index,
            "call_start_ns": index,
            "call_end_ns": index + 1,
            "query_latency_ms": 1.0,
            "host_call_end_monotonic_ns": index + 1,
            "error_message": "",
            "latest_rt_sequence": index,
            "rt_sequence_after_call": index + 1,
            "time_since_last_success_ms": 20.0,
        })
        rows.append(row)
    rows[12]["query_latency_ms"] = 1_500.0
    rows[12]["latency_ms"] = 1_500.0
    windows = characterize_wrench_longrun__event_windows(rows)
    assert len(windows) == 1
    assert len(windows[0]["previous_successful"]) == 10
    assert windows[0]["current"]["sequence_id"] == 13
    assert len(windows[0]["next"]) == 10


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
# Source: tests/test_empty_load_attempt_audit.py
# ============================================================================

import csv as empty_load_attempt_audit__csv
import json as empty_load_attempt_audit__json

from scripts.audit_empty_load_attempts import analyze_attempt as empty_load_attempt_audit__analyze_attempt


def test_empty_load_attempt_audit__audit_separates_query_storage_and_sample_age(tmp_path):
    events = [
        dict(type="QUERY_SUCCESS", event_id=5, query_id=1,
             intent_ns=1_000_000_000, native_start_ns=1_110_000_000,
             completion_ns=1_111_000_000, frame=dict(host_monotonic_time_s=1.1105)),
        dict(type="DURABLE_ACK", child_event_id=5, durable_ns=1_165_000_000,
             timestamp_ns=1_180_000_000),
        dict(type="FAILURE_LATCH", reason="wrench_sample_stale_or_clock_invalid"),
        dict(type="CLEANUP_OUTCOME", outcome=dict(NORMAL_CLEANUP=True,
             failure_latch=["wrench_sample_stale_or_clock_invalid"])),
    ]
    (tmp_path / "wrench_query_events.jsonl").write_text(
        "\n".join(empty_load_attempt_audit__json.dumps(e) for e in events), encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with (tmp_path / "trajectory_command.csv").open("w", newline="") as stream:
        empty_load_attempt_audit__csv.writer(stream).writerow(["host_time_s", "x"])
    report = empty_load_attempt_audit__analyze_attempt(tmp_path)
    query = report["queries"][0]
    assert query["native_duration_ms"] == 1
    assert query["intent_to_native_ms"] == 110
    assert query["completion_to_durable_ms"] == 54
    assert round(query["sample_age_at_ack_ms"], 6) == 69.5
    assert report["cleanup_log_contradiction"]
    assert report["zero_recorded_trajectory_commands"]
    assert len(report["source_sha256"]) == 3


# ============================================================================
# Source: tests/test_empty_load_release_claim.py
# ============================================================================

"""Local single-use ledger tests; approvals and hardware are test doubles."""
from concurrent.futures import ThreadPoolExecutor as empty_load_release_claim__ThreadPoolExecutor
from dataclasses import replace as empty_load_release_claim__replace
from datetime import datetime as empty_load_release_claim__datetime, timedelta as empty_load_release_claim__timedelta, timezone as empty_load_release_claim__timezone
import hashlib as empty_load_release_claim__hashlib
import json as empty_load_release_claim__json
import multiprocessing as empty_load_release_claim__mp

import pytest as empty_load_release_claim__pytest

import safety.empty_load_commissioning_release as empty_load_release_claim__releases


def empty_load_release_claim___claim_in_spawned_process(empty_load_release_claim__release, root, results):
    empty_load_release_claim__releases._ROOT = root
    restored = empty_load_release_claim__replace(empty_load_release_claim__release, _token=empty_load_release_claim__releases._VALIDATED)
    try:
        restored.claim_execution()
    except PermissionError:
        results.put(False)
    else:
        results.put(True)


@empty_load_release_claim__pytest.fixture
def empty_load_release_claim__release(tmp_path, monkeypatch):
    monkeypatch.setattr(empty_load_release_claim__releases, "_ROOT", tmp_path)
    now = empty_load_release_claim__datetime.now(empty_load_release_claim__timezone.utc)
    payload = dict(issued_at=(now-empty_load_release_claim__timedelta(minutes=1)).isoformat(),
                   expires_at=(now+empty_load_release_claim__timedelta(hours=1)).isoformat())
    path = tmp_path / "approval.json"
    path.write_text(empty_load_release_claim__json.dumps(payload), encoding="utf-8")
    return empty_load_release_claim__releases.ValidatedEmptyLoadCommissioningRelease(
        payload_json=empty_load_release_claim__json.dumps(payload), approval_path=str(path),
        approval_sha256=empty_load_release_claim__hashlib.sha256(path.read_bytes()).hexdigest(),
        files=(), _token=empty_load_release_claim__releases._VALIDATED)


def test_empty_load_release_claim__concurrent_claims_have_one_winner_and_survive_reconstruction(empty_load_release_claim__release):
    def claim(_):
        try:
            return empty_load_release_claim__release.claim_execution()
        except PermissionError:
            return None

    with empty_load_release_claim__ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert sum(r is not None for r in results) == 1
    restored = empty_load_release_claim__replace(empty_load_release_claim__release)
    with empty_load_release_claim__pytest.raises(PermissionError, match="already claimed"):
        restored.require_unused()
    with empty_load_release_claim__pytest.raises(PermissionError, match="already claimed"):
        restored.claim_execution()
    # Runtime validity remains checkable by the process that owns the claim.
    restored.require_current(check_files=True)


def test_empty_load_release_claim__spawned_processes_cannot_both_claim_same_release(empty_load_release_claim__release, tmp_path):
    context = empty_load_release_claim__mp.get_context("spawn")
    results = context.Queue()
    processes = [context.Process(target=empty_load_release_claim___claim_in_spawned_process,
        args=(empty_load_release_claim__release, tmp_path, results)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert sorted(results.get(timeout=2) for _ in processes) == [False, True]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        results.close()


def test_empty_load_release_claim__failed_claim_flush_stays_consumed(empty_load_release_claim__release, monkeypatch):
    def fail_fsync(_):
        raise OSError("offline storage fault")
    monkeypatch.setattr(empty_load_release_claim__releases.os, "fsync", fail_fsync)
    with empty_load_release_claim__pytest.raises(OSError, match="storage fault"):
        empty_load_release_claim__release.claim_execution()
    with empty_load_release_claim__pytest.raises(PermissionError, match="already claimed"):
        empty_load_release_claim__release.require_unused()


def test_empty_load_release_claim__changed_approval_is_rejected_before_claim(empty_load_release_claim__release):
    from pathlib import Path
    Path(empty_load_release_claim__release.approval_path).write_text("{}", encoding="utf-8")
    with empty_load_release_claim__pytest.raises(PermissionError, match="reviewed file changed"):
        empty_load_release_claim__release.claim_execution()
    assert not empty_load_release_claim__release._claim_path().exists()


@empty_load_release_claim__pytest.mark.parametrize("already_claimed", [False, True])
def test_empty_load_release_claim__executor_claims_before_first_intent_and_never_reuses(empty_load_release_claim__release, tmp_path, monkeypatch, already_claimed):
    from collection.episode_logger import EpisodeLogger
    from control.robot_trajectory_executor import RokaeMotionExecutor
    Acquisition = robot_trajectory_executor__Acquisition
    Motion = robot_trajectory_executor__Motion
    _preflight = robot_trajectory_executor___preflight
    _safety = robot_trajectory_executor___safety
    _trajectory = robot_trajectory_executor___trajectory

    # Runtime identity checks have separate coverage; isolate the real claim
    # integration using a synthetic approval and the offline motion adapter.
    monkeypatch.setattr(empty_load_release_claim__releases.ValidatedEmptyLoadCommissioningRelease, "require_runtime", lambda *_: None)
    if already_claimed:
        empty_load_release_claim__release.claim_execution()
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    original_start = motion.start_cartesian_hold

    def start(pose):
        assert empty_load_release_claim__release._claim_path().is_file()
        assert logger.row_counts["trajectory_command"] == 1
        return original_start(pose)

    motion.start_cartesian_hold = start
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    preflight = empty_load_release_claim__replace(_preflight(), release_approval=empty_load_release_claim__release)
    try:
        if already_claimed:
            with empty_load_release_claim__pytest.raises(PermissionError, match="already claimed"):
                executor.execute(_trajectory(), preflight)
            assert not any(c[0] == "start" for c in motion.calls)
            assert logger.row_counts["trajectory_command"] == 0
        else:
            assert executor.execute(_trajectory(), preflight).completed
    finally:
        logger.close(completed=False)


def test_empty_load_release_claim__preflight_failure_does_not_consume_release(empty_load_release_claim__release, tmp_path):
    from collection.episode_logger import EpisodeLogger
    from control.robot_trajectory_executor import RokaeMotionExecutor
    Acquisition = robot_trajectory_executor__Acquisition
    Motion = robot_trajectory_executor__Motion
    _preflight = robot_trajectory_executor___preflight
    _safety = robot_trajectory_executor___safety
    _trajectory = robot_trajectory_executor___trajectory
    logger = EpisodeLogger(tmp_path / "episode").start()
    try:
        executor = RokaeMotionExecutor(Motion(), Acquisition(), logger, _safety())
        with empty_load_release_claim__pytest.raises(PermissionError, match="blocked_test"):
            executor.execute(_trajectory(), empty_load_release_claim__replace(_preflight(False), release_approval=empty_load_release_claim__release))
        assert not empty_load_release_claim__release._claim_path().exists()
    finally:
        logger.close(completed=False)


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
            fsync_entered = episode_logger__threading.Event()
            release_fsync = episode_logger__threading.Event()
            caller_done = episode_logger__threading.Event()
            errors: list[BaseException] = []

            def blocking_fsync(_descriptor: int) -> None:
                fsync_entered.set()
                if not release_fsync.wait(timeout=2.0):
                    raise TimeoutError("test did not release fsync")

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

            with episode_logger__patch("collection.episode_logger.os.fsync", side_effect=blocking_fsync):
                caller = episode_logger__threading.Thread(target=append_bounded, name="bounded-caller")
                caller.start()
                self.assertTrue(fsync_entered.wait(timeout=1.0))
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
                # This must finish while fsync still owns the logger I/O lock.
                self.assertTrue(later_done.wait(timeout=0.2))
                self.assertEqual(len(later_errors), 1)
                self.assertIsInstance(later_errors[0], episode_logger__EpisodeLoggerError)

                release_fsync.set()
                caller.join(timeout=1.0)
                later.join(timeout=1.0)
                self.assertTrue(logger.wait_for_pending_writes(1.0))

            self.assertEqual(logger.status, "failed")
            logger.close(completed=False)
            metadata = episode_logger__json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failed_stream"], "trajectory_command")

    def test_signal_failure_returns_while_fsync_holds_logger_lock(self) -> None:
        with episode_logger__tempfile.TemporaryDirectory() as root:
            episode = episode_logger__Path(root) / "episode_signal_failure"
            logger = episode_logger__EpisodeLogger(episode, clock=episode_logger___DeterministicClock()).start()
            fsync_entered = episode_logger__threading.Event()
            release_fsync = episode_logger__threading.Event()
            append_done = episode_logger__threading.Event()
            signal_done = episode_logger__threading.Event()

            def blocking_fsync(_descriptor: int) -> None:
                fsync_entered.set()
                if not release_fsync.wait(timeout=2.0):
                    raise TimeoutError("test did not release fsync")

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

            with episode_logger__patch("collection.episode_logger.os.fsync", side_effect=blocking_fsync):
                writer = episode_logger__threading.Thread(target=append_command)
                writer.start()
                self.assertTrue(fsync_entered.wait(timeout=1.0))
                signaler = episode_logger__threading.Thread(target=publish_failure)
                signaler.start()
                self.assertTrue(signal_done.wait(timeout=0.2))
                self.assertTrue(logger.failure_event.is_set())
                self.assertFalse(logger.healthy_signal)
                release_fsync.set()
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
# Source: tests/test_execution_preflight.py
# ============================================================================

"""Fail-closed first-trial execution gate tests."""



from dataclasses import replace as execution_preflight__replace
from pathlib import Path as execution_preflight__Path

import pytest as execution_preflight__pytest

from collection.real_robot_acquisition import AcquisitionHealth as execution_preflight__AcquisitionHealth
import control.execution_preflight as execution_preflight__preflight_module
from control.execution_preflight import (
    OPERATOR_CONFIRMATION as execution_preflight__OPERATOR_CONFIRMATION,
    evaluate_execution_preflight as execution_preflight__evaluate_execution_preflight,
    evaluate_offline_execution_request as execution_preflight__evaluate_offline_execution_request,
)
from control.start_anchor import FixedTcpOrientation as execution_preflight__FixedTcpOrientation, StartAnchor as execution_preflight__StartAnchor
from control.start_anchored_relative_trajectory import (
    RehabFrameConfig as execution_preflight__RehabFrameConfig,
    build_start_anchored_relative_trajectory as execution_preflight__build_start_anchored_relative_trajectory,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH as execution_preflight__DEFAULT_REFERENCE_PATH
from safety.experiment_safety import ExperimentSafetyConfig as execution_preflight__ExperimentSafetyConfig


execution_preflight__SOURCE_CANDIDATE_DIRECTORY = (
    artifact_path("lower_limb_sim/data/reference_candidates")
)


class execution_preflight__ReadyLogger:
    ready = True
    healthy = True


class execution_preflight__ConnectedAdapter:
    def is_connected(self):
        return True

    def get_robot_state_summary(self):
        return {
            "connected": True,
            "joint_position_rad": [0.0] * 6,
            "collision_state": False,
            "collision_state_query_valid": True,
            "robot_metadata": {
                "robot_model": "xMate6-Pro",
                "robot_serial_number": "SN-PREFLIGHT",
                "controller_version": "controller-test",
                "joint_soft_limits_rad": [[-2.0, 2.0]] * 6,
                "sdk_tool_payload": {
                    "toolset_load_mass_kg": 1.25,
                    "toolset_load_cog_m": [0.0, 0.0, 0.08],
                    "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
                    "sdk_available_tool_names": ["rehab_cuff"],
                    "sdk_available_workobject_names": ["rehab_fixture"],
                    "active_hmi_tool_workobject_verified": False,
                },
            },
        }


def execution_preflight___inputs():
    frame = execution_preflight__RehabFrameConfig((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), True, "reviewed test")
    pose = (0.4, 0.0, 0.3, 0.1, 0.2, 0.3)
    trajectory, audit, _ = execution_preflight__build_start_anchored_relative_trajectory(
        current_tcp_start_pose=pose,
        rehab_frame=frame,
    )
    first = trajectory.iloc[0]
    anchor = execution_preflight__StartAnchor(
        capture_host_time_s=1.0,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=execution_preflight__FixedTcpOrientation(pose[3:]),
        robot_joint_positions=(0.0,) * 6,
        trajectory_id=audit.trajectory_id,
        reference_start_q_hip=float(first["q_hip_ref"]),
        reference_start_q_knee=float(first["q_knee_ref"]),
        anchor_id="anchor_reviewed",
        robot_model="xMate6-Pro",
        robot_serial_number="SN-PREFLIGHT",
        controller_version="controller-test",
        tool_name="rehab_cuff",
        workpiece_name="rehab_fixture",
        reviewed=True,
        notes="synthetic test",
    )
    safety = execution_preflight__ExperimentSafetyConfig(
        max_tcp_speed_m_s=10.0,
        max_tcp_acceleration_m_s2=10.0,
        max_start_anchor_position_error_m=0.01,
        max_start_anchor_orientation_error_rad=0.01,
        max_command_lateness_s=0.1,
        max_force_n=100.0,
        max_torque_nm=100.0,
        max_state_age_s=1.0,
        max_wrench_age_s=1.0,
        max_state_wrench_skew_s=1.0,
        workspace_min_base_m=(-2.0, -2.0, -2.0),
        workspace_max_base_m=(2.0, 2.0, 2.0),
        expected_robot_model="xMate6-Pro",
        expected_robot_serial_number="SN-PREFLIGHT",
        expected_controller_version="controller-test",
        reviewed_tool_name="rehab_cuff",
        reviewed_workpiece_name="rehab_fixture",
        reviewed_payload_mass_kg=1.25,
        reviewed_payload_cog_m=(0.0, 0.0, 0.08),
        reviewed_payload_inertia_kg_m2=(0.01, 0.02, 0.03),
        reviewed_joint_soft_limits_rad=tuple((-2.0, 2.0) for _ in range(6)),
        reviewed_rt_filter_hz=25.0,
        reviewed_rt_network_tolerance_percent=20.0,
        robot_identity_reviewed=True,
        tool_workpiece_reviewed=True,
        payload_configuration_reviewed=True,
        collision_configuration_reviewed=True,
        joint_soft_limits_reviewed=True,
        realtime_configuration_reviewed=True,
        reviewed=True,
        notes="synthetic test values only",
    )
    health = execution_preflight__AcquisitionHealth(
        host_time_s=10.0,
        state_time_s=9.99,
        wrench_time_s=9.98,
        state_age_s=0.01,
        wrench_age_s=0.02,
        state_wrench_skew_s=0.01,
        state_valid=True,
        wrench_valid=True,
        state_thread_alive=True,
        wrench_thread_alive=True,
        alignment_thread_alive=True,
        query_duration_ms=2.0,
        force_magnitude_n=5.0,
        torque_magnitude_nm=1.0,
        valid=True,
        invalid_reason="",
    )
    return frame, trajectory, audit, anchor, safety, health


def execution_preflight___evaluate(**overrides):
    frame, trajectory, audit, anchor, safety, health = execution_preflight___inputs()
    values = dict(
        mode="execute",
        enable_motion=True,
        operator_confirmation=execution_preflight__OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id,
        frame=frame,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
        audit=audit,
        acquisition_health=health,
        logger=execution_preflight__ReadyLogger(),
        robot_adapter=execution_preflight__ConnectedAdapter(),
        current_tcp_pose_base=anchor.tcp_pose_base,
    )
    values.update(overrides)
    return execution_preflight__evaluate_execution_preflight(**values)


def test_execution_preflight__reference_freeze_keeps_even_reviewed_slow_request_no_go():
    result = execution_preflight___evaluate()
    assert not result.allowed
    assert result.reasons == (
        "reference_release_not_approved_for_first_robot_trial",
    )
    assert result.evaluation_phase == "live"
    assert result.trajectory_sha256


def test_execution_preflight__callback_state_can_satisfy_collision_gate_but_not_reference_release():
    from hardware.safety_events import SafetyEventMonitor

    monitor = SafetyEventMonitor(1.0)
    generation = monitor.begin()
    monitor.registered(generation)

    class EventAdapter(execution_preflight__ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            status = monitor.snapshot()
            summary.update(collision_state=status.collision_state,
                           collision_state_query_valid=status.valid,
                           collision_state_source="events")
            return summary

    adapter = EventAdapter()
    assert "runtime_collision_state_unavailable" in execution_preflight___evaluate(robot_adapter=adapter).reasons
    monitor.receive(generation, {"collided": False})
    assert execution_preflight___evaluate(robot_adapter=adapter).reasons == (
        "reference_release_not_approved_for_first_robot_trial",
    )
    monitor.receive(generation, {"collided": True})
    assert "runtime_collision_detected" in execution_preflight___evaluate(robot_adapter=adapter).reasons
    monitor.invalidate("disconnected")
    assert "runtime_collision_state_unavailable" in execution_preflight___evaluate(robot_adapter=adapter).reasons


def test_execution_preflight__execute_requires_explicit_enable_and_operator_confirmation():
    assert "enable_motion_flag_missing" in execution_preflight___evaluate(enable_motion=False).reasons
    assert "operator_confirmation_missing" in execution_preflight___evaluate(operator_confirmation="yes").reasons


def test_execution_preflight__unreviewed_frame_anchor_and_safety_are_each_blocking():
    frame, _, _, anchor, safety, _ = execution_preflight___inputs()
    assert "rehab_frame_not_reviewed" in execution_preflight___evaluate(frame=execution_preflight__replace(frame, reviewed=False)).reasons
    assert "start_anchor_not_reviewed" in execution_preflight___evaluate(anchor=execution_preflight__replace(anchor, reviewed=False)).reasons
    result = execution_preflight___evaluate(safety=execution_preflight__replace(safety, reviewed=False))
    assert "experiment_safety_not_reviewed" in result.reasons


def test_execution_preflight__nominal_reference_is_not_first_trial_whitelisted():
    frame, _, _, anchor, safety, health = execution_preflight___inputs()
    nominal = execution_preflight__SOURCE_CANDIDATE_DIRECTORY / "reference_measured_asymmetric_closed_nominal.csv"
    trajectory, audit, _ = execution_preflight__build_start_anchored_relative_trajectory(
        nominal,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    result = execution_preflight___evaluate(
        trajectory=trajectory,
        audit=audit,
        anchor=execution_preflight__replace(anchor, trajectory_id=audit.trajectory_id),
        safety=safety,
        acquisition_health=health,
    )
    assert "trajectory_not_whitelisted_for_first_robot_trial" in result.reasons
    assert "trajectory_audit_invalid" in result.reasons
    assert "trajectory_rows_invalid" in result.reasons


def test_execution_preflight__stale_or_dead_stream_is_blocking():
    *_, health = execution_preflight___inputs()
    result = execution_preflight___evaluate(
        acquisition_health=execution_preflight__replace(
            health,
            state_age_s=2.0,
            state_thread_alive=False,
            valid=False,
        )
    )
    assert "state_stale" in result.reasons
    assert "state_thread_not_alive" in result.reasons


def test_execution_preflight__robot_must_already_be_at_reviewed_anchor():
    *_, anchor, _, _ = execution_preflight___inputs()
    displaced = list(anchor.tcp_pose_base)
    displaced[0] += 0.1
    result = execution_preflight___evaluate(current_tcp_pose_base=displaced)
    assert "robot_not_at_reviewed_start_anchor_position" in result.reasons


def test_execution_preflight__anchor_reference_start_joint_values_are_bound_to_active_first_row():
    *_, anchor, _, _ = execution_preflight___inputs()
    result = execution_preflight___evaluate(
        anchor=execution_preflight__replace(anchor, reference_start_q_hip=anchor.reference_start_q_hip + 0.01)
    )
    assert "anchor_reference_start_joint_mismatch" in result.reasons


def test_execution_preflight__runtime_robot_identity_must_match_anchor_and_reviewed_config():
    class WrongSerial(execution_preflight__ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["robot_metadata"]["robot_serial_number"] = "SN-WRONG"
            return summary

    result = execution_preflight___evaluate(robot_adapter=WrongSerial())
    assert "runtime_robot_serial_number_mismatch" in result.reasons


def test_execution_preflight__declared_tool_and_workpiece_must_match_anchor_and_sdk_lists():
    *_, anchor, _, _ = execution_preflight___inputs()
    result = execution_preflight___evaluate(anchor=execution_preflight__replace(anchor, tool_name="other_tool"))
    assert "anchor_tool_name_mismatch" in result.reasons

    class MissingWorkpiece(execution_preflight__ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["robot_metadata"]["sdk_tool_payload"][
                "sdk_available_workobject_names"
            ] = []
            return summary

    result = execution_preflight___evaluate(robot_adapter=MissingWorkpiece())
    assert "reviewed_workpiece_name_not_reported_by_sdk" in result.reasons


def test_execution_preflight__payload_collision_and_soft_limits_fail_closed():
    class UnsafeSummary(execution_preflight__ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["collision_state"] = True
            summary["robot_metadata"]["sdk_tool_payload"][
                "toolset_load_mass_kg"
            ] = 2.0
            summary["robot_metadata"]["joint_soft_limits_rad"][0] = [-1.0, 1.0]
            return summary

    reasons = execution_preflight___evaluate(robot_adapter=UnsafeSummary()).reasons
    assert "runtime_collision_detected" in reasons
    assert "runtime_payload_mass_mismatch" in reasons
    assert "runtime_joint_soft_limits_mismatch" in reasons


def test_execution_preflight__unavailable_collision_query_and_current_joint_outside_limits_block():
    class MissingCollision(execution_preflight__ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["collision_state"] = None
            summary["collision_state_query_valid"] = False
            summary["joint_position_rad"][2] = 3.0
            return summary

    reasons = execution_preflight___evaluate(robot_adapter=MissingCollision()).reasons
    assert "runtime_collision_state_unavailable" in reasons
    assert "runtime_joint_position_outside_soft_limits" in reasons


def test_execution_preflight__row_validity_requires_real_booleans_and_motion_derivatives_are_recomputed():
    _, trajectory, _, _, _, _ = execution_preflight___inputs()
    string_flags = trajectory.copy(deep=True)
    string_flags["trajectory_valid"] = "True"
    reasons = execution_preflight___evaluate(trajectory=string_flags).reasons
    assert "trajectory_valid_column_not_strict_boolean" in reasons

    falsified_derivatives = trajectory.copy(deep=True)
    falsified_derivatives[
        ["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]
    ] = 0.0
    falsified_derivatives[
        ["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]
    ] = 0.0
    reasons = execution_preflight___evaluate(trajectory=falsified_derivatives).reasons
    assert "trajectory_velocity_columns_mismatch" in reasons
    assert "trajectory_acceleration_columns_mismatch" in reasons


def test_execution_preflight__offline_preflight_is_explicitly_unbound_from_executor():
    frame, trajectory, audit, anchor, safety, _ = execution_preflight___inputs()
    result = execution_preflight__evaluate_offline_execution_request(
        mode="execute",
        enable_motion=True,
        operator_confirmation=execution_preflight__OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id,
        frame=frame,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
        audit=audit,
    )
    assert not result.allowed
    assert "reference_release_not_approved_for_first_robot_trial" in result.reasons
    assert result.evaluation_phase == "offline"
    with execution_preflight__pytest.raises(PermissionError, match="not_live_bound"):
        result.require_live_trajectory_binding(trajectory)


def test_execution_preflight__first_trial_reference_hash_is_pinned_not_only_compared_to_same_file(
    monkeypatch,
):
    original_loader = execution_preflight__preflight_module.load_closed_reference_trajectory

    def wrong_hash_loader(path):
        frame, metadata = original_loader(path)
        return frame, {**metadata, "sha256": "0" * 64}

    monkeypatch.setattr(
        execution_preflight__preflight_module,
        "load_closed_reference_trajectory",
        wrong_hash_loader,
    )
    assert "trajectory_not_exact_official_slow_reference" in execution_preflight___evaluate().reasons


# ============================================================================
# Source: tests/test_identify_real_episode.py
# ============================================================================

"""Real identification interface must never fabricate missing evidence."""



import json as identify_real_episode__json

import pandas as identify_real_episode__pd
import pytest as identify_real_episode__pytest

from scripts.identify_real_episode import identify_real_episode as identify_real_episode__identify_real_episode


def identify_real_episode___empty_episode(path):
    path.mkdir()
    (path / "metadata.json").write_text("{}\n", encoding="utf-8")
    identify_real_episode__pd.DataFrame(
        columns=[
            "host_time_s", "tcp_x", "tcp_y", "tcp_z", "valid", "invalid_reason"
        ]
    ).to_csv(path / "robot_state.csv", index=False)
    identify_real_episode__pd.DataFrame(
        columns=[
            "query_start_s", "query_end_s", "fx", "fy", "fz", "frame_type", "valid"
        ]
    ).to_csv(path / "robot_wrench.csv", index=False)


def test_identify_real_episode__missing_reviewed_config_creates_no_fake_outputs(tmp_path):
    episode = tmp_path / "episode"
    identify_real_episode___empty_episode(episode)
    with identify_real_episode__pytest.raises(FileNotFoundError, match="reviewed real identification config"):
        identify_real_episode__identify_real_episode(episode)
    assert not (episode / "identified_parameters.json").exists()
    assert not (episode / "prediction_metrics.csv").exists()


def test_identify_real_episode__unreviewed_template_is_rejected_before_reading_empty_data(tmp_path):
    episode = tmp_path / "episode"
    identify_real_episode___empty_episode(episode)
    template = {
        "schema_version": 1,
        "reviewed": False,
        "raw_wrench_frame": None,
        "R_rehab_from_raw_wrench": None,
        "force_sign_robot_on_leg": None,
        "assumed_wrench_delay_s": None,
        "baseline_subject_template": {
            "mass_thigh_kg": None,
            "mass_shank_kg": None,
            "com_thigh_m": None,
            "com_shank_m": None,
            "inertia_thigh_kg_m2": None,
            "inertia_shank_kg_m2": None,
            "q0_hip_rad": None,
            "q0_knee_rad": None,
            "gravity_m_s2": None,
        },
        "notes": "draft",
    }
    config = episode / "identification_config.json"
    config.write_text(identify_real_episode__json.dumps(template), encoding="utf-8")
    with identify_real_episode__pytest.raises(PermissionError, match="not reviewed"):
        identify_real_episode__identify_real_episode(episode)
    assert not (episode / "identified_parameters.json").exists()
    assert not (episode / "prediction_metrics.csv").exists()




# ============================================================================
# Source: tests/test_lab_session.py
# ============================================================================

from copy import deepcopy as lab_session__deepcopy
from datetime import datetime as lab_session__datetime, timedelta as lab_session__timedelta, timezone as lab_session__timezone
import json as lab_session__json
from pathlib import Path as lab_session__Path
from types import SimpleNamespace as lab_session__SimpleNamespace

import pytest as lab_session__pytest

import scripts.lab_session as lab_session__lab
from scripts.prepare_empty_load_commissioning_release import bounded_expiry as lab_session__bounded_expiry


@lab_session__pytest.fixture
def lab_session__profile():
    return lab_session__deepcopy(lab_session__lab.load(section_path("lab_next_visit")))


@lab_session__pytest.mark.parametrize("action", ["diagnostics", "events", "capture", "observe", "empty", "round"])
def test_lab_session__device_actions_require_explicit_live_before_subprocess(tmp_path, lab_session__profile, action):
    with lab_session__pytest.raises(PermissionError, match="--live"):
        lab_session__lab.execute(action, lab_session__profile, tmp_path, runner=lambda *a, **k: lab_session__pytest.fail("device command"))


def test_lab_session__diagnostics_retains_independent_failures_without_retry(tmp_path, lab_session__profile):
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        assert "--enable-motion" not in command
        assert "--prepare-realtime" not in command
        assert "--operator-confirmation" not in command
        return lab_session__SimpleNamespace(returncode=1 if command[4] == "scripts.diagnose_collision_events" else 0)
    result = lab_session__lab.execute("diagnostics", lab_session__profile, tmp_path, live=True, runner=runner)
    assert len(commands) == 3
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert [s["returncode"] for s in result["steps"]] == [1, 0, 0]
    assert all((tmp_path / name / "session_status.json").exists() for name in ("events", "capture", "observe"))


def test_lab_session__missing_empty_release_never_reaches_motion_or_confirmation(tmp_path, lab_session__profile):
    result = lab_session__lab.execute("empty", lab_session__profile, tmp_path, live=True,
        runner=lambda *a, **k: lab_session__pytest.fail("motion command"),
        confirm=lambda *_: lab_session__pytest.fail("invalid release should fail first"))
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert "required path" in result["reason"]


def test_lab_session__empty_precheck_failure_prevents_motion(tmp_path, lab_session__profile, monkeypatch):
    monkeypatch.setattr(lab_session__lab, "empty_release", lambda _: lab_session__SimpleNamespace(approval_sha256="test", payload={}))
    lab_session__profile["empty_load"]["release"] = "test-release.json"
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return lab_session__SimpleNamespace(returncode=1)
    result = lab_session__lab.execute("empty", lab_session__profile, tmp_path, live=True, runner=runner,
                         confirm=lambda *_: lab_session__pytest.fail("precheck did not pass"))
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert len(calls) == 1 and "scripts.check_empty_load_commissioning_release" in calls[0]


@lab_session__pytest.mark.parametrize("seconds,allowed", [(600, True), (1800, True), (1801, False)])
def test_lab_session__empty_release_lifetime_is_enforced(lab_session__profile, monkeypatch, seconds, allowed):
    import safety.empty_load_commissioning_release as releases
    now = lab_session__datetime.now(lab_session__timezone.utc)
    payload = dict(issued_at=now.isoformat(), expires_at=(now+lab_session__timedelta(seconds=seconds)).isoformat(),
                   single_cycle=True, training_allowed=False)
    monkeypatch.setattr(releases, "load_and_validate_empty_load_commissioning_release",
                        lambda *a, **k: lab_session__SimpleNamespace(payload=payload))
    lab_session__profile["empty_load"]["release"] = "test-release.json"
    if allowed:
        assert lab_session__lab.empty_release(lab_session__profile).payload == payload
    else:
        with lab_session__pytest.raises(PermissionError, match="lifetime"):
            lab_session__lab.empty_release(lab_session__profile)


@lab_session__pytest.mark.parametrize("confirmed", [False, True])
def test_lab_session__empty_uses_only_existing_release_after_precheck_and_manual_confirmation(tmp_path, lab_session__profile, monkeypatch, confirmed):
    from control import start_anchor
    lab_session__profile["empty_load"]["release"] = "test-release.json"
    payload = dict(expires_at="future-for-test-only", wrench_hz=50)
    monkeypatch.setattr(lab_session__lab, "empty_release", lambda _: lab_session__SimpleNamespace(approval_sha256="test", payload=payload))
    monkeypatch.setattr(start_anchor, "load_start_anchor", lambda _: lab_session__SimpleNamespace(anchor_id="test-anchor"))
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return lab_session__SimpleNamespace(returncode=0)
    result = lab_session__lab.execute("empty", lab_session__profile, tmp_path, live=True, runner=runner,
        confirm=lambda _: "I CONFIRM EMPTY LOAD ONE CYCLE" if confirmed else "cancel")
    assert len(calls) == (2 if confirmed else 1)
    assert result["status"] == ("COMPLETE" if confirmed else "BLOCKED_OR_FAILED")
    if confirmed:
        command = calls[1]
        assert command[4] == "scripts.run_rehab_experiment"
        assert "--prepare-realtime" in command and "--enable-motion" in command
        assert command[command.index("--collision-source")+1] == "query"
        assert "scripts.train_robot_leg_round" not in command


def test_lab_session__capture_only_config_cannot_masquerade_as_training(tmp_path, lab_session__profile, monkeypatch):
    import scripts.run_robot_leg_round as rounds
    monkeypatch.setattr(rounds, "check_config", lambda _: [])
    monkeypatch.setattr(rounds, "paths", lambda _: dict(trial_kind="dummy_leg_capture_only"))
    result = lab_session__lab.execute("round", lab_session__profile, tmp_path, live=True,
                         runner=lambda *a, **k: lab_session__pytest.fail("motion command"))
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert result["blockers"] == ["selected configuration does not enable real training"]


@lab_session__pytest.mark.parametrize("confirmed", [False, True])
def test_lab_session__complete_round_requires_preview_confirmation_and_real_update(tmp_path, lab_session__profile, monkeypatch, confirmed):
    import scripts.run_robot_leg_round as rounds
    monkeypatch.setattr(rounds, "check_config", lambda _: [])
    monkeypatch.setattr(rounds, "paths", lambda _: dict(trial_kind="dummy_leg_reference"))
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        directory = lab_session__Path(command[command.index("--session") + 1])
        directory.mkdir(exist_ok=True)
        if "run" in command:
            lab_session__lab.write(directory / "round_status.json", dict(status="ROUND_COMPLETE", training_status="MODEL_UPDATED"))
        return lab_session__SimpleNamespace(returncode=0)
    result = lab_session__lab.execute("round", lab_session__profile, tmp_path, live=True, runner=runner,
        confirm=lambda _: rounds.CONFIRMATION if confirmed else "cancel")
    assert len(calls) == (2 if confirmed else 1)
    assert result["status"] == ("COMPLETE" if confirmed else "BLOCKED_OR_FAILED")
    if confirmed:
        assert result["round_report"].endswith("report.md")


def test_lab_session__offline_training_rejects_incomplete_round(tmp_path, lab_session__profile):
    original = tmp_path / "original"
    original.mkdir()
    lab_session__lab.write(original / "round_status.json", dict(status="ROUND_FAILED", training_status="NOT_UPDATED"))
    result = lab_session__lab.execute("train", lab_session__profile, tmp_path, session=original,
        runner=lambda *a, **k: lab_session__pytest.fail("must not train failed data"))
    assert result["status"] == "BLOCKED_OR_FAILED"


def test_lab_session__initialization_has_no_measurements_and_no_device_command(tmp_path, lab_session__profile, monkeypatch):
    import scripts.run_robot_leg_round as rounds
    config = rounds.paths(lab_session__lab.path(lab_session__profile["round_config"]))
    reference = artifact_path("reference_release/reference_measured_asymmetric_closed_slow.csv")
    config["candidates"][0]["path"] = str(reference)
    monkeypatch.setattr(rounds, "paths", lambda _: config)
    def runner(command, **kwargs):
        assert "--validate-only" in command
        assert command[4] == "scripts.train_robot_leg_round"
        lab_session__lab.write(lab_session__Path(command[command.index("--output")+1]), dict(status="ALGORITHM_INITIALIZED"))
        return lab_session__SimpleNamespace(returncode=0)
    result = lab_session__lab.execute("initialize", lab_session__profile, tmp_path, runner=runner)
    assert result["status"] == "COMPLETE"
    assert lab_session__lab.load(tmp_path / "initialization_request.json")["history"] == []


@lab_session__pytest.mark.parametrize("minutes", [0, -1, 31, float("nan"), float("inf")])
def test_lab_session__invalid_short_release_lifetime(minutes):
    now = lab_session__datetime.now(lab_session__timezone.utc)
    with lab_session__pytest.raises(ValueError):
        lab_session__bounded_expiry((now+lab_session__timedelta(days=1)).isoformat(), minutes, now)


def test_lab_session__release_expiry_never_extends_reviewed_period():
    now = lab_session__datetime.now(lab_session__timezone.utc)
    assert lab_session__bounded_expiry((now+lab_session__timedelta(days=1)).isoformat(), 30, now) == now+lab_session__timedelta(minutes=30)
    assert lab_session__bounded_expiry((now+lab_session__timedelta(minutes=5)).isoformat(), 30, now) == now+lab_session__timedelta(minutes=5)
    with lab_session__pytest.raises(PermissionError):
        lab_session__bounded_expiry((now-lab_session__timedelta(seconds=1)).isoformat(), 30, now)


def test_lab_session__release_requires_explicit_confirmation_not_replayed_request(tmp_path, lab_session__profile):
    result = lab_session__lab.execute("release", lab_session__profile, tmp_path,
        runner=lambda *a, **k: lab_session__pytest.fail("unconfirmed approval generated"), confirm=lambda _: "")
    assert result["status"] == "BLOCKED_OR_FAILED"


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
from scripts.import_dummy_leg_episode import main as leg_episode_import__import_main


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
# Source: tests/test_loaded_round_lifecycle.py
# ============================================================================

"""SYNTHETIC HARDWARE ONLY. No SDK imports, connection or laboratory evidence."""
from dataclasses import replace as loaded_round_lifecycle__replace
from pathlib import Path as loaded_round_lifecycle__Path
import subprocess as loaded_round_lifecycle__subprocess
import time as loaded_round_lifecycle__time
from types import SimpleNamespace as loaded_round_lifecycle__SimpleNamespace

import pandas as loaded_round_lifecycle__pd
import pytest as loaded_round_lifecycle__pytest

from collection.twincat_leg import atomic_json as loaded_round_lifecycle__atomic_json
from control.robot_trajectory_executor import RokaeMotionExecutor as loaded_round_lifecycle__RokaeMotionExecutor
from safety.robot_trial_release import read_json as loaded_round_lifecycle__read_json, sha256_file as loaded_round_lifecycle__sha256_file
loaded_round_lifecycle__prepared_round = robot_leg_round__prepared_round
loaded_round_lifecycle__SyntheticCapture = robot_leg_round__SyntheticCapture
loaded_round_lifecycle__BaseAcquisition = robot_trajectory_executor__Acquisition
loaded_round_lifecycle__Motion = robot_trajectory_executor__Motion
import scripts.run_rehab_experiment as loaded_round_lifecycle__runner
import scripts.run_robot_leg_round as loaded_round_lifecycle__rounds


def loaded_round_lifecycle__install_hardware(monkeypatch, directory, fault=None):
    calls = []
    capture_instances = []
    anchor = loaded_round_lifecycle__read_json(directory / "anchor.json")["tcp_pose_base"]

    class Native:
        def get_robot_mode(self):
            return "MOVING" if fault == "busy" else "IDLE"
        def enable_realtime(self, *, load):
            calls.append("prepare_rt")
            assert load == 0.0
            if fault == "prepare":
                raise RuntimeError("SYNTHETIC_PARTIAL_PREPARATION_FAILURE")
        def stop_realtime(self, *, switch_to_nrt):
            calls.append("stop_rt")
        def disable(self):
            calls.append("power_off")
            if fault == "cleanup":
                raise RuntimeError("SYNTHETIC_POWER_OFF_FAILURE")

    summary = loaded_round_lifecycle__BaseAcquisition().summary
    summary["robot_metadata"].update(robot_serial_number="SN-RUNNER", xcore_sdk_version="0.7.0")
    summary.update(collision_state_source="events", safety_event_status=dict(max_age_s=.5))
    if fault in ("unknown_collision", "collision", "259"):
        summary.update(collision_state_query_valid=fault == "collision",
                       collision_state=True if fault == "collision" else None,
                       collision_state_invalid_reason="259" if fault == "259" else "unknown")

    class Adapter:
        native_robot = Native()
        def connect(self): calls.append("connect")
        def disconnect(self): calls.append("disconnect")
        def is_connected(self): return True
        def get_robot_state_summary(self): return summary
        def read_tcp_pose(self): return anchor

    class Acquisition(loaded_round_lifecycle__BaseAcquisition):
        live_producer_names = ()
        def __init__(self, adapter, logger, **kwargs):
            super().__init__()
            self.adapter = adapter
        def start(self, *, manage_connection):
            assert manage_connection is False
            calls.append("streams_start")
        def wait_until_healthy(self, **kwargs): return True
        def latest_state_frame(self):
            return loaded_round_lifecycle__SimpleNamespace(valid=True, tcp_position_m=anchor[:3], tcp_orientation_rad=anchor[3:])
        def latest_health(self):
            if "hold" in calls and fault == "disconnect":
                return loaded_round_lifecycle__replace(self.health, valid=False, invalid_reason="SYNTHETIC_DISCONNECT")
            if "hold" in calls and fault == "query_hang":
                return loaded_round_lifecycle__replace(self.health, valid=False, invalid_reason="query_pending_too_long")
            return self.health
        def stop(self): calls.append("streams_stop")

    class FakeMotion(loaded_round_lifecycle__Motion):
        def start_cartesian_hold(self, pose):
            calls.append("hold")
            if fault == "cancel":
                raise KeyboardInterrupt("SYNTHETIC_OPERATOR_CANCEL")
            super().start_cartesian_hold(pose)
        def send_cartesian_target(self, pose):
            calls.append("send")
            super().send_cartesian_target(pose)
        def request_stop(self, reason):
            calls.append("motion_stop")
            super().request_stop(reason)

    class VirtualExecutor(loaded_round_lifecycle__RokaeMotionExecutor):
        """Only scheduler waiting is virtual; real release and gates still run."""
        def __init__(self, *args, **kwargs):
            self.test_time = loaded_round_lifecycle__time.perf_counter()
            super().__init__(*args, clock=loaded_round_lifecycle__SimpleNamespace(now_s=lambda: self.test_time), **kwargs)
        def _wait_until(self, deadline_s):
            self.test_time = deadline_s
            reasons = self._runtime_reasons()
            if reasons:
                raise RuntimeError("runtime safety gate: " + ";".join(reasons))

    class Capture(loaded_round_lifecycle__SyntheticCapture):
        stopped = False
        def __init__(self, *args):
            super().__init__(*args)
            capture_instances.append(self)
        def start(self):
            super().start()
            self.raw_sha256 = loaded_round_lifecycle__sha256_file(self.directory / "leg.csv")
        def stop(self):
            self.stopped = True
            calls.append("leg_stop")

    monkeypatch.setattr(loaded_round_lifecycle__runner, "RealRobotAcquisition", Acquisition)
    monkeypatch.setattr(loaded_round_lifecycle__runner, "RokaeMotionExecutor", VirtualExecutor)
    monkeypatch.setattr(loaded_round_lifecycle__runner, "build_live_wrench_provider", lambda *a, **k: object())
    monkeypatch.setattr(loaded_round_lifecycle__rounds, "_configured_adapter_factory", lambda **k: lambda ip: Adapter())
    if fault == "disk":
        def disk_failure(self, **kwargs):
            raise OSError("SYNTHETIC_DISK_FULL")
        monkeypatch.setattr(loaded_round_lifecycle__runner.EpisodeLogger, "append_trajectory_command_bounded", disk_failure)

    def execute(**kwargs):
        assert kwargs["prepare_realtime"] is True
        original_observer = kwargs["dispatch_observer"]
        synthetic_dispatch = loaded_round_lifecycle__pd.read_csv(directory / "leg/dispatch.csv").itertuples()
        def observer(t, start, end):
            # Virtual time is not a hardware alignment measurement.
            row = next(synthetic_dispatch)
            assert t == loaded_round_lifecycle__pytest.approx(row.trajectory_time_s)
            original_observer(t, row.dispatch_start_ns, row.dispatch_end_ns)
        kwargs["dispatch_observer"] = observer
        return loaded_round_lifecycle__runner.run_execute(**kwargs, motion_factory=lambda adapter: FakeMotion())

    return Capture, execute, calls, capture_instances


@loaded_round_lifecycle__pytest.mark.parametrize("fault", [None, "disconnect", "query_hang", "disk", "cancel", "training_timeout",
                                  "prepare", "cleanup", "unknown_collision", "collision", "259", "busy"])
def test_loaded_round_lifecycle__full_loaded_round_lifecycle(tmp_path, monkeypatch, fault):
    directory = loaded_round_lifecycle__prepared_round(tmp_path)
    state = loaded_round_lifecycle__read_json(directory / "round_status.json")
    state["software_verification_only"] = True
    loaded_round_lifecycle__rounds.report(directory, state)
    Capture, execute, calls, captures = loaded_round_lifecycle__install_hardware(monkeypatch, directory, fault)
    real_worker = loaded_round_lifecycle__rounds.run_algorithm_worker
    def worker(*args, **kwargs):
        if not kwargs.get("validate_only"):
            assert calls.index("motion_stop") < calls.index("stop_rt") < calls.index("power_off") < calls.index("disconnect")
            assert captures[0].stopped
            calls.append("train")
            if fault == "training_timeout":
                # Exercise subprocess.run's actual timeout/kill/wait path.
                return real_worker(*args[:3], .001, **kwargs)
        return real_worker(*args, **kwargs)
    monkeypatch.setattr(loaded_round_lifecycle__rounds, "run_algorithm_worker", worker)
    if fault is None:
        result = loaded_round_lifecycle__rounds.run(directory, confirmation=loaded_round_lifecycle__rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
        assert result["status"] == "ROUND_COMPLETE"
        assert result["training_status"] == "MODEL_UPDATED"
    else:
        with loaded_round_lifecycle__pytest.raises((Exception, KeyboardInterrupt)):
            loaded_round_lifecycle__rounds.run(directory, confirmation=loaded_round_lifecycle__rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
        result = loaded_round_lifecycle__read_json(directory / "round_status.json")
        assert result["status"] == "ROUND_FAILED"
        assert result["training_status"] == "NOT_UPDATED"
        assert not (directory / "model_update.json").exists()
        assert ("train" in calls) == (fault == "training_timeout")
    assert captures[0].stopped
    assert calls.count("connect") == 1 and calls.count("disconnect") == 1
    assert calls.index("stop_rt") < calls.index("power_off") < calls.index("disconnect")
    assert (directory / "leg/leg.csv").is_file()
    assert loaded_round_lifecycle__sha256_file(directory / "leg/leg.csv") == captures[0].raw_sha256
    assert "SYNTHETIC HARDWARE" in (directory / "report.md").read_text(encoding="utf-8")
    if fault in ("prepare", "unknown_collision", "collision", "259", "busy"):
        assert "hold" not in calls
    if fault in ("unknown_collision", "collision", "259", "busy"):
        assert "prepare_rt" not in calls
    if fault in ("disconnect", "query_hang", "disk", "cancel"):
        assert "motion_stop" in calls and "send" not in calls


def test_loaded_round_lifecycle__capture_cleanup_precedes_failure_reporting_when_storage_fails(tmp_path, monkeypatch, capsys):
    directory = loaded_round_lifecycle__prepared_round(tmp_path, capture_only=True)
    Capture, execute, calls, captures = loaded_round_lifecycle__install_hardware(monkeypatch, directory, "disconnect")
    original_report = loaded_round_lifecycle__rounds.report
    original_atomic = loaded_round_lifecycle__rounds.atomic_json
    def report_with_failed_storage(path, state):
        if state["status"] == "ROUND_FAILED":
            assert captures[0].stopped
            def unavailable(*args, **kwargs):
                raise OSError("SYNTHETIC_STATUS_DISK_FAILURE")
            monkeypatch.setattr(loaded_round_lifecycle__rounds, "atomic_json", unavailable)
        try:
            return original_report(path, state)
        finally:
            monkeypatch.setattr(loaded_round_lifecycle__rounds, "atomic_json", original_atomic)
    monkeypatch.setattr(loaded_round_lifecycle__rounds, "report", report_with_failed_storage)
    with loaded_round_lifecycle__pytest.raises(RuntimeError, match="SYNTHETIC_DISCONNECT") as caught:
        loaded_round_lifecycle__rounds.run(directory, confirmation=loaded_round_lifecycle__rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
    assert "disconnect" in calls and captures[0].stopped
    assert (directory / "leg/leg.csv").is_file()
    assert "ROUND_FAILED" in (directory / "report.md").read_text(encoding="utf-8")
    assert "ROUND_REPORT_WRITE_FAILED" in capsys.readouterr().err
    assert any("could not be fully saved" in note for note in caught.value.__notes__)


def test_loaded_round_lifecycle__old_loaded_config_does_not_authorize_automatic_power_preparation(tmp_path):
    loaded_round_lifecycle__prepared_round(tmp_path, capture_only=True)
    p = tmp_path / "round.json"
    c = loaded_round_lifecycle__read_json(p)
    del c["realtime_preparation"]
    loaded_round_lifecycle__atomic_json(p, c)
    assert any("same_session" in reason for reason in loaded_round_lifecycle__rounds.check_config(p))


def test_loaded_round_lifecycle__idle_only_wrench_config_is_an_offline_blocker(tmp_path):
    loaded_round_lifecycle__prepared_round(tmp_path, capture_only=True)
    p = tmp_path / "wrench.json"
    c = loaded_round_lifecycle__read_json(p)
    c["allowed_operation_states"] = ["idle"]
    loaded_round_lifecycle__atomic_json(p, c)
    assert any("rtcontrolling" in reason for reason in loaded_round_lifecycle__rounds.check_config(tmp_path / "round.json"))


# ============================================================================
# Source: tests/test_offline_leg.py
# ============================================================================

"""Unit/schema regressions; no controller imports or lab file dependency."""
import json as offline_leg__json

import numpy as offline_leg__np
import pandas as offline_leg__pd
import pytest as offline_leg__pytest

from measurement_validation.leg_episode import CalibrationError as offline_leg__CalibrationError, JointCalibration as offline_leg__JointCalibration, read_leg_episode as offline_leg__read_leg_episode
from measurement_validation.offline_leg import calibrate as offline_leg__calibrate, inspect_file as offline_leg__inspect_file, main as offline_leg__main


def offline_leg__recording(tmp_path, *, name="record.csv", rows=1001):
    t = offline_leg__np.arange(rows) * 2.0
    frame = offline_leg__pd.DataFrame({"receive_elapsed_ms": t,
                          "host_timestamp_filetime_100ns": 134361865866568070 + offline_leg__np.arange(rows) * 20000,
                          "synchronization": "four_axes_same_process_image_snapshot"})
    for axis in (1, 4):
        frame[f"axis{axis}_position_raw"] = -100000
        frame[f"axis{axis}_position_motor_deg"] = -45.77637
        frame[f"axis{axis}_torque_3c1a_nm"] = -4.5
        frame[f"axis{axis}_statusword_hex"] = "0x0231"
        frame[f"axis{axis}_error_code_hex"] = "0x0000"
        frame[f"axis{axis}_operation_enabled"] = "false"
        frame[f"axis{axis}_fault"] = "false"
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path, frame


def test_offline_leg__real_pdo_schema_has_explicit_units_and_status(tmp_path):
    path, _ = offline_leg__recording(tmp_path)
    ep = offline_leg__read_leg_episode(path)
    assert ep.host_time_s[-1] == 2
    hip = ep.joints["hip"]
    assert hip.position_counts[0] == -100000
    assert hip.motor_position_deg[0] == offline_leg__pytest.approx(-45.77637)
    assert hip.statusword[0] == 561
    assert hip.error_code[0] == 0
    assert not hip.operation_enabled.any() and not hip.fault.any()
    assert hip.torque_nm[0] == -4.5
    assert hip.flexion_deg is None


def test_offline_leg__motor_degrees_never_become_counts(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    frame.drop(columns=["axis1_position_raw", "axis4_position_raw"]).to_csv(path, index=False)
    ep = offline_leg__read_leg_episode(path)
    assert ep.joints["hip"].position_counts is None
    assert ep.joints["hip"].motor_position_deg[0] == offline_leg__pytest.approx(-45.77637)
    cal = offline_leg__JointCalibration(1000, -100000, -1, -10, 90)
    with offline_leg__pytest.raises(offline_leg__CalibrationError, match="raw position"):
        offline_leg__read_leg_episode(path, calibration={"hip": cal, "knee": cal})


def test_offline_leg__time_overrides_and_filetime_subtract_before_float(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    assert offline_leg__read_leg_episode(path, column_map={"time": "receive_elapsed_ms"}).host_time_s[-1] == 2
    frame = frame.drop(columns="receive_elapsed_ms")
    frame.to_csv(path, index=False)
    assert offline_leg__read_leg_episode(path).host_time_s[1] == .002
    frame["host_timestamp_filetime_100ns"] = 134361865866568070 + offline_leg__np.arange(len(frame)) * 3
    frame.to_csv(path, index=False)
    assert offline_leg__read_leg_episode(path).host_time_s[1] == 3e-7


def test_offline_leg__quality_locates_gaps_missing_duplicate_clock_and_motion(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    frame.loc[500:, "receive_elapsed_ms"] += 50
    frame.loc[8, "host_timestamp_filetime_100ns"] = frame.loc[7, "host_timestamp_filetime_100ns"]
    frame.loc[25, "axis1_torque_3c1a_nm"] = offline_leg__np.nan
    frame.loc[700:, "axis1_position_raw"] -= 30000
    frame.to_csv(path, index=False)
    report = offline_leg__inspect_file(path)
    assert report["long_intervals"][0]["csv_line"] == 502
    assert report["timing_missing_estimate"] == 25
    assert report["secondary_clocks"]["host_timestamp_filetime_100ns"]["duplicate_csv_lines"] == [10]
    assert report["missing_values_by_column"]["axis1_torque_3c1a_nm"] == 1
    assert not report["joint_checks"]["hip"]["whole_segment_stability"]["stable"]
    assert report["usable_for_dynamics"] is False
    offline_leg__json.dumps(report, allow_nan=False)


def test_offline_leg__batch_continues_and_reports_bad_file(tmp_path):
    path, _ = offline_leg__recording(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("unrelated\n1\n", encoding="utf-8")
    out = tmp_path / "out"
    assert offline_leg__main(["quality", str(path), str(bad), "--output-dir", str(out)]) == 1
    results = offline_leg__json.loads((out / "summary.json").read_text())
    assert {x["status"] for x in results} == {"reported", "error"}


def test_offline_leg__calibration_keeps_validation_out_of_fit_and_knee_missing(tmp_path):
    points = []
    for name, angle, counts, role in [("zero", 0, -100000, "fit"), ("thirty", 30, -130000, "fit"),
                                      ("independent", 15, -115200, "validation"),
                                      ("returned", None, -100300, "return")]:
        path, frame = offline_leg__recording(tmp_path, name=name + ".csv")
        frame["axis1_position_raw"] = counts
        frame.to_csv(path, index=False)
        points.append(dict(id=name, joint="hip", file=path.name, start_s=0, end_s=2.01,
                           angle_deg=angle, role=role, **({"pair_id": "zero_return"} if name in ("zero", "returned") else {})))
    candidate = dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000)
    result = offline_leg__calibrate({"points": points}, tmp_path, candidate)
    assert result["fits"]["knee"] is None
    assert result["fits"]["hip"]["counts_per_degree"] == offline_leg__pytest.approx(1000)
    assert result["points"][2]["prediction_error_deg"] == offline_leg__pytest.approx(.2)
    assert result["points"][3]["prediction_error_deg"] is None
    assert result["pairs"][0]["difference_deg"] == offline_leg__pytest.approx(.3)
    assert not result["pairs"][0]["same_measured_angle"]
    path = tmp_path / points[1]["file"]
    frame = offline_leg__pd.read_csv(path)
    frame["axis1_position_raw"] = offline_leg__np.linspace(-130000, -100000, len(frame))
    frame.to_csv(path, index=False)
    result = offline_leg__calibrate({"points": points}, tmp_path, candidate)
    assert not result["points"][1]["accepted"]
    assert result["fits"]["hip"] is None


def test_offline_leg__source_hash_change_rejected(tmp_path):
    path, _ = offline_leg__recording(tmp_path)
    spec = dict(id="p", joint="hip", file=path.name, start_s=0, end_s=2,
                angle_deg=0, role="fit", sha256="wrong")
    with offline_leg__pytest.raises(ValueError, match="hash changed"):
        offline_leg__calibrate({"points": [spec]}, tmp_path,
                  dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000))


def test_offline_leg__overlap_cannot_masquerade_as_independent_validation(tmp_path):
    path, _ = offline_leg__recording(tmp_path)
    base = dict(joint="hip", file=path.name, start_s=0, end_s=2, angle_deg=0)
    with offline_leg__pytest.raises(ValueError, match="overlap"):
        offline_leg__calibrate({"points": [dict(base, id="fit", role="fit"), dict(base, id="check", role="validation")]},
                  tmp_path, dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000))


def test_offline_leg__all_missing_counts_reported_and_excluded_from_fit(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    frame["axis1_position_raw"] = offline_leg__np.nan
    frame["axis1_position_motor_deg"] = offline_leg__np.nan
    frame.to_csv(path, index=False)
    report = offline_leg__inspect_file(path)
    assert report["joint_checks"]["hip"]["position_counts"]["mean"] is None
    offline_leg__json.dumps(report, allow_nan=False)
    result = offline_leg__calibrate({"points": [dict(id="missing", joint="hip", file=path.name,
                       start_s=0, end_s=2, angle_deg=0, role="fit")]}, tmp_path,
                       dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000))
    assert result["points"][0]["accepted"] is False
    assert result["points"][0]["prediction_error_deg"] is None


def test_offline_leg__faults_and_missing_state_block_dynamics_even_with_calibration(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    frame.loc[3, "axis1_error_code_hex"] = "0x1234"
    frame.loc[3, "axis1_fault"] = "true"
    frame.to_csv(path, index=False)
    cal = offline_leg__JointCalibration(1000, -100000, -1, -10, 90)
    from measurement_validation.leg_episode import quality_report
    report = quality_report(offline_leg__read_leg_episode(path, calibration={"hip": cal, "knee": cal}))
    assert report["usable_for_dynamics"] is False
    assert "hip_nonzero_error" in report["findings"]
    assert "hip_fault_present" in report["findings"]


def test_offline_leg__utc_duplicates_are_distinct_from_receive_duplicates(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    utc = offline_leg__pd.Series(offline_leg__pd.date_range("2026-10-11", periods=len(frame), freq="2ms", tz="UTC"))
    utc.iloc[9] = utc.iloc[8]
    frame["host_timestamp_utc"] = utc.astype(str)
    frame.to_csv(path, index=False)
    result = offline_leg__inspect_file(path)
    assert result["receive_time_duplicates_csv_lines"] == []
    assert result["secondary_clocks"]["host_timestamp_utc"]["duplicate_csv_lines"] == [11]
    frame.drop(columns=["receive_elapsed_ms", "host_timestamp_filetime_100ns"]).to_csv(path, index=False)
    ep = offline_leg__read_leg_episode(path)
    assert ep.host_time_s[-1] == 2


def test_offline_leg__oct7_elapsed_ms_and_torqueclose_schema(tmp_path):
    path, frame = offline_leg__recording(tmp_path)
    frame.rename(columns={"receive_elapsed_ms": "elapsed_ms",
                          "axis1_torque_3c1a_nm": "axis1_torqueclose_nm",
                          "axis4_torque_3c1a_nm": "axis4_torqueclose_nm"}).to_csv(path, index=False)
    ep = offline_leg__read_leg_episode(path)
    assert ep.host_time_s[-1] == 2
    assert ep.joints["hip"].torque_nm[0] == -4.5


def test_offline_leg__sparse_constant_positions_do_not_pass_stability():
    from measurement_validation.offline_leg import stability
    check = stability(offline_leg__np.array([0, .5, 1.]), offline_leg__np.array([1, 1, 1.]), max_range=50,
                      max_slope=20, min_duration=.8)
    assert not check["stable"]
    assert "time_gap_exceeded" in check["reasons"]


# ============================================================================
# Source: tests/test_prepare_lab_bundle.py
# ============================================================================

from copy import deepcopy as prepare_lab_bundle__deepcopy
import hashlib as prepare_lab_bundle__hashlib
import json as prepare_lab_bundle__json

import pytest as prepare_lab_bundle__pytest

import scripts.prepare_lab_bundle as prepare_lab_bundle__bundle


def prepare_lab_bundle__fixture_repo(tmp_path, monkeypatch):
    root = tmp_path / "rehab_robot"
    files = export_sections(root / "config")
    config = prepare_lab_bundle__deepcopy(prepare_lab_bundle__bundle.load(files["robot_leg_round"]))
    profile = prepare_lab_bundle__bundle.load(files["lab_next_visit"])
    frozen = tmp_path / "standalone" / "frozen_algorithm"
    frozen.mkdir(parents=True)
    digest = lambda p: prepare_lab_bundle__hashlib.sha256(p.read_bytes()).hexdigest()
    config["algorithm_root"] = str(frozen)
    for name in config["algorithm_sha256"]:
        file = frozen / name
        file.write_bytes(("# exact fixture " + name + "\n").encode())
        config["algorithm_sha256"][name] = digest(file)
    external = frozen.parent / "model" / "coupled_model.py"
    external.parent.mkdir()
    external.write_bytes(b"# exact external model\n")
    internal = root / "mechanics.py"
    internal.write_bytes(b"# exact repository dependency\n")
    config["algorithm_dependencies"] = {str(external): digest(external), str(internal): digest(internal)}
    candidate = root / "reference.csv"
    candidate.write_bytes(b"time_s,q_hip_rad\n0,0\n")
    config["candidates"] = [dict(candidate_id="reference", path=str(candidate), sha256=digest(candidate))]
    config["reference_candidate_id"] = "reference"
    (root / "config/robot_leg_round.json").write_text(prepare_lab_bundle__json.dumps(config), encoding="utf-8")
    (root / "config/lab_next_visit.json").write_text(prepare_lab_bundle__json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(prepare_lab_bundle__bundle, "ROOT", root)
    return root, frozen, config


def test_prepare_lab_bundle__portable_draft_copies_exact_bytes_without_reviewing_or_filling_facts(tmp_path, monkeypatch):
    root, frozen, original = prepare_lab_bundle__fixture_repo(tmp_path, monkeypatch)
    source_bytes = (root / "config/robot_leg_round.json").read_bytes()
    output = prepare_lab_bundle__bundle.build(root / "results/lab_pack")
    draft = prepare_lab_bundle__bundle.load(output / "round_training_draft.json")
    assert draft["trial_kind"] == "dummy_leg_reference" and draft["reviewed"] is False
    assert draft["collision_event_max_age_s"] is None
    assert draft["max_leg_start_age_s"] is None
    assert {k: v["passed"] for k, v in draft["site_evidence"].items()} == {k: v["passed"] for k, v in original["site_evidence"].items()}
    for name, digest in draft["algorithm_sha256"].items():
        assert prepare_lab_bundle__bundle.sha(output / draft["algorithm_root"] / name) == digest
    for name, digest in draft["algorithm_dependencies"].items():
        assert prepare_lab_bundle__bundle.sha(root / name) == digest
    assert (root / "config/robot_leg_round.json").read_bytes() == source_bytes
    with prepare_lab_bundle__pytest.raises(FileExistsError):
        prepare_lab_bundle__bundle.build(output)


def test_prepare_lab_bundle__changed_frozen_source_blocks_before_creating_bundle(tmp_path, monkeypatch):
    root, frozen, _ = prepare_lab_bundle__fixture_repo(tmp_path, monkeypatch)
    (frozen / "policy.py").write_bytes(b"changed\n")
    output = root / "results/lab_pack"
    with prepare_lab_bundle__pytest.raises(ValueError, match="algorithm changed"):
        prepare_lab_bundle__bundle.build(output)
    assert not output.exists()


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
        "level_1_static_cells.csv",
        "episode_features.csv",
        "level_2_repeatability_groups.csv",
        "level_3_trajectory_effects.csv",
        "level_3_time_local_effects.csv",
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
# Source: tests/test_reference_commissioning.py
# ============================================================================



import json as reference_commissioning__json
from pathlib import Path as reference_commissioning__Path

import numpy as reference_commissioning__np
import pandas as reference_commissioning__pd
import pytest as reference_commissioning__pytest

from control.reference_commissioning import (
    build_reference_commissioning_artifact as reference_commissioning__build_reference_commissioning_artifact,
    load_reference_commissioning_policy as reference_commissioning__load_reference_commissioning_policy,
    retime_formal_reference as reference_commissioning__retime_formal_reference,
)


reference_commissioning__ROOT = reference_commissioning__Path(__file__).resolve().parents[1]


def test_reference_commissioning__repository_commissioning_policy_is_narrow_and_non_executable():
    policy = reference_commissioning__load_reference_commissioning_policy(
        section_path("reference_commissioning")
    )
    assert policy.controller_speed_ratio_percent == 3
    assert policy.max_tcp_speed_m_s == reference_commissioning__pytest.approx(0.03)
    assert policy.cycles_per_session == 1
    assert policy.allow_missing_sdk_collision_state


def test_reference_commissioning__retime_preserves_geometry_and_caps_speed():
    source = reference_commissioning__pd.read_csv(
        artifact_path("reference_release/reference_measured_asymmetric_closed_slow.csv")
    )
    retimed, timing = reference_commissioning__retime_formal_reference(source, max_tcp_speed_m_s=0.03)
    geometry = ["q_hip_rad", "q_knee_rad", "x_pull_m", "z_pull_m"]
    reference_commissioning__np.testing.assert_array_equal(source[geometry].to_numpy(), retimed[geometry].to_numpy())
    assert timing["uniform_time_scale"] > 1.0
    assert timing["retimed_max_pull_speed_m_s"] <= 0.03 + 1e-12
    assert timing["retimed_duration_s"] > timing["source_duration_s"]


def test_reference_commissioning__artifact_never_authorizes_motion(tmp_path):
    policy = reference_commissioning__load_reference_commissioning_policy(
        section_path("reference_commissioning")
    )
    manifest = reference_commissioning__build_reference_commissioning_artifact(
        policy, tmp_path / "session", blockers=["timing_not_reviewed"]
    )
    assert manifest["motion_authorized"] is False
    assert manifest["status"] == "PREPARED_NOT_AUTHORIZED"
    assert manifest["remaining_motion_blockers"] == ["timing_not_reviewed"]
    saved = reference_commissioning__json.loads(
        (tmp_path / "session/commissioning_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["artifact"]["rows"] == 401


def test_reference_commissioning__policy_rejects_self_authorized_execution(tmp_path):
    source = reference_commissioning__json.loads(
        section_path("reference_commissioning").read_text(encoding="utf-8")
    )
    source["execution_enabled"] = True
    source["trajectory"]["path"] = str(
        artifact_path("reference_release/reference_measured_asymmetric_closed_slow.csv")
    )
    path = tmp_path / "bad.json"
    path.write_text(reference_commissioning__json.dumps(source), encoding="utf-8")
    with reference_commissioning__pytest.raises(PermissionError, match="must not self-authorize"):
        reference_commissioning__load_reference_commissioning_policy(path)


# ============================================================================
# Source: tests/test_robot_release_workflow.py
# ============================================================================

"""Real release/runner/preflight/executor chain with synthetic hardware only."""
from copy import deepcopy as robot_release_workflow__deepcopy
from datetime import datetime as robot_release_workflow__datetime, timedelta as robot_release_workflow__timedelta, timezone as robot_release_workflow__timezone
import json as robot_release_workflow__json
from types import SimpleNamespace as robot_release_workflow__SimpleNamespace

import pytest as robot_release_workflow__pytest

from control.execution_preflight import OPERATOR_CONFIRMATION as robot_release_workflow__OPERATOR_CONFIRMATION, evaluate_offline_execution_request as robot_release_workflow__evaluate_offline_execution_request
from control.robot_trajectory_executor import RokaeMotionExecutor as robot_release_workflow__RokaeMotionExecutor
from control.start_anchor import load_start_anchor as robot_release_workflow__load_start_anchor
from control.start_anchored_relative_trajectory import build_start_anchored_relative_trajectory as robot_release_workflow__build_start_anchored_relative_trajectory, load_rehab_frame_config as robot_release_workflow__load_rehab_frame_config
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH as robot_release_workflow__DEFAULT_REFERENCE_PATH
from safety.experiment_safety import load_experiment_safety_config as robot_release_workflow__load_experiment_safety_config
from safety.robot_trial_release import (
    APPROVAL_CONFIRMATION as robot_release_workflow__APPROVAL_CONFIRMATION, EVIDENCE_KINDS as robot_release_workflow__EVIDENCE_KINDS, load_and_validate_robot_trial_release as robot_release_workflow__load_and_validate_robot_trial_release,
    read_json as robot_release_workflow__read_json, sha256_file as robot_release_workflow__sha256_file,
)
from scripts.prepare_robot_trial_release import main as robot_release_workflow__release_main
from scripts.acquire_robot_data import load_live_wrench_config as robot_release_workflow__load_live_wrench_config
import scripts.run_rehab_experiment as robot_release_workflow__runner
robot_release_workflow___wrench_config = robot_trial_release___wrench_config
robot_release_workflow___reviewed_files = run_rehab_experiment___reviewed_files
robot_release_workflow__Acquisition = robot_trajectory_executor__Acquisition
robot_release_workflow__Motion = robot_trajectory_executor__Motion


def robot_release_workflow__workflow(tmp_path):
    anchor, frame, safety = robot_release_workflow___reviewed_files(tmp_path)
    wrench = robot_release_workflow___wrench_config(tmp_path / "wrench_config.json")
    draft, approved = tmp_path / "draft.json", tmp_path / "releases/approved.json"
    expires = (robot_release_workflow__datetime.now(robot_release_workflow__timezone.utc) + robot_release_workflow__timedelta(hours=2)).isoformat()
    argv = ["--reference", str(robot_release_workflow__DEFAULT_REFERENCE_PATH), "--anchor", str(anchor),
            "--frame-config", str(frame), "--safety-config", str(safety), "--wrench-config", str(wrench),
            "--robot-model", "xMate6-Pro", "--robot-serial-number", "SN-RUNNER",
            "--controller-version", "controller-test", "--sdk-version", "0.7.0",
            "--reviewer", "safety", "--operator", "operator", "--expires-at", expires,
            "--collision-source", "events", "--collision-event-max-age-s", "0.5", "--wrench-hz", "50"]
    assert robot_release_workflow__release_main([*argv, "--output", str(draft)]) == 0
    payload = robot_release_workflow__read_json(draft)
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    records = {}
    for kind in robot_release_workflow__EVIDENCE_KINDS:
        path = site_dir / (kind + ".json")
        path.write_text('{"synthetic_test_evidence": true}', encoding="utf-8")
        records[kind] = dict(passed=True, path=path.name, sha256=robot_release_workflow__sha256_file(path))
    site = site_dir / "review.json"
    site.write_text(robot_release_workflow__json.dumps(dict(scope=payload["scope"], decision="approved", reviewer="safety",
                        reviewed_at=robot_release_workflow__datetime.now(robot_release_workflow__timezone.utc).isoformat(),
                        bundle_sha256=payload["bundle_sha256"], records=records)), encoding="utf-8")
    approve_args = [*argv, "--output", str(approved), "--approve", "--input-draft", str(draft),
                    "--site-review", str(site), "--confirmation", robot_release_workflow__APPROVAL_CONFIRMATION]
    inputs = dict(reference_path=robot_release_workflow__DEFAULT_REFERENCE_PATH, anchor_path=anchor,
                  frame_config_path=frame, safety_config_path=safety, wrench_config_path=wrench)
    return robot_release_workflow__SimpleNamespace(anchor=anchor, frame=frame, safety=safety, wrench=wrench,
                           draft=draft, approved=approved, site=site, approve_args=approve_args, inputs=inputs)


def robot_release_workflow__approve(w):
    assert robot_release_workflow__release_main(w.approve_args) == 0
    return robot_release_workflow__load_and_validate_robot_trial_release(w.approved, **w.inputs)


def robot_release_workflow__runner_args(w, tmp_path):
    return dict(episode_dir=tmp_path / "episode", trajectory_id="reference_measured_asymmetric_closed_slow",
                anchor_path=w.anchor, frame_config_path=w.frame,
                safety_config_path=w.safety, requested_anchor_id="anchor_runner_test",
                enable_motion=True, operator_confirmation=robot_release_workflow__OPERATOR_CONFIRMATION,
                robot_ip="192.0.2.1", local_ip="192.0.2.2", wrench_config_path=w.wrench,
                live_wrench_config=robot_release_workflow__load_live_wrench_config(w.wrench), wrench_hz=50.,
                collision_source="events", collision_event_max_age_s=.5, release_approval_path=w.approved)


class robot_release_workflow__FakeAdapter:
    def __init__(self):
        self.summary = robot_release_workflow__Acquisition().summary
        self.summary.update(collision_state_source="events", safety_event_status={"max_age_s": .5})
        self.summary["robot_metadata"].update(robot_serial_number="SN-RUNNER", xcore_sdk_version="0.7.0")

    def is_connected(self):
        return True

    def get_robot_state_summary(self):
        return robot_release_workflow__deepcopy(self.summary)

    def read_tcp_pose(self):
        return (.4, 0., .3, .1, .2, .3)


class robot_release_workflow__FakeAcquisition(robot_release_workflow__Acquisition):
    def __init__(self, adapter, logger, **kwargs):
        super().__init__()
        self.adapter = adapter
        self.stopped = False
        self.live_producer_names = ()

    def start(self, **kwargs):
        pass

    def stop(self):
        self.stopped = True

    def wait_until_healthy(self, **kwargs):
        return True

    def latest_state_frame(self):
        return robot_release_workflow__SimpleNamespace(valid=True, tcp_position_m=(.4, 0., .3), tcp_orientation_rad=(.1, .2, .3))


def robot_release_workflow__install_fakes(monkeypatch, *, motion=None):
    adapter = robot_release_workflow__FakeAdapter()
    motion = motion or robot_release_workflow__Motion()
    # Accelerate only the scheduler clock/wait; keep trajectory audit, bindings,
    # actual logger and command dispatch logic intact.
    class Clock:
        t = 0.
        def now_s(self):
            return self.t
    class FastExecutor(robot_release_workflow__RokaeMotionExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, clock=Clock())
        def _wait_until(self, deadline_s):
            self.clock.t = deadline_s
    monkeypatch.setattr(robot_release_workflow__runner, "RealRobotAcquisition", robot_release_workflow__FakeAcquisition)
    monkeypatch.setattr(robot_release_workflow__runner, "build_live_wrench_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(robot_release_workflow__runner, "RokaeMotionExecutor", FastExecutor)
    return adapter, motion


def test_robot_release_workflow__cli_to_real_runner_preflight_and_executor_success(tmp_path, monkeypatch):
    w = robot_release_workflow__workflow(tmp_path)
    validated = robot_release_workflow__approve(w)
    assert robot_release_workflow__read_json(w.draft)["approved"] is False
    assert validated.payload["wrench_hz"] == 50
    adapter, motion = robot_release_workflow__install_fakes(monkeypatch)
    result = robot_release_workflow__runner.run_execute(**robot_release_workflow__runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                                motion_factory=lambda a: motion)
    assert result["completed"] is True
    assert result["commands_dispatched"] > 2
    assert [call[0] for call in motion.calls].count("start") == 1
    assert motion.calls[-1] == ("stop", "trajectory_completed")
    metadata = robot_release_workflow__read_json(tmp_path / "episode/metadata.json")
    assert metadata["trial_release"]["approval"]["scope"] == "FIRST_ROBOT_TRIAL_EMPTY_LOAD"
    assert metadata["runtime_preflight"]["allowed"] is True


@robot_release_workflow__pytest.mark.parametrize("change", ["missing_evidence", "failed_evidence", "changed_input", "expired",
                                   "wrong_scope", "changed_code", "bad_frequency", "bad_reviewer"])
def test_robot_release_workflow__invalid_approval_is_never_published(tmp_path, change):
    w = robot_release_workflow__workflow(tmp_path)
    site = robot_release_workflow__read_json(w.site)
    draft = robot_release_workflow__read_json(w.draft)
    if change == "missing_evidence":
        site["records"].pop("setup_and_stop")
    elif change == "failed_evidence":
        site["records"]["controller_safety"]["passed"] = False
    elif change == "changed_input":
        w.safety.write_text(w.safety.read_text() + "\n")
    elif change == "expired":
        draft["expires_at"] = (robot_release_workflow__datetime.now(robot_release_workflow__timezone.utc) - robot_release_workflow__timedelta(seconds=1)).isoformat()
    elif change == "wrong_scope":
        draft["scope"] = site["scope"] = "FIRST_ROBOT_TRIAL"
    elif change == "changed_code":
        draft["code_sha256"]["control/robot_trajectory_executor.py"] = "bad"
    elif change == "bad_frequency":
        draft["wrench_hz"] = -1
    elif change == "bad_reviewer":
        site["reviewer"] = "other"
    w.site.write_text(robot_release_workflow__json.dumps(site), encoding="utf-8")
    w.draft.write_text(robot_release_workflow__json.dumps(draft), encoding="utf-8")
    with robot_release_workflow__pytest.raises((ValueError, PermissionError)):
        robot_release_workflow__release_main(w.approve_args)
    assert not w.approved.exists()


@robot_release_workflow__pytest.mark.parametrize("change", ["frequency", "source", "event_age", "wrench_object", "identity", "reviewer"])
def test_robot_release_workflow__runner_or_approval_mismatch_never_connects(tmp_path, change):
    w = robot_release_workflow__workflow(tmp_path)
    robot_release_workflow__approve(w)
    kwargs = robot_release_workflow__runner_args(w, tmp_path)
    if change == "frequency":
        kwargs["wrench_hz"] = 25
    elif change == "source":
        kwargs.update(collision_source="query", collision_event_max_age_s=None)
    elif change == "event_age":
        kwargs["collision_event_max_age_s"] = .7
    elif change == "wrench_object":
        kwargs["live_wrench_config"]["budgets"]["query_s"] = .6
    elif change in ("identity", "reviewer"):
        args = w.approve_args.copy()
        index = args.index("--robot-serial-number" if change == "identity" else "--reviewer") + 1
        args[index] = "other"
        # Another publication target proves CLI rejection before writing.
        args[args.index("--output") + 1] = str(tmp_path / "other.json")
        with robot_release_workflow__pytest.raises(PermissionError):
            robot_release_workflow__release_main(args)
        assert not (tmp_path / "other.json").exists()
        return
    calls = []
    with robot_release_workflow__pytest.raises(PermissionError):
        robot_release_workflow__runner.run_execute(**kwargs, adapter_factory=lambda ip: calls.append(ip))
    assert calls == []
    assert not (tmp_path / "episode").exists()


def test_robot_release_workflow__plain_dict_cannot_approve_preflight(tmp_path):
    w = robot_release_workflow__workflow(tmp_path)
    validated = robot_release_workflow__approve(w)
    frame, anchor = robot_release_workflow__load_rehab_frame_config(w.frame), robot_release_workflow__load_start_anchor(w.anchor)
    trajectory, audit, _ = robot_release_workflow__build_start_anchored_relative_trajectory(
        robot_release_workflow__DEFAULT_REFERENCE_PATH, current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    args = dict(mode="execute", enable_motion=True, operator_confirmation=robot_release_workflow__OPERATOR_CONFIRMATION,
                requested_anchor_id=anchor.anchor_id, frame=frame, anchor=anchor,
                safety=robot_release_workflow__load_experiment_safety_config(w.safety), trajectory=trajectory, audit=audit)
    assert robot_release_workflow__evaluate_offline_execution_request(**args, release_approval=validated).allowed
    assert not robot_release_workflow__evaluate_offline_execution_request(**args, release_approval=validated.payload).allowed


@robot_release_workflow__pytest.mark.parametrize("change", ["sdk", "event_age", "serial"])
def test_robot_release_workflow__runtime_identity_or_source_budget_blocks_before_attachment(tmp_path, monkeypatch, change):
    w = robot_release_workflow__workflow(tmp_path)
    robot_release_workflow__approve(w)
    adapter, motion = robot_release_workflow__install_fakes(monkeypatch)
    if change == "sdk":
        adapter.summary["robot_metadata"]["xcore_sdk_version"] = "other"
    elif change == "serial":
        adapter.summary["robot_metadata"]["robot_serial_number"] = "other"
    else:
        adapter.summary["safety_event_status"]["max_age_s"] = 9
    with robot_release_workflow__pytest.raises(PermissionError):
        robot_release_workflow__runner.run_execute(**robot_release_workflow__runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                           motion_factory=lambda a: motion)
    assert motion.calls == []


def test_robot_release_workflow__expiry_during_execution_stops_before_next_target(tmp_path, monkeypatch):
    import safety.robot_trial_release as release_module
    w = robot_release_workflow__workflow(tmp_path)
    robot_release_workflow__approve(w)
    class ExpiringMotion(robot_release_workflow__Motion):
        def start_cartesian_hold(self, pose):
            super().start_cartesian_hold(pose)
            real_datetime = robot_release_workflow__datetime
            class FutureDateTime:
                @staticmethod
                def now(tz):
                    return real_datetime.now(tz) + robot_release_workflow__timedelta(days=1)
                fromisoformat = staticmethod(real_datetime.fromisoformat)
            monkeypatch.setattr(release_module, "datetime", FutureDateTime)
    adapter, motion = robot_release_workflow__install_fakes(monkeypatch, motion=ExpiringMotion())
    with robot_release_workflow__pytest.raises(RuntimeError, match="expired"):
        robot_release_workflow__runner.run_execute(**robot_release_workflow__runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                           motion_factory=lambda a: motion)
    assert not any(call[0] == "send" for call in motion.calls)
    assert any(call[0] == "stop" for call in motion.calls)


def test_robot_release_workflow__actual_execute_cli_forwards_release_and_acquisition_options(tmp_path, monkeypatch):
    w = robot_release_workflow__workflow(tmp_path)
    robot_release_workflow__approve(w)
    adapter, motion = robot_release_workflow__install_fakes(monkeypatch)
    assert robot_release_workflow__runner.main([
        "--mode", "execute", "--ip", "192.0.2.1", "--local-ip", "192.0.2.2",
        "--episode-dir", str(tmp_path / "episode"), "--anchor", str(w.anchor),
        "--anchor-id", "anchor_runner_test", "--frame-config", str(w.frame),
        "--safety-config", str(w.safety), "--wrench-config", str(w.wrench), "--wrench-hz", "50",
        "--release-approval", str(w.approved), "--enable-motion",
        "--operator-confirmation", robot_release_workflow__OPERATOR_CONFIRMATION, "--collision-source", "events",
        "--collision-event-max-age-s", "0.5"],
        adapter_factory=lambda ip: adapter, motion_factory=lambda a: motion) == 0
    assert motion.calls[-1] == ("stop", "trajectory_completed")


@robot_release_workflow__pytest.mark.parametrize("change", ["evidence", "approval", "trajectory", "safety"])
def test_robot_release_workflow__changes_after_loading_block_preflight(tmp_path, change):
    w = robot_release_workflow__workflow(tmp_path)
    validated = robot_release_workflow__approve(w)
    frame, anchor = robot_release_workflow__load_rehab_frame_config(w.frame), robot_release_workflow__load_start_anchor(w.anchor)
    trajectory, audit, _ = robot_release_workflow__build_start_anchored_relative_trajectory(
        robot_release_workflow__DEFAULT_REFERENCE_PATH, current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    safety = robot_release_workflow__load_experiment_safety_config(w.safety)
    if change == "evidence":
        (w.site.parent / "setup_and_stop.json").write_text("changed")
    elif change == "approval":
        w.approved.write_text(w.approved.read_text() + "\n")
    elif change == "trajectory":
        trajectory.loc[1, "tcp_x_base"] += .001
    else:
        from dataclasses import replace
        safety = replace(safety, max_force_n=99.)
    preflight = robot_release_workflow__evaluate_offline_execution_request(
        mode="execute", enable_motion=True, operator_confirmation=robot_release_workflow__OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id, frame=frame, anchor=anchor,
        safety=safety, trajectory=trajectory, audit=audit, release_approval=validated)
    assert not preflight.allowed
    assert any("release_validation_failed" in r for r in preflight.reasons)


def test_robot_release_workflow__evidence_change_after_attachment_blocks_first_hold(tmp_path, monkeypatch):
    w = robot_release_workflow__workflow(tmp_path)
    robot_release_workflow__approve(w)
    class TamperingMotion(robot_release_workflow__Motion):
        def attach_externally_prepared(self, **kwargs):
            super().attach_externally_prepared(**kwargs)
            (w.site.parent / "controller_safety.json").write_text("changed")
    adapter, motion = robot_release_workflow__install_fakes(monkeypatch, motion=TamperingMotion())
    with robot_release_workflow__pytest.raises(PermissionError, match="reviewed file changed"):
        robot_release_workflow__runner.run_execute(**robot_release_workflow__runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                           motion_factory=lambda a: motion)
    assert not any(c[0] in ("start", "send") for c in motion.calls)
    assert any(c[0] == "stop" for c in motion.calls)


def test_robot_release_workflow__draft_cannot_execute_and_missing_frequency_cannot_generate_draft(tmp_path):
    w = robot_release_workflow__workflow(tmp_path)
    kwargs = robot_release_workflow__runner_args(w, tmp_path)
    kwargs["release_approval_path"] = w.draft
    calls = []
    with robot_release_workflow__pytest.raises(PermissionError, match="not approved"):
        robot_release_workflow__runner.run_execute(**kwargs, adapter_factory=lambda ip: calls.append(ip))
    assert calls == []
    args = w.approve_args.copy()
    index = args.index("--wrench-hz")
    del args[index:index + 2]
    with robot_release_workflow__pytest.raises(SystemExit):
        robot_release_workflow__release_main(args)


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
from scripts.check_rt_state_timing import measure as rokae_diagnostics__measure_rt
from scripts.check_snapshot_alignment import measure as rokae_diagnostics__measure_alignment
from scripts.check_wrench_query_timing import measure as rokae_diagnostics__measure_query
from scripts.rokae_diagnostic_common import (
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


def test_rokae_safety_events__execute_cli_passes_events_configuration_to_runner(monkeypatch, capsys):
    import scripts.run_rehab_experiment as module

    calls = []
    monkeypatch.setattr(module, "run_execute", lambda **kwargs: calls.append(kwargs) or {})
    assert module.main(["--mode", "execute", "--ip", "192.0.2.1",
                        "--episode-dir", "unused", "--anchor", "unused.json",
                        "--anchor-id", "unused", "--collision-source", "events",
                        "--collision-event-max-age-s", "0.2"]) == 0
    assert calls[0]["collision_source"] == "events"
    assert calls[0]["collision_event_max_age_s"] == .2


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
# Source: tests/test_start_anchor_and_safety.py
# ============================================================================

"""Read-only start-anchor capture and fail-closed safety configuration tests."""



from dataclasses import replace as start_anchor_and_safety__replace
import json as start_anchor_and_safety__json
from pathlib import Path as start_anchor_and_safety__Path
from types import SimpleNamespace as start_anchor_and_safety__SimpleNamespace

import pytest as start_anchor_and_safety__pytest

from control.start_anchor import (
    FixedTcpOrientation as start_anchor_and_safety__FixedTcpOrientation,
    StartAnchor as start_anchor_and_safety__StartAnchor,
    capture_start_anchor as start_anchor_and_safety__capture_start_anchor,
    load_start_anchor as start_anchor_and_safety__load_start_anchor,
    save_start_anchor as start_anchor_and_safety__save_start_anchor,
)
from safety.experiment_safety import (
    ExperimentSafetyConfig as start_anchor_and_safety__ExperimentSafetyConfig,
    load_experiment_safety_config as start_anchor_and_safety__load_experiment_safety_config,
    require_execute_safety as start_anchor_and_safety__require_execute_safety,
)


start_anchor_and_safety__POSE = (0.41, -0.12, 0.36, 0.1, -0.2, 0.3)
start_anchor_and_safety__JOINTS = (0.0, 0.1, -0.2, 0.3, -0.4, 0.5)
start_anchor_and_safety__SOFT_LIMITS = tuple((-2.0, 2.0) for _ in range(6))


class start_anchor_and_safety__ReadOnlyFrameAdapter:
    def __init__(self) -> None:
        self.read_count = 0
        self.metadata_read_count = 0
        self.motion_calls: list[str] = []

    def read_state_frame(self):
        self.read_count += 1
        return start_anchor_and_safety__SimpleNamespace(
            valid=True,
            invalid_reason="",
            tcp_position_m=start_anchor_and_safety__POSE[:3],
            tcp_orientation_rad=start_anchor_and_safety__POSE[3:],
            joint_position_rad=start_anchor_and_safety__JOINTS,
        )

    def read_robot_metadata(self):
        self.metadata_read_count += 1
        return {
            "robot_model": "xMate6-Pro",
            "robot_serial_number": "SN-TEST-001",
            "controller_version": "4.2-test",
        }

    def _forbidden(self, name: str):
        self.motion_calls.append(name)
        raise AssertionError(f"capture called forbidden adapter method: {name}")

    def connect(self):
        self._forbidden("connect")

    def enable(self):
        self._forbidden("enable")

    def enable_realtime(self):
        self._forbidden("enable_realtime")

    def move_l(self, _target=None):
        self._forbidden("move_l")

    def move_j(self, _target=None):
        self._forbidden("move_j")

    def stop(self):
        self._forbidden("stop")


def start_anchor_and_safety___capture(adapter=None) -> start_anchor_and_safety__StartAnchor:
    return start_anchor_and_safety__capture_start_anchor(
        adapter or start_anchor_and_safety__ReadOnlyFrameAdapter(),
        trajectory_id="reference_measured_asymmetric_closed_slow",
        reference_start_q_hip=0.25,
        reference_start_q_knee=0.35,
        anchor_id="anchor_test_001",
        tool_name="rehab_cuff",
        workpiece_name="rehab_fixture",
        notes="operator will review after capture",
        clock=lambda: 1234.5,
    )


def start_anchor_and_safety___complete_safety(*, reviewed: bool = True) -> start_anchor_and_safety__ExperimentSafetyConfig:
    return start_anchor_and_safety__ExperimentSafetyConfig(
        max_tcp_speed_m_s=0.1,
        max_tcp_acceleration_m_s2=0.2,
        max_start_anchor_position_error_m=0.01,
        max_start_anchor_orientation_error_rad=0.01,
        max_command_lateness_s=0.1,
        max_force_n=10.0,
        max_torque_nm=2.0,
        max_state_age_s=0.05,
        max_wrench_age_s=0.05,
        max_state_wrench_skew_s=0.02,
        workspace_min_base_m=(0.1, -0.3, 0.1),
        workspace_max_base_m=(0.7, 0.3, 0.8),
        expected_robot_model="xMate6-Pro",
        expected_robot_serial_number="SN-TEST-001",
        expected_controller_version="4.2-test",
        reviewed_tool_name="rehab_cuff",
        reviewed_workpiece_name="rehab_fixture",
        reviewed_payload_mass_kg=1.25,
        reviewed_payload_cog_m=(0.0, 0.0, 0.08),
        reviewed_payload_inertia_kg_m2=(0.01, 0.02, 0.03),
        reviewed_joint_soft_limits_rad=start_anchor_and_safety__SOFT_LIMITS,
        reviewed_rt_filter_hz=25.0,
        reviewed_rt_network_tolerance_percent=20.0,
        robot_identity_reviewed=True,
        tool_workpiece_reviewed=True,
        payload_configuration_reviewed=True,
        collision_configuration_reviewed=True,
        joint_soft_limits_reviewed=True,
        realtime_configuration_reviewed=True,
        reviewed=reviewed,
        notes="synthetic unit-test values only",
    )


def test_start_anchor_and_safety__capture_reads_one_cached_state_and_never_calls_lifecycle_or_motion() -> None:
    adapter = start_anchor_and_safety__ReadOnlyFrameAdapter()
    anchor = start_anchor_and_safety___capture(adapter)
    assert adapter.read_count == 1
    assert adapter.metadata_read_count == 1
    assert adapter.motion_calls == []
    assert anchor.capture_host_time_s == 1234.5
    assert anchor.tcp_pose_base == start_anchor_and_safety__POSE
    assert anchor.tcp_position_base_m == start_anchor_and_safety__POSE[:3]
    assert anchor.tcp_orientation.strategy == "fixed"
    assert anchor.tcp_orientation.values_rad == start_anchor_and_safety__POSE[3:]
    assert anchor.robot_joint_positions == start_anchor_and_safety__JOINTS
    assert anchor.trajectory_id == "reference_measured_asymmetric_closed_slow"
    assert anchor.robot_model == "xMate6-Pro"
    assert anchor.robot_serial_number == "SN-TEST-001"
    assert anchor.controller_version == "4.2-test"
    assert anchor.tool_name == "rehab_cuff"
    assert anchor.workpiece_name == "rehab_fixture"
    assert anchor.reviewed is False


def test_start_anchor_and_safety__capture_supports_explicit_read_adapter_without_motion_methods() -> None:
    class ExplicitReadAdapter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def read_tcp_pose(self):
            self.calls.append("read_tcp_pose")
            return start_anchor_and_safety__POSE

        def read_joint_positions(self):
            self.calls.append("read_joint_positions")
            return start_anchor_and_safety__JOINTS

    adapter = ExplicitReadAdapter()
    anchor = start_anchor_and_safety___capture(adapter)
    assert adapter.calls == ["read_tcp_pose", "read_joint_positions"]
    assert anchor.tcp_pose_base == start_anchor_and_safety__POSE


def test_start_anchor_and_safety__anchor_json_round_trip_is_atomic_and_strict(tmp_path: start_anchor_and_safety__Path) -> None:
    path = tmp_path / "subject" / "anchor.json"
    anchor = start_anchor_and_safety___capture()
    assert start_anchor_and_safety__save_start_anchor(anchor, path) == path
    assert start_anchor_and_safety__load_start_anchor(path) == anchor
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    payload = start_anchor_and_safety__json.loads(path.read_text(encoding="utf-8"))
    assert payload["reviewed"] is False
    assert payload["tcp_orientation"]["strategy"] == "fixed"

    payload["unexpected"] = True
    path.write_text(start_anchor_and_safety__json.dumps(payload), encoding="utf-8")
    with start_anchor_and_safety__pytest.raises(ValueError, match="fields must be exactly"):
        start_anchor_and_safety__StartAnchor.load_json(path)


def test_start_anchor_and_safety__anchor_loader_rejects_truthy_review_and_inconsistent_pose(tmp_path: start_anchor_and_safety__Path) -> None:
    payload = start_anchor_and_safety___capture().to_dict()
    path = tmp_path / "anchor.json"
    payload["reviewed"] = 1
    path.write_text(start_anchor_and_safety__json.dumps(payload), encoding="utf-8")
    with start_anchor_and_safety__pytest.raises(ValueError, match="reviewed must be"):
        start_anchor_and_safety__StartAnchor.load_json(path)

    payload["reviewed"] = False
    payload["tcp_position_base_m"] = [9.0, 8.0, 7.0]
    path.write_text(start_anchor_and_safety__json.dumps(payload), encoding="utf-8")
    with start_anchor_and_safety__pytest.raises(ValueError, match="position does not match"):
        start_anchor_and_safety__StartAnchor.load_json(path)


def test_start_anchor_and_safety__invalid_cached_state_is_rejected_without_fallback_motion() -> None:
    adapter = start_anchor_and_safety__ReadOnlyFrameAdapter()
    adapter.read_state_frame = lambda: start_anchor_and_safety__SimpleNamespace(
        valid=False,
        invalid_reason="state_stale",
        tcp_position_m=start_anchor_and_safety__POSE[:3],
        tcp_orientation_rad=start_anchor_and_safety__POSE[3:],
        joint_position_rad=start_anchor_and_safety__JOINTS,
    )
    with start_anchor_and_safety__pytest.raises(RuntimeError, match="state_stale"):
        start_anchor_and_safety___capture(adapter)
    assert adapter.motion_calls == []


def test_start_anchor_and_safety__repository_safety_config_is_unreviewed_and_fail_closed() -> None:
    from dataclasses import replace

    config = start_anchor_and_safety__load_experiment_safety_config(section_path("experiment_safety"))
    assert config.reviewed is False
    allowed, reasons = config.validate_for_execution()
    assert allowed is False
    assert "experiment_safety_not_reviewed" in reasons
    with start_anchor_and_safety__pytest.raises(PermissionError, match="real robot execution blocked"):
        start_anchor_and_safety__require_execute_safety(config)

    incomplete = replace(config, reviewed=True, max_start_anchor_position_error_m=None)
    allowed, reasons = incomplete.validate_for_execution()
    assert allowed is False
    assert "max_start_anchor_position_error_m_not_configured" in reasons
    with start_anchor_and_safety__pytest.raises(PermissionError, match="max_start_anchor_position_error_m_not_configured"):
        start_anchor_and_safety__require_execute_safety(incomplete)

def test_start_anchor_and_safety__reviewed_flag_alone_never_bypasses_missing_limits() -> None:
    config = start_anchor_and_safety__ExperimentSafetyConfig(reviewed=True, notes="reviewed but incomplete")
    assert config.execution_allowed is False
    assert "max_force_n_not_configured" in config.execution_block_reasons()
    with start_anchor_and_safety__pytest.raises(PermissionError):
        config.require_execute_allowed()


def test_start_anchor_and_safety__complete_configuration_still_requires_boolean_review() -> None:
    draft = start_anchor_and_safety___complete_safety(reviewed=False)
    assert not draft.execution_allowed
    assert draft.execution_block_reasons() == ("experiment_safety_not_reviewed",)
    reviewed = start_anchor_and_safety__replace(draft, reviewed=True)
    assert reviewed.validate_for_execution() == (True, ())
    reviewed.require_execute_allowed()


def test_start_anchor_and_safety__safety_json_round_trip_and_workspace_validation(tmp_path: start_anchor_and_safety__Path) -> None:
    config = start_anchor_and_safety___complete_safety()
    path = tmp_path / "experiment_safety.json"
    config.save_json(path)
    assert start_anchor_and_safety__ExperimentSafetyConfig.load_json(path) == config
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))

    with start_anchor_and_safety__pytest.raises(ValueError, match="both be null or both be configured"):
        start_anchor_and_safety__ExperimentSafetyConfig(workspace_min_base_m=(0.0, 0.0, 0.0))
    with start_anchor_and_safety__pytest.raises(ValueError, match="strictly below"):
        start_anchor_and_safety__ExperimentSafetyConfig(
            workspace_min_base_m=(0.0, 0.0, 0.0),
            workspace_max_base_m=(0.0, 1.0, 1.0),
        )
    with start_anchor_and_safety__pytest.raises(ValueError, match="finite positive"):
        start_anchor_and_safety__ExperimentSafetyConfig(max_force_n=0.0)
    with start_anchor_and_safety__pytest.raises(ValueError, match=r"six \[lower, upper\] pairs"):
        start_anchor_and_safety__ExperimentSafetyConfig(reviewed_joint_soft_limits_rad=((-1.0, 1.0),))
    with start_anchor_and_safety__pytest.raises(ValueError, match="lower must be below upper"):
        start_anchor_and_safety__ExperimentSafetyConfig(
            reviewed_joint_soft_limits_rad=tuple((1.0, -1.0) for _ in range(6))
        )
    with start_anchor_and_safety__pytest.raises(ValueError, match="non-negative"):
        start_anchor_and_safety__ExperimentSafetyConfig(reviewed_payload_inertia_kg_m2=(-0.1, 0.0, 0.0))
    with start_anchor_and_safety__pytest.raises(ValueError, match=r"\[1, 1000\]"):
        start_anchor_and_safety__ExperimentSafetyConfig(reviewed_rt_filter_hz=1001.0)
    with start_anchor_and_safety__pytest.raises(ValueError, match=r"\[0, 100\]"):
        start_anchor_and_safety__ExperimentSafetyConfig(reviewed_rt_network_tolerance_percent=101.0)


def test_start_anchor_and_safety__orientation_requires_fixed_euler_representation() -> None:
    with start_anchor_and_safety__pytest.raises(ValueError, match="strategy"):
        start_anchor_and_safety__FixedTcpOrientation(values_rad=start_anchor_and_safety__POSE[3:], strategy="active")
    with start_anchor_and_safety__pytest.raises(ValueError, match="representation"):
        start_anchor_and_safety__FixedTcpOrientation(values_rad=start_anchor_and_safety__POSE[3:], representation="rotation_vector")


# ============================================================================
# Source: tests/test_start_anchored_relative.py
# ============================================================================

"""Regression tests for the frozen start-anchored experiment geometry."""



import inspect as start_anchored_relative__inspect
import json as start_anchored_relative__json
from pathlib import Path as start_anchored_relative__Path

import numpy as start_anchored_relative__np
import pytest as start_anchored_relative__pytest

from control.start_anchored_relative_trajectory import (
    ABSOLUTE_CALIBRATED_MODE as start_anchored_relative__ABSOLUTE_CALIBRATED_MODE,
    APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256 as start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256,
    APPROVED_FIRST_ROBOT_TRIAL_L1_M as start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_L1_M,
    APPROVED_FIRST_ROBOT_TRIAL_L2_M as start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_L2_M,
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID as start_anchored_relative__FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    RehabFrameConfig as start_anchored_relative__RehabFrameConfig,
    START_ANCHORED_RELATIVE_MODE as start_anchored_relative__START_ANCHORED_RELATIVE_MODE,
    build_start_anchored_relative_trajectory as start_anchored_relative__build_start_anchored_relative_trajectory,
    load_rehab_frame_config as start_anchored_relative__load_rehab_frame_config,
)
from lower_limb_sim.reference_closed_c2 import (
    APPROVED_HIP_ROM_DEG as start_anchored_relative__APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG as start_anchored_relative__APPROVED_KNEE_ROM_DEG,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_NOMINAL_ID as start_anchored_relative__MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH as start_anchored_relative__DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory as start_anchored_relative__load_closed_reference_trajectory,
)


start_anchored_relative__ANCHOR = start_anchored_relative__np.asarray((0.41, -0.12, 0.36, 0.1, -0.2, 0.3), dtype=float)
start_anchored_relative__SOURCE_CANDIDATE_DIRECTORY = (
    artifact_path("lower_limb_sim/data/reference_candidates")
)


def start_anchored_relative___identity_frame(*, reviewed: bool = False) -> start_anchored_relative__RehabFrameConfig:
    return start_anchored_relative__RehabFrameConfig(
        rehab_x_axis_in_base=(1.0, 0.0, 0.0),
        rehab_z_axis_in_base=(0.0, 0.0, 1.0),
        reviewed=reviewed,
        notes="offline test",
    )


def start_anchored_relative___build(frame: start_anchored_relative__RehabFrameConfig | None = None):
    return start_anchored_relative__build_start_anchored_relative_trajectory(
        current_tcp_start_pose=start_anchored_relative__ANCHOR,
        rehab_frame=frame or start_anchored_relative___identity_frame(),
    )


def test_start_anchored_relative__first_and_final_relative_samples_are_exactly_zero_and_anchor() -> None:
    trajectory, audit, _ = start_anchored_relative___build()
    relative = trajectory[["delta_x_R", "delta_y_R", "delta_z_R"]].to_numpy(float)
    target = trajectory[["tcp_x_base", "tcp_y_base", "tcp_z_base"]].to_numpy(float)
    assert start_anchored_relative__np.array_equal(relative[0], start_anchored_relative__np.zeros(3))
    assert start_anchored_relative__np.array_equal(relative[-1], start_anchored_relative__np.zeros(3))
    assert start_anchored_relative__np.array_equal(target[0], start_anchored_relative__ANCHOR[:3])
    assert start_anchored_relative__np.array_equal(target[-1], start_anchored_relative__ANCHOR[:3])
    assert audit.first_relative_displacement_zero
    assert audit.final_relative_displacement_zero
    assert audit.trajectory_valid


def test_start_anchored_relative__relative_mode_has_no_hip_center_input_or_output_dependency() -> None:
    signature = start_anchored_relative__inspect.signature(start_anchored_relative__build_start_anchored_relative_trajectory)
    assert "hip_center_in_base_m" not in signature.parameters
    trajectory, _, metadata = start_anchored_relative___build()
    assert not any("hip_center" in name for name in trajectory.columns)
    assert metadata["hip_center_required"] is False
    assert metadata["observed_ankle_used_as_pull_point"] is False
    assert metadata["experiment_mode"] == start_anchored_relative__START_ANCHORED_RELATIVE_MODE
    assert metadata["tool_offset_retained_for_mode"] == start_anchored_relative__ABSOLUTE_CALIBRATED_MODE


def test_start_anchored_relative__fixed_tcp_orientation_is_identical_to_anchor_for_every_sample() -> None:
    trajectory, audit, metadata = start_anchored_relative___build()
    orientation = trajectory[["tcp_rx", "tcp_ry", "tcp_rz"]].to_numpy(float)
    expected = start_anchored_relative__np.repeat(start_anchored_relative__ANCHOR[None, 3:], len(trajectory), axis=0)
    assert start_anchored_relative__np.array_equal(orientation, expected)
    assert audit.orientation_constant
    assert metadata["tcp_orientation_strategy"] == "fixed_at_start_anchor"


def test_start_anchored_relative__rehab_x_and_z_map_to_configured_base_directions() -> None:
    frame = start_anchored_relative__RehabFrameConfig(
        rehab_x_axis_in_base=(0.0, 1.0, 0.0),
        rehab_z_axis_in_base=(0.0, 0.0, 1.0),
        reviewed=True,
        notes="axis mapping test",
    )
    rotation = frame.rotation_base_from_rehab
    dx = 0.025
    dz = 0.040
    mapped_x = rotation @ start_anchored_relative__np.asarray((dx, 0.0, 0.0))
    mapped_z = rotation @ start_anchored_relative__np.asarray((0.0, 0.0, dz))
    assert start_anchored_relative__np.allclose(mapped_x, dx * start_anchored_relative__np.asarray(frame.rehab_x_axis_in_base))
    assert start_anchored_relative__np.allclose(mapped_z, dz * start_anchored_relative__np.asarray(frame.rehab_z_axis_in_base))
    assert start_anchored_relative__np.allclose(rotation.T @ rotation, start_anchored_relative__np.eye(3), atol=1e-12)
    assert start_anchored_relative__np.linalg.det(rotation) == start_anchored_relative__pytest.approx(1.0)


def test_start_anchored_relative__frame_rejects_nonunit_or_nonorthogonal_axes() -> None:
    with start_anchored_relative__pytest.raises(ValueError, match="norm approximately one"):
        start_anchored_relative__RehabFrameConfig((2.0, 0.0, 0.0), (0.0, 0.0, 1.0), False)
    with start_anchored_relative__pytest.raises(ValueError, match="approximately orthogonal"):
        start_anchored_relative__RehabFrameConfig((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), False)


def test_start_anchored_relative__unreviewed_frame_can_be_previewed_but_is_not_silently_reviewed() -> None:
    trajectory, audit, metadata = start_anchored_relative___build(start_anchored_relative___identity_frame(reviewed=False))
    assert audit.trajectory_valid
    assert trajectory["trajectory_valid"].all()
    assert metadata["rehab_frame"]["reviewed"] is False
    assert metadata["robot_execution_approved"] is False


def test_start_anchored_relative__template_requires_axes_and_review_is_never_inferred(tmp_path) -> None:
    path = tmp_path / "rehab_frame.json"
    path.write_text(
        start_anchored_relative__json.dumps(
            {
                "rehab_x_axis_in_base": [1.0, 0.0, 0.0],
                "rehab_z_axis_in_base": [0.0, 0.0, 1.0],
                "reviewed": False,
                "notes": "draft",
            }
        ),
        encoding="utf-8",
    )
    frame = start_anchored_relative__load_rehab_frame_config(path)
    assert frame.reviewed is False
    assert frame.notes == "draft"


def test_start_anchored_relative__template_null_axes_report_the_missing_measurements(tmp_path) -> None:
    path = tmp_path / "rehab_frame.json"
    path.write_text(
        start_anchored_relative__json.dumps(
            {
                "rehab_x_axis_in_base": None,
                "rehab_z_axis_in_base": None,
                "reviewed": False,
                "notes": "measure on site",
            }
        ),
        encoding="utf-8",
    )
    with start_anchored_relative__pytest.raises(
        ValueError,
        match="unmeasured axes: rehab_x_axis_in_base, rehab_z_axis_in_base",
    ):
        start_anchored_relative__load_rehab_frame_config(path)


def test_start_anchored_relative__current_slow_reference_keeps_frozen_rom_theta_and_fk() -> None:
    trajectory, audit, metadata = start_anchored_relative___build()
    hip_deg = start_anchored_relative__np.rad2deg(trajectory["q_hip_ref"].to_numpy(float))
    knee_deg = start_anchored_relative__np.rad2deg(trajectory["q_knee_ref"].to_numpy(float))
    assert hip_deg.min() >= start_anchored_relative__APPROVED_HIP_ROM_DEG[0] - 1e-10
    assert hip_deg.max() <= start_anchored_relative__APPROVED_HIP_ROM_DEG[1] + 1e-10
    assert knee_deg.min() >= start_anchored_relative__APPROVED_KNEE_ROM_DEG[0] - 1e-10
    assert knee_deg.max() <= start_anchored_relative__APPROVED_KNEE_ROM_DEG[1] + 1e-10
    assert start_anchored_relative__np.allclose(
        trajectory["theta_shank_ref"],
        trajectory["q_hip_ref"] - trajectory["q_knee_ref"],
        atol=1e-14,
        rtol=0.0,
    )
    assert audit.approved_rom_valid
    assert audit.theta_shank_definition_valid
    assert audit.pull_forward_kinematics_valid
    assert metadata["trajectory_id"] == start_anchored_relative__FIRST_ROBOT_TRIAL_TRAJECTORY_ID
    assert metadata["allowed_for_first_robot_trial"] is False
    assert metadata["first_robot_trial_candidate_whitelisted"] is True
    assert metadata["reference"]["sha256"] == (
        start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
    )
    assert metadata["reference_sha256_matches_approved_first_trial"] is False
    assert metadata["reference_sha256_matches_frozen_release"] is True
    assert start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256 == (
        "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    )
    geometry = metadata["equivalent_pull_point_geometry"]
    assert geometry["L1_hip_to_knee_m"] == start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_L1_M
    assert geometry["L2_knee_to_equivalent_shank_strap_pull_point_m"] == (
        start_anchored_relative__APPROVED_FIRST_ROBOT_TRIAL_L2_M
    )
    assert geometry["matches_approved_first_trial_geometry"] is True


def test_start_anchored_relative__nonclosed_reference_is_invalid_not_silently_forced_closed() -> None:
    source, _ = start_anchored_relative__load_closed_reference_trajectory(start_anchored_relative__DEFAULT_REFERENCE_PATH)
    source.loc[len(source) - 1, "q_hip_rad"] += 0.05
    source.loc[len(source) - 1, "theta_shank_rad"] = (
        source.loc[len(source) - 1, "q_hip_rad"]
        - source.loc[len(source) - 1, "q_knee_rad"]
    )
    trajectory, audit, _ = start_anchored_relative__build_start_anchored_relative_trajectory(
        source,
        current_tcp_start_pose=start_anchored_relative__ANCHOR,
        rehab_frame=start_anchored_relative___identity_frame(),
    )
    assert not start_anchored_relative__np.array_equal(
        trajectory[["delta_x_R", "delta_y_R", "delta_z_R"]].iloc[-1].to_numpy(float),
        start_anchored_relative__np.zeros(3),
    )
    assert not audit.final_relative_displacement_zero
    assert "reference_not_closed" in audit.invalid_reasons
    assert not trajectory["trajectory_valid"].any()


def test_start_anchored_relative__nominal_reference_is_retained_but_not_first_trial_whitelisted() -> None:
    nominal = start_anchored_relative__SOURCE_CANDIDATE_DIRECTORY / "reference_measured_asymmetric_closed_nominal.csv"
    trajectory, audit, metadata = start_anchored_relative__build_start_anchored_relative_trajectory(
        nominal,
        current_tcp_start_pose=start_anchored_relative__ANCHOR,
        rehab_frame=start_anchored_relative___identity_frame(),
    )
    assert not audit.trajectory_valid
    assert "source_formal_gate_invalid" in audit.invalid_reasons
    assert trajectory["trajectory_id"].eq(start_anchored_relative__MEASURED_ASYMMETRIC_NOMINAL_ID).all()
    assert metadata["allowed_for_first_robot_trial"] is False


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
from scripts.run_wrench_stationary_validation import ObservedAdapter as state_frame_capture__ObservedAdapter


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


class state_frame_capture__ListEvidence:
    def __init__(self):
        self.rows = []

    def put(self, row):
        self.rows.append(row)


def test_state_frame_capture__observed_adapter_audits_each_drained_frame_without_latest_ahead_of_batch():
    source = state_frame_capture__ControlledSource()
    source.connect()
    source.start_state_stream()
    evidence = state_frame_capture__ListEvidence()
    observed = state_frame_capture__ObservedAdapter(source, evidence, lambda *args, **kwargs: None,
                               "offline-session", "offline-run", {})
    assert observed.begin_state_capture() is True
    source.publish(2)
    source.publish(3)
    assert observed.read_state_frame().sequence_id == 3
    assert observed.read_state_frame().sequence_id == 3
    assert evidence.rows == [], "a health/latest read must not jump audit ahead of buffered frames"
    assert tuple(item.sequence_id for item in observed.drain_state_frames()) == (1, 2, 3)
    assert [row["sequence_id"] for row in evidence.rows] == [1, 2, 3]
    assert [row["sequence_gap"] for row in evidence.rows] == [0, 0, 0]
    assert observed.drain_state_frames() == ()
    assert observed.read_state_frame().sequence_id == 3
    assert len(evidence.rows) == 3
    observed.end_state_capture()


def test_state_frame_capture__observed_adapter_legacy_latest_still_deduplicates_audit():
    legacy = state_frame_capture__RokaeRobotAdapter(native_robot=state_frame_capture__LegacyNative())
    evidence = state_frame_capture__ListEvidence()
    observed = state_frame_capture__ObservedAdapter(legacy, evidence, lambda *args, **kwargs: None,
                               "offline-session", "offline-run", {})
    assert observed.begin_state_capture() is False
    observed.read_state_frame()
    observed.read_state_frame()
    assert [row["sequence_id"] for row in evidence.rows] == [7]


# ============================================================================
# Source: tests/test_state_wrench_timing_audit.py
# ============================================================================



import argparse as state_wrench_timing_audit__argparse
import json as state_wrench_timing_audit__json
import time as state_wrench_timing_audit__time

import pytest as state_wrench_timing_audit__pytest

from scripts.audit_state_wrench_timing import (
    NOT_DEFINED as state_wrench_timing_audit__NOT_DEFINED,
    UNDEFINED as state_wrench_timing_audit__UNDEFINED,
    build_gates as state_wrench_timing_audit__build_gates,
    consecutive_error_runs as state_wrench_timing_audit__consecutive_error_runs,
    error_histogram as state_wrench_timing_audit__error_histogram,
    error_timeline as state_wrench_timing_audit__error_timeline,
    metric_summary as state_wrench_timing_audit__metric_summary,
    run_compare as state_wrench_timing_audit__run_compare,
)
from scripts.wrench_process_isolation import RtProcessSupervisor as state_wrench_timing_audit__RtProcessSupervisor, WrenchProcessSupervisor as state_wrench_timing_audit__WrenchProcessSupervisor


def test_state_wrench_timing_audit__metric_summary_has_exact_median_and_percentiles() -> None:
    result = state_wrench_timing_audit__metric_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == {
        "count": 5,
        "mean": 3.0,
        "median": 3.0,
        "p95": state_wrench_timing_audit__pytest.approx(4.8),
        "p99": state_wrench_timing_audit__pytest.approx(4.96),
        "max": 5.0,
    }


def test_state_wrench_timing_audit__error_263_timeline_and_consecutive_runs_are_not_suppressed() -> None:
    events = [
        {"sequence_id": 1, "call_start_ns": 0, "call_end_ns": 1_000_000, "query_latency_ms": 1.0, "success": True, "error_code": None},
        {"sequence_id": 2, "call_start_ns": 20_000_000, "call_end_ns": 30_000_000, "query_latency_ms": 10.0, "success": False, "error_code": 263, "error_message": "failure", "operation_state": "IDLE"},
        {"sequence_id": 3, "call_start_ns": 40_000_000, "call_end_ns": 45_000_000, "query_latency_ms": 5.0, "success": False, "error_code": 263, "error_message": "failure", "operation_state": "IDLE"},
        {"sequence_id": 4, "call_start_ns": 60_000_000, "call_end_ns": 61_000_000, "query_latency_ms": 1.0, "success": True, "error_code": None},
    ]
    assert state_wrench_timing_audit__error_histogram(events) == {"263": 2}
    timeline = state_wrench_timing_audit__error_timeline(events)
    assert [item["error_code"] for item in timeline] == [263, 263]
    assert all(item["recovered"] for item in timeline)
    runs = state_wrench_timing_audit__consecutive_error_runs(events)
    assert runs == [{
        "first_sequence_id": 2,
        "last_sequence_id": 3,
        "count": 2,
        "onset_ns": 20_000_000,
        "end_ns": 45_000_000,
        "duration_ms": 25.0,
        "codes": [263, 263],
        "recovered": True,
        "recovery_sequence_id": 4,
        "recovery_timestamp_ns": 61_000_000,
    }]


def test_state_wrench_timing_audit__missing_formal_timing_thresholds_remain_undefined() -> None:
    summary = {
        "completion": {"completed_requested_duration": True},
        "rt_source": {"ring_overwrite_count": 0, "sequence_gap_count": 0},
        "wrench": {"event_drop_count": 0, "formal_stale_count": state_wrench_timing_audit__NOT_DEFINED},
        "state_transitions": {"before": "idle", "after": "idle", "non_idle_transitions": []},
        "workers": {
            "rt_hung_event_count": 0, "rt_crash_event_count": 0,
            "wrench_hung_event_count": 0, "wrench_crash_event_count": 0,
            "rt_cleanup": {"worker_exited": True, "graceful_disconnect_confirmed": True},
            "wrench_cleanup": {"worker_exited": True, "graceful_disconnect_confirmed": True},
        },
        "rt_ipc": {"formal_stale_count": state_wrench_timing_audit__NOT_DEFINED},
        "supervisor": {"formal_late_cycle_count": state_wrench_timing_audit__NOT_DEFINED},
    }
    gates = state_wrench_timing_audit__build_gates(
        case="test_b",
        summary=summary,
        safety={"max_state_age_s": None, "max_wrench_age_s": None, "max_command_lateness_s": None},
    )
    assert gates["data_integrity"]["status"] == "PASS"
    assert gates["operation_state_stability"]["status"] == "PASS"
    assert gates["rt_source_timing"]["status"] == state_wrench_timing_audit__UNDEFINED
    assert gates["rt_ipc_freshness"]["evidence"]["threshold"] == state_wrench_timing_audit__NOT_DEFINED
    assert gates["wrench_error_reliability"]["status"] == state_wrench_timing_audit__UNDEFINED


def test_state_wrench_timing_audit__rt_source_ring_is_contiguous_and_nonblocking() -> None:
    rt = state_wrench_timing_audit__RtProcessSupervisor(stale_age_ms=None)
    try:
        rt.start({"offline": True, "publish_hz": 125.0})
        deadline = state_wrench_timing_audit__time.monotonic() + 3.0
        while state_wrench_timing_audit__time.monotonic() < deadline and not rt.poll().rt_valid:
            state_wrench_timing_audit__time.sleep(0.01)
        state_wrench_timing_audit__time.sleep(0.15)
        events = rt.drain_source_events()
        assert len(events) >= 10
        sequences = [event["rt_sequence"] for event in events]
        assert sequences == list(range(sequences[0], sequences[-1] + 1))
        assert rt.source_ring_overwrite_count == 0
        cleanup = rt.stop_normally(3.0)
        assert cleanup["worker_exited"]
    finally:
        if rt.alive:
            rt.terminate()
        rt.close()


def test_state_wrench_timing_audit__wrench_event_stream_preserves_every_error_263_result() -> None:
    wrench = state_wrench_timing_audit__WrenchProcessSupervisor(stale_age_ms=None)
    events = []
    try:
        wrench.start(mode="offline", config={
            "behavior": "error263", "delay_s": 0.001, "target_hz": 50.0,
            "event_queue_size": 256,
        })
        deadline = state_wrench_timing_audit__time.monotonic() + 0.4
        while state_wrench_timing_audit__time.monotonic() < deadline:
            wrench.poll()
            events.extend(wrench.drain_events())
            state_wrench_timing_audit__time.sleep(0.005)
        cleanup = wrench.stop_normally(3.0)
        events.extend(wrench.drain_events())
        assert cleanup["worker_exited"]
        assert len(events) >= 5
        assert all(event["error_code"] == 263 for event in events)
        assert [event["sequence_id"] for event in events] == list(range(1, len(events) + 1))
        assert wrench.metadata["event_drop_count"] == 0
        assert wrench.metadata["failure_count"] == len(events)
    finally:
        if wrench.alive:
            wrench.terminate()
        wrench.close()


def test_state_wrench_timing_audit__comparison_report_generation_is_deterministic(tmp_path) -> None:
    def payload(case: str, offset: float) -> dict:
        return {
            "case": case,
            "completion": {
                "completed_requested_duration": True,
                "observed_duration_s": 10.0,
                "requested_duration_s": 10.0,
                "fatal_error": None,
            },
            "rt_source": {"interval_ms": {"p99": 8.0 + offset, "max": 9.0 + offset}},
            "rt_ipc": {"age_ms": {"p95": 7.0 + offset, "p99": 8.0 + offset, "max": 10.0 + offset}},
            "supervisor": {"loop_interval_ms": {"p99": 11.0 + offset, "max": 20.0 + offset}},
            "wrench": {"request_count": 0 if case == "test_a" else 100, "failure_count": 0, "error_code_histogram": {}, "error_count_263": 0},
        }

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(state_wrench_timing_audit__json.dumps(payload("test_a", 0.0)), encoding="utf-8")
    b_path.write_text(state_wrench_timing_audit__json.dumps(payload("test_b", 1.0)), encoding="utf-8")
    result = state_wrench_timing_audit__run_compare(state_wrench_timing_audit__argparse.Namespace(
        test_a_summary=str(a_path), test_b_summary=str(b_path), output_dir=str(tmp_path)
    ))
    assert result["comparison_gate"]["status"] == state_wrench_timing_audit__UNDEFINED
    assert result["descriptive_delta"]["rt_ipc_p99_ms"] == 1.0
    assert all((tmp_path / name).exists() for name in (
        next(path.name for path in tmp_path.glob("state_wrench_timing_comparison_*.json")),
        next(path.name for path in tmp_path.glob("state_wrench_timing_comparison_*.md")),
        next(path.name for path in tmp_path.glob("state_wrench_timing_comparison_*.png")),
    ))


# ============================================================================
# Source: tests/test_static_validation_logging.py
# ============================================================================

"""Offline tests for the default-off static validation label sidecar."""



import ast as static_validation_logging__ast
from copy import deepcopy as static_validation_logging__deepcopy
import csv as static_validation_logging__csv
import json as static_validation_logging__json
from pathlib import Path as static_validation_logging__Path

import pytest as static_validation_logging__pytest

from scripts.static_validation_logging import (
    CSV_FIELDS as static_validation_logging__CSV_FIELDS,
    DEFAULT_CONFIG_PATH as static_validation_logging__DEFAULT_CONFIG_PATH,
    DEFAULT_ENABLED as static_validation_logging__DEFAULT_ENABLED,
    DEFAULT_SCHEMA_PATH as static_validation_logging__DEFAULT_SCHEMA_PATH,
    PHASES as static_validation_logging__PHASES,
    SCHEMA_VERSION as static_validation_logging__SCHEMA_VERSION,
    StaticValidationCell as static_validation_logging__StaticValidationCell,
    StaticValidationLabelLogger as static_validation_logging__StaticValidationLabelLogger,
    StaticValidationLoggingDisabled as static_validation_logging__StaticValidationLoggingDisabled,
    StaticValidationLoggingError as static_validation_logging__StaticValidationLoggingError,
    build_static_validation_record as static_validation_logging__build_static_validation_record,
    load_static_validation_logging_config as static_validation_logging__load_static_validation_logging_config,
)
from scripts.dry_run_static_validation_logging import run_dry_run as static_validation_logging__run_dry_run


static_validation_logging__PROTOCOL_SHA = "c88799b838f6304765acb643a706b1a6f1bbe02b1ee4f6c07ed9c486eab2f5c1"
static_validation_logging__ROOT = static_validation_logging__Path(__file__).resolve().parents[1]


def static_validation_logging__cell(*, repeat_id: int = 1, direction_id: str = "+WORLD_X") -> static_validation_logging__StaticValidationCell:
    return static_validation_logging__StaticValidationCell(
        session_id="STATIC_TEST_SESSION",
        protocol_sha256=static_validation_logging__PROTOCOL_SHA,
        pose_id="P0_CURRENT_SAFE_STATIONARY",
        direction_id=direction_id,
        load_level_id="L1_REVIEWED_LOW",
        repeat_id=repeat_id,
    )


def static_validation_logging__wrench(*, start: float = 10.0, end: float = 10.004) -> dict:
    return {
        "force_query_started_s": start,
        "force_query_finished_s": end,
        "host_monotonic_time_s": (start + end) / 2.0,
        "cartesian_force_raw_n": [1.2, -0.3, 0.4],
        "raw_force_frame": "world",
        "valid": True,
        "invalid_reason": "",
    }


def static_validation_logging__state(*, sequence_id: int = 7) -> dict:
    return {
        "host_monotonic_time_s": 10.005,
        "sequence_id": sequence_id,
        "tcp_position_m": [0.31, 0.02, 0.44],
        "tcp_orientation_rad": [0.01, 0.02, 0.03],
        "joint_position_rad": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "operation_state": "stationary",
        "valid": True,
        "invalid_reason": "",
    }


def static_validation_logging__metadata(*, verified: bool = True) -> dict:
    return {
        "robot_model": "mock-model",
        "robot_serial_number": "mock-serial",
        "controller_version": "mock-controller",
        "active_tool_name": "mock-tool",
        "active_workobject_name": "mock-workobject",
        "sdk_tool_payload": {
            "active_hmi_tool_workobject_verified": verified,
            "toolset_load_mass_kg": 1.5,
            "toolset_load_cog_m": [0.0, 0.0, 0.05],
            "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
            "toolset_end_translation_m": [0.0, 0.0, 0.1],
            "toolset_end_rpy_rad": [0.0, 0.0, 0.0],
            "sdk_available_tool_names": ["mock-tool"],
            "sdk_available_workobject_names": ["mock-workobject"],
        },
    }


def static_validation_logging__build(**overrides) -> dict:
    arguments = {
        "cell": static_validation_logging__cell(),
        "phase": "PRE",
        "phase_sample_index": 0,
        "raw_measurement_source": "robot_wrench.csv",
        "raw_measurement_id": "raw-wrench-0001",
        "wrench_sample": static_validation_logging__wrench(),
        "robot_state": static_validation_logging__state(),
        "robot_metadata": static_validation_logging__metadata(),
    }
    arguments.update(overrides)
    return static_validation_logging__build_static_validation_record(**arguments)


def static_validation_logging__read_rows(path: static_validation_logging__Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(static_validation_logging__csv.DictReader(stream))


def static_validation_logging__make_logger(path: static_validation_logging__Path, *, enabled: bool) -> static_validation_logging__StaticValidationLabelLogger:
    return static_validation_logging__StaticValidationLabelLogger(
        path,
        session_id="STATIC_TEST_SESSION",
        protocol_sha256=static_validation_logging__PROTOCOL_SHA,
        raw_measurement_source="robot_wrench.csv",
        enabled=enabled,
        run_metadata={"evidence_role": "OFFLINE_TEST_ONLY"},
    )


def test_static_validation_logging__default_off_identity_is_redundantly_explicit() -> None:
    config = static_validation_logging__load_static_validation_logging_config()
    assert static_validation_logging__DEFAULT_ENABLED is False
    assert type(config["enabled"]) is bool
    assert config["schema_version"] == 1
    assert static_validation_logging__Path(config["record_schema"]) == section_path("static_validation_record_schema_v1")
    assert "does not authorize robot connection" in config["notes"]


def test_static_validation_logging__disabled_logger_creates_no_directory_or_file(tmp_path: static_validation_logging__Path) -> None:
    output = tmp_path / "disabled"
    logger = static_validation_logging__make_logger(output, enabled=False)
    assert logger.active is False
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingDisabled, match="default-off"):
        logger.start()
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingDisabled, match="default-off"):
        logger.append(
            cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=0,
            raw_measurement_id="raw-1", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    assert not output.exists()


def test_static_validation_logging__schema_required_fields_exactly_match_csv_fields() -> None:
    schema = static_validation_logging__json.loads(static_validation_logging__DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == static_validation_logging__SCHEMA_VERSION
    assert set(schema["required"]) == set(static_validation_logging__CSV_FIELDS)
    assert set(schema["properties"]) == set(static_validation_logging__CSV_FIELDS)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["phase"]["enum"] == list(static_validation_logging__PHASES)


def test_static_validation_logging__schema_contains_every_requested_identity_measurement_and_status_field() -> None:
    required = set(static_validation_logging__json.loads(static_validation_logging__DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))["required"])
    assert {
        "session_id", "protocol_sha256", "pose_id", "direction_id",
        "load_level_id", "repeat_id", "phase", "query_start_s",
        "query_end_s", "query_midpoint_s", "query_latency_ms",
        "fx_raw_n", "fy_raw_n", "fz_raw_n", "valid", "status",
    } <= required
    assert {
        "tcp_x_m", "tcp_y_m", "tcp_z_m", "q1_rad", "q6_rad",
        "robot_operation_state", "active_tool_name", "tcp_translation_m_json",
        "payload_mass_kg", "payload_cog_m_json",
    } <= required


def test_static_validation_logging__valid_record_copies_labels_timing_wrench_state_and_metadata() -> None:
    row = static_validation_logging__build()
    assert row["session_id"] == "STATIC_TEST_SESSION"
    assert row["protocol_sha256"] == static_validation_logging__PROTOCOL_SHA
    assert row["pose_id"] == "P0_CURRENT_SAFE_STATIONARY"
    assert row["direction_id"] == "+WORLD_X"
    assert row["load_level_id"] == "L1_REVIEWED_LOW"
    assert row["repeat_id"] == 1 and row["phase"] == "PRE"
    assert row["query_start_s"] == 10.0
    assert row["query_midpoint_s"] == static_validation_logging__pytest.approx(10.002)
    assert row["query_end_s"] == 10.004
    assert row["query_latency_ms"] == static_validation_logging__pytest.approx(4.0)
    assert (row["fx_raw_n"], row["fy_raw_n"], row["fz_raw_n"]) == (1.2, -0.3, 0.4)
    assert row["tcp_x_m"] == 0.31 and row["q6_rad"] == 0.5
    assert row["active_tool_name"] == "mock-tool"
    assert row["payload_mass_kg"] == 1.5
    assert row["valid"] is True and row["status"] == "VALID"


def test_static_validation_logging__midpoint_and_latency_use_existing_query_bounds_without_new_clock() -> None:
    sample = static_validation_logging__wrench(start=21.0, end=21.008)
    sample.pop("host_monotonic_time_s")
    row = static_validation_logging__build(wrench_sample=sample)
    assert row["query_midpoint_s"] == static_validation_logging__pytest.approx(21.004)
    assert row["query_latency_ms"] == static_validation_logging__pytest.approx(8.0)
    source = static_validation_logging__Path(static_validation_logging__ROOT / "scripts/static_validation_logging.py").read_text(encoding="utf-8")
    assert "perf_counter" not in source
    assert "time.time" not in source


def test_static_validation_logging__timestamp_order_and_midpoint_consistency_fail_closed() -> None:
    reversed_row = static_validation_logging__build(wrench_sample=static_validation_logging__wrench(start=10.1, end=10.0))
    assert reversed_row["valid"] is False
    assert reversed_row["query_midpoint_s"] is None
    assert reversed_row["query_latency_ms"] is None
    assert "query_timestamp_order_invalid" in reversed_row["invalid_reason"]

    inconsistent = static_validation_logging__wrench()
    inconsistent["host_monotonic_time_s"] = 99.0
    inconsistent_row = static_validation_logging__build(wrench_sample=inconsistent)
    assert inconsistent_row["valid"] is False
    assert "query_midpoint_inconsistent" in inconsistent_row["invalid_reason"]


def test_static_validation_logging__phase_labels_accept_only_pre_load_post() -> None:
    assert [static_validation_logging__build(phase=phase)["phase"] for phase in static_validation_logging__PHASES] == ["PRE", "LOAD", "POST"]
    with static_validation_logging__pytest.raises(ValueError, match="phase must be one of"):
        static_validation_logging__build(phase="BASELINE")
    with static_validation_logging__pytest.raises(ValueError, match="phase must be one of"):
        static_validation_logging__build(phase="PRE_LOAD_ZERO")


def test_static_validation_logging__cell_identity_is_stable_and_unique_across_repeat_or_direction() -> None:
    first = static_validation_logging__cell()
    same = static_validation_logging__cell()
    next_repeat = static_validation_logging__cell(repeat_id=2)
    next_direction = static_validation_logging__cell(direction_id="-WORLD_X")
    assert first.cell_id == same.cell_id
    assert len({first.cell_id, next_repeat.cell_id, next_direction.cell_id}) == 3
    assert all(len(value) == 64 for value in {first.cell_id, next_repeat.cell_id, next_direction.cell_id})


def test_static_validation_logging__record_identity_links_one_raw_measurement_to_one_label() -> None:
    row = static_validation_logging__build(raw_measurement_id="raw-wrench-0042")
    assert row["raw_measurement_source"] == "robot_wrench.csv"
    assert row["raw_measurement_id"] == "raw-wrench-0042"
    assert len(row["record_id"]) == 64
    assert len(row["cell_id"]) == 64
    changed = static_validation_logging__build(raw_measurement_id="raw-wrench-0043")
    assert row["record_id"] != changed["record_id"]


def test_static_validation_logging__raw_wrench_and_state_inputs_are_not_mutated() -> None:
    raw_wrench = static_validation_logging__wrench()
    raw_state = static_validation_logging__state()
    raw_metadata = static_validation_logging__metadata()
    originals = static_validation_logging__deepcopy((raw_wrench, raw_state, raw_metadata))
    static_validation_logging__build(wrench_sample=raw_wrench, robot_state=raw_state, robot_metadata=raw_metadata)
    assert (raw_wrench, raw_state, raw_metadata) == originals


def test_static_validation_logging__missing_numeric_data_remains_none_and_record_is_invalid() -> None:
    row = static_validation_logging__build(
        wrench_sample={
            "force_query_started_s": None,
            "force_query_finished_s": None,
            "cartesian_force_raw_n": [None, float("nan"), None],
            "raw_force_frame": "world",
        },
        robot_state=None,
        robot_metadata=None,
    )
    assert row["query_start_s"] is None
    assert row["query_end_s"] is None
    assert row["query_midpoint_s"] is None
    assert row["query_latency_ms"] is None
    assert row["fx_raw_n"] is None and row["fy_raw_n"] is None and row["fz_raw_n"] is None
    assert row["tcp_x_m"] is None and row["q1_rad"] is None
    assert row["valid"] is False and row["status"] == "INVALID"
    for reason in ("missing_query_timestamp", "raw_force_missing_or_nonfinite", "robot_state_missing", "robot_metadata_missing"):
        assert reason in row["invalid_reason"]


def test_static_validation_logging__unverified_active_tool_metadata_fails_closed_without_fake_name() -> None:
    robot_metadata = static_validation_logging__metadata(verified=False)
    robot_metadata["active_tool_name"] = None
    row = static_validation_logging__build(robot_metadata=robot_metadata)
    assert row["active_tool_name"] is None
    assert row["active_hmi_tool_workobject_verified"] is False
    assert row["valid"] is False
    assert "active_tool_workobject_not_verified" in row["invalid_reason"]


def test_static_validation_logging__identity_fields_are_required_instead_of_written_as_ambiguous_blanks() -> None:
    with static_validation_logging__pytest.raises(ValueError, match="session_id"):
        static_validation_logging__StaticValidationCell("", static_validation_logging__PROTOCOL_SHA, "P0", "+X", "L1", 1)
    with static_validation_logging__pytest.raises(ValueError, match="protocol_sha256"):
        static_validation_logging__StaticValidationCell("S", "not-a-sha", "P0", "+X", "L1", 1)
    with static_validation_logging__pytest.raises(ValueError, match="repeat_id"):
        static_validation_logging__StaticValidationCell("S", static_validation_logging__PROTOCOL_SHA, "P0", "+X", "L1", 0)
    with static_validation_logging__pytest.raises(ValueError, match="raw_measurement_id"):
        static_validation_logging__build(raw_measurement_id="")


def test_static_validation_logging__logger_enforces_pre_load_post_order_per_cell(tmp_path: static_validation_logging__Path) -> None:
    logger = static_validation_logging__make_logger(tmp_path / "ordered", enabled=True).start()
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="first phase.*PRE"):
        logger.append(
            cell=static_validation_logging__cell(), phase="LOAD", phase_sample_index=0,
            raw_measurement_id="bad-load-first", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    logger.append(
        cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="pre-0", wrench_sample=static_validation_logging__wrench(),
        robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
    )
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="without reversal or skip"):
        logger.append(
            cell=static_validation_logging__cell(), phase="POST", phase_sample_index=0,
            raw_measurement_id="post-skips-load", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    logger.append(
        cell=static_validation_logging__cell(), phase="LOAD", phase_sample_index=0,
        raw_measurement_id="load-0", wrench_sample=static_validation_logging__wrench(),
        robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
    )
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="without reversal or skip"):
        logger.append(
            cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=1,
            raw_measurement_id="pre-after-load", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    logger.append(
        cell=static_validation_logging__cell(), phase="POST", phase_sample_index=0,
        raw_measurement_id="post-0", wrench_sample=static_validation_logging__wrench(),
        robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
    )
    logger.close()
    assert [row["phase"] for row in static_validation_logging__read_rows(logger.label_path)] == ["PRE", "LOAD", "POST"]


def test_static_validation_logging__logger_rejects_duplicate_phase_index_record_and_raw_link(tmp_path: static_validation_logging__Path) -> None:
    logger = static_validation_logging__make_logger(tmp_path / "duplicates", enabled=True).start()
    logger.append(
        cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="raw-0", wrench_sample=static_validation_logging__wrench(),
        robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
    )
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="duplicate phase_sample_index"):
        logger.append(
            cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=0,
            raw_measurement_id="raw-1", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="raw measurement is already linked"):
        logger.append(
            cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=1,
            raw_measurement_id="raw-0", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    logger.close()


def test_static_validation_logging__logger_writes_missing_values_as_blank_not_zero(tmp_path: static_validation_logging__Path) -> None:
    logger = static_validation_logging__make_logger(tmp_path / "missing", enabled=True).start()
    logger.append(
        cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="missing-0",
        wrench_sample={"force_query_started_s": None, "cartesian_force_raw_n": None},
        robot_state=None,
        robot_metadata=None,
    )
    logger.close()
    row = static_validation_logging__read_rows(logger.label_path)[0]
    for field in ("query_start_s", "query_end_s", "query_midpoint_s", "query_latency_ms", "fx_raw_n", "fy_raw_n", "fz_raw_n", "tcp_x_m", "q1_rad"):
        assert row[field] == ""
    assert row["valid"] == "False"
    assert row["status"] == "INVALID"
    assert row["invalid_reason"]


def test_static_validation_logging__logger_metadata_links_sidecar_to_source_and_never_claims_robot_action(tmp_path: static_validation_logging__Path) -> None:
    logger = static_validation_logging__make_logger(tmp_path / "metadata", enabled=True).start()
    logger.append(
        cell=static_validation_logging__cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="raw-0", wrench_sample=static_validation_logging__wrench(),
        robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
    )
    logger.close()
    payload = static_validation_logging__json.loads(logger.metadata_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "STATIC_TEST_SESSION"
    assert payload["protocol_sha256"] == static_validation_logging__PROTOCOL_SHA
    assert payload["raw_measurement_source"] == "robot_wrench.csv"
    assert payload["label_file"] == "static_validation_labels.csv"
    assert len(payload["label_file_sha256"]) == 64
    assert payload["raw_wrench_overwritten"] is False
    assert payload["robot_connected_by_logger"] is False
    assert payload["robot_action_count"] == 0
    assert payload["control_or_safety_modified"] is False


def test_static_validation_logging__output_collision_never_overwrites_existing_raw_or_label_files(tmp_path: static_validation_logging__Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "robot_wrench.csv"
    protected.write_text("do-not-overwrite\n", encoding="utf-8")
    logger = static_validation_logging__make_logger(output, enabled=True)
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="never overwritten"):
        logger.start()
    assert protected.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_static_validation_logging__cell_must_match_logger_session_and_protocol(tmp_path: static_validation_logging__Path) -> None:
    logger = static_validation_logging__make_logger(tmp_path / "identity", enabled=True).start()
    wrong = static_validation_logging__StaticValidationCell("OTHER_SESSION", static_validation_logging__PROTOCOL_SHA, "P0", "+X", "L1", 1)
    with static_validation_logging__pytest.raises(static_validation_logging__StaticValidationLoggingError, match="does not match logger"):
        logger.append(
            cell=wrong, phase="PRE", phase_sample_index=0,
            raw_measurement_id="raw-0", wrench_sample=static_validation_logging__wrench(),
            robot_state=static_validation_logging__state(), robot_metadata=static_validation_logging__metadata(),
        )
    logger.close()


def test_static_validation_logging__dry_run_uses_synthetic_data_and_passes_all_layer_checks(tmp_path: static_validation_logging__Path) -> None:
    result = static_validation_logging__run_dry_run(tmp_path / "dry_run")
    assert result["dry_run_status"] == "PASS"
    assert result["evidence_role"] == "OFFLINE_SYNTHETIC_DRY_RUN_ONLY"
    assert result["default_off_verified"] is True
    assert result["schema_complete"] is True
    assert result["row_count"] == 12
    assert result["cell_count"] == 2
    assert result["record_ids_unique"] is True
    assert result["cell_ids_unique_by_repeat"] is True
    assert result["phase_sequences_valid"] is True
    assert result["timestamp_order_valid"] is True
    assert result["raw_input_unchanged"] is True
    assert result["missing_value_fail_closed"] is True
    assert result["robot_constructed_or_connected"] is False
    assert result["robot_action_count"] == 0
    assert result["physical_load_count"] == 0
    assert result["human_data_count"] == 0


def test_static_validation_logging__dry_run_script_is_default_off_without_explicit_run_flag() -> None:
    source = (static_validation_logging__ROOT / "scripts/dry_run_static_validation_logging.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--run", action="store_true"' in source
    assert "if not args.run:" in source
    assert "no files written" in source


def test_static_validation_logging__new_layer_has_no_hardware_control_safety_or_robot_action_imports() -> None:
    path = static_validation_logging__ROOT / "scripts/static_validation_logging.py"
    tree = static_validation_logging__ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in static_validation_logging__ast.walk(tree):
        if isinstance(node, static_validation_logging__ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, static_validation_logging__ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, static_validation_logging__ast.Call):
            if isinstance(node.func, static_validation_logging__ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, static_validation_logging__ast.Attribute):
                called.add(node.func.attr)
    assert not any(name.startswith(("hardware", "control", "safety")) for name in imported)
    assert not {"connect", "enable", "move", "drag", "stop", "calibrate_force_sensors"}.intersection(called)


def test_static_validation_logging__existing_collection_and_control_paths_do_not_import_new_layer() -> None:
    for relative in (
        "collection/collector.py",
        "collection/episode_logger.py",
        "collection/real_robot_acquisition.py",
        "control/robot_trajectory_executor.py",
    ):
        text = (static_validation_logging__ROOT / relative).read_text(encoding="utf-8")
        assert "static_validation_logging" not in text


def test_static_validation_logging__config_adds_no_force_or_safety_threshold() -> None:
    config = static_validation_logging__json.loads(section_path("static_validation_logging").read_text(encoding="utf-8"))
    assert set(config) == {"enabled", "notes", "record_schema", "schema_version"}
    assert not any("force" in key.lower() or "threshold" in key.lower() or "limit" in key.lower() for key in config)


# ============================================================================
# Source: tests/test_stationary_identity_contract.py
# ============================================================================

"""Offline contract tests: real metadata shape, no SDK or robot connection."""
from types import SimpleNamespace as stationary_identity_contract__SimpleNamespace

import pytest as stationary_identity_contract__pytest

from collection.real_robot_acquisition import RealRobotAcquisition as stationary_identity_contract__RealRobotAcquisition
from hardware.windows.rokae_xcore import RokaeRobot as stationary_identity_contract__RokaeRobot
from scripts import run_wrench_stationary_validation as stationary_identity_contract__runner
from scripts.prepare_wrench_stationary_validation import prepare as stationary_identity_contract__prepare


def stationary_identity_contract__metadata_from_native_contract(sdk_version):
    native = stationary_identity_contract__SimpleNamespace(
        ip_address="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE",
        _robot_info=stationary_identity_contract__SimpleNamespace(type="fixture-model", id="fixture-serial", version="fixture-controller"),
        _sdk_version=sdk_version, state_interval_ms=8,
        _joint_soft_limits_rad=None, _joint_soft_limit_error=None,
        collision_source="events", get_safety_event_status=lambda: {},
        get_tool_payload_metadata=lambda: {},
    )
    # Exercise the actual producer of the metadata key, without constructing SDK objects.
    result = stationary_identity_contract__RokaeRobot.get_robot_metadata(native)
    assert "xcore_sdk_version" in result and "sdk_version" not in result
    return result


def stationary_identity_contract__expected_identity():
    return dict(robot_model="fixture-model", robot_serial="fixture-serial",
                controller_version="fixture-controller", sdk_version="fixture-sdk")


@stationary_identity_contract__pytest.fixture(autouse=False)
def stationary_identity_contract__forbid_native_sdk_and_network(monkeypatch):
    import socket
    import sys
    from hardware.windows import rokae_xcore
    def forbidden(*args, **kwargs):
        raise AssertionError("native SDK/network forbidden in identity contract tests")
    monkeypatch.setattr(rokae_xcore, "_load_sdk", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert "xCoreSDK_python" not in sys.modules
    yield
    assert "xCoreSDK_python" not in sys.modules


@stationary_identity_contract__pytest.mark.parametrize("sdk_version,accepted", [("fixture-sdk", True), ("wrong-sdk", False), (None, False)])
@stationary_identity_contract__pytest.mark.usefixtures('stationary_identity_contract__forbid_native_sdk_and_network')
def test_stationary_identity_contract__stationary_parent_uses_native_sdk_metadata(tmp_path, monkeypatch, sdk_version, accepted):
    calls = []

    class FakeAdapter(stationary_identity_contract__runner.OfflineStateAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.native_robot = stationary_identity_contract__SimpleNamespace(
                _rt_controller=None, get_robot_mode=lambda: "IDLE",
                _call=lambda *args: stationary_identity_contract__SimpleNamespace(name="off"),
                _robot=stationary_identity_contract__SimpleNamespace(powerState=lambda: None),
            )
        def read_robot_metadata(self):
            return stationary_identity_contract__metadata_from_native_contract(sdk_version)

    def stop_before_acquisition(self, **kwargs):
        calls.append("identity_verified")
        raise RuntimeError("OFFLINE_FIXTURE_STOP_BEFORE_ACQUISITION")

    original_hashes = stationary_identity_contract__runner.source_hashes()
    monkeypatch.setattr(stationary_identity_contract__runner, "ROOT", tmp_path)
    monkeypatch.setattr(stationary_identity_contract__runner, "source_hashes", lambda: original_hashes)
    # Only this offline fixture bypasses authorization; the real hardware constructor is replaced.
    monkeypatch.setattr(stationary_identity_contract__runner, "authorize", lambda *args, **kwargs: None)
    monkeypatch.setattr("hardware.rokae_adapter.RokaeRobotAdapter", FakeAdapter)
    monkeypatch.setattr(stationary_identity_contract__RealRobotAcquisition, "start", stop_before_acquisition)
    request = stationary_identity_contract__prepare(tmp_path / "request", 50.)
    request.update(
        connection=dict(robot_ip="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE"),
        expected_identity=stationary_identity_contract__expected_identity(), allowed_power_states=["off"],
        budgets=dict.fromkeys(("startup_s", "query_s", "progress_s", "sample_age_s", "ack_s", "log_s",
                               "shutdown_s", "terminate_s", "state_age_s", "skew_s"), 2.),
        lifecycle_budgets=dict.fromkeys(("connect_s", "stop_s", "end_state_s", "disconnect_s", "logger_s"), 2.),
    )
    request["budgets"]["provenance"] = "NOT_SAFETY_THRESHOLDS"
    _, summary = stationary_identity_contract__runner.run_case(request, "A", tmp_path / "outputs", mode="LIVE_STATIONARY",
        authorization={"authorization_id": "OFFLINE_METADATA_CONTRACT_FIXTURE"})
    assert bool(calls) is accepted
    failures = ";".join(summary["failures"])
    assert ("parent_identity_mismatch:sdk_version" not in failures) is accepted
    if accepted:
        assert "OFFLINE_FIXTURE_STOP_BEFORE_ACQUISITION" in failures
    assert not summary["completed_900s"]


@stationary_identity_contract__pytest.mark.parametrize("sdk_version,accepted", [("fixture-sdk", True), ("wrong-sdk", False), (None, False)])
@stationary_identity_contract__pytest.mark.usefixtures('stationary_identity_contract__forbid_native_sdk_and_network')
def test_stationary_identity_contract__acquisition_parent_child_uses_native_sdk_metadata(sdk_version, accepted):
    calls = []
    def stop_before_stream():
        calls.append("identity_verified")
        raise RuntimeError("OFFLINE_FIXTURE_STOP_BEFORE_STREAM")
    adapter = stationary_identity_contract__SimpleNamespace(
        native_robot=stationary_identity_contract__SimpleNamespace(ip_address="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE"),
        is_connected=lambda: True,
        read_robot_metadata=lambda: stationary_identity_contract__metadata_from_native_contract(sdk_version),
        start_state_stream=stop_before_stream, stop_state_stream=lambda: None, disconnect=lambda: None,
    )
    provider = stationary_identity_contract__SimpleNamespace(
        config=dict(mode="live", rate_hz=50., robot_ip="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE",
                    expected_identity=stationary_identity_contract__expected_identity()), process=None,
    )
    logger = stationary_identity_contract__SimpleNamespace(assert_healthy=lambda: None, signal_failure=lambda *args, **kwargs: None,
                             mark_failed=lambda *args, **kwargs: None)
    acquisition = stationary_identity_contract__RealRobotAcquisition(adapter, logger, wrench_provider=provider)
    error = RuntimeError if accepted else PermissionError
    reason = "OFFLINE_FIXTURE_STOP_BEFORE_STREAM" if accepted else "parent_child_identity_binding_mismatch:sdk_version"
    with stationary_identity_contract__pytest.raises(error, match=reason):
        acquisition.start(manage_connection=False)
    assert bool(calls) is accepted


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

from collection.collector import DataCollector as units_and_frames__DataCollector
from collection.safety_guard import SafetyGuard as units_and_frames__SafetyGuard
from collection.snapshot import compose_robot_state_sample as units_and_frames__compose_robot_state_sample
from collection.state import (
    InternalWrenchFrame as units_and_frames__InternalWrenchFrame,
    KinematicStateFrame as units_and_frames__KinematicStateFrame,
    calculate_software_bias as units_and_frames__calculate_software_bias,
    rpy_euler_xyz_rotation_matrix as units_and_frames__rpy_euler_xyz_rotation_matrix,
    rotate_vector as units_and_frames__rotate_vector,
    transform_wrench as units_and_frames__transform_wrench,
)
from collection.trajectory import (
    TrajectoryGeometry as units_and_frames__TrajectoryGeometry,
    project_along_tangent as units_and_frames__project_along_tangent,
)
from config import settings as units_and_frames__settings
from hardware.windows.rokae_internal_wrench import RokaeInternalWrenchSource as units_and_frames__RokaeInternalWrenchSource
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


class units_and_frames___LiveRobot:
    def __init__(self) -> None:
        self.index = 0
        self.stopped = False
        self.collision = False

    def get_state_frame(self) -> units_and_frames__KinematicStateFrame:
        self.index += 1
        now_s = units_and_frames__host_time_s()
        return units_and_frames___state_frame(
            now_s,
            sequence_id=self.index,
            position=(min(0.9, 0.05 * self.index), 0.0, 0.0),
            collision=self.collision,
        )

    def get_robot_metadata(self):
        return {"robot_model": "fake-rokae", "xcore_sdk_version": "0.7.0"}

    def get_joint_soft_limits_rad(self):
        return tuple((-2.0, 2.0) for _ in range(6))

    def get_collision_state(self):
        return self.collision

    def stop(self):
        self.stopped = True


class units_and_frames___LiveWrench:
    @property
    def bias(self):
        return (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)

    def snapshot(self, now_s: float) -> units_and_frames__InternalWrenchFrame:
        return units_and_frames___wrench_frame(now_s, sequence_id=1)


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

    def test_tangent_projection_and_retraced_path_continuity(self):
        geometry = units_and_frames__TrajectoryGeometry(units_and_frames__np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        outbound, _ = geometry.project((0.5, 0.0, 0.0))
        inbound, _ = geometry.project((0.5, 0.0, 0.0), reference_arc_length_m=1.5)
        self.assertEqual(outbound.tangent_base, (1.0, 0.0, 0.0))
        self.assertEqual(inbound.tangent_base, (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(units_and_frames__project_along_tangent((3.0, 0.0, 0.0), outbound.tangent_base), 3.0)
        self.assertAlmostEqual(units_and_frames__project_along_tangent((-2.0, 0.0, 0.0), inbound.tangent_base), 2.0)
        zero = units_and_frames__TrajectoryGeometry(units_and_frames__np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        projection, reason = zero.project((0.0, 0.0, 0.0))
        self.assertIsNone(projection)
        self.assertEqual(reason, "trajectory_zero_length")

    def test_software_bias_and_internal_wrench_raw_corrected_fields(self):
        self.assertEqual(
            units_and_frames__calculate_software_bias([(1, 2, 3, 4, 5, 6), (3, 4, 5, 6, 7, 8)]),
            (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        )
        source = units_and_frames__RokaeInternalWrenchSource(object())
        source._bias = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
        frame = source._frame_from_raw({
            "host_monotonic_time_s": 3.0,
            "cartesian_force_raw_n": [4.0, 5.0, 6.0],
            "cartesian_torque_raw_nm": [0.4, 0.5, 0.6],
            "joint_measured_torque_nm": [1, 2, 3, 4, 5, 6],
            "joint_external_torque_nm": [11, 12, 13, 14, 15, 16],
            "raw_force_frame": "world",
        })
        self.assertEqual(frame.cartesian_force_raw_n, (4.0, 5.0, 6.0))
        self.assertEqual(frame.cartesian_force_corrected_n, (3.0, 4.0, 5.0))
        self.assertEqual(frame.joint_measured_torque_nm[0], 1.0)
        self.assertEqual(frame.joint_external_torque_nm[0], 11.0)

    def test_collector_csv_metadata_and_snapshot_schema(self):
        with units_and_frames__tempfile.TemporaryDirectory() as data_dir:
            with units_and_frames__patch.object(units_and_frames__settings, "DATA_DIR", data_dir):
                collector = units_and_frames__DataCollector(units_and_frames___LiveRobot(), units_and_frames___LiveWrench(), "subject", "session")
                collector.start_episode(units_and_frames__np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
                for _ in range(12):
                    self.assertTrue(collector.record_sample())
                    units_and_frames__time.sleep(0.001)
                output = collector.end_episode()

            with open(output, newline="", encoding="utf-8") as stream:
                row = next(units_and_frames__csv.DictReader(stream))
            for name in (
                "sample_time_s", "state_internal_skew_ms", "joint_measured_torque_1_nm",
                "joint_external_torque_1_nm", "fx_raw_n", "fx_corrected_n", "fx_base_n",
                "force_tangent_n", "force_query_duration_ms", "force_estimate_valid",
            ):
                self.assertIn(name, row)
            self.assertEqual(row["schema_version"], "3")
            self.assertEqual(row["frame"], "base")
            self.assertNotIn("comfort", row)
            with open(output.replace(".csv", ".json"), encoding="utf-8") as stream:
                metadata = units_and_frames__json.load(stream)
            self.assertIn("started_at", metadata)
            self.assertIn("finished_at", metadata)
            self.assertNotIn("comfort_label", metadata)
            self.assertNotIn("pain_label", metadata)

    def test_invalid_episode_is_saved_and_sample_index_does_not_repeat(self):
        class OneInvalidWrench(units_and_frames___LiveWrench):
            def __init__(self):
                self.calls = 0

            def snapshot(self, now_s):
                self.calls += 1
                return units_and_frames___wrench_frame(now_s, valid=self.calls != 1,
                                     reason="simulated_wrench_gap" if self.calls == 1 else "")

        with units_and_frames__tempfile.TemporaryDirectory() as data_dir:
            with units_and_frames__patch.object(units_and_frames__settings, "DATA_DIR", data_dir), units_and_frames__patch.object(
                units_and_frames__settings, "WRITE_INVALID_SAMPLES", False
            ):
                collector = units_and_frames__DataCollector(units_and_frames___LiveRobot(), OneInvalidWrench(), "subject", "session")
                collector.start_episode()
                self.assertFalse(collector.record_sample())
                self.assertTrue(collector.record_sample())
                output = collector.abort_episode("simulated_abort")
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(units_and_frames__csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_index"], "1")
            with open(output.replace(".csv", ".json"), encoding="utf-8") as stream:
                metadata = units_and_frames__json.load(stream)
            self.assertFalse(metadata["completed"])
            self.assertEqual(metadata["stop_reason"], "simulated_abort")

    def test_background_sampler_runs_independently_at_configured_cadence(self):
        with units_and_frames__tempfile.TemporaryDirectory() as data_dir:
            with units_and_frames__patch.object(units_and_frames__settings, "DATA_DIR", data_dir):
                collector = units_and_frames__DataCollector(units_and_frames___LiveRobot(), units_and_frames___LiveWrench(), "subject", "session")
                collector.start_episode()
                collector.start_background_sampling(sample_hz=50.0)
                units_and_frames__time.sleep(0.14)
                output = collector.end_episode()
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(units_and_frames__csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 4)

    def test_stream_failure_and_collision_are_visible_to_safety(self):
        adapter = units_and_frames__RokaeRobot("192.0.2.1")
        adapter.is_connected = True
        adapter.robot_mode = "IDLE"
        adapter._accept_state([0.3, 0.0, 0.4, 0.0, 0.0, 0.0], [0.0] * 6, 1.0)
        adapter._mark_state_error(RuntimeError("simulated stream failure"))
        state = adapter.get_state_frame()
        self.assertFalse(state.valid)
        self.assertIn("robot_state_stream_error", state.invalid_reason)

        robot = units_and_frames___LiveRobot()
        robot.collision = True
        guard = units_and_frames__SafetyGuard(robot=robot, wrench_source=units_and_frames___LiveWrench())
        with units_and_frames__patch.object(units_and_frames__settings, "CONTROLLER_COLLISION_CONFIGURATION_CONFIRMED", True):
            with self.assertRaisesRegex(RuntimeError, "robot_collision"):
                guard.check()
        self.assertTrue(robot.stopped)


if __name__ == "__main__":
    units_and_frames__unittest.main()


# ============================================================================
# Source: tests/test_validate_wrench_process_live.py
# ============================================================================

from scripts.validate_wrench_process_live import summarize as validate_wrench_process_live__summarize


def test_validate_wrench_process_live__live_summary_distinguishes_rt_period_from_consumer_gap() -> None:
    rows = [
        {
            "main_loop_timestamp_ns": 1_000_000_000,
            "loop_period_ms": None,
            "rt_sequence": 10,
            "rt_timestamp_ns": 990_000_000,
            "rt_age_ms": 10.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_sequence": 2,
            "wrench_age_ms": 15.0,
            "wrench_valid": True,
            "wrench_stale": False,
            "worker_alive": True,
            "worker_hung": False,
            "heartbeat_age_ms": 5.0,
            "last_error_code": None,
        },
        {
            "main_loop_timestamp_ns": 1_010_000_000,
            "loop_period_ms": 10.0,
            "rt_sequence": 12,
            "rt_timestamp_ns": 1_006_000_000,
            "rt_age_ms": 4.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_sequence": 3,
            "wrench_age_ms": 5.0,
            "wrench_valid": True,
            "wrench_stale": False,
            "worker_alive": True,
            "worker_hung": False,
            "heartbeat_age_ms": 3.0,
            "last_error_code": None,
        },
    ]
    result = validate_wrench_process_live__summarize(rows)
    assert result["rt_source_period_ms"]["mean"] == 8.0
    assert result["rt_sequence_advance"] == 2
    assert result["wrench_sequence_advance"] == 1
    assert result["main_loop_rate_hz"] == 100.0


# ============================================================================
# Source: tests/test_validate_wrench_process_pair.py
# ============================================================================

import csv as validate_wrench_process_pair__csv

from scripts.validate_wrench_process_live import summarize as validate_wrench_process_pair__summarize
from scripts.validate_wrench_process_pair import CSV_FIELDS as validate_wrench_process_pair__CSV_FIELDS, _row_for_summary as validate_wrench_process_pair___row_for_summary, _run_loop as validate_wrench_process_pair___run_loop
from scripts.wrench_process_isolation import RtProcessSupervisor as validate_wrench_process_pair__RtProcessSupervisor, WrenchProcessSupervisor as validate_wrench_process_pair__WrenchProcessSupervisor


def test_validate_wrench_process_pair__pair_summary_projection_keeps_rt_and_wrench_fault_domains_distinct() -> None:
    row = {
        "main_loop_timestamp_ns": 10,
        "loop_period_ms": 10.0,
        "rt_sequence": 4,
        "rt_timestamp_ns": 8,
        "rt_age_ms": 2.0,
        "rt_valid": True,
        "operation_state": "IDLE",
        "wrench_sequence": 3,
        "wrench_age_ms": 20.0,
        "wrench_valid": True,
        "wrench_stale": False,
        "wrench_worker_alive": True,
        "wrench_worker_hung": False,
        "wrench_heartbeat_age_ms": 5.0,
        "last_error_code": 263,
    }
    projected = validate_wrench_process_pair___row_for_summary(row)
    assert projected["worker_hung"] is False
    assert projected["last_error_code"] == 263
    assert projected["rt_sequence"] == 4


def test_validate_wrench_process_pair__pure_supervisor_keeps_100hz_with_two_offline_processes(tmp_path) -> None:
    rt = validate_wrench_process_pair__RtProcessSupervisor(stale_age_ms=50, worker_hung_ms=500)
    wrench = validate_wrench_process_pair__WrenchProcessSupervisor(stale_age_ms=150, worker_hung_ms=500)
    rt.start({"offline": True, "publish_hz": 125.0})
    wrench.start(mode="offline", config={
        "behavior": "normal", "delay_s": 0.001, "target_hz": 20.0,
    })
    rows = []
    path = tmp_path / "pair.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = validate_wrench_process_pair__csv.DictWriter(stream, fieldnames=validate_wrench_process_pair__CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        validate_wrench_process_pair___run_loop(
            stage="overlap",
            duration_s=1.0,
            rt=rt,
            wrench=wrench,
            rows=rows,
            writer=writer,
            stream=stream,
            stop_on_fault=True,
        )
    wrench_cleanup = wrench.stop_normally()
    rt_cleanup = rt.stop_normally()
    wrench.close()
    rt.close()
    projected = [validate_wrench_process_pair___row_for_summary(row) for row in rows]
    result = validate_wrench_process_pair__summarize(projected)
    assert result["main_loop_rate_hz"] >= 80.0
    assert result["rt_sequence_advance"] > 20
    assert result["wrench_sequence_advance"] > 5
    assert wrench_cleanup["worker_exited"]
    assert rt_cleanup["worker_exited"]


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
# Source: tests/test_wrench_nonblocking_diagnostic.py
# ============================================================================



import threading as wrench_nonblocking_diagnostic__threading
import time as wrench_nonblocking_diagnostic__time

from scripts.wrench_nonblocking_diagnostic import LatestWrenchCache as wrench_nonblocking_diagnostic__LatestWrenchCache, NonBlockingWrenchWorker as wrench_nonblocking_diagnostic__NonBlockingWrenchWorker


def test_wrench_nonblocking_diagnostic__cache_becomes_stale_without_blocking_snapshot() -> None:
    cache = wrench_nonblocking_diagnostic__LatestWrenchCache(stale_threshold_ms=20.0)
    cache.set_source_alive(True)
    cache.publish_success({"force": 1}, 1_000_000_000)
    fresh = cache.snapshot(1_010_000_000)
    stale = cache.snapshot(1_030_000_000)
    assert fresh.valid and not fresh.stale and fresh.age_ms == 10.0
    assert not stale.valid and stale.stale and stale.age_ms == 30.0
    assert stale.source_alive


def test_wrench_nonblocking_diagnostic__blocked_worker_does_not_block_main_snapshot() -> None:
    entered = wrench_nonblocking_diagnostic__threading.Event()
    release = wrench_nonblocking_diagnostic__threading.Event()

    def query():
        entered.set()
        release.wait(timeout=1.0)
        return {"force": 1}

    worker = wrench_nonblocking_diagnostic__NonBlockingWrenchWorker(query, target_hz=50, stale_threshold_ms=20)
    worker.start()
    assert entered.wait(timeout=0.5)
    before = wrench_nonblocking_diagnostic__time.perf_counter()
    snapshot = worker.cache.snapshot()
    assert wrench_nonblocking_diagnostic__time.perf_counter() - before < 0.05
    assert snapshot.query_in_flight and snapshot.source_alive
    release.set()
    worker.request_stop()
    assert worker.join(timeout=1.0)


def test_wrench_nonblocking_diagnostic__worker_exception_is_visible_and_recoverable() -> None:
    calls = 0

    def query():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("xCoreSDK getEndTorque failed (263): synthetic")
        return {"force": 2}

    worker = wrench_nonblocking_diagnostic__NonBlockingWrenchWorker(query, target_hz=200, stale_threshold_ms=100)
    worker.start()
    deadline = wrench_nonblocking_diagnostic__time.monotonic() + 1.0
    saw_error = False
    saw_recovery = False
    while wrench_nonblocking_diagnostic__time.monotonic() < deadline:
        snapshot = worker.cache.snapshot()
        saw_error |= snapshot.last_error_code == 263
        saw_recovery |= snapshot.valid and snapshot.sequence_id >= 2
        if saw_error and saw_recovery:
            break
        wrench_nonblocking_diagnostic__time.sleep(0.001)
    worker.request_stop()
    assert worker.join(timeout=1.0)
    assert saw_error and saw_recovery
    final = worker.cache.snapshot()
    assert final.valid
    assert final.last_error_code == 263


def test_wrench_nonblocking_diagnostic__worker_reports_scheduled_and_exact_call_timestamps() -> None:
    rows = []
    worker = wrench_nonblocking_diagnostic__NonBlockingWrenchWorker(
        lambda: {"force": 1},
        target_hz=200,
        stale_threshold_ms=100,
        on_result=rows.append,
    )
    worker.start()
    deadline = wrench_nonblocking_diagnostic__time.monotonic() + 1.0
    while len(rows) < 2 and wrench_nonblocking_diagnostic__time.monotonic() < deadline:
        wrench_nonblocking_diagnostic__time.sleep(0.001)
    worker.request_stop()
    assert worker.join(timeout=1.0)
    assert len(rows) >= 2
    first = rows[0]
    assert first["scheduled_time_ns"] <= first["call_start_ns"] <= first["call_end_ns"]
    assert first["query_latency_ms"] == first["latency_ms"]


# ============================================================================
# Source: tests/test_wrench_process_isolation.py
# ============================================================================



import queue as wrench_process_isolation__queue
import time as wrench_process_isolation__time

import pytest as wrench_process_isolation__pytest

from scripts.validate_wrench_process_isolation import offline_case_passes as wrench_process_isolation__offline_case_passes, run_offline_case as wrench_process_isolation__run_offline_case
from scripts.wrench_process_isolation import (
    RtProcessSupervisor as wrench_process_isolation__RtProcessSupervisor,
    WrenchProcessSupervisor as wrench_process_isolation__WrenchProcessSupervisor,
    drain_latest as wrench_process_isolation__drain_latest,
    publish_latest as wrench_process_isolation__publish_latest,
)


def wrench_process_isolation___wait_for(supervisor: wrench_process_isolation__WrenchProcessSupervisor, predicate, timeout_s: float = 2.0):
    deadline = wrench_process_isolation__time.monotonic() + timeout_s
    latest = supervisor.poll()
    while wrench_process_isolation__time.monotonic() < deadline:
        latest = supervisor.poll()
        if predicate(latest):
            return latest
        wrench_process_isolation__time.sleep(0.005)
    return latest


def test_wrench_process_isolation__bounded_latest_snapshot_replaces_old_value_without_blocking() -> None:
    channel: wrench_process_isolation__queue.Queue = wrench_process_isolation__queue.Queue(maxsize=1)
    assert wrench_process_isolation__publish_latest(channel, {"sequence_id": 1})
    started = wrench_process_isolation__time.perf_counter()
    assert wrench_process_isolation__publish_latest(channel, {"sequence_id": 2})
    assert wrench_process_isolation__time.perf_counter() - started < 0.05
    assert wrench_process_isolation__drain_latest(channel) == {"sequence_id": 2}


@wrench_process_isolation__pytest.mark.parametrize(
    ("name", "behavior", "delay_s", "expect_code"),
    [
        ("normal", "normal", 0.001, None),
        ("slow_40ms", "normal", 0.040, None),
        ("error_263", "error263", 0.001, 263),
        ("worker_exception", "exception", 0.001, None),
    ],
)
def test_wrench_process_isolation__normal_slow_and_error_ipc(name, behavior, delay_s, expect_code) -> None:
    supervisor = wrench_process_isolation__WrenchProcessSupervisor(stale_age_ms=150, worker_hung_ms=500)
    supervisor.start(mode="offline", config={
        "behavior": behavior,
        "delay_s": delay_s,
        "target_hz": 20.0,
    })
    observation = wrench_process_isolation___wait_for(
        supervisor,
        # Heartbeat errors can arrive before the independent result queue.
        lambda item: item.wrench_sequence >= 1
        and (expect_code is None or item.last_error_code == expect_code)
        and (name != "worker_exception" or "synthetic worker exception" in str(item.last_error)),
    )
    cleanup = supervisor.stop_normally()
    supervisor.close()
    assert cleanup["worker_exited"]
    assert observation.wrench_sequence >= 1
    if expect_code is not None:
        assert observation.last_error_code == expect_code
    if name == "worker_exception":
        assert "synthetic worker exception" in str(observation.last_error)


def test_wrench_process_isolation__stale_and_heartbeat_timeout_are_independent() -> None:
    supervisor = wrench_process_isolation__WrenchProcessSupervisor(stale_age_ms=50, worker_hung_ms=200)
    supervisor.start(mode="offline", config={
        "behavior": "normal", "delay_s": 0.5, "target_hz": 20.0,
    })
    stale = wrench_process_isolation___wait_for(supervisor, lambda item: item.wrench_stale, timeout_s=0.15)
    hung = wrench_process_isolation___wait_for(supervisor, lambda item: item.worker_hung, timeout_s=0.5)
    cleanup = supervisor.terminate()
    supervisor.close()
    assert stale.wrench_stale
    assert hung.worker_hung
    assert cleanup["worker_terminated"]
    assert cleanup["graceful_disconnect_confirmed"] is False


def test_wrench_process_isolation__starting_worker_uses_separate_startup_timeout() -> None:
    supervisor = wrench_process_isolation__WrenchProcessSupervisor(
        stale_age_ms=50,
        worker_hung_ms=10,
        worker_startup_hung_ms=500,
    )
    supervisor.start(mode="offline", config={
        "behavior": "normal", "delay_s": 0.001, "target_hz": 20.0,
    })
    first = supervisor.poll()
    assert not first.worker_hung
    cleanup = supervisor.stop_normally()
    supervisor.close()
    assert cleanup["worker_exited"]


def test_wrench_process_isolation__permanent_hung_worker_is_terminatable_and_parent_survives() -> None:
    supervisor = wrench_process_isolation__WrenchProcessSupervisor(stale_age_ms=50, worker_hung_ms=100)
    supervisor.start(mode="offline", config={
        "behavior": "permanent", "permanent_sleep_s": 3600.0,
    })
    hung = wrench_process_isolation___wait_for(supervisor, lambda item: item.worker_hung, timeout_s=1.0)
    parent_counter = 0
    for _ in range(20):
        supervisor.poll()
        parent_counter += 1
    cleanup = supervisor.terminate()
    death = supervisor.poll()
    supervisor.close()
    assert hung.worker_hung
    assert parent_counter == 20
    assert cleanup["worker_terminated"]
    assert not death.worker_alive


def test_wrench_process_isolation__worker_crash_becomes_visible_without_killing_parent() -> None:
    supervisor = wrench_process_isolation__WrenchProcessSupervisor(stale_age_ms=50, worker_hung_ms=500)
    supervisor.start(mode="offline", config={
        "behavior": "crash", "delay_s": 0.001, "crash_after": 1,
    })
    dead = wrench_process_isolation___wait_for(supervisor, lambda item: not item.worker_alive and item.worker_exitcode is not None)
    supervisor.join(0.1)
    supervisor.close()
    assert not dead.worker_alive
    assert dead.worker_exitcode == 17


def test_wrench_process_isolation__offline_case_acceptance_rejects_low_parent_rate() -> None:
    result = {
        "scenario": "normal",
        "main_loop_rate_hz": 5.0,
        "parent_work_counter": 1,
        "main_loop_tick_count": 1,
        "maximum_wrench_sequence_seen": 1,
        "cleanup": {"worker_exited": True},
    }
    assert not wrench_process_isolation__offline_case_passes(result)


def test_wrench_process_isolation__real_ten_second_offline_block_does_not_block_parent_loop() -> None:
    result = wrench_process_isolation__run_offline_case({
        "name": "long_block_10s",
        "config": {"behavior": "normal", "delay_s": 10.0},
        "duration_s": 10.4,
        "hung_ms": 250.0,
    })
    assert result["pass"]
    assert result["main_loop_tick_count"] >= 800
    assert result["worker_hung_seen"]
    assert result["wrench_stale_seen"]


def test_wrench_process_isolation__saturated_process_ipc_keeps_worker_and_parent_progressing() -> None:
    result = wrench_process_isolation__run_offline_case({
        "name": "ipc_saturation",
        "config": {"behavior": "normal", "delay_s": 0.0001, "target_hz": 2000.0},
        "duration_s": 0.8,
    })
    assert result["pass"]
    assert result["maximum_wrench_sequence_seen"] > result["main_loop_tick_count"]


def test_wrench_process_isolation__rt_process_publishes_plain_state_and_stops_cleanly() -> None:
    supervisor = wrench_process_isolation__RtProcessSupervisor(stale_age_ms=50, worker_hung_ms=200)
    supervisor.start({"offline": True, "publish_hz": 125.0})
    observation = wrench_process_isolation___wait_for(
        supervisor,
        lambda item: item.rt_valid and item.rt_sequence > 2,
    )
    cleanup = supervisor.stop_normally()
    final = supervisor.poll()
    supervisor.close()
    assert observation.rt_age_ms is not None and observation.rt_age_ms < 50
    assert observation.operation_state == "IDLE"
    assert cleanup["worker_exited"]
    assert cleanup["graceful_disconnect_confirmed"] is True
    assert not final.worker_alive


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
    acquisition=RealRobotAcquisition(FakeAdapter(),EpisodeLogger(tmp_path/'unused'),wrench_provider=p)
    acquisition._runtime_freshness_armed=True
    stale=FakeAdapter().read_state_frame()
    acquisition._latest_state=wrench_production_process__replace(stale,host_monotonic_time_s=wrench_production_process__time.perf_counter()-10)
    acquisition._latest_wrench=p.last_good
    h=acquisition.latest_health()
    assert not h.valid
    assert 'state_freshness_failure' in p.failure_latch


def test_wrench_production_process__stationary_preparation_no_live_entry(tmp_path):
    from scripts.prepare_wrench_stationary_validation import prepare
    manifest=prepare(tmp_path/'prepared',50)
    assert manifest['duration_s']==900 and manifest['intended_wrench_hz']==50
    assert not manifest['live_runner_ready'] and not manifest['robot_connected']
    with wrench_production_process__pytest.raises(ValueError): prepare(tmp_path/'invalid',float('nan'))


def test_wrench_production_process__existing_control_rejects_historical_good_unhealthy_stream():
    import pandas as pd
    from control.execution_preflight import evaluate_execution_preflight
    from control.robot_trajectory_executor import RokaeMotionExecutor
    _safety = robot_trajectory_executor___safety
    _health = robot_trajectory_executor___health
    from safety.experiment_safety import ExperimentSafetyConfig
    h = wrench_production_process__replace(_health(), valid=False, invalid_reason='sdk_error',
                wrench_valid=True, wrench_stream_healthy=False, wrench_failure_latch=('sdk_error',))
    # Deliberately unapproved fake request; evaluate the real health gate without
    # loading or mutating an official trajectory release or calling any SDK.
    audit = wrench_production_process__SimpleNamespace(trajectory_id='fake',sample_count=0,**{k:True for k in (
        'trajectory_valid','first_target_equals_anchor','final_target_equals_anchor',
        'first_relative_displacement_zero','final_relative_displacement_zero',
        'approved_rom_valid','theta_shank_definition_valid','all_samples_finite','time_strictly_increasing')})
    result = evaluate_execution_preflight(mode='offline',enable_motion=False,operator_confirmation='',
        requested_anchor_id='fake',frame=wrench_production_process__SimpleNamespace(reviewed=False),
        anchor=wrench_production_process__SimpleNamespace(reviewed=False,anchor_id='fake',trajectory_id='fake',tcp_pose_base=(0.,)*6),
        safety=ExperimentSafetyConfig(),trajectory=pd.DataFrame(),audit=audit,acquisition_health=h,
        logger=wrench_production_process__SimpleNamespace(ready=True,healthy=True),robot_adapter=wrench_production_process__SimpleNamespace(is_connected=False))
    assert not result.allowed
    assert 'acquisition_streams_unhealthy' in result.reasons
    with wrench_production_process__pytest.raises(PermissionError): result.require_allowed()
    executor = object.__new__(RokaeMotionExecutor)
    executor.logger = wrench_production_process__SimpleNamespace(healthy_signal=True)
    executor.acquisition = wrench_production_process__SimpleNamespace(latest_health=lambda:h)
    executor.safety = _safety()
    assert 'sdk_error' in executor._runtime_reasons()
    # No motion object exists in this test; no execute() or motion API called.


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


# ============================================================================
# Source: tests/test_wrench_stationary_runner.py
# ============================================================================

"""Offline integration only. Real spawned production provider; no live SDK."""
import json as wrench_stationary_runner__json
import os as wrench_stationary_runner__os
from pathlib import Path as wrench_stationary_runner__Path
import sys as wrench_stationary_runner__sys
import time as wrench_stationary_runner__time

import pytest as wrench_stationary_runner__pytest

from scripts.prepare_wrench_stationary_validation import prepare as wrench_stationary_runner__prepare
from scripts import run_wrench_stationary_validation as wrench_stationary_runner__runner


@wrench_stationary_runner__pytest.fixture
def wrench_stationary_runner__request_record(tmp_path):
    return wrench_stationary_runner__prepare(tmp_path / "request", 50.)


@wrench_stationary_runner__pytest.fixture
def wrench_stationary_runner__outputs(tmp_path, request):
    root = wrench_stationary_runner__Path(wrench_stationary_runner__os.environ.get("STATIONARY_TEST_OUTPUT", tmp_path)) / request.node.name
    root.mkdir(parents=True, exist_ok=False)
    return root


@wrench_stationary_runner__pytest.mark.parametrize("case", ["A", "B"])
def test_wrench_stationary_runner__normal_case(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs, case, monkeypatch):
    # A hardware constructor or any native wrench call in parent is a test failure.
    from hardware.rokae_adapter import RokaeRobotAdapter
    def forbidden(*args, **kwargs): raise AssertionError("hardware forbidden")
    monkeypatch.setattr(RokaeRobotAdapter, "__init__", forbidden)
    path, summary = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, case, wrench_stationary_runner__outputs, offline_seconds=1.2)
    assert summary["offline_dry_run_pass"], summary["failures"]
    assert summary["requested_duration_s"] == 900
    assert not summary["completed_900s"]
    assert summary["intended_wrench_rate_hz"] == 50
    assert not summary["robot_connected"]
    assert summary["rt_sample_count"] > 0
    assert summary["timing_acceptance"] == "UNDEFINED"
    for name in ("protocol_snapshot.json", "rt_state.csv", "wrench_query_events.jsonl", "wrench_samples.csv",
                 "health_timeline.jsonl", "process_lifecycle.jsonl", "cleanup_outcome.json", "end_state.json", "case_summary.json"):
        assert (path / name).exists()
    if case == "A":
        assert summary["wrench_status"] == "N/A_INTENTIONALLY_DISABLED"
        assert all(not r["valid"] for r in wrench_stationary_runner__runner.rows(path / "health_timeline.jsonl"))
    else:
        assert summary["wrench_successful_queries"] > 0
        assert summary["cleanup"]["child"]["GRACEFUL_SDK_DISCONNECT"] is None
        assert summary["cleanup"]["child"]["process_exited"]


@wrench_stationary_runner__pytest.mark.parametrize("behavior,expected", [("error263", "263"), ("never", None)])
def test_wrench_stationary_runner__faults_preserved(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs, behavior, expected):
    path, summary = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "B", wrench_stationary_runner__outputs, offline_seconds=3., behavior=behavior)
    assert summary["failures"]
    assert not summary["completed_900s"]
    assert summary["requested_duration_s"] == 900
    assert summary["actual_duration_s"] < 3
    if expected:
        assert summary["sdk_error_histogram"] == {"263": 1}
        assert summary["wrench_started_queries"] == 2
    else:
        assert summary["pending_queries"]
        assert summary["wrench_completed_queries"] == 0
        assert summary["cleanup"]["child"]["FORCED_WORKER_TERMINATION"]
        assert summary["cleanup"]["child"]["GRACEFUL_SDK_DISCONNECT"] is False


def test_wrench_stationary_runner__missing_final_state(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs):
    path, summary = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "A", wrench_stationary_runner__outputs, offline_seconds=.1, final_state=False)
    end = wrench_stationary_runner__json.loads((path / "end_state.json").read_text())
    assert end["final_operation_state"] is None and end["reason"]
    assert "final_operation_state_missing" in summary["failures"]


def test_wrench_stationary_runner__state_failure(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs):
    _, summary = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "A", wrench_stationary_runner__outputs, offline_seconds=1., state_failure=True)
    assert "state_unhealthy" in summary["failures"]


@wrench_stationary_runner__pytest.mark.parametrize("authorization", [None, {}, {"reviewed": True}])
def test_wrench_stationary_runner__live_missing_auth_before_sdk(wrench_stationary_runner__request_record, tmp_path, authorization, monkeypatch):
    from hardware.rokae_adapter import RokaeRobotAdapter
    calls = []
    monkeypatch.setattr(RokaeRobotAdapter, "__init__", lambda *a, **k: calls.append("SDK"))
    with wrench_stationary_runner__pytest.raises(PermissionError):
        wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "B", tmp_path / "unused", mode="LIVE_STATIONARY", authorization=authorization)
    assert not calls and not (tmp_path / "unused").exists()
    assert "xCoreSDK_python" not in wrench_stationary_runner__sys.modules


@wrench_stationary_runner__pytest.mark.parametrize("key,value", [("schema_version", 2), ("duration_s", 1), ("intended_wrench_hz", 0), ("retry", True)])
def test_wrench_stationary_runner__request_rejection(wrench_stationary_runner__request_record, tmp_path, key, value):
    wrench_stationary_runner__request_record[key] = value
    with wrench_stationary_runner__pytest.raises(ValueError): wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "A", tmp_path)


def test_wrench_stationary_runner__comparison_incomplete(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs):
    _, a = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "A", wrench_stationary_runner__outputs, offline_seconds=.1)
    _, b = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "B", wrench_stationary_runner__outputs, offline_seconds=2., behavior="error263")
    result = wrench_stationary_runner__runner.compare(a, b)
    assert result["status"] == "INCOMPLETE"
    assert result["causality"] == "NOT_ESTABLISHED"
    wrench_stationary_runner__runner.write_json(wrench_stationary_runner__outputs / "ab_comparison.json", result)


def test_wrench_stationary_runner__runner_has_no_control_dependency():
    import ast
    tree = ast.parse(wrench_stationary_runner__Path(wrench_stationary_runner__runner.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("control", "safety"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"execute", "servo", "setPowerState", "setOperateMode", "setMotionControlMode", "move", "movej", "movel"}


def wrench_stationary_runner__complete_authorization(wrench_stationary_runner__request_record):
    # Deliberately fictional OFFLINE artifact; this helper NEVER invokes run_case live.
    wrench_stationary_runner__request_record.update(budgets=dict(startup_s=5., query_s=.5, progress_s=2., sample_age_s=2.,
        ack_s=2., log_s=2., shutdown_s=.5, terminate_s=.5, state_age_s=2., skew_s=2.,
        provenance="OFFLINE_AUTHORIZATION_VALIDATOR_FIXTURE"),
        lifecycle_budgets=dict.fromkeys(("connect_s", "stop_s", "end_state_s", "disconnect_s", "logger_s"), 1.),
        connection=dict(robot_ip="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE"),
        expected_identity=dict(robot_model="OFFLINE", robot_serial="OFFLINE", controller_version="OFFLINE", sdk_version="OFFLINE"),
        allowed_power_states=["off"], criteria=dict(max_supervisor_interval_s=.5))
    a = dict(schema_version=1, scope="LIVE_STATIONARY_NO_MOTION", reviewed=True,
        request_sha256=wrench_stationary_runner__runner.digest(wrench_stationary_runner__request_record), case="A", source_hashes=wrench_stationary_runner__runner.source_hashes(),
        previous_session_recovery_confirmed=True, operator_present=True, stop_procedure_reviewed=True,
        parent_native_lifecycle_reviewed=True, end_state_path="existing_session_operationState",
        expires_at="2099-01-01T00:00:00Z")
    for key in ("authorization_id", "reviewer", "operator", "stop_procedure", "recovery_record", "setup_record", "vendor_session_record"):
        a[key] = "OFFLINE_FIXTURE"
    return a


@wrench_stationary_runner__pytest.mark.parametrize("tamper", ["request", "source", "expiry", "ack", "budget", "case"])
def test_wrench_stationary_runner__authorization_binding(wrench_stationary_runner__request_record, tamper):
    a = wrench_stationary_runner__complete_authorization(wrench_stationary_runner__request_record)
    ack = "OFFLINE_FIXTURE:OFFLINE_FIXTURE"
    wrench_stationary_runner__runner.authorize(wrench_stationary_runner__request_record, "A", a, ack)  # No adapter construction.
    if tamper == "request": wrench_stationary_runner__request_record["intended_wrench_hz"] = 20.
    elif tamper == "source": a["source_hashes"] = {}
    elif tamper == "expiry": a["expires_at"] = "2000-01-01T00:00:00Z"
    elif tamper == "ack": ack = "yes"
    elif tamper == "budget":
        wrench_stationary_runner__request_record["budgets"] = None
        a["request_sha256"] = wrench_stationary_runner__runner.digest(wrench_stationary_runner__request_record)
    else: a["case"] = "B"
    with wrench_stationary_runner__pytest.raises(PermissionError): wrench_stationary_runner__runner.authorize(wrench_stationary_runner__request_record, "A", a, ack)


def test_wrench_stationary_runner__b_requires_reviewed_a(wrench_stationary_runner__request_record):
    a = wrench_stationary_runner__complete_authorization(wrench_stationary_runner__request_record)
    a["case"] = "B"
    with wrench_stationary_runner__pytest.raises(PermissionError, match="requires reviewed successful A"):
        wrench_stationary_runner__runner.authorize(wrench_stationary_runner__request_record, "B", a, "OFFLINE_FIXTURE:OFFLINE_FIXTURE")


def test_wrench_stationary_runner__failed_stop_does_not_retry_disconnect(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs, monkeypatch):
    calls = []
    def stop(self): raise RuntimeError("injected_stop_failure")
    def disconnect(self): calls.append("disconnect")
    monkeypatch.setattr(wrench_stationary_runner__runner.OfflineStateAdapter, "stop_state_stream", stop)
    monkeypatch.setattr(wrench_stationary_runner__runner.OfflineStateAdapter, "disconnect", disconnect)
    _, summary = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "A", wrench_stationary_runner__outputs, offline_seconds=.1)
    assert not calls
    assert not summary["cleanup"]["parent"]["normal_cleanup"]
    assert summary["final_robot_operation_state"] is None
    assert summary["failures"]


def test_wrench_stationary_runner__telemetry_failure_aborts(wrench_stationary_runner__request_record, wrench_stationary_runner__outputs, monkeypatch):
    monkeypatch.setattr(wrench_stationary_runner__runner.Evidence, "check", lambda self: "injected_telemetry_failure")
    _, summary = wrench_stationary_runner__runner.run_case(wrench_stationary_runner__request_record, "A", wrench_stationary_runner__outputs, offline_seconds=1.)
    assert summary["failures"] and not summary["completed_900s"]


def test_wrench_stationary_runner__incomplete_jsonl_retained(tmp_path):
    path = tmp_path / "torn.jsonl"
    path.write_text('{"type":"good"}\n{"type":', encoding="utf-8")
    failures = []
    assert wrench_stationary_runner__runner.rows(path, failures) == [{"type": "good"}]
    assert failures and "telemetry_corrupt" in failures[0]
    assert path.read_text().endswith('{"type":')


def test_wrench_stationary_runner__bounded_call_does_not_claim_cancel():
    value, unresolved = wrench_stationary_runner__runner.bounded_call(lambda: wrench_stationary_runner__time.sleep(.05), .001)
    assert unresolved and "UNRESOLVED_PARENT_CALL" in value["error"]
    wrench_stationary_runner__time.sleep(.06)


# ============================================================================
# Source: experiment_algorithm/test_migration.py
# ============================================================================

"""Offline migration checks; no robot interface or motion is used."""
from config.bundle import load_section as algorithm_migration__load_section
from dataclasses import asdict as algorithm_migration__asdict
import importlib as algorithm_migration__importlib
import json as algorithm_migration__json
from pathlib import Path as algorithm_migration__Path
import subprocess as algorithm_migration__subprocess
import sys as algorithm_migration__sys
from types import SimpleNamespace as algorithm_migration__SimpleNamespace

import numpy as algorithm_migration__np
import pandas as algorithm_migration__pd
import pytest as algorithm_migration__pytest

from scripts import train_robot_leg_round as algorithm_migration__worker
from experiment_algorithm import policy as algorithm_migration__policy, surrogate as algorithm_migration__surrogate

algorithm_migration__ROOT = algorithm_migration__Path(__file__).resolve().parents[1]


def algorithm_migration__request_template():
    configured = algorithm_migration__load_section("robot_leg_round", resolved=True)
    configured["algorithm_root"] = "experiment_algorithm"
    # Freeze actual checkout bytes for this synthetic request, never production configuration.
    configured["algorithm_sha256"] = {
        name: algorithm_migration__worker.sha256_file(algorithm_migration__ROOT / "experiment_algorithm" / name)
        for name in configured["algorithm_sha256"]}
    configured["algorithm_dependencies"] = {
        algorithm_migration__Path(name).relative_to(algorithm_migration__ROOT).as_posix():
            algorithm_migration__worker.sha256_file(name)
        for name in configured["algorithm_dependencies"]}
    for candidate in configured["candidates"]:
        candidate["path"] = algorithm_migration__Path(candidate["path"]).relative_to(algorithm_migration__ROOT).as_posix()
    reference = configured["candidates"][0]
    return dict(configured, setup_sha256="0" * 64, reference_sha256=reference["sha256"], history=[])


def test_algorithm_migration__config_import_is_namespaced():
    robot_config = algorithm_migration__importlib.import_module("config")
    algorithm_config = algorithm_migration__importlib.import_module("experiment_algorithm.config")
    assert algorithm_migration__Path(robot_config.__file__).parent == algorithm_migration__ROOT / "config"
    assert algorithm_config.AlgorithmConfig is algorithm_migration__policy.AlgorithmConfig
    assert not any(name in algorithm_migration__sys.modules for name in ("surrogate", "policy", "coupled_model"))
    assert "lower_limb_sim.e3_low_budget.run" not in algorithm_migration__sys.modules
    assert "external_simulation.myoleg_reference_trajectory_replay_v1.build_and_replay" not in algorithm_migration__sys.modules


def test_algorithm_migration__validate_only_with_repo_paths_from_other_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = algorithm_migration__worker.train(algorithm_migration__request_template(), validate_only=True)
    assert result["status"] == "ALGORITHM_INITIALIZED"
    assert result["source_hashes_checked"] is True
    assert result["public_prior"]["includes_foot_and_auxiliary_inertia"] is True


@algorithm_migration__pytest.mark.parametrize("field", ["algorithm_sha256", "algorithm_dependencies"])
def test_algorithm_migration__changed_hash_is_rejected_before_model(monkeypatch, field):
    request = algorithm_migration__request_template()
    name = next(iter(request[field]))
    request[field][name] = "0" * 64
    def forbidden(*args, **kwargs):
        algorithm_migration__pytest.fail("model must not be imported after failed integrity check")
    monkeypatch.setattr(algorithm_migration__worker.importlib, "import_module", forbidden)
    with algorithm_migration__pytest.raises(ValueError, match="changed"):
        algorithm_migration__worker.train(request, validate_only=True)


def test_algorithm_migration__unhashed_package_helper_is_rejected():
    request = algorithm_migration__request_template()
    helper = next(name for name in request["algorithm_dependencies"]
                  if algorithm_migration__Path(name).as_posix().endswith("experiment_algorithm/coordinates.py"))
    request["algorithm_dependencies"].pop(helper)
    with algorithm_migration__pytest.raises(ValueError, match="package dependency hashes"):
        algorithm_migration__worker.train(request, validate_only=True)


def test_algorithm_migration__other_algorithm_root_is_rejected(tmp_path):
    request = algorithm_migration__request_template()
    request["algorithm_root"] = str(tmp_path)
    with algorithm_migration__pytest.raises(ValueError, match="algorithm_root"):
        algorithm_migration__worker.train(request, validate_only=True)


def test_algorithm_migration__candidate_hash_is_checked():
    request = algorithm_migration__request_template()
    request["candidates"][0]["sha256"] = "0" * 64
    with algorithm_migration__pytest.raises(ValueError, match="candidate trajectory hash"):
        algorithm_migration__worker.train(request, validate_only=True)


def test_algorithm_migration__worker_cli_validate_only(tmp_path):
    request = tmp_path / "request.json"
    output = tmp_path / "validation.json"
    request.write_text(algorithm_migration__json.dumps(algorithm_migration__request_template()), encoding="utf-8")
    completed = algorithm_migration__subprocess.run([algorithm_migration__sys.executable, "-B", "-m", "scripts.train_robot_leg_round",
        str(request), "--output", str(output), "--validate-only"], cwd=algorithm_migration__ROOT,
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert algorithm_migration__json.loads(output.read_text(encoding="utf-8"))["status"] == "ALGORITHM_INITIALIZED"


def test_algorithm_migration__training_with_synthetic_direct_torque_and_measurement_hash(tmp_path):
    request = algorithm_migration__request_template()
    reference = request["candidates"][0]
    frame = algorithm_migration__pd.read_csv(algorithm_migration__ROOT / reference["path"])
    from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
    from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
    template = baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS["baseline"])
    matrix, offset = algorithm_migration__surrogate.mechanical.design_matrix(frame, template)
    assert algorithm_migration__np.array_equal(offset, algorithm_migration__np.zeros(len(offset)))
    theta = algorithm_migration__np.array([1.05, 11., 9., 1.2, .9, 1.5])
    torque = (matrix @ theta).reshape(2, -1).T
    frame[["tau_hip_nm", "tau_knee_nm"]] = torque
    measured = tmp_path / "synthetic.csv"
    frame.to_csv(measured, index=False)
    request["history"] = [dict(episode_id="SYNTHETIC_ONLY", candidate_id=reference["candidate_id"],
        measured_path=str(measured), measured_sha256=algorithm_migration__worker.sha256_file(measured),
        endpoints=dict(E3=1., E2=1., peak_ratio=1.))]
    result = algorithm_migration__worker.train(request)
    assert result["status"] == "MODEL_UPDATED"
    assert result["diagnostics"]["train_rmse"] < 1e-5
    assert len(result["theta"]) == 6
    assert result["action"]["stop_reason"] == "DOMAIN_EXHAUSTED"
    assert algorithm_migration__np.isfinite(result["predictions"]["mean"]).all()
    request["history"][0]["measured_sha256"] = "0" * 64
    with algorithm_migration__pytest.raises(ValueError, match="measured training frame changed"):
        algorithm_migration__worker.train(request)


def test_algorithm_migration__greedy_selects_mean_and_retains_feasibility_gate():
    points = [algorithm_migration__SimpleNamespace(candidate_id=name, beta_flex=0., beta_extend=0.) for name in ("ref", "best", "unsafe")]
    class Domain(list):
        reference = points[0]
    context = algorithm_migration__SimpleNamespace(domain=Domain(points), common_candidate_id="ref")
    diagnostics = dict(rank=6, condition=1., normalized_train_rmse=0., scaled_parameter_change=0.,
                       prequential_count=1, latest_prequential=dict(normalized_torque_rmse=0., E3_abs_error=0.))
    model = algorithm_migration__SimpleNamespace(diagnostics=diagnostics)
    model.fit = lambda measurements: model
    model.predict_all = lambda: dict(mean=algorithm_migration__np.array([[1., 1., 1.], [.8, 1., 1.], [.1, 2., 2.]]),
                                    std=algorithm_migration__np.full((3, 3), .001))
    observations = [algorithm_migration__SimpleNamespace(observation=algorithm_migration__SimpleNamespace(valid=True, candidate_id="ref",
        endpoint_value=1., endpoint_uncertainty=None, trial_index=1), E2=1., peak_ratio=1., feasible=True)]
    config = algorithm_migration__policy.AlgorithmConfig(min_model_trials=1)
    action = algorithm_migration__policy.Policy(context, method="GREEDY", surrogate=model, config=config).next_action(observations)
    assert action.candidate_id == "best", algorithm_migration__asdict(action)


# ============================================================================
# Evidence archive and unified configuration integrity
# ============================================================================

import hashlib as integrity__hashlib
import json as integrity__json
from types import SimpleNamespace as integrity__SimpleNamespace
import zipfile as integrity__zipfile

import pytest as integrity__pytest

from config import bundle as bundle_integrity__module
from utils import evidence as evidence_integrity__module


def evidence_integrity__write_archive(sandbox, *, name=None, blob=None, size=None):
    name = sandbox.name if name is None else name
    index = dict(
        entries={name: dict(sha256=integrity__hashlib.sha256(sandbox.payload).hexdigest(),
                           size=len(sandbox.payload) if size is None else size)},
        runtime=[name], source_commit="SYNTHETIC_ONLY", limitations=[],
    )
    digest = index["entries"][name]["sha256"]
    with integrity__zipfile.ZipFile(sandbox.archive, "w") as archive:
        archive.writestr("index.json", integrity__json.dumps(index))
        archive.writestr("blobs/" + digest, sandbox.payload if blob is None else blob)


@integrity__pytest.fixture
def evidence_integrity__sandbox(tmp_path, monkeypatch):
    sandbox = integrity__SimpleNamespace(
        archive=tmp_path / "evidence.zip", cache=tmp_path / "cache",
        name="repository/reference_release/frozen.csv", payload=b"time_s,q_rad\n0,0\n",
    )
    evidence_integrity__write_archive(sandbox)
    monkeypatch.setattr(evidence_integrity__module, "ARCHIVE", sandbox.archive)
    monkeypatch.setattr(evidence_integrity__module, "CACHE", sandbox.cache)
    monkeypatch.setattr(evidence_integrity__module, "_READY", None)
    return sandbox


def test_evidence_integrity__verified_archive_materializes_exact_runtime_bytes(evidence_integrity__sandbox):
    sandbox = evidence_integrity__sandbox
    original = sandbox.archive.read_bytes()
    assert evidence_integrity__module.read_bytes(sandbox.name) == sandbox.payload
    root = evidence_integrity__module.runtime_root()
    assert root == sandbox.cache / "repository"
    path = evidence_integrity__module.artifact_path("reference_release/frozen.csv")
    assert path == root / "reference_release/frozen.csv"
    assert path.read_bytes() == sandbox.payload
    result = evidence_integrity__module.verify()
    assert result["originals"] == result["unique_blobs"] == 1
    assert result["source_commit"] == "SYNTHETIC_ONLY"
    assert sandbox.archive.read_bytes() == original


@integrity__pytest.mark.parametrize("entrypoint", ["read_bytes", "verify", "runtime_root"])
@integrity__pytest.mark.parametrize("corruption", ["hash", "size"])
def test_evidence_integrity__tampered_zip_is_rejected(evidence_integrity__sandbox, entrypoint, corruption):
    sandbox = evidence_integrity__sandbox
    if corruption == "hash":
        evidence_integrity__write_archive(sandbox, blob=b"X" * len(sandbox.payload))
    else:
        evidence_integrity__write_archive(sandbox, size=len(sandbox.payload) + 1)
    with integrity__pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        operation = getattr(evidence_integrity__module, entrypoint)
        operation(sandbox.name) if entrypoint == "read_bytes" else operation()
    assert not (sandbox.cache / sandbox.name).exists()


@integrity__pytest.mark.parametrize("entrypoint", ["read_bytes", "verify", "runtime_root"])
def test_evidence_integrity__invalid_zip_never_becomes_runtime_evidence(evidence_integrity__sandbox, entrypoint):
    sandbox = evidence_integrity__sandbox
    sandbox.archive.write_bytes(b"SYNTHETIC_INVALID_ZIP")
    with integrity__pytest.raises(integrity__zipfile.BadZipFile):
        operation = getattr(evidence_integrity__module, entrypoint)
        operation(sandbox.name) if entrypoint == "read_bytes" else operation()
    assert not sandbox.cache.exists()


@integrity__pytest.mark.parametrize("entrypoint", ["read_bytes", "artifact_path"])
@integrity__pytest.mark.parametrize("name", [
    "", "../escape.csv", "/absolute.csv", "repository/../escape.csv",
    "C:/escape.csv", "C:escape.csv", "repository\\escape.csv",
])
def test_evidence_integrity__unsafe_paths_are_rejected(evidence_integrity__sandbox, entrypoint, name):
    with integrity__pytest.raises(ValueError, match="invalid evidence path"):
        getattr(evidence_integrity__module, entrypoint)(name)
    assert not (evidence_integrity__sandbox.archive.parent / "escape.csv").exists()


@integrity__pytest.mark.parametrize("name", [
    "repository/../../escape.csv", "raw/recording.csv", "/absolute.csv", "C:/escape.csv",
])
def test_evidence_integrity__runtime_index_cannot_escape_repository(evidence_integrity__sandbox, name):
    sandbox = evidence_integrity__sandbox
    evidence_integrity__write_archive(sandbox, name=name)
    with integrity__pytest.raises(ValueError, match="invalid evidence path|originate in the repository"):
        evidence_integrity__module.runtime_root()
    assert not sandbox.cache.exists()
    assert not (sandbox.archive.parent / "escape.csv").exists()


@integrity__pytest.fixture
def bundle_integrity__sandbox(tmp_path, monkeypatch):
    module = bundle_integrity__module
    root = tmp_path / "checkout"
    source = root / "config/experiment.json"
    source.parent.mkdir(parents=True)
    original = module.BUNDLE_PATH.read_bytes()
    source.write_bytes(original)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "BUNDLE_PATH", source)
    monkeypatch.setattr(module, "CACHE_ROOT", tmp_path / "config-cache")
    monkeypatch.setattr(evidence_integrity__module, "artifact_path", lambda relative: root / "evidence" / relative)
    return integrity__SimpleNamespace(root=root, source=source, original=original)


@integrity__pytest.mark.parametrize("name", [
    "robot_leg_round", "lab_next_visit", "experiment_safety",
    "static_validation_logging", "formal_experiment_manifest",
])
def test_bundle_integrity__all_loading_routes_use_one_authority(bundle_integrity__sandbox, name):
    sandbox = bundle_integrity__sandbox
    module = bundle_integrity__module
    raw = integrity__json.loads(sandbox.original)["sections"][name]
    assert module.load_section(name) == raw
    path = module.section_path(name)
    rendered = integrity__json.loads(path.read_text(encoding="utf-8"))
    assert module.load_section(name, resolved=True) == rendered
    assert module.resolve_config_path(sandbox.root / "config" / (name + ".json")) == path
    assert module.resolve_config_path(sandbox.source, section=name) == path
    assert module.resolve_config_path(path) == path
    assert sandbox.source.read_bytes() == sandbox.original


def test_bundle_integrity__loaded_sections_do_not_mutate_authority(bundle_integrity__sandbox):
    module = bundle_integrity__module
    expected = integrity__json.loads(bundle_integrity__sandbox.original)["sections"]["robot_leg_round"]
    loaded = module.load_section("robot_leg_round")
    loaded["site_evidence"]["controller_safety"]["synthetic_marker"] = True
    loaded["candidates"][0]["path"] = "SYNTHETIC_CHANGED"
    assert module.load_section("robot_leg_round") == expected
    assert bundle_integrity__sandbox.source.read_bytes() == bundle_integrity__sandbox.original


def test_bundle_integrity__bundle_requires_section_and_run_files_pass_through(bundle_integrity__sandbox):
    module = bundle_integrity__module
    with integrity__pytest.raises(ValueError, match="explicit configuration section"):
        module.resolve_config_path(bundle_integrity__sandbox.source)
    run_file = bundle_integrity__sandbox.root / "reviewed_run.json"
    run_file.write_text('{"reviewed": false}', encoding="utf-8")
    assert module.resolve_config_path(run_file) == run_file
    assert integrity__json.loads(run_file.read_text(encoding="utf-8")) == {"reviewed": False}


def test_bundle_integrity__stale_cache_cannot_select_previous_configuration(bundle_integrity__sandbox):
    sandbox = bundle_integrity__sandbox
    module = bundle_integrity__module
    previous = module.section_path("robot_leg_round")
    previous_bytes = previous.read_bytes()
    data = integrity__json.loads(sandbox.original)
    data["sections"]["robot_leg_round"]["note"] += " SYNTHETIC_AUTHORITY_CHANGE"
    sandbox.source.write_text(integrity__json.dumps(data), encoding="utf-8")
    current = module.section_path("robot_leg_round")
    assert current != previous
    with integrity__pytest.raises(PermissionError, match="stale or unknown materialized configuration"):
        module.resolve_config_path(previous)
    assert previous.read_bytes() == previous_bytes
    assert module.resolve_config_path(current) == current
    assert integrity__json.loads(current.read_text(encoding="utf-8"))["note"].endswith("SYNTHETIC_AUTHORITY_CHANGE")


@integrity__pytest.mark.parametrize("name", ["robot_leg_round", "experiment_safety"])
def test_bundle_integrity__tampered_materialized_view_is_rejected(bundle_integrity__sandbox, name):
    path = bundle_integrity__module.section_path(name)
    path.write_bytes(path.read_bytes() + b"\n")
    with integrity__pytest.raises(PermissionError, match="materialized configuration changed"):
        bundle_integrity__module.resolve_config_path(path)


@integrity__pytest.mark.parametrize("relative", ["unknown.json", "nested/robot_leg_round.json"])
def test_bundle_integrity__unknown_cache_views_are_rejected(bundle_integrity__sandbox, relative):
    current = bundle_integrity__module.section_path("robot_leg_round")
    unknown = current.parent / relative
    unknown.parent.mkdir(parents=True, exist_ok=True)
    unknown.write_bytes(current.read_bytes())
    with integrity__pytest.raises(PermissionError, match="stale or unknown materialized configuration"):
        bundle_integrity__module.resolve_config_path(unknown)


# ============================================================================
# Completed-session archive lifecycle; all paths and ZIPs are temporary
# ============================================================================


def session_archive__snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@integrity__pytest.fixture
def session_archive__sandbox(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    archive = root / "data/evidence.zip"
    archive.parent.mkdir(parents=True)
    sandbox = integrity__SimpleNamespace(
        root=root, archive=archive, cache=root / "cache",
        name="repository/reference_release/frozen.csv", payload=b"time_s,q_rad\n0,0\n",
        session=root / "results/lab/run-001",
    )
    evidence_integrity__write_archive(sandbox)
    monkeypatch.setattr(evidence_integrity__module, "ROOT", root)
    monkeypatch.setattr(evidence_integrity__module, "ARCHIVE", archive)
    monkeypatch.setattr(evidence_integrity__module, "CACHE", sandbox.cache)
    monkeypatch.setattr(evidence_integrity__module, "_READY", None)
    for name, value in {
        "round_status.json": b'{"status":"ROUND_COMPLETE"}\n',
        "leg/leg.csv": sandbox.payload,
        "episode/robot_state.csv": sandbox.payload,
        "release_ledger.json": b'{"claimed":true}\n',
        "measured_metrics.json": b'{"E3":1.25}\n',
    }.items():
        path = sandbox.session / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return sandbox


@integrity__pytest.mark.parametrize("status", [
    "ROUND_COMPLETE", "CAPTURE_ONLY_COMPLETE", "EMPTY_LOAD_COMMISSIONING_COMPLETE",
])
def test_session_archive__completed_round_is_retained_and_repeated_without_duplicate_blobs(session_archive__sandbox, status):
    sandbox = session_archive__sandbox
    module = evidence_integrity__module
    (sandbox.session / "round_status.json").write_text(integrity__json.dumps({"status": status}), encoding="utf-8")
    logs = session_archive__snapshot(sandbox.session)
    first = module.archive_session(sandbox.session)
    assert first == dict(session="sessions/lab/run-001", files=5, new_blobs=3, original_logs_retained=True)
    with integrity__zipfile.ZipFile(sandbox.archive) as archive:
        original_index = integrity__json.loads(archive.read("index.json"))
        original_names = archive.namelist()
        assert len(original_names) == len(set(original_names))
        assert len([name for name in original_names if name.startswith("blobs/")]) == 4
        assert set(original_index["entries"]) == {sandbox.name, *(first["session"] + "/" + name for name in logs)}
        for name, expected in logs.items():
            item = original_index["entries"][first["session"] + "/" + name]
            assert item["source"] == first["session"] + "/" + name
            assert archive.read("blobs/" + item["sha256"]) == expected
    summary_path = sandbox.root / "data/experiment_summary.csv"
    original_summary = summary_path.read_bytes()
    assert original_summary
    second = module.archive_session(sandbox.session)
    assert second == dict(first, new_blobs=0)
    with integrity__zipfile.ZipFile(sandbox.archive) as archive:
        assert integrity__json.loads(archive.read("index.json")) == original_index
        assert sorted(archive.namelist()) == sorted(original_names)
    assert module.read_bytes(sandbox.name) == sandbox.payload
    assert module.verify()["unique_blobs"] == 4
    assert summary_path.read_bytes() == original_summary
    assert session_archive__snapshot(sandbox.session) == logs
    assert not sandbox.archive.with_suffix(".lock").exists()


@integrity__pytest.mark.parametrize("status", ["ROUND_RUNNING", "ROUND_FAILED", None])
def test_session_archive__incomplete_round_is_rejected_without_changing_evidence(session_archive__sandbox, status):
    sandbox = session_archive__sandbox
    (sandbox.session / "round_status.json").write_text(integrity__json.dumps({"status": status}), encoding="utf-8")
    original = sandbox.archive.read_bytes()
    logs = session_archive__snapshot(sandbox.session)
    with integrity__pytest.raises(ValueError, match="only closed, completed rounds"):
        evidence_integrity__module.archive_session(sandbox.session)
    assert sandbox.archive.read_bytes() == original
    assert session_archive__snapshot(sandbox.session) == logs
    assert not (sandbox.root / "data/experiment_summary.csv").exists()
    assert not sandbox.archive.with_suffix(".lock").exists()


@integrity__pytest.mark.parametrize("relative", ["leg/leg.csv", "round_status.json"])
def test_session_archive__changed_content_cannot_overwrite_same_session(session_archive__sandbox, relative):
    sandbox = session_archive__sandbox
    evidence_integrity__module.archive_session(sandbox.session)
    original = sandbox.archive.read_bytes()
    summary_path = sandbox.root / "data/experiment_summary.csv"
    summary = summary_path.read_bytes()
    path = sandbox.session / relative
    if relative.endswith(".json"):
        path.write_text(integrity__json.dumps({"status": "ROUND_COMPLETE", "note": "SYNTHETIC_CHANGED"}), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"1,1\n")
    logs = session_archive__snapshot(sandbox.session)
    with integrity__pytest.raises(ValueError, match="archived session identity already has different content"):
        evidence_integrity__module.archive_session(sandbox.session)
    assert sandbox.archive.read_bytes() == original
    assert summary_path.read_bytes() == summary
    assert session_archive__snapshot(sandbox.session) == logs
    assert not sandbox.archive.with_suffix(".lock").exists()


def test_session_archive__outside_results_is_rejected_without_changing_evidence(session_archive__sandbox):
    sandbox = session_archive__sandbox
    outside = sandbox.root / "not-results/run-001"
    outside.mkdir(parents=True)
    (outside / "round_status.json").write_bytes(b'{"status":"ROUND_COMPLETE"}\n')
    original = sandbox.archive.read_bytes()
    logs = session_archive__snapshot(outside)
    with integrity__pytest.raises(ValueError, match="session must be inside this repository's results"):
        evidence_integrity__module.archive_session(outside)
    assert sandbox.archive.read_bytes() == original
    assert session_archive__snapshot(outside) == logs
    assert not sandbox.archive.with_suffix(".lock").exists()


def test_session_archive__cli_archives_only_temporary_completed_session(session_archive__sandbox, capsys):
    sandbox = session_archive__sandbox
    logs = session_archive__snapshot(sandbox.session)
    evidence_integrity__module.main(["archive-session", "--session", str(sandbox.session)])
    result = integrity__json.loads(capsys.readouterr().out)
    assert result == dict(session="sessions/lab/run-001", files=5, new_blobs=3, original_logs_retained=True)
    assert evidence_integrity__module.read_bytes("sessions/lab/run-001/leg/leg.csv") == sandbox.payload
    assert session_archive__snapshot(sandbox.session) == logs
