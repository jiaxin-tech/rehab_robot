"""Offline concurrency tests for independent real-episode acquisition."""

from __future__ import annotations

import threading
import time

import pytest

from collection.episode_logger import EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition
from collection.state import KinematicStateFrame
from hardware.rokae_adapter import RobotWrenchFrame
from utils.clock import host_time_s


class FakeAdapter:
    def __init__(self, *, block_wrench: bool = False) -> None:
        self.connected = False
        self.sequence = 0
        self.calls: list[str] = []
        self.block_wrench = block_wrench
        self.wrench_entered = threading.Event()
        self.release_wrench = threading.Event()

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
        now = host_time_s()
        return KinematicStateFrame(
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
        start = host_time_s()
        end = host_time_s()
        return RobotWrenchFrame(
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


class _SignalOnlyLogger:
    """Logger double that rejects every lock-taking failure path."""

    def __init__(self) -> None:
        self.failure_event = threading.Event()
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


class _LifecycleLogger(_SignalOnlyLogger):
    def __init__(self) -> None:
        super().__init__()
        self.durable_failures: list[str] = []

    @property
    def healthy(self):
        return not self.failure_event.is_set()

    def mark_failed(self, reason: str) -> None:
        self.durable_failures.append(reason)


class _ControlledThread:
    def __init__(
        self,
        *,
        name: str,
        fail_start: bool,
        remain_alive_after_join: bool,
        stop_event: threading.Event,
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


class _ControlledThreadFactory:
    def __init__(self, *, stuck_state: bool) -> None:
        self.stuck_state = stuck_state
        self.stop_event: threading.Event | None = None
        self.created: dict[str, _ControlledThread] = {}

    def __call__(self, *, target, name: str, daemon: bool):
        assert self.stop_event is not None
        thread = _ControlledThread(
            name=name,
            fail_start=name == "real-episode-wrench",
            remain_alive_after_join=(
                self.stuck_state and name == "real-episode-state"
            ),
            stop_event=self.stop_event,
        )
        self.created[name] = thread
        return thread


def test_independent_streams_publish_and_health_is_visible(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    adapter = FakeAdapter()
    acquisition = RealRobotAcquisition(adapter, logger).start()
    try:
        assert acquisition.wait_until_healthy(1.0)
        health = acquisition.latest_health()
        assert health.valid
        assert health.state_age_s is not None
        assert health.wrench_age_s is not None
        assert health.state_wrench_skew_s is not None
        assert logger.row_counts["robot_state"] > 0
        assert logger.row_counts["robot_wrench"] > 0
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


def test_blocking_wrench_does_not_block_state_or_command_logging(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    adapter = FakeAdapter(block_wrench=True)
    acquisition = RealRobotAcquisition(adapter, logger).start()
    assert adapter.wrench_entered.wait(timeout=0.5)
    try:
        time.sleep(0.04)
        assert logger.row_counts["robot_state"] >= 2
        logger.append_trajectory_command(
            host_time_s=host_time_s(),
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


def test_acquisition_never_calls_motion_methods(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    adapter = FakeAdapter()
    acquisition = RealRobotAcquisition(adapter, logger).start()
    acquisition.stop()
    logger.close(completed=True)
    assert not any(
        token in call
        for call in adapter.calls
        for token in ("enable", "power", "move", "drag", "calibrate", "clear_error")
    )


def test_stuck_wrench_refuses_concurrent_disconnect(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    adapter = FakeAdapter(block_wrench=True)
    acquisition = RealRobotAcquisition(
        adapter,
        logger,
        join_timeout_s=0.02,
    ).start()
    assert adapter.wrench_entered.wait(timeout=0.5)
    with pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        acquisition.stop()
    assert "stop_state_stream" not in adapter.calls
    assert "disconnect" not in adapter.calls
    assert logger.failed
    adapter.release_wrench.set()
    logger.close(completed=False)


def test_stuck_producer_stop_uses_only_nonblocking_failure_signal():
    logger = _SignalOnlyLogger()
    adapter = FakeAdapter(block_wrench=True)
    acquisition = RealRobotAcquisition(
        adapter,
        logger,
        join_timeout_s=0.02,
    ).start()
    assert adapter.wrench_entered.wait(timeout=0.5)

    with pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        acquisition.stop()

    assert logger.failure_event.is_set()
    assert logger.signals
    assert logger.signals[-1][1] == "acquisition_stop"
    assert "stop_state_stream" not in adapter.calls
    assert "disconnect" not in adapter.calls

    adapter.release_wrench.set()
    acquisition.stop()
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]


def test_partial_start_failure_joins_started_thread_before_cleanup():
    logger = _LifecycleLogger()
    adapter = FakeAdapter()
    factory = _ControlledThreadFactory(stuck_state=False)
    acquisition = RealRobotAcquisition(
        adapter,
        logger,
        thread_factory=factory,
    )
    factory.stop_event = acquisition._stop_event

    with pytest.raises(RuntimeError, match="synthetic start failure"):
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


def test_partial_start_failure_refuses_cleanup_while_started_thread_is_alive():
    logger = _LifecycleLogger()
    adapter = FakeAdapter()
    factory = _ControlledThreadFactory(stuck_state=True)
    acquisition = RealRobotAcquisition(
        adapter,
        logger,
        join_timeout_s=0.01,
        thread_factory=factory,
    )
    factory.stop_event = acquisition._stop_event

    with pytest.raises(RuntimeError, match="refusing SDK cleanup/disconnect"):
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
