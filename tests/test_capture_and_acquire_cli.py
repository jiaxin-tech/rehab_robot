"""No-motion regressions for capture and acquire entry points."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from collection.state import KinematicStateFrame
from hardware.rokae_adapter import RobotWrenchFrame
import scripts.acquire_robot_data as acquire_cli
from scripts.acquire_robot_data import run_acquisition
from scripts.capture_start_anchor import run_capture
from utils.clock import host_time_s


class FakeAdapter:
    def __init__(self, *, fail_on=()):
        self.connected = False
        self.calls = []
        self.sequence = 0
        self._state_thread = threading.Thread()
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
        now = host_time_s()
        return RobotWrenchFrame(
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


def _track_episode_logger_close(monkeypatch):
    original = acquire_cli.EpisodeLogger
    instances = []

    class TrackingEpisodeLogger(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            instances.append(self)

        def close(self, *args, **kwargs):
            self.close_calls += 1
            return super().close(*args, **kwargs)

    monkeypatch.setattr(acquire_cli, "EpisodeLogger", TrackingEpisodeLogger)
    return instances


class StopFailureAcquisition:
    stop_message = "simulated acquisition.stop failure"

    def __init__(self, adapter, logger):
        self.adapter = adapter
        self.logger = logger
        self.background_error = None

    def start(self, *, manage_connection):
        assert manage_connection is False
        return self

    def wait_until_healthy(self, timeout_s):
        return True

    def latest_health(self):
        return SimpleNamespace(invalid_reason="")

    def stop(self):
        self.adapter.calls.append("acquisition.stop")
        raise RuntimeError(self.stop_message)


class RefusingStopAcquisition(StopFailureAcquisition):
    stop_message = (
        "acquisition_threads_did_not_stop:wrench; refusing SDK disconnect "
        "while a native query may still be active"
    )


def test_capture_saves_unreviewed_anchor_and_never_moves(tmp_path):
    adapter = FakeAdapter()
    path = tmp_path / "anchor.json"
    result = run_capture(
        robot_ip="192.0.2.1",
        output_path=path,
        anchor_id="anchor_cli_test",
        tool_name="rehab_cuff",
        workpiece_name="rehab_fixture",
        adapter_factory=lambda _ip: adapter,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def test_capture_partial_connect_failure_still_attempts_disconnect(tmp_path):
    adapter = FakeAdapter(fail_on="connect")
    with pytest.raises(RuntimeError, match="simulated connect failure"):
        run_capture(
            robot_ip="192.0.2.1",
            output_path=tmp_path / "anchor.json",
            adapter_factory=lambda _ip: adapter,
        )
    assert adapter.calls == ["connect", "disconnect"]


def test_capture_stream_cleanup_failure_does_not_skip_disconnect(tmp_path):
    adapter = FakeAdapter(fail_on="stop_state_stream")
    with pytest.raises(RuntimeError, match="anchor capture cleanup failure"):
        run_capture(
            robot_ip="192.0.2.1",
            output_path=tmp_path / "anchor.json",
            tool_name="rehab_cuff",
            workpiece_name="rehab_fixture",
            adapter_factory=lambda _ip: adapter,
        )
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]
    assert adapter.connected is False


def test_acquire_writes_five_files_without_motion(tmp_path):
    adapter = FakeAdapter()
    episode = tmp_path / "episode"
    result = run_acquisition(
        robot_ip="192.0.2.1",
        episode_dir=episode,
        duration_s=0.03,
        adapter_factory=lambda _ip: adapter,
    )
    assert result["completed"]
    assert result["motion_commanded"] is False
    for filename in (
        "robot_state.csv",
        "robot_wrench.csv",
        "trajectory_command.csv",
        "aligned_snapshot.csv",
        "metadata.json",
    ):
        assert (episode / filename).is_file()
    assert "connect" in adapter.calls
    assert "disconnect" in adapter.calls
    assert not any(
        token in call
        for call in adapter.calls
        for token in ("enable", "power", "move", "drag", "calibrate", "clear_error")
    )


def test_acquire_factory_failure_still_closes_logger(monkeypatch, tmp_path):
    loggers = _track_episode_logger_close(monkeypatch)

    def fail_factory(_robot_ip):
        raise RuntimeError("simulated adapter factory failure")

    with pytest.raises(RuntimeError, match="adapter factory failure"):
        run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "factory_failure",
            duration_s=0.01,
            adapter_factory=fail_factory,
        )

    assert len(loggers) == 1
    assert loggers[0].close_calls == 1
    assert loggers[0].status == "failed"


def test_acquire_connect_failure_attempts_stop_disconnect_and_close(
    monkeypatch,
    tmp_path,
):
    loggers = _track_episode_logger_close(monkeypatch)
    adapter = FakeAdapter(fail_on="connect")

    with pytest.raises(RuntimeError, match="simulated connect failure"):
        run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "connect_failure",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
        )

    assert adapter.calls == ["connect", "stop_state_stream", "disconnect"]
    assert loggers[0].close_calls == 1
    assert loggers[0].status == "failed"


def test_acquire_stop_failure_retries_cleanup_stages_independently(
    monkeypatch,
    tmp_path,
):
    loggers = _track_episode_logger_close(monkeypatch)
    monkeypatch.setattr(acquire_cli, "RealRobotAcquisition", StopFailureAcquisition)
    adapter = FakeAdapter(fail_on="stop_state_stream")

    with pytest.raises(RuntimeError, match="acquisition.stop failure"):
        run_acquisition(
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


def test_acquire_does_not_bypass_live_thread_disconnect_refusal(
    monkeypatch,
    tmp_path,
):
    loggers = _track_episode_logger_close(monkeypatch)
    monkeypatch.setattr(
        acquire_cli,
        "RealRobotAcquisition",
        RefusingStopAcquisition,
    )
    adapter = FakeAdapter()

    with pytest.raises(RuntimeError, match="refusing SDK disconnect"):
        run_acquisition(
            robot_ip="192.0.2.1",
            episode_dir=tmp_path / "refused_disconnect",
            duration_s=0.01,
            adapter_factory=lambda _ip: adapter,
        )

    assert adapter.calls == ["connect", "acquisition.stop"]
    assert adapter.connected is True
    assert loggers[0].close_calls == 0
    assert loggers[0].status == "failed"
