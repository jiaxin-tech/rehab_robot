import csv
from dataclasses import replace
import hashlib
import json

import pytest

from collection.state import KinematicStateFrame
from scripts import record_rehab_trajectory as recorder


class Clock:
    def __init__(self):
        self.value = 100.

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def frame(clock, sequence):
    return KinematicStateFrame(
        sequence_id=sequence, host_monotonic_time_s=clock(), wall_time_iso="2026-11-14T00:00:00Z",
        robot_device_time_s=None, valid=True, invalid_reason="",
        tcp_position_m=(.7+sequence*.001, -.1, .3), tcp_orientation_rad=(-3., .1, .7),
        tcp_linear_velocity_mps=None, tcp_angular_velocity_radps=None, velocity_source="unavailable",
        joint_position_rad=(.1, .2, .3, .4, .5, .6), joint_velocity_radps=None,
        pose_time_s=clock(), joint_time_s=clock(), velocity_time_s=None,
        operation_state="DRAG", collision_state=None, controller_error=None,
    )


class StateOnlyAdapter:
    """Every unlisted interface, especially native/control access, fails."""
    def __init__(self, clock, reader=None, *, cleanup_error=None, connect_error=False,
                 retained=False):
        self.clock = clock
        self.reader = reader
        self.calls = []
        self.reads = 0
        self.cleanup_error = cleanup_error
        self.connect_error = connect_error
        self.retained = retained

    def __getattr__(self, name):
        raise AssertionError("forbidden/non-state interface: " + name)

    def connect(self):
        self.calls.append("connect")
        if self.connect_error:
            raise RuntimeError("connect failed")

    def start_state_stream(self):
        self.calls.append("start_state_stream")

    def begin_state_capture(self):
        self.calls.append("begin_state_capture")
        return self.retained

    def read_state_frame(self):
        self.calls.append("read_state_frame")
        self.reads += 1
        return self.reader(self.reads) if self.reader else frame(self.clock, self.reads)

    def drain_state_frames(self):
        self.calls.append("drain_state_frames")
        return ()

    def end_state_capture(self):
        self.calls.append("end_state_capture")

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")
        if self.cleanup_error == "stop_state_stream":
            raise RuntimeError("stop failure")

    def disconnect(self):
        self.calls.append("disconnect")
        if self.cleanup_error == "disconnect":
            raise RuntimeError("disconnect failure")


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "robot": {"ip": "192.0.2.1", "local_ip": "192.0.2.2", "robot_class": "xMateRobot",
                  "collision_source": "controller", "collision_event_max_age_s": None},
        "motion": {"max_state_age_s": .1},
    }), encoding="utf-8")
    return path


def run(config_path, tmp_path, adapter, clock, **kwargs):
    return recorder.record(config_path, tmp_path / "recording", duration_s=.025, live=True,
                           adapter_factory=lambda c: adapter, clock=clock, sleep=clock.sleep,
                           announce=lambda message: None, **kwargs)


def read_rows(tmp_path):
    with (tmp_path / "recording/recorded_tcp.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_raw_state_only_complete_and_hashes(config_path, tmp_path):
    clock = Clock()
    adapter = StateOnlyAdapter(clock, retained=True)
    state = run(config_path, tmp_path, adapter, clock)
    assert state["status"] == "COMPLETE"
    assert state["ready_for_replay"] is False
    assert state["summary"]["actual_span_s"] > 0
    assert not any(state["project_commands_sent"].values())
    assert adapter.calls[-4:] == ["stop_state_stream", "drain_state_frames", "end_state_capture", "disconnect"]
    rows = read_rows(tmp_path)
    assert float(rows[0]["tcp_x"]) == .701
    assert float(rows[0]["tcp_rx"]) == -3.
    assert float(rows[0]["q6"]) == .6
    assert rows[0]["operation_state"] == "DRAG"
    assert rows[0]["robot_device_time_s"] == ""
    manifest = json.loads((tmp_path / "recording/recording_manifest.json").read_text())
    for name, digest in manifest["output_files_sha256"].items():
        assert hashlib.sha256((tmp_path / "recording" / name).read_bytes()).hexdigest() == digest
    assert manifest["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert "scripts/record_rehab_trajectory.py" in manifest["source_sha256"]
    assert "config/bundle.py" in manifest["source_sha256"]
    assert state["operation_state_semantics"] == manifest["operation_state_semantics"]
    assert "not independent verification of OEM drag mode" in state["operation_state_semantics"]
    assert "controller/session side effects" in manifest["vendor_session_side_effects"]


def test_duplicates_are_events_not_new_state_samples(config_path, tmp_path):
    clock = Clock()
    repeated = frame(clock, 1)
    adapter = StateOnlyAdapter(clock, reader=lambda i: repeated)
    state = run(config_path, tmp_path, adapter, clock)
    assert len(read_rows(tmp_path)) == 1
    assert state["duplicate_polls"] >= 3
    assert all(e["kind"] == "duplicate_state" for e in state["events"])
    assert state["status"] == "FAILED"  # one repeated point is not a complete trajectory


def test_equal_timestamp_with_new_sequence_is_not_new_sample(config_path, tmp_path):
    clock = Clock()
    stamp = clock()
    adapter = StateOnlyAdapter(clock, reader=lambda i: replace(frame(clock, i), host_monotonic_time_s=stamp))
    state = run(config_path, tmp_path, adapter, clock)
    assert state["summary"]["state_samples"] == 1
    assert state["duplicate_polls"] > 0


def test_sequence_regression_is_preserved_but_invalid(config_path, tmp_path):
    clock = Clock()
    adapter = StateOnlyAdapter(clock, reader=lambda i: frame(clock, 10 if i == 1 else i))
    run(config_path, tmp_path, adapter, clock)
    rows = read_rows(tmp_path)
    assert rows[0]["valid"] == "True"
    assert rows[1]["valid"] == "False"
    assert "state_sequence_not_increasing" in rows[1]["invalid_reason"]


def test_invalid_stale_and_nonincreasing_frames_preserved(config_path, tmp_path):
    clock = Clock()
    def read(i):
        f = frame(clock, i)
        if i == 2:
            return replace(f, valid=False, invalid_reason="vendor_state_error")
        if i == 3:
            return replace(f, host_monotonic_time_s=99.)
        return f
    state = run(config_path, tmp_path, StateOnlyAdapter(clock, read), clock)
    rows = read_rows(tmp_path)
    assert rows[1]["frame_invalid_reason"] == "vendor_state_error"
    assert rows[1]["valid"] == "False"
    assert "state_stale_or_host_time_invalid" in rows[2]["invalid_reason"]
    assert "state_host_time_not_increasing" in rows[2]["invalid_reason"]
    assert state["stale_polls"] == 1
    assert state["summary"]["invalid_rows"] == 2


@pytest.mark.parametrize("exception,expected", [(KeyboardInterrupt, "CANCELLED"), (RuntimeError, "FAILED")])
def test_cancel_or_read_error_retains_partial_and_cleanup(config_path, tmp_path, exception, expected):
    clock = Clock()
    def read(i):
        if i == 3:
            raise exception("interrupted read")
        return frame(clock, i)
    adapter = StateOnlyAdapter(clock, read)
    state = run(config_path, tmp_path, adapter, clock)
    assert state["status"] == expected
    assert state["partial"] is True
    assert state["summary"]["valid_samples"] == 2
    assert adapter.calls[-2:] == ["stop_state_stream", "disconnect"]
    if expected == "FAILED":
        assert read_rows(tmp_path)[-1]["row_kind"] == "read_error"
    manifest = json.loads((tmp_path / "recording/recording_manifest.json").read_text())
    assert len(manifest["output_files_sha256"]) == 2


@pytest.mark.parametrize("failure", ["stop_state_stream", "disconnect"])
def test_cleanup_error_is_failed_and_disconnect_is_attempted(config_path, tmp_path, failure):
    clock = Clock()
    adapter = StateOnlyAdapter(clock, cleanup_error=failure)
    state = run(config_path, tmp_path, adapter, clock)
    assert state["status"] == "FAILED"
    assert state["partial"] is True
    assert "cleanup_failed:" + failure in state["error"]
    assert adapter.calls[-1] == "disconnect"
    assert any(row.get("ok") is False for row in state["cleanup"])


def test_connection_error_preserved_and_cleaned(config_path, tmp_path):
    clock = Clock()
    adapter = StateOnlyAdapter(clock, connect_error=True)
    state = run(config_path, tmp_path, adapter, clock)
    assert state["status"] == "FAILED"
    assert "connect failed" in state["error"]
    assert adapter.calls == ["connect", "stop_state_stream", "disconnect"]


def test_refuse_overwrite_and_no_live_before_adapter(config_path, tmp_path):
    created = []
    factory = lambda c: created.append(c)
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        recorder.record(config_path, directory, live=True, adapter_factory=factory)
    with pytest.raises(PermissionError):
        recorder.record(config_path, tmp_path / "not_live", adapter_factory=factory)
    for duration in [0., 301., float("nan")]:
        with pytest.raises(ValueError):
            recorder.record(config_path, tmp_path / "invalid", live=True, duration_s=duration,
                            adapter_factory=factory)
    assert created == []
    assert (directory / "keep").read_text() == "original"


def test_configured_robot_factory_uses_only_observation_adapter(config_path, monkeypatch):
    import hardware.rokae_adapter
    calls = []
    monkeypatch.setattr(hardware.rokae_adapter, "RokaeRobotAdapter",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    recorder._adapter(json.loads(config_path.read_text()))
    assert calls == [(("192.0.2.1",), dict(local_ip="192.0.2.2", robot_class="xMateRobot",
                   collision_source="controller", collision_event_max_age_s=None))]
