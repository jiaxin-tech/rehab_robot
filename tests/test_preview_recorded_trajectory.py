"""Offline, synthetic file tests; no devices or controller imports."""
import csv
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import preview_recorded_trajectory as previewer


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_recording(tmp_path, *, status="COMPLETE", poses=None, times=None, invalid=None):
    directory = tmp_path / "recording"
    directory.mkdir()
    poses = poses if poses is not None else [[.7, -.1, .3, 0, 0, 0], [.72, -.08, .35, 0, 0, 0], [.7, -.1, .3, 0, 0, 0]]
    times = [100+i*.05 for i in range(len(poses))] if times is None else times
    invalid = invalid or set()
    rows = []
    for index, (pose, time) in enumerate(zip(poses, times)):
        row = dict(row_index=index, row_kind="state", host_time_s=time,
            query_start_s=time+.001, query_end_s=time+.002, sequence_id=index+1,
            frame_valid=index not in invalid, valid=index not in invalid,
            invalid_reason="fake invalid" if index in invalid else "")
        row.update(zip(previewer.POSE_FIELDS, pose))
        rows.append(row)
    with (directory / "recorded_tcp.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=previewer.REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "config_snapshot.json", {"robot": {"ip": "fake"}})
    write_json(directory / "recording_status.json", {"schema_version": 1, "status": status,
        "partial": status != "COMPLETE", "ready_for_replay": False, "error": None})
    manifest = dict(schema_version=1, ready_for_replay=False,
        config_sha256=sha(directory / "config_snapshot.json"),
        config_snapshot_sha256=sha(directory / "config_snapshot.json"),
        units={"position": "base_m", "orientation": "euler_xyz_rad", "joints": "rad"},
        source_sha256={"recorder": "recorded-fake-source"}, source_git={"commit": "fake"})
    manifest["output_files_sha256"] = {name: sha(directory / name) for name in ("recorded_tcp.csv", "recording_status.json")}
    write_json(directory / "recording_manifest.json", manifest)
    return directory


def refresh(directory):
    path = directory / "recording_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_files_sha256"] = {name: sha(directory / name) for name in ("recorded_tcp.csv", "recording_status.json")}
    write_json(path, manifest)


def test_complete_closed_preview_preserves_absolute_and_relative_geometry(tmp_path):
    recording = make_recording(tmp_path)
    original = {path.name: path.read_bytes() for path in recording.iterdir()}
    output = tmp_path / "preview"
    result = previewer.preview(recording, output)
    assert result["status"] == "PREVIEW_COMPLETE" and result["ok"]
    assert result["ready_for_replay"] is False and result["hardware_accessed"] is False
    assert result["input_hashes_verified"] is True
    assert result["position_closure_error_m"] == 0
    assert result["maximum_orientation_change_deg"] == 0
    assert result["review_flags"] == []
    assert result["coordinates_rescaled"] is False
    assert result["absolute_base_xyz_m"]["z"]["max"] == .35
    assert result["relative_to_first_valid_xyz_m"]["z"]["max"] == pytest.approx(.05)
    assert result["max_z_candidate"]["classification"] == "MAX_Z_CANDIDATE_UNCONFIRMED"
    assert result["max_z_candidate"]["valid_sample_index"] == 1
    assert result["branch_split_confirmed"] is False
    assert {path.name: path.read_bytes() for path in recording.iterdir()} == original
    assert set(result["generated_files_sha256"]) == {"trajectory_3d.png", "projections.png", "xyz_time.png", "orientation_time.png", "closure_review.png"}
    for name, digest in result["generated_files_sha256"].items():
        assert sha(output / name) == digest
        with Image.open(output / name) as image:
            assert image.width >= 1000 and image.height >= 700
    assert json.loads((output / "summary.json").read_text())["ready_for_replay"] is False


@pytest.mark.parametrize("name", ["recorded_tcp.csv", "recording_status.json", "config_snapshot.json"])
def test_modified_original_sha_rejected_without_plots(tmp_path, name):
    recording = make_recording(tmp_path)
    with (recording / name).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "FAILED" and result["ready_for_replay"] is False
    assert "SHA-256 mismatch" in result["error"]
    assert not list((tmp_path / "preview").glob("*.png"))


@pytest.mark.parametrize("times", [[100, 100.1, 100.05], [100, 100, 100.1]])
def test_disordered_or_repeated_time_is_not_sorted_or_repaired(tmp_path, times):
    recording = make_recording(tmp_path, times=times)
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "FAILED"
    assert "duplicated or out of order" in result["error"]


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "RECORDING"])
def test_partial_recording_can_be_reviewed_without_replay_authorization(tmp_path, status):
    recording = make_recording(tmp_path, status=status)
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "PARTIAL" and result["ok"]
    assert result["recording_status"] == status
    assert result["ready_for_replay"] is False
    assert "PARTIAL_RECORDING_OR_INVALID_ROWS" in result["review_flags"]


def test_invalid_rows_and_time_gap_break_drawn_segments(tmp_path):
    recording = make_recording(tmp_path, poses=[[.7, 0, .3, 0, 0, 0], [.8, 0, .4, 0, 0, 0], [.7, 0, .3, 0, 0, 0]],
        times=[100, 100.05, 100.3], invalid={1})
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "PARTIAL"
    assert result["valid_sample_count"] == 2 and len(result["discarded_rows"]) == 1
    assert result["plotted_contiguous_segments"] == 2
    assert result["gaps"][0]["elapsed_s"] == pytest.approx(.3)
    assert "TIME_GAPS_EXCEED_REVIEW_SCREEN" in result["review_flags"]
    assert result["samples_interpolated"] is False


def test_open_end_and_large_orientation_are_review_flags_not_geometry_repairs(tmp_path):
    recording = make_recording(tmp_path, poses=[[.7, 0, .3, 0, 0, 0], [.72, 0, .35, 0, 0, .3]])
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["position_closure_error_m"] == pytest.approx(np.sqrt(.02**2+.05**2))
    assert result["maximum_orientation_change_deg"] == pytest.approx(np.degrees(.3))
    assert result["orientation_closure_error_deg"] == pytest.approx(np.degrees(.3))
    assert set(result["review_flags"]) == {"END_POSITION_DIFFERS_FROM_START", "LARGE_ORIENTATION_CHANGE", "END_ORIENTATION_DIFFERS_FROM_START"}
    assert result["end_pose_base_m_rad"][5] == .3


def test_euler_wrap_does_not_invent_a_full_orientation_turn(tmp_path):
    recording = make_recording(tmp_path, poses=[[.7, 0, .3, 0, 0, np.pi-.01], [.7, 0, .3, 0, 0, -np.pi+.01]])
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["maximum_orientation_change_deg"] == pytest.approx(np.degrees(.02))
    assert "LARGE_ORIENTATION_CHANGE" not in result["review_flags"]


def test_nonfinite_valid_pose_rejected(tmp_path):
    recording = make_recording(tmp_path, poses=[[.7, 0, .3, 0, 0, 0], [float("nan"), 0, .4, 0, 0, 0]])
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "FAILED"
    assert "must be finite" in result["error"]


def test_manifest_paths_cannot_read_outside_session(tmp_path):
    recording = make_recording(tmp_path)
    manifest_path = recording / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_files_sha256"]["../outside.txt"] = "anything"
    write_json(manifest_path, manifest)
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "FAILED"
    assert "without a path" in result["error"]


def test_units_cannot_silently_be_guessed_or_scaled(tmp_path):
    recording = make_recording(tmp_path)
    manifest_path = recording / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["units"]["position"] = "mm"
    write_json(manifest_path, manifest)
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "FAILED"
    assert "base metres" in result["error"]


def test_offline_boundary_and_existing_output_refusal(tmp_path, monkeypatch):
    recording = make_recording(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("offline preview attempted a network connection")
    monkeypatch.setattr(socket, "socket", forbidden)
    assert previewer.main(["--session", str(recording), "--output-dir", str(tmp_path / "preview")]) == 0
    with pytest.raises(FileExistsError):
        previewer.preview(recording, tmp_path / "preview")
    source = Path(previewer.__file__).read_text(encoding="utf-8")
    assert "from hardware" not in source and "import hardware" not in source
    assert "subprocess" not in source and "ctypes" not in source


def test_cli_failure_writes_diagnostic_summary_and_returns_one(tmp_path):
    recording = make_recording(tmp_path, invalid={0, 1})
    assert previewer.main(["--session", str(recording), "--output-dir", str(tmp_path / "preview")]) == 1
    assert json.loads((tmp_path / "preview/summary.json").read_text())["ready_for_replay"] is False


def test_plot_write_failure_cannot_be_reported_as_completed_preview(tmp_path, monkeypatch):
    recording = make_recording(tmp_path)
    def fail_plot(*args):
        raise OSError("simulated image write failure")
    monkeypatch.setattr(previewer, "_plot", fail_plot)
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["status"] == "FAILED" and result["ok"] is False
    assert result["ready_for_replay"] is False
    assert "image write failure" in result["error"]


def test_current_recorder_output_integrates_with_preview_without_a_real_adapter(tmp_path, monkeypatch):
    from scripts import record_rehab_trajectory as recorder

    class Clock:
        value = 100.
        def __call__(self):
            return self.value
        def sleep(self, duration):
            self.value += duration

    clock = Clock()

    class StateOnlyAdapter:
        reads = 0
        def connect(self):
            pass
        def start_state_stream(self):
            pass
        def begin_state_capture(self):
            return False
        def read_state_frame(self):
            self.reads += 1
            return SimpleNamespace(sequence_id=self.reads, host_monotonic_time_s=clock(),
                wall_time_iso="2026-11-14T00:00:00Z", robot_device_time_s=None,
                valid=True, invalid_reason="", tcp_position_m=(.7, -.1, .3+self.reads*.0001),
                tcp_orientation_rad=(0., 0., 0.), joint_position_rad=(0.,)*6, operation_state="DRAG")
        def stop_state_stream(self):
            pass
        def disconnect(self):
            pass
        def __getattr__(self, name):
            raise AssertionError("forbidden device interface: " + name)

    config = tmp_path / "fake_config.json"
    write_json(config, {"robot": {"ip": "192.0.2.1", "local_ip": "192.0.2.2",
        "robot_class": "xMateRobot", "collision_source": "controller"},
        "motion": {"max_state_age_s": .1}})
    monkeypatch.setattr(recorder, "_source_git", lambda: {"commit": "SYNTHETIC_ONLY"})
    recording = tmp_path / "recording"
    state = recorder.record(config, recording, duration_s=.025, live=True,
        adapter_factory=lambda c: StateOnlyAdapter(), clock=clock, sleep=clock.sleep,
        announce=lambda message: None)
    assert state["status"] == "COMPLETE"
    result = previewer.preview(recording, tmp_path / "preview")
    assert result["ok"] is True and result["input_hashes_verified"] is True
    assert result["valid_sample_count"] == state["summary"]["valid_samples"]
    assert result["ready_for_replay"] is False
