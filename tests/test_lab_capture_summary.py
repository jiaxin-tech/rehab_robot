import hashlib
import json

import pytest

from scripts.lab_capture_summary import summarize_capture


def fixture_capture(directory, *, findings=None):
    directory.mkdir()
    raw = directory / "leg.csv"
    raw.write_bytes(b"time_s,value\n0,1\n1,2\n")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    segment = dict(status="CAPTURE_COMPLETE", motion_commanded=False, rows=2, csv_sha256=digest)
    quality = dict(rows=2, sha256=digest, findings=[] if findings is None else findings,
                   sample_rate_hz=1., usable_for_dynamics=not findings,
                   long_intervals=[], receive_time_duplicates_csv_lines=[])
    for name, data in (("segment_status.json", segment), ("quality.json", quality)):
        (directory / name).write_text(json.dumps(data), encoding="utf-8")
    return raw, segment, quality


def test_complete_recording_keeps_quality_findings_distinct(tmp_path):
    directory = tmp_path / "capture"
    fixture_capture(directory, findings=["receive_time_gaps", "hip_has_no_joint_angle_calibration"])
    result = summarize_capture(directory)
    assert result["recording_complete"] is True
    assert result["usable_for_dynamics"] is False
    assert result["findings"] == ["receive_time_gaps", "hip_has_no_joint_angle_calibration"]
    assert result["motion_approved"] is False


@pytest.mark.parametrize("kind", ["raw_changed", "partial", "motion", "quality_hash", "row_count"])
def test_capture_evidence_must_bind_complete_current_raw_file(tmp_path, kind):
    directory = tmp_path / "capture"
    raw, segment, quality = fixture_capture(directory)
    if kind == "raw_changed":
        raw.write_bytes(raw.read_bytes() + b"2,3\n")
    if kind == "partial":
        segment["status"] = "CAPTURE_FAILED"
    if kind == "motion":
        segment["motion_commanded"] = True
    if kind == "quality_hash":
        quality["sha256"] = "0" * 64
    if kind == "row_count":
        segment["rows"] = 3
    (directory / "segment_status.json").write_text(json.dumps(segment), encoding="utf-8")
    (directory / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    with pytest.raises(ValueError):
        summarize_capture(directory)
