"""Unit/schema regressions; no controller imports or lab file dependency."""
import json

import numpy as np
import pandas as pd
import pytest

from measurement_validation.leg_episode import CalibrationError, JointCalibration, read_leg_episode
from measurement_validation.offline_leg import calibrate, inspect_file, main


def recording(tmp_path, *, name="record.csv", rows=1001):
    t = np.arange(rows) * 2.0
    frame = pd.DataFrame({"receive_elapsed_ms": t,
                          "host_timestamp_filetime_100ns": 134361865866568070 + np.arange(rows) * 20000,
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


def test_real_pdo_schema_has_explicit_units_and_status(tmp_path):
    path, _ = recording(tmp_path)
    ep = read_leg_episode(path)
    assert ep.host_time_s[-1] == 2
    hip = ep.joints["hip"]
    assert hip.position_counts[0] == -100000
    assert hip.motor_position_deg[0] == pytest.approx(-45.77637)
    assert hip.statusword[0] == 561
    assert hip.error_code[0] == 0
    assert not hip.operation_enabled.any() and not hip.fault.any()
    assert hip.torque_nm[0] == -4.5
    assert hip.flexion_deg is None


def test_motor_degrees_never_become_counts(tmp_path):
    path, frame = recording(tmp_path)
    frame.drop(columns=["axis1_position_raw", "axis4_position_raw"]).to_csv(path, index=False)
    ep = read_leg_episode(path)
    assert ep.joints["hip"].position_counts is None
    assert ep.joints["hip"].motor_position_deg[0] == pytest.approx(-45.77637)
    cal = JointCalibration(1000, -100000, -1, -10, 90)
    with pytest.raises(CalibrationError, match="raw position"):
        read_leg_episode(path, calibration={"hip": cal, "knee": cal})


def test_time_overrides_and_filetime_subtract_before_float(tmp_path):
    path, frame = recording(tmp_path)
    assert read_leg_episode(path, column_map={"time": "receive_elapsed_ms"}).host_time_s[-1] == 2
    frame = frame.drop(columns="receive_elapsed_ms")
    frame.to_csv(path, index=False)
    assert read_leg_episode(path).host_time_s[1] == .002
    frame["host_timestamp_filetime_100ns"] = 134361865866568070 + np.arange(len(frame)) * 3
    frame.to_csv(path, index=False)
    assert read_leg_episode(path).host_time_s[1] == 3e-7


def test_quality_locates_gaps_missing_duplicate_clock_and_motion(tmp_path):
    path, frame = recording(tmp_path)
    frame.loc[500:, "receive_elapsed_ms"] += 50
    frame.loc[8, "host_timestamp_filetime_100ns"] = frame.loc[7, "host_timestamp_filetime_100ns"]
    frame.loc[25, "axis1_torque_3c1a_nm"] = np.nan
    frame.loc[700:, "axis1_position_raw"] -= 30000
    frame.to_csv(path, index=False)
    report = inspect_file(path)
    assert report["long_intervals"][0]["csv_line"] == 502
    assert report["timing_missing_estimate"] == 25
    assert report["secondary_clocks"]["host_timestamp_filetime_100ns"]["duplicate_csv_lines"] == [10]
    assert report["missing_values_by_column"]["axis1_torque_3c1a_nm"] == 1
    assert not report["joint_checks"]["hip"]["whole_segment_stability"]["stable"]
    assert report["usable_for_dynamics"] is False
    json.dumps(report, allow_nan=False)


def test_batch_continues_and_reports_bad_file(tmp_path):
    path, _ = recording(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("unrelated\n1\n", encoding="utf-8")
    out = tmp_path / "out"
    assert main(["quality", str(path), str(bad), "--output-dir", str(out)]) == 1
    results = json.loads((out / "summary.json").read_text())
    assert {x["status"] for x in results} == {"reported", "error"}


def test_calibration_keeps_validation_out_of_fit_and_knee_missing(tmp_path):
    points = []
    for name, angle, counts, role in [("zero", 0, -100000, "fit"), ("thirty", 30, -130000, "fit"),
                                      ("independent", 15, -115200, "validation"),
                                      ("returned", None, -100300, "return")]:
        path, frame = recording(tmp_path, name=name + ".csv")
        frame["axis1_position_raw"] = counts
        frame.to_csv(path, index=False)
        points.append(dict(id=name, joint="hip", file=path.name, start_s=0, end_s=2.01,
                           angle_deg=angle, role=role, **({"pair_id": "zero_return"} if name in ("zero", "returned") else {})))
    candidate = dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000)
    result = calibrate({"points": points}, tmp_path, candidate)
    assert result["fits"]["knee"] is None
    assert result["fits"]["hip"]["counts_per_degree"] == pytest.approx(1000)
    assert result["points"][2]["prediction_error_deg"] == pytest.approx(.2)
    assert result["points"][3]["prediction_error_deg"] is None
    assert result["pairs"][0]["difference_deg"] == pytest.approx(.3)
    assert not result["pairs"][0]["same_measured_angle"]
    path = tmp_path / points[1]["file"]
    frame = pd.read_csv(path)
    frame["axis1_position_raw"] = np.linspace(-130000, -100000, len(frame))
    frame.to_csv(path, index=False)
    result = calibrate({"points": points}, tmp_path, candidate)
    assert not result["points"][1]["accepted"]
    assert result["fits"]["hip"] is None


def test_source_hash_change_rejected(tmp_path):
    path, _ = recording(tmp_path)
    spec = dict(id="p", joint="hip", file=path.name, start_s=0, end_s=2,
                angle_deg=0, role="fit", sha256="wrong")
    with pytest.raises(ValueError, match="hash changed"):
        calibrate({"points": [spec]}, tmp_path,
                  dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000))


def test_overlap_cannot_masquerade_as_independent_validation(tmp_path):
    path, _ = recording(tmp_path)
    base = dict(joint="hip", file=path.name, start_s=0, end_s=2, angle_deg=0)
    with pytest.raises(ValueError, match="overlap"):
        calibrate({"points": [dict(base, id="fit", role="fit"), dict(base, id="check", role="validation")]},
                  tmp_path, dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000))


def test_all_missing_counts_reported_and_excluded_from_fit(tmp_path):
    path, frame = recording(tmp_path)
    frame["axis1_position_raw"] = np.nan
    frame["axis1_position_motor_deg"] = np.nan
    frame.to_csv(path, index=False)
    report = inspect_file(path)
    assert report["joint_checks"]["hip"]["position_counts"]["mean"] is None
    json.dumps(report, allow_nan=False)
    result = calibrate({"points": [dict(id="missing", joint="hip", file=path.name,
                       start_s=0, end_s=2, angle_deg=0, role="fit")]}, tmp_path,
                       dict(flexion_sign=-1, zero_counts=-100000, counts_per_degree=1000))
    assert result["points"][0]["accepted"] is False
    assert result["points"][0]["prediction_error_deg"] is None


def test_faults_and_missing_state_block_dynamics_even_with_calibration(tmp_path):
    path, frame = recording(tmp_path)
    frame.loc[3, "axis1_error_code_hex"] = "0x1234"
    frame.loc[3, "axis1_fault"] = "true"
    frame.to_csv(path, index=False)
    cal = JointCalibration(1000, -100000, -1, -10, 90)
    from measurement_validation.leg_episode import quality_report
    report = quality_report(read_leg_episode(path, calibration={"hip": cal, "knee": cal}))
    assert report["usable_for_dynamics"] is False
    assert "hip_nonzero_error" in report["findings"]
    assert "hip_fault_present" in report["findings"]


def test_utc_duplicates_are_distinct_from_receive_duplicates(tmp_path):
    path, frame = recording(tmp_path)
    utc = pd.Series(pd.date_range("2026-10-11", periods=len(frame), freq="2ms", tz="UTC"))
    utc.iloc[9] = utc.iloc[8]
    frame["host_timestamp_utc"] = utc.astype(str)
    frame.to_csv(path, index=False)
    result = inspect_file(path)
    assert result["receive_time_duplicates_csv_lines"] == []
    assert result["secondary_clocks"]["host_timestamp_utc"]["duplicate_csv_lines"] == [11]
    frame.drop(columns=["receive_elapsed_ms", "host_timestamp_filetime_100ns"]).to_csv(path, index=False)
    ep = read_leg_episode(path)
    assert ep.host_time_s[-1] == 2


def test_oct7_elapsed_ms_and_torqueclose_schema(tmp_path):
    path, frame = recording(tmp_path)
    frame.rename(columns={"receive_elapsed_ms": "elapsed_ms",
                          "axis1_torque_3c1a_nm": "axis1_torqueclose_nm",
                          "axis4_torque_3c1a_nm": "axis4_torqueclose_nm"}).to_csv(path, index=False)
    ep = read_leg_episode(path)
    assert ep.host_time_s[-1] == 2
    assert ep.joints["hip"].torque_nm[0] == -4.5


def test_sparse_constant_positions_do_not_pass_stability():
    from measurement_validation.offline_leg import stability
    check = stability(np.array([0, .5, 1.]), np.array([1, 1, 1.]), max_range=50,
                      max_slope=20, min_duration=.8)
    assert not check["stable"]
    assert "time_gap_exceeded" in check["reasons"]
