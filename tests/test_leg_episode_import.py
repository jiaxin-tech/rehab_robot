"""Regression tests for the TwinCAT dummy-leg episode importer.

The fixtures mirror the 2026-10-07 acquisition exactly enough to keep the two
real traps visible: axis-sequential reads at ~2.6 Hz, and a static torque
offset that moves with pose.  No fixture is presented as identification data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from measurement_validation.leg_episode import (
    CalibrationError,
    JointCalibration,
    LegImportError,
    MOTOR_SIDE_DEG_PER_COUNT,
    quality_report,
    read_leg_episode,
)
from scripts.import_dummy_leg_episode import main as import_main


def _write_episode(path: Path, rows: int, rate_hz: float, *, atomic: bool = False) -> Path:
    """One CSV with the column names the handover record described."""

    times = np.arange(rows, dtype=float) / rate_hz
    # Hip flexion decreases counts; knee flexion increases counts.
    axis1_counts = 100_000.0 - np.linspace(0.0, 5_000.0, rows)
    axis4_counts = 200_000.0 + np.linspace(0.0, 9_000.0, rows)
    axis1_torque = np.full(rows, -34.5872)
    axis4_torque = np.full(rows, 2.6403)
    sync = "atomic_task_buffer" if atomic else "not_atomic_axes_read_sequentially"
    lines = ["host_time_s,sync_quality,axis1_position_motor_deg,axis4_position_motor_deg,"
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


def _calibration() -> dict[str, JointCalibration]:
    # Counts-per-degree and zero are illustrative only, never presented as the
    # real mechanism calibration.
    return {
        "hip": JointCalibration(counts_per_degree=1000.0, zero_counts=100_000.0,
                                flexion_sign=-1, safe_min_deg=-20.0, safe_max_deg=90.0),
        "knee": JointCalibration(counts_per_degree=1000.0, zero_counts=200_000.0,
                                 flexion_sign=+1, safe_min_deg=0.0, safe_max_deg=120.0),
    }


def test_axis_binding_does_not_cross_between_axis_one_and_four(tmp_path: Path):
    """A bare ``1`` inside ``3c1a`` must not steal axis 4's torque column."""

    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 12, 2.6))
    assert episode.joints["hip"].signals["torque_nm"] == "axis1_torque_3c1a"
    assert episode.joints["knee"].signals["torque_nm"] == "axis4_torque_3c1a"
    assert episode.joints["hip"].signals["position"] == "axis1_position_motor_deg"
    assert episode.joints["knee"].signals["position"] == "axis4_position_motor_deg"


def test_underscore_alias_names_still_resolve(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 5, 2.6))
    assert np.allclose(episode.host_time_s, np.arange(5) / 2.6)
    assert episode.sync_quality == "not_atomic_axes_read_sequentially"


def test_joint_angle_needs_a_real_calibration(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 5, 2.6))
    assert episode.joints["hip"].flexion_deg is None
    # Motor-side degrees are recorded but never promoted to a joint angle.
    assert episode.joints["hip"].motor_position_deg is not None


def test_calibration_converts_counts_to_signed_flexion(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 3, 2.6),
                               calibration=_calibration())
    hip = episode.joints["hip"].flexion_deg
    knee = episode.joints["knee"].flexion_deg
    assert hip[0] == pytest.approx(0.0)
    assert hip[-1] > 0.0          # hip counts fall during flexion, sign -1
    assert knee[0] == pytest.approx(0.0)
    assert knee[-1] > 0.0         # knee counts rise during flexion, sign +1


def test_missing_calibration_entry_is_rejected(tmp_path: Path):
    calibration = _calibration()
    del calibration["knee"]
    with pytest.raises(CalibrationError):
        read_leg_episode(_write_episode(tmp_path / "ep.csv", 3, 2.6), calibration=calibration)


def test_low_rate_and_non_atomic_reads_are_reported(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 27, 2.6))
    report = quality_report(episode, minimum_rate_hz=100.0)
    assert report["sample_rate_hz"] == pytest.approx(2.6, rel=1e-6)
    assert report["usable_for_dynamics"] is False
    assert report["non_atomic_axis_reads"] is True
    assert any(finding.startswith("sample_rate_below_100hz") for finding in report["findings"])
    assert "axis_channels_not_a_single_instant" in report["findings"]


def test_high_rate_atomic_episode_is_usable(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 101, 200.0, atomic=True))
    report = quality_report(episode, minimum_rate_hz=100.0)
    assert report["usable_for_dynamics"] is True
    assert report["non_atomic_axis_reads"] is False


def test_static_offset_is_descriptive_and_never_tared(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 27, 2.6))
    report = quality_report(episode)
    hip = report["joints"]["hip"]["raw_torque_nm"]
    assert hip["mean"] == pytest.approx(-34.5872, abs=1e-6)
    assert "never subtracted by default" in report["joints"]["hip"]["torque_offset_note"]


def test_time_column_absent_is_rejected(tmp_path: Path):
    path = tmp_path / "no_time.csv"
    path.write_text("axis1_position_motor_deg,axis1_torque_3c1a\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(LegImportError):
        read_leg_episode(path)


def test_column_map_override_is_honoured(tmp_path: Path):
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
    episode = read_leg_episode(path, column_map=column_map)
    assert episode.joints["hip"].signals["position"] == "hip_counts"
    assert episode.joints["knee"].signals["torque_nm"] == "knee_torque"


def test_motor_side_scale_matches_handover_record(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 3, 2.6))
    counts = episode.joints["hip"].position_counts
    assert np.allclose(episode.joints["hip"].motor_position_deg,
                       counts * MOTOR_SIDE_DEG_PER_COUNT)


def test_json_report_is_serialisable(tmp_path: Path):
    episode = read_leg_episode(_write_episode(tmp_path / "ep.csv", 12, 2.6))
    json.dumps(quality_report(episode), ensure_ascii=False)


def test_cli_writes_a_report_without_touching_hardware(tmp_path: Path, capsys):
    csv_path = _write_episode(tmp_path / "ep.csv", 12, 2.6)
    output = tmp_path / "report.json"
    assert import_main([str(csv_path), "--minimum-rate-hz", "100", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["file"] == "ep.csv"
    assert report["usable_for_dynamics"] is False


def test_cli_promotes_joint_angles_only_with_calibration(tmp_path: Path):
    csv_path = _write_episode(tmp_path / "ep.csv", 3, 2.6)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "hip": {"counts_per_degree": 1000.0, "zero_counts": 100000.0,
                "flexion_sign": -1, "safe_min_deg": -20.0, "safe_max_deg": 90.0},
        "knee": {"counts_per_degree": 1000.0, "zero_counts": 200000.0,
                 "flexion_sign": 1, "safe_min_deg": 0.0, "safe_max_deg": 120.0},
    }), encoding="utf-8")
    output = tmp_path / "report.json"
    assert import_main([str(csv_path), "--calibration", str(calibration_path),
                        "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["joints"]["hip"]["joint_angle_available"] is True
