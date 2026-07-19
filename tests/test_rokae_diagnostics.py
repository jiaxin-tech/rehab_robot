"""Cross-platform tests for read-only xCore diagnostic tooling."""

from __future__ import annotations

import time

from collection.snapshot import read_live_robot_state_sample
from collection.state import InternalWrenchFrame, KinematicStateFrame
from scripts.check_rt_state_timing import measure as measure_rt
from scripts.check_snapshot_alignment import measure as measure_alignment
from scripts.check_wrench_query_timing import measure as measure_query
from scripts.rokae_diagnostic_common import (
    invalid_reason_counts,
    pose_dependence_analysis,
    rotation_push_analysis,
    sequence_drops,
    write_report,
)


def _state(sequence_id: int) -> KinematicStateFrame:
    now = time.monotonic()
    return KinematicStateFrame(
        sequence_id=sequence_id, host_monotonic_time_s=now, wall_time_iso="now",
        robot_device_time_s=None, valid=True, invalid_reason="", tcp_position_m=(0.0, 0.0, 0.0),
        tcp_orientation_rad=(0.0, 0.0, 0.0), tcp_linear_velocity_mps=(0.0, 0.0, 0.0),
        tcp_angular_velocity_radps=(0.0, 0.0, 0.0), velocity_source="fake", joint_position_rad=(0.0,) * 6,
        joint_velocity_radps=(0.0,) * 6, pose_time_s=now, joint_time_s=now, velocity_time_s=now,
        operation_state="IDLE", collision_state=False, controller_error=None,
    )


def _wrench(sequence_id: int) -> InternalWrenchFrame:
    now = time.monotonic()
    return InternalWrenchFrame(
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


class FakeAdapter:
    def __init__(self) -> None:
        self.sequence = 0

    def get_state_frame(self) -> KinematicStateFrame:
        self.sequence += 2  # deliberately demonstrates detectable source gaps
        return _state(self.sequence)

    def get_end_wrench(self, frame: str) -> dict[str, object]:
        now = time.monotonic()
        return {"raw_force_frame": frame, "force_query_started_s": now - 0.0001, "force_query_finished_s": now}


class FakeSource:
    def __init__(self) -> None:
        self.sequence = 0

    def snapshot(self, now_s: float | None = None) -> InternalWrenchFrame:
        self.sequence += 1
        return _wrench(self.sequence)


def test_fake_adapter_timing_diagnostics_collect_rows() -> None:
    robot = FakeAdapter()
    rows, summary = measure_rt(robot, duration_s=0.004, poll_hz=2_000.0)
    assert rows and summary["dropped_frames"] >= 1
    query_rows, query_summary = measure_query(robot, duration_s=0.004, target_hz=2_000.0, reference_frame="world")
    assert query_rows and query_summary["query_duration_ms"]["count"] == len(query_rows)


def test_fake_adapter_snapshot_alignment_uses_project_snapshot() -> None:
    rows, summary = measure_alignment(FakeAdapter(), FakeSource(), duration_s=0.004, sample_hz=1_000.0)
    assert rows
    assert summary["force_query_duration_ms"]["count"] == len(rows)
    assert "base_wrench_rotation_requires_robot_validation" in summary["invalid_reasons"]


def test_rotation_analysis_reports_axis_sign_and_cross_axis() -> None:
    result = rotation_push_analysis([(0.0, 0.0, 0.0)], [(0.2, 4.0, 0.4)], "Y")
    assert result["principal_axis"] == "Y"
    assert result["principal_sign"] == "+"
    assert result["expected_axis_positive"] is True
    assert result["cross_axis_ratio"] is not None


def test_pose_dependence_is_not_gravity_compensation_claim() -> None:
    result = pose_dependence_analysis(
        [
            {"corrected_force_mean_n": (0.0, 0.0, 0.0), "corrected_torque_mean_nm": (0.0, 0.0, 0.0)},
            {"corrected_force_mean_n": (1.2, 0.0, 0.0), "corrected_torque_mean_nm": (0.0, 0.3, 0.0)},
        ], force_threshold_n=1.0, torque_threshold_nm=0.2,
    )
    assert result["software_bias_pose_dependence_observed"] is True
    assert "does not establish" in result["interpretation"]


def test_report_writes_csv_and_json(tmp_path) -> None:
    csv_path, json_path = write_report(tmp_path, "fake", [{"sample_index": 0, "invalid_reason": "a;b"}], {"ok": True})
    assert csv_path.exists() and json_path.exists()
    assert invalid_reason_counts([{"invalid_reason": "a;b"}]) == {"a": 1, "b": 1}
    assert sequence_drops([1, 4, 5]) == 2
