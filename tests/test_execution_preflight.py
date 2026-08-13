"""Fail-closed first-trial execution gate tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from collection.real_robot_acquisition import AcquisitionHealth
import control.execution_preflight as preflight_module
from control.execution_preflight import (
    OPERATOR_CONFIRMATION,
    evaluate_execution_preflight,
    evaluate_offline_execution_request,
)
from control.start_anchor import FixedTcpOrientation, StartAnchor
from control.start_anchored_relative_trajectory import (
    RehabFrameConfig,
    build_start_anchored_relative_trajectory,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from safety.experiment_safety import ExperimentSafetyConfig


SOURCE_CANDIDATE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "lower_limb_sim"
    / "data"
    / "reference_candidates"
)


class ReadyLogger:
    ready = True
    healthy = True


class ConnectedAdapter:
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


def _inputs():
    frame = RehabFrameConfig((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), True, "reviewed test")
    pose = (0.4, 0.0, 0.3, 0.1, 0.2, 0.3)
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        current_tcp_start_pose=pose,
        rehab_frame=frame,
    )
    first = trajectory.iloc[0]
    anchor = StartAnchor(
        capture_host_time_s=1.0,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=FixedTcpOrientation(pose[3:]),
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
    safety = ExperimentSafetyConfig(
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
    health = AcquisitionHealth(
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


def _evaluate(**overrides):
    frame, trajectory, audit, anchor, safety, health = _inputs()
    values = dict(
        mode="execute",
        enable_motion=True,
        operator_confirmation=OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id,
        frame=frame,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
        audit=audit,
        acquisition_health=health,
        logger=ReadyLogger(),
        robot_adapter=ConnectedAdapter(),
        current_tcp_pose_base=anchor.tcp_pose_base,
    )
    values.update(overrides)
    return evaluate_execution_preflight(**values)


def test_reference_freeze_keeps_even_reviewed_slow_request_no_go():
    result = _evaluate()
    assert not result.allowed
    assert result.reasons == (
        "reference_release_not_approved_for_first_robot_trial",
    )
    assert result.evaluation_phase == "live"
    assert result.trajectory_sha256


def test_execute_requires_explicit_enable_and_operator_confirmation():
    assert "enable_motion_flag_missing" in _evaluate(enable_motion=False).reasons
    assert "operator_confirmation_missing" in _evaluate(operator_confirmation="yes").reasons


def test_unreviewed_frame_anchor_and_safety_are_each_blocking():
    frame, _, _, anchor, safety, _ = _inputs()
    assert "rehab_frame_not_reviewed" in _evaluate(frame=replace(frame, reviewed=False)).reasons
    assert "start_anchor_not_reviewed" in _evaluate(anchor=replace(anchor, reviewed=False)).reasons
    result = _evaluate(safety=replace(safety, reviewed=False))
    assert "experiment_safety_not_reviewed" in result.reasons


def test_nominal_reference_is_not_first_trial_whitelisted():
    frame, _, _, anchor, safety, health = _inputs()
    nominal = SOURCE_CANDIDATE_DIRECTORY / "reference_measured_asymmetric_closed_nominal.csv"
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        nominal,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    result = _evaluate(
        trajectory=trajectory,
        audit=audit,
        anchor=replace(anchor, trajectory_id=audit.trajectory_id),
        safety=safety,
        acquisition_health=health,
    )
    assert "trajectory_not_whitelisted_for_first_robot_trial" in result.reasons
    assert "trajectory_audit_invalid" in result.reasons
    assert "trajectory_rows_invalid" in result.reasons


def test_stale_or_dead_stream_is_blocking():
    *_, health = _inputs()
    result = _evaluate(
        acquisition_health=replace(
            health,
            state_age_s=2.0,
            state_thread_alive=False,
            valid=False,
        )
    )
    assert "state_stale" in result.reasons
    assert "state_thread_not_alive" in result.reasons


def test_robot_must_already_be_at_reviewed_anchor():
    *_, anchor, _, _ = _inputs()
    displaced = list(anchor.tcp_pose_base)
    displaced[0] += 0.1
    result = _evaluate(current_tcp_pose_base=displaced)
    assert "robot_not_at_reviewed_start_anchor_position" in result.reasons


def test_anchor_reference_start_joint_values_are_bound_to_active_first_row():
    *_, anchor, _, _ = _inputs()
    result = _evaluate(
        anchor=replace(anchor, reference_start_q_hip=anchor.reference_start_q_hip + 0.01)
    )
    assert "anchor_reference_start_joint_mismatch" in result.reasons


def test_runtime_robot_identity_must_match_anchor_and_reviewed_config():
    class WrongSerial(ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["robot_metadata"]["robot_serial_number"] = "SN-WRONG"
            return summary

    result = _evaluate(robot_adapter=WrongSerial())
    assert "runtime_robot_serial_number_mismatch" in result.reasons


def test_declared_tool_and_workpiece_must_match_anchor_and_sdk_lists():
    *_, anchor, _, _ = _inputs()
    result = _evaluate(anchor=replace(anchor, tool_name="other_tool"))
    assert "anchor_tool_name_mismatch" in result.reasons

    class MissingWorkpiece(ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["robot_metadata"]["sdk_tool_payload"][
                "sdk_available_workobject_names"
            ] = []
            return summary

    result = _evaluate(robot_adapter=MissingWorkpiece())
    assert "reviewed_workpiece_name_not_reported_by_sdk" in result.reasons


def test_payload_collision_and_soft_limits_fail_closed():
    class UnsafeSummary(ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["collision_state"] = True
            summary["robot_metadata"]["sdk_tool_payload"][
                "toolset_load_mass_kg"
            ] = 2.0
            summary["robot_metadata"]["joint_soft_limits_rad"][0] = [-1.0, 1.0]
            return summary

    reasons = _evaluate(robot_adapter=UnsafeSummary()).reasons
    assert "runtime_collision_detected" in reasons
    assert "runtime_payload_mass_mismatch" in reasons
    assert "runtime_joint_soft_limits_mismatch" in reasons


def test_unavailable_collision_query_and_current_joint_outside_limits_block():
    class MissingCollision(ConnectedAdapter):
        def get_robot_state_summary(self):
            summary = super().get_robot_state_summary()
            summary["collision_state"] = None
            summary["collision_state_query_valid"] = False
            summary["joint_position_rad"][2] = 3.0
            return summary

    reasons = _evaluate(robot_adapter=MissingCollision()).reasons
    assert "runtime_collision_state_unavailable" in reasons
    assert "runtime_joint_position_outside_soft_limits" in reasons


def test_row_validity_requires_real_booleans_and_motion_derivatives_are_recomputed():
    _, trajectory, _, _, _, _ = _inputs()
    string_flags = trajectory.copy(deep=True)
    string_flags["trajectory_valid"] = "True"
    reasons = _evaluate(trajectory=string_flags).reasons
    assert "trajectory_valid_column_not_strict_boolean" in reasons

    falsified_derivatives = trajectory.copy(deep=True)
    falsified_derivatives[
        ["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]
    ] = 0.0
    falsified_derivatives[
        ["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]
    ] = 0.0
    reasons = _evaluate(trajectory=falsified_derivatives).reasons
    assert "trajectory_velocity_columns_mismatch" in reasons
    assert "trajectory_acceleration_columns_mismatch" in reasons


def test_offline_preflight_is_explicitly_unbound_from_executor():
    frame, trajectory, audit, anchor, safety, _ = _inputs()
    result = evaluate_offline_execution_request(
        mode="execute",
        enable_motion=True,
        operator_confirmation=OPERATOR_CONFIRMATION,
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
    with pytest.raises(PermissionError, match="not_live_bound"):
        result.require_live_trajectory_binding(trajectory)


def test_first_trial_reference_hash_is_pinned_not_only_compared_to_same_file(
    monkeypatch,
):
    original_loader = preflight_module.load_closed_reference_trajectory

    def wrong_hash_loader(path):
        frame, metadata = original_loader(path)
        return frame, {**metadata, "sha256": "0" * 64}

    monkeypatch.setattr(
        preflight_module,
        "load_closed_reference_trajectory",
        wrong_hash_loader,
    )
    assert "trajectory_not_exact_official_slow_reference" in _evaluate().reasons
