"""The unified runner must fail offline before any forbidden connection."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from config import settings
from control.execution_preflight import OPERATOR_CONFIRMATION
from control.start_anchor import FixedTcpOrientation, StartAnchor
from control.start_anchored_relative_trajectory import (
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    build_start_anchored_relative_trajectory,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.run_rehab_experiment import run_execute


SOURCE_CANDIDATE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "lower_limb_sim"
    / "data"
    / "reference_candidates"
)


def _reviewed_files(
    tmp_path, *, trajectory_id=FIRST_ROBOT_TRIAL_TRAJECTORY_ID
):
    pose = (0.4, 0.0, 0.3, 0.1, 0.2, 0.3)
    reference = (
        DEFAULT_REFERENCE_PATH
        if trajectory_id.endswith("slow")
        else SOURCE_CANDIDATE_DIRECTORY
        / "reference_measured_asymmetric_closed_nominal.csv"
    )
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        reference,
        current_tcp_start_pose=pose,
        rehab_frame=__import__(
            "control.start_anchored_relative_trajectory", fromlist=["RehabFrameConfig"]
        ).RehabFrameConfig((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), True),
    )
    anchor = StartAnchor(
        capture_host_time_s=1.0,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=FixedTcpOrientation(pose[3:]),
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
        json.dumps(
            {
                "rehab_x_axis_in_base": [1.0, 0.0, 0.0],
                "rehab_z_axis_in_base": [0.0, 0.0, 1.0],
                "reviewed": True,
                "notes": "synthetic test",
            }
        ),
        encoding="utf-8",
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


def _call(tmp_path, factory, **overrides):
    trajectory_id = overrides.pop(
        "trajectory_id", FIRST_ROBOT_TRIAL_TRAJECTORY_ID
    )
    anchor, frame, safety = _reviewed_files(tmp_path, trajectory_id=trajectory_id)
    values = dict(
        robot_ip="192.0.2.1",
        episode_dir=tmp_path / "episode",
        anchor_path=anchor,
        requested_anchor_id="anchor_runner_test",
        frame_config_path=frame,
        safety_config_path=safety,
        trajectory_id=trajectory_id,
        enable_motion=True,
        operator_confirmation=OPERATOR_CONFIRMATION,
        adapter_factory=factory,
    )
    values.update(overrides)
    return run_execute(**values)


def test_missing_enable_motion_fails_before_adapter_construction(tmp_path):
    calls = []
    with pytest.raises(PermissionError, match="enable_motion_flag_missing"):
        _call(
            tmp_path,
            lambda ip: calls.append(ip),
            enable_motion=False,
        )
    assert calls == []
    assert not (tmp_path / "episode").exists()


def test_wrong_operator_confirmation_fails_before_connection(tmp_path):
    calls = []
    with pytest.raises(PermissionError, match="operator_confirmation_missing"):
        _call(tmp_path, lambda ip: calls.append(ip), operator_confirmation="yes")
    assert calls == []


def test_nominal_is_refused_before_adapter_construction(tmp_path):
    calls = []
    with pytest.raises(PermissionError, match="not_whitelisted"):
        _call(
            tmp_path,
            lambda ip: calls.append(ip),
            trajectory_id=MEASURED_ASYMMETRIC_NOMINAL_ID,
        )
    assert calls == []


def test_real_default_factory_requires_explicit_local_interface_before_connection(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(settings, "ROBOT_LOCAL_IP", "")
    with pytest.raises(
        PermissionError,
        match="reference_release_not_approved_for_first_robot_trial",
    ):
        _call(tmp_path, None)
    assert not (tmp_path / "episode").exists()
