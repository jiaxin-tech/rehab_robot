"""Read-only start-anchor capture and fail-closed safety configuration tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from control.start_anchor import (
    FixedTcpOrientation,
    StartAnchor,
    capture_start_anchor,
    load_start_anchor,
    save_start_anchor,
)
from safety.experiment_safety import (
    ExperimentSafetyConfig,
    load_experiment_safety_config,
    require_execute_safety,
)


POSE = (0.41, -0.12, 0.36, 0.1, -0.2, 0.3)
JOINTS = (0.0, 0.1, -0.2, 0.3, -0.4, 0.5)
SOFT_LIMITS = tuple((-2.0, 2.0) for _ in range(6))


class ReadOnlyFrameAdapter:
    def __init__(self) -> None:
        self.read_count = 0
        self.metadata_read_count = 0
        self.motion_calls: list[str] = []

    def read_state_frame(self):
        self.read_count += 1
        return SimpleNamespace(
            valid=True,
            invalid_reason="",
            tcp_position_m=POSE[:3],
            tcp_orientation_rad=POSE[3:],
            joint_position_rad=JOINTS,
        )

    def read_robot_metadata(self):
        self.metadata_read_count += 1
        return {
            "robot_model": "xMate6-Pro",
            "robot_serial_number": "SN-TEST-001",
            "controller_version": "4.2-test",
        }

    def _forbidden(self, name: str):
        self.motion_calls.append(name)
        raise AssertionError(f"capture called forbidden adapter method: {name}")

    def connect(self):
        self._forbidden("connect")

    def enable(self):
        self._forbidden("enable")

    def enable_realtime(self):
        self._forbidden("enable_realtime")

    def move_l(self, _target=None):
        self._forbidden("move_l")

    def move_j(self, _target=None):
        self._forbidden("move_j")

    def stop(self):
        self._forbidden("stop")


def _capture(adapter=None) -> StartAnchor:
    return capture_start_anchor(
        adapter or ReadOnlyFrameAdapter(),
        trajectory_id="reference_measured_asymmetric_closed_slow",
        reference_start_q_hip=0.25,
        reference_start_q_knee=0.35,
        anchor_id="anchor_test_001",
        tool_name="rehab_cuff",
        workpiece_name="rehab_fixture",
        notes="operator will review after capture",
        clock=lambda: 1234.5,
    )


def _complete_safety(*, reviewed: bool = True) -> ExperimentSafetyConfig:
    return ExperimentSafetyConfig(
        max_tcp_speed_m_s=0.1,
        max_tcp_acceleration_m_s2=0.2,
        max_start_anchor_position_error_m=0.01,
        max_start_anchor_orientation_error_rad=0.01,
        max_command_lateness_s=0.1,
        max_force_n=10.0,
        max_torque_nm=2.0,
        max_state_age_s=0.05,
        max_wrench_age_s=0.05,
        max_state_wrench_skew_s=0.02,
        workspace_min_base_m=(0.1, -0.3, 0.1),
        workspace_max_base_m=(0.7, 0.3, 0.8),
        expected_robot_model="xMate6-Pro",
        expected_robot_serial_number="SN-TEST-001",
        expected_controller_version="4.2-test",
        reviewed_tool_name="rehab_cuff",
        reviewed_workpiece_name="rehab_fixture",
        reviewed_payload_mass_kg=1.25,
        reviewed_payload_cog_m=(0.0, 0.0, 0.08),
        reviewed_payload_inertia_kg_m2=(0.01, 0.02, 0.03),
        reviewed_joint_soft_limits_rad=SOFT_LIMITS,
        reviewed_rt_filter_hz=25.0,
        reviewed_rt_network_tolerance_percent=20.0,
        robot_identity_reviewed=True,
        tool_workpiece_reviewed=True,
        payload_configuration_reviewed=True,
        collision_configuration_reviewed=True,
        joint_soft_limits_reviewed=True,
        realtime_configuration_reviewed=True,
        reviewed=reviewed,
        notes="synthetic unit-test values only",
    )


def test_capture_reads_one_cached_state_and_never_calls_lifecycle_or_motion() -> None:
    adapter = ReadOnlyFrameAdapter()
    anchor = _capture(adapter)
    assert adapter.read_count == 1
    assert adapter.metadata_read_count == 1
    assert adapter.motion_calls == []
    assert anchor.capture_host_time_s == 1234.5
    assert anchor.tcp_pose_base == POSE
    assert anchor.tcp_position_base_m == POSE[:3]
    assert anchor.tcp_orientation.strategy == "fixed"
    assert anchor.tcp_orientation.values_rad == POSE[3:]
    assert anchor.robot_joint_positions == JOINTS
    assert anchor.trajectory_id == "reference_measured_asymmetric_closed_slow"
    assert anchor.robot_model == "xMate6-Pro"
    assert anchor.robot_serial_number == "SN-TEST-001"
    assert anchor.controller_version == "4.2-test"
    assert anchor.tool_name == "rehab_cuff"
    assert anchor.workpiece_name == "rehab_fixture"
    assert anchor.reviewed is False


def test_capture_supports_explicit_read_adapter_without_motion_methods() -> None:
    class ExplicitReadAdapter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def read_tcp_pose(self):
            self.calls.append("read_tcp_pose")
            return POSE

        def read_joint_positions(self):
            self.calls.append("read_joint_positions")
            return JOINTS

    adapter = ExplicitReadAdapter()
    anchor = _capture(adapter)
    assert adapter.calls == ["read_tcp_pose", "read_joint_positions"]
    assert anchor.tcp_pose_base == POSE


def test_anchor_json_round_trip_is_atomic_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "subject" / "anchor.json"
    anchor = _capture()
    assert save_start_anchor(anchor, path) == path
    assert load_start_anchor(path) == anchor
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reviewed"] is False
    assert payload["tcp_orientation"]["strategy"] == "fixed"

    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields must be exactly"):
        StartAnchor.load_json(path)


def test_anchor_loader_rejects_truthy_review_and_inconsistent_pose(tmp_path: Path) -> None:
    payload = _capture().to_dict()
    path = tmp_path / "anchor.json"
    payload["reviewed"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed must be"):
        StartAnchor.load_json(path)

    payload["reviewed"] = False
    payload["tcp_position_base_m"] = [9.0, 8.0, 7.0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="position does not match"):
        StartAnchor.load_json(path)


def test_invalid_cached_state_is_rejected_without_fallback_motion() -> None:
    adapter = ReadOnlyFrameAdapter()
    adapter.read_state_frame = lambda: SimpleNamespace(
        valid=False,
        invalid_reason="state_stale",
        tcp_position_m=POSE[:3],
        tcp_orientation_rad=POSE[3:],
        joint_position_rad=JOINTS,
    )
    with pytest.raises(RuntimeError, match="state_stale"):
        _capture(adapter)
    assert adapter.motion_calls == []


def test_repository_default_safety_config_is_all_null_and_fail_closed() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "experiment_safety.json"
    config = load_experiment_safety_config(path)
    assert config.reviewed is False
    assert config.max_tcp_speed_m_s is None
    assert config.max_tcp_acceleration_m_s2 is None
    assert config.max_start_anchor_position_error_m is None
    assert config.max_start_anchor_orientation_error_rad is None
    assert config.max_command_lateness_s is None
    assert config.max_force_n is None
    assert config.max_torque_nm is None
    assert config.max_state_age_s is None
    assert config.max_wrench_age_s is None
    assert config.max_state_wrench_skew_s is None
    assert config.workspace_min_base_m is None
    assert config.workspace_max_base_m is None
    assert config.expected_robot_model is None
    assert config.expected_robot_serial_number is None
    assert config.expected_controller_version is None
    assert config.reviewed_tool_name is None
    assert config.reviewed_workpiece_name is None
    assert config.reviewed_payload_mass_kg is None
    assert config.reviewed_payload_cog_m is None
    assert config.reviewed_payload_inertia_kg_m2 is None
    assert config.reviewed_joint_soft_limits_rad is None
    assert config.reviewed_rt_filter_hz is None
    assert config.reviewed_rt_network_tolerance_percent is None
    assert config.robot_identity_reviewed is False
    assert config.tool_workpiece_reviewed is False
    assert config.payload_configuration_reviewed is False
    assert config.collision_configuration_reviewed is False
    assert config.joint_soft_limits_reviewed is False
    assert config.realtime_configuration_reviewed is False
    allowed, reasons = config.validate_for_execution()
    assert allowed is False
    assert "experiment_safety_not_reviewed" in reasons
    assert "workspace_bounds_not_configured" in reasons
    with pytest.raises(PermissionError, match="real robot execution blocked"):
        require_execute_safety(config)


def test_reviewed_flag_alone_never_bypasses_missing_limits() -> None:
    config = ExperimentSafetyConfig(reviewed=True, notes="reviewed but incomplete")
    assert config.execution_allowed is False
    assert "max_force_n_not_configured" in config.execution_block_reasons()
    with pytest.raises(PermissionError):
        config.require_execute_allowed()


def test_complete_configuration_still_requires_boolean_review() -> None:
    draft = _complete_safety(reviewed=False)
    assert not draft.execution_allowed
    assert draft.execution_block_reasons() == ("experiment_safety_not_reviewed",)
    reviewed = replace(draft, reviewed=True)
    assert reviewed.validate_for_execution() == (True, ())
    reviewed.require_execute_allowed()


def test_safety_json_round_trip_and_workspace_validation(tmp_path: Path) -> None:
    config = _complete_safety()
    path = tmp_path / "experiment_safety.json"
    config.save_json(path)
    assert ExperimentSafetyConfig.load_json(path) == config
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))

    with pytest.raises(ValueError, match="both be null or both be configured"):
        ExperimentSafetyConfig(workspace_min_base_m=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="strictly below"):
        ExperimentSafetyConfig(
            workspace_min_base_m=(0.0, 0.0, 0.0),
            workspace_max_base_m=(0.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="finite positive"):
        ExperimentSafetyConfig(max_force_n=0.0)
    with pytest.raises(ValueError, match=r"six \[lower, upper\] pairs"):
        ExperimentSafetyConfig(reviewed_joint_soft_limits_rad=((-1.0, 1.0),))
    with pytest.raises(ValueError, match="lower must be below upper"):
        ExperimentSafetyConfig(
            reviewed_joint_soft_limits_rad=tuple((1.0, -1.0) for _ in range(6))
        )
    with pytest.raises(ValueError, match="non-negative"):
        ExperimentSafetyConfig(reviewed_payload_inertia_kg_m2=(-0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        ExperimentSafetyConfig(reviewed_rt_filter_hz=1001.0)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        ExperimentSafetyConfig(reviewed_rt_network_tolerance_percent=101.0)


def test_orientation_requires_fixed_euler_representation() -> None:
    with pytest.raises(ValueError, match="strategy"):
        FixedTcpOrientation(values_rad=POSE[3:], strategy="active")
    with pytest.raises(ValueError, match="representation"):
        FixedTcpOrientation(values_rad=POSE[3:], representation="rotation_vector")
