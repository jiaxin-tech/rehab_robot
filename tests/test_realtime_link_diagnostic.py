from __future__ import annotations

import json

import numpy as np
import pytest

from control.execution_preflight import OPERATOR_CONFIRMATION
from control.realtime_link_diagnostic import (
    FAR_CHECK_S,
    REALTIME_LINK_PROFILE,
    RETURN_CHECK_S,
    SETTLE_END_S,
    RealtimeLinkTrackingGuard,
    audit_realtime_link_trajectory,
    build_realtime_link_trajectory,
    evaluate_realtime_link_offline,
)
from control.start_anchor import FixedTcpOrientation, StartAnchor, save_start_anchor
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.test_realtime_motion import prepare, run


START_POSE = (0.7, 0.0, 0.4, 0.1, -0.2, 0.3)


def _safety() -> ExperimentSafetyConfig:
    return ExperimentSafetyConfig(
        max_tcp_speed_m_s=0.03,
        max_tcp_acceleration_m_s2=0.01,
        max_start_anchor_position_error_m=0.005,
        max_start_anchor_orientation_error_rad=0.05,
        max_command_lateness_s=0.1,
        max_force_n=100.0,
        max_torque_nm=200.0,
        max_state_age_s=0.1,
        max_wrench_age_s=0.05,
        max_state_wrench_skew_s=0.04,
        workspace_min_base_m=(0.0, -0.15, 0.1),
        workspace_max_base_m=(1.0, 1.0, 1.2),
        rt_filter_hz=50.0,
        rt_network_tolerance_percent=20,
    )


def _anchor() -> StartAnchor:
    return StartAnchor(
        capture_host_time_s=1.0,
        tcp_pose_base=START_POSE,
        tcp_position_base_m=START_POSE[:3],
        tcp_orientation=FixedTcpOrientation(values_rad=START_POSE[3:]),
        robot_joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        trajectory_id="measured_asymmetric_closed_v1",
        reference_start_q_hip=0.5,
        reference_start_q_knee=0.6,
        anchor_id="rt-link-test-anchor",
    )


def test_builder_produces_only_closed_one_millimetre_positive_z_path():
    trajectory = build_realtime_link_trajectory(START_POSE, step_m=0.001)
    xyz = trajectory[["tcp_x_base", "tcp_y_base", "tcp_z_base"]].to_numpy(float)
    relative = xyz - xyz[0]

    assert trajectory["trajectory_profile"].unique().tolist() == [REALTIME_LINK_PROFILE]
    assert np.allclose(relative[:, :2], 0.0)
    assert relative[:, 2].min() == pytest.approx(0.0)
    assert relative[:, 2].max() == pytest.approx(0.001)
    assert np.allclose(relative[-1], 0.0)
    reasons, speed, acceleration = audit_realtime_link_trajectory(
        trajectory, _safety()
    )
    assert reasons == ()
    assert speed < 0.002
    assert acceleration < 0.005


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    (
        (
            lambda table: table.__setitem__(
                "tcp_x_base", table["tcp_x_base"] + table["delta_z_R"]
            ),
            "diagnostic_lateral_motion_forbidden",
        ),
        (
            lambda table: table.__setitem__(
                "tcp_rx",
                table["tcp_rx"] + np.where(table.index == 10, 0.01, 0.0),
            ),
            "diagnostic_orientation_change_forbidden",
        ),
        (
            lambda table: table.__setitem__("trajectory_profile", "arbitrary"),
            "diagnostic_trajectory_profile_invalid",
        ),
    ),
)
def test_auditor_rejects_mutated_paths(mutate, expected_reason):
    trajectory = build_realtime_link_trajectory(START_POSE, step_m=0.001)
    mutate(trajectory)
    reasons, _, _ = audit_realtime_link_trajectory(trajectory, _safety())
    assert expected_reason in reasons


class _PoseAdapter:
    def __init__(self):
        self.pose = list(START_POSE)

    def read_tcp_pose(self):
        return tuple(self.pose)


def test_tracking_guard_rejects_dispatched_commands_without_actual_motion():
    adapter = _PoseAdapter()
    guard = RealtimeLinkTrackingGuard(adapter, START_POSE, 0.001)
    guard.observe_dispatch(SETTLE_END_S, 1, 2)
    assert guard.check() is None
    guard.observe_dispatch(FAR_CHECK_S, 1, 2)

    assert guard.check().startswith("diagnostic_no_observed_motion:")


def test_tracking_guard_requires_observed_move_and_return():
    adapter = _PoseAdapter()
    guard = RealtimeLinkTrackingGuard(adapter, START_POSE, 0.001)
    guard.observe_dispatch(SETTLE_END_S, 1, 2)
    assert guard.check() is None
    adapter.pose[2] += 0.0009
    guard.observe_dispatch(FAR_CHECK_S, 1, 2)
    assert guard.check() is None
    adapter.pose[2] = START_POSE[2]
    guard.observe_dispatch(RETURN_CHECK_S, 3, 4)
    assert guard.check() is None
    guard.require_complete()
    summary = guard.as_metadata()
    assert summary["far_verified"] is True
    assert summary["return_verified"] is True


def test_tracking_guard_does_not_accept_sideways_motion_as_positive_z_motion():
    adapter = _PoseAdapter()
    guard = RealtimeLinkTrackingGuard(adapter, START_POSE, 0.001)
    guard.observe_dispatch(SETTLE_END_S, 1, 2)
    assert guard.check() is None
    adapter.pose[0] += 0.001
    guard.observe_dispatch(FAR_CHECK_S, 1, 2)

    assert guard.check().startswith("diagnostic_no_observed_motion:")


def test_tracking_guard_measures_motion_from_post_settle_pose():
    adapter = _PoseAdapter()
    guard = RealtimeLinkTrackingGuard(adapter, START_POSE, 0.001)
    adapter.pose[0] -= 0.0002
    adapter.pose[2] -= 0.0012
    guard.observe_dispatch(SETTLE_END_S, 1, 2)
    assert guard.check() is None
    adapter.pose[2] += 0.001
    guard.observe_dispatch(FAR_CHECK_S, 3, 4)
    assert guard.check() is None
    adapter.pose[2] -= 0.001
    guard.observe_dispatch(RETURN_CHECK_S, 5, 6)
    assert guard.check() is None
    guard.require_complete()
    summary = guard.as_metadata()
    assert summary["settle_displacement_m"] == pytest.approx(
        np.hypot(0.0002, 0.0012)
    )


def test_offline_gate_binds_first_target_to_captured_anchor():
    anchor = _anchor()
    trajectory = build_realtime_link_trajectory(START_POSE, step_m=0.001)
    trajectory["tcp_x_base"] += 0.001
    preflight = evaluate_realtime_link_offline(
        enable_motion=True,
        confirm_unloaded=True,
        operator_confirmation=OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id,
        anchor=anchor,
        safety=_safety(),
        trajectory=trajectory,
    )

    assert preflight.allowed is False
    assert "diagnostic trajectory is not bound to the prepared anchor" in preflight.reasons


def test_prepare_is_observation_only_and_run_without_flags_never_constructs_adapter(
    tmp_path,
):
    session = tmp_path / "rt_link_001"

    def fake_capture(**kwargs):
        save_start_anchor(_anchor(), kwargs["output_path"])
        return {"saved": True, "motion_commanded": False}

    prepared = prepare(session, step_mm=1.0, capture_fn=fake_capture)
    assert prepared["status"] == "PREPARED"
    assert prepared["motion_commanded"] is False
    constructed = []

    def forbidden_adapter(_ip):
        constructed.append(True)
        raise AssertionError("adapter must not be constructed")

    with pytest.raises(PermissionError, match="--live is required"):
        run(
            session,
            live=False,
            confirm_unloaded=True,
            operator_confirmation=OPERATOR_CONFIRMATION,
            adapter_factory=forbidden_adapter,
        )
    assert constructed == []
    status = json.loads((session / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PREPARED"

