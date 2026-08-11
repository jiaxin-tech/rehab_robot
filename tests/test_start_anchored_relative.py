"""Regression tests for the frozen start-anchored experiment geometry."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from control.start_anchored_relative_trajectory import (
    ABSOLUTE_CALIBRATED_MODE,
    APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256,
    APPROVED_FIRST_ROBOT_TRIAL_L1_M,
    APPROVED_FIRST_ROBOT_TRIAL_L2_M,
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    RehabFrameConfig,
    START_ANCHORED_RELATIVE_MODE,
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from lower_limb_sim.reference_closed_c2 import (
    APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)


ANCHOR = np.asarray((0.41, -0.12, 0.36, 0.1, -0.2, 0.3), dtype=float)


def _identity_frame(*, reviewed: bool = False) -> RehabFrameConfig:
    return RehabFrameConfig(
        rehab_x_axis_in_base=(1.0, 0.0, 0.0),
        rehab_z_axis_in_base=(0.0, 0.0, 1.0),
        reviewed=reviewed,
        notes="offline test",
    )


def _build(frame: RehabFrameConfig | None = None):
    return build_start_anchored_relative_trajectory(
        current_tcp_start_pose=ANCHOR,
        rehab_frame=frame or _identity_frame(),
    )


def test_first_and_final_relative_samples_are_exactly_zero_and_anchor() -> None:
    trajectory, audit, _ = _build()
    relative = trajectory[["delta_x_R", "delta_y_R", "delta_z_R"]].to_numpy(float)
    target = trajectory[["tcp_x_base", "tcp_y_base", "tcp_z_base"]].to_numpy(float)
    assert np.array_equal(relative[0], np.zeros(3))
    assert np.array_equal(relative[-1], np.zeros(3))
    assert np.array_equal(target[0], ANCHOR[:3])
    assert np.array_equal(target[-1], ANCHOR[:3])
    assert audit.first_relative_displacement_zero
    assert audit.final_relative_displacement_zero
    assert audit.trajectory_valid


def test_relative_mode_has_no_hip_center_input_or_output_dependency() -> None:
    signature = inspect.signature(build_start_anchored_relative_trajectory)
    assert "hip_center_in_base_m" not in signature.parameters
    trajectory, _, metadata = _build()
    assert not any("hip_center" in name for name in trajectory.columns)
    assert metadata["hip_center_required"] is False
    assert metadata["observed_ankle_used_as_pull_point"] is False
    assert metadata["experiment_mode"] == START_ANCHORED_RELATIVE_MODE
    assert metadata["tool_offset_retained_for_mode"] == ABSOLUTE_CALIBRATED_MODE


def test_fixed_tcp_orientation_is_identical_to_anchor_for_every_sample() -> None:
    trajectory, audit, metadata = _build()
    orientation = trajectory[["tcp_rx", "tcp_ry", "tcp_rz"]].to_numpy(float)
    expected = np.repeat(ANCHOR[None, 3:], len(trajectory), axis=0)
    assert np.array_equal(orientation, expected)
    assert audit.orientation_constant
    assert metadata["tcp_orientation_strategy"] == "fixed_at_start_anchor"


def test_rehab_x_and_z_map_to_configured_base_directions() -> None:
    frame = RehabFrameConfig(
        rehab_x_axis_in_base=(0.0, 1.0, 0.0),
        rehab_z_axis_in_base=(0.0, 0.0, 1.0),
        reviewed=True,
        notes="axis mapping test",
    )
    rotation = frame.rotation_base_from_rehab
    dx = 0.025
    dz = 0.040
    mapped_x = rotation @ np.asarray((dx, 0.0, 0.0))
    mapped_z = rotation @ np.asarray((0.0, 0.0, dz))
    assert np.allclose(mapped_x, dx * np.asarray(frame.rehab_x_axis_in_base))
    assert np.allclose(mapped_z, dz * np.asarray(frame.rehab_z_axis_in_base))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_frame_rejects_nonunit_or_nonorthogonal_axes() -> None:
    with pytest.raises(ValueError, match="norm approximately one"):
        RehabFrameConfig((2.0, 0.0, 0.0), (0.0, 0.0, 1.0), False)
    with pytest.raises(ValueError, match="approximately orthogonal"):
        RehabFrameConfig((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), False)


def test_unreviewed_frame_can_be_previewed_but_is_not_silently_reviewed() -> None:
    trajectory, audit, metadata = _build(_identity_frame(reviewed=False))
    assert audit.trajectory_valid
    assert trajectory["trajectory_valid"].all()
    assert metadata["rehab_frame"]["reviewed"] is False
    assert metadata["robot_execution_approved"] is False


def test_template_requires_axes_and_review_is_never_inferred(tmp_path) -> None:
    path = tmp_path / "rehab_frame.json"
    path.write_text(
        json.dumps(
            {
                "rehab_x_axis_in_base": [1.0, 0.0, 0.0],
                "rehab_z_axis_in_base": [0.0, 0.0, 1.0],
                "reviewed": False,
                "notes": "draft",
            }
        ),
        encoding="utf-8",
    )
    frame = load_rehab_frame_config(path)
    assert frame.reviewed is False
    assert frame.notes == "draft"


def test_current_slow_reference_keeps_frozen_rom_theta_and_fk() -> None:
    trajectory, audit, metadata = _build()
    hip_deg = np.rad2deg(trajectory["q_hip_ref"].to_numpy(float))
    knee_deg = np.rad2deg(trajectory["q_knee_ref"].to_numpy(float))
    assert hip_deg.min() >= APPROVED_HIP_ROM_DEG[0] - 1e-10
    assert hip_deg.max() <= APPROVED_HIP_ROM_DEG[1] + 1e-10
    assert knee_deg.min() >= APPROVED_KNEE_ROM_DEG[0] - 1e-10
    assert knee_deg.max() <= APPROVED_KNEE_ROM_DEG[1] + 1e-10
    assert np.allclose(
        trajectory["theta_shank_ref"],
        trajectory["q_hip_ref"] - trajectory["q_knee_ref"],
        atol=1e-14,
        rtol=0.0,
    )
    assert audit.approved_rom_valid
    assert audit.theta_shank_definition_valid
    assert audit.pull_forward_kinematics_valid
    assert metadata["trajectory_id"] == FIRST_ROBOT_TRIAL_TRAJECTORY_ID
    assert metadata["allowed_for_first_robot_trial"] is True
    assert metadata["reference"]["sha256"] == (
        APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
    )
    assert metadata["reference_sha256_matches_approved_first_trial"] is True
    assert APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256 == (
        "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    )
    geometry = metadata["equivalent_pull_point_geometry"]
    assert geometry["L1_hip_to_knee_m"] == APPROVED_FIRST_ROBOT_TRIAL_L1_M
    assert geometry["L2_knee_to_equivalent_shank_strap_pull_point_m"] == (
        APPROVED_FIRST_ROBOT_TRIAL_L2_M
    )
    assert geometry["matches_approved_first_trial_geometry"] is True


def test_nonclosed_reference_is_invalid_not_silently_forced_closed() -> None:
    source, _ = load_closed_reference_trajectory(DEFAULT_REFERENCE_PATH)
    source.loc[len(source) - 1, "q_hip_rad"] += 0.05
    source.loc[len(source) - 1, "theta_shank_rad"] = (
        source.loc[len(source) - 1, "q_hip_rad"]
        - source.loc[len(source) - 1, "q_knee_rad"]
    )
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        source,
        current_tcp_start_pose=ANCHOR,
        rehab_frame=_identity_frame(),
    )
    assert not np.array_equal(
        trajectory[["delta_x_R", "delta_y_R", "delta_z_R"]].iloc[-1].to_numpy(float),
        np.zeros(3),
    )
    assert not audit.final_relative_displacement_zero
    assert "reference_not_closed" in audit.invalid_reasons
    assert not trajectory["trajectory_valid"].any()


def test_nominal_reference_is_retained_but_not_first_trial_whitelisted() -> None:
    nominal = DEFAULT_REFERENCE_PATH.with_name(
        "reference_measured_asymmetric_closed_nominal.csv"
    )
    trajectory, audit, metadata = build_start_anchored_relative_trajectory(
        nominal,
        current_tcp_start_pose=ANCHOR,
        rehab_frame=_identity_frame(),
    )
    assert not audit.trajectory_valid
    assert "source_formal_gate_invalid" in audit.invalid_reasons
    assert trajectory["trajectory_id"].eq(MEASURED_ASYMMETRIC_NOMINAL_ID).all()
    assert metadata["allowed_for_first_robot_trial"] is False
