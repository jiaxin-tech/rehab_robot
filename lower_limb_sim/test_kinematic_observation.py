"""Stage 4.5D IK reconstruction and observation-boundary tests."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lower_limb_sim.angle_reconstruction import (
    PHYSIOLOGICAL_BRANCH,
    reconstruct_joint_angles_from_pull_point,
)
from lower_limb_sim.geometry_calibration import AssumedGeometry, TrueGeometry
from lower_limb_sim.kinematic_observation import (
    INDEPENDENT_JOINT_MEASUREMENT,
    ORACLE_TRUE_JOINT_STATE,
    TCP_INVERSE_KINEMATICS,
    build_kinematic_observation,
    synthesize_independent_joint_measurements,
    synthesize_tcp_position_measurements,
)
from lower_limb_sim.kinematics import forward_kinematics


L1_M = 0.42
L2_M = 0.30


def _geometry(
    *,
    L1_m: float = L1_M,
    L2_m: float = L2_M,
    hip_x_m: float = 0.0,
    hip_z_m: float = 0.0,
) -> AssumedGeometry:
    return AssumedGeometry(
        L1_assumed_m=L1_m,
        L2_assumed_m=L2_m,
        hip_center_x_assumed_m=hip_x_m,
        hip_center_z_assumed_m=hip_z_m,
        q0_hip_assumed_rad=np.deg2rad(10.0),
        q0_knee_assumed_rad=np.deg2rad(15.0),
    )


def _smooth_angles(count: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.linspace(0.0, 2.0, count)
    phase = np.pi * time_s / time_s[-1]
    q_hip = np.deg2rad(30.0 + 20.0 * (1.0 - np.cos(phase)))
    q_knee = np.deg2rad(40.0 + 30.0 * (1.0 - np.cos(phase)))
    return time_s, q_hip, q_knee


def _measured_tcp_dataframe(count: int = 101) -> pd.DataFrame:
    time_s, q_hip, q_knee = _smooth_angles(count)
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    return pd.DataFrame(
        {
            "trajectory_id": "trajectory-a",
            "trajectory_family": "coupled",
            "speed_profile": "nominal",
            "dataset_split": "train",
            "phase": "flexion",
            "time_s": time_s,
            "trajectory_sample_index": np.arange(count),
            "x_pull_measured_m": x_pull,
            "z_pull_measured_m": z_pull,
            # These decoys must never influence a practical observation mode.
            "q_hip_true_rad": q_hip,
            "q_knee_true_rad": q_knee,
            "dq_hip_true_rad_s": np.gradient(q_hip, time_s),
            "dq_knee_true_rad_s": np.gradient(q_knee, time_s),
            "ddq_hip_true_rad_s2": np.gradient(
                np.gradient(q_hip, time_s), time_s
            ),
            "ddq_knee_true_rad_s2": np.gradient(
                np.gradient(q_knee, time_s), time_s
            ),
        }
    )


def test_matched_geometry_recovers_original_angles() -> None:
    time_s, q_hip, q_knee = _smooth_angles()
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )

    result = reconstruct_joint_angles_from_pull_point(
        x_pull,
        z_pull,
        assumed_geometry=_geometry(),
        time_s=time_s,
    )

    assert result.dataframe["ik_valid"].all()
    assert np.max(np.abs(result.q_hip_est_rad - q_hip)) < 1e-12
    assert np.max(np.abs(result.q_knee_est_rad - q_knee)) < 1e-12
    assert (
        result.dataframe["branch_selected"] == PHYSIOLOGICAL_BRANCH
    ).all()
    assert result.metadata["branch_selection_uses_true_angles"] is False


def test_reconstruction_retains_qhip_minus_qknee_shank_definition() -> None:
    q_hip = np.deg2rad(np.array([35.0, 48.0, 62.0]))
    q_knee = np.deg2rad(np.array([45.0, 75.0, 105.0]))
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    result = reconstruct_joint_angles_from_pull_point(
        x_pull,
        z_pull,
        assumed_geometry=_geometry(),
        maximum_joint_jump_rad=np.deg2rad(40.0),
    ).dataframe

    shank_angle_from_points = np.arctan2(
        z_pull - z_knee,
        x_pull - x_knee,
    )
    assert np.allclose(
        shank_angle_from_points,
        result["q_hip_est_rad"] - result["q_knee_est_rad"],
        atol=1e-12,
    )


def test_ik_interface_and_branch_selection_do_not_accept_true_q() -> None:
    signature = inspect.signature(reconstruct_joint_angles_from_pull_point)
    assert not any("true" in name.lower() for name in signature.parameters)
    assert not any(name.startswith("q_") for name in signature.parameters)

    original = _measured_tcp_dataframe()
    tampered = original.copy()
    tampered["q_hip_true_rad"] = 1e6
    tampered["q_knee_true_rad"] = -1e6
    tampered["dq_hip_true_rad_s"] = np.nan
    tampered["ddq_knee_true_rad_s2"] = np.inf
    result_a = build_kinematic_observation(
        original,
        TCP_INVERSE_KINEMATICS,
        assumed_geometry=_geometry(),
    )
    result_b = build_kinematic_observation(
        tampered,
        TCP_INVERSE_KINEMATICS,
        assumed_geometry=_geometry(),
    )
    columns = [
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "observation_valid",
    ]
    assert_frame_equal(result_a.dataframe[columns], result_b.dataframe[columns])
    assert not any("true" in column for column in result_a.dataframe)
    assert result_a.metadata["reads_true_joint_position"] is False
    assert result_a.metadata["reads_true_joint_velocity"] is False
    assert result_a.metadata["reads_true_joint_acceleration"] is False


def test_true_geometry_object_is_rejected_by_reconstruction() -> None:
    true_geometry = TrueGeometry(
        L1_true_m=L1_M,
        L2_true_m=L2_M,
        hip_center_x_true_m=0.0,
        hip_center_z_true_m=0.0,
        q0_hip_true_rad=0.0,
        q0_knee_true_rad=0.0,
    )
    with pytest.raises(TypeError, match="true_geometry"):
        reconstruct_joint_angles_from_pull_point(
            [0.5],
            [0.2],
            assumed_geometry=true_geometry,
        )


def test_assumed_hip_center_offset_changes_reconstructed_angles() -> None:
    q_hip = np.deg2rad(np.array([35.0, 45.0, 55.0]))
    q_knee = np.deg2rad(np.array([55.0, 70.0, 85.0]))
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    matched = reconstruct_joint_angles_from_pull_point(
        x_pull, z_pull, assumed_geometry=_geometry()
    ).dataframe
    shifted = reconstruct_joint_angles_from_pull_point(
        x_pull,
        z_pull,
        assumed_geometry=_geometry(hip_x_m=0.01, hip_z_m=-0.005),
    ).dataframe

    valid = matched["ik_valid"] & shifted["ik_valid"]
    assert valid.all()
    assert np.max(
        np.abs(
            shifted.loc[valid, "q_hip_est_rad"]
            - matched.loc[valid, "q_hip_est_rad"]
        )
    ) > np.deg2rad(0.2)


def test_assumed_L2_error_changes_knee_angle_reconstruction() -> None:
    q_hip = np.deg2rad(np.array([35.0, 45.0, 55.0]))
    q_knee = np.deg2rad(np.array([50.0, 75.0, 95.0]))
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    matched = reconstruct_joint_angles_from_pull_point(
        x_pull, z_pull, assumed_geometry=_geometry()
    ).dataframe
    wrong_L2 = reconstruct_joint_angles_from_pull_point(
        x_pull,
        z_pull,
        assumed_geometry=_geometry(L2_m=L2_M + 0.02),
    ).dataframe

    valid = matched["ik_valid"] & wrong_L2["ik_valid"]
    assert valid.any()
    knee_error = (
        wrong_L2.loc[valid, "q_knee_est_rad"]
        - matched.loc[valid, "q_knee_est_rad"]
    )
    assert np.max(np.abs(knee_error)) > np.deg2rad(1.0)


def test_geometrically_invalid_ik_is_rejected_with_reason() -> None:
    result = reconstruct_joint_angles_from_pull_point(
        [2.0, np.nan],
        [2.0, 0.1],
        assumed_geometry=_geometry(),
    ).dataframe

    assert not result["ik_valid"].any()
    assert result.loc[0, "ik_reason"] == "acos_domain_error"
    assert "nonfinite_pull_point" in result.loc[1, "ik_reason"]
    assert result[["q_hip_est_rad", "q_knee_est_rad"]].isna().all(axis=None)


def test_below_bed_solution_is_rejected() -> None:
    q_hip = np.deg2rad(20.0)
    q_knee = np.deg2rad(100.0)
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    assert z_pull < 0.0
    result = reconstruct_joint_angles_from_pull_point(
        [x_pull], [z_pull], assumed_geometry=_geometry()
    ).dataframe
    assert not result.loc[0, "ik_valid"]
    assert "bed_constraint_violation" in result.loc[0, "ik_reason"]


def test_angle_continuity_check_rejects_large_jump() -> None:
    q_hip = np.deg2rad(np.array([30.0, 32.0, 85.0]))
    q_knee = np.deg2rad(np.array([50.0, 52.0, 110.0]))
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    result = reconstruct_joint_angles_from_pull_point(
        x_pull,
        z_pull,
        assumed_geometry=_geometry(),
        maximum_joint_jump_rad=np.deg2rad(10.0),
    ).dataframe

    assert result.loc[:1, "ik_valid"].all()
    assert not result.loc[2, "ik_valid"]
    assert not result.loc[2, "joint_continuity_valid"]
    assert "joint_angle_jump" in result.loc[2, "ik_reason"]


def test_domain_clip_is_audited_and_never_silently_accepted() -> None:
    tiny_radial_error = 1e-12
    result = reconstruct_joint_angles_from_pull_point(
        [L1_M + L2_M + tiny_radial_error],
        [0.0],
        assumed_geometry=_geometry(),
        acos_domain_tolerance=1e-10,
    ).dataframe

    assert result.loc[0, "ik_domain_clip_applied"]
    assert result.loc[0, "ik_domain_clip_amount"] > 0.0
    # Knee=0 is outside the configured physiological range; domain clipping
    # must not override that independent validity check.
    assert not result.loc[0, "ik_valid"]


def test_valid_ik_samples_are_all_finite_and_reproducible() -> None:
    dataframe = _measured_tcp_dataframe()
    kwargs = {
        "assumed_geometry": _geometry(),
        "time_s": dataframe["time_s"],
        "trajectory_ids": dataframe["trajectory_id"],
    }
    first = reconstruct_joint_angles_from_pull_point(
        dataframe["x_pull_measured_m"],
        dataframe["z_pull_measured_m"],
        **kwargs,
    )
    second = reconstruct_joint_angles_from_pull_point(
        dataframe["x_pull_measured_m"],
        dataframe["z_pull_measured_m"],
        **kwargs,
    )
    assert_frame_equal(first.dataframe, second.dataframe)
    valid = first.dataframe["ik_valid"]
    required = [
        "q_hip_est_rad",
        "q_knee_est_rad",
        "ik_position_reconstruction_error_m",
        "ik_jacobian_condition_number",
    ]
    assert np.isfinite(first.dataframe.loc[valid, required]).all(axis=None)


def test_tcp_observation_reconstructs_derivatives_from_ik_angles() -> None:
    dataframe = _measured_tcp_dataframe()
    result = build_kinematic_observation(
        dataframe,
        TCP_INVERSE_KINEMATICS,
        assumed_geometry=_geometry(),
        derivative_method="savitzky_golay_offline",
    )

    output = result.dataframe
    assert output["observation_valid"].all()
    assert result.metadata["derivatives_reconstructed_from_observed_angles"]
    assert result.metadata["uses_future_samples"]
    assert not any("true" in column for column in output)
    assert np.isfinite(
        output.loc[output["observation_valid"], [
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]]
    ).all(axis=None)


def test_independent_observer_reads_only_noisy_angle_measurements() -> None:
    time_s, q_hip, q_knee = _smooth_angles()
    measured = synthesize_independent_joint_measurements(
        q_hip,
        q_knee,
        noise_standard_deviation_rad=np.deg2rad(0.5),
        random_seed=42,
    )
    dataframe = pd.DataFrame(
        {
            "time_s": time_s,
            "trajectory_id": "camera-a",
            "q_hip_true_rad": q_hip,
            "q_knee_true_rad": q_knee,
            "dq_hip_true_rad_s": 123.0,
            "dq_knee_true_rad_s": -456.0,
            **measured,
        }
    )
    first = build_kinematic_observation(
        dataframe,
        INDEPENDENT_JOINT_MEASUREMENT,
    )
    dataframe["q_hip_true_rad"] = -1e6
    dataframe["dq_hip_true_rad_s"] = np.nan
    second = build_kinematic_observation(
        dataframe,
        INDEPENDENT_JOINT_MEASUREMENT,
    )

    assert_frame_equal(first.dataframe, second.dataframe)
    assert not any("true" in column for column in first.dataframe)
    assert first.metadata["reads_true_joint_position"] is False
    assert first.metadata["reads_true_joint_velocity"] is False
    assert first.metadata["reads_true_joint_acceleration"] is False


def test_oracle_mode_is_explicitly_labelled_upper_bound() -> None:
    dataframe = _measured_tcp_dataframe()
    result = build_kinematic_observation(
        dataframe,
        ORACLE_TRUE_JOINT_STATE,
    )
    assert result.dataframe["observation_valid"].all()
    assert result.metadata["oracle_upper_bound_only"] is True
    assert result.metadata["practical_observation_mode"] is False
    assert result.metadata["reads_true_joint_velocity"] is True
    assert result.metadata["reads_true_joint_acceleration"] is True


def test_measurement_synthesis_is_seed_reproducible() -> None:
    _, q_hip, q_knee = _smooth_angles(25)
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1_M, L2_M
    )
    angle_a = synthesize_independent_joint_measurements(
        q_hip,
        q_knee,
        noise_standard_deviation_rad=0.01,
        random_seed=7,
    )
    angle_b = synthesize_independent_joint_measurements(
        q_hip,
        q_knee,
        noise_standard_deviation_rad=0.01,
        random_seed=7,
    )
    tcp_a = synthesize_tcp_position_measurements(
        x_pull,
        z_pull,
        noise_standard_deviation_m=0.001,
        random_seed=11,
    )
    tcp_b = synthesize_tcp_position_measurements(
        x_pull,
        z_pull,
        noise_standard_deviation_m=0.001,
        random_seed=11,
    )
    assert_frame_equal(angle_a, angle_b)
    assert_frame_equal(tcp_a, tcp_b)


def test_nonoracle_modes_reject_true_angle_reconstruction_options() -> None:
    with pytest.raises(ValueError, match="true angles/geometry"):
        build_kinematic_observation(
            _measured_tcp_dataframe(),
            TCP_INVERSE_KINEMATICS,
            assumed_geometry=_geometry(),
            reconstruction_options={"q_true": np.array([0.0])},
        )
