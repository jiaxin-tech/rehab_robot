"""阶段 4.5C 未见软件验证轨迹的回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    L1,
    L2,
    hip_range_deg,
    knee_range_deg,
)
from lower_limb_sim.generalization_trajectories import (
    BOUNDARY_TEST,
    GENERALIZATION_TRAJECTORIES,
    GENERALIZATION_TRAJECTORY_NAMES,
    INTERPOLATION_TEST,
    OUTSIDE_DOMAIN_TEST,
    SOFTWARE_VALIDATION_TRAJECTORY,
    TRAINING_HIP_RANGE_DEG,
    TRAINING_KNEE_RANGE_DEG,
    build_generalization_trajectory_set,
    generate_generalization_trajectory,
    validate_generalization_trajectory,
)
from lower_limb_sim.trajectory_profiles import (
    generate_identification_excitation_trajectory,
)


REQUIRED_KINEMATIC_COLUMNS = (
    "time_s",
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
)


@pytest.fixture(scope="module")
def trajectories() -> dict[str, pd.DataFrame]:
    return {
        name: generate_generalization_trajectory(name)
        for name in GENERALIZATION_TRAJECTORY_NAMES
    }


@pytest.fixture(scope="module")
def identification_state_amplitude_bounds() -> dict[str, float]:
    profiles = [
        generate_identification_excitation_trajectory(family, speed)
        for family in ("coupled", "hip_dominant", "knee_dominant")
        for speed in ("slow", "nominal", "fast")
    ]
    derivative_columns = (
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    return {
        column: max(float(profile[column].abs().max()) for profile in profiles)
        for column in derivative_columns
    }


def test_six_unique_families_and_required_split_assignment() -> None:
    assert GENERALIZATION_TRAJECTORY_NAMES == (
        "phase_shift_small",
        "amplitude_mix",
        "intermediate_speed",
        "asymmetric_flexion_extension",
        "boundary_near",
        "outside_domain",
    )
    assert len(set(GENERALIZATION_TRAJECTORY_NAMES)) == 6
    assert {
        name: spec.dataset_split
        for name, spec in GENERALIZATION_TRAJECTORIES.items()
    } == {
        "phase_shift_small": INTERPOLATION_TEST,
        "amplitude_mix": INTERPOLATION_TEST,
        "intermediate_speed": INTERPOLATION_TEST,
        "asymmetric_flexion_extension": INTERPOLATION_TEST,
        "boundary_near": BOUNDARY_TEST,
        "outside_domain": OUTSIDE_DOMAIN_TEST,
    }


def test_all_trajectories_are_explicitly_nonclinical_software_validation(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    for name, trajectory in trajectories.items():
        assert set(trajectory["generalization_family"]) == {name}
        assert set(trajectory["trajectory_family"]) == {name}
        assert set(trajectory["trajectory_id"]) == {
            f"{SOFTWARE_VALIDATION_TRAJECTORY}:{name}"
        }
        assert trajectory["software_validation_trajectory"].all()
        assert not trajectory["clinical_reference"].any()
        assert set(trajectory["dataset_split"]) == {
            GENERALIZATION_TRAJECTORIES[name].dataset_split
        }


def test_time_and_si_kinematics_are_finite_and_well_formed(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    for trajectory in trajectories.values():
        assert np.isfinite(
            trajectory[list(REQUIRED_KINEMATIC_COLUMNS)].to_numpy(dtype=float)
        ).all()
        assert np.all(np.diff(trajectory["time_s"].to_numpy(dtype=float)) > 0.0)
        assert trajectory["time_s"].iloc[0] == 0.0
        assert set(trajectory["phase"]) == {"flexion", "extension"}
        assert np.array_equal(
            trajectory["trajectory_sample_index"].to_numpy(dtype=int),
            np.arange(len(trajectory)),
        )
        endpoint_derivatives = trajectory.iloc[[0, -1]][
            [
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            ]
        ].to_numpy(dtype=float)
        assert np.allclose(endpoint_derivatives, 0.0, atol=1e-12, rtol=0.0)


def test_analytic_velocity_and_acceleration_match_sampled_derivatives(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    for name, trajectory in trajectories.items():
        time_s = trajectory["time_s"].to_numpy(dtype=float)
        flexion_end = int(np.flatnonzero(trajectory["phase"].eq("flexion"))[-1])
        mask = np.ones(len(trajectory), dtype=bool)
        mask[:3] = False
        mask[-3:] = False
        mask[max(0, flexion_end - 3) : flexion_end + 4] = False
        for q_column, dq_column in (
            ("q_hip_rad", "dq_hip_rad_s"),
            ("q_knee_rad", "dq_knee_rad_s"),
        ):
            numerical = np.gradient(
                trajectory[q_column].to_numpy(dtype=float),
                time_s,
                edge_order=2,
            )
            assert np.allclose(
                numerical[mask],
                trajectory[dq_column].to_numpy(dtype=float)[mask],
                atol=4e-5,
                rtol=2e-4,
            ), name
        for dq_column, ddq_column in (
            ("dq_hip_rad_s", "ddq_hip_rad_s2"),
            ("dq_knee_rad_s", "ddq_knee_rad_s2"),
        ):
            numerical = np.gradient(
                trajectory[dq_column].to_numpy(dtype=float),
                time_s,
                edge_order=2,
            )
            assert np.allclose(
                numerical[mask],
                trajectory[ddq_column].to_numpy(dtype=float)[mask],
                atol=5e-5,
                rtol=3e-4,
            ), name


def test_phase_shift_has_small_smooth_knee_lead(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    trajectory = trajectories["phase_shift_small"]
    flexion = trajectory.loc[trajectory["phase"].eq("flexion")]
    difference = (
        flexion["knee_path_progress"].to_numpy(dtype=float)
        - flexion["hip_path_progress"].to_numpy(dtype=float)
    )
    assert difference[0] == pytest.approx(0.0, abs=1e-14)
    assert difference[-1] == pytest.approx(0.0, abs=1e-14)
    assert difference.max() > 0.02
    assert difference.max() < 0.05
    assert (difference >= -1e-12).all()


def test_amplitude_mix_is_between_identification_path_amplitudes(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    trajectory = trajectories["amplitude_mix"]
    hip_amplitude_deg = np.rad2deg(
        trajectory["q_hip_rad"].max() - trajectory["q_hip_rad"].min()
    )
    knee_amplitude_deg = np.rad2deg(
        trajectory["q_knee_rad"].max() - trajectory["q_knee_rad"].min()
    )
    assert 50.0 < hip_amplitude_deg < 100.0
    assert 50.0 < knee_amplitude_deg < 100.0


def test_intermediate_and_asymmetric_timing_are_unseen_and_smooth(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    intermediate = trajectories["intermediate_speed"]
    turn_time = float(
        intermediate.loc[intermediate["phase"].eq("flexion"), "time_s"].iloc[-1]
    )
    assert 3.0 < turn_time < 6.0
    assert turn_time == pytest.approx(4.5)

    asymmetric = trajectories["asymmetric_flexion_extension"]
    flexion_duration = float(
        asymmetric.loc[asymmetric["phase"].eq("flexion"), "time_s"].iloc[-1]
    )
    total_duration = float(asymmetric["time_s"].iloc[-1])
    assert flexion_duration == pytest.approx(4.5)
    assert total_duration - flexion_duration == pytest.approx(8.0)
    assert flexion_duration != total_duration - flexion_duration


def test_nonoutside_trajectories_stay_inside_training_state_bounds(
    trajectories: dict[str, pd.DataFrame],
    identification_state_amplitude_bounds: dict[str, float],
) -> None:
    for name, trajectory in trajectories.items():
        if name == "outside_domain":
            continue
        hip_deg = np.rad2deg(trajectory["q_hip_rad"].to_numpy(dtype=float))
        knee_deg = np.rad2deg(trajectory["q_knee_rad"].to_numpy(dtype=float))
        assert hip_deg.min() >= TRAINING_HIP_RANGE_DEG[0] - 1e-12
        assert hip_deg.max() <= TRAINING_HIP_RANGE_DEG[1] + 1e-12
        assert knee_deg.min() >= TRAINING_KNEE_RANGE_DEG[0] - 1e-12
        assert knee_deg.max() <= TRAINING_KNEE_RANGE_DEG[1] + 1e-12
        assert not trajectory["outside_training_domain"].any()
        for column, training_bound in identification_state_amplitude_bounds.items():
            assert float(trajectory[column].abs().max()) <= training_bound + 1e-12


def test_boundary_near_approaches_but_does_not_cross_training_edges(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    trajectory = trajectories["boundary_near"]
    hip_deg = np.rad2deg(trajectory["q_hip_rad"].to_numpy(dtype=float))
    knee_deg = np.rad2deg(trajectory["q_knee_rad"].to_numpy(dtype=float))
    assert hip_deg.min() - TRAINING_HIP_RANGE_DEG[0] <= 1.0 + 1e-12
    assert TRAINING_HIP_RANGE_DEG[1] - hip_deg.max() <= 2.0 + 1e-12
    assert knee_deg.min() - TRAINING_KNEE_RANGE_DEG[0] <= 2.0 + 1e-12
    assert not trajectory["outside_training_domain"].any()
    assert set(trajectory["dataset_split"]) == {BOUNDARY_TEST}


def test_outside_domain_exceeds_training_only_and_stays_in_total_workspace(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    trajectory = trajectories["outside_domain"]
    hip_deg = np.rad2deg(trajectory["q_hip_rad"].to_numpy(dtype=float))
    knee_deg = np.rad2deg(trajectory["q_knee_rad"].to_numpy(dtype=float))
    outside = trajectory["outside_training_domain"].to_numpy(dtype=bool)
    assert outside.any()
    assert hip_deg.min() < TRAINING_HIP_RANGE_DEG[0]
    assert knee_deg.min() < TRAINING_KNEE_RANGE_DEG[0]
    assert knee_deg.max() > TRAINING_KNEE_RANGE_DEG[1]
    assert hip_deg.min() >= hip_range_deg[0] - 1e-12
    assert hip_deg.max() <= hip_range_deg[1] + 1e-12
    assert knee_deg.min() >= knee_range_deg[0] - 1e-12
    assert knee_deg.max() <= knee_range_deg[1] + 1e-12
    assert set(trajectory["dataset_split"]) == {OUTSIDE_DOMAIN_TEST}
    assert trajectory["trajectory_is_extrapolation"].all()


def test_all_trajectories_stay_above_bed_and_use_subtractive_shank_angle(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    for trajectory in trajectories.values():
        q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
        q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
        expected_x_pull = L1 * np.cos(q_hip) + L2 * np.cos(q_hip - q_knee)
        expected_z_pull = L1 * np.sin(q_hip) + L2 * np.sin(q_hip - q_knee)
        assert np.allclose(trajectory["x_pull_m"], expected_x_pull, atol=1e-12)
        assert np.allclose(trajectory["z_pull_m"], expected_z_pull, atol=1e-12)
        assert (trajectory["z_knee_m"] >= -1e-12).all()
        assert (trajectory["x_pull_m"] >= -1e-12).all()
        assert (trajectory["z_pull_m"] >= -1e-12).all()
        assert trajectory["workspace_valid"].all()


def test_builder_is_deterministic_and_preserves_local_monotonic_time() -> None:
    first = build_generalization_trajectory_set(sampling_frequency_hz=80.0)
    second = build_generalization_trajectory_set(sampling_frequency_hz=80.0)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert set(first["generalization_family"]) == set(
        GENERALIZATION_TRAJECTORY_NAMES
    )
    for _, trajectory in first.groupby("generalization_family", sort=False):
        assert np.all(np.diff(trajectory["time_s"].to_numpy(dtype=float)) > 0.0)


def test_builder_subset_and_invalid_arguments() -> None:
    subset = build_generalization_trajectory_set(
        ("phase_shift_small", "outside_domain"),
        sampling_frequency_hz=50.0,
    )
    assert list(subset["generalization_family"].drop_duplicates()) == [
        "phase_shift_small",
        "outside_domain",
    ]
    with pytest.raises(ValueError, match="Unknown generalization trajectory"):
        generate_generalization_trajectory("not_a_trajectory")
    with pytest.raises(ValueError, match="finite and positive"):
        generate_generalization_trajectory("phase_shift_small", 0.0)
    with pytest.raises(ValueError, match="at least one"):
        build_generalization_trajectory_set(())
    with pytest.raises(ValueError, match="duplicates"):
        build_generalization_trajectory_set(
            ("phase_shift_small", "phase_shift_small")
        )


def test_validator_fails_closed_for_false_domain_marker(
    trajectories: dict[str, pd.DataFrame],
) -> None:
    damaged = trajectories["outside_domain"].copy(deep=True)
    damaged["outside_training_domain"] = False
    with pytest.raises(ValueError, match="markers are inconsistent"):
        validate_generalization_trajectory(damaged)

