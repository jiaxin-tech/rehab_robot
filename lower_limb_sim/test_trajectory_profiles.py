"""解析最小 jerk 测试轨迹和速度缩放测试。"""

import numpy as np

from lower_limb_sim.compare_speed_profiles import (
    build_speed_profile_comparison,
)
from lower_limb_sim.config import dynamic_sampling_frequency_hz
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.simulate_dynamic_trajectory import (
    simulate_dynamic_trajectory,
)
from lower_limb_sim.trajectory_profiles import (
    generate_software_test_trajectory,
)


def _profiles():
    return {
        profile: generate_software_test_trajectory(profile)
        for profile in ("slow", "nominal", "fast")
    }


def test_start_and_end_velocity_are_zero() -> None:
    trajectory = generate_software_test_trajectory("nominal")
    columns = ["dq_hip_rad_s", "dq_knee_rad_s"]
    assert np.allclose(trajectory.iloc[[0, -1]][columns], 0.0, atol=1e-12)


def test_start_and_end_acceleration_are_zero() -> None:
    trajectory = generate_software_test_trajectory("nominal")
    columns = ["ddq_hip_rad_s2", "ddq_knee_rad_s2"]
    assert np.allclose(trajectory.iloc[[0, -1]][columns], 0.0, atol=1e-12)


def test_flexion_reaches_requested_maximum_angles() -> None:
    trajectory = generate_software_test_trajectory("nominal")
    flexion_end = trajectory.loc[trajectory["phase"] == "flexion"].iloc[-1]

    assert np.isclose(flexion_end["q_hip_rad"], np.deg2rad(120.0))
    assert np.isclose(flexion_end["q_knee_rad"], np.deg2rad(120.0))
    assert np.isclose(flexion_end["dq_hip_rad_s"], 0.0)
    assert np.isclose(flexion_end["ddq_hip_rad_s2"], 0.0)


def test_extension_returns_to_start_angles() -> None:
    trajectory = generate_software_test_trajectory("nominal")

    assert np.allclose(
        trajectory.iloc[-1][["q_hip_rad", "q_knee_rad"]].to_numpy(dtype=float),
        np.deg2rad([20.0, 20.0]),
    )


def test_time_is_strictly_increasing() -> None:
    trajectory = generate_software_test_trajectory("nominal")
    assert (np.diff(trajectory["time_s"]) > 0.0).all()


def test_flexion_extension_connection_is_not_duplicated() -> None:
    trajectory = generate_software_test_trajectory("nominal")
    maximum = np.deg2rad([120.0, 120.0])
    at_maximum = np.isclose(
        trajectory["q_hip_rad"],
        maximum[0],
        rtol=0.0,
        atol=1e-14,
    ) & np.isclose(
        trajectory["q_knee_rad"],
        maximum[1],
        rtol=0.0,
        atol=1e-14,
    )

    assert at_maximum.sum() == 1


def test_speed_profiles_share_identical_geometric_path() -> None:
    profiles = _profiles()
    ranges = []
    for trajectory in profiles.values():
        ranges.append(
            trajectory[
                ["q_hip_rad", "q_knee_rad"]
            ].agg(["min", "max"]).to_numpy()
        )
        hip_progress = (
            trajectory["q_hip_rad"] - trajectory["q_hip_rad"].min()
        ) / (
            trajectory["q_hip_rad"].max() - trajectory["q_hip_rad"].min()
        )
        knee_progress = (
            trajectory["q_knee_rad"] - trajectory["q_knee_rad"].min()
        ) / (
            trajectory["q_knee_rad"].max() - trajectory["q_knee_rad"].min()
        )
        assert np.allclose(hip_progress, knee_progress, atol=1e-12)
    assert np.allclose(ranges[0], ranges[1])
    assert np.allclose(ranges[0], ranges[2])


def test_faster_profiles_have_higher_peak_velocity_and_acceleration() -> None:
    profiles = _profiles()
    peak_velocity = [
        profiles[name]["dq_knee_rad_s"].abs().max()
        for name in ("slow", "nominal", "fast")
    ]
    peak_acceleration = [
        profiles[name]["ddq_knee_rad_s2"].abs().max()
        for name in ("slow", "nominal", "fast")
    ]

    assert peak_velocity[0] < peak_velocity[1] < peak_velocity[2]
    assert peak_acceleration[0] < peak_acceleration[1] < peak_acceleration[2]


def test_all_dynamic_trajectory_points_are_above_bed_and_reachable() -> None:
    subject = get_dynamic_subject("baseline")
    for profile in ("slow", "nominal", "fast"):
        trajectory = simulate_dynamic_trajectory(subject, profile)
        assert (trajectory["z_knee_m"] >= 0.0).all()
        assert (trajectory["z_pull_m"] >= 0.0).all()
        assert (trajectory["x_pull_m"] >= 0.0).all()
        assert not trajectory["jacobian_near_singular"].any()


def test_sampling_frequency_is_close_to_100_hz() -> None:
    trajectory = generate_software_test_trajectory("slow")
    observed_frequency = 1.0 / np.median(np.diff(trajectory["time_s"]))

    assert np.isclose(
        observed_frequency,
        dynamic_sampling_frequency_hz,
        rtol=1e-9,
    )


def test_dynamic_speed_comparison_validates_expected_scaling() -> None:
    subject = get_dynamic_subject("baseline")
    trajectories = {
        profile: simulate_dynamic_trajectory(subject, profile)
        for profile in ("slow", "nominal", "fast")
    }
    comparison = build_speed_profile_comparison(
        "baseline",
        trajectories=trajectories,
    )

    assert list(comparison["speed_profile"]) == ["slow", "nominal", "fast"]
    assert comparison["invalid_force_samples"].eq(0).all()
