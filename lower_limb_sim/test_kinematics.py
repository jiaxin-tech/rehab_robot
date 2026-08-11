"""二维下肢运动学、图谱和查询的离线测试。"""

import numpy as np

from lower_limb_sim.config import L1, L2, hip_range_deg, knee_range_deg
from lower_limb_sim.kinematics import forward_kinematics, inverse_kinematics
from lower_limb_sim.query import query_position
from lower_limb_sim.workspace_atlas import (
    build_workspace_atlas,
    save_workspace_atlas,
)


def test_forward_inverse_round_trip_random_angles() -> None:
    rng = np.random.default_rng(20260726)
    q_hip = np.deg2rad(rng.uniform(*hip_range_deg, size=1000))
    q_knee = np.deg2rad(rng.uniform(*knee_range_deg, size=1000))

    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    recovered_hip, recovered_knee, reachable = inverse_kinematics(
        x_pull,
        z_pull,
        L1,
        L2,
    )

    assert np.all(reachable)
    assert np.max(np.abs(recovered_hip - q_hip)) < 1e-6
    assert np.max(np.abs(recovered_knee - q_knee)) < 1e-6


def test_inverse_kinematics_marks_out_of_reach_target() -> None:
    q_hip, q_knee, reachable = inverse_kinematics(2.0, 0.0, L1, L2)

    assert reachable is False
    assert np.isnan(q_hip)
    assert np.isnan(q_knee)


def test_coupled_flexion_lifts_knee_and_retracts_pull_point() -> None:
    # “抬膝”是髋、膝共同屈曲的动作；膝点升高实际由 q_hip 增大产生。
    q_hip = np.deg2rad([20.0, 70.0])
    q_knee = np.deg2rad([20.0, 120.0])
    _, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)

    assert q_knee[1] > q_knee[0]
    assert z_knee[1] > z_knee[0]
    assert x_pull[1] < x_pull[0]
    assert np.all(z_pull >= 0.0)


def test_knee_flexion_alone_does_not_move_knee_joint() -> None:
    # 保护人体两连杆定义：q_knee 只能改变小腿/牵引点，不能改变膝点。
    q_hip = np.deg2rad(45.0)
    first = forward_kinematics(q_hip, np.deg2rad(20.0), L1, L2)
    second = forward_kinematics(q_hip, np.deg2rad(100.0), L1, L2)

    assert np.isclose(first[0], second[0])
    assert np.isclose(first[1], second[1])


def test_below_bed_postures_are_filtered() -> None:
    atlas = build_workspace_atlas(step_deg=5.0)
    below_bed = atlas["z_pull"] < 0.0

    assert below_bed.any()
    assert not atlas.loc[below_bed, "reachable"].any()
    assert (atlas.loc[atlas["reachable"], "z_pull"] >= 0.0).all()
    assert (atlas.loc[atlas["reachable"], "x_pull"] >= 0.0).all()


def test_query_position_after_loading_csv(tmp_path) -> None:
    atlas = build_workspace_atlas(step_deg=5.0)
    csv_path, _ = save_workspace_atlas(atlas, tmp_path)
    expected = atlas.loc[atlas["reachable"]].iloc[len(atlas) // 3]

    result = query_position(
        float(expected["x_pull"]),
        float(expected["z_pull"]),
        atlas_path=csv_path,
        max_distance=1e-9,
    )

    assert result["reachable"]
    assert result["distance_error"] < 1e-9
    assert np.isclose(result["q_hip"], expected["q_hip_rad"])
    assert np.isclose(result["q_knee"], expected["q_knee_rad"])
