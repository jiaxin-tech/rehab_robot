"""虚拟受试者准静态力矩、力映射和力地图测试。"""

import numpy as np
import pytest

from lower_limb_sim.build_force_map import build_force_map, save_force_map
from lower_limb_sim.config import L1, L2
from lower_limb_sim.force_mapping import endpoint_force_from_joint_torque
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.quasi_static_dynamics import (
    gravity_torque,
    passive_stiffness_torque,
    quasi_static_joint_torque,
)
from lower_limb_sim.virtual_subject import (
    BASELINE_SUBJECT,
    HEAVY_LEG_SUBJECT,
    HIP_STIFF_SUBJECT,
    KNEE_STIFF_SUBJECT,
    VirtualSubject,
)
from lower_limb_sim.workspace_atlas import build_workspace_atlas


def test_passive_stiffness_is_zero_at_neutral_angles() -> None:
    tau_hip, tau_knee = passive_stiffness_torque(
        BASELINE_SUBJECT.q0_hip_rad,
        BASELINE_SUBJECT.q0_knee_rad,
        BASELINE_SUBJECT,
    )

    assert tau_hip == 0.0
    assert tau_knee == 0.0


def test_hip_stiff_subject_increases_hip_stiffness_torque() -> None:
    q_hip = np.deg2rad(90.0)
    q_knee = np.deg2rad(60.0)
    baseline_hip, _ = passive_stiffness_torque(
        q_hip,
        q_knee,
        BASELINE_SUBJECT,
    )
    stiff_hip, _ = passive_stiffness_torque(
        q_hip,
        q_knee,
        HIP_STIFF_SUBJECT,
    )

    assert stiff_hip > baseline_hip


def test_knee_stiff_subject_increases_knee_stiffness_torque() -> None:
    q_hip = np.deg2rad(60.0)
    q_knee = np.deg2rad(110.0)
    _, baseline_knee = passive_stiffness_torque(
        q_hip,
        q_knee,
        BASELINE_SUBJECT,
    )
    _, stiff_knee = passive_stiffness_torque(
        q_hip,
        q_knee,
        KNEE_STIFF_SUBJECT,
    )

    assert stiff_knee > baseline_knee


def test_heavy_leg_increases_gravity_torque_norm() -> None:
    q_hip = np.deg2rad(55.0)
    q_knee = np.deg2rad(75.0)
    baseline = gravity_torque(q_hip, q_knee, BASELINE_SUBJECT, L1)
    heavy = gravity_torque(q_hip, q_knee, HEAVY_LEG_SUBJECT, L1)

    assert np.hypot(*heavy) > np.hypot(*baseline)


def test_endpoint_force_reconstructs_random_joint_torques() -> None:
    rng = np.random.default_rng(20260727)
    q_hips = np.deg2rad(rng.uniform(10.0, 110.0, size=200))
    q_knees = np.deg2rad(rng.uniform(10.0, 125.0, size=200))
    torque = quasi_static_joint_torque(
        q_hips,
        q_knees,
        BASELINE_SUBJECT,
        L1,
    )
    force = endpoint_force_from_joint_torque(
        q_hips,
        q_knees,
        torque.tau_total_hip_nm,
        torque.tau_total_knee_nm,
        L1,
        L2,
    )
    assert np.all(force.force_mapping_valid)

    jacobian = leg_jacobian(q_hips, q_knees, L1, L2)
    endpoint_force = np.stack(
        (force.fx_robot_on_leg_n, force.fz_robot_on_leg_n),
        axis=-1,
    )
    reconstructed = np.matmul(
        np.swapaxes(jacobian, -1, -2),
        endpoint_force[..., np.newaxis],
    )[..., 0]
    expected = np.stack(
        (torque.tau_total_hip_nm, torque.tau_total_knee_nm),
        axis=-1,
    )
    assert np.max(np.abs(reconstructed - expected)) < 1e-8


def test_singular_posture_invalidates_force_mapping() -> None:
    force = endpoint_force_from_joint_torque(
        np.deg2rad(40.0),
        0.0,
        10.0,
        5.0,
        L1,
        L2,
    )

    assert force.force_mapping_valid is False
    assert force.jacobian_near_singular is True
    assert force.invalid_reason == "jacobian_near_singular"
    assert np.isnan(force.fx_robot_on_leg_n)
    assert np.isnan(force.fz_robot_on_leg_n)


def test_excessive_force_is_invalidated_with_reason() -> None:
    force = endpoint_force_from_joint_torque(
        np.deg2rad(50.0),
        np.deg2rad(70.0),
        1000.0,
        1000.0,
        L1,
        L2,
        force_limit_n=100.0,
    )

    assert force.force_mapping_valid is False
    assert force.invalid_reason == "force_magnitude_limit_exceeded"
    assert np.isnan(force.force_magnitude_n)


def test_virtual_subject_rejects_invalid_physical_parameters() -> None:
    with pytest.raises(ValueError, match="mass_thigh_kg"):
        VirtualSubject(
            subject_id="invalid",
            mass_thigh_kg=0.0,
            mass_shank_kg=4.0,
            com_thigh_m=0.18,
            com_shank_m=0.16,
            k_hip_nm_per_rad=15.0,
            k_knee_nm_per_rad=12.0,
            q0_hip_rad=0.0,
            q0_knee_rad=0.0,
        )


def test_all_valid_force_map_rows_are_finite_and_save_as_npz(tmp_path) -> None:
    workspace = build_workspace_atlas(step_deg=5.0)
    force_map = build_force_map(BASELINE_SUBJECT, workspace)
    valid = force_map["force_mapping_valid"]
    finite_columns = [
        "tau_gravity_hip_nm",
        "tau_gravity_knee_nm",
        "tau_stiffness_hip_nm",
        "tau_stiffness_knee_nm",
        "tau_total_hip_nm",
        "tau_total_knee_nm",
        "fx_robot_on_leg_n",
        "fz_robot_on_leg_n",
        "force_magnitude_n",
        "jacobian_determinant",
        "jacobian_condition_number",
    ]

    assert valid.any()
    assert np.isfinite(
        force_map.loc[valid, finite_columns].to_numpy(dtype=float)
    ).all()
    csv_path, npz_path = save_force_map(force_map, tmp_path)
    assert csv_path.exists()
    with np.load(npz_path, allow_pickle=False) as saved:
        assert set(finite_columns).issubset(saved.files)
        assert len(saved["q_hip_rad"]) == len(force_map)
