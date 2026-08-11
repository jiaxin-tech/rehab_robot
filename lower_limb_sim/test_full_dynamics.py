"""完整双连杆动力学和动态力映射的离线测试。"""

import numpy as np

from lower_limb_sim.config import L1, L2
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.force_mapping import endpoint_force_from_joint_torque
from lower_limb_sim.full_dynamics import (
    center_of_mass_positions,
    damping_torque,
    gravity_vector,
    inverse_dynamics,
    kinetic_energy,
    mass_matrix,
    stiffness_torque,
)
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.quasi_static_dynamics import (
    gravity_torque,
    quasi_static_joint_torque,
)
from lower_limb_sim.simulate_dynamic_trajectory import (
    simulate_dynamic_trajectory,
)

SUBJECT = get_dynamic_subject("baseline")


def test_mass_matrix_is_symmetric() -> None:
    q_hip = np.deg2rad(np.array([10.0, 50.0, 100.0]))
    q_knee = np.deg2rad(np.array([10.0, 70.0, 120.0]))
    matrix = mass_matrix(q_hip, q_knee, SUBJECT, L1)

    assert np.allclose(matrix, np.swapaxes(matrix, -1, -2), atol=1e-12)


def test_mass_matrix_is_positive_definite() -> None:
    rng = np.random.default_rng(20260728)
    q_hip = np.deg2rad(rng.uniform(0.0, 120.0, size=500))
    q_knee = np.deg2rad(rng.uniform(5.0, 130.0, size=500))
    eigenvalues = np.linalg.eigvalsh(mass_matrix(q_hip, q_knee, SUBJECT, L1))

    assert np.isfinite(eigenvalues).all()
    assert (eigenvalues > 0.0).all()


def test_static_inverse_dynamics_matches_quasi_static_model() -> None:
    q_hip = np.deg2rad(65.0)
    q_knee = np.deg2rad(95.0)
    dynamic = inverse_dynamics(q_hip, q_knee, 0.0, 0.0, 0.0, 0.0, SUBJECT, L1)
    quasi_static = quasi_static_joint_torque(q_hip, q_knee, SUBJECT, L1)

    assert np.isclose(dynamic.tau_total_hip_nm, quasi_static.tau_total_hip_nm)
    assert np.isclose(dynamic.tau_total_knee_nm, quasi_static.tau_total_knee_nm)
    assert dynamic.tau_inertia_hip_nm == 0.0
    assert dynamic.tau_coriolis_hip_nm == 0.0
    assert dynamic.tau_damping_hip_nm == 0.0


def test_gravity_vector_matches_quasi_static_gravity() -> None:
    q_hip = np.deg2rad(np.array([20.0, 60.0, 100.0]))
    q_knee = np.deg2rad(np.array([30.0, 80.0, 120.0]))
    dynamic_gravity = gravity_vector(q_hip, q_knee, SUBJECT, L1)
    quasi_gravity = gravity_torque(q_hip, q_knee, SUBJECT, L1)

    assert np.allclose(dynamic_gravity[0], quasi_gravity[0], atol=1e-12)
    assert np.allclose(dynamic_gravity[1], quasi_gravity[1], atol=1e-12)


def test_zero_velocity_has_zero_damping() -> None:
    assert damping_torque(0.0, 0.0, SUBJECT) == (0.0, 0.0)


def test_neutral_angles_have_zero_stiffness() -> None:
    stiffness = stiffness_torque(
        SUBJECT.q0_hip_rad,
        SUBJECT.q0_knee_rad,
        SUBJECT,
    )
    assert stiffness == (0.0, 0.0)


def test_reversing_velocity_reverses_damping_torque() -> None:
    forward = np.asarray(damping_torque(0.4, -0.7, SUBJECT))
    reverse = np.asarray(damping_torque(-0.4, 0.7, SUBJECT))

    assert np.allclose(reverse, -forward)


def test_dynamic_endpoint_force_reconstructs_total_torque() -> None:
    q_hip = np.deg2rad(55.0)
    q_knee = np.deg2rad(85.0)
    dynamics = inverse_dynamics(
        q_hip,
        q_knee,
        0.3,
        0.5,
        -0.2,
        0.4,
        SUBJECT,
        L1,
    )
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        dynamics.tau_total_hip_nm,
        dynamics.tau_total_knee_nm,
        L1,
        L2,
    )
    reconstructed = leg_jacobian(q_hip, q_knee, L1, L2).T @ np.array(
        [force.fx_robot_on_leg_n, force.fz_robot_on_leg_n]
    )

    assert force.force_mapping_valid
    assert np.allclose(
        reconstructed,
        [dynamics.tau_total_hip_nm, dynamics.tau_total_knee_nm],
        atol=1e-9,
    )


def test_all_valid_dynamic_trajectory_values_are_finite() -> None:
    trajectory = simulate_dynamic_trajectory(SUBJECT, "fast")
    valid = trajectory["force_mapping_valid"].astype(bool)
    numeric = trajectory.select_dtypes(include=[np.number])

    assert valid.all()
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert trajectory["torque_reconstruction_error_nm"].max() < 1e-9


def test_shank_center_of_mass_uses_hip_minus_knee_angle() -> None:
    q_hip = np.deg2rad(70.0)
    q_knee = np.deg2rad(40.0)
    _, _, x_shank, z_shank = center_of_mass_positions(
        q_hip,
        q_knee,
        SUBJECT,
        L1,
    )
    expected = np.array(
        [
            L1 * np.cos(q_hip)
            + SUBJECT.com_shank_m * np.cos(q_hip - q_knee),
            L1 * np.sin(q_hip)
            + SUBJECT.com_shank_m * np.sin(q_hip - q_knee),
        ]
    )
    forbidden_plus_model = np.array(
        [
            L1 * np.cos(q_hip)
            + SUBJECT.com_shank_m * np.cos(q_hip + q_knee),
            L1 * np.sin(q_hip)
            + SUBJECT.com_shank_m * np.sin(q_hip + q_knee),
        ]
    )

    assert np.allclose([x_shank, z_shank], expected)
    assert not np.allclose([x_shank, z_shank], forbidden_plus_model)


def test_kinetic_energy_is_non_negative() -> None:
    rng = np.random.default_rng(20260728)
    q_hip = np.deg2rad(rng.uniform(0.0, 120.0, size=1000))
    q_knee = np.deg2rad(rng.uniform(5.0, 130.0, size=1000))
    dq_hip = rng.normal(0.0, 1.0, size=1000)
    dq_knee = rng.normal(0.0, 1.0, size=1000)
    energy = kinetic_energy(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
        SUBJECT,
        L1,
    )

    assert np.isfinite(energy).all()
    assert (energy >= 0.0).all()
