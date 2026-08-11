"""牵引点解析 Jacobian 的离线数值测试。"""

import numpy as np

from lower_limb_sim.config import L1, L2
from lower_limb_sim.jacobian import jacobian_diagnostics, leg_jacobian
from lower_limb_sim.kinematics import forward_kinematics


def _numerical_jacobian(q_hip: float, q_knee: float) -> np.ndarray:
    epsilon = 1e-7

    def pull_position(hip: float, knee: float) -> np.ndarray:
        _, _, x_pull, z_pull = forward_kinematics(hip, knee, L1, L2)
        return np.array([x_pull, z_pull])

    derivative_hip = (
        pull_position(q_hip + epsilon, q_knee)
        - pull_position(q_hip - epsilon, q_knee)
    ) / (2.0 * epsilon)
    derivative_knee = (
        pull_position(q_hip, q_knee + epsilon)
        - pull_position(q_hip, q_knee - epsilon)
    ) / (2.0 * epsilon)
    return np.column_stack((derivative_hip, derivative_knee))


def test_analytic_jacobian_matches_finite_difference() -> None:
    q_hip = np.deg2rad(60.0)
    q_knee = np.deg2rad(80.0)
    analytic = leg_jacobian(q_hip, q_knee, L1, L2)

    assert np.max(np.abs(analytic - _numerical_jacobian(q_hip, q_knee))) < 1e-6


def test_random_analytic_jacobians_match_numerical_values() -> None:
    rng = np.random.default_rng(20260727)
    q_hips = np.deg2rad(rng.uniform(5.0, 115.0, size=100))
    q_knees = np.deg2rad(rng.uniform(10.0, 125.0, size=100))

    errors = [
        np.max(
            np.abs(
                leg_jacobian(q_hip, q_knee, L1, L2)
                - _numerical_jacobian(q_hip, q_knee)
            )
        )
        for q_hip, q_knee in zip(q_hips, q_knees)
    ]
    assert max(errors) < 1e-6


def test_second_column_uses_hip_minus_knee_signs() -> None:
    q_hip = np.deg2rad(60.0)
    q_knee = np.deg2rad(20.0)
    shank_angle = q_hip - q_knee
    jacobian = leg_jacobian(q_hip, q_knee, L1, L2)

    assert np.isclose(jacobian[0, 1], L2 * np.sin(shank_angle))
    assert np.isclose(jacobian[1, 1], -L2 * np.cos(shank_angle))
    assert jacobian[0, 1] > 0.0
    assert jacobian[1, 1] < 0.0


def test_fully_extended_knee_is_detected_as_singular() -> None:
    diagnostics = jacobian_diagnostics(
        np.deg2rad(40.0),
        0.0,
        L1,
        L2,
    )

    assert abs(diagnostics.determinant) < 1e-12
    assert diagnostics.near_singular
    assert not np.isfinite(diagnostics.condition_number) or (
        diagnostics.condition_number > 100.0
    )


def test_jacobian_supports_broadcast_array_inputs() -> None:
    q_hip = np.deg2rad(np.array([[20.0], [40.0], [60.0]]))
    q_knee = np.deg2rad(np.array([30.0, 60.0]))
    jacobian = leg_jacobian(q_hip, q_knee, L1, L2)
    diagnostics = jacobian_diagnostics(q_hip, q_knee, L1, L2)

    assert jacobian.shape == (3, 2, 2, 2)
    assert np.asarray(diagnostics.determinant).shape == (3, 2)
    assert np.asarray(diagnostics.condition_number).shape == (3, 2)
    assert np.asarray(diagnostics.near_singular).shape == (3, 2)
