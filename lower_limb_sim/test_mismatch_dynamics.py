"""阶段 4.5C 复杂生成动力学的离线单元测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lower_limb_sim.config import L1
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.full_dynamics import inverse_dynamics, stiffness_torque
from lower_limb_sim.mismatch_dynamics import (
    coupling_potential_energy,
    coupling_stiffness_torque,
    cubic_stiffness_torque,
    mismatch_inverse_dynamics,
    nonlinear_damping_torque,
    nonlinear_stiffness_torque,
    quadratic_damping_torque,
    structured_residual_torque,
)
from lower_limb_sim.mismatch_subject import (
    mismatch_subject_from_dynamic_subject,
)


BASE_SUBJECT = get_dynamic_subject("baseline")
MATCHED_SUBJECT = mismatch_subject_from_dynamic_subject(BASE_SUBJECT)


def _representative_state() -> tuple[np.ndarray, ...]:
    return (
        np.deg2rad(np.array([20.0, 55.0, 100.0, 120.0])),
        np.deg2rad(np.array([20.0, 70.0, 100.0, 120.0])),
        np.array([0.0, 0.35, -0.7, 1.1]),
        np.array([0.0, -0.5, 0.8, -1.2]),
        np.array([0.2, -0.4, 0.9, -1.3]),
        np.array([-0.3, 0.6, -1.0, 1.4]),
    )


def test_zero_additional_terms_exactly_match_existing_inverse_dynamics() -> None:
    state = _representative_state()
    expected = inverse_dynamics(*state, BASE_SUBJECT, L1)
    observed = mismatch_inverse_dynamics(*state, MATCHED_SUBJECT, L1)

    for joint in ("hip", "knee"):
        assert np.array_equal(
            getattr(observed, f"tau_total_{joint}_nm"),
            getattr(expected, f"tau_total_{joint}_nm"),
        )
        assert np.array_equal(
            getattr(observed, f"tau_base_total_{joint}_nm"),
            getattr(expected, f"tau_total_{joint}_nm"),
        )
        assert np.count_nonzero(
            getattr(observed, f"tau_mismatch_{joint}_nm"),
        ) == 0


def test_k3_zero_reduces_to_existing_linear_stiffness() -> None:
    q_hip, q_knee = _representative_state()[:2]
    expected = stiffness_torque(q_hip, q_knee, BASE_SUBJECT)
    observed = nonlinear_stiffness_torque(q_hip, q_knee, MATCHED_SUBJECT)

    assert np.array_equal(observed[0], expected[0])
    assert np.array_equal(observed[1], expected[1])


def test_cubic_stiffness_has_larger_effect_at_large_flexion() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        k3_hip_nm_per_rad3=2.0,
        k3_knee_nm_per_rad3=1.5,
    )
    near = cubic_stiffness_torque(
        subject.q0_hip_rad + 0.20,
        subject.q0_knee_rad + 0.20,
        subject,
    )
    far = cubic_stiffness_torque(
        subject.q0_hip_rad + 1.20,
        subject.q0_knee_rad + 1.20,
        subject,
    )

    assert abs(far[0]) > 100.0 * abs(near[0])
    assert abs(far[1]) > 100.0 * abs(near[1])


def test_coupling_torque_matches_finite_difference_potential_gradient() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        k_coupling_nm_per_rad=5.0,
        k_coupling_asymmetry=0.75,
    )
    q_hip = np.deg2rad(67.0)
    q_knee = np.deg2rad(91.0)
    epsilon = 1e-7
    analytical = coupling_stiffness_torque(q_hip, q_knee, subject)
    numerical_hip = (
        coupling_potential_energy(q_hip + epsilon, q_knee, subject)
        - coupling_potential_energy(q_hip - epsilon, q_knee, subject)
    ) / (2.0 * epsilon)
    numerical_knee = (
        coupling_potential_energy(q_hip, q_knee + epsilon, subject)
        - coupling_potential_energy(q_hip, q_knee - epsilon, subject)
    ) / (2.0 * epsilon)

    assert np.allclose(analytical, [numerical_hip, numerical_knee], atol=1e-8)


def test_nonlinear_damping_is_zero_at_zero_velocity() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        b2_hip_nm_s2_per_rad2=0.5,
        b2_knee_nm_s2_per_rad2=0.4,
    )

    assert nonlinear_damping_torque(0.0, 0.0, subject) == (0.0, 0.0)
    assert quadratic_damping_torque(0.0, 0.0, subject) == (0.0, 0.0)


def test_nonlinear_damping_is_odd_under_velocity_reversal() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        b2_hip_nm_s2_per_rad2=0.5,
        b2_knee_nm_s2_per_rad2=0.4,
    )
    forward = np.asarray(nonlinear_damping_torque(0.7, -1.1, subject))
    reverse = np.asarray(nonlinear_damping_torque(-0.7, 1.1, subject))

    assert np.array_equal(reverse, -forward)


def test_structured_residual_is_deterministic_smooth_and_seeded() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        residual_torque_scale_nm=0.6,
        residual_torque_frequency=1.2,
    )
    state = (0.8, 1.1, 0.3, -0.4)
    first = np.asarray(structured_residual_torque(*state, subject, random_seed=17))
    repeated = np.asarray(
        structured_residual_torque(*state, subject, random_seed=17),
    )
    nearby = np.asarray(
        structured_residual_torque(
            state[0] + 1e-5,
            state[1],
            state[2],
            state[3],
            subject,
            random_seed=17,
        ),
    )
    other_seed = np.asarray(
        structured_residual_torque(*state, subject, random_seed=18),
    )

    assert np.array_equal(first, repeated)
    assert np.linalg.norm(nearby - first) < 1e-4
    assert not np.array_equal(first, other_seed)
    assert np.max(np.abs(first)) <= subject.residual_torque_scale_nm


def test_all_normal_vectorized_results_are_finite() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        k3_hip_nm_per_rad3=4.0,
        k3_knee_nm_per_rad3=3.5,
        k_coupling_nm_per_rad=7.0,
        k_coupling_asymmetry=0.9,
        b2_hip_nm_s2_per_rad2=0.7,
        b2_knee_nm_s2_per_rad2=0.55,
        residual_torque_scale_nm=1.5,
        residual_torque_frequency=1.6,
    )
    result = mismatch_inverse_dynamics(
        *_representative_state(),
        subject,
        L1,
        residual_random_seed=20260802,
    )

    for value in vars(result).values():
        assert np.isfinite(np.asarray(value)).all()


def test_total_torque_equals_sum_of_reported_components() -> None:
    subject = replace(
        MATCHED_SUBJECT,
        k3_hip_nm_per_rad3=1.0,
        k3_knee_nm_per_rad3=0.8,
        k_coupling_nm_per_rad=2.0,
        k_coupling_asymmetry=0.7,
        b2_hip_nm_s2_per_rad2=0.2,
        b2_knee_nm_s2_per_rad2=0.15,
        residual_torque_scale_nm=0.4,
        residual_torque_frequency=1.0,
    )
    result = mismatch_inverse_dynamics(
        *_representative_state(),
        subject,
        L1,
        residual_random_seed=99,
    )
    for joint in ("hip", "knee"):
        expected = sum(
            np.asarray(getattr(result, f"tau_{term}_{joint}_nm"))
            for term in (
                "inertia",
                "coriolis",
                "gravity",
                "damping",
                "stiffness",
                "residual",
            )
        )
        assert np.allclose(
            getattr(result, f"tau_total_{joint}_nm"),
            expected,
            atol=1e-12,
        )


def test_generator_keeps_hip_minus_knee_base_model() -> None:
    # 将 q_knee 取反会把基础模型的小腿绝对角变为 q_hip+q_knee，正好作为
    # 禁止定义的数值对照；复杂生成器必须与原 q_hip-q_knee 结果一致。
    q_hip = np.deg2rad(70.0)
    q_knee = np.deg2rad(40.0)
    observed = mismatch_inverse_dynamics(
        q_hip,
        q_knee,
        0.0,
        0.0,
        0.0,
        0.0,
        MATCHED_SUBJECT,
        L1,
    )
    expected = inverse_dynamics(
        q_hip,
        q_knee,
        0.0,
        0.0,
        0.0,
        0.0,
        BASE_SUBJECT,
        L1,
    )
    forbidden_plus_proxy = inverse_dynamics(
        q_hip,
        -q_knee,
        0.0,
        0.0,
        0.0,
        0.0,
        BASE_SUBJECT,
        L1,
    )

    observed_pair = np.array(
        [observed.tau_total_hip_nm, observed.tau_total_knee_nm],
    )
    expected_pair = np.array(
        [expected.tau_total_hip_nm, expected.tau_total_knee_nm],
    )
    forbidden_pair = np.array(
        [
            forbidden_plus_proxy.tau_total_hip_nm,
            forbidden_plus_proxy.tau_total_knee_nm,
        ],
    )
    assert np.array_equal(observed_pair, expected_pair)
    assert not np.allclose(observed_pair, forbidden_pair)

