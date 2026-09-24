import numpy as np

from lower_limb_sim.myoleg_benchmark.mechanism_candidates import (
    COORDINATION_AMPLITUDES,
    COORDINATION_LAGS,
    DURATION_SCALES,
    MechanismDomain,
)


def test_mechanism_domain_is_fixed_and_contains_reference():
    domain = MechanismDomain()
    assert len(domain) == len(DURATION_SCALES) * len(COORDINATION_LAGS) * len(COORDINATION_AMPLITUDES)
    assert domain.reference.features == (1.0, 0.0, 0.0)
    assert len({point.candidate_id for point in domain}) == len(domain)
    assert all(np.isfinite(point.features).all() if hasattr(point.features, "all") else all(np.isfinite(point.features)) for point in domain)


def test_duration_scaling_changes_time_and_derivatives_consistently():
    domain = MechanismDomain()
    reference = domain.reference
    slow = next(point for point in domain if point.features == (1.2, 0.0, 0.0))
    assert np.isclose(slow.time_s[-1] - slow.time_s[0], 1.2 * (reference.time_s[-1] - reference.time_s[0]))
    assert np.allclose(slow.trajectory.q, reference.trajectory.q)
    assert np.allclose(slow.trajectory.dq, reference.trajectory.dq / 1.2)
    assert np.allclose(slow.trajectory.ddq, reference.trajectory.ddq / 1.2**2)


def test_coordination_is_smooth_bounded_and_does_not_change_v1_domain():
    domain = MechanismDomain()
    changed = next(point for point in domain if point.features == (1.0, 0.12, 0.04))
    assert np.max(np.abs(changed.trajectory.q[:, 0] - domain.reference.trajectory.q[:, 0])) == 0.0
    assert np.max(np.abs(changed.trajectory.q[:, 1] - domain.reference.trajectory.q[:, 1])) > 0.0
    assert changed.kinematic["hip_speed_ratio"] <= 1.5
    assert changed.kinematic["knee_speed_ratio"] <= 1.5
    assert changed.kinematic["hip_accel_ratio"] <= 2.0
    assert changed.kinematic["knee_accel_ratio"] <= 2.0
