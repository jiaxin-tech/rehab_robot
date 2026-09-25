from __future__ import annotations

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark.physical_interaction import (
    PhysicalInteractionConfig,
    PhysicalInteractionProfile,
    compute_physical_resistance,
    endpoint_force_from_interaction,
    net_required_torque,
)


def _trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 1.0, 21)
    q = np.column_stack((0.35 * phase, 0.75 - 0.25 * phase))
    dq = np.gradient(q, phase, axis=0)
    return q, dq, phase


def test_interaction_has_explicit_nm_components_and_duration_effect() -> None:
    q, dq, phase = _trajectory()
    profile = PhysicalInteractionProfile(
        "P1", stiffness_nm_per_rad=(2.0, 3.0), damping_nm_s_per_rad=(0.4, 0.5),
        reference_q_rad=(0.0, 0.0),
    )
    fast = compute_physical_resistance(q, dq, phase, (0.9, 0.5, 0.5), profile)
    slow = compute_physical_resistance(q, dq, phase, (1.1, 0.5, 0.5), profile)
    assert fast.valid and slow.valid
    assert fast.external_resistance_tau_nm.shape == (21, 2)
    assert np.max(np.abs(fast.external_resistance_tau_nm)) > 0.0
    assert not np.allclose(fast.external_resistance_tau_nm, slow.external_resistance_tau_nm)
    assert np.all(fast.onset_gate >= 0.0) and np.all(fast.onset_gate <= 1.0)


def test_net_torque_and_jacobian_force_mapping_are_separate_from_native() -> None:
    q, dq, phase = _trajectory()
    profile = PhysicalInteractionProfile(
        "P2", stiffness_nm_per_rad=(1.0, 1.0), damping_nm_s_per_rad=(0.1, 0.1),
    )
    result = compute_physical_resistance(q, dq, phase, (1.0, 0.5, 0.75), profile)
    native = np.full((21, 2), 4.0)
    net = net_required_torque(native, result)
    assert np.allclose(net, native - result.external_resistance_tau_nm)
    mapped = endpoint_force_from_interaction(q, result, L1_m=0.42, L2_m=0.30)
    assert mapped["fx_n"].shape == (21,)
    assert mapped["fz_n"].shape == (21,)
    assert mapped["valid"].shape == (21,)
    assert np.all(mapped["valid"])


def test_interaction_fails_closed_on_torque_limit_and_bad_candidate() -> None:
    q, dq, phase = _trajectory()
    profile = PhysicalInteractionProfile(
        "P3", stiffness_nm_per_rad=(50.0, 50.0), damping_nm_s_per_rad=(10.0, 10.0),
    )
    limited = compute_physical_resistance(
        q, dq, phase, (1.0, 0.5, 0.5), profile,
        PhysicalInteractionConfig(max_resistance_torque_nm=0.01),
    )
    assert not limited.valid
    with pytest.raises(ValueError, match="INVALID_PHYSICAL_INTERACTION_RESULT"):
        endpoint_force_from_interaction(q, limited, L1_m=0.42, L2_m=0.30)
    with pytest.raises(ValueError, match="OUT_OF_RANGE"):
        compute_physical_resistance(q, dq, phase, (1.0, 1.5, 0.5), profile)
