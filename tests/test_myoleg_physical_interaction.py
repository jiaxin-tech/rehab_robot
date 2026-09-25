from __future__ import annotations

import numpy as np
import pytest

from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.myoleg_benchmark.physical_interaction import (
    PhysicalInteractionConfig, PhysicalInteractionProfile, compute_physical_resistance,
    endpoint_force_from_interaction, net_required_torque,
)


def test_spring_restores_position_in_both_motion_directions_and_at_rest():
    profile = PhysicalInteractionProfile("SPRING", (2.0, 4.0), (0.0, 0.0), (0.1, 0.2))
    q = np.array([[0.4, 0.5]] * 3)
    dq = np.array([[0.5, 0.3], [-0.5, -0.3], [0.0, 0.0]])
    result = compute_physical_resistance(q, dq, profile)
    np.testing.assert_allclose(result.external_resistance_tau_nm, [[-0.6, -1.2]] * 3)
    assert np.all(result.elastic_component_nm[1] * dq[1] > 0.0)
    np.testing.assert_allclose(result.potential_energy_j, [[0.09, 0.18]] * 3)
    equilibrium = compute_physical_resistance(np.array([[0.1, 0.2]]), np.zeros((1, 2)), profile)
    np.testing.assert_array_equal(equilibrium.external_resistance_tau_nm, 0.0)


def test_damping_is_passive_and_duration_enters_once_through_executed_velocity():
    profile = PhysicalInteractionProfile("DAMPER", (0.0, 0.0), (2.0, 3.0))
    q = np.array([[0.4, 0.6]] * 3)
    dq = np.array([[0.3, -0.2], [0.0, 0.0], [-0.4, 0.5]])
    first = compute_physical_resistance(q, dq, profile)
    slow = compute_physical_resistance(q, dq / 2.0, profile)
    np.testing.assert_allclose(slow.external_resistance_tau_nm, first.external_resistance_tau_nm / 2.0)
    assert np.all(first.viscous_component_nm * dq <= 0.0)
    np.testing.assert_allclose(first.viscous_component_nm, -dq * [2.0, 3.0])


def test_closed_cycle_spring_conserves_energy_and_damper_dissipates_it():
    time_s = np.linspace(0.0, 2.0 * np.pi, 2001)
    q = np.column_stack((0.2 + 0.2 * np.sin(time_s), 0.6 + 0.1 * np.cos(time_s)))
    dq = np.column_stack((0.2 * np.cos(time_s), -0.1 * np.sin(time_s)))
    profile = PhysicalInteractionProfile("PASSIVE", (3.0, 4.0), (0.5, 0.7), (0.1, 0.3), 1.5)
    result = compute_physical_resistance(q, dq, profile)
    spring_work = np.trapezoid(result.elastic_component_nm * dq, time_s, axis=0)
    total_work = np.trapezoid(result.external_resistance_tau_nm * dq, time_s, axis=0)
    dissipated = np.trapezoid(1.5 * np.array([0.5, 0.7]) * dq**2, time_s, axis=0)
    np.testing.assert_allclose(spring_work, 0.0, atol=1e-12)
    np.testing.assert_allclose(total_work, -dissipated, atol=1e-12)
    np.testing.assert_allclose(result.potential_energy_j[-1], result.potential_energy_j[0], atol=1e-12)


def test_open_spring_path_work_matches_potential_energy_change():
    time_s = np.linspace(0.0, 2.0, 101)
    q = np.column_stack((0.2 + 0.3 * time_s, 0.7 - 0.1 * time_s))
    dq = np.broadcast_to([0.3, -0.1], q.shape)
    result = compute_physical_resistance(q, dq, PhysicalInteractionProfile("SPRING", (2.0, 3.0), (0.0, 0.0)))
    work = np.trapezoid(result.external_resistance_tau_nm * dq, time_s, axis=0)
    np.testing.assert_allclose(work + result.potential_energy_j[-1] - result.potential_energy_j[0], 0.0, atol=1e-12)


def test_force_balance_counts_native_once_and_subtracts_both_external_torques():
    profile = PhysicalInteractionProfile("SIGNS", (2.0, 0.0), (0.0, 2.0))
    interaction = compute_physical_resistance(np.array([[0.5, 0.5]]), np.array([[0.0, -0.5]]), profile)
    native, assist = np.array([[4.0, -4.0]]), np.array([[0.5, -0.5]])
    np.testing.assert_allclose(interaction.external_resistance_tau_nm, [[-1.0, 1.0]])
    np.testing.assert_allclose(net_required_torque(native, interaction), [[5.0, -5.0]])
    np.testing.assert_allclose(net_required_torque(native, interaction, assist), [[4.5, -4.5]])


def test_jacobian_mapping_preserves_torque_and_virtual_work():
    q = np.array([[0.3, 0.7], [0.5, 0.8]])
    dq = np.array([[0.2, -0.1], [-0.3, 0.2]])
    result = compute_physical_resistance(q, dq, PhysicalInteractionProfile("FORCE", (1.0, 1.0), (0.1, 0.2)))
    mapped = endpoint_force_from_interaction(q, result, L1_m=0.42, L2_m=0.30)
    assert np.all(mapped["valid"])
    force = np.column_stack((mapped["fx_n"], mapped["fz_n"]))
    jacobian = leg_jacobian(q[:, 0], q[:, 1], 0.42, 0.30)
    np.testing.assert_allclose(np.einsum("nji,nj->ni", jacobian, force), result.external_resistance_tau_nm, atol=1e-12)
    endpoint_velocity = np.einsum("nij,nj->ni", jacobian, dq)
    np.testing.assert_allclose(np.sum(endpoint_velocity * force, axis=1),
                               np.sum(result.external_resistance_tau_nm * dq, axis=1), atol=1e-12)


def test_singular_force_mapping_is_invalid_without_clipping_or_inventing_force():
    q = np.array([[0.3, 0.0], [0.3, 0.7]])
    result = compute_physical_resistance(q, np.zeros_like(q), PhysicalInteractionProfile("FORCE", (1.0, 1.0), (0.0, 0.0)))
    mapped = endpoint_force_from_interaction(q, result, L1_m=0.42, L2_m=0.30)
    assert not mapped["valid"][0] and mapped["valid"][1]
    assert np.isnan(mapped["magnitude_n"][0])
    assert "singular" in mapped["invalid_reason"][0]
    limited = endpoint_force_from_interaction(q, result, L1_m=0.42, L2_m=0.30, force_limit_n=0.001)
    assert not np.any(limited["valid"])


def test_interaction_limit_fails_closed_without_silent_clipping():
    q = np.array([[0.3, 0.7]])
    result = compute_physical_resistance(q, np.zeros_like(q),
        PhysicalInteractionProfile("LIMIT", (50.0, 50.0), (0.0, 0.0)),
        PhysicalInteractionConfig(max_resistance_torque_nm=0.01))
    assert not result.valid
    assert np.max(np.abs(result.external_resistance_tau_nm)) > 0.01
    with pytest.raises(ValueError, match="INVALID_PHYSICAL_INTERACTION_RESULT"):
        net_required_torque(np.ones_like(q), result)
    with pytest.raises(ValueError, match="INVALID_PHYSICAL_INTERACTION_RESULT"):
        endpoint_force_from_interaction(q, result, L1_m=0.42, L2_m=0.30)


@pytest.mark.parametrize("q,dq", [
    (np.empty((0, 2)), np.empty((0, 2))), (np.zeros((2, 2)), np.zeros((2, 1))),
    (np.zeros((2, 3)), np.zeros((2, 3))), (np.array([[np.nan, 0.0]]), np.zeros((1, 2))),
    (np.zeros((1, 2)), np.array([[0.0, np.inf]])),
])
def test_malformed_or_nonfinite_trajectory_is_rejected(q, dq):
    with pytest.raises(ValueError):
        compute_physical_resistance(q, dq, PhysicalInteractionProfile("BAD", (1.0, 1.0), (1.0, 1.0)))


def test_nonfinite_or_wrong_shape_torque_inputs_are_rejected():
    q = np.array([[0.3, 0.7]])
    result = compute_physical_resistance(q, np.zeros_like(q), PhysicalInteractionProfile("VALID", (1.0, 1.0), (0.0, 0.0)))
    for native in (np.array([[np.nan, 1.0]]), np.ones((1, 1))):
        with pytest.raises(ValueError, match="NATIVE_TORQUE"):
            net_required_torque(native, result)
    for assistance in (np.array([[np.nan, 1.0]]), np.ones((1, 1))):
        with pytest.raises(ValueError, match="ASSISTANCE_TORQUE"):
            net_required_torque(np.ones_like(q), result, assistance)
