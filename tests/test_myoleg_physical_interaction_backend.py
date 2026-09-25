from copy import deepcopy

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark.controlled_actuation import ControlledActuationBackend, ControlledActuationDomain
from lower_limb_sim.myoleg_benchmark.physical_interaction import PhysicalInteractionConfig, PhysicalInteractionProfile
from lower_limb_sim.myoleg_benchmark.physical_interaction_backend import PhysicalInteractionBackend


class Native:
    provenance = {"source": "test double with prescribed torque"}

    def __init__(self, scale=3.0):
        self.scale = scale
        self.calls = []

    def requested(self, point):
        self.calls.append(point.candidate_id)
        return np.full((len(point.time_s), 2), self.scale)


@pytest.fixture(scope="module")
def domain():
    return ControlledActuationDomain()


def test_zero_profile_reproduces_existing_v3_controlled_actuation(domain):
    null = PhysicalInteractionProfile("NULL", (0.0, 0.0), (0.0, 0.0))
    physical = PhysicalInteractionBackend(Native(), domain, null)
    previous = ControlledActuationBackend(Native(), domain)
    for point in domain:
        np.testing.assert_array_equal(physical.requested(point), previous.requested(point))
        np.testing.assert_array_equal(physical.trace_for(point)["interaction_tau_nm"], 0.0)


def test_assistance_timing_and_share_do_not_change_subject_mechanics(domain):
    profile = PhysicalInteractionProfile("FIXED_SUBJECT", (1.0, 1.5), (0.2, 0.3), (0.1, 0.2))
    backend = PhysicalInteractionBackend(Native(), domain, profile)
    traces = []
    for point in domain:
        if point.duration_scale == 1.0:
            backend.requested(point)
            traces.append(backend.trace_for(point))
    assert len(traces) == 9
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace["interaction_tau_nm"], traces[0]["interaction_tau_nm"])
    assert any(not np.array_equal(trace["assistance_tau_nm"], traces[0]["assistance_tau_nm"])
               for trace in traces[1:])


def test_real_domain_duration_changes_damping_once_and_not_spring(domain):
    backend = PhysicalInteractionBackend(Native(), domain, PhysicalInteractionProfile("DURATION", (1.0, 1.5), (0.2, 0.3)))
    backend.requested(domain.reference)
    reference = backend.trace_for(domain.reference)
    for point in domain:
        if point.assistance_timing == 0.5 and point.hip_share == 0.5:
            backend.requested(point)
            trace = backend.trace_for(point)
            np.testing.assert_array_equal(trace["elastic_tau_nm"], reference["elastic_tau_nm"])
            np.testing.assert_allclose(trace["viscous_tau_nm"] * point.duration_scale,
                                       reference["viscous_tau_nm"], atol=1e-14)


def test_requested_only_trace_and_reference_normalization_with_same_mechanics(domain):
    native = Native()
    backend = PhysicalInteractionBackend(native, domain, PhysicalInteractionProfile("ACCESS", (1.0, 1.5), (0.2, 0.3)))
    point = domain.reference
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(point)
    assert native.calls == []
    baseline = backend.unassisted_reference()
    assert native.calls == [point.candidate_id]
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(point)
    response = backend.requested(point)
    trace = backend.trace_for(point)
    np.testing.assert_allclose(baseline, trace["native_tau_nm"] - trace["interaction_tau_nm"])
    np.testing.assert_allclose(response + trace["assistance_tau_nm"], baseline)
    np.testing.assert_allclose(response + trace["interaction_tau_nm"] + trace["assistance_tau_nm"],
                               trace["native_tau_nm"], atol=1e-12)
    assert trace["diagnostics"]["force_balance_max_abs_error_nm"] < 1e-12
    response[:] = np.nan
    trace["net_tau_nm"][:] = np.nan
    assert np.isfinite(backend.requested(point)).all()
    assert len(native.calls) == 2
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(domain.points[1])


def test_rejects_forged_candidate_identity(domain):
    backend = PhysicalInteractionBackend(Native(), domain, PhysicalInteractionProfile("IDENTITY", (1.0, 1.0), (0.0, 0.0)))
    forged = deepcopy(domain.reference)
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        backend.requested(forged)
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        backend.trace_for(forged)


def test_failed_interaction_is_not_cached_or_passed_to_native_backend(domain):
    native = Native()
    backend = PhysicalInteractionBackend(native, domain,
        PhysicalInteractionProfile("LIMIT", (50.0, 50.0), (0.0, 0.0)),
        PhysicalInteractionConfig(max_resistance_torque_nm=0.01))
    with pytest.raises(ValueError, match="TORQUE_LIMIT"):
        backend.requested(domain.reference)
    with pytest.raises(ValueError, match="TORQUE_LIMIT"):
        backend.unassisted_reference()
    assert native.calls == []
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(domain.reference)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_native_input_is_not_cached(domain, value):
    backend = PhysicalInteractionBackend(Native(value), domain, PhysicalInteractionProfile("FINITE", (1.0, 1.0), (0.0, 0.0)))
    with pytest.raises(ValueError, match="NATIVE_TORQUE"):
        backend.requested(domain.reference)
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(domain.reference)
