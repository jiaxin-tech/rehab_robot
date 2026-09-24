from types import SimpleNamespace

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark.controlled_actuation import (
    ControlledActuationBackend,
    ControlledActuationConfig,
    ControlledActuationDomain,
    assistance_waveform,
)


@pytest.fixture(scope="module")
def domain():
    return ControlledActuationDomain()


def test_frozen_domain_preserves_shape_rom_and_duration_derivatives(domain):
    assert len(domain) == 27
    assert len({point.features for point in domain}) == 27
    assert len({point.candidate_id for point in domain}) == 27
    assert domain.reference.features == (1.0, 0.5, 0.5)
    reference = domain.subject_reference
    for point in domain:
        np.testing.assert_array_equal(point.trajectory.q, reference.q)
        np.testing.assert_allclose(point.trajectory.dq, reference.dq / point.duration_scale)
        np.testing.assert_allclose(point.trajectory.ddq, reference.ddq / point.duration_scale**2)
        np.testing.assert_allclose(np.diff(point.time_s), np.diff(reference.time_s) * point.duration_scale)
        assert point.kinematic["rom_excess_rad"] == 0


def test_fixed_peak_allocation_and_smooth_branch_boundaries(domain):
    config = ControlledActuationConfig()
    phases = domain.subject_reference.phases
    for point in domain:
        wave = assistance_waveform(point, phases, config)
        tau = wave["assistance_tau_nm"]
        np.testing.assert_allclose(np.max(np.abs(tau), axis=0),
                                   config.assist_peak_nm * np.array([point.hip_share, 1 - point.hip_share]))
        assert np.max(np.sum(np.abs(tau), axis=1)) == pytest.approx(config.assist_peak_nm)
        for branch, sign in (("flexion", 1), ("extension", -1)):
            indices = np.flatnonzero(phases == branch)
            assert np.all(sign * tau[indices] >= 0)
            np.testing.assert_array_equal(tau[indices[[0, 1, -2, -1]]], 0)
            peak_index = indices[np.argmax(wave["pulse"][indices])]
            assert wave["branch_phase"][peak_index] == pytest.approx(point.assistance_timing, abs=.003)


def test_duration_changes_impulse_but_not_peak_and_share_preserves_l1_impulse(domain):
    config = ControlledActuationConfig()
    phases = domain.subject_reference.phases
    integrals = {}
    for point in domain:
        wave = assistance_waveform(point, phases, config)
        integrals[point.features] = np.trapezoid(np.sum(np.abs(wave["assistance_tau_nm"]), axis=1), point.time_s)
    for timing in (.25, .5, .75):
        for duration in (.9, 1., 1.1):
            for share in (.25, .5, .75):
                assert integrals[(duration, timing, share)] == pytest.approx(duration * integrals[(1., timing, .5)])


def test_requested_only_adapter_equation_no_truth_dependent_waveform_and_copy_isolation(domain):
    class Native:
        def __init__(self, scale):
            self.scale = scale
            self.calls = []

        def requested(self, point):
            self.calls.append(point.candidate_id)
            return np.full((len(point.time_s), 2), self.scale)

    point = domain.reference
    native = Native(3.)
    backend = ControlledActuationBackend(native, domain)
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(point)
    first = backend.requested(point)
    trace = backend.trace_for(point)
    np.testing.assert_allclose(first, trace["native_tau_nm"] - trace["assistance_tau_nm"])
    assert trace["diagnostics"]["actuation_valid"]
    assert trace["diagnostics"]["constraint_violations"] == []
    assert len(native.calls) == 1
    first[:] = np.nan
    trace["net_tau_nm"][:] = np.nan
    assert np.isfinite(backend.requested(point)).all()
    assert len(native.calls) == 1
    other = ControlledActuationBackend(Native(-17.), domain)
    other.requested(point)
    np.testing.assert_array_equal(backend.trace_for(point)["assistance_tau_nm"],
                                  other.trace_for(point)["assistance_tau_nm"])


def test_invalid_configuration_and_native_response_fail_closed(domain):
    for peak in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ASSISTANCE_PEAK"):
            ControlledActuationConfig(assist_peak_nm=peak)
    for width in (0., -.1, .26, float("nan")):
        with pytest.raises(ValueError, match="PULSE_WIDTH"):
            ControlledActuationConfig(pulse_half_width_phase=width)
    malformed = SimpleNamespace(requested=lambda point: np.full((len(point.time_s), 2), np.nan))
    backend = ControlledActuationBackend(malformed, domain)
    with pytest.raises(ValueError, match="INVALID_NATIVE"):
        backend.requested(domain.reference)
    with pytest.raises(ValueError, match="TRACE_NOT_REQUESTED"):
        backend.trace_for(domain.reference)
