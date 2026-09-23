"""Behavioral checks for the MyoLeg harness without model assets or truth tables."""

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.myoleg_benchmark import experiment as benchmark
from lower_limb_sim.myoleg_benchmark import run as runner
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain


class FakeBackend:
    """Only the requested candidate can reveal its synthetic torque trace."""

    def __init__(self, factors=None, invalid=None):
        self.factors = factors or {}
        self.invalid = invalid
        self.calls = []

    def requested(self, point):
        self.calls.append(point.candidate_id)
        if self.invalid == "exception":
            raise RuntimeError("synthetic replay failure")
        if self.invalid == "malformed":
            return np.ones((len(point.time_s), 1))
        if self.invalid == "nonfinite":
            return np.full((len(point.time_s), 2), np.nan)
        factor = self.factors.get(
            point.candidate_id,
            1.0 - .04 * point.beta_flex / .03
            + .02 * point.beta_extend / .03 + .03 * point.share_shift / .05,
        )
        return np.tile(np.array([10.0, 5.0]) * factor, (len(point.time_s), 1))


@pytest.fixture(scope="module")
def source():
    # This reads only the native reference/ROM, never cohort or held-out truth.
    return native_domain()


@pytest.fixture(scope="module")
def small_domain(source):
    return benchmark.Domain(source, "BETA_TIMING", grid=(0., .03), shifts=(0., .05))


def nonreference_points(domain):
    return [p for p in domain if p.candidate_id != domain.reference.candidate_id]


@pytest.mark.parametrize("family", benchmark.FAMILIES)
def test_domain_uses_only_kinematics_and_rejects_excess_speed(source, family, monkeypatch):
    def no_outcome_read(*args, **kwargs):
        raise AssertionError("candidate construction attempted to read outcomes")

    monkeypatch.setattr(pd, "read_csv", no_outcome_read)
    monkeypatch.setattr(np, "load", no_outcome_read)
    domain = benchmark.Domain(source, family, grid=(0.,), shifts=(-.4, 0., .4))
    assert domain.reference.beta == (0., 0.)
    assert domain.reference.share_shift == 0.
    assert len(domain.rejections) > 0
    assert all(r["reason"] == "KINEMATIC_COMPARISON_LIMIT" for r in domain.rejections)
    for point in domain:
        assert np.isfinite(point.trajectory.q).all()
        assert np.all(np.diff(point.time_s) > 0)


def test_space_filling_uses_time_share_as_third_coordinate(small_domain):
    env = benchmark.Environment(small_domain, FakeBackend(), subject_id="synthetic")
    reference = env.evaluate(small_domain.reference, 1)
    selected, _ = benchmark.Learner(small_domain, "SPACE_FILLING", 0).select([reference])
    # Dropping the third coordinate would select (.03, .03, 0) on this ordered grid.
    assert (*selected.beta, selected.share_shift) == (.03, .03, .05)


def test_gp_explicit_noise_changes_posterior_at_observed_point():
    quiet = benchmark.GaussianProcess3D()
    noisy = benchmark.GaussianProcess3D()
    quiet.fit([[0., 0., 0.]], [1.], [.001])
    noisy.fit([[0., 0., 0.]], [1.], [.5])
    quiet_mean, quiet_std = quiet.predict([[0., 0., 0.]])
    noisy_mean, noisy_std = noisy.predict([[0., 0., 0.]])
    variance = noisy.kernel.signal_std ** 2
    expected_mean = variance / (variance + .5 ** 2 + noisy.kernel.jitter)
    assert noisy_mean[0] == pytest.approx(expected_mean)
    assert 0 < noisy_mean[0] < quiet_mean[0] < 1
    assert noisy_std[0] > quiet_std[0]


@pytest.mark.parametrize("noise", [[-.1], [np.nan], []])
def test_gp_rejects_invalid_noise(noise):
    with pytest.raises(ValueError):
        benchmark.GaussianProcess3D().fit([[0., 0., 0.]], [1.], noise)


def test_valid_infeasible_response_keeps_endpoint_and_identification_payload(small_domain):
    point = nonreference_points(small_domain)[0]
    backend = FakeBackend({point.candidate_id: 1.2})
    env = benchmark.Environment(
        small_domain, backend, subject_id="synthetic", tier=.01, physics=True,
    )
    reference = env.evaluate(small_domain.reference, 1)
    excess = env.evaluate(point, 2)
    assert reference.feasible
    assert not excess.feasible
    assert excess.observation.valid
    assert excess.observation.invalid_reason is None
    assert excess.observation.endpoint_value == pytest.approx(1.2)
    payload = excess.observation.identification_payload
    assert payload is not None and payload.valid
    assert all(payload.sample_valid)
    jacobian_t = leg_jacobian(point.trajectory.q[:, 0], point.trajectory.q[:, 1], .42, .30).swapaxes(-1, -2)
    reconstructed = np.einsum("nij,nj->ni", jacobian_t, np.asarray(payload.planar_force_n))
    np.testing.assert_allclose(reconstructed, np.tile([12., 6.], (len(point.time_s), 1)))
    assert benchmark.recommend([reference, excess]) == small_domain.reference.candidate_id
    # An observed constraint violation remains useful objective data.
    learner = benchmark.Learner(small_domain, "PURE_BO_EI", 0)
    learner.select([reference, excess])
    assert len(learner.gp.x) == 2
    np.testing.assert_array_equal(learner.gp.x[1], point.features)
    assert env.calls == backend.calls == [small_domain.reference.candidate_id, point.candidate_id]


def test_paired_noise_is_independent_of_request_order(small_domain):
    first, second = nonreference_points(small_domain)[:2]

    def observe(order, seed):
        env = benchmark.Environment(
            small_domain, FakeBackend(), subject_id="synthetic", noise_std=.02,
            seed=seed, physics=True,
        )
        values = {}
        for index, point in enumerate([small_domain.reference, *order], 1):
            measurement = env.evaluate(point, index)
            assert measurement.observation.valid
            values[point.candidate_id] = measurement
        return values

    forward = observe([first, second], 12)
    reverse = observe([second, first], 12)
    another_seed = observe([first, second], 13)
    for candidate_id in forward:
        a, b = forward[candidate_id], reverse[candidate_id]
        assert a.observation.endpoint_value == b.observation.endpoint_value
        assert a.observation.endpoint_uncertainty > 0
        assert a.feasible == b.feasible
        np.testing.assert_array_equal(
            a.observation.identification_payload.planar_force_n,
            b.observation.identification_payload.planar_force_n,
        )
    assert not np.array_equal(
        forward[first.candidate_id].observation.identification_payload.planar_force_n,
        another_seed[first.candidate_id].observation.identification_payload.planar_force_n,
    )


class FakePhysicsAdapter:
    """Stable causal predictions; parameter estimation is tested elsewhere."""

    def __init__(self, domain, **kwargs):
        self.domain = domain
        self.history = []

    def fit(self, history):
        self.history = [o for o in history if o.valid]
        assert all(o.identification_payload is not None for o in self.history)

    def predict_value(self, point):
        base = np.mean([o.endpoint_value for o in self.history])
        return float(base - .02 * point.beta_flex / .03 + .01 * point.share_shift / .05)

    def metadata(self):
        return {"fit_candidate_ids": [o.candidate_id for o in self.history]}


@pytest.mark.parametrize("method", [m for m in benchmark.METHODS if m != "REFERENCE"])
def test_budget_prefix_unique_candidates_and_causal_updates(small_domain, monkeypatch, method):
    monkeypatch.setattr(benchmark, "Adapter", FakePhysicsAdapter)
    short_backend, long_backend = FakeBackend(), FakeBackend()
    kwargs = dict(subject_id="synthetic", method=method, noise_std=.01, seed=17)
    short = benchmark.run_sequence(small_domain, short_backend, budget=2, **kwargs)
    long = benchmark.run_sequence(small_domain, long_backend, budget=4, **kwargs)
    assert short["failure"] is long["failure"] is None
    assert short["rows"] == long["rows"][:2]
    assert len(long_backend.calls) == len(set(long_backend.calls)) == 4
    assert long_backend.calls[0] == small_domain.reference.candidate_id
    assert long_backend.calls == [row["candidate_id"] for row in long["rows"]]
    for diagnostic in long["diagnostics"]:
        if "fit_candidate_ids" in diagnostic:
            assert diagnostic["fit_candidate_ids"] == long_backend.calls[:diagnostic["after_trial"]]


def test_noisy_recommendation_uses_observations_not_latent_best(source, monkeypatch):
    domain = benchmark.Domain(source, "BETA_TIMING", grid=(0.,), shifts=(-.05, 0., .05))
    lower, higher = sorted(nonreference_points(domain), key=lambda p: p.share_shift)
    backend = FakeBackend({lower.candidate_id: .8, higher.candidate_id: .9})
    observed_factors = {domain.reference.candidate_id: 1., lower.candidate_id: .98, higher.candidate_id: .85}

    def noise_that_reverses_ranking(tau, *, candidate_id, **kwargs):
        return np.tile(np.array([10., 5.]) * observed_factors[candidate_id], (len(tau), 1))

    monkeypatch.setattr(benchmark, "paired_noise", noise_that_reverses_ranking)
    result = benchmark.run_sequence(
        domain, backend, subject_id="synthetic", method="SPACE_FILLING", budget=3,
        noise_std=.02,
    )
    assert result["failure"] is None
    assert result["rows"][-1]["recommendation_id"] == higher.candidate_id
    assert backend.factors[lower.candidate_id] < backend.factors[higher.candidate_id]
    assert all(row["recommendation_id"] in backend.calls[:row["trial"]] for row in result["rows"])
    evaluated = runner._evaluate_results(
        result, domain, backend, {"method": "SPACE_FILLING"}, [3], .01,
    )[0]
    # Score the recommendation, not the latent best among all executed points.
    assert evaluated["true_E3"] == pytest.approx(.9)
    assert evaluated["selection_loss"] == pytest.approx(.9)
    assert evaluated["improvement_pct"] == pytest.approx(10.)


@pytest.mark.parametrize("invalid", ["exception", "nonfinite", "malformed"])
def test_failed_reference_stops_without_fabricated_incumbent(small_domain, invalid):
    backend = FakeBackend(invalid=invalid)
    result = benchmark.run_sequence(
        small_domain, backend, subject_id="synthetic", method="PURE_BO_EI", budget=4,
    )
    assert result["failure"] == "REFERENCE_MEASUREMENT_FAILED"
    assert len(result["rows"]) == len(backend.calls) == 1
    row = result["rows"][0]
    assert not row["measurement_valid"] and not row["observed_feasible"]
    assert row["observed_E3"] is None and row["recommendation_id"] is None
    assert row["invalid_reason"]
    assert result["diagnostics"] == []


def test_duplicate_request_is_rejected_before_backend_call(small_domain):
    backend = FakeBackend()
    env = benchmark.Environment(small_domain, backend, subject_id="synthetic")
    env.evaluate(small_domain.reference, 1)
    with pytest.raises(ValueError, match="DUPLICATE_TRIAL"):
        env.evaluate(small_domain.reference, 2)
    assert backend.calls == [small_domain.reference.candidate_id]


@pytest.mark.parametrize("method", ["REFERENCE", "PURE_BO_EI"])
@pytest.mark.parametrize("invalid", ["exception", "nonfinite", "malformed"])
def test_runner_scores_reference_failure_without_retrying_truth(small_domain, method, invalid):
    backend = FakeBackend(invalid=invalid)
    result = benchmark.run_sequence(
        small_domain, backend, subject_id="synthetic", method=method, budget=4,
    )
    scored = runner._evaluate_results(result, small_domain, backend, {"method": method}, [1, 2, 4], .01)
    assert backend.calls == [small_domain.reference.candidate_id]
    for row in scored:
        assert row["status"] == "REFERENCE_MEASUREMENT_FAILED"
        assert row["selection_loss"] == 1.
        assert row["improvement_pct"] == 0.
        assert row["recommendation_id"] is None
        assert row["executed_trials"] == row["invalid_trials"] == 1
        assert not row["true_feasible"]
        assert row["true_E3"] is None or np.isnan(row["true_E3"])


def test_runner_does_not_reward_noisy_false_feasibility(small_domain, monkeypatch):
    point = nonreference_points(small_domain)[0]

    class UnequalLoads(FakeBackend):
        def requested(self, candidate):
            response = super().requested(candidate)
            if candidate.candidate_id == point.candidate_id:
                response[:] = [10.5, 3.25]  # Mean ratio .85, but hip ratio 1.05.
            return response

    def optimistic_measurement(tau, *, candidate_id, **kwargs):
        factor = .8 if candidate_id == point.candidate_id else 1.
        return np.tile(np.array([10., 5.]) * factor, (len(tau), 1))

    monkeypatch.setattr(benchmark, "paired_noise", optimistic_measurement)
    monkeypatch.setattr(benchmark.Learner, "select", lambda self, history: (point, {}))
    backend = UnequalLoads()
    result = benchmark.run_sequence(
        small_domain, backend, subject_id="synthetic", method="SPACE_FILLING",
        budget=2, noise_std=.03,
    )
    assert result["rows"][-1]["recommendation_id"] == point.candidate_id
    assert result["rows"][-1]["observed_feasible"]
    scored = runner._evaluate_results(result, small_domain, backend, {"method": "SPACE_FILLING"}, [2], .01)[0]
    assert scored["status"] == "complete"
    assert scored["true_E3"] == pytest.approx(.85)
    assert scored["true_E2"] == pytest.approx(1.05)
    assert not scored["true_feasible"]
    assert scored["selection_loss"] == 1.
    assert scored["improvement_pct"] == 0.


def test_runner_preserves_completed_prefix_when_later_selection_fails(small_domain, monkeypatch):
    point = nonreference_points(small_domain)[0]

    def fail_after_two(self, history):
        if len(history) == 1:
            return point, {}
        raise RuntimeError("synthetic selection failure")

    monkeypatch.setattr(benchmark.Learner, "select", fail_after_two)
    backend = FakeBackend({point.candidate_id: .9})
    result = benchmark.run_sequence(
        small_domain, backend, subject_id="synthetic", method="PURE_BO_EI", budget=4,
    )
    scored = runner._evaluate_results(result, small_domain, backend, {"method": "PURE_BO_EI"}, [1, 2, 4], .01)
    assert [row["executed_trials"] for row in scored] == [1, 2, 2]
    assert scored[0]["status"] == scored[1]["status"] == "complete"
    assert scored[1]["selection_loss"] == pytest.approx(.9)
    assert scored[2]["status"].startswith("SELECTION_FAILED")
    assert scored[2]["selection_loss"] == 1.
    assert not scored[2]["true_feasible"]
