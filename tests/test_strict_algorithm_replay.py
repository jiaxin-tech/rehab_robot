"""Probe-role, stop, failure-budget and causal-prefix regression checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_algorithm.strict_replay import replay_attempts
from experiment_algorithm.v2.config import AlgorithmConfig
from experiment_algorithm.v2.policy import Policy


class Surrogate:
    def __init__(self):
        self.diagnostics = dict(success=True, rank=0, condition=1., normalized_train_rmse=0.,
                                scaled_parameter_change=0., latest_prequential=dict(normalized_torque_rmse=0., E3_abs_error=0.))
        self.prefixes = []

    def fit(self, history):
        prefix = tuple(m.observation.candidate_id for m in history)
        if not self.prefixes or self.prefixes[-1] != prefix:
            self.prefixes.append(prefix)

    def predict_all(self):
        return dict(mean=np.array([[1., 1., 1.], [.8, .8, .8], [.7, .7, .7], [.6, .6, .6]]),
                    std=np.full((4, 3), .001))

    def information_scores(self, indices):
        return -np.asarray(indices)


def policy(*, budget=8):
    points = [SimpleNamespace(candidate_id=f"c{i}") for i in range(4)]
    class Domain(list):
        reference = points[0]
    context = SimpleNamespace(domain=Domain(points), common_candidate_id="c0")
    return Policy(context, AlgorithmConfig(total_budget=budget), method="GREEDY", surrogate=Surrogate())


def measurement(attempt, index):
    return SimpleNamespace(observation=SimpleNamespace(candidate_id=attempt["candidate_id"],
        episode_id=attempt["episode_id"], valid=True, endpoint_value=1., endpoint_uncertainty=.001, trial_index=index),
        feasible=True, E2=1., peak_ratio=1.)


def ledger():
    return [dict(attempt_index=i+1, episode_id=f"e{i}", candidate_id=f"c{i}",
                 role="REFERENCE" if i == 0 else "INFORMATION_PROBE", outcome="COMPLETED") for i in range(3)]


def test_prefix_replay_retains_max_probe_count_and_stop():
    controller = policy()
    result = replay_attempts(controller, ledger(), measurement)
    assert result["action"].role == "STOP" and result["action"].stop_reason == "IDENTIFICATION_INSUFFICIENT"
    assert list(result["policy_roles"].values()).count("INFORMATION_PROBE") == 2
    assert controller.surrogate.prefixes == [("c0",), ("c0", "c1"), ("c0", "c1", "c2")]
    extra = dict(attempt_index=4, episode_id="e3", candidate_id="c3", role="INFORMATION_PROBE", outcome="COMPLETED")
    with pytest.raises(ValueError, match="AFTER_POLICY_STOP"):
        replay_attempts(policy(), ledger()+[extra], measurement)


def test_failed_attempt_consumes_budget_and_cannot_be_appended_to():
    attempts = ledger()[:2]
    attempts[1].update(outcome="FAILED", failure_reason="acquisition dropout")
    result = replay_attempts(policy(), attempts, measurement)
    assert result["attempted_trials"] == 2 and result["completed_trials"] == 1
    assert result["failed_trials"] == 1 and result["remaining_trials"] == 6
    assert result["action"].stop_reason == "ATTEMPT_FAILED"
    with pytest.raises(ValueError, match="AFTER_POLICY_STOP"):
        replay_attempts(policy(), attempts+ledger()[2:], measurement)


@pytest.mark.parametrize("mutation", ["candidate", "role", "index", "budget", "failed_data"])
def test_inconsistent_ledger_is_rejected_before_future_data(mutation):
    attempts = ledger()
    controller = policy()
    if mutation == "candidate": attempts[1]["candidate_id"] = "c3"
    if mutation == "role": attempts[1]["role"] = "SEARCH"
    if mutation == "index": attempts[1]["attempt_index"] = 4
    if mutation == "budget": controller = policy(budget=2)
    if mutation == "failed_data": attempts[1].update(outcome="FAILED", failure_reason="bad", measured_path="not-allowed.csv")
    with pytest.raises(ValueError):
        replay_attempts(controller, attempts, measurement)
    assert not controller.surrogate.prefixes or max(map(len, controller.surrogate.prefixes)) == 1
