"""Small, observation-only tests for the SAST/EG-CPI-BO prototype."""

from dataclasses import replace

import numpy as np

from lower_limb_sim.myoleg_benchmark.personalized_v2 import (
    CandidateView,
    GateConfig,
    ObservationRecord,
    SASTBO,
    run_sequence,
)


def _candidates():
    safe = {"E2": (0.96, 0.005), "peak_ratio": (0.96, 0.005)}
    return [
        CandidateView("reference", (0.0, 0.0), safe),
        CandidateView("probe_a", (1.0, 0.0), safe),
        CandidateView("probe_b", (0.0, 1.0), safe),
        CandidateView("common", (0.5, 0.5), safe),
        CandidateView("candidate_c", (1.0, 1.0), safe),
    ]


def _obs(candidate, trial, value, uncertainty=0.001):
    return ObservationRecord(
        candidate_id=candidate.candidate_id,
        features=tuple(candidate.features),
        value=float(value),
        uncertainty=uncertainty,
        feasible=True,
        constraint_values={"E2": 0.96, "peak_ratio": 0.96},
        trial_index=trial,
    )


def test_null_cohort_keeps_personalization_inactive_and_uses_common_policy():
    candidates = _candidates()
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        common_predictor=lambda candidate: 0.0,
        preapproved_candidate_ids=("reference", "probe_a", "probe_b", "common"),
    )
    assert controller.select_next().mode == "REFERENCE"
    controller.observe(_obs(candidates[0], 1, 0.1))
    controller.observe(_obs(candidates[1], 2, 0.1))
    controller.observe(_obs(candidates[2], 3, 0.1))
    decision = controller.select_next()
    assert decision.mode == "COMMON_POLICY"
    assert decision.candidate.candidate_id == "common"
    assert decision.gate.status == "PERSONALIZATION_INACTIVE"
    assert decision.gate.posterior_probability < 0.95


def test_positive_cohort_activates_gate_and_selects_personalized_safe_candidate():
    candidates = _candidates()
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        common_predictor=lambda candidate: 0.0,
        preapproved_candidate_ids=("reference", "probe_a", "probe_b", "common"),
        config=GateConfig(interaction_threshold=0.05),
    )
    controller.observe(_obs(candidates[0], 1, 0.0))
    controller.observe(_obs(candidates[1], 2, -0.3))
    gate = controller.observe(_obs(candidates[2], 3, 0.3))
    assert gate.active
    decision = controller.select_next()
    assert decision.mode == "PERSONALIZED_BO"
    assert decision.candidate.candidate_id in {"common", "candidate_c"}
    assert decision.safe_candidate_count == 2


def test_safety_upper_confidence_bound_locks_when_candidate_is_not_certified():
    candidates = _candidates()
    unsafe = replace(candidates[-1], constraint_predictions={"E2": (1.0, 0.2), "peak_ratio": (1.0, 0.2)})
    candidates[-1] = unsafe
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        common_predictor=lambda candidate: 0.0,
        preapproved_candidate_ids=("reference", "probe_a", "probe_b", "common"),
        config=GateConfig(interaction_threshold=0.05),
    )
    controller.observe(_obs(candidates[0], 1, 0.0))
    controller.observe(_obs(candidates[1], 2, -0.3))
    controller.observe(_obs(candidates[2], 3, 0.3))
    decision = controller.select_next()
    assert decision.mode == "PERSONALIZED_BO"
    assert decision.candidate.candidate_id == "common"
    assert decision.safe_candidate_count == 1


def test_runner_exposes_only_observation_responses():
    candidates = _candidates()
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        common_predictor=lambda candidate: 0.0,
        preapproved_candidate_ids=("reference", "probe_a", "probe_b", "common"),
    )
    seen = []

    def evaluator(candidate, trial):
        seen.append(candidate.candidate_id)
        # The evaluator returns a single endpoint observation; no candidate
        # truth/ranking table is made available to the controller.
        return _obs(candidate, trial, 0.1)

    result = run_sequence(controller, evaluator, budget=3)
    assert seen == ["reference", "probe_a", "probe_b"]
    assert result["failure"] is None
    assert len(result["observations"]) == 3


def test_gate_requires_explicit_common_model():
    candidates = _candidates()
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        preapproved_candidate_ids=("reference", "probe_a", "probe_b", "common"),
    )
    for trial, candidate in enumerate(candidates[:3], 1):
        controller.observe(_obs(candidate, trial, float(trial)))
    gate = controller.last_gate
    assert not gate.active
    assert gate.reason == "COMMON_MODEL_REQUIRED"


def test_safety_lock_has_no_executable_candidate_and_recommendation_is_observed_only():
    candidates = _candidates()
    candidates[3] = replace(candidates[3], constraint_predictions={})
    candidates[4] = replace(candidates[4], constraint_predictions={})
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        common_predictor=lambda candidate: 0.0,
        preapproved_candidate_ids=("reference", "probe_a", "probe_b"),
    )
    controller.observe(_obs(candidates[0], 1, 0.2))
    controller.observe(_obs(candidates[1], 2, 0.2))
    controller.observe(_obs(candidates[2], 3, 0.2))
    decision = controller.select_next()
    assert decision.mode == "SAFETY_LOCK"
    assert decision.candidate is None
    assert controller.recommend() == "reference"


def test_inactive_gate_never_substitutes_an_unrelated_alternative_after_common_was_observed():
    candidates = _candidates()
    controller = SASTBO(
        candidates,
        reference_id="reference",
        calibration_probe_ids=("probe_a", "probe_b"),
        common_policy_id="common",
        common_predictor=lambda candidate: 0.0,
        preapproved_candidate_ids=("reference", "probe_a", "probe_b", "common"),
    )
    for trial, candidate in enumerate(candidates[:3], 1):
        controller.observe(_obs(candidate, trial, 0.1))
    controller.observe(_obs(candidates[3], 4, 0.1))
    decision = controller.select_next()
    assert decision.mode == "COMMON_POLICY"
    assert decision.candidate.candidate_id == "common"
