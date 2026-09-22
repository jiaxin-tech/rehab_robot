from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from personalization.active_diagnostic_v1 import (
    ACTIVE_DIAGNOSTIC_METHOD,
    PRIMARY_ADAPTATION_BUDGET as K4_BUDGET,
    average_symmetric_kl_gaussian,
    run_active_prior_diagnostic_arbitration,
)
from personalization.active_diagnostic_v1.diagnostic import DiagnosticSelection
from personalization.benchmarks.run_k4_vs_k5_information_budget_study_v1 import (
    REPEATED_K5,
)
from personalization.candidates import V3CandidateDomain
from personalization.environment import AnalyticBenchmarkEnvironment, make_primary_cases
from personalization.models.physics_graybox import PhysicsSubjectModel
from personalization.models.residual_gp import PhysicsInformedResidualModel
from personalization.models.standard_gp import StandardGaussianProcess
from personalization.observations import EpisodeObservation
from personalization.predictive_failover_v1 import (
    PHYSICS_MODE,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
)
from personalization.repeated_active_diagnostic_k5_v1 import (
    PRIMARY_BUDGET,
    REPEATED_ACTIVE_DIAGNOSTIC_METHOD,
    ROMGatedRepeatedDiagnosticEpisode,
    RepeatedDiagnosticEvidence,
    arbitrate_after_trial_3,
    causal_running_physics_offset,
    evidence_consistency,
    run_repeated_active_diagnostic_arbitration_k5,
    score_repeated_diagnostic_observation,
    verify_frozen_rule,
)
from personalization.rom_gated_v2 import (
    SubjectROMProfile,
    SubjectSpecificV3CandidateDomain,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
)
from personalization.selectors import LowerConfidenceBoundSelector


ROOT = Path(__file__).resolve().parents[1]


class SmoothTruthAdapter:
    def __init__(self, *, optimum=(0.0125, -0.01), bias=0.0, scale=1.0):
        self.optimum = optimum
        self.bias = float(bias)
        self.scale = float(scale)
        self.fit_count = 0

    def fit(self, history):
        self.fit_count = sum(item.valid for item in history)

    def predict_value(self, candidate):
        x = candidate.beta_flex / 0.03
        z = candidate.beta_extend / 0.03
        ox = self.optimum[0] / 0.03
        oz = self.optimum[1] / 0.03
        truth = 0.55 * (x - ox) ** 2 + 0.45 * (z - oz) ** 2 - 0.25
        return self.bias + self.scale * truth

    def metadata(self):
        return {
            "adapter": "GENERIC_REPEATED_DIAGNOSTIC_TEST_ADAPTER",
            "fit_count": self.fit_count,
        }


class InvalidDiagnosticEnvironment(AnalyticBenchmarkEnvironment):
    def __init__(self, *args, invalid_trial, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_trial = invalid_trial

    def evaluate(self, candidate, trial_index):
        if trial_index != self.invalid_trial:
            return super().evaluate(candidate, trial_index)
        return EpisodeObservation(
            episode_id=f"invalid-diagnostic:{trial_index}",
            trial_index=trial_index,
            candidate_id=candidate.candidate_id,
            beta_flex=candidate.beta_flex,
            beta_extend=candidate.beta_extend,
            endpoint_name=self.endpoint_name,
            endpoint_value=None,
            endpoint_unit=self.endpoint_unit,
            endpoint_uncertainty=self.case.noise_std,
            valid=False,
            invalid_reason="INJECTED_INVALID_DIAGNOSTIC",
            metadata={"classification": "OFFLINE_ALGORITHM_SEMANTIC_TEST"},
        )


def make_observation(candidate, trial, value, uncertainty=0.0):
    valid = value is not None
    return EpisodeObservation(
        episode_id=f"semantic:{trial}",
        trial_index=trial,
        candidate_id=candidate.candidate_id,
        beta_flex=candidate.beta_flex,
        beta_extend=candidate.beta_extend,
        endpoint_name="offline_semantic_cost",
        endpoint_value=value,
        endpoint_unit="normalized_cost",
        endpoint_uncertainty=uncertainty,
        valid=valid,
        invalid_reason=None if valid else "INJECTED_INVALID_DIAGNOSTIC",
        metadata={"classification": "OFFLINE_ALGORITHM_SEMANTIC_TEST"},
    )


def make_snapshot(candidate, trial, standard_mean, physics_mean, std=0.25):
    return PretrialPredictionSnapshot(
        trial_index=trial,
        candidate_id=candidate.candidate_id,
        beta_flex=candidate.beta_flex,
        beta_extend=candidate.beta_extend,
        history_size_before=trial - 1,
        standard_bo_mean_before=standard_mean,
        standard_bo_std_before=std,
        standard_bo_prediction_valid=True,
        physics_bo_mean_before=physics_mean,
        physics_bo_std_before=std,
        physics_bo_prediction_valid=True,
    )


def make_selection(candidate, trial, standard_mean, physics_mean, offset=0.0, std=0.25):
    snapshot = make_snapshot(candidate, trial, standard_mean, physics_mean, std)
    adjusted = physics_mean + offset
    return DiagnosticSelection(
        candidate=candidate,
        predictions=snapshot,
        physics_offset=offset,
        adjusted_physics_mean=adjusted,
        diagnostic_score=average_symmetric_kl_gaussian(
            standard_mean, std, adjusted, std
        ),
        standard_to_physics_kl=0.0,
        physics_to_standard_kl=0.0,
    )


@pytest.fixture(scope="module")
def small_domain():
    return V3CandidateDomain.regular_grid([-0.03, 0.0, 0.03])


@pytest.fixture(scope="module")
def rom_stage():
    case = make_offline_rom_development_cases()[1]
    _, profile = determine_synthetic_rom(case)
    return profile, SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)


def test_k5_rule_reuses_k4_score_margin_and_freezes_budget():
    payload = verify_frozen_rule()
    assert PRIMARY_BUDGET == 5
    assert K4_BUDGET == 4
    assert payload["diagnostic_candidate"]["score_id"] == (
        "AVERAGE_SYMMETRIC_KL_GAUSSIAN"
    )
    assert payload["arbitration"]["evidence_margin"] == pytest.approx(math.log(10.0))
    assert payload["arbitration"]["inconclusive_policy"] == PHYSICS_MODE


def test_causal_running_offset_uses_only_prior_prediction_observations(small_domain):
    candidates = small_domain.as_tuple()
    snapshot_1 = make_snapshot(candidates[4], 1, 0.0, 2.0)
    snapshot_2 = make_snapshot(candidates[0], 2, 0.0, 4.0)
    observation_1 = make_observation(candidates[4], 1, 0.0)
    observation_2 = make_observation(candidates[0], 2, 1.0)
    assert causal_running_physics_offset(
        [(snapshot_1, observation_1)], before_trial=2
    ) == pytest.approx(-2.0)
    assert causal_running_physics_offset(
        [(snapshot_1, observation_1), (snapshot_2, observation_2)],
        before_trial=3,
    ) == pytest.approx(np.median([-2.0, -3.0]))
    with pytest.raises(ValueError, match="current or future"):
        causal_running_physics_offset(
            [(snapshot_2, observation_2)], before_trial=2
        )


def test_two_prequential_scores_sum_exactly_before_trial_3_arbitration(small_domain):
    candidate_2, candidate_3 = small_domain.as_tuple()[:2]
    selection_2 = make_selection(candidate_2, 2, 0.0, 1.0)
    selection_3 = make_selection(candidate_3, 3, 0.0, 2.0)
    evidence_2 = score_repeated_diagnostic_observation(
        selection_2, make_observation(candidate_2, 2, 1.0)
    )
    evidence_3 = score_repeated_diagnostic_observation(
        selection_3, make_observation(candidate_3, 3, 2.0)
    )
    decision = arbitrate_after_trial_3((evidence_2, evidence_3))
    assert evidence_2.selection.predictions.history_size_before == 1
    assert evidence_3.selection.predictions.history_size_before == 2
    assert decision.cumulative_evidence == pytest.approx(
        evidence_2.score_difference + evidence_3.score_difference
    )
    assert decision.selected_expert == PHYSICS_MODE


def test_consistent_and_conflicting_evidence_labels_are_deterministic(small_domain):
    candidate_2, candidate_3 = small_domain.as_tuple()[:2]

    def evidence(trial, candidate, difference):
        selection = make_selection(candidate, trial, 0.0, 0.0)
        return RepeatedDiagnosticEvidence(
            trial,
            selection,
            True,
            0.0,
            0.0,
            difference,
            difference,
            "FAVORS_PHYSICS" if difference > 0 else "FAVORS_STANDARD",
            "VALID_PREQUENTIAL_GAUSSIAN_LOG_SCORE",
        )

    positive = evidence(2, candidate_2, 1.0)
    positive_3 = evidence(3, candidate_3, 2.0)
    negative_3 = evidence(3, candidate_3, -2.0)
    assert evidence_consistency((positive, positive_3)) == (
        "CONSISTENT_FAVORS_PHYSICS"
    )
    assert evidence_consistency((positive, negative_3)) == "CONFLICTING_EVIDENCE"


def test_k5_episode_is_reference_two_diagnostics_then_fixed_expert(small_domain):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=0)
    result = run_repeated_active_diagnostic_arbitration_k5(
        environment,
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    entries = result.ledger.entries
    assert result.method == REPEATED_ACTIVE_DIAGNOSTIC_METHOD
    assert len(entries) == 5
    assert entries[0].candidate == small_domain.reference
    assert entries[0].selected_next_candidate == entries[1].candidate
    assert entries[1].selected_next_candidate == entries[2].candidate
    assert entries[1].candidate != entries[2].candidate
    assert entries[1].pretrial_predictions.history_size_before == 1
    assert entries[2].pretrial_predictions.history_size_before == 2
    assert entries[1].arbitration_decision == "DEFERRED_UNTIL_AFTER_TRIAL_3"
    assert entries[1].selected_expert_after_trial_3 is None
    assert entries[2].arbitration_decision == result.arbitration.decision
    assert all(
        entry.selected_expert_after_trial_3 == result.selected_expert
        for entry in entries[2:]
    )
    assert len(set(result.ledger.executed_candidate_ids)) == 5
    assert environment.oracle_access_count == 0
    assert result.arbitration.cumulative_evidence == pytest.approx(
        sum(item.score_difference for item in result.diagnostic_evidence)
    )


def test_trial_3_offset_and_selection_use_d2_only(small_domain):
    case = make_primary_cases()[0]
    result = run_repeated_active_diagnostic_arbitration_k5(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=3),
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    trial_1 = result.ledger.entries[0]
    diagnostic_2 = result.diagnostic_selections[0]
    expected = np.median(
        [
            float(trial_1.observation.endpoint_value)
            - trial_1.pretrial_predictions.physics_bo_mean_before,
            float(result.ledger.entries[1].observation.endpoint_value)
            - diagnostic_2.predictions.physics_bo_mean_before,
        ]
    )
    assert result.diagnostic_selections[1].physics_offset == pytest.approx(expected)
    assert result.diagnostic_selections[1].predictions.history_size_before == 2


@pytest.mark.parametrize(
    ("adapter", "expected_expert"),
    [
        (SmoothTruthAdapter(), PHYSICS_MODE),
        (SmoothTruthAdapter(bias=100.0, scale=-50.0), STANDARD_MODE),
    ],
)
def test_selected_expert_exactly_controls_trials_4_5_and_final(
    small_domain, adapter, expected_expert
):
    case = make_primary_cases()[0]
    result = run_repeated_active_diagnostic_arbitration_k5(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=1),
        small_domain,
        physics_model=PhysicsSubjectModel(adapter),
    )
    assert result.selected_expert == expected_expert
    for index, entry in enumerate(result.ledger.entries[2:4], start=2):
        history = result.ledger.observations[: index + 1]
        if expected_expert == STANDARD_MODE:
            model = StandardGaussianProcess()
            selector_name = "Standard BO"
        else:
            model = PhysicsInformedResidualModel(
                PhysicsSubjectModel(SmoothTruthAdapter())
            )
            selector_name = "Physics-Informed BO"
        model.fit(history)
        expected = LowerConfidenceBoundSelector(
            name=selector_name, kappa=1.5
        ).select_next(history, small_domain, model)
        assert entry.selected_next_candidate == expected.candidate
        assert entry.acquisition_value == expected.acquisition_value

    if expected_expert == STANDARD_MODE:
        final_model = StandardGaussianProcess()
    else:
        final_model = PhysicsInformedResidualModel(
            PhysicsSubjectModel(SmoothTruthAdapter())
        )
    final_model.fit(result.ledger.observations)
    expected_final = min(
        small_domain,
        key=lambda item: (final_model.predict(item).mean, item.candidate_index),
    )
    assert result.model_recommended_final_candidate == expected_final


@pytest.mark.parametrize("invalid_trial", [2, 3])
def test_invalid_diagnostic_consumes_budget_without_evidence_or_retry(
    small_domain, invalid_trial
):
    case = make_primary_cases()[0]
    result = run_repeated_active_diagnostic_arbitration_k5(
        InvalidDiagnosticEnvironment(
            small_domain, case, seed=0, invalid_trial=invalid_trial
        ),
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    invalid_evidence = result.diagnostic_evidence[invalid_trial - 2]
    assert len(result.ledger.entries) == 5
    assert invalid_evidence.score_difference is None
    assert invalid_evidence.evidence_sign == "INVALID_NO_EVIDENCE"
    assert result.arbitration.valid_evidence_count == 1
    assert len(set(result.ledger.executed_candidate_ids)) == 5


@pytest.mark.parametrize("budget", [4, 6])
def test_k5_budget_is_exactly_enforced(small_domain, budget):
    case = make_primary_cases()[0]
    with pytest.raises(ValueError, match="frozen at K=5"):
        run_repeated_active_diagnostic_arbitration_k5(
            AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
            small_domain,
            physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
            budget=budget,
        )


def test_k4_historical_algorithm_remains_k4_and_unchanged(small_domain):
    case = make_primary_cases()[0]
    result = run_active_prior_diagnostic_arbitration(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    assert result.method == ACTIVE_DIAGNOSTIC_METHOD
    assert result.budget == 4
    assert len(result.ledger.entries) == 4


def test_k5_rom_must_remain_frozen(rom_stage):
    _, domain = rom_stage
    unfrozen = SubjectROMProfile(
        profile_id="UNFROZEN_REPEATED_DIAGNOSTIC_TEST",
        version=1,
        hip_min_rad=0.4,
        hip_max_rad=1.5,
        knee_min_rad=0.3,
        knee_max_rad=2.0,
        rom_status="DRAFT",
        provenance="ALGORITHM_DEVELOPMENT_ONLY",
        frozen=False,
    )
    with pytest.raises(RuntimeError, match="ROM_PROFILE_FROZEN"):
        ROMGatedRepeatedDiagnosticEpisode(
            episode_id="invalid", profile=unfrozen, domain=domain
        )


def test_repeated_diagnostic_has_no_oracle_case_label_or_robot_imports():
    package = ROOT / "personalization" / "repeated_active_diagnostic_k5_v1"
    forbidden_imports = {"hardware", "control", "collection", "safety"}
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "oracle_value" not in source
        assert "oracle_optimum" not in source
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.lstrip(".").split(".")[0]
                if root:
                    imported.add(root)
        assert not (imported & forbidden_imports)
    rule = json.loads(
        (package / "PRIMARY_REPEATED_ACTIVE_DIAGNOSTIC_K5_V1.json").read_text()
    )
    encoded = json.dumps(rule)
    for label in ("P0", "P1", "P2", "P3", "regret", "true_optimum"):
        assert label not in encoded


def test_formal_information_budget_artifact_records_mixed_decision():
    summary_path = (
        ROOT
        / "personalization"
        / "benchmarks"
        / "results_k4_vs_k5_information_budget_v1"
        / "study_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["status"] == (
        "K4_VS_K5_PRIOR_IDENTIFICATION_INFORMATION_BUDGET_STUDY_V1_COMPLETED"
    )
    assert payload["classification"] == (
        "OFFLINE_ALGORITHM_DEVELOPMENT_EVIDENCE_ONLY"
    )
    assert payload["run_count"] == 1440
    assert payload["historical_k4_equivalence"] == {
        "compared_metric_count": 144,
        "exact_within_1e_12": True,
        "historical_summary_path": (
            "personalization/benchmarks/results_active_diagnostic_v1/"
            "benchmark_summary.json"
        ),
        "maximum_absolute_error": 0.0,
    }

    decision = payload["study_decision"]
    assert decision["scientific_conclusion"] == "EVIDENCE_MIXED"
    assert decision["k4_total_p3_missed_rejection_count"] == 19
    assert decision["k5_total_p3_missed_rejection_count"] == 3
    assert decision["clear_identification_improvement"] is True
    assert decision["informative_benefit_retained_at_75_percent"] is False
    assert decision["maximum_informative_false_rejection_rate"] == 0.75
    assert decision["k5_p3_negative_transfer_at_most_0_05_each_noise"] is False
    assert payload["FREEZE_PRIMARY_PERSONALIZATION_BUDGET"] == "NONE"
    assert payload["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"] is False

    p3_by_noise = {row["noise_label"]: row for row in payload["p3_analysis"]}
    assert [p3_by_noise[label]["k5_missed_rejection_count"] for label in (
        "zero",
        "low",
        "moderate",
    )] == [0, 0, 3]
    assert p3_by_noise["moderate"]["k5_rejection_rate"] == 0.85
    assert p3_by_noise["moderate"]["mean_final_regret"][
        REPEATED_K5
    ] == pytest.approx(0.22473006908431178)
    assert p3_by_noise["moderate"][
        "k5_negative_transfer_vs_standard_k5"
    ] == pytest.approx(0.1798567957837764)

    moderate_missed = payload["p3_moderate_missed_rejection_analysis"]
    assert moderate_missed["k4_missed_rejection_count"] == 8
    assert moderate_missed["k5_missed_rejection_count"] == 3
    assert moderate_missed["k4_missed_corrected_by_k5_count"] == 5
    assert moderate_missed["k4_correct_but_k5_missed_count"] == 0

    assert len(payload["diagnostic_evidence_rows"]) == 240
    assert all(
        not row["diagnostic_candidate_repeated"]
        for row in payload["diagnostic_evidence_rows"]
    )
    assert payload["causal_audit"] == {
        "arbitration_occurs_after_trial_3": True,
        "case_label_available_to_algorithm": False,
        "future_truth_available_to_algorithm": False,
        "k4_algorithm_modified": False,
        "selected_expert_fixed_for_trials_4_and_5": True,
        "trial_2_prediction_history": "D1",
        "trial_3_prediction_history": "D2",
    }
