from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from personalization.active_diagnostic_v1 import (
    ACTIVE_DIAGNOSTIC_METHOD,
    ARBITRATION_MARGIN,
    DIAGNOSTIC_SCORE_ID,
    DiagnosticSelection,
    ROMGatedActiveDiagnosticEpisode,
    arbitrate_after_diagnostic,
    average_symmetric_kl_gaussian,
    causal_trial_1_physics_offset,
    run_active_prior_diagnostic_arbitration,
    select_active_diagnostic_candidate,
    verify_frozen_rule,
)
from personalization.candidates import V3CandidateDomain
from personalization.environment import AnalyticBenchmarkEnvironment, make_primary_cases
from personalization.models.base import Prediction
from personalization.models.physics_graybox import PhysicsSubjectModel
from personalization.models.residual_gp import PhysicsInformedResidualModel
from personalization.models.standard_gp import StandardGaussianProcess
from personalization.observations import EpisodeObservation
from personalization.predictive_failover_v1 import (
    PHYSICS_MODE,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
)
from personalization.rom_gated_v2 import (
    SubjectROMProfile,
    SubjectSpecificV3CandidateDomain,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
)
from personalization.selectors import LowerConfidenceBoundSelector


ROOT = Path(__file__).resolve().parents[1]


class TableModel:
    def __init__(self, means, stds=0.25):
        self.means = means
        self.stds = stds
        self.fit_history_lengths = []

    def fit(self, history):
        self.fit_history_lengths.append(len(history))

    def predict(self, candidate):
        mean = self.means[candidate.candidate_id]
        std = self.stds[candidate.candidate_id] if isinstance(self.stds, dict) else self.stds
        return Prediction(float(mean), float(std), True, {"table_model": True})

    def state_summary(self):
        return {"model": "generic_table_model"}


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
            "adapter": "GENERIC_ACTIVE_DIAGNOSTIC_TEST_ADAPTER",
            "fit_count": self.fit_count,
        }


class InvalidTrial2Environment(AnalyticBenchmarkEnvironment):
    def evaluate(self, candidate, trial_index):
        if trial_index != 2:
            return super().evaluate(candidate, trial_index)
        return EpisodeObservation(
            episode_id="invalid-diagnostic:2",
            trial_index=2,
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


def manual_selection(
    candidate,
    *,
    standard_mean,
    physics_mean,
    physics_offset=0.0,
    standard_std=0.25,
    physics_std=0.25,
):
    predictions = PretrialPredictionSnapshot(
        trial_index=2,
        candidate_id=candidate.candidate_id,
        beta_flex=candidate.beta_flex,
        beta_extend=candidate.beta_extend,
        history_size_before=1,
        standard_bo_mean_before=standard_mean,
        standard_bo_std_before=standard_std,
        standard_bo_prediction_valid=True,
        physics_bo_mean_before=physics_mean,
        physics_bo_std_before=physics_std,
        physics_bo_prediction_valid=True,
    )
    adjusted = physics_mean + physics_offset
    return DiagnosticSelection(
        candidate=candidate,
        predictions=predictions,
        physics_offset=physics_offset,
        adjusted_physics_mean=adjusted,
        diagnostic_score=average_symmetric_kl_gaussian(
            standard_mean, standard_std, adjusted, physics_std
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


def test_primary_active_diagnostic_rule_is_frozen():
    payload = verify_frozen_rule()
    assert payload["diagnostic_candidate"]["score_id"] == DIAGNOSTIC_SCORE_ID
    assert payload["arbitration"]["evidence_margin"] == pytest.approx(math.log(10.0))
    assert payload["inconclusive_policy"] == PHYSICS_MODE
    assert payload["primary_budget"] == 4
    assert payload["selected_expert_fixed_after_trial_2"] is True


def test_symmetric_kl_uses_mean_and_predictive_uncertainty_stably():
    assert average_symmetric_kl_gaussian(0.0, 0.0, 0.0, 0.0) == 0.0
    precise_disagreement = average_symmetric_kl_gaussian(0.0, 0.1, 1.0, 0.1)
    uncertain_disagreement = average_symmetric_kl_gaussian(0.0, 1.0, 1.0, 1.0)
    variance_only = average_symmetric_kl_gaussian(0.0, 0.1, 0.0, 0.5)
    assert precise_disagreement > uncertain_disagreement > 0.0
    assert variance_only > 0.0
    assert math.isfinite(average_symmetric_kl_gaussian(1.0, 0.0, -1.0, 0.0))


def test_identical_experts_use_deterministic_lowest_index_tie(small_domain):
    history = [make_observation(small_domain.reference, 1, 0.0)]
    means = {candidate.candidate_id: 0.0 for candidate in small_domain}
    first = select_active_diagnostic_candidate(
        history,
        small_domain,
        standard_model=TableModel(means),
        physics_model=TableModel(means),
        physics_offset=0.0,
    )
    second = select_active_diagnostic_candidate(
        history,
        small_domain,
        standard_model=TableModel(means),
        physics_model=TableModel(means),
        physics_offset=0.0,
    )
    assert first.candidate == second.candidate
    assert first.candidate.candidate_index == 0
    assert first.diagnostic_score == 0.0


def test_trial_2_candidate_maximizes_d1_distribution_disagreement(small_domain):
    history = [make_observation(small_domain.reference, 1, 0.0)]
    standard_means = {candidate.candidate_id: 0.0 for candidate in small_domain}
    physics_means = {candidate.candidate_id: 0.0 for candidate in small_domain}
    target = small_domain.as_tuple()[-1]
    physics_means[target.candidate_id] = 3.0
    selection = select_active_diagnostic_candidate(
        history,
        small_domain,
        standard_model=TableModel(standard_means),
        physics_model=TableModel(physics_means),
        physics_offset=0.0,
    )
    assert selection.candidate == target
    assert selection.predictions.history_size_before == 1
    assert selection.predictions.trial_index == 2


def test_causal_trial_1_offset_corrects_constant_bias(small_domain):
    reference = small_domain.reference
    trial_1 = PretrialPredictionSnapshot(
        1,
        reference.candidate_id,
        reference.beta_flex,
        reference.beta_extend,
        0,
        0.0,
        0.25,
        True,
        2.0,
        0.25,
        True,
    )
    offset = causal_trial_1_physics_offset(
        trial_1, make_observation(reference, 1, 0.0)
    )
    selection = manual_selection(
        small_domain.as_tuple()[0],
        standard_mean=0.0,
        physics_mean=3.0,
        physics_offset=offset,
    )
    decision = arbitrate_after_diagnostic(
        selection, make_observation(selection.candidate, 2, 1.0)
    )
    assert offset == pytest.approx(-2.0)
    assert selection.adjusted_physics_mean == pytest.approx(1.0)
    assert decision.selected_expert == PHYSICS_MODE
    assert decision.score_difference > ARBITRATION_MARGIN


def test_ranking_inversion_selects_standard_and_scaling_case_can_keep_physics(
    small_domain,
):
    candidate = small_domain.as_tuple()[0]
    inverted = manual_selection(
        candidate, standard_mean=1.0, physics_mean=-2.0
    )
    inverted_decision = arbitrate_after_diagnostic(
        inverted, make_observation(candidate, 2, 1.0)
    )
    scaling = manual_selection(
        candidate,
        standard_mean=-1.0,
        physics_mean=1.2,
        physics_offset=-0.2,
        standard_std=0.4,
        physics_std=0.4,
    )
    scaling_decision = arbitrate_after_diagnostic(
        scaling, make_observation(candidate, 2, 1.0)
    )
    assert inverted_decision.selected_expert == STANDARD_MODE
    assert scaling_decision.selected_expert == PHYSICS_MODE


def test_single_noisy_diagnostic_is_finite_and_can_be_inconclusive(small_domain):
    candidate = small_domain.as_tuple()[0]
    selection = manual_selection(
        candidate, standard_mean=0.0, physics_mean=0.2
    )
    decision = arbitrate_after_diagnostic(
        selection, make_observation(candidate, 2, 1.0, uncertainty=1.0)
    )
    assert math.isfinite(decision.standard_log_score)
    assert math.isfinite(decision.physics_log_score)
    assert abs(decision.score_difference) < ARBITRATION_MARGIN
    assert decision.selected_expert == PHYSICS_MODE
    assert decision.decision == "INCONCLUSIVE_DEFAULT_PHYSICS"


def test_invalid_trial_2_consumes_budget_and_uses_inconclusive_policy(small_domain):
    case = make_primary_cases()[0]
    result = run_active_prior_diagnostic_arbitration(
        InvalidTrial2Environment(small_domain, case, seed=0),
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    assert len(result.ledger.entries) == 4
    assert not result.ledger.entries[1].observation.valid
    assert result.arbitration.score_difference is None
    assert result.arbitration.selected_expert == PHYSICS_MODE
    assert result.arbitration.decision.startswith("INVALID_DIAGNOSTIC")


def test_episode_is_strict_reference_diagnostic_then_fixed_expert(small_domain):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=0)
    result = run_active_prior_diagnostic_arbitration(
        environment,
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    entries = result.ledger.entries
    assert result.method == ACTIVE_DIAGNOSTIC_METHOD
    assert entries[0].candidate == small_domain.reference
    assert entries[0].selected_next_candidate == entries[1].candidate
    assert entries[1].trial_role == "ACTIVE_PRIOR_DIAGNOSTIC_AND_MODEL_ARBITRATION"
    assert entries[1].pretrial_predictions.history_size_before == 1
    assert entries[1].diagnostic_score == result.diagnostic_selection.diagnostic_score
    assert all(
        entry.selected_expert_after_trial_2 == result.selected_expert
        for entry in entries[1:]
    )
    assert len(set(result.ledger.executed_candidate_ids)) == 4
    assert environment.oracle_access_count == 0


@pytest.mark.parametrize(
    ("adapter", "expected_expert"),
    [
        (SmoothTruthAdapter(), PHYSICS_MODE),
        (SmoothTruthAdapter(bias=100.0, scale=-50.0), STANDARD_MODE),
    ],
)
def test_trials_3_4_and_final_exactly_use_selected_frozen_expert(
    small_domain, adapter, expected_expert
):
    case = make_primary_cases()[0]
    result = run_active_prior_diagnostic_arbitration(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=1),
        small_domain,
        physics_model=PhysicsSubjectModel(adapter),
    )
    assert result.selected_expert == expected_expert
    for index, entry in enumerate(result.ledger.entries[1:3], start=1):
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


@pytest.mark.parametrize("budget", [3, 5])
def test_primary_active_diagnostic_budget_remains_k4(small_domain, budget):
    case = make_primary_cases()[0]
    with pytest.raises(ValueError, match="frozen at K=4"):
        run_active_prior_diagnostic_arbitration(
            AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
            small_domain,
            physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
            budget=budget,
        )


def test_rom_must_be_frozen(rom_stage):
    _, domain = rom_stage
    unfrozen = SubjectROMProfile(
        profile_id="UNFROZEN_ACTIVE_DIAGNOSTIC_TEST",
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
        ROMGatedActiveDiagnosticEpisode(
            episode_id="invalid", profile=unfrozen, domain=domain
        )


def test_algorithm_has_no_oracle_case_label_or_robot_stack_access():
    package = ROOT / "personalization" / "active_diagnostic_v1"
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
        (package / "PRIMARY_ACTIVE_DIAGNOSTIC_RULE_V1.json").read_text()
    )
    encoded = json.dumps(rule)
    for label in ("P0", "P1", "P2", "P3", "regret", "true_optimum"):
        assert label not in encoded


def test_formal_benchmark_preserves_history_and_reports_k4_limitation():
    results = (
        ROOT
        / "personalization"
        / "benchmarks"
        / "results_active_diagnostic_v1"
    )
    payload = json.loads((results / "benchmark_summary.json").read_text())
    assert payload["run_count"] == 1200
    assert payload["primary_budget"] == 4
    assert payload["candidate_count"] == 625
    assert payload["same_rom_fingerprint_for_all_paired_methods"] is True
    assert payload["historical_baseline_equivalence"] == {
        "compared_metric_count": 192,
        "exact_within_1e_12": True,
        "historical_summary_path": (
            "personalization/benchmarks/results_predictive_failover_v1/benchmark_summary.json"
        ),
        "maximum_absolute_error": 0.0,
    }
    assert payload["implementation_status"].endswith("IMPLEMENTED_WITH_LIMITATIONS")
    assert payload["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"] is False
    assert payload[
        "K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED"
    ] is True
    assert payload["robot_actions"] == payload["human_actions"] == 0
    assert payload["PINN_training"] == 0
    assert len(payload["diagnostic_selection_rows"]) == 240

    informative = [
        row
        for row in payload["arbitration_summary"]
        if row["prior_quality"] in {"P0", "P1", "P2"}
    ]
    assert sum(
        row["false_rejection_of_useful_physics_count"] for row in informative
    ) == 0
    assert payload["readiness_assessment"][
        "informative_positive_benefit_retention_fraction"
    ] == pytest.approx(1.043416036181235)

    p3 = {row["noise_label"]: row for row in payload["p3_analysis"]}
    assert [
        p3[label]["misleading_prior_rejection_rate"]
        for label in ("zero", "low", "moderate")
    ] == [0.75, 0.7, 0.6]
    assert [
        p3[label]["active_missed_rejection_count"]
        for label in ("zero", "low", "moderate")
    ] == [5, 6, 8]
    assert [
        p3[label]["missed_rejection_reduction_vs_passive"]
        for label in ("zero", "low", "moderate")
    ] == [5, 2, 1]
    assert p3["zero"]["active_improvement_vs_passive"] > 0.0
    assert p3["low"]["active_improvement_vs_passive"] > 0.0
    assert p3["moderate"]["active_improvement_vs_passive"] < 0.0
    assert len(payload["diagnostic_candidate_summary"]) == 6
