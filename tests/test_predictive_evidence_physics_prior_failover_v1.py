from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from personalization.candidates import V3CandidateDomain
from personalization.environment import AnalyticBenchmarkEnvironment, make_primary_cases
from personalization.models.physics_graybox import PhysicsSubjectModel
from personalization.models.residual_gp import PhysicsInformedResidualModel
from personalization.models.standard_gp import StandardGaussianProcess
from personalization.observations import EpisodeObservation
from personalization.predictive_failover_v1 import (
    FAILOVER_THRESHOLD,
    PHYSICS_MODE,
    PRIMARY_RULE_ID,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
    PredictiveEvidenceArbitrator,
    ROMGatedPredictiveFailoverEpisode,
    gaussian_log_predictive_density,
    run_predictive_evidence_failover,
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
            "adapter": "GENERIC_OFFLINE_SEMANTIC_TEST_ADAPTER",
            "fit_count": self.fit_count,
        }


def observation(
    snapshot: PretrialPredictionSnapshot,
    value: float | None,
    *,
    uncertainty: float = 0.0,
) -> EpisodeObservation:
    valid = value is not None
    return EpisodeObservation(
        episode_id=f"semantic:{snapshot.trial_index}",
        trial_index=snapshot.trial_index,
        candidate_id=snapshot.candidate_id,
        beta_flex=snapshot.beta_flex,
        beta_extend=snapshot.beta_extend,
        endpoint_name="offline_semantic_cost",
        endpoint_value=value,
        endpoint_unit="normalized_cost",
        endpoint_uncertainty=uncertainty,
        valid=valid,
        invalid_reason=None if valid else "INJECTED_INVALID_SEMANTIC_OBSERVATION",
        metadata={"classification": "OFFLINE_ALGORITHM_SEMANTIC_TEST"},
    )


def snapshot(
    trial: int,
    *,
    standard_mean: float,
    physics_mean: float,
    standard_std: float = 0.25,
    physics_std: float = 0.25,
) -> PretrialPredictionSnapshot:
    return PretrialPredictionSnapshot(
        trial_index=trial,
        candidate_id=f"semantic_{trial}",
        beta_flex=0.001 * trial,
        beta_extend=-0.001 * trial,
        history_size_before=trial - 1,
        standard_bo_mean_before=standard_mean,
        standard_bo_std_before=standard_std,
        standard_bo_prediction_valid=True,
        physics_bo_mean_before=physics_mean,
        physics_bo_std_before=physics_std,
        physics_bo_prediction_valid=True,
    )


def drive_semantic_sequence(rows):
    arbitrator = PredictiveEvidenceArbitrator()
    state = arbitrator.initial_state()
    for trial, row in enumerate(rows, start=1):
        current = snapshot(
            trial,
            standard_mean=row[1],
            physics_mean=row[2],
            standard_std=row[3] if len(row) > 3 else 0.25,
            physics_std=row[4] if len(row) > 4 else 0.25,
        )
        state = arbitrator.update(
            state,
            current,
            observation(
                current,
                row[0],
                uncertainty=row[5] if len(row) > 5 else 0.0,
            ),
        )
    return state


@pytest.fixture(scope="module")
def small_domain():
    return V3CandidateDomain.regular_grid([-0.03, 0.0, 0.03])


@pytest.fixture(scope="module")
def rom_stage():
    case = make_offline_rom_development_cases()[1]
    _, profile = determine_synthetic_rom(case)
    return profile, SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)


def test_primary_rule_is_frozen_before_formal_benchmark():
    payload = verify_frozen_rule()
    assert payload["primary_rule_id"] == PRIMARY_RULE_ID
    assert payload["failover"]["threshold"] == pytest.approx(-math.log(10.0))
    assert payload["calibration"]["mode"] == (
        "CAUSAL_RUNNING_MEDIAN_PHYSICS_OFFSET"
    )
    assert payload["primary_budget"] == 4
    assert payload["selection_semantics"]["continuous_mixture_used"] is False


def test_gaussian_log_score_uses_uncertainty_and_variance_floor():
    floored = gaussian_log_predictive_density(0.0, 0.0, 0.0, 0.0)
    expected = -0.5 * math.log(2.0 * math.pi * 0.05**2)
    assert floored == pytest.approx(expected)
    uncertain = gaussian_log_predictive_density(1.0, 0.0, 0.0, 1.0)
    overconfident = gaussian_log_predictive_density(1.0, 0.0, 0.0, 0.0)
    assert uncertain > overconfident
    assert math.isfinite(floored)


def test_causal_running_median_offset_uses_only_prior_trials():
    state = drive_semantic_sequence(
        [
            (0.0, 0.5, 2.0),
            (1.0, 0.0, 3.0),
            (2.0, 0.0, 4.0),
        ]
    )
    records = state.records
    assert records[0].physics_offset_before == 0.0
    assert records[1].physics_offset_before == pytest.approx(-2.0)
    assert records[1].physics_scoring_mean == pytest.approx(1.0)
    assert records[2].physics_offset_before == pytest.approx(-2.0)
    assert records[2].physics_scoring_mean == pytest.approx(2.0)
    assert not state.failed_over


def test_generic_accurate_and_scaling_cases_normally_stay_physics():
    accurate = drive_semantic_sequence(
        [(0.0, 0.8, 0.0), (1.0, 0.0, 1.0), (2.0, 0.2, 2.0)]
    )
    scaling = drive_semantic_sequence(
        [
            (0.0, 0.5, 0.0),
            (1.0, 0.0, 1.3, 0.4, 0.4),
            (2.0, 0.2, 2.4, 0.4, 0.4),
        ]
    )
    assert accurate.mode == PHYSICS_MODE
    assert accurate.cumulative_evidence > 0.0
    assert scaling.mode == PHYSICS_MODE


def test_generic_ranking_inversion_generates_failover_and_is_one_way():
    state = drive_semantic_sequence(
        [(0.0, 0.0, 2.0), (1.0, 1.0, 1.0), (2.0, 2.0, 0.0)]
    )
    assert state.cumulative_evidence < FAILOVER_THRESHOLD
    assert state.failover_trial == 2
    assert [record.mode_after for record in state.records] == [
        PHYSICS_MODE,
        STANDARD_MODE,
        STANDARD_MODE,
    ]


def test_single_uncertain_outlier_does_not_lightly_trigger_failover():
    state = drive_semantic_sequence(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.2, 0.25, 0.25, 1.0),
        ]
    )
    assert state.cumulative_evidence > FAILOVER_THRESHOLD
    assert not state.failed_over


def test_invalid_observation_consumes_trial_without_evidence_or_failover():
    arbitrator = PredictiveEvidenceArbitrator()
    first = snapshot(1, standard_mean=0.0, physics_mean=0.0)
    state = arbitrator.update(
        arbitrator.initial_state(), first, observation(first, 0.0)
    )
    second = snapshot(2, standard_mean=0.0, physics_mean=100.0)
    state = arbitrator.update(state, second, observation(second, None))
    assert state.trial_index == 2
    assert state.scored_evidence_count == 0
    assert state.cumulative_evidence == 0.0
    assert state.mode == PHYSICS_MODE
    assert state.records[-1].standard_log_score is None
    assert state.records[-1].physics_log_score is None


def test_end_to_end_ledger_stores_strict_pretrial_predictions(small_domain):
    case = make_primary_cases()[0]
    result = run_predictive_evidence_failover(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    assert len(result.ledger.entries) == 4
    for trial, entry in enumerate(result.ledger.entries, start=1):
        assert entry.pretrial_predictions.history_size_before == trial - 1
        assert entry.pretrial_predictions.trial_index == trial
        assert entry.standard_bo_mean_before == entry.pretrial_predictions.standard_bo_mean_before
        assert entry.physics_bo_mean_before == entry.pretrial_predictions.physics_bo_mean_before
        if trial == 1:
            assert entry.standard_log_score is None
            assert entry.physics_log_score is None
        else:
            assert math.isfinite(entry.standard_log_score)
            assert math.isfinite(entry.physics_log_score)


def test_no_oracle_or_case_label_is_available_to_arbitrator(small_domain):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=4)
    result = run_predictive_evidence_failover(
        environment,
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    assert environment.oracle_access_count == 0
    assert len(set(result.ledger.executed_candidate_ids)) == 4
    source = (
        ROOT / "personalization" / "predictive_failover_v1" / "evidence.py"
    ).read_text(encoding="utf-8")
    for prohibited in ("oracle_value", "oracle_optimum", "future_observation"):
        assert prohibited not in source


def test_post_failover_selection_exactly_equals_standard_bo_current_history(
    small_domain,
):
    case = make_primary_cases()[0]
    result = run_predictive_evidence_failover(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=2),
        small_domain,
        physics_model=PhysicsSubjectModel(
            SmoothTruthAdapter(bias=100.0, scale=-50.0)
        ),
    )
    assert result.failed_over
    checked = 0
    for index, entry in enumerate(result.ledger.entries):
        if entry.mode_after_observation != STANDARD_MODE or index == 3:
            continue
        history = result.ledger.observations[: index + 1]
        standard = StandardGaussianProcess()
        standard.fit(history)
        expected = LowerConfidenceBoundSelector(
            name="Standard BO", kappa=1.5
        ).select_next(history, small_domain, standard)
        assert entry.selected_next_candidate == expected.candidate
        assert entry.acquisition_value == expected.acquisition_value
        checked += 1
    assert checked >= 1
    standard = StandardGaussianProcess()
    standard.fit(result.ledger.observations)
    expected_final = min(
        small_domain,
        key=lambda item: (standard.predict(item).mean, item.candidate_index),
    )
    assert result.model_recommended_final_candidate == expected_final


def test_no_failover_selection_exactly_equals_fixed_physics_bo_current_history(
    small_domain,
):
    case = make_primary_cases()[0]
    result = run_predictive_evidence_failover(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
        small_domain,
        physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
    )
    assert not result.failed_over
    for index, entry in enumerate(result.ledger.entries[:-1]):
        history = result.ledger.observations[: index + 1]
        physics = PhysicsInformedResidualModel(
            PhysicsSubjectModel(SmoothTruthAdapter())
        )
        physics.fit(history)
        expected = LowerConfidenceBoundSelector(
            name="Physics-Informed BO", kappa=1.5
        ).select_next(history, small_domain, physics)
        assert entry.selected_next_candidate == expected.candidate
        assert entry.acquisition_value == expected.acquisition_value


@pytest.mark.parametrize("budget", [3, 5])
def test_primary_method_preserves_k4(small_domain, budget):
    case = make_primary_cases()[0]
    with pytest.raises(ValueError, match="frozen at K=4"):
        run_predictive_evidence_failover(
            AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
            small_domain,
            physics_model=PhysicsSubjectModel(SmoothTruthAdapter()),
            budget=budget,
        )


def test_rom_must_remain_frozen(rom_stage):
    _, domain = rom_stage
    unfrozen = SubjectROMProfile(
        profile_id="UNFROZEN_FAILOVER_TEST",
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
        ROMGatedPredictiveFailoverEpisode(
            episode_id="invalid", profile=unfrozen, domain=domain
        )


def test_failover_package_has_no_robot_stack_imports():
    forbidden = {"hardware", "control", "collection", "safety"}
    package = ROOT / "personalization" / "predictive_failover_v1"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.lstrip(".").split(".")[0]
                if root:
                    imported.add(root)
        assert not (imported & forbidden), (path, imported & forbidden)


def test_rule_artifact_contains_no_case_specific_threshold_tuning():
    payload = json.loads(
        (
            ROOT
            / "personalization"
            / "predictive_failover_v1"
            / "PRIMARY_FAILOVER_RULE_V1.json"
        ).read_text(encoding="utf-8")
    )
    encoded = json.dumps(payload)
    for label in ("P0", "P1", "P2", "P3", "regret", "true_optimum"):
        assert label not in encoded
    assert payload["failover"]["threshold_interpretation"].startswith(
        "log predictive Bayes factor"
    )


def test_formal_benchmark_preserves_baselines_and_reports_limitations():
    results = (
        ROOT
        / "personalization"
        / "benchmarks"
        / "results_predictive_failover_v1"
    )
    payload = json.loads((results / "benchmark_summary.json").read_text())
    assert payload["run_count"] == 960
    assert payload["primary_budget"] == 4
    assert payload["same_rom_fingerprint_for_all_paired_methods"] is True
    assert payload["historical_baseline_equivalence"]["exact_within_1e_12"] is True
    assert payload["historical_baseline_equivalence"]["maximum_absolute_error"] == 0.0
    assert payload["robot_actions"] == payload["human_actions"] == 0
    assert payload["PINN_training"] == 0
    assert payload["implementation_status"].endswith("IMPLEMENTED_WITH_LIMITATIONS")
    assert payload["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"] is False
    assert len(payload["evidence_trajectory_rows"]) == 240 * 4

    informative = [
        row
        for row in payload["failover_summary"]
        if row["prior_quality"] in {"P0", "P1", "P2"}
    ]
    assert sum(row["false_failover_count"] for row in informative) == 0
    assert all(row["failover_rate"] == 0.0 for row in informative)
    assert all(
        row["benefit_retention_fraction"] == pytest.approx(1.0)
        for row in payload["benefit_retention"]
    )

    p3 = {row["noise_label"]: row for row in payload["p3_analysis"]}
    assert [p3[label]["failover_count"] for label in ("zero", "low", "moderate")] == [
        10,
        12,
        11,
    ]
    assert p3["zero"]["failover_improvement_vs_adaptive"] < 0.0
    assert p3["low"]["failover_improvement_vs_adaptive"] < 0.0
    assert p3["moderate"]["failover_improvement_vs_adaptive"] > 0.0
