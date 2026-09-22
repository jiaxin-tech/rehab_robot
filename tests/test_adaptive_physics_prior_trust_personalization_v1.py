from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from personalization.adaptive_trust_v1 import (
    EXPECTED_RULE_SHA256,
    FixedTrustEstimator,
    PhysicsPriorTrustEstimator,
    PhysicsPriorTrustEvidence,
    PhysicsPriorTrustState,
    ROMGatedAdaptiveTrustEpisode,
    run_adaptive_trust_personalization,
    verify_frozen_rule,
)
from personalization.candidates import V3CandidateDomain
from personalization.environment import AnalyticBenchmarkEnvironment, make_primary_cases
from personalization.models.physics_graybox import (
    AnalyticDevelopmentPhysicsAdapter,
    PhysicsSubjectModel,
)
from personalization.rom_gated_v2 import (
    ROMGatedPersonalizationEpisode,
    SubjectROMProfile,
    SubjectSpecificV3CandidateDomain,
    assert_same_frozen_rom_comparison,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
)
from personalization.sequential import run_sequential_personalization


ROOT = Path(__file__).resolve().parents[1]


def physics(case, quality="P1") -> PhysicsSubjectModel:
    return PhysicsSubjectModel(
        AnalyticDevelopmentPhysicsAdapter(
            optimum_beta=case.optimum_beta,
            prior_quality=quality,
            landscape=case.landscape,
        )
    )


def update_sequence(
    pairs: list[tuple[float, float]],
    *,
    uncertainty: float = 0.0,
) -> tuple[PhysicsPriorTrustState, tuple[PhysicsPriorTrustEvidence, ...]]:
    estimator = PhysicsPriorTrustEstimator()
    state = estimator.initial_state()
    evidence = []
    for trial_index, (observed, predicted) in enumerate(pairs, start=1):
        evidence.append(
            PhysicsPriorTrustEvidence(
                trial_index=trial_index,
                candidate_id=f"c{trial_index}",
                beta_flex=0.0025 * trial_index,
                beta_extend=-0.0025 * trial_index,
                observed_value=observed,
                physics_prediction=predicted,
                observation_uncertainty=uncertainty,
            )
        )
        state = estimator.update(
            state,
            tuple(evidence),
            trial_index=trial_index,
            current_observation_valid=True,
        )
    return state, tuple(evidence)


@pytest.fixture(scope="module")
def small_domain() -> V3CandidateDomain:
    return V3CandidateDomain.regular_grid([-0.03, 0.0, 0.03])


@pytest.fixture(scope="module")
def rom_stage():
    case = make_offline_rom_development_cases()[1]
    _, profile = determine_synthetic_rom(case)
    domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)
    return profile, domain


def test_primary_rule_artifact_is_frozen_before_benchmark():
    payload = verify_frozen_rule()
    assert payload["primary_rule_id"] == (
        "PRIMARY_CAUSAL_OFFSET_INVARIANT_RANKING_TRUST_V1"
    )
    assert EXPECTED_RULE_SHA256 == (
        "326007fe40d752f52d411d04d2c732ed3a2a1ced543d618ac817af683813c67d"
    )
    assert payload["initial_trust_policy"]["trust_score"] == 1.0


def test_trust_score_is_bounded_versioned_and_immutable():
    state, _ = update_sequence([(0.0, 100.0), (2.0, -50.0), (-3.0, 25.0)])
    assert 0.0 <= state.trust_score <= 1.0
    assert state.metadata["version"] == 1
    assert state.metadata["semantic_scope"] == (
        "ALGORITHMIC_PHYSICS_PRIOR_TRUST_WEIGHT"
    )
    assert state.metadata["not_biological_probability"] is True
    with pytest.raises(FrozenInstanceError):
        state.trust_score = 0.5


def test_trust_update_is_deterministic():
    first, _ = update_sequence([(0.1, 0.0), (0.4, 0.2), (-0.2, 0.5)])
    second, _ = update_sequence([(0.1, 0.0), (0.4, 0.2), (-0.2, 0.5)])
    assert first.as_dict() == second.as_dict()


def test_future_evidence_is_rejected_and_rule_has_no_case_label_input():
    estimator = PhysicsPriorTrustEstimator()
    state = estimator.initial_state()
    future = PhysicsPriorTrustEvidence(2, "future", 0.0, 0.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="future trust evidence is prohibited"):
        estimator.update(
            state,
            (future,),
            trial_index=1,
            current_observation_valid=True,
        )
    source = (ROOT / "personalization" / "adaptive_trust_v1" / "trust.py").read_text()
    for label in ("P0", "P1", "P2", "P3", "optimum_beta", "oracle_value"):
        assert label not in source


def test_constant_magnitude_bias_does_not_remove_trust():
    state, _ = update_sequence([(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)])
    assert state.prediction_error_metrics["estimated_constant_offset"] == -2.0
    assert state.prediction_error_metrics["offset_removed_nrmse"] == 0.0
    assert state.ranking_consistency_metrics["discordant_pair_count"] == 0
    assert state.trust_score == pytest.approx(0.90625)


def test_scaling_bias_preserves_substantial_ranking_trust():
    state, _ = update_sequence([(0.0, -0.7), (1.0, 1.1), (2.0, 2.9)])
    assert state.ranking_consistency_metrics["concordant_pair_count"] == 3
    assert state.ranking_consistency_metrics["discordant_pair_count"] == 0
    assert state.trust_score > 0.8


def test_ranking_inversion_decreases_trust():
    state, _ = update_sequence([(0.0, 0.0), (1.0, -1.0), (2.0, -2.0)])
    assert state.ranking_consistency_metrics["discordant_pair_count"] == 3
    assert state.ranking_consistency_metrics["concordant_pair_count"] == 0
    assert state.trust_score < 0.3


def test_one_noisy_point_cannot_catastrophically_collapse_trust():
    state, _ = update_sequence([(100.0, -100.0)], uncertainty=10.0)
    assert state.evidence_count == 1
    assert state.ranking_consistency_metrics["comparable_pair_count"] == 0
    assert state.trust_score == 1.0


def test_invalid_observation_consumes_trial_but_adds_no_false_evidence(small_domain):
    base = make_primary_cases()[0]
    invalid_case = type(base)(
        base.name,
        base.landscape,
        base.optimum_beta,
        base.noise_std,
        frozenset({small_domain.reference.candidate_id}),
    )
    result = run_adaptive_trust_personalization(
        AnalyticBenchmarkEnvironment(small_domain, invalid_case, seed=3),
        small_domain,
        physics_model=physics(base),
    )
    first = result.ledger.entries[0]
    assert not first.observation.valid
    assert first.observation.endpoint_value is None
    assert first.physics_prior_trust_before == first.physics_prior_trust_after == 1.0
    assert first.trust_evidence_summary["evidence_count"] == 0
    assert "INVALID_OBSERVATION_IGNORED" in first.trust_evidence_summary["update_reason"]
    assert len(result.ledger.entries) == 4


def test_trust_zero_exactly_recovers_standard_bo(small_domain):
    case = make_primary_cases()[0]
    adaptive = run_adaptive_trust_personalization(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=2),
        small_domain,
        physics_model=physics(case),
        trust_estimator=FixedTrustEstimator(0.0),
    )
    standard = run_sequential_personalization(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=2),
        small_domain,
        method="Standard BO",
    )
    assert adaptive.ledger.executed_candidate_ids == standard.ledger.executed_candidate_ids
    assert (
        adaptive.model_recommended_final_candidate.candidate_id
        == standard.model_recommended_final_candidate.candidate_id
    )
    assert adaptive.final_model_summary["limit_semantics"]["trust_0"] == (
        "EXACT_STANDARD_BO"
    )


def test_trust_one_exactly_recovers_fixed_physics_bo(small_domain):
    case = make_primary_cases()[0]
    adaptive = run_adaptive_trust_personalization(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=2),
        small_domain,
        physics_model=physics(case),
        trust_estimator=FixedTrustEstimator(1.0),
    )
    fixed = run_sequential_personalization(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=2),
        small_domain,
        method="Physics-Informed BO",
        physics_model=physics(case),
    )
    assert adaptive.ledger.executed_candidate_ids == fixed.ledger.executed_candidate_ids
    assert (
        adaptive.model_recommended_final_candidate.candidate_id
        == fixed.model_recommended_final_candidate.candidate_id
    )
    assert adaptive.final_model_summary["limit_semantics"]["trust_1"] == (
        "EXACT_FIXED_PHYSICS_INFORMED_BO"
    )


def test_adaptation_has_no_oracle_access_and_no_duplicate_candidates(small_domain):
    case = make_primary_cases()[2]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=7)
    result = run_adaptive_trust_personalization(
        environment,
        small_domain,
        physics_model=physics(case, "P3"),
    )
    assert environment.oracle_access_count == 0
    assert len(result.ledger.executed_candidate_ids) == 4
    assert len(set(result.ledger.executed_candidate_ids)) == 4


@pytest.mark.parametrize("budget", [3, 5])
def test_primary_adaptive_method_enforces_k4(small_domain, budget):
    case = make_primary_cases()[0]
    with pytest.raises(ValueError, match="frozen at K=4"):
        run_adaptive_trust_personalization(
            AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
            small_domain,
            physics_model=physics(case),
            budget=budget,
        )


def test_rom_must_be_frozen_before_adaptive_episode(rom_stage):
    _, domain = rom_stage
    unfrozen = SubjectROMProfile(
        profile_id="UNFROZEN_TRUST_TEST",
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
        ROMGatedAdaptiveTrustEpisode(
            episode_id="invalid", profile=unfrozen, domain=domain
        )


def test_same_rom_fingerprint_across_standard_fixed_and_adaptive(rom_stage):
    profile, domain = rom_stage
    case = make_primary_cases()[0]
    standard = ROMGatedPersonalizationEpisode(
        episode_id="standard", profile=profile, domain=domain
    ).run(
        AnalyticBenchmarkEnvironment(domain, case, seed=0),
        method="Standard BO",
    )
    fixed = ROMGatedPersonalizationEpisode(
        episode_id="fixed", profile=profile, domain=domain
    ).run(
        AnalyticBenchmarkEnvironment(domain, case, seed=0),
        method="Physics-Informed BO",
        physics_model=physics(case),
    )
    adaptive = ROMGatedAdaptiveTrustEpisode(
        episode_id="adaptive", profile=profile, domain=domain
    ).run(
        AnalyticBenchmarkEnvironment(domain, case, seed=0),
        physics_model=physics(case),
    )
    results = {"standard": standard, "fixed": fixed, "adaptive": adaptive}
    assert assert_same_frozen_rom_comparison(results) == profile.fingerprint
    assert all(result.rom_profile_fingerprint == profile.fingerprint for result in results.values())


def test_full_causal_trust_fields_extend_one_personalization_ledger(small_domain):
    case = make_primary_cases()[0]
    result = run_adaptive_trust_personalization(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=1),
        small_domain,
        physics_model=physics(case),
    )
    previous_after = 1.0
    for index, entry in enumerate(result.ledger.entries, start=1):
        assert entry.trial_index == index
        assert entry.physics_prior_trust_before == previous_after
        assert entry.physics_prior_trust_after == result.trust_states[index - 1].trust_score
        assert entry.trust_evidence_summary["evidence_count"] <= index
        if index < 4:
            assert entry.selected_next_candidate == result.ledger.entries[index].candidate
            assert entry.trust_evidence_summary["used_to_select_next_trial"] is True
        else:
            assert entry.selected_next_candidate is None
            assert entry.trust_evidence_summary["used_to_select_next_trial"] is False
        previous_after = entry.physics_prior_trust_after
    payload = result.as_dict()
    assert "physics_prior_trust_before" in payload["ledger"]["entries"][0]
    assert "physics_prior_trust_after" in payload["ledger"]["entries"][0]
    assert "trust_evidence_summary" in payload["ledger"]["entries"][0]


def test_adaptive_package_has_no_robot_control_collection_or_safety_imports():
    forbidden = {"hardware", "control", "collection", "safety"}
    for path in (ROOT / "personalization" / "adaptive_trust_v1").rglob("*.py"):
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


def test_formal_benchmark_artifacts_preserve_frozen_rule_and_limitations():
    results = ROOT / "personalization" / "benchmarks" / "results_adaptive_trust_v1"
    payload = json.loads((results / "benchmark_summary.json").read_text())
    assert payload["primary_rule_sha256"] == EXPECTED_RULE_SHA256
    assert payload["run_count"] == 720
    assert payload["primary_budget"] == 4
    assert payload["same_rom_fingerprint_for_all_paired_methods"] is True
    assert payload["historical_V1_baseline_equivalence"] == {
        "compared_metric_count": 96,
        "exact_within_1e_12": True,
        "historical_summary_path": (
            "personalization/benchmarks/results_v1/benchmark_summary.json"
        ),
        "maximum_absolute_error": 0.0,
    }
    assert payload["causal_audit"][
        "environment_oracle_access_before_post_run_evaluation"
    ] == 0
    assert payload["robot_actions"] == payload["human_actions"] == 0
    assert payload["PINN_training"] == 0
    assert payload["adaptive_worse_than_both_count"] == 17
    assert payload["implementation_status"].endswith("IMPLEMENTED_WITH_LIMITATIONS")
    assert payload["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"] is False

    p3_moderate = next(
        row
        for row in payload["negative_transfer"]
        if row["prior_quality"] == "P3" and row["noise_label"] == "moderate"
    )
    assert p3_moderate["adaptive_negative_transfer_mean"] > (
        p3_moderate["fixed_negative_transfer_mean"]
    )

    figure_names = sorted(path.name for path in results.glob("figure_*.png"))
    assert figure_names == [
        "figure_1_p1_regret_and_trust.png",
        "figure_2_p3_method_comparison.png",
        "figure_3_prior_quality_regret_trust.png",
    ]

    for line in (results / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", maxsplit=1)
        assert hashlib.sha256((results / name).read_bytes()).hexdigest() == expected


def test_frozen_adversarial_audit_has_expected_trust_semantics():
    path = (
        ROOT
        / "personalization"
        / "benchmarks"
        / "results_adaptive_trust_v1"
        / "adversarial_trust_audit.json"
    )
    scenarios = json.loads(path.read_text())["scenarios"]
    assert scenarios["magnitude_bias_only"]["final_trust"] == pytest.approx(0.90625)
    assert scenarios["scaling_bias"]["final_trust"] > 0.8
    assert scenarios["ranking_inversion"]["final_trust"] < 0.3
    assert scenarios["one_noisy_point"]["final_trust"] == 1.0
    assert scenarios["invalid_point"]["final_trust"] == 1.0
