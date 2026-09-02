from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from personalization.benchmarks.metrics import evaluate_run
from personalization.candidates import V3_MANIFEST, V3_TABLE, Candidate, V3CandidateDomain
from personalization.environment import (
    AnalyticBenchmarkEnvironment,
    FrozenOfflineReplayEnvironment,
    RealRobotEnvironment,
    make_primary_cases,
)
from personalization.models.physics_graybox import (
    AnalyticDevelopmentPhysicsAdapter,
    FullDynamicsGrayBoxEndpointAdapter,
    PhysicsSubjectModel,
)
from personalization.models.residual_gp import (
    PhysicsInformedResidualModel,
    ResidualGaussianProcess,
)
from personalization.models.standard_gp import StandardGaussianProcess
from personalization.observations import EpisodeObservation
from personalization.sequential import METHODS, run_sequential_personalization


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def frozen_domain() -> V3CandidateDomain:
    return V3CandidateDomain.from_frozen_artifact()


@pytest.fixture()
def small_domain() -> V3CandidateDomain:
    return V3CandidateDomain.regular_grid([-0.03, 0.0, 0.03])


def physics(case, quality="P1") -> PhysicsSubjectModel:
    return PhysicsSubjectModel(
        AnalyticDevelopmentPhysicsAdapter(
            optimum_beta=case.optimum_beta,
            prior_quality=quality,
            landscape=case.landscape,
        )
    )


def test_frozen_v3_identity_and_reference(frozen_domain):
    assert len(frozen_domain) == 625
    assert frozen_domain.reference.candidate_id == "MYOLEG_V3_K0312"
    assert frozen_domain.reference.beta == (0.0, 0.0)
    assert min(item.beta_flex for item in frozen_domain) == -0.03
    assert max(item.beta_extend for item in frozen_domain) == 0.03


def test_invalid_observation_cannot_be_filled_with_zero():
    with pytest.raises(ValueError, match="missing endpoint"):
        EpisodeObservation(
            "bad", 1, "c", 0.0, 0.0, "mechanical_cost", 0.0, "u", None,
            False, "invalid",
        )


@pytest.mark.parametrize("budget", [3, 4, 5])
def test_budget_and_reference_cold_start(small_domain, budget):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=7)
    result = run_sequential_personalization(
        environment, small_domain, method="Space Filling", budget=budget
    )
    assert len(result.ledger.entries) == budget
    assert result.ledger.entries[0].candidate == small_domain.reference
    assert len(set(result.ledger.executed_candidate_ids)) == budget


def test_adaptation_never_accesses_future_oracle(small_domain):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=1)
    run_sequential_personalization(
        environment,
        small_domain,
        method="Physics-Informed BO",
        physics_model=physics(case),
    )
    assert environment.oracle_access_count == 0


def test_invalid_episode_excluded_and_consumes_budget(small_domain):
    base = make_primary_cases()[0]
    case = type(base)(
        base.name,
        base.landscape,
        base.optimum_beta,
        base.noise_std,
        frozenset({small_domain.reference.candidate_id}),
    )
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=3)
    result = run_sequential_personalization(
        environment,
        small_domain,
        method="Physics-Informed BO",
        budget=3,
        physics_model=physics(case),
    )
    assert len(result.ledger.entries) == 3
    assert not result.ledger.entries[0].observation.valid
    assert result.ledger.entries[0].observation.endpoint_value is None
    assert result.ledger.entries[0].residual_model_state_summary["training_count"] == 0


def test_random_selector_and_bo_are_deterministic(small_domain):
    case = make_primary_cases(0.03)[0]
    sequences = []
    for _ in range(2):
        environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=11)
        result = run_sequential_personalization(
            environment, small_domain, method="Random", seed=11
        )
        sequences.append(result.ledger.executed_candidate_ids)
    assert sequences[0] == sequences[1]

    bo_sequences = []
    for _ in range(2):
        environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=11)
        result = run_sequential_personalization(
            environment, small_domain, method="Standard BO", seed=11
        )
        bo_sequences.append(result.ledger.executed_candidate_ids)
    assert bo_sequences[0] == bo_sequences[1]


def test_residual_and_total_prediction_identity(small_domain):
    case = make_primary_cases()[0]
    model = PhysicsInformedResidualModel(physics(case))
    candidate = small_domain.reference
    physics_value = model.physics.predict(candidate).mean
    observation = EpisodeObservation(
        "e1", 1, candidate.candidate_id, *candidate.beta,
        "offline_synthetic_mechanical_cost", physics_value + 0.4,
        "normalized_cost", 0.0, True,
    )
    model.fit([observation])
    prediction = model.predict(candidate)
    assert prediction.metadata["residual_mean"] == pytest.approx(0.4, abs=1e-5)
    assert prediction.mean == pytest.approx(
        prediction.metadata["physics_mean"] + prediction.metadata["residual_mean"]
    )


def test_standard_bo_has_no_physics_access(small_domain):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=2)
    result = run_sequential_personalization(
        environment, small_domain, method="Standard BO"
    )
    assert result.final_model_summary["physics_access"] is False


def test_physics_informed_bo_records_only_causal_fit_count(small_domain):
    case = make_primary_cases()[0]
    environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=2)
    result = run_sequential_personalization(
        environment,
        small_domain,
        method="Physics-Informed BO",
        physics_model=physics(case),
    )
    for entry in result.ledger.entries:
        assert entry.physics_model_state_summary["fit_valid_episode_count"] <= entry.trial_index
        assert entry.residual_model_state_summary["training_count"] <= entry.trial_index


def test_bad_prior_zero_noise_and_noisy_cases_do_not_crash(small_domain):
    for noise in (0.0, 0.1):
        case = make_primary_cases(noise)[-1]
        environment = AnalyticBenchmarkEnvironment(small_domain, case, seed=5)
        result = run_sequential_personalization(
            environment,
            small_domain,
            method="Physics-Informed BO",
            physics_model=physics(case, "P3"),
        )
        assert len(result.ledger.entries) == 4
        metrics = evaluate_run(result, environment)
        assert metrics["final_regret"] >= -1e-12
        assert len(metrics["simple_regret_per_trial"]) == 4


def test_replay_reveals_only_executed_candidate(small_domain):
    values = {candidate.candidate_id: float(index) for index, candidate in enumerate(small_domain)}
    environment = FrozenOfflineReplayEnvironment(
        values, endpoint_name="frozen_mechanical_endpoint", endpoint_unit="u"
    )
    result = run_sequential_personalization(
        environment, small_domain, method="Space Filling", budget=3
    )
    assert environment.revealed_candidate_ids == list(result.ledger.executed_candidate_ids)
    assert len(environment.revealed_candidate_ids) == 3


def test_reference_repeat_is_explicit_exception(small_domain):
    case = make_primary_cases()[0]
    result = run_sequential_personalization(
        AnalyticBenchmarkEnvironment(small_domain, case, seed=0),
        small_domain,
        method="Reference",
        budget=4,
    )
    assert result.ledger.executed_candidate_ids == (small_domain.reference.candidate_id,) * 4
    assert result.ledger.duplicate_candidate_policy == "NO_DUPLICATE_CANDIDATE"


def test_real_robot_environment_fails_closed(small_domain):
    with pytest.raises(RuntimeError, match="REAL_ROBOT_ENVIRONMENT_DISABLED.*NOT_ROBOT_APPROVED"):
        RealRobotEnvironment().evaluate(small_domain.reference, 1)


def test_personalization_package_has_no_hardware_or_control_imports():
    forbidden = {"hardware", "control", "collection"}
    for path in (ROOT / "personalization").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden), (path, imported & forbidden)


def test_full_gray_box_adapter_reuses_effective_parameters(frozen_domain):
    adapter = FullDynamicsGrayBoxEndpointAdapter()
    value = adapter.predict_value(frozen_domain.reference)
    metadata = adapter.metadata()
    assert np.isfinite(value) and value > 0.0
    assert metadata["parameter_semantics"] == "effective_gray_box_parameters"
    assert set(metadata["parameter_names"]) == {
        "mass_scale", "k_hip_nm_per_rad", "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad", "b_knee_nm_s_per_rad",
    }
    assert "NOT_FUTURE_REAL_ENDPOINT" in metadata["endpoint_status"]


def test_frozen_artifacts_unchanged_by_algorithm_run(frozen_domain):
    before = {V3_TABLE: sha256(V3_TABLE), V3_MANIFEST: sha256(V3_MANIFEST)}
    case = make_primary_cases()[0]
    for method in METHODS:
        run_sequential_personalization(
            AnalyticBenchmarkEnvironment(frozen_domain, case, seed=0),
            frozen_domain,
            method=method,
            budget=3,
            physics_model=(
                physics(case) if method in {"Model-Only Greedy", "Physics-Informed BO"} else None
            ),
        )
    assert before == {path: sha256(path) for path in before}
