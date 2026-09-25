from __future__ import annotations

import json

import pytest

from lower_limb_sim.myoleg_benchmark.controlled_resistance_algorithm_comparison import (
    METHODS,
    run_benchmark,
    run_profile,
)
from lower_limb_sim.myoleg_benchmark.controlled_resistance_cohort import load_profiles


def test_comparison_methods_use_frozen_names_and_include_pure_ei() -> None:
    assert METHODS == ("COMMON_POLICY", "RANDOM", "RESIDUAL_GREEDY", "PURE_EI", "SAST_BO")
    assert "RESIDUAL_BO" not in METHODS


def test_common_policy_recommends_reference_after_calibration() -> None:
    profile = load_profiles("DEVELOPMENT")[2]
    result = run_profile(profile, "COMMON_POLICY", budget=8)
    assert result["recommendation_id"] == "resist:13"
    assert result["executed_trials"] == 5
    assert [row["trial_index"] for row in result["observations"]] == [1, 2, 3, 4, 5]
    assert result["policy_regret"] == pytest.approx(result["common_regret"])
    assert result["improvement_over_common"] == pytest.approx(0.0)
    assert result["termination"] == "COMMON_POLICY_FIXED"


@pytest.mark.parametrize("method", ["RANDOM", "RESIDUAL_GREEDY", "PURE_EI", "SAST_BO"])
def test_methods_record_only_unique_observed_candidates(method: str) -> None:
    profile = load_profiles("DEVELOPMENT")[2]
    result = run_profile(profile, method, budget=8, seed=0)
    observed_ids = [row["candidate_id"] for row in result["observations"]]
    assert len(observed_ids) == len(set(observed_ids))
    assert len(observed_ids) == result["executed_trials"]
    assert result["recommendation_id"] in set(observed_ids) or result["recommendation_id"] is None
    assert result["oracle_id"] not in {"__learner_input__"}


def test_benchmark_persists_sequences_separately_from_scalar_results(tmp_path) -> None:
    destination = run_benchmark(output_dir=tmp_path / "comparison", budget=5)
    results_header = (destination / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "observations" not in results_header
    assert "decisions" not in results_header
    logs = json.loads((destination / "decision_logs.json").read_text(encoding="utf-8"))
    assert logs
    assert all(row["observations"] for row in logs)
    assert {row["method"] for row in logs} == set(METHODS)


def test_k4_uses_two_calibration_probes_and_one_adaptive_trial() -> None:
    profile = load_profiles("DEVELOPMENT")[2]
    result = run_profile(profile, "PURE_EI", budget=4)
    assert result["executed_trials"] == 4
    assert [row["trial_index"] for row in result["observations"][:3]] == [1, 2, 3]
    assert result["decisions"][2]["mode"] == "CALIBRATION"
    assert result["decisions"][3]["mode"] == "PURE_EI"
