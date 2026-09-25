import json

import pandas as pd

from lower_limb_sim.myoleg_benchmark.controlled_resistance_algorithm_comparison import (
    METHODS, run_benchmark, run_profile,
)
from lower_limb_sim.myoleg_benchmark.controlled_resistance_cohort import build_profiles


def test_all_methods_are_observation_only_and_recommend_observed_candidates():
    profile = next(p for p in build_profiles() if p.arm == "EARLY" and p.split == "DEVELOPMENT")
    for method in METHODS:
        result = run_profile(profile, method, budget=8, seed=0)
        assert result["recommendation_id"] is None or result["executed_trials"] >= 5
        assert result["policy_regret"] is None or result["policy_regret"] >= -1e-12
        assert result["oracle_id"]


def test_algorithm_comparison_has_null_and_positive_rows_without_confirmatory_access(tmp_path):
    output = run_benchmark(output_dir=tmp_path / "algorithm")
    protocol = json.loads((output / "protocol.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    frame = pd.read_csv(output / "results.csv")
    assert set(frame.method) == set(METHODS)
    assert set(frame.scenario) == {"COMMON_RESISTANCE", "DIVERGENT_RESISTANCE"}
    assert protocol["confirmatory_access"] is False
    assert completion["confirmatory_access"] is False
