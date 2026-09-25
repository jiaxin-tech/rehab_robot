import json

import pandas as pd

from lower_limb_sim.myoleg_benchmark.controlled_resistance_experiment import run_profile, run_benchmark
from lower_limb_sim.myoleg_benchmark.controlled_resistance_cohort import build_profiles


def test_sast_resistance_profile_uses_only_development_and_returns_auditable_result():
    profile = next(p for p in build_profiles() if p.arm == "EARLY" and p.split == "DEVELOPMENT")
    result = run_profile(profile, budget=8)
    assert result["executed_trials"] == 8
    assert result["oracle_id"]
    assert result["recommendation_id"] in {None, *(o["candidate_id"] for o in result["observations"])}
    assert result["gate_status"] in {"PERSONALIZATION_ACTIVE", "PERSONALIZATION_INACTIVE"}


def test_development_pilot_writes_no_confirmatory_access(tmp_path):
    output = run_benchmark(output_dir=tmp_path / "resistance")
    protocol = json.loads((output / "protocol.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    assert completion["complete"] is True
    assert completion["confirmatory_access"] is False
    assert len(pd.read_csv(output / "results.csv")) == 6
    assert protocol["candidate_count"] == 27
    decision = json.loads((output / "decision.json").read_text())
    assert decision["null_gate_active_rate"] == 0
    assert decision["confirmatory_ready"] is False
