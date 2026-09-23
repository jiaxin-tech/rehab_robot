from pathlib import Path
import shutil
import uuid

from lower_limb_sim.myoleg_benchmark.controlled_positive_cohort import load_profiles
from lower_limb_sim.myoleg_benchmark.controlled_positive_experiment import run_benchmark, run_profile


def test_controlled_policy_gate_and_regret_contract():
    profiles = load_profiles("DEVELOPMENT")
    null = run_profile(next(p for p in profiles if p.scenario == "COMMON_OPTIMUM"), budget=8)
    positive = run_profile(next(p for p in profiles if p.scenario == "DIVERGENT_OPTIMA"), budget=8)
    assert null["gate_status"] == "PERSONALIZATION_INACTIVE"
    assert null["personalized_regret"] == null["common_regret"] == 0.0
    assert positive["gate_status"] == "PERSONALIZATION_ACTIVE"
    assert positive["personalized_regret"] < positive["common_regret"]
    assert positive["recommendation_id"] in {row["candidate_id"] for row in positive["observations"]}
    assert positive["termination"] == "BUDGET_EXHAUSTED"


def test_controlled_policy_writes_auditable_outputs():
    output = Path(".cache") / f"controlled-policy-test-{uuid.uuid4().hex}"
    try:
        run_benchmark(output_dir=output, budget=4, split="DEVELOPMENT")
        assert (output / "protocol.json").is_file()
        assert (output / "results.csv").is_file()
        assert (output / "decision_logs.json").is_file()
        assert (output / "REPORT.md").is_file()
        assert (output / "completion.json").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)
