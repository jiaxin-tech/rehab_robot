from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from lower_limb_sim.five_leg_mujoco_v1.endpoint_design import (
    ENDPOINT_DEFINITIONS,
    FEATURE_VECTOR_FIELDS,
    FEATURE_VECTOR_ID,
    STUDY_ID,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "lower_limb_sim" / "five_leg_mujoco_v1"
RESULTS = PACKAGE / "results_endpoint_design_v1"


@pytest.fixture(scope="module")
def payload():
    return json.loads((RESULTS / "study_summary.json").read_text(encoding="utf-8"))


def test_study_preserves_frozen_scope_and_predefines_unweighted_endpoints(payload):
    assert payload[STUDY_ID] == "COMPLETE"
    assert payload["source_audit"] == "V3_TRAJECTORY_OBJECTIVE_DISCRIMINABILITY_AUDIT_V1"
    assert payload["source_audit_preserved"] is True
    assert payload["five_leg_parameters_preserved"] is True
    assert payload["ROM_preserved"] is True
    assert payload["V3_preserved"] is True
    assert payload["E0_historical_definition_preserved"] is True
    assert payload["continuous_weight_optimization"] is False
    assert payload["endpoint_definitions_fixed_before_oracle_characterization"] is True
    assert payload["endpoint_definitions"] == ENDPOINT_DEFINITIONS
    assert payload["algorithm_runs"] == payload["robot_actions"] == payload["PINN_training"] == 0


def test_feature_vector_retains_joint_branch_rms_and_peaks(payload):
    feature = payload["feature_vector"]
    assert feature["feature_vector_id"] == FEATURE_VECTOR_ID
    assert feature["ordered_fields"] == list(FEATURE_VECTOR_FIELDS)
    assert feature["normalization"] == "each leg's own beta=(0,0) reference"
    assert feature["cohort_mean_normalization"] is False
    rows = payload["feature_rows"]
    assert len(rows) == 5 * 625
    references = [
        row for row in rows if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
    ]
    assert len(references) == 5
    assert all(row["feature_vector"] == pytest.approx([1.0] * 8) for row in references)
    for row in references:
        assert row["endpoint_E1"] == pytest.approx(1.0)
        assert row["endpoint_E2"] == pytest.approx(1.0)
        assert row["endpoint_E3"] == pytest.approx(1.0)
        assert row["endpoint_E4"] == pytest.approx(1.0)


def test_e1_and_e2_reduce_extreme_E0_compression_without_tuned_weights(payload):
    by_endpoint = {
        endpoint: [
            row for row in payload["endpoint_discriminability"] if row["endpoint_id"] == endpoint
        ]
        for endpoint in ENDPOINT_DEFINITIONS
    }
    assert min(row["relative_range_by_reference"] for row in by_endpoint["E0"]) < 0.0004
    assert min(row["relative_range_by_reference"] for row in by_endpoint["E1"]) > 0.0088
    assert min(row["relative_range_by_reference"] for row in by_endpoint["E2"]) > 0.011
    assert max(row["relative_range_by_reference"] for row in by_endpoint["E3"]) < 0.0005
    assert max(row["relative_range_by_reference"] for row in by_endpoint["E4"]) < 0.0005
    assert all(row["within_0.1_percent_optimum_count"] == 625 for row in by_endpoint["E3"])
    assert all(row["within_0.1_percent_optimum_count"] == 625 for row in by_endpoint["E4"])


def test_e1_interaction_is_decision_relevant_and_mechanical_not_weight_tuned(payload):
    cross = {row["endpoint_id"]: row for row in payload["cross_leg_decision_structure"]}
    assert cross["E0"]["normalized_interaction_energy_fraction"] < 0.004
    assert cross["E1"]["normalized_interaction_energy_fraction"] > 0.08
    assert cross["E1"]["minimum_pairwise_spearman"] < 0.41
    assert cross["E1"]["unique_oracle_count"] == 5
    assert cross["E2"]["unique_oracle_count"] == 2
    assert payload["selection_not_based_on_oracle_diversity"] is True
    assert all(
        row["arbitrary_continuous_weights"] is False
        for row in payload["endpoint_selection_characterization"]
    )


def test_anti_cancellation_examples_show_E0_hiding_joint_and_branch_tradeoffs(payload):
    summaries = payload["anti_cancellation_summary"]
    assert len(summaries) == 5
    assert sum(row["joint_tradeoff_count"] for row in summaries) > 2300
    assert sum(row["branch_tradeoff_count"] for row in summaries) == 5 * 312
    assert sum(
        row["joint_tradeoff_hidden_inside_E0_0_1_percent_count"] for row in summaries
    ) > 1300
    assert sum(
        row["branch_tradeoff_hidden_inside_E0_0_1_percent_count"] for row in summaries
    ) == 1173
    examples = payload["anti_cancellation_representatives"]
    assert len(examples) == 10
    assert all(abs(row["E0_relative"] - 1.0) <= 0.001 for row in examples)
    assert all(row["anti_cancellation_gap"] > 0.0 for row in examples)


def test_rms_endpoints_are_robust_but_peak_rankings_are_not(payload):
    robust = {row["endpoint_id"]: row for row in payload["robustness_summary"]}
    assert robust["BASELINE_RECOMPUTE_CHECK"]["maximum_E0_recompute_abs_error_nm"] == 0.0
    assert robust["E1"]["minimum_ranking_spearman"] > 0.9999
    assert robust["E1"]["exact_oracle_stability_fraction"] == pytest.approx(0.95)
    assert robust["E1"]["maximum_baseline_relative_regret_of_perturbed_oracle"] < 1.3e-5
    assert robust["E2"]["minimum_ranking_spearman"] > 0.999
    assert robust["E3"]["minimum_ranking_spearman"] < -0.8
    assert robust["E4"]["minimum_ranking_spearman"] < -0.8
    assert robust["E3"]["not_a_validated_sensor_noise_model"] is True
    assert robust["E4"]["not_a_validated_sensor_noise_model"] is True


def test_selection_and_measurement_boundary_are_explicit(payload):
    assert payload["MECHANICAL_ENDPOINT_REDESIGN_V1"] == "SUPPORTED_WITH_LIMITATIONS"
    assert payload["RECOMMENDED_PRIMARY_ENDPOINT"] == "E2"
    assert payload["READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY"] is True
    statuses = {row["current_status"] for row in payload["measurement_compatibility"]}
    assert statuses == {"DIRECTLY_MEASURABLE", "MODEL_DERIVED", "FUTURE_SENSOR_REQUIRED"}
    assert all(not row["real_world_validated"] for row in payload["measurement_compatibility"])


def test_outputs_exist_and_new_code_has_no_personalization_or_robot_runtime_imports(payload):
    required = {
        "study_summary.json",
        "mechanical_endpoint_feature_vector_v1.csv",
        "table_1_endpoint_discriminability.csv",
        "table_2_cross_leg_decision_structure.csv",
        "table_4_anti_cancellation_summary.csv",
        "table_7_robustness_scenarios.csv",
        "table_10_measurement_compatibility.csv",
        "table_11_endpoint_selection_characterization.csv",
    }
    assert all((RESULTS / filename).stat().st_size > 0 for filename in required)
    forbidden_import_roots = {"hardware", "control", "collection", "safety"}
    forbidden_algorithm_fragments = (
        "adaptive_trust_v1",
        "predictive_failover_v1",
        "active_diagnostic_v1",
        "StandardGaussianProcess",
        "LowerConfidenceBoundSelector",
    )
    for filename in ("endpoint_design.py", "run_endpoint_design.py"):
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_algorithm_fragments)
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])
        assert not imported & forbidden_import_roots
