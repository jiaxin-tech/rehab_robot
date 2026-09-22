from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from lower_limb_sim.five_leg_mujoco_v1.e2_necessity import (
    COMPONENT_FIELDS,
    E2_ENDPOINT_ID,
    STUDY_ID,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "lower_limb_sim" / "five_leg_mujoco_v1"
RESULTS = PACKAGE / "results_e2_necessity_v1"


@pytest.fixture(scope="module")
def payload():
    return json.loads((RESULTS / "study_summary.json").read_text(encoding="utf-8"))


def test_study_is_offline_and_preserves_every_frozen_component(payload):
    assert payload[STUDY_ID] == "COMPLETE"
    assert payload["classification"] == (
        "OFFLINE_SYNTHETIC_MUJOCO_E2_NECESSITY_EVALUATION_ONLY"
    )
    assert payload["source_endpoint_study"] == (
        "DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1"
    )
    assert payload["source_endpoint_study_preserved"] is True
    assert payload["five_leg_parameters_preserved"] is True
    assert payload["ROM_preserved"] is True
    assert payload["V3_preserved"] is True
    assert payload["E2_definition_frozen"] is True
    assert payload["candidate_count_per_leg"] == 625
    assert payload["algorithm_runs"] == payload["robot_actions"] == 0
    assert payload["PINN_training"] == 0


def test_complete_landscape_uses_frozen_reference_normalized_max_definition(payload):
    assert payload["E2_endpoint_id"] == E2_ENDPOINT_ID
    assert payload["E2_source_definition_name"] == "BRANCH_BALANCED_RELATIVE_RMS"
    assert payload["E2_definition"]["formula"] == (
        "max(r_hip_flex_RMS, r_hip_extend_RMS, "
        "r_knee_flex_RMS, r_knee_extend_RMS)"
    )
    rows = payload["E2_landscape_rows"]
    assert len(rows) == 5 * 625
    assert {row["leg_id"] for row in rows} == {
        "LEG_0_NOMINAL",
        "LEG_1_HEAVY_HIP_STIFF",
        "LEG_2_KNEE_DOMINANT",
        "LEG_3_NONLINEAR_COUPLED",
        "LEG_4_STRONG_STRUCTURAL_MISMATCH",
    }
    labels = [field.removeprefix("relative_").removesuffix("_rms") for field in COMPONENT_FIELDS]
    for row in rows:
        assert row["E2"] == pytest.approx(max(row[label] for label in labels))
    references = [
        row for row in rows if row["beta_flex"] == row["beta_extend"] == 0.0
    ]
    assert len(references) == 5
    assert all(row["E2"] == pytest.approx(1.0) for row in references)


def test_e2_oracles_and_near_oracle_counts_are_characterized(payload):
    rows = {row["leg_id"]: row for row in payload["E2_oracle_characterization"]}
    assert len(rows) == 5
    for leg_id in (
        "LEG_0_NOMINAL",
        "LEG_1_HEAVY_HIP_STIFF",
        "LEG_2_KNEE_DOMINANT",
        "LEG_3_NONLINEAR_COUPLED",
    ):
        assert rows[leg_id]["oracle_beta"] == [0.0, 0.0]
        assert rows[leg_id]["oracle_value"] == pytest.approx(1.0)
        assert rows[leg_id]["reference_to_oracle_improvement"] == 0.0
    exceptional = rows["LEG_4_STRONG_STRUCTURAL_MISMATCH"]
    assert exceptional["oracle_beta"] == [-0.03, 0.0225]
    assert exceptional["oracle_value"] == pytest.approx(0.9992936589002313)
    assert exceptional["reference_to_oracle_improvement_relative"] == pytest.approx(
        0.0007068403701731008
    )
    expected_near_counts = {
        "LEG_0_NOMINAL": [182, 304, 500, 625, 625],
        "LEG_1_HEAVY_HIP_STIFF": [182, 323, 525, 625, 625],
        "LEG_2_KNEE_DOMINANT": [10, 247, 525, 625, 625],
        "LEG_3_NONLINEAR_COUPLED": [182, 323, 525, 625, 625],
        "LEG_4_STRONG_STRUCTURAL_MISMATCH": [169, 340, 575, 625, 625],
    }
    keys = [
        "within_0.1_percent_oracle_count",
        "within_0.5_percent_oracle_count",
        "within_1_percent_oracle_count",
        "within_2_percent_oracle_count",
        "within_5_percent_oracle_count",
    ]
    assert {
        leg_id: [row[key] for key in keys] for leg_id, row in rows.items()
    } == expected_near_counts


def test_oracle_diversity_and_full_cross_leg_transfer_are_small_in_decision_terms(payload):
    diversity = payload["E2_oracle_diversity"]
    assert diversity["unique_oracle_count"] == 2
    assert diversity["diversity_alone_does_not_establish_personalization"] is True
    transfer = payload["E2_oracle_transfer_matrix"]
    assert len(transfer) == 25
    assert len({(row["source_leg"], row["target_leg"]) for row in transfer}) == 25
    nonzero = [row for row in transfer if row["relative_regret"] > 1.0e-12]
    assert len(nonzero) == 8
    assert max(row["relative_regret"] for row in transfer) == pytest.approx(
        0.00528568713334554
    )


def test_common_candidate_has_tiny_regret_and_large_universal_near_sets(payload):
    common = payload["E2_common_candidate"]
    assert common["common_candidate_index"] == 312
    assert common["common_beta"] == [0.0, 0.0]
    assert common["mean_relative_common_regret"] == pytest.approx(
        0.00014136807403462016
    )
    assert common["maximum_relative_common_regret"] == pytest.approx(
        0.0007068403701731008
    )
    assert [row["relative_common_regret"] for row in common["per_leg"]] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0007068403701731008]
    )
    assert [
        common[f"universal_within_{tolerance}_percent_count"]
        for tolerance in ("0.1", "0.5", "1", "2", "5")
    ] == [8, 228, 500, 625, 625]


def test_pairwise_ordering_interaction_and_branch_drivers_are_retained(payload):
    pairwise = payload["E0_E2_pairwise_landscape_similarity"]
    assert len(pairwise) == 20
    assert sum(row["endpoint_id"] == "E0" for row in pairwise) == 10
    assert sum(row["endpoint_id"] == "E2" for row in pairwise) == 10
    comparison = {row["endpoint_id"]: row for row in payload["E0_vs_E2_comparison"]}
    assert comparison["E0"]["minimum_pairwise_spearman"] == pytest.approx(
        0.991982157778324
    )
    assert comparison["E2"]["minimum_pairwise_spearman"] == pytest.approx(
        0.8146590115287
    )
    assert comparison["E0"]["normalized_interaction_energy_fraction"] == pytest.approx(
        0.0035999127178301124
    )
    assert comparison["E2"]["normalized_interaction_energy_fraction"] == pytest.approx(
        0.05228441639192297
    )
    drivers = {row["leg_id"]: row for row in payload["E2_branch_driver_summary"]}
    assert drivers["LEG_0_NOMINAL"]["oracle_active_worst_components"] == [
        "hip_flexion",
        "hip_extension",
        "knee_flexion",
        "knee_extension",
    ]
    assert drivers["LEG_4_STRONG_STRUCTURAL_MISMATCH"][
        "oracle_active_worst_components"
    ] == ["hip_flexion"]


def test_robustness_aware_decision_does_not_overclaim_personalization(payload):
    robust = payload["robustness_aware_necessity"]
    assert robust["not_a_validated_sensor_noise_model"] is True
    assert robust["minimum_E2_ranking_spearman"] > 0.999
    assert robust["E2_exact_oracle_stability_fraction"] == 1.0
    assert robust["maximum_E2_relative_endpoint_variability"] == pytest.approx(
        0.00018515195743713596
    )
    assert robust["maximum_transfer_regret_to_variability_ratio"] > 28.5
    assert robust["mean_common_regret_to_maximum_variability_ratio"] < 1.0
    assert payload["E2_PERSONALIZATION_NECESSITY"] == "NOT_SUPPORTED"
    assert payload["READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON"] is False


def test_outputs_exist_and_study_code_has_no_forbidden_runtime_imports(payload):
    required = {
        "study_summary.json",
        "e2_full_landscapes.csv",
        "table_1_e2_oracle_characterization.csv",
        "table_2_e2_oracle_transfer_matrix.csv",
        "table_3_e2_common_candidate_per_leg.csv",
        "table_4_e2_branch_driver_summary.csv",
        "table_5_e2_branch_driver_rows.csv",
        "table_6_e0_vs_e2_comparison.csv",
        "table_7_common_regret_vs_perturbation.csv",
        "table_8_e0_e2_pairwise_landscape_similarity.csv",
    }
    assert all((RESULTS / filename).stat().st_size > 0 for filename in required)
    forbidden_import_roots = {"hardware", "control", "collection", "safety"}
    forbidden_fragments = (
        "StandardGaussianProcess",
        "LowerConfidenceBoundSelector",
        "adaptive_trust_v1",
        "predictive_failover_v1",
        "active_diagnostic_v1",
    )
    for filename in ("e2_necessity.py", "run_e2_necessity.py"):
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_fragments)
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])
        assert not imported & forbidden_import_roots
