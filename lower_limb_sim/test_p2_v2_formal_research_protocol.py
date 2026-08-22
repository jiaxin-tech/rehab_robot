from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from .continuous_reference_neighborhood import OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
)
from .p2_v2_formal_research_protocol import (
    CUMULATIVE_RULE_ID,
    DESIGN_STATUS,
    FORMAL_DESIGN_ID,
    LOCAL_PROTOCOL_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_STATUS,
    PAIR_OUTCOME_STATUS,
    PAIRS_PER_LOCATION_CLASS,
    STOPPING_RULE_ID,
    attach_designated_local_validation_outcomes,
    build_designated_local_validation_pair_plan,
    cumulative_decision_rule_protocol,
    decision_value_exploration_stopping_protocol,
    designated_local_validation_protocol,
    enumerate_designated_local_pair_universe,
    minimum_p2_v2_revision_set,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .run_p2_revision_root_cause_audit import POLICY_ARTIFACT_DIRECTORY
from .run_p2_v2_formal_research_protocol import (
    CORE_SOURCE_PATH,
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    FIGURE_FILENAMES,
    JSON_FILENAMES,
    REPORT_FILENAMES,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def formal_lattice() -> pd.DataFrame:
    return geometrically_valid_parameter_lattice(
        pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    )


@pytest.fixture(scope="module")
def pair_plan() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "designated_local_validation_pair_plan.csv"
    )


@pytest.fixture(scope="module")
def strata() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "designated_local_validation_strata.csv"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _top_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_protocol_identifiers_are_exact() -> None:
    assert FORMAL_DESIGN_ID == "P2_V2_FORMAL_RESEARCH_PROTOCOL_DESIGN_V1"
    assert LOCAL_PROTOCOL_ID == "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1"
    assert CUMULATIVE_RULE_ID == "CUMULATIVE_DECISION_RULE_V1"
    assert STOPPING_RULE_ID == "DECISION_VALUE_EXPLORATION_STOPPING_V1"


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    names = (*CSV_FILENAMES, *JSON_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    for name in (*names, "metadata.json"):
        path = DEFAULT_OUTPUT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_output_hashes_match_metadata(metadata) -> None:
    for name, expected in metadata["output_sha256"].items():
        assert _sha(DEFAULT_OUTPUT_DIRECTORY / name) == expected


def test_design_statuses_remain_non_release(metadata) -> None:
    assert metadata["design_status"] == DESIGN_STATUS
    assert metadata["offline_method_status"] == OFFLINE_METHOD_STATUS
    assert metadata["human_readiness"] == NOT_HUMAN_READY
    assert metadata["robot_motion_approval"] == NOT_ROBOT_MOTION_APPROVED
    assert metadata["formal_personalization_executed"] is False
    assert metadata["real_robot_connected"] is False


def test_frozen_scientific_contract_is_unchanged(metadata) -> None:
    assert metadata["rom_protocol_version"] == ROM_PROTOCOL_VERSION
    assert metadata["hip_rom_deg"] == list(FORMAL_HIP_ROM_DEG) == [0.0, 120.0]
    assert metadata["knee_rom_deg"] == list(FORMAL_KNEE_ROM_DEG) == [5.0, 145.0]
    assert metadata["theta_shank_definition"] == THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert metadata["active_reference_id"] == ACTIVE_REFERENCE_ID
    assert metadata["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert _sha(ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256
    assert metadata["five_parameter_names"] == list(PARAMETER_NAMES)
    assert metadata["mechanical_objective_version"] == MECHANICAL_OBJECTIVE_VERSION
    assert metadata["generator_bounds"] == {
        key: list(value) for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
    }
    assert metadata["algorithm_equivalence_tolerance"] == OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert metadata["support_coverage_gate_percent"] == MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0


def test_formal_lattice_is_the_existing_generator_space(formal_lattice) -> None:
    assert len(formal_lattice) == 21025
    assert formal_lattice["geometrically_admissible"].astype(bool).all()


def test_pair_universe_is_complete_and_unique(formal_lattice) -> None:
    universe = enumerate_designated_local_pair_universe(formal_lattice)
    assert len(universe) == 173188
    assert universe["pair_id"].is_unique
    assert universe["selection_hash"].is_unique


def test_pair_universe_covers_all_coordinates_and_existing_trust_levels(
    formal_lattice,
) -> None:
    universe = enumerate_designated_local_pair_universe(formal_lattice)
    assert set(universe["coordinate"]) == {"hip", "knee", "phase"}
    assert set(universe["trust_level"]) == {"INITIAL", "HALF", "MINIMUM"}
    expected_steps = {
        "hip": {0.25, 0.5, 1.0},
        "knee": {0.25, 0.5, 1.0},
        "phase": {0.0025, 0.005, 0.01},
    }
    for coordinate, group in universe.groupby("coordinate"):
        assert set(np.round(group["trust_step"], 12)) == expected_steps[coordinate]


def test_pair_plan_is_balanced_and_pre_registered(pair_plan, strata) -> None:
    assert len(pair_plan) == 324
    assert len(strata) == 27
    assert pair_plan["pair_id"].is_unique
    assert strata["planned_pair_count"].eq(PAIRS_PER_LOCATION_CLASS).all()
    assert strata["selection_uses_prediction"].eq(False).all()
    assert strata["selection_uses_truth"].eq(False).all()
    assert strata["selection_uses_final_truth_landscape"].eq(False).all()


def test_pair_plan_is_deterministic(formal_lattice, pair_plan) -> None:
    regenerated, regenerated_strata = build_designated_local_validation_pair_plan(
        formal_lattice
    )
    assert regenerated["pair_id"].tolist() == pair_plan["pair_id"].tolist()
    assert len(regenerated_strata) == 27


def test_pair_plan_outcomes_are_intentionally_blank(pair_plan) -> None:
    assert pair_plan[["predicted_delta_J", "truth_delta_J", "e_delta_J"]].isna().all().all()
    assert pair_plan["outcome_status"].eq(PAIR_OUTCOME_STATUS).all()
    assert pair_plan["formal_P2_V2_guard_input"].eq(False).all()
    assert pair_plan["threshold_frozen"].eq(False).all()


def test_pair_plan_changes_exactly_one_generator_coordinate(pair_plan) -> None:
    deltas = pair_plan[
        ["delta_alpha_hip", "delta_alpha_knee", "delta_alpha_phase"]
    ].to_numpy(dtype=float)
    assert np.all(np.count_nonzero(np.abs(deltas) > 1e-12, axis=1) == 1)
    assert pair_plan["inside_existing_generator_bounds"].astype(bool).all()
    assert pair_plan["geometrically_admissible_pair"].astype(bool).all()
    assert pair_plan["search_range_expanded"].eq(False).all()


def test_alpha_distance_is_formal_generator_distance_only(pair_plan) -> None:
    assert set(pair_plan["alpha_distance_formal_grid_steps"]) == {1.0, 2.0, 4.0}
    assert pair_plan["alpha_distance_definition"].str.contains("NOT_PHYSICAL").all()


def test_pair_selection_never_uses_truth_or_prediction(pair_plan) -> None:
    assert pair_plan["truth_used_for_pair_enumeration"].eq(False).all()
    assert pair_plan["final_truth_landscape_used_for_pair_enumeration"].eq(False).all()
    assert pair_plan["used_for_model_fitting"].eq(False).all()
    assert pair_plan["used_for_adaptation_update"].eq(False).all()
    assert pair_plan["heldout_final_test"].eq(False).all()
    assert pair_plan["used_by_P2_V1"].eq(False).all()


def test_pair_plan_hash_is_frozen_in_protocol_and_metadata(metadata) -> None:
    expected = _sha(DEFAULT_OUTPUT_DIRECTORY / "designated_local_validation_pair_plan.csv")
    protocol = json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1.json").read_text(encoding="utf-8")
    )
    assert expected == metadata["designated_local_pair_plan_sha256"]
    assert expected == protocol["pair_generation"]["pair_plan_sha256"]


def test_designated_protocol_does_not_freeze_sample_or_uncertainty_threshold(metadata) -> None:
    protocol = designated_local_validation_protocol(
        pair_plan_sha256=metadata["designated_local_pair_plan_sha256"],
        planned_pair_count=metadata["designated_local_pair_plan_count"],
        universe_pair_count=metadata["designated_local_pair_universe_count"],
    )
    assert protocol["pair_generation"]["sample_count_status"].endswith("REVIEW_APPROVAL")
    assert protocol["uncertainty_statistics_to_report"] == ["max", "P95", "P99"]
    assert protocol["uncertainty_threshold_frozen"] is False
    assert protocol["candidate_neighborhood"]["clipping_allowed"] is False
    assert protocol["candidate_neighborhood"]["bounds_expansion_allowed"] is False
    assert protocol["candidate_neighborhood"]["physical_distance_used"] is False
    assert protocol["outcome_schema"]["existing_final_truth_landscape_allowed"] is False


def test_outcome_attachment_computes_error_without_reselecting(pair_plan) -> None:
    outcomes = pair_plan[["pair_id"]].copy()
    outcomes["predicted_delta_J"] = np.linspace(-0.01, 0.01, len(outcomes))
    outcomes["truth_delta_J"] = outcomes["predicted_delta_J"] + 0.002
    attached = attach_designated_local_validation_outcomes(pair_plan, outcomes)
    assert attached["pair_id"].tolist() == pair_plan["pair_id"].tolist()
    assert np.allclose(attached["e_delta_J"], 0.002)
    assert attached["pair_selection_changed_after_outcome"].eq(False).all()
    assert attached["truth_used_for_pair_selection"].eq(False).all()
    assert attached["formal_threshold_created"].eq(False).all()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "nonfinite"])
def test_outcome_attachment_rejects_invalid_outcome_sets(pair_plan, mutation) -> None:
    outcomes = pair_plan[["pair_id"]].copy()
    outcomes["predicted_delta_J"] = 0.0
    outcomes["truth_delta_J"] = 0.0
    if mutation == "missing":
        outcomes = outcomes.iloc[:-1]
    elif mutation == "extra":
        outcomes.loc[len(outcomes)] = ["not_planned", 0.0, 0.0]
    elif mutation == "duplicate":
        outcomes.loc[1, "pair_id"] = outcomes.loc[0, "pair_id"]
    else:
        outcomes.loc[0, "truth_delta_J"] = np.inf
    with pytest.raises(ValueError):
        attach_designated_local_validation_outcomes(pair_plan, outcomes)


def test_global_and_designated_local_comparison_is_honest() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "global_vs_designated_local_validation_design.csv"
    ).set_index("validation_class")
    global_row = table.loc["CURRENT_GLOBAL_IDENTIFICATION_PAIR"]
    local_row = table.loc["DESIGNATED_LOCAL_PAIR_PLAN"]
    assert global_row["pair_instance_count"] == 61
    assert global_row["outcomes_available"]
    assert global_row["alpha_pair_mappable"] == False
    assert local_row["pair_instance_count"] == 324
    assert local_row["outcomes_available"] == False
    assert local_row["alpha_pair_mappable"]
    assert np.isnan(local_row["P95_e_delta_J"])
    assert np.isnan(local_row["max_e_delta_J"])


def test_cumulative_rule_is_a_disabled_design_candidate() -> None:
    protocol = cumulative_decision_rule_protocol()
    rule = protocol["Rule_B"]
    assert rule["maximum_accumulation_step_candidates"] == [2, 3, 5]
    assert "same_generator_coordinate" in rule["direction_consistency"]
    assert "same_signed_direction" in rule["direction_consistency"]
    assert "never_use_posthoc_truth_to_extend_or_reselect_bundle" in rule["wrong_direction_prevention"]
    assert len(rule["uncertainty_aggregation_candidates"]) == 3
    assert protocol["rule_enabled"] is False
    assert protocol["maximum_steps_frozen"] is False
    assert protocol["uncertainty_aggregation_frozen"] is False
    assert protocol["numeric_threshold_created"] is False
    assert protocol["objective_modified"] is False
    assert protocol["generator_modified"] is False


def test_cumulative_candidate_matrix_is_not_policy() -> None:
    matrix = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "cumulative_decision_rule_candidate_matrix.csv"
    )
    assert len(matrix) == 9
    assert set(matrix["maximum_accumulation_steps_candidate"]) == {2, 3, 5}
    assert matrix["candidate_enabled"].eq(False).all()
    assert matrix["maximum_steps_frozen"].eq(False).all()
    assert matrix["uncertainty_aggregation_frozen"].eq(False).all()
    assert matrix["truth_used_by_policy"].eq(False).all()


def test_knee_stiff_single_vs_cumulative_evidence_is_reported_not_executed() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "single_vs_cumulative_rule_comparison.csv"
    ).set_index("rule")
    assert table.loc["RULE_A_SINGLE_STEP", "moves_exceeding_existing_0p005"] == 0
    assert table.loc["RULE_A_SINGLE_STEP", "maximum_observed_improvement"] == pytest.approx(0.004467187931, abs=1e-12)
    assert table.loc["RULE_B_MULTI_STEP_CANDIDATE", "moves_exceeding_existing_0p005"] == 3
    assert table.loc["RULE_B_MULTI_STEP_CANDIDATE", "maximum_observed_improvement"] == pytest.approx(0.022042232169, abs=1e-12)
    assert table["policy_enabled"].eq(False).all()


def test_decision_value_stopping_separates_all_four_values() -> None:
    protocol = decision_value_exploration_stopping_protocol()
    records = protocol["per_explore_record"]
    assert set(records) == {"SUPPORT_VALUE", "MODEL_VALUE", "PREDICTION_VALUE", "DECISION_VALUE"}
    assert protocol["stop_candidate"]["consecutive_K_candidates"] == [1, 2, 3]
    assert protocol["support_used_as_decision_value"] is False
    assert "support_growth_alone_is_not_a_continue_reason" in protocol["continue_candidate"]
    assert protocol["change_detection_numeric_tolerance"] is None
    assert protocol["consecutive_K_frozen"] is False
    assert protocol["automatic_stop_enabled"] is False
    assert protocol["truth_feature_used"] is False


def test_decision_value_stopping_shadow_is_never_applied() -> None:
    summary = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "decision_value_stopping_shadow_summary.csv"
    ).set_index("consecutive_zero_value_candidate")
    assert summary["historical_trials_potentially_avoided"].to_dict() == {1: 25, 2: 21, 3: 17}
    assert summary["later_exploit_trials_in_frozen_history"].eq(0).all()
    assert summary["later_accepted_best_changes_in_frozen_history"].eq(0).all()
    assert summary["automatic_stop_executed"].eq(False).all()
    assert summary["candidate_frozen"].eq(False).all()
    assert summary["truth_feature_used"].eq(False).all()


def test_minimum_revision_set_preserves_all_frozen_boundaries() -> None:
    revisions = minimum_p2_v2_revision_set()
    assert len(revisions) == 6
    assert any("designated_local" in item for item in revisions)
    assert any("cumulative" in item for item in revisions)
    assert any("stopping" in item for item in revisions)
    assert any("0p005" in item and "90percent" in item for item in revisions)


def test_decision_matrix_answers_all_required_questions() -> None:
    matrix = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "P2_V2_FORMAL_DESIGN_DECISION_MATRIX.csv"
    ).set_index("question")
    assert set(matrix.index) == {
        "local_validation_in_formal_P2_V2",
        "cumulative_improvement_solves_stepwise_problem",
        "decision_value_stopping_reduces_meaningless_exploration",
        "formal_P2_V2_ready",
    }
    assert matrix["P2_V1_modified"].eq(False).all()
    assert matrix.loc["formal_P2_V2_ready", "recommendation"] == "NO_OFFLINE_METHOD_REQUIRES_REVISION"


def test_p2_v1_policy_definition_is_unchanged(metadata) -> None:
    policy_path = POLICY_ARTIFACT_DIRECTORY / "policy_definition.json"
    stored = json.loads(policy_path.read_text(encoding="utf-8"))
    assert stored == policy_definitions()
    assert _sha(policy_path) == metadata["policy_definition_source_sha256"]
    assert _text_sha(inspect.getsource(apply_research_decision_guard)) == metadata["decision_guard_source_sha256"]
    assert _text_sha(inspect.getsource(select_exploit_candidate)) == metadata["exploit_selector_source_sha256"]
    assert _text_sha(inspect.getsource(rank_exploration_frontier)) == metadata["exploration_ranker_source_sha256"]
    assert metadata["current_P2_V1_remains_default"] is True
    assert metadata["current_P2_V1_modified"] is False


def test_metadata_explicitly_disables_every_candidate(metadata) -> None:
    assert metadata["local_uncertainty_threshold_frozen"] is False
    assert metadata["cumulative_rule_enabled"] is False
    assert metadata["cumulative_max_steps_frozen"] is False
    assert metadata["cumulative_uncertainty_aggregation_frozen"] is False
    assert metadata["automatic_stopping_enabled"] is False
    assert metadata["stopping_candidate_frozen"] is False
    assert metadata["formal_threshold_created"] is False
    assert metadata["truth_used_to_modify_formal_policy"] is False


def test_protocol_sources_do_not_import_robot_side_packages() -> None:
    runner = Path(__file__).with_name("run_p2_v2_formal_research_protocol.py")
    forbidden = {"hardware", "control", "collection", "safety", "xCoreSDK_python"}
    assert _top_level_import_roots(CORE_SOURCE_PATH).isdisjoint(forbidden)
    assert _top_level_import_roots(runner).isdisjoint(forbidden)


def test_runner_does_not_call_p2_or_connect_robot() -> None:
    runner = Path(__file__).with_name("run_p2_v2_formal_research_protocol.py")
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_policy" not in calls
    assert "connect" not in calls
    assert "execute" not in calls


def test_protected_robot_packages_have_no_tracked_diff(metadata) -> None:
    completed = subprocess.run(
        ["git", "diff", "--", "hardware", "control", "collection", "safety"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.stdout == ""
    assert metadata["protected_package_git_diff_empty"] is True
    assert metadata["protected_package_diff_unchanged"] is True


def test_reports_preserve_scope_and_final_statuses() -> None:
    local = (DEFAULT_OUTPUT_DIRECTORY / "LOCAL_VALIDATION_PROTOCOL_REPORT.md").read_text(encoding="utf-8")
    final = (DEFAULT_OUTPUT_DIRECTORY / "P2_V2_FORMAL_DESIGN_REPORT.md").read_text(encoding="utf-8")
    leakage = (DEFAULT_OUTPUT_DIRECTORY / "DATA_LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert LOCAL_PROTOCOL_ID in local
    assert "final truth landscape is forbidden" in local
    assert CUMULATIVE_RULE_ID in final
    assert STOPPING_RULE_ID in final
    assert OFFLINE_METHOD_STATUS in final
    assert NOT_HUMAN_READY in final
    assert NOT_ROBOT_MOTION_APPROVED in final
    assert "No prediction, truth, final truth landscape" in leakage


def test_core_and_runner_hashes_match_metadata(metadata) -> None:
    runner = Path(__file__).with_name("run_p2_v2_formal_research_protocol.py")
    assert _sha(CORE_SOURCE_PATH) == metadata["protocol_core_source_sha256"]
    assert _sha(runner) == metadata["runner_source_sha256"]
