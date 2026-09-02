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
from .p2_v2_offline_research_prototype import (
    DEFAULT_CONTROLS,
    FROZEN_LOCAL_PROTOCOL_ID,
    FROZEN_PAIR_PLAN_SHA256,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_STATUS,
    PROTOTYPE_ID,
    PROTOTYPE_STATUS,
    OfflinePrototypeControls,
    assign_frozen_pairs_to_cases,
    local_uncertainty_metrics,
    minimum_p2_v2_change_set,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .run_p2_revision_root_cause_audit import POLICY_ARTIFACT_DIRECTORY
from .run_p2_v2_offline_research_prototype import (
    CORE_SOURCE_PATH,
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    FIGURE_FILENAMES,
    JSON_FILENAMES,
    REPORT_FILENAMES,
    generate_artifacts,
)
from .run_p2_v2_formal_research_protocol import (
    DEFAULT_OUTPUT_DIRECTORY as FORMAL_PROTOCOL_ARTIFACT_DIRECTORY,
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
def pair_plan() -> pd.DataFrame:
    return pd.read_csv(
        FORMAL_PROTOCOL_ARTIFACT_DIRECTORY
        / "designated_local_validation_pair_plan.csv"
    )


@pytest.fixture(scope="module")
def local_results() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_validation_results.csv")


@pytest.fixture(scope="module")
def guard_comparison() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_guard_comparison.csv"
    ).set_index("guard_id")


@pytest.fixture(scope="module")
def cumulative() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "cumulative_rule_comparison.csv"
    ).set_index("rule_id")


@pytest.fixture(scope="module")
def stopping() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "decision_value_stopping_comparison.csv"
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


def test_prototype_identifiers_are_exact() -> None:
    assert PROTOTYPE_ID == "P2_V2_OFFLINE_RESEARCH_PROTOTYPE_IMPLEMENTATION_V1"
    assert PROTOTYPE_STATUS == "DEFAULT_OFF_SHADOW_EVALUATION_NOT_FORMAL_POLICY"
    assert FROZEN_LOCAL_PROTOCOL_ID == "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1"
    assert FROZEN_PAIR_PLAN_SHA256 == "ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55"


def test_default_controls_keep_every_behavior_off() -> None:
    controls = DEFAULT_CONTROLS.to_dict()
    assert controls["p2_v1_remains_default"] is True
    assert all(
        value is False
        for name, value in controls.items()
        if name != "p2_v1_remains_default"
    )
    DEFAULT_CONTROLS.require_default_off()


@pytest.mark.parametrize(
    "field",
    [
        "local_guard_policy_override_enabled",
        "cumulative_rule_enabled",
        "automatic_stopping_enabled",
        "truth_policy_input_enabled",
        "formal_personalization_enabled",
        "robot_execution_enabled",
    ],
)
def test_enabling_any_prototype_behavior_is_rejected(field: str) -> None:
    with pytest.raises(PermissionError):
        OfflinePrototypeControls(**{field: True}).require_default_off()


def test_disabling_p2_v1_default_is_rejected() -> None:
    with pytest.raises(PermissionError):
        OfflinePrototypeControls(p2_v1_remains_default=False).require_default_off()


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    names = (*CSV_FILENAMES, *JSON_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    for name in (*names, "metadata.json"):
        path = DEFAULT_OUTPUT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_output_hashes_match_metadata(metadata) -> None:
    for name, expected in metadata["output_sha256"].items():
        assert _sha(DEFAULT_OUTPUT_DIRECTORY / name) == expected


def test_frozen_pair_plan_sha_is_exact() -> None:
    path = (
        FORMAL_PROTOCOL_ARTIFACT_DIRECTORY
        / "designated_local_validation_pair_plan.csv"
    )
    assert _sha(path) == FROZEN_PAIR_PLAN_SHA256


def test_local_results_contain_exactly_the_frozen_pair_set(pair_plan, local_results) -> None:
    assert len(local_results) == len(pair_plan) == 324
    assert local_results["pair_id"].is_unique
    assert set(local_results["pair_id"]) == set(pair_plan["pair_id"])


def test_case_assignment_is_balanced(local_results) -> None:
    counts = local_results.groupby("case_id").size()
    assert len(counts) == 9
    assert counts.eq(36).all()


def test_case_assignment_is_deterministic_and_truth_free(pair_plan, local_results) -> None:
    cases = sorted(local_results["case_id"].unique())
    assigned = assign_frozen_pairs_to_cases(pair_plan, cases).set_index("pair_id")
    observed = local_results.set_index("pair_id")
    assert assigned["case_id"].to_dict() == observed["case_id"].to_dict()
    assert assigned["evaluation_assignment_hash"].to_dict() == observed[
        "evaluation_assignment_hash"
    ].to_dict()
    assert assigned["prediction_used_for_case_assignment"].eq(False).all()
    assert assigned["truth_used_for_case_assignment"].eq(False).all()
    assert assigned["final_truth_landscape_used_for_case_assignment"].eq(False).all()


def test_assignment_rejects_duplicate_pair_ids(pair_plan) -> None:
    invalid = pd.concat([pair_plan, pair_plan.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        assign_frozen_pairs_to_cases(invalid, ["case"])


def test_local_result_alpha_and_trust_fields_match_plan(pair_plan, local_results) -> None:
    plan = pair_plan.set_index("pair_id")
    result = local_results.set_index("pair_id")
    mappings = {
        "alpha_i_hip": "candidate_alpha_A_hip",
        "alpha_i_knee": "candidate_alpha_A_knee",
        "alpha_i_phase": "candidate_alpha_A_phase",
        "alpha_j_hip": "candidate_alpha_B_hip",
        "alpha_j_knee": "candidate_alpha_B_knee",
        "alpha_j_phase": "candidate_alpha_B_phase",
        "trust_step": "trust_step",
        "alpha_distance_formal_grid_steps": "alpha_distance_formal_grid_steps",
    }
    for source, target in mappings.items():
        assert np.allclose(plan.loc[result.index, source], result[target])
    assert plan.loc[result.index, "trust_level"].tolist() == result[
        "trust_level"
    ].tolist()


def test_local_result_deltas_and_errors_are_computed_exactly(local_results) -> None:
    assert np.allclose(
        local_results["predicted_delta_J"],
        local_results["J_pred_B"] - local_results["J_pred_A"],
        atol=1e-11,
        rtol=0.0,
    )
    assert np.allclose(
        local_results["truth_delta_J"],
        local_results["J_truth_B"] - local_results["J_truth_A"],
        atol=1e-11,
        rtol=0.0,
    )
    assert np.allclose(
        local_results["e_delta_J"],
        np.abs(local_results["predicted_delta_J"] - local_results["truth_delta_J"]),
        atol=1e-12,
        rtol=0.0,
    )


def test_local_results_never_reselect_or_modify_policy(local_results) -> None:
    assert local_results["pair_selection_changed_after_truth"].eq(False).all()
    assert local_results["truth_used_for_pair_selection"].eq(False).all()
    assert local_results["truth_used_to_modify_formal_policy"].eq(False).all()
    assert local_results["formal_guard_input"].eq(False).all()
    assert local_results["P2_V1_modified"].eq(False).all()
    assert local_results["evaluation_role"].eq(
        "FRESH_OFFLINE_DESIGNATED_OUTCOME_AFTER_PAIR_PLAN_FREEZE"
    ).all()


def test_every_result_records_the_frozen_pair_sha(local_results) -> None:
    assert local_results["pair_plan_sha256"].eq(FROZEN_PAIR_PLAN_SHA256).all()


def test_local_uncertainty_metrics_are_exact(local_results, metadata) -> None:
    metrics = local_uncertainty_metrics(local_results)
    assert metrics["local_max"] == pytest.approx(0.0016827379049442204, abs=1e-14)
    assert metrics["local_P95"] == pytest.approx(0.000430956758923898, abs=1e-14)
    assert metrics["local_P99"] == pytest.approx(0.001276942013587856, abs=1e-14)
    assert metrics == pytest.approx(metadata["local_uncertainty_metrics"], abs=1e-14)


def test_local_uncertainty_metrics_reject_nonfinite_data(local_results) -> None:
    invalid = local_results.copy()
    invalid.loc[0, "e_delta_J"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        local_uncertainty_metrics(invalid)


def test_guard_comparison_contains_g0_through_g3(guard_comparison) -> None:
    assert list(guard_comparison.index) == [
        "G0_CURRENT_GLOBAL_UNCERTAINTY_REPLAY",
        "G1_DESIGNATED_LOCAL_MAX_SHADOW",
        "G2_DESIGNATED_LOCAL_P95_SHADOW",
        "G3_DESIGNATED_LOCAL_P99_SHADOW",
    ]
    assert guard_comparison["candidate_count"].eq(341).all()


def test_local_p95_reduces_missed_improvement_without_false_increase(
    guard_comparison,
) -> None:
    g0 = guard_comparison.loc["G0_CURRENT_GLOBAL_UNCERTAINTY_REPLAY"]
    g2 = guard_comparison.loc["G2_DESIGNATED_LOCAL_P95_SHADOW"]
    assert g0["missed_improvement_count"] == 7
    assert g2["missed_improvement_count"] == 3
    assert g2["change_vs_G0_missed_improvement_count"] == -4
    assert g0["false_improvement_count"] == 0
    assert g2["false_improvement_count"] == 0
    assert g2["change_vs_G0_false_improvement_count"] == 0


def test_local_max_and_p99_remain_more_conservative(guard_comparison) -> None:
    assert guard_comparison.loc[
        "G1_DESIGNATED_LOCAL_MAX_SHADOW", "missed_improvement_count"
    ] == 27
    assert guard_comparison.loc[
        "G3_DESIGNATED_LOCAL_P99_SHADOW", "missed_improvement_count"
    ] == 23


def test_guard_shadow_does_not_freeze_or_modify_policy(guard_comparison) -> None:
    assert guard_comparison["shadow_only"].astype(bool).all()
    assert guard_comparison["threshold_frozen"].eq(False).all()
    assert guard_comparison["formal_policy_modified"].eq(False).all()
    assert guard_comparison["truth_used_to_modify_formal_policy"].eq(False).all()


def test_guard_detail_is_a_four_way_shadow() -> None:
    detail = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_guard_shadow_detail.csv")
    assert len(detail) == 341 * 4
    assert detail.groupby("guard_id").size().eq(341).all()
    assert detail["trajectory_executed"].eq(False).all()
    assert detail["formal_policy_modified"].eq(False).all()


def test_cumulative_rules_have_exact_lengths(cumulative) -> None:
    assert cumulative.loc["RULE_A_SINGLE_STEP", "trajectory_sequence_length"] == 1
    assert cumulative.loc["RULE_B_TWO_STEP_CUMULATIVE", "trajectory_sequence_length"] == 2
    assert cumulative.loc["RULE_C_THREE_STEP_CUMULATIVE", "trajectory_sequence_length"] == 3
    assert cumulative.loc["RULE_D_FIVE_STEP_CUMULATIVE", "trajectory_sequence_length"] == 5


def test_single_step_does_not_cross_existing_tolerance(cumulative) -> None:
    row = cumulative.loc["RULE_A_SINGLE_STEP"]
    assert row["predicted_cumulative_delta_J"] == pytest.approx(-0.004467187931, abs=1e-12)
    assert not row["passes_existing_0p005_before_bundle_uncertainty"]
    assert row["recovered_improvement"] == 0.0
    assert not row["false_acceptance"]


@pytest.mark.parametrize(
    ("rule_id", "expected"),
    [
        ("RULE_B_TWO_STEP_CUMULATIVE", 0.00890528510875),
        ("RULE_C_THREE_STEP_CUMULATIVE", 0.0133137854648),
        ("RULE_D_FIVE_STEP_CUMULATIVE", 0.0220422321691),
    ],
)
def test_cumulative_candidates_recover_knee_improvement_without_false_acceptance(
    cumulative, rule_id: str, expected: float
) -> None:
    row = cumulative.loc[rule_id]
    assert row["passes_existing_0p005_before_bundle_uncertainty"]
    assert row["recovered_improvement"] == pytest.approx(expected, abs=1e-12)
    assert not row["false_acceptance"]
    assert row["direction_consistent"]
    assert row["same_generator_coordinate"]
    assert row["inside_existing_generator_bounds"]


def test_cumulative_uncertainty_and_rules_remain_unfrozen(cumulative) -> None:
    assert cumulative["uncertainty_constraint_applied"].eq(False).all()
    assert cumulative["uncertainty_constraint_status"].eq(
        "UNFROZEN_REQUIRES_DESIGNATED_BUNDLE_VALIDATION"
    ).all()
    assert cumulative["rule_frozen"].eq(False).all()
    assert cumulative["rule_enabled"].eq(False).all()
    assert cumulative["trajectory_executed"].eq(False).all()
    assert cumulative["truth_used_to_modify_formal_policy"].eq(False).all()


def test_stopping_comparison_contains_four_cases_and_four_strategies(stopping) -> None:
    assert len(stopping) == 16
    assert stopping["case_id"].nunique() == 4
    assert set(stopping["strategy_id"]) == {
        "CURRENT_P2_V1_HISTORY",
        "K1_DECISION_VALUE_STOP_SHADOW",
        "K2_DECISION_VALUE_STOP_SHADOW",
        "K3_DECISION_VALUE_STOP_SHADOW",
    }


def test_stopping_shadow_reduces_exploration_without_missed_opportunity(stopping) -> None:
    summary = stopping.groupby("strategy_id").agg(
        explores=("exploration_count", "sum"),
        reduction=("exploration_reduction_vs_current", "sum"),
        missed=("missed_opportunity", "sum"),
    )
    assert summary.loc["CURRENT_P2_V1_HISTORY"].to_dict() == {
        "explores": 32,
        "reduction": 0,
        "missed": 0,
    }
    assert summary.loc["K1_DECISION_VALUE_STOP_SHADOW"].to_dict() == {
        "explores": 7,
        "reduction": 25,
        "missed": 0,
    }
    assert summary.loc["K2_DECISION_VALUE_STOP_SHADOW"].to_dict() == {
        "explores": 11,
        "reduction": 21,
        "missed": 0,
    }
    assert summary.loc["K3_DECISION_VALUE_STOP_SHADOW"].to_dict() == {
        "explores": 15,
        "reduction": 17,
        "missed": 0,
    }


def test_stopping_shadow_preserves_final_best_trajectory(stopping) -> None:
    assert stopping["final_best_trajectory"].eq(
        stopping["current_final_best_trajectory"]
    ).all()
    assert stopping["later_exploit_count"].eq(0).all()
    assert stopping["later_accepted_improvement_count"].eq(0).all()


def test_stopping_shadow_reports_support_loss_separately(stopping) -> None:
    support = stopping.groupby("strategy_id")["support_increase"].sum().to_dict()
    assert support == {
        "CURRENT_P2_V1_HISTORY": 33200,
        "K1_DECISION_VALUE_STOP_SHADOW": 19450,
        "K2_DECISION_VALUE_STOP_SHADOW": 21650,
        "K3_DECISION_VALUE_STOP_SHADOW": 23850,
    }
    assert stopping["support_increase_not_used_as_decision_value"].astype(bool).all()


def test_stopping_candidates_are_never_enabled(stopping) -> None:
    assert stopping["automatic_stop_executed"].eq(False).all()
    assert stopping["candidate_frozen"].eq(False).all()
    assert stopping["truth_feature_used"].eq(False).all()
    assert stopping["formal_policy_modified"].eq(False).all()


def test_minimum_change_set_has_only_default_off_research_changes() -> None:
    changes = minimum_p2_v2_change_set()
    assert len(changes) == 6
    assert any("local_uncertainty" in item for item in changes)
    assert any("cumulative" in item for item in changes)
    assert any("stopping" in item for item in changes)
    assert all("hardware" not in item and "objective" not in item for item in changes)


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


def test_metadata_preserves_non_release_statuses(metadata) -> None:
    assert metadata["offline_method_status"] == OFFLINE_METHOD_STATUS
    assert metadata["human_readiness"] == NOT_HUMAN_READY
    assert metadata["robot_motion_approval"] == NOT_ROBOT_MOTION_APPROVED
    assert metadata["P2_V1_remains_default"] is True
    assert metadata["P2_V1_executed_by_runner"] is False
    assert metadata["P2_V1_modified"] is False
    assert metadata["formal_personalization_executed"] is False
    assert metadata["real_robot_connected"] is False


def test_metadata_keeps_all_candidates_unfrozen(metadata) -> None:
    assert metadata["local_uncertainty_threshold_frozen"] is False
    assert metadata["cumulative_rule_enabled"] is False
    assert metadata["cumulative_rule_frozen"] is False
    assert metadata["cumulative_bundle_uncertainty_frozen"] is False
    assert metadata["automatic_stopping_enabled"] is False
    assert metadata["stopping_K_frozen"] is False
    assert metadata["truth_used_to_modify_formal_policy"] is False
    assert metadata["truth_used_for_automatic_stopping"] is False


def test_p2_v1_policy_definition_and_functions_are_unchanged(metadata) -> None:
    path = POLICY_ARTIFACT_DIRECTORY / "policy_definition.json"
    assert json.loads(path.read_text(encoding="utf-8")) == policy_definitions()
    assert _sha(path) == metadata["policy_definition_source_sha256"]
    assert _text_sha(inspect.getsource(apply_research_decision_guard)) == metadata[
        "decision_guard_source_sha256"
    ]
    assert _text_sha(inspect.getsource(select_exploit_candidate)) == metadata[
        "exploit_selector_source_sha256"
    ]
    assert _text_sha(inspect.getsource(rank_exploration_frontier)) == metadata[
        "exploration_ranker_source_sha256"
    ]


def test_prototype_sources_do_not_import_robot_side_packages() -> None:
    runner = Path(__file__).with_name("run_p2_v2_offline_research_prototype.py")
    forbidden = {"hardware", "control", "collection", "safety", "xCoreSDK_python"}
    assert _top_level_import_roots(CORE_SOURCE_PATH).isdisjoint(forbidden)
    assert _top_level_import_roots(runner).isdisjoint(forbidden)


def test_runner_does_not_call_p2_connect_or_execute() -> None:
    runner = Path(__file__).with_name("run_p2_v2_offline_research_prototype.py")
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_policy" not in calls
    assert "connect" not in calls
    assert "execute" not in calls


def test_runner_refuses_to_overwrite_existing_artifacts(tmp_path) -> None:
    (tmp_path / "sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_artifacts(tmp_path)
    assert (tmp_path / "sentinel").read_text(encoding="utf-8") == "preserve"


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
    assert metadata["safety_configuration_modified"] is False


def test_reports_answer_all_questions_and_keep_boundaries() -> None:
    report = (DEFAULT_OUTPUT_DIRECTORY / "P2_V2_PROTOTYPE_EVALUATION_REPORT.md").read_text(encoding="utf-8")
    leakage = (DEFAULT_OUTPUT_DIRECTORY / "DATA_LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    for number in range(1, 6):
        assert f"回答 {number}" in report
    assert "missed improvement 从 7 降到 3" in report
    assert "false improvement 从 0 变为 0" in report
    assert OFFLINE_METHOD_STATUS in report
    assert NOT_HUMAN_READY in report
    assert NOT_ROBOT_MOTION_APPROVED in report
    assert "Truth does not reselect a pair or case" in leakage


def test_core_and_runner_hashes_match_metadata(metadata) -> None:
    runner = Path(__file__).with_name("run_p2_v2_offline_research_prototype.py")
    assert _sha(CORE_SOURCE_PATH) == metadata["prototype_core_source_sha256"]
    assert _sha(runner) == metadata["runner_source_sha256"]


def test_formal_lattice_still_has_21025_points() -> None:
    lattice = geometrically_valid_parameter_lattice(
        pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    )
    assert len(lattice) == 21025
    assert lattice["geometrically_admissible"].astype(bool).all()
