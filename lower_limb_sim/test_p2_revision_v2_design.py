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
from .p2_revision_v2_design import (
    DESIGN_PROTOCOL_ID,
    DESIGN_STATUS,
    EXPLORATION_STOPPING_CANDIDATE_ID,
    LOCAL_PROTOCOL_ID,
    OFFLINE_METHOD_STATUS,
    P2_V2_IMPLEMENTATION_STATUS,
    local_decision_validation_protocol,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .run_p2_revision_v2_design import (
    CORE_SOURCE_PATH,
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    ESTIMATOR_SOURCE,
    FIGURE_FILENAMES,
    GENERATOR_SOURCE,
    JSON_FILENAMES,
    MECHANICAL_OBJECTIVE_SOURCE,
    POLICY_ARTIFACT_DIRECTORY,
    POLICY_CORE_SOURCE,
    REPORT_FILENAMES,
)


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def local_pairs() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "retrospective_local_decision_pair_errors.csv"
    )


@pytest.fixture(scope="module")
def guard_summary() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_guard_counterfactual_summary.csv"
    ).set_index("guard_id")


@pytest.fixture(scope="module")
def exploration() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "exploration_value_components.csv"
    )


@pytest.fixture(scope="module")
def specificity() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "subject_specificity_gap.csv"
    ).set_index("subject_id")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_protocol_identifiers_are_exact() -> None:
    assert DESIGN_PROTOCOL_ID == "P2_REVISION_V2_DESIGN_ANALYSIS_V1"
    assert LOCAL_PROTOCOL_ID == "LOCAL_DECISION_VALIDATION_PROTOCOL_V1"
    assert (
        EXPLORATION_STOPPING_CANDIDATE_ID
        == "EXPLORATION_VALUE_AWARE_STOPPING_CANDIDATE_V1"
    )


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    names = (*CSV_FILENAMES, *JSON_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    for name in (*names, "metadata.json"):
        path = DEFAULT_OUTPUT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_output_hashes_match_metadata(metadata) -> None:
    for name, expected in metadata["output_sha256"].items():
        assert _sha(DEFAULT_OUTPUT_DIRECTORY / name) == expected


def test_design_status_remains_not_frozen(metadata) -> None:
    assert DESIGN_STATUS == "REVISION_DESIGN_NOT_FROZEN"
    assert metadata["design_status"] == DESIGN_STATUS
    assert metadata["local_candidate_threshold_frozen"] is False
    assert metadata["human_threshold_created"] is False


def test_offline_method_still_requires_revision(metadata) -> None:
    assert OFFLINE_METHOD_STATUS == "OFFLINE_METHOD_REQUIRES_REVISION"
    assert metadata["offline_method_status"] == OFFLINE_METHOD_STATUS


def test_no_p2_or_formal_personalization_was_executed(metadata) -> None:
    assert metadata["current_P2_executed"] is False
    assert metadata["current_P2_behavior_modified"] is False
    assert metadata["formal_personalization_executed"] is False
    assert metadata["counterfactual_trajectory_executed"] is False


def test_local_pair_protocol_uses_no_physical_distance() -> None:
    protocol = local_decision_validation_protocol()
    definition = protocol["local_pair_definition"]
    assert definition["physical_distance_threshold"] is None
    assert definition["euclidean_physical_distance_used"] is False
    assert definition["clipping_allowed"] is False


def test_local_pair_protocol_uses_existing_generator_and_trust_relation() -> None:
    definition = local_decision_validation_protocol()["local_pair_definition"]
    assert definition["both_points"] == "geometrically_admissible_generator_alpha_points"
    assert "exactly_one" in definition["relationship"]
    assert set(definition["allowed_existing_trust_levels"]) == {
        "hip",
        "knee",
        "phase",
    }


def test_future_local_validation_is_separated_from_fitting_and_final_test() -> None:
    requirements = local_decision_validation_protocol()[
        "future_designated_validation_requirements"
    ]
    assert requirements["must_not_be_model_fitting_data"] is True
    assert requirements["must_not_be_adaptation_executed_outcome"] is True
    assert requirements["must_not_be_heldout_final_test"] is True
    assert requirements["must_stratify_by_model_support_status"] is True
    assert requirements["must_cover_all_existing_trust_levels_before_generalizing"] is True
    assert requirements["minimum_sample_count"] is None


def test_retrospective_local_pairs_have_expected_count(local_pairs) -> None:
    assert len(local_pairs) == 341
    assert local_pairs["case_id"].nunique() == 9


def test_each_local_pair_changes_exactly_one_coordinate(local_pairs) -> None:
    assert local_pairs["changed_coordinate_count"].eq(1).all()
    assert set(local_pairs["changed_coordinate"]) == {"hip", "knee", "phase"}
    assert local_pairs["formal_local_candidate_relationship_valid"].astype(bool).all()


def test_local_pair_steps_are_existing_trust_levels(local_pairs) -> None:
    expected = {
        "hip": {0.25, 0.5, 1.0},
        "knee": {0.25, 0.5, 1.0},
        "phase": {0.0025, 0.005, 0.01},
    }
    for coordinate, group in local_pairs.groupby("changed_coordinate"):
        assert set(np.round(group["absolute_coordinate_step"], 12)).issubset(
            expected[coordinate]
        )


def test_retrospective_pairs_are_not_designated_validation(local_pairs) -> None:
    assert not local_pairs["designated_validation"].astype(bool).any()
    assert not local_pairs["used_to_modify_current_policy"].astype(bool).any()
    assert not local_pairs["truth_used_as_policy_input"].astype(bool).any()
    assert not local_pairs["physical_distance_used"].astype(bool).any()


def test_global_vs_local_sample_counts() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "global_vs_local_pair_error_distribution.csv"
    ).set_index("pair_class")
    assert table.loc[
        "CURRENT_GLOBAL_IDENTIFICATION_EXCITATION_PAIR_INSTANCES",
        "pair_instance_count",
    ] == 61
    assert table.loc[
        "RETROSPECTIVE_LOCAL_DECISION_OPPORTUNITY_PAIRS",
        "pair_instance_count",
    ] == 341


def test_local_error_distribution_values_are_reproduced() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "global_vs_local_pair_error_distribution.csv"
    ).set_index("pair_class")
    row = table.loc["RETROSPECTIVE_LOCAL_DECISION_OPPORTUNITY_PAIRS"]
    assert row["p95_e_delta_J"] == pytest.approx(2.412448282e-05)
    assert row["p99_e_delta_J"] == pytest.approx(0.000863147481512)
    assert row["max_e_delta_J"] == pytest.approx(0.00170839435891)


def test_local_prediction_actual_correlations_are_reported() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "global_vs_local_pair_error_distribution.csv"
    ).set_index("pair_class")
    row = table.loc["RETROSPECTIVE_LOCAL_DECISION_OPPORTUNITY_PAIRS"]
    assert row["pearson_delta_pred_vs_actual"] == pytest.approx(0.99876563)
    assert row["spearman_delta_pred_vs_actual"] == pytest.approx(0.99844549)


def test_retrospective_local_strata_expose_support_and_scale_limits() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_pair_stratum_summary.csv"
    )
    counts = table.groupby("model_supported")["pair_count"].sum()
    assert counts.loc[True] == 274
    assert counts.loc[False] == 67
    observed = set(
        zip(table["changed_coordinate"], table["absolute_coordinate_step"])
    )
    assert observed == {("hip", 1.0), ("knee", 1.0), ("phase", 0.01)}
    assert table["retrospective_evidence_only"].astype(bool).all()
    assert not table["threshold_frozen"].astype(bool).any()


def test_only_max_p95_p99_uncertainty_candidates_are_created() -> None:
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_uncertainty_candidates.csv")
    assert set(table["candidate_id"]) == {
        "LOCAL_MAX_UNCERTAINTY_CANDIDATE",
        "LOCAL_P95_UNCERTAINTY_CANDIDATE",
        "LOCAL_P99_UNCERTAINTY_CANDIDATE",
    }


def test_no_local_uncertainty_candidate_is_frozen() -> None:
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_uncertainty_candidates.csv")
    assert not table["threshold_frozen"].astype(bool).any()
    assert not table["current_P2_modified"].astype(bool).any()


def test_leave_one_case_out_candidate_excludes_evaluation_case() -> None:
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_uncertainty_candidates.csv")
    loco = table.loc[table["calibration_scope"].eq("LEAVE_ONE_CASE_OUT")]
    assert len(loco) == 27
    assert loco["excluded_evaluation_case_id"].nunique() == 9
    assert loco["calibration_case_count"].eq(8).all()


def test_g0_counterfactual_metrics_are_preserved(guard_summary) -> None:
    row = guard_summary.loc["G0_CURRENT_GLOBAL_MAX"]
    assert row["would_exploit_candidate_count"] == 20
    assert row["missed_improvement_candidate_count"] == 7
    assert row["false_improvement_candidate_count"] == 0
    assert row["conservative_stop_round_count"] == 7
    assert bool(row["current_behavior_replay"])


def test_local_p95_counterfactual_reduces_missed_without_observed_false(guard_summary) -> None:
    row = guard_summary.loc["G2_LOCAL_P95_CANDIDATE"]
    assert row["would_exploit_candidate_count"] == 24
    assert row["missed_improvement_candidate_count"] == 3
    assert row["false_improvement_candidate_count"] == 0
    assert row["change_vs_G0_missed_improvement_candidate_count"] == -4


def test_local_max_and_p99_are_more_conservative_than_g0(guard_summary) -> None:
    assert guard_summary.loc[
        "G1_LOCAL_MAX_CANDIDATE", "missed_improvement_candidate_count"
    ] == 27
    assert guard_summary.loc[
        "G2_LOCAL_P99_CANDIDATE", "missed_improvement_candidate_count"
    ] == 19


def test_guard_counterfactual_did_not_execute_or_modify_policy() -> None:
    detail = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_guard_counterfactual_detail.csv"
    )
    assert not detail["policy_modified"].astype(bool).any()
    assert not detail["trajectory_executed"].astype(bool).any()
    assert not detail["truth_used_to_construct_policy"].astype(bool).any()


def test_no_guard_threshold_is_frozen(guard_summary) -> None:
    assert not guard_summary["threshold_frozen"].astype(bool).any()


def test_exploration_value_dimensions_are_separate(exploration) -> None:
    assert int(exploration["MODEL_VALUE"].sum()) == 0
    assert int(exploration["SUPPORT_VALUE"].sum()) == 32
    assert int(exploration["DECISION_VALUE"].sum()) == 3
    assert not exploration["support_is_decision_value"].astype(bool).any()


def test_exploration_records_all_requested_observables(exploration) -> None:
    required = {
        "information_gain",
        "new_supported_points",
        "theta_change_l2",
        "RMS_prediction_map_change",
        "validation_deltaJ_error_change",
        "newly_enabled_exploit_candidates",
        "best_J_change",
    }
    assert required.issubset(exploration.columns)


def test_knee_stiff_has_eight_support_only_zero_decision_explores(exploration) -> None:
    knee = exploration.loc[exploration["subject_id"].eq("knee_stiff")]
    assert len(knee) == 8
    assert knee["SUPPORT_VALUE"].astype(bool).all()
    assert not knee["MODEL_VALUE"].astype(bool).any()
    assert not knee["DECISION_VALUE"].astype(bool).any()
    assert knee["exact_zero_decision_value_round"].astype(bool).all()


def test_matched_late_trials_are_support_only(exploration) -> None:
    selected = exploration.loc[
        exploration["subject_id"].isin(("baseline", "hip_stiff", "heavy_leg"))
        & exploration["iteration"].between(7, 13)
    ]
    assert len(selected) == 21
    assert selected["SUPPORT_VALUE"].astype(bool).all()
    assert not selected["MODEL_VALUE"].astype(bool).any()
    assert not selected["DECISION_VALUE"].astype(bool).any()


def test_stopping_candidates_are_disabled_and_not_frozen() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "exploration_stopping_counterfactual.csv"
    )
    assert not table["candidate_enabled"].astype(bool).any()
    assert not table["threshold_frozen"].astype(bool).any()
    assert not table["current_policy_modified"].astype(bool).any()
    assert not table["truth_feature_used"].astype(bool).any()


def test_stopping_candidate_historical_trial_counts() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "exploration_stopping_counterfactual.csv"
    )
    totals = table.groupby("stopping_candidate_id")[
        [
            "executed_trials_avoided",
            "later_exploit_trials_in_frozen_history",
            "later_accepted_best_changes_in_frozen_history",
        ]
    ].sum()
    assert totals.loc[
        "S1_STOP_AFTER_FIRST_EXACT_ZERO_DECISION_VALUE_EXPLORE",
        "executed_trials_avoided",
    ] == 25
    assert totals.loc[
        "S2_STOP_AFTER_TWO_CONSECUTIVE_EXACT_ZERO_DECISION_VALUE_EXPLORES",
        "executed_trials_avoided",
    ] == 21
    assert not totals["later_exploit_trials_in_frozen_history"].any()
    assert not totals["later_accepted_best_changes_in_frozen_history"].any()


def test_subject_specificity_has_four_subjects(specificity) -> None:
    assert set(specificity.index) == {
        "baseline",
        "hip_stiff",
        "knee_stiff",
        "heavy_leg",
    }


def test_only_knee_stiff_has_meaningful_truth_gap(specificity) -> None:
    meaningful = specificity.loc[
        ~specificity["gap_within_existing_0p005_equivalence"].astype(bool)
    ]
    assert list(meaningful.index) == ["knee_stiff"]
    assert meaningful.iloc[0]["J_truth_regret"] == pytest.approx(0.0252261269059)


def test_knee_stiff_one_step_improvement_is_below_unchanged_tolerance(specificity) -> None:
    row = specificity.loc["knee_stiff"]
    assert row["best_one_step_truth_delta_J_at_P2_selected_posthoc"] == pytest.approx(
        -0.00446718793104
    )
    assert not bool(row["best_one_step_improvement_exceeds_existing_0p005"])
    assert "subthreshold" in row["C_search_policy_contribution"]


def test_generator_contains_all_observed_truth_optima(specificity) -> None:
    assert specificity["generator_contains_observed_truth_optimum"].astype(bool).all()
    assert not specificity["generator_expansion_justified"].astype(bool).any()
    assert not specificity["generator_modified"].astype(bool).any()


def test_objective_is_not_modified_or_recommended_for_change(specificity, metadata) -> None:
    assert not specificity["objective_modified"].astype(bool).any()
    assert metadata["mechanical_objective_modified"] is False
    assert metadata["design_recommendation"]["objective_change_needed"] is False


def test_recommendation_is_to_implement_only_after_design_freeze(metadata) -> None:
    assert (
        metadata["design_recommendation"]["implementation_status"]
        == P2_V2_IMPLEMENTATION_STATUS
    )
    assert "AFTER_DESIGN_FREEZE" in P2_V2_IMPLEMENTATION_STATUS


def test_active_reference_and_formal_protocol_are_unchanged(metadata) -> None:
    assert ACTIVE_REFERENCE_ID == "reference_measured_asymmetric_closed_slow"
    assert metadata["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["active_reference_sha256_observed"] == _sha(ACTIVE_REFERENCE_PATH)
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"


def test_model_objective_bounds_tolerance_and_support_gate_are_unchanged(metadata) -> None:
    assert tuple(PARAMETER_NAMES) == tuple(metadata["five_parameter_names"])
    assert MECHANICAL_OBJECTIVE_VERSION == "mechanical_joint_torque_objective_v1"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert {
        key: tuple(value) for key, value in metadata["generator_bounds"].items()
    } == OFFLINE_PERSONALIZATION_SEARCH_BOUNDS


def test_current_policy_definition_is_identical_to_previous_artifact() -> None:
    previous = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    assert previous == policy_definitions()


def test_policy_and_frozen_source_hashes_match_metadata(metadata) -> None:
    assert metadata["policy_core_source_sha256"] == _sha(POLICY_CORE_SOURCE)
    assert metadata["mechanical_objective_source_sha256"] == _sha(
        MECHANICAL_OBJECTIVE_SOURCE
    )
    assert metadata["generator_source_sha256"] == _sha(GENERATOR_SOURCE)
    assert metadata["estimator_source_sha256"] == _sha(ESTIMATOR_SOURCE)
    assert metadata["design_core_source_sha256"] == _sha(CORE_SOURCE_PATH)


def test_guard_and_selector_function_hashes_match_metadata(metadata) -> None:
    assert metadata["decision_guard_source_sha256"] == _text_sha(
        inspect.getsource(apply_research_decision_guard)
    )
    assert metadata["exploit_selector_source_sha256"] == _text_sha(
        inspect.getsource(select_exploit_candidate)
    )
    assert metadata["exploration_ranker_source_sha256"] == _text_sha(
        inspect.getsource(rank_exploration_frontier)
    )


def test_truth_is_posthoc_only(metadata) -> None:
    assert metadata["truth_used_to_modify_policy"] is False
    assert metadata["truth_used_for_model_fitting"] is False
    assert metadata["truth_used_for_candidate_proposal"] is False
    assert metadata["truth_used_as_stopping_feature"] is False
    assert metadata["heldout_final_test_used"] is False


def test_protected_robot_packages_have_no_git_diff(metadata) -> None:
    completed = subprocess.run(
        ["git", "diff", "--", "hardware", "control", "collection", "safety"],
        cwd=CORE_SOURCE_PATH.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""
    assert metadata["protected_package_diff_unchanged"] is True
    assert metadata["protected_package_git_diff_empty"] is True


def test_design_modules_do_not_import_robot_packages() -> None:
    for path in (CORE_SOURCE_PATH, Path(__file__).with_name("run_p2_revision_v2_design.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        assert not roots.intersection({"hardware", "control", "collection", "safety", "xCoreSDK_python"})


def test_runner_does_not_call_p2_execution() -> None:
    path = Path(__file__).with_name("run_p2_revision_v2_design.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_policy" not in calls
    assert "connect" not in calls


def test_reports_preserve_research_only_boundaries() -> None:
    report = (DEFAULT_OUTPUT_DIRECTORY / REPORT_FILENAMES[0]).read_text(
        encoding="utf-8"
    )
    prompt = (DEFAULT_OUTPUT_DIRECTORY / REPORT_FILENAMES[1]).read_text(
        encoding="utf-8"
    )
    assert "不能冻结任何候选阈值" in report
    assert "support 不是 decision value" in report
    assert "不需要" in report
    assert "default-off" in prompt
    assert "truth 只用于最后的 post-hoc" in prompt


def test_final_readiness_is_not_human_or_robot_ready(metadata) -> None:
    assert metadata["research_status"] == "RESEARCH_ONLY"
    assert metadata["not_human_ready"] is True
    assert metadata["not_robot_motion_approved"] is True
    assert metadata["real_robot_connected"] is False
