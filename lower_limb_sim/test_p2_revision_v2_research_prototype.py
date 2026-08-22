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
from .large_artifact_reproduction import artifact_entry
from .p2_revision_v2_research_prototype import (
    CUMULATIVE_RULE_ASSESSMENT,
    DEFAULT_PROTOTYPE_CONTROLS,
    EXPLORATION_VALUE_PROTOCOL_ID,
    LOCAL_PROTOCOL_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_STATUS,
    PROTOTYPE_ID,
    PROTOTYPE_STATUS,
    PrototypeControls,
    build_formal_local_neighborhood,
    build_local_validation_pairs,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .run_p2_revision_v2_research_prototype import (
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
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .sequential_personalization import SearchAlpha, TrustRegionSteps, shrink_steps


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
def local_pairs() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_validation_pairs.csv")


@pytest.fixture(scope="module")
def guard_summary() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_guard_counterfactual_summary.csv"
    ).set_index("guard_id")


@pytest.fixture(scope="module")
def exploration() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "exploration_value_history.csv")


@pytest.fixture(scope="module")
def cumulative() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "knee_stiff_cumulative_improvement.csv"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_protocol_identifiers_are_exact() -> None:
    assert PROTOTYPE_ID == "P2_REVISION_V2_RESEARCH_PROTOTYPE_V1"
    assert LOCAL_PROTOCOL_ID == "LOCAL_DECISION_VALIDATION_PROTOCOL_V1"
    assert (
        EXPLORATION_VALUE_PROTOCOL_ID
        == "DECISION_VALUE_AWARE_EXPLORATION_SHADOW_V1"
    )


def test_default_controls_are_strictly_off() -> None:
    controls = DEFAULT_PROTOTYPE_CONTROLS.to_dict()
    assert controls["current_p2_remains_default"] is True
    assert controls["local_uncertainty_policy_override_enabled"] is False
    assert controls["exploration_automatic_stop_enabled"] is False
    assert controls["cumulative_decision_rule_enabled"] is False
    assert controls["truth_policy_input_enabled"] is False
    assert controls["robot_execution_enabled"] is False
    assert controls["formal_policy_created"] is False
    assert controls["threshold_frozen"] is False


@pytest.mark.parametrize(
    "field",
    [
        "local_uncertainty_policy_override_enabled",
        "exploration_automatic_stop_enabled",
        "cumulative_decision_rule_enabled",
        "truth_policy_input_enabled",
        "robot_execution_enabled",
    ],
)
def test_behavior_switches_are_rejected_by_shadow_boundary(field: str) -> None:
    values = {field: True}
    with pytest.raises(PermissionError):
        PrototypeControls(**values).require_default_off()


def test_disabling_current_p2_default_is_rejected() -> None:
    with pytest.raises(PermissionError):
        PrototypeControls(current_p2_remains_default=False).require_default_off()


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    names = (*CSV_FILENAMES, *JSON_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    for name in (*names, "metadata.json"):
        path = DEFAULT_OUTPUT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_all_output_hashes_match_metadata(metadata) -> None:
    for name, expected in metadata["output_sha256"].items():
        assert _sha(DEFAULT_OUTPUT_DIRECTORY / name) == expected


def test_formal_lattice_is_the_existing_21025_point_space(formal_lattice) -> None:
    assert len(formal_lattice) == 21025
    assert formal_lattice["geometrically_admissible"].astype(bool).all()


def test_initial_local_neighborhood_contains_current_and_six_neighbors(
    formal_lattice,
) -> None:
    table = build_formal_local_neighborhood(
        formal_lattice, SearchAlpha(), TrustRegionSteps()
    )
    assert len(table) == 7
    assert table["neighborhood_role"].eq("CURRENT").sum() == 1
    assert table["included_as_local_validation_neighbor"].astype(bool).sum() == 6


def test_existing_half_and_minimum_trust_levels_are_supported(formal_lattice) -> None:
    half = shrink_steps(TrustRegionSteps())
    minimum = shrink_steps(half)
    half_table = build_formal_local_neighborhood(
        formal_lattice, SearchAlpha(), half
    )
    minimum_table = build_formal_local_neighborhood(
        formal_lattice, SearchAlpha(), minimum
    )
    assert set(half_table.loc[half_table.changed_coordinate.eq("hip"), "trust_step_hip"]) == {0.5}
    assert set(minimum_table.loc[minimum_table.changed_coordinate.eq("hip"), "trust_step_hip"]) == {0.25}
    assert half_table["included_as_local_validation_neighbor"].astype(bool).sum() == 6
    assert minimum_table["included_as_local_validation_neighbor"].astype(bool).sum() == 6


def test_nonexisting_trust_level_is_rejected(formal_lattice) -> None:
    with pytest.raises(ValueError, match="not an existing trust-region level"):
        build_formal_local_neighborhood(
            formal_lattice,
            SearchAlpha(),
            TrustRegionSteps(hip_deg=0.75, knee_deg=1.0, phase=0.01),
        )


def test_boundary_neighborhood_is_not_clipped_or_expanded(formal_lattice) -> None:
    table = build_formal_local_neighborhood(
        formal_lattice,
        SearchAlpha(knee_delta_deg=-5.0),
        TrustRegionSteps(),
    )
    outside = table.loc[
        np.isclose(table["candidate_alpha_knee"], -6.0, atol=1e-12, rtol=0.0)
    ]
    assert len(outside) == 1
    assert not bool(outside.iloc[0]["inside_existing_generator_bounds"])
    assert not bool(outside.iloc[0]["included_as_local_validation_neighbor"])
    assert not table["pointwise_clipping_applied"].astype(bool).any()
    assert not table["search_range_expanded"].astype(bool).any()


def test_alpha_distance_is_formal_not_physical(formal_lattice) -> None:
    table = build_formal_local_neighborhood(
        formal_lattice, SearchAlpha(), TrustRegionSteps()
    )
    neighbors = table.loc[table["included_as_local_validation_neighbor"].astype(bool)]
    assert neighbors["alpha_distance_formal_grid_steps"].eq(4.0).all()
    assert not neighbors["physical_distance_used"].astype(bool).any()
    assert neighbors["alpha_distance_definition"].str.contains("NOT_PHYSICAL").all()


def test_generic_local_pair_builder_computes_required_fields(formal_lattice) -> None:
    neighborhood = build_formal_local_neighborhood(
        formal_lattice, SearchAlpha(), TrustRegionSteps()
    )
    points = neighborhood.loc[
        neighborhood["formal_lattice_member"].astype(bool),
        ["trajectory_id", "candidate_alpha_hip", "candidate_alpha_knee", "candidate_alpha_phase"],
    ].rename(
        columns={
            "candidate_alpha_hip": "hip_delta",
            "candidate_alpha_knee": "knee_delta",
            "candidate_alpha_phase": "phase_delta",
        }
    )
    prediction = points.copy()
    truth = points.copy()
    prediction["J_pred"] = 1.0 + 0.001 * np.arange(len(points))
    prediction["model_supported"] = True
    truth["J_truth"] = 1.0 + 0.0012 * np.arange(len(points))
    pairs = build_local_validation_pairs(
        neighborhood,
        prediction,
        truth,
        case_id="unit_case",
        iteration=1,
    )
    assert len(pairs) == 6
    assert pairs["pair_id"].nunique() == 6
    assert {
        "predicted_delta_J",
        "truth_delta_J_posthoc",
        "e_delta_J",
        "alpha_distance_formal_grid_steps",
    }.issubset(pairs.columns)
    np.testing.assert_allclose(
        pairs["e_delta_J"],
        np.abs(pairs["predicted_delta_J"] - pairs["truth_delta_J_posthoc"]),
    )
    assert not pairs["truth_used_by_formal_policy"].astype(bool).any()


def test_historical_neighborhood_rows_validate_current_candidates() -> None:
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_neighborhood_history.csv")
    assert len(table) == 427
    assert table.groupby(["case_id", "iteration"])["neighborhood_role"].apply(
        lambda values: int((values == "CURRENT").sum()) == 1
    ).all()
    historical = table.loc[table["historical_pair_present"].astype(bool)]
    assert len(historical) == 341
    assert historical["included_as_local_validation_neighbor"].astype(bool).all()


def test_protocol_examples_cover_all_existing_trust_levels() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_neighborhood_protocol_examples.csv"
    )
    assert len(table) == 21
    assert set(table["trust_level_label"]) == {"INITIAL", "HALF", "MINIMUM"}
    assert table["example_only_not_policy_input"].astype(bool).all()


def test_local_validation_pairs_have_required_count_and_unique_ids(local_pairs) -> None:
    assert len(local_pairs) == 341
    assert local_pairs["pair_id"].nunique() == 341
    assert local_pairs["case_id"].nunique() == 9


def test_local_pairs_remain_inside_existing_generator(local_pairs) -> None:
    assert local_pairs["inside_existing_generator_bounds"].astype(bool).all()
    assert local_pairs["formal_lattice_current_member"].astype(bool).all()
    assert local_pairs["formal_lattice_candidate_member"].astype(bool).all()
    assert local_pairs["geometrically_admissible_pair"].astype(bool).all()
    assert not local_pairs["search_range_expanded"].astype(bool).any()


def test_local_pair_truth_is_posthoc_only(local_pairs) -> None:
    assert local_pairs["research_metric_only"].astype(bool).all()
    assert not local_pairs["threshold_frozen"].astype(bool).any()
    assert not local_pairs["truth_used_by_formal_policy"].astype(bool).any()
    assert local_pairs["truth_used_only_for_posthoc_metric"].astype(bool).all()
    assert not local_pairs["current_P2_modified"].astype(bool).any()


def test_local_pair_errors_are_exact(local_pairs) -> None:
    np.testing.assert_allclose(
        local_pairs["e_delta_J"],
        np.abs(
            local_pairs["predicted_delta_J"]
            - local_pairs["truth_delta_J_posthoc"]
        ),
        atol=1e-12,
    )


def test_local_uncertainty_research_metrics_are_reproduced() -> None:
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_uncertainty_metrics.csv")
    pooled = table.loc[table["metric_scope"].eq("POOLED_RETROSPECTIVE")].iloc[0]
    assert pooled["local_max_error"] == pytest.approx(0.00170839435891)
    assert pooled["local_P95_error"] == pytest.approx(2.412448282e-05)
    assert pooled["local_P99_error"] == pytest.approx(0.000863147481512)
    assert not table["threshold_frozen"].astype(bool).any()
    assert not table["formal_guard_uses_metric"].astype(bool).any()


def test_local_metrics_use_leave_one_case_out_for_counterfactual() -> None:
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "local_uncertainty_metrics.csv")
    loco = table.loc[table["metric_scope"].eq("LEAVE_ONE_CASE_OUT")]
    assert len(loco) == 9
    assert loco["case_count"].eq(8).all()
    assert loco["excluded_evaluation_case_id"].nunique() == 9


def test_g0_guard_replay_is_unchanged(guard_summary) -> None:
    row = guard_summary.loc["G0_CURRENT_GLOBAL_GUARD_REPLAY"]
    assert row["would_exploit_candidate_count"] == 20
    assert row["missed_improvement_candidate_count"] == 7
    assert row["false_improvement_candidate_count"] == 0
    assert row["conservative_stop_round_count"] == 7


def test_g1_local_max_is_more_conservative(guard_summary) -> None:
    row = guard_summary.loc["G1_LOCAL_MAX_RESEARCH_METRIC"]
    assert row["would_exploit_candidate_count"] == 0
    assert row["missed_improvement_candidate_count"] == 27
    assert row["false_improvement_candidate_count"] == 0


def test_g2_local_p95_reduces_missed_without_observed_false(guard_summary) -> None:
    row = guard_summary.loc["G2_LOCAL_P95_RESEARCH_METRIC"]
    assert row["would_exploit_candidate_count"] == 24
    assert row["missed_improvement_candidate_count"] == 3
    assert row["false_improvement_candidate_count"] == 0
    assert row["change_vs_G0_missed_improvement_candidate_count"] == -4
    assert row["change_vs_G0_false_improvement_candidate_count"] == 0


def test_counterfactual_never_executes_or_modifies_policy() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "local_guard_counterfactual_detail.csv"
    )
    assert not table["threshold_frozen"].astype(bool).any()
    assert not table["formal_policy_modified"].astype(bool).any()
    assert not table["trajectory_executed"].astype(bool).any()
    assert not table["truth_used_to_modify_formal_policy"].astype(bool).any()


def test_exploration_history_records_requested_support_model_decision_fields(
    exploration,
) -> None:
    required = {
        "new_supported_points",
        "parameter_delta_l2",
        "prediction_map_RMS_delta",
        "best_trajectory_changed",
        "predicted_local_ranking_changed",
        "predicted_global_ranking_changed",
        "exploit_eligibility_changed",
        "SUPPORT_VALUE",
        "MODEL_VALUE",
        "DECISION_VALUE",
    }
    assert required.issubset(exploration.columns)


def test_exploration_value_counts_are_reproduced(exploration) -> None:
    assert len(exploration) == 32
    assert int(exploration["SUPPORT_VALUE"].sum()) == 32
    assert int(exploration["MODEL_VALUE"].sum()) == 0
    assert int(exploration["DECISION_VALUE"].sum()) == 3
    assert int(exploration["zero_model_and_decision_value_explore"].sum()) == 29


def test_ranking_change_is_explicit_and_zero_in_history(exploration) -> None:
    assert not exploration["predicted_local_ranking_changed"].astype(bool).any()
    assert not exploration["predicted_global_ranking_changed"].astype(bool).any()


def test_support_is_never_relabelled_as_decision_value(exploration) -> None:
    assert not exploration["support_is_decision_value"].astype(bool).any()


def test_exploration_scoring_does_not_stop_or_reduce_trials(exploration) -> None:
    assert set(exploration["prototype_action"]) == {
        "SHADOW_SCORE_ONLY_NO_AUTOMATIC_STOP"
    }
    assert not exploration["automatic_stop_triggered"].astype(bool).any()
    summary = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "exploration_value_summary.csv"
    ).iloc[0]
    assert summary["actual_explore_trials_avoided_by_prototype"] == 0
    assert not bool(summary["automatic_stop_enabled"])
    assert not bool(summary["scoring_reduces_exploration_by_itself"])


def test_exploration_scoring_uses_no_truth_landscape(exploration) -> None:
    assert not exploration["truth_landscape_used_for_scoring"].astype(bool).any()
    assert not exploration["formal_policy_modified"].astype(bool).any()


def test_knee_cumulative_path_uses_existing_zero_to_minus_five_direction(
    cumulative,
) -> None:
    np.testing.assert_allclose(cumulative["alpha_hip"], 0.0)
    np.testing.assert_allclose(cumulative["alpha_phase"], 0.0)
    np.testing.assert_allclose(cumulative["alpha_knee"], [0, -1, -2, -3, -4, -5])


def test_all_knee_single_steps_are_below_unchanged_tolerance(cumulative) -> None:
    selected = cumulative.loc[cumulative["step_index"].gt(0)]
    assert not selected["single_step_exceeds_existing_0p005"].astype(bool).any()
    assert selected["single_step_improvement_magnitude"].max() == pytest.approx(
        0.004467187931
    )


def test_knee_cumulative_improvement_crosses_at_two_steps(cumulative) -> None:
    crossing = cumulative.loc[
        cumulative["cumulative_exceeds_existing_0p005"].astype(bool)
    ]
    assert int(crossing.iloc[0]["step_index"]) == 2
    assert crossing.iloc[0]["cumulative_improvement_magnitude"] == pytest.approx(
        0.008905285109
    )
    assert cumulative.iloc[-1]["cumulative_improvement_magnitude"] == pytest.approx(
        0.022042232169
    )


def test_cumulative_assessment_requires_research_candidate_not_policy() -> None:
    payload = json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "cumulative_improvement_assessment.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["cumulative_decision_rule_research_design_required"] is True
    assert payload["assessment"] == CUMULATIVE_RULE_ASSESSMENT
    assert payload["cumulative_rule_enabled"] is False
    assert payload["objective_modified"] is False
    assert payload["truth_used_to_modify_policy"] is False
    assert payload["formal_policy_approval"] is False


def test_cumulative_audit_does_not_modify_objective_or_generator(cumulative) -> None:
    assert not cumulative["mechanical_objective_modified"].astype(bool).any()
    assert not cumulative["generator_direction_modified"].astype(bool).any()
    assert not cumulative["truth_used_by_policy"].astype(bool).any()


def test_active_reference_and_protocol_are_unchanged(metadata) -> None:
    assert ACTIVE_REFERENCE_ID == "reference_measured_asymmetric_closed_slow"
    assert ACTIVE_REFERENCE_SHA256 == metadata["active_reference_sha256"]
    assert _sha(ACTIVE_REFERENCE_PATH) == metadata["active_reference_sha256_observed"]
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"


def test_model_objective_generator_tolerance_and_support_are_unchanged(metadata) -> None:
    assert tuple(metadata["five_parameter_names"]) == tuple(PARAMETER_NAMES)
    assert MECHANICAL_OBJECTIVE_VERSION == "mechanical_joint_torque_objective_v1"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert {
        name: tuple(values) for name, values in metadata["generator_bounds"].items()
    } == OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
    assert metadata["mechanical_objective_modified"] is False
    assert metadata["generator_modified"] is False
    assert metadata["five_parameter_model_modified"] is False


def test_current_p2_definition_is_identical_to_prior_artifact() -> None:
    prior = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    assert prior == policy_definitions()


def test_current_p2_and_frozen_source_hashes_match_metadata(metadata) -> None:
    assert metadata["policy_core_source_sha256"] == _sha(POLICY_CORE_SOURCE)
    assert metadata["mechanical_objective_source_sha256"] == _sha(
        MECHANICAL_OBJECTIVE_SOURCE
    )
    assert metadata["generator_source_sha256"] == _sha(GENERATOR_SOURCE)
    assert metadata["estimator_source_sha256"] == _sha(ESTIMATOR_SOURCE)
    assert metadata["prototype_core_source_sha256"] == _sha(CORE_SOURCE_PATH)


def test_existing_guard_and_selectors_are_unchanged(metadata) -> None:
    assert metadata["decision_guard_source_sha256"] == _text_sha(
        inspect.getsource(apply_research_decision_guard)
    )
    assert metadata["exploit_selector_source_sha256"] == _text_sha(
        inspect.getsource(select_exploit_candidate)
    )
    assert metadata["exploration_ranker_source_sha256"] == _text_sha(
        inspect.getsource(rank_exploration_frontier)
    )


def test_input_artifact_hashes_still_match_metadata(metadata) -> None:
    known = {
        "parameter_map": DEFAULT_PARAMETER_MAP_PATH,
        "root_counterfactual": Path(
            "lower_limb_sim/formal_artifacts/p2_revision_root_cause_audit_v1/counterfactual_guard_comparison.csv"
        ),
        "root_exploration": Path(
            "lower_limb_sim/formal_artifacts/p2_revision_root_cause_audit_v1/exploration_value_decomposition.csv"
        ),
        "convergence_best_history": Path(
            "lower_limb_sim/formal_artifacts/sequential_personalization_convergence_stopping_audit_v1/best_trajectory_stability.csv"
        ),
        "convergence_landscape_evolution": Path(
            "lower_limb_sim/formal_artifacts/sequential_personalization_convergence_stopping_audit_v1/prediction_landscape_evolution.csv"
        ),
        "current_policy_definition": POLICY_ARTIFACT_DIRECTORY / "policy_definition.json",
        "current_policy_summary": POLICY_ARTIFACT_DIRECTORY / "scenario_sequential_summary.csv",
        "current_policy_trial_history": POLICY_ARTIFACT_DIRECTORY / "sequential_trial_history.csv",
    }
    for name, path in known.items():
        assert metadata["source_input_sha256"][name] == _sha(path)
    assert metadata["source_input_sha256"]["root_knee_truth_landscape"] == (
        artifact_entry("truth_landscape_knee_stiff")["sha256"]
    )


def test_truth_never_modifies_formal_policy(metadata) -> None:
    assert metadata["truth_used_to_modify_formal_policy"] is False
    assert metadata["truth_used_for_proposal_or_ranking"] is False
    assert metadata["truth_used_for_automatic_stop"] is False
    assert metadata["heldout_final_test_used"] is False
    assert metadata["formal_personalization_executed"] is False


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


def test_prototype_modules_do_not_import_robot_packages() -> None:
    paths = (
        CORE_SOURCE_PATH,
        Path(__file__).with_name("run_p2_revision_v2_research_prototype.py"),
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        assert not roots.intersection(
            {"hardware", "control", "collection", "safety", "xCoreSDK_python"}
        )


def test_runner_never_calls_existing_p2_or_robot_connection() -> None:
    path = Path(__file__).with_name("run_p2_revision_v2_research_prototype.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_policy" not in calls
    assert "connect" not in calls


def test_report_answers_all_five_questions_without_freezing() -> None:
    report = (DEFAULT_OUTPUT_DIRECTORY / "P2_REVISION_V2_PROTOTYPE_REPORT.md").read_text(
        encoding="utf-8"
    )
    for number in range(1, 6):
        assert f"回答 {number}" in report
    assert "actual avoided=0" in report
    assert "不值得替换/冻结" in report
    assert "theta_shank = q_hip - q_knee" in report


def test_final_status_remains_offline_not_human_not_robot_ready(metadata) -> None:
    assert PROTOTYPE_STATUS == "DEFAULT_OFF_RESEARCH_SHADOW_ONLY"
    assert metadata["offline_method_status"] == OFFLINE_METHOD_STATUS
    assert OFFLINE_METHOD_STATUS == "OFFLINE_METHOD_REQUIRES_REVISION"
    assert metadata["human_readiness"] == NOT_HUMAN_READY
    assert metadata["robot_motion_approval"] == NOT_ROBOT_MOTION_APPROVED
    assert metadata["real_robot_connected"] is False
    assert metadata["formal_threshold_created"] is False
