from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
)
from .formal_protocol import (
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
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    POLICY_IDS,
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    run_policy,
    select_exploit_candidate,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_OUTPUT_DIRECTORY as PREVIOUS_STAGE_DIRECTORY,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    FIGURE_FILENAMES,
    REPORT_FILENAMES,
)
from .sequential_personalization import SearchAlpha
from .sequential_personalization_convergence_stopping_audit import (
    AUDIT_PROTOCOL_ID,
    BOUNDARY_OPTIMUM_DIAGNOSTIC,
    CORRECT_CONSERVATIVE_STOP,
    CORRECT_LOCAL_STOP,
    DECISION_VALUE_OBSERVED,
    EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
    HORIZON_STATUS,
    INFORMATIVE_BUT_LOW_DECISION_VALUE,
    MISSED_IMPROVEMENT,
    OFFLINE_METHOD_REQUIRES_REVISION,
    POST_DECISION_TRUTH_ROLE,
    PREMATURE_CONSERVATIVE_STOP,
    REFERENCE_LOCALLY_COMPETITIVE,
    TRIAL_BUDGETS,
    build_marginal_improvement,
    corridor_boundary_distances,
    normalized_alpha_distance,
    parameter_bound_distances,
)


@pytest.fixture(scope="module")
def metadata():
    return json.loads((DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def natural():
    return pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "natural_stopping_summary.csv")


@pytest.fixture(scope="module")
def p2_natural(natural):
    return natural.loc[natural["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]


@pytest.fixture(scope="module")
def rounds():
    cases = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_cases.csv")
    summary = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_summary.csv")
    return cases, summary


def test_protocol_identifier_is_convergence_stopping_audit_v1():
    assert AUDIT_PROTOCOL_ID == "SEQUENTIAL_PERSONALIZATION_CONVERGENCE_AND_STOPPING_AUDIT_V1"


def test_extended_horizon_is_twenty_virtual_trials():
    assert EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON == 20


def test_extended_horizon_is_not_human_threshold():
    assert HORIZON_STATUS == "OFFLINE_VIRTUAL_DIAGNOSTIC_CAP_NOT_HUMAN_THRESHOLD"


def test_budget_sensitivity_uses_required_horizons():
    assert TRIAL_BUDGETS == (3, 6, 12, 20)


def test_default_first_protocol_still_rejects_non_six_budget():
    with pytest.raises(ValueError, match="first research protocol"):
        run_policy(
            None,
            POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
            None,
            None,
            trial_budget=20,
        )


def test_extended_horizon_requires_explicit_opt_in(metadata):
    assert metadata["extended_offline_diagnostic_horizon"] == 20
    assert metadata["diagnostic_horizon_status"] == HORIZON_STATUS


def test_policy_definitions_are_byte_semantically_unchanged(metadata):
    previous = json.loads(
        (PREVIOUS_STAGE_DIRECTORY / "policy_definition.json").read_text(encoding="utf-8")
    )
    assert previous == policy_definitions()
    assert metadata["policy_definition_changed"] is False


def test_decision_guard_source_is_frozen(metadata):
    observed = hashlib.sha256(
        inspect.getsource(apply_research_decision_guard).encode("utf-8")
    ).hexdigest()
    assert observed == metadata["decision_guard_source_sha256"]


def test_exploit_selector_source_is_frozen(metadata):
    observed = hashlib.sha256(
        inspect.getsource(select_exploit_candidate).encode("utf-8")
    ).hexdigest()
    assert observed == metadata["exploit_selector_source_sha256"]


def test_exploration_ranker_source_is_frozen(metadata):
    observed = hashlib.sha256(
        inspect.getsource(rank_exploration_frontier).encode("utf-8")
    ).hexdigest()
    assert observed == metadata["exploration_ranker_source_sha256"]


def test_algorithm_equivalence_tolerance_is_unchanged():
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005


def test_support_gate_is_unchanged():
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0


def test_mechanical_objective_is_unchanged():
    assert MECHANICAL_OBJECTIVE_VERSION == "mechanical_joint_torque_objective_v1"


def test_active_reference_sha_is_unchanged():
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"


def test_formal_rom_is_unchanged():
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)


def test_theta_shank_remains_subtraction():
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"


def test_five_parameter_model_is_unchanged():
    assert tuple(PARAMETER_NAMES) == (
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    )


def test_all_p2_cases_naturally_stop_before_cap(p2_natural):
    assert len(p2_natural) == 9
    assert p2_natural["natural_stop_reached"].astype(bool).all()
    assert not p2_natural["diagnostic_cap_reached"].astype(bool).any()


def test_six_trial_budget_is_not_mislabeled_natural_stop():
    budget = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "trial_budget_sensitivity.csv")
    selected = budget.loc[
        budget["case_id"].eq("baseline__matched_linear")
        & budget["offline_research_budget"].eq(6)
    ].iloc[0]
    assert not selected["natural_stop_observed_within_horizon"]
    assert selected["stop_reason"] == "DIAGNOSTIC_BUDGET_TRUNCATION_AT_6"


def test_diagnostic_cap_and_natural_stop_are_separate_fields(p2_natural):
    assert {"natural_stop_reached", "natural_stop_iteration", "diagnostic_cap_reached"}.issubset(p2_natural.columns)


def test_baseline_natural_stop_occurs_after_thirteen_executed_trials(p2_natural):
    row = p2_natural.loc[p2_natural["case_id"].eq("baseline__matched_linear")].iloc[0]
    assert row["executed_trial_count"] == 13
    assert row["natural_stop_iteration"] == 14


def test_knee_stiff_natural_stop_occurs_after_eight_explores(p2_natural):
    row = p2_natural.loc[p2_natural["case_id"].eq("knee_stiff__matched_linear")].iloc[0]
    assert row["executed_trial_count"] == 8
    assert row["natural_stop_iteration"] == 9


def test_parameter_bound_distance_is_correct_at_reference():
    result = parameter_bound_distances(SearchAlpha())
    assert result["distance_to_parameter_bounds_formal_steps"] == pytest.approx(8.0)
    assert not result["on_generator_parameter_boundary"]


def test_parameter_bound_distance_is_zero_at_knee_lower_bound():
    result = parameter_bound_distances(SearchAlpha(knee_delta_deg=-5.0))
    assert result["distance_to_knee_lower_bound"] == 0.0
    assert result["distance_to_parameter_bounds_formal_steps"] == 0.0
    assert result["on_generator_parameter_boundary"]


def test_normalized_alpha_distance_uses_formal_grid_steps():
    assert normalized_alpha_distance((0, 0, 0), (0.25, 0.25, 0.0025)) == pytest.approx(np.sqrt(3.0))


def test_corridor_distance_is_finite_and_signed():
    result = corridor_boundary_distances(SearchAlpha(hip_delta_deg=0.25))
    assert np.isfinite(result["distance_to_joint_corridor_boundary_deg"])
    assert np.isfinite(result["distance_to_pull_corridor_boundary_mm"])
    assert result["outside_joint_corridor"]
    assert result["outside_pull_corridor"]


def test_marginal_improvement_calculation_is_exact():
    history = pd.DataFrame(
        {
            "best_actual_J_before": [1.0, 0.99],
            "best_actual_J_after": [0.99, 0.985],
            "accepted_improvement": [True, True],
            "trial_purpose": ["EXPLOIT", "EXPLOIT"],
        }
    )
    output = build_marginal_improvement(history)
    np.testing.assert_allclose(output["marginal_best_J_improvement"], [0.01, 0.005])
    np.testing.assert_allclose(output["cumulative_J_improvement"], [0.01, 0.015])


def test_marginal_audit_does_not_create_threshold():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "marginal_improvement_by_trial.csv")
    assert not table["new_minimum_useful_improvement_threshold_created"].astype(bool).any()


def test_post_decision_truth_role_is_explicit():
    assert POST_DECISION_TRUTH_ROLE == "POST_DECISION_EVALUATION_ONLY_NO_POLICY_FEEDBACK"
    cases = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_cases.csv")
    assert set(cases["truth_role"]) == {POST_DECISION_TRUTH_ROLE}


def test_truth_is_only_computed_after_decision_frozen():
    cases = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_cases.csv")
    assert cases["policy_decision_frozen_before_truth"].astype(bool).all()


def test_post_decision_truth_never_feeds_back_to_policy():
    cases = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_cases.csv")
    assert not cases["truth_fed_back_to_policy"].astype(bool).any()


def test_seven_missed_improvement_rounds_are_found(rounds):
    cases, summary = rounds
    assert cases[["case_id", "iteration"]].drop_duplicates().shape[0] == 7
    all_row = summary.loc[summary["scenario_group"].eq("ALL_SCENARIOS")].iloc[0]
    assert all_row["missed_opportunity_rounds"] == 7
    assert all_row["rounds_with_true_local_improvement_available"] == 27


def test_missed_improvement_rate_formula(rounds):
    _, summary = rounds
    all_row = summary.loc[summary["scenario_group"].eq("ALL_SCENARIOS")].iloc[0]
    assert all_row["missed_improvement_rate"] == pytest.approx(7 / 27)
    assert not all_row["threshold_tuned_from_metric"]


def test_missed_candidates_have_required_fields():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_cases.csv")
    assert {
        "alpha_hip",
        "alpha_knee",
        "alpha_phase",
        "delta_J_pred",
        "delta_J_truth",
        "decision_guard_margin",
        "model_supported",
        "domain_coverage",
        "why_guard_rejected",
    }.issubset(table.columns)
    assert set(table["diagnostic_status"]) == {MISSED_IMPROVEMENT}


def test_matched_first_round_misses_are_unsupported_provenance():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "missed_improvement_cases.csv")
    matched = table.loc[table["case_class"].eq("MATCHED_POSITIVE_CONTROL")]
    assert len(matched) == 3
    assert not matched["model_supported"].astype(bool).any()
    assert set(matched["why_guard_rejected"]) == {"UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE"}


def test_four_zero_action_mismatch_stops_are_premature():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "correct_stop_audit.csv")
    scenarios = {
        "nonlinear_stiffness_mild",
        "hip_knee_coupling_mild",
        "structured_residual",
        "combined_mild",
    }
    selected = table.loc[table["scenario_name"].isin(scenarios)]
    assert len(selected) == 4
    assert set(selected["conservative_stop_classification"]) == {PREMATURE_CONSERVATIVE_STOP}


def test_nonlinear_damping_stop_is_correct_conservative():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "correct_stop_audit.csv")
    row = table.loc[table["scenario_name"].eq("nonlinear_damping_mild")].iloc[0]
    assert row["correct_stop_classification"] == CORRECT_LOCAL_STOP
    assert row["conservative_stop_classification"] == CORRECT_CONSERVATIVE_STOP


def test_knee_stiff_reference_is_locally_competitive():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "knee_stiff_extended_audit.csv")
    assert table["knee_stiff_diagnostic_status"].str.contains(REFERENCE_LOCALLY_COMPETITIVE).all()
    assert not table["true_local_improvement_available"].astype(bool).any()
    assert table["best_local_delta_J_truth"].min() == pytest.approx(-0.004467, abs=1e-6)


def test_knee_stiff_has_no_later_exploit():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "knee_stiff_extended_audit.csv")
    executed = table.loc[table["trial_purpose"].notna()]
    assert len(executed) == 8
    assert set(executed["trial_purpose"]) == {"EXPLORE"}


def test_knee_stiff_support_region_continues_to_expand():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "knee_stiff_extended_audit.csv")
    executed = table.loc[table["supported_point_count"].notna()]
    assert executed["supported_point_count"].is_monotonic_increasing
    assert executed.iloc[-1]["supported_point_count"] > executed.iloc[0]["supported_point_count"]


def test_knee_stiff_prediction_landscape_does_not_change():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "knee_stiff_extended_audit.csv")
    assert table["RMS_map_change"].fillna(0.0).eq(0.0).all()
    assert table["max_abs_map_change"].fillna(0.0).eq(0.0).all()


def test_knee_stiff_matched_parameters_remain_stable():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "model_parameter_evolution.csv")
    selected = table.loc[table["case_id"].eq("knee_stiff__matched_linear")]
    delta_columns = [column for column in selected if column.endswith("_delta")]
    assert selected[delta_columns].abs().to_numpy().max() == 0.0


def test_information_gain_and_decision_value_are_separate():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "exploration_decision_value.csv")
    low = table.loc[table["exploration_decision_value_status"].eq(INFORMATIVE_BUT_LOW_DECISION_VALUE)]
    assert len(low) == 29
    assert (low["incremental_log_information_gain"] > 0.0).all()
    assert not low["decision_value_observed_within_2_rounds"].astype(bool).any()


def test_three_explores_enable_exploit_within_one_and_two_rounds():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "exploration_decision_value.csv")
    assert int(table["enabled_exploit_within_1_round"].sum()) == 3
    assert int(table["enabled_exploit_within_2_rounds"].sum()) == 3
    observed = table.loc[table["decision_value_observed_within_2_rounds"].astype(bool)]
    assert set(observed["exploration_decision_value_status"]) == {DECISION_VALUE_OBSERVED}


def test_repeated_knee_exploration_is_audited_without_new_stop():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "repeated_exploration_audit.csv")
    row = table.loc[table["case_id"].eq("knee_stiff__matched_linear")].iloc[0]
    assert row["consecutive_explore_count"] == 8
    assert row["information_gain_trend_per_round"] < 0.0
    assert not row["future_exploit_within_2_rounds_observed"]
    assert not row["diminishing_return_stop_created"]


def test_boundary_optimum_diagnostic_is_not_declared_an_error():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "boundary_chasing_audit.csv")
    selected = table.loc[table["final_optimum_diagnostic_status"].eq(BOUNDARY_OPTIMUM_DIAGNOSTIC)]
    assert selected["case_id"].nunique() == 4
    assert not selected["boundary_is_error_claim"].astype(bool).any()


def test_sustained_exploit_paths_march_to_knee_lower_bound():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "boundary_chasing_audit.csv")
    case_ids = {
        "baseline__matched_linear",
        "hip_stiff__matched_linear",
        "heavy_leg__matched_linear",
        "baseline__nonlinear_damping_mild",
    }
    selected = table.loc[table["case_id"].isin(case_ids)]
    assert selected["exploit_monotonic_march_toward_boundary"].astype(bool).all()
    assert selected["final_best_on_generator_boundary"].astype(bool).all()


def test_subject_path_divergence_finds_three_identical_paths():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "subject_path_divergence.csv")
    identical = table.loc[table["final_paths_identical"].astype(bool)]
    assert len(identical) == 3
    assert (identical["final_alpha_difference_formal_steps"] == 0.0).all()


def test_knee_stiff_final_path_differs_by_twenty_formal_steps():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "subject_path_divergence.csv")
    selected = table.loc[
        table["subject_a"].eq("knee_stiff") | table["subject_b"].eq("knee_stiff")
    ]
    assert len(selected) == 3
    assert (selected["final_alpha_difference_formal_steps"] == 20.0).all()


def test_full_prediction_map_count_remains_21025():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "prediction_landscape_evolution.csv")
    assert table["full_map_point_count"].eq(21025).all()


def test_matched_map_values_remain_stable_while_support_grows():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "prediction_landscape_evolution.csv")
    matched = table.loc[table["scenario_name"].eq("matched_linear")]
    assert matched["RMS_map_change"].eq(0.0).all()
    assert matched["max_abs_map_change"].eq(0.0).all()


def test_mismatch_equivalent_parameters_are_allowed_to_change():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "model_parameter_evolution.csv")
    selected = table.loc[table["case_id"].eq("baseline__nonlinear_damping_mild")]
    delta_columns = [column for column in selected if column.endswith("_delta")]
    assert selected[delta_columns].abs().to_numpy().max() > 0.0
    assert set(selected["parameter_interpretation"]) == {
        "LOCAL_EQUIVALENT_DYNAMICS_NOT_PHYSIOLOGICAL_TISSUE_CHANGE"
    }


def test_budget_sensitivity_is_reproducible_and_complete():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "trial_budget_sensitivity.csv")
    assert len(table) == 9 * len(TRIAL_BUDGETS)
    assert set(table["offline_research_budget"]) == set(TRIAL_BUDGETS)
    assert not table["human_trial_recommendation"].astype(bool).any()


def test_six_trial_best_equals_extended_final_best_in_all_cases():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "best_trajectory_stability.csv")
    values = table.groupby("case_id")["six_trial_best_equals_extended_final_best"].first()
    assert len(values) == 9
    assert values.astype(bool).all()


def test_extended_false_improvement_remains_zero_for_all_policies():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "false_improvement_extended_audit.csv")
    assert set(table["policy_id"]) == set(POLICY_IDS)
    assert not table["executed_false_improvement"].astype(bool).any()


def test_freeze_readiness_requires_revision(metadata):
    readiness = metadata["readiness"]
    assert readiness["status"] == OFFLINE_METHOD_REQUIRES_REVISION
    assert readiness["missed_opportunity_round_count"] == 7
    assert readiness["informative_but_low_decision_value_explore_count"] == 29
    assert readiness["boundary_optimum_case_count"] == 4


def test_readiness_never_implies_human_or_robot_approval(metadata):
    readiness = metadata["readiness"]
    assert readiness["not_human_ready"]
    assert readiness["not_robot_motion_approved"]
    assert not metadata["formal_human_ready_model_created"]
    assert not metadata["robot_motion_approved"]
    assert not metadata["real_robot_connected"]


def test_unapproved_threshold_statuses_remain_frozen():
    assert INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS.endswith("REQUIRES_REVIEW")
    assert GLOBAL_MODEL_RELIABILITY_RULE_STATUS.endswith("NOT_FROZEN")


def test_new_audit_modules_do_not_import_robot_packages():
    for filename in (
        "sequential_personalization_convergence_stopping_audit.py",
        "run_sequential_personalization_convergence_stopping_audit.py",
    ):
        tree = ast.parse(Path(__file__).with_name(filename).read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not any(
            module.startswith(("hardware", "control", "collection", "safety", "xCoreSDK"))
            for module in modules
        )


def test_formal_artifact_contract_is_complete():
    expected = {*CSV_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES, "metadata.json"}
    assert expected.issubset({path.name for path in DEFAULT_OUTPUT_DIRECTORY.iterdir()})


def test_all_hashed_outputs_match_metadata(metadata):
    for filename, expected in metadata["output_sha256"].items():
        observed = hashlib.sha256((DEFAULT_OUTPUT_DIRECTORY / filename).read_bytes()).hexdigest()
        assert observed == expected


def test_data_leakage_report_records_post_decision_only_truth():
    report = (DEFAULT_OUTPUT_DIRECTORY / "DATA_LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert "after each EXPLOIT, EXPLORE, or STOP decision was frozen: `true`" in report
    assert "Post-decision truth was fed back to policy: `false`" in report
    assert "Held-out final-test data were not loaded" in report


def test_all_required_figures_are_valid_nonempty_pngs():
    for filename in FIGURE_FILENAMES:
        path = DEFAULT_OUTPUT_DIRECTORY / filename
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000

