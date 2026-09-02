from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    build_predicted_map,
    build_trajectory_component_cache,
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    validate_active_reference_file,
)
from .mechanical_objective import (
    MechanicalTorqueMetrics,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
    evaluate_mechanical_objective,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    DECISION_GUARD_STATUS,
    DECISION_UNCERTAINTY_VERSION,
    FRONTIER_DISTANCE_ROLE,
    GEOMETRICALLY_INADMISSIBLE,
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
    PAIRWISE_BOUND_STATUS,
    POLICY_DECISION_GUARDED_EXPLOIT_ONLY,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    POLICY_IDS,
    POLICY_SUPPORTED_ONLY_GREEDY,
    PROTOCOL_ID,
    RESEARCH_EXPLOIT_ELIGIBLE,
    RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET,
    ResearchDecisionUncertainty,
    SUPPORTED_BUT_DECISION_UNRELIABLE,
    SUPPORT_ROLE,
    SelectionGatedVirtualTruthOracle,
    TRIAL_PURPOSE_EXPLORE,
    UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE,
    _model_for_iteration,
    alpha_key_from_row,
    apply_research_decision_guard,
    build_initial_research_state,
    build_local_exploration_frontier,
    evaluate_validation_pairwise_uncertainty,
    frozen_baseline_metadata,
    local_prediction_candidates,
    policy_definitions,
    rank_exploration_frontier,
    run_policy,
    select_exploit_candidate,
)
from .run_research_decision_guarded_sequential_personalization import (
    ANALYSIS_CASES,
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_PARAMETER_MAP_PATH,
    FIGURE_FILENAMES,
    REPORT_FILENAMES,
)
from .continuous_reference_neighborhood import generate_personalized_trajectory
from .sequential_personalization import SearchAlpha, TrustRegionSteps, accept_actual_trial


@pytest.fixture(scope="module")
def lattice() -> pd.DataFrame:
    raw = pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    return geometrically_valid_parameter_lattice(raw)


@pytest.fixture(scope="module")
def cache(lattice: pd.DataFrame):
    return build_trajectory_component_cache(lattice)


@pytest.fixture(scope="module")
def baseline_state():
    return build_initial_research_state("baseline", "matched_linear")


@pytest.fixture(scope="module")
def baseline_map(baseline_state, lattice, cache):
    model = _model_for_iteration(
        baseline_state, baseline_state.parameters, baseline_state.domain_data, 0
    )
    result, _ = build_predicted_map(model, lattice, cache, batch_size=256)
    return result


@pytest.fixture(scope="module")
def baseline_p2(baseline_state, lattice, cache):
    return run_policy(
        baseline_state,
        POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
        lattice,
        cache,
    )


def _uncertainty(bound: float = 0.01, pair_count: int = 1):
    audit = pd.DataFrame(
        {
            "e_delta_J": [bound] if pair_count else [],
            "current_item_prediction_error_abs": [0.01] if pair_count else [],
            "candidate_item_prediction_error_abs": [0.02] if pair_count else [],
        }
    )
    return ResearchDecisionUncertainty(
        case_id="unit",
        iteration=0,
        pairwise_audit=audit,
        maximum_observed_e_delta_j=bound,
        p95_observed_e_delta_j=bound,
        p99_observed_e_delta_j=bound,
        validation_pair_count=pair_count,
        bound_used_by_guard=bound,
    )


def _guard_table(*, candidate_j: float, candidate_supported: bool = True, geometry: bool = True):
    return pd.DataFrame(
        [
            {
                "trajectory_id": "reference",
                "hip_delta": 0.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": 1.0,
                "model_supported": True,
                "domain_coverage": 100.0,
                "distance_to_supported_region": 0.0,
                "geometrically_admissible": True,
            },
            {
                "trajectory_id": "candidate",
                "hip_delta": 1.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": candidate_j,
                "model_supported": candidate_supported,
                "domain_coverage": 100.0 if candidate_supported else 80.0,
                "distance_to_supported_region": 0.0 if candidate_supported else 1.0,
                "geometrically_admissible": geometry,
            },
        ]
    )


def test_protocol_identifier_is_research_only_v1():
    assert PROTOCOL_ID == "RESEARCH_ONLY_DECISION_GUARDED_SEQUENTIAL_PERSONALIZATION_V1"


def test_rom_protocol_v2_is_unchanged():
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)


def test_active_reference_sha_is_unchanged():
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    validate_active_reference_file()


def test_theta_shank_definition_is_subtraction():
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    generated = generate_personalized_trajectory()
    np.testing.assert_allclose(
        generated.trajectory["theta_shank_rad"],
        generated.trajectory["q_hip_rad"] - generated.trajectory["q_knee_rad"],
        atol=1e-12,
    )


def test_five_parameter_model_is_unchanged():
    assert tuple(PARAMETER_NAMES) == (
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    )


def test_mechanical_objective_formula_is_unchanged():
    reference = MechanicalTorqueMetrics(2.0, 4.0, 3.0, 5.0, 1.0, 2.0)
    candidate = MechanicalTorqueMetrics(1.0, 2.0, 2.0, 4.0, 0.5, 1.0)
    result = evaluate_mechanical_objective(
        trajectory_id="unit",
        metrics=candidate,
        reference_metrics=reference,
        hip_rms_deviation_deg=0.0,
        knee_rms_deviation_deg=0.0,
    )
    assert result.mechanical_cost_j_rms == pytest.approx(math.sqrt((0.5**2 + 0.5**2) / 2))


def test_existing_equivalence_tolerance_keeps_algorithm_semantics():
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert not accept_actual_trial(0.996, 1.0)
    assert accept_actual_trial(0.994, 1.0)


def test_three_policy_variants_are_fixed():
    assert POLICY_IDS == (
        POLICY_SUPPORTED_ONLY_GREEDY,
        POLICY_DECISION_GUARDED_EXPLOIT_ONLY,
        POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    )


def test_p0_is_explicitly_a_nonrecommended_sanity_comparator():
    definitions = policy_definitions()
    assert definitions["policies"][POLICY_SUPPORTED_ONLY_GREEDY]["role"] == "SANITY_COMPARATOR_NOT_RECOMMENDED"


def test_decision_guard_is_not_a_formal_threshold():
    definitions = policy_definitions()
    assert definitions["decision_guard"]["formal_threshold"] is False
    assert definitions["decision_guard"]["uncertainty_bound_status"] == PAIRWISE_BOUND_STATUS


def test_support_is_provenance_not_reliability():
    assert SUPPORT_ROLE == "DATA_PROVENANCE_NOT_RELIABILITY_APPROVAL"


def test_distance_is_locality_not_reliability():
    assert FRONTIER_DISTANCE_ROLE == "LOCALITY_CONTROL_NOT_RELIABILITY_SCORE"


def test_support_gate_remains_ninety_percent():
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert policy_definitions()["support_coverage_gate_percent"] == 90.0


def test_research_trial_budget_reuses_existing_offline_budget():
    assert RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET == 6
    assert policy_definitions()["trial_budget_status"] == "RESEARCH_ONLY_NOT_HUMAN_SAFETY_THRESHOLD"


def test_initial_model_remains_diagnostic_only(baseline_state):
    assert baseline_state.model_status == "DIAGNOSTIC_ONLY"
    assert baseline_state.human_readiness == "NOT_HUMAN_READY"
    assert baseline_state.approval_status == NOT_APPROVED_FOR_ROBOT_PERSONALIZATION


def test_initial_fitting_data_contains_no_truth_parameter_columns(baseline_state):
    forbidden = {"truth_parameters", "scenario_parameters", "heldout_final"}
    assert forbidden.isdisjoint(baseline_state.fitting_data.columns)


def test_decision_uncertainty_uses_designated_validation_only(baseline_state):
    uncertainty = evaluate_validation_pairwise_uncertainty(
        baseline_state, baseline_state.parameters, iteration=0
    )
    assert uncertainty.validation_pair_count >= 1
    assert set(uncertainty.pairwise_audit["calibration_data_role"]) == {"DESIGNATED_VALIDATION_ONLY"}
    assert not uncertainty.pairwise_audit["heldout_final_test_used"].any()


def test_pairwise_delta_j_error_is_directly_audited(baseline_state):
    uncertainty = evaluate_validation_pairwise_uncertainty(
        baseline_state, baseline_state.parameters, iteration=0
    )
    audit = uncertainty.pairwise_audit
    np.testing.assert_allclose(
        audit["e_delta_J"],
        np.abs(audit["delta_J_pred"] - audit["delta_J_actual"]),
    )


def test_max_p95_p99_are_research_candidates(baseline_state):
    uncertainty = evaluate_validation_pairwise_uncertainty(
        baseline_state, baseline_state.parameters, iteration=0
    )
    assert uncertainty.bound_used_by_guard == uncertainty.maximum_observed_e_delta_j
    assert uncertainty.bound_status == "RESEARCH_CANDIDATE_ONLY_NOT_FORMAL_THRESHOLD"
    assert uncertainty.p95_observed_e_delta_j <= uncertainty.maximum_observed_e_delta_j + 1e-15
    assert uncertainty.p99_observed_e_delta_j <= uncertainty.maximum_observed_e_delta_j + 1e-15


def test_support_alone_cannot_approve_exploit():
    guarded = apply_research_decision_guard(
        _guard_table(candidate_j=0.997), SearchAlpha(), _uncertainty(0.001)
    )
    candidate = guarded.loc[guarded["trajectory_id"].eq("candidate")].iloc[0]
    assert candidate["decision_guard_status"] == SUPPORTED_BUT_DECISION_UNRELIABLE
    assert not candidate["research_exploit_eligible"]


def test_supported_strong_improvement_can_pass_research_guard():
    guarded = apply_research_decision_guard(
        _guard_table(candidate_j=0.98), SearchAlpha(), _uncertainty(0.01)
    )
    candidate = guarded.loc[guarded["trajectory_id"].eq("candidate")].iloc[0]
    assert candidate["decision_guard_status"] == RESEARCH_EXPLOIT_ELIGIBLE
    assert candidate["improvement_margin"] == pytest.approx(0.005)


def test_unsupported_candidate_can_have_j_but_not_exploit():
    guarded = apply_research_decision_guard(
        _guard_table(candidate_j=0.8, candidate_supported=False),
        SearchAlpha(),
        _uncertainty(0.01),
    )
    candidate = guarded.loc[guarded["trajectory_id"].eq("candidate")].iloc[0]
    assert math.isfinite(candidate["J_pred"])
    assert candidate["decision_guard_status"] == UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE


def test_geometrically_invalid_candidate_cannot_exploit():
    guarded = apply_research_decision_guard(
        _guard_table(candidate_j=0.8, geometry=False), SearchAlpha(), _uncertainty(0.01)
    )
    candidate = guarded.loc[guarded["trajectory_id"].eq("candidate")].iloc[0]
    assert candidate["decision_guard_status"] == GEOMETRICALLY_INADMISSIBLE


def test_no_validation_evidence_cannot_approve_exploit():
    guarded = apply_research_decision_guard(
        _guard_table(candidate_j=0.8), SearchAlpha(), _uncertainty(0.0, pair_count=0)
    )
    assert not guarded["research_exploit_eligible"].any()


def test_local_exploit_candidates_use_existing_trust_region(baseline_map):
    local = local_prediction_candidates(baseline_map, SearchAlpha(), TrustRegionSteps())
    assert len(local) == 7
    assert all(
        max(abs(row[0]), abs(row[1]), abs(row[2]) * 100.0) <= 1.0 + 1e-12
        for row in local[["hip_delta", "knee_delta", "phase_delta"]].itertuples(index=False, name=None)
    )


def test_global_map_contains_all_21025_geometric_points(lattice, baseline_map):
    assert len(lattice) == 21025
    assert len(baseline_map) == 21025
    assert np.isfinite(baseline_map["J_pred"]).all()


def test_global_minimum_is_never_direct_execution_instruction(baseline_p2):
    assert not baseline_p2.summary["global_minimum_executed_directly"]
    assert not baseline_p2.prediction_map_history["global_minimum_executed_directly"].any()


def test_frontier_is_exactly_one_formal_step(baseline_map):
    frontier = build_local_exploration_frontier(baseline_map, {(0.0, 0.0, 0.0)})
    assert frontier["adjacent_formal_step"].all()
    assert not frontier["intermediate_layer_skipped"].any()
    for key in frontier.apply(alpha_key_from_row, axis=1):
        assert key in {
            (GRID_HIP_STEP_DEG, 0.0, 0.0),
            (-GRID_HIP_STEP_DEG, 0.0, 0.0),
            (0.0, GRID_KNEE_STEP_DEG, 0.0),
            (0.0, -GRID_KNEE_STEP_DEG, 0.0),
            (0.0, 0.0, GRID_PHASE_STEP),
            (0.0, 0.0, -GRID_PHASE_STEP),
        }


def test_exploration_ranking_is_information_first_without_truth(baseline_map, baseline_state):
    frontier = build_local_exploration_frontier(baseline_map, {(0.0, 0.0, 0.0)})
    ranked = rank_exploration_frontier(
        frontier.loc[~frontier["model_supported"].astype(bool)].head(2),
        baseline_state.fitting_data,
        baseline_state.parameters,
    )
    assert not ranked["truth_used_for_exploration_rank"].any()
    assert not ranked["J_pred_used_as_primary_exploration_rank"].any()
    assert ranked.iloc[0]["incremental_log_information_gain"] >= ranked.iloc[-1]["incremental_log_information_gain"]


def test_virtual_truth_requires_current_selection_token():
    oracle = SelectionGatedVirtualTruthOracle("baseline", "matched_linear")
    generated = generate_personalized_trajectory()
    trajectory = generated.trajectory.copy()
    trajectory["trajectory_id"] = generated.metadata["trajectory_id"]
    token = oracle.declare_selected(generated.metadata["trajectory_id"], TRIAL_PURPOSE_EXPLORE)
    wrong = type(token)("wrong", token.trajectory_id, token.trial_purpose, token.serial)
    with pytest.raises(PermissionError):
        oracle.execute(wrong, trajectory)


def test_virtual_selection_token_executes_exactly_once():
    oracle = SelectionGatedVirtualTruthOracle("baseline", "matched_linear")
    generated = generate_personalized_trajectory(hip_amplitude_delta_deg=0.25)
    trajectory = generated.trajectory.copy()
    trajectory["trajectory_id"] = generated.metadata["trajectory_id"]
    token = oracle.declare_selected(generated.metadata["trajectory_id"], TRIAL_PURPOSE_EXPLORE)
    oracle.execute(token, trajectory)
    assert oracle.truth_calls == 1
    with pytest.raises(PermissionError):
        oracle.execute(token, trajectory)


def test_proposal_never_accesses_truth(baseline_p2):
    assert baseline_p2.truth_access_audit["truth_calls_unchanged_during_every_proposal"]


def test_exactly_one_trial_per_iteration(baseline_p2):
    assert baseline_p2.truth_access_audit["exactly_one_trajectory_per_iteration"]
    assert baseline_p2.trial_history["executed_trial_count_this_iteration"].eq(1).all()


def test_baseline_p2_explores_then_exploits(baseline_p2):
    purposes = baseline_p2.trial_history["trial_purpose"].tolist()
    assert purposes[0] == "EXPLORE"
    assert "EXPLOIT" in purposes[1:]


def test_false_improvement_cannot_update_best_by_actual_rule():
    assert not accept_actual_trial(1.01, 1.0)


def test_valid_exploration_data_updates_model_even_if_not_accepted(baseline_p2):
    explore = baseline_p2.trial_history.loc[
        baseline_p2.trial_history["trial_purpose"].eq("EXPLORE")
    ].iloc[0]
    assert not explore["accepted_improvement"]
    assert explore["valid_data_added_to_model_update"]
    assert explore["model_updates_this_iteration"] == 1


def test_model_updates_at_most_once_per_trial(baseline_p2):
    assert baseline_p2.trial_history["model_updates_this_iteration"].le(1).all()
    assert baseline_p2.parameter_history["updates_this_trial"].le(1).all()


def test_model_is_fixed_within_every_trial(baseline_p2):
    assert baseline_p2.parameter_history["within_trial_model_fixed"].all()


def test_whole_map_recomputed_after_every_model_update(baseline_p2):
    assert baseline_p2.summary["whole_map_recomputation_count"] == baseline_p2.summary["model_update_count"] + 1
    assert baseline_p2.prediction_map_history["complete_map_recomputed"].all()


def test_support_update_does_not_update_reliability_from_support(baseline_p2):
    assert not baseline_p2.known_region_history["reliability_updated_from_support"].any()
    assert set(baseline_p2.known_region_history["support_role"]) == {SUPPORT_ROLE}


def test_uncertainty_history_never_uses_heldout(baseline_p2):
    assert not baseline_p2.uncertainty_history["heldout_final_test_used"].any()
    assert not baseline_p2.uncertainty_pairwise_audit["heldout_final_test_used"].any()


def test_baseline_matched_sanity_is_reproducible(baseline_p2):
    assert baseline_p2.summary["number_of_executed_trials"] == 6
    assert baseline_p2.summary["number_of_executed_false_improvements"] == 0
    assert baseline_p2.summary["final_best_actual_J"] == pytest.approx(0.9712287596140415)


@pytest.mark.parametrize(
    "scenario_name",
    [
        "nonlinear_stiffness_mild",
        "hip_knee_coupling_mild",
        "nonlinear_damping_mild",
        "structured_residual",
        "combined_mild",
    ],
)
def test_all_required_mismatch_scenarios_are_registered(scenario_name):
    assert ("baseline", scenario_name, "MILD_MODEL_MISMATCH") in ANALYSIS_CASES


def test_four_matched_subjects_are_registered():
    matched = {subject for subject, scenario, _ in ANALYSIS_CASES if scenario == "matched_linear"}
    assert matched == {"baseline", "hip_stiff", "knee_stiff", "heavy_leg"}


def test_frozen_statuses_remain_unapproved():
    assert INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS.endswith("REQUIRES_REVIEW")
    assert GLOBAL_MODEL_RELIABILITY_RULE_STATUS.endswith("NOT_FROZEN")
    metadata = frozen_baseline_metadata()
    assert not metadata["formal_human_ready_theta_0_created"]
    assert not metadata["formal_personalization_approval_created"]
    assert not metadata["real_robot_connected"]


def test_core_and_runner_do_not_import_robot_packages():
    for path in (
        Path(__file__).with_name("research_decision_guarded_sequential_personalization.py"),
        Path(__file__).with_name("run_research_decision_guarded_sequential_personalization.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
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
    expected = {
        *CSV_FILENAMES,
        "policy_definition.json",
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
        "metadata.json",
    }
    assert expected.issubset({path.name for path in DEFAULT_OUTPUT_DIRECTORY.iterdir()})


def test_formal_artifact_metadata_matches_runtime():
    metadata = json.loads((DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["protocol_id"] == PROTOCOL_ID
    assert metadata["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["geometrically_admissible_point_count"] == 21025
    assert metadata["whole_map_recomputed_after_every_successful_update"]
    assert not metadata["heldout_final_test_used"]
    assert not metadata["real_robot_connected"]


def test_formal_policy_outputs_are_deterministic_and_complete():
    summary = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "scenario_sequential_summary.csv")
    assert len(summary) == len(ANALYSIS_CASES) * len(POLICY_IDS)
    assert set(summary["policy_id"]) == set(POLICY_IDS)
    assert set(summary["case_id"]) == {f"{subject}__{scenario}" for subject, scenario, _ in ANALYSIS_CASES}


def test_formal_policy_comparison_has_no_robot_recommendation():
    comparison = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "policy_comparison.csv")
    assert set(comparison["policy_id"]) == set(POLICY_IDS)
    assert comparison["research_only"].astype(bool).all()
    assert not comparison["recommended_for_human_or_robot"].astype(bool).any()


def test_formal_data_leakage_report_records_no_heldout_or_preselection_truth():
    report = (DEFAULT_OUTPUT_DIRECTORY / "DATA_LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert "Held-out final test: not loaded" in report
    assert "truth-call count remained unchanged: `true`" in report
    assert "Human/robot approval created: `false`" in report


def test_research_uncertainty_csv_contains_pairwise_errors_and_research_bounds():
    table = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "research_decision_uncertainty.csv")
    required = {
        "delta_J_pred",
        "delta_J_actual",
        "e_delta_J",
        "maximum_observed_validation_e_delta_J",
        "p95_observed_validation_e_delta_J",
        "p99_observed_validation_e_delta_J",
        "guard_uncertainty_bound_status",
    }
    assert required.issubset(table.columns)
    assert set(table["guard_uncertainty_bound_status"]) == {PAIRWISE_BOUND_STATUS}


def test_formal_figures_are_nonempty_pngs():
    for filename in FIGURE_FILENAMES:
        path = DEFAULT_OUTPUT_DIRECTORY / filename
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000


def test_decision_guard_status_is_research_only_candidate():
    assert DECISION_GUARD_STATUS == "RESEARCH_ONLY_VIRTUAL_CANDIDATE"
    assert DECISION_UNCERTAINTY_VERSION == "RESEARCH_DECISION_UNCERTAINTY_V1"
