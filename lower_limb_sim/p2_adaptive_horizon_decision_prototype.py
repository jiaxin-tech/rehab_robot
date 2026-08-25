"""Default-off adaptive-horizon endpoint decision research prototype.

The prototype starts from one formal generator-grid step and only escalates
through 2, 3, and 5 steps when no shorter endpoint passes the already-frozen
two-gate decision semantics and the predicted objective keeps improving along
one coordinate in one signed direction.  Only the selected endpoint is
executed in the offline shadow.  Latent intermediate trajectories are never
executed, and every endpoint execution invalidates the authorization before a
model refit and full prediction-map recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    TrajectoryComponentCache,
    build_predicted_map,
)
from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_decision_rule_semantics_audit import SemanticsCalibration
from .p2_multi_step_decision_framework_analysis import (
    FRAMEWORKS,
    MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
    SMALL_STEP_SOURCE_PATH,
    FrameworkSpec,
    FrozenFrameworkManifestGate,
    _AXES,
    _AXIS_INDEX,
    _GRID_STEP,
    _key,
    _map_lookup,
    evaluate_endpoint_candidates,
    framework_uncertainty,
)
from .research_decision_guarded_sequential_personalization import (
    EXECUTED_FALSE_IMPROVEMENT,
    RESEARCH_ONLY,
    STOP_MAX_PERSONALIZATION_TRIALS,
    STOP_MODEL_UPDATE_FAILURE,
    STOP_NO_GEOMETRICALLY_VALID_CANDIDATE,
    STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER,
    STOP_PATIENT_ENVELOPE_BOUNDARY,
    TRIAL_PURPOSE_EXPLOIT,
    TRIAL_PURPOSE_EXPLORE,
    InitialResearchState,
    PolicyRunResult,
    SelectionGatedVirtualTruthOracle,
    _actual_objective,
    _fit_updated_model,
    _model_for_iteration,
    alpha_from_row,
    build_local_exploration_frontier,
    rank_exploration_frontier,
)
from .sequential_personalization import SearchAlpha, accept_actual_trial


PROTOTYPE_ID = "P2_ADAPTIVE_HORIZON_DECISION_PROTOTYPE_V1"
MANIFEST_ID = "P2_ADAPTIVE_HORIZON_DECISION_PROTOTYPE_MANIFEST_V1"
H1_ID = "H1_SINGLE_STEP_ONLY"
H2_ID = "H2_FIXED_BUNDLE_5"
H3_ID = "H3_ADAPTIVE_HORIZON"
ADAPTIVE_HORIZON_SEQUENCE = (1, 2, 3, 5)
DEFAULT_ENABLED = False
OFFLINE_ONLY = "OFFLINE_ONLY"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"

PRIOR_FRAMEWORK_MANIFEST_SHA256 = (
    "a640f7c897291bf044f2f67dd41c84af87cd867be8140030cc4304a4b57d5731"
)
MODULE_DIR = Path(__file__).resolve().parent
PRIOR_FRAMEWORK_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_multi_step_decision_framework_analysis_v1"
)
PRIOR_FRAMEWORK_MANIFEST_PATH = PRIOR_FRAMEWORK_DIRECTORY / "MANIFEST.json"

_SPEC_BY_HORIZON = {spec.horizon_steps: spec for spec in FRAMEWORKS}
if tuple(_SPEC_BY_HORIZON) != ADAPTIVE_HORIZON_SEQUENCE:
    raise RuntimeError("frozen multi-step horizon sequence changed")


@dataclass(frozen=True)
class AdaptiveEndpointEvaluation:
    candidates: pd.DataFrame
    selected: pd.Series | None
    evaluated_horizons: tuple[int, ...]
    escalation_stopped_reason: str


def manifest_payload(
    calibration: SemanticsCalibration,
    *,
    checkpoint_commit: str,
    protected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze adaptive semantics before development truth is accessed."""

    return {
        "manifest_id": MANIFEST_ID,
        "prototype_id": PROTOTYPE_ID,
        "status": "FROZEN_BEFORE_DEVELOPMENT_SHADOW_TRUTH",
        "checkpoint_commit": str(checkpoint_commit),
        "prior_framework_manifest_sha256": PRIOR_FRAMEWORK_MANIFEST_SHA256,
        "comparators": [
            {
                "prototype_variant_id": H1_ID,
                "decision_horizon": "FIXED_1",
                "source": "FROZEN_PRIOR_MULTI_STEP_ARTIFACT",
            },
            {
                "prototype_variant_id": H2_ID,
                "decision_horizon": "FIXED_5",
                "source": "FROZEN_PRIOR_MULTI_STEP_ARTIFACT",
            },
            {
                "prototype_variant_id": H3_ID,
                "decision_horizon": "ADAPTIVE_1_2_3_5",
                "source": "THIS_DEFAULT_OFF_SHADOW",
            },
        ],
        "adaptive_rule": {
            "horizon_evaluation_order": list(ADAPTIVE_HORIZON_SEQUENCE),
            "start_horizon_steps": 1,
            "select_first_horizon_with_any_eligible_endpoint": True,
            "within_horizon_selection": (
                "lowest_J_pred_then_trajectory_id_among_eligible_endpoints"
            ),
            "magnitude_gate": "predicted_endpoint_improvement > 0.005",
            "direction_gate": "deltaJ_pred + independent_scale_P95[horizon] < 0",
            "direction_consistency": (
                "all finite per-grid-step predicted objective increments from the "
                "current endpoint are strictly negative on one coordinate and one "
                "signed direction"
            ),
            "direction_consistency_numeric_threshold": 0.0,
            "escalate_only_if_no_shorter_endpoint_passes": True,
            "escalate_only_if_at_least_one_nondecision-valid_direction_is_consistent": True,
            "truth_used_for_horizon_selection": False,
            "new_distance_definition": False,
            "new_threshold_frozen": False,
        },
        "uncertainty_P95_by_horizon": {
            str(horizon): framework_uncertainty(_SPEC_BY_HORIZON[horizon], calibration)
            for horizon in ADAPTIVE_HORIZON_SEQUENCE
        },
        "endpoint_execution": {
            "authorization_scope": "SELECTED_DIRECT_ENDPOINT_ONLY",
            "intermediate_trajectories_executed": False,
            "latent_nodes_checked_with_existing_nondecision_gates": True,
            "model_refit_after_endpoint_execution": True,
            "whole_map_recomputed_after_endpoint_execution": True,
            "authorization_invalidated_after_execution": True,
        },
        "evaluation_questions_frozen_before_truth": {
            "approaches_bundle_5_performance_if": [
                "small_step_recovery_equals_H2",
                "mean_final_J_not_worse_than_H2_by_0.005",
                "mean_global_regret_not_worse_than_H2_by_0.005",
            ],
            "reduces_bundle_5_trial_cost_if": "total_trials < H2_total_trials",
            "reduces_boundary_collapse_if": [
                "boundary_saturated_case_count < H2",
                "or unique_final_alpha_count > H2",
            ],
            "results_may_modify_these_criteria": False,
        },
        "data_roles": {
            "development": "ORIGINAL_9_PLUS_POST_REJECTION_DEVELOPMENT_6_SHADOW_ONLY",
            "independent_calibration": "RESIDUAL_SCALE_ONLY",
            "prior_framework_artifacts": "FROZEN_H1_H2_COMPARATORS_ONLY",
            "heldout_final_test": "NOT_READ",
            "prospective": "NOT_RUN",
        },
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "protected_source_sha256": dict(protected_source_sha256),
        "P2_V1_modified": False,
        "objective_modified": False,
        "five_parameter_model_modified": False,
        "generator_modified": False,
        "ROM_modified": False,
        "support_gate_modified": False,
        "default_enabled": False,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
    }


def _path_prediction_consistency(
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    row: Mapping[str, Any],
    horizon_steps: int,
) -> tuple[bool, str, str]:
    """Check strict predicted improvement at every existing grid node."""

    axis = str(row["coordinate"])
    direction = str(row["direction"])
    sign = -1.0 if direction == "NEGATIVE" else 1.0
    axis_index = _AXIS_INDEX[axis]
    lookup = _map_lookup(prediction_map)
    start = current.key()
    path: list[tuple[float, float, float]] = []
    values: list[float] = []
    identifiers: list[str] = []
    for step_number in range(horizon_steps + 1):
        point = list(start)
        point[axis_index] += sign * _GRID_STEP[axis] * step_number
        key = _key(point)
        if key not in lookup:
            return False, "", "PATH_NODE_MISSING"
        path.append(key)
        values.append(float(lookup[key]["J_pred"]))
        identifiers.append(str(lookup[key]["trajectory_id"]))
    differences = np.diff(np.asarray(values, dtype=float))
    finite = bool(np.isfinite(values).all() and np.isfinite(differences).all())
    consistent = bool(finite and np.all(differences < 0.0))
    reason = "STRICTLY_IMPROVING" if consistent else (
        "NONFINITE_PREDICTION" if not finite else "DIRECTION_NOT_STRICTLY_IMPROVING"
    )
    return consistent, ";".join(identifiers), reason


def evaluate_adaptive_endpoint_candidates(
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    calibration: SemanticsCalibration,
    *,
    executed_keys: set[tuple[float, float, float]],
    patient_validity_cache: dict[tuple[float, float, float], bool],
) -> AdaptiveEndpointEvaluation:
    """Evaluate 1→2→3→5 and stop at the first eligible horizon."""

    evaluated: list[int] = []
    frames: list[pd.DataFrame] = []
    selected: pd.Series | None = None
    stop_reason = "MAXIMUM_HORIZON_EVALUATED_WITHOUT_ELIGIBLE_ENDPOINT"
    for horizon in ADAPTIVE_HORIZON_SEQUENCE:
        spec = _SPEC_BY_HORIZON[horizon]
        table = evaluate_endpoint_candidates(
            prediction_map,
            current,
            spec,
            calibration,
            executed_keys=executed_keys,
            patient_validity_cache=patient_validity_cache,
        ).copy()
        table["adaptive_horizon_steps"] = horizon
        table["adaptive_evaluation_order"] = len(evaluated) + 1
        table["direction_consistency_pass"] = False
        table["direction_consistency_path_ids"] = ""
        table["direction_consistency_reason"] = "NOT_A_DIRECT_ENDPOINT"
        table["adaptive_nondecision_path_valid"] = False
        table["adaptive_research_exploit_eligible"] = False
        direct_indices = table.index[
            table["candidate_type"].eq("DIRECT_ENDPOINT_CANDIDATE")
        ]
        for index in direct_indices:
            consistent, path_ids, reason = _path_prediction_consistency(
                prediction_map, current, table.loc[index], horizon
            )
            nondecision_valid = bool(
                table.at[index, "all_latent_nodes_geometry_valid"]
                and table.at[index, "all_latent_nodes_provenance_valid"]
                and table.at[index, "all_latent_nodes_model_supported"]
                and table.at[index, "all_latent_nodes_patient_envelope_valid"]
            )
            table.at[index, "direction_consistency_pass"] = consistent
            table.at[index, "direction_consistency_path_ids"] = path_ids
            table.at[index, "direction_consistency_reason"] = reason
            table.at[index, "adaptive_nondecision_path_valid"] = nondecision_valid
            table.at[index, "adaptive_research_exploit_eligible"] = bool(
                consistent
                and nondecision_valid
                and table.at[index, "research_exploit_eligible"]
            )
        evaluated.append(horizon)
        eligible = table.loc[
            table["adaptive_research_exploit_eligible"].astype(bool)
        ]
        if not eligible.empty:
            selected = eligible.sort_values(
                ["J_pred", "trajectory_id"], kind="mergesort"
            ).iloc[0].copy()
            stop_reason = "FIRST_ELIGIBLE_HORIZON_SELECTED"
            table["adaptive_horizon_selected"] = False
            table.loc[
                table["trajectory_id"].astype(str).eq(str(selected["trajectory_id"])),
                "adaptive_horizon_selected",
            ] = True
            frames.append(table)
            break
        table["adaptive_horizon_selected"] = False
        frames.append(table)
        can_escalate = bool(
            (
                table["direction_consistency_pass"].astype(bool)
                & table["adaptive_nondecision_path_valid"].astype(bool)
            ).any()
        )
        if not can_escalate:
            stop_reason = "NO_CONSISTENT_NONDECISION_VALID_DIRECTION"
            break
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    candidates["evaluated_horizon_sequence"] = ";".join(map(str, evaluated))
    candidates["adaptive_escalation_stop_reason"] = stop_reason
    return AdaptiveEndpointEvaluation(
        candidates=candidates,
        selected=selected,
        evaluated_horizons=tuple(evaluated),
        escalation_stopped_reason=stop_reason,
    )


def run_adaptive_shadow(
    state: InitialResearchState,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    manifest_gate: FrozenFrameworkManifestGate,
    calibration: SemanticsCalibration,
    *,
    patient_validity_cache: dict[tuple[float, float, float], bool] | None = None,
    trial_budget: int = MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
) -> PolicyRunResult:
    """Run the adaptive endpoint-only development shadow."""

    if DEFAULT_ENABLED:
        raise PermissionError("adaptive prototype must remain default-off")
    if trial_budget != MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON:
        raise ValueError("diagnostic horizon is fixed")
    manifest_gate.require_frozen()
    patient_cache = patient_validity_cache if patient_validity_cache is not None else {}
    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    model = _model_for_iteration(state, parameters, domain_data, 0)
    prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
    initial_prediction_map = prediction_map.copy(deep=True)
    oracle = SelectionGatedVirtualTruthOracle(state.subject_id, state.scenario_name)

    reference = generate_personalized_trajectory()
    reference_trajectory = reference.trajectory.copy(deep=True)
    reference_id = str(reference.metadata["trajectory_id"])
    reference_trajectory["trajectory_id"] = reference_id
    token = oracle.declare_selected(reference_id, "REFERENCE_NORMALIZATION")
    manifest_gate.record_truth_access("REFERENCE_NORMALIZATION")
    reference_metrics = oracle.execute(token, reference_trajectory).actual_metrics

    operating_alpha = SearchAlpha()
    operating_actual_j = 1.0
    best_alpha = SearchAlpha()
    best_actual_j = 1.0
    executed_keys = {operating_alpha.key()}
    history_rows: list[dict[str, Any]] = []
    guard_frames: list[pd.DataFrame] = []
    stop_reason = ""
    model_update_count = 0
    cumulative_regret = 0.0

    for iteration in range(1, trial_budget + 1):
        truth_before_proposal = oracle.truth_calls
        evaluation = evaluate_adaptive_endpoint_candidates(
            prediction_map,
            operating_alpha,
            calibration,
            executed_keys=executed_keys,
            patient_validity_cache=patient_cache,
        )
        guarded = evaluation.candidates.copy()
        guarded["iteration"] = iteration
        guarded["policy_id"] = H3_ID
        guarded["framework_id"] = H3_ID
        guarded["subject_id"] = state.subject_id
        guarded["scenario_name"] = state.scenario_name
        guarded["selected_for_execution"] = False
        guarded["selection_mode"] = ""
        selected = evaluation.selected
        purpose = TRIAL_PURPOSE_EXPLOIT
        selected_horizon = 0
        selection_mode = ""
        frontier_ranked = pd.DataFrame()
        if selected is not None:
            selected_horizon = int(selected["adaptive_horizon_steps"])
            selection_mode = f"ADAPTIVE_DIRECT_ENDPOINT_H{selected_horizon}"
        else:
            frontier = build_local_exploration_frontier(prediction_map, executed_keys)
            if not frontier.empty:
                frontier = frontier.loc[~frontier["model_supported"].astype(bool)].copy()
            frontier_ranked = rank_exploration_frontier(
                frontier, fitting_data, parameters
            )
            valid = (
                frontier_ranked.loc[
                    frontier_ranked["exploration_candidate_valid"].astype(bool)
                ]
                if not frontier_ranked.empty
                else frontier_ranked
            )
            if not valid.empty:
                selected = valid.iloc[0].copy()
                purpose = TRIAL_PURPOSE_EXPLORE
                selection_mode = TRIAL_PURPOSE_EXPLORE
        if selected is None:
            if guarded.empty:
                stop_reason = STOP_NO_GEOMETRICALLY_VALID_CANDIDATE
            else:
                frontier_any = build_local_exploration_frontier(
                    prediction_map, executed_keys
                )
                unsupported = (
                    frontier_any.loc[~frontier_any["model_supported"].astype(bool)]
                    if not frontier_any.empty
                    else frontier_any
                )
                stop_reason = (
                    STOP_PATIENT_ENVELOPE_BOUNDARY
                    if not unsupported.empty
                    and not frontier_ranked.empty
                    and not frontier_ranked["exploration_candidate_valid"]
                    .astype(bool)
                    .any()
                    else STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
                )
            guarded["policy_decision"] = "STOP"
            guarded["prospective_stop_reason"] = stop_reason
            guard_frames.append(guarded)
            break

        selected_id = str(selected["trajectory_id"])
        selected_alpha = alpha_from_row(selected)
        if selected_alpha.key() == operating_alpha.key():
            raise RuntimeError("current endpoint cannot be re-executed")
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            difference = np.asarray(selected_alpha.key()) - np.asarray(
                operating_alpha.key()
            )
            changed = np.flatnonzero(np.abs(difference) > 1e-12)
            if len(changed) != 1:
                raise RuntimeError("adaptive endpoint attempted a mixed-axis jump")
            axis = _AXES[int(changed[0])]
            expected = selected_horizon * _GRID_STEP[axis]
            if not np.isclose(
                abs(difference[changed[0]]), expected, atol=1e-12, rtol=0.0
            ):
                raise RuntimeError("adaptive endpoint used a non-frozen horizon")
            guarded.loc[
                guarded["trajectory_id"].astype(str).eq(selected_id)
                & guarded["adaptive_horizon_steps"].eq(selected_horizon),
                ["selected_for_execution", "selection_mode"],
            ] = [True, selection_mode]
        guarded["policy_decision"] = purpose
        guarded["prospective_stop_reason"] = ""
        guard_frames.append(guarded)
        if oracle.truth_calls != truth_before_proposal:
            raise RuntimeError("adaptive proposal or horizon selection accessed truth")

        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = selected_id
        selection_token = oracle.declare_selected(selected_id, purpose)
        manifest_gate.record_truth_access("SELECTED_ADAPTIVE_ENDPOINT_OR_EXPLORATION")
        execution = oracle.execute(selection_token, trajectory)
        if not execution.observation_valid:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
            break
        actual = _actual_objective(selected_id, execution, reference_metrics)
        current_row = _map_lookup(prediction_map)[operating_alpha.key()]
        predicted_j = float(selected["J_pred"])
        delta_pred = predicted_j - float(current_row["J_pred"])
        operating_before = operating_alpha
        operating_actual_before = operating_actual_j
        delta_actual = actual.mechanical_cost_j_rms - operating_actual_before
        best_before = best_actual_j
        accepted = accept_actual_trial(actual.mechanical_cost_j_rms, best_actual_j)
        if accepted:
            best_actual_j = actual.mechanical_cost_j_rms
            best_alpha = selected_alpha
            operating_alpha = selected_alpha
            operating_actual_j = actual.mechanical_cost_j_rms
        cumulative_regret += max(actual.mechanical_cost_j_rms - best_before, 0.0)
        executed_keys.add(selected_alpha.key())
        false_improvement = bool(
            purpose == TRIAL_PURPOSE_EXPLOIT
            and delta_pred < -OBJECTIVE_EQUIVALENCE_TOLERANCE
            and delta_actual >= -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )

        fitting_data = pd.concat(
            (fitting_data, execution.estimator_observations), ignore_index=True
        )
        domain_data = pd.concat(
            (domain_data, execution.estimator_observations), ignore_index=True
        )
        estimation = _fit_updated_model(fitting_data, parameters)
        if estimation.optimizer_success:
            parameters = dict(estimation.estimated_parameters)
            model_update_count += 1
        else:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
        model = _model_for_iteration(state, parameters, domain_data, iteration)
        prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
        history_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": H3_ID,
                "framework_id": H3_ID,
                "horizon_steps": selected_horizon,
                "iteration": iteration,
                "trial_purpose": purpose,
                "selection_mode": selection_mode,
                "trajectory_id": selected_id,
                "alpha_hip": selected_alpha.hip_delta_deg,
                "alpha_knee": selected_alpha.knee_delta_deg,
                "alpha_phase": selected_alpha.phase_delta,
                "operating_alpha_hip_before": operating_before.hip_delta_deg,
                "operating_alpha_knee_before": operating_before.knee_delta_deg,
                "operating_alpha_phase_before": operating_before.phase_delta,
                "operating_alpha_hip_after": operating_alpha.hip_delta_deg,
                "operating_alpha_knee_after": operating_alpha.knee_delta_deg,
                "operating_alpha_phase_after": operating_alpha.phase_delta,
                "J_pred": predicted_j,
                "actual_J": actual.mechanical_cost_j_rms,
                "operating_actual_J_before": operating_actual_before,
                "best_actual_J_before": best_before,
                "best_actual_J_after": best_actual_j,
                "delta_J_pred_endpoint": delta_pred,
                "delta_J_actual_vs_operating": delta_actual,
                "accepted_meaningful_improvement": accepted,
                "executed_false_improvement": false_improvement,
                "evaluated_horizon_sequence": ";".join(
                    map(str, evaluation.evaluated_horizons)
                ),
                "adaptive_escalation_stop_reason": (
                    evaluation.escalation_stopped_reason
                ),
                "latent_intermediate_count": (
                    max(selected_horizon - 1, 0)
                    if purpose == TRIAL_PURPOSE_EXPLOIT
                    else 0
                ),
                "intermediate_execution_count": 0,
                "endpoint_execution_count": (
                    1 if purpose == TRIAL_PURPOSE_EXPLOIT else 0
                ),
                "model_refit_after_execution": bool(estimation.optimizer_success),
                "full_map_recomputed_after_execution": True,
                "authorization_invalidated_after_execution": (
                    purpose == TRIAL_PURPOSE_EXPLOIT
                ),
                "truth_accessed_before_selection": False,
                "manifest_sha_verified_before_truth": True,
                "execution_status": (
                    EXECUTED_FALSE_IMPROVEMENT
                    if false_improvement
                    else "EXECUTED"
                ),
                "cumulative_regret_vs_best_before": cumulative_regret,
                "stop_reason_after_iteration": stop_reason,
            }
        )
        if stop_reason:
            break

    if not stop_reason:
        stop_reason = STOP_MAX_PERSONALIZATION_TRIALS
        if history_rows:
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
    history = pd.DataFrame(history_rows)
    guard_audit = pd.concat(guard_frames, ignore_index=True, sort=False)
    # Every evaluated horizon contains the same current operating point.  The
    # post-policy truth auditor requires one row per trajectory and iteration;
    # only those repeated origin rows are collapsed.  Distinct endpoints from
    # every evaluated horizon remain in the audit.
    guard_audit = guard_audit.drop_duplicates(
        subset=["iteration", "trajectory_id"], keep="first"
    ).reset_index(drop=True)
    number_executed = len(history)
    exploit_history = (
        history.loc[history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT)]
        if number_executed
        else history
    )
    horizon_usage = {
        str(horizon): int(exploit_history["horizon_steps"].eq(horizon).sum())
        if not exploit_history.empty
        else 0
        for horizon in ADAPTIVE_HORIZON_SEQUENCE
    }
    summary = {
        "case_id": f"{state.subject_id}__{state.scenario_name}",
        "subject_id": state.subject_id,
        "scenario_name": state.scenario_name,
        "policy_id": H3_ID,
        "framework_id": H3_ID,
        "horizon_steps": -1,
        "calibrated_uncertainty": np.nan,
        "research_status": RESEARCH_ONLY,
        "number_of_executed_trials": number_executed,
        "number_of_exploit_trials": len(exploit_history),
        "number_of_explore_trials": int(
            history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLORE).sum()
        )
        if number_executed
        else 0,
        "number_of_executed_false_improvements": int(
            history["executed_false_improvement"].sum()
        )
        if number_executed
        else 0,
        "first_exploit_iteration": (
            int(exploit_history["iteration"].min())
            if not exploit_history.empty
            else np.nan
        ),
        "decision_latency_trials": (
            int(exploit_history["iteration"].min()) - 1
            if not exploit_history.empty
            else number_executed
        ),
        "latent_intermediate_nodes_skipped": int(
            history["latent_intermediate_count"].sum()
        )
        if number_executed
        else 0,
        "intermediate_trajectory_executions": int(
            history["intermediate_execution_count"].sum()
        )
        if number_executed
        else 0,
        "endpoint_executions": int(history["endpoint_execution_count"].sum())
        if number_executed
        else 0,
        "reference_actual_J": 1.0,
        "final_best_actual_J": best_actual_j,
        "actual_J_reduction_from_reference": 1.0 - best_actual_j,
        "cumulative_regret_vs_best_before": cumulative_regret,
        "model_update_count": model_update_count,
        "final_best_alpha_hip": best_alpha.hip_delta_deg,
        "final_best_alpha_knee": best_alpha.knee_delta_deg,
        "final_best_alpha_phase": best_alpha.phase_delta,
        "final_operating_alpha_hip": operating_alpha.hip_delta_deg,
        "final_operating_alpha_knee": operating_alpha.knee_delta_deg,
        "final_operating_alpha_phase": operating_alpha.phase_delta,
        "adaptive_horizon_usage": horizon_usage,
        "stop_reason": stop_reason,
        "trial_budget": trial_budget,
        "heldout_final_test_used": False,
        "calibration_cases_used_for_policy_outcomes": False,
        "prospective_cohort_run": False,
        "new_policy_default_enabled": False,
        "human_ready": False,
        "robot_motion_approved": False,
        "model_reliability_status": "RESEARCH_DIAGNOSTIC_NOT_FORMALLY_RELIABLE",
    }
    return PolicyRunResult(
        subject_id=state.subject_id,
        scenario_name=state.scenario_name,
        policy_id=H3_ID,
        trial_history=history,
        decision_guard_audit=guard_audit,
        parameter_history=pd.DataFrame(),
        prediction_map_history=pd.DataFrame(),
        known_region_history=pd.DataFrame(),
        uncertainty_history=pd.DataFrame(),
        uncertainty_pairwise_audit=pd.DataFrame(),
        exploration_information_gain=pd.DataFrame(),
        false_improvement_audit=pd.DataFrame(),
        summary=summary,
        initial_prediction_map=initial_prediction_map,
        final_prediction_map=prediction_map,
        truth_access_audit={
            "manifest_verified_before_every_truth": True,
            "proposal_truth_accessed": False,
            "heldout_final_test_used": False,
        },
    )


def adaptive_small_step_recovery(
    calibration: SemanticsCalibration,
) -> pd.DataFrame:
    """Apply the frozen adaptive rule before attaching posthoc truth."""

    source = pd.read_csv(SMALL_STEP_SOURCE_PATH)
    rows: list[dict[str, Any]] = []
    for path_id, group in source.groupby("path_id", sort=True):
        ordered = group.sort_values("step_number")
        first = ordered.iloc[0]
        selected_horizon: int | None = None
        predicted_delta = np.nan
        direction_consistent = False
        magnitude_gate = False
        direction_gate = False
        evaluated: list[int] = []
        for horizon in ADAPTIVE_HORIZON_SEQUENCE:
            prefix = ordered.loc[ordered["step_number"].le(horizon)]
            endpoint = ordered.loc[ordered["step_number"].eq(horizon)]
            if len(prefix) != horizon or endpoint.empty:
                break
            increments = prefix["single_step_deltaJ_pred"].to_numpy(dtype=float)
            direction_consistent = bool(
                np.isfinite(increments).all() and np.all(increments < 0.0)
            )
            evaluated.append(horizon)
            if not direction_consistent:
                break
            predicted_delta = float(
                endpoint.iloc[0]["cumulative_endpoint_deltaJ_pred"]
            )
            magnitude_gate = bool(
                -predicted_delta > OBJECTIVE_EQUIVALENCE_TOLERANCE
            )
            direction_gate = bool(
                predicted_delta
                + framework_uncertainty(_SPEC_BY_HORIZON[horizon], calibration)
                < 0.0
            )
            if magnitude_gate and direction_gate:
                selected_horizon = horizon
                break
        # Truth is attached only after the predicted-only horizon is fixed.
        truth_delta = np.nan
        recovered = False
        if selected_horizon is not None:
            selected = ordered.loc[
                ordered["step_number"].eq(selected_horizon)
            ].iloc[0]
            truth_delta = float(selected["cumulative_endpoint_deltaJ_truth"])
            recovered = bool(
                truth_delta < -OBJECTIVE_EQUIVALENCE_TOLERANCE
            )
        rows.append(
            {
                "path_id": path_id,
                "case_id": first["case_id"],
                "coordinate": first["coordinate"],
                "direction": first["direction"],
                "prototype_variant_id": H3_ID,
                "evaluated_horizon_sequence": ";".join(map(str, evaluated)),
                "selected_horizon_steps": selected_horizon,
                "predicted_endpoint_delta_J": predicted_delta,
                "truth_endpoint_delta_J_posthoc": truth_delta,
                "direction_consistency_pass": direction_consistent,
                "magnitude_gate_pass": magnitude_gate,
                "direction_gate_pass": direction_gate,
                "recovered_small_step_path": recovered,
                "intermediate_trajectories_executed": False,
                "truth_used_for_authorization": False,
                "truth_attached_posthoc_only": True,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ADAPTIVE_HORIZON_SEQUENCE",
    "DEFAULT_ENABLED",
    "H1_ID",
    "H2_ID",
    "H3_ID",
    "MANIFEST_ID",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_APPROVED",
    "OFFLINE_ONLY",
    "PRIOR_FRAMEWORK_MANIFEST_PATH",
    "PRIOR_FRAMEWORK_MANIFEST_SHA256",
    "PROTOTYPE_ID",
    "AdaptiveEndpointEvaluation",
    "adaptive_small_step_recovery",
    "evaluate_adaptive_endpoint_candidates",
    "manifest_payload",
    "run_adaptive_shadow",
]
