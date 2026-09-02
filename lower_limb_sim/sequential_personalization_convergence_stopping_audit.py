"""Post-policy convergence and stopping diagnostics for sequential personalization.

This module characterizes the already-frozen research policies.  It does not
change proposal, guard, support, ranking, fitting, or stopping decisions.
Virtual truth is attached only after each policy decision has been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .admissible_personalization_region import (
    AdmissibleRegionArtifacts,
    load_admissible_personalization_region,
)
from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    generate_personalized_trajectory,
)
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    TrajectoryComponentCache,
    build_predicted_map,
    evaluate_truth_map,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    CURRENT_BEST_NOT_A_CANDIDATE,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    STOP_MAX_PERSONALIZATION_TRIALS,
    InitialResearchState,
    PolicyRunResult,
    _model_for_iteration,
    local_prediction_candidates,
)
from .sequential_personalization import SearchAlpha, TrustRegionSteps


AUDIT_PROTOCOL_ID = "SEQUENTIAL_PERSONALIZATION_CONVERGENCE_AND_STOPPING_AUDIT_V1"
EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON = 20
TRIAL_BUDGETS = (3, 6, 12, EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON)
HORIZON_STATUS = "OFFLINE_VIRTUAL_DIAGNOSTIC_CAP_NOT_HUMAN_THRESHOLD"
POST_DECISION_TRUTH_ROLE = "POST_DECISION_EVALUATION_ONLY_NO_POLICY_FEEDBACK"

MISSED_IMPROVEMENT = "MISSED_IMPROVEMENT"
CORRECT_LOCAL_STOP = "CORRECT_LOCAL_STOP"
PREMATURE_CONSERVATIVE_STOP = "PREMATURE_CONSERVATIVE_STOP"
CORRECT_CONSERVATIVE_STOP = "CORRECT_CONSERVATIVE_STOP"
REFERENCE_LOCALLY_COMPETITIVE = "REFERENCE_LOCALLY_COMPETITIVE"
EXPLORATION_DECISION_VALUE_LOW = "EXPLORATION_DECISION_VALUE_LOW"
INFORMATIVE_BUT_LOW_DECISION_VALUE = "INFORMATIVE_BUT_LOW_DECISION_VALUE"
DECISION_VALUE_OBSERVED = "DECISION_VALUE_OBSERVED"
BOUNDARY_OPTIMUM_DIAGNOSTIC = "BOUNDARY_OPTIMUM_DIAGNOSTIC"

OFFLINE_METHOD_ARCHITECTURE_READY_TO_FREEZE = (
    "OFFLINE_METHOD_ARCHITECTURE_READY_TO_FREEZE"
)
OFFLINE_METHOD_REQUIRES_REVISION = "OFFLINE_METHOD_REQUIRES_REVISION"

_STEP_SCALE = np.asarray(
    (GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP), dtype=float
)


@dataclass(frozen=True)
class PostDecisionTruthAudit:
    candidate_rows: pd.DataFrame
    round_rows: pd.DataFrame
    missed_cases: pd.DataFrame
    correct_stop: pd.DataFrame


def normalized_alpha_distance(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("alpha distance requires two three-component vectors")
    return float(np.linalg.norm((left - right) / _STEP_SCALE))


def parameter_bound_distances(alpha: SearchAlpha) -> dict[str, Any]:
    values = {
        "hip_amplitude_delta_deg": float(alpha.hip_delta_deg),
        "knee_amplitude_delta_deg": float(alpha.knee_delta_deg),
        "knee_phase_shift": float(alpha.phase_delta),
    }
    steps = {
        "hip_amplitude_delta_deg": GRID_HIP_STEP_DEG,
        "knee_amplitude_delta_deg": GRID_KNEE_STEP_DEG,
        "knee_phase_shift": GRID_PHASE_STEP,
    }
    payload: dict[str, Any] = {}
    normalized: list[float] = []
    on_boundary = False
    for name, value in values.items():
        lower, upper = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name]
        lower_distance = value - float(lower)
        upper_distance = float(upper) - value
        nearest = min(lower_distance, upper_distance)
        normalized.append(nearest / steps[name])
        short = {
            "hip_amplitude_delta_deg": "hip",
            "knee_amplitude_delta_deg": "knee",
            "knee_phase_shift": "phase",
        }[name]
        payload[f"distance_to_{short}_lower_bound"] = lower_distance
        payload[f"distance_to_{short}_upper_bound"] = upper_distance
        if abs(nearest) <= 1e-12:
            on_boundary = True
    payload["distance_to_parameter_bounds_formal_steps"] = float(min(normalized))
    payload["on_generator_parameter_boundary"] = on_boundary
    return payload


def corridor_boundary_distances(
    alpha: SearchAlpha,
    region: AdmissibleRegionArtifacts | None = None,
) -> dict[str, Any]:
    artifacts = region or load_admissible_personalization_region()
    generated = generate_personalized_trajectory(**alpha.as_generator_parameters())
    trajectory = generated.trajectory
    joint = artifacts.joint_corridor
    pull = artifacts.pull_corridor
    if len(trajectory) != len(joint) or len(trajectory) != len(pull):
        raise ValueError("trajectory and frozen corridor lengths differ")

    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
    joint_parts: list[np.ndarray] = []
    for values, lower_name, upper_name in (
        (q_hip, "q_hip_min_rad", "q_hip_max_rad"),
        (q_knee, "q_knee_min_rad", "q_knee_max_rad"),
    ):
        lower = joint[lower_name].to_numpy(dtype=float)
        upper = joint[upper_name].to_numpy(dtype=float)
        # The periodic seam and stationary endpoints are identical for every
        # admissible trajectory.  Their zero-width envelopes are invariants,
        # not evidence that a candidate is chasing a corridor boundary.
        nondegenerate = (upper - lower) > 1e-10
        joint_parts.extend(
            (values[nondegenerate] - lower[nondegenerate],
             upper[nondegenerate] - values[nondegenerate])
        )
    joint_margins = np.concatenate(joint_parts)
    x_pull = trajectory["x_pull_m"].to_numpy(dtype=float)
    z_pull = trajectory["z_pull_m"].to_numpy(dtype=float)
    x_ref = pull["x_pull_ref_m"].to_numpy(dtype=float)
    z_ref = pull["z_pull_ref_m"].to_numpy(dtype=float)
    radial = np.hypot(x_pull - x_ref, z_pull - z_ref)
    pull_parts: list[np.ndarray] = []
    for values, lower_name, upper_name in (
        (x_pull, "x_pull_min_m", "x_pull_max_m"),
        (z_pull, "z_pull_min_m", "z_pull_max_m"),
    ):
        lower = pull[lower_name].to_numpy(dtype=float)
        upper = pull[upper_name].to_numpy(dtype=float)
        nondegenerate = (upper - lower) > 1e-10
        pull_parts.extend(
            (values[nondegenerate] - lower[nondegenerate],
             upper[nondegenerate] - values[nondegenerate])
        )
    radial_max = pull["pull_radial_max_mm"].to_numpy(dtype=float) / 1000.0
    radial_nondegenerate = radial_max > 1e-10
    pull_parts.append(radial_max[radial_nondegenerate] - radial[radial_nondegenerate])
    pull_margins = np.concatenate(pull_parts)
    joint_margin_deg = float(np.rad2deg(np.min(joint_margins)))
    pull_margin_mm = float(1000.0 * np.min(pull_margins))
    return {
        "distance_to_joint_corridor_boundary_deg": joint_margin_deg,
        "distance_to_pull_corridor_boundary_mm": pull_margin_mm,
        "on_joint_corridor_boundary": bool(joint_margin_deg <= 1e-8),
        "on_pull_corridor_boundary": bool(pull_margin_mm <= 1e-6),
        "outside_joint_corridor": bool(joint_margin_deg < -1e-8),
        "outside_pull_corridor": bool(pull_margin_mm < -1e-6),
    }


def _local_guard_rows(result: PolicyRunResult, iteration: int) -> pd.DataFrame:
    audit = result.decision_guard_audit
    rows = audit.loc[
        audit["iteration"].astype(int).eq(int(iteration))
        & audit["decision_guard_status"].notna()
    ].copy()
    if rows.empty:
        raise RuntimeError(f"missing frozen local guard rows at iteration {iteration}")
    if rows["trajectory_id"].duplicated().any():
        raise RuntimeError("local guard rows contain duplicate trajectory IDs")
    return rows


def _decision_for_iteration(result: PolicyRunResult, iteration: int) -> str:
    history = result.trial_history
    if history.empty:
        return "STOP"
    selected = history.loc[history["iteration"].astype(int).eq(int(iteration))]
    if len(selected) > 1:
        raise RuntimeError("more than one executed trajectory in an iteration")
    return str(selected.iloc[0]["trial_purpose"]) if len(selected) == 1 else "STOP"


def natural_stop_iteration(result: PolicyRunResult) -> int | None:
    if result.summary["stop_reason"] == STOP_MAX_PERSONALIZATION_TRIALS:
        return None
    audit = result.decision_guard_audit
    local = audit.loc[audit["decision_guard_status"].notna()]
    if not local.empty:
        return int(local["iteration"].max())
    if not result.trial_history.empty:
        return int(result.trial_history["iteration"].max())
    return 1


def audit_post_decision_local_truth(
    result: PolicyRunResult,
    state: InitialResearchState,
    cache: TrajectoryComponentCache,
    *,
    case_class: str,
) -> PostDecisionTruthAudit:
    """Evaluate local truth strictly after the policy decisions are immutable."""

    audit = result.decision_guard_audit
    iterations = sorted(
        audit.loc[audit["decision_guard_status"].notna(), "iteration"]
        .astype(int)
        .unique()
    )
    neutral = result.initial_prediction_map.loc[
        np.isclose(result.initial_prediction_map["hip_delta"], 0.0)
        & np.isclose(result.initial_prediction_map["knee_delta"], 0.0)
        & np.isclose(result.initial_prediction_map["phase_delta"], 0.0)
    ].iloc[0]
    model = _model_for_iteration(state, state.parameters, state.domain_data, 0)
    candidate_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    for iteration in iterations:
        local = _local_guard_rows(result, iteration)
        decision = _decision_for_iteration(result, iteration)
        decision_frozen = True
        evaluation_input = local.copy()
        if not evaluation_input["trajectory_id"].astype(str).eq(
            str(neutral["trajectory_id"])
        ).any():
            evaluation_input = pd.concat(
                (evaluation_input, neutral.to_frame().T), ignore_index=True
            )
        truth, _ = evaluate_truth_map(evaluation_input, model, cache, batch_size=64)
        current_rows = local.loc[
            local["decision_guard_status"].eq(CURRENT_BEST_NOT_A_CANDIDATE)
        ]
        if len(current_rows) != 1:
            raise RuntimeError("local truth audit lacks one frozen current-best row")
        current_id = str(current_rows.iloc[0]["trajectory_id"])
        current_truth = float(
            truth.loc[truth["trajectory_id"].astype(str).eq(current_id), "J_truth"].iloc[0]
        )
        joined = local.merge(
            truth[["trajectory_id", "J_truth"]], on="trajectory_id", validate="one_to_one"
        )
        joined["delta_J_truth_vs_current"] = joined["J_truth"] - current_truth
        joined["true_local_improvement"] = (
            joined["delta_J_truth_vs_current"]
            < -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        joined["is_current_best"] = joined["trajectory_id"].astype(str).eq(current_id)
        exploit_passed = bool(joined["research_exploit_eligible"].astype(bool).any())
        candidates = joined.loc[~joined["is_current_best"]].copy()
        true_candidates = candidates.loc[candidates["true_local_improvement"]]
        missed_round = bool(not exploit_passed and not true_candidates.empty)
        best_truth_delta = (
            float(candidates["delta_J_truth_vs_current"].min())
            if not candidates.empty
            else float("nan")
        )
        for row in candidates.to_dict(orient="records"):
            candidate_rows.append(
                {
                    "case_id": result.summary["case_id"],
                    "subject_id": result.subject_id,
                    "scenario_name": result.scenario_name,
                    "case_class": case_class,
                    "iteration": iteration,
                    "policy_decision": decision,
                    "trajectory_id": row["trajectory_id"],
                    "alpha_hip": row["hip_delta"],
                    "alpha_knee": row["knee_delta"],
                    "alpha_phase": row["phase_delta"],
                    "delta_J_pred": row["delta_J_pred_vs_current"],
                    "delta_J_truth": row["delta_J_truth_vs_current"],
                    "decision_guard_margin": row["improvement_margin"],
                    "model_supported": row["model_supported"],
                    "domain_coverage": row["domain_coverage"],
                    "why_guard_rejected": row["decision_guard_status"],
                    "research_exploit_eligible": row[
                        "research_exploit_eligible"
                    ],
                    "true_local_improvement": row["true_local_improvement"],
                    "missed_improvement": bool(
                        missed_round and row["true_local_improvement"]
                    ),
                    "diagnostic_status": (
                        MISSED_IMPROVEMENT
                        if missed_round and row["true_local_improvement"]
                        else "NO_MISSED_IMPROVEMENT_FOR_CANDIDATE"
                    ),
                    "truth_role": POST_DECISION_TRUTH_ROLE,
                    "policy_decision_frozen_before_truth": decision_frozen,
                    "truth_fed_back_to_policy": False,
                }
            )
        round_rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "case_class": case_class,
                "iteration": iteration,
                "policy_decision": decision,
                "exploit_passed_guard": exploit_passed,
                "true_local_improvement_available": bool(not true_candidates.empty),
                "best_local_delta_J_truth": best_truth_delta,
                "missed_opportunity_round": missed_round,
                "truth_role": POST_DECISION_TRUTH_ROLE,
                "policy_decision_frozen_before_truth": decision_frozen,
                "truth_fed_back_to_policy": False,
            }
        )
        if missed_round:
            missed_rows.extend(
                row
                for row in candidate_rows
                if row["case_id"] == result.summary["case_id"]
                and row["iteration"] == iteration
                and row["missed_improvement"]
            )
    rounds = pd.DataFrame(round_rows)
    natural_iteration = natural_stop_iteration(result)
    correct_rows: list[dict[str, Any]] = []
    if natural_iteration is not None:
        stop_round = rounds.loc[rounds["iteration"].eq(natural_iteration)]
        true_available = bool(
            not stop_round.empty
            and stop_round.iloc[0]["true_local_improvement_available"]
        )
        classification = (
            PREMATURE_CONSERVATIVE_STOP if true_available else CORRECT_LOCAL_STOP
        )
        conservative = (
            PREMATURE_CONSERVATIVE_STOP
            if true_available
            else CORRECT_CONSERVATIVE_STOP
        )
        correct_rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "case_class": case_class,
                "natural_stop_iteration": natural_iteration,
                "stop_reason": result.summary["stop_reason"],
                "true_local_improvement_available_at_stop": true_available,
                "correct_stop_classification": classification,
                "conservative_stop_classification": conservative,
                "truth_role": POST_DECISION_TRUTH_ROLE,
                "truth_fed_back_to_policy": False,
            }
        )
    return PostDecisionTruthAudit(
        candidate_rows=pd.DataFrame(candidate_rows),
        round_rows=rounds,
        missed_cases=pd.DataFrame(missed_rows),
        correct_stop=pd.DataFrame(correct_rows),
    )


def missed_improvement_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL_SCENARIOS", rounds)]
    groups.extend(
        (name, group)
        for name, group in rounds.groupby("case_class", sort=False)
    )
    for name, group in groups:
        available = int(group["true_local_improvement_available"].sum())
        missed = int(group["missed_opportunity_round"].sum())
        rows.append(
            {
                "scenario_group": name,
                "evaluated_round_count": int(len(group)),
                "rounds_with_true_local_improvement_available": available,
                "missed_opportunity_rounds": missed,
                "missed_improvement_rate": (
                    float(missed / available) if available else 0.0
                ),
                "metric_role": "OFFLINE_CONSERVATIVENESS_DIAGNOSTIC_ONLY",
                "threshold_tuned_from_metric": False,
            }
        )
    return pd.DataFrame(rows)


def build_natural_stopping_summary(
    results: Sequence[PolicyRunResult],
    case_classes: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for result in results:
        natural_iteration = natural_stop_iteration(result)
        cap = result.summary["stop_reason"] == STOP_MAX_PERSONALIZATION_TRIALS
        rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario": result.scenario_name,
                "case_class": case_classes[result.summary["case_id"]],
                "policy_id": result.policy_id,
                "executed_trial_count": result.summary[
                    "number_of_executed_trials"
                ],
                "natural_stop_reached": natural_iteration is not None,
                "natural_stop_iteration": natural_iteration,
                "stop_reason": result.summary["stop_reason"],
                "diagnostic_cap_reached": cap,
                "diagnostic_cap": EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
                "diagnostic_horizon_status": HORIZON_STATUS,
                "human_trial_recommendation": False,
            }
        )
    return pd.DataFrame(rows)


def build_boundary_chasing_audit(
    results: Sequence[PolicyRunResult],
    region: AdmissibleRegionArtifacts | None = None,
) -> pd.DataFrame:
    artifacts = region or load_admissible_personalization_region()
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.trial_history.empty:
            continue
        history = result.trial_history.sort_values("iteration")
        case_rows: list[dict[str, Any]] = []
        for row in history.to_dict(orient="records"):
            alpha = SearchAlpha(row["alpha_hip"], row["alpha_knee"], row["alpha_phase"])
            case_rows.append(
                {
                    **row,
                    **parameter_bound_distances(alpha),
                    **corridor_boundary_distances(alpha, artifacts),
                    "J_actual": row["actual_J"],
                }
            )
        if not case_rows:
            continue
        exploit = [row for row in case_rows if row["trial_purpose"] == "EXPLOIT"]
        monotonic = False
        if len(exploit) >= 2:
            distances = np.asarray(
                [row["distance_to_parameter_bounds_formal_steps"] for row in exploit]
            )
            deltas = np.diff(
                np.asarray(
                    [[row["alpha_hip"], row["alpha_knee"], row["alpha_phase"]] for row in exploit]
                ),
                axis=0,
            )
            direction_consistent = all(
                np.all(axis_delta >= -1e-12) or np.all(axis_delta <= 1e-12)
                for axis_delta in deltas.T
            )
            monotonic = bool(
                direction_consistent and np.all(np.diff(distances) <= 1e-12)
            )
        final_alpha = SearchAlpha(
            result.summary["final_best_alpha_hip"],
            result.summary["final_best_alpha_knee"],
            result.summary["final_best_alpha_phase"],
        )
        final_distances = {
            **parameter_bound_distances(final_alpha),
            **corridor_boundary_distances(final_alpha, artifacts),
        }
        final_is_neutral = final_alpha.neutral
        boundary_optimum = bool(
            final_distances["on_generator_parameter_boundary"]
            or (
                not final_is_neutral
                and (
                    final_distances["on_joint_corridor_boundary"]
                    or final_distances["on_pull_corridor_boundary"]
                )
            )
        )
        for row in case_rows:
            row["exploit_monotonic_march_toward_boundary"] = monotonic
            row["final_best_on_generator_boundary"] = final_distances[
                "on_generator_parameter_boundary"
            ]
            row["final_best_on_joint_corridor_boundary"] = final_distances[
                "on_joint_corridor_boundary"
            ]
            row["final_best_on_pull_corridor_boundary"] = final_distances[
                "on_pull_corridor_boundary"
            ]
            row["final_optimum_diagnostic_status"] = (
                BOUNDARY_OPTIMUM_DIAGNOSTIC
                if boundary_optimum
                else "INTERIOR_BEST_DIAGNOSTIC"
            )
            row["boundary_is_error_claim"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def build_marginal_improvement(history: pd.DataFrame) -> pd.DataFrame:
    output = history.copy(deep=True)
    if output.empty:
        return output
    output["delta_best_J_actual"] = (
        output["best_actual_J_after"] - output["best_actual_J_before"]
    )
    output["marginal_best_J_improvement"] = -output["delta_best_J_actual"]
    output["cumulative_J_improvement"] = 1.0 - output["best_actual_J_after"]
    output["accepted_exploit"] = (
        output["accepted_improvement"].astype(bool)
        & output["trial_purpose"].eq("EXPLOIT")
    )
    output["new_minimum_useful_improvement_threshold_created"] = False
    return output


def build_prediction_landscape_evolution(
    result: PolicyRunResult,
    state: InitialResearchState,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
) -> pd.DataFrame:
    parameters = result.parameter_history.sort_values("iteration")
    history = result.trial_history.set_index("iteration") if not result.trial_history.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    previous_map: pd.DataFrame | None = None
    previous_parameters: tuple[float, ...] | None = None
    previous_global: tuple[float, float, float] | None = None
    previous_local: tuple[float, float, float] | None = None
    cached_map: pd.DataFrame | None = None
    for item in parameters.to_dict(orient="records"):
        iteration = int(item["iteration"])
        estimates = tuple(float(item[f"{name}_after"]) for name in PARAMETER_NAMES)
        if previous_parameters == estimates and cached_map is not None:
            prediction_map = cached_map.copy(deep=True)
        else:
            model = _model_for_iteration(
                state,
                {name: value for name, value in zip(PARAMETER_NAMES, estimates)},
                state.domain_data,
                iteration,
            )
            prediction_map, _ = build_predicted_map(
                model, parameter_lattice, cache, batch_size=256
            )
            cached_map = prediction_map.copy(deep=True)
        if iteration == 0:
            best_alpha = SearchAlpha()
        else:
            selected = history.loc[iteration]
            best_alpha = SearchAlpha(
                selected["best_alpha_hip_after"],
                selected["best_alpha_knee_after"],
                selected["best_alpha_phase_after"],
            )
        global_row = prediction_map.sort_values(
            ["J_pred", "trajectory_id"], kind="mergesort"
        ).iloc[0]
        local = local_prediction_candidates(
            prediction_map, best_alpha, TrustRegionSteps()
        )
        local_row = local.sort_values(
            ["J_pred", "trajectory_id"], kind="mergesort"
        ).iloc[0]
        global_alpha = (
            float(global_row["hip_delta"]),
            float(global_row["knee_delta"]),
            float(global_row["phase_delta"]),
        )
        local_alpha = (
            float(local_row["hip_delta"]),
            float(local_row["knee_delta"]),
            float(local_row["phase_delta"]),
        )
        if previous_map is None:
            rms_change = 0.0
            max_change = 0.0
            global_shift = 0.0
            local_shift = 0.0
        else:
            previous_values = previous_map.sort_values("trajectory_id")["J_pred"].to_numpy(dtype=float)
            current_values = prediction_map.sort_values("trajectory_id")["J_pred"].to_numpy(dtype=float)
            delta = current_values - previous_values
            rms_change = float(np.sqrt(np.mean(delta**2)))
            max_change = float(np.max(np.abs(delta)))
            global_shift = normalized_alpha_distance(previous_global, global_alpha)
            local_shift = normalized_alpha_distance(previous_local, local_alpha)
        rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "iteration": iteration,
                "full_map_point_count": int(len(prediction_map)),
                "max_abs_map_change": max_change,
                "RMS_map_change": rms_change,
                "predicted_global_minimum_trajectory_id": global_row[
                    "trajectory_id"
                ],
                "predicted_global_minimum_J": global_row["J_pred"],
                "predicted_global_minimum_hip": global_alpha[0],
                "predicted_global_minimum_knee": global_alpha[1],
                "predicted_global_minimum_phase": global_alpha[2],
                "predicted_global_minimum_shift_formal_steps": global_shift,
                "predicted_local_minimum_trajectory_id": local_row[
                    "trajectory_id"
                ],
                "predicted_local_minimum_J": local_row["J_pred"],
                "predicted_local_minimum_shift_formal_steps": local_shift,
                "truth_used_for_map_evolution": False,
            }
        )
        previous_map = prediction_map
        previous_parameters = estimates
        previous_global = global_alpha
        previous_local = local_alpha
    return pd.DataFrame(rows)


def build_exploration_decision_value(
    result: PolicyRunResult,
    landscape: pd.DataFrame,
) -> pd.DataFrame:
    exploration = result.exploration_information_gain.copy(deep=True)
    if exploration.empty:
        return exploration
    history = result.trial_history
    parameter = result.parameter_history.set_index("iteration")
    map_rows = landscape.set_index("iteration")
    rows = []
    for item in exploration.to_dict(orient="records"):
        iteration = int(item["iteration"])
        later = history.loc[history["iteration"].astype(int).gt(iteration)]
        within_one = later.loc[later["iteration"].astype(int).le(iteration + 1)]
        within_two = later.loc[later["iteration"].astype(int).le(iteration + 2)]
        enabled_one = bool(within_one["trial_purpose"].eq("EXPLOIT").any())
        enabled_two = bool(within_two["trial_purpose"].eq("EXPLOIT").any())
        parameter_row = parameter.loc[iteration]
        parameter_change = float(
            np.linalg.norm(
                [float(parameter_row[f"{name}_delta"]) for name in PARAMETER_NAMES]
            )
        )
        decision_value = bool(
            enabled_two or item["accepted_as_best"]
        )
        rows.append(
            {
                **item,
                "enabled_exploit_within_1_round": enabled_one,
                "enabled_exploit_within_2_rounds": enabled_two,
                "parameter_change_l2": parameter_change,
                "prediction_map_RMS_change": float(
                    map_rows.loc[iteration, "RMS_map_change"]
                ),
                "prediction_map_max_abs_change": float(
                    map_rows.loc[iteration, "max_abs_map_change"]
                ),
                "information_gain_observed": bool(
                    item["incremental_log_information_gain"] > 0.0
                ),
                "decision_value_observed_within_2_rounds": decision_value,
                "exploration_decision_value_status": (
                    DECISION_VALUE_OBSERVED
                    if decision_value
                    else INFORMATIVE_BUT_LOW_DECISION_VALUE
                ),
                "new_stop_threshold_created": False,
            }
        )
    return pd.DataFrame(rows)


def build_repeated_exploration_audit(
    result: PolicyRunResult,
    exploration_value: pd.DataFrame,
) -> pd.DataFrame:
    if result.trial_history.empty or exploration_value.empty:
        return pd.DataFrame()
    history = result.trial_history.sort_values("iteration")
    explore_by_iteration = exploration_value.set_index("iteration")
    runs: list[list[int]] = []
    current: list[int] = []
    for row in history.to_dict(orient="records"):
        if row["trial_purpose"] == "EXPLORE":
            current.append(int(row["iteration"]))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    rows = []
    for run_index, iterations in enumerate(runs, start=1):
        values = explore_by_iteration.loc[iterations]
        if isinstance(values, pd.Series):
            values = values.to_frame().T
        x = np.arange(len(values), dtype=float)
        slope = lambda column: (
            float(np.polyfit(x, values[column].to_numpy(dtype=float), 1)[0])
            if len(values) >= 2
            else 0.0
        )
        rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "explore_run_id": run_index,
                "start_iteration": min(iterations),
                "end_iteration": max(iterations),
                "consecutive_explore_count": len(iterations),
                "information_gain_first": float(
                    values.iloc[0]["incremental_log_information_gain"]
                ),
                "information_gain_last": float(
                    values.iloc[-1]["incremental_log_information_gain"]
                ),
                "information_gain_trend_per_round": slope(
                    "incremental_log_information_gain"
                ),
                "known_region_growth_total": int(
                    values["new_supported_point_count"].sum()
                ),
                "known_region_growth_trend_per_round": slope(
                    "new_supported_point_count"
                ),
                "map_change_RMS_mean": float(
                    values["prediction_map_RMS_change"].mean()
                ),
                "map_change_RMS_trend_per_round": slope(
                    "prediction_map_RMS_change"
                ),
                "best_improvement_observed": bool(
                    values["accepted_as_best"].astype(bool).any()
                ),
                "future_exploit_within_2_rounds_observed": bool(
                    values["enabled_exploit_within_2_rounds"].astype(bool).any()
                ),
                "diminishing_return_stop_created": False,
            }
        )
    return pd.DataFrame(rows)


def build_best_trajectory_stability(result: PolicyRunResult) -> pd.DataFrame:
    rows = [
        {
            "case_id": result.summary["case_id"],
            "subject_id": result.subject_id,
            "scenario_name": result.scenario_name,
            "iteration": 0,
            "best_alpha_hip": 0.0,
            "best_alpha_knee": 0.0,
            "best_alpha_phase": 0.0,
            "best_actual_J": 1.0,
            "best_changed_this_iteration": False,
        }
    ]
    unchanged = 0
    history = (
        result.trial_history.sort_values("iteration")
        if not result.trial_history.empty
        else pd.DataFrame()
    )
    for item in history.to_dict(orient="records"):
        changed = bool(item["accepted_improvement"])
        unchanged = 0 if changed else unchanged + 1
        rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "iteration": int(item["iteration"]),
                "best_alpha_hip": item["best_alpha_hip_after"],
                "best_alpha_knee": item["best_alpha_knee_after"],
                "best_alpha_phase": item["best_alpha_phase_after"],
                "best_actual_J": item["best_actual_J_after"],
                "best_changed_this_iteration": changed,
                "consecutive_trials_without_best_change": unchanged,
            }
        )
    output = pd.DataFrame(rows)
    final = output.iloc[-1]
    at_six = output.loc[output["iteration"].le(6)].iloc[-1]
    same = bool(
        np.allclose(
            final[["best_alpha_hip", "best_alpha_knee", "best_alpha_phase", "best_actual_J"]].to_numpy(dtype=float),
            at_six[["best_alpha_hip", "best_alpha_knee", "best_alpha_phase", "best_actual_J"]].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
    )
    output["six_trial_best_equals_extended_final_best"] = same
    output["extended_natural_stop_reason"] = result.summary["stop_reason"]
    return output


def build_trial_budget_sensitivity(result: PolicyRunResult) -> pd.DataFrame:
    history = (
        result.trial_history.sort_values("iteration")
        if not result.trial_history.empty
        else pd.DataFrame(
            columns=(
                "iteration",
                "trial_purpose",
                "best_actual_J_after",
                "best_alpha_hip_after",
                "best_alpha_knee_after",
                "best_alpha_phase_after",
                "cumulative_regret_vs_best_before",
            )
        )
    )
    stop_iteration = natural_stop_iteration(result)
    rows = []
    for budget in TRIAL_BUDGETS:
        prefix = history.loc[history["iteration"].astype(int).le(budget)]
        if prefix.empty:
            best_j = 1.0
            alpha = (0.0, 0.0, 0.0)
            regret = 0.0
        else:
            last = prefix.iloc[-1]
            best_j = float(last["best_actual_J_after"])
            alpha = (
                float(last["best_alpha_hip_after"]),
                float(last["best_alpha_knee_after"]),
                float(last["best_alpha_phase_after"]),
            )
            regret = float(last["cumulative_regret_vs_best_before"])
        natural_reached = bool(
            stop_iteration is not None and stop_iteration <= budget
        )
        rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "offline_research_budget": budget,
                "budget_status": HORIZON_STATUS,
                "executed_trials": int(len(prefix)),
                "EXPLORE_count": int(prefix["trial_purpose"].eq("EXPLORE").sum()),
                "EXPLOIT_count": int(prefix["trial_purpose"].eq("EXPLOIT").sum()),
                "final_best_J": best_j,
                "final_alpha_hip": alpha[0],
                "final_alpha_knee": alpha[1],
                "final_alpha_phase": alpha[2],
                "cumulative_regret": regret,
                "natural_stop_observed_within_horizon": natural_reached,
                "stop_reason": (
                    result.summary["stop_reason"]
                    if natural_reached
                    else f"DIAGNOSTIC_BUDGET_TRUNCATION_AT_{budget}"
                ),
                "human_trial_recommendation": False,
            }
        )
    return pd.DataFrame(rows)


def build_subject_path_divergence(
    results: Sequence[PolicyRunResult],
) -> pd.DataFrame:
    selected = [
        result
        for result in results
        if result.scenario_name == "matched_linear"
        and result.policy_id == POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
    ]
    paths: dict[str, pd.DataFrame] = {}
    for result in selected:
        stability = build_best_trajectory_stability(result)
        paths[result.subject_id] = stability
    rows = []
    subject_ids = sorted(paths)
    for left_index, left_id in enumerate(subject_ids):
        for right_id in subject_ids[left_index + 1 :]:
            left = paths[left_id]
            right = paths[right_id]
            common = min(int(left["iteration"].max()), int(right["iteration"].max()))
            distances = []
            for iteration in range(common + 1):
                lrow = left.loc[left["iteration"].eq(iteration)].iloc[-1]
                rrow = right.loc[right["iteration"].eq(iteration)].iloc[-1]
                distances.append(
                    normalized_alpha_distance(
                        (lrow["best_alpha_hip"], lrow["best_alpha_knee"], lrow["best_alpha_phase"]),
                        (rrow["best_alpha_hip"], rrow["best_alpha_knee"], rrow["best_alpha_phase"]),
                    )
                )
            lfinal = left.iloc[-1]
            rfinal = right.iloc[-1]
            final_distance = normalized_alpha_distance(
                (lfinal["best_alpha_hip"], lfinal["best_alpha_knee"], lfinal["best_alpha_phase"]),
                (rfinal["best_alpha_hip"], rfinal["best_alpha_knee"], rfinal["best_alpha_phase"]),
            )
            rows.append(
                {
                    "subject_a": left_id,
                    "subject_b": right_id,
                    "common_iteration_count_including_reference": common + 1,
                    "mean_best_path_difference_formal_steps": float(np.mean(distances)),
                    "max_best_path_difference_formal_steps": float(np.max(distances)),
                    "final_alpha_difference_formal_steps": final_distance,
                    "final_paths_identical": bool(final_distance <= 1e-12),
                }
            )
    return pd.DataFrame(rows)


def classify_knee_stiff(
    result: PolicyRunResult,
    truth_rounds: pd.DataFrame,
) -> str:
    if result.subject_id != "knee_stiff":
        raise ValueError("knee-stiff classification requires knee_stiff result")
    any_true = bool(truth_rounds["true_local_improvement_available"].any())
    any_exploit = bool(result.trial_history["trial_purpose"].eq("EXPLOIT").any())
    if any_true and not any_exploit:
        return "MISSED_LOCAL_IMPROVEMENT"
    if not any_true:
        return f"{REFERENCE_LOCALLY_COMPETITIVE};{EXPLORATION_DECISION_VALUE_LOW}"
    return "LATER_EXPLOIT_OBSERVED"


def freeze_readiness_audit(
    *,
    all_rounds: pd.DataFrame,
    exploration_value: pd.DataFrame,
    boundary: pd.DataFrame,
    natural_stopping: pd.DataFrame,
    false_improvement: pd.DataFrame,
) -> dict[str, Any]:
    missed_rounds = int(all_rounds["missed_opportunity_round"].sum())
    low_value_explores = int(
        exploration_value["exploration_decision_value_status"]
        .eq(INFORMATIVE_BUT_LOW_DECISION_VALUE)
        .sum()
    ) if not exploration_value.empty else 0
    boundary_cases = int(
        boundary.loc[
            boundary["final_optimum_diagnostic_status"].eq(
                BOUNDARY_OPTIMUM_DIAGNOSTIC
            ),
            "case_id",
        ].nunique()
    ) if not boundary.empty else 0
    explore_then_exploit = int(
        exploration_value["enabled_exploit_within_2_rounds"].astype(bool).sum()
    ) if not exploration_value.empty else 0
    false_count = int(
        false_improvement["executed_false_improvement"].astype(bool).sum()
    ) if not false_improvement.empty else 0
    cap_cases = int(natural_stopping["diagnostic_cap_reached"].astype(bool).sum())
    reasons = []
    if missed_rounds:
        reasons.append("post_decision_missed_local_improvements_observed")
    if low_value_explores:
        reasons.append("informative_but_low_decision_value_exploration_observed")
    if boundary_cases:
        reasons.append("boundary_optimum_diagnostics_require_method_interpretation")
    if false_count:
        reasons.append("executed_false_improvements_observed")
    if cap_cases:
        reasons.append("some_policies_did_not_naturally_stop_by_diagnostic_cap")
    status = (
        OFFLINE_METHOD_REQUIRES_REVISION
        if reasons
        else OFFLINE_METHOD_ARCHITECTURE_READY_TO_FREEZE
    )
    return {
        "status": status,
        "explore_update_future_exploit_chain_observed_count": explore_then_exploit,
        "whole_map_recomputation_architecture_stable": True,
        "missed_opportunity_round_count": missed_rounds,
        "informative_but_low_decision_value_explore_count": low_value_explores,
        "boundary_optimum_case_count": boundary_cases,
        "executed_false_improvement_count": false_count,
        "diagnostic_cap_case_count": cap_cases,
        "revision_reasons": reasons,
        "not_human_ready": True,
        "not_robot_motion_approved": True,
        "thresholds_modified": False,
    }


__all__ = [
    "AUDIT_PROTOCOL_ID",
    "BOUNDARY_OPTIMUM_DIAGNOSTIC",
    "CORRECT_CONSERVATIVE_STOP",
    "CORRECT_LOCAL_STOP",
    "DECISION_VALUE_OBSERVED",
    "EXPLORATION_DECISION_VALUE_LOW",
    "EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON",
    "HORIZON_STATUS",
    "INFORMATIVE_BUT_LOW_DECISION_VALUE",
    "MISSED_IMPROVEMENT",
    "OFFLINE_METHOD_ARCHITECTURE_READY_TO_FREEZE",
    "OFFLINE_METHOD_REQUIRES_REVISION",
    "POST_DECISION_TRUTH_ROLE",
    "PREMATURE_CONSERVATIVE_STOP",
    "PostDecisionTruthAudit",
    "REFERENCE_LOCALLY_COMPETITIVE",
    "TRIAL_BUDGETS",
    "audit_post_decision_local_truth",
    "build_best_trajectory_stability",
    "build_boundary_chasing_audit",
    "build_exploration_decision_value",
    "build_marginal_improvement",
    "build_natural_stopping_summary",
    "build_prediction_landscape_evolution",
    "build_repeated_exploration_audit",
    "build_subject_path_divergence",
    "build_trial_budget_sensitivity",
    "classify_knee_stiff",
    "corridor_boundary_distances",
    "freeze_readiness_audit",
    "missed_improvement_summary",
    "natural_stop_iteration",
    "normalized_alpha_distance",
    "parameter_bound_distances",
]
