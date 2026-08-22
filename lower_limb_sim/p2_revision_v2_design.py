"""Offline design analysis for a possible P2 Revision V2.

This module does not implement or modify P2.  It turns already-frozen,
post-hoc audit artifacts into candidate protocol definitions and retrospective
counterfactuals.  Truth remains an evaluation label and never becomes a
policy, fitting, calibration-freeze, or stopping input.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .sequential_personalization import (
    INITIAL_STEP_HIP_DEG,
    INITIAL_STEP_KNEE_DEG,
    INITIAL_STEP_PHASE,
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
)


DESIGN_PROTOCOL_ID = "P2_REVISION_V2_DESIGN_ANALYSIS_V1"
LOCAL_PROTOCOL_ID = "LOCAL_DECISION_VALIDATION_PROTOCOL_V1"
EXPLORATION_STOPPING_CANDIDATE_ID = (
    "EXPLORATION_VALUE_AWARE_STOPPING_CANDIDATE_V1"
)
DESIGN_STATUS = "REVISION_DESIGN_NOT_FROZEN"
OFFLINE_METHOD_STATUS = "OFFLINE_METHOD_REQUIRES_REVISION"
RETROSPECTIVE_LOCAL_ROLE = (
    "RETROSPECTIVE_LOCAL_DECISION_OPPORTUNITY_CANDIDATE_NOT_DESIGNATED_VALIDATION"
)
LOCAL_UNCERTAINTY_CANDIDATE_STATUS = (
    "RESEARCH_CANDIDATE_ONLY_REQUIRES_INDEPENDENT_DESIGNATED_VALIDATION"
)
P2_V2_IMPLEMENTATION_STATUS = (
    "P2_V2_RESEARCH_PROTOTYPE_WORTH_IMPLEMENTING_AFTER_DESIGN_FREEZE"
)

_ALPHA_COLUMNS = ("hip", "knee", "phase")
_GRID_STEPS = np.asarray(
    (GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP), dtype=float
)
_TRUST_LEVELS = {
    "hip": (INITIAL_STEP_HIP_DEG, INITIAL_STEP_HIP_DEG / 2.0, MINIMUM_STEP_HIP_DEG),
    "knee": (
        INITIAL_STEP_KNEE_DEG,
        INITIAL_STEP_KNEE_DEG / 2.0,
        MINIMUM_STEP_KNEE_DEG,
    ),
    "phase": (
        INITIAL_STEP_PHASE,
        INITIAL_STEP_PHASE / 2.0,
        MINIMUM_STEP_PHASE,
    ),
}


def local_decision_validation_protocol() -> dict[str, Any]:
    """Return the design candidate without creating a numerical threshold."""

    return {
        "protocol_id": LOCAL_PROTOCOL_ID,
        "status": DESIGN_STATUS,
        "purpose": (
            "estimate decision-error uncertainty on the same formal coordinate "
            "relationship used by a future local P2 decision"
        ),
        "local_pair_definition": {
            "both_points": "geometrically_admissible_generator_alpha_points",
            "relationship": (
                "exactly_one_of_hip_knee_phase_differs_and_the_signed_difference_"
                "equals_the_current_trust_region_step_for_that_coordinate"
            ),
            "allowed_existing_trust_levels": {
                name: list(values) for name, values in _TRUST_LEVELS.items()
            },
            "formal_grid_steps": {
                "hip_deg": GRID_HIP_STEP_DEG,
                "knee_deg": GRID_KNEE_STEP_DEG,
                "phase": GRID_PHASE_STEP,
            },
            "physical_distance_threshold": None,
            "euclidean_physical_distance_used": False,
            "clipping_allowed": False,
        },
        "future_designated_validation_requirements": {
            "split_role": "DESIGNATED_LOCAL_DECISION_VALIDATION_ONLY",
            "must_be_predeclared_before_policy_evaluation": True,
            "must_not_be_model_fitting_data": True,
            "must_not_be_adaptation_executed_outcome": True,
            "must_not_be_heldout_final_test": True,
            "must_preserve_pair_current_candidate_identity": True,
            "must_stratify_by_coordinate_and_existing_trust_level": True,
            "must_stratify_by_model_support_status": True,
            "must_cover_all_existing_trust_levels_before_generalizing": True,
            "minimum_sample_count": None,
            "sample_count_requires_review": True,
        },
        "candidate_statistics_only": ["max", "P95", "P99"],
        "threshold_frozen": False,
        "current_P2_modified": False,
        "retrospective_evidence_role": RETROSPECTIVE_LOCAL_ROLE,
    }


def _current_best_before_iteration(
    best_history: pd.DataFrame,
    case_id: str,
    iteration: int,
) -> tuple[float, float, float]:
    rows = best_history.loc[
        best_history["case_id"].astype(str).eq(str(case_id))
        & best_history["iteration"].astype(int).lt(int(iteration))
    ].sort_values("iteration")
    if rows.empty:
        return (0.0, 0.0, 0.0)
    row = rows.iloc[-1]
    return (
        float(row["best_alpha_hip"]),
        float(row["best_alpha_knee"]),
        float(row["best_alpha_phase"]),
    )


def build_retrospective_local_pair_errors(
    root_counterfactual: pd.DataFrame,
    best_history: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct frozen candidate-current relationships; do not calibrate P2."""

    g0 = root_counterfactual.loc[
        root_counterfactual["guard_id"].eq("G0_CURRENT_GLOBAL_MAX")
    ].copy()
    rows: list[dict[str, Any]] = []
    for item in g0.to_dict(orient="records"):
        current = _current_best_before_iteration(
            best_history, str(item["case_id"]), int(item["iteration"])
        )
        candidate = np.asarray(
            (item["alpha_hip"], item["alpha_knee"], item["alpha_phase"]),
            dtype=float,
        )
        difference = candidate - np.asarray(current, dtype=float)
        changed = ~np.isclose(difference, 0.0, atol=1e-12, rtol=0.0)
        changed_count = int(changed.sum())
        axis_index = int(np.flatnonzero(changed)[0]) if changed_count == 1 else -1
        axis = _ALPHA_COLUMNS[axis_index] if axis_index >= 0 else "INVALID"
        absolute_step = abs(float(difference[axis_index])) if axis_index >= 0 else np.nan
        allowed = (
            any(
                math.isclose(absolute_step, level, abs_tol=1e-12, rel_tol=0.0)
                for level in _TRUST_LEVELS[axis]
            )
            if axis in _TRUST_LEVELS
            else False
        )
        normalized = difference / _GRID_STEPS
        rows.append(
            {
                "case_id": item["case_id"],
                "subject_id": item["subject_id"],
                "scenario_name": item["scenario_name"],
                "iteration": int(item["iteration"]),
                "current_alpha_hip": current[0],
                "current_alpha_knee": current[1],
                "current_alpha_phase": current[2],
                "candidate_trajectory_id": item["trajectory_id"],
                "candidate_alpha_hip": float(candidate[0]),
                "candidate_alpha_knee": float(candidate[1]),
                "candidate_alpha_phase": float(candidate[2]),
                "delta_alpha_hip": float(difference[0]),
                "delta_alpha_knee": float(difference[1]),
                "delta_alpha_phase": float(difference[2]),
                "changed_coordinate_count": changed_count,
                "changed_coordinate": axis,
                "signed_coordinate_direction": (
                    "POSITIVE" if axis_index >= 0 and difference[axis_index] > 0 else "NEGATIVE"
                ),
                "absolute_coordinate_step": absolute_step,
                "formal_grid_step_multiplier": (
                    abs(float(normalized[axis_index])) if axis_index >= 0 else np.nan
                ),
                "formal_local_candidate_relationship_valid": bool(
                    changed_count == 1 and allowed
                ),
                "pair_relationship": "SIGNED_COORDINATE_TRUST_REGION_MOVE",
                "physical_distance_used": False,
                "delta_J_pred": float(item["delta_J_pred"]),
                "delta_J_actual_posthoc": float(
                    item["delta_J_truth_post_decision"]
                ),
                "e_delta_J": abs(
                    float(item["delta_J_pred"])
                    - float(item["delta_J_truth_post_decision"])
                ),
                "model_supported": bool(item["model_supported"]),
                "current_G0_would_exploit": bool(item["would_exploit"]),
                "true_improvement_posthoc": bool(item["true_improvement"]),
                "evidence_role": RETROSPECTIVE_LOCAL_ROLE,
                "designated_validation": False,
                "used_to_modify_current_policy": False,
                "truth_used_as_policy_input": False,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty or not output["formal_local_candidate_relationship_valid"].all():
        raise RuntimeError("historical decision opportunities violate local-pair design")
    return output


def _correlations(table: pd.DataFrame) -> dict[str, float]:
    pred = table["delta_J_pred"].astype(float)
    actual_column = (
        "delta_J_actual_posthoc"
        if "delta_J_actual_posthoc" in table
        else "delta_J_actual"
    )
    actual = table[actual_column].astype(float)
    error = table["e_delta_J"].astype(float)
    return {
        "pearson_delta_pred_vs_actual": float(pred.corr(actual, method="pearson")),
        "spearman_delta_pred_vs_actual": float(pred.corr(actual, method="spearman")),
        "pearson_error_vs_abs_actual_delta": float(
            error.corr(actual.abs(), method="pearson")
        ),
        "spearman_error_vs_abs_actual_delta": float(
            error.corr(actual.abs(), method="spearman")
        ),
    }


def build_global_vs_local_error_distribution(
    global_provenance: pd.DataFrame,
    local_pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = (
        (
            "CURRENT_GLOBAL_IDENTIFICATION_EXCITATION_PAIR_INSTANCES",
            global_provenance,
            "DESIGNATED_VALIDATION_BUT_NOT_PERSONALIZATION_ALPHA_SCALE",
        ),
        (
            "RETROSPECTIVE_LOCAL_DECISION_OPPORTUNITY_PAIRS",
            local_pairs,
            RETROSPECTIVE_LOCAL_ROLE,
        ),
    )
    for pair_class, table, role in sources:
        errors = table["e_delta_J"].to_numpy(dtype=float)
        rows.append(
            {
                "pair_class": pair_class,
                "pair_instance_count": int(len(table)),
                "unique_case_count": int(table["case_id"].nunique()),
                "mean_e_delta_J": float(np.mean(errors)),
                "p95_e_delta_J": float(np.percentile(errors, 95)),
                "p99_e_delta_J": float(np.percentile(errors, 99)),
                "max_e_delta_J": float(np.max(errors)),
                **_correlations(table),
                "formal_local_alpha_relationship": pair_class.startswith(
                    "RETROSPECTIVE_LOCAL"
                ),
                "evidence_role": role,
                "threshold_frozen": False,
            }
        )
    return pd.DataFrame(rows)


def build_local_uncertainty_candidates(local_pairs: pd.DataFrame) -> pd.DataFrame:
    """Create pooled and leave-one-case-out candidates, never a threshold."""

    rows: list[dict[str, Any]] = []
    candidate_functions = {
        "LOCAL_MAX_UNCERTAINTY_CANDIDATE": lambda values: np.max(values),
        "LOCAL_P95_UNCERTAINTY_CANDIDATE": lambda values: np.percentile(values, 95),
        "LOCAL_P99_UNCERTAINTY_CANDIDATE": lambda values: np.percentile(values, 99),
    }
    cases = sorted(local_pairs["case_id"].astype(str).unique())
    for scope, excluded in [("POOLED_RETROSPECTIVE", ""), *[("LEAVE_ONE_CASE_OUT", case) for case in cases]]:
        calibration = (
            local_pairs
            if not excluded
            else local_pairs.loc[~local_pairs["case_id"].astype(str).eq(excluded)]
        )
        values = calibration["e_delta_J"].to_numpy(dtype=float)
        for candidate_id, function in candidate_functions.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "calibration_scope": scope,
                    "excluded_evaluation_case_id": excluded,
                    "calibration_pair_count": int(len(calibration)),
                    "calibration_case_count": int(calibration["case_id"].nunique()),
                    "candidate_uncertainty_bound": float(function(values)),
                    "evidence_role": RETROSPECTIVE_LOCAL_ROLE,
                    "candidate_status": LOCAL_UNCERTAINTY_CANDIDATE_STATUS,
                    "threshold_frozen": False,
                    "current_P2_modified": False,
                    "heldout_final_test_used": False,
                }
            )
    return pd.DataFrame(rows)


def build_local_pair_stratum_summary(local_pairs: pd.DataFrame) -> pd.DataFrame:
    """Expose scale/support coverage limits of the retrospective evidence."""

    rows: list[dict[str, Any]] = []
    grouped = local_pairs.groupby(
        ["changed_coordinate", "absolute_coordinate_step", "model_supported"],
        sort=True,
    )
    for (coordinate, step, supported), group in grouped:
        errors = group["e_delta_J"].to_numpy(dtype=float)
        rows.append(
            {
                "changed_coordinate": coordinate,
                "absolute_coordinate_step": float(step),
                "model_supported": bool(supported),
                "pair_count": int(len(group)),
                "case_count": int(group["case_id"].nunique()),
                "mean_e_delta_J": float(np.mean(errors)),
                "p95_e_delta_J": float(np.percentile(errors, 95)),
                "p99_e_delta_J": float(np.percentile(errors, 99)),
                "max_e_delta_J": float(np.max(errors)),
                **_correlations(group),
                "retrospective_evidence_only": True,
                "threshold_frozen": False,
            }
        )
    return pd.DataFrame(rows)


def build_local_guard_counterfactual(
    local_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use leave-one-case-out bounds for G1/G2 retrospective evaluation."""

    candidate_to_guard = {
        "LOCAL_MAX_UNCERTAINTY_CANDIDATE": "G1_LOCAL_MAX_CANDIDATE",
        "LOCAL_P95_UNCERTAINTY_CANDIDATE": "G2_LOCAL_P95_CANDIDATE",
        "LOCAL_P99_UNCERTAINTY_CANDIDATE": "G2_LOCAL_P99_CANDIDATE",
    }
    bound_lookup = {
        (str(row.excluded_evaluation_case_id), str(row.candidate_id)): float(
            row.candidate_uncertainty_bound
        )
        for row in candidates.loc[
            candidates["calibration_scope"].eq("LEAVE_ONE_CASE_OUT")
        ].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for item in local_pairs.to_dict(orient="records"):
        guards: list[tuple[str, float, str]] = [
            (
                "G0_CURRENT_GLOBAL_MAX",
                np.nan,
                "CURRENT_FROZEN_PER_CASE_ITERATION_BOUND",
            )
        ]
        for candidate_id, guard_id in candidate_to_guard.items():
            guards.append(
                (
                    guard_id,
                    bound_lookup[(str(item["case_id"]), candidate_id)],
                    "LEAVE_ONE_CASE_OUT_RETROSPECTIVE_LOCAL_BOUND",
                )
            )
        true_improvement = bool(item["true_improvement_posthoc"])
        for guard_id, bound, bound_role in guards:
            if guard_id == "G0_CURRENT_GLOBAL_MAX":
                would_exploit = bool(item["current_G0_would_exploit"])
                effective_bound = np.nan
            else:
                effective_bound = float(bound)
                would_exploit = bool(
                    item["model_supported"]
                    and -float(item["delta_J_pred"])
                    - effective_bound
                    - OBJECTIVE_EQUIVALENCE_TOLERANCE
                    > 0.0
                )
            rows.append(
                {
                    **item,
                    "guard_id": guard_id,
                    "uncertainty_bound": effective_bound,
                    "bound_role": bound_role,
                    "would_exploit": would_exploit,
                    "false_improvement_posthoc": bool(
                        would_exploit and not true_improvement
                    ),
                    "missed_improvement_posthoc": bool(
                        not would_exploit and true_improvement
                    ),
                    "counterfactual_only": True,
                    "policy_modified": False,
                    "trajectory_executed": False,
                    "truth_used_to_construct_policy": False,
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    g0_metrics: dict[str, int] | None = None
    for guard_id, group in detail.groupby("guard_id", sort=False):
        rounds = group.groupby(["case_id", "iteration"], as_index=False).agg(
            true_improvement_available=("true_improvement_posthoc", "any"),
            would_exploit_any=("would_exploit", "any"),
            false_improvement_any=("false_improvement_posthoc", "any"),
        )
        metrics = {
            "would_exploit_candidate_count": int(group["would_exploit"].sum()),
            "missed_improvement_candidate_count": int(
                group["missed_improvement_posthoc"].sum()
            ),
            "false_improvement_candidate_count": int(
                group["false_improvement_posthoc"].sum()
            ),
            "conservative_stop_round_count": int(
                (
                    rounds["true_improvement_available"]
                    & ~rounds["would_exploit_any"]
                ).sum()
            ),
        }
        if guard_id == "G0_CURRENT_GLOBAL_MAX":
            g0_metrics = metrics
        summary_rows.append(
            {
                "guard_id": guard_id,
                **metrics,
                "candidate_status": (
                    "CURRENT_BEHAVIOR_REPLAY_NOT_REVISED"
                    if guard_id == "G0_CURRENT_GLOBAL_MAX"
                    else LOCAL_UNCERTAINTY_CANDIDATE_STATUS
                ),
                "counterfactual_only": guard_id != "G0_CURRENT_GLOBAL_MAX",
                "threshold_frozen": False,
                "current_behavior_replay": guard_id == "G0_CURRENT_GLOBAL_MAX",
            }
        )
    summary = pd.DataFrame(summary_rows)
    if g0_metrics is None:
        raise RuntimeError("counterfactual summary lacks G0")
    for name in (
        "missed_improvement_candidate_count",
        "false_improvement_candidate_count",
        "conservative_stop_round_count",
    ):
        summary[f"change_vs_G0_{name}"] = summary[name] - g0_metrics[name]
    return detail, summary


def build_exploration_value_components(
    exploration: pd.DataFrame,
) -> pd.DataFrame:
    output = exploration.copy(deep=True)
    output["MODEL_VALUE"] = (
        output["theta_changed_exactly"].astype(bool)
        | output["prediction_map_changed_exactly"].astype(bool)
        | output["validation_error_changed_exactly"].astype(bool)
    )
    output["SUPPORT_VALUE"] = output["new_supported_points"].astype(float).gt(0.0)
    output["DECISION_VALUE"] = (
        output["newly_enabled_exploit_candidates"].astype(int).gt(0)
        | output["best_changed_under_existing_0p005_rule"].astype(bool)
    )
    output["support_is_decision_value"] = False
    output["exact_zero_decision_value_round"] = (
        ~output["MODEL_VALUE"]
        & output["SUPPORT_VALUE"]
        & ~output["DECISION_VALUE"]
    )
    output["stopping_feature_uses_truth"] = False
    output["new_numeric_threshold_created"] = False
    return output


def _first_consecutive_low_value_trigger(
    case: pd.DataFrame,
    consecutive_required: int,
) -> int | None:
    run = 0
    previous_iteration: int | None = None
    for row in case.sort_values("iteration").itertuples(index=False):
        iteration = int(row.iteration)
        consecutive = previous_iteration is not None and iteration == previous_iteration + 1
        if bool(row.exact_zero_decision_value_round):
            run = run + 1 if consecutive else 1
        else:
            run = 0
        if run >= consecutive_required:
            return iteration
        previous_iteration = iteration
    return None


def build_exploration_stopping_counterfactual(
    components: pd.DataFrame,
    executed_history: pd.DataFrame,
    natural_stopping: pd.DataFrame,
) -> pd.DataFrame:
    p2_natural = natural_stopping.loc[
        natural_stopping["policy_id"].eq("P2_DECISION_GUARDED_EXPLORE_EXPLOIT")
    ].set_index("case_id")
    rows: list[dict[str, Any]] = []
    candidate_definitions = (
        ("S0_CURRENT_P2", 0),
        ("S1_STOP_AFTER_FIRST_EXACT_ZERO_DECISION_VALUE_EXPLORE", 1),
        ("S2_STOP_AFTER_TWO_CONSECUTIVE_EXACT_ZERO_DECISION_VALUE_EXPLORES", 2),
    )
    for case_id, case in components.groupby("case_id", sort=False):
        history = executed_history.loc[
            executed_history["case_id"].astype(str).eq(str(case_id))
        ].sort_values("iteration")
        current_count = int(p2_natural.loc[case_id, "executed_trial_count"])
        for candidate_id, required in candidate_definitions:
            trigger = (
                None
                if required == 0
                else _first_consecutive_low_value_trigger(case, required)
            )
            later = (
                history.iloc[0:0]
                if trigger is None
                else history.loc[history["iteration"].astype(int).gt(trigger)]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "stopping_candidate_id": candidate_id,
                    "consecutive_exact_zero_rounds_required": (
                        required if required else np.nan
                    ),
                    "trigger_observed": trigger is not None,
                    "counterfactual_stop_after_iteration": trigger,
                    "current_executed_trial_count": current_count,
                    "counterfactual_executed_trial_count": (
                        current_count if trigger is None else int(trigger)
                    ),
                    "executed_trials_avoided": (
                        0 if trigger is None else max(current_count - int(trigger), 0)
                    ),
                    "later_exploit_trials_in_frozen_history": int(
                        later["trial_purpose"].eq("EXPLOIT").sum()
                    ),
                    "later_accepted_best_changes_in_frozen_history": int(
                        later["accepted_improvement"].astype(bool).sum()
                    ),
                    "features": (
                        "exact_parameter_change;exact_map_change;validation_error_change;"
                        "new_exploit_eligibility;best_change;support_and_information_reported_separately"
                    ),
                    "support_alone_causes_continuation": False,
                    "truth_feature_used": False,
                    "candidate_enabled": False,
                    "threshold_frozen": False,
                    "current_policy_modified": False,
                }
            )
    return pd.DataFrame(rows)


def build_subject_specificity_gap(
    truth_summary: pd.DataFrame,
    local_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in truth_summary.to_dict(orient="records"):
        truth_alpha = np.asarray(
            (
                item["alpha_truth_global_hip"],
                item["alpha_truth_global_knee"],
                item["alpha_truth_global_phase"],
            ),
            dtype=float,
        )
        selected_alpha = np.asarray(
            (
                item["p2_final_alpha_hip"],
                item["p2_final_alpha_knee"],
                item["p2_final_alpha_phase"],
            ),
            dtype=float,
        )
        gap = selected_alpha - truth_alpha
        regret = float(item["p2_final_truth_regret_vs_global"])
        equivalent = regret <= OBJECTIVE_EQUIVALENCE_TOLERANCE + 1e-12
        knee_stiff = str(item["subject_id"]) == "knee_stiff"
        final_local_best = np.nan
        final_local_pair_count = 0
        if local_pairs is not None:
            subject_pairs = local_pairs.loc[
                local_pairs["subject_id"].astype(str).eq(str(item["subject_id"]))
                & local_pairs["scenario_name"].astype(str).eq("matched_linear")
            ]
            at_selected = subject_pairs.loc[
                np.isclose(
                    subject_pairs["current_alpha_hip"].astype(float),
                    selected_alpha[0],
                )
                & np.isclose(
                    subject_pairs["current_alpha_knee"].astype(float),
                    selected_alpha[1],
                )
                & np.isclose(
                    subject_pairs["current_alpha_phase"].astype(float),
                    selected_alpha[2],
                )
            ]
            final_local_pair_count = int(len(at_selected))
            if not at_selected.empty:
                final_local_best = float(at_selected["delta_J_actual_posthoc"].min())
        rows.append(
            {
                "subject_id": item["subject_id"],
                "truth_optimum_hip": truth_alpha[0],
                "truth_optimum_knee": truth_alpha[1],
                "truth_optimum_phase": truth_alpha[2],
                "P2_selected_hip": selected_alpha[0],
                "P2_selected_knee": selected_alpha[1],
                "P2_selected_phase": selected_alpha[2],
                "alpha_gap_hip": gap[0],
                "alpha_gap_knee": gap[1],
                "alpha_gap_phase": gap[2],
                "alpha_gap_formal_grid_euclidean": float(
                    np.linalg.norm(gap / _GRID_STEPS)
                ),
                "J_truth_global": item["J_truth_global"],
                "J_truth_at_P2_selected": item["J_truth_at_p2_final"],
                "J_truth_regret": regret,
                "gap_within_existing_0p005_equivalence": equivalent,
                "retrospective_local_pair_count_at_P2_selected": final_local_pair_count,
                "best_one_step_truth_delta_J_at_P2_selected_posthoc": final_local_best,
                "best_one_step_improvement_exceeds_existing_0p005": bool(
                    np.isfinite(final_local_best)
                    and -final_local_best
                    > OBJECTIVE_EQUIVALENCE_TOLERANCE + 1e-12
                ),
                "A_objective_contribution": (
                    "common_knee_reduction_boundary_tendency_but_complete_optima_remain_subject_dependent"
                ),
                "B_generator_direction_contribution": (
                    "truth_optimum_is_inside_existing_space_but_touches_knee_lower_bound;"
                    "boundary_limits_identification_beyond_minus5_not_observed_gap"
                ),
                "generator_contains_observed_truth_optimum": True,
                "truth_optimum_touches_generator_knee_boundary": bool(
                    math.isclose(truth_alpha[1], -5.0, abs_tol=1e-12)
                ),
                "generator_expansion_justified": False,
                "C_search_policy_contribution": (
                    "local_stepwise_acceptance_cannot_accumulate_repeated_subthreshold_knee_moves"
                    if knee_stiff
                    else "alpha_difference_exists_but_is_objective_equivalent_under_frozen_tolerance"
                ),
                "D_guard_contribution": (
                    "uncertainty_not_primary;frozen_0p005_term_dominates_each_one_step_margin"
                    if knee_stiff
                    else "no_decision_relevant_final_gap_beyond_frozen_equivalence"
                ),
                "specificity_gap_classification": (
                    "MEANINGFUL_GAP_LOCAL_STEP_TOLERANCE_AND_POLICY_ACCUMULATION_LIMIT"
                    if not equivalent
                    else "ALPHA_GAP_WITHIN_FROZEN_OBJECTIVE_EQUIVALENCE"
                ),
                "truth_used_to_modify_policy": False,
                "objective_modified": False,
                "generator_modified": False,
            }
        )
    return pd.DataFrame(rows)


def design_recommendation(
    guard_summary: pd.DataFrame,
    stopping: pd.DataFrame,
    specificity: pd.DataFrame,
) -> dict[str, Any]:
    p95 = guard_summary.loc[
        guard_summary["guard_id"].eq("G2_LOCAL_P95_CANDIDATE")
    ].iloc[0]
    s1 = stopping.loc[
        stopping["stopping_candidate_id"].eq(
            "S1_STOP_AFTER_FIRST_EXACT_ZERO_DECISION_VALUE_EXPLORE"
        )
    ]
    meaningful_gap = specificity.loc[
        ~specificity["gap_within_existing_0p005_equivalence"].astype(bool)
    ]
    return {
        "local_uncertainty_next_version": (
            "PROMISING_AS_AN_ARCHITECTURE;P95_RETROSPECTIVE_CANDIDATE_REQUIRES_"
            "INDEPENDENT_DESIGNATED_LOCAL_VALIDATION"
        ),
        "local_P95_change_in_missed_candidates_vs_G0": int(
            p95["change_vs_G0_missed_improvement_candidate_count"]
        ),
        "local_P95_change_in_false_candidates_vs_G0": int(
            p95["change_vs_G0_false_improvement_candidate_count"]
        ),
        "decision_value_aware_exploration_next_version": (
            "PROMISING_AS_RESEARCH_CANDIDATE_NO_STOP_RULE_FROZEN"
        ),
        "S1_historical_trials_avoided": int(s1["executed_trials_avoided"].sum()),
        "S1_later_exploits_in_frozen_history": int(
            s1["later_exploit_trials_in_frozen_history"].sum()
        ),
        "objective_change_needed": False,
        "objective_reason": (
            "retains_subject_discrimination_and_is_frozen;boundary_interpretation_not_"
            "an_authorization_to_change_formula"
        ),
        "generator_direction_expansion_needed": False,
        "generator_reason": (
            "observed_truth_optima_are_already_inside_current_space;expansion_would_"
            "extend_boundary_chasing_without_scientific_review"
        ),
        "meaningful_subject_gap_count": int(len(meaningful_gap)),
        "minimum_revision_set": [
            "add_a_reviewed_local_decision_uncertainty_provider_but_keep_current_G0_default_until_frozen",
            "add_separate_MODEL_VALUE_SUPPORT_VALUE_DECISION_VALUE_observability",
            "add_a_disabled_decision_value_stopping_candidate_with_audit_output",
            "retain_all_frozen_objective_model_generator_reference_ROM_tolerance_and_support_gates",
        ],
        "implementation_status": P2_V2_IMPLEMENTATION_STATUS,
        "design_status": DESIGN_STATUS,
        "offline_method_status": OFFLINE_METHOD_STATUS,
        "formal_personalization_executed": False,
        "current_P2_modified": False,
    }


__all__ = [
    "DESIGN_PROTOCOL_ID",
    "DESIGN_STATUS",
    "EXPLORATION_STOPPING_CANDIDATE_ID",
    "LOCAL_PROTOCOL_ID",
    "LOCAL_UNCERTAINTY_CANDIDATE_STATUS",
    "OFFLINE_METHOD_STATUS",
    "P2_V2_IMPLEMENTATION_STATUS",
    "RETROSPECTIVE_LOCAL_ROLE",
    "build_exploration_stopping_counterfactual",
    "build_exploration_value_components",
    "build_global_vs_local_error_distribution",
    "build_local_guard_counterfactual",
    "build_local_pair_stratum_summary",
    "build_local_uncertainty_candidates",
    "build_retrospective_local_pair_errors",
    "build_subject_specificity_gap",
    "design_recommendation",
    "local_decision_validation_protocol",
]
