"""Formal research-design protocols for a possible P2 V2.

This module creates pre-registered plans and shadow rule candidates.  It does
not modify P2 V1, evaluate formal personalization, or use final truth values to
choose designated local-validation pairs.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
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


FORMAL_DESIGN_ID = "P2_V2_FORMAL_RESEARCH_PROTOCOL_DESIGN_V1"
LOCAL_PROTOCOL_ID = "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1"
CUMULATIVE_RULE_ID = "CUMULATIVE_DECISION_RULE_V1"
STOPPING_RULE_ID = "DECISION_VALUE_EXPLORATION_STOPPING_V1"
DESIGN_STATUS = "FORMAL_RESEARCH_PROTOCOL_DESIGNED_NOT_POLICY_FROZEN"
OFFLINE_METHOD_STATUS = "OFFLINE_METHOD_REQUIRES_REVISION"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"
PAIRS_PER_LOCATION_CLASS = 12
PAIR_OUTCOME_STATUS = "PENDING_INDEPENDENT_EVALUATION_AFTER_PLAN_FREEZE"
SAMPLE_COUNT_STATUS = "PILOT_DESIGN_COUNT_REQUIRES_POWER_AND_REVIEW_APPROVAL"

_COORDINATE_COLUMNS = {
    "hip": "hip_delta",
    "knee": "knee_delta",
    "phase": "phase_delta",
}
_BOUND_KEYS = {
    "hip": "hip_amplitude_delta_deg",
    "knee": "knee_amplitude_delta_deg",
    "phase": "knee_phase_shift",
}
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
_GRID_STEPS = {
    "hip": GRID_HIP_STEP_DEG,
    "knee": GRID_KNEE_STEP_DEG,
    "phase": GRID_PHASE_STEP,
}
_LOCATION_CLASSES = ("LOWER_EDGE", "INTERIOR", "UPPER_EDGE")
_UNCERTAINTY_AGGREGATION_CANDIDATES = (
    "SUM_OF_STEPWISE_LOCAL_BOUNDS_WORST_CASE",
    "EMPIRICAL_BLOCK_P95_REQUIRES_NEW_DESIGNATED_BUNDLES",
    "RSS_ONLY_AFTER_RESIDUAL_INDEPENDENCE_VALIDATED",
)
_ACCUMULATION_STEP_CANDIDATES = (2, 3, 5)
_STOP_CONSECUTIVE_CANDIDATES = (1, 2, 3)


def _alpha_key(values: Sequence[float]) -> tuple[float, float, float]:
    return tuple(round(float(value), 12) for value in values)  # type: ignore[return-value]


def _pair_identity(
    coordinate: str,
    trust_step: float,
    alpha_i: tuple[float, float, float],
    alpha_j: tuple[float, float, float],
) -> tuple[str, str]:
    payload = (
        f"{LOCAL_PROTOCOL_ID}|{coordinate}|{trust_step:.12g}|"
        f"{alpha_i}|{alpha_j}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"dlv_pair_{digest[:24]}", digest


def _location_class(
    coordinate: str,
    alpha_i: tuple[float, float, float],
    alpha_j: tuple[float, float, float],
) -> str:
    index = ("hip", "knee", "phase").index(coordinate)
    lower, upper = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[_BOUND_KEYS[coordinate]]
    if math.isclose(alpha_i[index], float(lower), abs_tol=1e-12, rel_tol=0.0):
        return "LOWER_EDGE"
    if math.isclose(alpha_j[index], float(upper), abs_tol=1e-12, rel_tol=0.0):
        return "UPPER_EDGE"
    return "INTERIOR"


def enumerate_designated_local_pair_universe(
    parameter_lattice: pd.DataFrame,
) -> pd.DataFrame:
    """Enumerate pairs using geometry and trust levels only; no truth input."""

    required = {
        "trajectory_id",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "geometrically_admissible",
    }
    missing = required.difference(parameter_lattice.columns)
    if missing:
        raise ValueError(f"parameter lattice missing columns: {sorted(missing)}")
    if not parameter_lattice["geometrically_admissible"].astype(bool).all():
        raise ValueError("designated pair universe accepts geometry-valid lattice only")
    lookup = {
        _alpha_key((row.hip_delta, row.knee_delta, row.phase_delta)): str(
            row.trajectory_id
        )
        for row in parameter_lattice.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    coordinate_index = {"hip": 0, "knee": 1, "phase": 2}
    for coordinate, levels in _TRUST_LEVELS.items():
        index = coordinate_index[coordinate]
        for level_number, step in enumerate(levels):
            for alpha_i, trajectory_i in lookup.items():
                candidate = list(alpha_i)
                candidate[index] += float(step)
                alpha_j = _alpha_key(candidate)
                trajectory_j = lookup.get(alpha_j)
                if trajectory_j is None:
                    continue
                pair_id, selection_hash = _pair_identity(
                    coordinate, float(step), alpha_i, alpha_j
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "selection_hash": selection_hash,
                        "coordinate": coordinate,
                        "trust_level": ("INITIAL", "HALF", "MINIMUM")[
                            level_number
                        ],
                        "trust_step": float(step),
                        "location_class": _location_class(
                            coordinate, alpha_i, alpha_j
                        ),
                        "trajectory_i": trajectory_i,
                        "trajectory_j": trajectory_j,
                        "alpha_i_hip": alpha_i[0],
                        "alpha_i_knee": alpha_i[1],
                        "alpha_i_phase": alpha_i[2],
                        "alpha_j_hip": alpha_j[0],
                        "alpha_j_knee": alpha_j[1],
                        "alpha_j_phase": alpha_j[2],
                        "delta_alpha_hip": alpha_j[0] - alpha_i[0],
                        "delta_alpha_knee": alpha_j[1] - alpha_i[1],
                        "delta_alpha_phase": alpha_j[2] - alpha_i[2],
                        "alpha_distance_formal_grid_steps": float(
                            step / _GRID_STEPS[coordinate]
                        ),
                        "alpha_distance_definition": (
                            "NORMALIZED_FORMAL_GENERATOR_GRID_NOT_PHYSICAL_DISTANCE"
                        ),
                        "canonical_orientation": "LOWER_COORDINATE_TO_HIGHER_COORDINATE",
                        "reverse_pair_error_is_symmetric": True,
                        "inside_existing_generator_bounds": True,
                        "geometrically_admissible_pair": True,
                        "search_range_expanded": False,
                        "truth_used_for_pair_enumeration": False,
                        "final_truth_landscape_used_for_pair_enumeration": False,
                    }
                )
    output = pd.DataFrame(rows)
    if output["pair_id"].duplicated().any():
        raise RuntimeError("designated local pair universe contains duplicate IDs")
    return output


def build_designated_local_validation_pair_plan(
    parameter_lattice: pd.DataFrame,
    *,
    pairs_per_location_class: int = PAIRS_PER_LOCATION_CLASS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash-select a balanced pilot plan before any prediction/truth outcome."""

    if pairs_per_location_class < 1:
        raise ValueError("pairs_per_location_class must be positive")
    universe = enumerate_designated_local_pair_universe(parameter_lattice)
    selected_frames: list[pd.DataFrame] = []
    strata_rows: list[dict[str, Any]] = []
    group_columns = ["coordinate", "trust_level", "trust_step", "location_class"]
    for keys, group in universe.groupby(group_columns, sort=True):
        ordered = group.sort_values("selection_hash", kind="mergesort")
        if len(ordered) < pairs_per_location_class:
            raise RuntimeError(f"pair universe stratum too small: {keys}")
        selected = ordered.head(pairs_per_location_class).copy()
        selected["within_stratum_hash_rank"] = np.arange(
            1, len(selected) + 1, dtype=int
        )
        selected_frames.append(selected)
        coordinate, trust_level, trust_step, location_class = keys
        strata_rows.append(
            {
                "coordinate": coordinate,
                "trust_level": trust_level,
                "trust_step": trust_step,
                "location_class": location_class,
                "universe_pair_count": int(len(group)),
                "planned_pair_count": int(len(selected)),
                "selection_rule": (
                    "LEXICOGRAPHIC_LOWEST_SHA256_WITHIN_GEOMETRY_TRUST_LOCATION_STRATUM"
                ),
                "selection_uses_prediction": False,
                "selection_uses_truth": False,
                "selection_uses_final_truth_landscape": False,
                "sample_count_status": SAMPLE_COUNT_STATUS,
            }
        )
    plan = pd.concat(selected_frames, ignore_index=True, sort=False)
    plan["predicted_delta_J"] = None
    plan["truth_delta_J"] = None
    plan["e_delta_J"] = None
    plan["outcome_status"] = PAIR_OUTCOME_STATUS
    plan["used_for_model_fitting"] = False
    plan["used_for_adaptation_update"] = False
    plan["heldout_final_test"] = False
    plan["used_by_P2_V1"] = False
    plan["formal_P2_V2_guard_input"] = False
    plan["threshold_frozen"] = False
    plan["plan_must_be_hashed_before_outcome_evaluation"] = True
    plan = plan.sort_values(
        ["coordinate", "trust_step", "location_class", "selection_hash"],
        kind="mergesort",
    ).reset_index(drop=True)
    return plan, pd.DataFrame(strata_rows)


def attach_designated_local_validation_outcomes(
    frozen_plan: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Validate a future outcome table without allowing pair reselection."""

    required = {"pair_id", "predicted_delta_J", "truth_delta_J"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"designated outcomes missing columns: {sorted(missing)}")
    if outcomes["pair_id"].duplicated().any():
        raise ValueError("designated outcomes contain duplicate pair IDs")
    planned_ids = set(frozen_plan["pair_id"].astype(str))
    outcome_ids = set(outcomes["pair_id"].astype(str))
    if outcome_ids != planned_ids:
        raise ValueError("outcome pair IDs must exactly match the frozen plan")
    selected = outcomes.loc[:, list(required)].copy()
    selected["predicted_delta_J"] = pd.to_numeric(
        selected["predicted_delta_J"], errors="raise"
    )
    selected["truth_delta_J"] = pd.to_numeric(
        selected["truth_delta_J"], errors="raise"
    )
    if not np.isfinite(
        selected[["predicted_delta_J", "truth_delta_J"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("designated outcomes must be finite")
    output = frozen_plan.drop(
        columns=["predicted_delta_J", "truth_delta_J", "e_delta_J", "outcome_status"]
    ).merge(selected, on="pair_id", how="left", validate="one_to_one")
    output["e_delta_J"] = np.abs(
        output["predicted_delta_J"] - output["truth_delta_J"]
    )
    output["outcome_status"] = "INDEPENDENT_DESIGNATED_OUTCOME_ATTACHED"
    output["pair_selection_changed_after_outcome"] = False
    output["truth_used_for_pair_selection"] = False
    output["formal_threshold_created"] = False
    return output


def designated_local_validation_protocol(
    *,
    pair_plan_sha256: str,
    planned_pair_count: int,
    universe_pair_count: int,
) -> dict[str, Any]:
    return {
        "protocol_id": LOCAL_PROTOCOL_ID,
        "status": DESIGN_STATUS,
        "candidate_neighborhood": {
            "space": "existing_geometry_valid_generator_parameter_lattice_only",
            "relationship": "one_coordinate_differs_by_one_existing_trust_step",
            "trust_levels": {
                name: list(values) for name, values in _TRUST_LEVELS.items()
            },
            "clipping_allowed": False,
            "bounds_expansion_allowed": False,
            "physical_distance_used": False,
        },
        "pair_generation": {
            "orientation": "canonical_low_coordinate_to_high_coordinate",
            "strata": "coordinate_x_trust_level_x_lower_interior_upper_location",
            "pairs_per_location_class": PAIRS_PER_LOCATION_CLASS,
            "selection": (
                "lowest_SHA256_within_each_stratum_using_alpha_and_protocol_only"
            ),
            "universe_pair_count": universe_pair_count,
            "planned_pair_count": planned_pair_count,
            "sample_count_status": SAMPLE_COUNT_STATUS,
            "pair_plan_sha256": pair_plan_sha256,
            "pair_plan_must_be_frozen_before_prediction_or_truth": True,
        },
        "outcome_schema": {
            "required": [
                "pair_id",
                "alpha_i",
                "alpha_j",
                "predicted_delta_J",
                "truth_delta_J",
                "e_delta_J=abs(predicted_delta_J-truth_delta_J)",
            ],
            "prediction_model_checkpoint_must_be_predeclared": True,
            "truth_source": (
                "new_independent_designated_offline_evaluation_after_plan_freeze"
            ),
            "existing_final_truth_landscape_allowed": False,
        },
        "data_separation": {
            "model_fitting": False,
            "adaptation_update": False,
            "heldout_final_test": False,
            "P2_V1_input": False,
            "formal_P2_V2_guard_input_before_review": False,
        },
        "uncertainty_statistics_to_report": ["max", "P95", "P99"],
        "uncertainty_threshold_frozen": False,
        "current_P2_modified": False,
    }


def compare_global_and_designated_local_validation(
    global_validation: pd.DataFrame,
    designated_plan: pd.DataFrame,
) -> pd.DataFrame:
    global_errors = global_validation["e_delta_J"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "validation_class": "CURRENT_GLOBAL_IDENTIFICATION_PAIR",
                "pair_instance_count": int(len(global_validation)),
                "alpha_pair_mappable": False,
                "coordinates_covered": "IDENTIFICATION_EXCITATION_NOT_ALPHA",
                "trust_levels_covered": "NONE",
                "selection_uses_final_truth_landscape": False,
                "outcomes_available": True,
                "mean_e_delta_J": float(np.mean(global_errors)),
                "P95_e_delta_J": float(np.percentile(global_errors, 95)),
                "P99_e_delta_J": float(np.percentile(global_errors, 99)),
                "max_e_delta_J": float(np.max(global_errors)),
                "role": "CURRENT_G0_PROVENANCE_NOT_LOCAL_DECISION_SCALE",
            },
            {
                "validation_class": "DESIGNATED_LOCAL_PAIR_PLAN",
                "pair_instance_count": int(len(designated_plan)),
                "alpha_pair_mappable": True,
                "coordinates_covered": "hip;knee;phase",
                "trust_levels_covered": "INITIAL;HALF;MINIMUM",
                "selection_uses_final_truth_landscape": False,
                "outcomes_available": False,
                "mean_e_delta_J": np.nan,
                "P95_e_delta_J": np.nan,
                "P99_e_delta_J": np.nan,
                "max_e_delta_J": np.nan,
                "role": "PRE_REGISTERED_PLAN_PENDING_INDEPENDENT_OUTCOMES",
            },
        ]
    )


def cumulative_decision_rule_protocol() -> dict[str, Any]:
    return {
        "rule_id": CUMULATIVE_RULE_ID,
        "status": DESIGN_STATUS,
        "Rule_A": {
            "name": "single_step_improvement",
            "existing_margin": "-delta_J_pred-U_step-0.005>0",
            "limitation": "cannot_accumulate_repeated_sub_0p005_same_direction_moves",
        },
        "Rule_B": {
            "name": "multi_step_accumulated_improvement_bundle",
            "candidate_formula": "-sum(delta_J_pred)-U_bundle-0.005>0",
            "maximum_accumulation_step_candidates": list(
                _ACCUMULATION_STEP_CANDIDATES
            ),
            "direction_consistency": [
                "same_generator_coordinate",
                "same_signed_direction",
                "each_step_is_an_existing_trust_neighbor",
                "predicted_step_delta_J_must_remain_improving",
                "predicted_endpoint_ranking_must_not_reverse",
            ],
            "path_constraints": [
                "every_intermediate_alpha_inside_existing_generator_bounds",
                "every_intermediate_trajectory_geometry_valid",
                "every_intermediate_candidate_model_supported",
                "model_checkpoint_fixed_within_a_shadow_bundle",
            ],
            "uncertainty_aggregation_candidates": list(
                _UNCERTAINTY_AGGREGATION_CANDIDATES
            ),
            "wrong_direction_prevention": [
                "predeclare_direction_from_prediction_before_truth",
                "reject_bundle_on_predicted_sign_or_rank_reversal",
                "reject_bundle_on_support_or_uncertainty_failure",
                "never_use_posthoc_truth_to_extend_or_reselect_bundle",
                "evaluate_truth_only_after_shadow_bundle_selection_is_frozen",
            ],
        },
        "maximum_steps_frozen": False,
        "uncertainty_aggregation_frozen": False,
        "numeric_threshold_created": False,
        "rule_enabled": False,
        "objective_modified": False,
        "generator_modified": False,
        "truth_used_to_modify_policy": False,
        "current_P2_modified": False,
    }


def build_cumulative_rule_candidate_matrix(
    knee_cumulative_history: pd.DataFrame,
) -> pd.DataFrame:
    history = knee_cumulative_history.set_index("step_index")
    rows: list[dict[str, Any]] = []
    for maximum_steps in _ACCUMULATION_STEP_CANDIDATES:
        if maximum_steps not in history.index:
            raise ValueError(f"knee cumulative evidence lacks step {maximum_steps}")
        observed = history.loc[maximum_steps]
        for aggregation in _UNCERTAINTY_AGGREGATION_CANDIDATES:
            rows.append(
                {
                    "rule_id": CUMULATIVE_RULE_ID,
                    "maximum_accumulation_steps_candidate": maximum_steps,
                    "uncertainty_aggregation_candidate": aggregation,
                    "direction_consistency_required": True,
                    "same_coordinate_required": True,
                    "same_sign_required": True,
                    "intermediate_geometry_and_support_required": True,
                    "posthoc_knee_stiff_cumulative_improvement": float(
                        observed["cumulative_improvement_magnitude"]
                    ),
                    "posthoc_exceeds_existing_0p005": bool(
                        observed["cumulative_exceeds_existing_0p005"]
                    ),
                    "truth_role": "POST_HOC_RULE_DESIGN_EVIDENCE_ONLY",
                    "truth_used_by_policy": False,
                    "candidate_enabled": False,
                    "maximum_steps_frozen": False,
                    "uncertainty_aggregation_frozen": False,
                    "objective_modified": False,
                    "current_P2_modified": False,
                }
            )
    return pd.DataFrame(rows)


def compare_single_and_cumulative_rules(
    knee_cumulative_history: pd.DataFrame,
) -> pd.DataFrame:
    non_reference = knee_cumulative_history.loc[
        knee_cumulative_history["step_index"].gt(0)
    ]
    return pd.DataFrame(
        [
            {
                "rule": "RULE_A_SINGLE_STEP",
                "evaluated_move_count": int(len(non_reference)),
                "maximum_observed_improvement": float(
                    non_reference["single_step_improvement_magnitude"].max()
                ),
                "moves_exceeding_existing_0p005": int(
                    non_reference["single_step_exceeds_existing_0p005"].sum()
                ),
                "addresses_stepwise_problem": False,
                "policy_enabled": False,
            },
            {
                "rule": "RULE_B_MULTI_STEP_CANDIDATE",
                "evaluated_move_count": int(len(_ACCUMULATION_STEP_CANDIDATES)),
                "maximum_observed_improvement": float(
                    non_reference["cumulative_improvement_magnitude"].max()
                ),
                "moves_exceeding_existing_0p005": int(
                    non_reference.loc[
                        non_reference["step_index"].isin(
                            _ACCUMULATION_STEP_CANDIDATES
                        ),
                        "cumulative_exceeds_existing_0p005",
                    ].sum()
                ),
                "addresses_stepwise_problem": True,
                "policy_enabled": False,
            },
        ]
    )


def decision_value_exploration_stopping_protocol() -> dict[str, Any]:
    return {
        "rule_id": STOPPING_RULE_ID,
        "status": DESIGN_STATUS,
        "per_explore_record": {
            "SUPPORT_VALUE": "coverage_increase",
            "MODEL_VALUE": "parameter_change",
            "PREDICTION_VALUE": "prediction_map_change",
            "DECISION_VALUE": [
                "predicted_ranking_change",
                "best_trajectory_change_under_existing_0p005",
                "exploit_eligibility_change_as_supplemental_signal",
            ],
        },
        "continue_candidate": [
            "continue_if_DECISION_VALUE",
            "continue_shadow_observation_if_MODEL_VALUE_or_PREDICTION_VALUE",
            "support_growth_alone_is_not_a_continue_reason",
        ],
        "stop_candidate": {
            "condition": (
                "K_consecutive_explores_with_zero_MODEL_PREDICTION_DECISION_value_"
                "and_no_pending_exploit"
            ),
            "consecutive_K_candidates": list(_STOP_CONSECUTIVE_CANDIDATES),
            "requires_valid_observation_and_no_failure": True,
        },
        "change_detection_numeric_tolerance": None,
        "change_detection_tolerance_requires_review": True,
        "consecutive_K_frozen": False,
        "automatic_stop_enabled": False,
        "support_used_as_decision_value": False,
        "truth_feature_used": False,
        "current_P2_modified": False,
    }


def _first_consecutive_zero_value_iteration(
    case_history: pd.DataFrame,
    required: int,
) -> int | None:
    run = 0
    previous: int | None = None
    for row in case_history.sort_values("iteration").itertuples(index=False):
        iteration = int(row.iteration)
        zero_value = bool(
            not row.parameter_changed_exactly
            and not row.prediction_map_changed_exactly
            and not row.predicted_local_ranking_changed
            and not row.predicted_global_ranking_changed
            and not row.best_trajectory_changed
            and not row.exploit_eligibility_changed
        )
        if zero_value:
            run = run + 1 if previous is not None and iteration == previous + 1 else 1
        else:
            run = 0
        if run >= required:
            return iteration
        previous = iteration
    return None


def build_decision_value_stopping_shadow(
    exploration_history: pd.DataFrame,
    executed_history: pd.DataFrame,
    natural_stopping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p2_counts = natural_stopping.loc[
        natural_stopping["policy_id"].eq("P2_DECISION_GUARDED_EXPLORE_EXPLOIT")
    ].set_index("case_id")
    rows: list[dict[str, Any]] = []
    for case_id, case in exploration_history.groupby("case_id", sort=True):
        current_count = int(p2_counts.loc[case_id, "executed_trial_count"])
        executed = executed_history.loc[
            executed_history["case_id"].astype(str).eq(str(case_id))
        ]
        for required in _STOP_CONSECUTIVE_CANDIDATES:
            trigger = _first_consecutive_zero_value_iteration(case, required)
            later = (
                executed.iloc[0:0]
                if trigger is None
                else executed.loc[executed["iteration"].astype(int).gt(trigger)]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "consecutive_zero_value_candidate": required,
                    "trigger_observed": trigger is not None,
                    "shadow_stop_after_iteration": trigger,
                    "current_executed_trial_count": current_count,
                    "shadow_executed_trial_count": (
                        current_count if trigger is None else int(trigger)
                    ),
                    "historical_trials_potentially_avoided": (
                        0 if trigger is None else max(current_count - int(trigger), 0)
                    ),
                    "later_exploit_trials_in_frozen_history": int(
                        later["trial_purpose"].eq("EXPLOIT").sum()
                    ),
                    "later_accepted_best_changes_in_frozen_history": int(
                        later["accepted_improvement"].astype(bool).sum()
                    ),
                    "support_alone_causes_continue": False,
                    "automatic_stop_executed": False,
                    "candidate_frozen": False,
                    "truth_feature_used": False,
                    "current_P2_modified": False,
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("consecutive_zero_value_candidate", as_index=False)
        .agg(
            evaluated_case_count=("case_id", "count"),
            trigger_case_count=("trigger_observed", "sum"),
            historical_trials_potentially_avoided=(
                "historical_trials_potentially_avoided",
                "sum",
            ),
            later_exploit_trials_in_frozen_history=(
                "later_exploit_trials_in_frozen_history",
                "sum",
            ),
            later_accepted_best_changes_in_frozen_history=(
                "later_accepted_best_changes_in_frozen_history",
                "sum",
            ),
        )
    )
    summary["automatic_stop_executed"] = False
    summary["candidate_frozen"] = False
    summary["truth_feature_used"] = False
    summary["current_P2_modified"] = False
    return detail, summary


def minimum_p2_v2_revision_set() -> list[str]:
    return [
        "pre_registered_designated_local_pair_plan_with_immutable_hash_and_independent_outcomes",
        "reviewed_local_uncertainty_provider_defaulting_to_P2_V1_until_approved",
        "default_off_cumulative_bundle_evaluator_with_direction_path_and_uncertainty_guards",
        "separate_support_model_prediction_decision_value_history",
        "default_off_shadow_stopping_candidate_with_no_support_only_continue",
        "preserve_reference_ROM_model_objective_generator_0p005_and_90percent_gate",
    ]


__all__ = [
    "CUMULATIVE_RULE_ID",
    "DESIGN_STATUS",
    "FORMAL_DESIGN_ID",
    "LOCAL_PROTOCOL_ID",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_MOTION_APPROVED",
    "OFFLINE_METHOD_STATUS",
    "PAIR_OUTCOME_STATUS",
    "PAIRS_PER_LOCATION_CLASS",
    "SAMPLE_COUNT_STATUS",
    "STOPPING_RULE_ID",
    "attach_designated_local_validation_outcomes",
    "build_cumulative_rule_candidate_matrix",
    "build_decision_value_stopping_shadow",
    "build_designated_local_validation_pair_plan",
    "compare_global_and_designated_local_validation",
    "compare_single_and_cumulative_rules",
    "cumulative_decision_rule_protocol",
    "decision_value_exploration_stopping_protocol",
    "designated_local_validation_protocol",
    "enumerate_designated_local_pair_universe",
    "minimum_p2_v2_revision_set",
]
