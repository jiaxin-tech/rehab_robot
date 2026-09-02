"""Pure post-hoc analyses for the frozen BUNDLE_5 development shadow.

This module does not define, select, or execute a new policy.  It only turns
already-frozen development trajectories plus post-policy truth landscapes into
auditable boundary, subject-specificity, and trial-value tables.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
)
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
)
from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE


AUDIT_ID = "P2_BUNDLE5_BOUNDARY_AND_SUBJECT_SPECIFICITY_AUDIT_V1"
MANIFEST_ID = "BUNDLE5_AUDIT_MANIFEST_V1"
FINAL_IDENTIFIED = "BUNDLE5_ROOT_CAUSE_IDENTIFIED"
FINAL_MORE_EVIDENCE = "MORE_BOUNDARY_EVIDENCE_REQUIRED"
BOUNDARY_TRUTH = "TRUTH_LANDSCAPE_CONCENTRATION"
BOUNDARY_POLICY = "POLICY_INDUCED_COLLAPSE"
BOUNDARY_MIXED = "MIXED_TRUTH_AND_POLICY_EFFECT"
OBJECTIVE_REVIEW = "OBJECTIVE_SUBJECT_DISCRIMINATION_REQUIRES_REVIEW"
OBJECTIVE_NO_CHANGE = "OBJECTIVE_CHANGE_NOT_JUSTIFIED"
TRIAL_USEFUL = "TRIAL_COST_DOMINATED_BY_USEFUL_IMPROVEMENT"
TRIAL_LOW_TAIL = "TRIAL_COST_CONTAINS_SUBSTANTIAL_LOW_VALUE_TAIL"
TRIAL_MIXED = "MIXED_TRIAL_VALUE"
OFFLINE_ONLY = "OFFLINE_ONLY"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"

ADAPTIVE_MANIFEST_SHA256 = (
    "ace9586f98bfc5142ee310539e6c42b02d7164a3fbb91fdad8d7352110c96f9b"
)
MODULE_DIR = Path(__file__).resolve().parent
ADAPTIVE_ARTIFACT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_adaptive_horizon_decision_prototype_v1"
)
ADAPTIVE_MANIFEST_PATH = ADAPTIVE_ARTIFACT_DIRECTORY / "MANIFEST.json"
MULTI_STEP_ARTIFACT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_multi_step_decision_framework_analysis_v1"
)

_AXIS_COLUMN = {"HIP": "hip_delta", "KNEE": "knee_delta", "PHASE": "phase_delta"}
_AXIS_STEP = {
    "HIP": GRID_HIP_STEP_DEG,
    "KNEE": GRID_KNEE_STEP_DEG,
    "PHASE": GRID_PHASE_STEP,
}
_AXIS_BOUND = {
    "HIP": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["hip_amplitude_delta_deg"],
    "KNEE": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_amplitude_delta_deg"],
    "PHASE": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_phase_shift"],
}


def manifest_payload(
    *, checkpoint_commit: str, protected_source_sha256: Mapping[str, str]
) -> dict[str, Any]:
    """Freeze diagnostic definitions before the deterministic H2 replay."""

    return {
        "manifest_id": MANIFEST_ID,
        "audit_id": AUDIT_ID,
        "status": "FROZEN_BEFORE_POSTHOC_H2_REPLAY",
        "checkpoint_commit": str(checkpoint_commit),
        "adaptive_manifest_sha256": ADAPTIVE_MANIFEST_SHA256,
        "policy_under_audit": "H2_FIXED_BUNDLE_5",
        "new_policy_designed": False,
        "replay_contract": {
            "trajectory_sequence_source": (
                "frozen p2_multi_step_decision_framework_analysis_v1/"
                "framework_trial_history.csv BUNDLE_5 rows"
            ),
            "trajectory_selection_recomputed_or_changed": False,
            "every_replayed_row_must_match_frozen_alpha_purpose_J_and_acceptance": True,
            "posthoc_diagnostics_added": [
                "parameter_change",
                "prediction_map_change",
                "support_change",
                "future_exploit_eligibility_change",
                "candidate_truth_after_policy_decision",
            ],
            "truth_used_for_policy_authorization": False,
        },
        "truth_landscape_scope": {
            "global": "existing_frozen_geometrically_admissible_parameter_lattice",
            "reference_centered_local": (
                "reference plus six existing one-grid-unit single-axis neighbors "
                "that are present in the frozen lattice"
            ),
            "distance_invented": False,
            "truth_is_posthoc_only": True,
        },
        "boundary_collapse_classification": {
            "TRUTH_LANDSCAPE_CONCENTRATION": (
                "one unique full truth optimum and H2 equals that optimum"
            ),
            "POLICY_INDUCED_COLLAPSE": (
                "truth full optima differ, H2 is uniform, and no truth-optimum "
                "coordinate is uniformly equal to the corresponding H2 coordinate"
            ),
            "MIXED_TRUTH_AND_POLICY_EFFECT": (
                "truth full optima differ, H2 is uniform, and at least one "
                "truth-optimum coordinate is uniformly equal to H2 while another "
                "truth-optimum coordinate remains diverse"
            ),
            "results_may_change_rules": False,
        },
        "objective_review_rule": {
            "requires_both": [
                "modal_full_truth_optimum_count >= 12_of_15",
                "truth_axis_profile_best_direction_is_KNEE_NEGATIVE_BOUNDARY >= 12_of_15",
            ],
            "otherwise": OBJECTIVE_NO_CHANGE,
            "threshold_role": "POSTHOC_REVIEW_TRIAGE_ONLY_NOT_A_POLICY_GATE",
            "objective_may_be_changed_in_this_task": False,
        },
        "trial_value_definition": {
            "DIRECT_DECISION_VALUE": (
                "actual best J or best alpha changes, or future exploit eligibility changes"
            ),
            "MODEL_INFORMATION_VALUE": (
                "parameters or full prediction map change exactly, without direct or support value"
            ),
            "SUPPORT_ONLY_VALUE": (
                "supported-point count increases without direct or model value"
            ),
            "MULTIPLE_VALUES": "two or more of decision/model/support value are present",
            "POST_OPTIMUM_LOW_VALUE": (
                "strictly after first final-alpha arrival with no best-J/alpha or "
                "future-exploit-eligibility change; model/support growth does not "
                "by itself become decision value"
            ),
            "BOUNDARY_CHASING_LOW_VALUE": (
                "an exploit is executed after best knee is already -5 and it has "
                "no best-J/alpha or future-eligibility change"
            ),
            "all_change_checks": "exact research numerical diagnostics; no new threshold",
        },
        "trial_cost_classification": {
            TRIAL_LOW_TAIL: "low-value classifications >= 50_percent_of_trials",
            TRIAL_USEFUL: "useful classifications >= 75_percent_of_trials",
            TRIAL_MIXED: "otherwise",
            "role": "POSTHOC_DESCRIPTION_ONLY_NOT_A_STOPPING_RULE",
        },
        "data_roles": {
            "development": "ORIGINAL_9_PLUS_POST_REJECTION_DEVELOPMENT_6",
            "independent_calibration": "EXISTING_UNCERTAINTY_ONLY_NOT_OUTCOME_EVIDENCE",
            "prospective": "NOT_GENERATED",
            "heldout_final_test": "NOT_READ",
        },
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "generator_bounds": {
            key: list(value)
            for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
        },
        "protected_source_sha256": dict(protected_source_sha256),
        "default_enabled": False,
        "P2_V1_modified": False,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
    }


def _alpha_key(row: Mapping[str, Any], *, prefix: str = "") -> tuple[float, float, float]:
    return tuple(
        round(float(row[f"{prefix}{name}"]), 12)
        for name in ("hip", "knee", "phase")
    )


def build_truth_optimum_by_case(
    truth_landscapes: Mapping[str, pd.DataFrame],
    h2_case_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_lookup = h2_case_summary.set_index("case_id")
    for case_id in sorted(truth_landscapes):
        truth = truth_landscapes[case_id].copy()
        global_best = truth.sort_values(
            ["J_truth", "trajectory_id"], kind="mergesort"
        ).iloc[0]
        local_mask = (
            (
                truth[["hip_delta", "knee_delta", "phase_delta"]]
                .abs()
                .gt(1e-12)
                .sum(axis=1)
                <= 1
            )
            & truth["hip_delta"].abs().le(GRID_HIP_STEP_DEG + 1e-12)
            & truth["knee_delta"].abs().le(GRID_KNEE_STEP_DEG + 1e-12)
            & truth["phase_delta"].abs().le(GRID_PHASE_STEP + 1e-12)
        )
        local = truth.loc[local_mask]
        if local.empty:
            raise RuntimeError(f"reference-centered local truth set empty: {case_id}")
        local_best = local.sort_values(
            ["J_truth", "trajectory_id"], kind="mergesort"
        ).iloc[0]
        frozen = summary_lookup.loc[case_id]
        final_key = (
            float(frozen["final_best_alpha_hip"]),
            float(frozen["final_best_alpha_knee"]),
            float(frozen["final_best_alpha_phase"]),
        )
        final = truth.loc[
            np.isclose(truth["hip_delta"], final_key[0], atol=1e-12, rtol=0.0)
            & np.isclose(truth["knee_delta"], final_key[1], atol=1e-12, rtol=0.0)
            & np.isclose(truth["phase_delta"], final_key[2], atol=1e-12, rtol=0.0)
        ]
        if len(final) != 1:
            raise RuntimeError(f"H2 final alpha absent from truth landscape: {case_id}")
        final = final.iloc[0]
        global_key = (
            float(global_best["hip_delta"]),
            float(global_best["knee_delta"]),
            float(global_best["phase_delta"]),
        )
        if not (
            np.isclose(global_key[0], frozen["truth_optimum_alpha_hip"], atol=1e-12)
            and np.isclose(global_key[1], frozen["truth_optimum_alpha_knee"], atol=1e-12)
            and np.isclose(global_key[2], frozen["truth_optimum_alpha_phase"], atol=1e-12)
            and np.isclose(
                float(global_best["J_truth"]),
                float(frozen["truth_optimum_J"]),
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise RuntimeError(f"recomputed truth optimum differs from checkpoint: {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "subject_id": frozen["subject_id"],
                "scenario_name": frozen["scenario_name"],
                "truth_global_trajectory_id": global_best["trajectory_id"],
                "truth_global_alpha_hip": global_key[0],
                "truth_global_alpha_knee": global_key[1],
                "truth_global_alpha_phase": global_key[2],
                "truth_global_J": float(global_best["J_truth"]),
                "truth_local_trajectory_id": local_best["trajectory_id"],
                "truth_local_alpha_hip": float(local_best["hip_delta"]),
                "truth_local_alpha_knee": float(local_best["knee_delta"]),
                "truth_local_alpha_phase": float(local_best["phase_delta"]),
                "truth_local_J": float(local_best["J_truth"]),
                "truth_local_candidate_count": len(local),
                "H2_final_alpha_hip": final_key[0],
                "H2_final_alpha_knee": final_key[1],
                "H2_final_alpha_phase": final_key[2],
                "J_truth_at_H2_final": float(final["J_truth"]),
                "H2_minus_truth_hip": final_key[0] - global_key[0],
                "H2_minus_truth_knee": final_key[1] - global_key[1],
                "H2_minus_truth_phase": final_key[2] - global_key[2],
                "global_regret": float(final["J_truth"] - global_best["J_truth"]),
                "H2_full_alpha_matches_truth_global": bool(
                    np.allclose(final_key, global_key, atol=1e-12, rtol=0.0)
                ),
                "truth_hip_at_boundary": bool(
                    np.isclose(global_key[0], _AXIS_BOUND["HIP"][0], atol=1e-12)
                    or np.isclose(global_key[0], _AXIS_BOUND["HIP"][1], atol=1e-12)
                ),
                "truth_knee_at_boundary": bool(
                    np.isclose(global_key[1], _AXIS_BOUND["KNEE"][0], atol=1e-12)
                    or np.isclose(global_key[1], _AXIS_BOUND["KNEE"][1], atol=1e-12)
                ),
                "truth_phase_at_boundary": bool(
                    np.isclose(global_key[2], _AXIS_BOUND["PHASE"][0], atol=1e-12)
                    or np.isclose(global_key[2], _AXIS_BOUND["PHASE"][1], atol=1e-12)
                ),
                "truth_used_for_policy_selection": False,
                "truth_role": "POSTHOC_AUDIT_ONLY",
            }
        )
    return pd.DataFrame(rows)


def build_subject_discrimination(optimum: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(optimum)), 2):
        left = optimum.iloc[left_index]
        right = optimum.iloc[right_index]
        truth_left = _alpha_key(left, prefix="truth_global_alpha_")
        truth_right = _alpha_key(right, prefix="truth_global_alpha_")
        h2_left = _alpha_key(left, prefix="H2_final_alpha_")
        h2_right = _alpha_key(right, prefix="H2_final_alpha_")
        truth_differs = truth_left != truth_right
        h2_differs = h2_left != h2_right
        if truth_differs and not h2_differs:
            label = "SUBJECT_DIFFERENCE_COLLAPSED_BY_POLICY"
        elif not truth_differs and not h2_differs:
            label = "NO_TRUTH_DISCRIMINATION_TO_PRESERVE"
        elif truth_differs and h2_differs:
            label = "TRUTH_DIFFERENCE_PRESERVED_BY_POLICY"
        else:
            label = "POLICY_DIFFERENCE_WITHOUT_TRUTH_OPTIMUM_DIFFERENCE"
        rows.append(
            {
                "case_id_A": left["case_id"],
                "case_id_B": right["case_id"],
                "truth_optimum_full_alpha_differs": truth_differs,
                "H2_final_full_alpha_differs": h2_differs,
                "truth_optimum_J_absolute_difference": abs(
                    float(left["truth_global_J"]) - float(right["truth_global_J"])
                ),
                "pair_classification": label,
                "truth_used_for_policy_selection": False,
            }
        )
    return pd.DataFrame(rows)


def build_truth_axis_profiles(
    truth_landscapes: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id in sorted(truth_landscapes):
        truth = truth_landscapes[case_id]
        reference = truth.loc[
            np.isclose(truth["hip_delta"], 0.0)
            & np.isclose(truth["knee_delta"], 0.0)
            & np.isclose(truth["phase_delta"], 0.0)
        ]
        if len(reference) != 1:
            raise RuntimeError(f"truth profile reference missing: {case_id}")
        reference_j = float(reference.iloc[0]["J_truth"])
        for axis, column in _AXIS_COLUMN.items():
            other = [name for name in _AXIS_COLUMN.values() if name != column]
            profile = truth.loc[
                np.isclose(truth[other[0]], 0.0, atol=1e-12, rtol=0.0)
                & np.isclose(truth[other[1]], 0.0, atol=1e-12, rtol=0.0)
            ].sort_values(column)
            for direction, sign in (("NEGATIVE", -1.0), ("POSITIVE", 1.0)):
                directional = profile.loc[profile[column] * sign > 1e-12].copy()
                directional["distance_steps"] = (
                    directional[column].abs() / _AXIS_STEP[axis]
                )
                directional = directional.sort_values("distance_steps")
                if directional.empty:
                    continue
                near = directional.iloc[0]
                best = directional.sort_values(
                    ["J_truth", "trajectory_id"], kind="mergesort"
                ).iloc[0]
                values_from_reference = np.concatenate(
                    ([reference_j], directional["J_truth"].to_numpy(dtype=float))
                )
                monotonic = bool(np.all(np.diff(values_from_reference) < 0.0))
                best_alpha = float(best[column])
                lower, upper = _AXIS_BOUND[axis]
                best_at_boundary = bool(
                    np.isclose(best_alpha, lower, atol=1e-12, rtol=0.0)
                    or np.isclose(best_alpha, upper, atol=1e-12, rtol=0.0)
                )
                for item in directional.to_dict(orient="records"):
                    rows.append(
                        {
                            "case_id": case_id,
                            "axis": axis,
                            "direction": direction,
                            "alpha_hip": item["hip_delta"],
                            "alpha_knee": item["knee_delta"],
                            "alpha_phase": item["phase_delta"],
                            "axis_alpha": item[column],
                            "distance_in_existing_grid_steps": item[
                                "distance_steps"
                            ],
                            "J_truth": item["J_truth"],
                            "delta_J_truth_vs_reference": item["J_truth"]
                            - reference_j,
                            "model_supported_at_initial_state": item[
                                "model_supported"
                            ],
                            "domain_coverage_at_initial_state": item[
                                "domain_coverage"
                            ],
                            "reference_near_local_delta_J": float(
                                near["J_truth"] - reference_j
                            ),
                            "best_axis_direction_alpha": best_alpha,
                            "best_axis_direction_J": float(best["J_truth"]),
                            "best_axis_direction_improvement": float(
                                reference_j - best["J_truth"]
                            ),
                            "best_axis_direction_at_generator_boundary": (
                                best_at_boundary
                            ),
                            "truth_monotonic_improvement_from_reference_to_boundary": (
                                monotonic
                            ),
                            "truth_used_for_policy_selection": False,
                        }
                    )
    return pd.DataFrame(rows)


def classify_boundary_collapse(optimum: pd.DataFrame) -> str:
    truth_columns = [
        "truth_global_alpha_hip",
        "truth_global_alpha_knee",
        "truth_global_alpha_phase",
    ]
    h2_columns = ["H2_final_alpha_hip", "H2_final_alpha_knee", "H2_final_alpha_phase"]
    truth_unique = len(optimum[truth_columns].drop_duplicates())
    h2_unique = len(optimum[h2_columns].drop_duplicates())
    if truth_unique == 1 and h2_unique == 1:
        truth_key = tuple(optimum[truth_columns].iloc[0])
        h2_key = tuple(optimum[h2_columns].iloc[0])
        return BOUNDARY_TRUTH if truth_key == h2_key else BOUNDARY_POLICY
    if truth_unique > 1 and h2_unique == 1:
        h2 = optimum[h2_columns].iloc[0].to_numpy(dtype=float)
        uniform_matching_coordinates = 0
        diverse_coordinates = 0
        for index, truth_column in enumerate(truth_columns):
            values = optimum[truth_column].to_numpy(dtype=float)
            if len(np.unique(np.round(values, 12))) > 1:
                diverse_coordinates += 1
            elif np.isclose(values[0], h2[index], atol=1e-12, rtol=0.0):
                uniform_matching_coordinates += 1
        if uniform_matching_coordinates and diverse_coordinates:
            return BOUNDARY_MIXED
        return BOUNDARY_POLICY
    return BOUNDARY_MIXED


def classify_objective_status(
    optimum: pd.DataFrame, axis_profiles: pd.DataFrame
) -> str:
    keys = optimum[
        [
            "truth_global_alpha_hip",
            "truth_global_alpha_knee",
            "truth_global_alpha_phase",
        ]
    ].astype(str).agg("|".join, axis=1)
    modal_count = int(keys.value_counts().max())
    knee_negative_boundary = axis_profiles.loc[
        axis_profiles["axis"].eq("KNEE")
        & axis_profiles["direction"].eq("NEGATIVE")
    ].groupby("case_id")["best_axis_direction_at_generator_boundary"].first()
    knee_boundary_count = int(knee_negative_boundary.sum())
    return (
        OBJECTIVE_REVIEW
        if modal_count >= 12 and knee_boundary_count >= 12
        else OBJECTIVE_NO_CHANGE
    )


def classify_trial_values(
    diagnostics: pd.DataFrame, h2_case_summary: pd.DataFrame
) -> pd.DataFrame:
    output = diagnostics.copy()
    final_lookup = h2_case_summary.set_index("case_id")
    classifications: list[str] = []
    first_final: dict[str, int] = {}
    for case_id, group in output.groupby("case_id", sort=False):
        frozen = final_lookup.loc[case_id]
        final_key = np.asarray(
            [
                frozen["final_best_alpha_hip"],
                frozen["final_best_alpha_knee"],
                frozen["final_best_alpha_phase"],
            ],
            dtype=float,
        )
        after = group[
            ["best_alpha_hip_after", "best_alpha_knee_after", "best_alpha_phase_after"]
        ].to_numpy(dtype=float)
        matches = np.all(np.isclose(after, final_key, atol=1e-12, rtol=0.0), axis=1)
        first_final[case_id] = int(group.loc[matches, "iteration"].min())
    for row in output.to_dict(orient="records"):
        direct = bool(
            row["actual_best_J_improvement"] > 0.0
            or row["changed_best_alpha"]
            or row["changed_future_exploit_eligibility"]
        )
        model = bool(
            row["parameter_changed_exactly"]
            or row["prediction_map_changed_exactly"]
        )
        support = bool(row["support_point_increase"] > 0)
        after_final = int(row["iteration"]) > first_final[str(row["case_id"])]
        boundary_before = bool(
            np.isclose(
                float(row["best_alpha_knee_before"]),
                _AXIS_BOUND["KNEE"][0],
                atol=1e-12,
                rtol=0.0,
            )
        )
        if not direct and after_final:
            classification = "POST_OPTIMUM_LOW_VALUE"
        elif (
            not direct
            and boundary_before
            and str(row["trial_purpose"]) == "EXPLOIT"
        ):
            classification = "BOUNDARY_CHASING_LOW_VALUE"
        elif sum((direct, model, support)) >= 2:
            classification = "MULTIPLE_VALUES"
        elif direct:
            classification = "DIRECT_DECISION_VALUE"
        elif support:
            classification = "SUPPORT_ONLY_VALUE"
        elif model:
            classification = "MODEL_INFORMATION_VALUE"
        else:
            classification = "POST_OPTIMUM_LOW_VALUE"
        classifications.append(classification)
    output["trial_value_classification"] = classifications
    output["classification_is_posthoc_only"] = True
    output["classification_used_to_stop_policy"] = False
    return output


def build_boundary_timeline(
    trial_values: pd.DataFrame, h2_case_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary = h2_case_summary.set_index("case_id")
    for case_id, group in trial_values.groupby("case_id", sort=True):
        ordered = group.sort_values("iteration")
        frozen = summary.loc[case_id]
        final = np.asarray(
            [
                frozen["final_best_alpha_hip"],
                frozen["final_best_alpha_knee"],
                frozen["final_best_alpha_phase"],
            ],
            dtype=float,
        )
        after = ordered[
            ["best_alpha_hip_after", "best_alpha_knee_after", "best_alpha_phase_after"]
        ].to_numpy(dtype=float)
        final_mask = np.all(np.isclose(after, final, atol=1e-12, rtol=0.0), axis=1)
        knee_mask = np.isclose(
            ordered["best_alpha_knee_after"],
            _AXIS_BOUND["KNEE"][0],
            atol=1e-12,
            rtol=0.0,
        )
        first_final = int(ordered.loc[final_mask, "iteration"].min())
        first_knee = int(ordered.loc[knee_mask, "iteration"].min())
        after_final = ordered.loc[ordered["iteration"].gt(first_final)]
        after_knee = ordered.loc[ordered["iteration"].gt(first_knee)]
        rows.append(
            {
                "case_id": case_id,
                "subject_id": frozen["subject_id"],
                "scenario_name": frozen["scenario_name"],
                "first_trial_reaching_knee_minus_5": first_knee,
                "first_trial_reaching_final_alpha": first_final,
                "total_trials": len(ordered),
                "trials_after_first_final_alpha": len(after_final),
                "accepted_actions_after_first_final_alpha": int(
                    after_final["accepted_meaningful_improvement"].sum()
                ),
                "executed_actions_after_first_final_alpha": len(after_final),
                "explore_actions_after_knee_minus_5": int(
                    after_knee["trial_purpose"].eq("EXPLORE").sum()
                ),
                "endpoint_actions_after_knee_minus_5": int(
                    after_knee["trial_purpose"].eq("EXPLOIT").sum()
                ),
                "model_updates_after_knee_minus_5": int(
                    after_knee["model_refit_after_execution"].sum()
                ),
                "best_J_changed_after_knee_minus_5": bool(
                    after_knee["actual_best_J_improvement"].gt(0.0).any()
                ),
                "best_alpha_changed_after_knee_minus_5": bool(
                    after_knee["changed_best_alpha"].any()
                ),
                "prediction_map_changed_exactly_after_knee_minus_5": bool(
                    after_knee["prediction_map_change_max_abs"].gt(0.0).any()
                ),
                "maximum_map_change_after_knee_minus_5": (
                    float(after_knee["prediction_map_change_max_abs"].max())
                    if not after_knee.empty
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_best_j_progression(trial_values: pd.DataFrame) -> pd.DataFrame:
    output = trial_values[
        [
            "case_id",
            "subject_id",
            "scenario_name",
            "iteration",
            "trial_purpose",
            "best_actual_J_after",
            "best_executed_J_pred_so_far",
            "map_best_supported_J_pred_after",
            "best_alpha_hip_after",
            "best_alpha_knee_after",
            "best_alpha_phase_after",
            "actual_best_J_improvement",
            "trial_value_classification",
        ]
    ].copy()
    output = output.rename(
        columns={
            "iteration": "executed_trial_index",
            "best_actual_J_after": "best_actual_J_so_far",
        }
    )
    return output


def build_gain_timing(
    progression: pd.DataFrame,
    h1_summary: pd.DataFrame,
    h2_summary: pd.DataFrame,
) -> pd.DataFrame:
    h1 = h1_summary.set_index("case_id")
    h2 = h2_summary.set_index("case_id")
    rows: list[dict[str, Any]] = []
    for case_id, group in progression.groupby("case_id", sort=True):
        ordered = group.sort_values("executed_trial_index")
        h1_final = float(h1.loc[case_id, "final_best_actual_J"])
        h2_final = float(h2.loc[case_id, "final_best_actual_J"])
        total_gain = h1_final - h2_final
        fractions = (
            (h1_final - ordered["best_actual_J_so_far"]) / total_gain
            if total_gain > 0.0
            else pd.Series(np.zeros(len(ordered)), index=ordered.index)
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "H1_final_J": h1_final,
            "H2_final_J": h2_final,
            "H2_gain_vs_H1": total_gain,
            "total_H2_trials": len(ordered),
        }
        for target in (0.50, 0.80, 0.90, 0.95):
            reached = ordered.loc[fractions.ge(target)]
            row[f"trial_reaching_{int(target * 100)}pct_final_gain"] = (
                int(reached.iloc[0]["executed_trial_index"])
                if not reached.empty
                else np.nan
            )
        for quarter in (0.25, 0.50, 0.75, 1.00):
            index = max(int(np.ceil(len(ordered) * quarter)) - 1, 0)
            row[f"gain_fraction_at_{int(quarter * 100)}pct_trials"] = float(
                fractions.iloc[index]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_trial_cost(trial_values: pd.DataFrame) -> str:
    low = trial_values["trial_value_classification"].isin(
        ("POST_OPTIMUM_LOW_VALUE", "BOUNDARY_CHASING_LOW_VALUE")
    )
    useful = ~low
    if float(low.mean()) >= 0.50:
        return TRIAL_LOW_TAIL
    if float(useful.mean()) >= 0.75:
        return TRIAL_USEFUL
    return TRIAL_MIXED


__all__ = [
    "ADAPTIVE_MANIFEST_PATH",
    "ADAPTIVE_MANIFEST_SHA256",
    "AUDIT_ID",
    "BOUNDARY_MIXED",
    "BOUNDARY_POLICY",
    "BOUNDARY_TRUTH",
    "FINAL_IDENTIFIED",
    "FINAL_MORE_EVIDENCE",
    "MANIFEST_ID",
    "MULTI_STEP_ARTIFACT_DIRECTORY",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_APPROVED",
    "OBJECTIVE_NO_CHANGE",
    "OBJECTIVE_REVIEW",
    "OFFLINE_ONLY",
    "TRIAL_LOW_TAIL",
    "TRIAL_MIXED",
    "TRIAL_USEFUL",
    "build_best_j_progression",
    "build_boundary_timeline",
    "build_gain_timing",
    "build_subject_discrimination",
    "build_truth_axis_profiles",
    "build_truth_optimum_by_case",
    "classify_boundary_collapse",
    "classify_objective_status",
    "classify_trial_cost",
    "classify_trial_values",
    "manifest_payload",
]
