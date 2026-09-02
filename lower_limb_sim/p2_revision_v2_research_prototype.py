"""Default-off, offline-only P2 Revision V2 research prototype.

The module implements shadow analysis interfaces next to the existing P2.  It
does not replace, wrap, or modify the current policy.  Virtual truth is allowed
only in explicitly post-hoc local-validation and cumulative-improvement audit
outputs; it is never a proposal, ranking, guard-execution, or stopping input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping

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
    SearchAlpha,
    TrustRegionSteps,
    build_coordinate_neighborhood,
)


PROTOTYPE_ID = "P2_REVISION_V2_RESEARCH_PROTOTYPE_V1"
LOCAL_PROTOCOL_ID = "LOCAL_DECISION_VALIDATION_PROTOCOL_V1"
EXPLORATION_VALUE_PROTOCOL_ID = "DECISION_VALUE_AWARE_EXPLORATION_SHADOW_V1"
CUMULATIVE_ANALYSIS_ID = "KNEE_STIFF_CUMULATIVE_IMPROVEMENT_AUDIT_V1"
PROTOTYPE_STATUS = "DEFAULT_OFF_RESEARCH_SHADOW_ONLY"
OFFLINE_METHOD_STATUS = "OFFLINE_METHOD_REQUIRES_REVISION"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"
LOCAL_PAIR_ROLE = (
    "RETROSPECTIVE_VIRTUAL_LOCAL_PROTOCOL_PROTOTYPE_NOT_FORMAL_CALIBRATION"
)
LOCAL_METRIC_STATUS = "RESEARCH_METRIC_ONLY_NOT_A_FROZEN_THRESHOLD"
CUMULATIVE_RULE_ASSESSMENT = (
    "CUMULATIVE_RULE_RESEARCH_CANDIDATE_REQUIRED_TO_ADDRESS_OBSERVED_KNEE_GAP_"
    "NOT_FORMAL_POLICY_APPROVAL"
)

_ALPHA_COLUMNS = ("hip", "knee", "phase")
_GRID_STEPS = np.asarray(
    (GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP), dtype=float
)
_ALLOWED_TRUST_LEVELS = {
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


@dataclass(frozen=True)
class PrototypeControls:
    """All behavior-changing switches remain off in this prototype."""

    current_p2_remains_default: bool = True
    local_uncertainty_policy_override_enabled: bool = False
    exploration_automatic_stop_enabled: bool = False
    cumulative_decision_rule_enabled: bool = False
    truth_policy_input_enabled: bool = False
    robot_execution_enabled: bool = False
    shadow_artifact_generation_enabled: bool = True

    def require_default_off(self) -> None:
        if not self.current_p2_remains_default:
            raise PermissionError("the existing P2 must remain the default policy")
        forbidden = {
            "local_uncertainty_policy_override_enabled": (
                self.local_uncertainty_policy_override_enabled
            ),
            "exploration_automatic_stop_enabled": (
                self.exploration_automatic_stop_enabled
            ),
            "cumulative_decision_rule_enabled": self.cumulative_decision_rule_enabled,
            "truth_policy_input_enabled": self.truth_policy_input_enabled,
            "robot_execution_enabled": self.robot_execution_enabled,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise PermissionError(
                "default-off research prototype forbids behavior switches: "
                + ", ".join(enabled)
            )
        if not self.shadow_artifact_generation_enabled:
            raise PermissionError("shadow artifact generation must be explicit")

    def to_dict(self) -> dict[str, Any]:
        self.require_default_off()
        return {
            "prototype_id": PROTOTYPE_ID,
            "prototype_status": PROTOTYPE_STATUS,
            **asdict(self),
            "formal_policy_created": False,
            "threshold_frozen": False,
            "automatic_stop_created": False,
            "offline_method_status": OFFLINE_METHOD_STATUS,
            "human_readiness": NOT_HUMAN_READY,
            "robot_motion_approval": NOT_ROBOT_MOTION_APPROVED,
        }


DEFAULT_PROTOTYPE_CONTROLS = PrototypeControls()


def _alpha_values(alpha: SearchAlpha) -> np.ndarray:
    return np.asarray(
        (alpha.hip_delta_deg, alpha.knee_delta_deg, alpha.phase_delta),
        dtype=float,
    )


def _inside_generator_bounds(alpha: SearchAlpha) -> bool:
    values = {
        "hip_amplitude_delta_deg": alpha.hip_delta_deg,
        "knee_amplitude_delta_deg": alpha.knee_delta_deg,
        "knee_phase_shift": alpha.phase_delta,
    }
    return all(
        float(lower) - 1e-12 <= float(values[name]) <= float(upper) + 1e-12
        for name, (lower, upper) in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
    )


def _validate_existing_trust_steps(steps: TrustRegionSteps) -> None:
    values = {"hip": steps.hip_deg, "knee": steps.knee_deg, "phase": steps.phase}
    for coordinate, value in values.items():
        if not any(
            math.isclose(float(value), float(allowed), abs_tol=1e-12, rel_tol=0.0)
            for allowed in _ALLOWED_TRUST_LEVELS[coordinate]
        ):
            raise ValueError(
                f"{coordinate} trust step {value} is not an existing trust-region level"
            )


def _lattice_lookup(parameter_lattice: pd.DataFrame) -> dict[tuple[float, float, float], dict[str, Any]]:
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
    lookup: dict[tuple[float, float, float], dict[str, Any]] = {}
    for row in parameter_lattice.to_dict(orient="records"):
        key = (
            round(float(row["hip_delta"]), 12),
            round(float(row["knee_delta"]), 12),
            round(float(row["phase_delta"]), 12),
        )
        if key in lookup:
            raise ValueError(f"parameter lattice contains duplicate alpha {key}")
        lookup[key] = row
    return lookup


def build_formal_local_neighborhood(
    parameter_lattice: pd.DataFrame,
    current: SearchAlpha,
    steps: TrustRegionSteps,
    *,
    case_id: str = "",
    iteration: int = 0,
) -> pd.DataFrame:
    """Build current plus signed coordinate neighbors without clipping or expansion."""

    _validate_existing_trust_steps(steps)
    lookup = _lattice_lookup(parameter_lattice)
    current_values = _alpha_values(current)
    rows: list[dict[str, Any]] = []
    for candidate in build_coordinate_neighborhood(current, steps):
        candidate_values = _alpha_values(candidate)
        difference = candidate_values - current_values
        changed = ~np.isclose(difference, 0.0, atol=1e-12, rtol=0.0)
        changed_count = int(changed.sum())
        changed_index = int(np.flatnonzero(changed)[0]) if changed_count == 1 else -1
        coordinate = _ALPHA_COLUMNS[changed_index] if changed_index >= 0 else "CURRENT"
        key = candidate.key()
        lattice_row = lookup.get(key)
        inside = _inside_generator_bounds(candidate)
        lattice_member = lattice_row is not None
        admissible = bool(
            lattice_member and lattice_row["geometrically_admissible"]
        )
        relationship_valid = bool(
            changed_count == 0
            or (
                changed_count == 1
                and math.isclose(
                    abs(float(difference[changed_index])),
                    float(
                        (steps.hip_deg, steps.knee_deg, steps.phase)[changed_index]
                    ),
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
            )
        )
        if changed_count == 0 and (not inside or not admissible):
            raise ValueError("current alpha must be inside the admissible generator lattice")
        rows.append(
            {
                "case_id": case_id,
                "iteration": int(iteration),
                "neighborhood_role": "CURRENT" if changed_count == 0 else "NEIGHBOR",
                "trajectory_id": (
                    str(lattice_row["trajectory_id"])
                    if lattice_member
                    else f"not_in_lattice_{key}"
                ),
                "current_alpha_hip": current.hip_delta_deg,
                "current_alpha_knee": current.knee_delta_deg,
                "current_alpha_phase": current.phase_delta,
                "candidate_alpha_hip": candidate.hip_delta_deg,
                "candidate_alpha_knee": candidate.knee_delta_deg,
                "candidate_alpha_phase": candidate.phase_delta,
                "delta_alpha_hip": float(difference[0]),
                "delta_alpha_knee": float(difference[1]),
                "delta_alpha_phase": float(difference[2]),
                "changed_coordinate_count": changed_count,
                "changed_coordinate": coordinate,
                "signed_coordinate_direction": (
                    "CURRENT"
                    if changed_index < 0
                    else "POSITIVE"
                    if difference[changed_index] > 0.0
                    else "NEGATIVE"
                ),
                "alpha_distance_formal_grid_steps": float(
                    np.sum(np.abs(difference / _GRID_STEPS))
                ),
                "alpha_distance_definition": (
                    "NORMALIZED_FORMAL_GENERATOR_GRID_L1_NOT_PHYSICAL_DISTANCE"
                ),
                "physical_distance_used": False,
                "trust_step_hip": steps.hip_deg,
                "trust_step_knee": steps.knee_deg,
                "trust_step_phase": steps.phase,
                "inside_existing_generator_bounds": inside,
                "formal_lattice_member": lattice_member,
                "geometrically_admissible": admissible,
                "formal_neighbor_relationship_valid": relationship_valid,
                "included_as_local_validation_neighbor": bool(
                    changed_count == 1
                    and relationship_valid
                    and inside
                    and admissible
                ),
                "pointwise_clipping_applied": False,
                "search_range_expanded": False,
                "current_P2_modified": False,
            }
        )
    return pd.DataFrame(rows)


def _unique_value_for_alpha(
    table: pd.DataFrame,
    alpha: SearchAlpha,
    value_column: str,
) -> pd.Series:
    required = {"hip_delta", "knee_delta", "phase_delta", value_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"value map missing columns: {sorted(missing)}")
    selected = table.loc[
        np.isclose(table["hip_delta"], alpha.hip_delta_deg, atol=1e-12, rtol=0.0)
        & np.isclose(
            table["knee_delta"], alpha.knee_delta_deg, atol=1e-12, rtol=0.0
        )
        & np.isclose(table["phase_delta"], alpha.phase_delta, atol=1e-12, rtol=0.0)
    ]
    if len(selected) != 1:
        raise ValueError(f"value map lacks unique alpha {alpha.key()}")
    return selected.iloc[0]


def build_local_validation_pairs(
    neighborhood: pd.DataFrame,
    prediction_map: pd.DataFrame,
    truth_map: pd.DataFrame,
    *,
    case_id: str,
    iteration: int,
) -> pd.DataFrame:
    """Build research pairs; truth is retained only as an offline metric label."""

    current_rows = neighborhood.loc[neighborhood["neighborhood_role"].eq("CURRENT")]
    if len(current_rows) != 1:
        raise ValueError("local neighborhood must contain exactly one current alpha")
    current_item = current_rows.iloc[0]
    current = SearchAlpha(
        float(current_item["current_alpha_hip"]),
        float(current_item["current_alpha_knee"]),
        float(current_item["current_alpha_phase"]),
    )
    current_pred = _unique_value_for_alpha(prediction_map, current, "J_pred")
    current_truth = _unique_value_for_alpha(truth_map, current, "J_truth")
    rows: list[dict[str, Any]] = []
    candidates = neighborhood.loc[
        neighborhood["included_as_local_validation_neighbor"].astype(bool)
    ]
    for item in candidates.to_dict(orient="records"):
        candidate = SearchAlpha(
            float(item["candidate_alpha_hip"]),
            float(item["candidate_alpha_knee"]),
            float(item["candidate_alpha_phase"]),
        )
        candidate_pred = _unique_value_for_alpha(prediction_map, candidate, "J_pred")
        candidate_truth = _unique_value_for_alpha(truth_map, candidate, "J_truth")
        delta_pred = float(candidate_pred["J_pred"] - current_pred["J_pred"])
        delta_truth = float(candidate_truth["J_truth"] - current_truth["J_truth"])
        identity = (
            f"{case_id}|{int(iteration)}|{current.key()}|{candidate.key()}|"
            f"{LOCAL_PROTOCOL_ID}"
        )
        rows.append(
            {
                "pair_id": "local_pair_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "case_id": case_id,
                "iteration": int(iteration),
                "current_trajectory_id": str(current_pred.get("trajectory_id", "")),
                "candidate_trajectory_id": str(
                    candidate_pred.get("trajectory_id", item["trajectory_id"])
                ),
                "current_alpha_hip": current.hip_delta_deg,
                "current_alpha_knee": current.knee_delta_deg,
                "current_alpha_phase": current.phase_delta,
                "candidate_alpha_hip": candidate.hip_delta_deg,
                "candidate_alpha_knee": candidate.knee_delta_deg,
                "candidate_alpha_phase": candidate.phase_delta,
                "delta_alpha_hip": item["delta_alpha_hip"],
                "delta_alpha_knee": item["delta_alpha_knee"],
                "delta_alpha_phase": item["delta_alpha_phase"],
                "changed_coordinate": item["changed_coordinate"],
                "alpha_distance_formal_grid_steps": item[
                    "alpha_distance_formal_grid_steps"
                ],
                "alpha_distance_definition": item["alpha_distance_definition"],
                "predicted_delta_J": delta_pred,
                "truth_delta_J_posthoc": delta_truth,
                "e_delta_J": abs(delta_pred - delta_truth),
                "model_supported": bool(candidate_pred.get("model_supported", True)),
                "current_model_supported": bool(
                    current_pred.get("model_supported", True)
                ),
                "true_improvement_posthoc": bool(
                    delta_truth < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                ),
                "local_pair_role": LOCAL_PAIR_ROLE,
                "research_metric_only": True,
                "threshold_frozen": False,
                "truth_used_by_formal_policy": False,
                "truth_used_only_for_posthoc_metric": True,
                "current_P2_modified": False,
            }
        )
    return pd.DataFrame(rows)


def format_retrospective_local_pairs(
    design_pairs: pd.DataFrame,
    parameter_lattice: pd.DataFrame,
) -> pd.DataFrame:
    """Adapt the frozen design opportunities into the prototype pair schema."""

    lookup = _lattice_lookup(parameter_lattice)
    rows: list[dict[str, Any]] = []
    for item in design_pairs.to_dict(orient="records"):
        current_key = tuple(
            round(float(item[f"current_alpha_{name}"]), 12)
            for name in _ALPHA_COLUMNS
        )
        candidate_key = tuple(
            round(float(item[f"candidate_alpha_{name}"]), 12)
            for name in _ALPHA_COLUMNS
        )
        if current_key not in lookup or candidate_key not in lookup:
            raise RuntimeError("retrospective local pair is outside formal lattice")
        if not bool(lookup[current_key]["geometrically_admissible"]):
            raise RuntimeError("retrospective current alpha is inadmissible")
        if not bool(lookup[candidate_key]["geometrically_admissible"]):
            raise RuntimeError("retrospective candidate alpha is inadmissible")
        identity = (
            f"{item['case_id']}|{int(item['iteration'])}|{current_key}|"
            f"{candidate_key}|{LOCAL_PROTOCOL_ID}"
        )
        rows.append(
            {
                "pair_id": "local_pair_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "case_id": item["case_id"],
                "subject_id": item["subject_id"],
                "scenario_name": item["scenario_name"],
                "iteration": int(item["iteration"]),
                "current_trajectory_id": str(lookup[current_key]["trajectory_id"]),
                "candidate_trajectory_id": str(
                    item.get("candidate_trajectory_id", lookup[candidate_key]["trajectory_id"])
                ),
                **{
                    f"current_alpha_{name}": float(item[f"current_alpha_{name}"])
                    for name in _ALPHA_COLUMNS
                },
                **{
                    f"candidate_alpha_{name}": float(item[f"candidate_alpha_{name}"])
                    for name in _ALPHA_COLUMNS
                },
                **{
                    f"delta_alpha_{name}": float(item[f"delta_alpha_{name}"])
                    for name in _ALPHA_COLUMNS
                },
                "changed_coordinate": item["changed_coordinate"],
                "signed_coordinate_direction": item["signed_coordinate_direction"],
                "alpha_distance_formal_grid_steps": float(
                    item["formal_grid_step_multiplier"]
                ),
                "alpha_distance_definition": (
                    "NORMALIZED_FORMAL_GENERATOR_GRID_L1_NOT_PHYSICAL_DISTANCE"
                ),
                "physical_distance_used": False,
                "inside_existing_generator_bounds": True,
                "formal_lattice_current_member": True,
                "formal_lattice_candidate_member": True,
                "geometrically_admissible_pair": True,
                "predicted_delta_J": float(item["delta_J_pred"]),
                "truth_delta_J_posthoc": float(item["delta_J_actual_posthoc"]),
                "e_delta_J": float(item["e_delta_J"]),
                "model_supported": bool(item["model_supported"]),
                "current_G0_would_exploit": bool(item["current_G0_would_exploit"]),
                "true_improvement_posthoc": bool(item["true_improvement_posthoc"]),
                "local_pair_role": LOCAL_PAIR_ROLE,
                "research_metric_only": True,
                "threshold_frozen": False,
                "truth_used_by_formal_policy": False,
                "truth_used_only_for_posthoc_metric": True,
                "current_P2_modified": False,
                "search_range_expanded": False,
            }
        )
    output = pd.DataFrame(rows)
    if output["pair_id"].duplicated().any():
        raise RuntimeError("local validation pair identifiers are not unique")
    return output


def local_uncertainty_metrics(local_pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled and leave-one-case-out research metrics."""

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str]] = [("POOLED_RETROSPECTIVE", "")]
    scopes.extend(
        ("LEAVE_ONE_CASE_OUT", case)
        for case in sorted(local_pairs["case_id"].astype(str).unique())
    )
    for scope, excluded_case in scopes:
        selected = (
            local_pairs
            if not excluded_case
            else local_pairs.loc[
                ~local_pairs["case_id"].astype(str).eq(excluded_case)
            ]
        )
        errors = selected["e_delta_J"].to_numpy(dtype=float)
        rows.append(
            {
                "metric_scope": scope,
                "excluded_evaluation_case_id": excluded_case,
                "pair_count": int(len(selected)),
                "case_count": int(selected["case_id"].nunique()),
                "local_max_error": float(np.max(errors)),
                "local_P95_error": float(np.percentile(errors, 95)),
                "local_P99_error": float(np.percentile(errors, 99)),
                "metric_status": LOCAL_METRIC_STATUS,
                "threshold_frozen": False,
                "formal_guard_uses_metric": False,
                "heldout_final_test_used": False,
                "current_P2_modified": False,
            }
        )
    return pd.DataFrame(rows)


def evaluate_local_guard_counterfactual(
    local_pairs: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare G0/G1/G2 in shadow mode; never expose a live policy output."""

    loco = metrics.loc[metrics["metric_scope"].eq("LEAVE_ONE_CASE_OUT")].set_index(
        "excluded_evaluation_case_id"
    )
    rows: list[dict[str, Any]] = []
    for item in local_pairs.to_dict(orient="records"):
        case_metrics = loco.loc[str(item["case_id"])]
        guards = (
            ("G0_CURRENT_GLOBAL_GUARD_REPLAY", np.nan),
            ("G1_LOCAL_MAX_RESEARCH_METRIC", float(case_metrics["local_max_error"])),
            ("G2_LOCAL_P95_RESEARCH_METRIC", float(case_metrics["local_P95_error"])),
        )
        true_improvement = bool(item["true_improvement_posthoc"])
        for guard_id, bound in guards:
            if guard_id.startswith("G0_"):
                would_exploit = bool(item["current_G0_would_exploit"])
                bound_role = "CURRENT_P2_BEHAVIOR_REPLAY"
            else:
                would_exploit = bool(
                    item["model_supported"]
                    and -float(item["predicted_delta_J"])
                    - float(bound)
                    - OBJECTIVE_EQUIVALENCE_TOLERANCE
                    > 0.0
                )
                bound_role = "LEAVE_ONE_CASE_OUT_RESEARCH_METRIC"
            rows.append(
                {
                    **item,
                    "guard_id": guard_id,
                    "uncertainty_metric": bound,
                    "uncertainty_metric_role": bound_role,
                    "would_exploit_counterfactual": would_exploit,
                    "missed_improvement_posthoc": bool(
                        not would_exploit and true_improvement
                    ),
                    "false_improvement_posthoc": bool(
                        would_exploit and not true_improvement
                    ),
                    "counterfactual_only": not guard_id.startswith("G0_"),
                    "threshold_frozen": False,
                    "formal_policy_modified": False,
                    "trajectory_executed": False,
                    "truth_used_to_modify_formal_policy": False,
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for guard_id, group in detail.groupby("guard_id", sort=False):
        rounds = group.groupby(["case_id", "iteration"], as_index=False).agg(
            true_improvement_available=("true_improvement_posthoc", "any"),
            would_exploit_any=("would_exploit_counterfactual", "any"),
        )
        summary_rows.append(
            {
                "guard_id": guard_id,
                "pair_count": int(len(group)),
                "would_exploit_candidate_count": int(
                    group["would_exploit_counterfactual"].sum()
                ),
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
                "threshold_frozen": False,
                "formal_policy_modified": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    g0 = summary.loc[
        summary["guard_id"].eq("G0_CURRENT_GLOBAL_GUARD_REPLAY")
    ].iloc[0]
    for column in (
        "missed_improvement_candidate_count",
        "false_improvement_candidate_count",
        "conservative_stop_round_count",
    ):
        summary[f"change_vs_G0_{column}"] = summary[column] - int(g0[column])
    return detail, summary


def build_exploration_value_history(
    exploration: pd.DataFrame,
    landscape_evolution: pd.DataFrame,
) -> pd.DataFrame:
    """Score executed explores after the fact; never emit an automatic stop."""

    landscape = landscape_evolution.set_index(["case_id", "iteration"])
    rows: list[dict[str, Any]] = []
    for item in exploration.to_dict(orient="records"):
        case_id = str(item["case_id"])
        iteration = int(item["iteration"])
        before_key = (case_id, iteration - 1)
        after_key = (case_id, iteration)
        if before_key not in landscape.index or after_key not in landscape.index:
            raise RuntimeError(
                f"prediction landscape lacks before/after explore rows {case_id}:{iteration}"
            )
        before = landscape.loc[before_key]
        after = landscape.loc[after_key]
        local_ranking_changed = bool(
            str(before["predicted_local_minimum_trajectory_id"])
            != str(after["predicted_local_minimum_trajectory_id"])
        )
        global_ranking_changed = bool(
            str(before["predicted_global_minimum_trajectory_id"])
            != str(after["predicted_global_minimum_trajectory_id"])
        )
        parameter_changed = bool(item["theta_changed_exactly"])
        map_changed = bool(item["prediction_map_changed_exactly"])
        validation_changed = bool(item["validation_error_changed_exactly"])
        model_value = bool(parameter_changed or map_changed or validation_changed)
        support_value = bool(float(item["new_supported_points"]) > 0.0)
        best_changed = bool(item["best_changed_under_existing_0p005_rule"])
        exploit_change = bool(int(item["newly_enabled_exploit_candidates"]) > 0)
        decision_value = bool(
            best_changed
            or local_ranking_changed
            or global_ranking_changed
            or exploit_change
        )
        rows.append(
            {
                "case_id": case_id,
                "subject_id": item["subject_id"],
                "scenario_name": item["scenario_name"],
                "iteration": iteration,
                "trajectory_id": item["trajectory_id"],
                "information_gain": float(item["information_gain"]),
                "new_supported_points": int(item["new_supported_points"]),
                "SUPPORT_VALUE": support_value,
                "parameter_delta_l2": float(item["theta_change_l2"]),
                "parameter_changed_exactly": parameter_changed,
                "prediction_map_RMS_delta": float(item["RMS_prediction_map_change"]),
                "prediction_map_max_delta": float(item["max_prediction_map_change"]),
                "prediction_map_changed_exactly": map_changed,
                "validation_uncertainty_change": float(
                    item["validation_deltaJ_error_change"]
                ),
                "validation_uncertainty_changed_exactly": validation_changed,
                "MODEL_VALUE": model_value,
                "best_J_change": float(item["best_J_change"]),
                "best_trajectory_changed": best_changed,
                "predicted_local_rank_1_before": str(
                    before["predicted_local_minimum_trajectory_id"]
                ),
                "predicted_local_rank_1_after": str(
                    after["predicted_local_minimum_trajectory_id"]
                ),
                "predicted_local_ranking_changed": local_ranking_changed,
                "predicted_global_rank_1_before": str(
                    before["predicted_global_minimum_trajectory_id"]
                ),
                "predicted_global_rank_1_after": str(
                    after["predicted_global_minimum_trajectory_id"]
                ),
                "predicted_global_ranking_changed": global_ranking_changed,
                "newly_enabled_exploit_candidates": int(
                    item["newly_enabled_exploit_candidates"]
                ),
                "newly_enabled_exploit_trajectory_ids": str(
                    item.get("newly_enabled_exploit_trajectory_ids", "")
                ),
                "exploit_eligibility_changed": exploit_change,
                "DECISION_VALUE": decision_value,
                "support_is_decision_value": False,
                "zero_model_and_decision_value_explore": bool(
                    support_value and not model_value and not decision_value
                ),
                "prototype_action": "SHADOW_SCORE_ONLY_NO_AUTOMATIC_STOP",
                "automatic_stop_triggered": False,
                "truth_landscape_used_for_scoring": False,
                "formal_policy_modified": False,
            }
        )
    return pd.DataFrame(rows)


def exploration_value_summary(history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "explore_trial_count": int(len(history)),
                "support_value_trial_count": int(history["SUPPORT_VALUE"].sum()),
                "model_value_trial_count": int(history["MODEL_VALUE"].sum()),
                "decision_value_trial_count": int(history["DECISION_VALUE"].sum()),
                "zero_model_and_decision_value_trial_count": int(
                    history["zero_model_and_decision_value_explore"].sum()
                ),
                "actual_explore_trials_avoided_by_prototype": 0,
                "automatic_stop_enabled": False,
                "scoring_reduces_exploration_by_itself": False,
                "research_interpretation": (
                    "IDENTIFIES_LOW_DECISION_VALUE_EXPLORES_BUT_DOES_NOT_STOP"
                ),
                "formal_policy_modified": False,
            }
        ]
    )


def build_knee_stiff_cumulative_improvement(
    truth_landscape: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit the existing 1-degree knee direction without changing objective."""

    required = {"trajectory_id", "hip_delta", "knee_delta", "phase_delta", "J_truth"}
    missing = required.difference(truth_landscape.columns)
    if missing:
        raise ValueError(f"truth landscape missing columns: {sorted(missing)}")
    lower = float(OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_amplitude_delta_deg"][0])
    step = float(INITIAL_STEP_KNEE_DEG)
    knee_values = np.arange(0.0, lower - 1e-12, -step)
    selected = truth_landscape.loc[
        np.isclose(truth_landscape["hip_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(truth_landscape["phase_delta"], 0.0, atol=1e-12, rtol=0.0)
        & truth_landscape["knee_delta"].isin(knee_values)
    ].copy()
    selected["_order"] = selected["knee_delta"].map(
        {value: index for index, value in enumerate(knee_values)}
    )
    selected = selected.sort_values("_order")
    if len(selected) != len(knee_values):
        raise RuntimeError("knee-stiff truth landscape lacks the existing knee path")
    j_values = selected["J_truth"].to_numpy(dtype=float)
    single_delta = np.concatenate(([np.nan], np.diff(j_values)))
    cumulative_delta = j_values - j_values[0]
    output = pd.DataFrame(
        {
            "step_index": np.arange(len(selected), dtype=int),
            "trajectory_id": selected["trajectory_id"].astype(str).to_numpy(),
            "alpha_hip": selected["hip_delta"].to_numpy(dtype=float),
            "alpha_knee": selected["knee_delta"].to_numpy(dtype=float),
            "alpha_phase": selected["phase_delta"].to_numpy(dtype=float),
            "J_truth_posthoc": j_values,
            "single_step_delta_J_truth": single_delta,
            "single_step_improvement_magnitude": np.maximum(-single_delta, 0.0),
            "single_step_exceeds_existing_0p005": np.where(
                np.isnan(single_delta),
                False,
                -single_delta > OBJECTIVE_EQUIVALENCE_TOLERANCE + 1e-12,
            ),
            "cumulative_delta_J_truth_from_reference": cumulative_delta,
            "cumulative_improvement_magnitude": np.maximum(-cumulative_delta, 0.0),
            "cumulative_exceeds_existing_0p005": (
                -cumulative_delta > OBJECTIVE_EQUIVALENCE_TOLERANCE + 1e-12
            ),
            "mechanical_objective_modified": False,
            "generator_direction_modified": False,
            "truth_used_by_policy": False,
            "truth_role": "POST_HOC_CUMULATIVE_METHOD_DIAGNOSTIC_ONLY",
        }
    )
    non_reference = output.loc[output["step_index"].gt(0)]
    first_cumulative = output.loc[
        output["cumulative_exceeds_existing_0p005"].astype(bool)
    ]
    all_single_subthreshold = bool(
        ~non_reference["single_step_exceeds_existing_0p005"].astype(bool).any()
    )
    cumulative_crosses = not first_cumulative.empty
    assessment = {
        "analysis_id": CUMULATIVE_ANALYSIS_ID,
        "subject_id": "knee_stiff",
        "path_definition": (
            "existing_generator_knee_direction_from_0_to_minus5_in_existing_1deg_trust_steps"
        ),
        "single_step_count": int(len(non_reference)),
        "all_single_steps_below_existing_0p005": all_single_subthreshold,
        "maximum_single_step_improvement": float(
            non_reference["single_step_improvement_magnitude"].max()
        ),
        "final_five_step_cumulative_improvement": float(
            output.iloc[-1]["cumulative_improvement_magnitude"]
        ),
        "cumulative_improvement_crosses_existing_0p005": cumulative_crosses,
        "first_step_count_crossing_existing_0p005": (
            int(first_cumulative.iloc[0]["step_index"])
            if cumulative_crosses
            else None
        ),
        "first_cumulative_improvement_crossing_existing_0p005": (
            float(first_cumulative.iloc[0]["cumulative_improvement_magnitude"])
            if cumulative_crosses
            else None
        ),
        "cumulative_decision_rule_research_design_required": bool(
            all_single_subthreshold and cumulative_crosses
        ),
        "assessment": (
            CUMULATIVE_RULE_ASSESSMENT
            if all_single_subthreshold and cumulative_crosses
            else "CUMULATIVE_RULE_NOT_SUPPORTED_BY_THIS_DIAGNOSTIC"
        ),
        "cumulative_rule_enabled": False,
        "objective_modified": False,
        "truth_used_to_modify_policy": False,
        "formal_policy_approval": False,
    }
    return output, assessment


__all__ = [
    "CUMULATIVE_ANALYSIS_ID",
    "CUMULATIVE_RULE_ASSESSMENT",
    "DEFAULT_PROTOTYPE_CONTROLS",
    "EXPLORATION_VALUE_PROTOCOL_ID",
    "LOCAL_METRIC_STATUS",
    "LOCAL_PAIR_ROLE",
    "LOCAL_PROTOCOL_ID",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_MOTION_APPROVED",
    "OFFLINE_METHOD_STATUS",
    "PROTOTYPE_ID",
    "PROTOTYPE_STATUS",
    "PrototypeControls",
    "build_exploration_value_history",
    "build_formal_local_neighborhood",
    "build_knee_stiff_cumulative_improvement",
    "build_local_validation_pairs",
    "evaluate_local_guard_counterfactual",
    "exploration_value_summary",
    "format_retrospective_local_pairs",
    "local_uncertainty_metrics",
]
