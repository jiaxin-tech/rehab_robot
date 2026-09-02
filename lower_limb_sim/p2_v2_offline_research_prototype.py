"""Default-off offline implementation of the P2 V2 research candidates.

The functions in this module evaluate a frozen local-validation plan and
historical shadow counterfactuals.  They do not call, wrap, or replace P2 V1.
Virtual truth is attached only after pair assignment and prediction inputs are
fixed, and it never becomes an input to a formal policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .decision_relevant_global_model_reliability import (
    DiagnosticInitialModel,
    build_predicted_map,
    build_trajectory_component_cache,
    evaluate_truth_map,
)
from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .geometry_error_metrics import StateDomainBounds
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE


PROTOTYPE_ID = "P2_V2_OFFLINE_RESEARCH_PROTOTYPE_IMPLEMENTATION_V1"
PROTOTYPE_STATUS = "DEFAULT_OFF_SHADOW_EVALUATION_NOT_FORMAL_POLICY"
FROZEN_LOCAL_PROTOCOL_ID = "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1"
FROZEN_PAIR_PLAN_SHA256 = (
    "ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55"
)
OFFLINE_METHOD_STATUS = "OFFLINE_METHOD_REQUIRES_REVISION"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"

_ALPHA_COLUMNS = ("hip_delta", "knee_delta", "phase_delta")
_PAIR_ALPHA_COLUMNS = {
    "A": ("alpha_i_hip", "alpha_i_knee", "alpha_i_phase"),
    "B": ("alpha_j_hip", "alpha_j_knee", "alpha_j_phase"),
}
_GUARD_IDS = (
    "G0_CURRENT_GLOBAL_UNCERTAINTY_REPLAY",
    "G1_DESIGNATED_LOCAL_MAX_SHADOW",
    "G2_DESIGNATED_LOCAL_P95_SHADOW",
    "G3_DESIGNATED_LOCAL_P99_SHADOW",
)
_CUMULATIVE_RULES = (
    ("RULE_A_SINGLE_STEP", 1),
    ("RULE_B_TWO_STEP_CUMULATIVE", 2),
    ("RULE_C_THREE_STEP_CUMULATIVE", 3),
    ("RULE_D_FIVE_STEP_CUMULATIVE", 5),
)
_STOPPING_K_CANDIDATES = (1, 2, 3)


@dataclass(frozen=True)
class OfflinePrototypeControls:
    p2_v1_remains_default: bool = True
    local_guard_policy_override_enabled: bool = False
    cumulative_rule_enabled: bool = False
    automatic_stopping_enabled: bool = False
    truth_policy_input_enabled: bool = False
    formal_personalization_enabled: bool = False
    robot_execution_enabled: bool = False

    def require_default_off(self) -> None:
        if not self.p2_v1_remains_default:
            raise PermissionError("P2 V1 must remain the default policy")
        enabled = {
            name: value
            for name, value in self.to_dict().items()
            if name != "p2_v1_remains_default" and value
        }
        if enabled:
            raise PermissionError(
                f"P2 V2 offline prototype controls must remain off: {sorted(enabled)}"
            )

    def to_dict(self) -> dict[str, bool]:
        return {
            "p2_v1_remains_default": self.p2_v1_remains_default,
            "local_guard_policy_override_enabled": (
                self.local_guard_policy_override_enabled
            ),
            "cumulative_rule_enabled": self.cumulative_rule_enabled,
            "automatic_stopping_enabled": self.automatic_stopping_enabled,
            "truth_policy_input_enabled": self.truth_policy_input_enabled,
            "formal_personalization_enabled": self.formal_personalization_enabled,
            "robot_execution_enabled": self.robot_execution_enabled,
        }


DEFAULT_CONTROLS = OfflinePrototypeControls()


def diagnostic_models_from_frozen_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, DiagnosticInitialModel]:
    """Reconstruct pre-existing diagnostic checkpoints without refitting."""

    definitions = metadata.get("diagnostic_initial_models")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("diagnostic model metadata is missing")
    models: dict[str, DiagnosticInitialModel] = {}
    for definition in definitions:
        case_id = str(definition["case_id"])
        domain = definition["identification_domain"]
        model = DiagnosticInitialModel(
            subject_id=str(definition["subject_id"]),
            scenario_name=str(definition["scenario_name"]),
            selected_trial_id=int(definition["selected_actual_trial_id"]),
            parameters={
                str(name): float(value)
                for name, value in definition["parameters"].items()
            },
            identification_domain=StateDomainBounds(
                columns=tuple(str(value) for value in domain["columns"]),
                lower=tuple(float(value) for value in domain["lower"]),
                upper=tuple(float(value) for value in domain["upper"]),
                valid_training_samples=int(domain["valid_training_samples"]),
            ),
            identification_dataset_sha256=str(
                definition["identification_dataset_sha256"]
            ),
        )
        if case_id != f"{model.subject_id}__{model.scenario_name}":
            raise ValueError(f"diagnostic model case identity mismatch: {case_id}")
        if case_id in models:
            raise ValueError(f"duplicate diagnostic model case: {case_id}")
        models[case_id] = model
    return models


def assign_frozen_pairs_to_cases(
    frozen_plan: pd.DataFrame,
    case_ids: Sequence[str],
) -> pd.DataFrame:
    """Assign pairs evenly using hashes only; prediction/truth are not inputs."""

    if frozen_plan["pair_id"].duplicated().any():
        raise ValueError("frozen pair plan contains duplicate pair IDs")
    cases = sorted({str(value) for value in case_ids})
    if not cases:
        raise ValueError("at least one pre-existing evaluation case is required")
    assignment = frozen_plan.loc[:, ["pair_id"]].copy()
    assignment["evaluation_assignment_hash"] = assignment["pair_id"].map(
        lambda pair_id: hashlib.sha256(
            f"{PROTOTYPE_ID}|CASE_ASSIGNMENT|{pair_id}".encode("utf-8")
        ).hexdigest()
    )
    assignment = assignment.sort_values(
        ["evaluation_assignment_hash", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)
    assignment["case_id"] = [cases[index % len(cases)] for index in range(len(assignment))]
    assignment["assignment_rank"] = np.arange(1, len(assignment) + 1, dtype=int)
    assignment["case_assignment_rule"] = (
        "ROUND_ROBIN_SORTED_SHA256_OF_PROTOCOL_AND_PAIR_ID_OVER_SORTED_EXISTING_CASE_IDS"
    )
    assignment["prediction_used_for_case_assignment"] = False
    assignment["truth_used_for_case_assignment"] = False
    assignment["final_truth_landscape_used_for_case_assignment"] = False
    output = frozen_plan.merge(assignment, on="pair_id", validate="one_to_one")
    return output.sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _parameter_rows(
    parameter_lattice: pd.DataFrame,
    trajectory_ids: Sequence[str],
) -> pd.DataFrame:
    lookup = parameter_lattice.set_index("trajectory_id", drop=False)
    identifiers = sorted({str(value) for value in trajectory_ids})
    missing = set(identifiers).difference(lookup.index.astype(str))
    if missing:
        raise ValueError(f"pair plan trajectories missing from formal lattice: {sorted(missing)}")
    selected = lookup.loc[identifiers].copy().reset_index(drop=True)
    required = {
        "trajectory_id",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "parent_reference_sha256",
        "geometrically_admissible",
    }
    missing_columns = required.difference(selected.columns)
    if missing_columns:
        raise ValueError(f"formal lattice missing columns: {sorted(missing_columns)}")
    return selected.loc[:, sorted(required)].copy()


def _point_lookup(table: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if table["trajectory_id"].duplicated().any():
        raise RuntimeError("evaluated point table contains duplicate trajectory IDs")
    return {
        str(row["trajectory_id"]): row
        for row in table.to_dict(orient="records")
    }


def generate_designated_local_validation_results(
    frozen_plan: pd.DataFrame,
    parameter_lattice: pd.DataFrame,
    diagnostic_models: Mapping[str, DiagnosticInitialModel],
) -> pd.DataFrame:
    """Freshly evaluate the already selected 324 pairs in offline simulation."""

    assigned = assign_frozen_pairs_to_cases(frozen_plan, diagnostic_models.keys())
    all_ids = [
        *assigned["trajectory_i"].astype(str),
        *assigned["trajectory_j"].astype(str),
    ]
    reference = parameter_lattice.loc[
        np.isclose(parameter_lattice["hip_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(parameter_lattice["knee_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(parameter_lattice["phase_delta"], 0.0, atol=1e-12, rtol=0.0)
    ]
    if len(reference) != 1:
        raise RuntimeError("formal lattice must contain the active reference exactly once")
    reference_id = str(reference.iloc[0]["trajectory_id"])
    master_points = _parameter_rows(parameter_lattice, [*all_ids, reference_id])
    cache = build_trajectory_component_cache(master_points)
    rows: list[dict[str, Any]] = []
    for case_id, case_pairs in assigned.groupby("case_id", sort=True):
        model = diagnostic_models[str(case_id)]
        case_ids = [
            *case_pairs["trajectory_i"].astype(str),
            *case_pairs["trajectory_j"].astype(str),
            reference_id,
        ]
        points = _parameter_rows(parameter_lattice, case_ids)
        predicted, prediction_audit = build_predicted_map(model, points, cache)
        if not prediction_audit["truth_evaluated_during_prediction"] is False:
            raise RuntimeError("truth leaked into designated prediction stage")
        evaluated, truth_audit = evaluate_truth_map(predicted, model, cache)
        if truth_audit["truth_used_for_pre_evaluation_ranking"]:
            raise RuntimeError("truth leaked into designated pair selection")
        lookup = _point_lookup(evaluated)
        for item in case_pairs.to_dict(orient="records"):
            point_a = lookup[str(item["trajectory_i"])]
            point_b = lookup[str(item["trajectory_j"])]
            predicted_delta = float(point_b["J_pred"] - point_a["J_pred"])
            truth_delta = float(point_b["J_truth"] - point_a["J_truth"])
            rows.append(
                {
                    "pair_id": item["pair_id"],
                    "case_id": case_id,
                    "subject_id": model.subject_id,
                    "scenario_name": model.scenario_name,
                    "evaluation_assignment_hash": item["evaluation_assignment_hash"],
                    "assignment_rank": int(item["assignment_rank"]),
                    "case_assignment_rule": item["case_assignment_rule"],
                    "candidate_A_trajectory_id": item["trajectory_i"],
                    "candidate_B_trajectory_id": item["trajectory_j"],
                    "candidate_alpha_A_hip": float(item["alpha_i_hip"]),
                    "candidate_alpha_A_knee": float(item["alpha_i_knee"]),
                    "candidate_alpha_A_phase": float(item["alpha_i_phase"]),
                    "candidate_alpha_B_hip": float(item["alpha_j_hip"]),
                    "candidate_alpha_B_knee": float(item["alpha_j_knee"]),
                    "candidate_alpha_B_phase": float(item["alpha_j_phase"]),
                    "changed_coordinate": item["coordinate"],
                    "trust_level": item["trust_level"],
                    "trust_step": float(item["trust_step"]),
                    "alpha_distance_formal_grid_steps": float(
                        item["alpha_distance_formal_grid_steps"]
                    ),
                    "alpha_distance_definition": item["alpha_distance_definition"],
                    "J_pred_A": float(point_a["J_pred"]),
                    "J_pred_B": float(point_b["J_pred"]),
                    "predicted_delta_J": predicted_delta,
                    "J_truth_A": float(point_a["J_truth"]),
                    "J_truth_B": float(point_b["J_truth"]),
                    "truth_delta_J": truth_delta,
                    "e_delta_J": abs(predicted_delta - truth_delta),
                    "candidate_A_model_supported": bool(point_a["model_supported"]),
                    "candidate_B_model_supported": bool(point_b["model_supported"]),
                    "diagnostic_model_trial_id": model.selected_trial_id,
                    "diagnostic_identification_dataset_sha256": (
                        model.identification_dataset_sha256
                    ),
                    "pair_plan_sha256": FROZEN_PAIR_PLAN_SHA256,
                    "pair_selection_changed_after_truth": False,
                    "prediction_used_for_case_assignment": False,
                    "truth_used_for_case_assignment": False,
                    "truth_used_for_pair_selection": False,
                    "truth_used_to_modify_formal_policy": False,
                    "evaluation_role": (
                        "FRESH_OFFLINE_DESIGNATED_OUTCOME_AFTER_PAIR_PLAN_FREEZE"
                    ),
                    "formal_guard_input": False,
                    "P2_V1_modified": False,
                }
            )
    output = pd.DataFrame(rows).sort_values("pair_id", kind="mergesort")
    if len(output) != len(frozen_plan) or not output["pair_id"].is_unique:
        raise RuntimeError("designated local result count/identity mismatch")
    if set(output["pair_id"]) != set(frozen_plan["pair_id"]):
        raise RuntimeError("designated local results changed the frozen pair set")
    if not np.isfinite(
        output[["predicted_delta_J", "truth_delta_J", "e_delta_J"]].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError("designated local results contain non-finite outcomes")
    return output.reset_index(drop=True)


def local_uncertainty_metrics(results: pd.DataFrame) -> dict[str, float]:
    errors = results["e_delta_J"].to_numpy(dtype=float)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("designated local errors must be finite and non-empty")
    return {
        "local_max": float(np.max(errors)),
        "local_P95": float(np.percentile(errors, 95)),
        "local_P99": float(np.percentile(errors, 99)),
    }


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ValueError(f"{name} must be a strict boolean")


def evaluate_local_uncertainty_guards_shadow(
    historical_guard_candidates: pd.DataFrame,
    metrics: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay G0 and evaluate G1/G2/G3 without returning a policy action."""

    required = {
        "case_id",
        "subject_id",
        "scenario_name",
        "iteration",
        "trajectory_id",
        "delta_J_pred",
        "delta_J_truth_post_decision",
        "model_supported",
        "would_exploit",
        "uncertainty_bound",
    }
    missing = required.difference(historical_guard_candidates.columns)
    if missing:
        raise ValueError(f"historical guard candidates missing: {sorted(missing)}")
    guards = (
        (_GUARD_IDS[0], None, "CURRENT_FROZEN_PER_CASE_ITERATION_BOUND"),
        (_GUARD_IDS[1], float(metrics["local_max"]), "DESIGNATED_LOCAL_MAX"),
        (_GUARD_IDS[2], float(metrics["local_P95"]), "DESIGNATED_LOCAL_P95"),
        (_GUARD_IDS[3], float(metrics["local_P99"]), "DESIGNATED_LOCAL_P99"),
    )
    rows: list[dict[str, Any]] = []
    for item in historical_guard_candidates.to_dict(orient="records"):
        model_supported = _strict_bool(item["model_supported"], name="model_supported")
        true_improvement = bool(
            float(item["delta_J_truth_post_decision"])
            < -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        for guard_id, candidate_bound, bound_role in guards:
            if candidate_bound is None:
                effective_bound = float(item["uncertainty_bound"])
                would_exploit = _strict_bool(
                    item["would_exploit"], name="would_exploit"
                )
            else:
                effective_bound = candidate_bound
                would_exploit = bool(
                    model_supported
                    and -float(item["delta_J_pred"])
                    - effective_bound
                    - OBJECTIVE_EQUIVALENCE_TOLERANCE
                    > 0.0
                )
            rows.append(
                {
                    "case_id": item["case_id"],
                    "subject_id": item["subject_id"],
                    "scenario_name": item["scenario_name"],
                    "iteration": int(item["iteration"]),
                    "trajectory_id": item["trajectory_id"],
                    "guard_id": guard_id,
                    "uncertainty_bound": effective_bound,
                    "uncertainty_bound_role": bound_role,
                    "predicted_delta_J": float(item["delta_J_pred"]),
                    "truth_delta_J_posthoc": float(
                        item["delta_J_truth_post_decision"]
                    ),
                    "model_supported": model_supported,
                    "would_exploit_shadow": would_exploit,
                    "true_improvement_posthoc": true_improvement,
                    "missed_improvement": bool(
                        true_improvement and not would_exploit
                    ),
                    "false_improvement": bool(
                        would_exploit and not true_improvement
                    ),
                    "conservative_rejection": bool(
                        not would_exploit and not true_improvement
                    ),
                    "counterfactual_only": guard_id != _GUARD_IDS[0],
                    "threshold_frozen": False,
                    "formal_policy_modified": False,
                    "trajectory_executed": False,
                    "truth_used_to_modify_formal_policy": False,
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("guard_id", sort=False, as_index=False)
        .agg(
            candidate_count=("trajectory_id", "count"),
            would_exploit_count=("would_exploit_shadow", "sum"),
            missed_improvement_count=("missed_improvement", "sum"),
            false_improvement_count=("false_improvement", "sum"),
            conservative_rejection_count=("conservative_rejection", "sum"),
            uncertainty_bound_min=("uncertainty_bound", "min"),
            uncertainty_bound_max=("uncertainty_bound", "max"),
        )
    )
    g0 = summary.set_index("guard_id").loc[_GUARD_IDS[0]]
    for column in (
        "missed_improvement_count",
        "false_improvement_count",
        "conservative_rejection_count",
    ):
        summary[f"change_vs_G0_{column}"] = summary[column] - int(g0[column])
    summary["shadow_only"] = True
    summary["threshold_frozen"] = False
    summary["formal_policy_modified"] = False
    summary["truth_used_to_modify_formal_policy"] = False
    return detail, summary


def evaluate_knee_stiff_cumulative_shadow(
    evaluated_knee_path: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate fixed 1/2/3/5-step knee bundles before an uncertainty choice."""

    selected = evaluated_knee_path.loc[
        np.isclose(evaluated_knee_path["hip_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(evaluated_knee_path["phase_delta"], 0.0, atol=1e-12, rtol=0.0)
        & evaluated_knee_path["knee_delta"].isin([0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
    ].sort_values("knee_delta", ascending=False, kind="mergesort")
    if len(selected) != 6:
        raise ValueError("knee_stiff cumulative path must contain 0 to -5 deg")
    by_knee = selected.set_index("knee_delta")
    origin = by_knee.loc[0.0]
    rows: list[dict[str, Any]] = []
    for rule_id, steps in _CUMULATIVE_RULES:
        endpoint = by_knee.loc[-float(steps)]
        sequence = [
            str(by_knee.loc[-float(index)]["trajectory_id"])
            for index in range(0, steps + 1)
        ]
        predicted_delta = float(endpoint["J_pred"] - origin["J_pred"])
        truth_delta = float(endpoint["J_truth"] - origin["J_truth"])
        accepts_before_uncertainty = bool(
            -predicted_delta - OBJECTIVE_EQUIVALENCE_TOLERANCE > 0.0
        )
        true_improvement = bool(
            truth_delta < -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        rows.append(
            {
                "case_id": "knee_stiff__matched_linear",
                "rule_id": rule_id,
                "trajectory_sequence": ";".join(sequence),
                "trajectory_sequence_length": steps,
                "trajectory_point_count": steps + 1,
                "start_trajectory_id": sequence[0],
                "end_trajectory_id": sequence[-1],
                "end_alpha_knee": -float(steps),
                "predicted_cumulative_delta_J": predicted_delta,
                "truth_cumulative_delta_J_posthoc": truth_delta,
                "passes_existing_0p005_before_bundle_uncertainty": (
                    accepts_before_uncertainty
                ),
                "true_improvement_posthoc": true_improvement,
                "recovered_improvement": (
                    max(-truth_delta, 0.0)
                    if accepts_before_uncertainty and true_improvement
                    else 0.0
                ),
                "false_acceptance": bool(
                    accepts_before_uncertainty and not true_improvement
                ),
                "direction_consistent": True,
                "same_generator_coordinate": True,
                "inside_existing_generator_bounds": True,
                "uncertainty_constraint_applied": False,
                "uncertainty_constraint_status": (
                    "UNFROZEN_REQUIRES_DESIGNATED_BUNDLE_VALIDATION"
                ),
                "rule_frozen": False,
                "rule_enabled": False,
                "trajectory_executed": False,
                "truth_used_to_select_sequence": False,
                "truth_used_to_modify_formal_policy": False,
            }
        )
    return pd.DataFrame(rows)


def freshly_evaluate_knee_stiff_path(
    parameter_lattice: pd.DataFrame,
    diagnostic_model: DiagnosticInitialModel,
) -> pd.DataFrame:
    """Generate the predeclared 0,-1,...,-5 knee path without truth selection."""

    path_points = parameter_lattice.loc[
        np.isclose(parameter_lattice["hip_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(parameter_lattice["phase_delta"], 0.0, atol=1e-12, rtol=0.0)
        & parameter_lattice["knee_delta"].isin([0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
    ].copy()
    if len(path_points) != 6:
        raise RuntimeError("formal lattice lacks the predeclared knee path")
    # The initial knee-stiff checkpoint labels this whole path unsupported.
    # build_predicted_map also computes distance-to-support, so add the fixed
    # lower generator corner solely as a bookkeeping anchor.  It is selected
    # by alpha order, not by prediction or truth, and is removed on return.
    support_bookkeeping_anchor = parameter_lattice.sort_values(
        list(_ALPHA_COLUMNS), kind="mergesort"
    ).iloc[[0]]
    points = pd.concat(
        [path_points, support_bookkeeping_anchor], ignore_index=True
    ).drop_duplicates("trajectory_id")
    cache = build_trajectory_component_cache(points)
    predicted, _ = build_predicted_map(diagnostic_model, points, cache)
    evaluated, _ = evaluate_truth_map(predicted, diagnostic_model, cache)
    return evaluated.loc[
        evaluated["trajectory_id"].isin(path_points["trajectory_id"])
    ].reset_index(drop=True)


def _zero_decision_value(row: Mapping[str, Any]) -> bool:
    return bool(
        not _strict_bool(row["parameter_changed_exactly"], name="parameter_changed")
        and not _strict_bool(
            row["prediction_map_changed_exactly"], name="prediction_map_changed"
        )
        and not _strict_bool(
            row["predicted_local_ranking_changed"], name="local_ranking_changed"
        )
        and not _strict_bool(
            row["predicted_global_ranking_changed"], name="global_ranking_changed"
        )
        and not _strict_bool(
            row["best_trajectory_changed"], name="best_trajectory_changed"
        )
        and not _strict_bool(
            row["exploit_eligibility_changed"], name="exploit_eligibility_changed"
        )
    )


def _stopping_trigger(case_exploration: pd.DataFrame, required: int) -> int | None:
    run = 0
    previous: int | None = None
    for item in case_exploration.sort_values("iteration").to_dict(orient="records"):
        iteration = int(item["iteration"])
        if _zero_decision_value(item):
            run = run + 1 if previous is not None and iteration == previous + 1 else 1
        else:
            run = 0
        if run >= required:
            return iteration
        previous = iteration
    return None


def _alpha_to_trajectory_lookup(parameter_lattice: pd.DataFrame) -> dict[tuple[float, float, float], str]:
    return {
        tuple(round(float(row[column]), 12) for column in _ALPHA_COLUMNS): str(
            row["trajectory_id"]
        )
        for row in parameter_lattice.to_dict(orient="records")
    }


def evaluate_decision_value_stopping_shadow(
    exploration_history: pd.DataFrame,
    executed_history: pd.DataFrame,
    parameter_lattice: pd.DataFrame,
) -> pd.DataFrame:
    """Replay K=1/2/3 stop points; never stop the historical or live policy."""

    alpha_lookup = _alpha_to_trajectory_lookup(parameter_lattice)
    rows: list[dict[str, Any]] = []
    for case_id, case_explore in exploration_history.groupby("case_id", sort=True):
        case_execution = executed_history.loc[
            executed_history["case_id"].astype(str).eq(str(case_id))
        ].sort_values("iteration", kind="mergesort")
        if case_execution.empty:
            raise ValueError(f"executed history missing case: {case_id}")
        full_last = case_execution.iloc[-1]
        full_alpha = tuple(
            round(float(full_last[f"best_alpha_{axis}_after"]), 12)
            for axis in ("hip", "knee", "phase")
        )
        full_best = alpha_lookup[full_alpha]
        full_explore_count = int(len(case_explore))
        full_support = int(case_explore["new_supported_points"].sum())
        candidates: list[tuple[str, int | None, int | None]] = [
            ("CURRENT_P2_V1_HISTORY", None, None),
            *[
                (f"K{required}_DECISION_VALUE_STOP_SHADOW", required, _stopping_trigger(case_explore, required))
                for required in _STOPPING_K_CANDIDATES
            ],
        ]
        for strategy_id, required, trigger in candidates:
            if trigger is None:
                retained_explore = case_explore
                retained_execution = case_execution
                trigger_observed = False
            else:
                retained_explore = case_explore.loc[
                    case_explore["iteration"].astype(int).le(trigger)
                ]
                retained_execution = case_execution.loc[
                    case_execution["iteration"].astype(int).le(trigger)
                ]
                trigger_observed = True
            last = retained_execution.iloc[-1]
            alpha = tuple(
                round(float(last[f"best_alpha_{axis}_after"]), 12)
                for axis in ("hip", "knee", "phase")
            )
            final_best = alpha_lookup[alpha]
            later = (
                case_execution.iloc[0:0]
                if trigger is None
                else case_execution.loc[
                    case_execution["iteration"].astype(int).gt(trigger)
                ]
            )
            missed_opportunity = bool(
                final_best != full_best
                or later["accepted_improvement"].map(
                    lambda value: _strict_bool(value, name="accepted_improvement")
                ).any()
            )
            exploration_count = int(len(retained_explore))
            support_increase = int(retained_explore["new_supported_points"].sum())
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case_explore["subject_id"].iloc[0],
                    "scenario_name": case_explore["scenario_name"].iloc[0],
                    "strategy_id": strategy_id,
                    "consecutive_zero_value_K": required,
                    "trigger_observed": trigger_observed,
                    "shadow_stop_after_iteration": trigger,
                    "exploration_count": exploration_count,
                    "exploration_reduction_vs_current": (
                        full_explore_count - exploration_count
                    ),
                    "final_best_trajectory": final_best,
                    "current_final_best_trajectory": full_best,
                    "missed_opportunity": missed_opportunity,
                    "later_exploit_count": int(
                        later["trial_purpose"].astype(str).eq("EXPLOIT").sum()
                    ),
                    "later_accepted_improvement_count": int(
                        later["accepted_improvement"].map(
                            lambda value: _strict_bool(
                                value, name="accepted_improvement"
                            )
                        ).sum()
                    ),
                    "support_increase": support_increase,
                    "support_increase_current": full_support,
                    "support_increase_not_used_as_decision_value": True,
                    "automatic_stop_executed": False,
                    "candidate_frozen": False,
                    "truth_feature_used": False,
                    "formal_policy_modified": False,
                }
            )
    return pd.DataFrame(rows)


def minimum_p2_v2_change_set() -> list[str]:
    return [
        "review_and_freeze_an_independent_designated_local_uncertainty_statistic",
        "add_a_default_off_local_uncertainty_provider_while_P2_V1_remains_default",
        "prospectively_validate_same_direction_cumulative_bundles_and_uncertainty_aggregation",
        "add_default_off_cumulative_bundle_evaluation_for_knee_stepwise_candidates",
        "record_support_model_prediction_and_decision_value_separately",
        "prospectively_validate_default_off_K_stopping_before_any_policy_enablement",
    ]


__all__ = [
    "DEFAULT_CONTROLS",
    "FROZEN_LOCAL_PROTOCOL_ID",
    "FROZEN_PAIR_PLAN_SHA256",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_MOTION_APPROVED",
    "OFFLINE_METHOD_STATUS",
    "OfflinePrototypeControls",
    "PROTOTYPE_ID",
    "PROTOTYPE_STATUS",
    "assign_frozen_pairs_to_cases",
    "diagnostic_models_from_frozen_metadata",
    "evaluate_decision_value_stopping_shadow",
    "evaluate_knee_stiff_cumulative_shadow",
    "evaluate_local_uncertainty_guards_shadow",
    "freshly_evaluate_knee_stiff_path",
    "generate_designated_local_validation_results",
    "local_uncertainty_metrics",
    "minimum_p2_v2_change_set",
]
