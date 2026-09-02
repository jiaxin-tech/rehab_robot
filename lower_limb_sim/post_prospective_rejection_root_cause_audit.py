"""Post-prospective development audit for the rejected P2 V2A revision.

All functions in this module operate on the already revealed prospective
cohort or on geometry-only candidate plans.  They cannot revise the immutable
prospective conclusion and do not define or execute a new policy version.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
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
from .p2_v2_prospective_offline_validation import (
    FINAL_REJECTS,
    LOCAL_P95,
    ProspectivePolicySpec,
)


AUDIT_ID = "POST_PROSPECTIVE_REJECTION_ROOT_CAUSE_AUDIT_V1"
AUDIT_DATA_ROLE = "POST_PROSPECTIVE_DEVELOPMENT_ONLY"
PROSPECTIVE_CONCLUSION = FINAL_REJECTS
PROSPECTIVE_START_COMMIT = "d7fe80945ae625fffc7919e1735e9e2df8c8fa00"
PROSPECTIVE_MANIFEST_SHA256 = (
    "94d33675b2ae51ef80154c3bba92f31b87852267f3cffbaaacc75c3ce0aa1876"
)
BUNDLE_PROTOCOL_ID = "DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1"
BUNDLE_OUTCOME_STATUS = "PENDING_FUTURE_INDEPENDENT_BUNDLE_CALIBRATION"
FINAL_STATUS_IDENTIFIED = "NEXT_REVISION_ROOT_CAUSE_IDENTIFIED"
FINAL_STATUS_MORE_ANALYSIS = "MORE_POST_PROSPECTIVE_ANALYSIS_REQUIRED"
FINAL_STATUSES = (FINAL_STATUS_IDENTIFIED, FINAL_STATUS_MORE_ANALYSIS)

MODULE_DIR = Path(__file__).resolve().parent
PROSPECTIVE_ARTIFACT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_v2_prospective_offline_validation_v1"
)
PROSPECTIVE_MANIFEST_PATH = (
    PROSPECTIVE_ARTIFACT_DIRECTORY
    / "P2_V2_PROSPECTIVE_EVALUATION_MANIFEST_V1.json"
)

FACTORIAL_SPECS = (
    (
        "A0_G0_C0_S0_ORIGINAL_P2_V1",
        ProspectivePolicySpec(
            "P2_V1_G0_C0_S0",
            "G0_CURRENT_GLOBAL_MAX",
            "C0_SINGLE_STEP",
            "S0_CURRENT_CONTINUATION",
            None,
            "HISTORICAL_PROSPECTIVE_PRIMARY_A0",
        ),
        "HISTORICAL_PROSPECTIVE_PRIMARY",
    ),
    (
        "A1_G2_C0_S0_POST_HOC",
        ProspectivePolicySpec(
            "POST_HOC_A1_G2_C0_S0",
            "G2_FROZEN_LOCAL_P95",
            "C0_SINGLE_STEP",
            "S0_CURRENT_CONTINUATION",
            None,
            "POST_HOC_COUNTERFACTUAL_ONLY",
        ),
        "POST_HOC_COUNTERFACTUAL_ONLY",
    ),
    (
        "A2_G0_C0_S2_POST_HOC",
        ProspectivePolicySpec(
            "POST_HOC_A2_G0_C0_S2",
            "G0_CURRENT_GLOBAL_MAX",
            "C0_SINGLE_STEP",
            "S2_DECISION_VALUE_K2",
            2,
            "POST_HOC_COUNTERFACTUAL_ONLY",
        ),
        "POST_HOC_COUNTERFACTUAL_ONLY",
    ),
    (
        "A3_G2_C0_S2_REJECTED_V2A",
        ProspectivePolicySpec(
            "P2_V2A_G2_C0_S2",
            "G2_FROZEN_LOCAL_P95",
            "C0_SINGLE_STEP",
            "S2_DECISION_VALUE_K2",
            2,
            "HISTORICAL_PROSPECTIVE_PRIMARY_A3_REJECTED",
        ),
        "HISTORICAL_PROSPECTIVE_PRIMARY",
    ),
)

_FACTORIAL_BY_POLICY = {
    spec.policy_variant_id: (factorial_id, evidence_role)
    for factorial_id, spec, evidence_role in FACTORIAL_SPECS
}
_ALPHA_COLUMNS = ("hip_delta", "knee_delta", "phase_delta")
_COORDINATE_INDEX = {"hip": 0, "knee": 1, "phase": 2}
_GRID_STEPS = {
    "hip": GRID_HIP_STEP_DEG,
    "knee": GRID_KNEE_STEP_DEG,
    "phase": GRID_PHASE_STEP,
}
_BOUND_KEYS = {
    "hip": "hip_amplitude_delta_deg",
    "knee": "knee_amplitude_delta_deg",
    "phase": "knee_phase_shift",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_immutable_prospective_artifacts() -> dict[str, Any]:
    metadata_path = PROSPECTIVE_ARTIFACT_DIRECTORY / "metadata.json"
    metadata = pd.read_json(metadata_path, typ="series").to_dict()
    if metadata["final_status"] != PROSPECTIVE_CONCLUSION:
        raise RuntimeError("immutable prospective rejection conclusion changed")
    if metadata["prospective_manifest_sha256"] != PROSPECTIVE_MANIFEST_SHA256:
        raise RuntimeError("immutable prospective manifest SHA changed")
    if metadata["prospective_start_commit_sha"] != PROSPECTIVE_START_COMMIT:
        raise RuntimeError("immutable prospective start commit changed")
    if sha256_file(PROSPECTIVE_MANIFEST_PATH) != PROSPECTIVE_MANIFEST_SHA256:
        raise RuntimeError("prospective manifest file bytes changed")
    return metadata


def attach_factorial_identity(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy(deep=True)
    output["factorial_variant_id"] = output["policy_id"].map(
        lambda value: _FACTORIAL_BY_POLICY[str(value)][0]
    )
    output["evidence_role"] = output["policy_id"].map(
        lambda value: _FACTORIAL_BY_POLICY[str(value)][1]
    )
    output["data_role"] = AUDIT_DATA_ROLE
    output["prospective_conclusion"] = PROSPECTIVE_CONCLUSION
    output["prospective_conclusion_revised"] = False
    output["truth_used_to_modify_historical_policy"] = False
    return output


def verify_historical_reproduction(
    factorial_summary: pd.DataFrame,
    historical_summary: pd.DataFrame,
    *,
    atol: float = 1e-11,
) -> pd.DataFrame:
    metric_columns = (
        "number_of_executed_trials",
        "number_of_explore_trials",
        "number_of_exploit_trials",
        "missed_improvement_rounds",
        "premature_conservative_stops",
        "number_of_executed_false_improvements",
        "final_best_actual_J",
        "global_truth_regret",
        "low_decision_value_exploration_count",
        "correct_local_stops",
        "final_best_alpha_hip",
        "final_best_alpha_knee",
        "final_best_alpha_phase",
    )
    rows: list[dict[str, Any]] = []
    for policy_id in ("P2_V1_G0_C0_S0", "P2_V2A_G2_C0_S2"):
        current = factorial_summary.loc[
            factorial_summary["policy_id"].eq(policy_id)
        ].set_index("case_id")
        frozen = historical_summary.loc[
            historical_summary["policy_id"].eq(policy_id)
        ].set_index("case_id")
        if set(current.index) != set(frozen.index):
            raise RuntimeError(f"historical reproduction cases differ for {policy_id}")
        for case_id in sorted(current.index):
            for metric in metric_columns:
                observed = float(current.loc[case_id, metric])
                expected = float(frozen.loc[case_id, metric])
                exact = bool(np.isclose(observed, expected, atol=atol, rtol=0.0))
                rows.append(
                    {
                        "case_id": case_id,
                        "policy_id": policy_id,
                        "metric": metric,
                        "historical_value": expected,
                        "replayed_value": observed,
                        "absolute_difference": abs(observed - expected),
                        "reproduced": exact,
                    }
                )
                if not exact:
                    raise RuntimeError(
                        f"historical prospective result did not reproduce: "
                        f"{policy_id}/{case_id}/{metric}"
                    )
    return pd.DataFrame(rows)


def factorial_decomposition(summary: pd.DataFrame) -> pd.DataFrame:
    """Return transparent case-wise 2x2 simple effects and interaction."""

    metrics = (
        "number_of_executed_trials",
        "number_of_explore_trials",
        "number_of_exploit_trials",
        "missed_improvement_rounds",
        "premature_conservative_stops",
        "number_of_executed_false_improvements",
        "final_best_actual_J",
        "global_truth_regret",
        "low_decision_value_exploration_count",
        "correct_local_stops",
        "final_best_alpha_hip",
        "final_best_alpha_knee",
        "final_best_alpha_phase",
    )
    by_factor = summary.set_index(["case_id", "factorial_variant_id"])
    ids = [item[0] for item in FACTORIAL_SPECS]
    rows: list[dict[str, Any]] = []
    for case_id in sorted(summary["case_id"].unique()):
        for metric in metrics:
            a0, a1, a2, a3 = (
                float(by_factor.loc[(case_id, identifier), metric])
                for identifier in ids
            )
            guard_at_s0 = a1 - a0
            guard_at_s2 = a3 - a2
            stopping_at_g0 = a2 - a0
            stopping_at_g2 = a3 - a1
            interaction = a3 - a2 - a1 + a0
            candidates = {
                "GUARD_EFFECT": 0.5 * (guard_at_s0 + guard_at_s2),
                "STOPPING_EFFECT": 0.5 * (stopping_at_g0 + stopping_at_g2),
                "GUARD_STOPPING_INTERACTION": interaction,
            }
            maximum = max(abs(value) for value in candidates.values())
            dominant = (
                "NONE_SUFFICIENT_EXPLAINS_FAILURE"
                if np.isclose(maximum, 0.0, atol=1e-15, rtol=0.0)
                else max(candidates, key=lambda name: abs(candidates[name]))
            )
            rows.append(
                {
                    "case_id": case_id,
                    "metric": metric,
                    "A0": a0,
                    "A1": a1,
                    "A2": a2,
                    "A3": a3,
                    "A3_minus_A0_total": a3 - a0,
                    "guard_effect_at_S0_A1_minus_A0": guard_at_s0,
                    "guard_effect_at_S2_A3_minus_A2": guard_at_s2,
                    "stopping_effect_at_G0_A2_minus_A0": stopping_at_g0,
                    "stopping_effect_at_G2_A3_minus_A1": stopping_at_g2,
                    "factorial_guard_main_effect": candidates["GUARD_EFFECT"],
                    "factorial_stopping_main_effect": candidates["STOPPING_EFFECT"],
                    "guard_stopping_interaction": interaction,
                    "dominant_direct_effect": dominant,
                    "complex_statistical_model_used": False,
                    "data_role": AUDIT_DATA_ROLE,
                }
            )
    return pd.DataFrame(rows)


def _current_alpha_for_iteration(
    history: pd.DataFrame,
    case_id: str,
    policy_id: str,
    iteration: int,
) -> tuple[float, float, float]:
    selected = history.loc[
        history["case_id"].eq(case_id) & history["policy_id"].eq(policy_id)
    ].sort_values("iteration")
    exact = selected.loc[selected["iteration"].astype(int).eq(int(iteration))]
    if not exact.empty:
        row = exact.iloc[0]
        return tuple(float(row[f"best_alpha_{axis}_before"]) for axis in ("hip", "knee", "phase"))
    before = selected.loc[selected["iteration"].astype(int).lt(int(iteration))]
    if before.empty:
        return (0.0, 0.0, 0.0)
    row = before.iloc[-1]
    return tuple(float(row[f"best_alpha_{axis}_after"]) for axis in ("hip", "knee", "phase"))


def _mechanism_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    predicted_delta = float(row["delta_J_pred"])
    truth_delta = float(row["delta_J_truth"])
    guard_status = str(row["guard_status"])
    if predicted_delta >= 0.0 and truth_delta < -OBJECTIVE_EQUIVALENCE_TOLERANCE:
        labels.append("MODEL_PREDICTED_WRONG_DIRECTION")
    if "UNSUPPORTED_PROVENANCE" in guard_status:
        labels.append("SUPPORT_PROVENANCE_BLOCKED")
    if (
        predicted_delta < 0.0
        and -predicted_delta <= OBJECTIVE_EQUIVALENCE_TOLERANCE
    ):
        labels.append("SINGLE_STEP_TOLERANCE_BLOCKED")
    if (
        "SUPPORTED_BUT_DECISION_UNRELIABLE" in guard_status
        and -predicted_delta > OBJECTIVE_EQUIVALENCE_TOLERANCE
    ):
        labels.append("GUARD_BLOCKED_TRUE_IMPROVEMENT")
    if str(row["policy_decision"]) == "STOP":
        labels.append("EXPLORATION_STOPPED_BEFORE_REACHING_CANDIDATE")
    if not labels:
        labels.append("GUARD_BLOCKED_TRUE_IMPROVEMENT")
    return tuple(dict.fromkeys(labels))


def missed_round_root_cause(
    candidate_audit: pd.DataFrame,
    history: pd.DataFrame,
    guard_audits: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify all factorial missed candidates without changing a decision."""

    missed = candidate_audit.loc[candidate_audit["missed_improvement"].astype(bool)].copy()
    g0_bounds: dict[tuple[str, int], list[float]] = {}
    guard_details: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for policy_id, guard in guard_audits.items():
        local = guard.loc[guard["decision_guard_status"].notna()].copy()
        for item in local.to_dict(orient="records"):
            key = (
                str(item["subject_id"]) + "__" + str(item["scenario_name"]),
                str(policy_id),
                int(item["iteration"]),
                str(item["trajectory_id"]),
            )
            guard_details[key] = item
            if str(item.get("guard_id")) == "G0_CURRENT_GLOBAL_MAX":
                g0_bounds.setdefault((key[0], key[2]), []).append(
                    float(item["validation_uncertainty_bound"])
                )

    # Build stopping-rule counterfactuals only from states that the four
    # factorial replays actually visited.  S0 and S2 are compared within the
    # same guard, current alpha, and candidate alpha; no truth label enters
    # this lookup.  When S2 already stopped before a later S0 state, its answer
    # is deterministically false and is labelled as such below.
    state_decisions: dict[
        tuple[str, str, int, tuple[float, ...], tuple[float, ...], str], bool
    ] = {}
    s2_stop_iteration: dict[tuple[str, str], int] = {}
    for observed in candidate_audit.to_dict(orient="records"):
        observed_policy = str(observed["policy_id"])
        factorial_id = _FACTORIAL_BY_POLICY[observed_policy][0]
        guard_factor = "G0" if "_G0_" in factorial_id else "G2"
        stopping_factor = "S0" if "_S0" in factorial_id else "S2"
        observed_case = str(observed["case_id"])
        observed_iteration = int(observed["iteration"])
        observed_current = _current_alpha_for_iteration(
            history, observed_case, observed_policy, observed_iteration
        )
        observed_candidate = tuple(
            round(float(observed[f"alpha_{axis}"]), 12)
            for axis in ("hip", "knee", "phase")
        )
        decision_key = (
            observed_case,
            guard_factor,
            observed_iteration,
            tuple(round(value, 12) for value in observed_current),
            observed_candidate,
            stopping_factor,
        )
        would_continue = str(observed["policy_decision"]) != "STOP"
        previous = state_decisions.get(decision_key)
        if previous is not None and previous != would_continue:
            raise RuntimeError("inconsistent stopping decision for identical state")
        state_decisions[decision_key] = would_continue
        if stopping_factor == "S2" and not would_continue:
            stop_key = (observed_case, guard_factor)
            s2_stop_iteration[stop_key] = min(
                observed_iteration,
                s2_stop_iteration.get(stop_key, observed_iteration),
            )
    rows: list[dict[str, Any]] = []
    for item in missed.to_dict(orient="records"):
        case_id = str(item["case_id"])
        policy_id = str(item["policy_id"])
        iteration = int(item["iteration"])
        details = guard_details[(case_id, policy_id, iteration, str(item["trajectory_id"]))]
        current = _current_alpha_for_iteration(history, case_id, policy_id, iteration)
        labels = _mechanism_labels(item)
        g0_candidates = g0_bounds.get((case_id, iteration), [])
        g0_bound = float(g0_candidates[0]) if g0_candidates else np.nan
        evidence_role = _FACTORIAL_BY_POLICY[policy_id][1]
        factorial_id = _FACTORIAL_BY_POLICY[policy_id][0]
        guard_factor = "G0" if "_G0_" in factorial_id else "G2"
        candidate_alpha = tuple(
            round(float(item[f"alpha_{axis}"]), 12)
            for axis in ("hip", "knee", "phase")
        )
        decision_key_prefix = (
            case_id,
            guard_factor,
            iteration,
            tuple(round(value, 12) for value in current),
            candidate_alpha,
        )
        s0_key = (*decision_key_prefix, "S0")
        s2_key = (*decision_key_prefix, "S2")
        s0_continue = state_decisions.get(s0_key)
        s2_continue = state_decisions.get(s2_key)
        s0_basis = "MATCHED_OBSERVED_FACTORIAL_STATE" if s0_continue is not None else "STATE_NOT_OBSERVED"
        s2_basis = "MATCHED_OBSERVED_FACTORIAL_STATE" if s2_continue is not None else "STATE_NOT_OBSERVED"
        if s2_continue is None:
            stop_iteration = s2_stop_iteration.get((case_id, guard_factor))
            if stop_iteration is not None and iteration > stop_iteration:
                s2_continue = False
                s2_basis = "S2_ALREADY_STOPPED_BEFORE_THIS_S0_STATE"
        rows.append(
            {
                "case_id": case_id,
                "subject_id": item["subject_id"],
                "scenario": item["scenario_name"],
                "policy_id": policy_id,
                "factorial_variant_id": _FACTORIAL_BY_POLICY[policy_id][0],
                "evidence_role": evidence_role,
                "historical_prospective_primary": evidence_role == "HISTORICAL_PROSPECTIVE_PRIMARY",
                "iteration": iteration,
                "current_alpha_hip": current[0],
                "current_alpha_knee": current[1],
                "current_alpha_phase": current[2],
                "candidate_alpha_hip": item["alpha_hip"],
                "candidate_alpha_knee": item["alpha_knee"],
                "candidate_alpha_phase": item["alpha_phase"],
                "deltaJ_pred": item["delta_J_pred"],
                "deltaJ_truth_posthoc": item["delta_J_truth"],
                "equivalence_tolerance_contribution": OBJECTIVE_EQUIVALENCE_TOLERANCE,
                "G0_uncertainty_contribution": g0_bound,
                "G0_contribution_role": (
                    "ACTUAL_VARIANT_BOUND"
                    if str(details.get("guard_id")) == "G0_CURRENT_GLOBAL_MAX"
                    else "SAME_CASE_ITERATION_OTHER_G0_PATH_NOT_STATE_IDENTICAL"
                ),
                "G2_uncertainty_contribution": LOCAL_P95,
                "model_supported": bool(details["model_supported"]),
                "domain_coverage": float(details["domain_coverage"]),
                "S0_would_continue": s0_continue,
                "S0_counterfactual_basis": s0_basis,
                "S2_would_continue": s2_continue,
                "S2_counterfactual_basis": s2_basis,
                "guard_rejection_reason": item["guard_status"],
                "mechanism_components": ";".join(labels),
                "rejection_mechanism": (
                    labels[0] if len(labels) == 1 else "MULTIPLE_FACTORS"
                ),
                "truth_fed_back_to_historical_decision": False,
                "data_role": AUDIT_DATA_ROLE,
            }
        )
    output = pd.DataFrame(rows)
    summary = (
        output.groupby(
            ["historical_prospective_primary", "rejection_mechanism"],
            as_index=False,
            sort=False,
        )
        .agg(
            missed_candidate_count=("candidate_alpha_hip", "count"),
            affected_case_count=("case_id", "nunique"),
            affected_round_count=("iteration", "nunique"),
        )
    )
    summary["prospective_conclusion"] = PROSPECTIVE_CONCLUSION
    summary["prospective_conclusion_revised"] = False
    return output, summary


def detailed_small_step_audit(
    detected_paths: pd.DataFrame,
    prediction_truth_by_case: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand every detected path to the fixed 1..5 formal-neighbor sequence."""

    detected = detected_paths.loc[
        detected_paths["small_step_accumulation_case"].astype(bool)
    ].copy()
    if len(detected) != 9:
        raise RuntimeError("expected exactly nine prospective accumulation paths")
    detail_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for path_number, item in enumerate(detected.to_dict(orient="records"), start=1):
        case_id = str(item["case_id"])
        coordinate = str(item["coordinate"])
        direction = str(item["direction"])
        sign = 1.0 if direction == "POSITIVE" else -1.0
        index = _COORDINATE_INDEX[coordinate]
        step_size = _GRID_STEPS[coordinate]
        table = prediction_truth_by_case[case_id]
        lookup = {
            tuple(round(float(row[column]), 12) for column in _ALPHA_COLUMNS): row
            for row in table.to_dict(orient="records")
        }
        start = (0.0, 0.0, 0.0)
        values: list[Mapping[str, Any]] = []
        for step_number in range(0, 6):
            key_list = list(start)
            key_list[index] += sign * step_number * step_size
            key = tuple(round(value, 12) for value in key_list)
            if key not in lookup:
                raise RuntimeError(f"small-step path point missing: {case_id}/{key}")
            values.append(lookup[key])
        start_pred = float(values[0]["J_pred"])
        start_truth = float(values[0]["J_truth"])
        truth_minimum_steps = next(
            (
                step
                for step in range(1, 6)
                if float(values[step]["J_truth"]) - start_truth
                < -OBJECTIVE_EQUIVALENCE_TOLERANCE
            ),
            None,
        )
        predicted_cumulative = []
        truth_cumulative = []
        for step_number in range(1, 6):
            point = values[step_number]
            previous = values[step_number - 1]
            single_pred = float(point["J_pred"] - previous["J_pred"])
            single_truth = float(point["J_truth"] - previous["J_truth"])
            cumulative_pred = float(point["J_pred"] - start_pred)
            cumulative_truth = float(point["J_truth"] - start_truth)
            predicted_cumulative.append(cumulative_pred)
            truth_cumulative.append(cumulative_truth)
            detail_rows.append(
                {
                    "path_id": f"prospective_accumulation_{path_number:02d}",
                    "case_id": case_id,
                    "coordinate": coordinate,
                    "direction": direction,
                    "detected_bundle_length": int(item["bundle_length"]),
                    "step_number": step_number,
                    "step_alpha_hip": float(point["hip_delta"]),
                    "step_alpha_knee": float(point["knee_delta"]),
                    "step_alpha_phase": float(point["phase_delta"]),
                    "J_pred": float(point["J_pred"]),
                    "J_truth_posthoc": float(point["J_truth"]),
                    "single_step_deltaJ_pred": single_pred,
                    "single_step_deltaJ_truth": single_truth,
                    "cumulative_endpoint_deltaJ_pred": cumulative_pred,
                    "cumulative_endpoint_deltaJ_truth": cumulative_truth,
                    "single_step_pred_passes_0p005": single_pred < -OBJECTIVE_EQUIVALENCE_TOLERANCE,
                    "single_step_truth_passes_0p005": single_truth < -OBJECTIVE_EQUIVALENCE_TOLERANCE,
                    "minimum_steps_for_truth_cumulative_gt_0p005": truth_minimum_steps,
                    "within_original_detected_bundle": step_number <= int(item["bundle_length"]),
                    "same_formal_parameter_direction": True,
                    "mixed_axis_or_turn_required": False,
                    "truth_used_to_modify_policy": False,
                    "data_role": AUDIT_DATA_ROLE,
                }
            )
        declared_length = int(item["bundle_length"])
        direction_correct = bool(
            np.sign(predicted_cumulative[declared_length - 1])
            == np.sign(truth_cumulative[declared_length - 1])
        )
        magnitude_ordering = bool(
            np.all(np.diff(-np.asarray(predicted_cumulative[:declared_length])) >= -1e-12)
            and np.all(np.diff(-np.asarray(truth_cumulative[:declared_length])) >= -1e-12)
        )
        for length in (2, 3, 5):
            pred = predicted_cumulative[length - 1]
            truth = truth_cumulative[length - 1]
            residual_rows.append(
                {
                    "path_id": f"prospective_accumulation_{path_number:02d}",
                    "case_id": case_id,
                    "coordinate": coordinate,
                    "direction": direction,
                    "bundle_length": length,
                    "predicted_endpoint_deltaJ": pred,
                    "truth_endpoint_deltaJ_posthoc": truth,
                    "e_deltaJ_bundle_posthoc": abs(pred - truth),
                    "cumulative_direction_correct": np.sign(pred) == np.sign(truth),
                    "declared_path_direction_correct": direction_correct,
                    "declared_path_magnitude_ordering_correct": magnitude_ordering,
                    "prediction_usefulness": (
                        "CUMULATIVE_SIGNAL_PRESENT"
                        if direction_correct and magnitude_ordering
                        else "CUMULATIVE_MODEL_UNRELIABLE"
                    ),
                    "same_formal_parameter_direction": True,
                    "mixed_axis_or_turn_required": False,
                    "calibration_role": "POST_HOC_CHARACTERIZATION_NOT_FUTURE_CALIBRATION",
                    "future_bundle_uncertainty_updated": False,
                    "data_role": AUDIT_DATA_ROLE,
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(residual_rows)


def _alpha_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return tuple(round(float(row[column]), 12) for column in _ALPHA_COLUMNS)


def _location_class(
    coordinate: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> str:
    index = _COORDINATE_INDEX[coordinate]
    lower, upper = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[_BOUND_KEYS[coordinate]]
    if np.isclose(start[index], lower, atol=1e-12, rtol=0.0) or np.isclose(
        end[index], lower, atol=1e-12, rtol=0.0
    ):
        return "LOWER_BOUNDARY"
    if np.isclose(start[index], upper, atol=1e-12, rtol=0.0) or np.isclose(
        end[index], upper, atol=1e-12, rtol=0.0
    ):
        return "UPPER_BOUNDARY"
    return "INTERIOR"


def _bundle_identity(
    coordinate: str,
    direction: str,
    length: int,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[str, str]:
    payload = (
        f"{BUNDLE_PROTOCOL_ID}|{coordinate}|{direction}|{length}|{start}|{end}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"dbv_bundle_{digest[:24]}", digest


def enumerate_bundle_pair_universe(parameter_lattice: pd.DataFrame) -> pd.DataFrame:
    """Use generator geometry only; prediction and truth are absent by design."""

    required = {"trajectory_id", *_ALPHA_COLUMNS, "geometrically_admissible"}
    missing = required.difference(parameter_lattice.columns)
    if missing:
        raise ValueError(f"bundle lattice missing columns: {sorted(missing)}")
    if not parameter_lattice["geometrically_admissible"].astype(bool).all():
        raise ValueError("bundle plan accepts geometry-valid lattice only")
    lookup = {
        _alpha_key(row): str(row["trajectory_id"])
        for row in parameter_lattice.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for coordinate, direction, length in itertools.product(
        ("hip", "knee", "phase"), ("NEGATIVE", "POSITIVE"), (2, 3, 5)
    ):
        index = _COORDINATE_INDEX[coordinate]
        signed_step = _GRID_STEPS[coordinate] * (
            -1.0 if direction == "NEGATIVE" else 1.0
        )
        for start, start_id in lookup.items():
            path = []
            for step_number in range(length + 1):
                point = list(start)
                point[index] += signed_step * step_number
                path.append(tuple(round(value, 12) for value in point))
            if any(point not in lookup for point in path):
                continue
            end = path[-1]
            bundle_id, selection_hash = _bundle_identity(
                coordinate, direction, length, start, end
            )
            rows.append(
                {
                    "bundle_pair_id": bundle_id,
                    "selection_hash": selection_hash,
                    "coordinate": coordinate,
                    "direction": direction,
                    "bundle_length": length,
                    "location_class": _location_class(coordinate, start, end),
                    "start_trajectory_id": start_id,
                    "endpoint_trajectory_id": lookup[end],
                    "intermediate_trajectory_ids": ";".join(
                        lookup[point] for point in path[1:-1]
                    ),
                    "start_alpha_hip": start[0],
                    "start_alpha_knee": start[1],
                    "start_alpha_phase": start[2],
                    "endpoint_alpha_hip": end[0],
                    "endpoint_alpha_knee": end[1],
                    "endpoint_alpha_phase": end[2],
                    "formal_grid_step": _GRID_STEPS[coordinate],
                    "endpoint_distance_formal_steps": length,
                    "formal_neighbor_continuous": True,
                    "direction_consistent": True,
                    "geometrically_admissible_path": True,
                    "generator_bounds_expanded": False,
                    "prediction_used_for_plan": False,
                    "truth_used_for_plan": False,
                    "prospective_error_used_for_plan": False,
                }
            )
    output = pd.DataFrame(rows)
    if output["bundle_pair_id"].duplicated().any():
        raise RuntimeError("bundle validation universe contains duplicate IDs")
    return output


def build_designated_bundle_pair_plan(
    parameter_lattice: pd.DataFrame,
    *,
    pairs_per_stratum: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash-select balanced bundle pairs before any future bundle truth."""

    if pairs_per_stratum < 1:
        raise ValueError("pairs_per_stratum must be positive")
    universe = enumerate_bundle_pair_universe(parameter_lattice)
    group_columns = ["coordinate", "direction", "bundle_length", "location_class"]
    selected_frames: list[pd.DataFrame] = []
    strata_rows: list[dict[str, Any]] = []
    for keys, group in universe.groupby(group_columns, sort=True):
        ordered = group.sort_values("selection_hash", kind="mergesort")
        if len(ordered) < pairs_per_stratum:
            raise RuntimeError(f"bundle-plan stratum too small: {keys}")
        selected = ordered.head(pairs_per_stratum).copy()
        selected["within_stratum_hash_rank"] = np.arange(1, len(selected) + 1)
        selected_frames.append(selected)
        coordinate, direction, length, location = keys
        strata_rows.append(
            {
                "coordinate": coordinate,
                "direction": direction,
                "bundle_length": length,
                "location_class": location,
                "universe_pair_count": len(group),
                "planned_pair_count": len(selected),
                "selection_rule": "LOWEST_SHA256_WITHIN_GEOMETRY_DIRECTION_LENGTH_LOCATION_STRATUM",
                "prediction_used_for_selection": False,
                "truth_used_for_selection": False,
                "prospective_error_used_for_selection": False,
            }
        )
    plan = pd.concat(selected_frames, ignore_index=True, sort=False).sort_values(
        "bundle_pair_id", kind="mergesort", ignore_index=True
    )
    plan["predicted_delta_J"] = ""
    plan["truth_delta_J"] = ""
    plan["e_deltaJ_bundle"] = ""
    plan["outcome_status"] = BUNDLE_OUTCOME_STATUS
    plan["calibration_truth_generated_in_this_task"] = False
    plan["data_role"] = "FUTURE_INDEPENDENT_BUNDLE_CALIBRATION_PLAN_ONLY"
    strata = pd.DataFrame(strata_rows).sort_values(
        group_columns, kind="mergesort", ignore_index=True
    )
    return plan, strata


def stopping_removed_trial_value_audit(
    summaries: pd.DataFrame,
    histories: pd.DataFrame,
    exploration: pd.DataFrame,
) -> pd.DataFrame:
    """Compare S2 with S0 under each fixed guard using post-hoc labels only."""

    comparisons = (
        ("G0", "P2_V1_G0_C0_S0", "POST_HOC_A2_G0_C0_S2"),
        ("G2", "POST_HOC_A1_G2_C0_S0", "P2_V2A_G2_C0_S2"),
    )
    rows: list[dict[str, Any]] = []
    for guard_id, s0_policy, s2_policy in comparisons:
        for case_id in sorted(summaries["case_id"].unique()):
            s0_summary = summaries.loc[
                summaries["case_id"].eq(case_id)
                & summaries["policy_id"].eq(s0_policy)
            ].iloc[0]
            s2_summary = summaries.loc[
                summaries["case_id"].eq(case_id)
                & summaries["policy_id"].eq(s2_policy)
            ].iloc[0]
            s0_history = histories.loc[
                histories["case_id"].eq(case_id)
                & histories["policy_id"].eq(s0_policy)
            ].sort_values("iteration")
            s2_history = histories.loc[
                histories["case_id"].eq(case_id)
                & histories["policy_id"].eq(s2_policy)
            ].sort_values("iteration")
            fixed_k_triggered = str(s2_summary["stop_reason"]) == "STOP_DECISION_VALUE_ZERO_RUN_K"
            stop_after = int(s2_history["iteration"].max()) if not s2_history.empty else 0
            removed = s0_history.loc[s0_history["iteration"].astype(int).gt(stop_after)]
            later_useful = bool(
                not removed.empty
                and (
                    removed["accepted_improvement"].astype(bool).any()
                    or (
                        removed["delta_J_actual"].astype(float)
                        < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                    ).any()
                )
            )
            if not fixed_k_triggered:
                classification = "INDETERMINATE"
            elif later_useful:
                classification = "TRUNCATED_USEFUL_EXPLORATION_CHAIN"
            else:
                classification = "REMOVED_LOW_VALUE_EXPLORATION"
            s2_explore = exploration.loc[
                exploration["case_id"].eq(case_id)
                & exploration["policy_id"].eq(s2_policy)
            ].sort_values("iteration")
            trigger_features = s2_explore.iloc[-1] if not s2_explore.empty else None
            rows.append(
                {
                    "case_id": case_id,
                    "guard_context": guard_id,
                    "S0_policy_id": s0_policy,
                    "S2_policy_id": s2_policy,
                    "S2_first_early_stop_after_iteration": stop_after if fixed_k_triggered else np.nan,
                    "S2_fixed_K_triggered": fixed_k_triggered,
                    "S0_executed_trials": int(s0_summary["number_of_executed_trials"]),
                    "S2_executed_trials": int(s2_summary["number_of_executed_trials"]),
                    "removed_trial_count": int(len(removed)),
                    "trigger_support_growth": (
                        float(trigger_features["support_growth"])
                        if trigger_features is not None
                        else np.nan
                    ),
                    "trigger_information_gain": (
                        float(trigger_features["information_gain"])
                        if trigger_features is not None
                        else np.nan
                    ),
                    "trigger_parameter_change_observed": (
                        bool(trigger_features["parameter_change_observed"])
                        if trigger_features is not None
                        else np.nan
                    ),
                    "trigger_prediction_map_change_observed": (
                        bool(trigger_features["prediction_map_change_observed"])
                        if trigger_features is not None
                        else np.nan
                    ),
                    "trigger_exploit_eligibility_after": (
                        bool(trigger_features["exploit_eligibility_after"])
                        if trigger_features is not None
                        else np.nan
                    ),
                    "later_useful_action_under_S0": later_useful,
                    "later_S0_exploit_count": int(removed["trial_purpose"].eq("EXPLOIT").sum()) if not removed.empty else 0,
                    "later_S0_accepted_improvement_count": int(removed["accepted_improvement"].astype(bool).sum()) if not removed.empty else 0,
                    "removed_trial_value_classification": classification,
                    "future_action_used_by_historical_stopping": False,
                    "data_role": AUDIT_DATA_ROLE,
                }
            )
    return pd.DataFrame(rows)


def premature_stop_root_cause(
    historical_summary: pd.DataFrame,
    historical_failure: pd.DataFrame,
    historical_candidates: pd.DataFrame,
) -> pd.DataFrame:
    premature = historical_failure.loc[
        historical_failure["failure_mode"].eq("PREMATURE_CONSERVATIVE_STOP")
        & historical_failure["observed"].astype(bool)
    ].copy()
    if len(premature) != 24:
        raise RuntimeError("expected the immutable 24 prospective premature stops")
    rows: list[dict[str, Any]] = []
    for item in premature.to_dict(orient="records"):
        case_id, policy_id = str(item["case_id"]), str(item["policy_id"])
        summary = historical_summary.loc[
            historical_summary["case_id"].eq(case_id)
            & historical_summary["policy_id"].eq(policy_id)
        ].iloc[0]
        candidates = historical_candidates.loc[
            historical_candidates["case_id"].eq(case_id)
            & historical_candidates["policy_id"].eq(policy_id)
            & historical_candidates["missed_improvement"].astype(bool)
        ].copy()
        stop_candidates = candidates.loc[candidates["policy_decision"].eq("STOP")]
        if not stop_candidates.empty:
            candidates = stop_candidates.loc[
                stop_candidates["iteration"].eq(stop_candidates["iteration"].max())
            ]
        labels: list[str] = []
        for candidate in candidates.to_dict(orient="records"):
            labels.extend(_mechanism_labels(candidate))
        labels = list(dict.fromkeys(labels))
        if str(summary["stop_reason"]) == "STOP_DECISION_VALUE_ZERO_RUN_K":
            primary = "EXPLORATION_STOPPED_BEFORE_REACHING_CANDIDATE"
        elif any("UNSUPPORTED_PROVENANCE" in str(value) for value in candidates.get("guard_status", [])):
            primary = "SUPPORT_PROVENANCE_BLOCKED"
        elif labels:
            primary = labels[0] if len(labels) == 1 else "MULTIPLE_FACTORS"
        else:
            primary = "INDETERMINATE"
        rows.append(
            {
                "case_id": case_id,
                "policy_id": policy_id,
                "stop_reason": summary["stop_reason"],
                "executed_trials": int(summary["number_of_executed_trials"]),
                "missed_candidate_count_at_or_before_stop": len(candidates),
                "mechanism_components": ";".join(labels),
                "primary_stop_root_cause": primary,
                "prospective_conclusion": PROSPECTIVE_CONCLUSION,
                "prospective_conclusion_revised": False,
                "data_role": AUDIT_DATA_ROLE,
            }
        )
    return pd.DataFrame(rows)


def root_cause_matrix(
    factorial: pd.DataFrame,
    small_steps: pd.DataFrame,
    stopping: pd.DataFrame,
    premature: pd.DataFrame,
) -> pd.DataFrame:
    mean_factorial = factorial.groupby("metric", as_index=True).mean(numeric_only=True)
    missed = mean_factorial.loc["missed_improvement_rounds"]
    final_j = mean_factorial.loc["final_best_actual_J"]
    regret = mean_factorial.loc["global_truth_regret"]
    truncations = int(
        stopping["removed_trial_value_classification"]
        .eq("TRUNCATED_USEFUL_EXPLORATION_CHAIN")
        .sum()
    )
    removed_low = int(
        stopping["removed_trial_value_classification"]
        .eq("REMOVED_LOW_VALUE_EXPLORATION")
        .sum()
    )
    signal = int(
        small_steps["prediction_usefulness"].eq("CUMULATIVE_SIGNAL_PRESENT").sum()
    )
    unreliable = int(
        small_steps["prediction_usefulness"].eq("CUMULATIVE_MODEL_UNRELIABLE").sum()
    )
    final_stopping_zero = bool(
        np.isclose(
            final_j["factorial_stopping_main_effect"], 0.0, atol=1e-12, rtol=0.0
        )
    )
    regret_stopping_zero = bool(
        np.isclose(
            regret["factorial_stopping_main_effect"], 0.0, atol=1e-12, rtol=0.0
        )
    )
    return pd.DataFrame(
        [
            {
                "problem": "HIGHER_MISSED_IMPROVEMENT_UNDER_REJECTED_V2A",
                "evidence": (
                    f"mean A3-A0={missed['A3_minus_A0_total']:.9g}; "
                    f"guard main={missed['factorial_guard_main_effect']:.9g}; "
                    f"stopping main={missed['factorial_stopping_main_effect']:.9g}; "
                    f"interaction={missed['guard_stopping_interaction']:.9g}"
                ),
                "primary_cause": "GUARD_EFFECT",
                "secondary_cause": (
                    "STOPPING_AND_INTERACTION_REDUCED_OBSERVED_OPPORTUNITY_COUNT"
                ),
                "objective_change_required": False,
                "generator_enlargement_required": False,
                "model_enlargement_required": False,
            },
            {
                "problem": "HIGHER_FINAL_J_AND_REGRET",
                "evidence": (
                    f"mean final-J A3-A0={final_j['A3_minus_A0_total']:.9g}; "
                    f"guard={final_j['factorial_guard_main_effect']:.9g}; "
                    f"stopping={final_j['factorial_stopping_main_effect']:.9g}; "
                    f"interaction={final_j['guard_stopping_interaction']:.9g}; "
                    f"regret A3-A0={regret['A3_minus_A0_total']:.9g}"
                ),
                "primary_cause": "GUARD_EFFECT",
                "secondary_cause": (
                    "NO_SEPARABLE_STOPPING_OUTCOME_EFFECT_OBSERVED"
                    if final_stopping_zero and regret_stopping_zero
                    else "GUARD_STOPPING_INTERACTION"
                ),
                "objective_change_required": False,
                "generator_enlargement_required": False,
                "model_enlargement_required": False,
            },
            {
                "problem": "SMALL_STEP_ACCUMULATION",
                "evidence": f"9 repeated paths; signal={signal}; unreliable={unreliable}",
                "primary_cause": "SINGLE_STEP_DECISION_STRUCTURE",
                "secondary_cause": "BUNDLE_UNCERTAINTY_NOT_CALIBRATED",
                "objective_change_required": False,
                "generator_enlargement_required": False,
                "model_enlargement_required": False,
            },
            {
                "problem": "EXCESS_EXPLORATION_UNDER_P2_V1",
                "evidence": f"removed-low-value comparisons={removed_low}",
                "primary_cause": "NO_DECISION_VALUE_STOPPING_IN_S0",
                "secondary_cause": "SUPPORT_GROWTH_OVERVALUED",
                "objective_change_required": False,
                "generator_enlargement_required": False,
                "model_enlargement_required": False,
            },
            {
                "problem": "OVER_EARLY_STOP_UNDER_V2A",
                "evidence": (
                    f"useful-chain truncations={truncations}; "
                    f"immutable premature-stop labels={len(premature)}; "
                    f"final-J stopping main={final_j['factorial_stopping_main_effect']:.9g}"
                ),
                "primary_cause": "GUARD_EFFECT",
                "secondary_cause": (
                    "FIXED_K_OUTCOME_HARM_NOT_IDENTIFIED_IN_THIS_FACTORIAL"
                    if final_stopping_zero
                    else "FIXED_K_STOPPING_STRUCTURE_INSUFFICIENT"
                ),
                "objective_change_required": False,
                "generator_enlargement_required": False,
                "model_enlargement_required": False,
            },
        ]
    ).assign(data_role=AUDIT_DATA_ROLE)


__all__ = [
    "AUDIT_DATA_ROLE",
    "AUDIT_ID",
    "BUNDLE_OUTCOME_STATUS",
    "BUNDLE_PROTOCOL_ID",
    "FACTORIAL_SPECS",
    "FINAL_STATUSES",
    "FINAL_STATUS_IDENTIFIED",
    "FINAL_STATUS_MORE_ANALYSIS",
    "PROSPECTIVE_CONCLUSION",
    "PROSPECTIVE_MANIFEST_SHA256",
    "PROSPECTIVE_START_COMMIT",
    "attach_factorial_identity",
    "build_designated_bundle_pair_plan",
    "detailed_small_step_audit",
    "enumerate_bundle_pair_universe",
    "factorial_decomposition",
    "missed_round_root_cause",
    "premature_stop_root_cause",
    "root_cause_matrix",
    "sha256_file",
    "stopping_removed_trial_value_audit",
    "verify_historical_reproduction",
    "verify_immutable_prospective_artifacts",
]
