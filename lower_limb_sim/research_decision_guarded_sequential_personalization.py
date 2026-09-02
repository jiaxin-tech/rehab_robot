"""Research-only decision-guarded sequential personalization.

This module joins the existing diagnostic initial-identification, five-
parameter estimator, full geometrically admissible prediction map, fixed
support provenance, validation-only pairwise decision uncertainty, local
exploitation, and information-driven frontier exploration.  Virtual truth is
available only through a selection-gated execution oracle and a separate
post-policy evaluation layer.  Nothing here creates a human-ready model,
formal reliability threshold, robot-personalization approval, or motion API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    identification_lower_bounds,
    identification_upper_bounds,
)
from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    DIAGNOSTIC_INITIAL_MODEL,
    DIAGNOSTIC_ONLY,
    DiagnosticInitialModel,
    GRID_DISTANCE_DEFINITION,
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    IMPROVE,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    NEUTRAL,
    NOT_HUMAN_READY,
    TrajectoryComponentCache,
    WORSE,
    build_predicted_map,
    classify_improvement_direction,
    diagnostic_model_from_sequential_result,
    evaluate_truth_map,
)
from .dynamic_subject import get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    validate_active_reference_file,
)
from .geometry_error_metrics import StateDomainBounds
from .identifiability_analysis import numerical_sensitivity_matrix
from .initial_identification_acceptance_rule import build_validation_observations
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
    MechanicalObjectiveResult,
    MechanicalTorqueMetrics,
    compute_torque_metrics,
    evaluate_mechanical_objective,
)
from .parameter_estimator import (
    PARAMETER_NAMES,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    estimate_subject_parameters,
    measured_joint_torque,
    predict_joint_torque,
)
from .run_model_mismatch_experiment import project_estimator_inputs
from .safeguarded_sequential_initial_identification import (
    SequentialIdentificationResult,
    VirtualIdentificationOracle,
    default_virtual_patient_envelope,
    run_sequential_initial_identification,
)
from .sequential_personalization import (
    MAX_EXECUTED_TRIALS,
    SearchAlpha,
    Stage45CVirtualTruthOracle,
    TrustRegionSteps,
    accept_actual_trial,
    build_coordinate_neighborhood,
    shrink_steps,
)


PROTOCOL_ID = "RESEARCH_ONLY_DECISION_GUARDED_SEQUENTIAL_PERSONALIZATION_V1"
DECISION_UNCERTAINTY_VERSION = "RESEARCH_DECISION_UNCERTAINTY_V1"
DECISION_GUARD_STATUS = "RESEARCH_ONLY_VIRTUAL_CANDIDATE"
GLOBAL_MODEL_RELIABILITY_RULE_STATUS = "GLOBAL_MODEL_RELIABILITY_RULE_NOT_FROZEN"
INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS = (
    "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW"
)
REAL_ROBOT_HARD_SAFEGUARD_STATUS = "NOT_DEFINED_NOT_APPROVED"
RESEARCH_ONLY = "RESEARCH_ONLY"
NOT_APPROVED_FOR_ROBOT_PERSONALIZATION = "NOT_APPROVED_FOR_ROBOT_PERSONALIZATION"
RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET = MAX_EXECUTED_TRIALS

POLICY_SUPPORTED_ONLY_GREEDY = "P0_SUPPORTED_ONLY_GREEDY"
POLICY_DECISION_GUARDED_EXPLOIT_ONLY = "P1_DECISION_GUARDED_EXPLOIT_ONLY"
POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT = (
    "P2_DECISION_GUARDED_EXPLORE_EXPLOIT"
)
POLICY_IDS = (
    POLICY_SUPPORTED_ONLY_GREEDY,
    POLICY_DECISION_GUARDED_EXPLOIT_ONLY,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
)

TRIAL_PURPOSE_EXPLOIT = "EXPLOIT"
TRIAL_PURPOSE_EXPLORE = "EXPLORE"
SUPPORTED_BUT_DECISION_UNRELIABLE = "SUPPORTED_BUT_DECISION_UNRELIABLE"
RESEARCH_EXPLOIT_ELIGIBLE = "RESEARCH_EXPLOIT_ELIGIBLE"
UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE = (
    "UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE"
)
CURRENT_BEST_NOT_A_CANDIDATE = "CURRENT_BEST_NOT_A_CANDIDATE"
NO_INDEPENDENT_VALIDATION_EVIDENCE = "NO_INDEPENDENT_VALIDATION_EVIDENCE"
GEOMETRICALLY_INADMISSIBLE = "GEOMETRICALLY_INADMISSIBLE"
EXECUTED_FALSE_IMPROVEMENT = "EXECUTED_FALSE_IMPROVEMENT"
MODEL_RELIABILITY_DEGRADED = "MODEL_RELIABILITY_DEGRADED"

STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER = (
    "STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER"
)
STOP_MAX_PERSONALIZATION_TRIALS = "STOP_MAX_PERSONALIZATION_TRIALS"
STOP_MODEL_UPDATE_FAILURE = "STOP_MODEL_UPDATE_FAILURE"
STOP_MODEL_ADEQUACY_DEGRADED = "STOP_MODEL_ADEQUACY_DEGRADED"
STOP_PATIENT_ENVELOPE_BOUNDARY = "STOP_PATIENT_ENVELOPE_BOUNDARY"
STOP_NO_GEOMETRICALLY_VALID_CANDIDATE = (
    "STOP_NO_GEOMETRICALLY_VALID_CANDIDATE"
)

PAIRWISE_BOUND_TYPE = "MAX_OBSERVED_VALIDATION_PAIRWISE_DELTA_J_ERROR"
PAIRWISE_BOUND_STATUS = "RESEARCH_CANDIDATE_ONLY_NOT_FORMAL_THRESHOLD"
FRONTIER_DISTANCE_ROLE = "LOCALITY_CONTROL_NOT_RELIABILITY_SCORE"
SUPPORT_ROLE = "DATA_PROVENANCE_NOT_RELIABILITY_APPROVAL"
MAP_TRUTH_ROLE = "POST_POLICY_EVALUATION_ONLY_NOT_POLICY_INPUT"

_STATE_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
)
_DOMAIN_COLUMNS = (
    "q_hip_est_rad",
    "q_knee_est_rad",
    "dq_hip_est_rad_s",
    "dq_knee_est_rad_s",
    "ddq_hip_est_rad_s2",
    "ddq_knee_est_rad_s2",
)


@dataclass(frozen=True)
class InitialResearchState:
    subject_id: str
    scenario_name: str
    selected_trial_id: int
    parameters: Mapping[str, float]
    fitting_data: pd.DataFrame
    domain_data: pd.DataFrame
    validation_data: pd.DataFrame
    sequential_result: SequentialIdentificationResult
    model_status: str = DIAGNOSTIC_ONLY
    research_status: str = RESEARCH_ONLY
    human_readiness: str = NOT_HUMAN_READY
    approval_status: str = NOT_APPROVED_FOR_ROBOT_PERSONALIZATION


@dataclass(frozen=True)
class ResearchDecisionUncertainty:
    case_id: str
    iteration: int
    pairwise_audit: pd.DataFrame
    maximum_observed_e_delta_j: float
    p95_observed_e_delta_j: float
    p99_observed_e_delta_j: float
    validation_pair_count: int
    bound_used_by_guard: float
    bound_type: str = PAIRWISE_BOUND_TYPE
    bound_status: str = PAIRWISE_BOUND_STATUS


@dataclass(frozen=True)
class SelectionToken:
    token: str
    trajectory_id: str
    trial_purpose: str
    serial: int


@dataclass(frozen=True)
class PolicyRunResult:
    subject_id: str
    scenario_name: str
    policy_id: str
    trial_history: pd.DataFrame
    decision_guard_audit: pd.DataFrame
    parameter_history: pd.DataFrame
    prediction_map_history: pd.DataFrame
    known_region_history: pd.DataFrame
    uncertainty_history: pd.DataFrame
    uncertainty_pairwise_audit: pd.DataFrame
    exploration_information_gain: pd.DataFrame
    false_improvement_audit: pd.DataFrame
    summary: Mapping[str, Any]
    initial_prediction_map: pd.DataFrame
    final_prediction_map: pd.DataFrame
    truth_access_audit: Mapping[str, Any]


def _dataframe_sha256(dataframe: pd.DataFrame) -> str:
    payload = dataframe.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _domain_from_data(dataframe: pd.DataFrame) -> StateDomainBounds:
    values = dataframe.loc[:, _STATE_COLUMNS].to_numpy(dtype=float)
    selected = values[np.isfinite(values).all(axis=1)]
    if selected.size == 0:
        raise ValueError("current identification/adaptation data has no finite states")
    return StateDomainBounds(
        columns=_DOMAIN_COLUMNS,
        lower=tuple(np.min(selected, axis=0)),
        upper=tuple(np.max(selected, axis=0)),
        valid_training_samples=int(len(selected)),
    )


def _model_for_iteration(
    state: InitialResearchState,
    parameters: Mapping[str, float],
    domain_data: pd.DataFrame,
    iteration: int,
) -> DiagnosticInitialModel:
    return DiagnosticInitialModel(
        subject_id=state.subject_id,
        scenario_name=state.scenario_name,
        selected_trial_id=state.selected_trial_id + int(iteration),
        parameters=dict(parameters),
        identification_domain=_domain_from_data(domain_data),
        identification_dataset_sha256=_dataframe_sha256(domain_data),
    )


def build_initial_research_state(
    subject_id: str,
    scenario_name: str,
) -> InitialResearchState:
    """Reconstruct the approved diagnostic source without freezing theta_0."""

    sequential = run_sequential_initial_identification(
        VirtualIdentificationOracle(subject_id, scenario_name),
        default_virtual_patient_envelope(),
        stop_rule=None,
    )
    diagnostic = diagnostic_model_from_sequential_result(sequential)
    executed = sequential.executed_identification_data.loc[
        sequential.executed_identification_data["trial_id"]
        .astype(int)
        .le(diagnostic.selected_trial_id)
    ].copy()
    fitting = executed.loc[
        executed["within_identification_role"].astype(str).eq("train")
    ].copy()
    if fitting.empty:
        raise RuntimeError("diagnostic initial state has no fitting rows")
    # The sequential-identification table predates the estimator projection's
    # explicit invalid-reason field.  It contains only valid observations, so
    # add the missing schema field without changing any measurement value.
    if "invalid_reason" not in fitting:
        fitting["invalid_reason"] = ""
    if "invalid_reason" not in executed:
        executed["invalid_reason"] = ""
    fitting = project_estimator_inputs(fitting)
    domain = project_estimator_inputs(executed)
    validation = build_validation_observations(subject_id, scenario_name)
    return InitialResearchState(
        subject_id=subject_id,
        scenario_name=scenario_name,
        selected_trial_id=diagnostic.selected_trial_id,
        parameters=dict(diagnostic.parameters),
        fitting_data=fitting,
        domain_data=domain,
        validation_data=validation,
        sequential_result=sequential,
    )


def _objective_from_metrics(
    trajectory_id: str,
    metrics: MechanicalTorqueMetrics,
    reference_metrics: MechanicalTorqueMetrics,
) -> float:
    return float(
        evaluate_mechanical_objective(
            trajectory_id=trajectory_id,
            metrics=metrics,
            reference_metrics=reference_metrics,
            hip_rms_deviation_deg=0.0,
            knee_rms_deviation_deg=0.0,
        ).mechanical_cost_j_rms
    )


def evaluate_validation_pairwise_uncertainty(
    state: InitialResearchState,
    parameters: Mapping[str, float],
    *,
    iteration: int,
) -> ResearchDecisionUncertainty:
    """Calibrate pairwise delta-J residuals from designated validation only."""

    validation = state.validation_data.copy(deep=True).reset_index(drop=True)
    if set(validation["dataset_split"].astype(str)) != {"validation"}:
        raise ValueError("decision uncertainty accepts designated validation only")
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    safe = project_estimator_inputs(validation)
    measured_hip, measured_knee = measured_joint_torque(safe, L1, L2)
    predicted_hip, predicted_knee = predict_joint_torque(
        safe, template, parameters, L1
    )
    item_rows: list[dict[str, Any]] = []
    groups = list(validation.groupby("trajectory_id", sort=False))
    if len(groups) < 2:
        raise RuntimeError("pairwise decision uncertainty needs >=2 validation items")
    actual_metrics: dict[str, MechanicalTorqueMetrics] = {}
    predicted_metrics: dict[str, MechanicalTorqueMetrics] = {}
    for trajectory_id, group in groups:
        indices = group.index.to_numpy(dtype=int)
        time_s = group["time_s"].to_numpy(dtype=float)
        actual_metrics[str(trajectory_id)] = compute_torque_metrics(
            time_s, measured_hip[indices], measured_knee[indices]
        )
        predicted_metrics[str(trajectory_id)] = compute_torque_metrics(
            time_s, predicted_hip[indices], predicted_knee[indices]
        )
    reference_id = str(groups[0][0])
    item_actual: dict[str, float] = {}
    item_predicted: dict[str, float] = {}
    for trajectory_id, _ in groups:
        key = str(trajectory_id)
        item_actual[key] = _objective_from_metrics(
            key, actual_metrics[key], actual_metrics[reference_id]
        )
        item_predicted[key] = _objective_from_metrics(
            key, predicted_metrics[key], predicted_metrics[reference_id]
        )
        item_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "iteration": int(iteration),
                "validation_item_id": key,
                "validation_reference_item_id": reference_id,
                "J_actual": item_actual[key],
                "J_pred": item_predicted[key],
                "absolute_item_prediction_error": abs(
                    item_predicted[key] - item_actual[key]
                ),
                "data_role": "DESIGNATED_VALIDATION_ONLY",
                "used_for_model_fitting": False,
                "heldout_final_test": False,
            }
        )
    pair_rows: list[dict[str, Any]] = []
    ids = [str(item[0]) for item in groups]
    for current_id, candidate_id in itertools.combinations(ids, 2):
        delta_pred = item_predicted[candidate_id] - item_predicted[current_id]
        delta_actual = item_actual[candidate_id] - item_actual[current_id]
        pair_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "iteration": int(iteration),
                "current_validation_trajectory_id": current_id,
                "candidate_validation_trajectory_id": candidate_id,
                "delta_J_pred": delta_pred,
                "delta_J_actual": delta_actual,
                "e_delta_J": abs(delta_pred - delta_actual),
                "current_item_prediction_error_abs": abs(
                    item_predicted[current_id] - item_actual[current_id]
                ),
                "candidate_item_prediction_error_abs": abs(
                    item_predicted[candidate_id] - item_actual[candidate_id]
                ),
                "calibration_data_role": "DESIGNATED_VALIDATION_ONLY",
                "adaptation_executed_outcome_used_directly": False,
                "heldout_final_test_used": False,
                "bound_status": PAIRWISE_BOUND_STATUS,
            }
        )
    audit = pd.DataFrame(pair_rows)
    errors = audit["e_delta_J"].to_numpy(dtype=float)
    maximum = float(np.max(errors))
    return ResearchDecisionUncertainty(
        case_id=f"{state.subject_id}__{state.scenario_name}",
        iteration=int(iteration),
        pairwise_audit=audit,
        maximum_observed_e_delta_j=maximum,
        p95_observed_e_delta_j=float(np.percentile(errors, 95)),
        p99_observed_e_delta_j=float(np.percentile(errors, 99)),
        validation_pair_count=int(len(audit)),
        bound_used_by_guard=maximum,
    )


def uncertainty_summary_row(
    uncertainty: ResearchDecisionUncertainty,
) -> dict[str, Any]:
    audit = uncertainty.pairwise_audit
    correlation = float("nan")
    if (
        len(audit) >= 3
        and audit["current_item_prediction_error_abs"].nunique() > 1
        and audit["candidate_item_prediction_error_abs"].nunique() > 1
    ):
        correlation = float(
            audit["current_item_prediction_error_abs"].corr(
                audit["candidate_item_prediction_error_abs"], method="spearman"
            )
        )
    return {
        "case_id": uncertainty.case_id,
        "iteration": uncertainty.iteration,
        "decision_uncertainty_version": DECISION_UNCERTAINTY_VERSION,
        "validation_pair_count": uncertainty.validation_pair_count,
        "maximum_observed_validation_e_delta_J": (
            uncertainty.maximum_observed_e_delta_j
        ),
        "p95_observed_validation_e_delta_J": uncertainty.p95_observed_e_delta_j,
        "p99_observed_validation_e_delta_J": uncertainty.p99_observed_e_delta_j,
        "guard_uncertainty_bound": uncertainty.bound_used_by_guard,
        "guard_uncertainty_bound_type": uncertainty.bound_type,
        "guard_uncertainty_bound_status": uncertainty.bound_status,
        "current_vs_candidate_item_error_spearman": correlation,
        "correlation_sample_sufficient": bool(len(audit) >= 3),
        "heldout_final_test_used": False,
        "formal_threshold_created": False,
    }


def alpha_key_from_row(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        round(float(row["hip_delta"]), 12),
        round(float(row["knee_delta"]), 12),
        round(float(row["phase_delta"]), 12),
    )


def alpha_from_row(row: Mapping[str, Any]) -> SearchAlpha:
    return SearchAlpha(
        hip_delta_deg=float(row["hip_delta"]),
        knee_delta_deg=float(row["knee_delta"]),
        phase_delta=float(row["phase_delta"]),
    )


def _row_for_alpha(table: pd.DataFrame, alpha: SearchAlpha) -> pd.Series:
    selected = table.loc[
        np.isclose(table["hip_delta"], alpha.hip_delta_deg, atol=1e-12, rtol=0.0)
        & np.isclose(table["knee_delta"], alpha.knee_delta_deg, atol=1e-12, rtol=0.0)
        & np.isclose(table["phase_delta"], alpha.phase_delta, atol=1e-12, rtol=0.0)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"formal map lacks unique alpha {alpha.key()}")
    return selected.iloc[0].copy()


def local_prediction_candidates(
    prediction_map: pd.DataFrame,
    current_best: SearchAlpha,
    steps: TrustRegionSteps,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for alpha in build_coordinate_neighborhood(current_best, steps):
        try:
            rows.append(_row_for_alpha(prediction_map, alpha))
        except RuntimeError:
            continue
    if not rows:
        return pd.DataFrame(columns=prediction_map.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def apply_research_decision_guard(
    local_candidates: pd.DataFrame,
    current_best: SearchAlpha,
    uncertainty: ResearchDecisionUncertainty,
) -> pd.DataFrame:
    """Attach the research-only exploit guard; support alone can never pass."""

    if local_candidates.empty:
        return local_candidates.copy()
    output = local_candidates.copy(deep=True)
    current = _row_for_alpha(output, current_best)
    current_predicted_j = float(current["J_pred"])
    current_supported = bool(current["model_supported"])
    output["candidate_type"] = "LOCAL_EXPLOIT_NEIGHBORHOOD"
    output["current_best_J_pred"] = current_predicted_j
    output["delta_J_pred_vs_current"] = output["J_pred"] - current_predicted_j
    output["predicted_improvement_magnitude"] = -output["delta_J_pred_vs_current"]
    output["validation_uncertainty_bound"] = uncertainty.bound_used_by_guard
    output["algorithm_equivalence_tolerance"] = OBJECTIVE_EQUIVALENCE_TOLERANCE
    output["improvement_margin"] = (
        output["predicted_improvement_magnitude"]
        - uncertainty.bound_used_by_guard
        - OBJECTIVE_EQUIVALENCE_TOLERANCE
    )
    output["support_role"] = SUPPORT_ROLE
    output["distance_role"] = FRONTIER_DISTANCE_ROLE
    output["decision_guard_rule_status"] = DECISION_GUARD_STATUS
    output["validation_pair_count"] = uncertainty.validation_pair_count
    output["current_model_supported"] = current_supported
    statuses: list[str] = []
    eligible: list[bool] = []
    for row in output.to_dict(orient="records"):
        is_current = alpha_key_from_row(row) == current_best.key()
        if is_current:
            status = CURRENT_BEST_NOT_A_CANDIDATE
        elif not bool(row["geometrically_admissible"]):
            status = GEOMETRICALLY_INADMISSIBLE
        elif uncertainty.validation_pair_count < 1:
            status = NO_INDEPENDENT_VALIDATION_EVIDENCE
        elif not current_supported or not bool(row["model_supported"]):
            status = UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE
        elif float(row["improvement_margin"]) <= 0.0:
            status = SUPPORTED_BUT_DECISION_UNRELIABLE
        else:
            status = RESEARCH_EXPLOIT_ELIGIBLE
        statuses.append(status)
        eligible.append(status == RESEARCH_EXPLOIT_ELIGIBLE)
    output["decision_guard_status"] = statuses
    output["research_exploit_eligible"] = eligible
    output["support_alone_approved_exploit"] = False
    output["formal_personalization_approval"] = False
    return output


def select_exploit_candidate(
    guarded_candidates: pd.DataFrame,
    policy_id: str,
) -> pd.Series | None:
    if policy_id not in POLICY_IDS:
        raise ValueError(f"unknown policy: {policy_id}")
    if guarded_candidates.empty:
        return None
    candidates = guarded_candidates.loc[
        ~guarded_candidates["decision_guard_status"].eq(
            CURRENT_BEST_NOT_A_CANDIDATE
        )
    ].copy()
    if policy_id == POLICY_SUPPORTED_ONLY_GREEDY:
        candidates = candidates.loc[
            candidates["geometrically_admissible"].astype(bool)
            & candidates["model_supported"].astype(bool)
            & (candidates["delta_J_pred_vs_current"] < 0.0)
        ]
    else:
        candidates = candidates.loc[
            candidates["research_exploit_eligible"].astype(bool)
        ]
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["J_pred", "trajectory_id"], kind="mergesort"
    ).iloc[0].copy()


def _neighbor_keys(key: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    hip, knee, phase = key
    raw = (
        (hip + GRID_HIP_STEP_DEG, knee, phase),
        (hip - GRID_HIP_STEP_DEG, knee, phase),
        (hip, knee + GRID_KNEE_STEP_DEG, phase),
        (hip, knee - GRID_KNEE_STEP_DEG, phase),
        (hip, knee, phase + GRID_PHASE_STEP),
        (hip, knee, phase - GRID_PHASE_STEP),
    )
    return tuple(tuple(round(value, 12) for value in item) for item in raw)


def build_local_exploration_frontier(
    prediction_map: pd.DataFrame,
    executed_alpha_keys: set[tuple[float, float, float]],
) -> pd.DataFrame:
    """Return the next unexecuted formal-grid layer without skipping."""

    by_key = {
        alpha_key_from_row(row): row
        for row in prediction_map.to_dict(orient="records")
    }
    frontier_keys: set[tuple[float, float, float]] = set()
    for executed in executed_alpha_keys:
        for neighbor in _neighbor_keys(executed):
            if neighbor in by_key and neighbor not in executed_alpha_keys:
                frontier_keys.add(neighbor)
    rows = [by_key[key] for key in sorted(frontier_keys)]
    frontier = pd.DataFrame(rows)
    if frontier.empty:
        return frontier
    frontier["candidate_type"] = "LOCAL_EXPLORATION_FRONTIER"
    frontier["adjacent_formal_step"] = True
    frontier["intermediate_layer_skipped"] = False
    frontier["frontier_distance_steps"] = 1.0
    frontier["distance_role"] = FRONTIER_DISTANCE_ROLE
    frontier["support_role"] = SUPPORT_ROLE
    return frontier.reset_index(drop=True)


def _predicted_estimator_observations(
    trajectory: pd.DataFrame,
    template: Any,
    parameters: Mapping[str, float],
) -> pd.DataFrame:
    hip, knee = predict_joint_torque(trajectory, template, parameters, L1)
    force = endpoint_force_from_joint_torque(
        trajectory["q_hip_rad"].to_numpy(dtype=float),
        trajectory["q_knee_rad"].to_numpy(dtype=float),
        hip,
        knee,
        L1,
        L2,
    )
    valid = np.asarray(force.force_mapping_valid, dtype=bool)
    return pd.DataFrame(
        {
            **{
                column: trajectory[column].to_numpy(dtype=float)
                for column in _STATE_COLUMNS
            },
            "fx_observed_n": np.asarray(force.fx_robot_on_leg_n, dtype=float),
            "fz_observed_n": np.asarray(force.fz_robot_on_leg_n, dtype=float),
            "sample_valid": valid,
            "force_mapping_valid": valid,
            "wrench_is_stale": False,
            "invalid_reason": np.where(valid, "", force.invalid_reason),
        }
    )


def _information_metrics(matrix: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    minimum = float(singular[-1])
    condition = float(singular[0] / minimum) if minimum > 0.0 else float("inf")
    information = matrix.T @ matrix
    sensitivities = np.sqrt(np.maximum(np.diag(information), 0.0))
    covariance = np.linalg.pinv(information)
    uncertainty_trace = float(np.trace(covariance))
    return {
        "minimum_singular_value": minimum,
        "condition_number": condition,
        "weakest_parameter_sensitivity": float(np.min(sensitivities)),
        "uncertainty_trace_proxy": uncertainty_trace,
    }


def _log_information_gain(prior: np.ndarray, combined: np.ndarray) -> float:
    identity = np.eye(len(PARAMETER_NAMES))
    before_sign, before = np.linalg.slogdet(identity + prior.T @ prior)
    after_sign, after = np.linalg.slogdet(identity + combined.T @ combined)
    if before_sign <= 0 or after_sign <= 0:
        raise RuntimeError("information matrix log determinant is invalid")
    return float(after - before)


def _state_coverage_gain(prior: pd.DataFrame, candidate: pd.DataFrame) -> float:
    a = prior.loc[:, _STATE_COLUMNS].to_numpy(dtype=float)[::8]
    b = candidate.loc[:, _STATE_COLUMNS].to_numpy(dtype=float)[::8]
    if len(a) == 0:
        return 1.0
    combined = np.vstack((a, b))
    scale = np.std(combined, axis=0)
    scale[scale < 1e-9] = 1.0
    a = a / scale
    b = b / scale
    distances = np.full(len(b), np.inf)
    for start in range(0, len(a), 128):
        block = np.linalg.norm(b[:, None, :] - a[None, start : start + 128], axis=2)
        distances = np.minimum(distances, np.min(block, axis=1))
    return float(np.mean(distances))


def rank_exploration_frontier(
    frontier: pd.DataFrame,
    fitting_data: pd.DataFrame,
    parameters: Mapping[str, float],
) -> pd.DataFrame:
    """Rank frontier by information first; J_pred is diagnostic only."""

    if frontier.empty:
        return frontier.copy()
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    prior_matrix, prior_data, torque_scales = numerical_sensitivity_matrix(
        fitting_data,
        template,
        parameters,
        L1,
        L2,
        torque_scales_nm=(1.0, 1.0),
    )
    prior_metrics = _information_metrics(prior_matrix)
    envelope = default_virtual_patient_envelope()
    rows: list[dict[str, Any]] = []
    for row in frontier.to_dict(orient="records"):
        alpha = alpha_from_row(row)
        generated = generate_personalized_trajectory(**alpha.as_generator_parameters())
        patient_valid = envelope.contains(generated.trajectory)
        predicted = _predicted_estimator_observations(
            generated.trajectory, template, parameters
        )
        candidate_matrix, candidate_data, _ = numerical_sensitivity_matrix(
            predicted,
            template,
            parameters,
            L1,
            L2,
            torque_scales_nm=torque_scales,
        )
        combined = np.vstack((prior_matrix, candidate_matrix))
        resulting = _information_metrics(combined)
        combined_deviation = float(
            math.sqrt(
                float(generated.metadata["hip_rms_deviation_deg"]) ** 2
                + float(generated.metadata["knee_rms_deviation_deg"]) ** 2
            )
        )
        rows.append(
            {
                **row,
                "patient_envelope_valid": patient_valid,
                "hard_global_constraints_valid": bool(row["geometrically_admissible"]),
                "exploration_candidate_valid": bool(
                    patient_valid and row["geometrically_admissible"]
                ),
                "incremental_log_information_gain": _log_information_gain(
                    prior_matrix, combined
                ),
                "minimum_singular_value_before": prior_metrics[
                    "minimum_singular_value"
                ],
                "resulting_minimum_singular_value": resulting[
                    "minimum_singular_value"
                ],
                "minimum_singular_value_gain": resulting[
                    "minimum_singular_value"
                ]
                - prior_metrics["minimum_singular_value"],
                "condition_number_before": prior_metrics["condition_number"],
                "resulting_condition_number": resulting["condition_number"],
                "conditioning_improvement": prior_metrics["condition_number"]
                - resulting["condition_number"],
                "weakest_parameter_sensitivity_before": prior_metrics[
                    "weakest_parameter_sensitivity"
                ],
                "resulting_weakest_parameter_sensitivity": resulting[
                    "weakest_parameter_sensitivity"
                ],
                "weakest_parameter_sensitivity_gain": resulting[
                    "weakest_parameter_sensitivity"
                ]
                - prior_metrics["weakest_parameter_sensitivity"],
                "uncertainty_trace_before": prior_metrics[
                    "uncertainty_trace_proxy"
                ],
                "resulting_uncertainty_trace": resulting[
                    "uncertainty_trace_proxy"
                ],
                "incremental_state_regressor_coverage": _state_coverage_gain(
                    prior_data, candidate_data
                ),
                "combined_joint_deviation_deg": combined_deviation,
                "max_pull_deviation_mm": float(
                    generated.metadata["pull_max_deviation_mm"]
                ),
                "J_pred_used_as_primary_exploration_rank": False,
                "truth_used_for_exploration_rank": False,
                "frontier_distance_used_as_reliability_score": False,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        [
            "exploration_candidate_valid",
            "incremental_log_information_gain",
            "resulting_minimum_singular_value",
            "resulting_condition_number",
            "resulting_weakest_parameter_sensitivity",
            "incremental_state_regressor_coverage",
            "combined_joint_deviation_deg",
            "max_pull_deviation_mm",
            "trajectory_id",
        ],
        ascending=[False, False, False, True, False, False, True, True, True],
        kind="mergesort",
        ignore_index=True,
    )
    output["exploration_rank"] = np.arange(1, len(output) + 1)
    output["selected_for_exploration"] = False
    return output


class SelectionGatedVirtualTruthOracle:
    """Require an explicit one-candidate selection token before virtual truth."""

    def __init__(self, subject_id: str, scenario_name: str) -> None:
        self.subject_id = str(subject_id)
        self.scenario_name = str(scenario_name)
        self._backend = Stage45CVirtualTruthOracle(subject_id, scenario_name)
        self._serial = 0
        self._pending: SelectionToken | None = None
        self._executed_tokens: set[str] = set()
        self._audit: list[dict[str, Any]] = []

    @property
    def truth_calls(self) -> int:
        return self._backend.truth_calls

    @property
    def audit(self) -> pd.DataFrame:
        return pd.DataFrame(self._audit)

    def declare_selected(
        self, trajectory_id: str, trial_purpose: str
    ) -> SelectionToken:
        if self._pending is not None:
            raise RuntimeError("exactly one pending selected trajectory is allowed")
        if trial_purpose not in (
            TRIAL_PURPOSE_EXPLOIT,
            TRIAL_PURPOSE_EXPLORE,
            "REFERENCE_NORMALIZATION",
        ):
            raise ValueError("invalid virtual trial purpose")
        self._serial += 1
        payload = (
            f"{self.subject_id}|{self.scenario_name}|{trajectory_id}|"
            f"{trial_purpose}|{self._serial}"
        )
        token = SelectionToken(
            token=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            trajectory_id=str(trajectory_id),
            trial_purpose=str(trial_purpose),
            serial=self._serial,
        )
        self._pending = token
        self._audit.append(
            {
                "serial": token.serial,
                "event": "SELECTED_BEFORE_TRUTH",
                "trajectory_id": token.trajectory_id,
                "trial_purpose": token.trial_purpose,
                "truth_calls": self.truth_calls,
            }
        )
        return token

    def execute(self, token: SelectionToken, trajectory: pd.DataFrame) -> Any:
        if self._pending is None or token != self._pending:
            raise PermissionError("virtual truth requires the current selection token")
        if token.token in self._executed_tokens:
            raise PermissionError("a selection token can execute exactly once")
        trajectory_ids = (
            set(trajectory["trajectory_id"].astype(str))
            if "trajectory_id" in trajectory
            else {token.trajectory_id}
        )
        if len(trajectory_ids) != 1 or next(iter(trajectory_ids)) != token.trajectory_id:
            raise PermissionError("selected trajectory identity does not match execution")
        before = self.truth_calls
        result = self._backend.simulate(trajectory)
        if self.truth_calls != before + 1:
            raise RuntimeError("virtual execution must perform exactly one truth call")
        self._executed_tokens.add(token.token)
        self._pending = None
        self._audit.append(
            {
                "serial": token.serial,
                "event": "TRUTH_EXECUTED_AFTER_SELECTION",
                "trajectory_id": token.trajectory_id,
                "trial_purpose": token.trial_purpose,
                "truth_calls": self.truth_calls,
            }
        )
        return result


def _actual_objective(
    trajectory_id: str,
    execution: Any,
    reference_metrics: MechanicalTorqueMetrics,
) -> MechanicalObjectiveResult:
    return evaluate_mechanical_objective(
        trajectory_id=trajectory_id,
        metrics=execution.actual_metrics,
        reference_metrics=reference_metrics,
        hip_rms_deviation_deg=0.0,
        knee_rms_deviation_deg=0.0,
    )


def _fit_updated_model(
    fitting_data: pd.DataFrame,
    parameters: Mapping[str, float],
) -> ParameterEstimationResult:
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    return estimate_subject_parameters(
        project_estimator_inputs(fitting_data),
        template,
        L1,
        L2,
        initial_guess=parameters,
        bounds=(identification_lower_bounds, identification_upper_bounds),
    )


def _parameter_uncertainty_trace(
    fitting_data: pd.DataFrame,
    parameters: Mapping[str, float],
) -> float:
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    matrix, _, _ = numerical_sensitivity_matrix(
        fitting_data,
        template,
        parameters,
        L1,
        L2,
        torque_scales_nm=(1.0, 1.0),
    )
    return float(np.trace(np.linalg.pinv(matrix.T @ matrix)))


def _map_summary(
    prediction_map: pd.DataFrame,
    current_best: SearchAlpha,
    *,
    iteration: int,
    previous_map: pd.DataFrame | None,
) -> dict[str, Any]:
    best_row = _row_for_alpha(prediction_map, current_best)
    global_best = prediction_map.sort_values(
        ["J_pred", "trajectory_id"], kind="mergesort"
    ).iloc[0]
    row: dict[str, Any] = {
        "iteration": int(iteration),
        "geometrically_admissible_point_count": int(len(prediction_map)),
        "supported_point_count": int(prediction_map["model_supported"].sum()),
        "unsupported_point_count": int((~prediction_map["model_supported"]).sum()),
        "current_best_trajectory_id": str(best_row["trajectory_id"]),
        "current_best_J_pred": float(best_row["J_pred"]),
        "global_predicted_minimum_trajectory_id": str(global_best["trajectory_id"]),
        "global_predicted_minimum_J": float(global_best["J_pred"]),
        "global_predicted_minimum_hip_delta": float(global_best["hip_delta"]),
        "global_predicted_minimum_knee_delta": float(global_best["knee_delta"]),
        "global_predicted_minimum_phase_delta": float(global_best["phase_delta"]),
        "global_minimum_executed_directly": False,
        "truth_used_for_map": False,
        "complete_map_recomputed": True,
    }
    if previous_map is None:
        row.update(
            {
                "mean_abs_J_pred_map_change": 0.0,
                "max_abs_J_pred_map_change": 0.0,
                "supported_point_count_change": 0,
                "global_predicted_minimum_changed": False,
            }
        )
    else:
        merged = previous_map[["trajectory_id", "J_pred"]].merge(
            prediction_map[["trajectory_id", "J_pred"]],
            on="trajectory_id",
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        delta = np.abs(merged["J_pred_after"] - merged["J_pred_before"])
        previous_global = previous_map.sort_values(
            ["J_pred", "trajectory_id"], kind="mergesort"
        ).iloc[0]
        row.update(
            {
                "mean_abs_J_pred_map_change": float(delta.mean()),
                "max_abs_J_pred_map_change": float(delta.max()),
                "supported_point_count_change": int(
                    prediction_map["model_supported"].sum()
                    - previous_map["model_supported"].sum()
                ),
                "global_predicted_minimum_changed": str(
                    previous_global["trajectory_id"]
                )
                != str(global_best["trajectory_id"]),
            }
        )
    return row


def _post_policy_final_local_regret(
    final_prediction_map: pd.DataFrame,
    final_model: DiagnosticInitialModel,
    cache: TrajectoryComponentCache,
    current_best: SearchAlpha,
    steps: TrustRegionSteps,
    uncertainty: ResearchDecisionUncertainty,
) -> dict[str, Any]:
    local_pred = local_prediction_candidates(final_prediction_map, current_best, steps)
    guarded = apply_research_decision_guard(local_pred, current_best, uncertainty)
    selected = select_exploit_candidate(
        guarded, POLICY_DECISION_GUARDED_EXPLOIT_ONLY
    )
    if selected is None:
        selected = _row_for_alpha(local_pred, current_best)
    # The shared truth evaluator normalizes every objective against the active
    # reference and therefore requires the neutral row even when the final
    # best-centered neighborhood no longer contains it.  Add that row only to
    # this explicitly post-policy audit; keep the regret comparison local.
    evaluation_input = local_pred.copy()
    reference_id = str(_row_for_alpha(final_prediction_map, SearchAlpha())["trajectory_id"])
    if not evaluation_input["trajectory_id"].astype(str).eq(reference_id).any():
        evaluation_input = pd.concat(
            (
                evaluation_input,
                _row_for_alpha(final_prediction_map, SearchAlpha()).to_frame().T,
            ),
            ignore_index=True,
        )
    evaluated_all, _ = evaluate_truth_map(
        evaluation_input, final_model, cache, batch_size=64
    )
    evaluated = evaluated_all.loc[
        evaluated_all["trajectory_id"].astype(str).isin(
            set(local_pred["trajectory_id"].astype(str))
        )
    ].copy()
    truth_best = evaluated.sort_values(
        ["J_truth", "trajectory_id"], kind="mergesort"
    ).iloc[0]
    selected_truth = evaluated.loc[
        evaluated["trajectory_id"].eq(str(selected["trajectory_id"]))
    ].iloc[0]
    return {
        "final_local_recommended_trajectory_id": str(selected["trajectory_id"]),
        "final_local_truth_best_trajectory_id": str(truth_best["trajectory_id"]),
        "final_local_regret": max(
            float(selected_truth["J_truth"] - truth_best["J_truth"]), 0.0
        ),
        "final_local_regret_truth_role": MAP_TRUTH_ROLE,
    }


def run_policy(
    state: InitialResearchState,
    policy_id: str,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    *,
    trial_budget: int = RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET,
    allow_extended_offline_diagnostic_horizon: bool = False,
) -> PolicyRunResult:
    """Run one isolated policy with at most one virtual trajectory per iteration."""

    if policy_id not in POLICY_IDS:
        raise ValueError(f"unknown policy: {policy_id}")
    if trial_budget != RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET and not bool(
        allow_extended_offline_diagnostic_horizon
    ):
        raise ValueError("first research protocol reuses the existing offline budget")
    if trial_budget < 1:
        raise ValueError("trial_budget must be positive")
    validate_active_reference_file()
    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    model = _model_for_iteration(state, parameters, domain_data, 0)
    prediction_map, prediction_meta = build_predicted_map(
        model, parameter_lattice, cache, batch_size=256
    )
    initial_prediction_map = prediction_map.copy(deep=True)
    uncertainty = evaluate_validation_pairwise_uncertainty(
        state, parameters, iteration=0
    )
    oracle = SelectionGatedVirtualTruthOracle(state.subject_id, state.scenario_name)
    reference = generate_personalized_trajectory()
    reference_trajectory = reference.trajectory.copy(deep=True)
    reference_trajectory["trajectory_id"] = str(reference.metadata["trajectory_id"])
    reference_token = oracle.declare_selected(
        str(reference.metadata["trajectory_id"]), "REFERENCE_NORMALIZATION"
    )
    reference_execution = oracle.execute(reference_token, reference_trajectory)
    reference_metrics = reference_execution.actual_metrics
    current_best = SearchAlpha()
    best_actual_j = 1.0
    steps = TrustRegionSteps()
    executed_keys: set[tuple[float, float, float]] = {current_best.key()}
    history_rows: list[dict[str, Any]] = []
    guard_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    parameter_rows.append(
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": policy_id,
            "iteration": 0,
            "model_update_count": 0,
            "model_update_success": True,
            "within_trial_model_fixed": True,
            "updates_this_trial": 0,
            **{
                f"{name}_before": float(parameters[name])
                for name in PARAMETER_NAMES
            },
            **{
                f"{name}_after": float(parameters[name])
                for name in PARAMETER_NAMES
            },
            **{f"{name}_delta": 0.0 for name in PARAMETER_NAMES},
            "uncertainty_trace_before": _parameter_uncertainty_trace(
                fitting_data, parameters
            ),
            "uncertainty_trace_after": _parameter_uncertainty_trace(
                fitting_data, parameters
            ),
        }
    )
    map_rows = [
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": policy_id,
            **_map_summary(
                prediction_map, current_best, iteration=0, previous_map=None
            ),
        }
    ]
    known_rows = [
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": policy_id,
            "iteration": 0,
            "executed_known_alpha_count": 1,
            "supported_point_count": int(prediction_map["model_supported"].sum()),
            "new_supported_point_count": 0,
            "support_role": SUPPORT_ROLE,
            "reliability_updated_from_support": False,
        }
    ]
    uncertainty_rows = [
        {
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": policy_id,
            **uncertainty_summary_row(uncertainty),
        }
    ]
    uncertainty_pair_frames = [
        uncertainty.pairwise_audit.assign(policy_id=policy_id)
    ]
    exploration_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []
    stop_reason = ""
    model_update_count = 0
    cumulative_regret = 0.0
    previous_uncertainty_bound = uncertainty.bound_used_by_guard
    proposal_truth_audit: list[dict[str, Any]] = []

    for iteration in range(1, trial_budget + 1):
        truth_before_proposal = oracle.truth_calls
        local = local_prediction_candidates(prediction_map, current_best, steps)
        guarded = apply_research_decision_guard(local, current_best, uncertainty)
        guarded["iteration"] = iteration
        guarded["policy_id"] = policy_id
        guarded["scenario_name"] = state.scenario_name
        guarded["subject_id"] = state.subject_id
        guarded["selected_for_execution"] = False
        guarded["selection_mode"] = ""
        exploit = select_exploit_candidate(guarded, policy_id)
        selected: pd.Series | None = exploit
        purpose = TRIAL_PURPOSE_EXPLOIT
        frontier_ranked = pd.DataFrame()
        if selected is None and policy_id == POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT:
            frontier = build_local_exploration_frontier(
                prediction_map, executed_keys
            )
            if not frontier.empty:
                # Exploration is reserved for the not-yet-well-covered frontier.
                frontier = frontier.loc[
                    ~frontier["model_supported"].astype(bool)
                ].copy()
            frontier_ranked = rank_exploration_frontier(
                frontier, fitting_data, parameters
            )
            valid_frontier = frontier_ranked.loc[
                frontier_ranked.get(
                    "exploration_candidate_valid", pd.Series(dtype=bool)
                ).astype(bool)
            ] if not frontier_ranked.empty else frontier_ranked
            if not valid_frontier.empty:
                selected = valid_frontier.iloc[0].copy()
                purpose = TRIAL_PURPOSE_EXPLORE
                frontier_ranked.loc[
                    frontier_ranked["trajectory_id"].eq(
                        str(selected["trajectory_id"])
                    ),
                    "selected_for_exploration",
                ] = True
                frontier_ranked["iteration"] = iteration
                frontier_ranked["policy_id"] = policy_id
                frontier_ranked["scenario_name"] = state.scenario_name
                frontier_ranked["subject_id"] = state.subject_id
                frontier_ranked["selected_for_execution"] = frontier_ranked[
                    "selected_for_exploration"
                ]
                frontier_ranked["selection_mode"] = TRIAL_PURPOSE_EXPLORE
        if selected is None:
            if local.empty:
                stop_reason = STOP_NO_GEOMETRICALLY_VALID_CANDIDATE
            elif policy_id == POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT:
                frontier_any = build_local_exploration_frontier(
                    prediction_map, executed_keys
                )
                if frontier_any.empty:
                    stop_reason = STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
                else:
                    unsupported_frontier = frontier_any.loc[
                        ~frontier_any["model_supported"].astype(bool)
                    ]
                    stop_reason = (
                        STOP_PATIENT_ENVELOPE_BOUNDARY
                        if not unsupported_frontier.empty
                        and not frontier_ranked.empty
                        and not frontier_ranked["exploration_candidate_valid"]
                        .astype(bool)
                        .any()
                        else STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
                    )
            else:
                stop_reason = STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
            guard_frames.append(guarded)
            if not frontier_ranked.empty:
                guard_frames.append(frontier_ranked)
            break

        selected_id = str(selected["trajectory_id"])
        selected_alpha = alpha_from_row(selected)
        if selected_alpha.key() == current_best.key():
            raise RuntimeError("current best cannot be re-executed as a trial")
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            if selected_alpha.key() not in {
                alpha.key() for alpha in build_coordinate_neighborhood(current_best, steps)
            }:
                raise RuntimeError("exploit attempted a nonlocal/global jump")
            guarded.loc[
                guarded["trajectory_id"].eq(selected_id),
                ["selected_for_execution", "selection_mode"],
            ] = [True, purpose]
        guard_frames.append(guarded)
        if not frontier_ranked.empty:
            guard_frames.append(frontier_ranked)
        if oracle.truth_calls != truth_before_proposal:
            raise RuntimeError("proposal/ranking accessed virtual truth")
        proposal_truth_audit.append(
            {
                "iteration": iteration,
                "truth_calls_before_proposal": truth_before_proposal,
                "truth_calls_after_proposal": oracle.truth_calls,
                "selected_trajectory_id": selected_id,
                "trial_purpose": purpose,
            }
        )

        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = selected_id
        token = oracle.declare_selected(selected_id, purpose)
        execution = oracle.execute(token, trajectory)
        if not execution.observation_valid:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
            break
        actual = _actual_objective(selected_id, execution, reference_metrics)
        predicted_j = float(selected["J_pred"])
        current_predicted_j = float(_row_for_alpha(prediction_map, current_best)["J_pred"])
        delta_pred = predicted_j - current_predicted_j
        delta_actual = actual.mechanical_cost_j_rms - best_actual_j
        predicted_direction = str(classify_improvement_direction([delta_pred])[0])
        actual_direction = str(classify_improvement_direction([delta_actual])[0])
        false_improvement = bool(
            predicted_direction == IMPROVE and actual_direction != IMPROVE
        )
        accepted = accept_actual_trial(actual.mechanical_cost_j_rms, best_actual_j)
        best_before = best_actual_j
        best_alpha_before = current_best
        if accepted:
            best_actual_j = actual.mechanical_cost_j_rms
            current_best = selected_alpha
        elif purpose == TRIAL_PURPOSE_EXPLOIT:
            steps = shrink_steps(steps)
        cumulative_regret += max(actual.mechanical_cost_j_rms - best_before, 0.0)
        executed_keys.add(selected_alpha.key())

        parameters_before = dict(parameters)
        uncertainty_trace_before = _parameter_uncertainty_trace(
            fitting_data, parameters_before
        )
        adaptation = execution.estimator_observations.copy(deep=True)
        fitting_data = pd.concat((fitting_data, adaptation), ignore_index=True)
        domain_data = pd.concat((domain_data, adaptation), ignore_index=True)
        estimation = _fit_updated_model(fitting_data, parameters_before)
        if not estimation.optimizer_success:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
        else:
            parameters = dict(estimation.estimated_parameters)
            model_update_count += 1
        uncertainty_trace_after = _parameter_uncertainty_trace(
            fitting_data, parameters
        )
        previous_map = prediction_map
        model = _model_for_iteration(state, parameters, domain_data, iteration)
        prediction_map, _ = build_predicted_map(
            model, parameter_lattice, cache, batch_size=256
        )
        next_uncertainty = evaluate_validation_pairwise_uncertainty(
            state, parameters, iteration=iteration
        )
        new_supported = int(
            prediction_map["model_supported"].sum()
            - previous_map["model_supported"].sum()
        )
        map_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": policy_id,
                **_map_summary(
                    prediction_map,
                    current_best,
                    iteration=iteration,
                    previous_map=previous_map,
                ),
            }
        )
        known_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": policy_id,
                "iteration": iteration,
                "executed_known_alpha_count": len(executed_keys),
                "supported_point_count": int(prediction_map["model_supported"].sum()),
                "new_supported_point_count": new_supported,
                "support_role": SUPPORT_ROLE,
                "reliability_updated_from_support": False,
            }
        )
        uncertainty_rows.append(
            {
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": policy_id,
                **uncertainty_summary_row(next_uncertainty),
            }
        )
        uncertainty_pair_frames.append(
            next_uncertainty.pairwise_audit.assign(policy_id=policy_id)
        )
        parameter_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": policy_id,
                "iteration": iteration,
                "model_update_count": model_update_count,
                "model_update_success": bool(estimation.optimizer_success),
                "within_trial_model_fixed": True,
                "updates_this_trial": int(estimation.optimizer_success),
                **{
                    f"{name}_before": float(parameters_before[name])
                    for name in PARAMETER_NAMES
                },
                **{f"{name}_after": float(parameters[name]) for name in PARAMETER_NAMES},
                **{
                    f"{name}_delta": float(parameters[name] - parameters_before[name])
                    for name in PARAMETER_NAMES
                },
                "uncertainty_trace_before": uncertainty_trace_before,
                "uncertainty_trace_after": uncertainty_trace_after,
            }
        )
        history_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": policy_id,
                "iteration": iteration,
                "executed_trial_count_this_iteration": 1,
                "trial_purpose": purpose,
                "trajectory_id": selected_id,
                "alpha_hip": selected_alpha.hip_delta_deg,
                "alpha_knee": selected_alpha.knee_delta_deg,
                "alpha_phase": selected_alpha.phase_delta,
                "best_alpha_hip_before": best_alpha_before.hip_delta_deg,
                "best_alpha_knee_before": best_alpha_before.knee_delta_deg,
                "best_alpha_phase_before": best_alpha_before.phase_delta,
                "best_alpha_hip_after": current_best.hip_delta_deg,
                "best_alpha_knee_after": current_best.knee_delta_deg,
                "best_alpha_phase_after": current_best.phase_delta,
                "J_pred": predicted_j,
                "actual_J": actual.mechanical_cost_j_rms,
                "best_actual_J_before": best_before,
                "best_actual_J_after": best_actual_j,
                "delta_J_pred": delta_pred,
                "delta_J_actual": delta_actual,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "decision_uncertainty_bound": uncertainty.bound_used_by_guard,
                "improvement_margin": float(
                    -delta_pred
                    - uncertainty.bound_used_by_guard
                    - OBJECTIVE_EQUIVALENCE_TOLERANCE
                ),
                "model_supported": bool(selected["model_supported"]),
                "domain_coverage": float(selected["domain_coverage"]),
                "distance_to_supported_region": float(
                    selected["distance_to_supported_region"]
                ),
                "distance_role": FRONTIER_DISTANCE_ROLE,
                "accepted_improvement": accepted,
                "executed_false_improvement": false_improvement,
                "execution_status": (
                    EXECUTED_FALSE_IMPROVEMENT if false_improvement else "EXECUTED"
                ),
                "valid_data_added_to_model_update": True,
                "model_updates_this_iteration": int(estimation.optimizer_success),
                "model_reliability_status_after_iteration": (
                    "RESEARCH_DIAGNOSTIC_NOT_FORMALLY_RELIABLE"
                ),
                "truth_accessed_before_selection": False,
                "cumulative_regret_vs_best_before": cumulative_regret,
                "stop_reason_after_iteration": stop_reason,
            }
        )
        false_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": policy_id,
                "iteration": iteration,
                "trial_purpose": purpose,
                "trajectory_id": selected_id,
                "delta_J_pred": delta_pred,
                "delta_J_actual": delta_actual,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "executed_false_improvement": false_improvement,
                "false_improvement_executed_as_exploit": bool(
                    false_improvement and purpose == TRIAL_PURPOSE_EXPLOIT
                ),
                "best_updated": accepted,
            }
        )
        if purpose == TRIAL_PURPOSE_EXPLORE:
            selected_frontier = frontier_ranked.loc[
                frontier_ranked["trajectory_id"].eq(selected_id)
            ].iloc[0]
            exploration_rows.append(
                {
                    "case_id": f"{state.subject_id}__{state.scenario_name}",
                    "subject_id": state.subject_id,
                    "scenario_name": state.scenario_name,
                    "policy_id": policy_id,
                    "iteration": iteration,
                    "trajectory_id": selected_id,
                    "incremental_log_information_gain": float(
                        selected_frontier["incremental_log_information_gain"]
                    ),
                    "new_supported_point_count": new_supported,
                    "parameter_uncertainty_trace_change": (
                        uncertainty_trace_after - uncertainty_trace_before
                    ),
                    "validation_e_delta_J_change": (
                        next_uncertainty.bound_used_by_guard
                        - uncertainty.bound_used_by_guard
                    ),
                    "actual_J": actual.mechanical_cost_j_rms,
                    "accepted_as_best": accepted,
                    "later_enabled_reliable_exploit": False,
                    "truth_used_for_ranking": False,
                }
            )
        if (
            not stop_reason
            and next_uncertainty.bound_used_by_guard
            > previous_uncertainty_bound + OBJECTIVE_EQUIVALENCE_TOLERANCE
        ):
            stop_reason = STOP_MODEL_ADEQUACY_DEGRADED
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
            history_rows[-1]["model_reliability_status_after_iteration"] = (
                MODEL_RELIABILITY_DEGRADED
            )
        previous_uncertainty_bound = next_uncertainty.bound_used_by_guard
        uncertainty = next_uncertainty
        if stop_reason:
            break
    if not stop_reason:
        stop_reason = STOP_MAX_PERSONALIZATION_TRIALS
        if history_rows:
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason

    history = pd.DataFrame(history_rows)
    if exploration_rows and not history.empty:
        for row in exploration_rows:
            row["later_enabled_reliable_exploit"] = bool(
                (
                    history["iteration"].astype(int).gt(int(row["iteration"]))
                    & history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT)
                ).any()
            )
    final_model = _model_for_iteration(
        state, parameters, domain_data, model_update_count
    )
    final_local = _post_policy_final_local_regret(
        prediction_map, final_model, cache, current_best, steps, uncertainty
    )
    number_executed = int(len(history))
    final_trial_is_best = bool(
        number_executed > 0
        and SearchAlpha(
            hip_delta_deg=float(history.iloc[-1]["alpha_hip"]),
            knee_delta_deg=float(history.iloc[-1]["alpha_knee"]),
            phase_delta=float(history.iloc[-1]["alpha_phase"]),
        ).key()
        == current_best.key()
    )
    summary = {
        "case_id": f"{state.subject_id}__{state.scenario_name}",
        "subject_id": state.subject_id,
        "scenario_name": state.scenario_name,
        "policy_id": policy_id,
        "research_status": RESEARCH_ONLY,
        "model_status": DIAGNOSTIC_ONLY,
        "human_readiness": NOT_HUMAN_READY,
        "approval_status": NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
        "number_of_executed_trials": number_executed,
        "number_of_exploit_trials": int(
            history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT).sum()
        ) if number_executed else 0,
        "number_of_explore_trials": int(
            history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLORE).sum()
        ) if number_executed else 0,
        "number_of_accepted_improvements": int(
            history["accepted_improvement"].sum()
        ) if number_executed else 0,
        "number_of_executed_false_improvements": int(
            history["executed_false_improvement"].sum()
        ) if number_executed else 0,
        "number_of_exploit_false_improvements": int(
            (
                history["executed_false_improvement"].astype(bool)
                & history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT)
            ).sum()
        ) if number_executed else 0,
        "reference_actual_J": 1.0,
        "final_best_actual_J": best_actual_j,
        "actual_J_reduction_from_reference": 1.0 - best_actual_j,
        "cumulative_regret_vs_best_before": cumulative_regret,
        "model_update_count": model_update_count,
        "final_best_alpha_hip": current_best.hip_delta_deg,
        "final_best_alpha_knee": current_best.knee_delta_deg,
        "final_best_alpha_phase": current_best.phase_delta,
        "final_best_equals_last_executed_trial": final_trial_is_best,
        "final_supported_point_count": int(prediction_map["model_supported"].sum()),
        "known_region_growth": int(prediction_map["model_supported"].sum())
        - int(map_rows[0]["supported_point_count"]),
        "final_validation_e_delta_J_bound": uncertainty.bound_used_by_guard,
        "stop_reason": stop_reason,
        "trial_budget": trial_budget,
        "trial_budget_status": "RESEARCH_ONLY_NOT_HUMAN_SAFETY_THRESHOLD",
        "whole_map_recomputation_count": len(map_rows),
        "truth_calls_including_reference_normalization": oracle.truth_calls,
        "heldout_final_test_used": False,
        "global_minimum_executed_directly": False,
        "support_used_as_reliability_score": False,
        "model_reliability_status": (
            MODEL_RELIABILITY_DEGRADED
            if stop_reason == STOP_MODEL_ADEQUACY_DEGRADED
            else "RESEARCH_DIAGNOSTIC_NOT_FORMALLY_RELIABLE"
        ),
        **final_local,
    }
    guard_audit = (
        pd.concat(guard_frames, ignore_index=True, sort=False)
        if guard_frames
        else pd.DataFrame()
    )
    truth_audit = {
        "proposal_truth_call_audit": proposal_truth_audit,
        "truth_calls_unchanged_during_every_proposal": all(
            row["truth_calls_before_proposal"]
            == row["truth_calls_after_proposal"]
            for row in proposal_truth_audit
        ),
        "oracle_event_audit": oracle.audit.to_dict(orient="records"),
        "exactly_one_trajectory_per_iteration": bool(
            history.empty
            or history["executed_trial_count_this_iteration"].eq(1).all()
        ),
        "heldout_final_test_used": False,
        "post_policy_local_truth_evaluation_role": MAP_TRUTH_ROLE,
    }
    return PolicyRunResult(
        subject_id=state.subject_id,
        scenario_name=state.scenario_name,
        policy_id=policy_id,
        trial_history=history,
        decision_guard_audit=guard_audit,
        parameter_history=pd.DataFrame(parameter_rows),
        prediction_map_history=pd.DataFrame(map_rows),
        known_region_history=pd.DataFrame(known_rows),
        uncertainty_history=pd.DataFrame(uncertainty_rows),
        uncertainty_pairwise_audit=pd.concat(
            uncertainty_pair_frames, ignore_index=True
        ),
        exploration_information_gain=pd.DataFrame(exploration_rows),
        false_improvement_audit=pd.DataFrame(false_rows),
        summary=summary,
        initial_prediction_map=initial_prediction_map,
        final_prediction_map=prediction_map,
        truth_access_audit=truth_audit,
    )


def policy_definitions() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "decision_uncertainty_version": DECISION_UNCERTAINTY_VERSION,
        "policies": {
            POLICY_SUPPORTED_ONLY_GREEDY: {
                "role": "SANITY_COMPARATOR_NOT_RECOMMENDED",
                "exploit": "lowest_predicted_J_among_supported_local_trust_region_neighbors",
                "decision_guard_required": False,
                "exploration_allowed": False,
            },
            POLICY_DECISION_GUARDED_EXPLOIT_ONLY: {
                "role": "RESEARCH_DIAGNOSTIC",
                "exploit": "local_candidate_requires_validation_calibrated_margin_and_support_provenance",
                "decision_guard_required": True,
                "exploration_allowed": False,
            },
            POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT: {
                "role": "RESEARCH_DIAGNOSTIC",
                "exploit": "same_as_P1",
                "explore": "one_formal_grid_layer_information_gain_first",
                "decision_guard_required": True,
                "exploration_allowed": True,
            },
        },
        "exploit_steps": {
            "initial_hip_deg": TrustRegionSteps().hip_deg,
            "initial_knee_deg": TrustRegionSteps().knee_deg,
            "initial_phase": TrustRegionSteps().phase,
            "minimum_hip_deg": GRID_HIP_STEP_DEG,
            "minimum_knee_deg": GRID_KNEE_STEP_DEG,
            "minimum_phase": GRID_PHASE_STEP,
        },
        "exploration_frontier_steps": {
            "hip_deg": GRID_HIP_STEP_DEG,
            "knee_deg": GRID_KNEE_STEP_DEG,
            "phase": GRID_PHASE_STEP,
        },
        "decision_guard": {
            "improvement_margin_formula": (
                "-delta_J_pred - max_observed_validation_e_delta_J - 0.005"
            ),
            "uncertainty_bound_status": PAIRWISE_BOUND_STATUS,
            "support_role": SUPPORT_ROLE,
            "distance_role": FRONTIER_DISTANCE_ROLE,
            "formal_threshold": False,
        },
        "support_coverage_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "trial_budget": RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET,
        "trial_budget_status": "RESEARCH_ONLY_NOT_HUMAN_SAFETY_THRESHOLD",
        "heldout_final_test_used": False,
        "human_ready_theta_0_created": False,
        "robot_execution_allowed": False,
    }


def frozen_baseline_metadata() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "five_parameter_names": list(PARAMETER_NAMES),
        "mechanical_objective_version": MECHANICAL_OBJECTIVE_VERSION,
        "mechanical_objective_formula": "sqrt((R_h^2 + R_k^2) / 2)",
        "algorithm_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "support_coverage_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "distance_definition": GRID_DISTANCE_DEFINITION,
        "distance_role": FRONTIER_DISTANCE_ROLE,
        "support_role": SUPPORT_ROLE,
        "initial_identification_acceptance_status": (
            INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS
        ),
        "global_model_reliability_rule_status": (
            GLOBAL_MODEL_RELIABILITY_RULE_STATUS
        ),
        "real_robot_hard_safeguard_status": REAL_ROBOT_HARD_SAFEGUARD_STATUS,
        "formal_human_ready_theta_0_created": False,
        "formal_personalization_approval_created": False,
        "real_robot_connected": False,
    }


__all__ = [
    "DECISION_GUARD_STATUS",
    "DECISION_UNCERTAINTY_VERSION",
    "EXECUTED_FALSE_IMPROVEMENT",
    "FRONTIER_DISTANCE_ROLE",
    "GEOMETRICALLY_INADMISSIBLE",
    "GLOBAL_MODEL_RELIABILITY_RULE_STATUS",
    "InitialResearchState",
    "MODEL_RELIABILITY_DEGRADED",
    "NOT_APPROVED_FOR_ROBOT_PERSONALIZATION",
    "PAIRWISE_BOUND_STATUS",
    "POLICY_DECISION_GUARDED_EXPLOIT_ONLY",
    "POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT",
    "POLICY_IDS",
    "POLICY_SUPPORTED_ONLY_GREEDY",
    "PROTOCOL_ID",
    "PolicyRunResult",
    "RESEARCH_EXPLOIT_ELIGIBLE",
    "RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET",
    "ResearchDecisionUncertainty",
    "STOP_MAX_PERSONALIZATION_TRIALS",
    "STOP_MODEL_ADEQUACY_DEGRADED",
    "STOP_MODEL_UPDATE_FAILURE",
    "STOP_NO_GEOMETRICALLY_VALID_CANDIDATE",
    "STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER",
    "STOP_PATIENT_ENVELOPE_BOUNDARY",
    "SUPPORTED_BUT_DECISION_UNRELIABLE",
    "SUPPORT_ROLE",
    "SelectionGatedVirtualTruthOracle",
    "TRIAL_PURPOSE_EXPLOIT",
    "TRIAL_PURPOSE_EXPLORE",
    "UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE",
    "alpha_from_row",
    "alpha_key_from_row",
    "apply_research_decision_guard",
    "build_initial_research_state",
    "build_local_exploration_frontier",
    "evaluate_validation_pairwise_uncertainty",
    "frozen_baseline_metadata",
    "local_prediction_candidates",
    "policy_definitions",
    "rank_exploration_frontier",
    "run_policy",
    "select_exploit_candidate",
    "uncertainty_summary_row",
]
