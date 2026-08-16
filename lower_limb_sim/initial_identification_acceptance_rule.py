"""Two-gate acceptance analysis for sequential initial identification.

``INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_V1`` is an offline research audit.
It deliberately separates numerical parameter identifiability from predictive
model adequacy.  No threshold is silently promoted to a formal rule: the
default rule is incomplete and therefore fail-closed.

Only the frozen TRAIN observations produced by the preceding sequential
identification stage and the predeclared VALIDATION trajectories are used.
The held-out final-test split is not generated, read, or accepted by this
module.  Nothing here authorizes robot motion or personalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import L1, L2, identification_initial_guess, identification_parameter_scales
from .dynamic_subject import get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import compute_torque_metrics, evaluate_mechanical_objective
from .mismatch_dynamics import mismatch_inverse_dynamics
from .mismatch_metrics import compute_trajectory_metrics
from .mismatch_scenarios import get_mismatch_scenario
from .parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
    measured_joint_torque,
    predict_joint_torque,
)
from .run_model_mismatch_experiment import (
    ESTIMATOR_INPUT_COLUMNS,
    project_estimator_inputs,
)
from .safeguarded_sequential_initial_identification import (
    MAX_INITIAL_IDENTIFICATION_TRIALS,
    SequentialIdentificationResult,
)
from .trajectory_profiles import generate_identification_excitation_trajectory


PROTOCOL_ID = "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_V1"
PARAMETER_IDENTIFIABILITY_GATE = "PARAMETER_IDENTIFIABILITY_GATE"
MODEL_ADEQUACY_GATE = "MODEL_ADEQUACY_GATE"
PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW = (
    "PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW"
)
MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW = (
    "MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW"
)
INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW = (
    "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW"
)
ID_CONTINUE_NEEDS_INFORMATION = "ID_CONTINUE_NEEDS_INFORMATION"
ID_PARAMETER_IDENTIFIABLE_MODEL_ADEQUACY_PENDING = (
    "ID_PARAMETER_IDENTIFIABLE_MODEL_ADEQUACY_PENDING"
)
INITIAL_IDENTIFICATION_COMPLETE = "INITIAL_IDENTIFICATION_COMPLETE"
MODEL_INADEQUATE_FOR_PERSONALIZATION = "MODEL_INADEQUATE_FOR_PERSONALIZATION"
INITIAL_IDENTIFICATION_INSUFFICIENT = "INITIAL_IDENTIFICATION_INSUFFICIENT"
MODEL_STRUCTURE_LIMITATION = "MODEL_STRUCTURE_LIMITATION"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
NOT_APPROVED_FOR_PERSONALIZATION = "NOT_APPROVED_FOR_PERSONALIZATION"
FORMALLY_REVIEWED = "FORMALLY_REVIEWED"
RESEARCH_ONLY_NOT_FROZEN = "RESEARCH_ONLY_NOT_FROZEN"

VALIDATION_TRAJECTORY_SPECS = (
    ("hip_dominant", "slow"),
    ("knee_dominant", "fast"),
)
HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS = (
    ("coupled", "nominal"),
    ("hip_dominant", "fast"),
    ("knee_dominant", "slow"),
)

_PARAMETER_SCALES = {
    name: float(identification_parameter_scales[name]) for name in PARAMETER_NAMES
}


@dataclass(frozen=True)
class ParameterIdentifiabilityThresholds:
    """Candidate thresholds for the identifiability gate.

    ``review_status`` must be ``FORMALLY_REVIEWED`` before these values can
    produce a formal pass/fail decision.  ``minimum_rank=5`` is structural;
    every other numeric limit requires a documented scientific review.
    """

    minimum_rank: int = len(PARAMETER_NAMES)
    minimum_singular_value: float | None = None
    maximum_condition_number: float | None = None
    maximum_abs_parameter_correlation: float | None = None
    maximum_uncertainty_proxy: float | None = None
    minimum_parameter_sensitivity: float | None = None
    maximum_normalized_parameter_change: float | None = None
    review_status: str = PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW
    evidence_source: str = "NOT_DEFINED"

    @property
    def complete(self) -> bool:
        values = (
            self.minimum_singular_value,
            self.maximum_condition_number,
            self.maximum_abs_parameter_correlation,
            self.maximum_uncertainty_proxy,
            self.minimum_parameter_sensitivity,
            self.maximum_normalized_parameter_change,
        )
        return bool(
            self.review_status == FORMALLY_REVIEWED
            and self.minimum_rank == len(PARAMETER_NAMES)
            and all(value is not None and math.isfinite(float(value)) for value in values)
        )


@dataclass(frozen=True)
class ModelAdequacyThresholds:
    """Candidate thresholds for independent VALIDATION prediction quality."""

    maximum_hip_rmse_nm: float | None = None
    maximum_knee_rmse_nm: float | None = None
    maximum_combined_rmse_nm: float | None = None
    maximum_combined_nrmse_percent: float | None = None
    maximum_validation_e_j: float | None = None
    maximum_validation_relative_e_j_percent: float | None = None
    review_status: str = MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW
    evidence_source: str = "NOT_DEFINED"

    @property
    def complete(self) -> bool:
        values = (
            self.maximum_hip_rmse_nm,
            self.maximum_knee_rmse_nm,
            self.maximum_combined_rmse_nm,
            self.maximum_combined_nrmse_percent,
            self.maximum_validation_e_j,
            self.maximum_validation_relative_e_j_percent,
        )
        return bool(
            self.review_status == FORMALLY_REVIEWED
            and all(value is not None and math.isfinite(float(value)) for value in values)
        )


@dataclass(frozen=True)
class GateDecision:
    gate_name: str
    passed: bool | None
    status: str
    reasons: tuple[str, ...]
    per_parameter_pass: Mapping[str, bool | None] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceDecision:
    trial_id: int
    final_status: str
    theta_hat_0: Mapping[str, float] | None
    personalization_prerequisite: bool
    diagnostic_model_status: str


def _finite_nonnegative(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(number) and number >= 0.0)


def evaluate_parameter_identifiability(
    trial_metrics: Mapping[str, Any],
    per_parameter_metrics: pd.DataFrame,
    thresholds: ParameterIdentifiabilityThresholds | None = None,
) -> GateDecision:
    """Evaluate the parameter gate without consulting prediction residuals."""

    rule = thresholds or ParameterIdentifiabilityThresholds()
    required_global = {
        "rank",
        "minimum_singular_value",
        "condition_number",
        "maximum_abs_parameter_correlation",
        "maximum_normalized_parameter_change",
    }
    missing_global = required_global.difference(trial_metrics)
    if missing_global:
        raise ValueError(f"identifiability metrics missing: {sorted(missing_global)}")
    required_parameter = {"parameter", "sensitivity", "uncertainty_proxy"}
    missing_parameter = required_parameter.difference(per_parameter_metrics.columns)
    if missing_parameter:
        raise ValueError(
            f"per-parameter metrics missing: {sorted(missing_parameter)}"
        )
    names = set(per_parameter_metrics["parameter"].astype(str))
    if names != set(PARAMETER_NAMES):
        raise ValueError("identifiability gate requires exactly all five parameters")
    if not rule.complete:
        return GateDecision(
            gate_name=PARAMETER_IDENTIFIABILITY_GATE,
            passed=None,
            status=PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW,
            reasons=("complete_formally_reviewed_numeric_thresholds_not_available",),
            per_parameter_pass={name: None for name in PARAMETER_NAMES},
        )

    checks = {
        "rank": int(trial_metrics["rank"]) >= rule.minimum_rank,
        "minimum_singular_value": float(trial_metrics["minimum_singular_value"])
        >= float(rule.minimum_singular_value),
        "condition_number": float(trial_metrics["condition_number"])
        <= float(rule.maximum_condition_number),
        "parameter_correlation": float(
            trial_metrics["maximum_abs_parameter_correlation"]
        )
        <= float(rule.maximum_abs_parameter_correlation),
        "parameter_stability": float(
            trial_metrics["maximum_normalized_parameter_change"]
        )
        <= float(rule.maximum_normalized_parameter_change),
    }
    per_parameter: dict[str, bool] = {}
    for name in PARAMETER_NAMES:
        row = per_parameter_metrics.loc[
            per_parameter_metrics["parameter"].astype(str).eq(name)
        ].iloc[0]
        per_parameter[name] = bool(
            float(row["sensitivity"]) >= float(rule.minimum_parameter_sensitivity)
            and float(row["uncertainty_proxy"])
            <= float(rule.maximum_uncertainty_proxy)
        )
    reasons = tuple(name for name, passed in checks.items() if not passed)
    reasons += tuple(
        f"parameter_support:{name}" for name, passed in per_parameter.items() if not passed
    )
    return GateDecision(
        gate_name=PARAMETER_IDENTIFIABILITY_GATE,
        passed=not reasons,
        status=(
            "PARAMETER_IDENTIFIABILITY_PASS"
            if not reasons
            else "PARAMETER_IDENTIFIABILITY_FAIL"
        ),
        reasons=reasons,
        per_parameter_pass=per_parameter,
    )


def evaluate_model_adequacy(
    metrics: Mapping[str, Any],
    thresholds: ModelAdequacyThresholds | None = None,
) -> GateDecision:
    """Evaluate only independent VALIDATION metrics, never training RMSE."""

    forbidden = {name for name in metrics if "training" in str(name).lower()}
    if forbidden:
        raise ValueError(f"model-adequacy input contains training metrics: {sorted(forbidden)}")
    required = {
        "validation_hip_rmse_nm",
        "validation_knee_rmse_nm",
        "validation_combined_rmse_nm",
        "validation_combined_nrmse_percent",
        "validation_e_j",
        "validation_relative_e_j_percent",
    }
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"model-adequacy metrics missing: {sorted(missing)}")
    if not all(_finite_nonnegative(metrics[name]) for name in required):
        raise ValueError("model-adequacy metrics must be finite and non-negative")
    rule = thresholds or ModelAdequacyThresholds()
    if not rule.complete:
        return GateDecision(
            gate_name=MODEL_ADEQUACY_GATE,
            passed=None,
            status=MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW,
            reasons=("complete_formally_reviewed_validation_thresholds_not_available",),
        )
    checks = {
        "hip_rmse": float(metrics["validation_hip_rmse_nm"])
        <= float(rule.maximum_hip_rmse_nm),
        "knee_rmse": float(metrics["validation_knee_rmse_nm"])
        <= float(rule.maximum_knee_rmse_nm),
        "combined_rmse": float(metrics["validation_combined_rmse_nm"])
        <= float(rule.maximum_combined_rmse_nm),
        "combined_nrmse": float(metrics["validation_combined_nrmse_percent"])
        <= float(rule.maximum_combined_nrmse_percent),
        "validation_e_j": float(metrics["validation_e_j"])
        <= float(rule.maximum_validation_e_j),
        "validation_relative_e_j": float(
            metrics["validation_relative_e_j_percent"]
        )
        <= float(rule.maximum_validation_relative_e_j_percent),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return GateDecision(
        gate_name=MODEL_ADEQUACY_GATE,
        passed=not reasons,
        status="MODEL_ADEQUACY_PASS" if not reasons else "MODEL_ADEQUACY_FAIL",
        reasons=reasons,
    )


def determine_acceptance_state(
    trial_id: int,
    parameter_decision: GateDecision,
    model_decision: GateDecision,
    temporary_parameters: Mapping[str, float],
) -> AcceptanceDecision:
    """Apply the fail-closed state machine and freeze only after two passes."""

    if not 1 <= int(trial_id) <= MAX_INITIAL_IDENTIFICATION_TRIALS:
        raise ValueError("trial_id must be in 1..5")
    if parameter_decision.gate_name != PARAMETER_IDENTIFIABILITY_GATE:
        raise ValueError("first decision must be the parameter-identifiability gate")
    if model_decision.gate_name != MODEL_ADEQUACY_GATE:
        raise ValueError("second decision must be the model-adequacy gate")
    parameters = {name: float(temporary_parameters[name]) for name in PARAMETER_NAMES}
    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError("temporary parameters must be finite")

    if parameter_decision.passed is None:
        status = PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW
    elif parameter_decision.passed is False:
        status = (
            INITIAL_IDENTIFICATION_INSUFFICIENT
            if trial_id == MAX_INITIAL_IDENTIFICATION_TRIALS
            else ID_CONTINUE_NEEDS_INFORMATION
        )
    elif model_decision.passed is None:
        status = ID_PARAMETER_IDENTIFIABLE_MODEL_ADEQUACY_PENDING
    elif model_decision.passed is False:
        status = MODEL_INADEQUATE_FOR_PERSONALIZATION
    else:
        status = INITIAL_IDENTIFICATION_COMPLETE
    complete = status == INITIAL_IDENTIFICATION_COMPLETE
    return AcceptanceDecision(
        trial_id=int(trial_id),
        final_status=status,
        theta_hat_0=parameters if complete else None,
        personalization_prerequisite=complete,
        diagnostic_model_status=(
            "APPROVED_INITIAL_MODEL" if complete else f"{DIAGNOSTIC_ONLY};{NOT_APPROVED_FOR_PERSONALIZATION}"
        ),
    )


def parameter_stability_by_trial(
    result: SequentialIdentificationResult,
) -> pd.DataFrame:
    """Return parameter deltas relative to the prior accumulated estimate."""

    estimates = result.parameter_estimates.copy(deep=True)
    required = {"trial_id", "parameter", "estimate"}
    if required.difference(estimates.columns):
        raise ValueError("sequential result lacks parameter estimates")
    previous = dict(identification_initial_guess)
    rows: list[dict[str, Any]] = []
    for trial_id in sorted(estimates["trial_id"].astype(int).unique()):
        current_rows = estimates.loc[estimates["trial_id"].astype(int).eq(trial_id)]
        current = {
            str(row.parameter): float(row.estimate)
            for row in current_rows.itertuples(index=False)
        }
        if set(current) != set(PARAMETER_NAMES):
            raise ValueError("each trial must estimate exactly five parameters")
        for name in PARAMETER_NAMES:
            delta = current[name] - previous[name]
            rows.append(
                {
                    "subject_id": result.subject_id,
                    "scenario_name": result.truth_scenario,
                    "case_id": f"{result.subject_id}__{result.truth_scenario}",
                    "trial_id": int(trial_id),
                    "temporary_parameter_name": f"theta_hat_ID_{trial_id}",
                    "parameter": name,
                    "estimate": current[name],
                    "previous_estimate": previous[name],
                    "previous_state": "initial_guess" if trial_id == 1 else f"theta_hat_ID_{trial_id - 1}",
                    "delta_theta": delta,
                    "absolute_delta_theta": abs(delta),
                    "parameter_scale": _PARAMETER_SCALES[name],
                    "normalized_parameter_change": abs(delta) / _PARAMETER_SCALES[name],
                    "parameter_interpretation": "local_equivalent_dynamics_parameter",
                    "is_tissue_material_change": False,
                }
            )
        previous = current
    table = pd.DataFrame(rows)
    table["maximum_normalized_parameter_change_in_trial"] = table.groupby(
        ["case_id", "trial_id"]
    )["normalized_parameter_change"].transform("max")
    return table


def build_parameter_identifiability_table(
    result: SequentialIdentificationResult,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Combine global matrix diagnostics with all five parameter diagnostics."""

    per_parameter = result.parameter_identifiability.copy(deep=True).rename(
        columns={"truth_scenario": "scenario_name"}
    )
    global_metrics = result.trial_history.loc[
        :,
        [
            "trial_id",
            "rank",
            "minimum_singular_value",
            "condition_number",
            "maximum_abs_parameter_correlation",
            "highest_correlation_pair",
            "weakest_parameter",
            "training_residual_rmse_nm",
            "within_identification_validation_residual_rmse_nm",
        ],
    ].copy()
    table = per_parameter.merge(global_metrics, on="trial_id", validate="many_to_one")
    table["case_id"] = f"{result.subject_id}__{result.truth_scenario}"
    stability_columns = stability.loc[
        :,
        [
            "trial_id",
            "parameter",
            "estimate",
            "delta_theta",
            "normalized_parameter_change",
            "maximum_normalized_parameter_change_in_trial",
        ],
    ]
    table = table.merge(
        stability_columns,
        on=["trial_id", "parameter"],
        validate="one_to_one",
    )
    table["gate"] = PARAMETER_IDENTIFIABILITY_GATE
    table["validation_residual_used_by_gate"] = False
    table["threshold_status"] = PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW
    return table


def build_validation_observations(subject_id: str, scenario_name: str) -> pd.DataFrame:
    """Generate only the two predeclared VALIDATION trajectories.

    Generator truth creates virtual endpoint-force observations, then the
    strict estimator projection removes scenario and torque-truth fields.
    The returned table adds only public validation identity columns.
    """

    base_subject = get_dynamic_subject(subject_id)
    scenario = get_mismatch_scenario(scenario_name)
    generator_subject = scenario.create_subject(base_subject)
    frames: list[pd.DataFrame] = []
    for family, speed in VALIDATION_TRAJECTORY_SPECS:
        profile = generate_identification_excitation_trajectory(family, speed).copy()
        qh = profile["q_hip_rad"].to_numpy(dtype=float)
        qk = profile["q_knee_rad"].to_numpy(dtype=float)
        dynamics = mismatch_inverse_dynamics(
            qh,
            qk,
            profile["dq_hip_rad_s"].to_numpy(dtype=float),
            profile["dq_knee_rad_s"].to_numpy(dtype=float),
            profile["ddq_hip_rad_s2"].to_numpy(dtype=float),
            profile["ddq_knee_rad_s2"].to_numpy(dtype=float),
            generator_subject,
            L1,
            residual_random_seed=scenario.random_seed,
        )
        force = endpoint_force_from_joint_torque(
            qh,
            qk,
            np.asarray(dynamics.tau_total_hip_nm, dtype=float),
            np.asarray(dynamics.tau_total_knee_nm, dtype=float),
            L1,
            L2,
        )
        valid = np.asarray(force.force_mapping_valid, dtype=bool)
        raw = pd.DataFrame(
            {
                "q_hip_rad": qh,
                "q_knee_rad": qk,
                "dq_hip_rad_s": profile["dq_hip_rad_s"].to_numpy(dtype=float),
                "dq_knee_rad_s": profile["dq_knee_rad_s"].to_numpy(dtype=float),
                "ddq_hip_rad_s2": profile["ddq_hip_rad_s2"].to_numpy(dtype=float),
                "ddq_knee_rad_s2": profile["ddq_knee_rad_s2"].to_numpy(dtype=float),
                "fx_observed_n": np.asarray(force.fx_robot_on_leg_n, dtype=float),
                "fz_observed_n": np.asarray(force.fz_robot_on_leg_n, dtype=float),
                "sample_valid": valid,
                "force_mapping_valid": valid,
                "wrench_is_stale": False,
                "invalid_reason": np.asarray(force.invalid_reason, dtype=str),
            }
        )
        safe = project_estimator_inputs(raw)
        safe.insert(0, "time_s", profile["time_s"].to_numpy(dtype=float))
        safe.insert(0, "speed_profile", speed)
        safe.insert(0, "trajectory_family", family)
        safe.insert(0, "trajectory_id", f"identification_excitation_trajectory:{family}:{speed}")
        safe.insert(0, "dataset_split", "validation")
        safe["theta_shank_rad"] = safe["q_hip_rad"] - safe["q_knee_rad"]
        frames.append(safe)
    output = pd.concat(frames, ignore_index=True)
    forbidden_ids = {
        f"identification_excitation_trajectory:{family}:{speed}"
        for family, speed in HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS
    }
    if forbidden_ids.intersection(output["trajectory_id"].astype(str)):
        raise RuntimeError("held-out final-test trajectory entered validation")
    if tuple(project_estimator_inputs(output).columns) != ESTIMATOR_INPUT_COLUMNS:
        raise RuntimeError("strict estimator projection changed")
    return output


def _mechanical_validation_error(
    validation: pd.DataFrame,
    measured_hip: np.ndarray,
    measured_knee: np.ndarray,
    predicted_hip: np.ndarray,
    predicted_knee: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return duration-weighted J values using validation-only references."""

    actual_values: list[float] = []
    predicted_values: list[float] = []
    durations: list[float] = []
    for _, group in validation.groupby("trajectory_id", sort=False):
        index = group.index.to_numpy(dtype=int)
        time = group["time_s"].to_numpy(dtype=float)
        actual_metrics = compute_torque_metrics(time, measured_hip[index], measured_knee[index])
        predicted_metrics = compute_torque_metrics(
            time, predicted_hip[index], predicted_knee[index]
        )
        actual = evaluate_mechanical_objective(
            trajectory_id=str(group["trajectory_id"].iloc[0]),
            metrics=actual_metrics,
            reference_metrics=actual_metrics,
            hip_rms_deviation_deg=0.0,
            knee_rms_deviation_deg=0.0,
        ).mechanical_cost_j_rms
        predicted = evaluate_mechanical_objective(
            trajectory_id=str(group["trajectory_id"].iloc[0]),
            metrics=predicted_metrics,
            reference_metrics=actual_metrics,
            hip_rms_deviation_deg=0.0,
            knee_rms_deviation_deg=0.0,
        ).mechanical_cost_j_rms
        actual_values.append(float(actual))
        predicted_values.append(float(predicted))
        durations.append(float(time[-1] - time[0]))
    weights = np.asarray(durations, dtype=float)
    j_actual = float(np.average(actual_values, weights=weights))
    j_predicted = float(np.average(predicted_values, weights=weights))
    error = abs(j_predicted - j_actual)
    relative = 100.0 * error / max(abs(j_actual), np.finfo(float).eps)
    return j_actual, j_predicted, error, relative


def evaluate_validation_by_trial(
    result: SequentialIdentificationResult,
    validation_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every temporary estimate on independent VALIDATION data."""

    if set(validation_observations["dataset_split"].astype(str)) != {"validation"}:
        raise ValueError("model adequacy accepts only the validation split")
    forbidden = {
        f"identification_excitation_trajectory:{family}:{speed}"
        for family, speed in HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS
    }
    if forbidden.intersection(validation_observations["trajectory_id"].astype(str)):
        raise ValueError("held-out final-test data is prohibited")
    safe = project_estimator_inputs(validation_observations)
    measured_hip, measured_knee = measured_joint_torque(safe, L1, L2)
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    estimates = result.parameter_estimates.pivot(
        index="trial_id", columns="parameter", values="estimate"
    )
    rows: list[dict[str, Any]] = []
    for trial_id, estimate_row in estimates.sort_index().iterrows():
        parameters = {name: float(estimate_row[name]) for name in PARAMETER_NAMES}
        predicted_hip, predicted_knee = predict_joint_torque(safe, template, parameters, L1)
        predictions = validation_observations.copy(deep=True).reset_index(drop=True)
        predictions["tau_measured_hip_nm"] = measured_hip
        predictions["tau_measured_knee_nm"] = measured_knee
        predictions["tau_predicted_identified_hip_nm"] = predicted_hip
        predictions["tau_predicted_identified_knee_nm"] = predicted_knee
        trajectory_metrics = compute_trajectory_metrics(
            predictions, prediction_model="identified", link_1_m=L1, link_2_m=L2
        )
        aggregate_predictions = predictions.drop(
            columns=["trajectory_family", "speed_profile"], errors="ignore"
        ).copy()
        aggregate_predictions["trajectory_id"] = "validation_aggregate"
        aggregate_metric = compute_trajectory_metrics(
            aggregate_predictions,
            prediction_model="identified",
            link_1_m=L1,
            link_2_m=L2,
        ).iloc[0]
        count = int(aggregate_metric["valid_torque_sample_count"])
        hip_rmse = float(aggregate_metric["hip_torque_rmse_nm"])
        knee_rmse = float(aggregate_metric["knee_torque_rmse_nm"])
        combined_rmse = float(aggregate_metric["combined_torque_rmse_nm"])
        combined_nrmse = float(aggregate_metric["combined_nrmse_percent"])
        j_actual, j_predicted, e_j, relative_e_j = _mechanical_validation_error(
            validation_observations.reset_index(drop=True),
            measured_hip,
            measured_knee,
            predicted_hip,
            predicted_knee,
        )
        rows.append(
            {
                "case_id": f"{result.subject_id}__{result.truth_scenario}",
                "subject_id": result.subject_id,
                "scenario_name": result.truth_scenario,
                "trial_id": int(trial_id),
                "temporary_parameter_name": f"theta_hat_ID_{trial_id}",
                "validation_trajectory_count": int(len(trajectory_metrics)),
                "validation_sample_count": int(count),
                "validation_hip_rmse_nm": hip_rmse,
                "validation_knee_rmse_nm": knee_rmse,
                "validation_combined_rmse_nm": combined_rmse,
                "validation_combined_nrmse_percent": combined_nrmse,
                "validation_nrmse_unreliable_small_range": bool(
                    aggregate_metric["combined_nrmse_unreliable_small_range"]
                ),
                "validation_j_actual": j_actual,
                "validation_j_predicted": j_predicted,
                "validation_e_j": e_j,
                "validation_relative_e_j_percent": relative_e_j,
                "mechanical_objective_interpretation": (
                    "validation_self_normalized_precursor_not_global_reliability_rule"
                ),
                "gate": MODEL_ADEQUACY_GATE,
                "training_residual_used_by_gate": False,
                "heldout_final_test_used": False,
                "truth_parameters_used_by_decision": False,
                "threshold_status": MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW,
            }
        )
    return pd.DataFrame(rows)


def identification_marginal_gain_table(
    identifiability: pd.DataFrame,
    stability: pd.DataFrame,
    adequacy: pd.DataFrame,
    information_gain: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Trial i-1 -> i continuous gains without inventing a plateau cutoff."""

    global_ident = identifiability.groupby(
        ["case_id", "subject_id", "scenario_name", "trial_id"], as_index=False
    ).agg(
        rank=("rank", "first"),
        minimum_singular_value=("minimum_singular_value", "first"),
        condition_number=("condition_number", "first"),
        maximum_abs_parameter_correlation=("maximum_abs_parameter_correlation", "first"),
        maximum_uncertainty_proxy=("uncertainty_proxy", "max"),
        minimum_parameter_sensitivity=("sensitivity", "min"),
    )
    stability_summary = stability.groupby(["case_id", "trial_id"], as_index=False).agg(
        maximum_normalized_parameter_change=("normalized_parameter_change", "max")
    )
    table = global_ident.merge(stability_summary, on=["case_id", "trial_id"])
    table = table.merge(
        adequacy[
            [
                "case_id",
                "trial_id",
                "validation_combined_rmse_nm",
                "validation_e_j",
            ]
        ],
        on=["case_id", "trial_id"],
        validate="one_to_one",
    )
    gains = information_gain.copy(deep=True).rename(columns={"truth_scenario": "scenario_name"})
    table = table.merge(
        gains[["subject_id", "scenario_name", "trial_id", "incremental_log_information_gain"]],
        on=["subject_id", "scenario_name", "trial_id"],
        validate="one_to_one",
    ).sort_values(["case_id", "trial_id"])
    rows: list[dict[str, Any]] = []
    for case_id, group in table.groupby("case_id", sort=False):
        group = group.sort_values("trial_id").reset_index(drop=True)
        for index in range(1, len(group)):
            previous = group.iloc[index - 1]
            current = group.iloc[index]
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": current["subject_id"],
                    "scenario_name": current["scenario_name"],
                    "from_trial": int(previous["trial_id"]),
                    "to_trial": int(current["trial_id"]),
                    "minimum_singular_value_improvement": float(current["minimum_singular_value"] - previous["minimum_singular_value"]),
                    "condition_number_improvement": float(previous["condition_number"] - current["condition_number"]),
                    "correlation_improvement": float(previous["maximum_abs_parameter_correlation"] - current["maximum_abs_parameter_correlation"]),
                    "maximum_uncertainty_reduction": float(previous["maximum_uncertainty_proxy"] - current["maximum_uncertainty_proxy"]),
                    "minimum_parameter_sensitivity_improvement": float(current["minimum_parameter_sensitivity"] - previous["minimum_parameter_sensitivity"]),
                    "maximum_normalized_parameter_change": float(current["maximum_normalized_parameter_change"]),
                    "validation_rmse_improvement_nm": float(previous["validation_combined_rmse_nm"] - current["validation_combined_rmse_nm"]),
                    "validation_e_j_improvement": float(previous["validation_e_j"] - current["validation_e_j"]),
                    "selected_candidate_incremental_log_information_gain": float(current["incremental_log_information_gain"]),
                    "plateau_threshold_applied": False,
                    "formal_stop_decision_from_this_row": False,
                }
            )
    return pd.DataFrame(rows)


def diagnose_model_structure_limitation(
    identifiability: pd.DataFrame,
    adequacy: pd.DataFrame,
) -> str:
    """Return a trend diagnosis, not a thresholded acceptance decision."""

    ident = identifiability.groupby("trial_id", as_index=False).agg(
        rank=("rank", "first"),
        minimum_singular_value=("minimum_singular_value", "first"),
        maximum_uncertainty_proxy=("uncertainty_proxy", "max"),
    ).sort_values("trial_id")
    model = adequacy.sort_values("trial_id")
    full_rank = bool((ident["rank"].astype(int) == len(PARAMETER_NAMES)).all())
    information_improved = bool(
        ident["minimum_singular_value"].iloc[-1]
        > ident["minimum_singular_value"].iloc[0]
        and ident["maximum_uncertainty_proxy"].iloc[-1]
        < ident["maximum_uncertainty_proxy"].iloc[0]
    )
    errors = model["validation_combined_rmse_nm"].to_numpy(dtype=float)
    # This is a floating-point zero check, not an adequacy threshold.  Exact
    # matched-model recovery should not be mislabeled because of nanometre-scale
    # numerical solver noise.
    numerically_nonzero = not bool(np.allclose(errors, 0.0, atol=1e-8, rtol=0.0))
    error_not_resolved = bool(
        numerically_nonzero and errors[-1] >= np.min(errors[1:])
    )
    if full_rank and information_improved and error_not_resolved:
        return MODEL_STRUCTURE_LIMITATION
    if not full_rank:
        return "PARAMETER_IDENTIFIABILITY_FAILURE_PATTERN"
    return "NO_STRUCTURE_LIMITATION_DIAGNOSIS_FROM_TREND_ONLY"


def frozen_baseline_metadata() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "five_parameter_names": list(PARAMETER_NAMES),
        "maximum_initial_identification_trials": MAX_INITIAL_IDENTIFICATION_TRIALS,
        "validation_trajectory_specs": [list(item) for item in VALIDATION_TRAJECTORY_SPECS],
        "heldout_final_test_trajectory_specs": [
            list(item) for item in HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS
        ],
        "heldout_final_test_generated_or_read": False,
        "acceptance_rule_status": INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW,
        "real_robot_connected": False,
        "personalization_executed": False,
        "global_prediction_reliability_rule_frozen": False,
    }


__all__ = [
    "AcceptanceDecision",
    "DIAGNOSTIC_ONLY",
    "FORMALLY_REVIEWED",
    "GateDecision",
    "HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS",
    "ID_CONTINUE_NEEDS_INFORMATION",
    "ID_PARAMETER_IDENTIFIABLE_MODEL_ADEQUACY_PENDING",
    "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW",
    "INITIAL_IDENTIFICATION_COMPLETE",
    "INITIAL_IDENTIFICATION_INSUFFICIENT",
    "MODEL_ADEQUACY_GATE",
    "MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW",
    "MODEL_INADEQUATE_FOR_PERSONALIZATION",
    "MODEL_STRUCTURE_LIMITATION",
    "ModelAdequacyThresholds",
    "NOT_APPROVED_FOR_PERSONALIZATION",
    "PARAMETER_IDENTIFIABILITY_GATE",
    "PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW",
    "PROTOCOL_ID",
    "ParameterIdentifiabilityThresholds",
    "VALIDATION_TRAJECTORY_SPECS",
    "build_parameter_identifiability_table",
    "build_validation_observations",
    "determine_acceptance_state",
    "diagnose_model_structure_limitation",
    "evaluate_model_adequacy",
    "evaluate_parameter_identifiability",
    "evaluate_validation_by_trial",
    "frozen_baseline_metadata",
    "identification_marginal_gain_table",
    "parameter_stability_by_trial",
]
