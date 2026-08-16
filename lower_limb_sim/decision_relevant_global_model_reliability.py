"""Offline decision-relevant reliability characterization.

This module compares a diagnostic five-parameter model with virtual truth over
the complete geometrically valid personalization lattice.  Prediction is built
first; virtual truth is attached only in the post-prediction evaluation layer.
No function freezes a reliability threshold, approves a human-ready model, or
executes personalization/robot motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .config import L1
from .continuous_reference_neighborhood import generate_personalized_trajectory
from .dynamic_subject import get_dynamic_subject
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import StateDomainBounds
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
    MechanicalTorqueMetrics,
    compute_torque_metrics,
)
from .mismatch_dynamics import mismatch_inverse_dynamics
from .mismatch_scenarios import get_mismatch_scenario
from .parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
    candidate_subject_from_parameters,
)
from .safeguarded_sequential_initial_identification import (
    SUPPORTED_PREDICTION,
    UNSUPPORTED_EXTRAPOLATION,
    SequentialIdentificationResult,
)
from .sequential_personalization import (
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
)


PROTOCOL_ID = "DECISION_RELEVANT_GLOBAL_MODEL_RELIABILITY_CHARACTERIZATION_V1"
GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS = "GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS"
GLOBAL_MODEL_RELIABILITY_RULE_STATUS = "NOT_FROZEN_REQUIRES_REVIEW"
INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS = (
    "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW"
)
DIAGNOSTIC_INITIAL_MODEL = "DIAGNOSTIC_INITIAL_MODEL"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
NOT_APPROVED_FOR_PERSONALIZATION = "NOT_APPROVED_FOR_PERSONALIZATION"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
RESEARCH_DECISION_EQUIVALENCE_BAND = "RESEARCH_DECISION_EQUIVALENCE_BAND"
IMPROVE = "IMPROVE"
NEUTRAL = "NEUTRAL"
WORSE = "WORSE"
FALSE_IMPROVEMENT = "FALSE_IMPROVEMENT"
MODEL_INADEQUATE_FOR_PRECISE_DYNAMICS = "MODEL_INADEQUATE_FOR_PRECISE_DYNAMICS"
POTENTIALLY_USEFUL_FOR_LOCAL_DECISION = "POTENTIALLY_USEFUL_FOR_LOCAL_DECISION"
MODEL_INADEQUATE_FOR_DECISION = "MODEL_INADEQUATE_FOR_DECISION"
MIXED_DECISION_UTILITY_REQUIRES_REVIEW = "MIXED_DECISION_UTILITY_REQUIRES_REVIEW"
NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW = (
    "NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW"
)
MODEL_SUPPORT_COVERAGE_GATE_PERCENT = 90.0

GRID_HIP_STEP_DEG = MINIMUM_STEP_HIP_DEG
GRID_KNEE_STEP_DEG = MINIMUM_STEP_KNEE_DEG
GRID_PHASE_STEP = MINIMUM_STEP_PHASE
GRID_DISTANCE_DEFINITION = (
    "euclidean_distance_to_nearest_supported_alpha_after_scaling_hip_by_0.25_deg_"
    "knee_by_0.25_deg_and_phase_by_0.0025"
)
LOCAL_RADIUS_DEFINITION = (
    "chebyshev_radius_in_formal_grid_steps_from_reference_alpha_0_0_0"
)
RELATIVE_ERROR_DEFINITION = "100 * abs(J_pred - J_truth) / abs(J_truth)"

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
_GEOMETRY_GATE_COLUMNS = (
    "alpha_bounds_valid",
    "global_rom_valid",
    "workspace_valid",
    "jacobian_valid",
    "force_mapping_valid",
    "velocity_valid",
    "acceleration_valid",
    "closure_valid",
    "continuity_valid",
    "asymmetry_valid",
    "finite_valid",
)


@dataclass(frozen=True)
class DiagnosticInitialModel:
    subject_id: str
    scenario_name: str
    selected_trial_id: int
    parameters: Mapping[str, float]
    identification_domain: StateDomainBounds
    identification_dataset_sha256: str
    model_status: str = DIAGNOSTIC_ONLY
    approval_status: str = NOT_APPROVED_FOR_PERSONALIZATION
    human_readiness: str = NOT_HUMAN_READY


@dataclass(frozen=True)
class TrajectoryComponentCache:
    time_s: np.ndarray
    hip: Mapping[float, tuple[np.ndarray, np.ndarray, np.ndarray]]
    knee: Mapping[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]]

    def batch(self, rows: pd.DataFrame) -> tuple[np.ndarray, ...]:
        hip_components = [self.hip[float(value)] for value in rows["hip_delta"]]
        knee_components = [
            self.knee[(float(knee), float(phase))]
            for knee, phase in zip(rows["knee_delta"], rows["phase_delta"])
        ]
        qh = np.stack([item[0] for item in hip_components])
        dqh = np.stack([item[1] for item in hip_components])
        ddqh = np.stack([item[2] for item in hip_components])
        qk = np.stack([item[0] for item in knee_components])
        dqk = np.stack([item[1] for item in knee_components])
        ddqk = np.stack([item[2] for item in knee_components])
        return qh, qk, dqh, dqk, ddqh, ddqk


def _dataframe_sha256(dataframe: pd.DataFrame) -> str:
    payload = dataframe.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identification_domain(dataframe: pd.DataFrame) -> StateDomainBounds:
    values = dataframe.loc[:, _STATE_COLUMNS].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    selected = values[finite]
    if selected.size == 0:
        raise ValueError("diagnostic identification data has no finite states")
    return StateDomainBounds(
        columns=_DOMAIN_COLUMNS,
        lower=tuple(np.min(selected, axis=0)),
        upper=tuple(np.max(selected, axis=0)),
        valid_training_samples=int(len(selected)),
    )


def diagnostic_model_from_sequential_result(
    result: SequentialIdentificationResult,
) -> DiagnosticInitialModel:
    """Select an actually accumulated temporary model without formal freezing."""

    executed_trial_ids = result.executed_identification_data["trial_id"].astype(int)
    if executed_trial_ids.empty:
        raise ValueError("sequential result has no actually executed trials")
    if result.truth_scenario == "matched_linear":
        selected_trial = 2
        if int(executed_trial_ids.max()) < selected_trial:
            raise ValueError("matched diagnostic model requires actually executed Trial 2")
    else:
        selected_trial = int(executed_trial_ids.max())
    estimates = result.parameter_estimates.loc[
        result.parameter_estimates["trial_id"].astype(int).eq(selected_trial)
    ]
    parameters = {
        str(row.parameter): float(row.estimate)
        for row in estimates.itertuples(index=False)
    }
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError("diagnostic model must contain exactly five parameters")
    executed = result.executed_identification_data.loc[
        result.executed_identification_data["trial_id"].astype(int).le(selected_trial)
    ].copy()
    if executed.empty:
        raise ValueError("selected diagnostic trial has no accumulated observations")
    return DiagnosticInitialModel(
        subject_id=result.subject_id,
        scenario_name=result.truth_scenario,
        selected_trial_id=selected_trial,
        parameters=parameters,
        identification_domain=_identification_domain(executed),
        identification_dataset_sha256=_dataframe_sha256(executed),
    )


def geometrically_valid_parameter_lattice(parameter_map: pd.DataFrame) -> pd.DataFrame:
    """Keep every non-domain geometry-valid point; do not use support as geometry."""

    required = {
        "trajectory_id",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "parent_reference_sha256",
        *_GEOMETRY_GATE_COLUMNS,
    }
    missing = required.difference(parameter_map.columns)
    if missing:
        raise ValueError(f"parameter lattice missing columns: {sorted(missing)}")
    if not parameter_map["parent_reference_sha256"].astype(str).eq(
        ACTIVE_REFERENCE_SHA256
    ).all():
        raise RuntimeError("parameter lattice parent reference SHA mismatch")
    mask = np.ones(len(parameter_map), dtype=bool)
    for column in _GEOMETRY_GATE_COLUMNS:
        mask &= parameter_map[column].fillna(False).astype(bool).to_numpy()
    selected = parameter_map.loc[
        mask,
        [
            "trajectory_id",
            "hip_delta",
            "knee_delta",
            "phase_delta",
            "parent_reference_sha256",
        ],
    ].copy()
    selected["geometrically_admissible"] = True
    if selected.empty:
        raise ValueError("geometrically valid lattice is empty")
    return selected.reset_index(drop=True)


def build_trajectory_component_cache(parameter_lattice: pd.DataFrame) -> TrajectoryComponentCache:
    """Build the same separable trajectory cache used by the preceding map."""

    hip: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    knee: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    time_s: np.ndarray | None = None
    for hip_delta in sorted(parameter_lattice["hip_delta"].astype(float).unique()):
        trajectory = generate_personalized_trajectory(
            hip_amplitude_delta_deg=float(hip_delta),
            knee_amplitude_delta_deg=0.0,
            knee_phase_shift=0.0,
        ).trajectory
        if time_s is None:
            time_s = trajectory["time_s"].to_numpy(dtype=float)
        hip[float(hip_delta)] = tuple(
            trajectory[column].to_numpy(dtype=float)
            for column in ("q_hip_rad", "dq_hip_rad_s", "ddq_hip_rad_s2")
        )
    keys = sorted(
        {
            (float(knee), float(phase))
            for knee, phase in zip(
                parameter_lattice["knee_delta"], parameter_lattice["phase_delta"]
            )
        }
    )
    for knee_delta, phase_delta in keys:
        trajectory = generate_personalized_trajectory(
            hip_amplitude_delta_deg=0.0,
            knee_amplitude_delta_deg=knee_delta,
            knee_phase_shift=phase_delta,
        ).trajectory
        knee[(knee_delta, phase_delta)] = tuple(
            trajectory[column].to_numpy(dtype=float)
            for column in ("q_knee_rad", "dq_knee_rad_s", "ddq_knee_rad_s2")
        )
    if time_s is None:
        raise RuntimeError("trajectory cache did not initialize time")
    return TrajectoryComponentCache(time_s=time_s, hip=hip, knee=knee)


def _batch_rms(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0.0 or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("trajectory time must be strictly increasing")
    return np.sqrt(np.trapezoid(values**2, time_s, axis=1) / duration)


def mechanical_objective_from_torque_batch(
    time_s: np.ndarray,
    hip_torque_nm: np.ndarray,
    knee_torque_nm: np.ndarray,
    reference_metrics: MechanicalTorqueMetrics,
) -> np.ndarray:
    """Vectorized, formula-identical evaluation of the frozen objective J."""

    hip = np.asarray(hip_torque_nm, dtype=float)
    knee = np.asarray(knee_torque_nm, dtype=float)
    if hip.ndim != 2 or knee.shape != hip.shape:
        raise ValueError("batched hip/knee torque must have equal 2-D shapes")
    hip_ratio = _batch_rms(time_s, hip) / reference_metrics.hip_rms_torque_nm
    knee_ratio = _batch_rms(time_s, knee) / reference_metrics.knee_rms_torque_nm
    return np.sqrt((hip_ratio**2 + knee_ratio**2) / 2.0)


def _reference_states(cache: TrajectoryComponentCache) -> tuple[np.ndarray, ...]:
    h = cache.hip[0.0]
    k = cache.knee[(0.0, 0.0)]
    return h[0], k[0], h[1], k[1], h[2], k[2]


def _reference_metrics_predicted(
    model: DiagnosticInitialModel,
    cache: TrajectoryComponentCache,
) -> MechanicalTorqueMetrics:
    qh, qk, dqh, dqk, ddqh, ddqk = _reference_states(cache)
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    subject = candidate_subject_from_parameters(template, model.parameters)
    dynamics = inverse_dynamics(qh, qk, dqh, dqk, ddqh, ddqk, subject, L1)
    return compute_torque_metrics(
        cache.time_s,
        np.asarray(dynamics.tau_total_hip_nm, dtype=float),
        np.asarray(dynamics.tau_total_knee_nm, dtype=float),
    )


def _coverage(states: tuple[np.ndarray, ...], domain: StateDomainBounds) -> np.ndarray:
    stacked = np.stack(states, axis=2)
    lower = np.asarray(domain.lower, dtype=float)[None, None, :]
    upper = np.asarray(domain.upper, dtype=float)[None, None, :]
    member = np.isfinite(stacked).all(axis=2) & (stacked >= lower).all(axis=2) & (
        stacked <= upper
    ).all(axis=2)
    return 100.0 * np.mean(member, axis=1)


def distance_to_supported_region(
    parameter_points: pd.DataFrame,
    supported: Sequence[bool],
) -> np.ndarray:
    """Return nearest-support Euclidean distance in frozen grid-step units."""

    coordinates = parameter_points.loc[
        :, ["hip_delta", "knee_delta", "phase_delta"]
    ].to_numpy(dtype=float)
    coordinates /= np.asarray(
        [GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP], dtype=float
    )
    supported_mask = np.asarray(supported, dtype=bool)
    if supported_mask.shape != (len(parameter_points),):
        raise ValueError("supported mask length mismatch")
    if not np.any(supported_mask):
        raise ValueError("distance to support is undefined without supported points")
    tree = cKDTree(coordinates[supported_mask])
    distance, _ = tree.query(coordinates, k=1)
    distance = np.asarray(distance, dtype=float)
    distance[supported_mask] = 0.0
    return distance


def build_predicted_map(
    model: DiagnosticInitialModel,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    *,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build J_pred/support without evaluating or receiving J_truth."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    predicted_subject = candidate_subject_from_parameters(template, model.parameters)
    reference_metrics = _reference_metrics_predicted(model, cache)
    j_parts: list[np.ndarray] = []
    coverage_parts: list[np.ndarray] = []
    for start in range(0, len(parameter_lattice), batch_size):
        rows = parameter_lattice.iloc[start : start + batch_size]
        states = cache.batch(rows)
        dynamics = inverse_dynamics(*states, predicted_subject, L1)
        j_parts.append(
            mechanical_objective_from_torque_batch(
                cache.time_s,
                np.asarray(dynamics.tau_total_hip_nm, dtype=float),
                np.asarray(dynamics.tau_total_knee_nm, dtype=float),
                reference_metrics,
            )
        )
        coverage_parts.append(_coverage(states, model.identification_domain))
    output = parameter_lattice.copy(deep=True)
    output.insert(0, "case_id", f"{model.subject_id}__{model.scenario_name}")
    output.insert(1, "subject_id", model.subject_id)
    output.insert(2, "scenario_name", model.scenario_name)
    output["J_pred"] = np.concatenate(j_parts)
    output["domain_coverage"] = np.concatenate(coverage_parts)
    output["model_supported"] = (
        output["domain_coverage"] >= MODEL_SUPPORT_COVERAGE_GATE_PERCENT
    )
    output["prediction_label"] = np.where(
        output["model_supported"], SUPPORTED_PREDICTION, UNSUPPORTED_EXTRAPOLATION
    )
    output["distance_to_supported_region"] = distance_to_supported_region(
        output, output["model_supported"]
    )
    output["distance_to_supported_region_definition"] = GRID_DISTANCE_DEFINITION
    output["can_calculate_equals_can_trust"] = False
    output["diagnostic_model_trial_id"] = model.selected_trial_id
    output["diagnostic_model_type"] = DIAGNOSTIC_INITIAL_MODEL
    output["diagnostic_model_status"] = model.model_status
    output["diagnostic_model_approval_status"] = model.approval_status
    output["diagnostic_model_human_readiness"] = model.human_readiness
    output["diagnostic_identification_dataset_sha256"] = (
        model.identification_dataset_sha256
    )
    reference = output.loc[
        np.isclose(output["hip_delta"], 0.0)
        & np.isclose(output["knee_delta"], 0.0)
        & np.isclose(output["phase_delta"], 0.0)
    ]
    if len(reference) != 1 or not np.isclose(
        float(reference.iloc[0]["J_pred"]), 1.0, atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("predicted active-reference J is not one")
    metadata = {
        "J_pred_reference": float(reference.iloc[0]["J_pred"]),
        "J_pred_reference_verified_one": True,
        "truth_evaluated_during_prediction": False,
        "truth_used_for_fitting": False,
        "truth_used_for_prediction_ranking": False,
    }
    return output, metadata


def select_predicted_best(
    prediction_table: pd.DataFrame,
    *,
    supported_only: bool = False,
) -> pd.Series:
    """Select by J_pred only; J_truth is explicitly prohibited."""

    if "J_truth" in prediction_table.columns:
        raise ValueError("predicted-best selection must occur before truth attachment")
    required = {"trajectory_id", "J_pred", "model_supported"}
    missing = required.difference(prediction_table.columns)
    if missing:
        raise ValueError(f"prediction table missing: {sorted(missing)}")
    table = prediction_table.loc[
        prediction_table["model_supported"].astype(bool)
        if supported_only
        else np.ones(len(prediction_table), dtype=bool)
    ].copy()
    if table.empty:
        raise ValueError("predicted-best candidate set is empty")
    return table.sort_values(
        ["J_pred", "trajectory_id"], kind="mergesort"
    ).iloc[0].copy()


def _reference_metrics_truth(
    subject_id: str,
    scenario_name: str,
    cache: TrajectoryComponentCache,
) -> MechanicalTorqueMetrics:
    qh, qk, dqh, dqk, ddqh, ddqk = _reference_states(cache)
    scenario = get_mismatch_scenario(scenario_name)
    truth_subject = scenario.create_subject(get_dynamic_subject(subject_id))
    dynamics = mismatch_inverse_dynamics(
        qh,
        qk,
        dqh,
        dqk,
        ddqh,
        ddqk,
        truth_subject,
        L1,
        residual_random_seed=scenario.random_seed,
    )
    return compute_torque_metrics(
        cache.time_s,
        np.asarray(dynamics.tau_total_hip_nm, dtype=float),
        np.asarray(dynamics.tau_total_knee_nm, dtype=float),
    )


def evaluate_truth_map(
    prediction_table: pd.DataFrame,
    model: DiagnosticInitialModel,
    cache: TrajectoryComponentCache,
    *,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach J_truth after prediction/support/ranking inputs are frozen."""

    if "J_truth" in prediction_table:
        raise ValueError("truth has already been attached")
    scenario = get_mismatch_scenario(model.scenario_name)
    truth_subject = scenario.create_subject(get_dynamic_subject(model.subject_id))
    reference_metrics = _reference_metrics_truth(
        model.subject_id, model.scenario_name, cache
    )
    j_parts: list[np.ndarray] = []
    for start in range(0, len(prediction_table), batch_size):
        rows = prediction_table.iloc[start : start + batch_size]
        states = cache.batch(rows)
        dynamics = mismatch_inverse_dynamics(
            *states,
            truth_subject,
            L1,
            residual_random_seed=scenario.random_seed,
        )
        j_parts.append(
            mechanical_objective_from_torque_batch(
                cache.time_s,
                np.asarray(dynamics.tau_total_hip_nm, dtype=float),
                np.asarray(dynamics.tau_total_knee_nm, dtype=float),
                reference_metrics,
            )
        )
    output = prediction_table.copy(deep=True)
    output["J_truth"] = np.concatenate(j_parts)
    reference = output.loc[
        np.isclose(output["hip_delta"], 0.0)
        & np.isclose(output["knee_delta"], 0.0)
        & np.isclose(output["phase_delta"], 0.0)
    ]
    if len(reference) != 1 or not np.isclose(
        float(reference.iloc[0]["J_truth"]), 1.0, atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("truth active-reference J is not one")
    output["delta_J_pred"] = output["J_pred"] - float(reference.iloc[0]["J_pred"])
    output["delta_J_truth"] = output["J_truth"] - float(reference.iloc[0]["J_truth"])
    output["e_J_abs"] = np.abs(output["J_pred"] - output["J_truth"])
    output["e_J_relative_percent"] = (
        100.0 * output["e_J_abs"] / np.abs(output["J_truth"])
    )
    output["predicted_direction"] = classify_improvement_direction(
        output["delta_J_pred"].to_numpy(dtype=float)
    )
    output["truth_direction"] = classify_improvement_direction(
        output["delta_J_truth"].to_numpy(dtype=float)
    )
    output["decision_sign_agreement"] = (
        output["predicted_direction"] == output["truth_direction"]
    )
    output["false_improvement"] = (
        output["predicted_direction"].eq(IMPROVE)
        & ~output["truth_direction"].eq(IMPROVE)
    )
    output["decision_equivalence_band"] = OBJECTIVE_EQUIVALENCE_TOLERANCE
    output["decision_equivalence_band_status"] = RESEARCH_DECISION_EQUIVALENCE_BAND
    output["predicted_percentile_rank"] = output["J_pred"].rank(
        method="average", pct=True
    )
    output["truth_percentile_rank"] = output["J_truth"].rank(
        method="average", pct=True
    )
    output["absolute_rank_percentile_error"] = np.abs(
        output["predicted_percentile_rank"] - output["truth_percentile_rank"]
    )
    metadata = {
        "J_truth_reference": float(reference.iloc[0]["J_truth"]),
        "J_truth_reference_verified_one": True,
        "truth_evaluation_stage": "POST_PREDICTION_EVALUATION_ONLY",
        "truth_used_for_fitting": False,
        "truth_used_for_support_construction": False,
        "truth_used_for_pre_evaluation_ranking": False,
    }
    return output, metadata


def classify_improvement_direction(
    delta_j: Sequence[float] | np.ndarray,
    *,
    equivalence_tolerance: float = OBJECTIVE_EQUIVALENCE_TOLERANCE,
) -> np.ndarray:
    """Classify with the existing 0.005 algorithmic equivalence tolerance."""

    values = np.asarray(delta_j, dtype=float)
    tolerance = float(equivalence_tolerance)
    if not np.isfinite(values).all() or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("direction inputs/tolerance must be finite")
    return np.where(
        values < -tolerance,
        IMPROVE,
        np.where(values > tolerance, WORSE, NEUTRAL),
    )


def _spearman(table: pd.DataFrame) -> tuple[float, float, int]:
    selected = table.loc[
        np.isfinite(table["J_pred"].to_numpy(dtype=float))
        & np.isfinite(table["J_truth"].to_numpy(dtype=float))
    ]
    if len(selected) < 2:
        return float("nan"), float("nan"), int(len(selected))
    result = spearmanr(
        selected["J_pred"].to_numpy(dtype=float),
        selected["J_truth"].to_numpy(dtype=float),
    )
    return float(result.statistic), float(result.pvalue), int(len(selected))


def global_rank_consistency(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, case in table.groupby("case_id", sort=False):
        scopes = {
            "ALL_GEOMETRICALLY_ADMISSIBLE": case,
            "SUPPORTED_ONLY": case.loc[case["model_supported"].astype(bool)],
            "UNSUPPORTED_ONLY": case.loc[~case["model_supported"].astype(bool)],
        }
        for scope, selected in scopes.items():
            correlation, pvalue, count = _spearman(selected)
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "scope": scope,
                    "point_count": count,
                    "spearman_rank_correlation": correlation,
                    "spearman_pvalue": pvalue,
                    "metric_used_as_pass_fail_threshold": False,
                }
            )
    return pd.DataFrame(rows)


def _grid_chebyshev_distance(table: pd.DataFrame) -> np.ndarray:
    values = table.loc[:, ["hip_delta", "knee_delta", "phase_delta"]].to_numpy(
        dtype=float
    )
    values /= np.asarray(
        [GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP], dtype=float
    )
    return np.max(np.abs(values), axis=1)


def local_rank_consistency(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, case in table.groupby("case_id", sort=False):
        distance = _grid_chebyshev_distance(case)
        for radius in (1, 2, 3):
            selected = case.loc[distance <= radius + 1e-12]
            correlation, pvalue, count = _spearman(selected)
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "radius_grid_steps": radius,
                    "radius_definition": LOCAL_RADIUS_DEFINITION,
                    "hip_step_deg": GRID_HIP_STEP_DEG,
                    "knee_step_deg": GRID_KNEE_STEP_DEG,
                    "phase_step": GRID_PHASE_STEP,
                    "point_count": count,
                    "spearman_rank_correlation": correlation,
                    "spearman_pvalue": pvalue,
                    "metric_used_as_pass_fail_threshold": False,
                }
            )
    return pd.DataFrame(rows)


def _truth_best(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        raise ValueError("truth-best candidate set is empty")
    return table.sort_values(["J_truth", "trajectory_id"], kind="mergesort").iloc[
        0
    ].copy()


def predicted_best_regret(
    evaluated: pd.DataFrame,
    predicted_best_ids: Mapping[tuple[str, str], str],
) -> pd.DataFrame:
    """Evaluate preselected predicted bests against post-attached truth."""

    rows: list[dict[str, Any]] = []
    for case_id, case in evaluated.groupby("case_id", sort=False):
        for scope, supported_only in (
            ("GLOBAL", False),
            ("SUPPORTED_ONLY", True),
        ):
            selected = (
                case.loc[case["model_supported"].astype(bool)]
                if supported_only
                else case
            )
            predicted_id = predicted_best_ids[(str(case_id), scope)]
            predicted = selected.loc[selected["trajectory_id"].eq(predicted_id)]
            if len(predicted) != 1:
                raise RuntimeError("preselected predicted best is not unique")
            predicted = predicted.iloc[0]
            truth = _truth_best(selected)
            regret = float(predicted["J_truth"] - truth["J_truth"])
            if regret < -1e-12:
                raise RuntimeError("predicted-best regret cannot be negative")
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "scope": scope,
                    "point_count": int(len(selected)),
                    "alpha_predicted_best_trajectory_id": predicted_id,
                    "alpha_predicted_best_hip_delta": float(predicted["hip_delta"]),
                    "alpha_predicted_best_knee_delta": float(predicted["knee_delta"]),
                    "alpha_predicted_best_phase_delta": float(predicted["phase_delta"]),
                    "J_pred_at_predicted_best": float(predicted["J_pred"]),
                    "J_truth_at_predicted_best": float(predicted["J_truth"]),
                    "alpha_truth_best_trajectory_id": str(truth["trajectory_id"]),
                    "alpha_truth_best_hip_delta": float(truth["hip_delta"]),
                    "alpha_truth_best_knee_delta": float(truth["knee_delta"]),
                    "alpha_truth_best_phase_delta": float(truth["phase_delta"]),
                    "J_truth_at_truth_best": float(truth["J_truth"]),
                    "diagnostic_regret": max(regret, 0.0),
                    "truth_used_for_predicted_best_selection": False,
                    "truth_used_for_post_selection_evaluation": True,
                }
            )
    return pd.DataFrame(rows)


def one_step_coordinate_neighborhood(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.loc[:, ["hip_delta", "knee_delta", "phase_delta"]].to_numpy(
        dtype=float
    ) / np.asarray(
        [GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP], dtype=float
    )
    axial_distance = np.sum(np.abs(normalized), axis=1)
    return table.loc[axial_distance <= 1.0 + 1e-12].copy()


def local_decision_regret(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, case in table.groupby("case_id", sort=False):
        local = one_step_coordinate_neighborhood(case)
        supported = local.loc[local["model_supported"].astype(bool)].copy()
        current_rows = local.loc[
            np.isclose(local["hip_delta"], 0.0)
            & np.isclose(local["knee_delta"], 0.0)
            & np.isclose(local["phase_delta"], 0.0)
        ]
        if len(current_rows) != 1:
            raise RuntimeError("local neighborhood must contain the reference once")
        current_id = str(current_rows.iloc[0]["trajectory_id"])
        if supported.empty:
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "current_best_trajectory_id": current_id,
                    "local_neighborhood_definition": (
                        "reference_plus_six_signed_coordinate_moves_at_formal_minimum_steps"
                    ),
                    "local_point_count": int(len(local)),
                    "supported_local_point_count": 0,
                    "predicted_local_best_trajectory_id": "",
                    "truth_local_best_trajectory_id": "",
                    "predicted_best_equals_truth_best": False,
                    "J_truth_at_predicted_local_best": float("nan"),
                    "J_truth_at_truth_local_best": float("nan"),
                    "local_decision_regret": float("nan"),
                    "selected_candidate_false_improvement": False,
                    "local_false_improvement_count": 0,
                    "diagnostic_local_utility_label": (
                        NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW
                    ),
                    "utility_label_is_formal_reliability_approval": False,
                    "truth_used_for_candidate_selection": False,
                }
            )
            continue
        predicted_projection = supported.drop(columns=["J_truth"], errors="ignore")
        predicted = select_predicted_best(predicted_projection)
        selected = supported.loc[
            supported["trajectory_id"].eq(str(predicted["trajectory_id"]))
        ].iloc[0]
        truth = _truth_best(supported)
        regret = float(selected["J_truth"] - truth["J_truth"])
        if regret < -1e-12:
            raise RuntimeError("local decision regret cannot be negative")
        local_false_improvement_count = int(supported["false_improvement"].sum())
        exact_best_match = str(selected["trajectory_id"]) == str(truth["trajectory_id"])
        selected_false_improvement = bool(selected["false_improvement"])
        if exact_best_match and local_false_improvement_count == 0:
            utility = POTENTIALLY_USEFUL_FOR_LOCAL_DECISION
        elif selected_false_improvement:
            utility = MODEL_INADEQUATE_FOR_DECISION
        else:
            utility = MIXED_DECISION_UTILITY_REQUIRES_REVIEW
        rows.append(
            {
                "case_id": case_id,
                "subject_id": case["subject_id"].iloc[0],
                "scenario_name": case["scenario_name"].iloc[0],
                "current_best_trajectory_id": current_id,
                "local_neighborhood_definition": (
                    "reference_plus_six_signed_coordinate_moves_at_formal_minimum_steps"
                ),
                "local_point_count": int(len(local)),
                "supported_local_point_count": int(len(supported)),
                "predicted_local_best_trajectory_id": str(selected["trajectory_id"]),
                "truth_local_best_trajectory_id": str(truth["trajectory_id"]),
                "predicted_best_equals_truth_best": exact_best_match,
                "J_truth_at_predicted_local_best": float(selected["J_truth"]),
                "J_truth_at_truth_local_best": float(truth["J_truth"]),
                "local_decision_regret": max(regret, 0.0),
                "selected_candidate_false_improvement": selected_false_improvement,
                "local_false_improvement_count": local_false_improvement_count,
                "diagnostic_local_utility_label": utility,
                "utility_label_is_formal_reliability_approval": False,
                "truth_used_for_candidate_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _error_statistics(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    if array.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def scenario_reliability_summary(
    table: pd.DataFrame,
    ranks: pd.DataFrame,
    regrets: pd.DataFrame,
    local_regrets: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, case in table.groupby("case_id", sort=False):
        for scope, selected in (
            ("OVERALL", case),
            ("SUPPORTED", case.loc[case["model_supported"].astype(bool)]),
            ("UNSUPPORTED", case.loc[~case["model_supported"].astype(bool)]),
        ):
            stats = _error_statistics(selected["e_J_abs"])
            relative_stats = _error_statistics(selected["e_J_relative_percent"])
            rank_scope = {
                "OVERALL": "ALL_GEOMETRICALLY_ADMISSIBLE",
                "SUPPORTED": "SUPPORTED_ONLY",
                "UNSUPPORTED": "UNSUPPORTED_ONLY",
            }[scope]
            rank_row = ranks.loc[
                ranks["case_id"].eq(case_id) & ranks["scope"].eq(rank_scope)
            ].iloc[0]
            local = local_regrets.loc[local_regrets["case_id"].eq(case_id)].iloc[0]
            regret_scope = {
                "OVERALL": "GLOBAL",
                "SUPPORTED": "SUPPORTED_ONLY",
                "UNSUPPORTED": None,
            }[scope]
            if regret_scope is None:
                predicted_best_regret_value = float("nan")
            else:
                global_regret = regrets.loc[
                    regrets["case_id"].eq(case_id)
                    & regrets["scope"].eq(regret_scope)
                ].iloc[0]
                predicted_best_regret_value = float(
                    global_regret["diagnostic_regret"]
                )
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "scope": scope,
                    "point_count": int(len(selected)),
                    "e_J_abs_mean": stats["mean"],
                    "e_J_abs_median": stats["median"],
                    "e_J_abs_p90": stats["p90"],
                    "e_J_abs_p95": stats["p95"],
                    "e_J_abs_p99": stats["p99"],
                    "e_J_abs_max": stats["max"],
                    "e_J_relative_mean_percent": relative_stats["mean"],
                    "e_J_relative_median_percent": relative_stats["median"],
                    "e_J_relative_p90_percent": relative_stats["p90"],
                    "e_J_relative_p95_percent": relative_stats["p95"],
                    "e_J_relative_p99_percent": relative_stats["p99"],
                    "e_J_relative_max_percent": relative_stats["max"],
                    "decision_sign_agreement_rate": float(
                        selected["decision_sign_agreement"].mean()
                    )
                    if len(selected)
                    else float("nan"),
                    "false_improvement_count": int(selected["false_improvement"].sum()),
                    "false_improvement_rate": float(selected["false_improvement"].mean())
                    if len(selected)
                    else float("nan"),
                    "spearman_rank_correlation": float(
                        rank_row["spearman_rank_correlation"]
                    ),
                    "predicted_best_regret": predicted_best_regret_value,
                    "local_decision_regret": float(local["local_decision_regret"]),
                    "diagnostic_local_utility_label": str(
                        local["diagnostic_local_utility_label"]
                    ),
                    "formal_reliability_status": GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
                }
            )
    return pd.DataFrame(rows)


def decision_sign_agreement_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    directions = (IMPROVE, NEUTRAL, WORSE)
    for case_id, case in table.groupby("case_id", sort=False):
        for scope, selected in (
            ("OVERALL", case),
            ("SUPPORTED", case.loc[case["model_supported"].astype(bool)]),
            ("UNSUPPORTED", case.loc[~case["model_supported"].astype(bool)]),
        ):
            row: dict[str, Any] = {
                "case_id": case_id,
                "subject_id": case["subject_id"].iloc[0],
                "scenario_name": case["scenario_name"].iloc[0],
                "scope": scope,
                "point_count": int(len(selected)),
                "agreement_count": int(selected["decision_sign_agreement"].sum()),
                "agreement_rate": float(selected["decision_sign_agreement"].mean())
                if len(selected)
                else float("nan"),
                "false_improvement_count": int(selected["false_improvement"].sum()),
                "false_improvement_rate": float(selected["false_improvement"].mean())
                if len(selected)
                else float("nan"),
            }
            for predicted in directions:
                for truth in directions:
                    row[f"pred_{predicted.lower()}__truth_{truth.lower()}_count"] = int(
                        (
                            selected["predicted_direction"].eq(predicted)
                            & selected["truth_direction"].eq(truth)
                        ).sum()
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_vs_support_distance(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped_table = table.copy(deep=False)
    distance_key = np.round(
        grouped_table["distance_to_supported_region"].to_numpy(dtype=float), 6
    )
    for (case_id, distance), group in grouped_table.groupby(
        ["case_id", distance_key], sort=True
    ):
        correlation, _, count = _spearman(group)
        rows.append(
            {
                "case_id": case_id,
                "subject_id": group["subject_id"].iloc[0],
                "scenario_name": group["scenario_name"].iloc[0],
                "distance_to_supported_region": float(distance),
                "distance_definition": GRID_DISTANCE_DEFINITION,
                "point_count": count,
                "mean_e_J_abs": float(group["e_J_abs"].mean()),
                "median_e_J_abs": float(group["e_J_abs"].median()),
                "decision_sign_accuracy": float(group["decision_sign_agreement"].mean()),
                "false_improvement_rate": float(group["false_improvement"].mean()),
                "mean_absolute_rank_percentile_error": float(
                    group["absolute_rank_percentile_error"].mean()
                ),
                "within_group_spearman_rank_correlation": correlation,
            }
        )
    return pd.DataFrame(rows)


def reliability_vs_domain_coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped_table = table.copy(deep=False)
    coverage_key = np.round(grouped_table["domain_coverage"].to_numpy(dtype=float), 6)
    for (case_id, coverage), group in grouped_table.groupby(
        ["case_id", coverage_key], sort=True
    ):
        correlation, _, count = _spearman(group)
        rows.append(
            {
                "case_id": case_id,
                "subject_id": group["subject_id"].iloc[0],
                "scenario_name": group["scenario_name"].iloc[0],
                "domain_coverage_percent": float(coverage),
                "existing_90_percent_gate_pass": bool(
                    float(coverage) >= MODEL_SUPPORT_COVERAGE_GATE_PERCENT
                ),
                "point_count": count,
                "mean_e_J_abs": float(group["e_J_abs"].mean()),
                "median_e_J_abs": float(group["e_J_abs"].median()),
                "decision_sign_accuracy": float(group["decision_sign_agreement"].mean()),
                "false_improvement_rate": float(group["false_improvement"].mean()),
                "mean_absolute_rank_percentile_error": float(
                    group["absolute_rank_percentile_error"].mean()
                ),
                "within_group_spearman_rank_correlation": correlation,
                "coverage_gate_modified": False,
            }
        )
    return pd.DataFrame(rows)


def false_improvement_cases(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "subject_id",
        "scenario_name",
        "trajectory_id",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "J_pred",
        "J_truth",
        "delta_J_pred",
        "delta_J_truth",
        "domain_coverage",
        "model_supported",
        "prediction_label",
        "distance_to_supported_region",
        "distance_to_supported_region_definition",
        "predicted_direction",
        "truth_direction",
        "decision_equivalence_band",
    ]
    return table.loc[table["false_improvement"].astype(bool), columns].copy()


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
        "decision_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "decision_equivalence_tolerance_status": RESEARCH_DECISION_EQUIVALENCE_BAND,
        "model_support_coverage_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "global_model_reliability_rule_status": GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
        "initial_identification_acceptance_status": INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
        "formal_theta_hat_0_available": False,
        "human_ready_theta_hat_0_available": False,
        "personalization_executed": False,
        "real_robot_connected": False,
    }


__all__ = [
    "DIAGNOSTIC_INITIAL_MODEL",
    "DIAGNOSTIC_ONLY",
    "FALSE_IMPROVEMENT",
    "GLOBAL_MODEL_RELIABILITY_RULE_STATUS",
    "GRID_DISTANCE_DEFINITION",
    "GRID_HIP_STEP_DEG",
    "GRID_KNEE_STEP_DEG",
    "GRID_PHASE_STEP",
    "IMPROVE",
    "MODEL_INADEQUATE_FOR_DECISION",
    "MODEL_INADEQUATE_FOR_PRECISE_DYNAMICS",
    "MODEL_SUPPORT_COVERAGE_GATE_PERCENT",
    "MIXED_DECISION_UTILITY_REQUIRES_REVIEW",
    "NEUTRAL",
    "NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW",
    "NOT_APPROVED_FOR_PERSONALIZATION",
    "NOT_HUMAN_READY",
    "POTENTIALLY_USEFUL_FOR_LOCAL_DECISION",
    "PROTOCOL_ID",
    "RESEARCH_DECISION_EQUIVALENCE_BAND",
    "RELATIVE_ERROR_DEFINITION",
    "TrajectoryComponentCache",
    "WORSE",
    "build_predicted_map",
    "build_trajectory_component_cache",
    "classify_improvement_direction",
    "decision_sign_agreement_summary",
    "diagnostic_model_from_sequential_result",
    "distance_to_supported_region",
    "evaluate_truth_map",
    "false_improvement_cases",
    "frozen_baseline_metadata",
    "geometrically_valid_parameter_lattice",
    "global_rank_consistency",
    "local_decision_regret",
    "local_rank_consistency",
    "mechanical_objective_from_torque_batch",
    "predicted_best_regret",
    "reliability_vs_domain_coverage",
    "reliability_vs_support_distance",
    "scenario_reliability_summary",
    "select_predicted_best",
]
