"""Fail-closed offline sequential initial identification.

This module implements ``SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1``.
It is intentionally isolated from robot, collection, safety, and formal
personalization code.  The only executable experiments are deterministic
virtual-subject experiments.

Three constraint layers are kept separate throughout:

``GLOBAL_MODEL_CONSTRAINTS``
    Mathematical ROM, workspace, Jacobian, force-mapping, continuity, and
    finite-value checks.  They are not patient-safety limits.
``PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE``
    A pre-supplied local region in which a candidate may be considered.  This
    module never expands it and never attempts to discover an injury boundary.
``REAL_ROBOT_HARD_SAFEGUARD``
    ``NOT_DEFINED_NOT_APPROVED``.  Nothing here authorizes physical motion.

The existing frozen 24 s rehabilitation reference is used only as a geometric
parent.  Identification excitation duration is an independent research-design
variable and has no human-safety interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    force_magnitude_limit_n,
    identification_initial_guess,
    identification_parameter_scales,
    jacobian_condition_limit,
    jacobian_det_threshold,
)
from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    TOTAL_DURATION_S,
    generate_personalized_trajectory,
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
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import StateDomainBounds, classify_state_domain
from .jacobian import jacobian_diagnostics
from .kinematics import forward_kinematics
from .mechanical_objective import compute_torque_metrics as mechanical_torque_metrics
from .mismatch_dynamics import mismatch_inverse_dynamics
from .mismatch_scenarios import get_mismatch_scenario
from .parameter_estimator import (
    PARAMETER_NAMES,
    BaselineSubjectTemplate,
    baseline_template_from_dynamic_subject,
    compute_torque_metrics,
    estimate_subject_parameters,
    predict_joint_torque,
)
from .identifiability_analysis import numerical_sensitivity_matrix


PROTOCOL_ID = "SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1"
MAX_INITIAL_IDENTIFICATION_TRIALS = 5
AUTO_EXPAND_PATIENT_ENVELOPE = False
GLOBAL_MODEL_CONSTRAINTS = "GLOBAL_MODEL_CONSTRAINTS"
PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE = (
    "PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE"
)
REAL_ROBOT_HARD_SAFEGUARD = "NOT_DEFINED_NOT_APPROVED"
IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW = (
    "IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW"
)
VIRTUAL_RESEARCH_STOP_RULE_STATUS = (
    "VIRTUAL_RESEARCH_COMPARISON_ONLY_NOT_FROZEN"
)
RESEARCH_DURATION_LABEL = (
    "RESEARCH_DESIGN_RANGE_NOT_HUMAN_SAFETY_LIMIT"
)
INITIAL_IDENTIFICATION_COMPLETE = "INITIAL_IDENTIFICATION_COMPLETE"
INITIAL_IDENTIFICATION_INSUFFICIENT = "INITIAL_IDENTIFICATION_INSUFFICIENT"
SUPPORTED_PREDICTION = "SUPPORTED_PREDICTION"
UNSUPPORTED_EXTRAPOLATION = "UNSUPPORTED_EXTRAPOLATION"

_STATE_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
)
_HASH_COLUMNS = ("time_s", *_STATE_COLUMNS, "theta_shank_rad")
_PARAMETER_SCALES = np.asarray(
    [identification_parameter_scales[name] for name in PARAMETER_NAMES],
    dtype=float,
)


@dataclass(frozen=True)
class PatientOperationalEnvelope:
    """Pre-approved local region for an offline virtual subject fixture.

    The numeric values are inputs, not learned limits.  ``clinical_safety_*``
    fields deliberately do not exist.
    """

    envelope_id: str
    patient_hip_min_deg: float
    patient_hip_max_deg: float
    patient_knee_min_deg: float
    patient_knee_max_deg: float
    source_status: str = "SYNTHETIC_TEST_FIXTURE_NOT_CLINICAL_SAFETY_LIMIT"
    values_are_clinical_safety_limits: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.patient_hip_min_deg,
                self.patient_hip_max_deg,
                self.patient_knee_min_deg,
                self.patient_knee_max_deg,
            ),
            dtype=float,
        )
        if not str(self.envelope_id).strip():
            raise ValueError("envelope_id must not be empty")
        if not np.isfinite(values).all():
            raise ValueError("patient operational envelope must be finite")
        if not self.patient_hip_min_deg < self.patient_hip_max_deg:
            raise ValueError("patient hip envelope must be increasing")
        if not self.patient_knee_min_deg < self.patient_knee_max_deg:
            raise ValueError("patient knee envelope must be increasing")
        if (
            self.patient_hip_min_deg < FORMAL_HIP_ROM_DEG[0]
            or self.patient_hip_max_deg > FORMAL_HIP_ROM_DEG[1]
            or self.patient_knee_min_deg < FORMAL_KNEE_ROM_DEG[0]
            or self.patient_knee_max_deg > FORMAL_KNEE_ROM_DEG[1]
        ):
            raise ValueError(
                "patient operational envelope must be contained in global model ROM"
            )
        if not str(self.source_status).strip():
            raise ValueError("source_status must not be empty")
        if self.values_are_clinical_safety_limits:
            raise ValueError("this offline module cannot accept clinical safety limits")

    def contains(self, trajectory: pd.DataFrame, *, tolerance_deg: float = 1e-9) -> bool:
        hip = np.rad2deg(trajectory["q_hip_rad"].to_numpy(dtype=float))
        knee = np.rad2deg(trajectory["q_knee_rad"].to_numpy(dtype=float))
        return bool(
            np.isfinite(np.column_stack((hip, knee))).all()
            and np.min(hip) >= self.patient_hip_min_deg - tolerance_deg
            and np.max(hip) <= self.patient_hip_max_deg + tolerance_deg
            and np.min(knee) >= self.patient_knee_min_deg - tolerance_deg
            and np.max(knee) <= self.patient_knee_max_deg + tolerance_deg
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "constraint_layer": PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE,
            "auto_expand": AUTO_EXPAND_PATIENT_ENVELOPE,
            "is_global_model_rom": False,
            "is_real_robot_hard_safeguard": False,
        }


@dataclass(frozen=True)
class IdentificationExcitationSpec:
    candidate_id: str
    hip_amplitude_delta_deg: float
    knee_amplitude_delta_deg: float
    knee_phase_shift: float
    excitation_duration_s: float
    duration_design_status: str = RESEARCH_DURATION_LABEL

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.hip_amplitude_delta_deg,
                self.knee_amplitude_delta_deg,
                self.knee_phase_shift,
                self.excitation_duration_s,
            ),
            dtype=float,
        )
        if not str(self.candidate_id).strip() or not np.isfinite(values).all():
            raise ValueError("identification excitation specification is invalid")
        if self.excitation_duration_s <= 0.0:
            raise ValueError("excitation_duration_s must be positive")
        if self.duration_design_status != RESEARCH_DURATION_LABEL:
            raise ValueError("duration must remain a research-design value")
        for name in (
            "hip_amplitude_delta_deg",
            "knee_amplitude_delta_deg",
            "knee_phase_shift",
        ):
            lower, upper = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name]
            value = float(getattr(self, name))
            if value < lower or value > upper:
                raise ValueError(f"{name} lies outside the existing offline design bounds")

    @property
    def excursion_norm(self) -> float:
        return float(
            math.sqrt(
                self.hip_amplitude_delta_deg**2
                + self.knee_amplitude_delta_deg**2
                + (100.0 * self.knee_phase_shift) ** 2
            )
        )


@dataclass(frozen=True)
class ResearchIdentifiabilityStopRule:
    """Explicit virtual-research comparator, never a patient release rule."""

    minimum_rank: int
    minimum_singular_value: float
    maximum_condition_number: float
    maximum_abs_parameter_correlation: float
    maximum_uncertainty_proxy: float
    minimum_parameter_sensitivity: float
    maximum_validation_rmse_nm: float
    rule_status: str = VIRTUAL_RESEARCH_STOP_RULE_STATUS

    def __post_init__(self) -> None:
        if self.rule_status != VIRTUAL_RESEARCH_STOP_RULE_STATUS:
            raise ValueError("research stop rule must retain its non-frozen status")
        values = np.asarray(
            (
                self.minimum_singular_value,
                self.maximum_condition_number,
                self.maximum_abs_parameter_correlation,
                self.maximum_uncertainty_proxy,
                self.minimum_parameter_sensitivity,
                self.maximum_validation_rmse_nm,
            ),
            dtype=float,
        )
        if self.minimum_rank != len(PARAMETER_NAMES):
            raise ValueError("all five parameters must be required")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("research stop-rule values must be finite and non-negative")
        if self.maximum_abs_parameter_correlation > 1.0:
            raise ValueError("correlation threshold cannot exceed one")


@dataclass(frozen=True)
class GeneratedIdentificationExcitation:
    spec: IdentificationExcitationSpec
    trajectory: pd.DataFrame
    global_constraint_audit: dict[str, Any]
    patient_envelope_valid: bool
    candidate_valid: bool
    invalid_reason: str
    trajectory_sha256: str


@dataclass(frozen=True)
class SequentialIdentificationResult:
    subject_id: str
    truth_scenario: str
    status: str
    trials_required: int
    theta_hat_0: dict[str, float] | None
    d_init: pd.DataFrame | None
    executed_identification_data: pd.DataFrame
    trial_history: pd.DataFrame
    trial_candidates: pd.DataFrame
    parameter_identifiability: pd.DataFrame
    parameter_estimates: pd.DataFrame
    incremental_information_gain: pd.DataFrame
    patient_envelope_history: pd.DataFrame
    summary: dict[str, Any]
    selection_audit: dict[str, Any]


def default_virtual_patient_envelope() -> PatientOperationalEnvelope:
    return PatientOperationalEnvelope(
        envelope_id="VIRTUAL_RESEARCH_ENVELOPE_DEFAULT",
        patient_hip_min_deg=20.0,
        patient_hip_max_deg=115.0,
        patient_knee_min_deg=15.0,
        patient_knee_max_deg=135.0,
    )


def limited_rom_virtual_patient_envelope() -> PatientOperationalEnvelope:
    return PatientOperationalEnvelope(
        envelope_id="LIMITED_ROM_VIRTUAL_SUBJECT_SYNTHETIC_FIXTURE",
        patient_hip_min_deg=28.70,
        patient_hip_max_deg=112.10,
        patient_knee_min_deg=18.20,
        patient_knee_max_deg=124.90,
    )


def default_virtual_research_candidate_pool() -> tuple[IdentificationExcitationSpec, ...]:
    """Predeclared duration-sensitivity pool; values are not human limits."""

    raw = (
        ("ID-A-neutral-d20", 0.0, 0.0, 0.0, 20.0),
        ("ID-B-hip-d16", -1.5, 0.0, 0.0, 16.0),
        ("ID-C-knee-d16", 0.0, -1.5, 0.0, 16.0),
        ("ID-D-phase-pos-d18", 0.0, 0.0, 0.018, 18.0),
        ("ID-E-phase-neg-d18", 0.0, 0.0, -0.018, 18.0),
        ("ID-F-coupled-pos-d14", -1.0, -1.0, 0.015, 14.0),
        ("ID-G-coupled-neg-d14", -1.0, -1.0, -0.015, 14.0),
        ("ID-H-hip-d22", -2.0, 0.0, 0.0, 22.0),
        ("ID-I-knee-d22", 0.0, -2.0, 0.0, 22.0),
        ("ID-J-opposed-d18", 1.0, -1.0, 0.0, 18.0),
        ("ID-K-opposed-d20", -1.0, 1.0, 0.0, 20.0),
        ("ID-L-slow-phase-d28", -0.5, -0.5, 0.020, 28.0),
    )
    return tuple(IdentificationExcitationSpec(*values) for values in raw)


def _dataframe_sha256(dataframe: pd.DataFrame) -> str:
    payload = dataframe.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trajectory_sha256(dataframe: pd.DataFrame) -> str:
    return _dataframe_sha256(dataframe.loc[:, _HASH_COLUMNS])


def _append_reason(reasons: list[str], valid: bool, reason: str) -> None:
    if not valid:
        reasons.append(reason)


def _global_constraint_audit(trajectory: pd.DataFrame) -> dict[str, Any]:
    qh = trajectory["q_hip_rad"].to_numpy(dtype=float)
    qk = trajectory["q_knee_rad"].to_numpy(dtype=float)
    dqh = trajectory["dq_hip_rad_s"].to_numpy(dtype=float)
    dqk = trajectory["dq_knee_rad_s"].to_numpy(dtype=float)
    ddqh = trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float)
    ddqk = trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float)
    time = trajectory["time_s"].to_numpy(dtype=float)
    theta = trajectory["theta_shank_rad"].to_numpy(dtype=float)
    finite = bool(
        np.isfinite(np.column_stack((time, qh, qk, dqh, dqk, ddqh, ddqk, theta))).all()
    )
    time_valid = bool(len(time) >= 3 and np.all(np.diff(time) > 0.0))
    theta_valid = bool(np.allclose(theta, qh - qk, atol=1e-12, rtol=0.0))
    hip_deg = np.rad2deg(qh)
    knee_deg = np.rad2deg(qk)
    rom_valid = bool(
        np.min(hip_deg) >= FORMAL_HIP_ROM_DEG[0]
        and np.max(hip_deg) <= FORMAL_HIP_ROM_DEG[1]
        and np.min(knee_deg) >= FORMAL_KNEE_ROM_DEG[0]
        and np.max(knee_deg) <= FORMAL_KNEE_ROM_DEG[1]
    )
    xk, zk, xp, zp = forward_kinematics(qh, qk, L1, L2)
    workspace_valid = bool(
        np.isfinite(np.column_stack((xk, zk, xp, zp))).all()
        and np.all(xp >= 0.0)
        and np.all(zp >= 0.0)
        and np.all(zk >= 0.0)
    )
    jac = jacobian_diagnostics(qh, qk, L1, L2)
    determinant = np.asarray(jac.determinant, dtype=float)
    condition = np.asarray(jac.condition_number, dtype=float)
    jacobian_valid = bool(
        np.isfinite(np.column_stack((determinant, condition))).all()
        and not np.asarray(jac.near_singular, dtype=bool).any()
        and np.all(np.abs(determinant) >= jacobian_det_threshold)
        and np.all(condition <= jacobian_condition_limit)
    )
    baseline = get_dynamic_subject("baseline")
    dynamics = inverse_dynamics(qh, qk, dqh, dqk, ddqh, ddqk, baseline, L1)
    force = endpoint_force_from_joint_torque(
        qh,
        qk,
        dynamics.tau_total_hip_nm,
        dynamics.tau_total_knee_nm,
        L1,
        L2,
    )
    force_mapping_valid = bool(
        np.asarray(force.force_mapping_valid, dtype=bool).all()
        and np.isfinite(force.force_magnitude_n).all()
        and np.max(force.force_magnitude_n) <= force_magnitude_limit_n
    )
    seam_c2_valid = bool(
        np.allclose([dqh[0], dqk[0], dqh[-1], dqk[-1]], 0.0, atol=1e-10)
        and np.allclose([ddqh[0], ddqk[0], ddqh[-1], ddqk[-1]], 0.0, atol=1e-9)
        and np.allclose([qh[0], qk[0]], [qh[-1], qk[-1]], atol=1e-10)
    )
    gates = {
        "finite_valid": finite,
        "strictly_increasing_time_valid": time_valid,
        "theta_shank_valid": theta_valid,
        "global_model_rom_valid": rom_valid,
        "workspace_valid": workspace_valid,
        "jacobian_valid": jacobian_valid,
        "force_mapping_valid": force_mapping_valid,
        "c2_cycle_seam_valid": seam_c2_valid,
    }
    return {
        "constraint_layer": GLOBAL_MODEL_CONSTRAINTS,
        "global_model_rom_is_patient_safety_rom": False,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        **gates,
        "all_global_model_constraints_valid": bool(all(gates.values())),
        "minimum_abs_jacobian_determinant": float(np.min(np.abs(determinant))),
        "maximum_jacobian_condition": float(np.max(condition)),
        "maximum_baseline_model_force_n": float(np.max(force.force_magnitude_n)),
    }


def generate_identification_excitation(
    spec: IdentificationExcitationSpec,
    envelope: PatientOperationalEnvelope,
) -> GeneratedIdentificationExcitation:
    """Generate a variable-duration C2 excitation without changing reference."""

    if not isinstance(spec, IdentificationExcitationSpec):
        raise TypeError("spec must be IdentificationExcitationSpec")
    if not isinstance(envelope, PatientOperationalEnvelope):
        raise TypeError("envelope must be PatientOperationalEnvelope")
    validate_active_reference_file()
    generated = generate_personalized_trajectory(
        hip_amplitude_delta_deg=spec.hip_amplitude_delta_deg,
        knee_amplitude_delta_deg=spec.knee_amplitude_delta_deg,
        knee_phase_shift=spec.knee_phase_shift,
    )
    trajectory = generated.trajectory.copy(deep=True)
    scale = float(TOTAL_DURATION_S / spec.excitation_duration_s)
    trajectory["time_s"] = (
        trajectory["time_s"].to_numpy(dtype=float) / scale
    )
    for column in ("dq_hip_rad_s", "dq_knee_rad_s"):
        trajectory[column] = trajectory[column].to_numpy(dtype=float) * scale
    for column in ("ddq_hip_rad_s2", "ddq_knee_rad_s2"):
        trajectory[column] = trajectory[column].to_numpy(dtype=float) * scale**2
    if "minimum_jerk_phase_rate_s_inv" in trajectory:
        trajectory["minimum_jerk_phase_rate_s_inv"] *= scale
    if "minimum_jerk_phase_acceleration_s_inv2" in trajectory:
        trajectory["minimum_jerk_phase_acceleration_s_inv2"] *= scale**2
    trajectory["theta_shank_rad"] = (
        trajectory["q_hip_rad"].to_numpy(dtype=float)
        - trajectory["q_knee_rad"].to_numpy(dtype=float)
    )
    trajectory["trajectory_id"] = spec.candidate_id
    trajectory["excitation_duration_s"] = spec.excitation_duration_s
    trajectory["duration_design_status"] = spec.duration_design_status
    trajectory["parent_reference_id"] = ACTIVE_REFERENCE_ID
    trajectory["parent_reference_sha256"] = ACTIVE_REFERENCE_SHA256
    trajectory["identification_protocol"] = PROTOCOL_ID
    trajectory["formal_robot_execution_allowed"] = False
    audit = _global_constraint_audit(trajectory)
    patient_valid = envelope.contains(trajectory)
    reasons: list[str] = []
    _append_reason(
        reasons,
        bool(audit["all_global_model_constraints_valid"]),
        "global_model_constraints_invalid",
    )
    _append_reason(reasons, patient_valid, "outside_current_patient_operational_envelope")
    return GeneratedIdentificationExcitation(
        spec=spec,
        trajectory=trajectory,
        global_constraint_audit=audit,
        patient_envelope_valid=patient_valid,
        candidate_valid=not reasons,
        invalid_reason=";".join(reasons),
        trajectory_sha256=_trajectory_sha256(trajectory),
    )


def _design_observations(
    excitation: GeneratedIdentificationExcitation,
    template: BaselineSubjectTemplate,
    parameters: Mapping[str, float],
) -> pd.DataFrame:
    trajectory = excitation.trajectory
    tau_hip, tau_knee = predict_joint_torque(trajectory, template, parameters, L1)
    force = endpoint_force_from_joint_torque(
        trajectory["q_hip_rad"].to_numpy(dtype=float),
        trajectory["q_knee_rad"].to_numpy(dtype=float),
        tau_hip,
        tau_knee,
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
        }
    )


def _matrix_metrics(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("identifiability matrix must have five columns")
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    minimum = float(singular[-1])
    condition = float(singular[0] / minimum) if minimum > tolerance else float("inf")
    information = matrix.T @ matrix
    covariance_shape = np.linalg.pinv(information)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance_shape), 0.0))
    denom = np.outer(uncertainty, uncertainty)
    correlation = np.divide(
        covariance_shape,
        denom,
        out=np.zeros_like(covariance_shape),
        where=denom > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    off_diagonal = np.abs(correlation - np.eye(len(PARAMETER_NAMES)))
    pair_index = np.unravel_index(int(np.argmax(off_diagonal)), off_diagonal.shape)
    sensitivities = np.sqrt(np.maximum(np.diag(information), 0.0))
    weakest_index = int(np.argmin(sensitivities))
    return {
        "rank": rank,
        "singular_values": [float(value) for value in singular],
        "minimum_singular_value": minimum,
        "condition_number": condition,
        "information_matrix": information,
        "parameter_correlation": correlation,
        "maximum_abs_parameter_correlation": float(off_diagonal[pair_index]),
        "highest_correlation_pair": (
            f"{PARAMETER_NAMES[pair_index[0]]}|{PARAMETER_NAMES[pair_index[1]]}"
            if pair_index[0] != pair_index[1]
            else "NONE"
        ),
        "uncertainty_proxy": uncertainty,
        "parameter_sensitivity": sensitivities,
        "weakest_parameter": PARAMETER_NAMES[weakest_index],
        "weakest_parameter_sensitivity": float(sensitivities[weakest_index]),
    }


def _state_coverage_gain(prior: pd.DataFrame, candidate: pd.DataFrame) -> tuple[float, bool]:
    candidate_state = candidate.loc[:, _STATE_COLUMNS].to_numpy(dtype=float)[::8]
    if prior.empty:
        return 1.0, False
    prior_state = prior.loc[:, _STATE_COLUMNS].to_numpy(dtype=float)[::8]
    combined = np.vstack((prior_state, candidate_state))
    scale = np.std(combined, axis=0)
    scale[scale < 1e-9] = 1.0
    a = prior_state / scale
    b = candidate_state / scale
    minimum_distances = np.full(len(b), np.inf)
    for start in range(0, len(a), 128):
        distance = np.linalg.norm(
            b[:, None, :] - a[None, start : start + 128, :], axis=2
        )
        minimum_distances = np.minimum(minimum_distances, np.min(distance, axis=1))
    gain = float(np.mean(minimum_distances))
    return gain, bool(np.max(minimum_distances) < 1e-8)


def _log_information_gain(prior: np.ndarray, combined: np.ndarray) -> float:
    identity = np.eye(len(PARAMETER_NAMES))
    _, before = np.linalg.slogdet(identity + prior.T @ prior)
    _, after = np.linalg.slogdet(identity + combined.T @ combined)
    return float(after - before)


def select_next_identification_excitation(
    executed_data: pd.DataFrame,
    template: BaselineSubjectTemplate,
    current_parameters: Mapping[str, float],
    envelope: PatientOperationalEnvelope,
    candidate_specs: Sequence[IdentificationExcitationSpec],
    *,
    already_executed_candidate_ids: Iterable[str] = (),
) -> tuple[GeneratedIdentificationExcitation | None, pd.DataFrame, dict[str, Any]]:
    """Select by deterministic identifiability lexicography only.

    The signature intentionally has no truth subject, truth parameters,
    held-out test data, or mechanical-personalization objective.
    """

    prior = executed_data.copy(deep=True)
    if prior.empty:
        prior_matrix = np.empty((0, len(PARAMETER_NAMES)), dtype=float)
        current_diagnosis = {
            "weakest_parameter": "NOT_YET_OBSERVED",
            "highest_correlation_pair": "NOT_YET_OBSERVED",
        }
    else:
        prior_matrix, _, _ = numerical_sensitivity_matrix(
            prior,
            template,
            current_parameters,
            L1,
            L2,
            torque_scales_nm=(1.0, 1.0),
        )
        metrics = _matrix_metrics(prior_matrix)
        current_diagnosis = {
            "weakest_parameter": metrics["weakest_parameter"],
            "highest_correlation_pair": metrics["highest_correlation_pair"],
        }
    executed_ids = set(map(str, already_executed_candidate_ids))
    rows: list[dict[str, Any]] = []
    generated_by_id: dict[str, GeneratedIdentificationExcitation] = {}
    for spec in candidate_specs:
        generated = generate_identification_excitation(spec, envelope)
        generated_by_id[spec.candidate_id] = generated
        not_previously_executed = spec.candidate_id not in executed_ids
        candidate_observation = _design_observations(
            generated, template, current_parameters
        )
        candidate_matrix, _, _ = numerical_sensitivity_matrix(
            candidate_observation,
            template,
            current_parameters,
            L1,
            L2,
            torque_scales_nm=(1.0, 1.0),
        )
        combined = np.vstack((prior_matrix, candidate_matrix))
        metrics = _matrix_metrics(combined)
        coverage_gain, design_duplicate = _state_coverage_gain(prior, candidate_observation)
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "hip_amplitude_delta_deg": spec.hip_amplitude_delta_deg,
                "knee_amplitude_delta_deg": spec.knee_amplitude_delta_deg,
                "knee_phase_shift": spec.knee_phase_shift,
                "excitation_duration_s": spec.excitation_duration_s,
                "duration_design_status": spec.duration_design_status,
                "global_constraints_valid": bool(
                    generated.global_constraint_audit[
                        "all_global_model_constraints_valid"
                    ]
                ),
                "patient_envelope_valid": generated.patient_envelope_valid,
                "not_previously_executed": not_previously_executed,
                "design_duplicate": design_duplicate,
                "candidate_valid": bool(
                    generated.candidate_valid
                    and not_previously_executed
                    and not design_duplicate
                ),
                "invalid_reason": (
                    generated.invalid_reason
                    or ("previously_executed" if not not_previously_executed else "")
                    or ("duplicate_information" if design_duplicate else "")
                ),
                "resulting_rank": metrics["rank"],
                "resulting_minimum_singular_value": metrics[
                    "minimum_singular_value"
                ],
                "resulting_condition_number": metrics["condition_number"],
                "resulting_maximum_abs_correlation": metrics[
                    "maximum_abs_parameter_correlation"
                ],
                "resulting_weakest_parameter": metrics["weakest_parameter"],
                "resulting_weakest_parameter_sensitivity": metrics[
                    "weakest_parameter_sensitivity"
                ],
                "incremental_log_information_gain": _log_information_gain(
                    prior_matrix, combined
                ),
                "incremental_state_regressor_coverage": coverage_gain,
                "excursion_norm": spec.excursion_norm,
                "trajectory_sha256": generated.trajectory_sha256,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return None, table, current_diagnosis
    ranked = table.sort_values(
        [
            "candidate_valid",
            "resulting_rank",
            "resulting_minimum_singular_value",
            "resulting_condition_number",
            "resulting_maximum_abs_correlation",
            "resulting_weakest_parameter_sensitivity",
            "incremental_state_regressor_coverage",
            "excursion_norm",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True, False, False, True, True],
        kind="mergesort",
        ignore_index=True,
    )
    ranked["deterministic_candidate_rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = False
    valid = ranked.loc[ranked["candidate_valid"].astype(bool)]
    if valid.empty:
        return None, ranked, current_diagnosis
    selected_id = str(valid.iloc[0]["candidate_id"])
    ranked.loc[ranked["candidate_id"].eq(selected_id), "selected"] = True
    current_diagnosis["selected_candidate_id"] = selected_id
    return generated_by_id[selected_id], ranked, current_diagnosis


class VirtualIdentificationOracle:
    """Truth-isolated virtual execution; selection never receives this state."""

    def __init__(self, subject_id: str, truth_scenario: str = "matched_linear") -> None:
        base_id = "baseline" if subject_id == "LIMITED_ROM_VIRTUAL_SUBJECT" else subject_id
        base = get_dynamic_subject(base_id)
        self._truth_subject = get_mismatch_scenario(truth_scenario).create_subject(base)
        self.subject_id = str(subject_id)
        self.truth_scenario = str(truth_scenario)
        self.execution_count = 0

    def execute(self, excitation: GeneratedIdentificationExcitation) -> pd.DataFrame:
        if not excitation.candidate_valid:
            raise PermissionError("invalid excitation cannot be virtually executed")
        trajectory = excitation.trajectory
        qh = trajectory["q_hip_rad"].to_numpy(dtype=float)
        qk = trajectory["q_knee_rad"].to_numpy(dtype=float)
        dynamics = mismatch_inverse_dynamics(
            qh,
            qk,
            trajectory["dq_hip_rad_s"].to_numpy(dtype=float),
            trajectory["dq_knee_rad_s"].to_numpy(dtype=float),
            trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float),
            trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float),
            self._truth_subject,
            L1,
            residual_random_seed=get_mismatch_scenario(
                self.truth_scenario
            ).random_seed,
        )
        force = endpoint_force_from_joint_torque(
            qh,
            qk,
            dynamics.tau_total_hip_nm,
            dynamics.tau_total_knee_nm,
            L1,
            L2,
        )
        valid = np.asarray(force.force_mapping_valid, dtype=bool)
        observation = pd.DataFrame(
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
                "trajectory_sample_index": np.arange(len(trajectory), dtype=int),
                "candidate_id": excitation.spec.candidate_id,
                "excitation_duration_s": excitation.spec.excitation_duration_s,
                "trajectory_sha256": excitation.trajectory_sha256,
            }
        )
        observation["within_identification_role"] = np.where(
            observation["trajectory_sample_index"].to_numpy(dtype=int) % 5 == 0,
            "validation",
            "train",
        )
        self.execution_count += 1
        return observation


def _completion_audit(
    metrics: Mapping[str, Any],
    validation_rmse_nm: float,
    rule: ResearchIdentifiabilityStopRule | None,
) -> tuple[bool, list[str], dict[str, bool]]:
    if rule is None:
        return False, [IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW], {
            name: False for name in PARAMETER_NAMES
        }
    sensitivity = np.asarray(metrics["parameter_sensitivity"], dtype=float)
    uncertainty = np.asarray(metrics["uncertainty_proxy"], dtype=float)
    per_parameter = {
        name: bool(
            sensitivity[index] >= rule.minimum_parameter_sensitivity
            and uncertainty[index] <= rule.maximum_uncertainty_proxy
        )
        for index, name in enumerate(PARAMETER_NAMES)
    }
    checks = {
        "rank_insufficient": int(metrics["rank"]) >= rule.minimum_rank,
        "minimum_singular_value_insufficient": float(
            metrics["minimum_singular_value"]
        )
        >= rule.minimum_singular_value,
        "condition_number_exceeded": float(metrics["condition_number"])
        <= rule.maximum_condition_number,
        "parameter_correlation_exceeded": float(
            metrics["maximum_abs_parameter_correlation"]
        )
        <= rule.maximum_abs_parameter_correlation,
        "validation_residual_exceeded": validation_rmse_nm
        <= rule.maximum_validation_rmse_nm,
        "one_or_more_parameters_unsupported": all(per_parameter.values()),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return not reasons, reasons, per_parameter


def run_sequential_initial_identification(
    oracle: VirtualIdentificationOracle,
    envelope: PatientOperationalEnvelope,
    *,
    stop_rule: ResearchIdentifiabilityStopRule | None = None,
    candidate_specs: Sequence[IdentificationExcitationSpec] | None = None,
    max_trials: int = MAX_INITIAL_IDENTIFICATION_TRIALS,
) -> SequentialIdentificationResult:
    """Execute at most five offline virtual trials and fail closed."""

    if not isinstance(oracle, VirtualIdentificationOracle):
        raise TypeError("oracle must be VirtualIdentificationOracle")
    if max_trials != MAX_INITIAL_IDENTIFICATION_TRIALS:
        raise ValueError("formal architecture fixes the maximum at exactly five")
    if AUTO_EXPAND_PATIENT_ENVELOPE:
        raise RuntimeError("patient operational envelope auto-expansion is prohibited")
    specs = tuple(candidate_specs or default_virtual_research_candidate_pool())
    if len({spec.candidate_id for spec in specs}) != len(specs):
        raise ValueError("candidate IDs must be unique")
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    executed = pd.DataFrame()
    current_parameters = dict(identification_initial_guess)
    previous_parameters: dict[str, float] | None = None
    executed_ids: list[str] = []
    history_rows: list[dict[str, Any]] = []
    candidate_tables: list[pd.DataFrame] = []
    ident_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    envelope_rows: list[dict[str, Any]] = []
    completion = False
    completion_reasons = [IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW]
    per_parameter_supported = {name: False for name in PARAMETER_NAMES}
    last_metrics: dict[str, Any] | None = None
    last_validation_rmse = float("nan")

    for trial_id in range(1, MAX_INITIAL_IDENTIFICATION_TRIALS + 1):
        selected, audit, diagnosis = select_next_identification_excitation(
            executed,
            template,
            current_parameters,
            envelope,
            specs,
            already_executed_candidate_ids=executed_ids,
        )
        if not audit.empty:
            audit.insert(0, "trial_id", trial_id)
            audit["diagnosed_weakest_parameter_before_trial"] = diagnosis[
                "weakest_parameter"
            ]
            audit["diagnosed_highest_correlation_pair_before_trial"] = diagnosis[
                "highest_correlation_pair"
            ]
            candidate_tables.append(audit)
        if selected is None:
            completion_reasons = ["NO_NEW_CONSTRAINT_VALID_INFORMATIVE_CANDIDATE"]
            break
        observation = oracle.execute(selected)
        observation.insert(0, "trial_id", trial_id)
        executed = pd.concat((executed, observation), ignore_index=True)
        executed_ids.append(selected.spec.candidate_id)
        training = executed.loc[
            executed["within_identification_role"].eq("train")
        ].copy()
        validation = executed.loc[
            executed["within_identification_role"].eq("validation")
        ].copy()
        estimation = estimate_subject_parameters(
            training,
            template,
            L1,
            L2,
            initial_guess=current_parameters,
        )
        if not estimation.optimizer_success:
            completion_reasons = ["TEMPORARY_PARAMETER_FIT_FAILED"]
            break
        previous_parameters = dict(current_parameters)
        current_parameters = dict(estimation.estimated_parameters)
        sensitivity, _, _ = numerical_sensitivity_matrix(
            training,
            template,
            current_parameters,
            L1,
            L2,
            torque_scales_nm=(1.0, 1.0),
        )
        last_metrics = _matrix_metrics(sensitivity)
        validation_metrics = compute_torque_metrics(
            validation,
            template,
            current_parameters,
            L1,
            L2,
        )
        last_validation_rmse = float(
            validation_metrics["torque_rmse_combined_nm"]
        )
        completion, completion_reasons, per_parameter_supported = _completion_audit(
            last_metrics, last_validation_rmse, stop_rule
        )
        selected_audit = audit.loc[audit["selected"].astype(bool)].iloc[0]
        history_rows.append(
            {
                "subject_id": oracle.subject_id,
                "truth_scenario": oracle.truth_scenario,
                "trial_id": trial_id,
                "candidate_id": selected.spec.candidate_id,
                "excitation_duration_s": selected.spec.excitation_duration_s,
                "duration_design_status": selected.spec.duration_design_status,
                "trajectory_sha256": selected.trajectory_sha256,
                "executed": True,
                "temporary_parameter_name": f"theta_hat_ID_{trial_id}",
                "rank": last_metrics["rank"],
                "minimum_singular_value": last_metrics["minimum_singular_value"],
                "condition_number": last_metrics["condition_number"],
                "maximum_abs_parameter_correlation": last_metrics[
                    "maximum_abs_parameter_correlation"
                ],
                "highest_correlation_pair": last_metrics[
                    "highest_correlation_pair"
                ],
                "weakest_parameter": last_metrics["weakest_parameter"],
                "training_residual_rmse_nm": estimation.residual_statistics[
                    "torque_rmse_combined_nm"
                ],
                "within_identification_validation_residual_rmse_nm": last_validation_rmse,
                "all_five_parameters_supported": all(per_parameter_supported.values()),
                "identification_complete": completion,
                "completion_audit_reason": ";".join(completion_reasons),
                "truth_used_for_selection": False,
                "heldout_test_used_for_selection": False,
                "mechanical_personalization_j_used_for_selection": False,
            }
        )
        gain_rows.append(
            {
                "subject_id": oracle.subject_id,
                "truth_scenario": oracle.truth_scenario,
                "trial_id": trial_id,
                "candidate_id": selected.spec.candidate_id,
                "diagnosed_weakest_parameter_before_trial": diagnosis[
                    "weakest_parameter"
                ],
                "diagnosed_highest_correlation_pair_before_trial": diagnosis[
                    "highest_correlation_pair"
                ],
                "incremental_log_information_gain": float(
                    selected_audit["incremental_log_information_gain"]
                ),
                "incremental_state_regressor_coverage": float(
                    selected_audit["incremental_state_regressor_coverage"]
                ),
                "duplicate_information": bool(selected_audit["design_duplicate"]),
            }
        )
        envelope_rows.append(
            {
                "subject_id": oracle.subject_id,
                "truth_scenario": oracle.truth_scenario,
                "trial_id": trial_id,
                **envelope.as_dict(),
                "candidate_within_envelope": True,
                "constraint_violation_used_to_expand_envelope": False,
            }
        )
        covariance_uncertainty = np.asarray(
            last_metrics["uncertainty_proxy"], dtype=float
        )
        sensitivities = np.asarray(
            last_metrics["parameter_sensitivity"], dtype=float
        )
        correlations = np.asarray(
            last_metrics["parameter_correlation"], dtype=float
        )
        standard_errors = estimation.parameter_standard_errors
        for index, name in enumerate(PARAMETER_NAMES):
            relative_change = (
                abs(current_parameters[name] - previous_parameters[name])
                / _PARAMETER_SCALES[index]
                if previous_parameters is not None
                else float("nan")
            )
            ident_rows.append(
                {
                    "subject_id": oracle.subject_id,
                    "truth_scenario": oracle.truth_scenario,
                    "trial_id": trial_id,
                    "parameter": name,
                    "sensitivity": float(sensitivities[index]),
                    "uncertainty_proxy": float(covariance_uncertainty[index]),
                    "optimizer_standard_error": float(standard_errors[name]),
                    "recovery_stability_relative_change": float(relative_change),
                    "maximum_abs_correlation_with_other_parameter": float(
                        np.max(
                            np.abs(
                                np.delete(correlations[index, :], index)
                            )
                        )
                    ),
                    "parameter_supported_by_research_rule": bool(
                        per_parameter_supported[name]
                    ),
                }
            )
            estimate_rows.append(
                {
                    "subject_id": oracle.subject_id,
                    "truth_scenario": oracle.truth_scenario,
                    "trial_id": trial_id,
                    "temporary_parameter_name": f"theta_hat_ID_{trial_id}",
                    "parameter": name,
                    "estimate": float(current_parameters[name]),
                    "parameter_interpretation": "local_equivalent_dynamics_parameter",
                    "is_tissue_material_constant": False,
                    "frozen_as_theta_hat_0": bool(completion),
                }
            )
        if completion:
            break

    trial_history = pd.DataFrame(history_rows)
    candidate_history = (
        pd.concat(candidate_tables, ignore_index=True)
        if candidate_tables
        else pd.DataFrame()
    )
    identifiability = pd.DataFrame(ident_rows)
    estimates = pd.DataFrame(estimate_rows)
    gains = pd.DataFrame(gain_rows)
    envelope_history = pd.DataFrame(envelope_rows)
    trials_required = int(oracle.execution_count)
    if trials_required > MAX_INITIAL_IDENTIFICATION_TRIALS:
        raise RuntimeError("implementation attempted a prohibited sixth trial")
    if completion:
        status = INITIAL_IDENTIFICATION_COMPLETE
        theta_hat_0 = dict(current_parameters)
        d_init = executed.copy(deep=True)
        d_init_sha = _dataframe_sha256(d_init)
    else:
        status = INITIAL_IDENTIFICATION_INSUFFICIENT
        theta_hat_0 = None
        d_init = None
        d_init_sha = None
    summary = {
        "protocol_id": PROTOCOL_ID,
        "subject_id": oracle.subject_id,
        "truth_scenario": oracle.truth_scenario,
        "status": status,
        "trials_required": trials_required,
        "maximum_initial_identification_trials": MAX_INITIAL_IDENTIFICATION_TRIALS,
        "theta_hat_0_frozen": theta_hat_0 is not None,
        "initial_subject_model": theta_hat_0,
        "initial_identification_trial_count": trials_required if completion else None,
        "initial_identification_dataset_sha": d_init_sha,
        "executed_identification_data_sha": _dataframe_sha256(executed)
        if not executed.empty
        else None,
        "executed_candidate_ids": executed_ids,
        "personalization_interface_ready": completion,
        "personalization_executed": False,
        "real_robot_personalization_allowed": False,
        "real_robot_hard_safeguard": REAL_ROBOT_HARD_SAFEGUARD,
        "identifiability_stop_rule_status": (
            IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW
            if stop_rule is None
            else stop_rule.rule_status
        ),
        "completion_audit_reason": completion_reasons,
        "global_model_rom_is_patient_safety_rom": False,
        "patient_envelope_auto_expanded": False,
        "constraint_violation_used_to_discover_boundary": False,
        "heldout_test_used_for_selection": False,
        "truth_used_for_selection": False,
        "mechanical_personalization_j_used_for_selection": False,
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "active_reference_duration_s": TOTAL_DURATION_S,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "last_validation_rmse_nm": last_validation_rmse,
        "last_identifiability_metrics": (
            {
                key: value
                for key, value in last_metrics.items()
                if key not in {
                    "information_matrix",
                    "parameter_correlation",
                    "uncertainty_proxy",
                    "parameter_sensitivity",
                }
            }
            if last_metrics is not None
            else None
        ),
    }
    selection_audit = {
        "selection_inputs": [
            "executed_identification_data_so_far",
            "current_temporary_five_parameter_estimate",
            "global_model_constraints",
            "current_patient_operational_envelope",
            "predeclared_virtual_research_candidate_pool",
        ],
        "prohibited_inputs_absent": [
            "truth_subject_label",
            "truth_five_parameters",
            "future_trial_outcome",
            "heldout_test",
            "mechanical_personalization_J",
        ],
        "truth_oracle_calls": oracle.execution_count,
        "candidate_selection_calls_precede_truth_execution": True,
    }
    return SequentialIdentificationResult(
        subject_id=oracle.subject_id,
        truth_scenario=oracle.truth_scenario,
        status=status,
        trials_required=trials_required,
        theta_hat_0=theta_hat_0,
        d_init=d_init,
        executed_identification_data=executed,
        trial_history=trial_history,
        trial_candidates=candidate_history,
        parameter_identifiability=identifiability,
        parameter_estimates=estimates,
        incremental_information_gain=gains,
        patient_envelope_history=envelope_history,
        summary=summary,
        selection_audit=selection_audit,
    )


def load_initial_identification_domain_bounds(path: str | Path) -> StateDomainBounds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bounds = payload.get("bounds")
    if not isinstance(bounds, Mapping):
        raise RuntimeError("state-domain bounds are missing")
    return StateDomainBounds(
        columns=tuple(map(str, bounds["columns"])),
        lower=tuple(map(float, bounds["lower"])),
        upper=tuple(map(float, bounds["upper"])),
        valid_training_samples=int(bounds["valid_training_samples"]),
    )


def prediction_support(
    trajectory: pd.DataFrame,
    domain_bounds: StateDomainBounds,
    *,
    minimum_coverage_percent: float = 90.0,
) -> tuple[float, bool]:
    state = trajectory.loc[:, _STATE_COLUMNS].copy(deep=True)
    state.columns = domain_bounds.columns
    state["state_estimation_valid"] = np.isfinite(
        state.loc[:, domain_bounds.columns].to_numpy(dtype=float)
    ).all(axis=1)
    member = np.asarray(classify_state_domain(state, domain_bounds), dtype=bool)
    coverage = 100.0 * float(np.mean(member))
    return coverage, bool(coverage >= minimum_coverage_percent)


def predict_mechanical_cost(
    trajectory: pd.DataFrame,
    template: BaselineSubjectTemplate,
    theta_hat_0: Mapping[str, float],
    reference_metrics: Any,
) -> float:
    hip, knee = predict_joint_torque(trajectory, template, theta_hat_0, L1)
    metrics = mechanical_torque_metrics(trajectory["time_s"], hip, knee)
    return float(
        math.sqrt(
            (
                (metrics.hip_rms_torque_nm / reference_metrics.hip_rms_torque_nm) ** 2
                + (
                    metrics.knee_rms_torque_nm
                    / reference_metrics.knee_rms_torque_nm
                )
                ** 2
            )
            / 2.0
        )
    )


__all__ = [
    "AUTO_EXPAND_PATIENT_ENVELOPE",
    "GLOBAL_MODEL_CONSTRAINTS",
    "IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW",
    "INITIAL_IDENTIFICATION_COMPLETE",
    "INITIAL_IDENTIFICATION_INSUFFICIENT",
    "IdentificationExcitationSpec",
    "MAX_INITIAL_IDENTIFICATION_TRIALS",
    "PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE",
    "PROTOCOL_ID",
    "PatientOperationalEnvelope",
    "REAL_ROBOT_HARD_SAFEGUARD",
    "RESEARCH_DURATION_LABEL",
    "ResearchIdentifiabilityStopRule",
    "SUPPORTED_PREDICTION",
    "SequentialIdentificationResult",
    "UNSUPPORTED_EXTRAPOLATION",
    "VIRTUAL_RESEARCH_STOP_RULE_STATUS",
    "VirtualIdentificationOracle",
    "default_virtual_patient_envelope",
    "default_virtual_research_candidate_pool",
    "generate_identification_excitation",
    "limited_rom_virtual_patient_envelope",
    "load_initial_identification_domain_bounds",
    "predict_mechanical_cost",
    "prediction_support",
    "run_sequential_initial_identification",
    "select_next_identification_excitation",
]
