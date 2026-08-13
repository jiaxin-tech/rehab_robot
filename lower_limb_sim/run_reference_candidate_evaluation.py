"""Stage 5C reference execution, local identification and candidate screening.

The command is fail-closed: without an explicit knee ROM pair it saves the two
reference versions and their audit, but it does not run dynamics, parameter
identification or candidate screening.  This module is software-only and does
not import robot control, acquisition, safety, hardware or SDK code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    hip_range_deg,
    jacobian_condition_limit,
    knee_range_deg,
    reference_trajectory_data_dir,
)
from .force_mapping import endpoint_force_from_joint_torque
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import (
    ESTIMATED_DOMAIN_STATE_COLUMNS,
    StateDomainBounds,
    classify_state_domain,
)
from .kinematics import forward_kinematics
from .formal_protocol import (
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
)
from .parameter_estimator import (
    baseline_template_from_dynamic_subject,
    candidate_subject_from_parameters,
)
from .dynamic_subject import get_dynamic_subject
from .reference_execution_trajectory import (
    CLOSED_REFERENCE,
    MEASURED_REFERENCE,
    ExecutionRomAudit,
    HipRomApproval,
    KneeRomApproval,
    apply_execution_rom_policy,
    build_execution_reference_versions,
    closed_execution_phase_path,
    closure_metrics,
)
from .reference_local_excitation import (
    LOCAL_DOMAIN_MODEL,
    LOCAL_TRAJECTORY_DURATION_S,
    LocalIdentificationResult,
    build_local_domain_coverage,
    build_local_identification_dataset,
    build_local_phase_paths,
    fit_local_identification_domain,
    fit_local_subject_parameters,
    perturb_closed_phase_path,
    retime_local_phase_paths,
)
from .reference_trajectory_retiming import (
    MODEL_ANGLE_DEFINITION,
    load_processed_reference_cycle,
    retime_reference_path,
)


STAGE5C_MODEL_VERSION = "lower_limb_sim_reference_candidates_v1"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data" / "reference_candidates"
SUBJECT_IDS = ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
LOCAL_DOMAIN_MINIMUM_PERCENT = 90.0

CANDIDATE_SPECIFICATIONS: dict[str, dict[str, float | str]] = {
    "C0": {"description": "reference_closed_symmetric_slow"},
    "C1": {"description": "hip_amplitude_minus_3deg", "hip_minus_deg": 3.0},
    "C2": {"description": "hip_amplitude_minus_5deg", "hip_minus_deg": 5.0},
    "C3": {"description": "knee_amplitude_minus_3deg", "knee_minus_deg": 3.0},
    "C4": {"description": "knee_amplitude_minus_5deg", "knee_minus_deg": 5.0},
    "C5": {"description": "knee_phase_advance_3pct", "phase_shift": 0.03},
    "C6": {"description": "knee_phase_delay_3pct", "phase_shift": -0.03},
    "C7": {"description": "total_duration_plus_20pct", "duration_scale": 1.2},
    "C8": {
        "description": "hip_minus_3deg_plus_knee_phase_advance_3pct",
        "hip_minus_deg": 3.0,
        "phase_shift": 0.03,
    },
}


@dataclass(frozen=True)
class CandidateEvaluationResult:
    trajectories: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    feasibility: pd.DataFrame
    subject_comparison: pd.DataFrame
    pareto: pd.DataFrame


@dataclass(frozen=True)
class Stage5CResult:
    execution_versions: pd.DataFrame
    rom_audit: ExecutionRomAudit
    local_identification_dataset: pd.DataFrame
    local_parameter_estimates: pd.DataFrame
    local_domain_coverage: pd.DataFrame
    candidate_metrics: pd.DataFrame
    candidate_feasibility: pd.DataFrame
    candidate_subject_comparison: pd.DataFrame
    candidate_pareto: pd.DataFrame
    candidate_trajectories: dict[str, pd.DataFrame]
    metadata: dict[str, object]
    output_paths: dict[str, Path]
    visualization_paths: dict[str, Path]
    skipped_visualizations: dict[str, str]


def build_candidate_phase_paths(
    reference_versions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the fixed C0--C8 set around the approved closed reference."""

    base = closed_execution_phase_path(reference_versions)
    if not base["formal_execution_allowed"].fillna(False).astype(bool).all():
        raise PermissionError("candidate generation is blocked by the ROM gate.")
    paths: dict[str, pd.DataFrame] = {}
    base_maxima = {
        "hip": float(base["q_hip_reference_rad"].max()),
        "knee": float(base["q_knee_reference_rad"].max()),
    }
    for candidate_id, specification in CANDIDATE_SPECIFICATIONS.items():
        path = perturb_closed_phase_path(
            base,
            trajectory_id=candidate_id,
            hip_amplitude_reduction_deg=float(specification.get("hip_minus_deg", 0.0)),
            knee_amplitude_reduction_deg=float(
                specification.get("knee_minus_deg", 0.0)
            ),
            knee_phase_shift_fraction=float(specification.get("phase_shift", 0.0)),
        )
        path["candidate_id"] = candidate_id
        path["candidate_description"] = str(specification["description"])
        path["candidate_is_fast_stress_test"] = False
        if float(path["q_hip_reference_rad"].max()) > base_maxima["hip"] + 1e-12:
            raise RuntimeError(f"{candidate_id} increased the reference hip maximum.")
        if float(path["q_knee_reference_rad"].max()) > base_maxima["knee"] + 1e-12:
            raise RuntimeError(f"{candidate_id} increased the reference knee maximum.")
        paths[candidate_id] = path
    return paths


def retime_candidate_phase_paths(
    phase_paths: Mapping[str, pd.DataFrame],
    *,
    samples_per_segment: int = 201,
) -> dict[str, pd.DataFrame]:
    """Retime C0--C8; fast is deliberately absent from the candidate set."""

    if set(phase_paths) != set(CANDIDATE_SPECIFICATIONS):
        raise ValueError("candidate phase paths must contain exactly C0 through C8.")
    trajectories: dict[str, pd.DataFrame] = {}
    for candidate_id in CANDIDATE_SPECIFICATIONS:
        scale = float(CANDIDATE_SPECIFICATIONS[candidate_id].get("duration_scale", 1.0))
        one_way_duration = 12.0 * scale
        trajectory = retime_reference_path(
            phase_paths[candidate_id],
            profile=candidate_id,
            flexion_duration_s=one_way_duration,
            extension_duration_s=one_way_duration,
            samples_per_segment=samples_per_segment,
        )
        trajectory.insert(0, "candidate_id", candidate_id)
        trajectory["candidate_description"] = str(
            CANDIDATE_SPECIFICATIONS[candidate_id]["description"]
        )
        trajectory["main_candidate_profile"] = "slow"
        trajectory["software_stress_test"] = False
        trajectory["ranking_eligible_profile"] = True
        trajectories[candidate_id] = trajectory
    return trajectories


def _state_for_domain(trajectory: pd.DataFrame) -> pd.DataFrame:
    source = (
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    states = trajectory.loc[:, source].copy(deep=True)
    states.columns = ESTIMATED_DOMAIN_STATE_COLUMNS
    states["state_estimation_valid"] = trajectory["trajectory_sample_valid"].astype(
        bool
    ).to_numpy()
    return states


def _peak_abs(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else np.nan


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan


def _trajectory_smoothness(trajectory: pd.DataFrame) -> float:
    time_s = trajectory["time_s"].to_numpy(float)
    jerk_hip = np.gradient(
        trajectory["ddq_hip_rad_s2"].to_numpy(float), time_s, edge_order=2
    )
    jerk_knee = np.gradient(
        trajectory["ddq_knee_rad_s2"].to_numpy(float), time_s, edge_order=2
    )
    return float(np.trapezoid(jerk_hip**2 + jerk_knee**2, time_s))


def _append_reason(reasons: list[str], condition: bool, token: str) -> None:
    if condition:
        reasons.append(token)


def evaluate_candidate_trajectories(
    trajectories: Mapping[str, pd.DataFrame],
    estimated_parameters_by_subject: Mapping[str, Mapping[str, float]],
    domain_bounds: StateDomainBounds,
    *,
    approved_knee_rom: KneeRomApproval,
    approved_hip_rom: HipRomApproval | None = None,
    minimum_domain_percent: float = LOCAL_DOMAIN_MINIMUM_PERCENT,
    condition_limit: float = jacobian_condition_limit,
) -> CandidateEvaluationResult:
    """Evaluate C0--C8 with fitted parameters and fixed hard constraints."""

    if set(trajectories) != set(CANDIDATE_SPECIFICATIONS):
        raise ValueError("candidate trajectories must contain exactly C0 through C8.")
    if set(estimated_parameters_by_subject) != set(SUBJECT_IDS):
        raise ValueError("estimated parameters are required for all four subjects.")
    if not np.isfinite(minimum_domain_percent) or not 0.0 <= minimum_domain_percent <= 100.0:
        raise ValueError("minimum_domain_percent must lie in [0, 100].")

    baseline_template = baseline_template_from_dynamic_subject(
        get_dynamic_subject("baseline")
    )
    subject_rows: list[dict[str, object]] = []
    feasibility_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    for candidate_id in CANDIDATE_SPECIFICATIONS:
        trajectory = trajectories[candidate_id]
        q_hip = trajectory["q_hip_rad"].to_numpy(float)
        q_knee = trajectory["q_knee_rad"].to_numpy(float)
        time_s = trajectory["time_s"].to_numpy(float)
        theta_ok = np.allclose(
            trajectory["theta_shank_rad"].to_numpy(float), q_hip - q_knee, atol=1e-14
        )
        active_hip_rom = (
            hip_range_deg
            if approved_hip_rom is None
            else approved_hip_rom.as_list()
        )
        hip_min, hip_max = np.deg2rad(np.asarray(active_hip_rom, dtype=float))
        knee_min, knee_max = np.deg2rad(
            np.asarray(approved_knee_rom.as_list(), dtype=float)
        )
        rom_valid = bool(
            theta_ok
            and np.all((q_hip >= hip_min - 1e-12) & (q_hip <= hip_max + 1e-12))
            and np.all((q_knee >= knee_min - 1e-12) & (q_knee <= knee_max + 1e-12))
        )
        x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
        geometry = np.column_stack((x_knee, z_knee, x_pull, z_pull))
        workspace_valid = bool(
            np.isfinite(geometry).all()
            and np.all(x_pull >= -1e-12)
            and np.all(z_pull >= -1e-12)
            and np.all(z_knee >= -1e-12)
        )
        closure = closure_metrics(trajectory)
        closed = bool(
            abs(closure["q_hip_closure_error_deg"]) <= 1e-8
            and abs(closure["q_knee_closure_error_deg"]) <= 1e-8
            and closure["pull_point_closure_error_m"] <= 1e-10
        )
        states = _state_for_domain(trajectory)
        domain_membership = classify_state_domain(states, domain_bounds)
        domain_percent = 100.0 * float(domain_membership.sum()) / len(trajectory)
        domain_valid = domain_percent >= minimum_domain_percent
        trajectory["domain_membership_estimated"] = domain_membership
        trajectory["domain_model"] = LOCAL_DOMAIN_MODEL

        candidate_subject_rows: list[dict[str, object]] = []
        all_force_valid = True
        maximum_condition = 0.0
        for subject_id in SUBJECT_IDS:
            dynamic_subject = candidate_subject_from_parameters(
                baseline_template, estimated_parameters_by_subject[subject_id]
            )
            dynamics = inverse_dynamics(
                q_hip,
                q_knee,
                trajectory["dq_hip_rad_s"].to_numpy(float),
                trajectory["dq_knee_rad_s"].to_numpy(float),
                trajectory["ddq_hip_rad_s2"].to_numpy(float),
                trajectory["ddq_knee_rad_s2"].to_numpy(float),
                dynamic_subject,
                L1,
            )
            tau_hip = np.asarray(dynamics.tau_total_hip_nm, dtype=float)
            tau_knee = np.asarray(dynamics.tau_total_knee_nm, dtype=float)
            force = endpoint_force_from_joint_torque(
                q_hip, q_knee, tau_hip, tau_knee, L1, L2
            )
            force_valid = np.asarray(force.force_mapping_valid, dtype=bool)
            condition = np.asarray(force.jacobian_condition_number, dtype=float)
            finite_condition = condition[np.isfinite(condition)]
            subject_max_condition = (
                float(np.max(finite_condition)) if finite_condition.size else np.inf
            )
            maximum_condition = max(maximum_condition, subject_max_condition)
            all_force_valid = all_force_valid and bool(force_valid.all())
            torque_rate_hip = np.gradient(tau_hip, time_s, edge_order=2)
            torque_rate_knee = np.gradient(tau_knee, time_s, edge_order=2)
            force_magnitude = np.asarray(force.force_magnitude_n, dtype=float)
            row = {
                "candidate_id": candidate_id,
                "candidate_description": CANDIDATE_SPECIFICATIONS[candidate_id][
                    "description"
                ],
                "subject_id": subject_id,
                "dynamics_parameter_source": "reference_local_estimated_parameters",
                "peak_abs_tau_hip_nm": _peak_abs(tau_hip),
                "peak_abs_tau_knee_nm": _peak_abs(tau_knee),
                "rms_combined_torque_nm": float(
                    np.sqrt(np.mean(tau_hip**2 + tau_knee**2))
                ),
                "peak_torque_rate_nm_s": _peak_abs(
                    np.hypot(torque_rate_hip, torque_rate_knee)
                ),
                "joint_jerk_cost": _trajectory_smoothness(trajectory),
                "maximum_jacobian_condition": subject_max_condition,
                "out_of_domain_percent": 100.0 - domain_percent,
                "peak_force_n": _peak_abs(force_magnitude),
                "rms_force_n": _rms(force_magnitude),
                "force_mapping_valid_percent": 100.0 * float(force_valid.sum()) / len(force_valid),
                "total_duration_s": float(time_s[-1] - time_s[0]),
                "force_is_software_relative_metric_only": True,
                "force_is_real_robot_safety_threshold": False,
                "simulation_status": "software_only",
            }
            candidate_subject_rows.append(row)
            subject_rows.append(row)

        reasons: list[str] = []
        _append_reason(reasons, not rom_valid, "rom_violation")
        _append_reason(reasons, not closed, "not_closed")
        _append_reason(reasons, not workspace_valid, "workspace_invalid")
        _append_reason(reasons, not all_force_valid, "force_mapping_invalid")
        _append_reason(
            reasons,
            not np.isfinite(maximum_condition) or maximum_condition > condition_limit,
            "jacobian_condition_exceeded",
        )
        _append_reason(reasons, not domain_valid, "identification_domain_insufficient")
        feasible = not reasons
        feasibility_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_description": CANDIDATE_SPECIFICATIONS[candidate_id][
                    "description"
                ],
                "candidate_feasible": feasible,
                "infeasible_reasons": ";".join(reasons),
                "rom_valid": rom_valid,
                "closed": closed,
                "workspace_valid": workspace_valid,
                "force_mapping_valid": all_force_valid,
                "jacobian_condition_valid": maximum_condition <= condition_limit,
                "identification_domain_valid": domain_valid,
                "in_domain_percent": domain_percent,
                "minimum_required_in_domain_percent": minimum_domain_percent,
                "maximum_jacobian_condition": maximum_condition,
                "jacobian_condition_limit": condition_limit,
                **closure,
            }
        )
        frame = pd.DataFrame(candidate_subject_rows)
        metrics_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_description": CANDIDATE_SPECIFICATIONS[candidate_id][
                    "description"
                ],
                "candidate_feasible": feasible,
                "peak_abs_tau_hip_nm": float(frame["peak_abs_tau_hip_nm"].max()),
                "peak_abs_tau_knee_nm": float(frame["peak_abs_tau_knee_nm"].max()),
                "rms_combined_torque_nm": float(frame["rms_combined_torque_nm"].max()),
                "peak_torque_rate_nm_s": float(frame["peak_torque_rate_nm_s"].max()),
                "joint_jerk_cost": float(frame["joint_jerk_cost"].iloc[0]),
                "maximum_jacobian_condition": maximum_condition,
                "out_of_domain_percent": 100.0 - domain_percent,
                "peak_force_n": float(frame["peak_force_n"].max()),
                "rms_force_n": float(frame["rms_force_n"].max()),
                "force_mapping_valid_percent": float(
                    frame["force_mapping_valid_percent"].min()
                ),
                "total_duration_s": float(time_s[-1] - time_s[0]),
                "ranking_eligible": feasible,
                "fast_excluded_from_main_ranking": True,
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    feasibility = pd.DataFrame(feasibility_rows)
    comparison = pd.DataFrame(subject_rows)
    pareto = build_candidate_pareto(metrics)
    return CandidateEvaluationResult(
        trajectories=dict(trajectories),
        metrics=metrics,
        feasibility=feasibility,
        subject_comparison=comparison,
        pareto=pareto,
    )


def build_candidate_pareto(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute an unweighted non-dominated comparison for feasible candidates."""

    objectives = [
        "peak_abs_tau_hip_nm",
        "peak_abs_tau_knee_nm",
        "rms_combined_torque_nm",
        "joint_jerk_cost",
        "total_duration_s",
    ]
    required = {"candidate_id", "candidate_feasible", *objectives}
    missing = required.difference(candidate_metrics.columns)
    if missing:
        raise ValueError(f"candidate metrics missing Pareto fields: {sorted(missing)}")
    feasible = candidate_metrics.loc[
        candidate_metrics["candidate_feasible"].astype(bool)
    ].copy(deep=True)
    rows: list[dict[str, object]] = []
    values = feasible.loc[:, objectives].to_numpy(float)
    for index, (_, row) in enumerate(feasible.iterrows()):
        dominated_by: list[str] = []
        for other_index, (_, other) in enumerate(feasible.iterrows()):
            if index == other_index:
                continue
            a = values[other_index]
            b = values[index]
            if np.all(a <= b + 1e-12) and np.any(a < b - 1e-12):
                dominated_by.append(str(other["candidate_id"]))
        rows.append(
            {
                **row.to_dict(),
                "pareto_front": not dominated_by,
                "dominated_by": ";".join(dominated_by),
                "pareto_method": "unweighted_minimize_five_objectives",
                "comfort_score_computed": False,
            }
        )
    return pd.DataFrame(rows)


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {
        "local_dataset": pd.DataFrame(
            columns=["subject_id", "trajectory_id", "dataset_split", "sample_valid"]
        ),
        "parameter_estimates": pd.DataFrame(
            columns=["subject_id", "parameter", "estimated_value"]
        ),
        "domain_coverage": pd.DataFrame(
            columns=[
                "trajectory_id",
                "in_domain_sample_count",
                "out_of_domain_sample_count",
                "in_domain_percent",
            ]
        ),
        "candidate_metrics": pd.DataFrame(
            columns=["candidate_id", "candidate_feasible"]
        ),
        "candidate_feasibility": pd.DataFrame(
            columns=["candidate_id", "candidate_feasible", "infeasible_reasons"]
        ),
        "candidate_subject_comparison": pd.DataFrame(
            columns=["candidate_id", "subject_id"]
        ),
        "candidate_pareto": pd.DataFrame(
            columns=["candidate_id", "pareto_front", "dominated_by"]
        ),
    }


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, StateDomainBounds):
        return value.as_serializable_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _build_metadata(
    *,
    source_metadata: Mapping[str, object],
    rom_audit: ExecutionRomAudit,
    execution_versions: pd.DataFrame,
    local_result: LocalIdentificationResult | None,
    candidate_result: CandidateEvaluationResult | None,
    rom_approval_source: str | None,
) -> dict[str, object]:
    measured = execution_versions.loc[
        execution_versions["reference_version"].eq(MEASURED_REFERENCE)
    ]
    closed = execution_versions.loc[
        execution_versions["reference_version"].eq(CLOSED_REFERENCE)
    ]
    metadata: dict[str, object] = {
        "stage": "5C_reference_execution_local_identification_candidate_screening",
        "model_version": STAGE5C_MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stage": "5A_processed_cycle_and_5B_path_shape",
        "source_cycle": source_metadata.get("selected_cycle"),
        "source_timing_status": "unknown",
        "retimed_trajectory": True,
        "retimed_timing_is_original": False,
        "model_angle_definition": MODEL_ANGLE_DEFINITION,
        "configured_hip_range_deg": list(map(float, hip_range_deg)),
        "configured_knee_range_deg": list(map(float, knee_range_deg)),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "rom_audit": rom_audit.as_dict(),
        "approved_hip_rom_deg": list(rom_audit.approved_hip_range_deg),
        "approved_knee_rom_deg": (
            None
            if rom_audit.approved_knee_range_deg is None
            else list(rom_audit.approved_knee_range_deg)
        ),
        "rom_approval_status": (
            "approved"
            if rom_audit.hip_approval_supplied
            and rom_audit.knee_approval_supplied
            and rom_audit.formal_execution_allowed
            else "not_fully_approved"
        ),
        "rom_approval_source": rom_approval_source,
        "reference_max_knee_flexion_deg": float(
            np.rad2deg(closed["q_knee_original_rad"].max())
        ),
        "reference_path_preserved": bool(
            not rom_audit.rom_mapping_applied
            and np.array_equal(
                closed["q_hip_reference_rad"].to_numpy(float),
                closed["q_hip_original_rad"].to_numpy(float),
            )
            and np.array_equal(
                closed["q_knee_reference_rad"].to_numpy(float),
                closed["q_knee_original_rad"].to_numpy(float),
            )
        ),
        "approved_knee_min_deg": (
            None
            if rom_audit.approved_knee_range_deg is None
            else rom_audit.approved_knee_range_deg[0]
        ),
        "approved_knee_max_deg": (
            None
            if rom_audit.approved_knee_range_deg is None
            else rom_audit.approved_knee_range_deg[1]
        ),
        "formal_execution_allowed": rom_audit.formal_execution_allowed,
        "formal_evaluation_allowed": rom_audit.formal_execution_allowed,
        "rom_mapping_applied": rom_audit.rom_mapping_applied,
        "mapping_formula": rom_audit.mapping_formula,
        "measured_reference": {
            "repeatable_loop": False,
            "closure": closure_metrics(measured),
            "q_hip_range_deg": [
                float(np.rad2deg(measured["q_hip_original_rad"].min())),
                float(np.rad2deg(measured["q_hip_original_rad"].max())),
            ],
            "q_knee_range_deg": [
                float(np.rad2deg(measured["q_knee_original_rad"].min())),
                float(np.rad2deg(measured["q_knee_original_rad"].max())),
            ],
            "extension_provenance": "measured_skeleton_extension",
        },
        "closed_reference": {
            "repeatable_loop": True,
            "closure": closure_metrics(closed),
            "q_hip_range_deg": [
                float(np.rad2deg(closed["q_hip_reference_rad"].min())),
                float(np.rad2deg(closed["q_hip_reference_rad"].max())),
            ],
            "q_knee_range_deg": [
                float(np.rad2deg(closed["q_knee_reference_rad"].min())),
                float(np.rad2deg(closed["q_knee_reference_rad"].max())),
            ],
            "extension_provenance": "synthetic_time_reverse_of_measured_flexion",
            "symmetric_return_is_original_measurement": False,
        },
        "main_speed_profiles": {
            "slow": {"flexion_s": 12.0, "extension_s": 12.0},
            "nominal": {"flexion_s": 6.0, "extension_s": 6.0},
        },
        "fast_profile": {
            "status": "software_stress_test_only",
            "included_in_candidate_ranking": False,
        },
        "local_dataset_split": {
            "train": [
                "reference_slow",
                "reference_nominal",
                "hip_amplitude_minus_3deg",
                "knee_amplitude_minus_3deg",
            ],
            "validation": ["knee_phase_advance_3pct"],
            "test": ["knee_phase_delay_3pct"],
        },
        "test_used_for_parameter_fit": False,
        "test_used_for_domain_fit": False,
        "test_used_for_candidate_adjustment": False,
        "domain_model": LOCAL_DOMAIN_MODEL,
        "domain_minimum_percent_fixed_before_evaluation": LOCAL_DOMAIN_MINIMUM_PERCENT,
        "nearest_neighbour_threshold_widened": False,
        "candidate_dynamics_parameter_source": "reference_local_estimated_parameters",
        "L1_m": L1,
        "L2_m": L2,
        "L2_definition": "knee_to_strap_equivalent_pull_point",
        "observed_ankle_is_pull_point": False,
        "force_metrics_are_software_relative_only": True,
        "force_metrics_are_real_robot_safety_thresholds": False,
        "clinical_validation_status": "not_clinically_validated",
        "simulation_status": "software_only",
        "real_robot_code_used": False,
        "hardware_used": False,
        "repository_real_robot_path_audit_status": (
            "not_asserted_by_runtime; verify with version-control path audit"
        ),
        "formal_outputs_generated": local_result is not None and candidate_result is not None,
        "planarity_rmse_m": source_metadata.get("planarity_rmse_m"),
        "planarity_max_error_m": source_metadata.get("planarity_max_error_m"),
        "geometry_uncertainty_remains": True,
        "software_version_or_git_commit": "workspace_snapshot_no_commit_created",
    }
    if local_result is not None:
        metadata["local_domain_bounds"] = local_result.domain_bounds
        metadata["local_parameter_estimates"] = (
            local_result.parameter_estimates.to_dict(orient="records")
        )
        metadata["local_domain_coverage"] = (
            local_result.domain_coverage.to_dict(orient="records")
        )
    if candidate_result is not None:
        metadata["pareto_candidates"] = candidate_result.pareto.loc[
            candidate_result.pareto.get("pareto_front", pd.Series(False)).astype(bool),
            "candidate_id",
        ].astype(str).tolist()
    return _json_ready(metadata)  # type: ignore[return-value]


def run_reference_candidate_evaluation(
    *,
    processed_directory: str | Path = reference_trajectory_data_dir,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    cycle_index: int | None = None,
    approved_hip_range_deg: tuple[float, float] | None = FORMAL_HIP_ROM_DEG,
    approved_knee_range_deg: tuple[float, float] | None = FORMAL_KNEE_ROM_DEG,
    rom_approval_source: str | None = None,
    apply_smooth_rom_mapping: bool = False,
    samples_per_segment: int = 201,
    save_outputs: bool = True,
    generate_plots: bool = True,
) -> Stage5CResult:
    """Run legacy Stage 5C under the one formal ROM protocol.

    ``None`` remains available only for explicit legacy audit tests.  Every
    non-null runner approval must exactly match the manifest; command-line
    values cannot create a second active ROM or authorize amplitude mapping.
    """

    if approved_hip_range_deg is not None and tuple(
        map(float, approved_hip_range_deg)
    ) != FORMAL_HIP_ROM_DEG:
        raise ValueError("Stage 5C hip ROM must exactly match ROM_PROTOCOL_V2")
    if approved_knee_range_deg is not None and tuple(
        map(float, approved_knee_range_deg)
    ) != FORMAL_KNEE_ROM_DEG:
        raise ValueError("Stage 5C knee ROM must exactly match ROM_PROTOCOL_V2")
    if apply_smooth_rom_mapping:
        raise ValueError("formal Stage 5C forbids ROM amplitude mapping")

    source_cycle, source_metadata = load_processed_reference_cycle(
        processed_directory, cycle_index=cycle_index
    )
    versions = build_execution_reference_versions(
        source_cycle, samples_per_segment=samples_per_segment
    )
    hip_approval = (
        None
        if approved_hip_range_deg is None
        else HipRomApproval(*map(float, approved_hip_range_deg))
    )
    knee_approval = (
        None
        if approved_knee_range_deg is None
        else KneeRomApproval(*map(float, approved_knee_range_deg))
    )
    versions, rom_audit = apply_execution_rom_policy(
        versions,
        approved_hip_rom=hip_approval,
        approved_knee_rom=knee_approval,
        apply_smooth_rom_mapping=apply_smooth_rom_mapping,
    )

    empty = _empty_tables()
    local_result: LocalIdentificationResult | None = None
    candidate_result: CandidateEvaluationResult | None = None
    candidate_trajectories: dict[str, pd.DataFrame] = {}
    if rom_audit.formal_execution_allowed:
        local_phase_paths = build_local_phase_paths(versions)
        local_trajectories = retime_local_phase_paths(
            local_phase_paths, samples_per_segment=samples_per_segment
        )
        local_dataset = build_local_identification_dataset(local_trajectories)
        parameter_table, optimizer_results, parameters = fit_local_subject_parameters(
            local_dataset
        )
        bounds = fit_local_identification_domain(local_dataset)
        domain_coverage = build_local_domain_coverage(local_trajectories, bounds)
        local_result = LocalIdentificationResult(
            dataset=local_dataset,
            parameter_estimates=parameter_table,
            optimizer_results=optimizer_results,
            estimated_parameters_by_subject=parameters,
            domain_bounds=bounds,
            domain_coverage=domain_coverage,
        )
        candidate_phase_paths = build_candidate_phase_paths(versions)
        candidate_trajectories = retime_candidate_phase_paths(
            candidate_phase_paths, samples_per_segment=samples_per_segment
        )
        if knee_approval is None:
            raise RuntimeError("formal evaluation cannot proceed without approval.")
        candidate_result = evaluate_candidate_trajectories(
            candidate_trajectories,
            parameters,
            bounds,
            approved_hip_rom=hip_approval,
            approved_knee_rom=knee_approval,
        )

    local_dataset = (
        local_result.dataset if local_result is not None else empty["local_dataset"]
    )
    parameter_estimates = (
        local_result.parameter_estimates
        if local_result is not None
        else empty["parameter_estimates"]
    )
    domain_coverage = (
        local_result.domain_coverage
        if local_result is not None
        else empty["domain_coverage"]
    )
    candidate_metrics = (
        candidate_result.metrics
        if candidate_result is not None
        else empty["candidate_metrics"]
    )
    feasibility = (
        candidate_result.feasibility
        if candidate_result is not None
        else empty["candidate_feasibility"]
    )
    comparison = (
        candidate_result.subject_comparison
        if candidate_result is not None
        else empty["candidate_subject_comparison"]
    )
    pareto = (
        candidate_result.pareto
        if candidate_result is not None
        else empty["candidate_pareto"]
    )
    metadata = _build_metadata(
        source_metadata=source_metadata,
        rom_audit=rom_audit,
        execution_versions=versions,
        local_result=local_result,
        candidate_result=candidate_result,
        rom_approval_source=rom_approval_source,
    )
    metadata["parent_reference_id"] = None
    metadata["parent_reference_sha256"] = None
    metadata["final_result_eligible"] = False
    metadata["final_result_ineligible_reason"] = (
        "legacy_symmetric_candidate_pipeline_not_bound_to_frozen_active_reference"
    )
    metadata["selected_source_cycle_angle_ranges_deg"] = {
        "q_hip": [
            float(np.rad2deg(source_cycle["q_hip_rad"].min())),
            float(np.rad2deg(source_cycle["q_hip_rad"].max())),
        ],
        "q_knee": [
            float(np.rad2deg(source_cycle["q_knee_rad"].min())),
            float(np.rad2deg(source_cycle["q_knee_rad"].max())),
        ],
    }

    output_dir = Path(output_directory)
    output_paths: dict[str, Path] = {}
    table_map = {
        "reference_execution_versions.csv": versions,
        "reference_local_identification_dataset.csv": local_dataset,
        "reference_local_parameter_estimates.csv": parameter_estimates,
        "reference_local_domain_coverage.csv": domain_coverage,
        "reference_candidate_metrics.csv": candidate_metrics,
        "reference_candidate_feasibility.csv": feasibility,
        "reference_candidate_subject_comparison.csv": comparison,
        "reference_candidate_pareto.csv": pareto,
    }
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, dataframe in table_map.items():
            path = output_dir / filename
            dataframe.to_csv(path, index=False)
            output_paths[filename] = path

    visualization_paths: dict[str, Path] = {}
    skipped_visualizations: dict[str, str] = {}
    if generate_plots:
        from .visualize_reference_candidates import (
            generate_reference_candidate_visualizations,
        )

        candidate_samples = (
            pd.concat(candidate_trajectories.values(), ignore_index=True)
            if candidate_trajectories
            else pd.DataFrame()
        )
        visualization = generate_reference_candidate_visualizations(
            versions,
            local_dataset,
            domain_coverage,
            candidate_metrics,
            comparison,
            pareto,
            metadata,
            output_dir,
            candidate_trajectories=candidate_samples,
        )
        visualization_paths = dict(visualization.paths)
        skipped_visualizations = dict(visualization.skipped)
        output_paths.update(visualization_paths)

    metadata["generated_files"] = sorted(output_paths)
    metadata["skipped_visualizations"] = skipped_visualizations
    if save_outputs:
        metadata_path = output_dir / "metadata.json"
        output_paths["metadata.json"] = metadata_path
        metadata["generated_files"] = sorted(output_paths)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)

    return Stage5CResult(
        execution_versions=versions,
        rom_audit=rom_audit,
        local_identification_dataset=local_dataset,
        local_parameter_estimates=parameter_estimates,
        local_domain_coverage=domain_coverage,
        candidate_metrics=candidate_metrics,
        candidate_feasibility=feasibility,
        candidate_subject_comparison=comparison,
        candidate_pareto=pareto,
        candidate_trajectories=candidate_trajectories,
        metadata=metadata,
        output_paths=output_paths,
        visualization_paths=visualization_paths,
        skipped_visualizations=skipped_visualizations,
    )


def _approved_pair(
    parser: argparse.ArgumentParser,
    minimum: float | None,
    maximum: float | None,
    joint_name: str,
) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None:
        parser.error(
            f"--approved-{joint_name}-min-deg and --approved-{joint_name}-max-deg "
            "must be supplied together"
        )
    return float(minimum), float(maximum)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5C software-only reference execution and candidate screening. "
            "Formal evaluation requires an explicit run-local knee ROM pair."
        )
    )
    parser.add_argument(
        "--processed-directory", type=Path, default=reference_trajectory_data_dir
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--cycle-index", type=int)
    parser.add_argument(
        "--rom-approval-source",
        type=str,
        help="Human-readable provenance for the run-local ROM approval.",
    )
    parser.add_argument("--samples-per-segment", type=int, default=201)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    approved_hip = FORMAL_HIP_ROM_DEG
    approved_knee = FORMAL_KNEE_ROM_DEG
    result = run_reference_candidate_evaluation(
        processed_directory=args.processed_directory,
        output_directory=args.output_directory,
        cycle_index=args.cycle_index,
        approved_hip_range_deg=approved_hip,
        approved_knee_range_deg=approved_knee,
        rom_approval_source=(
            args.rom_approval_source or "formal_experiment_manifest"
        ),
        apply_smooth_rom_mapping=False,
        samples_per_segment=args.samples_per_segment,
        generate_plots=not args.no_plots,
    )
    print("Stage 5C completed (software-only).")
    print(f"Output directory: {Path(args.output_directory).resolve()}")
    print(f"Formal configured knee ROM: {knee_range_deg} deg")
    print(f"Approved hip ROM: {result.rom_audit.approved_hip_range_deg}")
    print(f"Approved knee ROM: {result.rom_audit.approved_knee_range_deg}")
    print(f"ROM mapping applied: {result.rom_audit.rom_mapping_applied}")
    print(f"Formal execution allowed: {result.rom_audit.formal_execution_allowed}")
    if not result.rom_audit.formal_execution_allowed:
        print("Formal dynamics/identification/candidate screening blocked: " + ";".join(result.rom_audit.block_reasons))
    else:
        print(result.local_domain_coverage.to_string(index=False))
        print(result.candidate_feasibility.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_SPECIFICATIONS",
    "CandidateEvaluationResult",
    "Stage5CResult",
    "build_candidate_pareto",
    "build_candidate_phase_paths",
    "evaluate_candidate_trajectories",
    "retime_candidate_phase_paths",
    "run_reference_candidate_evaluation",
]
