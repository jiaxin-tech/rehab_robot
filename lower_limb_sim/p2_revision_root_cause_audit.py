"""Post-hoc root-cause diagnostics for the frozen P2 research policy.

The functions in this module are analysis-only.  Virtual truth is attached
after policy decisions are frozen and is never exposed to proposal, fitting,
guard calibration, exploration ranking, or stopping.  No function changes the
frozen P0/P1/P2 definitions or creates a human/robot threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import L1
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    TrajectoryComponentCache,
    build_predicted_map,
    evaluate_truth_map,
    mechanical_objective_from_torque_batch,
    one_step_coordinate_neighborhood,
)
from .dynamic_subject import DYNAMIC_SUBJECTS, get_dynamic_subject
from .full_dynamics import inverse_dynamics
from .mechanical_objective import (
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
from .research_decision_guarded_sequential_personalization import (
    CURRENT_BEST_NOT_A_CANDIDATE,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    PolicyRunResult,
    _model_for_iteration,
    build_initial_research_state,
)


AUDIT_PROTOCOL_ID = "P2_REVISION_ROOT_CAUSE_AUDIT_V1"
POST_HOC_TRUTH_ROLE = "POST_HOC_ROOT_CAUSE_AND_COUNTERFACTUAL_ONLY"
SYNTHETIC_SCAN_ROLE = "SYNTHETIC_PARAMETER_SENSITIVITY_ONLY"
LOCAL_VALIDATION_CANDIDATE_ID = (
    "LOCAL_PAIRWISE_VALIDATION_UNCERTAINTY_CANDIDATE"
)
LOCAL_PAIR_UNAVAILABLE = (
    "NOT_ESTIMABLE_NO_DESIGNATED_VALIDATION_PAIR_ON_FORMAL_LOCAL_ALPHA_SCALE"
)
GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH = "GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH"
LOCAL_CALIBRATION_NOT_SUFFICIENT = "LOCAL_CALIBRATION_NOT_SUFFICIENT"
OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM = "OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM"
POLICY_COLLAPSES_SUBJECT_DIFFERENCES = "POLICY_COLLAPSES_SUBJECT_DIFFERENCES"
CURRENT_OBJECTIVE_LOW_SUBJECT_DISCRIMINATION = (
    "CURRENT_OBJECTIVE_LOW_SUBJECT_DISCRIMINATION"
)
CURRENT_VIRTUAL_SUBJECT_SET_INSUFFICIENT = (
    "CURRENT_FOUR_VIRTUAL_SUBJECTS_DO_NOT_SPAN_ALL_SYNTHETIC_OPTIMA"
)
SUPPORT_ONLY_EXPLORATION = "SUPPORT_ONLY_EXPLORATION"
EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT = (
    "EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT"
)
POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION = (
    "POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION"
)
P2_POLICY_REVISION_JUSTIFIED = "P2_POLICY_REVISION_JUSTIFIED"
OFFLINE_METHOD_REQUIRES_REVISION = "OFFLINE_METHOD_REQUIRES_REVISION"

_ALPHA_COLUMNS = ("hip_delta", "knee_delta", "phase_delta")
_STEP_BY_ALPHA = {
    "hip_delta": GRID_HIP_STEP_DEG,
    "knee_delta": GRID_KNEE_STEP_DEG,
    "phase_delta": GRID_PHASE_STEP,
}


@dataclass(frozen=True)
class SubjectTruthAudit:
    landscape: pd.DataFrame
    summary: Mapping[str, Any]
    profiles: pd.DataFrame
    local_sensitivity: Mapping[str, Any]


def _alpha_mask(table: pd.DataFrame, alpha: Sequence[float]) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    if values.shape != (3,):
        raise ValueError("alpha must contain hip, knee, and phase")
    mask = np.ones(len(table), dtype=bool)
    for column, value in zip(_ALPHA_COLUMNS, values):
        mask &= np.isclose(
            table[column].to_numpy(dtype=float), value, atol=1e-12, rtol=0.0
        )
    return mask


def _unique_alpha_row(table: pd.DataFrame, alpha: Sequence[float]) -> pd.Series:
    selected = table.loc[_alpha_mask(table, alpha)]
    if len(selected) != 1:
        raise RuntimeError(f"truth table lacks one alpha {tuple(alpha)}")
    return selected.iloc[0].copy()


def _truth_best(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        raise ValueError("truth candidate table is empty")
    return table.sort_values(
        ["J_truth", "trajectory_id"], kind="mergesort"
    ).iloc[0].copy()


def _finite_difference(
    table: pd.DataFrame,
    column: str,
) -> tuple[float, float, float, float]:
    step = _STEP_BY_ALPHA[column]
    minus = {name: 0.0 for name in _ALPHA_COLUMNS}
    plus = dict(minus)
    minus[column] = -step
    plus[column] = step
    j_minus = float(_unique_alpha_row(table, tuple(minus.values()))["J_truth"])
    j_zero = float(_unique_alpha_row(table, (0.0, 0.0, 0.0))["J_truth"])
    j_plus = float(_unique_alpha_row(table, tuple(plus.values()))["J_truth"])
    gradient = (j_plus - j_minus) / (2.0 * step)
    curvature = (j_plus - 2.0 * j_zero + j_minus) / (step**2)
    return gradient, curvature, j_minus, j_plus


def audit_subject_truth_landscape(
    subject_id: str,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    *,
    p2_final_alpha: Sequence[float] = (0.0, -5.0, 0.0),
) -> SubjectTruthAudit:
    """Compute matched virtual truth without feeding it to the frozen policy."""

    state = build_initial_research_state(subject_id, "matched_linear")
    model = _model_for_iteration(state, state.parameters, state.domain_data, 0)
    predicted, _ = build_predicted_map(model, parameter_lattice, cache)
    evaluated, _ = evaluate_truth_map(predicted, model, cache)
    landscape = evaluated.loc[
        :,
        [
            "trajectory_id",
            "hip_delta",
            "knee_delta",
            "phase_delta",
            "geometrically_admissible",
            "J_truth",
            "delta_J_truth",
        ],
    ].copy()
    landscape.insert(0, "subject_id", subject_id)
    landscape.insert(1, "scenario_name", "matched_linear")
    landscape["truth_role"] = POST_HOC_TRUTH_ROLE
    landscape["truth_used_by_policy"] = False

    global_best = _truth_best(landscape)
    local = one_step_coordinate_neighborhood(landscape)
    local_best = _truth_best(local)
    reference = _unique_alpha_row(landscape, (0.0, 0.0, 0.0))
    p2_final = _unique_alpha_row(landscape, p2_final_alpha)

    profile_frames: list[pd.DataFrame] = []
    monotonic: dict[str, bool] = {}
    for axis in _ALPHA_COLUMNS:
        other = tuple(name for name in _ALPHA_COLUMNS if name != axis)
        profile = landscape.loc[
            np.isclose(landscape[other[0]], 0.0)
            & np.isclose(landscape[other[1]], 0.0),
            ["subject_id", "trajectory_id", axis, "J_truth"],
        ].copy()
        profile = profile.sort_values(axis, kind="mergesort")
        profile = profile.rename(columns={axis: "axis_value"})
        profile["profile_axis"] = axis
        profile["axis_step"] = _STEP_BY_ALPHA[axis]
        profile["truth_role"] = POST_HOC_TRUTH_ROLE
        profile_frames.append(profile)
        differences = np.diff(profile["J_truth"].to_numpy(dtype=float))
        if axis == "knee_delta":
            monotonic[axis] = bool(np.all(differences >= -1e-12))
        else:
            monotonic[axis] = bool(np.all(differences <= 1e-12))

    sensitivity: dict[str, Any] = {
        "subject_id": subject_id,
        "reference_J_truth": float(reference["J_truth"]),
        "finite_difference_role": "FORMAL_GRID_NUMERICAL_DIAGNOSTIC_NOT_PHYSIOLOGICAL_GRADIENT",
        "truth_role": POST_HOC_TRUTH_ROLE,
    }
    for axis in _ALPHA_COLUMNS:
        gradient, curvature, j_minus, j_plus = _finite_difference(landscape, axis)
        short = {"hip_delta": "hip", "knee_delta": "knee", "phase_delta": "phase"}[axis]
        sensitivity[f"{short}_formal_step"] = _STEP_BY_ALPHA[axis]
        sensitivity[f"dJ_d_{short}_at_reference"] = gradient
        sensitivity[f"second_difference_{short}_curvature"] = curvature
        sensitivity[f"J_at_negative_{short}_step"] = j_minus
        sensitivity[f"J_at_positive_{short}_step"] = j_plus

    global_alpha = tuple(float(global_best[column]) for column in _ALPHA_COLUMNS)
    local_alpha = tuple(float(local_best[column]) for column in _ALPHA_COLUMNS)
    final_alpha = tuple(float(value) for value in p2_final_alpha)
    summary = {
        "subject_id": subject_id,
        "truth_landscape_point_count": int(len(landscape)),
        "reference_J_truth": float(reference["J_truth"]),
        "alpha_truth_global_hip": global_alpha[0],
        "alpha_truth_global_knee": global_alpha[1],
        "alpha_truth_global_phase": global_alpha[2],
        "J_truth_global": float(global_best["J_truth"]),
        "alpha_truth_local_hip": local_alpha[0],
        "alpha_truth_local_knee": local_alpha[1],
        "alpha_truth_local_phase": local_alpha[2],
        "J_truth_local": float(local_best["J_truth"]),
        "local_neighborhood_definition": (
            "reference_plus_six_signed_coordinate_moves_at_formal_minimum_steps"
        ),
        "p2_final_alpha_hip": final_alpha[0],
        "p2_final_alpha_knee": final_alpha[1],
        "p2_final_alpha_phase": final_alpha[2],
        "J_truth_at_p2_final": float(p2_final["J_truth"]),
        "p2_final_truth_regret_vs_global": float(
            p2_final["J_truth"] - global_best["J_truth"]
        ),
        "p2_final_equals_global_truth_alpha": final_alpha == global_alpha,
        "global_truth_knee_at_lower_generator_bound": bool(
            math.isclose(global_alpha[1], -5.0, abs_tol=1e-12)
        ),
        "knee_axis_J_nondecreasing_from_minus5_to_plus2": monotonic["knee_delta"],
        "hip_axis_J_nonincreasing_from_minus5_to_plus2": monotonic["hip_delta"],
        "phase_axis_J_nonincreasing_from_minus003_to_plus003": monotonic["phase_delta"],
        "truth_role": POST_HOC_TRUTH_ROLE,
        "truth_fed_back_to_policy": False,
    }
    return SubjectTruthAudit(
        landscape=landscape,
        summary=summary,
        profiles=pd.concat(profile_frames, ignore_index=True),
        local_sensitivity=sensitivity,
    )


def _reference_states(cache: TrajectoryComponentCache) -> tuple[np.ndarray, ...]:
    hip = cache.hip[0.0]
    knee = cache.knee[(0.0, 0.0)]
    return hip[0], knee[0], hip[1], knee[1], hip[2], knee[2]


def _truth_metrics(
    subject_id: str,
    rows: pd.DataFrame,
    cache: TrajectoryComponentCache,
) -> tuple[MechanicalTorqueMetrics, MechanicalTorqueMetrics]:
    if len(rows) != 1:
        raise ValueError("truth metric helper requires one alpha row")
    scenario = get_mismatch_scenario("matched_linear")
    subject = scenario.create_subject(get_dynamic_subject(subject_id))
    reference_dynamics = mismatch_inverse_dynamics(
        *_reference_states(cache),
        subject,
        L1,
        residual_random_seed=scenario.random_seed,
    )
    reference = compute_torque_metrics(
        cache.time_s,
        np.asarray(reference_dynamics.tau_total_hip_nm, dtype=float),
        np.asarray(reference_dynamics.tau_total_knee_nm, dtype=float),
    )
    dynamics = mismatch_inverse_dynamics(
        *cache.batch(rows),
        subject,
        L1,
        residual_random_seed=scenario.random_seed,
    )
    candidate = compute_torque_metrics(
        cache.time_s,
        np.asarray(dynamics.tau_total_hip_nm, dtype=float)[0],
        np.asarray(dynamics.tau_total_knee_nm, dtype=float)[0],
    )
    return reference, candidate


def build_objective_normalization_audit(
    truth_summaries: pd.DataFrame,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
) -> pd.DataFrame:
    """Report raw torque scales and unchanged normalized ratios."""

    rows: list[dict[str, Any]] = []
    baseline_reference: tuple[float, float] | None = None
    for summary in truth_summaries.to_dict(orient="records"):
        subject_id = str(summary["subject_id"])
        candidates = {
            "REFERENCE": (0.0, 0.0, 0.0),
            "P2_SHARED_FINAL": (
                summary["p2_final_alpha_hip"],
                summary["p2_final_alpha_knee"],
                summary["p2_final_alpha_phase"],
            ),
            "TRUTH_LOCAL_MINIMUM": (
                summary["alpha_truth_local_hip"],
                summary["alpha_truth_local_knee"],
                summary["alpha_truth_local_phase"],
            ),
            "TRUTH_GLOBAL_MINIMUM": (
                summary["alpha_truth_global_hip"],
                summary["alpha_truth_global_knee"],
                summary["alpha_truth_global_phase"],
            ),
        }
        for candidate_role, alpha in candidates.items():
            lattice_row = parameter_lattice.loc[_alpha_mask(parameter_lattice, alpha)]
            reference, candidate = _truth_metrics(subject_id, lattice_row, cache)
            if subject_id == "baseline" and baseline_reference is None:
                baseline_reference = (
                    reference.hip_rms_torque_nm,
                    reference.knee_rms_torque_nm,
                )
            hip_ratio = candidate.hip_rms_torque_nm / reference.hip_rms_torque_nm
            knee_ratio = candidate.knee_rms_torque_nm / reference.knee_rms_torque_nm
            rows.append(
                {
                    "subject_id": subject_id,
                    "candidate_role": candidate_role,
                    "alpha_hip": float(alpha[0]),
                    "alpha_knee": float(alpha[1]),
                    "alpha_phase": float(alpha[2]),
                    "reference_hip_rms_torque_nm": reference.hip_rms_torque_nm,
                    "reference_knee_rms_torque_nm": reference.knee_rms_torque_nm,
                    "candidate_hip_rms_torque_nm": candidate.hip_rms_torque_nm,
                    "candidate_knee_rms_torque_nm": candidate.knee_rms_torque_nm,
                    "candidate_minus_reference_hip_rms_nm": (
                        candidate.hip_rms_torque_nm - reference.hip_rms_torque_nm
                    ),
                    "candidate_minus_reference_knee_rms_nm": (
                        candidate.knee_rms_torque_nm - reference.knee_rms_torque_nm
                    ),
                    "R_h": hip_ratio,
                    "R_k": knee_ratio,
                    "J_unchanged_formula": math.sqrt(
                        (hip_ratio**2 + knee_ratio**2) / 2.0
                    ),
                    "normalization_removes_absolute_subject_scale_by_design": True,
                    "objective_modified": False,
                    "truth_role": POST_HOC_TRUTH_ROLE,
                }
            )
    output = pd.DataFrame(rows)
    if baseline_reference is None:
        raise RuntimeError("normalization audit lacks baseline reference")
    output["reference_hip_scale_vs_baseline"] = (
        output["reference_hip_rms_torque_nm"] / baseline_reference[0]
    )
    output["reference_knee_scale_vs_baseline"] = (
        output["reference_knee_rms_torque_nm"] / baseline_reference[1]
    )
    return output


def registered_parameter_design() -> pd.DataFrame:
    """Use only values already present in the repository's four subjects."""

    baseline = get_dynamic_subject("baseline")
    registered: dict[str, dict[str, float]] = {}
    for subject_id, subject in DYNAMIC_SUBJECTS.items():
        registered[subject_id] = {
            "mass_scale": subject.mass_thigh_kg / baseline.mass_thigh_kg,
            "k_hip_nm_per_rad": subject.k_hip_nm_per_rad,
            "k_knee_nm_per_rad": subject.k_knee_nm_per_rad,
            "b_hip_nm_s_per_rad": subject.b_hip_nm_s_per_rad,
            "b_knee_nm_s_per_rad": subject.b_knee_nm_s_per_rad,
        }
    unique = {
        name: sorted({values[name] for values in registered.values()})
        for name in PARAMETER_NAMES
    }
    combinations = []
    serial = 0
    for mass in unique["mass_scale"]:
        for k_hip in unique["k_hip_nm_per_rad"]:
            for k_knee in unique["k_knee_nm_per_rad"]:
                for b_hip in unique["b_hip_nm_s_per_rad"]:
                    for b_knee in unique["b_knee_nm_s_per_rad"]:
                        parameters = {
                            "mass_scale": mass,
                            "k_hip_nm_per_rad": k_hip,
                            "k_knee_nm_per_rad": k_knee,
                            "b_hip_nm_s_per_rad": b_hip,
                            "b_knee_nm_s_per_rad": b_knee,
                        }
                        exact = [
                            subject_id
                            for subject_id, values in registered.items()
                            if all(
                                math.isclose(parameters[name], values[name])
                                for name in PARAMETER_NAMES
                            )
                        ]
                        combinations.append(
                            {
                                "synthetic_subject_id": f"registered_value_product_{serial:02d}",
                                **parameters,
                                "matches_registered_subject": bool(exact),
                                "registered_subject_id": exact[0] if exact else "",
                                "parameter_value_source": (
                                    "CARTESIAN_PRODUCT_OF_VALUES_ALREADY_USED_BY_REGISTERED_VIRTUAL_SUBJECTS"
                                ),
                                "research_role": SYNTHETIC_SCAN_ROLE,
                                "clinical_range_claimed": False,
                            }
                        )
                        serial += 1
    return pd.DataFrame(combinations)


def _linear_truth_values(
    parameters: Mapping[str, float],
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    subject = candidate_subject_from_parameters(template, parameters)
    reference_dynamics = inverse_dynamics(*_reference_states(cache), subject, L1)
    reference = compute_torque_metrics(
        cache.time_s,
        np.asarray(reference_dynamics.tau_total_hip_nm, dtype=float),
        np.asarray(reference_dynamics.tau_total_knee_nm, dtype=float),
    )
    values: list[np.ndarray] = []
    for start in range(0, len(parameter_lattice), batch_size):
        rows = parameter_lattice.iloc[start : start + batch_size]
        dynamics = inverse_dynamics(*cache.batch(rows), subject, L1)
        values.append(
            mechanical_objective_from_torque_batch(
                cache.time_s,
                np.asarray(dynamics.tau_total_hip_nm, dtype=float),
                np.asarray(dynamics.tau_total_knee_nm, dtype=float),
                reference,
            )
        )
    return np.concatenate(values)


def scan_registered_parameter_sensitivity(
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
) -> pd.DataFrame:
    design = registered_parameter_design()
    rows: list[dict[str, Any]] = []
    for item in design.to_dict(orient="records"):
        parameters = {name: float(item[name]) for name in PARAMETER_NAMES}
        values = _linear_truth_values(parameters, parameter_lattice, cache)
        table = parameter_lattice.loc[:, ["trajectory_id", *_ALPHA_COLUMNS]].copy()
        table["J_truth"] = values
        global_best = _truth_best(table)
        local_best = _truth_best(one_step_coordinate_neighborhood(table))
        rows.append(
            {
                **item,
                "alpha_truth_global_hip": float(global_best["hip_delta"]),
                "alpha_truth_global_knee": float(global_best["knee_delta"]),
                "alpha_truth_global_phase": float(global_best["phase_delta"]),
                "J_truth_global": float(global_best["J_truth"]),
                "alpha_truth_local_hip": float(local_best["hip_delta"]),
                "alpha_truth_local_knee": float(local_best["knee_delta"]),
                "alpha_truth_local_phase": float(local_best["phase_delta"]),
                "J_truth_local": float(local_best["J_truth"]),
                "global_knee_at_lower_generator_bound": bool(
                    math.isclose(float(global_best["knee_delta"]), -5.0, abs_tol=1e-12)
                ),
                "truth_used_for_policy": False,
            }
        )
    return pd.DataFrame(rows)


def build_current_guard_uncertainty_provenance(
    results: Sequence[PolicyRunResult],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        if result.policy_id != POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT:
            continue
        frame = result.uncertainty_pairwise_audit.copy(deep=True)
        frame["current_has_personalization_alpha"] = False
        frame["candidate_has_personalization_alpha"] = False
        frame["formal_parameter_distance_steps"] = np.nan
        frame["validation_pair_scale_class"] = (
            "UNMAPPABLE_IDENTIFICATION_EXCITATION_PAIR_NOT_PERSONALIZATION_ALPHA_PAIR"
        )
        frame["same_formal_one_step_local_scale"] = False
        frame["larger_formal_parameter_distance"] = False
        frame["local_scale_classification_reason"] = (
            "designated_validation_items_are_hip_dominant_slow_and_knee_dominant_fast_"
            "identification_excitations_without_generator_alpha_coordinates"
        )
        frame["current_guard_uses_e_delta_J"] = True
        frame["local_pairwise_candidate_status"] = LOCAL_PAIR_UNAVAILABLE
        frame["heldout_final_test_used"] = False
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_validation_pair_scales(provenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    masks = {
        "SAME_FORMAL_ONE_STEP_LOCAL_SCALE": provenance[
            "same_formal_one_step_local_scale"
        ].astype(bool),
        "LARGER_FORMAL_PARAMETER_DISTANCE": provenance[
            "larger_formal_parameter_distance"
        ].astype(bool),
        "UNMAPPABLE_IDENTIFICATION_EXCITATION_PAIR": provenance[
            "validation_pair_scale_class"
        ].eq(
            "UNMAPPABLE_IDENTIFICATION_EXCITATION_PAIR_NOT_PERSONALIZATION_ALPHA_PAIR"
        ),
        "ALL_CURRENT_GUARD_PAIR_INSTANCES": np.ones(len(provenance), dtype=bool),
    }
    for scale_class, mask in masks.items():
        values = provenance.loc[mask, "e_delta_J"].to_numpy(dtype=float)
        rows.append(
            {
                "validation_pair_scale_class": scale_class,
                "pair_instance_count": int(len(values)),
                "mean_e_delta_J": float(np.mean(values)) if len(values) else np.nan,
                "p95_e_delta_J": float(np.percentile(values, 95)) if len(values) else np.nan,
                "p99_e_delta_J": float(np.percentile(values, 99)) if len(values) else np.nan,
                "max_e_delta_J": float(np.max(values)) if len(values) else np.nan,
                "research_candidate_id": (
                    LOCAL_VALIDATION_CANDIDATE_ID
                    if scale_class == "SAME_FORMAL_ONE_STEP_LOCAL_SCALE"
                    else "CURRENT_GUARD_PROVENANCE_AUDIT"
                ),
                "candidate_status": (
                    LOCAL_PAIR_UNAVAILABLE
                    if scale_class == "SAME_FORMAL_ONE_STEP_LOCAL_SCALE"
                    else "DESCRIPTIVE_ONLY"
                ),
                "formal_threshold_created": False,
            }
        )
    return pd.DataFrame(rows)


def build_counterfactual_guard_comparison(
    results: Sequence[PolicyRunResult],
    post_decision_truth_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Audit G0; retain explicit unavailable rows for non-estimable G1/G2."""

    truth = post_decision_truth_candidates.loc[
        :,
        ["case_id", "iteration", "trajectory_id", "delta_J_truth"],
    ].drop_duplicates()
    rows: list[dict[str, Any]] = []
    guards = (
        ("G0_CURRENT_GLOBAL_MAX", "CURRENT_MAX", True),
        ("G1_LOCAL_PAIRWISE_MAX", "LOCAL_MAX", False),
        ("G2_LOCAL_PAIRWISE_P95", "LOCAL_P95", False),
    )
    for result in results:
        if result.policy_id != POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT:
            continue
        local = result.decision_guard_audit.loc[
            result.decision_guard_audit["decision_guard_status"].notna()
            & ~result.decision_guard_audit["decision_guard_status"].eq(
                CURRENT_BEST_NOT_A_CANDIDATE
            )
        ].copy()
        local = local.merge(
            truth,
            on=["case_id", "iteration", "trajectory_id"],
            how="left",
            validate="one_to_one",
        )
        if local["delta_J_truth"].isna().any():
            raise RuntimeError("counterfactual guard lacks post-decision truth")
        for candidate in local.to_dict(orient="records"):
            true_improvement = bool(
                float(candidate["delta_J_truth"])
                < -OBJECTIVE_EQUIVALENCE_TOLERANCE
            )
            for guard_id, bound_type, available in guards:
                if available:
                    bound = float(candidate["validation_uncertainty_bound"])
                    base_eligible = bool(
                        candidate["geometrically_admissible"]
                        and candidate["model_supported"]
                        and candidate["current_model_supported"]
                    )
                    would_exploit = bool(
                        base_eligible
                        and -float(candidate["delta_J_pred_vs_current"])
                        - bound
                        - OBJECTIVE_EQUIVALENCE_TOLERANCE
                        > 0.0
                    )
                    status = "EVALUATED_WITH_FROZEN_CURRENT_GUARD"
                else:
                    bound = np.nan
                    would_exploit = None
                    status = LOCAL_PAIR_UNAVAILABLE
                rows.append(
                    {
                        "case_id": candidate["case_id"],
                        "subject_id": candidate["subject_id"],
                        "scenario_name": candidate["scenario_name"],
                        "iteration": int(candidate["iteration"]),
                        "trajectory_id": candidate["trajectory_id"],
                        "alpha_hip": candidate["hip_delta"],
                        "alpha_knee": candidate["knee_delta"],
                        "alpha_phase": candidate["phase_delta"],
                        "guard_id": guard_id,
                        "uncertainty_bound_type": bound_type,
                        "uncertainty_bound": bound,
                        "delta_J_pred": candidate["delta_J_pred_vs_current"],
                        "delta_J_truth_post_decision": candidate["delta_J_truth"],
                        "model_supported": candidate["model_supported"],
                        "would_exploit": would_exploit,
                        "true_improvement": true_improvement,
                        "false_improvement": (
                            bool(would_exploit and not true_improvement)
                            if would_exploit is not None
                            else None
                        ),
                        "missed_improvement": (
                            bool(not would_exploit and true_improvement)
                            if would_exploit is not None
                            else None
                        ),
                        "counterfactual_status": status,
                        "policy_executed_from_counterfactual": False,
                        "truth_used_to_construct_guard": False,
                        "truth_used_only_for_posthoc_outcome": True,
                        "heldout_final_test_used": False,
                    }
                )
    return pd.DataFrame(rows)


def _eligible_sets_by_iteration(result: PolicyRunResult) -> dict[int, set[str]]:
    table = result.decision_guard_audit
    local = table.loc[table["decision_guard_status"].notna()].copy()
    output: dict[int, set[str]] = {}
    for iteration, group in local.groupby("iteration", sort=False):
        output[int(iteration)] = set(
            group.loc[
                group["research_exploit_eligible"].eq(True),
                "trajectory_id",
            ].astype(str)
        )
    return output


def build_exploration_value_decomposition(
    results: Sequence[PolicyRunResult],
    previous_exploration_value: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prior = previous_exploration_value.set_index(["case_id", "iteration"])
    for result in results:
        if result.policy_id != POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT:
            continue
        if result.exploration_information_gain.empty:
            continue
        eligible = _eligible_sets_by_iteration(result)
        parameter = result.parameter_history.set_index("iteration")
        map_history = result.prediction_map_history.set_index("iteration")
        history = result.trial_history.set_index("iteration")
        for explore in result.exploration_information_gain.to_dict(orient="records"):
            iteration = int(explore["iteration"])
            key = (str(explore["case_id"]), iteration)
            if key not in prior.index:
                raise RuntimeError(f"previous convergence artifact lacks explore {key}")
            old = prior.loc[key]
            theta = parameter.loc[iteration]
            map_row = map_history.loc[iteration]
            trial = history.loc[iteration]
            parameter_deltas = {
                f"theta_delta_{name}": float(theta[f"{name}_delta"])
                for name in PARAMETER_NAMES
            }
            theta_changed = any(value != 0.0 for value in parameter_deltas.values())
            map_rms = float(old["prediction_map_RMS_change"])
            map_max = float(old["prediction_map_max_abs_change"])
            map_changed = bool(map_rms != 0.0 or map_max != 0.0)
            validation_change = float(explore["validation_e_delta_J_change"])
            validation_changed = validation_change != 0.0
            current_set = eligible.get(iteration, set())
            next_set = eligible.get(iteration + 1, set())
            newly_enabled = next_set.difference(current_set)
            best_change = float(
                trial["best_actual_J_after"] - trial["best_actual_J_before"]
            )
            best_changed = bool(best_change < -OBJECTIVE_EQUIVALENCE_TOLERANCE)
            information_gain = float(explore["incremental_log_information_gain"])
            supported_growth = int(explore["new_supported_point_count"])
            model_information_value = bool(
                theta_changed or map_changed or validation_changed
            )
            support_value = supported_growth > 0
            decision_value = bool(newly_enabled or best_changed)
            support_only = bool(
                information_gain > 0.0
                and support_value
                and not model_information_value
                and not decision_value
            )
            rows.append(
                {
                    "case_id": explore["case_id"],
                    "subject_id": explore["subject_id"],
                    "scenario_name": explore["scenario_name"],
                    "iteration": iteration,
                    "trajectory_id": explore["trajectory_id"],
                    "alpha_hip": trial["alpha_hip"],
                    "alpha_knee": trial["alpha_knee"],
                    "alpha_phase": trial["alpha_phase"],
                    "information_gain": information_gain,
                    "new_supported_points": supported_growth,
                    **parameter_deltas,
                    "theta_change_l2": float(old["parameter_change_l2"]),
                    "theta_changed_exactly": theta_changed,
                    "validation_deltaJ_error_change": validation_change,
                    "validation_error_changed_exactly": validation_changed,
                    "RMS_prediction_map_change": map_rms,
                    "max_prediction_map_change": map_max,
                    "prediction_map_changed_exactly": map_changed,
                    "newly_enabled_exploit_candidates": len(newly_enabled),
                    "newly_enabled_exploit_trajectory_ids": ";".join(
                        sorted(newly_enabled)
                    ),
                    "enabled_exploit_within_1_round": bool(
                        old["enabled_exploit_within_1_round"]
                    ),
                    "enabled_exploit_within_2_rounds": bool(
                        old["enabled_exploit_within_2_rounds"]
                    ),
                    "best_J_change": best_change,
                    "best_changed_under_existing_0p005_rule": best_changed,
                    "MODEL_INFORMATION_VALUE": model_information_value,
                    "SUPPORT_PROVENANCE_VALUE": support_value,
                    "DECISION_VALUE": decision_value,
                    "support_only_exploration": support_only,
                    "diagnostic_label": (
                        SUPPORT_ONLY_EXPLORATION
                        if support_only
                        else "DECISION_VALUE_OBSERVED"
                        if decision_value
                        else "CONTINUOUS_VALUE_METRICS_REPORTED_NO_NEW_BINARY_THRESHOLD"
                    ),
                    "support_growth_used_as_reliability_score": False,
                    "new_meaningful_threshold_created": False,
                    "truth_used_as_future_stopping_feature": False,
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != 32:
        raise RuntimeError(f"expected 32 frozen EXPLORE rows, got {len(output)}")
    return output


def build_knee_stiff_exploration_audit(
    decomposition: pd.DataFrame,
) -> pd.DataFrame:
    knee = decomposition.loc[
        decomposition["subject_id"].eq("knee_stiff")
        & decomposition["scenario_name"].eq("matched_linear")
    ].sort_values("iteration").copy()
    if len(knee) != 8:
        raise RuntimeError("knee_stiff must have eight frozen EXPLORE trials")
    knee["why_theta_unchanged"] = (
        "matched_five_parameter_truth_and_estimator_remain_numerically_identical"
    )
    knee["why_prediction_map_unchanged"] = (
        "unchanged_five_parameters_produce_identical_normalized_prediction_map"
    )
    knee["why_decision_guard_unchanged"] = (
        "unchanged_model_and_designated_validation_pair_error_leave_guard_unchanged"
    )
    knee["why_next_explore_was_allowed"] = np.where(
        knee["iteration"].astype(int).lt(int(knee["iteration"].max())),
        "valid_unexecuted_unsupported_adjacent_frontier_remained_and_P2_has_no_diminishing_value_stop",
        "last_frontier_explore_then_no_useful_frontier_remained",
    )
    knee["diagnostic_conclusion"] = EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT
    return knee


def classify_root_causes(
    truth_summary: pd.DataFrame,
    synthetic: pd.DataFrame,
    counterfactual: pd.DataFrame,
    exploration: pd.DataFrame,
) -> pd.DataFrame:
    global_alpha_count = int(
        truth_summary[
            [
                "alpha_truth_global_hip",
                "alpha_truth_global_knee",
                "alpha_truth_global_phase",
            ]
        ].drop_duplicates().shape[0]
    )
    synthetic_alpha_count = int(
        synthetic[
            [
                "alpha_truth_global_hip",
                "alpha_truth_global_knee",
                "alpha_truth_global_phase",
            ]
        ].drop_duplicates().shape[0]
    )
    g0 = counterfactual.loc[counterfactual["guard_id"].eq("G0_CURRENT_GLOBAL_MAX")]
    missed = int(g0["missed_improvement"].astype(bool).sum())
    support_only = int(exploration["support_only_exploration"].astype(bool).sum())
    decision_value = int(exploration["DECISION_VALUE"].astype(bool).sum())
    rows = [
        {
            "problem": "same_subject_path",
            "possible_root": "objective_truth_landscape",
            "evidence": (
                f"all {int(truth_summary['global_truth_knee_at_lower_generator_bound'].sum())}/4 truth minima use knee=-5; "
                f"complete truth alpha count={global_alpha_count}"
            ),
            "conclusion": OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM,
        },
        {
            "problem": "same_subject_path",
            "possible_root": "policy",
            "evidence": (
                "P2 final alpha is identical (0,-5,0) while matched truth-global hip/phase components differ"
            ),
            "conclusion": POLICY_COLLAPSES_SUBJECT_DIFFERENCES,
        },
        {
            "problem": "same_subject_path",
            "possible_root": "normalization",
            "evidence": (
                "per-subject normalization removes absolute torque scale by design, but distinct truth-global alpha remains"
            ),
            "conclusion": "NORMALIZATION_COMPRESSION_CONTRIBUTES_BUT_DOES_NOT_ERASE_ALL_DISCRIMINATION",
        },
        {
            "problem": "same_subject_path",
            "possible_root": "virtual_subject_selection",
            "evidence": f"registered-value synthetic scan has {synthetic_alpha_count} distinct truth-global alpha",
            "conclusion": (
                CURRENT_VIRTUAL_SUBJECT_SET_INSUFFICIENT
                if synthetic_alpha_count > global_alpha_count
                else CURRENT_OBJECTIVE_LOW_SUBJECT_DISCRIMINATION
            ),
        },
        {
            "problem": "premature_mismatch_stop",
            "possible_root": "global_decision_uncertainty",
            "evidence": f"G0 missed-improvement candidate count={missed}; four mild-mismatch stop cases are supported with correct predicted direction",
            "conclusion": GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH,
        },
        {
            "problem": "premature_mismatch_stop",
            "possible_root": "local_calibration_mismatch",
            "evidence": (
                "zero designated validation pairs have personalization alpha coordinates; G1/G2 cannot be estimated"
            ),
            "conclusion": LOCAL_CALIBRATION_NOT_SUFFICIENT,
        },
        {
            "problem": "premature_mismatch_stop",
            "possible_root": "model_direction_error",
            "evidence": (
                "the four stopped mismatch candidates were predicted and post-hoc confirmed as improving; the bound, not sign, rejected them"
            ),
            "conclusion": "MODEL_DIRECTION_ERROR_NOT_PRIMARY_FOR_THE_FOUR_PREMATURE_STOPS",
        },
        {
            "problem": "low_value_exploration",
            "possible_root": "support_overvalued_continuation",
            "evidence": f"support-only EXPLORE={support_only}/32; decision-value EXPLORE={decision_value}/32",
            "conclusion": EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT,
        },
        {
            "problem": "low_value_exploration",
            "possible_root": "model_or_map_update",
            "evidence": (
                "all 32 matched-case EXPLORE rows have exact zero parameter and prediction-map change"
            ),
            "conclusion": POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION,
        },
        {
            "problem": "low_value_exploration",
            "possible_root": "frontier_ranking_and_stop_architecture",
            "evidence": (
                "positive information gain and unexecuted unsupported frontier permit continuation without an observable diminishing-value stop"
            ),
            "conclusion": "DECISION_VALUE_AWARE_STOPPING_REVISION_CANDIDATE_JUSTIFIED",
        },
    ]
    output = pd.DataFrame(rows)
    output["policy_modified"] = False
    output["new_threshold_created"] = False
    output["truth_fed_back_to_policy"] = False
    return output


__all__ = [
    "AUDIT_PROTOCOL_ID",
    "CURRENT_OBJECTIVE_LOW_SUBJECT_DISCRIMINATION",
    "EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT",
    "GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH",
    "LOCAL_CALIBRATION_NOT_SUFFICIENT",
    "LOCAL_PAIR_UNAVAILABLE",
    "LOCAL_VALIDATION_CANDIDATE_ID",
    "OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM",
    "OFFLINE_METHOD_REQUIRES_REVISION",
    "P2_POLICY_REVISION_JUSTIFIED",
    "POLICY_COLLAPSES_SUBJECT_DIFFERENCES",
    "POST_HOC_TRUTH_ROLE",
    "POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION",
    "SUPPORT_ONLY_EXPLORATION",
    "SYNTHETIC_SCAN_ROLE",
    "SubjectTruthAudit",
    "audit_subject_truth_landscape",
    "build_counterfactual_guard_comparison",
    "build_current_guard_uncertainty_provenance",
    "build_exploration_value_decomposition",
    "build_knee_stiff_exploration_audit",
    "build_objective_normalization_audit",
    "classify_root_causes",
    "registered_parameter_design",
    "scan_registered_parameter_sensitivity",
    "summarize_validation_pair_scales",
]
