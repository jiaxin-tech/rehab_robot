"""Endpoint-only multi-step decision framework analysis for offline P2 shadow.

The framework changes decision horizon only.  Every exploit candidate is a
direct formal-grid endpoint; latent intermediate grid nodes are audited but are
never executed.  No policy is registered or enabled by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    TrajectoryComponentCache,
    build_predicted_map,
)
from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_decision_rule_semantics_audit import (
    EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
    EXPECTED_CALIBRATION_MANIFEST_SHA256,
    SemanticsCalibration,
    load_semantics_calibration,
    sha256_file,
)
from .research_decision_guarded_sequential_personalization import (
    CURRENT_BEST_NOT_A_CANDIDATE,
    EXECUTED_FALSE_IMPROVEMENT,
    GEOMETRICALLY_INADMISSIBLE,
    MODEL_RELIABILITY_DEGRADED,
    NO_INDEPENDENT_VALIDATION_EVIDENCE,
    RESEARCH_ONLY,
    RESEARCH_EXPLOIT_ELIGIBLE,
    STOP_MAX_PERSONALIZATION_TRIALS,
    STOP_MODEL_UPDATE_FAILURE,
    STOP_NO_GEOMETRICALLY_VALID_CANDIDATE,
    STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER,
    STOP_PATIENT_ENVELOPE_BOUNDARY,
    TRIAL_PURPOSE_EXPLOIT,
    TRIAL_PURPOSE_EXPLORE,
    UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE,
    InitialResearchState,
    PolicyRunResult,
    SelectionGatedVirtualTruthOracle,
    _actual_objective,
    _fit_updated_model,
    _model_for_iteration,
    alpha_from_row,
    build_local_exploration_frontier,
    rank_exploration_frontier,
)
from .safeguarded_sequential_initial_identification import (
    default_virtual_patient_envelope,
)
from .sequential_personalization import SearchAlpha, accept_actual_trial


ANALYSIS_ID = "P2_MULTI_STEP_DECISION_FRAMEWORK_ANALYSIS_V1"
MANIFEST_ID = "P2_MULTI_STEP_DECISION_FRAMEWORK_ANALYSIS_MANIFEST_V1"
OFFLINE_ONLY = "OFFLINE_ONLY"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"
DEFAULT_ENABLED = False
MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON = 20
PRIOR_SEMANTICS_MANIFEST_SHA256 = (
    "2e97a2b812acff15284c469756e5a0b0dedad307a7f8b8410276dd415c593b65"
)
EXPECTED_SMALL_STEP_SOURCE_SHA256 = (
    "4fd0f87335ff6a4e114d17410bf1d5c6acc2f97e142600dea52ee4e6569752d9"
)

MODULE_DIR = Path(__file__).resolve().parent
PRIOR_SEMANTICS_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_decision_rule_semantics_audit_v1"
)
PRIOR_SEMANTICS_MANIFEST_PATH = (
    PRIOR_SEMANTICS_DIRECTORY
    / "DECISION_RULE_SEMANTICS_CANDIDATE_MANIFEST_V1.json"
)
SMALL_STEP_SOURCE_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "post_prospective_rejection_root_cause_audit_v1"
    / "prospective_small_step_accumulation.csv"
)

_AXES = ("hip", "knee", "phase")
_AXIS_INDEX = {"hip": 0, "knee": 1, "phase": 2}
_GRID_STEP = {
    "hip": GRID_HIP_STEP_DEG,
    "knee": GRID_KNEE_STEP_DEG,
    "phase": GRID_PHASE_STEP,
}


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class FrameworkSpec:
    framework_id: str
    horizon_steps: int
    residual_scale: str
    role: str
    default_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FRAMEWORKS = (
    FrameworkSpec("SINGLE_STEP", 1, "INDEPENDENT_ONE_STEP_P95", "BASELINE_HORIZON"),
    FrameworkSpec("BUNDLE_2", 2, "INDEPENDENT_2_STEP_ENDPOINT_P95", "SHORT_BUNDLE"),
    FrameworkSpec("BUNDLE_3", 3, "INDEPENDENT_3_STEP_ENDPOINT_P95", "MEDIUM_BUNDLE"),
    FrameworkSpec("BUNDLE_5", 5, "INDEPENDENT_5_STEP_ENDPOINT_P95", "LONG_BUNDLE"),
)


def framework_uncertainty(
    spec: FrameworkSpec, calibration: SemanticsCalibration
) -> float:
    if spec.horizon_steps == 1:
        return float(calibration.one_step_p95)
    if spec.horizon_steps not in (2, 3, 5):
        raise PermissionError("only frozen 1/2/3/5-step horizons are allowed")
    return float(calibration.bundle_scale_p95[spec.horizon_steps])


def manifest_payload(
    calibration: SemanticsCalibration,
    *,
    checkpoint_commit: str,
    protected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze the complete analysis before any development truth access."""

    uncertainties = {
        spec.framework_id: framework_uncertainty(spec, calibration)
        for spec in FRAMEWORKS
    }
    return {
        "manifest_id": MANIFEST_ID,
        "analysis_id": ANALYSIS_ID,
        "status": "FROZEN_BEFORE_DEVELOPMENT_SHADOW_TRUTH",
        "checkpoint_commit": checkpoint_commit,
        "prior_semantics_manifest_sha256": PRIOR_SEMANTICS_MANIFEST_SHA256,
        "frameworks": [spec.as_dict() for spec in FRAMEWORKS],
        "formal_generator_grid_unit": {
            "hip_delta_deg": GRID_HIP_STEP_DEG,
            "knee_delta_deg": GRID_KNEE_STEP_DEG,
            "phase_delta": GRID_PHASE_STEP,
        },
        "decision_rule_common_to_all_frameworks": {
            "magnitude_gate": "predicted_endpoint_improvement > 0.005",
            "direction_interval": "[deltaJ_pred-U_scale_P95, deltaJ_pred+U_scale_P95]",
            "direction_gate": "deltaJ_pred + U_scale_P95 < 0",
            "additive_margin_used": False,
            "objective_tolerance_changed": False,
            "percentile_searched": False,
        },
        "framework_uncertainty_P95": uncertainties,
        "candidate_relationship": {
            "one_coordinate_only": True,
            "same_signed_direction": True,
            "endpoint_distance": "horizon_steps * existing_formal_generator_grid_unit",
            "physical_distance_invented": False,
            "generator_bounds_expanded": False,
        },
        "endpoint_authorization": {
            "authorization_scope": "DIRECT_ENDPOINT_CANDIDATE_ONLY",
            "intermediate_trajectories_executed": False,
            "latent_intermediate_nodes_must_exist": True,
            "latent_intermediate_nodes_geometry_valid": True,
            "latent_intermediate_nodes_reference_provenance_valid": True,
            "latent_intermediate_nodes_model_supported_at_90_percent": True,
            "latent_intermediate_nodes_patient_envelope_valid": True,
            "future_truth_used_for_authorization": False,
            "model_refit_after_endpoint_execution": True,
            "whole_map_recomputed_after_endpoint_execution": True,
            "authorization_expires_after_endpoint_execution": True,
        },
        "selection_rule": (
            "lowest_J_pred_then_trajectory_id_among_gate_passed_unexecuted_endpoints"
        ),
        "trial_budget": MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
        "recommendation_rule_frozen_before_shadow": {
            "prototype_eligible_if": [
                "small_step_recovery_greater_than_SINGLE_STEP",
                "total_false_improvement_not_greater_than_SINGLE_STEP",
                "mean_final_J_not_worse_than_SINGLE_STEP_by_0.005",
                "mean_global_regret_not_worse_than_SINGLE_STEP_by_0.005",
            ],
            "ranking_for_eligible_bundles": [
                "largest_small_step_recovery",
                "lowest_mean_final_J",
                "fewest_total_trials",
                "lowest_calibrated_uncertainty",
                "shortest_horizon",
            ],
            "single_step_failure_is_horizon_limited_if": [
                "at_least_one_bundle_recovers_more_small_step_paths",
                "that_bundle_does_not_increase_false_improvement",
                "that_bundle_improves_or_preserves_final_J_and_regret_within_0.005",
            ],
            "results_may_change_rule": False,
        },
        "benefit_uncertainty_analysis": {
            "benefit_metrics": [
                "small_step_recovery",
                "final_J_reduction_vs_SINGLE_STEP",
                "global_regret_reduction_vs_SINGLE_STEP",
            ],
            "uncertainty_metric": "independent_scale_P95_endpoint_residual",
            "monotonic_claim_limited_to_bundle_2_3_5": True,
            "inferential_statistics_claimed": False,
        },
        "data_roles": {
            "development": "ORIGINAL_9_PLUS_POST_REJECTION_DEVELOPMENT_6_SHADOW_ONLY",
            "independent_calibration": "RESIDUAL_SCALE_ONLY",
            "heldout_final_test": "NOT_READ",
            "prospective": "NOT_RUN",
        },
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA256,
        "bundle_pair_plan_sha256": EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
        "small_step_source_sha256": EXPECTED_SMALL_STEP_SOURCE_SHA256,
        "protected_source_sha256": dict(protected_source_sha256),
        "truth_used_to_define_or_rank_frameworks": False,
        "P2_V1_modified": False,
        "new_policy_implemented": False,
        "default_enabled": False,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
    }


class FrozenFrameworkManifestGate:
    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = str(expected_sha256)
        self.truth_access_count = 0
        self.truth_access_stages: list[str] = []

    def require_frozen(self) -> None:
        if not self.path.is_file() or sha256_file(self.path) != self.expected_sha256:
            raise PermissionError("development truth requires frozen framework manifest")

    def record_truth_access(self, stage: str) -> None:
        self.require_frozen()
        self.truth_access_count += 1
        self.truth_access_stages.append(str(stage))


def _key(alpha: SearchAlpha | Sequence[float]) -> tuple[float, float, float]:
    values = alpha.key() if isinstance(alpha, SearchAlpha) else tuple(alpha)
    return tuple(round(float(value), 12) for value in values)


def _map_lookup(table: pd.DataFrame) -> dict[tuple[float, float, float], dict[str, Any]]:
    return {
        _key((row["hip_delta"], row["knee_delta"], row["phase_delta"])): row
        for row in table.to_dict(orient="records")
    }


def _patient_valid(
    point: tuple[float, float, float],
    cache: dict[tuple[float, float, float], bool],
) -> bool:
    if point not in cache:
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=point[0],
            knee_amplitude_delta_deg=point[1],
            knee_phase_shift=point[2],
        )
        cache[point] = default_virtual_patient_envelope().contains(
            generated.trajectory
        )
    return cache[point]


def evaluate_endpoint_candidates(
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    spec: FrameworkSpec,
    calibration: SemanticsCalibration,
    *,
    executed_keys: set[tuple[float, float, float]],
    patient_validity_cache: dict[tuple[float, float, float], bool],
) -> pd.DataFrame:
    """Return current plus six direct endpoints for one frozen horizon."""

    lookup = _map_lookup(prediction_map)
    start = current.key()
    if start not in lookup:
        raise RuntimeError("current alpha missing from prediction map")
    current_row = dict(lookup[start])
    start_j = float(current_row["J_pred"])
    bound = framework_uncertainty(spec, calibration)
    rows: list[dict[str, Any]] = []
    current_row.update(
        {
            "candidate_type": "CURRENT_OPERATING_ENDPOINT_ORIGIN",
            "current_best_J_pred": start_j,
            "delta_J_pred_vs_current": 0.0,
            "predicted_improvement_magnitude": 0.0,
            "framework_id": spec.framework_id,
            "horizon_steps": spec.horizon_steps,
            "calibrated_uncertainty": bound,
            "magnitude_gate_pass": False,
            "direction_gate_pass": False,
            "decision_guard_status": CURRENT_BEST_NOT_A_CANDIDATE,
            "research_exploit_eligible": False,
            "latent_intermediate_trajectory_ids": "",
            "latent_intermediate_count": 0,
            "intermediate_execution_count": 0,
            "endpoint_execution_count_if_selected": 0,
            "truth_used_for_authorization": False,
        }
    )
    rows.append(current_row)
    for axis in _AXES:
        axis_index = _AXIS_INDEX[axis]
        for direction, sign in (("NEGATIVE", -1.0), ("POSITIVE", 1.0)):
            path: list[tuple[float, float, float]] = []
            for step_number in range(spec.horizon_steps + 1):
                point = list(start)
                point[axis_index] += sign * _GRID_STEP[axis] * step_number
                path.append(_key(point))
            path_exists = all(point in lookup for point in path)
            if not path_exists:
                continue
            path_rows = [lookup[point] for point in path]
            endpoint = dict(path_rows[-1])
            endpoint_key = path[-1]
            geometric = all(
                bool(row["geometrically_admissible"]) for row in path_rows
            )
            provenance = all(
                str(row["parent_reference_sha256"]) == ACTIVE_REFERENCE_SHA256
                for row in path_rows
            )
            supported = all(bool(row["model_supported"]) for row in path_rows) and all(
                float(row["domain_coverage"]) >= MODEL_SUPPORT_COVERAGE_GATE_PERCENT
                for row in path_rows
            )
            patient = all(
                _patient_valid(point, patient_validity_cache) for point in path
            )
            unexecuted = endpoint_key not in executed_keys
            delta = float(endpoint["J_pred"]) - start_j
            improvement = -delta
            magnitude_gate = improvement > OBJECTIVE_EQUIVALENCE_TOLERANCE
            direction_gate = delta + bound < 0.0
            authorized = bool(
                geometric
                and provenance
                and supported
                and patient
                and unexecuted
                and magnitude_gate
                and direction_gate
            )
            if not geometric:
                status = GEOMETRICALLY_INADMISSIBLE
            elif not provenance or not supported or not patient:
                status = UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE
            elif not unexecuted:
                status = "EXECUTED_ENDPOINT_NOT_REELIGIBLE"
            elif not magnitude_gate:
                status = "ENDPOINT_MAGNITUDE_GATE_FAILED"
            elif not direction_gate:
                status = "ENDPOINT_DIRECTION_GATE_FAILED"
            elif authorized:
                status = RESEARCH_EXPLOIT_ELIGIBLE
            else:
                status = NO_INDEPENDENT_VALIDATION_EVIDENCE
            endpoint.update(
                {
                    "candidate_type": "DIRECT_ENDPOINT_CANDIDATE",
                    "current_best_J_pred": start_j,
                    "delta_J_pred_vs_current": delta,
                    "predicted_improvement_magnitude": improvement,
                    "framework_id": spec.framework_id,
                    "horizon_steps": spec.horizon_steps,
                    "coordinate": axis,
                    "direction": direction,
                    "calibrated_uncertainty": bound,
                    "magnitude_gate_pass": magnitude_gate,
                    "direction_gate_pass": direction_gate,
                    "magnitude_margin": improvement
                    - OBJECTIVE_EQUIVALENCE_TOLERANCE,
                    "direction_margin": improvement - bound,
                    "all_latent_nodes_geometry_valid": geometric,
                    "all_latent_nodes_provenance_valid": provenance,
                    "all_latent_nodes_model_supported": supported,
                    "all_latent_nodes_patient_envelope_valid": patient,
                    "endpoint_not_previously_executed": unexecuted,
                    "decision_guard_status": status,
                    "research_exploit_eligible": authorized,
                    "latent_intermediate_trajectory_ids": ";".join(
                        str(lookup[point]["trajectory_id"]) for point in path[1:-1]
                    ),
                    "latent_intermediate_count": max(spec.horizon_steps - 1, 0),
                    "intermediate_execution_count": 0,
                    "endpoint_execution_count_if_selected": 1,
                    "authorization_scope": "DIRECT_ENDPOINT_CANDIDATE_ONLY",
                    "truth_used_for_authorization": False,
                    "support_alone_approved_exploit": False,
                    "formal_personalization_approval": False,
                }
            )
            rows.append(endpoint)
    return pd.DataFrame(rows)


def select_endpoint_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    eligible = candidates.loc[candidates["research_exploit_eligible"].astype(bool)]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["J_pred", "trajectory_id"], kind="mergesort"
    ).iloc[0].copy()


def run_framework_shadow(
    state: InitialResearchState,
    spec: FrameworkSpec,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    manifest_gate: FrozenFrameworkManifestGate,
    calibration: SemanticsCalibration,
    *,
    patient_validity_cache: dict[tuple[float, float, float], bool] | None = None,
    trial_budget: int = MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
) -> PolicyRunResult:
    """Run one direct-endpoint development shadow and refit after each trial."""

    if spec.default_enabled or DEFAULT_ENABLED:
        raise PermissionError("framework analysis must remain default-off")
    if trial_budget != MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON:
        raise ValueError("diagnostic horizon is fixed")
    manifest_gate.require_frozen()
    patient_cache = patient_validity_cache if patient_validity_cache is not None else {}
    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    model = _model_for_iteration(state, parameters, domain_data, 0)
    prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
    initial_prediction_map = prediction_map.copy(deep=True)
    oracle = SelectionGatedVirtualTruthOracle(state.subject_id, state.scenario_name)

    reference = generate_personalized_trajectory()
    reference_trajectory = reference.trajectory.copy(deep=True)
    reference_id = str(reference.metadata["trajectory_id"])
    reference_trajectory["trajectory_id"] = reference_id
    token = oracle.declare_selected(reference_id, "REFERENCE_NORMALIZATION")
    manifest_gate.record_truth_access("REFERENCE_NORMALIZATION")
    reference_metrics = oracle.execute(token, reference_trajectory).actual_metrics

    operating_alpha = SearchAlpha()
    operating_actual_j = 1.0
    best_alpha = SearchAlpha()
    best_actual_j = 1.0
    executed_keys = {operating_alpha.key()}
    history_rows: list[dict[str, Any]] = []
    guard_frames: list[pd.DataFrame] = []
    stop_reason = ""
    model_update_count = 0
    cumulative_regret = 0.0

    for iteration in range(1, trial_budget + 1):
        truth_before_proposal = oracle.truth_calls
        guarded = evaluate_endpoint_candidates(
            prediction_map,
            operating_alpha,
            spec,
            calibration,
            executed_keys=executed_keys,
            patient_validity_cache=patient_cache,
        )
        guarded["iteration"] = iteration
        guarded["policy_id"] = spec.framework_id
        guarded["subject_id"] = state.subject_id
        guarded["scenario_name"] = state.scenario_name
        guarded["selected_for_execution"] = False
        guarded["selection_mode"] = ""
        selected = select_endpoint_candidate(guarded)
        purpose = TRIAL_PURPOSE_EXPLOIT
        selection_mode = "DIRECT_ENDPOINT"
        frontier_ranked = pd.DataFrame()
        if selected is None:
            frontier = build_local_exploration_frontier(prediction_map, executed_keys)
            if not frontier.empty:
                frontier = frontier.loc[~frontier["model_supported"].astype(bool)].copy()
            frontier_ranked = rank_exploration_frontier(frontier, fitting_data, parameters)
            valid = (
                frontier_ranked.loc[
                    frontier_ranked["exploration_candidate_valid"].astype(bool)
                ]
                if not frontier_ranked.empty
                else frontier_ranked
            )
            if not valid.empty:
                selected = valid.iloc[0].copy()
                purpose = TRIAL_PURPOSE_EXPLORE
                selection_mode = TRIAL_PURPOSE_EXPLORE
        if selected is None:
            if guarded.empty:
                stop_reason = STOP_NO_GEOMETRICALLY_VALID_CANDIDATE
            else:
                frontier_any = build_local_exploration_frontier(
                    prediction_map, executed_keys
                )
                unsupported = (
                    frontier_any.loc[~frontier_any["model_supported"].astype(bool)]
                    if not frontier_any.empty
                    else frontier_any
                )
                stop_reason = (
                    STOP_PATIENT_ENVELOPE_BOUNDARY
                    if not unsupported.empty
                    and not frontier_ranked.empty
                    and not frontier_ranked["exploration_candidate_valid"].astype(bool).any()
                    else STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
                )
            guarded["policy_decision"] = "STOP"
            guarded["prospective_stop_reason"] = stop_reason
            guard_frames.append(guarded)
            break

        selected_id = str(selected["trajectory_id"])
        selected_alpha = alpha_from_row(selected)
        if selected_alpha.key() == operating_alpha.key():
            raise RuntimeError("current endpoint cannot be re-executed")
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            difference = np.asarray(selected_alpha.key()) - np.asarray(
                operating_alpha.key()
            )
            changed = np.flatnonzero(np.abs(difference) > 1e-12)
            if len(changed) != 1:
                raise RuntimeError("endpoint exploit attempted a mixed-axis jump")
            axis = _AXES[int(changed[0])]
            expected = spec.horizon_steps * _GRID_STEP[axis]
            if not np.isclose(
                abs(difference[changed[0]]), expected, atol=1e-12, rtol=0.0
            ):
                raise RuntimeError("endpoint exploit used the wrong frozen horizon")
            guarded.loc[
                guarded["trajectory_id"].astype(str).eq(selected_id),
                ["selected_for_execution", "selection_mode"],
            ] = [True, selection_mode]
        guarded["policy_decision"] = purpose
        guarded["prospective_stop_reason"] = ""
        guard_frames.append(guarded)
        if oracle.truth_calls != truth_before_proposal:
            raise RuntimeError("proposal or endpoint authorization accessed truth")

        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = selected_id
        selection_token = oracle.declare_selected(selected_id, purpose)
        manifest_gate.record_truth_access("SELECTED_ENDPOINT_OR_EXPLORATION")
        execution = oracle.execute(selection_token, trajectory)
        if not execution.observation_valid:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
            break
        actual = _actual_objective(selected_id, execution, reference_metrics)
        current_row = _map_lookup(prediction_map)[operating_alpha.key()]
        predicted_j = float(selected["J_pred"])
        delta_pred = predicted_j - float(current_row["J_pred"])
        operating_before = operating_alpha
        operating_actual_before = operating_actual_j
        delta_actual = actual.mechanical_cost_j_rms - operating_actual_before
        best_before = best_actual_j
        accepted = accept_actual_trial(actual.mechanical_cost_j_rms, best_actual_j)
        if accepted:
            best_actual_j = actual.mechanical_cost_j_rms
            best_alpha = selected_alpha
            operating_alpha = selected_alpha
            operating_actual_j = actual.mechanical_cost_j_rms
        cumulative_regret += max(actual.mechanical_cost_j_rms - best_before, 0.0)
        executed_keys.add(selected_alpha.key())
        false_improvement = bool(
            purpose == TRIAL_PURPOSE_EXPLOIT
            and delta_pred < -OBJECTIVE_EQUIVALENCE_TOLERANCE
            and delta_actual >= -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )

        fitting_data = pd.concat(
            (fitting_data, execution.estimator_observations), ignore_index=True
        )
        domain_data = pd.concat(
            (domain_data, execution.estimator_observations), ignore_index=True
        )
        estimation = _fit_updated_model(fitting_data, parameters)
        if estimation.optimizer_success:
            parameters = dict(estimation.estimated_parameters)
            model_update_count += 1
        else:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
        model = _model_for_iteration(state, parameters, domain_data, iteration)
        prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
        history_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.framework_id,
                "framework_id": spec.framework_id,
                "horizon_steps": spec.horizon_steps,
                "iteration": iteration,
                "trial_purpose": purpose,
                "selection_mode": selection_mode,
                "trajectory_id": selected_id,
                "alpha_hip": selected_alpha.hip_delta_deg,
                "alpha_knee": selected_alpha.knee_delta_deg,
                "alpha_phase": selected_alpha.phase_delta,
                "operating_alpha_hip_before": operating_before.hip_delta_deg,
                "operating_alpha_knee_before": operating_before.knee_delta_deg,
                "operating_alpha_phase_before": operating_before.phase_delta,
                "operating_alpha_hip_after": operating_alpha.hip_delta_deg,
                "operating_alpha_knee_after": operating_alpha.knee_delta_deg,
                "operating_alpha_phase_after": operating_alpha.phase_delta,
                "J_pred": predicted_j,
                "actual_J": actual.mechanical_cost_j_rms,
                "operating_actual_J_before": operating_actual_before,
                "best_actual_J_before": best_before,
                "best_actual_J_after": best_actual_j,
                "delta_J_pred_endpoint": delta_pred,
                "delta_J_actual_vs_operating": delta_actual,
                "accepted_meaningful_improvement": accepted,
                "executed_false_improvement": false_improvement,
                "latent_intermediate_count": (
                    max(spec.horizon_steps - 1, 0)
                    if purpose == TRIAL_PURPOSE_EXPLOIT
                    else 0
                ),
                "intermediate_execution_count": 0,
                "endpoint_execution_count": (
                    1 if purpose == TRIAL_PURPOSE_EXPLOIT else 0
                ),
                "model_refit_after_execution": bool(estimation.optimizer_success),
                "full_map_recomputed_after_execution": True,
                "authorization_invalidated_after_execution": (
                    purpose == TRIAL_PURPOSE_EXPLOIT
                ),
                "truth_accessed_before_selection": False,
                "manifest_sha_verified_before_truth": True,
                "execution_status": (
                    EXECUTED_FALSE_IMPROVEMENT
                    if false_improvement
                    else "EXECUTED"
                ),
                "cumulative_regret_vs_best_before": cumulative_regret,
                "stop_reason_after_iteration": stop_reason,
            }
        )
        if stop_reason:
            break

    if not stop_reason:
        stop_reason = STOP_MAX_PERSONALIZATION_TRIALS
        if history_rows:
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
    history = pd.DataFrame(history_rows)
    guard_audit = pd.concat(guard_frames, ignore_index=True, sort=False)
    number_executed = len(history)
    exploit_history = (
        history.loc[history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT)]
        if number_executed
        else history
    )
    summary = {
        "case_id": f"{state.subject_id}__{state.scenario_name}",
        "subject_id": state.subject_id,
        "scenario_name": state.scenario_name,
        "policy_id": spec.framework_id,
        "framework_id": spec.framework_id,
        "horizon_steps": spec.horizon_steps,
        "calibrated_uncertainty": framework_uncertainty(spec, calibration),
        "research_status": RESEARCH_ONLY,
        "number_of_executed_trials": number_executed,
        "number_of_exploit_trials": len(exploit_history),
        "number_of_explore_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLORE).sum()) if number_executed else 0,
        "number_of_executed_false_improvements": int(history["executed_false_improvement"].sum()) if number_executed else 0,
        "first_exploit_iteration": (
            int(exploit_history["iteration"].min()) if not exploit_history.empty else np.nan
        ),
        "decision_latency_trials": (
            int(exploit_history["iteration"].min()) - 1
            if not exploit_history.empty
            else number_executed
        ),
        "latent_intermediate_nodes_skipped": int(history["latent_intermediate_count"].sum()) if number_executed else 0,
        "intermediate_trajectory_executions": int(history["intermediate_execution_count"].sum()) if number_executed else 0,
        "endpoint_executions": int(history["endpoint_execution_count"].sum()) if number_executed else 0,
        "reference_actual_J": 1.0,
        "final_best_actual_J": best_actual_j,
        "actual_J_reduction_from_reference": 1.0 - best_actual_j,
        "cumulative_regret_vs_best_before": cumulative_regret,
        "model_update_count": model_update_count,
        "final_best_alpha_hip": best_alpha.hip_delta_deg,
        "final_best_alpha_knee": best_alpha.knee_delta_deg,
        "final_best_alpha_phase": best_alpha.phase_delta,
        "final_operating_alpha_hip": operating_alpha.hip_delta_deg,
        "final_operating_alpha_knee": operating_alpha.knee_delta_deg,
        "final_operating_alpha_phase": operating_alpha.phase_delta,
        "stop_reason": stop_reason,
        "trial_budget": trial_budget,
        "heldout_final_test_used": False,
        "calibration_cases_used_for_policy_outcomes": False,
        "prospective_cohort_run": False,
        "new_policy_default_enabled": False,
        "human_ready": False,
        "robot_motion_approved": False,
        "model_reliability_status": "RESEARCH_DIAGNOSTIC_NOT_FORMALLY_RELIABLE",
    }
    return PolicyRunResult(
        subject_id=state.subject_id,
        scenario_name=state.scenario_name,
        policy_id=spec.framework_id,
        trial_history=history,
        decision_guard_audit=guard_audit,
        parameter_history=pd.DataFrame(),
        prediction_map_history=pd.DataFrame(),
        known_region_history=pd.DataFrame(),
        uncertainty_history=pd.DataFrame(),
        uncertainty_pairwise_audit=pd.DataFrame(),
        exploration_information_gain=pd.DataFrame(),
        false_improvement_audit=pd.DataFrame(),
        summary=summary,
        initial_prediction_map=initial_prediction_map,
        final_prediction_map=prediction_map,
        truth_access_audit={
            "manifest_verified_before_every_truth": True,
            "proposal_truth_accessed": False,
            "heldout_final_test_used": False,
        },
    )


def small_step_recovery(
    calibration: SemanticsCalibration,
) -> pd.DataFrame:
    source = pd.read_csv(SMALL_STEP_SOURCE_PATH)
    rows: list[dict[str, Any]] = []
    for path_id, group in source.groupby("path_id", sort=True):
        ordered = group.sort_values("step_number")
        first = ordered.iloc[0]
        for spec in FRAMEWORKS:
            selected = ordered.loc[ordered["step_number"].eq(spec.horizon_steps)]
            if selected.empty:
                predicted = truth = np.nan
                gate_a = gate_b = recovered = False
            else:
                item = selected.iloc[0]
                predicted = float(item["cumulative_endpoint_deltaJ_pred"])
                truth = float(item["cumulative_endpoint_deltaJ_truth"])
                improvement = -predicted
                gate_a = improvement > OBJECTIVE_EQUIVALENCE_TOLERANCE
                gate_b = predicted + framework_uncertainty(spec, calibration) < 0.0
                recovered = bool(
                    gate_a
                    and gate_b
                    and truth < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                )
            rows.append(
                {
                    "path_id": path_id,
                    "case_id": first["case_id"],
                    "coordinate": first["coordinate"],
                    "direction": first["direction"],
                    "framework_id": spec.framework_id,
                    "horizon_steps": spec.horizon_steps,
                    "predicted_endpoint_delta_J": predicted,
                    "truth_endpoint_delta_J_posthoc": truth,
                    "calibrated_uncertainty": framework_uncertainty(
                        spec, calibration
                    ),
                    "magnitude_gate_pass": gate_a,
                    "direction_gate_pass": gate_b,
                    "recovered_small_step_path": recovered,
                    "intermediate_trajectories_executed": False,
                    "truth_used_for_authorization": False,
                    "truth_attached_posthoc_only": True,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ANALYSIS_ID",
    "DEFAULT_ENABLED",
    "FRAMEWORKS",
    "MANIFEST_ID",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_APPROVED",
    "OFFLINE_ONLY",
    "PRIOR_SEMANTICS_MANIFEST_SHA256",
    "FrameworkSpec",
    "FrozenFrameworkManifestGate",
    "canonical_json_bytes",
    "evaluate_endpoint_candidates",
    "framework_uncertainty",
    "load_semantics_calibration",
    "manifest_payload",
    "run_framework_shadow",
    "select_endpoint_candidate",
    "small_step_recovery",
]
