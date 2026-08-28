"""Generate the frozen, post-hoc BUNDLE_5 boundary/specificity audit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    generate_personalized_trajectory,
)
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    TrajectoryComponentCache,
    build_predicted_map,
    build_trajectory_component_cache,
    evaluate_truth_map,
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    validate_active_reference_file,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_adaptive_horizon_decision_prototype import (
    ADAPTIVE_HORIZON_SEQUENCE,
)
from .p2_bundle5_boundary_subject_specificity_audit import (
    ADAPTIVE_MANIFEST_PATH,
    ADAPTIVE_MANIFEST_SHA256,
    AUDIT_ID,
    BOUNDARY_MIXED,
    FINAL_IDENTIFIED,
    FINAL_MORE_EVIDENCE,
    MANIFEST_ID,
    MULTI_STEP_ARTIFACT_DIRECTORY,
    NOT_HUMAN_READY,
    NOT_ROBOT_APPROVED,
    OBJECTIVE_NO_CHANGE,
    OFFLINE_ONLY,
    build_best_j_progression,
    build_boundary_timeline,
    build_gain_timing,
    build_subject_discrimination,
    build_truth_axis_profiles,
    build_truth_optimum_by_case,
    classify_boundary_collapse,
    classify_objective_status,
    classify_trial_cost,
    classify_trial_values,
    manifest_payload,
)
from .p2_decision_rule_semantics_audit import sha256_file
from .p2_multi_step_decision_framework_analysis import (
    FRAMEWORKS,
    FrozenFrameworkManifestGate,
    _map_lookup,
    evaluate_endpoint_candidates,
    load_semantics_calibration,
    select_endpoint_candidate,
)
from .p2_v2_prospective_offline_validation import (
    EXPECTED_GEOMETRIC_LATTICE_SIZE,
    dynamic_subject_for_id,
    registered_prospective_subject,
)
from .post_prospective_rejection_root_cause_audit import (
    verify_immutable_prospective_artifacts,
)
from .research_decision_guarded_sequential_personalization import (
    TRIAL_PURPOSE_EXPLOIT,
    TRIAL_PURPOSE_EXPLORE,
    SelectionGatedVirtualTruthOracle,
    _actual_objective,
    _fit_updated_model,
    _model_for_iteration,
    alpha_from_row,
    build_initial_research_state,
)
from .run_p2_multi_step_decision_framework_analysis import (
    _case_table,
    _write_csv,
    _write_json,
    _write_text,
)
from .run_p2_v2_prospective_offline_validation import _protected_source_hashes
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .sequential_personalization import SearchAlpha, accept_actual_trial


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_bundle5_boundary_subject_specificity_audit.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_p2_bundle5_boundary_subject_specificity_audit.py"
ADAPTIVE_CORE_PATH = MODULE_DIR / "p2_adaptive_horizon_decision_prototype.py"
ADAPTIVE_RUNNER_PATH = MODULE_DIR / "run_p2_adaptive_horizon_decision_prototype.py"
ADAPTIVE_TEST_PATH = MODULE_DIR / "test_p2_adaptive_horizon_decision_prototype.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_bundle5_boundary_subject_specificity_audit_v1"
)
REQUIRED_OUTPUT_FILENAMES = (
    "BUNDLE5_AUDIT_MANIFEST_V1.json",
    "bundle5_truth_optimum_by_case.csv",
    "bundle5_subject_discrimination.csv",
    "bundle5_axis_direction_decision_audit.csv",
    "bundle5_truth_axis_profiles.csv",
    "bundle5_boundary_arrival_timeline.csv",
    "bundle5_best_J_progression.csv",
    "bundle5_trial_value_audit.csv",
    "bundle5_objective_decomposition.csv",
    "BUNDLE5_BOUNDARY_ROOT_CAUSE_MATRIX.csv",
    "BUNDLE5_BOUNDARY_SUBJECT_SPECIFICITY_REPORT.md",
    "DATA_ROLE_AUDIT.md",
    "metadata.json",
)

_ADAPTIVE_REQUIRED = (
    ADAPTIVE_CORE_PATH,
    ADAPTIVE_RUNNER_PATH,
    ADAPTIVE_TEST_PATH,
    ADAPTIVE_MANIFEST_PATH,
    ADAPTIVE_MANIFEST_PATH.with_name("metadata.json"),
    ADAPTIVE_MANIFEST_PATH.with_name("adaptive_trial_history.csv"),
    ADAPTIVE_MANIFEST_PATH.with_name("horizon_usage.csv"),
)
_H2_SPEC = next(spec for spec in FRAMEWORKS if spec.framework_id == "BUNDLE_5")


@dataclass
class ReplayResult:
    trial_diagnostics: pd.DataFrame
    axis_decisions: pd.DataFrame
    truth_landscape: pd.DataFrame
    objective_decomposition: pd.DataFrame


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    return completed.stdout.rstrip("\n")


def _checkpoint_preflight() -> dict[str, Any]:
    if sha256_file(ADAPTIVE_MANIFEST_PATH) != ADAPTIVE_MANIFEST_SHA256:
        raise RuntimeError("BUNDLE5_AUDIT_REQUIRES_CHECKPOINT")
    head = _git_output("rev-parse", "HEAD")
    adaptive_commit = _git_output(
        "log", "-1", "--format=%H", "--", str(ADAPTIVE_CORE_PATH.relative_to(PROJECT_ROOT))
    )
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", adaptive_commit, head],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        timeout=20.0,
    )
    if ancestor_check.returncode != 0:
        raise RuntimeError("BUNDLE5_AUDIT_REQUIRES_CHECKPOINT")
    verified: dict[str, str] = {}
    for path in _ADAPTIVE_REQUIRED:
        relative = str(path.relative_to(PROJECT_ROOT))
        try:
            _git_output("ls-files", "--error-unmatch", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("BUNDLE5_AUDIT_REQUIRES_CHECKPOINT") from exc
        committed = subprocess.run(
            ["git", "show", f"{adaptive_commit}:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise RuntimeError("BUNDLE5_AUDIT_REQUIRES_CHECKPOINT")
        verified[relative] = sha256_file(path)
    return {
        "checkpoint_commit": head,
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "adaptive_checkpoint_commit": adaptive_commit,
        "adaptive_checkpoint_is_current_HEAD": adaptive_commit == head,
        "adaptive_checkpoint_is_ancestor_of_current_HEAD": True,
        "adaptive_manifest_sha256": ADAPTIVE_MANIFEST_SHA256,
        "tracked_adaptive_artifact_sha256": verified,
    }


def _source_hashes() -> dict[str, str]:
    hashes = _protected_source_hashes()
    hashes.update(
        {
            "audit_core": sha256_file(CORE_SOURCE_PATH),
            "audit_runner": sha256_file(RUNNER_SOURCE_PATH),
            "adaptive_core": sha256_file(ADAPTIVE_CORE_PATH),
            "adaptive_runner": sha256_file(ADAPTIVE_RUNNER_PATH),
            "adaptive_test": sha256_file(ADAPTIVE_TEST_PATH),
            "adaptive_manifest": sha256_file(ADAPTIVE_MANIFEST_PATH),
            "frozen_H2_history": sha256_file(
                MULTI_STEP_ARTIFACT_DIRECTORY / "framework_trial_history.csv"
            ),
            "frozen_H2_summary": sha256_file(
                MULTI_STEP_ARTIFACT_DIRECTORY / "framework_case_summary.csv"
            ),
        }
    )
    return hashes


def _eligible_ids(table: pd.DataFrame) -> set[str]:
    return set(
        table.loc[table["research_exploit_eligible"].astype(bool), "trajectory_id"]
        .astype(str)
        .tolist()
    )


def _parameter_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> tuple[dict[str, float], bool]:
    if set(before) != set(after):
        raise RuntimeError("five-parameter keys changed during replay")
    delta = {key: float(after[key]) - float(before[key]) for key in sorted(before)}
    return delta, any(value != 0.0 for value in delta.values())


def _attach_guard_truth(
    guarded: pd.DataFrame,
    neutral_row: pd.Series,
    truth_model: Any,
    cache: TrajectoryComponentCache,
    gate: FrozenFrameworkManifestGate,
    *,
    case_id: str,
    subject_id: str,
    scenario_name: str,
    iteration: int,
    decision_state: str,
    frozen_selected_id: str | None,
) -> pd.DataFrame:
    gate.record_truth_access("POST_DECISION_AXIS_CANDIDATE_TRUTH")
    evaluation = guarded.copy()
    if not evaluation["trajectory_id"].astype(str).eq(
        str(neutral_row["trajectory_id"])
    ).any():
        evaluation = pd.concat(
            (evaluation, neutral_row.to_frame().T), ignore_index=True, sort=False
        )
    truth, _ = evaluate_truth_map(evaluation, truth_model, cache, batch_size=64)
    current = truth.loc[truth["candidate_type"].eq("CURRENT_OPERATING_ENDPOINT_ORIGIN")]
    if len(current) != 1:
        raise RuntimeError("axis audit does not contain exactly one current endpoint")
    current_truth = float(current.iloc[0]["J_truth"])
    direct = truth.loc[truth["candidate_type"].eq("DIRECT_ENDPOINT_CANDIDATE")].copy()
    direct["case_id"] = case_id
    direct["subject_id"] = subject_id
    direct["scenario_name"] = scenario_name
    direct["iteration"] = iteration
    direct["decision_state"] = decision_state
    direct["axis_direction"] = (
        direct["coordinate"].str.upper() + "_" + direct["direction"].str.upper()
    )
    direct["predicted_cumulative_delta_J"] = direct["delta_J_pred_vs_current"]
    direct["meaningful_improvement_component"] = (
        direct["predicted_improvement_magnitude"]
        - OBJECTIVE_EQUIVALENCE_TOLERANCE
    )
    direct["decision_evidence_margin"] = direct["direction_margin"]
    direct["truth_cumulative_delta_J_posthoc"] = direct["J_truth"] - current_truth
    direct["authorized"] = direct["research_exploit_eligible"].astype(bool)
    direct["selected"] = direct["trajectory_id"].astype(str).eq(
        str(frozen_selected_id)
    ) if frozen_selected_id is not None else False
    direct["all_nondecision_gates_available"] = (
        direct["all_latent_nodes_geometry_valid"].astype(bool)
        & direct["all_latent_nodes_provenance_valid"].astype(bool)
        & direct["all_latent_nodes_model_supported"].astype(bool)
        & direct["all_latent_nodes_patient_envelope_valid"].astype(bool)
    )
    direct["truth_direction_supports_improvement"] = direct[
        "truth_cumulative_delta_J_posthoc"
    ].lt(0.0)
    direct["truth_meaningful_improvement"] = direct[
        "truth_cumulative_delta_J_posthoc"
    ].lt(-OBJECTIVE_EQUIVALENCE_TOLERANCE)
    direct["truth_used_for_authorization"] = False
    direct["truth_attached_posthoc_only"] = True
    authorized = direct.loc[direct["authorized"]].sort_values(
        ["J_pred", "trajectory_id"], kind="mergesort"
    )
    rank = {identifier: index + 1 for index, identifier in enumerate(authorized["trajectory_id"].astype(str))}
    direct["authorized_prediction_rank"] = direct["trajectory_id"].astype(str).map(rank)
    columns = [
        "case_id",
        "subject_id",
        "scenario_name",
        "iteration",
        "decision_state",
        "trajectory_id",
        "axis_direction",
        "coordinate",
        "direction",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "predicted_cumulative_delta_J",
        "calibrated_uncertainty",
        "meaningful_improvement_component",
        "decision_evidence_margin",
        "truth_cumulative_delta_J_posthoc",
        "magnitude_gate_pass",
        "direction_gate_pass",
        "all_nondecision_gates_available",
        "authorized",
        "authorized_prediction_rank",
        "selected",
        "truth_direction_supports_improvement",
        "truth_meaningful_improvement",
        "domain_coverage",
        "model_supported",
        "decision_guard_status",
        "truth_used_for_authorization",
        "truth_attached_posthoc_only",
    ]
    return direct[columns]


def _objective_row(
    *, case_id: str, role: str, alpha: tuple[float, float, float], result: Any
) -> dict[str, Any]:
    values = result.as_dict()
    hip_ratio = float(values["hip_rms_ratio"])
    knee_ratio = float(values["knee_rms_ratio"])
    return {
        "case_id": case_id,
        "trajectory_role": role,
        "alpha_hip": alpha[0],
        "alpha_knee": alpha[1],
        "alpha_phase": alpha[2],
        **values,
        "hip_squared_ratio_reduction_vs_reference": 1.0 - hip_ratio**2,
        "knee_squared_ratio_reduction_vs_reference": 1.0 - knee_ratio**2,
        "knee_component_reduction_exceeds_hip": bool(
            1.0 - knee_ratio**2 > 1.0 - hip_ratio**2
        ),
        "truth_role": "POSTHOC_OBJECTIVE_DECOMPOSITION_ONLY",
        "used_for_policy_selection": False,
    }


def _reference_objective_row(
    case_id: str, reference_metrics: Any
) -> dict[str, Any]:
    metrics = reference_metrics.as_dict()
    return {
        "case_id": case_id,
        "trajectory_role": "REFERENCE",
        "alpha_hip": 0.0,
        "alpha_knee": 0.0,
        "alpha_phase": 0.0,
        "trajectory_id": "REFERENCE_NORMALIZATION",
        "hip_rms_ratio": 1.0,
        "knee_rms_ratio": 1.0,
        "mechanical_cost_j_rms": 1.0,
        **metrics,
        **{f"reference_{key}": value for key, value in metrics.items()},
        "hip_squared_ratio_reduction_vs_reference": 0.0,
        "knee_squared_ratio_reduction_vs_reference": 0.0,
        "knee_component_reduction_exceeds_hip": False,
        "truth_role": "POSTHOC_OBJECTIVE_DECOMPOSITION_ONLY",
        "used_for_policy_selection": False,
    }


def _replay_case(
    *,
    case: Mapping[str, Any],
    frozen_history: pd.DataFrame,
    h2_summary: pd.Series,
    lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    gate: FrozenFrameworkManifestGate,
    patient_cache: dict[tuple[float, float, float], bool],
) -> ReplayResult:
    case_id = str(case["case_id"])
    subject_id = str(case["subject_id"])
    scenario_name = str(case["scenario_name"])
    gate.record_truth_access("INITIAL_IDENTIFICATION_AFTER_AUDIT_MANIFEST_FREEZE")
    state = build_initial_research_state(subject_id, scenario_name)
    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    model = _model_for_iteration(state, parameters, domain_data, 0)
    truth_model = model
    prediction_map, _ = build_predicted_map(model, lattice, cache)
    initial_prediction_map = prediction_map.copy(deep=True)
    neutral_row = initial_prediction_map.loc[
        np.isclose(initial_prediction_map["hip_delta"], 0.0)
        & np.isclose(initial_prediction_map["knee_delta"], 0.0)
        & np.isclose(initial_prediction_map["phase_delta"], 0.0)
    ].iloc[0].copy()
    calibration = load_semantics_calibration()
    oracle = SelectionGatedVirtualTruthOracle(subject_id, scenario_name)

    reference = generate_personalized_trajectory()
    reference_trajectory = reference.trajectory.copy(deep=True)
    reference_id = str(reference.metadata["trajectory_id"])
    reference_trajectory["trajectory_id"] = reference_id
    token = oracle.declare_selected(reference_id, "REFERENCE_NORMALIZATION")
    gate.record_truth_access("REFERENCE_NORMALIZATION")
    reference_execution = oracle.execute(token, reference_trajectory)
    reference_metrics = reference_execution.actual_metrics

    operating_alpha = SearchAlpha()
    operating_actual_j = 1.0
    best_alpha = SearchAlpha()
    best_actual_j = 1.0
    executed_keys = {operating_alpha.key()}
    trial_rows: list[dict[str, Any]] = []
    axis_frames: list[pd.DataFrame] = []
    objective_by_key: dict[tuple[float, float, float], Any] = {}
    best_executed_pred = np.inf

    ordered = frozen_history.sort_values("iteration")
    for frozen in ordered.to_dict(orient="records"):
        iteration = int(frozen["iteration"])
        purpose = str(frozen["trial_purpose"])
        guarded = evaluate_endpoint_candidates(
            prediction_map,
            operating_alpha,
            _H2_SPEC,
            calibration,
            executed_keys=executed_keys,
            patient_validity_cache=patient_cache,
        )
        selected_endpoint = select_endpoint_candidate(guarded)
        frozen_id = str(frozen["trajectory_id"])
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            if selected_endpoint is None or str(selected_endpoint["trajectory_id"]) != frozen_id:
                raise RuntimeError(f"H2 replay exploit differs from checkpoint: {case_id}/{iteration}")
        elif selected_endpoint is not None:
            raise RuntimeError(f"H2 replay exploration had eligible exploit: {case_id}/{iteration}")
        axis_frames.append(
            _attach_guard_truth(
                guarded,
                neutral_row,
                truth_model,
                cache,
                gate,
                case_id=case_id,
                subject_id=subject_id,
                scenario_name=scenario_name,
                iteration=iteration,
                decision_state=purpose,
                frozen_selected_id=frozen_id if purpose == TRIAL_PURPOSE_EXPLOIT else None,
            )
        )
        eligible_before = _eligible_ids(guarded)
        selected_alpha = SearchAlpha(
            hip_delta_deg=float(frozen["alpha_hip"]),
            knee_delta_deg=float(frozen["alpha_knee"]),
            phase_delta=float(frozen["alpha_phase"]),
        )
        selected_map_row = _map_lookup(prediction_map)[selected_alpha.key()]
        if not np.isclose(
            float(selected_map_row["J_pred"]), float(frozen["J_pred"]), atol=1e-11, rtol=0.0
        ):
            raise RuntimeError(f"H2 replay J_pred differs from checkpoint: {case_id}/{iteration}")
        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = frozen_id
        selection_token = oracle.declare_selected(frozen_id, purpose)
        gate.record_truth_access("FROZEN_H2_TRAJECTORY_REPLAY")
        execution = oracle.execute(selection_token, trajectory)
        actual = _actual_objective(frozen_id, execution, reference_metrics)
        objective_by_key[selected_alpha.key()] = actual
        if not np.isclose(
            actual.mechanical_cost_j_rms, float(frozen["actual_J"]), atol=1e-11, rtol=0.0
        ):
            raise RuntimeError(f"H2 replay actual J differs from checkpoint: {case_id}/{iteration}")

        parameters_before = dict(parameters)
        map_before = prediction_map.copy(deep=True)
        support_before = int(map_before["model_supported"].sum())
        operating_before = operating_alpha
        best_before = best_alpha
        best_actual_before = best_actual_j
        accepted = accept_actual_trial(actual.mechanical_cost_j_rms, best_actual_j)
        if accepted:
            best_actual_j = actual.mechanical_cost_j_rms
            best_alpha = selected_alpha
            operating_alpha = selected_alpha
            operating_actual_j = actual.mechanical_cost_j_rms
        if bool(accepted) != bool(frozen["accepted_meaningful_improvement"]):
            raise RuntimeError(f"H2 replay acceptance differs from checkpoint: {case_id}/{iteration}")
        executed_keys.add(selected_alpha.key())
        fitting_data = pd.concat(
            (fitting_data, execution.estimator_observations), ignore_index=True
        )
        domain_data = pd.concat(
            (domain_data, execution.estimator_observations), ignore_index=True
        )
        estimation = _fit_updated_model(fitting_data, parameters)
        if not estimation.optimizer_success:
            raise RuntimeError(f"H2 posthoc replay model refit failed: {case_id}/{iteration}")
        parameters = dict(estimation.estimated_parameters)
        model = _model_for_iteration(state, parameters, domain_data, iteration)
        prediction_map, _ = build_predicted_map(model, lattice, cache)
        if len(prediction_map) != len(map_before) or not prediction_map[
            "trajectory_id"
        ].astype(str).equals(map_before["trajectory_id"].astype(str)):
            raise RuntimeError("prediction map grid changed during H2 replay")
        map_delta = prediction_map["J_pred"].to_numpy(dtype=float) - map_before[
            "J_pred"
        ].to_numpy(dtype=float)
        guarded_after = evaluate_endpoint_candidates(
            prediction_map,
            operating_alpha,
            _H2_SPEC,
            calibration,
            executed_keys=executed_keys,
            patient_validity_cache=patient_cache,
        )
        eligible_after = _eligible_ids(guarded_after)
        parameter_delta, parameter_changed = _parameter_delta(
            parameters_before, parameters
        )
        best_executed_pred = min(best_executed_pred, float(frozen["J_pred"]))
        best_after_key = best_alpha.key()
        frozen_after = (
            float(frozen["operating_alpha_hip_after"]),
            float(frozen["operating_alpha_knee_after"]),
            float(frozen["operating_alpha_phase_after"]),
        )
        if not np.allclose(best_after_key, frozen_after, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"H2 replay alpha differs from checkpoint: {case_id}/{iteration}")
        if not np.isclose(
            best_actual_j, float(frozen["best_actual_J_after"]), atol=1e-11, rtol=0.0
        ):
            raise RuntimeError(f"H2 replay best J differs from checkpoint: {case_id}/{iteration}")
        trial_rows.append(
            {
                "case_id": case_id,
                "subject_id": subject_id,
                "scenario_name": scenario_name,
                "iteration": iteration,
                "trial_purpose": purpose,
                "trajectory_id": frozen_id,
                "alpha_hip": selected_alpha.hip_delta_deg,
                "alpha_knee": selected_alpha.knee_delta_deg,
                "alpha_phase": selected_alpha.phase_delta,
                "best_alpha_hip_before": best_before.hip_delta_deg,
                "best_alpha_knee_before": best_before.knee_delta_deg,
                "best_alpha_phase_before": best_before.phase_delta,
                "best_alpha_hip_after": best_alpha.hip_delta_deg,
                "best_alpha_knee_after": best_alpha.knee_delta_deg,
                "best_alpha_phase_after": best_alpha.phase_delta,
                "actual_J": actual.mechanical_cost_j_rms,
                "best_actual_J_before": best_actual_before,
                "best_actual_J_after": best_actual_j,
                "actual_best_J_improvement": best_actual_before - best_actual_j,
                "changed_best_alpha": best_before.key() != best_alpha.key(),
                "accepted_meaningful_improvement": accepted,
                "J_pred_at_execution": float(frozen["J_pred"]),
                "best_executed_J_pred_so_far": best_executed_pred,
                "map_best_supported_J_pred_after": float(
                    prediction_map.loc[
                        prediction_map["model_supported"].astype(bool), "J_pred"
                    ].min()
                ),
                "eligible_endpoint_ids_before": ";".join(sorted(eligible_before)),
                "eligible_endpoint_ids_after": ";".join(sorted(eligible_after)),
                "changed_future_exploit_eligibility": eligible_before != eligible_after,
                "future_eligibility_added": ";".join(sorted(eligible_after - eligible_before)),
                "future_eligibility_removed": ";".join(sorted(eligible_before - eligible_after)),
                "parameter_delta_json": json.dumps(parameter_delta, sort_keys=True),
                "parameter_changed_exactly": parameter_changed,
                "prediction_map_change_rms": float(np.sqrt(np.mean(map_delta**2))),
                "prediction_map_change_max_abs": float(np.max(np.abs(map_delta))),
                "prediction_map_changed_exactly": bool(np.any(map_delta != 0.0)),
                "supported_point_count_before": support_before,
                "supported_point_count_after": int(
                    prediction_map["model_supported"].sum()
                ),
                "support_point_increase": int(
                    prediction_map["model_supported"].sum()
                )
                - support_before,
                "model_refit_after_execution": True,
                "full_map_recomputed_after_execution": True,
                "frozen_history_row_verified": True,
                "truth_used_to_select_or_stop": False,
            }
        )

    final_iteration = int(ordered["iteration"].max()) + 1
    final_guard = evaluate_endpoint_candidates(
        prediction_map,
        operating_alpha,
        _H2_SPEC,
        calibration,
        executed_keys=executed_keys,
        patient_validity_cache=patient_cache,
    )
    if select_endpoint_candidate(final_guard) is not None:
        raise RuntimeError(f"H2 replay final stop still has eligible endpoint: {case_id}")
    axis_frames.append(
        _attach_guard_truth(
            final_guard,
            neutral_row,
            truth_model,
            cache,
            gate,
            case_id=case_id,
            subject_id=subject_id,
            scenario_name=scenario_name,
            iteration=final_iteration,
            decision_state="FINAL_STOP_DECISION",
            frozen_selected_id=None,
        )
    )
    gate.record_truth_access("POST_REPLAY_FULL_TRUTH_LANDSCAPE")
    truth_landscape, _ = evaluate_truth_map(
        initial_prediction_map, truth_model, cache
    )
    optimum = truth_landscape.sort_values(
        ["J_truth", "trajectory_id"], kind="mergesort"
    ).iloc[0]
    global_key = (
        float(optimum["hip_delta"]),
        float(optimum["knee_delta"]),
        float(optimum["phase_delta"]),
    )
    final_key = (
        float(h2_summary["final_best_alpha_hip"]),
        float(h2_summary["final_best_alpha_knee"]),
        float(h2_summary["final_best_alpha_phase"]),
    )
    objective_rows = [_reference_objective_row(case_id, reference_metrics)]
    for role, alpha in (
        ("TRUTH_GLOBAL_OPTIMUM", global_key),
        ("H2_FINAL", final_key),
    ):
        result = objective_by_key.get(tuple(round(value, 12) for value in alpha))
        if result is None:
            generated = generate_personalized_trajectory(
                hip_amplitude_delta_deg=alpha[0],
                knee_amplitude_delta_deg=alpha[1],
                knee_phase_shift=alpha[2],
            )
            trajectory = generated.trajectory.copy(deep=True)
            trajectory_id = str(generated.metadata["trajectory_id"])
            trajectory["trajectory_id"] = trajectory_id
            token = oracle.declare_selected(trajectory_id, TRIAL_PURPOSE_EXPLORE)
            gate.record_truth_access(f"POSTHOC_OBJECTIVE_{role}")
            execution = oracle.execute(token, trajectory)
            result = _actual_objective(trajectory_id, execution, reference_metrics)
        objective_rows.append(
            _objective_row(case_id=case_id, role=role, alpha=alpha, result=result)
        )
    return ReplayResult(
        trial_diagnostics=pd.DataFrame(trial_rows),
        axis_decisions=pd.concat(axis_frames, ignore_index=True, sort=False),
        truth_landscape=truth_landscape,
        objective_decomposition=pd.DataFrame(objective_rows),
    )


def _axis_summary(axis_audit: pd.DataFrame) -> pd.DataFrame:
    return (
        axis_audit.groupby("axis_direction", as_index=False)
        .agg(
            candidate_count=("trajectory_id", "count"),
            mean_predicted_improvement=(
                "predicted_cumulative_delta_J",
                lambda values: float((-values).mean()),
            ),
            mean_calibrated_uncertainty=("calibrated_uncertainty", "mean"),
            nondecision_availability_rate=("all_nondecision_gates_available", "mean"),
            authorization_count=("authorized", "sum"),
            selection_count=("selected", "sum"),
            mean_truth_improvement=(
                "truth_cumulative_delta_J_posthoc",
                lambda values: float((-values).mean()),
            ),
            truth_improvement_agreement_rate=(
                "truth_direction_supports_improvement",
                "mean",
            ),
        )
    )


def _root_matrix(
    *,
    optimum: pd.DataFrame,
    axis_audit: pd.DataFrame,
    timeline: pd.DataFrame,
    trial_values: pd.DataFrame,
    boundary_classification: str,
    objective_status: str,
    trial_cost_status: str,
) -> pd.DataFrame:
    truth_unique = len(
        optimum[
            [
                "truth_global_alpha_hip",
                "truth_global_alpha_knee",
                "truth_global_alpha_phase",
            ]
        ].drop_duplicates()
    )
    truth_knee_minus5 = int(
        np.isclose(optimum["truth_global_alpha_knee"], -5.0).sum()
    )
    selected = axis_audit.loc[axis_audit["selected"].astype(bool)]
    low = int(
        trial_values["trial_value_classification"].isin(
            ("POST_OPTIMUM_LOW_VALUE", "BOUNDARY_CHASING_LOW_VALUE")
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "problem": "15/15 H2 final same alpha",
                "possible_cause": "truth landscape + policy ranking",
                "evidence": (
                    f"truth full optima={truth_unique}; truth knee=-5={truth_knee_minus5}/15; "
                    "H2 full alpha unique=1"
                ),
                "conclusion": boundary_classification,
            },
            {
                "problem": "116 trials",
                "possible_cause": "initial exploration plus four useful boundary endpoints per case",
                "evidence": (
                    f"low-value={low}/116; trials after final alpha="
                    f"{int(timeline['trials_after_first_final_alpha'].sum())}"
                ),
                "conclusion": trial_cost_status,
            },
            {
                "problem": "H3 degenerates to H5",
                "possible_cause": "short horizons fail unchanged confidence gates",
                "evidence": "adaptive endpoint usage H1=0,H2=0,H3=1,H5=59",
                "conclusion": "HORIZON_ESCALATION_DRIVEN_BY_FROZEN_DECISION_EVIDENCE",
            },
            {
                "problem": "high J improvement but weak specificity",
                "possible_cause": "uniform truth knee direction plus policy loss of hip/phase diversity",
                "evidence": (
                    f"selected knee-negative={int(selected['axis_direction'].eq('KNEE_NEGATIVE').sum())}/"
                    f"{len(selected)}; objective status={objective_status}"
                ),
                "conclusion": boundary_classification,
            },
            {
                "problem": "knee-negative dominance",
                "possible_cause": "prediction magnitude / availability / uncertainty / ranking / truth",
                "evidence": "all horizons use the same H5 uncertainty; see axis-direction audit",
                "conclusion": "PREDICTION_MAGNITUDE_AND_TRUTH_DIRECTION_WITH_EQUAL_UNCERTAINTY",
            },
            {
                "problem": "objective subject discrimination",
                "possible_cause": "normalization / torque response / generator geometry",
                "evidence": f"full truth optima={truth_unique}; knee boundary={truth_knee_minus5}/15",
                "conclusion": objective_status,
            },
        ]
    )


def _data_role_audit(checkpoint: Mapping[str, Any]) -> str:
    return f"""# DATA_ROLE_AUDIT

- Audit: `{AUDIT_ID}`
- Adaptive checkpoint: `{checkpoint['adaptive_checkpoint_commit']}`
- Adaptive manifest SHA-256: `{ADAPTIVE_MANIFEST_SHA256}`
- Policy-outcome evidence: the existing 9 DEVELOPMENT plus 6
  POST_REJECTION_DEVELOPMENT cases only.
- Independent calibration cases: retained only through the already-frozen H5
  uncertainty value; they are not loaded as policy outcomes and are not used
  to choose a conclusion.
- Held-out final test: not read.
- Future prospective cohort: not generated or run.
- Truth landscapes and candidate truth: attached only after each frozen H2
  decision/trajectory sequence is fixed. They do not feed model fitting,
  candidate ranking, authorization, stopping, or policy changes.
- This is software/offline synthetic evidence, not human, robot-motion, safety,
  comfort, or clinical evidence.
"""


def _report(
    *,
    manifest_sha: str,
    optimum: pd.DataFrame,
    discrimination: pd.DataFrame,
    axis_audit: pd.DataFrame,
    profiles: pd.DataFrame,
    timeline: pd.DataFrame,
    trial_values: pd.DataFrame,
    gain_timing: pd.DataFrame,
    objective: pd.DataFrame,
    boundary_classification: str,
    objective_status: str,
    trial_cost_status: str,
    final_status: str,
) -> str:
    truth_keys = optimum[
        [
            "truth_global_alpha_hip",
            "truth_global_alpha_knee",
            "truth_global_alpha_phase",
        ]
    ]
    truth_unique = len(truth_keys.drop_duplicates())
    unique_hip = truth_keys.iloc[:, 0].nunique()
    unique_knee = truth_keys.iloc[:, 1].nunique()
    unique_phase = truth_keys.iloc[:, 2].nunique()
    knee_minus5 = int(np.isclose(truth_keys.iloc[:, 1], -5.0).sum())
    exact_common = int(
        (
            np.isclose(truth_keys.iloc[:, 0], 0.0)
            & np.isclose(truth_keys.iloc[:, 1], -5.0)
            & np.isclose(truth_keys.iloc[:, 2], 0.0)
        ).sum()
    )
    pair_collapsed = int(
        discrimination["pair_classification"]
        .eq("SUBJECT_DIFFERENCE_COLLAPSED_BY_POLICY")
        .sum()
    )
    axis_summary = _axis_summary(axis_audit).set_index("axis_direction")
    knee = axis_summary.loc["KNEE_NEGATIVE"]
    selected = axis_audit.loc[axis_audit["selected"].astype(bool)]
    selected_agreement = float(
        selected["truth_direction_supports_improvement"].mean()
    )
    low_mask = trial_values["trial_value_classification"].isin(
        ("POST_OPTIMUM_LOW_VALUE", "BOUNDARY_CHASING_LOW_VALUE")
    )
    low_count = int(low_mask.sum())
    class_counts = trial_values["trial_value_classification"].value_counts()
    gain_medians = {
        target: float(gain_timing[f"trial_reaching_{target}pct_final_gain"].median())
        for target in (50, 80, 90, 95)
    }
    knee_profile = profiles.loc[
        profiles["axis"].eq("KNEE") & profiles["direction"].eq("NEGATIVE")
    ]
    truth_knee_boundary_cases = int(
        knee_profile.groupby("case_id")[
            "best_axis_direction_at_generator_boundary"
        ].first().sum()
    )
    h2_objective = objective.loc[objective["trajectory_role"].eq("H2_FINAL")]
    knee_component_cases = int(
        h2_objective["knee_component_reduction_exceeds_hip"].sum()
    )
    after_boundary = int(timeline["executed_actions_after_first_final_alpha"].sum())
    return f"""# {AUDIT_ID}

Manifest SHA-256: `{manifest_sha}`

## Plain-language findings

### A. Why did adaptive almost degenerate to BUNDLE_5?

The unchanged 1/2/3-step endpoint evidence almost never passed both frozen
decision gates. The same H5 uncertainty applies to every axis at H5, while
knee-negative had the strongest predicted endpoint improvement and remained
available. Therefore the adaptive sequence escalated to H5 for 59/60 endpoint
authorizations; this audit does not define a replacement rule.

### B. How diverse are the 15 truth optima?

There are `{truth_unique}` unique full alpha optima: hip has `{unique_hip}`
values, knee `{unique_knee}`, and phase `{unique_phase}`. Pairwise analysis
finds `{pair_collapsed}` pairs where truth optima differ but H2 final alpha is
the same.

### C. Is truth itself generally at knee=-5?

Yes: `{knee_minus5}/15` global truth optima have knee=-5, and the negative-knee
axis profile reaches its best value at that boundary in
`{truth_knee_boundary_cases}/15` cases. But only `{exact_common}/15` truth
optima equal the H2 common full alpha `(0,-5,0)`.

### D. Is the 15/15 common H2 alpha caused by truth or policy?

`{boundary_classification}`. Truth supplies a uniform knee-negative boundary
direction, while truth still contains hip/phase diversity that H2 does not
preserve.

### E. Why does knee-negative dominate H5 decisions?

KNEE_NEGATIVE was selected `{int(knee['selection_count'])}` times. Its mean
predicted improvement is `{float(knee['mean_predicted_improvement']):.9f}` and
availability is `{float(knee['nondecision_availability_rate']):.3f}`. All H5
directions use the same calibrated uncertainty, so uncertainty cannot explain
the axis preference. Selected direction agrees with posthoc truth direction in
`{selected_agreement:.3f}` of executions. Dominance is primarily predicted
magnitude plus the truth mechanical response, followed by deterministic
lowest-J ranking—not a favorable axis-specific uncertainty.

### F. When is the H2-over-H1 J gain obtained?

Median within-case trials to obtain 50/80/90/95% of final H2 gain are
`{gain_medians[50]:.1f}`, `{gain_medians[80]:.1f}`, `{gain_medians[90]:.1f}`,
and `{gain_medians[95]:.1f}` respectively. These are diagnostics, not stopping
thresholds.

### G. How many of 116 trials are an obvious low-decision-value tail?

`{low_count}/116`. Classification counts: {json.dumps(class_counts.to_dict(), ensure_ascii=False)}.
Final trial-cost interpretation: `{trial_cost_status}`.

### H. Are there many ineffective actions after reaching the boundary/final alpha?

Yes, in the matched-model cases: actions strictly after first arrival at final
alpha total `{after_boundary}`. They are exploration/refit actions that can
enlarge support or alter the model, but they do not change the best trajectory,
best J, or subsequent exploit eligibility in this frozen replay. They are
therefore classified posthoc as a low-decision-value tail, not used to define a
stopping rule.

### I. Does the objective lack all subject discrimination?

No. The full truth landscape contains `{truth_unique}` distinct full-alpha
optima, even though knee=-5 is common. At H2 final, knee torque contributes more
normalized squared-ratio reduction than hip in `{knee_component_cases}/15`
cases. The formula normalizes each joint to its own reference and weights the
two normalized RMS ratios equally; the common knee direction is therefore a
combined torque-response/generator-geometry effect, not evidence of an unequal
hard-coded knee weight.

### J. Is objective review scientifically justified now?

`{objective_status}` under the manifest-frozen review criterion. This task does
not change the objective.

### K. What should be studied next?

The evidence points first to **policy subject-discrimination/ranking** and
second to **trial-efficiency accounting**. It does not justify objective change
or further automatic P2 expansion. Any next study needs a new checkpoint and a
separate protocol; none is implemented here.

### L. Final status

`{final_status}`

## Evidence boundary

- DEVELOPMENT + POST_REJECTION_DEVELOPMENT only.
- Calibration cases provide frozen uncertainty only; no outcome selection.
- No held-out final test, future prospective cohort, human use, or robot motion.
- P2 V1, H1/H2/H3, reference, ROM, theta definition, five-parameter model,
  objective, generator/bounds, 0.005 tolerance, and 90% support gate unchanged.
- Final operational states remain `OFFLINE_ONLY`, `NOT_HUMAN_READY`, and
  `NOT_ROBOT_APPROVED`.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = _checkpoint_preflight()
    verify_immutable_prospective_artifacts()
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference SHA changed")
    if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
        raise RuntimeError("ROM protocol changed")
    if tuple(FORMAL_HIP_ROM_DEG) != (0.0, 120.0):
        raise RuntimeError("hip ROM changed")
    if tuple(FORMAL_KNEE_ROM_DEG) != (5.0, 145.0):
        raise RuntimeError("knee ROM changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    expected_bounds = {
        "hip_amplitude_delta_deg": (-5.0, 2.0),
        "knee_amplitude_delta_deg": (-5.0, 2.0),
        "knee_phase_shift": (-0.03, 0.03),
    }
    if OFFLINE_PERSONALIZATION_SEARCH_BOUNDS != expected_bounds:
        raise RuntimeError("generator bounds changed")

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protected_before = _source_hashes()
    manifest = manifest_payload(
        checkpoint_commit=str(checkpoint["checkpoint_commit"]),
        protected_source_sha256=protected_before,
    )
    manifest_path = output / "BUNDLE5_AUDIT_MANIFEST_V1.json"
    _write_json(manifest_path, manifest, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    gate = FrozenFrameworkManifestGate(manifest_path, manifest_sha)
    gate.require_frozen()

    cases = _case_table()
    if len(cases) != 15:
        raise RuntimeError("development case count changed")
    calibration_cases_loaded_as_outcomes = False
    lattice = geometrically_valid_parameter_lattice(pd.read_csv(parameter_map_path))
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometrically admissible lattice changed")
    cache = build_trajectory_component_cache(lattice)
    frozen_summary_all = pd.read_csv(
        MULTI_STEP_ARTIFACT_DIRECTORY / "framework_case_summary.csv"
    )
    h2_summary = frozen_summary_all.loc[
        frozen_summary_all["framework_id"].eq("BUNDLE_5")
    ].copy()
    h1_summary = frozen_summary_all.loc[
        frozen_summary_all["framework_id"].eq("SINGLE_STEP")
    ].copy()
    frozen_history_all = pd.read_csv(
        MULTI_STEP_ARTIFACT_DIRECTORY / "framework_trial_history.csv"
    )
    h2_history = frozen_history_all.loc[
        frozen_history_all["framework_id"].eq("BUNDLE_5")
    ].copy()
    if len(h2_history) != 116 or len(h2_summary) != 15:
        raise RuntimeError("frozen H2 evidence changed")

    trial_frames: list[pd.DataFrame] = []
    axis_frames: list[pd.DataFrame] = []
    objective_frames: list[pd.DataFrame] = []
    truth_landscapes: dict[str, pd.DataFrame] = {}
    patient_cache: dict[tuple[float, float, float], bool] = {}
    summary_lookup = h2_summary.set_index("case_id")
    for case in cases.to_dict(orient="records"):
        case_id = str(case["case_id"])

        def run_case() -> None:
            replay = _replay_case(
                case=case,
                frozen_history=h2_history.loc[h2_history["case_id"].eq(case_id)],
                h2_summary=summary_lookup.loc[case_id],
                lattice=lattice,
                cache=cache,
                gate=gate,
                patient_cache=patient_cache,
            )
            trial_frames.append(replay.trial_diagnostics)
            axis_frames.append(replay.axis_decisions)
            truth_landscapes[case_id] = replay.truth_landscape
            objective_frames.append(replay.objective_decomposition)

        if case["development_origin"] == "POST_REJECTION_DEVELOPMENT":
            subject = dynamic_subject_for_id(str(case["subject_id"]))
            with registered_prospective_subject(subject):
                run_case()
        else:
            run_case()

    diagnostics = pd.concat(trial_frames, ignore_index=True, sort=False)
    axis_audit = pd.concat(axis_frames, ignore_index=True, sort=False)
    objective = pd.concat(objective_frames, ignore_index=True, sort=False)
    if len(diagnostics) != 116 or not diagnostics["frozen_history_row_verified"].all():
        raise RuntimeError("H2 replay did not verify all 116 frozen rows")
    optimum = build_truth_optimum_by_case(truth_landscapes, h2_summary)
    discrimination = build_subject_discrimination(optimum)
    profiles = build_truth_axis_profiles(truth_landscapes)
    boundary_classification = classify_boundary_collapse(optimum)
    objective_status = classify_objective_status(optimum, profiles)
    trial_values = classify_trial_values(diagnostics, h2_summary)
    timeline = build_boundary_timeline(trial_values, h2_summary)
    progression = build_best_j_progression(trial_values)
    gain_timing = build_gain_timing(progression, h1_summary, h2_summary)
    trial_cost_status = classify_trial_cost(trial_values)
    final_status = (
        FINAL_IDENTIFIED
        if len(optimum) == 15
        and len(discrimination) == 105
        and len(trial_values) == 116
        and boundary_classification in (
            "TRUTH_LANDSCAPE_CONCENTRATION",
            "POLICY_INDUCED_COLLAPSE",
            "MIXED_TRUTH_AND_POLICY_EFFECT",
        )
        else FINAL_MORE_EVIDENCE
    )
    root_matrix = _root_matrix(
        optimum=optimum,
        axis_audit=axis_audit,
        timeline=timeline,
        trial_values=trial_values,
        boundary_classification=boundary_classification,
        objective_status=objective_status,
        trial_cost_status=trial_cost_status,
    )

    _write_csv(output / "bundle5_truth_optimum_by_case.csv", optimum)
    _write_csv(output / "bundle5_subject_discrimination.csv", discrimination)
    _write_csv(output / "bundle5_axis_direction_decision_audit.csv", axis_audit)
    _write_csv(output / "bundle5_truth_axis_profiles.csv", profiles)
    _write_csv(output / "bundle5_boundary_arrival_timeline.csv", timeline)
    _write_csv(output / "bundle5_best_J_progression.csv", progression)
    _write_csv(output / "bundle5_trial_value_audit.csv", trial_values)
    _write_csv(output / "bundle5_objective_decomposition.csv", objective)
    _write_csv(output / "bundle5_performance_gain_timing.csv", gain_timing)
    _write_csv(output / "BUNDLE5_BOUNDARY_ROOT_CAUSE_MATRIX.csv", root_matrix)
    _write_text(output / "DATA_ROLE_AUDIT.md", _data_role_audit(checkpoint))
    _write_text(
        output / "BUNDLE5_BOUNDARY_SUBJECT_SPECIFICITY_REPORT.md",
        _report(
            manifest_sha=manifest_sha,
            optimum=optimum,
            discrimination=discrimination,
            axis_audit=axis_audit,
            profiles=profiles,
            timeline=timeline,
            trial_values=trial_values,
            gain_timing=gain_timing,
            objective=objective,
            boundary_classification=boundary_classification,
            objective_status=objective_status,
            trial_cost_status=trial_cost_status,
            final_status=final_status,
        ),
    )

    protected_after = _source_hashes()
    if protected_before != protected_after:
        raise RuntimeError("protected source changed during BUNDLE_5 audit")
    if sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("BUNDLE_5 audit manifest changed after replay")
    if calibration_cases_loaded_as_outcomes:
        raise RuntimeError("calibration cases entered policy outcome audit")
    if diagnostics["truth_used_to_select_or_stop"].astype(bool).any():
        raise RuntimeError("posthoc truth entered H2 replay decisions")

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "metadata.json":
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    truth_columns = [
        "truth_global_alpha_hip",
        "truth_global_alpha_knee",
        "truth_global_alpha_phase",
    ]
    trial_counts = trial_values["trial_value_classification"].value_counts().to_dict()
    low_count = int(
        trial_values["trial_value_classification"].isin(
            ("POST_OPTIMUM_LOW_VALUE", "BOUNDARY_CHASING_LOW_VALUE")
        ).sum()
    )
    metadata = {
        "audit_id": AUDIT_ID,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "checkpoint": checkpoint,
        "adaptive_manifest_sha256": ADAPTIVE_MANIFEST_SHA256,
        "final_status": final_status,
        "boundary_collapse_classification": boundary_classification,
        "objective_status": objective_status,
        "trial_cost_status": trial_cost_status,
        "truth_full_alpha_unique_count": len(optimum[truth_columns].drop_duplicates()),
        "truth_hip_unique_count": optimum[truth_columns[0]].nunique(),
        "truth_knee_unique_count": optimum[truth_columns[1]].nunique(),
        "truth_phase_unique_count": optimum[truth_columns[2]].nunique(),
        "truth_knee_minus_5_count": int(
            np.isclose(optimum[truth_columns[1]], -5.0).sum()
        ),
        "truth_exact_H2_common_alpha_count": int(
            (
                np.isclose(optimum[truth_columns[0]], 0.0)
                & np.isclose(optimum[truth_columns[1]], -5.0)
                & np.isclose(optimum[truth_columns[2]], 0.0)
            ).sum()
        ),
        "trial_value_classification_counts": trial_counts,
        "obvious_low_decision_value_tail_count": low_count,
        "post_final_alpha_execution_count": int(
            timeline["executed_actions_after_first_final_alpha"].sum()
        ),
        "frozen_H2_trial_count": len(diagnostics),
        "frozen_H2_rows_verified": True,
        "calibration_cases_used_as_policy_outcomes": False,
        "future_prospective_generated": False,
        "held_out_final_test_read": False,
        "truth_profiles_used_for_policy": False,
        "P2_V1_modified": False,
        "objective_modified": False,
        "generator_modified": False,
        "generator_bounds_modified": False,
        "five_parameter_model_modified": False,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "robot_connected": False,
        "new_policy_implemented": False,
        "default_enabled": False,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "artifact_manifest": artifacts,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "metadata.json", metadata)
    missing = [name for name in REQUIRED_OUTPUT_FILENAMES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing BUNDLE_5 audit artifacts: {missing}")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the post-hoc frozen BUNDLE_5 boundary audit."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_dir, arguments.parameter_map)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
