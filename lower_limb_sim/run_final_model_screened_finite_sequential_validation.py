"""Run the final default-off finite sequential-validation offline study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    TrajectoryComponentCache,
    build_predicted_map,
    build_trajectory_component_cache,
    evaluate_truth_map,
    geometrically_valid_parameter_lattice,
)
from .final_model_screened_finite_sequential_validation import (
    BEST_VALIDATED_TRAJECTORY,
    DEFAULT_ENABLED,
    FINAL_STATUS_RULE,
    MANIFEST_ID,
    MAX_MODEL_SCREENED_CANDIDATES,
    MAX_VALIDATION_TRIALS,
    METHOD_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_APPROVED,
    OFFLINE_ONLY,
    FrozenShortlist,
    FrozenShortlistTruthGate,
    assert_complete_candidate_trajectory,
    canonical_json_bytes,
    freeze_model_screened_shortlist,
    method_manifest_payload,
    rerank_remaining_frozen_candidates,
    select_best_validated,
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
from .generate_final_method_animation import (
    FIGURE_NAMES,
    GIF_NAMES,
    generate_all_visuals,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_decision_rule_semantics_audit import sha256_file
from .p2_v2_prospective_offline_validation import (
    dynamic_subject_for_id,
    registered_prospective_subject,
)
from .post_prospective_rejection_root_cause_audit import (
    verify_immutable_prospective_artifacts,
)
from .research_decision_guarded_sequential_personalization import (
    _actual_objective,
    _fit_updated_model,
    _model_for_iteration,
    build_initial_research_state,
)
from .run_p2_multi_step_decision_framework_analysis import _case_table
from .run_p2_v2_prospective_offline_validation import _protected_source_hashes
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .sequential_personalization import Stage45CVirtualTruthOracle


FINAL_SUPPORTED = "FINAL_SIMPLIFIED_METHOD_SUPPORTED"
FINAL_LIMITED = "FINAL_SIMPLIFIED_METHOD_SUPPORTED_WITH_LIMITATIONS"
FINAL_NOT_SUPPORTED = "FINAL_SIMPLIFIED_METHOD_NOT_SUPPORTED"

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "final_model_screened_finite_sequential_validation.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_final_model_screened_finite_sequential_validation.py"
ANIMATION_SOURCE_PATH = MODULE_DIR / "generate_final_method_animation.py"
TEST_SOURCE_PATH = MODULE_DIR / "test_final_model_screened_finite_sequential_validation.py"
PREVIOUS_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_bundle5_boundary_subject_specificity_audit_v1"
)
PREVIOUS_MANIFEST_PATH = PREVIOUS_DIRECTORY / "BUNDLE5_AUDIT_MANIFEST_V1.json"
PREVIOUS_MANIFEST_SHA256 = (
    "b959444e8df39a05693f873aaa3060cb5c21a4525d7f1bbda9c81aa96f1762c8"
)
ADAPTIVE_COMPARISON_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_adaptive_horizon_decision_prototype_v1"
    / "adaptive_vs_fixed_comparison.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "final_model_screened_finite_sequential_validation_v1"
)
REQUIRED_TABLES = (
    "candidate_shortlist_manifest.csv",
    "candidate_execution_history.csv",
    "model_update_history.csv",
    "final_subject_summary.csv",
    "predicted_vs_truth_optimum.csv",
    "budget_sensitivity.csv",
    "matched_mismatch_summary.csv",
    "subject_specificity_summary.csv",
    "false_improvement_audit.csv",
    "p2_vs_finite_method_comparison.csv",
)
REQUIRED_DOCUMENTS = (
    "FINAL_METHOD_MANIFEST_V1.json",
    "FINAL_METHOD_VALIDATION_REPORT.md",
    "DATA_ROLE_AUDIT.md",
    "VISUALIZATION_GUIDE.md",
    "metadata.json",
)
PREVIOUS_REQUIRED = (
    MODULE_DIR / "p2_bundle5_boundary_subject_specificity_audit.py",
    MODULE_DIR / "run_p2_bundle5_boundary_subject_specificity_audit.py",
    MODULE_DIR / "test_p2_bundle5_boundary_subject_specificity_audit.py",
    PREVIOUS_MANIFEST_PATH,
    PREVIOUS_DIRECTORY / "metadata.json",
    PREVIOUS_DIRECTORY / "BUNDLE5_BOUNDARY_SUBJECT_SPECIFICITY_REPORT.md",
)


@dataclass
class PreparedCase:
    role: dict[str, Any]
    state: Any
    initial_model: Any
    initial_prediction_map: pd.DataFrame
    shortlist: FrozenShortlist


@dataclass
class CaseEvaluation:
    execution_history: pd.DataFrame
    model_history: pd.DataFrame
    summary: dict[str, Any]
    optimum: dict[str, Any]
    false_improvements: pd.DataFrame
    truth_map: pd.DataFrame


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
    if sha256_file(PREVIOUS_MANIFEST_PATH) != PREVIOUS_MANIFEST_SHA256:
        raise RuntimeError("FINAL_SIMPLIFIED_METHOD_REQUIRES_CHECKPOINT")
    head = _git_output("rev-parse", "HEAD")
    previous_commit = _git_output(
        "log",
        "-1",
        "--format=%H",
        "--",
        str(PREVIOUS_REQUIRED[0].relative_to(PROJECT_ROOT)),
    )
    if previous_commit != head:
        raise RuntimeError("FINAL_SIMPLIFIED_METHOD_REQUIRES_CHECKPOINT")
    verified: dict[str, str] = {}
    for path in PREVIOUS_REQUIRED:
        relative = str(path.relative_to(PROJECT_ROOT))
        try:
            _git_output("ls-files", "--error-unmatch", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("FINAL_SIMPLIFIED_METHOD_REQUIRES_CHECKPOINT") from exc
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        # The prerequisite is the existence of an immutable scientific
        # checkpoint.  Later maintenance of an audit runner/test must not make
        # that historical checkpoint disappear; current source hashes are
        # recorded separately by _source_hashes().
        verified[relative] = hashlib.sha256(committed).hexdigest()
    return {
        "checkpoint_commit": head,
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "previous_stage_commit": previous_commit,
        "previous_stage_is_independent_current_HEAD": True,
        "previous_manifest_sha256": PREVIOUS_MANIFEST_SHA256,
        "verified_previous_files": verified,
    }


def _source_hashes() -> dict[str, str]:
    hashes = _protected_source_hashes()
    hashes.update(
        {
            "finite_method_core": sha256_file(CORE_SOURCE_PATH),
            "finite_method_runner": sha256_file(RUNNER_SOURCE_PATH),
            "finite_method_animation": sha256_file(ANIMATION_SOURCE_PATH),
            "previous_bundle5_manifest": sha256_file(PREVIOUS_MANIFEST_PATH),
            "previous_adaptive_comparison": sha256_file(ADAPTIVE_COMPARISON_PATH),
        }
    )
    return hashes


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any], *, canonical: bool = False) -> None:
    data = (
        canonical_json_bytes(_json_safe(dict(payload)))
        if canonical
        else (
            json.dumps(
                _json_safe(dict(payload)),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    _atomic_bytes(path, data)


def _write_text(path: Path, content: str) -> None:
    _atomic_bytes(path, content.encode("utf-8"))


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    _atomic_bytes(
        path,
        table.to_csv(
            index=False, lineterminator="\n", float_format="%.12g"
        ).encode("utf-8"),
    )


def _parameters_json(parameters: Mapping[str, float]) -> str:
    return json.dumps(
        {key: float(parameters[key]) for key in sorted(parameters)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _alpha_from_row(row: Mapping[str, Any], prefix: str = "") -> tuple[float, float, float]:
    return (
        float(row[f"{prefix}hip_delta"]),
        float(row[f"{prefix}knee_delta"]),
        float(row[f"{prefix}phase_delta"]),
    )


def _alpha_id(alpha: tuple[float, float, float]) -> str:
    generated = generate_personalized_trajectory(
        hip_amplitude_delta_deg=alpha[0],
        knee_amplitude_delta_deg=alpha[1],
        knee_phase_shift=alpha[2],
    )
    return str(generated.metadata["trajectory_id"])


def _prepare_cases(
    cases: pd.DataFrame,
    lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
) -> list[PreparedCase]:
    prepared: list[PreparedCase] = []
    for role in cases.to_dict(orient="records"):
        def prepare_one() -> None:
            state = build_initial_research_state(
                str(role["subject_id"]), str(role["scenario_name"])
            )
            model = _model_for_iteration(
                state, dict(state.parameters), state.domain_data.copy(deep=True), 0
            )
            prediction, prediction_metadata = build_predicted_map(
                model, lattice, cache
            )
            if prediction_metadata["truth_evaluated_during_prediction"]:
                raise RuntimeError(
                    "candidate prediction accessed truth before shortlist freeze"
                )
            shortlist = freeze_model_screened_shortlist(
                prediction, case_id=str(role["case_id"])
            )
            prepared.append(
                PreparedCase(
                    role=dict(role),
                    state=state,
                    initial_model=model,
                    initial_prediction_map=prediction,
                    shortlist=shortlist,
                )
            )

        if role["development_origin"] == "POST_REJECTION_DEVELOPMENT":
            subject = dynamic_subject_for_id(str(role["subject_id"]))
            with registered_prospective_subject(subject):
                prepare_one()
        else:
            prepare_one()
    return prepared


def _shortlist_table(
    prepared: Sequence[PreparedCase], manifest_sha256: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in prepared:
        for candidate in item.shortlist.candidates:
            rows.append(
                {
                    **item.role,
                    "shortlist_freeze_token": item.shortlist.freeze_token,
                    "global_manifest_sha256": manifest_sha256,
                    **candidate.as_dict(),
                    "selection_rule": item.shortlist.selection_rule,
                    "truth_read_before_freeze": False,
                    "support_role": "DATA_PROVENANCE_AND_APPLICABILITY_NOT_RELIABILITY_SCORE",
                    "candidate_addition_after_freeze_allowed": False,
                    "candidate_truth_used_for_shortlist": False,
                }
            )
    return pd.DataFrame(rows)


def _reference_execution(backend: Stage45CVirtualTruthOracle) -> tuple[Any, str]:
    generated = generate_personalized_trajectory()
    trajectory = generated.trajectory.copy(deep=True)
    trajectory_id = str(generated.metadata["trajectory_id"])
    trajectory["trajectory_id"] = trajectory_id
    assert_complete_candidate_trajectory(
        trajectory, expected_trajectory_id=trajectory_id
    )
    return backend.simulate(trajectory), trajectory_id


def _evaluate_case(
    prepared: PreparedCase,
    *,
    cache: TrajectoryComponentCache,
    global_manifest_sha256: str,
) -> CaseEvaluation:
    case_id = str(prepared.role["case_id"])
    state = prepared.state
    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    prediction_map = prepared.initial_prediction_map.copy(deep=True)
    initial_map = prediction_map.copy(deep=True)
    backend = Stage45CVirtualTruthOracle(
        str(prepared.role["subject_id"]), str(prepared.role["scenario_name"])
    )
    gate = FrozenShortlistTruthGate(
        prepared.shortlist,
        global_manifest_sha256=global_manifest_sha256,
        manifest_persisted=True,
    )
    reference_execution, reference_id = _reference_execution(backend)
    reference_metrics = reference_execution.actual_metrics
    executed_ids: list[str] = []
    validated_rows: list[dict[str, Any]] = [
        {
            "trajectory_id": reference_id,
            "validated_J": 1.0,
            "validation_role": "REFERENCE_BASELINE",
            "hip_delta": 0.0,
            "knee_delta": 0.0,
            "phase_delta": 0.0,
        }
    ]
    execution_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = [
        {
            **prepared.role,
            "model_iteration": 0,
            "after_validation_round": 0,
            "theta_hat_json": _parameters_json(parameters),
            **{f"theta_hat_{key}": float(value) for key, value in parameters.items()},
            "fitting_sample_count": len(fitting_data),
            "domain_sample_count": len(domain_data),
            "full_landscape_recomputed": True,
            "remaining_frozen_candidates_json": json.dumps(
                list(prepared.shortlist.trajectory_ids), separators=(",", ":")
            ),
            "diagnostic_full_predicted_optimum_id": str(
                prediction_map.loc[
                    prediction_map["model_supported"].astype(bool)
                ].sort_values(["J_pred", "trajectory_id"], kind="mergesort").iloc[0][
                    "trajectory_id"
                ]
            ),
            "diagnostic_new_optimum_added_to_shortlist": False,
        }
    ]
    candidate_lookup = {
        candidate.trajectory_id: candidate for candidate in prepared.shortlist.candidates
    }
    best_j = 1.0
    best_id = reference_id
    best_alpha = (0.0, 0.0, 0.0)
    for round_index in range(1, min(MAX_VALIDATION_TRIALS, len(candidate_lookup)) + 1):
        ranking = rerank_remaining_frozen_candidates(
            prepared.shortlist,
            prediction_map,
            executed_trajectory_ids=executed_ids,
        )
        selected = ranking.iloc[0]
        trajectory_id = str(selected["trajectory_id"])
        frozen = candidate_lookup[trajectory_id]
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=frozen.hip_delta,
            knee_amplitude_delta_deg=frozen.knee_delta,
            knee_phase_shift=frozen.phase_delta,
        )
        if str(generated.metadata["trajectory_sha256"]) != frozen.trajectory_sha256:
            raise RuntimeError("frozen candidate trajectory SHA changed")
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = trajectory_id
        assert_complete_candidate_trajectory(
            trajectory, expected_trajectory_id=trajectory_id
        )
        truth_token = gate.authorize(trajectory_id)
        truth_calls_before = backend.truth_calls
        execution = backend.simulate(trajectory)
        if backend.truth_calls != truth_calls_before + 1:
            raise RuntimeError("one complete candidate must produce one truth call")
        gate.complete(trajectory_id, truth_token)
        actual = _actual_objective(trajectory_id, execution, reference_metrics)
        actual_j = float(actual.mechanical_cost_j_rms)
        executed_ids.append(trajectory_id)
        validated_rows.append(
            {
                "trajectory_id": trajectory_id,
                "validated_J": actual_j,
                "validation_role": "COMPLETE_CANDIDATE_VIRTUAL_VALIDATION",
                "hip_delta": frozen.hip_delta,
                "knee_delta": frozen.knee_delta,
                "phase_delta": frozen.phase_delta,
            }
        )
        if actual_j < best_j:
            best_j = actual_j
            best_id = trajectory_id
            best_alpha = frozen.alpha_key
        parameters_before = dict(parameters)
        fitting_data = pd.concat(
            (fitting_data, execution.estimator_observations), ignore_index=True
        )
        domain_data = pd.concat(
            (domain_data, execution.estimator_observations), ignore_index=True
        )
        estimation = _fit_updated_model(fitting_data, parameters)
        if not estimation.optimizer_success:
            raise RuntimeError(f"finite validation refit failed: {case_id}/{round_index}")
        parameters = dict(estimation.estimated_parameters)
        model = _model_for_iteration(state, parameters, domain_data, round_index)
        fixed_lattice = prepared.initial_prediction_map.loc[
            :,
            [
                "trajectory_id",
                "hip_delta",
                "knee_delta",
                "phase_delta",
                "parent_reference_sha256",
                "geometrically_admissible",
            ],
        ].copy()
        prediction_map, map_metadata = build_predicted_map(
            model, fixed_lattice, cache
        )
        if map_metadata["truth_evaluated_during_prediction"]:
            raise RuntimeError("post-validation full prediction map accessed truth")
        diagnostic = prediction_map.loc[
            prediction_map["model_supported"].astype(bool)
        ].sort_values(["J_pred", "trajectory_id"], kind="mergesort").iloc[0]
        remaining = rerank_remaining_frozen_candidates(
            prepared.shortlist,
            prediction_map,
            executed_trajectory_ids=executed_ids,
        )
        execution_rows.append(
            {
                **prepared.role,
                "round": round_index,
                "candidate_id": f"C{frozen.shortlist_ordinal}",
                "trajectory_id": trajectory_id,
                "shortlist_freeze_token": prepared.shortlist.freeze_token,
                "truth_authorization_token": truth_token,
                "whole_trajectory_execution": True,
                "trajectory_sample_count": len(trajectory),
                "trajectory_duration_s": float(trajectory["time_s"].iloc[-1] - trajectory["time_s"].iloc[0]),
                "trajectory_sha256": frozen.trajectory_sha256,
                "hip_delta": frozen.hip_delta,
                "knee_delta": frozen.knee_delta,
                "phase_delta": frozen.phase_delta,
                "current_frozen_rank": int(selected["current_frozen_rank"]),
                "J_pred_before_execution": float(selected["J_pred"]),
                "domain_coverage_before_execution": float(selected["domain_coverage"]),
                "model_supported_before_execution": bool(selected["model_supported"]),
                "actual_J": actual_j,
                "predicted_improvement": bool(float(selected["J_pred"]) < 1.0),
                "truth_improvement": bool(actual_j < 1.0),
                "false_improvement": bool(float(selected["J_pred"]) < 1.0 and actual_j > 1.0),
                "best_validated_trajectory_id_after": best_id,
                "best_validated_J_after": best_j,
                "theta_refit_after_execution": True,
                "full_landscape_recomputed_after_execution": True,
                "remaining_frozen_candidates_after": ";".join(remaining["trajectory_id"].astype(str)),
                "diagnostic_full_predicted_optimum_after": str(diagnostic["trajectory_id"]),
                "diagnostic_optimum_is_frozen_candidate": str(diagnostic["trajectory_id"]) in set(prepared.shortlist.trajectory_ids),
                "diagnostic_optimum_added_to_shortlist": False,
                "new_candidate_execution_allowed": False,
                "truth_used_for_shortlist_or_ranking": False,
            }
        )
        parameter_delta = {
            key: float(parameters[key]) - float(parameters_before[key])
            for key in sorted(parameters)
        }
        model_rows.append(
            {
                **prepared.role,
                "model_iteration": round_index,
                "after_validation_round": round_index,
                "theta_hat_json": _parameters_json(parameters),
                **{f"theta_hat_{key}": float(value) for key, value in parameters.items()},
                "parameter_delta_json": json.dumps(parameter_delta, sort_keys=True, separators=(",", ":")),
                "fitting_sample_count": len(fitting_data),
                "domain_sample_count": len(domain_data),
                "full_landscape_recomputed": True,
                "remaining_frozen_candidates_json": json.dumps(
                    remaining["trajectory_id"].astype(str).tolist(), separators=(",", ":")
                ),
                "remaining_frozen_J_pred_json": json.dumps(
                    [float(value) for value in remaining["J_pred"]], separators=(",", ":")
                ),
                "diagnostic_full_predicted_optimum_id": str(diagnostic["trajectory_id"]),
                "diagnostic_new_optimum_added_to_shortlist": False,
            }
        )

    validated = pd.DataFrame(validated_rows)
    final = select_best_validated(validated.drop(columns=[], errors="ignore"))
    truth_map, truth_metadata = evaluate_truth_map(
        initial_map, prepared.initial_model, cache
    )
    if truth_metadata["truth_used_for_pre_evaluation_ranking"]:
        raise RuntimeError("truth was used by prediction ranking")
    truth_ranked = truth_map.sort_values(
        ["J_truth", "trajectory_id"], kind="mergesort"
    )
    truth_optimum = truth_ranked.iloc[0]
    initial_predicted = initial_map.loc[
        initial_map["model_supported"].astype(bool)
        & ~(
            np.isclose(initial_map["hip_delta"], 0.0)
            & np.isclose(initial_map["knee_delta"], 0.0)
            & np.isclose(initial_map["phase_delta"], 0.0)
        )
    ].sort_values(["J_pred", "trajectory_id"], kind="mergesort").iloc[0]
    truth_lookup = truth_map.set_index("trajectory_id")
    for row in execution_rows:
        expected = float(truth_lookup.loc[str(row["trajectory_id"]), "J_truth"])
        if not np.isclose(float(row["actual_J"]), expected, atol=1e-11, rtol=0.0):
            raise RuntimeError("candidate virtual execution differs from truth map")
    shortlist_truth = truth_map.loc[
        truth_map["trajectory_id"].astype(str).isin(prepared.shortlist.trajectory_ids)
    ].sort_values(["J_truth", "trajectory_id"], kind="mergesort")
    shortlist_best = shortlist_truth.iloc[0]
    execution_frame = pd.DataFrame(execution_rows)
    b1_best = min(1.0, float(execution_frame.iloc[0]["actual_J"])) if not execution_frame.empty else 1.0
    final_alpha = (
        float(final["hip_delta"]),
        float(final["knee_delta"]),
        float(final["phase_delta"]),
    )
    initial_truth = float(
        truth_lookup.loc[str(initial_predicted["trajectory_id"]), "J_truth"]
    )
    summary = {
        **prepared.role,
        "initial_identification_trial_count": int(state.selected_trial_id),
        "initial_theta_hat_json": _parameters_json(state.parameters),
        "frozen_candidate_count": len(prepared.shortlist.candidates),
        "executed_validation_count": len(execution_frame),
        "B0_reference_J": 1.0,
        "B1_predicted_best_only_J": b1_best,
        "B1_false_improvement_count": int(
            execution_frame.head(1)["false_improvement"].sum()
        ),
        "B2_final_best_validated_J": float(final["validated_J"]),
        "B3_truth_global_optimum_J": float(truth_optimum["J_truth"]),
        "reference_improvement": 1.0 - float(final["validated_J"]),
        "global_regret": float(final["validated_J"]) - float(truth_optimum["J_truth"]),
        "false_improvement_count": int(execution_frame["false_improvement"].sum()),
        "shortlist_hit_truth_global_optimum": str(truth_optimum["trajectory_id"]) in set(prepared.shortlist.trajectory_ids),
        "shortlist_best_truth_J": float(shortlist_best["J_truth"]),
        "shortlist_near_optimum_regret": float(shortlist_best["J_truth"] - truth_optimum["J_truth"]),
        "initial_predicted_optimum_trajectory_id": str(initial_predicted["trajectory_id"]),
        "initial_predicted_optimum_alpha_hip": float(initial_predicted["hip_delta"]),
        "initial_predicted_optimum_alpha_knee": float(initial_predicted["knee_delta"]),
        "initial_predicted_optimum_alpha_phase": float(initial_predicted["phase_delta"]),
        "truth_global_optimum_trajectory_id": str(truth_optimum["trajectory_id"]),
        "truth_global_alpha_hip": float(truth_optimum["hip_delta"]),
        "truth_global_alpha_knee": float(truth_optimum["knee_delta"]),
        "truth_global_alpha_phase": float(truth_optimum["phase_delta"]),
        "best_validated_trajectory_id": str(final["trajectory_id"]),
        "best_validated_role": BEST_VALIDATED_TRAJECTORY,
        "best_validated_alpha_hip": final_alpha[0],
        "best_validated_alpha_knee": final_alpha[1],
        "best_validated_alpha_phase": final_alpha[2],
        "final_theta_hat_json": _parameters_json(parameters),
        "truth_used_for_shortlist": False,
        "truth_used_for_reranking": False,
        "held_out_final_test_read": False,
    }
    optimum_row = {
        **prepared.role,
        "initial_predicted_optimum_trajectory_id": str(initial_predicted["trajectory_id"]),
        "initial_predicted_optimum_alpha_hip": float(initial_predicted["hip_delta"]),
        "initial_predicted_optimum_alpha_knee": float(initial_predicted["knee_delta"]),
        "initial_predicted_optimum_alpha_phase": float(initial_predicted["phase_delta"]),
        "initial_predicted_optimum_J_pred": float(initial_predicted["J_pred"]),
        "truth_J_at_initial_predicted_optimum": initial_truth,
        "initial_optimum_absolute_J_error": abs(float(initial_predicted["J_pred"]) - initial_truth),
        "truth_global_optimum_trajectory_id": str(truth_optimum["trajectory_id"]),
        "truth_global_alpha_hip": float(truth_optimum["hip_delta"]),
        "truth_global_alpha_knee": float(truth_optimum["knee_delta"]),
        "truth_global_alpha_phase": float(truth_optimum["phase_delta"]),
        "truth_global_optimum_J": float(truth_optimum["J_truth"]),
        "shortlist_hit": str(truth_optimum["trajectory_id"]) in set(prepared.shortlist.trajectory_ids),
        "shortlist_best_truth_J": float(shortlist_best["J_truth"]),
        "shortlist_regret": float(shortlist_best["J_truth"] - truth_optimum["J_truth"]),
        "candidate_truth_read_after_freeze_only": True,
        "truth_used_for_policy": False,
    }
    false_frame = execution_frame.loc[
        execution_frame["false_improvement"].astype(bool)
    ].copy()
    if false_frame.empty:
        false_frame = execution_frame.head(0).copy()
    return CaseEvaluation(
        execution_history=execution_frame,
        model_history=pd.DataFrame(model_rows),
        summary=summary,
        optimum=optimum_row,
        false_improvements=false_frame,
        truth_map=truth_map,
    )


def _budget_table(
    summaries: pd.DataFrame,
    execution: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries.to_dict(orient="records"):
        case_id = str(summary["case_id"])
        history = execution.loc[execution["case_id"].eq(case_id)].sort_values("round")
        optimum = float(summary["B3_truth_global_optimum_J"])
        for budget in range(0, MAX_VALIDATION_TRIALS + 1):
            used = history.head(budget)
            best = min([1.0, *used["actual_J"].astype(float).tolist()])
            rows.append(
                {
                    "case_id": case_id,
                    "case_class": summary["case_class"],
                    "budget": budget,
                    "executed_candidate_count": len(used),
                    "final_best_validated_J": best,
                    "reference_improvement": 1.0 - best,
                    "global_regret": best - optimum,
                    "false_improvement_count": int(used["false_improvement"].sum()),
                    "same_frozen_shortlist_all_budgets": True,
                    "used_to_change_registered_budget": False,
                }
            )
    return pd.DataFrame(rows)


def _pairwise_preservation(
    truth: pd.DataFrame,
    representation: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[int, int, float]:
    truth_lookup = truth.set_index("case_id")
    rep_lookup = representation.set_index("case_id")
    differing = 0
    preserved = 0
    case_ids = sorted(set(truth_lookup.index).intersection(rep_lookup.index))
    for left_index, left in enumerate(case_ids):
        for right in case_ids[left_index + 1 :]:
            truth_left = tuple(truth_lookup.loc[left, ["truth_global_alpha_hip", "truth_global_alpha_knee", "truth_global_alpha_phase"]])
            truth_right = tuple(truth_lookup.loc[right, ["truth_global_alpha_hip", "truth_global_alpha_knee", "truth_global_alpha_phase"]])
            if truth_left == truth_right:
                continue
            differing += 1
            if tuple(rep_lookup.loc[left, list(columns)]) != tuple(rep_lookup.loc[right, list(columns)]):
                preserved += 1
    return differing, preserved, float(preserved / differing) if differing else float("nan")


def _subject_specificity(
    summaries: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    truth = summaries.copy()
    representations: list[tuple[str, pd.DataFrame, tuple[str, str, str]]] = [
        (
            "TRUTH_GLOBAL_OPTIMUM",
            summaries,
            ("truth_global_alpha_hip", "truth_global_alpha_knee", "truth_global_alpha_phase"),
        ),
        (
            "INITIAL_PREDICTED_OPTIMUM",
            summaries,
            ("initial_predicted_optimum_alpha_hip", "initial_predicted_optimum_alpha_knee", "initial_predicted_optimum_alpha_phase"),
        ),
        (
            "BEST_VALIDATED_TRAJECTORY",
            summaries,
            ("best_validated_alpha_hip", "best_validated_alpha_knee", "best_validated_alpha_phase"),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, table, columns in representations:
        values = table.loc[:, list(columns)].copy()
        differing, preserved, rate = _pairwise_preservation(truth, table, columns)
        rows.append(
            {
                "representation": name,
                "unique_full_alpha_count": len(values.drop_duplicates()),
                "unique_hip_count": values[columns[0]].nunique(),
                "unique_knee_count": values[columns[1]].nunique(),
                "unique_phase_count": values[columns[2]].nunique(),
                "truth_different_pair_count": differing,
                "truth_difference_preserved_pair_count": preserved,
                "pairwise_subject_difference_preservation_rate": rate,
                "truth_used_to_create_difference": False,
            }
        )
    old = summaries[["case_id"]].copy()
    old["hip"] = 0.0
    old["knee"] = -5.0
    old["phase"] = 0.0
    differing, preserved, rate = _pairwise_preservation(
        truth, old, ("hip", "knee", "phase")
    )
    rows.append(
        {
            "representation": "OLD_BUNDLE5_FINAL",
            "unique_full_alpha_count": 1,
            "unique_hip_count": 1,
            "unique_knee_count": 1,
            "unique_phase_count": 1,
            "truth_different_pair_count": differing,
            "truth_difference_preserved_pair_count": preserved,
            "pairwise_subject_difference_preservation_rate": rate,
            "truth_used_to_create_difference": False,
        }
    )
    signatures = (
        shortlist.sort_values(["case_id", "shortlist_ordinal"])
        .groupby("case_id")
        .apply(lambda group: tuple(group["trajectory_id"].astype(str)), include_groups=False)
    )
    rows.append(
        {
            "representation": "FROZEN_SHORTLIST_SIGNATURE",
            "unique_full_alpha_count": signatures.nunique(),
            "unique_hip_count": shortlist["hip_delta"].nunique(),
            "unique_knee_count": shortlist["knee_delta"].nunique(),
            "unique_phase_count": shortlist["phase_delta"].nunique(),
            "truth_different_pair_count": np.nan,
            "truth_difference_preserved_pair_count": np.nan,
            "pairwise_subject_difference_preservation_rate": np.nan,
            "truth_used_to_create_difference": False,
        }
    )
    return pd.DataFrame(rows)


def _matched_mismatch(
    summaries: pd.DataFrame,
    optimum: pd.DataFrame,
    execution: pd.DataFrame,
) -> pd.DataFrame:
    joined = summaries.merge(
        optimum[["case_id", "initial_optimum_absolute_J_error"]], on="case_id"
    )
    ranking_rows: list[dict[str, Any]] = []
    for case_id, group in execution.groupby("case_id"):
        correlation = group["J_pred_before_execution"].corr(group["actual_J"], method="spearman")
        ranking_rows.append({"case_id": case_id, "candidate_ranking_spearman": correlation})
    joined = joined.merge(pd.DataFrame(ranking_rows), on="case_id", how="left")
    return (
        joined.groupby("case_class", as_index=False)
        .agg(
            case_count=("case_id", "count"),
            mean_initial_optimum_absolute_J_error=("initial_optimum_absolute_J_error", "mean"),
            mean_candidate_ranking_spearman=("candidate_ranking_spearman", "mean"),
            false_improvement_count=("false_improvement_count", "sum"),
            shortlist_hit_rate=("shortlist_hit_truth_global_optimum", "mean"),
            mean_shortlist_regret=("shortlist_near_optimum_regret", "mean"),
            mean_final_J=("B2_final_best_validated_J", "mean"),
            mean_global_regret=("global_regret", "mean"),
            unique_best_validated_alpha_count=("best_validated_trajectory_id", "nunique"),
        )
    )


def _comparison(
    summaries: pd.DataFrame,
    specificity: pd.DataFrame,
) -> pd.DataFrame:
    unique_lookup = specificity.set_index("representation")["unique_full_alpha_count"]
    rows = [
        {
            "method_id": "B0_REFERENCE_ONLY",
            "trial_count": 0,
            "mean_final_J": 1.0,
            "mean_global_regret": float((1.0 - summaries["B3_truth_global_optimum_J"]).mean()),
            "false_improvement": 0,
            "unique_final_alpha_count": 1,
            "candidate_set_size_per_case": 0,
            "truth_used_for_selection": False,
        },
        {
            "method_id": "B1_PREDICTED_GLOBAL_BEST_ONLY",
            "trial_count": len(summaries),
            "mean_final_J": float(summaries["B1_predicted_best_only_J"].mean()),
            "mean_global_regret": float((summaries["B1_predicted_best_only_J"] - summaries["B3_truth_global_optimum_J"]).mean()),
            "false_improvement": int(
                summaries["B1_false_improvement_count"].sum()
            ),
            "unique_final_alpha_count": np.nan,
            "candidate_set_size_per_case": 1,
            "truth_used_for_selection": False,
        },
        {
            "method_id": "B2_FROZEN_TOP3_FINITE_SEQUENTIAL_VALIDATION",
            "trial_count": int(summaries["executed_validation_count"].sum()),
            "mean_final_J": float(summaries["B2_final_best_validated_J"].mean()),
            "mean_global_regret": float(summaries["global_regret"].mean()),
            "false_improvement": int(summaries["false_improvement_count"].sum()),
            "unique_final_alpha_count": int(unique_lookup[BEST_VALIDATED_TRAJECTORY]),
            "candidate_set_size_per_case": MAX_MODEL_SCREENED_CANDIDATES,
            "truth_used_for_selection": "MEASUREMENT_DECIDES_ONLY_WITHIN_FROZEN_SET",
        },
        {
            "method_id": "B3_ORACLE_TRUTH_GLOBAL_OPTIMUM",
            "trial_count": 0,
            "mean_final_J": float(summaries["B3_truth_global_optimum_J"].mean()),
            "mean_global_regret": 0.0,
            "false_improvement": 0,
            "unique_final_alpha_count": int(unique_lookup["TRUTH_GLOBAL_OPTIMUM"]),
            "candidate_set_size_per_case": 21025,
            "truth_used_for_selection": "OFFLINE_LOWER_BOUND_ONLY_NOT_POLICY",
        },
    ]
    old = pd.read_csv(ADAPTIVE_COMPARISON_PATH)
    for row in old.to_dict(orient="records"):
        rows.append(
            {
                "method_id": str(row["prototype_variant_id"]),
                "trial_count": int(row["trial_count"]),
                "mean_final_J": float(row["mean_final_J"]),
                "mean_global_regret": float(row["mean_global_regret"]),
                "false_improvement": int(row["false_improvement"]),
                "unique_final_alpha_count": int(row["unique_final_alpha_count"]),
                "candidate_set_size_per_case": "OPEN_SEQUENTIAL_RESEARCH_SPACE",
                "truth_used_for_selection": False,
            }
        )
    output = pd.DataFrame(rows)
    output["initial_identification_cost_excluded_as_common_baseline"] = True
    output["clinical_head_to_head_claim"] = False
    return output


def _status(
    comparison: pd.DataFrame,
    summaries: pd.DataFrame,
    specificity: pd.DataFrame,
) -> str:
    lookup = comparison.set_index("method_id")
    b1 = lookup.loc["B1_PREDICTED_GLOBAL_BEST_ONLY"]
    b2 = lookup.loc["B2_FROZEN_TOP3_FINITE_SEQUENTIAL_VALIDATION"]
    unique = specificity.set_index("representation")["unique_full_alpha_count"]
    monotone = (
        float(b2["mean_final_J"]) <= float(b1["mean_final_J"]) + 1e-12
        and float(b2["mean_global_regret"])
        <= float(b1["mean_global_regret"]) + 1e-12
        and int(unique[BEST_VALIDATED_TRAJECTORY]) > int(unique["OLD_BUNDLE5_FINAL"])
    )
    if not monotone:
        return FINAL_NOT_SUPPORTED
    perfect = (
        int(summaries["false_improvement_count"].sum()) == 0
        and bool(summaries["shortlist_hit_truth_global_optimum"].all())
    )
    return FINAL_SUPPORTED if perfect else FINAL_LIMITED


def _visualization_slices(
    prepared: Sequence[PreparedCase],
    evaluations: Sequence[CaseEvaluation],
    summaries: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    chosen: list[str] = []
    seen_truth: set[tuple[float, float, float]] = set()
    for row in summaries.sort_values("case_id").to_dict(orient="records"):
        key = (
            float(row["truth_global_alpha_hip"]),
            float(row["truth_global_alpha_knee"]),
            float(row["truth_global_alpha_phase"]),
        )
        if key not in seen_truth:
            chosen.append(str(row["case_id"]))
            seen_truth.add(key)
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise RuntimeError("three distinct truth landscapes are unavailable for GIF")
    prepared_lookup = {str(item.role["case_id"]): item for item in prepared}
    evaluation_lookup = {
        str(item.summary["case_id"]): item for item in evaluations
    }
    frames: list[pd.DataFrame] = []
    for case_id in chosen:
        initial = prepared_lookup[case_id].initial_prediction_map
        truth = evaluation_lookup[case_id].truth_map[
            ["trajectory_id", "J_truth"]
        ]
        merged = initial.merge(truth, on="trajectory_id", validate="one_to_one")
        predicted_indices = merged.groupby(
            ["hip_delta", "phase_delta"], sort=True
        )["J_pred"].idxmin()
        frame = merged.loc[predicted_indices].copy()
        truth_minimum = (
            merged.groupby(["hip_delta", "phase_delta"], as_index=False)[
                "J_truth"
            ]
            .min()
            .rename(columns={"J_truth": "J_truth_projected_minimum_over_knee"})
        )
        frame = frame.drop(columns="J_truth").merge(
            truth_minimum,
            on=["hip_delta", "phase_delta"],
            validate="one_to_one",
        )
        frame = frame.rename(
            columns={
                "J_truth_projected_minimum_over_knee": "J_truth",
                "knee_delta": "knee_at_predicted_minimum",
            }
        )
        frame["case_id"] = case_id
        frame["projection_definition"] = (
            "hip_phase_projection_with_J_pred_and_J_truth_each_minimized_over_"
            "the_formal_knee_axis"
        )
        frame["visualization_selection_role"] = (
            "LEXICAL_FIRST_THREE_DISTINCT_TRUTH_OPTIMA_POSTHOC_ONLY"
        )
        frame["truth_used_for_policy"] = False
        frames.append(
            frame[
                [
                    "case_id",
                    "trajectory_id",
                    "hip_delta",
                    "knee_at_predicted_minimum",
                    "phase_delta",
                    "J_pred",
                    "J_truth",
                    "model_supported",
                    "projection_definition",
                    "visualization_selection_role",
                    "truth_used_for_policy",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _data_role_document(checkpoint: Mapping[str, Any]) -> str:
    return f"""# DATA_ROLE_AUDIT

- Method: `{METHOD_ID}`
- Checkpoint: `{checkpoint['checkpoint_commit']}`
- Policy-validation outcomes: existing 9 `ORIGINAL_P2_DEVELOPMENT` plus 6
  `POST_REJECTION_DEVELOPMENT` virtual cases only.
- Initial identification observations are the existing safeguarded sequential
  architecture and are available before candidate screening.
- Candidate virtual truth is inaccessible until all 15 prediction-only
  shortlists and `FINAL_METHOD_MANIFEST_V1.json` are frozen and persisted.
- Independent calibration cases are not loaded or used for candidate-rule
  tuning.
- `HELD_OUT_FINAL_TEST` is not read.
- No future prospective cohort is generated.
- B3 truth landscapes are post-policy offline lower-bound comparators only.
- No human, robot-motion, comfort, safety, or clinical evidence is produced.
"""


def _visualization_guide() -> str:
    return """# VISUALIZATION_GUIDE

- `01_method_overview.png`: finite method flow and the prediction/measurement boundary.
- `02_subject_specific_landscape_example.png`: initial prediction versus post-freeze truth for one real simulation case.
- `03_frozen_shortlist_on_landscape.png`: C1–C3 locked on the initial landscape.
- `04_sequential_validation_progress.png`: whole-trajectory candidate trials and best validated J.
- `05_reference_vs_validated_J.png`: reference versus final measured/truth selection for all cases.
- `06_predicted_vs_truth_optimum.png`: model screening error at the initial predicted optimum.
- `07_subject_specific_alpha.png`: final validated hip, knee, and phase coordinates.
- `08_budget_sensitivity.png`: posthoc Budget 0/1/2/3 comparison on the same shortlist.
- `09_p2_vs_finite_method_trial_cost.png`: research-architecture trial-count comparison, not a clinical head-to-head study.
- `FINAL_METHOD_WORKFLOW_ANIMATION.gif`: a 12-stage, 16:9 explanation for supervisor presentations, the manuscript Method presentation, and possible future supplementary material.
- `SUBJECT_SPECIFIC_LANDSCAPE_COMPARISON.gif`: three unmodified simulation landscapes with different truth optima; the selected subjects are determined posthoc only for visualization and never affect the policy.

Every colored J surface is a whole-trajectory mechanical objective. It is not
comfort, an instantaneous reward, or a single robot-control step.
"""


def _report(
    *,
    status: str,
    comparison: pd.DataFrame,
    budget: pd.DataFrame,
    matched: pd.DataFrame,
    specificity: pd.DataFrame,
    summaries: pd.DataFrame,
    manifest_sha256: str,
) -> str:
    lookup = comparison.set_index("method_id")
    b0 = lookup.loc["B0_REFERENCE_ONLY"]
    b1 = lookup.loc["B1_PREDICTED_GLOBAL_BEST_ONLY"]
    b2 = lookup.loc["B2_FROZEN_TOP3_FINITE_SEQUENTIAL_VALIDATION"]
    b3 = lookup.loc["B3_ORACLE_TRUTH_GLOBAL_OPTIMUM"]
    budget_mean = budget.groupby("budget", as_index=False).agg(
        mean_J=("final_best_validated_J", "mean"),
        mean_regret=("global_regret", "mean"),
        false_improvement=("false_improvement_count", "sum"),
        candidate_cost=("executed_candidate_count", "sum"),
    )
    specificity_lookup = specificity.set_index("representation")
    old_unique = int(specificity_lookup.loc["OLD_BUNDLE5_FINAL", "unique_full_alpha_count"])
    new_unique = int(specificity_lookup.loc[BEST_VALIDATED_TRAJECTORY, "unique_full_alpha_count"])
    old_h2 = lookup.loc["H2_FIXED_BUNDLE_5"]
    reduction = 1.0 - float(b2["trial_count"]) / float(old_h2["trial_count"])
    matched_lines = "\n".join(
        f"- {row.case_class}: mean J={row.mean_final_J:.6f}, regret={row.mean_global_regret:.6f}, prediction error={row.mean_initial_optimum_absolute_J_error:.6f}, false improvements={int(row.false_improvement_count)}."
        for row in matched.itertuples(index=False)
    )
    budget_lines = "\n".join(
        f"- Budget {int(row.budget)}: mean J={row.mean_J:.6f}, regret={row.mean_regret:.6f}, false improvements={int(row.false_improvement)}, candidate cost={int(row.candidate_cost)}."
        for row in budget_mean.itertuples(index=False)
    )
    stop = status in (FINAL_SUPPORTED, FINAL_LIMITED)
    return f"""# FINAL_MODEL_SCREENED_FINITE_SEQUENTIAL_VALIDATION_V1

Manifest SHA-256: `{manifest_sha256}`

## A–G. Method in plain language

The subject first completes the unchanged initial identification protocol. The
five-parameter model predicts J for all 21,025 geometrically valid grid points,
then freezes at most three supported candidates before candidate truth is read.
Each round validates one complete trajectory, refits the model, recomputes the
full landscape, and reranks only the remaining frozen candidates. No C4 or
later candidate can enter. The final trajectory is the lowest measured/virtual-
truth J among the reference and actually validated candidates.

This is still “step by step,” but one step is one **whole-trajectory trial**, not
a local alpha-grid perturbation. Search is finite because the persisted
shortlist is immutable and the registered budget is at most three. The full
landscape is recomputed to diagnose model change; a new predicted optimum is
never physically/virtually added. Prediction screens, measurement decides.

## H. B0/B1/B2/B3

| Comparator | Candidate trials | Mean final J | Mean regret | False improvements |
|---|---:|---:|---:|---:|
| B0 reference | {int(b0.trial_count)} | {b0.mean_final_J:.6f} | {b0.mean_global_regret:.6f} | {int(b0.false_improvement)} |
| B1 predicted best only | {int(b1.trial_count)} | {b1.mean_final_J:.6f} | {b1.mean_global_regret:.6f} | {int(b1.false_improvement)} |
| B2 frozen Top-3 finite | {int(b2.trial_count)} | {b2.mean_final_J:.6f} | {b2.mean_global_regret:.6f} | {int(b2.false_improvement)} |
| B3 truth oracle | offline lower bound | {b3.mean_final_J:.6f} | 0 | 0 |

## I. Budget sensitivity

{budget_lines}

These values do not change the preregistered three-candidate research budget.

## J. Matched versus mismatch

{matched_lines}

Any mismatch gap is an `APPLICABILITY_LIMITATION`; the five-parameter model is
not changed here.

## K. Subject specificity

The old BUNDLE_5 final alpha has `{old_unique}` unique full alpha; the finite
method has `{new_unique}`. This difference arises from unmodified simulation
measurements, not a diversity reward or truth-driven shortlist.

## L. Trial-count comparison

Old fixed BUNDLE_5 used `{int(old_h2.trial_count)}` personalization trials over
15 cases. B2 uses `{int(b2.trial_count)}` complete candidate validations, a
`{100.0 * reduction:.1f}%` reduction. Initial identification is excluded from
both sides as the common baseline. This is an offline architecture comparison,
not a prospective or clinical head-to-head conclusion.

## M–N. Visualizations

Nine static figures and two deterministic GIFs are generated. The mandatory
GIFs are `FINAL_METHOD_WORKFLOW_ANIMATION.gif` and
`SUBJECT_SPECIFIC_LANDSCAPE_COMPARISON.gif`.

## O–P. Final interpretation

Final status: `{status}`

`STOP_FURTHER_P2_EXPANSION = {str(stop).lower()}`

The evidence remains `OFFLINE_ONLY`, `NOT_HUMAN_READY`, and
`NOT_ROBOT_APPROVED`. If supported with limitations, the next work is method
freeze, manuscript Method/Offline Results writing, and separately governed
limited fixed-candidate physical validation—not P2 V3 or another optimizer.
"""


def generate_artifacts(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_preflight()
    verify_immutable_prospective_artifacts()
    validate_active_reference_file(ACTIVE_REFERENCE_PATH)
    if ACTIVE_REFERENCE_SHA256 != sha256_file(ACTIVE_REFERENCE_PATH):
        raise RuntimeError("active reference SHA changed")
    source_before = _source_hashes()
    parameter_map = pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    lattice = geometrically_valid_parameter_lattice(parameter_map)
    if len(lattice) != 21025:
        raise RuntimeError("formal generator lattice changed")
    cache = build_trajectory_component_cache(lattice)
    cases = _case_table()
    prepared = _prepare_cases(cases, lattice, cache)

    manifest = method_manifest_payload(
        checkpoint=checkpoint,
        source_hashes=source_before,
        shortlists=[item.shortlist for item in prepared],
    )
    manifest.update(
        {
            "case_count": len(prepared),
            "case_roles": cases.to_dict(orient="records"),
            "geometric_lattice_size": len(lattice),
            "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
            "formal_hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
            "formal_knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
            "status_rule": FINAL_STATUS_RULE,
            "manifest_persisted_before_candidate_truth": True,
        }
    )
    manifest_path = output / "FINAL_METHOD_MANIFEST_V1.json"
    _write_json(manifest_path, manifest, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    shortlist_table = _shortlist_table(prepared, manifest_sha)
    _write_csv(output / "candidate_shortlist_manifest.csv", shortlist_table)

    evaluations: list[CaseEvaluation] = []
    for item in prepared:
        def evaluate_one() -> None:
            evaluations.append(
                _evaluate_case(
                    item, cache=cache, global_manifest_sha256=manifest_sha
                )
            )

        if item.role["development_origin"] == "POST_REJECTION_DEVELOPMENT":
            subject = dynamic_subject_for_id(str(item.role["subject_id"]))
            with registered_prospective_subject(subject):
                evaluate_one()
        else:
            evaluate_one()
    execution = pd.concat(
        [item.execution_history for item in evaluations], ignore_index=True
    )
    model_history = pd.concat(
        [item.model_history for item in evaluations], ignore_index=True
    )
    summaries = pd.DataFrame([item.summary for item in evaluations])
    optimum = pd.DataFrame([item.optimum for item in evaluations])
    false_frames = [item.false_improvements for item in evaluations]
    false_audit = pd.concat(false_frames, ignore_index=True)
    if false_audit.empty:
        false_audit = execution.head(0).copy()
    budget = _budget_table(summaries, execution)
    specificity = _subject_specificity(summaries, shortlist_table)
    matched = _matched_mismatch(summaries, optimum, execution)
    comparison = _comparison(summaries, specificity)
    visualization_slices = _visualization_slices(
        prepared, evaluations, summaries, shortlist_table
    )
    status = _status(comparison, summaries, specificity)
    stop_expansion = status in (FINAL_SUPPORTED, FINAL_LIMITED)

    tables = {
        "candidate_execution_history.csv": execution,
        "model_update_history.csv": model_history,
        "final_subject_summary.csv": summaries,
        "predicted_vs_truth_optimum.csv": optimum,
        "budget_sensitivity.csv": budget,
        "matched_mismatch_summary.csv": matched,
        "subject_specificity_summary.csv": specificity,
        "false_improvement_audit.csv": false_audit,
        "p2_vs_finite_method_comparison.csv": comparison,
        "visualization_landscape_slice.csv": visualization_slices,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)
    _write_text(output / "DATA_ROLE_AUDIT.md", _data_role_document(checkpoint))
    _write_text(output / "VISUALIZATION_GUIDE.md", _visualization_guide())
    report = _report(
        status=status,
        comparison=comparison,
        budget=budget,
        matched=matched,
        specificity=specificity,
        summaries=summaries,
        manifest_sha256=manifest_sha,
    )
    _write_text(output / "FINAL_METHOD_VALIDATION_REPORT.md", report)
    generate_all_visuals(output)

    source_after = _source_hashes()
    if source_before != source_after:
        raise RuntimeError("protected frozen source changed during validation")
    artifact_names = [
        *REQUIRED_TABLES,
        *REQUIRED_DOCUMENTS[:-1],
        "visualization_landscape_slice.csv",
        *FIGURE_NAMES,
        *GIF_NAMES,
    ]
    artifact_manifest = {
        name: {
            "sha256": sha256_file(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in artifact_names
    }
    finite = comparison.set_index("method_id").loc[
        "B2_FROZEN_TOP3_FINITE_SEQUENTIAL_VALIDATION"
    ]
    metadata = {
        "method_id": METHOD_ID,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "checkpoint": checkpoint,
        "final_status": status,
        "STOP_FURTHER_P2_EXPANSION": stop_expansion,
        "default_enabled": DEFAULT_ENABLED,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "case_count": len(summaries),
        "frozen_shortlist_count": len(shortlist_table),
        "executed_candidate_validation_count": len(execution),
        "maximum_candidate_validations_per_case": int(
            summaries["executed_validation_count"].max()
        ),
        "mean_final_J": float(finite["mean_final_J"]),
        "mean_global_regret": float(finite["mean_global_regret"]),
        "false_improvement_count": int(finite["false_improvement"]),
        "shortlist_hit_rate": float(
            summaries["shortlist_hit_truth_global_optimum"].mean()
        ),
        "unique_best_validated_alpha_count": int(
            summaries[
                [
                    "best_validated_alpha_hip",
                    "best_validated_alpha_knee",
                    "best_validated_alpha_phase",
                ]
            ].drop_duplicates().shape[0]
        ),
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "candidate_shortlists_frozen_before_truth": True,
        "manifest_persisted_before_candidate_truth": True,
        "new_candidate_after_freeze_added": False,
        "whole_trajectory_trials": True,
        "model_refit_after_every_candidate": True,
        "full_landscape_recomputed_after_every_candidate": True,
        "remaining_frozen_candidates_only_reranked": True,
        "P2_explore_exploit_run": False,
        "bundle_run": False,
        "adaptive_horizon_run": False,
        "calibration_cases_used_for_rule_tuning": False,
        "held_out_final_test_read": False,
        "new_prospective_cohort_generated": False,
        "robot_connected": False,
        "hardware_control_collection_safety_modified": False,
        "protected_source_sha256_before": source_before,
        "protected_source_sha256_after": source_after,
        "artifact_manifest": artifact_manifest,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
