"""Prospective, default-off offline validation engine for the P2 V2 candidates.

The cohort and policy bundle are constructed without virtual-truth outcomes.
Every personalization truth call is guarded by a frozen-manifest check and an
explicit one-trajectory selection token.  This module does not register a new
runtime policy, change P2 V1, or import robot-side code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    generate_personalized_trajectory,
)
from .decision_relevant_global_model_reliability import (
    DiagnosticInitialModel,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    TrajectoryComponentCache,
    build_predicted_map,
    evaluate_truth_map,
)
from .dynamic_subject import DYNAMIC_SUBJECTS, DynamicVirtualSubject, get_dynamic_subject
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
)
from .parameter_estimator import PARAMETER_NAMES
from .p2_v2_offline_research_prototype import (
    FROZEN_LOCAL_PROTOCOL_ID,
    FROZEN_PAIR_PLAN_SHA256,
    local_uncertainty_metrics,
)
from .research_decision_guarded_sequential_personalization import (
    CURRENT_BEST_NOT_A_CANDIDATE,
    EXECUTED_FALSE_IMPROVEMENT,
    MAP_TRUTH_ROLE,
    MODEL_RELIABILITY_DEGRADED,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    RESEARCH_ONLY,
    STOP_MAX_PERSONALIZATION_TRIALS,
    STOP_MODEL_ADEQUACY_DEGRADED,
    STOP_MODEL_UPDATE_FAILURE,
    STOP_NO_GEOMETRICALLY_VALID_CANDIDATE,
    STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER,
    STOP_PATIENT_ENVELOPE_BOUNDARY,
    TRIAL_PURPOSE_EXPLOIT,
    TRIAL_PURPOSE_EXPLORE,
    InitialResearchState,
    PolicyRunResult,
    ResearchDecisionUncertainty,
    SelectionGatedVirtualTruthOracle,
    _actual_objective,
    _fit_updated_model,
    _map_summary,
    _model_for_iteration,
    _parameter_uncertainty_trace,
    _row_for_alpha,
    alpha_from_row,
    apply_research_decision_guard,
    build_initial_research_state,
    build_local_exploration_frontier,
    local_prediction_candidates,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .sequential_personalization import (
    SearchAlpha,
    TrustRegionSteps,
    accept_actual_trial,
    build_coordinate_neighborhood,
    shrink_steps,
)


PROTOCOL_ID = "P2_V2_PROSPECTIVE_OFFLINE_VALIDATION_V1"
MANIFEST_ID = "P2_V2_PROSPECTIVE_EVALUATION_MANIFEST_V1"
PROSPECTIVE_STATUS = "OFFLINE_SYNTHETIC_RESEARCH_ONLY"
DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
HELD_OUT_FINAL_TEST = "HELD_OUT_FINAL_TEST_NOT_READ"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"
OFFLINE_METHOD_REQUIRES_REVISION = "OFFLINE_METHOD_REQUIRES_REVISION"
P2_V2_DEFAULT_ENABLED = False
INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS = (
    "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW"
)
GLOBAL_MODEL_RELIABILITY_STATUS = "GLOBAL_MODEL_RELIABILITY_RULE_NOT_FROZEN_FOR_HUMANS"

LOCAL_MAX = 0.0016827379049442204
LOCAL_P95 = 0.000430956758923898
LOCAL_P99 = 0.001276942013587856
PROSPECTIVE_SUBJECT_SELECTION_SEED = 20260823
MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON = 20
EXPECTED_GEOMETRIC_LATTICE_SIZE = 21025

FINAL_SUPPORTS = "P2_V2_PROSPECTIVE_EVIDENCE_SUPPORTS_OFFLINE_FREEZE"
FINAL_INSUFFICIENT = "P2_V2_PROSPECTIVE_EVIDENCE_INSUFFICIENT"
FINAL_REJECTS = "P2_V2_PROSPECTIVE_EVIDENCE_REJECTS_CURRENT_REVISION"
FINAL_STATUSES = (FINAL_SUPPORTS, FINAL_INSUFFICIENT, FINAL_REJECTS)

STOP_DECISION_VALUE_K = "STOP_DECISION_VALUE_ZERO_RUN_K"

PAIR_PLAN_PATH = (
    Path(__file__).resolve().parent
    / "formal_artifacts"
    / "p2_v2_formal_research_protocol_v1"
    / "designated_local_validation_pair_plan.csv"
)
LOCAL_RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "formal_artifacts"
    / "p2_v2_offline_research_prototype_v1"
    / "local_validation_results.csv"
)

DEVELOPMENT_CASES = (
    "baseline__matched_linear",
    "hip_stiff__matched_linear",
    "knee_stiff__matched_linear",
    "heavy_leg__matched_linear",
    "baseline__nonlinear_stiffness_mild",
    "baseline__hip_knee_coupling_mild",
    "baseline__nonlinear_damping_mild",
    "baseline__structured_residual",
    "baseline__combined_mild",
)


@dataclass(frozen=True)
class ProspectivePolicySpec:
    policy_variant_id: str
    guard_id: str
    cumulative_rule_id: str
    stopping_rule_id: str
    stopping_k: int | None
    role: str
    active_in_prospective_execution: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICY_VARIANTS = (
    ProspectivePolicySpec(
        "P2_V1_G0_C0_S0",
        "G0_CURRENT_GLOBAL_MAX",
        "C0_SINGLE_STEP",
        "S0_CURRENT_CONTINUATION",
        None,
        "PRIMARY_P2_V1",
    ),
    ProspectivePolicySpec(
        "P2_V2A_G2_C0_S2",
        "G2_FROZEN_LOCAL_P95",
        "C0_SINGLE_STEP",
        "S2_DECISION_VALUE_K2",
        2,
        "PRIMARY_P2_V2A_DEFAULT_OFF",
    ),
    ProspectivePolicySpec(
        "P2_V2A_G3_C0_S2_SENSITIVITY",
        "G3_FROZEN_LOCAL_P99",
        "C0_SINGLE_STEP",
        "S2_DECISION_VALUE_K2",
        2,
        "SENSITIVITY_LOCAL_P99",
    ),
    ProspectivePolicySpec(
        "P2_V2A_G2_C0_S1_SENSITIVITY",
        "G2_FROZEN_LOCAL_P95",
        "C0_SINGLE_STEP",
        "S1_DECISION_VALUE_K1",
        1,
        "SENSITIVITY_STOPPING_K1",
    ),
    ProspectivePolicySpec(
        "P2_V2A_G2_C0_S3_SENSITIVITY",
        "G2_FROZEN_LOCAL_P95",
        "C0_SINGLE_STEP",
        "S3_DECISION_VALUE_K3",
        3,
        "SENSITIVITY_STOPPING_K3",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def stable_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the exact canonical bytes written by the prospective runner."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _subject_candidate_grid() -> list[tuple[float, float, float, float]]:
    return list(
        itertools.product(
            (0.88, 0.96, 1.04, 1.12),
            (11.0, 18.0, 24.0, 27.0),
            (9.0, 16.0, 21.0, 26.0),
            (0.9, 1.1),
        )
    )


def prospective_subject_definitions() -> tuple[dict[str, Any], ...]:
    """Select three unused combinations by SHA only, never by truth outcome."""

    ranked: list[tuple[str, tuple[float, float, float, float]]] = []
    for values in _subject_candidate_grid():
        identity = "|".join(
            (
                str(PROSPECTIVE_SUBJECT_SELECTION_SEED),
                *(f"{value:.12g}" for value in values),
            )
        )
        ranked.append((hashlib.sha256(identity.encode("utf-8")).hexdigest(), values))
    baseline = get_dynamic_subject("baseline")
    output: list[dict[str, Any]] = []
    for index, (selection_hash, values) in enumerate(sorted(ranked)[:3], start=1):
        mass_scale, k_hip, k_knee, damping_scale = values
        subject_id = f"prospective_subject_{index:03d}"
        subject = DynamicVirtualSubject(
            subject_id=subject_id,
            mass_thigh_kg=baseline.mass_thigh_kg * mass_scale,
            mass_shank_kg=baseline.mass_shank_kg * mass_scale,
            com_thigh_m=baseline.com_thigh_m,
            com_shank_m=baseline.com_shank_m,
            inertia_thigh_kg_m2=baseline.inertia_thigh_kg_m2 * mass_scale,
            inertia_shank_kg_m2=baseline.inertia_shank_kg_m2 * mass_scale,
            b_hip_nm_s_per_rad=baseline.b_hip_nm_s_per_rad * damping_scale,
            b_knee_nm_s_per_rad=baseline.b_knee_nm_s_per_rad * damping_scale,
            k_hip_nm_per_rad=k_hip,
            k_knee_nm_per_rad=k_knee,
            q0_hip_rad=baseline.q0_hip_rad,
            q0_knee_rad=baseline.q0_knee_rad,
            gravity_m_s2=baseline.gravity_m_s2,
        )
        output.append(
            {
                "subject_id": subject_id,
                "selection_hash": selection_hash,
                "selection_seed": PROSPECTIVE_SUBJECT_SELECTION_SEED,
                "selection_rule": "LOWEST_SHA256_OVER_FIXED_DISCRETE_SYNTHETIC_GRID",
                "candidate_grid_size": len(ranked),
                "mass_scale": mass_scale,
                "damping_scale": damping_scale,
                "parameters": subject.as_metadata_dict(),
                "parameter_interpretation": "OFFLINE_EQUIVALENT_SYNTHETIC_NOT_CLINICAL",
                "truth_used_for_selection": False,
            }
        )
    return tuple(output)


def prospective_case_rows() -> pd.DataFrame:
    subjects = prospective_subject_definitions()
    mismatch_names = (
        "nonlinear_stiffness_strong",
        "hip_knee_coupling_strong",
        "combined_strong",
    )
    rows: list[dict[str, Any]] = []
    for index, definition in enumerate(subjects):
        subject_id = str(definition["subject_id"])
        for scenario_name, case_class in (
            ("matched_linear", "PROSPECTIVE_MATCHED"),
            (mismatch_names[index], "PROSPECTIVE_MODEL_MISMATCH"),
        ):
            case_id = f"{subject_id}__{scenario_name}"
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "scenario_name": scenario_name,
                    "case_class": case_class,
                    "data_split": "PROSPECTIVE",
                    "research_status": PROSPECTIVE_STATUS,
                    "primary_prospective_cohort": True,
                    "development_case": case_id in DEVELOPMENT_CASES,
                    "subject_selection_hash": definition["selection_hash"],
                    "subject_selection_seed": PROSPECTIVE_SUBJECT_SELECTION_SEED,
                    "truth_used_for_case_selection": False,
                    "heldout_final_test": False,
                }
            )
    output = pd.DataFrame(rows)
    if output["case_id"].duplicated().any():
        raise RuntimeError("prospective case IDs must be unique")
    if output["development_case"].astype(bool).any():
        raise RuntimeError("development case entered prospective cohort")
    return output


def dynamic_subject_for_id(subject_id: str) -> DynamicVirtualSubject:
    definitions = {
        str(item["subject_id"]): item for item in prospective_subject_definitions()
    }
    try:
        parameters = definitions[str(subject_id)]["parameters"]
    except KeyError as exc:
        raise ValueError(f"unknown prospective subject: {subject_id}") from exc
    return DynamicVirtualSubject(**parameters)


@contextmanager
def registered_prospective_subject(subject: DynamicVirtualSubject) -> Iterator[None]:
    """Temporarily expose one frozen synthetic subject to unchanged offline code."""

    if subject.subject_id in DYNAMIC_SUBJECTS:
        raise RuntimeError("prospective subject ID collides with the development registry")
    before_keys = tuple(DYNAMIC_SUBJECTS)
    DYNAMIC_SUBJECTS[subject.subject_id] = subject
    try:
        yield
    finally:
        current = DYNAMIC_SUBJECTS.pop(subject.subject_id, None)
        if current is not subject or tuple(DYNAMIC_SUBJECTS) != before_keys:
            raise RuntimeError("temporary prospective subject registry was not restored")


def validate_frozen_local_evidence(
    pair_plan_path: Path = PAIR_PLAN_PATH,
    results_path: Path = LOCAL_RESULTS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan_sha = _sha256_file(pair_plan_path)
    if plan_sha != FROZEN_PAIR_PLAN_SHA256:
        raise RuntimeError("LOCAL_VALIDATION_PROTOCOL_PROVENANCE_FAILURE")
    plan = pd.read_csv(pair_plan_path)
    results = pd.read_csv(results_path)
    if len(plan) != 324 or len(results) != 324:
        raise RuntimeError("LOCAL_VALIDATION_PROTOCOL_PROVENANCE_FAILURE")
    if set(plan["pair_id"].astype(str)) != set(results["pair_id"].astype(str)):
        raise RuntimeError("LOCAL_VALIDATION_PROTOCOL_PROVENANCE_FAILURE")
    metrics = local_uncertainty_metrics(results)
    expected = {"local_max": LOCAL_MAX, "local_P95": LOCAL_P95, "local_P99": LOCAL_P99}
    for name, value in expected.items():
        if not np.isclose(metrics[name], value, atol=1e-12, rtol=0.0):
            raise RuntimeError("LOCAL_VALIDATION_PROTOCOL_PROVENANCE_FAILURE")
    provenance = pd.DataFrame(
        [
            {
                "protocol_id": FROZEN_LOCAL_PROTOCOL_ID,
                "pair_plan_sha256": plan_sha,
                "pair_count": len(plan),
                "results_sha256": _sha256_file(results_path),
                "metric_id": name,
                "metric_value": metrics[name],
                "source_split": "FROZEN_DESIGNATED_LOCAL_VALIDATION",
                "prospective_case_contributed": False,
                "threshold_updated_from_prospective_truth": False,
            }
            for name in ("local_max", "local_P95", "local_P99")
        ]
    )
    return provenance, metrics


def audit_bundle_uncertainty(pair_plan: pd.DataFrame) -> pd.DataFrame:
    """Fail closed because the frozen plan has no predesignated bundle residuals."""

    bundle_columns = {
        "designated_bundle_id",
        "bundle_length",
        "bundle_endpoint_error",
        "bundle_path_direction_consistent",
    }
    has_bundle_schema = bundle_columns.issubset(pair_plan.columns)
    rows = [
        {
            "cumulative_rule_id": "C0_SINGLE_STEP",
            "bundle_length": 1,
            "calibration_status": "ACTIVE_SINGLE_STEP_NOT_A_BUNDLE",
            "designated_bundle_evidence_count": 324,
            "active_prospective_policy": True,
            "uncertainty_formula": "FROZEN_SINGLE_PAIR_LOCAL_BOUND",
            "n_times_one_step_assumed": False,
            "sqrt_n_times_one_step_assumed": False,
        }
    ]
    for length in (2, 3, 5):
        count = (
            int(pair_plan["bundle_length"].astype(int).eq(length).sum())
            if has_bundle_schema
            else 0
        )
        rows.append(
            {
                "cumulative_rule_id": f"C{length}_{length}_STEP_CUMULATIVE",
                "bundle_length": length,
                "calibration_status": "SHADOW_ONLY_NOT_CALIBRATED",
                "designated_bundle_evidence_count": count,
                "active_prospective_policy": False,
                "uncertainty_formula": "NONE_FAIL_CLOSED",
                "n_times_one_step_assumed": False,
                "sqrt_n_times_one_step_assumed": False,
            }
        )
    return pd.DataFrame(rows)


def build_prospective_manifest(
    prospective_start_commit_sha: str,
    *,
    protected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build all immutable prospective inputs without accessing truth."""

    cases = prospective_case_rows()
    subjects = prospective_subject_definitions()
    return {
        "manifest_id": MANIFEST_ID,
        "protocol_id": PROTOCOL_ID,
        "prospective_start_commit_sha": str(prospective_start_commit_sha),
        "manifest_must_precede_truth": True,
        "cohort_immutable_after_manifest_freeze": True,
        "prospective_subject_selection_seed": PROSPECTIVE_SUBJECT_SELECTION_SEED,
        "prospective_subject_definitions": list(subjects),
        "prospective_cases": cases.to_dict(orient="records"),
        "development_cases": [
            {"case_id": case_id, "data_split": DEVELOPMENT_ONLY, "primary_metrics": False}
            for case_id in DEVELOPMENT_CASES
        ],
        "held_out_final_test": {
            "status": HELD_OUT_FINAL_TEST,
            "read_during_experiment": False,
            "contributes_to_metrics": False,
        },
        "frozen_scientific_baseline": {
            "rom_protocol_version": ROM_PROTOCOL_VERSION,
            "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
            "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
            "active_reference_id": ACTIVE_REFERENCE_ID,
            "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
            "theta_shank_definition": THETA_SHANK_DEFINITION,
            "five_parameter_names": list(PARAMETER_NAMES),
            "mechanical_objective_version": MECHANICAL_OBJECTIVE_VERSION,
            "algorithm_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
            "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
            "generator_bounds": {
                name: list(values)
                for name, values in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
            },
        },
        "local_validation": {
            "protocol_id": FROZEN_LOCAL_PROTOCOL_ID,
            "pair_plan_sha256": FROZEN_PAIR_PLAN_SHA256,
            "pair_count": 324,
            "local_max": LOCAL_MAX,
            "local_P95": LOCAL_P95,
            "local_P99": LOCAL_P99,
            "prospective_outcomes_may_update_calibration": False,
        },
        "bundle_uncertainty": {
            "C2": "SHADOW_ONLY_NOT_CALIBRATED",
            "C3": "SHADOW_ONLY_NOT_CALIBRATED",
            "C5": "SHADOW_ONLY_NOT_CALIBRATED",
            "P2_V2B_active": False,
            "n_times_bound_prohibited": True,
            "sqrt_n_times_bound_prohibited": True,
        },
        "operational_envelope_fixture": "VIRTUAL_RESEARCH_ENVELOPE_DEFAULT",
        "initial_identification_protocol": "SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1_UNCHANGED",
        "maximum_research_diagnostic_horizon": MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
        "policy_variants": [spec.as_dict() for spec in POLICY_VARIANTS],
        "inactive_policy_candidates": [
            {
                "policy_variant_id": "P2_V2B_G2_C2_S2",
                "status": "NOT_ACTIVE_BUNDLE_UNCERTAINTY_NOT_CALIBRATED",
            }
        ],
        "decision_value_observables": [
            "parameter_change",
            "prediction_map_change",
            "validation_uncertainty_change",
            "exploit_eligibility_change",
            "information_gain",
            "support_growth",
            "predicted_ranking_change",
            "predicted_best_trajectory_change",
        ],
        "decision_value_forbidden_inputs": [
            "future_truth",
            "future_exploit",
            "future_best",
        ],
        "evaluation_metrics": [
            "executed_trials",
            "explore_count",
            "exploit_count",
            "executed_false_improvement",
            "missed_improvement_rounds",
            "premature_conservative_stops",
            "correct_local_stops",
            "final_best_actual_J",
            "J_reduction_from_reference",
            "local_regret",
            "cumulative_regret",
            "support_growth",
            "low_decision_value_exploration_count",
            "natural_stop_reason",
            "final_alpha",
        ],
        "truth_access_policy": {
            "policy_action_frozen_before_truth": True,
            "one_selection_token_per_executed_trajectory": True,
            "post_policy_landscape_truth_role": "POST_POLICY_EVALUATION_ONLY",
            "truth_may_update_policy_or_calibration": False,
        },
        "pre_registered_final_status_rule": {
            "supports": (
                "V2A total missed rounds <= V1, false improvements <= V1, "
                "mean final J and mean regret each <= V1 + 0.005, and mean trials <= V1"
            ),
            "rejects": (
                "V2A false improvements > V1 or mean final J/regret exceeds V1 by >0.005"
            ),
            "otherwise": FINAL_INSUFFICIENT,
        },
        "protected_source_sha256": dict(protected_source_sha256),
        "P2_V2_default_enabled": False,
        "human_ready": False,
        "robot_motion_approved": False,
        "truth_used_to_construct_manifest": False,
    }


class FrozenManifestGate:
    """Require immutable on-disk manifest bytes before any prospective truth."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = str(expected_sha256)
        self.truth_access_count = 0

    def require_frozen(self) -> None:
        if not self.path.is_file() or _sha256_file(self.path) != self.expected_sha256:
            raise PermissionError("prospective truth requires the frozen manifest")

    def record_truth_access(self) -> None:
        self.require_frozen()
        self.truth_access_count += 1


def _static_uncertainty(
    state: InitialResearchState,
    iteration: int,
    guard_id: str,
) -> ResearchDecisionUncertainty:
    if guard_id == "G2_FROZEN_LOCAL_P95":
        bound, bound_type = LOCAL_P95, "FROZEN_DESIGNATED_LOCAL_P95"
    elif guard_id == "G3_FROZEN_LOCAL_P99":
        bound, bound_type = LOCAL_P99, "FROZEN_DESIGNATED_LOCAL_P99"
    else:
        raise ValueError(f"guard has no frozen local uncertainty: {guard_id}")
    return ResearchDecisionUncertainty(
        case_id=f"{state.subject_id}__{state.scenario_name}",
        iteration=int(iteration),
        pairwise_audit=pd.DataFrame(),
        maximum_observed_e_delta_j=LOCAL_MAX,
        p95_observed_e_delta_j=LOCAL_P95,
        p99_observed_e_delta_j=LOCAL_P99,
        validation_pair_count=324,
        bound_used_by_guard=bound,
        bound_type=bound_type,
        bound_status="FROZEN_RESEARCH_CALIBRATION_DEFAULT_OFF_PROSPECTIVE",
    )


def _uncertainty_for_policy(
    state: InitialResearchState,
    parameters: Mapping[str, float],
    iteration: int,
    spec: ProspectivePolicySpec,
) -> ResearchDecisionUncertainty:
    if spec.guard_id == "G0_CURRENT_GLOBAL_MAX":
        from .research_decision_guarded_sequential_personalization import (
            evaluate_validation_pairwise_uncertainty,
        )

        return evaluate_validation_pairwise_uncertainty(
            state, parameters, iteration=iteration
        )
    return _static_uncertainty(state, iteration, spec.guard_id)


def _strict_map_changed(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    joined = before[["trajectory_id", "J_pred"]].merge(
        after[["trajectory_id", "J_pred"]],
        on="trajectory_id",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    return bool(
        np.any(
            joined["J_pred_before"].to_numpy(dtype=float)
            != joined["J_pred_after"].to_numpy(dtype=float)
        )
    )


def _predicted_rank_signature(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        table.sort_values(["J_pred", "trajectory_id"], kind="mergesort")[
            "trajectory_id"
        ].astype(str)
    )


def _next_exploit_available(
    prediction_map: pd.DataFrame,
    current_best: SearchAlpha,
    steps: TrustRegionSteps,
    uncertainty: ResearchDecisionUncertainty,
) -> bool:
    local = local_prediction_candidates(prediction_map, current_best, steps)
    guarded = apply_research_decision_guard(local, current_best, uncertainty)
    return (
        select_exploit_candidate(
            guarded, POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
        )
        is not None
    )


def run_prospective_policy(
    state: InitialResearchState,
    spec: ProspectivePolicySpec,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    manifest_gate: FrozenManifestGate,
    *,
    trial_budget: int = MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
) -> PolicyRunResult:
    """Run one isolated prospective policy without changing the P2 V1 module."""

    if not spec.active_in_prospective_execution:
        raise PermissionError("inactive policy candidate cannot be executed")
    if spec.cumulative_rule_id != "C0_SINGLE_STEP":
        raise PermissionError("uncalibrated cumulative policy must fail closed")
    if trial_budget != MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON:
        raise ValueError("prospective diagnostic horizon is frozen")
    manifest_gate.require_frozen()

    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    model = _model_for_iteration(state, parameters, domain_data, 0)
    prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
    initial_prediction_map = prediction_map.copy(deep=True)
    uncertainty = _uncertainty_for_policy(state, parameters, 0, spec)
    oracle = SelectionGatedVirtualTruthOracle(state.subject_id, state.scenario_name)

    reference = generate_personalized_trajectory()
    reference_trajectory = reference.trajectory.copy(deep=True)
    reference_id = str(reference.metadata["trajectory_id"])
    reference_trajectory["trajectory_id"] = reference_id
    reference_token = oracle.declare_selected(reference_id, "REFERENCE_NORMALIZATION")
    manifest_gate.record_truth_access()
    reference_execution = oracle.execute(reference_token, reference_trajectory)
    reference_metrics = reference_execution.actual_metrics

    current_best = SearchAlpha()
    best_actual_j = 1.0
    steps = TrustRegionSteps()
    executed_keys: set[tuple[float, float, float]] = {current_best.key()}
    history_rows: list[dict[str, Any]] = []
    guard_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = [
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": spec.policy_variant_id,
            **_map_summary(prediction_map, current_best, iteration=0, previous_map=None),
        }
    ]
    known_rows: list[dict[str, Any]] = [
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": spec.policy_variant_id,
            "iteration": 0,
            "executed_known_alpha_count": 1,
            "supported_point_count": int(prediction_map["model_supported"].sum()),
            "new_supported_point_count": 0,
        }
    ]
    uncertainty_rows: list[dict[str, Any]] = [
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": spec.policy_variant_id,
            "iteration": 0,
            "guard_id": spec.guard_id,
            "uncertainty_bound": uncertainty.bound_used_by_guard,
            "uncertainty_source": uncertainty.bound_type,
            "prospective_outcome_updated_uncertainty": False,
        }
    ]
    exploration_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []
    proposal_truth_rows: list[dict[str, Any]] = []
    stop_reason = ""
    model_update_count = 0
    cumulative_regret = 0.0
    zero_value_run = 0
    previous_uncertainty_bound = uncertainty.bound_used_by_guard

    for iteration in range(1, trial_budget + 1):
        truth_before_proposal = oracle.truth_calls
        local = local_prediction_candidates(prediction_map, current_best, steps)
        guarded = apply_research_decision_guard(local, current_best, uncertainty)
        guarded["iteration"] = iteration
        guarded["policy_id"] = spec.policy_variant_id
        guarded["scenario_name"] = state.scenario_name
        guarded["subject_id"] = state.subject_id
        guarded["selected_for_execution"] = False
        guarded["selection_mode"] = ""
        guarded["guard_id"] = spec.guard_id
        guarded["cumulative_rule_id"] = spec.cumulative_rule_id
        guarded["stopping_rule_id"] = spec.stopping_rule_id
        exploit = select_exploit_candidate(
            guarded, POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
        )
        selected = exploit
        purpose = TRIAL_PURPOSE_EXPLOIT
        frontier_ranked = pd.DataFrame()
        if selected is None:
            frontier = build_local_exploration_frontier(prediction_map, executed_keys)
            if not frontier.empty:
                frontier = frontier.loc[~frontier["model_supported"].astype(bool)].copy()
            frontier_ranked = rank_exploration_frontier(
                frontier, fitting_data, parameters
            )
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
                frontier_ranked["selected_for_exploration"] = frontier_ranked[
                    "trajectory_id"
                ].astype(str).eq(str(selected["trajectory_id"]))
                frontier_ranked["iteration"] = iteration
                frontier_ranked["policy_id"] = spec.policy_variant_id
                frontier_ranked["scenario_name"] = state.scenario_name
                frontier_ranked["subject_id"] = state.subject_id
                frontier_ranked["selection_mode"] = TRIAL_PURPOSE_EXPLORE

        if selected is None:
            if local.empty:
                stop_reason = STOP_NO_GEOMETRICALLY_VALID_CANDIDATE
            else:
                frontier_any = build_local_exploration_frontier(
                    prediction_map, executed_keys
                )
                if frontier_any.empty:
                    stop_reason = STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
                else:
                    unsupported = frontier_any.loc[
                        ~frontier_any["model_supported"].astype(bool)
                    ]
                    stop_reason = (
                        STOP_PATIENT_ENVELOPE_BOUNDARY
                        if not unsupported.empty
                        and not frontier_ranked.empty
                        and not frontier_ranked["exploration_candidate_valid"]
                        .astype(bool)
                        .any()
                        else STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER
                    )
            guarded["policy_decision"] = "STOP"
            guarded["prospective_stop_reason"] = stop_reason
            guard_frames.append(guarded)
            break

        selected_id = str(selected["trajectory_id"])
        selected_alpha = alpha_from_row(selected)
        if selected_alpha.key() == current_best.key():
            raise RuntimeError("current best cannot be re-executed")
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            allowed = {
                alpha.key()
                for alpha in build_coordinate_neighborhood(current_best, steps)
            }
            if selected_alpha.key() not in allowed:
                raise RuntimeError("prospective exploit attempted a nonlocal jump")
            guarded.loc[
                guarded["trajectory_id"].astype(str).eq(selected_id),
                ["selected_for_execution", "selection_mode"],
            ] = [True, purpose]
        guarded["policy_decision"] = purpose
        guarded["prospective_stop_reason"] = ""
        guard_frames.append(guarded)
        if not frontier_ranked.empty:
            guard_frames.append(frontier_ranked)
        if oracle.truth_calls != truth_before_proposal:
            raise RuntimeError("proposal or ranking accessed prospective truth")
        proposal_truth_rows.append(
            {
                "iteration": iteration,
                "truth_calls_before_proposal": truth_before_proposal,
                "truth_calls_after_proposal": oracle.truth_calls,
                "selected_trajectory_id": selected_id,
                "trial_purpose": purpose,
                "manifest_verified_before_truth": True,
            }
        )

        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = selected_id
        token = oracle.declare_selected(selected_id, purpose)
        manifest_gate.record_truth_access()
        execution = oracle.execute(token, trajectory)
        if not execution.observation_valid:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
            break

        actual = _actual_objective(selected_id, execution, reference_metrics)
        predicted_j = float(selected["J_pred"])
        current_predicted_j = float(
            _row_for_alpha(prediction_map, current_best)["J_pred"]
        )
        delta_pred = predicted_j - current_predicted_j
        delta_actual = actual.mechanical_cost_j_rms - best_actual_j
        false_improvement = bool(
            delta_pred < -OBJECTIVE_EQUIVALENCE_TOLERANCE
            and delta_actual >= -OBJECTIVE_EQUIVALENCE_TOLERANCE
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
        trace_before = _parameter_uncertainty_trace(fitting_data, parameters_before)
        adaptation = execution.estimator_observations.copy(deep=True)
        fitting_data = pd.concat((fitting_data, adaptation), ignore_index=True)
        domain_data = pd.concat((domain_data, adaptation), ignore_index=True)
        estimation = _fit_updated_model(fitting_data, parameters_before)
        if estimation.optimizer_success:
            parameters = dict(estimation.estimated_parameters)
            model_update_count += 1
        else:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
        trace_after = _parameter_uncertainty_trace(fitting_data, parameters)
        previous_map = prediction_map
        previous_rank = _predicted_rank_signature(previous_map)
        previous_global_best = previous_rank[0]
        model = _model_for_iteration(state, parameters, domain_data, iteration)
        prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
        next_rank = _predicted_rank_signature(prediction_map)
        next_uncertainty = _uncertainty_for_policy(
            state, parameters, iteration, spec
        )
        new_supported = int(
            prediction_map["model_supported"].sum()
            - previous_map["model_supported"].sum()
        )
        map_changed = _strict_map_changed(previous_map, prediction_map)
        rank_changed = previous_rank != next_rank
        global_best_changed = previous_global_best != next_rank[0]
        parameter_changed = any(
            float(parameters[name]) != float(parameters_before[name])
            for name in PARAMETER_NAMES
        )
        uncertainty_changed = (
            float(next_uncertainty.bound_used_by_guard)
            != float(uncertainty.bound_used_by_guard)
        )
        exploit_available_after = _next_exploit_available(
            prediction_map, current_best, steps, next_uncertainty
        )

        decision_value_zero = bool(
            purpose == TRIAL_PURPOSE_EXPLORE
            and not parameter_changed
            and not map_changed
            and not rank_changed
            and not global_best_changed
            and not uncertainty_changed
            and not exploit_available_after
        )
        if purpose == TRIAL_PURPOSE_EXPLORE:
            zero_value_run = zero_value_run + 1 if decision_value_zero else 0
        else:
            zero_value_run = 0

        if not frontier_ranked.empty and purpose == TRIAL_PURPOSE_EXPLORE:
            selected_frontier = frontier_ranked.loc[
                frontier_ranked["trajectory_id"].astype(str).eq(selected_id)
            ].iloc[0]
            information_gain = float(
                selected_frontier["incremental_log_information_gain"]
            )
            exploration_rows.append(
                {
                    "case_id": f"{state.subject_id}__{state.scenario_name}",
                    "subject_id": state.subject_id,
                    "scenario_name": state.scenario_name,
                    "policy_id": spec.policy_variant_id,
                    "iteration": iteration,
                    "trajectory_id": selected_id,
                    "support_growth": new_supported,
                    "information_gain": information_gain,
                    "parameter_change_observed": parameter_changed,
                    "parameter_delta_norm": float(
                        np.linalg.norm(
                            [parameters[name] - parameters_before[name] for name in PARAMETER_NAMES]
                        )
                    ),
                    "prediction_map_change_observed": map_changed,
                    "predicted_ranking_changed": rank_changed,
                    "predicted_best_trajectory_changed": global_best_changed,
                    "validation_uncertainty_changed": uncertainty_changed,
                    "exploit_eligibility_after": exploit_available_after,
                    "decision_value_zero": decision_value_zero,
                    "consecutive_zero_decision_value": zero_value_run,
                    "support_alone_used_as_decision_value": False,
                    "future_truth_used_by_stopping": False,
                    "future_exploit_used_by_stopping": False,
                    "future_best_used_by_stopping": False,
                }
            )

        map_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.policy_variant_id,
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
                "policy_id": spec.policy_variant_id,
                "iteration": iteration,
                "executed_known_alpha_count": len(executed_keys),
                "supported_point_count": int(prediction_map["model_supported"].sum()),
                "new_supported_point_count": new_supported,
            }
        )
        uncertainty_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.policy_variant_id,
                "iteration": iteration,
                "guard_id": spec.guard_id,
                "uncertainty_bound": next_uncertainty.bound_used_by_guard,
                "uncertainty_source": next_uncertainty.bound_type,
                "prospective_outcome_updated_uncertainty": False,
            }
        )
        parameter_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.policy_variant_id,
                "iteration": iteration,
                "model_update_success": bool(estimation.optimizer_success),
                "parameter_changed_exactly": parameter_changed,
                "uncertainty_trace_before": trace_before,
                "uncertainty_trace_after": trace_after,
                **{f"{name}_before": parameters_before[name] for name in PARAMETER_NAMES},
                **{f"{name}_after": parameters[name] for name in PARAMETER_NAMES},
            }
        )
        history_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.policy_variant_id,
                "policy_role": spec.role,
                "iteration": iteration,
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
                "decision_uncertainty_bound": uncertainty.bound_used_by_guard,
                "improvement_margin": (
                    -delta_pred
                    - uncertainty.bound_used_by_guard
                    - OBJECTIVE_EQUIVALENCE_TOLERANCE
                ),
                "accepted_improvement": accepted,
                "executed_false_improvement": false_improvement,
                "execution_status": (
                    EXECUTED_FALSE_IMPROVEMENT if false_improvement else "EXECUTED"
                ),
                "model_supported": bool(selected["model_supported"]),
                "domain_coverage": float(selected["domain_coverage"]),
                "support_growth": new_supported,
                "decision_value_zero": decision_value_zero,
                "consecutive_zero_decision_value": zero_value_run,
                "cumulative_regret_vs_best_before": cumulative_regret,
                "truth_accessed_before_selection": False,
                "selection_token": token.token,
                "manifest_sha_verified_before_truth": True,
                "stop_reason_after_iteration": stop_reason,
            }
        )
        false_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.policy_variant_id,
                "iteration": iteration,
                "trial_purpose": purpose,
                "trajectory_id": selected_id,
                "delta_J_pred": delta_pred,
                "delta_J_actual": delta_actual,
                "executed_false_improvement": false_improvement,
                "best_updated": accepted,
            }
        )

        if (
            not stop_reason
            and spec.guard_id == "G0_CURRENT_GLOBAL_MAX"
            and next_uncertainty.bound_used_by_guard
            > previous_uncertainty_bound + OBJECTIVE_EQUIVALENCE_TOLERANCE
        ):
            stop_reason = STOP_MODEL_ADEQUACY_DEGRADED
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
        if (
            not stop_reason
            and purpose == TRIAL_PURPOSE_EXPLORE
            and spec.stopping_k is not None
            and zero_value_run >= spec.stopping_k
            and not exploit_available_after
        ):
            stop_reason = STOP_DECISION_VALUE_K
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
            final_local = local_prediction_candidates(prediction_map, current_best, steps)
            final_guard = apply_research_decision_guard(
                final_local, current_best, next_uncertainty
            )
            final_guard["iteration"] = iteration + 1
            final_guard["policy_id"] = spec.policy_variant_id
            final_guard["scenario_name"] = state.scenario_name
            final_guard["subject_id"] = state.subject_id
            final_guard["selected_for_execution"] = False
            final_guard["selection_mode"] = "STOP"
            final_guard["policy_decision"] = "STOP"
            final_guard["prospective_stop_reason"] = stop_reason
            final_guard["guard_id"] = spec.guard_id
            final_guard["cumulative_rule_id"] = spec.cumulative_rule_id
            final_guard["stopping_rule_id"] = spec.stopping_rule_id
            guard_frames.append(final_guard)
        previous_uncertainty_bound = next_uncertainty.bound_used_by_guard
        uncertainty = next_uncertainty
        if stop_reason:
            break

    if not stop_reason:
        stop_reason = STOP_MAX_PERSONALIZATION_TRIALS
        if history_rows:
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason

    history = pd.DataFrame(history_rows)
    guard_audit = (
        pd.concat(guard_frames, ignore_index=True, sort=False)
        if guard_frames
        else pd.DataFrame()
    )
    number_executed = len(history)
    summary = {
        "case_id": f"{state.subject_id}__{state.scenario_name}",
        "subject_id": state.subject_id,
        "scenario_name": state.scenario_name,
        "policy_id": spec.policy_variant_id,
        "policy_role": spec.role,
        "guard_id": spec.guard_id,
        "cumulative_rule_id": spec.cumulative_rule_id,
        "stopping_rule_id": spec.stopping_rule_id,
        "stopping_k": spec.stopping_k,
        "research_status": RESEARCH_ONLY,
        "number_of_executed_trials": number_executed,
        "number_of_exploit_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT).sum()) if number_executed else 0,
        "number_of_explore_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLORE).sum()) if number_executed else 0,
        "number_of_executed_false_improvements": int(history["executed_false_improvement"].sum()) if number_executed else 0,
        "reference_actual_J": 1.0,
        "final_best_actual_J": best_actual_j,
        "actual_J_reduction_from_reference": 1.0 - best_actual_j,
        "cumulative_regret_vs_best_before": cumulative_regret,
        "model_update_count": model_update_count,
        "final_best_alpha_hip": current_best.hip_delta_deg,
        "final_best_alpha_knee": current_best.knee_delta_deg,
        "final_best_alpha_phase": current_best.phase_delta,
        "initial_supported_point_count": int(map_rows[0]["supported_point_count"]),
        "final_supported_point_count": int(prediction_map["model_supported"].sum()),
        "known_region_growth": int(prediction_map["model_supported"].sum()) - int(map_rows[0]["supported_point_count"]),
        "low_decision_value_exploration_count": int(sum(bool(row["decision_value_zero"]) for row in exploration_rows)),
        "stop_reason": stop_reason,
        "trial_budget": trial_budget,
        "whole_map_recomputation_count": len(map_rows),
        "truth_calls_including_reference_normalization": oracle.truth_calls,
        "heldout_final_test_used": False,
        "prospective_truth_updated_local_calibration": False,
        "support_used_as_reliability_score": False,
        "P2_V2_default_enabled": False,
        "human_ready": False,
        "robot_motion_approved": False,
        "model_reliability_status": (
            MODEL_RELIABILITY_DEGRADED
            if stop_reason == STOP_MODEL_ADEQUACY_DEGRADED
            else "RESEARCH_DIAGNOSTIC_NOT_FORMALLY_RELIABLE"
        ),
    }
    truth_audit = {
        "proposal_truth_call_audit": proposal_truth_rows,
        "truth_calls_unchanged_during_every_proposal": all(
            row["truth_calls_before_proposal"] == row["truth_calls_after_proposal"]
            for row in proposal_truth_rows
        ),
        "oracle_event_audit": oracle.audit.to_dict(orient="records"),
        "manifest_verified_before_every_truth": True,
        "heldout_final_test_used": False,
        "post_policy_truth_role": MAP_TRUTH_ROLE,
    }
    return PolicyRunResult(
        subject_id=state.subject_id,
        scenario_name=state.scenario_name,
        policy_id=spec.policy_variant_id,
        trial_history=history,
        decision_guard_audit=guard_audit,
        parameter_history=pd.DataFrame(parameter_rows),
        prediction_map_history=pd.DataFrame(map_rows),
        known_region_history=pd.DataFrame(known_rows),
        uncertainty_history=pd.DataFrame(uncertainty_rows),
        uncertainty_pairwise_audit=pd.DataFrame(),
        exploration_information_gain=pd.DataFrame(exploration_rows),
        false_improvement_audit=pd.DataFrame(false_rows),
        summary=summary,
        initial_prediction_map=initial_prediction_map,
        final_prediction_map=prediction_map,
        truth_access_audit=truth_audit,
    )


def post_policy_local_truth_audit(
    result: PolicyRunResult,
    state: InitialResearchState,
    cache: TrajectoryComponentCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach local truth only after the complete policy path is immutable."""

    local_audit = result.decision_guard_audit.loc[
        result.decision_guard_audit["decision_guard_status"].notna()
    ].copy()
    if local_audit.empty:
        return pd.DataFrame(), pd.DataFrame()
    model = _model_for_iteration(state, state.parameters, state.domain_data, 0)
    neutral = result.initial_prediction_map.loc[
        np.isclose(result.initial_prediction_map["hip_delta"], 0.0)
        & np.isclose(result.initial_prediction_map["knee_delta"], 0.0)
        & np.isclose(result.initial_prediction_map["phase_delta"], 0.0)
    ].iloc[0]
    candidate_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    for iteration, local in local_audit.groupby("iteration", sort=True):
        if local["trajectory_id"].duplicated().any():
            raise RuntimeError("post-policy local audit contains duplicate candidates")
        evaluation = local.copy()
        if not evaluation["trajectory_id"].astype(str).eq(str(neutral["trajectory_id"])).any():
            evaluation = pd.concat((evaluation, neutral.to_frame().T), ignore_index=True)
        truth, _ = evaluate_truth_map(evaluation, model, cache, batch_size=64)
        current = local.loc[
            local["decision_guard_status"].eq(CURRENT_BEST_NOT_A_CANDIDATE)
        ]
        if len(current) != 1:
            raise RuntimeError("post-policy local audit lacks one current point")
        current_id = str(current.iloc[0]["trajectory_id"])
        current_truth = float(
            truth.loc[truth["trajectory_id"].astype(str).eq(current_id), "J_truth"].iloc[0]
        )
        joined = local.merge(
            truth[["trajectory_id", "J_truth"]], on="trajectory_id", validate="one_to_one"
        )
        joined["delta_J_truth_vs_current"] = joined["J_truth"] - current_truth
        joined["is_current"] = joined["trajectory_id"].astype(str).eq(current_id)
        joined["true_local_improvement"] = (
            joined["delta_J_truth_vs_current"] < -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        candidates = joined.loc[~joined["is_current"]].copy()
        guard_passed = bool(joined["research_exploit_eligible"].astype(bool).any())
        true_available = bool(candidates["true_local_improvement"].any())
        missed = bool(not guard_passed and true_available)
        decision = str(local["policy_decision"].iloc[0])
        round_rows.append(
            {
                "case_id": result.summary["case_id"],
                "subject_id": result.subject_id,
                "scenario_name": result.scenario_name,
                "policy_id": result.policy_id,
                "iteration": int(iteration),
                "policy_decision": decision,
                "true_local_improvement_available": true_available,
                "missed_improvement_round": missed,
                "best_local_delta_J_truth": float(candidates["delta_J_truth_vs_current"].min()) if not candidates.empty else np.nan,
                "policy_complete_before_truth_audit": True,
                "truth_fed_back_to_policy": False,
            }
        )
        for row in candidates.to_dict(orient="records"):
            candidate_rows.append(
                {
                    "case_id": result.summary["case_id"],
                    "subject_id": result.subject_id,
                    "scenario_name": result.scenario_name,
                    "policy_id": result.policy_id,
                    "iteration": int(iteration),
                    "policy_decision": decision,
                    "trajectory_id": row["trajectory_id"],
                    "alpha_hip": row["hip_delta"],
                    "alpha_knee": row["knee_delta"],
                    "alpha_phase": row["phase_delta"],
                    "delta_J_pred": row["delta_J_pred_vs_current"],
                    "delta_J_truth": row["delta_J_truth_vs_current"],
                    "guard_status": row["decision_guard_status"],
                    "true_local_improvement": row["true_local_improvement"],
                    "missed_improvement": bool(missed and row["true_local_improvement"]),
                    "policy_complete_before_truth_audit": True,
                    "truth_fed_back_to_policy": False,
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(round_rows)


def evaluate_full_truth_landscape(
    result: PolicyRunResult,
    state: InitialResearchState,
    cache: TrajectoryComponentCache,
) -> pd.DataFrame:
    model = _model_for_iteration(state, state.parameters, state.domain_data, 0)
    evaluated, _ = evaluate_truth_map(result.initial_prediction_map, model, cache)
    return evaluated


def small_step_accumulation_audit(
    truth_landscape: pd.DataFrame,
    case_id: str,
) -> pd.DataFrame:
    lookup = {
        (
            round(float(row["hip_delta"]), 12),
            round(float(row["knee_delta"]), 12),
            round(float(row["phase_delta"]), 12),
        ): row
        for row in truth_landscape.to_dict(orient="records")
    }
    origin = (0.0, 0.0, 0.0)
    rows: list[dict[str, Any]] = []
    steps = ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.0025))
    names = ("hip", "knee", "phase")
    for name, step in zip(names, steps):
        for sign in (-1.0, 1.0):
            for length in (2, 3, 5):
                keys = [
                    tuple(round(origin[j] + sign * index * step[j], 12) for j in range(3))
                    for index in range(length + 1)
                ]
                if any(key not in lookup for key in keys):
                    continue
                values = [float(lookup[key]["J_truth"]) for key in keys]
                deltas = np.diff(values)
                occurrence = bool(
                    np.all(deltas < 0.0)
                    and np.all(np.abs(deltas) < OBJECTIVE_EQUIVALENCE_TOLERANCE)
                    and values[-1] - values[0] < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "coordinate": name,
                        "direction": "POSITIVE" if sign > 0 else "NEGATIVE",
                        "bundle_length": length,
                        "start_alpha": str(keys[0]),
                        "end_alpha": str(keys[-1]),
                        "single_step_delta_J_max_abs": float(np.max(np.abs(deltas))),
                        "endpoint_delta_J": values[-1] - values[0],
                        "small_step_accumulation_case": occurrence,
                        "bundle_rule_status": "SHADOW_ONLY_NOT_CALIBRATED",
                        "truth_used_to_create_bundle": False,
                    }
                )
    return pd.DataFrame(rows)


def classify_final_status(policy_summary: pd.DataFrame) -> str:
    """Apply the manifest's pre-registered aggregate rule without tuning."""

    primary = policy_summary.loc[
        policy_summary["policy_id"].isin(
            ("P2_V1_G0_C0_S0", "P2_V2A_G2_C0_S2")
        )
    ]
    grouped = primary.groupby("policy_id", sort=False).agg(
        missed=("missed_improvement_rounds", "sum"),
        false=("number_of_executed_false_improvements", "sum"),
        final_j=("final_best_actual_J", "mean"),
        regret=("global_truth_regret", "mean"),
        trials=("number_of_executed_trials", "mean"),
    )
    if set(grouped.index) != {"P2_V1_G0_C0_S0", "P2_V2A_G2_C0_S2"}:
        return FINAL_INSUFFICIENT
    v1 = grouped.loc["P2_V1_G0_C0_S0"]
    v2 = grouped.loc["P2_V2A_G2_C0_S2"]
    if (
        int(v2["false"]) > int(v1["false"])
        or float(v2["final_j"]) > float(v1["final_j"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
        or float(v2["regret"]) > float(v1["regret"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
    ):
        return FINAL_REJECTS
    if (
        int(v2["missed"]) <= int(v1["missed"])
        and int(v2["false"]) <= int(v1["false"])
        and float(v2["final_j"]) <= float(v1["final_j"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
        and float(v2["regret"]) <= float(v1["regret"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
        and float(v2["trials"]) <= float(v1["trials"])
    ):
        return FINAL_SUPPORTS
    return FINAL_INSUFFICIENT


__all__ = [
    "DEVELOPMENT_CASES",
    "EXPECTED_GEOMETRIC_LATTICE_SIZE",
    "FINAL_INSUFFICIENT",
    "FINAL_REJECTS",
    "FINAL_STATUSES",
    "FINAL_SUPPORTS",
    "FrozenManifestGate",
    "GLOBAL_MODEL_RELIABILITY_STATUS",
    "HELD_OUT_FINAL_TEST",
    "INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS",
    "LOCAL_MAX",
    "LOCAL_P95",
    "LOCAL_P99",
    "MANIFEST_ID",
    "MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_MOTION_APPROVED",
    "OFFLINE_METHOD_REQUIRES_REVISION",
    "P2_V2_DEFAULT_ENABLED",
    "POLICY_VARIANTS",
    "PROSPECTIVE_STATUS",
    "PROTOCOL_ID",
    "ProspectivePolicySpec",
    "audit_bundle_uncertainty",
    "build_prospective_manifest",
    "classify_final_status",
    "dynamic_subject_for_id",
    "evaluate_full_truth_landscape",
    "post_policy_local_truth_audit",
    "prospective_case_rows",
    "prospective_subject_definitions",
    "registered_prospective_subject",
    "run_prospective_policy",
    "small_step_accumulation_audit",
    "stable_manifest_sha256",
    "validate_frozen_local_evidence",
]
