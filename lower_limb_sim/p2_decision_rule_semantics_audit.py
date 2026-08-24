"""Default-off decision-rule semantics audit for P2 research.

This module does not register or replace a policy.  It separates the frozen
0.005 meaningful-improvement magnitude check from independently calibrated
model-direction evidence, then runs only development shadow comparators behind
a manifest SHA gate.
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
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    CURRENT_BEST_NOT_A_CANDIDATE,
    EXECUTED_FALSE_IMPROVEMENT,
    GEOMETRICALLY_INADMISSIBLE,
    MODEL_RELIABILITY_DEGRADED,
    NO_INDEPENDENT_VALIDATION_EVIDENCE,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    RESEARCH_ONLY,
    RESEARCH_EXPLOIT_ELIGIBLE,
    STOP_MAX_PERSONALIZATION_TRIALS,
    STOP_MODEL_ADEQUACY_DEGRADED,
    STOP_MODEL_UPDATE_FAILURE,
    STOP_NO_GEOMETRICALLY_VALID_CANDIDATE,
    STOP_NO_RELIABLE_IMPROVEMENT_NO_USEFUL_FRONTIER,
    STOP_PATIENT_ENVELOPE_BOUNDARY,
    TRIAL_PURPOSE_EXPLOIT,
    TRIAL_PURPOSE_EXPLORE,
    UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE,
    InitialResearchState,
    PolicyRunResult,
    ResearchDecisionUncertainty,
    SelectionGatedVirtualTruthOracle,
    _actual_objective,
    _fit_updated_model,
    _model_for_iteration,
    _row_for_alpha,
    alpha_from_row,
    apply_research_decision_guard,
    build_local_exploration_frontier,
    evaluate_validation_pairwise_uncertainty,
    local_prediction_candidates,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .safeguarded_sequential_initial_identification import (
    default_virtual_patient_envelope,
)
from .sequential_personalization import (
    INITIAL_STEP_HIP_DEG,
    INITIAL_STEP_KNEE_DEG,
    INITIAL_STEP_PHASE,
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
    SearchAlpha,
    TrustRegionSteps,
    accept_actual_trial,
    build_coordinate_neighborhood,
    shrink_steps,
)


AUDIT_ID = "P2_DECISION_RULE_SEMANTICS_AUDIT_V1"
MANIFEST_ID = "DECISION_RULE_SEMANTICS_CANDIDATE_MANIFEST_V1"
ADDITIVE_ASSUMPTION = "ADDITIVE_MARGIN_IS_DESIGN_ASSUMPTION"
PRIMARY_BLOCKER = "ADDITIVE_MARGIN_SEMANTICS_IS_PRIMARY_BLOCKER"
NOT_SUFFICIENT = "DECISION_SEMANTICS_NOT_SUFFICIENT_TO_EXPLAIN_FAILURE"
MORE_EVIDENCE = "MORE_EVIDENCE_REQUIRED"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"
DEFAULT_ENABLED = False
CALIBRATED_BUNDLE_LENGTHS = (2, 3, 5)
MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON = 20
EXPECTED_CALIBRATION_MANIFEST_SHA256 = (
    "08f930692704c24f10f85f094eabf45fc5e0842ec3f479345e62bee892df1729"
)
EXPECTED_BUNDLE_PAIR_PLAN_SHA256 = (
    "3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84"
)
EXPECTED_SMALL_STEP_SOURCE_SHA256 = (
    "4fd0f87335ff6a4e114d17410bf1d5c6acc2f97e142600dea52ee4e6569752d9"
)

MODULE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_next_revision_independent_calibration_v1"
)
CALIBRATION_MANIFEST_PATH = (
    CALIBRATION_DIRECTORY / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
)
POST_REJECTION_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "post_prospective_rejection_root_cause_audit_v1"
)
BUNDLE_PLAN_PATH = POST_REJECTION_DIRECTORY / "designated_bundle_validation_pair_plan.csv"
SMALL_STEP_SOURCE_PATH = POST_REJECTION_DIRECTORY / "prospective_small_step_accumulation.csv"

_AXES = ("hip", "knee", "phase")
_AXIS_INDEX = {"hip": 0, "knee": 1, "phase": 2}
_GRID_STEP = {
    "hip": GRID_HIP_STEP_DEG,
    "knee": GRID_KNEE_STEP_DEG,
    "phase": GRID_PHASE_STEP,
}
_TRUST_STEPS = {
    "hip": {
        "INITIAL": INITIAL_STEP_HIP_DEG,
        "HALF": INITIAL_STEP_HIP_DEG / 2.0,
        "MINIMUM": MINIMUM_STEP_HIP_DEG,
    },
    "knee": {
        "INITIAL": INITIAL_STEP_KNEE_DEG,
        "HALF": INITIAL_STEP_KNEE_DEG / 2.0,
        "MINIMUM": MINIMUM_STEP_KNEE_DEG,
    },
    "phase": {
        "INITIAL": INITIAL_STEP_PHASE,
        "HALF": INITIAL_STEP_PHASE / 2.0,
        "MINIMUM": MINIMUM_STEP_PHASE,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
class SemanticSpec:
    semantic_id: str
    direct_rule: str
    bundle_rule: str | None
    role: str
    default_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SEMANTIC_VARIANTS = (
    SemanticSpec(
        "S0_CURRENT_ADDITIVE_MARGIN",
        "CURRENT_DYNAMIC_U_ADDITIVE",
        "BUNDLE_SCALE_P95_ADDITIVE",
        "CURRENT_RESEARCH_CANDIDATE_COMPARATOR",
    ),
    SemanticSpec(
        "S1_TWO_GATE_DIRECTION_AND_MAGNITUDE",
        "CALIBRATION_SIGN_MAJORITY_AND_MAGNITUDE",
        None,
        "CATEGORICAL_DIRECTION_EVIDENCE_COMPARATOR",
    ),
    SemanticSpec(
        "S2_UNCERTAINTY_INTERVAL_DIRECTION_GATE",
        "ONE_STEP_P95_INTERVAL_AND_MAGNITUDE",
        None,
        "TRANSPARENT_INTERVAL_COMPARATOR",
    ),
    SemanticSpec(
        "S3_BUNDLE_ENDPOINT_TWO_GATE",
        "ONE_STEP_P95_INTERVAL_AND_MAGNITUDE",
        "BUNDLE_SCALE_P95_INTERVAL_AND_MAGNITUDE",
        "CUMULATIVE_ENDPOINT_SEMANTICS_COMPARATOR",
    ),
)


@dataclass(frozen=True)
class SemanticsCalibration:
    one_step_p95: float
    bundle_scale_p95: Mapping[int, float]
    direction_evidence: pd.DataFrame
    one_step_pair_count: int
    bundle_pair_count_by_scale: Mapping[int, int]


def load_semantics_calibration() -> SemanticsCalibration:
    """Load independent residual evidence, never policy-performance cases."""

    if sha256_file(CALIBRATION_MANIFEST_PATH) != EXPECTED_CALIBRATION_MANIFEST_SHA256:
        raise RuntimeError("independent calibration manifest SHA changed")
    if sha256_file(BUNDLE_PLAN_PATH) != EXPECTED_BUNDLE_PAIR_PLAN_SHA256:
        raise RuntimeError("designated bundle pair plan SHA changed")
    if sha256_file(SMALL_STEP_SOURCE_PATH) != EXPECTED_SMALL_STEP_SOURCE_SHA256:
        raise RuntimeError("historical small-step audit source SHA changed")

    one = pd.read_csv(CALIBRATION_DIRECTORY / "independent_one_step_residuals.csv")
    if len(one) != 324:
        raise RuntimeError("independent one-step calibration count changed")
    if one["used_by_policy"].astype(bool).any() or one[
        "heldout_final_test_used"
    ].astype(bool).any():
        raise RuntimeError("calibration role flags changed")
    one["direction_agreement"] = np.sign(one["deltaJ_pred"].astype(float)) == np.sign(
        one["deltaJ_truth"].astype(float)
    )
    evidence = (
        one.groupby(["coordinate", "trust_level"], as_index=False, sort=True)
        .agg(
            calibration_pair_count=("pair_id", "count"),
            direction_support_count=("direction_agreement", "sum"),
        )
    )
    evidence["direction_contradiction_count"] = (
        evidence["calibration_pair_count"] - evidence["direction_support_count"]
    )
    evidence["direction_supported_by_majority"] = (
        evidence["direction_support_count"]
        > evidence["direction_contradiction_count"]
    )
    evidence["majority_rule"] = "support_count > contradiction_count"
    evidence["probability_claimed"] = False
    evidence["threshold_tuned_on_development"] = False

    summary = pd.read_csv(CALIBRATION_DIRECTORY / "one_step_residual_summary.csv")
    overall = summary.loc[summary["summary_scope"].eq("OVERALL")]
    if len(overall) != 1:
        raise RuntimeError("one-step residual summary changed")

    bundle_p95: dict[int, float] = {}
    bundle_counts: dict[int, int] = {}
    bundle_summary = pd.read_csv(CALIBRATION_DIRECTORY / "bundle_residual_summary.csv")
    for length in CALIBRATED_BUNDLE_LENGTHS:
        label = f"{length}-step"
        row = bundle_summary.loc[
            bundle_summary["summary_scope"].eq("OVERALL")
            & bundle_summary["decision_scale"].eq(label)
        ]
        if len(row) != 1:
            raise RuntimeError(f"missing {label} independent residual summary")
        residuals = pd.read_csv(
            CALIBRATION_DIRECTORY / f"independent_bundle_{length}step_residuals.csv"
        )
        if residuals["used_by_policy"].astype(bool).any() or residuals[
            "heldout_final_test_used"
        ].astype(bool).any():
            raise RuntimeError(f"{label} calibration role flags changed")
        if len(residuals) != int(row.iloc[0]["n"]):
            raise RuntimeError(f"{label} independent residual count changed")
        bundle_p95[length] = float(row.iloc[0]["P95"])
        bundle_counts[length] = len(residuals)
    return SemanticsCalibration(
        one_step_p95=float(overall.iloc[0]["P95"]),
        bundle_scale_p95=bundle_p95,
        direction_evidence=evidence,
        one_step_pair_count=len(one),
        bundle_pair_count_by_scale=bundle_counts,
    )


def candidate_manifest_payload(
    calibration: SemanticsCalibration,
    *,
    checkpoint_commit: str,
    protected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return the complete comparator definition before development truth."""

    return {
        "manifest_id": MANIFEST_ID,
        "audit_id": AUDIT_ID,
        "status": "FROZEN_BEFORE_DEVELOPMENT_SHADOW_TRUTH",
        "checkpoint_commit": checkpoint_commit,
        "default_enabled": False,
        "P2_V1_replaced": False,
        "policy_implemented": False,
        "prospective_cohort_run": False,
        "heldout_final_test_read_allowed": False,
        "data_roles": {
            "development": "ORIGINAL_9_PLUS_POST_REJECTION_DEVELOPMENT_6_SHADOW_ONLY",
            "independent_calibration": "RESIDUAL_AND_DIRECTION_EVIDENCE_ONLY",
            "calibration_policy_performance": "FORBIDDEN",
            "heldout_final_test": "NOT_READ",
            "prospective": "NOT_RUN",
        },
        "current_semantics": {
            "meaningful_improvement_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
            "tolerance_role": "MECHANICAL_OBJECTIVE_EQUIVALENCE_AND_MINIMUM_MEANINGFUL_MAGNITUDE",
            "uncertainty_role": "EMPIRICAL_ABSOLUTE_ERROR_BOUND_ON_PREDICTED_DELTA_J",
            "current_direct_rule": "predicted_improvement > 0.005 + current_global_max_U",
            "current_bundle_candidate_rule": "predicted_endpoint_improvement > 0.005 + scale_P95_U",
            "theoretical_or_preregistered_necessity_for_addition_found": False,
            "classification": ADDITIVE_ASSUMPTION,
        },
        "semantic_variants": [item.as_dict() for item in SEMANTIC_VARIANTS],
        "rules": {
            "S0": {
                "gate": "I_pred > 0.005 + U",
                "direct_U": "CURRENT_DYNAMIC_GLOBAL_PAIRWISE_MAX",
                "bundle_U": "INDEPENDENT_SCALE_P95",
            },
            "S1": {
                "magnitude_gate": "I_pred > 0.005",
                "direction_gate": "same_axis_and_trust_level_calibration_support_count > contradiction_count",
                "direction_source_pair_count": calibration.one_step_pair_count,
                "probability_claimed": False,
            },
            "S2": {
                "magnitude_gate": "I_pred > 0.005",
                "direction_interval": "[deltaJ_pred-U_P95, deltaJ_pred+U_P95]",
                "direction_gate": "deltaJ_pred + U_P95 < 0",
                "one_step_P95": calibration.one_step_p95,
                "confidence_or_probability_claimed": False,
            },
            "S3": {
                "direct_gate": "S2",
                "bundle_magnitude_gate": "I_endpoint_pred > 0.005",
                "bundle_direction_gate": "deltaJ_endpoint_pred + U_scale_P95 < 0",
                "bundle_lengths": list(CALIBRATED_BUNDLE_LENGTHS),
                "bundle_scale_P95": {
                    str(k): v for k, v in calibration.bundle_scale_p95.items()
                },
                "authorization_scope": "NEXT_ONE_FORMAL_GRID_STEP_ONLY",
                "authorization_expires_after_execution": True,
                "model_refit_after_every_execution": True,
                "whole_map_recomputed_after_every_execution": True,
            },
        },
        "S1_direction_evidence": calibration.direction_evidence.to_dict(
            orient="records"
        ),
        "selection_and_interpretation_frozen": {
            "direct_selection": "lowest_J_pred_then_trajectory_id_among_authorized",
            "bundle_selection": "largest_min_of_magnitude_and_direction_margin_then_shortest_length_then_id",
            "primary_blocker_if": [
                "S3_recovers_more_of_9_paths_than_S0",
                "S3_has_more_bundle_authorizations_than_S0",
                "S3_missed_improvement_not_higher_than_S0",
                "S3_total_false_improvement_not_higher_than_S0",
                "S3_mean_final_J_not_worse_than_S0_by_0.005",
                "S3_mean_regret_not_worse_than_S0_by_0.005",
            ],
            "not_sufficient_if": "S3_does_not_recover_more_of_9_paths_than_S0",
            "otherwise": MORE_EVIDENCE,
            "development_results_may_change_gate": False,
            "percentile_may_change_after_shadow": False,
            "tolerance_may_change_after_shadow": False,
            "variant_may_be_added_after_shadow": False,
        },
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA256,
        "bundle_pair_plan_sha256": EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
        "small_step_source_sha256": EXPECTED_SMALL_STEP_SOURCE_SHA256,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "protected_source_sha256": dict(protected_source_sha256),
        "truth_used_to_create_or_select_semantics": False,
        "human_ready": NOT_HUMAN_READY,
        "robot_motion_approved": NOT_ROBOT_MOTION_APPROVED,
    }


class FrozenSemanticsManifestGate:
    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = str(expected_sha256)
        self.truth_access_count = 0
        self.truth_access_stages: list[str] = []

    def require_frozen(self) -> None:
        if not self.path.is_file() or sha256_file(self.path) != self.expected_sha256:
            raise PermissionError("development truth requires frozen semantics manifest")

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


def _candidate_axis_and_trust(
    row: Mapping[str, Any], current: SearchAlpha
) -> tuple[str | None, str | None]:
    delta = np.asarray(
        (float(row["hip_delta"]), float(row["knee_delta"]), float(row["phase_delta"]))
    ) - np.asarray(current.key())
    changed = np.flatnonzero(np.abs(delta) > 1e-12)
    if len(changed) != 1:
        return None, None
    axis = _AXES[int(changed[0])]
    distance = abs(float(delta[changed[0]]))
    for trust, expected in _TRUST_STEPS[axis].items():
        if np.isclose(distance, expected, atol=1e-12, rtol=0.0):
            return axis, trust
    return axis, None


def apply_semantic_decision_guard(
    local_candidates: pd.DataFrame,
    current: SearchAlpha,
    dynamic_uncertainty: ResearchDecisionUncertainty,
    spec: SemanticSpec,
    calibration: SemanticsCalibration,
) -> pd.DataFrame:
    """Apply a frozen research comparator without changing the P2 function."""

    output = apply_research_decision_guard(
        local_candidates, current, dynamic_uncertainty
    )
    if output.empty:
        return output
    evidence_lookup = {
        (str(row["coordinate"]), str(row["trust_level"])): row
        for row in calibration.direction_evidence.to_dict(orient="records")
    }
    gate_a: list[bool] = []
    gate_b: list[bool] = []
    statuses: list[str] = []
    eligible: list[bool] = []
    axes: list[str] = []
    trusts: list[str] = []
    direction_support: list[float] = []
    direction_contradiction: list[float] = []
    semantic_u: list[float] = []

    for row in output.to_dict(orient="records"):
        is_current = _key((row["hip_delta"], row["knee_delta"], row["phase_delta"])) == current.key()
        improvement = float(row["predicted_improvement_magnitude"])
        axis, trust = _candidate_axis_and_trust(row, current)
        axes.append(axis or "")
        trusts.append(trust or "")
        evidence = evidence_lookup.get((axis, trust)) if axis and trust else None
        support_count = float(evidence["direction_support_count"]) if evidence else np.nan
        contradiction_count = (
            float(evidence["direction_contradiction_count"]) if evidence else np.nan
        )
        direction_support.append(support_count)
        direction_contradiction.append(contradiction_count)
        magnitude_ok = improvement > OBJECTIVE_EQUIVALENCE_TOLERANCE
        if spec.direct_rule == "CURRENT_DYNAMIC_U_ADDITIVE":
            bound = float(dynamic_uncertainty.bound_used_by_guard)
            direction_ok = improvement > bound
            semantic_ok = improvement > OBJECTIVE_EQUIVALENCE_TOLERANCE + bound
        elif spec.direct_rule == "CALIBRATION_SIGN_MAJORITY_AND_MAGNITUDE":
            bound = np.nan
            direction_ok = bool(evidence is not None and support_count > contradiction_count)
            semantic_ok = magnitude_ok and direction_ok
        elif spec.direct_rule == "ONE_STEP_P95_INTERVAL_AND_MAGNITUDE":
            bound = float(calibration.one_step_p95)
            direction_ok = float(row["delta_J_pred_vs_current"]) + bound < 0.0
            semantic_ok = magnitude_ok and direction_ok
        else:
            raise ValueError(f"unknown direct semantic rule: {spec.direct_rule}")
        semantic_u.append(bound)
        gate_a.append(magnitude_ok)
        gate_b.append(direction_ok)

        if is_current:
            status = CURRENT_BEST_NOT_A_CANDIDATE
        elif not bool(row["geometrically_admissible"]):
            status = GEOMETRICALLY_INADMISSIBLE
        elif dynamic_uncertainty.validation_pair_count < 1:
            status = NO_INDEPENDENT_VALIDATION_EVIDENCE
        elif not bool(row["current_model_supported"]) or not bool(
            row["model_supported"]
        ):
            status = UNSUPPORTED_PROVENANCE_NOT_EXPLOIT_ELIGIBLE
        elif not magnitude_ok:
            status = "SEMANTIC_MAGNITUDE_GATE_FAILED"
        elif not direction_ok:
            status = "SEMANTIC_DIRECTION_GATE_FAILED"
        elif semantic_ok:
            status = RESEARCH_EXPLOIT_ELIGIBLE
        else:
            status = "SEMANTIC_RULE_NOT_AUTHORIZED"
        statuses.append(status)
        eligible.append(status == RESEARCH_EXPLOIT_ELIGIBLE)

    output["semantic_id"] = spec.semantic_id
    output["semantic_rule"] = spec.direct_rule
    output["candidate_coordinate"] = axes
    output["candidate_trust_level"] = trusts
    output["magnitude_gate_threshold"] = OBJECTIVE_EQUIVALENCE_TOLERANCE
    output["magnitude_gate_pass"] = gate_a
    output["direction_gate_pass"] = gate_b
    output["semantic_uncertainty_bound"] = semantic_u
    output["direction_support_count"] = direction_support
    output["direction_contradiction_count"] = direction_contradiction
    output["decision_guard_status"] = statuses
    output["research_exploit_eligible"] = eligible
    output["support_alone_approved_exploit"] = False
    output["formal_personalization_approval"] = False
    output["truth_used_for_semantic_gate"] = False
    return output


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


def evaluate_bundle_semantics(
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    spec: SemanticSpec,
    calibration: SemanticsCalibration,
    *,
    iteration: int,
    patient_validity_cache: dict[tuple[float, float, float], bool],
) -> pd.DataFrame:
    if spec.bundle_rule is None:
        return pd.DataFrame()
    lookup = _map_lookup(prediction_map)
    start = current.key()
    if start not in lookup:
        raise RuntimeError("current alpha missing from prediction map")
    start_j = float(lookup[start]["J_pred"])
    rows: list[dict[str, Any]] = []
    for axis in _AXES:
        axis_index = _AXIS_INDEX[axis]
        for direction, sign in (("NEGATIVE", -1.0), ("POSITIVE", 1.0)):
            for length in CALIBRATED_BUNDLE_LENGTHS:
                path = []
                for step_number in range(length + 1):
                    point = list(start)
                    point[axis_index] += sign * _GRID_STEP[axis] * step_number
                    path.append(_key(point))
                path_exists = all(point in lookup for point in path)
                path_rows = [lookup[point] for point in path] if path_exists else []
                geometric = bool(
                    path_exists
                    and all(bool(row["geometrically_admissible"]) for row in path_rows)
                )
                provenance = bool(
                    path_exists
                    and all(
                        str(row["parent_reference_sha256"])
                        == ACTIVE_REFERENCE_SHA256
                        for row in path_rows
                    )
                )
                supported = bool(
                    path_exists
                    and all(bool(row["model_supported"]) for row in path_rows)
                    and all(
                        float(row["domain_coverage"])
                        >= MODEL_SUPPORT_COVERAGE_GATE_PERCENT
                        for row in path_rows
                    )
                )
                patient = bool(
                    path_exists
                    and all(
                        _patient_valid(point, patient_validity_cache) for point in path
                    )
                )
                endpoint_j = float(path_rows[-1]["J_pred"]) if path_exists else np.nan
                delta = endpoint_j - start_j if path_exists else np.nan
                improvement = -delta if path_exists else np.nan
                bound = float(calibration.bundle_scale_p95[length])
                magnitude_margin = improvement - OBJECTIVE_EQUIVALENCE_TOLERANCE
                direction_margin = improvement - bound
                additive_margin = (
                    improvement - OBJECTIVE_EQUIVALENCE_TOLERANCE - bound
                )
                gate_a = bool(path_exists and magnitude_margin > 0.0)
                gate_b = bool(path_exists and direction_margin > 0.0)
                base_valid = geometric and provenance and supported and patient
                if spec.bundle_rule == "BUNDLE_SCALE_P95_ADDITIVE":
                    authorized = bool(base_valid and additive_margin > 0.0)
                    selection_margin = additive_margin
                elif spec.bundle_rule == "BUNDLE_SCALE_P95_INTERVAL_AND_MAGNITUDE":
                    authorized = bool(base_valid and gate_a and gate_b)
                    selection_margin = min(magnitude_margin, direction_margin)
                else:
                    raise ValueError(f"unknown bundle semantic rule: {spec.bundle_rule}")
                identity = f"{spec.semantic_id}|{iteration}|{start}|{axis}|{direction}|{length}"
                authorization_id = "semantic_bundle_" + hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
                rows.append(
                    {
                        "authorization_id": authorization_id,
                        "semantic_id": spec.semantic_id,
                        "semantic_rule": spec.bundle_rule,
                        "iteration": iteration,
                        "coordinate": axis,
                        "direction": direction,
                        "bundle_length": length,
                        "start_trajectory_id": str(lookup[start]["trajectory_id"]),
                        "first_step_trajectory_id": (
                            str(lookup[path[1]]["trajectory_id"]) if path_exists else ""
                        ),
                        "endpoint_trajectory_id": (
                            str(lookup[path[-1]]["trajectory_id"]) if path_exists else ""
                        ),
                        "start_alpha_hip": start[0],
                        "start_alpha_knee": start[1],
                        "start_alpha_phase": start[2],
                        "first_step_alpha_hip": path[1][0],
                        "first_step_alpha_knee": path[1][1],
                        "first_step_alpha_phase": path[1][2],
                        "endpoint_alpha_hip": path[-1][0],
                        "endpoint_alpha_knee": path[-1][1],
                        "endpoint_alpha_phase": path[-1][2],
                        "predicted_cumulative_delta_J": delta,
                        "predicted_endpoint_improvement": improvement,
                        "calibrated_uncertainty": bound,
                        "meaningful_improvement_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
                        "magnitude_gate_pass": gate_a,
                        "direction_gate_pass": gate_b,
                        "magnitude_margin": magnitude_margin,
                        "direction_margin": direction_margin,
                        "additive_margin": additive_margin,
                        "semantic_selection_margin": selection_margin,
                        "path_exists": path_exists,
                        "all_intermediate_nodes_admissible": geometric,
                        "all_nodes_model_supported": supported,
                        "patient_envelope_compliant": patient,
                        "reference_provenance_valid": provenance,
                        "bundle_evidence_authorizes_next_step": authorized,
                        "selected_authorization": False,
                        "authorized_execution_count": 1 if authorized else 0,
                        "queued_later_steps": False,
                        "authorization_lifetime": "CURRENT_ITERATION_ONLY",
                        "authorization_invalidated_after_execution": False,
                        "truth_used_for_authorization": False,
                    }
                )
    return pd.DataFrame(rows)


def select_bundle_authorization(options: pd.DataFrame) -> pd.Series | None:
    if options.empty:
        return None
    eligible = options.loc[
        options["bundle_evidence_authorizes_next_step"].astype(bool)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        [
            "semantic_selection_margin",
            "bundle_length",
            "endpoint_trajectory_id",
            "authorization_id",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0].copy()


def _augment_guard_with_bundle_step(
    guarded: pd.DataFrame,
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    dynamic_uncertainty: ResearchDecisionUncertainty,
    spec: SemanticSpec,
    calibration: SemanticsCalibration,
    authorization: pd.Series,
) -> pd.DataFrame:
    output = guarded.copy(deep=True)
    output["authorization_mode"] = "DIRECT_ONE_STEP"
    selected_id = str(authorization["first_step_trajectory_id"])
    mask = output["trajectory_id"].astype(str).eq(selected_id)
    if not mask.any():
        formal = local_prediction_candidates(
            prediction_map,
            current,
            TrustRegionSteps(
                hip_deg=MINIMUM_STEP_HIP_DEG,
                knee_deg=MINIMUM_STEP_KNEE_DEG,
                phase=MINIMUM_STEP_PHASE,
            ),
        )
        candidate = apply_semantic_decision_guard(
            formal, current, dynamic_uncertainty, spec, calibration
        )
        candidate = candidate.loc[candidate["trajectory_id"].astype(str).eq(selected_id)]
        if len(candidate) != 1:
            raise RuntimeError("authorized formal first step missing from prediction map")
        output = pd.concat((output, candidate), ignore_index=True, sort=False)
        mask = output["trajectory_id"].astype(str).eq(selected_id)
    output.loc[mask, "research_exploit_eligible"] = True
    output.loc[mask, "decision_guard_status"] = "BUNDLE_ENDPOINT_TWO_GATE_FIRST_STEP_AUTHORIZED"
    output.loc[mask, "authorization_mode"] = "BUNDLE_SUPPORTED_ONE_STEP"
    output.loc[mask, "bundle_authorization_id"] = str(authorization["authorization_id"])
    output.loc[mask, "bundle_length"] = int(authorization["bundle_length"])
    return output


def run_semantic_shadow(
    state: InitialResearchState,
    spec: SemanticSpec,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    manifest_gate: FrozenSemanticsManifestGate,
    calibration: SemanticsCalibration,
    *,
    patient_validity_cache: dict[tuple[float, float, float], bool] | None = None,
    trial_budget: int = MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
) -> tuple[PolicyRunResult, pd.DataFrame]:
    """Run one manifest-gated synthetic development shadow."""

    if spec.default_enabled or DEFAULT_ENABLED:
        raise PermissionError("semantics comparator must remain default-off")
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
    dynamic_uncertainty = evaluate_validation_pairwise_uncertainty(
        state, parameters, iteration=0
    )
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
    steps = TrustRegionSteps()
    executed_keys = {operating_alpha.key()}
    history_rows: list[dict[str, Any]] = []
    guard_frames: list[pd.DataFrame] = []
    bundle_frames: list[pd.DataFrame] = []
    stop_reason = ""
    model_update_count = 0
    cumulative_regret = 0.0
    previous_dynamic_bound = float(dynamic_uncertainty.bound_used_by_guard)

    for iteration in range(1, trial_budget + 1):
        truth_before_proposal = oracle.truth_calls
        local = local_prediction_candidates(prediction_map, operating_alpha, steps)
        guarded = apply_semantic_decision_guard(
            local, operating_alpha, dynamic_uncertainty, spec, calibration
        )
        guarded["iteration"] = iteration
        guarded["policy_id"] = spec.semantic_id
        guarded["scenario_name"] = state.scenario_name
        guarded["subject_id"] = state.subject_id
        guarded["selected_for_execution"] = False
        guarded["selection_mode"] = ""
        guarded["authorization_mode"] = "DIRECT_ONE_STEP"

        selected = select_exploit_candidate(
            guarded, POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
        )
        selected_by_bundle = False
        selected_authorization: pd.Series | None = None
        if selected is None and spec.bundle_rule is not None:
            options = evaluate_bundle_semantics(
                prediction_map,
                operating_alpha,
                spec,
                calibration,
                iteration=iteration,
                patient_validity_cache=patient_cache,
            )
            options["case_id"] = f"{state.subject_id}__{state.scenario_name}"
            options["subject_id"] = state.subject_id
            options["scenario_name"] = state.scenario_name
            selected_authorization = select_bundle_authorization(options)
            if selected_authorization is not None:
                selected_id = str(selected_authorization["first_step_trajectory_id"])
                selected = prediction_map.loc[
                    prediction_map["trajectory_id"].astype(str).eq(selected_id)
                ].iloc[0].copy()
                selected_by_bundle = True
                options.loc[
                    options["authorization_id"].astype(str).eq(
                        str(selected_authorization["authorization_id"])
                    ),
                    ["selected_authorization", "authorization_invalidated_after_execution"],
                ] = [True, True]
                guarded = _augment_guard_with_bundle_step(
                    guarded,
                    prediction_map,
                    operating_alpha,
                    dynamic_uncertainty,
                    spec,
                    calibration,
                    selected_authorization,
                )
            bundle_frames.append(options)

        purpose = TRIAL_PURPOSE_EXPLOIT
        selection_mode = (
            "BUNDLE_SUPPORTED_ONE_STEP" if selected_by_bundle else "DIRECT_ONE_STEP"
        )
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
            if local.empty:
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
            raise RuntimeError("current operating alpha cannot be re-executed")
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            if selected_by_bundle:
                difference = np.asarray(selected_alpha.key()) - np.asarray(
                    operating_alpha.key()
                )
                changed = np.flatnonzero(np.abs(difference) > 1e-12)
                if len(changed) != 1:
                    raise RuntimeError("bundle attempted a mixed-axis step")
                axis = _AXES[int(changed[0])]
                if not np.isclose(
                    abs(difference[changed[0]]), _GRID_STEP[axis], atol=1e-12, rtol=0.0
                ):
                    raise RuntimeError("bundle attempted a non-adjacent step")
            else:
                allowed = {
                    alpha.key()
                    for alpha in build_coordinate_neighborhood(operating_alpha, steps)
                }
                if selected_alpha.key() not in allowed:
                    raise RuntimeError("direct exploit attempted a nonlocal jump")
            guarded.loc[
                guarded["trajectory_id"].astype(str).eq(selected_id),
                ["selected_for_execution", "selection_mode"],
            ] = [True, selection_mode]
        guarded["policy_decision"] = purpose
        guarded["prospective_stop_reason"] = ""
        guard_frames.append(guarded)
        if oracle.truth_calls != truth_before_proposal:
            raise RuntimeError("proposal or authorization accessed truth")

        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = selected_id
        selection_token = oracle.declare_selected(selected_id, purpose)
        manifest_gate.record_truth_access("SELECTED_ONE_TRAJECTORY")
        execution = oracle.execute(selection_token, trajectory)
        if not execution.observation_valid:
            stop_reason = STOP_MODEL_UPDATE_FAILURE
            break
        actual = _actual_objective(selected_id, execution, reference_metrics)
        predicted_j = float(selected["J_pred"])
        current_predicted_j = float(
            _row_for_alpha(prediction_map, operating_alpha)["J_pred"]
        )
        delta_pred = predicted_j - current_predicted_j
        operating_before = operating_alpha
        operating_actual_before = operating_actual_j
        delta_actual = actual.mechanical_cost_j_rms - operating_actual_before
        best_before = best_actual_j
        accepted = accept_actual_trial(actual.mechanical_cost_j_rms, best_actual_j)
        if accepted:
            best_actual_j = actual.mechanical_cost_j_rms
            best_alpha = selected_alpha
        if selected_by_bundle:
            operating_alpha = selected_alpha
            operating_actual_j = actual.mechanical_cost_j_rms
        elif accepted:
            operating_alpha = selected_alpha
            operating_actual_j = actual.mechanical_cost_j_rms
        elif purpose == TRIAL_PURPOSE_EXPLOIT:
            steps = shrink_steps(steps)
        cumulative_regret += max(actual.mechanical_cost_j_rms - best_before, 0.0)
        executed_keys.add(selected_alpha.key())
        direct_false = bool(
            not selected_by_bundle
            and purpose == TRIAL_PURPOSE_EXPLOIT
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
        next_dynamic_uncertainty = evaluate_validation_pairwise_uncertainty(
            state, parameters, iteration=iteration
        )
        history_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.semantic_id,
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
                "delta_J_pred_one_step": delta_pred,
                "delta_J_actual_vs_operating": delta_actual,
                "accepted_meaningful_improvement": accepted,
                "executed_false_improvement": direct_false,
                "bundle_authorization_id": (
                    str(selected_authorization["authorization_id"])
                    if selected_authorization is not None
                    else ""
                ),
                "bundle_length": (
                    int(selected_authorization["bundle_length"])
                    if selected_authorization is not None
                    else np.nan
                ),
                "bundle_authorized_execution_count": 1 if selected_by_bundle else 0,
                "queued_later_bundle_steps": False,
                "bundle_authorization_invalidated": selected_by_bundle,
                "model_refit_after_execution": bool(estimation.optimizer_success),
                "full_map_recomputed_after_execution": True,
                "truth_accessed_before_selection": False,
                "manifest_sha_verified_before_truth": True,
                "execution_status": (
                    EXECUTED_FALSE_IMPROVEMENT if direct_false else "EXECUTED"
                ),
                "cumulative_regret_vs_best_before": cumulative_regret,
                "stop_reason_after_iteration": stop_reason,
            }
        )
        if (
            not stop_reason
            and next_dynamic_uncertainty.bound_used_by_guard
            > previous_dynamic_bound + OBJECTIVE_EQUIVALENCE_TOLERANCE
        ):
            stop_reason = STOP_MODEL_ADEQUACY_DEGRADED
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
        previous_dynamic_bound = float(next_dynamic_uncertainty.bound_used_by_guard)
        dynamic_uncertainty = next_dynamic_uncertainty
        if stop_reason:
            break

    if not stop_reason:
        stop_reason = STOP_MAX_PERSONALIZATION_TRIALS
        if history_rows:
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
    history = pd.DataFrame(history_rows)
    guard_audit = pd.concat(guard_frames, ignore_index=True, sort=False)
    bundle_history = (
        pd.concat(bundle_frames, ignore_index=True, sort=False)
        if bundle_frames
        else pd.DataFrame()
    )
    number_executed = len(history)
    summary = {
        "case_id": f"{state.subject_id}__{state.scenario_name}",
        "subject_id": state.subject_id,
        "scenario_name": state.scenario_name,
        "policy_id": spec.semantic_id,
        "semantic_role": spec.role,
        "research_status": RESEARCH_ONLY,
        "number_of_executed_trials": number_executed,
        "number_of_exploit_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT).sum()) if number_executed else 0,
        "number_of_explore_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLORE).sum()) if number_executed else 0,
        "number_of_bundle_authorized_trials": int(history["selection_mode"].eq("BUNDLE_SUPPORTED_ONE_STEP").sum()) if number_executed else 0,
        "number_of_executed_false_improvements": int(history["executed_false_improvement"].sum()) if number_executed else 0,
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
        "new_policy_default_enabled": False,
        "human_ready": False,
        "robot_motion_approved": False,
        "model_reliability_status": (
            MODEL_RELIABILITY_DEGRADED
            if stop_reason == STOP_MODEL_ADEQUACY_DEGRADED
            else "RESEARCH_DIAGNOSTIC_NOT_FORMALLY_RELIABLE"
        ),
    }
    return (
        PolicyRunResult(
            subject_id=state.subject_id,
            scenario_name=state.scenario_name,
            policy_id=spec.semantic_id,
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
        ),
        bundle_history,
    )


def attach_bundle_posthoc_truth(
    history: pd.DataFrame, truth_landscapes: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    frames = []
    for case_id, group in history.groupby("case_id", sort=False):
        truth = truth_landscapes[str(case_id)]
        values = dict(
            zip(truth["trajectory_id"].astype(str), truth["J_truth"].astype(float))
        )
        output = group.copy()
        output["start_J_truth_posthoc"] = output["start_trajectory_id"].map(values)
        output["endpoint_J_truth_posthoc"] = output["endpoint_trajectory_id"].map(values)
        output["truth_cumulative_delta_J_posthoc"] = (
            output["endpoint_J_truth_posthoc"] - output["start_J_truth_posthoc"]
        )
        output["truth_bundle_meaningful_improvement"] = output[
            "truth_cumulative_delta_J_posthoc"
        ].lt(-OBJECTIVE_EQUIVALENCE_TOLERANCE)
        output["bundle_endpoint_false_improvement"] = (
            output["selected_authorization"].astype(bool)
            & ~output["truth_bundle_meaningful_improvement"].astype(bool)
        )
        output["posthoc_truth_fed_back_to_policy"] = False
        frames.append(output)
    return pd.concat(frames, ignore_index=True, sort=False)


def small_step_semantic_recovery(
    calibration: SemanticsCalibration,
) -> pd.DataFrame:
    source = pd.read_csv(SMALL_STEP_SOURCE_PATH)
    rows: list[dict[str, Any]] = []
    for path_id, group in source.groupby("path_id", sort=True):
        ordered = group.sort_values("step_number")
        first = ordered.iloc[0]
        for spec in SEMANTIC_VARIANTS:
            candidates = []
            if spec.bundle_rule is not None:
                for length in CALIBRATED_BUNDLE_LENGTHS:
                    found = ordered.loc[ordered["step_number"].eq(length)]
                    if found.empty:
                        continue
                    item = found.iloc[0]
                    improvement = -float(item["cumulative_endpoint_deltaJ_pred"])
                    bound = calibration.bundle_scale_p95[length]
                    magnitude_margin = improvement - OBJECTIVE_EQUIVALENCE_TOLERANCE
                    direction_margin = improvement - bound
                    additive_margin = magnitude_margin - bound
                    authorized = (
                        additive_margin > 0.0
                        if spec.bundle_rule == "BUNDLE_SCALE_P95_ADDITIVE"
                        else magnitude_margin > 0.0 and direction_margin > 0.0
                    )
                    candidates.append(
                        {
                            "length": length,
                            "predicted": float(item["cumulative_endpoint_deltaJ_pred"]),
                            "truth": float(item["cumulative_endpoint_deltaJ_truth"]),
                            "bound": bound,
                            "magnitude_margin": magnitude_margin,
                            "direction_margin": direction_margin,
                            "additive_margin": additive_margin,
                            "selection_margin": (
                                additive_margin
                                if spec.bundle_rule == "BUNDLE_SCALE_P95_ADDITIVE"
                                else min(magnitude_margin, direction_margin)
                            ),
                            "authorized": authorized,
                        }
                    )
            eligible = [candidate for candidate in candidates if candidate["authorized"]]
            selected = (
                sorted(
                    eligible,
                    key=lambda item: (-item["selection_margin"], item["length"]),
                )[0]
                if eligible
                else None
            )
            rows.append(
                {
                    "path_id": path_id,
                    "case_id": first["case_id"],
                    "coordinate": first["coordinate"],
                    "direction": first["direction"],
                    "semantic_id": spec.semantic_id,
                    "single_step_predicted_improvement": -float(first["single_step_deltaJ_pred"]),
                    "single_step_magnitude_gate_pass": -float(first["single_step_deltaJ_pred"]) > OBJECTIVE_EQUIVALENCE_TOLERANCE,
                    "bundle_available": spec.bundle_rule is not None,
                    "authorized_bundle_length": selected["length"] if selected else np.nan,
                    "authorized_predicted_delta_J": selected["predicted"] if selected else np.nan,
                    "authorized_truth_delta_J_posthoc": selected["truth"] if selected else np.nan,
                    "calibrated_uncertainty": selected["bound"] if selected else np.nan,
                    "magnitude_margin": selected["magnitude_margin"] if selected else np.nan,
                    "direction_margin": selected["direction_margin"] if selected else np.nan,
                    "additive_margin": selected["additive_margin"] if selected else np.nan,
                    "recovered_small_step_path": bool(
                        selected is not None
                        and selected["truth"] < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                    ),
                    "truth_used_for_authorization": False,
                    "truth_attached_posthoc_only": True,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ADDITIVE_ASSUMPTION",
    "AUDIT_ID",
    "CALIBRATED_BUNDLE_LENGTHS",
    "DEFAULT_ENABLED",
    "EXPECTED_BUNDLE_PAIR_PLAN_SHA256",
    "EXPECTED_CALIBRATION_MANIFEST_SHA256",
    "MANIFEST_ID",
    "MORE_EVIDENCE",
    "NOT_SUFFICIENT",
    "PRIMARY_BLOCKER",
    "SEMANTIC_VARIANTS",
    "FrozenSemanticsManifestGate",
    "SemanticSpec",
    "SemanticsCalibration",
    "apply_semantic_decision_guard",
    "attach_bundle_posthoc_truth",
    "candidate_manifest_payload",
    "canonical_json_bytes",
    "evaluate_bundle_semantics",
    "load_semantics_calibration",
    "run_semantic_shadow",
    "select_bundle_authorization",
    "sha256_file",
    "small_step_semantic_recovery",
]
