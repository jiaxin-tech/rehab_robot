"""Default-off development shadow for the next P2 policy design.

The module adds no runtime policy.  It consumes independent calibration only
as frozen uncertainty evidence, then evaluates a predeclared candidate set on
development cases.  A bundle can authorize exactly one adjacent formal-grid
step; every authorization expires after that simulated execution.
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
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
    SearchAlpha,
    TrustRegionSteps,
    accept_actual_trial,
    build_coordinate_neighborhood,
    shrink_steps,
)


DESIGN_ID = "P2_NEXT_REVISION_POLICY_DESIGN_V1"
CANDIDATE_MANIFEST_ID = "POLICY_DESIGN_CANDIDATE_MANIFEST_V1"
BUNDLE_POLICY_ID = "BUNDLE_SUPPORTED_ONE_STEP_COMMITMENT_V1"
DATA_ROLE_DEVELOPMENT = "DEVELOPMENT_POLICY_SHADOW_ONLY"
DATA_ROLE_CALIBRATION = "INDEPENDENT_CALIBRATION_UNCERTAINTY_ONLY"
FINAL_READY = "POLICY_CANDIDATE_READY_FOR_NEW_PROSPECTIVE"
FINAL_REVISE = "POLICY_DESIGN_REQUIRES_REVISION"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"
P2_NEXT_REVISION_DEFAULT_ENABLED = False
STOP_DECISION_VALUE_K = "STOP_DECISION_VALUE_ZERO_RUN_K"
EXPECTED_CALIBRATION_MANIFEST_SHA256 = (
    "08f930692704c24f10f85f094eabf45fc5e0842ec3f479345e62bee892df1729"
)
EXPECTED_LOCAL_PAIR_PLAN_SHA256 = (
    "ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55"
)
EXPECTED_BUNDLE_PAIR_PLAN_SHA256 = (
    "3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84"
)
OLD_DEVELOPMENT_LOCAL_P95 = 0.000430956758923898
CALIBRATED_BUNDLE_LENGTHS = (2, 3, 5)
MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON = 20

MODULE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_next_revision_independent_calibration_v1"
)
CALIBRATION_MANIFEST_PATH = (
    CALIBRATION_DIRECTORY / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
)
BUNDLE_PLAN_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "post_prospective_rejection_root_cause_audit_v1"
    / "designated_bundle_validation_pair_plan.csv"
)
LOCAL_PLAN_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_v2_formal_research_protocol_v1"
    / "designated_local_validation_pair_plan.csv"
)

_AXES = ("hip", "knee", "phase")
_AXIS_INDEX = {"hip": 0, "knee": 1, "phase": 2}
_GRID_STEP = {
    "hip": GRID_HIP_STEP_DEG,
    "knee": GRID_KNEE_STEP_DEG,
    "phase": GRID_PHASE_STEP,
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


def stable_manifest_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class PolicyDesignSpec:
    policy_id: str
    direct_decision_id: str
    bundle_decision_id: str | None
    bundle_percentile: str | None
    bundle_axis_stratified: bool
    stopping_rule_id: str
    stopping_k: int | None
    role: str
    default_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICY_VARIANTS = (
    PolicyDesignSpec(
        "R0_P2_V1_G0_NO_BUNDLE_S0",
        "D0_CURRENT_P2_V1_ONE_STEP",
        None,
        None,
        False,
        "S0_CURRENT_CONTINUATION",
        None,
        "P2_V1_COMPARATOR",
    ),
    PolicyDesignSpec(
        "R1_G0_BUNDLE_SCALE_P95_S0",
        "D0_CURRENT_P2_V1_ONE_STEP",
        BUNDLE_POLICY_ID,
        "P95",
        False,
        "S0_CURRENT_CONTINUATION",
        None,
        "PRIMARY_BUNDLE_EFFECT",
    ),
    PolicyDesignSpec(
        "R2_G0_BUNDLE_SCALE_P95_S2",
        "D0_CURRENT_P2_V1_ONE_STEP",
        BUNDLE_POLICY_ID,
        "P95",
        False,
        "S2_DECISION_VALUE_K2",
        2,
        "PRIMARY_MINIMAL_NEXT_REVISION",
    ),
    PolicyDesignSpec(
        "R3_G0_BUNDLE_SCALE_P99_S2",
        "D0_CURRENT_P2_V1_ONE_STEP",
        BUNDLE_POLICY_ID,
        "P99",
        False,
        "S2_DECISION_VALUE_K2",
        2,
        "P99_SENSITIVITY",
    ),
    PolicyDesignSpec(
        "R4_G0_BUNDLE_SCALE_AXIS_P95_S2",
        "D0_CURRENT_P2_V1_ONE_STEP",
        BUNDLE_POLICY_ID,
        "P95",
        True,
        "S2_DECISION_VALUE_K2",
        2,
        "SCALE_AXIS_SENSITIVITY",
    ),
)


@dataclass(frozen=True)
class CalibrationUncertainty:
    one_step_p95: float
    one_step_p99: float
    scale_p95: Mapping[int, float]
    scale_p99: Mapping[int, float]
    scale_axis_p95: Mapping[int, Mapping[str, float]]
    source_row_counts: Mapping[str, int]

    def bundle_bound(self, spec: PolicyDesignSpec, length: int, axis: str) -> float:
        if length not in CALIBRATED_BUNDLE_LENGTHS:
            raise PermissionError("only calibrated 2/3/5-step bundles may be used")
        if spec.bundle_percentile == "P99":
            return float(self.scale_p99[length])
        if spec.bundle_percentile != "P95":
            raise PermissionError("bundle percentile is not predeclared")
        if spec.bundle_axis_stratified:
            return float(self.scale_axis_p95[length][axis])
        return float(self.scale_p95[length])


def load_calibration_uncertainty() -> CalibrationUncertainty:
    """Load only independent residual summaries; never policy outcomes."""

    if sha256_file(CALIBRATION_MANIFEST_PATH) != EXPECTED_CALIBRATION_MANIFEST_SHA256:
        raise RuntimeError("independent calibration manifest SHA changed")
    if sha256_file(LOCAL_PLAN_PATH) != EXPECTED_LOCAL_PAIR_PLAN_SHA256:
        raise RuntimeError("designated local pair plan SHA changed")
    if sha256_file(BUNDLE_PLAN_PATH) != EXPECTED_BUNDLE_PAIR_PLAN_SHA256:
        raise RuntimeError("designated bundle pair plan SHA changed")

    one = pd.read_csv(CALIBRATION_DIRECTORY / "one_step_residual_summary.csv")
    one = one.loc[one["summary_scope"].eq("OVERALL")].iloc[0]
    bundle = pd.read_csv(CALIBRATION_DIRECTORY / "bundle_residual_summary.csv")
    overall = bundle.loc[bundle["summary_scope"].eq("OVERALL")].copy()
    scale_p95: dict[int, float] = {}
    scale_p99: dict[int, float] = {}
    axis_p95: dict[int, dict[str, float]] = {}
    row_counts: dict[str, int] = {"one_step": int(one["n"])}
    for length in CALIBRATED_BUNDLE_LENGTHS:
        label = f"{length}-step"
        row = overall.loc[overall["decision_scale"].eq(label)]
        if len(row) != 1:
            raise RuntimeError(f"missing independent {label} residual summary")
        scale_p95[length] = float(row.iloc[0]["P95"])
        scale_p99[length] = float(row.iloc[0]["P99"])
        residuals = pd.read_csv(
            CALIBRATION_DIRECTORY / f"independent_bundle_{length}step_residuals.csv"
        )
        if len(residuals) != int(row.iloc[0]["n"]):
            raise RuntimeError(f"independent {label} residual count changed")
        axis_p95[length] = {}
        for axis in _AXES:
            values = residuals.loc[
                residuals["coordinate"].astype(str).eq(axis), "e_deltaJ_bundle"
            ].to_numpy(dtype=float)
            if len(values) == 0:
                raise RuntimeError(f"missing independent {label}/{axis} residuals")
            axis_p95[length][axis] = float(np.quantile(values, 0.95, method="linear"))
            row_counts[f"{label}_{axis}"] = len(values)
    return CalibrationUncertainty(
        one_step_p95=float(one["P95"]),
        one_step_p99=float(one["P99"]),
        scale_p95=scale_p95,
        scale_p99=scale_p99,
        scale_axis_p95=axis_p95,
        source_row_counts=row_counts,
    )


def candidate_manifest_payload(
    uncertainty: CalibrationUncertainty,
    *,
    checkpoint_commit: str,
    protected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Create the complete candidate set without development truth outcomes."""

    return {
        "manifest_id": CANDIDATE_MANIFEST_ID,
        "design_id": DESIGN_ID,
        "status": "FROZEN_BEFORE_DEVELOPMENT_SHADOW_TRUTH",
        "checkpoint_commit": checkpoint_commit,
        "default_enabled": False,
        "P2_V1_replaced": False,
        "prospective_cohort_created": False,
        "held_out_final_test_read_allowed": False,
        "calibration_cases_used_for_policy_outcomes": False,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA256,
        "local_pair_plan_sha256": EXPECTED_LOCAL_PAIR_PLAN_SHA256,
        "bundle_pair_plan_sha256": EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
        "data_roles": {
            "development": "ORIGINAL_9_PLUS_REJECTED_PROSPECTIVE_6_POLICY_SHADOW",
            "independent_calibration": "TWELVE_CASES_RESIDUAL_DISTRIBUTIONS_ONLY",
            "future_prospective": "NOT_CREATED",
            "held_out_final_test": "NOT_READ",
        },
        "direct_comparators": [
            {
                "id": "D0_CURRENT_P2_V1_ONE_STEP",
                "uncertainty": "CURRENT_GLOBAL_PAIRWISE_MAX",
                "policy_branch": True,
            },
            {
                "id": "D1_INDEPENDENT_ONE_STEP_P95_RESEARCH_COMPARATOR",
                "uncertainty": uncertainty.one_step_p95,
                "percentile": "P95",
                "policy_branch": False,
                "threshold_frozen": False,
            },
        ],
        "bundle_uncertainty_candidates": {
            "percentile_levels": ["P95", "P99"],
            "percentile_search_performed": False,
            "scale_p95": {str(k): v for k, v in uncertainty.scale_p95.items()},
            "scale_p99": {str(k): v for k, v in uncertainty.scale_p99.items()},
            "scale_axis_p95": {
                str(k): dict(v) for k, v in uncertainty.scale_axis_p95.items()
            },
            "analytic_n_times_U1_used": False,
            "analytic_sqrt_n_U1_used": False,
        },
        "bundle_rule": {
            "id": BUNDLE_POLICY_ID,
            "allowed_lengths": list(CALIBRATED_BUNDLE_LENGTHS),
            "same_axis_required": True,
            "same_direction_required": True,
            "formal_adjacent_grid_required": True,
            "all_intermediate_nodes_admissible": True,
            "all_nodes_patient_envelope_compliant": True,
            "all_nodes_model_supported_at_90_percent": True,
            "lower_bound_margin": "-deltaJ_pred_start_endpoint-U_bundle-0.005",
            "authorization_scope": "NEXT_ONE_FORMAL_GRID_STEP_ONLY",
            "authorization_invalidated_after_execution": True,
            "model_refit_after_every_execution": True,
            "full_map_recomputed_after_every_execution": True,
        },
        "policy_variants": [spec.as_dict() for spec in POLICY_VARIANTS],
        "stopping_candidates": ["S0_CURRENT_CONTINUATION", "S2_DECISION_VALUE_K2"],
        "stopping_k_values": [2],
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "protected_source_sha256": dict(protected_source_sha256),
        "truth_used_to_create_candidates": False,
        "truth_may_modify_candidates_in_this_task": False,
        "human_ready": NOT_HUMAN_READY,
        "robot_motion_approved": NOT_ROBOT_MOTION_APPROVED,
    }


class FrozenPolicyDesignManifestGate:
    """Fail closed unless the exact pre-shadow candidate manifest is present."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = str(expected_sha256)
        self.truth_access_count = 0
        self.truth_access_stages: list[str] = []

    def require_frozen(self) -> None:
        if not self.path.is_file() or sha256_file(self.path) != self.expected_sha256:
            raise PermissionError("shadow truth requires frozen policy candidate manifest")

    def record_truth_access(self, stage: str) -> None:
        self.require_frozen()
        self.truth_access_count += 1
        self.truth_access_stages.append(str(stage))


def _static_direct_uncertainty(
    state: InitialResearchState,
    iteration: int,
    uncertainty: CalibrationUncertainty,
) -> ResearchDecisionUncertainty:
    return ResearchDecisionUncertainty(
        case_id=f"{state.subject_id}__{state.scenario_name}",
        iteration=int(iteration),
        pairwise_audit=pd.DataFrame(),
        maximum_observed_e_delta_j=float(uncertainty.one_step_p99),
        p95_observed_e_delta_j=float(uncertainty.one_step_p95),
        p99_observed_e_delta_j=float(uncertainty.one_step_p99),
        validation_pair_count=int(uncertainty.source_row_counts["one_step"]),
        bound_used_by_guard=float(uncertainty.one_step_p95),
        bound_type="INDEPENDENT_CALIBRATION_ONE_STEP_P95_RESEARCH_COMPARATOR",
        bound_status="PREDECLARED_RESEARCH_COMPARATOR_NOT_POLICY",
    )


def _key(alpha: SearchAlpha | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(alpha, SearchAlpha):
        values = alpha.key()
    else:
        values = tuple(float(value) for value in alpha)
    return tuple(round(float(value), 12) for value in values)


def _map_lookup(table: pd.DataFrame) -> dict[tuple[float, float, float], dict[str, Any]]:
    return {
        _key((row["hip_delta"], row["knee_delta"], row["phase_delta"])): row
        for row in table.to_dict(orient="records")
    }


def _patient_valid(
    key: tuple[float, float, float],
    cache: dict[tuple[float, float, float], bool],
) -> bool:
    if key not in cache:
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=key[0],
            knee_amplitude_delta_deg=key[1],
            knee_phase_shift=key[2],
        )
        cache[key] = default_virtual_patient_envelope().contains(generated.trajectory)
    return cache[key]


def evaluate_bundle_options(
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    spec: PolicyDesignSpec,
    uncertainty: CalibrationUncertainty,
    *,
    iteration: int,
    patient_validity_cache: dict[tuple[float, float, float], bool],
) -> pd.DataFrame:
    """Evaluate only predeclared straight formal-neighbor bundles."""

    if spec.bundle_decision_id is None:
        return pd.DataFrame()
    lookup = _map_lookup(prediction_map)
    start = current.key()
    if start not in lookup:
        raise RuntimeError("current alpha missing from prediction map")
    start_j = float(lookup[start]["J_pred"])
    rows: list[dict[str, Any]] = []
    for axis in _AXES:
        index = _AXIS_INDEX[axis]
        for direction, sign in (("NEGATIVE", -1.0), ("POSITIVE", 1.0)):
            for length in CALIBRATED_BUNDLE_LENGTHS:
                path: list[tuple[float, float, float]] = []
                for step_number in range(length + 1):
                    point = list(start)
                    point[index] += sign * _GRID_STEP[axis] * step_number
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
                    and all(_patient_valid(point, patient_validity_cache) for point in path)
                )
                endpoint_j = float(path_rows[-1]["J_pred"]) if path_exists else np.nan
                delta = endpoint_j - start_j if path_exists else np.nan
                bound = uncertainty.bundle_bound(spec, length, axis)
                margin = (
                    -delta - bound - OBJECTIVE_EQUIVALENCE_TOLERANCE
                    if path_exists
                    else np.nan
                )
                authorized = bool(
                    geometric
                    and provenance
                    and supported
                    and patient
                    and np.isfinite(margin)
                    and margin >= -1e-15
                )
                identity = (
                    f"{spec.policy_id}|{iteration}|{start}|{axis}|{direction}|{length}"
                )
                authorization_id = "bundle_auth_" + hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
                rows.append(
                    {
                        "authorization_id": authorization_id,
                        "iteration": int(iteration),
                        "policy_id": spec.policy_id,
                        "bundle_policy_id": BUNDLE_POLICY_ID,
                        "coordinate": axis,
                        "direction": direction,
                        "bundle_length": length,
                        "start_trajectory_id": (
                            str(lookup[start]["trajectory_id"]) if path_exists else ""
                        ),
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
                        "intermediate_trajectory_ids": (
                            ";".join(str(lookup[item]["trajectory_id"]) for item in path[1:-1])
                            if path_exists
                            else ""
                        ),
                        "predicted_cumulative_delta_J": delta,
                        "calibrated_uncertainty": bound,
                        "uncertainty_percentile": spec.bundle_percentile,
                        "uncertainty_stratification": (
                            "SCALE_AXIS" if spec.bundle_axis_stratified else "SCALE_ONLY"
                        ),
                        "meaningful_improvement_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
                        "bundle_lower_bound_margin": margin,
                        "path_exists": path_exists,
                        "formal_neighbor_continuous": path_exists,
                        "same_axis": True,
                        "direction_consistent": True,
                        "mixed_axis": False,
                        "direction_reversal": False,
                        "all_intermediate_nodes_admissible": geometric,
                        "patient_envelope_compliant": patient,
                        "all_nodes_model_supported": supported,
                        "support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
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
            "bundle_lower_bound_margin",
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
    uncertainty: ResearchDecisionUncertainty,
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
        candidate = apply_research_decision_guard(formal, current, uncertainty)
        candidate = candidate.loc[candidate["trajectory_id"].astype(str).eq(selected_id)]
        if len(candidate) != 1:
            raise RuntimeError("authorized formal first step missing from local map")
        candidate = candidate.copy()
        candidate["authorization_mode"] = "BUNDLE_SUPPORTED_ONE_STEP"
        output = pd.concat((output, candidate), ignore_index=True, sort=False)
        mask = output["trajectory_id"].astype(str).eq(selected_id)
    output.loc[mask, "research_exploit_eligible"] = True
    output.loc[mask, "decision_guard_status"] = "BUNDLE_SUPPORTED_FIRST_STEP_AUTHORIZED"
    output.loc[mask, "improvement_margin"] = float(
        authorization["bundle_lower_bound_margin"]
    )
    output.loc[mask, "validation_uncertainty_bound"] = float(
        authorization["calibrated_uncertainty"]
    )
    output.loc[mask, "authorization_mode"] = "BUNDLE_SUPPORTED_ONE_STEP"
    output.loc[mask, "bundle_authorization_id"] = str(
        authorization["authorization_id"]
    )
    output.loc[mask, "bundle_length"] = int(authorization["bundle_length"])
    return output


def _rank_signature(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        table.sort_values(["J_pred", "trajectory_id"], kind="mergesort")[
            "trajectory_id"
        ].astype(str)
    )


def _map_changed(before: pd.DataFrame, after: pd.DataFrame) -> bool:
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


def _next_decision_available(
    prediction_map: pd.DataFrame,
    current: SearchAlpha,
    steps: TrustRegionSteps,
    direct_uncertainty: ResearchDecisionUncertainty,
    spec: PolicyDesignSpec,
    calibration: CalibrationUncertainty,
    patient_cache: dict[tuple[float, float, float], bool],
    iteration: int,
) -> tuple[bool, bool]:
    local = local_prediction_candidates(prediction_map, current, steps)
    guarded = apply_research_decision_guard(local, current, direct_uncertainty)
    direct = select_exploit_candidate(
        guarded, POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
    ) is not None
    bundle = False
    if not direct and spec.bundle_decision_id is not None:
        bundle = select_bundle_authorization(
            evaluate_bundle_options(
                prediction_map,
                current,
                spec,
                calibration,
                iteration=iteration,
                patient_validity_cache=patient_cache,
            )
        ) is not None
    return direct, bundle


def run_policy_design_shadow(
    state: InitialResearchState,
    spec: PolicyDesignSpec,
    parameter_lattice: pd.DataFrame,
    cache: TrajectoryComponentCache,
    manifest_gate: FrozenPolicyDesignManifestGate,
    calibration: CalibrationUncertainty,
    *,
    patient_validity_cache: dict[tuple[float, float, float], bool] | None = None,
    trial_budget: int = MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
) -> tuple[PolicyRunResult, pd.DataFrame, pd.DataFrame]:
    """Run one isolated development shadow without registering a P2 policy."""

    if spec.default_enabled or P2_NEXT_REVISION_DEFAULT_ENABLED:
        raise PermissionError("next-revision policy must remain default-off")
    if trial_budget != MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON:
        raise ValueError("development diagnostic horizon is fixed")
    manifest_gate.require_frozen()
    patient_cache = patient_validity_cache if patient_validity_cache is not None else {}

    parameters = dict(state.parameters)
    fitting_data = state.fitting_data.copy(deep=True)
    domain_data = state.domain_data.copy(deep=True)
    model = _model_for_iteration(state, parameters, domain_data, 0)
    prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
    initial_prediction_map = prediction_map.copy(deep=True)
    direct_uncertainty = evaluate_validation_pairwise_uncertainty(
        state, parameters, iteration=0
    )
    oracle = SelectionGatedVirtualTruthOracle(state.subject_id, state.scenario_name)

    reference = generate_personalized_trajectory()
    reference_trajectory = reference.trajectory.copy(deep=True)
    reference_id = str(reference.metadata["trajectory_id"])
    reference_trajectory["trajectory_id"] = reference_id
    reference_token = oracle.declare_selected(reference_id, "REFERENCE_NORMALIZATION")
    manifest_gate.record_truth_access("REFERENCE_NORMALIZATION")
    reference_execution = oracle.execute(reference_token, reference_trajectory)
    reference_metrics = reference_execution.actual_metrics

    operating_alpha = SearchAlpha()
    operating_actual_j = 1.0
    best_alpha = SearchAlpha()
    best_actual_j = 1.0
    steps = TrustRegionSteps()
    executed_keys: set[tuple[float, float, float]] = {operating_alpha.key()}
    history_rows: list[dict[str, Any]] = []
    guard_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = [
        {
            "case_id": f"{state.subject_id}__{state.scenario_name}",
            "subject_id": state.subject_id,
            "scenario_name": state.scenario_name,
            "policy_id": spec.policy_id,
            **_map_summary(prediction_map, operating_alpha, iteration=0, previous_map=None),
        }
    ]
    known_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    exploration_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []
    bundle_frames: list[pd.DataFrame] = []
    comparator_frames: list[pd.DataFrame] = []
    stop_reason = ""
    model_update_count = 0
    cumulative_regret = 0.0
    zero_value_run = 0
    previous_direct_bound = float(direct_uncertainty.bound_used_by_guard)

    for iteration in range(1, trial_budget + 1):
        truth_before_proposal = oracle.truth_calls
        local = local_prediction_candidates(prediction_map, operating_alpha, steps)
        guarded = apply_research_decision_guard(local, operating_alpha, direct_uncertainty)
        guarded["iteration"] = iteration
        guarded["policy_id"] = spec.policy_id
        guarded["scenario_name"] = state.scenario_name
        guarded["subject_id"] = state.subject_id
        guarded["selected_for_execution"] = False
        guarded["selection_mode"] = ""
        guarded["guard_id"] = "G0_CURRENT_GLOBAL_MAX"
        guarded["bundle_decision_id"] = spec.bundle_decision_id or "NO_BUNDLE"
        guarded["stopping_rule_id"] = spec.stopping_rule_id
        guarded["authorization_mode"] = "DIRECT_ONE_STEP"

        if spec.policy_id == POLICY_VARIANTS[0].policy_id:
            d1 = _static_direct_uncertainty(state, iteration - 1, calibration)
            candidates = guarded.loc[
                ~guarded["decision_guard_status"].eq(CURRENT_BEST_NOT_A_CANDIDATE)
            ].copy()
            candidates["D0_current_guard_authorized"] = candidates[
                "research_exploit_eligible"
            ].astype(bool)
            candidates["D1_independent_P95_margin"] = (
                -candidates["delta_J_pred_vs_current"].astype(float)
                - d1.bound_used_by_guard
                - OBJECTIVE_EQUIVALENCE_TOLERANCE
            )
            candidates["D1_independent_P95_authorized"] = (
                candidates["model_supported"].astype(bool)
                & candidates["current_model_supported"].astype(bool)
                & candidates["D1_independent_P95_margin"].ge(-1e-15)
            )
            candidates["old_G2_development_P95_margin"] = (
                -candidates["delta_J_pred_vs_current"].astype(float)
                - OLD_DEVELOPMENT_LOCAL_P95
                - OBJECTIVE_EQUIVALENCE_TOLERANCE
            )
            candidates["old_G2_development_P95_authorized"] = (
                candidates["model_supported"].astype(bool)
                & candidates["current_model_supported"].astype(bool)
                & candidates["old_G2_development_P95_margin"].ge(-1e-15)
            )
            candidates["independent_one_step_P95"] = d1.bound_used_by_guard
            candidates["truth_used_for_comparator"] = False
            comparator_frames.append(candidates)

        exploit = select_exploit_candidate(
            guarded, POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
        )
        selected = exploit
        selected_by_bundle = False
        selected_authorization: pd.Series | None = None
        options = pd.DataFrame()
        if selected is None and spec.bundle_decision_id is not None:
            options = evaluate_bundle_options(
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
                    direct_uncertainty,
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
                selection_mode = TRIAL_PURPOSE_EXPLORE
                frontier_ranked["selected_for_exploration"] = frontier_ranked[
                    "trajectory_id"
                ].astype(str).eq(str(selected["trajectory_id"]))
                frontier_ranked["iteration"] = iteration
                frontier_ranked["policy_id"] = spec.policy_id
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
        if selected_alpha.key() == operating_alpha.key():
            raise RuntimeError("current operating alpha cannot be re-executed")
        if purpose == TRIAL_PURPOSE_EXPLOIT:
            if selected_by_bundle:
                differences = np.asarray(selected_alpha.key()) - np.asarray(
                    operating_alpha.key()
                )
                changed = np.flatnonzero(np.abs(differences) > 1e-12)
                if len(changed) != 1:
                    raise RuntimeError("bundle authorization attempted a mixed-axis step")
                axis = _AXES[int(changed[0])]
                if not np.isclose(
                    abs(differences[changed[0]]), _GRID_STEP[axis], atol=1e-12, rtol=0.0
                ):
                    raise RuntimeError("bundle authorization attempted a non-adjacent step")
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
        if not frontier_ranked.empty:
            guard_frames.append(frontier_ranked)
        if oracle.truth_calls != truth_before_proposal:
            raise RuntimeError("proposal or bundle authorization accessed truth")

        generated = generate_personalized_trajectory(
            **selected_alpha.as_generator_parameters()
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = selected_id
        # The frozen virtual oracle accepts the original EXPLOIT/EXPLORE
        # purpose vocabulary.  Bundle provenance stays in ``selection_mode``;
        # it does not create a new execution primitive.
        token = oracle.declare_selected(selected_id, purpose)
        manifest_gate.record_truth_access("SELECTED_ONE_TRAJECTORY")
        execution = oracle.execute(token, trajectory)
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
        delta_actual_operating = actual.mechanical_cost_j_rms - operating_actual_before
        best_before = best_actual_j
        best_alpha_before = best_alpha
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
            and delta_actual_operating >= -OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        bundle_step_harmful = bool(selected_by_bundle and delta_actual_operating > 0.0)

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
        previous_rank = _rank_signature(previous_map)
        previous_global_best = previous_rank[0]
        model = _model_for_iteration(state, parameters, domain_data, iteration)
        prediction_map, _ = build_predicted_map(model, parameter_lattice, cache)
        next_rank = _rank_signature(prediction_map)
        next_direct_uncertainty = evaluate_validation_pairwise_uncertainty(
            state, parameters, iteration=iteration
        )
        new_supported = int(
            prediction_map["model_supported"].sum()
            - previous_map["model_supported"].sum()
        )
        map_changed = _map_changed(previous_map, prediction_map)
        rank_changed = previous_rank != next_rank
        global_best_changed = previous_global_best != next_rank[0]
        parameter_changed = any(
            float(parameters[name]) != float(parameters_before[name])
            for name in PARAMETER_NAMES
        )
        uncertainty_changed = bool(
            float(next_direct_uncertainty.bound_used_by_guard)
            != float(direct_uncertainty.bound_used_by_guard)
        )
        direct_available_after, bundle_available_after = _next_decision_available(
            prediction_map,
            operating_alpha,
            steps,
            next_direct_uncertainty,
            spec,
            calibration,
            patient_cache,
            iteration + 1,
        )
        decision_available_after = direct_available_after or bundle_available_after
        decision_value_zero = bool(
            purpose == TRIAL_PURPOSE_EXPLORE
            and not parameter_changed
            and not map_changed
            and not rank_changed
            and not global_best_changed
            and not uncertainty_changed
            and not decision_available_after
        )
        if purpose == TRIAL_PURPOSE_EXPLORE:
            zero_value_run = zero_value_run + 1 if decision_value_zero else 0
        else:
            zero_value_run = 0

        information_gain = np.nan
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
                    "policy_id": spec.policy_id,
                    "iteration": iteration,
                    "trajectory_id": selected_id,
                    "support_growth": new_supported,
                    "information_gain": information_gain,
                    "parameter_change_observed": parameter_changed,
                    "prediction_map_change_observed": map_changed,
                    "predicted_ranking_changed": rank_changed,
                    "predicted_best_trajectory_changed": global_best_changed,
                    "validation_uncertainty_changed": uncertainty_changed,
                    "direct_exploit_eligibility_after": direct_available_after,
                    "bundle_exploit_eligibility_after": bundle_available_after,
                    "exploit_eligibility_after": decision_available_after,
                    "decision_value_zero": decision_value_zero,
                    "consecutive_zero_decision_value": zero_value_run,
                    "support_alone_used_as_decision_value": False,
                    "future_truth_used_by_stopping": False,
                }
            )

        map_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "subject_id": state.subject_id,
                "scenario_name": state.scenario_name,
                "policy_id": spec.policy_id,
                **_map_summary(
                    prediction_map,
                    operating_alpha,
                    iteration=iteration,
                    previous_map=previous_map,
                ),
            }
        )
        known_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "policy_id": spec.policy_id,
                "iteration": iteration,
                "executed_known_alpha_count": len(executed_keys),
                "supported_point_count": int(prediction_map["model_supported"].sum()),
                "new_supported_point_count": new_supported,
            }
        )
        uncertainty_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "policy_id": spec.policy_id,
                "iteration": iteration,
                "direct_guard_bound": next_direct_uncertainty.bound_used_by_guard,
                "direct_guard_source": next_direct_uncertainty.bound_type,
                "bundle_percentile": spec.bundle_percentile or "NONE",
                "bundle_axis_stratified": spec.bundle_axis_stratified,
                "development_truth_updated_uncertainty": False,
            }
        )
        parameter_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "policy_id": spec.policy_id,
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
                "policy_id": spec.policy_id,
                "policy_role": spec.role,
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
                "best_alpha_hip_before": best_alpha_before.hip_delta_deg,
                "best_alpha_knee_before": best_alpha_before.knee_delta_deg,
                "best_alpha_phase_before": best_alpha_before.phase_delta,
                "best_alpha_hip_after": best_alpha.hip_delta_deg,
                "best_alpha_knee_after": best_alpha.knee_delta_deg,
                "best_alpha_phase_after": best_alpha.phase_delta,
                "J_pred": predicted_j,
                "actual_J": actual.mechanical_cost_j_rms,
                "operating_actual_J_before": operating_actual_before,
                "best_actual_J_before": best_before,
                "best_actual_J_after": best_actual_j,
                "delta_J_pred_one_step": delta_pred,
                "delta_J_actual_vs_operating": delta_actual_operating,
                "direct_decision_uncertainty": direct_uncertainty.bound_used_by_guard,
                "accepted_meaningful_improvement": accepted,
                "executed_false_improvement": direct_false,
                "bundle_step_harmful": bundle_step_harmful,
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
                "support_growth": new_supported,
                "decision_value_zero": decision_value_zero,
                "consecutive_zero_decision_value": zero_value_run,
                "cumulative_regret_vs_best_before": cumulative_regret,
                "truth_accessed_before_selection": False,
                "selection_token": token.token,
                "manifest_sha_verified_before_truth": True,
                "execution_status": (
                    EXECUTED_FALSE_IMPROVEMENT if direct_false else "EXECUTED"
                ),
                "stop_reason_after_iteration": stop_reason,
            }
        )
        false_rows.append(
            {
                "case_id": f"{state.subject_id}__{state.scenario_name}",
                "policy_id": spec.policy_id,
                "iteration": iteration,
                "selection_mode": selection_mode,
                "executed_false_improvement": direct_false,
                "bundle_step_harmful": bundle_step_harmful,
            }
        )

        if (
            not stop_reason
            and next_direct_uncertainty.bound_used_by_guard
            > previous_direct_bound + OBJECTIVE_EQUIVALENCE_TOLERANCE
        ):
            stop_reason = STOP_MODEL_ADEQUACY_DEGRADED
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
        if (
            not stop_reason
            and purpose == TRIAL_PURPOSE_EXPLORE
            and spec.stopping_k is not None
            and zero_value_run >= spec.stopping_k
            and not decision_available_after
        ):
            stop_reason = STOP_DECISION_VALUE_K
            history_rows[-1]["stop_reason_after_iteration"] = stop_reason
            # Persist the actual post-update decision state at which S2 stops.
            # Without this row, a later truth audit would incorrectly label
            # the pre-execution state as the stopping state.
            final_local = local_prediction_candidates(
                prediction_map, operating_alpha, steps
            )
            final_guard = apply_research_decision_guard(
                final_local, operating_alpha, next_direct_uncertainty
            )
            final_guard["iteration"] = iteration + 1
            final_guard["policy_id"] = spec.policy_id
            final_guard["scenario_name"] = state.scenario_name
            final_guard["subject_id"] = state.subject_id
            final_guard["selected_for_execution"] = False
            final_guard["selection_mode"] = "STOP"
            final_guard["guard_id"] = "G0_CURRENT_GLOBAL_MAX"
            final_guard["bundle_decision_id"] = spec.bundle_decision_id or "NO_BUNDLE"
            final_guard["stopping_rule_id"] = spec.stopping_rule_id
            final_guard["authorization_mode"] = "DIRECT_ONE_STEP"
            final_guard["policy_decision"] = "STOP"
            final_guard["prospective_stop_reason"] = stop_reason
            guard_frames.append(final_guard)
        previous_direct_bound = float(next_direct_uncertainty.bound_used_by_guard)
        direct_uncertainty = next_direct_uncertainty
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
    bundle_history = (
        pd.concat(bundle_frames, ignore_index=True, sort=False)
        if bundle_frames
        else pd.DataFrame()
    )
    comparator_history = (
        pd.concat(comparator_frames, ignore_index=True, sort=False)
        if comparator_frames
        else pd.DataFrame()
    )
    number_executed = len(history)
    summary = {
        "case_id": f"{state.subject_id}__{state.scenario_name}",
        "subject_id": state.subject_id,
        "scenario_name": state.scenario_name,
        "policy_id": spec.policy_id,
        "policy_role": spec.role,
        "guard_id": "G0_CURRENT_GLOBAL_MAX",
        "direct_decision_id": spec.direct_decision_id,
        "bundle_decision_id": spec.bundle_decision_id or "NO_BUNDLE",
        "bundle_percentile": spec.bundle_percentile or "NONE",
        "bundle_axis_stratified": spec.bundle_axis_stratified,
        "stopping_rule_id": spec.stopping_rule_id,
        "stopping_k": spec.stopping_k,
        "research_status": RESEARCH_ONLY,
        "number_of_executed_trials": number_executed,
        "number_of_exploit_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT).sum()) if number_executed else 0,
        "number_of_explore_trials": int(history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLORE).sum()) if number_executed else 0,
        "number_of_bundle_authorized_trials": int(history["selection_mode"].eq("BUNDLE_SUPPORTED_ONE_STEP").sum()) if number_executed else 0,
        "number_of_executed_false_improvements": int(history["executed_false_improvement"].sum()) if number_executed else 0,
        "number_of_harmful_bundle_steps": int(history["bundle_step_harmful"].sum()) if number_executed else 0,
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
        "initial_supported_point_count": int(map_rows[0]["supported_point_count"]),
        "final_supported_point_count": int(prediction_map["model_supported"].sum()),
        "known_region_growth": int(prediction_map["model_supported"].sum()) - int(map_rows[0]["supported_point_count"]),
        "low_decision_value_exploration_count": int(sum(bool(row["decision_value_zero"]) for row in exploration_rows)),
        "stop_reason": stop_reason,
        "trial_budget": trial_budget,
        "whole_map_recomputation_count": len(map_rows),
        "truth_calls_including_reference_normalization": oracle.truth_calls,
        "heldout_final_test_used": False,
        "calibration_cases_used_for_policy_outcomes": False,
        "support_used_as_decision_value": False,
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
            policy_id=spec.policy_id,
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
            truth_access_audit={
                "manifest_verified_before_every_truth": True,
                "proposal_truth_accessed": False,
                "heldout_final_test_used": False,
            },
        ),
        bundle_history,
        comparator_history,
    )


def attach_bundle_posthoc_truth(
    history: pd.DataFrame,
    truth_landscapes: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach endpoint truth only after all policy paths are complete."""

    if history.empty:
        return history.copy()
    frames: list[pd.DataFrame] = []
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
        output["bundle_induced_false_improvement"] = (
            output["selected_authorization"].astype(bool)
            & ~output["truth_bundle_meaningful_improvement"].astype(bool)
        )
        output["posthoc_truth_fed_back_to_policy"] = False
        frames.append(output)
    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate_policy_summary(case_summary: pd.DataFrame) -> pd.DataFrame:
    order = [spec.policy_id for spec in POLICY_VARIANTS]
    output = (
        case_summary.groupby("policy_id", as_index=False, sort=False)
        .agg(
            case_count=("case_id", "count"),
            trials=("number_of_executed_trials", "sum"),
            EXPLORE=("number_of_explore_trials", "sum"),
            EXPLOIT=("number_of_exploit_trials", "sum"),
            bundle_authorized_trials=("number_of_bundle_authorized_trials", "sum"),
            missed_improvement=("missed_improvement_rounds", "sum"),
            false_improvement=("number_of_executed_false_improvements", "sum"),
            harmful_bundle_steps=("number_of_harmful_bundle_steps", "sum"),
            final_J=("final_best_actual_J", "mean"),
            regret=("global_truth_regret", "mean"),
            low_value_exploration=("low_decision_value_exploration_count", "sum"),
        )
        .set_index("policy_id")
        .reindex(order)
        .reset_index()
    )
    output["policy_outcomes_used_to_select_candidate"] = False
    output["data_role"] = DATA_ROLE_DEVELOPMENT
    return output


__all__ = [
    "BUNDLE_POLICY_ID",
    "CALIBRATED_BUNDLE_LENGTHS",
    "CANDIDATE_MANIFEST_ID",
    "CalibrationUncertainty",
    "DESIGN_ID",
    "EXPECTED_BUNDLE_PAIR_PLAN_SHA256",
    "EXPECTED_CALIBRATION_MANIFEST_SHA256",
    "EXPECTED_LOCAL_PAIR_PLAN_SHA256",
    "FINAL_READY",
    "FINAL_REVISE",
    "FrozenPolicyDesignManifestGate",
    "MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON",
    "OLD_DEVELOPMENT_LOCAL_P95",
    "P2_NEXT_REVISION_DEFAULT_ENABLED",
    "POLICY_VARIANTS",
    "PolicyDesignSpec",
    "aggregate_policy_summary",
    "attach_bundle_posthoc_truth",
    "candidate_manifest_payload",
    "canonical_json_bytes",
    "evaluate_bundle_options",
    "load_calibration_uncertainty",
    "run_policy_design_shadow",
    "select_bundle_authorization",
    "sha256_file",
    "stable_manifest_sha256",
]
