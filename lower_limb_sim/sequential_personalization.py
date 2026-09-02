"""Deterministic offline sequential personalization around the frozen reference.

The estimator is the existing five-parameter gray-box model.  Candidate
selection can see only its estimates and the continuous generator's feasibility
audit.  A Stage-4.5C virtual truth is queried only after one candidate has been
selected for simulated execution.  Nothing in this module authorizes hardware
motion or supplies a robot safety threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .admissible_personalization_region import (
    AdmissibleRegionArtifacts,
    evaluate_admissible_personalization_region,
    load_admissible_personalization_region,
)
from .config import (
    L1,
    L2,
    identification_initial_guess,
    identification_lower_bounds,
    identification_upper_bounds,
)
from .continuous_reference_neighborhood import (
    FIXED_TIME_SCALE,
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    GeneratedTrajectory,
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
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
    MechanicalObjectiveResult,
    MechanicalTorqueMetrics,
    compute_torque_metrics,
    evaluate_mechanical_objective,
    rank_feasible_candidates,
)
from .mismatch_dynamics import mismatch_inverse_dynamics
from .mismatch_scenarios import get_mismatch_scenario
from .parameter_estimator import (
    PARAMETER_NAMES,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    estimate_subject_parameters,
    predict_joint_torque,
)
from .run_model_mismatch_experiment import (
    ESTIMATOR_INPUT_COLUMNS,
    project_estimator_inputs,
)


SEQUENTIAL_PERSONALIZATION_VERSION = "deterministic_coordinate_trust_region_v1"
FORMAL_SUBJECT_IDS = ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
FORMAL_TRUTH_SCENARIOS = ("matched_linear", "combined_mild")
MAX_EXECUTED_TRIALS = 6
INITIAL_STEP_HIP_DEG = 1.0
INITIAL_STEP_KNEE_DEG = 1.0
INITIAL_STEP_PHASE = 0.01
MINIMUM_STEP_HIP_DEG = 0.25
MINIMUM_STEP_KNEE_DEG = 0.25
MINIMUM_STEP_PHASE = 0.0025
MODEL_RELIABILITY_THRESHOLD: float | None = None

INITIAL_IDENTIFICATION_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_local_active_asymmetric"
    / "identification_dataset.csv"
)
INITIAL_IDENTIFICATION_CONFIG_PATH = (
    INITIAL_IDENTIFICATION_DATASET_PATH.parent / "experiment_config.json"
)

STOP_NO_FEASIBLE_NEIGHBOR = "STOP_NO_FEASIBLE_NEIGHBOR"
STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE = (
    "STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE"
)
STOP_MINIMUM_STEP_WITHOUT_ACCEPTED_IMPROVEMENT = (
    "STOP_MINIMUM_STEP_WITHOUT_ACCEPTED_IMPROVEMENT"
)
STOP_MODEL_UPDATE_FAILED = "MODEL_UPDATE_FAILED"
STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD = (
    "STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD"
)
STOP_MODEL_RELIABILITY_GATE_EXCEEDED = "STOP_MODEL_RELIABILITY_GATE_EXCEEDED"
STOP_MAX_TRIALS = "STOP_MAX_TRIALS"

_PARAMETER_HISTORY_NAMES = {
    "mass_scale": "mass_scale_hat",
    "k_hip_nm_per_rad": "K_hip_hat",
    "k_knee_nm_per_rad": "K_knee_hat",
    "b_hip_nm_s_per_rad": "B_hip_hat",
    "b_knee_nm_s_per_rad": "B_knee_hat",
}


@dataclass(frozen=True)
class SearchAlpha:
    hip_delta_deg: float = 0.0
    knee_delta_deg: float = 0.0
    phase_delta: float = 0.0

    def __post_init__(self) -> None:
        numbers = np.asarray(
            (self.hip_delta_deg, self.knee_delta_deg, self.phase_delta),
            dtype=float,
        )
        if not np.isfinite(numbers).all():
            raise ValueError("search alpha must contain finite values")

    @property
    def neutral(self) -> bool:
        return bool(
            self.hip_delta_deg == 0.0
            and self.knee_delta_deg == 0.0
            and self.phase_delta == 0.0
        )

    def as_generator_parameters(self) -> dict[str, float]:
        return {
            "hip_amplitude_delta_deg": float(self.hip_delta_deg),
            "knee_amplitude_delta_deg": float(self.knee_delta_deg),
            "knee_phase_shift": float(self.phase_delta),
            "time_scale": FIXED_TIME_SCALE,
        }

    def key(self) -> tuple[float, float, float]:
        return (
            round(float(self.hip_delta_deg), 12),
            round(float(self.knee_delta_deg), 12),
            round(float(self.phase_delta), 12),
        )


@dataclass(frozen=True)
class TrustRegionSteps:
    hip_deg: float = INITIAL_STEP_HIP_DEG
    knee_deg: float = INITIAL_STEP_KNEE_DEG
    phase: float = INITIAL_STEP_PHASE

    def __post_init__(self) -> None:
        values = np.asarray((self.hip_deg, self.knee_deg, self.phase), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("trust-region steps must be finite and positive")

    @property
    def at_minimum(self) -> bool:
        return bool(
            self.hip_deg <= MINIMUM_STEP_HIP_DEG + 1e-15
            and self.knee_deg <= MINIMUM_STEP_KNEE_DEG + 1e-15
            and self.phase <= MINIMUM_STEP_PHASE + 1e-15
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    alpha: SearchAlpha
    generated: GeneratedTrajectory | None
    row: dict[str, Any]


@dataclass(frozen=True)
class ProposalResult:
    candidate: CandidateEvaluation | None
    candidate_audit: pd.DataFrame
    current_predicted_j: float | None
    predicted_improvement: float | None
    stop_reason: str


@dataclass(frozen=True)
class TruthExecution:
    estimator_observations: pd.DataFrame
    actual_hip_torque_nm: np.ndarray
    actual_knee_torque_nm: np.ndarray
    actual_metrics: MechanicalTorqueMetrics
    observation_valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class SubjectPersonalizationResult:
    subject_id: str
    truth_scenario: str
    history: pd.DataFrame
    candidate_audit: pd.DataFrame
    torque_audit: pd.DataFrame
    parameter_history: pd.DataFrame
    heldout_generalization: pd.DataFrame
    final_alpha: SearchAlpha
    final_trajectory: GeneratedTrajectory
    summary: dict[str, Any]
    data_leakage_audit: dict[str, Any]


EstimatorFunction = Callable[..., ParameterEstimationResult]
GeneratorFunction = Callable[..., GeneratedTrajectory]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_search_bounds(alpha: SearchAlpha) -> bool:
    values = {
        "hip_amplitude_delta_deg": alpha.hip_delta_deg,
        "knee_amplitude_delta_deg": alpha.knee_delta_deg,
        "knee_phase_shift": alpha.phase_delta,
    }
    return all(
        OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name][0] - 1e-15
        <= float(value)
        <= OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name][1] + 1e-15
        for name, value in values.items()
    )


def build_coordinate_neighborhood(
    current: SearchAlpha,
    steps: TrustRegionSteps,
) -> tuple[SearchAlpha, ...]:
    """Build current and six signed coordinate moves without clipping."""

    raw = (
        current,
        SearchAlpha(current.hip_delta_deg + steps.hip_deg, current.knee_delta_deg, current.phase_delta),
        SearchAlpha(current.hip_delta_deg - steps.hip_deg, current.knee_delta_deg, current.phase_delta),
        SearchAlpha(current.hip_delta_deg, current.knee_delta_deg + steps.knee_deg, current.phase_delta),
        SearchAlpha(current.hip_delta_deg, current.knee_delta_deg - steps.knee_deg, current.phase_delta),
        SearchAlpha(current.hip_delta_deg, current.knee_delta_deg, current.phase_delta + steps.phase),
        SearchAlpha(current.hip_delta_deg, current.knee_delta_deg, current.phase_delta - steps.phase),
    )
    unique: list[SearchAlpha] = []
    seen: set[tuple[float, float, float]] = set()
    for alpha in raw:
        if alpha.key() not in seen:
            unique.append(alpha)
            seen.add(alpha.key())
    if len(unique) > 7:
        raise RuntimeError("coordinate neighborhood exceeded seven points")
    return tuple(unique)


def shrink_steps(steps: TrustRegionSteps) -> TrustRegionSteps:
    """Halve all steps and clamp only to the frozen minimum step sizes."""

    return TrustRegionSteps(
        hip_deg=max(steps.hip_deg * 0.5, MINIMUM_STEP_HIP_DEG),
        knee_deg=max(steps.knee_deg * 0.5, MINIMUM_STEP_KNEE_DEG),
        phase=max(steps.phase * 0.5, MINIMUM_STEP_PHASE),
    )


def accept_actual_trial(
    actual_j: float,
    best_actual_j: float,
    *,
    tolerance: float = OBJECTIVE_EQUIVALENCE_TOLERANCE,
) -> bool:
    return bool(float(actual_j) < float(best_actual_j) - float(tolerance))


def rejection_stop_reason(steps_after_shrink: TrustRegionSteps) -> str:
    """Return the fixed minimum-step stop marker after a rejected execution."""

    return (
        STOP_MINIMUM_STEP_WITHOUT_ACCEPTED_IMPROVEMENT
        if steps_after_shrink.at_minimum
        else ""
    )


def boundary_saturation_audit(alpha: SearchAlpha) -> dict[str, Any]:
    parameters = {
        "hip_amplitude_delta_deg": alpha.hip_delta_deg,
        "knee_amplitude_delta_deg": alpha.knee_delta_deg,
        "knee_phase_shift": alpha.phase_delta,
    }
    names: list[str] = []
    directions: list[str] = []
    for name, value in parameters.items():
        lower, upper = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name]
        if math.isclose(float(value), float(lower), abs_tol=1e-12, rel_tol=0.0):
            names.append(name)
            directions.append("lower")
        if math.isclose(float(value), float(upper), abs_tol=1e-12, rel_tol=0.0):
            names.append(name)
            directions.append("upper")
    return {
        "OBJECTIVE_BOUNDARY_SATURATION": bool(names),
        "boundary_parameter": ";".join(names),
        "boundary_direction": ";".join(directions),
        "search_bounds_expanded": False,
        "global_optimum_claimed": False,
    }


def _reference_deviation_from_generated(generated: GeneratedTrajectory) -> tuple[float, float]:
    return (
        float(generated.metadata["hip_rms_deviation_deg"]),
        float(generated.metadata["knee_rms_deviation_deg"]),
    )


def _predict_objective(
    generated: GeneratedTrajectory,
    parameters: Mapping[str, float],
    template,
    reference_metrics: MechanicalTorqueMetrics,
) -> tuple[MechanicalObjectiveResult, np.ndarray, np.ndarray]:
    trajectory = generated.trajectory
    hip, knee = predict_joint_torque(trajectory, template, parameters, L1)
    metrics = compute_torque_metrics(trajectory["time_s"], hip, knee)
    hip_deviation, knee_deviation = _reference_deviation_from_generated(generated)
    objective = evaluate_mechanical_objective(
        trajectory_id=str(generated.metadata["trajectory_id"]),
        metrics=metrics,
        reference_metrics=reference_metrics,
        hip_rms_deviation_deg=hip_deviation,
        knee_rms_deviation_deg=knee_deviation,
    )
    return objective, hip, knee


def evaluate_candidate_neighborhood(
    *,
    current: SearchAlpha,
    steps: TrustRegionSteps,
    estimated_parameters: Mapping[str, float],
    template,
    generator: GeneratorFunction = generate_personalized_trajectory,
    admissible_region: AdmissibleRegionArtifacts | None = None,
) -> tuple[list[CandidateEvaluation], MechanicalTorqueMetrics]:
    """Generate/audit/predict a local neighborhood without any truth access."""

    verified_region = (
        load_admissible_personalization_region()
        if admissible_region is None
        else admissible_region
    )
    reference = generator(**SearchAlpha().as_generator_parameters())
    reference_admissibility = evaluate_admissible_personalization_region(
        reference, region=verified_region
    )
    if not reference_admissibility.trajectory_admissible:
        raise RuntimeError("frozen reference failed the unified admissibility gate")
    reference_hip, reference_knee = predict_joint_torque(
        reference.trajectory, template, estimated_parameters, L1
    )
    reference_metrics = compute_torque_metrics(
        reference.trajectory["time_s"], reference_hip, reference_knee
    )
    evaluations: list[CandidateEvaluation] = []
    for alpha in build_coordinate_neighborhood(current, steps):
        base_row: dict[str, Any] = {
            "alpha_hip": alpha.hip_delta_deg,
            "alpha_knee": alpha.knee_delta_deg,
            "alpha_phase": alpha.phase_delta,
            "time_scale": FIXED_TIME_SCALE,
            "parent_reference_id": ACTIVE_REFERENCE_ID,
            "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        }
        if not _inside_search_bounds(alpha):
            base_row.update(
                {
                    "trajectory_id": f"out_of_bounds_{alpha.key()}",
                    "trajectory_feasible": False,
                    "trajectory_admissible": False,
                    "generator_trajectory_feasible": False,
                    "invalid_reason": "generator_bounds",
                    "domain_coverage": math.nan,
                }
            )
            evaluations.append(CandidateEvaluation(alpha, None, base_row))
            continue
        generated = generator(**alpha.as_generator_parameters())
        admissibility = evaluate_admissible_personalization_region(
            generated, region=verified_region
        )
        base_row.update(
            {
                "trajectory_id": str(generated.metadata["trajectory_id"]),
                "trajectory_feasible": bool(admissibility.trajectory_admissible),
                "trajectory_admissible": bool(admissibility.trajectory_admissible),
                "generator_trajectory_feasible": bool(
                    generated.constraints.trajectory_feasible
                ),
                "invalid_reason": str(admissibility.invalid_reason),
                "domain_coverage": float(admissibility.domain_coverage_percent),
                "first_invalid_sample": admissibility.first_invalid_sample,
                "invalid_phase": admissibility.invalid_phase,
                "joint_corridor_valid": admissibility.joint_corridor_valid,
                "pull_corridor_valid": admissibility.pull_corridor_valid,
                "trajectory_sha256": str(generated.metadata["trajectory_sha256"]),
                "reference_deviation": math.nan,
                "combined_peak_ratio": math.nan,
                "combined_torque_rate_ratio": math.nan,
                "mechanical_cost_j_rms": math.nan,
            }
        )
        if admissibility.trajectory_admissible:
            objective, _, _ = _predict_objective(
                generated, estimated_parameters, template, reference_metrics
            )
            base_row.update(objective.as_dict())
        evaluations.append(CandidateEvaluation(alpha, generated, base_row))
    return evaluations, reference_metrics


def propose_next_trial(
    *,
    current: SearchAlpha,
    steps: TrustRegionSteps,
    estimated_parameters: Mapping[str, float],
    template,
    generator: GeneratorFunction = generate_personalized_trajectory,
    admissible_region: AdmissibleRegionArtifacts | None = None,
    equivalence_tolerance: float = OBJECTIVE_EQUIVALENCE_TOLERANCE,
) -> ProposalResult:
    """Select exactly one predicted-improving feasible candidate, or stop."""

    evaluations, _ = evaluate_candidate_neighborhood(
        current=current,
        steps=steps,
        estimated_parameters=estimated_parameters,
        template=template,
        generator=generator,
        admissible_region=admissible_region,
    )
    audit = pd.DataFrame([item.row for item in evaluations])
    feasible = audit.loc[audit["trajectory_feasible"].astype(bool)].copy()
    if feasible.empty:
        return ProposalResult(None, audit, None, None, STOP_NO_FEASIBLE_NEIGHBOR)
    current_mask = (
        np.isclose(feasible["alpha_hip"], current.hip_delta_deg, atol=1e-12, rtol=0.0)
        & np.isclose(feasible["alpha_knee"], current.knee_delta_deg, atol=1e-12, rtol=0.0)
        & np.isclose(feasible["alpha_phase"], current.phase_delta, atol=1e-12, rtol=0.0)
    )
    if current_mask.sum() != 1:
        raise RuntimeError("current trust-region center is not uniquely feasible")
    current_j = float(feasible.loc[current_mask, "mechanical_cost_j_rms"].iloc[0])
    if int((~current_mask).sum()) == 0:
        return ProposalResult(
            None,
            audit,
            current_j,
            None,
            STOP_NO_FEASIBLE_NEIGHBOR,
        )
    ranked = rank_feasible_candidates(
        audit, equivalence_tolerance=equivalence_tolerance
    )
    audit = audit.merge(
        ranked[
            [
                "trajectory_id",
                "deterministic_rank",
                "mechanically_equivalent_to_minimum",
            ]
        ],
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    )
    selected_id = str(ranked.iloc[0]["trajectory_id"])
    selected = next(item for item in evaluations if item.row["trajectory_id"] == selected_id)
    predicted_j = float(selected.row["mechanical_cost_j_rms"])
    improvement = current_j - predicted_j
    if selected.alpha.key() == current.key() or improvement < equivalence_tolerance - 1e-15:
        return ProposalResult(
            None,
            audit,
            current_j,
            improvement,
            STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE,
        )
    return ProposalResult(selected, audit, current_j, improvement, "")


class Stage45CVirtualTruthOracle:
    """Private post-proposal truth layer backed by an existing mismatch scenario."""

    def __init__(self, subject_id: str, scenario_name: str) -> None:
        self.subject_id = str(subject_id)
        self.scenario_name = str(scenario_name)
        base = get_dynamic_subject(self.subject_id)
        scenario = get_mismatch_scenario(self.scenario_name)
        self._truth_subject = scenario.create_subject(base)
        self._residual_random_seed = int(scenario.random_seed)
        self._truth_calls = 0

    @property
    def truth_calls(self) -> int:
        return self._truth_calls

    def simulate(self, trajectory: pd.DataFrame) -> TruthExecution:
        required = {
            "time_s",
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        }
        missing = required.difference(trajectory.columns)
        if missing:
            raise ValueError(f"truth trajectory missing columns: {sorted(missing)}")
        self._truth_calls += 1
        q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
        q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
        dynamics = mismatch_inverse_dynamics(
            q_hip,
            q_knee,
            trajectory["dq_hip_rad_s"].to_numpy(dtype=float),
            trajectory["dq_knee_rad_s"].to_numpy(dtype=float),
            trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float),
            trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float),
            self._truth_subject,
            L1,
            residual_random_seed=self._residual_random_seed,
        )
        hip = np.asarray(dynamics.tau_total_hip_nm, dtype=float)
        knee = np.asarray(dynamics.tau_total_knee_nm, dtype=float)
        force = endpoint_force_from_joint_torque(q_hip, q_knee, hip, knee, L1, L2)
        force_valid = np.asarray(force.force_mapping_valid, dtype=bool)
        finite = np.isfinite(
            np.column_stack(
                (
                    q_hip,
                    q_knee,
                    trajectory["dq_hip_rad_s"],
                    trajectory["dq_knee_rad_s"],
                    trajectory["ddq_hip_rad_s2"],
                    trajectory["ddq_knee_rad_s2"],
                    force.fx_robot_on_leg_n,
                    force.fz_robot_on_leg_n,
                )
            )
        ).all(axis=1)
        sample_valid = force_valid & finite
        reasons = np.asarray(force.invalid_reason, dtype=str)
        reasons = np.where(sample_valid, "", reasons)
        observation = pd.DataFrame(
            {
                "q_hip_rad": q_hip,
                "q_knee_rad": q_knee,
                "dq_hip_rad_s": trajectory["dq_hip_rad_s"].to_numpy(dtype=float),
                "dq_knee_rad_s": trajectory["dq_knee_rad_s"].to_numpy(dtype=float),
                "ddq_hip_rad_s2": trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float),
                "ddq_knee_rad_s2": trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float),
                "fx_observed_n": np.asarray(force.fx_robot_on_leg_n, dtype=float),
                "fz_observed_n": np.asarray(force.fz_robot_on_leg_n, dtype=float),
                "sample_valid": sample_valid,
                "force_mapping_valid": force_valid,
                "wrench_is_stale": False,
                "invalid_reason": reasons,
            }
        )
        projected = project_estimator_inputs(observation)
        invalid = sorted(set(reasons[~sample_valid])) if (~sample_valid).any() else []
        return TruthExecution(
            estimator_observations=projected,
            actual_hip_torque_nm=hip,
            actual_knee_torque_nm=knee,
            actual_metrics=compute_torque_metrics(trajectory["time_s"], hip, knee),
            observation_valid=bool(sample_valid.all()),
            invalid_reason=";".join(value for value in invalid if value),
        )


def _load_identification_rows(
    subject_id: str,
    split: str,
    *,
    path: str | Path = INITIAL_IDENTIFICATION_DATASET_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one declared role; only the projected table may reach fitting."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"formal reference-local dataset missing: {source}")
    dataframe = pd.read_csv(source)
    required_provenance = {
        "subject_id",
        "trajectory_id",
        "dataset_split",
        "active_reference_identifier",
        "active_reference_sha256",
        *ESTIMATOR_INPUT_COLUMNS,
    }
    # The persisted clean dataset predates the explicit stale flag; it is
    # deterministic software data, so the missing flag is added before the
    # strict estimator projection.
    if "wrench_is_stale" not in dataframe:
        dataframe["wrench_is_stale"] = False
    missing = required_provenance.difference(dataframe.columns)
    if missing:
        raise RuntimeError(f"identification dataset missing columns: {sorted(missing)}")
    selected = dataframe.loc[
        dataframe["subject_id"].astype(str).eq(str(subject_id))
        & dataframe["dataset_split"].astype(str).eq(str(split))
    ].copy()
    if selected.empty:
        raise RuntimeError(f"no {split!r} rows for subject {subject_id!r}")
    if not selected["active_reference_identifier"].astype(str).eq(ACTIVE_REFERENCE_ID).all():
        raise RuntimeError("identification data belongs to another reference")
    if not selected["active_reference_sha256"].astype(str).eq(ACTIVE_REFERENCE_SHA256).all():
        raise RuntimeError("identification data parent SHA mismatch")
    role_counts = dataframe.loc[
        dataframe["subject_id"].astype(str).eq(str(subject_id)), "dataset_split"
    ].astype(str).value_counts().to_dict()
    audit = {
        "source_path": str(source.resolve()),
        "source_sha256": _file_sha256(source),
        "requested_role": str(split),
        "returned_rows": int(len(selected)),
        "available_role_row_counts_not_passed_to_estimator": {
            str(key): int(value) for key, value in role_counts.items()
        },
    }
    return project_estimator_inputs(selected), audit


def _fit_parameters(
    training: pd.DataFrame,
    template,
    *,
    initial_guess: Mapping[str, float],
    estimator: EstimatorFunction,
) -> ParameterEstimationResult:
    projected = project_estimator_inputs(training)
    return estimator(
        projected,
        template,
        L1,
        L2,
        initial_guess=initial_guess,
        bounds=(identification_lower_bounds, identification_upper_bounds),
    )


def _parameter_columns(
    parameters: Mapping[str, float],
    *,
    suffix: str = "",
) -> dict[str, float]:
    return {
        f"{output}{suffix}": float(parameters[name])
        for name, output in _PARAMETER_HISTORY_NAMES.items()
    }


def _parameter_delta_columns(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    return {
        f"parameter_delta_{name}": float(after[name]) - float(before[name])
        for name in PARAMETER_NAMES
    }


def _actual_objective(
    generated: GeneratedTrajectory,
    execution: TruthExecution,
    reference_metrics: MechanicalTorqueMetrics,
) -> MechanicalObjectiveResult:
    hip_deviation, knee_deviation = _reference_deviation_from_generated(generated)
    return evaluate_mechanical_objective(
        trajectory_id=str(generated.metadata["trajectory_id"]),
        metrics=execution.actual_metrics,
        reference_metrics=reference_metrics,
        hip_rms_deviation_deg=hip_deviation,
        knee_rms_deviation_deg=knee_deviation,
    )


def _torque_audit_table(
    *,
    subject_id: str,
    scenario_name: str,
    trial_id: int,
    trajectory: pd.DataFrame,
    predicted_hip: np.ndarray,
    predicted_knee: np.ndarray,
    actual: TruthExecution,
    accepted: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "truth_scenario": scenario_name,
            "trial_id": int(trial_id),
            "sample_index": np.arange(len(trajectory), dtype=int),
            "time_s": trajectory["time_s"].to_numpy(dtype=float),
            "predicted_hip_torque_nm": np.asarray(predicted_hip, dtype=float),
            "actual_hip_torque_nm": actual.actual_hip_torque_nm,
            "predicted_knee_torque_nm": np.asarray(predicted_knee, dtype=float),
            "actual_knee_torque_nm": actual.actual_knee_torque_nm,
            "accepted": bool(accepted),
            "data_role": "sequential_executed_adaptation_audit",
            "used_for_next_model_update": bool(actual.observation_valid),
        }
    )


def _history_row(
    *,
    subject_id: str,
    scenario_name: str,
    trial_id: int,
    current: SearchAlpha,
    steps: TrustRegionSteps,
    steps_after: TrustRegionSteps,
    proposed: SearchAlpha,
    best_alpha_after: SearchAlpha,
    best_trajectory_id_after: str,
    predicted: MechanicalObjectiveResult,
    predicted_improvement: float,
    actual: MechanicalObjectiveResult,
    best_before: float,
    best_after: float,
    accepted: bool,
    rejection_reason: str,
    generated: GeneratedTrajectory,
    parameters_before: Mapping[str, float],
    parameters_after: Mapping[str, float],
    model_update_success: bool,
    stop_reason: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject_id": subject_id,
        "truth_scenario": scenario_name,
        "iteration": int(trial_id),
        "trial_id": int(trial_id),
        "executed": True,
        "trajectory_id": str(generated.metadata["trajectory_id"]),
        "trajectory_sha256": str(generated.metadata["trajectory_sha256"]),
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "alpha_current_hip": current.hip_delta_deg,
        "alpha_current_knee": current.knee_delta_deg,
        "alpha_current_phase": current.phase_delta,
        "step_hip": steps.hip_deg,
        "step_knee": steps.knee_deg,
        "step_phase": steps.phase,
        "step_hip_after": steps_after.hip_deg,
        "step_knee_after": steps_after.knee_deg,
        "step_phase_after": steps_after.phase,
        "proposed_alpha_hip": proposed.hip_delta_deg,
        "proposed_alpha_knee": proposed.knee_delta_deg,
        "proposed_alpha_phase": proposed.phase_delta,
        "best_alpha_after_hip": best_alpha_after.hip_delta_deg,
        "best_alpha_after_knee": best_alpha_after.knee_delta_deg,
        "best_alpha_after_phase": best_alpha_after.phase_delta,
        "best_accepted_trajectory_id_after": str(best_trajectory_id_after),
        "predicted_J": predicted.mechanical_cost_j_rms,
        "actual_J": actual.mechanical_cost_j_rms,
        "best_J_before": best_before,
        "best_J_after": best_after,
        "predicted_improvement": float(predicted_improvement),
        "actual_improvement": best_before - actual.mechanical_cost_j_rms,
        "prediction_error": actual.mechanical_cost_j_rms - predicted.mechanical_cost_j_rms,
        "accepted": bool(accepted),
        "rejection_reason": rejection_reason,
        "trajectory_feasible": bool(generated.constraints.trajectory_feasible),
        "domain_coverage": float(generated.constraints.domain_coverage),
        "hip_rms_torque": actual.metrics.hip_rms_torque_nm,
        "knee_rms_torque": actual.metrics.knee_rms_torque_nm,
        "hip_peak_torque": actual.metrics.hip_peak_abs_torque_nm,
        "knee_peak_torque": actual.metrics.knee_peak_abs_torque_nm,
        "torque_rate": float(
            math.sqrt(
                (
                    actual.metrics.hip_rms_torque_rate_nm_s**2
                    + actual.metrics.knee_rms_torque_rate_nm_s**2
                )
                / 2.0
            )
        ),
        "torque_rate_ratio": actual.combined_torque_rate_ratio,
        "combined_peak_ratio": actual.combined_peak_ratio,
        "predicted_hip_rms_torque": predicted.metrics.hip_rms_torque_nm,
        "predicted_knee_rms_torque": predicted.metrics.knee_rms_torque_nm,
        "predicted_hip_peak_torque": predicted.metrics.hip_peak_abs_torque_nm,
        "predicted_knee_peak_torque": predicted.metrics.knee_peak_abs_torque_nm,
        "reference_deviation": actual.reference_deviation,
        "model_update_success": bool(model_update_success),
        "stop_reason": stop_reason,
    }
    row.update(_parameter_columns(parameters_before))
    row.update(_parameter_columns(parameters_after, suffix="_after_update"))
    row.update(_parameter_delta_columns(parameters_before, parameters_after))
    return row


def _fit_or_failure(
    training: pd.DataFrame,
    template,
    parameters: Mapping[str, float],
    estimator: EstimatorFunction,
) -> tuple[dict[str, float], ParameterEstimationResult | None, str]:
    try:
        result = _fit_parameters(
            training,
            template,
            initial_guess=parameters,
            estimator=estimator,
        )
    except Exception as exc:  # saved as an explicit fail-closed experiment outcome
        return dict(parameters), None, f"{type(exc).__name__}: {exc}"
    if not result.optimizer_success:
        return dict(parameters), result, result.optimizer_message
    values = dict(result.estimated_parameters)
    if set(values) != set(PARAMETER_NAMES) or not np.isfinite(
        np.asarray(list(values.values()), dtype=float)
    ).all():
        return dict(parameters), result, "estimator returned invalid parameter schema"
    return values, result, ""


def _final_heldout_evaluation(
    *,
    subject_id: str,
    oracle: Stage45CVirtualTruthOracle,
    parameters: Mapping[str, float],
    template,
    actual_reference_metrics: MechanicalTorqueMetrics,
    path: str | Path,
) -> pd.DataFrame:
    heldout, _ = _load_identification_rows(subject_id, "test", path=path)
    # State/trajectory identity is loaded separately for final-only grouping;
    # no held-out observation or force column is passed to fitting.
    source = pd.read_csv(path)
    source = source.loc[
        source["subject_id"].astype(str).eq(subject_id)
        & source["dataset_split"].astype(str).eq("test")
    ].copy()
    if len(source) != len(heldout):
        raise RuntimeError("held-out state and strict-role projections disagree")
    rows: list[dict[str, Any]] = []
    for trajectory_id, trajectory in source.groupby("trajectory_id", sort=True):
        trajectory = trajectory.reset_index(drop=True)
        execution = oracle.simulate(trajectory)
        predicted_hip, predicted_knee = predict_joint_torque(
            trajectory, template, parameters, L1
        )
        residual_hip = execution.actual_hip_torque_nm - predicted_hip
        residual_knee = execution.actual_knee_torque_nm - predicted_knee
        actual_objective = evaluate_mechanical_objective(
            trajectory_id=str(trajectory_id),
            metrics=execution.actual_metrics,
            reference_metrics=actual_reference_metrics,
            hip_rms_deviation_deg=0.0,
            knee_rms_deviation_deg=0.0,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "truth_scenario": oracle.scenario_name,
                "trajectory_id": str(trajectory_id),
                "data_role": "held_out_final_test_only",
                "used_for_proposal": False,
                "used_for_parameter_fitting": False,
                "used_for_ranking": False,
                "used_for_stopping": False,
                "valid_samples": int(execution.observation_valid) * len(trajectory),
                "actual_J_vs_frozen_reference": actual_objective.mechanical_cost_j_rms,
                "prediction_rmse_hip_nm": float(np.sqrt(np.mean(residual_hip**2))),
                "prediction_rmse_knee_nm": float(np.sqrt(np.mean(residual_knee**2))),
                "prediction_rmse_combined_nm": float(
                    np.sqrt(np.mean(np.concatenate((residual_hip, residual_knee)) ** 2))
                ),
            }
        )
    return pd.DataFrame(rows)


def run_subject_personalization(
    subject_id: str,
    truth_scenario: str,
    *,
    max_trials: int = MAX_EXECUTED_TRIALS,
    model_reliability_threshold: float | None = MODEL_RELIABILITY_THRESHOLD,
    identification_dataset_path: str | Path = INITIAL_IDENTIFICATION_DATASET_PATH,
    generator: GeneratorFunction = generate_personalized_trajectory,
    estimator: EstimatorFunction = estimate_subject_parameters,
    truth_oracle_factory: Callable[[str, str], Any] = Stage45CVirtualTruthOracle,
) -> SubjectPersonalizationResult:
    """Run one isolated subject/scenario experiment, including final-only test."""

    if subject_id not in FORMAL_SUBJECT_IDS:
        raise ValueError(f"unsupported formal subject: {subject_id}")
    if truth_scenario not in FORMAL_TRUTH_SCENARIOS:
        raise ValueError(f"unsupported sequential truth scenario: {truth_scenario}")
    if isinstance(max_trials, bool) or int(max_trials) < 1:
        raise ValueError("max_trials must be a positive integer")
    max_trials = int(max_trials)
    if model_reliability_threshold is not None:
        threshold = float(model_reliability_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("model reliability threshold must be finite and non-negative")

    validate_active_reference_file()
    base_subject = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(base_subject)
    initial_training, initial_audit = _load_identification_rows(
        subject_id, "train", path=identification_dataset_path
    )
    initial_result = _fit_parameters(
        initial_training,
        template,
        initial_guess=identification_initial_guess,
        estimator=estimator,
    )
    if not initial_result.optimizer_success:
        raise RuntimeError(f"initial identification failed: {initial_result.optimizer_message}")
    parameters = dict(initial_result.estimated_parameters)
    oracle = truth_oracle_factory(subject_id, truth_scenario)
    best_alpha = SearchAlpha()
    steps = TrustRegionSteps()
    reference = generator(**best_alpha.as_generator_parameters())
    if str(reference.metadata["parent_reference_sha256"]) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("continuous generator did not propagate the frozen parent SHA")

    reference_execution = oracle.simulate(reference.trajectory)
    if not reference_execution.observation_valid:
        raise RuntimeError(
            "reference truth observation is invalid: " + reference_execution.invalid_reason
        )
    actual_reference_metrics = reference_execution.actual_metrics
    predicted_ref_hip, predicted_ref_knee = predict_joint_torque(
        reference.trajectory, template, parameters, L1
    )
    predicted_reference_metrics = compute_torque_metrics(
        reference.trajectory["time_s"], predicted_ref_hip, predicted_ref_knee
    )
    predicted_reference = evaluate_mechanical_objective(
        trajectory_id=str(reference.metadata["trajectory_id"]),
        metrics=predicted_reference_metrics,
        reference_metrics=predicted_reference_metrics,
        hip_rms_deviation_deg=0.0,
        knee_rms_deviation_deg=0.0,
    )
    actual_reference = _actual_objective(
        reference, reference_execution, actual_reference_metrics
    )
    best_actual_j = actual_reference.mechanical_cost_j_rms
    best_generated = reference
    adaptation_frames = [reference_execution.estimator_observations.copy(deep=True)]
    combined_training = pd.concat((initial_training, *adaptation_frames), ignore_index=True)
    updated_parameters, update_result, update_error = _fit_or_failure(
        combined_training, template, parameters, estimator
    )
    model_update_success = update_result is not None and not update_error
    initial_stop = STOP_MODEL_UPDATE_FAILED if not model_update_success else ""
    history_rows = [
        _history_row(
            subject_id=subject_id,
            scenario_name=truth_scenario,
            trial_id=0,
            current=SearchAlpha(),
            steps=steps,
            steps_after=steps,
            proposed=SearchAlpha(),
            best_alpha_after=SearchAlpha(),
            best_trajectory_id_after=str(reference.metadata["trajectory_id"]),
            predicted=predicted_reference,
            predicted_improvement=0.0,
            actual=actual_reference,
            best_before=1.0,
            best_after=1.0,
            accepted=True,
            rejection_reason="",
            generated=reference,
            parameters_before=parameters,
            parameters_after=updated_parameters,
            model_update_success=model_update_success,
            stop_reason=initial_stop,
        )
    ]
    torque_frames = [
        _torque_audit_table(
            subject_id=subject_id,
            scenario_name=truth_scenario,
            trial_id=0,
            trajectory=reference.trajectory,
            predicted_hip=predicted_ref_hip,
            predicted_knee=predicted_ref_knee,
            actual=reference_execution,
            accepted=True,
        )
    ]
    parameter_rows = [
        {
            "subject_id": subject_id,
            "truth_scenario": truth_scenario,
            "trial_id": 0,
            "model_update_success": model_update_success,
            "model_update_error": update_error,
            "adaptation_trial_count": 1,
            **_parameter_columns(updated_parameters),
            **_parameter_delta_columns(parameters, updated_parameters),
        }
    ]
    parameters = updated_parameters
    candidate_frames: list[pd.DataFrame] = []
    stop_reason = initial_stop
    proposal_truth_call_audit: list[dict[str, int]] = []

    for trial_id in range(1, max_trials):
        if stop_reason:
            break
        center_before = best_alpha
        steps_before = steps
        truth_calls_before = oracle.truth_calls
        proposal = propose_next_trial(
            current=center_before,
            steps=steps_before,
            estimated_parameters=parameters,
            template=template,
            generator=generator,
        )
        truth_calls_after = oracle.truth_calls
        proposal_truth_call_audit.append(
            {
                "trial_id": trial_id,
                "truth_calls_before_proposal": truth_calls_before,
                "truth_calls_after_proposal": truth_calls_after,
            }
        )
        if truth_calls_after != truth_calls_before:
            raise RuntimeError("proposal accessed the virtual truth oracle")
        audit = proposal.candidate_audit.copy()
        audit.insert(0, "iteration", trial_id)
        audit.insert(0, "truth_scenario", truth_scenario)
        audit.insert(0, "subject_id", subject_id)
        audit["proposed_for_execution"] = False
        if proposal.candidate is not None:
            audit.loc[
                audit["trajectory_id"].astype(str).eq(
                    str(proposal.candidate.row["trajectory_id"])
                ),
                "proposed_for_execution",
            ] = True
        candidate_frames.append(audit)
        if proposal.candidate is None:
            stop_reason = proposal.stop_reason
            break
        candidate = proposal.candidate
        if candidate.generated is None:
            raise RuntimeError("selected proposal lacks a generated trajectory")
        generated = candidate.generated
        predicted_reference_hip, predicted_reference_knee = predict_joint_torque(
            reference.trajectory, template, parameters, L1
        )
        current_predicted_reference_metrics = compute_torque_metrics(
            reference.trajectory["time_s"],
            predicted_reference_hip,
            predicted_reference_knee,
        )
        predicted, predicted_hip, predicted_knee = _predict_objective(
            generated, parameters, template, current_predicted_reference_metrics
        )
        execution = oracle.simulate(generated.trajectory)
        actual = _actual_objective(generated, execution, actual_reference_metrics)
        best_before = best_actual_j
        accepted = bool(
            execution.observation_valid
            and accept_actual_trial(actual.mechanical_cost_j_rms, best_actual_j)
        )
        if not execution.observation_valid:
            rejection_reason = "truth_observation_invalid"
        elif accepted:
            rejection_reason = ""
        elif actual.mechanical_cost_j_rms >= best_actual_j:
            rejection_reason = "actual_not_better_than_verified_best"
        else:
            rejection_reason = "actual_improvement_within_equivalence_tolerance"
        if accepted:
            best_alpha = candidate.alpha
            best_actual_j = actual.mechanical_cost_j_rms
            best_generated = generated
        else:
            steps = shrink_steps(steps)

        parameters_before = dict(parameters)
        model_update_success = False
        update_error = "truth observation invalid"
        update_result = None
        if execution.observation_valid:
            adaptation_frames.append(execution.estimator_observations.copy(deep=True))
            combined_training = pd.concat(
                (initial_training, *adaptation_frames), ignore_index=True
            )
            parameters, update_result, update_error = _fit_or_failure(
                combined_training, template, parameters_before, estimator
            )
            model_update_success = update_result is not None and not update_error
        if not model_update_success:
            stop_reason = STOP_MODEL_UPDATE_FAILED

        prediction_error = actual.mechanical_cost_j_rms - predicted.mechanical_cost_j_rms
        if not stop_reason:
            if model_reliability_threshold is None:
                stop_reason = STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD
            elif abs(prediction_error) > float(model_reliability_threshold):
                stop_reason = STOP_MODEL_RELIABILITY_GATE_EXCEEDED
                steps = shrink_steps(steps)
            elif not accepted:
                stop_reason = rejection_stop_reason(steps)

        history_rows.append(
            _history_row(
                subject_id=subject_id,
                scenario_name=truth_scenario,
                trial_id=trial_id,
                current=center_before,
                steps=steps_before,
                steps_after=steps,
                proposed=candidate.alpha,
                best_alpha_after=best_alpha,
                best_trajectory_id_after=str(best_generated.metadata["trajectory_id"]),
                predicted=predicted,
                predicted_improvement=float(proposal.predicted_improvement),
                actual=actual,
                best_before=best_before,
                best_after=best_actual_j,
                accepted=accepted,
                rejection_reason=rejection_reason,
                generated=generated,
                parameters_before=parameters_before,
                parameters_after=parameters,
                model_update_success=model_update_success,
                stop_reason=stop_reason,
            )
        )
        torque_frames.append(
            _torque_audit_table(
                subject_id=subject_id,
                scenario_name=truth_scenario,
                trial_id=trial_id,
                trajectory=generated.trajectory,
                predicted_hip=predicted_hip,
                predicted_knee=predicted_knee,
                actual=execution,
                accepted=accepted,
            )
        )
        parameter_rows.append(
            {
                "subject_id": subject_id,
                "truth_scenario": truth_scenario,
                "trial_id": trial_id,
                "model_update_success": model_update_success,
                "model_update_error": update_error,
                "adaptation_trial_count": len(adaptation_frames),
                **_parameter_columns(parameters),
                **_parameter_delta_columns(parameters_before, parameters),
            }
        )

    if not stop_reason:
        stop_reason = STOP_MAX_TRIALS
    history_rows[-1]["stop_reason"] = stop_reason
    final_trajectory = generator(**best_alpha.as_generator_parameters())
    if str(final_trajectory.metadata["trajectory_sha256"]) != str(
        best_generated.metadata["trajectory_sha256"]
    ):
        raise RuntimeError("final fallback trajectory is not the verified accepted best")
    boundary = boundary_saturation_audit(best_alpha)
    heldout = _final_heldout_evaluation(
        subject_id=subject_id,
        oracle=oracle,
        parameters=parameters,
        template=template,
        actual_reference_metrics=actual_reference_metrics,
        path=identification_dataset_path,
    )
    history = pd.DataFrame(history_rows)
    candidate_audit = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame()
    )
    torque_audit = pd.concat(torque_frames, ignore_index=True)
    parameter_history = pd.DataFrame(parameter_rows)
    accepted_rows = history.loc[history["accepted"].astype(bool)]
    final_row = accepted_rows.iloc[-1]
    reference_peak = float(
        math.sqrt(
            (
                actual_reference.metrics.hip_peak_abs_torque_nm**2
                + actual_reference.metrics.knee_peak_abs_torque_nm**2
            )
            / 2.0
        )
    )
    final_peak = float(
        math.sqrt(
            (final_row["hip_peak_torque"] ** 2 + final_row["knee_peak_torque"] ** 2)
            / 2.0
        )
    )
    reference_rate = float(
        math.sqrt(
            (
                actual_reference.metrics.hip_rms_torque_rate_nm_s**2
                + actual_reference.metrics.knee_rms_torque_rate_nm_s**2
            )
            / 2.0
        )
    )
    final_rate = float(final_row["torque_rate"])
    summary: dict[str, Any] = {
        "subject": subject_id,
        "truth_scenario": truth_scenario,
        "number_of_executed_trials": int(len(history)),
        "number_of_accepted_trials": int(history["accepted"].astype(bool).sum()),
        "number_of_accepted_improvements": int(
            history.loc[history["trial_id"].astype(int).gt(0), "accepted"].astype(bool).sum()
        ),
        "final_hip_delta": best_alpha.hip_delta_deg,
        "final_knee_delta": best_alpha.knee_delta_deg,
        "final_phase_delta": best_alpha.phase_delta,
        "reference_actual_J": 1.0,
        "final_actual_J": best_actual_j,
        "mechanical_reduction_percent": 100.0 * (1.0 - best_actual_j),
        "reference_peak_torque": reference_peak,
        "final_peak_torque": final_peak,
        "reference_torque_rate": reference_rate,
        "final_torque_rate": final_rate,
        "final_reference_deviation": float(final_row["reference_deviation"]),
        "boundary_saturation": bool(boundary["OBJECTIVE_BOUNDARY_SATURATION"]),
        "boundary_parameter": boundary["boundary_parameter"],
        "boundary_direction": boundary["boundary_direction"],
        "stop_reason": stop_reason,
        "final_trajectory_id": str(final_trajectory.metadata["trajectory_id"]),
        "final_trajectory_sha256": str(final_trajectory.metadata["trajectory_sha256"]),
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
    }
    leakage = {
        "subject_id": subject_id,
        "truth_scenario": truth_scenario,
        "initial_identification_role": "train",
        "initial_identification_audit": initial_audit,
        "sequential_executed_adaptation_trials": len(adaptation_frames),
        "validation_rows_used_for_proposal_or_fitting": 0,
        "heldout_rows_used_for_proposal": 0,
        "heldout_rows_used_for_parameter_fitting": 0,
        "heldout_rows_used_for_ranking": 0,
        "heldout_rows_used_for_stopping": 0,
        "heldout_evaluation_timing": "once_after_personalization_stop",
        "heldout_final_trajectory_count": int(len(heldout)),
        "estimator_input_columns": list(ESTIMATOR_INPUT_COLUMNS),
        "estimator_received_subject_or_scenario_id": False,
        "estimator_received_truth_parameters": False,
        "proposal_received_truth_parameters": False,
        "proposal_truth_call_audit": proposal_truth_call_audit,
        "truth_calls_unchanged_during_every_proposal": all(
            row["truth_calls_before_proposal"] == row["truth_calls_after_proposal"]
            for row in proposal_truth_call_audit
        ),
        "truth_scenario_queried_only_after_selection": True,
        "data_leakage_detected": False,
    }
    return SubjectPersonalizationResult(
        subject_id=subject_id,
        truth_scenario=truth_scenario,
        history=history,
        candidate_audit=candidate_audit,
        torque_audit=torque_audit,
        parameter_history=parameter_history,
        heldout_generalization=heldout,
        final_alpha=best_alpha,
        final_trajectory=final_trajectory,
        summary=summary,
        data_leakage_audit=leakage,
    )


def run_sequential_personalization_experiment(
    *,
    subject_ids: Sequence[str] = FORMAL_SUBJECT_IDS,
    truth_scenarios: Sequence[str] = FORMAL_TRUTH_SCENARIOS,
    max_trials: int = MAX_EXECUTED_TRIALS,
    model_reliability_threshold: float | None = MODEL_RELIABILITY_THRESHOLD,
) -> dict[tuple[str, str], SubjectPersonalizationResult]:
    """Run independent subject/scenario searches with no shared mutable state."""

    results: dict[tuple[str, str], SubjectPersonalizationResult] = {}
    for scenario in truth_scenarios:
        for subject_id in subject_ids:
            key = (str(subject_id), str(scenario))
            if key in results:
                raise ValueError(f"duplicate experiment key: {key}")
            results[key] = run_subject_personalization(
                key[0],
                key[1],
                max_trials=max_trials,
                model_reliability_threshold=model_reliability_threshold,
            )
    return results


def optimizer_metadata() -> dict[str, Any]:
    return {
        "sequential_personalization_version": SEQUENTIAL_PERSONALIZATION_VERSION,
        "mechanical_objective_version": MECHANICAL_OBJECTIVE_VERSION,
        "mechanical_objective_formula": "sqrt((R_h^2 + R_k^2) / 2)",
        "mechanically_equivalent_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "equivalence_tolerance_is_robot_safety_threshold": False,
        "ranking_tie_break": [
            "reference_deviation",
            "combined_peak_ratio",
            "combined_torque_rate_ratio",
            "trajectory_id_lexical",
        ],
        "initial_alpha": [0.0, 0.0, 0.0],
        "initial_steps": [
            INITIAL_STEP_HIP_DEG,
            INITIAL_STEP_KNEE_DEG,
            INITIAL_STEP_PHASE,
        ],
        "minimum_steps": [
            MINIMUM_STEP_HIP_DEG,
            MINIMUM_STEP_KNEE_DEG,
            MINIMUM_STEP_PHASE,
        ],
        "maximum_executed_trials": MAX_EXECUTED_TRIALS,
        "step_expansion_enabled": False,
        "model_reliability_threshold": MODEL_RELIABILITY_THRESHOLD,
        "missing_threshold_policy": STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD,
        "offline_search_bounds": {
            name: list(values)
            for name, values in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
        },
        "duration_optimization_enabled": False,
        "duration_s": 24.0,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "formal_truth_scenarios": list(FORMAL_TRUTH_SCENARIOS),
        "formal_subject_ids": list(FORMAL_SUBJECT_IDS),
        "optimizer_model": "existing_five_parameter_gray_box",
        "virtual_truth_model": "existing_stage_4_5c_mismatch_infrastructure",
        "hardware_used": False,
        "robot_motion_authorized": False,
        "clinical_or_comfort_claim": False,
    }


__all__ = [
    "FORMAL_SUBJECT_IDS",
    "FORMAL_TRUTH_SCENARIOS",
    "INITIAL_IDENTIFICATION_DATASET_PATH",
    "MAX_EXECUTED_TRIALS",
    "MODEL_RELIABILITY_THRESHOLD",
    "SearchAlpha",
    "Stage45CVirtualTruthOracle",
    "SubjectPersonalizationResult",
    "TrustRegionSteps",
    "accept_actual_trial",
    "boundary_saturation_audit",
    "build_coordinate_neighborhood",
    "evaluate_candidate_neighborhood",
    "optimizer_metadata",
    "propose_next_trial",
    "rejection_stop_reason",
    "run_sequential_personalization_experiment",
    "run_subject_personalization",
    "shrink_steps",
]
