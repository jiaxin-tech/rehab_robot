"""Reference-neighbourhood excitation, clean observations and local domain.

Only software-generated observations are produced here.  The estimator-facing
table intentionally contains no true parameter or ``tau_total`` columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import (
    L1,
    L2,
    identification_initial_guess,
    identification_lower_bounds,
    identification_upper_bounds,
)
from .dynamic_subject import DynamicVirtualSubject, get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import (
    ESTIMATED_DOMAIN_STATE_COLUMNS,
    StateDomainBounds,
    classify_state_domain,
    fit_state_domain_bounds,
)
from .parameter_estimator import (
    PARAMETER_NAMES,
    REQUIRED_OBSERVATION_COLUMNS,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    compute_torque_metrics,
    estimate_subject_parameters,
)
from .reference_execution_trajectory import (
    CLOSED_REFERENCE,
    closed_execution_phase_path,
)
from .reference_trajectory_retiming import retime_reference_path


LOCAL_TRAJECTORY_SPLIT = {
    "reference_slow": "train",
    "reference_nominal": "train",
    "hip_amplitude_minus_3deg": "train",
    "knee_amplitude_minus_3deg": "train",
    "knee_phase_advance_3pct": "validation",
    "knee_phase_delay_3pct": "test",
}
LOCAL_TRAJECTORY_DURATION_S = {
    "reference_slow": 12.0,
    "reference_nominal": 6.0,
    "hip_amplitude_minus_3deg": 6.0,
    "knee_amplitude_minus_3deg": 6.0,
    "knee_phase_advance_3pct": 6.0,
    "knee_phase_delay_3pct": 6.0,
}
SUBJECT_IDS = ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
LOCAL_DOMAIN_MODEL = "axis_aligned_6d_reference_neighbourhood_train_only"


@dataclass(frozen=True)
class LocalIdentificationResult:
    dataset: pd.DataFrame
    parameter_estimates: pd.DataFrame
    optimizer_results: dict[str, ParameterEstimationResult]
    estimated_parameters_by_subject: dict[str, dict[str, float]]
    domain_bounds: StateDomainBounds
    domain_coverage: pd.DataFrame


def _endpoint_zero_bump(global_phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(global_phase, dtype=float)
    if np.any(phase < -1e-12) or np.any(phase > 1.0 + 1e-12):
        raise ValueError("global phase must remain in [0, 1].")
    return np.sin(np.pi * phase) ** 2


def _phase_warp(local_phase: np.ndarray, shift_fraction: float) -> np.ndarray:
    """A monotone endpoint-preserving local phase warp without clipping."""

    phase = np.asarray(local_phase, dtype=float)
    shift = float(shift_fraction)
    if not np.isfinite(shift) or abs(shift) > 0.10:
        raise ValueError("phase shift must be finite and no larger than 10 percent.")
    warped = phase + shift * np.sin(np.pi * phase) ** 2
    if not np.all(np.diff(warped) > 0.0):
        raise ValueError("requested phase warp is not strictly monotone.")
    if not np.isclose(warped[0], 0.0) or not np.isclose(warped[-1], 1.0):
        raise RuntimeError("phase warp did not preserve its endpoints.")
    return warped


def perturb_closed_phase_path(
    base_phase_path: pd.DataFrame,
    *,
    trajectory_id: str,
    hip_amplitude_reduction_deg: float = 0.0,
    knee_amplitude_reduction_deg: float = 0.0,
    knee_phase_shift_fraction: float = 0.0,
) -> pd.DataFrame:
    """Create one smooth closed reference-neighbourhood path.

    Amplitude changes use ``sin(pi*global_phase)^2`` and are therefore zero at
    the cycle endpoints.  Knee phase changes use a monotone warp independently
    on the two symmetric branches and preserve both endpoint and peak values.
    No pointwise angle clipping is used.
    """

    required = {
        "cycle_phase",
        "segment_phase",
        "global_phase",
        "q_hip_reference_rad",
        "q_knee_reference_rad",
    }
    missing = required.difference(base_phase_path.columns)
    if missing:
        raise ValueError(f"closed phase path missing columns: {sorted(missing)}")
    output = base_phase_path.copy(deep=True)
    if set(output["cycle_phase"].astype(str)) != {"flexion", "extension"}:
        raise ValueError("closed phase path must contain flexion and extension.")

    bump = _endpoint_zero_bump(output["global_phase"].to_numpy(float))
    hip_delta = -np.deg2rad(float(hip_amplitude_reduction_deg)) * bump
    knee_delta = -np.deg2rad(float(knee_amplitude_reduction_deg)) * bump
    output["q_hip_reference_rad"] = (
        output["q_hip_reference_rad"].to_numpy(float) + hip_delta
    )

    base_knee = output["q_knee_reference_rad"].to_numpy(float)
    shifted_knee = base_knee.copy()
    if abs(float(knee_phase_shift_fraction)) > 0.0:
        for phase_name in ("flexion", "extension"):
            mask = output["cycle_phase"].astype(str).eq(phase_name).to_numpy()
            local_phase = output.loc[mask, "segment_phase"].to_numpy(float)
            knee_values = base_phase_path.loc[mask, "q_knee_reference_rad"].to_numpy(float)
            warped = _phase_warp(local_phase, float(knee_phase_shift_fraction))
            shifted_knee[mask] = PchipInterpolator(
                local_phase, knee_values, extrapolate=False
            )(warped)
    output["q_knee_reference_rad"] = shifted_knee + knee_delta
    output["theta_shank_reference_rad"] = (
        output["q_hip_reference_rad"] - output["q_knee_reference_rad"]
    )
    output["trajectory_id"] = trajectory_id
    output["hip_amplitude_reduction_deg"] = float(hip_amplitude_reduction_deg)
    output["knee_amplitude_reduction_deg"] = float(knee_amplitude_reduction_deg)
    output["knee_phase_shift_fraction"] = float(knee_phase_shift_fraction)
    output["perturbation_basis"] = "sin(pi*phase)^2_endpoint_zero"
    output["pointwise_angle_clipping_applied"] = False

    for phase_name in ("flexion", "extension"):
        segment = output.loc[output["cycle_phase"].eq(phase_name)]
        if not np.all(np.diff(segment["segment_phase"].to_numpy(float)) > 0.0):
            raise RuntimeError("perturbed path phase is not strictly increasing.")
    if not np.allclose(
        output["theta_shank_reference_rad"],
        output["q_hip_reference_rad"] - output["q_knee_reference_rad"],
        atol=1e-14,
    ):
        raise RuntimeError("theta_shank convention was not preserved.")
    return output


def build_local_phase_paths(reference_versions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the six fixed Stage-5C local-identification paths."""

    base = closed_execution_phase_path(reference_versions)
    specifications = {
        "reference_slow": {},
        "reference_nominal": {},
        "hip_amplitude_minus_3deg": {"hip_amplitude_reduction_deg": 3.0},
        "knee_amplitude_minus_3deg": {"knee_amplitude_reduction_deg": 3.0},
        "knee_phase_advance_3pct": {"knee_phase_shift_fraction": 0.03},
        "knee_phase_delay_3pct": {"knee_phase_shift_fraction": -0.03},
    }
    return {
        name: perturb_closed_phase_path(base, trajectory_id=name, **settings)
        for name, settings in specifications.items()
    }


def retime_local_phase_paths(
    phase_paths: Mapping[str, pd.DataFrame],
    *,
    samples_per_segment: int = 201,
) -> dict[str, pd.DataFrame]:
    """Retime all local paths using the existing minimum-jerk implementation."""

    trajectories: dict[str, pd.DataFrame] = {}
    for trajectory_id, phase_path in phase_paths.items():
        if trajectory_id not in LOCAL_TRAJECTORY_DURATION_S:
            raise ValueError(f"unknown local trajectory {trajectory_id!r}.")
        duration = LOCAL_TRAJECTORY_DURATION_S[trajectory_id]
        trajectory = retime_reference_path(
            phase_path,
            profile=trajectory_id,
            flexion_duration_s=duration,
            extension_duration_s=duration,
            samples_per_segment=samples_per_segment,
        )
        trajectory.insert(0, "trajectory_id", trajectory_id)
        trajectory["dataset_split"] = LOCAL_TRAJECTORY_SPLIT[trajectory_id]
        trajectory["identification_trajectory_type"] = (
            "reference_neighbourhood_identification_excitation"
        )
        trajectory["clinical_reference_status"] = "not_clinically_validated"
        trajectories[trajectory_id] = trajectory
    return trajectories


def _clean_observation_table(
    trajectory: pd.DataFrame,
    subject: DynamicVirtualSubject,
) -> pd.DataFrame:
    q_hip = trajectory["q_hip_rad"].to_numpy(float)
    q_knee = trajectory["q_knee_rad"].to_numpy(float)
    dynamics = inverse_dynamics(
        q_hip,
        q_knee,
        trajectory["dq_hip_rad_s"].to_numpy(float),
        trajectory["dq_knee_rad_s"].to_numpy(float),
        trajectory["ddq_hip_rad_s2"].to_numpy(float),
        trajectory["ddq_knee_rad_s2"].to_numpy(float),
        subject,
        L1,
    )
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        dynamics.tau_total_hip_nm,
        dynamics.tau_total_knee_nm,
        L1,
        L2,
    )
    fields = [
        "trajectory_id",
        "dataset_split",
        "time_s",
        "cycle_phase",
        "global_phase",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "theta_shank_rad",
        "x_pull_m",
        "z_pull_m",
    ]
    output = trajectory.loc[:, fields].copy(deep=True)
    output.insert(0, "subject_id", subject.subject_id)
    output["noise_scenario"] = "clean_reference_local"
    output["fx_observed_n"] = force.fx_robot_on_leg_n
    output["fz_observed_n"] = force.fz_robot_on_leg_n
    output["force_mapping_valid"] = force.force_mapping_valid
    output["jacobian_condition_number"] = force.jacobian_condition_number
    finite = np.isfinite(
        output[
            [
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
                "fx_observed_n",
                "fz_observed_n",
            ]
        ].to_numpy(float)
    ).all(axis=1)
    output["sample_valid"] = (
        finite
        & np.asarray(force.force_mapping_valid, dtype=bool)
        & trajectory["trajectory_sample_valid"].astype(bool).to_numpy()
    )
    output["invalid_reason"] = np.where(
        output["sample_valid"].to_numpy(bool), "", np.asarray(force.invalid_reason)
    )
    output["state_source"] = "estimated_or_observed_joint_state_no_truth_columns"
    output["model_angle_definition"] = "theta_shank = q_hip - q_knee"
    return output


def build_local_identification_dataset(
    trajectories: Mapping[str, pd.DataFrame],
    *,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> pd.DataFrame:
    """Generate clean force observations for four virtual subjects."""

    expected = set(LOCAL_TRAJECTORY_SPLIT)
    if set(trajectories) != expected:
        raise ValueError("local trajectories must exactly match the fixed Stage-5C set.")
    frames: list[pd.DataFrame] = []
    for subject_id in subject_ids:
        subject = get_dynamic_subject(subject_id)
        for trajectory_id in LOCAL_TRAJECTORY_SPLIT:
            table = _clean_observation_table(trajectories[trajectory_id], subject)
            if not table["trajectory_id"].eq(trajectory_id).all():
                raise RuntimeError("trajectory identity changed during dataset creation.")
            frames.append(table)
    dataset = pd.concat(frames, ignore_index=True)
    forbidden = [
        column
        for column in dataset.columns
        if column.startswith("true_")
        or column.startswith("tau_total")
        or column in PARAMETER_NAMES
    ]
    if forbidden:
        raise RuntimeError(f"estimator dataset leaked forbidden fields: {forbidden}")
    if not dataset["sample_valid"].fillna(False).astype(bool).all():
        invalid = dataset.loc[
            ~dataset["sample_valid"].fillna(False).astype(bool),
            ["trajectory_id", "subject_id", "invalid_reason"],
        ]
        summary = invalid.drop_duplicates().head(8).to_dict(orient="records")
        raise RuntimeError(
            "reference-neighbourhood excitation entered an invalid force/Jacobian "
            f"region: {summary}"
        )
    return dataset


def fit_local_subject_parameters(
    dataset: pd.DataFrame,
    *,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> tuple[pd.DataFrame, dict[str, ParameterEstimationResult], dict[str, dict[str, float]]]:
    """Fit each subject from train rows only and evaluate held-out splits."""

    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    rows: list[dict[str, object]] = []
    optimizer_results: dict[str, ParameterEstimationResult] = {}
    estimates: dict[str, dict[str, float]] = {}
    for subject_id in subject_ids:
        subject_rows = dataset.loc[dataset["subject_id"].eq(subject_id)]
        training = subject_rows.loc[subject_rows["dataset_split"].eq("train")]
        if not training["trajectory_id"].isin(
            [name for name, split in LOCAL_TRAJECTORY_SPLIT.items() if split == "train"]
        ).all():
            raise RuntimeError("validation/test trajectory entered local fitting.")
        estimator_columns = [*REQUIRED_OBSERVATION_COLUMNS, "force_mapping_valid"]
        optimizer_input = training.loc[:, estimator_columns].copy(deep=True)
        result = estimate_subject_parameters(
            optimizer_input,
            template,
            L1,
            L2,
            initial_guess=identification_initial_guess,
            bounds=(identification_lower_bounds, identification_upper_bounds),
        )
        optimizer_results[subject_id] = result
        estimates[subject_id] = dict(result.estimated_parameters)

        # Truth is loaded only after fitting and is used solely for this audit table.
        truth_subject = get_dynamic_subject(subject_id)
        scales = np.asarray(
            [
                truth_subject.mass_thigh_kg / baseline.mass_thigh_kg,
                truth_subject.mass_shank_kg / baseline.mass_shank_kg,
                truth_subject.inertia_thigh_kg_m2 / baseline.inertia_thigh_kg_m2,
                truth_subject.inertia_shank_kg_m2 / baseline.inertia_shank_kg_m2,
            ]
        )
        true_parameters = {
            "mass_scale": float(scales[0]),
            "k_hip_nm_per_rad": truth_subject.k_hip_nm_per_rad,
            "k_knee_nm_per_rad": truth_subject.k_knee_nm_per_rad,
            "b_hip_nm_s_per_rad": truth_subject.b_hip_nm_s_per_rad,
            "b_knee_nm_s_per_rad": truth_subject.b_knee_nm_s_per_rad,
        }
        for parameter in PARAMETER_NAMES:
            true_value = float(true_parameters[parameter])
            estimated_value = float(result.estimated_parameters[parameter])
            absolute_error = abs(estimated_value - true_value)
            row: dict[str, object] = {
                "subject_id": subject_id,
                "parameter": parameter,
                "true_value_evaluation_only": true_value,
                "estimated_value": estimated_value,
                "absolute_error": absolute_error,
                "relative_error_percent": (
                    100.0 * absolute_error / abs(true_value)
                    if true_value != 0.0
                    else np.nan
                ),
                "optimizer_success": result.optimizer_success,
                "training_sample_count": result.valid_training_samples,
                "fit_split": "train_only",
            }
            for split in ("train", "validation", "test"):
                metrics = compute_torque_metrics(
                    subject_rows.loc[subject_rows["dataset_split"].eq(split)],
                    template,
                    result.estimated_parameters,
                    L1,
                    L2,
                )
                row[f"{split}_torque_rmse_combined_nm"] = metrics[
                    "torque_rmse_combined_nm"
                ]
            rows.append(row)
    return pd.DataFrame(rows), optimizer_results, estimates


def _estimated_state_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    source_columns = (
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    table = dataframe.loc[:, source_columns].copy(deep=True)
    table.columns = ESTIMATED_DOMAIN_STATE_COLUMNS
    table["state_estimation_valid"] = dataframe["sample_valid"].astype(bool).to_numpy()
    return table


def fit_local_identification_domain(dataset: pd.DataFrame) -> StateDomainBounds:
    """Fit the six-state local domain from training observations only."""

    training = dataset.loc[dataset["dataset_split"].eq("train")]
    if training.empty:
        raise ValueError("local dataset has no training rows.")
    if not training["trajectory_id"].isin(
        [name for name, split in LOCAL_TRAJECTORY_SPLIT.items() if split == "train"]
    ).all():
        raise RuntimeError("non-training local trajectory entered domain fitting.")
    # The four subjects share exactly the same prescribed state trajectory.
    # Remove those repeats so the reported domain sample count represents
    # independent states rather than four copies of each state.
    training = training.drop_duplicates(
        subset=[
            "trajectory_id",
            "time_s",
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    )
    return fit_state_domain_bounds(_estimated_state_table(training))


def _coverage_missing_groups(states: pd.DataFrame, bounds: StateDomainBounds) -> str:
    values = states.loc[:, bounds.columns].to_numpy(float)
    lower = np.asarray(bounds.lower)
    upper = np.asarray(bounds.upper)
    outside = (values < lower) | (values > upper) | ~np.isfinite(values)
    groups: list[str] = []
    if outside[:, 0:2].any():
        groups.append("q")
    if outside[:, 2:4].any():
        groups.append("dq")
    if outside[:, 4:6].any():
        groups.append("ddq")
    return ";".join(groups)


def build_local_domain_coverage(
    trajectories: Mapping[str, pd.DataFrame],
    bounds: StateDomainBounds,
) -> pd.DataFrame:
    """Report train-built domain coverage for all six local trajectories."""

    rows: list[dict[str, object]] = []
    for trajectory_id, trajectory in trajectories.items():
        observed = trajectory.copy(deep=True)
        observed["sample_valid"] = trajectory["trajectory_sample_valid"].astype(bool)
        states = _estimated_state_table(observed)
        membership = classify_state_domain(states, bounds)
        in_count = int(membership.sum())
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_label": (
                    "phase_advance"
                    if trajectory_id == "knee_phase_advance_3pct"
                    else "phase_delay"
                    if trajectory_id == "knee_phase_delay_3pct"
                    else trajectory_id
                ),
                "dataset_split": LOCAL_TRAJECTORY_SPLIT[trajectory_id],
                "in_domain_sample_count": in_count,
                "out_of_domain_sample_count": int(len(trajectory) - in_count),
                "in_domain_percent": 100.0 * in_count / len(trajectory),
                "missing_state_variables": _coverage_missing_groups(states, bounds),
                "domain_model": LOCAL_DOMAIN_MODEL,
                "domain_fitted_from_split": "train",
                "domain_training_sample_count": bounds.valid_training_samples,
            }
        )
    return pd.DataFrame(rows)


def run_local_identification(
    reference_versions: pd.DataFrame,
    *,
    samples_per_segment: int = 201,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> LocalIdentificationResult:
    """Build local excitation, fit five parameters and rebuild the local domain."""

    closed = reference_versions.loc[
        reference_versions["reference_version"].eq(CLOSED_REFERENCE)
    ]
    if closed.empty or "formal_execution_allowed" not in closed:
        raise ValueError("local identification requires a ROM-gated closed reference.")
    if not closed["formal_execution_allowed"].fillna(False).astype(bool).all():
        raise PermissionError(
            "local identification is blocked until knee ROM is explicitly approved."
        )
    phase_paths = build_local_phase_paths(reference_versions)
    trajectories = retime_local_phase_paths(
        phase_paths, samples_per_segment=samples_per_segment
    )
    dataset = build_local_identification_dataset(trajectories, subject_ids=subject_ids)
    parameter_table, optimizer_results, parameters = fit_local_subject_parameters(
        dataset, subject_ids=subject_ids
    )
    bounds = fit_local_identification_domain(dataset)
    coverage = build_local_domain_coverage(trajectories, bounds)
    return LocalIdentificationResult(
        dataset=dataset,
        parameter_estimates=parameter_table,
        optimizer_results=optimizer_results,
        estimated_parameters_by_subject=parameters,
        domain_bounds=bounds,
        domain_coverage=coverage,
    )


__all__ = [
    "LOCAL_DOMAIN_MODEL",
    "LOCAL_TRAJECTORY_DURATION_S",
    "LOCAL_TRAJECTORY_SPLIT",
    "LocalIdentificationResult",
    "build_local_domain_coverage",
    "build_local_identification_dataset",
    "build_local_phase_paths",
    "fit_local_identification_domain",
    "fit_local_subject_parameters",
    "perturb_closed_phase_path",
    "retime_local_phase_paths",
    "run_local_identification",
]
