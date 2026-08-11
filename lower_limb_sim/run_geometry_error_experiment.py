"""Run Stage 4.5D virtual geometry/kinematic-observation experiments.

The generator and evaluation layers may retain simulator truth, but the
five-parameter estimator receives a hard-whitelisted table containing only
reconstructed motion, observed endpoint force, and validity flags.  Every
non-oracle derivative is reconstructed from the measured angle stream.

This module is offline software validation only.  It does not import or call
robot control, acquisition, safety, hardware, or SDK code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    dynamic_sampling_frequency_hz,
    geometry_default_offline_derivative_method,
    geometry_error_data_dir,
    geometry_error_model_version,
    geometry_error_random_seed,
    geometry_ik_domain_clip_tolerance,
    geometry_max_derivative_gap_s,
    geometry_max_joint_jump_deg,
    geometry_noise_seed_count,
    geometry_noisy_offline_derivative_method,
    geometry_savgol_polynomial_order,
    geometry_savgol_window_duration_s,
    hip_range_deg,
    identification_initial_guess,
    identification_loss,
    identification_lower_bounds,
    identification_upper_bounds,
)
from .derivative_estimation import DerivativeEstimationConfig
from .dynamic_subject import DYNAMIC_SUBJECTS, get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .geometry_calibration import (
    AssumedGeometry,
    TrueGeometry,
    calibration_error_from_geometries,
    true_geometry_from_dynamic_subject,
)
from .geometry_error_metrics import (
    attach_and_evaluate_domain_membership,
    build_geometry_sensitivity_ranking,
    combined_nrmse_percent,
    compute_kinematic_reconstruction_metrics,
    compute_observation_metrics,
    compute_parameter_error_table,
    q0_stiffness_correlation_analysis,
)
from .geometry_error_scenarios import (
    BASE_GEOMETRY_ERROR_SCENARIOS,
    GEOMETRY_ERROR_SCENARIOS,
    GEOMETRY_SENSITIVITY_VARIANTS,
    INDEPENDENT_JOINT_MEASUREMENT,
    OBSERVATION_MODES,
    ORACLE_TRUE_JOINT_STATE,
    TCP_INVERSE_KINEMATICS,
    GeometryErrorScenario,
    get_geometry_error_scenario,
)
from .jacobian import jacobian_diagnostics, leg_jacobian
from .kinematic_observation import build_kinematic_observation
from .kinematics import forward_kinematics
from .parameter_estimator import (
    PARAMETER_NAMES,
    BaselineSubjectTemplate,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    estimate_subject_parameters,
    predict_joint_torque,
)
from .run_model_mismatch_experiment import (
    ALL_EVALUATION_SPLITS,
    _git_commit,
    _parameters_from_subject,
    _upsert_csv,
    _write_json,
    build_model_mismatch_dataset,
)


GEOMETRY_ESTIMATOR_INPUT_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
    "fx_observed_n",
    "fz_observed_n",
    "sample_valid",
    "force_mapping_valid",
    "wrench_is_stale",
    "invalid_reason",
)

TRUE_STATE_COLUMNS = (
    "q_hip_true_rad",
    "q_knee_true_rad",
    "dq_hip_true_rad_s",
    "dq_knee_true_rad_s",
    "ddq_hip_true_rad_s2",
    "ddq_knee_true_rad_s2",
)

ESTIMATED_STATE_COLUMNS = (
    "q_hip_est_rad",
    "q_knee_est_rad",
    "dq_hip_est_rad_s",
    "dq_knee_est_rad_s",
    "ddq_hip_est_rad_s2",
    "ddq_knee_est_rad_s2",
)

OBSERVATION_CONTEXT_COLUMNS = (
    "subject_id",
    "trajectory_id",
    "trajectory_family",
    "trajectory_name",
    "speed_profile",
    "dataset_split",
    "phase",
    "time_s",
    "trajectory_sample_index",
    "fx_observed_n",
    "fz_observed_n",
    "force_mapping_valid",
    "wrench_is_stale",
)

SPLIT_FILENAMES = {
    "train": "training_data.csv",
    "validation": "validation_data.csv",
    "interpolation_test": "interpolation_test_data.csv",
    "boundary_test": "boundary_test_data.csv",
    "outside_domain_test": "outside_domain_test_data.csv",
}

CORE_FOUR_SUBJECT_SCENARIOS = (
    "matched_geometry",
    "L2_error_2cm",
    "hip_center_combined_error_2cm",
    "q0_error_5deg",
    "combined_geometry_mild",
    "combined_geometry_strong",
)


def project_geometry_estimator_inputs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the complete and exclusive input visible to the estimator."""

    missing = set(GEOMETRY_ESTIMATOR_INPUT_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(f"geometry observation is missing: {sorted(missing)}")
    projected = dataframe.loc[:, GEOMETRY_ESTIMATOR_INPUT_COLUMNS].copy()
    projected.attrs.clear()
    forbidden = [
        column
        for column in projected.columns
        if "true" in column.lower()
        or "scenario" in column.lower()
        or "subject" in column.lower()
        or "split" in column.lower()
        or "geometry" in column.lower()
    ]
    if forbidden or tuple(projected.columns) != GEOMETRY_ESTIMATOR_INPUT_COLUMNS:
        raise RuntimeError(f"estimator leakage boundary failed: {forbidden}")
    return projected


def fit_five_parameter_observation(
    training_dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    assumed_geometry: AssumedGeometry,
    *,
    initial_guess: Mapping[str, float] | Sequence[float] = (
        identification_initial_guess
    ),
    bounds: tuple[
        Mapping[str, float] | Sequence[float],
        Mapping[str, float] | Sequence[float],
    ] = (identification_lower_bounds, identification_upper_bounds),
    loss: str = identification_loss,
) -> ParameterEstimationResult:
    """Fit on one training table; no test/truth/scenario argument exists."""

    estimator_input = project_geometry_estimator_inputs(training_dataframe)
    return estimate_subject_parameters(
        estimator_input,
        baseline_subject_template,
        assumed_geometry.L1_assumed_m,
        assumed_geometry.L2_assumed_m,
        initial_guess=initial_guess,
        bounds=bounds,
        loss=loss,
    )


def _scenario_with_seed(
    scenario_name: str,
    random_seed: int | None,
) -> GeometryErrorScenario:
    scenario = get_geometry_error_scenario(scenario_name)
    return scenario if random_seed is None else scenario.with_random_seed(random_seed)


def _true_and_observed_trajectories(
    subject_id: str,
    scenario: GeometryErrorScenario,
    true_geometry: TrueGeometry,
    *,
    sampling_frequency_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate truth once, then expose only sensor-like observations."""

    source = build_model_mismatch_dataset(
        subject_id,
        "matched_linear",
        sampling_frequency_hz=sampling_frequency_hz,
    ).reset_index(drop=True)
    q_hip = source["q_hip_rad"].to_numpy(dtype=float)
    q_knee = source["q_knee_rad"].to_numpy(dtype=float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        true_geometry.L1_true_m,
        true_geometry.L2_true_m,
    )
    x_knee = np.asarray(x_knee, dtype=float) + true_geometry.hip_center_x_true_m
    z_knee = np.asarray(z_knee, dtype=float) + true_geometry.hip_center_z_true_m
    x_pull = np.asarray(x_pull, dtype=float) + true_geometry.hip_center_x_true_m
    z_pull = np.asarray(z_pull, dtype=float) + true_geometry.hip_center_z_true_m

    identity = [
        column
        for column in OBSERVATION_CONTEXT_COLUMNS
        if column in source.columns
    ]
    truth = source.loc[:, identity].copy()
    truth["scenario_name"] = scenario.scenario_name
    for destination, source_column in zip(
        TRUE_STATE_COLUMNS,
        (
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ),
    ):
        truth[destination] = source[source_column].to_numpy(dtype=float)
    truth["x_knee_true_m"] = x_knee
    truth["z_knee_true_m"] = z_knee
    truth["x_pull_true_m"] = x_pull
    truth["z_pull_true_m"] = z_pull
    truth["fx_true_n"] = source["fx_observed_n"].to_numpy(dtype=float)
    truth["fz_true_n"] = source["fz_observed_n"].to_numpy(dtype=float)

    x_measured, z_measured = scenario.apply_tcp_position_noise(
        x_pull,
        z_pull,
        random_seed=scenario.random_seed,
    )
    q_hip_measured, q_knee_measured = scenario.apply_independent_angle_noise(
        q_hip,
        q_knee,
        random_seed=scenario.random_seed + 1,
    )
    observed = source.loc[:, identity].copy()
    observed["scenario_name"] = scenario.scenario_name
    observed["x_pull_measured_m"] = np.asarray(x_measured, dtype=float)
    observed["z_pull_measured_m"] = np.asarray(z_measured, dtype=float)
    observed["q_hip_measured_rad"] = np.asarray(q_hip_measured, dtype=float)
    observed["q_knee_measured_rad"] = np.asarray(q_knee_measured, dtype=float)
    observed["measurement_valid"] = np.isfinite(
        observed[
            [
                "x_pull_measured_m",
                "z_pull_measured_m",
                "q_hip_measured_rad",
                "q_knee_measured_rad",
                "fx_observed_n",
                "fz_observed_n",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    return truth, observed


def _savgol_window_length(sampling_frequency_hz: float) -> int:
    requested = int(round(geometry_savgol_window_duration_s * sampling_frequency_hz))
    requested = max(requested, geometry_savgol_polynomial_order + 2, 5)
    if requested % 2 == 0:
        requested += 1
    return requested


def _derivative_config(sampling_frequency_hz: float) -> DerivativeEstimationConfig:
    return DerivativeEstimationConfig(
        savgol_window_length=_savgol_window_length(sampling_frequency_hz),
        savgol_polynomial_order=geometry_savgol_polynomial_order,
        maximum_time_gap_s=geometry_max_derivative_gap_s,
    )


def _derivative_method_for_mode(
    scenario: GeometryErrorScenario,
    observation_mode: str,
) -> str:
    noisy = (
        observation_mode == TCP_INVERSE_KINEMATICS
        and scenario.tcp_position_noise_std_m > 0.0
    ) or (
        observation_mode == INDEPENDENT_JOINT_MEASUREMENT
        and scenario.independent_angle_noise_std_rad > 0.0
    )
    return (
        geometry_noisy_offline_derivative_method
        if noisy
        else geometry_default_offline_derivative_method
    )


def _observation_input(
    truth: pd.DataFrame,
    observed: pd.DataFrame,
    observation_mode: str,
) -> pd.DataFrame:
    context = [
        column for column in OBSERVATION_CONTEXT_COLUMNS if column in observed
    ]
    if observation_mode == ORACLE_TRUE_JOINT_STATE:
        output = observed.loc[:, context].copy()
        for column in TRUE_STATE_COLUMNS:
            output[column] = truth[column].to_numpy(dtype=float)
        return output
    if observation_mode == TCP_INVERSE_KINEMATICS:
        return observed.loc[
            :,
            [*context, "x_pull_measured_m", "z_pull_measured_m"],
        ].copy()
    if observation_mode == INDEPENDENT_JOINT_MEASUREMENT:
        return observed.loc[
            :,
            [*context, "q_hip_measured_rad", "q_knee_measured_rad"],
        ].copy()
    raise ValueError(f"unknown observation mode {observation_mode!r}")


def _matvec_transpose(jacobian: np.ndarray, force: np.ndarray) -> np.ndarray:
    return np.matmul(
        np.swapaxes(jacobian, -1, -2),
        force[..., np.newaxis],
    )[..., 0]


def _reconstruct_one_mode(
    truth: pd.DataFrame,
    observed: pd.DataFrame,
    assumed_geometry: AssumedGeometry,
    scenario: GeometryErrorScenario,
    observation_mode: str,
    *,
    sampling_frequency_hz: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    mode_input = _observation_input(truth, observed, observation_mode)
    derivative_method = _derivative_method_for_mode(scenario, observation_mode)
    kwargs: dict[str, object] = {
        "derivative_method": derivative_method,
        "derivative_config": _derivative_config(sampling_frequency_hz),
        "derivative_group_columns": ("trajectory_id",),
    }
    if observation_mode == TCP_INVERSE_KINEMATICS:
        kwargs["assumed_geometry"] = assumed_geometry
        kwargs["reconstruction_options"] = {
            "acos_domain_tolerance": geometry_ik_domain_clip_tolerance,
            "maximum_joint_jump_rad": np.deg2rad(geometry_max_joint_jump_deg),
        }
    result = build_kinematic_observation(
        mode_input,
        observation_mode,
        **kwargs,
    )
    frame = result.dataframe.reset_index(drop=True).copy()
    frame["scenario_name"] = scenario.scenario_name
    frame["observation_mode"] = observation_mode
    frame["derivative_method"] = derivative_method
    canonical_to_estimated = {
        "q_hip_rad": "q_hip_est_rad",
        "q_knee_rad": "q_knee_est_rad",
        "dq_hip_rad_s": "dq_hip_est_rad_s",
        "dq_knee_rad_s": "dq_knee_est_rad_s",
        "ddq_hip_rad_s2": "ddq_hip_est_rad_s2",
        "ddq_knee_rad_s2": "ddq_knee_est_rad_s2",
    }
    for canonical, estimated in canonical_to_estimated.items():
        if estimated not in frame:
            frame[estimated] = frame[canonical].to_numpy(dtype=float)
    frame["state_estimation_valid"] = frame["observation_valid"].astype(bool)
    if "ik_valid" not in frame:
        frame["ik_valid"] = frame["observation_valid"].astype(bool)
    if "joint_continuity_valid" not in frame:
        frame["joint_continuity_valid"] = frame["observation_valid"].astype(bool)
    if "ik_reason" not in frame:
        frame["ik_reason"] = ""
    for column in TRUE_STATE_COLUMNS:
        frame[column] = truth[column].to_numpy(dtype=float)
    for column in (
        "x_pull_true_m",
        "z_pull_true_m",
        "fx_true_n",
        "fz_true_n",
    ):
        frame[column] = truth[column].to_numpy(dtype=float)
    for column in (
        "x_pull_measured_m",
        "z_pull_measured_m",
        "q_hip_measured_rad",
        "q_knee_measured_rad",
    ):
        frame[column] = observed[column].to_numpy(dtype=float)

    q_hip_true = frame["q_hip_true_rad"].to_numpy(dtype=float)
    q_knee_true = frame["q_knee_true_rad"].to_numpy(dtype=float)
    q_hip_est = frame["q_hip_est_rad"].to_numpy(dtype=float)
    q_knee_est = frame["q_knee_est_rad"].to_numpy(dtype=float)
    # Truth lengths are attached only to the generator/evaluation table attrs;
    # no estimator projection preserves attrs.
    L1_true_m = float(truth.attrs["L1_true_m"])
    L2_true_m = float(truth.attrs["L2_true_m"])
    true_jacobian = leg_jacobian(q_hip_true, q_knee_true, L1_true_m, L2_true_m)
    finite_est = np.isfinite(q_hip_est) & np.isfinite(q_knee_est)
    assumed_jacobian = leg_jacobian(
        np.where(finite_est, q_hip_est, 0.0),
        np.where(finite_est, q_knee_est, 0.0),
        assumed_geometry.L1_assumed_m,
        assumed_geometry.L2_assumed_m,
    )
    assumed_jacobian[~finite_est] = np.nan
    force = frame[["fx_observed_n", "fz_observed_n"]].to_numpy(dtype=float)
    tau_true = _matvec_transpose(true_jacobian, force)
    tau_est = _matvec_transpose(assumed_jacobian, force)
    frame["tau_measured_true_hip_nm"] = tau_true[:, 0]
    frame["tau_measured_true_knee_nm"] = tau_true[:, 1]
    frame["tau_measured_est_hip_nm"] = tau_est[:, 0]
    frame["tau_measured_est_knee_nm"] = tau_est[:, 1]
    frame["jacobian_frobenius_error"] = np.linalg.norm(
        assumed_jacobian - true_jacobian,
        axis=(1, 2),
    )
    true_diagnostics = jacobian_diagnostics(
        q_hip_true, q_knee_true, L1_true_m, L2_true_m
    )
    assumed_diagnostics = jacobian_diagnostics(
        np.where(finite_est, q_hip_est, np.nan),
        np.where(finite_est, q_knee_est, np.nan),
        assumed_geometry.L1_assumed_m,
        assumed_geometry.L2_assumed_m,
    )
    true_condition = np.asarray(true_diagnostics.condition_number, dtype=float)
    assumed_condition = np.asarray(
        assumed_diagnostics.condition_number, dtype=float
    )
    condition_error = np.abs(assumed_condition - true_condition)
    condition_error[~np.isfinite(condition_error)] = np.nan
    frame["jacobian_condition_true"] = true_condition
    frame["jacobian_condition_assumed"] = assumed_condition
    frame["jacobian_condition_error"] = condition_error

    x_knee_assumed = np.full(len(frame), np.nan)
    z_knee_assumed = np.full(len(frame), np.nan)
    x_pull_assumed = np.full(len(frame), np.nan)
    z_pull_assumed = np.full(len(frame), np.nan)
    if finite_est.any():
        selected = np.flatnonzero(finite_est)
        fk = forward_kinematics(
            q_hip_est[selected],
            q_knee_est[selected],
            assumed_geometry.L1_assumed_m,
            assumed_geometry.L2_assumed_m,
        )
        x_knee_assumed[selected] = (
            np.asarray(fk[0], dtype=float)
            + assumed_geometry.hip_center_x_assumed_m
        )
        z_knee_assumed[selected] = (
            np.asarray(fk[1], dtype=float)
            + assumed_geometry.hip_center_z_assumed_m
        )
        x_pull_assumed[selected] = (
            np.asarray(fk[2], dtype=float)
            + assumed_geometry.hip_center_x_assumed_m
        )
        z_pull_assumed[selected] = (
            np.asarray(fk[3], dtype=float)
            + assumed_geometry.hip_center_z_assumed_m
        )
    frame["x_knee_assumed_m"] = x_knee_assumed
    frame["z_knee_assumed_m"] = z_knee_assumed
    frame["x_pull_assumed_reconstructed_m"] = x_pull_assumed
    frame["z_pull_assumed_reconstructed_m"] = z_pull_assumed

    force_valid = frame["force_mapping_valid"].fillna(False).astype(bool).to_numpy()
    finite_force = np.isfinite(force).all(axis=1)
    finite_state = np.isfinite(
        frame.loc[:, ESTIMATED_STATE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)
    sample_valid = (
        frame["state_estimation_valid"].astype(bool).to_numpy()
        & force_valid
        & finite_force
        & finite_state
        & np.isfinite(tau_est).all(axis=1)
    )
    frame["sample_valid"] = sample_valid
    frame["wrench_is_stale"] = False
    frame["invalid_reason"] = np.where(
        sample_valid,
        "",
        frame["observation_reason"].fillna("").astype(str),
    )
    empty_reason = (~sample_valid) & frame["invalid_reason"].eq("").to_numpy()
    frame.loc[empty_reason, "invalid_reason"] = "invalid_force_or_state"
    return frame, result.metadata


def build_geometry_observation_dataset(
    subject_id: str,
    scenario_name: str,
    *,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int | None = None,
    observation_modes: Sequence[str] = OBSERVATION_MODES,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    TrueGeometry,
    AssumedGeometry,
    dict[str, dict[str, object]],
]:
    """Build truth, sensor observations, and reconstructed states."""

    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    modes = tuple(observation_modes)
    if not modes or set(modes).difference(OBSERVATION_MODES):
        raise ValueError(f"observation_modes must be a subset of {OBSERVATION_MODES}")
    subject = get_dynamic_subject(subject_id)
    scenario = _scenario_with_seed(scenario_name, random_seed)
    true_geometry = true_geometry_from_dynamic_subject(subject)
    assumed_geometry = scenario.create_assumed_geometry(true_geometry)
    truth, observed = _true_and_observed_trajectories(
        subject_id,
        scenario,
        true_geometry,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    truth.attrs.update(true_geometry.as_metadata_dict())
    frames = []
    metadata: dict[str, dict[str, object]] = {}
    for mode in modes:
        frame, mode_metadata = _reconstruct_one_mode(
            truth,
            observed,
            assumed_geometry,
            scenario,
            mode,
            sampling_frequency_hz=sampling_frequency_hz,
        )
        frames.append(frame)
        metadata[mode] = mode_metadata
    reconstructed = pd.concat(frames, ignore_index=True, sort=False)
    expected_splits = set(ALL_EVALUATION_SPLITS)
    if set(reconstructed["dataset_split"].astype(str)) != expected_splits:
        raise RuntimeError("geometry observation split coverage is incomplete.")
    return (
        truth,
        observed,
        reconstructed,
        true_geometry,
        assumed_geometry,
        metadata,
    )


def _assumed_template(assumed_geometry: AssumedGeometry) -> BaselineSubjectTemplate:
    baseline = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    return replace(
        baseline,
        q0_hip_rad=assumed_geometry.q0_hip_assumed_rad,
        q0_knee_rad=assumed_geometry.q0_knee_assumed_rad,
    )


def _fit_all_modes(
    reconstructed: pd.DataFrame,
    assumed_geometry: AssumedGeometry,
    *,
    loss: str,
) -> tuple[dict[str, ParameterEstimationResult], dict[str, pd.DataFrame]]:
    template = _assumed_template(assumed_geometry)
    estimates: dict[str, ParameterEstimationResult] = {}
    inputs: dict[str, pd.DataFrame] = {}
    for mode, group in reconstructed.groupby("observation_mode", sort=False):
        training = group.loc[group["dataset_split"].eq("train")]
        projected = project_geometry_estimator_inputs(training)
        inputs[str(mode)] = projected
        estimates[str(mode)] = fit_five_parameter_observation(
            projected,
            template,
            assumed_geometry,
            loss=loss,
        )
    return estimates, inputs


def _prediction_metrics_for_mode(
    dataframe: pd.DataFrame,
    assumed_geometry: AssumedGeometry,
    template: BaselineSubjectTemplate,
    generic_parameters: Mapping[str, float],
    identified_parameters: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = dataframe.copy()
    q_hip = output["q_hip_rad"].to_numpy(dtype=float)
    q_knee = output["q_knee_rad"].to_numpy(dtype=float)
    parameter_sets = {
        "generic": generic_parameters,
        "identified": identified_parameters,
    }
    state_valid = (
        output["sample_valid"].astype(bool).to_numpy()
        & np.isfinite(
            output[
                [
                    "q_hip_rad",
                    "q_knee_rad",
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1)
    )
    for model, parameters in parameter_sets.items():
        predicted_hip = np.full(len(output), np.nan)
        predicted_knee = np.full(len(output), np.nan)
        if state_valid.any():
            selected = output.loc[state_valid]
            predicted_valid = predict_joint_torque(
                selected,
                template,
                parameters,
                assumed_geometry.L1_assumed_m,
            )
            predicted_hip[state_valid] = np.asarray(predicted_valid[0], dtype=float)
            predicted_knee[state_valid] = np.asarray(predicted_valid[1], dtype=float)
        output[f"tau_{model}_hip_nm"] = predicted_hip
        output[f"tau_{model}_knee_nm"] = predicted_knee
        fx_predicted = np.full(len(output), np.nan)
        fz_predicted = np.full(len(output), np.nan)
        mapping_valid = np.zeros(len(output), dtype=bool)
        if state_valid.any():
            force = endpoint_force_from_joint_torque(
                q_hip[state_valid],
                q_knee[state_valid],
                predicted_hip[state_valid],
                predicted_knee[state_valid],
                assumed_geometry.L1_assumed_m,
                assumed_geometry.L2_assumed_m,
            )
            fx_predicted[state_valid] = np.asarray(
                force.fx_robot_on_leg_n, dtype=float
            )
            fz_predicted[state_valid] = np.asarray(
                force.fz_robot_on_leg_n, dtype=float
            )
            mapping_valid[state_valid] = np.asarray(
                force.force_mapping_valid, dtype=bool
            )
        output[f"fx_{model}_n"] = fx_predicted
        output[f"fz_{model}_n"] = fz_predicted
        output[f"{model}_prediction_valid"] = (
            state_valid & mapping_valid
        )

    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for split, split_frame in output.groupby("dataset_split", sort=False):
        split_metrics: dict[str, dict[str, float]] = {}
        for model in ("generic", "identified"):
            true_tau = split_frame[
                ["tau_measured_true_hip_nm", "tau_measured_true_knee_nm"]
            ].to_numpy(dtype=float)
            predicted_tau = split_frame[
                [f"tau_{model}_hip_nm", f"tau_{model}_knee_nm"]
            ].to_numpy(dtype=float)
            valid = (
                split_frame["sample_valid"].astype(bool).to_numpy()
                & np.isfinite(true_tau).all(axis=1)
                & np.isfinite(predicted_tau).all(axis=1)
            )
            if not valid.any():
                torque_rmse = torque_nrmse = torque_peak = np.nan
            else:
                residual = predicted_tau[valid] - true_tau[valid]
                torque_rmse = float(np.sqrt(np.mean(residual**2)))
                torque_nrmse = combined_nrmse_percent(
                    true_tau[valid, 0],
                    true_tau[valid, 1],
                    predicted_tau[valid, 0],
                    predicted_tau[valid, 1],
                )
                denominator = max(float(np.max(np.abs(true_tau[valid]))), 1e-12)
                torque_peak = float(100.0 * np.max(np.abs(residual)) / denominator)

            true_force = split_frame[["fx_true_n", "fz_true_n"]].to_numpy(float)
            predicted_force = split_frame[
                [f"fx_{model}_n", f"fz_{model}_n"]
            ].to_numpy(float)
            force_valid = (
                split_frame[f"{model}_prediction_valid"].astype(bool).to_numpy()
                & np.isfinite(true_force).all(axis=1)
                & np.isfinite(predicted_force).all(axis=1)
            )
            if not force_valid.any():
                force_rmse = force_peak = np.nan
            else:
                force_residual = predicted_force[force_valid] - true_force[force_valid]
                force_rmse = float(np.sqrt(np.mean(force_residual**2)))
                force_denominator = max(
                    float(np.max(np.linalg.norm(true_force[force_valid], axis=1))),
                    1e-12,
                )
                force_peak = float(
                    100.0
                    * np.max(np.linalg.norm(force_residual, axis=1))
                    / force_denominator
                )
            metrics = {
                "combined_torque_rmse_nm": torque_rmse,
                "combined_torque_nrmse_percent": torque_nrmse,
                "peak_torque_error_percent": torque_peak,
                "endpoint_force_rmse_n": force_rmse,
                "endpoint_force_peak_error_percent": force_peak,
            }
            split_metrics[model] = metrics
            rows.append(
                {
                    "subject_id": str(split_frame["subject_id"].iloc[0]),
                    "scenario_name": str(split_frame["scenario_name"].iloc[0]),
                    "observation_mode": str(
                        split_frame["observation_mode"].iloc[0]
                    ),
                    "dataset_split": str(split),
                    "prediction_model": model,
                    "valid_torque_samples": int(valid.sum()),
                    "valid_force_samples": int(force_valid.sum()),
                    **metrics,
                }
            )
        generic_rmse = split_metrics["generic"]["combined_torque_rmse_nm"]
        identified_rmse = split_metrics["identified"]["combined_torque_rmse_nm"]
        improvement = (
            100.0 * (generic_rmse - identified_rmse) / generic_rmse
            if np.isfinite(generic_rmse) and generic_rmse > 1e-6
            else np.nan
        )
        comparisons.append(
            {
                "subject_id": str(split_frame["subject_id"].iloc[0]),
                "scenario_name": str(split_frame["scenario_name"].iloc[0]),
                "observation_mode": str(
                    split_frame["observation_mode"].iloc[0]
                ),
                "dataset_split": str(split),
                "generic_combined_torque_rmse_nm": generic_rmse,
                "identified_combined_torque_rmse_nm": identified_rmse,
                "generic_vs_identified_improvement_percent": improvement,
            }
        )
    return output, pd.DataFrame(rows).merge(
        pd.DataFrame(comparisons)[
            [
                "observation_mode",
                "dataset_split",
                "generic_vs_identified_improvement_percent",
            ]
        ],
        on=("observation_mode", "dataset_split"),
        how="left",
    ), pd.DataFrame(comparisons)


def _predict_all_modes(
    reconstructed: pd.DataFrame,
    assumed_geometry: AssumedGeometry,
    estimates: Mapping[str, ParameterEstimationResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_subject = get_dynamic_subject("baseline")
    generic_parameters = _parameters_from_subject(
        baseline_subject, baseline_subject
    )
    template = _assumed_template(assumed_geometry)
    predictions = []
    metrics = []
    comparisons = []
    for mode, group in reconstructed.groupby("observation_mode", sort=False):
        predicted, mode_metrics, mode_comparison = _prediction_metrics_for_mode(
            group,
            assumed_geometry,
            template,
            generic_parameters,
            estimates[str(mode)].estimated_parameters,
        )
        predictions.append(predicted)
        metrics.append(mode_metrics)
        comparisons.append(mode_comparison)
    return (
        pd.concat(predictions, ignore_index=True, sort=False),
        pd.concat(metrics, ignore_index=True, sort=False),
        pd.concat(comparisons, ignore_index=True, sort=False),
    )


def _observation_mode_comparison(
    kinematic_metrics: pd.DataFrame,
    observation_metrics: pd.DataFrame,
    parameter_errors: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
) -> pd.DataFrame:
    kinematics = kinematic_metrics.loc[
        kinematic_metrics["dataset_split"].eq("interpolation_test")
    ].copy()
    kinematics["joint_angle_rmse_deg"] = np.sqrt(
        (
            kinematics["q_hip_rmse_deg"] ** 2
            + kinematics["q_knee_rmse_deg"] ** 2
        )
        / 2.0
    )
    observation = observation_metrics.loc[
        observation_metrics["dataset_split"].eq("interpolation_test")
    ]
    prediction = prediction_metrics.loc[
        prediction_metrics["dataset_split"].eq("interpolation_test")
        & prediction_metrics["prediction_model"].eq("identified")
    ]
    columns = ["subject_id", "scenario_name", "observation_mode"]
    result = kinematics[
        [
            *columns,
            "joint_angle_rmse_deg",
            "q_hip_rmse_deg",
            "q_knee_rmse_deg",
            "invalid_ik_rate",
        ]
    ].merge(
        observation[
            [*columns, "tau_observation_combined_rmse_nm"]
        ],
        on=columns,
        how="left",
    ).merge(
        parameter_errors[
            [*columns, "maximum_parameter_error_percent"]
        ],
        on=columns,
        how="left",
    ).merge(
        prediction[
            [
                *columns,
                "combined_torque_rmse_nm",
                "combined_torque_nrmse_percent",
                "endpoint_force_rmse_n",
                "generic_vs_identified_improvement_percent",
            ]
        ],
        on=columns,
        how="left",
    )
    tcp = result.loc[
        result["observation_mode"].eq(TCP_INVERSE_KINEMATICS),
        "combined_torque_nrmse_percent",
    ]
    independent = result.loc[
        result["observation_mode"].eq(INDEPENDENT_JOINT_MEASUREMENT),
        "combined_torque_nrmse_percent",
    ]
    relative_improvement = np.nan
    if len(tcp) and len(independent) and float(tcp.iloc[0]) > 0.0:
        relative_improvement = float(
            100.0 * (float(tcp.iloc[0]) - float(independent.iloc[0])) / float(tcp.iloc[0])
        )
    result["independent_vs_tcp_nrmse_improvement_percent"] = relative_improvement
    result["oracle_upper_bound_only"] = result["observation_mode"].eq(
        ORACLE_TRUE_JOINT_STATE
    )
    return result


def _error_source(scenario: GeometryErrorScenario) -> str:
    name = scenario.scenario_name
    if name.startswith("L1_"):
        return "L1_error"
    if name.startswith("L2_"):
        return "L2_error"
    if name.startswith("hip_center_x"):
        return "hip_center_x_error"
    if name.startswith("hip_center_z"):
        return "hip_center_z_error"
    if name.startswith("hip_center_combined"):
        return "hip_center_combined_error"
    if name.startswith("q0_"):
        return "q0_error"
    if name.startswith("tcp_position"):
        return "tcp_position_noise"
    if name.startswith("independent_angle"):
        return "independent_angle_noise"
    if name.startswith("combined"):
        return "combined_geometry_and_noise"
    return "matched_geometry"


def _sensitivity_rows(
    scenario: GeometryErrorScenario,
    random_seed: int,
    kinematic_metrics: pd.DataFrame,
    observation_metrics: pd.DataFrame,
    parameter_errors: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
) -> pd.DataFrame:
    comparison = _observation_mode_comparison(
        kinematic_metrics,
        observation_metrics,
        parameter_errors,
        prediction_metrics,
    ).copy()
    comparison["error_source"] = _error_source(scenario)
    comparison["random_seed"] = int(random_seed)
    comparison["interpolation_combined_nrmse_percent"] = comparison[
        "combined_torque_nrmse_percent"
    ]
    return comparison[
        [
            "subject_id",
            "scenario_name",
            "observation_mode",
            "error_source",
            "random_seed",
            "joint_angle_rmse_deg",
            "tau_observation_combined_rmse_nm",
            "interpolation_combined_nrmse_percent",
            "maximum_parameter_error_percent",
            "invalid_ik_rate",
        ]
    ]


def _estimator_ready_export(reconstructed: pd.DataFrame) -> pd.DataFrame:
    safe_context = [
        column
        for column in (
            "observation_mode",
            "trajectory_id",
            "trajectory_family",
            "trajectory_name",
            "speed_profile",
            "phase",
            "time_s",
            "dataset_split",
            *GEOMETRY_ESTIMATOR_INPUT_COLUMNS,
        )
        if column in reconstructed.columns
    ]
    result = reconstructed.loc[:, safe_context].copy()
    forbidden = [column for column in result if "true" in column.lower()]
    if forbidden:
        raise RuntimeError(f"truth leaked into exported estimator data: {forbidden}")
    return result


def run_geometry_error_experiment(
    subject_id: str,
    scenario_name: str,
    *,
    output_root: str | Path = geometry_error_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int | None = None,
    loss: str = identification_loss,
    observation_modes: Sequence[str] = OBSERVATION_MODES,
    make_plots: bool = True,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Run one complete Stage 4.5D experiment without test-set fitting."""

    scenario = _scenario_with_seed(scenario_name, random_seed)
    effective_seed = scenario.random_seed
    (
        truth,
        observed,
        reconstructed,
        true_geometry,
        assumed_geometry,
        observation_metadata,
    ) = build_geometry_observation_dataset(
        subject_id,
        scenario_name,
        sampling_frequency_hz=sampling_frequency_hz,
        random_seed=effective_seed,
        observation_modes=observation_modes,
    )
    estimates, estimator_inputs = _fit_all_modes(
        reconstructed,
        assumed_geometry,
        loss=loss,
    )
    predictions, prediction_metrics, generic_comparison = _predict_all_modes(
        reconstructed,
        assumed_geometry,
        estimates,
    )
    predictions, domain_metrics, domain_bounds = (
        attach_and_evaluate_domain_membership(predictions)
    )
    kinematic_metrics = compute_kinematic_reconstruction_metrics(predictions)
    observation_metrics = compute_observation_metrics(predictions)

    # Loading virtual parameter truth occurs only after every optimizer call.
    subject = get_dynamic_subject(subject_id)
    baseline_subject = get_dynamic_subject("baseline")
    true_parameters = _parameters_from_subject(subject, baseline_subject)
    parameter_errors = compute_parameter_error_table(
        subject_id,
        scenario_name,
        {
            mode: estimate.estimated_parameters
            for mode, estimate in estimates.items()
        },
        true_parameters,
    )
    q0_rows = []
    template = _assumed_template(assumed_geometry)
    for mode, estimate in estimates.items():
        q0_audit = q0_stiffness_correlation_analysis(
            estimator_inputs[mode],
            template,
            estimate.estimated_parameters,
            assumed_geometry.L1_assumed_m,
        )
        q0_rows.append(
            {
                "subject_id": subject_id,
                "scenario_name": scenario_name,
                "observation_mode": mode,
                **q0_audit,
            }
        )
    q0_correlation = pd.DataFrame(q0_rows)
    identification_metrics = parameter_errors.merge(
        q0_correlation,
        on=("subject_id", "scenario_name", "observation_mode"),
        how="left",
    )
    for mode, estimate in estimates.items():
        selected = identification_metrics["observation_mode"].eq(mode)
        identification_metrics.loc[selected, "optimizer_success"] = (
            estimate.optimizer_success
        )
        identification_metrics.loc[selected, "valid_training_samples"] = (
            estimate.valid_training_samples
        )
        identification_metrics.loc[selected, "optimizer_cost"] = estimate.cost

    mode_comparison = _observation_mode_comparison(
        kinematic_metrics,
        observation_metrics,
        parameter_errors,
        prediction_metrics,
    )
    sensitivity_rows = _sensitivity_rows(
        scenario,
        effective_seed,
        kinematic_metrics,
        observation_metrics,
        parameter_errors,
        prediction_metrics,
    )

    root = Path(output_root)
    destination = root / subject_id / scenario_name
    figure_paths: list[Path] = []
    summary_tables: dict[str, pd.DataFrame] = {}
    metadata = {
        "stage": "4.5D_geometry_and_kinematic_observation_error",
        "model_version": geometry_error_model_version,
        "software_version_or_git_commit": _git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "scenario_name": scenario_name,
        "scenario_configuration": scenario.as_metadata_dict(),
        "random_seed": int(effective_seed),
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "observation_modes": list(observation_modes),
        "observation_metadata": observation_metadata,
        "angle_definition": "theta_shank = q_hip - q_knee",
        "hip_angle_limit_deg": list(hip_range_deg),
        "estimated_parameter_names": list(PARAMETER_NAMES),
        "main_estimator_parameter_count": len(PARAMETER_NAMES),
        "q0_in_main_estimator": False,
        "fit_split": ["train"],
        "validation_used_for_parameter_fit": False,
        "test_used_for_fit_filter_derivative_geometry_or_model_selection": False,
        "derivative_configuration_pre_fixed": True,
        "estimator_input_columns": list(GEOMETRY_ESTIMATOR_INPUT_COLUMNS),
        "true_state_available_to_nonoracle_estimator": False,
        "true_geometry_available_to_estimator": False,
        "scenario_or_subject_id_available_to_estimator": False,
        "runtime_domain_columns": [
            "q_hip_est_rad",
            "q_knee_est_rad",
            "dq_hip_est_rad_s",
            "dq_knee_est_rad_s",
            "ddq_hip_est_rad_s2",
            "ddq_knee_est_rad_s2",
        ],
        "true_domain_membership_evaluation_only": True,
        "static_L2_calibration_error_only": True,
        "dynamic_strap_slip_modelled": False,
        "oracle_upper_bound_only": True,
        "software_virtual_data_only": True,
        "real_robot_connected": False,
        "disclaimer": (
            "Offline virtual geometry/observation-chain validation only; not a "
            "real-patient estimate, clinical threshold, robot command, or safety "
            "limit."
        ),
    }

    if save_outputs:
        destination.mkdir(parents=True, exist_ok=True)
        truth.to_csv(destination / "true_trajectory.csv", index=False)
        observed.to_csv(destination / "observed_trajectory.csv", index=False)
        predictions.to_csv(destination / "reconstructed_state.csv", index=False)
        estimator_export = _estimator_ready_export(predictions)
        for split, filename in SPLIT_FILENAMES.items():
            estimator_export.loc[
                estimator_export["dataset_split"].eq(split)
            ].to_csv(destination / filename, index=False)
        kinematic_metrics.to_csv(destination / "kinematic_metrics.csv", index=False)
        identification_metrics.to_csv(
            destination / "identification_metrics.csv", index=False
        )
        prediction_metrics.to_csv(destination / "prediction_metrics.csv", index=False)
        domain_metrics.to_csv(destination / "domain_metrics.csv", index=False)
        generic_comparison.to_csv(
            destination / "generic_vs_identified_comparison.csv", index=False
        )
        mode_comparison.to_csv(
            destination / "observation_mode_comparison.csv", index=False
        )
        q0_correlation.to_csv(destination / "q0_k_correlation.csv", index=False)
        _write_json(
            destination / "estimated_parameters.json",
            {
                mode: estimate.as_serializable_dict()
                for mode, estimate in estimates.items()
            },
        )
        _write_json(
            destination / "true_geometry.json",
            {
                **true_geometry.as_metadata_dict(),
                "evaluation_only": True,
                "available_to_estimator": False,
            },
        )
        _write_json(
            destination / "assumed_geometry.json",
            assumed_geometry.as_metadata_dict(),
        )
        _write_json(destination / "metadata.json", metadata)

        summaries = root / "summaries"
        summary_tables["experiment_summary"] = _upsert_csv(
            summaries / "experiment_summary.csv",
            sensitivity_rows,
            ("subject_id", "scenario_name", "observation_mode", "random_seed"),
        )
        summary_tables["observation_mode_comparison"] = _upsert_csv(
            summaries / "observation_mode_comparison.csv",
            mode_comparison,
            ("subject_id", "scenario_name", "observation_mode"),
        )
        summary_tables["parameter_errors"] = _upsert_csv(
            summaries / "parameter_errors.csv",
            parameter_errors,
            ("subject_id", "scenario_name", "observation_mode"),
        )
        summary_tables["prediction_metrics"] = _upsert_csv(
            summaries / "prediction_metrics.csv",
            prediction_metrics,
            (
                "subject_id",
                "scenario_name",
                "observation_mode",
                "dataset_split",
                "prediction_model",
            ),
        )
        ranking = build_geometry_sensitivity_ranking(
            summary_tables["experiment_summary"]
        )
        ranking.to_csv(summaries / "geometry_sensitivity_ranking.csv", index=False)
        summary_tables["geometry_sensitivity_ranking"] = ranking
        if make_plots:
            from .visualize_geometry_error import generate_geometry_error_visualizations

            figure_paths = generate_geometry_error_visualizations(
                predictions=predictions,
                kinematic_metrics=kinematic_metrics,
                identification_metrics=identification_metrics,
                prediction_metrics=prediction_metrics,
                observation_mode_comparison=mode_comparison,
                sensitivity_ranking=ranking,
                subject_id=subject_id,
                scenario_name=scenario_name,
                output_directory=destination,
            )

    return {
        "subject_id": subject_id,
        "scenario_name": scenario_name,
        "random_seed": effective_seed,
        "output_dir": destination,
        "truth": truth,
        "observed": observed,
        "reconstructed_state": predictions,
        "true_geometry": true_geometry,
        "assumed_geometry": assumed_geometry,
        "calibration_errors": calibration_error_from_geometries(
            true_geometry, assumed_geometry
        ),
        "estimates": estimates,
        "estimator_inputs": estimator_inputs,
        "kinematic_metrics": kinematic_metrics,
        "observation_metrics": observation_metrics,
        "identification_metrics": identification_metrics,
        "prediction_metrics": prediction_metrics,
        "domain_metrics": domain_metrics,
        "domain_bounds": domain_bounds,
        "generic_vs_identified_comparison": generic_comparison,
        "observation_mode_comparison": mode_comparison,
        "sensitivity_rows": sensitivity_rows,
        "q0_correlation": q0_correlation,
        "metadata": metadata,
        "figure_paths": figure_paths,
        "summary_tables": summary_tables,
    }


def _noise_study_modes(scenario: GeometryErrorScenario) -> tuple[str, ...]:
    modes = []
    if scenario.tcp_position_noise_std_m > 0.0:
        modes.append(TCP_INVERSE_KINEMATICS)
    if scenario.independent_angle_noise_std_rad > 0.0:
        modes.append(INDEPENDENT_JOINT_MEASUREMENT)
    return tuple(modes)


def run_noise_seed_study(
    subject_id: str = "baseline",
    *,
    scenario_names: Iterable[str] = BASE_GEOMETRY_ERROR_SCENARIOS,
    number_of_seeds: int = geometry_noise_seed_count,
    output_root: str | Path = geometry_error_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run at least 20 repeats for each randomised observation scenario."""

    if number_of_seeds < 1:
        raise ValueError("number_of_seeds must be positive.")
    rows = []
    for scenario_name in scenario_names:
        scenario = get_geometry_error_scenario(scenario_name)
        modes = _noise_study_modes(scenario)
        if not scenario.randomized or not modes:
            continue
        for offset in range(number_of_seeds):
            result = run_geometry_error_experiment(
                subject_id,
                scenario_name,
                output_root=output_root,
                sampling_frequency_hz=sampling_frequency_hz,
                random_seed=geometry_error_random_seed + 1000 * (offset + 1),
                observation_modes=modes,
                make_plots=False,
                save_outputs=False,
            )
            rows.append(result["sensitivity_rows"])
    if not rows:
        raise ValueError("scenario_names contains no randomised scenario.")
    runs = pd.concat(rows, ignore_index=True)
    ranking = build_geometry_sensitivity_ranking(runs)
    statistic_rows: list[dict[str, object]] = []
    statistic_metrics = (
        "joint_angle_rmse_deg",
        "tau_observation_combined_rmse_nm",
        "interpolation_combined_nrmse_percent",
        "maximum_parameter_error_percent",
        "invalid_ik_rate",
    )
    for (scenario_name, mode), group in runs.groupby(
        ["scenario_name", "observation_mode"], sort=False
    ):
        for metric in statistic_metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            statistic_rows.append(
                {
                    "subject_id": subject_id,
                    "scenario_name": scenario_name,
                    "observation_mode": mode,
                    "metric": metric,
                    "run_count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": (
                        float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                    ),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95.0)),
                    "worst_case": float(np.max(values)),
                }
            )
    scenario_statistics = pd.DataFrame(statistic_rows)
    summaries = Path(output_root) / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    runs.to_csv(summaries / "noise_seed_runs.csv", index=False)
    scenario_statistics.to_csv(
        summaries / "noise_seed_scenario_statistics.csv", index=False
    )
    ranking.to_csv(summaries / "noise_seed_sensitivity_ranking.csv", index=False)
    accumulated = _upsert_csv(
        summaries / "experiment_summary.csv",
        runs,
        ("subject_id", "scenario_name", "observation_mode", "random_seed"),
    )
    accumulated_ranking = build_geometry_sensitivity_ranking(accumulated)
    accumulated_ranking.to_csv(
        summaries / "geometry_sensitivity_ranking.csv", index=False
    )
    return runs, accumulated_ranking


def _print_result(result: Mapping[str, object]) -> None:
    comparison = result["observation_mode_comparison"]
    assert isinstance(comparison, pd.DataFrame)
    print(f"{result['subject_id']}/{result['scenario_name']}")
    for row in comparison.itertuples(index=False):
        print(
            f"  {row.observation_mode}: q_RMSE={row.joint_angle_rmse_deg:.4g} deg, "
            f"interpolation NRMSE={row.combined_torque_nrmse_percent:.4g}%"
        )
    print(result["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", nargs="?", choices=tuple(DYNAMIC_SUBJECTS))
    parser.add_argument("scenario_name", nargs="?", choices=GEOMETRY_ERROR_SCENARIOS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-baseline", action="store_true")
    group.add_argument("--all-sensitivity", action="store_true")
    group.add_argument("--all-core-subjects", action="store_true")
    group.add_argument("--noise-seed-study", action="store_true")
    parser.add_argument("--output-root", type=Path, default=geometry_error_data_dir)
    parser.add_argument(
        "--sampling-frequency-hz",
        type=float,
        default=dynamic_sampling_frequency_hz,
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--noise-seeds", type=int, default=geometry_noise_seed_count)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.noise_seed_study:
        runs, ranking = run_noise_seed_study(
            "baseline",
            number_of_seeds=args.noise_seeds,
            output_root=args.output_root,
            sampling_frequency_hz=args.sampling_frequency_hz,
        )
        print(f"noise seed runs: {len(runs)}")
        print(ranking.to_string(index=False))
        if not args.no_plots:
            from .visualize_geometry_error import (
                generate_geometry_error_summary_visualizations,
            )

            generate_geometry_error_summary_visualizations(
                args.output_root / "summaries"
            )
        return
    if args.all_baseline:
        experiments = [
            ("baseline", scenario) for scenario in BASE_GEOMETRY_ERROR_SCENARIOS
        ]
    elif args.all_sensitivity:
        experiments = [
            ("baseline", scenario) for scenario in GEOMETRY_SENSITIVITY_VARIANTS
        ]
    elif args.all_core_subjects:
        experiments = [
            (subject, scenario)
            for subject in DYNAMIC_SUBJECTS
            for scenario in CORE_FOUR_SUBJECT_SCENARIOS
        ]
    else:
        if args.subject_id is None or args.scenario_name is None:
            parser.error(
                "provide subject_id and scenario_name, or choose a batch flag"
            )
        experiments = [(args.subject_id, args.scenario_name)]

    for subject_id, scenario_name in experiments:
        result = run_geometry_error_experiment(
            subject_id,
            scenario_name,
            output_root=args.output_root,
            sampling_frequency_hz=args.sampling_frequency_hz,
            random_seed=args.random_seed,
            make_plots=not args.no_plots,
        )
        _print_result(result)
    if len(experiments) > 1 and not args.no_plots:
        from .visualize_geometry_error import (
            generate_geometry_error_summary_visualizations,
        )

        generate_geometry_error_summary_visualizations(
            args.output_root / "summaries"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "CORE_FOUR_SUBJECT_SCENARIOS",
    "GEOMETRY_ESTIMATOR_INPUT_COLUMNS",
    "build_geometry_observation_dataset",
    "fit_five_parameter_observation",
    "project_geometry_estimator_inputs",
    "run_geometry_error_experiment",
    "run_noise_seed_study",
]
