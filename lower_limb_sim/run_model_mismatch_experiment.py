"""Run stage 4.5C virtual-subject model-mismatch experiments.

The complex generator is deliberately isolated from the existing five-parameter
estimator.  Only q, dq, ddq and the reconstructed endpoint wrench projection are
passed into :func:`estimate_subject_parameters`; every complex torque term is
retained solely in the saved audit dataset and final evaluation layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    dynamic_sampling_frequency_hz,
    hip_range_deg,
    identification_dataset_split,
    identification_initial_guess,
    identification_loss,
    identification_lower_bounds,
    identification_parameter_names,
    identification_upper_bounds,
    model_mismatch_data_dir,
    model_mismatch_model_version,
    model_mismatch_random_seed,
)
from .dynamic_subject import DYNAMIC_SUBJECTS, DynamicVirtualSubject, get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .generalization_trajectories import (
    GENERALIZATION_TRAJECTORY_NAMES,
    build_generalization_trajectory_set,
)
from .mismatch_dynamics import MismatchDynamicsResult, mismatch_inverse_dynamics
from .mismatch_metrics import (
    build_generic_vs_identified_comparison,
    compute_all_model_metrics,
    evaluate_model_mismatch_predictions,
)
from .mismatch_scenarios import (
    MISMATCH_SCENARIOS,
    MismatchScenario,
    get_mismatch_scenario,
)
from .observation_model import joint_torque_from_endpoint_force
from .parameter_estimator import (
    PARAMETER_NAMES,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    estimate_subject_parameters,
    measured_joint_torque,
    predict_joint_torque,
)
from .trajectory_profiles import generate_identification_excitation_trajectory


# This is the complete and intentionally narrow estimator interface.  In
# particular, it excludes subject/scenario IDs, split labels, true torques and
# all nonlinear/coupling/residual generator fields.
ESTIMATOR_INPUT_COLUMNS = (
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

IDENTIFICATION_SPLITS_USED = ("train", "validation")
GENERALIZATION_SPLITS = (
    "interpolation_test",
    "boundary_test",
    "outside_domain_test",
)
ALL_EVALUATION_SPLITS = (*IDENTIFICATION_SPLITS_USED, *GENERALIZATION_SPLITS)

_DYNAMICS_TERM_COLUMNS = {
    "tau_inertia_hip_nm": "tau_generator_inertia_hip_nm",
    "tau_inertia_knee_nm": "tau_generator_inertia_knee_nm",
    "tau_coriolis_hip_nm": "tau_generator_coriolis_hip_nm",
    "tau_coriolis_knee_nm": "tau_generator_coriolis_knee_nm",
    "tau_gravity_hip_nm": "tau_generator_gravity_hip_nm",
    "tau_gravity_knee_nm": "tau_generator_gravity_knee_nm",
    "tau_linear_damping_hip_nm": "tau_generator_linear_damping_hip_nm",
    "tau_linear_damping_knee_nm": "tau_generator_linear_damping_knee_nm",
    "tau_linear_stiffness_hip_nm": "tau_generator_linear_stiffness_hip_nm",
    "tau_linear_stiffness_knee_nm": "tau_generator_linear_stiffness_knee_nm",
    "tau_nonlinear_stiffness_hip_nm": (
        "tau_generator_nonlinear_stiffness_hip_nm"
    ),
    "tau_nonlinear_stiffness_knee_nm": (
        "tau_generator_nonlinear_stiffness_knee_nm"
    ),
    "tau_coupling_hip_nm": "tau_generator_coupling_hip_nm",
    "tau_coupling_knee_nm": "tau_generator_coupling_knee_nm",
    "tau_nonlinear_damping_hip_nm": "tau_generator_nonlinear_damping_hip_nm",
    "tau_nonlinear_damping_knee_nm": "tau_generator_nonlinear_damping_knee_nm",
    "tau_residual_hip_nm": "tau_generator_structured_residual_hip_nm",
    "tau_residual_knee_nm": "tau_generator_structured_residual_knee_nm",
    "tau_mismatch_hip_nm": "tau_generator_total_mismatch_hip_nm",
    "tau_mismatch_knee_nm": "tau_generator_total_mismatch_knee_nm",
    "tau_base_total_hip_nm": "tau_generator_linear_base_total_hip_nm",
    "tau_base_total_knee_nm": "tau_generator_linear_base_total_knee_nm",
}


def _json_safe(value: object) -> object:
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _append_reason(
    current: np.ndarray,
    mask: np.ndarray,
    reason: str,
) -> np.ndarray:
    result = np.asarray(current, dtype=object).copy()
    for index in np.flatnonzero(np.asarray(mask, dtype=bool)):
        previous = str(result[index])
        result[index] = f"{previous};{reason}" if previous else reason
    return result.astype(str)


def project_estimator_inputs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the only columns the five-parameter estimator may access.

    This explicit projection is the primary data-leakage boundary.  The output
    intentionally has no attrs because pandas metadata could otherwise carry a
    scenario name or complex generator truth beside an apparently clean table.
    """

    missing = set(ESTIMATOR_INPUT_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(
            "model-mismatch dataframe is missing estimator observations: "
            f"{sorted(missing)}"
        )
    projected = dataframe.loc[:, ESTIMATOR_INPUT_COLUMNS].copy()
    projected.attrs.clear()
    if tuple(projected.columns) != ESTIMATOR_INPUT_COLUMNS:
        raise RuntimeError("estimator input whitelist was not preserved.")
    return projected


def _identification_profiles(
    sampling_frequency_hz: float,
) -> list[pd.DataFrame]:
    profiles: list[pd.DataFrame] = []
    for (family, speed), split in identification_dataset_split.items():
        # The three pre-existing stage-4 test trajectories stay out of fitting
        # and are not relabelled; stage 4.5C has dedicated unseen paths below.
        if split not in IDENTIFICATION_SPLITS_USED:
            continue
        profile = generate_identification_excitation_trajectory(
            family,
            speed,
            sampling_frequency_hz=sampling_frequency_hz,
        ).copy()
        profile["trajectory_id"] = (
            f"identification_excitation_trajectory:{family}:{speed}"
        )
        profile["dataset_split"] = split
        profile["software_validation_trajectory"] = True
        profile["clinical_reference"] = False
        profile["outside_training_domain"] = False
        profile["workspace_valid"] = True
        profile["generalization_family"] = ""
        profiles.append(profile)
    return profiles


def _all_motion_profiles(sampling_frequency_hz: float) -> list[pd.DataFrame]:
    profiles = _identification_profiles(sampling_frequency_hz)
    generalization = build_generalization_trajectory_set(
        sampling_frequency_hz=sampling_frequency_hz
    )
    profiles.extend(
        group.reset_index(drop=True)
        for _, group in generalization.groupby("trajectory_id", sort=False)
    )
    return profiles


def _simulate_profile(
    profile: pd.DataFrame,
    subject_id: str,
    scenario: MismatchScenario,
    generator_subject,
) -> pd.DataFrame:
    q_hip = profile["q_hip_rad"].to_numpy(dtype=float)
    q_knee = profile["q_knee_rad"].to_numpy(dtype=float)
    dq_hip = profile["dq_hip_rad_s"].to_numpy(dtype=float)
    dq_knee = profile["dq_knee_rad_s"].to_numpy(dtype=float)
    ddq_hip = profile["ddq_hip_rad_s2"].to_numpy(dtype=float)
    ddq_knee = profile["ddq_knee_rad_s2"].to_numpy(dtype=float)
    dynamics = mismatch_inverse_dynamics(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
        ddq_hip,
        ddq_knee,
        generator_subject,
        L1,
        residual_random_seed=scenario.random_seed,
    )
    true_hip = np.asarray(dynamics.tau_total_hip_nm, dtype=float)
    true_knee = np.asarray(dynamics.tau_total_knee_nm, dtype=float)
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        true_hip,
        true_knee,
        L1,
        L2,
    )
    measured_hip, measured_knee = joint_torque_from_endpoint_force(
        q_hip,
        q_knee,
        force.fx_robot_on_leg_n,
        force.fz_robot_on_leg_n,
        L1,
        L2,
    )
    workspace_valid = (
        profile["workspace_valid"].astype(bool).to_numpy()
        if "workspace_valid" in profile
        else np.ones(len(profile), dtype=bool)
    )
    valid = np.asarray(force.force_mapping_valid, dtype=bool) & workspace_valid
    finite_observation = np.isfinite(
        np.column_stack(
            (
                q_hip,
                q_knee,
                dq_hip,
                dq_knee,
                ddq_hip,
                ddq_knee,
                force.fx_robot_on_leg_n,
                force.fz_robot_on_leg_n,
                measured_hip,
                measured_knee,
            )
        )
    ).all(axis=1)
    invalid_reason = np.asarray(force.invalid_reason, dtype=str)
    invalid_reason = _append_reason(
        invalid_reason,
        ~workspace_valid,
        "workspace_invalid",
    )
    invalid_reason = _append_reason(
        invalid_reason,
        ~finite_observation,
        "non_finite_observation",
    )
    valid &= finite_observation

    output = profile.copy()
    output["subject_id"] = subject_id
    output["scenario_name"] = scenario.scenario_name
    output["fx_observed_n"] = np.asarray(force.fx_robot_on_leg_n, dtype=float)
    output["fz_observed_n"] = np.asarray(force.fz_robot_on_leg_n, dtype=float)
    output["fx_complex_true_n"] = output["fx_observed_n"]
    output["fz_complex_true_n"] = output["fz_observed_n"]
    output["force_magnitude_observed_n"] = np.asarray(
        force.force_magnitude_n,
        dtype=float,
    )
    output["tau_measured_hip_nm"] = measured_hip
    output["tau_measured_knee_nm"] = measured_knee
    output["tau_complex_true_hip_nm"] = true_hip
    output["tau_complex_true_knee_nm"] = true_knee
    output["torque_reconstruction_consistency_error_nm"] = np.where(
        valid,
        np.hypot(measured_hip - true_hip, measured_knee - true_knee),
        np.nan,
    )
    output["sample_valid"] = valid
    output["force_mapping_valid"] = np.asarray(
        force.force_mapping_valid,
        dtype=bool,
    )
    output["wrench_is_stale"] = False
    output["invalid_reason"] = invalid_reason
    output["jacobian_determinant"] = force.jacobian_determinant
    output["jacobian_condition_number"] = force.jacobian_condition_number
    output["generator_truth_for_audit_only"] = True
    for result_field, output_column in _DYNAMICS_TERM_COLUMNS.items():
        output[output_column] = np.asarray(
            getattr(dynamics, result_field),
            dtype=float,
        )
    return output


def build_model_mismatch_dataset(
    subject_id: str,
    scenario_name: str,
    *,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
) -> pd.DataFrame:
    """Generate train/validation and six dedicated unseen trajectories."""

    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    base_subject = get_dynamic_subject(subject_id)
    scenario = get_mismatch_scenario(scenario_name)
    generator_subject = scenario.create_subject(base_subject)
    frames = [
        _simulate_profile(
            profile,
            subject_id,
            scenario,
            generator_subject,
        )
        for profile in _all_motion_profiles(float(sampling_frequency_hz))
    ]
    dataset = pd.concat(frames, ignore_index=True)
    if set(dataset["dataset_split"].astype(str)) != set(ALL_EVALUATION_SPLITS):
        raise RuntimeError("model-mismatch dataset split coverage is incomplete.")
    dataset.attrs.update(
        {
            "subject_id": subject_id,
            "scenario_name": scenario_name,
            "generator_parameters": dict(scenario.generator_parameters),
            "angle_definition": "theta_shank = q_hip - q_knee",
            "software_virtual_data_only": True,
        }
    )
    return dataset


def split_model_mismatch_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the five fixed evaluation splits without changing labels."""

    observed = set(dataframe["dataset_split"].astype(str))
    missing = set(ALL_EVALUATION_SPLITS).difference(observed)
    unexpected = observed.difference(ALL_EVALUATION_SPLITS)
    if missing or unexpected:
        raise ValueError(
            f"invalid model-mismatch splits; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        split: dataframe.loc[dataframe["dataset_split"].eq(split)].copy()
        for split in ALL_EVALUATION_SPLITS
    }


def _parameters_from_subject(
    subject: DynamicVirtualSubject,
    baseline: DynamicVirtualSubject,
) -> dict[str, float]:
    scales = np.asarray(
        (
            subject.mass_thigh_kg / baseline.mass_thigh_kg,
            subject.mass_shank_kg / baseline.mass_shank_kg,
            subject.inertia_thigh_kg_m2 / baseline.inertia_thigh_kg_m2,
            subject.inertia_shank_kg_m2 / baseline.inertia_shank_kg_m2,
        ),
        dtype=float,
    )
    if not np.allclose(scales, scales[0], atol=1e-12, rtol=0.0):
        raise ValueError("subject does not use the common mass/inertia scale.")
    return {
        "mass_scale": float(scales[0]),
        "k_hip_nm_per_rad": float(subject.k_hip_nm_per_rad),
        "k_knee_nm_per_rad": float(subject.k_knee_nm_per_rad),
        "b_hip_nm_s_per_rad": float(subject.b_hip_nm_s_per_rad),
        "b_knee_nm_s_per_rad": float(subject.b_knee_nm_s_per_rad),
    }


def _predict_models(
    dataframe: pd.DataFrame,
    generic_parameters: Mapping[str, float],
    identified_parameters: Mapping[str, float],
    template,
) -> pd.DataFrame:
    """Predict both models; measured torque is rebuilt only from J.T @ F."""

    measured_hip, measured_knee = measured_joint_torque(dataframe, L1, L2)
    generic_hip, generic_knee = predict_joint_torque(
        dataframe,
        template,
        generic_parameters,
        L1,
    )
    identified_hip, identified_knee = predict_joint_torque(
        dataframe,
        template,
        identified_parameters,
        L1,
    )
    q_hip = dataframe["q_hip_rad"].to_numpy(dtype=float)
    q_knee = dataframe["q_knee_rad"].to_numpy(dtype=float)
    generic_force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        generic_hip,
        generic_knee,
        L1,
        L2,
    )
    identified_force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        identified_hip,
        identified_knee,
        L1,
        L2,
    )
    sample_valid = dataframe["sample_valid"].astype(bool).to_numpy()
    result = dataframe.copy()
    # Canonical evaluation truth is reconstructed from observed wrench here;
    # tau_complex_true remains an audit field and is never consumed.
    result["tau_true_hip_nm"] = measured_hip
    result["tau_true_knee_nm"] = measured_knee
    result["tau_generic_hip_nm"] = generic_hip
    result["tau_generic_knee_nm"] = generic_knee
    result["tau_identified_hip_nm"] = identified_hip
    result["tau_identified_knee_nm"] = identified_knee
    result["fx_true_n"] = dataframe["fx_observed_n"].to_numpy(dtype=float)
    result["fz_true_n"] = dataframe["fz_observed_n"].to_numpy(dtype=float)
    result["fx_generic_n"] = np.asarray(
        generic_force.fx_robot_on_leg_n,
        dtype=float,
    )
    result["fz_generic_n"] = np.asarray(
        generic_force.fz_robot_on_leg_n,
        dtype=float,
    )
    result["fx_identified_n"] = np.asarray(
        identified_force.fx_robot_on_leg_n,
        dtype=float,
    )
    result["fz_identified_n"] = np.asarray(
        identified_force.fz_robot_on_leg_n,
        dtype=float,
    )
    result["generic_prediction_valid"] = sample_valid & np.asarray(
        generic_force.force_mapping_valid,
        dtype=bool,
    )
    result["identified_prediction_valid"] = sample_valid & np.asarray(
        identified_force.force_mapping_valid,
        dtype=bool,
    )
    result["generic_residual_hip_nm"] = measured_hip - generic_hip
    result["generic_residual_knee_nm"] = measured_knee - generic_knee
    result["identified_residual_hip_nm"] = measured_hip - identified_hip
    result["identified_residual_knee_nm"] = measured_knee - identified_knee
    return result


def _parameter_shift_table(
    subject_id: str,
    scenario_name: str,
    generic_parameters: Mapping[str, float],
    generator_linear_parameters: Mapping[str, float],
    identified_parameters: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for parameter in PARAMETER_NAMES:
        generic = float(generic_parameters[parameter])
        generator_linear = float(generator_linear_parameters[parameter])
        identified = float(identified_parameters[parameter])
        rows.append(
            {
                "subject_id": subject_id,
                "scenario_name": scenario_name,
                "parameter": parameter,
                "generic_value": generic,
                "generator_linear_value": generator_linear,
                "identified_value": identified,
                "identified_equivalent_value": identified,
                "identified_minus_generic": identified - generic,
                "identified_minus_generator_linear": identified - generator_linear,
                "identified_vs_generic_percent": (
                    100.0 * (identified - generic) / abs(generic)
                    if generic != 0.0
                    else np.nan
                ),
                "interpretation": "trajectory_local_equivalent_parameter",
            }
        )
    return pd.DataFrame(rows)


def _split_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    drop_identity = [
        column
        for column in (
            "trajectory_id",
            "trajectory_family",
            "trajectory_name",
            "speed_profile",
        )
        if column in predictions
    ]
    return compute_all_model_metrics(predictions.drop(columns=drop_identity))


def _upsert_csv(
    path: Path,
    rows: pd.DataFrame,
    key_columns: Iterable[str],
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        combined = pd.concat((pd.read_csv(path), rows), ignore_index=True)
    else:
        combined = rows.copy()
    keys = list(key_columns)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def run_model_mismatch_experiment(
    subject_id: str,
    scenario_name: str,
    *,
    output_root: str | Path = model_mismatch_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int = model_mismatch_random_seed,
    loss: str = identification_loss,
    make_plots: bool = True,
) -> dict[str, object]:
    """Run one reliable-alignment, software-only model-mismatch experiment."""

    # Random seed is metadata for orchestration reproducibility; each scenario
    # owns its fixed residual seed so scenario output is call-order independent.
    if isinstance(random_seed, bool) or not isinstance(
        random_seed,
        (int, np.integer),
    ):
        raise TypeError("random_seed must be an integer.")
    base_subject = get_dynamic_subject(subject_id)
    baseline_subject = get_dynamic_subject("baseline")
    scenario = get_mismatch_scenario(scenario_name)
    template = baseline_template_from_dynamic_subject(baseline_subject)
    dataset = build_model_mismatch_dataset(
        subject_id,
        scenario_name,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    splits = split_model_mismatch_dataset(dataset)

    estimator_training_data = project_estimator_inputs(splits["train"])
    # This is the only optimizer call.  Validation and all unseen test splits
    # are intentionally unavailable to the fitting interface.
    estimate = estimate_subject_parameters(
        estimator_training_data,
        template,
        L1,
        L2,
        initial_guess=identification_initial_guess,
        bounds=(identification_lower_bounds, identification_upper_bounds),
        loss=loss,
    )
    generic_parameters = _parameters_from_subject(
        baseline_subject,
        baseline_subject,
    )
    generator_linear_parameters = _parameters_from_subject(
        base_subject,
        baseline_subject,
    )
    predictions = _predict_models(
        dataset,
        generic_parameters,
        estimate.estimated_parameters,
        template,
    )
    metric_bundle = evaluate_model_mismatch_predictions(
        predictions,
        link_1_m=L1,
        link_2_m=L2,
    )
    trajectory_metrics = metric_bundle.trajectory_metrics.copy()
    trajectory_metrics["metric_scope"] = "trajectory"
    split_metrics = _split_metric_table(predictions)
    split_metrics["metric_scope"] = "split"
    prediction_metrics = pd.concat(
        (trajectory_metrics, split_metrics),
        ignore_index=True,
        sort=False,
    )
    generic_comparison = build_generic_vs_identified_comparison(
        trajectory_metrics
    )
    parameter_shift = _parameter_shift_table(
        subject_id,
        scenario_name,
        generic_parameters,
        generator_linear_parameters,
        estimate.estimated_parameters,
    )

    root = Path(output_root)
    destination = root / subject_id / scenario_name
    destination.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train": "training_data.csv",
        "validation": "validation_data.csv",
        "interpolation_test": "interpolation_test_data.csv",
        "boundary_test": "boundary_test_data.csv",
        "outside_domain_test": "outside_domain_test_data.csv",
    }
    for split, filename in filenames.items():
        splits[split].to_csv(destination / filename, index=False)
    predictions.to_csv(destination / "predicted_vs_true_torque.csv", index=False)
    prediction_metrics.to_csv(destination / "prediction_metrics.csv", index=False)
    generic_comparison.to_csv(
        destination / "generic_vs_identified_comparison.csv",
        index=False,
    )
    metric_bundle.residual_feature_correlations.to_csv(
        destination / "residual_feature_correlations.csv",
        index=False,
    )
    metric_bundle.residual_diagnostics.to_csv(
        destination / "residual_diagnostics.csv",
        index=False,
    )
    parameter_shift.to_csv(destination / "parameter_shift.csv", index=False)

    estimated_payload = {
        **estimate.as_serializable_dict(),
        "parameter_interpretation": "trajectory_local_equivalent_linear_parameters",
        "not_direct_tissue_properties": True,
        "generator_truth_used_for_fitting": False,
    }
    _write_json(destination / "estimated_parameters.json", estimated_payload)
    generator_payload = {
        **scenario.as_metadata_dict(),
        "base_dynamic_subject": base_subject.as_metadata_dict(),
        "generator_linear_parameters": generator_linear_parameters,
        "saved_for_post_fit_audit_only": True,
        "available_to_estimator": False,
    }
    _write_json(destination / "generator_parameters.json", generator_payload)

    train_ids = sorted(splits["train"]["trajectory_id"].astype(str).unique())
    validation_ids = sorted(
        splits["validation"]["trajectory_id"].astype(str).unique()
    )
    test_ids = {
        split: sorted(splits[split]["trajectory_id"].astype(str).unique())
        for split in GENERALIZATION_SPLITS
    }
    excluded_stage4_test_ids = sorted(
        f"identification_excitation_trajectory:{family}:{speed}"
        for (family, speed), split in identification_dataset_split.items()
        if split == "test"
    )
    metadata = {
        "stage": "4.5C_model_mismatch_generalization",
        "model_version": model_mismatch_model_version,
        "software_version_or_git_commit": _git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "scenario_name": scenario_name,
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "random_seed": int(random_seed),
        "generator_random_seed": scenario.random_seed,
        "angle_definition": "theta_shank = q_hip - q_knee",
        "hip_angle_limit_deg": list(hip_range_deg),
        "time_alignment_condition": "reliable_zero_delay",
        "variable_delay_is_primary_experiment_variable": False,
        "estimator_model": scenario.estimator_model_description,
        "estimated_parameter_names": list(identification_parameter_names),
        "estimated_parameter_interpretation": (
            "equivalent linear parameters local to the training trajectory region"
        ),
        "estimator_input_columns": list(ESTIMATOR_INPUT_COLUMNS),
        "train_trajectory_ids": train_ids,
        "validation_trajectory_ids": validation_ids,
        "test_trajectory_ids": test_ids,
        "excluded_existing_stage4_test_trajectory_ids": excluded_stage4_test_ids,
        "fit_split": ["train"],
        "validation_used_for_parameter_fit": False,
        "test_used_for_parameter_fit_or_model_selection": False,
        "generator_parameters_available_to_estimator": False,
        "tau_complex_true_available_to_estimator": False,
        "scenario_name_available_to_estimator": False,
        "nrmse_formula": (
            "RMSE / (max(true_torque) - min(true_torque) + epsilon) * 100%"
        ),
        "nrmse_small_range_is_flagged": True,
        "outside_domain_is_extrapolation": True,
        "software_validation_trajectory": True,
        "clinical_reference_trajectory": False,
        "software_virtual_subject_only": True,
        "real_robot_control_connected": False,
        "disclaimer": (
            "Complex virtual-subject generator versus a simplified five-parameter "
            "model. Not a real-patient estimate, clinical validation, robot "
            "command, or safety threshold."
        ),
    }
    _write_json(destination / "metadata.json", metadata)

    summary_dir = root / "summaries"
    _upsert_csv(
        summary_dir / "equivalent_parameters.csv",
        parameter_shift,
        ("subject_id", "scenario_name", "parameter"),
    )
    split_summary = split_metrics.copy()
    _upsert_csv(
        summary_dir / "split_prediction_metrics.csv",
        split_summary,
        ("subject_id", "scenario_name", "dataset_split", "prediction_model"),
    )
    _upsert_csv(
        summary_dir / "generic_vs_identified_comparison.csv",
        generic_comparison,
        (
            "subject_id",
            "scenario_name",
            "trajectory_id",
            "dataset_split",
            "trajectory_family",
            "speed_profile",
        ),
    )

    figure_paths: list[Path] = []
    if make_plots:
        from .visualize_model_mismatch import (
            generate_model_mismatch_visualizations,
        )

        figure_paths = generate_model_mismatch_visualizations(
            predictions,
            prediction_metrics,
            generic_comparison,
            metric_bundle.residual_feature_correlations,
            parameter_shift,
            subject_id,
            scenario_name,
            destination,
        )

    return {
        "subject_id": subject_id,
        "scenario_name": scenario_name,
        "output_dir": destination,
        "estimate": estimate,
        "dataset": dataset,
        "splits": splits,
        "predictions": predictions,
        "prediction_metrics": prediction_metrics,
        "trajectory_metrics": trajectory_metrics,
        "split_metrics": split_metrics,
        "generic_vs_identified": generic_comparison,
        "residual_feature_correlations": (
            metric_bundle.residual_feature_correlations
        ),
        "residual_diagnostics": metric_bundle.residual_diagnostics,
        "parameter_shift": parameter_shift,
        "figure_paths": figure_paths,
        "metadata": metadata,
    }


def _print_result(result: Mapping[str, object]) -> None:
    estimate = result["estimate"]
    assert isinstance(estimate, ParameterEstimationResult)
    split_metrics = result["split_metrics"]
    assert isinstance(split_metrics, pd.DataFrame)
    interpolation = split_metrics.loc[
        split_metrics["dataset_split"].eq("interpolation_test")
        & split_metrics["prediction_model"].eq("identified")
    ]
    nrmse = float(interpolation["combined_nrmse_percent"].iloc[0])
    print(
        f"{result['subject_id']}/{result['scenario_name']}: "
        f"success={estimate.optimizer_success}, "
        f"interpolation NRMSE={nrmse:.4g}%, "
        f"equivalent_parameters={estimate.estimated_parameters}"
    )
    print(result["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", nargs="?", choices=tuple(DYNAMIC_SUBJECTS))
    parser.add_argument("scenario_name", nargs="?", choices=MISMATCH_SCENARIOS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-baseline", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=model_mismatch_data_dir,
    )
    parser.add_argument(
        "--sampling-frequency-hz",
        type=float,
        default=dynamic_sampling_frequency_hz,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.all_baseline:
        experiments = [("baseline", scenario) for scenario in MISMATCH_SCENARIOS]
    elif args.all:
        experiments = [
            (subject, scenario)
            for subject in DYNAMIC_SUBJECTS
            for scenario in MISMATCH_SCENARIOS
        ]
    else:
        if args.subject_id is None or args.scenario_name is None:
            parser.error(
                "provide subject_id and scenario_name, or use --all-baseline/--all"
            )
        experiments = [(args.subject_id, args.scenario_name)]

    for subject_id, scenario_name in experiments:
        result = run_model_mismatch_experiment(
            subject_id,
            scenario_name,
            output_root=args.output_root,
            sampling_frequency_hz=args.sampling_frequency_hz,
            random_seed=model_mismatch_random_seed,
            make_plots=not args.no_plots,
        )
        _print_result(result)


if __name__ == "__main__":
    main()
