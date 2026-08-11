"""批量运行虚拟受试者五参数动力学辨识实验。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    dynamic_sampling_frequency_hz,
    identification_data_dir,
    identification_dataset_split,
    identification_initial_guess,
    identification_lower_bounds,
    identification_loss,
    identification_model_version,
    identification_parameter_names,
    identification_random_seed,
    identification_trajectory_endpoints_deg,
    identification_upper_bounds,
)
from .dynamic_subject import (
    DYNAMIC_SUBJECTS,
    DynamicVirtualSubject,
    get_dynamic_subject,
)
from .identifiability_analysis import (
    IdentifiabilityResult,
    compare_excitation_sets,
    force_amplitude_sensitivity_analysis,
    save_identifiability_outputs,
)
from .identification_dataset import (
    build_identification_dataset,
    split_identification_dataset,
)
from .noise_models import NOISE_SCENARIOS
from .parameter_estimator import (
    PARAMETER_NAMES,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    compute_torque_metrics,
    estimate_subject_parameters,
    measured_joint_torque,
    predict_joint_torque,
    valid_observations,
)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def _true_parameters_for_evaluation(
    subject: DynamicVirtualSubject,
    baseline: DynamicVirtualSubject,
) -> dict[str, float]:
    """最终评价层才调用；估计器模块不导入或查询虚拟真值注册表。"""

    scales = np.array(
        [
            subject.mass_thigh_kg / baseline.mass_thigh_kg,
            subject.mass_shank_kg / baseline.mass_shank_kg,
            subject.inertia_thigh_kg_m2 / baseline.inertia_thigh_kg_m2,
            subject.inertia_shank_kg_m2 / baseline.inertia_shank_kg_m2,
        ]
    )
    if not np.allclose(scales, scales[0], atol=1e-12):
        raise ValueError(
            "truth subject is outside the common mass/inertia scale model."
        )
    return {
        "mass_scale": float(scales[0]),
        "k_hip_nm_per_rad": subject.k_hip_nm_per_rad,
        "k_knee_nm_per_rad": subject.k_knee_nm_per_rad,
        "b_hip_nm_s_per_rad": subject.b_hip_nm_s_per_rad,
        "b_knee_nm_s_per_rad": subject.b_knee_nm_s_per_rad,
    }


def _parameter_evaluation_table(
    subject_id: str,
    noise_scenario: str,
    true_parameters: Mapping[str, float],
    estimated_parameters: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for parameter in PARAMETER_NAMES:
        true_value = float(true_parameters[parameter])
        estimated_value = float(estimated_parameters[parameter])
        absolute_error = abs(estimated_value - true_value)
        relative_error = (
            100.0 * absolute_error / abs(true_value)
            if true_value != 0.0
            else float("nan")
        )
        rows.append(
            {
                "subject_id": subject_id,
                "noise_scenario": noise_scenario,
                "parameter": parameter,
                "true_value": true_value,
                "estimated_value": estimated_value,
                "absolute_error": absolute_error,
                "relative_error_percent": relative_error,
            }
        )
    return pd.DataFrame(rows)


def _prediction_table(
    dataframe: pd.DataFrame,
    template,
    parameters: Mapping[str, float],
) -> pd.DataFrame:
    frames = []
    for split in ("train", "validation", "test"):
        split_data = valid_observations(
            dataframe.loc[dataframe["dataset_split"].eq(split)]
        )
        measured_hip, measured_knee = measured_joint_torque(
            split_data,
            L1,
            L2,
        )
        predicted_hip, predicted_knee = predict_joint_torque(
            split_data,
            template,
            parameters,
            L1,
        )
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": split_data["subject_id"].to_numpy(),
                    "noise_scenario": split_data["noise_scenario"].to_numpy(),
                    "dataset_split": split,
                    "trajectory_family": split_data[
                        "trajectory_family"
                    ].to_numpy(),
                    "speed_profile": split_data["speed_profile"].to_numpy(),
                    "phase": split_data["phase"].to_numpy(),
                    "time_s": split_data["time_s"].to_numpy(dtype=float),
                    "trajectory_sample_index": split_data[
                        "trajectory_sample_index"
                    ].to_numpy(dtype=int),
                    "fx_observed_n": split_data["fx_observed_n"].to_numpy(
                        dtype=float
                    ),
                    "fz_observed_n": split_data["fz_observed_n"].to_numpy(
                        dtype=float
                    ),
                    "tau_measured_hip_nm": measured_hip,
                    "tau_measured_knee_nm": measured_knee,
                    "tau_predicted_hip_nm": predicted_hip,
                    "tau_predicted_knee_nm": predicted_knee,
                    "torque_residual_hip_nm": measured_hip - predicted_hip,
                    "torque_residual_knee_nm": measured_knee - predicted_knee,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _metric_table(
    splits: Mapping[str, pd.DataFrame],
    template,
    estimated_parameters: Mapping[str, float],
    subject_id: str,
    noise_scenario: str,
) -> pd.DataFrame:
    rows = []
    for split, dataframe in splits.items():
        metrics = compute_torque_metrics(
            dataframe,
            template,
            estimated_parameters,
            L1,
            L2,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "noise_scenario": noise_scenario,
                "dataset_split": split,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _upsert_csv(
    path: Path,
    new_rows: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    if path.exists():
        current = pd.read_csv(path)
        combined = pd.concat((current, new_rows), ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined = combined.sort_values(key_columns).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _maximum_correlation(
    result: IdentifiabilityResult,
) -> dict[str, float | str]:
    correlation = np.asarray(result.parameter_correlation, dtype=float)
    upper_indices = np.triu_indices(len(PARAMETER_NAMES), 1)
    flat_index = int(np.argmax(np.abs(correlation[upper_indices])))
    first = int(upper_indices[0][flat_index])
    second = int(upper_indices[1][flat_index])
    return {
        "parameter_1": PARAMETER_NAMES[first],
        "parameter_2": PARAMETER_NAMES[second],
        "correlation": float(correlation[first, second]),
    }


def _clean_acceptance(
    parameter_table: pd.DataFrame,
    metric_table: pd.DataFrame,
) -> dict[str, object]:
    limits = {
        "mass_scale": 2.0,
        "k_hip_nm_per_rad": 2.0,
        "k_knee_nm_per_rad": 2.0,
        "b_hip_nm_s_per_rad": 5.0,
        "b_knee_nm_s_per_rad": 5.0,
    }
    parameter_checks = {
        parameter: bool(
            parameter_table.loc[
                parameter_table["parameter"].eq(parameter),
                "relative_error_percent",
            ].iloc[0]
            < limit
        )
        for parameter, limit in limits.items()
    }
    test_rmse = float(
        metric_table.loc[
            metric_table["dataset_split"].eq("test"),
            "torque_rmse_combined_nm",
        ].iloc[0]
    )
    return {
        "parameter_relative_error_limits_percent": limits,
        "parameter_checks": parameter_checks,
        "test_torque_rmse_combined_nm": test_rmse,
        "all_parameter_checks_passed": all(parameter_checks.values()),
        "clean_acceptance_passed": all(parameter_checks.values())
        and test_rmse < 1e-6,
    }


def run_identification_experiment(
    subject_id: str,
    noise_scenario: str,
    *,
    output_root: str | Path = identification_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int = identification_random_seed,
    loss: str = identification_loss,
    make_plots: bool = True,
) -> dict[str, object]:
    """运行一名虚拟受试者、一个场景并保存全部规定产物。"""

    subject = get_dynamic_subject(subject_id)
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    dataset = build_identification_dataset(
        subject,
        noise_scenario,
        sampling_frequency_hz=sampling_frequency_hz,
        random_seed=random_seed,
    )
    noise_metadata = dataset.attrs["noise_metadata"]
    splits = split_identification_dataset(dataset)
    if not splits["train"]["dataset_split"].eq("train").all():
        raise RuntimeError("non-training trajectories entered the optimizer input.")

    estimate = estimate_subject_parameters(
        splits["train"],
        template,
        L1,
        L2,
        initial_guess=identification_initial_guess,
        bounds=(identification_lower_bounds, identification_upper_bounds),
        loss=loss,
    )
    true_parameters = _true_parameters_for_evaluation(subject, baseline)
    parameter_table = _parameter_evaluation_table(
        subject_id,
        noise_scenario,
        true_parameters,
        estimate.estimated_parameters,
    )
    metric_table = _metric_table(
        splits,
        template,
        estimate.estimated_parameters,
        subject_id,
        noise_scenario,
    )
    predictions = _prediction_table(
        dataset,
        template,
        estimate.estimated_parameters,
    )

    identifiability = compare_excitation_sets(
        dataset,
        template,
        estimate.estimated_parameters,
        L1,
        L2,
    )
    force_amplitude = force_amplitude_sensitivity_analysis(
        dataset,
        template,
        estimate.estimated_parameters,
        L1,
        L2,
    )
    valid_train = valid_observations(splits["train"])
    geometry_limit = float(
        valid_train["jacobian_condition_number"].quantile(0.95)
    )
    geometry_filtered_train = splits["train"].loc[
        splits["train"]["jacobian_condition_number"] <= geometry_limit
    ]
    geometry_filtered_estimate = estimate_subject_parameters(
        geometry_filtered_train,
        template,
        L1,
        L2,
        initial_guess=identification_initial_guess,
        bounds=(identification_lower_bounds, identification_upper_bounds),
        loss=loss,
    )

    root = Path(output_root)
    destination = root / subject_id / noise_scenario
    destination.mkdir(parents=True, exist_ok=True)
    for split, dataframe in splits.items():
        filename = {
            "train": "training_data.csv",
            "validation": "validation_data.csv",
            "test": "test_data.csv",
        }[split]
        dataframe.to_csv(destination / filename, index=False)
    parameter_table.to_csv(destination / "parameter_estimates.csv", index=False)
    metric_table.to_csv(destination / "dataset_metrics.csv", index=False)
    predictions.to_csv(destination / "predicted_vs_measured.csv", index=False)
    force_amplitude.to_csv(
        destination / "force_amplitude_sensitivity.csv",
        index=False,
    )
    save_identifiability_outputs(
        identifiability,
        destination,
        force_amplitude_analysis=force_amplitude,
    )

    _write_json(
        destination / "estimated_parameters.json",
        estimate.as_serializable_dict(),
    )
    _write_json(
        destination / "metrics.json",
        {
            row["dataset_split"]: {
                key: value
                for key, value in row.items()
                if key
                not in {"subject_id", "noise_scenario", "dataset_split"}
            }
            for row in metric_table.to_dict(orient="records")
        },
    )
    _write_json(
        destination / "geometry_filtered_estimated_parameters.json",
        {
            "jacobian_condition_95th_percentile_limit": geometry_limit,
            "full_valid_training_samples": len(valid_train),
            "filtered_valid_training_samples": (
                geometry_filtered_estimate.valid_training_samples
            ),
            "full_sample_estimate": estimate.estimated_parameters,
            "geometry_filtered_estimate": (
                geometry_filtered_estimate.estimated_parameters
            ),
            "filter_reason": (
                "Sensitivity-only removal of the top 5% Jacobian-condition "
                "region; no sample was removed because of force magnitude."
            ),
        },
    )

    complete_identifiability = identifiability[
        "C_all_families_all_speeds"
    ]
    summary = {
        "subject_id": subject_id,
        "noise_scenario": noise_scenario,
        "optimizer_success": estimate.optimizer_success,
        "optimizer_message": estimate.optimizer_message,
        "estimated_parameters": estimate.estimated_parameters,
        "parameter_evaluation": parameter_table.to_dict(orient="records"),
        "dataset_metrics": metric_table.to_dict(orient="records"),
        "identifiability": {
            name: result.as_serializable_dict()
            for name, result in identifiability.items()
        },
        "highest_absolute_parameter_correlation": _maximum_correlation(
            complete_identifiability
        ),
        "optimizer_success_is_not_identifiability_proof": True,
        "full_set_numerical_rank": complete_identifiability.numerical_rank,
        "software_virtual_subject_only": True,
    }
    if noise_scenario == "clean":
        summary["clean_acceptance"] = _clean_acceptance(
            parameter_table,
            metric_table,
        )
    _write_json(destination / "identification_summary.json", summary)

    metadata = {
        "model_version": identification_model_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "noise_scenario": noise_scenario,
        "noise_model": noise_metadata,
        "sampling_frequency_hz": sampling_frequency_hz,
        "random_seed": random_seed,
        "trajectory_id": "identification_excitation_trajectory",
        "trajectory_endpoints_deg": identification_trajectory_endpoints_deg,
        "trajectory_split": {
            f"{family}/{speed}": split
            for (family, speed), split in identification_dataset_split.items()
        },
        "angle_definition": "theta_shank = q_hip - q_knee",
        "known_parameters": [
            "L1",
            "L2",
            "center_of_mass_distances",
            "neutral_angles",
            "baseline_mass_and_inertia_template",
        ],
        "estimated_parameters": list(identification_parameter_names),
        "truth_parameters_available_to_estimator": False,
        "tau_total_available_to_estimator": False,
        "test_split_used_for_fitting": False,
        "force_samples_scaled_or_clipped_for_identification": False,
        "disclaimer": (
            "Software-only virtual-subject identification. Not a real patient "
            "estimate, robot command, clinical force, or safety threshold."
        ),
    }
    _write_json(destination / "metadata.json", metadata)

    root.mkdir(parents=True, exist_ok=True)
    aggregate_parameters = _upsert_csv(
        root / "parameter_estimates.csv",
        parameter_table,
        ["subject_id", "noise_scenario", "parameter"],
    )
    _upsert_csv(
        root / "dataset_metrics.csv",
        metric_table,
        ["subject_id", "noise_scenario", "dataset_split"],
    )
    aggregate_summary_path = root / "identification_summary.json"
    if aggregate_summary_path.exists():
        aggregate_summary = json.loads(
            aggregate_summary_path.read_text(encoding="utf-8")
        )
    else:
        aggregate_summary = {
            "model_version": identification_model_version,
            "experiments": {},
            "disclaimer": (
                "All entries are software virtual-subject identification only."
            ),
        }
    aggregate_summary["experiments"][
        f"{subject_id}/{noise_scenario}"
    ] = summary
    _write_json(aggregate_summary_path, aggregate_summary)

    figure_paths: list[Path] = []
    if make_plots:
        from .visualize_identification import (
            generate_identification_visualizations,
        )

        figure_paths = generate_identification_visualizations(
            parameter_table,
            predictions,
            identifiability,
            aggregate_parameters,
            subject_id,
            noise_scenario,
            destination,
        )

    return {
        "subject_id": subject_id,
        "noise_scenario": noise_scenario,
        "output_dir": destination,
        "estimate": estimate,
        "parameter_table": parameter_table,
        "metric_table": metric_table,
        "identifiability": identifiability,
        "figure_paths": figure_paths,
        "summary": summary,
    }


def _print_result(result: Mapping[str, object]) -> None:
    estimate = result["estimate"]
    assert isinstance(estimate, ParameterEstimationResult)
    metric_table = result["metric_table"]
    assert isinstance(metric_table, pd.DataFrame)
    test_rmse = metric_table.loc[
        metric_table["dataset_split"].eq("test"),
        "torque_rmse_combined_nm",
    ].iloc[0]
    print(
        f"{result['subject_id']}/{result['noise_scenario']}: "
        f"success={estimate.optimizer_success}, "
        f"test RMSE={test_rmse:.6g} N·m, "
        f"parameters={estimate.estimated_parameters}"
    )
    print(result["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", nargs="?", choices=tuple(DYNAMIC_SUBJECTS))
    parser.add_argument("noise_scenario", nargs="?", choices=NOISE_SCENARIOS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all-clean",
        action="store_true",
        help="运行四名虚拟受试者的 clean 辨识。",
    )
    group.add_argument(
        "--all-scenarios",
        action="store_true",
        help="运行四名虚拟受试者的全部 clean/noise/angle 场景。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=identification_data_dir,
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=dynamic_sampling_frequency_hz,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=identification_random_seed,
    )
    parser.add_argument(
        "--loss",
        choices=("soft_l1", "linear"),
        default=identification_loss,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.all_clean:
        jobs = [(subject_id, "clean") for subject_id in DYNAMIC_SUBJECTS]
    elif args.all_scenarios:
        jobs = [
            (subject_id, scenario)
            for subject_id in DYNAMIC_SUBJECTS
            for scenario in NOISE_SCENARIOS
        ]
    elif args.subject_id and args.noise_scenario:
        jobs = [(args.subject_id, args.noise_scenario)]
    else:
        parser.error(
            "provide SUBJECT_ID NOISE_SCENARIO, --all-clean, or --all-scenarios"
        )

    for subject_id, scenario in jobs:
        result = run_identification_experiment(
            subject_id,
            scenario,
            output_root=args.output_dir,
            sampling_frequency_hz=args.sampling_frequency,
            random_seed=args.seed,
            loss=args.loss,
            make_plots=not args.no_plots,
        )
        _print_result(result)


if __name__ == "__main__":
    main()
