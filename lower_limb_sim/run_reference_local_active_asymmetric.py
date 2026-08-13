"""Run and save the formal active-asymmetric reference-local P0.1 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config
from .formal_protocol import ROM_PROTOCOL_VERSION, THETA_SHANK_DEFINITION
from .parameter_estimator import PARAMETER_NAMES
from .reference_local_active_asymmetric import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    DOMAIN_MODEL,
    HELD_OUT_TRAJECTORY_IDS,
    MODEL_VERSION,
    RANDOM_SEED,
    SPLIT_DEFINITION_ID,
    SPLIT_BY_TRAJECTORY,
    TRAINING_TRAJECTORY_IDS,
    TRAJECTORY_SPECIFICATIONS,
    ActiveReferenceLocalResult,
    run_active_reference_local_identification,
    sha256_file,
)
from .reference_local_excitation import LOCAL_TRAJECTORY_SPLIT, SUBJECT_IDS
from .visualize_reference_local_active_asymmetric import (
    FIGURE_DEFINITIONS,
    generate_active_reference_local_figures,
)


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_local_active_asymmetric"
)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _workspace_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "as_serializable_dict"):
        return _json_ready(value.as_serializable_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _provenance_frame(
    dataframe: pd.DataFrame,
    *,
    generated_at_utc: str,
    git_commit: str | None,
    reference_sha256: str,
) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    provenance = {
        "generated_at_utc": generated_at_utc,
        "git_commit": git_commit,
        "workspace_dirty_at_generation": _workspace_dirty(),
        "experiment_model_version": MODEL_VERSION,
        "model_configuration_id": MODEL_VERSION,
        "split_definition_id": SPLIT_DEFINITION_ID,
        "active_reference_identifier": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": reference_sha256,
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": reference_sha256,
        "random_seed": RANDOM_SEED,
    }
    for column, value in provenance.items():
        if column not in output:
            output[column] = value
    return output


def _truth_parameter_summary(parameter_errors: pd.DataFrame) -> dict[str, object]:
    return {
        "mean_relative_error_percent": float(
            parameter_errors["relative_error_percent"].mean()
        ),
        "maximum_relative_error_percent": float(
            parameter_errors["relative_error_percent"].max()
        ),
        "maximum_error_subject": str(
            parameter_errors.loc[
                parameter_errors["relative_error_percent"].idxmax(), "subject_id"
            ]
        ),
        "maximum_error_parameter": str(
            parameter_errors.loc[
                parameter_errors["relative_error_percent"].idxmax(), "parameter"
            ]
        ),
    }


def _prediction_summary(result: ActiveReferenceLocalResult) -> dict[str, object]:
    identified = result.prediction_metrics.loc[
        result.prediction_metrics["prediction_model"].eq("identified")
    ]
    heldout = identified.loc[identified["dataset_split"].eq("test")]
    exact_active = heldout.loc[
        heldout["trajectory_id"].eq("heldout_active_reference_slow")
    ]
    heldout_comparison = result.generic_vs_identified.loc[
        result.generic_vs_identified["split"].eq("test")
    ]
    return {
        "heldout_trajectory_ids": list(HELD_OUT_TRAJECTORY_IDS),
        "heldout_group_count": int(len(heldout)),
        "heldout_mean_combined_torque_rmse_nm": float(
            heldout["combined_torque_rmse_nm"].mean()
        ),
        "heldout_maximum_combined_torque_rmse_nm": float(
            heldout["combined_torque_rmse_nm"].max()
        ),
        "heldout_mean_combined_nrmse_percent": float(
            heldout["combined_nrmse_percent"].mean()
        ),
        "heldout_maximum_combined_nrmse_percent": float(
            heldout["combined_nrmse_percent"].max()
        ),
        "exact_active_reference_mean_rmse_nm": float(
            exact_active["combined_torque_rmse_nm"].mean()
        ),
        "exact_active_reference_maximum_rmse_nm": float(
            exact_active["combined_torque_rmse_nm"].max()
        ),
        "generic_to_identified_mean_improvement_percent": float(
            heldout_comparison["improvement_percent"].mean()
        ),
        "generic_to_identified_minimum_improvement_percent": float(
            heldout_comparison["improvement_percent"].min()
        ),
        "test_used_for_parameter_fit": False,
    }


def _identifiability_summary(result: ActiveReferenceLocalResult) -> dict[str, object]:
    summary = result.identifiability_summary
    strongest = summary.loc[
        summary["maximum_absolute_parameter_correlation"].idxmax()
    ]
    return {
        "all_subjects_full_rank_five": bool(
            summary["full_rank_five_parameter_model"].astype(bool).all()
        ),
        "rank_values": sorted(set(summary["numerical_rank"].astype(int))),
        "condition_number_minimum": float(summary["condition_number"].min()),
        "condition_number_maximum": float(summary["condition_number"].max()),
        "maximum_absolute_parameter_correlation": float(
            strongest["maximum_absolute_parameter_correlation"]
        ),
        "strongest_correlation_parameter_1": str(
            strongest["strongest_correlation_parameter_1"]
        ),
        "strongest_correlation_parameter_2": str(
            strongest["strongest_correlation_parameter_2"]
        ),
        "strongest_signed_correlation": float(strongest["strongest_correlation"]),
        "highly_correlated_pair_count_total": int(
            summary["highly_correlated_pair_count"].sum()
        ),
        "weakest_information_parameters": sorted(
            set(summary["weakest_information_parameter"].astype(str))
        ),
        "interpretation": (
            "local numerical identifiability for the adopted five-parameter "
            "equivalent model, not unique physiological recovery"
        ),
    }


def _coverage_summary(result: ActiveReferenceLocalResult) -> dict[str, object]:
    coverage = result.domain_coverage
    roles = {
        str(row["trajectory_id"]): {
            "dataset_split": str(row["dataset_split"]),
            "evaluation_role": str(row["evaluation_role"]),
            "in_domain_percent": float(row["in_domain_percent"]),
            "outside_domain_percent": float(row["outside_domain_percent"]),
            "missing_state_variables": str(row["missing_state_variables"]),
        }
        for _, row in coverage.iterrows()
    }
    return {
        "domain_model": DOMAIN_MODEL,
        "state_variables": list(result.domain_bounds.columns),
        "domain_fitted_from_split": "train_only",
        "domain_training_sample_count": result.domain_bounds.valid_training_samples,
        "trajectories": roles,
        "minimum_validation_coverage_percent": float(
            coverage.loc[
                coverage["dataset_split"].eq("validation"), "in_domain_percent"
            ].min()
        ),
        "minimum_test_coverage_percent": float(
            coverage.loc[
                coverage["dataset_split"].eq("test"), "in_domain_percent"
            ].min()
        ),
        "coverage_limit": (
            "axis-aligned marginal bounds do not capture correlations or "
            "constitute a statistical confidence region"
        ),
    }


def _experiment_config(
    result: ActiveReferenceLocalResult,
    context: Mapping[str, object],
    targeted_test_result: str,
    full_test_result: str,
) -> dict[str, object]:
    return {
        **context,
        "experiment_type": "formal_offline_synthetic_experiment",
        "scientific_question": (
            "For the current active closed asymmetric reference, does the "
            "existing reference-local five-parameter pipeline provide local "
            "coverage, numerical identifiability, and held-out prediction?"
        ),
        "random_seed_used": RANDOM_SEED,
        "stochastic_component_present": False,
        "prespecified_virtual_subject_ids": list(SUBJECT_IDS),
        "active_reference": result.reference.summary,
        "trajectory_specifications": [item.__dict__ for item in TRAJECTORY_SPECIFICATIONS],
        "split_definition": {
            "train": list(TRAINING_TRAJECTORY_IDS),
            "validation": [
                key for key, split in SPLIT_BY_TRAJECTORY.items() if split == "validation"
            ],
            "test": list(HELD_OUT_TRAJECTORY_IDS),
        },
        "five_parameter_estimator": {
            "parameter_names": list(PARAMETER_NAMES),
            "initial_generic_parameters": config.identification_initial_guess,
            "lower_bounds": config.identification_lower_bounds,
            "upper_bounds": config.identification_upper_bounds,
            "parameter_scales": config.identification_parameter_scales,
            "loss": config.identification_loss,
            "maximum_function_evaluations": 500,
            "fit_split": "train_only",
            "observation_columns": [
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
                "fx_observed_n",
                "fz_observed_n",
                "sample_valid",
            ],
            "generator_truth_available_to_estimator": False,
        },
        "model_configuration": {
            "L1_m": config.L1,
            "L2_m": config.L2,
            "L2_definition": "knee_to_strap_equivalent_pull_point",
            "theta_shank_convention": "theta_shank = q_hip - q_knee",
            "active_hip_rom_deg": result.reference.metadata["approved_hip_rom_deg"],
            "active_knee_rom_deg": result.reference.metadata["approved_knee_rom_deg"],
            "rom_protocol_version": ROM_PROTOCOL_VERSION,
            "jacobian_determinant_threshold": config.jacobian_det_threshold,
            "jacobian_condition_limit": config.jacobian_condition_limit,
            "force_magnitude_software_anomaly_limit_n": config.force_magnitude_limit_n,
            "velocity_limit": None,
            "velocity_limit_status": "not_configured_offline_peak_report_only",
            "acceleration_limit": None,
            "acceleration_limit_status": "not_configured_offline_peak_report_only",
        },
        "verification": {
            "targeted_tests": targeted_test_result,
            "full_regression": full_test_result,
        },
        "prohibited_code_used": False,
        "robot_connection_performed": False,
        "robot_motion_command_sent": False,
    }


def save_formal_result(
    result: ActiveReferenceLocalResult,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    generate_plots: bool = True,
    targeted_test_result: str = "not_recorded",
    full_test_result: str = "not_recorded",
) -> dict[str, Path]:
    """Save one immutable formal result directory; refuse overwrite."""

    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"formal output directory already exists and is non-empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_directory = destination / "trajectories"
    trajectory_directory.mkdir()

    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit()
    reference_sha = str(result.reference.summary["active_reference_sha256"])
    context = {
        "generated_at_utc": generated_at,
        "git_commit": git_commit,
        "software_version_or_git_commit": git_commit or MODEL_VERSION,
        "workspace_dirty_at_generation": _workspace_dirty(),
        "experiment_model_version": MODEL_VERSION,
        "model_configuration_id": MODEL_VERSION,
        "split_definition_id": SPLIT_DEFINITION_ID,
        "active_reference_identifier": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": reference_sha,
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": reference_sha,
        "random_seed": RANDOM_SEED,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
    }
    paths: dict[str, Path] = {}

    json_payloads = {
        "experiment_config.json": _experiment_config(
            result,
            context,
            targeted_test_result,
            full_test_result,
        ),
        "reference_metadata.json": {**context, **result.reference.summary},
        "state_domain_bounds.json": {
            **context,
            "domain_model": DOMAIN_MODEL,
            "bounds": result.domain_bounds,
        },
    }
    for filename, payload in json_payloads.items():
        path = destination / filename
        _write_json(path, payload)
        paths[filename] = path

    csv_outputs = {
        "excitation_metadata.csv": result.excitation_metadata,
        "identification_dataset.csv": result.dataset,
        "domain_coverage.csv": result.domain_coverage,
        "identified_parameters.csv": result.identified_parameters,
        "parameter_errors.csv": result.parameter_errors,
        "prediction_metrics.csv": result.prediction_metrics,
        "generic_vs_identified.csv": result.generic_vs_identified,
        "identifiability_summary.csv": result.identifiability_summary,
        "parameter_correlations.csv": result.parameter_correlations,
        "sensitivity_singular_values.csv": result.singular_values,
        "legacy_identifiability_summary.csv": result.legacy_identifiability_summary,
        "legacy_prediction_metrics.csv": result.legacy_prediction_metrics,
        "legacy_comparison.csv": result.legacy_comparison,
        "held_out_predictions.csv": result.prediction_samples.loc[
            result.prediction_samples["dataset_split"].eq("test")
        ],
    }
    for filename, dataframe in csv_outputs.items():
        path = destination / filename
        _provenance_frame(
            dataframe,
            generated_at_utc=generated_at,
            git_commit=git_commit,
            reference_sha256=reference_sha,
        ).to_csv(path, index=False)
        paths[filename] = path

    for trajectory_id, trajectory in result.trajectories.items():
        path = trajectory_directory / f"{trajectory_id}.csv"
        _provenance_frame(
            trajectory,
            generated_at_utc=generated_at,
            git_commit=git_commit,
            reference_sha256=reference_sha,
        ).to_csv(path, index=False)
        paths[f"trajectories/{trajectory_id}.csv"] = path

    figure_paths: Mapping[str, Path] = {}
    if generate_plots:
        figure_paths = generate_active_reference_local_figures(result, destination)
        paths.update(figure_paths)
    figure_manifest = {
        **context,
        "generating_script": (
            "lower_limb_sim/visualize_reference_local_active_asymmetric.py"
        ),
        "figures": {
            filename: {
                **definition,
                "generated": filename in figure_paths,
                "output_path": str(figure_paths[filename]) if filename in figure_paths else None,
            }
            for filename, definition in FIGURE_DEFINITIONS.items()
        },
    }
    figure_manifest_path = destination / "figure_manifest.json"
    _write_json(figure_manifest_path, figure_manifest)
    paths["figure_manifest.json"] = figure_manifest_path

    reference_sha_after = sha256_file(ACTIVE_REFERENCE_PATH)
    if reference_sha_after != reference_sha:
        raise RuntimeError("active reference changed during the formal experiment.")

    targeted_passed = targeted_test_result.lower().startswith("passed")
    full_passed = full_test_result.lower().startswith("passed")
    acceptance = {
        "current_active_reference_only": True,
        "no_legacy_symmetric_trajectory_in_new_dataset": bool(
            set(result.dataset["trajectory_id"].astype(str)).isdisjoint(
                LOCAL_TRAJECTORY_SPLIT
            )
        ),
        "task_local_coverage_recomputed": not result.domain_coverage.empty,
        "formal_five_parameter_train_only_estimator_used": bool(
            list(PARAMETER_NAMES)
            == list(config.identification_parameter_names)
            and not result.identified_parameters["test_used_for_fit"].astype(bool).any()
        ),
        "identifiability_recomputed": not result.identifiability_summary.empty,
        "genuine_heldout_prediction_present": bool(
            set(HELD_OUT_TRAJECTORY_IDS).issubset(
                set(
                    result.prediction_metrics.loc[
                        result.prediction_metrics["dataset_split"].eq("test"),
                        "trajectory_id",
                    ].astype(str)
                )
            )
        ),
        "parameter_recovery_and_prediction_separate": True,
        "multiple_prespecified_virtual_subjects_present": bool(
            set(result.identified_parameters["subject_id"].astype(str))
            == set(SUBJECT_IDS)
        ),
        "new_directory_without_legacy_overwrite": True,
        "targeted_tests_passed": targeted_passed,
        "full_regression_passed": full_passed,
        "formal_status_explicit_in_run_summary": True,
        "no_real_robot_hardware_safety_code_modified_or_used": True,
    }
    formal_ready = bool(all(acceptance.values()))
    run_summary = {
        **context,
        "status": "FORMAL" if formal_ready else "PRELIMINARY",
        "formal_ready": formal_ready,
        "scientific_scope": (
            "matched clean offline virtual-subject active-asymmetric "
            "reference-local identification"
        ),
        "reference": result.reference.summary,
        "excitation_trajectory_count": len(result.trajectories),
        "training_trajectory_ids": list(TRAINING_TRAJECTORY_IDS),
        "heldout_trajectory_ids": list(HELD_OUT_TRAJECTORY_IDS),
        "coverage": _coverage_summary(result),
        "identifiability": _identifiability_summary(result),
        "parameter_recovery": _truth_parameter_summary(result.parameter_errors),
        "heldout_prediction": _prediction_summary(result),
        "legacy_symmetric_retrospective_comparison": result.legacy_comparison.to_dict(
            orient="records"
        ),
        "failure_boundary": {
            "trajectory_id": "heldout_boundary_speed_plus_10pct",
            "in_domain_percent": float(
                result.domain_coverage.loc[
                    result.domain_coverage["trajectory_id"].eq(
                        "heldout_boundary_speed_plus_10pct"
                    ),
                    "in_domain_percent",
                ].iloc[0]
            ),
            "interpretation": (
                "axis-aligned q/dq/ddq support loss under a 10 percent faster "
                "than nominal held-out profile; matched-model prediction alone "
                "must not be used to waive the domain warning"
            ),
        },
        "acceptance_criteria": acceptance,
        "paper_registry_update_required_after_generation": True,
        "verification": {
            "targeted_tests": targeted_test_result,
            "full_regression": full_test_result,
        },
        "claim_allowed": [
            "task-local equivalent dynamics identification",
            "active-asymmetric reference-local excitation",
            "local numerical identifiability for the adopted model",
            "held-out task-local prediction in matched clean simulation",
            "transfer from legacy symmetric to current active asymmetric task",
        ],
        "claim_not_allowed": [
            "true human parameter recovery",
            "global dynamics identification",
            "clinical or comfort improvement",
            "real robot or human validation",
        ],
    }
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
        if path.is_file()
    }
    run_summary["artifact_sha256"] = artifact_hashes
    run_summary_path = destination / "run_summary.json"
    _write_json(run_summary_path, run_summary)
    paths["run_summary.json"] = run_summary_path
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Formal offline active-asymmetric reference-local five-parameter "
            "identification validation."
        )
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--samples-per-segment", type=int, default=201)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--targeted-test-result", default="not_recorded")
    parser.add_argument("--full-test-result", default="not_recorded")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_active_reference_local_identification(
        samples_per_segment=args.samples_per_segment
    )
    paths = save_formal_result(
        result,
        args.output_directory,
        generate_plots=not args.no_plots,
        targeted_test_result=args.targeted_test_result,
        full_test_result=args.full_test_result,
    )
    summary = json.loads(paths["run_summary.json"].read_text(encoding="utf-8"))
    print(f"status={summary['status']}")
    print(f"active_reference={ACTIVE_REFERENCE_ID}")
    print(f"output_directory={args.output_directory}")
    print(
        "condition_number_range="
        f"{summary['identifiability']['condition_number_minimum']:.6g}.."
        f"{summary['identifiability']['condition_number_maximum']:.6g}"
    )
    print(
        "heldout_max_rmse_nm="
        f"{summary['heldout_prediction']['heldout_maximum_combined_torque_rmse_nm']:.6g}"
    )
    print(f"artifact_count={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "build_argument_parser",
    "main",
    "save_formal_result",
]
