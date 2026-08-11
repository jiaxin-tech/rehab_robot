from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lower_limb_sim.mismatch_metrics import (
    RESIDUAL_FEATURE_COLUMNS,
    compute_residual_diagnostics,
    compute_residual_feature_correlations,
    compute_trajectory_metrics,
    evaluate_model_mismatch_predictions,
)
from lower_limb_sim.visualize_model_mismatch import (
    REQUIRED_FIGURE_FILENAMES,
    RESIDUAL_FEATURE_FIGURE_FILENAMES,
    generate_model_mismatch_visualizations,
)


def _prediction_table() -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for index, (family, split) in enumerate(
        (
            ("phase_shift_small", "interpolation_test"),
            ("boundary_near", "boundary_test"),
        )
    ):
        time = np.linspace(0.0, 2.0, 41)
        phase = np.pi * time / 2.0
        q_hip = 0.35 + 0.25 * np.sin(phase)
        q_knee = 0.55 + 0.35 * np.sin(phase + 0.15 * index)
        dq_hip = 0.25 * (np.pi / 2.0) * np.cos(phase)
        dq_knee = 0.35 * (np.pi / 2.0) * np.cos(phase + 0.15 * index)
        ddq_hip = -0.25 * (np.pi / 2.0) ** 2 * np.sin(phase)
        ddq_knee = -0.35 * (np.pi / 2.0) ** 2 * np.sin(
            phase + 0.15 * index
        )
        tau_hip = 12.0 + 8.0 * q_hip + 1.4 * dq_hip + 0.7 * ddq_hip
        tau_knee = 7.0 + 6.0 * q_knee + 0.9 * dq_knee + 0.5 * ddq_knee
        tables.append(
            pd.DataFrame(
                {
                    "subject_id": "baseline",
                    "scenario_name": "nonlinear_stiffness_mild",
                    "trajectory_id": "software_validation_trajectory",
                    "trajectory_family": family,
                    "dataset_split": split,
                    "phase": "flexion",
                    "time_s": time,
                    "trajectory_sample_index": np.arange(len(time)),
                    "q_hip_rad": q_hip,
                    "q_knee_rad": q_knee,
                    "dq_hip_rad_s": dq_hip,
                    "dq_knee_rad_s": dq_knee,
                    "ddq_hip_rad_s2": ddq_hip,
                    "ddq_knee_rad_s2": ddq_knee,
                    "tau_true_hip_nm": tau_hip,
                    "tau_true_knee_nm": tau_knee,
                    "tau_generic_hip_nm": tau_hip - 2.0 - 0.3 * q_hip,
                    "tau_generic_knee_nm": tau_knee + 1.5 + 0.2 * q_knee,
                    "tau_identified_hip_nm": tau_hip,
                    "tau_identified_knee_nm": tau_knee,
                    "sample_valid": True,
                    "generic_prediction_valid": True,
                    "identified_prediction_valid": True,
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def test_all_requested_metrics_and_generic_comparison() -> None:
    bundle = evaluate_model_mismatch_predictions(_prediction_table())
    required_metrics = {
        "hip_torque_rmse_nm",
        "knee_torque_rmse_nm",
        "combined_torque_rmse_nm",
        "hip_nrmse_percent",
        "knee_nrmse_percent",
        "combined_nrmse_percent",
        "hip_mae_nm",
        "knee_mae_nm",
        "hip_peak_error_percent",
        "knee_peak_error_percent",
        "hip_correlation",
        "knee_correlation",
        "hip_vaf",
        "knee_vaf",
        "endpoint_force_rmse_n",
        "endpoint_force_peak_error_percent",
        "residual_mean_hip_nm",
        "residual_mean_knee_nm",
        "residual_angle_correlation",
        "residual_velocity_correlation",
    }
    assert required_metrics.issubset(bundle.trajectory_metrics.columns)
    identified = bundle.trajectory_metrics.loc[
        bundle.trajectory_metrics["prediction_model"].eq("identified")
    ]
    assert np.allclose(identified["combined_torque_rmse_nm"], 0.0)
    assert np.allclose(identified["endpoint_force_rmse_n"], 0.0)
    assert np.allclose(identified["hip_correlation"], 1.0)
    assert np.allclose(identified["knee_vaf"], 100.0)
    assert (bundle.generic_vs_identified["improvement_percent"] > 99.9).all()


def test_fixed_nrmse_formula_and_small_range_flag() -> None:
    predictions = _prediction_table().iloc[:5].copy()
    predictions["tau_true_hip_nm"] = np.arange(5.0)
    predictions["tau_identified_hip_nm"] = predictions["tau_true_hip_nm"] - 1.0
    predictions["tau_true_knee_nm"] = 3.0
    predictions["tau_identified_knee_nm"] = 2.0
    metrics = compute_trajectory_metrics(predictions, "identified").iloc[0]
    assert np.isclose(metrics["hip_torque_rmse_nm"], 1.0)
    assert np.isclose(metrics["hip_nrmse_percent"], 25.0)
    assert np.isnan(metrics["knee_nrmse_percent"])
    assert bool(metrics["knee_nrmse_unreliable_small_range"])
    assert bool(metrics["nrmse_unreliable_small_range"])


def test_residual_correlations_keep_constant_features_explicit() -> None:
    predictions = _prediction_table().iloc[:20].copy()
    predictions["ddq_hip_rad_s2"] = 0.0
    correlations = compute_residual_feature_correlations(
        predictions,
        "identified",
    )
    assert set(RESIDUAL_FEATURE_COLUMNS).issubset(set(correlations["feature"]))
    constant = correlations.loc[
        correlations["feature"].eq("ddq_hip_rad_s2")
    ]
    assert not constant["correlation_valid"].any()
    assert constant["correlation"].isna().all()
    assert set(constant["invalid_reason"]) == {"constant_residual_or_feature"}


def test_nonlinear_residual_diagnostic_is_only_a_flag() -> None:
    predictions = _prediction_table().iloc[:41].copy()
    q_hip = predictions["q_hip_rad"].to_numpy(dtype=float)
    nonlinear = q_hip**3
    design = np.column_stack((np.ones(len(q_hip)), q_hip))
    nonlinear -= design @ np.linalg.lstsq(design, nonlinear, rcond=None)[0]
    predictions["tau_identified_hip_nm"] = (
        predictions["tau_true_hip_nm"] - 5.0 * nonlinear
    )
    diagnostics = compute_residual_diagnostics(
        predictions,
        "identified",
        correlation_threshold=0.8,
    ).iloc[0]
    assert bool(diagnostics["possible_nonlinear_stiffness_mismatch"])
    assert bool(diagnostics["diagnostic_only_not_mechanism_proof"])
    assert "not prove" in diagnostics["diagnostic_note"]


def test_visualizer_generates_required_and_residual_plots(tmp_path: Path) -> None:
    predictions = _prediction_table()
    predictions["ddq_hip_rad_s2"] = 0.0
    bundle = evaluate_model_mismatch_predictions(predictions)
    parameters = pd.DataFrame(
        {
            "parameter": (
                "mass_scale",
                "k_hip_nm_per_rad",
                "k_knee_nm_per_rad",
                "b_hip_nm_s_per_rad",
                "b_knee_nm_s_per_rad",
            ),
            "generic_value": (1.0, 10.0, 10.0, 1.0, 1.0),
            "identified_value": (1.05, 15.0, 13.0, 1.4, 1.2),
        }
    )
    paths = generate_model_mismatch_visualizations(
        predictions,
        bundle.trajectory_metrics,
        bundle.generic_vs_identified,
        bundle.residual_feature_correlations,
        parameters,
        "baseline",
        "nonlinear_stiffness_mild",
        tmp_path,
    )
    expected_names = set(REQUIRED_FIGURE_FILENAMES) | set(
        RESIDUAL_FEATURE_FIGURE_FILENAMES.values()
    )
    assert {path.name for path in paths} == expected_names
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)

