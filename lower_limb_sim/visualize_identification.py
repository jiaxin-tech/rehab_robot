"""第四阶段参数辨识结果的静态可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .identifiability_analysis import IdentifiabilityResult
from .parameter_estimator import PARAMETER_NAMES

DISPLAY_NAMES = {
    "mass_scale": "mass scale",
    "k_hip_nm_per_rad": "K hip",
    "k_knee_nm_per_rad": "K knee",
    "b_hip_nm_s_per_rad": "B hip",
    "b_knee_nm_s_per_rad": "B knee",
}
PARAMETER_UNITS = {
    "mass_scale": "ratio",
    "k_hip_nm_per_rad": "N·m/rad",
    "k_knee_nm_per_rad": "N·m/rad",
    "b_hip_nm_s_per_rad": "N·m·s/rad",
    "b_knee_nm_s_per_rad": "N·m·s/rad",
}


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _true_vs_estimated(
    parameter_table: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for axis, parameter in zip(axes, PARAMETER_NAMES):
        row = parameter_table.loc[parameter_table["parameter"].eq(parameter)].iloc[0]
        axis.bar(
            ["true", "estimated"],
            [row["true_value"], row["estimated_value"]],
            color=["#4C78A8", "#F58518"],
        )
        axis.set_title(DISPLAY_NAMES[parameter])
        axis.set_ylabel(PARAMETER_UNITS[parameter])
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle(f"True vs estimated parameters — {identity}")
    return _save(figure, path)


def _predicted_vs_measured(
    predictions: pd.DataFrame,
    joint: str,
    identity: str,
    path: Path,
) -> Path:
    measured = predictions[f"tau_measured_{joint}_nm"].to_numpy(dtype=float)
    predicted = predictions[f"tau_predicted_{joint}_nm"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    colors = {"train": "#4C78A8", "validation": "#54A24B", "test": "#E45756"}
    for split, group in predictions.groupby("dataset_split", sort=False):
        axis.scatter(
            group[f"tau_measured_{joint}_nm"],
            group[f"tau_predicted_{joint}_nm"],
            s=6,
            alpha=0.35,
            label=split,
            color=colors.get(str(split)),
        )
    lower = float(min(np.min(measured), np.min(predicted)))
    upper = float(max(np.max(measured), np.max(predicted)))
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="ideal")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"measured {joint} torque (N·m)")
    axis.set_ylabel(f"predicted {joint} torque (N·m)")
    axis.set_title(f"Predicted vs measured {joint} torque — {identity}")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(figure, path)


def _residuals_vs_time(
    predictions: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    selected = predictions.loc[predictions["dataset_split"].eq("test")].copy()
    selected["display_sample"] = np.arange(len(selected))
    for axis, joint in zip(axes, ("hip", "knee")):
        axis.plot(
            selected["display_sample"],
            selected[f"torque_residual_{joint}_nm"],
            linewidth=0.9,
            color="#E45756" if joint == "hip" else "#4C78A8",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(f"{joint} residual\n(N·m)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("concatenated test sample (trajectory-local time preserved in CSV)")
    figure.suptitle(f"Test torque residuals — {identity}")
    return _save(figure, path)


def _relative_errors(
    parameter_table: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        [DISPLAY_NAMES[name] for name in parameter_table["parameter"]],
        parameter_table["relative_error_percent"],
        color="#F58518",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("relative error (%)")
    axis.set_title(f"Parameter relative errors — {identity}")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, path)


def _correlation_heatmap(
    complete_result: IdentifiabilityResult,
    identity: str,
    path: Path,
) -> Path:
    correlation = np.asarray(complete_result.parameter_correlation, dtype=float)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [DISPLAY_NAMES[name] for name in PARAMETER_NAMES]
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(correlation[row, column]) > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="local parameter correlation")
    axis.set_title(f"Parameter correlation — {identity}")
    return _save(figure, path)


def _singular_values(
    results: Mapping[str, IdentifiabilityResult],
    identity: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for name, result in results.items():
        if name == "C_without_extreme_geometry":
            continue
        axis.semilogy(
            np.arange(1, len(result.singular_values) + 1),
            result.singular_values,
            marker="o",
            label=name,
        )
    axis.set_xlabel("singular value index")
    axis.set_ylabel("scaled sensitivity singular value")
    axis.set_title(f"Excitation-set sensitivity singular values — {identity}")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    return _save(figure, path)


def _clean_vs_noise(
    aggregate_parameter_table: pd.DataFrame,
    subject_id: str,
    identity: str,
    path: Path,
) -> Path:
    selected = aggregate_parameter_table.loc[
        aggregate_parameter_table["subject_id"].eq(subject_id)
    ]
    comparison = (
        selected.groupby("noise_scenario", sort=False)["relative_error_percent"]
        .mean()
        .sort_values()
    )
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    axis.bar(comparison.index, comparison.values, color="#72B7B2")
    axis.set_ylabel("mean absolute parameter relative error (%)")
    axis.set_title(f"Clean vs noise identification — {identity}")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, path)


def generate_identification_visualizations(
    parameter_table: pd.DataFrame,
    predictions: pd.DataFrame,
    identifiability_results: Mapping[str, IdentifiabilityResult],
    aggregate_parameter_table: pd.DataFrame,
    subject_id: str,
    noise_scenario: str,
    output_dir: str | Path,
) -> list[Path]:
    """生成题设要求的八张辨识图。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity = f"{subject_id}/{noise_scenario}"
    complete = identifiability_results["C_all_families_all_speeds"]
    return [
        _true_vs_estimated(
            parameter_table,
            identity,
            destination / "true_vs_estimated_parameters.png",
        ),
        _predicted_vs_measured(
            predictions,
            "hip",
            identity,
            destination / "predicted_vs_measured_hip_torque.png",
        ),
        _predicted_vs_measured(
            predictions,
            "knee",
            identity,
            destination / "predicted_vs_measured_knee_torque.png",
        ),
        _residuals_vs_time(
            predictions,
            identity,
            destination / "torque_residuals_vs_time.png",
        ),
        _relative_errors(
            parameter_table,
            identity,
            destination / "parameter_relative_errors.png",
        ),
        _correlation_heatmap(
            complete,
            identity,
            destination / "parameter_correlation_heatmap.png",
        ),
        _singular_values(
            identifiability_results,
            identity,
            destination / "sensitivity_singular_values.png",
        ),
        _clean_vs_noise(
            aggregate_parameter_table,
            subject_id,
            identity,
            destination / "clean_vs_noise_comparison.png",
        ),
    ]
