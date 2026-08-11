"""Matplotlib reports for Stage 4.5D virtual geometry-error experiments."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lower_limb_sim_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry_error_scenarios import (
    INDEPENDENT_JOINT_MEASUREMENT,
    ORACLE_TRUE_JOINT_STATE,
    TCP_INVERSE_KINEMATICS,
)


MODE_LABELS = {
    ORACLE_TRUE_JOINT_STATE: "Oracle (upper bound)",
    TCP_INVERSE_KINEMATICS: "TCP inverse kinematics",
    INDEPENDENT_JOINT_MEASUREMENT: "Independent joint measurement",
}
MODE_COLORS = {
    ORACLE_TRUE_JOINT_STATE: "#4C78A8",
    TCP_INVERSE_KINEMATICS: "#F58518",
    INDEPENDENT_JOINT_MEASUREMENT: "#54A24B",
}
TRUE_COLOR = "#222222"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _representative_trajectory(dataframe: pd.DataFrame) -> pd.DataFrame:
    preferred = dataframe.loc[dataframe["dataset_split"].eq("interpolation_test")]
    if preferred.empty:
        preferred = dataframe
    identity_candidates = (
        "trajectory_id",
        "trajectory_name",
        "trajectory_family",
    )
    for column in identity_candidates:
        if column in preferred and preferred[column].notna().any():
            identity = preferred.loc[preferred[column].notna(), column].iloc[0]
            return preferred.loc[preferred[column].eq(identity)].copy()
    return preferred.copy()


def _mode_groups(dataframe: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for mode, group in dataframe.groupby("observation_mode", sort=False):
        yield str(mode), group.sort_values("time_s")


def _save(fig: plt.Figure, destination: Path, filename: str) -> Path:
    path = destination / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _line_comparison(
    data: pd.DataFrame,
    *,
    true_columns: tuple[str, str],
    estimated_columns: tuple[str, str],
    y_labels: tuple[str, str],
    title: str,
    convert: float = 1.0,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    first_mode = next(iter(_mode_groups(data)))[1]
    time = first_mode["time_s"].to_numpy(dtype=float)
    for axis, truth_column, estimated_column, y_label in zip(
        axes, true_columns, estimated_columns, y_labels
    ):
        axis.plot(
            time,
            convert * first_mode[truth_column].to_numpy(dtype=float),
            color=TRUE_COLOR,
            linewidth=2.0,
            label="Virtual truth (evaluation only)",
        )
        for mode, group in _mode_groups(data):
            axis.plot(
                group["time_s"].to_numpy(dtype=float),
                convert * group[estimated_column].to_numpy(dtype=float),
                color=MODE_COLORS.get(mode),
                linewidth=1.2,
                alpha=0.9,
                label=MODE_LABELS.get(mode, mode),
            )
        axis.set_ylabel(y_label)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    return fig


def generate_geometry_error_visualizations(
    *,
    predictions: pd.DataFrame,
    kinematic_metrics: pd.DataFrame,
    identification_metrics: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    observation_mode_comparison: pd.DataFrame,
    sensitivity_ranking: pd.DataFrame,
    subject_id: str,
    scenario_name: str,
    output_directory: str | Path,
) -> list[Path]:
    """Create the eleven required, unit-labelled PNG reports."""

    _configure_style()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    representative = _representative_trajectory(predictions)
    prefix = f"{subject_id} / {scenario_name}"
    paths: list[Path] = []

    fig = _line_comparison(
        representative,
        true_columns=("q_hip_true_rad", "q_knee_true_rad"),
        estimated_columns=("q_hip_est_rad", "q_knee_est_rad"),
        y_labels=("Hip angle (deg)", "Knee angle (deg)"),
        title=f"{prefix}: true vs reconstructed joint angles",
        convert=180.0 / np.pi,
    )
    paths.append(_save(fig, destination, "true_vs_reconstructed_angles.png"))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis, estimated, truth, label in (
        (axes[0], "q_hip_est_rad", "q_hip_true_rad", "Hip angle error (deg)"),
        (axes[1], "q_knee_est_rad", "q_knee_true_rad", "Knee angle error (deg)"),
    ):
        for mode, group in _mode_groups(representative):
            error = np.rad2deg(
                group[estimated].to_numpy(float) - group[truth].to_numpy(float)
            )
            axis.plot(
                group["time_s"],
                error,
                color=MODE_COLORS.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )
        axis.axhline(0.0, color=TRUE_COLOR, linewidth=0.8)
        axis.set_ylabel(label)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{prefix}: angle reconstruction error")
    paths.append(_save(fig, destination, "angle_error_vs_time.png"))

    fig = _line_comparison(
        representative,
        true_columns=("dq_hip_true_rad_s", "dq_knee_true_rad_s"),
        estimated_columns=("dq_hip_est_rad_s", "dq_knee_est_rad_s"),
        y_labels=("Hip velocity (rad/s)", "Knee velocity (rad/s)"),
        title=f"{prefix}: true vs estimated velocity",
    )
    paths.append(_save(fig, destination, "true_vs_estimated_velocity.png"))

    fig = _line_comparison(
        representative,
        true_columns=("ddq_hip_true_rad_s2", "ddq_knee_true_rad_s2"),
        estimated_columns=("ddq_hip_est_rad_s2", "ddq_knee_est_rad_s2"),
        y_labels=("Hip acceleration (rad/s²)", "Knee acceleration (rad/s²)"),
        title=f"{prefix}: true vs estimated acceleration",
    )
    paths.append(_save(fig, destination, "true_vs_estimated_acceleration.png"))

    fig, axis = plt.subplots(figsize=(7.5, 6))
    first_mode = next(iter(_mode_groups(representative)))[1]
    axis.plot(
        first_mode["x_pull_true_m"],
        first_mode["z_pull_true_m"],
        color=TRUE_COLOR,
        linewidth=2.2,
        label="True pull point",
    )
    for mode, group in _mode_groups(representative):
        axis.plot(
            group["x_pull_assumed_reconstructed_m"],
            group["z_pull_assumed_reconstructed_m"],
            color=MODE_COLORS.get(mode),
            linewidth=1.2,
            label=MODE_LABELS.get(mode, mode),
        )
    axis.set_xlabel("Bed direction x (m)")
    axis.set_ylabel("Vertical z (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(fontsize=8)
    axis.set_title(f"{prefix}: true and assumed pull-point paths")
    paths.append(_save(fig, destination, "true_vs_assumed_pull_point.png"))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for mode, group in _mode_groups(representative):
        axes[0].plot(
            group["time_s"],
            group["jacobian_frobenius_error"],
            color=MODE_COLORS.get(mode),
            label=MODE_LABELS.get(mode, mode),
        )
        axes[1].plot(
            group["time_s"],
            group["jacobian_condition_error"],
            color=MODE_COLORS.get(mode),
        )
    axes[0].set_ylabel("Jacobian Frobenius error")
    axes[1].set_ylabel("Condition-number error")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{prefix}: Jacobian error")
    paths.append(_save(fig, destination, "jacobian_error_vs_time.png"))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis, joint in zip(axes, ("hip", "knee")):
        for mode, group in _mode_groups(representative):
            error = (
                group[f"tau_measured_est_{joint}_nm"].to_numpy(float)
                - group[f"tau_measured_true_{joint}_nm"].to_numpy(float)
            )
            axis.plot(
                group["time_s"],
                error,
                color=MODE_COLORS.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )
        axis.axhline(0.0, color=TRUE_COLOR, linewidth=0.8)
        axis.set_ylabel(f"{joint.title()} torque error (N·m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{prefix}: reconstructed observation torque error")
    paths.append(_save(fig, destination, "torque_observation_error.png"))

    parameter_columns = (
        ("mass_scale_error_percent", "Mass scale"),
        ("k_hip_error_percent", "K hip"),
        ("k_knee_error_percent", "K knee"),
        ("b_hip_error_percent", "B hip"),
        ("b_knee_error_percent", "B knee"),
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(parameter_columns), dtype=float)
    modes = list(identification_metrics["observation_mode"].astype(str).unique())
    width = 0.8 / max(len(modes), 1)
    for index, mode in enumerate(modes):
        row = identification_metrics.loc[
            identification_metrics["observation_mode"].eq(mode)
        ].iloc[0]
        axis.bar(
            x + (index - (len(modes) - 1) / 2.0) * width,
            [float(row[column]) for column, _ in parameter_columns],
            width=width,
            color=MODE_COLORS.get(mode),
            label=MODE_LABELS.get(mode, mode),
        )
    axis.set_xticks(x, [label for _, label in parameter_columns])
    axis.set_ylabel("Absolute relative error (%)")
    axis.set_title(f"{prefix}: five-parameter estimation error")
    axis.legend(fontsize=8)
    paths.append(_save(fig, destination, "parameter_error_by_scenario.png"))

    identified = prediction_metrics.loc[
        prediction_metrics["prediction_model"].eq("identified")
    ].copy()
    splits = list(identified["dataset_split"].astype(str).unique())
    fig, axis = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(splits), dtype=float)
    modes = list(identified["observation_mode"].astype(str).unique())
    width = 0.8 / max(len(modes), 1)
    for index, mode in enumerate(modes):
        mode_rows = identified.loc[identified["observation_mode"].eq(mode)].set_index(
            "dataset_split"
        )
        values = [
            float(mode_rows.loc[split, "combined_torque_nrmse_percent"])
            for split in splits
        ]
        axis.bar(
            x + (index - (len(modes) - 1) / 2.0) * width,
            values,
            width=width,
            color=MODE_COLORS.get(mode),
            label=MODE_LABELS.get(mode, mode),
        )
    axis.set_xticks(x, [split.replace("_", "\n") for split in splits])
    axis.set_ylabel("Combined torque NRMSE (%)")
    axis.set_title(f"{prefix}: identified-model prediction by fixed split")
    axis.legend(fontsize=8)
    paths.append(_save(fig, destination, "prediction_nrmse_by_scenario.png"))

    comparison = observation_mode_comparison.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    mode_names = comparison["observation_mode"].astype(str).tolist()
    labels = [MODE_LABELS.get(mode, mode) for mode in mode_names]
    colors = [MODE_COLORS.get(mode) for mode in mode_names]
    axes[0].bar(labels, comparison["joint_angle_rmse_deg"], color=colors)
    axes[0].set_ylabel("Joint-angle RMSE (deg)")
    axes[1].bar(
        labels,
        comparison["combined_torque_nrmse_percent"],
        color=colors,
    )
    axes[1].set_ylabel("Interpolation torque NRMSE (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    fig.suptitle(f"{prefix}: observation-mode comparison")
    paths.append(_save(fig, destination, "observation_mode_comparison.png"))

    ranking = sensitivity_ranking.loc[
        sensitivity_ranking["ranking_metric"].eq(
            "interpolation_combined_nrmse_percent"
        )
    ].sort_values("mean", ascending=True)
    fig, axis = plt.subplots(figsize=(9.5, max(4.5, 0.38 * len(ranking) + 2)))
    if ranking.empty:
        axis.text(0.5, 0.5, "No accumulated sensitivity rows", ha="center")
        axis.set_axis_off()
    else:
        positions = np.arange(len(ranking))
        error = ranking["std"].fillna(0.0).to_numpy(dtype=float)
        category_labels = ranking["error_source"].astype(str)
        if "observation_mode" in ranking:
            category_labels = category_labels.str.cat(
                ranking["observation_mode"].astype(str).map(
                    lambda value: MODE_LABELS.get(value, value)
                ),
                sep=" / ",
            )
        axis.barh(
            positions,
            ranking["mean"].to_numpy(dtype=float),
            xerr=error,
            color="#4C78A8",
            alpha=0.85,
        )
        axis.set_yticks(positions, category_labels)
        axis.set_xlabel("Mean interpolation torque NRMSE (%) ± 1 SD")
        axis.set_ylabel("Geometry / observation error source")
    axis.set_title("Accumulated Stage 4.5D geometry sensitivity ranking")
    paths.append(_save(fig, destination, "geometry_sensitivity_ranking.png"))

    return paths


def generate_geometry_error_summary_visualizations(
    summary_directory: str | Path,
) -> list[Path]:
    """Refresh four cross-scenario figures from accumulated CSV summaries."""

    _configure_style()
    destination = Path(summary_directory)
    required = {
        "parameter": destination / "parameter_errors.csv",
        "prediction": destination / "prediction_metrics.csv",
        "comparison": destination / "observation_mode_comparison.csv",
        "ranking": destination / "geometry_sensitivity_ranking.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing geometry summary tables: {missing}")
    parameter = pd.read_csv(required["parameter"])
    prediction = pd.read_csv(required["prediction"])
    comparison = pd.read_csv(required["comparison"])
    ranking = pd.read_csv(required["ranking"])
    paths: list[Path] = []

    parameter_view = parameter.loc[
        parameter["observation_mode"].eq(TCP_INVERSE_KINEMATICS)
    ].sort_values("maximum_parameter_error_percent", ascending=False).head(20)
    fig, axis = plt.subplots(
        figsize=(10, max(5, 0.32 * len(parameter_view) + 2))
    )
    axis.barh(
        np.arange(len(parameter_view)),
        parameter_view["maximum_parameter_error_percent"].to_numpy(float),
        color=MODE_COLORS[TCP_INVERSE_KINEMATICS],
    )
    axis.set_yticks(
        np.arange(len(parameter_view)),
        parameter_view["scenario_name"].astype(str),
    )
    axis.invert_yaxis()
    axis.set_xlabel("Maximum five-parameter error (%)")
    axis.set_ylabel("Scenario (TCP inverse kinematics)")
    axis.set_title("Stage 4.5D cross-scenario parameter error")
    paths.append(_save(fig, destination, "parameter_error_by_scenario.png"))

    prediction_view = prediction.loc[
        prediction["observation_mode"].eq(TCP_INVERSE_KINEMATICS)
        & prediction["dataset_split"].eq("interpolation_test")
        & prediction["prediction_model"].eq("identified")
    ].sort_values("combined_torque_nrmse_percent", ascending=False).head(20)
    fig, axis = plt.subplots(
        figsize=(10, max(5, 0.32 * len(prediction_view) + 2))
    )
    axis.barh(
        np.arange(len(prediction_view)),
        prediction_view["combined_torque_nrmse_percent"].to_numpy(float),
        color=MODE_COLORS[TCP_INVERSE_KINEMATICS],
    )
    axis.set_yticks(
        np.arange(len(prediction_view)),
        prediction_view["scenario_name"].astype(str),
    )
    axis.invert_yaxis()
    axis.set_xlabel("Interpolation combined torque NRMSE (%)")
    axis.set_ylabel("Scenario (identified TCP model)")
    axis.set_title("Stage 4.5D unseen-trajectory prediction sensitivity")
    paths.append(_save(fig, destination, "prediction_nrmse_by_scenario.png"))

    aggregate = comparison.groupby("observation_mode", sort=False).agg(
        joint_angle_rmse_deg=("joint_angle_rmse_deg", "mean"),
        combined_torque_nrmse_percent=(
            "combined_torque_nrmse_percent",
            "mean",
        ),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = [
        MODE_LABELS.get(mode, mode)
        for mode in aggregate["observation_mode"].astype(str)
    ]
    colors = [
        MODE_COLORS.get(mode)
        for mode in aggregate["observation_mode"].astype(str)
    ]
    axes[0].bar(labels, aggregate["joint_angle_rmse_deg"], color=colors)
    axes[0].set_ylabel("Mean interpolation joint-angle RMSE (deg)")
    axes[1].bar(
        labels,
        aggregate["combined_torque_nrmse_percent"],
        color=colors,
    )
    axes[1].set_ylabel("Mean interpolation torque NRMSE (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    fig.suptitle("Stage 4.5D accumulated observation-mode comparison")
    paths.append(_save(fig, destination, "observation_mode_comparison.png"))

    ranking_view = ranking.loc[
        ranking["ranking_metric"].eq(
            "interpolation_combined_nrmse_percent"
        )
    ].sort_values("mean", ascending=False).head(20).iloc[::-1]
    labels = ranking_view["error_source"].astype(str)
    if "observation_mode" in ranking_view:
        labels = labels.str.cat(
            ranking_view["observation_mode"].astype(str).map(
                lambda value: MODE_LABELS.get(value, value)
            ),
            sep=" / ",
        )
    fig, axis = plt.subplots(
        figsize=(10, max(5, 0.34 * len(ranking_view) + 2))
    )
    positions = np.arange(len(ranking_view))
    axis.barh(
        positions,
        ranking_view["mean"].to_numpy(float),
        xerr=ranking_view["std"].fillna(0.0).to_numpy(float),
        color="#4C78A8",
        alpha=0.85,
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Mean interpolation torque NRMSE (%) ± 1 SD")
    axis.set_ylabel("Error source / observation mode")
    axis.set_title("Stage 4.5D accumulated geometry sensitivity ranking")
    paths.append(_save(fig, destination, "geometry_sensitivity_ranking.png"))
    return paths


__all__ = [
    "generate_geometry_error_summary_visualizations",
    "generate_geometry_error_visualizations",
]
