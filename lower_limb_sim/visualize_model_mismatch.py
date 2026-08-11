"""Static visual review for stage 4.5C model-mismatch experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mismatch_metrics import resolve_mismatch_prediction_columns


MODEL_COLORS = {
    "generic": "#9D9DA1",
    "identified": "#F58518",
    "true": "#4C78A8",
}
SPLIT_COLORS = {
    "train": "#4C78A8",
    "validation": "#54A24B",
    "interpolation_test": "#F58518",
    "boundary_test": "#E45756",
    "outside_domain_test": "#B279A2",
}
SPLIT_MARKERS = {
    "train": "o",
    "validation": "s",
    "interpolation_test": "^",
    "boundary_test": "D",
    "outside_domain_test": "X",
}
PARAMETER_DISPLAY_NAMES = {
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
REQUIRED_FIGURE_FILENAMES = (
    "true_vs_predicted_torque.png",
    "generic_vs_identified_rmse.png",
    "error_by_trajectory.png",
    "nrmse_by_split.png",
    "peak_error_by_split.png",
    "residual_over_trajectory.png",
    "residual_feature_correlations.png",
    "parameter_shift.png",
)
RESIDUAL_FEATURE_FIGURE_FILENAMES = {
    "q_hip_rad": "residual_vs_q_hip.png",
    "q_knee_rad": "residual_vs_q_knee.png",
    "dq_hip_rad_s": "residual_vs_dq_hip.png",
    "dq_knee_rad_s": "residual_vs_dq_knee.png",
}
FEATURE_AXIS_LABELS = {
    "q_hip_rad": "hip angle (rad)",
    "q_knee_rad": "knee angle (rad)",
    "dq_hip_rad_s": "hip angular velocity (rad/s)",
    "dq_knee_rad_s": "knee angular velocity (rad/s)",
}


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _empty_axis(axis: plt.Axes, message: str = "No finite data available") -> None:
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#666666",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _identity(subject_id: str, scenario_name: str) -> str:
    return f"subject={subject_id} | scenario={scenario_name}"


def _split_column(dataframe: pd.DataFrame) -> str | None:
    return next(
        (column for column in ("dataset_split", "split") if column in dataframe),
        None,
    )


def _trajectory_label(row: pd.Series | Mapping[str, object]) -> str:
    primary = ""
    for column in ("trajectory_family", "trajectory_name", "trajectory_id"):
        value = row.get(column) if hasattr(row, "get") else None
        if value is None or pd.isna(value):
            continue
        text = str(value)
        if text:
            primary = text
            break
    if not primary:
        primary = "trajectory"
    speed = row.get("speed_profile") if hasattr(row, "get") else None
    if speed is not None and not pd.isna(speed) and str(speed):
        speed_text = str(speed)
        if speed_text != primary:
            return f"{primary}/{speed_text}"
    return primary


def _short_split(value: object) -> str:
    return {
        "train": "train",
        "validation": "validation",
        "interpolation_test": "interpolation",
        "boundary_test": "boundary",
        "outside_domain_test": "outside-domain",
    }.get(str(value), str(value))


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[finite], y_values[finite]


def _true_vs_predicted_torque(
    predictions: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    try:
        generic = resolve_mismatch_prediction_columns(predictions, "generic")
        identified = resolve_mismatch_prediction_columns(predictions, "identified")
    except ValueError:
        generic = {}
        identified = {}
    split_column = _split_column(predictions)
    splits: Sequence[object]
    if split_column is None or predictions.empty:
        splits = ("all",)
    else:
        splits = tuple(predictions[split_column].drop_duplicates())

    for axis, joint in zip(axes, ("hip", "knee")):
        plotted_values: list[np.ndarray] = []
        for model, columns in (("generic", generic), ("identified", identified)):
            true_column = columns.get(f"true_{joint}")
            predicted_column = columns.get(f"predicted_{joint}")
            if true_column is None or predicted_column is None:
                continue
            for split in splits:
                if split_column is None or split == "all":
                    group = predictions
                else:
                    group = predictions.loc[predictions[split_column].eq(split)]
                true, predicted = _finite_pair(
                    group[true_column],
                    group[predicted_column],
                )
                if true.size == 0:
                    continue
                plotted_values.extend((true, predicted))
                axis.scatter(
                    true,
                    predicted,
                    s=10,
                    alpha=0.35,
                    color=MODEL_COLORS[model],
                    marker=SPLIT_MARKERS.get(str(split), "o"),
                    label=f"{model} | {split}",
                )
        if plotted_values:
            combined = np.concatenate(plotted_values)
            lower = float(np.min(combined))
            upper = float(np.max(combined))
            if np.isclose(lower, upper):
                padding = max(abs(lower) * 0.05, 1.0)
                lower -= padding
                upper += padding
            axis.plot(
                [lower, upper],
                [lower, upper],
                "k--",
                linewidth=1.0,
                label="ideal",
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(f"true {joint} torque (N·m)")
            axis.set_ylabel(f"predicted {joint} torque (N·m)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=6, ncol=2)
        else:
            _empty_axis(axis)
        axis.set_title(f"{joint.capitalize()} joint")
    figure.suptitle(f"True vs predicted torque | {identity}")
    return _save(figure, path)


def _generic_vs_identified_rmse(
    comparison: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(11.0, 5.0))
    required = {"generic_rmse", "identified_rmse"}
    if comparison.empty or not required.issubset(comparison.columns):
        _empty_axis(axis)
    else:
        table = comparison.copy().reset_index(drop=True)
        labels = [_trajectory_label(row) for _, row in table.iterrows()]
        split_column = _split_column(table)
        if split_column is not None:
            labels = [
                f"{label}\n[{_short_split(split)}]"
                for label, split in zip(labels, table[split_column].astype(str))
            ]
        x = np.arange(len(table), dtype=float)
        width = 0.38
        axis.bar(
            x - width / 2,
            table["generic_rmse"],
            width,
            label="generic baseline",
            color=MODEL_COLORS["generic"],
        )
        axis.bar(
            x + width / 2,
            table["identified_rmse"],
            width,
            label="identified equivalent",
            color=MODEL_COLORS["identified"],
        )
        axis.set_xticks(x, labels=labels, rotation=35, ha="right")
        axis.set_ylabel("combined torque RMSE (N·m)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")
    axis.set_title(f"Generic vs identified prediction RMSE | {identity}")
    return _save(figure, path)


def _identified_metric_rows(trajectory_metrics: pd.DataFrame) -> pd.DataFrame:
    table = trajectory_metrics.copy()
    if "metric_scope" in table and table["metric_scope"].eq("trajectory").any():
        table = table.loc[table["metric_scope"].eq("trajectory")]
    if "prediction_model" not in table:
        return table
    return table.loc[table["prediction_model"].eq("identified")].copy()


def _error_by_trajectory(
    trajectory_metrics: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(11.0, 5.0))
    table = _identified_metric_rows(trajectory_metrics).reset_index(drop=True)
    required = {
        "hip_torque_rmse_nm",
        "knee_torque_rmse_nm",
        "combined_torque_rmse_nm",
    }
    if table.empty or not required.issubset(table.columns):
        _empty_axis(axis)
    else:
        labels = [_trajectory_label(row) for _, row in table.iterrows()]
        split_column = _split_column(table)
        if split_column is not None:
            labels = [
                f"{label}\n[{_short_split(split)}]"
                for label, split in zip(labels, table[split_column].astype(str))
            ]
        x = np.arange(len(table), dtype=float)
        width = 0.25
        for offset, column, label, color in (
            (-width, "hip_torque_rmse_nm", "hip", "#4C78A8"),
            (0.0, "knee_torque_rmse_nm", "knee", "#E45756"),
            (width, "combined_torque_rmse_nm", "combined", "#72B7B2"),
        ):
            axis.bar(x + offset, table[column], width, label=label, color=color)
        axis.set_xticks(x, labels=labels, rotation=35, ha="right")
        axis.set_ylabel("identified torque RMSE (N·m)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axis.set_title(f"Prediction error by trajectory | {identity}")
    return _save(figure, path)


def _aggregate_by_split(
    trajectory_metrics: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    split_column = _split_column(trajectory_metrics)
    if (
        trajectory_metrics.empty
        or split_column is None
        or value_column not in trajectory_metrics
    ):
        return pd.DataFrame()
    table = trajectory_metrics.copy()
    if "metric_scope" in table and table["metric_scope"].eq("split").any():
        table = table.loc[table["metric_scope"].eq("split")]
    table[value_column] = pd.to_numeric(table[value_column], errors="coerce")
    if "prediction_model" not in table:
        table["prediction_model"] = "identified"
    return (
        table.groupby([split_column, "prediction_model"], sort=False, dropna=False)[
            value_column
        ]
        .mean()
        .rename("value")
        .reset_index()
        .rename(columns={split_column: "split"})
    )


def _split_bars(
    aggregate: pd.DataFrame,
    ylabel: str,
    title: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    if aggregate.empty or not np.isfinite(aggregate["value"]).any():
        _empty_axis(axis)
    else:
        splits = list(dict.fromkeys(aggregate["split"].astype(str)))
        models = list(dict.fromkeys(aggregate["prediction_model"].astype(str)))
        x = np.arange(len(splits), dtype=float)
        width = min(0.8 / max(len(models), 1), 0.38)
        for index, model in enumerate(models):
            lookup = (
                aggregate.loc[aggregate["prediction_model"].astype(str).eq(model)]
                .set_index(aggregate.loc[
                    aggregate["prediction_model"].astype(str).eq(model)
                ]["split"].astype(str))["value"]
            )
            values = [lookup.get(split, np.nan) for split in splits]
            offset = (index - (len(models) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width,
                label=model,
                color=MODEL_COLORS.get(model),
            )
        axis.set_xticks(x, labels=splits, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axis.set_title(title)
    return _save(figure, path)


def _nrmse_by_split(
    trajectory_metrics: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    aggregate = _aggregate_by_split(
        trajectory_metrics,
        "combined_nrmse_percent",
    )
    return _split_bars(
        aggregate,
        "mean combined NRMSE (%)",
        f"NRMSE by split | {identity}",
        path,
    )


def _peak_error_by_split(
    trajectory_metrics: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    table = trajectory_metrics.copy()
    if "combined_peak_error_percent" not in table and {
        "hip_peak_error_percent",
        "knee_peak_error_percent",
    }.issubset(table.columns):
        table["combined_peak_error_percent"] = table[
            ["hip_peak_error_percent", "knee_peak_error_percent"]
        ].max(axis=1)
    aggregate = _aggregate_by_split(table, "combined_peak_error_percent")
    return _split_bars(
        aggregate,
        "mean combined peak error (%)",
        f"Peak torque error by split | {identity}",
        path,
    )


def _residual_over_trajectory(
    predictions: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=False)
    try:
        columns = resolve_mismatch_prediction_columns(predictions, "identified")
    except ValueError:
        columns = {}
    group_columns = [
        column
        for column in (
            "dataset_split",
            "split",
            "trajectory_family",
            "trajectory_name",
            "trajectory_id",
            "speed_profile",
        )
        if column in predictions
    ]
    # Avoid grouping twice by equivalent split aliases.
    if "dataset_split" in group_columns and "split" in group_columns:
        group_columns.remove("split")
    if group_columns:
        grouper: str | list[str]
        grouper = group_columns[0] if len(group_columns) == 1 else group_columns
        groups: Iterable[tuple[object, pd.DataFrame]] = predictions.groupby(
            grouper,
            sort=False,
            dropna=False,
        )
    else:
        groups = (("trajectory", predictions),)

    any_plotted = [False, False]
    for group_key, group in groups:
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        identity_values = dict(zip(group_columns, key_tuple))
        label = _trajectory_label(identity_values)
        split = str(
            identity_values.get(
                "dataset_split",
                identity_values.get("split", "all"),
            )
        )
        label = f"{label} [{_short_split(split)}]"
        if "time_s" in group:
            x = pd.to_numeric(group["time_s"], errors="coerce").to_numpy(
                dtype=float
            )
            xlabel = "trajectory-local time (s)"
        else:
            x = np.arange(len(group), dtype=float)
            xlabel = "trajectory-local sample"
        for axis_index, (axis, joint) in enumerate(zip(axes, ("hip", "knee"))):
            true_column = columns.get(f"true_{joint}")
            predicted_column = columns.get(f"predicted_{joint}")
            if true_column is None or predicted_column is None:
                continue
            true = pd.to_numeric(group[true_column], errors="coerce").to_numpy(
                dtype=float
            )
            predicted = pd.to_numeric(
                group[predicted_column], errors="coerce"
            ).to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(true) & np.isfinite(predicted)
            if not np.any(finite):
                continue
            axis.plot(
                x[finite],
                (true - predicted)[finite],
                linewidth=0.9,
                alpha=0.8,
                color=SPLIT_COLORS.get(split),
                label=label,
            )
            any_plotted[axis_index] = True
            axis.set_ylabel(f"{joint} residual (N·m)")
            axis.set_xlabel(xlabel)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.25)
    for plotted, axis in zip(any_plotted, axes):
        if plotted:
            axis.legend(fontsize=6, ncol=2)
        else:
            _empty_axis(axis)
    figure.suptitle(f"Identified-model residual over trajectory | {identity}")
    return _save(figure, path)


def _strongest_signed(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return np.nan
    return float(finite.iloc[int(np.argmax(np.abs(finite.to_numpy(dtype=float))))])


def _residual_feature_correlations(
    correlations: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    required = {"residual_joint", "feature", "correlation"}
    if correlations.empty or not required.issubset(correlations.columns):
        _empty_axis(axis)
    else:
        table = correlations.copy()
        if "prediction_model" in table:
            selected = table.loc[table["prediction_model"].eq("identified")]
            if not selected.empty:
                table = selected
        pivot = table.pivot_table(
            index="residual_joint",
            columns="feature",
            values="correlation",
            aggfunc=_strongest_signed,
            dropna=False,
        )
        desired_features = [
            feature
            for feature in (
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            )
            if feature in pivot.columns
        ]
        pivot = pivot.reindex(index=["hip", "knee"], columns=desired_features)
        values = pivot.to_numpy(dtype=float)
        if values.size == 0:
            _empty_axis(axis)
        else:
            color_map = plt.get_cmap("coolwarm").copy()
            color_map.set_bad("#D9D9D9")
            image = axis.imshow(
                np.ma.masked_invalid(values),
                cmap=color_map,
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
            )
            axis.set_xticks(
                range(len(pivot.columns)),
                labels=list(pivot.columns),
                rotation=30,
                ha="right",
            )
            axis.set_yticks(
                range(len(pivot.index)),
                labels=[f"{joint} residual" for joint in pivot.index],
            )
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    value = values[row, column]
                    label = "NA" if not np.isfinite(value) else f"{value:.2f}"
                    axis.text(
                        column,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=(
                            "white"
                            if np.isfinite(value) and abs(value) > 0.55
                            else "black"
                        ),
                    )
            figure.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title(
        "Residual-feature correlations (diagnostic, not mechanism proof)"
        f" | {identity}"
    )
    return _save(figure, path)


def _residual_vs_feature(
    predictions: pd.DataFrame,
    feature: str,
    identity: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=False)
    try:
        columns = resolve_mismatch_prediction_columns(predictions, "identified")
    except ValueError:
        columns = {}
    split_column = _split_column(predictions)
    if split_column is None or predictions.empty:
        groups: Iterable[tuple[object, pd.DataFrame]] = (("all", predictions),)
    else:
        groups = predictions.groupby(split_column, sort=False, dropna=False)

    plotted = [False, False]
    for split, group in groups:
        if feature not in group:
            continue
        feature_values = pd.to_numeric(
            group[feature], errors="coerce"
        ).to_numpy(dtype=float)
        for axis_index, (axis, joint) in enumerate(zip(axes, ("hip", "knee"))):
            true_column = columns.get(f"true_{joint}")
            predicted_column = columns.get(f"predicted_{joint}")
            if true_column is None or predicted_column is None:
                continue
            true = pd.to_numeric(group[true_column], errors="coerce").to_numpy(
                dtype=float
            )
            predicted = pd.to_numeric(
                group[predicted_column], errors="coerce"
            ).to_numpy(dtype=float)
            finite = (
                np.isfinite(feature_values)
                & np.isfinite(true)
                & np.isfinite(predicted)
            )
            if not np.any(finite):
                continue
            axis.scatter(
                feature_values[finite],
                (true - predicted)[finite],
                s=10,
                alpha=0.35,
                color=SPLIT_COLORS.get(str(split)),
                marker=SPLIT_MARKERS.get(str(split), "o"),
                label=str(split),
            )
            plotted[axis_index] = True
    for was_plotted, axis, joint in zip(plotted, axes, ("hip", "knee")):
        if was_plotted:
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xlabel(FEATURE_AXIS_LABELS[feature])
            axis.set_ylabel(f"{joint} residual (N·m)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        else:
            _empty_axis(axis)
        axis.set_title(f"{joint.capitalize()} residual")
    figure.suptitle(
        f"Residual vs {FEATURE_AXIS_LABELS[feature]} | {identity}\n"
        "Pattern is diagnostic only; it does not prove a physiological mechanism"
    )
    return _save(figure, path)


def _parameter_columns(
    parameter_table: pd.DataFrame,
) -> tuple[str | None, str | None, str | None]:
    parameter_column = next(
        (
            column
            for column in ("parameter", "parameter_name", "name")
            if column in parameter_table
        ),
        None,
    )
    generic_column = next(
        (
            column
            for column in (
                "generic_value",
                "baseline_value",
                "generic_baseline_value",
                "initial_value",
                "true_value",
            )
            if column in parameter_table
        ),
        None,
    )
    identified_column = next(
        (
            column
            for column in (
                "identified_value",
                "estimated_value",
                "identified_equivalent_value",
            )
            if column in parameter_table
        ),
        None,
    )
    return parameter_column, generic_column, identified_column


def _parameter_shift(
    parameter_table: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 5, figsize=(14.5, 3.6))
    parameter_column, generic_column, identified_column = _parameter_columns(
        parameter_table
    )
    if (
        parameter_table.empty
        or parameter_column is None
        or generic_column is None
        or identified_column is None
    ):
        for axis in axes:
            _empty_axis(axis, "No parameter comparison")
    else:
        available = {
            str(row[parameter_column]): row
            for _, row in parameter_table.iterrows()
        }
        parameters = list(PARAMETER_DISPLAY_NAMES)
        for axis, parameter in zip(axes, parameters):
            row = available.get(parameter)
            if row is None:
                _empty_axis(axis, "Parameter unavailable")
                axis.set_title(PARAMETER_DISPLAY_NAMES[parameter])
                continue
            generic_value = pd.to_numeric(
                pd.Series([row[generic_column]]), errors="coerce"
            ).iloc[0]
            identified_value = pd.to_numeric(
                pd.Series([row[identified_column]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(generic_value) or not np.isfinite(identified_value):
                _empty_axis(axis, "Non-finite value")
            else:
                axis.bar(
                    ["generic", "identified"],
                    [generic_value, identified_value],
                    color=[MODEL_COLORS["generic"], MODEL_COLORS["identified"]],
                )
                axis.tick_params(axis="x", rotation=20)
                axis.set_ylabel(PARAMETER_UNITS[parameter])
                axis.grid(axis="y", alpha=0.25)
            axis.set_title(PARAMETER_DISPLAY_NAMES[parameter])
    figure.suptitle(
        "Generic-to-identified equivalent parameter shift"
        f" | {identity}\nEquivalent parameters are not direct tissue measurements"
    )
    return _save(figure, path)


def generate_model_mismatch_visualizations(
    predictions: pd.DataFrame,
    trajectory_metrics: pd.DataFrame,
    generic_vs_identified: pd.DataFrame,
    residual_correlations: pd.DataFrame,
    parameter_table: pd.DataFrame,
    subject_id: str,
    scenario_name: str,
    output_dir: str | Path,
) -> list[Path]:
    """Generate the eight required model-mismatch figures using Matplotlib.

    Empty tables and constant residual features produce explicit ``No data`` or
    ``NA`` panels rather than raising during batch experiments.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity = _identity(subject_id, scenario_name)
    required_paths = [
        _true_vs_predicted_torque(
            predictions,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[0],
        ),
        _generic_vs_identified_rmse(
            generic_vs_identified,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[1],
        ),
        _error_by_trajectory(
            trajectory_metrics,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[2],
        ),
        _nrmse_by_split(
            trajectory_metrics,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[3],
        ),
        _peak_error_by_split(
            trajectory_metrics,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[4],
        ),
        _residual_over_trajectory(
            predictions,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[5],
        ),
        _residual_feature_correlations(
            residual_correlations,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[6],
        ),
        _parameter_shift(
            parameter_table,
            identity,
            destination / REQUIRED_FIGURE_FILENAMES[7],
        ),
    ]
    residual_paths = [
        _residual_vs_feature(
            predictions,
            feature,
            identity,
            destination / filename,
        )
        for feature, filename in RESIDUAL_FEATURE_FIGURE_FILENAMES.items()
    ]
    return [*required_paths, *residual_paths]


# A compact alias for command-line orchestration.
visualize_model_mismatch = generate_model_mismatch_visualizations
