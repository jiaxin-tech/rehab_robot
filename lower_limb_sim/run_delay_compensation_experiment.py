"""运行阶段 4.5A 固定 wrench 延迟估计与补偿对照实验。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    delay_compensation_data_dir,
    delay_compensation_model_version,
    delay_search_common_margin_s,
    delay_search_max_ms,
    delay_search_min_ms,
    delay_search_step_ms,
    delay_search_values_ms,
    dynamic_sampling_frequency_hz,
    identification_initial_guess,
    identification_loss,
    identification_lower_bounds,
    identification_random_seed,
    identification_upper_bounds,
    known_delay_experiments_ms,
    max_alignment_interpolation_gap_s,
)
from .delay_estimator import DelayEstimationResult, estimate_wrench_delay
from .dynamic_subject import DYNAMIC_SUBJECTS, get_dynamic_subject
from .identification_dataset import (
    build_identification_dataset,
    split_identification_dataset,
)
from .parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
    compute_torque_metrics,
    estimate_subject_parameters,
)
from .run_identification import (
    _json_safe,
    _parameter_evaluation_table,
    _true_parameters_for_evaluation,
)
from .simulate_dynamic_trajectory import _safe_git_commit
from .timestamp_alignment import (
    align_wrench_to_state_timestamps,
    synthesize_delayed_wrench_dataset,
)

COMPENSATION_METHODS = ("uncompensated", "known_truth", "automatic_estimated")
COMPENSATION_METHOD_LABELS = {
    "uncompensated": "uncorrected",
    "known_truth": "known_delay_compensation",
    "automatic_estimated": "automatic_delay_compensation",
}


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


def _delay_directory_name(delay_ms: int | float) -> str:
    rounded = int(round(float(delay_ms)))
    sign = "minus_" if rounded < 0 else ""
    return f"delay_{sign}{abs(rounded):03d}ms"


def _evaluate_metrics(
    splits: Mapping[str, pd.DataFrame],
    template,
    parameters: Mapping[str, float],
) -> dict[str, dict[str, float | int]]:
    return {
        split: compute_torque_metrics(
            dataframe,
            template,
            parameters,
            L1,
            L2,
        )
        for split, dataframe in splits.items()
    }


def _summary_row(
    subject_id: str,
    true_delay_ms: float,
    selection: DelayEstimationResult,
    method: str,
    applied_delay_ms: float,
    parameter_table: pd.DataFrame,
    metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, object]:
    parameter_errors = {
        str(row.parameter): float(row.relative_error_percent)
        for row in parameter_table.itertuples(index=False)
    }
    estimated_parameters = {
        str(row.parameter): float(row.estimated_value)
        for row in parameter_table.itertuples(index=False)
    }
    true_parameters = {
        str(row.parameter): float(row.true_value)
        for row in parameter_table.itertuples(index=False)
    }
    row: dict[str, object] = {
        "subject_id": subject_id,
        "true_delay_ms": true_delay_ms,
        "estimated_delay_ms": selection.estimated_delay_ms,
        "delay_error_ms": abs(selection.estimated_delay_ms - true_delay_ms),
        "search_boundary_hit": selection.search_boundary_hit,
        "search_warning": selection.search_warning or "",
        "alignment_mode": selection.alignment_mode,
        "compensation_method": method,
        "comparison_label": COMPENSATION_METHOD_LABELS[method],
        "applied_delay_ms": applied_delay_ms,
        "parameter_mean_relative_error_percent": float(
            np.mean(list(parameter_errors.values()))
        ),
        "parameter_max_relative_error_percent": float(
            np.max(list(parameter_errors.values()))
        ),
        **{
            f"{name}_relative_error_percent": value
            for name, value in parameter_errors.items()
        },
        **{
            f"estimated_{name}": value
            for name, value in estimated_parameters.items()
        },
        **{
            f"true_{name}": value
            for name, value in true_parameters.items()
        },
    }
    for split, split_metrics in metrics.items():
        for key, value in split_metrics.items():
            row[f"{split}_{key}"] = value
    return row


def _invalid_sample_counts(
    delayed: pd.DataFrame,
    compensated: pd.DataFrame,
) -> dict[str, int]:
    delayed_reason = delayed.get(
        "invalid_reason",
        pd.Series("", index=delayed.index, dtype=str),
    ).fillna("").astype(str)
    compensated_reason = compensated.get(
        "invalid_reason",
        pd.Series("", index=compensated.index, dtype=str),
    ).fillna("").astype(str)
    stale = compensated.get(
        "wrench_is_stale",
        pd.Series(False, index=compensated.index, dtype=bool),
    ).astype(bool)
    return {
        "valid_samples_before": int(
            delayed["sample_valid"].astype(bool).sum()
        ),
        "valid_samples_after": int(
            compensated["sample_valid"].astype(bool).sum()
        ),
        "invalid_gap_samples": int(
            compensated_reason.str.contains(
                "gap|no_bracketing",
                case=False,
                regex=True,
            ).sum()
        ),
        "invalid_dropout_samples": int(
            (
                delayed_reason.str.contains("dropout", case=False)
                | compensated_reason.str.contains("dropout", case=False)
            ).sum()
        ),
        "invalid_stale_samples": int(
            (
                stale
                | delayed_reason.str.contains("stale|freeze", case=False)
                | compensated_reason.str.contains(
                    "stale|freeze",
                    case=False,
                )
            ).sum()
        ),
    }


def _flat_experiment_summary(
    detailed_summary: pd.DataFrame,
    selection: DelayEstimationResult,
    invalid_counts: Mapping[str, int],
) -> pd.DataFrame:
    rows = {
        str(row.compensation_method): row
        for row in detailed_summary.itertuples(index=False)
    }
    before = rows["uncompensated"]
    known = rows["known_truth"]
    automatic = rows["automatic_estimated"]

    def value(row: object, name: str) -> float:
        return float(getattr(row, name))

    flat = {
        "subject_id": before.subject_id,
        "noise_scenario": "fixed_wrench_delay",
        "true_delay_ms": value(before, "true_delay_ms"),
        "estimated_delay_ms": value(before, "estimated_delay_ms"),
        "delay_error_ms": value(before, "delay_error_ms"),
        "search_boundary_hit": selection.search_boundary_hit,
        "search_warning": selection.search_warning or "",
        "alignment_mode": selection.alignment_mode,
        "uncorrected_parameter_error_percent": value(
            before,
            "parameter_mean_relative_error_percent",
        ),
        "known_compensated_parameter_error_percent": value(
            known,
            "parameter_mean_relative_error_percent",
        ),
        "automatic_compensated_parameter_error_percent": value(
            automatic,
            "parameter_mean_relative_error_percent",
        ),
        "uncorrected_train_rmse_nm": value(
            before,
            "train_torque_rmse_combined_nm",
        ),
        "uncorrected_validation_rmse_nm": value(
            before,
            "validation_torque_rmse_combined_nm",
        ),
        "uncorrected_test_rmse_nm": value(
            before,
            "test_torque_rmse_combined_nm",
        ),
        "known_train_rmse_nm": value(
            known,
            "train_torque_rmse_combined_nm",
        ),
        "known_validation_rmse_nm": value(
            known,
            "validation_torque_rmse_combined_nm",
        ),
        "known_test_rmse_nm": value(
            known,
            "test_torque_rmse_combined_nm",
        ),
        "automatic_train_rmse_nm": value(
            automatic,
            "train_torque_rmse_combined_nm",
        ),
        "automatic_validation_rmse_nm": value(
            automatic,
            "validation_torque_rmse_combined_nm",
        ),
        "automatic_test_rmse_nm": value(
            automatic,
            "test_torque_rmse_combined_nm",
        ),
        "b_hip_error_before_percent": value(
            before,
            "b_hip_nm_s_per_rad_relative_error_percent",
        ),
        "b_hip_error_after_percent": value(
            automatic,
            "b_hip_nm_s_per_rad_relative_error_percent",
        ),
        "b_knee_error_before_percent": value(
            before,
            "b_knee_nm_s_per_rad_relative_error_percent",
        ),
        "b_knee_error_after_percent": value(
            automatic,
            "b_knee_nm_s_per_rad_relative_error_percent",
        ),
        **dict(invalid_counts),
    }
    return pd.DataFrame([flat])


def _plot_delay_search_curve(
    curve: pd.DataFrame,
    subject_id: str,
    true_delay_ms: float,
    estimated_delay_ms: float,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        curve["candidate_delay_ms"],
        curve["validation_torque_rmse_combined_nm"],
        color="#4C78A8",
        linewidth=1.6,
    )
    axis.axvline(
        true_delay_ms,
        color="#54A24B",
        linestyle="--",
        label=f"true {true_delay_ms:g} ms",
    )
    axis.axvline(
        estimated_delay_ms,
        color="#E45756",
        linestyle=":",
        label=f"estimated {estimated_delay_ms:g} ms",
    )
    axis.scatter(
        [estimated_delay_ms],
        [
            curve.loc[
                curve["selected"].astype(bool),
                "validation_torque_rmse_combined_nm",
            ].iloc[0]
        ],
        color="#E45756",
        zorder=3,
    )
    axis.set_xlabel("candidate wrench delay (ms)")
    axis.set_ylabel("validation combined torque RMSE (N·m)")
    axis.set_title(
        f"Delay search — {subject_id} — simulated {true_delay_ms:g} ms"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_before_after_parameter_error(
    summary: pd.DataFrame,
    subject_id: str,
    true_delay_ms: float,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("mass", "K hip", "K knee", "B hip", "B knee")
    colors = ("#9D9D9D", "#54A24B", "#E45756")
    x = np.arange(len(PARAMETER_NAMES))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for index, method in enumerate(COMPENSATION_METHODS):
        row = summary.loc[summary["compensation_method"].eq(method)].iloc[0]
        values = [
            row[f"{name}_relative_error_percent"] for name in PARAMETER_NAMES
        ]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method,
            color=colors[index],
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("absolute relative parameter error (%)")
    axis.set_title(
        f"Before/after delay compensation — {subject_id} — "
        f"{true_delay_ms:g} ms"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_delay_estimation_accuracy(
    summary: pd.DataFrame,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    lower = float(
        min(summary["true_delay_ms"].min(), summary["estimated_delay_ms"].min())
    )
    upper = float(
        max(summary["true_delay_ms"].max(), summary["estimated_delay_ms"].max())
    )
    padding = max(2.0, 0.05 * max(upper - lower, 1.0))
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label="ideal",
    )
    for subject_id, group in summary.groupby("subject_id", sort=True):
        ordered = group.sort_values("true_delay_ms")
        axis.plot(
            ordered["true_delay_ms"],
            ordered["estimated_delay_ms"],
            marker="o",
            linewidth=1.4,
            label=str(subject_id),
        )
    axis.set_xlabel("true simulated delay (ms)")
    axis.set_ylabel("estimated delay (ms)")
    axis.set_title("Offline fixed-delay estimation accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_test_rmse_before_after(
    summary: pd.DataFrame,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = tuple(sorted(summary["subject_id"].astype(str).unique()))
    columns = 2 if len(subjects) > 1 else 1
    rows = int(np.ceil(len(subjects) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.0, 3.8 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    styles = {
        "uncorrected_test_rmse_nm": (
            "uncorrected",
            "#777777",
            "--",
            "o",
        ),
        "known_test_rmse_nm": (
            "known delay",
            "#54A24B",
            "-.",
            "s",
        ),
        "automatic_test_rmse_nm": (
            "automatic",
            "#E45756",
            "-",
            "x",
        ),
    }
    legend_handles = None
    legend_labels = None
    for panel_index, (subject_id, group) in enumerate(
        summary.groupby("subject_id", sort=True)
    ):
        axis = axes.flat[panel_index]
        ordered = group.sort_values("true_delay_ms")
        for column, (label, color, linestyle, marker) in styles.items():
            axis.plot(
                ordered["true_delay_ms"],
                ordered[column],
                marker=marker,
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                label=label,
            )
        axis.set_title(str(subject_id))
        axis.grid(alpha=0.25)
        if panel_index % columns == 0:
            axis.set_ylabel("test combined torque RMSE (N·m)")
        if panel_index >= columns * (rows - 1):
            axis.set_xlabel("true simulated delay (ms)")
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
    for unused in axes.flat[len(subjects) :]:
        unused.set_visible(False)
    figure.suptitle(
        "Test torque error before and after delay compensation",
        y=0.995,
    )
    if legend_handles is not None:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.955),
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_damping_error_before_after(
    summary: pd.DataFrame,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = tuple(sorted(summary["subject_id"].astype(str).unique()))
    columns = 2 if len(subjects) > 1 else 1
    rows = int(np.ceil(len(subjects) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.0, 3.8 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    series = (
        (
            "B hip before",
            "b_hip_error_before_percent",
            "#4C78A8",
            "--",
            "o",
        ),
        (
            "B hip automatic",
            "b_hip_error_after_percent",
            "#4C78A8",
            "-",
            "x",
        ),
        (
            "B knee before",
            "b_knee_error_before_percent",
            "#F58518",
            "--",
            "s",
        ),
        (
            "B knee automatic",
            "b_knee_error_after_percent",
            "#F58518",
            "-",
            "+",
        ),
    )
    legend_handles = None
    legend_labels = None
    for panel_index, (subject_id, group) in enumerate(
        summary.groupby("subject_id", sort=True)
    ):
        axis = axes.flat[panel_index]
        ordered = group.sort_values("true_delay_ms")
        for label, column, color, linestyle, marker in series:
            axis.plot(
                ordered["true_delay_ms"],
                ordered[column],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.4,
                label=label,
            )
        axis.set_title(str(subject_id))
        axis.grid(alpha=0.25)
        if panel_index % columns == 0:
            axis.set_ylabel("absolute relative error (%)")
        if panel_index >= columns * (rows - 1):
            axis.set_xlabel("true simulated delay (ms)")
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
    for unused in axes.flat[len(subjects) :]:
        unused.set_visible(False)
    figure.suptitle(
        "Damping parameter error before and after compensation",
        y=0.995,
    )
    if legend_handles is not None:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.955),
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _trajectory_ids(dataframe: pd.DataFrame) -> list[str]:
    return sorted(
        {
            (
                f"{row.trajectory_id}:"
                f"{row.trajectory_family}:{row.speed_profile}"
            )
            for row in dataframe[
                ["trajectory_id", "trajectory_family", "speed_profile"]
            ].drop_duplicates().itertuples(index=False)
        }
    )


def run_single_delay_compensation_experiment(
    subject_id: str,
    true_delay_ms: float,
    *,
    clean_dataset: pd.DataFrame | None = None,
    output_root: str | Path = delay_compensation_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    make_plots: bool = True,
    loss: str = identification_loss,
) -> dict[str, object]:
    """运行一个已知虚拟延迟；真值只在选择完成后用于评价和已知补偿。"""

    if not np.isfinite(true_delay_ms) or true_delay_ms < 0.0:
        raise ValueError("true_delay_ms must be finite and non-negative.")
    subject = get_dynamic_subject(subject_id)
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    if clean_dataset is None:
        clean_dataset = build_identification_dataset(
            subject,
            "clean",
            sampling_frequency_hz=sampling_frequency_hz,
        )
    delayed = synthesize_delayed_wrench_dataset(
        clean_dataset,
        float(true_delay_ms) / 1000.0,
    )
    raw_splits = split_identification_dataset(delayed)

    # 此调用只传 train、validation 和观测模型配置。true delay 与 test 都
    # 不在 estimate_wrench_delay 的函数签名中。
    selection = estimate_wrench_delay(
        raw_splits["train"],
        raw_splits["validation"],
        template,
        L1,
        L2,
        loss=loss,
    )

    true_parameters = _true_parameters_for_evaluation(subject, baseline)
    applied_delays_ms = {
        "uncompensated": 0.0,
        "known_truth": float(true_delay_ms),
        "automatic_estimated": selection.estimated_delay_ms,
    }
    method_rows: list[dict[str, object]] = []
    method_parameter_tables: dict[str, pd.DataFrame] = {}
    automatic_dataset: pd.DataFrame | None = None
    for method in COMPENSATION_METHODS:
        applied_delay_ms = applied_delays_ms[method]
        aligned = align_wrench_to_state_timestamps(
            delayed,
            applied_delay_ms / 1000.0,
            mode="offline_only",
            max_interpolation_gap_s=max_alignment_interpolation_gap_s,
            evaluation_margin_s=delay_search_common_margin_s,
        ).dataframe
        aligned_splits = split_identification_dataset(aligned)
        if method == "automatic_estimated":
            parameters = selection.estimated_parameters
            automatic_dataset = aligned
        else:
            estimate = estimate_subject_parameters(
                aligned_splits["train"],
                template,
                L1,
                L2,
                initial_guess=identification_initial_guess,
                bounds=(
                    identification_lower_bounds,
                    identification_upper_bounds,
                ),
                loss=loss,
            )
            parameters = estimate.estimated_parameters
        parameter_table = _parameter_evaluation_table(
            subject_id,
            f"delay_{true_delay_ms:g}ms_{method}",
            true_parameters,
            parameters,
        )
        metrics = _evaluate_metrics(aligned_splits, template, parameters)
        method_parameter_tables[method] = parameter_table
        method_rows.append(
            _summary_row(
                subject_id,
                float(true_delay_ms),
                selection,
                method,
                applied_delay_ms,
                parameter_table,
                metrics,
            )
        )
    if automatic_dataset is None:
        raise RuntimeError("automatic compensated dataset was not generated.")

    summary = pd.DataFrame(method_rows)
    invalid_counts = _invalid_sample_counts(delayed, automatic_dataset)
    flat_summary = _flat_experiment_summary(
        summary,
        selection,
        invalid_counts,
    )
    # Preserve the existing three-row method comparison while also exposing
    # the requested one-row before/after contract in the same CSV.
    common_flat_values = flat_summary.iloc[0].to_dict()
    for column, value in common_flat_values.items():
        if column not in summary.columns:
            summary[column] = value
    destination = (
        Path(output_root)
        / subject_id
        / _delay_directory_name(true_delay_ms)
    )
    destination.mkdir(parents=True, exist_ok=True)
    selection.search_curve.to_csv(
        destination / "delay_search_curve.csv",
        index=False,
    )
    summary.to_csv(
        destination / "delay_compensation_summary.csv",
        index=False,
    )
    automatic_dataset.to_csv(
        destination / "compensated_dataset.csv",
        index=False,
    )
    parameter_details = pd.concat(
        [
            table.assign(compensation_method=method)
            for method, table in method_parameter_tables.items()
        ],
        ignore_index=True,
    )
    parameter_details.to_csv(
        destination / "before_after_parameter_errors.csv",
        index=False,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    git_commit = _safe_git_commit()
    metadata = {
        "model_version": delay_compensation_model_version,
        "software_version": delay_compensation_model_version,
        "git_commit": git_commit,
        "software_version_or_git_commit": (
            git_commit or delay_compensation_model_version
        ),
        "generated_at_utc": created_at,
        "created_at": created_at,
        "subject_id": subject_id,
        "noise_scenario": "fixed_wrench_delay",
        "true_delay_ms": float(true_delay_ms),
        "estimated_delay_ms": selection.estimated_delay_ms,
        "delay_error_ms": abs(selection.estimated_delay_ms - true_delay_ms),
        "search_boundary_hit": selection.search_boundary_hit,
        "search_warning": selection.search_warning,
        "positive_delay_definition": "F_obs(t) = F_true(t - delay)",
        "timestamp_columns": [
            "state_timestamp_s",
            "wrench_timestamp_s",
            "wrench_age_s",
            "state_wrench_skew_s",
        ],
        "timestamp_clock": "trajectory-local simulated monotonic seconds",
        "automatic_estimator_input_columns": list(
            selection.estimator_input_columns
        ),
        "delay_selection_splits": list(selection.delay_selection_splits),
        "test_used_for_delay_selection": False,
        "true_delay_passed_to_automatic_estimator": False,
        "delay_search_min_ms": delay_search_min_ms,
        "delay_search_max_ms": delay_search_max_ms,
        "delay_search_step_ms": delay_search_step_ms,
        "search_range_ms": [delay_search_min_ms, delay_search_max_ms],
        "search_step_ms": delay_search_step_ms,
        "alignment_mode": "offline_only",
        "offline_only": True,
        "offline_only_warning": (
            "offline_only uses future samples for bidirectional interpolation "
            "and must not be treated as an online controller."
        ),
        "online_equivalence_claimed": False,
        "causal_history_mode_available_separately": True,
        "max_interpolation_gap_s": max_alignment_interpolation_gap_s,
        "maximum_interpolation_gap_ms": (
            1000.0 * max_alignment_interpolation_gap_s
        ),
        "long_freeze_or_dropout_interpolated": False,
        "train_trajectory_ids": _trajectory_ids(raw_splits["train"]),
        "validation_trajectory_ids": _trajectory_ids(
            raw_splits["validation"]
        ),
        "test_trajectory_ids": _trajectory_ids(raw_splits["test"]),
        "random_seed": identification_random_seed,
        "common_training_samples": selection.common_training_samples,
        "common_validation_samples": selection.common_validation_samples,
        **invalid_counts,
        "angle_definition": "theta_shank = q_hip - q_knee",
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "scope_excludes": [
            "model_mismatch",
            "trajectory_optimization",
            "PINN",
            "MPC",
            "real_robot_motion",
            "real_robot_safety_thresholds",
        ],
        "disclaimer": (
            "Offline software-only virtual-data delay identification; not an "
            "online controller, real patient estimate, or robot safety result."
        ),
    }
    _write_json(destination / "metadata.json", metadata)
    figure_paths: list[Path] = []
    if make_plots:
        figure_paths = [
            _plot_delay_search_curve(
                selection.search_curve,
                subject_id,
                float(true_delay_ms),
                selection.estimated_delay_ms,
                destination / "delay_search_curve.png",
            ),
            _plot_before_after_parameter_error(
                summary,
                subject_id,
                float(true_delay_ms),
                destination / "before_after_parameter_error.png",
            ),
        ]
    return {
        "subject_id": subject_id,
        "true_delay_ms": float(true_delay_ms),
        "selection": selection,
        "summary": summary,
        "flat_summary": flat_summary,
        "parameter_details": parameter_details,
        "compensated_dataset": automatic_dataset,
        "metadata": metadata,
        "output_dir": destination,
        "figure_paths": figure_paths,
    }


def run_delay_compensation_experiments(
    subject_id: str = "baseline",
    *,
    true_delays_ms: Sequence[float] = known_delay_experiments_ms,
    output_root: str | Path = delay_compensation_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    make_plots: bool = True,
    loss: str = identification_loss,
) -> pd.DataFrame:
    """对同一受试者批量运行 0/8/16/24/32/40 ms。"""

    subject = get_dynamic_subject(subject_id)
    clean = build_identification_dataset(
        subject,
        "clean",
        sampling_frequency_hz=sampling_frequency_hz,
    )
    results = [
        run_single_delay_compensation_experiment(
            subject_id,
            delay_ms,
            clean_dataset=clean,
            output_root=output_root,
            sampling_frequency_hz=sampling_frequency_hz,
            make_plots=make_plots,
            loss=loss,
        )
        for delay_ms in true_delays_ms
    ]
    aggregate = pd.concat(
        [result["summary"] for result in results],
        ignore_index=True,
    )
    flat_aggregate = pd.concat(
        [result["flat_summary"] for result in results],
        ignore_index=True,
    )
    subject_dir = Path(output_root) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(
        subject_dir / "delay_compensation_summary.csv",
        index=False,
    )
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    all_summary_path = output_root_path / "all_delay_summary.csv"
    if all_summary_path.exists():
        existing = pd.read_csv(all_summary_path)
        compatible = {
            "subject_id",
            "true_delay_ms",
        }.issubset(existing.columns)
        if compatible:
            keys = set(
                zip(
                    flat_aggregate["subject_id"].astype(str),
                    flat_aggregate["true_delay_ms"].astype(float),
                    strict=True,
                )
            )
            retain = [
                (str(row.subject_id), float(row.true_delay_ms)) not in keys
                for row in existing.itertuples(index=False)
            ]
            flat_aggregate = pd.concat(
                [existing.loc[retain], flat_aggregate],
                ignore_index=True,
            )
    flat_aggregate = flat_aggregate.sort_values(
        ["subject_id", "true_delay_ms"],
        kind="mergesort",
    ).reset_index(drop=True)
    flat_aggregate.to_csv(all_summary_path, index=False)
    if make_plots:
        _plot_delay_estimation_accuracy(
            flat_aggregate,
            output_root_path / "delay_estimation_accuracy.png",
        )
        _plot_test_rmse_before_after(
            flat_aggregate,
            output_root_path / "test_rmse_before_after.png",
        )
        _plot_damping_error_before_after(
            flat_aggregate,
            output_root_path / "damping_error_before_after.png",
        )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject_id",
        nargs="?",
        choices=tuple(DYNAMIC_SUBJECTS),
        default="baseline",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="仅运行一个非负虚拟固定延迟；省略时运行全部六个延迟。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=delay_compensation_data_dir,
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=dynamic_sampling_frequency_hz,
    )
    parser.add_argument(
        "--loss",
        choices=("soft_l1", "linear"),
        default=identification_loss,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    delays = (
        (args.delay_ms,)
        if args.delay_ms is not None
        else known_delay_experiments_ms
    )
    summary = run_delay_compensation_experiments(
        args.subject_id,
        true_delays_ms=delays,
        output_root=args.output_dir,
        sampling_frequency_hz=args.sampling_frequency,
        make_plots=not args.no_plots,
        loss=args.loss,
    )
    selected = summary.loc[
        summary["compensation_method"].eq("automatic_estimated")
    ]
    print(
        selected[
            [
                "subject_id",
                "true_delay_ms",
                "estimated_delay_ms",
                "delay_error_ms",
                "test_torque_rmse_combined_nm",
            ]
        ].to_string(index=False)
    )
    print(Path(args.output_dir) / args.subject_id)


if __name__ == "__main__":
    main()
