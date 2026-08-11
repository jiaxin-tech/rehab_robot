"""阶段 4.5B 变化延迟跟踪、样本匹配和辨识结果可视化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .parameter_estimator import PARAMETER_NAMES

METHOD_ORDER = (
    "row_index_alignment",
    "global_fixed_delay",
    "causal_history_latest",
    "causal_buffered_matching",
)
METHOD_LABELS = {
    "row_index_alignment": "row index",
    "global_fixed_delay": "global fixed",
    "causal_history_latest": "causal latest",
    "causal_buffered_matching": "causal buffered",
}
METHOD_COLORS = {
    "row_index_alignment": "#777777",
    "global_fixed_delay": "#54A24B",
    "causal_history_latest": "#F58518",
    "causal_buffered_matching": "#E45756",
}
PARAMETER_LABELS = {
    "mass_scale": "mass scale",
    "k_hip_nm_per_rad": "K hip",
    "k_knee_nm_per_rad": "K knee",
    "b_hip_nm_s_per_rad": "B hip",
    "b_knee_nm_s_per_rad": "B knee",
}


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _tracking_axis(tracking: pd.DataFrame) -> np.ndarray:
    if "tracking_sample_index" in tracking:
        return tracking["tracking_sample_index"].to_numpy(dtype=float)
    return np.arange(len(tracking), dtype=float)


def _true_vs_estimated_delay(
    tracking: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    x = _tracking_axis(tracking)
    true_delay_ms = np.round(
        1000.0 * tracking["true_window_delay_s"].to_numpy(dtype=float),
        decimals=6,
    )
    estimated_delay_ms = np.round(
        tracking["estimated_delay_ms"].to_numpy(dtype=float),
        decimals=6,
    )
    figure, axis = plt.subplots(figsize=(10.0, 4.5))
    axis.plot(
        x,
        true_delay_ms,
        color="#4C78A8",
        linewidth=1.3,
        label="true window delay",
    )
    axis.plot(
        x,
        estimated_delay_ms,
        color="#E45756",
        linewidth=1.3,
        label="filtered estimate",
    )
    axis.set_xlabel("causal update index")
    axis.set_ylabel("delay (ms)")
    axis.set_title(f"True vs estimated delay — {identity}")
    axis.grid(alpha=0.25)
    axis.legend()
    finite = np.concatenate((true_delay_ms, estimated_delay_ms))
    finite = finite[np.isfinite(finite)]
    if len(finite):
        span = float(np.ptp(finite))
        padding = max(1.0, 0.08 * span)
        axis.set_ylim(float(np.min(finite) - padding), float(np.max(finite) + padding))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    return _save(figure, path)


def _delay_error(
    tracking: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    x = _tracking_axis(tracking)
    error = (
        tracking["estimated_delay_ms"].to_numpy(dtype=float)
        - 1000.0 * tracking["true_window_delay_s"].to_numpy(dtype=float)
    )
    error = np.round(error, decimals=6)
    figure, axis = plt.subplots(figsize=(10.0, 4.2))
    axis.plot(x, error, color="#E45756", linewidth=1.0)
    axis.axhline(0.0, color="#777777", linewidth=0.9)
    axis.set_xlabel("causal update index")
    axis.set_ylabel("estimated - true delay (ms)")
    axis.set_title(f"Delay tracking error — {identity}")
    axis.grid(alpha=0.25)
    finite = error[np.isfinite(error)]
    if len(finite):
        magnitude = max(float(np.max(np.abs(finite))), 1.0)
        axis.set_ylim(-1.1 * magnitude, 1.1 * magnitude)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    return _save(figure, path)


def _delay_confidence(
    tracking: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    x = _tracking_axis(tracking)
    valid = tracking["delay_update_valid"].astype(bool).to_numpy()
    figure, axis = plt.subplots(figsize=(10.0, 4.2))
    axis.plot(
        x,
        tracking["delay_confidence"],
        color="#4C78A8",
        linewidth=1.0,
    )
    axis.scatter(
        x[~valid],
        tracking.loc[~valid, "delay_confidence"],
        marker="x",
        color="#E45756",
        label="held/rejected update",
        zorder=3,
    )
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("causal update index")
    axis.set_ylabel("delay confidence (0–1)")
    axis.set_title(f"Delay confidence — {identity}")
    axis.grid(alpha=0.25)
    if (~valid).any():
        axis.legend()
    return _save(figure, path)


def _state_match_error(
    matched: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    selected = matched.loc[
        matched["alignment_method"].eq("causal_buffered_matching")
    ].reset_index(drop=True)
    valid = selected["alignment_valid"].astype(bool).to_numpy()
    x = np.arange(len(selected))
    error_ms = 1000.0 * selected["state_match_error_s"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10.0, 4.2))
    axis.scatter(
        x[valid],
        error_ms[valid],
        s=5,
        alpha=0.45,
        color="#4C78A8",
        label="valid",
    )
    rejected_y = np.zeros((~valid).sum())
    axis.scatter(
        x[~valid],
        rejected_y,
        s=10,
        marker="x",
        color="#E45756",
        label="rejected",
    )
    axis.set_xlabel("concatenated arrival sample")
    axis.set_ylabel("state match error (ms)")
    axis.set_title(f"Buffered state match error — {identity}")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save(figure, path)


def _torque_prediction_comparison(
    comparison: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    ordered = comparison.set_index("alignment_method").reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for offset, (column, label, color) in enumerate(
        (
            ("hip_test_rmse_nm", "hip", "#4C78A8"),
            ("knee_test_rmse_nm", "knee", "#F58518"),
            ("test_torque_rmse_nm", "combined", "#E45756"),
        )
    ):
        axis.bar(
            x + (offset - 1) * width,
            ordered[column],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(
        x,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("test torque RMSE (N·m)")
    axis.set_title(f"Torque prediction comparison — {identity}")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save(figure, path)


def _parameter_error_by_method(
    parameter_table: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    pivot = parameter_table.pivot(
        index="parameter",
        columns="alignment_method",
        values="relative_error_percent",
    ).reindex(PARAMETER_NAMES)
    x = np.arange(len(PARAMETER_NAMES))
    width = 0.18
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    for index, method in enumerate(METHOD_ORDER):
        axis.bar(
            x + (index - 1.5) * width,
            pivot[method],
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axis.set_xticks(
        x,
        [PARAMETER_LABELS[name] for name in PARAMETER_NAMES],
    )
    axis.set_ylabel("absolute relative error (%)")
    axis.set_title(f"Parameter error by alignment method — {identity}")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save(figure, path)


def _valid_rejected_samples(
    comparison: pd.DataFrame,
    identity: str,
    path: Path,
) -> Path:
    ordered = comparison.set_index("alignment_method").reindex(METHOD_ORDER)
    valid = ordered["valid_match_count"].to_numpy(dtype=float)
    rejected = ordered["rejected_match_count"].to_numpy(dtype=float)
    x = np.arange(len(METHOD_ORDER))
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.bar(x, valid, label="valid", color="#54A24B")
    axis.bar(x, rejected, bottom=valid, label="rejected", color="#E45756")
    axis.set_xticks(
        x,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("sample count")
    axis.set_title(f"Valid and rejected samples — {identity}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save(figure, path)


def generate_variable_delay_visualizations(
    raw_dataset: pd.DataFrame,
    matched_dataset: pd.DataFrame,
    delay_tracking_history: pd.DataFrame,
    method_comparison: pd.DataFrame,
    parameter_estimates: pd.DataFrame,
    subject_id: str,
    delay_scenario: str,
    output_dir: str | Path,
) -> list[Path]:
    """生成阶段 4.5B 规定的七张离线实验图。"""

    del raw_dataset  # raw data are represented by tracking/matching summaries.
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity = f"{subject_id}/{delay_scenario}"
    return [
        _true_vs_estimated_delay(
            delay_tracking_history,
            identity,
            destination / "true_vs_estimated_delay.png",
        ),
        _delay_error(
            delay_tracking_history,
            identity,
            destination / "delay_error_vs_time.png",
        ),
        _delay_confidence(
            delay_tracking_history,
            identity,
            destination / "delay_confidence_vs_time.png",
        ),
        _state_match_error(
            matched_dataset,
            identity,
            destination / "state_match_error_vs_time.png",
        ),
        _torque_prediction_comparison(
            method_comparison,
            identity,
            destination / "torque_prediction_comparison.png",
        ),
        _parameter_error_by_method(
            parameter_estimates,
            identity,
            destination / "parameter_error_by_method.png",
        ),
        _valid_rejected_samples(
            method_comparison,
            identity,
            destination / "valid_rejected_samples.png",
        ),
    ]
