"""阶段 4.5B：变化延迟条件下的因果缓存式时间匹配实验。

本入口只回放软件虚拟数据。它比较错误的行号对齐、阶段 4.5A 全局固定
延迟、无补偿的因果 latest 和按 wrench 样本时间戳查询历史状态的因果
buffered matching。任何自动估计接口都不会收到 test 或 ``true_delay``。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .causal_sample_matcher import CausalSampleMatcher
from .config import (
    L1,
    L2,
    delay_search_common_margin_s,
    delay_tracker_filter_alpha,
    delay_tracker_minimum_effective_samples,
    delay_tracker_minimum_excitation_score,
    delay_tracker_search_max_ms,
    delay_tracker_search_min_ms,
    delay_tracker_search_step_ms,
    delay_tracker_update_interval_s,
    delay_tracker_window_duration_s,
    dynamic_sampling_frequency_hz,
    identification_initial_guess,
    identification_loss,
    identification_lower_bounds,
    identification_random_seed,
    identification_upper_bounds,
    max_alignment_interpolation_gap_s,
    max_state_interpolation_interval_s,
    max_state_match_error_s,
    max_wrench_age_s,
    maximum_delay_change_per_update_ms,
    state_buffer_duration_s,
    variable_delay_core_scenarios,
    variable_delay_data_dir,
    variable_delay_model_version,
    variable_delay_random_seed,
    variable_delay_scenarios,
)
from .delay_estimator import DelayEstimationResult, estimate_wrench_delay
from .dynamic_subject import DYNAMIC_SUBJECTS, DynamicVirtualSubject, get_dynamic_subject
from .identification_dataset import build_identification_dataset
from .parameter_estimator import (
    PARAMETER_NAMES,
    BaselineSubjectTemplate,
    baseline_template_from_dynamic_subject,
    compute_torque_metrics,
    estimate_subject_parameters,
    measured_joint_torque,
    predict_joint_torque,
    valid_observations,
)
from .state_history_buffer import StateHistoryBuffer
from .timestamp_alignment import align_wrench_to_state_timestamps
from .variable_delay_models import (
    VARIABLE_DELAY_SCENARIOS,
    apply_variable_delay_scenario,
)
from .visualize_variable_delay import generate_variable_delay_visualizations
from .windowed_delay_tracker import WindowedDelayTracker


ALIGNMENT_METHODS = (
    "row_index_alignment",
    "global_fixed_delay",
    "causal_history_latest",
    "causal_buffered_matching",
)
TRAJECTORY_GROUP_COLUMNS = ("trajectory_family", "speed_profile")

# Explicit estimator projection. In particular, scenario names, delay truth,
# timestamp-age audit fields and saved tau values never reach parameter fitting.
ESTIMATOR_INPUT_COLUMNS = (
    "trajectory_id",
    "trajectory_family",
    "speed_profile",
    "phase",
    "time_s",
    "trajectory_sample_index",
    "dataset_split",
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


def _safe_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _split(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"dataset_split"}
    if not required.issubset(dataframe.columns):
        raise ValueError("variable-delay dataframe needs dataset_split.")
    observed = set(dataframe["dataset_split"].astype(str))
    expected = {"train", "validation", "test"}
    if observed != expected:
        raise ValueError(
            f"expected train/validation/test, got {sorted(observed)}."
        )
    return {
        split: dataframe.loc[dataframe["dataset_split"].eq(split)].copy()
        for split in ("train", "validation", "test")
    }


def _estimator_projection(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = set(ESTIMATOR_INPUT_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"estimator projection is missing columns: {sorted(missing)}"
        )
    projected = dataframe.loc[:, ESTIMATOR_INPUT_COLUMNS].copy(deep=True)
    projected.attrs.clear()
    forbidden = [
        column
        for column in projected
        if column.startswith(("true_", "ground_truth", "tau_total"))
    ]
    if forbidden:
        raise RuntimeError(f"ground-truth leakage in projection: {forbidden}")
    return projected.reset_index(drop=True)


def _true_parameters_for_evaluation(
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
        raise ValueError("subject is outside the common mass-scale model.")
    return {
        "mass_scale": float(scales[0]),
        "k_hip_nm_per_rad": float(subject.k_hip_nm_per_rad),
        "k_knee_nm_per_rad": float(subject.k_knee_nm_per_rad),
        "b_hip_nm_s_per_rad": float(subject.b_hip_nm_s_per_rad),
        "b_knee_nm_s_per_rad": float(subject.b_knee_nm_s_per_rad),
    }


def _parameter_table(
    subject_id: str,
    scenario: str,
    method: str,
    truth: Mapping[str, float],
    estimate: Mapping[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in PARAMETER_NAMES:
        true_value = float(truth[name])
        estimated_value = float(estimate[name])
        absolute_error = abs(estimated_value - true_value)
        relative_error = (
            100.0 * absolute_error / abs(true_value)
            if true_value != 0.0
            else np.nan
        )
        rows.append(
            {
                "subject_id": subject_id,
                "delay_scenario": scenario,
                "alignment_method": method,
                "parameter": name,
                "true_value": true_value,
                "estimated_value": estimated_value,
                "absolute_error": absolute_error,
                "relative_error_percent": relative_error,
            }
        )
    return pd.DataFrame(rows)


def _append_invalid_reason(
    dataframe: pd.DataFrame,
    mask: np.ndarray,
    reason: str,
) -> None:
    selected = np.asarray(mask, dtype=bool)
    current = dataframe.loc[selected, "invalid_reason"].fillna("").astype(str)
    dataframe.loc[selected, "invalid_reason"] = np.where(
        current.eq(""),
        reason,
        current + ";" + reason,
    )


def _prepare_row_index_alignment(raw: pd.DataFrame) -> pd.DataFrame:
    aligned = raw.copy(deep=True)
    aligned.attrs.clear()
    valid = (
        aligned["sample_valid"].astype(bool).to_numpy()
        & aligned["force_mapping_valid"].astype(bool).to_numpy()
        & ~aligned["wrench_is_stale"].astype(bool).to_numpy()
        & np.isfinite(
            aligned[["fx_observed_n", "fz_observed_n"]].to_numpy(dtype=float)
        ).all(axis=1)
    )
    aligned["sample_valid"] = valid
    aligned["alignment_valid"] = valid
    aligned["alignment_method"] = "row_index_alignment"
    aligned["alignment_mode"] = "incorrect_row_index_baseline"
    aligned["state_match_error_s"] = np.abs(
        aligned["state_timestamp_s"].to_numpy(dtype=float)
        - aligned["wrench_sample_timestamp_s"].to_numpy(dtype=float)
    )
    aligned["state_match_reason"] = np.where(valid, "", "invalid_raw_wrench")
    return aligned


def _preserve_event_rejections(
    aligned: pd.DataFrame,
    raw: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    result = aligned.copy(deep=True)
    blocked = (
        raw["is_dropout"].astype(bool).to_numpy()
        | raw["is_stale"].astype(bool).to_numpy()
    )
    if blocked.any():
        result.loc[blocked, ["fx_observed_n", "fz_observed_n"]] = np.nan
        result.loc[blocked, "sample_valid"] = False
        result.loc[blocked, "alignment_valid"] = False
        _append_invalid_reason(result, blocked, "source_event_rejected")
    result["alignment_method"] = method
    return result


def _prepare_global_fixed_alignment(
    raw: pd.DataFrame,
    delay_s: float,
) -> pd.DataFrame:
    result = align_wrench_to_state_timestamps(
        raw,
        delay_s,
        mode="offline_only",
        max_interpolation_gap_s=max_alignment_interpolation_gap_s,
        evaluation_margin_s=delay_search_common_margin_s,
    ).dataframe
    result["state_match_error_s"] = np.abs(
        raw["true_delay_s"].to_numpy(dtype=float) - float(delay_s)
    )
    result["state_match_reason"] = result["alignment_invalid_reason"]
    return _preserve_event_rejections(result, raw, "global_fixed_delay")


def _prepare_causal_latest_alignment(raw: pd.DataFrame) -> pd.DataFrame:
    result = align_wrench_to_state_timestamps(
        raw,
        0.0,
        mode="causal_history",
        max_interpolation_gap_s=max_alignment_interpolation_gap_s,
        evaluation_margin_s=0.0,
    ).dataframe
    result["state_match_error_s"] = np.abs(
        raw["state_timestamp_s"].to_numpy(dtype=float)
        - raw["wrench_sample_timestamp_s"].to_numpy(dtype=float)
    )
    result["state_match_reason"] = result["alignment_invalid_reason"]
    return _preserve_event_rejections(result, raw, "causal_history_latest")


def _tracker_for_replay() -> WindowedDelayTracker:
    return WindowedDelayTracker(
        window_duration_s=delay_tracker_window_duration_s,
        update_interval_s=delay_tracker_update_interval_s,
        minimum_delay_ms=delay_tracker_search_min_ms,
        maximum_delay_ms=delay_tracker_search_max_ms,
        delay_step_ms=delay_tracker_search_step_ms,
        smoothing_alpha=delay_tracker_filter_alpha,
        maximum_delay_change_ms=maximum_delay_change_per_update_ms,
        excitation_threshold=delay_tracker_minimum_excitation_score,
        minimum_effective_samples=delay_tracker_minimum_effective_samples,
        minimum_confidence=0.25,
        initial_delay_ms=16.0,
    )


def _window_search_curve(
    group: pd.DataFrame,
    history: pd.DataFrame,
    candidates_ms: np.ndarray,
    trajectory_key: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    arrival = group["wrench_arrival_timestamp_s"].to_numpy(dtype=float)
    sample = group["wrench_sample_timestamp_s"].to_numpy(dtype=float)
    base_valid = (
        group["sample_valid"].astype(bool).to_numpy()
        & ~group["wrench_is_stale"].astype(bool).to_numpy()
        & np.isfinite(arrival)
        & np.isfinite(sample)
    )
    for record in history.itertuples(index=False):
        in_window = (
            group["time_s"].to_numpy(dtype=float) >= record.window_start_s - 1e-12
        ) & (
            group["time_s"].to_numpy(dtype=float) <= record.window_end_s + 1e-12
        )
        valid = base_valid & in_window
        if valid.any():
            observed_ms = 1000.0 * (arrival[valid] - sample[valid])
            score = np.sqrt(
                np.mean(
                    (
                        observed_ms[:, np.newaxis]
                        - candidates_ms[np.newaxis, :]
                    )
                    ** 2,
                    axis=0,
                )
            )
        else:
            score = np.full(len(candidates_ms), np.nan)
        selected = np.zeros(len(candidates_ms), dtype=bool)
        if int(record.selected_candidate_index) >= 0:
            selected[int(record.selected_candidate_index)] = True
        rows.append(
            pd.DataFrame(
                {
                    "trajectory_key": trajectory_key,
                    "window_update_sequence": int(record.update_sequence),
                    "window_start_s": float(record.window_start_s),
                    "window_end_s": float(record.window_end_s),
                    "candidate_delay_ms": candidates_ms,
                    "delay_score": score,
                    "delay_score_unit": "ms_timestamp_residual",
                    "effective_sample_count": int(record.effective_sample_count),
                    "selected": selected,
                    "delay_update_valid": bool(record.delay_update_valid),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _track_training_delays(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    training = raw.loc[raw["dataset_split"].eq("train")].copy()
    histories: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    by_trajectory: dict[tuple[str, str], pd.DataFrame] = {}
    tracking_index = 0
    for keys, group in training.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        family, speed = str(keys[0]), str(keys[1])
        group = group.reset_index(drop=True)
        tracker = _tracker_for_replay()
        history = tracker.track(group)
        if history.empty:
            by_trajectory[(family, speed)] = history
            continue
        true_window_delay: list[float] = []
        for record in history.itertuples(index=False):
            selected = group.loc[
                group["time_s"].between(
                    float(record.window_start_s) - 1e-12,
                    float(record.window_end_s) + 1e-12,
                )
                & group["sample_valid"].astype(bool)
                & ~group["wrench_is_stale"].astype(bool)
            ]
            true_window_delay.append(
                float(selected["true_delay_s"].median())
                if not selected.empty
                else np.nan
            )
        history["true_window_delay_s"] = true_window_delay
        history["trajectory_family"] = family
        history["speed_profile"] = speed
        history["trajectory_key"] = f"{family}/{speed}"
        history["tracking_sample_index"] = np.arange(
            tracking_index,
            tracking_index + len(history),
            dtype=int,
        )
        tracking_index += len(history)
        # Runtime matching receives only observable tracker outputs. The
        # true-window delay is appended solely to the separate evaluation
        # table used for saved metrics and plots.
        by_trajectory[(family, speed)] = history.drop(
            columns=["true_window_delay_s"],
        ).copy()
        histories.append(history)
        curve = _window_search_curve(
            group,
            history,
            tracker.candidate_delays_ms,
            f"{family}/{speed}",
        )
        if not curve.empty:
            curves.append(curve)
    history_table = (
        pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    )
    curve_table = (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    )
    return history_table, curve_table, by_trajectory


def _delay_for_arrival(
    local_history: pd.DataFrame,
    arrival_s: float,
    fallback_delay_ms: float,
) -> tuple[float, float]:
    if local_history.empty:
        return fallback_delay_ms / 1000.0, 0.0
    available = local_history.loc[
        local_history["window_end_s"].le(arrival_s + 1e-12)
    ]
    if available.empty:
        return fallback_delay_ms / 1000.0, 0.0
    latest = available.iloc[-1]
    return (
        float(latest["estimated_delay_ms"]) / 1000.0,
        float(latest["delay_confidence"]),
    )


def _prepare_causal_buffered_alignment(
    raw: pd.DataFrame,
    tracker_histories: Mapping[tuple[str, str], pd.DataFrame],
    fallback_delay_ms: float,
) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []
    for keys, group in raw.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        family, speed = str(keys[0]), str(keys[1])
        history = tracker_histories.get((family, speed), pd.DataFrame())
        buffer = StateHistoryBuffer(
            history_duration_s=state_buffer_duration_s,
            max_state_interval_s=max_state_interpolation_interval_s,
        )
        matcher = CausalSampleMatcher(
            max_wrench_age_s=max_wrench_age_s,
            max_match_error_s=max_state_match_error_s,
            max_state_interval_s=max_state_interpolation_interval_s,
        )
        for row in group.itertuples(index=False):
            arrival = float(row.wrench_arrival_timestamp_s)
            buffer.append(
                float(row.state_timestamp_s),
                float(row.q_hip_rad),
                float(row.q_knee_rad),
                float(row.dq_hip_rad_s),
                float(row.dq_knee_rad_s),
                float(row.ddq_hip_rad_s2),
                float(row.ddq_knee_rad_s2),
            )
            estimated_delay_s, tracker_confidence = _delay_for_arrival(
                history,
                arrival,
                fallback_delay_ms,
            )
            match = matcher.match(
                buffer,
                arrival_timestamp_s=arrival,
                current_timestamp_s=arrival,
                fx_observed_n=float(row.fx_observed_n),
                fz_observed_n=float(row.fz_observed_n),
                estimated_delay_s=estimated_delay_s,
                sample_timestamp_s=float(row.wrench_sample_timestamp_s),
                sample_timestamp_reliable=True,
                wrench_valid=(
                    bool(row.force_mapping_valid)
                    and not bool(row.is_dropout)
                    and np.isfinite(float(row.fx_observed_n))
                    and np.isfinite(float(row.fz_observed_n))
                ),
                wrench_is_stale=bool(row.wrench_is_stale),
            )
            values = row._asdict()
            values.update(match.as_dict())
            values.update(
                {
                    "alignment_method": "causal_buffered_matching",
                    "alignment_mode": "causal_buffered_history_only",
                    "sample_valid": bool(match.valid),
                    "alignment_valid": bool(match.valid),
                    "force_mapping_valid": bool(row.force_mapping_valid),
                    "invalid_reason": str(match.invalid_reason),
                    "state_match_reason": str(match.invalid_reason),
                    "tracker_delay_confidence": tracker_confidence,
                    "state_match_confidence": match.confidence,
                    "delay_confidence": tracker_confidence,
                    "estimated_delay_s": estimated_delay_s,
                }
            )
            output_rows.append(values)
    result = pd.DataFrame(output_rows)
    result.attrs.clear()
    return result


def _method_metrics(
    dataframe: pd.DataFrame,
    template: BaselineSubjectTemplate,
    parameters: Mapping[str, float],
) -> dict[str, dict[str, float | int]]:
    safe = _estimator_projection(dataframe)
    return {
        split: compute_torque_metrics(
            selected,
            template,
            parameters,
            L1,
            L2,
        )
        for split, selected in _split(safe).items()
    }


def _prediction_table(
    dataframe: pd.DataFrame,
    template: BaselineSubjectTemplate,
    parameters: Mapping[str, float],
    method: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    safe = _estimator_projection(dataframe)
    for split, selected in _split(safe).items():
        valid = valid_observations(selected)
        measured_hip, measured_knee = measured_joint_torque(valid, L1, L2)
        predicted_hip, predicted_knee = predict_joint_torque(
            valid,
            template,
            parameters,
            L1,
        )
        frames.append(
            pd.DataFrame(
                {
                    "alignment_method": method,
                    "dataset_split": split,
                    "trajectory_family": valid["trajectory_family"].to_numpy(),
                    "speed_profile": valid["speed_profile"].to_numpy(),
                    "time_s": valid["time_s"].to_numpy(dtype=float),
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


def _rejection_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    valid = dataframe["alignment_valid"].astype(bool).to_numpy()
    reason = dataframe["invalid_reason"].fillna("").astype(str)
    return {
        "dropout_rejected_count": int(
            (dataframe["is_dropout"].astype(bool).to_numpy() & ~valid).sum()
        ),
        "stale_rejected_count": int(
            (dataframe["is_stale"].astype(bool).to_numpy() & ~valid).sum()
        ),
        "long_tail_rejected_count": int(
            (dataframe["is_long_tail"].astype(bool).to_numpy() & ~valid).sum()
        ),
        "state_history_expired_count": int(
            reason.str.contains("state_history_expired", regex=False).sum()
        ),
        "wrench_age_rejected_count": int(
            reason.str.contains("wrench_age_limit_exceeded", regex=False).sum()
        ),
    }


def _delay_metrics(
    method: str,
    raw: pd.DataFrame,
    global_delay_ms: float,
    tracking_history: pd.DataFrame,
) -> dict[str, float]:
    if method == "global_fixed_delay":
        true_ms = 1000.0 * raw.loc[
            raw["sample_valid"].astype(bool), "true_delay_s"
        ].to_numpy(dtype=float)
        estimate_ms = np.full(len(true_ms), global_delay_ms)
    elif method == "causal_buffered_matching" and not tracking_history.empty:
        true_ms = (
            1000.0
            * tracking_history["true_window_delay_s"].to_numpy(dtype=float)
        )
        estimate_ms = tracking_history["estimated_delay_ms"].to_numpy(dtype=float)
    else:
        true_ms = 1000.0 * raw.loc[
            raw["sample_valid"].astype(bool), "true_delay_s"
        ].to_numpy(dtype=float)
        estimate_ms = np.zeros(len(true_ms))
    finite = np.isfinite(true_ms) & np.isfinite(estimate_ms)
    if not finite.any():
        return {
            "mean_delay_error_ms": np.nan,
            "delay_mae_ms": np.nan,
            "delay_rmse_ms": np.nan,
            "median_delay_error_ms": np.nan,
            "p95_delay_error_ms": np.nan,
            "delay_p95_error_ms": np.nan,
            "maximum_delay_error_ms": np.nan,
        }
    absolute_error = np.abs(estimate_ms[finite] - true_ms[finite])
    signed_error = estimate_ms[finite] - true_ms[finite]
    mean_absolute_error = float(np.mean(absolute_error))
    p95_absolute_error = float(np.quantile(absolute_error, 0.95))
    return {
        "mean_delay_error_ms": mean_absolute_error,
        "delay_mae_ms": mean_absolute_error,
        "delay_rmse_ms": float(np.sqrt(np.mean(signed_error**2))),
        "median_delay_error_ms": float(np.median(absolute_error)),
        "p95_delay_error_ms": p95_absolute_error,
        "delay_p95_error_ms": p95_absolute_error,
        "maximum_delay_error_ms": float(np.max(absolute_error)),
    }


def _fit_method(
    dataframe: pd.DataFrame,
    template: BaselineSubjectTemplate,
    subject_id: str,
    scenario: str,
    method: str,
    truth: Mapping[str, float],
    raw: pd.DataFrame,
    global_delay_ms: float,
    global_search_boundary_hit: bool,
    tracking_history: pd.DataFrame,
    loss: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    safe = _estimator_projection(dataframe)
    splits = _split(safe)
    estimate = estimate_subject_parameters(
        splits["train"],
        template,
        L1,
        L2,
        initial_guess=identification_initial_guess,
        bounds=(identification_lower_bounds, identification_upper_bounds),
        loss=loss,
    )
    parameters = estimate.estimated_parameters
    parameter_table = _parameter_table(
        subject_id,
        scenario,
        method,
        truth,
        parameters,
    )
    metrics = _method_metrics(dataframe, template, parameters)
    prediction = _prediction_table(dataframe, template, parameters, method)
    valid = dataframe["alignment_valid"].astype(bool).to_numpy()
    match_error = dataframe.loc[valid, "state_match_error_s"].to_numpy(
        dtype=float
    )
    finite_match_error = match_error[np.isfinite(match_error)]
    valid_match_rate = float(np.mean(valid))
    median_match_error_ms = (
        1000.0 * float(np.median(finite_match_error))
        if len(finite_match_error)
        else np.nan
    )
    p95_match_error_ms = (
        1000.0 * float(np.quantile(finite_match_error, 0.95))
        if len(finite_match_error)
        else np.nan
    )
    maximum_match_error_ms = (
        1000.0 * float(np.max(finite_match_error))
        if len(finite_match_error)
        else np.nan
    )
    if method == "causal_buffered_matching" and not tracking_history.empty:
        update_valid = tracking_history["delay_update_valid"].astype(bool)
        update_success_rate = float(update_valid.mean())
        boundary_hit_count = int(
            tracking_history["search_boundary_hit"].astype(bool).sum()
        )
        low_confidence_update_count = int(
            tracking_history["low_confidence"].astype(bool).sum()
        )
    elif method == "global_fixed_delay":
        update_success_rate = 1.0
        boundary_hit_count = int(global_search_boundary_hit)
        low_confidence_update_count = 0
    else:
        update_success_rate = 0.0
        boundary_hit_count = 0
        low_confidence_update_count = 0
    row: dict[str, object] = {
        "subject_id": subject_id,
        "delay_scenario": scenario,
        "alignment_method": method,
        "optimizer_success": estimate.optimizer_success,
        "optimizer_message": estimate.optimizer_message,
        "valid_match_count": int(valid.sum()),
        "rejected_match_count": int((~valid).sum()),
        "matching_acceptance_rate": valid_match_rate,
        "valid_match_rate": valid_match_rate,
        "mean_state_match_error_ms": (
            1000.0 * float(np.mean(finite_match_error))
            if len(finite_match_error)
            else np.nan
        ),
        "median_state_match_error_ms": median_match_error_ms,
        "p95_state_match_error_ms": p95_match_error_ms,
        "maximum_state_match_error_ms": maximum_match_error_ms,
        "delay_update_success_rate": update_success_rate,
        "boundary_hit_count": boundary_hit_count,
        "low_confidence_update_count": low_confidence_update_count,
        "train_rmse_nm": metrics["train"]["torque_rmse_combined_nm"],
        "validation_rmse_nm": metrics["validation"][
            "torque_rmse_combined_nm"
        ],
        "test_rmse_nm": metrics["test"]["torque_rmse_combined_nm"],
        "hip_test_rmse_nm": metrics["test"]["torque_rmse_hip_nm"],
        "knee_test_rmse_nm": metrics["test"]["torque_rmse_knee_nm"],
        "test_torque_rmse_nm": metrics["test"][
            "torque_rmse_combined_nm"
        ],
        **_rejection_counts(dataframe),
        **_delay_metrics(
            method,
            raw,
            global_delay_ms,
            tracking_history,
        ),
    }
    for split, values in metrics.items():
        for key, value in values.items():
            row[f"{split}_{key}"] = value
        row[f"{split}_torque_rmse_nm"] = values[
            "torque_rmse_combined_nm"
        ]
    for item in parameter_table.itertuples(index=False):
        row[f"{item.parameter}_estimate"] = item.estimated_value
        row[f"{item.parameter}_relative_error_percent"] = (
            item.relative_error_percent
        )
    parameter_metric_aliases = {
        "mass_scale": "mass_scale_error_percent",
        "k_hip_nm_per_rad": "k_hip_error_percent",
        "k_knee_nm_per_rad": "k_knee_error_percent",
        "b_hip_nm_s_per_rad": "b_hip_error_percent",
        "b_knee_nm_s_per_rad": "b_knee_error_percent",
    }
    for parameter, alias in parameter_metric_aliases.items():
        value = parameter_table.loc[
            parameter_table["parameter"].eq(parameter),
            "relative_error_percent",
        ].iloc[0]
        row[alias] = float(value)
    row["mean_parameter_error_percent"] = float(
        parameter_table["relative_error_percent"].mean()
    )
    row["maximum_parameter_error_percent"] = float(
        parameter_table["relative_error_percent"].max()
    )
    return row, parameter_table, prediction


def _trajectory_ids(dataframe: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in (
            "trajectory_id",
            "trajectory_family",
            "speed_profile",
        )
        if column in dataframe
    ]
    return [
        "/".join(str(value) for value in row)
        for row in dataframe[columns].drop_duplicates().itertuples(
            index=False,
            name=None,
        )
    ]


def run_single_variable_delay_experiment(
    subject_id: str,
    delay_scenario: str,
    *,
    clean_dataset: pd.DataFrame | None = None,
    output_root: str | Path = variable_delay_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int = variable_delay_random_seed,
    make_plots: bool = True,
    loss: str = identification_loss,
    global_candidate_delays_s: Sequence[float] | None = None,
) -> dict[str, object]:
    """运行一个受试者/变化延迟场景并保存阶段 4.5B 全部产物。"""

    if delay_scenario not in VARIABLE_DELAY_SCENARIOS:
        raise ValueError(
            f"unknown scenario {delay_scenario!r}; "
            f"choose one of {', '.join(VARIABLE_DELAY_SCENARIOS)}."
        )
    subject = get_dynamic_subject(subject_id)
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    if clean_dataset is None:
        clean_dataset = build_identification_dataset(
            subject,
            "clean",
            sampling_frequency_hz=sampling_frequency_hz,
        )
    application = apply_variable_delay_scenario(
        clean_dataset,
        delay_scenario,
        random_seed=random_seed,
    )
    raw = application.dataframe
    raw_splits = _split(raw)

    # 4.5A comparison: exactly train fit + validation selection. test and
    # true_delay have no place in estimate_wrench_delay's signature.
    global_selection: DelayEstimationResult = estimate_wrench_delay(
        raw_splits["train"],
        raw_splits["validation"],
        template,
        L1,
        L2,
        candidate_delays_s=global_candidate_delays_s,
        loss=loss,
    )

    tracking_history, search_windows, tracker_histories = _track_training_delays(
        raw
    )
    finite_tracker = (
        tracking_history.loc[
            np.isfinite(
                tracking_history["estimated_delay_ms"].to_numpy(dtype=float)
            )
        ]
        if not tracking_history.empty
        else tracking_history
    )
    fallback_delay_ms = (
        float(finite_tracker["estimated_delay_ms"].median())
        if not finite_tracker.empty
        else global_selection.estimated_delay_ms
    )
    method_datasets = {
        "row_index_alignment": _prepare_row_index_alignment(raw),
        "global_fixed_delay": _prepare_global_fixed_alignment(
            raw,
            global_selection.estimated_delay_s,
        ),
        "causal_history_latest": _prepare_causal_latest_alignment(raw),
        "causal_buffered_matching": _prepare_causal_buffered_alignment(
            raw,
            tracker_histories,
            fallback_delay_ms,
        ),
    }
    if tuple(method_datasets) != ALIGNMENT_METHODS:
        raise RuntimeError("alignment method order/coverage changed unexpectedly.")

    # Ground truth is loaded only after all delay estimators and method datasets
    # have been built. It is passed only to the final evaluation table.
    truth = _true_parameters_for_evaluation(subject, baseline)
    comparison_rows: list[dict[str, object]] = []
    parameter_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    for method, dataframe in method_datasets.items():
        row, parameter_table, prediction = _fit_method(
            dataframe,
            template,
            subject_id,
            delay_scenario,
            method,
            truth,
            raw,
            global_selection.estimated_delay_ms,
            global_selection.search_boundary_hit,
            tracking_history,
            loss,
        )
        comparison_rows.append(row)
        parameter_tables.append(parameter_table)
        prediction_tables.append(prediction)
    comparison = pd.DataFrame(comparison_rows)
    parameter_estimates = pd.concat(parameter_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    matched = pd.concat(
        [
            dataframe.assign(alignment_method=method)
            for method, dataframe in method_datasets.items()
        ],
        ignore_index=True,
    )

    destination = Path(output_root) / subject_id / delay_scenario
    destination.mkdir(parents=True, exist_ok=True)
    raw.to_csv(destination / "raw_variable_delay_dataset.csv", index=False)
    matched.to_csv(destination / "matched_dataset.csv", index=False)
    tracking_history.to_csv(
        destination / "delay_tracking_history.csv",
        index=False,
    )
    search_windows.to_csv(
        destination / "delay_search_windows.csv",
        index=False,
    )
    parameter_estimates.to_csv(
        destination / "parameter_estimates_by_method.csv",
        index=False,
    )
    parameter_estimates.to_csv(
        destination / "parameter_estimates.csv",
        index=False,
    )
    comparison.to_csv(
        destination / "method_comparison.csv",
        index=False,
    )
    predictions.to_csv(
        destination / "torque_predictions_by_method.csv",
        index=False,
    )
    global_selection.search_curve.to_csv(
        destination / "global_fixed_delay_search_curve.csv",
        index=False,
    )
    metrics_payload = {
        "subject_id": subject_id,
        "delay_scenario": delay_scenario,
        "methods": comparison.to_dict(orient="records"),
        "delay_metric_units": "ms",
        "state_match_error_units": "ms",
        "torque_metric_units": "N*m",
        "test_used_for_delay_estimation": False,
        "true_delay_used_for_estimation": False,
    }
    _write_json(destination / "metrics.json", metrics_payload)

    tracker_valid = (
        tracking_history["delay_update_valid"].astype(bool)
        if not tracking_history.empty
        else pd.Series(dtype=bool)
    )
    tracker_errors = (
        tracking_history["estimated_delay_ms"].to_numpy(dtype=float)
        - 1000.0
        * tracking_history["true_window_delay_s"].to_numpy(dtype=float)
        if not tracking_history.empty
        else np.asarray([], dtype=float)
    )
    finite_tracker_errors = tracker_errors[np.isfinite(tracker_errors)]
    created_at = datetime.now(timezone.utc).isoformat()
    git_commit = _safe_git_commit()
    metadata = {
        "model_version": variable_delay_model_version,
        "software_version_or_git_commit": (
            git_commit or variable_delay_model_version
        ),
        "created_at": created_at,
        "subject_id": subject_id,
        "delay_scenario": delay_scenario,
        "random_seed": int(random_seed),
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "positive_delay_definition": (
            "wrench_arrival_timestamp_s - wrench_sample_timestamp_s"
        ),
        "timestamp_columns": [
            "state_timestamp_s",
            "wrench_arrival_timestamp_s",
            "wrench_sample_timestamp_s",
            "wrench_age_s",
            "state_wrench_skew_s",
        ],
        "state_history_duration_s": state_buffer_duration_s,
        "maximum_state_interpolation_interval_ms": (
            1000.0 * max_state_interpolation_interval_s
        ),
        "maximum_state_match_error_ms": 1000.0 * max_state_match_error_s,
        "maximum_wrench_age_ms": 1000.0 * max_wrench_age_s,
        "delay_tracker_window_duration_s": delay_tracker_window_duration_s,
        "delay_tracker_update_interval_s": delay_tracker_update_interval_s,
        "delay_tracker_search_min_ms": delay_tracker_search_min_ms,
        "delay_tracker_search_max_ms": delay_tracker_search_max_ms,
        "delay_tracker_search_step_ms": delay_tracker_search_step_ms,
        "delay_tracker_filter_alpha": delay_tracker_filter_alpha,
        "maximum_delay_change_per_update_ms": (
            maximum_delay_change_per_update_ms
        ),
        "delay_tracker_minimum_effective_samples": (
            delay_tracker_minimum_effective_samples
        ),
        "delay_tracker_minimum_excitation_score": (
            delay_tracker_minimum_excitation_score
        ),
        "tracker_input_splits": ["train"],
        "tracker_test_access": False,
        "tracker_validation_access": False,
        "true_delay_passed_to_tracker": False,
        "runtime_tracker_history_truth_columns": [],
        "global_delay_selection_splits": list(
            global_selection.delay_selection_splits
        ),
        "global_delay_test_access": False,
        "global_delay_true_delay_access": False,
        "global_fixed_estimated_delay_ms": (
            global_selection.estimated_delay_ms
        ),
        "global_fixed_search_boundary_hit": (
            global_selection.search_boundary_hit
        ),
        "tracker_update_count": int(len(tracking_history)),
        "tracker_valid_update_count": int(tracker_valid.sum()),
        "tracker_held_update_count": int((~tracker_valid).sum()),
        "tracker_delay_mean_absolute_error_ms": (
            float(np.mean(np.abs(finite_tracker_errors)))
            if len(finite_tracker_errors)
            else None
        ),
        "tracker_delay_p95_absolute_error_ms": (
            float(np.quantile(np.abs(finite_tracker_errors), 0.95))
            if len(finite_tracker_errors)
            else None
        ),
        "train_trajectory_ids": _trajectory_ids(raw_splits["train"]),
        "validation_trajectory_ids": _trajectory_ids(
            raw_splits["validation"]
        ),
        "test_trajectory_ids": _trajectory_ids(raw_splits["test"]),
        "four_method_definitions": {
            "row_index_alignment": (
                "intentionally wrong baseline pairing force row t with state row t"
            ),
            "global_fixed_delay": (
                "Stage 4.5A offline-only bidirectional fixed-delay compensation"
            ),
            "causal_history_latest": (
                "latest arrived wrench without timestamp compensation"
            ),
            "causal_buffered_matching": (
                "arrival-time query of already buffered state using reliable "
                "wrench sample timestamp, with estimated-delay fallback"
            ),
        },
        "primary_buffered_timestamp_source": (
            "reliable simulated wrench_sample_timestamp_s"
        ),
        "tracker_delay_fallback_exercised_in_primary_experiments": False,
        "tracker_delay_fallback_unit_tested": True,
        "buffered_result_interpretation": (
            "Primary buffered RMSE validates reliable sample timestamps plus "
            "causal state-history matching. Delay-fallback behavior is tested "
            "separately and is not claimed as equivalent."
        ),
        "tracker_score_field_compatibility_note": (
            "best_validation_rmse_nm and second_best_validation_rmse_nm are "
            "required compatibility field names; validation_score_unit records "
            "ms_timestamp_residual for the default timestamp scorer."
        ),
        "offline_only_warning": (
            "global_fixed_delay uses future samples and is not an online "
            "controller; causal_buffered_matching uses history only."
        ),
        "long_gap_interpolation_allowed": False,
        "stale_or_dropout_revalidated": False,
        "true_delay_columns_for_evaluation_only": [
            "true_delay_s",
            "generated_base_delay_s",
        ],
        "estimator_input_columns": list(ESTIMATOR_INPUT_COLUMNS),
        "scenario_metadata": application.metadata,
        "scope_excludes": [
            "model_mismatch",
            "trajectory_optimization",
            "PINN",
            "MPC",
            "tactile",
            "real_robot_motion",
            "real_robot_control",
            "real_robot_safety_thresholds",
        ],
        "disclaimer": (
            "Software-only virtual-data causal alignment validation; not a "
            "real patient estimate, online controller, or robot safety result."
        ),
    }
    _write_json(destination / "metadata.json", metadata)
    figure_paths: list[Path] = []
    if make_plots:
        figure_paths = generate_variable_delay_visualizations(
            raw,
            matched,
            tracking_history,
            comparison,
            parameter_estimates,
            subject_id,
            delay_scenario,
            destination,
        )
    return {
        "subject_id": subject_id,
        "delay_scenario": delay_scenario,
        "raw_dataset": raw,
        "matched_dataset": matched,
        "delay_tracking_history": tracking_history,
        "delay_search_windows": search_windows,
        "parameter_estimates": parameter_estimates,
        "method_comparison": comparison,
        "predictions": predictions,
        "global_delay_selection": global_selection,
        "metadata": metadata,
        "output_dir": destination,
        "figure_paths": figure_paths,
    }


def run_variable_delay_experiments(
    subjects: Sequence[str],
    scenarios: Sequence[str],
    *,
    output_root: str | Path = variable_delay_data_dir,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int = variable_delay_random_seed,
    make_plots: bool = True,
    loss: str = identification_loss,
) -> pd.DataFrame:
    """批量运行并在根目录保存四方法聚合摘要。"""

    results: list[pd.DataFrame] = []
    clean_cache: dict[str, pd.DataFrame] = {}
    for subject_id in subjects:
        subject = get_dynamic_subject(subject_id)
        clean_cache[subject_id] = build_identification_dataset(
            subject,
            "clean",
            sampling_frequency_hz=sampling_frequency_hz,
        )
        for scenario in scenarios:
            result = run_single_variable_delay_experiment(
                subject_id,
                scenario,
                clean_dataset=clean_cache[subject_id],
                output_root=output_root,
                sampling_frequency_hz=sampling_frequency_hz,
                random_seed=random_seed,
                make_plots=make_plots,
                loss=loss,
            )
            results.append(result["method_comparison"])
    aggregate = pd.concat(results, ignore_index=True)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(
        destination / "all_variable_delay_summary.csv",
        index=False,
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4.5B software-only variable wrench-delay alignment replay."
        )
    )
    parser.add_argument("subject_id", nargs="?")
    parser.add_argument("delay_scenario", nargs="?")
    parser.add_argument(
        "--all-baseline",
        action="store_true",
        help="run baseline across all ten variable-delay scenarios",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all four virtual subjects across the six core scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=variable_delay_data_dir,
    )
    parser.add_argument(
        "--sampling-frequency-hz",
        type=float,
        default=dynamic_sampling_frequency_hz,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=variable_delay_random_seed,
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--loss", default=identification_loss)
    args = parser.parse_args()

    if args.all_baseline and args.all:
        parser.error("choose only one of --all-baseline and --all.")
    if args.all_baseline:
        subjects = ("baseline",)
        scenarios = tuple(variable_delay_scenarios)
    elif args.all:
        subjects = tuple(DYNAMIC_SUBJECTS)
        scenarios = tuple(variable_delay_core_scenarios)
    else:
        if args.subject_id is None or args.delay_scenario is None:
            parser.error(
                "provide subject_id and delay_scenario, or use "
                "--all-baseline/--all."
            )
        subjects = (args.subject_id,)
        scenarios = (args.delay_scenario,)
    summary = run_variable_delay_experiments(
        subjects,
        scenarios,
        output_root=args.output_dir,
        sampling_frequency_hz=args.sampling_frequency_hz,
        random_seed=args.random_seed,
        make_plots=not args.no_plots,
        loss=args.loss,
    )
    columns = [
        "subject_id",
        "delay_scenario",
        "alignment_method",
        "test_rmse_nm",
        "mean_parameter_error_percent",
        "matching_acceptance_rate",
    ]
    print(summary.loc[:, columns].to_string(index=False))
    print(Path(args.output_dir))


if __name__ == "__main__":
    main()
