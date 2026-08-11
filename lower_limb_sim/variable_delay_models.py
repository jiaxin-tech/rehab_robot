"""阶段 4.5B 可复现的变化 wrench 延迟、抖动、长尾和缺失模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    max_alignment_interpolation_gap_s,
    variable_delay_random_seed,
    variable_delay_scenarios,
)
from .observation_model import joint_torque_from_endpoint_force

TRAJECTORY_GROUP_COLUMNS = ("trajectory_family", "speed_profile")
VARIABLE_DELAY_SCENARIOS = tuple(variable_delay_scenarios)
TIME_TOLERANCE_S = 1e-12


@dataclass(frozen=True)
class VariableDelayApplicationResult:
    dataframe: pd.DataFrame
    metadata: dict[str, object]


def _append_reason(
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


def _delay_profile(
    time_s: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """返回基础延迟和长尾标记；全部时间单位为秒。"""

    count = len(time_s)
    long_tail = np.zeros(count, dtype=bool)
    if scenario == "fixed_16ms":
        delay = np.full(count, 0.016)
    elif scenario == "piecewise_delay":
        delay = np.select(
            [time_s < 4.0, time_s < 8.0],
            [0.008, 0.024],
            default=0.016,
        ).astype(float)
    elif scenario == "gradual_drift":
        duration = max(float(time_s[-1] - time_s[0]), np.finfo(float).eps)
        progress = (time_s - time_s[0]) / duration
        delay = 0.008 + 0.024 * progress
    elif scenario == "jitter_low":
        delay = np.clip(rng.normal(0.016, 0.002, count), 0.004, 0.040)
    elif scenario == "jitter_medium":
        delay = np.clip(rng.normal(0.024, 0.005, count), 0.004, 0.060)
    elif scenario == "bimodal_delay":
        slow_mode = rng.random(count) < 0.20
        delay = np.where(
            slow_mode,
            rng.normal(0.032, 0.0015, count),
            rng.normal(0.008, 0.0015, count),
        )
        delay = np.clip(delay, 0.003, 0.045)
    elif scenario == "long_tail":
        delay = np.clip(rng.normal(0.016, 0.002, count), 0.006, 0.032)
        long_tail = rng.random(count) < 0.025
        # A small upper shoulder beyond 100 ms exercises the configured age
        # rejection without changing the dominant requested 50–100 ms tail.
        delay[long_tail] = rng.uniform(0.050, 0.105, int(long_tail.sum()))
    elif scenario == "stale_freeze":
        delay = np.full(count, 0.016)
    elif scenario == "dropout_5pct":
        delay = np.full(count, 0.016)
    elif scenario == "combined_realistic":
        delay = np.clip(rng.normal(0.016, 0.0035, count), 0.004, 0.032)
        long_tail = rng.random(count) < 0.012
        delay[long_tail] = rng.uniform(0.032, 0.050, int(long_tail.sum()))
    else:
        raise ValueError(
            f"Unknown variable delay scenario {scenario!r}; choose one of "
            f"{', '.join(VARIABLE_DELAY_SCENARIOS)}."
        )
    return np.asarray(delay, dtype=float), long_tail


def _valid_linear_force_interpolation(
    time_s: np.ndarray,
    fx: np.ndarray,
    fz: np.ndarray,
    source_valid: np.ndarray,
    target_s: np.ndarray,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = np.flatnonzero(source_valid)
    output_fx = np.full(len(target_s), np.nan)
    output_fz = np.full(len(target_s), np.nan)
    output_valid = np.zeros(len(target_s), dtype=bool)
    if not len(valid_indices):
        return output_fx, output_fz, output_valid

    valid_time = time_s[valid_indices]
    valid_fx = fx[valid_indices]
    valid_fz = fz[valid_indices]
    insertion = np.searchsorted(valid_time, target_s, side="left")
    clipped = np.minimum(insertion, len(valid_time) - 1)
    exact = (
        insertion < len(valid_time)
    ) & (
        np.abs(valid_time[clipped] - target_s) <= TIME_TOLERANCE_S
    )
    if exact.any():
        source = clipped[exact]
        output_fx[exact] = valid_fx[source]
        output_fz[exact] = valid_fz[source]
        output_valid[exact] = True

    interpolate = ~exact & (insertion > 0) & (insertion < len(valid_time))
    if interpolate.any():
        right = insertion[interpolate]
        left = right - 1
        gap = valid_time[right] - valid_time[left]
        accepted_local = gap <= float(max_gap_s) + TIME_TOLERANCE_S
        targets = np.flatnonzero(interpolate)[accepted_local]
        if len(targets):
            accepted_left = left[accepted_local]
            accepted_right = right[accepted_local]
            alpha = (
                target_s[targets] - valid_time[accepted_left]
            ) / (
                valid_time[accepted_right] - valid_time[accepted_left]
            )
            output_fx[targets] = valid_fx[accepted_left] + alpha * (
                valid_fx[accepted_right] - valid_fx[accepted_left]
            )
            output_fz[targets] = valid_fz[accepted_left] + alpha * (
                valid_fz[accepted_right] - valid_fz[accepted_left]
            )
            output_valid[targets] = True
    return output_fx, output_fz, output_valid


def _freeze_intervals(
    time_s: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float,
    duration_range_s: tuple[float, float],
    force_one: bool,
) -> list[np.ndarray]:
    if len(time_s) < 6 or (not force_one and rng.random() > probability):
        return []
    dt = float(np.median(np.diff(time_s)))
    minimum = max(2, int(np.ceil(duration_range_s[0] / dt)))
    maximum = max(minimum, int(np.ceil(duration_range_s[1] / dt)))
    length = min(int(rng.integers(minimum, maximum + 1)), len(time_s) - 2)
    start = int(rng.integers(1, max(2, len(time_s) - length)))
    return [np.arange(start, start + length, dtype=int)]


def apply_variable_delay_scenario(
    clean_dataframe: pd.DataFrame,
    scenario: str,
    *,
    random_seed: int = variable_delay_random_seed,
    maximum_force_interpolation_gap_s: float = (
        max_alignment_interpolation_gap_s
    ),
    L1_m: float = L1,
    L2_m: float = L2,
) -> VariableDelayApplicationResult:
    """把 clean 力流变成按到达时刻记录的变化延迟观测。

    每行状态时刻和 wrench 到达时刻相同；wrench 的物理测量时刻为
    ``arrival - delay``。自动算法不得读取保存的 ``true_delay_s``，
    但可以读取独立设备本应提供的 ``wrench_sample_timestamp_s``。
    """

    if scenario not in VARIABLE_DELAY_SCENARIOS:
        raise ValueError(
            f"Unknown variable delay scenario {scenario!r}; choose one of "
            f"{', '.join(VARIABLE_DELAY_SCENARIOS)}."
        )
    if not isinstance(random_seed, (int, np.integer)):
        raise ValueError("random_seed must be an integer.")
    required = {
        "time_s",
        "trajectory_family",
        "speed_profile",
        "q_hip_rad",
        "q_knee_rad",
        "fx_observed_n",
        "fz_observed_n",
        "sample_valid",
        "force_mapping_valid",
        "invalid_reason",
    }
    missing = required.difference(clean_dataframe.columns)
    if missing:
        raise ValueError(f"clean variable-delay data is missing: {sorted(missing)}")

    dataframe = clean_dataframe.copy(deep=True).reset_index(drop=True)
    dataframe.attrs.clear()
    rng = np.random.default_rng(int(random_seed))
    count = len(dataframe)
    arrival = dataframe["time_s"].to_numpy(dtype=float)
    sample_timestamp = np.full(count, np.nan)
    observed_fx = np.full(count, np.nan)
    observed_fz = np.full(count, np.nan)
    source_available = np.zeros(count, dtype=bool)
    is_long_tail = np.zeros(count, dtype=bool)
    is_dropout = np.zeros(count, dtype=bool)
    is_stale = np.zeros(count, dtype=bool)
    freeze_duration = np.zeros(count, dtype=float)
    base_delay = np.full(count, np.nan)
    forced_freeze_created = False

    for _, group in dataframe.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        indices = group.index.to_numpy(dtype=int)
        time_s = group["time_s"].to_numpy(dtype=float)
        delay, group_long_tail = _delay_profile(time_s, scenario, rng)
        target_source_time = time_s - delay
        clean_valid = (
            group["sample_valid"].astype(bool).to_numpy()
            & group["force_mapping_valid"].astype(bool).to_numpy()
            & np.isfinite(group["fx_observed_n"].to_numpy(dtype=float))
            & np.isfinite(group["fz_observed_n"].to_numpy(dtype=float))
        )
        fx, fz, available = _valid_linear_force_interpolation(
            time_s,
            group["fx_observed_n"].to_numpy(dtype=float),
            group["fz_observed_n"].to_numpy(dtype=float),
            clean_valid,
            target_source_time,
            maximum_force_interpolation_gap_s,
        )

        group_dropout = np.zeros(len(group), dtype=bool)
        if scenario == "dropout_5pct":
            group_dropout = rng.random(len(group)) < 0.05
        elif scenario == "combined_realistic":
            group_dropout = rng.random(len(group)) < 0.03

        group_stale = np.zeros(len(group), dtype=bool)
        group_freeze_duration = np.zeros(len(group), dtype=float)
        intervals: list[np.ndarray] = []
        if scenario == "stale_freeze":
            intervals = _freeze_intervals(
                time_s,
                rng,
                probability=0.45,
                duration_range_s=(0.10, 0.25),
                force_one=not forced_freeze_created,
            )
        elif scenario == "combined_realistic":
            intervals = _freeze_intervals(
                time_s,
                rng,
                probability=0.12,
                duration_range_s=(0.10, 0.25),
                force_one=False,
            )
        for interval in intervals:
            if not len(interval) or interval[0] == 0:
                continue
            held_index = int(interval[0] - 1)
            group_stale[interval] = True
            target_source_time[interval] = target_source_time[held_index]
            fx[interval] = fx[held_index]
            fz[interval] = fz[held_index]
            available[interval] = available[held_index]
            group_freeze_duration[interval] = (
                time_s[interval] - time_s[held_index]
            )
            forced_freeze_created = True

        sample_timestamp[indices] = target_source_time
        observed_fx[indices] = fx
        observed_fz[indices] = fz
        source_available[indices] = available
        is_long_tail[indices] = group_long_tail
        is_dropout[indices] = group_dropout
        is_stale[indices] = group_stale
        freeze_duration[indices] = group_freeze_duration
        base_delay[indices] = delay

    observed_fx[is_dropout] = np.nan
    observed_fz[is_dropout] = np.nan
    true_delay = arrival - sample_timestamp
    raw_state_valid = dataframe["force_mapping_valid"].astype(bool).to_numpy()
    final_valid = (
        source_available
        & raw_state_valid
        & ~is_dropout
        & ~is_stale
        & np.isfinite(observed_fx)
        & np.isfinite(observed_fz)
    )

    dataframe["fx_observed_n"] = observed_fx
    dataframe["fz_observed_n"] = observed_fz
    dataframe["state_timestamp_s"] = arrival
    dataframe["wrench_arrival_timestamp_s"] = arrival
    dataframe["wrench_timestamp_s"] = arrival
    dataframe["wrench_sample_timestamp_s"] = sample_timestamp
    dataframe["true_delay_s"] = true_delay
    dataframe["generated_base_delay_s"] = base_delay
    dataframe["wrench_age_s"] = true_delay
    dataframe["state_wrench_skew_s"] = arrival - sample_timestamp
    dataframe["delay_scenario"] = scenario
    dataframe["noise_scenario"] = f"variable_delay/{scenario}"
    dataframe["is_dropout"] = is_dropout
    dataframe["is_stale"] = is_stale
    dataframe["wrench_is_stale"] = is_stale
    dataframe["is_long_tail"] = is_long_tail
    dataframe["freeze_duration_s"] = freeze_duration
    dataframe["sample_valid"] = final_valid
    dataframe["invalid_reason"] = ""
    _append_reason(dataframe, ~source_available, "wrench_source_history_unavailable")
    _append_reason(dataframe, is_dropout, "wrench_dropout")
    _append_reason(dataframe, is_stale, "stale_wrench_freeze")
    _append_reason(dataframe, ~raw_state_valid, "invalid_force_mapping")

    tau_hip, tau_knee = joint_torque_from_endpoint_force(
        dataframe["q_hip_rad"].to_numpy(dtype=float),
        dataframe["q_knee_rad"].to_numpy(dtype=float),
        observed_fx,
        observed_fz,
        L1_m,
        L2_m,
    )
    dataframe["tau_measured_hip_nm"] = tau_hip
    dataframe["tau_measured_knee_nm"] = tau_knee
    dataframe["force_magnitude_observed_n"] = np.hypot(
        observed_fx,
        observed_fz,
    )
    metadata = {
        "delay_scenario": scenario,
        "random_seed": int(random_seed),
        "positive_delay_definition": (
            "wrench_arrival_timestamp_s - wrench_sample_timestamp_s"
        ),
        "timestamp_clock": "trajectory-local simulated monotonic seconds",
        "future_wrench_used": False,
        "true_delay_available_to_estimators": False,
        "sample_timestamp_is_observed_simulated_device_time": True,
        "dropout_samples": int(is_dropout.sum()),
        "stale_samples": int(is_stale.sum()),
        "long_tail_samples": int(is_long_tail.sum()),
        "source_history_unavailable_samples": int((~source_available).sum()),
        "maximum_true_delay_s": float(np.nanmax(true_delay)),
        "angle_definition": "theta_shank = q_hip - q_knee",
    }
    dataframe.attrs.update(metadata)
    return VariableDelayApplicationResult(dataframe, metadata)

