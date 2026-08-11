"""可复现的 wrench 噪声、时序异常和独立角度噪声模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import identification_random_seed

NOISE_SCENARIOS = (
    "clean",
    "force_noise_low",
    "force_noise_medium",
    "force_bias",
    "timing_delay_16ms",
    "timing_delay_32ms",
    "random_dropout_5pct",
    "stale_freeze",
    "combined_realistic",
    "advanced_angle_noise",
)
TRAJECTORY_GROUP_COLUMNS = ("trajectory_family", "speed_profile")


@dataclass(frozen=True)
class NoiseApplicationResult:
    dataframe: pd.DataFrame
    metadata: dict[str, object]


def _append_invalid_reason(
    dataframe: pd.DataFrame,
    mask: np.ndarray | pd.Series,
    reason: str,
) -> None:
    selected = np.asarray(mask, dtype=bool)
    current = dataframe.loc[selected, "invalid_reason"].fillna("").astype(str)
    dataframe.loc[selected, "invalid_reason"] = np.where(
        current.eq(""),
        reason,
        current + ";" + reason,
    )


def _finite_valid_force_mask(dataframe: pd.DataFrame) -> np.ndarray:
    return (
        dataframe["sample_valid"].astype(bool).to_numpy()
        & np.isfinite(dataframe["fx_observed_n"].to_numpy(dtype=float))
        & np.isfinite(dataframe["fz_observed_n"].to_numpy(dtype=float))
    )


def _apply_gaussian_force_noise(
    dataframe: pd.DataFrame,
    rng: np.random.Generator,
    standard_deviation_n: float,
) -> dict[str, float]:
    valid = _finite_valid_force_mask(dataframe)
    fx_noise = rng.normal(0.0, standard_deviation_n, valid.sum())
    fz_noise = rng.normal(0.0, standard_deviation_n, valid.sum())
    dataframe.loc[valid, "fx_observed_n"] += fx_noise
    dataframe.loc[valid, "fz_observed_n"] += fz_noise
    return {
        "configured_standard_deviation_n": standard_deviation_n,
        "realized_fx_mean_n": float(np.mean(fx_noise)) if len(fx_noise) else 0.0,
        "realized_fx_std_n": float(np.std(fx_noise)) if len(fx_noise) else 0.0,
        "realized_fz_mean_n": float(np.mean(fz_noise)) if len(fz_noise) else 0.0,
        "realized_fz_std_n": float(np.std(fz_noise)) if len(fz_noise) else 0.0,
    }


def _apply_causal_delay(
    dataframe: pd.DataFrame,
    delay_by_group_s: dict[tuple[str, str], float],
) -> dict[str, object]:
    delayed_fx = dataframe["fx_observed_n"].to_numpy(dtype=float).copy()
    delayed_fz = dataframe["fz_observed_n"].to_numpy(dtype=float).copy()
    unavailable = np.zeros(len(dataframe), dtype=bool)

    for keys, group in dataframe.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        key = (str(keys[0]), str(keys[1]))
        delay_s = float(delay_by_group_s[key])
        indices = group.index.to_numpy(dtype=int)
        time_s = group["time_s"].to_numpy(dtype=float)
        source_time = time_s - delay_s
        original_fx = group["fx_observed_n"].to_numpy(dtype=float)
        original_fz = group["fz_observed_n"].to_numpy(dtype=float)
        available = (
            source_time >= time_s[0]
        ) & np.isfinite(original_fx) & np.isfinite(original_fz)
        # np.interp 在 t-delay 处只使用当前时刻之前的源样本；首段不回填。
        delayed_fx[indices[available]] = np.interp(
            source_time[available],
            time_s,
            original_fx,
        )
        delayed_fz[indices[available]] = np.interp(
            source_time[available],
            time_s,
            original_fz,
        )
        unavailable[indices[~available]] = True
        dataframe.loc[indices, "wrench_delay_s"] = delay_s

    delayed_fx[unavailable] = np.nan
    delayed_fz[unavailable] = np.nan
    dataframe["fx_observed_n"] = delayed_fx
    dataframe["fz_observed_n"] = delayed_fz
    dataframe.loc[unavailable, "sample_valid"] = False
    _append_invalid_reason(dataframe, unavailable, "wrench_delay_no_past_sample")
    return {
        "delay_by_trajectory_s": {
            f"{family}/{speed}": delay
            for (family, speed), delay in delay_by_group_s.items()
        },
        "causal_unavailable_samples": int(unavailable.sum()),
        "future_fill_used": False,
    }


def _apply_dropout(
    dataframe: pd.DataFrame,
    rng: np.random.Generator,
    probability: float,
) -> dict[str, float | int]:
    candidates = _finite_valid_force_mask(dataframe)
    dropout = candidates & (rng.random(len(dataframe)) < probability)
    dataframe.loc[dropout, ["fx_observed_n", "fz_observed_n"]] = np.nan
    dataframe.loc[dropout, "sample_valid"] = False
    _append_invalid_reason(dataframe, dropout, "wrench_dropout")
    return {
        "configured_probability": probability,
        "dropped_samples": int(dropout.sum()),
        "realized_fraction_of_candidates": (
            float(dropout.sum() / candidates.sum()) if candidates.sum() else 0.0
        ),
    }


def _freeze_one_group(
    dataframe: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
    duration_range_s: tuple[float, float],
) -> np.ndarray:
    if len(indices) < 5:
        return np.empty(0, dtype=int)
    time = dataframe.loc[indices, "time_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(time)))
    minimum_samples = max(2, int(np.ceil(duration_range_s[0] / dt)))
    maximum_samples = max(minimum_samples, int(np.ceil(duration_range_s[1] / dt)))
    length = int(rng.integers(minimum_samples, maximum_samples + 1))
    length = min(length, len(indices) - 2)
    candidates = [
        offset
        for offset in range(1, len(indices) - length)
        if np.isfinite(
            dataframe.loc[
                indices[offset - 1],
                ["fx_observed_n", "fz_observed_n"],
            ].to_numpy(dtype=float)
        ).all()
    ]
    if not candidates:
        return np.empty(0, dtype=int)
    start = int(rng.choice(candidates))
    frozen = indices[start : start + length]
    held = dataframe.loc[
        indices[start - 1],
        ["fx_observed_n", "fz_observed_n"],
    ].to_numpy(dtype=float)
    dataframe.loc[frozen, "fx_observed_n"] = held[0]
    dataframe.loc[frozen, "fz_observed_n"] = held[1]
    return frozen


def _apply_stale_freeze(
    dataframe: pd.DataFrame,
    rng: np.random.Generator,
    *,
    group_probability: float,
    duration_range_s: tuple[float, float] = (0.20, 0.25),
) -> dict[str, object]:
    all_frozen: list[np.ndarray] = []
    groups = list(
        dataframe.groupby(list(TRAJECTORY_GROUP_COLUMNS), sort=False)
    )
    for _, group in groups:
        if rng.random() <= group_probability:
            frozen = _freeze_one_group(
                dataframe,
                group.index.to_numpy(dtype=int),
                rng,
                duration_range_s,
            )
            if len(frozen):
                all_frozen.append(frozen)
    # 固定种子下也保证该场景至少包含一个可审计冻结区间。
    if not all_frozen and groups:
        frozen = _freeze_one_group(
            dataframe,
            groups[0][1].index.to_numpy(dtype=int),
            rng,
            duration_range_s,
        )
        if len(frozen):
            all_frozen.append(frozen)

    frozen_mask = np.zeros(len(dataframe), dtype=bool)
    for indices in all_frozen:
        frozen_mask[indices] = True
    dataframe.loc[frozen_mask, "wrench_is_stale"] = True
    dataframe.loc[frozen_mask, "sample_valid"] = False
    _append_invalid_reason(dataframe, frozen_mask, "stale_wrench_freeze")
    return {
        "duration_range_s": list(duration_range_s),
        "freeze_intervals": len(all_frozen),
        "stale_samples": int(frozen_mask.sum()),
        "long_freeze_interpolated": False,
    }


def _apply_advanced_angle_noise(
    dataframe: pd.DataFrame,
    rng: np.random.Generator,
    noise_standard_deviation_deg: float = 0.15,
) -> dict[str, object]:
    sigma_rad = float(np.deg2rad(noise_standard_deviation_deg))
    for _, group in dataframe.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        indices = group.index.to_numpy(dtype=int)
        time = group["time_s"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(time)))
        count = len(indices)
        window = min(31, count if count % 2 else count - 1)
        if window < 5:
            raise ValueError("advanced angle noise needs at least five samples.")
        polynomial_order = min(3, window - 2)
        for joint in ("hip", "knee"):
            # 本分支只读取 q；不会读取仿真 ground-truth dq/ddq。
            noisy_q = (
                group[f"q_{joint}_rad"].to_numpy(dtype=float)
                + rng.normal(0.0, sigma_rad, count)
            )
            filtered_q = savgol_filter(
                noisy_q,
                window_length=window,
                polyorder=polynomial_order,
                deriv=0,
                delta=dt,
                mode="interp",
            )
            filtered_dq = savgol_filter(
                noisy_q,
                window_length=window,
                polyorder=polynomial_order,
                deriv=1,
                delta=dt,
                mode="interp",
            )
            filtered_ddq = savgol_filter(
                noisy_q,
                window_length=window,
                polyorder=polynomial_order,
                deriv=2,
                delta=dt,
                mode="interp",
            )
            dataframe.loc[indices, f"q_{joint}_rad"] = filtered_q
            dataframe.loc[indices, f"dq_{joint}_rad_s"] = filtered_dq
            dataframe.loc[indices, f"ddq_{joint}_rad_s2"] = filtered_ddq
    dataframe["angle_observation_source"] = (
        "noisy_q_savgol_filtered_and_differentiated"
    )
    return {
        "q_noise_standard_deviation_deg": noise_standard_deviation_deg,
        "filter": "Savitzky-Golay",
        "filter_mode": "offline symmetric smoothing with interpolated edges",
        "window_samples": 31,
        "polynomial_order": 3,
        "dq_ddq_source": "filtered_noisy_q_only",
        "ground_truth_dq_ddq_used": False,
    }


def apply_noise_scenario(
    clean_dataframe: pd.DataFrame,
    scenario: str,
    random_seed: int = identification_random_seed,
) -> NoiseApplicationResult:
    """对 clean 观测应用一个独立且可复现的场景。"""

    if scenario not in NOISE_SCENARIOS:
        raise ValueError(
            f"Unknown noise scenario {scenario!r}; choose one of: "
            f"{', '.join(NOISE_SCENARIOS)}."
        )
    dataframe = clean_dataframe.copy(deep=True).reset_index(drop=True)
    dataframe["invalid_reason"] = dataframe["invalid_reason"].fillna("").astype(str)
    dataframe["noise_scenario"] = scenario
    dataframe["wrench_is_stale"] = False
    dataframe["wrench_delay_s"] = 0.0
    # 这里的 wrench_timestamp_s 是主机记录到该 wrench 样本的原始时间戳；
    # wrench_age_s 表示虚拟信号内容相对该记录时刻的年龄。自动延迟估计器
    # 会显式删除 age/delay/场景元数据，只从运动与力的验证误差估计延迟。
    dataframe["state_timestamp_s"] = dataframe["time_s"].to_numpy(dtype=float)
    dataframe["wrench_timestamp_s"] = dataframe["time_s"].to_numpy(dtype=float)
    dataframe["wrench_age_s"] = 0.0
    dataframe["state_wrench_skew_s"] = (
        dataframe["state_timestamp_s"] - dataframe["wrench_timestamp_s"]
    )
    dataframe["angle_observation_source"] = "simulation_ground_truth_q_dq_ddq"
    rng = np.random.default_rng(random_seed)
    details: dict[str, object] = {}

    if scenario == "force_noise_low":
        details["force_noise"] = _apply_gaussian_force_noise(dataframe, rng, 0.5)
    elif scenario == "force_noise_medium":
        details["force_noise"] = _apply_gaussian_force_noise(dataframe, rng, 2.0)
    elif scenario == "force_bias":
        valid = _finite_valid_force_mask(dataframe)
        dataframe.loc[valid, "fx_observed_n"] += 1.0
        dataframe.loc[valid, "fz_observed_n"] -= 1.0
        details["force_bias_n"] = {"fx": 1.0, "fz": -1.0}
    elif scenario in {"timing_delay_16ms", "timing_delay_32ms"}:
        delay = 0.016 if scenario.endswith("16ms") else 0.032
        delay_by_group = {
            (str(keys[0]), str(keys[1])): delay
            for keys, _ in dataframe.groupby(
                list(TRAJECTORY_GROUP_COLUMNS),
                sort=False,
            )
        }
        details["timing_delay"] = _apply_causal_delay(
            dataframe,
            delay_by_group,
        )
        dataframe["wrench_age_s"] = delay
    elif scenario == "random_dropout_5pct":
        details["dropout"] = _apply_dropout(dataframe, rng, 0.05)
    elif scenario == "stale_freeze":
        details["stale_freeze"] = _apply_stale_freeze(
            dataframe,
            rng,
            group_probability=0.45,
        )
    elif scenario == "combined_realistic":
        delay_by_group = {
            (str(keys[0]), str(keys[1])): float(rng.uniform(0.016, 0.032))
            for keys, _ in dataframe.groupby(
                list(TRAJECTORY_GROUP_COLUMNS),
                sort=False,
            )
        }
        details["timing_delay"] = _apply_causal_delay(
            dataframe,
            delay_by_group,
        )
        for keys, group in dataframe.groupby(
            list(TRAJECTORY_GROUP_COLUMNS),
            sort=False,
        ):
            key = (str(keys[0]), str(keys[1]))
            dataframe.loc[group.index, "wrench_age_s"] = delay_by_group[key]
        details["force_noise"] = _apply_gaussian_force_noise(dataframe, rng, 0.75)
        details["dropout"] = _apply_dropout(dataframe, rng, 0.02)
        details["stale_freeze"] = _apply_stale_freeze(
            dataframe,
            rng,
            group_probability=0.12,
        )
    elif scenario == "advanced_angle_noise":
        details["angle_noise"] = _apply_advanced_angle_noise(dataframe, rng)

    metadata = {
        "scenario": scenario,
        "random_seed": int(random_seed),
        "details": details,
        "invalid_samples": int((~dataframe["sample_valid"].astype(bool)).sum()),
        "stale_samples": int(dataframe["wrench_is_stale"].sum()),
        "future_sample_fill_used": False,
    }
    return NoiseApplicationResult(dataframe=dataframe, metadata=metadata)
