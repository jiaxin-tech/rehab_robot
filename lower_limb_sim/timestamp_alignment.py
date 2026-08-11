"""阶段 4.5A：wrench 时间戳、延迟合成和受限时间轴对齐。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import L1, L2, max_alignment_interpolation_gap_s
from .observation_model import joint_torque_from_endpoint_force

TRAJECTORY_GROUP_COLUMNS = ("trajectory_family", "speed_profile")
ALIGNMENT_MODES = ("causal_history", "offline_only")
TIME_TOLERANCE_S = 1e-12


@dataclass(frozen=True)
class TimestampAlignmentResult:
    dataframe: pd.DataFrame
    metadata: dict[str, object]


def _append_reason(
    reasons: np.ndarray,
    mask: np.ndarray,
    reason: str,
) -> None:
    selected = np.asarray(mask, dtype=bool)
    current = reasons[selected].astype(str)
    reasons[selected] = np.where(
        current == "",
        reason,
        np.char.add(np.char.add(current, ";"), reason),
    )


def _validate_delay(delay_s: float, *, allow_negative: bool) -> float:
    delay = float(delay_s)
    if not np.isfinite(delay):
        raise ValueError("delay_s must be finite.")
    if not allow_negative and delay < 0.0:
        raise ValueError("synthetic physical wrench delay cannot be negative.")
    return delay


def _validate_max_gap(max_gap_s: float) -> float:
    gap = float(max_gap_s)
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError("max_interpolation_gap_s must be finite and positive.")
    return gap


def _offline_linear_interpolation(
    source_effective_time_s: np.ndarray,
    source_available_time_s: np.ndarray,
    source_fx: np.ndarray,
    source_fz: np.ndarray,
    source_valid: np.ndarray,
    target_time_s: np.ndarray,
    max_gap_s: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """双向线性插值；不外推，也不跨越超过 max_gap 的无效区间。"""

    valid_indices = np.flatnonzero(source_valid)
    count = len(target_time_s)
    fx = np.full(count, np.nan)
    fz = np.full(count, np.nan)
    valid = np.zeros(count, dtype=bool)
    gap = np.full(count, np.nan)
    future_lookahead = np.full(count, np.nan)
    if not len(valid_indices):
        return fx, fz, valid, gap, future_lookahead

    effective = source_effective_time_s[valid_indices]
    available = source_available_time_s[valid_indices]
    if np.any(np.diff(effective) <= 0.0):
        raise ValueError("valid wrench timestamps must be strictly increasing.")
    source_fx_valid = source_fx[valid_indices]
    source_fz_valid = source_fz[valid_indices]
    insertion = np.searchsorted(effective, target_time_s, side="left")
    clipped_right = np.minimum(insertion, len(effective) - 1)
    exact = (
        insertion < len(effective)
    ) & (
        np.abs(effective[clipped_right] - target_time_s)
        <= TIME_TOLERANCE_S
    )
    if exact.any():
        right = clipped_right[exact]
        fx[exact] = source_fx_valid[right]
        fz[exact] = source_fz_valid[right]
        valid[exact] = True
        gap[exact] = 0.0
        future_lookahead[exact] = np.maximum(
            available[right] - target_time_s[exact],
            0.0,
        )

    interpolate = ~exact & (insertion > 0) & (insertion < len(effective))
    if interpolate.any():
        right = insertion[interpolate]
        left = right - 1
        bracket_gap = effective[right] - effective[left]
        allowed = bracket_gap <= max_gap_s + TIME_TOLERANCE_S
        target_indices = np.flatnonzero(interpolate)
        gap[target_indices] = bracket_gap
        accepted_targets = target_indices[allowed]
        if len(accepted_targets):
            accepted_left = left[allowed]
            accepted_right = right[allowed]
            alpha = (
                target_time_s[accepted_targets]
                - effective[accepted_left]
            ) / (effective[accepted_right] - effective[accepted_left])
            fx[accepted_targets] = (
                source_fx_valid[accepted_left]
                + alpha
                * (
                    source_fx_valid[accepted_right]
                    - source_fx_valid[accepted_left]
                )
            )
            fz[accepted_targets] = (
                source_fz_valid[accepted_left]
                + alpha
                * (
                    source_fz_valid[accepted_right]
                    - source_fz_valid[accepted_left]
                )
            )
            valid[accepted_targets] = True
            future_lookahead[accepted_targets] = np.maximum(
                np.maximum(
                    available[accepted_left],
                    available[accepted_right],
                )
                - target_time_s[accepted_targets],
                0.0,
            )
    return fx, fz, valid, gap, future_lookahead


def _causal_history_alignment(
    source_effective_time_s: np.ndarray,
    source_available_time_s: np.ndarray,
    source_fx: np.ndarray,
    source_fz: np.ndarray,
    source_valid: np.ndarray,
    target_time_s: np.ndarray,
    max_gap_s: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """只使用在目标时刻已经到达的最新历史 wrench，禁止未来样本。"""

    valid_indices = np.flatnonzero(source_valid)
    count = len(target_time_s)
    fx = np.full(count, np.nan)
    fz = np.full(count, np.nan)
    valid = np.zeros(count, dtype=bool)
    gap = np.full(count, np.nan)
    future_lookahead = np.zeros(count)
    if not len(valid_indices):
        return fx, fz, valid, gap, future_lookahead

    effective = source_effective_time_s[valid_indices]
    available = source_available_time_s[valid_indices]
    if np.any(np.diff(effective) <= 0.0) or np.any(np.diff(available) <= 0.0):
        raise ValueError("valid wrench timestamps must be strictly increasing.")
    available_count = np.searchsorted(
        available,
        target_time_s + TIME_TOLERANCE_S,
        side="right",
    )
    effective_count = np.searchsorted(
        effective,
        target_time_s + TIME_TOLERANCE_S,
        side="right",
    )
    selected = np.minimum(available_count, effective_count) - 1
    has_history = selected >= 0
    if has_history.any():
        targets = np.flatnonzero(has_history)
        source = selected[has_history]
        hold_age = target_time_s[targets] - effective[source]
        gap[targets] = np.maximum(hold_age, 0.0)
        allowed = (
            hold_age >= -TIME_TOLERANCE_S
        ) & (
            hold_age <= max_gap_s + TIME_TOLERANCE_S
        )
        accepted = targets[allowed]
        accepted_source = source[allowed]
        fx[accepted] = source_fx[valid_indices[accepted_source]]
        fz[accepted] = source_fz[valid_indices[accepted_source]]
        valid[accepted] = True
    return fx, fz, valid, gap, future_lookahead


def _recompute_measured_torque(
    dataframe: pd.DataFrame,
    L1_m: float,
    L2_m: float,
) -> None:
    tau_hip, tau_knee = joint_torque_from_endpoint_force(
        dataframe["q_hip_rad"].to_numpy(dtype=float),
        dataframe["q_knee_rad"].to_numpy(dtype=float),
        dataframe["fx_observed_n"].to_numpy(dtype=float),
        dataframe["fz_observed_n"].to_numpy(dtype=float),
        L1_m,
        L2_m,
    )
    dataframe["tau_measured_hip_nm"] = tau_hip
    dataframe["tau_measured_knee_nm"] = tau_knee
    dataframe["force_magnitude_observed_n"] = np.hypot(
        dataframe["fx_observed_n"].to_numpy(dtype=float),
        dataframe["fz_observed_n"].to_numpy(dtype=float),
    )


def synthesize_delayed_wrench_dataset(
    clean_dataframe: pd.DataFrame,
    delay_s: float,
    *,
    max_interpolation_gap_s: float = max_alignment_interpolation_gap_s,
    L1_m: float = L1,
    L2_m: float = L2,
) -> pd.DataFrame:
    """生成 ``F_obs(t)=F_clean(t-delay)`` 的虚拟原始记录。

    ``wrench_timestamp_s`` 是主机记录该 wrench 数据行的原始单调时间戳，
    与 ``state_timestamp_s`` 使用同一仿真时钟。``wrench_age_s`` 保存虚拟
    信号内容年龄，但自动延迟估计入口会删除该字段以及全部 attrs。
    """

    delay = _validate_delay(delay_s, allow_negative=False)
    max_gap = _validate_max_gap(max_interpolation_gap_s)
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
    }
    missing = required.difference(clean_dataframe.columns)
    if missing:
        raise ValueError(f"clean delay dataset is missing: {sorted(missing)}")

    dataframe = clean_dataframe.copy(deep=True).reset_index(drop=True)
    dataframe.attrs.clear()
    if "wrench_delay_s" in dataframe:
        dataframe = dataframe.drop(columns=["wrench_delay_s"])
    dataframe["noise_scenario"] = "timestamp_alignment_experiment"
    dataframe["state_timestamp_s"] = dataframe["time_s"].to_numpy(dtype=float)
    dataframe["wrench_timestamp_s"] = dataframe["time_s"].to_numpy(dtype=float)
    dataframe["wrench_age_s"] = delay
    dataframe["state_wrench_skew_s"] = (
        dataframe["state_timestamp_s"] - dataframe["wrench_timestamp_s"]
    )
    dataframe["raw_invalid_reason"] = dataframe["invalid_reason"].fillna("")
    delayed_fx = np.full(len(dataframe), np.nan)
    delayed_fz = np.full(len(dataframe), np.nan)
    delayed_valid = np.zeros(len(dataframe), dtype=bool)
    synthesis_gap = np.full(len(dataframe), np.nan)

    for _, group in dataframe.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        indices = group.index.to_numpy(dtype=int)
        time_s = group["time_s"].to_numpy(dtype=float)
        source_valid = (
            group["sample_valid"].astype(bool).to_numpy()
            & group["force_mapping_valid"].astype(bool).to_numpy()
            & np.isfinite(group["fx_observed_n"].to_numpy(dtype=float))
            & np.isfinite(group["fz_observed_n"].to_numpy(dtype=float))
        )
        if "wrench_is_stale" in group:
            source_valid &= ~group["wrench_is_stale"].astype(bool).to_numpy()
        target_source_time = time_s - delay
        fx, fz, valid, gap, _ = _offline_linear_interpolation(
            time_s,
            time_s,
            group["fx_observed_n"].to_numpy(dtype=float),
            group["fz_observed_n"].to_numpy(dtype=float),
            source_valid,
            target_source_time,
            max_gap,
        )
        delayed_fx[indices] = fx
        delayed_fz[indices] = fz
        delayed_valid[indices] = valid
        synthesis_gap[indices] = gap

    state_valid = (
        dataframe["force_mapping_valid"].astype(bool).to_numpy()
        & np.isfinite(
            dataframe[
                [
                    "q_hip_rad",
                    "q_knee_rad",
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1)
    )
    final_valid = delayed_valid & state_valid
    dataframe["fx_observed_n"] = delayed_fx
    dataframe["fz_observed_n"] = delayed_fz
    dataframe["sample_valid"] = final_valid
    dataframe["alignment_valid"] = final_valid
    dataframe["alignment_mode"] = "raw_uncompensated"
    dataframe["alignment_offline_only"] = False
    dataframe["alignment_used_future"] = False
    dataframe["alignment_gap_s"] = synthesis_gap
    dataframe["alignment_invalid_reason"] = ""
    missing_history = ~delayed_valid
    dataframe.loc[
        missing_history,
        "alignment_invalid_reason",
    ] = "synthetic_wrench_delay_no_history_or_gap"
    dataframe["invalid_reason"] = dataframe["alignment_invalid_reason"]
    _recompute_measured_torque(dataframe, L1_m, L2_m)
    dataframe.attrs.update(
        {
            "timestamp_clock": "trajectory-local simulated monotonic seconds",
            "timestamp_semantics": (
                "state/wrench timestamps are raw acquisition timestamps; "
                "wrench_age is evaluation-only signal content age"
            ),
            "true_delay_available_to_automatic_estimator": False,
        }
    )
    return dataframe


def align_wrench_to_state_timestamps(
    dataframe: pd.DataFrame,
    assumed_delay_s: float,
    *,
    mode: str = "offline_only",
    max_interpolation_gap_s: float = max_alignment_interpolation_gap_s,
    evaluation_margin_s: float = 0.0,
    L1_m: float = L1,
    L2_m: float = L2,
) -> TimestampAlignmentResult:
    """将 wrench 流对齐到状态时间轴。

    正候选表示 wrench 落后。有效物理时间定义为
    ``wrench_timestamp_s - assumed_delay_s``。``offline_only`` 可使用状态
    时刻之后才到达的 wrench 样本；``causal_history`` 严格只使用当前和
    历史已到达样本，并以不超过配置间隔的保持值输出。
    """

    delay = _validate_delay(assumed_delay_s, allow_negative=True)
    max_gap = _validate_max_gap(max_interpolation_gap_s)
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"mode must be one of {ALIGNMENT_MODES}.")
    if (
        not np.isfinite(evaluation_margin_s)
        or evaluation_margin_s < 0.0
    ):
        raise ValueError("evaluation_margin_s must be finite and non-negative.")
    required = {
        "state_timestamp_s",
        "wrench_timestamp_s",
        "trajectory_family",
        "speed_profile",
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
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"timestamp alignment dataframe is missing: {sorted(missing)}")

    aligned = dataframe.copy(deep=True).reset_index(drop=True)
    aligned.attrs.clear()
    if "fx_raw_observed_n" not in aligned:
        aligned["fx_raw_observed_n"] = aligned["fx_observed_n"]
        aligned["fz_raw_observed_n"] = aligned["fz_observed_n"]
    raw_fx = aligned["fx_raw_observed_n"].to_numpy(dtype=float)
    raw_fz = aligned["fz_raw_observed_n"].to_numpy(dtype=float)
    output_fx = np.full(len(aligned), np.nan)
    output_fz = np.full(len(aligned), np.nan)
    output_valid = np.zeros(len(aligned), dtype=bool)
    output_gap = np.full(len(aligned), np.nan)
    output_lookahead = np.full(len(aligned), np.nan)
    outside_common_margin = np.zeros(len(aligned), dtype=bool)

    for _, group in aligned.groupby(
        list(TRAJECTORY_GROUP_COLUMNS),
        sort=False,
    ):
        indices = group.index.to_numpy(dtype=int)
        state_time = group["state_timestamp_s"].to_numpy(dtype=float)
        raw_wrench_time = group["wrench_timestamp_s"].to_numpy(dtype=float)
        if (
            not np.isfinite(state_time).all()
            or not np.isfinite(raw_wrench_time).all()
            or np.any(np.diff(state_time) <= 0.0)
            or np.any(np.diff(raw_wrench_time) <= 0.0)
        ):
            raise ValueError("timestamps must be finite and strictly increasing.")
        source_valid = (
            group["sample_valid"].astype(bool).to_numpy()
            & np.isfinite(raw_fx[indices])
            & np.isfinite(raw_fz[indices])
        )
        if "wrench_is_stale" in group:
            source_valid &= ~group["wrench_is_stale"].astype(bool).to_numpy()
        effective_time = raw_wrench_time - delay
        if mode == "offline_only":
            result = _offline_linear_interpolation(
                effective_time,
                raw_wrench_time,
                raw_fx[indices],
                raw_fz[indices],
                source_valid,
                state_time,
                max_gap,
            )
        else:
            result = _causal_history_alignment(
                effective_time,
                raw_wrench_time,
                raw_fx[indices],
                raw_fz[indices],
                source_valid,
                state_time,
                max_gap,
            )
        fx, fz, valid, gap, lookahead = result
        output_fx[indices] = fx
        output_fz[indices] = fz
        output_valid[indices] = valid
        output_gap[indices] = gap
        output_lookahead[indices] = lookahead
        if evaluation_margin_s > 0.0:
            outside_common_margin[indices] = (
                (state_time < state_time[0] + evaluation_margin_s - TIME_TOLERANCE_S)
                | (
                    state_time
                    > state_time[-1] - evaluation_margin_s + TIME_TOLERANCE_S
                )
            )

    state_valid = (
        aligned["force_mapping_valid"].astype(bool).to_numpy()
        & np.isfinite(
            aligned[
                [
                    "q_hip_rad",
                    "q_knee_rad",
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1)
    )
    final_valid = output_valid & state_valid & ~outside_common_margin
    reasons = np.full(len(aligned), "", dtype=object)
    _append_reason(
        reasons,
        ~output_valid & np.isnan(output_gap),
        "alignment_no_bracketing_sample",
    )
    _append_reason(
        reasons,
        ~output_valid & np.isfinite(output_gap),
        "alignment_gap_exceeded",
    )
    _append_reason(reasons, ~state_valid, "invalid_state_or_force_mapping")
    _append_reason(reasons, outside_common_margin, "common_search_margin")

    aligned["fx_observed_n"] = output_fx
    aligned["fz_observed_n"] = output_fz
    aligned["sample_valid"] = final_valid
    aligned["alignment_valid"] = final_valid
    aligned["alignment_mode"] = mode
    aligned["alignment_offline_only"] = mode == "offline_only"
    aligned["alignment_used_future"] = output_lookahead > TIME_TOLERANCE_S
    aligned["alignment_future_lookahead_s"] = output_lookahead
    aligned["alignment_gap_s"] = output_gap
    aligned["alignment_timestamp_s"] = aligned["state_timestamp_s"]
    aligned["wrench_effective_timestamp_s"] = (
        aligned["wrench_timestamp_s"] - delay
    )
    aligned["applied_delay_compensation_s"] = delay
    aligned["alignment_invalid_reason"] = reasons.astype(str)
    aligned["invalid_reason"] = aligned["alignment_invalid_reason"]
    _recompute_measured_torque(aligned, L1_m, L2_m)
    metadata = {
        "alignment_mode": mode,
        "offline_only": mode == "offline_only",
        "causal": mode == "causal_history",
        "assumed_delay_s": delay,
        "positive_delay_definition": "F_obs(t) = F_true(t - delay)",
        "max_interpolation_gap_s": max_gap,
        "evaluation_margin_s": float(evaluation_margin_s),
        "valid_samples": int(final_valid.sum()),
        "invalid_samples": int((~final_valid).sum()),
        "future_samples_used": int(
            (aligned["alignment_used_future"] & final_valid).sum()
        ),
        "extrapolation_used": False,
        "cross_trajectory_interpolation_used": False,
    }
    aligned.attrs.update(metadata)
    return TimestampAlignmentResult(aligned, metadata)
