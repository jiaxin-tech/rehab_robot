"""阶段 4.5B：因果滑动窗口 wrench 延迟跟踪。

正延迟继续表示 wrench 落后运动状态。跟踪器只把当前更新时刻及其之前的
观测交给评分器；它不是在线控制器，也不读取仿真 ``true_delay``、
``wrench_age``、场景名或受试者真值字段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_WINDOW_DURATION_S = 2.0
DEFAULT_UPDATE_INTERVAL_S = 0.5
DEFAULT_MINIMUM_DELAY_MS = -50.0
DEFAULT_MAXIMUM_DELAY_MS = 80.0
DEFAULT_DELAY_STEP_MS = 1.0
DEFAULT_SMOOTHING_ALPHA = 0.35
DEFAULT_MAXIMUM_DELAY_CHANGE_MS = 10.0

_TIME_TOLERANCE_S = 1e-12
_ALLOWED_SPLITS = {"train", "online"}
_TRACKER_INPUT_COLUMNS = (
    "trajectory_id",
    "trajectory_family",
    "speed_profile",
    "phase",
    "trajectory_sample_index",
    "dataset_split",
    "time_s",
    "stream_timestamp_s",
    "state_timestamp_s",
    "wrench_timestamp_s",
    "wrench_source_timestamp_s",
    "wrench_sample_timestamp_s",
    "wrench_arrival_timestamp_s",
    "sample_timestamp_s",
    "arrival_timestamp_s",
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
)
_REQUIRED_MOTION_COLUMNS = (
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
)

DelayScoringCallback = Callable[[pd.DataFrame, np.ndarray], object]


@dataclass(frozen=True)
class WindowedDelayEstimate:
    """一次窗口更新的结果。

    前十一项和 ``delay_value_held`` 是阶段 4.5B 的稳定输出接口。
    """

    estimated_delay_ms: float
    delay_confidence: float
    delay_update_valid: bool
    delay_update_reason: str
    search_boundary_hit: bool
    window_start_s: float
    window_end_s: float
    effective_sample_count: int
    excitation_score: float
    best_validation_rmse_nm: float
    second_best_validation_rmse_nm: float
    delay_value_held: bool
    maximum_delay_change_limited: bool
    raw_estimated_delay_ms: float
    smoothed_delay_proposal_ms: float
    previous_delay_ms: float
    applied_delay_change_ms: float
    best_to_second_rmse_margin_nm: float
    local_rmse_curvature: float
    dq_rms_rad_s: float
    ddq_rms_rad_s2: float
    low_excitation: bool
    low_confidence: bool
    candidate_count: int
    selected_candidate_index: int
    validation_score_unit: str
    score_source: str
    causal_history_only: bool
    update_sequence: int

    @property
    def maximum_delay_change_limited_flag(self) -> bool:
        """兼容报告表中更显式的限幅字段名。"""

        return self.maximum_delay_change_limited

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "best_delay_score": self.best_validation_rmse_nm,
                "second_best_delay_score": (
                    self.second_best_validation_rmse_nm
                ),
                "delay_score_unit": self.validation_score_unit,
            }
        )
        return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _finite_non_negative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _time_column(dataframe: pd.DataFrame) -> str:
    # 因果窗口必须按到达/处理时间推进。真实长尾会让 sample/state 内容
    # 时间戳按到达顺序乱序，但这不是 freeze，也不应破坏在线窗口时间轴。
    for name in (
        "stream_timestamp_s",
        "wrench_arrival_timestamp_s",
        "arrival_timestamp_s",
        "time_s",
        "state_timestamp_s",
    ):
        if name in dataframe:
            return name
    raise ValueError(
        "windowed delay input needs stream_timestamp_s, "
        "state_timestamp_s, or time_s."
    )


def _stitch_local_time_if_needed(
    dataframe: pd.DataFrame,
    source_column: str,
) -> np.ndarray:
    """将多个按顺序存放、各自从零开始的历史 train 轨迹串接起来。"""

    raw = dataframe[source_column].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("tracker timestamps must be finite.")
    if len(raw) < 2:
        return raw.copy()
    differences = np.diff(raw)
    if np.all(differences > 0.0):
        return raw.copy()

    group_columns = [
        name
        for name in ("trajectory_id", "trajectory_family", "speed_profile")
        if name in dataframe
    ]
    if not group_columns:
        raise ValueError("tracker timestamps must be strictly increasing.")

    stitched = np.empty(len(dataframe), dtype=float)
    next_start = 0.0
    previous_key: tuple[str, ...] | None = None
    previous_end = 0.0
    previous_dt = 0.01
    for _, group in dataframe.groupby(group_columns, sort=False, dropna=False):
        indices = group.index.to_numpy(dtype=int)
        values = group[source_column].to_numpy(dtype=float)
        if len(values) > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError(
                "timestamps must increase within every trajectory stream."
            )
        key = tuple(str(group.iloc[0][name]) for name in group_columns)
        if key == previous_key:
            raise ValueError("a trajectory stream cannot appear in disjoint blocks.")
        local_dt = (
            float(np.median(np.diff(values)))
            if len(values) > 1
            else previous_dt
        )
        local_dt = local_dt if local_dt > 0.0 else previous_dt
        if previous_key is None:
            next_start = float(values[0])
        else:
            next_start = previous_end + max(previous_dt, local_dt)
        shifted = values - values[0] + next_start
        stitched[indices] = shifted
        previous_key = key
        previous_end = float(shifted[-1])
        previous_dt = local_dt
    if np.any(np.diff(stitched) <= 0.0):
        raise ValueError("stitched tracker timestamps must be strictly increasing.")
    return stitched


def sanitize_windowed_delay_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    """只保留观测白名单并拒绝 test/validation 数据。"""

    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("windowed delay input must be a non-empty DataFrame.")
    missing_motion = set(_REQUIRED_MOTION_COLUMNS).difference(dataframe.columns)
    if missing_motion:
        raise ValueError(
            f"windowed delay input is missing: {sorted(missing_motion)}"
        )
    if "dataset_split" in dataframe:
        observed = set(dataframe["dataset_split"].astype(str))
        disallowed = observed.difference(_ALLOWED_SPLITS)
        if disallowed:
            if "test" in disallowed:
                raise ValueError("test split is forbidden for online delay tracking.")
            raise ValueError(
                "windowed delay tracker accepts only train or online rows; "
                f"got {sorted(disallowed)}."
            )

    selected = [
        column for column in _TRACKER_INPUT_COLUMNS if column in dataframe
    ]
    sanitized = dataframe.loc[:, selected].copy(deep=True).reset_index(drop=True)
    if "dataset_split" not in sanitized:
        sanitized["dataset_split"] = "online"
    source_time = _time_column(sanitized)
    sanitized["tracker_time_s"] = _stitch_local_time_if_needed(
        sanitized,
        source_time,
    )
    sanitized.attrs.clear()
    return sanitized


class WindowedDelayTracker:
    """用历史滑动窗口跟踪缓慢变化的 wrench 相对延迟。

    ``scoring_callback(window, candidate_delays_ms)`` 应返回每个候选的验证
    RMSE。回调只会收到白名单化且不晚于当前更新时刻的有效历史样本。
    未提供回调时，默认使用明确的 sample/source 与 arrival 时间戳差；
    它绝不读取 ``true_delay`` 或 ``wrench_age``。
    """

    def __init__(
        self,
        *,
        window_duration_s: float = DEFAULT_WINDOW_DURATION_S,
        update_interval_s: float = DEFAULT_UPDATE_INTERVAL_S,
        minimum_delay_ms: float = DEFAULT_MINIMUM_DELAY_MS,
        maximum_delay_ms: float = DEFAULT_MAXIMUM_DELAY_MS,
        delay_step_ms: float = DEFAULT_DELAY_STEP_MS,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        alpha: float | None = None,
        maximum_delay_change_ms: float = (
            DEFAULT_MAXIMUM_DELAY_CHANGE_MS
        ),
        excitation_threshold: float = 0.5,
        dq_reference_rad_s: float = 0.10,
        ddq_reference_rad_s2: float = 0.50,
        minimum_effective_samples: int = 20,
        minimum_confidence: float = 0.50,
        initial_delay_ms: float = 0.0,
        scoring_callback: DelayScoringCallback | None = None,
        delay_scoring_callback: DelayScoringCallback | None = None,
    ) -> None:
        self.window_duration_s = _finite_positive(
            window_duration_s,
            "window_duration_s",
        )
        self.update_interval_s = _finite_positive(
            update_interval_s,
            "update_interval_s",
        )
        minimum = float(minimum_delay_ms)
        maximum = float(maximum_delay_ms)
        step = _finite_positive(delay_step_ms, "delay_step_ms")
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("delay grid limits must be finite.")
        if minimum >= maximum:
            raise ValueError("minimum_delay_ms must be below maximum_delay_ms.")
        count_float = (maximum - minimum) / step
        count = int(round(count_float))
        if not np.isclose(count_float, count, atol=1e-10, rtol=0.0):
            raise ValueError("delay grid range must be divisible by delay_step_ms.")
        self.candidate_delays_ms = np.linspace(
            minimum,
            maximum,
            count + 1,
        )
        selected_alpha = smoothing_alpha if alpha is None else alpha
        self.smoothing_alpha = float(selected_alpha)
        if (
            not np.isfinite(self.smoothing_alpha)
            or self.smoothing_alpha <= 0.0
            or self.smoothing_alpha > 1.0
        ):
            raise ValueError("smoothing alpha must lie in (0, 1].")
        self.maximum_delay_change_ms = _finite_positive(
            maximum_delay_change_ms,
            "maximum_delay_change_ms",
        )
        self.excitation_threshold = _finite_non_negative(
            excitation_threshold,
            "excitation_threshold",
        )
        self.dq_reference_rad_s = _finite_positive(
            dq_reference_rad_s,
            "dq_reference_rad_s",
        )
        self.ddq_reference_rad_s2 = _finite_positive(
            ddq_reference_rad_s2,
            "ddq_reference_rad_s2",
        )
        if int(minimum_effective_samples) != minimum_effective_samples:
            raise ValueError("minimum_effective_samples must be an integer.")
        self.minimum_effective_samples = int(minimum_effective_samples)
        if self.minimum_effective_samples <= 0:
            raise ValueError("minimum_effective_samples must be positive.")
        self.minimum_confidence = float(minimum_confidence)
        if (
            not np.isfinite(self.minimum_confidence)
            or self.minimum_confidence < 0.0
            or self.minimum_confidence > 1.0
        ):
            raise ValueError("minimum_confidence must lie in [0, 1].")
        self.initial_delay_ms = float(initial_delay_ms)
        if not np.isfinite(self.initial_delay_ms):
            raise ValueError("initial_delay_ms must be finite.")
        if (
            scoring_callback is not None
            and delay_scoring_callback is not None
            and scoring_callback is not delay_scoring_callback
        ):
            raise ValueError("provide only one delay scoring callback.")
        self.scoring_callback = (
            scoring_callback
            if scoring_callback is not None
            else delay_scoring_callback
        )
        self._current_delay_ms = self.initial_delay_ms
        self._last_update_time_s: float | None = None
        self._records: list[WindowedDelayEstimate] = []
        self._last_search_curve = pd.DataFrame()

    @property
    def current_delay_ms(self) -> float:
        return self._current_delay_ms

    @property
    def last_search_curve(self) -> pd.DataFrame:
        return self._last_search_curve.copy(deep=True)

    def reset(self, initial_delay_ms: float | None = None) -> None:
        if initial_delay_ms is not None:
            value = float(initial_delay_ms)
            if not np.isfinite(value):
                raise ValueError("initial_delay_ms must be finite.")
            self.initial_delay_ms = value
        self._current_delay_ms = self.initial_delay_ms
        self._last_update_time_s = None
        self._records = []
        self._last_search_curve = pd.DataFrame()

    def results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(record.as_dict() for record in self._records)

    @staticmethod
    def _effective_mask(window: pd.DataFrame) -> np.ndarray:
        valid = np.ones(len(window), dtype=bool)
        if "sample_valid" in window:
            valid &= window["sample_valid"].astype(bool).to_numpy()
        if "force_mapping_valid" in window:
            valid &= window["force_mapping_valid"].astype(bool).to_numpy()
        if "wrench_is_stale" in window:
            valid &= ~window["wrench_is_stale"].astype(bool).to_numpy()
        finite_columns = list(_REQUIRED_MOTION_COLUMNS)
        for name in (
            "q_hip_rad",
            "q_knee_rad",
            "fx_observed_n",
            "fz_observed_n",
        ):
            if name in window:
                finite_columns.append(name)
        valid &= np.isfinite(
            window[finite_columns].to_numpy(dtype=float)
        ).all(axis=1)
        return valid

    def _excitation(
        self,
        window: pd.DataFrame,
    ) -> tuple[float, float, float]:
        dq = window[
            ["dq_hip_rad_s", "dq_knee_rad_s"]
        ].to_numpy(dtype=float)
        ddq = window[
            ["ddq_hip_rad_s2", "ddq_knee_rad_s2"]
        ].to_numpy(dtype=float)
        dq_rms = float(np.sqrt(np.mean(np.sum(dq**2, axis=1))))
        ddq_rms = float(np.sqrt(np.mean(np.sum(ddq**2, axis=1))))
        score = float(
            np.hypot(
                dq_rms / self.dq_reference_rad_s,
                ddq_rms / self.ddq_reference_rad_s2,
            )
        )
        return score, dq_rms, ddq_rms

    def _timestamp_scores(
        self,
        window: pd.DataFrame,
    ) -> tuple[np.ndarray, str, str]:
        timestamp_pairs = (
            ("wrench_source_timestamp_s", "wrench_arrival_timestamp_s"),
            ("wrench_sample_timestamp_s", "wrench_arrival_timestamp_s"),
            ("sample_timestamp_s", "arrival_timestamp_s"),
            ("state_timestamp_s", "wrench_timestamp_s"),
        )
        for source_name, arrival_name in timestamp_pairs:
            if source_name not in window or arrival_name not in window:
                continue
            source = window[source_name].to_numpy(dtype=float)
            arrival = window[arrival_name].to_numpy(dtype=float)
            finite = np.isfinite(source) & np.isfinite(arrival)
            if not finite.any():
                continue
            observed_delay_ms = 1000.0 * (
                arrival[finite] - source[finite]
            )
            scores = np.sqrt(
                np.mean(
                    (
                        observed_delay_ms[:, np.newaxis]
                        - self.candidate_delays_ms[np.newaxis, :]
                    )
                    ** 2,
                    axis=0,
                )
            )
            return (
                scores,
                f"timestamp_difference:{arrival_name}-{source_name}",
                "ms_timestamp_residual",
            )
        raise ValueError(
            "default tracking needs reliable sample/source and arrival "
            "timestamps; otherwise provide scoring_callback."
        )

    def _callback_scores(
        self,
        window: pd.DataFrame,
    ) -> tuple[np.ndarray, str, str]:
        if self.scoring_callback is None:
            return self._timestamp_scores(window)
        result = self.scoring_callback(
            window.copy(deep=True),
            self.candidate_delays_ms.copy(),
        )
        if isinstance(result, pd.DataFrame):
            score_column = next(
                (
                    name
                    for name in (
                        "validation_rmse_nm",
                        "validation_torque_rmse_combined_nm",
                        "rmse_nm",
                        "score",
                    )
                    if name in result
                ),
                None,
            )
            if score_column is None:
                raise ValueError("scoring DataFrame has no RMSE/score column.")
            if "candidate_delay_ms" in result:
                lookup = result.set_index("candidate_delay_ms")[score_column]
                scores = lookup.reindex(self.candidate_delays_ms).to_numpy(
                    dtype=float
                )
            else:
                scores = result[score_column].to_numpy(dtype=float)
        elif isinstance(result, Mapping):
            key = next(
                (
                    name
                    for name in (
                        "validation_rmse_nm",
                        "rmse_nm",
                        "scores",
                        "score",
                    )
                    if name in result
                ),
                None,
            )
            if key is None:
                raise ValueError("scoring mapping has no RMSE/score values.")
            scores = np.asarray(result[key], dtype=float)
        elif isinstance(result, tuple):
            scores = np.asarray(result[0], dtype=float)
        else:
            scores = np.asarray(result, dtype=float)
        if scores.shape != self.candidate_delays_ms.shape:
            raise ValueError(
                "scoring_callback must return one value per delay candidate."
            )
        return scores, "injected_validation_rmse_callback", "N*m"

    def _confidence(
        self,
        *,
        best: float,
        second: float,
        curvature: float,
        effective_samples: int,
        excitation_score: float,
        boundary_hit: bool,
    ) -> float:
        scale = max(abs(best), abs(second), 1e-9)
        margin = max(second - best, 0.0)
        separation_score = float(
            np.clip(margin / (0.05 * scale + 1e-12), 0.0, 1.0)
        )
        curvature_scale = (
            curvature * float(self.candidate_delays_ms[1] - self.candidate_delays_ms[0]) ** 2
        )
        curvature_score = float(
            np.clip(curvature_scale / (0.10 * scale + 1e-12), 0.0, 1.0)
        )
        sample_score = float(
            np.clip(
                effective_samples / self.minimum_effective_samples,
                0.0,
                1.0,
            )
        )
        if self.excitation_threshold == 0.0:
            excitation_factor = 1.0
        else:
            excitation_factor = float(
                np.clip(
                    excitation_score / self.excitation_threshold,
                    0.0,
                    1.0,
                )
            )
        confidence = (
            0.40 * separation_score
            + 0.25 * curvature_score
            + 0.20 * sample_score
            + 0.15 * excitation_factor
        )
        if boundary_hit:
            confidence *= 0.35
        return float(np.clip(confidence, 0.0, 1.0))

    def _held_result(
        self,
        *,
        reason: str,
        window_start_s: float,
        window_end_s: float,
        effective_sample_count: int,
        excitation_score: float,
        dq_rms: float,
        ddq_rms: float,
        low_excitation: bool,
        confidence: float = 0.0,
        boundary_hit: bool = False,
        best: float = float("nan"),
        second: float = float("nan"),
        raw_delay_ms: float = float("nan"),
        curvature: float = float("nan"),
        selected_index: int = -1,
        score_unit: str = "N*m",
        score_source: str = "not_scored",
    ) -> WindowedDelayEstimate:
        previous = self._current_delay_ms
        result = WindowedDelayEstimate(
            estimated_delay_ms=previous,
            delay_confidence=confidence,
            delay_update_valid=False,
            delay_update_reason=reason,
            search_boundary_hit=boundary_hit,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            effective_sample_count=effective_sample_count,
            excitation_score=excitation_score,
            best_validation_rmse_nm=best,
            second_best_validation_rmse_nm=second,
            delay_value_held=True,
            maximum_delay_change_limited=False,
            raw_estimated_delay_ms=raw_delay_ms,
            smoothed_delay_proposal_ms=previous,
            previous_delay_ms=previous,
            applied_delay_change_ms=0.0,
            best_to_second_rmse_margin_nm=(
                second - best
                if np.isfinite(best) and np.isfinite(second)
                else float("nan")
            ),
            local_rmse_curvature=curvature,
            dq_rms_rad_s=dq_rms,
            ddq_rms_rad_s2=ddq_rms,
            low_excitation=low_excitation,
            low_confidence=reason in {
                "low_confidence",
                "search_boundary_low_confidence",
            },
            candidate_count=len(self.candidate_delays_ms),
            selected_candidate_index=selected_index,
            validation_score_unit=score_unit,
            score_source=score_source,
            causal_history_only=True,
            update_sequence=len(self._records),
        )
        self._last_update_time_s = window_end_s
        self._records.append(result)
        return result

    def _update_sanitized(
        self,
        history: pd.DataFrame,
        current_time_s: float,
    ) -> WindowedDelayEstimate:
        end = float(current_time_s)
        if not np.isfinite(end):
            raise ValueError("current_time_s must be finite.")
        if (
            self._last_update_time_s is not None
            and end <= self._last_update_time_s + _TIME_TOLERANCE_S
        ):
            raise ValueError("tracker update times must be strictly increasing.")
        start = end - self.window_duration_s
        in_window = (
            history["tracker_time_s"].to_numpy(dtype=float)
            >= start - _TIME_TOLERANCE_S
        ) & (
            history["tracker_time_s"].to_numpy(dtype=float)
            <= end + _TIME_TOLERANCE_S
        )
        window = history.loc[in_window].copy()
        effective_mask = self._effective_mask(window)
        effective = window.loc[effective_mask].copy()
        count = len(effective)
        if count:
            excitation, dq_rms, ddq_rms = self._excitation(effective)
        else:
            excitation = dq_rms = ddq_rms = 0.0
        low_excitation = excitation < self.excitation_threshold
        if count < self.minimum_effective_samples:
            return self._held_result(
                reason="insufficient_effective_samples",
                window_start_s=start,
                window_end_s=end,
                effective_sample_count=count,
                excitation_score=excitation,
                dq_rms=dq_rms,
                ddq_rms=ddq_rms,
                low_excitation=low_excitation,
            )
        if low_excitation:
            return self._held_result(
                reason="insufficient_excitation",
                window_start_s=start,
                window_end_s=end,
                effective_sample_count=count,
                excitation_score=excitation,
                dq_rms=dq_rms,
                ddq_rms=ddq_rms,
                low_excitation=True,
            )

        try:
            scores, score_source, score_unit = self._callback_scores(effective)
        except (KeyError, TypeError, ValueError) as exc:
            return self._held_result(
                reason=f"scoring_unavailable:{type(exc).__name__}",
                window_start_s=start,
                window_end_s=end,
                effective_sample_count=count,
                excitation_score=excitation,
                dq_rms=dq_rms,
                ddq_rms=ddq_rms,
                low_excitation=False,
            )
        finite = np.isfinite(scores)
        if finite.sum() < 2:
            return self._held_result(
                reason="insufficient_finite_delay_scores",
                window_start_s=start,
                window_end_s=end,
                effective_sample_count=count,
                excitation_score=excitation,
                dq_rms=dq_rms,
                ddq_rms=ddq_rms,
                low_excitation=False,
                score_source=score_source,
                score_unit=score_unit,
            )
        candidates = self.candidate_delays_ms
        curve = pd.DataFrame(
            {
                "candidate_delay_ms": candidates,
                "validation_rmse_nm": scores,
                "score_finite": finite,
            }
        )
        eligible = curve.loc[curve["score_finite"]].copy()
        eligible["absolute_candidate_delay_ms"] = eligible[
            "candidate_delay_ms"
        ].abs()
        eligible = eligible.sort_values(
            [
                "validation_rmse_nm",
                "absolute_candidate_delay_ms",
                "candidate_delay_ms",
            ],
            kind="mergesort",
        )
        best_row = eligible.iloc[0]
        second_row = eligible.iloc[1]
        best_index = int(best_row.name)
        best = float(best_row["validation_rmse_nm"])
        second = float(second_row["validation_rmse_nm"])
        raw_delay = float(best_row["candidate_delay_ms"])
        boundary = best_index in {0, len(candidates) - 1}
        step = float(candidates[1] - candidates[0])
        if 0 < best_index < len(candidates) - 1:
            curvature = float(
                max(
                    (
                        scores[best_index - 1]
                        - 2.0 * scores[best_index]
                        + scores[best_index + 1]
                    )
                    / step**2,
                    0.0,
                )
            )
        elif best_index == 0 and len(candidates) >= 3:
            curvature = float(
                max((scores[2] - 2.0 * scores[1] + scores[0]) / step**2, 0.0)
            )
        elif len(candidates) >= 3:
            curvature = float(
                max(
                    (
                        scores[-3] - 2.0 * scores[-2] + scores[-1]
                    )
                    / step**2,
                    0.0,
                )
            )
        else:
            curvature = 0.0
        confidence = self._confidence(
            best=best,
            second=second,
            curvature=curvature,
            effective_samples=count,
            excitation_score=excitation,
            boundary_hit=boundary,
        )
        curve["selected"] = False
        curve.loc[best_index, "selected"] = True
        self._last_search_curve = curve.drop(
            columns=["absolute_candidate_delay_ms"],
            errors="ignore",
        )
        if confidence < self.minimum_confidence:
            return self._held_result(
                reason=(
                    "search_boundary_low_confidence"
                    if boundary
                    else "low_confidence"
                ),
                window_start_s=start,
                window_end_s=end,
                effective_sample_count=count,
                excitation_score=excitation,
                dq_rms=dq_rms,
                ddq_rms=ddq_rms,
                low_excitation=False,
                confidence=confidence,
                boundary_hit=boundary,
                best=best,
                second=second,
                raw_delay_ms=raw_delay,
                curvature=curvature,
                selected_index=best_index,
                score_unit=score_unit,
                score_source=score_source,
            )

        previous = self._current_delay_ms
        smoothed = (
            self.smoothing_alpha * raw_delay
            + (1.0 - self.smoothing_alpha) * previous
        )
        proposed_change = smoothed - previous
        limited = abs(proposed_change) > self.maximum_delay_change_ms
        applied_change = float(
            np.clip(
                proposed_change,
                -self.maximum_delay_change_ms,
                self.maximum_delay_change_ms,
            )
        )
        estimated = previous + applied_change
        self._current_delay_ms = estimated
        result = WindowedDelayEstimate(
            estimated_delay_ms=estimated,
            delay_confidence=confidence,
            delay_update_valid=True,
            delay_update_reason=(
                "updated_with_maximum_delay_change_limit"
                if limited
                else "updated"
            ),
            search_boundary_hit=boundary,
            window_start_s=start,
            window_end_s=end,
            effective_sample_count=count,
            excitation_score=excitation,
            best_validation_rmse_nm=best,
            second_best_validation_rmse_nm=second,
            delay_value_held=False,
            maximum_delay_change_limited=limited,
            raw_estimated_delay_ms=raw_delay,
            smoothed_delay_proposal_ms=smoothed,
            previous_delay_ms=previous,
            applied_delay_change_ms=applied_change,
            best_to_second_rmse_margin_nm=second - best,
            local_rmse_curvature=curvature,
            dq_rms_rad_s=dq_rms,
            ddq_rms_rad_s2=ddq_rms,
            low_excitation=False,
            low_confidence=False,
            candidate_count=len(candidates),
            selected_candidate_index=best_index,
            validation_score_unit=score_unit,
            score_source=score_source,
            causal_history_only=True,
            update_sequence=len(self._records),
        )
        self._last_update_time_s = end
        self._records.append(result)
        return result

    def update(
        self,
        history_dataframe: pd.DataFrame,
        current_time_s: float | None = None,
    ) -> WindowedDelayEstimate:
        """从不晚于 ``current_time_s`` 的历史窗口执行一次更新。"""

        history = sanitize_windowed_delay_input(history_dataframe)
        end = (
            float(history["tracker_time_s"].max())
            if current_time_s is None
            else float(current_time_s)
        )
        return self._update_sanitized(history, end)

    def track(
        self,
        historical_dataframe: pd.DataFrame,
        *,
        reset: bool = True,
    ) -> pd.DataFrame:
        """按 0.5 s（默认）更新周期回放历史 train/online 观测。"""

        history = sanitize_windowed_delay_input(historical_dataframe)
        if reset:
            self.reset()
        minimum_time = float(history["tracker_time_s"].min())
        maximum_time = float(history["tracker_time_s"].max())
        first_update = minimum_time + self.window_duration_s
        if first_update > maximum_time + _TIME_TOLERANCE_S:
            return self.results_dataframe()
        count = int(
            np.floor(
                (maximum_time - first_update + _TIME_TOLERANCE_S)
                / self.update_interval_s
            )
        )
        update_times = first_update + np.arange(count + 1) * self.update_interval_s
        for update_time in update_times:
            self._update_sanitized(history, float(update_time))
        return self.results_dataframe()

    def process(
        self,
        historical_dataframe: pd.DataFrame,
        *,
        reset: bool = True,
    ) -> pd.DataFrame:
        """``track`` 的语义化别名。"""

        return self.track(historical_dataframe, reset=reset)

    def process_stream(
        self,
        historical_dataframe: pd.DataFrame,
        *,
        reset: bool = True,
    ) -> pd.DataFrame:
        """``track`` 的在线回放别名；每个窗口仍严格只见历史。"""

        return self.track(historical_dataframe, reset=reset)
