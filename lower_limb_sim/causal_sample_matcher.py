"""在线到达 wrench 与历史人体状态之间的严格因果匹配。

本模块只处理软件时间轴和二维观测，不读取机器人、硬件或真实参数。可靠的
wrench 样本时间戳优先；缺失时才使用 ``arrival - estimated_delay``。
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import asdict, dataclass
import threading
from typing import Any, Protocol

import numpy as np


TIME_TOLERANCE_S = 1e-12
DEFAULT_MAX_WRENCH_AGE_S = 0.100
DEFAULT_MAX_MATCH_ERROR_S = 0.005
DEFAULT_MAX_STATE_INTERVAL_S = 0.020


class StateHistoryProtocol(Protocol):
    """``StateHistoryBuffer`` 被匹配器使用的最小公开协议。"""

    @property
    def oldest_timestamp_s(self) -> float | None: ...

    @property
    def newest_timestamp_s(self) -> float | None: ...

    def query(
        self,
        target_timestamp_s: float,
        method: str = "linear_interpolation",
        max_state_interval_s: float | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class CausalSampleMatch:
    """一次 wrench 到达事件的完整匹配结果。"""

    valid: bool
    invalid_reason: str
    arrival_timestamp_s: float
    sample_timestamp_s: float
    target_state_timestamp_s: float
    matched_state_timestamp_s: float
    current_timestamp_s: float
    wrench_age_s: float
    processing_age_s: float
    state_wrench_skew_s: float
    state_match_error_s: float
    q_hip_rad: float
    q_knee_rad: float
    dq_hip_rad_s: float
    dq_knee_rad_s: float
    ddq_hip_rad_s2: float
    ddq_knee_rad_s2: float
    fx_observed_n: float
    fz_observed_n: float
    used_true_sample_timestamp: bool
    used_interpolation: bool
    confidence: float
    estimated_delay_s: float = np.nan
    causal_state_only: bool = True

    def as_dict(self) -> dict[str, bool | float | str]:
        """返回可直接用于 DataFrame/JSON 的标量字典。"""

        result = asdict(self)
        result.update(
            {
                "alignment_valid": self.valid,
                "state_timestamp_s": self.matched_state_timestamp_s,
                "wrench_arrival_timestamp_s": self.arrival_timestamp_s,
                "wrench_sample_timestamp_s": self.sample_timestamp_s,
                "delay_confidence": self.confidence,
            }
        )
        return result

    @property
    def alignment_valid(self) -> bool:
        return self.valid

    @property
    def state_timestamp_s(self) -> float:
        return self.matched_state_timestamp_s

    @property
    def wrench_arrival_timestamp_s(self) -> float:
        return self.arrival_timestamp_s

    @property
    def wrench_sample_timestamp_s(self) -> float:
        return self.sample_timestamp_s

    @property
    def delay_confidence(self) -> float:
        return self.confidence


def _finite_scalar(value: float | int | np.number) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _timestamp_or_nan(value: float | None) -> float:
    return float(value) if value is not None and _finite_scalar(value) else np.nan


def _invalid_match(
    reason: str,
    *,
    arrival_timestamp_s: float,
    sample_timestamp_s: float,
    target_state_timestamp_s: float,
    current_timestamp_s: float,
    fx_observed_n: float,
    fz_observed_n: float,
    used_true_sample_timestamp: bool,
    matched_state_timestamp_s: float = np.nan,
    state_match_error_s: float = np.nan,
    used_interpolation: bool = False,
) -> CausalSampleMatch:
    wrench_age = arrival_timestamp_s - target_state_timestamp_s
    processing_age = current_timestamp_s - target_state_timestamp_s
    skew = matched_state_timestamp_s - target_state_timestamp_s
    return CausalSampleMatch(
        valid=False,
        invalid_reason=reason,
        arrival_timestamp_s=arrival_timestamp_s,
        sample_timestamp_s=sample_timestamp_s,
        target_state_timestamp_s=target_state_timestamp_s,
        matched_state_timestamp_s=matched_state_timestamp_s,
        current_timestamp_s=current_timestamp_s,
        wrench_age_s=wrench_age,
        processing_age_s=processing_age,
        state_wrench_skew_s=skew,
        state_match_error_s=state_match_error_s,
        q_hip_rad=np.nan,
        q_knee_rad=np.nan,
        dq_hip_rad_s=np.nan,
        dq_knee_rad_s=np.nan,
        ddq_hip_rad_s2=np.nan,
        ddq_knee_rad_s2=np.nan,
        fx_observed_n=fx_observed_n,
        fz_observed_n=fz_observed_n,
        used_true_sample_timestamp=used_true_sample_timestamp,
        used_interpolation=used_interpolation,
        confidence=0.0,
    )


class CausalSampleMatcher:
    """保存最近可靠样本时间戳，并执行 fail-closed 因果匹配。"""

    def __init__(
        self,
        *,
        max_wrench_age_s: float = DEFAULT_MAX_WRENCH_AGE_S,
        max_match_error_s: float = DEFAULT_MAX_MATCH_ERROR_S,
        max_state_interval_s: float = DEFAULT_MAX_STATE_INTERVAL_S,
    ) -> None:
        for name, value in {
            "max_wrench_age_s": max_wrench_age_s,
            "max_match_error_s": max_match_error_s,
            "max_state_interval_s": max_state_interval_s,
        }.items():
            if not _finite_scalar(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.max_wrench_age_s = float(max_wrench_age_s)
        self.max_match_error_s = float(max_match_error_s)
        self.max_state_interval_s = float(max_state_interval_s)
        self._last_reliable_sample_timestamp_s: float | None = None
        # 按 sample/content timestamp 排序保存近期已成功匹配的唯一样本。
        # 到达顺序可以因长尾和抖动而不同于 sample timestamp 顺序，因此
        # 不能只和“上一条”比较，也不能把合法迟到样本当作冻结。
        self._seen_reliable_sample_timestamps_s: list[float] = []
        self._lock = threading.Lock()

    @property
    def last_reliable_sample_timestamp_s(self) -> float | None:
        """已成功匹配样本中最新（数值最大）的可靠时间戳。"""

        with self._lock:
            return self._last_reliable_sample_timestamp_s

    @property
    def seen_reliable_sample_count(self) -> int:
        """当前防重复时间窗内已成功匹配的唯一 sample timestamp 数量。"""

        with self._lock:
            return len(self._seen_reliable_sample_timestamps_s)

    def reset(self) -> None:
        """清除重复/冻结检测状态。"""

        with self._lock:
            self._last_reliable_sample_timestamp_s = None
            self._seen_reliable_sample_timestamps_s.clear()

    def _prune_seen_sample_timestamps(self, arrival_timestamp_s: float) -> None:
        """删除已不可能通过 wrench age 门限的旧防重复记录。"""

        cutoff = (
            arrival_timestamp_s
            - self.max_wrench_age_s
            - TIME_TOLERANCE_S
        )
        first_retained = bisect_left(
            self._seen_reliable_sample_timestamps_s,
            cutoff,
        )
        if first_retained:
            del self._seen_reliable_sample_timestamps_s[:first_retained]

    def _sample_timestamp_was_seen(self, sample_timestamp_s: float) -> bool:
        timestamps = self._seen_reliable_sample_timestamps_s
        insertion = bisect_left(timestamps, sample_timestamp_s)
        neighbors = []
        if insertion < len(timestamps):
            neighbors.append(timestamps[insertion])
        if insertion > 0:
            neighbors.append(timestamps[insertion - 1])
        return any(
            abs(timestamp - sample_timestamp_s) <= TIME_TOLERANCE_S
            for timestamp in neighbors
        )

    def match(
        self,
        state_history_buffer: StateHistoryProtocol,
        *,
        arrival_timestamp_s: float,
        current_timestamp_s: float,
        fx_observed_n: float,
        fz_observed_n: float,
        estimated_delay_s: float,
        sample_timestamp_s: float | None = None,
        sample_timestamp_reliable: bool = False,
        wrench_valid: bool = True,
        wrench_is_stale: bool = False,
    ) -> CausalSampleMatch:
        """匹配一个刚到达的 wrench；只查询到达时已经缓存的状态。"""

        with self._lock:
            return self._match_locked(
                state_history_buffer,
                arrival_timestamp_s=arrival_timestamp_s,
                current_timestamp_s=current_timestamp_s,
                fx_observed_n=fx_observed_n,
                fz_observed_n=fz_observed_n,
                estimated_delay_s=estimated_delay_s,
                sample_timestamp_s=sample_timestamp_s,
                sample_timestamp_reliable=sample_timestamp_reliable,
                wrench_valid=wrench_valid,
                wrench_is_stale=wrench_is_stale,
            )

    def _match_locked(
        self,
        state_history_buffer: StateHistoryProtocol,
        *,
        arrival_timestamp_s: float,
        current_timestamp_s: float,
        fx_observed_n: float,
        fz_observed_n: float,
        estimated_delay_s: float,
        sample_timestamp_s: float | None,
        sample_timestamp_reliable: bool,
        wrench_valid: bool,
        wrench_is_stale: bool,
    ) -> CausalSampleMatch:
        arrival = _timestamp_or_nan(arrival_timestamp_s)
        current = _timestamp_or_nan(current_timestamp_s)
        sample = _timestamp_or_nan(sample_timestamp_s)
        force_x = _timestamp_or_nan(fx_observed_n)
        force_z = _timestamp_or_nan(fz_observed_n)
        use_sample_time = bool(sample_timestamp_reliable)

        if use_sample_time:
            target = sample
        elif _finite_scalar(arrival) and _finite_scalar(estimated_delay_s):
            target = arrival - float(estimated_delay_s)
        else:
            target = np.nan

        invalid_arguments = (
            not _finite_scalar(arrival)
            or not _finite_scalar(current)
            or not _finite_scalar(estimated_delay_s)
            or (use_sample_time and not _finite_scalar(sample))
        )
        if invalid_arguments:
            return _invalid_match(
                "non_finite_timestamp_or_delay",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )
        if not wrench_valid or not _finite_scalar(force_x) or not _finite_scalar(
            force_z
        ):
            return _invalid_match(
                "wrench_dropout_or_non_finite_force",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )
        if wrench_is_stale:
            return _invalid_match(
                "stale_or_frozen_wrench",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )
        if arrival > current + TIME_TOLERANCE_S:
            return _invalid_match(
                "arrival_timestamp_in_future",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )
        if target > arrival + TIME_TOLERANCE_S:
            return _invalid_match(
                "wrench_target_timestamp_after_arrival",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )
        if use_sample_time:
            self._prune_seen_sample_timestamps(arrival)
            if self._sample_timestamp_was_seen(sample):
                return _invalid_match(
                    "duplicate_wrench_sample_timestamp",
                    arrival_timestamp_s=arrival,
                    sample_timestamp_s=sample,
                    target_state_timestamp_s=target,
                    current_timestamp_s=current,
                    fx_observed_n=force_x,
                    fz_observed_n=force_z,
                    used_true_sample_timestamp=True,
                )

        wrench_age = arrival - target
        if wrench_age < -TIME_TOLERANCE_S:
            return _invalid_match(
                "negative_wrench_age",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )
        if wrench_age > self.max_wrench_age_s + TIME_TOLERANCE_S:
            return _invalid_match(
                "wrench_age_limit_exceeded",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
            )

        oldest = getattr(state_history_buffer, "oldest_timestamp_s", None)
        newest = getattr(state_history_buffer, "newest_timestamp_s", None)
        if oldest is not None and _finite_scalar(oldest):
            if target < float(oldest) - TIME_TOLERANCE_S:
                return _invalid_match(
                    "state_history_expired",
                    arrival_timestamp_s=arrival,
                    sample_timestamp_s=sample,
                    target_state_timestamp_s=target,
                    current_timestamp_s=current,
                    fx_observed_n=force_x,
                    fz_observed_n=force_z,
                    used_true_sample_timestamp=use_sample_time,
                )
        # A future state in the caller-provided cache would let interpolation
        # use information unavailable at wrench arrival. Reject the event.
        if newest is not None and _finite_scalar(newest):
            if float(newest) > arrival + TIME_TOLERANCE_S:
                return _invalid_match(
                    "state_history_contains_future_state",
                    arrival_timestamp_s=arrival,
                    sample_timestamp_s=sample,
                    target_state_timestamp_s=target,
                    current_timestamp_s=current,
                    fx_observed_n=force_x,
                    fz_observed_n=force_z,
                    used_true_sample_timestamp=use_sample_time,
                )

        state_match = state_history_buffer.query(
            target,
            method="linear_interpolation",
            max_state_interval_s=self.max_state_interval_s,
        )
        matched_timestamp = _timestamp_or_nan(
            getattr(state_match, "matched_timestamp_s", None)
        )
        match_error = _timestamp_or_nan(
            getattr(state_match, "time_error_s", None)
        )
        used_interpolation = bool(
            getattr(state_match, "used_interpolation", False)
        )
        if not bool(getattr(state_match, "valid", False)):
            reason = str(getattr(state_match, "reason", "state_history_invalid"))
            return _invalid_match(
                reason or "state_history_invalid",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
                matched_state_timestamp_s=matched_timestamp,
                state_match_error_s=match_error,
                used_interpolation=used_interpolation,
            )
        if (
            not _finite_scalar(matched_timestamp)
            or not _finite_scalar(match_error)
        ):
            return _invalid_match(
                "non_finite_state_match",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
                matched_state_timestamp_s=matched_timestamp,
                state_match_error_s=match_error,
                used_interpolation=used_interpolation,
            )
        if matched_timestamp > arrival + TIME_TOLERANCE_S:
            return _invalid_match(
                "matched_state_timestamp_after_arrival",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
                matched_state_timestamp_s=matched_timestamp,
                state_match_error_s=match_error,
                used_interpolation=used_interpolation,
            )
        if match_error > self.max_match_error_s + TIME_TOLERANCE_S:
            return _invalid_match(
                "state_match_error_limit_exceeded",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
                matched_state_timestamp_s=matched_timestamp,
                state_match_error_s=match_error,
                used_interpolation=used_interpolation,
            )

        state_values = {
            name: _timestamp_or_nan(getattr(state_match, name, None))
            for name in (
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            )
        }
        if not all(_finite_scalar(value) for value in state_values.values()):
            return _invalid_match(
                "non_finite_matched_state",
                arrival_timestamp_s=arrival,
                sample_timestamp_s=sample,
                target_state_timestamp_s=target,
                current_timestamp_s=current,
                fx_observed_n=force_x,
                fz_observed_n=force_z,
                used_true_sample_timestamp=use_sample_time,
                matched_state_timestamp_s=matched_timestamp,
                state_match_error_s=match_error,
                used_interpolation=used_interpolation,
            )

        confidence = 1.0 if use_sample_time else 0.75
        if used_interpolation:
            confidence *= 0.90
        confidence *= max(
            0.0,
            1.0 - match_error / self.max_match_error_s,
        )
        if use_sample_time:
            insort(self._seen_reliable_sample_timestamps_s, sample)
            self._last_reliable_sample_timestamp_s = (
                sample
                if self._last_reliable_sample_timestamp_s is None
                else max(self._last_reliable_sample_timestamp_s, sample)
            )
        return CausalSampleMatch(
            valid=True,
            invalid_reason="",
            arrival_timestamp_s=arrival,
            sample_timestamp_s=sample,
            target_state_timestamp_s=target,
            matched_state_timestamp_s=matched_timestamp,
            current_timestamp_s=current,
            wrench_age_s=wrench_age,
            processing_age_s=current - target,
            state_wrench_skew_s=matched_timestamp - target,
            state_match_error_s=match_error,
            q_hip_rad=state_values["q_hip_rad"],
            q_knee_rad=state_values["q_knee_rad"],
            dq_hip_rad_s=state_values["dq_hip_rad_s"],
            dq_knee_rad_s=state_values["dq_knee_rad_s"],
            ddq_hip_rad_s2=state_values["ddq_hip_rad_s2"],
            ddq_knee_rad_s2=state_values["ddq_knee_rad_s2"],
            fx_observed_n=force_x,
            fz_observed_n=force_z,
            used_true_sample_timestamp=use_sample_time,
            used_interpolation=used_interpolation,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            estimated_delay_s=float(estimated_delay_s),
        )


def match_arriving_wrench_to_state(
    wrench_arrival_timestamp_s: float | StateHistoryProtocol,
    wrench_sample_timestamp_s: float | None = None,
    fx: float | None = None,
    fz: float | None = None,
    estimated_delay_s: float | None = None,
    state_history_buffer: StateHistoryProtocol | None = None,
    *,
    arrival_timestamp_s: float | None = None,
    current_timestamp_s: float | None = None,
    fx_observed_n: float | None = None,
    fz_observed_n: float | None = None,
    sample_timestamp_s: float | None = None,
    sample_timestamp_reliable: bool | None = None,
    wrench_valid: bool = True,
    wrench_is_stale: bool = False,
    matcher: CausalSampleMatcher | None = None,
) -> CausalSampleMatch:
    """函数式入口，兼容题设的六个位置参数和显式审计关键字。

    题设位置参数顺序为 ``arrival, sample, Fx, Fz, estimated_delay,
    state_history_buffer``。既有调用也可继续把 buffer 作为第一个位置参数，
    再使用 ``arrival_timestamp_s`` 等关键字。跨样本冻结检测时应复用
    ``matcher`` 实例。
    """

    if hasattr(wrench_arrival_timestamp_s, "query"):
        if state_history_buffer is not None:
            raise TypeError("state_history_buffer was supplied twice.")
        state_history_buffer = wrench_arrival_timestamp_s
        resolved_arrival = arrival_timestamp_s
        resolved_sample = sample_timestamp_s
        resolved_fx = fx_observed_n
        resolved_fz = fz_observed_n
        reliable = bool(sample_timestamp_reliable)
    else:
        resolved_arrival = (
            float(wrench_arrival_timestamp_s)
            if arrival_timestamp_s is None
            else arrival_timestamp_s
        )
        resolved_sample = (
            wrench_sample_timestamp_s
            if sample_timestamp_s is None
            else sample_timestamp_s
        )
        resolved_fx = fx if fx_observed_n is None else fx_observed_n
        resolved_fz = fz if fz_observed_n is None else fz_observed_n
        reliable = (
            resolved_sample is not None
            if sample_timestamp_reliable is None
            else bool(sample_timestamp_reliable)
        )
    if state_history_buffer is None:
        raise TypeError("state_history_buffer is required.")
    if resolved_arrival is None:
        raise TypeError("wrench arrival timestamp is required.")
    if resolved_fx is None or resolved_fz is None:
        raise TypeError("Fx and Fz are required.")
    if estimated_delay_s is None:
        raise TypeError("estimated_delay_s is required.")
    resolved_current = (
        float(resolved_arrival)
        if current_timestamp_s is None
        else current_timestamp_s
    )

    active_matcher = matcher if matcher is not None else CausalSampleMatcher()
    return active_matcher.match(
        state_history_buffer,
        arrival_timestamp_s=float(resolved_arrival),
        current_timestamp_s=float(resolved_current),
        fx_observed_n=float(resolved_fx),
        fz_observed_n=float(resolved_fz),
        estimated_delay_s=estimated_delay_s,
        sample_timestamp_s=resolved_sample,
        sample_timestamp_reliable=reliable,
        wrench_valid=wrench_valid,
        wrench_is_stale=wrench_is_stale,
    )
