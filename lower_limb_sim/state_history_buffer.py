"""有限长度、纯软件的下肢运动状态历史缓冲区。

缓冲区原样保存人体广义坐标 ``q_hip`` 和 ``q_knee``。任何下游几何计算
都必须继续使用 ``theta_shank = q_hip - q_knee``；本模块不把膝角转换成
普通机器人常见的相加角度，也不依赖机器人硬件或采集代码。
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .config import max_alignment_interpolation_gap_s

DEFAULT_HISTORY_DURATION_S = 2.0
MATCH_METHODS = ("nearest", "linear_interpolation")
TIME_TOLERANCE_S = 1e-12


@dataclass(frozen=True)
class _StateSample:
    timestamp_s: float
    q_hip_rad: float
    q_knee_rad: float
    dq_hip_rad_s: float
    dq_knee_rad_s: float
    ddq_hip_rad_s2: float
    ddq_knee_rad_s2: float


@dataclass(frozen=True)
class StateMatchResult:
    """一次历史状态查询的数值、时间误差和有效性。

    ``time_error_s`` 是匹配时刻与目标时刻之间的绝对差。线性插值成功时
    匹配时刻就是目标时刻，因此误差为零。无有效匹配时，时间与状态数值
    均为 NaN，并由 ``reason`` 给出稳定的机器可读原因。
    """

    target_timestamp_s: float
    matched_timestamp_s: float
    time_error_s: float
    valid: bool
    reason: str
    used_interpolation: bool
    q_hip_rad: float
    q_knee_rad: float
    dq_hip_rad_s: float
    dq_knee_rad_s: float
    ddq_hip_rad_s2: float
    ddq_knee_rad_s2: float

    @property
    def matched_state_timestamp_s(self) -> float:
        return self.matched_timestamp_s

    @property
    def state_time_error_s(self) -> float:
        return self.time_error_s

    @property
    def state_match_valid(self) -> bool:
        return self.valid

    @property
    def state_match_reason(self) -> str:
        return self.reason

    def as_dict(self) -> dict[str, bool | float | str]:
        result = asdict(self)
        result.update(
            {
                "matched_state_timestamp_s": self.matched_timestamp_s,
                "state_time_error_s": self.time_error_s,
                "state_match_valid": self.valid,
                "state_match_reason": self.reason,
            }
        )
        return result


class StateHistoryBuffer:
    """保存最近一段严格递增的髋膝运动状态。

    查询只会访问已经通过 :meth:`append` 放入缓冲区的样本。调用方若按
    数据到达顺序 append，再在当前 arrival timestamp 查询，nearest 和
    linear_interpolation 都不会读取尚未到达的状态。目标晚于最新缓存
    状态时不会外推，而是返回 ``state_history_future_query``。
    """

    def __init__(
        self,
        history_duration_s: float = DEFAULT_HISTORY_DURATION_S,
        max_state_interval_s: float = max_alignment_interpolation_gap_s,
    ) -> None:
        duration = float(history_duration_s)
        maximum_interval = float(max_state_interval_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("history_duration_s must be finite and positive.")
        if not np.isfinite(maximum_interval) or maximum_interval <= 0.0:
            raise ValueError(
                "max_state_interval_s must be finite and positive."
            )
        self._history_duration_s = duration
        self._max_state_interval_s = maximum_interval
        self._samples: deque[_StateSample] = deque()

    @property
    def history_duration_s(self) -> float:
        return self._history_duration_s

    @property
    def max_state_interval_s(self) -> float:
        return self._max_state_interval_s

    @property
    def oldest_timestamp_s(self) -> float | None:
        return self._samples[0].timestamp_s if self._samples else None

    @property
    def newest_timestamp_s(self) -> float | None:
        return self._samples[-1].timestamp_s if self._samples else None

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def append(
        self,
        timestamp_s: float,
        q_hip_rad: float,
        q_knee_rad: float,
        dq_hip_rad_s: float,
        dq_knee_rad_s: float,
        ddq_hip_rad_s2: float,
        ddq_knee_rad_s2: float,
    ) -> None:
        """追加一个有限状态；时间戳必须相对最新样本严格递增。"""

        names = (
            "timestamp_s",
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        )
        values = np.asarray(
            (
                timestamp_s,
                q_hip_rad,
                q_knee_rad,
                dq_hip_rad_s,
                dq_knee_rad_s,
                ddq_hip_rad_s2,
                ddq_knee_rad_s2,
            ),
            dtype=float,
        )
        if not np.isfinite(values).all():
            invalid = [
                name
                for name, value in zip(names, values)
                if not np.isfinite(value)
            ]
            raise ValueError(
                f"state history values must be finite: {invalid}"
            )
        timestamp = float(values[0])
        if self._samples and timestamp <= self._samples[-1].timestamp_s:
            raise ValueError(
                "state timestamps must be strictly increasing."
            )
        sample = _StateSample(
            timestamp_s=timestamp,
            q_hip_rad=float(values[1]),
            q_knee_rad=float(values[2]),
            dq_hip_rad_s=float(values[3]),
            dq_knee_rad_s=float(values[4]),
            ddq_hip_rad_s2=float(values[5]),
            ddq_knee_rad_s2=float(values[6]),
        )
        self._samples.append(sample)
        self._prune_expired(timestamp)

    def append_state(
        self,
        timestamp_s: float,
        q_hip_rad: float,
        q_knee_rad: float,
        dq_hip_rad_s: float,
        dq_knee_rad_s: float,
        ddq_hip_rad_s2: float,
        ddq_knee_rad_s2: float,
    ) -> None:
        """``append`` 的显式同义入口，便于调用端表达状态语义。"""

        self.append(
            timestamp_s,
            q_hip_rad,
            q_knee_rad,
            dq_hip_rad_s,
            dq_knee_rad_s,
            ddq_hip_rad_s2,
            ddq_knee_rad_s2,
        )

    def _prune_expired(self, newest_timestamp_s: float) -> None:
        cutoff = newest_timestamp_s - self._history_duration_s
        while (
            self._samples
            and self._samples[0].timestamp_s < cutoff - TIME_TOLERANCE_S
        ):
            self._samples.popleft()

    @staticmethod
    def _invalid_result(
        target_timestamp_s: float,
        reason: str,
        *,
        matched_timestamp_s: float = np.nan,
        time_error_s: float = np.nan,
    ) -> StateMatchResult:
        return StateMatchResult(
            target_timestamp_s=target_timestamp_s,
            matched_timestamp_s=matched_timestamp_s,
            time_error_s=time_error_s,
            valid=False,
            reason=reason,
            used_interpolation=False,
            q_hip_rad=np.nan,
            q_knee_rad=np.nan,
            dq_hip_rad_s=np.nan,
            dq_knee_rad_s=np.nan,
            ddq_hip_rad_s2=np.nan,
            ddq_knee_rad_s2=np.nan,
        )

    @staticmethod
    def _sample_result(
        target_timestamp_s: float,
        sample: _StateSample,
    ) -> StateMatchResult:
        return StateMatchResult(
            target_timestamp_s=target_timestamp_s,
            matched_timestamp_s=sample.timestamp_s,
            time_error_s=abs(sample.timestamp_s - target_timestamp_s),
            valid=True,
            reason="",
            used_interpolation=False,
            q_hip_rad=sample.q_hip_rad,
            q_knee_rad=sample.q_knee_rad,
            dq_hip_rad_s=sample.dq_hip_rad_s,
            dq_knee_rad_s=sample.dq_knee_rad_s,
            ddq_hip_rad_s2=sample.ddq_hip_rad_s2,
            ddq_knee_rad_s2=sample.ddq_knee_rad_s2,
        )

    def query(
        self,
        target_timestamp_s: float,
        method: str = "linear_interpolation",
        max_state_interval_s: float | None = None,
    ) -> StateMatchResult:
        """查询已缓存历史中的最近样本或线性插值状态。

        ``nearest`` 的门限作用于匹配样本的绝对时间误差；
        ``linear_interpolation`` 的门限作用于左右状态样本的时间跨度。
        精确命中缓存样本时不需要插值，也不受相邻样本跨度影响。
        """

        target = float(target_timestamp_s)
        if not np.isfinite(target):
            raise ValueError("target_timestamp_s must be finite.")
        if method not in MATCH_METHODS:
            raise ValueError(f"method must be one of {MATCH_METHODS}.")
        maximum_interval = (
            self._max_state_interval_s
            if max_state_interval_s is None
            else float(max_state_interval_s)
        )
        if not np.isfinite(maximum_interval) or maximum_interval <= 0.0:
            raise ValueError(
                "max_state_interval_s must be finite and positive."
            )
        if not self._samples:
            return self._invalid_result(
                target,
                "state_history_no_bracket",
            )

        oldest = self._samples[0].timestamp_s
        newest = self._samples[-1].timestamp_s
        if target < oldest - TIME_TOLERANCE_S:
            return self._invalid_result(
                target,
                "state_history_expired",
            )
        if target > newest + TIME_TOLERANCE_S:
            return self._invalid_result(
                target,
                "state_history_future_query",
            )

        samples = tuple(self._samples)
        timestamps = np.fromiter(
            (sample.timestamp_s for sample in samples),
            dtype=float,
            count=len(samples),
        )
        insertion = int(np.searchsorted(timestamps, target, side="left"))
        if (
            insertion < len(samples)
            and abs(timestamps[insertion] - target) <= TIME_TOLERANCE_S
        ):
            return self._sample_result(target, samples[insertion])

        if method == "nearest":
            candidates: list[_StateSample] = []
            if insertion > 0:
                candidates.append(samples[insertion - 1])
            if insertion < len(samples):
                candidates.append(samples[insertion])
            if not candidates:
                return self._invalid_result(
                    target,
                    "state_history_no_bracket",
                )
            # 等距时选更早样本，结果稳定且更保守。
            matched = min(
                candidates,
                key=lambda sample: (
                    abs(sample.timestamp_s - target),
                    sample.timestamp_s,
                ),
            )
            time_error = abs(matched.timestamp_s - target)
            if time_error > maximum_interval + TIME_TOLERANCE_S:
                return self._invalid_result(
                    target,
                    "state_history_gap_exceeded",
                    matched_timestamp_s=matched.timestamp_s,
                    time_error_s=time_error,
                )
            return self._sample_result(target, matched)

        if insertion <= 0 or insertion >= len(samples):
            return self._invalid_result(
                target,
                "state_history_no_bracket",
            )
        left = samples[insertion - 1]
        right = samples[insertion]
        bracket_interval = right.timestamp_s - left.timestamp_s
        if bracket_interval > maximum_interval + TIME_TOLERANCE_S:
            return self._invalid_result(
                target,
                "state_history_gap_exceeded",
            )
        alpha = (target - left.timestamp_s) / bracket_interval

        def interpolate(attribute: str) -> float:
            left_value = float(getattr(left, attribute))
            right_value = float(getattr(right, attribute))
            return left_value + alpha * (right_value - left_value)

        return StateMatchResult(
            target_timestamp_s=target,
            matched_timestamp_s=target,
            time_error_s=0.0,
            valid=True,
            reason="",
            used_interpolation=True,
            q_hip_rad=interpolate("q_hip_rad"),
            q_knee_rad=interpolate("q_knee_rad"),
            dq_hip_rad_s=interpolate("dq_hip_rad_s"),
            dq_knee_rad_s=interpolate("dq_knee_rad_s"),
            ddq_hip_rad_s2=interpolate("ddq_hip_rad_s2"),
            ddq_knee_rad_s2=interpolate("ddq_knee_rad_s2"),
        )
