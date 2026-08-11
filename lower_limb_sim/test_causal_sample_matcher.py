"""在线 wrench 到达事件与历史状态因果匹配的离线单元测试。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from lower_limb_sim.causal_sample_matcher import (
    CausalSampleMatcher,
    match_arriving_wrench_to_state,
)
from lower_limb_sim.state_history_buffer import StateHistoryBuffer


def _history(
    start_s: float = 0.80,
    stop_s: float = 1.00,
    step_s: float = 0.010,
) -> StateHistoryBuffer:
    buffer = StateHistoryBuffer(
        history_duration_s=2.0,
        max_state_interval_s=0.020,
    )
    count = int(round((stop_s - start_s) / step_s))
    for timestamp in np.linspace(start_s, stop_s, count + 1):
        buffer.append(
            timestamp_s=float(timestamp),
            q_hip_rad=float(timestamp),
            q_knee_rad=float(2.0 * timestamp),
            dq_hip_rad_s=float(3.0 * timestamp),
            dq_knee_rad_s=float(4.0 * timestamp),
            ddq_hip_rad_s2=float(5.0 * timestamp),
            ddq_knee_rad_s2=float(6.0 * timestamp),
        )
    return buffer


def _match(
    buffer,
    *,
    arrival: float = 1.0,
    current: float = 1.0,
    delay: float = 0.020,
    sample: float | None = None,
    reliable: bool = False,
    valid: bool = True,
    stale: bool = False,
    matcher: CausalSampleMatcher | None = None,
):
    return match_arriving_wrench_to_state(
        buffer,
        arrival_timestamp_s=arrival,
        current_timestamp_s=current,
        fx_observed_n=12.0,
        fz_observed_n=-7.0,
        estimated_delay_s=delay,
        sample_timestamp_s=sample,
        sample_timestamp_reliable=reliable,
        wrench_valid=valid,
        wrench_is_stale=stale,
        matcher=matcher,
    )


def test_reliable_sample_timestamp_has_priority_over_estimated_delay() -> None:
    result = _match(
        _history(),
        delay=0.010,
        sample=0.950,
        reliable=True,
    )

    assert result.valid
    assert result.used_true_sample_timestamp
    assert np.isclose(result.target_state_timestamp_s, 0.950)
    assert np.isclose(result.matched_state_timestamp_s, 0.950)
    assert np.isclose(result.wrench_age_s, 0.050)
    assert np.isclose(result.state_wrench_skew_s, 0.0, atol=1e-12)
    assert np.isclose(result.q_hip_rad, 0.950)
    assert np.isclose(result.q_knee_rad, 1.900)
    assert result.fx_observed_n == 12.0
    assert result.fz_observed_n == -7.0
    assert 0.0 < result.confidence <= 1.0


def test_missing_reliable_timestamp_falls_back_to_arrival_minus_delay() -> None:
    result = _match(_history(), delay=0.025)

    assert result.valid
    assert not result.used_true_sample_timestamp
    assert np.isnan(result.sample_timestamp_s)
    assert np.isclose(result.target_state_timestamp_s, 0.975)
    assert np.isclose(result.matched_state_timestamp_s, 0.975)
    assert np.isclose(result.wrench_age_s, 0.025)
    assert result.used_interpolation
    assert np.isclose(result.q_hip_rad, 0.975)


def test_result_contains_finite_state_derivatives_force_and_timing() -> None:
    result = _match(_history(), delay=0.025)
    serialized = result.as_dict()
    required_finite = (
        "arrival_timestamp_s",
        "target_state_timestamp_s",
        "matched_state_timestamp_s",
        "current_timestamp_s",
        "wrench_age_s",
        "processing_age_s",
        "state_wrench_skew_s",
        "state_match_error_s",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "fx_observed_n",
        "fz_observed_n",
        "confidence",
    )

    assert result.valid
    assert all(np.isfinite(serialized[name]) for name in required_finite)
    assert result.causal_state_only


@pytest.mark.parametrize(
    ("kwargs", "expected_reason"),
    (
        ({"valid": False}, "wrench_dropout_or_non_finite_force"),
        ({"stale": True}, "stale_or_frozen_wrench"),
        (
            {"arrival": 1.010, "current": 1.000},
            "arrival_timestamp_in_future",
        ),
        ({"delay": 0.101}, "wrench_age_limit_exceeded"),
    ),
)
def test_dropout_stale_future_arrival_and_excessive_age_are_rejected(
    kwargs: dict[str, object],
    expected_reason: str,
) -> None:
    result = _match(_history(), **kwargs)

    assert not result.valid
    assert result.invalid_reason == expected_reason
    assert result.confidence == 0.0
    assert np.isnan(result.q_hip_rad)


def test_non_finite_force_is_treated_as_dropout() -> None:
    result = match_arriving_wrench_to_state(
        _history(),
        arrival_timestamp_s=1.0,
        current_timestamp_s=1.0,
        fx_observed_n=np.nan,
        fz_observed_n=-7.0,
        estimated_delay_s=0.020,
    )

    assert not result.valid
    assert result.invalid_reason == "wrench_dropout_or_non_finite_force"


def test_duplicate_reliable_sample_timestamp_is_rejected() -> None:
    matcher = CausalSampleMatcher()
    buffer = _history(stop_s=0.980)
    first = _match(
        buffer,
        arrival=0.980,
        current=0.980,
        sample=0.950,
        reliable=True,
        matcher=matcher,
    )
    duplicate = _match(
        buffer,
        arrival=1.000,
        current=1.000,
        sample=0.950,
        reliable=True,
        matcher=matcher,
    )

    assert first.valid
    assert matcher.last_reliable_sample_timestamp_s == 0.950
    assert not duplicate.valid
    assert duplicate.invalid_reason == "duplicate_wrench_sample_timestamp"


def test_unique_late_sample_timestamp_is_not_misclassified_as_freeze() -> None:
    matcher = CausalSampleMatcher()
    # 缓存只到第一条 wrench 的到达时刻，后续迟到事件仍可查询这段历史。
    buffer = _history(stop_s=0.980)
    newer_content = _match(
        buffer,
        arrival=0.980,
        current=0.980,
        sample=0.950,
        reliable=True,
        matcher=matcher,
    )
    late_unique_content = _match(
        buffer,
        arrival=1.000,
        current=1.000,
        sample=0.930,
        reliable=True,
        matcher=matcher,
    )

    assert newer_content.valid
    assert late_unique_content.valid
    assert late_unique_content.invalid_reason == ""
    assert np.isclose(late_unique_content.q_hip_rad, 0.930)
    # “last”保持为最新内容时间，不因合法迟到样本倒退。
    assert matcher.last_reliable_sample_timestamp_s == 0.950
    assert matcher.seen_reliable_sample_count == 2


def test_duplicate_is_rejected_even_after_unique_out_of_order_arrival() -> None:
    matcher = CausalSampleMatcher()
    buffer = _history(stop_s=0.970)
    for arrival, sample in ((0.970, 0.940), (0.990, 0.960), (1.000, 0.950)):
        assert _match(
            buffer,
            arrival=arrival,
            current=arrival,
            sample=sample,
            reliable=True,
            matcher=matcher,
        ).valid

    repeated = _match(
        buffer,
        arrival=1.000,
        current=1.000,
        sample=0.950,
        reliable=True,
        matcher=matcher,
    )

    assert not repeated.valid
    assert repeated.invalid_reason == "duplicate_wrench_sample_timestamp"


def test_duplicate_timestamp_detection_uses_numeric_tolerance() -> None:
    matcher = CausalSampleMatcher()
    buffer = _history(stop_s=0.980)
    first = _match(
        buffer,
        arrival=0.980,
        current=0.980,
        sample=0.950,
        reliable=True,
        matcher=matcher,
    )
    repeated = _match(
        buffer,
        arrival=1.000,
        current=1.000,
        sample=0.950 + 0.5e-12,
        reliable=True,
        matcher=matcher,
    )

    assert first.valid
    assert not repeated.valid
    assert repeated.invalid_reason == "duplicate_wrench_sample_timestamp"


def test_cache_expired_target_is_rejected() -> None:
    buffer = _history(start_s=0.90, stop_s=0.94)
    result = _match(
        buffer,
        arrival=0.940,
        current=0.940,
        sample=0.890,
        reliable=True,
    )

    assert not result.valid
    assert result.invalid_reason == "state_history_expired"


@dataclass(frozen=True)
class _FakeStateMatch:
    valid: bool = True
    reason: str = ""
    matched_timestamp_s: float = 0.986
    time_error_s: float = 0.006
    used_interpolation: bool = False
    q_hip_rad: float = 0.2
    q_knee_rad: float = 0.4
    dq_hip_rad_s: float = 0.6
    dq_knee_rad_s: float = 0.8
    ddq_hip_rad_s2: float = 1.0
    ddq_knee_rad_s2: float = 1.2


class _LargeErrorBuffer:
    oldest_timestamp_s = 0.8
    newest_timestamp_s = 1.0

    def query(self, *args, **kwargs):
        return _FakeStateMatch()


def test_state_match_error_over_5ms_is_rejected() -> None:
    result = _match(_LargeErrorBuffer(), delay=0.020)

    assert not result.valid
    assert result.invalid_reason == "state_match_error_limit_exceeded"
    assert np.isclose(result.state_match_error_s, 0.006)


class _QueryMustNotRun:
    oldest_timestamp_s = 0.8
    newest_timestamp_s = 1.010

    def query(self, *args, **kwargs):
        raise AssertionError("future state cache must be rejected before query")


def test_state_after_wrench_arrival_is_never_used() -> None:
    result = _match(_QueryMustNotRun(), arrival=1.0, current=1.0)

    assert not result.valid
    assert result.invalid_reason == "state_history_contains_future_state"


def test_reliable_timestamp_must_be_finite_and_not_after_arrival() -> None:
    missing = _match(_history(), sample=None, reliable=True)
    future = _match(_history(), sample=1.001, reliable=True)

    assert not missing.valid
    assert missing.invalid_reason == "non_finite_timestamp_or_delay"
    assert not future.valid
    assert future.invalid_reason == "wrench_target_timestamp_after_arrival"


def test_true_timestamp_match_has_higher_confidence_than_delay_fallback() -> None:
    reliable = _match(
        _history(),
        sample=0.950,
        reliable=True,
        delay=0.050,
    )
    fallback = _match(_history(), delay=0.050)

    assert reliable.valid
    assert fallback.valid
    assert reliable.confidence > fallback.confidence


def test_matcher_configuration_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="max_wrench_age_s"):
        CausalSampleMatcher(max_wrench_age_s=0.0)
    with pytest.raises(ValueError, match="max_match_error_s"):
        CausalSampleMatcher(max_match_error_s=-0.001)
    with pytest.raises(ValueError, match="max_state_interval_s"):
        CausalSampleMatcher(max_state_interval_s=np.nan)
