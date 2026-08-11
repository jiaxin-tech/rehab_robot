"""有限状态历史缓冲区的严格时间与查询边界测试。"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from lower_limb_sim.state_history_buffer import (
    DEFAULT_HISTORY_DURATION_S,
    StateHistoryBuffer,
    StateMatchResult,
)


def _state_values(timestamp_s: float) -> tuple[float, ...]:
    """六个彼此不同的仿射状态，便于验证所有插值字段。"""

    return (
        0.40 + 2.0 * timestamp_s,
        0.20 + 1.0 * timestamp_s,
        0.10 + 3.0 * timestamp_s,
        -0.20 + 4.0 * timestamp_s,
        0.30 - 5.0 * timestamp_s,
        -0.40 + 6.0 * timestamp_s,
    )


def _append(buffer: StateHistoryBuffer, timestamp_s: float) -> None:
    buffer.append(timestamp_s, *_state_values(timestamp_s))


def test_default_buffer_is_finite_and_at_least_two_seconds() -> None:
    buffer = StateHistoryBuffer()

    assert DEFAULT_HISTORY_DURATION_S >= 2.0
    assert buffer.history_duration_s >= 2.0
    assert buffer.max_state_interval_s > 0.0
    assert len(buffer) == 0
    assert buffer.oldest_timestamp_s is None
    assert buffer.newest_timestamp_s is None


def test_append_requires_strictly_increasing_timestamps() -> None:
    buffer = StateHistoryBuffer()
    _append(buffer, 0.01)

    with pytest.raises(ValueError, match="strictly increasing"):
        _append(buffer, 0.01)
    with pytest.raises(ValueError, match="strictly increasing"):
        _append(buffer, 0.0)


@pytest.mark.parametrize("invalid_index", range(7))
def test_append_rejects_nonfinite_timestamp_or_state(
    invalid_index: int,
) -> None:
    buffer = StateHistoryBuffer()
    values = [0.01, *_state_values(0.01)]
    values[invalid_index] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        buffer.append(*values)


def test_new_samples_prune_states_older_than_history_duration() -> None:
    buffer = StateHistoryBuffer(history_duration_s=0.03)
    for timestamp_s in np.arange(5, dtype=float) * 0.01:
        _append(buffer, float(timestamp_s))

    assert buffer.oldest_timestamp_s == pytest.approx(0.01)
    assert buffer.newest_timestamp_s == pytest.approx(0.04)
    assert len(buffer) == 4

    expired = buffer.query(0.0, method="nearest")
    assert not expired.valid
    assert expired.reason == "state_history_expired"
    assert np.isnan(expired.matched_timestamp_s)
    assert np.isnan(expired.q_hip_rad)


def test_default_duration_bounds_long_running_buffer_size() -> None:
    buffer = StateHistoryBuffer()
    for index in range(501):
        _append(buffer, index * 0.01)

    assert buffer.newest_timestamp_s == pytest.approx(5.0)
    assert buffer.oldest_timestamp_s >= 3.0 - 1e-12
    assert len(buffer) <= 202


def test_empty_history_returns_explicit_no_bracket_reason() -> None:
    result = StateHistoryBuffer().query(0.0)

    assert not result.valid
    assert result.reason == "state_history_no_bracket"
    assert not result.used_interpolation
    assert np.isnan(result.time_error_s)


def test_query_after_newest_state_is_rejected_as_future() -> None:
    buffer = StateHistoryBuffer()
    _append(buffer, 0.0)
    _append(buffer, 0.01)

    result = buffer.query(0.011, method="nearest")

    assert not result.valid
    assert result.reason == "state_history_future_query"
    assert np.isnan(result.q_knee_rad)


def test_nearest_query_returns_timestamp_and_absolute_time_error() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)
    _append(buffer, 0.02)

    result = buffer.query(0.006, method="nearest")

    assert result.valid
    assert result.reason == ""
    assert not result.used_interpolation
    assert result.matched_timestamp_s == pytest.approx(0.0)
    assert result.time_error_s == pytest.approx(0.006)
    assert result.q_hip_rad == pytest.approx(_state_values(0.0)[0])


def test_nearest_query_uses_older_sample_for_exact_tie() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)
    _append(buffer, 0.02)

    result = buffer.query(0.01, method="nearest")

    assert result.valid
    assert result.matched_timestamp_s == pytest.approx(0.0)
    assert result.time_error_s == pytest.approx(0.01)


def test_nearest_query_honors_maximum_time_error() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)
    _append(buffer, 0.05)

    result = buffer.query(0.024, method="nearest")

    assert not result.valid
    assert result.reason == "state_history_gap_exceeded"
    assert result.matched_timestamp_s == pytest.approx(0.0)
    assert result.time_error_s == pytest.approx(0.024)


def test_linear_interpolation_returns_all_six_affine_states() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)
    _append(buffer, 0.02)

    result = buffer.query(0.007, method="linear_interpolation")
    expected = _state_values(0.007)
    observed = (
        result.q_hip_rad,
        result.q_knee_rad,
        result.dq_hip_rad_s,
        result.dq_knee_rad_s,
        result.ddq_hip_rad_s2,
        result.ddq_knee_rad_s2,
    )

    assert result.valid
    assert result.used_interpolation
    assert result.matched_timestamp_s == pytest.approx(0.007)
    assert result.time_error_s == 0.0
    assert observed == pytest.approx(expected)
    # q_hip、q_knee 独立插值；小腿绝对角仍由 q_hip - q_knee 得到。
    assert result.q_hip_rad - result.q_knee_rad == pytest.approx(
        expected[0] - expected[1]
    )


def test_exact_linear_query_does_not_claim_interpolation() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.005)
    _append(buffer, 0.0)
    _append(buffer, 0.10)

    result = buffer.query(0.10, method="linear_interpolation")

    assert result.valid
    assert not result.used_interpolation
    assert result.matched_timestamp_s == pytest.approx(0.10)
    assert result.time_error_s == 0.0


def test_linear_interpolation_rejects_bracket_over_maximum_interval() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)
    _append(buffer, 0.03)

    result = buffer.query(0.01, method="linear_interpolation")

    assert not result.valid
    assert result.reason == "state_history_gap_exceeded"
    assert not result.used_interpolation
    assert np.isnan(result.matched_timestamp_s)


def test_query_can_apply_an_explicit_interval_override() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)
    _append(buffer, 0.03)

    rejected = buffer.query(0.01, method="linear_interpolation")
    accepted = buffer.query(
        0.01,
        method="linear_interpolation",
        max_state_interval_s=0.03,
    )

    assert not rejected.valid
    assert accepted.valid
    assert accepted.used_interpolation


def test_query_only_uses_states_that_have_already_been_cached() -> None:
    buffer = StateHistoryBuffer(max_state_interval_s=0.02)
    _append(buffer, 0.0)

    before_right_bracket_arrives = buffer.query(
        0.005,
        method="linear_interpolation",
    )
    _append(buffer, 0.01)
    after_right_bracket_arrives = buffer.query(
        0.005,
        method="linear_interpolation",
    )

    assert not before_right_bracket_arrives.valid
    assert before_right_bracket_arrives.reason == "state_history_future_query"
    assert after_right_bracket_arrives.valid
    assert after_right_bracket_arrives.used_interpolation


def test_query_result_has_stable_required_fields() -> None:
    expected_fields = {
        "target_timestamp_s",
        "matched_timestamp_s",
        "time_error_s",
        "valid",
        "reason",
        "used_interpolation",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    }

    assert {field.name for field in fields(StateMatchResult)} == expected_fields


def test_clear_removes_all_cached_history() -> None:
    buffer = StateHistoryBuffer()
    _append(buffer, 0.0)
    buffer.clear()

    assert len(buffer) == 0
    assert buffer.oldest_timestamp_s is None
    assert buffer.newest_timestamp_s is None
    assert buffer.query(0.0).reason == "state_history_no_bracket"


@pytest.mark.parametrize(
    ("method", "maximum_interval"),
    (
        ("unsupported", None),
        ("nearest", 0.0),
        ("linear_interpolation", np.inf),
    ),
)
def test_query_rejects_invalid_method_or_interval(
    method: str,
    maximum_interval: float | None,
) -> None:
    buffer = StateHistoryBuffer()
    _append(buffer, 0.0)

    with pytest.raises(ValueError):
        buffer.query(
            0.0,
            method=method,
            max_state_interval_s=maximum_interval,
        )


def test_query_rejects_nonfinite_target_timestamp() -> None:
    buffer = StateHistoryBuffer()
    _append(buffer, 0.0)

    with pytest.raises(ValueError, match="target_timestamp_s"):
        buffer.query(np.nan)
