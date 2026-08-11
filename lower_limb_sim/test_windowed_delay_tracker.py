"""阶段 4.5B 因果滑动窗口延迟跟踪测试。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.windowed_delay_tracker import (
    WindowedDelayTracker,
    sanitize_windowed_delay_input,
)


REQUIRED_RESULT_FIELDS = {
    "estimated_delay_ms",
    "delay_confidence",
    "delay_update_valid",
    "delay_update_reason",
    "search_boundary_hit",
    "window_start_s",
    "window_end_s",
    "effective_sample_count",
    "excitation_score",
    "best_validation_rmse_nm",
    "second_best_validation_rmse_nm",
    "delay_value_held",
    "maximum_delay_change_limited",
}


def _stream(
    *,
    duration_s: float = 10.0,
    sampling_frequency_hz: float = 20.0,
    moving: bool = True,
    dataset_split: str = "train",
    timestamp_delay_ms: float | None = None,
) -> pd.DataFrame:
    count = int(round(duration_s * sampling_frequency_hz)) + 1
    time_s = np.linspace(0.0, duration_s, count)
    if moving:
        angular_frequency = 2.0 * np.pi * 0.5
        q_hip = 0.6 + 0.20 * np.sin(angular_frequency * time_s)
        q_knee = 0.9 + 0.25 * np.sin(
            angular_frequency * time_s + 0.35
        )
        dq_hip = 0.20 * angular_frequency * np.cos(
            angular_frequency * time_s
        )
        dq_knee = 0.25 * angular_frequency * np.cos(
            angular_frequency * time_s + 0.35
        )
        ddq_hip = -0.20 * angular_frequency**2 * np.sin(
            angular_frequency * time_s
        )
        ddq_knee = -0.25 * angular_frequency**2 * np.sin(
            angular_frequency * time_s + 0.35
        )
    else:
        q_hip = np.full(count, 0.6)
        q_knee = np.full(count, 0.9)
        dq_hip = np.zeros(count)
        dq_knee = np.zeros(count)
        ddq_hip = np.zeros(count)
        ddq_knee = np.zeros(count)
    dataframe = pd.DataFrame(
        {
            "dataset_split": dataset_split,
            "stream_timestamp_s": time_s,
            "state_timestamp_s": time_s,
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "dq_hip_rad_s": dq_hip,
            "dq_knee_rad_s": dq_knee,
            "ddq_hip_rad_s2": ddq_hip,
            "ddq_knee_rad_s2": ddq_knee,
            "fx_observed_n": 20.0 + 2.0 * np.sin(time_s),
            "fz_observed_n": 35.0 + 3.0 * np.cos(time_s),
            "sample_valid": True,
            "force_mapping_valid": True,
            "wrench_is_stale": False,
        }
    )
    if timestamp_delay_ms is not None:
        dataframe["wrench_source_timestamp_s"] = (
            time_s - timestamp_delay_ms / 1000.0
        )
        dataframe["wrench_arrival_timestamp_s"] = time_s
    return dataframe


def _schedule_scorer(
    schedule_ms: Callable[[float], float],
    *,
    observed_windows: list[tuple[float, float]] | None = None,
):
    """建立确定性的凸 RMSE 曲线；仅供测试注入。"""

    def score(window: pd.DataFrame, candidates_ms: np.ndarray) -> np.ndarray:
        start = float(window["tracker_time_s"].min())
        end = float(window["tracker_time_s"].max())
        if observed_windows is not None:
            observed_windows.append((start, end))
        target = float(schedule_ms(end))
        return 0.10 + 0.01 * (candidates_ms - target) ** 2

    return score


def _fast_tracker(schedule_ms, **kwargs) -> WindowedDelayTracker:
    options = {
        "alpha": 1.0,
        "maximum_delay_change_ms": 200.0,
        "minimum_confidence": 0.50,
        "scoring_callback": _schedule_scorer(schedule_ms),
    }
    options.update(kwargs)
    return WindowedDelayTracker(**options)


def test_fixed_delay_and_default_window_schedule_are_tracked_causally() -> None:
    observed_windows: list[tuple[float, float]] = []
    tracker = WindowedDelayTracker(
        alpha=1.0,
        maximum_delay_change_ms=200.0,
        scoring_callback=_schedule_scorer(
            lambda _: 24.0,
            observed_windows=observed_windows,
        ),
    )

    result = tracker.track(_stream())

    assert set(result["raw_estimated_delay_ms"]) == {24.0}
    assert set(result["estimated_delay_ms"]) == {24.0}
    assert result["delay_update_valid"].all()
    assert np.allclose(result["window_end_s"] - result["window_start_s"], 2.0)
    assert np.allclose(np.diff(result["window_end_s"]), 0.5)
    assert np.allclose(
        [end for _, end in observed_windows],
        result["window_end_s"],
    )
    # 回调收到的最后一个样本不晚于当前更新时刻。
    assert all(
        observed_end <= update_end + 1e-12
        for (_, observed_end), update_end in zip(
            observed_windows,
            result["window_end_s"],
        )
    )


def test_piecewise_delay_change_is_followed_on_the_next_update() -> None:
    tracker = _fast_tracker(lambda end: 10.0 if end < 5.0 else 40.0)

    result = tracker.track(_stream(duration_s=8.0))

    before = result.loc[result["window_end_s"] < 5.0]
    after = result.loc[result["window_end_s"] >= 5.0]
    assert (before["estimated_delay_ms"] == 10.0).all()
    assert (after["estimated_delay_ms"] == 40.0).all()


def test_linear_delay_drift_is_resolved_to_the_nearest_grid_point() -> None:
    schedule = lambda end: 5.0 + 3.0 * end
    tracker = _fast_tracker(schedule)

    result = tracker.track(_stream(duration_s=9.0))
    expected = result["window_end_s"].map(schedule).to_numpy(dtype=float)

    assert np.max(
        np.abs(result["raw_estimated_delay_ms"].to_numpy() - expected)
    ) <= 0.5 + 1e-12
    assert np.all(np.diff(result["estimated_delay_ms"]) >= 0.0)


def test_alpha_smoothing_reduces_deterministic_jitter() -> None:
    def jittered(end: float) -> float:
        update_index = int(round(end / 0.5))
        return 20.0 + (6.0 if update_index % 2 else -6.0)

    tracker = _fast_tracker(jittered, alpha=0.25)
    result = tracker.track(_stream(duration_s=12.0))
    settled = result.iloc[5:]

    assert (
        settled["estimated_delay_ms"].std()
        < settled["raw_estimated_delay_ms"].std()
    )
    assert set(result["raw_estimated_delay_ms"]) == {14.0, 26.0}


def test_still_window_holds_previous_delay_without_calling_scorer() -> None:
    def forbidden_scorer(window, candidates):
        raise AssertionError("low-excitation windows must not be scored")

    tracker = WindowedDelayTracker(
        initial_delay_ms=17.0,
        scoring_callback=forbidden_scorer,
    )

    result = tracker.track(_stream(moving=False, duration_s=5.0))

    assert (result["estimated_delay_ms"] == 17.0).all()
    assert (~result["delay_update_valid"]).all()
    assert result["delay_value_held"].all()
    assert (
        result["delay_update_reason"] == "insufficient_excitation"
    ).all()
    assert (result["excitation_score"] == 0.0).all()


def test_flat_search_curve_is_low_confidence_and_holds_value() -> None:
    tracker = WindowedDelayTracker(
        initial_delay_ms=9.0,
        minimum_confidence=0.80,
        scoring_callback=lambda window, candidates: np.ones(len(candidates)),
    )

    result = tracker.track(_stream(duration_s=4.0))

    assert (result["estimated_delay_ms"] == 9.0).all()
    assert (~result["delay_update_valid"]).all()
    assert result["delay_value_held"].all()
    assert (result["delay_update_reason"] == "low_confidence").all()
    assert (result["delay_confidence"] < 0.80).all()


def test_maximum_delay_change_is_limited_and_explicitly_marked() -> None:
    tracker = _fast_tracker(
        lambda _: 60.0,
        maximum_delay_change_ms=5.0,
        initial_delay_ms=0.0,
    )

    result = tracker.track(_stream(duration_s=4.0))

    assert result["estimated_delay_ms"].iloc[0] == 5.0
    assert np.allclose(np.diff(result["estimated_delay_ms"]), 5.0)
    assert result["maximum_delay_change_limited"].all()
    assert (
        result["delay_update_reason"]
        == "updated_with_maximum_delay_change_limit"
    ).all()
    assert (np.abs(result["applied_delay_change_ms"]) <= 5.0).all()


def test_out_of_range_delay_sets_search_boundary_flag() -> None:
    tracker = _fast_tracker(
        lambda _: 100.0,
        minimum_confidence=0.0,
    )

    result = tracker.track(_stream(duration_s=4.0))

    assert (result["raw_estimated_delay_ms"] == 80.0).all()
    assert result["search_boundary_hit"].all()
    assert (result["candidate_count"] == 131).all()
    assert tracker.last_search_curve["candidate_delay_ms"].iloc[0] == -50.0
    assert tracker.last_search_curve["candidate_delay_ms"].iloc[-1] == 80.0


def test_tracking_is_exactly_reproducible() -> None:
    scorer = _schedule_scorer(
        lambda end: 18.0 + 2.0 * np.sin(0.5 * end)
    )
    first = WindowedDelayTracker(
        alpha=0.4,
        maximum_delay_change_ms=7.0,
        scoring_callback=scorer,
    ).track(_stream(duration_s=7.0))
    second = WindowedDelayTracker(
        alpha=0.4,
        maximum_delay_change_ms=7.0,
        scoring_callback=scorer,
    ).track(_stream(duration_s=7.0))

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_test_and_validation_splits_are_rejected() -> None:
    tracker = _fast_tracker(lambda _: 20.0)

    with pytest.raises(ValueError, match="test split is forbidden"):
        tracker.track(_stream(dataset_split="test"))
    with pytest.raises(ValueError, match="only train or online"):
        tracker.track(_stream(dataset_split="validation"))


def test_required_output_fields_are_present() -> None:
    result = _fast_tracker(lambda _: 16.0).track(
        _stream(duration_s=3.0)
    )

    assert REQUIRED_RESULT_FIELDS.issubset(result.columns)
    assert result.loc[0, "effective_sample_count"] >= 20
    assert 0.0 <= result.loc[0, "delay_confidence"] <= 1.0


def test_default_timestamp_scorer_uses_reliable_arrival_difference() -> None:
    tracker = WindowedDelayTracker(
        alpha=1.0,
        maximum_delay_change_ms=200.0,
    )

    result = tracker.track(
        _stream(
            duration_s=4.0,
            dataset_split="online",
            timestamp_delay_ms=25.0,
        )
    )

    assert (result["raw_estimated_delay_ms"] == 25.0).all()
    assert (result["estimated_delay_ms"] == 25.0).all()
    assert result["score_source"].str.startswith(
        "timestamp_difference:"
    ).all()
    assert (result["validation_score_unit"] == "ms_timestamp_residual").all()


def test_late_unique_sample_timestamps_use_arrival_order_for_windows() -> None:
    dataframe = _stream(
        duration_s=4.0,
        dataset_split="online",
    ).drop(columns=["stream_timestamp_s"])
    arrival = dataframe["state_timestamp_s"].to_numpy(dtype=float)
    alternating_delay_ms = np.where(
        np.arange(len(dataframe)) % 2,
        80.0,
        0.0,
    )
    dataframe["wrench_arrival_timestamp_s"] = arrival
    dataframe["wrench_sample_timestamp_s"] = (
        arrival - alternating_delay_ms / 1000.0
    )
    # 80 ms长尾大于50 ms采样间隔，因此sample timestamp会按到达顺序乱序。
    assert (np.diff(dataframe["wrench_sample_timestamp_s"]) < 0.0).any()

    sanitized = sanitize_windowed_delay_input(dataframe)
    result = WindowedDelayTracker(
        alpha=1.0,
        maximum_delay_change_ms=200.0,
        minimum_confidence=0.0,
    ).track(dataframe)

    assert np.allclose(
        sanitized["tracker_time_s"],
        dataframe["wrench_arrival_timestamp_s"],
    )
    assert np.all(np.diff(sanitized["tracker_time_s"]) > 0.0)
    assert not result.empty
    assert (result["effective_sample_count"] >= 20).all()


def test_true_delay_and_age_fields_never_reach_scoring_callback() -> None:
    seen_columns: list[set[str]] = []

    def auditing_scorer(
        window: pd.DataFrame,
        candidates_ms: np.ndarray,
    ) -> np.ndarray:
        seen_columns.append(set(window.columns))
        return 0.1 + 0.01 * (candidates_ms - 22.0) ** 2

    dataframe = _stream(duration_s=4.0)
    dataframe["true_delay_ms"] = 75.0
    dataframe["wrench_age_s"] = 0.075
    dataframe["wrench_delay_s"] = 0.075
    dataframe["noise_scenario"] = "answer_is_75ms"
    dataframe["subject_id"] = "truth_lookup_decoy"
    dataframe.attrs["true_delay_ms"] = 75.0

    result = WindowedDelayTracker(
        alpha=1.0,
        maximum_delay_change_ms=200.0,
        scoring_callback=auditing_scorer,
    ).track(dataframe)

    assert (result["estimated_delay_ms"] == 22.0).all()
    forbidden = {
        "true_delay_ms",
        "wrench_age_s",
        "wrench_delay_s",
        "noise_scenario",
        "subject_id",
    }
    assert seen_columns
    assert all(columns.isdisjoint(forbidden) for columns in seen_columns)
    sanitized = sanitize_windowed_delay_input(dataframe)
    assert sanitized.attrs == {}
    assert set(sanitized.columns).isdisjoint(forbidden)
