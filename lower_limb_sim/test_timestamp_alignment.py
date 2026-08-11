"""阶段 4.5A 时间戳延迟合成与对齐的快速边界测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lower_limb_sim.timestamp_alignment import (
    align_wrench_to_state_timestamps,
    synthesize_delayed_wrench_dataset,
)


DT_S = 0.010
MAX_GAP_S = 0.020


def _observation_dataframe(
    time_s: np.ndarray | None = None,
    *,
    trajectory_family: str = "coupled",
    speed_profile: str = "nominal",
    force_offset_n: float = 0.0,
) -> pd.DataFrame:
    """建立100 Hz量级、力随时间线性变化的最小合法观测表。"""

    if time_s is None:
        time_s = np.arange(10, dtype=float) * DT_S
    time_s = np.asarray(time_s, dtype=float)
    count = len(time_s)
    return pd.DataFrame(
        {
            "trajectory_id": "identification_excitation_trajectory",
            "trajectory_family": trajectory_family,
            "speed_profile": speed_profile,
            "phase": "flexion",
            "time_s": time_s,
            "trajectory_sample_index": np.arange(count, dtype=int),
            "dataset_split": "train",
            "q_hip_rad": np.deg2rad(35.0) + 0.05 * time_s,
            "q_knee_rad": np.deg2rad(60.0) + 0.03 * time_s,
            "dq_hip_rad_s": np.full(count, 0.05),
            "dq_knee_rad_s": np.full(count, 0.03),
            "ddq_hip_rad_s2": np.zeros(count),
            "ddq_knee_rad_s2": np.zeros(count),
            "fx_observed_n": force_offset_n + 10.0 + 200.0 * time_s,
            "fz_observed_n": force_offset_n - 5.0 + 100.0 * time_s,
            "sample_valid": np.ones(count, dtype=bool),
            "force_mapping_valid": np.ones(count, dtype=bool),
            "wrench_is_stale": np.zeros(count, dtype=bool),
            "invalid_reason": "",
        }
    )


def _timestamped_dataframe(
    dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = _observation_dataframe() if dataframe is None else dataframe
    return synthesize_delayed_wrench_dataset(
        source,
        0.0,
        max_interpolation_gap_s=MAX_GAP_S,
    )


@pytest.mark.parametrize("mode", ("offline_only", "causal_history"))
def test_zero_delay_alignment_is_a_noop(mode: str) -> None:
    clean = _observation_dataframe()
    timestamped = _timestamped_dataframe(clean)

    result = align_wrench_to_state_timestamps(
        timestamped,
        0.0,
        mode=mode,
        max_interpolation_gap_s=MAX_GAP_S,
    )
    aligned = result.dataframe

    assert aligned["sample_valid"].all()
    assert np.allclose(aligned["fx_observed_n"], clean["fx_observed_n"])
    assert np.allclose(aligned["fz_observed_n"], clean["fz_observed_n"])
    assert np.allclose(aligned["state_timestamp_s"], clean["time_s"])
    assert np.allclose(aligned["wrench_timestamp_s"], clean["time_s"])
    assert np.allclose(aligned["wrench_age_s"], 0.0)
    assert np.allclose(aligned["state_wrench_skew_s"], 0.0)
    assert not aligned["alignment_used_future"].any()
    assert result.metadata["extrapolation_used"] is False


def test_fractional_positive_delay_uses_past_signal_without_extrapolation() -> None:
    clean = _observation_dataframe()
    delay_s = 0.016

    delayed = synthesize_delayed_wrench_dataset(
        clean,
        delay_s,
        max_interpolation_gap_s=MAX_GAP_S,
    )
    has_history = clean["time_s"].to_numpy() >= delay_s

    assert (~delayed.loc[~has_history, "sample_valid"]).all()
    assert delayed.loc[
        ~has_history,
        ["fx_observed_n", "fz_observed_n"],
    ].isna().all(axis=None)
    source_time = clean.loc[has_history, "time_s"].to_numpy() - delay_s
    assert np.allclose(
        delayed.loc[has_history, "fx_observed_n"],
        10.0 + 200.0 * source_time,
    )
    assert np.allclose(
        delayed.loc[has_history, "fz_observed_n"],
        -5.0 + 100.0 * source_time,
    )
    assert np.allclose(delayed["wrench_age_s"], delay_s)
    assert (
        delayed.loc[~has_history, "alignment_invalid_reason"]
        .str.contains("no_history_or_gap")
        .all()
    )


def test_synthetic_physical_delay_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        synthesize_delayed_wrench_dataset(
            _observation_dataframe(),
            -DT_S,
        )


def test_negative_alignment_candidate_is_supported_without_extrapolation() -> None:
    timestamped = _timestamped_dataframe()

    result = align_wrench_to_state_timestamps(
        timestamped,
        -DT_S,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    )
    aligned = result.dataframe
    valid = aligned["sample_valid"].to_numpy(dtype=bool)

    assert not valid[0]
    assert valid[1:].all()
    assert np.allclose(
        aligned.loc[valid, "fx_observed_n"].to_numpy(),
        timestamped["fx_observed_n"].to_numpy()[:-1],
    )
    assert np.allclose(aligned["applied_delay_compensation_s"], -DT_S)
    assert result.metadata["assumed_delay_s"] == pytest.approx(-DT_S)
    assert result.metadata["extrapolation_used"] is False


def test_offline_compensation_recovers_delayed_affine_signal() -> None:
    clean = _observation_dataframe()
    delayed = synthesize_delayed_wrench_dataset(
        clean,
        DT_S,
        max_interpolation_gap_s=MAX_GAP_S,
    )

    result = align_wrench_to_state_timestamps(
        delayed,
        DT_S,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    )
    aligned = result.dataframe
    valid = aligned["sample_valid"].to_numpy(dtype=bool)

    assert valid.any()
    assert np.allclose(
        aligned.loc[valid, "fx_observed_n"],
        clean.loc[valid, "fx_observed_n"],
    )
    assert np.allclose(
        aligned.loc[valid, "fz_observed_n"],
        clean.loc[valid, "fz_observed_n"],
    )
    assert aligned.loc[valid, "alignment_used_future"].all()
    assert result.metadata["offline_only"] is True
    assert result.metadata["causal"] is False
    assert result.metadata["future_samples_used"] == int(valid.sum())


def test_causal_history_never_uses_future_wrench_samples() -> None:
    delayed = synthesize_delayed_wrench_dataset(
        _observation_dataframe(),
        DT_S,
        max_interpolation_gap_s=MAX_GAP_S,
    )

    result = align_wrench_to_state_timestamps(
        delayed,
        DT_S,
        mode="causal_history",
        max_interpolation_gap_s=MAX_GAP_S,
    )
    aligned = result.dataframe
    valid = aligned["sample_valid"].to_numpy(dtype=bool)

    assert valid.any()
    assert not aligned["alignment_used_future"].any()
    assert np.allclose(aligned["alignment_future_lookahead_s"], 0.0)
    assert (
        aligned.loc[valid, "wrench_timestamp_s"]
        <= aligned.loc[valid, "alignment_timestamp_s"] + 1e-12
    ).all()
    assert np.allclose(
        aligned.loc[valid, "fx_observed_n"],
        aligned.loc[valid, "fx_raw_observed_n"],
    )
    assert result.metadata["causal"] is True
    assert result.metadata["future_samples_used"] == 0


def test_offline_interpolation_accepts_exactly_20ms_dropout_gap() -> None:
    raw = _timestamped_dataframe(
        _observation_dataframe(np.array([0.0, 0.01, 0.02]))
    )
    raw.loc[1, ["fx_observed_n", "fz_observed_n"]] = np.nan
    raw.loc[1, "sample_valid"] = False
    raw.loc[1, "invalid_reason"] = "wrench_dropout"

    aligned = align_wrench_to_state_timestamps(
        raw,
        0.0,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    ).dataframe

    assert aligned.loc[1, "sample_valid"]
    assert aligned.loc[1, "alignment_gap_s"] == pytest.approx(MAX_GAP_S)
    assert aligned.loc[1, "fx_observed_n"] == pytest.approx(12.0)
    assert aligned.loc[1, "fz_observed_n"] == pytest.approx(-4.0)
    assert aligned.loc[1, "alignment_used_future"]


def test_offline_interpolation_rejects_gap_larger_than_20ms() -> None:
    raw = _timestamped_dataframe(
        _observation_dataframe(np.array([0.0, 0.01, 0.02, 0.03]))
    )
    raw.loc[[1, 2], ["fx_observed_n", "fz_observed_n"]] = np.nan
    raw.loc[[1, 2], "sample_valid"] = False

    aligned = align_wrench_to_state_timestamps(
        raw,
        0.0,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    ).dataframe

    assert (~aligned.loc[[1, 2], "sample_valid"]).all()
    assert np.allclose(aligned.loc[[1, 2], "alignment_gap_s"], 0.03)
    assert (
        aligned.loc[[1, 2], "alignment_invalid_reason"]
        .str.contains("alignment_gap_exceeded")
        .all()
    )


def test_causal_hold_is_limited_to_20ms() -> None:
    raw = _timestamped_dataframe(
        _observation_dataframe(
            np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        )
    )
    raw.loc[[1, 2, 3], ["fx_observed_n", "fz_observed_n"]] = np.nan
    raw.loc[[1, 2, 3], "sample_valid"] = False

    aligned = align_wrench_to_state_timestamps(
        raw,
        0.0,
        mode="causal_history",
        max_interpolation_gap_s=MAX_GAP_S,
    ).dataframe

    assert aligned.loc[1, "sample_valid"]
    assert aligned.loc[2, "sample_valid"]
    assert aligned.loc[2, "alignment_gap_s"] == pytest.approx(MAX_GAP_S)
    assert not aligned.loc[3, "sample_valid"]
    assert aligned.loc[3, "alignment_gap_s"] == pytest.approx(0.03)
    assert "alignment_gap_exceeded" in aligned.loc[
        3,
        "alignment_invalid_reason",
    ]
    assert not aligned["alignment_used_future"].any()


def test_finite_stale_sample_is_not_used_as_interpolation_source() -> None:
    raw = _timestamped_dataframe(
        _observation_dataframe(np.array([0.0, 0.01, 0.02]))
    )
    raw.loc[1, "wrench_is_stale"] = True
    raw.loc[1, "fx_observed_n"] = 1e6
    raw.loc[1, "fz_observed_n"] = -1e6

    aligned = align_wrench_to_state_timestamps(
        raw,
        0.0,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    ).dataframe

    assert aligned.loc[1, "sample_valid"]
    assert aligned.loc[1, "fx_observed_n"] == pytest.approx(12.0)
    assert aligned.loc[1, "fz_observed_n"] == pytest.approx(-4.0)
    assert aligned.loc[1, "alignment_gap_s"] == pytest.approx(MAX_GAP_S)


def test_long_stale_freeze_is_not_silently_interpolated() -> None:
    time_s = np.arange(26, dtype=float) * DT_S
    raw = _timestamped_dataframe(_observation_dataframe(time_s))
    stale_indices = np.arange(1, len(raw) - 1)
    raw.loc[stale_indices, "wrench_is_stale"] = True
    raw.loc[stale_indices, "fx_observed_n"] = raw.loc[0, "fx_observed_n"]
    raw.loc[stale_indices, "fz_observed_n"] = raw.loc[0, "fz_observed_n"]

    aligned = align_wrench_to_state_timestamps(
        raw,
        0.0,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    ).dataframe

    assert (~aligned.loc[stale_indices, "sample_valid"]).all()
    assert (
        aligned.loc[stale_indices, "alignment_invalid_reason"]
        .str.contains("alignment_gap_exceeded")
        .all()
    )
    assert aligned.loc[[0, len(aligned) - 1], "sample_valid"].all()


@pytest.mark.parametrize(
    "missing_column",
    (
        "state_timestamp_s",
        "wrench_timestamp_s",
        "q_hip_rad",
        "ddq_knee_rad_s2",
        "fx_observed_n",
        "sample_valid",
        "force_mapping_valid",
    ),
)
def test_alignment_rejects_missing_required_columns(
    missing_column: str,
) -> None:
    raw = _timestamped_dataframe().drop(columns=[missing_column])

    with pytest.raises(ValueError, match="missing"):
        align_wrench_to_state_timestamps(raw, 0.0)


@pytest.mark.parametrize(
    ("timestamp_column", "invalid_kind"),
    (
        ("state_timestamp_s", "duplicate"),
        ("wrench_timestamp_s", "duplicate"),
        ("state_timestamp_s", "non_finite"),
        ("wrench_timestamp_s", "non_finite"),
    ),
)
def test_alignment_rejects_nonmonotonic_or_nonfinite_timestamps(
    timestamp_column: str,
    invalid_kind: str,
) -> None:
    raw = _timestamped_dataframe()
    if invalid_kind == "duplicate":
        raw.loc[2, timestamp_column] = raw.loc[1, timestamp_column]
    else:
        raw.loc[2, timestamp_column] = np.nan

    with pytest.raises(ValueError, match="finite and strictly increasing"):
        align_wrench_to_state_timestamps(raw, 0.0)


def test_valid_outputs_have_required_audit_columns_and_finite_values() -> None:
    aligned = align_wrench_to_state_timestamps(
        _timestamped_dataframe(),
        0.0,
        mode="offline_only",
        max_interpolation_gap_s=MAX_GAP_S,
    ).dataframe
    required_columns = {
        "state_timestamp_s",
        "wrench_timestamp_s",
        "wrench_age_s",
        "state_wrench_skew_s",
        "alignment_timestamp_s",
        "wrench_effective_timestamp_s",
        "applied_delay_compensation_s",
        "alignment_valid",
        "alignment_mode",
        "alignment_offline_only",
        "alignment_used_future",
        "alignment_future_lookahead_s",
        "alignment_gap_s",
        "alignment_invalid_reason",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
    }
    assert required_columns.issubset(aligned.columns)

    valid = aligned["sample_valid"].to_numpy(dtype=bool)
    finite_columns = [
        "state_timestamp_s",
        "wrench_timestamp_s",
        "alignment_timestamp_s",
        "wrench_effective_timestamp_s",
        "applied_delay_compensation_s",
        "alignment_future_lookahead_s",
        "alignment_gap_s",
        "fx_observed_n",
        "fz_observed_n",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
    ]
    assert np.isfinite(
        aligned.loc[valid, finite_columns].to_numpy(dtype=float)
    ).all()
    assert np.all(np.diff(aligned["state_timestamp_s"]) > 0.0)
    assert np.all(np.diff(aligned["wrench_timestamp_s"]) > 0.0)


def test_alignment_is_reproducible_and_does_not_mutate_input() -> None:
    source = _observation_dataframe()
    original = source.copy(deep=True)

    first_delayed = synthesize_delayed_wrench_dataset(source, 0.016)
    second_delayed = synthesize_delayed_wrench_dataset(source, 0.016)
    first = align_wrench_to_state_timestamps(
        first_delayed,
        0.016,
        mode="offline_only",
    )
    second = align_wrench_to_state_timestamps(
        second_delayed,
        0.016,
        mode="offline_only",
    )

    assert_frame_equal(source, original)
    assert_frame_equal(first_delayed, second_delayed)
    assert_frame_equal(first.dataframe, second.dataframe)
    assert first.metadata == second.metadata


def test_delay_history_never_crosses_trajectory_groups() -> None:
    first = _observation_dataframe(
        np.arange(5, dtype=float) * DT_S,
        trajectory_family="coupled",
        force_offset_n=0.0,
    )
    second = _observation_dataframe(
        np.arange(5, dtype=float) * DT_S,
        trajectory_family="hip_dominant",
        force_offset_n=1000.0,
    )
    combined = pd.concat((first, second), ignore_index=True)

    delayed = synthesize_delayed_wrench_dataset(
        combined,
        DT_S,
        max_interpolation_gap_s=MAX_GAP_S,
    )
    first_rows = delayed.groupby(
        ["trajectory_family", "speed_profile"],
        sort=False,
    ).head(1)

    assert (~first_rows["sample_valid"]).all()
    assert first_rows[["fx_observed_n", "fz_observed_n"]].isna().all(axis=None)
