"""Stage 4.5D tests for measured-angle derivative estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lower_limb_sim.derivative_estimation import (
    DERIVATIVE_METHODS,
    DERIVATIVE_OUTPUT_COLUMNS,
    DerivativeEstimationConfig,
    causal_backward_difference,
    causal_filter_and_difference,
    central_difference_offline,
    estimate_joint_derivatives,
    savitzky_golay_offline,
)


DT_S = 0.01


def _smooth_angles(
    count: int = 401,
    *,
    noise_scale_rad: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    time_s = np.arange(count, dtype=float) * DT_S
    q_hip = 0.4 + 0.25 * np.sin(1.1 * time_s)
    q_knee = 0.9 - 0.20 * np.cos(0.8 * time_s)
    deterministic_noise = (
        np.sin(31.0 * time_s) + 0.4 * np.sin(47.0 * time_s + 0.3)
    )
    dataframe = pd.DataFrame(
        {
            "trajectory_id": "trajectory_a",
            "phase": "flexion",
            "time_s": time_s,
            "q_hip_est_rad": q_hip + noise_scale_rad * deterministic_noise,
            "q_knee_est_rad": (
                q_knee - 0.8 * noise_scale_rad * deterministic_noise
            ),
            "ik_valid": True,
        }
    )
    truth = {
        "dq_hip": 0.25 * 1.1 * np.cos(1.1 * time_s),
        "ddq_hip": -0.25 * 1.1**2 * np.sin(1.1 * time_s),
        "dq_knee": 0.20 * 0.8 * np.sin(0.8 * time_s),
        "ddq_knee": 0.20 * 0.8**2 * np.cos(0.8 * time_s),
    }
    return dataframe, truth


def _combined_derivative_rmse(
    result: pd.DataFrame,
    truth: dict[str, np.ndarray],
) -> float:
    valid = result["derivative_valid"].to_numpy(dtype=bool)
    estimated = np.column_stack(
        (
            result.loc[valid, "dq_hip_est_rad_s"],
            result.loc[valid, "dq_knee_est_rad_s"],
            result.loc[valid, "ddq_hip_est_rad_s2"],
            result.loc[valid, "ddq_knee_est_rad_s2"],
        )
    )
    expected = np.column_stack(
        (
            truth["dq_hip"][valid],
            truth["dq_knee"][valid],
            truth["ddq_hip"][valid],
            truth["ddq_knee"][valid],
        )
    )
    return float(np.sqrt(np.mean(np.square(estimated - expected))))


def test_ground_truth_velocity_and_acceleration_columns_are_not_read() -> None:
    dataframe, _ = _smooth_angles()
    first = dataframe.assign(
        dq_hip_rad_s=1.0e9,
        dq_knee_rad_s=-1.0e9,
        ddq_hip_rad_s2=3.0e8,
        ddq_knee_rad_s2=-3.0e8,
    )
    second = dataframe.assign(
        dq_hip_rad_s=-7.0,
        dq_knee_rad_s=11.0,
        ddq_hip_rad_s2=13.0,
        ddq_knee_rad_s2=17.0,
    )

    result_a = savitzky_golay_offline(first)
    result_b = savitzky_golay_offline(second)

    for column in DERIVATIVE_OUTPUT_COLUMNS:
        if result_a.dataframe[column].dtype.kind in "fc":
            assert np.allclose(
                result_a.dataframe[column],
                result_b.dataframe[column],
                equal_nan=True,
            )
        else:
            assert result_a.dataframe[column].equals(
                result_b.dataframe[column]
            )
    assert result_a.metadata["ground_truth_dq_ddq_used"] is False
    assert not any(
        column.startswith(("dq_", "ddq_"))
        for column in result_a.metadata["input_columns_read"]
    )


def test_explicit_true_or_derivative_angle_sources_are_rejected() -> None:
    dataframe, _ = _smooth_angles()
    dataframe["q_hip_true_rad"] = dataframe["q_hip_est_rad"]
    dataframe["q_knee_true_rad"] = dataframe["q_knee_est_rad"]
    with pytest.raises(ValueError, match="prohibited"):
        central_difference_offline(
            dataframe,
            angle_columns=("q_hip_true_rad", "q_knee_true_rad"),
        )
    with pytest.raises(ValueError, match="prohibited"):
        central_difference_offline(
            dataframe.assign(dq_angle=np.arange(len(dataframe))),
            angle_columns=("dq_angle", "q_knee_est_rad"),
        )


@pytest.mark.parametrize(
    "algorithm",
    (central_difference_offline, savitzky_golay_offline),
)
def test_offline_methods_are_explicitly_marked_as_using_future_samples(
    algorithm,
) -> None:
    dataframe, _ = _smooth_angles()
    result = algorithm(dataframe)

    assert result.metadata["offline_only"] is True
    assert result.metadata["causal"] is False
    assert result.metadata["uses_future_samples"] is True
    assert result.dataframe["uses_future_samples"].all()
    assert np.allclose(
        result.dataframe.loc[
            result.dataframe["derivative_valid"], "filter_delay_s"
        ],
        0.0,
    )


@pytest.mark.parametrize(
    "algorithm",
    (causal_backward_difference, causal_filter_and_difference),
)
def test_causal_methods_are_prefix_invariant_and_never_use_future(
    algorithm,
) -> None:
    dataframe, _ = _smooth_angles()
    cutoff = 220
    changed_future = dataframe.copy(deep=True)
    changed_future.loc[cutoff + 1 :, "q_hip_est_rad"] += 100.0
    changed_future.loc[cutoff + 1 :, "q_knee_est_rad"] -= 80.0

    original = algorithm(dataframe)
    changed = algorithm(changed_future)
    derivative_columns = list(DERIVATIVE_OUTPUT_COLUMNS[:4])

    assert np.allclose(
        original.dataframe.loc[:cutoff, derivative_columns],
        changed.dataframe.loc[:cutoff, derivative_columns],
        equal_nan=True,
    )
    assert original.metadata["offline_only"] is False
    assert original.metadata["causal"] is True
    assert original.metadata["uses_future_samples"] is False
    assert not original.dataframe["uses_future_samples"].any()


@pytest.mark.parametrize("method", DERIVATIVE_METHODS)
def test_smooth_trajectory_derivative_error_is_finite_and_bounded(
    method: str,
) -> None:
    dataframe, truth = _smooth_angles()
    result = estimate_joint_derivatives(dataframe, method=method)
    rmse = _combined_derivative_rmse(result.dataframe, truth)

    assert result.dataframe["derivative_valid"].any()
    assert np.isfinite(rmse)
    if method.endswith("_offline"):
        assert rmse < 0.015
    else:
        assert rmse < 0.035
    assert np.isfinite(
        result.dataframe.loc[
            result.dataframe["derivative_valid"],
            list(DERIVATIVE_OUTPUT_COLUMNS[:4]),
        ].to_numpy(dtype=float)
    ).all()


def test_increasing_angle_noise_does_not_falsely_reduce_derivative_error() -> None:
    clean, truth = _smooth_angles(noise_scale_rad=0.0)
    low, _ = _smooth_angles(noise_scale_rad=2.0e-4)
    high, _ = _smooth_angles(noise_scale_rad=2.0e-3)

    clean_error = _combined_derivative_rmse(
        savitzky_golay_offline(clean).dataframe, truth
    )
    low_error = _combined_derivative_rmse(
        savitzky_golay_offline(low).dataframe, truth
    )
    high_error = _combined_derivative_rmse(
        savitzky_golay_offline(high).dataframe, truth
    )

    assert low_error > clean_error
    assert high_error > low_error


def test_long_time_gap_is_split_and_never_crossed_by_differentiation() -> None:
    time_s = np.concatenate((np.arange(40) * DT_S, 1.0 + np.arange(40) * DT_S))
    # A position offset after the missing interval makes cross-gap
    # differentiation obviously wrong; each local segment still has slope 2.
    q_hip = np.concatenate((2.0 * time_s[:40], 50.0 + 2.0 * time_s[40:]))
    q_knee = np.concatenate((-time_s[:40], -30.0 - time_s[40:]))
    dataframe = pd.DataFrame(
        {
            "time_s": time_s,
            "q_hip_est_rad": q_hip,
            "q_knee_est_rad": q_knee,
            "trajectory_id": "gap_test",
            "phase": "flexion",
            "ik_valid": True,
        }
    )

    result = causal_backward_difference(
        dataframe,
        maximum_time_gap_s=0.025,
    )
    output = result.dataframe

    assert result.metadata["long_gap_boundary_count"] == 1
    assert not output.loc[40:41, "derivative_valid"].any()
    assert "long_time_gap_boundary" in output.loc[40, "derivative_reason"]
    assert output.loc[42, "derivative_valid"]
    assert output.loc[42, "dq_hip_est_rad_s"] == pytest.approx(2.0)
    assert output.loc[42, "dq_knee_est_rad_s"] == pytest.approx(-1.0)
    assert abs(output.loc[42, "ddq_hip_est_rad_s2"]) < 1e-9


def test_invalid_angle_sample_splits_offline_segments() -> None:
    dataframe, _ = _smooth_angles(count=101)
    dataframe.loc[50, "ik_valid"] = False
    dataframe.loc[51:, "q_hip_est_rad"] += 25.0
    dataframe.loc[51:, "q_knee_est_rad"] -= 10.0

    result = central_difference_offline(dataframe)
    output = result.dataframe

    assert not output.loc[50, "derivative_valid"]
    assert "source_angle_invalid" in output.loc[50, "derivative_reason"]
    assert output.loc[[49, 51], "derivative_valid"].all()
    assert np.max(np.abs(output.loc[[49, 51], "dq_hip_est_rad_s"])) < 1.0


def test_phase_boundaries_are_never_crossed() -> None:
    time = np.arange(31, dtype=float) * DT_S
    first = pd.DataFrame(
        {
            "trajectory_id": "same",
            "phase": "flexion",
            "time_s": time,
            "q_hip_est_rad": 2.0 * time,
            "q_knee_est_rad": -time,
            "ik_valid": True,
        }
    )
    second = pd.DataFrame(
        {
            "trajectory_id": "same",
            "phase": "extension",
            "time_s": time,
            "q_hip_est_rad": 100.0 + 3.0 * time,
            "q_knee_est_rad": -100.0 + 4.0 * time,
            "ik_valid": True,
        }
    )
    result = central_difference_offline(
        pd.concat((first, second), ignore_index=True)
    )

    assert result.metadata["segment_count"] == 2
    assert np.allclose(result.dataframe.loc[:30, "dq_hip_est_rad_s"], 2.0)
    assert np.allclose(result.dataframe.loc[31:, "dq_hip_est_rad_s"], 3.0)
    assert np.allclose(result.dataframe.loc[31:, "dq_knee_est_rad_s"], 4.0)


@pytest.mark.parametrize("method", DERIVATIVE_METHODS)
def test_fixed_input_and_configuration_are_reproducible(method: str) -> None:
    dataframe, _ = _smooth_angles(noise_scale_rad=5e-4)
    config = DerivativeEstimationConfig(
        savgol_window_length=21,
        savgol_polynomial_order=3,
        causal_filter_window_length=5,
        maximum_time_gap_s=0.025,
    )

    first = estimate_joint_derivatives(dataframe, method=method, config=config)
    second = estimate_joint_derivatives(dataframe, method=method, config=config)

    assert_frame_equal(first.dataframe, second.dataframe, check_exact=True)
    assert first.metadata == second.metadata


def test_array_input_has_unified_schema_and_quality_flags() -> None:
    dataframe, _ = _smooth_angles(count=101)
    result = central_difference_offline(
        dataframe["time_s"].to_numpy(),
        dataframe["q_hip_est_rad"].to_numpy(),
        dataframe["q_knee_est_rad"].to_numpy(),
    )

    assert set(DERIVATIVE_OUTPUT_COLUMNS).issubset(result.dataframe.columns)
    assert result.metadata["array_input"] is True
    assert result.dataframe["derivative_valid"].all()


def test_nonuniform_savgol_segment_is_invalid_instead_of_silently_misused() -> None:
    dataframe, _ = _smooth_angles(count=101)
    dataframe.loc[60:, "time_s"] += 0.001
    result = savitzky_golay_offline(
        dataframe,
        maximum_time_gap_s=0.03,
    )

    assert not result.dataframe["derivative_valid"].any()
    assert result.dataframe["derivative_reason"].eq(
        "nonuniform_time_for_savgol"
    ).all()

