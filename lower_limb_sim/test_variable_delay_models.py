"""阶段 4.5B 变化 wrench 延迟和缺失场景的基础回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lower_limb_sim.config import (
    variable_delay_random_seed,
    variable_delay_scenarios,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.identification_dataset import build_identification_dataset
from lower_limb_sim.variable_delay_models import (
    VARIABLE_DELAY_SCENARIOS,
    apply_variable_delay_scenario,
)
from lower_limb_sim.windowed_delay_tracker import (
    sanitize_windowed_delay_input,
)


TRAJECTORY_KEYS = ["trajectory_family", "speed_profile"]
EXPECTED_SCENARIOS = {
    "fixed_16ms",
    "piecewise_delay",
    "gradual_drift",
    "jitter_low",
    "jitter_medium",
    "bimodal_delay",
    "long_tail",
    "stale_freeze",
    "dropout_5pct",
    "combined_realistic",
}


@pytest.fixture(scope="module")
def variable_delay_suite():
    clean = build_identification_dataset(
        get_dynamic_subject("baseline"),
        "clean",
        sampling_frequency_hz=100.0,
    )
    results = {
        scenario: apply_variable_delay_scenario(
            clean,
            scenario,
            random_seed=variable_delay_random_seed,
        )
        for scenario in VARIABLE_DELAY_SCENARIOS
    }
    return {"clean": clean, "results": results}


def test_all_ten_configured_scenarios_are_covered(
    variable_delay_suite,
) -> None:
    assert set(variable_delay_scenarios) == EXPECTED_SCENARIOS
    assert set(VARIABLE_DELAY_SCENARIOS) == EXPECTED_SCENARIOS
    assert set(variable_delay_suite["results"]) == EXPECTED_SCENARIOS
    assert len(VARIABLE_DELAY_SCENARIOS) == 10


@pytest.mark.parametrize("scenario", tuple(variable_delay_scenarios))
def test_timestamp_and_true_delay_fields_are_consistent(
    variable_delay_suite,
    scenario: str,
) -> None:
    application = variable_delay_suite["results"][scenario]
    dataframe = application.dataframe
    required = {
        "state_timestamp_s",
        "wrench_arrival_timestamp_s",
        "wrench_timestamp_s",
        "wrench_sample_timestamp_s",
        "true_delay_s",
        "generated_base_delay_s",
        "wrench_age_s",
        "state_wrench_skew_s",
        "delay_scenario",
        "noise_scenario",
    }

    assert required.issubset(dataframe.columns)
    assert np.allclose(dataframe["state_timestamp_s"], dataframe["time_s"])
    assert np.allclose(
        dataframe["wrench_arrival_timestamp_s"],
        dataframe["time_s"],
    )
    assert np.allclose(dataframe["wrench_timestamp_s"], dataframe["time_s"])
    reconstructed_delay = (
        dataframe["wrench_arrival_timestamp_s"]
        - dataframe["wrench_sample_timestamp_s"]
    )
    assert np.allclose(
        dataframe["true_delay_s"],
        reconstructed_delay,
        atol=1e-12,
    )
    assert np.allclose(dataframe["wrench_age_s"], reconstructed_delay)
    assert np.allclose(
        dataframe["state_wrench_skew_s"],
        reconstructed_delay,
    )
    assert np.isfinite(
        dataframe[
            [
                "wrench_sample_timestamp_s",
                "true_delay_s",
                "generated_base_delay_s",
            ]
        ].to_numpy(dtype=float)
    ).all()
    assert (dataframe["true_delay_s"] >= 0.0).all()
    assert (
        dataframe["wrench_sample_timestamp_s"]
        <= dataframe["wrench_arrival_timestamp_s"] + 1e-12
    ).all()
    assert dataframe["delay_scenario"].eq(scenario).all()
    assert dataframe["noise_scenario"].eq(
        f"variable_delay/{scenario}"
    ).all()
    assert application.metadata["delay_scenario"] == scenario
    assert application.metadata["future_wrench_used"] is False
    assert (
        application.metadata["positive_delay_definition"]
        == "wrench_arrival_timestamp_s - wrench_sample_timestamp_s"
    )


@pytest.mark.parametrize("scenario", tuple(variable_delay_scenarios))
def test_arrival_and_state_time_are_strictly_increasing_within_trajectory(
    variable_delay_suite,
    scenario: str,
) -> None:
    dataframe = variable_delay_suite["results"][scenario].dataframe
    monotonic_columns = (
        "time_s",
        "state_timestamp_s",
        "wrench_arrival_timestamp_s",
        "wrench_timestamp_s",
    )

    for _, trajectory in dataframe.groupby(TRAJECTORY_KEYS, sort=False):
        for column in monotonic_columns:
            assert (
                np.diff(trajectory[column].to_numpy(dtype=float)) > 0.0
            ).all()


def test_fixed_and_piecewise_delay_profiles(
    variable_delay_suite,
) -> None:
    fixed = variable_delay_suite["results"]["fixed_16ms"].dataframe
    piecewise = variable_delay_suite["results"]["piecewise_delay"].dataframe

    assert np.allclose(fixed["true_delay_s"], 0.016, atol=1e-12)
    expected = np.select(
        [piecewise["time_s"] < 4.0, piecewise["time_s"] < 8.0],
        [0.008, 0.024],
        default=0.016,
    )
    assert np.allclose(
        piecewise["generated_base_delay_s"],
        expected,
        atol=1e-12,
    )
    assert set(
        np.round(piecewise["generated_base_delay_s"], decimals=6)
    ) == {0.008, 0.016, 0.024}


def test_gradual_drift_is_monotonic_from_8ms_to_32ms(
    variable_delay_suite,
) -> None:
    dataframe = variable_delay_suite["results"]["gradual_drift"].dataframe

    for _, trajectory in dataframe.groupby(TRAJECTORY_KEYS, sort=False):
        delay = trajectory["generated_base_delay_s"].to_numpy(dtype=float)
        assert np.isclose(delay[0], 0.008)
        assert np.isclose(delay[-1], 0.032)
        assert (np.diff(delay) >= -1e-12).all()


def test_low_and_medium_jitter_have_expected_scale(
    variable_delay_suite,
) -> None:
    low = variable_delay_suite["results"]["jitter_low"].dataframe[
        "generated_base_delay_s"
    ]
    medium = variable_delay_suite["results"]["jitter_medium"].dataframe[
        "generated_base_delay_s"
    ]

    assert low.between(0.004, 0.040).all()
    assert medium.between(0.004, 0.060).all()
    assert abs(float(low.mean()) - 0.016) < 0.0005
    assert abs(float(medium.mean()) - 0.024) < 0.001
    assert 0.0015 < float(low.std()) < 0.0025
    assert 0.0040 < float(medium.std()) < 0.0060
    assert medium.std() > low.std()


def test_bimodal_delay_contains_both_requested_modes(
    variable_delay_suite,
) -> None:
    delay = variable_delay_suite["results"]["bimodal_delay"].dataframe[
        "generated_base_delay_s"
    ]
    lower_mode = delay < 0.015
    upper_mode = delay > 0.025

    assert lower_mode.mean() > 0.60
    assert 0.10 < upper_mode.mean() < 0.30
    assert (~lower_mode & ~upper_mode).mean() < 0.10


def test_long_tail_samples_are_explicit_and_between_50_and_105ms(
    variable_delay_suite,
) -> None:
    application = variable_delay_suite["results"]["long_tail"]
    dataframe = application.dataframe
    tail = dataframe["is_long_tail"].astype(bool)

    assert tail.any()
    assert dataframe.loc[
        tail,
        "generated_base_delay_s",
    ].between(0.050, 0.105).all()
    assert (
        dataframe.loc[~tail, "generated_base_delay_s"] <= 0.032
    ).all()
    assert int(tail.sum()) == application.metadata["long_tail_samples"]
    assert application.metadata["maximum_true_delay_s"] >= 0.050


def test_dropout_samples_are_invalid_nan_and_close_to_five_percent(
    variable_delay_suite,
) -> None:
    application = variable_delay_suite["results"]["dropout_5pct"]
    dataframe = application.dataframe
    dropout = dataframe["is_dropout"].astype(bool)

    assert 0.03 < float(dropout.mean()) < 0.07
    assert (~dataframe.loc[dropout, "sample_valid"].astype(bool)).all()
    assert dataframe.loc[
        dropout,
        ["fx_observed_n", "fz_observed_n"],
    ].isna().all(axis=None)
    assert dataframe.loc[
        dropout,
        ["tau_measured_hip_nm", "tau_measured_knee_nm"],
    ].isna().all(axis=None)
    assert dataframe.loc[dropout, "invalid_reason"].str.contains(
        "wrench_dropout"
    ).all()
    assert int(dropout.sum()) == application.metadata["dropout_samples"]


def test_stale_freeze_samples_are_invalid_and_keep_repeated_timestamp(
    variable_delay_suite,
) -> None:
    application = variable_delay_suite["results"]["stale_freeze"]
    dataframe = application.dataframe
    stale = dataframe["is_stale"].astype(bool)

    assert stale.any()
    assert dataframe["wrench_is_stale"].astype(bool).equals(stale)
    assert (~dataframe.loc[stale, "sample_valid"].astype(bool)).all()
    assert dataframe.loc[stale, "invalid_reason"].str.contains(
        "stale_wrench_freeze"
    ).all()
    assert dataframe.loc[stale, "freeze_duration_s"].max() >= 0.10
    duplicated_sample_time = False
    for _, trajectory in dataframe.groupby(TRAJECTORY_KEYS, sort=False):
        stale_trajectory = trajectory.loc[trajectory["is_stale"].astype(bool)]
        if len(stale_trajectory) >= 2:
            duplicated_sample_time |= (
                stale_trajectory["wrench_sample_timestamp_s"].nunique() == 1
            )
    assert duplicated_sample_time
    assert int(stale.sum()) == application.metadata["stale_samples"]


def test_combined_realistic_contains_all_declared_fault_types(
    variable_delay_suite,
) -> None:
    application = variable_delay_suite["results"]["combined_realistic"]
    dataframe = application.dataframe

    assert dataframe["is_dropout"].any()
    assert dataframe["is_stale"].any()
    assert dataframe["is_long_tail"].any()
    invalid_fault = (
        dataframe["is_dropout"].astype(bool)
        | dataframe["is_stale"].astype(bool)
    )
    assert (~dataframe.loc[invalid_fault, "sample_valid"].astype(bool)).all()
    assert application.metadata["dropout_samples"] > 0
    assert application.metadata["stale_samples"] > 0
    assert application.metadata["long_tail_samples"] > 0


@pytest.mark.parametrize("scenario", tuple(variable_delay_scenarios))
def test_fixed_seed_reproduces_every_scenario_exactly(
    variable_delay_suite,
    scenario: str,
) -> None:
    reference = variable_delay_suite["results"][scenario]
    repeated = apply_variable_delay_scenario(
        variable_delay_suite["clean"],
        scenario,
        random_seed=variable_delay_random_seed,
    )

    assert_frame_equal(reference.dataframe, repeated.dataframe, check_exact=True)
    assert reference.metadata == repeated.metadata


@pytest.mark.parametrize("scenario", tuple(variable_delay_scenarios))
def test_all_valid_samples_are_finite_and_not_dropout_or_stale(
    variable_delay_suite,
    scenario: str,
) -> None:
    dataframe = variable_delay_suite["results"][scenario].dataframe
    valid = dataframe["sample_valid"].astype(bool)
    finite_columns = [
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "fx_observed_n",
        "fz_observed_n",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
        "force_magnitude_observed_n",
        "state_timestamp_s",
        "wrench_arrival_timestamp_s",
        "wrench_sample_timestamp_s",
        "true_delay_s",
    ]

    assert valid.any()
    assert np.isfinite(
        dataframe.loc[valid, finite_columns].to_numpy(dtype=float)
    ).all()
    assert (~dataframe.loc[valid, "is_dropout"].astype(bool)).all()
    assert (~dataframe.loc[valid, "is_stale"].astype(bool)).all()
    assert dataframe.loc[valid, "force_mapping_valid"].astype(bool).all()


def test_true_delay_fields_are_evaluation_only(
    variable_delay_suite,
) -> None:
    application = variable_delay_suite["results"]["combined_realistic"]
    dataframe = application.dataframe
    training = dataframe.loc[dataframe["dataset_split"].eq("train")].copy()
    sanitized = sanitize_windowed_delay_input(training)
    evaluation_only = {
        "true_delay_s",
        "generated_base_delay_s",
        "wrench_age_s",
        "state_wrench_skew_s",
        "delay_scenario",
        "noise_scenario",
        "is_long_tail",
        "freeze_duration_s",
    }

    assert application.metadata["true_delay_available_to_estimators"] is False
    assert (
        application.metadata[
            "sample_timestamp_is_observed_simulated_device_time"
        ]
        is True
    )
    assert evaluation_only.issubset(dataframe.columns)
    assert set(sanitized.columns).isdisjoint(evaluation_only)
    assert sanitized.attrs == {}
    # 设备本身可靠提供的 sample timestamp 是观测量，不是真值标签。
    assert "wrench_sample_timestamp_s" in sanitized.columns


def test_unknown_scenario_and_invalid_seed_are_rejected(
    variable_delay_suite,
) -> None:
    clean = variable_delay_suite["clean"]
    with pytest.raises(ValueError, match="Unknown variable delay scenario"):
        apply_variable_delay_scenario(clean, "not_a_scenario")
    with pytest.raises(ValueError, match="random_seed must be an integer"):
        apply_variable_delay_scenario(
            clean,
            "fixed_16ms",
            random_seed=1.5,
        )
