"""阶段 4.5A 固定 wrench 延迟自动估计的离线测试。"""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import lower_limb_sim.delay_estimator as delay_estimator_module
from lower_limb_sim.config import L1, L2
from lower_limb_sim.delay_estimator import (
    DELAY_ESTIMATOR_INPUT_COLUMNS,
    DelayEstimationResult,
    estimate_wrench_delay,
    sanitize_delay_estimation_input,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.identification_dataset import (
    build_identification_dataset,
    split_identification_dataset,
)
from lower_limb_sim.parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
)
from lower_limb_sim.timestamp_alignment import synthesize_delayed_wrench_dataset


EXACT_GRID_DELAYS_MS = (0.0, 8.0, 16.0, 24.0, 32.0, 40.0)
SUBGRID_DELAYS_MS = (7.4, 15.6, 31.5)
SEARCH_CURVE_REQUIRED_COLUMNS = {
    "candidate_delay_ms",
    "train_rmse_nm",
    "validation_rmse_nm",
    "train_valid_samples",
    "validation_valid_samples",
    "valid_train_samples",
    "valid_validation_samples",
    "valid_sample_count",
    "train_coverage_ratio",
    "validation_coverage_ratio",
    "optimizer_success",
    "optimizer_cost",
    "validation_torque_rmse_hip_nm",
    "validation_torque_rmse_knee_nm",
    "validation_torque_rmse_combined_nm",
    "failure_reason",
    "selected",
    *(f"train_estimate_{name}" for name in PARAMETER_NAMES),
}


class _DelayTestHarness:
    """缓存代价较高的 101 候选搜索，避免断言间重复运行。"""

    def __init__(self) -> None:
        subject = get_dynamic_subject("baseline")
        self.template = baseline_template_from_dynamic_subject(subject)
        # 50 Hz 是 max interpolation gap 允许的最低均匀采样率；仍使用
        # 1 ms 延迟网格，同时将本文件的总运行时间控制在合理范围。
        self.clean = build_identification_dataset(
            subject,
            "clean",
            sampling_frequency_hz=50.0,
        )
        self._splits: dict[float, dict[str, pd.DataFrame]] = {}
        self._results: dict[float, DelayEstimationResult] = {}

    @staticmethod
    def _key(delay_ms: float) -> float:
        return round(float(delay_ms), 6)

    def splits(self, delay_ms: float) -> dict[str, pd.DataFrame]:
        key = self._key(delay_ms)
        if key not in self._splits:
            delayed = synthesize_delayed_wrench_dataset(
                self.clean,
                key / 1000.0,
            )
            self._splits[key] = split_identification_dataset(delayed)
        return {
            split: dataframe.copy(deep=True)
            for split, dataframe in self._splits[key].items()
        }

    def result(self, delay_ms: float) -> DelayEstimationResult:
        key = self._key(delay_ms)
        if key not in self._results:
            splits = self.splits(key)
            self._results[key] = estimate_wrench_delay(
                splits["train"],
                splits["validation"],
                self.template,
                L1,
                L2,
            )
        return self._results[key]


@pytest.fixture(scope="module")
def delay_harness() -> _DelayTestHarness:
    return _DelayTestHarness()


def _assert_same_result(
    first: DelayEstimationResult,
    second: DelayEstimationResult,
) -> None:
    assert first.estimated_delay_s == second.estimated_delay_s
    assert first.estimated_parameters == second.estimated_parameters
    pd.testing.assert_frame_equal(
        first.search_curve,
        second.search_curve,
        check_exact=True,
    )


@pytest.mark.parametrize("true_delay_ms", EXACT_GRID_DELAYS_MS)
def test_integer_grid_delays_are_recovered_within_one_millisecond(
    delay_harness: _DelayTestHarness,
    true_delay_ms: float,
) -> None:
    result = delay_harness.result(true_delay_ms)

    assert result.optimizer_success
    assert abs(result.estimated_delay_ms - true_delay_ms) <= 1.0
    assert result.estimated_delay_ms == pytest.approx(
        round(result.estimated_delay_ms),
        abs=1e-12,
    )


@pytest.mark.parametrize("true_delay_ms", SUBGRID_DELAYS_MS)
def test_subgrid_delays_are_recovered_at_one_millisecond_grid_accuracy(
    delay_harness: _DelayTestHarness,
    true_delay_ms: float,
) -> None:
    result = delay_harness.result(true_delay_ms)

    assert result.optimizer_success
    assert abs(result.estimated_delay_ms - true_delay_ms) <= 1.0
    assert result.estimated_delay_ms == pytest.approx(
        round(result.estimated_delay_ms),
        abs=1e-12,
    )


def test_default_search_has_exactly_101_ordered_candidates(
    delay_harness: _DelayTestHarness,
) -> None:
    curve = delay_harness.result(16.0).search_curve
    candidates = curve["candidate_delay_ms"].to_numpy(dtype=float)

    assert len(curve) == 101
    assert candidates[0] == -50.0
    assert candidates[-1] == 50.0
    assert np.allclose(np.diff(candidates), 1.0, atol=1e-12, rtol=0.0)
    assert curve["selected"].astype(bool).sum() == 1


def test_search_curve_contains_all_audit_and_selection_fields(
    delay_harness: _DelayTestHarness,
) -> None:
    curve = delay_harness.result(16.0).search_curve

    assert SEARCH_CURVE_REQUIRED_COLUMNS.issubset(curve.columns)
    selected = curve.loc[curve["selected"].astype(bool)]
    assert len(selected) == 1
    assert np.isfinite(
        selected[
            [
                "optimizer_cost",
                "validation_torque_rmse_hip_nm",
                "validation_torque_rmse_knee_nm",
                "validation_torque_rmse_combined_nm",
            ]
        ].to_numpy(dtype=float)
    ).all()


def test_selection_is_validation_only_and_matches_documented_tie_order(
    delay_harness: _DelayTestHarness,
) -> None:
    result = delay_harness.result(16.0)
    eligible = result.search_curve.loc[
        result.search_curve["optimizer_success"].astype(bool)
        & np.isfinite(
            result.search_curve["validation_torque_rmse_combined_nm"]
        )
    ].copy()
    eligible["absolute_candidate_delay_ms"] = eligible[
        "candidate_delay_ms"
    ].abs()
    expected = eligible.sort_values(
        [
            "validation_torque_rmse_combined_nm",
            "absolute_candidate_delay_ms",
            "candidate_delay_ms",
        ],
        kind="mergesort",
    ).iloc[0]

    assert result.estimated_delay_ms == expected["candidate_delay_ms"]
    assert result.delay_selection_splits == ("train", "validation")
    assert result.test_used_for_delay_selection is False


def test_validation_changes_selection_but_not_per_candidate_train_fits(
    delay_harness: _DelayTestHarness,
) -> None:
    train_16 = delay_harness.splits(16.0)["train"]
    validation_16 = delay_harness.splits(16.0)["validation"]
    validation_24 = delay_harness.splits(24.0)["validation"]
    candidates = np.arange(0.010, 0.031, 0.002)

    matched = estimate_wrench_delay(
        train_16,
        validation_16,
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=candidates,
    )
    changed_validation = estimate_wrench_delay(
        train_16,
        validation_24,
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=candidates,
    )

    assert matched.estimated_delay_ms == 16.0
    assert changed_validation.estimated_delay_ms == 24.0
    train_parameter_columns = [
        f"train_estimate_{name}" for name in PARAMETER_NAMES
    ]
    assert np.allclose(
        matched.search_curve[train_parameter_columns],
        changed_validation.search_curve[train_parameter_columns],
        atol=0.0,
        rtol=0.0,
    )
    assert not np.allclose(
        matched.search_curve["validation_torque_rmse_combined_nm"],
        changed_validation.search_curve[
            "validation_torque_rmse_combined_nm"
        ],
        atol=0.0,
        rtol=0.0,
    )


def test_estimator_interface_accepts_neither_test_data_nor_true_delay() -> None:
    parameters = signature(estimate_wrench_delay).parameters

    assert not any("test" in name.lower() for name in parameters)
    assert not any("true" in name.lower() for name in parameters)
    assert "training_dataframe" in parameters
    assert "validation_dataframe" in parameters


def test_modifying_test_split_cannot_change_selection_and_run_is_reproducible(
    delay_harness: _DelayTestHarness,
) -> None:
    splits = delay_harness.splits(16.0)
    candidates = (0.015, 0.016, 0.017)
    first = estimate_wrench_delay(
        splits["train"],
        splits["validation"],
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=candidates,
    )

    splits["test"].loc[:, "fx_observed_n"] = 1e12
    splits["test"].loc[:, "fz_observed_n"] = -1e12
    second = estimate_wrench_delay(
        splits["train"],
        splits["validation"],
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=candidates,
    )

    _assert_same_result(first, second)


def test_truth_and_scenario_leakage_fields_do_not_affect_delay_search(
    delay_harness: _DelayTestHarness,
) -> None:
    splits = delay_harness.splits(16.0)
    poisoned_train = splits["train"].copy(deep=True)
    poisoned_validation = splits["validation"].copy(deep=True)
    for dataframe in (poisoned_train, poisoned_validation):
        dataframe["wrench_age_s"] = 9.999
        dataframe["wrench_delay_s"] = -123.0
        dataframe["true_delay_s"] = 456.0
        dataframe["noise_scenario"] = "timing_delay_9999ms"
        dataframe["subject_id"] = "decoy_subject"
        dataframe["tau_total_hip_nm"] = 1e12
        dataframe["invalid_reason"] = "contains_the_answer"
        dataframe.attrs["noise_metadata"] = {"true_delay_s": 789.0}

    clean_sanitized = sanitize_delay_estimation_input(
        splits["train"],
        "train",
    )
    poisoned_sanitized = sanitize_delay_estimation_input(
        poisoned_train,
        "train",
    )
    pd.testing.assert_frame_equal(
        clean_sanitized,
        poisoned_sanitized,
        check_exact=True,
    )
    assert tuple(clean_sanitized.columns[:-1]) == DELAY_ESTIMATOR_INPUT_COLUMNS
    assert clean_sanitized.attrs == {}

    candidates = (0.015, 0.016, 0.017)
    reference = estimate_wrench_delay(
        splits["train"],
        splits["validation"],
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=candidates,
    )
    poisoned = estimate_wrench_delay(
        poisoned_train,
        poisoned_validation,
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=candidates,
    )
    _assert_same_result(reference, poisoned)


def test_tie_break_is_stable_for_reversed_symmetric_candidates(
    delay_harness: _DelayTestHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {
        "mass_scale": 1.0,
        "k_hip_nm_per_rad": 15.0,
        "k_knee_nm_per_rad": 12.0,
        "b_hip_nm_s_per_rad": 2.0,
        "b_knee_nm_s_per_rad": 1.5,
    }

    def fake_estimate(*args, **kwargs):
        return SimpleNamespace(
            estimated_parameters=parameters,
            optimizer_success=True,
            optimizer_message="synthetic tie",
            cost=1.0,
        )

    def tied_validation_metrics(*args, **kwargs):
        return {
            "torque_rmse_hip_nm": 1.0,
            "torque_rmse_knee_nm": 1.0,
            "torque_rmse_combined_nm": 1.0,
        }

    monkeypatch.setattr(
        delay_estimator_module,
        "estimate_subject_parameters",
        fake_estimate,
    )
    monkeypatch.setattr(
        delay_estimator_module,
        "compute_torque_metrics",
        tied_validation_metrics,
    )
    splits = delay_harness.splits(0.0)

    forward = estimate_wrench_delay(
        splits["train"],
        splits["validation"],
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=(-0.001, 0.001),
    )
    reversed_order = estimate_wrench_delay(
        splits["train"],
        splits["validation"],
        delay_harness.template,
        L1,
        L2,
        candidate_delays_s=(0.001, -0.001),
    )

    assert forward.estimated_delay_ms == -1.0
    _assert_same_result(forward, reversed_order)


def test_delay_beyond_grid_selects_nearest_search_boundary(
    delay_harness: _DelayTestHarness,
) -> None:
    result = delay_harness.result(55.0)
    selected = result.search_curve.loc[result.search_curve["selected"]].iloc[0]

    assert result.estimated_delay_ms == 50.0
    assert selected["candidate_delay_ms"] == 50.0
    assert result.search_curve["candidate_delay_ms"].max() == 50.0


def test_boundary_selection_exposes_a_machine_readable_warning(
    delay_harness: _DelayTestHarness,
) -> None:
    result = delay_harness.result(55.0)

    assert result.search_boundary_hit is True
    assert result.search_warning == (
        "selected_search_boundary_possible_out_of_search_range"
    )
