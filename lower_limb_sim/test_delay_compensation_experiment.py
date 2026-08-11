"""阶段 4.5A 固定 wrench 延迟估计与补偿的端到端回归测试。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    identification_lower_bounds,
    identification_upper_bounds,
    max_alignment_interpolation_gap_s,
)
from lower_limb_sim.delay_estimator import estimate_wrench_delay
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.identification_dataset import build_identification_dataset
from lower_limb_sim.parameter_estimator import PARAMETER_NAMES
from lower_limb_sim.run_delay_compensation_experiment import (
    COMPENSATION_METHODS,
    run_single_delay_compensation_experiment,
)
from lower_limb_sim.timestamp_alignment import (
    align_wrench_to_state_timestamps,
    synthesize_delayed_wrench_dataset,
)


EXPERIMENT_DELAYS_MS = (0.0, 16.0, 32.0)
METHOD_SET = set(COMPENSATION_METHODS)


@pytest.fixture(scope="module")
def delay_experiment_suite(tmp_path_factory):
    """复用一份 100 Hz clean 数据，避免三个实验重复生成轨迹。"""

    output_root = tmp_path_factory.mktemp("delay_compensation_experiment")
    clean = build_identification_dataset(
        get_dynamic_subject("baseline"),
        "clean",
        sampling_frequency_hz=100.0,
    )
    results = {
        delay_ms: run_single_delay_compensation_experiment(
            "baseline",
            delay_ms,
            clean_dataset=clean,
            output_root=output_root,
            sampling_frequency_hz=100.0,
            make_plots=delay_ms == 0.0,
            loss="linear",
        )
        for delay_ms in EXPERIMENT_DELAYS_MS
    }
    return {
        "clean": clean,
        "output_root": output_root,
        "results": results,
    }


def _method_rows(result: dict[str, object]) -> pd.DataFrame:
    summary = result["summary"]
    assert isinstance(summary, pd.DataFrame)
    return summary.set_index("compensation_method", drop=False)


def _parameter_details(result: dict[str, object]) -> pd.DataFrame:
    output_dir = Path(result["output_dir"])
    return pd.read_csv(output_dir / "before_after_parameter_errors.csv")


def test_zero_delay_is_a_no_op_and_does_not_degrade_results(
    delay_experiment_suite,
) -> None:
    result = delay_experiment_suite["results"][0.0]
    rows = _method_rows(result)
    raw = rows.loc["uncompensated"]

    assert abs(result["selection"].estimated_delay_ms) <= 2.0
    for method in ("known_truth", "automatic_estimated"):
        observed = rows.loc[method]
        assert abs(float(observed["applied_delay_ms"])) <= 2.0
        for field in (
            "parameter_mean_relative_error_percent",
            "parameter_max_relative_error_percent",
            "validation_torque_rmse_combined_nm",
            "test_torque_rmse_combined_nm",
        ):
            assert float(observed[field]) <= float(raw[field]) + 1e-8


@pytest.mark.parametrize("delay_ms", (16.0, 32.0))
def test_known_and_automatic_compensation_improve_delayed_results(
    delay_experiment_suite,
    delay_ms: float,
) -> None:
    result = delay_experiment_suite["results"][delay_ms]
    rows = _method_rows(result)
    raw = rows.loc["uncompensated"]

    for method in ("known_truth", "automatic_estimated"):
        compensated = rows.loc[method]
        for split in ("train", "validation", "test"):
            field = f"{split}_torque_rmse_combined_nm"
            assert float(compensated[field]) < float(raw[field])
        assert (
            float(compensated["parameter_mean_relative_error_percent"])
            < float(raw["parameter_mean_relative_error_percent"])
        )
        assert (
            float(compensated["parameter_max_relative_error_percent"])
            < float(raw["parameter_max_relative_error_percent"])
        )
        for parameter in (
            "b_hip_nm_s_per_rad",
            "b_knee_nm_s_per_rad",
        ):
            field = f"{parameter}_relative_error_percent"
            assert float(compensated[field]) < float(raw[field])


@pytest.mark.parametrize("delay_ms", (16.0, 32.0))
def test_automatic_delay_and_parameters_are_close_to_known_compensation(
    delay_experiment_suite,
    delay_ms: float,
) -> None:
    result = delay_experiment_suite["results"][delay_ms]
    rows = _method_rows(result)
    raw = rows.loc["uncompensated"]
    known = rows.loc["known_truth"]
    automatic = rows.loc["automatic_estimated"]

    assert abs(result["selection"].estimated_delay_ms - delay_ms) <= 2.0
    for field in (
        "validation_torque_rmse_combined_nm",
        "test_torque_rmse_combined_nm",
    ):
        permitted_difference = max(1e-3, 0.10 * float(raw[field]))
        assert abs(float(automatic[field]) - float(known[field])) <= (
            permitted_difference
        )

    details = _parameter_details(result).set_index(
        ["compensation_method", "parameter"]
    )
    for parameter in PARAMETER_NAMES:
        known_value = float(
            details.loc[("known_truth", parameter), "estimated_value"]
        )
        automatic_value = float(
            details.loc[("automatic_estimated", parameter), "estimated_value"]
        )
        assert np.isclose(
            automatic_value,
            known_value,
            rtol=0.05,
            atol=0.05,
        )


def test_all_three_methods_are_saved_and_parameters_stay_in_bounds(
    delay_experiment_suite,
) -> None:
    for result in delay_experiment_suite["results"].values():
        output_dir = Path(result["output_dir"])
        in_memory = result["summary"]
        saved = pd.read_csv(output_dir / "delay_compensation_summary.csv")
        details = _parameter_details(result)

        assert set(in_memory["compensation_method"]) == METHOD_SET
        assert set(in_memory["comparison_label"]) == {
            "uncorrected",
            "known_delay_compensation",
            "automatic_delay_compensation",
        }
        assert set(saved["compensation_method"]) == METHOD_SET
        assert saved["compensation_method"].value_counts().eq(1).all()
        assert set(details["compensation_method"]) == METHOD_SET
        assert set(details["parameter"]) == set(PARAMETER_NAMES)
        assert len(details) == len(METHOD_SET) * len(PARAMETER_NAMES)

        for row in details.itertuples(index=False):
            lower = identification_lower_bounds[row.parameter]
            upper = identification_upper_bounds[row.parameter]
            assert lower - 1e-12 <= row.estimated_value <= upper + 1e-12


def test_test_split_is_used_only_for_final_evaluation(
    delay_experiment_suite,
) -> None:
    signature = inspect.signature(estimate_wrench_delay)
    forbidden_arguments = {
        "test_dataframe",
        "test_df",
        "true_delay",
        "true_delay_ms",
        "noise_scenario",
    }
    assert forbidden_arguments.isdisjoint(signature.parameters)

    for result in delay_experiment_suite["results"].values():
        output_dir = Path(result["output_dir"])
        metadata = json.loads(
            (output_dir / "metadata.json").read_text(encoding="utf-8")
        )
        curve = pd.read_csv(output_dir / "delay_search_curve.csv")
        selection = result["selection"]

        assert tuple(selection.delay_selection_splits) == (
            "train",
            "validation",
        )
        assert selection.test_used_for_delay_selection is False
        assert metadata["delay_selection_splits"] == ["train", "validation"]
        assert metadata["test_used_for_delay_selection"] is False
        assert metadata["true_delay_passed_to_automatic_estimator"] is False
        assert not any(column.startswith("test_") for column in curve.columns)
        assert not any("true_delay" in column for column in curve.columns)
        assert any(
            column.startswith("test_")
            for column in result["summary"].columns
        )


def test_compensated_valid_samples_and_saved_metrics_are_finite(
    delay_experiment_suite,
) -> None:
    required_valid_columns = [
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
        "state_timestamp_s",
        "wrench_timestamp_s",
        "alignment_timestamp_s",
        "wrench_effective_timestamp_s",
        "alignment_gap_s",
    ]
    for result in delay_experiment_suite["results"].values():
        output_dir = Path(result["output_dir"])
        dataset = pd.read_csv(output_dir / "compensated_dataset.csv")
        valid = dataset["sample_valid"].astype(bool)

        assert valid.any()
        assert dataset.loc[valid, "alignment_valid"].astype(bool).all()
        assert set(dataset.loc[valid, "dataset_split"]) == {
            "train",
            "validation",
            "test",
        }
        assert np.isfinite(
            dataset.loc[valid, required_valid_columns].to_numpy(dtype=float)
        ).all()

        summary = pd.read_csv(
            output_dir / "delay_compensation_summary.csv"
        )
        # 空的可选 warning 文本经 CSV round-trip 会被 pandas 推断为 NaN
        # 浮点列；它不是数值结果或有效样本字段。
        numeric = summary.drop(
            columns=["search_warning"],
            errors="ignore",
        ).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_alignment_does_not_fill_a_gap_longer_than_20ms(
    delay_experiment_suite,
) -> None:
    clean = delay_experiment_suite["clean"]
    delayed = synthesize_delayed_wrench_dataset(clean, 0.016)
    damaged = delayed.copy(deep=True)
    group = damaged.loc[
        damaged["trajectory_family"].eq("coupled")
        & damaged["speed_profile"].eq("fast")
    ]
    missing_indices = group.index.to_numpy()[100:103]
    damaged.loc[missing_indices, "sample_valid"] = False
    damaged.loc[
        missing_indices,
        ["fx_observed_n", "fz_observed_n"],
    ] = np.nan

    alignment = align_wrench_to_state_timestamps(
        damaged,
        0.016,
        mode="offline_only",
        max_interpolation_gap_s=max_alignment_interpolation_gap_s,
    )
    aligned = alignment.dataframe
    long_gap = aligned["alignment_invalid_reason"].str.contains(
        "alignment_gap_exceeded",
        na=False,
    )

    assert np.isclose(max_alignment_interpolation_gap_s, 0.020)
    assert long_gap.any()
    assert (
        aligned.loc[long_gap, "alignment_gap_s"]
        > max_alignment_interpolation_gap_s
    ).all()
    assert (~aligned.loc[long_gap, "sample_valid"].astype(bool)).all()
    assert aligned.loc[
        long_gap,
        ["fx_observed_n", "fz_observed_n"],
    ].isna().all(axis=None)
    assert alignment.metadata["extrapolation_used"] is False
    assert alignment.metadata["cross_trajectory_interpolation_used"] is False


def test_complete_artifacts_and_required_fields_are_present(
    delay_experiment_suite,
) -> None:
    required_files = {
        "delay_search_curve.csv",
        "delay_compensation_summary.csv",
        "compensated_dataset.csv",
        "before_after_parameter_errors.csv",
        "metadata.json",
    }
    required_curve_fields = {
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
        "selected",
    }
    required_summary_fields = {
        "subject_id",
        "true_delay_ms",
        "estimated_delay_ms",
        "delay_error_ms",
        "compensation_method",
        "applied_delay_ms",
        "parameter_mean_relative_error_percent",
        "parameter_max_relative_error_percent",
        "train_torque_rmse_combined_nm",
        "validation_torque_rmse_combined_nm",
        "test_torque_rmse_combined_nm",
        "search_boundary_hit",
        "alignment_mode",
        "uncorrected_parameter_error_percent",
        "known_compensated_parameter_error_percent",
        "automatic_compensated_parameter_error_percent",
        "uncorrected_train_rmse_nm",
        "uncorrected_validation_rmse_nm",
        "uncorrected_test_rmse_nm",
        "known_train_rmse_nm",
        "known_validation_rmse_nm",
        "known_test_rmse_nm",
        "automatic_train_rmse_nm",
        "automatic_validation_rmse_nm",
        "automatic_test_rmse_nm",
        "b_hip_error_before_percent",
        "b_hip_error_after_percent",
        "b_knee_error_before_percent",
        "b_knee_error_after_percent",
        "valid_samples_before",
        "valid_samples_after",
        "invalid_gap_samples",
        "invalid_dropout_samples",
        "invalid_stale_samples",
    } | {
        f"{parameter}_relative_error_percent"
        for parameter in PARAMETER_NAMES
    }
    required_dataset_fields = {
        "dataset_split",
        "state_timestamp_s",
        "wrench_timestamp_s",
        "wrench_age_s",
        "state_wrench_skew_s",
        "fx_raw_observed_n",
        "fz_raw_observed_n",
        "fx_observed_n",
        "fz_observed_n",
        "sample_valid",
        "alignment_valid",
        "alignment_mode",
        "alignment_offline_only",
        "alignment_used_future",
        "alignment_future_lookahead_s",
        "alignment_gap_s",
        "alignment_timestamp_s",
        "wrench_effective_timestamp_s",
        "applied_delay_compensation_s",
        "alignment_invalid_reason",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
    }
    required_parameter_fields = {
        "subject_id",
        "noise_scenario",
        "parameter",
        "true_value",
        "estimated_value",
        "absolute_error",
        "relative_error_percent",
        "compensation_method",
    }
    required_metadata_fields = {
        "model_version",
        "software_version_or_git_commit",
        "created_at",
        "subject_id",
        "noise_scenario",
        "true_delay_ms",
        "estimated_delay_ms",
        "delay_error_ms",
        "positive_delay_definition",
        "automatic_estimator_input_columns",
        "delay_selection_splits",
        "test_used_for_delay_selection",
        "true_delay_passed_to_automatic_estimator",
        "search_range_ms",
        "search_step_ms",
        "delay_search_min_ms",
        "delay_search_max_ms",
        "delay_search_step_ms",
        "alignment_mode",
        "offline_only",
        "offline_only_warning",
        "max_interpolation_gap_s",
        "maximum_interpolation_gap_ms",
        "train_trajectory_ids",
        "validation_trajectory_ids",
        "test_trajectory_ids",
        "random_seed",
        "long_freeze_or_dropout_interpolated",
        "common_training_samples",
        "common_validation_samples",
        "angle_definition",
        "model_angle_definition",
        "disclaimer",
    }

    for result in delay_experiment_suite["results"].values():
        output_dir = Path(result["output_dir"])
        assert required_files.issubset(
            path.name for path in output_dir.iterdir() if path.is_file()
        )
        if float(result["true_delay_ms"]) == 0.0:
            assert len(result["figure_paths"]) == 2
        else:
            assert result["figure_paths"] == []

        curve = pd.read_csv(output_dir / "delay_search_curve.csv")
        summary = pd.read_csv(
            output_dir / "delay_compensation_summary.csv"
        )
        dataset = pd.read_csv(output_dir / "compensated_dataset.csv")
        parameter_details = _parameter_details(result)
        metadata = json.loads(
            (output_dir / "metadata.json").read_text(encoding="utf-8")
        )

        assert required_curve_fields.issubset(curve.columns)
        assert curve["selected"].astype(bool).sum() == 1
        assert required_summary_fields.issubset(summary.columns)
        assert required_dataset_fields.issubset(dataset.columns)
        assert required_parameter_fields.issubset(parameter_details.columns)
        assert required_metadata_fields.issubset(metadata)
        if float(result["true_delay_ms"]) == 0.0:
            assert {
                "delay_search_curve.png",
                "before_after_parameter_error.png",
            }.issubset(path.name for path in output_dir.iterdir())
