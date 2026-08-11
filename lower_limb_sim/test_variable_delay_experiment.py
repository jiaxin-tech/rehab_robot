"""阶段 4.5B 变化延迟实验入口的端到端回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    identification_lower_bounds,
    identification_upper_bounds,
    max_wrench_age_s,
    variable_delay_random_seed,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.identification_dataset import build_identification_dataset
from lower_limb_sim.parameter_estimator import PARAMETER_NAMES
from lower_limb_sim.run_variable_delay_experiment import (
    ALIGNMENT_METHODS,
    ESTIMATOR_INPUT_COLUMNS,
    main,
    run_single_variable_delay_experiment,
)
from lower_limb_sim.visualize_variable_delay import (
    generate_variable_delay_visualizations,
)


SCENARIOS = (
    "fixed_16ms",
    "piecewise_delay",
    "gradual_drift",
    "combined_realistic",
    "dropout_5pct",
    "stale_freeze",
    "long_tail",
)
SMALL_GLOBAL_DELAY_GRID_S = (0.0, 0.008, 0.016, 0.024, 0.032)
METHOD_SET = set(ALIGNMENT_METHODS)


@pytest.fixture(scope="module")
def variable_delay_experiment_suite(tmp_path_factory):
    """复用 50 Hz clean 数据和四个场景，避免重复构建/优化。"""

    output_root = tmp_path_factory.mktemp("variable_delay_experiment")
    clean = build_identification_dataset(
        get_dynamic_subject("baseline"),
        "clean",
        sampling_frequency_hz=50.0,
    )
    results = {
        scenario: run_single_variable_delay_experiment(
            "baseline",
            scenario,
            clean_dataset=clean,
            output_root=output_root,
            sampling_frequency_hz=50.0,
            random_seed=variable_delay_random_seed,
            make_plots=False,
            loss="linear",
            global_candidate_delays_s=SMALL_GLOBAL_DELAY_GRID_S,
        )
        for scenario in SCENARIOS
    }
    return {
        "clean": clean,
        "output_root": output_root,
        "results": results,
    }


def _comparison(result: dict[str, object]) -> pd.DataFrame:
    comparison = result["method_comparison"]
    assert isinstance(comparison, pd.DataFrame)
    return comparison.set_index("alignment_method", drop=False)


def _buffered_rows(result: dict[str, object]) -> pd.DataFrame:
    matched = result["matched_dataset"]
    assert isinstance(matched, pd.DataFrame)
    return matched.loc[
        matched["alignment_method"].eq("causal_buffered_matching")
    ].reset_index(drop=True)


def test_all_four_alignment_methods_are_present_once(
    variable_delay_experiment_suite,
) -> None:
    for result in variable_delay_experiment_suite["results"].values():
        comparison = result["method_comparison"]
        parameters = result["parameter_estimates"]
        matched = result["matched_dataset"]

        assert set(comparison["alignment_method"]) == METHOD_SET
        assert comparison["alignment_method"].value_counts().eq(1).all()
        assert set(parameters["alignment_method"]) == METHOD_SET
        assert set(matched["alignment_method"]) == METHOD_SET
        assert len(parameters) == len(METHOD_SET) * len(PARAMETER_NAMES)


def test_buffered_fixed16_rmse_beats_row_and_latest_and_is_near_global(
    variable_delay_experiment_suite,
) -> None:
    result = variable_delay_experiment_suite["results"]["fixed_16ms"]
    comparison = _comparison(result)
    buffered = float(
        comparison.loc["causal_buffered_matching", "test_rmse_nm"]
    )
    row = float(comparison.loc["row_index_alignment", "test_rmse_nm"])
    latest = float(
        comparison.loc["causal_history_latest", "test_rmse_nm"]
    )
    global_fixed = float(
        comparison.loc["global_fixed_delay", "test_rmse_nm"]
    )

    assert buffered < 0.10 * row
    assert buffered < 0.10 * latest
    assert buffered <= max(1.5 * global_fixed, 0.005)
    assert (
        abs(result["global_delay_selection"].estimated_delay_ms - 16.0)
        <= 2.0
    )


def test_buffered_piecewise_rmse_beats_global_fixed_delay(
    variable_delay_experiment_suite,
) -> None:
    comparison = _comparison(
        variable_delay_experiment_suite["results"]["piecewise_delay"]
    )
    buffered = float(
        comparison.loc["causal_buffered_matching", "test_rmse_nm"]
    )
    global_fixed = float(
        comparison.loc["global_fixed_delay", "test_rmse_nm"]
    )

    assert buffered < 0.10 * global_fixed


def test_buffered_gradual_drift_delay_mae_beats_global_fixed(
    variable_delay_experiment_suite,
) -> None:
    comparison = _comparison(
        variable_delay_experiment_suite["results"]["gradual_drift"]
    )

    assert (
        comparison.loc["causal_buffered_matching", "delay_mae_ms"]
        < comparison.loc["global_fixed_delay", "delay_mae_ms"]
    )
    assert (
        comparison.loc["causal_buffered_matching", "test_rmse_nm"]
        < comparison.loc["global_fixed_delay", "test_rmse_nm"]
    )


def test_combined_buffered_rmse_beats_row_index_and_causal_latest(
    variable_delay_experiment_suite,
) -> None:
    comparison = _comparison(
        variable_delay_experiment_suite["results"]["combined_realistic"]
    )
    buffered = comparison.loc[
        "causal_buffered_matching",
        "test_rmse_nm",
    ]

    assert buffered < comparison.loc[
        "row_index_alignment",
        "test_rmse_nm",
    ]
    assert buffered < comparison.loc[
        "causal_history_latest",
        "test_rmse_nm",
    ]


def test_all_estimated_parameters_stay_inside_physical_bounds(
    variable_delay_experiment_suite,
) -> None:
    for result in variable_delay_experiment_suite["results"].values():
        parameter_table = result["parameter_estimates"]
        for row in parameter_table.itertuples(index=False):
            lower = identification_lower_bounds[row.parameter]
            upper = identification_upper_bounds[row.parameter]
            assert lower - 1e-12 <= row.estimated_value <= upper + 1e-12


def test_six_required_csvs_metadata_and_supplementary_curves_are_saved(
    variable_delay_experiment_suite,
) -> None:
    required_csvs = {
        "raw_variable_delay_dataset.csv",
        "matched_dataset.csv",
        "delay_tracking_history.csv",
        "delay_search_windows.csv",
        "parameter_estimates.csv",
        "parameter_estimates_by_method.csv",
        "method_comparison.csv",
        "torque_predictions_by_method.csv",
    }
    supplementary_csvs = {
        "global_fixed_delay_search_curve.csv",
    }
    for result in variable_delay_experiment_suite["results"].values():
        output_dir = Path(result["output_dir"])
        files = {
            path.name for path in output_dir.iterdir() if path.is_file()
        }

        assert required_csvs.issubset(files)
        assert supplementary_csvs.issubset(files)
        assert "metadata.json" in files
        assert "metrics.json" in files
        assert result["figure_paths"] == []
        for filename in required_csvs | supplementary_csvs:
            dataframe = pd.read_csv(
                output_dir / filename,
                low_memory=False,
            )
            assert not dataframe.empty
        metrics = json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        assert len(metrics["methods"]) == len(ALIGNMENT_METHODS)
        assert metrics["test_used_for_delay_estimation"] is False
        assert metrics["true_delay_used_for_estimation"] is False


def test_method_comparison_contains_all_required_metric_contracts(
    variable_delay_experiment_suite,
) -> None:
    required = {
        "delay_mae_ms",
        "delay_rmse_ms",
        "delay_p95_error_ms",
        "maximum_delay_error_ms",
        "delay_update_success_rate",
        "boundary_hit_count",
        "low_confidence_update_count",
        "valid_match_rate",
        "median_state_match_error_ms",
        "p95_state_match_error_ms",
        "maximum_state_match_error_ms",
        "dropout_rejected_count",
        "stale_rejected_count",
        "long_tail_rejected_count",
        "state_history_expired_count",
        "mass_scale_error_percent",
        "k_hip_error_percent",
        "k_knee_error_percent",
        "b_hip_error_percent",
        "b_knee_error_percent",
        "train_torque_rmse_nm",
        "validation_torque_rmse_nm",
        "test_torque_rmse_nm",
        "hip_test_rmse_nm",
        "knee_test_rmse_nm",
    }
    for result in variable_delay_experiment_suite["results"].values():
        comparison = result["method_comparison"]
        assert required.issubset(comparison.columns)


def test_actual_visualizer_generates_all_seven_required_figures(
    variable_delay_experiment_suite,
    tmp_path,
) -> None:
    result = variable_delay_experiment_suite["results"]["fixed_16ms"]
    paths = generate_variable_delay_visualizations(
        result["raw_dataset"],
        result["matched_dataset"],
        result["delay_tracking_history"],
        result["method_comparison"],
        result["parameter_estimates"],
        "baseline",
        "fixed_16ms",
        tmp_path,
    )
    expected = {
        "true_vs_estimated_delay.png",
        "delay_error_vs_time.png",
        "delay_confidence_vs_time.png",
        "state_match_error_vs_time.png",
        "torque_prediction_comparison.png",
        "parameter_error_by_method.png",
        "valid_rejected_samples.png",
    }

    assert len(paths) == 7
    assert {Path(path).name for path in paths} == expected
    assert all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths)


def test_every_valid_matched_sample_has_finite_state_force_and_error(
    variable_delay_experiment_suite,
) -> None:
    finite_columns = [
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "fx_observed_n",
        "fz_observed_n",
        "state_timestamp_s",
        "wrench_arrival_timestamp_s",
        "wrench_sample_timestamp_s",
        "state_match_error_s",
    ]
    for result in variable_delay_experiment_suite["results"].values():
        matched = result["matched_dataset"]
        for method, method_rows in matched.groupby(
            "alignment_method",
            sort=False,
        ):
            valid = method_rows["alignment_valid"].astype(bool)
            assert valid.any(), method
            assert method_rows.loc[valid, "sample_valid"].astype(bool).all()
            assert np.isfinite(
                method_rows.loc[valid, finite_columns].to_numpy(dtype=float)
            ).all()


@pytest.mark.parametrize(
    ("scenario", "fault_column"),
    (
        ("dropout_5pct", "is_dropout"),
        ("stale_freeze", "is_stale"),
    ),
)
def test_buffer_never_revalidates_dropout_or_stale_events(
    variable_delay_experiment_suite,
    scenario: str,
    fault_column: str,
) -> None:
    buffered = _buffered_rows(
        variable_delay_experiment_suite["results"][scenario]
    )
    fault = buffered[fault_column].astype(bool)

    assert fault.any()
    assert (~buffered.loc[fault, "alignment_valid"].astype(bool)).all()
    assert (~buffered.loc[fault, "sample_valid"].astype(bool)).all()
    if fault_column == "is_dropout":
        assert buffered.loc[
            fault,
            ["fx_observed_n", "fz_observed_n"],
        ].isna().all(axis=None)
    else:
        # Freeze保留“上一力值”用于审计，但 invalid gate 必须始终关闭。
        assert np.isfinite(
            buffered.loc[
                fault,
                ["fx_observed_n", "fz_observed_n"],
            ].to_numpy(dtype=float)
        ).all()
        assert buffered.loc[fault, "invalid_reason"].eq(
            "stale_or_frozen_wrench"
        ).all()


def test_buffer_rejects_long_tail_older_than_wrench_age_limit(
    variable_delay_experiment_suite,
) -> None:
    result = variable_delay_experiment_suite["results"]["long_tail"]
    buffered = _buffered_rows(result)
    too_old = (
        buffered["is_long_tail"].astype(bool)
        & (buffered["true_delay_s"] > max_wrench_age_s)
    )

    assert too_old.any()
    assert (~buffered.loc[too_old, "alignment_valid"].astype(bool)).all()
    assert buffered.loc[too_old, "invalid_reason"].str.contains(
        "wrench_age_limit_exceeded"
    ).all()
    comparison = _comparison(result)
    assert (
        comparison.loc[
            "causal_buffered_matching",
            "long_tail_rejected_count",
        ]
        == int(too_old.sum())
    )


def test_tracker_uses_only_train_and_never_receives_truth_or_holdout(
    variable_delay_experiment_suite,
) -> None:
    forbidden_estimator_fields = {
        "true_delay_s",
        "generated_base_delay_s",
        "wrench_age_s",
        "state_wrench_skew_s",
        "delay_scenario",
        "noise_scenario",
        "subject_id",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
    }
    assert set(ESTIMATOR_INPUT_COLUMNS).isdisjoint(forbidden_estimator_fields)

    for result in variable_delay_experiment_suite["results"].values():
        metadata = result["metadata"]
        raw = result["raw_dataset"]
        tracking = result["delay_tracking_history"]
        train_keys = set(
            raw.loc[
                raw["dataset_split"].eq("train"),
                ["trajectory_family", "speed_profile"],
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        tracked_keys = set(
            tracking[["trajectory_family", "speed_profile"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        assert tracked_keys
        assert tracked_keys.issubset(train_keys)
        assert metadata["tracker_input_splits"] == ["train"]
        assert metadata["tracker_test_access"] is False
        assert metadata["tracker_validation_access"] is False
        assert metadata["true_delay_passed_to_tracker"] is False
        assert metadata["runtime_tracker_history_truth_columns"] == []
        assert metadata["global_delay_selection_splits"] == [
            "train",
            "validation",
        ]
        assert metadata["global_delay_test_access"] is False
        assert metadata["global_delay_true_delay_access"] is False
        assert set(metadata["estimator_input_columns"]).isdisjoint(
            forbidden_estimator_fields
        )


def test_buffered_confidence_keeps_tracker_and_state_match_meanings_separate(
    variable_delay_experiment_suite,
) -> None:
    result = variable_delay_experiment_suite["results"]["fixed_16ms"]
    buffered = _buffered_rows(result)
    required = {
        "tracker_delay_confidence",
        "state_match_confidence",
        "delay_confidence",
    }

    assert required.issubset(buffered.columns)
    assert np.allclose(
        buffered["delay_confidence"],
        buffered["tracker_delay_confidence"],
    )
    assert (
        buffered["state_match_confidence"]
        >= buffered["tracker_delay_confidence"]
    ).any()
    metadata = result["metadata"]
    assert (
        metadata["primary_buffered_timestamp_source"]
        == "reliable simulated wrench_sample_timestamp_s"
    )
    assert (
        metadata["tracker_delay_fallback_exercised_in_primary_experiments"]
        is False
    )


def test_metadata_preserves_model_scope_timestamps_and_angle_definition(
    variable_delay_experiment_suite,
) -> None:
    required = {
        "model_version",
        "software_version_or_git_commit",
        "created_at",
        "subject_id",
        "delay_scenario",
        "random_seed",
        "sampling_frequency_hz",
        "model_angle_definition",
        "positive_delay_definition",
        "timestamp_columns",
        "state_history_duration_s",
        "maximum_state_interpolation_interval_ms",
        "maximum_state_match_error_ms",
        "maximum_wrench_age_ms",
        "tracker_input_splits",
        "tracker_test_access",
        "tracker_validation_access",
        "true_delay_passed_to_tracker",
        "four_method_definitions",
        "offline_only_warning",
        "long_gap_interpolation_allowed",
        "stale_or_dropout_revalidated",
        "true_delay_columns_for_evaluation_only",
        "scope_excludes",
        "disclaimer",
    }
    for result in variable_delay_experiment_suite["results"].values():
        output_dir = Path(result["output_dir"])
        saved = json.loads(
            (output_dir / "metadata.json").read_text(encoding="utf-8")
        )

        assert required.issubset(saved)
        assert saved["model_angle_definition"] == (
            "theta_shank = q_hip - q_knee"
        )
        assert saved["positive_delay_definition"] == (
            "wrench_arrival_timestamp_s - wrench_sample_timestamp_s"
        )
        assert set(saved["four_method_definitions"]) == METHOD_SET
        assert saved["long_gap_interpolation_allowed"] is False
        assert saved["stale_or_dropout_revalidated"] is False
        assert saved["true_delay_columns_for_evaluation_only"] == [
            "true_delay_s",
            "generated_base_delay_s",
        ]
        assert "real_robot_control" in saved["scope_excludes"]
        assert saved == result["metadata"]


def test_runner_rejects_unknown_subject_and_scenario(
    variable_delay_experiment_suite,
) -> None:
    clean = variable_delay_experiment_suite["clean"]
    with pytest.raises(ValueError, match="unknown scenario"):
        run_single_variable_delay_experiment(
            "baseline",
            "not_a_scenario",
            clean_dataset=clean,
            make_plots=False,
        )
    with pytest.raises(ValueError, match="Unknown dynamic subject"):
        run_single_variable_delay_experiment(
            "not_a_subject",
            "fixed_16ms",
            clean_dataset=clean,
            make_plots=False,
        )


@pytest.mark.parametrize(
    "arguments",
    (
        ["run_variable_delay_experiment"],
        [
            "run_variable_delay_experiment",
            "--all",
            "--all-baseline",
        ],
    ),
)
def test_cli_rejects_missing_or_conflicting_mode_arguments(
    monkeypatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
