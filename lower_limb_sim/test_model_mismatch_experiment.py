"""阶段 4.5C 统一实验管线、泛化评价和泄漏边界测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    L1,
    L2,
    identification_lower_bounds,
    identification_parameter_names,
    identification_upper_bounds,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.run_model_mismatch_experiment import (
    ALL_EVALUATION_SPLITS,
    ESTIMATOR_INPUT_COLUMNS,
    build_model_mismatch_dataset,
    project_estimator_inputs,
    run_model_mismatch_experiment,
)


SCENARIOS_NEEDED = (
    "matched_linear",
    "nonlinear_stiffness_mild",
    "nonlinear_stiffness_strong",
    "hip_knee_coupling_mild",
    "combined_mild",
)


@pytest.fixture(scope="module")
def experiments(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    root = tmp_path_factory.mktemp("model_mismatch_experiments")
    return {
        scenario: run_model_mismatch_experiment(
            "baseline",
            scenario,
            output_root=root,
            sampling_frequency_hz=10.0,
            make_plots=False,
        )
        for scenario in SCENARIOS_NEEDED
    }


def _split_metric(
    result: dict[str, object],
    split: str,
    model: str = "identified",
) -> pd.Series:
    metrics = result["split_metrics"]
    assert isinstance(metrics, pd.DataFrame)
    selected = metrics.loc[
        metrics["dataset_split"].eq(split)
        & metrics["prediction_model"].eq(model)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def _trajectory_metric(
    result: dict[str, object],
    family: str,
    model: str = "identified",
) -> pd.Series:
    metrics = result["trajectory_metrics"]
    assert isinstance(metrics, pd.DataFrame)
    selected = metrics.loc[
        metrics["trajectory_family"].eq(family)
        & metrics["prediction_model"].eq(model)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_matched_linear_recovers_stage4_clean_performance(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["matched_linear"]
    estimate = result["estimate"]
    baseline = get_dynamic_subject("baseline")
    expected = {
        "mass_scale": 1.0,
        "k_hip_nm_per_rad": baseline.k_hip_nm_per_rad,
        "k_knee_nm_per_rad": baseline.k_knee_nm_per_rad,
        "b_hip_nm_s_per_rad": baseline.b_hip_nm_s_per_rad,
        "b_knee_nm_s_per_rad": baseline.b_knee_nm_s_per_rad,
    }
    assert estimate.optimizer_success
    for parameter, value in expected.items():
        assert estimate.estimated_parameters[parameter] == pytest.approx(
            value,
            abs=2e-6,
        )
    interpolation = _split_metric(result, "interpolation_test")
    assert interpolation["combined_torque_rmse_nm"] < 1e-6
    assert interpolation["combined_nrmse_percent"] < 1e-5


def test_estimator_remains_exactly_five_bounded_parameters(
    experiments: dict[str, dict[str, object]],
) -> None:
    expected_names = tuple(identification_parameter_names)
    assert len(expected_names) == 5
    for result in experiments.values():
        estimate = result["estimate"]
        assert tuple(estimate.estimated_parameters) == expected_names
        for parameter, value in estimate.estimated_parameters.items():
            assert identification_lower_bounds[parameter] <= value
            assert value <= identification_upper_bounds[parameter]


def test_tau_complex_and_generator_terms_are_removed_before_estimator() -> None:
    dataset = build_model_mismatch_dataset(
        "baseline",
        "combined_mild",
        sampling_frequency_hz=5.0,
    )
    assert "tau_complex_true_hip_nm" in dataset
    assert any(column.startswith("tau_generator_") for column in dataset)
    projected = project_estimator_inputs(
        dataset.loc[dataset["dataset_split"].eq("train")]
    )
    assert tuple(projected.columns) == ESTIMATOR_INPUT_COLUMNS
    assert not projected.attrs
    assert not any(
        token in column
        for column in projected
        for token in (
            "true",
            "generator",
            "scenario",
            "subject_id",
            "dataset_split",
            "trajectory",
        )
    )


def test_leakage_only_fields_cannot_change_estimator_projection() -> None:
    dataset = build_model_mismatch_dataset(
        "baseline",
        "combined_mild",
        sampling_frequency_hz=5.0,
    )
    train = dataset.loc[dataset["dataset_split"].eq("train")].copy()
    expected = project_estimator_inputs(train)
    tampered = train.copy()
    tampered["scenario_name"] = "forbidden_test_marker"
    tampered["tau_complex_true_hip_nm"] += 1e6
    generator_columns = [
        column for column in tampered if column.startswith("tau_generator_")
    ]
    tampered.loc[:, generator_columns] = -12345.0
    observed = project_estimator_inputs(tampered)
    pd.testing.assert_frame_equal(expected, observed, check_exact=True)


def test_optimizer_call_receives_only_train_whitelist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lower_limb_sim.run_model_mismatch_experiment as runner

    original = runner.estimate_subject_parameters
    captured: dict[str, object] = {}

    def checked_estimator(training_dataframe, *args, **kwargs):
        captured["columns"] = tuple(training_dataframe.columns)
        captured["attrs"] = dict(training_dataframe.attrs)
        captured["rows"] = len(training_dataframe)
        return original(training_dataframe, *args, **kwargs)

    monkeypatch.setattr(runner, "estimate_subject_parameters", checked_estimator)
    result = runner.run_model_mismatch_experiment(
        "baseline",
        "combined_mild",
        output_root=tmp_path,
        sampling_frequency_hz=5.0,
        make_plots=False,
    )
    train = result["splits"]["train"]
    assert captured == {
        "columns": ESTIMATOR_INPUT_COLUMNS,
        "attrs": {},
        "rows": len(train),
    }


def test_interpolation_and_outside_domain_never_enter_fit_or_model_selection(
    experiments: dict[str, dict[str, object]],
) -> None:
    metadata = experiments["combined_mild"]["metadata"]
    train_ids = set(metadata["train_trajectory_ids"])
    validation_ids = set(metadata["validation_trajectory_ids"])
    test_ids = {
        trajectory
        for values in metadata["test_trajectory_ids"].values()
        for trajectory in values
    }
    assert train_ids.isdisjoint(validation_ids | test_ids)
    assert metadata["fit_split"] == ["train"]
    assert metadata["validation_used_for_parameter_fit"] is False
    assert metadata["test_used_for_parameter_fit_or_model_selection"] is False
    assert metadata["generator_parameters_available_to_estimator"] is False
    assert metadata["tau_complex_true_available_to_estimator"] is False


def test_identified_model_improves_over_generic_in_mild_mismatch(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["combined_mild"]
    identified = _split_metric(result, "interpolation_test", "identified")
    generic = _split_metric(result, "interpolation_test", "generic")
    assert identified["combined_torque_rmse_nm"] < generic[
        "combined_torque_rmse_nm"
    ]
    assert identified["combined_nrmse_percent"] < 15.0


def test_strong_mismatch_error_is_reported_not_filtered(
    experiments: dict[str, dict[str, object]],
) -> None:
    mild = _split_metric(
        experiments["nonlinear_stiffness_mild"],
        "interpolation_test",
    )
    strong_result = experiments["nonlinear_stiffness_strong"]
    strong = _split_metric(strong_result, "interpolation_test")
    assert strong["combined_torque_rmse_nm"] > mild["combined_torque_rmse_nm"]
    assert strong["combined_nrmse_percent"] > mild["combined_nrmse_percent"]
    trajectory_metrics = strong_result["trajectory_metrics"]
    identified = trajectory_metrics.loc[
        trajectory_metrics["prediction_model"].eq("identified")
    ]
    assert len(identified) == 12
    assert identified["metric_valid"].all()
    assert (identified["valid_torque_sample_count"] > 0).all()


def test_nonlinear_stiffness_has_larger_large_angle_residual(
    experiments: dict[str, dict[str, object]],
) -> None:
    strong = experiments["nonlinear_stiffness_strong"]["predictions"]
    assert isinstance(strong, pd.DataFrame)
    valid = strong["sample_valid"].astype(bool).to_numpy()
    angle = np.maximum(
        strong["q_hip_rad"].to_numpy(dtype=float),
        strong["q_knee_rad"].to_numpy(dtype=float),
    )
    residual = np.hypot(
        strong["identified_residual_hip_nm"].to_numpy(dtype=float),
        strong["identified_residual_knee_nm"].to_numpy(dtype=float),
    )
    lower = valid & (angle <= np.quantile(angle[valid], 0.25))
    upper = valid & (angle >= np.quantile(angle[valid], 0.75))
    assert residual[upper].mean() > 1.5 * residual[lower].mean()


def test_coupling_is_exposed_on_phase_shift_relative_to_matched_model(
    experiments: dict[str, dict[str, object]],
) -> None:
    matched = _trajectory_metric(experiments["matched_linear"], "phase_shift_small")
    coupled = _trajectory_metric(
        experiments["hip_knee_coupling_mild"],
        "phase_shift_small",
    )
    assert coupled["combined_torque_rmse_nm"] > (
        matched["combined_torque_rmse_nm"] + 1e-4
    )
    assert coupled["combined_nrmse_percent"] > matched[
        "combined_nrmse_percent"
    ]
    coupling_metrics = experiments["hip_knee_coupling_mild"][
        "trajectory_metrics"
    ]
    interpolation = coupling_metrics.loc[
        coupling_metrics["prediction_model"].eq("identified")
        & coupling_metrics["dataset_split"].eq("interpolation_test")
    ]
    worst_interpolation = interpolation.sort_values(
        "combined_torque_rmse_nm",
        ascending=False,
    ).iloc[0]
    assert worst_interpolation["trajectory_family"] == "phase_shift_small"


def test_outside_domain_is_not_better_than_interpolation_for_strong_case(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["nonlinear_stiffness_strong"]
    interpolation = _split_metric(result, "interpolation_test")
    outside = _split_metric(result, "outside_domain_test")
    assert outside["combined_torque_rmse_nm"] >= interpolation[
        "combined_torque_rmse_nm"
    ]
    assert outside["combined_nrmse_percent"] >= interpolation[
        "combined_nrmse_percent"
    ]


def test_all_fixed_splits_and_required_outputs_are_saved(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["combined_mild"]
    assert set(result["splits"]) == set(ALL_EVALUATION_SPLITS)
    output_dir = Path(result["output_dir"])
    required = {
        "training_data.csv",
        "validation_data.csv",
        "interpolation_test_data.csv",
        "boundary_test_data.csv",
        "outside_domain_test_data.csv",
        "estimated_parameters.json",
        "generator_parameters.json",
        "prediction_metrics.csv",
        "generic_vs_identified_comparison.csv",
        "residual_feature_correlations.csv",
        "predicted_vs_true_torque.csv",
        "metadata.json",
    }
    assert required.issubset(path.name for path in output_dir.iterdir())
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["angle_definition"] == "theta_shank = q_hip - q_knee"
    assert metadata["hip_angle_limit_deg"] == [0.0, 120.0]


def test_subtractive_shank_angle_is_preserved_in_unseen_geometry(
    experiments: dict[str, dict[str, object]],
) -> None:
    dataset = experiments["combined_mild"]["dataset"]
    unseen = dataset.loc[dataset["dataset_split"].eq("outside_domain_test")]
    q_hip = unseen["q_hip_rad"].to_numpy(dtype=float)
    q_knee = unseen["q_knee_rad"].to_numpy(dtype=float)
    expected_x = L1 * np.cos(q_hip) + L2 * np.cos(q_hip - q_knee)
    expected_z = L1 * np.sin(q_hip) + L2 * np.sin(q_hip - q_knee)
    assert np.allclose(unseen["x_pull_m"], expected_x, atol=1e-12, rtol=0.0)
    assert np.allclose(unseen["z_pull_m"], expected_z, atol=1e-12, rtol=0.0)


def test_dataset_generation_is_deterministic() -> None:
    first = build_model_mismatch_dataset(
        "baseline",
        "structured_residual",
        sampling_frequency_hz=5.0,
    )
    second = build_model_mismatch_dataset(
        "baseline",
        "structured_residual",
        sampling_frequency_hz=5.0,
    )
    pd.testing.assert_frame_equal(first, second, check_exact=True)
