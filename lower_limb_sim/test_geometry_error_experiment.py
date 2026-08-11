"""Stage 4.5D end-to-end, leakage-boundary, and regression tests.

The experiments use a reduced 25 Hz sampling rate to keep the full three-mode
checks fast while remaining above the configured derivative-gap limit.  No
test data, simulator truth, or true geometry is exposed to the estimator.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    hip_range_deg,
    identification_lower_bounds,
    identification_upper_bounds,
)
from lower_limb_sim.geometry_calibration import AssumedGeometry, TrueGeometry
from lower_limb_sim.geometry_error_metrics import (
    TRUE_DOMAIN_STATE_COLUMNS,
    classify_state_domain,
    fit_state_domain_bounds,
)
from lower_limb_sim.geometry_error_scenarios import (
    INDEPENDENT_JOINT_MEASUREMENT,
    OBSERVATION_MODES,
    ORACLE_TRUE_JOINT_STATE,
    TCP_INVERSE_KINEMATICS,
)
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.parameter_estimator import PARAMETER_NAMES
from lower_limb_sim.run_geometry_error_experiment import (
    GEOMETRY_ESTIMATOR_INPUT_COLUMNS,
    build_geometry_observation_dataset,
    fit_five_parameter_observation,
    project_geometry_estimator_inputs,
    run_geometry_error_experiment,
)
from lower_limb_sim.run_model_mismatch_experiment import (
    ALL_EVALUATION_SPLITS,
    run_model_mismatch_experiment,
)


SAMPLING_FREQUENCY_HZ = 25.0
SCENARIOS = (
    "matched_geometry",
    "combined_geometry_mild",
    "combined_geometry_strong",
)


@pytest.fixture(scope="module")
def experiment_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("geometry_error_experiments")


@pytest.fixture(scope="module")
def experiments(experiment_root: Path) -> dict[str, dict[str, object]]:
    return {
        scenario: run_geometry_error_experiment(
            "baseline",
            scenario,
            output_root=experiment_root,
            sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
            make_plots=False,
        )
        for scenario in SCENARIOS
    }


@pytest.fixture(scope="module")
def stage45c_matched(experiment_root: Path) -> dict[str, object]:
    return run_model_mismatch_experiment(
        "baseline",
        "matched_linear",
        output_root=experiment_root / "stage45c_reference",
        sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
        make_plots=False,
    )


def _mode_row(dataframe: pd.DataFrame, mode: str) -> pd.Series:
    selected = dataframe.loc[dataframe["observation_mode"].eq(mode)]
    assert len(selected) == 1
    return selected.iloc[0]


def _prediction_row(
    result: dict[str, object],
    *,
    mode: str,
    split: str,
    model: str,
) -> pd.Series:
    metrics = result["prediction_metrics"]
    assert isinstance(metrics, pd.DataFrame)
    selected = metrics.loc[
        metrics["observation_mode"].eq(mode)
        & metrics["dataset_split"].eq(split)
        & metrics["prediction_model"].eq(model)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_oracle_mode_recovers_stage45c_matched_result(
    experiments: dict[str, dict[str, object]],
    stage45c_matched: dict[str, object],
) -> None:
    geometry_result = experiments["matched_geometry"]
    oracle = geometry_result["estimates"][ORACLE_TRUE_JOINT_STATE]
    stage45c = stage45c_matched["estimate"]

    assert oracle.optimizer_success
    assert stage45c.optimizer_success
    assert tuple(oracle.estimated_parameters) == tuple(PARAMETER_NAMES)
    for parameter in PARAMETER_NAMES:
        assert oracle.estimated_parameters[parameter] == pytest.approx(
            stage45c.estimated_parameters[parameter],
            abs=2e-8,
        )
    oracle_interpolation = _prediction_row(
        geometry_result,
        mode=ORACLE_TRUE_JOINT_STATE,
        split="interpolation_test",
        model="identified",
    )
    assert oracle_interpolation["combined_torque_rmse_nm"] < 1e-8
    assert oracle_interpolation["combined_torque_nrmse_percent"] < 1e-6


def test_tcp_pipeline_passes_no_true_joint_state_to_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lower_limb_sim.run_geometry_error_experiment as runner

    original = runner.build_kinematic_observation
    captured: dict[str, tuple[str, ...]] = {}

    def checked_observer(dataframe, observation_mode, *args, **kwargs):
        captured[str(observation_mode)] = tuple(dataframe.columns)
        return original(dataframe, observation_mode, *args, **kwargs)

    monkeypatch.setattr(runner, "build_kinematic_observation", checked_observer)
    runner.build_geometry_observation_dataset(
        "baseline",
        "matched_geometry",
        sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
        observation_modes=(TCP_INVERSE_KINEMATICS,),
    )

    columns = captured[TCP_INVERSE_KINEMATICS]
    assert {"x_pull_measured_m", "z_pull_measured_m"}.issubset(columns)
    assert not any("true" in column.lower() for column in columns)
    assert "q_hip_measured_rad" not in columns
    assert "q_knee_measured_rad" not in columns


def test_independent_pipeline_uses_only_measured_joint_angles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lower_limb_sim.run_geometry_error_experiment as runner

    original = runner.build_kinematic_observation
    captured: dict[str, tuple[str, ...]] = {}

    def checked_observer(dataframe, observation_mode, *args, **kwargs):
        captured[str(observation_mode)] = tuple(dataframe.columns)
        return original(dataframe, observation_mode, *args, **kwargs)

    monkeypatch.setattr(runner, "build_kinematic_observation", checked_observer)
    truth, observed, reconstructed, *_ = runner.build_geometry_observation_dataset(
        "baseline",
        "independent_angle_noise_medium",
        sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
        observation_modes=(INDEPENDENT_JOINT_MEASUREMENT,),
    )

    columns = captured[INDEPENDENT_JOINT_MEASUREMENT]
    assert {"q_hip_measured_rad", "q_knee_measured_rad"}.issubset(columns)
    assert not any("true" in column.lower() for column in columns)
    assert "x_pull_measured_m" not in columns
    np.testing.assert_array_equal(
        reconstructed["q_hip_est_rad"].to_numpy(dtype=float),
        observed["q_hip_measured_rad"].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        reconstructed["q_knee_est_rad"].to_numpy(dtype=float),
        observed["q_knee_measured_rad"].to_numpy(dtype=float),
    )
    assert not np.allclose(
        reconstructed["q_hip_est_rad"],
        truth["q_hip_true_rad"],
    )


def test_only_training_split_reaches_fit_and_filter_choice_is_prefixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lower_limb_sim.run_geometry_error_experiment as runner

    original_projection = runner.project_geometry_estimator_inputs
    projected_source_splits: list[set[str]] = []

    def checked_projection(dataframe: pd.DataFrame) -> pd.DataFrame:
        if "dataset_split" in dataframe:
            projected_source_splits.append(
                set(dataframe["dataset_split"].astype(str))
            )
        return original_projection(dataframe)

    monkeypatch.setattr(
        runner,
        "project_geometry_estimator_inputs",
        checked_projection,
    )
    result = runner.run_geometry_error_experiment(
        "baseline",
        "tcp_position_noise_low",
        output_root=tmp_path,
        sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
        observation_modes=(TCP_INVERSE_KINEMATICS,),
        make_plots=False,
        save_outputs=False,
    )

    assert projected_source_splits == [{"train"}]
    reconstructed = result["reconstructed_state"]
    assert reconstructed.groupby("observation_mode")[
        "derivative_method"
    ].nunique().eq(1).all()
    metadata = result["metadata"]
    assert metadata["fit_split"] == ["train"]
    assert metadata["validation_used_for_parameter_fit"] is False
    assert (
        metadata[
            "test_used_for_fit_filter_derivative_geometry_or_model_selection"
        ]
        is False
    )
    assert metadata["derivative_configuration_pre_fixed"] is True


def test_main_estimator_remains_five_parameters_and_q0_is_audit_only(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["combined_geometry_mild"]
    true_geometry = result["true_geometry"]
    assumed_geometry = result["assumed_geometry"]
    assert assumed_geometry.q0_hip_assumed_rad != true_geometry.q0_hip_true_rad
    assert assumed_geometry.q0_knee_assumed_rad != true_geometry.q0_knee_true_rad

    for estimate in result["estimates"].values():
        assert tuple(estimate.estimated_parameters) == tuple(PARAMETER_NAMES)
        assert len(estimate.estimated_parameters) == 5
        assert not any("q0" in name.lower() for name in estimate.estimated_parameters)
    q0_audit = result["q0_correlation"]
    assert len(q0_audit) == len(OBSERVATION_MODES)
    assert q0_audit["main_estimator_parameter_count"].eq(5).all()
    assert not q0_audit["q0_included_in_main_estimator"].astype(bool).any()
    metadata = result["metadata"]
    assert metadata["main_estimator_parameter_count"] == 5
    assert metadata["q0_in_main_estimator"] is False


@pytest.mark.parametrize(
    "scenario_name",
    ["combined_geometry_mild", "combined_geometry_strong"],
)
def test_mild_and_strong_scenarios_are_completely_reported_without_deletion(
    experiments: dict[str, dict[str, object]],
    scenario_name: str,
) -> None:
    result = experiments[scenario_name]
    reconstructed = result["reconstructed_state"]
    truth = result["truth"]
    assert len(reconstructed) == len(truth) * len(OBSERVATION_MODES)
    assert set(reconstructed["dataset_split"].astype(str)) == set(
        ALL_EVALUATION_SPLITS
    )
    assert set(reconstructed["observation_mode"].astype(str)) == set(
        OBSERVATION_MODES
    )

    expected_mode_splits = {
        (mode, split)
        for mode in OBSERVATION_MODES
        for split in ALL_EVALUATION_SPLITS
    }
    kinematic_metrics = result["kinematic_metrics"]
    assert set(
        zip(
            kinematic_metrics["observation_mode"],
            kinematic_metrics["dataset_split"],
        )
    ) == expected_mode_splits
    expected_predictions = {
        (mode, split, model)
        for mode in OBSERVATION_MODES
        for split in ALL_EVALUATION_SPLITS
        for model in ("generic", "identified")
    }
    prediction_metrics = result["prediction_metrics"]
    assert set(
        zip(
            prediction_metrics["observation_mode"],
            prediction_metrics["dataset_split"],
            prediction_metrics["prediction_model"],
        )
    ) == expected_predictions
    assert (prediction_metrics["valid_torque_samples"] > 0).all()


def test_strong_error_is_retained_and_larger_than_mild_error(
    experiments: dict[str, dict[str, object]],
) -> None:
    mild = _prediction_row(
        experiments["combined_geometry_mild"],
        mode=TCP_INVERSE_KINEMATICS,
        split="interpolation_test",
        model="identified",
    )
    strong = _prediction_row(
        experiments["combined_geometry_strong"],
        mode=TCP_INVERSE_KINEMATICS,
        split="interpolation_test",
        model="identified",
    )
    assert strong["combined_torque_rmse_nm"] > mild[
        "combined_torque_rmse_nm"
    ]
    assert strong["combined_torque_nrmse_percent"] > mild[
        "combined_torque_nrmse_percent"
    ]
    strong_parameter_error = _mode_row(
        experiments["combined_geometry_strong"]["identification_metrics"],
        TCP_INVERSE_KINEMATICS,
    )["maximum_parameter_error_percent"]
    mild_parameter_error = _mode_row(
        experiments["combined_geometry_mild"]["identification_metrics"],
        TCP_INVERSE_KINEMATICS,
    )["maximum_parameter_error_percent"]
    assert strong_parameter_error > mild_parameter_error


def test_identified_model_improves_over_generic_for_mild_geometry_error(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["combined_geometry_mild"]
    generic = _prediction_row(
        result,
        mode=TCP_INVERSE_KINEMATICS,
        split="interpolation_test",
        model="generic",
    )
    identified = _prediction_row(
        result,
        mode=TCP_INVERSE_KINEMATICS,
        split="interpolation_test",
        model="identified",
    )
    assert identified["combined_torque_rmse_nm"] < generic[
        "combined_torque_rmse_nm"
    ]
    assert identified["combined_torque_nrmse_percent"] < generic[
        "combined_torque_nrmse_percent"
    ]
    assert identified["generic_vs_identified_improvement_percent"] > 0.0


def test_true_geometry_is_evaluation_only_and_never_reaches_estimator(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["combined_geometry_mild"]
    assert isinstance(result["true_geometry"], TrueGeometry)
    assert isinstance(result["assumed_geometry"], AssumedGeometry)
    signature = inspect.signature(fit_five_parameter_observation)
    assert "assumed_geometry" in signature.parameters
    assert not any("true" in name.lower() for name in signature.parameters)

    for dataframe in result["estimator_inputs"].values():
        assert tuple(dataframe.columns) == GEOMETRY_ESTIMATOR_INPUT_COLUMNS
        assert not dataframe.attrs
        assert not any("true" in column.lower() for column in dataframe)
    metadata = result["metadata"]
    assert metadata["true_geometry_available_to_estimator"] is False
    assert metadata["true_state_available_to_nonoracle_estimator"] is False
    assert metadata["true_domain_membership_evaluation_only"] is True


def test_runtime_domain_classifier_is_invariant_to_true_state_tampering(
    experiments: dict[str, dict[str, object]],
) -> None:
    reconstructed = experiments["combined_geometry_mild"][
        "reconstructed_state"
    ]
    frame = reconstructed.loc[
        reconstructed["observation_mode"].eq(TCP_INVERSE_KINEMATICS)
    ].copy()
    training = frame.loc[frame["dataset_split"].eq("train")]
    bounds = fit_state_domain_bounds(training)
    expected = classify_state_domain(frame, bounds)

    tampered = frame.copy()
    for offset, column in enumerate(TRUE_DOMAIN_STATE_COLUMNS, start=1):
        tampered[column] = 1e6 * offset
    observed = classify_state_domain(tampered, bounds)
    np.testing.assert_array_equal(observed, expected)
    assert not any("true" in column.lower() for column in bounds.columns)
    with pytest.raises(ValueError, match="non-true"):
        fit_state_domain_bounds(training, columns=TRUE_DOMAIN_STATE_COLUMNS)


def test_estimator_whitelist_blocks_parameter_scenario_split_and_geometry_leaks(
    experiments: dict[str, dict[str, object]],
) -> None:
    reconstructed = experiments["combined_geometry_mild"][
        "reconstructed_state"
    ]
    training = reconstructed.loc[
        reconstructed["observation_mode"].eq(TCP_INVERSE_KINEMATICS)
        & reconstructed["dataset_split"].eq("train")
    ].copy()
    expected = project_geometry_estimator_inputs(training)
    tampered = training.copy()
    tampered.attrs["true_geometry"] = {"L1_true_m": 999.0}
    tampered["subject_id"] = "leak-subject"
    tampered["scenario_name"] = "leak-scenario"
    tampered["dataset_split"] = "outside_domain_test"
    tampered["L1_true_m"] = 999.0
    tampered["L2_assumed_m"] = -999.0
    tampered["mass_scale_true"] = 42.0
    tampered["q_hip_true_rad"] = -1e6
    observed = project_geometry_estimator_inputs(tampered)

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    assert tuple(observed.columns) == GEOMETRY_ESTIMATOR_INPUT_COLUMNS
    assert not observed.attrs
    forbidden_tokens = ("true", "subject", "scenario", "split", "geometry")
    assert not any(
        token in column.lower()
        for column in observed
        for token in forbidden_tokens
    )


def test_all_required_experiment_outputs_and_geometry_namespaces_are_saved(
    experiments: dict[str, dict[str, object]],
) -> None:
    output_directory = Path(
        experiments["combined_geometry_mild"]["output_dir"]
    )
    required = {
        "true_trajectory.csv",
        "observed_trajectory.csv",
        "reconstructed_state.csv",
        "training_data.csv",
        "validation_data.csv",
        "interpolation_test_data.csv",
        "boundary_test_data.csv",
        "outside_domain_test_data.csv",
        "estimated_parameters.json",
        "true_geometry.json",
        "assumed_geometry.json",
        "kinematic_metrics.csv",
        "identification_metrics.csv",
        "prediction_metrics.csv",
        "domain_metrics.csv",
        "generic_vs_identified_comparison.csv",
        "observation_mode_comparison.csv",
        "q0_k_correlation.csv",
        "metadata.json",
    }
    assert required.issubset(path.name for path in output_directory.iterdir())

    true_geometry = json.loads(
        (output_directory / "true_geometry.json").read_text(encoding="utf-8")
    )
    assumed_geometry = json.loads(
        (output_directory / "assumed_geometry.json").read_text(encoding="utf-8")
    )
    assert true_geometry["evaluation_only"] is True
    assert true_geometry["available_to_estimator"] is False
    assert all("true" in key or key in {"evaluation_only", "available_to_estimator"}
               for key in true_geometry)
    assert all("assumed" in key for key in assumed_geometry)


def test_stage45d_preserves_subtractive_shank_angle_and_120_degree_hip_limit(
    experiments: dict[str, dict[str, object]],
) -> None:
    result = experiments["combined_geometry_mild"]
    truth = result["truth"]
    true_geometry = result["true_geometry"]
    q_hip = truth["q_hip_true_rad"].to_numpy(dtype=float)
    q_knee = truth["q_knee_true_rad"].to_numpy(dtype=float)
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        true_geometry.L1_true_m,
        true_geometry.L2_true_m,
    )
    expected_x = (
        true_geometry.L1_true_m * np.cos(q_hip)
        + true_geometry.L2_true_m * np.cos(q_hip - q_knee)
        + true_geometry.hip_center_x_true_m
    )
    expected_z = (
        true_geometry.L1_true_m * np.sin(q_hip)
        + true_geometry.L2_true_m * np.sin(q_hip - q_knee)
        + true_geometry.hip_center_z_true_m
    )
    np.testing.assert_allclose(x_pull + true_geometry.hip_center_x_true_m, expected_x)
    np.testing.assert_allclose(z_pull + true_geometry.hip_center_z_true_m, expected_z)
    np.testing.assert_allclose(truth["x_pull_true_m"], expected_x)
    np.testing.assert_allclose(truth["z_pull_true_m"], expected_z)
    assert tuple(float(value) for value in hip_range_deg) == (0.0, 120.0)
    assert result["metadata"]["angle_definition"] == (
        "theta_shank = q_hip - q_knee"
    )
    assert result["metadata"]["hip_angle_limit_deg"] == [0.0, 120.0]


def test_estimates_remain_inside_existing_physical_bounds(
    experiments: dict[str, dict[str, object]],
) -> None:
    for result in experiments.values():
        for estimate in result["estimates"].values():
            for parameter, value in estimate.estimated_parameters.items():
                assert identification_lower_bounds[parameter] <= value
                assert value <= identification_upper_bounds[parameter]


def test_stage45d_has_no_real_robot_sdk_or_runtime_stack_import() -> None:
    package_directory = Path(__file__).resolve().parent
    repository_root = package_directory.parent
    implementation_files = (
        "geometry_calibration.py",
        "geometry_error_scenarios.py",
        "angle_reconstruction.py",
        "kinematic_observation.py",
        "derivative_estimation.py",
        "geometry_error_metrics.py",
        "run_geometry_error_experiment.py",
        "visualize_geometry_error.py",
    )
    forbidden_roots = {
        "hardware",
        "control",
        "collection",
        "xCoreSDK",
        "rokae",
    }
    for filename in implementation_files:
        path = package_directory / filename
        assert path.is_file()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])
        assert imported_roots.isdisjoint(forbidden_roots)

    # Worktree cleanliness was a rollout-local assertion when Stage 4.5D was
    # first added.  The repository now intentionally changes the robot runtime;
    # the durable boundary is that this offline stage does not import it.
