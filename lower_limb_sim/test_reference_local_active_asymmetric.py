from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .reference_local_active_asymmetric import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    HELD_OUT_TRAJECTORY_IDS,
    SPLIT_BY_TRAJECTORY,
    TRAINING_TRAJECTORY_IDS,
    TRAJECTORY_SPECIFICATIONS,
    build_active_local_trajectories,
    build_excitation_metadata,
    load_active_reference,
    run_active_reference_local_identification,
    sha256_file,
)
from .run_reference_local_active_asymmetric import save_formal_result
from .visualize_reference_local_active_asymmetric import (
    FIGURE_DEFINITIONS,
    generate_active_reference_local_figures,
)


@pytest.fixture(scope="module")
def baseline_result():
    return run_active_reference_local_identification(subject_ids=("baseline",))


def test_manifest_declared_active_reference_is_exact_closed_c2_asymmetric():
    reference = load_active_reference()
    summary = reference.summary
    assert summary["active_reference_identifier"] == ACTIVE_REFERENCE_ID
    assert summary["active_reference_sha256"] == sha256_file(ACTIVE_REFERENCE_PATH)
    assert summary["trajectory_duration_s"] == pytest.approx(24.0)
    assert summary["sample_count"] == 401
    assert summary["sampling_interval"]["by_branch"]["flexion"][
        "median_s"
    ] == pytest.approx(0.068)
    assert summary["sampling_interval"]["by_branch"]["extension"][
        "median_s"
    ] == pytest.approx(0.052)
    assert summary["continuity_level"] == "C2"
    assert summary["is_asymmetric"] is True
    assert summary["q_start_equals_q_end"] is True
    assert summary["q_closure_error_rad"] == pytest.approx(0.0, abs=1e-14)
    assert summary["pull_point_closure_error_m"] == pytest.approx(0.0, abs=1e-14)
    assert summary["model_angle_identity_valid"] is True


def test_new_trajectory_set_has_no_silent_legacy_symmetric_fallback():
    reference = load_active_reference()
    trajectories = build_active_local_trajectories(reference)
    expected = {item.trajectory_id for item in TRAJECTORY_SPECIFICATIONS}
    assert set(trajectories) == expected
    assert set(TRAINING_TRAJECTORY_IDS).isdisjoint(HELD_OUT_TRAJECTORY_IDS)
    assert all(
        SPLIT_BY_TRAJECTORY[item] == "train" for item in TRAINING_TRAJECTORY_IDS
    )
    for trajectory_id, trajectory in trajectories.items():
        assert trajectory["active_reference_identifier"].eq(ACTIVE_REFERENCE_ID).all()
        assert trajectory["active_reference_sha256"].eq(
            reference.summary["active_reference_sha256"]
        ).all()
        assert "symmetric" not in trajectory_id
        assert "reference_closed_c2" not in trajectory_id
    heldout = trajectories["heldout_active_reference_slow"]
    assert heldout["active_reference_exact_samples"].astype(bool).all()
    np.testing.assert_array_equal(
        heldout["time_s"].to_numpy(dtype=float),
        reference.active["time_s"].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        heldout[["q_hip_rad", "q_knee_rad"]].to_numpy(dtype=float),
        reference.active[["q_hip_rad", "q_knee_rad"]].to_numpy(dtype=float),
    )


def test_every_excitation_preserves_closure_shank_sign_and_offline_gates():
    reference = load_active_reference()
    trajectories = build_active_local_trajectories(reference)
    metadata = build_excitation_metadata(trajectories)
    assert metadata["offline_feasible"].astype(bool).all()
    assert metadata["closed"].astype(bool).all()
    assert metadata["c2_stationary_seam"].astype(bool).all()
    assert metadata["rom_valid"].astype(bool).all()
    assert metadata["workspace_valid"].astype(bool).all()
    assert metadata["jacobian_valid"].astype(bool).all()
    assert not metadata["velocity_limit_configured"].astype(bool).any()
    assert not metadata["acceleration_limit_configured"].astype(bool).any()
    for trajectory in trajectories.values():
        np.testing.assert_allclose(
            trajectory["theta_shank_rad"],
            trajectory["q_hip_rad"] - trajectory["q_knee_rad"],
            atol=1e-14,
            rtol=0.0,
        )


def test_train_only_fit_and_heldout_predictions_are_separate(baseline_result):
    dataset = baseline_result.dataset
    training_ids = set(
        dataset.loc[dataset["dataset_split"].eq("train"), "trajectory_id"]
    )
    heldout_ids = set(
        dataset.loc[dataset["dataset_split"].eq("test"), "trajectory_id"]
    )
    assert training_ids == set(TRAINING_TRAJECTORY_IDS)
    assert heldout_ids == set(HELD_OUT_TRAJECTORY_IDS)
    assert training_ids.isdisjoint(heldout_ids)
    assert not baseline_result.identified_parameters["test_used_for_fit"].astype(
        bool
    ).any()
    assert baseline_result.identified_parameters["fit_split"].eq("train_only").all()
    forbidden = [
        column
        for column in dataset.columns
        if column.startswith("true_")
        or column.startswith("tau_total")
        or column
        in {
            "mass_scale",
            "k_hip_nm_per_rad",
            "k_knee_nm_per_rad",
            "b_hip_nm_s_per_rad",
            "b_knee_nm_s_per_rad",
        }
    ]
    assert forbidden == []
    predicted_test_ids = set(
        baseline_result.prediction_metrics.loc[
            baseline_result.prediction_metrics["dataset_split"].eq("test"),
            "trajectory_id",
        ]
    )
    assert predicted_test_ids == set(HELD_OUT_TRAJECTORY_IDS)


def test_active_local_identifiability_and_domain_are_recomputed(baseline_result):
    summary = baseline_result.identifiability_summary.iloc[0]
    assert summary["numerical_rank"] == 5
    assert bool(summary["full_rank_five_parameter_model"])
    assert summary["condition_number"] < 100.0
    assert summary["highly_correlated_pair_count"] == 0
    assert baseline_result.domain_bounds.valid_training_samples == 6 * 401
    exact_active = baseline_result.domain_coverage.loc[
        baseline_result.domain_coverage["trajectory_id"].eq(
            "heldout_active_reference_slow"
        )
    ].iloc[0]
    assert exact_active["dataset_split"] == "test"
    assert exact_active["in_domain_percent"] > 99.0
    boundary = baseline_result.domain_coverage.loc[
        baseline_result.domain_coverage["trajectory_id"].eq(
            "heldout_boundary_speed_plus_10pct"
        )
    ].iloc[0]
    assert boundary["outside_domain_percent"] > 0.0


def test_active_local_computation_is_deterministic(baseline_result):
    repeated = run_active_reference_local_identification(subject_ids=("baseline",))
    pd.testing.assert_frame_equal(
        baseline_result.identified_parameters,
        repeated.identified_parameters,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        baseline_result.domain_coverage,
        repeated.domain_coverage,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        baseline_result.identifiability_summary,
        repeated.identifiability_summary,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        baseline_result.prediction_metrics,
        repeated.prediction_metrics,
        check_exact=True,
    )


def test_formal_output_metadata_identifies_active_reference_and_refuses_overwrite(
    baseline_result,
    tmp_path: Path,
):
    before = sha256_file(ACTIVE_REFERENCE_PATH)
    output = tmp_path / "formal"
    paths = save_formal_result(
        baseline_result,
        output,
        generate_plots=False,
        targeted_test_result="passed: targeted test fixture",
        full_test_result="passed: full test fixture",
    )
    assert before == sha256_file(ACTIVE_REFERENCE_PATH)
    required = {
        "experiment_config.json",
        "reference_metadata.json",
        "excitation_metadata.csv",
        "domain_coverage.csv",
        "identified_parameters.csv",
        "prediction_metrics.csv",
        "parameter_errors.csv",
        "identifiability_summary.csv",
        "generic_vs_identified.csv",
        "run_summary.json",
    }
    assert required.issubset(paths)
    summary = json.loads(paths["run_summary.json"].read_text(encoding="utf-8"))
    assert summary["active_reference_identifier"] == ACTIVE_REFERENCE_ID
    assert summary["active_reference_sha256"] == before
    assert summary["acceptance_criteria"]["current_active_reference_only"] is True
    assert summary["acceptance_criteria"][
        "no_legacy_symmetric_trajectory_in_new_dataset"
    ] is True
    config = json.loads(paths["experiment_config.json"].read_text(encoding="utf-8"))
    assert config["split_definition"]["train"] == list(TRAINING_TRAJECTORY_IDS)
    assert config["split_definition"]["test"] == list(HELD_OUT_TRAJECTORY_IDS)
    with pytest.raises(FileExistsError):
        save_formal_result(baseline_result, output, generate_plots=False)


def test_all_five_scientific_figures_are_generated(baseline_result, tmp_path: Path):
    paths = generate_active_reference_local_figures(baseline_result, tmp_path)
    assert set(paths) == set(FIGURE_DEFINITIONS)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_active_local_modules_do_not_import_prohibited_runtime_packages():
    module_directory = Path(__file__).resolve().parent
    for name in (
        "reference_local_active_asymmetric.py",
        "run_reference_local_active_asymmetric.py",
        "visualize_reference_local_active_asymmetric.py",
    ):
        source = (module_directory / name).read_text(encoding="utf-8")
        for token in (
            "from hardware",
            "import hardware",
            "from control",
            "import control",
            "from collection",
            "import collection",
            "from safety",
            "import safety",
            "xCoreSDK",
            "RokaeRobot",
        ):
            assert token not in source
