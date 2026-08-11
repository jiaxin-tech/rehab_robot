"""Stage 5A reference marker trajectory import tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.config import L1, L2
from lower_limb_sim.reference_trajectory_import import (
    REQUIRED_COLUMNS,
    import_reference_trajectory_csv,
    import_reference_trajectory_dataframe,
    validate_reference_trajectory_dataframe,
)
from lower_limb_sim.run_reference_trajectory import (
    DYNAMIC_FILENAMES,
    run_reference_trajectory,
)
from lower_limb_sim.visualize_reference_trajectory import (
    DYNAMIC_FIGURES,
    FIGURE_FILENAMES,
    GEOMETRY_FIGURES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REAL_REFERENCE_CSV = REPOSITORY_ROOT / "bone_return_3_leg.csv"


def _synthetic_bilateral_markers(
    q_hip_rad: np.ndarray,
    q_knee_rad: np.ndarray,
    *,
    first_frame: int = 100,
    right_knee_lateral_offset_m: float = 0.0,
) -> pd.DataFrame:
    q_hip = np.asarray(q_hip_rad, dtype=float)
    q_knee = np.asarray(q_knee_rad, dtype=float)
    if q_hip.ndim != 1 or q_hip.shape != q_knee.shape:
        raise ValueError("synthetic angles must be equal-length 1-D arrays.")
    count = len(q_hip)
    thigh_length_m = 0.42
    anatomical_shank_length_m = 0.36
    theta_shank = q_hip - q_knee

    right_hip = np.tile(np.array([0.0, -0.20, 0.0]), (count, 1))
    right_knee = right_hip + np.column_stack(
        (
            thigh_length_m * np.cos(q_hip),
            np.full(count, right_knee_lateral_offset_m),
            thigh_length_m * np.sin(q_hip),
        )
    )
    right_ankle = right_knee + np.column_stack(
        (
            anatomical_shank_length_m * np.cos(theta_shank),
            np.zeros(count),
            anatomical_shank_length_m * np.sin(theta_shank),
        )
    )
    # The left/contralateral leg is stationary and establishes +x toward feet.
    left_hip = np.tile(np.array([0.0, 0.20, 0.0]), (count, 1))
    left_knee = np.tile(np.array([0.42, 0.20, 0.0]), (count, 1))
    left_ankle = np.tile(np.array([0.78, 0.20, 0.0]), (count, 1))
    dataframe = pd.DataFrame({"Frame": np.arange(first_frame, first_frame + count)})
    for name, values in (
        ("LHip", left_hip),
        ("RHip", right_hip),
        ("LKnee", left_knee),
        ("RKnee", right_knee),
        ("LAnkle", left_ankle),
        ("RAnkle", right_ankle),
    ):
        for axis_index, axis in enumerate("XYZ"):
            dataframe[f"{name}_{axis}"] = values[:, axis_index]
    return dataframe


def _scale_marker_coordinates(dataframe_m: pd.DataFrame, scale: float) -> pd.DataFrame:
    output = dataframe_m.copy()
    coordinate_columns = [column for column in REQUIRED_COLUMNS if column != "Frame"]
    output.loc[:, coordinate_columns] *= scale
    return output


def test_required_frame_and_bilateral_leg_columns_are_enforced() -> None:
    dataframe = _synthetic_bilateral_markers(
        np.deg2rad(np.linspace(20.0, 60.0, 10)),
        np.deg2rad(np.linspace(30.0, 90.0, 10)),
    )
    assert tuple(dataframe.columns) == REQUIRED_COLUMNS
    with pytest.raises(ValueError, match="missing required columns"):
        import_reference_trajectory_dataframe(
            dataframe.drop(columns="RAnkle_Z"),
            coordinate_unit="m",
        )


def test_missing_required_values_are_rejected_but_optional_nan_is_irrelevant() -> None:
    dataframe = _synthetic_bilateral_markers(
        np.deg2rad(np.linspace(20.0, 60.0, 10)),
        np.deg2rad(np.linspace(30.0, 90.0, 10)),
    )
    dataframe["optional_marker"] = np.nan
    result = import_reference_trajectory_dataframe(
        dataframe,
        coordinate_unit="m",
    )
    assert len(result) == len(dataframe)
    invalid = dataframe.copy()
    invalid.loc[3, "RKnee_Z"] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        import_reference_trajectory_dataframe(invalid, coordinate_unit="m")


@pytest.mark.parametrize(
    "frames",
    (
        [1, 2, 2, 3],
        [1, 3, 2, 4],
    ),
)
def test_frame_must_be_strictly_increasing(frames: list[int]) -> None:
    dataframe = _synthetic_bilateral_markers(
        np.deg2rad(np.linspace(20.0, 60.0, 4)),
        np.deg2rad(np.linspace(30.0, 90.0, 4)),
    )
    dataframe["Frame"] = frames
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_reference_trajectory_dataframe(dataframe)


def test_frame_gaps_are_retained_and_explicitly_audited() -> None:
    dataframe = _synthetic_bilateral_markers(
        np.deg2rad(np.linspace(20.0, 60.0, 5)),
        np.deg2rad(np.linspace(30.0, 90.0, 5)),
    )
    dataframe["Frame"] = [100, 101, 104, 105, 109]
    result = import_reference_trajectory_dataframe(
        dataframe,
        coordinate_unit="m",
        primary_motion_leg="right",
    )

    assert not result.frame_audit.continuous_unit_steps
    assert result.frame_audit.gap_count == 2
    assert result.frame_audit.missing_frame_count == 5
    assert result.frame_audit.largest_step == 4
    assert result.frame_audit.gap_after_frames == (101, 105)
    assert result.trajectory["Frame"].tolist() == [100, 101, 104, 105, 109]
    assert result.trajectory["frame_contiguous_from_previous"].tolist() == [
        True,
        True,
        False,
        True,
        False,
    ]


def test_coordinate_unit_must_be_explicit_and_only_mm_or_m() -> None:
    dataframe = _synthetic_bilateral_markers(
        np.deg2rad(np.linspace(20.0, 60.0, 10)),
        np.deg2rad(np.linspace(30.0, 90.0, 10)),
    )
    with pytest.raises(TypeError, match="coordinate_unit"):
        import_reference_trajectory_dataframe(dataframe)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="explicitly 'mm' or 'm'"):
        import_reference_trajectory_dataframe(dataframe, coordinate_unit="cm")


def test_explicit_mm_and_m_inputs_produce_identical_SI_results() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 70.0, 31))
    q_knee = np.deg2rad(np.linspace(30.0, 100.0, 31))
    dataframe_m = _synthetic_bilateral_markers(q_hip, q_knee)
    dataframe_mm = _scale_marker_coordinates(dataframe_m, 1000.0)
    result_m = import_reference_trajectory_dataframe(
        dataframe_m,
        coordinate_unit="m",
    )
    result_mm = import_reference_trajectory_dataframe(
        dataframe_mm,
        coordinate_unit="mm",
    )

    assert_frame_equal(result_m.marker_dataframe_m, result_mm.marker_dataframe_m)
    numeric_columns = result_m.trajectory.select_dtypes(include=[np.number]).columns
    assert np.allclose(
        result_m.trajectory[numeric_columns],
        result_mm.trajectory[numeric_columns],
        equal_nan=True,
        atol=1e-12,
    )
    assert result_mm.metadata["coordinate_scale_to_m"] == pytest.approx(1e-3)


def test_synthetic_matched_angles_are_recovered_in_local_sagittal_frame() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 80.0, 61))
    q_knee = np.deg2rad(np.linspace(30.0, 110.0, 61))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, q_knee),
        coordinate_unit="m",
    )

    assert result.primary_motion_leg == "right"
    assert np.max(np.abs(result["q_hip_rad"] - q_hip)) < 1e-12
    assert np.max(np.abs(result["q_knee_rad"] - q_knee)) < 1e-12
    assert np.max(
        np.abs(result["theta_shank_rad"] - (q_hip - q_knee))
    ) < 1e-12
    assert result["joint_range_valid"].all()


def test_theta_shank_is_strictly_qhip_minus_qknee() -> None:
    q_hip = np.deg2rad(np.linspace(10.0, 90.0, 41))
    q_knee = np.deg2rad(np.linspace(20.0, 120.0, 41))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, q_knee),
        coordinate_unit="m",
    )
    assert np.array_equal(
        result["theta_shank_definition"].unique(),
        np.array(["q_hip - q_knee"], dtype=object),
    )
    assert np.allclose(
        result["theta_shank_rad"],
        result["q_hip_rad"] - result["q_knee_rad"],
        atol=1e-15,
    )
    assert result.metadata["model_angle_definition"] == (
        "theta_shank = q_hip - q_knee"
    )
    assert np.max(
        np.abs(
            result["theta_shank_projected_rad"]
            - result["theta_shank_rad"]
        )
    ) < 1e-12


def test_knee_angle_is_nonnegative_and_opposite_projection_branch_is_audited() -> None:
    q_hip = np.deg2rad(np.array([30.0, 35.0, 40.0]))
    apparent_hyperextension = np.deg2rad(np.array([-4.0, -2.0, -1.0]))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, apparent_hyperextension),
        coordinate_unit="m",
        primary_motion_leg="right",
    )

    assert np.all(result["q_knee_rad"] >= 0.0)
    assert np.allclose(result["q_knee_rad"], np.abs(apparent_hyperextension))
    assert not result["angle_valid"].any()
    assert result["angle_invalid_reason"].str.contains(
        "projected_knee_branch_opposes_flexion"
    ).all()
    assert np.allclose(
        result["theta_shank_rad"],
        result["q_hip_rad"] - result["q_knee_rad"],
    )


def test_out_of_range_angles_are_flagged_without_clipping() -> None:
    q_hip = np.deg2rad(np.array([30.0, 80.0, 130.0]))
    q_knee = np.deg2rad(np.array([40.0, 100.0, 145.0]))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, q_knee),
        coordinate_unit="m",
        primary_motion_leg="right",
    )
    output = result.trajectory

    assert output.loc[2, "q_hip_deg"] == pytest.approx(130.0)
    assert output.loc[2, "q_knee_deg"] == pytest.approx(145.0)
    assert not output.loc[2, "joint_range_valid"]
    assert "q_hip_out_of_range" in output.loc[2, "joint_range_reason"]
    assert "q_knee_out_of_range" in output.loc[2, "joint_range_reason"]
    assert not output["joint_angles_clipped"].any()
    assert result.metadata["joint_angle_clipping_applied"] is False


def test_axes_are_orthonormal_and_follow_foot_and_flexion_directions() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 80.0, 41))
    q_knee = np.deg2rad(np.linspace(30.0, 100.0, 41))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, q_knee),
        coordinate_unit="m",
    )
    frame = result.sagittal_frame
    axes = np.stack((frame.x_axis, frame.z_axis, frame.plane_normal))

    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    assert frame.x_axis == pytest.approx(np.array([1.0, 0.0, 0.0]))
    assert frame.z_axis == pytest.approx(np.array([0.0, 0.0, 1.0]))
    assert np.all(result["RKnee_z_local_m"] > 0.0)
    assert frame.sagittal_plane_normal_world == frame.lateral_axis_world


def test_out_of_plane_error_is_saved_without_changing_sagittal_projection() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 70.0, 31))
    q_knee = np.deg2rad(np.linspace(30.0, 100.0, 31))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(
            q_hip,
            q_knee,
            right_knee_lateral_offset_m=0.02,
        ),
        coordinate_unit="m",
    )

    assert np.allclose(result["RKnee_planarity_error_m"], 0.02, atol=1e-12)
    assert np.allclose(result["q_hip_rad"], q_hip, atol=1e-12)
    assert result.metadata["primary_leg_planarity_rmse_m"] > 0.0


def test_fixed_plane_origin_does_not_hide_moving_hip_out_of_plane_motion() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 70.0, 31))
    q_knee = np.deg2rad(np.linspace(30.0, 100.0, 31))
    dataframe = _synthetic_bilateral_markers(q_hip, q_knee)
    lateral_shift = np.linspace(-0.03, 0.03, len(dataframe))
    for marker in ("RHip", "RKnee", "RAnkle"):
        dataframe[f"{marker}_Y"] += lateral_shift
    result = import_reference_trajectory_dataframe(
        dataframe,
        coordinate_unit="m",
        primary_motion_leg="right",
    )

    hip_error = result["RHip_planarity_error_m"].to_numpy(dtype=float)
    assert hip_error.max() > 0.02
    assert np.sqrt(np.mean(hip_error**2)) > 0.01
    assert "fixed_median_primary_hip" in result.sagittal_frame.origin_policy


def test_bilateral_segment_lengths_and_primary_motion_leg_are_reported() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 80.0, 61))
    q_knee = np.deg2rad(np.linspace(30.0, 110.0, 61))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, q_knee),
        coordinate_unit="m",
    )
    lengths = result.segment_lengths

    assert result.motion_leg_audit.selection_mode == "auto_motion_score"
    assert result.motion_leg_audit.right_motion_score_m > (
        result.motion_leg_audit.left_motion_score_m
    )
    assert lengths.left_thigh.median_m == pytest.approx(0.42)
    assert lengths.right_thigh.median_m == pytest.approx(0.42)
    assert lengths.left_shank_to_ankle.median_m == pytest.approx(0.36)
    assert lengths.right_shank_to_ankle.median_m == pytest.approx(0.36)


def test_observed_ankle_is_retained_and_never_aliased_to_pull_point() -> None:
    q_hip = np.deg2rad(np.linspace(20.0, 70.0, 31))
    q_knee = np.deg2rad(np.linspace(30.0, 100.0, 31))
    result = import_reference_trajectory_dataframe(
        _synthetic_bilateral_markers(q_hip, q_knee),
        coordinate_unit="m",
    )

    assert np.allclose(
        result["observed_ankle_x_local_m"], result["RAnkle_x_local_m"]
    )
    assert np.allclose(
        result["observed_ankle_z_local_m"], result["RAnkle_z_local_m"]
    )
    assert not result["observed_ankle_is_pull_point"].any()
    assert not any(
        column.lower().startswith(("x_pull", "z_pull"))
        or column.lower().endswith(("pull_x", "pull_z"))
        for column in result.trajectory
    )
    assert result.metadata["observed_ankle_is_pull_point"] is False
    assert result.metadata["pull_point_reconstruction_performed"] is False

    # Existing FK L2 is knee-to-strap pull distance, not anatomical shank
    # length.  Therefore its endpoint is intentionally different from RAnkle.
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, 0.42, 0.30)
    ankle = result[["RAnkle_x_local_m", "RAnkle_z_local_m"]].to_numpy()
    assert np.max(np.linalg.norm(ankle - np.column_stack((x_pull, z_pull)), axis=1)) > 0.05


def test_real_csv_import_has_expected_frames_right_leg_and_metric_lengths() -> None:
    result = import_reference_trajectory_csv(
        REAL_REFERENCE_CSV,
        coordinate_unit="mm",
    )

    assert len(result) == 550
    assert result.frame_audit.first_frame == 5550
    assert result.frame_audit.last_frame == 6099
    assert result.frame_audit.continuous_unit_steps
    assert result.primary_motion_leg == "right"
    assert result.segment_lengths.right_thigh.median_m == pytest.approx(
        0.37658, abs=2e-4
    )
    assert result.segment_lengths.right_shank_to_ankle.median_m == pytest.approx(
        0.35036, abs=2e-4
    )
    assert np.isfinite(
        result[["q_hip_rad", "q_knee_rad", "theta_shank_rad"]]
    ).all(axis=None)


def test_import_does_not_mutate_source_dataframe() -> None:
    dataframe = _synthetic_bilateral_markers(
        np.deg2rad(np.linspace(20.0, 70.0, 31)),
        np.deg2rad(np.linspace(30.0, 100.0, 31)),
    )
    expected = dataframe.copy(deep=True)
    import_reference_trajectory_dataframe(dataframe, coordinate_unit="m")
    assert_frame_equal(dataframe, expected)


def test_import_module_has_no_dynamics_force_or_hardware_dependency() -> None:
    import lower_limb_sim.reference_trajectory_import as module

    source = inspect.getsource(module)
    forbidden_imports = (
        "full_dynamics",
        "parameter_estimator",
        "force_mapping",
        "observation_model",
        "hardware",
        "collection",
        "control",
        "xCoreSDK",
        "rokae",
    )
    assert not any(f"import {name}" in source for name in forbidden_imports)
    assert module.import_reference_trajectory_dataframe.__module__ == (
        "lower_limb_sim.reference_trajectory_import"
    )


def _synthetic_repeated_reference_csv(path: Path) -> tuple[Path, float]:
    fps = 50.0
    phase = np.linspace(0.0, 7.0 * np.pi, 421)
    q_hip = np.deg2rad(20.0 + 40.0 * (1.0 - np.cos(phase)))
    q_knee = np.deg2rad(20.0 + 50.0 * (1.0 - np.cos(phase)))
    dataframe = _synthetic_bilateral_markers(
        q_hip,
        q_knee,
        first_frame=1000,
    )
    dataframe.to_csv(path, index=False)
    return path, fps


def test_geometry_only_run_does_not_fabricate_derivatives_or_dynamics(
    tmp_path: Path,
) -> None:
    result = run_reference_trajectory(
        REAL_REFERENCE_CSV,
        coordinate_unit="mm",
        leg="right",
        output_directory=tmp_path,
        generate_visualizations=True,
    )

    assert not result.derivatives_available
    assert not result.dynamics_available
    assert not result.dynamics_by_subject
    assert result.selected_cycle[
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].isna().all(axis=None)
    assert result.selected_cycle["derivative_reason"].eq(
        "fps_not_provided"
    ).all()
    for filename in DYNAMIC_FILENAMES.values():
        assert not (tmp_path / filename).exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["fps"] is None
    assert metadata["dynamic_gate_reason"] == "fps_not_provided"
    assert set(metadata["skipped_dynamic_files"]) == set(DYNAMIC_FILENAMES.values())
    assert "metadata.json" in metadata["generated_files"]
    assert set(result.visualization_paths) == set(GEOMETRY_FIGURES)
    assert set(result.skipped_visualizations) == set(DYNAMIC_FIGURES)
    assert all((tmp_path / filename).is_file() for filename in GEOMETRY_FIGURES)
    assert all(not (tmp_path / filename).exists() for filename in DYNAMIC_FIGURES)


def test_detected_tail_is_incomplete_and_not_default_selection(
    tmp_path: Path,
) -> None:
    result = run_reference_trajectory(
        REAL_REFERENCE_CSV,
        coordinate_unit="mm",
        leg="right",
        output_directory=tmp_path,
        generate_visualizations=False,
    )
    cycles = result.detected_cycles
    assert (~cycles["cycle_complete"]).any()
    assert not bool(cycles.iloc[-1]["cycle_complete"])
    assert result.selection.cycle_complete
    assert result.selection.cycle_index != int(cycles.iloc[-1]["cycle_index"])
    assert int(cycles["selected"].sum()) == 1
    selected_row = cycles.loc[cycles["selected"]].iloc[0]
    assert int(selected_row["cycle_index"]) == result.selection.cycle_index
    # Saved frame values are original CSV identifiers, not zero-based rows.
    assert int(cycles["start_frame"].min()) >= 5550
    assert result.metadata["selected_cycle"]["start_frame"] >= 5550
    assert result.selection.start_frame >= 5550
    assert result.selection.end_frame <= 6099


def test_selected_contract_reuses_forward_kinematics_and_keeps_ankle_separate(
    tmp_path: Path,
) -> None:
    result = run_reference_trajectory(
        REAL_REFERENCE_CSV,
        coordinate_unit="mm",
        leg="right",
        output_directory=tmp_path,
        generate_visualizations=False,
    )
    selected = result.selected_cycle
    required = {
        "Frame",
        "time_s",
        "phase",
        "cycle_phase",
        "q_hip_raw_rad",
        "q_knee_raw_rad",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "x_knee_m",
        "z_knee_m",
        "x_pull_m",
        "z_pull_m",
        "x_ankle_observed_m",
        "z_ankle_observed_m",
        "planarity_error_m",
        "joint_limit_valid",
        "trajectory_sample_valid",
        "invalid_reason",
    }
    assert required.issubset(selected.columns)
    expected = forward_kinematics(
        selected["q_hip_rad"].to_numpy(),
        selected["q_knee_rad"].to_numpy(),
        L1,
        L2,
    )
    for actual_column, expected_values in zip(
        ("x_knee_m", "z_knee_m", "x_pull_m", "z_pull_m"),
        expected,
    ):
        assert np.allclose(selected[actual_column], expected_values)
    ankle = selected[["x_ankle_observed_m", "z_ankle_observed_m"]].to_numpy()
    pull = selected[["x_pull_m", "z_pull_m"]].to_numpy()
    assert np.max(np.linalg.norm(ankle - pull, axis=1)) > 0.01
    assert np.allclose(
        selected["theta_shank_rad"],
        selected["q_hip_rad"] - selected["q_knee_rad"],
        atol=1e-15,
    )
    with np.load(tmp_path / "reference_selected_cycle.npz", allow_pickle=False) as archive:
        assert required.difference(archive.files) == set()
        assert np.array_equal(archive["Frame"], selected["Frame"].to_numpy())


def test_explicit_fps_enables_existing_derivative_and_four_subject_dynamics(
    tmp_path: Path,
) -> None:
    csv_path, fps = _synthetic_repeated_reference_csv(tmp_path / "reference.csv")
    output = tmp_path / "processed"
    result = run_reference_trajectory(
        csv_path,
        coordinate_unit="m",
        fps=fps,
        leg="right",
        cycle_index=1,
        output_directory=output,
        generate_visualizations=True,
    )

    assert result.derivatives_available
    assert result.dynamics_available
    assert set(result.dynamics_by_subject) == set(DYNAMIC_FILENAMES)
    assert set(result.visualization_paths) == set(FIGURE_FILENAMES)
    assert not result.skipped_visualizations
    assert all((output / filename).is_file() for filename in FIGURE_FILENAMES)
    valid_derivative = result.selected_cycle["derivative_valid"].astype(bool)
    assert valid_derivative.any()
    derivative_columns = [
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    ]
    assert np.isfinite(
        result.selected_cycle.loc[valid_derivative, derivative_columns]
    ).all(axis=None)
    for subject_id, filename in DYNAMIC_FILENAMES.items():
        assert (output / filename).is_file()
        dynamics = result.dynamics_by_subject[subject_id]
        assert {
            "tau_inertia_hip_nm",
            "tau_coriolis_hip_nm",
            "tau_gravity_hip_nm",
            "tau_damping_hip_nm",
            "tau_stiffness_hip_nm",
            "tau_total_hip_nm",
            "tau_total_knee_nm",
            "fx_robot_on_leg_n",
            "fz_robot_on_leg_n",
            "force_magnitude_n",
            "jacobian_condition_number",
            "force_mapping_valid",
        }.issubset(dynamics.columns)
        valid_force = dynamics["force_mapping_valid"].astype(bool)
        assert valid_force.any()
        assert np.isfinite(
            dynamics.loc[
                valid_force,
                ["fx_robot_on_leg_n", "fz_robot_on_leg_n", "force_magnitude_n"],
            ]
        ).all(axis=None)
        assert dynamics["model_angle_definition"].eq(
            "theta_shank = q_hip - q_knee"
        ).all()


def test_manual_start_end_are_source_frame_identifiers(tmp_path: Path) -> None:
    result = run_reference_trajectory(
        REAL_REFERENCE_CSV,
        coordinate_unit="mm",
        leg="right",
        start_frame=5937,
        end_frame=6040,
        output_directory=tmp_path,
        generate_visualizations=False,
    )
    assert int(result.selected_cycle["Frame"].iloc[0]) == 5937
    assert int(result.selected_cycle["Frame"].iloc[-1]) == 6040
    assert result.selection.manual_selection
    assert result.selection.start_frame == 5937
    assert result.selection.end_frame == 6040


def test_cycle_detection_never_crosses_a_missing_source_frame(tmp_path: Path) -> None:
    csv_path, _ = _synthetic_repeated_reference_csv(tmp_path / "with_gap.csv")
    dataframe = pd.read_csv(csv_path)
    missing_frame = 1120
    dataframe = dataframe.loc[dataframe["Frame"].ne(missing_frame)].copy()
    dataframe.to_csv(csv_path, index=False)
    result = run_reference_trajectory(
        csv_path,
        coordinate_unit="m",
        leg="right",
        output_directory=tmp_path / "processed_gap",
        generate_visualizations=False,
    )

    assert result.import_result.frame_audit.gap_count == 1
    spans_gap = (
        result.detected_cycles["start_frame"].lt(missing_frame)
        & result.detected_cycles["end_frame"].gt(missing_frame)
    )
    assert not spans_gap.any()


def test_dynamic_peak_summary_uses_only_in_range_dynamic_samples(
    tmp_path: Path,
) -> None:
    result = run_reference_trajectory(
        REAL_REFERENCE_CSV,
        coordinate_unit="mm",
        fps=100.0,
        leg="right",
        output_directory=tmp_path,
        generate_visualizations=False,
    )
    baseline = result.dynamics_by_subject["baseline"]
    valid = baseline["dynamic_sample_valid"].astype(bool)
    summary = result.metadata["dynamic_peak_summary"]["baseline"]
    assert summary["peak_definition"] == "dynamic_sample_valid_only"
    assert summary["dynamic_sample_valid_samples"] == int(valid.sum())
    assert summary["peak_abs_hip_torque_nm"] == pytest.approx(
        baseline.loc[valid, "tau_total_hip_nm"].abs().max()
    )
    assert summary["peak_abs_knee_torque_nm"] == pytest.approx(
        baseline.loc[valid, "tau_total_knee_nm"].abs().max()
    )
    assert summary["peak_force_magnitude_n"] == pytest.approx(
        baseline.loc[valid, "force_magnitude_n"].max()
    )


def test_runner_has_no_real_robot_control_or_sdk_imports() -> None:
    import lower_limb_sim.run_reference_trajectory as module

    source = inspect.getsource(module)
    forbidden = (
        "from hardware",
        "from control",
        "from collection",
        "from safety",
        "import rokae",
        "import xCoreSDK",
        "import hardware",
        "import control",
    )
    assert not any(token in source for token in forbidden)
