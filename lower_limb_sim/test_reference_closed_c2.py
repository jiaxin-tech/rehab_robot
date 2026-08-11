"""Regression tests for the offline ``reference_closed_c2`` candidate."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from lower_limb_sim.reference_closed_c2 import (
    APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG,
    C2_REFERENCE,
    C2ReferenceError,
    compare_c2_with_pchip,
    fit_reference_closed_c2,
    retime_reference_closed_c2,
)
from lower_limb_sim.run_reference_c2 import (
    DEFAULT_INPUT_PATH,
    PROFILE_DURATIONS_S,
    run_reference_c2,
)
from lower_limb_sim.visualize_reference_c2 import FIGURE_FILENAMES


@pytest.fixture(scope="module")
def reference_versions() -> pd.DataFrame:
    dataframe = pd.read_csv(DEFAULT_INPUT_PATH)
    closed = dataframe.loc[
        dataframe["reference_version"].eq("reference_closed_symmetric")
    ]
    assert closed["formal_execution_allowed"].astype(bool).all()
    assert closed["q_knee_approved_max_deg"].eq(145.0).all()
    assert not closed["rom_mapping_applied"].astype(bool).any()
    return dataframe


@pytest.fixture(scope="module")
def c2_model(reference_versions: pd.DataFrame):
    return fit_reference_closed_c2(reference_versions)


@pytest.fixture(scope="module")
def c2_trajectories(c2_model):
    return {
        profile: retime_reference_closed_c2(
            c2_model,
            profile=profile,
            flexion_duration_s=duration,
            extension_duration_s=duration,
            samples_per_segment=201,
        )
        for profile, duration in PROFILE_DURATIONS_S.items()
    }


@pytest.fixture(scope="module")
def comparison(reference_versions, c2_model, c2_trajectories):
    table, _ = compare_c2_with_pchip(
        reference_versions,
        c2_model,
        c2_trajectories,
        durations_s=PROFILE_DURATIONS_S,
        samples_per_segment=201,
    )
    return table


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_145_degree_approval_is_consumed_without_rom_mapping(
    c2_model,
) -> None:
    assert c2_model.approved_hip_rom_deg == APPROVED_HIP_ROM_DEG
    assert c2_model.approved_knee_rom_deg == APPROVED_KNEE_ROM_DEG
    assert c2_model.shape_audit.rom_violation_count == 0
    assert not c2_model.shape_audit.pointwise_clip_applied
    assert not c2_model.phase_path["rom_mapping_applied"].any()
    assert c2_model.phase_path["joint_limit_valid"].all()


def test_source_reference_dataframe_is_not_modified(reference_versions) -> None:
    before = reference_versions.copy(deep=True)
    fit_reference_closed_c2(reference_versions)
    assert_frame_equal(reference_versions, before, check_exact=True)


def test_c2_start_and_peak_flexion_postures_are_exactly_preserved(
    reference_versions,
    c2_model,
) -> None:
    source = reference_versions.loc[
        reference_versions["reference_version"].eq("reference_closed_symmetric")
        & reference_versions["cycle_phase"].eq("flexion")
    ].sort_values("segment_phase")
    c2 = c2_model.phase_path.loc[c2_model.phase_path["cycle_phase"].eq("flexion")]
    peak = int(np.argmax(source["q_knee_reference_rad"].to_numpy(float)))
    np.testing.assert_allclose(
        c2[["q_hip_rad", "q_knee_rad"]].iloc[0],
        source[["q_hip_reference_rad", "q_knee_reference_rad"]].iloc[0],
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        c2[["q_hip_rad", "q_knee_rad"]].iloc[peak],
        source[["q_hip_reference_rad", "q_knee_reference_rad"]].iloc[peak],
        atol=1e-14,
        rtol=0.0,
    )
    assert c2_model.shape_audit.start_pose_error_deg == pytest.approx(0.0, abs=1e-13)
    assert c2_model.shape_audit.peak_pose_error_deg == pytest.approx(0.0, abs=1e-13)


def test_c2_phase_extension_is_exact_flexion_reverse_and_is_closed(c2_model) -> None:
    flexion = c2_model.phase_path.loc[
        c2_model.phase_path["cycle_phase"].eq("flexion")
    ].reset_index(drop=True)
    extension = c2_model.phase_path.loc[
        c2_model.phase_path["cycle_phase"].eq("extension")
    ].reset_index(drop=True)
    # The shared peak is persisted only once, so extension starts with the next
    # reverse-path sample and global_phase remains strictly increasing.
    assert np.all(np.diff(c2_model.phase_path["global_phase"].to_numpy(float)) > 0.0)
    for column in ("q_hip_rad", "q_knee_rad", "x_pull_m", "z_pull_m"):
        np.testing.assert_array_equal(
            extension[column].to_numpy(float), flexion[column].to_numpy(float)[-2::-1]
        )
    for joint in ("hip", "knee"):
        np.testing.assert_allclose(
            extension[f"dq_{joint}_ds_rad"].to_numpy(float),
            -flexion[f"dq_{joint}_ds_rad"].to_numpy(float)[-2::-1],
            atol=1e-13,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            extension[f"d2q_{joint}_ds2_rad"].to_numpy(float),
            flexion[f"d2q_{joint}_ds2_rad"].to_numpy(float)[-2::-1],
            atol=1e-12,
            rtol=0.0,
        )
    np.testing.assert_array_equal(
        c2_model.phase_path["theta_shank_rad"].to_numpy(float),
        c2_model.phase_path["q_hip_rad"].to_numpy(float)
        - c2_model.phase_path["q_knee_rad"].to_numpy(float),
    )


def test_full_cycle_global_phase_is_c4_at_turning_and_loop_seams(c2_model) -> None:
    """Audit both joins, not merely the internal flexion B-spline knots."""

    tolerance = c2_model.shape_audit.full_cycle_continuity_tolerance_rad
    for spline in (c2_model.hip_spline, c2_model.knee_spline):
        for order in range(5):
            forward_scale = float(2**order)
            reverse_scale = float((-2) ** order)

            turning_left = forward_scale * float(spline(1.0, order))
            turning_right = reverse_scale * float(spline(1.0, order))
            loop_left = reverse_scale * float(spline(0.0, order))
            loop_right = forward_scale * float(spline(0.0, order))

            assert turning_right == pytest.approx(turning_left, abs=tolerance)
            assert loop_right == pytest.approx(loop_left, abs=tolerance)

    audit = c2_model.shape_audit
    assert audit.internal_spline_continuity_order == 4
    assert audit.full_cycle_global_phase_continuity_order == 4
    assert audit.reflection_boundary_derivative_orders == (1, 3)
    for joint in ("hip", "knee"):
        for order_name in ("first", "second", "third", "fourth"):
            assert (
                getattr(
                    audit,
                    f"{joint}_full_cycle_max_{order_name}_derivative_jump_rad",
                )
                <= tolerance
            )


def test_retimed_slow_and_nominal_paths_are_closed_and_spatially_identical(
    c2_trajectories,
) -> None:
    slow = c2_trajectories["slow"]
    nominal = c2_trajectories["nominal"]
    for column in ("q_hip_rad", "q_knee_rad", "x_knee_m", "z_knee_m", "x_pull_m", "z_pull_m"):
        np.testing.assert_allclose(slow[column], nominal[column], atol=1e-14, rtol=0.0)
        assert slow[column].iloc[-1] == pytest.approx(slow[column].iloc[0], abs=1e-14)
    for trajectory in c2_trajectories.values():
        flexion = trajectory.iloc[:201]
        extension = trajectory.iloc[201:]
        for column in ("q_hip_rad", "q_knee_rad", "x_pull_m", "z_pull_m"):
            np.testing.assert_allclose(
                extension[column].to_numpy(float),
                flexion[column].to_numpy(float)[-2::-1],
                atol=1e-13,
                rtol=0.0,
            )


def test_retimed_endpoints_have_zero_velocity_and_acceleration(c2_trajectories) -> None:
    columns = [
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    ]
    for trajectory in c2_trajectories.values():
        np.testing.assert_allclose(
            trajectory.iloc[[0, 200, -1]][columns], 0.0, atol=1e-12, rtol=0.0
        )
        assert trajectory["trajectory_sample_valid"].all()
        assert trajectory["formal_execution_allowed"].all()
        assert not trajectory["robot_execution_approved"].any()
        assert trajectory["formal_execution_scope"].eq(
            "offline_reference_rom_and_shape_gate_only"
        ).all()


def test_extension_chain_rule_signs_and_duration_scaling(
    c2_model,
    c2_trajectories,
) -> None:
    """Independently reconstruct q, dq and ddq on both retimed branches."""

    for trajectory in c2_trajectories.values():
        for joint, spline in (
            ("hip", c2_model.hip_spline),
            ("knee", c2_model.knee_spline),
        ):
            source_phase = trajectory["source_flexion_phase"].to_numpy(float)
            direction = np.where(trajectory["cycle_phase"].eq("flexion"), 1.0, -1.0)
            phase_rate = trajectory["minimum_jerk_phase_rate_s_inv"].to_numpy(float)
            phase_acceleration = trajectory[
                "minimum_jerk_phase_acceleration_s_inv2"
            ].to_numpy(float)
            q_s = direction * spline(source_phase, 1)
            expected_dq = q_s * phase_rate
            expected_ddq = spline(source_phase, 2) * phase_rate**2 + (
                q_s * phase_acceleration
            )
            np.testing.assert_allclose(
                trajectory[f"q_{joint}_rad"], spline(source_phase), atol=1e-14, rtol=0.0
            )
            np.testing.assert_allclose(
                trajectory[f"dq_{joint}_rad_s"], expected_dq, atol=1e-14, rtol=0.0
            )
            np.testing.assert_allclose(
                trajectory[f"ddq_{joint}_rad_s2"],
                expected_ddq,
                atol=1e-13,
                rtol=0.0,
            )

    slow = c2_trajectories["slow"]
    nominal = c2_trajectories["nominal"]
    for joint in ("hip", "knee"):
        np.testing.assert_allclose(
            nominal[f"dq_{joint}_rad_s"],
            2.0 * slow[f"dq_{joint}_rad_s"],
            atol=1e-13,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            nominal[f"ddq_{joint}_rad_s2"],
            4.0 * slow[f"ddq_{joint}_rad_s2"],
            atol=1e-12,
            rtol=0.0,
        )


def test_c2_second_derivative_is_continuous_and_acceleration_warnings_are_removed(
    c2_model,
    comparison,
) -> None:
    assert c2_model.shape_audit.continuity_order >= 2
    assert c2_model.shape_audit.hip_second_derivative_max_knot_jump_rad < 1e-9
    assert c2_model.shape_audit.knee_second_derivative_max_knot_jump_rad < 1e-9
    assert comparison["original_acceleration_warning_count"].eq(12).all()
    assert comparison["c2_acceleration_warning_count"].eq(0).all()
    assert comparison["acceleration_warning_count_reduction"].eq(12).all()


def test_c2_deviation_is_quantified_and_below_explicit_acceptance_gates(
    c2_model,
    comparison,
) -> None:
    audit = c2_model.shape_audit
    assert 0.0 < audit.hip_max_deviation_deg <= audit.maximum_hip_deviation_gate_deg
    assert 0.0 < audit.knee_max_deviation_deg <= audit.maximum_knee_deviation_gate_deg
    assert 0.0 < audit.pull_point_max_deviation_mm <= audit.maximum_pull_deviation_gate_mm
    assert audit.hip_rms_deviation_deg <= audit.hip_max_deviation_deg
    assert audit.knee_rms_deviation_deg <= audit.knee_max_deviation_deg
    assert audit.pull_point_rms_deviation_mm <= audit.pull_point_max_deviation_mm
    required = {
        "hip_max_deviation_deg",
        "knee_max_deviation_deg",
        "hip_rms_deviation_deg",
        "knee_rms_deviation_deg",
        "pull_point_max_deviation_mm",
        "pull_point_rms_deviation_mm",
        "original_acceleration_warning_count",
        "c2_acceleration_warning_count",
    }
    assert required.issubset(comparison.columns)


def test_significant_shape_change_is_rejected_fail_closed(reference_versions) -> None:
    with pytest.raises(C2ReferenceError, match="shape_deviation_exceeded"):
        fit_reference_closed_c2(
            reference_versions,
            maximum_hip_deviation_deg=0.01,
            maximum_knee_deviation_deg=0.01,
            maximum_pull_deviation_mm=0.05,
        )


def test_unapproved_or_mapped_source_is_rejected(reference_versions) -> None:
    unapproved = reference_versions.copy(deep=True)
    closed = unapproved["reference_version"].eq("reference_closed_symmetric")
    unapproved.loc[closed, "formal_execution_allowed"] = False
    with pytest.raises(C2ReferenceError, match="not formally ROM-approved"):
        fit_reference_closed_c2(unapproved)

    mapped = reference_versions.copy(deep=True)
    mapped.loc[closed, "rom_mapping_applied"] = True
    with pytest.raises(C2ReferenceError, match="ROM-mapped source"):
        fit_reference_closed_c2(mapped)


def test_runner_writes_four_tables_three_figures_and_does_not_overwrite_source(
    tmp_path: Path,
) -> None:
    source_hash = _sha256(DEFAULT_INPUT_PATH)
    result = run_reference_c2(output_directory=tmp_path)
    assert _sha256(DEFAULT_INPUT_PATH) == source_hash
    expected_tables = {
        "reference_closed_c2_phase.csv",
        "reference_closed_c2_slow.csv",
        "reference_closed_c2_nominal.csv",
        "reference_c2_comparison.csv",
        "reference_c2_metadata.json",
    }
    assert expected_tables.issubset(result.output_paths)
    assert set(FIGURE_FILENAMES).issubset(result.output_paths)
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.output_paths.values())
    metadata = json.loads((tmp_path / "reference_c2_metadata.json").read_text())
    assert metadata["approved_hip_rom_deg"] == [0.0, 120.0]
    assert metadata["approved_knee_rom_deg"] == [5.0, 145.0]
    assert metadata["rom_mapping_applied"] is False
    assert metadata["source_reference_overwritten"] is False
    assert metadata["shape_preserved_within_audit_gates"] is True
    assert metadata["model_version"].endswith("_v2")
    assert metadata["internal_spline_continuity_order"] == 4
    assert metadata["full_cycle_global_phase_continuity_order"] == 4
    assert metadata["reflection_boundary_conditions"]["flexion_start"] == {
        "derivative_order_1": 0.0,
        "derivative_order_3": 0.0,
    }
    assert metadata["acceleration_warning_is_sampling_resolution_dependent"] is True
    assert metadata["formal_execution_scope"] == (
        "offline_reference_rom_and_shape_gate_only"
    )
    assert "immutable_source_retained" in metadata["reference_path_preserved_meaning"]
    assert metadata["robot_execution_approved"] is False


def test_runner_refuses_real_robot_and_shared_configuration_output_roots() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    for protected_name in ("hardware", "control", "collection", "safety", "config", "scripts"):
        with pytest.raises(ValueError, match="protected directory"):
            run_reference_c2(
                output_directory=repository_root / protected_name / "c2-forbidden",
                save_outputs=False,
                generate_plots=False,
            )


def test_c2_source_has_no_pointwise_clip_or_real_robot_imports() -> None:
    package_dir = Path(__file__).resolve().parent
    forbidden = {"hardware", "control", "collection", "safety", "sdk", "xcoresdk", "rokae"}
    for filename in (
        "reference_closed_c2.py",
        "run_reference_c2.py",
        "visualize_reference_c2.py",
    ):
        tree = ast.parse((package_dir / filename).read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            if isinstance(node, ast.Call):
                is_clip = (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "clip"
                ) or (isinstance(node.func, ast.Name) and node.func.id == "clip")
                assert not is_clip, f"pointwise clip found in {filename}"
        for imported in imports:
            assert set(imported.lower().split(".")).isdisjoint(forbidden)
