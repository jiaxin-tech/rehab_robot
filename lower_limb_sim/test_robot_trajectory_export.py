"""Stage 6A offline H/B/T transform and command-file regression tests.

Every calibration in this module is synthetic and scoped to ``tmp_path`` or
an in-memory dataframe.  The tests never import or touch robot hardware,
control, collection, safety, or SDK code.
"""

from __future__ import annotations

import ast
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.robot_coordinate_transform import (
    MODEL_ANGLE_DEFINITION,
    RobotFrameCalibration,
    TcpOrientation,
    calibration_from_mapping,
    human_pull_points_to_base,
    load_calibration_json,
    pull_points_base_to_tcp_origins,
    tcp_origins_to_pull_points_base,
)
from lower_limb_sim.robot_trajectory_audit import (
    REQUIRED_COMMAND_COLUMNS,
    audit_robot_trajectory,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
)
from lower_limb_sim.run_robot_trajectory_export import (
    COMMAND_FILENAME,
    DEFAULT_REFERENCE_PATH,
    METADATA_FILENAME,
    PROJECT_ROOT,
    STAGE5C_PCHIP_REFERENCE_PATH,
    build_robot_trajectory,
    dry_run_robot_trajectory,
    load_closed_reference_trajectory,
    run_robot_trajectory_export,
)
from lower_limb_sim.visualize_robot_trajectory import FIGURE_FILENAMES


STAGE6A_SOURCE_FILES = (
    "robot_coordinate_transform.py",
    "robot_trajectory_audit.py",
    "run_robot_trajectory_export.py",
    "visualize_robot_trajectory.py",
)


def _synthetic_reference(sample_count: int = 201) -> pd.DataFrame:
    """Return a smooth, closed, FK-consistent reference with explicit approval."""

    time_s = np.linspace(0.0, 2.0, sample_count)
    q_hip = 0.70 + 0.10 * np.sin(np.pi * time_s)
    q_knee = 1.00 + 0.15 * np.sin(np.pi * time_s)
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "theta_shank_rad": q_hip - q_knee,
            "x_pull_human_m": x_pull,
            "z_pull_human_m": z_pull,
            "trajectory_sample_valid": True,
            "formal_execution_allowed": True,
            "invalid_reason": "",
            "retimed_timing_is_original": False,
            "approved_hip_min_deg": 0.0,
            "approved_hip_max_deg": 120.0,
            "approved_knee_min_deg": 5.0,
            "approved_knee_max_deg": 145.0,
        }
    )


def _identity_calibration(
    *,
    hip_center: tuple[float, float, float] = (1.0, 2.0, 3.0),
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    representation: str = "euler_xyz_rad",
    approved_hip_rom: tuple[float, float] = (0.0, 120.0),
    approved_knee_rom: tuple[float, float] = (5.0, 145.0),
) -> RobotFrameCalibration:
    return RobotFrameCalibration(
        hip_center_in_base_m=hip_center,
        human_x_axis_in_base=(1.0, 0.0, 0.0),
        human_z_axis_in_base=(0.0, 0.0, 1.0),
        tool_offset_m=tool_offset,
        tcp_orientation=TcpOrientation(representation, orientation),
        approved_hip_rom_deg=approved_hip_rom,
        approved_knee_rom_deg=approved_knee_rom,
        reviewed=True,
    )


def _write_reference(tmp_path: Path, dataframe: pd.DataFrame | None = None) -> Path:
    path = tmp_path / "closed_reference.csv"
    (dataframe if dataframe is not None else _synthetic_reference()).to_csv(
        path, index=False
    )
    return path


def test_calibration_is_explicit_right_handed_and_rejects_bad_axes() -> None:
    calibration = RobotFrameCalibration(
        hip_center_in_base_m=(0.4, -0.2, 0.7),
        human_x_axis_in_base=(0.0, 1.0, 0.0),
        human_z_axis_in_base=(0.0, 0.0, 1.0),
        tool_offset_m=(0.0, 0.0, 0.0),
        tcp_orientation=TcpOrientation("rotation_vector_rad", (0.0, 0.0, 0.0)),
        approved_hip_rom_deg=(0.0, 120.0),
        approved_knee_rom_deg=(5.0, 145.0),
        reviewed=True,
    )
    expected = np.column_stack(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    np.testing.assert_allclose(calibration.rotation_base_from_human, expected)
    np.testing.assert_allclose(
        calibration.rotation_base_from_human.T
        @ calibration.rotation_base_from_human,
        np.eye(3),
        atol=1e-15,
    )
    assert np.linalg.det(calibration.rotation_base_from_human) == pytest.approx(1.0)
    assert calibration.transform_is_orthogonal

    with pytest.raises(ValueError, match="unit vectors"):
        RobotFrameCalibration(
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            TcpOrientation("euler_xyz_rad", (0.0, 0.0, 0.0)),
            (0.0, 120.0),
            (5.0, 145.0),
            reviewed=True,
        )
    with pytest.raises(ValueError, match="orthogonal"):
        RobotFrameCalibration(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            TcpOrientation("euler_xyz_rad", (0.0, 0.0, 0.0)),
            (0.0, 120.0),
            (5.0, 145.0),
            reviewed=True,
        )


def test_calibration_mapping_has_no_implicit_laboratory_defaults() -> None:
    complete = {
        "hip_center_in_base_m": [0.4, -0.2, 0.7],
        "human_x_axis_in_base": [1.0, 0.0, 0.0],
        "human_z_axis_in_base": [0.0, 0.0, 1.0],
        "tool_offset_m": [0.02, 0.0, -0.01],
        "tcp_orientation": {
            "representation": "rotation_vector_rad",
            "values_rad": [0.0, 0.0, 0.0],
        },
        "approved_hip_rom_deg": [0.0, 120.0],
        "approved_knee_rom_deg": [5.0, 145.0],
        "reviewed": True,
        "notes": "synthetic unit test only",
    }
    assert calibration_from_mapping(complete).hip_center_in_base_m == (0.4, -0.2, 0.7)
    for field in complete:
        incomplete = dict(complete)
        incomplete.pop(field)
        with pytest.raises(ValueError, match="missing required fields"):
            calibration_from_mapping(incomplete)


@pytest.mark.parametrize("reviewed_value", [False, 1, "true", None])
def test_unreviewed_or_truthy_calibration_cannot_enable_export(
    reviewed_value: object,
) -> None:
    mapping = {
        "hip_center_in_base_m": [0.4, -0.2, 0.7],
        "human_x_axis_in_base": [1.0, 0.0, 0.0],
        "human_z_axis_in_base": [0.0, 0.0, 1.0],
        "tool_offset_m": [0.02, 0.0, -0.01],
        "tcp_orientation": {
            "representation": "euler_xyz_rad",
            "values_rad": [0.0, 0.0, 0.0],
        },
        "approved_hip_rom_deg": [0.0, 120.0],
        "approved_knee_rom_deg": [5.0, 145.0],
        "reviewed": reviewed_value,
        "notes": "synthetic unit test only",
    }
    with pytest.raises(ValueError, match="reviewed must be the boolean true"):
        calibration_from_mapping(mapping)


def test_repository_calibration_template_contains_no_fake_values_and_is_blocked() -> None:
    template_path = (
        Path(__file__).resolve().parent
        / "config"
        / "robot_rehab_calibration_template.json"
    )
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    for field in (
        "hip_center_in_base_m",
        "human_x_axis_in_base",
        "human_z_axis_in_base",
        "tool_offset_m",
        "tcp_orientation",
    ):
        assert payload[field] is None
    assert payload["approved_hip_rom_deg"] == [0, 120]
    assert payload["approved_knee_rom_deg"] == [5, 145]
    assert payload["reviewed"] is False
    with pytest.raises(ValueError, match="reviewed must be the boolean true"):
        load_calibration_json(template_path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("hip_center_in_base_m", [np.nan, 0.0, 0.0]),
        ("human_x_axis_in_base", [np.inf, 0.0, 0.0]),
        ("tool_offset_m", [0.0, -np.inf, 0.0]),
        ("approved_knee_rom_deg", [5.0, np.nan]),
    ],
)
def test_reviewed_calibration_still_rejects_nonfinite_fields(
    field: str,
    invalid_value: object,
) -> None:
    mapping: dict[str, object] = {
        "hip_center_in_base_m": [0.4, -0.2, 0.7],
        "human_x_axis_in_base": [1.0, 0.0, 0.0],
        "human_z_axis_in_base": [0.0, 0.0, 1.0],
        "tool_offset_m": [0.02, 0.0, -0.01],
        "tcp_orientation": {
            "representation": "euler_xyz_rad",
            "values_rad": [0.0, 0.0, 0.0],
        },
        "approved_hip_rom_deg": [0.0, 120.0],
        "approved_knee_rom_deg": [5.0, 145.0],
        "reviewed": True,
        "notes": "synthetic unit test only",
    }
    mapping[field] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        calibration_from_mapping(mapping)


def test_visualization_labels_use_the_same_positive_human_axis_formula() -> None:
    source = (
        Path(__file__).resolve().parent / "visualize_robot_trajectory.py"
    ).read_text(encoding="utf-8")
    assert "hip + R_BH p_H" in source
    assert "hip_B + R_BH [x_H, 0, z_H]" in source
    assert "hip − R_BH p_H" not in source
    assert "hip_B − R_BH [x_H, 0, z_H]" not in source


def test_orthogonality_tolerance_cannot_be_inflated_to_bypass_validation() -> None:
    with pytest.raises(ValueError, match="orthogonality_tolerance"):
        RobotFrameCalibration(
            hip_center_in_base_m=(0.0, 0.0, 0.0),
            human_x_axis_in_base=(1.0, 0.0, 0.0),
            human_z_axis_in_base=(1.0, 0.0, 0.0),
            tool_offset_m=(0.0, 0.0, 0.0),
            tcp_orientation=TcpOrientation("euler_xyz_rad", (0.0, 0.0, 0.0)),
            approved_hip_rom_deg=(0.0, 120.0),
            approved_knee_rom_deg=(5.0, 145.0),
            reviewed=True,
            orthogonality_tolerance=100.0,
        )

    mapping = {
        "hip_center_in_base_m": [0.0, 0.0, 0.0],
        "human_x_axis_in_base": [1.0, 0.0, 0.0],
        "human_z_axis_in_base": [1.0, 0.0, 0.0],
        "tool_offset_m": [0.0, 0.0, 0.0],
        "tcp_orientation": {
            "representation": "euler_xyz_rad",
            "values_rad": [0.0, 0.0, 0.0],
        },
        "approved_hip_rom_deg": [0.0, 120.0],
        "approved_knee_rom_deg": [5.0, 145.0],
        "reviewed": True,
        "notes": "synthetic unit test only",
        "orthogonality_tolerance": 100.0,
    }
    with pytest.raises(ValueError, match="orthogonality_tolerance"):
        calibration_from_mapping(mapping)


def test_human_to_base_transform_maps_positive_human_axes_in_same_direction() -> None:
    calibration = _identity_calibration()
    x_pull = np.asarray((0.20, 0.35))
    z_pull = np.asarray((0.40, 0.10))
    actual = human_pull_points_to_base(x_pull, z_pull, calibration)
    expected = np.asarray(((1.20, 2.0, 3.40), (1.35, 2.0, 3.10)))
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.0)

    origin = human_pull_points_to_base(0.0, 0.0, calibration)
    plus_x = human_pull_points_to_base(0.125, 0.0, calibration)
    plus_z = human_pull_points_to_base(0.0, 0.075, calibration)
    np.testing.assert_allclose(
        plus_x - origin,
        0.125 * np.asarray(calibration.human_x_axis_in_base),
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        plus_z - origin,
        0.075 * np.asarray(calibration.human_z_axis_in_base),
        atol=1e-15,
        rtol=0.0,
    )

    rotated = RobotFrameCalibration(
        hip_center_in_base_m=(0.4, -0.2, 0.7),
        human_x_axis_in_base=(0.0, 1.0, 0.0),
        human_z_axis_in_base=(0.0, 0.0, 1.0),
        tool_offset_m=(0.0, 0.0, 0.0),
        tcp_orientation=TcpOrientation("euler_xyz_rad", (0.0, 0.0, 0.0)),
        approved_hip_rom_deg=(0.0, 120.0),
        approved_knee_rom_deg=(5.0, 145.0),
        reviewed=True,
    )
    rotated_origin = human_pull_points_to_base(0.0, 0.0, rotated)
    np.testing.assert_allclose(
        human_pull_points_to_base(0.2, 0.0, rotated) - rotated_origin,
        0.2 * np.asarray(rotated.human_x_axis_in_base),
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        human_pull_points_to_base(0.0, 0.3, rotated) - rotated_origin,
        0.3 * np.asarray(rotated.human_z_axis_in_base),
        atol=1e-15,
        rtol=0.0,
    )


def test_tool_offset_is_expressed_in_tcp_frame_and_rotated_into_base() -> None:
    calibration = _identity_calibration(
        tool_offset=(0.10, 0.0, 0.0),
        orientation=(0.0, 0.0, np.pi / 2.0),
    )
    pull_base = np.asarray(((0.8, 2.0, 2.6), (0.7, 2.0, 2.5)))
    tcp_base = pull_points_base_to_tcp_origins(pull_base, calibration)
    expected_tcp = pull_base - np.asarray((0.0, 0.10, 0.0))
    np.testing.assert_allclose(tcp_base, expected_tcp, atol=2e-15, rtol=0.0)
    np.testing.assert_allclose(
        tcp_origins_to_pull_points_base(tcp_base, calibration),
        pull_base,
        atol=2e-15,
        rtol=0.0,
    )
    assert not np.allclose(tcp_base, pull_base - np.asarray((0.10, 0.0, 0.0)))


def test_command_schema_angle_convention_and_rigid_transform_metrics() -> None:
    reference = _synthetic_reference()
    calibration = _identity_calibration(
        hip_center=(0.5, 0.1, 0.4),
        tool_offset=(0.02, -0.01, 0.03),
        orientation=(0.0, 0.0, 0.3),
    )
    command, audit, transform = build_robot_trajectory(reference, calibration)
    required = {
        *REQUIRED_COMMAND_COLUMNS,
        "trajectory_valid",
        "invalid_reason",
        "theta_shank_rad",
        "tcp_vx_base_m_s",
        "tcp_vy_base_m_s",
        "tcp_vz_base_m_s",
        "tcp_ax_base_m_s2",
        "tcp_ay_base_m_s2",
        "tcp_az_base_m_s2",
    }
    assert required.issubset(command.columns)
    np.testing.assert_allclose(
        command["theta_shank_rad"],
        command["q_hip_rad"] - command["q_knee_rad"],
        atol=1e-14,
        rtol=0.0,
    )
    assert command["model_angle_definition"].eq(MODEL_ANGLE_DEFINITION).all()
    assert command["trajectory_valid"].all()
    assert audit.all_samples_finite
    assert audit.position_continuous
    assert audit.velocity_continuous
    assert audit.acceleration_continuous
    assert not audit.obvious_single_frame_jump_detected
    assert audit.start_end_closed
    assert audit.start_end_closure_error_m == pytest.approx(0.0, abs=1e-14)
    assert audit.tool_offset_correctly_applied
    assert audit.maximum_tool_offset_reconstruction_error_m <= 2e-16
    assert audit.safety_thresholds_applied is False
    assert transform["all_pull_points_consistent_with_forward_kinematics"] is True

    time_s = reference["time_s"].to_numpy(float)
    human_path = np.column_stack(
        (
            reference["x_pull_human_m"].to_numpy(float),
            np.zeros(len(reference)),
            reference["z_pull_human_m"].to_numpy(float),
        )
    )
    human_velocity = np.column_stack(
        [np.gradient(human_path[:, axis], time_s, edge_order=2) for axis in range(3)]
    )
    human_acceleration = np.column_stack(
        [np.gradient(human_velocity[:, axis], time_s, edge_order=2) for axis in range(3)]
    )
    assert audit.maximum_cartesian_speed_m_s == pytest.approx(
        np.linalg.norm(human_velocity, axis=1).max(), rel=1e-12
    )
    assert audit.maximum_cartesian_acceleration_m_s2 == pytest.approx(
        np.linalg.norm(human_acceleration, axis=1).max(), rel=1e-12
    )


def test_source_rom_invalid_sample_is_preserved_in_command_validity() -> None:
    reference = _synthetic_reference()
    middle = len(reference) // 2
    reference.loc[middle, "trajectory_sample_valid"] = False
    reference.loc[middle, "invalid_reason"] = "outside_active_rom"
    command, audit, _ = build_robot_trajectory(reference, _identity_calibration())
    assert not bool(command.loc[middle, "trajectory_valid"])
    assert "outside_active_rom" in str(command.loc[middle, "invalid_reason"])
    assert audit.invalid_sample_count >= 1


def test_missing_or_false_source_formal_gate_cannot_become_valid_after_transform() -> None:
    reference = _synthetic_reference()
    reference["formal_execution_allowed"] = False
    command, audit, transform = build_robot_trajectory(reference, _identity_calibration())
    assert not command["trajectory_valid"].any()
    assert command["invalid_reason"].str.contains(
        "source_reference_formal_gate_not_approved"
    ).all()
    assert not audit.trajectory_all_samples_valid
    assert transform["source_formal_execution_allowed_all"] is False

    missing_gate = _synthetic_reference().drop(columns="formal_execution_allowed")
    missing_command, _, _ = build_robot_trajectory(
        missing_gate, _identity_calibration()
    )
    assert not missing_command["trajectory_valid"].any()


def test_missing_source_sample_validity_is_fail_closed_even_with_formal_gate() -> None:
    reference = _synthetic_reference().drop(columns="trajectory_sample_valid")
    assert reference["formal_execution_allowed"].all()
    command, audit, transform = build_robot_trajectory(
        reference, _identity_calibration()
    )
    assert not command["source_trajectory_valid"].any()
    assert not command["trajectory_valid"].any()
    assert command["invalid_reason"].str.contains(
        "source_sample_validity_missing"
    ).all()
    assert command["invalid_reason"].str.contains(
        "source_reference_sample_invalid"
    ).all()
    assert audit.trajectory_all_samples_valid is False
    assert transform["source_sample_validity_field_present"] is False


def test_source_boolean_strings_are_parsed_strictly_not_by_python_truthiness() -> None:
    reference = _synthetic_reference()
    reference["trajectory_sample_valid"] = "False"
    reference["formal_execution_allowed"] = "False"
    command, _, transform = build_robot_trajectory(
        reference, _identity_calibration()
    )
    assert not command["source_trajectory_valid"].any()
    assert not command["source_reference_formal_execution_allowed"].any()
    assert not command["trajectory_valid"].any()
    assert transform["source_formal_execution_allowed_all"] is False

    invalid_encoding = _synthetic_reference()
    invalid_encoding["trajectory_sample_valid"] = "not_a_boolean"
    invalid_command, _, _ = build_robot_trajectory(
        invalid_encoding, _identity_calibration()
    )
    assert not invalid_command["source_trajectory_valid"].any()
    assert invalid_command["invalid_reason"].str.contains(
        "source_sample_validity_encoding_invalid"
    ).all()


def test_fk_inconsistent_pull_point_is_reported_not_silently_transformed() -> None:
    reference = _synthetic_reference()
    middle = len(reference) // 2
    reference.loc[middle, "x_pull_human_m"] += 0.01
    command, _, transform = build_robot_trajectory(reference, _identity_calibration())
    assert not bool(command.loc[middle, "trajectory_valid"])
    assert "human_pull_point_inconsistent" in str(
        command.loc[middle, "invalid_reason"]
    )
    assert transform["all_pull_points_consistent_with_forward_kinematics"] is False


def test_source_and_calibration_approved_rom_must_match_exactly() -> None:
    reference = _synthetic_reference()
    calibration = _identity_calibration(approved_knee_rom=(5.0, 130.0))
    command, audit, transform = build_robot_trajectory(reference, calibration)

    assert transform["source_approved_knee_rom_deg"] == [5.0, 145.0]
    assert transform["calibration_approved_knee_rom_deg"] == [5.0, 130.0]
    assert transform["source_calibration_knee_rom_exact_match"] is False
    assert not command["trajectory_valid"].any()
    assert command["invalid_reason"].str.contains(
        "source_approved_knee_rom_mismatch_with_calibration"
    ).all()
    assert audit.trajectory_all_samples_valid is False


def test_calibration_rom_is_enforced_per_sample_not_only_saved_as_metadata() -> None:
    reference = _synthetic_reference()
    middle = len(reference) // 2
    reference.loc[middle, "q_knee_rad"] = np.deg2rad(146.0)
    q_hip = reference["q_hip_rad"].to_numpy(float)
    q_knee = reference["q_knee_rad"].to_numpy(float)
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    reference["x_pull_human_m"] = x_pull
    reference["z_pull_human_m"] = z_pull

    command, audit, transform = build_robot_trajectory(
        reference, _identity_calibration()
    )
    assert transform["knee_rom_violation_sample_count"] == 1
    assert not bool(command.loc[middle, "trajectory_valid"])
    assert "knee_outside_calibration_approved_rom" in str(
        command.loc[middle, "invalid_reason"]
    )
    assert audit.invalid_sample_count >= 1


def test_loader_normalizes_stage5c_and_c2_source_rom_aliases(tmp_path: Path) -> None:
    c2_path = _write_reference(tmp_path)
    c2, c2_metadata = load_closed_reference_trajectory(c2_path)
    assert c2_metadata["source_approved_hip_rom_deg"] == [0.0, 120.0]
    assert c2_metadata["source_approved_knee_rom_deg"] == [5.0, 145.0]
    assert c2["source_approved_knee_max_deg"].eq(145.0).all()

    stage5 = _synthetic_reference().rename(
        columns={
            "approved_hip_min_deg": "q_hip_approved_min_deg",
            "approved_hip_max_deg": "q_hip_approved_max_deg",
            "approved_knee_min_deg": "q_knee_approved_min_deg",
            "approved_knee_max_deg": "q_knee_approved_max_deg",
        }
    )
    stage5_path = tmp_path / "stage5_aliases.csv"
    stage5.to_csv(stage5_path, index=False)
    normalized, metadata = load_closed_reference_trajectory(stage5_path)
    assert metadata["source_approved_hip_rom_deg"] == [0.0, 120.0]
    assert metadata["source_approved_knee_rom_deg"] == [5.0, 145.0]
    assert normalized["source_approved_hip_min_deg"].eq(0.0).all()


def test_loader_rejects_missing_or_conflicting_source_rom_approval(
    tmp_path: Path,
) -> None:
    missing = _synthetic_reference().drop(columns="approved_knee_max_deg")
    with pytest.raises(ValueError, match="missing explicit source ROM approval"):
        load_closed_reference_trajectory(_write_reference(tmp_path, missing))

    conflict = _synthetic_reference()
    conflict["source_approved_knee_max_deg"] = 130.0
    with pytest.raises(ValueError, match="conflicting source ROM approval aliases"):
        load_closed_reference_trajectory(_write_reference(tmp_path, conflict))


def test_nonfinite_sample_never_leaves_a_valid_row_with_nonfinite_derived_values() -> None:
    command, _, _ = build_robot_trajectory(
        _synthetic_reference(), _identity_calibration()
    )
    command.loc[len(command) // 2, "tcp_x_base_m"] = np.nan
    audited, audit = audit_robot_trajectory(command, _identity_calibration())
    assert not audit.all_samples_finite
    assert not audit.trajectory_all_samples_valid
    numeric_outputs = [
        *REQUIRED_COMMAND_COLUMNS,
        "tcp_vx_base_m_s",
        "tcp_vy_base_m_s",
        "tcp_vz_base_m_s",
        "tcp_ax_base_m_s2",
        "tcp_ay_base_m_s2",
        "tcp_az_base_m_s2",
        "tcp_speed_m_s",
        "tcp_acceleration_m_s2",
    ]
    valid_numeric = audited.loc[
        audited["trajectory_valid"].astype(bool), numeric_outputs
    ].to_numpy(float)
    assert np.isfinite(valid_numeric).all()


def test_single_frame_position_jump_is_detected_and_marked_invalid() -> None:
    command, _, _ = build_robot_trajectory(
        _synthetic_reference(), _identity_calibration()
    )
    middle = len(command) // 2
    command.loc[middle, "tcp_x_base_m"] += 1.0
    command.loc[middle, "pull_x_base_m"] += 1.0
    audited, audit = audit_robot_trajectory(command, _identity_calibration())
    assert audit.obvious_single_frame_jump_detected
    assert not audit.position_continuous
    assert audit.position_jump_sample_count >= 1
    assert audited["invalid_reason"].str.contains("single_frame_position_jump").any()


def test_loader_rejects_bad_time_or_additive_shank_convention(tmp_path: Path) -> None:
    wrong_theta = _synthetic_reference()
    wrong_theta["theta_shank_rad"] = (
        wrong_theta["q_hip_rad"] + wrong_theta["q_knee_rad"]
    )
    with pytest.raises(ValueError, match="theta_shank"):
        load_closed_reference_trajectory(_write_reference(tmp_path, wrong_theta))

    duplicate_time = _synthetic_reference()
    duplicate_time.loc[10, "time_s"] = duplicate_time.loc[9, "time_s"]
    duplicate_path = tmp_path / "duplicate_time.csv"
    duplicate_time.to_csv(duplicate_path, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_closed_reference_trajectory(duplicate_path)


def test_mixed_phase_formal_gate_retimes_to_globally_blocked_reference(
    tmp_path: Path,
) -> None:
    phase_versions = pd.read_csv(STAGE5C_PCHIP_REFERENCE_PATH)
    closed_mask = phase_versions["reference_version"].eq(
        "reference_closed_symmetric"
    )
    closed_indices = phase_versions.index[closed_mask]
    assert len(closed_indices) > 2
    phase_versions.loc[closed_mask, "formal_execution_allowed"] = False
    phase_versions.loc[closed_indices[0], "formal_execution_allowed"] = True
    path = tmp_path / "mixed_formal_phase_reference.csv"
    phase_versions.to_csv(path, index=False)

    retimed, metadata = load_closed_reference_trajectory(
        path, samples_per_segment=31
    )
    assert metadata["stage6a_retimed_from_phase"] is True
    assert metadata["source_formal_execution_allowed_all"] is False
    assert not retimed["formal_execution_allowed"].any()


def test_missing_calibration_is_fail_closed_and_writes_metadata_only(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "blocked"
    result = run_robot_trajectory_export(
        input_reference=_write_reference(tmp_path),
        calibration=None,
        output_directory=output_directory,
        generate_plots=True,
    )
    assert result.blocked
    assert result.block_reasons == ("explicit_H_B_T_calibration_missing",)
    assert result.trajectory.empty
    assert result.audit is None
    assert "reference_robot_trajectory" not in result.output_paths
    assert not (output_directory / COMMAND_FILENAME).exists()
    assert (output_directory / METADATA_FILENAME).is_file()
    assert not any((output_directory / filename).exists() for filename in FIGURE_FILENAMES)
    metadata = json.loads((output_directory / METADATA_FILENAME).read_text())
    assert metadata["robot_execution_approved"] is False
    assert metadata["trajectory_generated_offline_only"] is True
    assert metadata["calibration"] is None
    assert metadata["calibration_reviewed"] is False
    assert metadata["laboratory_coordinates_hardcoded"] is False
    assert metadata["reference_robot_trajectory_csv_generated"] is False


def test_export_cannot_use_reviewed_calibration_with_mismatched_rom_approval(
    tmp_path: Path,
) -> None:
    result = run_robot_trajectory_export(
        input_reference=_write_reference(tmp_path),
        calibration=_identity_calibration(approved_knee_rom=(5.0, 130.0)),
        output_directory=tmp_path / "mismatched_approval",
        save_outputs=False,
        generate_plots=False,
    )
    assert result.blocked
    assert "source_calibration_knee_rom_approval_mismatch" in result.block_reasons
    assert "source_reference_contains_invalid_samples" in result.block_reasons
    assert result.metadata["preexecution_audit_passed"] is False
    assert result.metadata["transform_audit"][
        "source_calibration_knee_rom_exact_match"
    ] is False
    assert not result.trajectory["trajectory_valid"].any()


def test_explicit_synthetic_calibration_generates_csv_metadata_and_three_previews(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "generated"
    result = run_robot_trajectory_export(
        input_reference=_write_reference(tmp_path),
        calibration=_identity_calibration(
            hip_center=(0.5, 0.1, 0.4),
            tool_offset=(0.02, -0.01, 0.03),
            orientation=(0.0, 0.0, 0.3),
        ),
        calibration_source="synthetic_test_only",
        output_directory=output_directory,
        generate_plots=True,
    )
    assert not result.blocked
    assert result.audit is not None
    assert result.audit.trajectory_all_samples_valid
    assert result.metadata["robot_execution_approved"] is False
    assert result.metadata["trajectory_generated_offline_only"] is True
    assert result.metadata["real_robot_safety_thresholds_configured"] is False
    assert result.metadata["reported_extrema_are_safety_limits"] is False
    assert result.metadata["robot_sdk_imported"] is False
    assert result.metadata["robot_connection_attempted"] is False
    assert result.metadata["robot_servo_power_or_motion_command_sent"] is False
    assert result.metadata["calibration_reviewed"] is True
    assert result.metadata["human_to_base_formula"].startswith(
        "p_pull_B = hip_center_B +"
    )
    assert result.metadata["tool_offset_definition"] == (
        "vector_from_tcp_origin_to_actual_strap_connection_pull_point_expressed_in_T"
    )
    assert (output_directory / COMMAND_FILENAME).is_file()
    assert (output_directory / METADATA_FILENAME).is_file()
    exported = pd.read_csv(output_directory / COMMAND_FILENAME)
    assert {*REQUIRED_COMMAND_COLUMNS, "trajectory_valid", "invalid_reason"}.issubset(
        exported.columns
    )
    assert set(result.visualization_paths) == set(FIGURE_FILENAMES)
    assert result.skipped_visualizations == {}
    for filename in FIGURE_FILENAMES:
        path = output_directory / filename
        assert path.is_file()
        assert path.stat().st_size > 1_000


def test_current_measured_asymmetric_reference_transforms_without_jump_flags(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "current_stage5c_audit"
    result = run_robot_trajectory_export(
        input_reference=DEFAULT_REFERENCE_PATH,
        calibration=_identity_calibration(),
        calibration_source="synthetic_transform_test_only_not_experiment_approval",
        output_directory=output_directory,
        generate_plots=False,
    )
    assert not result.blocked
    assert result.audit is not None
    assert result.audit.position_jump_sample_count == 0
    assert result.audit.velocity_jump_sample_count == 0
    assert result.audit.acceleration_jump_sample_count == 0
    assert result.block_reasons == ()
    assert result.metadata["preexecution_audit_passed"] is True
    assert result.metadata["robot_execution_approved"] is False
    assert result.metadata["trajectory_generated_offline_only"] is True
    assert (output_directory / COMMAND_FILENAME).is_file()
    saved = pd.read_csv(output_directory / COMMAND_FILENAME)
    assert len(saved) == 401
    assert saved["trajectory_valid"].all()
    assert result.metadata["source_reference"]["reference_version"] == (
        MEASURED_ASYMMETRIC_CLOSED_REFERENCE
    )
    assert result.metadata["source_reference"]["stage6a_retimed_from_phase"] is False


def test_export_rejects_rotation_vector_command_orientation(tmp_path: Path) -> None:
    rotation_vector_calibration = _identity_calibration(
        representation="rotation_vector_rad"
    )
    with pytest.raises(ValueError, match="requires tcp_orientation representation"):
        run_robot_trajectory_export(
            input_reference=_write_reference(tmp_path),
            calibration=rotation_vector_calibration,
            output_directory=tmp_path / "must_not_be_created",
            save_outputs=False,
            generate_plots=False,
        )


@pytest.mark.parametrize(
    "protected_destination",
    (
        PROJECT_ROOT / "hardware",
        PROJECT_ROOT / "hardware" / "stage6a_must_not_write_here",
    ),
)
def test_export_refuses_hardware_directory_and_descendants(
    tmp_path: Path,
    protected_destination: Path,
) -> None:
    child = PROJECT_ROOT / "hardware" / "stage6a_must_not_write_here"
    assert not child.exists()
    with pytest.raises(ValueError, match="protected robot/configuration directory"):
        run_robot_trajectory_export(
            input_reference=_write_reference(tmp_path),
            calibration=None,
            output_directory=protected_destination,
            save_outputs=True,
            generate_plots=False,
        )
    assert not child.exists()


def test_dry_run_only_reads_and_never_attempts_robot_io(tmp_path: Path) -> None:
    output_directory = tmp_path / "export"
    export = run_robot_trajectory_export(
        input_reference=_write_reference(tmp_path),
        calibration=_identity_calibration(),
        calibration_source="synthetic_test_only",
        output_directory=output_directory,
        generate_plots=False,
    )
    command_path = export.output_paths["reference_robot_trajectory"]
    files_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    stream = StringIO()
    result = dry_run_robot_trajectory(command_path, print_samples=3, stream=stream)
    files_after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert files_after == files_before
    assert result["dry_run_valid"] is True
    assert result["rows_printed"] == 3
    assert result["sdk_imported"] is False
    assert result["robot_connection_attempted"] is False
    assert result["robot_power_or_motion_command_sent"] is False
    assert "tcp_x_base_m" in stream.getvalue()


def test_dry_run_rejects_forged_schema_theta_closure_jump_and_rotvec(
    tmp_path: Path,
) -> None:
    command, _, _ = build_robot_trajectory(
        _synthetic_reference(), _identity_calibration()
    )

    missing_schema = command.drop(columns="theta_shank_rad")
    missing_path = tmp_path / "dry_missing_schema.csv"
    missing_schema.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        dry_run_robot_trajectory(missing_path)

    for name, mutate in (
        (
            "wrong_theta",
            lambda frame: frame.assign(
                theta_shank_rad=frame["q_hip_rad"] + frame["q_knee_rad"]
            ),
        ),
        (
            "not_closed",
            lambda frame: frame.assign(
                tcp_x_base_m=np.where(
                    frame.index == frame.index[-1],
                    frame["tcp_x_base_m"] + 0.05,
                    frame["tcp_x_base_m"],
                )
            ),
        ),
        (
            "single_frame_jump",
            lambda frame: frame.assign(
                tcp_x_base_m=np.where(
                    frame.index == len(frame) // 2,
                    frame["tcp_x_base_m"] + 1.0,
                    frame["tcp_x_base_m"],
                )
            ),
        ),
        (
            "rotation_vector_representation",
            lambda frame: frame.assign(
                tcp_orientation_representation="rotation_vector_rad"
            ),
        ),
    ):
        forged = mutate(command.copy(deep=True))
        path = tmp_path / f"dry_{name}.csv"
        forged.to_csv(path, index=False)
        result = dry_run_robot_trajectory(path)
        assert result["dry_run_valid"] is False, name


def test_dry_run_recomputes_source_gates_instead_of_trusting_stored_validity(
    tmp_path: Path,
) -> None:
    command, _, _ = build_robot_trajectory(
        _synthetic_reference(), _identity_calibration()
    )
    assert command["trajectory_valid"].all()
    assert command["invalid_reason"].eq("").all()

    for column in (
        "source_trajectory_valid",
        "source_reference_formal_execution_allowed",
    ):
        forged = command.copy(deep=True)
        forged[column] = False
        path = tmp_path / f"dry_forged_{column}.csv"
        forged.to_csv(path, index=False)
        result = dry_run_robot_trajectory(path)
        assert result["dry_run_valid"] is False
        assert result[
            "source_trajectory_valid_all"
            if column == "source_trajectory_valid"
            else "source_reference_formal_execution_allowed_all"
        ] is False


def test_dry_run_recomputes_fk_after_synchronized_q_and_theta_tampering(
    tmp_path: Path,
) -> None:
    command, _, _ = build_robot_trajectory(
        _synthetic_reference(), _identity_calibration()
    )
    forged = command.copy(deep=True)
    middle = len(forged) // 2
    forged.loc[middle, "q_hip_rad"] += 0.10
    forged.loc[middle, "theta_shank_rad"] = (
        forged.loc[middle, "q_hip_rad"] - forged.loc[middle, "q_knee_rad"]
    )
    path = tmp_path / "dry_forged_q_and_theta_with_stale_pull.csv"
    forged.to_csv(path, index=False)
    result = dry_run_robot_trajectory(path)
    assert result["theta_shank_definition_valid"] is True
    assert result["pull_forward_kinematics_valid"] is False
    assert result["maximum_pull_forward_kinematics_error_m"] > 0.01
    assert result["dry_run_valid"] is False


def test_runner_import_and_dry_run_are_lazy_without_matplotlib_side_effects(
    tmp_path: Path,
) -> None:
    command, _, _ = build_robot_trajectory(
        _synthetic_reference(), _identity_calibration()
    )
    command_path = tmp_path / "lazy_dry_run.csv"
    command.to_csv(command_path, index=False)
    isolated_tmp = tmp_path / "isolated_tmp"
    isolated_tmp.mkdir()
    script = """
import json
import os
from pathlib import Path
import sys

assert 'MPLCONFIGDIR' not in os.environ
import lower_limb_sim.run_robot_trajectory_export as runner
assert 'lower_limb_sim.visualize_robot_trajectory' not in sys.modules
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert 'MPLCONFIGDIR' not in os.environ
result = runner.dry_run_robot_trajectory(sys.argv[1])
assert result['dry_run_valid'] is True
assert 'lower_limb_sim.visualize_robot_trajectory' not in sys.modules
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert 'MPLCONFIGDIR' not in os.environ
assert not (Path(os.environ['TMPDIR']) / 'lower_limb_sim_matplotlib').exists()
print(json.dumps(result))
"""
    environment = os.environ.copy()
    environment.pop("MPLCONFIGDIR", None)
    environment["TMPDIR"] = str(isolated_tmp)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(command_path)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["dry_run_valid"] is True


def test_stage6a_sources_have_no_robot_stack_or_sdk_imports() -> None:
    package_directory = Path(__file__).resolve().parent
    forbidden_prefixes = ("hardware", "control", "collection", "safety")
    forbidden_fragments = ("rokae", "xcore", "sdk")
    for filename in STAGE6A_SOURCE_FILES:
        tree = ast.parse((package_directory / filename).read_text(encoding="utf-8"))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        for module in imported_modules:
            normalized = module.lower()
            top_level = normalized.split(".", maxsplit=1)[0]
            assert top_level not in forbidden_prefixes, (filename, module)
            assert not any(fragment in normalized for fragment in forbidden_fragments), (
                filename,
                module,
            )
