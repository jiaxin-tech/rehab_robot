"""Offline pre-execution audit for Stage-6A TCP trajectory files.

The reported maxima are measurements of the supplied trajectory, not robot
safety limits.  Jump flags use a dimensionless, configurable robust outlier
ratio solely as a data-quality diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import L1, L2
from .kinematics import forward_kinematics
from .robot_coordinate_transform import (
    MODEL_ANGLE_DEFINITION,
    RobotFrameCalibration,
    tcp_origins_to_pull_points_base,
)


REQUIRED_COMMAND_COLUMNS = (
    "time_s",
    "q_hip_rad",
    "q_knee_rad",
    "x_pull_human_m",
    "z_pull_human_m",
    "tcp_x_base_m",
    "tcp_y_base_m",
    "tcp_z_base_m",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
)

DRY_RUN_REQUIRED_COLUMNS = (
    *REQUIRED_COMMAND_COLUMNS,
    "theta_shank_rad",
    "tcp_orientation_representation",
    "model_angle_definition",
    "trajectory_valid",
    "invalid_reason",
    "source_trajectory_valid",
    "source_reference_formal_execution_allowed",
    "robot_execution_approved",
    "trajectory_generated_offline_only",
)


@dataclass(frozen=True)
class RobotTrajectoryAudit:
    sample_count: int
    all_samples_finite: bool
    time_strictly_increasing: bool
    position_continuous: bool
    velocity_continuous: bool
    acceleration_continuous: bool
    obvious_single_frame_jump_detected: bool
    position_jump_sample_count: int
    velocity_jump_sample_count: int
    acceleration_jump_sample_count: int
    start_end_closure_error_m: float
    start_end_closed: bool
    maximum_displacement_from_start_m: float
    total_path_length_m: float
    maximum_single_step_displacement_m: float
    maximum_cartesian_speed_m_s: float
    maximum_cartesian_acceleration_m_s2: float
    tcp_x_range_base_m: tuple[float, float]
    tcp_y_range_base_m: tuple[float, float]
    tcp_z_range_base_m: tuple[float, float]
    transform_is_orthogonal: bool
    transform_orthogonality_error: float
    transform_determinant: float
    tool_offset_correctly_applied: bool
    maximum_tool_offset_reconstruction_error_m: float
    theta_shank_definition_valid: bool
    trajectory_all_samples_valid: bool
    invalid_sample_count: int
    jump_ratio_threshold: float
    closure_tolerance_m: float
    tool_offset_tolerance_m: float
    safety_thresholds_applied: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _append_reason(reasons: np.ndarray, mask: np.ndarray, token: str) -> None:
    for index in np.flatnonzero(mask):
        current = str(reasons[index])
        reasons[index] = f"{current};{token}" if current else token


def _range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


def _robust_jump_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    """Flag unusually large nonnegative increments without a physical limit."""

    array = np.asarray(values, dtype=float)
    result = np.zeros(array.shape, dtype=bool)
    finite = array[np.isfinite(array)]
    if finite.size < 5:
        return result
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, median * 0.05, np.finfo(float).eps)
    diagnostic_limit = max(median * ratio, median + ratio * scale)
    result[np.isfinite(array)] = array[np.isfinite(array)] > diagnostic_limit
    return result


def _derivatives(
    time_s: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) < 3 or not np.isfinite(positions).all():
        shape = positions.shape
        return np.full(shape, np.nan), np.full(shape, np.nan)
    velocity = np.column_stack(
        [np.gradient(positions[:, axis], time_s, edge_order=2) for axis in range(3)]
    )
    acceleration = np.column_stack(
        [np.gradient(velocity[:, axis], time_s, edge_order=2) for axis in range(3)]
    )
    return velocity, acceleration


def audit_robot_trajectory(
    trajectory: pd.DataFrame,
    calibration: RobotFrameCalibration,
    *,
    jump_ratio_threshold: float = 10.0,
    closure_tolerance_m: float = 1e-8,
    tool_offset_tolerance_m: float = 1e-10,
) -> tuple[pd.DataFrame, RobotTrajectoryAudit]:
    """Attach derivative/validity columns and compute the Stage-6A audit."""

    missing = set(REQUIRED_COMMAND_COLUMNS).difference(trajectory.columns)
    if missing:
        raise ValueError(f"robot trajectory missing columns: {sorted(missing)}")
    ratio = float(jump_ratio_threshold)
    closure_tolerance = float(closure_tolerance_m)
    offset_tolerance = float(tool_offset_tolerance_m)
    if not np.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("jump_ratio_threshold must be finite and greater than one.")
    if not np.isfinite(closure_tolerance) or closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance_m must be finite and positive.")
    if not np.isfinite(offset_tolerance) or offset_tolerance <= 0.0:
        raise ValueError("tool_offset_tolerance_m must be finite and positive.")

    output = trajectory.copy(deep=True).reset_index(drop=True)
    time_s = output["time_s"].to_numpy(dtype=float)
    if len(time_s) < 3:
        raise ValueError("robot trajectory needs at least three samples.")
    time_finite = np.isfinite(time_s).all()
    time_increasing = bool(time_finite and np.all(np.diff(time_s) > 0.0))
    if not time_increasing:
        raise ValueError("time_s must be finite and strictly increasing.")

    numeric = output.loc[:, REQUIRED_COMMAND_COLUMNS].to_numpy(dtype=float)
    finite_sample = np.isfinite(numeric).all(axis=1)
    all_finite = bool(finite_sample.all())
    positions = output[
        ["tcp_x_base_m", "tcp_y_base_m", "tcp_z_base_m"]
    ].to_numpy(dtype=float)
    velocity, acceleration = _derivatives(time_s, positions)
    output[["tcp_vx_base_m_s", "tcp_vy_base_m_s", "tcp_vz_base_m_s"]] = velocity
    output[
        ["tcp_ax_base_m_s2", "tcp_ay_base_m_s2", "tcp_az_base_m_s2"]
    ] = acceleration
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    output["tcp_speed_m_s"] = speed
    output["tcp_acceleration_m_s2"] = acceleration_magnitude
    derivative_finite_sample = np.isfinite(
        np.column_stack((velocity, acceleration, speed, acceleration_magnitude))
    ).all(axis=1)

    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    velocity_steps = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
    acceleration_steps = np.linalg.norm(np.diff(acceleration, axis=0), axis=1)
    position_jump_edges = _robust_jump_mask(position_steps, ratio)
    velocity_jump_edges = _robust_jump_mask(velocity_steps, ratio)
    acceleration_jump_edges = _robust_jump_mask(acceleration_steps, ratio)
    position_jump_samples = np.concatenate(([False], position_jump_edges))
    velocity_jump_samples = np.concatenate(([False], velocity_jump_edges))
    acceleration_jump_samples = np.concatenate(([False], acceleration_jump_edges))
    output["single_frame_position_jump"] = position_jump_samples
    output["velocity_discontinuity_outlier"] = velocity_jump_samples
    output["acceleration_discontinuity_outlier"] = acceleration_jump_samples

    closure_error = float(np.linalg.norm(positions[-1] - positions[0]))
    start_end_closed = bool(np.isfinite(closure_error) and closure_error <= closure_tolerance)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    path_length = float(np.sum(position_steps)) if np.isfinite(position_steps).all() else np.nan

    theta_expected = (
        output["q_hip_rad"].to_numpy(float)
        - output["q_knee_rad"].to_numpy(float)
    )
    if "theta_shank_rad" in output:
        theta_valid_samples = np.isclose(
            output["theta_shank_rad"].to_numpy(float),
            theta_expected,
            atol=1e-12,
            rtol=0.0,
        )
    else:
        output["theta_shank_rad"] = theta_expected
        theta_valid_samples = np.ones(len(output), dtype=bool)
    theta_valid = bool(theta_valid_samples.all())
    output["model_angle_definition"] = MODEL_ANGLE_DEFINITION

    pull_columns = ("pull_x_base_m", "pull_y_base_m", "pull_z_base_m")
    if set(pull_columns).issubset(output.columns):
        expected_pull = output.loc[:, pull_columns].to_numpy(dtype=float)
        reconstructed_pull = tcp_origins_to_pull_points_base(positions, calibration)
        offset_error = np.linalg.norm(reconstructed_pull - expected_pull, axis=1)
    else:
        offset_error = np.full(len(output), np.nan)
    output["tool_offset_reconstruction_error_m"] = offset_error
    maximum_offset_error = _finite_max(offset_error)
    tool_offset_correct = bool(
        np.isfinite(maximum_offset_error)
        and maximum_offset_error <= offset_tolerance
    )

    source_reason = (
        output["source_invalid_reason"].fillna("").astype(str).to_numpy()
        if "source_invalid_reason" in output
        else output["invalid_reason"].fillna("").astype(str).to_numpy()
        if "invalid_reason" in output
        else np.full(len(output), "", dtype=object)
    )
    reasons = np.asarray(source_reason, dtype=object).copy()
    _append_reason(reasons, ~finite_sample, "non_finite_sample")
    _append_reason(
        reasons,
        ~derivative_finite_sample,
        "cartesian_derivative_non_finite_or_unavailable",
    )
    _append_reason(reasons, position_jump_samples, "single_frame_position_jump")
    _append_reason(reasons, velocity_jump_samples, "velocity_discontinuity_outlier")
    _append_reason(
        reasons,
        acceleration_jump_samples,
        "acceleration_discontinuity_outlier",
    )
    _append_reason(reasons, ~theta_valid_samples, "theta_shank_definition_mismatch")
    if "source_trajectory_valid" in output:
        source_valid = output["source_trajectory_valid"].fillna(False).astype(bool).to_numpy()
        _append_reason(reasons, ~source_valid, "source_reference_sample_invalid")
    if "source_reference_formal_execution_allowed" in output:
        formal_allowed = output[
            "source_reference_formal_execution_allowed"
        ].fillna(False).astype(bool).to_numpy()
        _append_reason(
            reasons,
            ~formal_allowed,
            "source_reference_formal_gate_not_approved",
        )
    if not start_end_closed:
        _append_reason(reasons, np.ones(len(output), dtype=bool), "not_closed")
    if not calibration.transform_is_orthogonal:
        _append_reason(
            reasons,
            np.ones(len(output), dtype=bool),
            "coordinate_transform_not_orthogonal",
        )
    if not tool_offset_correct:
        _append_reason(
            reasons,
            np.ones(len(output), dtype=bool),
            "tool_offset_application_invalid",
        )
    output["invalid_reason"] = reasons.astype(str)
    output["trajectory_valid"] = reasons == ""

    audit = RobotTrajectoryAudit(
        sample_count=len(output),
        all_samples_finite=all_finite,
        time_strictly_increasing=time_increasing,
        position_continuous=bool(all_finite and not position_jump_samples.any()),
        velocity_continuous=bool(
            np.isfinite(velocity).all() and not velocity_jump_samples.any()
        ),
        acceleration_continuous=bool(
            np.isfinite(acceleration).all() and not acceleration_jump_samples.any()
        ),
        obvious_single_frame_jump_detected=bool(position_jump_samples.any()),
        position_jump_sample_count=int(position_jump_samples.sum()),
        velocity_jump_sample_count=int(velocity_jump_samples.sum()),
        acceleration_jump_sample_count=int(acceleration_jump_samples.sum()),
        start_end_closure_error_m=closure_error,
        start_end_closed=start_end_closed,
        maximum_displacement_from_start_m=_finite_max(displacement),
        total_path_length_m=path_length,
        maximum_single_step_displacement_m=_finite_max(position_steps),
        maximum_cartesian_speed_m_s=_finite_max(speed),
        maximum_cartesian_acceleration_m_s2=_finite_max(acceleration_magnitude),
        tcp_x_range_base_m=_range(positions[:, 0]),
        tcp_y_range_base_m=_range(positions[:, 1]),
        tcp_z_range_base_m=_range(positions[:, 2]),
        transform_is_orthogonal=calibration.transform_is_orthogonal,
        transform_orthogonality_error=calibration.transform_orthogonality_error,
        transform_determinant=calibration.transform_determinant,
        tool_offset_correctly_applied=tool_offset_correct,
        maximum_tool_offset_reconstruction_error_m=maximum_offset_error,
        theta_shank_definition_valid=theta_valid,
        trajectory_all_samples_valid=bool(output["trajectory_valid"].all()),
        invalid_sample_count=int((~output["trajectory_valid"]).sum()),
        jump_ratio_threshold=ratio,
        closure_tolerance_m=closure_tolerance,
        tool_offset_tolerance_m=offset_tolerance,
        safety_thresholds_applied=False,
    )
    return output, audit


def _strict_boolean_series(values: pd.Series) -> tuple[np.ndarray, bool]:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool), bool(values.notna().all())
    normalized = values.astype("string").str.strip().str.lower()
    tokens = normalized.fillna("").to_numpy(dtype=str)
    recognized = np.isin(tokens, ("true", "false", "1", "0"))
    parsed = np.isin(tokens, ("true", "1"))
    return parsed, bool(recognized.all())


def validate_dry_run_command_file(dataframe: pd.DataFrame) -> dict[str, object]:
    """Independently re-check an exported CSV without calibration or robot I/O.

    The dry-run does not trust a stored ``trajectory_valid=True`` by itself.  It
    recomputes the angle identity, time/order, closure and position-jump checks.
    Tool-offset recovery still belongs to the calibration-aware export audit.
    """

    missing = set(DRY_RUN_REQUIRED_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(f"dry-run command file missing columns: {sorted(missing)}")
    if len(dataframe) < 3:
        raise ValueError("dry-run command file needs at least three samples.")
    time_s = dataframe["time_s"].to_numpy(dtype=float)
    time_valid = bool(np.isfinite(time_s).all() and np.all(np.diff(time_s) > 0.0))
    finite = bool(
        np.isfinite(dataframe.loc[:, REQUIRED_COMMAND_COLUMNS].to_numpy(float)).all()
    )
    stored_valid_values, stored_valid_encoding = _strict_boolean_series(
        dataframe["trajectory_valid"]
    )
    stored_valid = bool(stored_valid_encoding and stored_valid_values.all())
    invalid_reasons_empty = bool(
        dataframe["invalid_reason"].fillna("").astype(str).str.strip().eq("").all()
    )
    source_valid_values, source_valid_encoding = _strict_boolean_series(
        dataframe["source_trajectory_valid"]
    )
    source_valid_all = bool(source_valid_encoding and source_valid_values.all())
    formal_values, formal_encoding = _strict_boolean_series(
        dataframe["source_reference_formal_execution_allowed"]
    )
    source_formal_gate_all = bool(formal_encoding and formal_values.all())
    robot_approved_values, robot_approved_encoding = _strict_boolean_series(
        dataframe["robot_execution_approved"]
    )
    robot_execution_approved_all_false = bool(
        robot_approved_encoding and not robot_approved_values.any()
    )
    offline_values, offline_encoding = _strict_boolean_series(
        dataframe["trajectory_generated_offline_only"]
    )
    trajectory_offline_only_all_true = bool(
        offline_encoding and offline_values.all()
    )

    q_hip = dataframe["q_hip_rad"].to_numpy(dtype=float)
    q_knee = dataframe["q_knee_rad"].to_numpy(dtype=float)
    theta = dataframe["theta_shank_rad"].to_numpy(dtype=float)
    theta_valid = bool(
        np.isfinite(theta).all()
        and np.allclose(theta, q_hip - q_knee, atol=1e-12, rtol=0.0)
    )
    _, _, x_pull_fk, z_pull_fk = forward_kinematics(q_hip, q_knee, L1, L2)
    pull_error = np.hypot(
        dataframe["x_pull_human_m"].to_numpy(dtype=float) - x_pull_fk,
        dataframe["z_pull_human_m"].to_numpy(dtype=float) - z_pull_fk,
    )
    maximum_pull_fk_error = (
        float(np.max(pull_error)) if np.isfinite(pull_error).all() else float("nan")
    )
    pull_forward_kinematics_valid = bool(
        np.isfinite(maximum_pull_fk_error) and maximum_pull_fk_error <= 1e-10
    )
    model_definition_valid = bool(
        dataframe["model_angle_definition"]
        .fillna("")
        .astype(str)
        .eq(MODEL_ANGLE_DEFINITION)
        .all()
    )

    representation = (
        dataframe["tcp_orientation_representation"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    # Stage-6A command columns rx/ry/rz are canonical active XYZ Euler/RPY.
    orientation_representation_valid = bool(
        representation.eq("euler_xyz_rad").all()
    )
    orientation = dataframe[
        ["tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad"]
    ].to_numpy(dtype=float)
    orientation_constant = bool(
        np.isfinite(orientation).all()
        and np.allclose(orientation, orientation[0], atol=1e-12, rtol=0.0)
    )

    positions = dataframe[
        ["tcp_x_base_m", "tcp_y_base_m", "tcp_z_base_m"]
    ].to_numpy(dtype=float)
    closure_error = (
        float(np.linalg.norm(positions[-1] - positions[0]))
        if np.isfinite(positions[[0, -1]]).all()
        else float("nan")
    )
    start_end_closed = bool(
        np.isfinite(closure_error) and closure_error <= 1e-8
    )
    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    position_jump_count = int(_robust_jump_mask(position_steps, 10.0).sum())
    position_jump_free = position_jump_count == 0

    dry_run_valid = bool(
        time_valid
        and finite
        and stored_valid
        and invalid_reasons_empty
        and source_valid_all
        and source_formal_gate_all
        and robot_execution_approved_all_false
        and trajectory_offline_only_all_true
        and theta_valid
        and pull_forward_kinematics_valid
        and model_definition_valid
        and orientation_representation_valid
        and orientation_constant
        and start_end_closed
        and position_jump_free
    )
    return {
        "sample_count": int(len(dataframe)),
        "time_strictly_increasing": time_valid,
        "required_values_all_finite": finite,
        "trajectory_valid_encoding_recognized": stored_valid_encoding,
        "stored_trajectory_valid_all": stored_valid,
        "stored_invalid_reasons_all_empty": invalid_reasons_empty,
        "source_trajectory_valid_encoding_recognized": source_valid_encoding,
        "source_trajectory_valid_all": source_valid_all,
        "source_formal_gate_encoding_recognized": formal_encoding,
        "source_reference_formal_execution_allowed_all": source_formal_gate_all,
        "robot_execution_approved_encoding_recognized": robot_approved_encoding,
        "robot_execution_approved_all_false": robot_execution_approved_all_false,
        "trajectory_offline_only_encoding_recognized": offline_encoding,
        "trajectory_generated_offline_only_all_true": trajectory_offline_only_all_true,
        "theta_shank_definition_valid": theta_valid,
        "pull_forward_kinematics_valid": pull_forward_kinematics_valid,
        "maximum_pull_forward_kinematics_error_m": maximum_pull_fk_error,
        "model_angle_definition_valid": model_definition_valid,
        "tcp_orientation_representation_valid": orientation_representation_valid,
        "tcp_orientation_constant": orientation_constant,
        "start_end_closure_error_m": closure_error,
        "start_end_closed": start_end_closed,
        "position_jump_sample_count": position_jump_count,
        "obvious_single_frame_jump_detected": not position_jump_free,
        "dry_run_valid": dry_run_valid,
        "robot_execution_approved": False,
        "trajectory_generated_offline_only": True,
        "sdk_imported": False,
        "robot_connection_attempted": False,
        "robot_power_or_motion_command_sent": False,
    }


__all__ = [
    "DRY_RUN_REQUIRED_COLUMNS",
    "REQUIRED_COMMAND_COLUMNS",
    "RobotTrajectoryAudit",
    "audit_robot_trajectory",
    "validate_dry_run_command_file",
]
