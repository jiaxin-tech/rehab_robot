"""Run Stage 5A skeleton-reference import and optional software dynamics.

The source CSV has no reliable sampling-rate metadata.  Geometry, un-clipped
angles, cycle detection, and forward kinematics are therefore available
without FPS, while derivatives and dynamics are only produced after the caller
supplies an explicit positive ``fps`` value.

This module reuses the existing kinematics, derivative estimation, full
dynamics, force mapping, and virtual-subject definitions.  It has no robot
control, acquisition, safety, hardware, or SDK dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import (
    L1,
    L2,
    hip_range_deg,
    knee_range_deg,
    reference_savgol_polynomial_order,
    reference_savgol_window_length,
    reference_trajectory_data_dir,
    reference_trajectory_model_version,
)
from .derivative_estimation import (
    DerivativeEstimationConfig,
    estimate_joint_derivatives,
)
from .dynamic_subject import DYNAMIC_SUBJECTS, get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .full_dynamics import InverseDynamicsResult, inverse_dynamics
from .kinematics import forward_kinematics
from .reference_trajectory_cycles import (
    CycleDetectionConfig,
    ReferenceCycleSelection,
    detect_flexion_extension_cycles,
    extract_selected_cycle,
    select_representative_cycle,
)
from .reference_trajectory_import import (
    ReferenceTrajectoryImportResult,
    import_reference_trajectory_csv,
)


SOURCE_TRAJECTORY_TYPE = "provided_rehabilitation_reference"
SIMULATION_STATUS = "software_only"
SUBJECT_IDS = ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
DYNAMIC_FILENAMES = {
    subject_id: f"reference_dynamic_{subject_id}.csv"
    for subject_id in SUBJECT_IDS
}


@dataclass(frozen=True)
class ReferenceTrajectoryRunResult:
    """In-memory data and written paths for one Stage 5A run."""

    import_result: ReferenceTrajectoryImportResult
    full_angles: pd.DataFrame
    detected_cycles: pd.DataFrame
    selected_cycle: pd.DataFrame
    selection: ReferenceCycleSelection
    dynamics_by_subject: dict[str, pd.DataFrame]
    metadata: dict[str, object]
    output_paths: dict[str, Path]
    visualization_paths: dict[str, Path]
    skipped_visualizations: dict[str, str]

    @property
    def derivatives_available(self) -> bool:
        return bool(self.metadata.get("derivatives_available", False))

    @property
    def dynamics_available(self) -> bool:
        return bool(self.metadata.get("dynamics_available", False))


def _validate_fps(fps: float | None) -> float | None:
    if fps is None:
        return None
    value = float(fps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("fps must be a finite positive value when provided.")
    return value


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _append_reason(base: np.ndarray, mask: np.ndarray, reason: str) -> None:
    for index in np.flatnonzero(np.asarray(mask, dtype=bool)):
        current = str(base[index])
        base[index] = f"{current};{reason}" if current else reason


def _merge_reason_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    reasons = np.full(len(dataframe), "", dtype=object)
    for column in columns:
        if column not in dataframe:
            continue
        values = dataframe[column].fillna("").astype(str).to_numpy()
        for index, value in enumerate(values):
            for token in filter(None, value.split(";")):
                existing = str(reasons[index]).split(";") if reasons[index] else []
                if token not in existing:
                    reasons[index] = (
                        f"{reasons[index]};{token}" if reasons[index] else token
                    )
    return reasons.astype(str)


def _contiguous_runs(frame: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for index in range(len(frame)):
        begins = bool(valid[index]) and (
            index == 0
            or not bool(valid[index - 1])
            or frame[index] - frame[index - 1] != 1
        )
        if begins:
            start = index
        ends = start is not None and (
            index == len(frame) - 1
            or not bool(valid[index + 1])
            or frame[index + 1] - frame[index] != 1
        )
        if ends:
            runs.append(np.arange(start, index + 1, dtype=int))
            start = None
    return runs


def _smooth_angles(full_angles: pd.DataFrame) -> pd.DataFrame:
    """Add filtered q columns without overwriting the imported raw angles."""

    output = full_angles.copy(deep=True)
    frame = output["Frame"].to_numpy(dtype=int)
    raw_hip = output["q_hip_raw_rad"].to_numpy(dtype=float)
    raw_knee = output["q_knee_raw_rad"].to_numpy(dtype=float)
    finite = np.isfinite(raw_hip) & np.isfinite(raw_knee)
    filtered_hip = np.full(len(output), np.nan)
    filtered_knee = np.full(len(output), np.nan)
    for run in _contiguous_runs(frame, finite):
        window = min(reference_savgol_window_length, len(run))
        if window % 2 == 0:
            window -= 1
        minimum = reference_savgol_polynomial_order + 2
        if minimum % 2 == 0:
            minimum += 1
        if window < minimum:
            filtered_hip[run] = raw_hip[run]
            filtered_knee[run] = raw_knee[run]
            continue
        filtered_hip[run] = savgol_filter(
            raw_hip[run],
            window_length=window,
            polyorder=reference_savgol_polynomial_order,
            mode="interp",
        )
        filtered_knee[run] = savgol_filter(
            raw_knee[run],
            window_length=window,
            polyorder=reference_savgol_polynomial_order,
            mode="interp",
        )
    output["q_hip_rad"] = filtered_hip
    output["q_knee_rad"] = filtered_knee
    output["q_hip_deg"] = np.rad2deg(filtered_hip)
    output["q_knee_deg"] = np.rad2deg(filtered_knee)
    output["theta_shank_rad"] = filtered_hip - filtered_knee
    output["theta_shank_deg"] = np.rad2deg(output["theta_shank_rad"])
    output["angle_filter_method"] = "savitzky_golay_offline_by_frame"
    output["angle_filter_uses_future_samples"] = True
    return output


def build_full_reference_angles(
    imported: ReferenceTrajectoryImportResult,
) -> pd.DataFrame:
    """Normalize import columns for cycle, plotting, and output contracts."""

    output = imported.trajectory.copy(deep=True)
    prefix = "R" if imported.primary_motion_leg == "right" else "L"
    output["q_hip_raw_rad"] = output["q_hip_rad"].to_numpy(dtype=float)
    output["q_knee_raw_rad"] = output["q_knee_rad"].to_numpy(dtype=float)
    output["q_hip_raw_deg"] = np.rad2deg(output["q_hip_raw_rad"])
    output["q_knee_raw_deg"] = np.rad2deg(output["q_knee_raw_rad"])
    output["x_hip_projected_m"] = output[f"{prefix}Hip_x_local_m"]
    output["z_hip_projected_m"] = output[f"{prefix}Hip_z_local_m"]
    output["x_knee_observed_m"] = output[f"{prefix}Knee_x_local_m"]
    output["z_knee_observed_m"] = output[f"{prefix}Knee_z_local_m"]
    output["x_ankle_observed_m"] = output["observed_ankle_x_local_m"]
    output["z_ankle_observed_m"] = output["observed_ankle_z_local_m"]
    plane_errors = output[
        [f"{prefix}{joint}_planarity_error_m" for joint in ("Hip", "Knee", "Ankle")]
    ].to_numpy(dtype=float)
    output["planarity_error_m"] = np.sqrt(np.mean(plane_errors**2, axis=1))
    output["joint_limit_valid"] = output["joint_range_valid"].astype(bool)
    output["joint_limit_reason"] = output["joint_range_reason"].astype(str)
    output["invalid_reason"] = _merge_reason_columns(
        output,
        ("angle_invalid_reason", "joint_limit_reason"),
    )
    output["source_trajectory_type"] = SOURCE_TRAJECTORY_TYPE
    output["simulation_status"] = SIMULATION_STATUS
    return _smooth_angles(output)


def _stage5_cycle_config() -> CycleDetectionConfig:
    """Settings that reject small marker jitter while retaining full cycles."""

    return CycleDetectionConfig(
        smoothing_window_frames=21,
        smoothing_polynomial_order=3,
        minimum_extrema_distance_frames=30,
        minimum_cycle_duration_frames=50,
        minimum_peak_prominence_rad=float(np.deg2rad(15.0)),
        minimum_knee_excursion_rad=float(np.deg2rad(25.0)),
        # Frame is an integer source identifier.  Any missing frame produces a
        # step >= 2 and must split the detector rather than be smoothed across.
        maximum_time_gap_factor=1.01,
    )


def _position_for_source_frame(frame: np.ndarray, requested: int, name: str) -> int:
    matches = np.flatnonzero(frame == int(requested))
    if len(matches) != 1:
        raise ValueError(f"{name}={requested} is not present exactly once in Frame.")
    return int(matches[0])


def _select_cycle(
    full_angles: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    cycle_index: int | None,
    start_frame: int | None,
    end_frame: int | None,
) -> ReferenceCycleSelection:
    if (start_frame is None) != (end_frame is None):
        raise ValueError("start_frame and end_frame must be supplied together.")
    frame = full_angles["Frame"].to_numpy(dtype=int)
    start_position = (
        None
        if start_frame is None
        else _position_for_source_frame(frame, int(start_frame), "start_frame")
    )
    end_position = (
        None
        if end_frame is None
        else _position_for_source_frame(frame, int(end_frame), "end_frame")
    )
    return select_representative_cycle(
        cycles,
        cycle_index=cycle_index,
        start_frame=start_position,
        end_frame=end_position,
    )


def _cycles_with_source_frames(
    cycles: pd.DataFrame,
    full_angles: pd.DataFrame,
    selection: ReferenceCycleSelection,
) -> pd.DataFrame:
    output = cycles.copy(deep=True)
    if output.empty:
        return output
    frame = full_angles["Frame"].to_numpy(dtype=int)
    for name in ("start", "peak_flexion", "end"):
        frame_column = f"{name}_frame"
        position_column = f"{name}_position"
        positions = output[frame_column].to_numpy(dtype=int)
        output[position_column] = positions
        output[frame_column] = frame[positions]
    output["selected"] = False
    if selection.cycle_index is not None:
        matched = output["cycle_index"].eq(int(selection.cycle_index))
        if int(matched.sum()) != 1:
            raise RuntimeError("selected cycle_index is not unique in cycle table.")
        output.loc[matched, "selected"] = True
    output.attrs.clear()
    return output


def _selected_cycle_with_geometry(
    full_angles: pd.DataFrame,
    selection: ReferenceCycleSelection,
    fps: float | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = extract_selected_cycle(full_angles, selection)
    peak_position = selection.peak_flexion_frame
    absolute_positions = selected["source_frame"].to_numpy(dtype=int)
    selected["cycle_phase"] = np.where(
        absolute_positions <= int(peak_position),
        "flexion",
        "extension",
    ) if peak_position is not None else "manual_unspecified"
    selected["phase"] = selected["cycle_phase"]

    if fps is None:
        selected["time_s"] = np.nan
        for column in (
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ):
            selected[column] = np.nan
        selected["derivative_valid"] = False
        selected["derivative_reason"] = "fps_not_provided"
        derivative_metadata: dict[str, object] = {
            "method": None,
            "offline_only": None,
            "uses_future_samples": None,
            "fps_required": True,
            "status": "not_computed_without_explicit_fps",
        }
    else:
        selected["time_s"] = (
            selected["Frame"].to_numpy(dtype=float)
            - float(selected["Frame"].iloc[0])
        ) / fps
        derivative_config = DerivativeEstimationConfig(
            savgol_window_length=reference_savgol_window_length,
            savgol_polynomial_order=reference_savgol_polynomial_order,
            maximum_time_gap_s=2.5 / fps,
        )
        derivative = estimate_joint_derivatives(
            selected,
            method="savitzky_golay_offline",
            config=derivative_config,
            time_column="time_s",
            angle_columns=("q_hip_raw_rad", "q_knee_raw_rad"),
            valid_column="angle_valid",
            group_columns=(),
        )
        selected = derivative.dataframe
        rename = {
            "dq_hip_est_rad_s": "dq_hip_rad_s",
            "dq_knee_est_rad_s": "dq_knee_rad_s",
            "ddq_hip_est_rad_s2": "ddq_hip_rad_s2",
            "ddq_knee_est_rad_s2": "ddq_knee_rad_s2",
        }
        selected.rename(columns=rename, inplace=True)
        derivative_metadata = dict(derivative.metadata)
        derivative_metadata["fps_required"] = True
        derivative_metadata["source_fps"] = fps
        derivative_metadata["status"] = "computed_from_filtered_angles"

    q_hip = selected["q_hip_rad"].to_numpy(dtype=float)
    q_knee = selected["q_knee_rad"].to_numpy(dtype=float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    selected["x_knee_m"] = x_knee
    selected["z_knee_m"] = z_knee
    selected["x_pull_m"] = x_pull
    selected["z_pull_m"] = z_pull
    # Explicit final aliases required by the reference-trajectory contract.
    selected["x_ankle_observed_m"] = selected["observed_ankle_x_local_m"]
    selected["z_ankle_observed_m"] = selected["observed_ankle_z_local_m"]

    hip_min, hip_max = np.deg2rad(hip_range_deg)
    knee_min, knee_max = np.deg2rad(knee_range_deg)
    finite_q = np.isfinite(q_hip) & np.isfinite(q_knee)
    selected["joint_limit_valid"] = (
        finite_q
        & (q_hip >= hip_min)
        & (q_hip <= hip_max)
        & (q_knee >= knee_min)
        & (q_knee <= knee_max)
    )
    joint_reasons = np.full(len(selected), "", dtype=object)
    _append_reason(joint_reasons, finite_q & ((q_hip < hip_min) | (q_hip > hip_max)), "q_hip_out_of_range")
    _append_reason(joint_reasons, finite_q & ((q_knee < knee_min) | (q_knee > knee_max)), "q_knee_out_of_range")
    _append_reason(joint_reasons, ~finite_q, "nonfinite_filtered_joint_angle")
    selected["joint_limit_reason"] = joint_reasons.astype(str)
    derivative_valid = selected["derivative_valid"].astype(bool).to_numpy()
    selected["trajectory_sample_valid"] = (
        selected["angle_valid"].astype(bool).to_numpy()
        & selected["joint_limit_valid"].astype(bool).to_numpy()
        & (derivative_valid if fps is not None else True)
    )
    validity_reason_columns = ["angle_invalid_reason", "joint_limit_reason"]
    if fps is not None:
        validity_reason_columns.append("derivative_reason")
    selected["invalid_reason"] = _merge_reason_columns(
        selected,
        validity_reason_columns,
    )
    selected["source_trajectory_type"] = SOURCE_TRAJECTORY_TYPE
    selected["simulation_status"] = SIMULATION_STATUS
    return selected, derivative_metadata


def _safe_inverse_dynamics(
    selected: pd.DataFrame,
    subject_id: str,
) -> pd.DataFrame:
    subject = get_dynamic_subject(subject_id)
    output = selected.copy(deep=True)
    output.insert(0, "subject_id", subject_id)
    q_hip = output["q_hip_rad"].to_numpy(dtype=float)
    q_knee = output["q_knee_rad"].to_numpy(dtype=float)
    dq_hip = output["dq_hip_rad_s"].to_numpy(dtype=float)
    dq_knee = output["dq_knee_rad_s"].to_numpy(dtype=float)
    ddq_hip = output["ddq_hip_rad_s2"].to_numpy(dtype=float)
    ddq_knee = output["ddq_knee_rad_s2"].to_numpy(dtype=float)
    input_valid = np.isfinite(
        np.column_stack((q_hip, q_knee, dq_hip, dq_knee, ddq_hip, ddq_knee))
    ).all(axis=1) & output["angle_valid"].astype(bool).to_numpy()
    output["dynamic_input_valid"] = input_valid

    torque_fields = [item.name for item in fields(InverseDynamicsResult)]
    for column in torque_fields:
        output[column] = np.nan
    if input_valid.any():
        dynamics = inverse_dynamics(
            q_hip[input_valid],
            q_knee[input_valid],
            dq_hip[input_valid],
            dq_knee[input_valid],
            ddq_hip[input_valid],
            ddq_knee[input_valid],
            subject,
            L1,
        )
        for column in torque_fields:
            output.loc[input_valid, column] = np.asarray(
                getattr(dynamics, column), dtype=float
            )

    force_columns = (
        "fx_robot_on_leg_n",
        "fz_robot_on_leg_n",
        "force_magnitude_n",
        "jacobian_determinant",
        "jacobian_condition_number",
    )
    for column in force_columns:
        output[column] = np.nan
    output["jacobian_near_singular"] = True
    output["force_mapping_valid"] = False
    force_reason = np.full(len(output), "dynamic_input_invalid", dtype=object)
    if input_valid.any():
        force = endpoint_force_from_joint_torque(
            q_hip[input_valid],
            q_knee[input_valid],
            output.loc[input_valid, "tau_total_hip_nm"].to_numpy(dtype=float),
            output.loc[input_valid, "tau_total_knee_nm"].to_numpy(dtype=float),
            L1,
            L2,
        )
        for column in force_columns:
            output.loc[input_valid, column] = np.asarray(
                getattr(force, column), dtype=float
            )
        output.loc[input_valid, "jacobian_near_singular"] = np.asarray(
            force.jacobian_near_singular, dtype=bool
        )
        output.loc[input_valid, "force_mapping_valid"] = np.asarray(
            force.force_mapping_valid, dtype=bool
        )
        force_reason[input_valid] = np.asarray(force.invalid_reason, dtype=str)
    output["force_mapping_reason"] = force_reason.astype(str)
    output["dynamic_sample_valid"] = (
        input_valid
        & output["joint_limit_valid"].astype(bool).to_numpy()
        & output["force_mapping_valid"].astype(bool).to_numpy()
    )
    output["model_angle_definition"] = "theta_shank = q_hip - q_knee"
    output["source_trajectory_type"] = SOURCE_TRAJECTORY_TYPE
    output["simulation_status"] = SIMULATION_STATUS
    return output


def simulate_reference_cycle_dynamics(
    selected_cycle: pd.DataFrame,
    *,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> dict[str, pd.DataFrame]:
    """Run existing inverse dynamics for a time-calibrated selected cycle."""

    required = {
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    }
    missing = required.difference(selected_cycle.columns)
    if missing:
        raise ValueError(f"selected cycle is missing dynamic inputs: {sorted(missing)}")
    if not np.isfinite(selected_cycle["time_s"].to_numpy(dtype=float)).all():
        raise ValueError("explicit fps is required before dynamics can be run.")
    unknown = set(subject_ids).difference(DYNAMIC_SUBJECTS)
    if unknown:
        raise ValueError(f"unknown dynamic subjects: {sorted(unknown)}")
    return {
        subject_id: _safe_inverse_dynamics(selected_cycle, subject_id)
        for subject_id in subject_ids
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _save_npz(dataframe: pd.DataFrame, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for column in dataframe.columns:
        values = dataframe[column].to_numpy()
        if values.dtype == object:
            values = dataframe[column].fillna("").astype(str).to_numpy(dtype=str)
        arrays[column] = values
    np.savez_compressed(path, **arrays)


def _peak_summary(dynamics: pd.DataFrame) -> dict[str, object]:
    force_mapping_valid = dynamics["force_mapping_valid"].astype(bool).to_numpy()
    dynamic_valid = dynamics["dynamic_sample_valid"].astype(bool).to_numpy()

    def peak_abs(column: str, mask: np.ndarray) -> float | None:
        values = dynamics[column].to_numpy(dtype=float)
        finite = np.isfinite(values) & mask
        return float(np.max(np.abs(values[finite]))) if finite.any() else None

    return {
        "peak_abs_hip_torque_nm": peak_abs("tau_total_hip_nm", dynamic_valid),
        "peak_abs_knee_torque_nm": peak_abs("tau_total_knee_nm", dynamic_valid),
        "peak_force_magnitude_n": (
            float(
                np.nanmax(
                    dynamics.loc[dynamic_valid, "force_magnitude_n"].to_numpy(dtype=float)
                )
            )
            if dynamic_valid.any()
            else None
        ),
        "peak_definition": "dynamic_sample_valid_only",
        "all_mapped_peak_abs_hip_torque_nm": peak_abs(
            "tau_total_hip_nm", force_mapping_valid
        ),
        "all_mapped_peak_abs_knee_torque_nm": peak_abs(
            "tau_total_knee_nm", force_mapping_valid
        ),
        "all_mapped_peak_force_magnitude_n": peak_abs(
            "force_magnitude_n", force_mapping_valid
        ),
        "force_mapping_valid_samples": int(force_mapping_valid.sum()),
        "dynamic_sample_valid_samples": int(dynamic_valid.sum()),
    }


def run_reference_trajectory(
    input_path: str | Path,
    *,
    coordinate_unit: str,
    fps: float | None = None,
    leg: str = "right",
    start_frame: int | None = None,
    end_frame: int | None = None,
    cycle_index: int | None = None,
    output_directory: str | Path = reference_trajectory_data_dir,
    generate_visualizations: bool = True,
) -> ReferenceTrajectoryRunResult:
    """Import one marker CSV and write the complete auditable Stage 5A output."""

    source_fps = _validate_fps(fps)
    if cycle_index is not None and (start_frame is not None or end_frame is not None):
        raise ValueError("choose cycle_index or start_frame/end_frame, not both.")
    imported = import_reference_trajectory_csv(
        input_path,
        coordinate_unit=coordinate_unit,
        primary_motion_leg=leg,
    )
    full_angles = build_full_reference_angles(imported)
    cycle_config = _stage5_cycle_config()
    cycles_internal = detect_flexion_extension_cycles(
        full_angles,
        config=cycle_config,
        q_hip_column="q_hip_raw_rad",
        q_knee_column="q_knee_raw_rad",
        time_column="Frame",
    )
    selection = _select_cycle(
        full_angles,
        cycles_internal,
        cycle_index=cycle_index,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    cycles_output = _cycles_with_source_frames(
        cycles_internal,
        full_angles,
        selection,
    )
    selected, derivative_metadata = _selected_cycle_with_geometry(
        full_angles,
        selection,
        source_fps,
    )
    dynamics = (
        simulate_reference_cycle_dynamics(selected)
        if source_fps is not None
        else {}
    )

    frame_values = full_angles["Frame"].to_numpy(dtype=int)
    selected_source_frames = {
        "start_frame": int(frame_values[selection.start_frame]),
        "peak_flexion_frame": (
            int(frame_values[selection.peak_flexion_frame])
            if selection.peak_flexion_frame is not None
            else None
        ),
        "end_frame": int(frame_values[selection.end_frame]),
    }
    public_selection = replace(
        selection,
        start_frame=selected_source_frames["start_frame"],
        peak_flexion_frame=selected_source_frames["peak_flexion_frame"],
        end_frame=selected_source_frames["end_frame"],
    )
    prefix = "R" if imported.primary_motion_leg == "right" else "L"
    joint_plane_columns = {
        joint.lower(): f"{prefix}{joint}_planarity_error_m"
        for joint in ("Hip", "Knee", "Ankle")
    }
    planarity_by_joint = {
        joint: {
            "rmse_m": float(
                np.sqrt(np.mean(full_angles[column].to_numpy(dtype=float) ** 2))
            ),
            "max_error_m": float(full_angles[column].max()),
        }
        for joint, column in joint_plane_columns.items()
    }
    complete_cycles = int(cycles_output["cycle_complete"].astype(bool).sum())
    metadata: dict[str, object] = {
        **imported.metadata,
        "stage": "5A",
        "model_version": reference_trajectory_model_version,
        "software_version_or_git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path).expanduser().resolve()),
        "source_unit": coordinate_unit,
        "source_unit_declared_by_caller": True,
        "source_unit_present_in_csv_metadata": False,
        "unit_uncertainty_remains": True,
        "fps": source_fps,
        "fps_declared_by_caller": source_fps is not None,
        "fps_present_in_csv_metadata": False,
        "timing_uncertainty_remains": source_fps is None,
        "primary_motion_leg": imported.primary_motion_leg,
        "L1_m": L1,
        "L2_m": L2,
        "L2_definition": "knee_to_equivalent_strap_pull_point_not_anatomical_ankle",
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "source_trajectory_type": SOURCE_TRAJECTORY_TYPE,
        "simulation_status": SIMULATION_STATUS,
        "clinical_validation_status": "not_established_from_source_csv",
        "configured_hip_range_deg": list(hip_range_deg),
        "configured_knee_range_deg": list(knee_range_deg),
        "angle_filter": {
            "method": "savitzky_golay_offline_by_frame",
            "window_length": reference_savgol_window_length,
            "polynomial_order": reference_savgol_polynomial_order,
            "uses_future_samples": True,
        },
        "derivatives_available": source_fps is not None,
        "dynamics_available": source_fps is not None,
        "derivative_estimation": derivative_metadata,
        "dynamic_gate_reason": (
            None if source_fps is not None else "fps_not_provided"
        ),
        "cycle_detection_config": asdict(cycle_config),
        "detected_cycle_count": int(len(cycles_output)),
        "complete_cycle_count": complete_cycles,
        "incomplete_cycle_count": int(len(cycles_output) - complete_cycles),
        "selected_cycle": {
            **selection.as_dict(),
            **selected_source_frames,
            "start_position": selection.start_frame,
            "peak_flexion_position": selection.peak_flexion_frame,
            "end_position": selection.end_frame,
        },
        "selection_reason": selection.selection_reason,
        "planarity_rmse_m": imported.metadata["primary_leg_planarity_rmse_m"],
        "planarity_max_error_m": imported.metadata["primary_leg_planarity_max_m"],
        "planarity_by_joint": planarity_by_joint,
        "raw_angle_ranges_deg": {
            "q_hip_min": float(full_angles["q_hip_raw_deg"].min()),
            "q_hip_max": float(full_angles["q_hip_raw_deg"].max()),
            "q_knee_min": float(full_angles["q_knee_raw_deg"].min()),
            "q_knee_max": float(full_angles["q_knee_raw_deg"].max()),
        },
        "angle_invalid_samples": int((~full_angles["angle_valid"]).sum()),
        "joint_limit_invalid_samples": int((~full_angles["joint_limit_valid"]).sum()),
        "angle_jump_threshold_deg": 20.0,
        "angle_jump_samples": int(
            (
                (full_angles["q_hip_raw_deg"].diff().abs() > 20.0)
                | (full_angles["q_knee_raw_deg"].diff().abs() > 20.0)
            ).sum()
        ),
        "observed_ankle_is_pull_point": False,
        "dynamics_subjects": list(dynamics),
        "dynamic_peak_summary": {
            subject_id: _peak_summary(dataframe)
            for subject_id, dataframe in dynamics.items()
        },
        "real_robot_code_used": False,
        "real_robot_code_modified": False,
    }

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    def save_csv(name: str, dataframe: pd.DataFrame) -> None:
        path = destination / name
        dataframe.to_csv(path, index=False)
        output_paths[name] = path

    save_csv("reference_full_raw.csv", imported.source_dataframe)
    save_csv("reference_full_angles.csv", full_angles)
    save_csv("detected_cycles.csv", cycles_output)
    save_csv("reference_selected_cycle.csv", selected)
    npz_path = destination / "reference_selected_cycle.npz"
    _save_npz(selected, npz_path)
    output_paths[npz_path.name] = npz_path
    for subject_id, dataframe in dynamics.items():
        save_csv(DYNAMIC_FILENAMES[subject_id], dataframe)

    visualization_paths: dict[str, Path] = {}
    skipped_visualizations: dict[str, str] = {}
    if generate_visualizations:
        from .visualize_reference_trajectory import (
            generate_reference_trajectory_visualizations,
        )

        visualization = generate_reference_trajectory_visualizations(
            imported.marker_dataframe_m,
            full_angles,
            cycles_output,
            selected,
            dynamics,
            metadata,
            destination,
        )
        visualization_paths = dict(visualization.paths)
        skipped_visualizations = dict(visualization.skipped)
        output_paths.update(visualization_paths)
    skipped_dynamic_files = (
        {}
        if source_fps is not None
        else {
            filename: "fps_not_provided; dynamics were not computed"
            for filename in DYNAMIC_FILENAMES.values()
        }
    )
    metadata["skipped_dynamic_files"] = skipped_dynamic_files
    metadata["skipped_visualizations"] = skipped_visualizations
    metadata["generated_files"] = sorted([*output_paths, "metadata.json"])
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    output_paths[metadata_path.name] = metadata_path
    return ReferenceTrajectoryRunResult(
        import_result=imported,
        full_angles=full_angles,
        detected_cycles=cycles_output,
        selected_cycle=selected,
        selection=public_selection,
        dynamics_by_subject=dynamics,
        metadata=metadata,
        output_paths=output_paths,
        visualization_paths=visualization_paths,
        skipped_visualizations=skipped_visualizations,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a 3-D bilateral-leg skeleton reference into the existing "
            "2-D lower-limb model. FPS is never assumed."
        )
    )
    parser.add_argument("--input", required=True, help="Input marker CSV path.")
    parser.add_argument("--fps", type=float, default=None, help="Trusted source FPS.")
    parser.add_argument("--unit", choices=("mm", "m"), required=True)
    parser.add_argument("--leg", choices=("right", "left"), default="right")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--cycle-index", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        default=str(reference_trajectory_data_dir),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_reference_trajectory(
        args.input,
        coordinate_unit=args.unit,
        fps=args.fps,
        leg=args.leg,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        cycle_index=args.cycle_index,
        output_directory=args.output_directory,
        generate_visualizations=not args.no_plots,
    )
    meta = result.metadata
    selected = meta["selected_cycle"]
    print(
        f"Stage 5A: {len(result.full_angles)} frames, "
        f"{meta['complete_cycle_count']} complete cycles, "
        f"selected cycle {selected['cycle_index']} "
        f"({selected['start_frame']}..{selected['end_frame']})."
    )
    if result.dynamics_available:
        print(f"Derivatives/dynamics computed with explicit fps={meta['fps']:.6g} Hz.")
    else:
        print(
            "FPS was not provided: q/geometry/cycles were saved, but dq/ddq and "
            "all dynamic force outputs remain unavailable."
        )
    print(f"Output directory: {Path(args.output_directory).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DYNAMIC_FILENAMES",
    "ReferenceTrajectoryRunResult",
    "SIMULATION_STATUS",
    "SOURCE_TRAJECTORY_TYPE",
    "SUBJECT_IDS",
    "build_full_reference_angles",
    "main",
    "run_reference_trajectory",
    "simulate_reference_cycle_dynamics",
]
