"""Stage 5B reference-path retiming and software-only dynamics evaluation.

The original skeleton CSV has no trustworthy frame rate.  This module therefore
uses only the *shape* of a Stage-5A processed cycle, parameterises its flexion
and extension branches by geometric path phase, and applies a newly prescribed
minimum-jerk clock.  The new clock is not an estimate of the source motion.

The leg convention remains strictly::

    theta_shank = q_hip - q_knee

No robot-control, acquisition, safety, hardware, or SDK module is imported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import (
    L1,
    L2,
    hip_range_deg,
    identification_data_dir,
    identification_trajectory_id,
    knee_range_deg,
    reference_phase_samples_per_segment,
    reference_retiming_data_dir,
    reference_retiming_durations_s,
    reference_retiming_model_version,
    reference_trajectory_data_dir,
)
from .dynamic_subject import DYNAMIC_SUBJECTS, get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import (
    ESTIMATED_DOMAIN_STATE_COLUMNS,
    StateDomainBounds,
    classify_state_domain,
    fit_state_domain_bounds,
)
from .kinematics import forward_kinematics
from .trajectory_profiles import minimum_jerk_profile


SOURCE_TRAJECTORY_TYPE = "provided_rehabilitation_reference"
SIMULATION_STATUS = "software_only"
MODEL_ANGLE_DEFINITION = "theta_shank = q_hip - q_knee"
SOURCE_TIMING_STATUS = "unknown"
DOMAIN_MODEL = "axis_aligned_6d_training_box"
DEFAULT_PROFILES = ("slow", "nominal", "fast")
SUBJECT_IDS = ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")


@dataclass(frozen=True)
class ApprovedRom:
    """Optional explicit range approval for either joint, in degrees."""

    hip_deg: tuple[float, float] | None = None
    knee_deg: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        configured = {"hip_deg": hip_range_deg, "knee_deg": knee_range_deg}
        for name, value in (("hip_deg", self.hip_deg), ("knee_deg", self.knee_deg)):
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != (2,) or not np.isfinite(array).all():
                raise ValueError(f"approved {name} must contain two finite values.")
            if not array[0] < array[1]:
                raise ValueError(f"approved {name} minimum must be below maximum.")
            configured_minimum, configured_maximum = configured[name]
            if (
                array[0] < configured_minimum - 1e-12
                or array[1] > configured_maximum + 1e-12
            ):
                raise ValueError(
                    f"approved {name} must remain within configured ROM "
                    f"[{configured_minimum}, {configured_maximum}] deg."
                )

    def as_dict(self) -> dict[str, list[float] | None]:
        return {
            "hip_deg": None if self.hip_deg is None else list(map(float, self.hip_deg)),
            "knee_deg": (
                None if self.knee_deg is None else list(map(float, self.knee_deg))
            ),
        }


@dataclass(frozen=True)
class RomAudit:
    """Range/mapping decision made before any dynamic evaluation."""

    original_angle_range_deg: dict[str, list[float]]
    configured_angle_range_deg: dict[str, list[float]]
    approved_angle_range_deg: dict[str, list[float] | None]
    effective_angle_range_deg: dict[str, list[float]]
    rom_mapping_applied: bool
    rom_mapping_applied_by_joint: dict[str, bool]
    mapping_formula: dict[str, str | None]
    trajectory_requires_rom_confirmation: bool
    confirmation_reasons: tuple[str, ...]
    dynamics_allowed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceRetimingResult:
    """All in-memory tables and paths produced by one Stage-5B run."""

    source_cycle: pd.DataFrame
    phase_path: pd.DataFrame
    retimed_by_profile: dict[str, pd.DataFrame]
    dynamics_by_profile_subject: dict[str, dict[str, pd.DataFrame]]
    retiming_summary: pd.DataFrame
    subject_comparison: pd.DataFrame
    domain_audit: pd.DataFrame
    rom_audit: RomAudit
    metadata: dict[str, object]
    output_paths: dict[str, Path]
    visualization_paths: dict[str, Path]
    skipped_visualizations: dict[str, str]


def _validate_samples_per_segment(samples_per_segment: int) -> int:
    if isinstance(samples_per_segment, bool) or not isinstance(
        samples_per_segment, (int, np.integer)
    ):
        raise TypeError("samples_per_segment must be an integer.")
    value = int(samples_per_segment)
    if value < 3:
        raise ValueError("samples_per_segment must be at least 3.")
    return value


def _validate_duration(value: float, name: str) -> float:
    duration = float(value)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return duration


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def load_processed_reference_cycle(
    processed_directory: str | Path = reference_trajectory_data_dir,
    *,
    cycle_index: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a complete Stage-5A cycle without reparsing the marker CSV."""

    directory = Path(processed_directory)
    metadata_path = directory / "metadata.json"
    cycles_path = directory / "detected_cycles.csv"
    selected_path = directory / "reference_selected_cycle.csv"
    full_path = directory / "reference_full_angles.csv"
    for path in (metadata_path, cycles_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Stage-5A output is missing: {path}")

    metadata = _load_json(metadata_path)
    cycles = pd.read_csv(cycles_path)
    selected_metadata = metadata.get("selected_cycle", {})
    selected_index = (
        selected_metadata.get("cycle_index")
        if isinstance(selected_metadata, Mapping)
        else None
    )

    if cycle_index is None or (
        selected_index is not None and int(cycle_index) == int(selected_index)
    ):
        cycle = pd.read_csv(selected_path)
        effective_index = selected_index
    else:
        if not full_path.is_file():
            raise FileNotFoundError(
                f"cycle selection requires Stage-5A full angles: {full_path}"
            )
        match = cycles.loc[cycles["cycle_index"].eq(int(cycle_index))]
        if len(match) != 1:
            raise ValueError(f"cycle_index={cycle_index} was not detected exactly once.")
        row = match.iloc[0]
        if not bool(row["cycle_complete"]):
            raise ValueError("an incomplete cycle cannot be used as a Stage-5B reference.")
        if "cycle_continuous" in row and not bool(row["cycle_continuous"]):
            raise ValueError("a discontinuous cycle cannot be retimed safely.")
        full = pd.read_csv(full_path)
        start_frame = int(row["start_frame"])
        peak_frame = int(row["peak_flexion_frame"])
        end_frame = int(row["end_frame"])
        cycle = full.loc[
            full["Frame"].between(start_frame, end_frame, inclusive="both")
        ].copy()
        if cycle.empty or int(cycle["Frame"].iloc[0]) != start_frame or int(
            cycle["Frame"].iloc[-1]
        ) != end_frame:
            raise ValueError("selected processed cycle frame bounds are incomplete.")
        cycle.insert(0, "source_frame", cycle["Frame"].to_numpy(dtype=int))
        cycle["cycle_phase"] = np.where(
            cycle["Frame"].to_numpy(dtype=int) <= peak_frame,
            "flexion",
            "extension",
        )
        cycle["phase"] = cycle["cycle_phase"]
        cycle["trajectory_sample_valid"] = cycle.get(
            "angle_valid", pd.Series(True, index=cycle.index)
        ).fillna(False).astype(bool)
        effective_index = int(cycle_index)
        metadata = dict(metadata)
        metadata["selected_cycle"] = {
            key: _json_ready(value)
            for key, value in row.to_dict().items()
            if key
            in {
                "cycle_index",
                "start_frame",
                "peak_flexion_frame",
                "end_frame",
                "duration_frames",
                "q_hip_range_deg",
                "q_knee_range_deg",
                "cycle_complete",
                "cycle_quality_score",
            }
        }
        metadata["selection_reason"] = "explicit Stage-5B cycle_index selection"

    required = {
        "Frame",
        "cycle_phase",
        "q_hip_raw_rad",
        "q_knee_raw_rad",
        "q_hip_rad",
        "q_knee_rad",
        "x_ankle_observed_m",
        "z_ankle_observed_m",
    }
    missing = required.difference(cycle.columns)
    if missing:
        raise ValueError(f"processed reference cycle missing columns: {sorted(missing)}")
    frame = cycle["Frame"].to_numpy(dtype=float)
    if not np.isfinite(frame).all() or not np.all(np.diff(frame) > 0.0):
        raise ValueError("processed reference Frame must be finite and increasing.")
    if set(cycle["cycle_phase"].astype(str)) != {"flexion", "extension"}:
        raise ValueError("processed cycle must contain flexion and extension.")
    if "source_frame" not in cycle:
        cycle.insert(0, "source_frame", cycle["Frame"].to_numpy(dtype=int))
    cycle = cycle.reset_index(drop=True)
    metadata = dict(metadata)
    metadata["stage5b_cycle_index"] = (
        None if effective_index is None else int(effective_index)
    )
    return cycle, metadata


def _geometric_phase_coordinate(q_hip: np.ndarray, q_knee: np.ndarray) -> np.ndarray:
    points = np.column_stack((q_hip, q_knee))
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("each reference segment needs at least two finite angle samples.")
    increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    if cumulative[-1] <= 1e-12:
        raise ValueError("reference segment has no measurable joint-space excursion.")
    return cumulative / cumulative[-1]


def _strict_pchip_knots(
    coordinate: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average duplicate geometric knots so PCHIP receives a strict axis."""

    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    if coordinate.ndim != 1 or values.ndim != 1 or coordinate.shape != values.shape:
        raise ValueError("PCHIP coordinate and values must be equal-length vectors.")
    groups = np.concatenate(
        ([0], np.flatnonzero(np.diff(coordinate) > 1e-12) + 1, [len(coordinate)])
    )
    knot_x: list[float] = []
    knot_y: list[float] = []
    for start, stop in zip(groups[:-1], groups[1:]):
        if stop <= start:
            continue
        knot_x.append(float(np.mean(coordinate[start:stop])))
        knot_y.append(float(np.mean(values[start:stop])))
    x = np.asarray(knot_x, dtype=float)
    y = np.asarray(knot_y, dtype=float)
    if len(x) < 2 or not np.all(np.diff(x) > 0.0):
        raise ValueError("not enough distinct geometric phase knots for PCHIP.")
    x[0] = 0.0
    x[-1] = 1.0
    return x, y


def _pchip_values(
    coordinate: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    x, y = _strict_pchip_knots(coordinate, values)
    return np.asarray(PchipInterpolator(x, y, extrapolate=False)(target), dtype=float)


def _nearest_source_values(
    coordinate: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    right = np.searchsorted(coordinate, target, side="left")
    right = np.clip(right, 0, len(coordinate) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(coordinate[right] - target) <= np.abs(
        coordinate[left] - target
    )
    indices = np.where(choose_right, right, left)
    return np.asarray(values)[indices]


def build_reference_phase_path(
    selected_cycle: pd.DataFrame,
    *,
    samples_per_segment: int = reference_phase_samples_per_segment,
) -> pd.DataFrame:
    """Resample both geometric branches on local phase ``[0, 1]`` using PCHIP.

    Both copies of the peak are retained in this phase-only table so each branch
    has exactly ``samples_per_segment`` samples.  The duplicate is removed only
    after an actual time axis is assigned.
    """

    count = _validate_samples_per_segment(samples_per_segment)
    rows: list[pd.DataFrame] = []
    flexion_source = selected_cycle.loc[
        selected_cycle["cycle_phase"].astype(str).eq("flexion")
    ].copy()
    if flexion_source.empty:
        raise ValueError("flexion needs at least two source samples.")
    peak_source = flexion_source.iloc[[-1]].copy()
    for phase_name, global_offset in (("flexion", 0.0), ("extension", 0.5)):
        source = selected_cycle.loc[
            selected_cycle["cycle_phase"].astype(str).eq(phase_name)
        ].copy()
        # Stage 5A labels the peak sample as flexion.  Prepending that same row
        # to extension makes q_ref continuous at the branch join; the duplicate
        # is retained in the phase table and removed after assigning time.
        if phase_name == "extension" and (
            source.empty
            or float(source["Frame"].iloc[0]) != float(peak_source["Frame"].iloc[0])
        ):
            source = pd.concat((peak_source, source), ignore_index=True)
        if len(source) < 2:
            raise ValueError(f"{phase_name} needs at least two source samples.")
        q_hip = source["q_hip_rad"].to_numpy(dtype=float)
        q_knee = source["q_knee_rad"].to_numpy(dtype=float)
        coordinate = _geometric_phase_coordinate(q_hip, q_knee)
        target = np.linspace(0.0, 1.0, count)
        frame = source["Frame"].to_numpy(dtype=float)
        raw_hip = source["q_hip_raw_rad"].to_numpy(dtype=float)
        raw_knee = source["q_knee_raw_rad"].to_numpy(dtype=float)
        smooth_hip = _pchip_values(coordinate, q_hip, target)
        smooth_knee = _pchip_values(coordinate, q_knee, target)
        raw_hip_interp = _pchip_values(coordinate, raw_hip, target)
        raw_knee_interp = _pchip_values(coordinate, raw_knee, target)
        frame_interp = _pchip_values(coordinate, frame, target)
        ankle_x = _pchip_values(
            coordinate,
            source["x_ankle_observed_m"].to_numpy(dtype=float),
            target,
        )
        ankle_z = _pchip_values(
            coordinate,
            source["z_ankle_observed_m"].to_numpy(dtype=float),
            target,
        )
        planarity = (
            _pchip_values(
                coordinate,
                source["planarity_error_m"].to_numpy(dtype=float),
                target,
            )
            if "planarity_error_m" in source
            else np.full(count, np.nan)
        )
        source_valid_values = (
            source["angle_valid"].fillna(False).astype(bool).to_numpy()
            if "angle_valid" in source
            else np.ones(len(source), dtype=bool)
        )
        source_valid = _nearest_source_values(
            coordinate, target, source_valid_values
        ).astype(bool)
        global_phase = global_offset + 0.5 * target
        rows.append(
            pd.DataFrame(
                {
                    "global_phase": global_phase,
                    "segment_phase": target,
                    "cycle_phase": phase_name,
                    "source_frame": frame_interp,
                    "q_hip_raw_rad": raw_hip_interp,
                    "q_knee_raw_rad": raw_knee_interp,
                    "q_hip_smoothed_rad": smooth_hip,
                    "q_knee_smoothed_rad": smooth_knee,
                    "theta_shank_rad": smooth_hip - smooth_knee,
                    "x_ankle_observed_m": ankle_x,
                    "z_ankle_observed_m": ankle_z,
                    "planarity_error_m": planarity,
                    "source_angle_valid": source_valid,
                    "source_trajectory_type": SOURCE_TRAJECTORY_TYPE,
                    "source_timing_status": SOURCE_TIMING_STATUS,
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    if not np.allclose(
        output["theta_shank_rad"],
        output["q_hip_smoothed_rad"] - output["q_knee_smoothed_rad"],
        atol=1e-14,
    ):
        raise RuntimeError("theta_shank convention was not preserved.")
    return output


def _range_deg(values: pd.Series | np.ndarray) -> list[float]:
    array = np.rad2deg(np.asarray(values, dtype=float))
    return [float(np.nanmin(array)), float(np.nanmax(array))]


def _outside_range(values_rad: np.ndarray, limits_deg: Sequence[float]) -> np.ndarray:
    lower, upper = np.deg2rad(np.asarray(limits_deg, dtype=float))
    tolerance = 1e-12
    return (values_rad < lower - tolerance) | (values_rad > upper + tolerance)


def apply_approved_rom_mapping(
    phase_path: pd.DataFrame,
    *,
    approved_rom: ApprovedRom | None = None,
) -> tuple[pd.DataFrame, RomAudit]:
    """Apply an optional whole-path affine amplitude map; never pointwise clip."""

    approved = approved_rom or ApprovedRom()
    output = phase_path.copy(deep=True)
    originals = {
        "hip": output["q_hip_smoothed_rad"].to_numpy(dtype=float),
        "knee": output["q_knee_smoothed_rad"].to_numpy(dtype=float),
    }
    configured = {"hip": hip_range_deg, "knee": knee_range_deg}
    explicit = {"hip": approved.hip_deg, "knee": approved.knee_deg}
    mappings: dict[str, bool] = {}
    formulas: dict[str, str | None] = {}
    reasons: list[str] = []
    if "source_angle_valid" in output and not output[
        "source_angle_valid"
    ].fillna(False).astype(bool).all():
        reasons.append("source_angle_invalid")

    for joint in ("hip", "knee"):
        values = originals[joint]
        approved_limits = explicit[joint]
        active_limits = configured[joint] if approved_limits is None else approved_limits
        outside = _outside_range(values, active_limits)
        mapped = False
        if approved_limits is not None and bool(outside.any()):
            raw_min = float(np.min(values))
            raw_max = float(np.max(values))
            if raw_max - raw_min <= 1e-12:
                raise ValueError(f"cannot amplitude-map constant {joint} reference.")
            approved_min, approved_max = np.deg2rad(
                np.asarray(approved_limits, dtype=float)
            )
            normalized = (values - raw_min) / (raw_max - raw_min)
            values = approved_min + normalized * (approved_max - approved_min)
            mapped = True
            formulas[joint] = (
                "q_new = approved_min + "
                "(q_original - original_min) / (original_max - original_min) "
                "* (approved_max - approved_min)"
            )
        else:
            formulas[joint] = None

        if approved_limits is None and bool(outside.any()):
            reasons.append(f"{joint}_outside_configured_rom_without_explicit_approval")
        output[f"q_{joint}_reference_rad"] = values
        output[f"q_{joint}_rom_mapping_applied"] = mapped
        output[f"q_{joint}_approved_min_deg"] = float(active_limits[0])
        output[f"q_{joint}_approved_max_deg"] = float(active_limits[1])
        mappings[joint] = mapped

    output["rom_mapping_applied"] = mappings["hip"] | mappings["knee"]
    output["theta_shank_reference_rad"] = (
        output["q_hip_reference_rad"] - output["q_knee_reference_rad"]
    )
    hip_valid = ~_outside_range(
        output["q_hip_reference_rad"].to_numpy(dtype=float),
        configured["hip"] if explicit["hip"] is None else explicit["hip"],
    )
    knee_valid = ~_outside_range(
        output["q_knee_reference_rad"].to_numpy(dtype=float),
        configured["knee"] if explicit["knee"] is None else explicit["knee"],
    )
    output["joint_limit_valid"] = hip_valid & knee_valid
    output["trajectory_requires_rom_confirmation"] = bool(reasons)
    output["dynamics_allowed"] = not bool(reasons)
    if not reasons and not np.all(output["joint_limit_valid"]):
        raise RuntimeError("ROM mapping did not place the path in its active limits.")

    audit = RomAudit(
        original_angle_range_deg={
            "hip": _range_deg(originals["hip"]),
            "knee": _range_deg(originals["knee"]),
        },
        configured_angle_range_deg={
            "hip": list(map(float, hip_range_deg)),
            "knee": list(map(float, knee_range_deg)),
        },
        approved_angle_range_deg={
            "hip": None if approved.hip_deg is None else list(map(float, approved.hip_deg)),
            "knee": (
                None if approved.knee_deg is None else list(map(float, approved.knee_deg))
            ),
        },
        effective_angle_range_deg={
            "hip": _range_deg(output["q_hip_reference_rad"]),
            "knee": _range_deg(output["q_knee_reference_rad"]),
        },
        rom_mapping_applied=mappings["hip"] or mappings["knee"],
        rom_mapping_applied_by_joint=mappings,
        mapping_formula=formulas,
        trajectory_requires_rom_confirmation=bool(reasons),
        confirmation_reasons=tuple(reasons),
        dynamics_allowed=not bool(reasons),
    )
    return output, audit


def _segment_interpolator(
    phase_path: pd.DataFrame,
    phase_name: str,
    column: str,
) -> PchipInterpolator:
    segment = phase_path.loc[phase_path["cycle_phase"].eq(phase_name)]
    x = segment["segment_phase"].to_numpy(dtype=float)
    y = segment[column].to_numpy(dtype=float)
    return PchipInterpolator(x, y, extrapolate=False)


def retime_reference_path(
    phase_path: pd.DataFrame,
    *,
    profile: str,
    flexion_duration_s: float,
    extension_duration_s: float,
    samples_per_segment: int = reference_phase_samples_per_segment,
) -> pd.DataFrame:
    """Apply minimum jerk to path phase and analytically propagate derivatives.

    The correct chain rule is

    ``q_ddot = q_ss * s_dot**2 + q_s * s_ddot``.

    The plus sign follows directly from differentiating ``q_ref(s(t))``.  The
    direction of extension is already encoded by its path derivative.
    """

    count = _validate_samples_per_segment(samples_per_segment)
    flexion_duration = _validate_duration(flexion_duration_s, "flexion_duration_s")
    extension_duration = _validate_duration(
        extension_duration_s, "extension_duration_s"
    )
    required = {
        "cycle_phase",
        "segment_phase",
        "q_hip_reference_rad",
        "q_knee_reference_rad",
    }
    missing = required.difference(phase_path.columns)
    if missing:
        raise ValueError(f"phase path missing retiming columns: {sorted(missing)}")

    segments: list[pd.DataFrame] = []
    time_offset = 0.0
    for phase_name, duration, global_offset in (
        ("flexion", flexion_duration, 0.0),
        ("extension", extension_duration, 0.5),
    ):
        u = np.linspace(0.0, 1.0, count)
        path_s, path_s_dot, path_s_ddot = minimum_jerk_profile(u, duration)
        local_time = u * duration
        data: dict[str, object] = {
            "profile": profile,
            "time_s": time_offset + local_time,
            "cycle_phase": phase_name,
            "segment_phase": path_s,
            "global_phase": global_offset + 0.5 * path_s,
            "minimum_jerk_phase_rate_s_inv": path_s_dot,
            "minimum_jerk_phase_acceleration_s_inv2": path_s_ddot,
        }
        phase_segment = phase_path.loc[
            phase_path["cycle_phase"].eq(phase_name)
        ]
        if "source_angle_valid" in phase_segment:
            data["source_angle_valid"] = _nearest_source_values(
                phase_segment["segment_phase"].to_numpy(dtype=float),
                np.asarray(path_s, dtype=float),
                phase_segment["source_angle_valid"]
                .fillna(False)
                .astype(bool)
                .to_numpy(),
            ).astype(bool)
        else:
            data["source_angle_valid"] = np.ones(count, dtype=bool)
        for joint in ("hip", "knee"):
            interpolator = _segment_interpolator(
                phase_path, phase_name, f"q_{joint}_reference_rad"
            )
            q = np.asarray(interpolator(path_s), dtype=float)
            q_s = np.asarray(interpolator.derivative(1)(path_s), dtype=float)
            q_ss = np.asarray(interpolator.derivative(2)(path_s), dtype=float)
            data[f"q_{joint}_rad"] = q
            data[f"dq_{joint}_ds_rad"] = q_s
            data[f"d2q_{joint}_ds2_rad"] = q_ss
            data[f"dq_{joint}_rad_s"] = q_s * path_s_dot
            data[f"ddq_{joint}_rad_s2"] = (
                q_ss * path_s_dot**2 + q_s * path_s_ddot
            )
            for source_column, target_column in (
                (f"q_{joint}_raw_rad", f"q_{joint}_raw_rad"),
                (f"q_{joint}_smoothed_rad", f"q_{joint}_smoothed_rad"),
            ):
                source_interpolator = _segment_interpolator(
                    phase_path, phase_name, source_column
                )
                data[target_column] = np.asarray(
                    source_interpolator(path_s), dtype=float
                )
        for column in (
            "source_frame",
            "x_ankle_observed_m",
            "z_ankle_observed_m",
            "planarity_error_m",
        ):
            if column in phase_path:
                interpolator = _segment_interpolator(phase_path, phase_name, column)
                data[column] = np.asarray(interpolator(path_s), dtype=float)
        segments.append(pd.DataFrame(data))
        time_offset += duration

    # Keep the flexion peak only once so the time axis is strictly increasing.
    output = pd.concat((segments[0], segments[1].iloc[1:]), ignore_index=True)
    q_hip = output["q_hip_rad"].to_numpy(dtype=float)
    q_knee = output["q_knee_rad"].to_numpy(dtype=float)
    output["theta_shank_rad"] = q_hip - q_knee
    output["source_timing_status"] = SOURCE_TIMING_STATUS
    output["retimed_trajectory"] = True
    output["retimed_timing_is_original"] = False
    output["source_trajectory_type"] = SOURCE_TRAJECTORY_TYPE
    output["simulation_status"] = SIMULATION_STATUS
    output["observed_ankle_is_pull_point"] = False
    output["rom_mapping_applied"] = bool(phase_path["rom_mapping_applied"].iloc[0])
    output["trajectory_requires_rom_confirmation"] = bool(
        phase_path["trajectory_requires_rom_confirmation"].iloc[0]
    )
    output["dynamics_allowed"] = bool(phase_path["dynamics_allowed"].iloc[0])

    active_hip = (
        float(phase_path["q_hip_approved_min_deg"].iloc[0]),
        float(phase_path["q_hip_approved_max_deg"].iloc[0]),
    )
    active_knee = (
        float(phase_path["q_knee_approved_min_deg"].iloc[0]),
        float(phase_path["q_knee_approved_max_deg"].iloc[0]),
    )
    output["joint_limit_valid"] = ~(
        _outside_range(q_hip, active_hip) | _outside_range(q_knee, active_knee)
    )
    finite_state = np.isfinite(
        output[
            [
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    # Geometric/state validity is deliberately separate from ROM approval.
    # Out-of-ROM samples remain in the phase/retimed tables and domain audit;
    # the run-level ROM gate (not deletion) prevents dynamic evaluation.
    source_valid = output["source_angle_valid"].fillna(False).astype(bool).to_numpy()
    output["trajectory_sample_valid"] = (
        finite_state
        & output["joint_limit_valid"].astype(bool).to_numpy()
        & source_valid
    )
    invalid_reason = np.where(
        finite_state,
        "",
        "non_finite_retimed_state",
    ).astype(object)
    invalid_reason = np.where(
        output["joint_limit_valid"].to_numpy(dtype=bool),
        invalid_reason,
        np.where(
            invalid_reason == "",
            "outside_active_rom",
            invalid_reason.astype(str) + ";outside_active_rom",
        ),
    )
    invalid_reason = np.where(
        source_valid,
        invalid_reason,
        np.where(
            invalid_reason == "",
            "source_angle_invalid",
            invalid_reason.astype(str) + ";source_angle_invalid",
        ),
    )
    output["invalid_reason"] = invalid_reason.astype(str)

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    output["x_knee_m"] = x_knee
    output["z_knee_m"] = z_knee
    output["x_pull_m"] = x_pull
    output["z_pull_m"] = z_pull
    output["L1_m"] = L1
    output["L2_m"] = L2
    output["L2_definition"] = "knee_to_strap_equivalent_pull_point"

    if not np.all(np.diff(output["time_s"].to_numpy(dtype=float)) > 0.0):
        raise RuntimeError("retimed trajectory time must be strictly increasing.")
    endpoint_state = output.iloc[[0, -1]][
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    if not np.allclose(endpoint_state, 0.0, atol=1e-12):
        raise RuntimeError("minimum-jerk start/end dq and ddq must be zero.")
    if not np.allclose(
        output["theta_shank_rad"], q_hip - q_knee, atol=1e-14
    ):
        raise RuntimeError("theta_shank convention was not preserved.")
    return output


def _training_domain_bounds(subject_id: str) -> StateDomainBounds:
    path = identification_data_dir / subject_id / "clean" / "training_data.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"identification-domain training data are missing for {subject_id}: {path}"
        )
    canonical = pd.read_csv(
        path,
        usecols=[
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
            "sample_valid",
            "force_mapping_valid",
            "dataset_split",
            "trajectory_id",
        ],
    )
    if not canonical["dataset_split"].astype(str).eq("train").all():
        raise ValueError(
            f"{path} contains non-train rows; validation/test cannot define "
            "the identification domain."
        )
    if not canonical["trajectory_id"].astype(str).eq(
        identification_trajectory_id
    ).all():
        raise ValueError(
            f"{path} contains an unexpected trajectory_id for domain fitting."
        )
    rename = dict(
        zip(
            (
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            ),
            ESTIMATED_DOMAIN_STATE_COLUMNS,
        )
    )
    training = canonical.rename(columns=rename)
    training["state_estimation_valid"] = (
        training["sample_valid"].fillna(False).astype(bool)
        & training["force_mapping_valid"].fillna(False).astype(bool)
    )
    return fit_state_domain_bounds(training)


def _domain_reason(dataframe: pd.DataFrame, bounds: StateDomainBounds) -> np.ndarray:
    values = dataframe.loc[:, bounds.columns].to_numpy(dtype=float)
    lower = np.asarray(bounds.lower, dtype=float)
    upper = np.asarray(bounds.upper, dtype=float)
    reasons = np.full(len(dataframe), "", dtype=object)
    for row_index, row in enumerate(values):
        tokens: list[str] = []
        for column, value, minimum, maximum in zip(
            bounds.columns, row, lower, upper
        ):
            if not np.isfinite(value):
                tokens.append(f"non_finite_{column}")
            elif value < minimum:
                tokens.append(f"{column}_below_training_bound")
            elif value > maximum:
                tokens.append(f"{column}_above_training_bound")
        reasons[row_index] = ";".join(tokens)
    return reasons.astype(str)


def build_reference_domain_audit(
    retimed_by_profile: Mapping[str, pd.DataFrame],
    *,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> tuple[pd.DataFrame, dict[str, StateDomainBounds]]:
    """Classify six-dimensional state coverage using existing train-only bounds."""

    frames: list[pd.DataFrame] = []
    bounds_by_subject: dict[str, StateDomainBounds] = {}
    state_mapping = dict(
        zip(
            ESTIMATED_DOMAIN_STATE_COLUMNS,
            (
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            ),
        )
    )
    for subject_id in subject_ids:
        bounds = _training_domain_bounds(subject_id)
        bounds_by_subject[subject_id] = bounds
        for profile, trajectory in retimed_by_profile.items():
            estimated = pd.DataFrame(
                {
                    domain_column: trajectory[source_column].to_numpy(dtype=float)
                    for domain_column, source_column in state_mapping.items()
                }
            )
            finite = np.isfinite(estimated.to_numpy(dtype=float)).all(axis=1)
            estimated["state_estimation_valid"] = (
                finite
                & trajectory["source_angle_valid"].fillna(False).astype(bool)
            )
            membership = classify_state_domain(estimated, bounds)
            reason = _domain_reason(estimated, bounds)
            reason = np.where(
                estimated["state_estimation_valid"].to_numpy(dtype=bool)
                | (reason != ""),
                reason,
                "state_estimation_invalid",
            )
            reason = np.where(membership, "", reason)
            frames.append(
                pd.DataFrame(
                    {
                        "profile": profile,
                        "subject_id": subject_id,
                        "time_s": trajectory["time_s"].to_numpy(dtype=float),
                        "global_phase": trajectory["global_phase"].to_numpy(dtype=float),
                        "cycle_phase": trajectory["cycle_phase"].astype(str).to_numpy(),
                        "q_hip_rad": trajectory["q_hip_rad"].to_numpy(dtype=float),
                        "q_knee_rad": trajectory["q_knee_rad"].to_numpy(dtype=float),
                        "dq_hip_rad_s": trajectory["dq_hip_rad_s"].to_numpy(dtype=float),
                        "dq_knee_rad_s": trajectory["dq_knee_rad_s"].to_numpy(dtype=float),
                        "ddq_hip_rad_s2": trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float),
                        "ddq_knee_rad_s2": trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float),
                        "joint_limit_valid": trajectory["joint_limit_valid"]
                        .astype(bool)
                        .to_numpy(),
                        "state_estimation_valid": estimated[
                            "state_estimation_valid"
                        ].to_numpy(dtype=bool),
                        "domain_membership_estimated": membership,
                        "domain_invalid_reason": reason.astype(str),
                        "domain_model": DOMAIN_MODEL,
                        "domain_training_sample_count": bounds.valid_training_samples,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True), bounds_by_subject


def evaluate_reference_dynamics(
    retimed_by_profile: Mapping[str, pd.DataFrame],
    domain_audit: pd.DataFrame,
    *,
    dynamics_allowed: bool,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run existing inverse dynamics and force mapping after the ROM gate."""

    if not dynamics_allowed:
        return {}
    all_results: dict[str, dict[str, pd.DataFrame]] = {}
    for profile, trajectory in retimed_by_profile.items():
        subject_results: dict[str, pd.DataFrame] = {}
        for subject_id in subject_ids:
            if subject_id not in DYNAMIC_SUBJECTS:
                raise ValueError(f"unknown dynamic subject {subject_id!r}.")
            subject = get_dynamic_subject(subject_id)
            q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
            q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
            dynamics = inverse_dynamics(
                q_hip,
                q_knee,
                trajectory["dq_hip_rad_s"].to_numpy(dtype=float),
                trajectory["dq_knee_rad_s"].to_numpy(dtype=float),
                trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float),
                trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float),
                subject,
                L1,
            )
            force = endpoint_force_from_joint_torque(
                q_hip,
                q_knee,
                dynamics.tau_total_hip_nm,
                dynamics.tau_total_knee_nm,
                L1,
                L2,
            )
            output = trajectory.copy(deep=True)
            output.insert(1, "subject_id", subject_id)
            for name, values in asdict(dynamics).items():
                output[name] = values
            output["fx_robot_on_leg_n"] = force.fx_robot_on_leg_n
            output["fz_robot_on_leg_n"] = force.fz_robot_on_leg_n
            output["force_magnitude_n"] = force.force_magnitude_n
            output["jacobian_determinant"] = force.jacobian_determinant
            output["jacobian_condition_number"] = force.jacobian_condition_number
            output["jacobian_near_singular"] = force.jacobian_near_singular
            output["force_mapping_valid"] = force.force_mapping_valid
            output["force_mapping_invalid_reason"] = force.invalid_reason
            domain = domain_audit.loc[
                domain_audit["profile"].eq(profile)
                & domain_audit["subject_id"].eq(subject_id)
            ].reset_index(drop=True)
            if len(domain) != len(output):
                raise RuntimeError("domain audit and dynamic trajectory length differ.")
            output["domain_membership_estimated"] = domain[
                "domain_membership_estimated"
            ].to_numpy(dtype=bool)
            output["domain_invalid_reason"] = domain[
                "domain_invalid_reason"
            ].astype(str).to_numpy()
            output["domain_model"] = DOMAIN_MODEL
            finite_torque = np.isfinite(
                output[["tau_total_hip_nm", "tau_total_knee_nm"]].to_numpy(float)
            ).all(axis=1)
            output["dynamic_sample_valid"] = (
                output["trajectory_sample_valid"].astype(bool)
                & output["joint_limit_valid"].astype(bool)
                & output["force_mapping_valid"].astype(bool)
                & finite_torque
            )
            output["trajectory_executable_sample"] = (
                output["dynamic_sample_valid"].astype(bool)
                & output["domain_membership_estimated"].astype(bool)
            )
            output["clinical_validation_status"] = "not_clinically_validated"
            subject_results[subject_id] = output
        all_results[profile] = subject_results
    return all_results


def _finite_peak_abs(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = np.abs(array[np.isfinite(array)])
    return float(np.max(array)) if array.size else np.nan


def _finite_rms(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(array**2))) if array.size else np.nan


def build_retiming_summary(
    retimed_by_profile: Mapping[str, pd.DataFrame],
    *,
    durations_by_profile: Mapping[str, Mapping[str, float]],
    rom_audit: RomAudit,
) -> pd.DataFrame:
    rows = []
    for profile, trajectory in retimed_by_profile.items():
        duration = durations_by_profile[profile]
        q_hip_closure_deg = float(
            np.rad2deg(
                trajectory["q_hip_rad"].iloc[-1]
                - trajectory["q_hip_rad"].iloc[0]
            )
        )
        q_knee_closure_deg = float(
            np.rad2deg(
                trajectory["q_knee_rad"].iloc[-1]
                - trajectory["q_knee_rad"].iloc[0]
            )
        )
        pull_closure_m = float(
            np.hypot(
                trajectory["x_pull_m"].iloc[-1]
                - trajectory["x_pull_m"].iloc[0],
                trajectory["z_pull_m"].iloc[-1]
                - trajectory["z_pull_m"].iloc[0],
            )
        )
        rows.append(
            {
                "profile": profile,
                "flexion_duration_s": float(duration["flexion"]),
                "extension_duration_s": float(duration["extension"]),
                "total_duration_s": float(trajectory["time_s"].iloc[-1]),
                "sample_count": int(len(trajectory)),
                "q_hip_min_deg": _range_deg(trajectory["q_hip_rad"])[0],
                "q_hip_max_deg": _range_deg(trajectory["q_hip_rad"])[1],
                "q_knee_min_deg": _range_deg(trajectory["q_knee_rad"])[0],
                "q_knee_max_deg": _range_deg(trajectory["q_knee_rad"])[1],
                "peak_abs_dq_hip_rad_s": _finite_peak_abs(
                    trajectory["dq_hip_rad_s"]
                ),
                "peak_abs_dq_knee_rad_s": _finite_peak_abs(
                    trajectory["dq_knee_rad_s"]
                ),
                "peak_abs_ddq_hip_rad_s2": _finite_peak_abs(
                    trajectory["ddq_hip_rad_s2"]
                ),
                "peak_abs_ddq_knee_rad_s2": _finite_peak_abs(
                    trajectory["ddq_knee_rad_s2"]
                ),
                "cycle_closure_q_hip_error_deg": q_hip_closure_deg,
                "cycle_closure_q_knee_error_deg": q_knee_closure_deg,
                "cycle_closure_pull_error_m": pull_closure_m,
                "cycle_is_closed_for_repetition": bool(
                    abs(q_hip_closure_deg) <= 1e-6
                    and abs(q_knee_closure_deg) <= 1e-6
                ),
                "rom_mapping_applied": rom_audit.rom_mapping_applied,
                "trajectory_requires_rom_confirmation": (
                    rom_audit.trajectory_requires_rom_confirmation
                ),
                "dynamics_allowed": rom_audit.dynamics_allowed,
                "source_timing_status": SOURCE_TIMING_STATUS,
                "retimed_trajectory": True,
                "retimed_timing_is_original": False,
            }
        )
    return pd.DataFrame(rows)


def build_subject_comparison(
    retimed_by_profile: Mapping[str, pd.DataFrame],
    dynamics_by_profile_subject: Mapping[str, Mapping[str, pd.DataFrame]],
    domain_audit: pd.DataFrame,
    *,
    dynamics_allowed: bool,
    block_reason: str,
    subject_ids: Sequence[str] = SUBJECT_IDS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile, trajectory in retimed_by_profile.items():
        for subject_id in subject_ids:
            domain = domain_audit.loc[
                domain_audit["profile"].eq(profile)
                & domain_audit["subject_id"].eq(subject_id)
            ]
            membership = domain["domain_membership_estimated"].astype(bool)
            out_count = int((~membership).sum())
            out_percent = 100.0 * out_count / len(domain) if len(domain) else np.nan
            dynamic = dynamics_by_profile_subject.get(profile, {}).get(subject_id)
            if dynamic is None:
                metrics = {
                    "peak_abs_tau_hip_nm": np.nan,
                    "peak_abs_tau_knee_nm": np.nan,
                    "rms_combined_torque_nm": np.nan,
                    "peak_force_n": np.nan,
                    "rms_force_n": np.nan,
                    "maximum_jacobian_condition": np.nan,
                    "invalid_force_sample_count": 0,
                    "valid_force_sample_count": 0,
                }
                executable = False
            else:
                valid = dynamic["dynamic_sample_valid"].astype(bool).to_numpy()
                torque_hip = dynamic["tau_total_hip_nm"].to_numpy(dtype=float)[valid]
                torque_knee = dynamic["tau_total_knee_nm"].to_numpy(dtype=float)[valid]
                force = dynamic["force_magnitude_n"].to_numpy(dtype=float)[valid]
                condition = dynamic["jacobian_condition_number"].to_numpy(dtype=float)
                finite_condition = condition[np.isfinite(condition)]
                metrics = {
                    "peak_abs_tau_hip_nm": _finite_peak_abs(torque_hip),
                    "peak_abs_tau_knee_nm": _finite_peak_abs(torque_knee),
                    "rms_combined_torque_nm": (
                        float(np.sqrt(np.mean(torque_hip**2 + torque_knee**2)))
                        if len(torque_hip)
                        else np.nan
                    ),
                    "peak_force_n": _finite_peak_abs(force),
                    "rms_force_n": _finite_rms(force),
                    "maximum_jacobian_condition": (
                        float(np.max(finite_condition))
                        if finite_condition.size
                        else np.nan
                    ),
                    "invalid_force_sample_count": int(
                        (~dynamic["force_mapping_valid"].astype(bool)).sum()
                    ),
                    "valid_force_sample_count": int(valid.sum()),
                }
                executable = bool(
                    dynamic["trajectory_executable_sample"].astype(bool).all()
                )
            rows.append(
                {
                    "profile": profile,
                    "subject_id": subject_id,
                    **metrics,
                    "out_of_domain_sample_count": out_count,
                    "out_of_domain_percent": out_percent,
                    "dynamics_evaluated": dynamic is not None,
                    "dynamics_allowed": dynamics_allowed,
                    "dynamics_block_reason": "" if dynamic is not None else block_reason,
                    "trajectory_executable": executable,
                    "trajectory_sample_count": int(len(trajectory)),
                }
            )
    return pd.DataFrame(rows)


def _resolve_profiles(profiles: Iterable[str] | None) -> tuple[str, ...]:
    requested = DEFAULT_PROFILES if profiles is None else tuple(profiles)
    if not requested:
        raise ValueError("at least one retiming profile is required.")
    invalid = [name for name in requested if name not in reference_retiming_durations_s]
    if invalid:
        raise ValueError(f"unknown retiming profiles: {invalid}")
    if len(set(requested)) != len(requested):
        raise ValueError("retiming profiles must be unique.")
    return requested


def _resolve_durations(
    profiles: Sequence[str],
    *,
    flexion_duration_s: float | None,
    extension_duration_s: float | None,
) -> dict[str, dict[str, float]]:
    if (flexion_duration_s is not None or extension_duration_s is not None) and len(
        profiles
    ) != 1:
        raise ValueError("duration overrides require exactly one --profile.")
    resolved: dict[str, dict[str, float]] = {}
    for profile in profiles:
        defaults = reference_retiming_durations_s[profile]
        resolved[profile] = {
            "flexion": _validate_duration(
                defaults["flexion"]
                if flexion_duration_s is None
                else flexion_duration_s,
                "flexion_duration_s",
            ),
            "extension": _validate_duration(
                defaults["extension"]
                if extension_duration_s is None
                else extension_duration_s,
                "extension_duration_s",
            ),
        }
    return resolved


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, StateDomainBounds):
        return value.as_serializable_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _build_metadata(
    *,
    source_metadata: Mapping[str, object],
    source_processed_directory: str | Path,
    phase_path: pd.DataFrame,
    retiming_summary: pd.DataFrame,
    rom_audit: RomAudit,
    durations_by_profile: Mapping[str, Mapping[str, float]],
    bounds_by_subject: Mapping[str, StateDomainBounds],
    dynamics_by_profile_subject: Mapping[str, Mapping[str, pd.DataFrame]],
    generated_files: Sequence[str],
) -> dict[str, object]:
    selected_cycle = source_metadata.get("selected_cycle", {})
    metadata: dict[str, object] = {
        "stage": "5B_reference_path_retiming_and_dynamics_evaluation",
        "model_version": reference_retiming_model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stage": "5A_processed_reference_trajectory",
        "source_processed_directory": str(Path(source_processed_directory)),
        "source_cycle": selected_cycle,
        "cycle_index": source_metadata.get("stage5b_cycle_index"),
        "source_coordinate_unit": source_metadata.get("source_coordinate_unit"),
        "source_fps": source_metadata.get("fps"),
        "source_timing_status": SOURCE_TIMING_STATUS,
        "retimed_trajectory": True,
        "retimed_timing_is_original": False,
        "retimed_timing_warning": (
            "prescribed Stage-5B durations are not the original skeleton motion speed"
        ),
        "profiles": list(durations_by_profile),
        "retimed_durations_s": durations_by_profile,
        "phase_parameterization": "joint_space_geometric_arc_length_per_segment",
        "phase_interpolation": "PCHIP_shape_preserving",
        "phase_interpolation_continuity": (
            "C1_piecewise_cubic; second path derivative may change at knots"
        ),
        "samples_per_segment": int(
            phase_path.groupby("cycle_phase", sort=False).size().iloc[0]
        ),
        "minimum_jerk_controls": "path_phase_not_joint_endpoint_line",
        "chain_rule_formula": (
            "ddq = d2q_ds2 * ds_dt**2 + dq_ds * d2s_dt2"
        ),
        "chain_rule_sign_note": (
            "plus is the mathematical derivative of q_ref(s(t)); extension "
            "direction is already carried by dq/ds"
        ),
        "model_angle_definition": MODEL_ANGLE_DEFINITION,
        "L1_m": L1,
        "L2_m": L2,
        "L2_definition": "knee_to_strap_equivalent_pull_point",
        "observed_ankle_retained_for_comparison": True,
        "observed_ankle_is_pull_point": False,
        "approved_rom_policy": (
            "explicit mapping target must remain within configured model ROM"
        ),
        "rom_audit": rom_audit.as_dict(),
        "trajectory_requires_rom_confirmation": (
            rom_audit.trajectory_requires_rom_confirmation
        ),
        "dynamics_allowed": rom_audit.dynamics_allowed,
        "dynamics_evaluated": bool(dynamics_by_profile_subject),
        "dynamic_subject_ids": list(SUBJECT_IDS),
        "identification_domain_model": DOMAIN_MODEL,
        "identification_domain_bounds_by_subject": bounds_by_subject,
        "domain_is_clinical_safety_domain": False,
        "rms_combined_torque_definition": (
            "sqrt(mean(tau_hip_nm**2 + tau_knee_nm**2))"
        ),
        "source_trajectory_type": SOURCE_TRAJECTORY_TYPE,
        "simulation_status": SIMULATION_STATUS,
        "clinical_validation_status": "not_clinically_validated",
        "planarity_rmse_m": source_metadata.get("planarity_rmse_m"),
        "planarity_max_error_m": source_metadata.get("planarity_max_error_m"),
        "geometry_uncertainty_remains": True,
        "real_robot_code_used": False,
        "real_robot_code_modified": False,
        "hardware_used": False,
        "generated_files": list(generated_files),
        "retiming_summary": retiming_summary.to_dict(orient="records"),
    }
    return _json_ready(metadata)  # type: ignore[return-value]


def run_reference_retiming(
    *,
    processed_directory: str | Path = reference_trajectory_data_dir,
    output_directory: str | Path = reference_retiming_data_dir,
    cycle_index: int | None = None,
    profiles: Iterable[str] | None = None,
    flexion_duration_s: float | None = None,
    extension_duration_s: float | None = None,
    approved_hip_range_deg: tuple[float, float] | None = None,
    approved_knee_range_deg: tuple[float, float] | None = None,
    samples_per_segment: int = reference_phase_samples_per_segment,
    save_outputs: bool = True,
    generate_plots: bool = True,
) -> ReferenceRetimingResult:
    """Run the complete Stage-5B offline, software-only workflow."""

    selected_profiles = _resolve_profiles(profiles)
    durations = _resolve_durations(
        selected_profiles,
        flexion_duration_s=flexion_duration_s,
        extension_duration_s=extension_duration_s,
    )
    source_cycle, source_metadata = load_processed_reference_cycle(
        processed_directory, cycle_index=cycle_index
    )
    raw_phase_path = build_reference_phase_path(
        source_cycle, samples_per_segment=samples_per_segment
    )
    phase_path, rom_audit = apply_approved_rom_mapping(
        raw_phase_path,
        approved_rom=ApprovedRom(
            hip_deg=approved_hip_range_deg,
            knee_deg=approved_knee_range_deg,
        ),
    )
    retimed = {
        profile: retime_reference_path(
            phase_path,
            profile=profile,
            flexion_duration_s=durations[profile]["flexion"],
            extension_duration_s=durations[profile]["extension"],
            samples_per_segment=samples_per_segment,
        )
        for profile in selected_profiles
    }
    domain_audit, bounds = build_reference_domain_audit(retimed)
    dynamics = evaluate_reference_dynamics(
        retimed,
        domain_audit,
        dynamics_allowed=rom_audit.dynamics_allowed,
    )
    retiming_summary = build_retiming_summary(
        retimed, durations_by_profile=durations, rom_audit=rom_audit
    )
    block_reason = (
        ";".join(rom_audit.confirmation_reasons)
        if rom_audit.confirmation_reasons
        else ""
    )
    comparison = build_subject_comparison(
        retimed,
        dynamics,
        domain_audit,
        dynamics_allowed=rom_audit.dynamics_allowed,
        block_reason=block_reason,
    )

    output_dir = Path(output_directory)
    output_paths: dict[str, Path] = {}
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        tables: dict[str, pd.DataFrame] = {
            "reference_path_phase.csv": phase_path,
            "reference_retiming_summary.csv": retiming_summary,
            "reference_subject_comparison.csv": comparison,
            "reference_domain_audit.csv": domain_audit,
        }
        for profile, trajectory in retimed.items():
            tables[f"reference_trajectory_retimed_{profile}.csv"] = trajectory
        for profile, subjects in dynamics.items():
            for subject_id, dataframe in subjects.items():
                tables[f"reference_dynamic_{profile}_{subject_id}.csv"] = dataframe
        for filename, dataframe in tables.items():
            path = output_dir / filename
            dataframe.to_csv(path, index=False)
            output_paths[filename] = path

    metadata = _build_metadata(
        source_metadata=source_metadata,
        source_processed_directory=processed_directory,
        phase_path=phase_path,
        retiming_summary=retiming_summary,
        rom_audit=rom_audit,
        durations_by_profile=durations,
        bounds_by_subject=bounds,
        dynamics_by_profile_subject=dynamics,
        generated_files=sorted(output_paths),
    )
    if save_outputs:
        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        output_paths["metadata.json"] = metadata_path
        metadata["generated_files"] = sorted(output_paths)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)

    visualization_paths: dict[str, Path] = {}
    skipped_visualizations: dict[str, str] = {}
    if generate_plots:
        from .visualize_reference_retiming import (
            generate_reference_retiming_visualizations,
        )

        visual = generate_reference_retiming_visualizations(
            phase_path=phase_path,
            retimed_by_profile=retimed,
            dynamics_by_profile_subject=dynamics,
            comparison=comparison,
            domain_audit=domain_audit,
            metadata=metadata,
            output_directory=output_dir,
        )
        visualization_paths = dict(visual.paths)
        skipped_visualizations = dict(visual.skipped)
        output_paths.update(visualization_paths)
        metadata["generated_files"] = sorted(output_paths)
        metadata["skipped_visualizations"] = skipped_visualizations
        if save_outputs:
            with output_paths["metadata.json"].open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)

    return ReferenceRetimingResult(
        source_cycle=source_cycle,
        phase_path=phase_path,
        retimed_by_profile=retimed,
        dynamics_by_profile_subject=dynamics,
        retiming_summary=retiming_summary,
        subject_comparison=comparison,
        domain_audit=domain_audit,
        rom_audit=rom_audit,
        metadata=metadata,
        output_paths=output_paths,
        visualization_paths=visualization_paths,
        skipped_visualizations=skipped_visualizations,
    )


__all__ = [
    "ApprovedRom",
    "DEFAULT_PROFILES",
    "DOMAIN_MODEL",
    "MODEL_ANGLE_DEFINITION",
    "ReferenceRetimingResult",
    "RomAudit",
    "SOURCE_TIMING_STATUS",
    "SUBJECT_IDS",
    "apply_approved_rom_mapping",
    "build_reference_domain_audit",
    "build_reference_phase_path",
    "build_retiming_summary",
    "build_subject_comparison",
    "evaluate_reference_dynamics",
    "load_processed_reference_cycle",
    "retime_reference_path",
    "run_reference_retiming",
]
