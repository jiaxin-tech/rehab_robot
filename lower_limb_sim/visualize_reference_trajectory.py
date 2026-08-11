"""Stage 5A reference-trajectory figures and frame-based leg animation.

The functions in this module are deliberately presentation-only.  They never
estimate missing dynamics: when source FPS or dynamic columns are unavailable,
the corresponding output is returned in ``skipped`` with an explicit reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any


_MPL_CONFIG_DIRECTORY = Path(tempfile.gettempdir()) / "lower_limb_sim_matplotlib"
_MPL_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIRECTORY))

import matplotlib

matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import L1, L2, hip_range_deg, knee_range_deg
from .kinematics import forward_kinematics


FIGURE_FILENAMES = (
    "3d_right_leg_trajectory.png",
    "sagittal_plane_projection.png",
    "raw_and_filtered_joint_angles.png",
    "detected_cycles.png",
    "selected_cycle_angles.png",
    "selected_cycle_pull_path.png",
    "selected_cycle_leg_animation.gif",
    "reference_torque_components.png",
    "reference_endpoint_force.png",
    "reference_subject_comparison.png",
)

GEOMETRY_FIGURES = FIGURE_FILENAMES[:7]
DYNAMIC_FIGURES = FIGURE_FILENAMES[7:]

_COLORS = {
    "hip": "#4C78A8",
    "knee": "#F58518",
    "shank": "#54A24B",
    "pull": "#B279A2",
    "ankle": "#72B7B2",
    "raw": "#8A8A8A",
    "invalid": "#E45756",
    "bed": "#222222",
}


@dataclass(frozen=True)
class ReferenceTrajectoryVisualizationResult:
    """Generated paths and explicit reasons for omitted requested outputs."""

    paths: dict[str, Path]
    skipped: dict[str, str]

    @property
    def generated_paths(self) -> tuple[Path, ...]:
        """Return generated files in the requested canonical order."""

        return tuple(
            self.paths[name] for name in FIGURE_FILENAMES if name in self.paths
        )

    @property
    def all_requested_outputs_accounted_for(self) -> bool:
        return set(self.paths) | set(self.skipped) == set(FIGURE_FILENAMES)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _as_dataframe(value: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame or None.")
    return value.copy(deep=False)


def _numeric(dataframe: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    for column in candidates:
        if column in dataframe.columns:
            return pd.to_numeric(dataframe[column], errors="coerce").to_numpy(
                dtype=float
            )
    return None


def _first_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _angle_degrees(
    dataframe: pd.DataFrame,
    *,
    degree_candidates: Sequence[str],
    radian_candidates: Sequence[str],
) -> np.ndarray | None:
    value = _numeric(dataframe, degree_candidates)
    if value is not None:
        return value
    value = _numeric(dataframe, radian_candidates)
    if value is not None:
        return np.rad2deg(value)
    return None


def _angle_radians(
    dataframe: pd.DataFrame,
    *,
    radian_candidates: Sequence[str],
    degree_candidates: Sequence[str],
) -> np.ndarray | None:
    value = _numeric(dataframe, radian_candidates)
    if value is not None:
        return value
    value = _numeric(dataframe, degree_candidates)
    if value is not None:
        return np.deg2rad(value)
    return None


def _frames(dataframe: pd.DataFrame) -> np.ndarray:
    values = _numeric(dataframe, ("Frame", "frame", "frame_index", "sample_index"))
    if values is None:
        return np.arange(len(dataframe), dtype=float)
    return values


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for nested_key in ("video", "source", "import", "trajectory"):
        nested = metadata.get(nested_key)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def _source_fps(metadata: Mapping[str, Any]) -> float | None:
    value = _metadata_value(
        metadata,
        "fps",
        "video_fps",
        "frames_per_second",
        "source_fps",
    )
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if np.isfinite(fps) and fps > 0.0 else None


def _geometry_length(metadata: Mapping[str, Any], which: str) -> float:
    default = L1 if which == "L1" else L2
    value = _metadata_value(
        metadata,
        which,
        f"{which}_m",
        f"{which.lower()}_m",
        f"{which}_assumed_m",
        f"{which}_pull_point_m",
    )
    try:
        length = float(value)
    except (TypeError, ValueError):
        return float(default)
    return length if np.isfinite(length) and length > 0.0 else float(default)


def _joint_limits(metadata: Mapping[str, Any], joint: str) -> tuple[float, float]:
    fallback = hip_range_deg if joint == "hip" else knee_range_deg
    value = _metadata_value(
        metadata,
        f"{joint}_range_deg",
        f"q_{joint}_range_deg",
        f"{joint}_joint_limits_deg",
    )
    if value is None:
        joint_limits = metadata.get("joint_limits_deg")
        if isinstance(joint_limits, Mapping):
            value = joint_limits.get(joint) or joint_limits.get(f"q_{joint}")
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError):
        lower, upper = (float(item) for item in fallback)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        lower, upper = (float(item) for item in fallback)
    return lower, upper


def _boolean_mask(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    *,
    default: bool,
) -> np.ndarray:
    column = _first_column(dataframe, candidates)
    if column is None:
        return np.full(len(dataframe), default, dtype=bool)
    series = dataframe[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).to_numpy(dtype=float) != 0.0
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(("true", "yes", "valid", "complete", "1")).to_numpy()


def _finite_rows(*arrays: np.ndarray | None) -> np.ndarray:
    available = [np.asarray(array) for array in arrays if array is not None]
    if not available:
        return np.empty(0, dtype=bool)
    mask = np.ones(len(available[0]), dtype=bool)
    for array in available:
        if len(array) != len(mask):
            raise ValueError("Plotting arrays must have the same length.")
        mask &= np.isfinite(array)
    return mask


def _save_figure(
    figure: plt.Figure,
    output_directory: Path,
    filename: str,
) -> Path:
    path = output_directory / filename
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _skip(
    skipped: dict[str, str],
    filename: str,
    reason: str,
) -> None:
    skipped[filename] = reason


def _plot_original_3d(
    raw: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    primary_leg = str(_metadata_value(metadata, "primary_motion_leg") or "right").lower()
    prefix = "L" if primary_leg == "left" else "R"
    readable_side = "Left" if prefix == "L" else "Right"
    lowercase_side = readable_side.lower()
    candidates = {
        "hip": (
            (f"{prefix}Hip_X", f"{lowercase_side}_hip_x_m", f"{prefix.lower()}hip_x_m"),
            (f"{prefix}Hip_Y", f"{lowercase_side}_hip_y_m", f"{prefix.lower()}hip_y_m"),
            (f"{prefix}Hip_Z", f"{lowercase_side}_hip_z_m", f"{prefix.lower()}hip_z_m"),
        ),
        "knee": (
            (f"{prefix}Knee_X", f"{lowercase_side}_knee_x_m", f"{prefix.lower()}knee_x_m"),
            (f"{prefix}Knee_Y", f"{lowercase_side}_knee_y_m", f"{prefix.lower()}knee_y_m"),
            (f"{prefix}Knee_Z", f"{lowercase_side}_knee_z_m", f"{prefix.lower()}knee_z_m"),
        ),
        "ankle": (
            (f"{prefix}Ankle_X", f"{lowercase_side}_ankle_x_m", f"{prefix.lower()}ankle_x_m"),
            (f"{prefix}Ankle_Y", f"{lowercase_side}_ankle_y_m", f"{prefix.lower()}ankle_y_m"),
            (f"{prefix}Ankle_Z", f"{lowercase_side}_ankle_z_m", f"{prefix.lower()}ankle_z_m"),
        ),
    }
    coordinates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for joint, axes in candidates.items():
        values = tuple(_numeric(raw, axis_candidates) for axis_candidates in axes)
        if any(value is None for value in values):
            return None
        coordinates[joint] = values  # type: ignore[assignment]

    figure = plt.figure(figsize=(10.5, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    labels = {
        "hip": f"{readable_side} hip (original 3D)",
        "knee": f"{readable_side} knee (original 3D)",
        "ankle": f"{readable_side} ankle (original 3D)",
    }
    colors = {"hip": _COLORS["hip"], "knee": _COLORS["knee"], "ankle": _COLORS["ankle"]}
    all_values: list[np.ndarray] = []
    for joint, (x, y, z) in coordinates.items():
        valid = _finite_rows(x, y, z)
        if not np.any(valid):
            continue
        axis.plot(x[valid], y[valid], z[valid], color=colors[joint], label=labels[joint])
        axis.scatter(x[valid][0], y[valid][0], z[valid][0], color=colors[joint], marker="o", s=28)
        axis.scatter(x[valid][-1], y[valid][-1], z[valid][-1], color=colors[joint], marker="x", s=36)
        all_values.extend((x[valid], y[valid], z[valid]))
    if not all_values:
        plt.close(figure)
        return None

    finite_x = np.concatenate([coordinates[joint][0][_finite_rows(*coordinates[joint])] for joint in coordinates])
    finite_y = np.concatenate([coordinates[joint][1][_finite_rows(*coordinates[joint])] for joint in coordinates])
    finite_z = np.concatenate([coordinates[joint][2][_finite_rows(*coordinates[joint])] for joint in coordinates])
    # A data-range box aspect compresses the comparatively narrow original-X
    # direction until its tick labels collide.  A cubic display box preserves
    # readable axes while all values remain in their unmodified metric scale.
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=24.0, azim=-58.0)
    for axis_object in (axis.xaxis, axis.yaxis, axis.zaxis):
        axis_object.set_major_locator(MaxNLocator(nbins=5))
    axis.set_xlabel("Original X (m)")
    axis.set_ylabel("Original Y (m)")
    axis.set_zlabel("Original Z (m)")
    source_unit = _metadata_value(metadata, "source_unit", "source_coordinate_unit")
    note = "Values plotted in metres"
    if source_unit is not None:
        note += f"; source unit recorded as {source_unit}"
    axis.set_title(
        f"Original 3D {lowercase_side}-leg landmark trajectory\n{note}",
        pad=16,
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 0.96))
    return _save_figure(figure, destination, "3d_right_leg_trajectory.png")


def _projected_coordinates(
    dataframe: pd.DataFrame,
    joint: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    names = {
        "hip": (
            ("x_hip_projected_m", "x_hip_m", "hip_x_projected_m"),
            ("z_hip_projected_m", "z_hip_m", "hip_z_projected_m"),
        ),
        "knee": (
            ("x_knee_observed_m", "x_knee_projected_m", "x_knee_m"),
            ("z_knee_observed_m", "z_knee_projected_m", "z_knee_m"),
        ),
        "ankle": (
            ("x_ankle_observed_m", "x_ankle_projected_m", "x_ankle_m"),
            ("z_ankle_observed_m", "z_ankle_projected_m", "z_ankle_m"),
        ),
        "pull": (
            ("x_pull_m", "x_pull_projected_m"),
            ("z_pull_m", "z_pull_projected_m"),
        ),
    }
    x_names, z_names = names[joint]
    return _numeric(dataframe, x_names), _numeric(dataframe, z_names)


def _plot_sagittal_projection(
    angles: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    hip_x, hip_z = _projected_coordinates(angles, "hip")
    knee_x, knee_z = _projected_coordinates(angles, "knee")
    ankle_x, ankle_z = _projected_coordinates(angles, "ankle")
    if knee_x is None or knee_z is None or ankle_x is None or ankle_z is None:
        return None
    if hip_x is None or hip_z is None:
        hip_x = np.zeros(len(angles), dtype=float)
        hip_z = np.zeros(len(angles), dtype=float)

    finite = _finite_rows(hip_x, hip_z, knee_x, knee_z, ankle_x, ankle_z)
    if not np.any(finite):
        return None
    indices = np.flatnonzero(finite)
    snapshot_indices = np.unique(
        np.linspace(0, len(indices) - 1, min(10, len(indices))).astype(int)
    )

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    for local_index in snapshot_indices:
        index = indices[local_index]
        axis.plot(
            [hip_x[index], knee_x[index], ankle_x[index]],
            [hip_z[index], knee_z[index], ankle_z[index]],
            color="#C9C9C9",
            linewidth=1.0,
            alpha=0.65,
        )
    axis.plot(hip_x[finite], hip_z[finite], color=_COLORS["hip"], label="Hip — sagittal projection")
    axis.plot(knee_x[finite], knee_z[finite], color=_COLORS["knee"], label="Knee — sagittal projection")
    axis.plot(ankle_x[finite], ankle_z[finite], color=_COLORS["ankle"], label="Observed ankle — sagittal projection")

    pull_x, pull_z = _projected_coordinates(angles, "pull")
    if pull_x is not None and pull_z is not None:
        pull_valid = _finite_rows(pull_x, pull_z)
        axis.plot(
            pull_x[pull_valid],
            pull_z[pull_valid],
            color=_COLORS["pull"],
            linewidth=1.8,
            linestyle="--",
            label="L2 equivalent pull point (not the ankle)",
        )
    axis.axhline(0.0, color=_COLORS["bed"], linewidth=1.2, label="Bed: z = 0")
    axis.set_xlabel("Sagittal bed direction x (m)")
    axis.set_ylabel("Sagittal vertical z (m)")
    primary_leg = str(_metadata_value(metadata, "primary_motion_leg") or "right").lower()
    axis.set_title(
        f"{primary_leg.title()}-leg sagittal-plane projection from original 3D landmarks"
    )
    planarity = _numeric(angles, ("planarity_error_m",))
    if planarity is not None and np.isfinite(planarity).any():
        values = planarity[np.isfinite(planarity)]
        axis.text(
            0.02,
            0.05,
            f"Planarity error: median {np.median(values) * 1000:.1f} mm, "
            f"95th percentile {np.percentile(values, 95) * 1000:.1f} mm",
            transform=axis.transAxes,
            va="bottom",
        )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")
    return _save_figure(figure, destination, "sagittal_plane_projection.png")


def _draw_joint_limits(
    axis: plt.Axes,
    limits: tuple[float, float],
    *,
    joint: str,
) -> None:
    for value, name in zip(limits, ("lower", "upper")):
        axis.axhline(
            value,
            color=_COLORS["invalid"],
            linewidth=0.9,
            linestyle=":" if name == "lower" else "--",
            alpha=0.8,
            label=f"{joint.title()} {name} limit ({value:g}°)",
        )


def _plot_raw_and_filtered_angles(
    angles: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    raw_values = {
        "hip": _angle_degrees(
            angles,
            degree_candidates=("q_hip_raw_deg",),
            radian_candidates=("q_hip_raw_rad",),
        ),
        "knee": _angle_degrees(
            angles,
            degree_candidates=("q_knee_raw_deg",),
            radian_candidates=("q_knee_raw_rad",),
        ),
    }
    filtered_values = {
        "hip": _angle_degrees(
            angles,
            degree_candidates=("q_hip_deg", "q_hip_filtered_deg"),
            radian_candidates=("q_hip_rad", "q_hip_filtered_rad"),
        ),
        "knee": _angle_degrees(
            angles,
            degree_candidates=("q_knee_deg", "q_knee_filtered_deg"),
            radian_candidates=("q_knee_rad", "q_knee_filtered_rad"),
        ),
    }
    if all(value is None for value in (*raw_values.values(), *filtered_values.values())):
        return None

    frame = _frames(angles)
    joint_valid = _boolean_mask(
        angles,
        ("joint_limit_valid", "angle_valid"),
        default=True,
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, joint in zip(axes, ("hip", "knee")):
        raw = raw_values[joint]
        filtered = filtered_values[joint]
        if raw is not None:
            valid = _finite_rows(frame, raw)
            axis.plot(frame[valid], raw[valid], color=_COLORS["raw"], linewidth=1.0, alpha=0.75, label=f"Raw {joint} angle")
        if filtered is not None:
            valid = _finite_rows(frame, filtered)
            axis.plot(frame[valid], filtered[valid], color=_COLORS[joint], linewidth=1.8, label=f"Filtered {joint} angle")
            out = valid & ~joint_valid
            if np.any(out):
                axis.scatter(frame[out], filtered[out], color=_COLORS["invalid"], marker="x", s=22, label="Out of joint range / invalid")
        else:
            axis.text(0.02, 0.9, "Filtered angle unavailable — raw data not substituted", transform=axis.transAxes)
        _draw_joint_limits(axis, _joint_limits(metadata, joint), joint=joint)
        axis.set_ylabel(f"{joint.title()} angle (deg)")
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Frame")
    axes[0].set_title("Raw and filtered joint angles with configured joint ranges")
    return _save_figure(figure, destination, "raw_and_filtered_joint_angles.png")


def _cycle_complete(row: pd.Series) -> bool:
    value = row.get("cycle_complete", False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "complete"}
    return bool(value) if pd.notna(value) else False


def _plot_detected_cycles(
    angles: pd.DataFrame,
    cycles: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    hip = _angle_degrees(
        angles,
        degree_candidates=("q_hip_deg", "q_hip_filtered_deg", "q_hip_raw_deg"),
        radian_candidates=("q_hip_rad", "q_hip_filtered_rad", "q_hip_raw_rad"),
    )
    knee = _angle_degrees(
        angles,
        degree_candidates=("q_knee_deg", "q_knee_filtered_deg", "q_knee_raw_deg"),
        radian_candidates=("q_knee_rad", "q_knee_filtered_rad", "q_knee_raw_rad"),
    )
    if hip is None and knee is None:
        return None
    frame = _frames(angles)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, values, joint in zip(axes, (hip, knee), ("hip", "knee")):
        if values is not None:
            valid = _finite_rows(frame, values)
            axis.plot(frame[valid], values[valid], color=_COLORS[joint], linewidth=1.5, label=f"Filtered {joint} angle")
        _draw_joint_limits(axis, _joint_limits(metadata, joint), joint=joint)
        axis.set_ylabel(f"{joint.title()} angle (deg)")

    complete_label_used = False
    incomplete_label_used = False
    selected_label_used = False
    required = {"start_frame", "end_frame"}
    if cycles.empty or not required.issubset(cycles.columns):
        axes[0].text(0.02, 0.88, "No valid cycle intervals were supplied", transform=axes[0].transAxes)
    else:
        for _, row in cycles.sort_values("start_frame").iterrows():
            try:
                start = float(row["start_frame"])
                end = float(row["end_frame"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(start) and np.isfinite(end) and end >= start):
                continue
            complete = _cycle_complete(row)
            color = "#59A14F" if complete else _COLORS["invalid"]
            label = "Complete cycle" if complete else "Incomplete cycle"
            show_label = (complete and not complete_label_used) or (
                not complete and not incomplete_label_used
            )
            for axis in axes:
                axis.axvspan(start, end, color=color, alpha=0.12, label=label if show_label else None)
                if bool(row.get("selected", False)):
                    axis.axvspan(
                        start,
                        end,
                        facecolor="none",
                        edgecolor=_COLORS["hip"],
                        linewidth=2.0,
                        label=(
                            "Selected representative cycle"
                            if not selected_label_used
                            else None
                        ),
                    )
            complete_label_used |= complete
            incomplete_label_used |= not complete
            selected_label_used |= bool(row.get("selected", False))
            peak = row.get("peak_flexion_frame", np.nan)
            try:
                peak_value = float(peak)
            except (TypeError, ValueError):
                peak_value = np.nan
            if np.isfinite(peak_value):
                for axis in axes:
                    axis.axvline(peak_value, color=color, alpha=0.65, linewidth=0.8)
            cycle_id = row.get("cycle_index", "?")
            quality = row.get("cycle_quality_score", np.nan)
            quality_text = ""
            try:
                quality_float = float(quality)
                if np.isfinite(quality_float):
                    quality_text = f"; quality={quality_float:.2f}"
            except (TypeError, ValueError):
                pass
            axes[0].text(
                (start + end) / 2.0,
                0.98,
                f"Cycle {cycle_id}: {'complete' if complete else 'incomplete'}{quality_text}",
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                rotation=90,
            )
    joint_valid = _boolean_mask(angles, ("joint_limit_valid",), default=True)
    invalid = ~joint_valid & np.isfinite(frame)
    if np.any(invalid):
        for axis, values in zip(axes, (hip, knee)):
            if values is not None:
                shown = invalid & np.isfinite(values)
                axis.scatter(frame[shown], values[shown], color=_COLORS["invalid"], marker="x", s=20, label="Out of joint range")
    axes[0].set_title("Detected flexion-extension cycles: complete and incomplete intervals")
    axes[-1].set_xlabel("Frame")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8)
    return _save_figure(figure, destination, "detected_cycles.png")


def _selected_x_axis(
    selected: pd.DataFrame,
    fps: float | None,
) -> tuple[np.ndarray, str]:
    time = _numeric(selected, ("time_s",))
    if fps is not None and time is not None and np.isfinite(time).any():
        first = time[np.flatnonzero(np.isfinite(time))[0]]
        return time - first, "Selected-cycle time (s)"
    return _frames(selected), "Frame"


def _plot_selected_angles(
    selected: pd.DataFrame,
    metadata: Mapping[str, Any],
    fps: float | None,
    destination: Path,
) -> Path | None:
    hip = _angle_degrees(
        selected,
        degree_candidates=("q_hip_deg",),
        radian_candidates=("q_hip_rad",),
    )
    knee = _angle_degrees(
        selected,
        degree_candidates=("q_knee_deg",),
        radian_candidates=("q_knee_rad",),
    )
    if hip is None or knee is None:
        return None
    x, xlabel = _selected_x_axis(selected, fps)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    joint_valid = _boolean_mask(
        selected,
        ("joint_limit_valid", "trajectory_sample_valid"),
        default=True,
    )
    for axis, values, joint in zip(axes, (hip, knee), ("hip", "knee")):
        finite = _finite_rows(x, values)
        axis.plot(x[finite], values[finite], color=_COLORS[joint], linewidth=1.8, label=f"Selected-cycle {joint} angle")
        invalid = finite & ~joint_valid
        if np.any(invalid):
            axis.scatter(x[invalid], values[invalid], color=_COLORS["invalid"], marker="x", s=24, label="Out of range / invalid sample")
        _draw_joint_limits(axis, _joint_limits(metadata, joint), joint=joint)
        axis.set_ylabel(f"{joint.title()} angle (deg)")
        axis.legend(ncol=2, fontsize=8)
    if "cycle_phase" in selected:
        phases = selected["cycle_phase"].astype(str).to_numpy()
        transitions = np.flatnonzero(phases[1:] != phases[:-1]) + 1
        for index in transitions:
            if index < len(x) and np.isfinite(x[index]):
                for axis in axes:
                    axis.axvline(x[index], color="#777777", linestyle=":", linewidth=0.9)
                axes[0].text(
                    x[index],
                    0.86,
                    phases[index],
                    transform=axes[0].get_xaxis_transform(),
                    va="top",
                    rotation=90,
                    fontsize=8,
                )
    axes[0].set_title("Selected complete cycle joint angles and configured ranges")
    axes[-1].set_xlabel(xlabel)
    return _save_figure(figure, destination, "selected_cycle_angles.png")


def _selected_leg_geometry(
    selected: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray] | None:
    q_hip = _angle_radians(
        selected,
        radian_candidates=("q_hip_rad",),
        degree_candidates=("q_hip_deg",),
    )
    q_knee = _angle_radians(
        selected,
        radian_candidates=("q_knee_rad",),
        degree_candidates=("q_knee_deg",),
    )
    x_knee = _numeric(selected, ("x_knee_m", "x_knee_projected_m"))
    z_knee = _numeric(selected, ("z_knee_m", "z_knee_projected_m"))
    x_pull = _numeric(selected, ("x_pull_m", "x_pull_projected_m"))
    z_pull = _numeric(selected, ("z_pull_m", "z_pull_projected_m"))
    if any(value is None for value in (x_knee, z_knee, x_pull, z_pull)):
        if q_hip is None or q_knee is None:
            return None
        calculated = forward_kinematics(
            q_hip,
            q_knee,
            _geometry_length(metadata, "L1"),
            _geometry_length(metadata, "L2"),
        )
        x_knee, z_knee, x_pull, z_pull = (
            np.asarray(value, dtype=float) for value in calculated
        )

    hip_x = _numeric(selected, ("x_hip_projected_m", "x_hip_m"))
    hip_z = _numeric(selected, ("z_hip_projected_m", "z_hip_m"))
    if hip_x is None:
        hip_x = np.zeros(len(selected), dtype=float)
    if hip_z is None:
        hip_z = np.zeros(len(selected), dtype=float)
    return {
        "hip_x": np.asarray(hip_x, dtype=float),
        "hip_z": np.asarray(hip_z, dtype=float),
        "knee_x": np.asarray(x_knee, dtype=float),
        "knee_z": np.asarray(z_knee, dtype=float),
        "pull_x": np.asarray(x_pull, dtype=float),
        "pull_z": np.asarray(z_pull, dtype=float),
    }


def _plot_selected_pull_path(
    selected: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    geometry = _selected_leg_geometry(selected, metadata)
    if geometry is None:
        return None
    pull_x = geometry["pull_x"]
    pull_z = geometry["pull_z"]
    valid = _finite_rows(pull_x, pull_z)
    if not np.any(valid):
        return None
    figure, axis = plt.subplots(figsize=(8, 6.5))
    frame = _frames(selected)
    color_value = frame[valid] if np.isfinite(frame[valid]).all() else np.arange(np.count_nonzero(valid))
    scatter = axis.scatter(pull_x[valid], pull_z[valid], c=color_value, cmap="viridis", s=18, label="L2 equivalent pull point")
    figure.colorbar(scatter, ax=axis, label="Frame")
    first = np.flatnonzero(valid)[0]
    last = np.flatnonzero(valid)[-1]
    axis.scatter(pull_x[first], pull_z[first], marker="o", s=55, color="#59A14F", label="Selected-cycle start")
    axis.scatter(pull_x[last], pull_z[last], marker="s", s=45, color="#E15759", label="Selected-cycle end")

    ankle_x, ankle_z = _projected_coordinates(selected, "ankle")
    if ankle_x is not None and ankle_z is not None:
        ankle_valid = _finite_rows(ankle_x, ankle_z)
        axis.plot(ankle_x[ankle_valid], ankle_z[ankle_valid], color=_COLORS["ankle"], linewidth=1.2, linestyle="--", label="Observed ankle path")
    axis.axhline(0.0, color=_COLORS["bed"], linewidth=1.2, label="Bed: z = 0")
    axis.set_xlabel("Bed direction x (m)")
    axis.set_ylabel("Vertical z (m)")
    axis.set_title("Selected-cycle pull path — L2 endpoint is not the ankle")
    axis.text(
        0.02,
        0.96,
        "Model endpoint = equivalent strap pull point; observed ankle shown separately",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")
    return _save_figure(figure, destination, "selected_cycle_pull_path.png")


def _create_leg_animation(
    selected: pd.DataFrame,
    metadata: Mapping[str, Any],
    fps: float | None,
    destination: Path,
    *,
    maximum_frames: int = 120,
) -> Path | None:
    geometry = _selected_leg_geometry(selected, metadata)
    if geometry is None or selected.empty:
        return None
    valid = _finite_rows(*geometry.values())
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        return None
    selection = np.unique(
        np.linspace(
            0,
            len(valid_indices) - 1,
            min(maximum_frames, len(valid_indices)),
        ).astype(int)
    )
    frame_indices = valid_indices[selection]

    all_x = np.concatenate(
        [geometry["hip_x"][valid], geometry["knee_x"][valid], geometry["pull_x"][valid]]
    )
    all_z = np.concatenate(
        [geometry["hip_z"][valid], geometry["knee_z"][valid], geometry["pull_z"][valid], np.array([0.0])]
    )
    x_span = max(float(np.ptp(all_x)), 0.1)
    z_span = max(float(np.ptp(all_z)), 0.1)
    margin = 0.12 * max(x_span, z_span)

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    thigh_line, = axis.plot([], [], color=_COLORS["hip"], linewidth=3.0, label="Thigh: hip to knee")
    pull_link, = axis.plot([], [], color=_COLORS["pull"], linewidth=3.0, label="Knee to L2 pull point")
    hip_marker, = axis.plot([], [], marker="s", color=_COLORS["hip"], linestyle="None", markersize=8, label="Hip")
    knee_marker, = axis.plot([], [], marker="o", color=_COLORS["knee"], linestyle="None", markersize=8, label="Knee")
    pull_marker, = axis.plot([], [], marker="D", color=_COLORS["pull"], linestyle="None", markersize=7, label="Equivalent pull point (not ankle)")
    ankle_marker, = axis.plot([], [], marker="x", color=_COLORS["ankle"], linestyle="None", markersize=7, label="Observed ankle")
    status = axis.text(0.02, 0.97, "", transform=axis.transAxes, va="top")
    axis.axhline(0.0, color=_COLORS["bed"], linewidth=1.4, label="Bed: z = 0")
    axis.set_xlim(float(np.min(all_x)) - margin, float(np.max(all_x)) + margin)
    axis.set_ylim(min(-0.02, float(np.min(all_z)) - margin), float(np.max(all_z)) + margin)
    axis.set_xlabel("Bed direction x (m)")
    axis.set_ylabel("Vertical z (m)")
    axis.set_aspect("equal", adjustable="box")
    timing_note = (
        f"Source timing: {fps:g} fps"
        if fps is not None
        else "Frame-based preview only — source fps unavailable; not real-time"
    )
    axis.set_title(f"Selected-cycle leg animation\n{timing_note}")
    axis.legend(loc="best", fontsize=8)

    ankle_x, ankle_z = _projected_coordinates(selected, "ankle")
    frame_values = _frames(selected)
    phases = (
        selected["cycle_phase"].astype(str).to_numpy()
        if "cycle_phase" in selected
        else np.full(len(selected), "", dtype=object)
    )

    def update(index: int) -> tuple[Any, ...]:
        hx = geometry["hip_x"][index]
        hz = geometry["hip_z"][index]
        kx = geometry["knee_x"][index]
        kz = geometry["knee_z"][index]
        px = geometry["pull_x"][index]
        pz = geometry["pull_z"][index]
        thigh_line.set_data([hx, kx], [hz, kz])
        pull_link.set_data([kx, px], [kz, pz])
        hip_marker.set_data([hx], [hz])
        knee_marker.set_data([kx], [kz])
        pull_marker.set_data([px], [pz])
        if ankle_x is not None and ankle_z is not None and np.isfinite(ankle_x[index]) and np.isfinite(ankle_z[index]):
            ankle_marker.set_data([ankle_x[index]], [ankle_z[index]])
        else:
            ankle_marker.set_data([], [])
        status.set_text(f"Frame {frame_values[index]:g}  {phases[index]}")
        return thigh_line, pull_link, hip_marker, knee_marker, pull_marker, ankle_marker, status

    source_stride = max(1, int(np.ceil(len(valid_indices) / len(frame_indices))))
    writer_fps = min(20.0, fps / source_stride) if fps is not None else 10.0
    writer_fps = max(float(writer_fps), 1.0)
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000.0 / writer_fps,
        blit=False,
        repeat=True,
    )
    output_path = destination / "selected_cycle_leg_animation.gif"
    animation.save(output_path, writer=PillowWriter(fps=writer_fps))
    plt.close(figure)
    return output_path


def _time_axis_for_dynamics(
    dynamics: pd.DataFrame,
    fps: float,
) -> np.ndarray:
    time = _numeric(dynamics, ("time_s",))
    if time is not None and np.isfinite(time).all() and len(time):
        return time - time[0]
    frame = _frames(dynamics)
    if np.isfinite(frame).all() and len(frame):
        return (frame - frame[0]) / fps
    return np.arange(len(dynamics), dtype=float) / fps


def _preferred_subject(
    dynamics_by_subject: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> tuple[str, pd.DataFrame] | None:
    requested = _metadata_value(metadata, "reference_subject_id", "subject_id")
    if requested is not None and str(requested) in dynamics_by_subject:
        key = str(requested)
        return key, dynamics_by_subject[key]
    if "baseline" in dynamics_by_subject:
        return "baseline", dynamics_by_subject["baseline"]
    for key, value in dynamics_by_subject.items():
        return str(key), value
    return None


def _plot_torque_components(
    subject_id: str,
    dynamics: pd.DataFrame,
    fps: float,
    destination: Path,
) -> Path | None:
    components = (
        ("inertia", "tau_inertia"),
        ("coriolis/centrifugal", "tau_coriolis"),
        ("gravity", "tau_gravity"),
        ("damping", "tau_damping"),
        ("stiffness", "tau_stiffness"),
        ("total", "tau_total"),
    )
    available = {
        joint: [(label, f"{prefix}_{joint}_nm") for label, prefix in components if f"{prefix}_{joint}_nm" in dynamics]
        for joint in ("hip", "knee")
    }
    if not available["hip"] or not available["knee"]:
        return None
    time = _time_axis_for_dynamics(dynamics, fps)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    palette = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#222222")
    sample_valid = _boolean_mask(
        dynamics,
        ("dynamic_sample_valid",),
        default=True,
    )
    for axis, joint in zip(axes, ("hip", "knee")):
        for index, (label, column) in enumerate(available[joint]):
            values = pd.to_numeric(dynamics[column], errors="coerce").to_numpy(float)
            valid = _finite_rows(time, values) & sample_valid
            axis.plot(
                time[valid],
                values[valid],
                color=palette[index % len(palette)],
                linewidth=2.1 if label == "total" else 1.2,
                label=label,
            )
        axis.set_ylabel(f"{joint.title()} torque (N·m)")
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Selected-cycle time (s)")
    axes[0].set_title(
        f"Reference torque components — virtual subject: {subject_id} "
        "(valid in-range samples)"
    )
    return _save_figure(figure, destination, "reference_torque_components.png")


def _force_values(dynamics: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    fx = _numeric(dynamics, ("fx_robot_on_leg_n",))
    fz = _numeric(dynamics, ("fz_robot_on_leg_n",))
    magnitude = _numeric(dynamics, ("force_magnitude_n",))
    if magnitude is None and fx is not None and fz is not None:
        magnitude = np.hypot(fx, fz)
    return fx, fz, magnitude


def _plot_endpoint_force(
    subject_id: str,
    dynamics: pd.DataFrame,
    fps: float,
    destination: Path,
) -> Path | None:
    fx, fz, magnitude = _force_values(dynamics)
    if fx is None or fz is None or magnitude is None:
        return None
    time = _time_axis_for_dynamics(dynamics, fps)
    mapping_valid = _boolean_mask(dynamics, ("force_mapping_valid",), default=True)
    dynamic_valid = _boolean_mask(
        dynamics,
        ("dynamic_sample_valid",),
        default=True,
    )
    finite = _finite_rows(time, fx, fz, magnitude) & mapping_valid & dynamic_valid
    if not np.any(finite):
        return None
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(time[finite], fx[finite], color=_COLORS["hip"], label="Fx robot on leg")
    axis.plot(time[finite], fz[finite], color=_COLORS["knee"], label="Fz robot on leg")
    axis.plot(time[finite], magnitude[finite], color="#222222", linewidth=2.0, label="|F|")
    invalid = ~finite & np.isfinite(time)
    if np.any(invalid):
        axis.scatter(time[invalid], np.zeros(np.count_nonzero(invalid)), color=_COLORS["invalid"], marker="x", s=18, label="Invalid force mapping")
    axis.set_xlabel("Selected-cycle time (s)")
    axis.set_ylabel("Endpoint force (N)")
    axis.set_title(
        f"Reference endpoint force — virtual subject: {subject_id} "
        "(valid in-range samples)"
    )
    axis.legend(ncol=4)
    return _save_figure(figure, destination, "reference_endpoint_force.png")


def _plot_subject_comparison(
    dynamics_by_subject: Mapping[str, pd.DataFrame],
    fps: float,
    destination: Path,
) -> Path | None:
    usable: list[tuple[str, pd.DataFrame]] = []
    for subject_id, data in dynamics_by_subject.items():
        if not isinstance(data, pd.DataFrame) or data.empty:
            continue
        if "tau_total_hip_nm" not in data or "tau_total_knee_nm" not in data:
            continue
        if _force_values(data)[2] is None:
            continue
        usable.append((str(subject_id), data))
    if not usable:
        return None

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    palette = plt.get_cmap("tab10")
    for index, (subject_id, data) in enumerate(usable):
        time = _time_axis_for_dynamics(data, fps)
        hip = _numeric(data, ("tau_total_hip_nm",))
        knee = _numeric(data, ("tau_total_knee_nm",))
        magnitude = _force_values(data)[2]
        assert hip is not None and knee is not None and magnitude is not None
        sample_valid = _boolean_mask(
            data,
            ("dynamic_sample_valid",),
            default=True,
        )
        color = palette(index % 10)
        for axis, values in zip(axes, (hip, knee, magnitude)):
            valid = _finite_rows(time, values) & sample_valid
            axis.plot(time[valid], values[valid], color=color, label=subject_id)
    axes[0].set_ylabel("Hip total torque (N·m)")
    axes[1].set_ylabel("Knee total torque (N·m)")
    axes[2].set_ylabel("Endpoint |F| (N)")
    axes[2].set_xlabel("Selected-cycle time (s)")
    axes[0].set_title("Reference trajectory virtual-subject comparison")
    for axis in axes:
        axis.legend(ncol=min(4, len(usable)), fontsize=8)
    if len(usable) == 1:
        axes[0].text(0.99, 0.04, "Only one virtual subject supplied", transform=axes[0].transAxes, ha="right")
    return _save_figure(figure, destination, "reference_subject_comparison.png")


def generate_reference_trajectory_visualizations(
    raw_trajectory: pd.DataFrame,
    full_angles: pd.DataFrame,
    detected_cycles: pd.DataFrame,
    selected_cycle: pd.DataFrame | None,
    dynamics_by_subject: Mapping[str, pd.DataFrame] | None,
    metadata: Mapping[str, Any] | None,
    output_directory: str | Path,
) -> ReferenceTrajectoryVisualizationResult:
    """Generate the ten requested Stage 5A visual products.

    Parameters
    ----------
    raw_trajectory:
        Original right-leg 3-D landmark columns (for example ``RHip_X`` through
        ``RAnkle_Z``), already normalized to metres.
    full_angles:
        Full sagittal projection and raw/filtered joint-angle observations.
    detected_cycles:
        One row per candidate cycle with start, peak, end and completeness.
    selected_cycle:
        Selected complete cycle, or ``None`` if no cycle passed validation.
    dynamics_by_subject:
        Mapping from virtual subject ID to selected-cycle inverse-dynamics data.
        These curves are never synthesized when absent.
    metadata:
        Import and geometry metadata.  A positive source ``fps`` is required for
        the three dynamics figures; it is not inferred from row spacing.
    output_directory:
        Destination for the canonical PNG/GIF filenames.

    Returns
    -------
    ReferenceTrajectoryVisualizationResult
        Every requested filename appears in exactly one of ``paths`` or
        ``skipped``.  A frame-based GIF is allowed without FPS and is labelled
        as non-real-time; torque/force figures are not.
    """

    _configure_style()
    raw = _as_dataframe(raw_trajectory, "raw_trajectory")
    angles = _as_dataframe(full_angles, "full_angles")
    cycles = _as_dataframe(detected_cycles, "detected_cycles")
    selected = _as_dataframe(selected_cycle, "selected_cycle")
    metadata_map: Mapping[str, Any] = metadata if metadata is not None else {}
    if not isinstance(metadata_map, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    if dynamics_by_subject is None:
        dynamics: Mapping[str, pd.DataFrame] = {}
    elif isinstance(dynamics_by_subject, Mapping):
        dynamics = dynamics_by_subject
    else:
        raise TypeError("dynamics_by_subject must be a mapping or None.")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    fps = _source_fps(metadata_map)

    path = _plot_original_3d(raw, metadata_map, destination)
    if path is None:
        _skip(skipped, "3d_right_leg_trajectory.png", "original right-leg 3-D landmark columns are missing or non-finite")
    else:
        paths[path.name] = path

    path = _plot_sagittal_projection(angles, metadata_map, destination)
    if path is None:
        _skip(skipped, "sagittal_plane_projection.png", "sagittal hip/knee/ankle projection columns are missing or non-finite")
    else:
        paths[path.name] = path

    path = _plot_raw_and_filtered_angles(angles, metadata_map, destination)
    if path is None:
        _skip(skipped, "raw_and_filtered_joint_angles.png", "raw and filtered hip/knee angle columns are unavailable")
    else:
        paths[path.name] = path

    path = _plot_detected_cycles(angles, cycles, metadata_map, destination)
    if path is None:
        _skip(skipped, "detected_cycles.png", "no finite hip or knee angle series is available for cycle display")
    else:
        paths[path.name] = path

    if selected.empty:
        selected_reason = "no validated selected cycle was supplied"
        for filename in (
            "selected_cycle_angles.png",
            "selected_cycle_pull_path.png",
            "selected_cycle_leg_animation.gif",
        ):
            _skip(skipped, filename, selected_reason)
    else:
        path = _plot_selected_angles(selected, metadata_map, fps, destination)
        if path is None:
            _skip(skipped, "selected_cycle_angles.png", "selected cycle lacks finite filtered hip/knee angles")
        else:
            paths[path.name] = path
        path = _plot_selected_pull_path(selected, metadata_map, destination)
        if path is None:
            _skip(skipped, "selected_cycle_pull_path.png", "selected cycle lacks finite pull-point geometry and reconstructable angles")
        else:
            paths[path.name] = path
        path = _create_leg_animation(selected, metadata_map, fps, destination)
        if path is None:
            _skip(skipped, "selected_cycle_leg_animation.gif", "selected cycle has fewer than two finite leg-geometry frames")
        else:
            paths[path.name] = path

    if fps is None:
        reason = "source fps is unavailable; dynamics were not estimated and no synthetic dynamic curve was drawn"
        for filename in DYNAMIC_FIGURES:
            _skip(skipped, filename, reason)
    elif not dynamics:
        reason = "source fps is available, but no per-subject dynamics dataframe was supplied"
        for filename in DYNAMIC_FIGURES:
            _skip(skipped, filename, reason)
    else:
        preferred = _preferred_subject(dynamics, metadata_map)
        if preferred is None or not isinstance(preferred[1], pd.DataFrame) or preferred[1].empty:
            for filename in DYNAMIC_FIGURES:
                _skip(skipped, filename, "no non-empty per-subject dynamics dataframe was supplied")
        else:
            subject_id, reference_dynamics = preferred
            path = _plot_torque_components(subject_id, reference_dynamics, fps, destination)
            if path is None:
                _skip(skipped, "reference_torque_components.png", "required measured torque-component columns are missing")
            else:
                paths[path.name] = path
            path = _plot_endpoint_force(subject_id, reference_dynamics, fps, destination)
            if path is None:
                _skip(skipped, "reference_endpoint_force.png", "required finite Fx/Fz and force-validity samples are missing")
            else:
                paths[path.name] = path
            path = _plot_subject_comparison(dynamics, fps, destination)
            if path is None:
                _skip(skipped, "reference_subject_comparison.png", "no subject has total hip/knee torque and endpoint-force data")
            else:
                paths[path.name] = path

    # Guard against accidental silent omission when this module evolves.
    for filename in FIGURE_FILENAMES:
        if filename not in paths and filename not in skipped:
            skipped[filename] = "visualization was not generated; no explicit producer matched"
    return ReferenceTrajectoryVisualizationResult(paths=paths, skipped=skipped)


__all__ = [
    "DYNAMIC_FIGURES",
    "FIGURE_FILENAMES",
    "GEOMETRY_FIGURES",
    "ReferenceTrajectoryVisualizationResult",
    "generate_reference_trajectory_visualizations",
]
