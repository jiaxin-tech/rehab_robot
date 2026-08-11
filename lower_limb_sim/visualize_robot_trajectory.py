"""Offline-only Stage 6A previews for a generated robot trajectory command.

This module is deliberately presentation-only.  It accepts an already
generated trajectory table and calibration metadata, writes PNG previews, and
never imports a robot SDK or opens a hardware connection.  Empty, blocked, or
non-finite trajectory data are reported as skipped outputs rather than being
replaced with a synthetic path.

Coordinate convention shown in the figures
-------------------------------------------

``H`` is the human sagittal frame (``+x_H`` points from hip toward the foot and
``+z_H`` points upward), ``B`` is the robot Base frame, and ``T`` is the TCP
frame.  The Stage 6A mapping is rendered explicitly as

``p_connection_B = p_hip_B + R_BH @ [x_pull_H, 0, z_pull_H]``.

The plus sign follows the axis definitions: a positive human-frame displacement
maps in the same direction as the corresponding H axis expressed in B.  The
plotted TCP path is read from the command table and is kept distinct from the
mapped strap/connection path so that the configured tool offset remains visible
and auditable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any


_MPL_CONFIG_DIRECTORY = Path(tempfile.gettempdir()) / "lower_limb_sim_matplotlib"
_MPL_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIRECTORY))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_FILENAMES = (
    "robot_trajectory_preview.png",
    "robot_workspace_preview.png",
    "human_vs_robot_coordinate_preview.png",
)

_BASE_COLORS = ("#4C78A8", "#F58518", "#54A24B")
_ORIENTATION_COLORS = ("#B279A2", "#E45756", "#72B7B2")
_INVALID_COLOR = "#D62728"
_CONNECTION_COLOR = "#79706E"
_TCP_COLOR = "#4C78A8"


@dataclass(frozen=True)
class RobotTrajectoryVisualizationResult:
    """Paths generated and explicit reasons for every omitted Stage 6A plot."""

    paths: dict[str, Path]
    skipped: dict[str, str]

    @property
    def generated_paths(self) -> tuple[Path, ...]:
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


def _as_dataframe(value: pd.DataFrame | None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError("trajectory must be a pandas DataFrame or None.")
    return value.copy(deep=False)


def _as_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("calibration_metadata must be a mapping or None.")
    return value


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    """Read a calibration field from the common Stage 6A metadata sections."""

    if key in metadata:
        return metadata[key]
    for section_name in (
        "calibration",
        "coordinate_transform",
        "frames",
        "trajectory_generation",
        "stage6a",
    ):
        section = metadata.get(section_name)
        if isinstance(section, Mapping) and key in section:
            return section[key]
    return None


def _orientation_presentation(
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, str, str], str]:
    orientation = _metadata_value(metadata, "tcp_orientation")
    representation: Any = None
    if isinstance(orientation, Mapping):
        representation = orientation.get("representation")
    if representation is None:
        representation = _metadata_value(
            metadata,
            "tcp_orientation_representation",
        )
    if str(representation) == "euler_xyz_rad":
        return ("Euler x", "Euler y", "Euler z"), "TCP Euler XYZ angles in Base B (rad)"
    if str(representation) == "rotation_vector_rad":
        return ("rotvec x", "rotvec y", "rotvec z"), "TCP rotation vector in Base B (rad)"
    return ("component x", "component y", "component z"), "TCP orientation components in Base B (rad)"


def _vector3(value: Any) -> np.ndarray | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, Mapping):
        if all(key in value for key in ("x", "y", "z")):
            value = [value["x"], value["y"], value["z"]]
        elif all(key in value for key in ("rx", "ry", "rz")):
            value = [value["rx"], value["ry"], value["rz"]]
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 3 or not np.isfinite(array).all():
        return None
    return array


def _numeric(dataframe: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(dataframe[name], errors="coerce").to_numpy(dtype=float)


def _valid_mask(dataframe: pd.DataFrame) -> np.ndarray:
    values = dataframe["trajectory_valid"]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.eq(1.0)
    strings = values.astype("string").str.strip().str.lower()
    mask |= strings.isin(("true", "yes", "valid"))
    return mask.fillna(False).to_numpy(dtype=bool)


_REQUIRED_COLUMNS = (
    "time_s",
    "x_pull_human_m",
    "z_pull_human_m",
    "tcp_x_base_m",
    "tcp_y_base_m",
    "tcp_z_base_m",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
    "trajectory_valid",
)


def _trajectory_status(dataframe: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    if dataframe.empty:
        return None, "no offline robot trajectory was generated"
    missing = [column for column in _REQUIRED_COLUMNS if column not in dataframe]
    if missing:
        return None, "trajectory is missing required columns: " + ", ".join(missing)
    valid = _valid_mask(dataframe)
    finite_columns = _REQUIRED_COLUMNS[:-1]
    finite = np.ones(len(dataframe), dtype=bool)
    for column in finite_columns:
        finite &= np.isfinite(_numeric(dataframe, column))
    valid &= finite
    if not valid.any():
        return None, "trajectory contains no finite samples marked trajectory_valid"
    return valid, None


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def _derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2 or not np.all(np.diff(time_s) > 0.0):
        return np.full_like(values, np.nan, dtype=float)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, time_s, axis=0, edge_order=edge_order)


def _mark_invalid_times(axis: plt.Axes, time_s: np.ndarray, valid: np.ndarray) -> None:
    invalid = (~valid) & np.isfinite(time_s)
    if invalid.any():
        axis.scatter(
            time_s[invalid],
            np.full(invalid.sum(), axis.get_ylim()[0]),
            marker="x",
            s=14,
            color=_INVALID_COLOR,
            label="invalid command sample",
            zorder=5,
        )


def robot_trajectory_preview(
    trajectory: pd.DataFrame,
    calibration_metadata: Mapping[str, Any] | None,
    output_dir: str | Path,
) -> Path | None:
    """Plot TCP position/orientation and measured kinematic maxima versus time.

    No safety threshold is drawn or inferred.  Speed and acceleration are
    presentation diagnostics computed from the supplied command samples.
    """

    _configure_style()
    metadata = _as_metadata(calibration_metadata)
    dataframe = _as_dataframe(trajectory)
    valid, reason = _trajectory_status(dataframe)
    if reason is not None or valid is None:
        return None

    time_s = _numeric(dataframe, "time_s")
    position = np.column_stack(
        [_numeric(dataframe, name) for name in (
            "tcp_x_base_m", "tcp_y_base_m", "tcp_z_base_m"
        )]
    )
    orientation = np.column_stack(
        [_numeric(dataframe, name) for name in (
            "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad"
        )]
    )
    if "tcp_speed_m_s" in dataframe:
        speed = _numeric(dataframe, "tcp_speed_m_s")
    else:
        velocity = _derivative(position, time_s)
        speed = np.linalg.norm(velocity, axis=1)
    if "tcp_acceleration_m_s2" in dataframe:
        acceleration_magnitude = _numeric(dataframe, "tcp_acceleration_m_s2")
    else:
        velocity = _derivative(position, time_s)
        acceleration = _derivative(velocity, time_s)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    for index, (label, color) in enumerate(zip(("X_B", "Y_B", "Z_B"), _BASE_COLORS)):
        axes[0, 0].plot(time_s[valid], position[valid, index], color=color, label=label)
    axes[0, 0].set_ylabel("TCP position in Base B (m)")
    axes[0, 0].legend(ncol=3)

    orientation_labels, orientation_ylabel = _orientation_presentation(metadata)
    for index, (label, color) in enumerate(
        zip(orientation_labels, _ORIENTATION_COLORS)
    ):
        axes[0, 1].plot(time_s[valid], orientation[valid, index], color=color, label=label)
    axes[0, 1].set_ylabel(orientation_ylabel)
    axes[0, 1].legend(ncol=3)

    axes[1, 0].plot(time_s[valid], speed[valid], color=_TCP_COLOR)
    axes[1, 0].set_ylabel("Cartesian speed (m/s)")
    axes[1, 0].set_xlabel("Retimed trajectory time (s)")
    if np.isfinite(speed[valid]).any():
        axes[1, 0].set_title(f"Observed maximum: {np.nanmax(speed[valid]):.4f} m/s")

    axes[1, 1].plot(time_s[valid], acceleration_magnitude[valid], color=_ORIENTATION_COLORS[1])
    axes[1, 1].set_ylabel("Cartesian acceleration (m/s²)")
    axes[1, 1].set_xlabel("Retimed trajectory time (s)")
    if np.isfinite(acceleration_magnitude[valid]).any():
        axes[1, 1].set_title(
            f"Observed maximum: {np.nanmax(acceleration_magnitude[valid]):.4f} m/s²"
        )

    for axis in axes.flat:
        _mark_invalid_times(axis, time_s, valid)
    figure.suptitle(
        "Offline TCP command preview — actual extrema shown, no robot safety limits"
    )
    figure.text(
        0.5,
        0.01,
        "Generated offline only · robot_execution_approved = false · no SDK or motion command",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    return _save(figure, Path(output_dir), "robot_trajectory_preview.png")


def _set_equal_3d_axes(axis: plt.Axes, points: np.ndarray) -> None:
    finite_points = points[np.isfinite(points).all(axis=1)]
    if len(finite_points) == 0:
        return
    minima = np.min(finite_points, axis=0)
    maxima = np.max(finite_points, axis=0)
    center = 0.5 * (minima + maxima)
    half_range = max(float(np.max(maxima - minima)) * 0.56, 0.02)
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def robot_workspace_preview(
    trajectory: pd.DataFrame,
    calibration_metadata: Mapping[str, Any] | None,
    output_dir: str | Path,
) -> Path | None:
    """Render the finite, valid TCP path and XYZ ranges in robot Base frame."""

    _configure_style()
    del calibration_metadata
    dataframe = _as_dataframe(trajectory)
    valid, reason = _trajectory_status(dataframe)
    if reason is not None or valid is None:
        return None
    position = np.column_stack(
        [_numeric(dataframe, name) for name in (
            "tcp_x_base_m", "tcp_y_base_m", "tcp_z_base_m"
        )]
    )

    figure = plt.figure(figsize=(9.5, 7.4))
    axis = figure.add_subplot(111, projection="3d")
    path = position[valid]
    axis.plot(path[:, 0], path[:, 1], path[:, 2], color=_TCP_COLOR, linewidth=2.0, label="TCP command path")
    axis.scatter(*path[0], color=_BASE_COLORS[2], s=48, label="start")
    axis.scatter(*path[-1], color=_INVALID_COLOR, s=48, marker="s", label="end")
    axis.set_xlabel("TCP X in Base B (m)")
    axis.set_ylabel("TCP Y in Base B (m)")
    axis.set_zlabel("TCP Z in Base B (m)")
    axis.legend(loc="upper left")
    _set_equal_3d_axes(axis, path)

    ranges = [
        f"{label}: {path[:, index].min():.4f} … {path[:, index].max():.4f} m"
        for index, label in enumerate(("X_B", "Y_B", "Z_B"))
    ]
    figure.text(
        0.03,
        0.025,
        "\n".join(ranges) + "\nNo real-robot workspace or safety threshold applied.",
        fontsize=8,
        va="bottom",
    )
    axis.set_title("Offline TCP workspace preview in robot Base frame B")
    figure.subplots_adjust(bottom=0.17)
    return _save(figure, Path(output_dir), "robot_workspace_preview.png")


def _calibration_vectors(
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    hip = _vector3(_metadata_value(metadata, "hip_center_in_base_m"))
    human_x = _vector3(_metadata_value(metadata, "human_x_axis_in_base"))
    human_z = _vector3(_metadata_value(metadata, "human_z_axis_in_base"))
    tool_offset = _vector3(_metadata_value(metadata, "tool_offset_m"))
    if any(value is None for value in (hip, human_x, human_z, tool_offset)):
        return None
    assert hip is not None and human_x is not None and human_z is not None and tool_offset is not None
    human_y = np.cross(human_z, human_x)
    human_y_norm = np.linalg.norm(human_y)
    if human_y_norm <= 1e-12:
        return None
    human_y /= human_y_norm
    rotation = np.column_stack((human_x, human_y, human_z))
    return hip, rotation, tool_offset, human_y


def human_vs_robot_coordinate_preview(
    trajectory: pd.DataFrame,
    calibration_metadata: Mapping[str, Any] | None,
    output_dir: str | Path,
) -> Path | None:
    """Compare the human sagittal path with mapped connection and TCP paths.

    The plotted separation between the mapped connection and TCP paths is the
    actually applied offset in Base coordinates.  The configured ``tool_offset``
    is labeled in frame T and is not silently treated as a Base-frame vector.
    """

    _configure_style()
    dataframe = _as_dataframe(trajectory)
    metadata = _as_metadata(calibration_metadata)
    valid, reason = _trajectory_status(dataframe)
    calibration = _calibration_vectors(metadata)
    if reason is not None or valid is None or calibration is None:
        return None

    hip, rotation, tool_offset, _human_y = calibration
    x_h = _numeric(dataframe, "x_pull_human_m")
    z_h = _numeric(dataframe, "z_pull_human_m")
    human_points = np.column_stack((x_h, np.zeros(len(dataframe)), z_h))
    connection_base = hip[None, :] - human_points @ rotation.T
    tcp_base = np.column_stack(
        [_numeric(dataframe, name) for name in (
            "tcp_x_base_m", "tcp_y_base_m", "tcp_z_base_m"
        )]
    )

    figure = plt.figure(figsize=(12.0, 6.0))
    human_axis = figure.add_subplot(121)
    base_axis = figure.add_subplot(122, projection="3d")

    human_axis.plot(x_h[valid], z_h[valid], color=_CONNECTION_COLOR, linewidth=2.0)
    human_axis.scatter(x_h[valid][0], z_h[valid][0], color=_BASE_COLORS[2], s=40, label="start")
    human_axis.scatter(x_h[valid][-1], z_h[valid][-1], color=_INVALID_COLOR, marker="s", s=40, label="end")
    human_axis.scatter(0.0, 0.0, marker="o", color="black", s=25, label="hip origin H")
    axis_length_h = max(float(np.ptp(x_h[valid])), float(np.ptp(z_h[valid])), 0.1) * 0.32
    human_axis.arrow(0.0, 0.0, axis_length_h, 0.0, width=0.002, color=_BASE_COLORS[0], length_includes_head=True)
    human_axis.arrow(0.0, 0.0, 0.0, axis_length_h, width=0.002, color=_BASE_COLORS[2], length_includes_head=True)
    human_axis.text(axis_length_h, 0.0, "+x_H footward", color=_BASE_COLORS[0], va="bottom")
    human_axis.text(0.0, axis_length_h, "+z_H upward", color=_BASE_COLORS[2], ha="left")
    human_axis.set_xlabel("Pull-point x in human H (m)")
    human_axis.set_ylabel("Pull-point z in human H (m)")
    human_axis.set_aspect("equal", adjustable="datalim")
    human_axis.legend(loc="best")
    human_axis.set_title("Human sagittal reference H")

    connection_path = connection_base[valid]
    tcp_path = tcp_base[valid]
    base_axis.plot(
        connection_path[:, 0], connection_path[:, 1], connection_path[:, 2],
        color=_CONNECTION_COLOR, linestyle="--", linewidth=1.8,
        label="mapped strap point: hip + R_BH p_H",
    )
    base_axis.plot(
        tcp_path[:, 0], tcp_path[:, 1], tcp_path[:, 2],
        color=_TCP_COLOR, linewidth=2.0, label="TCP command after tool offset",
    )
    sample_indices = np.linspace(0, len(connection_path) - 1, min(7, len(connection_path)), dtype=int)
    for index in np.unique(sample_indices):
        base_axis.plot(
            [connection_path[index, 0], tcp_path[index, 0]],
            [connection_path[index, 1], tcp_path[index, 1]],
            [connection_path[index, 2], tcp_path[index, 2]],
            color=_ORIENTATION_COLORS[0], alpha=0.65, linewidth=0.9,
        )

    base_axis.scatter(*hip, color="black", marker="o", s=32, label="hip center in B")
    combined = np.vstack((connection_path, tcp_path, hip[None, :]))
    span = max(float(np.ptp(combined, axis=0).max()), 0.1)
    arrow_length = 0.18 * span
    for index, (axis_name, color) in enumerate(zip(("x_H", "y_H", "z_H"), _BASE_COLORS)):
        vector = rotation[:, index]
        base_axis.quiver(*hip, *(arrow_length * vector), color=color, arrow_length_ratio=0.15)
        endpoint = hip + arrow_length * vector
        base_axis.text(*endpoint, f"+{axis_name} in B", color=color)
    # TCP minus connection equals the negative rotated TCP-to-strap offset.
    actual_offset = tcp_path - connection_path
    actual_offset_norm = np.linalg.norm(actual_offset, axis=1)
    orthogonality_residual = np.linalg.norm(rotation.T @ rotation - np.eye(3))
    figure.text(
        0.51,
        0.025,
        "Transform: p_connection_B = hip_B + R_BH [x_H, 0, z_H]ᵀ\n"
        f"Configured tool_offset_T = [{tool_offset[0]:.4f}, {tool_offset[1]:.4f}, {tool_offset[2]:.4f}] m\n"
        f"Applied offset norm = {np.median(actual_offset_norm):.4f} m; "
        f"‖R_BHᵀR_BH − I‖_F = {orthogonality_residual:.2e}\n"
        "Purple connectors show the applied offset; no lab coordinates are hard-coded.",
        fontsize=8,
        va="bottom",
    )
    base_axis.set_xlabel("X_B (m)")
    base_axis.set_ylabel("Y_B (m)")
    base_axis.set_zlabel("Z_B (m)")
    base_axis.legend(loc="upper left", fontsize=8)
    base_axis.set_title("Mapped connection and TCP command in Base B")
    _set_equal_3d_axes(base_axis, combined)

    figure.suptitle(
        "Human H → robot Base B → TCP T (offline coordinate audit only)"
    )
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 0.95))
    return _save(
        figure,
        Path(output_dir),
        "human_vs_robot_coordinate_preview.png",
    )


def generate_robot_trajectory_visualizations(
    trajectory: pd.DataFrame | None,
    calibration_metadata: Mapping[str, Any] | None,
    output_dir: str | Path,
) -> RobotTrajectoryVisualizationResult:
    """Generate all Stage 6A previews, accounting explicitly for skipped files."""

    _configure_style()
    dataframe = _as_dataframe(trajectory)
    metadata = _as_metadata(calibration_metadata)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _valid, shared_reason = _trajectory_status(dataframe)
    paths: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    if shared_reason is not None:
        for filename in FIGURE_FILENAMES:
            skipped[filename] = shared_reason
        return RobotTrajectoryVisualizationResult(paths=paths, skipped=skipped)

    producers = (
        (
            "robot_trajectory_preview.png",
            lambda: robot_trajectory_preview(dataframe, metadata, destination),
            "finite trajectory samples were unavailable for time preview",
        ),
        (
            "robot_workspace_preview.png",
            lambda: robot_workspace_preview(dataframe, metadata, destination),
            "finite trajectory samples were unavailable for Base workspace preview",
        ),
        (
            "human_vs_robot_coordinate_preview.png",
            lambda: human_vs_robot_coordinate_preview(dataframe, metadata, destination),
            "required H-to-B calibration vectors are missing, non-finite, or degenerate",
        ),
    )
    for filename, producer, skip_reason in producers:
        path = producer()
        if path is None:
            skipped[filename] = skip_reason
        else:
            paths[filename] = path
    return RobotTrajectoryVisualizationResult(paths=paths, skipped=skipped)


__all__ = [
    "FIGURE_FILENAMES",
    "RobotTrajectoryVisualizationResult",
    "robot_trajectory_preview",
    "robot_workspace_preview",
    "human_vs_robot_coordinate_preview",
    "generate_robot_trajectory_visualizations",
]
