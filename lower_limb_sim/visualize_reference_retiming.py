"""Stage 5B plots for phase-normalized and software-retimed reference paths.

The original reference file has no trustworthy timing.  This module therefore
keeps source phase plots separate from software-retimed time plots and never
draws torque, force, or subject curves unless ROM approval and supplied dynamics
make those outputs meaningful.  Estimated-state domain coverage remains an
independent, non-dynamic audit against clean training bounds.
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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .config import L1, L2
from .kinematics import forward_kinematics


FIGURE_FILENAMES = (
    "reference_path_phase.png",
    "reference_retimed_angles.png",
    "reference_speed_comparison.png",
    "reference_pull_path.png",
    "reference_torque_comparison.png",
    "reference_force_comparison.png",
    "reference_subject_comparison.png",
    "reference_domain_coverage.png",
)
KINEMATIC_FIGURES = FIGURE_FILENAMES[:4]
DYNAMIC_FIGURES = FIGURE_FILENAMES[4:7]
DOMAIN_FIGURE = FIGURE_FILENAMES[7]

_PROFILE_ORDER = {"slow": 0, "nominal": 1, "fast": 2}
_PROFILE_COLORS = {
    "slow": "#4C78A8",
    "nominal": "#54A24B",
    "fast": "#E45756",
}
_HIP_COLOR = "#4C78A8"
_KNEE_COLOR = "#F58518"
_PULL_COLOR = "#B279A2"
_INVALID_COLOR = "#E45756"


@dataclass(frozen=True)
class ReferenceRetimingVisualizationResult:
    """Generated files and explicit reasons for all omitted requested plots."""

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


def _as_dataframe(value: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame or None.")
    return value.copy(deep=False)


def _numeric(dataframe: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    for column in candidates:
        if column in dataframe:
            return pd.to_numeric(dataframe[column], errors="coerce").to_numpy(
                dtype=float
            )
    return None


def _angle_degrees(
    dataframe: pd.DataFrame,
    *,
    radian_candidates: Sequence[str],
    degree_candidates: Sequence[str],
) -> np.ndarray | None:
    value = _numeric(dataframe, degree_candidates)
    if value is not None:
        return value
    value = _numeric(dataframe, radian_candidates)
    return None if value is None else np.rad2deg(value)


def _finite(*arrays: np.ndarray | None) -> np.ndarray:
    available = [np.asarray(array) for array in arrays if array is not None]
    if not available:
        return np.empty(0, dtype=bool)
    mask = np.ones(len(available[0]), dtype=bool)
    for value in available:
        if len(value) != len(mask):
            raise ValueError("Plotting arrays must have equal lengths.")
        mask &= np.isfinite(value)
    return mask


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for nested_name in (
        "retiming",
        "reference",
        "source",
        "rom",
        "rom_audit",
        "geometry",
    ):
        nested = metadata.get(nested_name)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def _profile_sort_key(profile: str) -> tuple[int, str]:
    normalized = profile.strip().lower()
    return (_PROFILE_ORDER.get(normalized, len(_PROFILE_ORDER)), normalized)


def _profile_color(profile: str, index: int) -> Any:
    return _PROFILE_COLORS.get(profile.strip().lower(), plt.get_cmap("tab10")(index % 10))


def _normalize_retimed(
    value: Mapping[str, pd.DataFrame] | pd.DataFrame | None,
) -> list[tuple[str, pd.DataFrame]]:
    items: list[tuple[str, pd.DataFrame]] = []
    if value is None:
        return items
    if isinstance(value, pd.DataFrame):
        profile_column = next(
            (name for name in ("profile", "speed_profile") if name in value),
            None,
        )
        if profile_column is None:
            items.append(("retimed", value.copy(deep=False)))
        else:
            for profile, group in value.groupby(profile_column, sort=False):
                items.append((str(profile), group.copy(deep=False)))
    elif isinstance(value, Mapping):
        for profile, dataframe in value.items():
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Each retimed_by_profile value must be a DataFrame.")
            items.append((str(profile), dataframe.copy(deep=False)))
    else:
        raise TypeError("retimed_by_profile must be a mapping, DataFrame, or None.")
    return sorted(items, key=lambda item: _profile_sort_key(item[0]))


def _normalize_dynamics(
    value: Mapping[Any, Any] | pd.DataFrame | None,
) -> list[tuple[str, str, pd.DataFrame]]:
    """Accept profile->subject->frame, tuple keys, or one flat DataFrame."""

    items: list[tuple[str, str, pd.DataFrame]] = []
    if value is None:
        return items
    if isinstance(value, pd.DataFrame):
        profile_column = next(
            (name for name in ("profile", "speed_profile") if name in value),
            None,
        )
        subject_column = next(
            (name for name in ("subject_id", "virtual_subject_id") if name in value),
            None,
        )
        if profile_column and subject_column:
            for (profile, subject), group in value.groupby(
                [profile_column, subject_column], sort=False
            ):
                items.append((str(profile), str(subject), group.copy(deep=False)))
        else:
            items.append(("retimed", "subject", value.copy(deep=False)))
    elif isinstance(value, Mapping):
        for outer_key, outer_value in value.items():
            if isinstance(outer_value, pd.DataFrame):
                if isinstance(outer_key, tuple) and len(outer_key) == 2:
                    profile, subject = outer_key
                else:
                    profile = str(outer_key)
                    subject_column = next(
                        (
                            name
                            for name in ("subject_id", "virtual_subject_id")
                            if name in outer_value
                        ),
                        None,
                    )
                    if subject_column and outer_value[subject_column].nunique() == 1:
                        subject = outer_value[subject_column].iloc[0]
                    else:
                        subject = "subject"
                items.append(
                    (str(profile), str(subject), outer_value.copy(deep=False))
                )
            elif isinstance(outer_value, Mapping):
                for subject, dataframe in outer_value.items():
                    if not isinstance(dataframe, pd.DataFrame):
                        raise TypeError(
                            "Each dynamics profile/subject value must be a DataFrame."
                        )
                    items.append(
                        (str(outer_key), str(subject), dataframe.copy(deep=False))
                    )
            else:
                raise TypeError(
                    "dynamics_by_profile_subject must contain DataFrames or nested mappings."
                )
    else:
        raise TypeError(
            "dynamics_by_profile_subject must be a mapping, DataFrame, or None."
        )
    return sorted(
        items,
        key=lambda item: (_profile_sort_key(item[0]), item[1]),
    )


def _phase(dataframe: pd.DataFrame) -> np.ndarray:
    value = _numeric(
        dataframe,
        ("global_phase", "normalized_phase", "phase", "phase_normalized"),
    )
    if value is None:
        if len(dataframe) <= 1:
            return np.zeros(len(dataframe), dtype=float)
        return np.linspace(0.0, 1.0, len(dataframe))
    finite = value[np.isfinite(value)]
    if finite.size and np.nanmax(finite) > 1.5:
        value = value / 100.0
    return value


def _time(dataframe: pd.DataFrame) -> np.ndarray | None:
    value = _numeric(dataframe, ("time_s", "retimed_time_s"))
    if value is None or not np.isfinite(value).any():
        return None
    return value


def _save(
    figure: plt.Figure,
    output_directory: Path,
    filename: str,
) -> Path:
    path = output_directory / filename
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _source_timing_note(metadata: Mapping[str, Any]) -> str:
    status = str(
        _metadata_value(metadata, "source_timing_status", "timing_status")
        or "unknown"
    )
    return (
        f"Source timing: {status}; displayed seconds are software-retimed and "
        "are not recovered source timing"
    )


def _rom_gate(metadata: Mapping[str, Any], has_dynamics: bool) -> tuple[bool, str]:
    key_value = _metadata_value(
        metadata,
        "rom_approved",
        "ROM_approved",
        "range_of_motion_approved",
        "rom_status",
        "ROM_status",
        "rom_approval_status",
    )
    if isinstance(key_value, bool):
        approved = key_value
    elif key_value is None:
        approved = has_dynamics
    else:
        approved = str(key_value).strip().lower() in {
            "approved",
            "confirmed",
            "true",
            "yes",
            "pass",
            "passed",
        }
    explicit_dynamics = _metadata_value(
        metadata,
        "dynamics_allowed",
        "dynamics_available",
        "dynamics_used",
    )
    if explicit_dynamics is False:
        approved = False
    if not approved:
        return (
            False,
            "approved ROM is unavailable; torque, force, and subject-dynamics "
            "figures were not generated and no synthetic curves were drawn",
        )
    if not has_dynamics:
        return (
            False,
            "ROM is approved, but no profile/subject dynamics dataframe was supplied",
        )
    return True, ""


def _joint_limits(
    metadata: Mapping[str, Any],
    joint: str,
) -> tuple[float, float, str] | None:
    limit_source = "Approved"
    value = _metadata_value(
        metadata,
        f"approved_{joint}_range_deg",
        f"{joint}_rom_deg",
        f"mapped_{joint}_range_deg",
    )
    if value is None:
        ranges = _metadata_value(metadata, "approved_rom_deg", "rom_ranges_deg")
        if isinstance(ranges, Mapping):
            value = ranges.get(joint)
            if value is None:
                value = ranges.get(f"q_{joint}")
    if value is None:
        rom_audit = metadata.get("rom_audit")
        if isinstance(rom_audit, Mapping):
            approved = rom_audit.get("approved_angle_range_deg")
            configured = rom_audit.get("configured_angle_range_deg")
            if isinstance(approved, Mapping):
                value = approved.get(joint)
            if value is None and isinstance(configured, Mapping):
                value = configured.get(joint)
                limit_source = "Configured"
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None
    return lower, upper, limit_source


def _draw_limits(
    axis: plt.Axes,
    limits: tuple[float, float, str] | None,
    joint: str,
) -> None:
    if limits is None:
        return
    lower, upper, limit_source = limits
    axis.axhline(
        lower,
        color=_INVALID_COLOR,
        linestyle=":",
        linewidth=0.9,
        label=f"{limit_source} {joint} lower ROM ({lower:g}°)",
    )
    axis.axhline(
        upper,
        color=_INVALID_COLOR,
        linestyle="--",
        linewidth=0.9,
        label=f"{limit_source} {joint} upper ROM ({upper:g}°)",
    )


def _plot_reference_phase(
    phase_path: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    if phase_path.empty:
        return None
    phase_percent = 100.0 * _phase(phase_path)
    angle_specs = {
        "hip": {
            "raw": _angle_degrees(
                phase_path,
                radian_candidates=("q_hip_raw_rad",),
                degree_candidates=("q_hip_raw_deg",),
            ),
            "smoothed": _angle_degrees(
                phase_path,
                radian_candidates=("q_hip_smoothed_rad", "q_hip_rad"),
                degree_candidates=("q_hip_smoothed_deg", "q_hip_deg"),
            ),
            "mapped": _angle_degrees(
                phase_path,
                radian_candidates=(
                    "q_hip_mapped_rad",
                    "q_hip_rom_mapped_rad",
                    "q_hip_reference_rad",
                ),
                degree_candidates=("q_hip_mapped_deg", "q_hip_rom_mapped_deg"),
            ),
        },
        "knee": {
            "raw": _angle_degrees(
                phase_path,
                radian_candidates=("q_knee_raw_rad",),
                degree_candidates=("q_knee_raw_deg",),
            ),
            "smoothed": _angle_degrees(
                phase_path,
                radian_candidates=("q_knee_smoothed_rad", "q_knee_rad"),
                degree_candidates=("q_knee_smoothed_deg", "q_knee_deg"),
            ),
            "mapped": _angle_degrees(
                phase_path,
                radian_candidates=(
                    "q_knee_mapped_rad",
                    "q_knee_rom_mapped_rad",
                    "q_knee_reference_rad",
                ),
                degree_candidates=("q_knee_mapped_deg", "q_knee_rom_mapped_deg"),
            ),
        },
    }
    if all(
        value is None
        for joint_values in angle_specs.values()
        for value in joint_values.values()
    ):
        return None
    for values in angle_specs.values():
        if (
            values["mapped"] is not None
            and values["smoothed"] is not None
            and np.allclose(values["mapped"], values["smoothed"], equal_nan=True)
        ):
            values["mapped"] = None
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, joint in zip(axes, ("hip", "knee")):
        color = _HIP_COLOR if joint == "hip" else _KNEE_COLOR
        values = angle_specs[joint]
        if values["raw"] is not None:
            valid = _finite(phase_percent, values["raw"])
            axis.plot(
                phase_percent[valid],
                values["raw"][valid],
                color="#999999",
                alpha=0.7,
                linewidth=1.0,
                label=f"Source {joint} angle (raw)",
            )
        if values["smoothed"] is not None:
            valid = _finite(phase_percent, values["smoothed"])
            axis.plot(
                phase_percent[valid],
                values["smoothed"][valid],
                color=color,
                linewidth=1.8,
                label=f"Source {joint} path (smoothed)",
            )
        if values["mapped"] is not None:
            valid = _finite(phase_percent, values["mapped"])
            axis.plot(
                phase_percent[valid],
                values["mapped"][valid],
                color=_PULL_COLOR,
                linewidth=1.5,
                linestyle="--",
                label=f"ROM-mapped {joint} path",
            )
        _draw_limits(axis, _joint_limits(metadata, joint), joint)
        axis.set_ylabel(f"{joint.title()} angle (deg)")
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            ncol=1,
            fontsize=8,
        )
    axes[-1].set_xlabel("Normalized reference-path phase (%)")
    axes[0].set_title("Reference path by phase — source timing unknown")
    axes[0].text(
        0.01,
        0.12,
        "Phase preserves path order only; it is not an original time axis",
        transform=axes[0].transAxes,
    )
    return _save(figure, destination, "reference_path_phase.png")


def _plot_retimed_angles(
    retimed: list[tuple[str, pd.DataFrame]],
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    if not retimed:
        return None
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    plotted = False
    for profile_index, (profile, dataframe) in enumerate(retimed):
        time = _time(dataframe)
        if time is None:
            continue
        color = _profile_color(profile, profile_index)
        for axis, joint in zip(axes, ("hip", "knee")):
            angle = _angle_degrees(
                dataframe,
                radian_candidates=(f"q_{joint}_rad",),
                degree_candidates=(f"q_{joint}_deg",),
            )
            if angle is None:
                continue
            valid = _finite(time, angle)
            if np.any(valid):
                axis.plot(
                    time[valid],
                    angle[valid],
                    color=color,
                    label=f"{profile} retimed profile",
                )
                plotted = True
    if not plotted:
        plt.close(figure)
        return None
    for axis, joint in zip(axes, ("hip", "knee")):
        _draw_limits(axis, _joint_limits(metadata, joint), joint)
        axis.set_ylabel(f"{joint.title()} angle (deg)")
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            ncol=1,
            fontsize=8,
        )
    axes[-1].set_xlabel("Software-retimed time (s)")
    axes[0].set_title(
        "Reference joint path under software-retimed duration profiles\n"
        + _source_timing_note(metadata),
        fontsize=10,
    )
    return _save(figure, destination, "reference_retimed_angles.png")


def _plot_speed_comparison(
    retimed: list[tuple[str, pd.DataFrame]],
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    if not retimed:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    specifications = (
        (axes[0, 0], "dq_hip_rad_s", "Hip velocity (rad/s)"),
        (axes[0, 1], "dq_knee_rad_s", "Knee velocity (rad/s)"),
        (axes[1, 0], "ddq_hip_rad_s2", "Hip acceleration (rad/s²)"),
        (axes[1, 1], "ddq_knee_rad_s2", "Knee acceleration (rad/s²)"),
    )
    plotted = False
    for profile_index, (profile, dataframe) in enumerate(retimed):
        time = _time(dataframe)
        if time is None:
            continue
        color = _profile_color(profile, profile_index)
        for axis, column, _ in specifications:
            values = _numeric(dataframe, (column,))
            if values is None:
                continue
            valid = _finite(time, values)
            if np.any(valid):
                axis.plot(time[valid], values[valid], color=color, label=profile)
                plotted = True
    for axis, _, ylabel in specifications:
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Software-retimed time (s)")
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "Retimed derivative unavailable — no curve synthesized",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    figure.suptitle(
        "Retimed speed and acceleration comparison\n"
        + _source_timing_note(metadata),
        fontsize=11,
    )
    # The figure itself remains useful as an explicit availability audit even if
    # derivatives are absent; no zero-valued placeholder is inserted.
    if not plotted:
        figure.suptitle(
            "Retimed speed comparison — derivatives unavailable\n"
            + _source_timing_note(metadata),
            fontsize=11,
        )
    return _save(figure, destination, "reference_speed_comparison.png")


def _length(metadata: Mapping[str, Any], name: str, fallback: float) -> float:
    value = _metadata_value(metadata, name, f"{name}_m", f"{name.lower()}_m")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if np.isfinite(parsed) and parsed > 0.0 else fallback


def _pull_path(
    dataframe: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    x_pull = _numeric(dataframe, ("x_pull_m", "x_pull"))
    z_pull = _numeric(dataframe, ("z_pull_m", "z_pull"))
    if x_pull is not None and z_pull is not None:
        return x_pull, z_pull
    q_hip = _numeric(dataframe, ("q_hip_rad",))
    q_knee = _numeric(dataframe, ("q_knee_rad",))
    if q_hip is None or q_knee is None:
        return None
    _, _, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        _length(metadata, "L1", float(L1)),
        _length(metadata, "L2", float(L2)),
    )
    return np.asarray(x_pull, dtype=float), np.asarray(z_pull, dtype=float)


def _plot_pull_path(
    retimed: list[tuple[str, pd.DataFrame]],
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    if not retimed:
        return None
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    plotted = False
    for profile_index, (profile, dataframe) in enumerate(retimed):
        path = _pull_path(dataframe, metadata)
        if path is None:
            continue
        x_pull, z_pull = path
        valid = _finite(x_pull, z_pull)
        if not np.any(valid):
            continue
        line_styles = ("-", "--", ":", "-.")
        axis.plot(
            x_pull[valid],
            z_pull[valid],
            color=_profile_color(profile, profile_index),
            linewidth=max(1.2, 3.6 - 0.8 * profile_index),
            linestyle=line_styles[profile_index % len(line_styles)],
            alpha=0.85,
            label=f"{profile} profile — same geometric path",
        )
        plotted = True
    if not plotted:
        plt.close(figure)
        return None
    axis.axhline(0.0, color="#222222", linewidth=1.2, label="Bed: z = 0")
    axis.set_xlabel("Bed direction x (m)")
    axis.set_ylabel("Vertical z (m)")
    axis.set_title("Software-retimed L2 equivalent pull path (not anatomical ankle)")
    axis.text(
        0.02,
        0.09,
        "Retiming changes traversal speed, not the phase-normalized geometry",
        transform=axis.transAxes,
        va="bottom",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best", fontsize=8)
    return _save(figure, destination, "reference_pull_path.png")


def _preferred_subject(
    dynamics: list[tuple[str, str, pd.DataFrame]],
    metadata: Mapping[str, Any],
) -> str | None:
    subjects = [subject for _, subject, _ in dynamics]
    requested = _metadata_value(metadata, "reference_subject_id", "subject_id")
    if requested is not None and str(requested) in subjects:
        return str(requested)
    if "baseline" in subjects:
        return "baseline"
    return subjects[0] if subjects else None


def _plot_torque_comparison(
    dynamics: list[tuple[str, str, pd.DataFrame]],
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    subject = _preferred_subject(dynamics, metadata)
    if subject is None:
        return None
    selected = [(profile, frame) for profile, candidate, frame in dynamics if candidate == subject]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    plotted = False
    for profile_index, (profile, dataframe) in enumerate(selected):
        time = _time(dataframe)
        if time is None:
            continue
        color = _profile_color(profile, profile_index)
        for axis, joint in zip(axes, ("hip", "knee")):
            torque = _numeric(dataframe, (f"tau_total_{joint}_nm",))
            if torque is None:
                continue
            valid = _finite(time, torque)
            if np.any(valid):
                axis.plot(time[valid], torque[valid], color=color, label=profile)
                plotted = True
    if not plotted:
        plt.close(figure)
        return None
    axes[0].set_title(f"Retimed total joint torque — virtual subject: {subject}")
    for axis, joint in zip(axes, ("hip", "knee")):
        axis.set_ylabel(f"{joint.title()} torque (N·m)")
        axis.set_xlabel("Software-retimed time (s)")
        axis.legend(fontsize=8)
    return _save(figure, destination, "reference_torque_comparison.png")


def _force_values(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    fx = _numeric(dataframe, ("fx_robot_on_leg_n",))
    fz = _numeric(dataframe, ("fz_robot_on_leg_n",))
    magnitude = _numeric(dataframe, ("force_magnitude_n",))
    if magnitude is None and fx is not None and fz is not None:
        magnitude = np.hypot(fx, fz)
    return fx, fz, magnitude


def _valid_force_mask(dataframe: pd.DataFrame, length: int) -> np.ndarray:
    if "force_mapping_valid" not in dataframe:
        return np.ones(length, dtype=bool)
    values = dataframe["force_mapping_valid"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0.0).to_numpy(float) != 0.0


def _plot_force_comparison(
    dynamics: list[tuple[str, str, pd.DataFrame]],
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    subject = _preferred_subject(dynamics, metadata)
    if subject is None:
        return None
    selected = [(profile, frame) for profile, candidate, frame in dynamics if candidate == subject]
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    plotted = False
    for profile_index, (profile, dataframe) in enumerate(selected):
        time = _time(dataframe)
        fx, fz, magnitude = _force_values(dataframe)
        if time is None or fx is None or fz is None or magnitude is None:
            continue
        color = _profile_color(profile, profile_index)
        mapping_valid = _valid_force_mask(dataframe, len(dataframe))
        for axis, values in zip(axes, (fx, fz, magnitude)):
            valid = _finite(time, values) & mapping_valid
            if np.any(valid):
                axis.plot(time[valid], values[valid], color=color, label=profile)
                plotted = True
    if not plotted:
        plt.close(figure)
        return None
    labels = ("Fx robot on leg (N)", "Fz robot on leg (N)", "Endpoint |F| (N)")
    axes[0].set_title(f"Retimed endpoint force — virtual subject: {subject}")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.set_xlabel("Software-retimed time (s)")
        axis.legend(fontsize=8)
    return _save(figure, destination, "reference_force_comparison.png")


def _comparison_profile(
    dynamics: list[tuple[str, str, pd.DataFrame]],
    comparison: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str | None:
    profiles = [profile for profile, _, _ in dynamics]
    requested = _metadata_value(
        metadata,
        "subject_comparison_profile",
        "comparison_profile",
    )
    if requested is not None and str(requested) in profiles:
        return str(requested)
    for column in ("profile", "speed_profile"):
        if column in comparison:
            marked = next(
                (
                    flag
                    for flag in ("selected_for_subject_comparison", "is_reference_profile")
                    if flag in comparison
                ),
                None,
            )
            if marked is not None:
                selected = comparison.loc[comparison[marked].fillna(False).astype(bool)]
                if not selected.empty:
                    candidate = str(selected[column].iloc[0])
                    if candidate in profiles:
                        return candidate
    if "nominal" in profiles:
        return "nominal"
    return profiles[0] if profiles else None


def _plot_subject_comparison(
    dynamics: list[tuple[str, str, pd.DataFrame]],
    comparison: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    profile = _comparison_profile(dynamics, comparison, metadata)
    if profile is None:
        return None
    selected = [(subject, frame) for candidate, subject, frame in dynamics if candidate == profile]
    if not selected:
        return None
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    plotted = False
    palette = plt.get_cmap("tab10")
    for subject_index, (subject, dataframe) in enumerate(selected):
        time = _time(dataframe)
        hip = _numeric(dataframe, ("tau_total_hip_nm",))
        knee = _numeric(dataframe, ("tau_total_knee_nm",))
        magnitude = _force_values(dataframe)[2]
        if time is None or hip is None or knee is None or magnitude is None:
            continue
        color = palette(subject_index % 10)
        for axis, values in zip(axes, (hip, knee, magnitude)):
            valid = _finite(time, values)
            if np.any(valid):
                axis.plot(time[valid], values[valid], color=color, label=subject)
                plotted = True
    if not plotted:
        plt.close(figure)
        return None
    labels = ("Hip total torque (N·m)", "Knee total torque (N·m)", "Endpoint |F| (N)")
    axes[0].set_title(f"Virtual-subject comparison — retimed profile: {profile}")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.set_xlabel("Software-retimed time (s)")
        axis.legend(ncol=min(4, len(selected)), fontsize=8)
    return _save(figure, destination, "reference_subject_comparison.png")


def _domain_membership(dataframe: pd.DataFrame) -> np.ndarray | None:
    column = next(
        (
            name
            for name in (
                "domain_membership_estimated",
                "inside_estimated_domain",
                "in_domain",
            )
            if name in dataframe
        ),
        None,
    )
    if column is None:
        return None
    values = dataframe[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).to_numpy(float) != 0.0
    return values.astype(str).str.strip().str.lower().isin(
        ("true", "yes", "1", "inside", "in_domain")
    ).to_numpy()


def _plot_domain_coverage(
    domain_audit: pd.DataFrame,
    destination: Path,
) -> Path | None:
    if domain_audit.empty:
        return None
    membership = _domain_membership(domain_audit)
    if membership is None:
        return None
    profile_column = next(
        (name for name in ("profile", "speed_profile") if name in domain_audit),
        None,
    )
    subject_column = next(
        (name for name in ("subject_id", "virtual_subject_id") if name in domain_audit),
        None,
    )
    working = domain_audit.copy()
    working["_membership"] = membership
    if profile_column is None:
        working["_profile"] = "retimed"
        profile_column = "_profile"
    if subject_column is None:
        working["_subject"] = "all subjects"
        subject_column = "_subject"
    categories = [
        (str(profile), str(subject), group)
        for (profile, subject), group in working.groupby(
            [profile_column, subject_column], sort=False
        )
    ]
    categories.sort(key=lambda item: (_profile_sort_key(item[0]), item[1]))
    if not categories:
        return None

    figure, (coverage_axis, timeline_axis) = plt.subplots(
        1,
        2,
        figsize=(13, max(5.5, 0.45 * len(categories) + 2.5)),
        gridspec_kw={"width_ratios": (1.0, 2.3)},
    )
    labels: list[str] = []
    percentages: list[float] = []
    for row, (profile, subject, group) in enumerate(categories):
        label = f"{profile} / {subject}"
        labels.append(label)
        member = group["_membership"].to_numpy(bool)
        percentages.append(100.0 * float(np.mean(member)) if len(member) else np.nan)
        phase = _phase(group)
        valid = np.isfinite(phase)
        timeline_axis.scatter(
            100.0 * phase[valid & member],
            np.full(np.count_nonzero(valid & member), row),
            color="#54A24B",
            marker="o",
            s=15,
        )
        timeline_axis.scatter(
            100.0 * phase[valid & ~member],
            np.full(np.count_nonzero(valid & ~member), row),
            color=_INVALID_COLOR,
            marker="x",
            s=20,
        )
    y = np.arange(len(categories))
    coverage_axis.barh(y, percentages, color="#4C78A8")
    coverage_axis.set_yticks(y, labels)
    coverage_axis.set_xlim(0.0, 100.0)
    coverage_axis.set_xlabel("Estimated-state in-domain samples (%)")
    for row, value in enumerate(percentages):
        if np.isfinite(value):
            coverage_axis.text(min(value + 1.0, 98.0), row, f"{value:.1f}%", va="center", fontsize=8)
    timeline_axis.set_yticks(y, labels)
    timeline_axis.set_xlim(0.0, 100.0)
    timeline_axis.set_xlabel("Retimed path phase (%)")
    timeline_axis.set_title("Estimated-state domain membership across path")
    timeline_axis.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="None", color="#54A24B", label="In estimated domain"),
            Line2D([], [], marker="x", linestyle="None", color=_INVALID_COLOR, label="Outside estimated domain"),
        ),
        fontsize=8,
    )
    figure.suptitle("Reference retiming domain coverage by profile and subject")
    return _save(figure, destination, "reference_domain_coverage.png")


def generate_reference_retiming_visualizations(
    phase_path: pd.DataFrame,
    retimed_by_profile: Mapping[str, pd.DataFrame] | pd.DataFrame | None,
    dynamics_by_profile_subject: Mapping[Any, Any] | pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    domain_audit: pd.DataFrame | None,
    metadata: Mapping[str, Any] | None,
    output_directory: str | Path,
) -> ReferenceRetimingVisualizationResult:
    """Generate all eight Stage 5B figures without inventing missing dynamics.

    The first four figures describe phase geometry and software-retimed
    kinematics.  Domain membership is evaluated independently from clean
    training bounds and therefore does not require dynamics.  Torque, force,
    and subject-comparison plots require approved ROM and supplied dynamics.  A
    requested filename is always present in exactly one of ``paths`` or
    ``skipped``.
    """

    _configure_style()
    phase = _as_dataframe(phase_path, "phase_path")
    retimed = _normalize_retimed(retimed_by_profile)
    dynamics = _normalize_dynamics(dynamics_by_profile_subject)
    comparison_frame = _as_dataframe(comparison, "comparison")
    domain_frame = _as_dataframe(domain_audit, "domain_audit")
    metadata_map: Mapping[str, Any] = metadata if metadata is not None else {}
    if not isinstance(metadata_map, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    skipped: dict[str, str] = {}

    producers = (
        (
            "reference_path_phase.png",
            lambda: _plot_reference_phase(phase, metadata_map, destination),
            "phase path lacks finite raw, smoothed, or mapped joint angles",
        ),
        (
            "reference_retimed_angles.png",
            lambda: _plot_retimed_angles(retimed, metadata_map, destination),
            "retimed profiles lack finite time and joint-angle columns",
        ),
        (
            "reference_speed_comparison.png",
            lambda: _plot_speed_comparison(retimed, metadata_map, destination),
            "no retimed profile was supplied",
        ),
        (
            "reference_pull_path.png",
            lambda: _plot_pull_path(retimed, metadata_map, destination),
            "retimed profiles lack finite pull-point geometry or reconstructable angles",
        ),
    )
    for filename, producer, reason in producers:
        path = producer()
        if path is None:
            skipped[filename] = reason
        else:
            paths[path.name] = path

    domain_path = _plot_domain_coverage(domain_frame, destination)
    if domain_path is None:
        skipped[DOMAIN_FIGURE] = (
            "domain audit lacks estimated-state membership samples"
        )
    else:
        paths[domain_path.name] = domain_path

    dynamics_allowed, gate_reason = _rom_gate(metadata_map, bool(dynamics))
    if not dynamics_allowed:
        for filename in DYNAMIC_FIGURES:
            skipped[filename] = gate_reason
    else:
        dynamic_producers = (
            (
                "reference_torque_comparison.png",
                lambda: _plot_torque_comparison(dynamics, metadata_map, destination),
                "supplied dynamics lack finite total hip/knee torque curves",
            ),
            (
                "reference_force_comparison.png",
                lambda: _plot_force_comparison(dynamics, metadata_map, destination),
                "supplied dynamics lack finite valid endpoint-force curves",
            ),
            (
                "reference_subject_comparison.png",
                lambda: _plot_subject_comparison(
                    dynamics,
                    comparison_frame,
                    metadata_map,
                    destination,
                ),
                "no common retimed profile has subject torque and force curves",
            ),
        )
        for filename, producer, reason in dynamic_producers:
            path = producer()
            if path is None:
                skipped[filename] = reason
            else:
                paths[path.name] = path

    for filename in FIGURE_FILENAMES:
        if filename not in paths and filename not in skipped:
            skipped[filename] = "no visualization producer accounted for this output"
    return ReferenceRetimingVisualizationResult(paths=paths, skipped=skipped)


__all__ = [
    "DOMAIN_FIGURE",
    "DYNAMIC_FIGURES",
    "FIGURE_FILENAMES",
    "KINEMATIC_FIGURES",
    "ReferenceRetimingVisualizationResult",
    "generate_reference_retiming_visualizations",
]
