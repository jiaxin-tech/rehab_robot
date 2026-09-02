"""Paper-ready, read-only visualizations of the frozen admissible region.

This module consumes the already frozen active reference and
``REFERENCE_CENTERED_ADMISSIBLE_REGION_V1`` artifacts.  It does not refit an
identification domain, change a gate, run sequential personalization, or
interact with robot hardware.  The visualized region is an offline model-use
region, not a real-robot or human-safety region.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .admissible_personalization_region import (
    DEFAULT_REGION_DIRECTORY,
    REAL_ROBOT_SAFETY_REGION_STATUS,
    REGION_CLASSIFICATION,
    REGION_VERSION,
    load_admissible_personalization_region,
)
from .continuous_reference_neighborhood import (
    GENERATOR_VERSION,
    generate_personalized_trajectory,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_EXPERIMENT_MANIFEST_PATH,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    PROJECT_ROOT,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from .reference_release import load_frozen_active_reference


VISUALIZATION_VERSION = "ADMISSIBLE_REGION_VISUALIZATION_V1"
ARTIFACT_STATUS = "FORMAL_OFFLINE_EXPLANATORY_VISUALIZATION"
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "lower_limb_sim"
    / "formal_artifacts"
    / "admissible_region_visualization"
)

PNG_FILENAMES = (
    "reference_admissible_region_overview.png",
    "hip_corridor_vs_phase_detailed.png",
    "knee_corridor_vs_phase_detailed.png",
    "pull_reference_corridor.png",
    "pull_deviation_vs_phase.png",
    "identification_region_parameter_slices.png",
    "identification_directional_support_summary.png",
    "dynamic_rehabilitation_snapshots.png",
)
GIF_FILENAME = "dynamic_rehabilitation_process.gif"
SUMMARY_FILENAME = "VISUALIZATION_SUMMARY.md"
METADATA_FILENAME = "metadata.json"

REPRESENTATIVE_POINTS: dict[str, tuple[float, float, float]] = {
    "neutral": (0.0, 0.0, 0.0),
    "hip negative": (-3.0, 0.0, 0.0),
    "knee negative": (0.0, -3.0, 0.0),
    "positive phase": (0.0, 0.0, 0.03),
    "negative phase": (0.0, 0.0, -0.03),
    "combined perturbation": (-3.0, -3.0, 0.03),
}
INITIAL_NEIGHBORS: dict[str, tuple[float, float, float]] = {
    "hip +": (1.0, 0.0, 0.0),
    "hip -": (-1.0, 0.0, 0.0),
    "knee +": (0.0, 1.0, 0.0),
    "knee -": (0.0, -1.0, 0.0),
    "phase +": (0.0, 0.0, 0.01),
    "phase -": (0.0, 0.0, -0.01),
}

_COLORS = {
    "reference": "#111111",
    "lower": "#2878B5",
    "upper": "#E67E22",
    "band": "#8DB9D8",
    "admissible": "#2B9E4B",
    "domain_rejected": "#D93434",
    "other_rejected": "#8C8C8C",
    "pull": "#6A3D9A",
    "hip": "#2878B5",
    "knee": "#E67E22",
    "phase": "#6A3D9A",
}

_PLOT_STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "lines.linewidth": 1.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def _project_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _save_figure(figure: plt.Figure, output: Path, filename: str) -> Path:
    path = output / filename
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_inputs() -> tuple[
    Any,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    region = load_admissible_personalization_region()
    reference = load_frozen_active_reference().trajectory.copy(deep=True)
    parameter_path = DEFAULT_REGION_DIRECTORY / "parameter_space_admissibility.csv"
    parameter_map = pd.read_csv(parameter_path)
    required_parameters = {
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "trajectory_admissible",
        "invalid_reason",
        "domain_coverage",
    }
    if not required_parameters.issubset(parameter_map.columns):
        missing = sorted(required_parameters - set(parameter_map.columns))
        raise RuntimeError(f"formal parameter map lacks columns: {missing}")
    if len(parameter_map) != int(region.summary["parameter_sample_count"]):
        raise RuntimeError("formal parameter-map row count is inconsistent")
    if len(reference) != len(region.joint_corridor):
        raise RuntimeError("active reference and corridor sample counts differ")
    np.testing.assert_allclose(
        reference["q_hip_rad"],
        region.joint_corridor["q_hip_ref_rad"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference["q_knee_rad"],
        region.joint_corridor["q_knee_ref_rad"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference["x_pull_m"],
        region.pull_corridor["x_pull_ref_m"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference["z_pull_m"],
        region.pull_corridor["z_pull_ref_m"],
        atol=1e-12,
        rtol=0.0,
    )
    candidates: dict[str, pd.DataFrame] = {}
    for label, (hip, knee, phase) in REPRESENTATIVE_POINTS.items():
        generated = generate_personalized_trajectory(None, hip, knee, phase)
        if not generated.constraints.trajectory_feasible:
            raise RuntimeError(f"representative trajectory is no longer admissible: {label}")
        candidates[label] = generated.trajectory.copy(deep=True)
    return region, reference, parameter_map, candidates


def _alpha_text(record: Mapping[str, Any]) -> str:
    return (
        f"({float(record['hip_delta']):+.2f}, "
        f"{float(record['knee_delta']):+.2f}, "
        f"{float(record['phase_delta']):+.3f})"
    )


def _joint_corridor_panel(
    axis: plt.Axes,
    corridor: pd.DataFrame,
    joint: str,
    *,
    compact: bool = False,
) -> None:
    phase = corridor["global_phase"].to_numpy(float)
    lower = np.rad2deg(corridor[f"q_{joint}_min_rad"].to_numpy(float))
    reference = np.rad2deg(corridor[f"q_{joint}_ref_rad"].to_numpy(float))
    upper = np.rad2deg(corridor[f"q_{joint}_max_rad"].to_numpy(float))
    axis.fill_between(
        phase,
        lower,
        upper,
        color=_COLORS["band"],
        alpha=0.34,
        label="phase-wise admissible envelope",
    )
    axis.plot(phase, lower, color=_COLORS["lower"], label="lower envelope")
    axis.plot(
        phase,
        reference,
        color=_COLORS["reference"],
        linewidth=2.6,
        label="active reference",
    )
    axis.plot(phase, upper, color=_COLORS["upper"], label="upper envelope")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("global phase")
    axis.set_ylabel(f"q_{joint} (deg)")
    axis.set_title(f"{joint.capitalize()} reference-centered corridor")
    if not compact:
        axis.legend(loc="best", ncol=2)


def _detailed_joint_figure(
    region: Any,
    output: Path,
    joint: str,
) -> Path:
    corridor = region.joint_corridor
    summary = region.summary["joint_extrema"]
    negative = summary[f"maximum_negative_{joint}_deviation_deg"]
    positive = summary[f"maximum_positive_{joint}_deviation_deg"]
    phase = corridor["global_phase"].to_numpy(float)
    lower = np.rad2deg(corridor[f"q_{joint}_min_rad"].to_numpy(float))
    upper = np.rad2deg(corridor[f"q_{joint}_max_rad"].to_numpy(float))
    width = upper - lower
    figure, (axis, width_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    _joint_corridor_panel(axis, corridor, joint)
    for label, record, color in (
        ("maximum negative", negative, _COLORS["lower"]),
        ("maximum positive", positive, _COLORS["upper"]),
    ):
        location = int(record["sample_index"])
        reference_value = np.rad2deg(
            float(corridor.loc[location, f"q_{joint}_ref_rad"])
        )
        value = float(record["value"])
        endpoint = reference_value + value
        axis.plot(
            [float(record["global_phase"]), float(record["global_phase"])],
            [reference_value, endpoint],
            color=color,
            linestyle="--",
            linewidth=1.3,
        )
        axis.scatter(
            [float(record["global_phase"])],
            [endpoint],
            color=color,
            s=42,
            zorder=6,
        )
        prefix = "≈" if abs(value) < 5e-10 else ""
        axis.annotate(
            f"{label}: {prefix}{value:+.3f}°\n"
            f"phase={float(record['global_phase']):.3f}\n"
            f"alpha={_alpha_text(record)}",
            xy=(float(record["global_phase"]), endpoint),
            xytext=(10, 14 if value >= 0 else -52),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9},
        )
    width_axis.fill_between(
        phase,
        0.0,
        width,
        color=_COLORS["band"],
        alpha=0.55,
        label=f"{joint} corridor width",
    )
    width_axis.plot(phase, width, color=_COLORS[joint], linewidth=1.4)
    width_axis.set_xlim(0.0, 1.0)
    width_axis.set_xlabel("global phase")
    width_axis.set_ylabel("width (deg)")
    width_axis.legend(loc="best")
    if joint == "knee":
        positive_value = float(positive["value"])
        nominal_delta = float(positive["knee_delta"])
        axis.text(
            0.015,
            0.035,
            "The +{:.3f}° instantaneous knee deviation exceeds its {:+.2f}° "
            "amplitude term because the simultaneous phase warp changes the "
            "same-phase angle.".format(positive_value, nominal_delta),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88},
        )
    filename = f"{joint}_corridor_vs_phase_detailed.png"
    return _save_figure(figure, output, filename)


def _plot_pull_envelope(axis: plt.Axes, pull: pd.DataFrame) -> None:
    stride = 2
    sampled = pull.iloc[::stride]
    for x_column, z_column in (
        ("x_pull_min_m", "z_pull_min_m"),
        ("x_pull_min_m", "z_pull_max_m"),
        ("x_pull_max_m", "z_pull_min_m"),
        ("x_pull_max_m", "z_pull_max_m"),
    ):
        axis.scatter(
            sampled[x_column],
            sampled[z_column],
            color=_COLORS["band"],
            alpha=0.12,
            s=9,
            edgecolors="none",
        )
    axis.plot(
        pull["x_pull_min_m"],
        pull["z_pull_min_m"],
        color=_COLORS["lower"],
        linewidth=1.1,
        label="phase-wise lower pair",
    )
    axis.plot(
        pull["x_pull_max_m"],
        pull["z_pull_max_m"],
        color=_COLORS["upper"],
        linewidth=1.1,
        label="phase-wise upper pair",
    )


def _pull_reference_figure(
    region: Any,
    candidates: Mapping[str, pd.DataFrame],
    output: Path,
) -> Path:
    pull = region.pull_corridor
    figure, axis = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)
    _plot_pull_envelope(axis, pull)
    candidate_colors = {
        "hip negative": "#4C78A8",
        "knee negative": "#F58518",
        "positive phase": "#54A24B",
        "negative phase": "#B279A2",
        "combined perturbation": "#E45756",
    }
    for label, trajectory in candidates.items():
        if label == "neutral":
            continue
        axis.plot(
            trajectory["x_pull_m"],
            trajectory["z_pull_m"],
            color=candidate_colors[label],
            linewidth=1.15,
            alpha=0.85,
            label=label,
        )
    axis.plot(
        pull["x_pull_ref_m"],
        pull["z_pull_ref_m"],
        color=_COLORS["reference"],
        linewidth=3.0,
        label="active reference / neutral",
        zorder=5,
    )
    maximum = region.summary["pull_extrema"]["maximum_radial_pull_deviation_mm"]
    extreme = generate_personalized_trajectory(
        None,
        float(maximum["hip_delta"]),
        float(maximum["knee_delta"]),
        float(maximum["phase_delta"]),
    ).trajectory
    index = int(maximum["sample_index"])
    reference_point = np.array(
        [pull.loc[index, "x_pull_ref_m"], pull.loc[index, "z_pull_ref_m"]],
        dtype=float,
    )
    candidate_point = np.array(
        [extreme.loc[index, "x_pull_m"], extreme.loc[index, "z_pull_m"]],
        dtype=float,
    )
    axis.plot(
        [reference_point[0], candidate_point[0]],
        [reference_point[1], candidate_point[1]],
        color=_COLORS["pull"],
        linestyle="--",
        linewidth=1.5,
        zorder=7,
    )
    axis.scatter(
        [candidate_point[0]],
        [candidate_point[1]],
        marker="*",
        s=128,
        color=_COLORS["pull"],
        zorder=8,
        label="maximum radial deviation",
    )
    axis.annotate(
        f"{float(maximum['value']):.3f} mm\n"
        f"phase={float(maximum['global_phase']):.3f}\n"
        f"alpha={_alpha_text(maximum)}",
        xy=candidate_point,
        xytext=(-110, 18),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": _COLORS["pull"]},
    )
    axis.set_xlabel("pull point x (m)")
    axis.set_ylabel("pull point z (m)")
    axis.set_title(
        "Reference-centered pull-point corridor (offline FK audit)\n"
        "Same-phase envelope bounds are not a robot-safety polygon"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best", ncol=2)
    return _save_figure(figure, output, "pull_reference_corridor.png")


def _pull_deviation_figure(region: Any, output: Path) -> Path:
    pull = region.pull_corridor
    phase = pull["global_phase"].to_numpy(float)
    reference_x = pull["x_pull_ref_m"].to_numpy(float)
    reference_z = pull["z_pull_ref_m"].to_numpy(float)
    max_abs_x = 1000.0 * np.maximum(
        np.abs(pull["x_pull_min_m"].to_numpy(float) - reference_x),
        np.abs(pull["x_pull_max_m"].to_numpy(float) - reference_x),
    )
    max_abs_z = 1000.0 * np.maximum(
        np.abs(pull["z_pull_min_m"].to_numpy(float) - reference_z),
        np.abs(pull["z_pull_max_m"].to_numpy(float) - reference_z),
    )
    radial = pull["pull_radial_max_mm"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    axis.fill_between(phase, 0.0, radial, color="#C6B2DC", alpha=0.35)
    axis.plot(
        phase,
        radial,
        color=_COLORS["pull"],
        linewidth=2.6,
        label="maximum radial deviation",
    )
    axis.plot(phase, max_abs_x, color=_COLORS["lower"], label="maximum |delta x|")
    axis.plot(phase, max_abs_z, color=_COLORS["upper"], label="maximum |delta z|")
    index = int(np.argmax(radial))
    axis.scatter([phase[index]], [radial[index]], color=_COLORS["pull"], s=50, zorder=5)
    axis.annotate(
        f"{radial[index]:.3f} mm at phase {phase[index]:.3f}",
        xy=(phase[index], radial[index]),
        xytext=(-135, -42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": _COLORS["pull"]},
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("global phase")
    axis.set_ylabel("deviation from same-phase reference pull point (mm)")
    axis.set_title("Phase-dependent pull-point deviation envelope")
    axis.legend(loc="upper left")
    return _save_figure(figure, output, "pull_deviation_vs_phase.png")


def _classification(table: pd.DataFrame) -> np.ndarray:
    admissible = table["trajectory_admissible"].astype(bool).to_numpy()
    reasons = table["invalid_reason"].fillna("").astype(str)
    domain = reasons.str.contains("identification_domain_insufficient", regex=False).to_numpy()
    return np.where(admissible, 0, np.where(domain, 1, 2))


def _scatter_parameter_slice(
    axis: plt.Axes,
    parameter_map: pd.DataFrame,
    phase_value: float,
    *,
    overlay_joint_neighbors: bool,
) -> None:
    selected = parameter_map.loc[np.isclose(parameter_map["phase_delta"], phase_value)]
    categories = _classification(selected)
    for code, color, label in (
        (0, _COLORS["admissible"], "admissible"),
        (1, _COLORS["domain_rejected"], "identification-domain rejected"),
        (2, _COLORS["other_rejected"], "other rejection"),
    ):
        rows = selected.loc[categories == code]
        if rows.empty:
            continue
        axis.scatter(
            rows["hip_delta"],
            rows["knee_delta"],
            color=color,
            marker="s",
            s=22,
            edgecolors="none",
            alpha=0.92,
            label=label,
        )
    if np.isclose(phase_value, 0.0):
        axis.scatter(
            [0.0],
            [0.0],
            color="black",
            marker="*",
            s=135,
            zorder=7,
            label="neutral",
        )
        if overlay_joint_neighbors:
            for label, (hip, knee, phase) in INITIAL_NEIGHBORS.items():
                if not np.isclose(phase, 0.0):
                    continue
                axis.scatter(
                    [hip],
                    [knee],
                    facecolors="none",
                    edgecolors="black",
                    marker="o",
                    linewidths=1.3,
                    s=75,
                    zorder=7,
                )
                axis.annotate(label, (hip, knee), xytext=(4, 5), textcoords="offset points", fontsize=7)
    axis.set_xlim(-5.35, 2.35)
    axis.set_ylim(-5.35, 2.35)
    axis.set_xlabel("hip delta (deg)")
    axis.set_ylabel("knee delta (deg)")
    axis.set_title(f"phase delta = {phase_value:+.03f}")


def _neighbor_rows(parameter_map: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, (hip, knee, phase) in INITIAL_NEIGHBORS.items():
        selected = parameter_map.loc[
            np.isclose(parameter_map["hip_delta"], hip)
            & np.isclose(parameter_map["knee_delta"], knee)
            & np.isclose(parameter_map["phase_delta"], phase)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"initial neighbor is absent or duplicated: {label}")
        row = selected.iloc[0]
        rows.append(
            {
                "label": label,
                "alpha": [hip, knee, phase],
                "trajectory_admissible": bool(row["trajectory_admissible"]),
                "domain_coverage_percent": float(row["domain_coverage"]),
                "invalid_reason": "" if pd.isna(row["invalid_reason"]) else str(row["invalid_reason"]),
            }
        )
    return rows


def _identification_slices_figure(parameter_map: pd.DataFrame, output: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)
    for axis, phase in zip(axes.flat[:3], (-0.03, 0.0, 0.03)):
        _scatter_parameter_slice(
            axis,
            parameter_map,
            phase,
            overlay_joint_neighbors=np.isclose(phase, 0.0),
        )
    info = axes.flat[3]
    info.axis("off")
    lines = ["Initial trust-region neighbors", "", "direction      alpha              status / domain coverage"]
    for row in _neighbor_rows(parameter_map):
        alpha = row["alpha"]
        status = "supported" if row["trajectory_admissible"] else "domain rejected"
        lines.append(
            f"{row['label']:<10} ({alpha[0]:+g}, {alpha[1]:+g}, {alpha[2]:+.2f})  "
            f"{status:<15} {row['domain_coverage_percent']:.1f}%"
        )
    lines.extend(
        [
            "",
            "Phase +/- neighbors lie at +/-0.01 and therefore",
            "are listed here rather than moved onto the +/-0.03 slices.",
            "All current rejections are identification-domain insufficiency.",
        ]
    )
    info.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=_COLORS["admissible"], label="admissible"),
        Line2D([0], [0], marker="s", linestyle="", color=_COLORS["domain_rejected"], label="identification-domain rejected"),
        Line2D([0], [0], marker="s", linestyle="", color=_COLORS["other_rejected"], label="other rejection (none observed)"),
        Line2D([0], [0], marker="*", linestyle="", color="black", markersize=11, label="neutral"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black", label="joint-direction initial neighbor"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(
        "Identification-supported parameter region\n"
        "Offline prediction-support evidence; not ROM, workspace, or safety",
        fontsize=14,
    )
    return _save_figure(figure, output, "identification_region_parameter_slices.png")


def _directional_support(parameter_map: pd.DataFrame) -> list[dict[str, Any]]:
    definitions = (
        ("hip -", "hip_delta", -1.0, "deg"),
        ("hip +", "hip_delta", 1.0, "deg"),
        ("knee -", "knee_delta", -1.0, "deg"),
        ("knee +", "knee_delta", 1.0, "deg"),
        ("phase -", "phase_delta", -1.0, "phase"),
        ("phase +", "phase_delta", 1.0, "phase"),
    )
    result: list[dict[str, Any]] = []
    all_columns = ("hip_delta", "knee_delta", "phase_delta")
    for label, column, sign, unit in definitions:
        other = [name for name in all_columns if name != column]
        ray = parameter_map.loc[
            np.isclose(parameter_map[other[0]], 0.0)
            & np.isclose(parameter_map[other[1]], 0.0)
            & (sign * parameter_map[column] > 0.0)
        ].copy()
        ray["distance"] = np.abs(ray[column].to_numpy(float))
        ray = ray.sort_values("distance")
        supported = ray.loc[ray["trajectory_admissible"].astype(bool)]
        rejected = ray.loc[~ray["trajectory_admissible"].astype(bool)]
        nearest_rejected = None if rejected.empty else rejected.iloc[0]
        farthest_supported = None if supported.empty else supported.iloc[-1]
        result.append(
            {
                "direction": label,
                "parameter_column": column,
                "unit": unit,
                "supported_count": int(len(supported)),
                "sampled_count": int(len(ray)),
                "supported_fraction": float(len(supported) / len(ray)),
                "nearest_sample_distance": float(ray.iloc[0]["distance"]),
                "farthest_supported_distance": None if farthest_supported is None else float(farthest_supported["distance"]),
                "nearest_rejected_distance": None if nearest_rejected is None else float(nearest_rejected["distance"]),
                "nearest_rejected_reason": "" if nearest_rejected is None or pd.isna(nearest_rejected["invalid_reason"]) else str(nearest_rejected["invalid_reason"]),
            }
        )
    return result


def _directional_support_figure(
    directional: list[dict[str, Any]],
    output: Path,
) -> Path:
    labels = [str(item["direction"]) for item in directional]
    percentages = 100.0 * np.asarray([item["supported_fraction"] for item in directional])
    colors = [
        _COLORS["admissible"] if value == 100.0 else (_COLORS["domain_rejected"] if value == 0.0 else _COLORS["upper"])
        for value in percentages
    ]
    figure, axis = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    y = np.arange(len(labels))
    axis.barh(y, percentages, color=colors, alpha=0.88, height=0.62)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 150.0)
    axis.set_xticks(np.arange(0.0, 101.0, 20.0))
    axis.set_xlabel("supported samples along neutral-centered direction (%)")
    axis.set_title(
        "Directional identification support around the active reference\n"
        "Fractions use the frozen parameter-map lattice"
    )
    for index, item in enumerate(directional):
        farthest = item["farthest_supported_distance"]
        nearest_rejected = item["nearest_rejected_distance"]
        unit = "" if item["unit"] == "phase" else "°"
        if farthest is None:
            detail = f"no supported point; nearest rejection {nearest_rejected:g}{unit}"
        elif nearest_rejected is None:
            detail = f"supported through {farthest:g}{unit} sampled bound"
        else:
            detail = f"supported through {farthest:g}{unit}; next rejection {nearest_rejected:g}{unit}"
        if item["nearest_rejected_reason"]:
            detail += "; identification domain"
        axis.text(
            min(percentages[index] + 2.0, 104.0),
            index,
            f"{percentages[index]:.1f}% ({item['supported_count']}/{item['sampled_count']}) — {detail}",
            va="center",
            fontsize=8.5,
        )
    axis.axvline(100.0, color="0.35", linestyle=":", linewidth=1.0)
    axis.text(
        0.0,
        -0.12,
        "Support describes model-evidence coverage only; it does not authorize motion.",
        transform=axis.transAxes,
        fontsize=8.5,
    )
    return _save_figure(figure, output, "identification_directional_support_summary.png")


def _pose_indices(reference: pd.DataFrame) -> list[tuple[str, int]]:
    flexion = reference.index[reference["cycle_phase"].astype(str).eq("flexion")].to_numpy(int)
    extension = reference.index[reference["cycle_phase"].astype(str).eq("extension")].to_numpy(int)
    if len(flexion) < 3 or len(extension) < 2:
        raise RuntimeError("active reference lacks both flexion and extension samples")
    return [
        ("start", 0),
        ("early flexion", int(flexion[len(flexion) // 2])),
        ("peak flexion", int(flexion[-1])),
        ("early extension", int(extension[len(extension) // 2])),
        ("end / closure", len(reference) - 1),
    ]


def _draw_pose(axis: plt.Axes, row: pd.Series, color: str, label: str) -> None:
    axis.plot(
        [0.0, float(row["x_knee_m"]), float(row["x_pull_m"])],
        [0.0, float(row["z_knee_m"]), float(row["z_pull_m"])],
        "-o",
        color=color,
        linewidth=2.2,
        markersize=5,
        label=label,
    )


def _motion_snapshots_figure(reference: pd.DataFrame, output: Path) -> Path:
    poses = _pose_indices(reference)
    figure, axes = plt.subplots(1, len(poses), figsize=(16.0, 4.2), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(poses)))
    all_x = np.concatenate(([0.0], reference["x_knee_m"], reference["x_pull_m"]))
    all_z = np.concatenate(([0.0], reference["z_knee_m"], reference["z_pull_m"]))
    margin = 0.05
    for axis, (label, index), color in zip(axes, poses, colors):
        row = reference.iloc[index]
        axis.plot(reference["x_pull_m"], reference["z_pull_m"], color="0.82", linewidth=1.0)
        _draw_pose(axis, row, color, label)
        axis.scatter([0.0], [0.0], marker="s", color="black", s=30, zorder=5)
        axis.set_title(f"{label}\nt={float(row['time_s']):.1f} s")
        axis.set_xlim(float(all_x.min()) - margin, float(all_x.max()) + margin)
        axis.set_ylim(min(-0.02, float(all_z.min()) - margin), float(all_z.max()) + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    figure.suptitle(
        "Dynamic rehabilitation cycle snapshots — offline 2D leg kinematics\n"
        "L2 ends at the strap-equivalent pull point; this is not a robot rendering",
        fontsize=13,
    )
    return _save_figure(figure, output, "dynamic_rehabilitation_snapshots.png")


def _motion_animation(
    reference: pd.DataFrame,
    output: Path,
    *,
    frame_count: int,
    writer_fps: float,
) -> Path:
    if frame_count < 10:
        raise ValueError("animation frame_count must be at least 10")
    if not np.isfinite(writer_fps) or writer_fps <= 0.0:
        raise ValueError("animation writer_fps must be positive and finite")
    peak = _pose_indices(reference)[2][1]
    frame_indices = np.unique(
        np.concatenate(
            (
                np.linspace(0, len(reference) - 1, min(frame_count, len(reference))).astype(int),
                np.asarray([0, peak, len(reference) - 1], dtype=int),
            )
        )
    )
    figure, (axis, information) = plt.subplots(
        1,
        2,
        figsize=(10.5, 6.2),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
        constrained_layout=True,
    )
    all_x = np.concatenate(([0.0], reference["x_knee_m"], reference["x_pull_m"]))
    all_z = np.concatenate(([0.0], reference["z_knee_m"], reference["z_pull_m"]))
    margin = 0.06
    axis.plot(reference["x_pull_m"], reference["z_pull_m"], color="0.80", linewidth=1.2, label="active reference pull path")
    trail, = axis.plot([], [], color=_COLORS["pull"], linewidth=2.0, alpha=0.72, label="completed pull path")
    thigh, = axis.plot([], [], color=_COLORS["hip"], linewidth=4.0, label="thigh")
    shank, = axis.plot([], [], color=_COLORS["knee"], linewidth=4.0, label="knee to L2 pull point")
    joints, = axis.plot([], [], "o", color="black", markersize=7)
    pull_marker, = axis.plot([], [], marker="D", color=_COLORS["pull"], linestyle="", markersize=8, label="strap-equivalent pull point")
    axis.scatter([0.0], [0.0], marker="s", color="black", s=45, zorder=6, label="hip origin")
    axis.set_xlim(float(all_x.min()) - margin, float(all_x.max()) + margin)
    axis.set_ylim(min(-0.02, float(all_z.min()) - margin), float(all_z.max()) + margin)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Active asymmetric rehabilitation cycle")
    axis.legend(loc="best", fontsize=7.5)
    information.axis("off")
    status = information.text(0.02, 0.92, "", va="top", fontsize=11)
    information.text(
        0.02,
        0.12,
        "Offline kinematic visualization\n"
        "theta_shank = q_hip - q_knee\n"
        "No robot connection or command\n"
        "Not a real-robot safety region",
        va="bottom",
        fontsize=9,
        color="0.28",
    )
    figure.suptitle("Dynamic rehabilitation process (24 s cycle, time-compressed)", fontsize=14)

    def update(frame_index: int) -> tuple[Any, ...]:
        row = reference.iloc[int(frame_index)]
        knee = (float(row["x_knee_m"]), float(row["z_knee_m"]))
        pull = (float(row["x_pull_m"]), float(row["z_pull_m"]))
        thigh.set_data([0.0, knee[0]], [0.0, knee[1]])
        shank.set_data([knee[0], pull[0]], [knee[1], pull[1]])
        joints.set_data([0.0, knee[0]], [0.0, knee[1]])
        pull_marker.set_data([pull[0]], [pull[1]])
        trail.set_data(
            reference["x_pull_m"].iloc[: int(frame_index) + 1],
            reference["z_pull_m"].iloc[: int(frame_index) + 1],
        )
        status.set_text(
            f"time       {float(row['time_s']):5.2f} s\n"
            f"phase      {float(row['global_phase']):5.3f}\n"
            f"segment    {str(row['cycle_phase'])}\n\n"
            f"q_hip      {np.rad2deg(float(row['q_hip_rad'])):6.2f} deg\n"
            f"q_knee     {np.rad2deg(float(row['q_knee_rad'])):6.2f} deg\n\n"
            f"x_pull     {float(row['x_pull_m']):6.3f} m\n"
            f"z_pull     {float(row['z_pull_m']):6.3f} m"
        )
        return thigh, shank, joints, pull_marker, trail, status

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000.0 / float(writer_fps),
        blit=False,
        repeat=True,
    )
    path = output / GIF_FILENAME
    animation.save(path, writer=PillowWriter(fps=float(writer_fps)))
    plt.close(figure)
    return path


def _overview_figure(
    region: Any,
    reference: pd.DataFrame,
    parameter_map: pd.DataFrame,
    output: Path,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), constrained_layout=True)
    _joint_corridor_panel(axes[0, 0], region.joint_corridor, "hip", compact=True)
    _joint_corridor_panel(axes[0, 1], region.joint_corridor, "knee", compact=True)
    _plot_pull_envelope(axes[0, 2], region.pull_corridor)
    axes[0, 2].plot(
        region.pull_corridor["x_pull_ref_m"],
        region.pull_corridor["z_pull_ref_m"],
        color=_COLORS["reference"],
        linewidth=2.6,
        label="active reference",
    )
    axes[0, 2].set_xlabel("pull x (m)")
    axes[0, 2].set_ylabel("pull z (m)")
    axes[0, 2].set_title("Same-phase pull corridor")
    axes[0, 2].set_aspect("equal", adjustable="box")
    axes[0, 2].legend(loc="best", fontsize=7)
    _scatter_parameter_slice(axes[1, 0], parameter_map, 0.0, overlay_joint_neighbors=True)
    axes[1, 0].set_title("Identification support (phase delta = 0)")
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 5))
    for (label, index), color in zip(_pose_indices(reference), colors):
        _draw_pose(axes[1, 1], reference.iloc[index], color, label)
    axes[1, 1].plot(reference["x_pull_m"], reference["z_pull_m"], color="0.82", linewidth=1.0)
    axes[1, 1].scatter([0.0], [0.0], marker="s", color="black", s=32)
    axes[1, 1].set_xlabel("x (m)")
    axes[1, 1].set_ylabel("z (m)")
    axes[1, 1].set_title("Dynamic cycle snapshots")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].legend(loc="best", fontsize=7)
    key = axes[1, 2]
    key.axis("off")
    key.text(
        0.0,
        1.0,
        "Interpretation key\n\n"
        "GLOBAL ROM / WORKSPACE\n"
        "Overall physical-model geometric range.\n\n"
        "REFERENCE-CENTERED JOINT CORRIDORS\n"
        "Phase-specific hip and knee envelopes around the frozen reference.\n\n"
        "PULL / REFERENCE CORRIDOR\n"
        "Same-phase Cartesian envelope recomputed by FK.\n\n"
        "IDENTIFICATION REGION\n"
        "Where existing training data support model use.\n\n"
        "identification region != safety region\n"
        "identification region != ROM\n"
        "identification region != workspace",
        va="top",
        fontsize=10,
        linespacing=1.25,
    )
    figure.suptitle(
        "Reference-centered offline admissible region — explanatory overview",
        fontsize=15,
    )
    return _save_figure(figure, output, "reference_admissible_region_overview.png")


def _summary_markdown(region: Any, directional: list[dict[str, Any]]) -> str:
    joint = region.summary["joint_extrema"]
    pull = region.summary["pull_extrema"]
    direction_lines = []
    for item in directional:
        farthest = item["farthest_supported_distance"]
        nearest_rejected = item["nearest_rejected_distance"]
        detail = (
            "none on the sampled ray"
            if farthest is None
            else f"through {farthest:g} {item['unit']}"
        )
        if nearest_rejected is not None:
            detail += f"; nearest rejection at {nearest_rejected:g} {item['unit']}"
        direction_lines.append(
            f"| {item['direction']} | {100.0 * item['supported_fraction']:.1f}% "
            f"({item['supported_count']}/{item['sampled_count']}) | {detail} |"
        )
    return f"""# Admissible-region visualization summary

## Conceptual distinction

**GLOBAL ROM / WORKSPACE** describes the overall range admitted by the formal
physical model and geometry.  Under `ROM_PROTOCOL_V2`, hip flexion is
0--120 deg and knee flexion is 5--145 deg.

**REFERENCE-CENTERED CORRIDOR** is the local, phase-dependent exploration region
around the frozen active reference.  The **hip corridor** and **knee corridor**
bound the same-phase joint angles generated by the frozen continuous family.
The **pull/reference corridor** is independently recomputed through formal FK
and bounds the same-phase strap-equivalent pull point in the x-z plane.

**IDENTIFICATION REGION** marks parameter points whose full trajectories retain
sufficient coverage by the existing identification training domain.  It is a
model-evidence boundary: `identification region != safety region`,
`identification region != ROM`, and `identification region != workspace`.
It does not approve real-robot or human motion.

## Caption-ready figure descriptions

- **`reference_admissible_region_overview.png`** — Overview of phase-wise hip
  and knee corridors, the FK pull corridor, the phase-zero identification
  support slice, and five snapshots of the continuous rehabilitation cycle.
- **`hip_corridor_vs_phase_detailed.png`** — The active hip reference and its
  phase-wise lower/upper admissible envelope.  The largest negative deviation is
  {float(joint['maximum_negative_hip_deviation_deg']['value']):.3f} deg at phase
  {float(joint['maximum_negative_hip_deviation_deg']['global_phase']):.3f}; the
  largest positive deviation is numerically
  {float(joint['maximum_positive_hip_deviation_deg']['value']):.3e} deg.
- **`knee_corridor_vs_phase_detailed.png`** — The active knee reference and its
  phase-wise envelope.  The deviations span
  {float(joint['maximum_negative_knee_deviation_deg']['value']):.3f} to
  +{float(joint['maximum_positive_knee_deviation_deg']['value']):.3f} deg.  The
  positive extremum includes phase-warp displacement and therefore exceeds its
  nominal +{float(joint['maximum_positive_knee_deviation_deg']['knee_delta']):.2f}
  deg amplitude term.
- **`pull_reference_corridor.png`** — The active reference, five non-neutral
  representative candidates, and the same-phase FK envelope.  The maximum
  radial deviation is {float(pull['maximum_radial_pull_deviation_mm']['value']):.3f}
  mm at phase {float(pull['maximum_radial_pull_deviation_mm']['global_phase']):.3f}.
  Signed cycle extrema are delta-x
  [{float(pull['minimum_delta_x_pull_mm']['value']):.3f},
  +{float(pull['maximum_delta_x_pull_mm']['value']):.3f}] mm and delta-z
  [{float(pull['minimum_delta_z_pull_mm']['value']):.3f},
  +{float(pull['maximum_delta_z_pull_mm']['value']):.3f}] mm.
- **`identification_region_parameter_slices.png`** — Three formal parameter-map
  slices distinguish admissible points from identification-domain rejection.
  No other rejection class is present in the frozen map.
- **`identification_directional_support_summary.png`** — Directional support on
  neutral-centered grid rays explains why negative hip/knee and both phase
  directions are easier to explore, while positive hip is unsupported from the
  nearest sampled +0.25 deg point.
- **`dynamic_rehabilitation_process.gif`** and
  **`dynamic_rehabilitation_snapshots.png`** — Time-compressed and static views
  of the complete 24 s asymmetric flexion-extension cycle.  They display only
  offline 2D human-leg kinematics and the strap-equivalent pull point, not a
  robot execution or safety demonstration.

## Directional identification support

| Direction | Supported sampled ray | Extent / nearest rejection |
|---|---:|---|
{chr(10).join(direction_lines)}

## Evidence boundary

These figures are formal offline explanatory artifacts derived from the frozen
reference and admissible-region data.  They do not modify the generator,
sequential optimizer, mechanical objective, reliability rule, ROM, hardware, or
safety configuration.  `REAL_ROBOT_SAFETY_REGION` remains
`{REAL_ROBOT_SAFETY_REGION_STATUS}`.
"""


def generate_admissible_region_visualizations(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    animation_frame_count: int = 96,
    animation_writer_fps: float = 12.0,
) -> dict[str, Path]:
    """Generate all read-only explanatory artifacts from frozen inputs."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    region, reference, parameter_map, candidates = _load_inputs()
    directional = _directional_support(parameter_map)
    created: dict[str, Path] = {}
    with plt.rc_context(_PLOT_STYLE):
        created["reference_admissible_region_overview.png"] = _overview_figure(
            region, reference, parameter_map, output
        )
        created["hip_corridor_vs_phase_detailed.png"] = _detailed_joint_figure(
            region, output, "hip"
        )
        created["knee_corridor_vs_phase_detailed.png"] = _detailed_joint_figure(
            region, output, "knee"
        )
        created["pull_reference_corridor.png"] = _pull_reference_figure(
            region, candidates, output
        )
        created["pull_deviation_vs_phase.png"] = _pull_deviation_figure(region, output)
        created["identification_region_parameter_slices.png"] = (
            _identification_slices_figure(parameter_map, output)
        )
        created["identification_directional_support_summary.png"] = (
            _directional_support_figure(directional, output)
        )
        created["dynamic_rehabilitation_snapshots.png"] = _motion_snapshots_figure(
            reference, output
        )
        created[GIF_FILENAME] = _motion_animation(
            reference,
            output,
            frame_count=animation_frame_count,
            writer_fps=animation_writer_fps,
        )
    summary_path = output / SUMMARY_FILENAME
    summary_path.write_text(_summary_markdown(region, directional), encoding="utf-8")
    created[SUMMARY_FILENAME] = summary_path

    input_paths = {
        "active_reference": ACTIVE_REFERENCE_PATH,
        "formal_experiment_manifest": FORMAL_EXPERIMENT_MANIFEST_PATH,
        "admissible_region_manifest": DEFAULT_REGION_DIRECTORY / "admissible_region_manifest.json",
        "joint_corridor": DEFAULT_REGION_DIRECTORY / "joint_corridor_by_phase.csv",
        "pull_corridor": DEFAULT_REGION_DIRECTORY / "pull_corridor_by_phase.csv",
        "parameter_space_admissibility": DEFAULT_REGION_DIRECTORY / "parameter_space_admissibility.csv",
    }
    metadata = {
        "schema_version": 1,
        "visualization_version": VISUALIZATION_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "created_git_commit": _git_commit(),
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "source_region_version": REGION_VERSION,
        "source_region_classification": REGION_CLASSIFICATION,
        "generator_version": GENERATOR_VERSION,
        "source_inputs": {
            label: {
                "path": _project_relative(path),
                "sha256": _file_sha256(path),
            }
            for label, path in input_paths.items()
        },
        "trajectory_sample_count": int(len(reference)),
        "parameter_sample_count": int(len(parameter_map)),
        "representative_candidates": [
            {
                "label": label,
                "alpha": list(REPRESENTATIVE_POINTS[label]),
                "trajectory_admissible": True,
            }
            for label in REPRESENTATIVE_POINTS
        ],
        "initial_trust_region_neighbors": _neighbor_rows(parameter_map),
        "directional_identification_support": directional,
        "animation": {
            "source_cycle_duration_s": float(reference["time_s"].iloc[-1]),
            "requested_frame_count": int(animation_frame_count),
            "writer_fps": float(animation_writer_fps),
            "time_compressed": True,
            "physical_realtime_claimed": False,
        },
        "scientific_boundaries": {
            "identification_region_is_rom": False,
            "identification_region_is_workspace": False,
            "identification_region_is_safety_region": False,
            "real_robot_safety_region_status": REAL_ROBOT_SAFETY_REGION_STATUS,
            "formal_sequential_personalization_rerun": False,
            "reliability_threshold_resolved": False,
            "generator_mathematics_modified": False,
            "reference_modified": False,
            "rom_modified": False,
            "hardware_modified": False,
            "safety_modified": False,
            "robot_connection_performed": False,
            "robot_command_sent": False,
        },
        "artifact_sha256": {
            name: _file_sha256(path) for name, path in sorted(created.items())
        },
    }
    metadata_path = output / METADATA_FILENAME
    _write_json(metadata_path, metadata)
    created[METADATA_FILENAME] = metadata_path
    return created


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frozen admissible-region explanatory visualizations."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--animation-frame-count", type=int, default=96)
    parser.add_argument("--animation-writer-fps", type=float, default=12.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    generated = generate_admissible_region_visualizations(
        args.output_directory,
        animation_frame_count=args.animation_frame_count,
        animation_writer_fps=args.animation_writer_fps,
    )
    for name, path in sorted(generated.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "GIF_FILENAME",
    "METADATA_FILENAME",
    "PNG_FILENAMES",
    "SUMMARY_FILENAME",
    "VISUALIZATION_VERSION",
    "generate_admissible_region_visualizations",
]
