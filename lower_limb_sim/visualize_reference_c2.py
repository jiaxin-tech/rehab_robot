"""Offline plots for the C2-continuous rehabilitation reference."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIRECTORY = Path(tempfile.gettempdir()) / "lower_limb_sim_matplotlib"
_MPL_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIRECTORY))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_FILENAMES = (
    "reference_c2_joint_comparison.png",
    "reference_c2_acceleration_comparison.png",
    "reference_c2_pull_path_comparison.png",
)


def _style() -> None:
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


def _save(figure: plt.Figure, output_directory: Path, filename: str) -> Path:
    path = output_directory / filename
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_joint_comparison(
    phase_path: pd.DataFrame,
    output_directory: str | Path,
) -> Path:
    """Compare the retained PCHIP samples and the accepted quintic spline."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    flexion = phase_path.loc[phase_path["cycle_phase"].eq("flexion")]
    phase = flexion["segment_phase"].to_numpy(float)
    _style()
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    for axis, joint, title in (
        (axes[0], "hip", "Hip flexion"),
        (axes[1], "knee", "Knee flexion"),
    ):
        original = np.rad2deg(flexion[f"q_{joint}_original_pchip_rad"])
        c2 = np.rad2deg(flexion[f"q_{joint}_rad"])
        axis.plot(phase, original, color="#79706E", linewidth=1.5, label="retained PCHIP")
        axis.plot(phase, c2, color="#4C78A8", linewidth=1.2, label="quintic C2+")
        axis.set_ylabel(f"{title} (deg)")
        axis.legend(loc="best")
    axes[-1].set_xlabel("Flexion path phase")
    figure.suptitle("Reference path: retained PCHIP vs reference_closed_c2")
    return _save(figure, output, FIGURE_FILENAMES[0])


def plot_acceleration_comparison(
    original_by_profile: dict[str, pd.DataFrame],
    c2_by_profile: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> Path:
    """Compare analytic joint accelerations for slow and nominal clocks."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    profiles = tuple(profile for profile in ("slow", "nominal") if profile in c2_by_profile)
    if not profiles or any(profile not in original_by_profile for profile in profiles):
        raise ValueError("both original and C2 acceleration data are required.")
    _style()
    figure, axes = plt.subplots(len(profiles), 2, figsize=(10.0, 3.4 * len(profiles)))
    axes = np.asarray(axes).reshape(len(profiles), 2)
    for row, profile in enumerate(profiles):
        original = original_by_profile[profile]
        c2 = c2_by_profile[profile]
        for column, joint in enumerate(("hip", "knee")):
            axis = axes[row, column]
            axis.plot(
                original["time_s"],
                original[f"ddq_{joint}_rad_s2"],
                color="#79706E",
                linewidth=1.1,
                alpha=0.8,
                label="PCHIP",
            )
            axis.plot(
                c2["time_s"],
                c2[f"ddq_{joint}_rad_s2"],
                color="#E45756",
                linewidth=1.2,
                label="quintic C2+",
            )
            axis.set_title(f"{profile}: {joint} acceleration")
            axis.set_xlabel("Retimed time (s)")
            axis.set_ylabel("Angular acceleration (rad/s²)")
            axis.legend(loc="best")
    figure.suptitle("Acceleration comparison (offline retimed trajectories)")
    return _save(figure, output, FIGURE_FILENAMES[1])


def plot_pull_path_comparison(
    phase_path: pd.DataFrame,
    output_directory: str | Path,
) -> Path:
    """Compare equivalent strap pull paths; the observed ankle is not used."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    flexion = phase_path.loc[phase_path["cycle_phase"].eq("flexion")]
    original_q_hip = flexion["q_hip_original_pchip_rad"].to_numpy(float)
    original_q_knee = flexion["q_knee_original_pchip_rad"].to_numpy(float)
    # Use the persisted model dimensions rather than observed ankle positions.
    l1 = float(flexion["L1_m"].iloc[0])
    l2 = float(flexion["L2_m"].iloc[0])
    from .kinematics import forward_kinematics

    _, _, original_x, original_z = forward_kinematics(
        original_q_hip, original_q_knee, l1, l2
    )
    _style()
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.plot(
        original_x,
        original_z,
        color="#79706E",
        linewidth=1.7,
        label="retained PCHIP pull path",
    )
    axis.plot(
        flexion["x_pull_m"],
        flexion["z_pull_m"],
        color="#54A24B",
        linewidth=1.3,
        label="reference_closed_c2 pull path",
    )
    axis.scatter(
        [flexion["x_pull_m"].iloc[0], flexion["x_pull_m"].iloc[-1]],
        [flexion["z_pull_m"].iloc[0], flexion["z_pull_m"].iloc[-1]],
        color=["#4C78A8", "#E45756"],
        s=28,
        zorder=3,
        label="start / peak flexion",
    )
    axis.set_xlabel("Equivalent pull-point x in human frame (m)")
    axis.set_ylabel("Equivalent pull-point z in human frame (m)")
    axis.set_title("Equivalent strap pull path (not observed ankle path)")
    axis.axis("equal")
    axis.legend(loc="best")
    return _save(figure, output, FIGURE_FILENAMES[2])


def generate_reference_c2_visualizations(
    phase_path: pd.DataFrame,
    original_by_profile: dict[str, pd.DataFrame],
    c2_by_profile: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Generate all three required Stage-C2 figures."""

    return {
        FIGURE_FILENAMES[0]: plot_joint_comparison(phase_path, output_directory),
        FIGURE_FILENAMES[1]: plot_acceleration_comparison(
            original_by_profile, c2_by_profile, output_directory
        ),
        FIGURE_FILENAMES[2]: plot_pull_path_comparison(phase_path, output_directory),
    }


__all__ = [
    "FIGURE_FILENAMES",
    "generate_reference_c2_visualizations",
    "plot_acceleration_comparison",
    "plot_joint_comparison",
    "plot_pull_path_comparison",
]
