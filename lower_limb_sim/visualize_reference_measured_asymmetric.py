"""Required audit figures for the measured asymmetric periodic reference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import L1, L2
from .kinematics import forward_kinematics
from .reference_measured_asymmetric import MeasuredAsymmetricPeriodicModel


FIGURE_FILENAMES = (
    "all_detected_cycles_closure.png",
    "selected_natural_cycle.png",
    "measured_flexion_vs_extension.png",
    "raw_vs_periodic_closed.png",
    "asymmetry_preservation.png",
    "new_reference_pull_path.png",
)


def _save(figure, destination: Path, filename: str) -> Path:
    path = destination / filename
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _branch_comparison(
    model: MeasuredAsymmetricPeriodicModel,
    *,
    sample_count: int = 1001,
) -> dict[str, np.ndarray]:
    raw = model.measured_raw
    phase = raw["global_phase"].to_numpy(dtype=float)
    raw_hip = PchipInterpolator(
        phase, raw["q_hip_measured_rad"].to_numpy(dtype=float)
    )
    raw_knee = PchipInterpolator(
        phase, raw["q_knee_measured_rad"].to_numpy(dtype=float)
    )
    local = np.linspace(0.0, 1.0, sample_count)
    flex_phase = model.peak_global_phase * local
    reverse_extension_phase = 1.0 - (1.0 - model.peak_global_phase) * local
    result: dict[str, np.ndarray] = {
        "local_phase": local,
        "raw_flex_hip": raw_hip(flex_phase),
        "raw_ext_hip": raw_hip(reverse_extension_phase),
        "raw_flex_knee": raw_knee(flex_phase),
        "raw_ext_knee": raw_knee(reverse_extension_phase),
        "closed_flex_hip": model.hip_spline(flex_phase),
        "closed_ext_hip": model.hip_spline(reverse_extension_phase),
        "closed_flex_knee": model.knee_spline(flex_phase),
        "closed_ext_knee": model.knee_spline(reverse_extension_phase),
    }
    for prefix in ("raw", "closed"):
        _, _, flex_x, flex_z = forward_kinematics(
            result[f"{prefix}_flex_hip"],
            result[f"{prefix}_flex_knee"],
            L1,
            L2,
        )
        _, _, ext_x, ext_z = forward_kinematics(
            result[f"{prefix}_ext_hip"],
            result[f"{prefix}_ext_knee"],
            L1,
            L2,
        )
        result[f"{prefix}_flex_x"] = flex_x
        result[f"{prefix}_flex_z"] = flex_z
        result[f"{prefix}_ext_x"] = ext_x
        result[f"{prefix}_ext_z"] = ext_z
    return result


def generate_measured_asymmetric_reference_visualizations(
    full_angles: pd.DataFrame,
    closure_audit: pd.DataFrame,
    model: MeasuredAsymmetricPeriodicModel,
    trajectories: Mapping[str, pd.DataFrame],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Generate the six requested figures without touching robot code."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Every detected candidate and its raw closure components.
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(
        full_angles["Frame"],
        np.rad2deg(full_angles["q_hip_rad"]),
        label="q_hip measured",
    )
    axes[0].plot(
        full_angles["Frame"],
        np.rad2deg(full_angles["q_knee_rad"]),
        label="q_knee measured",
    )
    for row in closure_audit.itertuples(index=False):
        color = "tab:green" if bool(getattr(row, "selected", False)) else "0.55"
        axes[0].axvspan(row.start_frame, row.end_frame, color=color, alpha=0.12)
        axes[0].axvline(row.peak_frame, color=color, linewidth=0.8, alpha=0.8)
        axes[0].text(
            row.peak_frame,
            axes[0].get_ylim()[1],
            str(row.cycle_candidate_id),
            ha="center",
            va="top",
            fontsize=8,
        )
    axes[0].set_ylabel("angle (deg)")
    axes[0].set_title("Detected full-joint natural cycle candidates")
    axes[0].legend(loc="best")
    labels = closure_audit["cycle_candidate_id"].astype(str).tolist()
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, closure_audit["closure_score"], 0.36, label="closure score")
    axes[1].bar(
        x + 0.18,
        closure_audit["pull_closure_error_mm"],
        0.36,
        label="pull closure (mm)",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("cycle candidate")
    axes[1].set_ylabel("audit value")
    axes[1].legend(loc="best")
    paths[FIGURE_FILENAMES[0]] = _save(figure, destination, FIGURE_FILENAMES[0])
    plt.close(figure)

    raw = model.measured_raw
    # 2. Selected natural cycle in source-frame coordinates.
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for phase_name, color in (("flexion", "tab:blue"), ("extension", "tab:orange")):
        branch = raw.loc[raw["cycle_phase"].eq(phase_name)]
        axes[0].plot(
            branch["Frame"],
            np.rad2deg(branch["q_hip_measured_rad"]),
            color=color,
            label=f"measured {phase_name}",
        )
        axes[1].plot(
            branch["Frame"],
            np.rad2deg(branch["q_knee_measured_rad"]),
            color=color,
            label=f"measured {phase_name}",
        )
    for axis, label in zip(axes, ("hip (deg)", "knee (deg)")):
        axis.axvline(model.start_frame, color="k", linestyle=":")
        axis.axvline(model.peak_frame, color="k", linestyle="--")
        axis.axvline(model.end_frame, color="k", linestyle=":")
        axis.set_ylabel(label)
        axis.legend(loc="best")
    axes[0].set_title("Selected measured natural flexion-extension cycle")
    axes[1].set_xlabel("source Frame")
    paths[FIGURE_FILENAMES[1]] = _save(figure, destination, FIGURE_FILENAMES[1])
    plt.close(figure)

    branches = _branch_comparison(model)
    local = branches["local_phase"]
    # 3. Direct measured flexion vs time-reversed measured extension comparison.
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, joint in zip(axes[:2], ("hip", "knee")):
        axis.plot(
            local,
            np.rad2deg(branches[f"raw_flex_{joint}"]),
            label="measured flexion",
        )
        axis.plot(
            local,
            np.rad2deg(branches[f"raw_ext_{joint}"]),
            label="measured extension, reversed only for comparison",
        )
        axis.set_xlabel("normalized branch phase")
        axis.set_ylabel(f"{joint} angle (deg)")
        axis.legend(loc="best", fontsize=8)
    axes[2].plot(
        branches["raw_flex_x"],
        branches["raw_flex_z"],
        label="measured flexion",
    )
    axes[2].plot(
        branches["raw_ext_x"],
        branches["raw_ext_z"],
        label="measured extension",
    )
    axes[2].set_xlabel("x_pull (m)")
    axes[2].set_ylabel("z_pull (m)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(loc="best", fontsize=8)
    figure.suptitle("Measured flexion and measured extension are different paths")
    paths[FIGURE_FILENAMES[2]] = _save(figure, destination, FIGURE_FILENAMES[2])
    plt.close(figure)

    # 4. Raw measured samples against the periodic-closed curve.
    phase = model.phase_path["global_phase"].to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, row_index, label in (
        ("hip", 0, "hip"),
        ("knee", 1, "knee"),
    ):
        measured_column = f"q_{column}_measured_interpolated_rad"
        reference_column = f"q_{column}_reference_rad"
        axes[row_index, 0].plot(
            phase,
            np.rad2deg(model.phase_path[measured_column]),
            label="measured interpolation",
        )
        axes[row_index, 0].plot(
            phase,
            np.rad2deg(model.phase_path[reference_column]),
            label="periodic closed",
        )
        axes[row_index, 0].set_ylabel(f"{label} (deg)")
        axes[row_index, 0].legend(loc="best")
        axes[row_index, 1].plot(
            phase,
            np.rad2deg(
                model.phase_path[reference_column]
                - model.phase_path[measured_column]
            ),
        )
        axes[row_index, 1].set_ylabel(f"{label} deviation (deg)")
    axes[1, 0].set_xlabel("global phase")
    axes[1, 1].set_xlabel("global phase")
    figure.suptitle("Raw measured path vs small periodic closure correction")
    paths[FIGURE_FILENAMES[3]] = _save(figure, destination, FIGURE_FILENAMES[3])
    plt.close(figure)

    # 5. Raw and closed flexion-extension differences on a common phase.
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(
        local,
        np.rad2deg(branches["raw_flex_hip"] - branches["raw_ext_hip"]),
        label="raw asymmetry",
    )
    axes[0].plot(
        local,
        np.rad2deg(
            branches["closed_flex_hip"] - branches["closed_ext_hip"]
        ),
        label="closed asymmetry",
    )
    axes[0].set_ylabel("hip branch difference (deg)")
    axes[1].plot(
        local,
        np.rad2deg(branches["raw_flex_knee"] - branches["raw_ext_knee"]),
        label="raw asymmetry",
    )
    axes[1].plot(
        local,
        np.rad2deg(
            branches["closed_flex_knee"] - branches["closed_ext_knee"]
        ),
        label="closed asymmetry",
    )
    axes[1].set_ylabel("knee branch difference (deg)")
    raw_pull_difference = 1000.0 * np.hypot(
        branches["raw_flex_x"] - branches["raw_ext_x"],
        branches["raw_flex_z"] - branches["raw_ext_z"],
    )
    closed_pull_difference = 1000.0 * np.hypot(
        branches["closed_flex_x"] - branches["closed_ext_x"],
        branches["closed_flex_z"] - branches["closed_ext_z"],
    )
    axes[2].plot(local, raw_pull_difference, label="raw asymmetry")
    axes[2].plot(local, closed_pull_difference, label="closed asymmetry")
    axes[2].set_ylabel("pull branch difference (mm)")
    for axis in axes:
        axis.set_xlabel("normalized branch phase")
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Periodic closure preserves measured branch asymmetry")
    paths[FIGURE_FILENAMES[4]] = _save(figure, destination, FIGURE_FILENAMES[4])
    plt.close(figure)

    # 6. New closed pull path, including slow/nominal identity.
    figure, axis = plt.subplots(figsize=(8, 7))
    for phase_name, color in (("flexion", "tab:blue"), ("extension", "tab:orange")):
        branch = raw.loc[raw["cycle_phase"].eq(phase_name)]
        axis.plot(
            branch["x_pull_m"],
            branch["z_pull_m"],
            color=color,
            linestyle=":",
            label=f"raw measured {phase_name}",
        )
    closed = trajectories["slow"]
    for phase_name, color in (("flexion", "tab:blue"), ("extension", "tab:orange")):
        branch = closed.loc[closed["cycle_phase"].eq(phase_name)]
        axis.plot(
            branch["x_pull_m"],
            branch["z_pull_m"],
            color=color,
            linewidth=2.0,
            label=f"periodic closed {phase_name}",
        )
    axis.scatter(
        [closed["x_pull_m"].iloc[0]],
        [closed["z_pull_m"].iloc[0]],
        color="k",
        marker="o",
        label="closed start/end",
    )
    axis.set_xlabel("x_pull (m)")
    axis.set_ylabel("z_pull (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best", fontsize=8)
    axis.set_title("New active measured-asymmetric pull path")
    paths[FIGURE_FILENAMES[5]] = _save(figure, destination, FIGURE_FILENAMES[5])
    plt.close(figure)

    return paths


__all__ = [
    "FIGURE_FILENAMES",
    "generate_measured_asymmetric_reference_visualizations",
]
