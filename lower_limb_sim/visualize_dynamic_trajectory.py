"""连续 ``software_test_trajectory`` 的静态图和腿部动画。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import dynamic_trajectory_data_dir
from .dynamic_subject import DYNAMIC_SUBJECTS
from .config import speed_profile_one_way_duration_s


def _identity(trajectory: pd.DataFrame) -> tuple[str, str, str]:
    subject_ids = trajectory["subject_id"].astype(str).unique()
    speed_profiles = trajectory["speed_profile"].astype(str).unique()
    trajectory_ids = trajectory["trajectory_id"].astype(str).unique()
    if len(subject_ids) != 1 or len(speed_profiles) != 1 or len(trajectory_ids) != 1:
        raise ValueError("trajectory must contain one subject, profile, and ID.")
    return subject_ids[0], speed_profiles[0], trajectory_ids[0]


def _time_series_plot(
    trajectory: pd.DataFrame,
    columns: tuple[str, str],
    labels: tuple[str, str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    subject_id, speed_profile, trajectory_id = _identity(trajectory)
    figure, axis = plt.subplots(figsize=(9, 5))
    for column, label in zip(columns, labels):
        axis.plot(trajectory["time_s"], trajectory[column], label=label)
    axis.set_xlabel("time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{title} — {subject_id}/{speed_profile} — {trajectory_id}"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _pull_point_path(trajectory: pd.DataFrame, output_path: Path) -> None:
    subject_id, speed_profile, trajectory_id = _identity(trajectory)
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        trajectory["x_pull_m"],
        trajectory["z_pull_m"],
        c=trajectory["time_s"],
        cmap="viridis",
        s=10,
    )
    figure.colorbar(scatter, ax=axis, label="time (s)")
    axis.scatter(
        trajectory["x_pull_m"].iloc[0],
        trajectory["z_pull_m"].iloc[0],
        marker="o",
        s=55,
        color="green",
        label="start/end",
    )
    flexion_end = trajectory.loc[trajectory["phase"] == "flexion"].iloc[-1]
    axis.scatter(
        flexion_end["x_pull_m"],
        flexion_end["z_pull_m"],
        marker="x",
        s=70,
        color="red",
        label="maximum flexion",
    )
    axis.axhline(0.0, color="black", linewidth=1.2, label="bed: z = 0")
    axis.set_xlabel("x_pull (m)")
    axis.set_ylabel("z_pull (m)")
    axis.set_title(
        f"Continuous pull-point path — {subject_id}/{speed_profile} — "
        f"{trajectory_id}"
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _joint_torque_components(
    trajectory: pd.DataFrame,
    output_path: Path,
) -> None:
    subject_id, speed_profile, trajectory_id = _identity(trajectory)
    components = (
        ("inertia", "tau_inertia"),
        ("coriolis/centrifugal", "tau_coriolis"),
        ("gravity", "tau_gravity"),
        ("damping", "tau_damping"),
        ("stiffness", "tau_stiffness"),
        ("total", "tau_total"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for axis, joint in zip(axes, ("hip", "knee")):
        for label, prefix in components:
            axis.plot(
                trajectory["time_s"],
                trajectory[f"{prefix}_{joint}_nm"],
                label=label,
                linewidth=1.4 if label != "total" else 2.2,
            )
        axis.set_ylabel(f"{joint} torque (N·m)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
    axes[0].set_title(
        f"Joint torque components — {subject_id}/{speed_profile} — "
        f"{trajectory_id}"
    )
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _endpoint_force_plot(
    trajectory: pd.DataFrame,
    output_path: Path,
) -> None:
    subject_id, speed_profile, trajectory_id = _identity(trajectory)
    valid = trajectory["force_mapping_valid"].astype(bool)
    figure, axis = plt.subplots(figsize=(9, 5))
    for column, label in (
        ("fx_robot_on_leg_n", "Fx robot on leg"),
        ("fz_robot_on_leg_n", "Fz robot on leg"),
        ("force_magnitude_n", "|F|"),
    ):
        axis.plot(
            trajectory.loc[valid, "time_s"],
            trajectory.loc[valid, column],
            label=label,
        )
    axis.set_xlabel("time (s)")
    axis.set_ylabel("force (N)")
    axis.set_title(
        f"Endpoint force — {subject_id}/{speed_profile} — {trajectory_id}"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _force_vector_along_path(
    trajectory: pd.DataFrame,
    output_path: Path,
    max_arrows: int = 60,
) -> None:
    subject_id, speed_profile, trajectory_id = _identity(trajectory)
    valid = trajectory.loc[trajectory["force_mapping_valid"].astype(bool)]
    step = max(1, int(np.ceil(len(valid) / max_arrows)))
    sampled = valid.iloc[::step]
    magnitude = sampled["force_magnitude_n"].to_numpy(dtype=float)
    safe_magnitude = np.maximum(magnitude, np.finfo(float).eps)
    arrow_length_m = 0.025
    u = arrow_length_m * sampled["fx_robot_on_leg_n"].to_numpy() / safe_magnitude
    v = arrow_length_m * sampled["fz_robot_on_leg_n"].to_numpy() / safe_magnitude

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        trajectory["x_pull_m"],
        trajectory["z_pull_m"],
        color="lightgray",
        linewidth=2.0,
        label="pull-point path",
    )
    quiver = axis.quiver(
        sampled["x_pull_m"],
        sampled["z_pull_m"],
        u,
        v,
        sampled["time_s"],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
    )
    figure.colorbar(quiver, ax=axis, label="time (s)")
    axis.axhline(0.0, color="black", linewidth=1.2, label="bed: z = 0")
    axis.set_xlabel("x_pull (m)")
    axis.set_ylabel("z_pull (m)")
    axis.set_title(
        f"Robot-on-leg force along path — {subject_id}/{speed_profile} — "
        f"{trajectory_id}"
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _leg_animation(
    trajectory: pd.DataFrame,
    output_path: Path,
    max_frames: int = 120,
) -> None:
    subject_id, speed_profile, trajectory_id = _identity(trajectory)
    frame_indices = np.unique(
        np.linspace(0, len(trajectory) - 1, min(max_frames, len(trajectory))).astype(
            int
        )
    )
    all_x = np.concatenate(
        (
            np.array([0.0]),
            trajectory["x_knee_m"].to_numpy(),
            trajectory["x_pull_m"].to_numpy(),
        )
    )
    all_z = np.concatenate(
        (
            np.array([0.0]),
            trajectory["z_knee_m"].to_numpy(),
            trajectory["z_pull_m"].to_numpy(),
        )
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    line, = axis.plot([], [], "-o", linewidth=3.0, markersize=7)
    status = axis.text(0.03, 0.95, "", transform=axis.transAxes, va="top")
    axis.axhline(0.0, color="black", linewidth=1.5, label="bed: z = 0")
    axis.scatter([0.0], [0.0], marker="s", s=60, color="black", label="hip")
    margin = 0.06
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(all_x.min()) - margin, float(all_x.max()) + margin)
    axis.set_ylim(min(-0.03, float(all_z.min()) - margin), float(all_z.max()) + margin)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title(
        f"Leg motion — {subject_id}/{speed_profile} — {trajectory_id}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")

    def update(frame_index: int) -> tuple[object, ...]:
        row = trajectory.iloc[frame_index]
        line.set_data(
            [0.0, row["x_knee_m"], row["x_pull_m"]],
            [0.0, row["z_knee_m"], row["z_pull_m"]],
        )
        status.set_text(
            f"time={row['time_s']:.2f} s\nphase={row['phase']}"
        )
        return line, status

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=50,
        blit=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=20))
    plt.close(figure)


def generate_dynamic_trajectory_plots(
    trajectory: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """生成7张静态图和1个 GIF 动画。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "angles": destination / "joint_angles_vs_time.png",
        "velocities": destination / "joint_velocities_vs_time.png",
        "accelerations": destination / "joint_accelerations_vs_time.png",
        "pull_path": destination / "pull_point_path.png",
        "animation": destination / "leg_animation.gif",
        "torques": destination / "joint_torque_components.png",
        "forces": destination / "endpoint_force_vs_time.png",
        "force_path": destination / "force_vector_along_path.png",
    }
    angle_data = trajectory.assign(
        q_hip_plot_deg=np.rad2deg(trajectory["q_hip_rad"]),
        q_knee_plot_deg=np.rad2deg(trajectory["q_knee_rad"]),
    )
    _time_series_plot(
        angle_data,
        ("q_hip_plot_deg", "q_knee_plot_deg"),
        ("q_hip", "q_knee"),
        "angle (deg)",
        "Joint angles",
        paths["angles"],
    )
    _time_series_plot(
        trajectory,
        ("dq_hip_rad_s", "dq_knee_rad_s"),
        ("dq_hip", "dq_knee"),
        "angular velocity (rad/s)",
        "Joint velocities",
        paths["velocities"],
    )
    _time_series_plot(
        trajectory,
        ("ddq_hip_rad_s2", "ddq_knee_rad_s2"),
        ("ddq_hip", "ddq_knee"),
        "angular acceleration (rad/s²)",
        "Joint accelerations",
        paths["accelerations"],
    )
    _pull_point_path(trajectory, paths["pull_path"])
    _leg_animation(trajectory, paths["animation"])
    _joint_torque_components(trajectory, paths["torques"])
    _endpoint_force_plot(trajectory, paths["forces"])
    _force_vector_along_path(trajectory, paths["force_path"])
    return tuple(paths.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", choices=tuple(DYNAMIC_SUBJECTS))
    parser.add_argument(
        "speed_profile",
        choices=tuple(speed_profile_one_way_duration_s),
    )
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    default_dir = (
        dynamic_trajectory_data_dir / args.subject_id / args.speed_profile
    )
    trajectory_path = args.trajectory or default_dir / "trajectory.csv"
    output_dir = args.output_dir or default_dir
    trajectory = pd.read_csv(trajectory_path)
    for path in generate_dynamic_trajectory_plots(trajectory, output_dir):
        print(path)


if __name__ == "__main__":
    main()
