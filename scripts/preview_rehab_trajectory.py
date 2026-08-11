"""Pure-offline preview for the measured-asymmetric periodic trajectory.

This module has no hardware import and cannot connect, power, stop, or move a
robot.  It converts a saved anchor plus a rehabilitation-frame draft into CSV,
JSON audit evidence, and four visual checks for operator review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from control.start_anchor import StartAnchor, load_start_anchor
from control.start_anchored_relative_trajectory import (
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from utils.provenance import current_git_commit


DEFAULT_FRAME_CONFIG = Path(__file__).resolve().parents[1] / "config" / "rehab_frame_config.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _plot_preview(trajectory, output_dir: Path) -> list[Path]:
    time_s = trajectory["time_s"].to_numpy(float)
    xyz = trajectory[["tcp_x_base", "tcp_y_base", "tcp_z_base"]].to_numpy(float)
    velocity = trajectory[["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]].to_numpy(float)
    acceleration = trajectory[["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]].to_numpy(float)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    paths: list[Path] = []

    figure = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1.8)
    axis.scatter(*xyz[0], color="green", s=45, label="start / closed end")
    axis.set_xlabel("Base X (m)")
    axis.set_ylabel("Base Y (m)")
    axis.set_zlabel("Base Z (m)")
    axis.set_title("Start-anchored TCP path (offline preview)")
    axis.legend()
    path_3d = output_dir / "trajectory_3d.png"
    figure.savefig(path_3d, dpi=160)
    plt.close(figure)
    paths.append(path_3d)

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)
    for index, label in enumerate(("X", "Y", "Z")):
        axes[index].plot(time_s, xyz[:, index])
        axes[index].set_ylabel(f"Base {label} (m)")
        axes[index].grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("TCP position versus time")
    xyz_path = output_dir / "xyz_time.png"
    figure.savefig(xyz_path, dpi=160)
    plt.close(figure)
    paths.append(xyz_path)

    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    axis.plot(time_s, speed)
    axis.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="TCP speed (offline derivative)")
    axis.grid(True, alpha=0.25)
    speed_path = output_dir / "speed_time.png"
    figure.savefig(speed_path, dpi=160)
    plt.close(figure)
    paths.append(speed_path)

    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    axis.plot(time_s, acceleration_norm)
    axis.set(
        xlabel="Time (s)",
        ylabel="Acceleration (m/s²)",
        title="TCP acceleration (offline derivative)",
    )
    axis.grid(True, alpha=0.25)
    acceleration_path = output_dir / "acceleration_time.png"
    figure.savefig(acceleration_path, dpi=160)
    plt.close(figure)
    paths.append(acceleration_path)
    return paths


def preview_trajectory(
    *,
    anchor: StartAnchor | str | Path,
    frame_config: str | Path,
    reference: str | Path = DEFAULT_REFERENCE_PATH,
    output_dir: str | Path,
    reference_start_tolerance_rad: float = 1e-9,
) -> dict[str, Any]:
    """Generate review artifacts without importing or touching hardware."""
    resolved_anchor = load_start_anchor(anchor) if isinstance(anchor, (str, Path)) else anchor
    if not isinstance(resolved_anchor, StartAnchor):
        raise TypeError("anchor must be a StartAnchor or JSON path")
    frame = load_rehab_frame_config(frame_config)
    trajectory, audit, metadata = build_start_anchored_relative_trajectory(
        reference,
        current_tcp_start_pose=resolved_anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    if audit.trajectory_id != resolved_anchor.trajectory_id:
        raise ValueError(
            "anchor trajectory_id does not match preview reference: "
            f"{resolved_anchor.trajectory_id!r} != {audit.trajectory_id!r}"
        )
    q_hip_start = float(trajectory.iloc[0]["q_hip_ref"])
    q_knee_start = float(trajectory.iloc[0]["q_knee_ref"])
    if not np.isclose(
        resolved_anchor.reference_start_q_hip,
        q_hip_start,
        atol=reference_start_tolerance_rad,
        rtol=0.0,
    ) or not np.isclose(
        resolved_anchor.reference_start_q_knee,
        q_knee_start,
        atol=reference_start_tolerance_rad,
        rtol=0.0,
    ):
        raise ValueError("anchor reference start joint values do not match preview reference")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "trajectory_preview.csv"
    trajectory.to_csv(csv_path, index=False)
    plot_paths = _plot_preview(trajectory, destination)
    preview_metadata = {
        **metadata,
        "git_commit": current_git_commit(),
        "preview_only": True,
        "hardware_accessed": False,
        "anchor": resolved_anchor.to_dict(),
        "frame_reviewed": frame.reviewed,
        "anchor_reviewed": resolved_anchor.reviewed,
        "trajectory_csv": csv_path.name,
        "plots": [path.name for path in plot_paths],
    }
    metadata_path = destination / "preview_metadata.json"
    _write_json(metadata_path, preview_metadata)
    return {
        "output_dir": str(destination),
        "trajectory_csv": str(csv_path),
        "metadata_json": str(metadata_path),
        "plots": [str(path) for path in plot_paths],
        "trajectory_valid": audit.trajectory_valid,
        "trajectory_id": audit.trajectory_id,
        "sample_count": audit.sample_count,
        "frame_reviewed": frame.reviewed,
        "anchor_reviewed": resolved_anchor.reviewed,
        "execution_approved": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline start-anchored trajectory preview")
    parser.add_argument("--anchor", required=True, help="reviewable StartAnchor JSON")
    parser.add_argument("--frame-config", default=str(DEFAULT_FRAME_CONFIG))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    summary = preview_trajectory(
        anchor=args.anchor,
        frame_config=args.frame_config,
        reference=args.reference,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["trajectory_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
