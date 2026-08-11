"""比较同一虚拟受试者相同路径的 slow、nominal、fast 动态指标。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    dynamic_trajectory_data_dir,
    speed_profile_one_way_duration_s,
)
from .dynamic_subject import DYNAMIC_SUBJECTS

PROFILE_ORDER = ("slow", "nominal", "fast")


def _joint_norm(trajectory: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.hypot(
        trajectory[f"{prefix}_hip_nm"].to_numpy(dtype=float),
        trajectory[f"{prefix}_knee_nm"].to_numpy(dtype=float),
    )


def summarize_speed_profile(trajectory: pd.DataFrame) -> dict[str, str | float | int]:
    """汇总一条轨迹的速度、力矩和末端力指标。"""

    subject_id = str(trajectory["subject_id"].iloc[0])
    speed_profile = str(trajectory["speed_profile"].iloc[0])
    valid = trajectory["force_mapping_valid"].astype(bool)
    force = trajectory.loc[valid, "force_magnitude_n"].to_numpy(dtype=float)
    return {
        "subject_id": subject_id,
        "trajectory_id": str(trajectory["trajectory_id"].iloc[0]),
        "speed_profile": speed_profile,
        "duration_s": float(trajectory["time_s"].iloc[-1]),
        "one_way_duration_s": speed_profile_one_way_duration_s[speed_profile],
        "peak_dq_hip_rad_s": float(trajectory["dq_hip_rad_s"].abs().max()),
        "peak_dq_knee_rad_s": float(trajectory["dq_knee_rad_s"].abs().max()),
        "peak_ddq_hip_rad_s2": float(trajectory["ddq_hip_rad_s2"].abs().max()),
        "peak_ddq_knee_rad_s2": float(trajectory["ddq_knee_rad_s2"].abs().max()),
        "peak_tau_inertia_nm": float(
            np.max(_joint_norm(trajectory, "tau_inertia"))
        ),
        "peak_tau_damping_nm": float(
            np.max(_joint_norm(trajectory, "tau_damping"))
        ),
        "peak_tau_total_nm": float(np.max(_joint_norm(trajectory, "tau_total"))),
        "peak_force_n": float(np.max(force)) if force.size else np.nan,
        "rms_force_n": (
            float(np.sqrt(np.mean(force**2))) if force.size else np.nan
        ),
        "invalid_force_samples": int((~valid).sum()),
        "valid_force_ratio": float(valid.mean()),
    }


def validate_speed_profiles(trajectories: dict[str, pd.DataFrame]) -> None:
    """验证三条轨迹只有时间尺度不同，动态指标按速度单调变化。"""

    if tuple(trajectories) != PROFILE_ORDER:
        raise ValueError(f"profiles must be ordered as {PROFILE_ORDER}.")
    angle_ranges = []
    for profile in PROFILE_ORDER:
        trajectory = trajectories[profile]
        angle_ranges.append(
            (
                trajectory["q_hip_rad"].min(),
                trajectory["q_hip_rad"].max(),
                trajectory["q_knee_rad"].min(),
                trajectory["q_knee_rad"].max(),
            )
        )
        # 两关节沿同一条直线关节路径，排除速度变化混入路径变化。
        progress_hip = (
            trajectory["q_hip_rad"] - trajectory["q_hip_rad"].min()
        ) / (
            trajectory["q_hip_rad"].max() - trajectory["q_hip_rad"].min()
        )
        progress_knee = (
            trajectory["q_knee_rad"] - trajectory["q_knee_rad"].min()
        ) / (
            trajectory["q_knee_rad"].max() - trajectory["q_knee_rad"].min()
        )
        if not np.allclose(progress_hip, progress_knee, atol=1e-12):
            raise ValueError(f"{profile} changes the configured geometric path.")
    if not all(
        np.allclose(angle_ranges[0], angle_range, atol=1e-12)
        for angle_range in angle_ranges[1:]
    ):
        raise ValueError("speed profiles do not share the same angle range.")

    summaries = {
        profile: summarize_speed_profile(trajectories[profile])
        for profile in PROFILE_ORDER
    }
    monotonic_metrics = (
        "peak_dq_hip_rad_s",
        "peak_dq_knee_rad_s",
        "peak_ddq_hip_rad_s2",
        "peak_ddq_knee_rad_s2",
        "peak_tau_inertia_nm",
        "peak_tau_damping_nm",
    )
    for metric in monotonic_metrics:
        values = [float(summaries[profile][metric]) for profile in PROFILE_ORDER]
        if not values[0] < values[1] < values[2]:
            raise ValueError(f"{metric} does not increase from slow to fast.")

    # 在相同 path_progress 的起点、中点和最大屈曲点比较准静态分项。
    for target_progress in (0.0, 0.5, 1.0):
        references: dict[str, np.ndarray] = {}
        for profile in PROFILE_ORDER:
            flexion = trajectories[profile].loc[
                trajectories[profile]["phase"] == "flexion"
            ]
            index = (flexion["path_progress"] - target_progress).abs().idxmin()
            row = flexion.loc[index]
            references[profile] = row[
                [
                    "tau_gravity_hip_nm",
                    "tau_gravity_knee_nm",
                    "tau_stiffness_hip_nm",
                    "tau_stiffness_knee_nm",
                ]
            ].to_numpy(dtype=float)
        if not all(
            np.allclose(references["slow"], references[profile], atol=1e-10)
            for profile in ("nominal", "fast")
        ):
            raise ValueError(
                "quasi-static terms changed at the same geometric posture."
            )


def build_speed_profile_comparison(
    subject_id: str,
    input_root: str | Path = dynamic_trajectory_data_dir,
    trajectories: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """加载、验证并汇总一名受试者的三种速度。"""

    if trajectories is None:
        root = Path(input_root) / subject_id
        trajectories = {
            profile: pd.read_csv(root / profile / "trajectory.csv")
            for profile in PROFILE_ORDER
        }
    validate_speed_profiles(trajectories)
    return pd.DataFrame(
        [summarize_speed_profile(trajectories[profile]) for profile in PROFILE_ORDER]
    )


def save_speed_profile_comparison(
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """保存速度比较 CSV 和六指标柱状图。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "speed_profile_comparison.csv"
    png_path = destination / "speed_profile_comparison.png"
    comparison.to_csv(csv_path, index=False)

    specs = (
        ("peak_dq_knee_rad_s", "peak knee velocity (rad/s)"),
        ("peak_ddq_knee_rad_s2", "peak knee acceleration (rad/s²)"),
        ("peak_tau_inertia_nm", "peak inertia torque norm (N·m)"),
        ("peak_tau_damping_nm", "peak damping torque norm (N·m)"),
        ("peak_force_n", "peak force (N)"),
        ("rms_force_n", "RMS force (N)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = ("#4c78a8", "#f2cf5b", "#e45756")
    for axis, (column, ylabel) in zip(axes.ravel(), specs):
        axis.bar(comparison["speed_profile"], comparison[column], color=colors)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    subject_id = str(comparison["subject_id"].iloc[0])
    figure.suptitle(
        f"Speed profile comparison — {subject_id} — software_test_trajectory"
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject_id",
        nargs="?",
        choices=tuple(DYNAMIC_SUBJECTS),
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=dynamic_trajectory_data_dir,
    )
    args = parser.parse_args()
    if args.all and args.subject_id is not None:
        parser.error("subject_id and --all cannot be used together.")
    if not args.all and args.subject_id is None:
        parser.error("provide subject_id or use --all.")

    subject_ids = tuple(DYNAMIC_SUBJECTS) if args.all else (args.subject_id,)
    for subject_id in subject_ids:
        comparison = build_speed_profile_comparison(
            subject_id,
            args.input_dir,
        )
        for path in save_speed_profile_comparison(
            comparison,
            args.input_dir / subject_id,
        ):
            print(path)


if __name__ == "__main__":
    main()
