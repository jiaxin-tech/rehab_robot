"""运行虚拟受试者 ``software_test_trajectory`` 完整动态仿真。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    dynamic_model_version,
    dynamic_sampling_frequency_hz,
    dynamic_trajectory_data_dir,
    hip_range_deg,
    knee_range_deg,
    speed_profile_one_way_duration_s,
    test_trajectory_end_deg,
    test_trajectory_start_deg,
)
from .dynamic_subject import (
    DYNAMIC_SUBJECTS,
    DynamicVirtualSubject,
    get_dynamic_subject,
)
from .force_mapping import endpoint_force_from_joint_torque
from .full_dynamics import inverse_dynamics
from .jacobian import jacobian_diagnostics, leg_jacobian
from .kinematics import forward_kinematics, inverse_kinematics
from .trajectory_profiles import (
    TRAJECTORY_ID,
    generate_software_test_trajectory,
)


def _safe_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _validate_geometry(trajectory: pd.DataFrame) -> None:
    tolerance = 1e-12
    if (trajectory["z_knee_m"] < -tolerance).any():
        raise ValueError("trajectory moves the knee below the bed.")
    if (trajectory["z_pull_m"] < -tolerance).any():
        raise ValueError("trajectory moves the pull point below the bed.")
    if (trajectory["x_pull_m"] < -tolerance).any():
        raise ValueError("trajectory moves the pull point behind the hip.")

    if not trajectory["q_hip_deg"].between(*hip_range_deg).all():
        raise ValueError("trajectory exceeds the configured hip range.")
    if not trajectory["q_knee_deg"].between(*knee_range_deg).all():
        raise ValueError("trajectory exceeds the configured knee range.")

    _, _, inverse_reachable = inverse_kinematics(
        trajectory["x_pull_m"].to_numpy(),
        trajectory["z_pull_m"].to_numpy(),
        L1,
        L2,
    )
    if not np.all(inverse_reachable):
        raise ValueError("trajectory contains a geometrically unreachable point.")
    if trajectory["jacobian_near_singular"].any():
        raise ValueError("trajectory approaches a configured Jacobian singularity.")

    flexion = trajectory.loc[trajectory["phase"] == "flexion"]
    if not flexion["z_knee_m"].iloc[-1] > flexion["z_knee_m"].iloc[0]:
        raise ValueError("flexion does not lift the knee.")
    if not flexion["x_pull_m"].iloc[-1] < flexion["x_pull_m"].iloc[0]:
        raise ValueError("flexion does not retract the pull point toward the hip.")


def simulate_dynamic_trajectory(
    subject: DynamicVirtualSubject,
    speed_profile: str,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
) -> pd.DataFrame:
    """计算一条连续轨迹的几何、完整力矩分项和二维牵引力。"""

    profile = generate_software_test_trajectory(
        speed_profile,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    q_hip = profile["q_hip_rad"].to_numpy(dtype=float)
    q_knee = profile["q_knee_rad"].to_numpy(dtype=float)
    dq_hip = profile["dq_hip_rad_s"].to_numpy(dtype=float)
    dq_knee = profile["dq_knee_rad_s"].to_numpy(dtype=float)
    ddq_hip = profile["ddq_hip_rad_s2"].to_numpy(dtype=float)
    ddq_knee = profile["ddq_knee_rad_s2"].to_numpy(dtype=float)

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        L1,
        L2,
    )
    dynamics = inverse_dynamics(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
        ddq_hip,
        ddq_knee,
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
    jacobian = leg_jacobian(q_hip, q_knee, L1, L2)
    endpoint_force = np.stack(
        (force.fx_robot_on_leg_n, force.fz_robot_on_leg_n),
        axis=-1,
    )
    reconstructed = np.matmul(
        np.swapaxes(jacobian, -1, -2),
        endpoint_force[..., np.newaxis],
    )[..., 0]
    expected_torque = np.stack(
        (dynamics.tau_total_hip_nm, dynamics.tau_total_knee_nm),
        axis=-1,
    )
    reconstruction_error = np.linalg.norm(
        reconstructed - expected_torque,
        axis=-1,
    )
    reconstruction_error = np.where(
        force.force_mapping_valid,
        reconstruction_error,
        np.nan,
    )
    diagnostics = jacobian_diagnostics(q_hip, q_knee, L1, L2)

    trajectory = pd.DataFrame(
        {
            "subject_id": subject.subject_id,
            "trajectory_id": TRAJECTORY_ID,
            "speed_profile": speed_profile,
            "phase": profile["phase"].to_numpy(),
            "time_s": profile["time_s"].to_numpy(),
            "path_progress": profile["path_progress"].to_numpy(),
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "q_hip_deg": np.rad2deg(q_hip),
            "q_knee_deg": np.rad2deg(q_knee),
            "dq_hip_rad_s": dq_hip,
            "dq_knee_rad_s": dq_knee,
            "ddq_hip_rad_s2": ddq_hip,
            "ddq_knee_rad_s2": ddq_knee,
            "x_knee_m": x_knee,
            "z_knee_m": z_knee,
            "x_pull_m": x_pull,
            "z_pull_m": z_pull,
            "tau_inertia_hip_nm": dynamics.tau_inertia_hip_nm,
            "tau_inertia_knee_nm": dynamics.tau_inertia_knee_nm,
            "tau_coriolis_hip_nm": dynamics.tau_coriolis_hip_nm,
            "tau_coriolis_knee_nm": dynamics.tau_coriolis_knee_nm,
            "tau_gravity_hip_nm": dynamics.tau_gravity_hip_nm,
            "tau_gravity_knee_nm": dynamics.tau_gravity_knee_nm,
            "tau_damping_hip_nm": dynamics.tau_damping_hip_nm,
            "tau_damping_knee_nm": dynamics.tau_damping_knee_nm,
            "tau_stiffness_hip_nm": dynamics.tau_stiffness_hip_nm,
            "tau_stiffness_knee_nm": dynamics.tau_stiffness_knee_nm,
            "tau_total_hip_nm": dynamics.tau_total_hip_nm,
            "tau_total_knee_nm": dynamics.tau_total_knee_nm,
            "fx_robot_on_leg_n": force.fx_robot_on_leg_n,
            "fz_robot_on_leg_n": force.fz_robot_on_leg_n,
            "force_magnitude_n": force.force_magnitude_n,
            "jacobian_determinant": force.jacobian_determinant,
            "jacobian_condition_number": force.jacobian_condition_number,
            "jacobian_near_singular": diagnostics.near_singular,
            "force_mapping_valid": force.force_mapping_valid,
            "invalid_reason": force.invalid_reason,
            "torque_reconstruction_error_nm": reconstruction_error,
        }
    )
    _validate_geometry(trajectory)
    validate_dynamic_results(trajectory)
    return trajectory


def validate_dynamic_results(trajectory: pd.DataFrame) -> None:
    """验证全部非力字段和所有有效力映射样本的数值有限性。"""

    required_numeric = [
        column
        for column in trajectory.columns
        if column
        not in {
            "subject_id",
            "trajectory_id",
            "speed_profile",
            "phase",
            "invalid_reason",
        }
        and column
        not in {
            "fx_robot_on_leg_n",
            "fz_robot_on_leg_n",
            "force_magnitude_n",
            "torque_reconstruction_error_nm",
        }
    ]
    if not np.isfinite(trajectory[required_numeric].to_numpy(dtype=float)).all():
        raise ValueError("dynamic trajectory contains non-finite model data.")
    valid = trajectory["force_mapping_valid"].to_numpy(dtype=bool)
    valid_force_columns = [
        "fx_robot_on_leg_n",
        "fz_robot_on_leg_n",
        "force_magnitude_n",
        "torque_reconstruction_error_nm",
    ]
    if not np.isfinite(
        trajectory.loc[valid, valid_force_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("valid force mappings contain non-finite values.")


def build_metadata(
    trajectory: pd.DataFrame,
    subject: DynamicVirtualSubject,
    speed_profile: str,
    sampling_frequency_hz: float,
) -> dict[str, object]:
    """建立可追溯的动态仿真 metadata。"""

    valid = trajectory["force_mapping_valid"].to_numpy(dtype=bool)
    reasons = Counter(
        reason
        for reason in trajectory.loc[~valid, "invalid_reason"].astype(str)
        if reason
    )
    return {
        "subject": subject.as_metadata_dict(),
        "trajectory_id": TRAJECTORY_ID,
        "speed_profile": speed_profile,
        "L1_m": L1,
        "L2_pull_point_m": L2,
        "sampling_frequency_hz": sampling_frequency_hz,
        "one_way_duration_s": speed_profile_one_way_duration_s[speed_profile],
        "total_duration_s": float(trajectory["time_s"].iloc[-1]),
        "start_angles_deg": {
            "q_hip": test_trajectory_start_deg[0],
            "q_knee": test_trajectory_start_deg[1],
        },
        "maximum_flexion_angles_deg": {
            "q_hip": test_trajectory_end_deg[0],
            "q_knee": test_trajectory_end_deg[1],
        },
        "angle_definition": {
            "theta_thigh": "q_hip",
            "theta_shank": "q_hip - q_knee",
        },
        "coordinate_system": {
            "origin": "hip_joint_center",
            "x_axis": "along_bed_from_hip_toward_feet",
            "z_axis": "vertical_up",
            "bed": "z = 0",
        },
        "force_direction": {
            "saved_force": "robot_on_leg",
            "opposite_reaction": "leg_on_robot = -robot_on_leg",
        },
        "model_includes": [
            "two_link_inertia_coupling",
            "coriolis_and_centrifugal",
            "gravity",
            "linear_joint_damping",
            "linear_passive_joint_stiffness",
        ],
        "model_excludes": [
            "active_muscle_force",
            "nonlinear_end_range_resistance",
            "spasticity",
            "friction",
            "strap_elasticity_and_slip",
            "distributed_contact_force",
            "tactile_pressure",
            "robot_dynamics",
            "real_wrench_delay_and_noise",
            "parameter_identification",
            "PINN",
            "MPC",
            "trajectory_optimization",
        ],
        "force_mapping": {
            "valid_samples": int(valid.sum()),
            "invalid_samples": int((~valid).sum()),
            "valid_ratio": float(valid.mean()),
            "invalid_reason_counts": dict(reasons),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": dynamic_model_version,
        "git_commit": _safe_git_commit(),
        "disclaimer": (
            "Software-validation virtual trajectory; not a clinical reference, "
            "patient force prediction, robot command, or safety threshold."
        ),
    }


def save_dynamic_trajectory(
    trajectory: pd.DataFrame,
    metadata: dict[str, object],
    output_root: str | Path = dynamic_trajectory_data_dir,
) -> tuple[Path, Path, Path]:
    """保存 CSV、无 pickle 的 NPZ 和 JSON metadata。"""

    subject_id = str(trajectory["subject_id"].iloc[0])
    speed_profile = str(trajectory["speed_profile"].iloc[0])
    destination = Path(output_root) / subject_id / speed_profile
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "trajectory.csv"
    npz_path = destination / "trajectory.npz"
    metadata_path = destination / "metadata.json"
    trajectory.to_csv(csv_path, index=False)

    arrays: dict[str, np.ndarray] = {}
    for column in trajectory.columns:
        values = trajectory[column].to_numpy()
        arrays[column] = values.astype(str) if values.dtype == object else values
    np.savez_compressed(npz_path, **arrays)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, npz_path, metadata_path


def _run_one(
    subject: DynamicVirtualSubject,
    speed_profile: str,
    sampling_frequency_hz: float,
    output_root: Path,
    make_plots: bool,
) -> pd.DataFrame:
    trajectory = simulate_dynamic_trajectory(
        subject,
        speed_profile,
        sampling_frequency_hz,
    )
    metadata = build_metadata(
        trajectory,
        subject,
        speed_profile,
        sampling_frequency_hz,
    )
    paths = save_dynamic_trajectory(trajectory, metadata, output_root)
    valid = trajectory["force_mapping_valid"].to_numpy(dtype=bool)
    print(
        f"{subject.subject_id}/{speed_profile}: samples={len(trajectory)}, "
        f"valid_force={int(valid.sum())}/{len(trajectory)} "
        f"({valid.mean():.3%})"
    )
    for path in paths:
        print(path)
    if make_plots:
        from .visualize_dynamic_trajectory import (
            generate_dynamic_trajectory_plots,
        )

        for path in generate_dynamic_trajectory_plots(
            trajectory,
            paths[0].parent,
        ):
            print(f"figure: {path}")
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject_id",
        nargs="?",
        choices=tuple(DYNAMIC_SUBJECTS),
    )
    parser.add_argument(
        "speed_profile",
        nargs="?",
        choices=tuple(speed_profile_one_way_duration_s),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="生成 4 名虚拟受试者 × 3 种速度的全部轨迹。",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=dynamic_sampling_frequency_hz,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dynamic_trajectory_data_dir,
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="只保存 CSV、NPZ、metadata，不生成图片或 GIF。",
    )
    args = parser.parse_args()
    if args.all and (args.subject_id is not None or args.speed_profile is not None):
        parser.error("positional arguments and --all cannot be used together.")
    if not args.all and (
        args.subject_id is None or args.speed_profile is None
    ):
        parser.error("provide subject_id and speed_profile, or use --all.")

    subject_ids = tuple(DYNAMIC_SUBJECTS) if args.all else (args.subject_id,)
    speed_profiles = (
        tuple(speed_profile_one_way_duration_s)
        if args.all
        else (args.speed_profile,)
    )
    for subject_id in subject_ids:
        subject = get_dynamic_subject(subject_id)
        for speed_profile in speed_profiles:
            _run_one(
                subject,
                speed_profile,
                args.sampling_frequency,
                args.output_dir,
                make_plots=not args.no_plots,
            )


if __name__ == "__main__":
    main()
