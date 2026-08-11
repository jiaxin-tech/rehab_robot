"""生成无真实参数字段的第四阶段动力学辨识数据集。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    dynamic_sampling_frequency_hz,
    identification_dataset_split,
    identification_random_seed,
    identification_trajectory_endpoints_deg,
    speed_profile_one_way_duration_s,
)
from .dynamic_subject import DynamicVirtualSubject
from .force_mapping import endpoint_force_from_joint_torque
from .full_dynamics import inverse_dynamics
from .jacobian import jacobian_diagnostics
from .kinematics import forward_kinematics
from .noise_models import apply_noise_scenario
from .observation_model import joint_torque_from_endpoint_force
from .trajectory_profiles import (
    IDENTIFICATION_TRAJECTORY_ID,
    generate_identification_excitation_trajectory,
)

IDENTIFICATION_TRAJECTORY_FAMILIES = tuple(
    identification_trajectory_endpoints_deg
)
IDENTIFICATION_SPEED_PROFILES = tuple(speed_profile_one_way_duration_s)
FORBIDDEN_IDENTIFICATION_COLUMN_PREFIXES = (
    "true_",
    "ground_truth",
    "tau_total",
)


def _append_reason(
    dataframe: pd.DataFrame,
    mask: np.ndarray,
    reason: str,
) -> None:
    current = dataframe.loc[mask, "invalid_reason"].fillna("").astype(str)
    dataframe.loc[mask, "invalid_reason"] = np.where(
        current.eq(""),
        reason,
        current + ";" + reason,
    )


def _simulate_clean_observation(
    subject: DynamicVirtualSubject,
    trajectory_family: str,
    speed_profile: str,
    sampling_frequency_hz: float,
    L1_m: float,
    L2_m: float,
) -> pd.DataFrame:
    profile = generate_identification_excitation_trajectory(
        trajectory_family,
        speed_profile,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    q_hip = profile["q_hip_rad"].to_numpy(dtype=float)
    q_knee = profile["q_knee_rad"].to_numpy(dtype=float)
    dq_hip = profile["dq_hip_rad_s"].to_numpy(dtype=float)
    dq_knee = profile["dq_knee_rad_s"].to_numpy(dtype=float)
    ddq_hip = profile["ddq_hip_rad_s2"].to_numpy(dtype=float)
    ddq_knee = profile["ddq_knee_rad_s2"].to_numpy(dtype=float)

    dynamics = inverse_dynamics(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
        ddq_hip,
        ddq_knee,
        subject,
        L1_m,
    )
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        dynamics.tau_total_hip_nm,
        dynamics.tau_total_knee_nm,
        L1_m,
        L2_m,
    )
    reconstructed_hip, reconstructed_knee = joint_torque_from_endpoint_force(
        q_hip,
        q_knee,
        force.fx_robot_on_leg_n,
        force.fz_robot_on_leg_n,
        L1_m,
        L2_m,
    )
    reconstruction_error = np.hypot(
        reconstructed_hip - np.asarray(dynamics.tau_total_hip_nm),
        reconstructed_knee - np.asarray(dynamics.tau_total_knee_nm),
    )
    reconstruction_error = np.where(
        force.force_mapping_valid,
        reconstruction_error,
        np.nan,
    )
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        L1_m,
        L2_m,
    )

    return pd.DataFrame(
        {
            "subject_id": subject.subject_id,
            "trajectory_id": IDENTIFICATION_TRAJECTORY_ID,
            "trajectory_family": trajectory_family,
            "speed_profile": speed_profile,
            "phase": profile["phase"].to_numpy(),
            "time_s": profile["time_s"].to_numpy(dtype=float),
            "trajectory_sample_index": np.arange(len(profile), dtype=int),
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "dq_hip_rad_s": dq_hip,
            "dq_knee_rad_s": dq_knee,
            "ddq_hip_rad_s2": ddq_hip,
            "ddq_knee_rad_s2": ddq_knee,
            "fx_observed_n": force.fx_robot_on_leg_n,
            "fz_observed_n": force.fz_robot_on_leg_n,
            "tau_measured_hip_nm": reconstructed_hip,
            "tau_measured_knee_nm": reconstructed_knee,
            "sample_valid": np.asarray(force.force_mapping_valid, dtype=bool),
            "force_mapping_valid": np.asarray(
                force.force_mapping_valid,
                dtype=bool,
            ),
            "invalid_reason": np.asarray(force.invalid_reason, dtype=str),
            "dataset_split": identification_dataset_split[
                (trajectory_family, speed_profile)
            ],
            "x_knee_m": x_knee,
            "z_knee_m": z_knee,
            "x_pull_m": x_pull,
            "z_pull_m": z_pull,
            "jacobian_determinant": force.jacobian_determinant,
            "jacobian_condition_number": force.jacobian_condition_number,
            "force_magnitude_observed_n": force.force_magnitude_n,
            "torque_reconstruction_consistency_error_nm": reconstruction_error,
        }
    )


def _recompute_observation_fields(
    dataframe: pd.DataFrame,
    L1_m: float,
    L2_m: float,
) -> None:
    q_hip = dataframe["q_hip_rad"].to_numpy(dtype=float)
    q_knee = dataframe["q_knee_rad"].to_numpy(dtype=float)
    fx = dataframe["fx_observed_n"].to_numpy(dtype=float)
    fz = dataframe["fz_observed_n"].to_numpy(dtype=float)
    tau_hip, tau_knee = joint_torque_from_endpoint_force(
        q_hip,
        q_knee,
        fx,
        fz,
        L1_m,
        L2_m,
    )
    dataframe["tau_measured_hip_nm"] = tau_hip
    dataframe["tau_measured_knee_nm"] = tau_knee
    dataframe["force_magnitude_observed_n"] = np.hypot(fx, fz)

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        L1_m,
        L2_m,
    )
    dataframe["x_knee_m"] = x_knee
    dataframe["z_knee_m"] = z_knee
    dataframe["x_pull_m"] = x_pull
    dataframe["z_pull_m"] = z_pull
    diagnostics = jacobian_diagnostics(q_hip, q_knee, L1_m, L2_m)
    dataframe["jacobian_determinant"] = diagnostics.determinant
    dataframe["jacobian_condition_number"] = diagnostics.condition_number

    finite_observation = np.isfinite(
        dataframe[
            [
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
                "fx_observed_n",
                "fz_observed_n",
                "tau_measured_hip_nm",
                "tau_measured_knee_nm",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    newly_invalid = dataframe["sample_valid"].astype(bool).to_numpy() & (
        ~finite_observation | np.asarray(diagnostics.near_singular, dtype=bool)
    )
    dataframe.loc[newly_invalid, "sample_valid"] = False
    _append_reason(dataframe, newly_invalid, "non_finite_or_singular_observation")


def validate_identification_dataset(dataframe: pd.DataFrame) -> None:
    """验证 split、几何、时间、测量来源和 ground-truth 字段隔离。"""

    required = {
        "subject_id",
        "trajectory_id",
        "trajectory_family",
        "speed_profile",
        "phase",
        "time_s",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "fx_observed_n",
        "fz_observed_n",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
        "sample_valid",
        "invalid_reason",
        "dataset_split",
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"identification dataset is missing: {sorted(missing)}")
    forbidden = [
        column
        for column in dataframe.columns
        if column.startswith(FORBIDDEN_IDENTIFICATION_COLUMN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(
            f"identification dataset leaks ground truth: {sorted(forbidden)}"
        )
    if set(dataframe["trajectory_id"]) != {IDENTIFICATION_TRAJECTORY_ID}:
        raise ValueError("all trajectories must be identification excitation.")

    expected_keys = set(identification_dataset_split)
    actual_keys = set(
        dataframe[["trajectory_family", "speed_profile"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        raise ValueError("identification trajectory family/speed coverage is invalid.")
    for keys, group in dataframe.groupby(
        ["trajectory_family", "speed_profile"],
        sort=False,
    ):
        if not np.all(np.diff(group["time_s"].to_numpy(dtype=float)) > 0.0):
            raise ValueError(f"trajectory {keys} time must be strictly increasing.")
        expected_split = identification_dataset_split[(str(keys[0]), str(keys[1]))]
        if set(group["dataset_split"]) != {expected_split}:
            raise ValueError(f"trajectory {keys} is assigned to the wrong split.")

    if (dataframe["x_pull_m"] < -1e-9).any():
        raise ValueError("identification trajectory moves behind the hip.")
    if (dataframe["z_pull_m"] < -1e-9).any():
        raise ValueError("identification trajectory moves below the bed.")
    if (dataframe["z_knee_m"] < -1e-9).any():
        raise ValueError("identification trajectory moves the knee below the bed.")

    valid = dataframe["sample_valid"].astype(bool)
    finite_columns = [
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "fx_observed_n",
        "fz_observed_n",
        "tau_measured_hip_nm",
        "tau_measured_knee_nm",
    ]
    if not np.isfinite(
        dataframe.loc[valid, finite_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("valid identification samples must be finite.")


def build_identification_dataset(
    subject: DynamicVirtualSubject,
    noise_scenario: str = "clean",
    *,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    random_seed: int = identification_random_seed,
    L1_m: float = L1,
    L2_m: float = L2,
    trajectory_families: Iterable[str] = IDENTIFICATION_TRAJECTORY_FAMILIES,
    speed_profiles: Iterable[str] = IDENTIFICATION_SPEED_PROFILES,
) -> pd.DataFrame:
    """生成 3 family × 3 speed，并在观测侧加入指定误差模型。"""

    requested_families = tuple(trajectory_families)
    requested_speeds = tuple(speed_profiles)
    expected_families = set(IDENTIFICATION_TRAJECTORY_FAMILIES)
    expected_speeds = set(IDENTIFICATION_SPEED_PROFILES)
    if set(requested_families) != expected_families:
        raise ValueError("the complete identification dataset needs all families.")
    if set(requested_speeds) != expected_speeds:
        raise ValueError("the complete identification dataset needs all speeds.")

    clean = pd.concat(
        [
            _simulate_clean_observation(
                subject,
                family,
                speed,
                sampling_frequency_hz,
                L1_m,
                L2_m,
            )
            for family in requested_families
            for speed in requested_speeds
        ],
        ignore_index=True,
    )
    noisy = apply_noise_scenario(clean, noise_scenario, random_seed=random_seed)
    dataframe = noisy.dataframe
    _recompute_observation_fields(dataframe, L1_m, L2_m)
    validate_identification_dataset(dataframe)
    dataframe.attrs["noise_metadata"] = noisy.metadata
    dataframe.attrs["sampling_frequency_hz"] = sampling_frequency_hz
    dataframe.attrs["angle_definition"] = "theta_shank = q_hip - q_knee"
    return dataframe


def split_identification_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """按预先固定的轨迹级 split 返回互不重叠的副本。"""

    validate_identification_dataset(dataframe)
    return {
        split: dataframe.loc[dataframe["dataset_split"].eq(split)].copy()
        for split in ("train", "validation", "test")
    }
