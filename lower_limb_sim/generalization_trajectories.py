"""阶段 4.5C 的未见软件验证轨迹。

本模块只生成关节运动学输入，不生成力矩或力，也不读取虚拟受试者参数。
所有几何检查复用 :func:`lower_limb_sim.kinematics.forward_kinematics`，因此
小腿方向始终是 ``q_hip - q_knee``。这些轨迹不是临床参考轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    dynamic_sampling_frequency_hz,
    hip_range_deg,
    identification_trajectory_endpoints_deg,
    knee_range_deg,
)
from .kinematics import forward_kinematics
from .trajectory_profiles import minimum_jerk_profile


SOFTWARE_VALIDATION_TRAJECTORY = "software_validation_trajectory"
INTERPOLATION_TEST = "interpolation_test"
BOUNDARY_TEST = "boundary_test"
OUTSIDE_DOMAIN_TEST = "outside_domain_test"
GENERALIZATION_SPLITS = (
    INTERPOLATION_TEST,
    BOUNDARY_TEST,
    OUTSIDE_DOMAIN_TEST,
)

_ANGLE_TOLERANCE_RAD = 1e-12
_WORKSPACE_TOLERANCE_M = 1e-12


@dataclass(frozen=True)
class GeneralizationTrajectorySpec:
    """一条未见轨迹的解析定义。角度以 degree 配置，输出统一为 SI。"""

    name: str
    dataset_split: str
    start_angles_deg: tuple[float, float]
    end_angles_deg: tuple[float, float]
    flexion_duration_s: float
    extension_duration_s: float
    speed_profile: str
    domain_relation: str
    description: str
    knee_phase_lead_strength: float = 0.0


GENERALIZATION_TRAJECTORIES: dict[str, GeneralizationTrajectorySpec] = {
    "phase_shift_small": GeneralizationTrajectorySpec(
        name="phase_shift_small",
        dataset_split=INTERPOLATION_TEST,
        start_angles_deg=(20.0, 20.0),
        end_angles_deg=(110.0, 120.0),
        flexion_duration_s=6.0,
        extension_duration_s=6.0,
        speed_profile="nominal_phase_shift",
        domain_relation="inside_training_domain",
        description="knee motion leads hip motion slightly and smoothly",
        knee_phase_lead_strength=3.0,
    ),
    "amplitude_mix": GeneralizationTrajectorySpec(
        name="amplitude_mix",
        dataset_split=INTERPOLATION_TEST,
        start_angles_deg=(25.0, 25.0),
        end_angles_deg=(95.0, 105.0),
        flexion_duration_s=6.0,
        extension_duration_s=6.0,
        speed_profile="nominal_amplitude_mix",
        domain_relation="inside_training_domain",
        description="unseen hip/knee amplitude ratio inside training bounds",
    ),
    "intermediate_speed": GeneralizationTrajectorySpec(
        name="intermediate_speed",
        dataset_split=INTERPOLATION_TEST,
        start_angles_deg=(20.0, 20.0),
        end_angles_deg=(70.0, 120.0),
        flexion_duration_s=4.5,
        extension_duration_s=4.5,
        speed_profile="intermediate_nominal_fast",
        domain_relation="inside_training_domain",
        description="coupled path at a duration between nominal and fast",
    ),
    "asymmetric_flexion_extension": GeneralizationTrajectorySpec(
        name="asymmetric_flexion_extension",
        dataset_split=INTERPOLATION_TEST,
        start_angles_deg=(25.0, 25.0),
        end_angles_deg=(85.0, 110.0),
        flexion_duration_s=4.5,
        extension_duration_s=8.0,
        speed_profile="asymmetric_flexion_extension",
        domain_relation="inside_training_domain",
        description="flexion is faster than extension",
    ),
    "boundary_near": GeneralizationTrajectorySpec(
        name="boundary_near",
        dataset_split=BOUNDARY_TEST,
        start_angles_deg=(21.0, 22.0),
        end_angles_deg=(118.0, 78.0),
        flexion_duration_s=6.0,
        extension_duration_s=6.0,
        speed_profile="nominal_boundary_near",
        domain_relation="near_training_boundary",
        description="approaches the trained hip-angle boundary without crossing it",
    ),
    "outside_domain": GeneralizationTrajectorySpec(
        name="outside_domain",
        dataset_split=OUTSIDE_DOMAIN_TEST,
        start_angles_deg=(10.0, 10.0),
        end_angles_deg=(115.0, 128.0),
        flexion_duration_s=6.0,
        extension_duration_s=6.0,
        speed_profile="nominal_outside_domain",
        domain_relation="outside_training_domain",
        description=(
            "exceeds trained angle coverage while remaining inside total human "
            "joint and bed workspace limits"
        ),
    ),
}

GENERALIZATION_TRAJECTORY_NAMES = tuple(GENERALIZATION_TRAJECTORIES)


def _training_position_limits_deg() -> tuple[tuple[float, float], tuple[float, float]]:
    endpoints = np.asarray(
        [
            endpoint
            for start_end in identification_trajectory_endpoints_deg.values()
            for endpoint in start_end
        ],
        dtype=float,
    )
    return (
        (float(endpoints[:, 0].min()), float(endpoints[:, 0].max())),
        (float(endpoints[:, 1].min()), float(endpoints[:, 1].max())),
    )


TRAINING_HIP_RANGE_DEG, TRAINING_KNEE_RANGE_DEG = (
    _training_position_limits_deg()
)
TRAINING_HIP_RANGE_RAD = tuple(np.deg2rad(TRAINING_HIP_RANGE_DEG))
TRAINING_KNEE_RANGE_RAD = tuple(np.deg2rad(TRAINING_KNEE_RANGE_DEG))


def _validate_sampling_frequency(sampling_frequency_hz: float) -> float:
    frequency = float(sampling_frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    return frequency


def _sample_segment(
    duration_s: float,
    sampling_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("trajectory duration must be finite and positive.")
    interval_count = max(1, int(round(duration * sampling_frequency_hz)))
    local_time_s = np.linspace(0.0, duration, interval_count + 1)
    return local_time_s, local_time_s / duration


def _phase_lead_progress(
    u: np.ndarray,
    duration_s: float,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最小 jerk 加端点二阶导数为零的微小相位超前项。"""

    base, base_velocity, base_acceleration = minimum_jerk_profile(u, duration_s)
    if strength == 0.0:
        return (
            np.asarray(base, dtype=float),
            np.asarray(base_velocity, dtype=float),
            np.asarray(base_acceleration, dtype=float),
        )

    # h=u^3(1-u)^3；h、h'、h'' 在 u=0/1 均为零，所以不会破坏
    # minimum-jerk 的端点速度和加速度。正 strength 表示膝相对髋提前。
    h = u**3 * (1.0 - u) ** 3
    dh_du = 3.0 * u**2 - 12.0 * u**3 + 15.0 * u**4 - 6.0 * u**5
    d2h_du2 = 6.0 * u - 36.0 * u**2 + 60.0 * u**3 - 30.0 * u**4
    progress = np.asarray(base, dtype=float) + strength * h
    velocity = np.asarray(base_velocity, dtype=float) + (
        strength * dh_du / duration_s
    )
    acceleration = np.asarray(base_acceleration, dtype=float) + (
        strength * d2h_du2 / duration_s**2
    )
    if (
        np.any(progress < -1e-12)
        or np.any(progress > 1.0 + 1e-12)
        or np.any(velocity < -1e-12)
    ):
        raise ValueError("phase lead strength produced a non-monotonic profile.")
    return progress, velocity, acceleration


def _joint_segment(
    start_rad: np.ndarray,
    end_rad: np.ndarray,
    duration_s: float,
    sampling_frequency_hz: float,
    knee_phase_lead_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_time_s, u = _sample_segment(duration_s, sampling_frequency_hz)
    hip_progress = _phase_lead_progress(u, duration_s, 0.0)
    knee_progress = _phase_lead_progress(
        u,
        duration_s,
        knee_phase_lead_strength,
    )
    progress = np.column_stack((hip_progress[0], knee_progress[0]))
    progress_velocity = np.column_stack((hip_progress[1], knee_progress[1]))
    progress_acceleration = np.column_stack((hip_progress[2], knee_progress[2]))
    delta = end_rad - start_rad
    q = start_rad + progress * delta
    dq = progress_velocity * delta
    ddq = progress_acceleration * delta
    return local_time_s, progress, q, dq, ddq


def _outside_training_position_domain(
    q_hip_rad: np.ndarray,
    q_knee_rad: np.ndarray,
) -> np.ndarray:
    return (
        (q_hip_rad < TRAINING_HIP_RANGE_RAD[0] - _ANGLE_TOLERANCE_RAD)
        | (q_hip_rad > TRAINING_HIP_RANGE_RAD[1] + _ANGLE_TOLERANCE_RAD)
        | (q_knee_rad < TRAINING_KNEE_RANGE_RAD[0] - _ANGLE_TOLERANCE_RAD)
        | (q_knee_rad > TRAINING_KNEE_RANGE_RAD[1] + _ANGLE_TOLERANCE_RAD)
    )


def generate_generalization_trajectory(
    name: str,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
) -> pd.DataFrame:
    """生成一条未见的屈曲-伸展软件验证轨迹。

    ``time_s`` 对每条轨迹从零开始并严格递增。所有 q、dq、ddq 分别使用
    rad、rad/s、rad/s²。``outside_domain`` 只超出训练角度覆盖，不超出
    配置的人体总关节范围或床面工作空间。
    """

    try:
        spec = GENERALIZATION_TRAJECTORIES[str(name)]
    except KeyError as exc:
        choices = ", ".join(GENERALIZATION_TRAJECTORY_NAMES)
        raise ValueError(
            f"Unknown generalization trajectory {name!r}; choose one of: "
            f"{choices}."
        ) from exc
    frequency = _validate_sampling_frequency(sampling_frequency_hz)
    start_rad = np.deg2rad(np.asarray(spec.start_angles_deg, dtype=float))
    end_rad = np.deg2rad(np.asarray(spec.end_angles_deg, dtype=float))

    flex = _joint_segment(
        start_rad,
        end_rad,
        spec.flexion_duration_s,
        frequency,
        spec.knee_phase_lead_strength,
    )
    extension = _joint_segment(
        end_rad,
        start_rad,
        spec.extension_duration_s,
        frequency,
        spec.knee_phase_lead_strength,
    )
    flex_time, flex_progress, flex_q, flex_dq, flex_ddq = flex
    ext_time, ext_progress, ext_q, ext_dq, ext_ddq = extension

    # 最大屈曲点只保留一次；解析的 extension 在连接点仍从零速、零加速度
    # 开始，只是跳过重复的 t=0 输出行。
    time_s = np.concatenate(
        (flex_time, spec.flexion_duration_s + ext_time[1:])
    )
    q = np.concatenate((flex_q, ext_q[1:]), axis=0)
    dq = np.concatenate((flex_dq, ext_dq[1:]), axis=0)
    ddq = np.concatenate((flex_ddq, ext_ddq[1:]), axis=0)
    hip_path_progress = np.concatenate(
        (flex_progress[:, 0], 1.0 - ext_progress[1:, 0])
    )
    knee_path_progress = np.concatenate(
        (flex_progress[:, 1], 1.0 - ext_progress[1:, 1])
    )
    phase = np.concatenate(
        (
            np.full(len(flex_time), "flexion", dtype="<U9"),
            np.full(len(ext_time) - 1, "extension", dtype="<U9"),
        )
    )

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q[:, 0],
        q[:, 1],
        L1,
        L2,
    )
    outside_training = _outside_training_position_domain(q[:, 0], q[:, 1])
    workspace_valid = (
        (np.asarray(z_knee) >= -_WORKSPACE_TOLERANCE_M)
        & (np.asarray(x_pull) >= -_WORKSPACE_TOLERANCE_M)
        & (np.asarray(z_pull) >= -_WORKSPACE_TOLERANCE_M)
    )
    trajectory = pd.DataFrame(
        {
            "trajectory_id": (
                f"{SOFTWARE_VALIDATION_TRAJECTORY}:{spec.name}"
            ),
            "generalization_family": spec.name,
            "trajectory_family": spec.name,
            "speed_profile": spec.speed_profile,
            "dataset_split": spec.dataset_split,
            "phase": phase,
            "time_s": time_s,
            "trajectory_sample_index": np.arange(len(time_s), dtype=int),
            "path_progress": hip_path_progress,
            "hip_path_progress": hip_path_progress,
            "knee_path_progress": knee_path_progress,
            "q_hip_rad": q[:, 0],
            "q_knee_rad": q[:, 1],
            "dq_hip_rad_s": dq[:, 0],
            "dq_knee_rad_s": dq[:, 1],
            "ddq_hip_rad_s2": ddq[:, 0],
            "ddq_knee_rad_s2": ddq[:, 1],
            "x_knee_m": x_knee,
            "z_knee_m": z_knee,
            "x_pull_m": x_pull,
            "z_pull_m": z_pull,
            "workspace_valid": workspace_valid,
            "outside_training_domain": outside_training,
            "outside_training_position_domain": outside_training,
            "software_validation_trajectory": True,
            "clinical_reference": False,
            "trajectory_is_extrapolation": (
                spec.dataset_split == OUTSIDE_DOMAIN_TEST
            ),
            "domain_relation": spec.domain_relation,
            "phase_relationship": (
                "knee_leads_hip"
                if spec.knee_phase_lead_strength > 0.0
                else "synchronous_minimum_jerk"
            ),
            "flexion_duration_s": spec.flexion_duration_s,
            "extension_duration_s": spec.extension_duration_s,
        }
    )
    trajectory.attrs.update(
        {
            "trajectory_kind": SOFTWARE_VALIDATION_TRAJECTORY,
            "generalization_family": spec.name,
            "dataset_split": spec.dataset_split,
            "angle_definition": "theta_shank = q_hip - q_knee",
            "angle_unit": "rad",
            "velocity_unit": "rad/s",
            "acceleration_unit": "rad/s^2",
            "training_position_domain_deg": {
                "hip": TRAINING_HIP_RANGE_DEG,
                "knee": TRAINING_KNEE_RANGE_DEG,
            },
            "clinical_reference": False,
        }
    )
    validate_generalization_trajectory(trajectory)
    return trajectory


def _validate_single_trajectory(
    trajectory: pd.DataFrame,
    spec: GeneralizationTrajectorySpec,
) -> None:
    time_s = trajectory["time_s"].to_numpy(dtype=float)
    if len(trajectory) < 3 or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("generalization trajectory time must be strictly increasing.")
    finite_columns = [
        "time_s",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "x_knee_m",
        "z_knee_m",
        "x_pull_m",
        "z_pull_m",
    ]
    if not np.isfinite(trajectory[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("generalization trajectory contains non-finite values.")
    if set(trajectory["phase"].astype(str)) != {"flexion", "extension"}:
        raise ValueError("trajectory must contain flexion and extension phases.")
    if set(trajectory["dataset_split"].astype(str)) != {spec.dataset_split}:
        raise ValueError("generalization trajectory split does not match its spec.")
    expected_id = f"{SOFTWARE_VALIDATION_TRAJECTORY}:{spec.name}"
    if set(trajectory["trajectory_id"].astype(str)) != {expected_id}:
        raise ValueError("generalization trajectory_id does not match its spec.")
    if not trajectory["software_validation_trajectory"].astype(bool).all():
        raise ValueError("trajectory is not marked as software validation data.")
    if trajectory["clinical_reference"].astype(bool).any():
        raise ValueError("generalization trajectories cannot be clinical references.")
    endpoint_derivatives = trajectory.iloc[[0, -1]][
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    if not np.allclose(endpoint_derivatives, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("trajectory endpoint velocity/acceleration must be zero.")

    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
    hip_limits_rad = np.deg2rad(hip_range_deg)
    knee_limits_rad = np.deg2rad(knee_range_deg)
    if (
        np.any(q_hip < hip_limits_rad[0] - _ANGLE_TOLERANCE_RAD)
        or np.any(q_hip > hip_limits_rad[1] + _ANGLE_TOLERANCE_RAD)
        or np.any(q_knee < knee_limits_rad[0] - _ANGLE_TOLERANCE_RAD)
        or np.any(q_knee > knee_limits_rad[1] + _ANGLE_TOLERANCE_RAD)
    ):
        raise ValueError("trajectory exceeds total human joint limits.")

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip,
        q_knee,
        L1,
        L2,
    )
    expected_geometry = np.column_stack((x_knee, z_knee, x_pull, z_pull))
    stored_geometry = trajectory[
        ["x_knee_m", "z_knee_m", "x_pull_m", "z_pull_m"]
    ].to_numpy(dtype=float)
    if not np.allclose(expected_geometry, stored_geometry, atol=1e-12, rtol=0.0):
        raise ValueError("stored workspace coordinates are inconsistent.")
    workspace_valid = (
        (np.asarray(z_knee) >= -_WORKSPACE_TOLERANCE_M)
        & (np.asarray(x_pull) >= -_WORKSPACE_TOLERANCE_M)
        & (np.asarray(z_pull) >= -_WORKSPACE_TOLERANCE_M)
    )
    stored_workspace_valid = trajectory["workspace_valid"].astype(bool)
    if not workspace_valid.all() or not stored_workspace_valid.all():
        raise ValueError("trajectory leaves the above-bed forward workspace.")

    expected_outside = _outside_training_position_domain(q_hip, q_knee)
    observed_outside = trajectory["outside_training_domain"].astype(bool).to_numpy()
    if not np.array_equal(expected_outside, observed_outside):
        raise ValueError("outside-training-domain markers are inconsistent.")
    if spec.dataset_split == OUTSIDE_DOMAIN_TEST:
        if not expected_outside.any():
            raise ValueError("outside_domain must leave training position coverage.")
    elif expected_outside.any():
        raise ValueError("non-extrapolation trajectory left training coverage.")


def validate_generalization_trajectory(trajectory: pd.DataFrame) -> None:
    """验证一条轨迹或由本模块构建的六轨迹拼接表。"""

    required = {
        "trajectory_id",
        "generalization_family",
        "trajectory_family",
        "speed_profile",
        "dataset_split",
        "phase",
        "time_s",
        "trajectory_sample_index",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "x_knee_m",
        "z_knee_m",
        "x_pull_m",
        "z_pull_m",
        "workspace_valid",
        "outside_training_domain",
        "software_validation_trajectory",
        "clinical_reference",
    }
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(
            f"generalization trajectory is missing columns: {sorted(missing)}"
        )
    if trajectory.empty:
        raise ValueError("generalization trajectory cannot be empty.")
    observed_names = set(trajectory["generalization_family"].astype(str))
    unknown = observed_names.difference(GENERALIZATION_TRAJECTORIES)
    if unknown:
        raise ValueError(f"unknown generalization families: {sorted(unknown)}")
    for name, group in trajectory.groupby("generalization_family", sort=False):
        _validate_single_trajectory(
            group.reset_index(drop=True),
            GENERALIZATION_TRAJECTORIES[str(name)],
        )


def build_generalization_trajectory_set(
    trajectory_names: Iterable[str] | None = None,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
) -> pd.DataFrame:
    """拼接所选未见轨迹；每个 family 的 ``time_s`` 独立从零开始。"""

    if trajectory_names is None:
        names = GENERALIZATION_TRAJECTORY_NAMES
    elif isinstance(trajectory_names, str):
        names = (trajectory_names,)
    else:
        names = tuple(str(name) for name in trajectory_names)
    if not names:
        raise ValueError("trajectory_names must contain at least one name.")
    if len(set(names)) != len(names):
        raise ValueError("trajectory_names cannot contain duplicates.")
    frequency = _validate_sampling_frequency(sampling_frequency_hz)
    frames = [
        generate_generalization_trajectory(name, frequency)
        for name in names
    ]
    result = pd.concat(frames, ignore_index=True)
    result.attrs.update(
        {
            "trajectory_kind": SOFTWARE_VALIDATION_TRAJECTORY,
            "generalization_families": list(names),
            "angle_definition": "theta_shank = q_hip - q_knee",
            "clinical_reference": False,
        }
    )
    validate_generalization_trajectory(result)
    return result


# Readable compatibility aliases for callers that prefer a plural builder.
build_generalization_trajectories = build_generalization_trajectory_set
generate_generalization_trajectory_set = build_generalization_trajectory_set


__all__ = [
    "BOUNDARY_TEST",
    "GENERALIZATION_SPLITS",
    "GENERALIZATION_TRAJECTORIES",
    "GENERALIZATION_TRAJECTORY_NAMES",
    "GeneralizationTrajectorySpec",
    "INTERPOLATION_TEST",
    "OUTSIDE_DOMAIN_TEST",
    "SOFTWARE_VALIDATION_TRAJECTORY",
    "TRAINING_HIP_RANGE_DEG",
    "TRAINING_HIP_RANGE_RAD",
    "TRAINING_KNEE_RANGE_DEG",
    "TRAINING_KNEE_RANGE_RAD",
    "build_generalization_trajectories",
    "build_generalization_trajectory_set",
    "generate_generalization_trajectory",
    "generate_generalization_trajectory_set",
    "validate_generalization_trajectory",
]
