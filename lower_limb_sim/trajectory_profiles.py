"""解析最小 jerk 的关节空间 ``software_test_trajectory``。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    dynamic_sampling_frequency_hz,
    identification_trajectory_endpoints_deg,
    identification_trajectory_id,
    speed_profile_one_way_duration_s,
    test_trajectory_end_deg,
    test_trajectory_start_deg,
)

TRAJECTORY_ID = "software_test_trajectory"
IDENTIFICATION_TRAJECTORY_ID = identification_trajectory_id


def minimum_jerk_profile(
    u: float | np.ndarray,
    duration_s: float,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """解析计算最小 jerk 的 ``s, ds/dt, d²s/dt²``。

    ``u`` 必须位于 [0, 1]，``duration_s`` 是该单程动作时长。
    """

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive.")
    u_array = np.asarray(u, dtype=float)
    if np.any(~np.isfinite(u_array)) or np.any((u_array < 0.0) | (u_array > 1.0)):
        raise ValueError("u must contain finite values in [0, 1].")

    s = 10.0 * u_array**3 - 15.0 * u_array**4 + 6.0 * u_array**5
    ds_du = 30.0 * u_array**2 - 60.0 * u_array**3 + 30.0 * u_array**4
    d2s_du2 = 60.0 * u_array - 180.0 * u_array**2 + 120.0 * u_array**3
    ds_dt = ds_du / duration_s
    d2s_dt2 = d2s_du2 / duration_s**2
    if u_array.ndim == 0:
        return float(s), float(ds_dt), float(d2s_dt2)
    return s, ds_dt, d2s_dt2


def generate_software_test_trajectory(
    speed_profile: str,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
    start_angles_deg: tuple[float, float] = test_trajectory_start_deg,
    end_angles_deg: tuple[float, float] = test_trajectory_end_deg,
) -> pd.DataFrame:
    """生成屈曲后伸展返回的解析最小 jerk 关节轨迹。

    连接点只保留一次：最大屈曲点属于 ``flexion``，``extension`` 从下一
    个严格递增的采样时刻开始。
    """

    if speed_profile not in speed_profile_one_way_duration_s:
        choices = ", ".join(speed_profile_one_way_duration_s)
        raise ValueError(
            f"Unknown speed profile {speed_profile!r}; choose one of: {choices}."
        )
    if (
        not np.isfinite(sampling_frequency_hz)
        or sampling_frequency_hz <= 0.0
    ):
        raise ValueError("sampling_frequency_hz must be finite and positive.")

    duration_s = speed_profile_one_way_duration_s[speed_profile]
    interval_count = max(1, int(round(duration_s * sampling_frequency_hz)))
    local_time = np.linspace(0.0, duration_s, interval_count + 1)
    u = local_time / duration_s
    s, ds_dt, d2s_dt2 = minimum_jerk_profile(u, duration_s)

    start = np.deg2rad(np.asarray(start_angles_deg, dtype=float))
    end = np.deg2rad(np.asarray(end_angles_deg, dtype=float))
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("start_angles_deg and end_angles_deg must contain two values.")
    delta = end - start

    flex_q = start + s[:, np.newaxis] * delta
    flex_dq = ds_dt[:, np.newaxis] * delta
    flex_ddq = d2s_dt2[:, np.newaxis] * delta

    extension_s = 1.0 - s
    extension_q = start + extension_s[:, np.newaxis] * delta
    extension_dq = -ds_dt[:, np.newaxis] * delta
    extension_ddq = -d2s_dt2[:, np.newaxis] * delta

    # 去掉 extension 的局部 u=0，避免在最大屈曲连接点重复时间和姿态。
    time_s = np.concatenate((local_time, duration_s + local_time[1:]))
    q = np.concatenate((flex_q, extension_q[1:]), axis=0)
    dq = np.concatenate((flex_dq, extension_dq[1:]), axis=0)
    ddq = np.concatenate((flex_ddq, extension_ddq[1:]), axis=0)
    phase = np.concatenate(
        (
            np.full(interval_count + 1, "flexion", dtype="<U9"),
            np.full(interval_count, "extension", dtype="<U9"),
        )
    )
    path_progress = np.concatenate((s, extension_s[1:]))

    trajectory = pd.DataFrame(
        {
            "trajectory_id": TRAJECTORY_ID,
            "speed_profile": speed_profile,
            "phase": phase,
            "time_s": time_s,
            "path_progress": path_progress,
            "q_hip_rad": q[:, 0],
            "q_knee_rad": q[:, 1],
            "dq_hip_rad_s": dq[:, 0],
            "dq_knee_rad_s": dq[:, 1],
            "ddq_hip_rad_s2": ddq[:, 0],
            "ddq_knee_rad_s2": ddq[:, 1],
        }
    )
    validate_trajectory_profile(trajectory)
    return trajectory


def generate_identification_excitation_trajectory(
    trajectory_family: str,
    speed_profile: str,
    sampling_frequency_hz: float = dynamic_sampling_frequency_hz,
) -> pd.DataFrame:
    """生成一条第四阶段辨识激励轨迹。

    三个 family 都复用解析最小 jerk 往返生成器。它们只用于软件虚拟
    受试者参数辨识，不是临床参考轨迹。
    """

    try:
        start_angles_deg, end_angles_deg = (
            identification_trajectory_endpoints_deg[trajectory_family]
        )
    except KeyError as exc:
        choices = ", ".join(identification_trajectory_endpoints_deg)
        raise ValueError(
            f"Unknown trajectory family {trajectory_family!r}; "
            f"choose one of: {choices}."
        ) from exc

    trajectory = generate_software_test_trajectory(
        speed_profile=speed_profile,
        sampling_frequency_hz=sampling_frequency_hz,
        start_angles_deg=start_angles_deg,
        end_angles_deg=end_angles_deg,
    ).copy()
    trajectory["trajectory_id"] = IDENTIFICATION_TRAJECTORY_ID
    trajectory.insert(1, "trajectory_family", trajectory_family)
    trajectory["clinical_reference"] = False
    validate_trajectory_profile(trajectory)
    return trajectory


def validate_trajectory_profile(trajectory: pd.DataFrame) -> None:
    """验证时间、端点和解析最小 jerk 的基本连续性。"""

    required = {
        "phase",
        "time_s",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    }
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")
    if len(trajectory) < 3:
        raise ValueError("trajectory must contain at least three samples.")
    if not np.all(np.diff(trajectory["time_s"].to_numpy(dtype=float)) > 0.0):
        raise ValueError("trajectory time must be strictly increasing.")

    endpoint_columns = (
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    endpoints = trajectory.iloc[[0, -1]][list(endpoint_columns)].to_numpy(
        dtype=float
    )
    if not np.allclose(endpoints, 0.0, atol=1e-12):
        raise ValueError("minimum-jerk endpoint velocity/acceleration must be zero.")
    if set(trajectory["phase"]) != {"flexion", "extension"}:
        raise ValueError("trajectory must contain flexion and extension phases.")
