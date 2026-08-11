"""仰卧位髋膝屈曲的二维正、逆运动学。"""

from __future__ import annotations

import numpy as np

from .config import hip_range_deg, knee_range_deg


def _validate_link_lengths(L1: float, L2: float) -> None:
    if not np.isfinite(L1) or not np.isfinite(L2) or L1 <= 0.0 or L2 <= 0.0:
        raise ValueError("L1 and L2 must be finite positive lengths.")


def forward_kinematics(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    L1: float,
    L2: float,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """计算膝关节和束缚带等效牵引点坐标。

    q_hip、q_knee 使用 rad，可以是标量或可广播的 NumPy 数组。
    膝屈曲采用小腿绝对角 ``q_hip - q_knee``。
    """

    _validate_link_lengths(L1, L2)
    q_hip_array, q_knee_array = np.broadcast_arrays(
        np.asarray(q_hip, dtype=float),
        np.asarray(q_knee, dtype=float),
    )

    x_knee = L1 * np.cos(q_hip_array)
    z_knee = L1 * np.sin(q_hip_array)
    shank_angle = q_hip_array - q_knee_array
    x_pull = x_knee + L2 * np.cos(shank_angle)
    z_pull = z_knee + L2 * np.sin(shank_angle)
    if q_hip_array.ndim == 0:
        return (
            float(x_knee),
            float(z_knee),
            float(x_pull),
            float(z_pull),
        )
    return x_knee, z_knee, x_pull, z_pull


def inverse_kinematics(
    x_pull: float | np.ndarray,
    z_pull: float | np.ndarray,
    L1: float,
    L2: float,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    bool | np.ndarray,
]:
    """求牵引点对应的仰卧屈髋、屈膝解。

    返回 ``q_hip, q_knee, reachable``。角度使用 rad。几何不可达或不在
    配置关节范围内的位置会返回 ``reachable=False``，对应角度为 NaN。
    该函数只处理关节几何约束；床面约束由工作空间图谱负责。
    """

    _validate_link_lengths(L1, L2)
    x_array, z_array = np.broadcast_arrays(
        np.asarray(x_pull, dtype=float),
        np.asarray(z_pull, dtype=float),
    )

    distance_squared = x_array**2 + z_array**2
    D = (distance_squared - L1**2 - L2**2) / (2.0 * L1 * L2)
    finite_input = np.isfinite(x_array) & np.isfinite(z_array)
    geometrically_reachable = finite_input & (D >= -1.0) & (D <= 1.0)

    # clip 仅用于保证 arccos 数值安全；原始 D 决定 reachable。
    q_knee = np.arccos(np.clip(D, -1.0, 1.0))
    q_hip = np.arctan2(z_array, x_array) + np.arctan2(
        L2 * np.sin(q_knee),
        L1 + L2 * np.cos(q_knee),
    )

    hip_min, hip_max = np.deg2rad(hip_range_deg)
    knee_min, knee_max = np.deg2rad(knee_range_deg)
    tolerance = 1e-12
    valid_posture = (
        (q_hip >= hip_min - tolerance)
        & (q_hip <= hip_max + tolerance)
        & (q_knee >= knee_min - tolerance)
        & (q_knee <= knee_max + tolerance)
    )
    reachable = geometrically_reachable & valid_posture

    q_hip = np.where(reachable, q_hip, np.nan)
    q_knee = np.where(reachable, q_knee, np.nan)
    if x_array.ndim == 0:
        return float(q_hip), float(q_knee), bool(reachable)
    return q_hip, q_knee, reachable
