"""束缚带牵引点相对于髋、膝屈曲角的二维 Jacobian。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import jacobian_condition_limit, jacobian_det_threshold


@dataclass(frozen=True)
class JacobianDiagnostics:
    """Jacobian 数值状态；标量输入返回标量，数组输入返回 NumPy 数组。"""

    determinant: float | np.ndarray
    condition_number: float | np.ndarray
    near_singular: bool | np.ndarray


def _validate_link_lengths(L1: float, L2: float) -> None:
    if not np.isfinite(L1) or not np.isfinite(L2) or L1 <= 0.0 or L2 <= 0.0:
        raise ValueError("L1 and L2 must be finite positive lengths.")


def leg_jacobian(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    L1: float,
    L2: float,
) -> np.ndarray:
    """返回 ``d[x_pull, z_pull] / d[q_hip, q_knee]``。

    本模型的膝屈曲角为正，但小腿绝对方向是
    ``q_hip - q_knee``，不是普通二连杆常见的 ``q1 + q2``。标量输入
    返回形状 ``(2, 2)``，可广播数组输入返回形状 ``(..., 2, 2)``。
    """

    _validate_link_lengths(L1, L2)
    q_hip_array, q_knee_array = np.broadcast_arrays(
        np.asarray(q_hip, dtype=float),
        np.asarray(q_knee, dtype=float),
    )
    shank_angle = q_hip_array - q_knee_array

    j11 = -L1 * np.sin(q_hip_array) - L2 * np.sin(shank_angle)
    j12 = L2 * np.sin(shank_angle)
    j21 = L1 * np.cos(q_hip_array) + L2 * np.cos(shank_angle)
    j22 = -L2 * np.cos(shank_angle)
    return np.stack(
        (
            np.stack((j11, j12), axis=-1),
            np.stack((j21, j22), axis=-1),
        ),
        axis=-2,
    )


def jacobian_diagnostics(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    L1: float,
    L2: float,
    det_threshold: float = jacobian_det_threshold,
    condition_limit: float = jacobian_condition_limit,
) -> JacobianDiagnostics:
    """计算 determinant、2-范数 condition number 和奇异性标记。"""

    if not np.isfinite(det_threshold) or det_threshold < 0.0:
        raise ValueError("det_threshold must be finite and non-negative.")
    if not np.isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be finite and positive.")

    jacobian = leg_jacobian(q_hip, q_knee, L1, L2)
    finite_matrix = np.all(np.isfinite(jacobian), axis=(-2, -1))
    identity = np.broadcast_to(np.eye(2), jacobian.shape)
    safe_jacobian = np.where(
        finite_matrix[..., np.newaxis, np.newaxis],
        jacobian,
        identity,
    )

    determinant = np.linalg.det(safe_jacobian)
    singular_values = np.linalg.svd(safe_jacobian, compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition_number = singular_values[..., 0] / singular_values[..., -1]
    determinant = np.where(finite_matrix, determinant, np.nan)
    condition_number = np.where(finite_matrix, condition_number, np.inf)
    near_singular = (
        ~finite_matrix
        | (np.abs(determinant) < det_threshold)
        | ~np.isfinite(condition_number)
        | (condition_number > condition_limit)
    )

    if jacobian.ndim == 2:
        return JacobianDiagnostics(
            determinant=float(determinant),
            condition_number=float(condition_number),
            near_singular=bool(near_singular),
        )
    return JacobianDiagnostics(determinant, condition_number, near_singular)
