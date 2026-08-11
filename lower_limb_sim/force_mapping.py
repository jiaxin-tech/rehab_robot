"""将髋膝关节需求力矩映射为束缚带牵引点二维力。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    force_magnitude_limit_n,
    jacobian_condition_limit,
    jacobian_det_threshold,
)
from .jacobian import jacobian_diagnostics, leg_jacobian


@dataclass(frozen=True)
class EndpointForceResult:
    """机器人通过束缚带施加在腿上的二维力及映射有效性。"""

    fx_robot_on_leg_n: float | np.ndarray
    fz_robot_on_leg_n: float | np.ndarray
    force_magnitude_n: float | np.ndarray
    jacobian_determinant: float | np.ndarray
    jacobian_condition_number: float | np.ndarray
    jacobian_near_singular: bool | np.ndarray
    force_mapping_valid: bool | np.ndarray
    invalid_reason: str | np.ndarray


def _append_reason(
    reasons: np.ndarray,
    mask: np.ndarray,
    reason: str,
) -> None:
    for index in np.ndindex(reasons.shape):
        if bool(mask[index]):
            current = str(reasons[index])
            reasons[index] = f"{current};{reason}" if current else reason


def endpoint_force_from_joint_torque(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    tau_hip: float | np.ndarray,
    tau_knee: float | np.ndarray,
    L1: float,
    L2: float,
    det_threshold: float = jacobian_det_threshold,
    condition_limit: float = jacobian_condition_limit,
    force_limit_n: float = force_magnitude_limit_n,
) -> EndpointForceResult:
    """由 ``tau = J.T @ F`` 使用伪逆计算机器人施加在腿上的力。

    接近奇异、输入/结果非有限或力幅值超过配置上限时，力字段返回 NaN，
    ``force_mapping_valid=False``，并记录明确的 ``invalid_reason``。禁止
    使用 ``inv(J.T)``。
    """

    if not np.isfinite(force_limit_n) or force_limit_n <= 0.0:
        raise ValueError("force_limit_n must be finite and positive.")

    q_hip_array, q_knee_array, tau_hip_array, tau_knee_array = (
        np.broadcast_arrays(
            np.asarray(q_hip, dtype=float),
            np.asarray(q_knee, dtype=float),
            np.asarray(tau_hip, dtype=float),
            np.asarray(tau_knee, dtype=float),
        )
    )
    jacobian = leg_jacobian(q_hip_array, q_knee_array, L1, L2)
    diagnostics = jacobian_diagnostics(
        q_hip_array,
        q_knee_array,
        L1,
        L2,
        det_threshold=det_threshold,
        condition_limit=condition_limit,
    )

    finite_input = (
        np.isfinite(q_hip_array)
        & np.isfinite(q_knee_array)
        & np.isfinite(tau_hip_array)
        & np.isfinite(tau_knee_array)
    )
    identity = np.broadcast_to(np.eye(2), jacobian.shape)
    safe_jacobian = np.where(
        finite_input[..., np.newaxis, np.newaxis],
        jacobian,
        identity,
    )
    torque = np.stack((tau_hip_array, tau_knee_array), axis=-1)
    safe_torque = np.where(finite_input[..., np.newaxis], torque, 0.0)

    transposed = np.swapaxes(safe_jacobian, -1, -2)
    force = np.matmul(np.linalg.pinv(transposed), safe_torque[..., np.newaxis])[
        ..., 0
    ]
    fx = force[..., 0]
    fz = force[..., 1]
    magnitude = np.hypot(fx, fz)
    finite_force = np.isfinite(fx) & np.isfinite(fz) & np.isfinite(magnitude)
    excessive_force = (
        (np.abs(fx) > force_limit_n)
        | (np.abs(fz) > force_limit_n)
        | (magnitude > force_limit_n)
    )
    near_singular = np.asarray(diagnostics.near_singular, dtype=bool)
    valid = finite_input & ~near_singular & finite_force & ~excessive_force

    reasons = np.full(valid.shape, "", dtype=object)
    _append_reason(reasons, ~finite_input, "non_finite_input")
    _append_reason(reasons, near_singular, "jacobian_near_singular")
    _append_reason(reasons, finite_input & ~finite_force, "non_finite_force")
    _append_reason(
        reasons,
        finite_input & finite_force & excessive_force,
        "force_magnitude_limit_exceeded",
    )

    fx = np.where(valid, fx, np.nan)
    fz = np.where(valid, fz, np.nan)
    magnitude = np.where(valid, magnitude, np.nan)
    if q_hip_array.ndim == 0:
        return EndpointForceResult(
            fx_robot_on_leg_n=float(fx),
            fz_robot_on_leg_n=float(fz),
            force_magnitude_n=float(magnitude),
            jacobian_determinant=float(diagnostics.determinant),
            jacobian_condition_number=float(diagnostics.condition_number),
            jacobian_near_singular=bool(near_singular),
            force_mapping_valid=bool(valid),
            invalid_reason=str(reasons.item()),
        )
    return EndpointForceResult(
        fx_robot_on_leg_n=fx,
        fz_robot_on_leg_n=fz,
        force_magnitude_n=magnitude,
        jacobian_determinant=diagnostics.determinant,
        jacobian_condition_number=diagnostics.condition_number,
        jacobian_near_singular=near_singular,
        force_mapping_valid=valid,
        invalid_reason=reasons.astype(str),
    )
