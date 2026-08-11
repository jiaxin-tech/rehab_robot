"""由束缚带牵引点二维力重建人体髋、膝广义力矩。"""

from __future__ import annotations

import numpy as np

from .jacobian import leg_jacobian


def joint_torque_from_endpoint_force(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    fx: float | np.ndarray,
    fz: float | np.ndarray,
    L1: float,
    L2: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """计算 ``tau_measured = J(q).T @ [Fx, Fz]``。

    Jacobian 由 :mod:`lower_limb_sim.jacobian` 统一提供，因此这里不会复制
    运动学公式，并继续严格使用 ``theta_shank = q_hip - q_knee``。
    """

    q_hip_array, q_knee_array, fx_array, fz_array = np.broadcast_arrays(
        np.asarray(q_hip, dtype=float),
        np.asarray(q_knee, dtype=float),
        np.asarray(fx, dtype=float),
        np.asarray(fz, dtype=float),
    )
    jacobian = leg_jacobian(q_hip_array, q_knee_array, L1, L2)
    endpoint_force = np.stack((fx_array, fz_array), axis=-1)
    torque = np.matmul(
        np.swapaxes(jacobian, -1, -2),
        endpoint_force[..., np.newaxis],
    )[..., 0]
    if q_hip_array.ndim == 0:
        return float(torque[0]), float(torque[1])
    return torque[..., 0], torque[..., 1]
