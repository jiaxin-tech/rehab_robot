"""基于 ``theta_shank = q_hip - q_knee`` 的完整双连杆动力学。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamic_subject import DynamicVirtualSubject
from .quasi_static_dynamics import gravity_torque, passive_stiffness_torque


@dataclass(frozen=True)
class InverseDynamicsResult:
    """完整逆动力学的全部力矩分项，单位均为 N·m。"""

    tau_inertia_hip_nm: float | np.ndarray
    tau_inertia_knee_nm: float | np.ndarray
    tau_coriolis_hip_nm: float | np.ndarray
    tau_coriolis_knee_nm: float | np.ndarray
    tau_gravity_hip_nm: float | np.ndarray
    tau_gravity_knee_nm: float | np.ndarray
    tau_damping_hip_nm: float | np.ndarray
    tau_damping_knee_nm: float | np.ndarray
    tau_stiffness_hip_nm: float | np.ndarray
    tau_stiffness_knee_nm: float | np.ndarray
    tau_total_hip_nm: float | np.ndarray
    tau_total_knee_nm: float | np.ndarray


def _validate_thigh_length(L1: float) -> None:
    if not np.isfinite(L1) or L1 <= 0.0:
        raise ValueError("L1 must be finite and positive.")


def _broadcast(*values: float | np.ndarray) -> tuple[np.ndarray, ...]:
    return np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in values))


def _pair_to_scalar_if_needed(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    if first.ndim == 0:
        return float(first), float(second)
    return first, second


def center_of_mass_positions(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
    L1: float,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """返回大腿和小腿质心 ``x_c1, z_c1, x_c2, z_c2``。

    小腿绝对角严格为 ``q_hip - q_knee``。
    """

    _validate_thigh_length(L1)
    q_hip_array, q_knee_array = _broadcast(q_hip, q_knee)
    shank_angle = q_hip_array - q_knee_array
    x_c1 = subject.com_thigh_m * np.cos(q_hip_array)
    z_c1 = subject.com_thigh_m * np.sin(q_hip_array)
    x_c2 = (
        L1 * np.cos(q_hip_array)
        + subject.com_shank_m * np.cos(shank_angle)
    )
    z_c2 = (
        L1 * np.sin(q_hip_array)
        + subject.com_shank_m * np.sin(shank_angle)
    )
    if q_hip_array.ndim == 0:
        return float(x_c1), float(z_c1), float(x_c2), float(z_c2)
    return x_c1, z_c1, x_c2, z_c2


def mass_matrix(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
    L1: float,
) -> np.ndarray:
    """返回广义坐标 ``[q_hip, q_knee]`` 下的 2×2 质量矩阵。

    该解析式由题设动能展开得到。因为
    ``theta_shank_dot=dq_hip-dq_knee``，非对角项带负号。
    """

    _validate_thigh_length(L1)
    q_hip_array, q_knee_array = _broadcast(q_hip, q_knee)
    finite = np.isfinite(q_hip_array) & np.isfinite(q_knee_array)
    if not np.all(finite):
        raise ValueError("q_hip and q_knee must be finite.")

    a = (
        subject.mass_thigh_kg * subject.com_thigh_m**2
        + subject.inertia_thigh_kg_m2
        + subject.mass_shank_kg * L1**2
    )
    b = (
        subject.mass_shank_kg * subject.com_shank_m**2
        + subject.inertia_shank_kg_m2
    )
    coupling = subject.mass_shank_kg * L1 * subject.com_shank_m
    cosine = np.cos(q_knee_array)
    m11 = a + b + 2.0 * coupling * cosine
    m12 = -(b + coupling * cosine)
    m22 = np.broadcast_to(b, q_hip_array.shape)
    return np.stack(
        (
            np.stack((m11, m12), axis=-1),
            np.stack((m12, m22), axis=-1),
        ),
        axis=-2,
    )


def coriolis_vector(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    dq_hip: float | np.ndarray,
    dq_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
    L1: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """返回已经包含速度乘积的 ``C(q, q_dot) @ q_dot``。

    表达式由上述质量矩阵的 Christoffel 符号得到，不是 ``q1+q2`` 模型。
    """

    _validate_thigh_length(L1)
    q_hip_array, q_knee_array, dq_hip_array, dq_knee_array = _broadcast(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
    )
    coupling_sine = (
        subject.mass_shank_kg
        * L1
        * subject.com_shank_m
        * np.sin(q_knee_array)
    )
    tau_hip = coupling_sine * (
        dq_knee_array**2 - 2.0 * dq_hip_array * dq_knee_array
    )
    tau_knee = coupling_sine * dq_hip_array**2
    return _pair_to_scalar_if_needed(tau_hip, tau_knee)


def gravity_vector(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
    L1: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """返回与第二阶段准静态模型完全一致的重力向量。"""

    return gravity_torque(q_hip, q_knee, subject, L1)


def damping_torque(
    dq_hip: float | np.ndarray,
    dq_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """返回对角线性阻尼需求力矩 ``B @ q_dot``。"""

    dq_hip_array, dq_knee_array = _broadcast(dq_hip, dq_knee)
    tau_hip = subject.b_hip_nm_s_per_rad * dq_hip_array
    tau_knee = subject.b_knee_nm_s_per_rad * dq_knee_array
    return _pair_to_scalar_if_needed(tau_hip, tau_knee)


def stiffness_torque(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """复用第二阶段的独立线性被动刚度模型。"""

    return passive_stiffness_torque(q_hip, q_knee, subject)


def kinetic_energy(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    dq_hip: float | np.ndarray,
    dq_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
    L1: float,
) -> float | np.ndarray:
    """由 ``0.5 * q_dot.T @ M @ q_dot`` 计算总动能，单位 J。"""

    q_hip_array, q_knee_array, dq_hip_array, dq_knee_array = _broadcast(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
    )
    matrix = mass_matrix(q_hip_array, q_knee_array, subject, L1)
    velocity = np.stack((dq_hip_array, dq_knee_array), axis=-1)
    energy = 0.5 * np.einsum(
        "...i,...ij,...j->...",
        velocity,
        matrix,
        velocity,
    )
    return float(energy) if energy.ndim == 0 else energy


def inverse_dynamics(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    dq_hip: float | np.ndarray,
    dq_knee: float | np.ndarray,
    ddq_hip: float | np.ndarray,
    ddq_knee: float | np.ndarray,
    subject: DynamicVirtualSubject,
    L1: float,
) -> InverseDynamicsResult:
    """计算惯性、科氏/离心、重力、阻尼、刚度和总需求力矩。"""

    (
        q_hip_array,
        q_knee_array,
        dq_hip_array,
        dq_knee_array,
        ddq_hip_array,
        ddq_knee_array,
    ) = _broadcast(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
        ddq_hip,
        ddq_knee,
    )
    matrix = mass_matrix(q_hip_array, q_knee_array, subject, L1)
    acceleration = np.stack((ddq_hip_array, ddq_knee_array), axis=-1)
    inertia = np.matmul(matrix, acceleration[..., np.newaxis])[..., 0]
    inertia_hip, inertia_knee = inertia[..., 0], inertia[..., 1]
    coriolis_hip, coriolis_knee = coriolis_vector(
        q_hip_array,
        q_knee_array,
        dq_hip_array,
        dq_knee_array,
        subject,
        L1,
    )
    gravity_hip, gravity_knee = gravity_vector(
        q_hip_array,
        q_knee_array,
        subject,
        L1,
    )
    damping_hip, damping_knee = damping_torque(
        dq_hip_array,
        dq_knee_array,
        subject,
    )
    stiffness_hip, stiffness_knee = stiffness_torque(
        q_hip_array,
        q_knee_array,
        subject,
    )

    components = (
        inertia_hip,
        inertia_knee,
        np.asarray(coriolis_hip),
        np.asarray(coriolis_knee),
        np.asarray(gravity_hip),
        np.asarray(gravity_knee),
        np.asarray(damping_hip),
        np.asarray(damping_knee),
        np.asarray(stiffness_hip),
        np.asarray(stiffness_knee),
    )
    if q_hip_array.ndim == 0:
        components = tuple(float(value) for value in components)
    (
        inertia_hip,
        inertia_knee,
        coriolis_hip,
        coriolis_knee,
        gravity_hip,
        gravity_knee,
        damping_hip,
        damping_knee,
        stiffness_hip,
        stiffness_knee,
    ) = components
    return InverseDynamicsResult(
        tau_inertia_hip_nm=inertia_hip,
        tau_inertia_knee_nm=inertia_knee,
        tau_coriolis_hip_nm=coriolis_hip,
        tau_coriolis_knee_nm=coriolis_knee,
        tau_gravity_hip_nm=gravity_hip,
        tau_gravity_knee_nm=gravity_knee,
        tau_damping_hip_nm=damping_hip,
        tau_damping_knee_nm=damping_knee,
        tau_stiffness_hip_nm=stiffness_hip,
        tau_stiffness_knee_nm=stiffness_knee,
        tau_total_hip_nm=(
            inertia_hip
            + coriolis_hip
            + gravity_hip
            + damping_hip
            + stiffness_hip
        ),
        tau_total_knee_nm=(
            inertia_knee
            + coriolis_knee
            + gravity_knee
            + damping_knee
            + stiffness_knee
        ),
    )
