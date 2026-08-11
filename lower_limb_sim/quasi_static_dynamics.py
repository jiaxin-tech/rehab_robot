"""虚拟受试者下肢的准静态重力与线性被动刚度力矩。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .virtual_subject import VirtualSubject


@dataclass(frozen=True)
class QuasiStaticJointTorque:
    """准静态关节需求力矩的分项结果，单位均为 N·m。"""

    tau_gravity_hip_nm: float | np.ndarray
    tau_gravity_knee_nm: float | np.ndarray
    tau_stiffness_hip_nm: float | np.ndarray
    tau_stiffness_knee_nm: float | np.ndarray
    tau_total_hip_nm: float | np.ndarray
    tau_total_knee_nm: float | np.ndarray


def _broadcast_angles(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.broadcast_arrays(
        np.asarray(q_hip, dtype=float),
        np.asarray(q_knee, dtype=float),
    )


def _maybe_scalar(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    if first.ndim == 0:
        return float(first), float(second)
    return first, second


def gravity_torque(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: VirtualSubject,
    L1: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """返回髋、膝准静态重力需求力矩。

    小腿绝对角为 ``q_hip - q_knee``，因此膝重力项带负号。
    """

    if not np.isfinite(L1) or L1 <= 0.0:
        raise ValueError("L1 must be finite and positive.")
    q_hip_array, q_knee_array = _broadcast_angles(q_hip, q_knee)
    shank_angle = q_hip_array - q_knee_array
    gravity = subject.gravity_m_s2

    tau_hip = (
        (
            subject.mass_thigh_kg * subject.com_thigh_m
            + subject.mass_shank_kg * L1
        )
        * gravity
        * np.cos(q_hip_array)
        + subject.mass_shank_kg
        * subject.com_shank_m
        * gravity
        * np.cos(shank_angle)
    )
    tau_knee = (
        -subject.mass_shank_kg
        * subject.com_shank_m
        * gravity
        * np.cos(shank_angle)
    )
    return _maybe_scalar(tau_hip, tau_knee)


def passive_stiffness_torque(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: VirtualSubject,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """返回相对中性角的线性被动髋、膝刚度需求力矩。"""

    q_hip_array, q_knee_array = _broadcast_angles(q_hip, q_knee)
    tau_hip = subject.k_hip_nm_per_rad * (
        q_hip_array - subject.q0_hip_rad
    )
    tau_knee = subject.k_knee_nm_per_rad * (
        q_knee_array - subject.q0_knee_rad
    )
    return _maybe_scalar(tau_hip, tau_knee)


def quasi_static_joint_torque(
    q_hip: float | np.ndarray,
    q_knee: float | np.ndarray,
    subject: VirtualSubject,
    L1: float,
) -> QuasiStaticJointTorque:
    """组合重力与线性被动刚度，保留全部力矩分项。"""

    gravity_hip, gravity_knee = gravity_torque(q_hip, q_knee, subject, L1)
    stiffness_hip, stiffness_knee = passive_stiffness_torque(
        q_hip,
        q_knee,
        subject,
    )
    return QuasiStaticJointTorque(
        tau_gravity_hip_nm=gravity_hip,
        tau_gravity_knee_nm=gravity_knee,
        tau_stiffness_hip_nm=stiffness_hip,
        tau_stiffness_knee_nm=stiffness_knee,
        tau_total_hip_nm=gravity_hip + stiffness_hip,
        tau_total_knee_nm=gravity_knee + stiffness_knee,
    )
