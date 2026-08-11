"""复杂虚拟受试者的动力学数据生成器。

基础惯性、科氏/离心、重力、线性阻尼和线性刚度全部复用
``full_dynamics.inverse_dynamics``。本模块只计算附加失配项，不复制现有
动力学、运动学或 Jacobian 公式。小腿角定义继续严格由基础模型保持为
``theta_shank = q_hip - q_knee``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .full_dynamics import damping_torque, inverse_dynamics, stiffness_torque
from .mismatch_subject import MismatchVirtualSubject


ScalarOrArray = float | np.ndarray
DEFAULT_RESIDUAL_RANDOM_SEED = 20260802


def _broadcast(*values: ScalarOrArray) -> tuple[np.ndarray, ...]:
    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=float) for value in values),
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("all kinematic inputs must be finite.")
    return arrays


def _pair_output(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    if first.ndim == 0:
        return float(first), float(second)
    return first, second


def _output(value: np.ndarray) -> ScalarOrArray:
    return float(value) if value.ndim == 0 else value


def cubic_stiffness_torque(
    q_hip: ScalarOrArray,
    q_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """返回三次刚度附加力矩，单位 N·m。"""

    q_hip_array, q_knee_array = _broadcast(q_hip, q_knee)
    delta_hip = q_hip_array - subject.q0_hip_rad
    delta_knee = q_knee_array - subject.q0_knee_rad
    hip = subject.k3_hip_nm_per_rad3 * delta_hip**3
    knee = subject.k3_knee_nm_per_rad3 * delta_knee**3
    return _pair_output(hip, knee)


def nonlinear_stiffness_torque(
    q_hip: ScalarOrArray,
    q_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """返回线性加三次项的独立关节被动刚度力矩。"""

    linear_hip, linear_knee = stiffness_torque(
        q_hip,
        q_knee,
        subject.base_dynamic_subject(),
    )
    cubic_hip, cubic_knee = cubic_stiffness_torque(q_hip, q_knee, subject)
    hip = np.asarray(linear_hip) + np.asarray(cubic_hip)
    knee = np.asarray(linear_knee) + np.asarray(cubic_knee)
    return _pair_output(hip, knee)


def coupling_potential_energy(
    q_hip: ScalarOrArray,
    q_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
) -> ScalarOrArray:
    """返回髋膝耦合势能，单位 J。

    ``V = 0.5*k*(delta_q_hip-r*delta_q_knee)^2``。该定义使两个耦合
    力矩都由同一势能梯度得到。
    """

    q_hip_array, q_knee_array = _broadcast(q_hip, q_knee)
    delta_hip = q_hip_array - subject.q0_hip_rad
    delta_knee = q_knee_array - subject.q0_knee_rad
    deformation = delta_hip - subject.k_coupling_asymmetry * delta_knee
    energy = 0.5 * subject.k_coupling_nm_per_rad * deformation**2
    return _output(energy)


def coupling_stiffness_torque(
    q_hip: ScalarOrArray,
    q_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """返回耦合势能对 ``q_hip``、``q_knee`` 的解析梯度。"""

    q_hip_array, q_knee_array = _broadcast(q_hip, q_knee)
    ratio = subject.k_coupling_asymmetry
    deformation = (
        q_hip_array
        - subject.q0_hip_rad
        - ratio * (q_knee_array - subject.q0_knee_rad)
    )
    hip = subject.k_coupling_nm_per_rad * deformation
    knee = -ratio * hip
    return _pair_output(hip, knee)


# 简短别名便于调用方表达“耦合项”，仍由上述势能梯度实现。
coupling_potential = coupling_potential_energy
coupling_torque = coupling_stiffness_torque


def quadratic_damping_torque(
    dq_hip: ScalarOrArray,
    dq_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """返回奇对称的速度平方附加阻尼力矩，单位 N·m。"""

    dq_hip_array, dq_knee_array = _broadcast(dq_hip, dq_knee)
    hip = subject.b2_hip_nm_s2_per_rad2 * np.abs(dq_hip_array) * dq_hip_array
    knee = subject.b2_knee_nm_s2_per_rad2 * np.abs(dq_knee_array) * dq_knee_array
    return _pair_output(hip, knee)


def nonlinear_damping_torque(
    dq_hip: ScalarOrArray,
    dq_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """返回线性加奇对称速度平方项的总阻尼需求力矩。"""

    linear_hip, linear_knee = damping_torque(
        dq_hip,
        dq_knee,
        subject.base_dynamic_subject(),
    )
    quadratic_hip, quadratic_knee = quadratic_damping_torque(
        dq_hip,
        dq_knee,
        subject,
    )
    hip = np.asarray(linear_hip) + np.asarray(quadratic_hip)
    knee = np.asarray(linear_knee) + np.asarray(quadratic_knee)
    return _pair_output(hip, knee)


def structured_residual_torque(
    q_hip: ScalarOrArray,
    q_knee: ScalarOrArray,
    dq_hip: ScalarOrArray,
    dq_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
    *,
    random_seed: int = DEFAULT_RESIDUAL_RANDOM_SEED,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """返回小幅、平滑、确定且可复现的结构残余力矩。

    随机种子只用于固定两个相位；没有逐样本白噪声，也没有内部可变状态。
    对同一状态、受试者和种子，该纯函数总是返回相同结果。
    """

    if isinstance(random_seed, bool) or not isinstance(
        random_seed,
        (int, np.integer),
    ):
        raise TypeError("random_seed must be an integer.")
    qh, qk, dqh, dqk = _broadcast(q_hip, q_knee, dq_hip, dq_knee)
    if subject.residual_torque_scale_nm == 0.0:
        zeros = np.zeros_like(qh)
        return _pair_output(zeros, zeros.copy())

    rng = np.random.default_rng(int(random_seed))
    phase_hip, phase_knee = rng.uniform(-np.pi, np.pi, size=2)
    frequency = subject.residual_torque_frequency
    hip_argument = frequency * (1.10 * qh - 0.35 * qk + 0.15 * dqh)
    knee_argument = frequency * (0.95 * qk + 0.25 * qh + 0.12 * dqk)
    hip = subject.residual_torque_scale_nm * np.sin(hip_argument + phase_hip)
    knee = 0.8 * subject.residual_torque_scale_nm * np.sin(
        knee_argument + phase_knee,
    )
    return _pair_output(hip, knee)


@dataclass(frozen=True)
class MismatchDynamicsResult:
    """复杂生成模型的全部关节力矩分项，单位均为 N·m。"""

    tau_inertia_hip_nm: ScalarOrArray
    tau_inertia_knee_nm: ScalarOrArray
    tau_coriolis_hip_nm: ScalarOrArray
    tau_coriolis_knee_nm: ScalarOrArray
    tau_gravity_hip_nm: ScalarOrArray
    tau_gravity_knee_nm: ScalarOrArray
    tau_linear_damping_hip_nm: ScalarOrArray
    tau_linear_damping_knee_nm: ScalarOrArray
    tau_linear_stiffness_hip_nm: ScalarOrArray
    tau_linear_stiffness_knee_nm: ScalarOrArray
    tau_nonlinear_stiffness_hip_nm: ScalarOrArray
    tau_nonlinear_stiffness_knee_nm: ScalarOrArray
    tau_coupling_hip_nm: ScalarOrArray
    tau_coupling_knee_nm: ScalarOrArray
    tau_nonlinear_damping_hip_nm: ScalarOrArray
    tau_nonlinear_damping_knee_nm: ScalarOrArray
    tau_residual_hip_nm: ScalarOrArray
    tau_residual_knee_nm: ScalarOrArray
    tau_damping_hip_nm: ScalarOrArray
    tau_damping_knee_nm: ScalarOrArray
    tau_stiffness_hip_nm: ScalarOrArray
    tau_stiffness_knee_nm: ScalarOrArray
    tau_mismatch_hip_nm: ScalarOrArray
    tau_mismatch_knee_nm: ScalarOrArray
    tau_base_total_hip_nm: ScalarOrArray
    tau_base_total_knee_nm: ScalarOrArray
    tau_total_hip_nm: ScalarOrArray
    tau_total_knee_nm: ScalarOrArray


def mismatch_inverse_dynamics(
    q_hip: ScalarOrArray,
    q_knee: ScalarOrArray,
    dq_hip: ScalarOrArray,
    dq_knee: ScalarOrArray,
    ddq_hip: ScalarOrArray,
    ddq_knee: ScalarOrArray,
    subject: MismatchVirtualSubject,
    L1: float,
    *,
    residual_random_seed: int = DEFAULT_RESIDUAL_RANDOM_SEED,
) -> MismatchDynamicsResult:
    """计算复杂生成模型力矩，同时保留每个失配分项。

    基础项由现有 :func:`full_dynamics.inverse_dynamics` 唯一计算。附加参数
    全为零时，``tau_total`` 与基础模型逐位一致。
    """

    if not isinstance(subject, MismatchVirtualSubject):
        raise TypeError("subject must be a MismatchVirtualSubject.")
    # 先广播和验证所有输入，使附加项与基础结果具有完全相同的形状。
    qh, qk, dqh, dqk, ddqh, ddqk = _broadcast(
        q_hip,
        q_knee,
        dq_hip,
        dq_knee,
        ddq_hip,
        ddq_knee,
    )
    base = inverse_dynamics(
        qh,
        qk,
        dqh,
        dqk,
        ddqh,
        ddqk,
        subject.base_dynamic_subject(),
        L1,
    )
    cubic_hip, cubic_knee = cubic_stiffness_torque(qh, qk, subject)
    coupling_hip, coupling_knee = coupling_stiffness_torque(qh, qk, subject)
    quadratic_hip, quadratic_knee = quadratic_damping_torque(dqh, dqk, subject)
    residual_hip, residual_knee = structured_residual_torque(
        qh,
        qk,
        dqh,
        dqk,
        subject,
        random_seed=residual_random_seed,
    )

    def array(value: ScalarOrArray) -> np.ndarray:
        return np.asarray(value, dtype=float)

    cubic_hip_array, cubic_knee_array = array(cubic_hip), array(cubic_knee)
    coupling_hip_array, coupling_knee_array = (
        array(coupling_hip),
        array(coupling_knee),
    )
    quadratic_hip_array, quadratic_knee_array = (
        array(quadratic_hip),
        array(quadratic_knee),
    )
    residual_hip_array, residual_knee_array = (
        array(residual_hip),
        array(residual_knee),
    )
    linear_damping_hip = array(base.tau_damping_hip_nm)
    linear_damping_knee = array(base.tau_damping_knee_nm)
    linear_stiffness_hip = array(base.tau_stiffness_hip_nm)
    linear_stiffness_knee = array(base.tau_stiffness_knee_nm)
    damping_hip = linear_damping_hip + quadratic_hip_array
    damping_knee = linear_damping_knee + quadratic_knee_array
    stiffness_hip = (
        linear_stiffness_hip + cubic_hip_array + coupling_hip_array
    )
    stiffness_knee = (
        linear_stiffness_knee + cubic_knee_array + coupling_knee_array
    )
    mismatch_hip = (
        cubic_hip_array
        + coupling_hip_array
        + quadratic_hip_array
        + residual_hip_array
    )
    mismatch_knee = (
        cubic_knee_array
        + coupling_knee_array
        + quadratic_knee_array
        + residual_knee_array
    )
    total_hip = array(base.tau_total_hip_nm) + mismatch_hip
    total_knee = array(base.tau_total_knee_nm) + mismatch_knee

    result_values = {
        "tau_inertia_hip_nm": array(base.tau_inertia_hip_nm),
        "tau_inertia_knee_nm": array(base.tau_inertia_knee_nm),
        "tau_coriolis_hip_nm": array(base.tau_coriolis_hip_nm),
        "tau_coriolis_knee_nm": array(base.tau_coriolis_knee_nm),
        "tau_gravity_hip_nm": array(base.tau_gravity_hip_nm),
        "tau_gravity_knee_nm": array(base.tau_gravity_knee_nm),
        "tau_linear_damping_hip_nm": linear_damping_hip,
        "tau_linear_damping_knee_nm": linear_damping_knee,
        "tau_linear_stiffness_hip_nm": linear_stiffness_hip,
        "tau_linear_stiffness_knee_nm": linear_stiffness_knee,
        "tau_nonlinear_stiffness_hip_nm": cubic_hip_array,
        "tau_nonlinear_stiffness_knee_nm": cubic_knee_array,
        "tau_coupling_hip_nm": coupling_hip_array,
        "tau_coupling_knee_nm": coupling_knee_array,
        "tau_nonlinear_damping_hip_nm": quadratic_hip_array,
        "tau_nonlinear_damping_knee_nm": quadratic_knee_array,
        "tau_residual_hip_nm": residual_hip_array,
        "tau_residual_knee_nm": residual_knee_array,
        "tau_damping_hip_nm": damping_hip,
        "tau_damping_knee_nm": damping_knee,
        "tau_stiffness_hip_nm": stiffness_hip,
        "tau_stiffness_knee_nm": stiffness_knee,
        "tau_mismatch_hip_nm": mismatch_hip,
        "tau_mismatch_knee_nm": mismatch_knee,
        "tau_base_total_hip_nm": array(base.tau_total_hip_nm),
        "tau_base_total_knee_nm": array(base.tau_total_knee_nm),
        "tau_total_hip_nm": total_hip,
        "tau_total_knee_nm": total_knee,
    }
    if not all(np.isfinite(value).all() for value in result_values.values()):
        raise FloatingPointError("mismatch dynamics produced a non-finite torque.")
    return MismatchDynamicsResult(
        **{name: _output(value) for name, value in result_values.items()},
    )


# 兼容更偏描述性的调用名称。
inverse_dynamics_with_mismatch = mismatch_inverse_dynamics
