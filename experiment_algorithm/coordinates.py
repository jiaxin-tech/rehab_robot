"""Supine coordinate projection and virtual-work tangent from the public replay."""
from typing import Any
import math

import mujoco
import numpy as np

TARGET_HIP = "hip_flexion_r"
TARGET_KNEE = "knee_angle_r"
SOURCE_KNEE_EQUALITY_COUNT = 14
RIGHT_KNEE_EQUALITY_COUNT = 7
JOINT_NAMES = (TARGET_HIP, TARGET_KNEE)


def name(model: mujoco.MjModel, objtype: Any, identifier: int) -> str:
    return mujoco.mj_id2name(model, objtype, int(identifier)) or ""


def joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if value < 0:
        raise RuntimeError(f"missing joint {joint_name}")
    return int(value)


def qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, joint_name)])


def dof_address(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, joint_name)])


def polynomial_derivative(coefficients: np.ndarray, value: float, order: int) -> float:
    total = 0.0
    for power in range(order, 5):
        multiplier = math.factorial(power) / math.factorial(power - order)
        total += multiplier * float(coefficients[power]) * value ** (power - order)
    return float(total)


def project_source_knee_state(
    model: mujoco.MjModel, data: mujoco.MjData, include_acceleration: bool
) -> None:
    for equality_id in range(SOURCE_KNEE_EQUALITY_COUNT):
        first_joint = int(model.eq_obj1id[equality_id])
        second_joint = int(model.eq_obj2id[equality_id])
        first_qadr = int(model.jnt_qposadr[first_joint])
        second_qadr = int(model.jnt_qposadr[second_joint])
        first_dadr = int(model.jnt_dofadr[first_joint])
        second_dadr = int(model.jnt_dofadr[second_joint])
        parent_q = float(data.qpos[second_qadr])
        coefficients = model.eq_data[equality_id]
        first = polynomial_derivative(coefficients, parent_q, 1)
        data.qpos[first_qadr] = polynomial_derivative(coefficients, parent_q, 0)
        data.qvel[first_dadr] = first * data.qvel[second_dadr]
        if include_acceleration:
            second = polynomial_derivative(coefficients, parent_q, 2)
            data.qacc[first_dadr] = (
                first * data.qacc[second_dadr]
                + second * data.qvel[second_dadr] ** 2
            )


def reset_to_target_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_target: np.ndarray,
    dq_target: np.ndarray,
    ddq_target: np.ndarray | None,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qpos[:7] = np.asarray(
        [0.0, 0.0, 1.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    )
    q_addresses = [qpos_address(model, joint) for joint in JOINT_NAMES]
    dof_addresses = [dof_address(model, joint) for joint in JOINT_NAMES]
    data.qpos[q_addresses] = q_target
    data.qvel[dof_addresses] = dq_target
    if ddq_target is not None:
        data.qacc[dof_addresses] = ddq_target
    project_source_knee_state(model, data, ddq_target is not None)
    if model.na:
        data.act[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0


def independent_coordinate_tangent(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Map project hip/knee velocities into the full constrained velocity space."""

    tangent = np.zeros((model.nv, 2), dtype=float)
    tangent[dof_address(model, TARGET_HIP), 0] = 1.0
    tangent[dof_address(model, TARGET_KNEE), 1] = 1.0
    for equality_id in range(RIGHT_KNEE_EQUALITY_COUNT):
        first_joint = int(model.eq_obj1id[equality_id])
        second_joint = int(model.eq_obj2id[equality_id])
        if name(model, mujoco.mjtObj.mjOBJ_JOINT, second_joint) != TARGET_KNEE:
            raise RuntimeError("unexpected right-knee equality parent")
        first_dadr = int(model.jnt_dofadr[first_joint])
        second_qadr = int(model.jnt_qposadr[second_joint])
        tangent[first_dadr, 1] = polynomial_derivative(
            model.eq_data[equality_id], float(data.qpos[second_qadr]), 1
        )
    return tangent
