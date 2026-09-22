"""Deterministic inverse-dynamics replay for frozen V3 trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .model import FrozenBenchmarkDefinition, MechanicalLegDefinition


@dataclass(frozen=True)
class TrajectoryReplayResult:
    valid: bool
    invalid_reason: str | None
    endpoint_value_nm: float | None
    hip_rms_torque_nm: float | None
    knee_rms_torque_nm: float | None
    hip_peak_abs_torque_nm: float | None
    knee_peak_abs_torque_nm: float | None
    tracking_rms_rad: float
    tracking_max_abs_rad: float
    sample_count: int
    tau_hip_nm: np.ndarray
    tau_knee_nm: np.ndarray

    def as_metrics(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "endpoint_value_nm": self.endpoint_value_nm,
            "hip_rms_torque_nm": self.hip_rms_torque_nm,
            "knee_rms_torque_nm": self.knee_rms_torque_nm,
            "hip_peak_abs_torque_nm": self.hip_peak_abs_torque_nm,
            "knee_peak_abs_torque_nm": self.knee_peak_abs_torque_nm,
            "tracking_rms_rad": self.tracking_rms_rad,
            "tracking_max_abs_rad": self.tracking_max_abs_rad,
            "sample_count": self.sample_count,
        }


def _time_weighted_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    return float(np.sqrt(np.trapezoid(values**2, time_s) / duration))


def custom_passive_torque_project_coordinates(
    leg: MechanicalLegDefinition,
    q_hip: float,
    q_knee: float,
) -> np.ndarray:
    """Return structural nonlinear/coupling passive torque in project coordinates."""

    hip_displacement = q_hip - leg.hip_neutral_rad
    knee_displacement = q_knee - leg.knee_neutral_rad
    hip = -leg.hip_cubic_stiffness_nm_per_rad3 * hip_displacement**3
    knee = -leg.knee_cubic_stiffness_nm_per_rad3 * knee_displacement**3
    coupling_displacement = (
        hip_displacement - leg.coupling_ratio * knee_displacement
    )
    hip -= leg.coupling_stiffness_nm_per_rad * coupling_displacement
    knee += (
        leg.coupling_stiffness_nm_per_rad
        * leg.coupling_ratio
        * coupling_displacement
    )
    return np.asarray([hip, knee], dtype=float)


def replay_trajectory(
    *,
    model: mujoco.MjModel,
    definition: FrozenBenchmarkDefinition,
    leg: MechanicalLegDefinition,
    time_s: np.ndarray,
    q_project: np.ndarray,
    dq_project: np.ndarray,
    ddq_project: np.ndarray,
) -> TrajectoryReplayResult:
    """Replay q/dq/ddq exactly and compute required project-coordinate torques."""

    time = np.asarray(time_s, dtype=float)
    q = np.asarray(q_project, dtype=float)
    dq = np.asarray(dq_project, dtype=float)
    ddq = np.asarray(ddq_project, dtype=float)
    if time.ndim != 1 or q.shape != (len(time), 2):
        raise ValueError("q_project must be shaped [samples, 2]")
    if dq.shape != q.shape or ddq.shape != q.shape:
        raise ValueError("q, dq, and ddq shapes must match")
    if len(time) < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be strictly increasing with at least 3 samples")
    if not all(np.isfinite(value).all() for value in (time, q, dq, ddq)):
        raise ValueError("trajectory arrays must be finite")

    outside = max(
        float(leg.hip_min_rad - np.min(q[:, 0])),
        float(np.max(q[:, 0]) - leg.hip_max_rad),
        float(leg.knee_min_rad - np.min(q[:, 1])),
        float(np.max(q[:, 1]) - leg.knee_max_rad),
        0.0,
    )
    if outside > 2.0e-5:
        return TrajectoryReplayResult(
            valid=False,
            invalid_reason="TRAJECTORY_OUTSIDE_FROZEN_ROM_NO_CLIPPING",
            endpoint_value_nm=None,
            hip_rms_torque_nm=None,
            knee_rms_torque_nm=None,
            hip_peak_abs_torque_nm=None,
            knee_peak_abs_torque_nm=None,
            tracking_rms_rad=math.inf,
            tracking_max_abs_rad=math.inf,
            sample_count=len(time),
            tau_hip_nm=np.asarray([], dtype=float),
            tau_knee_nm=np.asarray([], dtype=float),
        )

    data = mujoco.MjData(model)
    tau_project = np.empty_like(q)
    tracking_error = np.empty_like(q)
    for index in range(len(time)):
        # MuJoCo's second hinge coordinate is the negative project knee angle,
        # preserving theta_shank = q_hip - q_knee.
        data.qpos[:] = (q[index, 0], -q[index, 1])
        data.qvel[:] = (dq[index, 0], -dq[index, 1])
        data.qacc[:] = (ddq[index, 0], -ddq[index, 1])
        passive_project = custom_passive_torque_project_coordinates(
            leg, q[index, 0], q[index, 1]
        )
        data.qfrc_applied[:] = (passive_project[0], -passive_project[1])
        mujoco.mj_inverse(model, data)
        required_mujoco = np.asarray(data.qfrc_inverse - data.qfrc_applied)
        tau_project[index] = (required_mujoco[0], -required_mujoco[1])
        tracking_error[index] = (
            data.qpos[0] - q[index, 0],
            -data.qpos[1] - q[index, 1],
        )

    tracking_rms = float(np.sqrt(np.mean(tracking_error**2)))
    tracking_max = float(np.max(np.abs(tracking_error)))
    finite_torque = bool(np.isfinite(tau_project).all())
    tracking_valid = tracking_rms <= definition.tracking_failure_rms_rad
    if not finite_torque or not tracking_valid:
        reason = (
            "NONFINITE_INVERSE_DYNAMICS_TORQUE"
            if not finite_torque
            else "TRACKING_FAILURE"
        )
        return TrajectoryReplayResult(
            valid=False,
            invalid_reason=reason,
            endpoint_value_nm=None,
            hip_rms_torque_nm=None,
            knee_rms_torque_nm=None,
            hip_peak_abs_torque_nm=None,
            knee_peak_abs_torque_nm=None,
            tracking_rms_rad=tracking_rms,
            tracking_max_abs_rad=tracking_max,
            sample_count=len(time),
            tau_hip_nm=tau_project[:, 0],
            tau_knee_nm=tau_project[:, 1],
        )

    hip_rms = _time_weighted_rms(tau_project[:, 0], time)
    knee_rms = _time_weighted_rms(tau_project[:, 1], time)
    endpoint = _time_weighted_rms(
        np.sqrt(tau_project[:, 0] ** 2 + tau_project[:, 1] ** 2), time
    )
    return TrajectoryReplayResult(
        valid=True,
        invalid_reason=None,
        endpoint_value_nm=endpoint,
        hip_rms_torque_nm=hip_rms,
        knee_rms_torque_nm=knee_rms,
        hip_peak_abs_torque_nm=float(np.max(np.abs(tau_project[:, 0]))),
        knee_peak_abs_torque_nm=float(np.max(np.abs(tau_project[:, 1]))),
        tracking_rms_rad=tracking_rms,
        tracking_max_abs_rad=tracking_max,
        sample_count=len(time),
        tau_hip_nm=tau_project[:, 0],
        tau_knee_nm=tau_project[:, 1],
    )


__all__ = [
    "TrajectoryReplayResult",
    "custom_passive_torque_project_coordinates",
    "replay_trajectory",
]
