"""Declared open-loop assistance on prescribed-state MyoLeg required torque.

``CONTROLLED_ACTUATION_V3`` keeps the trajectory shape/ROM fixed and varies
duration, assistance timing within each flexion/extension branch, and hip
allocation. The adapter computes ``tau_net = tau_native - tau_assistance``.
This algebraic load-sharing proxy is not a forward actuator simulation,
muscle recruitment model, cuff measurement, or patient physiology validation.

All subjects receive the same movement-aligned waveform. Its direction comes
only from the public reference kinematics, never subject torque or an oracle.
The sampled peak of ``abs(hip) + abs(knee)`` is fixed at 2 Nm by default.
This fixes peak assistance, not impulse, work, or squared torque energy;
those quantities may change with duration/allocation and are recorded.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np

from .mechanism_candidates import make_mechanism_trajectory
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain


DURATION_SCALES = (0.9, 1.0, 1.1)
ASSISTANCE_TIMINGS = (0.25, 0.50, 0.75)
HIP_SHARES = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class ControlledActuationConfig:
    """One common, frozen assistance budget for an entire comparison."""

    assist_peak_nm: float = 2.0
    pulse_half_width_phase: float = 0.20

    def __post_init__(self):
        if not np.isfinite(self.assist_peak_nm) or self.assist_peak_nm <= 0:
            raise ValueError("INVALID_FIXED_ASSISTANCE_PEAK")
        if not np.isfinite(self.pulse_half_width_phase) or not 0 < self.pulse_half_width_phase <= 0.25:
            raise ValueError("INVALID_ASSISTANCE_PULSE_WIDTH")

    def as_dict(self) -> dict[str, object]:
        return {
            "assist_peak_nm": float(self.assist_peak_nm),
            "pulse_half_width_phase": float(self.pulse_half_width_phase),
            "normalization": "fixed sampled peak of abs(tau_hip) + abs(tau_knee), separately in each branch",
            "pulse": "C2 compact polynomial (1 - u^2)^3 for abs(u) < 1; sampled peak normalized to one",
            "timing_scope": "relative elapsed time within each flexion/extension branch; not gait stance",
            "direction": "sign of public reference end-minus-start joint angle within each branch",
            "response_equation": "tau_net = tau_native - tau_assistance",
            "scope": "CONTROLLED_ACTUATION; algebraic prescribed-state load-sharing proxy",
            "fixed_budget_exclusions": "impulse, work, and squared torque energy are not fixed",
        }


@dataclass(frozen=True)
class ControlledActuationPoint:
    candidate_id: str
    candidate_index: int
    duration_scale: float
    assistance_timing: float
    hip_share: float
    trajectory: Any
    time_s: np.ndarray
    kinematic: dict[str, float]

    @property
    def features(self) -> tuple[float, float, float]:
        return (self.duration_scale, self.assistance_timing, self.hip_share)

    @property
    def beta(self) -> tuple[float, float]:
        """Legacy scalar-endpoint slots; the V3 factors live in ``features``."""
        return (0.0, 0.0)


class ControlledActuationDomain:
    """Fixed 27-point controlled domain, independent of response values.

    ``reference`` is the middle *assisted* candidate. The pilot must evaluate
    the native backend at its trajectory separately for the unassisted
    normalization baseline; this assisted reference is not a no-assist arm.
    """

    family = "CONTROLLED_ACTUATION_V3"

    def __init__(self, source: Any | None = None):
        source = native_domain() if source is None else source
        self.profile = source.profile
        self.subject_reference = source.subject_reference
        self.points: list[ControlledActuationPoint] = []
        self.rejections: list[dict[str, object]] = []
        combinations = list(product(DURATION_SCALES, ASSISTANCE_TIMINGS, HIP_SHARES))
        combinations.sort(key=lambda p: (p != (1.0, 0.5, 0.5), p))
        for parameters in combinations:
            trajectory, kinematic = make_mechanism_trajectory(
                self.subject_reference, duration_scale=parameters[0],
                coordination_lag=0.0, coordination_amplitude=0.0,
            )
            index = len(self.points)
            self.points.append(ControlledActuationPoint(
                f"{self.family}:{index}", index, *parameters,
                SimpleNamespace(q=trajectory["q"], dq=trajectory["dq"], ddq=trajectory["ddq"]),
                trajectory["time_s"], kinematic,
            ))
        self.lookup = {point.candidate_id: point for point in self.points}
        self.reference = self.points[0]
        self.features = np.asarray([point.features for point in self.points])

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def by_id(self, candidate_id: str) -> ControlledActuationPoint:
        return self.lookup[candidate_id]


def assistance_waveform(point: ControlledActuationPoint, phases: np.ndarray,
                        config: ControlledActuationConfig) -> dict[str, np.ndarray]:
    """Produce a smooth, common open-loop pulse using only candidate kinematics.

    Branch phase spans the first and last samples carrying that branch label.
    Compact support keeps assistance zero near branch boundaries, including
    the one-sample gap between the flexion and extension label ranges.
    """
    time_s = np.asarray(point.time_s, dtype=float)
    q = np.asarray(point.trajectory.q, dtype=float)
    phases = np.asarray(phases)
    if (time_s.ndim != 1 or len(time_s) < 6 or q.shape != (len(time_s), 2)
            or phases.shape != time_s.shape or not np.isfinite(time_s).all()
            or not np.isfinite(q).all() or np.any(np.diff(time_s) <= 0)):
        raise ValueError("INVALID_ASSISTANCE_KINEMATICS")
    if set(phases.tolist()) != {"flexion", "extension"}:
        raise ValueError("ASSISTANCE_REQUIRES_FLEXION_EXTENSION_PHASES")
    if point.assistance_timing not in ASSISTANCE_TIMINGS or point.hip_share not in HIP_SHARES:
        raise ValueError("UNDECLARED_ASSISTANCE_CANDIDATE")
    pulse = np.zeros(len(time_s))
    branch_phase = np.zeros(len(time_s))
    direction = np.zeros_like(q)
    for branch in ("flexion", "extension"):
        indices = np.flatnonzero(phases == branch)
        if len(indices) < 3 or np.any(np.diff(indices) != 1):
            raise ValueError("INVALID_ASSISTANCE_BRANCH_SAMPLING")
        local_phase = ((time_s[indices] - time_s[indices[0]])
                       / (time_s[indices[-1]] - time_s[indices[0]]))
        u = (local_phase - point.assistance_timing) / config.pulse_half_width_phase
        local_pulse = np.maximum(1.0 - u**2, 0.0)**3
        peak = float(np.max(local_pulse))
        if peak <= 0:
            raise ValueError("ASSISTANCE_PULSE_UNRESOLVED_BY_SAMPLING")
        local_direction = np.sign(q[indices[-1]] - q[indices[0]])
        if np.any(local_direction == 0):
            raise ValueError("ASSISTANCE_DIRECTION_UNDEFINED")
        pulse[indices] = local_pulse / peak
        branch_phase[indices] = local_phase
        direction[indices] = local_direction
    allocation = np.asarray([point.hip_share, 1.0 - point.hip_share])
    torque = config.assist_peak_nm * pulse[:, None] * direction * allocation
    return {"branch_phase": branch_phase, "pulse": pulse, "assistance_tau_nm": torque}


class ControlledActuationBackend:
    """Wrap one private native backend and reveal only requested net traces.

    No subject-specific amplification, force clipping, or truth-derived pulse
    direction is applied. A movement-aligned pulse may increase required net
    torque; the evaluator must report that outcome and its load constraints.
    """

    def __init__(self, native_backend: Any, domain: ControlledActuationDomain,
                 config: ControlledActuationConfig | None = None):
        self.native_backend = native_backend
        self.domain = domain
        self.config = ControlledActuationConfig() if config is None else config
        self._traces: dict[str, dict[str, object]] = {}
        self.provenance = {
            "native": deepcopy(getattr(native_backend, "provenance", {})),
            "controlled_actuation": self.config.as_dict(),
            "candidate_family": domain.family,
        }

    def requested(self, point: ControlledActuationPoint) -> np.ndarray:
        if self.domain.by_id(point.candidate_id) is not point:
            raise ValueError("CONTROLLED_CANDIDATE_IDENTITY_MISMATCH")
        if point.candidate_id not in self._traces:
            wave = assistance_waveform(point, self.domain.subject_reference.phases, self.config)
            native_tau = np.asarray(self.native_backend.requested(point), dtype=float)
            if native_tau.shape != (len(point.time_s), 2) or not np.isfinite(native_tau).all():
                raise ValueError("INVALID_NATIVE_ACTUATION_INPUT")
            assistance_tau = wave["assistance_tau_nm"]
            net_tau = native_tau - assistance_tau
            peak = float(np.max(np.sum(np.abs(assistance_tau), axis=1)))
            violations = []
            if not np.isclose(peak, self.config.assist_peak_nm, rtol=0, atol=1e-12):
                violations.append("FIXED_ASSISTANCE_PEAK_VIOLATION")
            if not np.isfinite(net_tau).all():
                violations.append("NONFINITE_NET_TORQUE")
            diagnostics = {
                "assist_peak_l1_nm": peak,
                "joint_assist_peak_nm": np.max(np.abs(assistance_tau), axis=0).tolist(),
                "joint_abs_impulse_nm_s": np.trapezoid(np.abs(assistance_tau), point.time_s, axis=0).tolist(),
                "joint_signed_work_j": np.trapezoid(assistance_tau * point.trajectory.dq, point.time_s, axis=0).tolist(),
                "joint_squared_torque_integral_nm2_s": np.trapezoid(assistance_tau**2, point.time_s, axis=0).tolist(),
                "native_peak_nm": np.max(np.abs(native_tau), axis=0).tolist(),
                "net_peak_nm": np.max(np.abs(net_tau), axis=0).tolist(),
                "actuation_valid": not violations,
                "constraint_violations": violations,
            }
            if violations:
                raise ValueError(";".join(violations))
            self._traces[point.candidate_id] = {
                "time_s": np.asarray(point.time_s).copy(),
                **wave,
                "native_tau_nm": native_tau.copy(),
                "net_tau_nm": net_tau,
                "diagnostics": diagnostics,
            }
        return self._traces[point.candidate_id]["net_tau_nm"].copy()

    def trace_for(self, point: ControlledActuationPoint) -> dict[str, object]:
        """Evaluator audit of an already requested trace; never executes a trial."""
        if self.domain.by_id(point.candidate_id) is not point:
            raise ValueError("CONTROLLED_CANDIDATE_IDENTITY_MISMATCH")
        if point.candidate_id not in self._traces:
            raise ValueError("ACTUATION_TRACE_NOT_REQUESTED")
        return deepcopy(self._traces[point.candidate_id])


__all__ = [
    "DURATION_SCALES", "ASSISTANCE_TIMINGS", "HIP_SHARES",
    "ControlledActuationConfig", "ControlledActuationPoint", "ControlledActuationDomain",
    "assistance_waveform", "ControlledActuationBackend",
]
