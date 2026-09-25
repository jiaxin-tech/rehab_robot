"""Mechanically interpretable subject--trajectory interaction primitives.

This module is deliberately separate from the analytical resistance stress
field.  It consumes an executed trajectory (q, dq, branch phase) and a
declared subject mechanical profile, then returns an external resistance
torque in Nm.  It does not call MyoLeg, invent muscle activation, or claim
physiological validity.  Its purpose is to provide a unit-tested, auditable
adapter for future native/real-measurement pilots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..force_mapping import endpoint_force_from_joint_torque


@dataclass(frozen=True)
class PhysicalInteractionProfile:
    """Subject mechanical parameters with explicit SI units."""

    profile_id: str
    stiffness_nm_per_rad: tuple[float, float]
    damping_nm_s_per_rad: tuple[float, float]
    reference_q_rad: tuple[float, float] = (0.0, 0.0)
    resistance_scale: float = 1.0

    def __post_init__(self) -> None:
        if any(len(values) != 2 for values in (
            self.stiffness_nm_per_rad,
            self.damping_nm_s_per_rad,
            self.reference_q_rad,
        )):
            raise ValueError("PHYSICAL_INTERACTION_PROFILE_REQUIRES_TWO_JOINTS")
        values = (*self.stiffness_nm_per_rad, *self.damping_nm_s_per_rad,
                  *self.reference_q_rad, self.resistance_scale)
        if not self.profile_id or not np.isfinite(values).all():
            raise ValueError("INVALID_PHYSICAL_INTERACTION_PROFILE")
        if any(value < 0.0 for value in (*self.stiffness_nm_per_rad, *self.damping_nm_s_per_rad)):
            raise ValueError("MECHANICAL_COEFFICIENTS_MUST_BE_NONNEGATIVE")
        if self.resistance_scale <= 0.0:
            raise ValueError("RESISTANCE_SCALE_MUST_BE_POSITIVE")


@dataclass(frozen=True)
class PhysicalInteractionConfig:
    """Candidate-independent numerical limits and onset shape."""

    onset_width_phase: float = 0.10
    max_resistance_torque_nm: float = 20.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.onset_width_phase) or not 0.0 < self.onset_width_phase <= 0.5:
            raise ValueError("INVALID_INTERACTION_ONSET_WIDTH")
        if not np.isfinite(self.max_resistance_torque_nm) or self.max_resistance_torque_nm <= 0.0:
            raise ValueError("INVALID_INTERACTION_TORQUE_LIMIT")


@dataclass(frozen=True)
class PhysicalInteractionResult:
    """External torque and auditable diagnostics for one candidate."""

    external_resistance_tau_nm: np.ndarray
    onset_gate: np.ndarray
    elastic_component_nm: np.ndarray
    viscous_component_nm: np.ndarray
    valid: bool
    invalid_reason: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.external_resistance_tau_nm.shape)


def _validate_candidate(features: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(features, dtype=float).reshape(-1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("INVALID_PHYSICAL_INTERACTION_CANDIDATE")
    duration, timing, hip_share = map(float, values)
    if duration <= 0.0 or not 0.0 <= timing <= 1.0 or not 0.0 <= hip_share <= 1.0:
        raise ValueError("PHYSICAL_INTERACTION_CANDIDATE_OUT_OF_RANGE")
    return duration, timing, hip_share


def compute_physical_resistance(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    branch_phase: np.ndarray,
    features: tuple[float, float, float] | np.ndarray,
    profile: PhysicalInteractionProfile,
    config: PhysicalInteractionConfig | None = None,
) -> PhysicalInteractionResult:
    """Compute an external resistance torque for one executed trajectory.

    ``features`` are ``(duration_scale, assistance_timing, hip_share)``.
    The resistance is a smooth onset-gated combination of angle-dependent
    stiffness and velocity-dependent damping.  It opposes measured motion and
    is returned as an *external* torque; a required-drive trace should use
    ``tau_net = tau_native - external_resistance_tau_nm``.
    """

    duration_scale, assistance_timing, hip_share = _validate_candidate(features)
    cfg = PhysicalInteractionConfig() if config is None else config
    q, dq, phase = np.asarray(q_rad, dtype=float), np.asarray(dq_rad_s, dtype=float), np.asarray(branch_phase, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or dq.shape != q.shape or phase.shape != (len(q),):
        raise ValueError("PHYSICAL_INTERACTION_TRAJECTORY_SHAPE_MISMATCH")
    if not np.isfinite(q).all() or not np.isfinite(dq).all() or not np.isfinite(phase).all():
        raise ValueError("NONFINITE_PHYSICAL_INTERACTION_TRAJECTORY")
    if np.any(phase < 0.0) or np.any(phase > 1.0):
        raise ValueError("BRANCH_PHASE_MUST_LIE_IN_ZERO_ONE")

    # Logistic onset is smooth and bounded.  The duration factor makes the
    # velocity term respond to the candidate duration even when the supplied
    # trajectory was sampled on a common normalized phase grid.
    normalized_onset = (phase - assistance_timing) / cfg.onset_width_phase
    gate = 1.0 / (1.0 + np.exp(-np.clip(normalized_onset, -60.0, 60.0)))
    q_error = q - np.asarray(profile.reference_q_rad, dtype=float)
    elastic = np.abs(q_error) * np.asarray(profile.stiffness_nm_per_rad, dtype=float)
    viscous = np.abs(dq) * np.asarray(profile.damping_nm_s_per_rad, dtype=float) / duration_scale
    joint_weight = np.asarray((hip_share, 1.0 - hip_share), dtype=float)
    magnitude = profile.resistance_scale * gate[:, None] * (elastic + viscous) * joint_weight[None, :]

    motion_sign = np.sign(dq)
    fallback_sign = np.sign(q_error)
    direction = np.where(motion_sign == 0.0, fallback_sign, motion_sign)
    direction = np.where(direction == 0.0, 1.0, direction)
    external = -direction * magnitude
    max_abs = np.max(np.abs(external), axis=1)
    valid = bool(np.isfinite(external).all() and np.all(max_abs <= cfg.max_resistance_torque_nm))
    reason = "" if valid else "PHYSICAL_INTERACTION_TORQUE_LIMIT_OR_NONFINITE"
    return PhysicalInteractionResult(external, gate, elastic, viscous, valid, reason)


def net_required_torque(tau_native_nm: np.ndarray, result: PhysicalInteractionResult) -> np.ndarray:
    """Combine prescribed native drive torque and external resistance."""

    native = np.asarray(tau_native_nm, dtype=float)
    if native.shape != result.external_resistance_tau_nm.shape or not np.isfinite(native).all():
        raise ValueError("NATIVE_TORQUE_SHAPE_OR_FINITE_MISMATCH")
    if not result.valid:
        raise ValueError("INVALID_PHYSICAL_INTERACTION_RESULT")
    return native - result.external_resistance_tau_nm


def endpoint_force_from_interaction(
    q_rad: np.ndarray,
    result: PhysicalInteractionResult,
    *,
    L1_m: float,
    L2_m: float,
    force_limit_n: float = 500.0,
) -> dict[str, np.ndarray]:
    """Map external joint resistance to cuff-plane force through ``J.T @ F``."""

    q = np.asarray(q_rad, dtype=float)
    tau = result.external_resistance_tau_nm
    if not result.valid:
        raise ValueError("INVALID_PHYSICAL_INTERACTION_RESULT")
    if q.shape != tau.shape or q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("INTERACTION_FORCE_MAPPING_SHAPE_MISMATCH")
    mapped = endpoint_force_from_joint_torque(
        q[:, 0], q[:, 1], tau[:, 0], tau[:, 1], L1_m, L2_m, force_limit_n=force_limit_n,
    )
    valid = np.asarray(mapped.force_mapping_valid, dtype=bool)
    if valid.shape != (len(q),):
        valid = np.broadcast_to(valid, (len(q),))
    return {
        "fx_n": np.asarray(mapped.fx_robot_on_leg_n, dtype=float),
        "fz_n": np.asarray(mapped.fz_robot_on_leg_n, dtype=float),
        "magnitude_n": np.asarray(mapped.force_magnitude_n, dtype=float),
        "valid": valid,
    }


__all__ = [
    "PhysicalInteractionConfig",
    "PhysicalInteractionProfile",
    "PhysicalInteractionResult",
    "compute_physical_resistance",
    "endpoint_force_from_interaction",
    "net_required_torque",
]
