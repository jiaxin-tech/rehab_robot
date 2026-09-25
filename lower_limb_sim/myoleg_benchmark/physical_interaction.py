"""Mechanically interpretable subject--trajectory interaction primitives.

This module is deliberately separate from the analytical resistance stress
field.  It consumes an executed trajectory (q, dq) and a
declared subject mechanical profile, then returns an external resistance
torque in Nm.  It does not call MyoLeg, invent muscle activation, or claim
physiological validity.  Its purpose is to provide a unit-tested, auditable
adapter for native/real-measurement development pilots. The added spring and
damper are declared external mechanical elements, not a second calculation of
MyoLeg's existing passive tissue torque.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..force_mapping import endpoint_force_from_joint_torque


PHYSICAL_INTERACTION_FORMULA_VERSION = "EXTERNAL_LINEAR_SPRING_DAMPER_V2"


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
    """Candidate-independent numerical limit, not a clinical safety limit."""

    max_resistance_torque_nm: float = 20.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_resistance_torque_nm) or self.max_resistance_torque_nm <= 0.0:
            raise ValueError("INVALID_INTERACTION_TORQUE_LIMIT")


@dataclass(frozen=True)
class PhysicalInteractionResult:
    """External torque and auditable diagnostics for one candidate."""

    external_resistance_tau_nm: np.ndarray
    elastic_component_nm: np.ndarray
    viscous_component_nm: np.ndarray
    potential_energy_j: np.ndarray
    valid: bool
    invalid_reason: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.external_resistance_tau_nm.shape)


def compute_physical_resistance(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    profile: PhysicalInteractionProfile,
    config: PhysicalInteractionConfig | None = None,
) -> PhysicalInteractionResult:
    """Compute an external resistance torque for one executed trajectory.

    ``dq_rad_s`` is the executed velocity in rad/s. Duration must already be
    reflected in that velocity; no additional time scaling is applied here.
    Assistance timing and allocation do not change subject mechanics.

    External torque is ``-scale * (K * (q - q0) + B * dq)``. K has units
    Nm/rad and B has units Nm s/rad. The spring restores position toward q0,
    including when it releases stored energy on the return branch. Only the
    damper is necessarily opposed to motion at every sample. All component
    torques are signed in project-positive hip/knee flexion coordinates.
    """

    cfg = PhysicalInteractionConfig() if config is None else config
    q, dq = np.asarray(q_rad, dtype=float), np.asarray(dq_rad_s, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or not len(q) or dq.shape != q.shape:
        raise ValueError("PHYSICAL_INTERACTION_TRAJECTORY_SHAPE_MISMATCH")
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("NONFINITE_PHYSICAL_INTERACTION_TRAJECTORY")
    with np.errstate(over="ignore", invalid="ignore"):
        q_error = q - np.asarray(profile.reference_q_rad, dtype=float)
        stiffness = profile.resistance_scale * np.asarray(profile.stiffness_nm_per_rad, dtype=float)
        damping = profile.resistance_scale * np.asarray(profile.damping_nm_s_per_rad, dtype=float)
        elastic = -stiffness * q_error
        viscous = -damping * dq
        potential = 0.5 * stiffness * q_error**2
        external = elastic + viscous
    max_abs = np.max(np.abs(external), axis=1)
    valid = bool(all(np.isfinite(value).all() for value in (external, elastic, viscous, potential))
                 and np.all(max_abs <= cfg.max_resistance_torque_nm))
    reason = "" if valid else "PHYSICAL_INTERACTION_TORQUE_LIMIT_OR_NONFINITE"
    return PhysicalInteractionResult(external, elastic, viscous, potential, valid, reason)


def net_required_torque(
    tau_native_nm: np.ndarray,
    result: PhysicalInteractionResult,
    assistance_tau_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Combine drive demand with additional external interaction and assistance.

    Native already includes its own passive and muscle torque. Neither is
    added or subtracted again. External interaction and external assistance
    both reduce the remaining signed drive demand by their signed torque.
    """

    native = np.asarray(tau_native_nm, dtype=float)
    if (native.ndim != 2 or native.shape[1] != 2 or not len(native)
            or native.shape != result.external_resistance_tau_nm.shape or not np.isfinite(native).all()):
        raise ValueError("NATIVE_TORQUE_SHAPE_OR_FINITE_MISMATCH")
    if not result.valid or not np.isfinite(result.external_resistance_tau_nm).all():
        raise ValueError("INVALID_PHYSICAL_INTERACTION_RESULT")
    assistance = np.zeros_like(native) if assistance_tau_nm is None else np.asarray(assistance_tau_nm, dtype=float)
    if assistance.shape != native.shape or not np.isfinite(assistance).all():
        raise ValueError("ASSISTANCE_TORQUE_SHAPE_OR_FINITE_MISMATCH")
    with np.errstate(over="ignore", invalid="ignore"):
        net = native - result.external_resistance_tau_nm - assistance
    if not np.isfinite(net).all():
        raise ValueError("NONFINITE_NET_TORQUE")
    return net


def endpoint_force_from_interaction(
    q_rad: np.ndarray,
    result: PhysicalInteractionResult,
    *,
    L1_m: float,
    L2_m: float,
    force_limit_n: float = 500.0,
) -> dict[str, np.ndarray]:
    """Map interaction torque to equivalent planar force through ``J.T @ F``.

    This is the interaction force equivalent, not the required robot cuff
    force. Mapping net drive torque, and measuring actual cuff force, are
    separate operations. Invalid/singular samples retain invalid reasons.
    """

    q = np.asarray(q_rad, dtype=float)
    tau = result.external_resistance_tau_nm
    if not result.valid or not np.isfinite(tau).all():
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
        "invalid_reason": np.asarray(mapped.invalid_reason),
        "jacobian_condition_number": np.asarray(mapped.jacobian_condition_number, dtype=float),
    }


__all__ = [
    "PHYSICAL_INTERACTION_FORMULA_VERSION",
    "PhysicalInteractionConfig",
    "PhysicalInteractionProfile",
    "PhysicalInteractionResult",
    "compute_physical_resistance",
    "endpoint_force_from_interaction",
    "net_required_torque",
]
