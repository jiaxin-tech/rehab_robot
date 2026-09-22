"""Pure E0/E2 mechanics; no MuJoCo, robot, or cohort normalization dependency."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EndpointIdentity:
    endpoint_name: str
    unit: str

    def require(self, name: str, unit: str) -> None:
        if (name, unit) != (self.endpoint_name, self.unit):
            raise ValueError(f"ENDPOINT_IDENTITY_MISMATCH: expected {self}, got {(name, unit)}")


E0 = EndpointIdentity("E0_FULL_CYCLE_DUAL_JOINT_RMS", "N_m")
E2 = EndpointIdentity("E2_BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS", "dimensionless")


def time_rms(values, time_s) -> float:
    values, time = np.asarray(values, dtype=float), np.asarray(time_s, dtype=float)
    if (time.ndim != 1 or values.shape != time.shape or len(time) < 2
            or not np.isfinite(time).all() or not np.isfinite(values).all()
            or not np.all(np.diff(time) > 0)):
        raise ValueError("INVALID_ENDPOINT_TIME_SERIES")
    return float(np.sqrt(np.trapezoid(values ** 2, time) / (time[-1] - time[0])))


def full_cycle_dual_joint_rms(tau_hip, tau_knee, time_s) -> float:
    hip, knee = np.asarray(tau_hip, dtype=float), np.asarray(tau_knee, dtype=float)
    if hip.shape != knee.shape:
        raise ValueError("ENDPOINT_TORQUE_SHAPE_MISMATCH")
    return time_rms(np.sqrt(hip ** 2 + knee ** 2), time_s)


def branch_rms_components(tau_hip, tau_knee, time_s, branches) -> tuple[float, ...]:
    """Same trapezoidal branch RMS as frozen endpoint_design, in Hf/He/Kf/Ke order."""
    hip, knee, time = (np.asarray(x, dtype=float) for x in (tau_hip, tau_knee, time_s))
    labels = np.asarray(branches)
    if hip.shape != time.shape or knee.shape != time.shape or labels.shape != time.shape:
        raise ValueError("ENDPOINT_BRANCH_SHAPE_MISMATCH")
    time_rms(hip, time)  # Validate global time and all samples as well as branches.
    time_rms(knee, time)
    if not np.isin(labels, ("flexion", "extension")).all():
        raise ValueError("EXPLICIT_FLEXION_EXTENSION_LABELS_REQUIRED")
    return tuple(time_rms(torque[labels == branch], time[labels == branch])
                 for torque in (hip, knee) for branch in ("flexion", "extension"))


@dataclass(frozen=True)
class MechanicalReferenceContext:
    """Identity binding; model_state includes theta and known model geometry/template."""
    rom_profile_id: str
    rom_version: int
    rom_fingerprint: str
    reference_version: str
    model_state: tuple[float, ...]


@dataclass(frozen=True)
class BranchRMSReference:
    context: MechanicalReferenceContext
    components: tuple[float, ...]


def branch_balanced_reference_normalized_rms(
    components, reference: BranchRMSReference, *, context: MechanicalReferenceContext,
) -> float:
    if context != reference.context:
        raise ValueError("E2_REFERENCE_CONTEXT_MISMATCH")
    candidate = np.asarray(components, dtype=float)
    denominator = np.asarray(reference.components, dtype=float)
    if (candidate.shape != (4,) or denominator.shape != (4,)
            or not np.isfinite(candidate).all() or not np.isfinite(denominator).all()
            or np.any(candidate < 0) or np.any(denominator <= np.finfo(float).tiny)):
        # tiny is a floating-point underflow guard, never a physical force threshold.
        raise ValueError("E2_INVALID_OR_NUMERICALLY_ZERO_REFERENCE_COMPONENT")
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.max(candidate / denominator))
    if not np.isfinite(value):
        raise ValueError("E2_NONFINITE_NORMALIZATION")
    return value
