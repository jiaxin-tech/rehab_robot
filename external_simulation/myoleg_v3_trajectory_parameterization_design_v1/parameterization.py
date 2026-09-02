"""Branch-aware two-parameter MyoLeg-V3 coordination generator.

The transformation preserves the frozen hip trajectory and modifies only the
interior progression of the measured knee branch.  It contains no objective,
subject model, optimizer, clipping operation, or robot interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


PARAMETERIZATION_ID = "V3_PARAMETERIZATION_SEMANTICS_V1"
PARAMETER_ORDER = ("beta_flex", "beta_extend")


@dataclass(frozen=True)
class V3Trajectory:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    warped_segment_phase: np.ndarray
    warp_first_derivative: np.ndarray
    warp_second_derivative: np.ndarray
    beta_flex: float
    beta_extend: float


def interior_warp_basis(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return b(s)=64*s^3*(1-s)^3 and its first two derivatives.

    b, b', and b'' are exactly zero where s is 0 or 1.  Therefore
    w(s; beta)=s+beta*b(s) has identity value, first derivative and second
    derivative at both branch boundaries.
    """

    value = np.asarray(s, dtype=float)
    basis = 64.0 * (value**3 - 3.0 * value**4 + 3.0 * value**5 - value**6)
    first = 64.0 * (3.0 * value**2 - 12.0 * value**3 + 15.0 * value**4 - 6.0 * value**5)
    second = 64.0 * (6.0 * value - 36.0 * value**2 + 60.0 * value**3 - 30.0 * value**4)
    return basis, first, second


def branch_warp(s: np.ndarray, beta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis, first_basis, second_basis = interior_warp_basis(s)
    coefficient = np.asarray(beta, dtype=float)
    return (
        np.asarray(s, dtype=float) + coefficient * basis,
        1.0 + coefficient * first_basis,
        coefficient * second_basis,
    )


def generate_v3_trajectory(
    reference: Mapping[str, np.ndarray | float],
    beta_flex: float,
    beta_extend: float,
) -> V3Trajectory:
    """Generate one two-parameter branch-aware coordination trajectory.

    Positive beta advances progression along the measured knee branch at a
    fixed frozen hip/reference phase; negative beta delays it.  Flexion and
    extension coefficients act on disjoint branch interiors.

    ``[0, 0]`` returns direct copies of frozen q/dq/ddq so reference recovery is
    array-exact and does not depend on spline roundoff.
    """

    q_reference = np.asarray(reference["q"], dtype=float)
    dq_reference = np.asarray(reference["dq"], dtype=float)
    ddq_reference = np.asarray(reference["ddq"], dtype=float)
    segment_phase = np.asarray(reference["segment_phase"], dtype=float)
    phases = np.asarray(reference["phases"])
    flexion = phases == "flexion"
    beta = np.where(flexion, float(beta_flex), float(beta_extend))
    warped, warp_first, warp_second = branch_warp(segment_phase, beta)

    if float(beta_flex) == 0.0 and float(beta_extend) == 0.0:
        return V3Trajectory(
            q=q_reference.copy(),
            dq=dq_reference.copy(),
            ddq=ddq_reference.copy(),
            warped_segment_phase=segment_phase.copy(),
            warp_first_derivative=np.ones_like(segment_phase),
            warp_second_derivative=np.zeros_like(segment_phase),
            beta_flex=0.0,
            beta_extend=0.0,
        )

    q = q_reference.copy()
    dq = dq_reference.copy()
    ddq = ddq_reference.copy()
    peak = float(reference["peak"])
    start = np.where(flexion, 0.0, peak)
    span = np.where(flexion, peak, 1.0 - peak)
    global_knee_phase = start + span * warped
    phase_rate = np.asarray(reference["phase_rate"], dtype=float)
    phase_accel = np.asarray(reference["phase_accel"], dtype=float)
    knee_rate = span * warp_first * phase_rate
    knee_accel = span * (warp_second * phase_rate**2 + warp_first * phase_accel)
    spline = reference["knee_spline"]
    q[:, 1] = spline(global_knee_phase)
    dq[:, 1] = spline(global_knee_phase, 1) * knee_rate
    ddq[:, 1] = spline(global_knee_phase, 2) * knee_rate**2 + spline(global_knee_phase, 1) * knee_accel

    # The analytic endpoint conditions already imply equality.  Copying the
    # exact frozen anchor samples prevents interpolation roundoff from changing
    # the prescribed endpoints; this is endpoint anchoring, not clipping.
    anchors = np.isclose(segment_phase, 0.0, atol=1.0e-15) | np.isclose(segment_phase, 1.0, atol=1.0e-15)
    q[anchors] = q_reference[anchors]
    dq[anchors] = dq_reference[anchors]
    ddq[anchors] = ddq_reference[anchors]

    return V3Trajectory(
        q=q,
        dq=dq,
        ddq=ddq,
        warped_segment_phase=warped,
        warp_first_derivative=warp_first,
        warp_second_derivative=warp_second,
        beta_flex=float(beta_flex),
        beta_extend=float(beta_extend),
    )


def semantics_payload() -> dict[str, object]:
    return {
        "parameterization_id": PARAMETERIZATION_ID,
        "dimension": 2,
        "parameter_order": list(PARAMETER_ORDER),
        "mathematical_transformation": "w_b(s; beta_b)=s+beta_b*64*s^3*(1-s)^3; q_hip=q_hip_ref; q_knee=q_knee_ref_branch(w_b)",
        "sign_semantics": {
            "positive": "advances knee progression along the measured branch relative to frozen hip/reference phase",
            "negative": "delays knee progression along the measured branch relative to frozen hip/reference phase",
            "zero": "identity on that branch",
        },
        "branch_independence": {
            "beta_flex": "flexion interior only",
            "beta_extend": "extension interior only",
        },
        "hip_trajectory": "array-identical to frozen V2 reference for every candidate",
        "identity": "beta_flex=beta_extend=0 returns array-identical frozen q/dq/ddq",
        "endpoint_conditions": ["w(0)=0", "w(1)=1", "w'(0)=w'(1)=1", "w''(0)=w''(1)=0"],
        "pointwise_clipping": False,
        "objective_dependency": False,
        "subject_truth_dependency": False,
    }
