"""Kinematic-only trajectory parameterization prototypes for the V2 boundary audit.

These generators are research design objects.  They do not define V3 bounds,
do not call MyoLeg, and do not evaluate any objective or subject truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.interpolate import make_interp_spline


PROTOTYPE_ID = "MYOLEG_V3_KINEMATIC_ONLY_PARAMETERIZATION_PROTOTYPES_V1"


@dataclass(frozen=True)
class PrototypeTrajectory:
    """A prescribed two-joint trajectory and its analytic time derivatives."""

    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    parameterization_id: str
    parameters: Mapping[str, float]


def _branch_bump(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-height degree-6 interior Bezier/B-spline basis and derivatives.

    ``64*u^3*(1-u)^3`` and its first two derivatives are zero as needed at
    branch endpoints.  That makes the displacement C2-compatible without
    post-generation clipping.
    """

    value = np.asarray(u, dtype=float)
    bump = 64.0 * value**3 * (1.0 - value) ** 3
    first = 192.0 * value**2 * (1.0 - value) ** 2 * (1.0 - 2.0 * value)
    second = 384.0 * value - 2304.0 * value**2 + 3840.0 * value**3 - 1920.0 * value**4
    return bump, first, second


def _add_branch_displacement(
    reference: Mapping[str, np.ndarray],
    displacement_deg: np.ndarray,
    displacement_du_deg: np.ndarray,
    displacement_du2_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate = np.asarray(reference["phase_rate"], dtype=float)
    accel = np.asarray(reference["phase_accel"], dtype=float)
    q = np.asarray(reference["q"], dtype=float) + np.radians(displacement_deg)
    dq = np.asarray(reference["dq"], dtype=float) + np.radians(displacement_du_deg) * rate[:, None]
    ddq = np.asarray(reference["ddq"], dtype=float) + np.radians(
        displacement_du2_deg * rate[:, None] ** 2 + displacement_du_deg * accel[:, None]
    )
    return q, dq, ddq


def p2_interior_bspline(
    reference: Mapping[str, np.ndarray],
    *,
    hip_flex_deg: float,
    hip_extension_deg: float,
    knee_flex_deg: float,
    knee_extension_deg: float,
) -> PrototypeTrajectory:
    """Four-coefficient branch-wise interior joint perturbation (P2)."""

    u = np.asarray(reference["segment_phase"], dtype=float)
    flex = np.asarray(reference["phases"]) == "flexion"
    bump, first, second = _branch_bump(u)
    coefficients = np.column_stack(
        (
            np.where(flex, hip_flex_deg, hip_extension_deg),
            np.where(flex, knee_flex_deg, knee_extension_deg),
        )
    )
    q, dq, ddq = _add_branch_displacement(
        reference,
        bump[:, None] * coefficients,
        first[:, None] * coefficients,
        second[:, None] * coefficients,
    )
    return PrototypeTrajectory(
        q=q,
        dq=dq,
        ddq=ddq,
        parameterization_id="P2_INTERIOR_BSPLINE_JOINT_PERTURBATION",
        parameters={
            "hip_flex_deg": hip_flex_deg,
            "hip_extension_deg": hip_extension_deg,
            "knee_flex_deg": knee_flex_deg,
            "knee_extension_deg": knee_extension_deg,
        },
    )


def _branch_normals(reference: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Return branch-wise angle-space normals and the degeneracy fraction.

    Normals are computed separately on flexion and extension.  Low-speed
    samples inherit the nearest well-defined in-branch normal; this is an audit
    mechanism, not a claim that Euclidean angle-space normal has mechanical
    meaning.
    """

    q_deg = np.degrees(np.asarray(reference["q"], dtype=float))
    dq_deg_s = np.degrees(np.asarray(reference["dq"], dtype=float))
    phases = np.asarray(reference["phases"])
    normal = np.zeros_like(q_deg)
    degenerate = np.zeros(len(q_deg), dtype=bool)
    for branch in ("flexion", "extension"):
        indices = np.flatnonzero(phases == branch)
        tangent = dq_deg_s[indices]
        magnitude = np.linalg.norm(tangent, axis=1)
        threshold = max(float(np.max(magnitude)) * 1.0e-8, 1.0e-12)
        valid = magnitude > threshold
        degenerate[indices] = ~valid
        if not np.any(valid):
            raise RuntimeError(f"normal is undefined for complete {branch} branch")
        valid_positions = np.flatnonzero(valid)
        for local_index in range(len(indices)):
            source = local_index if valid[local_index] else valid_positions[np.argmin(np.abs(valid_positions - local_index))]
            vector = tangent[source] / magnitude[source]
            normal[indices[local_index]] = np.asarray([-vector[1], vector[0]])
        for local_index in range(1, len(indices)):
            if float(np.dot(normal[indices[local_index - 1]], normal[indices[local_index]])) < 0.0:
                normal[indices[local_index]] *= -1.0
    return normal, float(np.mean(degenerate))


def p3_joint_space_normal(
    reference: Mapping[str, np.ndarray],
    *,
    flex_normal_deg: float,
    extension_normal_deg: float,
) -> tuple[PrototypeTrajectory, float]:
    """Two-coefficient branch-wise Euclidean angle-space normal displacement."""

    u = np.asarray(reference["segment_phase"], dtype=float)
    flex = np.asarray(reference["phases"]) == "flexion"
    bump, _, _ = _branch_bump(u)
    normal, degeneracy_fraction = _branch_normals(reference)
    amplitude = np.where(flex, flex_normal_deg, extension_normal_deg)
    displacement = bump[:, None] * amplitude[:, None] * normal

    # Derivatives are evaluated from one periodic cubic interpolation because
    # the nearest-normal branch repair is not analytic at its switching points.
    # The spline passes through the generated positions and supplies a closed
    # C2 trajectory; no pointwise clipping or endpoint overwrite is applied.
    time_s = np.asarray(reference["time_s"], dtype=float)
    q = np.asarray(reference["q"], dtype=float) + np.radians(displacement)
    spline = make_interp_spline(time_s, q, k=3, bc_type="periodic", axis=0)
    dq = spline(time_s, 1)
    ddq = spline(time_s, 2)
    return (
        PrototypeTrajectory(
            q=q,
            dq=dq,
            ddq=ddq,
            parameterization_id="P3_JOINT_SPACE_NORMAL_DISPLACEMENT",
            parameters={"flex_normal_deg": flex_normal_deg, "extension_normal_deg": extension_normal_deg},
        ),
        degeneracy_fraction,
    )


def p4_coordination_function(
    reference: Mapping[str, np.ndarray],
    *,
    flex_knee_coordination_deg: float,
    extension_knee_coordination_deg: float,
) -> PrototypeTrajectory:
    """Two-coefficient branch-aware normalized-phase coordination perturbation.

    Hip remains unchanged as the fixed task coordinate.  Knee is perturbed
    only in the branch interior as a deterministic function of the frozen
    normalized hip-reference phase.  Flexion and extension are handled
    separately, so the measured asymmetry is retained without requiring the
    measured hip angle itself to be strictly single-valued.
    """

    q_ref = np.asarray(reference["q"], dtype=float)
    phases = np.asarray(reference["phases"])
    flex = phases == "flexion"
    progress = np.asarray(reference["segment_phase"], dtype=float)
    if np.any(progress < -1.0e-10) or np.any(progress > 1.0 + 1.0e-10):
        raise RuntimeError("normalized branch phase is outside [0,1]; P4 is fail-closed")
    bump, first, second = _branch_bump(progress)
    amplitude = np.where(flex, flex_knee_coordination_deg, extension_knee_coordination_deg)

    # Chain derivatives through the unchanged normalized branch phase, so no
    # post-generation clipping or re-timing is introduced.
    progress_rate = np.asarray(reference["phase_rate"], dtype=float)
    progress_accel = np.asarray(reference["phase_accel"], dtype=float)
    displacement = np.zeros((len(progress), 2), dtype=float)
    displacement_du = np.zeros_like(displacement)
    displacement_du2 = np.zeros_like(displacement)
    displacement[:, 1] = bump * amplitude
    # Here the helper's generic phase chain is not used: derivatives are with
    # respect to hip progress, whose rates were computed above.
    q = q_ref + np.radians(displacement)
    dq = np.asarray(reference["dq"], dtype=float).copy()
    ddq = np.asarray(reference["ddq"], dtype=float).copy()
    dq[:, 1] += np.radians(first * amplitude) * progress_rate
    ddq[:, 1] += np.radians(second * amplitude) * progress_rate**2 + np.radians(first * amplitude) * progress_accel
    return PrototypeTrajectory(
        q=q,
        dq=dq,
        ddq=ddq,
        parameterization_id="P4_BRANCH_AWARE_COORDINATION_FUNCTION",
        parameters={
            "flex_knee_coordination_deg": flex_knee_coordination_deg,
            "extension_knee_coordination_deg": extension_knee_coordination_deg,
        },
    )


def prototype_definitions() -> list[dict[str, object]]:
    """Frozen structural definitions; scores are assigned in the audit protocol."""

    return [
        {
            "parameterization_id": "P1_PHASE_COORDINATION_ONLY",
            "dimension": 1,
            "parameters": ["knee_branch_internal_phase_warp"],
            "generator": "unchanged frozen V2 phase_warp",
        },
        {
            "parameterization_id": "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION",
            "dimension": 4,
            "parameters": ["hip_flex", "hip_extension", "knee_flex", "knee_extension"],
            "generator": "branch-wise degree-6 interior Bezier/B-spline basis",
        },
        {
            "parameterization_id": "P3_JOINT_SPACE_NORMAL_DISPLACEMENT",
            "dimension": 2,
            "parameters": ["flex_normal", "extension_normal"],
            "generator": "branch-wise smooth envelope times angle-space normal",
        },
        {
            "parameterization_id": "P4_BRANCH_AWARE_COORDINATION_FUNCTION",
            "dimension": 2,
            "parameters": ["flex_knee_coordination", "extension_knee_coordination"],
            "generator": "knee perturbation as interior function of normalized hip-reference branch phase",
        },
    ]
