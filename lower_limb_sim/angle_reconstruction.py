"""Pull-point based joint-angle reconstruction for stage 4.5D.

The public routine in this module is deliberately limited to measured pull-point
coordinates and *assumed* geometry.  It delegates the actual inverse/forward
kinematics and Jacobian diagnostics to the established project modules, so the
lower-leg convention remains ``theta_shank = q_hip - q_knee``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .jacobian import jacobian_diagnostics
from .kinematics import forward_kinematics, inverse_kinematics


PHYSIOLOGICAL_BRANCH = "physiological_knee_flexion"
NO_BRANCH = "none"


@dataclass(frozen=True)
class JointAngleReconstructionResult:
    """Reconstructed samples plus audit metadata.

    ``dataframe`` contains one row per input pull point.  Invalid rows retain
    NaN angles and must not be passed to the dynamics parameter estimator.
    """

    dataframe: pd.DataFrame
    metadata: dict[str, object]

    def __getitem__(self, key: str) -> pd.Series:
        return self.dataframe[key]

    def __len__(self) -> int:
        return len(self.dataframe)

    @property
    def q_hip_est_rad(self) -> np.ndarray:
        return self.dataframe["q_hip_est_rad"].to_numpy(dtype=float)

    @property
    def q_knee_est_rad(self) -> np.ndarray:
        return self.dataframe["q_knee_est_rad"].to_numpy(dtype=float)

    @property
    def ik_valid(self) -> np.ndarray:
        return self.dataframe["ik_valid"].to_numpy(dtype=bool)

    @property
    def ik_reason(self) -> np.ndarray:
        return self.dataframe["ik_reason"].to_numpy(dtype=object)

    @property
    def branch_selected(self) -> np.ndarray:
        return self.dataframe["branch_selected"].to_numpy(dtype=object)

    @property
    def ik_position_reconstruction_error_m(self) -> np.ndarray:
        return self.dataframe[
            "ik_position_reconstruction_error_m"
        ].to_numpy(dtype=float)

    @property
    def joint_continuity_valid(self) -> np.ndarray:
        return self.dataframe["joint_continuity_valid"].to_numpy(dtype=bool)

    @property
    def ik_domain_clip_applied(self) -> np.ndarray:
        return self.dataframe["ik_domain_clip_applied"].to_numpy(dtype=bool)

    @property
    def ik_domain_clip_amount(self) -> np.ndarray:
        return self.dataframe["ik_domain_clip_amount"].to_numpy(dtype=float)


def _append_reason(reasons: np.ndarray, mask: np.ndarray, reason: str) -> None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return
    current = reasons[selected].astype(str)
    reasons[selected] = np.where(
        current == "",
        reason,
        np.char.add(np.char.add(current, ";"), reason),
    )


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _geometry_value(geometry: object, explicit_name: str, alias: str) -> float:
    if isinstance(geometry, Mapping):
        if explicit_name in geometry:
            return float(geometry[explicit_name])
        if alias in geometry:
            return float(geometry[alias])
    if hasattr(geometry, explicit_name):
        return float(getattr(geometry, explicit_name))
    if hasattr(geometry, alias):
        return float(getattr(geometry, alias))
    raise TypeError(
        "assumed_geometry must expose "
        f"{explicit_name!r} (or its safe alias {alias!r})."
    )


def _resolve_assumed_geometry(
    assumed_geometry: object | None,
    L1_assumed_m: float | None,
    L2_assumed_m: float | None,
    hip_center_x_assumed_m: float | None,
    hip_center_z_assumed_m: float | None,
) -> tuple[float, float, float, float]:
    """Resolve only assumed geometry, explicitly rejecting a truth object."""

    explicit = (
        L1_assumed_m,
        L2_assumed_m,
        hip_center_x_assumed_m,
        hip_center_z_assumed_m,
    )
    if assumed_geometry is not None and any(value is not None for value in explicit):
        raise ValueError(
            "provide assumed_geometry or explicit assumed values, not both."
        )
    if assumed_geometry is not None:
        # TrueGeometry has these explicit fields.  Rejecting it here makes an
        # accidental ground-truth geometry hand-off fail closed.
        forbidden = (
            "L1_true_m",
            "L2_true_m",
            "hip_center_x_true_m",
            "hip_center_z_true_m",
        )
        mapping_keys = (
            set(assumed_geometry)
            if isinstance(assumed_geometry, Mapping)
            else set()
        )
        if any(
            hasattr(assumed_geometry, name) or name in mapping_keys
            for name in forbidden
        ):
            raise TypeError("true_geometry cannot be used for IK reconstruction.")
        L1_value = _geometry_value(
            assumed_geometry, "L1_assumed_m", "L1_m"
        )
        L2_value = _geometry_value(
            assumed_geometry, "L2_assumed_m", "L2_m"
        )
        hip_x = _geometry_value(
            assumed_geometry, "hip_center_x_assumed_m", "hip_center_x_m"
        )
        hip_z = _geometry_value(
            assumed_geometry, "hip_center_z_assumed_m", "hip_center_z_m"
        )
    else:
        if any(value is None for value in explicit):
            raise ValueError(
                "all four explicit assumed geometry values are required."
            )
        L1_value = float(L1_assumed_m)  # type: ignore[arg-type]
        L2_value = float(L2_assumed_m)  # type: ignore[arg-type]
        hip_x = float(hip_center_x_assumed_m)  # type: ignore[arg-type]
        hip_z = float(hip_center_z_assumed_m)  # type: ignore[arg-type]
    return (
        _finite_positive(L1_value, "L1_assumed_m"),
        _finite_positive(L2_value, "L2_assumed_m"),
        _finite_scalar(hip_x, "hip_center_x_assumed_m"),
        _finite_scalar(hip_z, "hip_center_z_assumed_m"),
    )


def _as_one_dimensional(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional.")
    return array


def _trajectory_groups(
    trajectory_ids: Any | None,
    count: int,
) -> np.ndarray:
    if trajectory_ids is None:
        return np.full(count, "trajectory", dtype=object)
    groups = np.asarray(trajectory_ids, dtype=object)
    if groups.ndim == 0:
        groups = np.full(count, groups.item(), dtype=object)
    if groups.shape != (count,):
        raise ValueError("trajectory_ids must have one value per sample.")
    return groups


def _validate_group_time(
    time_s: Any | None,
    groups: np.ndarray,
    count: int,
) -> np.ndarray:
    if time_s is None:
        return np.arange(count, dtype=float)
    time = _as_one_dimensional(time_s, "time_s")
    if time.shape != (count,) or not np.isfinite(time).all():
        raise ValueError("time_s must contain one finite value per sample.")
    for group in pd.unique(groups):
        selected = np.flatnonzero(groups == group)
        # A trajectory may not reappear after another trajectory, because that
        # would make continuity and derivative boundaries ambiguous.
        if len(selected) > 1 and np.any(np.diff(selected) != 1):
            raise ValueError("each trajectory_id must form one contiguous block.")
        if len(selected) > 1 and np.any(np.diff(time[selected]) <= 0.0):
            raise ValueError("time_s must be strictly increasing per trajectory.")
    return time


def _clip_pull_point_to_acos_domain(
    x_relative: np.ndarray,
    z_relative: np.ndarray,
    D: np.ndarray,
    L1_m: float,
    L2_m: float,
    clip_allowed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project tiny round-off violations to the annulus boundary.

    This is only a numerical preconditioner for the existing IK routine.  It is
    never used for violations larger than the configured tolerance.
    """

    x_used = x_relative.copy()
    z_used = z_relative.copy()
    if not clip_allowed.any():
        return x_used, z_used
    distance = np.hypot(x_relative, z_relative)
    outer_radius = L1_m + L2_m
    inner_radius = abs(L1_m - L2_m)
    target_radius = np.where(D > 1.0, outer_radius, inner_radius)
    safe = clip_allowed & (distance > 0.0)
    scale = np.ones_like(distance)
    scale[safe] = target_radius[safe] / distance[safe]
    x_used[safe] *= scale[safe]
    z_used[safe] *= scale[safe]
    return x_used, z_used


def reconstruct_joint_angles_from_pull_point(
    x_pull_measured_m: float | np.ndarray | pd.Series,
    z_pull_measured_m: float | np.ndarray | pd.Series,
    *,
    assumed_geometry: object | None = None,
    L1_assumed_m: float | None = None,
    L2_assumed_m: float | None = None,
    hip_center_x_assumed_m: float | None = None,
    hip_center_z_assumed_m: float | None = None,
    time_s: float | np.ndarray | pd.Series | None = None,
    trajectory_ids: object | np.ndarray | pd.Series | None = None,
    bed_height_m: float = 0.0,
    acos_domain_tolerance: float = 1e-10,
    maximum_joint_jump_rad: float = np.deg2rad(20.0),
    maximum_position_reconstruction_error_m: float = 1e-8,
) -> JointAngleReconstructionResult:
    """Reconstruct the physiological knee-flexion IK branch.

    Parameters contain no true angle or true geometry.  The measured pull point
    is translated by the assumed hip centre, then passed to the project's
    :func:`kinematics.inverse_kinematics`.  Samples fail closed on IK domain,
    joint limits, bed clearance, Jacobian conditioning, continuity, or
    non-finite values.
    """

    L1_m, L2_m, hip_x_m, hip_z_m = _resolve_assumed_geometry(
        assumed_geometry,
        L1_assumed_m,
        L2_assumed_m,
        hip_center_x_assumed_m,
        hip_center_z_assumed_m,
    )
    x_world = _as_one_dimensional(x_pull_measured_m, "x_pull_measured_m")
    z_world = _as_one_dimensional(z_pull_measured_m, "z_pull_measured_m")
    if x_world.shape != z_world.shape:
        raise ValueError("x and z pull-point arrays must have the same shape.")
    count = len(x_world)
    groups = _trajectory_groups(trajectory_ids, count)
    time = _validate_group_time(time_s, groups, count)

    bed_height = _finite_scalar(bed_height_m, "bed_height_m")
    domain_tolerance = _finite_scalar(
        acos_domain_tolerance, "acos_domain_tolerance"
    )
    max_jump = _finite_scalar(maximum_joint_jump_rad, "maximum_joint_jump_rad")
    max_position_error = _finite_scalar(
        maximum_position_reconstruction_error_m,
        "maximum_position_reconstruction_error_m",
    )
    if domain_tolerance < 0.0 or max_jump <= 0.0 or max_position_error < 0.0:
        raise ValueError("tolerances must be non-negative and jump must be positive.")

    reasons = np.full(count, "", dtype=object)
    finite_pull = np.isfinite(x_world) & np.isfinite(z_world)
    _append_reason(reasons, ~finite_pull, "nonfinite_pull_point")
    x_relative = x_world - hip_x_m
    z_relative = z_world - hip_z_m

    with np.errstate(invalid="ignore", divide="ignore"):
        D = (
            x_relative**2 + z_relative**2 - L1_m**2 - L2_m**2
        ) / (2.0 * L1_m * L2_m)
    domain_clip_amount = np.maximum(np.abs(D) - 1.0, 0.0)
    domain_violation = finite_pull & (domain_clip_amount > 0.0)
    clip_allowed = domain_violation & (
        domain_clip_amount <= domain_tolerance
    )
    domain_rejected = domain_violation & ~clip_allowed
    _append_reason(reasons, domain_rejected, "acos_domain_error")

    x_used, z_used = _clip_pull_point_to_acos_domain(
        x_relative,
        z_relative,
        D,
        L1_m,
        L2_m,
        clip_allowed,
    )
    q_hip_candidate, q_knee_candidate, reachable = inverse_kinematics(
        x_used,
        z_used,
        L1_m,
        L2_m,
    )
    q_hip_candidate = np.asarray(q_hip_candidate, dtype=float)
    q_knee_candidate = np.asarray(q_knee_candidate, dtype=float)
    reachable = np.asarray(reachable, dtype=bool)
    inverse_rejected = finite_pull & ~domain_rejected & ~reachable
    _append_reason(
        reasons,
        inverse_rejected,
        "joint_range_or_nonphysiological_branch",
    )

    x_knee_rel = np.full(count, np.nan)
    z_knee_rel = np.full(count, np.nan)
    x_reconstructed_rel = np.full(count, np.nan)
    z_reconstructed_rel = np.full(count, np.nan)
    if reachable.any():
        indices = np.flatnonzero(reachable)
        fk = forward_kinematics(
            q_hip_candidate[indices],
            q_knee_candidate[indices],
            L1_m,
            L2_m,
        )
        x_knee_rel[indices] = np.asarray(fk[0], dtype=float)
        z_knee_rel[indices] = np.asarray(fk[1], dtype=float)
        x_reconstructed_rel[indices] = np.asarray(fk[2], dtype=float)
        z_reconstructed_rel[indices] = np.asarray(fk[3], dtype=float)

    x_knee_world = hip_x_m + x_knee_rel
    z_knee_world = hip_z_m + z_knee_rel
    x_reconstructed_world = hip_x_m + x_reconstructed_rel
    z_reconstructed_world = hip_z_m + z_reconstructed_rel
    position_error = np.hypot(
        x_reconstructed_world - x_world,
        z_reconstructed_world - z_world,
    )
    finite_reconstruction = np.isfinite(position_error)
    position_valid = finite_reconstruction & (
        position_error <= max_position_error
    )
    _append_reason(
        reasons,
        reachable & ~finite_reconstruction,
        "nonfinite_position_reconstruction",
    )
    _append_reason(
        reasons,
        reachable & finite_reconstruction & ~position_valid,
        "position_reconstruction_error",
    )

    bed_valid = reachable & (
        z_knee_world >= bed_height - 1e-12
    ) & (z_reconstructed_world >= bed_height - 1e-12)
    _append_reason(reasons, reachable & ~bed_valid, "bed_constraint_violation")

    jacobian_determinant = np.full(count, np.nan)
    jacobian_condition_number = np.full(count, np.inf)
    jacobian_valid = np.zeros(count, dtype=bool)
    if reachable.any():
        indices = np.flatnonzero(reachable)
        diagnostics = jacobian_diagnostics(
            q_hip_candidate[indices],
            q_knee_candidate[indices],
            L1_m,
            L2_m,
        )
        jacobian_determinant[indices] = np.asarray(
            diagnostics.determinant, dtype=float
        )
        jacobian_condition_number[indices] = np.asarray(
            diagnostics.condition_number, dtype=float
        )
        jacobian_valid[indices] = ~np.asarray(
            diagnostics.near_singular, dtype=bool
        )
    _append_reason(
        reasons,
        reachable & ~jacobian_valid,
        "jacobian_near_singular",
    )

    candidate_valid = (
        finite_pull
        & ~domain_rejected
        & reachable
        & position_valid
        & bed_valid
        & jacobian_valid
    )
    continuity_valid = np.zeros(count, dtype=bool)
    discontinuity = np.zeros(count, dtype=bool)
    for group in pd.unique(groups):
        previous: int | None = None
        for index in np.flatnonzero(groups == group):
            if not candidate_valid[index]:
                continue
            if previous is None:
                continuity_valid[index] = True
            else:
                jump = max(
                    abs(q_hip_candidate[index] - q_hip_candidate[previous]),
                    abs(q_knee_candidate[index] - q_knee_candidate[previous]),
                )
                continuity_valid[index] = bool(jump <= max_jump + 1e-12)
                discontinuity[index] = not continuity_valid[index]
            # A rejected jump is not allowed to move the continuity reference.
            if continuity_valid[index]:
                previous = index
    _append_reason(reasons, discontinuity, "joint_angle_jump")

    ik_valid = candidate_valid & continuity_valid
    q_hip_est = np.where(ik_valid, q_hip_candidate, np.nan)
    q_knee_est = np.where(ik_valid, q_knee_candidate, np.nan)
    branch = np.where(reachable, PHYSIOLOGICAL_BRANCH, NO_BRANCH)
    dataframe = pd.DataFrame(
        {
            "time_s": time,
            "trajectory_id": groups,
            "x_pull_measured_m": x_world,
            "z_pull_measured_m": z_world,
            "x_pull_relative_assumed_hip_m": x_relative,
            "z_pull_relative_assumed_hip_m": z_relative,
            "q_hip_est_rad": q_hip_est,
            "q_knee_est_rad": q_knee_est,
            "ik_valid": ik_valid,
            "ik_reason": np.where(ik_valid, "", reasons),
            "branch_selected": branch,
            "ik_position_reconstruction_error_m": position_error,
            "joint_continuity_valid": continuity_valid,
            "ik_domain_value_D": D,
            "ik_domain_clip_applied": clip_allowed,
            "ik_domain_clip_amount": domain_clip_amount,
            "ik_joint_range_valid": reachable,
            "ik_bed_constraint_valid": bed_valid,
            "ik_jacobian_valid": jacobian_valid,
            "ik_jacobian_determinant": jacobian_determinant,
            "ik_jacobian_condition_number": jacobian_condition_number,
            "x_knee_reconstructed_m": x_knee_world,
            "z_knee_reconstructed_m": z_knee_world,
            "x_pull_reconstructed_m": x_reconstructed_world,
            "z_pull_reconstructed_m": z_reconstructed_world,
        }
    )
    valid_finite = dataframe.loc[
        dataframe["ik_valid"],
        [
            "q_hip_est_rad",
            "q_knee_est_rad",
            "ik_position_reconstruction_error_m",
            "ik_jacobian_condition_number",
        ],
    ].to_numpy(dtype=float)
    if valid_finite.size and not np.isfinite(valid_finite).all():
        raise RuntimeError("valid IK samples must be finite.")
    metadata: dict[str, object] = {
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "branch_policy": PHYSIOLOGICAL_BRANCH,
        "branch_selection_uses_true_angles": False,
        "true_geometry_accessed": False,
        "L1_assumed_m": L1_m,
        "L2_assumed_m": L2_m,
        "hip_center_x_assumed_m": hip_x_m,
        "hip_center_z_assumed_m": hip_z_m,
        "bed_height_m": bed_height,
        "acos_domain_tolerance": domain_tolerance,
        "domain_clip_amount_definition": "max(abs(D)-1, 0)",
        "maximum_joint_jump_rad": max_jump,
        "valid_samples": int(ik_valid.sum()),
        "invalid_samples": int((~ik_valid).sum()),
        "joint_discontinuity_count": int(discontinuity.sum()),
        "domain_clip_count": int(clip_allowed.sum()),
    }
    return JointAngleReconstructionResult(dataframe=dataframe, metadata=metadata)


__all__ = [
    "JointAngleReconstructionResult",
    "NO_BRANCH",
    "PHYSIOLOGICAL_BRANCH",
    "reconstruct_joint_angles_from_pull_point",
]
