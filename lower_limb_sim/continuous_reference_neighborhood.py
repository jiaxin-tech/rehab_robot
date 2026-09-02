"""Continuous, offline-only perturbations around the frozen active reference.

This module parameterizes a three-dimensional neighbourhood of
``reference_measured_asymmetric_closed_slow``.  It does not optimize, select,
connect to hardware, or authorize robot motion.  Search bounds below are
software experiment bounds and are deliberately unrelated to robot limits.

The model convention is always ``theta_shank = q_hip - q_knee``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, PchipInterpolator, make_interp_spline

from .config import (
    L1,
    L2,
    force_magnitude_limit_n,
    jacobian_condition_limit,
    jacobian_det_threshold,
)
from .dynamic_subject import get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    PROJECT_ROOT,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import StateDomainBounds, classify_state_domain
from .jacobian import jacobian_diagnostics
from .kinematics import forward_kinematics
from .reference_release import (
    RELEASE_ACTIVE_REFERENCE_PATH,
    FrozenReferenceBundle,
    load_frozen_active_reference,
    verify_reference_sha256,
)
from .run_reference_candidate_evaluation import LOCAL_DOMAIN_MINIMUM_PERCENT


GENERATOR_VERSION = "continuous_asymmetric_reference_neighborhood_v1"
OFFLINE_PERSONALIZATION_SEARCH_BOUNDS: dict[str, tuple[float, float]] = {
    "hip_amplitude_delta_deg": (-5.0, 2.0),
    "knee_amplitude_delta_deg": (-5.0, 2.0),
    "knee_phase_shift": (-0.03, 0.03),
}
FIXED_TIME_SCALE = 1.0
FLEXION_DURATION_S = 13.6
EXTENSION_DURATION_S = 10.4
TOTAL_DURATION_S = 24.0
DOMAIN_BOUNDS_PATH = (
    PROJECT_ROOT
    / "lower_limb_sim"
    / "data"
    / "reference_local_active_asymmetric"
    / "state_domain_bounds.json"
)
ASYMMETRY_MINIMUM_RETENTION_RATIO = 0.80
CONTINUITY_POSITION_TOLERANCE_RAD = 1e-10
CONTINUITY_VELOCITY_TOLERANCE_RAD_S = 1e-10
CONTINUITY_ACCELERATION_TOLERANCE_RAD_S2 = 1e-9

_STATE_COLUMNS = (
    "q_hip_est_rad",
    "q_knee_est_rad",
    "dq_hip_est_rad_s",
    "dq_knee_est_rad_s",
    "ddq_hip_est_rad_s2",
    "ddq_knee_est_rad_s2",
)
_TRAJECTORY_HASH_COLUMNS = (
    "time_s",
    "cycle_phase",
    "segment_phase",
    "global_phase",
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
    "theta_shank_rad",
    "x_knee_m",
    "z_knee_m",
    "x_pull_m",
    "z_pull_m",
)


@dataclass(frozen=True)
class ContinuousParameters:
    hip_amplitude_delta_deg: float = 0.0
    knee_amplitude_delta_deg: float = 0.0
    knee_phase_shift: float = 0.0
    time_scale: float = FIXED_TIME_SCALE

    @property
    def neutral(self) -> bool:
        return bool(
            self.hip_amplitude_delta_deg == 0.0
            and self.knee_amplitude_delta_deg == 0.0
            and self.knee_phase_shift == 0.0
            and self.time_scale == FIXED_TIME_SCALE
        )


@dataclass(frozen=True)
class TrajectoryConstraintAudit:
    closure_valid: bool
    rom_valid: bool
    workspace_valid: bool
    jacobian_valid: bool
    force_mapping_valid: bool
    domain_coverage: float
    domain_coverage_valid: bool
    velocity_valid: bool
    acceleration_valid: bool
    asymmetry_valid: bool
    finite_valid: bool
    trajectory_feasible: bool
    invalid_reason: str
    minimum_required_domain_coverage_percent: float
    minimum_abs_jacobian_determinant: float
    maximum_jacobian_condition: float
    maximum_abs_dq_hip_rad_s: float
    maximum_abs_dq_knee_rad_s: float
    maximum_abs_ddq_hip_rad_s2: float
    maximum_abs_ddq_knee_rad_s2: float
    maximum_offline_baseline_force_n: float
    velocity_gate_definition: str
    acceleration_gate_definition: str
    force_mapping_model: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedTrajectory:
    trajectory: pd.DataFrame
    metadata: dict[str, Any]
    constraints: TrajectoryConstraintAudit
    continuity_audit: dict[str, Any]
    asymmetry_audit: dict[str, Any]


@dataclass(frozen=True)
class _ParentModel:
    hip_spline: BSpline
    knee_spline: BSpline
    peak_global_phase: float
    parent: pd.DataFrame


def _finite_parameter(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_parameters(parameters: ContinuousParameters) -> None:
    for name in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS:
        value = _finite_parameter(getattr(parameters, name), name)
        lower, upper = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name]
        if value < lower or value > upper:
            raise ValueError(
                f"{name}={value} outside offline_personalization_search_bounds "
                f"[{lower}, {upper}]; clipping is prohibited"
            )
    if _finite_parameter(parameters.time_scale, "time_scale") != FIXED_TIME_SCALE:
        raise ValueError("time_scale is frozen at 1.0 in generator version 1")


def _build_parent_model(bundle: FrozenReferenceBundle) -> _ParentModel:
    verify_reference_sha256(RELEASE_ACTIVE_REFERENCE_PATH)
    if bundle.manifest["reference_id"] != ACTIVE_REFERENCE_ID:
        raise PermissionError("only the frozen active asymmetric reference is allowed")
    if bundle.manifest["sha256"] != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("REFERENCE_HASH_MISMATCH: parent manifest")
    if not bundle.audit.valid:
        raise RuntimeError("frozen parent reference audit failed")
    if bundle.manifest["approved_for_offline_personalization"] is not True:
        raise PermissionError("parent is not approved for offline personalization")
    parent = bundle.trajectory.copy(deep=True)
    canonical = pd.read_csv(RELEASE_ACTIVE_REFERENCE_PATH)
    if not parent.equals(canonical):
        raise RuntimeError(
            "REFERENCE_HASH_MISMATCH: supplied parent content differs from canonical release"
        )
    phase = parent["global_phase"].to_numpy(dtype=float)
    if not np.all(np.diff(phase) > 0.0):
        raise RuntimeError("parent global phase must be strictly increasing")
    if not np.isclose(phase[0], 0.0) or not np.isclose(phase[-1], 1.0):
        raise RuntimeError("parent global phase must span [0, 1]")
    peak_rows = parent.loc[parent["cycle_phase"].astype(str).eq("flexion")]
    peak = float(peak_rows["global_phase"].iloc[-1])
    if not 0.0 < peak < 1.0:
        raise RuntimeError("parent flexion/extension peak phase is invalid")
    return _ParentModel(
        hip_spline=make_interp_spline(
            phase,
            parent["q_hip_rad"].to_numpy(dtype=float),
            k=3,
            bc_type="periodic",
        ),
        knee_spline=make_interp_spline(
            phase,
            parent["q_knee_rad"].to_numpy(dtype=float),
            k=3,
            bc_type="periodic",
        ),
        peak_global_phase=peak,
        parent=parent,
    )


def _smoothstep5(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic smootherstep and first two derivatives by its argument."""

    u = np.asarray(value, dtype=float)
    y = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    dy = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    ddy = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return y, dy, ddy


def _amplitude_basis(
    local_phase: np.ndarray,
    flexion_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth, first, second = _smoothstep5(local_phase)
    sign = np.where(flexion_mask, 1.0, -1.0)
    value = np.where(flexion_mask, smooth, 1.0 - smooth)
    return value, sign * first, sign * second


def _phase_warp_terms(
    local_phase: np.ndarray,
    shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Endpoint/peak-preserving C2 phase warp ``W(r, shift)``.

    ``W = r + shift * 64 r^3 (1-r)^3`` is applied independently to
    flexion and extension.  Positive shift advances the knee along each
    measured branch.  The derivative remains positive throughout the v1
    software search interval.
    """

    r = np.asarray(local_phase, dtype=float)
    bump = 64.0 * (r**3 - 3.0 * r**4 + 3.0 * r**5 - r**6)
    bump_first = 64.0 * (
        3.0 * r**2 - 12.0 * r**3 + 15.0 * r**4 - 6.0 * r**5
    )
    bump_second = 64.0 * (
        6.0 * r - 36.0 * r**2 + 60.0 * r**3 - 30.0 * r**4
    )
    warped = r + shift * bump
    first = 1.0 + shift * bump_first
    second = shift * bump_second
    if not np.isclose(warped[0], r[0], atol=1e-14, rtol=0.0) and np.isclose(
        r[0], 0.0
    ):
        raise RuntimeError("phase warp changed a branch start")
    if np.any(first <= 0.0):
        raise ValueError("phase warp is not strictly monotone")
    if np.any((warped < -1e-13) | (warped > 1.0 + 1e-13)):
        raise ValueError("phase warp left its branch")
    return warped, first, second


def _trajectory_sha256(trajectory: pd.DataFrame) -> str:
    missing = set(_TRAJECTORY_HASH_COLUMNS).difference(trajectory.columns)
    if missing:
        raise ValueError(f"trajectory hash columns missing: {sorted(missing)}")
    payload = trajectory.loc[:, _TRAJECTORY_HASH_COLUMNS].to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def parameterized_trajectory_id(parameters: ContinuousParameters) -> str:
    def signed(value: float, digits: int) -> str:
        return f"{value:+.{digits}f}"

    return (
        "asym_"
        f"h{signed(parameters.hip_amplitude_delta_deg, 3)}_"
        f"k{signed(parameters.knee_amplitude_delta_deg, 3)}_"
        f"p{signed(parameters.knee_phase_shift, 5)}"
    )


def _load_domain_bounds(path: str | Path = DOMAIN_BOUNDS_PATH) -> StateDomainBounds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("active_reference_identifier") != ACTIVE_REFERENCE_ID:
        raise RuntimeError("identification domain belongs to another reference")
    if payload.get("active_reference_sha256") != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("identification domain parent SHA mismatch")
    bounds = payload.get("bounds")
    if not isinstance(bounds, Mapping):
        raise RuntimeError("identification domain bounds are missing")
    columns = tuple(map(str, bounds.get("columns", ())))
    lower = tuple(float(value) for value in bounds.get("lower", ()))
    upper = tuple(float(value) for value in bounds.get("upper", ()))
    if columns != _STATE_COLUMNS or len(lower) != 6 or len(upper) != 6:
        raise RuntimeError("identification domain has an unexpected schema")
    if not np.isfinite(np.asarray([lower, upper], dtype=float)).all():
        raise RuntimeError("identification domain contains non-finite bounds")
    return StateDomainBounds(
        columns=columns,
        lower=lower,
        upper=upper,
        valid_training_samples=int(bounds["valid_training_samples"]),
    )


def _state_table(trajectory: pd.DataFrame) -> pd.DataFrame:
    source = (
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    state = trajectory.loc[:, source].copy(deep=True)
    state.columns = _STATE_COLUMNS
    state["state_estimation_valid"] = np.isfinite(
        state.to_numpy(dtype=float)
    ).all(axis=1)
    return state


def _branch_asymmetry(trajectory: pd.DataFrame) -> tuple[float, float, float]:
    flexion = trajectory.loc[trajectory["cycle_phase"].astype(str).eq("flexion")]
    extension = trajectory.loc[
        trajectory["cycle_phase"].astype(str).eq("extension")
    ]
    if len(flexion) < 3 or len(extension) < 3:
        raise ValueError("trajectory must contain both asymmetric branches")
    query = np.linspace(0.0, 1.0, 2001)
    extension_phase = np.concatenate(
        ([0.0], extension["segment_phase"].to_numpy(dtype=float))
    )

    def values(column: str) -> tuple[np.ndarray, np.ndarray]:
        flex = PchipInterpolator(
            flexion["segment_phase"].to_numpy(dtype=float),
            flexion[column].to_numpy(dtype=float),
        )(query)
        ext = PchipInterpolator(
            extension_phase,
            np.concatenate(
                ([float(flexion[column].iloc[-1])], extension[column].to_numpy(dtype=float))
            ),
        )(1.0 - query)
        return np.asarray(flex), np.asarray(ext)

    hip_flex, hip_ext = values("q_hip_rad")
    knee_flex, knee_ext = values("q_knee_rad")
    x_flex, x_ext = values("x_pull_m")
    z_flex, z_ext = values("z_pull_m")
    return (
        float(np.sqrt(np.mean(np.rad2deg(hip_flex - hip_ext) ** 2))),
        float(np.sqrt(np.mean(np.rad2deg(knee_flex - knee_ext) ** 2))),
        float(
            np.sqrt(
                np.mean(
                    ((x_flex - x_ext) * 1000.0) ** 2
                    + ((z_flex - z_ext) * 1000.0) ** 2
                )
            )
        ),
    )


def _asymmetry_audit(
    parent: pd.DataFrame,
    generated: pd.DataFrame,
) -> dict[str, Any]:
    parent_values = _branch_asymmetry(parent)
    generated_values = _branch_asymmetry(generated)
    ratios = tuple(
        generated_value / parent_value
        for generated_value, parent_value in zip(generated_values, parent_values)
    )
    valid = bool(
        min(generated_values) > 1.0
        and min(ratios) >= ASYMMETRY_MINIMUM_RETENTION_RATIO
    )
    return {
        "measured_extension_is_reversed_flexion": False,
        "comparison_only_reverses_extension_time_axis": True,
        "parent_hip_flexion_extension_asymmetry_rmse_deg": parent_values[0],
        "parent_knee_flexion_extension_asymmetry_rmse_deg": parent_values[1],
        "parent_pull_path_asymmetry_rmse_mm": parent_values[2],
        "generated_hip_flexion_extension_asymmetry_rmse_deg": generated_values[0],
        "generated_knee_flexion_extension_asymmetry_rmse_deg": generated_values[1],
        "generated_pull_path_asymmetry_rmse_mm": generated_values[2],
        "hip_asymmetry_retention_ratio": ratios[0],
        "knee_asymmetry_retention_ratio": ratios[1],
        "pull_asymmetry_retention_ratio": ratios[2],
        "minimum_asymmetry_retention_ratio": ASYMMETRY_MINIMUM_RETENTION_RATIO,
        "asymmetry_valid": valid,
    }


def _continuity_audit(
    model: _ParentModel,
    parameters: ContinuousParameters,
) -> dict[str, Any]:
    hip_delta = math.radians(parameters.hip_amplitude_delta_deg)
    knee_delta = math.radians(parameters.knee_amplitude_delta_deg)
    peak = model.peak_global_phase

    def position(branch: str, at_end: bool) -> np.ndarray:
        r = np.asarray([1.0 if at_end else 0.0])
        flex = np.asarray([branch == "flexion"])
        basis, _, _ = _amplitude_basis(r, flex)
        warped, _, _ = _phase_warp_terms(r, parameters.knee_phase_shift)
        start = 0.0 if branch == "flexion" else peak
        span = peak if branch == "flexion" else 1.0 - peak
        hip_phase = start + span * r
        knee_phase = start + span * warped
        return np.asarray(
            [
                float(model.hip_spline(hip_phase)[0] + hip_delta * basis[0]),
                float(model.knee_spline(knee_phase)[0] + knee_delta * basis[0]),
            ]
        )

    peak_jump = position("extension", False) - position("flexion", True)
    seam_jump = position("flexion", False) - position("extension", True)
    maximum_position_jump = float(
        max(np.max(np.abs(peak_jump)), np.max(np.abs(seam_jump)))
    )
    # Both branches use a minimum-jerk time law.  Its first and second time
    # derivatives are exactly zero at cycle seam and flexion peak.  The
    # amplitude basis and phase warp also have zero first/second endpoint
    # derivatives, so both qdot and qddot jumps are analytically zero.
    maximum_velocity_jump = 0.0
    maximum_acceleration_jump = 0.0
    return {
        "continuity_order": 2,
        "position_continuity_warning_count": int(
            maximum_position_jump > CONTINUITY_POSITION_TOLERANCE_RAD
        ),
        "velocity_continuity_warning_count": int(
            maximum_velocity_jump > CONTINUITY_VELOCITY_TOLERANCE_RAD_S
        ),
        "acceleration_continuity_warning_count": int(
            maximum_acceleration_jump > CONTINUITY_ACCELERATION_TOLERANCE_RAD_S2
        ),
        "maximum_position_jump_rad": maximum_position_jump,
        "maximum_velocity_jump_rad_s": maximum_velocity_jump,
        "maximum_acceleration_jump_rad_s2": maximum_acceleration_jump,
        "position_tolerance_rad": CONTINUITY_POSITION_TOLERANCE_RAD,
        "velocity_tolerance_rad_s": CONTINUITY_VELOCITY_TOLERANCE_RAD_S,
        "acceleration_tolerance_rad_s2": CONTINUITY_ACCELERATION_TOLERANCE_RAD_S2,
        "internal_representation": "periodic_cubic_B_spline_composed_with_C2_polynomials",
        "passed": bool(maximum_position_jump <= CONTINUITY_POSITION_TOLERANCE_RAD),
    }


def _reference_deviation(
    parent: pd.DataFrame,
    generated: pd.DataFrame,
) -> dict[str, float]:
    hip = np.rad2deg(
        generated["q_hip_rad"].to_numpy(dtype=float)
        - parent["q_hip_rad"].to_numpy(dtype=float)
    )
    knee = np.rad2deg(
        generated["q_knee_rad"].to_numpy(dtype=float)
        - parent["q_knee_rad"].to_numpy(dtype=float)
    )
    pull = 1000.0 * np.hypot(
        generated["x_pull_m"].to_numpy(dtype=float)
        - parent["x_pull_m"].to_numpy(dtype=float),
        generated["z_pull_m"].to_numpy(dtype=float)
        - parent["z_pull_m"].to_numpy(dtype=float),
    )
    return {
        "hip_max_deviation_deg": float(np.max(np.abs(hip))),
        "hip_rms_deviation_deg": float(np.sqrt(np.mean(hip**2))),
        "knee_max_deviation_deg": float(np.max(np.abs(knee))),
        "knee_rms_deviation_deg": float(np.sqrt(np.mean(knee**2))),
        "pull_max_deviation_mm": float(np.max(pull)),
        "pull_rms_deviation_mm": float(np.sqrt(np.mean(pull**2))),
    }


def evaluate_trajectory_constraints(
    trajectory: pd.DataFrame,
    *,
    asymmetry_audit: Mapping[str, Any],
    continuity_audit: Mapping[str, Any],
    domain_bounds_path: str | Path = DOMAIN_BOUNDS_PATH,
    minimum_domain_coverage_percent: float = LOCAL_DOMAIN_MINIMUM_PERCENT,
) -> TrajectoryConstraintAudit:
    """Evaluate fixed offline constraints without clipping or motion approval."""

    required = set(_TRAJECTORY_HASH_COLUMNS)
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(f"generated trajectory missing columns: {sorted(missing)}")
    numerical = trajectory.loc[:, [c for c in required if c != "cycle_phase"]]
    finite_valid = bool(np.isfinite(numerical.to_numpy(dtype=float)).all())
    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
    hip_deg = np.rad2deg(q_hip)
    knee_deg = np.rad2deg(q_knee)
    rom_valid = bool(
        np.all(
            (hip_deg >= FORMAL_HIP_ROM_DEG[0] - 1e-12)
            & (hip_deg <= FORMAL_HIP_ROM_DEG[1] + 1e-12)
        )
        and np.all(
            (knee_deg >= FORMAL_KNEE_ROM_DEG[0] - 1e-12)
            & (knee_deg <= FORMAL_KNEE_ROM_DEG[1] + 1e-12)
        )
    )
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    workspace_valid = bool(
        np.isfinite(np.column_stack((x_knee, z_knee, x_pull, z_pull))).all()
        and np.all(x_pull >= -1e-12)
        and np.all(z_pull >= -1e-12)
        and np.all(z_knee >= -1e-12)
    )
    diagnostics = jacobian_diagnostics(q_hip, q_knee, L1, L2)
    determinant = np.asarray(diagnostics.determinant, dtype=float)
    condition = np.asarray(diagnostics.condition_number, dtype=float)
    near_singular = np.asarray(diagnostics.near_singular, dtype=bool)
    minimum_determinant = float(np.min(np.abs(determinant)))
    maximum_condition = float(np.max(condition))
    jacobian_valid = bool(
        np.isfinite(determinant).all()
        and np.isfinite(condition).all()
        and not near_singular.any()
        and minimum_determinant >= jacobian_det_threshold
        and maximum_condition <= jacobian_condition_limit
    )
    closure_valid = bool(
        np.allclose(q_hip[[0, -1]], q_hip[0], atol=1e-12, rtol=0.0)
        and np.allclose(q_knee[[0, -1]], q_knee[0], atol=1e-12, rtol=0.0)
        and np.allclose(
            np.asarray([x_pull[-1], z_pull[-1]]),
            np.asarray([x_pull[0], z_pull[0]]),
            atol=1e-12,
            rtol=0.0,
        )
        and bool(continuity_audit.get("passed"))
    )
    derivatives = trajectory[
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    # No robot velocity/acceleration threshold is inferred here.  These two
    # fields only certify finite analytic derivatives for downstream gates.
    velocity_valid = bool(np.isfinite(derivatives[:, :2]).all())
    acceleration_valid = bool(np.isfinite(derivatives[:, 2:]).all())

    bounds = _load_domain_bounds(domain_bounds_path)
    membership = classify_state_domain(_state_table(trajectory), bounds)
    domain_coverage = 100.0 * float(np.mean(membership))
    domain_valid = bool(domain_coverage >= minimum_domain_coverage_percent)

    subject = get_dynamic_subject("baseline")
    dynamics = inverse_dynamics(
        q_hip,
        q_knee,
        trajectory["dq_hip_rad_s"].to_numpy(dtype=float),
        trajectory["dq_knee_rad_s"].to_numpy(dtype=float),
        trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float),
        trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float),
        subject,
        L1,
    )
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        dynamics.tau_total_hip_nm,
        dynamics.tau_total_knee_nm,
        L1,
        L2,
    )
    force_valid = bool(np.asarray(force.force_mapping_valid, dtype=bool).all())
    finite_force = np.asarray(force.force_magnitude_n, dtype=float)
    maximum_force = (
        float(np.nanmax(finite_force)) if np.isfinite(finite_force).any() else math.inf
    )
    asymmetry_valid = bool(asymmetry_audit.get("asymmetry_valid"))

    checks = {
        "closure_invalid": closure_valid,
        "rom_invalid": rom_valid,
        "workspace_invalid": workspace_valid,
        "jacobian_invalid": jacobian_valid,
        "force_mapping_invalid": force_valid,
        "domain_coverage_insufficient": domain_valid,
        "velocity_non_finite": velocity_valid,
        "acceleration_non_finite": acceleration_valid,
        "asymmetry_invalid": asymmetry_valid,
        "non_finite_trajectory": finite_valid,
    }
    invalid = [reason for reason, passed in checks.items() if not passed]
    feasible = not invalid
    return TrajectoryConstraintAudit(
        closure_valid=closure_valid,
        rom_valid=rom_valid,
        workspace_valid=workspace_valid,
        jacobian_valid=jacobian_valid,
        force_mapping_valid=force_valid,
        domain_coverage=domain_coverage,
        domain_coverage_valid=domain_valid,
        velocity_valid=velocity_valid,
        acceleration_valid=acceleration_valid,
        asymmetry_valid=asymmetry_valid,
        finite_valid=finite_valid,
        trajectory_feasible=feasible,
        invalid_reason=";".join(invalid),
        minimum_required_domain_coverage_percent=float(
            minimum_domain_coverage_percent
        ),
        minimum_abs_jacobian_determinant=minimum_determinant,
        maximum_jacobian_condition=maximum_condition,
        maximum_abs_dq_hip_rad_s=float(
            np.max(np.abs(trajectory["dq_hip_rad_s"]))
        ),
        maximum_abs_dq_knee_rad_s=float(
            np.max(np.abs(trajectory["dq_knee_rad_s"]))
        ),
        maximum_abs_ddq_hip_rad_s2=float(
            np.max(np.abs(trajectory["ddq_hip_rad_s2"]))
        ),
        maximum_abs_ddq_knee_rad_s2=float(
            np.max(np.abs(trajectory["ddq_knee_rad_s2"]))
        ),
        maximum_offline_baseline_force_n=maximum_force,
        velocity_gate_definition="finite_analytic_derivative_only_no_robot_limit",
        acceleration_gate_definition="finite_analytic_derivative_only_no_robot_limit",
        force_mapping_model=(
            "baseline_virtual_subject_software_only;limit="
            f"{force_magnitude_limit_n:g}N_existing_config"
        ),
    )


def _annotate_constraint_result(
    trajectory: pd.DataFrame,
    constraints: TrajectoryConstraintAudit,
) -> None:
    """Overwrite inherited parent validity with this candidate's fail-closed result."""

    for field in (
        "closure_valid",
        "rom_valid",
        "workspace_valid",
        "jacobian_valid",
        "force_mapping_valid",
        "domain_coverage_valid",
        "velocity_valid",
        "acceleration_valid",
        "asymmetry_valid",
        "finite_valid",
        "trajectory_feasible",
    ):
        trajectory[field] = bool(getattr(constraints, field))
    trajectory["domain_coverage"] = float(constraints.domain_coverage)
    trajectory["invalid_reason"] = str(constraints.invalid_reason)
    trajectory["joint_limit_valid"] = bool(constraints.rom_valid)
    trajectory["trajectory_sample_valid"] = bool(
        constraints.trajectory_feasible
    )
    trajectory["formal_execution_allowed"] = False
    trajectory["allowed_for_first_robot_trial"] = False


def generate_personalized_trajectory(
    parent_reference: FrozenReferenceBundle | None = None,
    hip_amplitude_delta_deg: float = 0.0,
    knee_amplitude_delta_deg: float = 0.0,
    knee_phase_shift: float = 0.0,
    *,
    time_scale: float = FIXED_TIME_SCALE,
    domain_bounds_path: str | Path = DOMAIN_BOUNDS_PATH,
) -> GeneratedTrajectory:
    """Generate and audit one continuous offline reference perturbation."""

    parameters = ContinuousParameters(
        hip_amplitude_delta_deg=_finite_parameter(
            hip_amplitude_delta_deg, "hip_amplitude_delta_deg"
        ),
        knee_amplitude_delta_deg=_finite_parameter(
            knee_amplitude_delta_deg, "knee_amplitude_delta_deg"
        ),
        knee_phase_shift=_finite_parameter(knee_phase_shift, "knee_phase_shift"),
        time_scale=_finite_parameter(time_scale, "time_scale"),
    )
    _validate_parameters(parameters)
    bundle = (
        load_frozen_active_reference()
        if parent_reference is None
        else parent_reference
    )
    if not isinstance(bundle, FrozenReferenceBundle):
        raise TypeError(
            "parent_reference must be the fail-closed FrozenReferenceBundle"
        )
    model = _build_parent_model(bundle)
    parent = model.parent
    trajectory = parent.copy(deep=True)
    trajectory_id = parameterized_trajectory_id(parameters)

    if not parameters.neutral:
        local_phase = parent["segment_phase"].to_numpy(dtype=float)
        flexion_mask = parent["cycle_phase"].astype(str).eq("flexion").to_numpy()
        branch_start = np.where(flexion_mask, 0.0, model.peak_global_phase)
        branch_span = np.where(
            flexion_mask,
            model.peak_global_phase,
            1.0 - model.peak_global_phase,
        )
        phase_rate = parent["minimum_jerk_phase_rate_s_inv"].to_numpy(dtype=float)
        phase_acceleration = parent[
            "minimum_jerk_phase_acceleration_s_inv2"
        ].to_numpy(dtype=float)
        basis, basis_first, basis_second = _amplitude_basis(
            local_phase, flexion_mask
        )
        warped, warp_first, warp_second = _phase_warp_terms(
            local_phase, parameters.knee_phase_shift
        )

        hip_phase = branch_start + branch_span * local_phase
        knee_phase = branch_start + branch_span * warped
        hip_phase_rate = branch_span * phase_rate
        hip_phase_acceleration = branch_span * phase_acceleration
        knee_phase_rate = branch_span * warp_first * phase_rate
        knee_phase_acceleration = branch_span * (
            warp_second * phase_rate**2 + warp_first * phase_acceleration
        )
        hip_delta = math.radians(parameters.hip_amplitude_delta_deg)
        knee_delta = math.radians(parameters.knee_amplitude_delta_deg)

        q_hip = np.asarray(model.hip_spline(hip_phase), dtype=float) + hip_delta * basis
        q_knee = np.asarray(model.knee_spline(knee_phase), dtype=float) + knee_delta * basis
        dq_hip = (
            np.asarray(model.hip_spline(hip_phase, 1), dtype=float) * hip_phase_rate
            + hip_delta * basis_first * phase_rate
        )
        dq_knee = (
            np.asarray(model.knee_spline(knee_phase, 1), dtype=float) * knee_phase_rate
            + knee_delta * basis_first * phase_rate
        )
        ddq_hip = (
            np.asarray(model.hip_spline(hip_phase, 2), dtype=float) * hip_phase_rate**2
            + np.asarray(model.hip_spline(hip_phase, 1), dtype=float)
            * hip_phase_acceleration
            + hip_delta
            * (basis_second * phase_rate**2 + basis_first * phase_acceleration)
        )
        ddq_knee = (
            np.asarray(model.knee_spline(knee_phase, 2), dtype=float)
            * knee_phase_rate**2
            + np.asarray(model.knee_spline(knee_phase, 1), dtype=float)
            * knee_phase_acceleration
            + knee_delta
            * (basis_second * phase_rate**2 + basis_first * phase_acceleration)
        )
        trajectory["q_hip_rad"] = q_hip
        trajectory["q_knee_rad"] = q_knee
        trajectory["dq_hip_rad_s"] = dq_hip
        trajectory["dq_knee_rad_s"] = dq_knee
        trajectory["ddq_hip_rad_s2"] = ddq_hip
        trajectory["ddq_knee_rad_s2"] = ddq_knee
        x_knee, z_knee, x_pull, z_pull = forward_kinematics(
            q_hip, q_knee, L1, L2
        )
        trajectory["theta_shank_rad"] = q_hip - q_knee
        trajectory["x_knee_m"] = x_knee
        trajectory["z_knee_m"] = z_knee
        trajectory["x_pull_m"] = x_pull
        trajectory["z_pull_m"] = z_pull
    # Candidate identity/provenance columns are annotations only.  The neutral
    # q/dq/ddq/FK values remain bit-for-bit equal to the parent numerical data.
    trajectory["trajectory_id"] = trajectory_id
    trajectory["reference_version"] = GENERATOR_VERSION
    trajectory["active_reference"] = False
    trajectory["allowed_for_first_robot_trial"] = False
    trajectory["formal_execution_allowed"] = False
    trajectory["retimed_timing_is_original"] = False
    trajectory["reference_provenance"] = (
        "continuous_perturbation_of_frozen_measured_asymmetric_reference"
    )

    continuity = _continuity_audit(model, parameters)
    asymmetry = _asymmetry_audit(parent, trajectory)
    constraints = evaluate_trajectory_constraints(
        trajectory,
        asymmetry_audit=asymmetry,
        continuity_audit=continuity,
        domain_bounds_path=domain_bounds_path,
    )
    _annotate_constraint_result(trajectory, constraints)
    deviation = _reference_deviation(parent, trajectory)
    candidate_sha = _trajectory_sha256(trajectory)
    neutral_max_error = float(
        np.max(
            np.abs(
                trajectory[
                    [
                        "q_hip_rad",
                        "q_knee_rad",
                        "dq_hip_rad_s",
                        "dq_knee_rad_s",
                        "ddq_hip_rad_s2",
                        "ddq_knee_rad_s2",
                    ]
                ].to_numpy(dtype=float)
                - parent[
                    [
                        "q_hip_rad",
                        "q_knee_rad",
                        "dq_hip_rad_s",
                        "dq_knee_rad_s",
                        "ddq_hip_rad_s2",
                        "ddq_knee_rad_s2",
                    ]
                ].to_numpy(dtype=float)
            )
        )
    )
    metadata: dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "parent_sha_verified_before_generation": True,
        **asdict(parameters),
        "offline_personalization_search_bounds": {
            key: list(value)
            for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
        },
        "bounds_are_real_robot_safety_limits": False,
        "duration_optimization_enabled": False,
        "flexion_duration_s": FLEXION_DURATION_S,
        "extension_duration_s": EXTENSION_DURATION_S,
        "total_duration_s": TOTAL_DURATION_S,
        "generator_version": GENERATOR_VERSION,
        "generator_git_commit": _git_commit(),
        "generator_source_sha256": _file_sha256(Path(__file__)),
        "domain_bounds_source": str(Path(domain_bounds_path).resolve()),
        "domain_bounds_sha256": _file_sha256(domain_bounds_path),
        "trajectory_sha256": candidate_sha,
        "trajectory_sha256_definition": (
            "sha256_of_utf8_canonical_csv_selected_trajectory_columns_"
            "float17g_lf"
        ),
        "trajectory_sha256_columns": list(_TRAJECTORY_HASH_COLUMNS),
        "neutral_generator_sha": candidate_sha if parameters.neutral else None,
        "neutral_reference_max_abs_state_error": neutral_max_error,
        "neutral_exact_numeric_state_copy": parameters.neutral,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "L1_m": L1,
        "L2_m": L2,
        "measured_extension_is_reversed_flexion": False,
        "phase_warp_formula": "W(r,dphi)=r+dphi*64*r^3*(1-r)^3 per branch",
        "amplitude_basis_formula": (
            "flexion b(r)=10r^3-15r^4+6r^5; "
            "extension b(r)=1-(10r^3-15r^4+6r^5)"
        ),
        "pointwise_clipping_applied": False,
        "optimizer_implemented": False,
        "robot_motion_authorized": False,
        **deviation,
        **constraints.as_dict(),
        "continuity_audit": continuity,
        "asymmetry_audit": asymmetry,
    }
    return GeneratedTrajectory(
        trajectory=trajectory,
        metadata=metadata,
        constraints=constraints,
        continuity_audit=continuity,
        asymmetry_audit=asymmetry,
    )


__all__ = [
    "ASYMMETRY_MINIMUM_RETENTION_RATIO",
    "ContinuousParameters",
    "DOMAIN_BOUNDS_PATH",
    "FIXED_TIME_SCALE",
    "GENERATOR_VERSION",
    "GeneratedTrajectory",
    "OFFLINE_PERSONALIZATION_SEARCH_BOUNDS",
    "TrajectoryConstraintAudit",
    "evaluate_trajectory_constraints",
    "generate_personalized_trajectory",
    "parameterized_trajectory_id",
]
