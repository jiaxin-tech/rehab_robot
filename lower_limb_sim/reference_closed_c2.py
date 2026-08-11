"""C2-continuous closed reference trajectory for the approved experiment.

The Stage-5C ``reference_closed_symmetric`` path is retained as the immutable
source.  This module fits only its measured flexion branch with a quintic
B-spline and creates extension by exact time reversal.  The fit is accepted
only when explicit angle/pull-point shape-deviation gates and the run-local ROM
approval all pass.  It never clips joint angles and never changes global
configuration.

The lower-limb convention remains strictly
``theta_shank = q_hip - q_knee``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, PchipInterpolator, make_interp_spline

from .config import L1, L2
from .kinematics import forward_kinematics
from .reference_execution_trajectory import CLOSED_REFERENCE, retime_closed_reference
from .reference_trajectory_retiming import MODEL_ANGLE_DEFINITION
from .trajectory_profiles import minimum_jerk_profile


C2_REFERENCE = "reference_closed_c2"
C2_MODEL_VERSION = "lower_limb_sim_reference_closed_c2_v2"
APPROVED_HIP_ROM_DEG = (0.0, 120.0)
APPROVED_KNEE_ROM_DEG = (5.0, 145.0)

# These are shape-preservation acceptance gates, not robot safety thresholds.
DEFAULT_MAX_HIP_DEVIATION_DEG = 0.5
DEFAULT_MAX_KNEE_DEVIATION_DEG = 0.5
DEFAULT_MAX_PULL_DEVIATION_MM = 2.5
DEFAULT_QUINTIC_ANCHOR_COUNT = 99
DEFAULT_AUDIT_SAMPLE_COUNT = 20_001
DEFAULT_ACCELERATION_WARNING_RATIO = 10.0
DEFAULT_FULL_CYCLE_CONTINUITY_TOLERANCE = 1e-6


class C2ReferenceError(ValueError):
    """Raised when the source or C2 fit cannot pass the offline gates."""


@dataclass(frozen=True)
class C2ShapeAudit:
    spline_degree: int
    continuity_order: int
    internal_spline_continuity_order: int
    full_cycle_global_phase_continuity_order: int
    full_cycle_continuity_tolerance_rad: float
    reflection_boundary_derivative_orders: tuple[int, ...]
    source_flexion_sample_count: int
    anchor_count: int
    dense_audit_sample_count: int
    hip_max_deviation_deg: float
    knee_max_deviation_deg: float
    hip_rms_deviation_deg: float
    knee_rms_deviation_deg: float
    pull_point_max_deviation_mm: float
    pull_point_rms_deviation_mm: float
    hip_second_derivative_max_knot_jump_rad: float
    knee_second_derivative_max_knot_jump_rad: float
    hip_full_cycle_max_first_derivative_jump_rad: float
    knee_full_cycle_max_first_derivative_jump_rad: float
    hip_full_cycle_max_second_derivative_jump_rad: float
    knee_full_cycle_max_second_derivative_jump_rad: float
    hip_full_cycle_max_third_derivative_jump_rad: float
    knee_full_cycle_max_third_derivative_jump_rad: float
    hip_full_cycle_max_fourth_derivative_jump_rad: float
    knee_full_cycle_max_fourth_derivative_jump_rad: float
    start_pose_error_deg: float
    peak_pose_error_deg: float
    maximum_hip_deviation_gate_deg: float
    maximum_knee_deviation_gate_deg: float
    maximum_pull_deviation_gate_mm: float
    rom_violation_count: int
    pointwise_clip_applied: bool
    fit_accepted: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class C2ReferenceModel:
    """Immutable spline pair and its audited phase table."""

    hip_spline: BSpline
    knee_spline: BSpline
    phase_path: pd.DataFrame
    shape_audit: C2ShapeAudit
    approved_hip_rom_deg: tuple[float, float]
    approved_knee_rom_deg: tuple[float, float]


def _finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all() or not array[0] < array[1]:
        raise C2ReferenceError(f"{name} must contain two increasing finite values.")
    return float(array[0]), float(array[1])


def _strict_bool(series: pd.Series, name: str) -> np.ndarray:
    """Parse persisted CSV booleans without treating arbitrary text as true."""

    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise C2ReferenceError(f"{name} contains missing values.")
        return series.to_numpy(dtype=bool)
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.isin(("true", "false", "1", "0"))
    if not bool(valid.all()):
        raise C2ReferenceError(f"{name} contains invalid boolean encodings.")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def _closed_flexion_source(
    reference_versions: pd.DataFrame,
    approved_hip_rom_deg: Sequence[float],
    approved_knee_rom_deg: Sequence[float],
) -> pd.DataFrame:
    required = {
        "reference_version",
        "cycle_phase",
        "segment_phase",
        "q_hip_reference_rad",
        "q_knee_reference_rad",
        "formal_execution_allowed",
        "rom_mapping_applied",
    }
    missing = required.difference(reference_versions.columns)
    if missing:
        raise C2ReferenceError(
            f"reference_closed_symmetric source missing columns: {sorted(missing)}"
        )
    closed = reference_versions.loc[
        reference_versions["reference_version"].astype(str).eq(CLOSED_REFERENCE)
    ].copy(deep=True)
    if closed.empty:
        raise C2ReferenceError(f"source does not contain {CLOSED_REFERENCE!r}.")

    formal = _strict_bool(closed["formal_execution_allowed"], "formal_execution_allowed")
    if not bool(formal.all()):
        raise C2ReferenceError(
            "reference_closed_symmetric is not formally ROM-approved; C2 generation "
            "remains fail-closed."
        )
    mapped = _strict_bool(closed["rom_mapping_applied"], "rom_mapping_applied")
    if bool(mapped.any()):
        raise C2ReferenceError(
            "the approved experiment requires the original amplitude; a ROM-mapped "
            "source is not accepted."
        )

    approved_hip = _finite_pair(approved_hip_rom_deg, "approved_hip_rom_deg")
    approved_knee = _finite_pair(approved_knee_rom_deg, "approved_knee_rom_deg")
    expected_approval_columns = {
        "q_hip_approved_min_deg": approved_hip[0],
        "q_hip_approved_max_deg": approved_hip[1],
        "q_knee_approved_min_deg": approved_knee[0],
        "q_knee_approved_max_deg": approved_knee[1],
    }
    for column, expected in expected_approval_columns.items():
        if column not in closed:
            raise C2ReferenceError(f"source approval audit column is missing: {column}")
        values = pd.to_numeric(closed[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(
            values, expected, atol=1e-12, rtol=0.0
        ):
            raise C2ReferenceError(
                f"source {column} does not match the explicit approved ROM value {expected}."
            )

    flexion = closed.loc[closed["cycle_phase"].astype(str).eq("flexion")].copy()
    extension = closed.loc[closed["cycle_phase"].astype(str).eq("extension")].copy()
    if len(flexion) < 7 or len(extension) != len(flexion):
        raise C2ReferenceError(
            "closed source needs equal flexion/extension branches and at least 7 samples."
        )
    flexion = flexion.sort_values("segment_phase", kind="mergesort").reset_index(drop=True)
    extension = extension.sort_values("segment_phase", kind="mergesort").reset_index(
        drop=True
    )
    phase = flexion["segment_phase"].to_numpy(float)
    states = flexion[["q_hip_reference_rad", "q_knee_reference_rad"]].to_numpy(float)
    if not np.isfinite(np.column_stack((phase, states))).all() or not np.all(
        np.diff(phase) > 0.0
    ):
        raise C2ReferenceError("flexion phase and joint angles must be finite and ordered.")
    if not np.isclose(phase[0], 0.0, atol=1e-12) or not np.isclose(
        phase[-1], 1.0, atol=1e-12
    ):
        raise C2ReferenceError("flexion segment_phase must cover exactly [0, 1].")
    for joint in ("hip", "knee"):
        forward = flexion[f"q_{joint}_reference_rad"].to_numpy(float)
        reverse = extension[f"q_{joint}_reference_rad"].to_numpy(float)
        if not np.allclose(reverse, forward[::-1], atol=1e-12, rtol=0.0):
            raise C2ReferenceError(
                "source extension must be the exact reversed flexion branch."
            )
    return flexion


def _anchor_indices(flexion: pd.DataFrame, anchor_count: int) -> np.ndarray:
    count = len(flexion)
    requested = int(anchor_count)
    if requested < 7:
        raise C2ReferenceError("quintic spline anchor_count must be at least 7.")
    requested = min(requested, count)
    indices = np.rint(np.linspace(0, count - 1, requested)).astype(int)
    # Preserve both joint maxima even if a future source has an interior peak.
    mandatory = np.asarray(
        [
            0,
            count - 1,
            int(np.argmax(flexion["q_hip_reference_rad"].to_numpy(float))),
            int(np.argmax(flexion["q_knee_reference_rad"].to_numpy(float))),
        ],
        dtype=int,
    )
    indices = np.unique(np.concatenate((indices, mandatory)))
    if len(indices) < 7:
        raise C2ReferenceError("not enough distinct anchors for a quintic spline.")
    return indices


def _maximum_second_derivative_knot_jump(spline: BSpline) -> float:
    """Numerically audit the actual simple interior B-spline knots."""

    knots = np.unique(np.asarray(spline.t, dtype=float))
    if len(knots) <= 2:
        return 0.0
    interior = knots[1:-1]
    left = np.nextafter(interior, -np.inf)
    right = np.nextafter(interior, np.inf)
    return float(np.max(np.abs(spline(left, 2) - spline(right, 2))))


def _closed_cycle_global_phase_derivative_jumps(
    spline: BSpline,
) -> dict[int, float]:
    """Return seam/loop jumps for the flexion-plus-reversed-extension cycle.

    Full-cycle global phase ``g`` maps to flexion phase as ``s=2g`` on the
    forward branch and ``s=2-2g`` on the reverse branch.  Even derivatives
    therefore meet automatically.  Odd derivatives meet only when the
    corresponding flexion endpoint derivative is zero.
    """

    jumps: dict[int, float] = {}
    for order in range(1, 5):
        scale_forward = float(2**order)
        scale_reverse = float((-2) ** order)
        start_value = float(spline(0.0, order))
        peak_value = float(spline(1.0, order))
        turning_jump = abs(scale_reverse * peak_value - scale_forward * peak_value)
        loop_jump = abs(scale_forward * start_value - scale_reverse * start_value)
        jumps[order] = max(turning_jump, loop_jump)
    return jumps


def _build_phase_table(
    flexion: pd.DataFrame,
    hip_spline: BSpline,
    knee_spline: BSpline,
) -> pd.DataFrame:
    phase = flexion["segment_phase"].to_numpy(float)
    source_hip = flexion["q_hip_reference_rad"].to_numpy(float)
    source_knee = flexion["q_knee_reference_rad"].to_numpy(float)
    q_hip = np.asarray(hip_spline(phase), dtype=float)
    q_knee = np.asarray(knee_spline(phase), dtype=float)
    flex = pd.DataFrame(
        {
            "reference_version": C2_REFERENCE,
            "cycle_phase": "flexion",
            "segment_phase": phase,
            "global_phase": 0.5 * phase,
            "source_flexion_phase": phase,
            "q_hip_original_pchip_rad": source_hip,
            "q_knee_original_pchip_rad": source_knee,
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "dq_hip_ds_rad": hip_spline(phase, 1),
            "dq_knee_ds_rad": knee_spline(phase, 1),
            "d2q_hip_ds2_rad": hip_spline(phase, 2),
            "d2q_knee_ds2_rad": knee_spline(phase, 2),
        }
    )
    extension = flex.iloc[::-1].reset_index(drop=True).copy(deep=True)
    extension["cycle_phase"] = "extension"
    extension["segment_phase"] = np.linspace(0.0, 1.0, len(extension))
    extension["global_phase"] = 0.5 + 0.5 * extension["segment_phase"]
    # Derivatives below are with respect to the local extension phase.
    extension["dq_hip_ds_rad"] *= -1.0
    extension["dq_knee_ds_rad"] *= -1.0

    # Store the shared peak-flexion sample only once so global_phase is strictly
    # increasing and can be consumed directly by later offline interpolation.
    output = pd.concat((flex, extension.iloc[1:]), ignore_index=True)
    output["theta_shank_rad"] = output["q_hip_rad"] - output["q_knee_rad"]
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        output["q_hip_rad"].to_numpy(float),
        output["q_knee_rad"].to_numpy(float),
        L1,
        L2,
    )
    output["x_knee_m"] = x_knee
    output["z_knee_m"] = z_knee
    output["x_pull_m"] = x_pull
    output["z_pull_m"] = z_pull
    output["spline_degree"] = 5
    output["continuity_order"] = 4
    output["internal_spline_continuity_order"] = 4
    output["full_cycle_global_phase_continuity_order"] = 4
    output["reflection_boundary_conditions"] = (
        "flexion_dq_ds_order_1_and_3_equal_zero_at_start_and_peak"
    )
    output["minimum_required_continuity_order"] = 2
    output["extension_is_exact_reverse_of_flexion"] = True
    output["shared_turning_sample_stored_once"] = True
    output["pointwise_angle_clipping_applied"] = False
    output["rom_mapping_applied"] = False
    output["reference_path_preserved"] = True
    output["source_reference_overwritten"] = False
    output["shape_preserved_within_audit_gates"] = True
    output["reference_path_preserved_meaning"] = (
        "immutable_source_retained_and_c2_shape_within_explicit_deviation_gates"
    )
    output["model_angle_definition"] = MODEL_ANGLE_DEFINITION
    output["source_timing_status"] = "unknown"
    output["retimed_trajectory"] = False
    output["retimed_timing_is_original"] = False
    output["source_trajectory_type"] = "provided_rehabilitation_reference"
    output["simulation_status"] = "software_only"
    output["formal_execution_scope"] = "offline_reference_rom_and_shape_gate_only"
    output["robot_execution_approved"] = False
    output["observed_ankle_is_pull_point"] = False
    output["L1_m"] = L1
    output["L2_m"] = L2
    output["L2_definition"] = "knee_to_strap_equivalent_pull_point"
    return output


def fit_reference_closed_c2(
    reference_versions: pd.DataFrame,
    *,
    approved_hip_rom_deg: Sequence[float] = APPROVED_HIP_ROM_DEG,
    approved_knee_rom_deg: Sequence[float] = APPROVED_KNEE_ROM_DEG,
    anchor_count: int = DEFAULT_QUINTIC_ANCHOR_COUNT,
    dense_audit_sample_count: int = DEFAULT_AUDIT_SAMPLE_COUNT,
    maximum_hip_deviation_deg: float = DEFAULT_MAX_HIP_DEVIATION_DEG,
    maximum_knee_deviation_deg: float = DEFAULT_MAX_KNEE_DEVIATION_DEG,
    maximum_pull_deviation_mm: float = DEFAULT_MAX_PULL_DEVIATION_MM,
) -> C2ReferenceModel:
    """Fit and fail-closed audit the approved closed reference path."""

    flexion = _closed_flexion_source(
        reference_versions, approved_hip_rom_deg, approved_knee_rom_deg
    )
    audit_count = int(dense_audit_sample_count)
    if audit_count < 1001:
        raise C2ReferenceError("dense_audit_sample_count must be at least 1001.")
    gates = np.asarray(
        (
            maximum_hip_deviation_deg,
            maximum_knee_deviation_deg,
            maximum_pull_deviation_mm,
        ),
        dtype=float,
    )
    if not np.isfinite(gates).all() or np.any(gates <= 0.0):
        raise C2ReferenceError("shape-deviation gates must be finite and positive.")

    phase = flexion["segment_phase"].to_numpy(float)
    source_hip = flexion["q_hip_reference_rad"].to_numpy(float)
    source_knee = flexion["q_knee_reference_rad"].to_numpy(float)
    indices = _anchor_indices(flexion, anchor_count)
    anchor_phase = phase[indices]
    # Exact reversal changes the sign of odd path derivatives.  Constraining
    # first and third derivatives to zero at both flexion endpoints makes the
    # reflected full cycle C4: even derivatives meet automatically, while the
    # constrained odd derivatives meet at zero.  The third-derivative
    # constraints also supply the additional boundary conditions required by
    # the quintic interpolating B-spline without imposing an unnecessary zero
    # curvature condition.
    reflection_boundary_conditions = (
        [(1, 0.0), (3, 0.0)],
        [(1, 0.0), (3, 0.0)],
    )
    hip_spline = make_interp_spline(
        anchor_phase,
        source_hip[indices],
        k=5,
        bc_type=reflection_boundary_conditions,
        check_finite=True,
    )
    knee_spline = make_interp_spline(
        anchor_phase,
        source_knee[indices],
        k=5,
        bc_type=reflection_boundary_conditions,
        check_finite=True,
    )

    dense_phase = np.linspace(0.0, 1.0, audit_count)
    original_hip = PchipInterpolator(phase, source_hip, extrapolate=False)(dense_phase)
    original_knee = PchipInterpolator(phase, source_knee, extrapolate=False)(dense_phase)
    c2_hip = np.asarray(hip_spline(dense_phase), dtype=float)
    c2_knee = np.asarray(knee_spline(dense_phase), dtype=float)
    if not np.isfinite(np.column_stack((c2_hip, c2_knee))).all():
        raise C2ReferenceError("quintic spline generated non-finite joint angles.")

    hip_error_deg = np.rad2deg(c2_hip - original_hip)
    knee_error_deg = np.rad2deg(c2_knee - original_knee)
    _, _, original_x, original_z = forward_kinematics(
        original_hip, original_knee, L1, L2
    )
    _, _, c2_x, c2_z = forward_kinematics(c2_hip, c2_knee, L1, L2)
    pull_error_mm = 1000.0 * np.hypot(c2_x - original_x, c2_z - original_z)

    hip_rom = np.deg2rad(_finite_pair(approved_hip_rom_deg, "approved_hip_rom_deg"))
    knee_rom = np.deg2rad(_finite_pair(approved_knee_rom_deg, "approved_knee_rom_deg"))
    rom_invalid = (
        (c2_hip < hip_rom[0] - 1e-12)
        | (c2_hip > hip_rom[1] + 1e-12)
        | (c2_knee < knee_rom[0] - 1e-12)
        | (c2_knee > knee_rom[1] + 1e-12)
    )
    source_peak_index = int(np.argmax(source_knee))
    source_peak_phase = phase[source_peak_index]
    peak_error_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    np.asarray(
                        (
                            hip_spline(source_peak_phase) - source_hip[source_peak_index],
                            knee_spline(source_peak_phase) - source_knee[source_peak_index],
                        )
                    )
                )
            )
        )
    )
    start_error_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    np.asarray(
                        (
                            hip_spline(phase[0]) - source_hip[0],
                            knee_spline(phase[0]) - source_knee[0],
                        )
                    )
                )
            )
        )
    )

    hip_max = float(np.max(np.abs(hip_error_deg)))
    knee_max = float(np.max(np.abs(knee_error_deg)))
    pull_max = float(np.max(pull_error_mm))
    rejection_reasons: list[str] = []
    if hip_max > float(maximum_hip_deviation_deg) + 1e-12:
        rejection_reasons.append("hip_shape_deviation_exceeded")
    if knee_max > float(maximum_knee_deviation_deg) + 1e-12:
        rejection_reasons.append("knee_shape_deviation_exceeded")
    if pull_max > float(maximum_pull_deviation_mm) + 1e-12:
        rejection_reasons.append("pull_path_deviation_exceeded")
    if bool(rom_invalid.any()):
        rejection_reasons.append("new_rom_violation")
    if start_error_deg > 1e-10:
        rejection_reasons.append("start_pose_not_preserved")
    if peak_error_deg > 1e-10:
        rejection_reasons.append("peak_flexion_pose_not_preserved")

    hip_cycle_jumps = _closed_cycle_global_phase_derivative_jumps(hip_spline)
    knee_cycle_jumps = _closed_cycle_global_phase_derivative_jumps(knee_spline)
    maximum_cycle_jump = max((*hip_cycle_jumps.values(), *knee_cycle_jumps.values()))
    if maximum_cycle_jump > DEFAULT_FULL_CYCLE_CONTINUITY_TOLERANCE:
        rejection_reasons.append("full_cycle_global_phase_continuity_failed")
    audit = C2ShapeAudit(
        spline_degree=5,
        continuity_order=4,
        internal_spline_continuity_order=4,
        full_cycle_global_phase_continuity_order=4,
        full_cycle_continuity_tolerance_rad=(
            DEFAULT_FULL_CYCLE_CONTINUITY_TOLERANCE
        ),
        reflection_boundary_derivative_orders=(1, 3),
        source_flexion_sample_count=len(flexion),
        anchor_count=len(indices),
        dense_audit_sample_count=audit_count,
        hip_max_deviation_deg=hip_max,
        knee_max_deviation_deg=knee_max,
        hip_rms_deviation_deg=float(np.sqrt(np.mean(hip_error_deg**2))),
        knee_rms_deviation_deg=float(np.sqrt(np.mean(knee_error_deg**2))),
        pull_point_max_deviation_mm=pull_max,
        pull_point_rms_deviation_mm=float(np.sqrt(np.mean(pull_error_mm**2))),
        hip_second_derivative_max_knot_jump_rad=_maximum_second_derivative_knot_jump(
            hip_spline
        ),
        knee_second_derivative_max_knot_jump_rad=_maximum_second_derivative_knot_jump(
            knee_spline
        ),
        hip_full_cycle_max_first_derivative_jump_rad=hip_cycle_jumps[1],
        knee_full_cycle_max_first_derivative_jump_rad=knee_cycle_jumps[1],
        hip_full_cycle_max_second_derivative_jump_rad=hip_cycle_jumps[2],
        knee_full_cycle_max_second_derivative_jump_rad=knee_cycle_jumps[2],
        hip_full_cycle_max_third_derivative_jump_rad=hip_cycle_jumps[3],
        knee_full_cycle_max_third_derivative_jump_rad=knee_cycle_jumps[3],
        hip_full_cycle_max_fourth_derivative_jump_rad=hip_cycle_jumps[4],
        knee_full_cycle_max_fourth_derivative_jump_rad=knee_cycle_jumps[4],
        start_pose_error_deg=start_error_deg,
        peak_pose_error_deg=peak_error_deg,
        maximum_hip_deviation_gate_deg=float(maximum_hip_deviation_deg),
        maximum_knee_deviation_gate_deg=float(maximum_knee_deviation_deg),
        maximum_pull_deviation_gate_mm=float(maximum_pull_deviation_mm),
        rom_violation_count=int(rom_invalid.sum()),
        pointwise_clip_applied=False,
        fit_accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
    )
    if rejection_reasons:
        raise C2ReferenceError(
            "reference_closed_c2 fit rejected: " + ";".join(rejection_reasons)
        )
    phase_path = _build_phase_table(flexion, hip_spline, knee_spline)
    phase_path["approved_hip_min_deg"] = float(hip_rom[0] * 180.0 / np.pi)
    phase_path["approved_hip_max_deg"] = float(hip_rom[1] * 180.0 / np.pi)
    phase_path["approved_knee_min_deg"] = float(knee_rom[0] * 180.0 / np.pi)
    phase_path["approved_knee_max_deg"] = float(knee_rom[1] * 180.0 / np.pi)
    phase_path["joint_limit_valid"] = True
    phase_path["trajectory_sample_valid"] = True
    phase_path["invalid_reason"] = ""
    phase_path["formal_execution_allowed"] = True
    return C2ReferenceModel(
        hip_spline,
        knee_spline,
        phase_path,
        audit,
        tuple(map(float, approved_hip_rom_deg)),
        tuple(map(float, approved_knee_rom_deg)),
    )


def retime_reference_closed_c2(
    model: C2ReferenceModel,
    *,
    profile: str,
    flexion_duration_s: float,
    extension_duration_s: float,
    samples_per_segment: int = 201,
) -> pd.DataFrame:
    """Apply minimum jerk along the C2 path using analytic chain derivatives."""

    count = int(samples_per_segment)
    durations = np.asarray((flexion_duration_s, extension_duration_s), dtype=float)
    if count < 3:
        raise C2ReferenceError("samples_per_segment must be at least 3.")
    if not np.isfinite(durations).all() or np.any(durations <= 0.0):
        raise C2ReferenceError("retiming durations must be finite and positive.")

    frames: list[pd.DataFrame] = []
    time_offset = 0.0
    for phase_name, duration, global_offset in (
        ("flexion", durations[0], 0.0),
        ("extension", durations[1], 0.5),
    ):
        u = np.linspace(0.0, 1.0, count)
        path_s, path_rate, path_acceleration = minimum_jerk_profile(u, duration)
        direction = 1.0 if phase_name == "flexion" else -1.0
        source_phase = path_s if phase_name == "flexion" else 1.0 - path_s
        values: dict[str, object] = {
            "reference_version": C2_REFERENCE,
            "trajectory_id": f"{C2_REFERENCE}_{profile}",
            "profile": profile,
            "time_s": time_offset + duration * u,
            "cycle_phase": phase_name,
            "segment_phase": path_s,
            "global_phase": global_offset + 0.5 * path_s,
            "source_flexion_phase": source_phase,
            "minimum_jerk_phase_rate_s_inv": path_rate,
            "minimum_jerk_phase_acceleration_s_inv2": path_acceleration,
        }
        for joint, spline in (
            ("hip", model.hip_spline),
            ("knee", model.knee_spline),
        ):
            q = np.asarray(spline(source_phase), dtype=float)
            q_local_s = direction * np.asarray(spline(source_phase, 1), dtype=float)
            q_local_ss = np.asarray(spline(source_phase, 2), dtype=float)
            values[f"q_{joint}_rad"] = q
            values[f"dq_{joint}_ds_rad"] = q_local_s
            values[f"d2q_{joint}_ds2_rad"] = q_local_ss
            values[f"dq_{joint}_rad_s"] = q_local_s * path_rate
            values[f"ddq_{joint}_rad_s2"] = (
                q_local_ss * path_rate**2 + q_local_s * path_acceleration
            )
        frames.append(pd.DataFrame(values))
        time_offset += duration

    output = pd.concat((frames[0], frames[1].iloc[1:]), ignore_index=True)
    q_hip = output["q_hip_rad"].to_numpy(float)
    q_knee = output["q_knee_rad"].to_numpy(float)
    output["theta_shank_rad"] = q_hip - q_knee
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    output["x_knee_m"] = x_knee
    output["z_knee_m"] = z_knee
    output["x_pull_m"] = x_pull
    output["z_pull_m"] = z_pull
    output["model_angle_definition"] = MODEL_ANGLE_DEFINITION
    output["spline_degree"] = model.shape_audit.spline_degree
    output["continuity_order"] = model.shape_audit.continuity_order
    output["internal_spline_continuity_order"] = (
        model.shape_audit.internal_spline_continuity_order
    )
    output["full_cycle_global_phase_continuity_order"] = (
        model.shape_audit.full_cycle_global_phase_continuity_order
    )
    output["reflection_boundary_conditions"] = (
        "flexion_dq_ds_order_1_and_3_equal_zero_at_start_and_peak"
    )
    output["extension_is_exact_reverse_of_flexion"] = True
    output["repeatable_loop"] = True
    output["pointwise_angle_clipping_applied"] = False
    output["rom_mapping_applied"] = False
    output["reference_path_preserved"] = True
    output["source_reference_overwritten"] = False
    output["shape_preserved_within_audit_gates"] = model.shape_audit.fit_accepted
    output["reference_path_preserved_meaning"] = (
        "immutable_source_retained_and_c2_shape_within_explicit_deviation_gates"
    )
    output["source_timing_status"] = "unknown"
    output["retimed_trajectory"] = True
    output["retimed_timing_is_original"] = False
    output["source_trajectory_type"] = "provided_rehabilitation_reference"
    output["simulation_status"] = "software_only"
    output["formal_execution_scope"] = "offline_reference_rom_and_shape_gate_only"
    output["robot_execution_approved"] = False
    output["observed_ankle_is_pull_point"] = False
    output["L1_m"] = L1
    output["L2_m"] = L2
    output["L2_definition"] = "knee_to_strap_equivalent_pull_point"
    output["approved_hip_min_deg"] = model.approved_hip_rom_deg[0]
    output["approved_hip_max_deg"] = model.approved_hip_rom_deg[1]
    output["approved_knee_min_deg"] = model.approved_knee_rom_deg[0]
    output["approved_knee_max_deg"] = model.approved_knee_rom_deg[1]
    hip_valid = (q_hip >= np.deg2rad(model.approved_hip_rom_deg[0]) - 1e-12) & (
        q_hip <= np.deg2rad(model.approved_hip_rom_deg[1]) + 1e-12
    )
    knee_valid = (q_knee >= np.deg2rad(model.approved_knee_rom_deg[0]) - 1e-12) & (
        q_knee <= np.deg2rad(model.approved_knee_rom_deg[1]) + 1e-12
    )
    finite = np.isfinite(
        output[
            [
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
                "x_knee_m",
                "z_knee_m",
                "x_pull_m",
                "z_pull_m",
            ]
        ].to_numpy(float)
    ).all(axis=1)
    output["joint_limit_valid"] = hip_valid & knee_valid
    output["trajectory_sample_valid"] = finite & hip_valid & knee_valid
    output["invalid_reason"] = np.where(
        ~finite,
        "non_finite_c2_state",
        np.where(~(hip_valid & knee_valid), "outside_approved_rom", ""),
    )
    output["formal_execution_allowed"] = bool(
        model.shape_audit.fit_accepted and output["trajectory_sample_valid"].all()
    )

    endpoint_state = output.iloc[[0, -1]][
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(float)
    if not np.allclose(endpoint_state, 0.0, atol=1e-12, rtol=0.0):
        raise C2ReferenceError("C2 retiming did not preserve zero endpoint dq/ddq.")
    if not np.allclose(
        output[["q_hip_rad", "q_knee_rad"]].iloc[0],
        output[["q_hip_rad", "q_knee_rad"]].iloc[-1],
        atol=1e-12,
        rtol=0.0,
    ):
        raise C2ReferenceError("C2 trajectory is not angle-closed.")
    if not np.allclose(
        output["theta_shank_rad"], q_hip - q_knee, atol=1e-14, rtol=0.0
    ):
        raise C2ReferenceError("theta_shank = q_hip - q_knee was not preserved.")
    if not bool(output["trajectory_sample_valid"].all()):
        raise C2ReferenceError("C2 retimed trajectory failed finite/ROM validity.")
    return output


def _robust_warning_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    """Match the Stage-6A dimensionless data-quality outlier diagnostic."""

    array = np.asarray(values, dtype=float)
    result = np.zeros(array.shape, dtype=bool)
    finite = array[np.isfinite(array)]
    if finite.size < 5:
        return result
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, median * 0.05, np.finfo(float).eps)
    limit = max(median * ratio, median + ratio * scale)
    result[np.isfinite(array)] = array[np.isfinite(array)] > limit
    return result


def pull_acceleration_warning_count(
    trajectory: pd.DataFrame,
    *,
    ratio: float = DEFAULT_ACCELERATION_WARNING_RATIO,
) -> int:
    """Count offline pull-acceleration increment outliers (not safety events)."""

    required = {"time_s", "x_pull_m", "z_pull_m"}
    missing = required.difference(trajectory.columns)
    if missing:
        raise C2ReferenceError(f"acceleration audit missing columns: {sorted(missing)}")
    time_s = trajectory["time_s"].to_numpy(float)
    positions = trajectory[["x_pull_m", "z_pull_m"]].to_numpy(float)
    if len(time_s) < 5 or not np.isfinite(np.column_stack((time_s, positions))).all():
        raise C2ReferenceError("acceleration audit requires at least five finite samples.")
    if not np.all(np.diff(time_s) > 0.0):
        raise C2ReferenceError("acceleration audit time must be strictly increasing.")
    velocity = np.column_stack(
        [np.gradient(positions[:, axis], time_s, edge_order=2) for axis in range(2)]
    )
    acceleration = np.column_stack(
        [np.gradient(velocity[:, axis], time_s, edge_order=2) for axis in range(2)]
    )
    increments = np.linalg.norm(np.diff(acceleration, axis=0), axis=1)
    return int(_robust_warning_mask(increments, float(ratio)).sum())


def compare_c2_with_pchip(
    reference_versions: pd.DataFrame,
    model: C2ReferenceModel,
    c2_trajectories: dict[str, pd.DataFrame],
    *,
    durations_s: dict[str, float],
    samples_per_segment: int = 201,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare the C2 trajectories with the retained PCHIP reference."""

    rows: list[dict[str, object]] = []
    originals: dict[str, pd.DataFrame] = {}
    for profile, duration in durations_s.items():
        if profile not in c2_trajectories:
            raise C2ReferenceError(f"missing C2 trajectory for profile {profile!r}.")
        original = retime_closed_reference(
            reference_versions,
            profile=f"reference_closed_symmetric_{profile}",
            flexion_duration_s=float(duration),
            extension_duration_s=float(duration),
            samples_per_segment=samples_per_segment,
        )
        originals[profile] = original
        c2 = c2_trajectories[profile]
        original_warning_count = pull_acceleration_warning_count(original)
        c2_warning_count = pull_acceleration_warning_count(c2)
        rows.append(
            {
                "profile": profile,
                **model.shape_audit.as_dict(),
                "original_acceleration_warning_count": original_warning_count,
                "c2_acceleration_warning_count": c2_warning_count,
                "acceleration_warning_count_reduction": (
                    original_warning_count - c2_warning_count
                ),
                "acceleration_warning_ratio": DEFAULT_ACCELERATION_WARNING_RATIO,
                "acceleration_warning_is_safety_threshold": False,
                "max_abs_dq_hip_rad_s": float(c2["dq_hip_rad_s"].abs().max()),
                "max_abs_dq_knee_rad_s": float(c2["dq_knee_rad_s"].abs().max()),
                "max_abs_ddq_hip_rad_s2": float(c2["ddq_hip_rad_s2"].abs().max()),
                "max_abs_ddq_knee_rad_s2": float(c2["ddq_knee_rad_s2"].abs().max()),
                "q_hip_closure_error_deg": float(
                    np.rad2deg(c2["q_hip_rad"].iloc[-1] - c2["q_hip_rad"].iloc[0])
                ),
                "q_knee_closure_error_deg": float(
                    np.rad2deg(c2["q_knee_rad"].iloc[-1] - c2["q_knee_rad"].iloc[0])
                ),
                "pull_point_closure_error_m": float(
                    np.hypot(
                        c2["x_pull_m"].iloc[-1] - c2["x_pull_m"].iloc[0],
                        c2["z_pull_m"].iloc[-1] - c2["z_pull_m"].iloc[0],
                    )
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    if not comparison["fit_accepted"].astype(bool).all():
        raise C2ReferenceError("C2 fit was not accepted.")
    if (comparison["c2_acceleration_warning_count"].to_numpy(int) > 0).any():
        raise C2ReferenceError(
            "C2 path remains fail-closed because acceleration warnings were not eliminated."
        )
    return comparison, originals


__all__ = [
    "APPROVED_HIP_ROM_DEG",
    "APPROVED_KNEE_ROM_DEG",
    "C2_MODEL_VERSION",
    "C2_REFERENCE",
    "C2ReferenceError",
    "C2ReferenceModel",
    "C2ShapeAudit",
    "compare_c2_with_pchip",
    "fit_reference_closed_c2",
    "pull_acceleration_warning_count",
    "retime_reference_closed_c2",
]
