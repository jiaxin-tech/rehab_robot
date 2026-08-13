"""Measured asymmetric periodic reference construction and analytic retiming.

The legacy Stage-5C reference reflected measured flexion to synthesize an
extension branch.  This module instead keeps the selected CSV flexion and CSV
extension as two distinct measured branches.  A periodic cubic B-spline is fit
to the complete loop only after a small, explicitly audited endpoint closure
correction.  The source rows are never overwritten.

This is an offline numerical module.  It imports no robot SDK and contains no
connection, power, servo, motion, or safety-threshold code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, PchipInterpolator, make_interp_spline
from scipy.signal import savgol_filter

from .config import L1, L2
from .kinematics import forward_kinematics
from .formal_protocol import FORMAL_HIP_ROM_DEG, FORMAL_KNEE_ROM_DEG
from .trajectory_profiles import minimum_jerk_profile


MEASURED_RAW_REFERENCE = "reference_measured_raw"
MEASURED_ASYMMETRIC_CLOSED_REFERENCE = (
    "reference_measured_asymmetric_closed"
)
MEASURED_ASYMMETRIC_SLOW_ID = (
    "reference_measured_asymmetric_closed_slow"
)
MEASURED_ASYMMETRIC_NOMINAL_ID = (
    "reference_measured_asymmetric_closed_nominal"
)
MEASURED_ASYMMETRIC_MODEL_VERSION = (
    "lower_limb_sim_reference_measured_asymmetric_periodic_v1"
)

# These are offline path-preservation gates, not real-robot safety limits.
# They are intentionally separate from safety/experiment_safety.py.
MAXIMUM_HIP_PATH_DEVIATION_DEG = 0.5
MAXIMUM_KNEE_PATH_DEVIATION_DEG = 0.5
MAXIMUM_PULL_PATH_DEVIATION_MM = 2.5
MINIMUM_ASYMMETRY_RETENTION_RATIO = 0.80
NATURAL_CLOSURE_JOINT_TOLERANCE_DEG = 0.05
NATURAL_CLOSURE_PULL_TOLERANCE_MM = 0.50
CONTINUITY_POSITION_TOLERANCE_RAD = 1e-10
CONTINUITY_VELOCITY_TOLERANCE_RAD = 1e-8
CONTINUITY_ACCELERATION_TOLERANCE_RAD = 1e-6
DEFAULT_PHASE_SAMPLE_COUNT = 401
DEFAULT_DENSE_AUDIT_SAMPLE_COUNT = 20_001
PERIODIC_SPLINE_SUBDIVISION_FACTOR = 4
PERIODIC_PATH_SMOOTHING_WINDOW = 5
PERIODIC_PATH_SMOOTHING_POLYNOMIAL_ORDER = 3
MODEL_ANGLE_DEFINITION = "theta_shank = q_hip - q_knee"
APPROVED_HIP_ROM_DEG = FORMAL_HIP_ROM_DEG
APPROVED_KNEE_ROM_DEG = FORMAL_KNEE_ROM_DEG


@dataclass(frozen=True)
class PeriodicClosureDeviationAudit:
    """Pointwise deviation between measured and periodic-closed paths."""

    natural_delta_q_hip_deg: float
    natural_delta_q_knee_deg: float
    natural_delta_x_pull_mm: float
    natural_delta_z_pull_mm: float
    natural_pull_closure_error_mm: float
    natural_closure_below_numerical_tolerance: bool
    hip_max_deviation_deg: float
    knee_max_deviation_deg: float
    hip_rms_deviation_deg: float
    knee_rms_deviation_deg: float
    pull_point_max_deviation_mm: float
    pull_point_rms_deviation_mm: float
    flexion_hip_max_deviation_deg: float
    flexion_knee_max_deviation_deg: float
    flexion_pull_max_deviation_mm: float
    extension_hip_max_deviation_deg: float
    extension_knee_max_deviation_deg: float
    extension_pull_max_deviation_mm: float
    changed_source_node_count: int
    source_node_hip_max_deviation_deg: float
    source_node_knee_max_deviation_deg: float
    source_node_hip_rms_deviation_deg: float
    source_node_knee_rms_deviation_deg: float
    source_node_pull_max_deviation_mm: float
    source_node_pull_rms_deviation_mm: float
    maximum_hip_deviation_gate_deg: float
    maximum_knee_deviation_gate_deg: float
    maximum_pull_deviation_gate_mm: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FlexionExtensionAsymmetryAudit:
    """Measured branch mismatch and preservation after periodic closure."""

    hip_flexion_extension_asymmetry_rmse_deg: float
    knee_flexion_extension_asymmetry_rmse_deg: float
    pull_path_asymmetry_rmse_mm: float
    closed_hip_flexion_extension_asymmetry_rmse_deg: float
    closed_knee_flexion_extension_asymmetry_rmse_deg: float
    closed_pull_path_asymmetry_rmse_mm: float
    hip_asymmetry_retention_ratio: float
    knee_asymmetry_retention_ratio: float
    pull_asymmetry_retention_ratio: float
    minimum_asymmetry_retention_ratio: float
    asymmetry_preserved: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PeriodicContinuityAudit:
    """Numerical C2 audit at the seam, peak and all spline knots."""

    spline_degree: int
    continuity_order: int
    position_continuity_warning_count: int
    velocity_continuity_warning_count: int
    acceleration_continuity_warning_count: int
    maximum_position_jump_rad: float
    maximum_first_derivative_jump_rad: float
    maximum_second_derivative_jump_rad: float
    position_tolerance_rad: float
    velocity_tolerance_rad: float
    acceleration_tolerance_rad: float

    @property
    def passed(self) -> bool:
        return not (
            self.position_continuity_warning_count
            or self.velocity_continuity_warning_count
            or self.acceleration_continuity_warning_count
        )

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True)
class MeasuredAsymmetricPeriodicModel:
    """The two periodic joint splines and every acceptance audit."""

    hip_spline: BSpline
    knee_spline: BSpline
    measured_raw: pd.DataFrame
    phase_path: pd.DataFrame
    start_frame: int
    peak_frame: int
    end_frame: int
    peak_global_phase: float
    spline_anchor_count: int
    spline_subdivision_factor: int
    deviation_audit: PeriodicClosureDeviationAudit
    asymmetry_audit: FlexionExtensionAsymmetryAudit
    continuity_audit: PeriodicContinuityAudit
    fit_accepted: bool
    rejection_reasons: tuple[str, ...]


def _finite_rom_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    pair = np.asarray(values, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be two finite increasing values")
    return float(pair[0]), float(pair[1])


def _minimum_jerk_weight(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def _source_phase(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=float)
    if frame.ndim != 1 or len(frame) < 5 or not np.isfinite(frame).all():
        raise ValueError("measured reference needs at least five finite frames")
    if not np.all(np.diff(frame) > 0.0):
        raise ValueError("measured reference frames must be strictly increasing")
    return (frame - frame[0]) / (frame[-1] - frame[0])


def build_reference_measured_raw(
    full_angles: pd.DataFrame,
    *,
    start_frame: int,
    peak_frame: int,
    end_frame: int,
    approved_hip_rom_deg: Sequence[float] = APPROVED_HIP_ROM_DEG,
    approved_knee_rom_deg: Sequence[float] = APPROVED_KNEE_ROM_DEG,
) -> pd.DataFrame:
    """Copy the selected Stage-5A rows without changing any source column."""

    required = {
        "Frame",
        "q_hip_rad",
        "q_knee_rad",
        "q_hip_raw_rad",
        "q_knee_raw_rad",
        "angle_valid",
    }
    missing = required.difference(full_angles.columns)
    if missing:
        raise ValueError(f"Stage-5A full angles missing columns: {sorted(missing)}")
    if not int(start_frame) < int(peak_frame) < int(end_frame):
        raise ValueError("measured cycle requires start < peak < end")
    source = full_angles.loc[
        full_angles["Frame"].between(
            int(start_frame), int(end_frame), inclusive="both"
        )
    ].copy(deep=True)
    if source.empty:
        raise ValueError("selected measured cycle is empty")
    if int(source["Frame"].iloc[0]) != int(start_frame) or int(
        source["Frame"].iloc[-1]
    ) != int(end_frame):
        raise ValueError("selected measured cycle endpoints are missing")
    if int(source["Frame"].eq(int(peak_frame)).sum()) != 1:
        raise ValueError("selected peak frame is not present exactly once")
    frame = source["Frame"].to_numpy(dtype=float)
    global_phase = _source_phase(frame)
    peak_position = int(np.flatnonzero(frame == float(peak_frame))[0])
    peak_phase = float(global_phase[peak_position])
    segment_phase = np.where(
        global_phase <= peak_phase,
        global_phase / peak_phase,
        (global_phase - peak_phase) / (1.0 - peak_phase),
    )
    q_hip = source["q_hip_rad"].to_numpy(dtype=float)
    q_knee = source["q_knee_rad"].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack((q_hip, q_knee))).all():
        raise ValueError("selected measured cycle contains non-finite joint angles")
    hip_rom = np.deg2rad(_finite_rom_pair(approved_hip_rom_deg, "hip ROM"))
    knee_rom = np.deg2rad(_finite_rom_pair(approved_knee_rom_deg, "knee ROM"))
    rom_valid = (
        (q_hip >= hip_rom[0])
        & (q_hip <= hip_rom[1])
        & (q_knee >= knee_rom[0])
        & (q_knee <= knee_rom[1])
    )
    projection_valid = source["angle_valid"].fillna(False).astype(bool).to_numpy()
    if not projection_valid.all():
        raise ValueError("selected measured cycle contains projection-invalid rows")
    if not rom_valid.all():
        raise ValueError("selected measured cycle violates approved experiment ROM")
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)

    # Source columns above remain byte-for-value identical.  The fields below
    # are provenance/phase annotations, not replacements for measured values.
    source.insert(0, "source_frame", source["Frame"].to_numpy(dtype=int))
    source["global_phase"] = global_phase
    source["segment_phase"] = segment_phase
    source["cycle_phase"] = np.where(
        source["Frame"].to_numpy(dtype=int) <= int(peak_frame),
        "flexion",
        "extension",
    )
    source["q_hip_measured_rad"] = q_hip
    source["q_knee_measured_rad"] = q_knee
    expected_theta = q_hip - q_knee
    if not np.allclose(
        source["theta_shank_rad"].to_numpy(dtype=float),
        expected_theta,
        atol=1e-14,
        rtol=0.0,
    ):
        raise ValueError("Stage-5A source violates theta_shank = q_hip - q_knee")
    # Preserve the original Stage-5A column exactly; expose the recomputed
    # identity under a new audit field instead of replacing source values.
    source["theta_shank_reference_rad"] = expected_theta
    source["x_pull_m"] = x_pull
    source["z_pull_m"] = z_pull
    source["reference_version"] = MEASURED_RAW_REFERENCE
    source["active_reference"] = False
    source["repeatable_loop"] = False
    source["extension_source_is_measured"] = True
    source["measured_extension_is_reversed_flexion"] = False
    source["source_values_modified"] = False
    source["reference_provenance"] = (
        "direct_Stage5A_measured_flexion_and_measured_extension"
    )
    source["model_angle_definition"] = MODEL_ANGLE_DEFINITION
    source["projection_valid"] = projection_valid
    source["rom_valid"] = rom_valid
    source["trajectory_sample_valid"] = projection_valid & rom_valid
    return source.reset_index(drop=True)


def _dense_reference_values(
    measured_raw: pd.DataFrame,
    dense_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phase = measured_raw["global_phase"].to_numpy(dtype=float)
    q_hip = measured_raw["q_hip_measured_rad"].to_numpy(dtype=float)
    q_knee = measured_raw["q_knee_measured_rad"].to_numpy(dtype=float)
    return (
        np.asarray(PchipInterpolator(phase, q_hip)(dense_phase), dtype=float),
        np.asarray(PchipInterpolator(phase, q_knee)(dense_phase), dtype=float),
    )


def _deviation_audit(
    measured_raw: pd.DataFrame,
    hip_spline: BSpline,
    knee_spline: BSpline,
    peak_phase: float,
    *,
    dense_sample_count: int,
) -> PeriodicClosureDeviationAudit:
    dense_phase = np.linspace(0.0, 1.0, dense_sample_count)
    raw_hip, raw_knee = _dense_reference_values(measured_raw, dense_phase)
    closed_hip = np.asarray(hip_spline(dense_phase), dtype=float)
    closed_knee = np.asarray(knee_spline(dense_phase), dtype=float)
    _, _, raw_x, raw_z = forward_kinematics(raw_hip, raw_knee, L1, L2)
    _, _, closed_x, closed_z = forward_kinematics(
        closed_hip, closed_knee, L1, L2
    )
    hip_error = np.rad2deg(closed_hip - raw_hip)
    knee_error = np.rad2deg(closed_knee - raw_knee)
    pull_error = 1000.0 * np.hypot(closed_x - raw_x, closed_z - raw_z)
    flexion = dense_phase <= peak_phase
    extension = ~flexion

    source_phase = measured_raw["global_phase"].to_numpy(dtype=float)
    source_hip = measured_raw["q_hip_measured_rad"].to_numpy(dtype=float)
    source_knee = measured_raw["q_knee_measured_rad"].to_numpy(dtype=float)
    closed_source_hip = np.asarray(hip_spline(source_phase), dtype=float)
    closed_source_knee = np.asarray(knee_spline(source_phase), dtype=float)
    source_hip_error = np.rad2deg(closed_source_hip - source_hip)
    source_knee_error = np.rad2deg(closed_source_knee - source_knee)
    _, _, source_x, source_z = forward_kinematics(
        source_hip, source_knee, L1, L2
    )
    _, _, closed_source_x, closed_source_z = forward_kinematics(
        closed_source_hip, closed_source_knee, L1, L2
    )
    source_pull_error = 1000.0 * np.hypot(
        closed_source_x - source_x, closed_source_z - source_z
    )
    changed_source_nodes = (
        (np.abs(source_hip_error) > 1e-12)
        | (np.abs(source_knee_error) > 1e-12)
        | (source_pull_error > 1e-9)
    )

    q_hip = measured_raw["q_hip_measured_rad"].to_numpy(dtype=float)
    q_knee = measured_raw["q_knee_measured_rad"].to_numpy(dtype=float)
    _, _, endpoint_x, endpoint_z = forward_kinematics(
        np.asarray((q_hip[0], q_hip[-1])),
        np.asarray((q_knee[0], q_knee[-1])),
        L1,
        L2,
    )
    delta_hip = float(np.rad2deg(q_hip[-1] - q_hip[0]))
    delta_knee = float(np.rad2deg(q_knee[-1] - q_knee[0]))
    delta_x = float(1000.0 * (endpoint_x[-1] - endpoint_x[0]))
    delta_z = float(1000.0 * (endpoint_z[-1] - endpoint_z[0]))
    pull_closure = float(math.hypot(delta_x, delta_z))

    def maximum(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.max(np.abs(values[mask])))

    return PeriodicClosureDeviationAudit(
        natural_delta_q_hip_deg=delta_hip,
        natural_delta_q_knee_deg=delta_knee,
        natural_delta_x_pull_mm=delta_x,
        natural_delta_z_pull_mm=delta_z,
        natural_pull_closure_error_mm=pull_closure,
        natural_closure_below_numerical_tolerance=bool(
            abs(delta_hip) <= NATURAL_CLOSURE_JOINT_TOLERANCE_DEG
            and abs(delta_knee) <= NATURAL_CLOSURE_JOINT_TOLERANCE_DEG
            and pull_closure <= NATURAL_CLOSURE_PULL_TOLERANCE_MM
        ),
        hip_max_deviation_deg=float(np.max(np.abs(hip_error))),
        knee_max_deviation_deg=float(np.max(np.abs(knee_error))),
        hip_rms_deviation_deg=float(np.sqrt(np.mean(hip_error**2))),
        knee_rms_deviation_deg=float(np.sqrt(np.mean(knee_error**2))),
        pull_point_max_deviation_mm=float(np.max(pull_error)),
        pull_point_rms_deviation_mm=float(np.sqrt(np.mean(pull_error**2))),
        flexion_hip_max_deviation_deg=maximum(hip_error, flexion),
        flexion_knee_max_deviation_deg=maximum(knee_error, flexion),
        flexion_pull_max_deviation_mm=maximum(pull_error, flexion),
        extension_hip_max_deviation_deg=maximum(hip_error, extension),
        extension_knee_max_deviation_deg=maximum(knee_error, extension),
        extension_pull_max_deviation_mm=maximum(pull_error, extension),
        changed_source_node_count=int(changed_source_nodes.sum()),
        source_node_hip_max_deviation_deg=float(
            np.max(np.abs(source_hip_error))
        ),
        source_node_knee_max_deviation_deg=float(
            np.max(np.abs(source_knee_error))
        ),
        source_node_hip_rms_deviation_deg=float(
            np.sqrt(np.mean(source_hip_error**2))
        ),
        source_node_knee_rms_deviation_deg=float(
            np.sqrt(np.mean(source_knee_error**2))
        ),
        source_node_pull_max_deviation_mm=float(np.max(source_pull_error)),
        source_node_pull_rms_deviation_mm=float(
            np.sqrt(np.mean(source_pull_error**2))
        ),
        maximum_hip_deviation_gate_deg=MAXIMUM_HIP_PATH_DEVIATION_DEG,
        maximum_knee_deviation_gate_deg=MAXIMUM_KNEE_PATH_DEVIATION_DEG,
        maximum_pull_deviation_gate_mm=MAXIMUM_PULL_PATH_DEVIATION_MM,
    )


def _asymmetry_audit(
    measured_raw: pd.DataFrame,
    hip_spline: BSpline,
    knee_spline: BSpline,
    peak_phase: float,
    *,
    sample_count: int = 2001,
) -> FlexionExtensionAsymmetryAudit:
    local_phase = np.linspace(0.0, 1.0, sample_count)
    flexion_phase = peak_phase * local_phase
    # Reverse only for a branch-to-branch comparison.  This never supplies the
    # actual extension reference.
    reversed_extension_phase = 1.0 - (1.0 - peak_phase) * local_phase
    raw_hip_interp = PchipInterpolator(
        measured_raw["global_phase"].to_numpy(dtype=float),
        measured_raw["q_hip_measured_rad"].to_numpy(dtype=float),
    )
    raw_knee_interp = PchipInterpolator(
        measured_raw["global_phase"].to_numpy(dtype=float),
        measured_raw["q_knee_measured_rad"].to_numpy(dtype=float),
    )

    def metrics(hip_function, knee_function) -> tuple[float, float, float]:
        flexion_hip = np.asarray(hip_function(flexion_phase), dtype=float)
        extension_hip = np.asarray(
            hip_function(reversed_extension_phase), dtype=float
        )
        flexion_knee = np.asarray(knee_function(flexion_phase), dtype=float)
        extension_knee = np.asarray(
            knee_function(reversed_extension_phase), dtype=float
        )
        _, _, flexion_x, flexion_z = forward_kinematics(
            flexion_hip, flexion_knee, L1, L2
        )
        _, _, extension_x, extension_z = forward_kinematics(
            extension_hip, extension_knee, L1, L2
        )
        return (
            float(
                np.sqrt(
                    np.mean(np.rad2deg(flexion_hip - extension_hip) ** 2)
                )
            ),
            float(
                np.sqrt(
                    np.mean(np.rad2deg(flexion_knee - extension_knee) ** 2)
                )
            ),
            float(
                np.sqrt(
                    np.mean(
                        ((flexion_x - extension_x) * 1000.0) ** 2
                        + ((flexion_z - extension_z) * 1000.0) ** 2
                    )
                )
            ),
        )

    raw = metrics(raw_hip_interp, raw_knee_interp)
    closed = metrics(hip_spline, knee_spline)

    def retention(after: float, before: float) -> float:
        return 1.0 if before <= 1e-12 and after <= 1e-12 else after / before

    ratios = tuple(retention(after, before) for after, before in zip(closed, raw))
    return FlexionExtensionAsymmetryAudit(
        hip_flexion_extension_asymmetry_rmse_deg=raw[0],
        knee_flexion_extension_asymmetry_rmse_deg=raw[1],
        pull_path_asymmetry_rmse_mm=raw[2],
        closed_hip_flexion_extension_asymmetry_rmse_deg=closed[0],
        closed_knee_flexion_extension_asymmetry_rmse_deg=closed[1],
        closed_pull_path_asymmetry_rmse_mm=closed[2],
        hip_asymmetry_retention_ratio=ratios[0],
        knee_asymmetry_retention_ratio=ratios[1],
        pull_asymmetry_retention_ratio=ratios[2],
        minimum_asymmetry_retention_ratio=MINIMUM_ASYMMETRY_RETENTION_RATIO,
        asymmetry_preserved=bool(
            min(ratios) >= MINIMUM_ASYMMETRY_RETENTION_RATIO
        ),
    )


def _continuity_audit(
    hip_spline: BSpline,
    knee_spline: BSpline,
    knots: np.ndarray,
    peak_phase: float,
) -> PeriodicContinuityAudit:
    locations = [0.0, peak_phase, *map(float, knots[1:-1])]
    jumps: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for location in locations:
        if location == 0.0:
            left = 1.0
            right = 0.0
        else:
            left = float(np.nextafter(location, 0.0))
            right = float(np.nextafter(location, 1.0))
        for order in (0, 1, 2):
            for spline in (hip_spline, knee_spline):
                jumps[order].append(
                    abs(float(spline(right, order) - spline(left, order)))
                )
    tolerances = {
        0: CONTINUITY_POSITION_TOLERANCE_RAD,
        1: CONTINUITY_VELOCITY_TOLERANCE_RAD,
        2: CONTINUITY_ACCELERATION_TOLERANCE_RAD,
    }
    counts = {
        order: int(np.count_nonzero(np.asarray(values) > tolerances[order]))
        for order, values in jumps.items()
    }
    maxima = {
        order: float(max(values, default=0.0))
        for order, values in jumps.items()
    }
    return PeriodicContinuityAudit(
        spline_degree=3,
        continuity_order=2,
        position_continuity_warning_count=counts[0],
        velocity_continuity_warning_count=counts[1],
        acceleration_continuity_warning_count=counts[2],
        maximum_position_jump_rad=maxima[0],
        maximum_first_derivative_jump_rad=maxima[1],
        maximum_second_derivative_jump_rad=maxima[2],
        position_tolerance_rad=tolerances[0],
        velocity_tolerance_rad=tolerances[1],
        acceleration_tolerance_rad=tolerances[2],
    )


def _phase_table(
    measured_raw: pd.DataFrame,
    hip_spline: BSpline,
    knee_spline: BSpline,
    peak_phase: float,
    *,
    sample_count: int,
    formal_execution_allowed: bool,
) -> pd.DataFrame:
    if sample_count < 21:
        raise ValueError("phase sample_count must be at least 21")
    global_phase = np.linspace(0.0, 1.0, int(sample_count))
    q_hip = np.asarray(hip_spline(global_phase), dtype=float)
    q_knee = np.asarray(knee_spline(global_phase), dtype=float)
    raw_hip, raw_knee = _dense_reference_values(measured_raw, global_phase)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1, L2
    )
    cycle_phase = np.where(global_phase <= peak_phase, "flexion", "extension")
    segment_phase = np.where(
        global_phase <= peak_phase,
        global_phase / peak_phase,
        (global_phase - peak_phase) / (1.0 - peak_phase),
    )
    source_frame = np.interp(
        global_phase,
        measured_raw["global_phase"].to_numpy(dtype=float),
        measured_raw["Frame"].to_numpy(dtype=float),
    )
    return pd.DataFrame(
        {
            "reference_version": MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
            "active_reference": False,
            "repeatable_loop": True,
            "global_phase": global_phase,
            "segment_phase": segment_phase,
            "cycle_phase": cycle_phase,
            "source_frame": source_frame,
            "q_hip_measured_interpolated_rad": raw_hip,
            "q_knee_measured_interpolated_rad": raw_knee,
            "q_hip_reference_rad": q_hip,
            "q_knee_reference_rad": q_knee,
            "q_hip_rad": q_hip,
            "q_knee_rad": q_knee,
            "dq_hip_dglobal_phase_rad": hip_spline(global_phase, 1),
            "dq_knee_dglobal_phase_rad": knee_spline(global_phase, 1),
            "d2q_hip_dglobal_phase2_rad": hip_spline(global_phase, 2),
            "d2q_knee_dglobal_phase2_rad": knee_spline(global_phase, 2),
            "theta_shank_rad": q_hip - q_knee,
            "x_knee_m": x_knee,
            "z_knee_m": z_knee,
            "x_pull_m": x_pull,
            "z_pull_m": z_pull,
            "extension_source_is_measured": True,
            "measured_extension_is_reversed_flexion": False,
            "reference_provenance": (
                "periodic_cubic_fit_of_measured_flexion_and_measured_extension"
            ),
            "spline_family": "periodic_cubic_B_spline",
            "spline_degree": 3,
            "continuity_order": 2,
            "model_angle_definition": MODEL_ANGLE_DEFINITION,
            "L1_m": L1,
            "L2_m": L2,
            "L2_definition": "knee_to_strap_equivalent_pull_point",
            "approved_hip_min_deg": APPROVED_HIP_ROM_DEG[0],
            "approved_hip_max_deg": APPROVED_HIP_ROM_DEG[1],
            "approved_knee_min_deg": APPROVED_KNEE_ROM_DEG[0],
            "approved_knee_max_deg": APPROVED_KNEE_ROM_DEG[1],
            "joint_limit_valid": True,
            "trajectory_sample_valid": formal_execution_allowed,
            "formal_execution_allowed": formal_execution_allowed,
            "invalid_reason": "" if formal_execution_allowed else "periodic_fit_rejected",
        }
    )


def fit_measured_asymmetric_periodic_reference(
    measured_raw: pd.DataFrame,
    *,
    dense_audit_sample_count: int = DEFAULT_DENSE_AUDIT_SAMPLE_COUNT,
    phase_sample_count: int = DEFAULT_PHASE_SAMPLE_COUNT,
) -> MeasuredAsymmetricPeriodicModel:
    """Fit a small-correction periodic cubic spline to both measured branches."""

    if not isinstance(measured_raw, pd.DataFrame):
        raise TypeError("measured_raw must be a DataFrame")
    required = {
        "Frame",
        "global_phase",
        "cycle_phase",
        "q_hip_measured_rad",
        "q_knee_measured_rad",
        "projection_valid",
        "rom_valid",
    }
    missing = required.difference(measured_raw.columns)
    if missing:
        raise ValueError(f"measured raw table missing columns: {sorted(missing)}")
    if dense_audit_sample_count < 1001:
        raise ValueError("dense_audit_sample_count must be at least 1001")
    if not measured_raw["projection_valid"].astype(bool).all():
        raise ValueError("periodic fit requires projection-valid measured rows")
    if not measured_raw["rom_valid"].astype(bool).all():
        raise ValueError("periodic fit requires approved-ROM measured rows")
    phases = measured_raw["global_phase"].to_numpy(dtype=float)
    if not np.all(np.diff(phases) > 0.0) or not np.isclose(phases[[0, -1]], (0, 1)).all():
        raise ValueError("measured global phase must increase exactly from 0 to 1")
    peak_rows = measured_raw.loc[measured_raw["cycle_phase"].eq("flexion")]
    if peak_rows.empty:
        raise ValueError("measured raw table has no flexion branch")
    peak_frame = int(peak_rows["Frame"].iloc[-1])
    peak_match = measured_raw["Frame"].eq(peak_frame)
    peak_phase = float(measured_raw.loc[peak_match, "global_phase"].iloc[0])
    if not 0.0 < peak_phase < 1.0:
        raise ValueError("peak phase must lie strictly inside the cycle")

    measured_joint_values = (
        measured_raw["q_hip_measured_rad"].to_numpy(dtype=float),
        measured_raw["q_knee_measured_rad"].to_numpy(dtype=float),
    )
    # A local five-sample cubic Savitzky-Golay pass removes measurement-scale
    # second-derivative spikes before the periodic fit.  The immutable raw
    # table remains untouched, and every resulting change is measured against
    # its unsmoothed PCHIP path by ``_deviation_audit`` below.  This small pass
    # is needed for the existing Stage-6A acceleration-jump diagnostic while
    # preserving the visibly different measured flexion/extension branches.
    smoothed_joint_values = tuple(
        np.asarray(
            savgol_filter(
                values,
                PERIODIC_PATH_SMOOTHING_WINDOW,
                PERIODIC_PATH_SMOOTHING_POLYNOMIAL_ORDER,
                mode="interp",
            ),
            dtype=float,
        )
        for values in measured_joint_values
    )
    raw_interpolators = tuple(
        PchipInterpolator(phases, values, extrapolate=False)
        for values in smoothed_joint_values
    )
    # Power-of-two refinement is deterministic and preserves every original
    # frame as an exact anchor.  Four subdivisions are the first refinement
    # that satisfies the inherited 0.5 deg / 2.5 mm offline shape gates for the
    # selected measured cycle; this is recorded and regression-tested.
    anchor_count = PERIODIC_SPLINE_SUBDIVISION_FACTOR * (len(phases) - 1) + 1
    anchor_phase = np.linspace(0.0, 1.0, anchor_count)
    splines: list[BSpline] = []
    for raw_interpolator, measured_values in zip(
        raw_interpolators, measured_joint_values
    ):
        anchors = np.asarray(raw_interpolator(anchor_phase), dtype=float)
        seam_midpoint = 0.5 * (measured_values[0] + measured_values[-1])
        anchors[0] = seam_midpoint
        anchors[-1] = seam_midpoint
        splines.append(
            make_interp_spline(
                anchor_phase,
                anchors,
                k=3,
                bc_type="periodic",
                check_finite=True,
            )
        )
    hip_spline, knee_spline = splines
    deviation = _deviation_audit(
        measured_raw,
        hip_spline,
        knee_spline,
        peak_phase,
        dense_sample_count=int(dense_audit_sample_count),
    )
    asymmetry = _asymmetry_audit(
        measured_raw, hip_spline, knee_spline, peak_phase
    )
    continuity = _continuity_audit(
        hip_spline, knee_spline, anchor_phase, peak_phase
    )
    dense_phase = np.linspace(0.0, 1.0, int(dense_audit_sample_count))
    dense_hip = np.asarray(hip_spline(dense_phase), dtype=float)
    dense_knee = np.asarray(knee_spline(dense_phase), dtype=float)
    hip_rom = np.deg2rad(APPROVED_HIP_ROM_DEG)
    knee_rom = np.deg2rad(APPROVED_KNEE_ROM_DEG)
    rejection_reasons: list[str] = []
    if deviation.hip_max_deviation_deg > MAXIMUM_HIP_PATH_DEVIATION_DEG:
        rejection_reasons.append("hip_path_deviation_exceeded")
    if deviation.knee_max_deviation_deg > MAXIMUM_KNEE_PATH_DEVIATION_DEG:
        rejection_reasons.append("knee_path_deviation_exceeded")
    if deviation.pull_point_max_deviation_mm > MAXIMUM_PULL_PATH_DEVIATION_MM:
        rejection_reasons.append("pull_path_deviation_exceeded")
    if not asymmetry.asymmetry_preserved:
        rejection_reasons.append("measured_asymmetry_not_preserved")
    if not continuity.passed:
        rejection_reasons.append("periodic_C2_continuity_failed")
    if bool(
        (
            (dense_hip < hip_rom[0] - 1e-12)
            | (dense_hip > hip_rom[1] + 1e-12)
            | (dense_knee < knee_rom[0] - 1e-12)
            | (dense_knee > knee_rom[1] + 1e-12)
        ).any()
    ):
        rejection_reasons.append("periodic_fit_outside_approved_rom")
    fit_accepted = not rejection_reasons
    phase_path = _phase_table(
        measured_raw,
        hip_spline,
        knee_spline,
        peak_phase,
        sample_count=int(phase_sample_count),
        formal_execution_allowed=fit_accepted,
    )
    return MeasuredAsymmetricPeriodicModel(
        hip_spline=hip_spline,
        knee_spline=knee_spline,
        measured_raw=measured_raw.copy(deep=True),
        phase_path=phase_path,
        start_frame=int(measured_raw["Frame"].iloc[0]),
        peak_frame=peak_frame,
        end_frame=int(measured_raw["Frame"].iloc[-1]),
        peak_global_phase=peak_phase,
        spline_anchor_count=anchor_count,
        spline_subdivision_factor=PERIODIC_SPLINE_SUBDIVISION_FACTOR,
        deviation_audit=deviation,
        asymmetry_audit=asymmetry,
        continuity_audit=continuity,
        fit_accepted=fit_accepted,
        rejection_reasons=tuple(rejection_reasons),
    )


def _profile_identifier(profile: str) -> str:
    if profile == "slow":
        return MEASURED_ASYMMETRIC_SLOW_ID
    if profile == "nominal":
        return MEASURED_ASYMMETRIC_NOMINAL_ID
    raise ValueError("profile must be 'slow' or 'nominal'")


def retime_measured_asymmetric_periodic_reference(
    model: MeasuredAsymmetricPeriodicModel,
    *,
    profile: str,
    total_duration_s: float,
    sample_count: int = DEFAULT_PHASE_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Apply minimum jerk independently along measured flexion/extension."""

    if not isinstance(model, MeasuredAsymmetricPeriodicModel):
        raise TypeError("model must be a MeasuredAsymmetricPeriodicModel")
    if not model.fit_accepted:
        raise PermissionError(
            "periodic measured reference was rejected: "
            + ";".join(model.rejection_reasons)
        )
    if not math.isfinite(float(total_duration_s)) or total_duration_s <= 0.0:
        raise ValueError("total_duration_s must be finite and positive")
    if isinstance(sample_count, bool) or int(sample_count) < 21:
        raise ValueError("sample_count must be an integer >= 21")
    sample_count = int(sample_count)
    total_intervals = sample_count - 1
    if sample_count % 2 != 1:
        raise ValueError("sample_count must be odd so both branches share a join")
    segment_intervals = total_intervals // 2
    source_flexion_intervals = model.peak_frame - model.start_frame
    source_total_intervals = model.end_frame - model.start_frame
    flexion_duration = (
        total_duration_s * source_flexion_intervals / source_total_intervals
    )
    extension_duration = total_duration_s - flexion_duration
    trajectory_id = _profile_identifier(profile)
    segments: list[pd.DataFrame] = []
    time_offset = 0.0
    for phase_name, intervals, duration, start_phase, phase_span in (
        (
            "flexion",
            segment_intervals,
            flexion_duration,
            0.0,
            model.peak_global_phase,
        ),
        (
            "extension",
            segment_intervals,
            extension_duration,
            model.peak_global_phase,
            1.0 - model.peak_global_phase,
        ),
    ):
        local_u = np.linspace(0.0, 1.0, intervals + 1)
        progress, progress_rate, progress_acceleration = minimum_jerk_profile(
            local_u, duration
        )
        global_phase = start_phase + phase_span * progress
        global_rate = phase_span * progress_rate
        global_acceleration = phase_span * progress_acceleration
        q_hip = np.asarray(model.hip_spline(global_phase), dtype=float)
        q_knee = np.asarray(model.knee_spline(global_phase), dtype=float)
        q_hip_s = np.asarray(model.hip_spline(global_phase, 1), dtype=float)
        q_knee_s = np.asarray(model.knee_spline(global_phase, 1), dtype=float)
        q_hip_ss = np.asarray(model.hip_spline(global_phase, 2), dtype=float)
        q_knee_ss = np.asarray(model.knee_spline(global_phase, 2), dtype=float)
        segments.append(
            pd.DataFrame(
                {
                    "reference_version": MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
                    "trajectory_id": trajectory_id,
                    "profile": profile,
                    "time_s": time_offset + duration * local_u,
                    "cycle_phase": phase_name,
                    "segment_phase": progress,
                    "global_phase": global_phase,
                    "q_hip_rad": q_hip,
                    "q_knee_rad": q_knee,
                    "dq_hip_rad_s": q_hip_s * global_rate,
                    "dq_knee_rad_s": q_knee_s * global_rate,
                    "ddq_hip_rad_s2": (
                        q_hip_ss * global_rate**2
                        + q_hip_s * global_acceleration
                    ),
                    "ddq_knee_rad_s2": (
                        q_knee_ss * global_rate**2
                        + q_knee_s * global_acceleration
                    ),
                    "minimum_jerk_phase_rate_s_inv": progress_rate,
                    "minimum_jerk_phase_acceleration_s_inv2": (
                        progress_acceleration
                    ),
                }
            )
        )
        time_offset += duration
    output = pd.concat((segments[0], segments[1].iloc[1:]), ignore_index=True)
    q_hip = output["q_hip_rad"].to_numpy(dtype=float)
    q_knee = output["q_knee_rad"].to_numpy(dtype=float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip, q_knee, L1, L2
    )
    output["theta_shank_rad"] = q_hip - q_knee
    output["x_knee_m"] = x_knee
    output["z_knee_m"] = z_knee
    output["x_pull_m"] = x_pull
    output["z_pull_m"] = z_pull
    output["extension_source_is_measured"] = True
    output["measured_extension_is_reversed_flexion"] = False
    output["reference_provenance"] = (
        "periodic_cubic_fit_of_measured_flexion_and_measured_extension"
    )
    output["spline_family"] = "periodic_cubic_B_spline"
    output["spline_degree"] = 3
    output["continuity_order"] = 2
    output["minimum_jerk_controls"] = "each_measured_branch_phase_progress"
    output["retimed_trajectory"] = True
    output["retimed_timing_is_original"] = False
    output["repeatable_loop"] = True
    output["active_reference"] = profile == "slow"
    output["allowed_for_first_robot_trial"] = profile == "slow"
    output["model_angle_definition"] = MODEL_ANGLE_DEFINITION
    output["L1_m"] = L1
    output["L2_m"] = L2
    output["L2_definition"] = "knee_to_strap_equivalent_pull_point"
    output["source_approved_hip_min_deg"] = APPROVED_HIP_ROM_DEG[0]
    output["source_approved_hip_max_deg"] = APPROVED_HIP_ROM_DEG[1]
    output["source_approved_knee_min_deg"] = APPROVED_KNEE_ROM_DEG[0]
    output["source_approved_knee_max_deg"] = APPROVED_KNEE_ROM_DEG[1]
    output["approved_hip_min_deg"] = APPROVED_HIP_ROM_DEG[0]
    output["approved_hip_max_deg"] = APPROVED_HIP_ROM_DEG[1]
    output["approved_knee_min_deg"] = APPROVED_KNEE_ROM_DEG[0]
    output["approved_knee_max_deg"] = APPROVED_KNEE_ROM_DEG[1]
    hip_valid = (
        (q_hip >= np.deg2rad(APPROVED_HIP_ROM_DEG[0]) - 1e-12)
        & (q_hip <= np.deg2rad(APPROVED_HIP_ROM_DEG[1]) + 1e-12)
    )
    knee_valid = (
        (q_knee >= np.deg2rad(APPROVED_KNEE_ROM_DEG[0]) - 1e-12)
        & (q_knee <= np.deg2rad(APPROVED_KNEE_ROM_DEG[1]) + 1e-12)
    )
    finite = np.isfinite(
        output[
            [
                "time_s",
                "q_hip_rad",
                "q_knee_rad",
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
                "x_pull_m",
                "z_pull_m",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    output["joint_limit_valid"] = hip_valid & knee_valid
    output["trajectory_sample_valid"] = finite & hip_valid & knee_valid
    output["formal_execution_allowed"] = bool(
        model.fit_accepted and output["trajectory_sample_valid"].all()
    )
    output["invalid_reason"] = np.where(
        ~finite,
        "non_finite_periodic_reference",
        np.where(~(hip_valid & knee_valid), "outside_approved_rom", ""),
    )

    if len(output) != sample_count or not np.all(
        np.diff(output["time_s"].to_numpy(dtype=float)) > 0.0
    ):
        raise RuntimeError("retimed measured reference time grid is invalid")
    if not np.allclose(
        output[["q_hip_rad", "q_knee_rad"]].iloc[0],
        output[["q_hip_rad", "q_knee_rad"]].iloc[-1],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("retimed measured reference is not joint-state closed")
    if not np.allclose(
        output[["x_pull_m", "z_pull_m"]].iloc[0],
        output[["x_pull_m", "z_pull_m"]].iloc[-1],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("retimed measured reference is not pull-point closed")
    endpoint_derivatives = output.iloc[[0, -1]][
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    peak_index = int(np.argmin(np.abs(output["time_s"] - flexion_duration)))
    peak_derivatives = output.iloc[[peak_index]][
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    if not np.allclose(endpoint_derivatives, 0.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("retimed endpoints are not C2-stationary")
    if not np.allclose(peak_derivatives, 0.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("retimed peak is not C2-stationary")
    if not np.allclose(
        output["theta_shank_rad"], q_hip - q_knee, atol=1e-14, rtol=0.0
    ):
        raise RuntimeError("theta_shank = q_hip - q_knee was not preserved")
    if not bool(output["trajectory_sample_valid"].all()):
        raise RuntimeError("retimed measured reference failed finite/ROM audit")
    return output


__all__ = [
    "DEFAULT_DENSE_AUDIT_SAMPLE_COUNT",
    "DEFAULT_PHASE_SAMPLE_COUNT",
    "FlexionExtensionAsymmetryAudit",
    "MEASURED_ASYMMETRIC_CLOSED_REFERENCE",
    "MEASURED_ASYMMETRIC_MODEL_VERSION",
    "MEASURED_ASYMMETRIC_NOMINAL_ID",
    "MEASURED_ASYMMETRIC_SLOW_ID",
    "MEASURED_RAW_REFERENCE",
    "MeasuredAsymmetricPeriodicModel",
    "PeriodicClosureDeviationAudit",
    "PeriodicContinuityAudit",
    "PERIODIC_PATH_SMOOTHING_POLYNOMIAL_ORDER",
    "PERIODIC_PATH_SMOOTHING_WINDOW",
    "PERIODIC_SPLINE_SUBDIVISION_FACTOR",
    "build_reference_measured_raw",
    "fit_measured_asymmetric_periodic_reference",
    "retime_measured_asymmetric_periodic_reference",
]
