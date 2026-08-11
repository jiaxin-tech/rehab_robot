"""Full-joint-state audit for measured flexion/extension cycle closure.

The legacy Stage-5A cycle table brackets cycles with knee extrema alone.  This
module deliberately does not reuse those boundaries for selection.  It forms
a one-dimensional phase signal from a PCA of the approved-ROM-normalized hip
and knee angles, detects all major flexion peaks and adjacent joint-state
valleys, and then jointly chooses a same-phase boundary on the measured
flexion and measured extension branches.

The boundary score retains signed hip, knee, pull-x, and pull-z errors.  A
candidate is eligible only when it is a complete two-joint excursion, every
sample passes the persisted projection audit, and every sample is inside the
explicit run-local 0--120 / 5--145 degree ROM.  This is an offline numerical
audit: it imports no robot SDK, hardware, control, or safety module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from .config import L1, L2
from .kinematics import forward_kinematics


APPROVED_HIP_ROM_DEG = (0.0, 120.0)
APPROVED_KNEE_ROM_DEG = (5.0, 145.0)

CLOSURE_AUDIT_COLUMNS = (
    "cycle_candidate_id",
    "legacy_cycle_index",
    "legacy_peak_frame",
    "legacy_peak_offset_frames",
    "start_frame",
    "peak_frame",
    "end_frame",
    "bracket_start_frame",
    "bracket_end_frame",
    "q_hip_start",
    "q_hip_end",
    "q_knee_start",
    "q_knee_end",
    "q_hip_start_deg",
    "q_hip_peak_deg",
    "q_hip_end_deg",
    "q_knee_start_deg",
    "q_knee_peak_deg",
    "q_knee_end_deg",
    "delta_q_hip_deg",
    "delta_q_knee_deg",
    "pull_start_x",
    "pull_start_z",
    "pull_end_x",
    "pull_end_z",
    "delta_x_pull_m",
    "delta_z_pull_m",
    "pull_closure_error_m",
    "pull_closure_error_mm",
    "closure_q_hip_component",
    "closure_q_knee_component",
    "closure_x_pull_component",
    "closure_z_pull_component",
    "closure_score",
    "hip_flexion_excursion_deg",
    "knee_flexion_excursion_deg",
    "hip_extension_excursion_deg",
    "knee_extension_excursion_deg",
    "major_peak_count",
    "cycle_complete",
    "projection_valid",
    "projection_invalid_frame_count",
    "rom_valid",
    "rom_violation_count",
    "dq_hip_start_rad_s",
    "dq_hip_end_rad_s",
    "dq_knee_start_rad_s",
    "dq_knee_end_rad_s",
    "delta_dq_hip",
    "delta_dq_knee",
    "delta_dq_hip_rad_s",
    "delta_dq_knee_rad_s",
    "derivative_valid",
    "derivative_invalid_reason",
    "direction_slope_hip_start_deg_per_sample",
    "direction_slope_hip_end_deg_per_sample",
    "direction_slope_knee_start_deg_per_sample",
    "direction_slope_knee_end_deg_per_sample",
    "phase_start",
    "phase_peak",
    "phase_end",
    "low_flexion_phase_cutoff",
    "boundary_pair_available",
    "boundary_invalid_reason",
    "eligible",
    "selected",
)


@dataclass(frozen=True)
class CycleClosureConfig:
    """Numerical settings for full-joint phase and closure auditing."""

    smoothing_window_frames: int = 21
    smoothing_polynomial_order: int = 3
    minimum_peak_distance_frames: int = 50
    minimum_peak_prominence: float = 0.35
    low_flexion_fraction: float = 0.20
    minimum_joint_excursion_deg: float = 25.0
    hip_score_scale_deg: float = 1.0
    knee_score_scale_deg: float = 1.0
    pull_x_score_scale_mm: float = 5.0
    pull_z_score_scale_mm: float = 5.0
    approved_hip_rom_deg: tuple[float, float] = APPROVED_HIP_ROM_DEG
    approved_knee_rom_deg: tuple[float, float] = APPROVED_KNEE_ROM_DEG

    def __post_init__(self) -> None:
        integers = {
            "smoothing_window_frames": self.smoothing_window_frames,
            "smoothing_polynomial_order": self.smoothing_polynomial_order,
            "minimum_peak_distance_frames": self.minimum_peak_distance_frames,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
        if self.smoothing_window_frames < 3 or self.smoothing_window_frames % 2 == 0:
            raise ValueError("smoothing_window_frames must be an odd integer >= 3.")
        if not 0 <= self.smoothing_polynomial_order < self.smoothing_window_frames:
            raise ValueError(
                "smoothing_polynomial_order must be non-negative and smaller "
                "than smoothing_window_frames."
            )
        if self.minimum_peak_distance_frames < 1:
            raise ValueError("minimum_peak_distance_frames must be positive.")
        positive = (
            "minimum_peak_prominence",
            "minimum_joint_excursion_deg",
            "hip_score_scale_deg",
            "knee_score_scale_deg",
            "pull_x_score_scale_mm",
            "pull_z_score_scale_mm",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.low_flexion_fraction) or not (
            0.0 < self.low_flexion_fraction < 1.0
        ):
            raise ValueError("low_flexion_fraction must be finite and in (0, 1).")
        approved_hip = _validate_rom(
            self.approved_hip_rom_deg, "approved_hip_rom_deg"
        )
        approved_knee = _validate_rom(
            self.approved_knee_rom_deg, "approved_knee_rom_deg"
        )
        if approved_hip != APPROVED_HIP_ROM_DEG:
            raise ValueError(
                f"approved_hip_rom_deg must remain {APPROVED_HIP_ROM_DEG}."
            )
        if approved_knee != APPROVED_KNEE_ROM_DEG:
            raise ValueError(
                f"approved_knee_rom_deg must remain {APPROVED_KNEE_ROM_DEG}."
            )


@dataclass(frozen=True)
class ReferenceCycleClosureAuditResult:
    """All audit products plus the immutable measured slice that was selected."""

    closure_audit: pd.DataFrame
    phase_audit: pd.DataFrame
    selected_candidate_id: int | None
    selected_candidate: pd.Series | None
    selected_measured_cycle: pd.DataFrame
    metadata: dict[str, object]


def _validate_rom(values: Sequence[float], name: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all() or array[0] >= array[1]:
        raise ValueError(f"{name} must contain two increasing finite values.")
    return float(array[0]), float(array[1])


def _strict_bool(series: pd.Series, name: str) -> np.ndarray:
    """Read persisted booleans without accepting arbitrary truthy strings."""

    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"{name} contains missing values.")
        return series.to_numpy(dtype=bool)
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.isin(("true", "false", "1", "0"))
    if not bool(valid.all()):
        raise ValueError(f"{name} contains invalid boolean encodings.")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def _coerce_full_angles(
    full_angles: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    frame_column: str,
    q_hip_column: str,
    q_knee_column: str,
    projection_valid_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(full_angles, pd.DataFrame):
        dataframe = full_angles.copy(deep=True)
    elif isinstance(full_angles, Mapping):
        dataframe = pd.DataFrame(full_angles)
    else:
        raise TypeError("full_angles must be a pandas DataFrame or a mapping.")
    required = {
        frame_column,
        q_hip_column,
        q_knee_column,
        projection_valid_column,
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"full_angles is missing columns: {sorted(missing)}")
    if len(dataframe) < 7:
        raise ValueError("full_angles must contain at least seven samples.")

    frame_float = pd.to_numeric(dataframe[frame_column], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(frame_float).all() or not np.allclose(
        frame_float, np.rint(frame_float), atol=0.0, rtol=0.0
    ):
        raise ValueError("source frames must all be finite integers.")
    frames = np.rint(frame_float).astype(np.int64)
    if not np.all(np.diff(frames) > 0):
        raise ValueError("source frames must be strictly increasing.")

    q_hip = pd.to_numeric(dataframe[q_hip_column], errors="coerce").to_numpy(float)
    q_knee = pd.to_numeric(dataframe[q_knee_column], errors="coerce").to_numpy(float)
    if not np.isfinite(q_hip).all() or not np.isfinite(q_knee).all():
        raise ValueError("q_hip and q_knee must be finite for the full audit window.")
    projection_valid = _strict_bool(
        dataframe[projection_valid_column], projection_valid_column
    )
    return dataframe, frames, q_hip, q_knee, projection_valid


def _effective_savgol(config: CycleClosureConfig, sample_count: int) -> tuple[int, int]:
    window = min(config.smoothing_window_frames, sample_count)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("not enough samples for joint-state smoothing.")
    polynomial_order = min(config.smoothing_polynomial_order, window - 1)
    return window, polynomial_order


def _full_joint_phase(
    q_hip_deg: np.ndarray,
    q_knee_deg: np.ndarray,
    config: CycleClosureConfig,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, tuple[int, int]],
    np.ndarray,
]:
    window, polynomial_order = _effective_savgol(config, len(q_hip_deg))
    q_hip_smoothed = np.asarray(
        savgol_filter(
            q_hip_deg,
            window_length=window,
            polyorder=polynomial_order,
            mode="interp",
        ),
        dtype=float,
    )
    q_knee_smoothed = np.asarray(
        savgol_filter(
            q_knee_deg,
            window_length=window,
            polyorder=polynomial_order,
            mode="interp",
        ),
        dtype=float,
    )
    hip_rom = _validate_rom(config.approved_hip_rom_deg, "approved_hip_rom_deg")
    knee_rom = _validate_rom(config.approved_knee_rom_deg, "approved_knee_rom_deg")
    # Only the approved ROM spans set scale.  PCA centering removes the choice
    # of angular origin, so no artificial branch-dependent offset is added.
    normalized = np.column_stack(
        (
            q_hip_smoothed / (hip_rom[1] - hip_rom[0]),
            q_knee_smoothed / (knee_rom[1] - knee_rom[0]),
        )
    )
    normalized_mean = normalized.mean(axis=0)
    centered = normalized - normalized_mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    total_variance = float(np.sum(singular_values**2))
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("full joint state must have non-zero finite variance.")
    principal_component = np.asarray(right_vectors[0], dtype=float)
    # Make positive phase mean increasing knee flexion; this fixes the otherwise
    # arbitrary PCA sign and makes branch direction checks deterministic.
    if principal_component[1] < 0.0:
        principal_component *= -1.0
    phase = centered @ principal_component
    explained_variance_ratio = float(singular_values[0] ** 2 / total_variance)
    return (
        q_hip_smoothed,
        q_knee_smoothed,
        normalized_mean,
        principal_component,
        explained_variance_ratio,
        (window, polynomial_order),
    ), phase


def _direction_slopes(
    q_hip_deg: np.ndarray,
    q_knee_deg: np.ndarray,
    window: int,
    polynomial_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    hip = savgol_filter(
        q_hip_deg,
        window_length=window,
        polyorder=polynomial_order,
        deriv=1,
        delta=1.0,
        mode="interp",
    )
    knee = savgol_filter(
        q_knee_deg,
        window_length=window,
        polyorder=polynomial_order,
        deriv=1,
        delta=1.0,
        mode="interp",
    )
    return np.asarray(hip, dtype=float), np.asarray(knee, dtype=float)


def _legacy_cross_reference(
    peak_frame: int,
    detected_cycles: pd.DataFrame | None,
) -> tuple[float, float, float]:
    if detected_cycles is None or detected_cycles.empty:
        return np.nan, np.nan, np.nan
    required = {"cycle_index", "peak_flexion_frame"}
    missing = required.difference(detected_cycles.columns)
    if missing:
        raise ValueError(f"detected_cycles is missing columns: {sorted(missing)}")
    legacy_peaks = pd.to_numeric(
        detected_cycles["peak_flexion_frame"], errors="coerce"
    ).to_numpy(float)
    legacy_ids = pd.to_numeric(
        detected_cycles["cycle_index"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(legacy_peaks) & np.isfinite(legacy_ids)
    if not valid.any():
        return np.nan, np.nan, np.nan
    valid_positions = np.flatnonzero(valid)
    position = int(
        valid_positions[np.argmin(np.abs(legacy_peaks[valid] - float(peak_frame)))]
    )
    legacy_peak = float(legacy_peaks[position])
    return float(legacy_ids[position]), legacy_peak, float(peak_frame - legacy_peak)


def _empty_candidate_row(
    candidate_id: int,
    peak_position: int,
    left_position: int,
    right_position: int,
    frames: np.ndarray,
    phase: np.ndarray,
    cutoff: float,
    detected_cycles: pd.DataFrame | None,
    reason: str,
    derivative_reason: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in CLOSURE_AUDIT_COLUMNS}
    legacy_id, legacy_peak, legacy_offset = _legacy_cross_reference(
        int(frames[peak_position]), detected_cycles
    )
    row.update(
        {
            "cycle_candidate_id": int(candidate_id),
            "legacy_cycle_index": legacy_id,
            "legacy_peak_frame": legacy_peak,
            "legacy_peak_offset_frames": legacy_offset,
            "peak_frame": int(frames[peak_position]),
            "bracket_start_frame": int(frames[left_position]),
            "bracket_end_frame": int(frames[right_position]),
            "phase_peak": float(phase[peak_position]),
            "low_flexion_phase_cutoff": float(cutoff),
            "major_peak_count": 0,
            "cycle_complete": False,
            "projection_valid": False,
            "projection_invalid_frame_count": np.nan,
            "rom_valid": False,
            "rom_violation_count": np.nan,
            "derivative_valid": False,
            "derivative_invalid_reason": derivative_reason,
            "boundary_pair_available": False,
            "boundary_invalid_reason": reason,
            "eligible": False,
            "selected": False,
        }
    )
    return row


def _physical_derivative_status(
    frames: np.ndarray,
    source_fps: float | None,
) -> tuple[bool, str, float | None]:
    if source_fps is None:
        return False, "source_fps_not_provided", None
    fps = float(source_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("source_fps must be finite and positive when provided.")
    if not np.all(np.diff(frames) == 1):
        return False, "non_unit_source_frame_steps", None
    return True, "", fps


def _candidate_from_boundaries(
    *,
    candidate_id: int,
    start_position: int,
    peak_position: int,
    end_position: int,
    left_position: int,
    right_position: int,
    frames: np.ndarray,
    q_hip: np.ndarray,
    q_knee: np.ndarray,
    q_hip_deg: np.ndarray,
    q_knee_deg: np.ndarray,
    x_pull: np.ndarray,
    z_pull: np.ndarray,
    phase: np.ndarray,
    cutoff: float,
    direction_hip: np.ndarray,
    direction_knee: np.ndarray,
    physical_derivative_valid: bool,
    physical_derivative_reason: str,
    source_fps: float | None,
    projection_valid: np.ndarray,
    major_peaks: np.ndarray,
    config: CycleClosureConfig,
    detected_cycles: pd.DataFrame | None,
) -> dict[str, object]:
    start = int(start_position)
    peak = int(peak_position)
    end = int(end_position)
    segment = slice(start, end + 1)

    delta_hip_deg = float(q_hip_deg[end] - q_hip_deg[start])
    delta_knee_deg = float(q_knee_deg[end] - q_knee_deg[start])
    delta_x_m = float(x_pull[end] - x_pull[start])
    delta_z_m = float(z_pull[end] - z_pull[start])
    delta_x_mm = 1000.0 * delta_x_m
    delta_z_mm = 1000.0 * delta_z_m
    hip_component = delta_hip_deg / config.hip_score_scale_deg
    knee_component = delta_knee_deg / config.knee_score_scale_deg
    x_component = delta_x_mm / config.pull_x_score_scale_mm
    z_component = delta_z_mm / config.pull_z_score_scale_mm
    score = float(
        np.sqrt(
            hip_component**2
            + knee_component**2
            + x_component**2
            + z_component**2
        )
    )

    hip_flexion = float(q_hip_deg[peak] - q_hip_deg[start])
    knee_flexion = float(q_knee_deg[peak] - q_knee_deg[start])
    hip_extension = float(q_hip_deg[peak] - q_hip_deg[end])
    knee_extension = float(q_knee_deg[peak] - q_knee_deg[end])
    major_peak_count = int(np.count_nonzero((major_peaks > start) & (major_peaks < end)))
    cycle_complete = bool(
        major_peak_count == 1
        and min(hip_flexion, knee_flexion, hip_extension, knee_extension)
        >= config.minimum_joint_excursion_deg
    )
    projection_invalid_count = int(np.count_nonzero(~projection_valid[segment]))
    projection_ok = projection_invalid_count == 0

    hip_rom = _validate_rom(config.approved_hip_rom_deg, "approved_hip_rom_deg")
    knee_rom = _validate_rom(config.approved_knee_rom_deg, "approved_knee_rom_deg")
    inside_rom = (
        (q_hip_deg[segment] >= hip_rom[0])
        & (q_hip_deg[segment] <= hip_rom[1])
        & (q_knee_deg[segment] >= knee_rom[0])
        & (q_knee_deg[segment] <= knee_rom[1])
    )
    rom_violation_count = int(np.count_nonzero(~inside_rom))
    rom_ok = rom_violation_count == 0

    if physical_derivative_valid:
        assert source_fps is not None
        conversion = float(np.deg2rad(1.0) * source_fps)
        dq_hip_start = float(direction_hip[start] * conversion)
        dq_hip_end = float(direction_hip[end] * conversion)
        dq_knee_start = float(direction_knee[start] * conversion)
        dq_knee_end = float(direction_knee[end] * conversion)
        delta_dq_hip = dq_hip_end - dq_hip_start
        delta_dq_knee = dq_knee_end - dq_knee_start
    else:
        dq_hip_start = dq_hip_end = np.nan
        dq_knee_start = dq_knee_end = np.nan
        delta_dq_hip = delta_dq_knee = np.nan

    legacy_id, legacy_peak, legacy_offset = _legacy_cross_reference(
        int(frames[peak]), detected_cycles
    )
    eligible = bool(cycle_complete and projection_ok and rom_ok)
    return {
        "cycle_candidate_id": int(candidate_id),
        "legacy_cycle_index": legacy_id,
        "legacy_peak_frame": legacy_peak,
        "legacy_peak_offset_frames": legacy_offset,
        "start_frame": int(frames[start]),
        "peak_frame": int(frames[peak]),
        "end_frame": int(frames[end]),
        "bracket_start_frame": int(frames[left_position]),
        "bracket_end_frame": int(frames[right_position]),
        "q_hip_start": float(q_hip[start]),
        "q_hip_end": float(q_hip[end]),
        "q_knee_start": float(q_knee[start]),
        "q_knee_end": float(q_knee[end]),
        "q_hip_start_deg": float(q_hip_deg[start]),
        "q_hip_peak_deg": float(q_hip_deg[peak]),
        "q_hip_end_deg": float(q_hip_deg[end]),
        "q_knee_start_deg": float(q_knee_deg[start]),
        "q_knee_peak_deg": float(q_knee_deg[peak]),
        "q_knee_end_deg": float(q_knee_deg[end]),
        "delta_q_hip_deg": delta_hip_deg,
        "delta_q_knee_deg": delta_knee_deg,
        "pull_start_x": float(x_pull[start]),
        "pull_start_z": float(z_pull[start]),
        "pull_end_x": float(x_pull[end]),
        "pull_end_z": float(z_pull[end]),
        "delta_x_pull_m": delta_x_m,
        "delta_z_pull_m": delta_z_m,
        "pull_closure_error_m": float(np.hypot(delta_x_m, delta_z_m)),
        "pull_closure_error_mm": float(np.hypot(delta_x_mm, delta_z_mm)),
        "closure_q_hip_component": float(hip_component),
        "closure_q_knee_component": float(knee_component),
        "closure_x_pull_component": float(x_component),
        "closure_z_pull_component": float(z_component),
        "closure_score": score,
        "hip_flexion_excursion_deg": hip_flexion,
        "knee_flexion_excursion_deg": knee_flexion,
        "hip_extension_excursion_deg": hip_extension,
        "knee_extension_excursion_deg": knee_extension,
        "major_peak_count": major_peak_count,
        "cycle_complete": cycle_complete,
        "projection_valid": projection_ok,
        "projection_invalid_frame_count": projection_invalid_count,
        "rom_valid": rom_ok,
        "rom_violation_count": rom_violation_count,
        "dq_hip_start_rad_s": dq_hip_start,
        "dq_hip_end_rad_s": dq_hip_end,
        "dq_knee_start_rad_s": dq_knee_start,
        "dq_knee_end_rad_s": dq_knee_end,
        "delta_dq_hip": delta_dq_hip,
        "delta_dq_knee": delta_dq_knee,
        "delta_dq_hip_rad_s": delta_dq_hip,
        "delta_dq_knee_rad_s": delta_dq_knee,
        "derivative_valid": physical_derivative_valid,
        "derivative_invalid_reason": physical_derivative_reason,
        "direction_slope_hip_start_deg_per_sample": float(direction_hip[start]),
        "direction_slope_hip_end_deg_per_sample": float(direction_hip[end]),
        "direction_slope_knee_start_deg_per_sample": float(direction_knee[start]),
        "direction_slope_knee_end_deg_per_sample": float(direction_knee[end]),
        "phase_start": float(phase[start]),
        "phase_peak": float(phase[peak]),
        "phase_end": float(phase[end]),
        "low_flexion_phase_cutoff": float(cutoff),
        "boundary_pair_available": True,
        "boundary_invalid_reason": "",
        "eligible": eligible,
        "selected": False,
    }


def select_best_cycle_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    """Return the eligible minimum-score candidate with deterministic ties."""

    required = {
        "cycle_candidate_id",
        "closure_score",
        "cycle_complete",
        "projection_valid",
        "rom_valid",
        "eligible",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing columns: {sorted(missing)}")
    strict_gate = _strict_bool(candidates["eligible"], "eligible")
    for gate in ("cycle_complete", "projection_valid", "rom_valid"):
        strict_gate &= _strict_bool(candidates[gate], gate)
    eligible = candidates.loc[
        strict_gate
        & np.isfinite(pd.to_numeric(candidates["closure_score"], errors="coerce"))
    ].copy()
    if eligible.empty:
        return None
    eligible["closure_score"] = pd.to_numeric(
        eligible["closure_score"], errors="raise"
    )
    eligible["cycle_candidate_id"] = pd.to_numeric(
        eligible["cycle_candidate_id"], errors="raise"
    )
    ordered = eligible.sort_values(
        ["closure_score", "cycle_candidate_id"], kind="mergesort"
    )
    return ordered.iloc[0].copy()


def audit_reference_cycle_closure(
    full_angles: pd.DataFrame | Mapping[str, Sequence[object]],
    detected_cycles: pd.DataFrame | None = None,
    *,
    source_fps: float | None = None,
    config: CycleClosureConfig | None = None,
    frame_column: str = "Frame",
    q_hip_column: str = "q_hip_rad",
    q_knee_column: str = "q_knee_rad",
    projection_valid_column: str = "angle_valid",
    L1_m: float = L1,
    L2_m: float = L2,
) -> ReferenceCycleClosureAuditResult:
    """Detect, score, strictly filter, and select natural measured cycles.

    ``detected_cycles`` is optional legacy provenance only.  It is cross-linked
    by nearest peak frame and never supplies a phase peak, bracket, boundary,
    score, or eligibility value.
    """

    effective_config = config or CycleClosureConfig()
    dataframe, frames, q_hip, q_knee, angle_valid = _coerce_full_angles(
        full_angles,
        frame_column=frame_column,
        q_hip_column=q_hip_column,
        q_knee_column=q_knee_column,
        projection_valid_column=projection_valid_column,
    )
    if not np.isfinite(L1_m) or not np.isfinite(L2_m) or L1_m <= 0.0 or L2_m <= 0.0:
        raise ValueError("L1_m and L2_m must be finite positive lengths.")

    q_hip_deg = np.rad2deg(q_hip)
    q_knee_deg = np.rad2deg(q_knee)
    phase_products, phase = _full_joint_phase(q_hip_deg, q_knee_deg, effective_config)
    (
        q_hip_smoothed,
        q_knee_smoothed,
        normalized_mean,
        principal_component,
        explained_variance_ratio,
        smoothing_parameters,
    ) = phase_products
    window, polynomial_order = smoothing_parameters
    direction_hip, direction_knee = _direction_slopes(
        q_hip_deg, q_knee_deg, window, polynomial_order
    )
    major_peaks, peak_properties = find_peaks(
        phase,
        distance=effective_config.minimum_peak_distance_frames,
        prominence=effective_config.minimum_peak_prominence,
    )
    joint_state_valleys, _ = find_peaks(
        -phase,
        distance=effective_config.minimum_peak_distance_frames,
        prominence=effective_config.minimum_peak_prominence,
    )
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1_m, L2_m)
    x_pull = np.asarray(x_pull, dtype=float)
    z_pull = np.asarray(z_pull, dtype=float)
    derivative_valid, derivative_reason, effective_fps = _physical_derivative_status(
        frames, source_fps
    )

    positions = np.arange(len(dataframe), dtype=int)
    rows: list[dict[str, object]] = []
    for candidate_id, peak_position_value in enumerate(major_peaks):
        peak_position = int(peak_position_value)
        preceding = joint_state_valleys[joint_state_valleys < peak_position]
        following = joint_state_valleys[joint_state_valleys > peak_position]
        left_position = int(preceding[-1]) if len(preceding) else 0
        right_position = int(following[0]) if len(following) else len(dataframe) - 1
        low_phase = max(float(phase[left_position]), float(phase[right_position]))
        cutoff = low_phase + effective_config.low_flexion_fraction * (
            float(phase[peak_position]) - low_phase
        )

        start_candidates = positions[
            (positions >= left_position)
            & (positions < peak_position)
            & (phase <= cutoff)
            & (direction_hip > 0.0)
            & (direction_knee > 0.0)
        ]
        end_candidates = positions[
            (positions > peak_position)
            & (positions <= right_position)
            & (phase <= cutoff)
            & (direction_hip < 0.0)
            & (direction_knee < 0.0)
        ]
        if len(start_candidates) == 0 or len(end_candidates) == 0:
            missing_branches = []
            if len(start_candidates) == 0:
                missing_branches.append("flexion_boundary_band_empty")
            if len(end_candidates) == 0:
                missing_branches.append("extension_boundary_band_empty")
            rows.append(
                _empty_candidate_row(
                    candidate_id,
                    peak_position,
                    left_position,
                    right_position,
                    frames,
                    phase,
                    cutoff,
                    detected_cycles,
                    ";".join(missing_branches),
                    derivative_reason,
                )
            )
            continue

        start_grid = start_candidates[:, None]
        end_grid = end_candidates[None, :]
        hip_errors = q_hip_deg[end_grid] - q_hip_deg[start_grid]
        knee_errors = q_knee_deg[end_grid] - q_knee_deg[start_grid]
        x_errors_mm = 1000.0 * (x_pull[end_grid] - x_pull[start_grid])
        z_errors_mm = 1000.0 * (z_pull[end_grid] - z_pull[start_grid])
        scores = np.sqrt(
            (hip_errors / effective_config.hip_score_scale_deg) ** 2
            + (knee_errors / effective_config.knee_score_scale_deg) ** 2
            + (x_errors_mm / effective_config.pull_x_score_scale_mm) ** 2
            + (z_errors_mm / effective_config.pull_z_score_scale_mm) ** 2
        )
        # Both candidate arrays are source-frame ordered.  C-order argmin thus
        # gives the requested stable tie-break by start frame, then end frame.
        pair_index = int(np.argmin(scores))
        start_subindex, end_subindex = np.unravel_index(pair_index, scores.shape)
        start_position = int(start_candidates[start_subindex])
        end_position = int(end_candidates[end_subindex])
        rows.append(
            _candidate_from_boundaries(
                candidate_id=candidate_id,
                start_position=start_position,
                peak_position=peak_position,
                end_position=end_position,
                left_position=left_position,
                right_position=right_position,
                frames=frames,
                q_hip=q_hip,
                q_knee=q_knee,
                q_hip_deg=q_hip_deg,
                q_knee_deg=q_knee_deg,
                x_pull=x_pull,
                z_pull=z_pull,
                phase=phase,
                cutoff=cutoff,
                direction_hip=direction_hip,
                direction_knee=direction_knee,
                physical_derivative_valid=derivative_valid,
                physical_derivative_reason=derivative_reason,
                source_fps=effective_fps,
                projection_valid=angle_valid,
                major_peaks=major_peaks,
                config=effective_config,
                detected_cycles=detected_cycles,
            )
        )

    closure_audit = pd.DataFrame(rows, columns=CLOSURE_AUDIT_COLUMNS)
    selected_candidate = select_best_cycle_candidate(closure_audit)
    selected_candidate_id: int | None = None
    selected_measured_cycle = dataframe.iloc[0:0].copy()
    if selected_candidate is not None:
        selected_candidate_id = int(selected_candidate["cycle_candidate_id"])
        selected_mask = closure_audit["cycle_candidate_id"].eq(selected_candidate_id)
        closure_audit.loc[selected_mask, "selected"] = True
        selected_candidate = closure_audit.loc[selected_mask].iloc[0].copy()
        start_frame = int(selected_candidate["start_frame"])
        end_frame = int(selected_candidate["end_frame"])
        selected_measured_cycle = dataframe.loc[
            (frames >= start_frame) & (frames <= end_frame)
        ].copy(deep=True)

    peak_mask = np.zeros(len(dataframe), dtype=bool)
    valley_mask = np.zeros(len(dataframe), dtype=bool)
    peak_mask[major_peaks] = True
    valley_mask[joint_state_valleys] = True
    phase_audit = pd.DataFrame(
        {
            "source_frame": frames,
            "q_hip_smoothed_deg": q_hip_smoothed,
            "q_knee_smoothed_deg": q_knee_smoothed,
            "full_joint_phase_pc1": phase,
            "direction_slope_hip_deg_per_sample": direction_hip,
            "direction_slope_knee_deg_per_sample": direction_knee,
            "major_flexion_peak": peak_mask,
            "joint_state_valley": valley_mask,
        }
    )
    metadata: dict[str, object] = {
        "algorithm": "approved_rom_normalized_full_joint_pca_phase_v1",
        "phase_inputs": [q_hip_column, q_knee_column],
        "legacy_detected_cycles_used_for_selection": False,
        "sample_count": int(len(dataframe)),
        "candidate_count": int(len(closure_audit)),
        "eligible_candidate_count": int(
            closure_audit["eligible"].fillna(False).astype(bool).sum()
        ),
        "selected_candidate_id": selected_candidate_id,
        "selection_reason": (
            "eligible_minimum_full_state_closure_score"
            if selected_candidate_id is not None
            else "no_eligible_cycle_candidate"
        ),
        "pca_normalized_mean": normalized_mean.astype(float).tolist(),
        "pca_pc1": principal_component.astype(float).tolist(),
        "pca_pc1_explained_variance_ratio": explained_variance_ratio,
        "major_peak_frames": frames[major_peaks].astype(int).tolist(),
        "major_peak_prominences": np.asarray(
            peak_properties.get("prominences", []), dtype=float
        ).tolist(),
        "joint_state_valley_frames": frames[joint_state_valleys].astype(int).tolist(),
        "source_fps": None if source_fps is None else float(source_fps),
        "physical_derivatives_available": derivative_valid,
        "physical_derivative_invalid_reason": derivative_reason,
        "physical_derivative_unit": "rad/s",
        "q_hip_start_end_unit": "rad",
        "q_knee_start_end_unit": "rad",
        "pull_start_end_unit": "m",
        "closure_score_definition": (
            "sqrt((delta_q_hip_deg/1deg)^2 + "
            "(delta_q_knee_deg/1deg)^2 + "
            "(delta_x_pull_mm/5mm)^2 + (delta_z_pull_mm/5mm)^2)"
        ),
        "link_lengths_m": {"L1": float(L1_m), "L2": float(L2_m)},
        "config": asdict(effective_config),
        "hardware_used": False,
        "robot_connection_performed": False,
    }
    return ReferenceCycleClosureAuditResult(
        closure_audit=closure_audit,
        phase_audit=phase_audit,
        selected_candidate_id=selected_candidate_id,
        selected_candidate=selected_candidate,
        selected_measured_cycle=selected_measured_cycle,
        metadata=metadata,
    )


__all__ = [
    "APPROVED_HIP_ROM_DEG",
    "APPROVED_KNEE_ROM_DEG",
    "CLOSURE_AUDIT_COLUMNS",
    "CycleClosureConfig",
    "ReferenceCycleClosureAuditResult",
    "audit_reference_cycle_closure",
    "select_best_cycle_candidate",
]
