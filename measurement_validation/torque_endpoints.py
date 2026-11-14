"""Pure normalization shared by real-round analysis and offline history replay."""
from __future__ import annotations

import numpy as np


RAW_ENDPOINT_METHOD = "RAW_SENSOR_TIME_WEIGHTED_WITH_INTERPOLATED_BOUNDARIES_V1"
PHASE_BOUNDARY_CONVENTION = "last_flexion_dispatch_target_shared_by_both_branches"


def raw_torque_endpoint_trace(time_s, tau_hip_nm, tau_knee_nm, *,
                              motion_start_s, phase_boundary_s, motion_end_s):
    """Keep every in-motion sensor sample and interpolate exact interval edges.

    Inputs have already passed the episode's acquisition, calibration and timing
    checks. This operation neither filters nor tares torque. Its piecewise-linear
    boundary values do not recover transients absent from the sensor itself.
    """
    time = np.asarray(time_s, float)
    hip, knee = np.asarray(tau_hip_nm, float), np.asarray(tau_knee_nm, float)
    edges = np.asarray([motion_start_s, phase_boundary_s, motion_end_s], float)
    if (time.ndim != 1 or len(time) < 2 or hip.shape != time.shape or knee.shape != time.shape
            or not np.isfinite(time).all() or not np.isfinite(hip).all() or not np.isfinite(knee).all()
            or not np.all(np.diff(time) > 0) or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0) or time[0] > edges[0] or time[-1] < edges[-1]):
        raise ValueError("finite increasing raw torque samples must bracket all motion/phase edges")
    inside = (time >= edges[0]) & (time <= edges[-1])
    grid = np.unique(np.r_[time[inside], edges])
    relative = grid - edges[0]
    boundary = float(edges[1] - edges[0])
    return dict(schema_version=1, time_s=relative.tolist(),
                phase=np.where(relative <= boundary, "flexion", "extension").tolist(),
                tau_hip_nm=np.interp(grid, time, hip).tolist(),
                tau_knee_nm=np.interp(grid, time, knee).tolist(),
                phase_boundary_time_s=boundary,
                phase_boundary_convention=PHASE_BOUNDARY_CONVENTION,
                raw_samples_in_motion_interval=int(np.sum(inside)))


def torque_metrics_from_trace(trace):
    """Recompute raw-grid RMS and peaks, sharing the declared turning point.

    Both branches contain the exact turning boundary; no time interval is
    silently dropped between the last flexion and first extension command.
    This is a numerical endpoint check, not proof of physical signal validity.
    """
    if not isinstance(trace, dict) or trace.get("schema_version") != 1:
        raise ValueError("raw torque endpoint trace schema_version must be 1")
    if trace.get("phase_boundary_convention") != PHASE_BOUNDARY_CONVENTION:
        raise ValueError("explicit shared dispatch phase-boundary convention required")
    try:
        time = np.asarray(trace["time_s"], float)
        hip = np.asarray(trace["tau_hip_nm"], float)
        knee = np.asarray(trace["tau_knee_nm"], float)
        labels = np.asarray(trace["phase"])
        boundary = float(trace["phase_boundary_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("complete numeric raw torque endpoint trace required") from exc
    if (time.ndim != 1 or len(time) < 3 or hip.shape != time.shape or knee.shape != time.shape
            or labels.shape != time.shape or not np.isfinite(time).all()
            or not np.isfinite(hip).all() or not np.isfinite(knee).all()
            or not np.all(np.diff(time) > 0) or time[0] != 0.
            or not np.isfinite(boundary) or not time[0] < boundary < time[-1]
            or not np.any(time == boundary)
            or not np.array_equal(labels, np.where(time <= boundary, "flexion", "extension"))):
        raise ValueError("invalid raw torque time, phase boundary, labels or samples")
    components = []
    for torque in (hip, knee):
        for mask in (time <= boundary, time >= boundary):
            branch_time = time[mask]
            components.append(float(np.sqrt(np.trapezoid(torque[mask] ** 2, branch_time)
                                            / (branch_time[-1] - branch_time[0]))))
    return dict(components_nm=components,
                peaks_nm=[float(np.max(np.abs(hip))), float(np.max(np.abs(knee)))])


def endpoints(metrics, baseline):
    reference = np.asarray(baseline["components_nm"], float)
    peaks = np.asarray(baseline["peaks_nm"], float)
    if reference.shape != (4,) or peaks.shape != (2,) or not np.isfinite([*reference, *peaks]).all() or np.any(reference <= 1e-12) or np.any(peaks <= 1e-12):
        raise ValueError("nonzero finite measured baseline required")
    actual = np.asarray(metrics["components_nm"], float)
    actual_peaks = np.asarray(metrics["peaks_nm"], float)
    if actual.shape != (4,) or actual_peaks.shape != (2,) or not np.isfinite([*actual, *actual_peaks]).all() or np.any(actual < 0) or np.any(actual_peaks < 0):
        raise ValueError("finite nonnegative measured endpoints required")
    ratios = actual / reference
    return dict(E3=float(ratios.mean()), E2=float(ratios.max()),
                peak_ratio=float(np.max(actual_peaks/peaks)))
