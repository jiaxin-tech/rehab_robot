"""Offline geometry and timing for a hash-bound drag-taught Cartesian loop."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from control.independent_tcp_trajectory import TcpAudit

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "taught_tcp"
TRAJECTORY_ID = "drag_taught_loop_v1"
MAX_SAMPLES = 200000
PHASES = ("lift", "high_reach", "return")
DURATIONS = ("lift_duration_s", "reach_duration_s", "return_duration_s")
SCALES = ("scale_x", "scale_y", "scale_z")
HOLDS = ("start_hold_s", "turn_hold_s", "end_hold_s")
SETTING_KEYS = {"geometry_path", "geometry_sha256", *SCALES, *DURATIONS,
                *HOLDS, "sample_period_s"}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    value = float(value)
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(name + " is outside its supported range")
    return value


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=object)
    if array.shape != shape or any(isinstance(v, bool) or not isinstance(v, Real)
                                   or not math.isfinite(v) for v in array.flat):
        raise ValueError(name + " has invalid dimensions or nonfinite/non-numeric values")
    return array.astype(float)


def _settings(settings):
    if not isinstance(settings, dict) or set(settings) != SETTING_KEYS:
        raise ValueError("taught settings require exactly: " + ", ".join(sorted(SETTING_KEYS)))
    spec = dict(settings)
    path = spec["geometry_path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("geometry_path must be a nonempty string")
    digest = spec["geometry_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest):
        raise ValueError("geometry_sha256 must be a lowercase SHA-256 digest")
    for name in (*SCALES, *DURATIONS, "sample_period_s"):
        spec[name] = _number(spec[name], name, positive=True)
    for name in HOLDS:
        spec[name] = _number(spec[name], name)
    if not .001 <= spec["sample_period_s"] <= 1:
        raise ValueError("sample_period_s must be in [0.001, 1]")
    if any(spec[name] < .001 for name in DURATIONS) or any(
            0 < spec[name] < .001 for name in HOLDS):
        raise ValueError("nonzero taught stage durations must be at least 0.001 seconds")
    if sum(spec[name] for name in (*DURATIONS, *HOLDS)) > 3600:
        raise ValueError("total taught cycle duration must not exceed 3600 seconds")
    return spec


def _load_geometry(spec):
    path = Path(spec["geometry_path"]).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    if len(raw) > 8_000_000:
        raise ValueError("taught geometry exceeds supported file size")
    if hashlib.sha256(raw).hexdigest() != spec["geometry_sha256"]:
        raise ValueError("taught geometry SHA-256 mismatch")
    geometry = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_strict_object)
    if not isinstance(geometry, dict) or set(geometry) != {
            "schema_version", "recorded_anchor_pose", "segments", "source"}:
        raise ValueError("invalid taught geometry schema fields")
    if type(geometry["schema_version"]) is not int or geometry["schema_version"] != 1:
        raise ValueError("unsupported taught geometry schema version")
    _finite_array(geometry["recorded_anchor_pose"], (6,), "recorded_anchor_pose")
    if not isinstance(geometry["source"], dict):
        raise ValueError("geometry source must be an object")
    json.dumps(geometry["source"], allow_nan=False)
    segments = geometry["segments"]
    if not isinstance(segments, list) or len(segments) != 3:
        raise ValueError("taught geometry requires exactly three segments")
    controls = []
    for phase, segment in zip(PHASES, segments):
        if not isinstance(segment, dict) or set(segment) != {
                "phase", "control_points_delta_base_m"} or segment["phase"] != phase:
            raise ValueError("taught segment order must be lift, high_reach, return")
        points = segment["control_points_delta_base_m"]
        if not isinstance(points, list) or not 4 <= len(points) <= 10000:
            raise ValueError("each cubic segment needs 4 to 10000 control points")
        control = _finite_array(points, (len(points), 3), phase + " controls")
        if np.all(control == control[0]):
            raise ValueError("taught segment must contain movement")
        controls.append(control)
    if not np.array_equal(controls[0][0], np.zeros(3)) or not np.array_equal(
            controls[-1][-1], np.zeros(3)):
        raise ValueError("taught geometry must start and close exactly at zero base delta")
    if any(not np.array_equal(a[-1], b[0]) for a, b in zip(controls, controls[1:])):
        raise ValueError("taught segment endpoints must join exactly")
    peak = controls[0][-1, 2]
    if peak <= 0 or any(np.max(control[:, 2]) > peak + 1e-12 for control in controls):
        raise ValueError("lift endpoint must be the highest taught base-Z control point")
    return geometry, controls


def _spline(control):
    count = len(control)
    interior = np.arange(1, count - 3, dtype=float) / (count - 3)
    knots = np.r_[np.zeros(4), interior, np.ones(4)]
    return BSpline(knots, control, 3, extrapolate=False)


def _segment_values(control, duration, fraction):
    u = np.asarray(fraction, dtype=float)
    progress = u**4 * (35 - 84*u + 70*u*u - 20*u**3)
    progress = np.where(u == 0, 0., np.where(u == 1, 1., progress))
    progress = np.clip(progress, 0., 1.)
    rate = 140*u**3*(1-u)**3 / duration
    acceleration = 420*u*u*(1-u)**2*(1-2*u) / duration**2
    spline = _spline(control)
    first = spline.derivative(1)(progress)
    second = spline.derivative(2)(progress)
    return (spline(progress), first*rate[:, None],
            second*rate[:, None]**2 + first*acceleration[:, None])


def _rotation(frame):
    rotation = np.asarray(frame.rotation_base_from_rehab, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(
                np.linalg.det(rotation), 1., atol=1e-12, rtol=0):
        raise ValueError("rehabilitation frame must define a right-handed orthonormal rotation")
    return rotation


def stage_kinematic_extrema(settings):
    """Dense analytic velocity/acceleration maxima for each declared stage."""
    spec = _settings(settings)
    _, controls = _load_geometry(spec)
    scales = np.array([spec[name] for name in SCALES])
    result = []
    for control, duration_name in zip(controls, DURATIONS):
        duration = spec[duration_name]
        count = min(200001, max(1001, math.ceil(duration/.01)+1, 20*len(control)))
        # Cubic spline acceleration can peak sharply at a knot. Include all
        # knots explicitly instead of relying only on uniform clock samples.
        knots = np.unique(_spline(control).t)
        lo, hi = np.zeros_like(knots), np.ones_like(knots)
        for _ in range(45):
            mid = (lo+hi)/2
            progress = mid**4*(35-84*mid+70*mid**2-20*mid**3)
            lo, hi = np.where(progress < knots, mid, lo), np.where(progress < knots, hi, mid)
        u = np.unique(np.r_[np.linspace(0., 1., count), (lo+hi)/2])
        _, velocity, acceleration = _segment_values(control*scales, duration, u)
        result.append((float(np.linalg.norm(velocity, axis=1).max()),
                       float(np.linalg.norm(acceleration, axis=1).max())))
    return tuple(result)


def build_taught_tcp_trajectory(settings, *, current_tcp_start_pose, rehab_frame):
    """Scale Base deltas; Rehab rotation only expresses diagnostic delta_R.

    Orientation stays fixed at the newly captured current anchor.
    """
    spec = _settings(settings)
    geometry, raw_controls = _load_geometry(spec)
    pose = _finite_array(current_tcp_start_pose, (6,), "current TCP anchor")
    rotation = _rotation(rehab_frame)
    scales = np.array([spec[name] for name in SCALES])
    controls = [control * scales for control in raw_controls]
    if not all(np.isfinite(control).all() for control in controls):
        raise ValueError("scaled taught control points are nonfinite")
    stages = [
        ("start_hold", spec["start_hold_s"], None, controls[0][0]),
        ("lift", spec["lift_duration_s"], controls[0], None),
        ("high_reach", spec["reach_duration_s"], controls[1], None),
        ("turn_hold", spec["turn_hold_s"], None, controls[1][-1]),
        ("return", spec["return_duration_s"], controls[2], None),
        ("end_hold", spec["end_hold_s"], None, controls[2][-1]),
    ]
    counts = [max(8, math.ceil(duration/spec["sample_period_s"])) if duration else 0
              for _, duration, _, _ in stages]
    if 1 + sum(counts) > MAX_SAMPLES:
        raise ValueError("taught trajectory has too many samples")
    times, phases = [0.], ["start_hold"]
    deltas, velocities, accelerations = [np.zeros(3)], [np.zeros(3)], [np.zeros(3)]
    boundaries = {}
    elapsed = 0.
    for (phase, duration, control, hold), count in zip(stages, counts):
        if not count:
            continue
        u = np.linspace(0., 1., count+1)[1:]
        if control is None:
            delta = np.repeat(hold[None, :], count, axis=0)
            velocity = acceleration = np.zeros_like(delta)
        else:
            delta, velocity, acceleration = _segment_values(control, duration, u)
            delta[-1] = control[-1]
        boundaries[phase] = {"start_s": elapsed, "end_s": elapsed+duration}
        times.extend((elapsed + duration*u).tolist())
        phases.extend([phase]*count)
        deltas.extend(delta)
        velocities.extend(velocity)
        accelerations.extend(acceleration)
        elapsed += duration
    delta = np.asarray(deltas)
    delta[0] = delta[-1] = 0.
    xyz = pose[:3] + delta
    if not np.isfinite(xyz).all():
        raise ValueError("taught absolute positions are nonfinite")
    relative_rehab = delta @ rotation
    velocity, acceleration = np.asarray(velocities), np.asarray(accelerations)
    table = pd.DataFrame({
        "time_s": times, "trajectory_id": TRAJECTORY_ID,
        "trajectory_profile": PROFILE, "trajectory_phase": phases,
        "trajectory_valid": True, "tcp_plan_json": json.dumps(spec, sort_keys=True),
    })
    for i, axis in enumerate("xyz"):
        table[f"delta_{axis}_R"] = relative_rehab[:, i]
        table[f"tcp_{axis}_base"] = xyz[:, i]
        table[f"tcp_v{axis}_base"] = velocity[:, i]
        table[f"tcp_a{axis}_base"] = acceleration[:, i]
        table[f"tcp_r{axis}"] = pose[3+i]
    return table, TcpAudit(TRAJECTORY_ID, len(table), True), {
        "mode": PROFILE, "tcp_settings": spec, "joint_reference_used": False,
        "joint_tracking_required": False, "duration_s": elapsed,
        "actual_peak_tcp_z_base_m": float(xyz[:, 2].max()),
        "geometry_sha256": spec["geometry_sha256"], "geometry_source": geometry["source"],
        "recorded_anchor_pose": geometry["recorded_anchor_pose"],
        "base_delta_not_rotated": True, "phase_boundaries": boundaries,
        "geometry_control_bounds_delta_base_m": {
            "min": np.vstack(controls).min(axis=0).tolist(),
            "max": np.vstack(controls).max(axis=0).tolist()},
    }


def audit_taught_tcp_trajectory(table, safety, *, expected_trajectory_id=None,
                               frame=None, anchor=None):
    """Reconstruct hash-bound content, then independently check path and timing."""
    reasons = []
    try:
        if expected_trajectory_id not in (None, TRAJECTORY_ID):
            reasons.append("tcp_trajectory_id_binding_mismatch")
        if not 3 <= len(table) <= MAX_SAMPLES:
            raise ValueError("invalid taught sample count")
        if table["trajectory_id"].tolist() != [TRAJECTORY_ID]*len(table):
            reasons.append("tcp_trajectory_id_invalid")
        if table["trajectory_profile"].tolist() != [PROFILE]*len(table):
            reasons.append("tcp_profile_invalid")
        if not all(type(v) in (bool, np.bool_) and v for v in table["trajectory_valid"]):
            reasons.append("tcp_trajectory_invalid_rows")
        plans = table["tcp_plan_json"].unique()
        if len(plans) != 1:
            raise ValueError("inconsistent taught plans")
        spec = _settings(json.loads(plans[0], object_pairs_hook=_strict_object))
        _, controls = _load_geometry(spec)
        pose_columns = [f"tcp_{a}_base" for a in "xyz"] + [f"tcp_r{a}" for a in "xyz"]
        pose = table.iloc[0][pose_columns].to_numpy(float)
        if anchor is not None:
            pose = np.asarray(anchor.tcp_pose_base, float)
        if frame is None:
            raise ValueError("taught trajectory audit requires the rehabilitation frame")
        expected, _, _ = build_taught_tcp_trajectory(spec, current_tcp_start_pose=pose, rehab_frame=frame)
        if table["trajectory_phase"].tolist() != expected["trajectory_phase"].tolist():
            reasons.append("tcp_phase_mismatch")
        numeric = ["time_s"] + pose_columns + [f"delta_{a}_R" for a in "xyz"] + [
            f"tcp_{d}{a}_base" for d in ("v", "a") for a in "xyz"]
        if len(table) != len(expected) or not np.allclose(
                table[numeric].to_numpy(float), expected[numeric].to_numpy(float), atol=1e-10, rtol=0):
            reasons.append("tcp_plan_content_mismatch")
        times = table.time_s.to_numpy(float)
        xyz = table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
        if not np.isfinite(times).all() or not np.isfinite(xyz).all() or np.any(np.diff(times) <= 0):
            raise ValueError("invalid taught positions or time order")
        velocity = np.gradient(xyz, times, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
        speed = float(np.linalg.norm(velocity, axis=1).max())
        accel = float(np.linalg.norm(acceleration, axis=1).max())
        scales = np.array([spec[name] for name in SCALES])
        scaled = [control * scales for control in controls]
        # Numerical dense analytic check, not a certified continuous extrema solver.
        for stage_speed, stage_accel in stage_kinematic_extrema(spec):
            speed = max(speed, stage_speed)
            accel = max(accel, stage_accel)
        if not np.isfinite([speed, accel]).all():
            raise ValueError("nonfinite taught derivatives")
        if speed > safety.max_tcp_speed_m_s + 1e-12:
            reasons.append("trajectory_speed_limit_exceeded")
        if accel > safety.max_tcp_acceleration_m_s2 + 1e-12:
            reasons.append("trajectory_acceleration_limit_exceeded")
        lower, upper = np.asarray(safety.workspace_min_base_m), np.asarray(safety.workspace_max_base_m)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(np.r_[lower, upper]).all() or np.any(lower >= upper):
            raise ValueError("invalid reviewed workspace")
        hull = pose[:3] + np.vstack(scaled)
        if ((hull < lower) | (hull > upper)).any() or ((xyz < lower) | (xyz > upper)).any():
            reasons.append("trajectory_outside_reviewed_workspace")
        return tuple(dict.fromkeys(reasons)), speed, accel
    except (OSError, KeyError, ValueError, TypeError, IndexError, OverflowError, AttributeError):
        return ("tcp_plan_invalid",), None, None
