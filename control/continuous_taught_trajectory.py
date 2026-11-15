"""Hash-bound continuous taught loop with one Cartesian arc-length clock.

The three rehabilitation phases are labels on a single smooth spatial curve.
Only the beginning and end have speed ramps; no phase boundary stops motion.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator, PPoly
from scipy.spatial import cKDTree

from control.independent_tcp_trajectory import TcpAudit
from control.taught_tcp_trajectory import (
    _finite_array, _strict_object, _rotation, _spline,
    _load_geometry as _load_segmented_geometry, _settings as _segmented_settings,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "continuous_taught_tcp"
TRAJECTORY_ID = "drag_taught_continuous_v1"
MAX_SAMPLES = 200000
PHASES = ("lift", "high_reach", "return")
SCALES = ("scale_x", "scale_y", "scale_z")
SETTING_KEYS = {"geometry_path", "geometry_sha256", *SCALES,
                "cycle_duration_s", "ramp_duration_s", "sample_period_s"}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def _settings(settings):
    if not isinstance(settings, dict) or set(settings) != SETTING_KEYS:
        raise ValueError("continuous taught settings require exactly: " + ", ".join(sorted(SETTING_KEYS)))
    spec = dict(settings)
    if not isinstance(spec["geometry_path"], str) or not spec["geometry_path"].strip():
        raise ValueError("geometry_path must be a nonempty string")
    digest = spec["geometry_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("geometry_sha256 must be a lowercase SHA-256 digest")
    for name in (*SCALES, "cycle_duration_s", "ramp_duration_s", "sample_period_s"):
        spec[name] = _positive(spec[name], name)
    if spec["cycle_duration_s"] > 3600 or 2*spec["ramp_duration_s"] >= spec["cycle_duration_s"]:
        raise ValueError("continuous cycle must be at most 3600 seconds with two ramps shorter than the cycle")
    if not .001 <= spec["sample_period_s"] <= .05:
        raise ValueError("continuous sample_period_s must be in [0.001, 0.05]")
    if spec["cycle_duration_s"]-2*spec["ramp_duration_s"] < 2*spec["sample_period_s"]:
        raise ValueError("continuous constant-speed portion needs at least two command intervals")
    if math.ceil(spec["cycle_duration_s"]/spec["sample_period_s"])+1 > MAX_SAMPLES:
        raise ValueError("continuous taught trajectory has too many samples")
    return spec


def _axis_extrema(spline):
    result = []
    for axis in range(3):
        scalar = BSpline(spline.t, spline.c[:, axis], spline.k, extrapolate=False)
        roots = PPoly.from_spline(scalar).derivative().roots(extrapolate=False)
        u = np.unique(np.r_[0., 1., roots[np.isfinite(roots) & (roots >= 0) & (roots <= 1)]])
        values = scalar(u)
        lo, hi = int(np.argmin(values)), int(np.argmax(values))
        result.append(dict(minimum=float(values[lo]), maximum=float(values[hi]),
                           minimum_parameter=float(u[lo]), maximum_parameter=float(u[hi])))
    return result


@lru_cache(maxsize=16)
def _cached_curve(control_tuple):
    control = np.asarray(control_tuple, float)
    spline = _spline(control)
    knots = np.unique(spline.t)
    count = max(20001, min(500001, 1000*len(control)+1))
    u = np.unique(np.r_[np.linspace(0., 1., count), knots])
    # A knot and a uniform-grid point can differ only by floating roundoff;
    # retaining both would produce a zero accumulated arc increment.
    u = u[np.r_[True, np.diff(u) > 1e-12]]
    u[0], u[-1] = 0., 1.
    first = spline.derivative(1)(u)
    parameter_speed = np.linalg.norm(first, axis=1)
    if not np.isfinite(parameter_speed).all() or parameter_speed.min() < 1e-8:
        raise ValueError("continuous taught curve must be regular; a cusp cannot be crossed at constant speed")
    arc = cumulative_trapezoid(parameter_speed, u, initial=0.)
    if not np.isfinite(arc).all() or arc[-1] <= 1e-8 or np.any(np.diff(arc) <= 0):
        raise ValueError("continuous taught curve has invalid arc length")
    extrema = _axis_extrema(spline)
    top_u, far_u = extrema[2]["maximum_parameter"], extrema[1]["maximum_parameter"]
    if extrema[2]["maximum"] <= 0 or not 0 < top_u < far_u < 1:
        raise ValueError("continuous curve must reach maximum Base Z before maximum Base Y")
    forward = PchipInterpolator(u, arc, extrapolate=False)
    inverse = CubicSpline(arc, u, bc_type=((1,1/parameter_speed[0]),
        (1,1/parameter_speed[-1])), extrapolate=False)
    # Check the exact minima of every quadratic inverse derivative, in addition
    # to endpoint/midpoint samples. A numerical inverse must remain monotone.
    inverse_vertex = np.divide(-inverse.c[1], 3*inverse.c[0],
        out=np.full_like(inverse.c[0],np.nan), where=inverse.c[0]!=0)
    inside = (inverse_vertex>0)&(inverse_vertex<np.diff(arc))
    check_arc = np.unique(np.r_[arc, (arc[:-1]+arc[1:])/2,
        arc[:-1][inside]+inverse_vertex[inside]])
    inverse_rate = inverse.derivative(1)(check_arc)
    if not np.isfinite(inverse_rate).all() or inverse_rate.min() <= 0:
        raise ValueError("continuous inverse arc map is not monotone")
    actual_parameter = np.clip(inverse(check_arc),0.,1.)
    actual_parameter_speed = np.linalg.norm(spline.derivative(1)(actual_parameter),axis=1)
    inverse_speed_error = float(np.max(np.abs(actual_parameter_speed*inverse_rate-1.)))
    if inverse_speed_error > 1e-4:
        raise ValueError("continuous inverse arc map exceeds the speed accuracy budget")
    fractions = np.asarray(forward([top_u, far_u]), float)/arc[-1]
    # The final grid point is always retained by the odd dense-grid count; knot
    # insertion can change parity, so include it explicitly in the coarse check.
    coarse_indices = np.unique(np.r_[np.arange(0, len(u), 2), len(u)-1])
    coarse_length = np.trapezoid(parameter_speed[coarse_indices], u[coarse_indices])
    return dict(spline=spline, parameter_grid=u, arc_grid=arc,
        forward=forward, inverse=inverse, length=float(arc[-1]), extrema=extrema,
        phase_fractions=fractions, phase_parameters=np.array([top_u, far_u]),
        integration_resolution=len(u), inverse_arc_speed_relative_error=inverse_speed_error,
        dense_coarse_length_difference_m=float(abs(arc[-1]-coarse_length)))


def _curve(control):
    return _cached_curve(tuple(tuple(float(v) for v in row) for row in control))


def _load_geometry(spec):
    path = Path(spec["geometry_path"]).expanduser()
    if not path.is_absolute():
        path = ROOT/path
    raw = path.read_bytes()
    if len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != spec["geometry_sha256"]:
        raise ValueError("continuous taught geometry SHA-256 mismatch or unsupported size")
    geometry = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_strict_object)
    if not isinstance(geometry, dict) or set(geometry) != {
            "schema_version", "recorded_anchor_pose", "control_points_delta_base_m",
            "phase_boundaries_arc_fraction", "source"}:
        raise ValueError("invalid continuous taught geometry fields")
    if type(geometry["schema_version"]) is not int or geometry["schema_version"] != 1:
        raise ValueError("unsupported continuous taught geometry version")
    _finite_array(geometry["recorded_anchor_pose"], (6,), "recorded_anchor_pose")
    points = geometry["control_points_delta_base_m"]
    if not isinstance(points, list) or not 4 <= len(points) <= 10000:
        raise ValueError("continuous cubic geometry requires 4 to 10000 control points")
    control = _finite_array(points, (len(points), 3), "continuous control points")
    if not np.array_equal(control[0], np.zeros(3)) or not np.array_equal(control[-1], np.zeros(3)):
        raise ValueError("continuous geometry must start and close exactly at zero Base delta")
    fractions = _finite_array(geometry["phase_boundaries_arc_fraction"], (2,), "phase boundaries")
    if not 0 < fractions[0] < fractions[1] < 1:
        raise ValueError("continuous phase boundaries must be ordered within the loop")
    if not isinstance(geometry["source"], dict):
        raise ValueError("continuous geometry source must be an object")
    json.dumps(geometry["source"], allow_nan=False)
    model = _curve(control)
    if not np.allclose(fractions, model["phase_fractions"], atol=1e-7, rtol=0):
        raise ValueError("continuous phase boundaries do not match the actual height and reach extrema")
    return geometry, control


def _clock(spec, length, times):
    t = np.asarray(times, float)
    total, ramp = spec["cycle_duration_s"], spec["ramp_duration_s"]
    plateau = length/(total-ramp)
    distance = plateau*(t-ramp/2)
    speed = np.full_like(t, plateau)
    acceleration = np.zeros_like(t)
    for mask, u, sign, is_end in (
            (t < ramp, t/ramp, 1., False),
            (t > total-ramp, (total-t)/ramp, -1., True)):
        f = u[mask]
        blend = f**3*(10-15*f+6*f*f)
        integral = f**4*(2.5-3*f+f*f)
        distance[mask] = (length-plateau*ramp*integral if is_end else plateau*ramp*integral)
        speed[mask] = plateau*blend
        acceleration[mask] = sign*plateau/ramp*(30*f*f*(1-f)**2)
    distance = np.clip(distance, 0., length)
    speed[(t == 0) | (t == total)] = 0.
    acceleration[(t == 0) | (t == total)] = 0.
    return distance, speed, acceleration


def _time_at_distance(spec, length, distances):
    distances = np.asarray(distances, float)
    lo, hi = np.zeros_like(distances), np.full_like(distances, spec["cycle_duration_s"])
    for _ in range(55):
        middle = (lo+hi)/2
        current = _clock(spec, length, middle)[0]
        lo, hi = np.where(current < distances, middle, lo), np.where(current < distances, hi, middle)
    return (lo+hi)/2


def _evaluate(model, spec, times):
    distance, speed, tangential_acceleration = _clock(spec, model["length"], times)
    parameter = np.clip(model["inverse"](distance), 0., 1.)
    spline = model["spline"]
    delta = spline(parameter)
    first, second = spline.derivative(1)(parameter), spline.derivative(2)(parameter)
    # Differentiate the same numerical inverse used for positions. This avoids
    # declaring an ideal constant velocity that disagrees with sampled poses.
    inverse_first = model["inverse"].derivative(1)(distance)
    inverse_second = model["inverse"].derivative(2)(distance)
    parameter_rate = inverse_first*speed
    parameter_acceleration = inverse_second*speed**2+inverse_first*tangential_acceleration
    velocity = first*parameter_rate[:,None]
    acceleration = second*parameter_rate[:,None]**2+first*parameter_acceleration[:,None]
    return delta, velocity, acceleration, distance, parameter


def _spec_model(settings):
    spec = _settings(settings)
    geometry, control = _load_geometry(spec)
    scaled = control*np.array([spec[name] for name in SCALES])
    if not np.isfinite(scaled).all():
        raise ValueError("scaled continuous geometry is nonfinite")
    return spec, geometry, scaled, _curve(scaled)


def kinematic_extrema(settings):
    """Dense curve and clock samples, including knots and both ramp endpoints."""
    spec, _, _, model = _spec_model(settings)
    # Uniform parameter samples retain spatial detail even where the curve turns
    # quickly. Uniform time samples cover the independent acceleration ramps.
    u = model["parameter_grid"]
    times = _time_at_distance(spec, model["length"], model["arc_grid"])
    times = np.unique(np.r_[times, np.linspace(0., spec["cycle_duration_s"],
        min(200001, max(2001, math.ceil(spec["cycle_duration_s"]/.005)+1))),
        spec["ramp_duration_s"], spec["cycle_duration_s"]-spec["ramp_duration_s"]])
    _, velocity, acceleration, _, _ = _evaluate(model, spec, times)
    return float(np.linalg.norm(velocity, axis=1).max()), float(np.linalg.norm(acceleration, axis=1).max())


def build_continuous_taught_trajectory(settings, *, current_tcp_start_pose, rehab_frame):
    spec, geometry, control, model = _spec_model(settings)
    pose = _finite_array(current_tcp_start_pose, (6,), "current TCP anchor")
    rotation = _rotation(rehab_frame)
    count = math.ceil(spec["cycle_duration_s"]/spec["sample_period_s"])
    t = np.linspace(0., spec["cycle_duration_s"], count+1)
    delta, velocity, acceleration, distance, _ = _evaluate(model, spec, t)
    delta[0] = delta[-1] = 0.
    xyz = pose[:3]+delta
    chord_speeds=np.linalg.norm(np.diff(xyz,axis=0),axis=1)/np.diff(t)
    plateau_mask=(t[:-1]>=spec["ramp_duration_s"])&(t[1:]<=spec["cycle_duration_s"]-spec["ramp_duration_s"])
    plateau_speed=model["length"]/(spec["cycle_duration_s"]-spec["ramp_duration_s"])
    chord_relative_error=float(np.max(np.abs(chord_speeds[plateau_mask]/plateau_speed-1)))
    fractions = model["phase_fractions"]
    phase_indices = np.searchsorted(fractions*model["length"], distance, side="left")
    phases = np.asarray(PHASES, object)[phase_indices]
    boundary_times = _time_at_distance(spec, model["length"], fractions*model["length"])
    table = pd.DataFrame(dict(time_s=t, trajectory_id=TRAJECTORY_ID,
        trajectory_profile=PROFILE, trajectory_phase=phases,
        trajectory_valid=True, tcp_plan_json=json.dumps(spec, sort_keys=True)))
    relative = delta@rotation
    for i, axis in enumerate("xyz"):
        table[f"delta_{axis}_R"] = relative[:, i]
        table[f"tcp_{axis}_base"] = xyz[:, i]
        table[f"tcp_v{axis}_base"] = velocity[:, i]
        table[f"tcp_a{axis}_base"] = acceleration[:, i]
        table[f"tcp_r{axis}"] = pose[3+i]
    metadata = dict(mode=PROFILE, tcp_settings=spec, duration_s=spec["cycle_duration_s"],
        joint_reference_used=False, joint_tracking_required=False, base_delta_not_rotated=True,
        recorded_anchor_pose=geometry["recorded_anchor_pose"], geometry_sha256=spec["geometry_sha256"],
        geometry_source=geometry["source"], constant_cartesian_speed_m_s=model["length"]/(spec["cycle_duration_s"]-spec["ramp_duration_s"]),
        path_length_m=model["length"], ramp_duration_s=spec["ramp_duration_s"], holds_s=0.,
        phase_boundaries={name:dict(start_s=float(begin), end_s=float(end)) for name,begin,end in
            zip(PHASES, np.r_[0., boundary_times], np.r_[boundary_times, spec["cycle_duration_s"]])},
        actual_phase_boundaries_arc_fraction=fractions.tolist(),
        arc_lookup_sample_count=model["integration_resolution"],
        arc_integration_dense_coarse_difference_m=model["dense_coarse_length_difference_m"],
        inverse_arc_speed_relative_error=model["inverse_arc_speed_relative_error"],
        sampled_midcourse_chord_speed_relative_error=chord_relative_error,
        actual_peak_tcp_z_base_m=pose[2]+model["extrema"][2]["maximum"],
        geometric_extrema_delta_base_m=model["extrema"],
        geometry_control_bounds_delta_base_m=dict(min=control.min(axis=0).tolist(),max=control.max(axis=0).tolist()))
    return table, TcpAudit(TRAJECTORY_ID, len(table), True), metadata


def audit_continuous_taught_trajectory(table, safety, *, expected_trajectory_id=None, frame=None, anchor=None):
    reasons = []
    try:
        if expected_trajectory_id not in (None, TRAJECTORY_ID):
            reasons.append("tcp_trajectory_id_binding_mismatch")
        if not 3 <= len(table) <= MAX_SAMPLES or frame is None:
            raise ValueError("continuous audit requires valid samples and actual rehabilitation frame")
        if table.trajectory_id.tolist() != [TRAJECTORY_ID]*len(table): reasons.append("tcp_trajectory_id_invalid")
        if table.trajectory_profile.tolist() != [PROFILE]*len(table): reasons.append("tcp_profile_invalid")
        if not all(type(v) in (bool,np.bool_) and v for v in table.trajectory_valid): reasons.append("tcp_trajectory_invalid_rows")
        plans = table.tcp_plan_json.unique()
        if len(plans) != 1: raise ValueError("inconsistent continuous trajectory plans")
        spec, _, control, _ = _spec_model(json.loads(plans[0], object_pairs_hook=_strict_object))
        pose_columns = [f"tcp_{a}_base" for a in "xyz"]+[f"tcp_r{a}" for a in "xyz"]
        pose = table.iloc[0][pose_columns].to_numpy(float) if anchor is None else np.asarray(anchor.tcp_pose_base,float)
        expected, _, metadata = build_continuous_taught_trajectory(spec,current_tcp_start_pose=pose,rehab_frame=frame)
        numeric = ["time_s"]+pose_columns+[f"delta_{a}_R" for a in "xyz"]+[
            f"tcp_{d}{a}_base" for d in ("v","a") for a in "xyz"]
        if len(table) != len(expected) or not np.allclose(table[numeric].to_numpy(float),expected[numeric].to_numpy(float),atol=1e-10,rtol=0):
            reasons.append("tcp_plan_content_mismatch")
        if table.trajectory_phase.tolist() != expected.trajectory_phase.tolist(): reasons.append("tcp_phase_mismatch")
        if metadata["sampled_midcourse_chord_speed_relative_error"] > .01:
            reasons.append("continuous_speed_uniformity_failed")
        t = table.time_s.to_numpy(float)
        xyz = table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
        if not np.isfinite(t).all() or not np.isfinite(xyz).all() or np.any(np.diff(t)<=0):
            raise ValueError("invalid continuous time or position")
        v = np.gradient(xyz,t,axis=0,edge_order=2)
        a = np.gradient(v,t,axis=0,edge_order=2)
        speed, accel = kinematic_extrema(spec)
        speed=max(speed,float(np.linalg.norm(v,axis=1).max()))
        accel=max(accel,float(np.linalg.norm(a,axis=1).max()))
        if speed>safety.max_tcp_speed_m_s+1e-12: reasons.append("trajectory_speed_limit_exceeded")
        if accel>safety.max_tcp_acceleration_m_s2+1e-12: reasons.append("trajectory_acceleration_limit_exceeded")
        lower,upper=np.asarray(safety.workspace_min_base_m,float),np.asarray(safety.workspace_max_base_m,float)
        if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite(np.r_[lower,upper]).all() or np.any(lower>=upper):
            raise ValueError("invalid reviewed workspace")
        hull=pose[:3]+control
        if ((hull<lower)|(hull>upper)).any() or ((xyz<lower)|(xyz>upper)).any():
            reasons.append("trajectory_outside_reviewed_workspace")
        return tuple(dict.fromkeys(reasons)),speed,accel
    except (OSError,KeyError,ValueError,TypeError,IndexError,OverflowError,AttributeError):
        return ("tcp_plan_invalid",),None,None


def geometry_from_taught(settings, control_spacing_m=.02):
    """Round segmented joins into one C2 curve and preserve peak Y/Z excursions.

    This is an explicit geometric derivative, not a replay of unmodified manual
    samples. Source shape changes and peak restoration scales are quantified.
    """
    spec = _segmented_settings(settings)
    spacing = _positive(control_spacing_m,"control_spacing_m")
    if not .005 <= spacing <= .1: raise ValueError("control spacing must be in [0.005, 0.1] metres")
    original, controls = _load_segmented_geometry(spec)
    scale = np.array([spec[name] for name in SCALES])
    parts = [_spline(control*scale)(np.linspace(0.,1.,max(5001,100*len(control)))) for control in controls]
    reference = np.vstack([parts[0],parts[1][1:],parts[2][1:]])
    distance = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
    keep = np.r_[True,np.diff(distance)>1e-12]
    grid = np.linspace(0.,distance[-1],max(4,math.ceil(distance[-1]/spacing)+1))
    global_control=np.column_stack([np.interp(grid,distance[keep],reference[keep,i]) for i in range(3)])
    global_control[0]=global_control[-1]=0.
    before = _axis_extrema(_spline(global_control))
    old_extrema=[_axis_extrema(_spline(control*scale)) for control in controls]
    targets=np.array([max(extrema[i]["maximum"] for extrema in old_extrema) for i in range(3)])
    restoration=np.ones(3)
    for axis in (1,2):
        if targets[axis]<=0 or before[axis]["maximum"]<=0:
            raise ValueError("positive recorded Y/Z excursions required")
        restoration[axis]=targets[axis]/before[axis]["maximum"]
    global_control*=restoration
    model=_curve(global_control)
    dense=model["spline"](np.linspace(0.,1.,100001))
    distances_to_new=cKDTree(dense).query(reference)[0]
    distances_to_old=cKDTree(reference).query(dense)[0]
    changes=dict(method="densify each old cubic branch, resample the complete loop in arc length, then one global clamped cubic B-spline",
        control_spacing_m=spacing,control_count=len(global_control),
        peak_restoration_base_axis_scales=restoration.tolist(),
        preserved_positive_excursions_yz_m=targets[1:].tolist(),
        old_to_new_maximum_nearest_distance_m=float(distances_to_new.max()),
        new_to_old_maximum_nearest_distance_m=float(distances_to_old.max()),
        old_path_length_m=float(distance[-1]),new_path_length_m=model["length"],
        phase_boundary_definition="actual global maximum Base Z, then actual global maximum Base Y",
        no_two_dimensional_projection=True,phase_joins_rounded=True)
    source=dict(parent_geometry_path=spec["geometry_path"],parent_geometry_sha256=spec["geometry_sha256"],
        parent_settings=spec,parent_recording_source=original["source"],processing=changes,
        processing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hardware_accessed=False,ready_for_replay=False)
    geometry=dict(schema_version=1,recorded_anchor_pose=original["recorded_anchor_pose"],
        control_points_delta_base_m=global_control.tolist(),
        phase_boundaries_arc_fraction=model["phase_fractions"].tolist(),source=source)
    diagnostics=dict(source_dense_delta_base_m=reference,new_dense_delta_base_m=dense,
        maximum_nearest_distance_m=float(max(distances_to_new.max(),distances_to_old.max())),
        target_excursions_m=targets,processing=changes)
    return geometry,diagnostics
