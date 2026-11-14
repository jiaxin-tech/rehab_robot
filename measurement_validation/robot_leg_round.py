"""Measured torque endpoints and kinematics on one local monotonic clock."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from collection.twincat_leg import positive
from measurement_validation.leg_episode import JointCalibration, read_leg_episode
from measurement_validation.torque_endpoints import (
    RAW_ENDPOINT_METHOD, endpoints, raw_torque_endpoint_trace, torque_metrics_from_trace,
)
from safety.robot_trial_release import read_json, sha256_file


def validate_analysis(config, calibration):
    if config.get("reviewed") is not True or config.get("alignment_reviewed") is not True:
        raise ValueError("reviewed cross-device timing and analysis configuration required")
    for key in ("max_gap_s", "max_read_duration_s", "max_dispatch_time_error_s", "derivative_rate_hz",
                "derivative_window_s", "max_start_angle_error_deg", "max_tracking_error_deg"):
        positive(config.get(key), key)
    for key in ("sensor_receive_delay_s", "command_application_delay_s"):
        v = config.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"reviewed finite nonnegative {key} required")
    if calibration.get("reviewed") is not True or calibration.get("independent_validation_passed") is not True:
        raise ValueError("independently validated hip AND knee calibration required")
    result = {}
    for name in ("hip", "knee"):
        values = calibration["joints"][name]
        if values is None:
            raise ValueError(f"missing {name} calibration")
        result[name] = JointCalibration(**{key: values[key] for key in
            ("counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")})
        if type(values.get("torque_sign_to_inverse_dynamics")) is not int or values["torque_sign_to_inverse_dynamics"] not in (-1, 1):
            raise ValueError(f"reviewed action/reaction torque sign required for {name}")
    return result


def aligned_measurement(csv_path, dispatch_path, reference_path, analysis, calibration):
    conversions = validate_analysis(analysis, calibration)
    episode = read_leg_episode(csv_path)
    frame = episode.source_frame
    reference = pd.read_csv(reference_path)
    dispatch = pd.read_csv(dispatch_path)
    expected = reference.time_s.to_numpy(float)
    if len(expected) < 5 or not np.isfinite(expected).all() or np.any(np.diff(expected) <= 0):
        raise ValueError("finite increasing reference time required")
    phases = reference.cycle_phase.to_numpy()
    transitions = phases[np.r_[True, phases[1:] != phases[:-1]]].tolist()
    if transitions != ["flexion", "extension"] or min(np.sum(phases == p) for p in transitions) < 2:
        raise ValueError("complete flexion then extension motion phases required")
    if (len(dispatch) != len(reference) or
            not np.allclose(dispatch.trajectory_time_s, expected, rtol=0, atol=1e-9)):
        raise ValueError("dispatch log does not cover the complete reference")
    required = ("host_monotonic_ns", "query_start_ns", "query_end_ns")
    if any(c not in frame for c in required):
        raise ValueError("same-host monotonic acquisition/query timestamps required")
    # Subtract integer origin before converting large tick values to floating seconds.
    origin_ns = int(dispatch.dispatch_end_ns.iloc[0])
    if np.any(dispatch.dispatch_start_ns > dispatch.dispatch_end_ns):
        raise ValueError("negative dispatch duration")
    receive = np.array([(int(v)-origin_ns)/1e9 for v in frame.host_monotonic_ns])
    sensor = receive - analysis["sensor_receive_delay_s"]
    target = np.array([(int(v)-origin_ns)/1e9 for v in dispatch.dispatch_end_ns]) + analysis["command_application_delay_s"]
    if np.any(np.diff(sensor) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("non-increasing sensor or dispatch clock")
    if np.max(np.abs((target-target[0]) - (expected-expected[0]))) > analysis["max_dispatch_time_error_s"]:
        raise ValueError("dispatch timing exceeded reviewed analysis budget")
    if sensor[0] > target[0] or sensor[-1] < target[-1]:
        raise ValueError("leg recording does not bracket all motion targets")
    lo = max(0, np.searchsorted(sensor, target[0])-1)
    hi = min(len(sensor)-1, np.searchsorted(sensor, target[-1]))
    if np.any(np.diff(sensor[lo:hi+1]) > analysis["max_gap_s"]):
        raise ValueError("long sensor interval in motion; no interpolation across missing intervals")
    durations = np.array([(int(b)-int(a))/1e9 for a, b in zip(frame.query_start_ns, frame.query_end_ns)])
    if np.any(durations < 0) or np.max(durations[lo:hi+1]) > analysis["max_read_duration_s"]:
        raise ValueError("ADS read-duration uncertainty exceeded reviewed budget")
    for joint in episode.joints.values():
        for attr in ("position_counts", "torque_nm", "statusword", "error_code", "fault", "operation_enabled"):
            values = getattr(joint, attr)
            if values is None or not np.isfinite(values[lo:hi+1]).all():
                raise ValueError(f"missing/nonfinite {joint.joint} {attr}")
        if np.any(joint.error_code[lo:hi+1] != 0) or np.any(joint.fault[lo:hi+1] != 0):
            raise ValueError(f"{joint.joint} fault during motion")
    n = int(np.ceil((target[-1]-target[0])*analysis["derivative_rate_hz"]))+1
    if n < 5 or n > 2_000_000:
        raise ValueError("invalid derivative resampling size")
    grid = np.linspace(target[0], target[-1], n)
    delta = grid[1]-grid[0]
    window = int(round(analysis["derivative_window_s"]/delta)) | 1
    if window < 5 or window > n:
        raise ValueError("derivative window must contain at least five samples and fit in recording")
    output = {"time_s": target-target[0], "phase": reference.cycle_phase.to_numpy(),
              "command_time_s": expected}
    for name, joint in episode.joints.items():
        degrees = conversions[name].to_degrees(joint.position_counts)
        conv = conversions[name]
        if np.any(degrees[lo:hi+1] < conv.safe_min_deg) or np.any(degrees[lo:hi+1] > conv.safe_max_deg):
            raise ValueError(f"{name} measured angle outside reviewed range")
        q = np.deg2rad(np.interp(grid, sensor, degrees))
        q_filtered = savgol_filter(q, window, 3, mode="interp")
        actual = np.interp(target, grid, q_filtered)
        planned = reference[f"q_{name}_rad"].to_numpy(float)
        if abs(np.rad2deg(actual[0]-planned[0])) > analysis["max_start_angle_error_deg"]:
            raise ValueError(f"{name} start angle inconsistent with reference")
        if np.max(np.abs(np.rad2deg(actual-planned))) > analysis["max_tracking_error_deg"]:
            raise ValueError(f"{name} tracking deviation invalidates this training episode")
        output[f"q_{name}_rad"] = actual
        output[f"dq_{name}_rad_s"] = np.interp(target, grid, savgol_filter(q, window, 3, deriv=1, delta=delta, mode="interp"))
        output[f"ddq_{name}_rad_s2"] = np.interp(target, grid, savgol_filter(q, window, 3, deriv=2, delta=delta, mode="interp"))
        sign = calibration["joints"][name]["torque_sign_to_inverse_dynamics"]
        output[f"tau_{name}_nm"] = sign * np.interp(target, sensor, joint.torque_nm)
    measured = pd.DataFrame(output)
    if not np.isfinite(measured.select_dtypes(include="number")).all().all():
        raise ValueError("nonfinite aligned training data")
    # The model keeps one measured row per dispatched command. Mechanical
    # endpoints instead use every raw sample in the complete motion interval:
    # resampling force to the coarser command grid can erase short load peaks.
    turn = int(np.flatnonzero(phases == "flexion")[-1])
    raw_torque = {name: calibration["joints"][name]["torque_sign_to_inverse_dynamics"]
                  * joint.torque_nm[lo:hi+1] for name, joint in episode.joints.items()}
    trace = raw_torque_endpoint_trace(sensor[lo:hi+1], raw_torque["hip"], raw_torque["knee"],
                                     motion_start_s=target[0], phase_boundary_s=target[turn],
                                     motion_end_s=target[-1])
    metrics = torque_metrics_from_trace(trace)
    config_digest = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return measured, dict(**metrics, endpoint_trace=trace,
        endpoint_sampling=dict(method=RAW_ENDPOINT_METHOD,
            phase_boundary_convention=trace["phase_boundary_convention"],
            raw_samples_in_motion_interval=trace["raw_samples_in_motion_interval"],
            endpoint_samples=len(trace["time_s"]), model_samples=len(measured),
            rms_integration="trapezoidal squared raw torque on actual sensor time; exact shared branch boundary",
            boundary_interpolation="piecewise linear raw torque at motion start, turning boundary and end",
            note="Keeps recorded transients; does not recover peaks absent from the sensor stream."),
        source_sha256=sha256_file(csv_path), dispatch_sha256=sha256_file(dispatch_path),
        reference_sha256=sha256_file(reference_path), synchronization="reviewed same-host receive/dispatch alignment; not device hardware synchronization",
        analysis_config_sha256=config_digest(analysis), calibration_config_sha256=config_digest(calibration),
        configuration_hash_encoding="UTF-8 JSON, sorted keys, separators comma/colon, no nonfinite values",
        derivative_method=dict(method="Savitzky-Golay degree 3 on uniformly resampled measured angles", window_samples=window, delta_s=delta),
        raw_torque_preserved=True, automatically_tared=False)
