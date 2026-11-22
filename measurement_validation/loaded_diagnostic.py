"""Whole-motion passive-leg evidence for a diagnostic, never a learner input."""
from pathlib import Path

import numpy as np

from measurement_validation.leg_episode import JointCalibration, quality_report, read_leg_episode


def audit_loaded_leg(directory, config, robot_audit):
    """Reject unobserved motion, invalid drive state, torque or calibrated ROM.

    This uses the same host monotonic clock as robot capture. It reports the
    stricter dynamics-quality assessment separately; no resampling fills gaps.
    """
    calibration = {name: JointCalibration(**{key: values[key] for key in
        ("counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")})
        for name, values in config["calibration"].items()}
    episode = read_leg_episode(Path(directory) / "leg/leg.csv", calibration=calibration)
    # The general importer prefers receive_elapsed_ms for within-file quality.
    # Robot/ADS alignment must use the actual same-host absolute clock instead.
    clock = episode.source_frame["host_monotonic_ns"].to_numpy(float) / 1e9
    if len(clock) < 3 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError("loaded leg requires finite ordered receive times")
    origin = float(robot_audit["schedule_origin_host_time_s"])
    end = origin + float(robot_audit["duration_s"])
    if not np.isfinite([origin, end]).all() or end <= origin:
        raise ValueError("invalid loaded motion time window")
    budget = config["acquisition"]["max_leg_start_age_s"]
    inside = (clock >= origin) & (clock <= end)
    samples = clock[inside]
    if (len(samples) < 3 or clock[0] > origin or clock[-1] < end
        or np.max(np.diff(np.r_[origin, samples, end])) > budget):
        raise ValueError("loaded leg does not cover the entire motion within the freshness budget")
    query = episode.source_frame[["query_start_ns", "query_end_ns"]].to_numpy(float) / 1e9
    if (not np.isfinite(query).all() or np.any(query[:, 1] < query[:, 0])
        or np.any(query[:, 1] - query[:, 0] > budget)
        or not np.allclose(query[:, 1], clock, atol=1e-8, rtol=0)):
        raise ValueError("invalid or stale loaded ADS query interval")
    axes = {}
    for name, joint in episode.joints.items():
        fields = (joint.position_counts, joint.flexion_deg, joint.torque_nm,
                  joint.statusword, joint.error_code, joint.fault, joint.operation_enabled)
        if any(values is None or not np.isfinite(values).all() for values in fields):
            raise ValueError(name + " missing or nonfinite passive-leg fields")
        status = joint.statusword.astype(np.int64)
        if np.any(joint.statusword != status) or np.any(status < 0) or np.any(status > 65535):
            raise ValueError(name + " invalid statusword")
        if np.any(joint.error_code != 0) or np.any(joint.fault != 0) or np.any(status & 8):
            raise ValueError(name + " drive fault in recording")
        if np.any(joint.operation_enabled != 0) or np.any(status & 4):
            raise ValueError(name + " drive enabled during passive recording")
        c = calibration[name]
        if np.any((joint.flexion_deg < c.safe_min_deg) | (joint.flexion_deg > c.safe_max_deg)):
            raise ValueError(name + " outside physical calibrated range")
        peak = float(np.max(np.abs(joint.torque_nm)))
        guard = config["leg"]["guards"][f"axis{joint.axis}"]
        if guard["expected_operation_enabled"] is not False or peak > guard["max_abs_torque_nm"]:
            raise ValueError(name + " passive torque/enable guard violated")
        axes[name] = {"angle_range_deg": [float(joint.flexion_deg.min()), float(joint.flexion_deg.max())],
                      "raw_torque_peak_nm": peak}
    quality = quality_report(episode, expected_period_s=1 / config["leg"]["expected_hz"])
    return {"purpose": "connection_diagnostic_only", "motion_rows": int(inside.sum()),
            "duration_s": end - origin, "max_motion_receive_gap_s": float(np.max(np.diff(np.r_[origin, samples, end]))),
            "axes": axes, "quality": quality, "training_authorized": False,
            "comfort_claim_authorized": False, "interpolated_samples": 0}
