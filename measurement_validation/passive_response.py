"""Descriptive passive-leg response, without a desired joint trajectory."""
from pathlib import Path
import numpy as np
import pandas as pd
from collection.twincat_leg import atomic_json
from measurement_validation.leg_episode import JointCalibration


def analyze_passive_response(directory, config):
    directory = Path(directory)
    leg = pd.read_csv(directory / "leg/leg.csv")
    timestamp = leg["host_monotonic_ns"].to_numpy(float) / 1e9
    if len(timestamp) < 2 or not np.isfinite(timestamp).all() or (np.diff(timestamp) <= 0).any():
        raise ValueError("passive response requires ordered finite leg timestamps")
    result = pd.DataFrame({"host_time_s": timestamp})
    for name, axis in (("hip", 1), ("knee", 4)):
        c = config["calibration"][name]
        conversion = JointCalibration(**{k: c[k] for k in (
            "counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")})
        result[name + "_deg"] = [conversion.to_degrees(v) for v in leg[f"axis{axis}_position_raw"]]
        result[name + "_torque_nm"] = leg[f"axis{axis}_torque_3c1a_nm"].to_numpy(float)
    wrench = pd.read_csv(directory / "robot/robot_wrench.csv")
    valid = wrench["valid"].astype(str).str.lower().eq("true")
    components = ["fx", "fy", "fz", "tx", "ty", "tz"]
    finite = np.isfinite(wrench[["query_start_s", "query_end_s"] + components].to_numpy(float)).all(axis=1)
    wrench = wrench.loc[valid & finite & (wrench["query_end_s"] >= wrench["query_start_s"])].copy()
    # Host query midpoint is an approximation, never a device timestamp.
    wrench["wrench_host_time_s"] = (wrench["query_start_s"] + wrench["query_end_s"])/2
    columns = ["wrench_host_time_s", "query_start_s", "query_end_s", "frame_type"] + components
    tolerance = min(config["motion"]["max_wrench_age_s"], config["motion"]["max_state_wrench_skew_s"])
    result = pd.merge_asof(result, wrench[columns].sort_values("wrench_host_time_s"),
        left_on="host_time_s", right_on="wrench_host_time_s", direction="backward", tolerance=tolerance)
    result["wrench_age_s"] = result["host_time_s"] - result["wrench_host_time_s"]
    result["wrench_matched"] = result["wrench_host_time_s"].notna()
    result["robot_wrench_force_magnitude_n"] = np.linalg.norm(result[["fx", "fy", "fz"]], axis=1)
    state = pd.read_csv(directory / "robot/robot_state.csv")
    state = state.loc[state["valid"].astype(str).str.lower().eq("true")].copy()
    state = state.rename(columns={"host_time_s": "robot_host_time_s"})
    state_columns = ["robot_host_time_s", "tcp_x", "tcp_y", "tcp_z"]
    state = state.loc[np.isfinite(state[state_columns].to_numpy(float)).all(axis=1)]
    result = pd.merge_asof(result, state[state_columns].sort_values("robot_host_time_s"),
        left_on="host_time_s", right_on="robot_host_time_s", direction="backward",
        tolerance=config["motion"]["max_state_age_s"])
    result["robot_state_matched"] = result["robot_host_time_s"].notna()
    result.to_csv(directory / "passive_response.csv", index=False)
    def span(column):
        values = result[column].to_numpy(float)
        values = values[np.isfinite(values)]
        return [float(values.min()), float(values.max())] if len(values) else None
    summary = {"mode": "independent_tcp", "rows": len(result),
        "duration_s": float(timestamp[-1]-timestamp[0]),
        "joint_tracking_error_applicable": False,
        "torque_offset_subtracted": False,
        "wrench_timestamp_source": "host_query_midpoint_approximation",
        "wrench_match_tolerance_s": tolerance,
        "wrench_matched_rows": int(result["wrench_matched"].sum()),
        "robot_state_matched_rows": int(result["robot_state_matched"].sum()),
        "ranges": {key: span(key) for key in ("hip_deg", "knee_deg", "hip_torque_nm",
            "knee_torque_nm", "robot_wrench_force_magnitude_n")},
        "force_interpretation": "robot TCP wrench magnitude; not isolated strap tension",
        "identification_ready": False}
    atomic_json(directory / "passive_response_summary.json", summary)
    return summary
