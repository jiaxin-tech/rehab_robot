"""Robot-only observations for the existing planar six-parameter model.

Reuse existing fixed-point geometry, inverse kinematics and J.T @ F; never
read dummy-leg sensors or substitute commanded joint angles for observations.
The existing observation equation is a point-force model. A free connection
moment, moving hip or slipping attachment is outside that model.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from collection.twincat_leg import positive
from control.start_anchored_relative_trajectory import RehabFrameConfig
from lower_limb_sim.kinematics import inverse_kinematics
from lower_limb_sim.jacobian import jacobian_diagnostics
from lower_limb_sim.observation_model import joint_torque_from_endpoint_force
from lower_limb_sim.mechanical_endpoints import branch_rms_components

SOURCE = "robot_tcp_point_force_v1"
ANALYSIS_FIELDS = ("max_gap_s", "max_read_duration_s", "max_dispatch_time_error_s",
                   "derivative_rate_hz", "derivative_window_s")


def _vector(value, name, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"robot_observation.{name} requires {shape} finite calibrated values")
    return result


def validate_robot_observation(config):
    if config["training"].get("observation_source") != "robot_only":
        raise ValueError("formal training requires observation_source=robot_only; dummy-leg sensors are validation only")
    spec = config.get("robot_observation", {})
    if spec.get("connection_model") != "fixed_tcp_to_pull_point":
        raise ValueError("robot_observation requires the existing fixed TCP-to-pull-point geometry")
    for key in ("hip_center_in_base_m", "tcp_to_pull_point_in_tcp_m", "wrench_origin_in_tcp_m"):
        _vector(spec.get(key), key, (3,))
    for key in ("L1_m", "L2_m", "max_out_of_plane_position_m",
                "max_out_of_plane_force_n", "max_connection_moment_nm"):
        positive(spec.get(key), "robot_observation." + key)
    rotation = _vector(spec.get("rotation_base_from_wrench"), "rotation_base_from_wrench", (3, 3))
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)):
        raise ValueError("robot_observation wrench rotation must be orthonormal and right handed")
    if spec.get("wrench_frame_type") not in ("world", "base"):
        raise ValueError("robot_observation requires a fixed world/base wrench expression frame")
    if type(spec.get("force_robot_on_leg_sign")) is not int or spec["force_robot_on_leg_sign"] not in (-1, 1):
        raise ValueError("robot_observation.force_robot_on_leg_sign must be calibrated +1 or -1")
    _vector(spec.get("unloaded_wrench_bias"), "unloaded_wrench_bias", (6,))
    if spec.get("compensation_verified") is not True:
        raise ValueError("robot_observation requires verified external-wrench/tool compensation; loaded taring is not calibration")
    for key in ("state_time_offset_s", "wrench_time_offset_s"):
        value = spec.get(key)
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError("robot_observation." + key + " requires a calibrated finite clock correction")
    for key in ANALYSIS_FIELDS:
        positive(config["analysis"].get(key), key)
    positive(config["training"].get("timeout_s"), "training.timeout_s")
    return spec


def _samples(frame, clock, columns, begin, end, gap, name):
    values = frame[columns].to_numpy(float)
    if (len(clock) < 3 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0)):
        raise ValueError(name + " requires ordered finite observation timestamps")
    good = frame.valid.astype(str).str.lower().eq("true").to_numpy() & np.isfinite(values).all(axis=1)
    active = (clock >= begin) & (clock <= end)
    if not good[active].all():
        raise ValueError(name + " has invalid measurements during motion")
    stamp, values = clock[good], values[good]
    if len(stamp) < 3 or stamp[0] > begin or stamp[-1] < end:
        raise ValueError(name + " does not bracket the full motion")
    lo, hi = max(0, np.searchsorted(stamp, begin)-1), min(len(stamp)-1, np.searchsorted(stamp, end))
    if np.max(np.diff(stamp[lo:hi+1])) > gap + 1e-9:
        raise ValueError(name + " gap exceeds analysis budget; no interpolation across missing data")
    return stamp[lo:hi+1], values[lo:hi+1]


def robot_measurement(directory, config, reference, execution):
    """Construct model-derived q and generalized torque from measured robot CSVs."""
    spec = validate_robot_observation(config)
    if execution["schedule_origin_source"] != "recorded_schedule_origin":
        raise ValueError("robot-only alignment requires the recorded scheduler origin")
    analysis = config["analysis"]
    time = reference.time_s.to_numpy(float)
    if len(time) < 5 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("original candidate needs finite increasing timestamps")
    if set(reference.cycle_phase) != {"flexion", "extension"}:
        raise ValueError("original candidate requires both motion phases")
    origin = execution["schedule_origin_host_time_s"]
    target = origin + time
    dispatch = pd.read_csv(directory / "dispatch.csv")
    if (len(dispatch) != len(time) or not np.allclose(dispatch.trajectory_time_s, time, atol=1e-9, rtol=0)
        or np.max(np.abs(dispatch.dispatch_end_ns.to_numpy(float)[1:]/1e9-target[1:])) > analysis["max_dispatch_time_error_s"]):
        raise ValueError("dispatch timing exceeded robot-observation analysis budget")
    robot = pd.read_csv(directory / "robot/robot_state.csv")
    state_clock = robot.host_time_s.to_numpy(float) + spec["state_time_offset_s"]
    stamp, poses = _samples(robot, state_clock,
        ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"],
        target[0], target[-1], analysis["max_gap_s"], "robot state")
    rotation_tcp = Rotation.from_euler("xyz", poses[:, 3:]).as_matrix()
    frame = RehabFrameConfig(tuple(config["frame"]["rehab_x_axis_in_base"]),
                             tuple(config["frame"]["rehab_z_axis_in_base"]))
    rotation = frame.rotation_base_from_rehab
    point = poses[:, :3] + rotation_tcp @ np.asarray(spec["tcp_to_pull_point_in_tcp_m"])
    human = (point-np.asarray(spec["hip_center_in_base_m"])) @ rotation
    if np.abs(human[:, 1]).max() > spec["max_out_of_plane_position_m"]:
        raise ValueError("measured pull point is outside the original sagittal model")
    n = int(np.ceil((target[-1]-target[0])*analysis["derivative_rate_hz"]))+1
    if n < 5 or n > 2_000_000:
        raise ValueError("invalid robot-observation derivative resampling size")
    grid = np.linspace(target[0], target[-1], n)
    delta = grid[1]-grid[0]
    window = int(round(analysis["derivative_window_s"]/delta)) | 1
    if window < 5 or window > n:
        raise ValueError("derivative window must contain at least five samples and fit the recording")
    qh, qk, reachable = inverse_kinematics(np.interp(grid, stamp, human[:, 0]),
        np.interp(grid, stamp, human[:, 2]), spec["L1_m"], spec["L2_m"])
    if not np.all(reachable):
        raise ValueError("measured TCP cannot reconstruct an admissible leg posture with calibrated geometry")
    if np.any(jacobian_diagnostics(qh, qk, spec["L1_m"], spec["L2_m"]).near_singular):
        raise ValueError("robot-only posture reconstruction is near a Jacobian singularity")
    output = {"time_s": time-time[0], "command_time_s": time,
              "phase": reference.cycle_phase.to_numpy(), "observation_source": SOURCE}
    for joint, angle in (("hip", qh), ("knee", qk)):
        for name, derivative in ((f"q_{joint}_rad", 0), (f"dq_{joint}_rad_s", 1), (f"ddq_{joint}_rad_s2", 2)):
            values = savgol_filter(angle, window, 3, deriv=derivative, delta=delta, mode="interp")
            output[name] = np.interp(target, grid, values)
    wrench = pd.read_csv(directory / "robot/robot_wrench.csv")
    query = wrench[["query_start_s", "query_end_s"]].to_numpy(float)
    wrench_clock = query.mean(axis=1) + spec["wrench_time_offset_s"]
    active = (wrench_clock >= target[0]) & (wrench_clock <= target[-1])
    duration = query[:, 1]-query[:, 0]
    if (not np.isfinite(query).all() or np.any(duration[active] < 0)
        or np.any(duration[active] > analysis["max_read_duration_s"])):
        raise ValueError("robot wrench query-duration uncertainty exceeds analysis budget")
    wt, raw = _samples(wrench, wrench_clock, ["fx", "fy", "fz", "tx", "ty", "tz"],
        target[0], target[-1], analysis["max_gap_s"], "robot wrench")
    used = (wrench_clock >= wt[0]) & (wrench_clock <= wt[-1])
    if np.any(duration[used] < 0) or np.any(duration[used] > analysis["max_read_duration_s"]):
        raise ValueError("bracketing robot wrench query-duration uncertainty exceeds analysis budget")
    if not wrench.loc[used, "frame_type"].eq(spec["wrench_frame_type"]).all():
        raise ValueError("robot wrench frame differs from observation calibration")
    corrected = raw - np.asarray(spec["unloaded_wrench_bias"])
    rw = np.asarray(spec["rotation_base_from_wrench"])
    force_base = corrected[:, :3] @ rw.T * spec["force_robot_on_leg_sign"]
    moment_base = corrected[:, 3:] @ rw.T * spec["force_robot_on_leg_sign"]
    # Check the point-force assumption at observed wrench times. The audit has
    # already bounded orientation changes; unwrap avoids an Euler +/-pi jump.
    check_time = np.clip(wt, target[0], target[-1])
    angles = np.unwrap(poses[:, 3:], axis=0)
    orientation = np.column_stack([np.interp(check_time, stamp, angles[:, axis]) for axis in range(3)])
    lever = Rotation.from_euler("xyz", orientation).apply(
        np.asarray(spec["tcp_to_pull_point_in_tcp_m"])-np.asarray(spec["wrench_origin_in_tcp_m"]))
    connection_moment = moment_base - np.cross(lever, force_base)
    if np.linalg.norm(connection_moment, axis=1).max() > spec["max_connection_moment_nm"]:
        raise ValueError("free connection moment is outside the existing J.T @ F point-force observation model")
    force_human = force_base @ rotation
    if np.abs(force_human[:, 1]).max() > spec["max_out_of_plane_force_n"]:
        raise ValueError("out-of-plane force is outside the original planar observation model")
    output["fx_observed_n"] = np.interp(target, wt, force_human[:, 0])
    output["fz_observed_n"] = np.interp(target, wt, force_human[:, 2])
    output["tau_hip_nm"], output["tau_knee_nm"] = joint_torque_from_endpoint_force(
        output["q_hip_rad"], output["q_knee_rad"], output["fx_observed_n"], output["fz_observed_n"],
        spec["L1_m"], spec["L2_m"])
    measured = pd.DataFrame(output)
    if not np.isfinite(measured.select_dtypes(include="number")).all().all():
        raise ValueError("nonfinite robot-only model observations")
    components = branch_rms_components(measured.tau_hip_nm, measured.tau_knee_nm, measured.time_s, measured.phase)
    peaks = np.abs(measured[["tau_hip_nm", "tau_knee_nm"]]).max(axis=0).to_numpy()
    return measured, {"components_nm": list(components), "peaks_nm": peaks.tolist(),
        "observation_source": SOURCE, "dummy_leg_data_used": False,
        "kinematics_source": "measured robot TCP + fixed calibrated geometry + existing inverse_kinematics",
        "torque_source": "model-derived generalized torque using existing joint_torque_from_endpoint_force",
        "not_direct_human_joint_torque": True,
        "limitations": "fixed hip, fixed attachment, planar admissible branch, point force, compensated external wrench; no identifiability guarantee",
        "derivative_method": {"method": "Savitzky-Golay degree 3", "window_samples": window, "delta_s": delta}}
