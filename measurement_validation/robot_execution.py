"""Execution evidence audit only; no optimizer, learner, or new training endpoint."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from control.prepared_trajectory import build_prepared_trajectory, assert_frozen_trajectory
from control.execution_preflight import audit_execution_trajectory_content
from measurement_validation.wrench_endpoints import load_window
from safety.experiment_safety import ExperimentSafetyConfig

EVIDENCE = ("prepared_config.json", "tcp_plan.json", "anchor.json", "tcp_safety.json",
            "tcp_trajectory.csv", "dispatch.csv", "robot/metadata.json",
            "robot/robot_wrench.csv", "robot/robot_state.csv", "robot/trajectory_command.csv")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence_hashes(directory):
    names = EVIDENCE + (("original_candidate.csv",) if (directory / "original_candidate.csv").is_file() else ())
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def _ordered(values, name):
    if len(values) < 3 or not np.isfinite(values).all() or (np.diff(values) <= 0).any():
        raise ValueError(name + " requires finite, strictly ordered samples")


def audit_robot_execution(directory, *, allow_empty=False):
    """Validate an entire executed schedule and measure each moving branch."""
    directory = Path(directory).resolve()
    state = read_json(directory / "round_status.json")
    if state.get("trajectory_mode") not in ("independent_tcp", "original_candidate") or (state.get("empty_load") and not allow_empty):
        raise ValueError("execution audit requires a completed passive TCP motion (or explicit empty-load audit)")
    if state["status"] not in ("CAPTURE_COMPLETE", "ROUND_COMPLETE", "EMPTY_LOAD_COMPLETE", "COMPLETE"):
        raise ValueError("execution audit requires a completed motion")
    execution = state.get("execution", {})
    if (execution.get("completed") is not True
        or execution.get("same_session_realtime_cleanup_completed") is not True):
        raise ValueError("execution audit requires completed realtime cleanup")
    config = read_json(directory / "prepared_config.json")
    plan = read_json(directory / "tcp_plan.json")
    anchor = load_start_anchor(directory / "anchor.json")
    if plan != state["tcp_plan"] or plan["anchor"] != anchor.to_dict():
        raise ValueError("execution plan or anchor changed")
    frame = load_rehab_frame_config(directory / "prepared_config.json")
    trajectory, _, _ = build_prepared_trajectory(plan,
        current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    assert_frozen_trajectory(trajectory, directory / "tcp_trajectory.csv")
    expected_safety = dict(config["motion"])
    for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        expected_safety[key] = min(expected_safety[key], config["empty_load"][key])
    if read_json(directory / "tcp_safety.json") != expected_safety:
        raise ValueError("execution safety differs from prepared configuration")
    safety = ExperimentSafetyConfig(**expected_safety)
    safety.require_execute_allowed()
    reasons, _, _ = audit_execution_trajectory_content(trajectory, safety,
        expected_trajectory_id=anchor.trajectory_id, frame=frame, anchor=anchor)
    if reasons:
        raise ValueError( "execution trajectory audit: " + ";".join(reasons))
    metadata = read_json(directory / "robot/metadata.json")
    if metadata["logger"]["status"] != "completed" or metadata["execution_result"]["completed"] is not True:
        raise ValueError("robot recording did not close successfully")
    dispatch = pd.read_csv(directory / "dispatch.csv")
    times = trajectory.time_s.to_numpy(float)
    if len(dispatch) != len(times) or execution.get("commands_dispatched") != len(times):
        raise ValueError("incomplete dispatch schedule")
    if not np.allclose(dispatch.trajectory_time_s, times, atol=1e-9, rtol=0):
        raise ValueError("dispatch times do not match frozen trajectory")
    commands = pd.read_csv(directory / "robot/trajectory_command.csv")
    command_pose = commands[[f"tcp_target_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]].to_numpy(float)
    expected_pose = trajectory[[f"tcp_{axis}_base" for axis in "xyz"] + [f"tcp_r{axis}" for axis in "xyz"]].to_numpy(float)
    if (len(commands) != len(times) or not commands.command_valid.astype(str).str.lower().eq("true").all()
        or not np.allclose(commands.trajectory_time_s, times, atol=1e-9, rtol=0)
        or not np.allclose(command_pose, expected_pose, atol=1e-10, rtol=0)):
        raise ValueError("command intent log does not match frozen trajectory")
    starts = dispatch.dispatch_start_ns.to_numpy(float) / 1e9
    ends = dispatch.dispatch_end_ns.to_numpy(float) / 1e9
    _ordered(starts, "dispatch")
    if not np.isfinite(ends).all() or (ends < starts).any():
        raise ValueError("invalid dispatch duration")
    origin = metadata["execution_result"].get("schedule_origin_host_time_s")
    origin_source = "recorded_schedule_origin"
    if origin is None:
        # Historical recordings did not retain the scheduler's origin.
        origin = float(np.median(starts[1:] - times[1:]))
        origin_source = "historical_dispatch_median_approximation"
    errors = starts[1:] - times[1:] - float(origin)
    if not np.isfinite(origin) or np.max(np.abs(errors)) > safety.max_command_lateness_s:
        raise ValueError("dispatch timing uncertainty exceeded")
    wrench = pd.read_csv(directory / "robot/robot_wrench.csv")
    components = ["fx", "fy", "fz", "tx", "ty", "tz"]
    query = wrench[["query_start_s", "query_end_s"]].to_numpy(float)
    clock = query.mean(axis=1)
    _ordered(clock, "wrench")
    raw = wrench[components].to_numpy(float)
    valid = (wrench.valid.astype(str).str.lower().eq("true").to_numpy()
             & np.isfinite(raw).all(axis=1) & (query[:, 1] >= query[:, 0])
             & (query[:, 1] - query[:, 0] <= config["acquisition"]["budgets"]["query_s"]))
    moving = np.zeros(len(clock), bool)
    force, torque = np.linalg.norm(raw[:, :3], axis=1), np.linalg.norm(raw[:, 3:], axis=1)
    if plan.get("mode") == "original_candidate":
        intervals = [(float(trajectory.loc[trajectory.trajectory_phase == phase, "time_s"].min()),
                      float(trajectory.loc[trajectory.trajectory_phase == phase, "time_s"].max()))
                     for phase in ("flexion", "extension")]
    else:
        phases = plan["tcp_settings"]
        intervals = ((phases["start_hold_s"], phases["start_hold_s"] + phases["lift_duration_s"]),
                     (phases["start_hold_s"] + phases["lift_duration_s"] + phases["peak_hold_s"], times[-1]))
    metrics = []
    for begin, end in intervals:
        mask = (clock >= origin + begin) & (clock <= origin + end)
        good = mask & valid
        stamp = clock[good]
        if mask.sum() < 3 or good.sum() < 3:
            raise ValueError("insufficient valid robot wrench in motion branch")
        if np.diff(np.r_[origin + begin, stamp, origin + end]).max() > safety.max_wrench_age_s:
            raise ValueError("wrench gap in motion; no interpolation across missing data")
        measured = load_window(clock, np.column_stack([force, torque]), valid,
                               origin + begin, origin + end, safety.max_wrench_age_s)
        moving[measured["source_indices"]] = True
        metrics.append(measured["rms"])
    if wrench.loc[moving, "frame_type"].nunique(dropna=False) != 1 or wrench.loc[moving, "frame_type"].isna().any():
        raise ValueError("mixed or missing wrench frame")
    cycle = (clock >= origin) & (clock <= origin + times[-1])
    whole = cycle & valid
    cycle_metrics = load_window(clock, np.column_stack([force, torque]), valid,
                                origin, origin + times[-1], safety.max_wrench_age_s)
    whole[cycle_metrics["source_indices"]] = True
    if (np.diff(np.r_[origin, clock[whole], origin + times[-1]]).max() > safety.max_wrench_age_s):
        raise ValueError("incomplete full-cycle wrench data")
    if wrench.loc[whole, "frame_type"].nunique(dropna=False) != 1 or wrench.loc[whole, "frame_type"].isna().any():
        raise ValueError("mixed or missing full-cycle wrench frame")
    if cycle_metrics["peaks"][0] > safety.max_force_n or cycle_metrics["peaks"][1] > safety.max_torque_nm:
        raise ValueError("recorded force or torque exceeds physical safety limit")
    robot = pd.read_csv(directory / "robot/robot_state.csv")
    robot_time = robot.host_time_s.to_numpy(float)
    _ordered(robot_time, "robot state")
    xyz = robot[["tcp_x", "tcp_y", "tcp_z"]].to_numpy(float)
    rpy = robot[["tcp_rx", "tcp_ry", "tcp_rz"]].to_numpy(float)
    good = (robot.valid.astype(str).str.lower().eq("true").to_numpy()
            & np.isfinite(xyz).all(axis=1) & np.isfinite(rpy).all(axis=1)
            & (robot_time >= origin) & (robot_time <= origin + times[-1]))
    stamp = robot_time[good]
    if len(stamp) < 3 or np.diff(np.r_[origin, stamp, origin + times[-1]]).max() > safety.max_state_age_s:
        raise ValueError("missing measured robot trajectory")
    expected = np.column_stack([np.interp(stamp-origin, times, trajectory[f"tcp_{a}_base"]) for a in "xyz"])
    if np.linalg.norm(xyz[good] - expected, axis=1).max() > safety.max_start_anchor_position_error_m:
        raise ValueError("robot did not follow planned TCP path")
    orientation_error = np.arctan2(np.sin(rpy[good]-anchor.tcp_pose_base[3:]), np.cos(rpy[good]-anchor.tcp_pose_base[3:]))
    if np.linalg.norm(orientation_error, axis=1).max() > safety.max_start_anchor_orientation_error_rad:
        raise ValueError("robot did not preserve planned TCP orientation")
    return {"directory": str(directory),
        "candidate_id": plan.get("candidate_id"),
        "schedule_origin_host_time_s": origin, "schedule_origin_source": origin_source,
        "components": np.asarray(metrics).T.reshape(-1).tolist(),
        "peaks": cycle_metrics["peaks"].tolist(),
        "endpoint_method": cycle_metrics["method"],
        "frame_type": str(wrench.loc[moving, "frame_type"].iloc[0]),
        "anchor_pose": list(anchor.tcp_pose_base),
        "evidence_hashes": evidence_hashes(directory)}

