"""Finite TCP candidate learning from robot measurements only.

Reuse the existing Matérn GP, expected improvement and constraint score. This
is the explicitly authorized robot-load method, not six-parameter inference.
No patient geometry, inverse kinematics or dummy-leg data enters this module.
"""
import hashlib
from pathlib import Path
import time

import numpy as np

from collection.twincat_leg import atomic_json, positive
from control.independent_tcp_trajectory import build_tcp_trajectory, audit_tcp_trajectory
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from control.start_anchor import load_start_anchor
from experiment_algorithm.surrogate import MaternGP
from experiment_algorithm.policy import expected_improvement, constraint_probability
from measurement_validation.robot_execution import audit_robot_execution, read_json
from safety.experiment_safety import ExperimentSafetyConfig
from measurement_validation.force_trial_ledger import identity as study_identity, snapshot as attempt_snapshot

METHOD = "TCP_FORCE_GP_EI"
FEATURES = ("horizontal_displacement_m", "lift_duration_s", "return_duration_s")
SETTINGS = ("target_peak_tcp_z_base_m", *FEATURES, "start_hold_s", "peak_hold_s", "sample_period_s")
OUTPUTS = ("load_score", "force_peak_fraction", "torque_peak_fraction")
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/tcp_force_training.py", "measurement_validation/robot_execution.py",
           "measurement_validation/force_trial_ledger.py", "measurement_validation/wrench_endpoints.py",
           "control/independent_tcp_trajectory.py", "experiment_algorithm/surrogate.py",
           "experiment_algorithm/policy.py")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCES}


def active(config):
    return (config.get("training", {}).get("mode") == "tcp_force"
            and config.get("trajectory", {}).get("mode") == "independent_tcp")


def validate_config(config, *, measurement=True):
    settings = config["force_training"]
    candidates = settings["candidates"]
    ids = [p["candidate_id"] for p in candidates]
    if not ids or any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("force training requires a nonempty unique TCP candidate catalog")
    if settings["reference_candidate_id"] not in ids:
        raise ValueError("force reference candidate missing")
    if type(settings["total_budget"]) is not int or settings["total_budget"] < 1:
        raise ValueError("force total_budget must be a positive integer")
    study_identity(config)
    if "evaluation_budget" in settings and (type(settings["evaluation_budget"]) is not int or settings["evaluation_budget"] < 1):
        raise ValueError("force evaluation_budget must be a positive integer")
    if "fixed_cycle_duration_s" in settings:
        positive(settings["fixed_cycle_duration_s"], "force_training.fixed_cycle_duration_s")
    for name in ("gp_length_scale", "gp_signal_floor", "gp_noise_std", "max_cycle_duration_s", "min_vertical_excursion_m"):
        positive(settings[name], "force_training." + name)
    scales = np.asarray(settings["feature_scales"], float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("force feature_scales requires three positive values")
    for key in ("horizontal_displacement_bounds_m", "branch_duration_bounds_s"):
        bounds = np.asarray(settings[key], float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("invalid force task bounds: " + key)
    if settings["branch_duration_bounds_s"][0] <= 0:
        raise ValueError("force task must retain a positive motion duration")
    if (not .5 < settings["feasibility_probability"] < 1
        or not np.isfinite(settings["minimum_gain"]) or settings["minimum_gain"] < 0):
        raise ValueError("invalid force decision threshold")
    if measurement:
        if settings.get("compensation_verified") is not True:
            raise ValueError("force training requires verified robot external-wrench/tool compensation")
        bias = np.asarray(settings.get("unloaded_wrench_bias"), float)
        if bias.shape != (6,) or not np.isfinite(bias).all():
            raise ValueError("force training requires six calibrated unloaded wrench bias values")
        if settings.get("wrench_frame_type") not in ("world", "base", "tool", "flange"):
            raise ValueError("force training requires an explicit robot wrench frame")
    return settings


def selected_settings(config, selection=None):
    spec = validate_config(config, measurement=False)
    cid = (selection or {}).get("candidate_id", spec["reference_candidate_id"])
    item = next((p for p in spec["candidates"] if p["candidate_id"] == cid), None)
    if item is None:
        raise ValueError("TCP candidate is outside the force catalog")
    base = {k: config["trajectory"][k] for k in SETTINGS}
    if set(item) != {"candidate_id", *FEATURES}:
        raise ValueError("force candidate may vary only horizontal displacement and lift/return duration")
    base.update({k: item[k] for k in FEATURES})
    return cid, base


def effective_settings(config, cid, pose, frame):
    _, settings = selected_settings(config, {"candidate_id": cid})
    table, _, _ = build_tcp_trajectory(settings, current_tcp_start_pose=pose, rehab_frame=frame)
    speed = min(config["motion"]["max_tcp_speed_m_s"], config["empty_load"]["max_tcp_speed_m_s"])
    accel = min(config["motion"]["max_tcp_acceleration_m_s2"], config["empty_load"]["max_tcp_acceleration_m_s2"])
    v = np.linalg.norm(table[[f"tcp_v{a}_base" for a in "xyz"]], axis=1).max()
    a = np.linalg.norm(table[[f"tcp_a{x}_base" for x in "xyz"]], axis=1).max()
    margin = config["empty_load"]["retiming_margin"]
    scale = max(1., margin*v/speed, np.sqrt(margin*a/accel))
    for key in ("lift_duration_s", "return_duration_s"):
        settings[key] *= scale
    return settings


def check_task(config, settings, pose):
    task = config["force_training"]
    lo, hi = task["horizontal_displacement_bounds_m"]
    if not lo <= settings["horizontal_displacement_m"] <= hi:
        raise ValueError("TCP candidate changes the required horizontal task")
    lo, hi = task["branch_duration_bounds_s"]
    if any(not lo <= settings[k] <= hi for k in ("lift_duration_s", "return_duration_s")):
        raise ValueError("TCP branch duration outside task bounds")
    if settings["target_peak_tcp_z_base_m"] - pose[2] < task["min_vertical_excursion_m"]:
        raise ValueError("insufficient vertical task excursion; low force alone is not task completion")
    duration = sum(settings[k] for k in ("start_hold_s", "lift_duration_s", "peak_hold_s", "return_duration_s"))
    if duration > task["max_cycle_duration_s"] + 1e-9:
        raise ValueError("TCP candidate exceeds task duration; cannot slow indefinitely")
    if "fixed_cycle_duration_s" in task and not np.isclose(duration, task["fixed_cycle_duration_s"], atol=1e-9, rtol=0):
        raise ValueError("retimed TCP candidate differs from fixed comparison cycle duration")


def freeze_binding(directory, config, plan, *, measurement=True, persist=True):
    """Freeze actual retimed candidate features and task membership."""
    task = validate_config(config, measurement=measurement)
    anchor = load_start_anchor(directory / "anchor.json")
    frame = load_rehab_frame_config(directory / "prepared_config.json")
    points = []
    motion = dict(config["motion"])
    for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        motion[key] = min(motion[key], config["empty_load"][key])
    for item in task["candidates"]:
        settings = effective_settings(config, item["candidate_id"], anchor.tcp_pose_base, frame)
        check_task(config, settings, anchor.tcp_pose_base)
        table, _, _ = build_tcp_trajectory(settings, current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
        reasons, _, _ = audit_tcp_trajectory(table, ExperimentSafetyConfig(**motion), frame=frame, anchor=anchor)
        points.append({"candidate_id": item["candidate_id"], "settings": settings, "blockers": list(reasons)})
    matches = [p for p in points if all(np.isclose(plan["tcp_settings"][k], p["settings"][k], atol=1e-9, rtol=0)
                                      for k in p["settings"])]
    if len(matches) != 1 or matches[0]["blockers"]:
        raise ValueError("prepared force TCP is not a unique executable catalog candidate")
    if len({tuple(p["settings"][k] for k in FEATURES) for p in points}) != len(points):
        raise ValueError("retiming collapsed force candidates to duplicate trajectories")
    result = {"method": METHOD, "candidate_id": matches[0]["candidate_id"], "candidates": points}
    if persist:
        atomic_json(directory / "tcp_force_binding.json", result)
    return result


def require_binding(directory, config, state):
    saved = read_json(directory / "tcp_force_binding.json")
    if saved != state.get("force_training_binding"):
        raise ValueError("force candidate binding changed")
    expected = freeze_binding(directory, config, state["tcp_plan"], persist=False)
    if saved != expected:
        raise ValueError("force catalog changed after preparation")
    return saved


def measure(directory):
    """Audit actual task execution before computing directly measured load."""
    import pandas as pd
    config = read_json(directory / "prepared_config.json")
    spec = validate_config(config)
    state = read_json(directory / "round_status.json")
    binding = require_binding(directory, config, state)
    audit = audit_robot_execution(directory)
    if audit["frame_type"] != spec["wrench_frame_type"]:
        raise ValueError("force training wrench frame differs from calibration")
    anchor = audit["anchor_pose"]
    check_task(config, state["tcp_plan"]["tcp_settings"], anchor)
    wrench = pd.read_csv(directory / "robot/robot_wrench.csv")
    clock = wrench[["query_start_s", "query_end_s"]].mean(axis=1).to_numpy(float)
    raw = wrench[["fx", "fy", "fz", "tx", "ty", "tz"]].to_numpy(float)
    corrected = raw - np.asarray(spec["unloaded_wrench_bias"])
    force, torque = np.linalg.norm(corrected[:, :3], axis=1), np.linalg.norm(corrected[:, 3:], axis=1)
    settings = state["tcp_plan"]["tcp_settings"]
    origin = audit["schedule_origin_host_time_s"]
    if audit["schedule_origin_source"] != "recorded_schedule_origin":
        raise ValueError("force training requires the recorded schedule origin")
    valid = (wrench.valid.astype(str).str.lower().eq("true").to_numpy()
             & np.isfinite(raw).all(axis=1)
             & (wrench.query_end_s >= wrench.query_start_s).to_numpy()
             & (wrench.query_end_s-wrench.query_start_s <= config["acquisition"]["budgets"]["query_s"]).to_numpy())
    intervals = ((settings["start_hold_s"], settings["start_hold_s"]+settings["lift_duration_s"]),
        (settings["start_hold_s"]+settings["lift_duration_s"]+settings["peak_hold_s"], state["tcp_plan"]["duration_s"]))
    from measurement_validation.wrench_endpoints import load_window
    components = []
    values = np.column_stack((force, torque))
    for begin, end in intervals:
        try:
            branch = load_window(clock, values, valid, origin+begin, origin+end,
                                 config["motion"]["max_wrench_age_s"])
        except ValueError as exc:
            raise ValueError("invalid moving wrench endpoint: " + str(exc)) from exc
        components.append(branch["rms"])
    whole = load_window(clock, values, valid, origin, origin+state["tcp_plan"]["duration_s"],
                        config["motion"]["max_wrench_age_s"])
    peak = np.asarray(whole["peaks"])
    limits = np.array([config["motion"]["max_force_n"], config["motion"]["max_torque_nm"]])
    if np.any(peak > limits):
        raise ValueError("corrected load exceeds physical force/torque limits")
    rms = np.asarray(components).mean(axis=0)
    values = np.r_[float(np.mean(rms/limits)), peak/limits]
    return {"candidate_id": binding["candidate_id"], "values": values.tolist(),
        "force_rms_n": float(rms[0]), "torque_rms_nm": float(rms[1]), "peaks": peak.tolist(),
        "frame_type": audit["frame_type"], "anchor_pose": anchor,
        "evidence_hashes": audit["evidence_hashes"], "dummy_leg_data_used": False,
        "endpoint_method": whole["method"], "study_identity": study_identity(config),
        "robot_task_complete": True, "patient_joint_angles_or_parameters_estimated": False}


def history_directories(directory):
    result, seen = [], set()
    while directory is not None:
        directory = Path(directory).resolve()
        if directory in seen:
            raise ValueError("cyclic force training history")
        seen.add(directory)
        result.append(directory)
        directory = read_json(directory / "round_status.json").get("parent_session")
    return result[::-1]


def train(directory):
    directory = Path(directory).resolve()
    config = read_json(directory / "prepared_config.json")
    spec = validate_config(config)
    state = read_json(directory / "round_status.json")
    if state.get("empty_load") or state.get("learning_mode") != "tcp_force":
        raise ValueError("force training requires a completed loaded robot task")
    if state.get("trial_purpose") == "evaluation":
        raise ValueError("evaluation recordings do not enter adaptive GP history")
    budget = attempt_snapshot(directory, config, state)
    if budget["study_identity"]["strict"] and not budget["complete"]:
        raise ValueError("strict force study requires complete pre-execution attempt accounting")
    binding = require_binding(directory, config, state)
    history, hashes, pose = [], {}, None
    for source in history_directories(directory):
        old = read_json(source / "prepared_config.json")
        if study_identity(old) != study_identity(config):
            raise ValueError("force setup or mechanical condition changed between trials")
        if any(old[k] != config[k] for k in ("force_training", "robot", "motion", "empty_load", "trajectory", "frame")):
            raise ValueError("force task/configuration changed between trials")
        if source != directory:
            verify_training(source)
        observation = measure(source)
        if pose is None:
            pose = np.asarray(observation["anchor_pose"])
        current = np.asarray(observation["anchor_pose"])
        if (np.linalg.norm(current[:3]-pose[:3]) > config["motion"]["max_start_anchor_position_error_m"]
            or np.linalg.norm(np.arctan2(np.sin(current[3:]-pose[3:]), np.cos(current[3:]-pose[3:]))) > config["motion"]["max_start_anchor_orientation_error_rad"]):
            raise ValueError("force comparison requires the same robot task anchor and tool orientation")
        history.append({"directory": str(source), **observation})
        hashes.update({str(source / name): digest for name, digest in observation["evidence_hashes"].items()})
        hashes[str(source / "tcp_force_binding.json")] = file_hash(source / "tcp_force_binding.json")
    if history[0]["candidate_id"] != spec["reference_candidate_id"]:
        raise ValueError("force reference must be the first acquired trial")
    points = binding["candidates"]
    ids = [p["candidate_id"] for p in points]
    x = np.array([[p["settings"][k] for k in FEATURES] for p in points])/np.asarray(spec["feature_scales"])
    index = np.array([ids.index(row["candidate_id"]) for row in history])
    y = np.array([row["values"] for row in history])
    prior = y[0]
    means, stds = [], []
    for column in range(3):
        residual = y[:, column]-prior[column]
        gp = MaternGP(spec["gp_length_scale"], max(spec["gp_signal_floor"], float(np.sqrt(np.mean(residual**2)))))
        gp.fit(x[index], residual, np.full(len(index), spec["gp_noise_std"]))
        mean, std = gp.predict(x)
        means.append(mean+prior[column]); stds.append(std)
    mean, std = np.asarray(means).T, np.asarray(stds).T
    _, feasible = constraint_probability(mean[:, 1:], std[:, 1:], [1., 1.])
    used = {row["candidate_id"] for row in history}
    available = [i for i, point in enumerate(points) if point["candidate_id"] not in used and not point["blockers"]]
    eligible = [i for i in available if feasible[i] >= spec["feasibility_probability"]]
    charged = budget["adaptive_attempts"] if budget["complete"] else len(history)
    stop = ("TOTAL_BUDGET_REACHED" if charged >= spec["total_budget"] else
            "DOMAIN_EXHAUSTED" if not available else "NO_PREDICTED_FEASIBLE_CANDIDATE" if not eligible else None)
    incumbent = min(history, key=lambda row: row["values"][0])
    if not stop and incumbent["values"][0]-float(np.min(mean[available, 0]-1.96*std[available, 0])) <= spec["minimum_gain"]:
        stop = "NO_EXPECTED_IMPROVEMENT"
    action = {"candidate_id": None, "role": "STOP", "stop_reason": stop}
    if stop is None:
        scores = expected_improvement(mean[eligible, 0], std[eligible, 0], incumbent["values"][0])*feasible[eligible]
        chosen = eligible[int(np.argmax(scores))]
        action = {"candidate_id": ids[chosen], "role": "TCP_SEARCH", "stop_reason": None}
    prediction_evaluation = {"status": "NO_PRETRIAL_PREDICTION"}
    if (directory / "pretrial_prediction.json").is_file():
        committed = read_json(directory / "pretrial_prediction.json")
        started = read_json(directory / "robot/metadata.json")["execution_result"]["started_host_time_s"]
        if (committed != state.get("pretrial_prediction") or committed.get("method") != METHOD
            or committed["candidate_id"] != binding["candidate_id"] or committed["committed_host_time_s"] >= started):
            raise ValueError("force prediction changed or was committed after motion")
        prediction_evaluation = {"status": "COMMITTED_BEFORE_MOTION",
            "absolute_error": np.abs(np.asarray(committed["mean"])-y[-1]).tolist()}
    result = {"method": METHOD, "feature_names": list(FEATURES), "output_names": list(OUTPUTS),
        "action": action, "recommended_observed_candidate_id": incumbent["candidate_id"],
        "candidates": ids, "predictions": {"mean": mean.tolist(), "std": std.tolist()},
        "history": history, "input_sha256": hashes, "code_sha256": code_hashes(),
        "attempt_accounting": budget, "trial_purpose": "adaptive",
        "prediction_evaluation": prediction_evaluation, "dummy_leg_data_used": False,
        "six_parameter_identification": False, "hardware_benefit_demonstrated": False,
        "limitations": "finite task catalog; GP scores/noise are engineering assumptions, not calibrated clinical safety; posture changes and repeat dependence are not modeled"}
    atomic_json(directory / "tcp_force_training.json", result)
    return result


def evaluate(directory):
    """Measure an explicitly separate repeat; never update/search the GP."""
    directory = Path(directory).resolve()
    config = read_json(directory / "prepared_config.json")
    state = read_json(directory / "round_status.json")
    if state.get("trial_purpose") != "evaluation" or not state.get("evaluation_source"):
        raise ValueError("explicit evaluation source and purpose required")
    observation = measure(directory)
    result = dict(method="TCP_FORCE_EVALUATION_ONLY", trial_purpose="evaluation", action=None,
                  observation=observation, source_session=state["evaluation_source"],
                  attempt_accounting=attempt_snapshot(directory, config, state),
                  input_sha256={str(directory / name): digest for name, digest in observation["evidence_hashes"].items()},
                  code_sha256=code_hashes(), enters_adaptive_history=False)
    result["input_sha256"][str(directory / "tcp_force_binding.json")] = file_hash(directory / "tcp_force_binding.json")
    atomic_json(directory / "tcp_force_evaluation.json", result)
    return result


def verify_evaluation(directory):
    directory = Path(directory).resolve()
    state = read_json(directory / "round_status.json")
    result = read_json(directory / "tcp_force_evaluation.json")
    if state.get("training_status") != "COMPLETE" or state.get("force_evaluation") != result:
        raise ValueError("force evaluation is incomplete or changed")
    if result.get("method") != "TCP_FORCE_EVALUATION_ONLY" or result.get("code_sha256") != code_hashes():
        raise ValueError("force evaluation implementation changed")
    for path, digest in result["input_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("force evaluation input changed")
    return result


def verify_training(directory):
    directory = Path(directory).resolve()
    state = read_json(directory / "round_status.json")
    result = read_json(directory / "tcp_force_training.json")
    if state.get("training_status") != "COMPLETE" or result != state.get("force_training"):
        raise ValueError("force training is incomplete or changed")
    if result.get("method") != METHOD or result.get("code_sha256") != code_hashes():
        raise ValueError("force training implementation changed; retrain")
    for path, digest in result["input_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("force training input changed; retrain")
    return result


def selected_candidate(directory):
    result = verify_training(directory)
    action = result["action"]
    if action["candidate_id"] is None:
        raise ValueError("force algorithm stopped: " + str(action["stop_reason"]))
    return {"candidate_id": action["candidate_id"]}


def commit_prediction(parent, directory, binding):
    result = verify_training(parent)
    i = result["candidates"].index(binding["candidate_id"])
    value = {"method": METHOD, "candidate_id": binding["candidate_id"],
        "output_names": list(OUTPUTS), "mean": result["predictions"]["mean"][i],
        "std": result["predictions"]["std"][i], "committed_host_time_s": time.perf_counter(),
        "source_training_sha256": file_hash(Path(parent) / "tcp_force_training.json")}
    atomic_json(directory / "pretrial_prediction.json", value)
    return value
