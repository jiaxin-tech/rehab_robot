"""Offline comparison adapter to the six-parameter/residual-GP/Greedy algorithm.

No optimizer, endpoint, GP or repeated-trial model is implemented here.
The original candidate execution uses the existing FK/coordinate converter.
This geometry-dependent path is not the deployed robot-load method.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from collection.twincat_leg import atomic_json
from config.bundle import resolve_config_path
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from measurement_validation.robot_execution import audit_robot_execution, evidence_hashes, read_json
from measurement_validation.robot_leg_round import endpoints
from measurement_validation.robot_observation import SOURCE, robot_measurement, validate_robot_observation

METHOD = "FROZEN_RESIDUAL_GP_MEAN_GREEDY"
ROOT = Path(__file__).resolve().parents[1]
CORE = ("experiment_algorithm/config.py", "experiment_algorithm/surrogate.py",
        "experiment_algorithm/policy.py", "experiment_algorithm/coupled_model.py",
        "experiment_algorithm/coordinate_consistent_model.py", "experiment_algorithm/coordinates.py")
OBSERVATION = ("measurement_validation/robot_observation.py", "lower_limb_sim/kinematics.py",
               "lower_limb_sim/jacobian.py", "lower_limb_sim/observation_model.py")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_hashes():
    return {name: file_hash(ROOT / name) for name in CORE}


def catalog(config):
    result = []
    for item in config["training"]["candidates"]:
        point = {key: item[key] for key in ("candidate_id", "beta_flex", "beta_extend", "share_shift")}
        if not isinstance(point["candidate_id"], str) or not point["candidate_id"].strip():
            raise ValueError("original algorithm candidate ID is missing")
        if not np.isfinite([point[k] for k in ("beta_flex", "beta_extend", "share_shift")]).all():
            raise ValueError("original algorithm candidate features must be finite")
        point["path"] = str(resolve_config_path(ROOT / item["path"]))
        result.append(point)
    if not result or len({p["candidate_id"] for p in result}) != len(result):
        raise ValueError("original algorithm requires a nonempty unique candidate catalog")
    return result


def resolve_binding(config, tcp_settings):
    raise ValueError("standalone seventh-order TCP path is not an original candidate; use the existing original_candidate FK converter, not a new mapping table")


def original_binding(config, plan):
    points = catalog(config)
    reference_id = config["training"]["reference_candidate_id"]
    if reference_id not in {p["candidate_id"] for p in points}:
        raise ValueError("original reference candidate is missing")
    if plan["candidate_id"] not in {p["candidate_id"] for p in points}:
        raise ValueError("selected original candidate is missing")
    selected = next(p for p in points if p["candidate_id"] == plan["candidate_id"])
    if file_hash(selected["path"]) != plan["reference_sha256"]:
        raise ValueError("frozen candidate does not match original catalog")
    return {"status": "BOUND", "method": METHOD, "candidate_id": plan["candidate_id"],
        "reference_candidate_id": reference_id, "source": "existing_candidate_fk_converter",
        "candidates": points, "candidate_sha256": {p["candidate_id"]: file_hash(p["path"]) for p in points}}


def freeze_binding(directory, config, plan):
    directory = Path(directory)
    try:
        binding = original_binding(config, plan) if plan.get("mode") == "original_candidate" else resolve_binding(config, plan["tcp_settings"])
        reference = pd.read_csv(next(p["path"] for p in binding["candidates"]
                                    if p["candidate_id"] == binding["candidate_id"]))
        from control.prepared_trajectory import build_prepared_trajectory
        actual, _, _ = build_prepared_trajectory(plan,
            current_tcp_start_pose=plan["anchor"]["tcp_pose_base"],
            rehab_frame=load_rehab_frame_config(directory / "prepared_config.json"))
        if (len(reference) != len(actual)
            or not np.allclose(reference.time_s, actual.time_s, atol=1e-9, rtol=0)):
            raise ValueError("original candidate timeline differs from executed TCP; supply a reviewed matching candidate, not automatic retiming")
        for point in binding["candidates"]:
            candidate = pd.read_csv(point["path"])
            columns = ["time_s", "q_hip_rad", "q_knee_rad", "dq_hip_rad_s", "dq_knee_rad_s",
                       "ddq_hip_rad_s2", "ddq_knee_rad_s2"]
            if (len(candidate) != len(reference) or not np.isfinite(candidate[columns].to_numpy(float)).all()
                or not np.array_equal(candidate.cycle_phase, reference.cycle_phase)
                or np.any(np.diff(candidate.time_s) <= 0)):
                raise ValueError("candidate catalog violates the unchanged original algorithm's kinematics/phase contract")
        validate_robot_observation(config)
    except (ValueError, KeyError, OSError) as exc:
        binding = {"status": "UNBOUND", "method": METHOD, "reason": str(exc)}
    atomic_json(directory / "original_training_binding.json", binding)
    return binding


def require_binding(directory, config, state):
    directory = Path(directory)
    binding = read_json(directory / "original_training_binding.json")
    if binding != state.get("original_training_binding") or binding.get("status") != "BOUND":
        raise ValueError("original algorithm training is blocked: " + str(binding.get("reason", "binding changed")))
    expected = (original_binding(config, state["tcp_plan"])
        if state["tcp_plan"].get("mode") == "original_candidate"
        else resolve_binding(config, state["tcp_plan"]["tcp_settings"]))
    if expected != binding:
        raise ValueError("original candidate catalog or reviewed binding changed")
    validate_robot_observation(config)
    return binding


def history_directories(directory):
    history, seen = [], set()
    while directory is not None:
        directory = Path(directory).resolve()
        if directory in seen:
            raise ValueError("cyclic original-algorithm history")
        seen.add(directory)
        history.append(directory)
        directory = read_json(directory / "round_status.json").get("parent_session")
    return history[::-1]


def _measurement(directory, config, state, binding):
    execution = audit_robot_execution(directory)
    candidate = next(item for item in binding["candidates"] if item["candidate_id"] == binding["candidate_id"])
    reference = pd.read_csv(candidate["path"])
    tcp = pd.read_csv(directory / "tcp_trajectory.csv")
    if len(reference) != len(tcp) or not np.allclose(reference.time_s, tcp.time_s, atol=1e-9, rtol=0):
        raise ValueError("original candidate and actual TCP timestamps differ")
    return robot_measurement(directory, config, reference, execution)


def train_original(directory, worker):
    directory = Path(directory).resolve()
    current_state = read_json(directory / "round_status.json")
    if current_state.get("empty_load") or current_state["status"] not in ("CAPTURE_COMPLETE", "ROUND_COMPLETE", "COMPLETE"):
        raise ValueError("original training requires a complete with-leg recording")
    config = read_json(directory / "prepared_config.json")
    current_binding = require_binding(directory, config, current_state)
    histories, hashes, baseline, used = [], {}, None, set()
    context = None
    for source in history_directories(directory):
        state = read_json(source / "round_status.json")
        saved = read_json(source / "prepared_config.json")
        if any(saved[key] != config[key] for key in ("robot", "frame", "robot_observation", "analysis", "training")):
            raise ValueError("original training history has a different setup or configuration")
        binding = require_binding(source, saved, state)
        if binding["candidates"] != current_binding["candidates"] or binding["candidate_sha256"] != current_binding["candidate_sha256"]:
            raise ValueError("original training history uses a different candidate catalog")
        cid = binding["candidate_id"]
        if cid in used:
            raise ValueError("DUPLICATE_CANDIDATE_NO_REPEAT_NOISE_MODEL: original algorithm does not accept repeated candidates as new learning observations")
        if not used and cid != binding["reference_candidate_id"]:
            raise ValueError("REFERENCE_MUST_BE_FIRST")
        used.add(cid)
        pose = np.asarray(state["tcp_plan"]["anchor"]["tcp_pose_base"], float)
        if context is None:
            context = pose
        orientation = np.arctan2(np.sin(pose[3:]-context[3:]), np.cos(pose[3:]-context[3:]))
        if (np.linalg.norm(pose[:3]-context[:3]) > config["motion"]["max_start_anchor_position_error_m"]
            or np.linalg.norm(orientation) > config["motion"]["max_start_anchor_orientation_error_rad"]):
            raise ValueError("original candidate/TCP binding requires the same declared task anchor")
        measured, metrics = _measurement(source, saved, state, binding)
        if baseline is None:
            baseline = metrics
        if source == directory:
            measured_path = source / "original_measured.csv"
            measured.to_csv(measured_path, index=False)
            atomic_json(source / "original_joint_metrics.json", metrics)
        else:
            verify_original_training(source)
            measured_path = source / "original_measured.csv"
            pd.testing.assert_frame_equal(pd.read_csv(measured_path), measured, check_exact=False, atol=1e-10, rtol=0)
        histories.append({"episode_id": str(source), "candidate_id": cid,
            "measured_path": str(measured_path), "endpoints": endpoints(metrics, baseline)})
        for name, digest in evidence_hashes(source).items():
            hashes[str(source / name)] = digest
        for name in ("original_training_binding.json", "original_measured.csv"):
            hashes[str(source / name)] = file_hash(source / name)
    for item in current_binding["candidates"]:
        hashes[item["path"]] = file_hash(item["path"])
    request = {"candidates": current_binding["candidates"],
        "reference_candidate_id": current_binding["reference_candidate_id"],
        "history": histories, "experiment_scope": "original_algorithm_robot_only_observation",
        "observation_source": SOURCE}
    atomic_json(directory / "original_training_request.json", request)
    result = worker(directory, request, config["training"]["timeout_s"])
    if (result.get("method") != METHOD or len(result.get("theta", [])) != 6
        or result.get("observation_source") != SOURCE):
        raise ValueError("worker did not return the original six-parameter residual-GP algorithm with robot-only observations")
    prediction_evaluation = {"status": "NOT_RECORDED_BEFORE_MOTION"}
    prediction_path = directory / "pretrial_prediction.json"
    if prediction_path.is_file():
        prediction = read_json(prediction_path)
        started = read_json(directory / "robot/metadata.json")["execution_result"]["started_host_time_s"]
        if (prediction != current_state.get("pretrial_prediction")
            or prediction["method"] != METHOD or prediction["candidate_id"] != current_binding["candidate_id"]
            or prediction["committed_host_time_s"] >= started):
            raise ValueError("original algorithm prediction was altered or recorded after motion")
        observed = histories[-1]["endpoints"]
        actual = np.array([observed[key] for key in ("E3", "E2", "peak_ratio")])
        prediction_evaluation = {"status": "COMMITTED_BEFORE_MOTION", "prediction": prediction,
            "absolute_error": np.abs(np.asarray(prediction["mean"])-actual).tolist()}
    wrapped = {"method": METHOD, "result": result, "history": histories,
        "binding": current_binding, "input_sha256": hashes, "core_sha256": core_hashes(),
        "observation_code_sha256": {name: file_hash(ROOT / name) for name in OBSERVATION},
        "observation_source": SOURCE, "dummy_leg_data_used": False,
        "adapter": "robot measured TCP and external wrench; existing IK and J.T @ F; reconstructed generalized torque",
        "robot_wrench_role": "identification observation, endpoint evaluation and execution protection",
        "prediction_evaluation": prediction_evaluation}
    atomic_json(directory / "original_training.json", wrapped)
    return wrapped


def verify_original_training(directory):
    directory = Path(directory).resolve()
    state = read_json(directory / "round_status.json")
    result = read_json(directory / "original_training.json")
    if state.get("training_status") != "COMPLETE" or state.get("original_training") != result:
        raise ValueError("original algorithm training is incomplete or changed")
    if result.get("method") != METHOD or result["core_sha256"] != core_hashes():
        raise ValueError("original algorithm changed; retrain")
    if result.get("observation_source") != SOURCE or result.get("dummy_leg_data_used") is not False:
        raise ValueError("historical dummy-leg-sensor training cannot drive formal robot-only recommendations")
    if result.get("observation_code_sha256") != {name: file_hash(ROOT / name) for name in OBSERVATION}:
        raise ValueError("robot observation implementation changed; retrain")
    for path, digest in result["input_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("original training input changed; retrain")
    return result


def selected_tcp_settings(directory):
    result = verify_original_training(directory)
    action = result["result"]["action"]
    if action["candidate_id"] is None or action["role"] == "STOP":
        raise ValueError("original algorithm stopped: " + str(action["stop_reason"]))
    if result["binding"].get("source") == "existing_candidate_fk_converter":
        if action["candidate_id"] not in {p["candidate_id"] for p in result["binding"]["candidates"]}:
            raise ValueError("original policy selected a candidate outside its catalog")
        return {"candidate_id": action["candidate_id"]}
    raise ValueError("training output did not use the existing original-candidate converter")


def commit_original_prediction(parent, directory, binding):
    result = verify_original_training(parent)
    model = result["result"]
    index = model["candidates"].index(binding["candidate_id"])
    prediction = {"method": METHOD, "candidate_id": binding["candidate_id"],
        "endpoint_family": ["E3", "E2", "peak_ratio"],
        "mean": model["predictions"]["mean"][index], "std": model["predictions"]["std"][index],
        "source_training_sha256": file_hash(Path(parent) / "original_training.json"),
        "committed_at": datetime.now(timezone.utc).isoformat(), "committed_host_time_s": time.perf_counter()}
    atomic_json(Path(directory) / "pretrial_prediction.json", prediction)
    return prediction
