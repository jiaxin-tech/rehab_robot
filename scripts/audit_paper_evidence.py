"""Create a read-only, reproducible inventory for the paper's evidence boundary."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from collection.twincat_leg import atomic_json
from config.bundle import ROOT, configuration
from control.independent_tcp_trajectory import TRAJECTORY_ID, build_tcp_trajectory
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from control.start_anchored_relative_trajectory import build_start_anchored_relative_trajectory
from control.execution_preflight import audit_execution_trajectory_content
from safety.experiment_safety import ExperimentSafetyConfig
from measurement_validation.robot_observation import SOURCE, ANALYSIS_FIELDS, validate_robot_observation
from scripts import tcp_force_training as force_training
from config.bundle import resolve_config_path

SOURCES = ("config/experiment.json", "control/independent_tcp_trajectory.py",
    "control/robot_trajectory_executor.py", "control/execution_preflight.py",
    "collection/real_robot_acquisition.py", "collection/twincat_leg.py",
    "hardware/windows/rokae_xcore.py", "scripts/lab_session.py",
    "scripts/train_robot_leg_round.py", "experiment_algorithm/config.py",
    "experiment_algorithm/policy.py", "experiment_algorithm/surrogate.py",
    "experiment_algorithm/coupled_model.py", "experiment_algorithm/coordinate_consistent_model.py",
    "experiment_algorithm/coordinates.py", "control/start_anchored_relative_trajectory.py",
    "lower_limb_sim/kinematics.py", "lower_limb_sim/config.py",
    "lower_limb_sim/jacobian.py", "lower_limb_sim/observation_model.py", "measurement_validation/robot_observation.py",
    "scripts/original_training_adapter.py", "control/prepared_trajectory.py", "measurement_validation/robot_execution.py", "measurement_validation/passive_response.py",
    "measurement_validation/robot_leg_round.py", "scripts/audit_paper_evidence.py",
    "tests/test_original_training_adapter.py", "scripts/tcp_force_training.py", "tests/test_tcp_force_training.py")

SOURCES += ("measurement_validation/wrench_endpoints.py", "measurement_validation/force_trial_ledger.py",
    "scripts/manual_leg_check.py", "config/paper_personalized_load.template.json", "pytest-core.ini",
    "tests/test_force_trial_ledger.py", "tests/test_manual_leg_check.py", "tests/test_wrench_endpoints.py")


def historical_path(relative):
    """Materialize hash-checked selected historical bytes in a disposable cache."""
    manifest = read_json(ROOT / "data/lab_evidence_20261114_manifest.json")
    archive_path = ROOT / "data/lab_evidence_20261114.zip"
    if sha256(archive_path) != manifest["archive_sha256"]:
        raise ValueError("historical evidence archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(relative)
    if hashlib.sha256(payload).hexdigest() != manifest["files"][relative]["sha256"]:
        raise ValueError("historical evidence file changed: " + relative)
    target = ROOT / ".cache/paper_evidence" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(junit=None):
    sessions = []
    for path in sorted((ROOT / "results").rglob("round_status.json")):
        state = read_json(path)
        sessions.append({"session": path.parent.name, "status": state.get("status"),
            "mode": state.get("trajectory_mode", "historical"),
            "training_status": state.get("training_status"),
            "original_training_file_available": (path.parent / "original_training.json").is_file(),
            "status_sha256": sha256(path)})
    diagnostic_path = historical_path("results/rt_link_004/status.json")
    diagnostic = read_json(diagnostic_path)
    old = historical_path("results/empty_load_118/round_status.json").parent
    for relative in ("anchor.json", "robot/robot_state.csv", "robot/metadata.json", "preview/trajectory_preview.csv"):
        historical_path("results/empty_load_118/" + relative)
    robot = pd.read_csv(old / "robot/robot_state.csv")
    xyz = robot.loc[robot.valid.astype(str).str.lower().eq("true"), ["tcp_x", "tcp_y", "tcp_z"]].to_numpy(float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    preview = pd.read_csv(old / "preview/trajectory_preview.csv")
    config_path = ROOT / "config/experiment.json"
    config = configuration(config_path)
    anchor = replace(load_start_anchor(old / "anchor.json"), trajectory_id=TRAJECTORY_ID)
    if force_training.active(config):
        cid = config["force_training"]["reference_candidate_id"]
        settings = force_training.effective_settings(config, cid, anchor.tcp_pose_base, load_rehab_frame_config(config_path))
        table, _, metadata = build_tcp_trajectory(settings,
            current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=load_rehab_frame_config(config_path))
        metadata.update(learning_mode="tcp_force", candidate_id=cid,
            patient_geometry_required=False, patient_joint_angles_or_parameters_estimated=False)
    elif config["trajectory"].get("mode") == "original_candidate":
        point = next(p for p in config["training"]["candidates"] if p["candidate_id"] == config["training"]["reference_candidate_id"])
        table, _, metadata = build_start_anchored_relative_trajectory(resolve_config_path(ROOT / point["path"]),
            current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=load_rehab_frame_config(config_path))
        metadata = {"mode": "original_candidate", "candidate_id": point["candidate_id"],
            "converter": "build_start_anchored_relative_trajectory", "joint_feedback_used_for_planning": False,
            "candidate_source": point["path"],
            "candidate_sha256": sha256(resolve_config_path(ROOT / point["path"])),
            "actual_peak_tcp_z_base_m": float(table.tcp_z_base.max()), "duration_s": float(table.time_s.iloc[-1])}
    else:
        table, _, metadata = build_tcp_trajectory(config["trajectory"],
            current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=load_rehab_frame_config(config_path))
    anchor = replace(anchor, trajectory_id=str(table.trajectory_id.iloc[0]))
    motion = dict(config["motion"])
    for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        motion[key] = min(motion[key], config["empty_load"][key])
    reasons, _, _ = audit_execution_trajectory_content(table, ExperimentSafetyConfig(**motion),
        expected_trajectory_id=anchor.trajectory_id, frame=load_rehab_frame_config(config_path), anchor=anchor)
    target_z = config["trajectory"].get("target_peak_tcp_z_base_m")
    height_met = target_z is None or bool(np.isclose(table.tcp_z_base.max(), target_z, atol=1e-9, rtol=0))
    metadata.update(requested_peak_tcp_z_base_m=target_z, peak_z_requirement_met=height_met,
        execution_audit_reasons=list(reasons),
        execution_limits={key: motion[key] for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2")},
        catalog_size=len(config["force_training"]["candidates"]) if force_training.active(config) else len(config["training"]["candidates"]),
        missing_robot_analysis=[key for key in ANALYSIS_FIELDS if config["analysis"].get(key) is None],
        robot_observation_source=SOURCE, dummy_leg_training_data_used=False,
        missing_robot_calibration=[key for key, value in config.get("robot_observation", {}).items()
            if value is None or value is False])
    try:
        if force_training.active(config):
            force_training.validate_config(config)
            metadata.update(robot_observation_source=force_training.METHOD,
                missing_robot_analysis=[], missing_robot_calibration=[])
        else:
            validate_robot_observation(config)
        metadata["robot_observation_calibration_status"] = "CONFIGURED_NOT_HARDWARE_VALIDATED"
    except (ValueError, KeyError, TypeError) as exc:
        metadata["robot_observation_calibration_status"] = "INCOMPLETE"
        metadata["robot_observation_blocker"] = str(exc)
    if force_training.active(config):
        metadata.update(robot_observation_source=force_training.METHOD, missing_robot_analysis=[],
            missing_robot_calibration=[key for key in ("unloaded_wrench_bias", "compensation_verified")
                if config["force_training"].get(key) is None or config["force_training"].get(key) is False])
    verification = {"kind": "OFFLINE_TESTS_ONLY", "status": "NOT_PROVIDED"}
    if junit is not None:
        tree = ET.parse(junit).getroot()
        suites = list(tree.iter("testsuite"))
        verification.update(status="REPORT_PROVIDED", report=str(junit.resolve()),
            sha256=sha256(junit),
            **{key: sum(int(s.attrib.get(key, 0)) for s in suites)
               for key in ("tests", "failures", "errors", "skipped")})
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "scope": "Current source tree, local results and selected hash-checked historical archive; no new hardware or clinical experiment.",
        "historical_evidence_archive_sha256": sha256(ROOT / "data/lab_evidence_20261114.zip"),
        "sessions": sessions,
        "rt_link_004": {"recorded_status": diagnostic["status"],
            "reported_tracking": diagnostic["result"]["tracking_verification"],
            "status_sha256": sha256(diagnostic_path),
            "raw_robot_state_available": (diagnostic_path.parent / "robot/robot_state.csv").is_file(),
            "interpretation": "Recorded 1 mm diagnostic outcome, not a current full-trajectory learning experiment."},
        "empty_load_118": {"recorded_status": read_json(old / "round_status.json")["status"],
            "robot_metadata_git_commit": read_json(old / "robot/metadata.json").get("git_commit"),
            "planned_peak_z_m": float(preview.tcp_z_base.max()),
            "valid_robot_samples": len(xyz),
            "max_distance_from_first_valid_tcp_m": float(np.linalg.norm(xyz-xyz[0], axis=1).max()),
            "interpretation": "Historical archived recording only; does not negate operator reports of later unarchived motion."},
        "current_plan_offline_example": {**metadata,
            "source_anchor": "results/empty_load_118/anchor.json",
            "peak_speed_m_s": float(np.linalg.norm(table[[f"tcp_v{a}_base" for a in "xyz"]], axis=1).max()),
            "peak_acceleration_m_s2": float(np.linalg.norm(table[[f"tcp_a{a}_base" for a in "xyz"]], axis=1).max()),
            "sample_count": len(table), "hardware_accessed": False},
        "verification": verification}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    result = inventory(args.junit)
    atomic_json(args.output.resolve(), result)
    print(f"Wrote {args.output}; {len(result['sessions'])} archived session states inspected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
