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
    "tests/test_force_trial_ledger.py", "tests/test_manual_leg_check.py", "tests/test_wrench_endpoints.py",
    "scripts/twincat_file_evidence.py", "tests/test_twincat_file_evidence.py",
    "scripts/record_rehab_trajectory.py", "scripts/preview_recorded_trajectory.py",
    "scripts/review_empty_recording.py", "tests/test_record_rehab_trajectory.py",
    "tests/test_preview_recorded_trajectory.py", "tests/test_review_empty_recording.py")

SOURCES += ("control/taught_tcp_trajectory.py", "scripts/design_taught_trajectory.py",
    "tests/test_taught_tcp_trajectory.py", "tests/test_design_taught_trajectory.py",
    "config/taught_tcp_empty.template.json", "scripts/capture_start_anchor.py",
    "scripts/run_rehab_experiment.py", "scripts/preview_rehab_trajectory.py", "tests/test_mainline.py")
SOURCES += ("collection/episode_logger.py", "tests/test_command_prelogging.py")

TAUGHT_DESIGN = "results/manual_lab/20261115_taught_design_v1"
SOURCES += tuple(TAUGHT_DESIGN + "/" + name for name in ("geometry.json", "settings.json", "summary.json"))
SOURCES += ("control/taught_tcp_timing.py", "scripts/preview_taught_timing.py", "tests/test_taught_tcp_timing.py")
TAUGHT_70S = "results/manual_lab/20261115_taught_70s_v1"
SOURCES += ("config/taught_tcp_70s.review.json",) + tuple(TAUGHT_70S + "/" + name
    for name in ("geometry.json", "settings.json", "summary.json"))
APPROVED_70S = "results/manual_lab/20261115_taught_70s_approved_v1"
SOURCES += ("config/taught_tcp_70s.empty.json",) + tuple(APPROVED_70S + "/" + name
    for name in ("geometry.json", "settings.json", "summary.json"))
CONTINUOUS_90S = "results/manual_lab/20261115_continuous_90s_v1"
SOURCES += ("control/continuous_taught_trajectory.py", "control/continuous_taught_timing.py",
    "scripts/design_continuous_taught_trajectory.py", "tests/test_continuous_taught_trajectory.py",
    "tests/test_design_continuous_taught_trajectory.py", "config/continuous_taught_90s.empty.json")
SOURCES += tuple(CONTINUOUS_90S + "/" + name for name in ("geometry.json", "settings.json", "summary.json"))
LOADED_DIAGNOSTIC = "results/manual_lab/20261122_loaded_diagnostic_v1"
RETURNED_90S = "results/manual_lab/20261122_143700/continuous_empty_01"
SOURCES += ("measurement_validation/loaded_trial.py", "measurement_validation/loaded_diagnostic.py",
    "scripts/design_loaded_diagnostic.py", "config/continuous_taught_90s.loaded_diagnostic.json",
    "tests/test_loaded_trial_protocol.py", "tests/test_loaded_diagnostic_session.py",
    "tests/test_loaded_wrench_freshness.py", "tests/test_design_loaded_diagnostic.py",
    "tests/test_passive_leg_snapshot.py", "docs/LOADED_DIAGNOSTIC_WORKFLOW.md",
    "docs/templates/RIGID_SHANK_SETUP_RECORD.md", "docs/lab/EMPTY_90S_REVIEW_2026-11-22.md",
    LOADED_DIAGNOSTIC + "/summary.json", RETURNED_90S + "/review_20261122/review.json")


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
    taught_summary = read_json(ROOT / TAUGHT_DESIGN / "summary.json")
    for name, expected in taught_summary["output_files_sha256"].items():
        if Path(name).name != name or sha256(ROOT / TAUGHT_DESIGN / name) != expected:
            raise ValueError("taught design artifact changed: " + name)
    for name, expected in taught_summary["processing"]["processing_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("taught design processing source changed: " + name)
    timing_summary = read_json(ROOT / TAUGHT_70S / "summary.json")
    for name, expected in timing_summary["output_files_sha256"].items():
        if Path(name).name != name or sha256(ROOT / TAUGHT_70S / name) != expected:
            raise ValueError("70s timing artifact changed: " + name)
    for name, expected in timing_summary["processing_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("70s timing source changed: " + name)
    approved_summary = read_json(ROOT / APPROVED_70S / "summary.json")
    if sha256(ROOT / "config/taught_tcp_70s.empty.json") != approved_summary["source_config_sha256"]:
        raise ValueError("approved 70s execution configuration changed")
    for name, expected in approved_summary["output_files_sha256"].items():
        if Path(name).name != name or sha256(ROOT / APPROVED_70S / name) != expected:
            raise ValueError("approved 70s timing artifact changed: " + name)
    for name, expected in approved_summary["processing_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("approved 70s timing source changed: " + name)
    continuous_summary = read_json(ROOT / CONTINUOUS_90S / "summary.json")
    if sha256(ROOT / "config/continuous_taught_90s.empty.json") != continuous_summary["output_config_sha256"]:
        raise ValueError("continuous 90s execution configuration changed")
    for name, expected in continuous_summary["output_files_sha256"].items():
        if Path(name).name != name or sha256(ROOT / CONTINUOUS_90S / name) != expected:
            raise ValueError("continuous 90s artifact changed: " + name)
    for name, expected in continuous_summary["processing_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("continuous 90s processing source changed: " + name)
    returned_70s = ROOT / "results/manual_lab/20261115_161440/taught_empty_01"
    from measurement_validation.robot_execution import audit_robot_execution, evidence_hashes
    try:
        returned_audit = dict(passed=True, result=audit_robot_execution(returned_70s, allow_empty=True))
    except ValueError as exc:
        returned_audit = dict(passed=False, error=str(exc))
    returned_90s = ROOT / RETURNED_90S
    try:
        returned_90s_audit = dict(passed=True, result=audit_robot_execution(returned_90s, allow_empty=True))
    except (ValueError, KeyError, OSError) as exc:
        returned_90s_audit = dict(passed=False, reason=str(exc))
    loaded_summary = read_json(ROOT / LOADED_DIAGNOSTIC / "summary.json")
    if sha256(ROOT / loaded_summary["output_config"]) != loaded_summary["output_config_sha256"]:
        raise ValueError("loaded diagnostic template changed")
    for name, expected in loaded_summary["output_files_sha256"].items():
        if Path(name).name != name or sha256(ROOT / LOADED_DIAGNOSTIC / name) != expected:
            raise ValueError("loaded diagnostic preview artifact changed: " + name)
    for name, expected in loaded_summary["source_code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("loaded diagnostic preview source changed: " + name)
    from measurement_validation.loaded_trial import validate_config as validate_loaded
    validate_loaded(configuration(ROOT / loaded_summary["output_config"]), require_review=False)
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
        "taught_plan_offline_design": {
            "source_summary": TAUGHT_DESIGN + "/summary.json",
            "configuration": "config/taught_tcp_empty.template.json",
            "status": taught_summary["status"], "settings": taught_summary["settings"],
            "timing": {k: v for k, v in taught_summary["timing"].items() if k != "iterations"},
            "output_hashes_verified": True, "processing_source_hashes_verified": True,
            "hardware_accessed": False, "new_hardware_experiment": False,
            "ready_for_replay": False, "loaded_learning_integrated": False},
        "requested_70s_timing": {
            "source_summary": TAUGHT_70S + "/summary.json", "status": timing_summary["status"],
            "execution_review": timing_summary["execution_review"],
            "shape_changed": False, "hardware_accessed": False, "ready_for_replay": False},
        "approved_70s_timing": {
            "source_summary": APPROVED_70S + "/summary.json",
            "execution_configuration": "config/taught_tcp_70s.empty.json",
            "status": approved_summary["status"], "execution_review": approved_summary["execution_review"],
            "scope": "User accepted 0.13 m/s and 0.45 m/s2 for this empty-load configuration only.",
            "shape_changed": False, "hardware_accessed": False, "ready_for_replay": False},
        "returned_70s_execution": {
            "session": "results/manual_lab/20261115_161440/taught_empty_01",
            "status": read_json(returned_70s / "round_status.json")["status"],
            "quality_audit": returned_audit, "evidence_sha256": evidence_hashes(returned_70s),
            "interpretation": "Physical 70s run; successful motion status does not establish complete wrench data or normal wrench-process cleanup."},
        "continuous_90s_plan": {
            "source_summary": CONTINUOUS_90S + "/summary.json", "status": continuous_summary["status"],
            "execution_configuration": "config/continuous_taught_90s.empty.json",
            "execution_review": continuous_summary["execution_review"],
            "speed_verification": continuous_summary["speed_verification"],
            "geometry_processing": continuous_summary["geometry_processing"],
            "preview_anchor_source_sha256": continuous_summary["preview_anchor_source_sha256"],
            "hardware_accessed": False, "ready_for_replay": False, "loaded_learning_integrated": False},
        "returned_90s_execution": {"session": RETURNED_90S,
            "status": read_json(returned_90s / "round_status.json")["status"],
            "quality_audit": returned_90s_audit, "evidence_sha256": evidence_hashes(returned_90s),
            "review": RETURNED_90S + "/review_20261122/review.json",
            "interpretation": "Complete empty motion; force coverage failed. No loaded experiment proved."},
        "loaded_diagnostic_preview": {"source_summary": LOADED_DIAGNOSTIC + "/summary.json",
            "status": loaded_summary["status"], "configuration": loaded_summary["output_config"],
            "hardware_accessed": False, "ready_for_replay": False, "loaded_learning_integrated": False,
            "actual_setup": "Fixed hip base, freely rotating hip/knee, rigid TCP fixture near shank knee",
            "execution_blocker": "Current fixed-orientation trajectory is not a reviewed coupled rigid-leg pose path",
            "output_hashes_verified": True},
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
