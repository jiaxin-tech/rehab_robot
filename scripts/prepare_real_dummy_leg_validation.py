"""Write an offline, blocked preparation record; never import robot acquisition."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from safety.experiment_safety import load_experiment_safety_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/real_dummy_leg_validation_v1"


def main() -> None:
    safety = load_experiment_safety_config(ROOT / "config/experiment_safety.json")
    release = json.loads((ROOT / "reference_release/reference_release_manifest.json").read_text())
    reasons = list(safety.execution_block_reasons())
    if release["approved_for_first_robot_trial"] is not True:
        reasons.append("reference_release_not_robot_approved")
    if not reasons:
        raise RuntimeError("Preparation-only script requires a blocked gate; use reviewed experiment workflow")
    # Refuse to overwrite any existing experiment evidence.
    OUT.mkdir(parents=True, exist_ok=False)
    for stage in ("stage_a_static", "stage_b_repeatability", "stage_c_trajectory_sensitivity"):
        (OUT / stage).mkdir()
        (OUT / stage / "NOT_EXECUTED.md").write_text(
            "No trials or raw episodes were collected. No figures generated.\n", encoding="utf-8")
    sources = ["config/experiment_safety.json", "config/rehab_frame_config.json",
               "config/formal_experiment_manifest.json", "reference_release/reference_release_manifest.json",
               "diagnostics/state_wrench_timing_comparison_20260814T093551709145Z.md",
               "diagnostics/wrench_hardware_validation_20260813T110502Z.md",
               "measurement_validation/analysis.py"]
    manifest = {
        "record_type": "PREPARATION_ONLY_NOT_FROZEN_EXECUTION_PROTOCOL",
        "stop_reason": "MOTION_STAGE_NOT_EXECUTED_DUE_TO_EXISTING_SAFETY_GATE",
        "safety_block_reasons": reasons, "live_preflight_executed": False,
        "robot_connection_attempted": False, "new_trial_count": 0,
        "dummy_leg_id": None, "cuff_setup_id": None, "robot_config_id": None,
        "rom_profile_id": None, "physical_installation_verified": False,
        "topology": "robot flange -> existing connector -> cuff / rigid plate / strap assembly -> dummy-leg shank",
        "preferred_research_family": "KEY_POSTURE_TIMING",
        "physical_experiment_family_frozen": False,
        "research_reference_parameters": {"alpha_flex": 0.0, "alpha_extend": 0.0, "time_share_shift": 0.0},
        "research_reference_source": "lower_limb_sim/myoleg_benchmark/experiment.py:Domain",
        "robot_reference_release": release,
        "research_reference_equals_robot_release_verified": False,
        "contrast_a": None, "contrast_b": None, "repeats": None,
        "static_duration_s": None, "trial_order": [], "actual_order": [],
        "required_before_freeze": ["physical setup IDs and secure installation", "reviewed ROM and exact family-specific trajectories",
                                   "explicit --repeats N or frozen protocol count", "static duration and acquisition settings",
                                   "predeclared balanced order and trajectory hashes", "existing safety and acquisition readiness"],
        "sources_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "historical_sdk_blocking_observed": True, "current_session_sdk_blocking_observed": False,
        "analysis_compatibility": "Existing beta-only grouping must be extended before KEY_POSTURE_TIMING analysis; never alias alpha to beta",
    }
    (OUT / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    schemas = {
        "trial_metadata.csv": "trial_id,stage,planned_order,actual_order,dummy_leg_id,cuff_setup_id,robot_config_id,rom_profile_id,trajectory_family,candidate_id,alpha_flex,alpha_extend,beta_flex,beta_extend,time_share_shift,duration_s,rom_json,trajectory_sha256,episode_path,executed,valid,invalid,reason",
        "sample_quality.csv": "trial_id,sample_id,timestamp,sample_valid,invalid,reason,missing,stale,stale_age_s,query_start_time,query_end_time,latency_s,sdk_code,last_successful_wrench_timestamp,last_robot_state_timestamp",
        "repeatability_summary.csv": "dummy_leg_id,condition_id,branch,feature,n_planned,n_attempted,n_valid,mean,standard_deviation,cv,cv_reason,pairwise_correlation,phase_aligned_rms_difference,peak_variation,timing_variability_s,baseline_drift,missing_fraction,invalid_fraction,status",
        "trajectory_sensitivity_summary.csv": "dummy_leg_id,reference_id,contrast_id,branch,feature,n_reference,n_contrast,effect_magnitude,within_repeat_sd,effect_noise_ratio,ratio_reason,curve_distance,consistency,status",
        "sdk_event_summary.csv": "trial_id,event_id,query_start_time,query_end_time,latency_s,sdk_code,reason,last_successful_wrench_timestamp,last_robot_state_timestamp,stale_age_s,source_path",
    }
    for name, header in schemas.items():
        with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(header.split(","))
    status = {
        "REAL_DUMMY_LEG_TRAJECTORY_VALIDATION_V1": "COMPLETE_WITH_LIMITATIONS",
        "HANDHELD_FORCE_GAUGE_USED": False, "HUMAN_SUBJECT_USED": False,
        "DUMMY_LEG_USED": False, "STATIC_BASELINE_EXECUTED": False,
        "STATIC_WRENCH_STABILITY": "INSUFFICIENT", "REFERENCE_REPEATABILITY_EXECUTED": False,
        "SAME_TRAJECTORY_REPEATABILITY": "NOT_EXECUTED", "TRAJECTORY_SENSITIVITY_EXECUTED": False,
        "TRAJECTORY_SENSITIVITY": "NOT_EXECUTED", "ABSOLUTE_FORCE_CALIBRATION_VALIDATED": False,
        "TASK_FORCE_DIRECTION_VALIDATED": False, "WRENCH_SDK_BLOCKING_OBSERVED": True,
        "DECISION_VALUE_EXPERIMENT_READY": False, "PERSONALIZATION_EXECUTED": False,
        "BO_EXECUTED": False, "ROBOT_CONTROL_CODE_MODIFIED": False, "SAFETY_THRESHOLDS_MODIFIED": False,
    }
    (OUT / "final_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(manifest["stop_reason"])
    print(f"Offline gate reasons: {len(reasons)}; no hardware connection; output: {OUT}")


if __name__ == "__main__":
    main()
