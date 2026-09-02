"""Freeze a non-human, static wrench validation protocol without executing it.

This module writes protocol artifacts from pinned source/config evidence.  It
does not import robot, control or safety modules, connect to hardware, apply a
load, move a robot, or calculate any physical validation result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


STAGE_ID = "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1"
PROTOCOL_ID = "STATIC_WRENCH_VALIDATION_PROTOCOL"
FORMAL_STATUS = "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED"
PARENT_STATUS = "WRENCH_AND_TASK_DIRECTION_PARTIALLY_RESOLVED_REQUIRES_STATIC_VALIDATION"
FUTURE_DECISIONS = (
    "STATIC_WRENCH_FRAME_SIGN_VALIDATED",
    "STATIC_WRENCH_FRAME_SIGN_PARTIALLY_VALIDATED",
    "STATIC_WRENCH_FRAME_SIGN_NOT_VALIDATED",
)
REQUESTED_WRENCH_FRAME = "world"
VERIFIED_WRENCH_FRAME = "NONE_PHYSICALLY_VERIFIED"
TASK_DIRECTION_STATUS = "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
NEXT_DEPENDENCY = "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1"
ACTIVE_REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
PARENT_PROTOCOL_SHA256 = "e23c99636816b771a4233d35fe13cdeee46e57de84f5972aa764de5db775a52b"
PARENT_STATUS_SHA256 = "5b74d5744f7de7d60aa050b72e5ed14b2684a70d05fe7a22381bc068cb61f473"
FROZEN_PROTOCOL_SHA256 = "c88799b838f6304765acb643a706b1a6f1bbe02b1ee4f6c07ed9c486eab2f5c1"
REPETITIONS_PER_CELL = 5
HOST_QUERIES_PER_WINDOW = 100

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1"
PROTOCOL_PATH = OUTPUT / "STATIC_WRENCH_VALIDATION_PROTOCOL.json"


INPUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "PARENT_RESOLUTION_PROTOCOL",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/WRENCH_RESOLUTION_PROTOCOL.json",
        "markers": (PARENT_STATUS, "api_request_frame_is_not_physical_verification", "primary_endpoint_may_not_be_computed_or_validated"),
        "exact_sha256": PARENT_PROTOCOL_SHA256,
    },
    {
        "id": "PARENT_FINAL_STATUS",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/FINAL_WRENCH_TASK_DIRECTION_STATUS.json",
        "markers": (PARENT_STATUS, '"BASE_WRENCH_ROTATION_VERIFIED": false', '"WRENCH_FORCE_SIGN_VERIFIED": false'),
        "exact_sha256": PARENT_STATUS_SHA256,
    },
    {
        "id": "PARENT_TASK_DEFINITION",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/TASK_DIRECTION_FORMAL_DEFINITION.json",
        "markers": ("ACTUAL_STRAP_PULL_LINE_OF_ACTION", "p_robot_attach_B(t)", "p_limb_attach_B(t)", "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"),
    },
    {
        "id": "PARENT_STATIC_WRENCH_PLAN",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/FUTURE_STATIC_WRENCH_VALIDATION_PLAN.md",
        "markers": ("stationary, non-human fixture", "positive and negative directions", "known lever arms", "BASE_WRENCH_ROTATION_VERIFIED=true"),
    },
    {
        "id": "PARENT_ROTATION_RESULTS",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/WRENCH_ROTATION_UNIT_TEST_RESULTS.csv",
        "markers": ("CANONICAL_ACTIVE_XYZ_ROTATION", "FULL_MOMENT_REFERENCE_SHIFT", "False"),
    },
    {
        "id": "FORMAL_EXPERIMENT_MANIFEST",
        "path": "config/formal_experiment_manifest.json",
        "markers": ("ROM_PROTOCOL_V2", "q_hip - q_knee", ACTIVE_REFERENCE_SHA256),
    },
    {
        "id": "EXPERIMENT_SAFETY",
        "path": "config/experiment_safety.json",
        "markers": ('"max_force_n": null', '"workspace_min_base_m": null', '"reviewed_tool_name": null', '"reviewed": false'),
    },
    {
        "id": "PROJECT_SETTINGS",
        "path": "config/settings.py",
        "markers": ('ROBOT_FORCE_RAW_FRAME          = "world"', "ROBOT_FORCE_HZ                 = 50", "BASE_WRENCH_ROTATION_VERIFIED  = False"),
    },
    {
        "id": "REHAB_FRAME_CONFIG",
        "path": "config/rehab_frame_config.json",
        "markers": ('"rehab_x_axis_in_base": null', '"rehab_z_axis_in_base": null', '"reviewed": false'),
    },
    {
        "id": "REAL_IDENTIFICATION_CONFIG",
        "path": "config/real_identification_config.json",
        "markers": ('"raw_wrench_frame": null', '"force_sign_robot_on_leg": null', '"assumed_wrench_delay_s": null', '"reviewed": false'),
    },
    {
        "id": "LOCAL_XCORE_SDK_STUB",
        "path": "hardware/windows/xcoresdk/xCoreSDK_python/__init__.pyi",
        "markers": ("def getEndTorque", "FrameType::world", "cart_force", "单位N"),
    },
    {
        "id": "XCORE_QUERY_PROVENANCE",
        "path": "hardware/windows/rokae_xcore.py",
        "markers": ("getEndTorque", "force_query_started_s", "force_query_finished_s", "host_monotonic_time_s"),
    },
    {
        "id": "INTERNAL_WRENCH_SOURCE",
        "path": "hardware/windows/rokae_internal_wrench.py",
        "markers": ("rotation_only_pending_robot_validation", "session-local software reference bias", "get_world_to_base_rotation"),
    },
    {
        "id": "WRENCH_MATH",
        "path": "collection/state.py",
        "markers": ("transform_wrench", "F_b=R F_a", "p×(R F_a)"),
    },
    {
        "id": "DIAGNOSTIC_COMMON",
        "path": "scripts/rokae_diagnostic_common.py",
        "markers": ("never enable, move, drag", "rotation_push_analysis", "intentionally makes no statement about SDK gravity compensation"),
    },
    {
        "id": "QUERY_TIMING_DIAGNOSTIC",
        "path": "scripts/check_wrench_query_timing.py",
        "markers": ("without robot motion", "perf_counter_ns", "query_duration_ms"),
    },
    {
        "id": "FRAME_ROTATION_DIAGNOSTIC",
        "path": "scripts/check_wrench_frame_rotation.py",
        "markers": ("never sends a motion command", "expected_axis_positive", "does not set BASE_WRENCH_ROTATION_VERIFIED"),
    },
    {
        "id": "POSE_DEPENDENCE_DIAGNOSTIC",
        "path": "scripts/check_wrench_pose_dependence.py",
        "markers": ("no motion is issued", "does not determine whether xCoreSDK performs gravity compensation"),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("CSV rows must not be empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def verify_inputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in INPUT_SPECS:
        path = ROOT / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing input: {spec['path']}")
        content = path.read_text(encoding="utf-8")
        missing = [marker for marker in spec["markers"] if marker not in content]
        if missing:
            raise RuntimeError(f"semantic marker mismatch {spec['id']}: {missing}")
        digest = sha256_file(path)
        expected = spec.get("exact_sha256")
        if expected is not None and digest != expected:
            raise RuntimeError(f"exact SHA mismatch {spec['id']}: {digest}")
        rows.append({
            "input_id": spec["id"], "path": spec["path"], "sha256": digest,
            "semantic_markers": list(spec["markers"]), "semantic_markers_pass": True,
        })
    parent = read_json(ROOT / INPUT_SPECS[1]["path"])
    safety = read_json(ROOT / "config/experiment_safety.json")
    rehab = read_json(ROOT / "config/rehab_frame_config.json")
    if not (parent["readiness_decision"] == PARENT_STATUS and parent["VERIFIED_WRENCH_FRAME"] == VERIFIED_WRENCH_FRAME and parent["WRENCH_FORCE_SIGN_VERIFIED"] is False and parent["BASE_WRENCH_ROTATION_VERIFIED"] is False):
        raise RuntimeError("parent wrench status changed")
    if safety["reviewed"] is not False or safety["max_force_n"] is not None:
        raise RuntimeError("unexpected safety review or force limit")
    if rehab["reviewed"] is not False:
        raise RuntimeError("unexpected rehab-frame approval")
    return rows


def validation_hypotheses() -> dict[str, Any]:
    return {
        "hypothesis_set_id": "STATIC_WRENCH_HYPOTHESES_V1",
        "frozen_before_physical_result": True,
        "hypotheses": [
            {"id": "H1", "statement": "For a known static physical force direction, the reported world-frame force has its primary response aligned with that physical direction under one globally determined sign convention.", "primary": True},
            {"id": "H2", "statement": "Reversing the physical force direction produces a reliable sign reversal in the corresponding reported component.", "primary": True},
            {"id": "H3", "statement": "For the same world-referenced physical force direction at different tool orientations, the reported world-frame direction remains world-consistent rather than rotating with the tool.", "primary": True},
            {"id": "H4", "statement": "Zero-load baseline wrench can be quantified and pose dependence can be identified without assuming its controller/sensor origin.", "primary": True},
            {"id": "H5", "statement": "Host query timestamps are sufficient for steady-state static averaging but do not validate controller-source timing or dynamic synchronization.", "primary": True},
        ],
        "post_result_hypothesis_modification_allowed": False,
    }


def protocol_payload(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "formal_status": FORMAL_STATUS,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_status_sha256": PARENT_STATUS_SHA256,
        "protocol_scope": "FORCE_ONLY_FX_FY_FZ_STATIC_NON_HUMAN",
        "moment_status": "NOT_FULLY_VALIDATED",
        "hypotheses": validation_hypotheses(),
        "static_pose_roles": ["P0_CURRENT_SAFE_STATIONARY", "P1_CONDITIONAL_ORIENTATION_A", "P2_CONDITIONAL_ORIENTATION_B"],
        "directions": ["+WORLD_X", "-WORLD_X", "+WORLD_Y", "-WORLD_Y", "+WORLD_Z", "-WORLD_Z"],
        "load_levels": ["L1_REVIEWED_LOW", "L2_REVIEWED_HIGH"],
        "force_magnitudes_n": {"L1_REVIEWED_LOW": None, "L2_REVIEWED_HIGH": None},
        "force_magnitude_status": "FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW",
        "repetitions_per_pose_direction_level": REPETITIONS_PER_CELL,
        "host_queries_per_pre_load_post_window": HOST_QUERIES_PER_WINDOW,
        "primary_response_contrast": "DeltaF = mean(F_load) - 0.5*(mean(F_pre_zero)+mean(F_post_zero)); raw windows also retained",
        "thresholds": {
            "direction_angle_error_max_deg": None,
            "cross_axis_leakage_max": None,
            "sign_reversal_min_component_n": None,
            "zero_drift_max_n": None,
            "pose_consistency_max_deg": None,
            "minimum_signal_to_zero_noise": None,
        },
        "threshold_status": "THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE",
        "future_result_decisions": list(FUTURE_DECISIONS),
        "update_conditions": "only all preregistered gates after authorized execution; this protocol changes no wrench flag",
        "input_files": inputs,
        "forbidden_operations": [
            "connect_robot", "power_or_enable_robot", "send_motion", "position_robot",
            "execute_rehabilitation_trajectory", "apply_human_or_operator_hand_load",
            "modify_hardware_control_or_safety", "choose_load_from_sdk_result",
            "change_hypothesis_direction_or_repetitions_after_result", "drop_failed_axis_posthoc",
            "compute_primary_endpoint", "run_PINN", "run_BO", "execute_protocol",
        ],
        "protocol_frozen_before_any_physical_result": True,
    }


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"prepare requires empty output directory: {OUTPUT}")
    inputs = verify_inputs()
    atomic_json(PROTOCOL_PATH, protocol_payload(inputs))
    atomic_json(OUTPUT / "INPUT_VERIFICATION.json", {
        "stage_id": STAGE_ID, "input_count": len(inputs),
        "all_inputs_present_and_semantically_verified": True, "inputs": inputs,
        "physical_result_files_read": [], "robot_access_count": 0,
        "human_data_access_count": 0, "physical_load_application_count": 0,
    })
    atomic_json(OUTPUT / "HARDWARE_ACCESS_AUDIT.json", {
        "protocol_design_only": True, "robot_constructed": False,
        "robot_connected": False, "power_or_enable_count": 0,
        "motion_or_position_command_count": 0, "load_application_count": 0,
        "human_or_operator_hand_loading_count": 0, "endpoint_computation_count": 0,
    })
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "input_count": len(inputs)}, indent=2))


def verify_freeze() -> dict[str, Any]:
    if FROZEN_PROTOCOL_SHA256 == "TO_BE_FROZEN_AFTER_PREPARE":
        raise RuntimeError("protocol SHA has not been frozen")
    if sha256_file(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("static wrench protocol SHA mismatch")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["input_files"] != verify_inputs():
        raise RuntimeError("frozen inputs changed")
    return protocol


def pose_plan() -> dict[str, Any]:
    return {
        "plan_id": "STATIC_POSE_PLAN_V1",
        "current_execution_authorized": False,
        "pose_dependence_minimum_distinct_orientations": 2,
        "planned_pose_count": 3,
        "poses": [
            {"pose_id": "P0_CURRENT_SAFE_STATIONARY", "role": "single-pose frame/sign/zero block", "joint_position_rad": None, "tcp_pose_base_m_rad": None, "positioning_authorized_by_protocol": False, "eligibility": "existing stationary pose only after independent identity/state/safety review", "required": True},
            {"pose_id": "P1_CONDITIONAL_ORIENTATION_A", "role": "first different tool orientation for H3/H4", "joint_position_rad": None, "tcp_pose_base_m_rad": None, "positioning_authorized_by_protocol": False, "eligibility": "exact pose and separate safe positioning procedure must be reviewed/frozen before result", "required": False},
            {"pose_id": "P2_CONDITIONAL_ORIENTATION_B", "role": "second non-degenerate orientation/pose replication", "joint_position_rad": None, "tcp_pose_base_m_rad": None, "positioning_authorized_by_protocol": False, "eligibility": "exact pose and separate safe positioning procedure must be reviewed/frozen before result", "required": False},
        ],
        "pose_exclusion_principles": ["not near joint limit", "not near singularity", "not near workspace boundary", "fixture/load line unobstructed", "stable operation state", "approved tool/load configuration"],
        "if_only_P0_available": {"H1_H2_single_pose_may_be_assessed": True, "H3_status": "POSE_DEPENDENCE_NOT_YET_VALIDATED", "full_world_frame_decision_allowed": False},
        "post_result_pose_selection_allowed": False,
    }


def load_direction_rows() -> list[dict[str, Any]]:
    directions = [
        ("PX", "+WORLD_X", "[1,0,0]"), ("NX", "-WORLD_X", "[-1,0,0]"),
        ("PY", "+WORLD_Y", "[0,1,0]"), ("NY", "-WORLD_Y", "[0,-1,0]"),
        ("PZ", "+WORLD_Z", "[0,0,1]"), ("NZ", "-WORLD_Z", "[0,0,-1]"),
    ]
    poses = [
        ("P0_CURRENT_SAFE_STATIONARY", "REQUIRED_IF_FUTURE_EXECUTION_AUTHORIZED"),
        ("P1_CONDITIONAL_ORIENTATION_A", "CONDITIONAL_ON_SEPARATE_POSE_APPROVAL"),
        ("P2_CONDITIONAL_ORIENTATION_B", "CONDITIONAL_ON_SEPARATE_POSE_APPROVAL"),
    ]
    levels = ["L1_REVIEWED_LOW", "L2_REVIEWED_HIGH"]
    rows: list[dict[str, Any]] = []
    for pose_id, pose_status in poses:
        for short, direction, unit in directions:
            for level in levels:
                axis = direction[-1]
                preferred = "calibrated bidirectional force gauge or cable/pulley fixture aligned by external metrology"
                if axis == "Z":
                    preferred = "calibrated hanging mass only if gravity aligns with registered world axis; otherwise calibrated force gauge/pulley"
                rows.append({
                    "cell_id": f"{pose_id}_{short}_{level}", "pose_id": pose_id,
                    "pose_status": pose_status, "direction_id": direction,
                    "applied_unit_vector_world": unit, "load_level_id": level,
                    "force_magnitude_n": None, "repetitions": REPETITIONS_PER_CELL,
                    "queries_per_pre_load_post_window": HOST_QUERIES_PER_WINDOW,
                    "preferred_application": preferred,
                    "same_direction_alternative": "independently calibrated rigid/cable fixture producing the same registered world vector",
                    "if_direction_not_safe_or_reliable": "NOT_EXECUTED_AND_AXIS_REMAINS_UNVALIDATED",
                    "posthoc_drop_or_direction_change_allowed": False,
                })
    return rows


def known_load_plan() -> str:
    return """# Known Load Application Plan

## Required validation equipment

- independently calibrated bidirectional force gauge or load cell with current certificate and uncertainty;
- rigid fixture or low-friction cable/pulley arrangement that does not require a person to hold the load;
- calibrated masses only for directions that can be registered to gravity without unsafe side loading;
- independent metrology/inclinometer/fixture registration for world axes and load line;
- non-human rigid end-effector attachment/phantom and secondary retention against dropped masses;
- fixture/load identifiers, calibration records and environmental metadata.

Operator hand pushing and human/subject loading are prohibited as primary calibrated evidence. The controller/TCP orientation alone cannot define a physical world load. Before execution, register the fixture axes to controller world using independent physical references and freeze the transform and its uncertainty.

Two load roles are preregistered: `L1_REVIEWED_LOW` and `L2_REVIEWED_HIGH`. Their N values are null: `FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW`. Reviewers must choose values before physical results that are above calibrated zero/noise resolution yet below the most conservative reviewed robot, fixture and load limits with margin. The SDK output cannot be used to choose or increase them. If only one level is approved, frame/sign may be assessed but magnitude linearity is unavailable.

Apply the complete frozen matrix. A cable/pulley or rigid force-gauge arrangement may replace another method only if it produces the same preregistered world vector. If an axis cannot be safely/reliably implemented, record `AXIS_NOT_EXECUTED`; do not substitute a different direction or delete it after seeing results. The future decision can then be partial or not validated, never full.
"""


def zero_bias_plan() -> str:
    return """# Zero, Bias and Drift Plan

For every pose block and every independent load repetition, retain raw `PRE_LOAD_ZERO`, `LOAD`, and `POST_LOAD_ZERO` windows. Each window targets 100 valid host queries; if a future reviewed maximum dwell expires first, the cell is incomplete rather than silently shortened.

For Fx/Fy/Fz separately report raw mean, median, SD, minimum/maximum, valid count, query duration and window timestamps. Report post-minus-pre drift, within-pose baseline dispersion, between-repetition drift and between-pose zero difference. Preserve raw values before any software zero.

The validation response contrast is preregistered as `DeltaF = mean(F_load) - 0.5*(mean(F_pre)+mean(F_post))`. This local contrast isolates the applied-load response for frame/sign testing; it is not a conclusion that simple zero subtraction is an appropriate production compensation model. Results must also be shown raw and with pre-only/post-only contrasts as diagnostics. Pose dependence cannot be labelled sensor error while controller compensation/tool/load state remains unresolved.
"""


def sign_metrics() -> dict[str, Any]:
    return {
        "metrics_id": "STATIC_SIGN_VALIDATION_METRICS_V1",
        "response_vector": "DeltaF_world = mean(load) - 0.5*(mean(pre_zero)+mean(post_zero))",
        "paired_axes": [["+WORLD_X", "-WORLD_X"], ["+WORLD_Y", "-WORLD_Y"], ["+WORLD_Z", "-WORLD_Z"]],
        "dominant_component": "dot(DeltaF_world, d_applied_world)",
        "raw_axis_sign_reversal": "DeltaF_plus[axis] * DeltaF_minus[axis] < 0",
        "global_sign_candidates": {
            "SAME_DIRECTION": "reported response aligns with applied force for every paired axis/pose/level",
            "OPPOSITE_DIRECTION": "reported response aligns with negative applied force for every paired axis/pose/level",
        },
        "sign_rule": "select at most one global convention from all preregistered paired cells; per-axis sign flipping is forbidden",
        "SIGN_REVERSAL_PASS": "all required + / - pairs reverse and meet preregistered signal/uncertainty gates under one global convention",
        "thresholds": {"minimum_component_n": None, "minimum_signal_to_zero_noise": None, "repeatability_limit": None},
        "threshold_status": "THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE",
        "magnitude_similarity_cannot_replace_sign": True,
        "post_result_flip_or_threshold_change_allowed": False,
    }


def frame_metrics() -> dict[str, Any]:
    return {
        "metrics_id": "STATIC_FRAME_DIRECTION_VALIDATION_METRICS_V1",
        "normalized_reported": "u_reported = normalize(DeltaF_world)",
        "same_direction_angle_rad": "theta_same = acos(clip(dot(u_reported, d_applied_world), -1, 1))",
        "opposite_direction_angle_rad": "theta_opposite = acos(clip(dot(u_reported, -d_applied_world), -1, 1))",
        "sign_handling": "report both angles before sign resolution; after paired +/- analysis use only the single global sign convention",
        "dominant_component_abs_n": "abs(dot(DeltaF_world, d_applied_world))",
        "orthogonal_vector": "F_orth = DeltaF_world - dot(DeltaF_world,d_applied_world)*d_applied_world",
        "CROSS_AXIS_LEAKAGE": "norm(F_orth) / max(abs(dot(DeltaF_world,d_applied_world)), epsilon_preregistered)",
        "pose_consistency_angle_rad": "acos(clip(dot(u_reported_pose_i,u_reported_pose_j),-1,1)) after one global sign convention",
        "secondary_magnitude_linearity": "if two approved known levels: preregistered regression of response magnitude versus applied magnitude; report slope/intercept/residual/R2",
        "primary_frame_gate_requires": ["all six directions executed", "one global sign convention", "direction-angle gate", "cross-axis gate", "at least two independently approved tool orientations", "world-axis registration valid", "repeatability gate"],
        "thresholds": {"direction_angle_max_rad": None, "cross_axis_leakage_max": None, "pose_consistency_angle_max_rad": None, "epsilon_n": None},
        "threshold_status": "THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE",
        "magnitude_calibration_required_for_frame_gate": False,
    }


def pose_consistency_plan() -> str:
    return """# Pose and Orientation Consistency Plan

H3 requires at least two separately approved non-degenerate tool orientations. Repeat the same registered world ±X/±Y/±Z loads without redefining axes from TCP orientation. Compare normalized response directions and cross-axis leakage across poses. A genuine world-expression output should stay world-consistent; a tool-following response will rotate with tool orientation.

P0 is the existing safe stationary pose candidate. P1/P2 contain no coordinates and this protocol authorizes no positioning. Before results, a separate safety-reviewed positioning procedure must freeze exact joint/TCP poses and prove they are away from joint limits, singularity and workspace boundaries. If no such procedure exists, execute P0 only and report `POSE_DEPENDENCE_NOT_YET_VALIDATED`; full world-frame validation is forbidden.

The previous 9/9 canonical cases remain `MATHEMATICAL_TRANSFORM_VERIFIED` only. Future physical world-axis registration plus known-load results must independently pass before any consideration of `BASE_WRENCH_ROTATION_VERIFIED=true`.
"""


def sampling_plan() -> str:
    return f"""# Sampling and Timestamp Plan

Master clock: `HOST_MONOTONIC_PERF_COUNTER_NS`.

For every PRE/LOAD/POST window record query start, query end, midpoint, publish time, query duration, sequence ID, raw requested frame, raw Fx/Fy/Fz, joint torque arrays for provenance, validity and invalid reason. Retain robot state host receive time, TCP/tool pose, operation state and state-wrench skew as context; no device/source timestamp exists.

Each window targets `{HOST_QUERIES_PER_WINDOW}` valid host queries. Each pose x direction x approved-level cell has exactly `{REPETITIONS_PER_CELL}` independent load applications. This supplies repeated means/SDs without allowing post-result additions. The nominal configured query target is 50 Hz, but observed query rate/source-update behavior must be reported rather than assumed; repeated identical values cannot prove unique controller updates.

Record latency distributions and missed/invalid queries. Future maximum dwell, query failure and steady-force acceptance rules require safety/calibration review and remain null. Host timing is sufficient for steady-state window averages only:

`STATIC_FRAME_VALIDATION != DYNAMIC_SYNCHRONIZATION_VALIDATION`

No result may claim transport delay, controller source latency or dynamic command-state-wrench alignment from this protocol.
"""


def safety_preconditions() -> dict[str, Any]:
    checks = [
        ("SITE_SAFETY_REVIEW", False, "site owner approves static non-human fixture and procedure"),
        ("EXPERIMENT_SAFETY_CONFIG_REVIEWED", False, "config/experiment_safety.json reviewed=true"),
        ("STATIC_FORCE_LIMITS_APPROVED", False, "both load values and fixture/robot margins approved"),
        ("MAXIMUM_STATIC_DWELL_APPROVED", False, "maximum load and zero window dwell approved"),
        ("ROBOT_IDENTITY_TOOL_PAYLOAD_TCP_VERIFIED", False, "model/serial/controller/tool/load/TCP match reviewed record"),
        ("COMPENSATION_STATE_RESOLVED", False, "controller/tool/load compensation state documented or explicitly unresolved for interpretation"),
        ("WORLD_AXIS_REGISTRATION_VALID", False, "independent physical world/fixture registration and uncertainty frozen"),
        ("POSES_AND_POSITIONING_SEPARATELY_APPROVED", False, "exact poses and any positioning procedure independently approved"),
        ("OPERATION_STATE_STABLE_CONFIRMED", False, "stationary safe state confirmed"),
        ("EMERGENCY_STOP_AND_OPERATOR_ROLES_READY", False, "stop access, roles and communications reviewed"),
        ("CALIBRATED_NONHUMAN_LOAD_EQUIPMENT_READY", False, "certificate/uncertainty/retention and fixture inspection pass"),
        ("NO_HUMAN_SUBJECT_OR_HAND_LOADING", True, "protocol prohibits human/hand calibrated load"),
        ("LOGGER_AND_RAW_PROVENANCE_VALIDATED", False, "all raw/time/metadata streams verified before load"),
        ("CLEANUP_DISCONNECT_PROCEDURE_REVIEWED", False, "safe release/cleanup/disconnect plan reviewed"),
        ("METRIC_THRESHOLDS_FROZEN_BEFORE_RESULT", False, "calibration-derived gates preregistered before reveal"),
    ]
    return {
        "gate_id": "STATIC_WRENCH_FUTURE_EXECUTION_PRECONDITIONS_V1",
        "current_authorization": "NOT_AUTHORIZED",
        "fail_closed": True,
        "checks": [{"check_id": check_id, "current_pass": current, "requirement": requirement} for check_id, current, requirement in checks],
        "all_required_to_authorize": True,
        "null_or_unreviewed_values_may_not_be_filled_by_protocol": True,
        "on_any_failure": "DO_NOT_CONNECT_OR_APPLY_LOAD; preserve NOT_AUTHORIZED",
        "this_protocol_authorizes_robot_positioning": False,
        "this_protocol_authorizes_physical_execution": False,
    }


def future_result_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "STATIC_WRENCH_FRAME_SIGN_FUTURE_RESULT_V1",
        "type": "object",
        "required": ["protocol_identity", "execution_authorization", "setup", "raw_data", "zero_results", "cell_results", "hypotheses", "decision", "update_eligibility"],
        "properties": {
            "protocol_identity": {"type": "object", "properties": {"protocol_sha256": {"const": FROZEN_PROTOCOL_SHA256}, "result_reveal_after_threshold_freeze": {"type": "boolean"}}},
            "execution_authorization": {"type": "object", "properties": {"authorized": {"const": True}, "precondition_gate_sha256": {"type": "string"}}},
            "setup": {"description": "robot/SDK/controller, active tool/payload/TCP, compensation state, fixture/calibration, world registration, exact pose hashes and approved load values"},
            "raw_data": {"description": "immutable PRE/LOAD/POST host-query and state records with checksums"},
            "zero_results": {"description": "per axis/pose pre/post mean median SD drift and pose dependence"},
            "cell_results": {"description": "all preregistered cells including invalid/not-executed cells; response, both direction angles, leakage and repeats"},
            "hypotheses": {"description": "H1-H5 pass/fail/indeterminate with preregistered evidence"},
            "decision": {"enum": list(FUTURE_DECISIONS)},
            "update_eligibility": {"type": "object", "properties": {"VERIFIED_WRENCH_FRAME": {"type": ["string", "null"]}, "WRENCH_FORCE_SIGN_VERIFIED": {"type": "boolean"}, "WRENCH_FORCE_SIGN_CONVENTION": {"enum": ["SAME_DIRECTION", "OPPOSITE_DIRECTION", None]}, "BASE_WRENCH_ROTATION_VERIFIED": {"type": "boolean"}, "moment_validated": {"const": False}, "task_direction_status": {"const": TASK_DIRECTION_STATUS}, "primary_endpoint_ready": {"const": False}}},
        },
        "full_validation_requirements": ["H1 PASS all axes", "H2 PASS all paired axes with one global sign", "H3 PASS at >=2 approved orientations", "H4 zero/bias characterized", "H5 interpretation respected", "all thresholds frozen before reveal", "no missing required cell"],
        "partial_axis_success_cannot_produce_full_validation": True,
        "moment_remains_not_fully_validated": True,
        "static_validation_does_not_validate_dynamic_synchronization": True,
    }


def next_dependency_plan() -> str:
    return f"""# Next Dependency Plan

The wrench protocol and actual strap geometry use different reference equipment, uncertainties and scientific gates, so they remain independent. A future successful static wrench result would establish force expression/sign only; it would not measure the two physical strap attachment points or validate their pose dependence.

Exact next dependency: `{NEXT_DEPENDENCY}`.

That protocol should register `p_limb_attach_B(t)` and `p_robot_attach_B(t)`, fixture/bed/base frames, placement repeatability, tautness/routing and direction uncertainty. Only after both wrench and geometry validation can the primary endpoint definition be reconsidered. This dependency was not executed.
"""


def report() -> str:
    return f"""# Static Wrench Frame/Sign Validation Protocol V1

## Formal status

`{FORMAL_STATUS}`

This stage froze a future force-only, static, non-human protocol. It performed no physical validation.

## Hypotheses and design

H1-H5 are frozen unchanged: axis/direction response, paired sign reversal, cross-pose world consistency, zero/pose bias characterization and host-timing-only interpretation. The full matrix contains three pose roles, six world directions and two load-level roles. Each cell has `{REPETITIONS_PER_CELL}` independent load applications and `{HOST_QUERIES_PER_WINDOW}` host queries per PRE/LOAD/POST window.

P0 is only the current stationary-pose role. P1/P2 contain no coordinates and require separate positioning approval. If only P0 is possible, H3 becomes `POSE_DEPENDENCE_NOT_YET_VALIDATED` and full world-frame validation is prohibited.

## Load and thresholds

Loads must come from calibrated non-human equipment, never a hand or subject. The two N values remain null: `FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW`. Direction-angle, leakage, sign, drift, pose-consistency and SNR thresholds remain null: `THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE`. They must be frozen before physical result reveal; the SDK result cannot tune them.

## Metrics

Raw PRE/LOAD/POST windows remain immutable. The preregistered response contrast is `DeltaF=mean(load)-0.5*(mean(pre)+mean(post))`. Both same-direction and opposite-direction angle errors are reported until paired +/- cells establish at most one global sign convention. Cross-axis leakage is orthogonal norm divided by dominant-axis magnitude. Magnitude linearity is secondary and available only with two approved known levels.

## Safety and current authorization

The current safety file is unreviewed, force/workspace/tool values are null, world/bed registration is absent and physical equipment is unspecified. Therefore future physical execution is `NOT_AUTHORIZED`. This protocol does not authorize connection, power, enable, positioning, load or motion.

## State preservation

`REQUESTED_WRENCH_FRAME={REQUESTED_WRENCH_FRAME}`; `VERIFIED_WRENCH_FRAME={VERIFIED_WRENCH_FRAME}`; `WRENCH_FORCE_SIGN_VERIFIED=false`; `BASE_WRENCH_ROTATION_VERIFIED=false`; `{TASK_DIRECTION_STATUS}`. Moment remains `NOT_FULLY_VALIDATED`, and no endpoint was computed/finalized/validated.

## Next dependency

`{NEXT_DEPENDENCY}`. It remains separate because wrench response and strap geometry require different evidence. It was not executed.
"""


def write_checksums() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in files) + "\n")


def execute() -> None:
    protocol = verify_freeze()
    atomic_json(OUTPUT / "VALIDATION_HYPOTHESES.json", validation_hypotheses())
    atomic_json(OUTPUT / "STATIC_POSE_PLAN.json", pose_plan())
    atomic_text(OUTPUT / "KNOWN_LOAD_APPLICATION_PLAN.md", known_load_plan())
    atomic_csv(OUTPUT / "LOAD_DIRECTION_MATRIX.csv", load_direction_rows())
    atomic_text(OUTPUT / "ZERO_BIAS_DRIFT_PLAN.md", zero_bias_plan())
    atomic_json(OUTPUT / "SIGN_VALIDATION_METRICS.json", sign_metrics())
    atomic_json(OUTPUT / "FRAME_DIRECTION_VALIDATION_METRICS.json", frame_metrics())
    atomic_text(OUTPUT / "POSE_CONSISTENCY_PLAN.md", pose_consistency_plan())
    atomic_text(OUTPUT / "SAMPLING_AND_TIMESTAMP_PLAN.md", sampling_plan())
    atomic_json(OUTPUT / "SAFETY_PRECONDITIONS.json", safety_preconditions())
    atomic_json(OUTPUT / "FUTURE_RESULT_SCHEMA.json", future_result_schema())
    atomic_text(OUTPUT / "NEXT_DEPENDENCY_PLAN.md", next_dependency_plan())
    atomic_text(OUTPUT / "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_REPORT.md", report())
    atomic_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID, "formal_status": FORMAL_STATUS,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "input_count": len(protocol["input_files"]),
        "direction_matrix_cell_count": len(load_direction_rows()),
        "repetitions_per_cell": REPETITIONS_PER_CELL,
        "host_queries_per_window": HOST_QUERIES_PER_WINDOW,
        "force_magnitudes_frozen": False, "metric_thresholds_frozen": False,
        "future_physical_execution_authorized": False,
        "physical_validation_performed": False,
        "requested_wrench_frame": REQUESTED_WRENCH_FRAME,
        "verified_wrench_frame": VERIFIED_WRENCH_FRAME,
        "wrench_force_sign_verified": False, "base_wrench_rotation_verified": False,
        "moment_fully_validated": False, "task_direction_status": TASK_DIRECTION_STATUS,
        "primary_endpoint_finalized": False, "primary_endpoint_validated": False,
        "robot_access_count": 0, "motion_command_count": 0,
        "physical_load_application_count": 0, "human_data_access_count": 0,
        "pinn_run_count": 0, "bo_run_count": 0,
        "hardware_control_safety_modified": False, "frozen_artifacts_modified": False,
        "not_human_ready": True, "not_robot_approved": True,
        "next_dependency": NEXT_DEPENDENCY, "next_dependency_executed": False,
    })
    write_checksums()
    print(json.dumps({
        "stage_id": STAGE_ID, "formal_status": FORMAL_STATUS,
        "future_execution": "NOT_AUTHORIZED", "direction_matrix_cells": len(load_direction_rows()),
        "next_dependency": NEXT_DEPENDENCY,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute()


if __name__ == "__main__":
    main()
