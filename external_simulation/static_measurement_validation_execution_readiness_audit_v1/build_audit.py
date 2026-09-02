"""Build a read-only execution-readiness audit; never execute validation.

This module verifies pinned protocol/config/code evidence and writes only the
requested readiness artifacts.  It does not import robot/control/safety
modules, connect to hardware, select load values, or collect physical data.
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


STAGE_ID = "STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_AUDIT_V1"
AUDIT_TYPE = "EXECUTION_READINESS_SAFETY_EQUIPMENT_CONFIG_REVIEW"
DECISION = "STATIC_MEASUREMENT_VALIDATION_EXECUTION_NOT_READY"
NEXT_ACTION = "RESOLVE_MINIMUM_BLOCKING_ITEMS"
WRENCH_PROTOCOL_STATUS = "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED"
GEOMETRY_PROTOCOL_STATUS = "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED"
WRENCH_PROTOCOL_SHA256 = "c88799b838f6304765acb643a706b1a6f1bbe02b1ee4f6c07ed9c486eab2f5c1"
GEOMETRY_PROTOCOL_SHA256 = "4da84b5ffde2bb5c7dc7b3baccf81cb1dbd8a7a0cb8c3bdd8d457a4f3399e337"
ACTIVE_REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
ALLOWED_CLASSIFICATIONS = {
    "REVIEWED_AND_READY",
    "DEFINED_BUT_NOT_REVIEWED",
    "MISSING",
    "NOT_APPLICABLE",
}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/static_measurement_validation_execution_readiness_audit_v1"


SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "WRENCH_PROTOCOL",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/STATIC_WRENCH_VALIDATION_PROTOCOL.json",
        "sha256": WRENCH_PROTOCOL_SHA256,
        "markers": (WRENCH_PROTOCOL_STATUS, "FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW", "THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE"),
    },
    {
        "id": "WRENCH_SAFETY_PRECONDITIONS",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/SAFETY_PRECONDITIONS.json",
        "sha256": "e554510c3ad63a67878068910135423c2cf68ac4746b4e319ed7a0754b23bb13",
        "markers": ("NOT_AUTHORIZED", "STATIC_FORCE_LIMITS_APPROVED", "LOGGER_AND_RAW_PROVENANCE_VALIDATED"),
    },
    {
        "id": "STATIC_POSE_PLAN",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/STATIC_POSE_PLAN.json",
        "sha256": "c7fd9207b1b5c4e8a1e84a6139056952e49df85de2040a927ac86c14f46c0d20",
        "markers": ("P0_CURRENT_SAFE_STATIONARY", "POSE_DEPENDENCE_NOT_YET_VALIDATED", "full_world_frame_decision_allowed"),
    },
    {
        "id": "KNOWN_LOAD_PLAN",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/KNOWN_LOAD_APPLICATION_PLAN.md",
        "sha256": "3fcd7a20b8a8f8afed66d70b66e99802e62f2fbaf6d527041ab820c999f7b993",
        "markers": ("calibrated bidirectional force gauge", "Operator hand pushing", "N values are null"),
    },
    {
        "id": "SAMPLING_PLAN",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/SAMPLING_AND_TIMESTAMP_PLAN.md",
        "sha256": "b20d12114fb418a94876d76ece00a980ae12d735d404fde8fce31497c798dffd",
        "markers": ("HOST_MONOTONIC_PERF_COUNTER_NS", "PRE/LOAD/POST", "STATIC_FRAME_VALIDATION != DYNAMIC_SYNCHRONIZATION_VALIDATION"),
    },
    {
        "id": "WRENCH_RESULT_SCHEMA",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/FUTURE_RESULT_SCHEMA.json",
        "sha256": "8a2089b9e7115878894ab20e1e6e7e8d3b2fa3ce7aef63d15f7ecb36bc866443",
        "markers": (WRENCH_PROTOCOL_SHA256, "immutable PRE/LOAD/POST", "active tool/payload/TCP"),
    },
    {
        "id": "GEOMETRY_PROTOCOL",
        "path": "external_simulation_audits/static_strap_pull_geometry_validation_protocol_v1/STRAP_GEOMETRY_VALIDATION_PROTOCOL.json",
        "sha256": GEOMETRY_PROTOCOL_SHA256,
        "markers": (GEOMETRY_PROTOCOL_STATUS, "REHAB_SETUP_FRAME", "GEOMETRY_PRELOAD_REQUIRES_INDEPENDENT_FIXTURE_AND_SAFETY_REVIEW"),
    },
    {
        "id": "GEOMETRY_VALIDITY_GATE",
        "path": "external_simulation_audits/static_strap_pull_geometry_validation_protocol_v1/GEOMETRY_VALIDITY_GATE.json",
        "sha256": "f6f9b4860b0aedf3382acfa7cc47e3b030cd49f0e69054181f5476c1ec788ae2",
        "markers": ("FAIL_CLOSED_NOT_EXECUTED", "calibrated T_B_R", "fallback_direction_allowed"),
    },
    {
        "id": "GEOMETRY_MEASUREMENT_PLAN",
        "path": "external_simulation_audits/static_strap_pull_geometry_validation_protocol_v1/STATIC_GEOMETRY_MEASUREMENT_PLAN.md",
        "sha256": "964b5762694edc8523d53faab1b3c445153af59ca6dc8fb13954e52a617322d8",
        "markers": ("10` complete remove/reattach", "rigid shank surrogate", "GEOMETRY_PRELOAD_REQUIRES_INDEPENDENT_FIXTURE_AND_SAFETY_REVIEW"),
    },
    {
        "id": "FRAME_CALIBRATION_PLAN",
        "path": "external_simulation_audits/static_strap_pull_geometry_validation_protocol_v1/ROBOT_TO_SETUP_FRAME_CALIBRATION_PLAN.md",
        "sha256": "153664e340d8d82fecbe02af9b426709b6143d397b00098c7695e5271cde0aae",
        "markers": ("at least three non-collinear", "T_B_R", "Robot TCP probing is optional only"),
    },
    {
        "id": "EXPERIMENT_SAFETY_CONFIG",
        "path": "config/experiment_safety.json",
        "sha256": "2f60266b6a16f7911a585aad85fe2bcb4191144672d337d3960fc113e5c36b66",
        "markers": ('"max_force_n": null', '"expected_robot_model": null', '"reviewed": false'),
    },
    {
        "id": "REAL_IDENTIFICATION_CONFIG",
        "path": "config/real_identification_config.json",
        "sha256": "8979412db4d3c7ef4d0886eaebc6edeaa2474679ef33ac434c1c57d6c3cd2ba4",
        "markers": ('"raw_wrench_frame": null', '"force_sign_robot_on_leg": null', '"reviewed": false'),
    },
    {
        "id": "REHAB_FRAME_CONFIG",
        "path": "config/rehab_frame_config.json",
        "sha256": "eeb4b28a359acc74ab9cf9dcdf5ebbf41bf8f40e4eae1020045011aecdd57606",
        "markers": ('"rehab_x_axis_in_base": null', '"rehab_z_axis_in_base": null', '"reviewed": false'),
    },
    {
        "id": "COLLECTOR",
        "path": "collection/collector.py",
        "sha256": "18e1731497c25259d0d225778c0b9b235786a30ec4d6af87e4c18578f1bb6513",
        "markers": ("force_query_started_s", "force_query_finished_s", "fx_raw_n", "get_robot_metadata"),
    },
    {
        "id": "EPISODE_LOGGER",
        "path": "collection/episode_logger.py",
        "sha256": "831d4326b247ed95ccc9b9e45690332711077d05f7397a6cbd74d808c910187a",
        "markers": ("query_start_s", "query_end_s", "The logger never substitutes a numeric zero"),
    },
    {
        "id": "ROBOT_STATE_SCHEMA",
        "path": "collection/state.py",
        "sha256": "d5c2b4c67baca13922e5d0e8fb80a5d10941e160d2e3990287299b8c3cef88fd",
        "markers": ("force_query_started_s", "force_query_finished_s", "cartesian_force_raw_n"),
    },
    {
        "id": "XCORE_ADAPTER",
        "path": "hardware/windows/rokae_xcore.py",
        "sha256": "5d5908b0e504a729357b91ca8b31158f7ed6c8c8b8e5b8659bcef25ece22cca3",
        "markers": ("get_end_wrench", "host_monotonic_time_s", "active_hmi_tool_workobject_verified"),
    },
    {
        "id": "READONLY_DIAGNOSTIC_HELPER",
        "path": "scripts/rokae_diagnostic_common.py",
        "sha256": "cab5f8f075228702aae319700d6fba370b2032a522475593bdeab09ee26001d7",
        "markers": ("never enable, move, drag", "readonly_connection", "robot.disconnect"),
    },
    {
        "id": "WRENCH_QUERY_DIAGNOSTIC",
        "path": "scripts/check_wrench_query_timing.py",
        "sha256": "01c905104eeee051da2bc3982c4d4c527daba43317cddfc0c6465e620f46d6f3",
        "markers": ("without robot motion", "perf_counter_ns", "query_duration_ms"),
    },
    {
        "id": "ACTIVE_REFERENCE",
        "path": "reference_release/reference_measured_asymmetric_closed_slow.csv",
        "sha256": ACTIVE_REFERENCE_SHA256,
        "markers": ("reference_measured_asymmetric_closed_slow", "theta_shank = q_hip - q_knee"),
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


def verify_sources() -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = ROOT / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing input: {spec['path']}")
        content = path.read_text(encoding="utf-8")
        missing = [marker for marker in spec["markers"] if marker not in content]
        if missing:
            raise RuntimeError(f"semantic marker mismatch {spec['id']}: {missing}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"exact SHA mismatch {spec['id']}: {digest}")
        verified.append({
            "input_id": spec["id"], "path": spec["path"], "sha256": digest,
            "semantic_markers": list(spec["markers"]), "semantic_markers_pass": True,
        })

    wrench = read_json(ROOT / SOURCE_SPECS[0]["path"])
    geometry = read_json(ROOT / SOURCE_SPECS[6]["path"])
    safety = read_json(ROOT / "config/experiment_safety.json")
    real_config = read_json(ROOT / "config/real_identification_config.json")
    rehab = read_json(ROOT / "config/rehab_frame_config.json")
    if wrench["formal_status"] != WRENCH_PROTOCOL_STATUS:
        raise RuntimeError("frozen wrench protocol status changed")
    if geometry["formal_status"] != GEOMETRY_PROTOCOL_STATUS:
        raise RuntimeError("frozen geometry protocol status changed")
    if safety["reviewed"] is not False or safety["max_force_n"] is not None:
        raise RuntimeError("unexpected safety approval or force limit")
    if real_config["reviewed"] is not False or rehab["reviewed"] is not False:
        raise RuntimeError("unexpected measurement/frame review")
    return verified


def safety_rows() -> list[dict[str, Any]]:
    rows = [
        ("S01", "site authorization", "site owner approval for static nonhuman procedure", "no approval record", "MISSING", True, True, True, "obtain signed site-specific review"),
        ("S02", "experiment safety", "experiment_safety reviewed flag", "reviewed=false", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "review every field and set only through site approval"),
        ("S03", "robot identity", "model/serial/controller identity", "all expected values null", "MISSING", True, True, True, "record and independently match live identity"),
        ("S04", "tool/TCP/payload", "tool, workpiece, payload and TCP configuration", "names/mass/CoG/inertia null; active HMI setup not verified", "MISSING", True, True, True, "freeze reviewed installed configuration"),
        ("S05", "static force limits", "reviewed external-force and robot/fixture margins", "max_force_n and max_torque_nm null", "MISSING", True, True, True, "approve limits before choosing load levels"),
        ("S06", "static duration", "maximum PRE/LOAD/POST dwell and total duration", "null/not recorded", "MISSING", True, True, True, "freeze dwell, timeout and maximum session duration"),
        ("S07", "workspace/joints/collision", "workspace, soft limits and collision configuration", "workspace/soft limits null; reviews false", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "verify exact P0; additionally review positioning path for P1/P2"),
        ("S08", "state freshness", "state/wrench age, skew and query failure limits", "schema exists; site safety values null", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "review static acquisition acceptance limits"),
        ("S09", "operation state", "stationary safe robot state", "protocol requirement exists; no current episode record", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "confirm and record on execution day"),
        ("S10", "emergency response", "E-stop access, supervisor/operator roles and communications", "no reviewed role/stop record", "MISSING", True, True, True, "freeze roles and abort authority"),
        ("S11", "abort conditions", "fixture slip, overload, state/logger failure and operator concern", "principle defined; thresholds/dwell/load-release sequence missing", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "freeze static-specific abort and load-release procedure"),
        ("S12", "cleanup/disconnect", "safe unload, fixture release, cleanup and disconnect", "read-only disconnect helper exists; physical unload procedure unreviewed", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "review unloading before disconnect and failure cleanup"),
        ("S13", "wrench semantics", "raw frame/sign/compensation interpretation", "raw_wrench_frame/sign/delay null; compensation unresolved", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "record controller compensation state; validation may resolve frame/sign"),
        ("S14", "rehab frame", "T_B_R / rehab axes calibration", "axes null; reviewed=false", "MISSING", False, True, True, "calibrate and freeze transform for geometry/full task direction"),
        ("S15", "P0 pose", "exact existing stationary pose and its eligibility", "P0 role defined; joint/TCP coordinates null", "DEFINED_BUT_NOT_REVIEWED", True, True, True, "freeze current pose hash and approve no-positioning procedure"),
        ("S16", "P1/P2 poses", "exact alternative orientations and positioning safety", "coordinates null; positioning unauthorized", "MISSING", False, True, True, "separate motion, path, collision and pose approval"),
        ("S17", "human safeguards", "human anthropometry/subject safety configuration", "no human permitted", "NOT_APPLICABLE", False, False, False, "do not introduce a human"),
        ("S18", "rehabilitation motion", "trajectory execution safety", "motion prohibited in this stage", "NOT_APPLICABLE", False, False, False, "readiness cannot authorize trajectory motion"),
    ]
    return [{
        "item_id": item_id, "category": category, "requirement": requirement,
        "current_evidence": evidence, "classification": classification,
        "mandatory_for_p0": p0, "mandatory_for_full_pose": full,
        "current_blocker": blocker, "resolution": resolution,
    } for item_id, category, requirement, evidence, classification, p0, full, blocker, resolution in rows]


def equipment_rows() -> list[dict[str, Any]]:
    rows = [
        ("E01", "calibrated bidirectional force gauge or load cell", "wrench", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E02", "current calibration certificate and uncertainty", "wrench", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E03", "hands-free rigid fixture or low-friction pulley", "wrench", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E04", "world/load-direction registration metrology", "wrench", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E05", "safe mounting and secondary load retention", "wrench", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E06", "known calibrated masses", "optional wrench method", "optional; no evidence", False, False, False, False, "NOT_APPLICABLE", False),
        ("E07", "rigid nonhuman loading attachment/phantom", "wrench", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E08", "rigid shank surrogate/cylinder", "geometry", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E09", "identified production strap/cuff", "geometry", "intended component only; no ID/dimensions", False, False, False, False, "MISSING", True),
        ("E10", "identified robot eyelet/hook and fixed fixture", "geometry", "intended component only; no local offset", False, False, False, False, "MISSING", True),
        ("E11", "repeatable surrogate jig and cuff placement landmarks", "geometry", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E12", "bed/setup reference markers and base-linked fiducials", "geometry", "frame definition exists; physical markers unconfirmed", False, False, False, False, "MISSING", True),
        ("E13", "calibrated 3-D digitizer/tracked pointer or calibrated multi-view camera", "geometry minimum viable method", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", True),
        ("E14", "calibrated ruler/caliper", "geometry supplemental", "NO_REPOSITORY_EVIDENCE", False, False, False, False, "MISSING", False),
        ("E15", "robot/setup registration artifacts", "geometry", "T_B_R is null", False, False, False, False, "MISSING", True),
        ("E16", "supervised E-stop and safe-load-release access", "shared safety", "required but no availability record", False, False, False, False, "MISSING", True),
        ("E17", "logging computer/storage and offline logger primitives", "data", "repository code exists; no static-protocol dry run", True, False, False, False, "DEFINED_BUT_NOT_REVIEWED", True),
    ]
    return [{
        "item_id": item_id, "equipment": equipment, "use": use,
        "availability": availability, "calibration_known": calibration,
        "traceable_magnitude_or_geometry": traceable, "direction_controllable": direction,
        "safe_mounting_confirmed": mounting, "classification": classification,
        "mandatory_blocker": blocker,
    } for item_id, equipment, use, availability, calibration, traceable, direction, mounting, classification, blocker in rows]


def checklist_rows() -> list[dict[str, Any]]:
    rows = [
        ("C01", "protocol", "wrench protocol SHA verified", True, True, "REVIEWED_AND_READY", WRENCH_PROTOCOL_SHA256, False, "[ ]"),
        ("C02", "protocol", "geometry protocol SHA verified", False, True, "REVIEWED_AND_READY", GEOMETRY_PROTOCOL_SHA256, False, "[ ]"),
        ("C03", "authorization", "site-specific static nonhuman procedure approved", True, True, "MISSING", "no approval record", True, "[ ]"),
        ("C04", "safety", "experiment safety config reviewed and checksum frozen", True, True, "DEFINED_BUT_NOT_REVIEWED", "reviewed=false", True, "[ ]"),
        ("C05", "robot", "identity/controller/tool/TCP/payload match reviewed record", True, True, "MISSING", "expected fields null", True, "[ ]"),
        ("C06", "pose", "P0 exact stationary pose approved without positioning", True, True, "DEFINED_BUT_NOT_REVIEWED", "P0 coordinates null", True, "[ ]"),
        ("C07", "pose", "P1/P2 exact poses and positioning procedure approved", False, True, "MISSING", "positioning unauthorized", True, "[ ]"),
        ("C08", "robot", "stable stationary operation state confirmed", True, True, "DEFINED_BUT_NOT_REVIEWED", "day-of evidence absent", True, "[ ]"),
        ("C09", "load", "calibrated load instrument and certificate verified", True, False, "MISSING", "no equipment record", True, "[ ]"),
        ("C10", "load", "hands-free fixture/direction registration/secondary retention inspected", True, False, "MISSING", "no equipment record", True, "[ ]"),
        ("C11", "load", "low/high N values and static force margins approved", True, False, "MISSING", "both values null", True, "[ ]"),
        ("C12", "threshold", "baseline/calibration-derived thresholds frozen before results", True, True, "MISSING", "thresholds null", True, "[ ]"),
        ("C13", "geometry", "rigid nonhuman shank surrogate and repeatable jig inspected", False, True, "MISSING", "no equipment record", True, "[ ]"),
        ("C14", "geometry", "production cuff/strap and robot eyelet IDs/placement frozen", False, True, "MISSING", "component identities absent", True, "[ ]"),
        ("C15", "geometry", "setup fiducials and calibrated 3-D measurement method ready", False, True, "MISSING", "no metrology record", True, "[ ]"),
        ("C16", "geometry", "T_B_R method, fiducials, convention and uncertainty frozen", False, True, "MISSING", "T_B_R null", True, "[ ]"),
        ("C17", "data", "host monotonic/query start/end/midpoint/Fx/Fy/Fz/state fields dry-run verified", True, False, "DEFINED_BUT_NOT_REVIEWED", "primitives exist; static dry run absent", True, "[ ]"),
        ("C18", "data", "pose/direction/repeat/load-level/PRE-LOAD-POST labels recorded", True, False, "MISSING", "no protocol-specific logger/state machine", True, "[ ]"),
        ("C19", "data", "active tool/TCP/config and external calibration metadata recorded", True, True, "DEFINED_BUT_NOT_REVIEWED", "metadata primitive exists; HMI active config unresolved", True, "[ ]"),
        ("C20", "data", "output directory, raw checksums, failure retention and storage verified", True, True, "DEFINED_BUT_NOT_REVIEWED", "logger primitives exist; protocol dry run absent", True, "[ ]"),
        ("C21", "safety", "E-stop access, supervisor, loader and logger roles confirmed", True, True, "MISSING", "no reviewed roles", True, "[ ]"),
        ("C22", "safety", "abort criteria, maximum dwell/session duration and load release frozen", True, True, "MISSING", "limits null", True, "[ ]"),
        ("C23", "safety", "cleanup/disconnect procedure rehearsed without load", True, True, "DEFINED_BUT_NOT_REVIEWED", "disconnect helper only", True, "[ ]"),
        ("C24", "boundary", "no human subject and no hand-applied formal load", True, True, "REVIEWED_AND_READY", "both protocols prohibit these", True, "[ ]"),
        ("C25", "boundary", "no rehabilitation motion; robot remains static", True, True, "REVIEWED_AND_READY", "protocol scope", True, "[ ]"),
        ("C26", "workflow", "PRE/LOAD/POST sequence and cell order manifest ready", True, False, "DEFINED_BUT_NOT_REVIEWED", "protocol defines windows/matrix; executable runner absent", True, "[ ]"),
        ("C27", "workflow", "wrench and geometry raw/result pipelines remain separate", True, True, "REVIEWED_AND_READY", "audit recommendation", True, "[ ]"),
        ("C28", "boundary", "readiness does not authorize human, motion or endpoint finalization", True, True, "REVIEWED_AND_READY", "frozen boundary", True, "[ ]"),
    ]
    return [{
        "check_id": check_id, "section": section, "check": check,
        "mandatory_for_p0": p0, "mandatory_for_full_pose_or_geometry": full,
        "current_status": status, "current_evidence": evidence,
        "blocking_if_fail": blocking, "day_of_execution_box": box,
    } for check_id, section, check, p0, full, status, evidence, blocking, box in rows]


def minimum_blockers() -> list[dict[str, str]]:
    return [
        {
            "id": "B1_SITE_SAFETY_AND_POSE",
            "blocking_item": "Complete site-specific safety/config review and freeze exact P0 eligibility: robot identity, tool/TCP/payload, workspace/joint/collision limits, stationary state, E-stop/operator roles, dwell/abort/unload/cleanup. P1/P2 require separate motion approval.",
        },
        {
            "id": "B2_TRACEABLE_LOAD_SYSTEM",
            "blocking_item": "Provide an identified, calibrated hands-free force gauge/load cell plus direction-controlled fixture, safe mounting/secondary retention, calibration certificate and uncertainty.",
        },
        {
            "id": "B3_LOAD_AND_THRESHOLD_FREEZE",
            "blocking_item": "Use independent baseline/calibration evidence and reviewed robot/fixture/instrument limits to freeze load magnitudes and all PASS thresholds before formal validation results; values remain null now.",
        },
        {
            "id": "B4_GEOMETRY_KIT_AND_FRAME",
            "blocking_item": "Provide and identify the rigid shank surrogate, production cuff/strap, robot eyelet, repeatable jig/fiducials and calibrated minimum-viable 3-D metrology; freeze exact configurations and T_B_R registration/uncertainty.",
        },
        {
            "id": "B5_STATIC_VALIDATION_LOGGER_DRY_RUN",
            "blocking_item": "Implement only the standalone protocol-specific labels/manifest layer over existing read-only acquisition, then pass an offline/no-load dry run for PRE/LOAD/POST, pose/direction/repeat/load IDs, tool/config/calibration metadata, invalid rows and immutable checksums.",
        },
    ]


def load_level_readiness() -> str:
    return """# Load-Level Readiness

## Current decision

`FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW` and `LOAD_LEVEL_BLOCKER` remain active. Both `L1_REVIEWED_LOW` and `L2_REVIEWED_HIGH` are null. No N value can be produced from current repository evidence.

## Frozen future determination pathway

A load may be selected only after all six constraints are jointly available:

1. it exceeds the independently measured zero/noise floor by a prefrozen signal criterion;
2. it is below a site-reviewed static-force limit;
3. it is inside the calibrated instrument and fixture ranges with uncertainty/margin;
4. it is far below the most conservative robot/hardware protective limit;
5. the hands-free fixture reproduces magnitude and direction;
6. magnitude is independently traceable rather than inferred from the robot output.

Use the most conservative upper bound. The SDK wrench result cannot choose or increase a load. Hand push, estimated manual force and human-subject loading are prohibited formal evidence.

## Threshold evidence split

| Threshold/evidence | Can freeze before physical validation? | Dedicated calibration dataset required? |
|---|---|---|
| instrument range/accuracy/uncertainty | yes, from current certificate | no if certificate is current and setup-relevant |
| fixture rating and site-reviewed robot safety ceiling | yes, from reviewed engineering records | no |
| geometry metrology resolution and registration tolerance | yes, from certificate plus endpoint error budget | a metrology phantom check is still required |
| minimum separation/angular endpoint-error tolerance | yes, prospectively from geometry/error budget | no formal validation outcome may tune it |
| robot wrench zero/noise floor and drift | no numeric value exists now | yes: independent PRE/POST unloaded calibration dataset |
| steady-load acceptance/SNR and sign minimum | definition can be prefrozen; numeric gate needs baseline variability | yes |
| cross-axis leakage/pose consistency gate | engineering maximum can be prefrozen | fixture registration/calibration evidence is required; not validation outcomes |
| setup repeatability PASS gate | tolerance can be prefrozen | actual ten setups test the gate but must not set it |

Calibration data and formal validation data must have separate run IDs and roles. Formal validation results may test a threshold but may not define or relax it.
"""


def geometry_readiness() -> str:
    return """# Geometry Measurement Readiness

## Decision

The protocol definition is complete enough to specify what must be measured, but the physical setup is not execution-ready. There is no repository evidence of the production cuff/eyelet identities, rigid surrogate, jig, fiducials, calibrated metrology or `T_B_R` result.

## Minimum viable geometry measurement method

Use a rigid `NON_HUMAN_SHANK_SURROGATE`, the actual identified cuff/strap and robot eyelet, a repeatable jig, and a rigid fiducial frame. The minimum viable metrology is either:

- a calibrated 3-D digitizer/tracked pointer observing the eyelet, at least two taut free-span fiducials, cuff landmarks and at least three non-collinear setup/base reference points; or
- calibrated multi-view camera/photogrammetry with a validated scale/frame target and equivalent point/line uncertainty.

Advanced motion capture is not mandatory. A ruler/caliper may supplement local offsets only when its calibrated uncertainty is demonstrably adequate; by itself it normally does not establish a common 3-D base/setup transform.

Required products are: labelled raw point observations; eyelet offset `p_attach_TCP`; fitted strap exit/free-span line and residual; setup/surrogate/cuff placement IDs; `T_B_R` transform, convention, fiducials, residual and covariance; ten remove/reattach setup records with three within-setup repeats; angular uncertainty output. Robot probing is not required and remains unauthorized.

The surrogate must be rigid, dimensionally repeatable, stable in the jig, compatible with the real cuff and permit repeatable placement landmarks. It is a mechanical surrogate, not a physiological limb model.
"""


def data_readiness() -> str:
    return """# Data Acquisition Readiness

## Existing reusable primitives

| Required datum | Current code evidence | Readiness |
|---|---|---|
| host monotonic timestamp | `perf_counter_ns` and state timestamps exist | DEFINED_BUT_NOT_REVIEWED for this physical protocol |
| wrench query start/end/midpoint | adapter records start/end and midpoint | DEFINED_BUT_NOT_REVIEWED |
| raw Fx/Fy/Fz and validity | adapter/state/collector fields exist | DEFINED_BUT_NOT_REVIEWED; physical frame/sign remains the validation target |
| robot TCP/joint/state | RT state and snapshot fields exist | DEFINED_BUT_NOT_REVIEWED |
| tool/TCP/config metadata | adapter/collector metadata exists | DEFINED_BUT_NOT_REVIEWED; active HMI tool/workobject remains unverified |
| pose/direction/repeat/load condition IDs | no static-validation field/state machine | MISSING |
| PRE/LOAD/POST label | protocol text only | MISSING |
| external calibrated load reading/uncertainty | no integrated formal record | MISSING |
| failure retention/checksums | generic episode logger primitives exist | DEFINED_BUT_NOT_REVIEWED for this protocol |

Existing primitives are sufficient to avoid redesigning collection. They are not an executed static-validation logger and must not be relabelled formal evidence.

## Minimum future implementation change

Add a standalone, default-off static-validation logger/runner around the existing read-only adapter and `EpisodeLogger`; do not change control behavior. Add only protocol cell metadata (`pose_id`, `direction_id`, `load_level_id`, `repeat_id`, `window_label`, calibrated-load value/uncertainty, fixture/calibration IDs), frozen run-manifest SHA, raw checksums and fail-closed invalid reasons. It must not enable, move, calibrate sensors, invoke SafetyGuard stop, or choose loads.

Before physical execution, pass fake-adapter/offline and supervised no-load dry runs proving headers, counts, PRE/LOAD/POST transitions, exception retention, flush/fsync and cleanup. Host timing supports static window averaging only and does not validate dynamic synchronization.
"""


def blockers_md() -> str:
    lines = [
        "# Minimum Blocking Items",
        "",
        f"Current decision: `{DECISION}`.",
        "",
        "Only the following five consolidated items block the first formal nonhuman static physical validation:",
        "",
    ]
    for index, item in enumerate(minimum_blockers(), start=1):
        lines.append(f"{index}. **{item['id']}** — {item['blocking_item']}")
        lines.append("")
    lines.extend([
        f"Next action: `{NEXT_ACTION}`.",
        "",
        "Do not open another measurement-semantics audit. Resolve these items with equipment records, site review, calibration evidence and the minimal logger dry run; then rerun this readiness audit against the same frozen protocols.",
    ])
    return "\n".join(lines) + "\n"


def status_payload(safety: list[dict[str, Any]], equipment: list[dict[str, Any]], checklist: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        return {name: sum(row[field] == name for row in rows) for name in sorted(ALLOWED_CLASSIFICATIONS)}

    return {
        "stage_id": STAGE_ID,
        "audit_type": AUDIT_TYPE,
        "decision": DECISION,
        "question": "Are all prerequisites sufficiently defined to authorize the first non-human static physical validation?",
        "answer": False,
        "wrench_protocol": {"status": WRENCH_PROTOCOL_STATUS, "sha256": WRENCH_PROTOCOL_SHA256},
        "geometry_protocol": {"status": GEOMETRY_PROTOCOL_STATUS, "sha256": GEOMETRY_PROTOCOL_SHA256},
        "P0_STATIC_VALIDATION_EXECUTABLE": False,
        "P0_future_scope_after_blockers": "single-pose force frame/sign evidence without robot positioning; cannot establish pose invariance",
        "POSE_DEPENDENCE_VALIDATION_BLOCKED": True,
        "P1_P2_requirements": ["exact pose coordinates", "motion/path and collision review", "workspace/joint-limit review", "separate positioning authorization", "supervised abort/stop procedure"],
        "load_level_status": "LOAD_LEVEL_BLOCKER",
        "geometry_equipment_status": "NOT_READY",
        "data_acquisition_status": "PRIMITIVES_EXIST_PROTOCOL_SPECIFIC_LOGGER_NOT_READY",
        "session_recommendation": "SEPARATE_PHYSICAL_SESSIONS_PREFERRED; same bench day allowed only with separate authorization, run manifests, raw data, checksums and result pipelines",
        "minimum_blocking_items": minimum_blockers(),
        "readiness_counts": {
            "safety_and_config": counts(safety, "classification"),
            "equipment_and_calibration": counts(equipment, "classification"),
            "checklist": counts(checklist, "current_status"),
        },
        "mandatory_checklist_fail_count": sum(row["blocking_if_fail"] and row["current_status"] != "REVIEWED_AND_READY" for row in checklist),
        "next_action": NEXT_ACTION,
        "execution_authorized": False,
        "static_wrench_validation_executed": False,
        "geometry_validation_executed": False,
        "robot_connected": False,
        "robot_powered_or_enabled": False,
        "robot_motion_count": 0,
        "physical_load_count": 0,
        "human_data_count": 0,
        "rehabilitation_motion_count": 0,
        "pinn_run_count": 0,
        "bo_run_count": 0,
        "verified_wrench_frame": "NONE_PHYSICALLY_VERIFIED",
        "wrench_force_sign_verified": False,
        "task_direction_status": "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION",
        "primary_endpoint_finalized": False,
        "primary_endpoint_validated": False,
        "not_human_ready": True,
        "not_robot_approved": True,
    }


def report(status: dict[str, Any]) -> str:
    return f"""# Static Measurement Validation Execution Readiness Audit V1

## Formal decision

`{DECISION}`

**Answer:** no. The two authoritative protocols are frozen and internally ready as designs, but current safety/config review, physical equipment/calibration, load/threshold freeze, geometry metrology/frame registration and protocol-specific logging are not sufficient to authorize the first formal nonhuman static validation.

## P0 versus P1/P2

P0 could eventually provide limited single-pose force direction/sign evidence without positioning motion. Today `P0_STATIC_VALIDATION_EXECUTABLE=false`: its exact joint/TCP pose is null, site safety/tool/TCP/payload/limits are unreviewed, load hardware and levels are absent, and the static logger has no accepted dry run. Even after P0 becomes ready, `POSE_DEPENDENCE_VALIDATION_BLOCKED` remains until at least one separately authorized non-degenerate orientation is available; P0 alone cannot establish full world-frame pose invariance.

P1/P2 additionally require exact poses, reviewed workspace/joint/collision margins, a safe positioning path/procedure, separate motion authorization, stationary-state confirmation after positioning and supervised stop/abort handling. This audit authorizes none of those actions.

## Equipment and load

No repository evidence establishes availability/calibration of a bidirectional force gauge/load cell, hands-free direction fixture, secondary retention or force calibration certificate. Both load levels remain null and `LOAD_LEVEL_BLOCKER` applies. Hand push, estimated manual force and human loading are forbidden formal evidence.

Geometry likewise lacks identified production strap/eyelet hardware, rigid shank surrogate, repeatable jig/fiducials, calibrated 3-D metrology and `T_B_R`. A calibrated tracked pointer/3-D digitizer or calibrated multi-view camera is sufficient in principle; advanced motion capture is not mandatory.

## Data acquisition

The repository already has reusable read-only primitives for host monotonic timing, wrench query start/end/midpoint, Fx/Fy/Fz, TCP/joint/state and tool/payload metadata. It does not have an accepted static-protocol state machine recording pose/direction/repeat/load IDs and PRE/LOAD/POST labels with the external calibrated reading. The minimum future code change is a standalone default-off logger layer plus offline/no-load dry run, not a control change.

## Session decision

Prefer separate physical wrench and geometry sessions for the first validation. A same-day bench setup is acceptable only as two independently authorized sessions with separate manifests, raw data, checksums and result pipelines. A PASS in one branch cannot imply a PASS in the other.

## Static-only boundary

Any future authorization remains nonhuman, supervised, static, externally calibrated and time-limited with prefrozen abort conditions. It cannot extend to rehabilitation motion, human contact, dynamic endpoint validation or robot probing/positioning not separately approved.

## Endpoint and next action

`VERIFIED_WRENCH_FRAME=NONE_PHYSICALLY_VERIFIED`; `WRENCH_FORCE_SIGN_VERIFIED=false`; `TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION`; `PRIMARY_ENDPOINT_FINALIZED=false`; `PRIMARY_ENDPOINT_VALIDATED=false`.

Next action: `{NEXT_ACTION}`. Do not execute either validation and do not create another measurement-semantics protocol.
"""


def build() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory must be empty: {OUTPUT}")
    inputs = verify_sources()
    safety = safety_rows()
    equipment = equipment_rows()
    checklist = checklist_rows()
    for rows, field in ((safety, "classification"), (equipment, "classification"), (checklist, "current_status")):
        if any(row[field] not in ALLOWED_CLASSIFICATIONS for row in rows):
            raise RuntimeError("invalid readiness classification")
    status = status_payload(safety, equipment, checklist)

    atomic_csv(OUTPUT / "READINESS_CHECKLIST.csv", checklist)
    atomic_csv(OUTPUT / "SAFETY_AND_CONFIG_READINESS.csv", safety)
    atomic_csv(OUTPUT / "EQUIPMENT_AND_CALIBRATION_READINESS.csv", equipment)
    atomic_text(OUTPUT / "LOAD_LEVEL_READINESS.md", load_level_readiness())
    atomic_text(OUTPUT / "GEOMETRY_MEASUREMENT_READINESS.md", geometry_readiness())
    atomic_text(OUTPUT / "DATA_ACQUISITION_READINESS.md", data_readiness())
    atomic_text(OUTPUT / "MINIMUM_BLOCKING_ITEMS.md", blockers_md())
    atomic_json(OUTPUT / "FINAL_EXECUTION_READINESS_STATUS.json", status)
    atomic_text(OUTPUT / "STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_REPORT.md", report(status))

    artifact_sha256 = {
        path.name: sha256_file(path)
        for path in sorted(OUTPUT.iterdir())
        if path.is_file()
    }
    atomic_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID,
        "audit_type": AUDIT_TYPE,
        "decision": DECISION,
        "next_action": NEXT_ACTION,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "verified_input_count": len(inputs),
        "verified_inputs": inputs,
        "artifact_sha256_excluding_metadata_and_checksums": artifact_sha256,
        "minimum_blocking_item_count": len(minimum_blockers()),
        "execution_authorized": False,
        "physical_validation_executed": False,
        "robot_access_count": 0,
        "motion_command_count": 0,
        "physical_load_count": 0,
        "human_data_count": 0,
        "pinn_run_count": 0,
        "bo_run_count": 0,
        "hardware_control_safety_modified": False,
        "research_outcome_definition_modified": False,
        "authoritative_protocols_modified": False,
        "not_human_ready": True,
        "not_robot_approved": True,
    })
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.name}" for path in files) + "\n")
    print(json.dumps({
        "stage_id": STAGE_ID,
        "decision": DECISION,
        "minimum_blocking_items": len(minimum_blockers()),
        "execution_authorized": False,
        "next_action": NEXT_ACTION,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
