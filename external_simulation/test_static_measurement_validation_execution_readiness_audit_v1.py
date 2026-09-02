"""Regression tests for the static-measurement execution-readiness audit."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from external_simulation.static_measurement_validation_execution_readiness_audit_v1 import build_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/static_measurement_validation_execution_readiness_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_text(name: str) -> str:
    return (OUT / name).read_text(encoding="utf-8")


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_only_required_formal_artifacts_are_generated() -> None:
    expected = {
        "STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_REPORT.md",
        "READINESS_CHECKLIST.csv",
        "SAFETY_AND_CONFIG_READINESS.csv",
        "EQUIPMENT_AND_CALIBRATION_READINESS.csv",
        "LOAD_LEVEL_READINESS.md",
        "GEOMETRY_MEASUREMENT_READINESS.md",
        "DATA_ACQUISITION_READINESS.md",
        "MINIMUM_BLOCKING_ITEMS.md",
        "FINAL_EXECUTION_READINESS_STATUS.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert {path.name for path in OUT.iterdir() if path.is_file()} == expected


def test_formal_decision_is_not_ready_and_fail_closed() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["decision"] == audit.DECISION == "STATIC_MEASUREMENT_VALIDATION_EXECUTION_NOT_READY"
    assert status["answer"] is False
    assert status["execution_authorized"] is False
    assert status["next_action"] == audit.NEXT_ACTION == "RESOLVE_MINIMUM_BLOCKING_ITEMS"


def test_both_authoritative_protocols_are_exactly_pinned_and_unchanged() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["wrench_protocol"] == {"status": audit.WRENCH_PROTOCOL_STATUS, "sha256": audit.WRENCH_PROTOCOL_SHA256}
    assert status["geometry_protocol"] == {"status": audit.GEOMETRY_PROTOCOL_STATUS, "sha256": audit.GEOMETRY_PROTOCOL_SHA256}
    assert sha256(ROOT / audit.SOURCE_SPECS[0]["path"]) == audit.WRENCH_PROTOCOL_SHA256
    assert sha256(ROOT / audit.SOURCE_SPECS[6]["path"]) == audit.GEOMETRY_PROTOCOL_SHA256
    metadata = load_json("metadata.json")
    assert metadata["authoritative_protocols_modified"] is False
    assert metadata["research_outcome_definition_modified"] is False


def test_all_20_inputs_are_semantically_and_byte_verified() -> None:
    metadata = load_json("metadata.json")
    assert metadata["verified_input_count"] == len(audit.SOURCE_SPECS) == 20
    assert len(metadata["verified_inputs"]) == 20
    for row in metadata["verified_inputs"]:
        assert row["semantic_markers_pass"] is True
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_safety_config_is_not_misclassified_as_ready() -> None:
    rows = {row["item_id"]: row for row in load_csv("SAFETY_AND_CONFIG_READINESS.csv")}
    assert rows["S02"]["classification"] == "DEFINED_BUT_NOT_REVIEWED"
    assert rows["S03"]["classification"] == "MISSING"
    assert rows["S04"]["classification"] == "MISSING"
    assert rows["S05"]["classification"] == "MISSING"
    assert rows["S02"]["current_evidence"] == "reviewed=false"
    assert rows["S05"]["current_evidence"] == "max_force_n and max_torque_nm null"


def test_all_readiness_classifications_are_from_frozen_vocabulary() -> None:
    for filename, field in (
        ("SAFETY_AND_CONFIG_READINESS.csv", "classification"),
        ("EQUIPMENT_AND_CALIBRATION_READINESS.csv", "classification"),
        ("READINESS_CHECKLIST.csv", "current_status"),
    ):
        assert {row[field] for row in load_csv(filename)} <= audit.ALLOWED_CLASSIFICATIONS


def test_not_applicable_items_do_not_hide_static_blockers() -> None:
    rows = {row["item_id"]: row for row in load_csv("SAFETY_AND_CONFIG_READINESS.csv")}
    assert rows["S17"]["classification"] == "NOT_APPLICABLE"
    assert rows["S18"]["classification"] == "NOT_APPLICABLE"
    assert rows["S17"]["current_blocker"] == "False"
    assert rows["S18"]["current_blocker"] == "False"
    assert rows["S01"]["current_blocker"] == "True"


def test_p0_is_not_currently_executable_even_without_positioning() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["P0_STATIC_VALIDATION_EXECUTABLE"] is False
    assert "without robot positioning" in status["P0_future_scope_after_blockers"]
    report = read_text("STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_REPORT.md")
    assert "exact joint/TCP pose is null" in report
    assert "P0 alone cannot establish full world-frame pose invariance" in report


def test_pose_dependence_remains_blocked_and_p1_p2_need_motion_review() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["POSE_DEPENDENCE_VALIDATION_BLOCKED"] is True
    assert set(status["P1_P2_requirements"]) == {
        "exact pose coordinates", "motion/path and collision review",
        "workspace/joint-limit review", "separate positioning authorization",
        "supervised abort/stop procedure",
    }
    row = next(row for row in load_csv("SAFETY_AND_CONFIG_READINESS.csv") if row["item_id"] == "S16")
    assert row["mandatory_for_p0"] == "False"
    assert row["mandatory_for_full_pose"] == "True"


def test_no_force_equipment_is_assumed_available_or_calibrated() -> None:
    rows = [row for row in load_csv("EQUIPMENT_AND_CALIBRATION_READINESS.csv") if row["use"] == "wrench"]
    assert len(rows) >= 6
    assert all(row["availability"] == "NO_REPOSITORY_EVIDENCE" for row in rows)
    assert all(row["calibration_known"] == "False" for row in rows)
    assert all(row["safe_mounting_confirmed"] == "False" for row in rows)
    assert all(row["classification"] == "MISSING" for row in rows)


def test_optional_calibrated_mass_is_not_a_hidden_mandatory_method() -> None:
    row = next(row for row in load_csv("EQUIPMENT_AND_CALIBRATION_READINESS.csv") if row["item_id"] == "E06")
    assert row["classification"] == "NOT_APPLICABLE"
    assert row["mandatory_blocker"] == "False"


def test_load_levels_remain_null_and_pathway_uses_six_constraints() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    text = read_text("LOAD_LEVEL_READINESS.md")
    assert status["load_level_status"] == "LOAD_LEVEL_BLOCKER"
    assert "Both `L1_REVIEWED_LOW` and `L2_REVIEWED_HIGH` are null" in text
    for number in range(1, 7):
        assert f"{number}." in text
    assert "SDK wrench result cannot choose or increase a load" in text
    assert "Hand push, estimated manual force and human-subject loading are prohibited" in text


def test_threshold_evidence_is_split_from_formal_validation_outcomes() -> None:
    text = read_text("LOAD_LEVEL_READINESS.md")
    for marker in (
        "instrument range/accuracy/uncertainty",
        "robot wrench zero/noise floor and drift",
        "geometry metrology resolution",
        "cross-axis leakage/pose consistency gate",
        "setup repeatability PASS gate",
    ):
        assert marker in text
    assert "Formal validation results may test a threshold but may not define or relax it" in text


def test_geometry_equipment_is_not_ready_and_does_not_require_advanced_mocap() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    text = read_text("GEOMETRY_MEASUREMENT_READINESS.md")
    assert status["geometry_equipment_status"] == "NOT_READY"
    assert "Advanced motion capture is not mandatory" in text
    assert "calibrated 3-D digitizer/tracked pointer" in text
    assert "calibrated multi-view camera/photogrammetry" in text
    assert "ruler/caliper" in text


def test_minimum_geometry_method_measures_line_frame_and_repeatability() -> None:
    text = read_text("GEOMETRY_MEASUREMENT_READINESS.md")
    for marker in (
        "eyelet offset `p_attach_TCP`",
        "fitted strap exit/free-span line",
        "`T_B_R` transform",
        "ten remove/reattach setup records",
        "angular uncertainty output",
    ):
        assert marker in text
    assert "Robot probing is not required and remains unauthorized" in text


def test_nonhuman_surrogate_is_mechanical_not_physiological() -> None:
    text = read_text("GEOMETRY_MEASUREMENT_READINESS.md")
    assert "`NON_HUMAN_SHANK_SURROGATE`" in text
    for marker in ("rigid", "dimensionally repeatable", "stable in the jig", "compatible with the real cuff"):
        assert marker in text
    assert "not a physiological limb model" in text


def test_robot_to_rehab_frame_requires_noncollinear_registered_points() -> None:
    metadata = load_json("metadata.json")
    input_lookup = {row["input_id"]: row for row in metadata["verified_inputs"]}
    assert "FRAME_CALIBRATION_PLAN" in input_lookup
    plan = (ROOT / input_lookup["FRAME_CALIBRATION_PLAN"]["path"]).read_text(encoding="utf-8")
    assert "at least three non-collinear" in plan
    assert "registration residuals, leave-one-out error and transform covariance" in plan
    assert "Robot TCP probing is optional only after independent robot safety authorization" in plan


def test_existing_acquisition_primitives_are_recognized_but_not_overclaimed() -> None:
    text = read_text("DATA_ACQUISITION_READINESS.md")
    for marker in (
        "host monotonic timestamp",
        "wrench query start/end/midpoint",
        "raw Fx/Fy/Fz and validity",
        "robot TCP/joint/state",
        "tool/TCP/config metadata",
    ):
        assert marker in text
    assert "DEFINED_BUT_NOT_REVIEWED" in text
    assert "must not be relabelled formal evidence" in text


def test_protocol_specific_labels_and_external_load_record_are_missing() -> None:
    text = read_text("DATA_ACQUISITION_READINESS.md")
    assert "pose/direction/repeat/load condition IDs" in text
    assert "PRE/LOAD/POST label" in text
    assert "external calibrated load reading/uncertainty" in text
    assert text.count("| MISSING |") >= 3
    assert load_json("FINAL_EXECUTION_READINESS_STATUS.json")["data_acquisition_status"] == "PRIMITIVES_EXIST_PROTOCOL_SPECIFIC_LOGGER_NOT_READY"


def test_minimum_future_logger_change_does_not_modify_control() -> None:
    text = read_text("DATA_ACQUISITION_READINESS.md")
    assert "standalone, default-off static-validation logger/runner" in text
    assert "do not change control behavior" in text
    for marker in ("`pose_id`", "`direction_id`", "`load_level_id`", "`repeat_id`", "`window_label`"):
        assert marker in text
    assert "must not enable, move, calibrate sensors, invoke SafetyGuard stop, or choose loads" in text


def test_day_of_execution_checklist_is_single_and_usable() -> None:
    rows = load_csv("READINESS_CHECKLIST.csv")
    assert len(rows) == 28
    assert [row["check_id"] for row in rows] == [f"C{index:02d}" for index in range(1, 29)]
    assert all(row["day_of_execution_box"] == "[ ]" for row in rows)
    assert {row["section"] for row in rows} >= {"protocol", "safety", "robot", "pose", "load", "geometry", "data", "workflow", "boundary"}


def test_any_current_mandatory_failure_blocks_authorization() -> None:
    rows = load_csv("READINESS_CHECKLIST.csv")
    computed = sum(
        row["blocking_if_fail"] == "True" and row["current_status"] != "REVIEWED_AND_READY"
        for row in rows
    )
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert computed == status["mandatory_checklist_fail_count"] == 22
    assert computed > 0
    assert status["execution_authorized"] is False


def test_minimum_blocker_list_is_short_and_exactly_five_consolidated_items() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    blockers = status["minimum_blocking_items"]
    assert len(blockers) == 5
    assert [row["id"] for row in blockers] == [
        "B1_SITE_SAFETY_AND_POSE",
        "B2_TRACEABLE_LOAD_SYSTEM",
        "B3_LOAD_AND_THRESHOLD_FREEZE",
        "B4_GEOMETRY_KIT_AND_FRAME",
        "B5_STATIC_VALIDATION_LOGGER_DRY_RUN",
    ]
    assert read_text("MINIMUM_BLOCKING_ITEMS.md").count("**B") == 5


def test_next_action_is_resolution_not_another_large_audit() -> None:
    text = read_text("MINIMUM_BLOCKING_ITEMS.md")
    assert "`RESOLVE_MINIMUM_BLOCKING_ITEMS`" in text
    assert "Do not open another measurement-semantics audit" in text
    assert load_json("FINAL_EXECUTION_READINESS_STATUS.json")["next_action"] == "RESOLVE_MINIMUM_BLOCKING_ITEMS"


def test_wrench_and_geometry_sessions_remain_independent() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["session_recommendation"].startswith("SEPARATE_PHYSICAL_SESSIONS_PREFERRED")
    report = read_text("STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_REPORT.md")
    assert "separate manifests, raw data, checksums and result pipelines" in report
    assert "A PASS in one branch cannot imply a PASS in the other" in report


def test_static_only_boundary_does_not_expand_to_motion_or_human() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["robot_connected"] is False
    assert status["robot_powered_or_enabled"] is False
    assert status["robot_motion_count"] == 0
    assert status["physical_load_count"] == 0
    assert status["human_data_count"] == 0
    assert status["rehabilitation_motion_count"] == 0
    report = read_text("STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_REPORT.md")
    assert "cannot extend to rehabilitation motion, human contact" in report


def test_no_pinn_bo_or_validation_execution_occurred() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    metadata = load_json("metadata.json")
    assert status["pinn_run_count"] == 0 and status["bo_run_count"] == 0
    assert status["static_wrench_validation_executed"] is False
    assert status["geometry_validation_executed"] is False
    assert metadata["physical_validation_executed"] is False
    assert metadata["robot_access_count"] == 0


def test_wrench_geometry_and_endpoint_states_remain_unresolved() -> None:
    status = load_json("FINAL_EXECUTION_READINESS_STATUS.json")
    assert status["verified_wrench_frame"] == "NONE_PHYSICALLY_VERIFIED"
    assert status["wrench_force_sign_verified"] is False
    assert status["task_direction_status"] == "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
    assert status["primary_endpoint_finalized"] is False
    assert status["primary_endpoint_validated"] is False
    assert status["not_human_ready"] is True
    assert status["not_robot_approved"] is True


def test_builder_does_not_import_or_invoke_robot_control_or_safety_code() -> None:
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)
    assert not any(name.startswith(("hardware", "control", "collection", "safety")) for name in imported)
    assert not {"connect", "enable", "move", "calibrate_force_sensors", "stop"}.intersection(called_attributes)
    assert load_json("metadata.json")["hardware_control_safety_modified"] is False


def test_active_reference_sha_and_angle_semantics_are_preserved() -> None:
    spec = next(item for item in audit.SOURCE_SPECS if item["id"] == "ACTIVE_REFERENCE")
    assert spec["sha256"] == audit.ACTIVE_REFERENCE_SHA256
    assert sha256(ROOT / spec["path"]) == audit.ACTIVE_REFERENCE_SHA256
    assert "theta_shank = q_hip - q_knee" in (ROOT / spec["path"]).read_text(encoding="utf-8")


def test_metadata_hashes_all_primary_artifacts_before_metadata() -> None:
    metadata = load_json("metadata.json")
    hashes = metadata["artifact_sha256_excluding_metadata_and_checksums"]
    assert set(hashes) == {
        "READINESS_CHECKLIST.csv", "SAFETY_AND_CONFIG_READINESS.csv",
        "EQUIPMENT_AND_CALIBRATION_READINESS.csv", "LOAD_LEVEL_READINESS.md",
        "GEOMETRY_MEASUREMENT_READINESS.md", "DATA_ACQUISITION_READINESS.md",
        "MINIMUM_BLOCKING_ITEMS.md", "FINAL_EXECUTION_READINESS_STATUS.json",
        "STATIC_MEASUREMENT_VALIDATION_EXECUTION_READINESS_REPORT.md",
    }
    assert all(sha256(OUT / name) == digest for name, digest in hashes.items())


def test_checksums_cover_all_outputs_except_checksum_manifest() -> None:
    recorded: dict[str, str] = {}
    for line in read_text("checksums.sha256").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    files = {path.name for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    assert set(recorded) == files
    assert all(sha256(OUT / name) == digest for name, digest in recorded.items())
