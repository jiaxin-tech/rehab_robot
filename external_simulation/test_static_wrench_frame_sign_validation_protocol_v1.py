"""Regression gates for the static wrench validation protocol design."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from external_simulation.static_wrench_frame_sign_validation_protocol_v1 import build_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1"


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


def test_protocol_was_frozen_before_any_physical_result() -> None:
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    assert sha256(OUT / "STATIC_WRENCH_VALIDATION_PROTOCOL.json") == protocol.FROZEN_PROTOCOL_SHA256
    assert frozen["protocol_id"] == protocol.PROTOCOL_ID
    assert frozen["formal_status"] == protocol.FORMAL_STATUS
    assert frozen["protocol_frozen_before_any_physical_result"] is True
    assert frozen["protocol_scope"] == "FORCE_ONLY_FX_FY_FZ_STATIC_NON_HUMAN"
    assert frozen["moment_status"] == "NOT_FULLY_VALIDATED"


def test_all_18_inputs_and_parent_status_are_pinned() -> None:
    verification = load_json("INPUT_VERIFICATION.json")
    assert verification["input_count"] == len(protocol.INPUT_SPECS) == 18
    assert verification["all_inputs_present_and_semantically_verified"] is True
    assert verification["physical_result_files_read"] == []
    for row in verification["inputs"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
        assert row["semantic_markers_pass"] is True
    assert sha256(ROOT / protocol.INPUT_SPECS[0]["path"]) == protocol.PARENT_PROTOCOL_SHA256
    assert sha256(ROOT / protocol.INPUT_SPECS[1]["path"]) == protocol.PARENT_STATUS_SHA256


def test_five_hypotheses_are_exactly_frozen_and_not_posthoc_mutable() -> None:
    hypotheses = load_json("VALIDATION_HYPOTHESES.json")
    assert [row["id"] for row in hypotheses["hypotheses"]] == ["H1", "H2", "H3", "H4", "H5"]
    statements = {row["id"]: row["statement"] for row in hypotheses["hypotheses"]}
    assert "known static physical force direction" in statements["H1"]
    assert "sign reversal" in statements["H2"]
    assert "different tool orientations" in statements["H3"]
    assert "Zero-load baseline" in statements["H4"]
    assert "do not validate controller-source timing" in statements["H5"]
    assert all(row["primary"] is True for row in hypotheses["hypotheses"])
    assert hypotheses["post_result_hypothesis_modification_allowed"] is False


def test_force_only_scope_does_not_claim_moment_validation() -> None:
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    schema = load_json("FUTURE_RESULT_SCHEMA.json")
    metadata = load_json("metadata.json")
    assert frozen["moment_status"] == "NOT_FULLY_VALIDATED"
    assert schema["moment_remains_not_fully_validated"] is True
    assert metadata["moment_fully_validated"] is False
    assert schema["properties"]["update_eligibility"]["properties"]["moment_validated"]["const"] is False


def test_pose_plan_is_small_prefrozen_and_contains_no_invented_pose_coordinates() -> None:
    plan = load_json("STATIC_POSE_PLAN.json")
    assert plan["planned_pose_count"] == 3
    assert plan["pose_dependence_minimum_distinct_orientations"] == 2
    assert [row["pose_id"] for row in plan["poses"]] == ["P0_CURRENT_SAFE_STATIONARY", "P1_CONDITIONAL_ORIENTATION_A", "P2_CONDITIONAL_ORIENTATION_B"]
    assert all(row["joint_position_rad"] is None and row["tcp_pose_base_m_rad"] is None for row in plan["poses"])
    assert all(row["positioning_authorized_by_protocol"] is False for row in plan["poses"])
    assert plan["post_result_pose_selection_allowed"] is False
    for marker in ("not near joint limit", "not near singularity", "not near workspace boundary"):
        assert marker in plan["pose_exclusion_principles"]


def test_single_pose_fails_closed_on_pose_dependence_and_full_world_frame() -> None:
    only = load_json("STATIC_POSE_PLAN.json")["if_only_P0_available"]
    assert only["H1_H2_single_pose_may_be_assessed"] is True
    assert only["H3_status"] == "POSE_DEPENDENCE_NOT_YET_VALIDATED"
    assert only["full_world_frame_decision_allowed"] is False
    assert "full world-frame validation is forbidden" in read_text("POSE_CONSISTENCY_PLAN.md")


def test_direction_matrix_contains_all_36_prefrozen_cells() -> None:
    rows = load_csv("LOAD_DIRECTION_MATRIX.csv")
    assert len(rows) == 3 * 6 * 2 == 36
    assert {row["pose_id"] for row in rows} == {"P0_CURRENT_SAFE_STATIONARY", "P1_CONDITIONAL_ORIENTATION_A", "P2_CONDITIONAL_ORIENTATION_B"}
    assert {row["direction_id"] for row in rows} == {"+WORLD_X", "-WORLD_X", "+WORLD_Y", "-WORLD_Y", "+WORLD_Z", "-WORLD_Z"}
    assert {row["load_level_id"] for row in rows} == {"L1_REVIEWED_LOW", "L2_REVIEWED_HIGH"}
    assert all(row["repetitions"] == "5" for row in rows)
    assert all(row["queries_per_pre_load_post_window"] == "100" for row in rows)
    assert all(row["posthoc_drop_or_direction_change_allowed"] == "False" for row in rows)


def test_force_magnitudes_and_safety_thresholds_are_not_invented() -> None:
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    rows = load_csv("LOAD_DIRECTION_MATRIX.csv")
    assert frozen["force_magnitude_status"] == "FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW"
    assert frozen["force_magnitudes_n"] == {"L1_REVIEWED_LOW": None, "L2_REVIEWED_HIGH": None}
    assert all(row["force_magnitude_n"] == "" for row in rows)
    assert all(value is None for value in frozen["thresholds"].values())
    assert frozen["threshold_status"] == "THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE"
    assert load_json("metadata.json")["force_magnitudes_frozen"] is False


def test_known_load_equipment_is_calibrated_nonhuman_and_world_registered() -> None:
    plan = read_text("KNOWN_LOAD_APPLICATION_PLAN.md")
    for marker in ("calibrated bidirectional force gauge", "rigid fixture", "calibrated masses", "independent metrology", "non-human rigid", "calibration records"):
        assert marker in plan
    assert "Operator hand pushing and human/subject loading are prohibited" in plan
    assert "controller/TCP orientation alone cannot define a physical world load" in plan
    assert "SDK output cannot be used to choose or increase" in plan


def test_unavailable_axis_is_not_posthoc_replaced_or_deleted() -> None:
    rows = load_csv("LOAD_DIRECTION_MATRIX.csv")
    assert all(row["if_direction_not_safe_or_reliable"] == "NOT_EXECUTED_AND_AXIS_REMAINS_UNVALIDATED" for row in rows)
    assert all("same registered world vector" in row["same_direction_alternative"] for row in rows)
    plan = read_text("KNOWN_LOAD_APPLICATION_PLAN.md")
    assert "do not substitute a different direction or delete it after seeing results" in plan


def test_every_repetition_has_pre_load_and_post_load_zero_windows() -> None:
    plan = read_text("ZERO_BIAS_DRIFT_PLAN.md")
    assert "every independent load repetition" in plan
    for marker in ("PRE_LOAD_ZERO", "LOAD", "POST_LOAD_ZERO"):
        assert marker in plan
    for marker in ("mean", "median", "SD", "post-minus-pre drift", "between-pose zero difference"):
        assert marker in plan
    assert "Preserve raw values" in plan


def test_zero_contrast_is_prefrozen_without_claiming_production_compensation() -> None:
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    plan = read_text("ZERO_BIAS_DRIFT_PLAN.md")
    assert frozen["primary_response_contrast"].startswith("DeltaF = mean(F_load)")
    assert "DeltaF = mean(F_load) - 0.5*(mean(F_pre)+mean(F_post))" in plan
    assert "not a conclusion that simple zero subtraction" in plan
    assert "cannot be labelled sensor error" in plan


def test_sign_metrics_require_paired_reversal_and_one_global_convention() -> None:
    metrics = load_json("SIGN_VALIDATION_METRICS.json")
    assert metrics["paired_axes"] == [["+WORLD_X", "-WORLD_X"], ["+WORLD_Y", "-WORLD_Y"], ["+WORLD_Z", "-WORLD_Z"]]
    assert "< 0" in metrics["raw_axis_sign_reversal"]
    assert set(metrics["global_sign_candidates"]) == {"SAME_DIRECTION", "OPPOSITE_DIRECTION"}
    assert "one global convention" in metrics["sign_rule"]
    assert "all required + / - pairs reverse" in metrics["SIGN_REVERSAL_PASS"]
    assert metrics["magnitude_similarity_cannot_replace_sign"] is True
    assert metrics["post_result_flip_or_threshold_change_allowed"] is False


def test_direction_angles_consider_same_and_opposite_before_sign_resolution() -> None:
    metrics = load_json("FRAME_DIRECTION_VALIDATION_METRICS.json")
    assert "dot(u_reported, d_applied_world)" in metrics["same_direction_angle_rad"]
    assert "dot(u_reported, -d_applied_world)" in metrics["opposite_direction_angle_rad"]
    assert "report both angles before sign resolution" in metrics["sign_handling"]
    assert "single global sign convention" in metrics["sign_handling"]


def test_cross_axis_leakage_formula_and_threshold_remain_explicit() -> None:
    metrics = load_json("FRAME_DIRECTION_VALIDATION_METRICS.json")
    assert "DeltaF_world - dot" in metrics["orthogonal_vector"]
    assert "norm(F_orth) / max(abs(dot" in metrics["CROSS_AXIS_LEAKAGE"]
    assert metrics["thresholds"]["cross_axis_leakage_max"] is None
    assert metrics["threshold_status"] == "THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE"


def test_pose_consistency_uses_same_world_direction_not_tcp_orientation() -> None:
    plan = read_text("POSE_CONSISTENCY_PLAN.md")
    assert "at least two separately approved non-degenerate tool orientations" in plan
    assert "without redefining axes from TCP orientation" in plan
    assert "world-consistent" in plan
    assert "tool-following response will rotate" in plan
    assert "P1/P2 contain no coordinates" in plan


def test_mathematical_rotation_is_not_relabelled_physical_validation() -> None:
    plan = read_text("POSE_CONSISTENCY_PLAN.md")
    metadata = load_json("metadata.json")
    assert "9/9 canonical cases remain `MATHEMATICAL_TRANSFORM_VERIFIED` only" in plan
    assert "physical world-axis registration" in plan
    assert metadata["base_wrench_rotation_verified"] is False


def test_compensation_and_setup_state_are_required_before_interpretation() -> None:
    gate = load_json("SAFETY_PRECONDITIONS.json")
    lookup = {row["check_id"]: row for row in gate["checks"]}
    assert lookup["ROBOT_IDENTITY_TOOL_PAYLOAD_TCP_VERIFIED"]["current_pass"] is False
    assert lookup["COMPENSATION_STATE_RESOLVED"]["current_pass"] is False
    assert "compensation state" in lookup["COMPENSATION_STATE_RESOLVED"]["requirement"]
    schema = load_json("FUTURE_RESULT_SCHEMA.json")
    assert "active tool/payload/TCP, compensation state" in schema["properties"]["setup"]["description"]


def test_host_sampling_is_static_only_and_does_not_claim_source_sync() -> None:
    plan = read_text("SAMPLING_AND_TIMESTAMP_PLAN.md")
    for marker in ("HOST_MONOTONIC_PERF_COUNTER_NS", "query start", "query end", "midpoint", "query duration", "`100` valid host queries", "exactly `5`"):
        assert marker in plan
    assert "observed query rate/source-update behavior must be reported rather than assumed" in plan
    assert "STATIC_FRAME_VALIDATION != DYNAMIC_SYNCHRONIZATION_VALIDATION" in plan
    assert "No result may claim transport delay" in plan


def test_repeat_count_cannot_be_extended_after_results() -> None:
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    assert frozen["repetitions_per_pose_direction_level"] == protocol.REPETITIONS_PER_CELL == 5
    assert frozen["host_queries_per_pre_load_post_window"] == protocol.HOST_QUERIES_PER_WINDOW == 100
    assert "change_hypothesis_direction_or_repetitions_after_result" in frozen["forbidden_operations"]


def test_magnitude_linearity_is_secondary_not_required_for_frame_gate() -> None:
    metrics = load_json("FRAME_DIRECTION_VALIDATION_METRICS.json")
    plan = read_text("KNOWN_LOAD_APPLICATION_PLAN.md")
    assert "if two approved known levels" in metrics["secondary_magnitude_linearity"]
    assert metrics["magnitude_calibration_required_for_frame_gate"] is False
    assert "If only one level is approved, frame/sign may be assessed but magnitude linearity is unavailable" in plan


def test_future_decision_schema_is_fail_closed_and_partial_axes_cannot_pass_full() -> None:
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    schema = load_json("FUTURE_RESULT_SCHEMA.json")
    assert frozen["future_result_decisions"] == list(protocol.FUTURE_DECISIONS)
    assert schema["properties"]["decision"]["enum"] == list(protocol.FUTURE_DECISIONS)
    assert schema["partial_axis_success_cannot_produce_full_validation"] is True
    requirements = schema["full_validation_requirements"]
    assert "H1 PASS all axes" in requirements
    assert "H2 PASS all paired axes with one global sign" in requirements
    assert "H3 PASS at >=2 approved orientations" in requirements
    assert "no missing required cell" in requirements


def test_protocol_itself_cannot_update_any_wrench_verification_flag() -> None:
    metadata = load_json("metadata.json")
    frozen = load_json("STATIC_WRENCH_VALIDATION_PROTOCOL.json")
    assert metadata["requested_wrench_frame"] == "world"
    assert metadata["verified_wrench_frame"] == "NONE_PHYSICALLY_VERIFIED"
    assert metadata["wrench_force_sign_verified"] is False
    assert metadata["base_wrench_rotation_verified"] is False
    assert "this protocol changes no wrench flag" in frozen["update_conditions"]


def test_task_direction_and_endpoint_remain_independent_and_unready() -> None:
    metadata = load_json("metadata.json")
    schema = load_json("FUTURE_RESULT_SCHEMA.json")
    update = schema["properties"]["update_eligibility"]["properties"]
    assert metadata["task_direction_status"] == "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
    assert metadata["primary_endpoint_finalized"] is False
    assert metadata["primary_endpoint_validated"] is False
    assert update["task_direction_status"]["const"] == "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
    assert update["primary_endpoint_ready"]["const"] is False


def test_safety_preconditions_currently_block_all_physical_execution() -> None:
    gate = load_json("SAFETY_PRECONDITIONS.json")
    metadata = load_json("metadata.json")
    assert gate["current_authorization"] == "NOT_AUTHORIZED"
    assert gate["fail_closed"] is True
    assert gate["this_protocol_authorizes_robot_positioning"] is False
    assert gate["this_protocol_authorizes_physical_execution"] is False
    assert gate["on_any_failure"].startswith("DO_NOT_CONNECT_OR_APPLY_LOAD")
    assert metadata["future_physical_execution_authorized"] is False
    assert metadata["physical_validation_performed"] is False


def test_protocol_performed_no_robot_load_human_pinn_or_bo_action() -> None:
    audit = load_json("HARDWARE_ACCESS_AUDIT.json")
    metadata = load_json("metadata.json")
    assert audit["protocol_design_only"] is True
    assert audit["robot_constructed"] is False and audit["robot_connected"] is False
    assert audit["power_or_enable_count"] == 0
    assert audit["motion_or_position_command_count"] == 0
    assert audit["load_application_count"] == 0
    assert audit["human_or_operator_hand_loading_count"] == 0
    assert audit["endpoint_computation_count"] == 0
    assert metadata["pinn_run_count"] == 0 and metadata["bo_run_count"] == 0
    assert metadata["hardware_control_safety_modified"] is False


def test_builder_does_not_import_hardware_control_collection_or_safety() -> None:
    tree = ast.parse(Path(protocol.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith(("hardware", "control", "collection", "safety")) for name in imported)


def test_next_dependency_is_geometry_protocol_and_was_not_executed() -> None:
    plan = read_text("NEXT_DEPENDENCY_PLAN.md")
    metadata = load_json("metadata.json")
    assert metadata["next_dependency"] == "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1"
    assert metadata["next_dependency_executed"] is False
    assert "different reference equipment, uncertainties and scientific gates" in plan
    assert "This dependency was not executed" in plan


def test_checksums_cover_all_outputs_except_checksum_manifest() -> None:
    recorded: dict[str, str] = {}
    for line in read_text("checksums.sha256").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    files = {str(path.relative_to(OUT)) for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    assert set(recorded) == files
    assert all(sha256(OUT / name) == digest for name, digest in recorded.items())
