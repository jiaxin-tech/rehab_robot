"""Regression gates for offline wrench/task-direction resolution V1."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from collection.state import rpy_euler_xyz_rotation_matrix, rotate_vector, transform_wrench, transpose_rotation
from external_simulation.wrench_frame_and_task_direction_resolution_v1 import build_resolution as resolution


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1"


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


def test_protocol_is_frozen_before_resolution_results() -> None:
    protocol = load_json("WRENCH_RESOLUTION_PROTOCOL.json")
    assert sha256(OUT / "WRENCH_RESOLUTION_PROTOCOL.json") == resolution.FROZEN_PROTOCOL_SHA256
    assert protocol["protocol_id"] == resolution.PROTOCOL_ID
    assert protocol["protocol_frozen_before_resolution_results"] is True
    assert protocol["parent_endpoint_status"] == resolution.PARENT_STATUS
    assert protocol["decision_rules"]["api_request_frame_is_not_physical_verification"] is True
    assert protocol["decision_rules"]["rotation_math_pass_is_not_physical_frame_verification"] is True
    assert protocol["decision_rules"]["primary_endpoint_may_not_be_computed_or_validated"] is True


def test_all_23_inputs_and_parent_shas_are_pinned() -> None:
    verification = load_json("INPUT_VERIFICATION.json")
    assert verification["input_count"] == len(resolution.INPUT_SPECS) == 23
    assert verification["all_inputs_present_and_semantically_verified"] is True
    assert verification["new_physical_evidence_files_used"] == []
    for row in verification["inputs"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
        assert row["semantic_markers_pass"] is True
    assert sha256(ROOT / resolution.INPUT_SPECS[0]["path"]) == resolution.PARENT_PROTOCOL_SHA256
    assert sha256(ROOT / resolution.INPUT_SPECS[1]["path"]) == resolution.PARENT_ENDPOINT_SHA256


def test_reference_rom_and_theta_shank_remain_frozen() -> None:
    formal = json.loads((ROOT / "config/formal_experiment_manifest.json").read_text(encoding="utf-8"))
    assert sha256(ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv") == resolution.ACTIVE_REFERENCE_SHA256
    assert formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
    assert formal["theta_shank_definition"] == "q_hip - q_knee"
    assert formal["active_reference_sha256"] == resolution.ACTIVE_REFERENCE_SHA256


def test_sdk_version_signature_shape_and_units_are_audited_without_guessing() -> None:
    audit = read_text("ROKAE_GET_END_TORQUE_SEMANTICS_AUDIT.md")
    assert "xCoreSDK `0.7.0`" in audit
    assert "getEndTorque(ref_type, joint_torque_measured, external_torque_measured, cart_torque, cart_force, ec)" in audit
    assert "6/6/3/3" in audit
    for marker in ("joint measured torque", "external joint torque", "Cartesian torque", "Cartesian force"):
        assert marker in audit
    assert "N*m" in audit and "XYZ; N" in audit
    assert "Actual Windows native library loaded in this macOS audit: not loaded" in audit


def test_requested_world_is_distinct_from_verified_physical_frame() -> None:
    status = load_json("FINAL_WRENCH_TASK_DIRECTION_STATUS.json")
    audit = read_text("ROKAE_GET_END_TORQUE_SEMANTICS_AUDIT.md")
    assert status["REQUESTED_WRENCH_FRAME"] == "world"
    assert status["VERIFIED_WRENCH_FRAME"] == "NONE_PHYSICALLY_VERIFIED"
    assert status["WORLD_WRENCH_VERIFIED"] is False
    assert "The request and documented expression label" in audit
    assert "do not prove the physical load sign" in audit


def test_unknown_wrench_semantics_remain_unverified() -> None:
    status = load_json("FINAL_WRENCH_TASK_DIRECTION_STATUS.json")
    assert status["WRENCH_FORCE_SIGN_VERIFIED"] is False
    assert status["WRENCH_SIGN_STATUS"] == resolution.WRENCH_SIGN_STATUS
    assert status["WRENCH_COMPENSATION_STATUS"] == resolution.WRENCH_COMPENSATION_STATUS
    assert "PARTIALLY_DOCUMENTED" in status["WRENCH_REFERENCE_POINT_STATUS"]
    assert status["WRENCH_TIMESTAMP_STATUS"] == resolution.WRENCH_TIMESTAMP_STATUS
    audit = read_text("WRENCH_SIGN_AND_REFERENCE_POINT_AUDIT.md")
    assert "robot-on-environment or environment-on-robot" in audit
    assert "not proof of controller compensation" in audit
    assert "not a controller measurement timestamp" in audit


def test_frame_chain_records_current_unknown_tool_tcp_and_bed_values() -> None:
    audit = read_text("ROBOT_FRAME_CHAIN_AUDIT.md")
    for marker in ("baseFrame()", "toolset.end", "active HMI tool/workobject", "TCP-to-robot strap", "rehab bed x/z axes", "start TCP anchor"):
        assert marker in audit
    assert "both null, reviewed=false" in audit
    assert "no frozen value" in audit
    assert "active selection proof" in audit
    safety = json.loads((ROOT / "config/experiment_safety.json").read_text(encoding="utf-8"))
    rehab = json.loads((ROOT / "config/rehab_frame_config.json").read_text(encoding="utf-8"))
    assert safety["reviewed_tool_name"] is None and safety["tool_workpiece_reviewed"] is False
    assert rehab["rehab_x_axis_in_base"] is None and rehab["rehab_z_axis_in_base"] is None
    assert rehab["reviewed"] is False


def test_identity_and_canonical_x_rotation_use_active_column_convention() -> None:
    identity = rpy_euler_xyz_rotation_matrix((0.0, 0.0, 0.0))
    assert np.allclose(identity, np.eye(3), atol=1e-12, rtol=0.0)
    rx = rpy_euler_xyz_rotation_matrix((math.pi / 2.0, 0.0, 0.0))
    assert np.allclose(rotate_vector(rx, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0), atol=1e-12, rtol=0.0)
    assert np.allclose(rotate_vector(rx, (0.0, 0.0, 1.0)), (0.0, -1.0, 0.0), atol=1e-12, rtol=0.0)


def test_canonical_y_and_z_rotations_are_correct() -> None:
    ry = rpy_euler_xyz_rotation_matrix((0.0, math.pi / 2.0, 0.0))
    rz = rpy_euler_xyz_rotation_matrix((0.0, 0.0, math.pi / 2.0))
    assert np.allclose(rotate_vector(ry, (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0), atol=1e-12, rtol=0.0)
    assert np.allclose(rotate_vector(ry, (1.0, 0.0, 0.0)), (0.0, 0.0, -1.0), atol=1e-12, rtol=0.0)
    assert np.allclose(rotate_vector(rz, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0), atol=1e-12, rtol=0.0)
    assert np.allclose(rotate_vector(rz, (0.0, 1.0, 0.0)), (-1.0, 0.0, 0.0), atol=1e-12, rtol=0.0)


def test_world_to_base_uses_transpose_inverse_and_full_moment_shift() -> None:
    world_from_base = rpy_euler_xyz_rotation_matrix((0.0, 0.0, math.pi / 2.0))
    base_from_world = transpose_rotation(world_from_base)
    assert np.allclose(rotate_vector(base_from_world, (0.0, 1.0, 0.0)), (1.0, 0.0, 0.0), atol=1e-12, rtol=0.0)
    force, moment = transform_wrench((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), np.eye(3), (0.0, 1.0, 0.0))
    assert np.allclose(force, (1.0, 0.0, 0.0), atol=1e-12, rtol=0.0)
    assert np.allclose(moment, (0.0, 0.0, -1.0), atol=1e-12, rtol=0.0)


def test_rotation_artifact_passes_math_but_never_claims_physical_verification() -> None:
    rows = load_csv("WRENCH_ROTATION_UNIT_TEST_RESULTS.csv")
    assert len(rows) == 9
    assert all(row["passed"] == "True" for row in rows)
    assert all(row["physical_frame_verified"] == "False" for row in rows)
    assert {row["test_id"] for row in rows} >= {"IDENTITY_X", "RX90_Y_TO_Z", "RY90_Z_TO_X", "RZ90_X_TO_Y", "INVERSE_WORLD_TO_BASE", "FULL_MOMENT_REFERENCE_SHIFT"}
    status = load_json("FINAL_WRENCH_TASK_DIRECTION_STATUS.json")
    assert status["ROTATION_MATH_INTERNALLY_VERIFIED"] is True
    assert status["BASE_WRENCH_ROTATION_VERIFIED"] is False


def test_force_rotation_is_distinguished_from_full_wrench_transform() -> None:
    math_audit = read_text("WRENCH_ROTATION_MATH_AUDIT.md")
    sign_audit = read_text("WRENCH_SIGN_AND_REFERENCE_POINT_AUDIT.md")
    assert "F_base = R_base_from_world F_world" in math_audit
    assert "M_B = R M_A + r x (R F_A)" in math_audit
    assert "translating the reference origin does not change that 3D force vector" in sign_audit
    assert "six-dimensional wrench norm cannot use rotation-only torque" in sign_audit


def test_strap_geometry_separates_measured_configured_and_assumed() -> None:
    audit = read_text("STRAP_PULL_GEOMETRY_AUDIT.md")
    for marker in ("MEASURED_GEOMETRY", "CONFIGURED_GEOMETRY", "ASSUMED_GEOMETRY"):
        assert marker in audit
    assert "L2=0.30 m" in audit
    assert "not an ankle" in audit
    assert "actual line of action remains unknown" in audit
    assert "theta_shank=q_hip-q_knee" in audit


def test_all_five_task_direction_candidates_are_compared_without_outcomes() -> None:
    rows = load_csv("TASK_DIRECTION_CANDIDATE_COMPARISON.csv")
    assert len(rows) == 5
    assert {row["candidate_id"] for row in rows} == {"A_TCP_PATH_TANGENT", "B_ACTUAL_STRAP_PULL_LINE", "C_ENDPOINT_TO_HIP", "D_FIXED_BED_PULL_AXIS", "E_EQUIVALENT_PULL_POINT_TO_MODEL_HIP"}
    assert all(row["physical_interpretation"] and row["computability"] and row["required_measurements"] for row in rows)
    recommended = next(row for row in rows if row["candidate_id"] == "B_ACTUAL_STRAP_PULL_LINE")
    assert recommended["status"] == "PRIMARY_RECOMMENDATION_REQUIRES_GEOMETRIC_VALIDATION"
    assert recommended["actual_strap_relation"] == "direct"


def test_task_direction_has_exact_points_frame_and_geometric_sign() -> None:
    task = load_json("TASK_DIRECTION_FORMAL_DEFINITION.json")
    assert task["status"] == resolution.TASK_DIRECTION_STATUS
    assert task["target_definition"] == "ACTUAL_STRAP_PULL_LINE_OF_ACTION"
    assert task["coordinate_frame"] == "robot base B after independent frame-chain validation"
    assert task["point_A"]["id"] == "p_limb_attach_B(t)"
    assert task["point_B"]["id"] == "p_robot_attach_B(t)"
    assert task["point_A"]["current_status"] == "NOT_MEASURED"
    assert task["point_B"]["current_status"] == "NOT_MEASURED"
    assert task["construction"] == "d_task_B(t) = normalize(p_robot_attach_B(t) - p_limb_attach_B(t))"
    assert task["positive_geometric_direction"] == "limb attachment toward robot attachment (robotward pull direction)"
    assert task["currently_computable_as_validated_physical_direction"] is False
    assert task["selection_used_mechanical_outcomes"] is False


def test_tcp_tangent_is_not_silently_equated_to_interaction_direction() -> None:
    task = load_json("TASK_DIRECTION_FORMAL_DEFINITION.json")
    candidates = {row["candidate_id"]: row for row in load_csv("TASK_DIRECTION_CANDIDATE_COMPARISON.csv")}
    assert task["commanded_tcp_tangent_equivalent"] is False
    assert "only after geometry demonstrates alignment" in task["equivalence_condition"]
    assert "not equivalent unless geometry proves" in candidates["A_TCP_PATH_TANGENT"]["actual_strap_relation"]


def test_task_force_positive_meaning_remains_null_until_wrench_sign_validation() -> None:
    task = load_json("TASK_DIRECTION_FORMAL_DEFINITION.json")
    status = load_json("FINAL_WRENCH_TASK_DIRECTION_STATUS.json")
    assert task["positive_force_meaning"] is None
    assert "UNRESOLVED" in task["positive_force_status"]
    assert status["TASK_DIRECTION_FORCE_SIGN_MEANING"] is None
    assert status["WRENCH_FORCE_SIGN_VERIFIED"] is False


def test_future_static_geometry_plan_uses_no_human_motion_or_invented_tolerance() -> None:
    plan = read_text("FUTURE_STATIC_GEOMETRY_VALIDATION_PLAN.md")
    for marker in ("robot-side strap eye", "limb-side attachment", "rigid rehabilitation fixture/phantom", "bed-plane", "removal/reinstallation", "angular uncertainty"):
        assert marker in plan
    assert "No rehabilitation motion" in plan
    assert "No numeric tolerance is invented" in plan
    assert "Do not substitute TCP" in plan


def test_future_static_wrench_plan_requires_nonhuman_known_sign_reversal() -> None:
    plan = read_text("FUTURE_STATIC_WRENCH_VALIDATION_PLAN.md")
    for marker in ("stationary, non-human fixture", "positive and negative directions", "three non-collinear/orthogonal axes", "multiple non-degenerate", "known lever arms", "effective source-update cadence"):
        assert marker in plan
    assert "No human supplies the test load" in plan
    assert "must not set `BASE_WRENCH_ROTATION_VERIFIED=true`" in plan
    assert "does not itself approve connection" in plan


def test_readiness_is_partial_and_primary_endpoint_was_not_evaluated() -> None:
    status = load_json("FINAL_WRENCH_TASK_DIRECTION_STATUS.json")
    metadata = load_json("metadata.json")
    assert status["readiness_decision"] == resolution.FORMAL_STATUS
    assert status["PRIMARY_ENDPOINT_VALIDATED"] is False
    assert status["PRIMARY_ENDPOINT_FINALIZED"] is False
    assert status["PRIMARY_ENDPOINT_VALUE_COMPUTED"] is False
    assert metadata["primary_endpoint_value_computed"] is False
    assert metadata["pinn_training_count"] == 0
    assert metadata["bo_run_count"] == 0


def test_stage_performed_no_hardware_or_human_action() -> None:
    audit = load_json("HARDWARE_ACCESS_AUDIT.json")
    metadata = load_json("metadata.json")
    assert audit["offline_source_and_math_audit_only"] is True
    assert audit["robot_adapter_constructed"] is False
    assert audit["robot_connected"] is False
    assert audit["power_or_enable_count"] == 0
    assert audit["motion_command_count"] == 0
    assert audit["calibration_call_count"] == 0
    assert audit["human_loading_count"] == 0
    assert metadata["hardware_control_safety_modified"] is False
    assert metadata["not_human_ready"] is True
    assert metadata["not_robot_approved"] is True


def test_builder_imports_only_pure_collection_math_not_hardware_control_or_safety() -> None:
    tree = ast.parse(Path(resolution.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "collection.state" in imported
    assert not any(name.startswith(("hardware", "control", "safety")) for name in imported)


def test_next_stage_is_static_protocol_and_is_not_executed() -> None:
    status = load_json("FINAL_WRENCH_TASK_DIRECTION_STATUS.json")
    report = read_text("WRENCH_FRAME_AND_TASK_DIRECTION_RESOLUTION_REPORT.md")
    assert status["next_stage"] == "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1"
    assert status["next_stage_executed"] is False
    assert resolution.NEXT_STAGE in report
    assert "It was not executed" in report


def test_checksums_cover_all_outputs_except_checksum_manifest() -> None:
    recorded: dict[str, str] = {}
    for line in read_text("checksums.sha256").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    files = {str(path.relative_to(OUT)) for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    assert set(recorded) == files
    assert all(sha256(OUT / name) == digest for name, digest in recorded.items())
