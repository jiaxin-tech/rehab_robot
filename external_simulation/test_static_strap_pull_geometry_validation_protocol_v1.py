"""Regression gates for static strap pull geometry protocol design."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from external_simulation.static_strap_pull_geometry_validation_protocol_v1 import build_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/static_strap_pull_geometry_validation_protocol_v1"


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


def test_protocol_is_frozen_before_physical_results() -> None:
    frozen = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")
    assert sha256(OUT / "STRAP_GEOMETRY_VALIDATION_PROTOCOL.json") == protocol.FROZEN_PROTOCOL_SHA256
    assert frozen["stage_id"] == protocol.STAGE_ID
    assert frozen["formal_status"] == protocol.FORMAL_STATUS
    assert frozen["protocol_scope"] == "STATIC_NONHUMAN_GEOMETRY_PROTOCOL_DESIGN_ONLY"
    assert frozen["protocol_frozen_before_any_physical_result"] is True


def test_all_13_sources_are_byte_pinned_and_no_result_was_read() -> None:
    verification = load_json("INPUT_VERIFICATION.json")
    assert verification["input_count"] == len(protocol.INPUT_SPECS) == 13
    assert verification["all_inputs_present_and_semantically_verified"] is True
    assert verification["physical_geometry_result_files_read"] == []
    assert verification["scientific_endpoint_outcomes_read"] == []
    for row in verification["inputs"]:
        assert row["semantic_markers_pass"] is True
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_parent_wrench_and_task_definition_are_exactly_pinned() -> None:
    frozen = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")
    assert frozen["parent_wrench_protocol_sha256"] == protocol.PARENT_WRENCH_PROTOCOL_SHA256
    assert frozen["parent_task_direction_definition_sha256"] == protocol.PARENT_TASK_DEFINITION_SHA256
    assert sha256(ROOT / protocol.INPUT_SPECS[0]["path"]) == protocol.PARENT_WRENCH_PROTOCOL_SHA256
    assert sha256(ROOT / protocol.INPUT_SPECS[1]["path"]) == protocol.PARENT_TASK_DEFINITION_SHA256


def test_mechanical_topology_separates_hardware_model_and_assumptions() -> None:
    topology = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["topology"]
    assert "physical robot-side strap eyelet/hook/fixture load-transfer center" in topology["ordered_chain"]
    assert any("hardware-specific" in item for item in topology["PHYSICAL_HARDWARE_DEFINED"])
    assert any("L2=0.30 m" in item for item in topology["MODEL_EQUIVALENT"])
    assert any("TCP origin equals" in item for item in topology["ASSUMED_NOT_VALIDATED"])
    audit = read_text("STRAP_MECHANICAL_TOPOLOGY_AUDIT.md")
    assert "It does not make the L2 point an ankle" in audit


def test_robot_attachment_is_physical_eyelet_not_convenience_origin() -> None:
    robot = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["attachments"]["robot_side"]
    assert robot["selected_definition"] == "PHYSICAL_STRAP_EYELET_OR_HOOK_LOAD_TRANSFER_CENTER"
    assert robot["not_selected_as_default"] == ["TCP_ORIGIN", "FLANGE_ORIGIN"]
    assert "T_B_TCP(t)" in robot["future_equation"]
    assert robot["local_coordinates_m"] is None
    assert "consistent with zero" in robot["tcp_origin_allowed_only_if"]


def test_wide_cuff_candidates_are_compared_without_claiming_truth() -> None:
    audit = read_text("LIMB_SIDE_EQUIVALENT_ATTACHMENT_AUDIT.md")
    for candidate in (
        "cuff geometric center",
        "strap exit/tangent point",
        "equivalent resultant-force application point",
        "model L2 point",
        "configuration-dependent contact/resultant",
    ):
        assert candidate in audit
    assert "does not currently admit a defensible unique physical" in audit
    assert "not declared the true resultant application point" in audit


def test_operational_limb_point_is_configuration_dependent_exit_line_point() -> None:
    limb = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["attachments"]["limb_side"]
    assert limb["selected_operational_definition"] == "EQUIVALENT_LINE_POINT_AT_OBSERVED_STRAP_EXIT"
    assert limb["configuration_dependent"] is True
    assert limb["global_constant_allowed"] is False
    assert "at least two independently digitized free-segment fiducials" in limb["direct_line_alternative"]
    assert {"slack", "multiple free-span lines", "slip"}.issubset(set(limb["invalid_if"]))


def test_single_line_is_explicit_model_approximation_not_contact_truth() -> None:
    audit = read_text("POINT_FORCE_APPROXIMATION_AUDIT.md")
    assert "`MODEL APPROXIMATION`" in audit
    assert "not as a statement that every pressure/contact force" in audit
    assert "net moment" in audit
    assert "`d_task=null`" in audit


def test_rehab_setup_frame_is_right_handed_and_tied_to_rigid_landmarks() -> None:
    frame = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["rehab_setup_frame"]
    assert frame["frame_id"] == "REHAB_SETUP_FRAME"
    assert "rigid bed/setup base" in frame["origin"]
    assert frame["axes"]["+y_R"] == "normalize(z_R cross x_R)"
    assert frame["handedness"] == "right-handed; x_R cross y_R = z_R"
    assert len(frame["physical_landmarks"]) >= 3
    assert frame["transform_to_robot_base"]["value"] is None


def test_both_endpoints_enter_one_validated_common_base_frame() -> None:
    chain = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["rehab_setup_frame"]["frame_chain"]
    assert "T_B_TCP(t)" in chain["robot_side"]
    assert "T_B_R" in chain["limb_side"]
    assert chain["common_computation_frame"] == "robot base B"
    assert "separately validated T_W_B" in chain["world_optional"]


def test_robot_to_setup_transform_uses_independent_registration() -> None:
    plan = read_text("ROBOT_TO_SETUP_FRAME_CALIBRATION_PLAN.md")
    assert "at least three non-collinear reference points" in plan
    assert "registration residuals, leave-one-out error and transform covariance" in plan
    assert "Robot TCP probing is optional only after independent robot safety authorization" in plan
    assert "Bed axes are not assumed equal to base/world axes" in plan
    assert "`T_B_R=null` and `d_task=null`" in plan


def test_static_protocol_is_nonhuman_and_equipment_is_not_assumed_available() -> None:
    plan = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["static_measurement"]
    assert "no human" in plan["subject"]
    assert plan["execution_authorized"] is False
    required = " ".join(plan["minimum_equipment"])
    for marker in ("rigid shank surrogate", "strap/cuff", "fiducials", "calibrated", "3-D digitizer"):
        assert marker in required
    assert "OPTIONAL_ONLY_AFTER_INDEPENDENT_ROBOT_SAFETY_AUTHORIZATION" in plan["robot_tcp_probing"]


def test_repeatability_count_and_sequence_are_frozen_before_results() -> None:
    plan = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["static_measurement"]
    assert plan["setup_repetitions"] == protocol.SETUP_REPETITIONS == 10
    assert plan["metrology_repeats_per_setup"] == protocol.METROLOGY_REPEATS_PER_SETUP == 3
    assert plan["post_result_repeat_extension_allowed"] is False
    assert any("remove cuff/strap completely" in step for step in plan["per_repetition_sequence"])
    assert any("remove strap before the next" in step for step in plan["per_repetition_sequence"])
    assert "cannot be extended until PASS" in read_text("STATIC_GEOMETRY_MEASUREMENT_PLAN.md")


def test_no_preload_or_unmeasured_jig_coordinates_are_invented() -> None:
    plan = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["static_measurement"]
    assert plan["geometry_preload_n"] is None
    assert plan["geometry_preload_status"] == "GEOMETRY_PRELOAD_REQUIRES_INDEPENDENT_FIXTURE_AND_SAFETY_REVIEW"
    assert all(row["exact_fixture_coordinates"] is None for row in plan["configuration_roles"])
    assert "BEFORE_RESULTS" in plan["configuration_status"]


def test_uncertainty_model_contains_correct_normalization_jacobian() -> None:
    model = load_json("TASK_DIRECTION_GEOMETRY_UNCERTAINTY_MODEL.json")
    definitions = " ".join(model["definitions"])
    assert "v = p_r - p_l" in definitions
    assert "d = v / L" in definitions
    assert "J_d_v = (I - d d^T) / L" in definitions
    assert "Sigma_v = Sigma_r + Sigma_l - Sigma_rl - Sigma_lr" in definitions
    assert "trace(Sigma_d)" in definitions


def test_uncertainty_separates_measurement_and_setup_variation() -> None:
    model = load_json("TASK_DIRECTION_GEOMETRY_UNCERTAINTY_MODEL.json")
    assert model["estimation"]["within_setup"] == "from 3 metrology repeats per setup"
    assert model["estimation"]["between_setup"] == "from 10 complete remove/reattach repetitions"
    assert model["estimation"]["monte_carlo_samples"] == protocol.MONTE_CARLO_SAMPLES == 100_000
    assert model["estimation"]["monte_carlo_seed"] == protocol.MONTE_CARLO_SEED
    assert model["scientific_endpoint_outcomes_used"] is False


def test_geometry_thresholds_remain_null_pending_independent_review() -> None:
    model = load_json("TASK_DIRECTION_GEOMETRY_UNCERTAINTY_MODEL.json")
    assert all(value is None for value in model["thresholds"].values())
    assert model["threshold_status"] == "THRESHOLDS_REQUIRE_METROLOGY_AND_ENDPOINT_ERROR_BUDGET_REVIEW_BEFORE_RESULTS"
    metadata = load_json("metadata.json")
    assert metadata["thresholds_frozen"] is False
    assert metadata["geometry_preload_frozen"] is False


def test_static_direction_is_not_silently_applied_to_dynamic_trajectory() -> None:
    plan = read_text("DYNAMIC_TASK_DIRECTION_RECONSTRUCTION_PLAN.md")
    assert "does not make the line constant during the 24 s trajectory" in plan
    for marker in ("A: TCP-derived", "B: both endpoints", "C: robot point", "D: direct external", "E: one static"):
        assert marker in plan
    assert "Missing information yields `d_task(t)=null`" in plan


def test_dynamic_minimum_information_and_no_tangent_fallback() -> None:
    plan = read_text("DYNAMIC_TASK_DIRECTION_RECONSTRUCTION_PLAN.md")
    for marker in ("`p_attach_TCP`", "`T_B_TCP(t)`", "`T_B_R`", "limb/surrogate pose", "strap/cuff/routing/tautness", "timing uncertainty"):
        assert marker in plan
    assert "`TCP_TRAJECTORY_TANGENT != STRAP_PULL_LINE_OF_ACTION`" in plan
    assert "diagnostics only" in plan


def test_model_mapping_remains_not_calibrated_and_l2_semantics_are_preserved() -> None:
    metadata = load_json("metadata.json")
    mapping = read_text("MODEL_PULL_POINT_TO_PHYSICAL_STRAP_MAPPING.md")
    assert metadata["model_mapping_status"] == protocol.MODEL_MAPPING_STATUS == "NOT_YET_CALIBRATED"
    assert "`L2=0.30 m`" in mapping
    assert "`theta_shank=q_hip-q_knee`" in mapping
    assert "not the ankle or a measured cuff attachment" in mapping
    assert "This protocol changes neither L2 nor model kinematics" in mapping
    assert metadata["reference_ROM_L2_modified"] is False


def test_geometric_sign_is_defined_but_wrench_sign_stays_separate() -> None:
    direction = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")["attachments"]["direction"]
    assert direction["construction"] == "d_task_B(t) = normalize(p_robot_attach_B(t) - p_limb_attach_B(t))"
    assert "limb/cuff exit toward robot attachment" in direction["positive_geometric_direction"]
    assert direction["wrench_physical_sign_resolved"] is False
    assert load_json("metadata.json")["wrench_frame_sign_validated_by_this_stage"] is False


def test_geometry_validity_gate_is_all_required_and_fail_closed() -> None:
    gate = load_json("GEOMETRY_VALIDITY_GATE.json")
    assert gate["current_status"] == "FAIL_CLOSED_NOT_EXECUTED"
    assert gate["all_checks_required"] is True
    assert len(gate["checks"]) == 11
    assert all(row["current_pass"] is False for row in gate["checks"])
    assert gate["on_any_failure"]["d_task"] is None
    assert gate["on_any_failure"]["fallback_direction_allowed"] is False
    assert set(gate["explicitly_prohibited_fallbacks"]) == {
        "TCP_TRAJECTORY_TANGENT", "FIXED_BED_DIRECTION", "MODEL_L2_DIRECTION", "GUESSED_ATTACHMENT_POINT"
    }


def test_future_result_classes_are_exact_and_protocol_cannot_validate_itself() -> None:
    frozen = load_json("STRAP_GEOMETRY_VALIDATION_PROTOCOL.json")
    schema = load_json("FUTURE_GEOMETRY_RESULT_SCHEMA.json")
    assert frozen["future_result_classes"] == list(protocol.RESULT_CLASSES)
    assert schema["properties"]["decision"]["enum"] == list(protocol.RESULT_CLASSES)
    assert schema["protocol_itself_must_not_emit_validated"] is True
    metadata = load_json("metadata.json")
    assert metadata["physical_geometry_validation_performed"] is False
    assert metadata["geometry_result_class"] is None


def test_tactile_is_neither_geometry_nor_comfort_truth() -> None:
    audit = read_text("POINT_FORCE_APPROXIMATION_AUDIT.md")
    assert "tactile array may characterize contact region, pressure centroid and changes" in audit
    assert "neither geometry ground truth nor comfort truth" in audit


def test_endpoint_dependency_requires_both_independent_physical_branches() -> None:
    graph = read_text("NEXT_DEPENDENCY_GRAPH.md")
    assert "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL" in graph
    assert "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL" in graph
    assert "physical static wrench validation + physical static geometry validation" in graph
    assert "PRIMARY_MECHANICAL_ENDPOINT_FINALIZATION_AND_VALIDATION_PROTOCOL" in graph
    assert "Geometry protocol readiness alone cannot finalize or validate `J_force`" in graph
    assert load_json("metadata.json")["next_stage_executed"] is False


def test_endpoint_and_readiness_states_remain_fail_closed() -> None:
    metadata = load_json("metadata.json")
    assert metadata["task_direction_status"] == "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
    assert metadata["primary_endpoint_finalized"] is False
    assert metadata["primary_endpoint_validated"] is False
    assert metadata["future_physical_execution_authorized"] is False
    assert metadata["not_human_ready"] is True
    assert metadata["not_robot_approved"] is True


def test_protocol_performed_no_hardware_motion_load_human_pinn_or_bo_action() -> None:
    audit = load_json("HARDWARE_ACCESS_AUDIT.json")
    metadata = load_json("metadata.json")
    assert audit["protocol_design_only"] is True
    assert audit["robot_constructed"] is False and audit["robot_connected"] is False
    assert audit["power_or_enable_count"] == 0
    assert audit["motion_or_probing_command_count"] == 0
    assert audit["formal_traction_load_count"] == 0
    assert audit["geometry_experiment_count"] == 0
    assert audit["human_subject_count"] == 0
    assert audit["endpoint_computation_count"] == 0
    assert metadata["pinn_run_count"] == 0 and metadata["bo_run_count"] == 0
    assert metadata["hardware_control_safety_modified"] is False


def test_builder_does_not_import_robot_control_collection_or_safety_modules() -> None:
    tree = ast.parse(Path(protocol.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith(("hardware", "control", "collection", "safety")) for name in imported)


def test_report_answers_all_ten_questions_and_preserves_status() -> None:
    report = read_text("STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_REPORT.md")
    for number in range(1, 11):
        assert f"{number}. **" in report
    assert protocol.FORMAL_STATUS in report
    assert "PRIMARY_ENDPOINT_FINALIZED=false" in report
    assert "PRIMARY_ENDPOINT_VALIDATED=false" in report
    assert "NOT_HUMAN_READY" in report and "NOT_ROBOT_APPROVED" in report


def test_future_schema_binds_protocol_and_repeat_counts() -> None:
    schema = load_json("FUTURE_GEOMETRY_RESULT_SCHEMA.json")
    props = schema["properties"]
    assert props["protocol_sha256"]["const"] == protocol.FROZEN_PROTOCOL_SHA256
    assert props["repeat_count"]["const"] == 10
    assert props["metrology_repeats_per_setup"]["const"] == 3
    assert props["wrench_sign_status"]["const"] == "REQUIRES_SEPARATE_STATIC_WRENCH_FRAME_SIGN_VALIDATION"
    assert "no post-result definition, threshold or repeat-count change" in schema["full_validation_requirements"]


def test_active_reference_and_frozen_model_inputs_are_unchanged() -> None:
    expected = {spec["id"]: spec["exact_sha256"] for spec in protocol.INPUT_SPECS}
    assert expected["ACTIVE_REFERENCE"] == protocol.ACTIVE_REFERENCE_SHA256
    for input_id in ("ACTIVE_REFERENCE", "FORMAL_EXPERIMENT_MANIFEST", "START_ANCHORED_TRAJECTORY", "MODEL_CONFIG", "MODEL_KINEMATICS"):
        spec = next(item for item in protocol.INPUT_SPECS if item["id"] == input_id)
        assert sha256(ROOT / spec["path"]) == expected[input_id]


def test_checksums_cover_every_formal_output_except_manifest_itself() -> None:
    recorded: dict[str, str] = {}
    for line in read_text("checksums.sha256").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    files = {
        str(path.relative_to(OUT))
        for path in OUT.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    }
    assert set(recorded) == files
    assert all(sha256(OUT / name) == digest for name, digest in recorded.items())
