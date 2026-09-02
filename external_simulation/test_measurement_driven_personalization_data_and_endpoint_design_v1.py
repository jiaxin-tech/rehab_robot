"""Regression gates for the evidence-only measurement/endpoint design."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from external_simulation.measurement_driven_personalization_data_and_endpoint_design_v1 import build_design as design


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1"


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


def load_inventory() -> list[dict[str, str]]:
    with (OUT / "REAL_MEASUREMENT_CHANNEL_INVENTORY.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_protocol_was_frozen_before_endpoint_decision_outputs() -> None:
    protocol = load_json("DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json")
    assert sha256(OUT / "DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json") == design.FROZEN_PROTOCOL_SHA256
    assert protocol["protocol_id"] == design.PROTOCOL_ID
    assert protocol["protocol_frozen_before_endpoint_decision_outputs"] is True
    assert protocol["candidate_primary_endpoint_finalized_before_stage"] is False
    assert protocol["decision_rules"]["endpoint_may_never_be_labelled_validated_in_this_stage"] is True


def test_all_inputs_are_pinned_and_parent_protocol_identity_is_exact() -> None:
    verification = load_json("INPUT_VERIFICATION.json")
    assert verification["input_count"] == len(design.INPUT_SPECS) == 21
    assert verification["all_inputs_present_and_semantically_verified"] is True
    assert verification["physical_wrench_evidence_available_to_this_stage"] is False
    for row in verification["inputs"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
        assert row["semantic_markers_pass"] is True
    assert sha256(ROOT / design.INPUT_SPECS[0]["path"]) == design.PARENT_PROTOCOL_SHA256


def test_reference_rom_and_theta_shank_are_unchanged() -> None:
    formal = json.loads((ROOT / "config/formal_experiment_manifest.json").read_text(encoding="utf-8"))
    assert sha256(ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv") == design.ACTIVE_REFERENCE_SHA256
    assert formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
    assert formal["theta_shank_definition"] == "q_hip - q_knee"
    assert formal["active_reference_sha256"] == design.ACTIVE_REFERENCE_SHA256


def test_hardware_access_audit_proves_design_only_execution() -> None:
    audit = load_json("HARDWARE_ACCESS_AUDIT.json")
    assert audit["design_only"] is True
    assert audit["robot_adapter_imported"] is False
    assert audit["robot_constructed"] is False
    assert audit["robot_connected"] is False
    assert audit["robot_powered_or_enabled"] is False
    assert audit["motion_command_count"] == 0
    assert audit["calibration_call_count"] == 0
    assert audit["human_collection_count"] == 0
    assert audit["pinn_training_count"] == 0
    assert audit["bo_run_count"] == 0


def test_builder_has_no_robot_or_optimizer_imports() -> None:
    tree = ast.parse(Path(design.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith(("hardware", "control", "collection", "safety", "myosuite", "sklearn", "torch")) for name in imported)


def test_measurement_inventory_is_explicit_and_none_is_research_ready_now() -> None:
    rows = load_inventory()
    assert len(rows) == 11
    assert {row["channel_id"] for row in rows} == {
        "ROBOT_Q", "ROBOT_DQ", "ROBOT_DDQ", "TCP_POSE", "TCP_VELOCITY",
        "TRACKING_ERROR", "CART_FORCE", "CART_TORQUE", "JOINT_TAU_MEASURED",
        "JOINT_TAU_EXTERNAL", "TACTILE_RAW",
    }
    assert all(row["formal_research_ready_now"] == "False" for row in rows)
    assert all(row["requires_future_validation"] == "True" for row in rows)
    force = next(row for row in rows if row["channel_id"] == "CART_FORCE")
    assert force["dimension"] == "3" and force["unit"] == "N"
    assert "current request world" in force["frame_or_reference"]
    assert force["sign_semantics"] == "not documented/physically verified"


def test_wrench_sdk_facts_and_unknowns_are_not_conflated() -> None:
    audit = read_text("WRENCH_SEMANTICS_AUDIT.md")
    assert design.WRENCH_STATUS in audit
    assert "world`, `flange`, and `tool`" in audit
    assert "six joint measured torques in N*m" in audit
    for marker in ("sign convention", "compensation/bias", "moment reference point", "device/source timestamp", "update cadence"):
        assert marker in audit
    assert "variable names and offline rotation unit tests cannot close the semantics gap" in audit


def test_base_wrench_rotation_remains_unverified_and_full_moment_rule_is_preserved() -> None:
    metadata = load_json("metadata.json")
    audit = read_text("WRENCH_SEMANTICS_AUDIT.md")
    settings = (ROOT / "config/settings.py").read_text(encoding="utf-8")
    assert metadata["base_wrench_rotation_verified"] is False
    assert "BASE_WRENCH_ROTATION_VERIFIED  = False" in settings
    assert "F_b = R_b_from_w F_w" in audit
    assert "tau_b = R tau_w + p x (R F_w)" in audit
    assert "Rotation-only is not a point transform" in audit


def test_task_direction_candidates_are_compared_without_outcome_selection() -> None:
    audit = read_text("TASK_DIRECTION_DEFINITION_AUDIT.md")
    for marker in ("instantaneous TCP tangent", "strap/pull line of action", "endpoint-to-hip direction", "fixed bed-plane axis", "2-DOF equivalent traction direction"):
        assert marker in audit
    assert design.TASK_DIRECTION_STATUS in audit
    assert "defensible target is B" in audit
    assert "never from whichever direction produces the smallest RMS" in audit
    assert "never relabel `L2` as ankle" in audit


def test_primary_endpoint_is_mathematical_candidate_but_definition_incomplete() -> None:
    endpoint = load_json("PRIMARY_MECHANICAL_ENDPOINT_DEFINITION.json")
    assert endpoint["endpoint_id"] == design.ENDPOINT_ID
    assert endpoint["decision_state"] == design.ENDPOINT_STATE
    assert endpoint["validated"] is False
    assert endpoint["ready_for_physical_validation"] is False
    assert endpoint["provisional_mathematical_definition"]["unit"] == "N"
    assert "dot(F_interaction^R" in endpoint["provisional_mathematical_definition"]["force_projection"]
    assert "sqrt(sum_i w_i F_task" in endpoint["provisional_mathematical_definition"]["time_weighted_rms"]
    assert endpoint["transient_exclusion"] is None
    assert endpoint["bias_rule"] is None
    assert endpoint["filter_rule"] is None


def test_sign_is_retained_and_absolute_value_is_not_arbitrarily_added() -> None:
    endpoint = load_json("PRIMARY_MECHANICAL_ENDPOINT_DEFINITION.json")
    direction = read_text("TASK_DIRECTION_DEFINITION_AUDIT.md")
    sign = endpoint["provisional_mathematical_definition"]["sign"]
    assert "signed F_task retained" in sign
    assert "RMS itself is sign-invariant" in sign
    assert "must not be created by an arbitrary absolute value" in direction


def test_bias_filter_and_delay_are_fail_closed_without_numeric_invention() -> None:
    policy = read_text("BIAS_FILTER_DELAY_POLICY.md")
    endpoint = load_json("PRIMARY_MECHANICAL_ENDPOINT_DEFINITION.json")
    assert "A leg already attached under strap preload is not a zero-force condition" in policy
    assert design.FILTER_STATUS in policy
    assert "No fixed delay from old simulation" in policy
    assert endpoint["bias_rule"] is None and endpoint["filter_rule"] is None
    assert "No numeric bias or drift threshold is invented" in policy


def test_master_timebase_and_synchronization_do_not_invent_device_time() -> None:
    schema = load_json("SYNCHRONIZATION_SCHEMA.json")
    assert schema["master_timebase"] == design.MASTER_TIMEBASE
    assert schema["robot_device_timestamp_available"] is False
    assert schema["alignment"]["fixed_delay_assumption"] is None
    assert all(value is None for value in schema["limits"].values())
    assert "never zero-fill" in schema["missing_behavior"]
    assert "not frozen scientific endpoint thresholds" in schema["compatibility_note"]


def test_future_episode_schema_covers_identity_streams_timing_and_no_oracle() -> None:
    schema = load_json("FUTURE_EPISODE_DATA_SCHEMA.json")
    assert set(schema["required"]) == {"identity", "frozen_contract", "candidate", "streams", "timing", "diagnostics", "validity", "derived"}
    contract = schema["properties"]["frozen_contract"]["properties"]
    assert contract["planned_duration_s"]["const"] == 24.0
    assert contract["theta_shank"]["const"] == "q_hip - q_knee"
    streams = schema["properties"]["streams"]["properties"]
    assert set(streams) == {"reference", "robot_state", "wrench", "trajectory_command", "tactile"}
    for forbidden in ("future_episode_outcome", "unexecuted_candidate_truth", "MyoLeg_oracle", "held_out_final_evaluation"):
        assert forbidden in schema["forbidden_oracle_fields"]


def test_episode_validity_gate_blocks_endpoint_and_learning_on_failure() -> None:
    gate = load_json("EPISODE_VALIDITY_GATE.json")
    assert gate["fail_closed"] is True
    ids = {check["check_id"] for check in gate["checks"]}
    assert {"FULL_DURATION_COMPLETED", "NO_SAFETY_ABORT", "TIMING_VALID", "MISSING_DATA_ACCEPTABLE", "SYNCHRONIZATION_VALID", "WRENCH_SEMANTICS_VALIDATED", "TASK_DIRECTION_VALIDATED", "TRACKING_WITHIN_REVIEWED_BOUNDS"} <= ids
    current = {check["check_id"]: check["current_pass"] for check in gate["checks"]}
    assert current["WRENCH_SEMANTICS_VALIDATED"] is False
    assert current["TASK_DIRECTION_VALIDATED"] is False
    assert all(value is None for value in gate["numeric_thresholds"].values())
    assert "null endpoint" in gate["on_failure"]
    assert "do not update gray-box or BO objective" in gate["on_failure"]


def test_repeatability_and_sensitivity_are_future_designs_not_claimed_results() -> None:
    repeatability = read_text("ENDPOINT_REPEATABILITY_VALIDATION_PLAN.md")
    sensitivity = read_text("ENDPOINT_SENSITIVITY_VALIDATION_PLAN.md")
    assert "N_REPEATS_NOT_YET_FROZEN" in repeatability
    for marker in ("mean", "SD", "CV", "ICC", "drift"):
        assert marker in repeatability
    assert "not an experiment" in repeatability
    assert "prespecified small trajectory perturbations" in sensitivity
    assert "repeatability noise" in sensitivity
    assert "No numeric effect threshold" in sensitivity


def test_secondary_diagnostics_and_safety_are_not_weighted_primary_terms() -> None:
    diagnostics = load_json("SECONDARY_MECHANICAL_DIAGNOSTICS.json")
    assert diagnostics["weighted_composite_objective_created"] is False
    assert diagnostics["safety_thresholds_are_constraints"] is True
    assert {row["id"] for row in diagnostics["diagnostics"]} >= {"PEAK_ABS_TASK_FORCE", "FULL_FORCE_NORM", "TRACKING_ERROR", "MECHANICAL_POWER_OR_WORK", "TACTILE_FEATURES"}


def test_tactile_is_nullable_secondary_and_not_comfort_truth() -> None:
    tactile = read_text("TACTILE_FUTURE_INTERFACE_AND_VALIDATION.md")
    for marker in ("raw matrix", "calibration ID", "missing mask", "saturation mask", "sampling", "latency", "strap-placement repeatability"):
        assert marker in tactile
    assert "not a primary endpoint" in tactile
    assert "PRESSURE_IS_NOT_COMFORT_TRUTH" in tactile


def test_causal_policy_and_bo_interface_reject_future_and_invalid_outcomes() -> None:
    policy = load_json("REAL_PERSONALIZATION_CAUSAL_DATA_POLICY_V1.json")
    assert policy["decision_rule"] == "when selecting beta_k, use frozen priors plus valid completed episodes 1..k-1 only"
    assert "do not add a normal objective observation" in policy["invalid_episode"]
    assert policy["bo_observation_interface"]["invalid_value"] is None
    assert "offline MyoLeg oracle" in policy["forbidden"]
    assert "held-out final evaluation" in policy["forbidden"]
    assert policy["gray_box_interface"]["five_parameter_model_modified"] is False


def test_readiness_graph_stops_at_exact_next_stage_without_execution() -> None:
    graph = read_text("READINESS_DEPENDENCY_GRAPH.md")
    metadata = load_json("metadata.json")
    for marker in ("Measurement semantics", "Frame, sign, point and bias validation", "Synchronization and delay validation", "Repeated identical-trial repeatability", "Gray-box identification", "PINN stop/go gate", "BO stop/go gate"):
        assert marker in graph
    assert metadata["next_stage"] == design.NEXT_STAGE
    assert metadata["next_stage_executed"] is False
    assert metadata["formal_status"] == design.ENDPOINT_STATE


def test_report_answers_all_ten_questions_and_preserves_readiness_boundary() -> None:
    report = read_text("MEASUREMENT_DRIVEN_PERSONALIZATION_DATA_AND_ENDPOINT_DESIGN_REPORT.md")
    for number in range(1, 11):
        assert f"## Q{number}." in report
    assert "NOT_HUMAN_READY / NOT_ROBOT_APPROVED" in report
    assert design.NEXT_STAGE in report
    metadata = load_json("metadata.json")
    assert metadata["not_human_ready"] is True
    assert metadata["not_robot_approved"] is True
    assert metadata["hardware_collection_control_safety_modified"] is False


def test_checksums_cover_every_artifact_except_the_checksum_file() -> None:
    lines = read_text("checksums.sha256").splitlines()
    recorded = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    files = {str(path.relative_to(OUT)) for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    assert set(recorded) == files
    assert all(sha256(OUT / name) == digest for name, digest in recorded.items())
