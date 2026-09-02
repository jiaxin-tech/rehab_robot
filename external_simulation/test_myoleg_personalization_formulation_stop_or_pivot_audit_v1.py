"""Regression gates for the evidence-only MyoLeg stop-or-pivot audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from external_simulation.myoleg_personalization_formulation_stop_or_pivot_audit_v1 import build_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_personalization_formulation_stop_or_pivot_audit_v1"


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


def test_protocol_is_frozen_and_pins_exact_evidence_files() -> None:
    protocol = load_json("STOP_OR_PIVOT_DECISION_PROTOCOL.json")
    assert sha256(OUT / "STOP_OR_PIVOT_DECISION_PROTOCOL.json") == audit.FROZEN_PROTOCOL_SHA256
    assert protocol["protocol_id"] == audit.PROTOCOL_ID
    assert protocol["protocol_frozen_before_route_assessment_artifacts_written"] is True
    assert protocol["does_not_claim_frozen_before_historical_evidence"] is True
    assert len(protocol["evidence_files"]) == len(audit.EVIDENCE_SPECS) == 13
    assert protocol["frozen_latest_conclusion"] == "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED"


def test_all_frozen_evidence_hashes_and_semantics_remain_current() -> None:
    verification = load_json("EVIDENCE_INPUT_VERIFICATION.json")
    assert verification["all_required_inputs_present"] is True
    assert verification["all_semantic_markers_pass"] is True
    assert verification["authoritative_s1_sha256"] == audit.FROZEN_S1_SHA256
    assert verification["authoritative_s1_sha_pass"] is True
    assert verification["scientific_arrays_read"] == 0
    assert verification["simulator_replays_run"] == 0
    for row in verification["evidence_files"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
        assert row["semantic_markers_pass"] is True
    s1 = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1/S1_STRUCTURAL_DEFINITION_AMENDED_V1.json"
    assert sha256(s1) == audit.FROZEN_S1_SHA256


def test_evidence_chain_preserves_key_frozen_quantities_without_new_oracle() -> None:
    evidence = read_text("CUMULATIVE_EVIDENCE_CHAIN.md")
    for marker in (
        "99.939077%", "0.033114%", "99.7742%", "24 subjects x 625 candidates",
        "24/24", "0.135074%", "fallback 0", "MAGNITUDE_ONLY",
    ):
        assert marker in evidence
    assert "does not say that humans lack individual differences" in evidence
    assert "outcome-driven" in evidence


def test_synthetic_personalization_is_stopped_without_discarding_myoleg() -> None:
    assessment = read_text("SYNTHETIC_PERSONALIZATION_CONTINUATION_ASSESSMENT.md")
    assert "`STOP`" in assessment
    assert "primary source of subject-specific preference truth" in assessment
    assert "does not discard MyoLeg as an engineering simulator" in assessment
    assert "Increasing z" in assessment
    assert "Repeatedly redesigning subjects until oracle diversity appears: **NOT ACCEPTABLE**" in assessment


def test_universal_mechanical_option_is_retained_only_as_secondary() -> None:
    option = read_text("UNIVERSAL_OPTIMIZATION_OPTION.md")
    assert audit.UNIVERSAL_OPTION_DECISION in option
    assert "beta=[+0.03,-0.03]" in option
    assert "common-candidate regret is zero" in option
    assert "not an unconstrained physical optimum" in option
    assert "does not answer" in option


def test_measurement_driven_pivot_is_the_single_primary_recommendation() -> None:
    decision = load_json("FINAL_PROJECT_DIRECTION_DECISION.json")
    assert decision["primary_recommendation"] == audit.PRIMARY_DECISION
    assert decision["option_a_continue_synthetic_myoleg_personalization"] == "STOP"
    assert decision["option_b_universal_mechanical_trajectory_optimization"] == audit.UNIVERSAL_OPTION_DECISION
    assert decision["option_c_measurement_driven_personalization"] == "PRIMARY_RECOMMENDATION"
    assert decision["next_independent_stage"] == audit.NEXT_STAGE
    assert decision["next_stage_executed"] is False


def test_mechanical_and_preference_questions_use_different_data() -> None:
    option = read_text("MEASUREMENT_DRIVEN_PERSONALIZATION_OPTION.md")
    assert "Mechanical-only primary question" in option
    assert "force, torque, pressure, and tracking measurements" in option
    assert "Preference/comfort primary question" in option
    assert "ratings or pairwise trajectory choices" in option
    assert "Pressure or torque alone is not preference truth" in option
    decision = load_json("FINAL_PROJECT_DIRECTION_DECISION.json")
    assert decision["mechanical_and_comfort_personalization_separated"] is True
    assert decision["mechanical_torque_is_comfort_truth"] is False
    assert decision["human_feedback_required_for_comfort_preference_claim"] is True


def test_conceptual_architecture_keeps_v3_and_assigns_distinct_model_bo_roles() -> None:
    option = read_text("MEASUREMENT_DRIVEN_PERSONALIZATION_OPTION.md")
    for marker in (
        "Fixed task/family", "beta_flex, beta_extend", "Physics prior",
        "Subject adaptation", "Low-budget selection", "Feedback target",
    ):
        assert marker in option
    assert "only that subject's executed-trial" in option
    assert "five-parameter gray-box identification" in option
    assert "refit only from executed trials" in option
    assert "equal trial budget" in option


def test_pinn_and_bo_are_not_added_without_measured_signal() -> None:
    pinn = read_text("PINN_ROLE_REASSESSMENT.md")
    bo = read_text("BO_ROLE_REASSESSMENT.md")
    assert audit.PINN_DECISION in pinn
    assert "physics baseline + subject-specific residual" in pinn
    assert "simpler residual and gray-box baselines" in pinn
    assert audit.BO_DECISION in bo
    assert "does not create a personalized objective" in bo
    assert "Mechanical BO" in bo and "Preference-based BO" in bo


def test_tactile_is_a_feature_not_comfort_ground_truth() -> None:
    tactile = read_text("TACTILE_AND_FEEDBACK_ROLE.md")
    assert "MEASURED_INTERACTION_FEATURE / POSSIBLE_COMFORT_CORRELATE" in tactile
    assert "Pressure is not automatically comfort" in tactile
    assert "direct subject feedback" in tactile
    assert "HUMAN_FEEDBACK_REQUIRED" in tactile
    assert "NOT_HUMAN_READY" in tactile
    assert "NOT_ROBOT_APPROVED" in tactile


def test_myoleg_keep_downgrade_stop_taxonomy_is_explicit() -> None:
    role = load_json("MYOLEG_FUTURE_ROLE.json")
    assert set(role) >= {"KEEP", "DOWNGRADE", "STOP"}
    assert "offline method development" in role["KEEP"]
    assert "patient population truth" in role["DOWNGRADE"]
    assert "arbitrary factor expansion to induce oracle diversity" in role["STOP"]
    assert role["myoleg_still_useful"] is True
    assert role["myoleg_is_future_patient_preference_truth"] is False


def test_existing_work_is_retained_with_claim_boundaries() -> None:
    mapping = read_text("EXISTING_WORK_RETENTION_MAP.md")
    for category in (
        "MAIN_METHOD_MATERIAL", "SUPPORTING_MATERIAL",
        "NEGATIVE_RESULT_FORMULATION_EVIDENCE", "SHOULD_NOT_DOMINATE_FINAL_PAPER",
        "STOPPED_PATH",
    ):
        assert category in mapping
    for work in ("2-DOF", "fixed-ROM", "MyoLeg mapping", "structural pilot"):
        assert work in mapping
    assert "Nothing is deleted because of the pivot" in mapping


def test_access_and_execution_boundaries_remain_zero_and_fail_closed() -> None:
    access = load_json("HELD_OUT_ACCESS_AUDIT.json")
    metadata = load_json("metadata.json")
    assert access["held_out_file_access_count"] == 0
    assert access["held_out_scientific_access_count"] == 0
    assert access["held_out_arrays_loaded"] == 0
    assert access["oracle_or_rank_or_regret_computed"] is False
    assert metadata["simulator_replay_count"] == 0
    assert metadata["scientific_array_load_count"] == 0
    assert metadata["new_cohort_count"] == 0
    assert metadata["new_virtual_subject_count"] == 0
    assert metadata["objective_modified"] is False
    assert metadata["normalization_modified"] is False
    assert metadata["v3_candidate_domain_modified"] is False
    assert metadata["s1_modified"] is False
    assert metadata["five_parameter_or_nn_or_pinn_trained"] is False
    assert metadata["bo_run"] is False
    assert metadata["robot_or_hardware"] is False
    assert metadata["human_ready"] is False
    assert metadata["robot_approved"] is False
    assert metadata["next_stage_executed"] is False


def test_builder_has_no_simulator_learner_optimizer_or_robot_import() -> None:
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "argparse", "hashlib", "json", "os", "pathlib", "typing"}
    assert not imports.intersection({"mujoco", "myosuite", "numpy", "scipy", "hardware", "control"})


def test_required_artifacts_and_checksums_are_complete() -> None:
    required = {
        "MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_REPORT.md",
        "STOP_OR_PIVOT_DECISION_PROTOCOL.json",
        "CUMULATIVE_EVIDENCE_CHAIN.md",
        "SYNTHETIC_PERSONALIZATION_CONTINUATION_ASSESSMENT.md",
        "UNIVERSAL_OPTIMIZATION_OPTION.md",
        "MEASUREMENT_DRIVEN_PERSONALIZATION_OPTION.md",
        "MYOLEG_FUTURE_ROLE.json",
        "PINN_ROLE_REASSESSMENT.md",
        "BO_ROLE_REASSESSMENT.md",
        "TACTILE_AND_FEEDBACK_ROLE.md",
        "EXISTING_WORK_RETENTION_MAP.md",
        "FINAL_PROJECT_DIRECTION_DECISION.json",
        "EVIDENCE_INPUT_VERIFICATION.json",
        "HELD_OUT_ACCESS_AUDIT.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert {path.name for path in OUT.iterdir() if path.is_file()} == required
    entries: dict[str, str] = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual)
    assert all(sha256(OUT / relative) == digest for relative, digest in entries.items())


def test_report_answers_core_decision_and_stops_at_next_stage() -> None:
    report = read_text("MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_REPORT.md")
    assert audit.PRIMARY_DECISION in report
    assert "does **not** justify" in report
    assert "does not prove that real patients share one best trajectory" in report
    assert audit.NEXT_STAGE in report
    assert "It was **not** executed here" in report
    assert "Held-out scientific access: 0" in report
    assert "Human ready: no" in report
    assert "Robot approved: no" in report
