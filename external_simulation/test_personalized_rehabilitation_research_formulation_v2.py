"""Regression gates for the evidence-only Research Formulation V2 stage."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from external_simulation.personalized_rehabilitation_research_formulation_v2 import build_formulation as form


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/personalized_rehabilitation_research_formulation_v2"


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


def load_channels() -> list[dict[str, str]]:
    with (OUT / "MEASUREMENT_CHANNELS_AND_ROLES.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_protocol_is_frozen_before_formulation_outputs() -> None:
    protocol = load_json("RESEARCH_FORMULATION_V2_PROTOCOL.json")
    assert sha256(OUT / "RESEARCH_FORMULATION_V2_PROTOCOL.json") == form.FROZEN_PROTOCOL_SHA256
    assert protocol["protocol_id"] == form.PROTOCOL_ID
    assert protocol["protocol_frozen_before_formulation_outputs_written"] is True
    assert protocol["inherited_primary_direction"] == form.PRIMARY_DIRECTION
    assert protocol["primary_thesis_formulation"] == form.PRIMARY_THESIS
    assert protocol["primary_personalization_source"] == form.PRIMARY_PERSONALIZATION_SOURCE
    assert protocol["primary_outcome_type"] == form.PRIMARY_OUTCOME_TYPE
    assert protocol["candidate_outcome_is_not_final_calibrated_endpoint"] is True


def test_all_16_inputs_and_stop_or_pivot_identity_remain_frozen() -> None:
    verification = load_json("INPUT_VERIFICATION.json")
    assert verification["input_count"] == len(form.INPUT_SPECS) == 16
    assert verification["all_inputs_present_and_semantically_verified"] is True
    for row in verification["inputs"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
        assert row["semantic_markers_pass"] is True
    stop_protocol = ROOT / "external_simulation_audits/myoleg_personalization_formulation_stop_or_pivot_audit_v1/STOP_OR_PIVOT_DECISION_PROTOCOL.json"
    assert sha256(stop_protocol) == form.STOP_OR_PIVOT_PROTOCOL_SHA256


def test_reference_rom_and_theta_shank_are_unchanged() -> None:
    reference = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
    formal = json.loads((ROOT / "config/formal_experiment_manifest.json").read_text(encoding="utf-8"))
    assert sha256(reference) == form.ACTIVE_REFERENCE_SHA256
    assert formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
    assert formal["theta_shank_definition"] == "q_hip - q_knee"
    assert formal["active_reference_sha256"] == form.ACTIVE_REFERENCE_SHA256


def test_primary_question_and_thesis_are_mechanical_not_comfort() -> None:
    final = load_json("FINAL_RESEARCH_FORMULATION_V2.json")
    questions = read_text("PRIMARY_RESEARCH_QUESTIONS.md")
    assert final["formal_decision"] == form.FORMAL_STATUS
    assert final["PRIMARY_THESIS_FORMULATION"] == "MECHANICAL_MEASUREMENT_DRIVEN_PERSONALIZATION"
    assert final["PRIMARY_PERSONALIZATION_SOURCE"] == "SUBJECT_SPECIFIC_MEASURED_INTERACTION_TRIALS"
    assert "lower independently evaluated mechanical interaction" in final["PRIMARY_RESEARCH_QUESTION_V2"]
    assert "does not contain a comfort" in questions
    assert "Optional preference-aware extension" in questions
    assert "Mechanical measurements may be covariates or constraints but cannot replace preference feedback" in questions


def test_primary_outcome_is_a_candidate_pending_independent_calibration() -> None:
    final = load_json("FINAL_RESEARCH_FORMULATION_V2.json")
    scope = read_text("MECHANICAL_VS_PREFERENCE_SCOPE.md")
    assert final["PRIMARY_OUTCOME_TYPE"] == form.PRIMARY_OUTCOME_TYPE
    assert final["PRIMARY_CANDIDATE_OUTCOME"] == form.PRIMARY_CANDIDATE_OUTCOME
    assert final["PRIMARY_CANDIDATE_OUTCOME_FINALIZED"] is False
    assert "not yet the final objective" in scope
    assert "Safety limits and data-validity gates remain constraints, never reward terms" in scope
    assert "No arbitrary all-signal weighted score" in scope


def test_v3_p4_family_and_cold_start_are_preserved_without_robot_range_approval() -> None:
    protocol = load_json("RESEARCH_FORMULATION_V2_PROTOCOL.json")
    trajectory = protocol["trajectory_parameterization"]
    assert trajectory["id"] == "P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3"
    assert trajectory["parameters"] == ["beta_flex", "beta_extend"]
    assert trajectory["mathematical_definition"] == "w_b(s;beta_b)=s+beta_b*64*s^3*(1-s)^3; q_hip=q_hip_ref; q_knee=q_knee_ref_branch(w_b)"
    assert trajectory["cold_start"] == {"beta_flex": 0.0, "beta_extend": 0.0}
    assert trajectory["offline_bounds_are_robot_approved"] is False
    assert trajectory["v4_redesign_allowed"] is False


def test_measurement_channel_audit_is_explicit_and_fail_closed() -> None:
    rows = load_channels()
    assert len(rows) == 11
    assert {row["channel_id"] for row in rows} == {
        "R_STATE_Q", "R_STATE_DQ", "R_STATE_DDQ", "R_TCP", "R_TRACK",
        "R_WRENCH", "R_JOINT_TAU", "T_RAW", "T_FEATURES", "H_RATING", "H_PAIRWISE",
    }
    assert all(row["available_now_for_formal_research"] == "False" for row in rows)
    assert all(row["requires_validation"] == "True" for row in rows)
    lookup = {row["channel_id"]: row for row in rows}
    assert lookup["R_WRENCH"]["software_status"] == "GETENDTORQUE_PATH_PRESENT_UNVALIDATED"
    assert "frame rotation" in lookup["R_WRENCH"]["limitation"]
    assert lookup["T_RAW"]["software_status"] == "NOT_IMPLEMENTED"
    assert lookup["H_PAIRWISE"]["optimization_target_role"] == "PREFERENCE_BO_BRANCH_ONLY"


def test_active_source_scan_does_not_invent_tactile_or_preference_implementation() -> None:
    scan = load_json("INPUT_VERIFICATION.json")["active_channel_source_scan"]
    assert scan["active_source_file_count"] > 0
    assert scan["tactile_implemented"] is False
    assert scan["direct_preference_implemented"] is False
    assert scan["tactile_acquisition_source_hits"] == []
    assert scan["direct_preference_acquisition_source_hits"] == []


def test_episode_is_causal_complete_trial_and_separates_adaptation_evaluation() -> None:
    episode = read_text("FUTURE_PERSONALIZATION_EPISODE_PROTOCOL.md")
    assert "Trial 1 starts at the frozen reference: `beta=[0,0]`" in episode
    assert "one complete" in episode
    assert "trials `1..k` only" in episode
    assert "Forbidden before execution" in episode
    assert "current/future trial outcome" in episode
    assert "cannot also be claimed as final performance evidence" in episode
    assert "separate final-evaluation block" in episode


def test_trial_budget_is_a_hypothesis_not_hardware_approval() -> None:
    protocol = load_json("RESEARCH_FORMULATION_V2_PROTOCOL.json")
    budget = protocol["trial_budget_hypothesis"]
    assert budget["primary_complete_adaptation_trials"] == 4
    assert budget["sensitivity_complete_adaptation_trials"] == [3, 5]
    assert budget["final_evaluation_is_separate"] is True
    assert budget["hardware_approved"] is False
    episode = read_text("FUTURE_PERSONALIZATION_EPISODE_PROTOCOL.md")
    assert "experimental-design hypothesis" in episode


def test_subject_model_hierarchy_and_effective_parameter_semantics() -> None:
    models = read_text("SUBJECT_MODEL_HIERARCHY.md")
    for marker in ("M0", "M1", "M2", "M3"):
        assert marker in models
    assert "never physiological truth" in models
    assert "gray-box physics prediction + subject-specific residual" in models
    assert "does not infer comfort" in models


def test_pinn_has_an_exact_task_and_six_stop_go_conditions() -> None:
    models = read_text("SUBJECT_MODEL_HIERARCHY.md")
    for marker in (
        "repeated measured trials exist", "M1 is evaluated", "systematic residual exists",
        "repeatable across repeated trials", "data volume supports", "equal-budget comparison",
    ):
        assert marker in models
    assert "PINN_NOT_JUSTIFIED" in models
    final = load_json("FINAL_RESEARCH_FORMULATION_V2.json")
    assert final["PINN_ROLE"] == form.PINN_ROLE


def test_selector_hierarchy_and_bo_semantics_are_complete() -> None:
    selectors = read_text("TRAJECTORY_SELECTOR_HIERARCHY.md")
    for marker in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        assert marker in selectors
    assert "BO is a selector, not the source of personalization" in selectors
    assert "subject-specific observations" in selectors
    assert "Mechanical BO optimizes" in selectors
    assert "Preference BO optimizes" in selectors
    assert "PERSONALIZED_BO_JUSTIFIED" in selectors


def test_pinn_and_bo_roles_solve_different_problems_and_are_not_run() -> None:
    roles = read_text("PINN_AND_BO_ROLE_DEFINITION.md")
    metadata = load_json("metadata.json")
    assert "prediction mismatch" in roles
    assert "low-budget trajectory selection" in roles
    assert metadata["pinn_training_count"] == 0
    assert metadata["bo_run_count"] == 0


def test_tactile_pipeline_and_minimum_validation_needs_are_frozen() -> None:
    tactile = read_text("TACTILE_ROLE_AND_VALIDATION_NEEDS.md")
    assert "raw pressure map -> timestamped preprocessing -> versioned episode features" in tactile
    for marker in ("calibration", "sampling rate", "latency", "synchronization", "repeatability", "spatial mapping"):
        assert marker in tactile
    assert "no tactile acquisition implementation" in tactile
    assert "pressure != comfort" in tactile


def test_preference_label_comparison_and_human_feedback_requirement() -> None:
    scope = read_text("MECHANICAL_VS_PREFERENCE_SCOPE.md")
    assert "Scalar rating" in scope and "Pairwise preference" in scope
    assert "scale drift" in scope
    assert "natural fit for preference BO" in scope
    assert "HUMAN_FEEDBACK_REQUIRED" in scope
    final = load_json("FINAL_RESEARCH_FORMULATION_V2.json")
    assert final["HUMAN_FEEDBACK_REQUIREMENT"] == form.HUMAN_FEEDBACK_REQUIREMENT


def test_validation_baselines_are_equal_budget_and_truth_isolation_is_explicit() -> None:
    validation = read_text("FUTURE_BASELINE_AND_VALIDATION_PLAN.md")
    for marker in (
        "S0 reference", "S1 common trajectory", "S2 random/space-filling",
        "model-free mechanical BO", "gray-box plus BO", "residual/PINN plus BO",
    ):
        assert marker in validation
    assert "exactly `K=4` complete adaptation trials" in validation
    assert "No method receives oracle, future, held-out" in validation
    assert "Adaptation and final evaluation data remain separate" in validation


def test_generalization_unit_and_hardware_readiness_are_independent() -> None:
    final = load_json("FINAL_RESEARCH_FORMULATION_V2.json")
    boundary = load_json("HARDWARE_READINESS_BOUNDARY.json")
    assert final["GENERALIZATION_UNIT"] == "NEW_REAL_SUBJECT_AFTER_INDEPENDENT_APPROVALS"
    assert final["ALGORITHM_FORMULATION_READY_NE_ROBOT_EXECUTION_READY"] is True
    assert final["NOT_HUMAN_READY"] is True
    assert final["NOT_ROBOT_APPROVED"] is True
    assert boundary["experiment_safety_reviewed"] is False
    assert boundary["real_identification_config_reviewed"] is False
    assert boundary["base_wrench_rotation_verified"] is False
    assert boundary["offline_v3_bounds_robot_approved"] is False
    assert boundary["hardware_or_human_action_performed"] is False


def test_architecture_diagram_contains_all_required_layers_and_safety_gate() -> None:
    diagram = read_text("FUTURE_METHOD_ARCHITECTURE.md")
    assert "```mermaid" in diagram
    for marker in (
        "Measured reference and fixed task", "V3 P4 family", "Trajectory selector",
        "One complete robot trial", "Robot state and validated wrench",
        "Episode feature extraction", "Subject-specific gray-box posterior",
        "Mechanical endpoint or direct preference observation model",
        "Independent reviewed safety and domain gate",
    ):
        assert marker in diagram
    assert "offline checks only" in diagram
    assert "Safety layer: independent of the optimizer" in diagram


def test_myoleg_role_and_stopped_synthetic_paths_remain_explicit() -> None:
    role = read_text("MYOLEG_FUTURE_ROLE_V2.md")
    assert "## KEEP" in role and "## DOWNGRADE" in role and "## STOP" in role
    assert "not a patient-population distribution" in role
    assert "Do not generate Cohort V2" in role
    assert "actual subject-specific information" in role


def test_existing_work_is_retained_without_dominating_the_main_method() -> None:
    mapping = read_text("EXISTING_WORK_RETENTION_MAP_V2.md")
    for category in (
        "MAIN_TEXT_CANDIDATE", "SUPPLEMENTARY_OR_APPENDIX",
        "DO_NOT_MAKE_CENTRAL_FINAL_METHOD_CLAIM", "HISTORICAL_DEVELOPMENT_ONLY",
        "NOT_COMPLETED_EVIDENCE",
    ):
        assert category in mapping
    assert "No frozen artifact is deleted or rewritten" in mapping


def test_no_experiment_optimizer_learner_robot_or_human_action_occurred() -> None:
    metadata = load_json("metadata.json")
    assert metadata["formulation_only"] is True
    assert metadata["held_out_scientific_access_count"] == 0
    assert metadata["simulator_experiment_count"] == 0
    assert metadata["robot_access_count"] == 0
    assert metadata["human_study_count"] == 0
    assert metadata["cohort_v2_generated"] is False
    assert metadata["s2_s3_expansion_run"] is False
    assert metadata["v4_v5_redesign_run"] is False
    assert metadata["objective_weight_search_run"] is False
    assert metadata["frozen_artifacts_modified"] is False
    assert metadata["next_stage_executed"] is False


def test_builder_imports_only_standard_evidence_generation_modules() -> None:
    tree = ast.parse(Path(form.__file__).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "argparse", "csv", "hashlib", "io", "json", "os", "pathlib", "typing"}
    assert not imports.intersection({"mujoco", "myosuite", "numpy", "scipy", "hardware", "control", "collection"})


def test_report_answers_q1_to_q10_and_stops_at_one_next_stage() -> None:
    report = read_text("PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_REPORT.md")
    assert all(f"## Q{index}." in report for index in range(1, 11))
    assert form.FORMAL_STATUS in report
    assert form.NEXT_STAGE in report
    assert "It was not executed" in report
    assert "NOT_HUMAN_READY / NOT_ROBOT_APPROVED" in report


def test_required_artifacts_and_checksums_are_complete() -> None:
    required = {
        "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_REPORT.md",
        "RESEARCH_FORMULATION_V2_PROTOCOL.json",
        "PRIMARY_RESEARCH_QUESTIONS.md",
        "MECHANICAL_VS_PREFERENCE_SCOPE.md",
        "MEASUREMENT_CHANNELS_AND_ROLES.csv",
        "SUBJECT_MODEL_HIERARCHY.md",
        "TRAJECTORY_SELECTOR_HIERARCHY.md",
        "PINN_AND_BO_ROLE_DEFINITION.md",
        "MYOLEG_FUTURE_ROLE_V2.md",
        "TACTILE_ROLE_AND_VALIDATION_NEEDS.md",
        "FUTURE_PERSONALIZATION_EPISODE_PROTOCOL.md",
        "FUTURE_BASELINE_AND_VALIDATION_PLAN.md",
        "FUTURE_METHOD_ARCHITECTURE.md",
        "EXISTING_WORK_RETENTION_MAP_V2.md",
        "FINAL_RESEARCH_FORMULATION_V2.json",
        "INPUT_VERIFICATION.json",
        "HARDWARE_READINESS_BOUNDARY.json",
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
