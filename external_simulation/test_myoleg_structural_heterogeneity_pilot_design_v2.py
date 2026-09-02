"""Regression gates for the amended-S1 structural pilot V2 design freeze."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from external_simulation.myoleg_structural_heterogeneity_pilot_design_v2 import build_design as design


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_authoritative_amended_s1_and_all_frozen_inputs_are_exact() -> None:
    assert {name: sha256(path) for name, path in design.FROZEN_PATHS.items()} == design.FROZEN_SHA
    verification = read_json("AMENDED_S1_INPUT_VERIFICATION.json")
    amended = json.loads(design.AMENDED_S1.read_text(encoding="utf-8"))
    assert verification["status"] == "PASS"
    assert verification["authoritative_definition_sha256"] == design.FROZEN_SHA["amended_s1"]
    assert verification["authoritative_factor_reconstruction"] == amended["factors"]
    assert tuple(verification["factor_ids"]) == design.EXPECTED_FACTOR_IDS
    assert verification["factor_count"] == 4


def test_v1_failed_design_is_preserved_and_not_overwritten() -> None:
    verification = read_json("AMENDED_S1_INPUT_VERIFICATION.json")
    protocol = read_json("STRUCTURAL_HETEROGENEITY_PILOT_V2_PROTOCOL.json")
    previous = verification["previous_design_failure"]
    assert previous == {
        "outcome": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY",
        "blocker": "S1_DEFINITION_INCOMPLETE",
        "report_sha256": design.FROZEN_SHA["v1_design_report"],
        "metadata_sha256": design.FROZEN_SHA["v1_design_metadata"],
        "preserved": True,
    }
    assert protocol["V1_failure_record_preserved"]["overwritten"] is False
    assert sha256(design.V1_DESIGN_REPORT) == design.FROZEN_SHA["v1_design_report"]
    assert sha256(design.V1_DESIGN_METADATA) == design.FROZEN_SHA["v1_design_metadata"]


def test_all_four_levels_are_synthetic_ready_and_one_shot() -> None:
    artifact = read_json("PILOT_V2_DIAGNOSTIC_LEVELS.json")
    assert artifact["status"] == "ALL_FACTORS_READY"
    assert artifact["population_range"] == "NOT_AVAILABLE_FOR_ALL_FOUR_FACTORS"
    assert artifact["pilot_diagnostic_level_is_population_range"] is False
    assert artifact["one_round_only"] is True
    assert artifact["post_outcome_level_tuning_allowed"] is False
    rows = artifact["levels"]
    assert len(rows) == 4
    assert {row["factor_id"] for row in rows} == set(design.EXPECTED_FACTOR_IDS)
    for row in rows:
        expected = design.LEVEL_MAGNITUDES[row["factor_id"]]
        assert row["population_range"] == "NOT_AVAILABLE"
        assert row["primary_negative_z"] == -expected["primary"]
        assert row["primary_positive_z"] == expected["primary"]
        assert row["fallback_negative_z"] == -expected["fallback"]
        assert row["fallback_positive_z"] == expected["fallback"]
        assert row["fallback_positive_z"] == row["primary_positive_z"] / 2.0
        assert row["maximum_fallback_attempts_per_sign"] == 1
        assert row["not_patient_distribution"] is True
        assert row["not_cohort_v2_bound"] is True
        assert row["precheck_status"] == "DIAGNOSTIC_LEVEL_READY"
        assert {"J", "oracle", "ranking", "trajectory response", "held-out truth"} <= set(row["selection_prohibited_inputs"])


def test_operator_only_prechecks_pass_and_do_not_reveal_scientific_response() -> None:
    artifact = read_json("AMENDED_S1_INPUT_VERIFICATION.json")
    checks = artifact["operator_prechecks"]
    assert len(checks) == 20
    assert artifact["operator_precheck_pass_count"] == 20
    assert all(row["pass"] for row in checks)
    assert all(row["only_declared_fields_and_members_changed"] for row in checks)
    assert all(row["gain_bias_synchronized"] and row["positive_parameter_domain"] for row in checks)
    assert all(row["topology_fingerprint_unchanged"] for row in checks)
    assert all(row["compiled_single_state_forward_finite"] and row["solver_warning_count"] == 0 for row in checks)
    assert all(row["multi_trajectory_scientific_response_read"] is False for row in checks)
    identities = [row for row in checks if row["role"] == "NOMINAL_IDENTITY"]
    assert len(identities) == 4
    assert all(row["nominal_identity_bitwise_when_z0"] for row in identities)
    lmax = [row for row in checks if row["factor_id"] == "S1F1_BIARTICULAR_LMAX"]
    assert all(row["lmin_lt_lmax"] for row in lmax)
    assert all(row["lmax_hard_domain_margin_log_units"] is not None for row in lmax)


def test_geometry_only_subset_is_exactly_the_frozen_13_points() -> None:
    rows = read_csv("PILOT_V2_V3_TRAJECTORY_SUBSET.csv")
    assert len(rows) == 13
    assert len({row["candidate_id"] for row in rows}) == 13
    expected_ids = {
        "MYOLEG_V3_K0312", "MYOLEG_V3_K0000", "MYOLEG_V3_K0024",
        "MYOLEG_V3_K0600", "MYOLEG_V3_K0624", "MYOLEG_V3_K0012",
        "MYOLEG_V3_K0612", "MYOLEG_V3_K0300", "MYOLEG_V3_K0324",
        "MYOLEG_V3_K0156", "MYOLEG_V3_K0168", "MYOLEG_V3_K0456",
        "MYOLEG_V3_K0468",
    }
    assert {row["candidate_id"] for row in rows} == expected_ids
    assert all(row["exact_coordinate_match"] == "True" for row in rows)
    assert all(float(row["nearest_grid_distance"]) == 0.0 for row in rows)
    assert all(row["selection_basis"] == "BETA_SPACE_GEOMETRY_ONLY" for row in rows)
    assert all(row["J_or_oracle_or_rank_used"] == "False" for row in rows)
    assert all("j_truth" not in row["source_columns_read"].lower() for row in rows)


def test_response_representations_are_frozen_and_mechanistic() -> None:
    artifact = read_json("PILOT_V2_RESPONSE_REPRESENTATIONS.json")
    rows = artifact["representations"]
    ids = {row["response_id"] for row in rows}
    assert artifact["primary_endpoint"] == "CONFIGURATION_DEPENDENT_NONPROPORTIONAL_MECHANICAL_RESPONSE"
    assert {"HIP_REQUIRED_TORQUE_RMS", "KNEE_REQUIRED_TORQUE_RMS", "FROZEN_COMBINED_J"} <= ids
    assert any(row["response_id"].startswith("S1F1_") for row in rows)
    assert any(row["response_id"].startswith("S1F2_RECTUS") for row in rows)
    assert any(row["response_id"].startswith("S1F2_HAMSTRING") for row in rows)
    assert any(row["response_id"].startswith("S1F3_") for row in rows)
    assert any(row["response_id"].startswith("S1F4_") for row in rows)
    combined = next(row for row in rows if row["response_id"] == "FROZEN_COMBINED_J")
    assert combined["role"] == "SECONDARY_DIAGNOSTIC_ONLY"
    assert combined["not_sole_endpoint"] is True
    assert artifact["gradient_stencil"]["h"] == 0.015
    assert len(artifact["gradient_stencil"]["candidate_ids"]) == 4
    assert artifact["post_outcome_representation_addition_allowed"] is False
    hip_component = next(row for row in rows if row["response_id"].startswith("S1F3_"))
    assert "glmed3_r and piri_r" in hip_component["semantic_limit"]


def test_numeric_nonproportionality_gate_has_effect_resolution() -> None:
    gate = read_json("PILOT_V2_NONPROPORTIONALITY_GATES.json")
    assert gate["frozen_before_scientific_pilot_outcomes"] is True
    assert gate["effect_resolution_gate"]["delta_response_RMS_min_Nm_inclusive"] == 1.0e-5
    assert gate["effect_resolution_gate"]["delta_response_RMS_over_nominal_RMS_min_inclusive"] == 1.0e-4
    assert gate["nonproportionality_thresholds"]["proportional_NRMSE_strictly_above"] == 1.0e-4
    assert gate["nonproportionality_thresholds"]["affine_R2_strictly_below"] == 0.9999
    assert gate["nonproportionality_thresholds"]["both_required_on_same_predeclared_response"] is True
    assert gate["threshold_tuning_after_outcome"] is False


def test_numeric_configuration_and_gradient_gates_are_direction_aware() -> None:
    configuration = read_json("PILOT_V2_CONFIGURATION_DEPENDENCE_GATES.json")
    gradient = read_json("PILOT_V2_GRADIENT_ROTATION_GATES.json")
    assert configuration["thresholds"] == {
        "effect_resolution_gate_from_nonproportionality_required": True,
        "normalized_spread_min_inclusive": 1.0e-4,
        "normalized_range_min_inclusive": 2.0e-4,
        "beta_polynomial_R2_min_inclusive": 0.25,
        "all_required_on_same_predeclared_response": True,
    }
    assert "beta_flex^2" in configuration["beta_dependence_model"]
    assert configuration["scale_like_rule"].startswith("configuration dependence alone is insufficient")
    assert gradient["direction_evidence"]["cosine_similarity_max_inclusive"] == 0.995
    assert math.isclose(gradient["direction_evidence"]["equivalent_minimum_angle_deg"], math.degrees(math.acos(0.995)))
    assert gradient["direction_evidence"]["unit_direction_component_change_max_min_inclusive"] == 0.05
    assert gradient["magnitude_only_gradient_change_is_direction_evidence"] is False
    assert gradient["full_sign_reversal_required"] is False
    assert gradient["classification_role"].startswith("supporting")


def test_integrity_fallback_is_single_and_never_effect_triggered() -> None:
    rules = read_json("PILOT_V2_INTEGRITY_AND_FALLBACK_RULES.json")
    gates = rules["future_replay_gates"]
    assert gates["solver_warning_count_max"] == 0
    assert gates["unexpected_contact_active_count_max"] == 0
    assert gates["tendon_limit_active_count_max"] == 0
    assert gates["truth_decomposition_residual_max_abs_Nm"] == 1.0e-8
    assert gates["sample_count_exact"] == 401
    assert gates["duration_s_exact"] == 24.0
    assert gates["only_declared_model_fields_changed"] is True
    assert rules["maximum_fallback_attempts_per_factor_sign"] == 1
    assert "integrity failure only" in rules["fallback_trigger"]
    assert rules["scientific_small_effect_action"] == "INCONCLUSIVE; fallback prohibited"
    assert rules["fallback_failure_action"].startswith("factor=INVALID_FOR_PILOT")


def test_cohort_v2_admission_is_conjunctive_versioned_and_not_automatic() -> None:
    rules = read_json("PILOT_V2_COHORT_ADMISSION_RULES.json")
    assert rules["future_stage"] == "MYOLEG_VIRTUAL_PATIENT_COHORT_V2_RANGE_AND_DESIGN"
    assert rules["automatic_admission"] is False
    assert len(rules["required_conjunction"]) == 4
    assert rules["classification_if_first_three_pass_but_range_path_missing"] == "COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP"
    assert set(rules["current_population_ranges"].values()) == {"NOT_AVAILABLE"}
    assert rules["new_versioned_split_required"] is True
    assert rules["old_v1_heldout_is_automatic_cohort_v2_confirmation"] is False
    assert rules["old_v1_heldout_scientific_access_count"] == 0
    relationships = {row["entity"]: row["relationship"] for row in rules["inherited_v1_background_relationships"]}
    assert relationships["FEMUR_MASS_INERTIA_SCALE"] == "RETAIN_AS_BACKGROUND"
    assert relationships["HIP_ONLY_PASSIVE_FP_MAX_SCALE"] == "SECONDARY_ONLY"
    assert relationships["HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE"] == "REMOVE_FROM_PERSONALIZATION_FOCUSED_COHORT"
    assert rules["produces_personalization_is_a_valid_pilot_conclusion"] is False


def test_exact_future_workload_is_small_and_nothing_was_executed() -> None:
    plan = read_json("PILOT_V2_EXECUTION_PLAN.json")
    metadata = read_json("metadata.json")
    assert plan["status"] == "READY_FOR_SEPARATE_FUTURE_EXECUTION"
    assert plan["model_structure"] == "ONE_FACTOR_AT_A_TIME"
    assert plan["model_label"] == "STRUCTURAL_DIAGNOSTIC_MODEL"
    assert plan["virtual_patient_label_prohibited"] is True
    assert plan["primary_structural_diagnostic_model_count"] == 9
    assert plan["trajectories_per_model"] == 13
    assert plan["expected_primary_replay_count"] == 117
    assert plan["maximum_optional_fallback_model_count"] == 8
    assert plan["maximum_optional_fallback_replay_count"] == 104
    assert plan["maximum_total_unique_model_count_including_fallback"] == 17
    assert plan["maximum_total_replay_count_including_fallback"] == 221
    assert plan["scientific_pilot_executed"] is False
    assert plan["automatic_execution_from_design_builder"] is False
    assert metadata["outcome"] == design.OUTCOME_GAPS
    assert metadata["scientific_models_generated"] == 0
    assert metadata["scientific_trajectory_replays"] == 0
    assert metadata["held_out_scientific_access_count"] == 0
    assert metadata["analysis_code_sha256"] == sha256(Path(design.__file__))


def test_protocol_and_report_answer_all_questions_and_stop() -> None:
    protocol = read_json("STRUCTURAL_HETEROGENEITY_PILOT_V2_PROTOCOL.json")
    report = (OUT / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_REPORT.md").read_text(encoding="utf-8")
    assert protocol["formal_outcome"] == design.OUTCOME_GAPS
    assert protocol["future_pilot_ready"] is True
    assert protocol["future_pilot_must_be_separately_invoked"] is True
    assert protocol["this_stage_scientific_pilot_execution_authorized"] is False
    guards = protocol["scope_guards"]
    assert guards["scientific_trajectory_pilot_executed"] is False
    assert guards["multi_trajectory_torque_response_read"] is False
    assert guards["J_or_oracle_or_rank_used_for_design"] is False
    assert guards["held_out_scientific_access_count"] == 0
    assert guards["cohort_v2_generated"] is False
    assert guards["new_truth_landscape_generated"] is False
    assert guards["five_parameter_or_NN_or_PINN_or_BO_run"] is False
    assert all(f"## Q{index}." in report for index in range(1, 11))
    assert "Scientific models/replays executed now: **0 / 0**" in report
    assert "Held-out scientific access: 0" in report


def test_required_artifacts_and_checksums_are_complete() -> None:
    required = {
        "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_REPORT.md",
        "STRUCTURAL_HETEROGENEITY_PILOT_V2_PROTOCOL.json",
        "AMENDED_S1_INPUT_VERIFICATION.json",
        "PILOT_V2_DIAGNOSTIC_LEVELS.json",
        "PILOT_V2_V3_TRAJECTORY_SUBSET.csv",
        "PILOT_V2_RESPONSE_REPRESENTATIONS.json",
        "PILOT_V2_NONPROPORTIONALITY_GATES.json",
        "PILOT_V2_CONFIGURATION_DEPENDENCE_GATES.json",
        "PILOT_V2_GRADIENT_ROTATION_GATES.json",
        "PILOT_V2_INTEGRITY_AND_FALLBACK_RULES.json",
        "PILOT_V2_COHORT_ADMISSION_RULES.json",
        "PILOT_V2_EXECUTION_PLAN.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert required == {path.name for path in OUT.iterdir() if path.is_file()}
    entries = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual)
    assert all(sha256(OUT / relative) == expected for relative, expected in entries.items())


def test_builder_has_no_scientific_replay_learner_optimizer_or_robot_path() -> None:
    source = Path(design.__file__).read_text(encoding="utf-8")
    assert "prescribed_truth(" not in source
    assert "compact_replay(" not in source
    assert "replay_v3_subject_candidate(" not in source
    assert "import hardware" not in source
    assert "import control" not in source
    assert "import torch" not in source
    assert "sklearn" not in source
    assert "held_out_landscape" not in source.lower()

