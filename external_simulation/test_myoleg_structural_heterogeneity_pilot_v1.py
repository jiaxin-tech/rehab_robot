"""Regression gates for the frozen amended-S1 structural diagnostic pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from external_simulation.myoleg_structural_heterogeneity_pilot_v1 import run_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_v1"


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


def test_every_frozen_design_and_scientific_input_sha_is_exact() -> None:
    assert {name: sha256(path) for name, path in pilot.FROZEN_PATHS.items()} == pilot.FROZEN_SHA
    verification = read_json("PILOT_INPUT_VERIFICATION.json")
    assert verification["status"] == "PASS"
    assert verification["verified_before_scientific_replay"] is True
    assert verification["frozen_input_sha256"] == pilot.FROZEN_SHA
    assert verification["authoritative_factor_ids"] == list(pilot.FACTOR_IDS)
    assert verification["design_artifact_count"] == 14
    assert verification["design_checksum_entry_count"] == 13
    assert verification["held_out_scientific_access_count"] == 0


def test_execution_protocol_was_frozen_and_unchanged_before_replay() -> None:
    protocol = read_json("STRUCTURAL_HETEROGENEITY_PILOT_EXECUTION_PROTOCOL.json")
    verification = read_json("PILOT_INPUT_VERIFICATION.json")
    assert sha256(OUT / "STRUCTURAL_HETEROGENEITY_PILOT_EXECUTION_PROTOCOL.json") == verification["execution_protocol_sha256"]
    assert protocol["frozen_before_first_scientific_structural_replay"] is True
    assert protocol["authoritative_definition_sha256"] == pilot.FROZEN_SHA["amended_s1"]
    assert protocol["pilot_design_protocol_sha256"] == pilot.FROZEN_SHA["design_protocol"]
    assert protocol["model_plan"]["primary_model_count"] == 9
    assert protocol["model_plan"]["primary_replay_count"] == 117
    assert protocol["scope_guards"]["candidate_scientific_replay_count"] == 13
    assert protocol["scope_guards"]["625_grid_scientific_search"] is False
    assert protocol["sign_symmetry"]["role"] == "DESCRIPTIVE_ONLY_NO_PREREGISTERED_SYMMETRY_THRESHOLD"
    assert protocol["next_stage_auto_execution"] is False


def test_exact_levels_subset_responses_and_gates_are_inherited() -> None:
    protocol = read_json("STRUCTURAL_HETEROGENEITY_PILOT_EXECUTION_PROTOCOL.json")
    levels = protocol["factor_levels"]
    assert [row["primary_positive_z"] for row in levels] == [0.01, 0.025, 0.025, 0.025]
    assert [row["fallback_positive_z"] for row in levels] == [0.005, 0.0125, 0.0125, 0.0125]
    assert all(row["population_range"] == "NOT_AVAILABLE" for row in levels)
    assert len(protocol["candidate_subset"]) == 13
    assert len({row["candidate_id"] for row in protocol["candidate_subset"]}) == 13
    assert all(row["selection_basis"] == "BETA_SPACE_GEOMETRY_ONLY" for row in protocol["candidate_subset"])
    response_ids = {row["response_id"] for row in protocol["response_representations"]["representations"]}
    assert {"HIP_REQUIRED_TORQUE_RMS", "KNEE_REQUIRED_TORQUE_RMS", "FROZEN_COMBINED_J"} <= response_ids
    nonprop = protocol["nonproportionality_gate"]
    configuration = protocol["configuration_dependence_gate"]
    gradient = protocol["gradient_rotation_gate"]
    assert nonprop["effect_resolution_gate"]["delta_response_RMS_min_Nm_inclusive"] == 1.0e-5
    assert nonprop["effect_resolution_gate"]["delta_response_RMS_over_nominal_RMS_min_inclusive"] == 1.0e-4
    assert nonprop["nonproportionality_thresholds"]["proportional_NRMSE_strictly_above"] == 1.0e-4
    assert nonprop["nonproportionality_thresholds"]["affine_R2_strictly_below"] == 0.9999
    assert configuration["thresholds"]["normalized_spread_min_inclusive"] == 1.0e-4
    assert configuration["thresholds"]["normalized_range_min_inclusive"] == 2.0e-4
    assert configuration["thresholds"]["beta_polynomial_R2_min_inclusive"] == 0.25
    assert gradient["direction_evidence"]["cosine_similarity_max_inclusive"] == 0.995
    assert gradient["direction_evidence"]["unit_direction_component_change_max_min_inclusive"] == 0.05


def test_all_nine_primary_models_and_117_replays_pass_integrity_without_fallback() -> None:
    models = read_csv("PILOT_MODEL_INTEGRITY_RESULTS.csv")
    replays = read_csv("PILOT_REPLAY_RESULTS.csv")
    fallback = read_json("PILOT_FALLBACK_USAGE_AUDIT.json")
    metadata = read_json("metadata.json")
    assert len(models) == 9
    assert all(row["model_label"] == "STRUCTURAL_DIAGNOSTIC_MODEL" for row in models)
    assert all(row["operator_precheck_pass"] == "True" for row in models)
    assert all(row["model_integrity_pass"] == "True" for row in models)
    assert all(row["only_declared_fields_changed"] == "True" for row in models)
    assert all(row["gain_bias_synchronized"] == "True" for row in models)
    assert all(row["topology_unchanged"] == "True" for row in models)
    assert len(replays) == 117
    assert all(row["integrity_pass"] == "True" for row in replays)
    assert len({row["model_id"] for row in replays}) == 9
    assert all(sum(item["model_id"] == row["model_id"] for item in replays) == 13 for row in models)
    assert fallback["fallback_model_count"] == 0
    assert fallback["fallback_replay_count"] == 0
    assert fallback["events"] == []
    assert fallback["fallback_triggered_by_small_effect_or_failed_scientific_gate"] is False
    assert fallback["larger_perturbation_attempted"] is False
    assert metadata["primary_model_integrity_pass_count"] == 9
    assert metadata["actual_replay_count"] == 117


def test_replay_integrity_values_satisfy_every_frozen_bound() -> None:
    rows = read_csv("PILOT_REPLAY_RESULTS.csv")
    gates = read_json("STRUCTURAL_HETEROGENEITY_PILOT_EXECUTION_PROTOCOL.json")["integrity_and_fallback_rules"]["future_replay_gates"]
    assert max(int(row["solver_warning_count"]) for row in rows) == 0
    assert max(float(row["source_equality_residual_max"]) for row in rows) <= gates["source_equality_residual_max"]
    assert max(int(row["joint_limit_active_count_max"]) for row in rows) <= gates["joint_limit_active_count_max"]
    assert max(int(row["tendon_limit_active_count_max"]) for row in rows) == 0
    assert max(int(row["contact_active_count_max"]) for row in rows) == 0
    assert max(float(row["joint_limit_contribution_max_abs_Nm"]) for row in rows) <= gates["joint_limit_contribution_max_abs_Nm"]
    assert max(float(row["joint_limit_contribution_max_relative"]) for row in rows) <= gates["joint_limit_contribution_max_relative"]
    assert max(float(row["truth_decomposition_residual_max_abs_Nm"]) for row in rows) <= gates["truth_decomposition_residual_max_abs_Nm"]
    assert max(float(row["V3_extrema_ROM_error_max_deg"]) for row in rows) <= gates["V3_extrema_ROM_error_max_deg"]
    assert all(row["branch_anchor_C2_preserved"] == "True" for row in rows)
    references = [row for row in rows if row["candidate_id"] == "MYOLEG_V3_K0312"]
    assert len(references) == 9
    assert all(abs(float(row["FROZEN_COMBINED_J"]) - 1.0) <= 1.0e-12 for row in references)


def test_all_preregistered_factor_response_rows_exist() -> None:
    nonprop = read_csv("PILOT_NONPROPORTIONALITY_RESULTS.csv")
    configuration = read_csv("PILOT_CONFIGURATION_DEPENDENCE_RESULTS.csv")
    gradient = read_csv("PILOT_GRADIENT_ROTATION_RESULTS.csv")
    mechanistic = read_csv("PILOT_FACTOR_MECHANISTIC_RESULTS.csv")
    identities = lambda rows: {(row["factor_id"], row["sign"], row["response_id"]) for row in rows}
    assert len(nonprop) == len(configuration) == len(gradient) == len(mechanistic) == 36
    assert identities(nonprop) == identities(configuration) == identities(gradient) == identities(mechanistic)
    assert {row["factor_id"] for row in nonprop} == set(pilot.FACTOR_IDS)
    assert {row["sign"] for row in nonprop} == {"-1", "1"}
    assert all(int(row["candidate_count"]) == 13 for row in nonprop)
    assert all(row["sign_symmetry_role"] == "DESCRIPTIVE_ONLY_NO_PREREGISTERED_SYMMETRY_THRESHOLD" for row in mechanistic)
    assert all(row["sign_result_used_to_change_factor_semantics"] == "False" for row in mechanistic)


def test_one_fit_and_configuration_row_recomputes_from_replay_results() -> None:
    replays = read_csv("PILOT_REPLAY_RESULTS.csv")
    metrics = read_csv("PILOT_NONPROPORTIONALITY_RESULTS.csv")
    configurations = read_csv("PILOT_CONFIGURATION_DEPENDENCE_RESULTS.csv")
    target = next(row for row in metrics if row["factor_id"] == "S1F1_BIARTICULAR_LMAX" and row["sign"] == "-1" and row["response_id"] == "HIP_REQUIRED_TORQUE_RMS")
    config = next(row for row in configurations if row["factor_id"] == target["factor_id"] and row["sign"] == target["sign"] and row["response_id"] == target["response_id"])
    nominal = sorted((row for row in replays if row["model_id"] == "STRUCTURAL_DIAGNOSTIC_NOMINAL"), key=lambda row: int(row["selection_order"]))
    changed = sorted((row for row in replays if row["model_id"] == target["model_id"]), key=lambda row: int(row["selection_order"]))
    x = np.asarray([float(row[target["response_id"]]) for row in nominal])
    y = np.asarray([float(row[target["response_id"]]) for row in changed])
    scale = float(np.sqrt(np.mean(x**2)))
    a = float(np.dot(x, y) / np.dot(x, x))
    residual = y - a * x
    assert math.isclose(float(target["proportional_a"]), a, rel_tol=1.0e-12, abs_tol=1.0e-15)
    assert math.isclose(float(target["proportional_NRMSE"]), float(np.sqrt(np.mean(residual**2)) / scale), rel_tol=1.0e-12, abs_tol=1.0e-15)
    delta = y - x
    beta = np.asarray([[float(row["beta_flex"]), float(row["beta_extend"])] for row in nominal])
    matrix = np.column_stack((np.ones(13), beta[:, 0], beta[:, 1], beta[:, 0] ** 2, beta[:, 1] ** 2, beta[:, 0] * beta[:, 1]))
    coefficients = np.linalg.lstsq(matrix, delta, rcond=None)[0]
    fitted = matrix @ coefficients
    r2 = 1.0 - float(np.sum((delta - fitted) ** 2)) / float(np.sum((delta - np.mean(delta)) ** 2))
    assert math.isclose(float(config["normalized_spread"]), float(np.std(delta) / scale), rel_tol=1.0e-12, abs_tol=1.0e-15)
    assert math.isclose(float(config["beta_polynomial_R2"]), r2, rel_tol=1.0e-12, abs_tol=1.0e-15)


def test_negative_result_is_preserved_under_frozen_gates() -> None:
    nonprop = read_csv("PILOT_NONPROPORTIONALITY_RESULTS.csv")
    gradient = read_csv("PILOT_GRADIENT_ROTATION_RESULTS.csv")
    classifications = read_csv("PILOT_FACTOR_CLASSIFICATION.csv")
    decision = read_json("FINAL_PILOT_DECISION.json")
    assert all(row["nonproportionality_gate_pass"] == "False" for row in nonprop)
    assert all(row["gradient_rotation_gate_pass"] == "False" for row in gradient)
    assert [row["classification"] for row in classifications] == ["MAGNITUDE_ONLY"] * 4
    assert all(row["integrity"] == "PASS" for row in classifications)
    assert all(row["negative_factor_sign_structural_pass"] == "False" for row in classifications)
    assert all(row["positive_factor_sign_structural_pass"] == "False" for row in classifications)
    assert decision["primary_decision"] == "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED"
    assert decision["structurally_informative_factor_count"] == 0
    assert decision["classification_counts"]["MAGNITUDE_ONLY"] == 4
    assert decision["larger_z_retry_recommended"] is False
    assert decision["recommended_independent_next_stage"] == "MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_AUDIT_V1"
    assert decision["next_stage_executed"] is False


def test_no_factor_is_admitted_and_population_gap_is_not_hidden() -> None:
    rows = read_csv("PILOT_COHORT_V2_ADMISSION_RESULTS.csv")
    metadata = read_json("metadata.json")
    assert len(rows) == 4
    assert all(row["admission_result"] == "NOT_ELIGIBLE_MAGNITUDE_ONLY" for row in rows)
    assert all(row["population_range"] == "NOT_AVAILABLE" for row in rows)
    assert all(row["population_range_calibration_pathway_defensible_now"] == "False" for row in rows)
    assert all(row["cohort_v2_generated"] == "False" for row in rows)
    assert all(row["synthetic_diagnostic_z_promoted_to_population_bound"] == "False" for row in rows)
    assert metadata["new_virtual_subjects"] == 0
    assert metadata["cohort_v2_generated"] is False
    assert metadata["population_ranges_available"] is False


def test_held_out_oracle_learner_optimizer_and_robot_boundaries_remain_closed() -> None:
    held_out = read_json("HELD_OUT_ACCESS_AUDIT.json")
    metadata = read_json("metadata.json")
    assert held_out["cohort_v1_manifest_sha256"] == pilot.FROZEN_SHA["cohort_v1_manifest"]
    assert held_out["held_out_metadata_or_model_or_truth_loaded"] is False
    assert held_out["held_out_replay_count"] == 0
    assert held_out["held_out_J_tau_oracle_rank_access_count"] == 0
    assert held_out["held_out_scientific_access_count"] == 0
    assert metadata["625_grid_scientific_search"] is False
    assert metadata["oracle_or_rank_or_regret_computed"] is False
    assert metadata["objective_or_normalization_modified"] is False
    assert metadata["S1_or_V3_or_V1_cohort_modified"] is False
    assert metadata["five_parameter_or_NN_or_PINN_or_BO_run"] is False
    assert metadata["held_out_scientific_access_count"] == 0
    assert metadata["robot_or_hardware"] is False


def test_all_eight_preregistered_figures_exist_and_are_valid_png() -> None:
    expected = {
        *(f"{factor_id}_nominal_vs_perturbed.png" for factor_id in pilot.FACTOR_IDS),
        "delta_response_across_13_beta_locations.png", "proportional_fit_residuals.png",
        "gradient_vectors.png", "factor_summary.png",
    }
    actual = {path.name for path in (OUT / "figures").glob("*.png")}
    assert actual == expected
    for name in expected:
        path = OUT / "figures" / name
        assert path.stat().st_size > 10_000
        with Image.open(path) as image:
            image.verify()


def test_report_answers_all_questions_and_stops() -> None:
    report = (OUT / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_REPORT.md").read_text(encoding="utf-8")
    assert "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED" in report
    assert all(f"## Q{token}." in report for token in ("1", "2", "3-Q7", "8", "9", "10"))
    assert "Primary integrity PASS: **9/9**" in report
    assert "Fallback models used: **0**" in report
    assert "Actual total replay count: **117**" in report
    assert "MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_AUDIT_V1" in report
    assert "The next stage was not executed automatically" in report


def test_checksums_cover_every_artifact_and_figure() -> None:
    entries = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert len(actual) == 23
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual)
    assert all(sha256(OUT / relative) == expected for relative, expected in entries.items())


def test_runner_has_no_subject_cohort_optimizer_or_robot_execution_path() -> None:
    source = Path(pilot.__file__).read_text(encoding="utf-8")
    assert "model_from_record(" not in source
    assert "worker_model(" not in source
    assert "replay_v3_subject_candidate(" not in source
    assert "import torch" not in source
    assert "import hardware" not in source
    assert "import control" not in source
    assert "bayesian" not in source.lower()
    assert "generate_cohort" not in source

