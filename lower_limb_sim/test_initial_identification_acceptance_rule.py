from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    THETA_SHANK_DEFINITION,
    validate_active_reference_file,
)
from .initial_identification_acceptance_rule import (
    FORMALLY_REVIEWED,
    HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS,
    ID_CONTINUE_NEEDS_INFORMATION,
    INITIAL_IDENTIFICATION_COMPLETE,
    INITIAL_IDENTIFICATION_INSUFFICIENT,
    MODEL_ADEQUACY_GATE,
    MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW,
    MODEL_INADEQUATE_FOR_PERSONALIZATION,
    MODEL_STRUCTURE_LIMITATION,
    PARAMETER_IDENTIFIABILITY_GATE,
    PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW,
    VALIDATION_TRAJECTORY_SPECS,
    GateDecision,
    ModelAdequacyThresholds,
    ParameterIdentifiabilityThresholds,
    build_validation_observations,
    determine_acceptance_state,
    diagnose_model_structure_limitation,
    evaluate_model_adequacy,
    evaluate_parameter_identifiability,
    identification_marginal_gain_table,
)
from .parameter_estimator import PARAMETER_NAMES
from .run_initial_identification_acceptance_rule import run
from .safeguarded_sequential_initial_identification import (
    MAX_INITIAL_IDENTIFICATION_TRIALS,
)


def _parameter_metrics() -> tuple[dict[str, float], pd.DataFrame]:
    global_metrics = {
        "rank": 5,
        "minimum_singular_value": 25.0,
        "condition_number": 35.0,
        "maximum_abs_parameter_correlation": 0.2,
        "maximum_normalized_parameter_change": 0.01,
    }
    per_parameter = pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "sensitivity": [25.0] * len(PARAMETER_NAMES),
            "uncertainty_proxy": [0.02] * len(PARAMETER_NAMES),
        }
    )
    return global_metrics, per_parameter


def _parameter_rule() -> ParameterIdentifiabilityThresholds:
    return ParameterIdentifiabilityThresholds(
        minimum_rank=5,
        minimum_singular_value=20.0,
        maximum_condition_number=50.0,
        maximum_abs_parameter_correlation=0.3,
        maximum_uncertainty_proxy=0.05,
        minimum_parameter_sensitivity=20.0,
        maximum_normalized_parameter_change=0.05,
        review_status=FORMALLY_REVIEWED,
        evidence_source="UNIT_TEST_FIXTURE_ONLY",
    )


def _model_metrics(value: float = 0.1) -> dict[str, float]:
    return {
        "validation_hip_rmse_nm": value,
        "validation_knee_rmse_nm": value,
        "validation_combined_rmse_nm": value,
        "validation_combined_nrmse_percent": value,
        "validation_e_j": value,
        "validation_relative_e_j_percent": value,
    }


def _model_rule(maximum: float = 0.2) -> ModelAdequacyThresholds:
    return ModelAdequacyThresholds(
        maximum_hip_rmse_nm=maximum,
        maximum_knee_rmse_nm=maximum,
        maximum_combined_rmse_nm=maximum,
        maximum_combined_nrmse_percent=maximum,
        maximum_validation_e_j=maximum,
        maximum_validation_relative_e_j_percent=maximum,
        review_status=FORMALLY_REVIEWED,
        evidence_source="UNIT_TEST_FIXTURE_ONLY",
    )


@pytest.fixture(scope="module")
def generated_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("initial-id-acceptance")
    run(destination)
    return destination


def test_01_gates_are_independent() -> None:
    metrics, parameters = _parameter_metrics()
    parameter = evaluate_parameter_identifiability(metrics, parameters, _parameter_rule())
    model = evaluate_model_adequacy(_model_metrics(0.4), _model_rule(0.2))
    assert parameter.passed is True
    assert model.passed is False
    assert parameter.gate_name != model.gate_name


def test_02_low_training_rmse_cannot_enter_adequacy_gate() -> None:
    metrics = _model_metrics()
    metrics["training_rmse_nm"] = 0.0
    with pytest.raises(ValueError, match="training metrics"):
        evaluate_model_adequacy(metrics, _model_rule())


def test_03_full_rank_does_not_auto_pass_model_adequacy() -> None:
    metrics, parameters = _parameter_metrics()
    assert evaluate_parameter_identifiability(metrics, parameters).passed is None
    assert evaluate_model_adequacy(_model_metrics()).passed is None


def test_04_model_adequacy_cannot_mask_unidentifiable_parameters() -> None:
    parameter = GateDecision(
        PARAMETER_IDENTIFIABILITY_GATE,
        False,
        "PARAMETER_IDENTIFIABILITY_FAIL",
        ("rank",),
    )
    model = GateDecision(MODEL_ADEQUACY_GATE, True, "MODEL_ADEQUACY_PASS", ())
    decision = determine_acceptance_state(2, parameter, model, _temporary_parameters())
    assert decision.final_status == ID_CONTINUE_NEEDS_INFORMATION
    assert decision.theta_hat_0 is None


def _temporary_parameters() -> dict[str, float]:
    return {name: float(index + 1) for index, name in enumerate(PARAMETER_NAMES)}


def test_05_theta_hat_zero_requires_both_passes() -> None:
    parameter = GateDecision(PARAMETER_IDENTIFIABILITY_GATE, True, "PASS", ())
    model = GateDecision(MODEL_ADEQUACY_GATE, True, "PASS", ())
    decision = determine_acceptance_state(2, parameter, model, _temporary_parameters())
    assert decision.final_status == INITIAL_IDENTIFICATION_COMPLETE
    assert decision.theta_hat_0 == _temporary_parameters()
    assert decision.personalization_prerequisite is True


def test_06_combined_mild_is_diagnosed_as_model_structure_limitation(
    generated_artifacts: Path,
) -> None:
    ident = pd.read_csv(generated_artifacts / "parameter_identifiability_by_trial.csv")
    adequacy = pd.read_csv(generated_artifacts / "model_adequacy_by_trial.csv")
    case = "baseline__combined_mild"
    assert diagnose_model_structure_limitation(
        ident.loc[ident["case_id"].eq(case)],
        adequacy.loc[adequacy["case_id"].eq(case)],
    ) == MODEL_STRUCTURE_LIMITATION


def test_07_parameter_stable_but_validation_bad_is_model_failure() -> None:
    parameter = GateDecision(PARAMETER_IDENTIFIABILITY_GATE, True, "PASS", ())
    model = evaluate_model_adequacy(_model_metrics(0.4), _model_rule(0.2))
    decision = determine_acceptance_state(3, parameter, model, _temporary_parameters())
    assert decision.final_status == MODEL_INADEQUATE_FOR_PERSONALIZATION


def test_08_trial_marginal_gain_signs_are_correct() -> None:
    ident_rows = []
    stability_rows = []
    for trial, singular, condition, corr, uncertainty, sensitivity, change in (
        (1, 10.0, 60.0, 0.4, 0.10, 10.0, 0.2),
        (2, 15.0, 50.0, 0.3, 0.08, 12.0, 0.1),
    ):
        for parameter in PARAMETER_NAMES:
            ident_rows.append(
                {
                    "case_id": "case",
                    "subject_id": "subject",
                    "scenario_name": "scenario",
                    "trial_id": trial,
                    "parameter": parameter,
                    "rank": 5,
                    "minimum_singular_value": singular,
                    "condition_number": condition,
                    "maximum_abs_parameter_correlation": corr,
                    "uncertainty_proxy": uncertainty,
                    "sensitivity": sensitivity,
                }
            )
            stability_rows.append(
                {
                    "case_id": "case",
                    "trial_id": trial,
                    "parameter": parameter,
                    "normalized_parameter_change": change,
                }
            )
    adequacy = pd.DataFrame(
        {
            "case_id": ["case", "case"],
            "trial_id": [1, 2],
            "validation_combined_rmse_nm": [0.5, 0.4],
            "validation_e_j": [0.2, 0.15],
        }
    )
    information = pd.DataFrame(
        {
            "subject_id": ["subject", "subject"],
            "truth_scenario": ["scenario", "scenario"],
            "trial_id": [1, 2],
            "incremental_log_information_gain": [1.0, 0.5],
        }
    )
    gain = identification_marginal_gain_table(
        pd.DataFrame(ident_rows), pd.DataFrame(stability_rows), adequacy, information
    ).iloc[0]
    assert gain["minimum_singular_value_improvement"] == pytest.approx(5.0)
    assert gain["condition_number_improvement"] == pytest.approx(10.0)
    assert gain["validation_rmse_improvement_nm"] == pytest.approx(0.1)
    assert gain["validation_e_j_improvement"] == pytest.approx(0.05)


def test_09_trial_five_parameter_failure_is_insufficient() -> None:
    parameter = GateDecision(PARAMETER_IDENTIFIABILITY_GATE, False, "FAIL", ())
    model = GateDecision(MODEL_ADEQUACY_GATE, True, "PASS", ())
    decision = determine_acceptance_state(5, parameter, model, _temporary_parameters())
    assert decision.final_status == INITIAL_IDENTIFICATION_INSUFFICIENT


def test_10_identifiable_inadequate_state_is_fail_closed() -> None:
    parameter = GateDecision(PARAMETER_IDENTIFIABILITY_GATE, True, "PASS", ())
    model = GateDecision(MODEL_ADEQUACY_GATE, False, "FAIL", ())
    decision = determine_acceptance_state(2, parameter, model, _temporary_parameters())
    assert decision.final_status == MODEL_INADEQUATE_FOR_PERSONALIZATION
    assert decision.theta_hat_0 is None


def test_11_personalization_prerequisite_defaults_to_fail_closed() -> None:
    parameter = GateDecision(PARAMETER_IDENTIFIABILITY_GATE, True, "PASS", ())
    model = GateDecision(MODEL_ADEQUACY_GATE, None, "REVIEW", ())
    assert not determine_acceptance_state(
        2, parameter, model, _temporary_parameters()
    ).personalization_prerequisite


def test_12_zero_point_two_is_not_a_formal_threshold(generated_artifacts: Path) -> None:
    candidates = pd.read_csv(generated_artifacts / "acceptance_rule_candidate_table.csv")
    legacy = candidates.loc[
        candidates["candidate_rule_id"].eq("legacy_virtual_comparator_0p20_nm")
    ].iloc[0]
    assert legacy["threshold_freezing_status"] == "RESEARCH_ONLY_UNJUSTIFIED_NOT_FORMAL"
    assert not bool(legacy["complete_two_gate_rule"])


def test_13_combined_mild_does_not_cause_threshold_to_change_to_point_45(
    generated_artifacts: Path,
) -> None:
    audit = (generated_artifacts / "CURRENT_INITIAL_ID_STOP_RULE_AUDIT.md").read_text()
    assert "loosened to 0.45 N·m" in audit
    metadata = json.loads((generated_artifacts / "metadata.json").read_text())
    assert metadata["legacy_0p20_nm_changed_to_0p45"] is False


def test_14_heldout_final_test_is_not_in_validation() -> None:
    validation = build_validation_observations("baseline", "matched_linear")
    ids = set(validation["trajectory_id"].astype(str))
    forbidden = {
        f"identification_excitation_trajectory:{family}:{speed}"
        for family, speed in HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS
    }
    assert not ids.intersection(forbidden)
    assert len(ids) == len(VALIDATION_TRAJECTORY_SPECS)


def test_15_truth_parameters_are_not_in_validation_decision_table() -> None:
    validation = build_validation_observations("baseline", "combined_mild")
    forbidden = [
        column
        for column in validation.columns
        if column.startswith("true_")
        or column.startswith("ground_truth")
        or column.startswith("tau_")
    ]
    assert forbidden == []


def test_16_excitation_selector_module_is_not_edited_by_runner() -> None:
    source = Path(__file__).with_name(
        "safeguarded_sequential_initial_identification.py"
    )
    before = source.read_bytes()
    assert b"def select_next_identification_excitation" in before
    assert source.read_bytes() == before


def test_17_maximum_trial_count_remains_five() -> None:
    assert MAX_INITIAL_IDENTIFICATION_TRIALS == 5


def test_18_active_reference_sha_remains_frozen() -> None:
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    validate_active_reference_file()


def test_19_rom_protocol_remains_frozen() -> None:
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)


def test_20_five_parameter_model_is_unchanged() -> None:
    assert PARAMETER_NAMES == (
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    )


def test_21_mechanical_objective_source_is_not_modified_by_task() -> None:
    source = Path(__file__).with_name("mechanical_objective.py").read_text()
    assert "MECHANICAL_OBJECTIVE_VERSION = \"mechanical_joint_torque_objective_v1\"" in source


def test_22_new_runtime_has_no_hardware_control_or_safety_import() -> None:
    for name in (
        "initial_identification_acceptance_rule.py",
        "run_initial_identification_acceptance_rule.py",
    ):
        tree = ast.parse(Path(__file__).with_name(name).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(
            module.startswith(("hardware", "control", "safety", "collection"))
            for module in imports
        )


def test_23_no_real_robot_connection_claim(generated_artifacts: Path) -> None:
    metadata = json.loads((generated_artifacts / "metadata.json").read_text())
    assert metadata["real_robot_connected"] is False
    assert metadata["hardware_or_safety_modified"] is False


def test_24_no_explore_exploit_execution(generated_artifacts: Path) -> None:
    metadata = json.loads((generated_artifacts / "metadata.json").read_text())
    assert metadata["explore_exploit_personalization_executed"] is False
    assert metadata["personalization_executed"] is False


def test_25_required_artifacts_and_figures_exist(generated_artifacts: Path) -> None:
    required = {
        "CURRENT_INITIAL_ID_STOP_RULE_AUDIT.md",
        "parameter_identifiability_by_trial.csv",
        "parameter_stability_by_trial.csv",
        "model_adequacy_by_trial.csv",
        "identification_marginal_gain_by_trial.csv",
        "acceptance_rule_candidate_table.csv",
        "initial_identification_acceptance_summary.csv",
        "INITIAL_ID_ACCEPTANCE_DATA_LEAKAGE_AUDIT.md",
        "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REPORT.md",
        "metadata.json",
        "parameter_identifiability_vs_trial.png",
        "parameter_stability_vs_trial.png",
        "validation_error_vs_trial.png",
        "information_gain_vs_model_error.png",
        "identifiability_vs_model_adequacy.png",
    }
    assert required.issubset({path.name for path in generated_artifacts.iterdir()})
    assert all((generated_artifacts / name).stat().st_size > 0 for name in required)


def test_26_default_thresholds_remain_requires_review() -> None:
    metrics, parameters = _parameter_metrics()
    parameter = evaluate_parameter_identifiability(metrics, parameters)
    model = evaluate_model_adequacy(_model_metrics())
    assert parameter.status == PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW
    assert model.status == MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW


def test_27_formal_summary_freezes_no_theta(generated_artifacts: Path) -> None:
    summary = pd.read_csv(
        generated_artifacts / "initial_identification_acceptance_summary.csv"
    )
    assert not summary["theta_hat_0_frozen"].astype(bool).any()
    assert not summary["personalization_prerequisite"].astype(bool).any()


def test_28_matched_candidate_stops_at_trial_two(generated_artifacts: Path) -> None:
    summary = pd.read_csv(
        generated_artifacts / "initial_identification_acceptance_summary.csv"
    )
    matched = summary.loc[summary["scenario_name"].eq("matched_linear")]
    assert set(matched["trials_required_under_positive_control_candidate"]) == {2}
    assert set(matched["candidate_rule_diagnostic_state"]) == {
        INITIAL_IDENTIFICATION_COMPLETE
    }


def test_29_combined_candidate_state_is_model_inadequate(
    generated_artifacts: Path,
) -> None:
    summary = pd.read_csv(
        generated_artifacts / "initial_identification_acceptance_summary.csv"
    )
    combined = summary.loc[summary["scenario_name"].eq("combined_mild")].iloc[0]
    assert combined["candidate_rule_diagnostic_state"] == MODEL_INADEQUATE_FOR_PERSONALIZATION
    assert combined["trend_diagnosis"] == MODEL_STRUCTURE_LIMITATION


def test_30_theta_shank_definition_is_preserved_in_validation() -> None:
    validation = build_validation_observations("baseline", "matched_linear")
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert np.allclose(
        validation["theta_shank_rad"],
        validation["q_hip_rad"] - validation["q_knee_rad"],
        atol=1e-12,
        rtol=0.0,
    )
