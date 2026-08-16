from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.continuous_reference_neighborhood import TOTAL_DURATION_S
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from lower_limb_sim.run_safeguarded_sequential_initial_identification import (
    VIRTUAL_RESEARCH_COMPARATOR_RULE,
    build_full_prediction_map,
    generate_formal_artifacts,
)
from lower_limb_sim.safeguarded_sequential_initial_identification import (
    AUTO_EXPAND_PATIENT_ENVELOPE,
    IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW,
    INITIAL_IDENTIFICATION_COMPLETE,
    INITIAL_IDENTIFICATION_INSUFFICIENT,
    MAX_INITIAL_IDENTIFICATION_TRIALS,
    REAL_ROBOT_HARD_SAFEGUARD,
    RESEARCH_DURATION_LABEL,
    SUPPORTED_PREDICTION,
    UNSUPPORTED_EXTRAPOLATION,
    IdentificationExcitationSpec,
    ResearchIdentifiabilityStopRule,
    VirtualIdentificationOracle,
    default_virtual_patient_envelope,
    default_virtual_research_candidate_pool,
    generate_identification_excitation,
    limited_rom_virtual_patient_envelope,
    run_sequential_initial_identification,
    select_next_identification_excitation,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
)


def _rule(minimum_singular_value: float) -> ResearchIdentifiabilityStopRule:
    return ResearchIdentifiabilityStopRule(
        minimum_rank=5,
        minimum_singular_value=minimum_singular_value,
        maximum_condition_number=50.0,
        maximum_abs_parameter_correlation=0.30,
        maximum_uncertainty_proxy=0.10,
        minimum_parameter_sensitivity=0.0,
        maximum_validation_rmse_nm=0.20,
    )


@pytest.fixture(scope="module")
def baseline_results() -> dict[int, object]:
    return {
        expected: run_sequential_initial_identification(
            VirtualIdentificationOracle("baseline"),
            default_virtual_patient_envelope(),
            stop_rule=_rule(threshold),
        )
        for expected, threshold in ((1, 14.0), (2, 20.0), (3, 24.0), (4, 27.0))
    }


@pytest.fixture(scope="module")
def no_rule_failure():
    return run_sequential_initial_identification(
        VirtualIdentificationOracle("baseline"),
        default_virtual_patient_envelope(),
    )


@pytest.fixture(scope="module")
def mismatch_failure():
    return run_sequential_initial_identification(
        VirtualIdentificationOracle("baseline", "combined_mild"),
        default_virtual_patient_envelope(),
        stop_rule=VIRTUAL_RESEARCH_COMPARATOR_RULE,
    )


@pytest.fixture(scope="module")
def limited_success():
    return run_sequential_initial_identification(
        VirtualIdentificationOracle("LIMITED_ROM_VIRTUAL_SUBJECT"),
        limited_rom_virtual_patient_envelope(),
        stop_rule=VIRTUAL_RESEARCH_COMPARATOR_RULE,
    )


def test_maximum_trial_count_is_exactly_five() -> None:
    assert MAX_INITIAL_IDENTIFICATION_TRIALS == 5


def test_trial_one_can_stop_early(baseline_results) -> None:
    assert baseline_results[1].status == INITIAL_IDENTIFICATION_COMPLETE
    assert baseline_results[1].trials_required == 1


@pytest.mark.parametrize("trial", [2, 3, 4])
def test_trials_two_three_and_four_can_stop_early(baseline_results, trial: int) -> None:
    assert baseline_results[trial].status == INITIAL_IDENTIFICATION_COMPLETE
    assert baseline_results[trial].trials_required == trial


def test_sixth_trial_is_impossible(no_rule_failure) -> None:
    assert no_rule_failure.trials_required == 5
    assert no_rule_failure.executed_identification_data["trial_id"].max() == 5


def test_failure_has_no_theta_hat_zero_and_blocks_interface(no_rule_failure) -> None:
    assert no_rule_failure.status == INITIAL_IDENTIFICATION_INSUFFICIENT
    assert no_rule_failure.theta_hat_0 is None
    assert no_rule_failure.d_init is None
    assert no_rule_failure.summary["personalization_interface_ready"] is False
    assert no_rule_failure.summary["real_robot_personalization_allowed"] is False


def test_default_stop_rule_fails_closed_pending_review(no_rule_failure) -> None:
    assert (
        no_rule_failure.summary["identifiability_stop_rule_status"]
        == IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW
    )


def test_identification_duration_is_independent_but_reference_is_unchanged() -> None:
    before = sha256_file(ACTIVE_REFERENCE_PATH)
    spec = default_virtual_research_candidate_pool()[0]
    excitation = generate_identification_excitation(
        spec, default_virtual_patient_envelope()
    )
    assert spec.excitation_duration_s != TOTAL_DURATION_S
    assert excitation.trajectory["time_s"].iloc[-1] == pytest.approx(
        spec.excitation_duration_s
    )
    assert excitation.trajectory["duration_design_status"].eq(
        RESEARCH_DURATION_LABEL
    ).all()
    assert sha256_file(ACTIVE_REFERENCE_PATH) == before == ACTIVE_REFERENCE_SHA256
    assert TOTAL_DURATION_S == 24.0


def test_variable_duration_preserves_c2_cycle_seam() -> None:
    spec = IdentificationExcitationSpec("duration-17", -1.0, -1.0, 0.01, 17.0)
    result = generate_identification_excitation(
        spec, default_virtual_patient_envelope()
    )
    assert result.global_constraint_audit["c2_cycle_seam_valid"] is True
    assert result.global_constraint_audit["finite_valid"] is True


def test_candidate_pool_duration_values_are_research_design_only() -> None:
    pool = default_virtual_research_candidate_pool()
    durations = {spec.excitation_duration_s for spec in pool}
    assert len(durations) > 1
    assert durations != {24.0}
    assert all(spec.duration_design_status == RESEARCH_DURATION_LABEL for spec in pool)


def test_candidate_ranking_is_deterministic_and_constraint_first() -> None:
    envelope = limited_rom_virtual_patient_envelope()
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    parameters = {
        "mass_scale": 1.0,
        "k_hip_nm_per_rad": 10.0,
        "k_knee_nm_per_rad": 10.0,
        "b_hip_nm_s_per_rad": 1.0,
        "b_knee_nm_s_per_rad": 1.0,
    }
    pool = (
        IdentificationExcitationSpec("valid", 0.0, 0.0, 0.0, 20.0),
        IdentificationExcitationSpec("outside", 2.0, 2.0, 0.0, 14.0),
    )
    first, first_audit, _ = select_next_identification_excitation(
        pd.DataFrame(), template, parameters, envelope, pool
    )
    second, second_audit, _ = select_next_identification_excitation(
        pd.DataFrame(), template, parameters, envelope, pool
    )
    assert first is not None and second is not None
    assert first.spec.candidate_id == second.spec.candidate_id == "valid"
    assert first_audit["candidate_id"].tolist() == second_audit["candidate_id"].tolist()
    assert bool(first_audit.iloc[0]["candidate_valid"])
    assert not bool(first_audit.iloc[-1]["candidate_valid"])


def test_global_model_rom_is_not_patient_safety_rom() -> None:
    excitation = generate_identification_excitation(
        default_virtual_research_candidate_pool()[0],
        default_virtual_patient_envelope(),
    )
    audit = excitation.global_constraint_audit
    assert audit["global_model_rom_is_patient_safety_rom"] is False
    assert tuple(audit["hip_rom_deg"]) == FORMAL_HIP_ROM_DEG
    assert tuple(audit["knee_rom_deg"]) == FORMAL_KNEE_ROM_DEG
    assert REAL_ROBOT_HARD_SAFEGUARD == "NOT_DEFINED_NOT_APPROVED"


def test_patient_envelope_can_be_much_smaller_than_global_rom() -> None:
    envelope = limited_rom_virtual_patient_envelope()
    assert envelope.patient_hip_min_deg > FORMAL_HIP_ROM_DEG[0]
    assert envelope.patient_hip_max_deg < FORMAL_HIP_ROM_DEG[1]
    assert envelope.patient_knee_min_deg > FORMAL_KNEE_ROM_DEG[0]
    assert envelope.patient_knee_max_deg < FORMAL_KNEE_ROM_DEG[1]
    assert "SYNTHETIC" in envelope.source_status


def test_candidate_must_be_inside_patient_envelope() -> None:
    outside = IdentificationExcitationSpec("outside-limited", 2.0, 2.0, 0.0, 18.0)
    generated = generate_identification_excitation(
        outside, limited_rom_virtual_patient_envelope()
    )
    assert generated.patient_envelope_valid is False
    assert generated.candidate_valid is False
    assert "outside_current_patient_operational_envelope" in generated.invalid_reason


def test_patient_envelope_never_auto_expands_or_probes_a_boundary() -> None:
    assert AUTO_EXPAND_PATIENT_ENVELOPE is False
    source = inspect.getsource(run_sequential_initial_identification)
    assert "constraint_violation_used_to_discover_boundary" in source
    result = run_sequential_initial_identification(
        VirtualIdentificationOracle("baseline"),
        default_virtual_patient_envelope(),
        stop_rule=_rule(14.0),
    )
    assert result.summary["patient_envelope_auto_expanded"] is False
    assert result.summary["constraint_violation_used_to_discover_boundary"] is False


def test_next_trial_uses_executed_data_and_changes_selection() -> None:
    envelope = default_virtual_patient_envelope()
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    initial = dict(
        mass_scale=1.0,
        k_hip_nm_per_rad=10.0,
        k_knee_nm_per_rad=10.0,
        b_hip_nm_s_per_rad=1.0,
        b_knee_nm_s_per_rad=1.0,
    )
    first, _, _ = select_next_identification_excitation(
        pd.DataFrame(), template, initial, envelope, default_virtual_research_candidate_pool()
    )
    assert first is not None
    data = VirtualIdentificationOracle("baseline").execute(first)
    second, _, _ = select_next_identification_excitation(
        data,
        template,
        initial,
        envelope,
        default_virtual_research_candidate_pool(),
        already_executed_candidate_ids=[first.spec.candidate_id],
    )
    assert second is not None
    assert second.spec.candidate_id != first.spec.candidate_id


def test_selector_interface_has_no_truth_or_heldout_or_mechanical_j() -> None:
    signature = inspect.signature(select_next_identification_excitation)
    names = set(signature.parameters)
    assert not {"truth", "truth_parameters", "heldout_test", "mechanical_j"} & names
    source = inspect.getsource(select_next_identification_excitation)
    assert "mechanical_personalization" not in source


def test_incremental_information_is_computed_and_selected_is_not_duplicate(
    baseline_results,
) -> None:
    gains = baseline_results[2].incremental_information_gain
    assert (gains["incremental_log_information_gain"] > 0.0).all()
    assert (~gains["duplicate_information"].astype(bool)).all()


def test_repeated_information_is_recognized() -> None:
    envelope = default_virtual_patient_envelope()
    first_spec = IdentificationExcitationSpec("same-a", -1.0, -1.0, 0.0, 18.0)
    duplicate_spec = IdentificationExcitationSpec("same-b", -1.0, -1.0, 0.0, 18.0)
    first = generate_identification_excitation(first_spec, envelope)
    executed = VirtualIdentificationOracle("baseline").execute(first)
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    selected, audit, _ = select_next_identification_excitation(
        executed,
        template,
        dict(
            mass_scale=1.0,
            k_hip_nm_per_rad=10.0,
            k_knee_nm_per_rad=10.0,
            b_hip_nm_s_per_rad=1.0,
            b_knee_nm_s_per_rad=1.0,
        ),
        envelope,
        [duplicate_spec],
    )
    assert selected is None
    assert bool(audit.iloc[0]["design_duplicate"])


def test_all_five_parameters_are_audited_individually(baseline_results) -> None:
    table = baseline_results[2].parameter_identifiability
    assert set(table["parameter"]) == set(PARAMETER_NAMES)
    for column in (
        "sensitivity",
        "uncertainty_proxy",
        "optimizer_standard_error",
        "recovery_stability_relative_change",
    ):
        assert column in table


def test_low_residual_cannot_replace_identifiability() -> None:
    impossible = _rule(1e9)
    result = run_sequential_initial_identification(
        VirtualIdentificationOracle("baseline"),
        default_virtual_patient_envelope(),
        stop_rule=impossible,
    )
    assert result.trial_history["training_residual_rmse_nm"].max() < 1e-6
    assert result.status == INITIAL_IDENTIFICATION_INSUFFICIENT
    assert result.theta_hat_0 is None


def test_d_init_contains_only_executed_identification_trials(baseline_results) -> None:
    result = baseline_results[2]
    assert result.d_init is not None
    assert set(result.d_init["trial_id"]) == {1, 2}
    assert set(result.d_init["candidate_id"]) == set(
        result.trial_history["candidate_id"]
    )
    assert 1 <= result.trials_required <= 5


def test_success_freezes_theta_hat_zero_for_all_five_parameters(baseline_results) -> None:
    result = baseline_results[2]
    assert result.theta_hat_0 is not None
    assert set(result.theta_hat_0) == set(PARAMETER_NAMES)
    assert result.summary["theta_hat_0_frozen"] is True
    assert result.summary["initial_identification_dataset_sha"]


def test_mild_mismatch_is_preserved_as_a_fail_closed_case(mismatch_failure) -> None:
    assert mismatch_failure.status == INITIAL_IDENTIFICATION_INSUFFICIENT
    assert mismatch_failure.trials_required == 5
    assert mismatch_failure.theta_hat_0 is None
    assert "validation_residual_exceeded" in mismatch_failure.summary[
        "completion_audit_reason"
    ]


def test_limited_rom_fixture_uses_smaller_envelope_without_expansion(limited_success) -> None:
    assert limited_success.status == INITIAL_IDENTIFICATION_COMPLETE
    assert limited_success.patient_envelope_history["auto_expand"].eq(False).all()
    assert limited_success.patient_envelope_history[
        "candidate_within_envelope"
    ].eq(True).all()


def _mini_parameter_map(path: Path) -> Path:
    rows = []
    for hip, identifier in ((0.0, "neutral"), (2.0, "positive-hip")):
        rows.append(
            {
                "hip_delta": hip,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "trajectory_id": identifier,
                "domain_coverage": 100.0,
                "global_rom_valid": True,
                "workspace_valid": True,
                "jacobian_valid": True,
                "force_mapping_valid": True,
                "closure_valid": True,
                "continuity_valid": True,
                "asymmetry_valid": True,
                "finite_valid": True,
                "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
            }
        )
    output = path / "mini_map.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def test_full_map_only_after_success_and_unsupported_points_still_get_j(
    tmp_path: Path,
    baseline_results,
    no_rule_failure,
) -> None:
    success = baseline_results[2]
    envelope = default_virtual_patient_envelope()
    detail, summary = build_full_prediction_map(
        [success, no_rule_failure],
        {
            "baseline__matched_linear": envelope,
        },
        parameter_map_path=_mini_parameter_map(tmp_path),
    )
    assert detail["J_pred"].notna().all()
    assert set(detail["prediction_label"]).issubset(
        {SUPPORTED_PREDICTION, UNSUPPORTED_EXTRAPOLATION}
    )
    unsupported = detail.loc[detail["prediction_label"].eq(UNSUPPORTED_EXTRAPOLATION)]
    assert not unsupported.empty
    assert unsupported["J_pred"].notna().all()
    failed_summary = summary.loc[
        ~summary["theta_hat_0_available"].astype(bool)
    ].iloc[0]
    assert bool(failed_summary["full_prediction_map_generated"]) is False


def test_theta_shank_definition_is_preserved() -> None:
    generated = generate_identification_excitation(
        default_virtual_research_candidate_pool()[0],
        default_virtual_patient_envelope(),
    ).trajectory
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert np.allclose(
        generated["theta_shank_rad"],
        generated["q_hip_rad"] - generated["q_knee_rad"],
    )


def test_module_has_no_hardware_safety_or_robot_connection_imports() -> None:
    module_path = Path(
        inspect.getsourcefile(run_sequential_initial_identification) or ""
    )
    runner_path = Path(inspect.getsourcefile(generate_formal_artifacts) or "")
    text = module_path.read_text(encoding="utf-8") + runner_path.read_text(encoding="utf-8")
    for forbidden in (
        "from .hardware",
        "from hardware",
        "import hardware",
        "from .safety",
        "from safety",
        "xCoreSDK",
        "connectToRobot",
        "enable_robot",
    ):
        assert forbidden not in text


def test_no_real_patient_safety_threshold_or_personalization_execution(
    baseline_results,
) -> None:
    result = baseline_results[2]
    assert result.summary["real_robot_hard_safeguard"] == "NOT_DEFINED_NOT_APPROVED"
    assert result.summary["personalization_executed"] is False
    assert result.summary["real_robot_personalization_allowed"] is False


def test_formal_artifact_smoke_contains_required_outputs(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    metadata = generate_formal_artifacts(output, maximum_prediction_points=12)
    required = {
        "identification_trial_history.csv",
        "identification_trial_candidates.csv",
        "parameter_identifiability_by_trial.csv",
        "parameter_estimates_by_trial.csv",
        "incremental_information_gain.csv",
        "patient_operational_envelope_history.csv",
        "subject_identification_summary.csv",
        "failure_case_summary.csv",
        "full_prediction_map_summary.csv",
        "SAFEGUARD_IDENTIFICATION_PERSONALIZATION_ARCHITECTURE.md",
        "DATA_LEAKAGE_AUDIT.md",
        "IDENTIFIABILITY_STOP_RULE_AUDIT.md",
        "METHOD_MIGRATION_AUDIT.md",
        "initial_subject_models.json",
        "initial_known_regions.json",
        "metadata.json",
        "sequential_initial_identification_flowchart.png",
        "parameter_identifiability_by_trial.png",
        "condition_number_by_trial.png",
        "parameter_correlation_by_trial.png",
        "excitation_trajectories_sequence.png",
        "patient_envelope_vs_global_rom.png",
        "identification_to_global_prediction_map.png",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    on_disk = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert on_disk["real_robot_motion_executed"] is False
    assert on_disk["real_patient_safety_thresholds_defined"] is False
    assert on_disk["personalization_executed"] is False
    assert metadata["full_prediction_map"]["formal_complete_lattice_used"] is False
