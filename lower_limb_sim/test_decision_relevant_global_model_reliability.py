from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.decision_relevant_global_model_reliability import (
    DIAGNOSTIC_INITIAL_MODEL,
    DIAGNOSTIC_ONLY,
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    GRID_DISTANCE_DEFINITION,
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    IMPROVE,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    NEUTRAL,
    NOT_APPROVED_FOR_PERSONALIZATION,
    NOT_HUMAN_READY,
    RESEARCH_DECISION_EQUIVALENCE_BAND,
    WORSE,
    build_predicted_map,
    build_trajectory_component_cache,
    classify_improvement_direction,
    decision_sign_agreement_summary,
    diagnostic_model_from_sequential_result,
    distance_to_supported_region,
    evaluate_truth_map,
    false_improvement_cases,
    frozen_baseline_metadata,
    geometrically_valid_parameter_lattice,
    global_rank_consistency,
    local_decision_regret,
    local_rank_consistency,
    mechanical_objective_from_torque_batch,
    one_step_coordinate_neighborhood,
    predicted_best_regret,
    reliability_vs_domain_coverage,
    reliability_vs_support_distance,
    scenario_reliability_summary,
    select_predicted_best,
)
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from lower_limb_sim.mechanical_objective import (
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
    compute_torque_metrics,
)
from lower_limb_sim.parameter_estimator import PARAMETER_NAMES
from lower_limb_sim.run_decision_relevant_global_model_reliability import (
    ANALYSIS_CASES,
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_PARAMETER_MAP_PATH,
    FIGURE_FILENAMES,
    REPORT_FILENAMES,
    _limited_lattice_for_test,
    build_explore_exploit_implication_table,
)
from lower_limb_sim.safeguarded_sequential_initial_identification import (
    SUPPORTED_PREDICTION,
    UNSUPPORTED_EXTRAPOLATION,
    VirtualIdentificationOracle,
    default_virtual_patient_envelope,
    run_sequential_initial_identification,
)


@pytest.fixture(scope="module")
def geometry_lattice() -> pd.DataFrame:
    return geometrically_valid_parameter_lattice(pd.read_csv(DEFAULT_PARAMETER_MAP_PATH))


@pytest.fixture(scope="module")
def diagnostic_maps(geometry_lattice) -> dict[str, object]:
    lattice = _limited_lattice_for_test(geometry_lattice, 100)
    cache = build_trajectory_component_cache(lattice)
    output: dict[str, object] = {"lattice": lattice, "cache": cache}
    for scenario in ("matched_linear", "combined_mild"):
        result = run_sequential_initial_identification(
            VirtualIdentificationOracle("baseline", scenario),
            default_virtual_patient_envelope(),
            stop_rule=None,
        )
        model = diagnostic_model_from_sequential_result(result)
        predicted, prediction_metadata = build_predicted_map(
            model, lattice, cache, batch_size=32
        )
        global_id = str(select_predicted_best(predicted)["trajectory_id"])
        supported_id = str(
            select_predicted_best(predicted, supported_only=True)["trajectory_id"]
        )
        evaluated, truth_metadata = evaluate_truth_map(
            predicted, model, cache, batch_size=32
        )
        output[scenario] = {
            "result": result,
            "model": model,
            "predicted": predicted,
            "evaluated": evaluated,
            "prediction_metadata": prediction_metadata,
            "truth_metadata": truth_metadata,
            "ids": {
                (f"baseline__{scenario}", "GLOBAL"): global_id,
                (f"baseline__{scenario}", "SUPPORTED_ONLY"): supported_id,
            },
        }
    return output


def test_protocol_baseline_is_frozen() -> None:
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)


def test_theta_shank_definition_is_difference() -> None:
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"


def test_active_reference_sha_is_unchanged() -> None:
    assert ACTIVE_REFERENCE_SHA256 == (
        "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    )
    assert sha256_file(ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256


def test_five_parameter_model_is_unchanged() -> None:
    assert PARAMETER_NAMES == (
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    )


def test_mechanical_equivalence_tolerance_is_reused_not_redefined() -> None:
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert frozen_baseline_metadata()["decision_equivalence_tolerance_status"] == (
        RESEARCH_DECISION_EQUIVALENCE_BAND
    )


def test_geometry_lattice_uses_all_non_domain_valid_points(geometry_lattice) -> None:
    original = pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    assert len(geometry_lattice) == 21025
    assert len(geometry_lattice) > int(original["trajectory_admissible"].sum())


def test_geometry_lattice_contains_reference_and_formal_steps(geometry_lattice) -> None:
    keys = set(
        map(
            tuple,
            geometry_lattice[["hip_delta", "knee_delta", "phase_delta"]].to_numpy(),
        )
    )
    assert (0.0, 0.0, 0.0) in keys
    assert (GRID_HIP_STEP_DEG, 0.0, 0.0) in keys
    assert (0.0, GRID_KNEE_STEP_DEG, 0.0) in keys
    assert (0.0, 0.0, GRID_PHASE_STEP) in keys


def test_matched_diagnostic_model_uses_actual_trial_two(diagnostic_maps) -> None:
    model = diagnostic_maps["matched_linear"]["model"]
    assert model.selected_trial_id == 2


def test_mismatch_diagnostic_model_uses_last_actual_trial(diagnostic_maps) -> None:
    case = diagnostic_maps["combined_mild"]
    assert case["model"].selected_trial_id == int(
        case["result"].executed_identification_data["trial_id"].max()
    )


def test_diagnostic_model_is_not_approved(diagnostic_maps) -> None:
    model = diagnostic_maps["combined_mild"]["model"]
    assert model.model_status == DIAGNOSTIC_ONLY
    assert model.approval_status == NOT_APPROVED_FOR_PERSONALIZATION
    assert model.human_readiness == NOT_HUMAN_READY


def test_unsupported_points_still_receive_finite_j_pred(diagnostic_maps) -> None:
    predicted = diagnostic_maps["combined_mild"]["predicted"]
    unsupported = predicted.loc[~predicted["model_supported"]]
    assert len(unsupported) > 0
    assert np.isfinite(unsupported["J_pred"]).all()
    assert set(unsupported["prediction_label"]) == {UNSUPPORTED_EXTRAPOLATION}


def test_prediction_table_has_no_truth_before_evaluation(diagnostic_maps) -> None:
    predicted = diagnostic_maps["combined_mild"]["predicted"]
    assert "J_truth" not in predicted
    assert diagnostic_maps["combined_mild"]["prediction_metadata"][
        "truth_evaluated_during_prediction"
    ] is False


def test_predicted_best_selection_rejects_truth_column(diagnostic_maps) -> None:
    evaluated = diagnostic_maps["combined_mild"]["evaluated"]
    with pytest.raises(ValueError, match="before truth attachment"):
        select_predicted_best(evaluated)


@pytest.mark.parametrize("scenario", ["matched_linear", "combined_mild"])
def test_reference_j_pred_and_truth_are_one(diagnostic_maps, scenario: str) -> None:
    evaluated = diagnostic_maps[scenario]["evaluated"]
    reference = evaluated.loc[
        np.isclose(evaluated["hip_delta"], 0.0)
        & np.isclose(evaluated["knee_delta"], 0.0)
        & np.isclose(evaluated["phase_delta"], 0.0)
    ].iloc[0]
    assert float(reference["J_pred"]) == pytest.approx(1.0, abs=1e-12)
    assert float(reference["J_truth"]) == pytest.approx(1.0, abs=1e-12)


def test_delta_j_is_relative_to_computed_reference(diagnostic_maps) -> None:
    evaluated = diagnostic_maps["combined_mild"]["evaluated"]
    assert np.allclose(evaluated["delta_J_pred"], evaluated["J_pred"] - 1.0)
    assert np.allclose(evaluated["delta_J_truth"], evaluated["J_truth"] - 1.0)


def test_improvement_direction_uses_existing_equivalence_band() -> None:
    values = np.array([-0.006, -0.005, 0.0, 0.005, 0.006])
    assert classify_improvement_direction(values).tolist() == [
        IMPROVE,
        NEUTRAL,
        NEUTRAL,
        NEUTRAL,
        WORSE,
    ]


def test_false_improvement_filter_is_explicit() -> None:
    row = {
        "case_id": "c",
        "subject_id": "s",
        "scenario_name": "x",
        "trajectory_id": "t",
        "hip_delta": 0.25,
        "knee_delta": 0.0,
        "phase_delta": 0.0,
        "J_pred": 0.99,
        "J_truth": 1.01,
        "delta_J_pred": -0.01,
        "delta_J_truth": 0.01,
        "domain_coverage": 80.0,
        "model_supported": False,
        "prediction_label": UNSUPPORTED_EXTRAPOLATION,
        "distance_to_supported_region": 1.0,
        "distance_to_supported_region_definition": GRID_DISTANCE_DEFINITION,
        "predicted_direction": IMPROVE,
        "truth_direction": WORSE,
        "decision_equivalence_band": 0.005,
        "false_improvement": True,
    }
    assert false_improvement_cases(pd.DataFrame([row]))["trajectory_id"].tolist() == [
        "t"
    ]


def test_distance_to_support_uses_formal_grid_step_units() -> None:
    points = pd.DataFrame(
        {
            "hip_delta": [0.0, 0.25, 0.5],
            "knee_delta": [0.0, 0.0, 0.0],
            "phase_delta": [0.0, 0.0, 0.0],
        }
    )
    assert distance_to_supported_region(points, [True, False, False]).tolist() == [
        0.0,
        1.0,
        2.0,
    ]


def test_distance_to_support_fails_without_supported_point() -> None:
    points = pd.DataFrame(
        {"hip_delta": [0.0], "knee_delta": [0.0], "phase_delta": [0.0]}
    )
    with pytest.raises(ValueError, match="without supported points"):
        distance_to_supported_region(points, [False])


def test_existing_coverage_gate_remains_ninety_percent() -> None:
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0


def test_supported_and_unsupported_labels_are_separate(diagnostic_maps) -> None:
    predicted = diagnostic_maps["combined_mild"]["predicted"]
    assert set(predicted["prediction_label"]) == {
        SUPPORTED_PREDICTION,
        UNSUPPORTED_EXTRAPOLATION,
    }


def test_global_rank_consistency_is_reproducible(diagnostic_maps) -> None:
    evaluated = diagnostic_maps["combined_mild"]["evaluated"]
    first = global_rank_consistency(evaluated)
    second = global_rank_consistency(evaluated)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["scope"]) == {
        "ALL_GEOMETRICALLY_ADMISSIBLE",
        "SUPPORTED_ONLY",
        "UNSUPPORTED_ONLY",
    }


def test_local_rank_uses_formal_grid_steps(diagnostic_maps) -> None:
    local = local_rank_consistency(diagnostic_maps["combined_mild"]["evaluated"])
    assert local["radius_grid_steps"].tolist() == [1, 2, 3]
    assert set(local["hip_step_deg"]) == {0.25}
    assert set(local["knee_step_deg"]) == {0.25}
    assert set(local["phase_step"]) == {0.0025}


def test_one_step_coordinate_neighborhood_contains_seven_points(diagnostic_maps) -> None:
    neighborhood = one_step_coordinate_neighborhood(
        diagnostic_maps["combined_mild"]["evaluated"]
    )
    assert len(neighborhood) == 7


def test_global_and_supported_regret_are_nonnegative(diagnostic_maps) -> None:
    case = diagnostic_maps["combined_mild"]
    regret = predicted_best_regret(case["evaluated"], case["ids"])
    assert set(regret["scope"]) == {"GLOBAL", "SUPPORTED_ONLY"}
    assert (regret["diagnostic_regret"] >= 0.0).all()
    assert not regret["truth_used_for_predicted_best_selection"].any()


def test_local_regret_fails_closed_when_no_supported_neighbor(diagnostic_maps) -> None:
    local = local_decision_regret(diagnostic_maps["matched_linear"]["evaluated"]).iloc[0]
    assert int(local["supported_local_point_count"]) == 0
    assert local["diagnostic_local_utility_label"] == (
        "NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW"
    )
    assert pd.isna(local["local_decision_regret"])


def test_summaries_keep_supported_and_unsupported_separate(diagnostic_maps) -> None:
    evaluated = diagnostic_maps["combined_mild"]["evaluated"]
    ranks = global_rank_consistency(evaluated)
    regrets = predicted_best_regret(evaluated, diagnostic_maps["combined_mild"]["ids"])
    local = local_decision_regret(evaluated)
    summary = scenario_reliability_summary(evaluated, ranks, regrets, local)
    assert summary["scope"].tolist() == ["OVERALL", "SUPPORTED", "UNSUPPORTED"]
    assert {
        "e_J_relative_mean_percent",
        "e_J_relative_median_percent",
        "e_J_relative_p90_percent",
        "e_J_relative_p95_percent",
        "e_J_relative_p99_percent",
        "e_J_relative_max_percent",
    }.issubset(summary.columns)


def test_sign_summary_contains_false_improvement_counts(diagnostic_maps) -> None:
    summary = decision_sign_agreement_summary(
        diagnostic_maps["combined_mild"]["evaluated"]
    )
    assert set(summary["scope"]) == {"OVERALL", "SUPPORTED", "UNSUPPORTED"}
    assert "false_improvement_count" in summary


def test_support_distance_analysis_preserves_definition(diagnostic_maps) -> None:
    table = reliability_vs_support_distance(
        diagnostic_maps["combined_mild"]["evaluated"]
    )
    assert set(table["distance_definition"]) == {GRID_DISTANCE_DEFINITION}


def test_domain_coverage_analysis_does_not_modify_gate(diagnostic_maps) -> None:
    table = reliability_vs_domain_coverage(
        diagnostic_maps["combined_mild"]["evaluated"]
    )
    assert not table["coverage_gate_modified"].any()
    assert table.loc[
        table["domain_coverage_percent"] >= 90.0,
        "existing_90_percent_gate_pass",
    ].all()


def test_matched_positive_control_is_near_ideal(diagnostic_maps) -> None:
    table = diagnostic_maps["matched_linear"]["evaluated"]
    assert float(table["e_J_abs"].max()) < 1e-9
    assert table["decision_sign_agreement"].all()


def test_mismatch_scenario_has_nonzero_prediction_error(diagnostic_maps) -> None:
    table = diagnostic_maps["combined_mild"]["evaluated"]
    assert float(table["e_J_abs"].max()) > 1e-5


def test_vectorized_mechanical_objective_keeps_formula() -> None:
    time = np.array([0.0, 0.5, 1.0])
    reference = compute_torque_metrics(time, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    result = mechanical_objective_from_torque_batch(
        time,
        np.array([[2.0, 2.0, 2.0]]),
        np.array([[2.0, 2.0, 2.0]]),
        reference,
    )
    assert result[0] == pytest.approx(np.sqrt((2.0**2 + 1.0**2) / 2.0))


def test_active_reference_is_not_rom_clipped() -> None:
    reference = pd.read_csv(ACTIVE_REFERENCE_PATH)
    knee = np.rad2deg(reference["q_knee_rad"].to_numpy(dtype=float))
    assert knee.min() >= FORMAL_KNEE_ROM_DEG[0]
    assert knee.max() <= FORMAL_KNEE_ROM_DEG[1]
    assert knee.max() > 120.0


def test_frozen_metadata_has_no_formal_or_human_ready_model() -> None:
    metadata = frozen_baseline_metadata()
    assert metadata["formal_theta_hat_0_available"] is False
    assert metadata["human_ready_theta_hat_0_available"] is False
    assert metadata["initial_identification_acceptance_status"] == (
        INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS
    )
    assert metadata["global_model_reliability_rule_status"] == (
        GLOBAL_MODEL_RELIABILITY_RULE_STATUS
    )


def test_new_modules_do_not_import_hardware_control_or_safety() -> None:
    for filename in (
        "decision_relevant_global_model_reliability.py",
        "run_decision_relevant_global_model_reliability.py",
    ):
        tree = ast.parse((Path(__file__).parent / filename).read_text(encoding="utf-8"))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any(
            name.startswith(("hardware", "control", "safety")) for name in imports
        )


def test_new_modules_have_no_robot_connection_calls() -> None:
    source = "\n".join(
        (Path(__file__).parent / filename).read_text(encoding="utf-8")
        for filename in (
            "decision_relevant_global_model_reliability.py",
            "run_decision_relevant_global_model_reliability.py",
        )
    )
    assert "connectToRobot(" not in source
    assert ".connect(" not in source
    assert "start_cartesian" not in source


def test_analysis_case_registry_has_four_matched_and_five_mismatch() -> None:
    assert sum(scenario == "matched_linear" for _, scenario, _ in ANALYSIS_CASES) == 4
    assert sum(scenario != "matched_linear" for _, scenario, _ in ANALYSIS_CASES) == 5


def test_formal_artifact_set_is_exactly_twenty_one_files() -> None:
    expected = set(CSV_FILENAMES + FIGURE_FILENAMES + REPORT_FILENAMES + ("metadata.json",))
    observed = {
        path.name for path in DEFAULT_OUTPUT_DIRECTORY.iterdir() if path.is_file()
    }
    assert observed == expected
    assert len(observed) == 21


def test_formal_metadata_keeps_all_release_boundaries_closed() -> None:
    metadata = json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["global_reliability_threshold_frozen"] is False
    assert metadata["initial_identification_acceptance_rule_modified"] is False
    assert metadata["formal_theta_hat_0_frozen"] is False
    assert metadata["human_ready_theta_hat_0_frozen"] is False
    assert metadata["heldout_final_test_read"] is False
    assert metadata["personalization_executed"] is False
    assert metadata["hardware_connected"] is False


def test_formal_map_contains_all_cases_and_all_geometric_points() -> None:
    metadata = json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["analysis_case_count"] == 9
    assert metadata["evaluated_point_count_per_case"] == 21025
    row_count = sum(
        1
        for _ in (DEFAULT_OUTPUT_DIRECTORY / "global_prediction_truth_comparison.csv").open(
            encoding="utf-8"
        )
    ) - 1
    assert row_count == 9 * 21025


def test_formal_figures_are_nonempty_png_files() -> None:
    for filename in FIGURE_FILENAMES:
        path = DEFAULT_OUTPUT_DIRECTORY / filename
        assert path.stat().st_size > 10_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_explore_exploit_table_is_conceptual_only() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "explore_exploit_implication_table.csv"
    )
    assert not table["is_executable_policy"].astype(bool).any()
    assert not table["trajectory_proposed"].astype(bool).any()
    assert not table["trajectory_executed"].astype(bool).any()


def test_heldout_final_test_is_absent_from_new_runtime_imports() -> None:
    source = (
        Path(__file__).parent / "run_decision_relevant_global_model_reliability.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS" not in imported_names

