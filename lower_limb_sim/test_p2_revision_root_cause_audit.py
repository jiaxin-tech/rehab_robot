from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from .continuous_reference_neighborhood import OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
)
from .large_artifact_reproduction import artifact_entry
from .p2_revision_root_cause_audit import (
    AUDIT_PROTOCOL_ID,
    EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT,
    GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH,
    LOCAL_CALIBRATION_NOT_SUFFICIENT,
    LOCAL_PAIR_UNAVAILABLE,
    OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM,
    OFFLINE_METHOD_REQUIRES_REVISION,
    P2_POLICY_REVISION_JUSTIFIED,
    POLICY_COLLAPSES_SUBJECT_DIFFERENCES,
    POST_HOC_TRUTH_ROLE,
    POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION,
    SUPPORT_ONLY_EXPLORATION,
    SYNTHETIC_SCAN_ROLE,
    registered_parameter_design,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .run_p2_revision_root_cause_audit import (
    CORE_SOURCE_PATH,
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    FIGURE_FILENAMES,
    GENERATOR_SOURCE,
    MECHANICAL_OBJECTIVE_SOURCE,
    POLICY_ARTIFACT_DIRECTORY,
    POLICY_CORE_SOURCE,
    REPORT_FILENAMES,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    DEFAULT_OUTPUT_DIRECTORY as CONVERGENCE_ARTIFACT_DIRECTORY,
)


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(
        (DEFAULT_OUTPUT_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def truth_summary() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "truth_landscape_summary.csv")


@pytest.fixture(scope="module")
def counterfactual() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "counterfactual_guard_comparison.csv"
    )


@pytest.fixture(scope="module")
def exploration() -> pd.DataFrame:
    return pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "exploration_value_decomposition.csv"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_identifier_is_exact() -> None:
    assert AUDIT_PROTOCOL_ID == "P2_REVISION_ROOT_CAUSE_AUDIT_V1"


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    manifest_managed = {
        artifact_entry(f"truth_landscape_{subject}")["expected_filename"]
        for subject in ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
    }
    normal_regression_files = (
        set(CSV_FILENAMES + REPORT_FILENAMES + FIGURE_FILENAMES + ("metadata.json",))
        - manifest_managed
    )
    for name in normal_regression_files:
        path = DEFAULT_OUTPUT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_all_recorded_artifact_hashes_match(metadata) -> None:
    manifest_managed = {
        artifact_entry(f"truth_landscape_{subject}")["expected_filename"]
        for subject in ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
    }
    for name, expected in metadata["output_sha256"].items():
        if name in manifest_managed:
            logical_id = Path(name).stem
            assert artifact_entry(logical_id)["sha256"] == expected
        else:
            assert _sha(DEFAULT_OUTPUT_DIRECTORY / name) == expected


def test_current_p2_replay_is_unchanged(metadata) -> None:
    assert metadata["policy_behavior_modified"] is False
    assert metadata["policy_replay"]["case_count"] == 9
    assert metadata["policy_replay"]["replay_matches_previous_convergence_summary"]
    assert metadata["policy_replay"]["mismatch_fields"] == []


def test_policy_definition_remains_semantically_identical() -> None:
    previous = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    assert previous == policy_definitions()


def test_current_policy_source_matches_previous_convergence_checkpoint(metadata) -> None:
    previous = json.loads(
        (CONVERGENCE_ARTIFACT_DIRECTORY / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert _sha(POLICY_CORE_SOURCE) == metadata["policy_core_source_sha256"]
    assert metadata["policy_core_source_sha256"] == previous["core_policy_source_sha256"]


def test_decision_guard_and_selectors_are_unchanged(metadata) -> None:
    assert hashlib.sha256(inspect.getsource(apply_research_decision_guard).encode()).hexdigest() == metadata["decision_guard_source_sha256"]
    assert hashlib.sha256(inspect.getsource(select_exploit_candidate).encode()).hexdigest() == metadata["exploit_selector_source_sha256"]
    assert hashlib.sha256(inspect.getsource(rank_exploration_frontier).encode()).hexdigest() == metadata["exploration_ranker_source_sha256"]


def test_mechanical_objective_is_unchanged(metadata) -> None:
    assert MECHANICAL_OBJECTIVE_VERSION == "mechanical_joint_torque_objective_v1"
    assert metadata["mechanical_objective_modified"] is False
    assert _sha(MECHANICAL_OBJECTIVE_SOURCE) == metadata["mechanical_objective_source_sha256"]


def test_truth_landscapes_do_not_feed_policy(metadata) -> None:
    assert metadata["truth_role"] == POST_HOC_TRUTH_ROLE
    assert metadata["truth_fed_back_to_policy"] is False
    assert metadata["truth_used_for_candidate_proposal"] is False
    assert metadata["truth_used_for_model_fitting"] is False


def test_each_truth_landscape_has_the_same_21025_point_space() -> None:
    summary = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / "truth_landscape_summary.csv")
    assert set(summary["subject_id"]) == {
        "baseline",
        "hip_stiff",
        "knee_stiff",
        "heavy_leg",
    }
    assert summary["truth_landscape_point_count"].eq(21025).all()
    assert set(summary["truth_role"]) == {POST_HOC_TRUTH_ROLE}
    for subject in summary["subject_id"]:
        entry = artifact_entry(f"truth_landscape_{subject}")
        assert entry["expected_row_count"] == 21025
        assert entry["required_for_normal_pytest"] is False
        assert entry["required_for_formal_reproduction"] is True


def test_reference_truth_is_exactly_one_for_every_subject(truth_summary) -> None:
    np.testing.assert_allclose(truth_summary["reference_J_truth"], 1.0, atol=1e-12)


def test_subject_truth_minima_are_the_recomputed_values(truth_summary) -> None:
    observed = truth_summary.set_index("subject_id")
    expected = {
        "baseline": (2.0, -5.0, 0.03, 0.9676330753362283),
        "hip_stiff": (1.25, -5.0, 0.03, 0.9709052056690225),
        "knee_stiff": (2.0, -5.0, 0.03, 0.9747738730941182),
        "heavy_leg": (2.0, -5.0, 0.0025, 0.9632031171674547),
    }
    for subject, values in expected.items():
        row = observed.loc[subject]
        assert row["alpha_truth_global_hip"] == pytest.approx(values[0])
        assert row["alpha_truth_global_knee"] == pytest.approx(values[1])
        assert row["alpha_truth_global_phase"] == pytest.approx(values[2])
        assert row["J_truth_global"] == pytest.approx(values[3])


def test_truth_local_minimum_uses_formal_one_step_neighborhood(truth_summary) -> None:
    assert set(truth_summary["local_neighborhood_definition"]) == {
        "reference_plus_six_signed_coordinate_moves_at_formal_minimum_steps"
    }
    assert truth_summary["alpha_truth_local_hip"].eq(0.0).all()
    assert truth_summary["alpha_truth_local_knee"].eq(-0.25).all()
    assert truth_summary["alpha_truth_local_phase"].eq(0.0).all()


def test_local_sensitivity_matches_formal_central_difference() -> None:
    sensitivity = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "subject_truth_local_sensitivity.csv"
    )
    assert sensitivity["hip_formal_step"].eq(GRID_HIP_STEP_DEG).all()
    assert sensitivity["knee_formal_step"].eq(GRID_KNEE_STEP_DEG).all()
    assert sensitivity["phase_formal_step"].eq(GRID_PHASE_STEP).all()
    calculated = (
        sensitivity["J_at_positive_knee_step"]
        - sensitivity["J_at_negative_knee_step"]
    ) / (2.0 * GRID_KNEE_STEP_DEG)
    np.testing.assert_allclose(calculated, sensitivity["dJ_d_knee_at_reference"])
    assert sensitivity["finite_difference_role"].str.contains("NOT_PHYSIOLOGICAL").all()


def test_normalization_audit_recomputes_unchanged_objective_formula() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "objective_normalization_subject_effect.csv"
    )
    expected = np.sqrt((table["R_h"] ** 2 + table["R_k"] ** 2) / 2.0)
    np.testing.assert_allclose(expected, table["J_unchanged_formula"])
    assert not table["objective_modified"].astype(bool).any()


def test_normalization_does_not_erase_every_subject_optimum(truth_summary) -> None:
    columns = [
        "alpha_truth_global_hip",
        "alpha_truth_global_knee",
        "alpha_truth_global_phase",
    ]
    assert truth_summary[columns].drop_duplicates().shape[0] == 3


def test_synthetic_scan_is_research_only_and_uses_registered_values() -> None:
    scan = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "synthetic_subject_optimum_map.csv"
    )
    design = registered_parameter_design()
    assert len(scan) == len(design) == 8
    assert set(scan["research_role"]) == {SYNTHETIC_SCAN_ROLE}
    assert not scan["clinical_range_claimed"].astype(bool).any()
    for name in PARAMETER_NAMES:
        assert set(scan[name]) == set(design[name])


def test_synthetic_scan_finds_more_than_one_complete_optimum() -> None:
    scan = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "synthetic_subject_optimum_map.csv"
    )
    alpha = scan[
        ["alpha_truth_global_hip", "alpha_truth_global_knee", "alpha_truth_global_phase"]
    ]
    assert alpha.drop_duplicates().shape[0] == 5
    assert scan["global_knee_at_lower_generator_bound"].astype(bool).all()


def test_current_guard_provenance_is_designated_validation_only() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "current_guard_uncertainty_provenance.csv"
    )
    assert len(table) == 61
    assert set(table["calibration_data_role"]) == {"DESIGNATED_VALIDATION_ONLY"}
    assert not table["heldout_final_test_used"].astype(bool).any()
    assert table["current_guard_uses_e_delta_J"].astype(bool).all()


def test_current_validation_pairs_cannot_be_mislabeled_local() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "current_guard_uncertainty_provenance.csv"
    )
    assert not table["same_formal_one_step_local_scale"].astype(bool).any()
    assert not table["larger_formal_parameter_distance"].astype(bool).any()
    assert table["formal_parameter_distance_steps"].isna().all()
    assert table["validation_pair_scale_class"].str.startswith("UNMAPPABLE").all()


def test_local_pairwise_candidate_uses_only_formal_local_scale_or_stays_unavailable() -> None:
    table = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "global_vs_local_validation_uncertainty.csv"
    )
    local = table.loc[
        table["validation_pair_scale_class"].eq("SAME_FORMAL_ONE_STEP_LOCAL_SCALE")
    ].iloc[0]
    assert local["pair_instance_count"] == 0
    assert pd.isna(local["max_e_delta_J"])
    assert local["candidate_status"] == LOCAL_PAIR_UNAVAILABLE
    assert not local["formal_threshold_created"]


def test_counterfactual_guards_execute_no_trajectory(counterfactual) -> None:
    assert not counterfactual["policy_executed_from_counterfactual"].astype(bool).any()


def test_truth_is_only_used_for_counterfactual_outcomes(counterfactual) -> None:
    assert not counterfactual["truth_used_to_construct_guard"].astype(bool).any()
    assert counterfactual["truth_used_only_for_posthoc_outcome"].astype(bool).all()


def test_g0_reproduces_seven_missed_and_zero_false_candidates(counterfactual) -> None:
    g0 = counterfactual.loc[
        counterfactual["guard_id"].eq("G0_CURRENT_GLOBAL_MAX")
    ]
    assert int(g0["missed_improvement"].astype(bool).sum()) == 7
    assert int(g0["false_improvement"].astype(bool).sum()) == 0


def test_g1_and_g2_are_unavailable_not_zero_uncertainty(counterfactual) -> None:
    local = counterfactual.loc[
        counterfactual["guard_id"].isin(
            ("G1_LOCAL_PAIRWISE_MAX", "G2_LOCAL_PAIRWISE_P95")
        )
    ]
    assert local["would_exploit"].isna().all()
    assert local["uncertainty_bound"].isna().all()
    assert set(local["counterfactual_status"]) == {LOCAL_PAIR_UNAVAILABLE}


def test_all_32_explore_rows_are_audited(exploration) -> None:
    assert len(exploration) == 32
    assert exploration[["case_id", "iteration"]].duplicated().sum() == 0


def test_support_and_decision_values_are_separate(exploration) -> None:
    assert exploration["SUPPORT_PROVENANCE_VALUE"].astype(bool).all()
    assert int(exploration["DECISION_VALUE"].astype(bool).sum()) == 3
    assert int(exploration["support_only_exploration"].astype(bool).sum()) == 29


def test_support_only_label_uses_exact_observable_equality(exploration) -> None:
    support_only = exploration.loc[
        exploration["support_only_exploration"].astype(bool)
    ]
    assert set(support_only["diagnostic_label"]) == {SUPPORT_ONLY_EXPLORATION}
    assert not support_only["theta_changed_exactly"].astype(bool).any()
    assert not support_only["prediction_map_changed_exactly"].astype(bool).any()
    assert not support_only["validation_error_changed_exactly"].astype(bool).any()
    assert support_only["newly_enabled_exploit_candidates"].eq(0).all()
    assert support_only["best_J_change"].eq(0.0).all()


def test_three_explores_open_new_exploit_candidates(exploration) -> None:
    decision = exploration.loc[exploration["DECISION_VALUE"].astype(bool)]
    assert len(decision) == 3
    assert decision["newly_enabled_exploit_candidates"].ge(1).all()
    assert decision["enabled_exploit_within_1_round"].astype(bool).all()


def test_knee_stiff_eight_explores_are_fully_covered() -> None:
    knee = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "knee_stiff_exploration_audit.csv"
    )
    assert len(knee) == 8
    assert knee["iteration"].tolist() == list(range(1, 9))
    assert knee["diagnostic_conclusion"].eq(
        EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT
    ).all()
    assert knee["theta_change_l2"].eq(0.0).all()
    assert knee["RMS_prediction_map_change"].eq(0.0).all()


def test_matched_trial_7_to_13_has_no_decision_value(exploration) -> None:
    selected = exploration.loc[
        exploration["subject_id"].isin(("baseline", "hip_stiff", "heavy_leg"))
        & exploration["iteration"].between(7, 13)
    ]
    assert len(selected) == 21
    assert selected["newly_enabled_exploit_candidates"].eq(0).all()
    assert not selected["DECISION_VALUE"].astype(bool).any()
    assert selected["support_only_exploration"].astype(bool).all()


def test_future_stopping_candidate_never_uses_truth_feature() -> None:
    report = (
        DEFAULT_OUTPUT_DIRECTORY / "EXPLORATION_STOPPING_CANDIDATE_ANALYSIS.md"
    ).read_text(encoding="utf-8")
    assert "truth_used_as_future_online_feature = false" in report
    assert "new_threshold_created = false" in report


def test_root_cause_matrix_contains_all_three_problems() -> None:
    matrix = pd.read_csv(
        DEFAULT_OUTPUT_DIRECTORY / "P2_REVISION_ROOT_CAUSE_MATRIX.csv"
    )
    assert set(matrix["problem"]) == {
        "same_subject_path",
        "premature_mismatch_stop",
        "low_value_exploration",
    }
    conclusions = set(matrix["conclusion"])
    assert OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM in conclusions
    assert POLICY_COLLAPSES_SUBJECT_DIFFERENCES in conclusions
    assert GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH in conclusions
    assert LOCAL_CALIBRATION_NOT_SUFFICIENT in conclusions
    assert EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT in conclusions
    assert POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION in conclusions


def test_equivalence_tolerance_is_unchanged() -> None:
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005


def test_support_gate_is_unchanged() -> None:
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0


def test_reference_sha_is_unchanged() -> None:
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"


def test_formal_rom_and_theta_convention_are_unchanged() -> None:
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"


def test_generator_bounds_and_source_are_unchanged(metadata) -> None:
    assert metadata["generator_modified"] is False
    assert metadata["generator_bounds"] == {
        key: list(value) for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
    }
    assert _sha(GENERATOR_SOURCE) == metadata["generator_source_sha256"]


def test_five_parameter_model_is_exactly_unchanged(metadata) -> None:
    assert tuple(PARAMETER_NAMES) == (
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    )
    assert metadata["five_parameter_model_modified"] is False


def test_hardware_control_collection_safety_have_no_tracked_diff(metadata) -> None:
    completed = subprocess.run(
        ["git", "diff", "--", "hardware", "control", "collection", "safety"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""
    assert metadata["protected_package_git_diff_empty"] is True


def test_new_modules_do_not_import_robot_packages() -> None:
    for path in (CORE_SOURCE_PATH, Path(__file__).with_name("run_p2_revision_root_cause_audit.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert imports.isdisjoint({"hardware", "control", "collection", "safety", "xCoreSDK_python"})


def test_no_robot_connection_or_human_threshold_was_created(metadata) -> None:
    assert metadata["real_robot_connected"] is False
    assert metadata["counterfactual_trajectory_executed"] is False
    assert metadata["human_threshold_created"] is False
    assert metadata["not_human_ready"] is True
    assert metadata["not_robot_motion_approved"] is True


def test_final_recommendation_keeps_offline_method_unready(metadata) -> None:
    assert metadata["final_recommendation"] == P2_POLICY_REVISION_JUSTIFIED
    assert metadata["revision_implementation_readiness"] == "REVISION_DESIGN_NOT_FROZEN"
    assert metadata["offline_method_status"] == OFFLINE_METHOD_REQUIRES_REVISION
    assert metadata["initial_identification_acceptance_status"] == INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS
    assert metadata["global_model_reliability_rule_status"] == GLOBAL_MODEL_RELIABILITY_RULE_STATUS


def test_objective_is_not_automatically_revised(metadata) -> None:
    assert metadata["objective_requires_scientific_review_status"] == (
        "OBJECTIVE_RETAINS_SOME_SUBJECT_DISCRIMINATION_NO_AUTOMATIC_REVISION"
    )
    assert metadata["mechanical_objective_modified"] is False
