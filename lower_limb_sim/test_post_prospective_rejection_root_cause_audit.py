from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    geometrically_valid_parameter_lattice,
)
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from lower_limb_sim.mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from lower_limb_sim.p2_v2_prospective_offline_validation import (
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
)
from lower_limb_sim.post_prospective_rejection_root_cause_audit import (
    AUDIT_DATA_ROLE,
    BUNDLE_OUTCOME_STATUS,
    BUNDLE_PROTOCOL_ID,
    FINAL_STATUSES,
    PROSPECTIVE_CONCLUSION,
    PROSPECTIVE_MANIFEST_PATH,
    PROSPECTIVE_MANIFEST_SHA256,
    PROSPECTIVE_START_COMMIT,
    build_designated_bundle_pair_plan,
    sha256_file,
    verify_immutable_prospective_artifacts,
)
from lower_limb_sim.run_p2_v2_prospective_offline_validation import (
    _protected_source_hashes,
)
from lower_limb_sim.run_post_prospective_rejection_root_cause_audit import (
    DEFAULT_OUTPUT_DIRECTORY,
    EXTRA_CSV_FILENAMES,
    FIGURE_FILENAMES,
    JSON_FILENAMES,
    REPORT_FILENAMES,
    REQUIRED_CSV_FILENAMES,
    _write_csv,
)
from lower_limb_sim.run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


ARTIFACT_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((ARTIFACT_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def factorial() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIRECTORY / "factorial_policy_summary.csv")


@pytest.fixture(scope="module")
def plan() -> pd.DataFrame:
    return pd.read_csv(
        ARTIFACT_DIRECTORY / "designated_bundle_validation_pair_plan.csv"
    )


def test_original_prospective_rejection_is_immutable(metadata) -> None:
    original = verify_immutable_prospective_artifacts()
    assert PROSPECTIVE_CONCLUSION == "P2_V2_PROSPECTIVE_EVIDENCE_REJECTS_CURRENT_REVISION"
    assert original["final_status"] == PROSPECTIVE_CONCLUSION
    assert metadata["prospective_conclusion"] == PROSPECTIVE_CONCLUSION
    assert metadata["prospective_conclusion_revised"] is False


def test_original_manifest_sha_is_unchanged(metadata) -> None:
    assert PROSPECTIVE_MANIFEST_SHA256 == (
        "94d33675b2ae51ef80154c3bba92f31b87852267f3cffbaaacc75c3ce0aa1876"
    )
    assert sha256_file(PROSPECTIVE_MANIFEST_PATH) == PROSPECTIVE_MANIFEST_SHA256
    assert metadata["prospective_manifest_sha256"] == PROSPECTIVE_MANIFEST_SHA256


def test_original_prospective_start_commit_is_unchanged(metadata) -> None:
    assert PROSPECTIVE_START_COMMIT == "d7fe80945ae625fffc7919e1735e9e2df8c8fa00"
    assert metadata["prospective_start_commit_sha"] == PROSPECTIVE_START_COMMIT


def test_a0_and_a3_reproduce_historical_results_exactly(metadata) -> None:
    reproduction = pd.read_csv(
        ARTIFACT_DIRECTORY / "factorial_historical_reproduction_audit.csv"
    )
    assert len(reproduction) == 2 * 6 * 13
    assert reproduction["reproduced"].astype(bool).all()
    assert np.allclose(reproduction["absolute_difference"], 0.0, atol=1e-11)
    assert metadata["A0_A3_historical_reproduction_all_passed"] is True


def test_a1_and_a2_are_posthoc_counterfactual_only(factorial, metadata) -> None:
    posthoc = factorial.loc[factorial["factorial_variant_id"].str.startswith(("A1", "A2"))]
    assert set(posthoc["evidence_role"]) == {"POST_HOC_COUNTERFACTUAL_ONLY"}
    assert set(posthoc["data_role"]) == {AUDIT_DATA_ROLE}
    assert metadata["A1_A2_post_hoc_counterfactual_only"] is True


def test_truth_did_not_modify_historical_policy(factorial, metadata) -> None:
    assert not factorial["truth_used_to_modify_historical_policy"].astype(bool).any()
    assert metadata["prospective_outcome_used_to_modify_historical_policy"] is False


def test_every_missed_candidate_has_complete_mechanism_and_stopping_audit(
    factorial,
) -> None:
    missed = pd.read_csv(
        ARTIFACT_DIRECTORY / "prospective_missed_round_root_cause.csv"
    )
    allowed = {
        "GUARD_BLOCKED_TRUE_IMPROVEMENT",
        "SINGLE_STEP_TOLERANCE_BLOCKED",
        "EXPLORATION_STOPPED_BEFORE_REACHING_CANDIDATE",
        "SUPPORT_PROVENANCE_BLOCKED",
        "MODEL_PREDICTED_WRONG_DIRECTION",
        "MULTIPLE_FACTORS",
    }
    assert len(missed) == int(factorial["missed_improvement_rounds"].sum())
    assert missed["rejection_mechanism"].notna().all()
    assert set(missed["rejection_mechanism"]).issubset(allowed)
    assert missed["S0_would_continue"].notna().all()
    assert missed["S2_would_continue"].notna().all()
    assert not missed["truth_fed_back_to_historical_decision"].astype(bool).any()


def test_all_nine_small_step_paths_have_steps_one_through_five() -> None:
    detail = pd.read_csv(
        ARTIFACT_DIRECTORY / "prospective_small_step_accumulation.csv"
    )
    assert detail["path_id"].nunique() == 9
    assert len(detail) == 45
    assert set(detail.groupby("path_id")["step_number"].apply(tuple)) == {
        (1, 2, 3, 4, 5)
    }
    assert detail["same_formal_parameter_direction"].astype(bool).all()
    assert not detail["mixed_axis_or_turn_required"].astype(bool).any()


def test_bundle_residuals_are_posthoc_characterization_only() -> None:
    residuals = pd.read_csv(
        ARTIFACT_DIRECTORY / "prospective_bundle_residual_characterization.csv"
    )
    assert len(residuals) == 27
    assert set(residuals["bundle_length"]) == {2, 3, 5}
    assert set(residuals["calibration_role"]) == {
        "POST_HOC_CHARACTERIZATION_NOT_FUTURE_CALIBRATION"
    }
    assert not residuals["future_bundle_uncertainty_updated"].astype(bool).any()


def test_bundle_pair_plan_uses_geometry_only(plan) -> None:
    assert len(plan) == 648
    assert not plan["prediction_used_for_plan"].astype(bool).any()
    assert not plan["truth_used_for_plan"].astype(bool).any()
    assert not plan["prospective_error_used_for_plan"].astype(bool).any()
    assert plan["formal_neighbor_continuous"].astype(bool).all()
    assert plan["direction_consistent"].astype(bool).all()
    assert not plan["generator_bounds_expanded"].astype(bool).any()


def test_bundle_plan_is_balanced_and_has_no_outcomes(plan) -> None:
    groups = plan.groupby(
        ["coordinate", "direction", "bundle_length", "location_class"]
    ).size()
    assert len(groups) == 54
    assert set(groups) == {12}
    assert set(plan["outcome_status"]) == {BUNDLE_OUTCOME_STATUS}
    for column in ("predicted_delta_J", "truth_delta_J", "e_deltaJ_bundle"):
        assert plan[column].isna().all()
    assert not plan["calibration_truth_generated_in_this_task"].astype(bool).any()


def test_bundle_plan_sha_is_deterministic(tmp_path, metadata) -> None:
    raw = pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    lattice = geometrically_valid_parameter_lattice(raw)
    rebuilt, _ = build_designated_bundle_pair_plan(lattice)
    path = tmp_path / "designated_bundle_validation_pair_plan.csv"
    _write_csv(path, rebuilt)
    assert sha256_file(path) == metadata["bundle_pair_plan_sha256"]
    assert metadata["bundle_pair_plan_sha256"] == (
        "3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84"
    )


def test_bundle_protocol_freezes_plan_before_future_truth(metadata) -> None:
    protocol = json.loads(
        (ARTIFACT_DIRECTORY / "DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["protocol_id"] == BUNDLE_PROTOCOL_ID
    assert protocol["truth_used_to_select_plan"] is False
    assert protocol["future_truth_generated_in_this_task"] is False
    assert protocol["bundle_uncertainty_calibrated_in_this_task"] is False
    assert metadata["future_bundle_calibration_truth_generated"] is False


def test_no_percentile_or_k_was_selected(metadata) -> None:
    assert metadata["new_percentile_selected"] is False
    assert metadata["K_tuned"] is False
    assert metadata["new_policy_implemented"] is False
    assert metadata["P2_V2_default_enabled"] is False


def test_equivalence_tolerance_and_support_gate_are_unchanged(metadata) -> None:
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert metadata["algorithm_equivalence_tolerance"] == 0.005
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert metadata["support_gate_percent"] == 90.0


def test_objective_generator_model_and_p2_v1_are_unchanged(metadata) -> None:
    before = metadata["protected_source_sha256_before"]
    after = metadata["protected_source_sha256_after"]
    for name in (
        "mechanical_objective",
        "generator",
        "five_parameter_estimator",
        "P2_V1_core",
    ):
        assert before[name] == after[name]


def test_reference_rom_and_theta_shank_are_unchanged(metadata) -> None:
    assert ACTIVE_REFERENCE_SHA256 == (
        "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    )
    assert metadata["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert tuple(FORMAL_HIP_ROM_DEG) == (0.0, 120.0)
    assert tuple(FORMAL_KNEE_ROM_DEG) == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert metadata["theta_shank_definition"] == THETA_SHANK_DEFINITION


def test_hardware_control_collection_and_safety_have_zero_diff(metadata) -> None:
    before = metadata["protected_source_sha256_before"]
    after = metadata["protected_source_sha256_after"]
    current = _protected_source_hashes()
    for name in ("tree:hardware", "tree:control", "tree:collection", "tree:safety"):
        assert before[name] == after[name] == current[name]


def test_no_robot_or_human_approval(metadata) -> None:
    assert metadata["robot_connected"] is False
    assert metadata["robot_motion_approval"] == NOT_ROBOT_MOTION_APPROVED
    assert metadata["human_readiness"] == NOT_HUMAN_READY


def test_factorial_identifies_guard_as_endpoint_failure_cause() -> None:
    matrix = pd.read_csv(
        ARTIFACT_DIRECTORY / "POST_PROSPECTIVE_REVISION_ROOT_CAUSE_MATRIX.csv"
    ).set_index("problem")
    assert matrix.loc["HIGHER_FINAL_J_AND_REGRET", "primary_cause"] == "GUARD_EFFECT"
    assert matrix.loc[
        "HIGHER_FINAL_J_AND_REGRET", "secondary_cause"
    ] == "NO_SEPARABLE_STOPPING_OUTCOME_EFFECT_OBSERVED"
    assert matrix.loc[
        "HIGHER_MISSED_IMPROVEMENT_UNDER_REJECTED_V2A", "primary_cause"
    ] == "GUARD_EFFECT"


def test_stopping_audit_does_not_claim_unobserved_useful_truncation() -> None:
    stopping = pd.read_csv(
        ARTIFACT_DIRECTORY / "stopping_removed_trial_value_audit.csv"
    )
    assert not stopping["future_action_used_by_historical_stopping"].astype(bool).any()
    assert not stopping["removed_trial_value_classification"].eq(
        "TRUNCATED_USEFUL_EXPLORATION_CHAIN"
    ).any()


def test_current_cases_are_permanently_development_only(metadata) -> None:
    assert metadata["development_used_after_rejection"] is True
    assert metadata["current_six_cases_may_support_future_prospective_claim"] is False
    assert metadata["future_revision_requires_new_independent_prospective_cohort"] is True


def test_final_status_is_allowed_and_no_next_policy_exists(metadata) -> None:
    assert metadata["final_status"] in FINAL_STATUSES
    assert metadata["new_policy_implemented"] is False


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    names = [
        *REQUIRED_CSV_FILENAMES,
        *EXTRA_CSV_FILENAMES,
        *JSON_FILENAMES,
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
        "metadata.json",
    ]
    for name in names:
        path = ARTIFACT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_artifact_manifest_hashes_match_files(metadata) -> None:
    for name, record in metadata["artifact_manifest"].items():
        path = ARTIFACT_DIRECTORY / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert path.stat().st_size == record["bytes"]


def test_new_source_has_no_robot_side_imports() -> None:
    paths = (
        Path("lower_limb_sim/post_prospective_rejection_root_cause_audit.py"),
        Path("lower_limb_sim/run_post_prospective_rejection_root_cause_audit.py"),
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in (
        "from hardware",
        "import hardware",
        "from control",
        "import control",
        "from collection",
        "import collection",
        "from safety",
        "import safety",
        "xCoreSDK",
    ):
        assert forbidden not in text
