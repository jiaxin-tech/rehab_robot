from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
)
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from lower_limb_sim.mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from lower_limb_sim.p2_next_revision_independent_calibration import (
    BUNDLE_SCALE_CALIBRATED,
    BUNDLE_SOURCE_PAIR_PLAN_PATH,
    BUNDLE_SOURCE_PAIR_PLAN_SHA256,
    CALIBRATION_CASE_COUNT,
    CALIBRATION_DATA_ROLE,
    CALIBRATION_ID,
    CALIBRATION_MANIFEST_ID,
    HELD_OUT_STATUS,
    LOCAL_SOURCE_PAIR_PLAN_SHA256,
    PROSPECTIVE_CONCLUSION,
    calibration_case_manifest,
    calibration_subject_definitions,
    residual_distribution,
    sha256_file,
)
from lower_limb_sim.p2_v2_prospective_offline_validation import (
    DEVELOPMENT_CASES,
    LOCAL_P95,
    prospective_case_rows,
    prospective_subject_definitions,
)
from lower_limb_sim.post_prospective_rejection_root_cause_audit import (
    PROSPECTIVE_MANIFEST_SHA256,
    PROSPECTIVE_START_COMMIT,
    verify_immutable_prospective_artifacts,
)
from lower_limb_sim.run_p2_next_revision_independent_calibration import (
    DEFAULT_OUTPUT_DIRECTORY,
    EXTRA_CSV_FILENAMES,
    FIGURE_FILENAMES,
    JSON_FILENAMES,
    REPORT_FILENAMES,
    REQUIRED_CSV_FILENAMES,
    _calibration_protected_hashes,
)


ARTIFACT_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY
MANIFEST_PATH = (
    ARTIFACT_DIRECTORY / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
)


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((ARTIFACT_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIRECTORY / "calibration_case_manifest.csv")


@pytest.fixture(scope="module")
def one_step() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIRECTORY / "independent_one_step_residuals.csv")


@pytest.fixture(scope="module")
def bundles() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(
                ARTIFACT_DIRECTORY / f"independent_bundle_{length}step_residuals.csv"
            )
            for length in (2, 3, 5)
        ],
        ignore_index=True,
    )


def test_post_prospective_audit_is_checkpointed() -> None:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            (
                "lower_limb_sim/formal_artifacts/"
                "post_prospective_rejection_root_cause_audit_v1/"
                "designated_bundle_validation_pair_plan.csv"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip()


def test_old_prospective_rejection_is_unchanged(metadata) -> None:
    old = verify_immutable_prospective_artifacts()
    assert old["final_status"] == PROSPECTIVE_CONCLUSION
    assert metadata["old_prospective_conclusion"] == PROSPECTIVE_CONCLUSION
    assert metadata["old_prospective_conclusion_revised"] is False


def test_old_prospective_manifest_and_start_commit_are_unchanged(metadata) -> None:
    assert metadata["old_prospective_manifest_sha256"] == PROSPECTIVE_MANIFEST_SHA256
    assert PROSPECTIVE_START_COMMIT == "d7fe80945ae625fffc7919e1735e9e2df8c8fa00"


def test_frozen_bundle_pair_plan_sha_is_unchanged(metadata) -> None:
    assert sha256_file(BUNDLE_SOURCE_PAIR_PLAN_PATH) == BUNDLE_SOURCE_PAIR_PLAN_SHA256
    assert metadata["bundle_source_pair_plan_sha256"] == (
        "3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84"
    )


def test_calibration_manifest_identity_and_sha(metadata, manifest) -> None:
    assert manifest["manifest_id"] == CALIBRATION_MANIFEST_ID
    assert manifest["calibration_id"] == CALIBRATION_ID
    assert manifest["status"] == "FROZEN_BEFORE_ANY_NEW_CALIBRATION_TRUTH"
    assert sha256_file(MANIFEST_PATH) == metadata["calibration_manifest_sha256"]
    assert metadata["calibration_manifest_sha256"] == (
        "08f930692704c24f10f85f094eabf45fc5e0842ec3f479345e62bee892df1729"
    )


def test_calibration_manifest_is_canonical_and_deterministic(metadata, manifest) -> None:
    canonical = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == metadata["calibration_manifest_sha256"]
    assert MANIFEST_PATH.read_bytes() == canonical


def test_calibration_cohort_has_six_matched_and_six_mismatch(cases) -> None:
    assert len(cases) == CALIBRATION_CASE_COUNT == 12
    assert cases["case_id"].nunique() == 12
    assert cases["calibration_category"].value_counts().to_dict() == {
        "MATCHED": 6,
        "MISMATCH": 6,
    }


def test_calibration_cohort_excludes_all_old_development_cases(cases) -> None:
    assert not set(cases["case_id"]).intersection(DEVELOPMENT_CASES)
    assert not cases["old_development_case"].astype(bool).any()


def test_calibration_cohort_excludes_rejected_prospective_cases(cases) -> None:
    rejected = set(prospective_case_rows()["case_id"].astype(str))
    assert not set(cases["case_id"]).intersection(rejected)
    assert not cases["rejected_prospective_case"].astype(bool).any()


def test_calibration_subject_parameter_signatures_are_unused() -> None:
    new = {item["parameter_signature"] for item in calibration_subject_definitions()}
    old = {
        "|".join(
            f"{float(value):.12g}"
            for value in (
                item["mass_scale"],
                item["parameters"]["k_hip_nm_per_rad"],
                item["parameters"]["k_knee_nm_per_rad"],
                item["damping_scale"],
            )
        )
        for item in prospective_subject_definitions()
    }
    assert len(new) == 6
    assert not new.intersection(old)


def test_case_selection_is_truth_free(cases, manifest) -> None:
    assert not cases[
        [
            "truth_used_for_case_selection",
            "truth_optimum_used_for_case_selection",
            "error_used_for_case_selection",
        ]
    ].astype(bool).any().any()
    assert manifest["case_selection"]["selection_uses_subject_specificity"] is False


def test_heldout_final_test_was_not_read(cases, metadata, manifest) -> None:
    assert not cases["heldout_final_test"].astype(bool).any()
    assert metadata["heldout_final_test_status"] == HELD_OUT_STATUS
    assert metadata["heldout_final_test_truth_access_count"] == 0
    assert manifest["heldout_final_test_read_allowed"] is False


def test_calibration_cases_are_not_reserved_for_future_prospective(cases, metadata) -> None:
    assert not cases["reserved_for_future_prospective"].astype(bool).any()
    assert metadata["reserved_for_future_prospective"] is False
    assert metadata["future_prospective_created"] is False


def test_local_plan_geometry_sha_is_reused_without_case_binding(metadata, manifest) -> None:
    path = ARTIFACT_DIRECTORY / "independent_local_calibration_pair_plan.csv"
    assert sha256_file(path) == LOCAL_SOURCE_PAIR_PLAN_SHA256
    assert metadata["local_source_pair_plan_sha256"] == LOCAL_SOURCE_PAIR_PLAN_SHA256
    assert manifest["local_validation_protocol"]["source_geometry_unchanged"] is True


def test_local_assignment_is_balanced_and_truth_free(manifest) -> None:
    assignment = pd.read_csv(
        ARTIFACT_DIRECTORY / "local_pair_assignment_manifest.csv"
    )
    assert len(assignment) == 324
    assert assignment["pair_id"].nunique() == 324
    assert set(assignment.groupby("assignment_stratum_id")["case_id"].nunique()) == {12}
    assert set(assignment.groupby("case_id").size()) == {27}
    assert not assignment[
        [
            "prediction_used_for_assignment",
            "truth_used_for_assignment",
            "error_used_for_assignment",
            "truth_optimum_used_for_assignment",
        ]
    ].astype(bool).any().any()
    assert sha256_file(
        ARTIFACT_DIRECTORY / "local_pair_assignment_manifest.csv"
    ) == manifest["local_validation_protocol"]["assignment_sha256"]


def test_bundle_assignment_is_balanced_and_truth_free(manifest) -> None:
    assignment = pd.read_csv(
        ARTIFACT_DIRECTORY / "bundle_pair_assignment_manifest.csv"
    )
    assert len(assignment) == 648
    assert assignment["bundle_pair_id"].nunique() == 648
    assert set(assignment.groupby("assignment_stratum_id")["case_id"].nunique()) == {12}
    assert set(assignment.groupby("case_id").size()) == {54}
    assert not assignment[
        [
            "prediction_used_for_assignment",
            "truth_used_for_assignment",
            "error_used_for_assignment",
            "truth_optimum_used_for_assignment",
        ]
    ].astype(bool).any().any()
    assert sha256_file(
        ARTIFACT_DIRECTORY / "bundle_pair_assignment_manifest.csv"
    ) == manifest["bundle_validation_protocol"]["assignment_sha256"]


def test_manifest_was_verified_before_every_truth_stage(metadata) -> None:
    audit = pd.read_csv(
        ARTIFACT_DIRECTORY / "calibration_truth_access_audit.csv"
    )
    assert len(audit) == 24 == metadata["calibration_truth_gate_access_count"]
    assert audit["manifest_verified_before_truth"].astype(bool).all()
    assert set(audit["manifest_sha256"]) == {metadata["calibration_manifest_sha256"]}
    assert not audit["truth_used_for_selection_or_assignment"].astype(bool).any()


def test_one_step_residual_is_directly_computed(one_step) -> None:
    expected = np.abs(one_step["deltaJ_pred"] - one_step["deltaJ_truth"])
    assert len(one_step) == 324
    assert np.allclose(one_step["e_deltaJ_1"], expected, atol=1e-12, rtol=0.0)
    assert set(one_step["residual_computation"]) == {
        "DIRECT_PAIRED_ENDPOINT_DIFFERENCE"
    }


@pytest.mark.parametrize("length", [2, 3, 5])
def test_each_bundle_residual_is_directly_computed(bundles, length) -> None:
    selected = bundles.loc[bundles["bundle_length"].eq(length)]
    expected = np.abs(selected["deltaJ_pred"] - selected["deltaJ_truth"])
    assert len(selected) == 216
    assert np.allclose(selected["e_deltaJ_bundle"], expected, atol=1e-12, rtol=0.0)
    assert set(selected["residual_computation"]) == {
        "DIRECT_START_TO_ENDPOINT_DIFFERENCE"
    }


def test_no_analytic_bundle_scaling_formula_was_used(bundles, metadata) -> None:
    assert not bundles[
        [
            "n_times_one_step_uncertainty_used",
            "sqrt_n_times_one_step_uncertainty_used",
            "analytic_scaling_formula_used",
        ]
    ].astype(bool).any().any()
    assert metadata["analytic_uncertainty_scaling_used"] is False


def test_one_step_reverse_direction_is_labelled_nonindependent(one_step, metadata) -> None:
    assert np.allclose(one_step["reverse_e_deltaJ_1"], one_step["e_deltaJ_1"])
    assert not one_step["negative_direction_is_independent_pair"].astype(bool).any()
    summary = pd.read_csv(
        ARTIFACT_DIRECTORY / "axis_direction_residual_summary.csv"
    )
    local = summary.loc[
        summary["decision_scale"].eq("1-step")
        & summary["summary_scope"].eq("AXIS_DIRECTION")
    ]
    assert set(local["direction"]) == {"POSITIVE", "NEGATIVE"}
    assert "SYMMETRIC_REVERSE_VIEW_NOT_AN_INDEPENDENT_PAIR" in set(
        local["direction_evidence_status"]
    )
    assert metadata["one_step_negative_direction_independent_pair_count"] == 0


def test_residual_summaries_use_direct_independent_samples(one_step) -> None:
    summary = pd.read_csv(
        ARTIFACT_DIRECTORY / "decision_scale_residual_comparison.csv"
    ).set_index("decision_scale")
    expected = residual_distribution(one_step["e_deltaJ_1"])
    for field in ("P90", "P95", "P99", "max"):
        assert np.isclose(summary.loc["1-step", field], expected[field], atol=1e-12)
    assert summary["direct_endpoint_residual"].astype(bool).all()
    assert not summary["threshold_selected"].astype(bool).any()


def test_development_and_independent_residuals_are_not_pooled(metadata) -> None:
    comparison = pd.read_csv(
        ARTIFACT_DIRECTORY / "development_vs_independent_calibration.csv"
    )
    assert not comparison[
        ["pooled_distribution_computed", "pooled_threshold_computed"]
    ].astype(bool).any().any()
    assert metadata["development_and_calibration_residuals_pooled"] is False
    old = comparison.loc[
        comparison["evidence_source"].eq("OLD_DEVELOPMENT_LOCAL")
    ].iloc[0]
    assert np.isclose(old["P95"], LOCAL_P95, atol=1e-12, rtol=0.0)


def test_all_bundle_scales_have_complete_research_calibration(metadata) -> None:
    feasibility = pd.read_csv(
        ARTIFACT_DIRECTORY / "bundle_scale_feasibility.csv"
    )
    assert set(feasibility["bundle_length"]) == {2, 3, 5}
    assert set(feasibility["calibration_status"]) == {BUNDLE_SCALE_CALIBRATED}
    assert not feasibility["formal_threshold_ready"].astype(bool).any()
    assert not feasibility["policy_enabled"].astype(bool).any()
    assert set(metadata["bundle_scale_status"].values()) == {BUNDLE_SCALE_CALIBRATED}


def test_bundle_scale_trend_is_empirical_not_a_frozen_formula(metadata) -> None:
    assert metadata["bundle_2_3_5_P95_strictly_increasing"] is True
    assert metadata["all_1_2_3_5_P95_strictly_increasing"] is False
    comparison = pd.read_csv(
        ARTIFACT_DIRECTORY / "decision_scale_residual_comparison.csv"
    )
    assert comparison["empirical_trend_only"].astype(bool).all()
    assert not comparison["formula_fitted_or_frozen"].astype(bool).any()


def test_no_new_percentile_k_or_policy_was_selected(metadata, manifest) -> None:
    assert metadata["new_percentile_selected"] is False
    assert metadata["new_K_selected"] is False
    assert metadata["new_policy_implemented"] is False
    assert metadata["cumulative_rule_enabled"] is False
    assert manifest["threshold_selection_allowed"] is False


def test_no_prospective_personalization_was_run(metadata, manifest) -> None:
    assert metadata["prospective_personalization_run"] is False
    assert manifest["prospective_personalization_allowed"] is False
    assert metadata["rejected_prospective_cases_used_in_residual_estimate"] is False


def test_equivalence_tolerance_and_support_gate_are_unchanged(metadata) -> None:
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert metadata["algorithm_equivalence_tolerance"] == 0.005
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert metadata["support_gate_percent"] == 90.0


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


def test_objective_generator_model_p2_and_robot_trees_are_unchanged(metadata) -> None:
    before = metadata["protected_source_sha256_before"]
    after = metadata["protected_source_sha256_after"]
    current = _calibration_protected_hashes()
    for name in (
        "mechanical_objective",
        "generator",
        "five_parameter_estimator",
        "P2_V1_core",
        "P2_V2A_definition",
        "tree:hardware",
        "tree:control",
        "tree:collection",
        "tree:safety",
    ):
        assert before[name] == after[name] == current[name]


def test_no_robot_or_human_approval(metadata) -> None:
    assert metadata["robot_connected"] is False
    assert metadata["robot_motion_approval"] == "NOT_ROBOT_MOTION_APPROVED"
    assert metadata["human_readiness"] == "NOT_HUMAN_READY"


def test_evidence_can_enter_design_but_no_design_was_created(metadata) -> None:
    assert metadata["next_revision_policy_design_evidence_available"] is True
    assert metadata["next_revision_policy_designed_or_enabled"] is False


def test_all_required_artifacts_exist_and_match_manifest(metadata) -> None:
    names = [
        *REQUIRED_CSV_FILENAMES,
        *EXTRA_CSV_FILENAMES,
        *JSON_FILENAMES,
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
    ]
    assert set(metadata["artifact_manifest"]) == set(names)
    for name in names:
        path = ARTIFACT_DIRECTORY / name
        assert path.is_file() and path.stat().st_size > 0, name
        record = metadata["artifact_manifest"][name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert path.stat().st_size == record["bytes"]


def test_new_source_has_no_robot_side_imports() -> None:
    paths = (
        Path("lower_limb_sim/p2_next_revision_independent_calibration.py"),
        Path("lower_limb_sim/run_p2_next_revision_independent_calibration.py"),
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


def test_recomputed_case_manifest_is_identical(cases) -> None:
    rebuilt = calibration_case_manifest()
    pd.testing.assert_frame_equal(rebuilt, cases, check_dtype=False)
