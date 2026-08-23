from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
from lower_limb_sim.p2_v2_prospective_offline_validation import (
    DEVELOPMENT_CASES,
    FINAL_STATUSES,
    FROZEN_PAIR_PLAN_SHA256,
    LOCAL_MAX,
    LOCAL_P95,
    LOCAL_P99,
    MANIFEST_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    POLICY_VARIANTS,
    ProspectivePolicySpec,
    audit_bundle_uncertainty,
    build_prospective_manifest,
    prospective_case_rows,
    prospective_subject_definitions,
    run_prospective_policy,
    stable_manifest_sha256,
    validate_frozen_local_evidence,
)
from lower_limb_sim.run_p2_v2_prospective_offline_validation import (
    CSV_FILENAMES,
    DEFAULT_OUTPUT_DIRECTORY,
    FIGURE_FILENAMES,
    MANIFEST_FILENAME,
    REPORT_FILENAMES,
    _protected_source_hashes,
)


ARTIFACT_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((ARTIFACT_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(
        (ARTIFACT_DIRECTORY / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIRECTORY / "prospective_case_manifest.csv")


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIRECTORY / "prospective_policy_summary.csv")


def test_prospective_start_commit_is_recorded(metadata) -> None:
    assert metadata["prospective_start_commit_sha"] == (
        "d7fe80945ae625fffc7919e1735e9e2df8c8fa00"
    )


def test_manifest_identity_and_freeze_precede_truth(metadata, manifest) -> None:
    assert manifest["manifest_id"] == MANIFEST_ID
    assert metadata["first_truth_after_manifest_freeze"] is True
    assert metadata["prospective_truth_gate_access_count"] > 0


def test_manifest_sha_is_deterministic(metadata, manifest) -> None:
    assert stable_manifest_sha256(manifest) == metadata["prospective_manifest_sha256"]
    path = ARTIFACT_DIRECTORY / MANIFEST_FILENAME
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata[
        "prospective_manifest_sha256"
    ]


def test_manifest_can_be_rebuilt_from_frozen_inputs(metadata, manifest) -> None:
    rebuilt = build_prospective_manifest(
        metadata["prospective_start_commit_sha"],
        protected_source_sha256=manifest["protected_source_sha256"],
    )
    assert stable_manifest_sha256(rebuilt) == metadata["prospective_manifest_sha256"]


def test_development_cases_are_excluded(cases) -> None:
    assert not set(cases["case_id"]).intersection(DEVELOPMENT_CASES)
    assert not cases["development_case"].astype(bool).any()


def test_prospective_cohort_has_three_matched_and_three_mismatch(cases) -> None:
    assert len(cases) == 6
    assert int(cases["case_class"].eq("PROSPECTIVE_MATCHED").sum()) == 3
    assert int(cases["case_class"].eq("PROSPECTIVE_MODEL_MISMATCH").sum()) == 3


def test_subject_selection_is_deterministic_and_truth_free() -> None:
    first = prospective_subject_definitions()
    second = prospective_subject_definitions()
    assert first == second
    assert len(first) == 3
    assert all(item["truth_used_for_selection"] is False for item in first)


def test_heldout_final_test_was_not_read(metadata, manifest) -> None:
    assert metadata["heldout_final_test"] == "HELD_OUT_FINAL_TEST_NOT_READ"
    assert manifest["held_out_final_test"]["read_during_experiment"] is False


def test_pair_plan_sha_and_count_are_unchanged() -> None:
    provenance, _ = validate_frozen_local_evidence()
    assert set(provenance["pair_plan_sha256"]) == {FROZEN_PAIR_PLAN_SHA256}
    assert set(provenance["pair_count"]) == {324}


def test_local_metrics_are_recomputed_from_frozen_designated_evidence() -> None:
    _, metrics = validate_frozen_local_evidence()
    assert np.isclose(metrics["local_max"], LOCAL_MAX, atol=1e-12, rtol=0.0)
    assert np.isclose(metrics["local_P95"], LOCAL_P95, atol=1e-12, rtol=0.0)
    assert np.isclose(metrics["local_P99"], LOCAL_P99, atol=1e-12, rtol=0.0)


def test_prospective_cases_do_not_contribute_to_local_calibration(cases) -> None:
    result = pd.read_csv(
        Path("lower_limb_sim/formal_artifacts/p2_v2_offline_research_prototype_v1/local_validation_results.csv")
    )
    assert not set(cases["case_id"]).intersection(set(result["case_id"]))


def test_prospective_truth_did_not_update_uncertainty(summary) -> None:
    assert not summary["prospective_truth_updated_local_calibration"].astype(bool).any()
    history = pd.read_csv(ARTIFACT_DIRECTORY / "prospective_trial_history.csv")
    p95 = history.loc[history["policy_id"].str.contains("G2")]
    p99 = history.loc[history["policy_id"].str.contains("G3")]
    assert np.allclose(p95["decision_uncertainty_bound"], LOCAL_P95, atol=1e-12)
    assert np.allclose(p99["decision_uncertainty_bound"], LOCAL_P99, atol=1e-12)


def test_bundle_uncertainty_never_uses_n_or_sqrt_n() -> None:
    audit = pd.read_csv(ARTIFACT_DIRECTORY / "bundle_uncertainty_audit.csv")
    assert not audit["n_times_one_step_assumed"].astype(bool).any()
    assert not audit["sqrt_n_times_one_step_assumed"].astype(bool).any()


def test_uncalibrated_cumulative_candidates_fail_closed() -> None:
    audit = pd.read_csv(ARTIFACT_DIRECTORY / "bundle_uncertainty_audit.csv")
    bundles = audit.loc[audit["bundle_length"].gt(1)]
    assert set(bundles["calibration_status"]) == {"SHADOW_ONLY_NOT_CALIBRATED"}
    assert not bundles["active_prospective_policy"].astype(bool).any()
    bad = ProspectivePolicySpec("BAD", "G2_FROZEN_LOCAL_P95", "C2_TWO_STEP", "S2", 2, "TEST")
    with pytest.raises(PermissionError, match="cumulative"):
        run_prospective_policy(None, bad, None, None, None)  # type: ignore[arg-type]


def test_truth_is_after_a_unique_selection_token() -> None:
    history = pd.read_csv(ARTIFACT_DIRECTORY / "prospective_trial_history.csv")
    assert not history["truth_accessed_before_selection"].astype(bool).any()
    assert history["manifest_sha_verified_before_truth"].astype(bool).all()
    assert not history[["case_id", "policy_id", "selection_token"]].duplicated().any()


def test_stopping_uses_no_future_information() -> None:
    history = pd.read_csv(
        ARTIFACT_DIRECTORY / "prospective_exploration_value_history.csv"
    )
    assert not history["future_truth_used_by_stopping"].astype(bool).any()
    assert not history["future_exploit_used_by_stopping"].astype(bool).any()
    assert not history["future_best_used_by_stopping"].astype(bool).any()
    assert not history["support_alone_used_as_decision_value"].astype(bool).any()


def test_p2_v1_variant_remains_g0_c0_s0(manifest) -> None:
    variants = {item["policy_variant_id"]: item for item in manifest["policy_variants"]}
    v1 = variants["P2_V1_G0_C0_S0"]
    assert (v1["guard_id"], v1["cumulative_rule_id"], v1["stopping_rule_id"]) == (
        "G0_CURRENT_GLOBAL_MAX",
        "C0_SINGLE_STEP",
        "S0_CURRENT_CONTINUATION",
    )


def test_p2_v2a_variant_is_frozen_g2_c0_s2(manifest) -> None:
    variants = {item["policy_variant_id"]: item for item in manifest["policy_variants"]}
    v2 = variants["P2_V2A_G2_C0_S2"]
    assert (v2["guard_id"], v2["cumulative_rule_id"], v2["stopping_k"]) == (
        "G2_FROZEN_LOCAL_P95",
        "C0_SINGLE_STEP",
        2,
    )


def test_p99_and_k_variants_are_sensitivity_only(manifest) -> None:
    sensitivity = [
        item for item in manifest["policy_variants"] if "SENSITIVITY" in item["role"]
    ]
    assert len(sensitivity) == 3
    assert {item["stopping_k"] for item in sensitivity} == {1, 2, 3}


def test_p2_v2_remains_default_off(metadata, manifest) -> None:
    assert metadata["P2_V2_default_enabled"] is False
    assert manifest["P2_V2_default_enabled"] is False


def test_equivalence_tolerance_remains_0p005(manifest) -> None:
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert manifest["frozen_scientific_baseline"]["algorithm_equivalence_tolerance"] == 0.005


def test_support_gate_remains_90_percent(manifest) -> None:
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert manifest["frozen_scientific_baseline"]["model_support_gate_percent"] == 90.0


def test_objective_generator_and_model_sources_are_unchanged(metadata) -> None:
    before = metadata["protected_source_sha256_before"]
    after = metadata["protected_source_sha256_after"]
    for name in ("mechanical_objective", "generator", "five_parameter_estimator", "P2_V1_core"):
        assert before[name] == after[name]


def test_active_reference_sha_is_unchanged(metadata, manifest) -> None:
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    assert manifest["frozen_scientific_baseline"]["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["protected_source_sha256_before"]["active_reference"] == ACTIVE_REFERENCE_SHA256


def test_rom_and_theta_shank_are_unchanged(manifest) -> None:
    baseline = manifest["frozen_scientific_baseline"]
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert tuple(FORMAL_HIP_ROM_DEG) == (0.0, 120.0)
    assert tuple(FORMAL_KNEE_ROM_DEG) == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert baseline["theta_shank_definition"] == "q_hip - q_knee"


def test_hardware_control_collection_safety_are_zero_diff(metadata) -> None:
    before = metadata["protected_source_sha256_before"]
    after = metadata["protected_source_sha256_after"]
    for name in ("tree:hardware", "tree:control", "tree:collection", "tree:safety"):
        assert before[name] == after[name]
    current = _protected_source_hashes()
    assert all(current[name] == after[name] for name in after)


def test_no_robot_connection_or_motion_approval(metadata) -> None:
    assert metadata["robot_connected"] is False
    assert metadata["robot_motion_approval"] == NOT_ROBOT_MOTION_APPROVED


def test_no_human_ready_approval(metadata) -> None:
    assert metadata["human_readiness"] == NOT_HUMAN_READY
    assert "REQUIRES_REVIEW" in metadata["initial_identification_acceptance_rule"]
    assert "NOT_FROZEN_FOR_HUMANS" in metadata["global_model_reliability_rule"]


def test_every_case_policy_variant_ran_once(cases, summary) -> None:
    assert len(summary) == len(cases) * len(POLICY_VARIANTS) == 30
    assert not summary[["case_id", "policy_id"]].duplicated().any()


def test_every_model_update_recomputed_full_map(summary) -> None:
    assert (
        summary["whole_map_recomputation_count"]
        == summary["number_of_executed_trials"] + 1
    ).all()


def test_small_step_paths_never_activate_cumulative_policy() -> None:
    audit = pd.read_csv(ARTIFACT_DIRECTORY / "cumulative_prospective_comparison.csv")
    assert audit["small_step_accumulation_case"].astype(bool).any()
    assert set(audit["bundle_rule_status"]) == {"SHADOW_ONLY_NOT_CALIBRATED"}


def test_subject_specificity_uses_matched_cases_only(cases) -> None:
    specificity = pd.read_csv(
        ARTIFACT_DIRECTORY / "prospective_subject_specificity.csv"
    )
    matched = set(cases.loc[cases["case_class"].eq("PROSPECTIVE_MATCHED"), "case_id"])
    assert set(specificity["case_id"]) == matched
    assert not specificity["truth_used_to_modify_policy"].astype(bool).any()


def test_boundary_optimum_is_not_automatically_policy_failure() -> None:
    specificity = pd.read_csv(
        ARTIFACT_DIRECTORY / "prospective_subject_specificity.csv"
    )
    truth_boundary = specificity["truth_optimum_on_generator_boundary"].astype(bool)
    assert truth_boundary.any()
    assert set(specificity.loc[truth_boundary, "boundary_classification"]) == {
        "OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM"
    }


def test_all_required_artifacts_exist_and_are_nonempty() -> None:
    names = [MANIFEST_FILENAME, *CSV_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES, "metadata.json"]
    for name in names:
        path = ARTIFACT_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_final_status_is_one_of_three_allowed_states(metadata) -> None:
    assert metadata["final_status"] in FINAL_STATUSES


def test_report_preserves_final_status_and_default_off(metadata) -> None:
    report = (ARTIFACT_DIRECTORY / "PROSPECTIVE_VALIDATION_REPORT.md").read_text(
        encoding="utf-8"
    )
    assert metadata["final_status"] in report
    assert "P2 V2 remains default-off" in report
    assert NOT_HUMAN_READY in report
    assert NOT_ROBOT_MOTION_APPROVED in report


def test_new_source_has_no_robot_side_imports() -> None:
    paths = (
        Path("lower_limb_sim/p2_v2_prospective_offline_validation.py"),
        Path("lower_limb_sim/run_p2_v2_prospective_offline_validation.py"),
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
