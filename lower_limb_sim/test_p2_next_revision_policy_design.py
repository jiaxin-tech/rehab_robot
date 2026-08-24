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
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_next_revision_policy_design import (
    CALIBRATED_BUNDLE_LENGTHS,
    EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
    EXPECTED_CALIBRATION_MANIFEST_SHA256,
    EXPECTED_LOCAL_PAIR_PLAN_SHA256,
    FINAL_REVISE,
    FrozenPolicyDesignManifestGate,
    POLICY_VARIANTS,
    candidate_manifest_payload,
    canonical_json_bytes,
    evaluate_bundle_options,
    load_calibration_uncertainty,
    select_bundle_authorization,
    sha256_file,
)
from .run_p2_next_revision_policy_design import (
    DEFAULT_OUTPUT_DIRECTORY,
    REQUIRED_OUTPUT_FILENAMES,
    _checkpoint_preflight,
    _case_table,
)


MODULE_DIR = Path(__file__).resolve().parent
ARTIFACT = DEFAULT_OUTPUT_DIRECTORY
METADATA_PATH = ARTIFACT / "metadata.json"
MANIFEST_PATH = ARTIFACT / "POLICY_DESIGN_CANDIDATE_MANIFEST_V1.json"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "1102654b003ca3899021dc2e43c3d682053b7e49082e46b3a722b0495db06166"
)


def _metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_independent_calibration_checkpoint_and_frozen_shas() -> None:
    checkpoint = _checkpoint_preflight()
    assert checkpoint["checkpoint_verified"] is True
    assert checkpoint["calibration_commit"] == checkpoint["checkpoint_commit"]
    assert checkpoint["calibration_artifact_tracked_count"] >= 30
    assert sha256_file(
        MODULE_DIR
        / "formal_artifacts"
        / "p2_next_revision_independent_calibration_v1"
        / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
    ) == EXPECTED_CALIBRATION_MANIFEST_SHA256
    assert sha256_file(
        MODULE_DIR
        / "formal_artifacts"
        / "post_prospective_rejection_root_cause_audit_v1"
        / "designated_bundle_validation_pair_plan.csv"
    ) == EXPECTED_BUNDLE_PAIR_PLAN_SHA256
    assert sha256_file(
        MODULE_DIR
        / "formal_artifacts"
        / "p2_v2_formal_research_protocol_v1"
        / "designated_local_validation_pair_plan.csv"
    ) == EXPECTED_LOCAL_PAIR_PLAN_SHA256


def test_candidate_manifest_is_exact_frozen_pretruth_file() -> None:
    manifest = _manifest()
    metadata = _metadata()
    assert sha256_file(MANIFEST_PATH) == EXPECTED_CANDIDATE_MANIFEST_SHA256
    assert metadata["candidate_manifest_sha256"] == EXPECTED_CANDIDATE_MANIFEST_SHA256
    assert manifest["status"] == "FROZEN_BEFORE_DEVELOPMENT_SHADOW_TRUTH"
    assert manifest["truth_used_to_create_candidates"] is False
    assert manifest["truth_may_modify_candidates_in_this_task"] is False
    assert MANIFEST_PATH.read_bytes() == canonical_json_bytes(manifest)


def test_frozen_manifest_gate_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(MANIFEST_PATH.read_bytes())
    gate = FrozenPolicyDesignManifestGate(path, EXPECTED_CANDIDATE_MANIFEST_SHA256)
    gate.require_frozen()
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="frozen policy candidate"):
        gate.record_truth_access("FORBIDDEN_AFTER_MUTATION")


def test_minimal_variants_percentiles_lengths_and_default_off() -> None:
    manifest = _manifest()
    assert [spec.policy_id for spec in POLICY_VARIANTS] == [
        "R0_P2_V1_G0_NO_BUNDLE_S0",
        "R1_G0_BUNDLE_SCALE_P95_S0",
        "R2_G0_BUNDLE_SCALE_P95_S2",
        "R3_G0_BUNDLE_SCALE_P99_S2",
        "R4_G0_BUNDLE_SCALE_AXIS_P95_S2",
    ]
    assert all(spec.default_enabled is False for spec in POLICY_VARIANTS)
    assert set(manifest["bundle_uncertainty_candidates"]["percentile_levels"]) == {
        "P95",
        "P99",
    }
    assert tuple(manifest["bundle_rule"]["allowed_lengths"]) == CALIBRATED_BUNDLE_LENGTHS
    assert manifest["bundle_uncertainty_candidates"]["percentile_search_performed"] is False
    assert manifest["bundle_uncertainty_candidates"]["analytic_n_times_U1_used"] is False
    assert manifest["bundle_uncertainty_candidates"]["analytic_sqrt_n_U1_used"] is False


def test_scale_axis_p95_is_one_uniform_percentile_not_axis_percentile_search() -> None:
    calibration = load_calibration_uncertainty()
    assert calibration.one_step_p95 == pytest.approx(0.00196762280892)
    assert calibration.scale_p95 == pytest.approx(
        {2: 0.00164217053717, 3: 0.00244544326845, 5: 0.00400496043747}
    )
    assert calibration.scale_p99 == pytest.approx(
        {2: 0.00338315735589, 3: 0.00519358473251, 5: 0.00792314538021}
    )
    assert calibration.scale_axis_p95[5]["knee"] > calibration.scale_axis_p95[5]["hip"]
    assert calibration.scale_axis_p95[5]["hip"] > calibration.scale_axis_p95[5]["phase"]


def test_bundle_geometry_is_same_axis_direction_and_authorizes_first_step_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for step in range(6):
        knee = -0.25 * step
        rows.append(
            {
                "case_id": "synthetic__unit",
                "subject_id": "synthetic",
                "scenario_name": "unit",
                "trajectory_id": f"knee_{step}",
                "hip_delta": 0.0,
                "knee_delta": knee,
                "phase_delta": 0.0,
                "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
                "geometrically_admissible": True,
                "J_pred": 1.0 - 0.004 * step,
                "domain_coverage": 100.0,
                "model_supported": True,
            }
        )
    table = pd.DataFrame(rows)
    monkeypatch.setattr(
        "lower_limb_sim.p2_next_revision_policy_design._patient_valid",
        lambda key, cache: True,
    )
    options = evaluate_bundle_options(
        table,
        current=__import__(
            "lower_limb_sim.sequential_personalization", fromlist=["SearchAlpha"]
        ).SearchAlpha(),
        spec=POLICY_VARIANTS[1],
        uncertainty=load_calibration_uncertainty(),
        iteration=1,
        patient_validity_cache={},
    )
    selected = select_bundle_authorization(options)
    assert selected is not None
    assert selected["coordinate"] == "knee"
    assert selected["direction"] == "NEGATIVE"
    assert int(selected["bundle_length"]) in CALIBRATED_BUNDLE_LENGTHS
    assert selected["first_step_trajectory_id"] == "knee_1"
    assert selected["endpoint_trajectory_id"] != selected["first_step_trajectory_id"]
    assert bool(selected["same_axis"])
    assert bool(selected["direction_consistent"])
    assert not bool(selected["mixed_axis"])
    assert not bool(selected["queued_later_steps"])
    assert int(selected["authorized_execution_count"]) == 1


def test_frozen_scientific_invariants_and_stopping_k() -> None:
    manifest = _manifest()
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert tuple(FORMAL_HIP_ROM_DEG) == (0.0, 120.0)
    assert tuple(FORMAL_KNEE_ROM_DEG) == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert manifest["model_support_gate_percent"] == 90.0
    assert manifest["stopping_k_values"] == [2]
    assert not any(spec.stopping_k not in (None, 2) for spec in POLICY_VARIANTS)


def test_data_roles_exclude_calibration_future_and_heldout() -> None:
    cases = _case_table()
    assert len(cases) == 15
    assert set(cases["development_origin"]) == {
        "ORIGINAL_P2_DEVELOPMENT",
        "REJECTED_PROSPECTIVE_NOW_DEVELOPMENT",
    }
    assert not cases["case_id"].str.contains("calibration_subject").any()
    metadata = _metadata()
    assert metadata["calibration_cases_used_for_policy_outcome_selection"] is False
    assert metadata["future_prospective_generated"] is False
    assert metadata["held_out_final_test_read"] is False
    assert metadata["development_truth_modified_candidate"] is False


def test_required_outputs_and_artifact_hash_manifest() -> None:
    metadata = _metadata()
    for name in REQUIRED_OUTPUT_FILENAMES:
        assert (ARTIFACT / name).is_file(), name
    for name, record in metadata["artifact_manifest"].items():
        path = ARTIFACT / name
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]


def test_every_execution_refits_and_recomputes_without_queued_bundle() -> None:
    history = pd.read_csv(ARTIFACT / "policy_shadow_trial_history.csv")
    assert history["model_refit_after_execution"].astype(bool).all()
    assert history["full_map_recomputed_after_execution"].astype(bool).all()
    assert not history["queued_later_bundle_steps"].astype(bool).any()
    assert history["bundle_authorized_execution_count"].isin((0, 1)).all()
    case_summary = pd.read_csv(ARTIFACT / "policy_shadow_case_summary.csv")
    assert (
        case_summary["whole_map_recomputation_count"]
        == case_summary["number_of_executed_trials"] + 1
    ).all()
    assert (
        case_summary["model_update_count"]
        == case_summary["number_of_executed_trials"]
    ).all()


def test_r0_reproduces_rejected_prospective_p2_v1_exactly() -> None:
    current = pd.read_csv(ARTIFACT / "policy_shadow_case_summary.csv")
    current = current.loc[current["policy_id"].eq(POLICY_VARIANTS[0].policy_id)]
    historical = pd.read_csv(
        MODULE_DIR
        / "formal_artifacts"
        / "post_prospective_rejection_root_cause_audit_v1"
        / "factorial_policy_summary.csv"
    )
    historical = historical.loc[historical["policy_id"].eq("P2_V1_G0_C0_S0")]
    joined = current.merge(historical, on="case_id", suffixes=("_new", "_old"))
    assert len(joined) == 6
    for column in (
        "number_of_executed_trials",
        "number_of_explore_trials",
        "number_of_exploit_trials",
        "missed_improvement_rounds",
        "final_best_actual_J",
        "global_truth_regret",
    ):
        assert np.array_equal(
            joined[f"{column}_new"].to_numpy(),
            joined[f"{column}_old"].to_numpy(),
        )


def test_shadow_result_is_honest_nonrecovery_and_not_policy_selection() -> None:
    summary = pd.read_csv(ARTIFACT / "policy_shadow_summary.csv")
    assert set(summary["bundle_authorized_trials"]) == {0}
    assert not summary["policy_outcomes_used_to_select_candidate"].astype(bool).any()
    small = pd.read_csv(ARTIFACT / "small_step_recovery_audit.csv")
    assert small["path_id"].nunique() == 9
    assert not small["first_step_authorized"].astype(bool).any()
    assert not small["recovered_true_accumulation_path"].astype(bool).any()
    assert small["bundle_scale_used"].notna().all()
    assert small["bundle_lower_bound_margin"].lt(0.0).all()
    assert _metadata()["final_status"] == FINAL_REVISE


def test_s2_reduces_exploration_but_is_not_declared_safe() -> None:
    stopping = pd.read_csv(ARTIFACT / "stopping_shadow_comparison.csv")
    r1 = stopping.loc[stopping["policy_id"].eq(POLICY_VARIANTS[1].policy_id)].iloc[0]
    r2 = stopping.loc[stopping["policy_id"].eq(POLICY_VARIANTS[2].policy_id)].iloc[0]
    assert int(r2["trials_removed_vs_R1"]) == 36
    assert int(r2["EXPLORE"]) < int(r1["EXPLORE"])
    assert float(r2["final_J_change_vs_R1"]) == 0.0
    failures = pd.read_csv(ARTIFACT / "new_failure_mode_audit.csv")
    assert failures.loc[
        failures["policy_id"].eq(POLICY_VARIANTS[2].policy_id)
        & failures["failure_mode"].eq("premature_S2_stop"),
        "observed",
    ].astype(bool).any()


def test_no_robot_import_and_protected_trees_unchanged() -> None:
    forbidden = {"hardware", "control", "collection", "safety", "xCoreSDK_python"}
    for path in (MODULE_DIR / "p2_next_revision_policy_design.py", MODULE_DIR / "run_p2_next_revision_policy_design.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert roots.isdisjoint(forbidden)
    metadata = _metadata()
    assert metadata["protected_source_sha256_before"] == metadata[
        "protected_source_sha256_after"
    ]
    assert metadata["P2_V1_replaced"] is False
    assert metadata["new_policy_default_enabled"] is False
    assert metadata["robot_connected"] is False
    assert metadata["human_ready"] == "NOT_HUMAN_READY"
    assert metadata["robot_motion_approved"] == "NOT_ROBOT_MOTION_APPROVED"


def test_manifest_reconstruction_is_independent_of_shadow_outputs() -> None:
    uncertainty = load_calibration_uncertainty()
    frozen = _manifest()
    reconstructed = candidate_manifest_payload(
        uncertainty,
        checkpoint_commit=frozen["checkpoint_commit"],
        protected_source_sha256=frozen["protected_source_sha256"],
    )
    assert reconstructed == frozen

