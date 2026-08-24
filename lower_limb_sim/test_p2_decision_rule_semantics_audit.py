from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from .formal_protocol import ACTIVE_REFERENCE_SHA256, THETA_SHANK_DEFINITION
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_decision_rule_semantics_audit import (
    ADDITIVE_ASSUMPTION,
    EXPECTED_CALIBRATION_MANIFEST_SHA256,
    SEMANTIC_VARIANTS,
    FrozenSemanticsManifestGate,
    SemanticsCalibration,
    apply_semantic_decision_guard,
    candidate_manifest_payload,
    canonical_json_bytes,
    evaluate_bundle_semantics,
    load_semantics_calibration,
    select_bundle_authorization,
    sha256_file,
    small_step_semantic_recovery,
)
from .research_decision_guarded_sequential_personalization import (
    ResearchDecisionUncertainty,
    apply_research_decision_guard,
)
from .run_p2_decision_rule_semantics_audit import (
    CORE_SOURCE_PATH,
    RUNNER_SOURCE_PATH,
    _case_table,
    _checkpoint_preflight,
)
from .sequential_personalization import SearchAlpha


def _uncertainty(bound: float = 0.002) -> ResearchDecisionUncertainty:
    return ResearchDecisionUncertainty(
        case_id="unit",
        iteration=0,
        pairwise_audit=pd.DataFrame(),
        maximum_observed_e_delta_j=bound,
        p95_observed_e_delta_j=bound,
        p99_observed_e_delta_j=bound,
        validation_pair_count=12,
        bound_used_by_guard=bound,
        bound_type="UNIT",
        bound_status="UNIT",
    )


def _local_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trajectory_id": "current",
                "hip_delta": 0.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": 1.0,
                "model_supported": True,
                "geometrically_admissible": True,
            },
            {
                "trajectory_id": "candidate",
                "hip_delta": 1.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": 0.994,
                "model_supported": True,
                "geometrically_admissible": True,
            },
        ]
    )


def test_calibration_roles_and_values_are_frozen() -> None:
    calibration = load_semantics_calibration()
    assert sha256_file(
        Path(__file__).parent
        / "formal_artifacts"
        / "p2_next_revision_independent_calibration_v1"
        / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
    ) == EXPECTED_CALIBRATION_MANIFEST_SHA256
    assert calibration.one_step_pair_count == 324
    assert calibration.one_step_p95 == pytest.approx(0.00196762280892)
    assert dict(calibration.bundle_scale_p95) == pytest.approx(
        {2: 0.00164217053717, 3: 0.00244544326845, 5: 0.00400496043747}
    )
    assert int(calibration.direction_evidence["direction_support_count"].sum()) == 306
    assert int(
        calibration.direction_evidence["direction_contradiction_count"].sum()
    ) == 18
    assert calibration.direction_evidence[
        "direction_supported_by_majority"
    ].astype(bool).all()


def test_manifest_freezes_semantics_without_development_truth(tmp_path: Path) -> None:
    calibration = load_semantics_calibration()
    payload = candidate_manifest_payload(
        calibration,
        checkpoint_commit="checkpoint",
        protected_source_sha256={"P2_V1": "abc"},
    )
    assert payload["current_semantics"]["classification"] == ADDITIVE_ASSUMPTION
    assert payload["rules"]["S0"]["gate"] == "I_pred > 0.005 + U"
    assert payload["rules"]["S2"]["magnitude_gate"] == "I_pred > 0.005"
    assert payload["rules"]["S2"]["direction_gate"] == "deltaJ_pred + U_P95 < 0"
    assert payload["rules"]["S3"]["authorization_scope"] == "NEXT_ONE_FORMAL_GRID_STEP_ONLY"
    assert payload["truth_used_to_create_or_select_semantics"] is False
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    gate = FrozenSemanticsManifestGate(path, digest)
    gate.record_truth_access("UNIT_AFTER_FREEZE")
    assert gate.truth_access_count == 1
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        gate.record_truth_access("MUTATED")


def test_two_gate_is_not_additive_and_does_not_change_base_guard() -> None:
    calibration = load_semantics_calibration()
    table = _local_table()
    uncertainty = _uncertainty(0.002)
    before = apply_research_decision_guard(table, SearchAlpha(), uncertainty)
    current = apply_semantic_decision_guard(
        table, SearchAlpha(), uncertainty, SEMANTIC_VARIANTS[0], calibration
    )
    interval = apply_semantic_decision_guard(
        table, SearchAlpha(), uncertainty, SEMANTIC_VARIANTS[2], calibration
    )
    after = apply_research_decision_guard(table, SearchAlpha(), uncertainty)
    selected_current = current.loc[current["trajectory_id"].eq("candidate")].iloc[0]
    selected_interval = interval.loc[interval["trajectory_id"].eq("candidate")].iloc[0]
    assert selected_current["magnitude_gate_pass"]
    assert selected_current["direction_gate_pass"]
    assert not selected_current["research_exploit_eligible"]
    assert selected_interval["magnitude_gate_pass"]
    assert selected_interval["direction_gate_pass"]
    assert selected_interval["research_exploit_eligible"]
    pd.testing.assert_series_equal(
        before["research_exploit_eligible"],
        after["research_exploit_eligible"],
        check_names=True,
    )


def test_S1_uses_calibration_direction_majority() -> None:
    calibration = load_semantics_calibration()
    output = apply_semantic_decision_guard(
        _local_table(), SearchAlpha(), _uncertainty(), SEMANTIC_VARIANTS[1], calibration
    )
    candidate = output.loc[output["trajectory_id"].eq("candidate")].iloc[0]
    assert candidate["candidate_coordinate"] == "hip"
    assert candidate["candidate_trust_level"] == "INITIAL"
    assert candidate["direction_support_count"] == 33
    assert candidate["direction_contradiction_count"] == 3
    assert candidate["research_exploit_eligible"]
    assert not candidate["truth_used_for_semantic_gate"]


def test_bundle_two_gate_authorizes_only_one_next_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = load_semantics_calibration()
    monkeypatch.setattr(
        "lower_limb_sim.p2_decision_rule_semantics_audit._patient_valid",
        lambda _point, _cache: True,
    )
    rows = []
    for step in range(6):
        rows.append(
            {
                "trajectory_id": f"hip_{step}",
                "hip_delta": 0.25 * step,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": 1.0 - 0.0012 * step,
                "geometrically_admissible": True,
                "model_supported": True,
                "domain_coverage": 100.0,
                "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
            }
        )
    table = pd.DataFrame(rows)
    s0 = evaluate_bundle_semantics(
        table,
        SearchAlpha(),
        SEMANTIC_VARIANTS[0],
        calibration,
        iteration=1,
        patient_validity_cache={},
    )
    s3 = evaluate_bundle_semantics(
        table,
        SearchAlpha(),
        SEMANTIC_VARIANTS[3],
        calibration,
        iteration=1,
        patient_validity_cache={},
    )
    assert select_bundle_authorization(s0) is None
    selected = select_bundle_authorization(s3)
    assert selected is not None
    assert selected["first_step_trajectory_id"] == "hip_1"
    assert selected["endpoint_trajectory_id"] == "hip_5"
    assert selected["authorized_execution_count"] == 1
    assert not selected["queued_later_steps"]
    assert not selected["truth_used_for_authorization"]


def test_nine_paths_are_recovered_only_by_frozen_S3_bundle_semantics() -> None:
    output = small_step_semantic_recovery(load_semantics_calibration())
    assert len(output) == 36
    counts = output.groupby("semantic_id")["recovered_small_step_path"].sum()
    assert int(counts[SEMANTIC_VARIANTS[0].semantic_id]) == 0
    assert int(counts[SEMANTIC_VARIANTS[1].semantic_id]) == 0
    assert int(counts[SEMANTIC_VARIANTS[2].semantic_id]) == 0
    assert int(counts[SEMANTIC_VARIANTS[3].semantic_id]) == 9
    assert not output["truth_used_for_authorization"].astype(bool).any()


def test_invariants_data_roles_and_no_robot_imports() -> None:
    checkpoint = _checkpoint_preflight()
    cases = _case_table()
    assert checkpoint["checkpoint_verified"]
    assert len(cases) == 15
    assert cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum() == 9
    assert cases["development_origin"].eq("POST_REJECTION_DEVELOPMENT").sum() == 6
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    source = CORE_SOURCE_PATH.read_text(encoding="utf-8") + RUNNER_SOURCE_PATH.read_text(
        encoding="utf-8"
    )
    assert "heldout_final_test_read_allowed\": False" in source
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
        "connectToRobot",
    ):
        assert forbidden not in source


def test_formal_artifacts_are_manifest_gated_and_remain_shadow_only() -> None:
    root = (
        Path(__file__).parent
        / "formal_artifacts"
        / "p2_decision_rule_semantics_audit_v1"
    )
    required = {
        "DECISION_RULE_SEMANTICS_CANDIDATE_MANIFEST_V1.json",
        "current_rule_semantics.md",
        "semantic_shadow_comparison.csv",
        "small_step_semantic_recovery.csv",
        "false_improvement_semantic_audit.csv",
        "DECISION_RULE_SEMANTICS_REPORT.md",
        "DATA_ROLE_AUDIT.md",
        "metadata.json",
    }
    assert required.issubset({path.name for path in root.iterdir()})
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    manifest = root / "DECISION_RULE_SEMANTICS_CANDIDATE_MANIFEST_V1.json"
    assert sha256_file(manifest) == metadata["candidate_manifest_sha256"]
    assert metadata["final_status"] == "MORE_EVIDENCE_REQUIRED"
    assert metadata["candidate_manifest_frozen_before_truth"]
    assert metadata["calibration_cases_used_for_policy_performance"] is False
    assert metadata["prospective_cohort_run"] is False
    assert metadata["held_out_final_test_read"] is False
    assert metadata["P2_V1_replaced"] is False
    assert metadata["new_policy_implemented"] is False
    assert metadata["robot_connected"] is False
    assert metadata["protected_source_sha256_before"] == metadata[
        "protected_source_sha256_after"
    ]

    summary = pd.read_csv(root / "semantic_shadow_comparison.csv").set_index(
        "semantic_id"
    )
    s0 = summary.loc[SEMANTIC_VARIANTS[0].semantic_id]
    s3 = summary.loc[SEMANTIC_VARIANTS[3].semantic_id]
    assert int(s0["bundle_authorizations"]) == 0
    assert int(s3["bundle_authorizations"]) == 64
    assert int(s0["total_false_improvement"]) == 0
    assert int(s3["total_false_improvement"]) == 0
    assert int(s3["missed_improvement"]) > int(s0["missed_improvement"])
