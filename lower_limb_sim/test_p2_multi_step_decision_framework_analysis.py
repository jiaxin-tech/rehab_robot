from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from .formal_protocol import ACTIVE_REFERENCE_SHA256, THETA_SHANK_DEFINITION
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_multi_step_decision_framework_analysis import (
    DEFAULT_ENABLED,
    FRAMEWORKS,
    PRIOR_SEMANTICS_MANIFEST_PATH,
    PRIOR_SEMANTICS_MANIFEST_SHA256,
    FrozenFrameworkManifestGate,
    canonical_json_bytes,
    evaluate_endpoint_candidates,
    framework_uncertainty,
    load_semantics_calibration,
    manifest_payload,
    select_endpoint_candidate,
)
from .run_p2_multi_step_decision_framework_analysis import (
    CORE_SOURCE_PATH,
    RUNNER_SOURCE_PATH,
    _case_table,
    _checkpoint_preflight,
)
from .sequential_personalization import SearchAlpha


def _prediction_map() -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def test_frameworks_and_scale_uncertainty_are_frozen() -> None:
    calibration = load_semantics_calibration()
    assert [(item.framework_id, item.horizon_steps) for item in FRAMEWORKS] == [
        ("SINGLE_STEP", 1),
        ("BUNDLE_2", 2),
        ("BUNDLE_3", 3),
        ("BUNDLE_5", 5),
    ]
    assert [framework_uncertainty(item, calibration) for item in FRAMEWORKS] == pytest.approx(
        [0.00196762280892, 0.00164217053717, 0.00244544326845, 0.00400496043747]
    )
    assert framework_uncertainty(FRAMEWORKS[1], calibration) < framework_uncertainty(
        FRAMEWORKS[0], calibration
    )
    assert framework_uncertainty(FRAMEWORKS[1], calibration) < framework_uncertainty(
        FRAMEWORKS[2], calibration
    ) < framework_uncertainty(FRAMEWORKS[3], calibration)
    assert DEFAULT_ENABLED is False


def test_manifest_freezes_endpoint_only_rules_before_truth(tmp_path: Path) -> None:
    calibration = load_semantics_calibration()
    payload = manifest_payload(
        calibration,
        checkpoint_commit="checkpoint",
        protected_source_sha256={"P2_V1": "abc"},
    )
    rule = payload["decision_rule_common_to_all_frameworks"]
    endpoint = payload["endpoint_authorization"]
    assert rule["magnitude_gate"] == "predicted_endpoint_improvement > 0.005"
    assert rule["direction_gate"] == "deltaJ_pred + U_scale_P95 < 0"
    assert rule["additive_margin_used"] is False
    assert endpoint["authorization_scope"] == "DIRECT_ENDPOINT_CANDIDATE_ONLY"
    assert endpoint["intermediate_trajectories_executed"] is False
    assert endpoint["model_refit_after_endpoint_execution"]
    assert endpoint["whole_map_recomputed_after_endpoint_execution"]
    assert payload["truth_used_to_define_or_rank_frameworks"] is False
    path = tmp_path / "MANIFEST.json"
    path.write_bytes(canonical_json_bytes(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    gate = FrozenFrameworkManifestGate(path, digest)
    gate.record_truth_access("AFTER_FREEZE")
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        gate.record_truth_access("AFTER_MUTATION")


def test_bundle_5_selects_direct_endpoint_and_never_intermediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lower_limb_sim.p2_multi_step_decision_framework_analysis._patient_valid",
        lambda _point, _cache: True,
    )
    calibration = load_semantics_calibration()
    candidates = evaluate_endpoint_candidates(
        _prediction_map(),
        current=SearchAlpha(),
        spec=FRAMEWORKS[3],
        calibration=calibration,
        executed_keys={(0.0, 0.0, 0.0)},
        patient_validity_cache={},
    )
    selected = select_endpoint_candidate(candidates)
    assert selected is not None
    assert selected["trajectory_id"] == "hip_5"
    assert selected["horizon_steps"] == 5
    assert selected["latent_intermediate_count"] == 4
    assert selected["latent_intermediate_trajectory_ids"] == "hip_1;hip_2;hip_3;hip_4"
    assert selected["intermediate_execution_count"] == 0
    assert selected["endpoint_execution_count_if_selected"] == 1
    assert selected["authorization_scope"] == "DIRECT_ENDPOINT_CANDIDATE_ONLY"
    assert not selected["truth_used_for_authorization"]


def test_single_step_same_map_fails_unchanged_magnitude_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lower_limb_sim.p2_multi_step_decision_framework_analysis._patient_valid",
        lambda _point, _cache: True,
    )
    candidates = evaluate_endpoint_candidates(
        _prediction_map(),
        current=SearchAlpha(),
        spec=FRAMEWORKS[0],
        calibration=load_semantics_calibration(),
        executed_keys={(0.0, 0.0, 0.0)},
        patient_validity_cache={},
    )
    candidate = candidates.loc[candidates["trajectory_id"].eq("hip_1")].iloc[0]
    assert not candidate["magnitude_gate_pass"]
    assert candidate["direction_gate_pass"] is False or not bool(
        candidate["direction_gate_pass"]
    )
    assert not candidate["research_exploit_eligible"]
    assert select_endpoint_candidate(candidates) is None


def test_checkpoint_invariants_and_no_robot_imports() -> None:
    checkpoint = _checkpoint_preflight()
    cases = _case_table()
    assert checkpoint["tracked_calibration_inputs_verified"]
    assert checkpoint["prior_semantics_manifest_sha_verified"]
    assert sha256_file(PRIOR_SEMANTICS_MANIFEST_PATH) == PRIOR_SEMANTICS_MANIFEST_SHA256
    assert len(cases) == 15
    assert cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum() == 9
    assert cases["development_origin"].eq("POST_REJECTION_DEVELOPMENT").sum() == 6
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    source = CORE_SOURCE_PATH.read_text(encoding="utf-8") + RUNNER_SOURCE_PATH.read_text(
        encoding="utf-8"
    )
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_framework_artifacts_are_consistent_and_offline_only() -> None:
    root = (
        Path(__file__).parent
        / "formal_artifacts"
        / "p2_multi_step_decision_framework_analysis_v1"
    )
    required = {
        "MANIFEST.json",
        "DECISION_FRAMEWORK_REPORT.md",
        "single_vs_bundle_comparison.csv",
        "small_step_recovery.csv",
        "subject_specificity_analysis.csv",
        "trial_efficiency_analysis.csv",
        "metadata.json",
    }
    assert required.issubset({path.name for path in root.iterdir()})
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert sha256_file(root / "MANIFEST.json") == metadata["manifest_sha256"]
    assert metadata["selected_framework_for_further_research"] == "BUNDLE_5"
    assert metadata["prototype_scope_if_recommended"] == "DEFAULT_OFF_OFFLINE_ONLY"
    assert metadata["total_intermediate_trajectory_executions"] == 0
    assert metadata["future_truth_used_for_authorization"] is False
    assert metadata["held_out_final_test_read"] is False
    assert metadata["prospective_cohort_run"] is False
    assert metadata["P2_V1_modified"] is False
    assert metadata["new_policy_implemented"] is False
    assert metadata["robot_connected"] is False
    assert metadata["protected_source_sha256_before"] == metadata[
        "protected_source_sha256_after"
    ]
    for name, record in metadata["artifact_manifest"].items():
        path = root / name
        assert sha256_file(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]

    comparison = pd.read_csv(root / "single_vs_bundle_comparison.csv").set_index(
        "framework_id"
    )
    assert int(comparison.loc["SINGLE_STEP", "small_step_recovery"]) == 0
    assert int(comparison.loc["BUNDLE_2", "small_step_recovery"]) == 0
    assert int(comparison.loc["BUNDLE_3", "small_step_recovery"]) == 0
    assert int(comparison.loc["BUNDLE_5", "small_step_recovery"]) == 9
    assert int(comparison.loc["BUNDLE_5", "false_improvement"]) == 0
    assert int(comparison["intermediate_trajectory_executions"].sum()) == 0

    specificity = pd.read_csv(root / "subject_specificity_analysis.csv")
    bundle_5 = specificity.loc[specificity["framework_id"].eq("BUNDLE_5")]
    assert int(bundle_5["unique_final_alpha_count_within_framework"].max()) == 1
    assert bundle_5["final_alpha_hip"].eq(0.0).all()
    assert bundle_5["final_alpha_knee"].eq(-5.0).all()
    assert bundle_5["final_alpha_phase"].eq(0.0).all()
