from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
from .p2_adaptive_horizon_decision_prototype import (
    ADAPTIVE_HORIZON_SEQUENCE,
    DEFAULT_ENABLED,
    H1_ID,
    H2_ID,
    H3_ID,
    PRIOR_FRAMEWORK_MANIFEST_PATH,
    PRIOR_FRAMEWORK_MANIFEST_SHA256,
    adaptive_small_step_recovery,
    evaluate_adaptive_endpoint_candidates,
    manifest_payload,
)
from .p2_multi_step_decision_framework_analysis import (
    canonical_json_bytes,
    load_semantics_calibration,
)
from .run_p2_adaptive_horizon_decision_prototype import (
    CORE_SOURCE_PATH,
    DEFAULT_OUTPUT_DIRECTORY,
    REQUIRED_OUTPUT_FILENAMES,
    RUNNER_SOURCE_PATH,
    _checkpoint_preflight,
)
from .sequential_personalization import SearchAlpha


def _prediction_map(step_costs: list[float]) -> pd.DataFrame:
    rows = []
    for step, cost in enumerate(step_costs):
        rows.append(
            {
                "trajectory_id": f"hip_positive_{step}",
                "hip_delta": 0.25 * step,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": cost,
                "geometrically_admissible": True,
                "model_supported": True,
                "domain_coverage": 100.0,
                "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
            }
        )
    return pd.DataFrame(rows)


def _evaluate(
    monkeypatch: pytest.MonkeyPatch, step_costs: list[float]
):
    monkeypatch.setattr(
        "lower_limb_sim.p2_multi_step_decision_framework_analysis._patient_valid",
        lambda _point, _cache: True,
    )
    return evaluate_adaptive_endpoint_candidates(
        _prediction_map(step_costs),
        SearchAlpha(),
        load_semantics_calibration(),
        executed_keys={(0.0, 0.0, 0.0)},
        patient_validity_cache={},
    )


def test_adaptive_manifest_freezes_default_off_endpoint_only_semantics() -> None:
    calibration = load_semantics_calibration()
    payload = manifest_payload(
        calibration,
        checkpoint_commit="checkpoint",
        protected_source_sha256={"P2_V1": "abc"},
    )
    adaptive = payload["adaptive_rule"]
    execution = payload["endpoint_execution"]
    assert tuple(adaptive["horizon_evaluation_order"]) == (1, 2, 3, 5)
    assert adaptive["select_first_horizon_with_any_eligible_endpoint"] is True
    assert adaptive["truth_used_for_horizon_selection"] is False
    assert execution["intermediate_trajectories_executed"] is False
    assert execution["model_refit_after_endpoint_execution"] is True
    assert payload["objective_equivalence_tolerance"] == 0.005
    assert payload["model_support_gate_percent"] == 90.0
    assert payload["default_enabled"] is False
    assert canonical_json_bytes(payload).endswith(b"\n")


def test_adaptive_accepts_h1_without_evaluating_longer_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate(monkeypatch, [1.0, 0.992, 0.984, 0.976, 0.968, 0.960])
    assert result.selected is not None
    assert int(result.selected["adaptive_horizon_steps"]) == 1
    assert result.evaluated_horizons == (1,)
    assert bool(result.selected["direction_consistency_pass"])


def test_adaptive_escalates_to_h5_and_executes_only_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate(
        monkeypatch,
        [1.0, 0.9988, 0.9976, 0.9964, 0.9952, 0.9940],
    )
    assert result.selected is not None
    assert int(result.selected["adaptive_horizon_steps"]) == 5
    assert result.evaluated_horizons == (1, 2, 3, 5)
    assert int(result.selected["latent_intermediate_count"]) == 4
    assert int(result.selected["intermediate_execution_count"]) == 0
    assert result.selected["authorization_scope"] == "DIRECT_ENDPOINT_CANDIDATE_ONLY"
    assert not bool(result.selected["truth_used_for_authorization"])


def test_adaptive_refuses_escalation_after_direction_inconsistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate(
        monkeypatch,
        [1.0, 0.999, 0.9995, 0.997, 0.994, 0.991],
    )
    assert result.selected is None
    assert result.evaluated_horizons == (1, 2)
    assert (
        result.escalation_stopped_reason
        == "NO_CONSISTENT_NONDECISION_VALID_DIRECTION"
    )


def test_small_step_adaptive_decision_is_prediction_only() -> None:
    table = adaptive_small_step_recovery(load_semantics_calibration())
    assert len(table) == 9
    assert table["truth_used_for_authorization"].eq(False).all()
    assert table["truth_attached_posthoc_only"].eq(True).all()
    assert set(table["selected_horizon_steps"].dropna().astype(int)).issubset(
        ADAPTIVE_HORIZON_SEQUENCE
    )


def test_checkpoint_constraints_and_no_robot_imports() -> None:
    checkpoint = _checkpoint_preflight()
    assert checkpoint["prior_framework_artifacts_tracked_and_verified"] is True
    assert hashlib.sha256(PRIOR_FRAMEWORK_MANIFEST_PATH.read_bytes()).hexdigest() == (
        PRIOR_FRAMEWORK_MANIFEST_SHA256
    )
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert tuple(FORMAL_HIP_ROM_DEG) == (0.0, 120.0)
    assert tuple(FORMAL_KNEE_ROM_DEG) == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert DEFAULT_ENABLED is False
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_adaptive_artifacts_are_consistent_and_default_off() -> None:
    root = DEFAULT_OUTPUT_DIRECTORY
    assert set(REQUIRED_OUTPUT_FILENAMES).issubset(
        {path.name for path in root.iterdir()}
    )
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert _sha256(root / "MANIFEST.json") == metadata["manifest_sha256"]
    assert metadata["default_enabled"] is False
    assert metadata["future_truth_used_for_authorization"] is False
    assert metadata["held_out_final_test_read"] is False
    assert metadata["prospective_cohort_run"] is False
    assert metadata["intermediate_trajectories_executed"] is False
    assert metadata["model_refit_after_every_execution"] is True
    assert metadata["full_map_recomputed_after_every_execution"] is True
    assert metadata["P2_V1_modified"] is False
    assert metadata["objective_modified"] is False
    assert metadata["five_parameter_model_modified"] is False
    assert metadata["generator_modified"] is False
    assert metadata["ROM_modified"] is False
    assert metadata["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["theta_shank_definition"] == "q_hip - q_knee"
    assert metadata["protected_source_sha256_before"] == metadata[
        "protected_source_sha256_after"
    ]
    for name, record in metadata["artifact_manifest"].items():
        path = root / name
        assert _sha256(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]

    comparison = pd.read_csv(root / "adaptive_vs_fixed_comparison.csv")
    assert comparison["prototype_variant_id"].tolist() == [H1_ID, H2_ID, H3_ID]
    assert comparison["intermediate_trajectory_executions"].eq(0).all()
    usage = pd.read_csv(root / "horizon_usage.csv")
    assert set(usage["horizon_steps"].astype(int)) == set(
        ADAPTIVE_HORIZON_SEQUENCE
    )
    assert usage["intermediate_trajectory_execution_count"].eq(0).all()
