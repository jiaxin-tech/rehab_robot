from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_bundle5_boundary_subject_specificity_audit import (
    ADAPTIVE_MANIFEST_PATH,
    ADAPTIVE_MANIFEST_SHA256,
    BOUNDARY_MIXED,
    FINAL_IDENTIFIED,
    OBJECTIVE_NO_CHANGE,
    classify_boundary_collapse,
    classify_trial_values,
    manifest_payload,
)
from .run_p2_bundle5_boundary_subject_specificity_audit import (
    CORE_SOURCE_PATH,
    DEFAULT_OUTPUT_DIRECTORY,
    REQUIRED_OUTPUT_FILENAMES,
    RUNNER_SOURCE_PATH,
    _checkpoint_preflight,
)


def test_adaptive_prototype_is_an_independent_checkpoint() -> None:
    checkpoint = _checkpoint_preflight()
    assert checkpoint["adaptive_checkpoint_is_current_HEAD"] is True
    assert checkpoint["adaptive_checkpoint_commit"] == checkpoint[
        "checkpoint_commit"
    ]
    assert hashlib.sha256(ADAPTIVE_MANIFEST_PATH.read_bytes()).hexdigest() == (
        ADAPTIVE_MANIFEST_SHA256
    )


def test_manifest_is_posthoc_only_and_freezes_no_new_policy() -> None:
    payload = manifest_payload(
        checkpoint_commit="checkpoint",
        protected_source_sha256={"P2_V1": "abc"},
    )
    assert payload["policy_under_audit"] == "H2_FIXED_BUNDLE_5"
    assert payload["new_policy_designed"] is False
    assert payload["replay_contract"]["trajectory_selection_recomputed_or_changed"] is False
    assert payload["replay_contract"]["truth_used_for_policy_authorization"] is False
    assert payload["data_roles"]["independent_calibration"] == (
        "EXISTING_UNCERTAINTY_ONLY_NOT_OUTCOME_EVIDENCE"
    )
    assert payload["data_roles"]["prospective"] == "NOT_GENERATED"
    assert payload["data_roles"]["heldout_final_test"] == "NOT_READ"
    assert payload["objective_equivalence_tolerance"] == 0.005
    assert payload["model_support_gate_percent"] == 90.0
    assert payload["default_enabled"] is False


def test_coordinate_truth_concentration_plus_full_diversity_is_mixed() -> None:
    optimum = pd.DataFrame(
        {
            "truth_global_alpha_hip": [2.0, -5.0],
            "truth_global_alpha_knee": [-5.0, -5.0],
            "truth_global_alpha_phase": [0.03, -0.025],
            "H2_final_alpha_hip": [0.0, 0.0],
            "H2_final_alpha_knee": [-5.0, -5.0],
            "H2_final_alpha_phase": [0.0, 0.0],
        }
    )
    assert classify_boundary_collapse(optimum) == BOUNDARY_MIXED


def test_post_optimum_support_growth_is_not_mislabeled_decision_value() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "case_id": "case",
                "iteration": 1,
                "trial_purpose": "EXPLOIT",
                "actual_best_J_improvement": 0.01,
                "changed_best_alpha": True,
                "changed_future_exploit_eligibility": True,
                "parameter_changed_exactly": True,
                "prediction_map_changed_exactly": True,
                "support_point_increase": 0,
                "best_alpha_knee_before": 0.0,
                "best_alpha_hip_after": 0.0,
                "best_alpha_knee_after": -5.0,
                "best_alpha_phase_after": 0.0,
            },
            {
                "case_id": "case",
                "iteration": 2,
                "trial_purpose": "EXPLORE",
                "actual_best_J_improvement": 0.0,
                "changed_best_alpha": False,
                "changed_future_exploit_eligibility": False,
                "parameter_changed_exactly": True,
                "prediction_map_changed_exactly": True,
                "support_point_increase": 550,
                "best_alpha_knee_before": -5.0,
                "best_alpha_hip_after": 0.0,
                "best_alpha_knee_after": -5.0,
                "best_alpha_phase_after": 0.0,
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "case_id": "case",
                "final_best_alpha_hip": 0.0,
                "final_best_alpha_knee": -5.0,
                "final_best_alpha_phase": 0.0,
            }
        ]
    )
    result = classify_trial_values(diagnostics, summary)
    assert result.iloc[0]["trial_value_classification"] == "MULTIPLE_VALUES"
    assert result.iloc[1]["trial_value_classification"] == "POST_OPTIMUM_LOW_VALUE"


def test_frozen_scientific_configuration_and_no_robot_imports() -> None:
    assert ACTIVE_REFERENCE_SHA256 == (
        "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    )
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert tuple(FORMAL_HIP_ROM_DEG) == (0.0, 120.0)
    assert tuple(FORMAL_KNEE_ROM_DEG) == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert OFFLINE_PERSONALIZATION_SEARCH_BOUNDS == {
        "hip_amplitude_delta_deg": (-5.0, 2.0),
        "knee_amplitude_delta_deg": (-5.0, 2.0),
        "knee_phase_shift": (-0.03, 0.03),
    }
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


def test_formal_bundle5_audit_artifacts_are_consistent() -> None:
    root = DEFAULT_OUTPUT_DIRECTORY
    assert set(REQUIRED_OUTPUT_FILENAMES).issubset(
        {path.name for path in root.iterdir()}
    )
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert _sha256(root / "BUNDLE5_AUDIT_MANIFEST_V1.json") == metadata[
        "manifest_sha256"
    ]
    assert metadata["final_status"] == FINAL_IDENTIFIED
    assert metadata["boundary_collapse_classification"] == BOUNDARY_MIXED
    assert metadata["objective_status"] == OBJECTIVE_NO_CHANGE
    assert metadata["truth_full_alpha_unique_count"] == 8
    assert metadata["truth_knee_minus_5_count"] == 15
    assert metadata["truth_exact_H2_common_alpha_count"] == 0
    assert metadata["frozen_H2_trial_count"] == 116
    assert metadata["frozen_H2_rows_verified"] is True
    assert metadata["calibration_cases_used_as_policy_outcomes"] is False
    assert metadata["future_prospective_generated"] is False
    assert metadata["held_out_final_test_read"] is False
    assert metadata["truth_profiles_used_for_policy"] is False
    assert metadata["new_policy_implemented"] is False
    assert metadata["robot_connected"] is False
    assert metadata["protected_source_sha256_before"] == metadata[
        "protected_source_sha256_after"
    ]
    for name, record in metadata["artifact_manifest"].items():
        path = root / name
        assert _sha256(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]

    trial = pd.read_csv(root / "bundle5_trial_value_audit.csv")
    assert len(trial) == 116
    assert trial["truth_used_to_select_or_stop"].eq(False).all()
    axis = pd.read_csv(root / "bundle5_axis_direction_decision_audit.csv")
    assert axis.loc[axis["selected"], "axis_direction"].eq("KNEE_NEGATIVE").all()
    assert axis["truth_used_for_authorization"].eq(False).all()

