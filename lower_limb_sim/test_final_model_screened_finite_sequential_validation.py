from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
)
from .final_model_screened_finite_sequential_validation import (
    DEFAULT_ENABLED,
    FINAL_STATUS_RULE,
    MAX_MODEL_SCREENED_CANDIDATES,
    MAX_VALIDATION_TRIALS,
    FrozenShortlistTruthGate,
    assert_complete_candidate_trajectory,
    freeze_model_screened_shortlist,
    method_manifest_payload,
    rerank_remaining_frozen_candidates,
    select_best_validated,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .generate_final_method_animation import (
    FIGURE_NAMES,
    GIF_NAMES,
    generate_workflow_animation,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .run_final_model_screened_finite_sequential_validation import (
    DEFAULT_OUTPUT_DIRECTORY,
    FINAL_LIMITED,
    FINAL_SUPPORTED,
    PREVIOUS_REQUIRED,
    PREVIOUS_MANIFEST_SHA256,
    PROJECT_ROOT,
    _checkpoint_preflight,
    _git_output,
)


def _prediction_map() -> pd.DataFrame:
    rows = []
    specifications = (
        (0.0, 0.0, 0.0, 1.0),
        (-1.0, -5.0, 0.0300, 0.9000),
        (-1.0, -5.0, 0.0275, 0.9001),
        (-2.0, -4.0, 0.0000, 0.9051),
        (-3.0, -3.0, -0.0100, 0.9102),
        (-4.0, -2.0, 0.0100, 0.9153),
    )
    for hip, knee, phase, predicted in specifications:
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=hip,
            knee_amplitude_delta_deg=knee,
            knee_phase_shift=phase,
        )
        rows.append(
            {
                "trajectory_id": generated.metadata["trajectory_id"],
                "hip_delta": hip,
                "knee_delta": knee,
                "phase_delta": phase,
                "J_pred": predicted,
                "domain_coverage": 95.0,
                "model_supported": True,
                "geometrically_admissible": True,
            }
        )
    return pd.DataFrame(rows)


def test_previous_scientific_stage_is_an_independent_checkpoint() -> None:
    # The V1 generator itself remains frozen and intentionally required its
    # immediate prerequisite to be current HEAD when V1 was created.  Later
    # formal checkpoint commits must not make this historical regression test
    # fail merely because HEAD advanced.  Verify ancestry and the immutable
    # manifest here without weakening the frozen generator preflight.
    head = _git_output("rev-parse", "HEAD")
    previous_commit = _git_output(
        "log",
        "-1",
        "--format=%H",
        "--",
        str(PREVIOUS_REQUIRED[0].relative_to(PROJECT_ROOT)),
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", previous_commit, head],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=20.0,
    )
    assert PREVIOUS_MANIFEST_SHA256 == (
        "b959444e8df39a05693f873aaa3060cb5c21a4525d7f1bbda9c81aa96f1762c8"
    )


def test_shortlist_freezes_before_truth_and_reuses_only_formal_equivalence() -> None:
    first = freeze_model_screened_shortlist(_prediction_map(), case_id="synthetic")
    second = freeze_model_screened_shortlist(_prediction_map(), case_id="synthetic")
    assert first == second
    assert len(first.candidates) == MAX_MODEL_SCREENED_CANDIDATES == 3
    assert first.truth_read_before_freeze is False
    assert [candidate.predicted_equivalence_band for candidate in first.candidates] == [0, 1, 2]
    assert all(candidate.model_supported for candidate in first.candidates)
    assert all(candidate.geometrically_admissible for candidate in first.candidates)
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    with pytest.raises(PermissionError, match="before truth"):
        freeze_model_screened_shortlist(
            _prediction_map().assign(J_truth=0.5), case_id="leak"
        )


def test_frozen_gate_rejects_fourth_candidate_and_repeat_execution() -> None:
    shortlist = freeze_model_screened_shortlist(_prediction_map(), case_id="synthetic")
    with pytest.raises(PermissionError, match="persisted"):
        FrozenShortlistTruthGate(
            shortlist, global_manifest_sha256="a" * 64, manifest_persisted=False
        )
    gate = FrozenShortlistTruthGate(
        shortlist, global_manifest_sha256="a" * 64, manifest_persisted=True
    )
    with pytest.raises(PermissionError, match="cannot enter"):
        gate.authorize("C4_NOT_FROZEN")
    selected = shortlist.trajectory_ids[0]
    token = gate.authorize(selected)
    gate.complete(selected, token)
    with pytest.raises(PermissionError, match="only once"):
        gate.authorize(selected)


def test_rerank_is_restricted_to_remaining_frozen_candidates() -> None:
    prediction = _prediction_map()
    shortlist = freeze_model_screened_shortlist(prediction, case_id="synthetic")
    outsider = prediction.iloc[[1]].copy()
    outsider["trajectory_id"] = "NEW_GLOBAL_OPTIMUM_AFTER_REFIT"
    outsider["hip_delta"] = 1.75
    outsider["J_pred"] = 0.1
    updated = pd.concat((prediction, outsider), ignore_index=True)
    ranking = rerank_remaining_frozen_candidates(
        shortlist,
        updated,
        executed_trajectory_ids=[shortlist.trajectory_ids[0]],
    )
    assert set(ranking["trajectory_id"]) == set(shortlist.trajectory_ids[1:])
    assert "NEW_GLOBAL_OPTIMUM_AFTER_REFIT" not in set(ranking["trajectory_id"])
    assert not ranking["candidate_addition_allowed"].astype(bool).any()


def test_candidate_is_a_complete_trajectory_and_theta_is_subtraction() -> None:
    generated = generate_personalized_trajectory(
        hip_amplitude_delta_deg=-1.0,
        knee_amplitude_delta_deg=-2.0,
        knee_phase_shift=0.01,
    )
    trajectory = generated.trajectory.copy(deep=True)
    identifier = str(generated.metadata["trajectory_id"])
    trajectory["trajectory_id"] = identifier
    assert_complete_candidate_trajectory(
        trajectory, expected_trajectory_id=identifier
    )
    assert len(trajectory) == 401
    assert np.allclose(
        trajectory["theta_shank_rad"],
        trajectory["q_hip_rad"] - trajectory["q_knee_rad"],
        atol=1e-12,
        rtol=0.0,
    )


def test_final_selection_uses_validated_J_not_prediction() -> None:
    validated = pd.DataFrame(
        {
            "trajectory_id": ["reference", "C1", "C2"],
            "validated_J": [1.0, 0.95, 0.91],
            "validation_role": ["REFERENCE", "CANDIDATE", "CANDIDATE"],
        }
    )
    assert select_best_validated(validated)["trajectory_id"] == "C2"
    with pytest.raises(PermissionError, match="must not use prediction"):
        select_best_validated(validated.assign(J_pred=[1.0, 0.8, 0.99]))


def test_manifest_is_default_off_finite_and_truth_isolated() -> None:
    shortlist = freeze_model_screened_shortlist(_prediction_map(), case_id="synthetic")
    manifest = method_manifest_payload(
        checkpoint={"checkpoint_commit": "a" * 40},
        source_hashes={"protected": "b" * 64},
        shortlists=[shortlist],
    )
    assert DEFAULT_ENABLED is False
    assert manifest["max_model_screened_candidates"] == 3
    assert manifest["max_validation_trials"] == MAX_VALIDATION_TRIALS == 3
    assert manifest["shortlist_frozen_before_candidate_truth"] is True
    assert manifest["new_candidate_after_freeze_allowed"] is False
    assert manifest["P2_explore_exploit_invoked"] is False
    assert manifest["bundle_invoked"] is False
    assert manifest["adaptive_horizon_invoked"] is False
    assert manifest["calibration_cases_used_for_rule_tuning"] is False
    assert manifest["held_out_final_test_read"] is False
    assert manifest["new_prospective_cohort_generated"] is False
    assert manifest["final_status_rule_frozen_before_candidate_truth"] == FINAL_STATUS_RULE


def test_frozen_scientific_constants_remain_exact() -> None:
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert MODEL_SUPPORT_COVERAGE_GATE_PERCENT == 90.0
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005


def test_method_sources_do_not_import_motion_or_old_policy_boundaries() -> None:
    directory = Path(__file__).resolve().parent
    sources = "\n".join(
        (directory / name).read_text(encoding="utf-8")
        for name in (
            "final_model_screened_finite_sequential_validation.py",
            "run_final_model_screened_finite_sequential_validation.py",
            "generate_final_method_animation.py",
        )
    )
    for forbidden in (
        "import hardware",
        "from hardware",
        "import control",
        "from control",
        "import collection",
        "from collection",
        "import safety",
        "from safety",
        "run_framework_shadow",
        "run_adaptive_shadow",
        "POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT",
        "connectToRobot",
    ):
        assert forbidden not in sources


def test_formal_artifacts_are_complete_and_scientifically_isolated() -> None:
    output = DEFAULT_OUTPUT_DIRECTORY
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    manifest_path = output / "FINAL_METHOD_MANIFEST_V1.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == metadata["manifest_sha256"]
    assert metadata["final_status"] in (FINAL_SUPPORTED, FINAL_LIMITED)
    assert metadata["STOP_FURTHER_P2_EXPANSION"] is True
    assert metadata["frozen_shortlist_count"] <= 15 * 3
    assert metadata["maximum_candidate_validations_per_case"] <= 3
    assert metadata["candidate_shortlists_frozen_before_truth"] is True
    assert metadata["manifest_persisted_before_candidate_truth"] is True
    assert metadata["new_candidate_after_freeze_added"] is False
    assert metadata["model_refit_after_every_candidate"] is True
    assert metadata["full_landscape_recomputed_after_every_candidate"] is True
    assert metadata["remaining_frozen_candidates_only_reranked"] is True
    assert metadata["P2_explore_exploit_run"] is False
    assert metadata["bundle_run"] is False
    assert metadata["adaptive_horizon_run"] is False
    assert metadata["held_out_final_test_read"] is False
    assert metadata["new_prospective_cohort_generated"] is False
    assert metadata["robot_connected"] is False
    assert metadata["hardware_control_collection_safety_modified"] is False
    assert metadata["protected_source_sha256_before"] == metadata["protected_source_sha256_after"]
    execution = pd.read_csv(output / "candidate_execution_history.csv")
    shortlist = pd.read_csv(output / "candidate_shortlist_manifest.csv")
    updates = pd.read_csv(output / "model_update_history.csv")
    assert execution.groupby("case_id").size().max() <= 3
    assert execution["whole_trajectory_execution"].astype(bool).all()
    assert execution["trajectory_sample_count"].eq(401).all()
    assert execution["theta_refit_after_execution"].astype(bool).all()
    assert execution["full_landscape_recomputed_after_execution"].astype(bool).all()
    assert not execution["diagnostic_optimum_added_to_shortlist"].astype(bool).any()
    assert not shortlist["truth_read_before_freeze"].astype(bool).any()
    assert set(execution["trajectory_id"]).issubset(set(shortlist["trajectory_id"]))
    assert updates.groupby("case_id").size().min() >= 2


def test_formal_gifs_are_multiframe_hd_and_deterministic(tmp_path: Path) -> None:
    output = DEFAULT_OUTPUT_DIRECTORY
    for name in GIF_NAMES:
        with Image.open(output / name) as image:
            assert image.size[0] >= 1280
            assert image.size[1] >= 720
            assert image.n_frames > 1
    first = tmp_path / "first.gif"
    second = tmp_path / "second.gif"
    generate_workflow_animation(output, output_path=first)
    generate_workflow_animation(output, output_path=second)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()
    for name in FIGURE_NAMES:
        assert (output / name).is_file()
