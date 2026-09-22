from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from external_simulation.myoleg_v3_trajectory_parameterization_design_v1.parameterization import (
    PARAMETERIZATION_ID,
    branch_warp,
)
from personalization.environment import RealRobotEnvironment
from personalization.ledger import ExecutedCandidateLedger
from personalization.rom_gated_v2 import (
    ROMBoundaryObservation,
    ROMCalibrationLedger,
    ROMDeterminationController,
    ROMGateStatus,
    ROMGatedPersonalizationEpisode,
    RealROMDeterminationInterface,
    SubjectROMProfile,
    SubjectSpecificFullDynamicsGrayBoxEndpointAdapter,
    SubjectSpecificOfflineMechanicalEnvironment,
    SubjectSpecificReferenceAdapter,
    SubjectSpecificV3CandidateDomain,
    SyntheticThresholdGate,
    UnavailableValidatedSafetyGate,
    assert_same_frozen_rom_comparison,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
    make_subject_physics_model,
)
from personalization.rom_gated_v2.rom import (
    CURRENT_TASK_SPECIFIC_ALLOWED_ROM,
    FROZEN_SUBJECT_ROM_PROFILE_V1,
    NOT_A_HUMAN_SAFETY_MODEL,
    P_LIMIT,
    VALIDATED_FORCE_THRESHOLD,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small_case():
    return make_offline_rom_development_cases()[0]


@pytest.fixture(scope="module")
def small_stage(small_case):
    controller, profile = determine_synthetic_rom(small_case)
    domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)
    return controller, profile, domain


def test_subject_rom_profile_schema_is_versioned_extensible_and_immutable(small_stage):
    _, profile, _ = small_stage
    assert profile.version == 1
    assert profile.rom_status == FROZEN_SUBJECT_ROM_PROFILE_V1
    assert profile.semantic_scope == CURRENT_TASK_SPECIFIC_ALLOWED_ROM
    assert profile.frozen is True
    assert profile.coupled_joint_constraint is None
    assert profile.configuration_validity_mask is None
    assert profile.trajectory_specific_constraint is None
    with pytest.raises(FrozenInstanceError):
        profile.hip_max_rad = 0.0
    with pytest.raises(TypeError):
        profile.metadata["new"] = "value"


def test_last_safe_boundary_stop_and_no_post_stop_expansion(small_case):
    controller, profile = determine_synthetic_rom(small_case)
    statuses = [entry.gate_result.status for entry in controller.ledger.entries]
    assert statuses == [
        ROMGateStatus.SAFE,
        ROMGateStatus.SAFE,
        ROMGateStatus.SAFE,
        ROMGateStatus.STOP,
    ]
    assert controller.last_safe_configuration.configuration_id.endswith("Q3")
    assert controller.first_stop_configuration.configuration_id.endswith("Q4")
    assert profile.hip_max_rad == pytest.approx(
        controller.last_safe_configuration.hip_angle_rad
    )
    assert profile.knee_max_rad == pytest.approx(
        controller.last_safe_configuration.knee_angle_rad
    )
    assert profile.hip_max_rad != controller.first_stop_configuration.hip_angle_rad
    with pytest.raises(RuntimeError, match="further expansion is prohibited"):
        controller.observe(controller.first_stop_configuration)


def test_invalid_rom_observation_fails_closed_and_cannot_freeze():
    controller = ROMDeterminationController(
        SyntheticThresholdGate(
            stop_threshold=1.0,
            threshold_policy_id="SYNTHETIC_INVALID_TEST_V1",
        )
    )
    result = controller.observe(
        ROMBoundaryObservation(
            observation_id="invalid-1",
            calibration_episode_id="invalid-calibration",
            configuration_id="missing-measurement",
            hip_angle_rad=None,
            knee_angle_rad=None,
            valid=False,
            measurement_quality="MISSING",
        )
    )
    assert result.status is ROMGateStatus.STOP
    assert result.reason == "INVALID_MEASUREMENT_FAIL_CLOSED"
    assert not controller.can_propose
    with pytest.raises(RuntimeError, match="without a defensibly SAFE observation"):
        controller.freeze_profile(
            profile_id="invalid",
            version=1,
            provenance="ALGORITHM_DEVELOPMENT_ONLY",
        )


def test_no_real_threshold_or_margin_is_fabricated(small_stage):
    _, profile, _ = small_stage
    assert P_LIMIT == "NOT_AVAILABLE"
    assert VALIDATED_FORCE_THRESHOLD == "NOT_AVAILABLE"
    assert profile.safety_margin_policy is None
    assert all(policy.startswith("SYNTHETIC_") for policy in profile.threshold_policy_ids)
    result = UnavailableValidatedSafetyGate().evaluate(
        ROMBoundaryObservation(
            observation_id="real-unavailable",
            calibration_episode_id="real-unavailable",
            configuration_id="q",
            hip_angle_rad=0.5,
            knee_angle_rad=0.5,
            valid=True,
        )
    )
    assert result.status is ROMGateStatus.STOP
    assert "NOT_AVAILABLE" in result.reason


def test_external_stop_has_fail_closed_semantics_not_comfort_label():
    gate = SyntheticThresholdGate(
        stop_threshold=10.0,
        threshold_policy_id="SYNTHETIC_EXTERNAL_STOP_V1",
    )
    result = gate.evaluate(
        ROMBoundaryObservation(
            observation_id="external-stop",
            calibration_episode_id="external-stop",
            configuration_id="q",
            hip_angle_rad=0.5,
            knee_angle_rad=0.5,
            valid=True,
            stop_requested=True,
            metadata={"synthetic_boundary_proxy": 0.0},
        )
    )
    assert result.status is ROMGateStatus.STOP
    assert result.reason == "EXTERNAL_STOP"
    assert "comfort" not in result.reason.lower()


def test_subject_reference_scaling_preserves_progression_timing_and_smoothness(small_stage):
    _, profile, domain = small_stage
    reference = domain.subject_reference
    assert np.min(reference.q[:, 0]) == pytest.approx(profile.hip_min_rad)
    assert np.max(reference.q[:, 0]) == pytest.approx(profile.hip_max_rad)
    assert np.min(reference.q[:, 1]) == pytest.approx(profile.knee_min_rad)
    assert np.max(reference.q[:, 1]) == pytest.approx(profile.knee_max_rad)
    assert reference.audit["branch_timing_preserved"] is True
    assert reference.audit["duration_preserved"] is True
    assert reference.audit["closure_preserved"] is True
    assert reference.audit["affine_progression_preserves_step_sign"] is True
    assert reference.audit["pointwise_clipping"] is False
    assert reference.audit["normalized_progression_max_abs_error"] < 1.0e-12
    assert reference.duration_s == 24.0


def test_beta_zero_is_array_exact_subject_reference(small_stage):
    _, _, domain = small_stage
    candidate = domain.reference
    reference = domain.subject_reference
    assert candidate.beta == (0.0, 0.0)
    assert candidate.canonical_beta_id == "MYOLEG_V3_K0312"
    assert np.array_equal(candidate.trajectory.q, reference.q)
    assert np.array_equal(candidate.trajectory.dq, reference.dq)
    assert np.array_equal(candidate.trajectory.ddq, reference.ddq)
    with pytest.raises(ValueError, match="read-only"):
        candidate.trajectory.q[0, 0] = 0.0


def test_all_625_candidates_preserve_subject_rom_without_clipping(small_stage):
    _, profile, domain = small_stage
    assert len(domain) == 625
    summary = domain.invariant_summary()
    assert summary["all_kinematic_gates_pass"] is True
    assert summary["maximum_extrema_error_rad"] <= 2.0e-5
    assert summary["maximum_closure_error"] <= 1.0e-9
    assert summary["minimum_warp_derivative"] > 0.0
    assert summary["pointwise_clipping_count"] == 0
    assert summary["duration_s"] == 24.0
    for candidate in domain:
        q = candidate.trajectory.q
        assert np.min(q[:, 0]) == pytest.approx(profile.hip_min_rad, abs=2.0e-5)
        assert np.max(q[:, 0]) == pytest.approx(profile.hip_max_rad, abs=2.0e-5)
        assert np.min(q[:, 1]) == pytest.approx(profile.knee_min_rad, abs=2.0e-5)
        assert np.max(q[:, 1]) == pytest.approx(profile.knee_max_rad, abs=2.0e-5)


def test_v3_operator_and_canonical_beta_grid_are_preserved(small_stage):
    _, _, domain = small_stage
    assert {candidate.v3_operator_version for candidate in domain} == {
        PARAMETERIZATION_ID
    }
    assert min(candidate.beta_flex for candidate in domain) == -0.03
    assert max(candidate.beta_flex for candidate in domain) == 0.03
    assert min(candidate.beta_extend for candidate in domain) == -0.03
    assert max(candidate.beta_extend for candidate in domain) == 0.03
    s = np.asarray([0.2, 0.5, 0.8])
    warped, _, _ = branch_warp(s, 0.02)
    expected = s + 0.02 * 64.0 * s**3 * (1.0 - s) ** 3
    assert np.allclose(warped, expected)


def test_subject_candidate_identity_includes_rom_and_canonical_beta(small_stage):
    _, first_profile, first_domain = small_stage
    second_case = make_offline_rom_development_cases()[1]
    _, second_profile = determine_synthetic_rom(second_case)
    second_domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(second_profile)
    assert len({item.candidate_id for item in first_domain}) == 625
    assert first_domain.reference.canonical_beta_id == second_domain.reference.canonical_beta_id
    assert first_domain.reference.candidate_id != second_domain.reference.candidate_id
    payload = first_domain.reference.as_dict()
    assert payload["rom_profile_id"] == first_profile.profile_id
    assert payload["rom_profile_fingerprint"] == first_profile.fingerprint


def test_stage1_requires_frozen_profile_and_k_excludes_rom_observations(small_stage, small_case):
    controller, profile, domain = small_stage
    unfrozen = SubjectROMProfile(
        profile_id="UNFROZEN",
        version=1,
        hip_min_rad=small_case.hip_min_rad,
        hip_max_rad=small_case.hip_max_rad,
        knee_min_rad=small_case.knee_min_rad,
        knee_max_rad=small_case.knee_max_rad,
        rom_status="DRAFT",
        provenance="ALGORITHM_DEVELOPMENT_ONLY",
        frozen=False,
    )
    with pytest.raises(RuntimeError, match="ROM_PROFILE_FROZEN"):
        SubjectSpecificReferenceAdapter().adapt(unfrozen)
    with pytest.raises(RuntimeError, match="Stage 1 cannot start"):
        ROMGatedPersonalizationEpisode(
            episode_id="bad", profile=unfrozen, domain=domain
        )

    episode = ROMGatedPersonalizationEpisode(
        episode_id="budget-separation", profile=profile, domain=domain
    )
    result = episode.run(
        SubjectSpecificOfflineMechanicalEnvironment(domain, small_case),
        method="Standard BO",
    )
    assert len(controller.ledger.entries) == 4
    assert len(result.sequential_result.ledger.entries) == 4
    assert isinstance(controller.ledger, ROMCalibrationLedger)
    assert isinstance(result.sequential_result.ledger, ExecutedCandidateLedger)
    assert result.as_dict()["ledger_type"] == "PERSONALIZATION_TRIAL_LEDGER"


def test_all_baselines_share_one_frozen_rom_and_stage1_starts_at_reference(small_stage, small_case):
    _, profile, domain = small_stage
    results = {}
    for method in (
        "Reference",
        "Random",
        "Space Filling",
        "Model-Only Greedy",
        "Standard BO",
        "Physics-Informed BO",
    ):
        episode = ROMGatedPersonalizationEpisode(
            episode_id=f"same-rom:{method}", profile=profile, domain=domain
        )
        result = episode.run(
            SubjectSpecificOfflineMechanicalEnvironment(domain, small_case),
            method=method,
            physics_model=(
                make_subject_physics_model(domain)
                if method in {"Model-Only Greedy", "Physics-Informed BO"}
                else None
            ),
        )
        results[method] = result
        assert result.sequential_result.ledger.entries[0].candidate == domain.reference
        assert len(result.sequential_result.ledger.entries) == 4
    assert assert_same_frozen_rom_comparison(results) == profile.fingerprint


def test_gray_box_consumes_subject_specific_q_dq_ddq(small_stage):
    _, profile, domain = small_stage
    adapter = SubjectSpecificFullDynamicsGrayBoxEndpointAdapter(domain)
    prediction = adapter.predict_value(domain.reference)
    metadata = adapter.metadata()
    assert math.isfinite(prediction) and prediction > 0.0
    assert metadata["rom_profile_fingerprint"] == profile.fingerprint
    assert metadata["trajectory_source"] == "SubjectROMProfile + beta -> actual q/dq/ddq"
    assert metadata["last_trajectory_shapes"] == {
        "q": (401, 2),
        "dq": (401, 2),
        "ddq": (401, 2),
    }
    assert metadata["parameter_semantics"] == "effective_gray_box_parameters"
    assert metadata["V1_fit_and_parameter_semantics_unchanged"] is True


def test_rom_change_closes_old_episode_and_requires_new_version(small_stage):
    _, old_profile, old_domain = small_stage
    case = make_offline_rom_development_cases()[1]
    _, new_profile = determine_synthetic_rom(case, version=2)
    new_domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(new_profile)
    old = ROMGatedPersonalizationEpisode(
        episode_id="old", profile=old_profile, domain=old_domain
    )
    restarted = old.restart_for_changed_rom(
        new_episode_id="new", new_profile=new_profile, new_domain=new_domain
    )
    assert old.status == "CLOSED_ROM_PROFILE_CHANGED"
    assert old.closed_reason == "ROM_CHANGE_INVALIDATES_PERSONALIZATION_EPISODE"
    assert restarted.status == "READY"
    assert restarted.profile.version == 2
    assert restarted.profile.fingerprint != old.profile.fingerprint


def test_real_rom_and_robot_interfaces_fail_closed(small_stage):
    _, _, domain = small_stage
    with pytest.raises(RuntimeError, match="REAL_ROM_DETERMINATION_DISABLED"):
        RealROMDeterminationInterface().determine()
    with pytest.raises(RuntimeError, match="REAL_ROBOT_ENVIRONMENT_DISABLED"):
        RealRobotEnvironment().evaluate(domain.reference, 1)


def test_v2_package_has_no_robot_control_collection_or_safety_imports():
    forbidden = {"hardware", "control", "collection", "safety"}
    for path in (ROOT / "personalization" / "rom_gated_v2").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.lstrip(".").split(".")[0]
                if root:
                    imported.add(root)
        assert not (imported & forbidden), (path, imported & forbidden)


def test_synthetic_gate_is_explicitly_not_a_human_safety_model():
    gate = SyntheticThresholdGate(
        stop_threshold=1.0,
        threshold_policy_id="SYNTHETIC_LABEL_AUDIT_V1",
    )
    assert NOT_A_HUMAN_SAFETY_MODEL in gate.classification
    with pytest.raises(ValueError, match="must start with SYNTHETIC"):
        SyntheticThresholdGate(
            stop_threshold=1.0,
            threshold_policy_id="PRESSURE_LIMIT_42",
        )


def test_recorded_offline_smoke_artifacts_are_complete_and_checksummed():
    result_dir = ROOT / "personalization" / "benchmarks" / "results_v2_rom_gated"
    summary_path = result_dir / "ROM_GATED_V2_SMOKE_SUMMARY.json"
    runs_path = result_dir / "ROM_GATED_V2_SMOKE_RUNS.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["case_count"] == 4
    assert summary["run_count"] == 16
    assert summary["all_runs_completed"] is True
    assert summary["all_stage1_runs_started_at_subject_reference"] is True
    assert summary["all_candidate_domains_have_625_members"] is True
    assert summary["all_subject_rom_invariants_pass"] is True
    assert summary["robot_actions"] == summary["human_actions"] == 0
    expected = {}
    for line in (result_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == expected[summary_path.name]
    assert hashlib.sha256(runs_path.read_bytes()).hexdigest() == expected[runs_path.name]
