from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from .continuous_reference_neighborhood import (
    GENERATOR_VERSION,
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    ContinuousParameters,
    generate_personalized_trajectory,
    parameterized_trajectory_id,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    sha256_file,
)
from .kinematics import forward_kinematics
from .reference_release import (
    RELEASE_ACTIVE_REFERENCE_PATH,
    load_frozen_active_reference,
)
from .run_continuous_reference_neighborhood import (
    run_continuous_reference_neighborhood,
)


@pytest.fixture(scope="module")
def parent():
    return load_frozen_active_reference()


@pytest.fixture(scope="module")
def representatives(parent):
    return {
        "neutral": generate_personalized_trajectory(parent),
        "hip": generate_personalized_trajectory(parent, -3.0, 0.0, 0.0),
        "knee": generate_personalized_trajectory(parent, 0.0, -3.0, 0.0),
        "advance": generate_personalized_trajectory(parent, 0.0, 0.0, 0.03),
        "delay": generate_personalized_trajectory(parent, 0.0, 0.0, -0.03),
    }


def test_neutral_strictly_reproduces_frozen_reference(parent, representatives):
    generated = representatives["neutral"]
    numerical = [
        "time_s",
        "segment_phase",
        "global_phase",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "theta_shank_rad",
        "x_knee_m",
        "z_knee_m",
        "x_pull_m",
        "z_pull_m",
    ]
    np.testing.assert_array_equal(
        generated.trajectory[numerical].to_numpy(),
        parent.trajectory[numerical].to_numpy(),
    )
    assert generated.metadata["neutral_reference_max_abs_state_error"] == 0.0
    assert generated.metadata["parent_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert generated.metadata["neutral_generator_sha"] == generated.metadata[
        "trajectory_sha256"
    ]


def test_joint_amplitudes_are_decoupled(parent, representatives):
    hip = representatives["hip"].trajectory
    knee = representatives["knee"].trajectory
    np.testing.assert_allclose(
        hip["q_knee_rad"], parent.trajectory["q_knee_rad"], atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        knee["q_hip_rad"], parent.trajectory["q_hip_rad"], atol=1e-14, rtol=0.0
    )
    assert representatives["hip"].metadata["hip_max_deviation_deg"] == pytest.approx(3.0)
    assert representatives["knee"].metadata["knee_max_deviation_deg"] == pytest.approx(3.0)


def test_positive_phase_advances_knee_without_moving_hip(parent, representatives):
    advance = representatives["advance"].trajectory
    delay = representatives["delay"].trajectory
    np.testing.assert_allclose(
        advance["q_hip_rad"], parent.trajectory["q_hip_rad"], atol=1e-14, rtol=0.0
    )
    flex = parent.trajectory["cycle_phase"].eq("flexion").to_numpy()
    interior = flex & parent.trajectory["segment_phase"].between(0.2, 0.8).to_numpy()
    assert float(np.mean(advance.loc[interior, "q_knee_rad"])) > float(
        np.mean(delay.loc[interior, "q_knee_rad"])
    )


@pytest.mark.parametrize("name", ["hip", "knee", "advance", "delay"])
def test_endpoints_fk_c2_and_asymmetry_are_preserved(parent, representatives, name):
    result = representatives[name]
    trajectory = result.trajectory
    np.testing.assert_allclose(
        trajectory[["q_hip_rad", "q_knee_rad"]].iloc[[0, -1]],
        parent.trajectory[["q_hip_rad", "q_knee_rad"]].iloc[[0, -1]],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        trajectory[["x_pull_m", "z_pull_m"]].iloc[0],
        trajectory[["x_pull_m", "z_pull_m"]].iloc[-1],
        atol=1e-12,
        rtol=0.0,
    )
    assert result.continuity_audit["position_continuity_warning_count"] == 0
    assert result.continuity_audit["velocity_continuity_warning_count"] == 0
    assert result.continuity_audit["acceleration_continuity_warning_count"] == 0
    assert result.asymmetry_audit["asymmetry_valid"] is True
    assert result.asymmetry_audit["measured_extension_is_reversed_flexion"] is False


def test_formal_rom_theta_and_recomputed_fk(representatives):
    for result in representatives.values():
        trajectory = result.trajectory
        q_hip = trajectory["q_hip_rad"].to_numpy(float)
        q_knee = trajectory["q_knee_rad"].to_numpy(float)
        np.testing.assert_allclose(
            trajectory["theta_shank_rad"], q_hip - q_knee, atol=1e-14, rtol=0.0
        )
        x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, 0.42, 0.30)
        np.testing.assert_allclose(trajectory["x_knee_m"], x_knee, atol=1e-14)
        np.testing.assert_allclose(trajectory["z_knee_m"], z_knee, atol=1e-14)
        np.testing.assert_allclose(trajectory["x_pull_m"], x_pull, atol=1e-14)
        np.testing.assert_allclose(trajectory["z_pull_m"], z_pull, atol=1e-14)
        assert result.metadata["hip_rom_deg"] == list(FORMAL_HIP_ROM_DEG)
        assert result.metadata["knee_rom_deg"] == list(FORMAL_KNEE_ROM_DEG)


def test_constraint_fields_and_domain_coverage_are_computed(representatives):
    required = {
        "closure_valid",
        "rom_valid",
        "workspace_valid",
        "jacobian_valid",
        "force_mapping_valid",
        "domain_coverage",
        "velocity_valid",
        "acceleration_valid",
        "asymmetry_valid",
        "finite_valid",
        "trajectory_feasible",
        "invalid_reason",
    }
    for result in representatives.values():
        assert required.issubset(result.constraints.as_dict())
        assert 0.0 <= result.constraints.domain_coverage <= 100.0
        assert result.constraints.workspace_valid
        assert result.constraints.jacobian_valid
        assert result.constraints.force_mapping_valid
        assert result.constraints.trajectory_feasible
        assert result.constraints.invalid_reason == ""


def test_failed_gate_is_reported_infeasible_without_clipping(representatives):
    trajectory = representatives["neutral"].trajectory.copy(deep=True)
    trajectory.loc[10, "q_knee_rad"] = np.deg2rad(146.0)
    trajectory.loc[10, "theta_shank_rad"] = (
        trajectory.loc[10, "q_hip_rad"] - trajectory.loc[10, "q_knee_rad"]
    )
    q_hip = trajectory["q_hip_rad"].to_numpy(float)
    q_knee = trajectory["q_knee_rad"].to_numpy(float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, 0.42, 0.30)
    trajectory["x_knee_m"] = x_knee
    trajectory["z_knee_m"] = z_knee
    trajectory["x_pull_m"] = x_pull
    trajectory["z_pull_m"] = z_pull
    from .continuous_reference_neighborhood import evaluate_trajectory_constraints

    audit = evaluate_trajectory_constraints(
        trajectory,
        asymmetry_audit=representatives["neutral"].asymmetry_audit,
        continuity_audit=representatives["neutral"].continuity_audit,
    )
    assert audit.rom_valid is False
    assert audit.trajectory_feasible is False
    assert "rom_invalid" in audit.invalid_reason
    assert np.rad2deg(trajectory.loc[10, "q_knee_rad"]) == pytest.approx(146.0)


def test_generated_validity_columns_are_candidate_specific_and_fail_closed(parent):
    infeasible = generate_personalized_trajectory(parent, 2.0, 2.0, 0.03)
    assert infeasible.constraints.domain_coverage_valid is False
    assert infeasible.constraints.trajectory_feasible is False
    assert not infeasible.trajectory["trajectory_sample_valid"].astype(bool).any()
    assert not infeasible.trajectory["trajectory_feasible"].astype(bool).any()
    assert infeasible.trajectory["invalid_reason"].eq(
        "domain_coverage_insufficient"
    ).all()
    assert not infeasible.trajectory["formal_execution_allowed"].astype(bool).any()
    assert not infeasible.trajectory["allowed_for_first_robot_trial"].astype(bool).any()


@pytest.mark.parametrize(
    "parameters,token",
    [
        ((-5.01, 0.0, 0.0), "hip_amplitude_delta_deg"),
        ((0.0, 2.01, 0.0), "knee_amplitude_delta_deg"),
        ((0.0, 0.0, 0.031), "knee_phase_shift"),
    ],
)
def test_software_search_bounds_reject_without_clipping(parent, parameters, token):
    with pytest.raises(ValueError, match=token):
        generate_personalized_trajectory(parent, *parameters)
    assert OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[token][0] < 0.0


def test_time_scale_is_fixed(parent):
    with pytest.raises(ValueError, match="time_scale is frozen"):
        generate_personalized_trajectory(parent, time_scale=1.01)


def test_mapping_is_continuous_near_alpha(parent):
    base = generate_personalized_trajectory(parent, -2.0, -1.5, 0.01)
    perturbed = generate_personalized_trajectory(parent, -1.999, -1.5, 0.01)
    maximum_change_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    perturbed.trajectory[["q_hip_rad", "q_knee_rad"]].to_numpy()
                    - base.trajectory[["q_hip_rad", "q_knee_rad"]].to_numpy()
                )
            )
        )
    )
    assert maximum_change_deg <= 0.0010000001


def test_candidate_identity_and_sha_are_stable(parent):
    first = generate_personalized_trajectory(parent, -2.0, -1.5, 0.01)
    second = generate_personalized_trajectory(parent, -2.0, -1.5, 0.01)
    assert first.metadata["trajectory_sha256"] == second.metadata["trajectory_sha256"]
    assert first.metadata["trajectory_id"] == parameterized_trajectory_id(
        ContinuousParameters(-2.0, -1.5, 0.01)
    )
    assert first.metadata["generator_version"] == GENERATOR_VERSION


def test_parent_loader_rejects_legacy_or_forged_parent():
    with pytest.raises(TypeError, match="FrozenReferenceBundle"):
        generate_personalized_trajectory(pd.read_csv(RELEASE_ACTIVE_REFERENCE_PATH))


def test_parent_sha_mismatch_fails_closed(parent):
    forged_manifest = dict(parent.manifest)
    forged_manifest["sha256"] = "0" * 64
    forged = replace(parent, manifest=forged_manifest)
    with pytest.raises(RuntimeError, match="REFERENCE_HASH_MISMATCH"):
        generate_personalized_trajectory(forged)


def test_in_memory_parent_content_mismatch_fails_closed(parent):
    forged_trajectory = parent.trajectory.copy(deep=True)
    forged_trajectory.loc[10, "q_knee_rad"] += 1e-6
    forged = replace(parent, trajectory=forged_trajectory)
    with pytest.raises(RuntimeError, match="REFERENCE_HASH_MISMATCH"):
        generate_personalized_trajectory(forged)


def test_grid_artifacts_and_figures_are_generated_without_parent_change(tmp_path):
    before = sha256_file(RELEASE_ACTIVE_REFERENCE_PATH)
    paths = run_continuous_reference_neighborhood(tmp_path)
    assert before == sha256_file(RELEASE_ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256
    expected = {
        "continuous_candidate_parameter_space.csv",
        "representative_regression_points.csv",
        "continuous_generator_metadata.json",
        "candidate_family_hip.png",
        "candidate_family_knee.png",
        "candidate_family_phase.png",
        "candidate_pull_paths.png",
    }
    assert set(paths) == expected
    grid = pd.read_csv(paths["continuous_candidate_parameter_space.csv"])
    assert len(grid) == 27
    assert grid["trajectory_sha256"].nunique() == 27
    metadata = json.loads(paths["continuous_generator_metadata.json"].read_text())
    assert metadata["grid_sample_count"] == 27
    assert metadata["parent_reference_id"] == ACTIVE_REFERENCE_ID
    assert metadata["parent_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["neutral_reference_max_abs_state_error"] == 0.0
    assert metadata["neutral_exact_numeric_state_copy"] is True
    assert metadata["trajectory_sha256_definition"].startswith("sha256_of_utf8")
    assert metadata["theta_shank_definition"] == "q_hip - q_knee"
    assert metadata["optimizer_implemented"] is False
    assert metadata["robot_connection_performed"] is False
    assert metadata["hardware_safety_thresholds_modified"] is False
    for filename, expected_sha in metadata["artifact_sha256"].items():
        assert sha256_file(paths[filename]) == expected_sha
    for filename in expected:
        assert paths[filename].is_file() and paths[filename].stat().st_size > 0


def test_module_has_no_hardware_or_motion_imports():
    paths = [
        Path(__file__).with_name("continuous_reference_neighborhood.py"),
        Path(__file__).with_name("run_continuous_reference_neighborhood.py"),
    ]
    forbidden = ("import hardware", "from hardware", "xCoreSDK", "Rokae", "connect(")
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert not any(token in source for token in forbidden)
    assert "def optimize" not in source
    assert "def select_best" not in source
