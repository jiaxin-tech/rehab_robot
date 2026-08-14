from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from .admissible_personalization_region import (
    DEFAULT_REGION_DIRECTORY,
    MODEL_RELIABILITY_RULE_STATUS,
    REAL_ROBOT_SAFETY_REGION_STATUS,
    REGION_VERSION,
    evaluate_admissible_personalization_region,
    load_admissible_personalization_region,
)
from .config import L1, L2
from .continuous_reference_neighborhood import generate_personalized_trajectory
from .formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    sha256_file,
)
from .kinematics import forward_kinematics
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .run_admissible_personalization_region import parameter_grid_axes
from .sequential_personalization import (
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
    MODEL_RELIABILITY_THRESHOLD,
    SearchAlpha,
    TrustRegionSteps,
    evaluate_candidate_neighborhood,
)
from .dynamic_subject import get_dynamic_subject
from .parameter_estimator import baseline_template_from_dynamic_subject


@pytest.fixture(scope="module")
def region():
    return load_admissible_personalization_region()


@pytest.fixture(scope="module")
def neutral():
    return generate_personalized_trajectory()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_active_reference_belongs_to_admissible_region(region, neutral):
    result = evaluate_admissible_personalization_region(neutral, region=region)
    assert result.trajectory_admissible is True
    assert result.invalid_reason == ""


def test_alpha_zero_is_admissible_and_checks_all_401_samples(region, neutral):
    result = evaluate_admissible_personalization_region(neutral, region=region)
    assert result.alpha_bounds_valid is True
    assert result.checked_sample_count == 401
    assert len(result.sample_audit) == 401
    assert result.first_invalid_sample is None


def test_global_rom_protocol_remains_v2():
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)


def test_phasewise_joint_corridor_contains_reference(region):
    corridor = region.joint_corridor
    assert (corridor.q_hip_min_rad <= corridor.q_hip_ref_rad).all()
    assert (corridor.q_hip_ref_rad <= corridor.q_hip_max_rad).all()
    assert (corridor.q_knee_min_rad <= corridor.q_knee_ref_rad).all()
    assert (corridor.q_knee_ref_rad <= corridor.q_knee_max_rad).all()


def test_phasewise_pull_corridor_contains_reference(region):
    corridor = region.pull_corridor
    assert (corridor.x_pull_min_m <= corridor.x_pull_ref_m).all()
    assert (corridor.x_pull_ref_m <= corridor.x_pull_max_m).all()
    assert (corridor.z_pull_min_m <= corridor.z_pull_ref_m).all()
    assert (corridor.z_pull_ref_m <= corridor.z_pull_max_m).all()
    assert (corridor.pull_radial_max_mm >= 0.0).all()


def test_joint_corridor_is_phase_dependent_generator_envelope(region):
    hip_width = np.rad2deg(
        region.joint_corridor.q_hip_max_rad - region.joint_corridor.q_hip_min_rad
    )
    knee_width = np.rad2deg(
        region.joint_corridor.q_knee_max_rad - region.joint_corridor.q_knee_min_rad
    )
    assert np.ptp(hip_width) > 1.0
    assert np.ptp(knee_width) > 1.0
    assert hip_width[0] == pytest.approx(0.0, abs=1e-10)
    assert knee_width[0] == pytest.approx(0.0, abs=1e-10)


def test_phase_warp_knee_deviation_is_included(region):
    generated = generate_personalized_trajectory(None, 0.0, 0.0, 0.03)
    assert generated.metadata["knee_max_deviation_deg"] > 4.0
    result = evaluate_admissible_personalization_region(generated, region=region)
    assert result.joint_corridor_valid is True


def test_pull_corridor_reference_is_recomputed_fk(region):
    joint = region.joint_corridor
    _, _, x_pull, z_pull = forward_kinematics(
        joint.q_hip_ref_rad.to_numpy(float),
        joint.q_knee_ref_rad.to_numpy(float),
        L1,
        L2,
    )
    np.testing.assert_allclose(x_pull, region.pull_corridor.x_pull_ref_m, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(z_pull, region.pull_corridor.z_pull_ref_m, atol=1e-12, rtol=0.0)


def test_single_joint_corridor_violation_rejects_entire_trajectory(region, neutral):
    trajectory = neutral.trajectory.copy(deep=True)
    index = 123
    trajectory.loc[index, "q_hip_rad"] = region.joint_corridor.loc[index, "q_hip_max_rad"] + 1e-4
    forged = replace(neutral, trajectory=trajectory)
    result = evaluate_admissible_personalization_region(forged, region=region)
    assert result.trajectory_admissible is False
    assert result.joint_corridor_valid is False
    assert result.first_invalid_sample == index
    assert result.invalid_phase == pytest.approx(float(trajectory.loc[index, "global_phase"]))


def test_global_rom_valid_does_not_bypass_local_joint_corridor(region, neutral):
    trajectory = neutral.trajectory.copy(deep=True)
    index = 150
    new_q = float(region.joint_corridor.loc[index, "q_hip_max_rad"] + 1e-4)
    assert FORMAL_HIP_ROM_DEG[0] < np.rad2deg(new_q) < FORMAL_HIP_ROM_DEG[1]
    trajectory.loc[index, "q_hip_rad"] = new_q
    result = evaluate_admissible_personalization_region(
        replace(neutral, trajectory=trajectory), region=region
    )
    assert result.global_rom_valid is True
    assert result.joint_corridor_valid is False
    assert result.trajectory_admissible is False


def test_joint_corridor_valid_does_not_bypass_pull_corridor(region):
    candidate = generate_personalized_trajectory(None, -1.0, 0.0, 0.0)
    narrow = region.pull_corridor.copy(deep=True)
    narrow["x_pull_min_m"] = narrow["x_pull_ref_m"]
    narrow["x_pull_max_m"] = narrow["x_pull_ref_m"]
    narrow["z_pull_min_m"] = narrow["z_pull_ref_m"]
    narrow["z_pull_max_m"] = narrow["z_pull_ref_m"]
    narrow["pull_radial_max_mm"] = 0.0
    narrowed_region = replace(region, pull_corridor=narrow)
    result = evaluate_admissible_personalization_region(
        candidate, region=narrowed_region
    )
    assert result.joint_corridor_valid is True
    assert result.pull_corridor_valid is False
    assert result.trajectory_admissible is False


def test_geometry_valid_does_not_bypass_insufficient_domain(region):
    rejected = pd.read_csv(DEFAULT_REGION_DIRECTORY / "rejected_parameter_samples.csv")
    row = rejected.loc[
        rejected.invalid_reason.astype(str).str.contains("identification_domain_insufficient")
        & rejected.workspace_valid.astype(bool)
        & rejected.jacobian_valid.astype(bool)
    ].iloc[0]
    generated = generate_personalized_trajectory(
        None, float(row.hip_delta), float(row.knee_delta), float(row.phase_delta)
    )
    result = evaluate_admissible_personalization_region(generated, region=region)
    assert result.workspace_valid is True
    assert result.jacobian_valid is True
    assert result.domain_valid is False
    assert result.trajectory_admissible is False
    assert "identification_domain_insufficient" in result.invalid_reason


def test_parameter_box_contains_interior_infeasible_samples(region):
    assert region.summary["parameter_box_contains_interior_infeasible_samples"] is True
    assert region.summary["interior_rejected_parameter_sample_count"] > 0
    assert region.summary["parameter_box_claimed_fully_feasible"] is False


def test_region_never_applies_clipping():
    table = pd.read_csv(DEFAULT_REGION_DIRECTORY / "parameter_space_admissibility.csv")
    assert not table.pointwise_clipping_applied.astype(bool).any()
    with pytest.raises(ValueError, match="clipping is prohibited"):
        generate_personalized_trajectory(None, -5.01, 0.0, 0.0)


def test_invalid_reason_and_first_invalid_sample_are_saved(region):
    rejected = pd.read_csv(DEFAULT_REGION_DIRECTORY / "rejected_parameter_samples.csv").iloc[0]
    generated = generate_personalized_trajectory(
        None,
        float(rejected.hip_delta),
        float(rejected.knee_delta),
        float(rejected.phase_delta),
    )
    result = evaluate_admissible_personalization_region(generated, region=region)
    assert result.invalid_reason
    assert result.first_invalid_sample is not None
    assert result.invalid_phase is not None
    assert result.first_invalid_sample_reason


def test_parent_sha_mismatch_fails_closed(region, neutral):
    metadata = {**neutral.metadata, "parent_reference_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="REFERENCE_HASH_MISMATCH"):
        evaluate_admissible_personalization_region(
            replace(neutral, metadata=metadata), region=region
        )


def test_legacy_symmetric_parent_fails_closed(region, neutral):
    metadata = {**neutral.metadata, "parent_reference_id": "reference_closed_symmetric"}
    with pytest.raises(PermissionError, match="legacy"):
        evaluate_admissible_personalization_region(
            replace(neutral, metadata=metadata), region=region
        )


def test_parameter_grid_is_full_frozen_minimum_step_lattice():
    axes = parameter_grid_axes()
    assert tuple(map(len, axes.values())) == (29, 29, 25)
    assert int(np.prod([len(values) for values in axes.values()])) == 21025
    assert all(np.any(np.isclose(values, 0.0)) for values in axes.values())


def test_representative_trajectories_are_formally_reported():
    table = pd.read_csv(
        DEFAULT_REGION_DIRECTORY / "representative_trajectory_admissibility.csv"
    )
    assert set(table.label) == {
        "neutral",
        "hip_negative",
        "knee_negative",
        "positive_phase",
        "negative_phase",
        "combined_perturbation",
    }


def test_sequential_ranking_receives_only_admissible_candidates(region):
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    parameters = {
        "mass_scale": 1.0,
        "k_hip_nm_per_rad": 15.0,
        "k_knee_nm_per_rad": 12.0,
        "b_hip_nm_s_per_rad": 2.0,
        "b_knee_nm_s_per_rad": 1.5,
    }
    evaluations, _ = evaluate_candidate_neighborhood(
        current=SearchAlpha(),
        steps=TrustRegionSteps(),
        estimated_parameters=parameters,
        template=template,
        admissible_region=region,
    )
    rows = pd.DataFrame([item.row for item in evaluations])
    assert (rows.trajectory_feasible == rows.trajectory_admissible).all()
    assert not rows.loc[~rows.trajectory_admissible.astype(bool), "mechanical_cost_j_rms"].notna().any()


def test_sequential_source_calls_unified_admissibility_api():
    source = inspect.getsource(evaluate_candidate_neighborhood)
    assert "evaluate_admissible_personalization_region" in source
    assert "trajectory_admissible" in source


def test_mechanical_objective_and_reliability_rule_remain_frozen():
    path = Path(__file__).resolve().parent / "mechanical_objective.py"
    assert _sha256(path) == "e20a391ecb8362346ed742a01593aa7a58f41ec2493b8c5defcbf385b3e18d67"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert MODEL_RELIABILITY_THRESHOLD is None
    assert MODEL_RELIABILITY_RULE_STATUS == "NOT_FROZEN"


def test_trust_region_steps_remain_frozen():
    assert MINIMUM_STEP_HIP_DEG == 0.25
    assert MINIMUM_STEP_KNEE_DEG == 0.25
    assert MINIMUM_STEP_PHASE == 0.0025


def test_hardware_and_safety_have_no_worktree_diff():
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        ["git", "diff", "--quiet", "--", "hardware", "safety"], cwd=root
    )
    assert completed.returncode == 0


def test_active_reference_sha_and_real_robot_boundary_are_unchanged(region):
    assert sha256_file(ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256
    assert region.manifest["region_version"] == REGION_VERSION
    assert region.manifest["real_robot_safety_region_status"] == REAL_ROBOT_SAFETY_REGION_STATUS
    assert region.manifest["offline_region_is_real_robot_safety_region"] is False
    assert region.manifest["formal_sequential_experiment_rerun"] is False
