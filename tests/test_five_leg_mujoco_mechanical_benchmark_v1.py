from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lower_limb_sim.five_leg_mujoco_v1 import (
    EXPECTED_LEG_IDS,
    build_leg_domain,
    custom_passive_torque_project_coordinates,
    load_frozen_benchmark_definition,
    make_mujoco_model,
    replay_trajectory,
)
from personalization.rom_gated_v2.rom import FROZEN_SUBJECT_ROM_PROFILE_V1


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def definition():
    return load_frozen_benchmark_definition()


@pytest.fixture(scope="module")
def nominal_stage(definition):
    leg = definition.legs[0]
    domain = build_leg_domain(leg)
    model = make_mujoco_model(definition, leg)
    return leg, domain, model


def test_exactly_five_deterministic_frozen_leg_definitions(definition):
    assert tuple(leg.leg_id for leg in definition.legs) == EXPECTED_LEG_IDS
    assert load_frozen_benchmark_definition() == definition
    encoded = json.dumps(json.loads((
        ROOT
        / "lower_limb_sim"
        / "five_leg_mujoco_v1"
        / "FROZEN_FIVE_LEG_MECHANICAL_PARAMETERS_V1.json"
    ).read_text(encoding="utf-8")))
    for legacy_label in ('"P0"', '"P1"', '"P2"', '"P3"'):
        assert legacy_label not in encoded


def test_leg_differences_are_structural_not_uniform_scaling(definition):
    legs = definition.legs
    mass_ratios = {
        round(leg.mass_thigh_kg / leg.mass_shank_kg, 8) for leg in legs
    }
    inertia_ratios = {
        round(leg.inertia_thigh_kg_m2 / leg.inertia_shank_kg_m2, 8)
        for leg in legs
    }
    assert len(mass_ratios) == 5
    assert len(inertia_ratios) == 5
    assert sum(leg.hip_cubic_stiffness_nm_per_rad3 > 0.0 for leg in legs) == 4
    assert sum(leg.coupling_stiffness_nm_per_rad > 0.0 for leg in legs) == 4
    assert len({leg.coupling_ratio for leg in legs}) == 5


def test_each_leg_has_a_frozen_synthetic_subject_rom(definition):
    profiles = [leg.make_rom_profile() for leg in definition.legs]
    assert len({profile.profile_id for profile in profiles}) == 5
    assert len({profile.fingerprint for profile in profiles}) == 5
    for profile in profiles:
        assert profile.frozen is True
        assert profile.rom_status == FROZEN_SUBJECT_ROM_PROFILE_V1
        assert profile.provenance == "SYNTHETIC_MUJOCO_BENCHMARK_ROM_ONLY"
        assert profile.metadata["not_clinical_population_evidence"] is True
        assert profile.metadata["parameters_frozen_before_landscape_generation"] is True


def test_subject_specific_domain_has_625_fixed_rom_candidates_without_clipping(
    nominal_stage,
):
    _, domain, _ = nominal_stage
    summary = domain.invariant_summary()
    assert len(domain) == 625
    assert summary["all_kinematic_gates_pass"] is True
    assert summary["pointwise_clipping_count"] == 0
    assert summary["maximum_extrema_error_rad"] <= 2.0e-5
    assert summary["minimum_warp_derivative"] > 0.0
    assert summary["duration_s"] == 24.0


def test_mujoco_model_is_fixed_pelvis_two_dof_and_preserves_coordinate_convention(
    definition, nominal_stage
):
    _, _, model = nominal_stage
    assert model.nq == model.nv == model.nu == 2
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee") >= 0
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "equivalent_cuff_interaction_point"
    )
    assert site_id >= 0

    q_hip = 0.8
    q_knee = 1.1
    data = mujoco.MjData(model)
    data.qpos[:] = (q_hip, -q_knee)
    mujoco.mj_forward(model, data)
    expected_x = (
        definition.thigh_length_m * math.cos(q_hip)
        + definition.cuff_distance_m * math.cos(q_hip - q_knee)
    )
    expected_z = (
        definition.thigh_length_m * math.sin(q_hip)
        + definition.cuff_distance_m * math.sin(q_hip - q_knee)
    )
    assert data.site_xpos[site_id, 0] == pytest.approx(expected_x, abs=1.0e-12)
    assert data.site_xpos[site_id, 2] == pytest.approx(expected_z, abs=1.0e-12)


def test_mujoco_inverse_dynamics_replay_is_deterministic_and_tracking_valid(
    definition, nominal_stage
):
    leg, domain, model = nominal_stage
    trajectory = domain.reference.trajectory
    kwargs = {
        "model": model,
        "definition": definition,
        "leg": leg,
        "time_s": domain.subject_reference.time_s,
        "q_project": trajectory.q,
        "dq_project": trajectory.dq,
        "ddq_project": trajectory.ddq,
    }
    first = replay_trajectory(**kwargs)
    second = replay_trajectory(**kwargs)
    assert first.valid is True
    assert first.invalid_reason is None
    assert first.endpoint_value_nm == second.endpoint_value_nm
    assert np.array_equal(first.tau_hip_nm, second.tau_hip_nm)
    assert np.array_equal(first.tau_knee_nm, second.tau_knee_nm)
    assert first.tracking_rms_rad <= definition.tracking_failure_rms_rad
    assert first.tracking_max_abs_rad == 0.0


def test_out_of_rom_trajectory_is_invalid_without_clipping(definition, nominal_stage):
    leg, domain, model = nominal_stage
    trajectory = domain.reference.trajectory
    outside = np.asarray(trajectory.q).copy()
    outside[10, 0] = leg.hip_max_rad + 0.01
    result = replay_trajectory(
        model=model,
        definition=definition,
        leg=leg,
        time_s=domain.subject_reference.time_s,
        q_project=outside,
        dq_project=trajectory.dq,
        ddq_project=trajectory.ddq,
    )
    assert result.valid is False
    assert result.invalid_reason == "TRAJECTORY_OUTSIDE_FROZEN_ROM_NO_CLIPPING"
    assert result.endpoint_value_nm is None


def test_structural_custom_passive_terms_are_active_only_as_declared(definition):
    q_hip, q_knee = 1.4, 1.7
    nominal = custom_passive_torque_project_coordinates(
        definition.legs[0], q_hip, q_knee
    )
    nonlinear = custom_passive_torque_project_coordinates(
        definition.legs[3], q_hip, q_knee
    )
    mismatch = custom_passive_torque_project_coordinates(
        definition.legs[4], q_hip, q_knee
    )
    assert np.array_equal(nominal, np.zeros(2))
    assert np.isfinite(nonlinear).all()
    assert np.isfinite(mismatch).all()
    assert not np.allclose(nonlinear, mismatch)


def test_new_package_has_no_personalization_algorithm_or_robot_imports():
    package = ROOT / "lower_limb_sim" / "five_leg_mujoco_v1"
    forbidden_roots = {"hardware", "control", "collection", "safety"}
    forbidden_algorithm_fragments = (
        "adaptive_trust_v1",
        "predictive_failover_v1",
        "active_diagnostic_v1",
        "repeated_active_diagnostic_k5_v1",
        "StandardGaussianProcess",
        "LowerConfidenceBoundSelector",
    )
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_algorithm_fragments)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])
        assert not (imported & forbidden_roots)


def test_formal_landscape_oracle_graybox_and_readiness_artifacts():
    results = (
        ROOT / "lower_limb_sim" / "five_leg_mujoco_v1" / "results"
    )
    payload = json.loads(
        (results / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert payload["FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1"] == "COMPLETE"
    assert payload["total_candidate_replays"] == 5 * 625
    assert payload["algorithm_runs"] == 0
    assert payload["robot_actions"] == 0
    assert payload["PINN_training"] == 0

    audits = payload["trajectory_audits"]
    assert len(audits) == 5
    assert all(item["valid_candidate_count"] == 625 for item in audits)
    assert all(item["invalid_candidate_count"] == 0 for item in audits)
    assert all(item["endpoint_reproducible"] for item in audits)
    assert all(item["maximum_tracking_rms_rad"] == 0.0 for item in audits)
    assert all(item["pointwise_clipping_count"] == 0 for item in audits)

    oracle = payload["oracle_characterization"]
    assert len(oracle) == 5
    assert all(item["oracle_beta"] == [-0.03, 0.03] for item in oracle)
    assert all(item["oracle_canonical_beta_id"] == "MYOLEG_V3_K0024" for item in oracle)
    assert all(item["valid_candidates"] == 625 for item in oracle)

    pairwise = payload["pairwise_landscape_analysis"]
    assert len(pairwise) == 10
    correlations = [item["spearman_rank_correlation"] for item in pairwise]
    assert min(correlations) == pytest.approx(0.991982157778324)
    assert max(correlations) == pytest.approx(0.9998579503563529)

    gray_box = payload["gray_box_prediction_quality"]
    assert len(gray_box) == 5
    assert all(item["fit_evidence"] == "REFERENCE_CANDIDATE_ONLY" for item in gray_box)
    assert all(item["fit_valid_episode_count"] == 1 for item in gray_box)
    assert all(not item["directionally_misleading"] for item in gray_box)
    assert all(item["predicted_oracle_beta"] == item["true_oracle_beta"] for item in gray_box)

    necessity = payload["personalization_necessity_analysis"]
    assert necessity["unique_oracle_beta_count"] == 1
    assert necessity["universal_within_5_percent_candidate_count"] == 625
    assert necessity["minimax_common_candidate"]["maximum_relative_regret"] == 0.0
    assert necessity["normalized_landscape_rank_one_explained_fraction"] == pytest.approx(
        0.9999969876252084
    )
    assert necessity["normalized_subject_by_trajectory_interaction_rms"] == pytest.approx(
        0.0017361869871018328
    )
    assert necessity["conclusion"] == (
        "MUJOCO_BENCHMARK_DOES_NOT_SUPPORT_PERSONALIZATION_NECESSITY"
    )
    assert payload[
        "READY_FOR_FROZEN_ALGORITHM_COMPARISON_ON_FIVE_LEG_MUJOCO"
    ] is False

    required_outputs = (
        "table_1_leg_parameters_and_rom.csv",
        "table_2_oracle_characterization.csv",
        "table_3_pairwise_rank_correlations.csv",
        "table_4_gray_box_prediction_quality.csv",
        "full_landscapes.csv",
        "figure_1_truth_landscapes.png",
        "figure_2_truth_vs_gray_box.png",
        "figure_3_oracle_near_oracle.png",
    )
    assert all((results / name).stat().st_size > 0 for name in required_outputs)
