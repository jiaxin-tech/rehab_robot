from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from lower_limb_sim.five_leg_mujoco_v1.discriminability_audit import (
    AUDIT_ID,
    build_leg_domain,
    decompose_candidate_response,
    load_frozen_benchmark_definition,
    make_mujoco_model,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "lower_limb_sim" / "five_leg_mujoco_v1"
RESULTS = PACKAGE / "results_discriminability_audit_v1"


@pytest.fixture(scope="module")
def payload():
    return json.loads((RESULTS / "audit_summary.json").read_text(encoding="utf-8"))


def test_audit_reuses_frozen_five_by_625_scope_and_changes_no_objective(payload):
    assert payload[AUDIT_ID] == "COMPLETE"
    assert payload["source_benchmark_id"] == "FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1"
    assert payload["source_benchmark_preserved"] is True
    assert payload["frozen_leg_count"] == 5
    assert payload["frozen_candidate_count_per_leg"] == 625
    assert payload["diagnostic_candidate_replays"] == 5 * 625
    assert payload["primary_endpoint_unchanged"] is True
    assert payload["beta_grid_unchanged"] is True
    assert payload["V3_operator_unchanged"] is True
    assert payload["coordinate_convention"] == "theta_shank = q_hip - q_knee"
    assert payload["algorithm_runs"] == payload["robot_actions"] == payload["PINN_training"] == 0


def test_v3_has_measurable_knee_only_separation_and_timing_effect(payload):
    trajectory = payload["trajectory_separability"]
    assert len(trajectory) == 5
    assert all(row["q_hip_maximum_instantaneous_separation_deg"] == 0.0 for row in trajectory)
    knee_maximum = [row["q_knee_maximum_instantaneous_separation_deg"] for row in trajectory]
    assert min(knee_maximum) > 8.0
    assert max(knee_maximum) < 10.1
    assert all(row["ddq_knee_maximum_instantaneous_separation_deg_s2"] > 24.0 for row in trajectory)
    timing = payload["timing_effects"]
    assert all(
        row["flexion_knee_progression_50_timing_range_s"]
        == pytest.approx(0.1676664424441805)
        for row in timing
    )
    assert all(
        row["extension_knee_progression_50_timing_range_s"]
        == pytest.approx(0.32036222785084334)
        for row in timing
    )


def test_component_decomposition_strictly_reconstructs_frozen_endpoint(payload):
    definition = load_frozen_benchmark_definition()
    leg = definition.legs[0]
    domain = build_leg_domain(leg)
    candidate = next(item for item in domain if item.candidate_index == 13)
    metrics, arrays = decompose_candidate_response(
        model=make_mujoco_model(definition, leg),
        leg=leg,
        time_s=domain.subject_reference.time_s,
        phases=domain.subject_reference.phases,
        q=candidate.trajectory.q,
        dq=candidate.trajectory.dq,
        ddq=candidate.trajectory.ddq,
    )
    assert metrics["component_reconstruction_max_abs_nm"] <= 1.0e-12
    assert np.max(np.linalg.norm(arrays["joint_limit_constraint"], axis=1)) > 0.0
    assert payload["component_decomposition"]["maximum_endpoint_recompute_abs_error_nm"] == 0.0
    assert payload["component_decomposition"]["maximum_reconstruction_abs_error_nm"] <= 2.0e-13


def test_primary_landscapes_are_flat_while_joint_and_time_local_signals_remain(payload):
    dynamic = payload["primary_landscape_dynamic_range"]
    assert all(row["within_5_percent_oracle_count"] == 625 for row in dynamic)
    assert min(row["relative_range_by_reference"] for row in dynamic) < 0.0004
    assert max(row["relative_range_by_reference"] for row in dynamic) < 0.022
    ordering = payload["joint_branch_ordering"]
    assert all(row["knee_oracle_beta"] == [-0.03, 0.03] for row in ordering)
    assert all(row["flexion_oracle_beta"][0] == -0.03 for row in ordering)
    assert all(row["extension_oracle_beta"][1] == 0.03 for row in ordering)
    assert min(row["hip_vs_knee_spearman"] for row in ordering) < -0.99
    local = payload["time_local_sensitivity"]
    assert max(row["total_largest_separation_nm"] for row in local) > 40.0
    assert min(row["total_top_10_percent_time_share_of_integrated_spread"] for row in local) > 0.47


def test_normalized_gradient_direction_is_universal_but_endpoint_compression_dominates(payload):
    gradients = payload["gradient_and_shape_by_leg"]
    assert all(row["global_oracle_beta"] == [-0.03, 0.03] for row in gradients)
    assert all(row["fraction_gradient_points_with_joint_boundary_direction"] == 1.0 for row in gradients)
    assert all(row["interior_strict_local_minimum_count"] == 0 for row in gradients)
    pairwise = payload["cross_leg_normalized_shape"]
    assert min(row["normalized_gradient_field_cosine_similarity"] for row in pairwise) > 0.993
    assert payload["PRIMARY_LIMITATION"] == "MECHANICAL_ENDPOINT_INSUFFICIENTLY_DISCRIMINATIVE"
    assert payload["NEXT_SCIENTIFIC_DIRECTION"] == "REVISIT_MECHANICAL_ENDPOINT"
    assert payload["personalization_algorithm_development_supported"] is False


def test_exactly_five_required_figures_and_no_forbidden_runtime_scope(payload):
    required = {
        "figure_1_selected_joint_space_paths.png",
        "figure_2_q_dq_ddq_spread_vs_time.png",
        "figure_3_normalized_landscapes.png",
        "figure_4_component_sensitivity.png",
        "figure_5_branch_joint_response.png",
    }
    assert {path.name for path in RESULTS.glob("figure_*.png")} == required
    assert all((RESULTS / filename).stat().st_size > 0 for filename in required)
    forbidden_import_roots = {"hardware", "control", "collection", "safety"}
    forbidden_algorithm_fragments = (
        "adaptive_trust_v1",
        "predictive_failover_v1",
        "active_diagnostic_v1",
        "StandardGaussianProcess",
        "LowerConfidenceBoundSelector",
    )
    for filename in (
        "discriminability_audit.py",
        "discriminability_plotting.py",
        "run_discriminability_audit.py",
    ):
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_algorithm_fragments)
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])
        assert not imported & forbidden_import_roots
