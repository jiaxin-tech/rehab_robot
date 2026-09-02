"""Regression tests for the development-only V2 parameterization audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_simulation.myoleg_v2_trajectory_parameterization_boundary_audit_v1 import build_audit
from external_simulation.myoleg_v2_trajectory_parameterization_boundary_audit_v1 import prototype_parameterizations


OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_trajectory_parameterization_boundary_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_protocol_was_frozen_before_development_values_and_is_unchanged() -> None:
    protocol_path = OUTPUT / "TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_PROTOCOL.json"
    protocol = read_json(protocol_path.name)
    assert sha256(protocol_path) == build_audit.EXPECTED_PROTOCOL_SHA256
    assert protocol["frozen_before_new_development_scientific_values_read"] is True
    assert protocol["population"]["development_count"] == 24
    assert protocol["population"]["held_out_scientific_values_allowed"] is False
    assert protocol["candidate_effect_diagnostics"]["causal_interpretation_forbidden"] is True
    assert protocol["candidate_effect_diagnostics"]["production_learner"] is False


def test_every_formal_artifact_checksum_passes() -> None:
    lines = (OUTPUT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 29
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = OUTPUT / relative.strip()
        assert path.is_file()
        assert sha256(path) == expected


def test_all_16675_candidates_have_outcome_free_kinematic_descriptors() -> None:
    rows = read_csv("CANDIDATE_KINEMATIC_DESCRIPTOR_TABLE.csv")
    assert len(rows) == 16675
    assert len({row["candidate_id"] for row in rows}) == 16675
    assert len({row["proposal_index"] for row in rows}) == 16675
    forbidden = {"j_truth", "mean_j", "oracle", "rank", "held_out"}
    assert forbidden.isdisjoint(rows[0])
    assert all(row["finite"] == "True" for row in rows)


def test_current_parameter_semantics_distinguish_task_amplitude_from_phase() -> None:
    rows = {row["parameter"]: row for row in read_csv("CURRENT_PARAMETER_SEMANTICS_AUDIT.csv")}
    assert rows["delta_hip_amp"]["changes_rehabilitation_task_amplitude"] == "True"
    assert rows["delta_knee_amp"]["changes_rehabilitation_task_amplitude"] == "True"
    assert rows["knee_phase_shift"]["changes_rehabilitation_task_amplitude"] == "False"
    assert all(row["changes_cycle_duration"] == "False" for row in rows.values())
    assert all(row["preserves_closure"] == "True" and row["preserves_C2"] == "True" for row in rows.values())
    assert float(rows["delta_hip_amp"]["delta_hip_range_deg"]) == pytest.approx(7.0, abs=2e-5)
    assert float(rows["delta_knee_amp"]["delta_knee_range_deg"]) == pytest.approx(5.5, abs=2e-3)


def test_rom_extrema_common_effect_is_descriptive_and_dominant() -> None:
    audit = read_json("ROM_EXTREMA_COMMON_EFFECT_AUDIT.json")
    assert audit["ROM_EXTREMA_EXPLAINED_COMMON_EFFECT"] == pytest.approx(0.9977422515561761, abs=1e-12)
    assert audit["interpretation"] == "DOMINANT"
    assert audit["association_is_not_causality"] is True
    assert audit["production_learner_trained"] is False


def test_matched_rom_grid_has_only_one_path_coordinate() -> None:
    summary = read_json("MATCHED_ROM_ANALYSIS_SUMMARY.json")
    rows = read_csv("MATCHED_ROM_ANALYSIS.csv")
    assert summary["matched_group_count"] == 667
    assert summary["candidate_count_per_complete_group_min"] == 25
    assert summary["candidate_count_per_complete_group_max"] == 25
    assert summary["maximum_independent_path_shape_dimensions_within_matched_ROM"] == 1
    assert summary["status"] == "CURRENT_GRID_CANNOT_IDENTIFY_FIXED_ROM_PATH_EFFECT"
    assert len(rows) == 667
    assert max(float(row["hip_extrema_span_within_group_deg"]) for row in rows) <= 1e-3
    assert max(float(row["knee_extrema_span_within_group_deg"]) for row in rows) <= 1e-3


def test_phase_only_slices_are_preregistered_common_monotonic() -> None:
    summary = read_json("PHASE_ONLY_SUBSPACE_SUMMARY.json")
    rows = read_csv("PHASE_ONLY_SUBSPACE_AUDIT.csv")
    assert summary["all_preregistered_slices_common_monotonic"] is True
    assert summary["direction"] == "phase increase worsens J"
    assert len(rows) == 6
    assert all(int(row["phase_value_count"]) == 25 for row in rows)
    assert all(int(row["subjects_monotonic_J_increasing_with_phase"]) == 24 for row in rows)
    assert all(int(row["subjects_sharing_common_oracle"]) == 24 for row in rows)
    assert all(float(row["common_oracle_phase"]) == pytest.approx(-0.03) for row in rows)


def test_phase_mechanism_reuses_only_frozen_development_replay_subset() -> None:
    summary = read_json("PHASE_MECHANISTIC_DECOMPOSITION_SUMMARY.json")
    rows = read_csv("PHASE_MECHANISTIC_DECOMPOSITION.csv")
    assert summary["subject_count"] == 6
    assert summary["replay_pair_count_used"] == 30
    assert summary["held_out_replay_pair_count"] == 0
    assert len(rows) == 60
    assert set(row["joint"] for row in rows) == {"hip", "knee"}


def test_boundary_sources_are_not_conflated_or_extrapolated() -> None:
    audit = read_json("BOUNDARY_SOURCE_AUDIT.json")
    classifications = {row["dimension"]: row["classification"] for row in audit["dimensions"]}
    assert classifications["hip amplitude upper"] == "A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE"
    assert classifications["knee amplitude upper"] == "B_MYOLEG_SIMULATOR_VALIDITY_LIMIT"
    assert classifications["knee phase lower"] == "A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE"
    assert audit["widen_current_bounds_as_primary_fix"] is False
    assert all(row["out_of_domain_optimum_claimed"] is False for row in audit["dimensions"])


def test_kinematic_prototypes_close_without_clipping_or_objective() -> None:
    rows = read_csv("KINEMATIC_PROTOTYPE_AUDIT.csv")
    assert {row["parameterization_id"] for row in rows} == {
        "P1_PHASE_COORDINATION_ONLY", "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION",
        "P3_JOINT_SPACE_NORMAL_DISPLACEMENT", "P4_BRANCH_AWARE_COORDINATION_FUNCTION",
    }
    assert max(int(row["dimension"]) for row in rows) <= 4
    assert all(row["finite"] == "True" for row in rows)
    assert all(row["post_generation_clipping_used"] == "False" for row in rows)
    assert all(row["myoleg_objective_evaluated"] == "False" for row in rows)
    assert all(float(row["q_closure_error_rad"]) <= 1e-12 for row in rows)
    assert all(float(row["dq_closure_error_rad_s"]) <= 1e-12 for row in rows)
    assert all(float(row["ddq_closure_error_rad_s2"]) <= 1e-12 for row in rows)
    p3 = next(row for row in rows if row["parameterization_id"].startswith("P3_"))
    assert float(p3["normal_low_speed_degeneracy_fraction"]) > 0.0


def test_interior_basis_has_zero_value_and_two_derivatives_at_boundaries() -> None:
    bump, first, second = prototype_parameterizations._branch_bump(np.asarray([0.0, 1.0]))
    assert np.array_equal(bump, np.zeros(2))
    assert np.array_equal(first, np.zeros(2))
    assert np.array_equal(second, np.zeros(2))


def test_v3_recommendation_is_structural_default_off_and_range_unfrozen() -> None:
    recommendation = read_json("V3_PARAMETERIZATION_RECOMMENDATION.json")
    assert recommendation["current_v2_decision"] == "NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION"
    assert recommendation["V3_PRIMARY_PARAMETERIZATION"]["id"] == "P4_BRANCH_AWARE_COORDINATION_FUNCTION"
    assert recommendation["V3_FALLBACK_PARAMETERIZATION"]["id"] == "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION"
    assert recommendation["V3_PRIMARY_PARAMETERIZATION"]["numeric_range_frozen"] is False
    assert recommendation["V3_FALLBACK_PARAMETERIZATION"]["numeric_range_frozen"] is False
    assert recommendation["decision_basis"]["selection_used_new_outcome_diversity"] is False
    assert recommendation["execute_next_stage_now"] is False


def test_held_out_stays_sealed_and_all_frozen_inputs_remain_unchanged() -> None:
    access = read_json("HELD_OUT_ACCESS_AUDIT.json")
    metadata = read_json("metadata.json")
    assert access["np_load_held_out_count"] == 0
    assert access["held_out_scientific_truth_access_count"] == 0
    assert access["held_out_j_oracle_rank_component_access_count"] == 0
    assert metadata["held_out_scientific_truth_access_count"] == 0
    for name, expected in build_audit.FROZEN_SHA.items():
        assert sha256(build_audit.frozen_paths()[name]) == expected


def test_scope_guards_and_required_figures() -> None:
    metadata = read_json("metadata.json")
    scope = metadata["scope"]
    assert scope == {
        "bo": False,
        "development_only": True,
        "five_parameter": False,
        "frozen_v2_modified": False,
        "held_out_truth": False,
        "new_MyoLeg_objective_replay": False,
        "new_v3_landscape": False,
        "nn_or_pinn": False,
        "offline_only": True,
        "robot_or_hardware": False,
    }
    figures = sorted((OUTPUT / "figures").glob("*.png"))
    assert len(figures) == 8
    assert all(path.stat().st_size > 20_000 for path in figures)
    assert metadata["analysis_code_sha256"] == sha256(Path(build_audit.__file__))
    assert metadata["prototype_code_sha256"] == sha256(Path(prototype_parameterizations.__file__))
