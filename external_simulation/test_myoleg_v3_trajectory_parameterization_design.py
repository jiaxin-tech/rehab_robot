"""Regression tests for the frozen MyoLeg-V3 parameterization design."""

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

from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import build_design
from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import parameterization


OUTPUT = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1"


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


def test_protocol_was_frozen_before_sweep_and_smoke() -> None:
    protocol = read_json("V3_PARAMETERIZATION_DESIGN_PROTOCOL.json")
    assert sha256(OUTPUT / "V3_PARAMETERIZATION_DESIGN_PROTOCOL.json") == build_design.EXPECTED_PROTOCOL_SHA256
    assert protocol["frozen_before_wide_kinematic_sweep"] is True
    assert protocol["frozen_before_nominal_simulator_smoke"] is True
    assert protocol["scientific_data_access"] == {
        "development_truth_allowed": False,
        "held_out_truth_allowed": False,
        "mechanical_objective_allowed": False,
        "subject_model_allowed": False,
    }
    assert protocol["parameterization"]["dimension"] == 2
    assert protocol["parameterization"]["pointwise_clipping"] is False


def test_warp_has_identity_value_and_two_derivatives_at_branch_endpoints() -> None:
    s = np.asarray([0.0, 1.0])
    basis, first, second = parameterization.interior_warp_basis(s)
    assert np.array_equal(basis, np.zeros(2))
    assert np.array_equal(first, np.zeros(2))
    assert np.array_equal(second, np.zeros(2))
    for beta in (-0.03, 0.0, 0.03):
        warped, derivative, second_derivative = parameterization.branch_warp(s, beta)
        assert np.array_equal(warped, s)
        assert np.array_equal(derivative, np.ones(2))
        assert np.array_equal(second_derivative, np.zeros(2))


def test_zero_beta_recovers_frozen_reference_array_exactly() -> None:
    recovery = read_json("REFERENCE_RECOVERY_AUDIT.json")
    assert recovery["pass"] is True
    assert recovery["q_array_equal"] is True
    assert recovery["dq_array_equal"] is True
    assert recovery["ddq_array_equal"] is True
    assert recovery["q_max_abs_error"] == 0.0
    assert recovery["dq_max_abs_error"] == 0.0
    assert recovery["ddq_max_abs_error"] == 0.0


def test_frozen_candidate_domain_is_two_dimensional_small_and_contains_reference() -> None:
    manifest = read_json("MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json")
    rows = read_csv("V3_KINEMATIC_CANDIDATE_TABLE.csv")
    assert manifest["manifest_id"] == "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1"
    assert manifest["protocol_sha256"] == build_design.EXPECTED_PROTOCOL_SHA256
    assert manifest["parameter_order"] == ["beta_flex", "beta_extend"]
    assert manifest["beta_range"] == [-0.03, 0.03]
    assert manifest["beta_step"] == 0.0025
    assert manifest["axis_count"] == 25
    assert manifest["candidate_count"] == 625
    assert len(rows) == 625
    assert len({row["candidate_id"] for row in rows}) == 625
    assert sha256(OUTPUT / "V3_KINEMATIC_CANDIDATE_TABLE.csv") == manifest["candidate_table_sha256"]
    reference = [row for row in rows if float(row["beta_flex"]) == 0.0 and float(row["beta_extend"]) == 0.0]
    assert len(reference) == 1
    assert reference[0]["included"] == "True"


def test_every_admitted_candidate_preserves_task_amplitude_and_hip_exactly() -> None:
    rows = read_csv("V3_TASK_INVARIANCE_AUDIT.csv")
    assert len(rows) == 625
    error_fields = (
        "hip_min_error_deg", "hip_max_error_deg", "hip_rom_error_deg",
        "knee_min_error_deg", "knee_max_error_deg", "knee_rom_error_deg",
    )
    assert max(float(row[key]) for row in rows for key in error_fields) <= 1.0e-3
    assert max(float(row["hip_q_max_abs_error_rad"]) for row in rows) == 0.0
    assert max(float(row["hip_dq_max_abs_error_rad_s"]) for row in rows) == 0.0
    assert max(float(row["hip_ddq_max_abs_error_rad_s2"]) for row in rows) == 0.0
    assert all(float(row["duration_s"]) == 24.0 and int(row["sample_count"]) == 401 for row in rows)
    assert all(row["kinematic_gate_pass"] == "True" and row["exclusion_reason"] == "" for row in rows)


def test_every_candidate_preserves_closure_C2_anchors_and_monotonicity() -> None:
    rows = read_csv("V3_C2_CLOSURE_AUDIT.csv")
    assert len(rows) == 625
    assert max(float(row["q_closure_error_rad"]) for row in rows) <= 1.0e-10
    assert max(float(row["dq_closure_error_rad_s"]) for row in rows) <= 1.0e-10
    assert max(float(row["ddq_closure_error_rad_s2"]) for row in rows) <= 1.0e-9
    assert max(float(row["branch_anchor_q_max_error_rad"]) for row in rows) == 0.0
    assert max(float(row["branch_anchor_dq_max_error_rad_s"]) for row in rows) == 0.0
    assert max(float(row["branch_anchor_ddq_max_error_rad_s2"]) for row in rows) == 0.0
    assert min(float(row["minimum_warp_derivative"]) for row in rows) >= 0.85
    assert all(row["kinematic_gate_pass"] == "True" for row in rows)


def test_wide_sweep_freezes_range_without_objective_or_outcome_columns() -> None:
    rows = read_csv("V3_BETA_RANGE_AUDIT.csv")
    assert len(rows) == 402
    assert {row["swept_branch"] for row in rows} == {"flex", "extend"}
    assert min(float(row["swept_beta"]) for row in rows) == -0.25
    assert max(float(row["swept_beta"]) for row in rows) == 0.25
    forbidden = {"J", "objective", "oracle", "rank", "subject_id", "held_out"}
    assert forbidden.isdisjoint(rows[0])
    extension_first_positive_outside = next(
        row for row in rows
        if row["swept_branch"] == "extend" and float(row["swept_beta"]) == 0.0325
    )
    assert extension_first_positive_outside["kinematic_gate_pass"] == "False"
    assert "maximum_knee_displacement" in extension_first_positive_outside["exclusion_reason"]


def test_fine_grid_is_selected_by_preregistered_resolution_rule() -> None:
    rows = {row["grid_name"]: row for row in read_csv("V3_GRID_RESOLUTION_AUDIT.csv")}
    assert rows["COARSE"]["grid_pass"] == "False"
    assert rows["MEDIUM"]["grid_pass"] == "False"
    assert rows["FINE"]["grid_pass"] == "True"
    assert int(rows["FINE"]["candidate_count"]) == 625
    assert float(rows["FINE"]["max_adjacent_change_fraction_of_reference_knee_rom"]) <= 0.005


def test_nominal_smoke_is_post_manifest_sparse_integrity_only_and_passes() -> None:
    manifest = read_json("MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json")
    rows = read_csv("V3_NOMINAL_MYOLEG_SMOKE.csv")
    assert manifest["smoke_completed_at_manifest_freeze"] is False
    assert len(rows) == 13
    assert all(row["smoke_integrity_pass"] == "True" for row in rows)
    assert all(row["model_role"] == "UNMODIFIED_NOMINAL_MYOLEG_V2" for row in rows)
    assert all(row["mechanical_objective_computed"] == "False" and row["ranking_computed"] == "False" for row in rows)
    assert max(int(row["solver_warning_count"]) for row in rows) == 0
    assert max(int(row["contact_active_count"]) for row in rows) == 0
    assert max(int(row["tendon_limit_active_count"]) for row in rows) == 0


def test_P4_remains_primary_and_P2_is_unexecuted_structural_fallback() -> None:
    rows = {row["option"]: row for row in read_csv("V3_P4_VS_P2_STRUCTURAL_COMPARISON.csv")}
    assert rows["P4_BRANCH_AWARE_COORDINATION_FUNCTION"]["role"] == "PRIMARY_IMPLEMENTED"
    assert int(rows["P4_BRANCH_AWARE_COORDINATION_FUNCTION"]["dimension"]) == 2
    assert rows["P2_INTERIOR_BSPLINE_JOINT_PERTURBATION"]["role"] == "FALLBACK_NOT_IMPLEMENTED"
    assert int(rows["P2_INTERIOR_BSPLINE_JOINT_PERTURBATION"]["dimension"]) == 4
    assert all(row["selection_used_J_or_subject_truth"] == "False" for row in rows.values())


def test_held_out_remains_sealed_and_frozen_inputs_are_unchanged() -> None:
    access = read_json("HELD_OUT_ACCESS_AUDIT.json")
    metadata = read_json("metadata.json")
    assert access["held_out_scientific_truth_access_count"] == 0
    assert access["np_load_held_out_count"] == 0
    assert metadata["development_truth_access_count"] == 0
    assert metadata["held_out_scientific_truth_access_count"] == 0
    assert metadata["mechanical_objective_evaluated"] is False
    assert metadata["full_V3_landscape_generated"] is False
    assert metadata["frozen_inputs_before"] == metadata["frozen_inputs_after"] == build_design.FROZEN_SHA
    for name, expected in build_design.FROZEN_SHA.items():
        assert sha256(build_design.frozen_paths()[name]) == expected


def test_all_formal_artifact_checksums_pass() -> None:
    lines = (OUTPUT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 19
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = OUTPUT / relative.strip()
        assert path.is_file()
        assert sha256(path) == expected


def test_final_status_stays_offline_default_off_and_next_stage_not_run() -> None:
    metadata = read_json("metadata.json")
    assert metadata["outcome"] == "MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_VALID_WITH_LIMITATIONS"
    assert metadata["status"] == ["OFFLINE_ONLY", "DEFAULT_OFF", "NOT_HUMAN_READY", "NOT_ROBOT_APPROVED"]
    assert metadata["recommended_next_stage"] == "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_GENERATION_V1"
    assert metadata["next_stage_executed"] is False

