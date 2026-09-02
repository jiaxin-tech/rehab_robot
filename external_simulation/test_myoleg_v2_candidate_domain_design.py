"""Frozen-evidence tests for MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_V1."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "external_simulation_audits" / "myoleg_v2_candidate_domain_design_v1"
BUILDER = ROOT / "external_simulation" / "myoleg_v2_candidate_domain_design_v1" / "build_candidate_domain.py"
COHORT_AUDIT = ROOT / "external_simulation_audits" / "myoleg_virtual_patient_cohort_generation_v1"
COHORT_MANIFEST = COHORT_AUDIT / "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
SAMPLING_MANIFEST = COHORT_AUDIT / "SAMPLING_FREEZE_MANIFEST.json"
V2_REFERENCE = ROOT / "external_simulation_audits" / "myoleg_knee_rom_compatibility_audit_v1" / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
FORMAL_REFERENCE = ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
FORMAL_MANIFEST = ROOT / "config" / "formal_experiment_manifest.json"
TRUTH_SEMANTICS = ROOT / "external_simulation_audits" / "myoleg_reference_trajectory_replay_v1" / "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
BASE_MODEL = ROOT / "external_simulation" / "myoleg_supine_rehab_v1" / "myoleg_supine_right_v1.xml"

FROZEN = {
    COHORT_MANIFEST: "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    SAMPLING_MANIFEST: "81451f87e817062e5e56cc1de13d2a71a148989db06454514818fa268300fecb",
    V2_REFERENCE: "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    FORMAL_REFERENCE: "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    FORMAL_MANIFEST: "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    TRUTH_SEMANTICS: "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    BASE_MODEL: "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
}
DETERMINISTIC = {
    "CANDIDATE_DOMAIN_PROTOCOL.json": "e5e973c84c59909f3307591b87b0597fb826a9b96f98118e1e306c224a38a40e",
    "ORIGINAL_PROPOSAL_GRID.csv": "4455fdb2a95257fc8e7eb76db8338b8e244238d3b9e1b1be6e7423ace5e8b6da",
    "V2_CANDIDATE_ADMISSION.csv": "13ff61daf55560953d1b1ff7a590af7f132b2f88d880dad0202cd3fb79e017b2",
    "SPARSE_MYOLEG_SMOKE_SET.csv": "729a799db359eacb4282170e3c20678fd9677a1e805381aac07b07943b996356",
    "TRUSTED_ROM_DECISION.json": "0383984f3b49835dd014d8de13f54d69d258fe8099a02d46dd398cd045a6e902",
    "MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_required_artifacts_and_checksums() -> None:
    required = {
        "MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_REPORT.md",
        "CANDIDATE_DOMAIN_PROTOCOL.json",
        "MYOLEG_BOUNDARY_ARTIFACT_AUDIT.csv",
        "TRUSTED_ROM_DECISION.json",
        "ORIGINAL_PROPOSAL_GRID.csv",
        "V2_CANDIDATE_ADMISSION.csv",
        "V2_CANDIDATE_EXCLUSION_SUMMARY.json",
        "PHASE_WARP_INTEGRITY.csv",
        "REFERENCE_NEIGHBORHOOD_AUDIT.json",
        "SPARSE_MYOLEG_SMOKE_SET.csv",
        "SPARSE_MYOLEG_SMOKE_RESULTS.csv",
        "MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json",
        "RUNTIME_AND_STORAGE_ESTIMATE.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert required == {path.name for path in ARTIFACTS.iterdir() if path.is_file()}
    for line in (ARTIFACTS / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(ARTIFACTS / relative.strip()) == expected


def test_frozen_cohort_lhs_reference_model_and_truth_are_unchanged() -> None:
    for path, expected in FROZEN.items():
        assert sha256(path) == expected, path
    for directory in (COHORT_AUDIT, ROOT / "external_simulation" / "cohorts" / "myoleg_virtual_patient_cohort_v1"):
        for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            assert sha256(directory / relative.strip()) == expected
    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    assert formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
    assert formal["theta_shank_definition"] == "q_hip - q_knee"


def test_protocol_was_frozen_before_results_with_engineering_artifact_gates() -> None:
    protocol = read_json("CANDIDATE_DOMAIN_PROTOCOL.json")
    assert protocol["frozen_before_new_boundary_or_smoke_results"] is True
    assert protocol["source_identities"]["cohort_manifest"] == FROZEN[COHORT_MANIFEST]
    thresholds = protocol["boundary_diagnostic"]["thresholds"]
    assert thresholds == {
        "absolute_limit_contribution_max_nm": 0.005,
        "contact_constraint_count": 0,
        "relative_limit_contribution_max": 0.0005,
        "solver_warning_count": 0,
        "source_equality_residual_max": 0.001,
        "classification": "SIMULATOR_ARTIFACT_GATE_NOT_HUMAN_SAFETY",
    }
    assert protocol["scope_guards"]["human_safety_claim"] is False
    assert protocol["sparse_smoke"]["candidate_count"] == 30


def test_original_proposal_grid_is_exact_21025_and_ids_are_stable() -> None:
    rows = read_csv("ORIGINAL_PROPOSAL_GRID.csv")
    assert len(rows) == 29 * 29 * 25 == 21025
    assert [int(row["proposal_index"]) for row in rows] == list(range(21025))
    assert all(row["candidate_id"] == f"MYOLEG_V2_P{index:05d}" for index, row in enumerate(rows))
    assert sorted({float(row["delta_hip_amp_deg"]) for row in rows}) == [round(-5.0 + 0.25 * i, 12) for i in range(29)]
    assert sorted({float(row["delta_knee_amp_deg"]) for row in rows}) == [round(-5.0 + 0.25 * i, 12) for i in range(29)]
    assert sorted({float(row["knee_phase_shift"]) for row in rows}) == [round(-0.03 + 0.0025 * i, 12) for i in range(25)]


def test_admission_is_geometry_only_and_all_included_candidates_meet_trusted_domain() -> None:
    rows = read_csv("V2_CANDIDATE_ADMISSION.csv")
    included = [row for row in rows if row["included"] == "True"]
    excluded = [row for row in rows if row["included"] == "False"]
    assert len(rows) == 21025
    assert len(included) == 16675
    assert len(excluded) == 4350
    assert {row["exclusion_reasons"] for row in excluded} == {"myoleg_knee_upper_trusted_bound"}
    assert all(float(row["q_knee_max_deg"]) <= 120.0 + 1.0e-12 for row in included)
    assert all(float(row["q_knee_min_deg"]) >= 18.3208 for row in included)
    assert all(-30.0001 <= float(row["q_hip_min_deg"]) for row in included)
    assert all(float(row["q_hip_max_deg"]) <= 120.0003 for row in included)
    assert all(float(row["minimum_abs_jacobian_determinant"]) >= 1.0e-4 for row in included)
    assert all(float(row["maximum_jacobian_condition"]) <= 100.0 for row in included)
    forbidden_fragments = ("j_pred", "truth_j", "objective", "rank", "model_coverage", "acquisition")
    assert not any(fragment in column.lower() for column in rows[0] for fragment in forbidden_fragments)


def test_reference_and_all_immediate_neighbors_are_retained() -> None:
    audit = read_json("REFERENCE_NEIGHBORHOOD_AUDIT.json")
    assert audit["reference"] == {
        "candidate_id": "MYOLEG_V2_P15012",
        "exact_v2_reference": True,
        "included": True,
        "proposal_index": 15012,
    }
    assert all(row["included"] for row in audit["immediate_neighbors"])
    assert {row["direction"] for row in audit["immediate_neighbors"]} == {
        "HIP_NEGATIVE", "HIP_POSITIVE", "KNEE_NEGATIVE", "KNEE_POSITIVE", "PHASE_NEGATIVE", "PHASE_POSITIVE"
    }
    assert audit["positive_knee_amplitude_available"] is True
    admission = read_csv("V2_CANDIDATE_ADMISSION.csv")
    positive_levels = sorted({float(row["delta_knee_amp_deg"]) for row in admission if row["included"] == "True" and float(row["delta_knee_amp_deg"]) > 0.0})
    assert positive_levels == [0.25, 0.5]


def test_phase_warp_and_every_candidate_closure_integrity_pass() -> None:
    phase = read_csv("PHASE_WARP_INTEGRITY.csv")
    assert len(phase) == 25
    assert all(row["integrity_pass"] == "True" for row in phase)
    assert all(float(row["minimum_warp_derivative"]) > 0.0 for row in phase)
    admission = read_csv("V2_CANDIDATE_ADMISSION.csv")
    assert all(float(row["joint_closure_error_rad"]) <= 1.0e-10 for row in admission)
    assert all(float(row["velocity_closure_error_rad_s"]) <= 1.0e-10 for row in admission)
    assert all(float(row["acceleration_closure_error_rad_s2"]) <= 1.0e-9 for row in admission)
    assert all(row["phase_warp_monotonic"] == "True" for row in admission)


def test_boundary_audit_covers_all_32_subjects_plus_nominal_and_reports_lower_caveat() -> None:
    rows = read_csv("MYOLEG_BOUNDARY_ARTIFACT_AUDIT.csv")
    assert len(rows) == 33 * (9 + 6) * 4 * 3
    assert len({row["subject_id"] for row in rows}) == 33
    upper = [row for row in rows if row["boundary_side"] == "UPPER"]
    assert all(row["artifact_gate_pass"] == "True" for row in upper)
    decision = read_json("TRUSTED_ROM_DECISION.json")
    assert decision["trusted_knee_upper_deg"] == 120.0
    assert decision["candidate_observed_knee_domain_deg"][0] > 18.32
    assert decision["candidate_observed_knee_domain_deg"][1] < 120.0
    assert decision["lower_angle_all_model_pass"]["0.1"] is False
    assert decision["lower_angle_all_model_pass"]["0.25"] is False
    assert "does not affect admission" not in decision["lower_boundary_interpretation"]
    assert "no proposal approaches" in decision["lower_boundary_interpretation"]


def test_sparse_smoke_is_deterministic_global_and_all_cases_pass() -> None:
    smoke_set = read_csv("SPARSE_MYOLEG_SMOKE_SET.csv")
    results = read_csv("SPARSE_MYOLEG_SMOKE_RESULTS.csv")
    assert len(smoke_set) == 30
    assert len({row["proposal_index"] for row in smoke_set}) == 30
    assert len(results) == 30 * 7 == 210
    assert all(row["smoke_integrity_pass"] == "True" for row in results)
    assert all(row["truth_semantic_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1" for row in results)
    assert all(row["truth_field"] == "TAU_MY0LEG_REQUIRED_DRIVE" for row in results)
    assert all(float(row["absolute_joint_limit_knee_contribution_nm"]) <= 0.005 for row in results)
    assert all(float(row["relative_joint_limit_contribution"]) <= 0.0005 for row in results)
    assert {row["split"] for row in results} == {"NOMINAL_CONTROL", "DEVELOPMENT", "HELD_OUT"}


def test_manifest_is_deterministic_and_freezes_one_global_candidate_set() -> None:
    for name, expected in DETERMINISTIC.items():
        assert sha256(ARTIFACTS / name) == expected, name
    manifest = read_json("MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json")
    assert manifest["source_proposal_count"] == 21025
    assert manifest["admissible_candidate_count"] == 16675
    assert len(manifest["ordered_included_candidates"]) == 16675
    assert manifest["one_global_candidate_set_for_all_32_subjects"] is True
    assert manifest["subject_specific_candidate_deletion"] is False
    assert manifest["sparse_smoke_status"] == "PASS"
    assert "runtime_s" not in manifest
    assert "generated_at" not in manifest


def test_scope_excludes_landscape_learners_bo_and_robot_code() -> None:
    metadata = read_json("metadata.json")
    manifest = read_json("MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json")
    for key in ("full_truth_landscape_generated", "five_parameter_fit", "nn_or_pinn", "bo", "robot_or_hardware"):
        assert metadata[key] is False
        assert manifest[key] is False
    assert not any("landscape" in path.name.lower() for path in ARTIFACTS.iterdir())
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint({"hardware", "control", "torch", "tensorflow", "sklearn", "gpytorch", "pymc"})

