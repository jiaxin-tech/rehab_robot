"""Artifact tests for MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1.

Normal repository pytest intentionally does not import MuJoCo.  The frozen
MyoSuite runtime builder performs the expensive numerical smoke checks; these
tests verify retained evidence, provenance, scope and scientific boundaries.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_virtual_patient_cohort_design_v1"
)
BUILDER = (
    ROOT
    / "external_simulation"
    / "myoleg_virtual_patient_cohort_design_v1"
    / "build_design_audit.py"
)
MODEL = (
    ROOT
    / "external_simulation"
    / "myoleg_supine_rehab_v1"
    / "myoleg_supine_right_v1.xml"
)
V2_REFERENCE = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
)
TRUTH_SEMANTICS = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_reference_trajectory_replay_v1"
    / "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
)
FORMAL_REFERENCE = (
    ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
FORMAL_MANIFEST = ROOT / "config" / "formal_experiment_manifest.json"

FROZEN = {
    MODEL: "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    V2_REFERENCE: "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    TRUTH_SEMANTICS: "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    FORMAL_REFERENCE: "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    FORMAL_MANIFEST: "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
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
        "MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_REPORT.md",
        "MYOLEG_PARAMETER_INVENTORY.csv",
        "PARAMETER_TAXONOMY.csv",
        "ANTHROPOMETRY_VARIATION_AUDIT.md",
        "MUSCLE_STRENGTH_VARIATION_AUDIT.md",
        "PASSIVE_PROPERTY_VARIATION_AUDIT.md",
        "BIARTICULAR_VARIATION_AUDIT.md",
        "LOW_ACTIVATION_VARIABILITY_AUDIT.md",
        "PARAMETER_RANGE_EVIDENCE.json",
        "SINGLE_PARAMETER_SENSITIVITY_RESULTS.csv",
        "PROPOSED_COHORT_SCHEMES.json",
        "PROPOSED_SUBJECT_MANIFEST_SCHEMA.json",
        "COHORT_INTEGRITY_GATES.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert required == {path.name for path in ARTIFACTS.iterdir() if path.is_file()}
    for line in (ARTIFACTS / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(ARTIFACTS / relative.strip()) == expected


def test_all_previous_myoleg_checksum_manifests_still_validate() -> None:
    prior_stages = (
        "myoleg_install_and_smoke_test_v1",
        "myoleg_supine_hip_knee_rehab_feasibility_v1",
        "myoleg_knee_rom_compatibility_audit_v1",
        "myoleg_reference_trajectory_replay_v1",
    )
    for stage in prior_stages:
        directory = ROOT / "external_simulation_audits" / stage
        manifest = directory / "checksums.sha256"
        assert manifest.is_file(), stage
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            assert sha256(directory / relative.strip()) == expected, (stage, relative)


def test_frozen_model_reference_truth_and_formal_protocol_are_unchanged() -> None:
    for path, expected in FROZEN.items():
        assert sha256(path) == expected
    semantic = json.loads(TRUTH_SEMANTICS.read_text(encoding="utf-8"))
    assert semantic["semantic_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    assert semantic["truth_field"] == "TAU_MY0LEG_REQUIRED_DRIVE"
    assert semantic["reduced_coordinate_equation"] == "tau_truth = T(q)^T*r"
    manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["rom_protocol_version"] == "ROM_PROTOCOL_V2"
    assert manifest["hip_rom_deg"] == [0.0, 120.0]
    assert manifest["knee_rom_deg"] == [5.0, 145.0]
    assert manifest["theta_shank_definition"] == "q_hip - q_knee"


def test_v2_primary_reference_identity_and_subject_specific_normalization() -> None:
    metadata = read_json("metadata.json")
    reference = metadata["myoleg_v2_reference"]
    assert reference["id"] == "NATIVE_ROM_REFERENCE_CANDIDATE"
    assert reference["sha256"] == FROZEN[V2_REFERENCE]
    assert reference["duration_s"] == 24.0
    assert reference["sample_count"] == 401
    assert reference["knee_max_deg"] == 119.5
    assert reference["regenerated"] is False
    assert reference["transformation_identity"]["pointwise_clipping_used"] is False
    truth = metadata["truth_semantics"]
    assert truth["version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    assert truth["field"] == "TAU_MY0LEG_REQUIRED_DRIVE"
    assert truth["subject_specific_reference_normalization"] is True
    assert truth["nominal_denominator_for_all_subjects"] is False


def test_actual_model_inventory_and_exclusive_taxonomy_are_complete() -> None:
    inventory = read_csv("MYOLEG_PARAMETER_INVENTORY.csv")
    metadata = read_json("metadata.json")
    counts = metadata["model_inventory_counts"]
    assert counts == {
        "bodies": 16,
        "equalities": 27,
        "geoms": 89,
        "inventory_rows": len(inventory),
        "joints": 29,
        "muscle_actuators": 80,
        "sites": 384,
        "tendons": 80,
    }
    muscle_names = {
        row["object_name"]
        for row in inventory
        if row["category"] == "MUSCLE_ACTUATOR"
    }
    assert len(muscle_names) == 80
    assert all(
        {"force", "fpmax", "range", "actuator_lengthrange", "tendon_transmission"}
        <= {row["field"] for row in inventory if row["object_name"] == muscle}
        for muscle in muscle_names
    )
    taxonomy = read_csv("PARAMETER_TAXONOMY.csv")
    family_ids = [row["family_id"] for row in taxonomy]
    assert len(family_ids) == len(set(family_ids))
    assert set(row["classification"] for row in taxonomy) == set("ABCDE")
    assert all(row["classification"] in set("ABCDE") for row in taxonomy)


def test_biarticular_membership_is_structural_not_name_guessed() -> None:
    inventory = read_csv("MYOLEG_PARAMETER_INVENTORY.csv")
    force_rows = [
        row
        for row in inventory
        if row["category"] == "MUSCLE_ACTUATOR"
        and row["field"] == "force"
        and row["target_leg"] == "True"
        and row["structural_group"] == "HIP_KNEE_BIARTICULAR"
    ]
    names = {row["object_name"] for row in force_rows}
    assert names == {
        "bflh_r",
        "grac_r",
        "recfem_r",
        "sart_r",
        "semimem_r",
        "semiten_r",
        "tfl_r",
    }
    assert all(row["spans_hip"] == row["spans_knee"] == "True" for row in force_rows)
    assert all(float(row["hip_moment_arm_max_abs_m"]) > 1e-7 for row in force_rows)
    assert all(float(row["knee_moment_arm_max_abs_m"]) > 1e-7 for row in force_rows)
    builder_source = BUILDER.read_text(encoding="utf-8")
    assert "compiled tendon transmission moment matrix over all 401 frozen V2 states" in builder_source


def test_small_perturbation_smoke_is_deterministic_and_not_a_scientific_range() -> None:
    rows = read_csv("SINGLE_PARAMETER_SENSITIVITY_RESULTS.csv")
    assert len(rows) == 15
    assert {row["perturbation"] for row in rows} == {
        "SMALL_NEGATIVE",
        "NOMINAL",
        "SMALL_POSITIVE",
    }
    assert {row["family_id"] for row in rows} == {
        "SEGMENT_MASS_INERTIA_COUPLED_SCALE",
        "MUSCLE_FORCE_CAPACITY_SCALE",
        "MUSCLE_PASSIVE_FP_MAX_SCALE",
        "BIARTICULAR_FORCE_CAPACITY_SCALE",
        "BIARTICULAR_PASSIVE_FP_MAX_SCALE",
    }
    for row in rows:
        assert row["smoke_value_is_scientific_range"] == "False"
        assert row["duration_s"] == "24.0"
        assert row["sample_count"] == "401"
        assert row["warning_count"] == "0"
        assert row["all_state_finite"] == "True"
        assert row["muscle_force_all_finite"] == "True"
        assert row["tendon_state_all_finite"] == "True"
        assert row["deterministic"] == "True"
        assert row["determinism_first_sha256"] == row["determinism_repeat_sha256"]
        assert row["all_integrity_gates_pass"] == "True"
        assert float(row["source_equality_residual_max"]) <= 1e-3
        assert float(row["algebraic_residual_max_nm"]) <= 1e-8
        assert float(row["controlled_knee_max_deg"]) <= 120.0
        assert float(row["peak_force_ratio_vs_nominal"]) <= 2.0


def test_nominal_replay_invariants_and_p0_force_fpmax_confounding_are_explicit() -> None:
    metadata = read_json("metadata.json")
    assert metadata["nominal_prior_replay_exact_match"] is True
    equivalence = metadata["smoke_test"]["p0_force_fpmax_equivalence"]
    checks = {
        key: value
        for key, value in equivalence.items()
        if key.endswith("_within_1e_12_nm")
    }
    assert len(checks) == 4
    assert all(checks.values())
    assert equivalence["global_negative_max_abs_difference_nm"] <= 1e-12
    assert equivalence["global_positive_max_abs_difference_nm"] <= 1e-12
    assert equivalence["biarticular_negative_max_abs_difference_nm"] <= 1e-12
    assert equivalence["biarticular_positive_max_abs_difference_nm"] <= 1e-12
    schemes = read_json("PROPOSED_COHORT_SCHEMES.json")
    assert schemes["final_scheme_frozen"] is False
    assert schemes["all_ranges_frozen"] is False
    for scheme in schemes["schemes"][:2]:
        factors = scheme["subject_level_factors"]
        assert not any("MUSCLE_FORCE" in factor for factor in factors)
        assert any("FP_MAX" in factor for factor in factors)


def test_ranges_remain_evidence_gaps_and_no_cohort_scheme_is_frozen() -> None:
    evidence = read_json("PARAMETER_RANGE_EVIDENCE.json")
    assert evidence["evidence_policy"]["rule"] == "No scientific cohort range is frozen by this stage."
    for family in evidence["families"]:
        assert family["status"] == "RANGE_REQUIRES_EXTERNAL_EVIDENCE"
        if family.get("smoke_values"):
            assert family["smoke_values_are_scientific_range"] is False
    metadata = read_json("metadata.json")
    assert metadata["outcome"] == "MYOLEG_COHORT_DESIGN_READY_WITH_EVIDENCE_GAPS"


def test_manifest_is_schema_only_and_freezes_truth_before_reveal() -> None:
    schema = read_json("PROPOSED_SUBJECT_MANIFEST_SCHEMA.json")
    assert schema["$id"] == "MYOLEG_VIRTUAL_SUBJECT_MANIFEST_SCHEMA_V1"
    assert schema["description"] == "Schema only. This design stage creates no subject instances."
    properties = schema["properties"]
    assert properties["base_myoleg_model_sha256"]["const"] == FROZEN[MODEL]
    assert properties["v2_reference_sha256"]["const"] == FROZEN[V2_REFERENCE]
    assert properties["truth_semantic_version"]["const"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    assert properties["truth_field"]["const"] == "TAU_MY0LEG_REQUIRED_DRIVE"
    assert properties["subject_specific_reference_normalization"]["const"] is True
    assert properties["frozen_before_truth_reveal"]["const"] is True
    assert "subjects" not in schema


def test_truth_learner_independence_and_forbidden_operations() -> None:
    metadata = read_json("metadata.json")
    assert metadata["truth_learner_parameterization_independence"] == "PASS"
    false_flags = {
        "cohort_generated",
        "candidate_landscape_generated",
        "five_parameter_fit",
        "nn_trained",
        "pinn_trained",
        "bo_run",
        "robot_connected",
        "hardware_accessed",
        "formal_reference_modified",
        "rom_protocol_modified",
        "v2_reference_modified",
        "truth_semantics_modified",
        "next_stage_executed",
    }
    assert all(metadata[key] is False for key in false_flags)
    assert not list(ARTIFACTS.glob("*.xml"))
    assert not list(ARTIFACTS.glob("*.npz"))
    assert not any("subject_" in path.name.lower() and path.name != "PROPOSED_SUBJECT_MANIFEST_SCHEMA.json" for path in ARTIFACTS.iterdir())


def test_builder_has_no_project_learner_or_robot_dependency() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint(
        {
            "lower_limb_sim",
            "hardware",
            "control",
            "collection",
            "safety",
            "torch",
            "tensorflow",
            "sklearn",
            "botorch",
        }
    )
    metadata = read_json("metadata.json")
    assert metadata["builder_script_sha256"] == sha256(BUILDER)

