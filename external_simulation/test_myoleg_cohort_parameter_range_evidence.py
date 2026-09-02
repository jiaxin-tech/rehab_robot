"""Artifact tests for MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_V1.

The expensive MyoLeg replay is retained as checksummed evidence.  Normal
repository pytest verifies its frozen inputs, range logic, outputs and scope
without importing MuJoCo or regenerating the cohort (which does not yet exist).
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
    / "myoleg_cohort_parameter_range_evidence_v1"
)
BUILDER = (
    ROOT
    / "external_simulation"
    / "myoleg_cohort_parameter_range_evidence_v1"
    / "build_range_evidence.py"
)
PRIOR = ROOT / "external_simulation_audits" / "myoleg_virtual_patient_cohort_design_v1"
MODEL = ROOT / "external_simulation" / "myoleg_supine_rehab_v1" / "myoleg_supine_right_v1.xml"
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
FORMAL_REFERENCE = ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
FORMAL_MANIFEST = ROOT / "config" / "formal_experiment_manifest.json"

FROZEN = {
    MODEL: "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    V2_REFERENCE: "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    TRUTH_SEMANTICS: "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    FORMAL_REFERENCE: "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    FORMAL_MANIFEST: "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    PRIOR / "PROPOSED_COHORT_SCHEMES.json": "a460befdbbfa7dc7f54078673067843467740ed6ecbf7c4cd5cee533e6269bff",
    PRIOR / "PARAMETER_TAXONOMY.csv": "0f8f3e6af995ad973bb1c941e9cc4e2efa96248ee1df85c65f44f38138bab33f",
    PRIOR / "PARAMETER_RANGE_EVIDENCE.json": "af250c583d856ba9891fb0449b4a964f9c469a3a0151ed560ce086534fec596c",
    PRIOR / "MYOLEG_PARAMETER_INVENTORY.csv": "b4eded805c353e65bb38325d64deb85bfbb3eff4c4ff127e9135c5be306ac417",
}

FACTORS = (
    "FEMUR_MASS_INERTIA_SCALE",
    "TIBIA_PATELLA_MASS_INERTIA_SCALE",
    "FOOT_COMPLEX_MASS_INERTIA_SCALE",
    "HIP_ONLY_PASSIVE_FP_MAX_SCALE",
    "KNEE_ONLY_PASSIVE_FP_MAX_SCALE",
    "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_required_artifacts_and_checksums() -> None:
    required = {
        "MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_REPORT.md",
        "PARAMETER_RANGE_LITERATURE_EVIDENCE.csv",
        "MYOLEG_MODEL_PROVENANCE.md",
        "ANTHROPOMETRY_RANGE_EVIDENCE.csv",
        "PASSIVE_PROPERTY_RANGE_EVIDENCE.csv",
        "PARAMETER_CORRELATION_AUDIT.csv",
        "PROPOSED_PARAMETER_RANGES.json",
        "RANGE_EVALUATION_MANIFEST.json",
        "RANGE_ENDPOINT_REPLAY_RESULTS.csv",
        "RANGE_INTERACTION_SMOKE_RESULTS.csv",
        "COHORT_SIZE_DESIGN_AUDIT.md",
        "COHORT_SAMPLING_DESIGN_AUDIT.md",
        "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1.json",
        "SOURCE_BIBLIOGRAPHY.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert required == {path.name for path in ARTIFACTS.iterdir() if path.is_file()}
    for line in (ARTIFACTS / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(ARTIFACTS / relative.strip()) == expected


def test_frozen_inputs_and_all_prior_design_artifacts_are_unchanged() -> None:
    for path, expected in FROZEN.items():
        assert sha256(path) == expected
    for line in (PRIOR / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(PRIOR / relative.strip()) == expected
    metadata = read_json("metadata.json")
    assert metadata["input_sha256_before"] == metadata["input_sha256_after"]
    assert metadata["prior_artifact_checksums_before"] == metadata["prior_artifact_checksums_after"]


def test_formal_rom_reference_and_truth_conventions_are_unchanged() -> None:
    manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["rom_protocol_version"] == "ROM_PROTOCOL_V2"
    assert manifest["hip_rom_deg"] == [0.0, 120.0]
    assert manifest["knee_rom_deg"] == [5.0, 145.0]
    assert manifest["theta_shank_definition"] == "q_hip - q_knee"
    assert manifest["active_reference_sha256"] == FROZEN[FORMAL_REFERENCE]
    truth = json.loads(TRUTH_SEMANTICS.read_text(encoding="utf-8"))
    assert truth["semantic_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    assert truth["truth_field"] == "TAU_MY0LEG_REQUIRED_DRIVE"
    assert truth["reduced_coordinate_equation"] == "tau_truth = T(q)^T*r"


def test_scheme_a_is_reused_exactly_and_p0_force_fpmax_semantics_are_frozen() -> None:
    prior = json.loads((PRIOR / "PROPOSED_COHORT_SCHEMES.json").read_text(encoding="utf-8"))
    scheme = next(item for item in prior["schemes"] if item["scheme_id"] == "SCHEME_A_MINIMAL_INTERPRETABLE")
    assert tuple(scheme["subject_level_factors"]) == FACTORS
    proposal = read_json("PROPOSED_PARAMETER_RANGES.json")
    assert proposal["scheme_id"] == "SCHEME_A_MINIMAL_INTERPRETABLE"
    assert tuple(item["factor_id"] for item in proposal["factors"]) == FACTORS
    assert proposal["p0_semantics"]["force_fpmax_indistinguishable"] is True
    assert proposal["p0_semantics"]["f0_factor_present"] is False
    assert not any("F0_" in factor for factor in FACTORS)
    assert proposal["range_selection_used_mechanical_objective"] is False
    assert proposal["range_selection_used_learner_or_landscape_outcome"] is False


def test_range_proposal_was_preregistered_and_has_evidence_for_every_factor() -> None:
    proposal = read_json("PROPOSED_PARAMETER_RANGES.json")
    manifest = read_json("RANGE_EVALUATION_MANIFEST.json")
    assert proposal["frozen_before_replay"] is True
    assert manifest["frozen_before_replay"] is True
    assert manifest["proposal_content_sha256"] == proposal["proposal_content_sha256"]
    assert manifest["no_range_tuning_after_results"] is True
    factors = {item["factor_id"]: item for item in proposal["factors"]}
    expected_primary = {
        FACTORS[0]: [0.88, 1.0, 1.12],
        FACTORS[1]: [0.87, 1.0, 1.13],
        FACTORS[2]: [0.82, 1.0, 1.18],
        FACTORS[3]: [0.95, 1.0, 1.05],
        FACTORS[4]: [0.95, 1.0, 1.05],
        FACTORS[5]: [0.95, 1.0, 1.05],
    }
    assert {factor: value["conservative"] for factor, value in factors.items()} == expected_primary
    assert all(value["sources"] and value["mapping_assumptions"] for value in factors.values())
    assert {factors[factor]["evidence_class"] for factor in FACTORS[:3]} == {"E2"}
    assert {factors[factor]["evidence_class"] for factor in FACTORS[3:]} == {"E4"}


def test_anthropometry_is_mass_anchored_and_inertia_is_only_an_approximation() -> None:
    rows = read_csv("ANTHROPOMETRY_RANGE_EVIDENCE.csv")
    assert len(rows) == 3
    assert {row["factor_id"] for row in rows} == set(FACTORS[:3])
    assert all(float(row["max_sex_mass_cv_pct"]) > 0.0 for row in rows)
    assert all(row["evidence_class"] == "E2" for row in rows)
    assert all(row["inertia_mapping_status"] == "INERTIA_SCALING_IS_MODELING_APPROXIMATION" for row in rows)
    report = (ARTIFACTS / "MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_REPORT.md").read_text(encoding="utf-8")
    assert "one\nscalar cannot represent all of them" in report


def test_fpmax_semantics_and_group_interpretation_are_not_overclaimed() -> None:
    rows = read_csv("PASSIVE_PROPERTY_RANGE_EVIDENCE.csv")
    assert len(rows) == 3
    assert {row["factor_id"] for row in rows} == set(FACTORS[3:])
    assert all(row["range_evidence_class"] == "E4" for row in rows)
    assert all(row["population_mapping_status"] == "NO_RELIABLE_POPULATION_TO_FPMAX_MAPPING" for row in rows)
    assert all(row["group_specific_range_supported"] == "NO_USE_COMMON_BASE_INTERVAL" for row in rows)
    biarticular = next(row for row in rows if row["frozen_structural_group"] == "HIP_KNEE_BIARTICULAR")
    assert set(biarticular["target_actuators"].split(";")) == {
        "bflh_r",
        "grac_r",
        "recfem_r",
        "sart_r",
        "semimem_r",
        "semiten_r",
        "tfl_r",
    }
    provenance = (ARTIFACTS / "MYOLEG_MODEL_PROVENANCE.md").read_text(encoding="utf-8")
    assert "close reference" in provenance
    assert "does not establish a one-to-one" in provenance
    assert "not** a directly measured human passive" in provenance


def test_literature_table_covers_the_full_evidence_hierarchy_and_traceability() -> None:
    rows = read_csv("PARAMETER_RANGE_LITERATURE_EVIDENCE.csv")
    assert {row["evidence_class"] for row in rows} == {"E1", "E2", "E3", "E4"}
    assert all(row["source"] and row["population_or_model"] for row in rows)
    assert all(row["reported_quantity"] and row["mapping_to_myoleg"] and row["limitation"] for row in rows)
    bibliography = read_json("SOURCE_BIBLIOGRAPHY.json")
    used_ids = {row["source_id"] for row in rows if row["source_id"] != "PRIOR_FROZEN_SMOKE"}
    assert used_ids <= set(bibliography)


def test_correlations_are_audited_without_inventing_covariance() -> None:
    rows = read_csv("PARAMETER_CORRELATION_AUDIT.csv")
    assert {row["classification"] for row in rows} == {
        "KNOWN_CORRELATED",
        "POSSIBLY_CORRELATED",
        "NO_USEFUL_EVIDENCE",
    }
    scheme_rows = [row for row in rows if row["scope"] == "SCHEME_A"]
    assert len(scheme_rows) == 15
    protocol = read_json("MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1.json")
    dependency = protocol["dependency_and_correlation"]
    assert dependency["quantitative_covariance_frozen"] is False
    assert dependency["marginal_independence_is_population_claim"] is False
    assert dependency["known_structural_overlap_excluded"] == "GLOBAL_RIGHT_TARGET_FP_MAX_SCALE"


def test_all_conservative_and_extended_endpoint_replays_pass() -> None:
    rows = read_csv("RANGE_ENDPOINT_REPLAY_RESULTS.csv")
    assert len(rows) == 30
    assert {row["factor_id"] for row in rows} == set(FACTORS)
    assert {row["endpoint"] for row in rows} == {
        "EXTENDED_LOWER",
        "CONSERVATIVE_LOWER",
        "NOMINAL",
        "CONSERVATIVE_UPPER",
        "EXTENDED_UPPER",
    }
    for row in rows:
        assert row["reference_sha256"] == FROZEN[V2_REFERENCE]
        assert row["duration_s"] == "24.0"
        assert row["sample_count"] == "401"
        assert row["truth_semantics_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
        assert row["truth_field"] == "TAU_MY0LEG_REQUIRED_DRIVE"
        assert row["warning_count"] == "0"
        assert row["all_state_finite"] == "True"
        assert row["muscle_state_all_finite"] == "True"
        assert row["tendon_state_all_finite"] == "True"
        assert row["all_integrity_gates_pass"] == "True"
        assert row["abnormal_force_concentration_observed"] == "False"
        assert float(row["source_equality_residual_max"]) <= 1.0e-3
        assert float(row["algebraic_residual_max_nm"]) <= 1.0e-8
        assert float(row["tracking_q_max_abs_deg"]) <= 1.0
        assert float(row["peak_force_ratio_vs_nominal"]) <= 2.0


def test_all_predeclared_interaction_corners_pass_and_duplicate_is_explicit() -> None:
    rows = read_csv("RANGE_INTERACTION_SMOKE_RESULTS.csv")
    assert len(rows) == 5
    assert {row["profile_id"] for row in rows} == {
        "ALL_CONSERVATIVE_LOW",
        "ALL_CONSERVATIVE_HIGH",
        "HIGH_MASS_HIGH_PASSIVE",
        "LOW_MASS_HIGH_PASSIVE",
        "REPRESENTATIVE_MIXED",
    }
    assert all(row["all_integrity_gates_pass"] == "True" for row in rows)
    duplicate = next(row for row in rows if row["profile_id"] == "HIGH_MASS_HIGH_PASSIVE")
    high = next(row for row in rows if row["profile_id"] == "ALL_CONSERVATIVE_HIGH")
    assert duplicate["duplicate_parameterization_of"] == "ALL_CONSERVATIVE_HIGH"
    assert duplicate["factor_scales_json"] == high["factor_scales_json"]
    assert duplicate["replay_sha256"] == high["replay_sha256"]


def test_generation_protocol_is_frozen_but_default_off_and_contains_no_subjects() -> None:
    protocol = read_json("MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1.json")
    assert protocol["status"] == "FROZEN_PROTOCOL_ONLY_DEFAULT_OFF"
    assert protocol["execution_authorized"] is False
    assert protocol["cohort_generated"] is False
    assert protocol["factor_order"] == list(FACTORS)
    sampling = protocol["sampling"]
    assert sampling["heterogeneous_subject_count"] == 32
    assert sampling["seed"] == 20260830
    assert len(sampling["development_indices_zero_based"]) == 24
    assert len(sampling["held_out_indices_zero_based"]) == 8
    assert set(sampling["development_indices_zero_based"]).isdisjoint(sampling["held_out_indices_zero_based"])
    assert sampling["split_frozen_before_learner_performance_reveal"] is True
    assert "subjects" not in protocol
    assert "representative patient cohort" in protocol["claim_boundary"]


def test_final_status_and_forbidden_operations_remain_false() -> None:
    metadata = read_json("metadata.json")
    assert metadata["outcome"] == "MYOLEG_COHORT_RANGES_READY_WITH_SYNTHETIC_LIMITATIONS"
    assert metadata["scheme_frozen"] is True
    assert metadata["range_proposal_frozen_before_replay"] is True
    assert metadata["all_endpoint_integrity_pass"] is True
    assert metadata["all_interaction_integrity_pass"] is True
    assert metadata["fpmax_population_mapping"] == "NOT_JUSTIFIED"
    assert metadata["mass_inertia_mapping"] == "INERTIA_SCALING_IS_MODELING_APPROXIMATION"
    false_flags = {
        "cohort_generated",
        "candidate_landscape_generated",
        "five_parameter_fit",
        "nn_trained",
        "pinn_trained",
        "bo_run",
        "robot_connected",
        "hardware_accessed",
        "control_modified",
        "safety_modified",
        "formal_reference_modified",
        "v2_reference_modified",
        "rom_protocol_modified",
        "truth_semantics_modified",
        "next_stage_executed",
    }
    assert all(metadata[key] is False for key in false_flags)
    assert metadata["subject_instances_generated"] == 0
    assert not list(ARTIFACTS.glob("*.xml"))
    assert not list(ARTIFACTS.glob("*.npz"))


def test_builder_has_no_learner_optimizer_or_robot_dependency() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imports: set[str] = set()
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
