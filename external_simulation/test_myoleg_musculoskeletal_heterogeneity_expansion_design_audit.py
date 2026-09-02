"""Regression gates for the offline MyoLeg heterogeneity expansion audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from external_simulation.myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1 import build_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_protocol_is_prefrozen_and_heldout_stays_sealed() -> None:
    protocol = read_json("MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_PROTOCOL.json")
    heldout = read_json("HELD_OUT_ACCESS_AUDIT.json")
    assert sha256(audit.PROTOCOL) == audit.FROZEN_SHA["protocol"]
    assert protocol["frozen_before_structural_model_inventory_or_local_sensitivity_results"] is True
    assert protocol["tiny_smoke_test_policy"]["maximum_factor_probes"] == 4
    assert protocol["tiny_smoke_test_policy"]["trajectory_candidate_ids"] == list(audit.TRAJECTORIES)
    assert heldout["held_out_scientific_access_count"] == 0
    assert heldout["held_out_np_load_count"] == 0
    assert heldout["held_out_replay_count"] == 0
    assert heldout["status"] == "SEALED_AND_FAIL_CLOSED"


def test_all_frozen_scientific_inputs_are_unchanged() -> None:
    assert {name: sha256(path) for name, path in audit.FROZEN_PATHS.items()} == audit.FROZEN_SHA
    metadata = read_json("metadata.json")
    assert metadata["objective_or_normalization_modified"] is False
    assert metadata["v3_parameterization_or_domain_modified"] is False
    assert metadata["fpmax_range_expanded"] is False
    assert metadata["new_subjects_generated"] == 0
    assert metadata["new_truth_landscape_generated"] is False
    assert metadata["analysis_code_sha256"] == sha256(Path(audit.__file__))


def test_inventory_uses_actual_fields_and_protects_lengthrange() -> None:
    inventory = {row["parameter_id"]: row for row in read_csv("MYOLEG_STRUCTURAL_PARAMETER_INVENTORY.csv")}
    assert {"A01", "A02", "A03", "A04", "A05", "B01", "B02", "C01", "C02", "D01", "D02", "D03", "E01", "E02", "F01"} <= set(inventory)
    assert inventory["A05"]["compiled_field"] == "actuator_lengthrange"
    assert "protected" in inventory["A05"]["audit_decision"]
    assert "stiffness=[0.0]" in inventory["E01"]["nominal_value"]
    assert "14 knee/patella" in inventory["F01"]["nominal_value"]
    assert all(row["source_model_path"] == "external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml" for row in inventory.values())


def test_biarticular_and_geometry_audits_use_real_model_objects() -> None:
    coupling = read_csv("BIARTICULAR_COUPLING_PARAMETER_AUDIT.csv")
    geometry = read_csv("MOMENT_ARM_GEOMETRY_AUDIT.csv")
    assert {row["muscle"] for row in coupling} == set(audit.BIARTICULAR)
    assert {row["muscle"] for row in geometry} == set(audit.BIARTICULAR)
    assert all(row["tendon"].endswith("_tendon") for row in coupling)
    assert all(int(row["tendon_path_element_count"]) >= 2 for row in coupling)
    assert all(row["artificial_coupling_coefficient_needed"] == "False" for row in coupling)
    assert all(row["consistency_class"] == "REQUIRES_REBUILD/CALIBRATION" for row in geometry)
    assert all(row["single_site_move_allowed"] == "False" for row in geometry)


def test_local_sensitivity_is_tiny_nominal_only_and_non_oracle() -> None:
    rows = read_csv("STRUCTURAL_LOCAL_SENSITIVITY_RESULTS.csv")
    assert len(rows) == 4
    run = [row for row in rows if row["status"] == "RUN"]
    skipped = [row for row in rows if row["status"] == "NOT_RUN_WITH_REASON"]
    assert len(run) == 3 and len(skipped) == 1
    assert all(row["scientific_role"] == "LOCAL_NUMERICAL_SENSITIVITY_ONLY" for row in rows)
    assert all(row["population_range_inferred"] == "False" for row in rows)
    assert all(int(row["trajectory_count"]) == 9 for row in run)
    assert all(int(row["warning_count_max"]) == 0 for row in run)
    assert all(float(row["decomposition_residual_max_nm"]) < 2e-13 for row in run)
    assert skipped[0]["probe_id"] == "P4_TENDON_ELASTICITY"
    metadata = read_json("metadata.json")
    assert metadata["local_sensitivity_uses_oracle_or_personalization_outcome"] is False
    assert metadata["local_sensitivity_population_bounds"] is False


def test_taxonomy_and_schemes_are_low_dimensional_and_truth_learner_independent() -> None:
    taxonomy = {row["factor"]: row for row in read_csv("HETEROGENEITY_FACTOR_TAXONOMY.csv")}
    assert taxonomy["background activation"]["taxonomy"] == "C"
    assert taxonomy["single-site attachment move"]["taxonomy"] == "E"
    assert taxonomy["actuator lengthrange"]["taxonomy"] == "E"
    assert taxonomy["joint damping"]["truth_learner_parameterization_independence"] == "overlaps Bhip/Bknee"
    schemes = read_json("PROPOSED_STRUCTURAL_HETEROGENEITY_SCHEMES.json")
    assert schemes["selection_uses_personalization_outcome"] is False
    assert schemes["population_bounds_frozen"] is False
    assert schemes["new_subjects_generated"] is False
    assert [row["dimensionality"] for row in schemes["schemes"]] == [4, 7, 6]
    assert schemes["schemes"][0]["range_status"] == "RANGE_REQUIRES_EXTERNAL_EVIDENCE"


def test_versioning_requires_v2_and_does_not_recycle_v1_heldout() -> None:
    metadata = read_json("metadata.json")
    plan = (OUT / "COHORT_VERSIONING_AND_SPLIT_PLAN.md").read_text(encoding="utf-8")
    assert metadata["cohort_v1_identity"] == "MYOLEG_VIRTUAL_PATIENT_COHORT_V1"
    assert metadata["future_cohort_identity"] == "MYOLEG_VIRTUAL_PATIENT_COHORT_V2"
    assert metadata["new_version_required"] is True
    assert "do not assume the V1 held-out set covers the V2 structural space" in plan
    assert "The pilot is not executed here" in plan


def test_formal_outcome_answers_all_questions_without_executing_pilot() -> None:
    metadata = read_json("metadata.json")
    report = (OUT / "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_REPORT.md").read_text(encoding="utf-8")
    assert metadata["outcome"] == audit.OUTCOME
    assert audit.OUTCOME in report
    assert all(f"## Q{index}." in report for index in range(1, 11))
    assert "Held-out scientific access: **0**" in report
    assert "not executed" in report
    assert metadata["next_stage_executed"] is False
    assert metadata["robot_or_hardware"] is False


def test_required_artifacts_and_checksum_coverage() -> None:
    required = {
        "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_REPORT.md",
        "MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_PROTOCOL.json",
        "MYOLEG_STRUCTURAL_PARAMETER_INVENTORY.csv", "MUSCLE_OPERATING_LENGTH_PARAMETER_AUDIT.csv",
        "BIARTICULAR_COUPLING_PARAMETER_AUDIT.csv", "MOMENT_ARM_GEOMETRY_AUDIT.csv",
        "SEGMENT_GEOMETRY_ANTHROPOMETRY_AUDIT.csv", "JOINT_PASSIVE_MECHANICS_AUDIT.csv",
        "TENDON_PROPERTY_AUDIT.csv", "MUSCLE_GROUP_HETEROGENEITY_AUDIT.csv",
        "STRUCTURAL_LOCAL_SENSITIVITY_RESULTS.csv", "HETEROGENEITY_FACTOR_TAXONOMY.csv",
        "PROPOSED_STRUCTURAL_HETEROGENEITY_SCHEMES.json", "COHORT_VERSIONING_AND_SPLIT_PLAN.md",
        "FUTURE_RANGE_EVIDENCE_REQUIREMENTS.csv", "HELD_OUT_ACCESS_AUDIT.json",
        "metadata.json", "checksums.sha256",
    }
    assert required <= {path.name for path in OUT.iterdir() if path.is_file()}
    entries = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual)
    assert all(sha256(OUT / relative) == expected for relative, expected in entries.items())


def test_source_has_no_robot_or_forbidden_scientific_execution() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "import hardware" not in source
    assert "import control" not in source
    assert "held_out_subject" not in source
    assert "external_simulation/data" not in source
    assert "replay_v3_subject_candidate" not in source
    assert "bayesian" not in source.lower()
