"""Regression gates for the fail-closed structural pilot design stage."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from external_simulation.myoleg_structural_heterogeneity_pilot_design_v1 import build_design as design


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v1"


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


def test_authoritative_s1_and_all_frozen_inputs_are_unchanged() -> None:
    assert {name: sha256(path) for name, path in design.FROZEN_PATHS.items()} == design.FROZEN_SHA
    schemes = json.loads(design.SCHEMES.read_text(encoding="utf-8"))
    s1 = next(row for row in schemes["schemes"] if row["scheme_id"] == "S1_MINIMAL_STRUCTURAL")
    assert s1["factors"] == design.EXPECTED_S1_FACTORS
    assert s1["actual_model_fields"] == design.EXPECTED_S1_FIELDS
    assert s1["dimensionality"] == 4


def test_exact_reconstruction_fails_closed_on_missing_targets_and_operations() -> None:
    rows = read_csv("S1_EXACT_FACTOR_DEFINITION.csv")
    assert len(rows) == 4
    assert [row["factor_name"] for row in rows] == design.EXPECTED_S1_FACTORS
    assert [row["actual_fields"] for row in rows] == design.EXPECTED_S1_FIELDS
    assert rows[2]["affected_objects"] == "NOT_SPECIFIED"
    assert rows[3]["affected_objects"] == "NOT_SPECIFIED"
    assert rows[2]["definition_status"] == design.BLOCKER
    assert rows[3]["definition_status"] == design.BLOCKER
    assert all(row["direction_and_coherent_update_rule_complete"] == "False" for row in rows)


def test_protocol_declares_not_ready_before_any_pilot_execution() -> None:
    protocol = read_json("STRUCTURAL_HETEROGENEITY_PILOT_PROTOCOL.json")
    assert protocol["formal_outcome"] == design.OUTCOME
    assert protocol["blocker"] == design.BLOCKER
    assert protocol["frozen_before_any_structural_pilot_scientific_outcome"] is True
    assert protocol["exact_reconstruction_gate"]["status"] == design.BLOCKER
    assert protocol["pilot_execution_authorized"] is False
    assert protocol["scope_guards"]["held_out_scientific_access"] == 0
    assert protocol["scope_guards"]["pilot_scientific_outcome_executed"] is False


def test_no_diagnostic_levels_or_executable_trajectory_subset_were_invented() -> None:
    levels = read_json("PILOT_DIAGNOSTIC_LEVELS.json")
    subset = read_csv("PILOT_V3_TRAJECTORY_SUBSET.csv")
    assert levels["status"] == "NOT_FROZEN"
    assert levels["levels"] == []
    assert levels["population_ranges"] == []
    assert levels["finite_fallback_level"] is None
    assert levels["execution_authorized"] is False
    assert len(subset) == 1
    assert subset[0]["candidate_id"] == ""
    assert subset[0]["status"] == "NOT_FROZEN"
    assert subset[0]["execution_authorized"] == "False"


def test_scientific_and_integrity_gate_artifacts_are_non_executable() -> None:
    scientific = read_json("NONPROPORTIONALITY_GATES.json")
    integrity = read_json("PILOT_INTEGRITY_GATES.json")
    assert scientific["status"] == "NOT_NUMERICALLY_FROZEN"
    assert scientific["execution_authorized"] is False
    assert scientific["personalization_is_not_a_success_metric"] is True
    assert scientific["factor_outcomes"] == [
        "STRUCTURALLY_INFORMATIVE", "MAGNITUDE_ONLY", "INCONCLUSIVE", "INVALID"
    ]
    assert integrity["primary_level"] is None
    assert integrity["fallback_level"] is None
    assert integrity["maximum_fallback_attempts"] == 0
    assert integrity["execution_authorized"] is False


def test_execution_plan_authorizes_zero_models_and_replays() -> None:
    plan = read_json("PILOT_EXECUTION_PLAN.json")
    metadata = read_json("metadata.json")
    assert plan["status"] == "BLOCKED"
    assert plan["structural_diagnostic_model_count"] == 0
    assert plan["trajectory_replay_count"] == 0
    assert plan["new_virtual_subject_count"] == 0
    assert plan["scientific_pilot_outcome_executed"] is False
    assert plan["execution_authorized"] is False
    assert metadata["diagnostic_models_generated"] == 0
    assert metadata["trajectory_replays_executed"] == 0
    assert metadata["held_out_scientific_access_count"] == 0
    assert metadata["analysis_code_sha256"] == sha256(Path(design.__file__))


def test_population_ranges_remain_unavailable_and_are_not_pilot_levels() -> None:
    rows = read_csv("PILOT_RANGE_EVIDENCE.csv")
    assert len(rows) == 4
    assert all(row["population_range"] == "NOT AVAILABLE" for row in rows)
    assert all(row["pilot_diagnostic_level"] == "NOT FROZEN" for row in rows)
    assert all(row["range_status"] == "RANGE_REQUIRES_EXTERNAL_EVIDENCE" for row in rows)
    assert all(row["pilot_level_is_population_bound"] == "False" for row in rows)


def test_cohort_v2_admission_is_versioned_and_never_automatic() -> None:
    rules = read_json("COHORT_V2_FACTOR_ADMISSION_RULES.json")
    assert rules["new_version_required"] is True
    assert rules["future_identity"] == "MYOLEG_VIRTUAL_PATIENT_COHORT_V2"
    assert rules["automatic_admission"] is False
    assert len(rules["required_conjunction"]) == 4
    assert rules["range_gap_status"] == "COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP"
    assert rules["old_v1_heldout_is_automatic_confirmation"] is False


def test_report_answers_all_questions_and_preserves_negative_evidence() -> None:
    report = (OUT / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_REPORT.md").read_text(encoding="utf-8")
    metadata = read_json("metadata.json")
    assert design.OUTCOME in report
    assert design.BLOCKER in report
    assert all(f"## Q{index}." in report for index in range(1, 11))
    assert "Structural models/replays: **0 / 0**" in report
    assert "Held-out scientific access: **0**" in report
    assert metadata["preserved_negative_results"] == [
        "V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED", "HETEROGENEITY_LIMITATION_DOMINANT"
    ]


def test_required_artifacts_and_checksums_cover_every_file() -> None:
    required = {
        "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_REPORT.md",
        "STRUCTURAL_HETEROGENEITY_PILOT_PROTOCOL.json", "S1_EXACT_FACTOR_DEFINITION.csv",
        "PILOT_FACTOR_SEMANTICS.csv", "PILOT_RANGE_EVIDENCE.csv",
        "PILOT_DIAGNOSTIC_LEVELS.json", "PILOT_V3_TRAJECTORY_SUBSET.csv",
        "NONPROPORTIONALITY_GATES.json", "PILOT_INTEGRITY_GATES.json",
        "COHORT_V2_FACTOR_ADMISSION_RULES.json", "PILOT_EXECUTION_PLAN.json",
        "SOURCE_AND_EVIDENCE_METADATA.json", "metadata.json", "checksums.sha256",
    }
    assert required == {path.name for path in OUT.iterdir() if path.is_file()}
    entries = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual)
    assert all(sha256(OUT / relative) == expected for relative, expected in entries.items())


def test_builder_has_no_simulator_robot_or_optimizer_execution_path() -> None:
    source = Path(design.__file__).read_text(encoding="utf-8")
    assert "import mujoco" not in source
    assert "import hardware" not in source
    assert "import control" not in source
    assert "prescribed_truth" not in source
    assert "replay_v3_subject_candidate" not in source
    assert "bayesian" not in source.lower()

