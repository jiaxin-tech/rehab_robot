"""Regression gates for the versioned S1 structural-factor amendment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from external_simulation.myoleg_s1_structural_factor_definition_amendment_v1 import build_amendment as amendment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1"


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


def test_protocol_was_frozen_before_membership_and_preserves_old_s1() -> None:
    protocol = read_json("S1_FACTOR_DEFINITION_AMENDMENT_PROTOCOL.json")
    assert sha256(amendment.PROTOCOL) == amendment.FROZEN_SHA["protocol"]
    assert protocol["frozen_before_final_group_membership_or_operator_candidate_results"] is True
    assert protocol["old_s1_sha256"] == amendment.FROZEN_SHA["old_s1"]
    assert protocol["versioning"]["overwrite_old_s1"] is False
    assert protocol["operator_selection_prohibited_inputs"] == [
        "J", "oracle", "ranking", "personalization", "gradient rotation outcome", "trajectory truth outcome"
    ]


def test_all_frozen_inputs_and_formal_conventions_are_unchanged() -> None:
    assert {name: sha256(path) for name, path in amendment.FROZEN_PATHS.items()} == amendment.FROZEN_SHA
    metadata = read_json("metadata.json")
    assert metadata["old_s1_overwritten"] is False
    assert metadata["v1_cohort_modified"] is False
    assert metadata["v3_parameterization_or_domain_modified"] is False
    assert metadata["objective_or_normalization_modified"] is False
    assert metadata["analysis_code_sha256"] == sha256(Path(amendment.__file__))


def test_mechanical_membership_uses_prefrozen_thresholds_and_401_states() -> None:
    rows = read_csv("S1_GROUP_MEMBERSHIP_AUDIT.csv")
    assert len(rows) == 40
    assert all(int(row["state_count"]) == 401 for row in rows)
    assert all(row["classification_basis"] == "unit positive muscle tension effect=-projected actuator moment; no J/oracle/rank" for row in rows)
    protocol = read_json("S1_FACTOR_DEFINITION_AMENDMENT_PROTOCOL.json")
    rules = protocol["membership_derivation"]
    assert rules["effective_moment_arm_abs_m"] == 1e-5
    assert rules["negligible_other_joint_max_abs_m"] == 1e-6
    assert rules["minimum_target_effective_state_fraction"] == 0.8
    assert rules["minimum_antagonist_negative_sign_fraction"] == 0.95


def test_exact_biarticular_rectus_and_hamstring_groups() -> None:
    rows = read_csv("S1_GROUP_MEMBERSHIP_AUDIT.csv")
    biarticular = [row["muscle"] for row in rows if row["biarticular_included"] == "True"]
    rectus = [row["muscle"] for row in rows if row["final_balance_group"] == "RECTUS_GROUP"]
    hamstring = [row["muscle"] for row in rows if row["final_balance_group"] == "HAMSTRING_GROUP"]
    unchanged = [row["muscle"] for row in rows if row["biarticular_included"] == "True" and row["final_balance_group"] == "UNCHANGED_BY_BALANCE_FACTOR"]
    assert biarticular == list(amendment.BIARTICULAR_EXPECTED)
    assert rectus == ["recfem_r"]
    assert hamstring == ["bflh_r", "semimem_r", "semiten_r"]
    assert unchanged == ["grac_r", "sart_r", "tfl_r"]
    assert next(row for row in rows if row["muscle"] == "grac_r")["mechanical_family_pattern"] == "HAMSTRING_LIKE"
    assert next(row for row in rows if row["muscle"] == "tfl_r")["mechanical_family_pattern"] == "RECTUS_LIKE"


def test_exact_hip_and_knee_antagonist_memberships_and_anatomy_review() -> None:
    hip = read_csv("HIP_ANTAGONIST_MEMBERSHIP.csv")
    knee = read_csv("KNEE_ANTAGONIST_MEMBERSHIP.csv")
    hip_members = [row["muscle"] for row in hip if row["mechanically_included"] == "True"]
    knee_members = [row["muscle"] for row in knee if row["mechanically_included"] == "True"]
    assert hip_members == ["addmagDist_r", "addmagIsch_r", "addmagMid_r", "glmax2_r", "glmax3_r", "glmed3_r", "piri_r"]
    assert knee_members == ["vasint_r", "vaslat_r", "vasmed_r"]
    assert all(float(row["target_negative_fraction"]) >= 0.95 for row in hip + knee if row["mechanically_included"] == "True")
    assert "AMBIGUOUS" in next(row for row in hip if row["muscle"] == "glmed3_r")["anatomical_review"]
    assert "AMBIGUOUS" in next(row for row in hip if row["muscle"] == "piri_r")["anatomical_review"]


def test_lmax_operator_is_exact_relative_and_does_not_touch_lengthrange() -> None:
    audit = read_json("BIARTICULAR_LMAX_OPERATOR_AUDIT.json")
    assert audit["chosen_operator"] == "L1_LOG_MULTIPLICATIVE_LMAX_ONLY"
    assert audit["members"] == list(amendment.BIARTICULAR_EXPECTED)
    assert audit["fields"] == ["actuator_gainprm[:,5]", "actuator_biasprm[:,5]"]
    assert "lmax_i0*exp(z)" in audit["operator"]
    assert "actuator_lengthrange" in audit["unchanged_fields"]
    assert "no population or pilot bound" in audit["hard_mathematical_domain"]
    assert audit["selection_used_scientific_outcome"] is False


def test_balance_operator_is_log_symmetric_and_nonfamily_is_unchanged() -> None:
    audit = read_json("RECTUS_HAMSTRING_BALANCE_OPERATOR_AUDIT.json")
    assert audit["chosen_operator"] == "B2_LOG_SYMMETRIC_FAMILY_CENTERS"
    assert audit["rectus_group"] == ["recfem_r"]
    assert audit["hamstring_group"] == ["bflh_r", "semimem_r", "semiten_r"]
    assert audit["unchanged_biarticular"] == ["grac_r", "sart_r", "tfl_r"]
    assert "exp(z)" in audit["operator"] and "exp(-z)" in audit["operator"]
    assert "removed" in audit["global_scale_policy"]
    assert audit["selection_used_scientific_outcome"] is False


def test_f0_operator_is_not_mislabeled_pure_active_strength() -> None:
    audit = read_json("F0_OPERATOR_AUDIT.json")
    assert audit["chosen_operator"] == "F1_LOG_GROUP_SCALE"
    assert audit["fields"] == ["actuator_gainprm[:,2]", "actuator_biasprm[:,2]"]
    assert "active-plus-passive" in audit["semantic"]
    assert "not pure active strength" in audit["semantic"]
    assert "own nominal F0" in audit["member_weighting"]
    assert audit["selection_used_scientific_outcome"] is False


def test_all_operator_identity_and_integrity_checks_pass_without_scientific_replay() -> None:
    rows = read_csv("OPERATOR_IDENTITY_AND_INTEGRITY_CHECKS.csv")
    assert len(rows) == 4
    assert all(row["nominal_identity_bitwise"] == "True" for row in rows)
    assert all(row["only_declared_members_and_field_changed"] == "True" for row in rows)
    assert all(row["gain_bias_synchronized"] == "True" for row in rows)
    assert all(row["positive_domain"] == "True" for row in rows)
    assert all(row["topology_fingerprint_unchanged"] == "True" for row in rows)
    assert all(int(row["solver_warning_count"]) == 0 for row in rows)
    assert all(row["scientific_trajectory_replay"] == "False" for row in rows)
    assert all(row["J_or_oracle_read"] == "False" for row in rows)
    assert all(row["pass"] == "True" for row in rows)


def test_v1_relationships_remove_duplicate_global_balance_degree() -> None:
    rows = {row["entity"]: row for row in read_csv("V1_S1_FACTOR_RELATIONSHIP.csv")}
    assert rows["FEMUR_MASS_INERTIA_SCALE"]["relationship"] == "RETAIN_AS_BACKGROUND"
    assert rows["TIBIA_PATELLA_MASS_INERTIA_SCALE"]["relationship"] == "RETAIN_AS_BACKGROUND"
    assert rows["FOOT_COMPLEX_MASS_INERTIA_SCALE"]["relationship"] == "RETAIN_AS_BACKGROUND"
    assert rows["HIP_ONLY_PASSIVE_FP_MAX_SCALE"]["relationship"] == "SECONDARY_ONLY"
    assert rows["KNEE_ONLY_PASSIVE_FP_MAX_SCALE"]["relationship"] == "SECONDARY_ONLY"
    assert rows["HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE"]["relationship"] == "REMOVE_FROM_PERSONALIZATION_FOCUSED_COHORT"


def test_amended_definition_is_complete_versioned_and_range_free() -> None:
    amended = read_json("S1_STRUCTURAL_DEFINITION_AMENDED_V1.json")
    metadata = read_json("metadata.json")
    assert amended["definition_id"] == amendment.AMENDED_ID
    assert amended["old_s1_sha256"] == amendment.FROZEN_SHA["old_s1"]
    assert amended["old_s1_overwritten"] is False
    assert amended["declared_dimensionality"] == 4
    assert amended["all_factors_resolved"] is True
    assert len(amended["factors"]) == 4
    assert amended["population_ranges_frozen"] is False
    assert amended["pilot_diagnostic_levels_frozen"] is False
    assert amended["scientific_outcome_used"] is False
    assert amended["J_or_oracle_or_rank_used"] is False
    assert metadata["amended_definition_sha256"] == sha256(OUT / "S1_STRUCTURAL_DEFINITION_AMENDED_V1.json")


def test_source_metadata_and_report_preserve_zero_access_stop_state() -> None:
    sources = read_json("SOURCE_AND_EVIDENCE_METADATA.json")
    metadata = read_json("metadata.json")
    report = (OUT / "MYOLEG_S1_STRUCTURAL_FACTOR_DEFINITION_AMENDMENT_REPORT.md").read_text(encoding="utf-8")
    assert sources["reference_states_used"] == 401
    assert sources["scientific_trajectory_truth_replay"] is False
    assert sources["J_or_oracle_or_rank_read"] is False
    assert sources["held_out_scientific_access_count"] == 0
    assert metadata["outcome"] == amendment.OUTCOME
    assert metadata["scientific_trajectory_replays"] == 0
    assert metadata["next_stage"] == "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2"
    assert metadata["next_stage_executed"] is False
    assert all(f"## Q{index}." in report for index in range(1, 11))
    assert "Held-out scientific access: **0**" in report


def test_required_artifacts_and_checksum_coverage() -> None:
    required = {
        "MYOLEG_S1_STRUCTURAL_FACTOR_DEFINITION_AMENDMENT_REPORT.md",
        "S1_FACTOR_DEFINITION_AMENDMENT_PROTOCOL.json", "S1_GROUP_MEMBERSHIP_AUDIT.csv",
        "HIP_ANTAGONIST_MEMBERSHIP.csv", "KNEE_ANTAGONIST_MEMBERSHIP.csv",
        "BIARTICULAR_LMAX_OPERATOR_AUDIT.md", "BIARTICULAR_LMAX_OPERATOR_AUDIT.json",
        "RECTUS_HAMSTRING_BALANCE_OPERATOR_AUDIT.md", "RECTUS_HAMSTRING_BALANCE_OPERATOR_AUDIT.json",
        "F0_OPERATOR_AUDIT.md", "F0_OPERATOR_AUDIT.json", "V1_S1_FACTOR_RELATIONSHIP.csv",
        "OPERATOR_IDENTITY_AND_INTEGRITY_CHECKS.csv", "S1_STRUCTURAL_DEFINITION_AMENDED_V1.json",
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


def test_builder_has_no_scientific_truth_optimizer_or_robot_path() -> None:
    source = Path(amendment.__file__).read_text(encoding="utf-8")
    assert "prescribed_truth(" not in source
    assert "replay_v3_subject_candidate" not in source
    assert "import hardware" not in source
    assert "import control" not in source
    assert "bayesian" not in source.lower()

