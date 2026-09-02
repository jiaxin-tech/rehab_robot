"""Regression gates for the development-only objective/heterogeneity decision audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from external_simulation.myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1 import build_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_protocol_was_frozen_before_new_scientific_diagnostics() -> None:
    protocol = load_json("OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL.json")
    execution = load_json("ANALYSIS_EXECUTION_FREEZE.json")
    assert sha256(OUT / "OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL.json") == audit.FROZEN_SHA["protocol"]
    assert protocol["frozen_before_new_development_diagnostics_reveal"] is True
    assert execution["frozen_before_compact_or_replay_scientific_arrays_read"] is True
    assert execution["protocol_sha256"] == audit.FROZEN_SHA["protocol"]
    assert protocol["objective_weight_search"] is False
    assert protocol["hypotheses"] == [
        "H_OBJ_NORMALIZATION_LOSS", "H_OBJ_RMS_COMPRESSION",
        "H_OBJ_COMPONENT_AGGREGATION", "H_HET_MULTIPLICATIVE_SCALING",
        "H_HET_INSUFFICIENT_STRUCTURAL_VARIATION",
    ]


def test_frozen_inputs_and_prior_v3_result_are_unchanged() -> None:
    assert sha256(audit.TRUTH_MANIFEST) == audit.FROZEN_SHA["truth"]
    assert sha256(audit.CANDIDATE_MANIFEST) == audit.FROZEN_SHA["candidate_manifest"]
    assert sha256(audit.CANDIDATE_TABLE) == audit.FROZEN_SHA["candidate_table"]
    assert sha256(audit.COHORT_MANIFEST) == audit.FROZEN_SHA["cohort"]
    assert sha256(audit.NECESSITY / "checksums.sha256") == audit.FROZEN_SHA["necessity_checksums"]
    metadata = load_json("metadata.json")
    assert metadata["formal_objective_unchanged"] is True
    assert metadata["normalization_unchanged"] is True
    assert metadata["v3_parameterization_unchanged"] is True
    assert metadata["v3_candidate_domain_unchanged"] is True
    assert metadata["cohort_unchanged"] is True
    assert metadata["subject_ranges_unchanged"] is True
    assert metadata["new_subject_factors_added"] is False
    assert metadata["analysis_code_sha256"] == sha256(Path(audit.__file__))


def test_exact_development_population_and_heldout_remains_sealed() -> None:
    access = load_json("HELD_OUT_ACCESS_AUDIT.json")
    assert access["held_out_subject_ids"] == list(audit.HELD_OUT_IDS)
    assert access["held_out_file_hashes_verified"] == 24
    assert access["held_out_np_load_count"] == 0
    assert access["held_out_replay_count"] == 0
    assert access["held_out_scientific_access_count"] == 0
    assert access["development_compact_subject_count"] == 24
    assert access["development_replay_pair_count"] == 136
    assert not set(access["development_compact_subject_ids_accessed"]).intersection(audit.HELD_OUT_IDS)
    assert not set(access["development_replay_subject_ids"]).intersection(audit.HELD_OUT_IDS)
    assert all(
        file_row["operation"] == "streaming_sha_only"
        for subject_row in access["held_out_file_verification"]
        for file_row in subject_row["files"]
    )


def test_compact_store_rejects_heldout_before_path_resolution() -> None:
    truth = {"chunks": []}
    candidates = [{"candidate_id": f"K{index}"} for index in range(625)]
    store = audit.CompactDevelopmentStore(truth, candidates, [f"DEV_{index:02d}" for index in range(24)])
    with pytest.raises(PermissionError, match="before path resolution"):
        store.load("MYOLEG_VP_004")


def test_replay_subset_is_geometry_frozen_and_matches_compact_truth() -> None:
    protocol = load_json("OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL.json")
    subset = protocol["replay_subset"]
    manifest = load_json("DEVELOPMENT_REPLAY_CACHE_MANIFEST.json")
    integrity = load_json("REPLAY_COMPACT_INTEGRITY.json")
    assert subset["uses_objective_or_candidate_performance"] is False
    assert subset["pair_count"] == manifest["pair_count"] == 136
    assert manifest["selection_uses_geometry_only"] is True
    assert manifest["compact_landscape_or_oracle_read_by_replay_api"] is False
    assert manifest["warning_count_max"] == 0
    assert manifest["decomposition_residual_max_nm"] <= 2e-14
    assert sha256(ROOT / manifest["path"]) == manifest["sha256"]
    assert integrity["passed"] is True
    assert integrity["hip_rms_max_abs_error_nm"] <= integrity["integrity_tolerance_nm"]
    assert integrity["knee_rms_max_abs_error_nm"] <= integrity["integrity_tolerance_nm"]


def test_required_artifacts_and_diagnostic_dimensions() -> None:
    required = (
        "MYOLEG_OBJECTIVE_AND_HETEROGENEITY_DECISION_REPORT.md",
        "OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL.json", "HELD_OUT_ACCESS_AUDIT.json",
        "RAW_VS_NORMALIZED_INTERACTION_AUDIT.csv", "TIME_RESOLVED_INTERACTION_AUDIT.csv",
        "PEAK_DIAGNOSTIC_AUDIT.csv", "FORCE_COMPONENT_INTERACTION_AUDIT.csv",
        "PASSIVE_HETEROGENEITY_AUDIT.csv", "MASS_INERTIA_HETEROGENEITY_AUDIT.csv",
        "AFFINE_SCALING_ACROSS_REPRESENTATIONS.csv", "SUBJECT_GRADIENT_DIRECTION_AUDIT.csv",
        "OBJECTIVE_INFORMATION_RETENTION_SUMMARY.csv", "HETEROGENEITY_ADEQUACY_DECISION.json",
        "OBJECTIVE_ADEQUACY_DECISION.json", "FINAL_BRANCH_DECISION.json",
        "FUTURE_MUSCULOSKELETAL_FACTOR_TAXONOMY.csv", "metadata.json", "checksums.sha256",
    )
    assert all((OUT / name).is_file() for name in required)
    assert len(load_csv("RAW_VS_NORMALIZED_INTERACTION_AUDIT.csv")) == 7
    assert len(load_csv("TIME_RESOLVED_INTERACTION_AUDIT.csv")) == 28
    assert len(load_csv("PEAK_DIAGNOSTIC_AUDIT.csv")) == 4
    assert len(load_csv("FORCE_COMPONENT_INTERACTION_AUDIT.csv")) >= 10
    assert len(load_csv("SUBJECT_GRADIENT_DIRECTION_AUDIT.csv")) == 296
    assert len(load_csv("SUBJECT_PARAMETER_GRADIENT_ASSOCIATIONS.csv")) == 324


def test_normalization_preserves_subject_ordering_and_directions() -> None:
    comparisons = {
        row["joint"]: row for row in load_csv("RAW_VS_NORMALIZED_INTERACTION_AUDIT.csv")
        if row["record_type"] == "NORMALIZATION_COMPARISON"
    }
    assert set(comparisons) == {"hip", "knee"}
    for row in comparisons.values():
        assert float(row["within_subject_raw_normalized_spearman_min"]) >= 0.999999
        assert int(row["adjacent_sign_change_count"]) == 0
        assert int(row["local_gradient_direction_change_count"]) == 0
        assert row["ordering_retained"] == "True"
        assert row["ordering_loss"] == "False"


def test_time_peak_and_component_views_remain_diagnostic_only() -> None:
    time_rows = load_csv("TIME_RESOLVED_INTERACTION_AUDIT.csv")
    windows = [row for row in time_rows if row["record_type"] == "BRANCH_PHASE_WINDOW"]
    assert len(windows) == 24
    assert {row["window"] for row in windows} == {
        "FLEXION_EARLY", "FLEXION_MID", "FLEXION_LATE",
        "EXTENSION_EARLY", "EXTENSION_MID", "EXTENSION_LATE",
    }
    assert all(row["representation_ordering_evidence"] == "False" for row in windows)
    peaks = load_csv("PEAK_DIAGNOSTIC_AUDIT.csv")
    assert all(row["scientific_role"] == "DIAGNOSTIC_ONLY_NO_OBJECTIVE_CHANGE" for row in peaks)
    components = load_csv("FORCE_COMPONENT_INTERACTION_AUDIT.csv")
    assert {row["component"] for row in components} == set(audit.COMPONENTS)
    assert all(row["representation_ordering_evidence"] == "False" for row in components)


def test_decision_follows_the_prefrozen_matrix_without_next_stage_execution() -> None:
    objective = load_json("OBJECTIVE_ADEQUACY_DECISION.json")
    heterogeneity = load_json("HETEROGENEITY_ADEQUACY_DECISION.json")
    branch = load_json("FINAL_BRANCH_DECISION.json")
    assert objective["limited_conjunction"] is False
    assert objective["status"] == audit.OBJECTIVE_ADEQUATE
    assert heterogeneity["adequate_conjunction"] is False
    assert heterogeneity["status"] == audit.HETEROGENEITY_LIMITED
    assert heterogeneity["parameters_modulate_response_magnitude_not_preference"] is True
    assert branch["decision"] == audit.DECISION_HETEROGENEITY
    assert branch["recommended_next_stage"] == "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_AUDIT_V1"
    assert branch["next_stage_executed"] is False


def test_no_forbidden_training_search_or_robot_action() -> None:
    metadata = load_json("metadata.json")
    assert metadata["objective_weight_search"] is False
    assert metadata["five_parameter_model_trained"] is False
    assert metadata["nn_or_pinn_trained"] is False
    assert metadata["bo_run"] is False
    assert metadata["robot_or_hardware"] is False
    assert metadata["next_stage_executed"] is False
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "import hardware" not in source
    assert "import control" not in source


def test_figures_and_report_cover_the_formal_questions() -> None:
    figures = sorted((OUT / "figures").glob("*.png"))
    assert len(figures) == 9
    assert all(path.stat().st_size > 20_000 for path in figures)
    report = (OUT / "MYOLEG_OBJECTIVE_AND_HETEROGENEITY_DECISION_REPORT.md").read_text(encoding="utf-8")
    assert all(f"### Q{index}." in report for index in range(1, 11))
    assert audit.DECISION_HETEROGENEITY in report
    assert "Held-out scientific access: **0**" in report
    assert "not establish physiological parameter values" in report
    assert "Next stage executed: **no**" in report


def test_checksums_cover_every_formal_artifact_without_self_reference() -> None:
    entries = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual)
    assert all(sha256(OUT / relative) == expected for relative, expected in entries.items())
