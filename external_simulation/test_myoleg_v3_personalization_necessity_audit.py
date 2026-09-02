from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from external_simulation.myoleg_v3_personalization_necessity_audit_v1 import build_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_simulation_audits/myoleg_v3_personalization_necessity_audit_v1"


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


def test_protocol_was_frozen_before_v3_outcome_reveal() -> None:
    protocol = load_json("V3_PERSONALIZATION_NECESSITY_PROTOCOL.json")
    assert sha256(OUT / "V3_PERSONALIZATION_NECESSITY_PROTOCOL.json") == audit.FROZEN_SHA["protocol"]
    assert protocol["frozen_before_development_truth_reveal"] is True
    assert protocol["top_sets"]["counts"] == {
        "top_1_percent": 7, "top_5_percent": 32, "top_10_percent": 63,
    }
    assert protocol["near_oracle"]["epsilons_j"] == [0.0001, 0.0005, 0.001]
    assert protocol["decision_logic"]["supported_all_required"]["material_interaction_increase_required"] is True
    assert protocol["comparative_interpretation"]["material_interaction_threshold_fraction"] == 0.0025
    assert len(protocol["v2_protocol_exact_differences"]) == 6


def test_frozen_inputs_and_objective_domain_remain_unchanged() -> None:
    assert sha256(audit.TRUTH_MANIFEST) == audit.FROZEN_SHA["truth"]
    assert sha256(audit.CANDIDATE_MANIFEST) == audit.FROZEN_SHA["candidate_manifest"]
    assert sha256(audit.CANDIDATE_TABLE) == audit.FROZEN_SHA["candidate_table"]
    assert sha256(audit.COHORT_MANIFEST) == audit.FROZEN_SHA["cohort"]
    assert sha256(audit.V2_PROTOCOL) == audit.FROZEN_SHA["v2_protocol"]
    metadata = load_json("metadata.json")
    assert metadata["scope"] == {
        "bo_run": False, "candidate_domain_modified": False,
        "human_or_clinical": False, "models_trained": False,
        "nn_or_pinn": False, "normalization_modified": False,
        "objective_modified": False, "offline_only": True,
        "parameterization_modified": False, "robot_or_hardware": False,
    }
    assert metadata["analysis_code_sha256"] == sha256(Path(audit.__file__))


def test_exact_development_population_and_heldout_seal() -> None:
    heldout = load_json("HELD_OUT_ACCESS_AUDIT.json")
    assert heldout["held_out_subject_ids"] == list(audit.HELD_OUT_IDS)
    assert heldout["held_out_scientific_access_count"] == 0
    assert heldout["held_out_files_opened_via_np_load"] == 0
    assert heldout["held_out_replay_count"] == 0
    assert heldout["held_out_files_sha256_verified"] == 24
    assert heldout["development_subject_count_scientifically_loaded"] == 24
    assert heldout["held_out_subject_ids_scientifically_loaded"] == []
    assert not set(heldout["v3_development_chunk_subject_ids"]).intersection(audit.HELD_OUT_IDS)


def test_truth_store_rejects_heldout_before_path_resolution() -> None:
    store = audit.DevelopmentTruthStore(
        {"chunks": []}, [f"DEV_{index:02d}" for index in range(24)],
        [{"candidate_id": f"K{index}", "beta_flex": "0", "beta_extend": "0"} for index in range(625)],
    )
    with pytest.raises(PermissionError, match="before path resolution"):
        store.load_subject("MYOLEG_VP_004")


def test_required_artifacts_and_exact_row_counts() -> None:
    required = (
        "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md",
        "V3_PERSONALIZATION_NECESSITY_PROTOCOL.json", "HELD_OUT_ACCESS_AUDIT.json",
        "V3_DEV_ORACLE_SUMMARY.csv", "V3_ORACLE_DIVERSITY.csv",
        "V3_RANK_CORRELATION.csv", "V3_TOPSET_OVERLAP.csv",
        "V3_COMMON_BASELINE.json", "V3_PERSONALIZATION_GAP.csv",
        "V3_TWO_WAY_VARIANCE_DECOMPOSITION.json", "V3_MULTIPLICATIVE_SIMILARITY_AUDIT.csv",
        "V3_LOCAL_DIRECTION_AUDIT.csv", "V3_GLOBAL_DIRECTIONAL_AGREEMENT.csv",
        "V3_ORACLE_TRANSFER_MATRIX.csv", "V3_NEAR_ORACLE_PLATEAU.csv",
        "V3_COMMON_NEAR_ORACLE_ANALYSIS.csv", "V3_SUBJECT_PARAMETER_ASSOCIATIONS.csv",
        "V2_V3_PERSONALIZATION_COMPARISON.csv", "V2_V3_PERSONALIZATION_COMPARISON.json",
        "metadata.json", "checksums.sha256",
    )
    assert all((OUT / name).is_file() for name in required)
    assert len(load_csv("V3_DEV_ORACLE_SUMMARY.csv")) == 24
    assert len(load_csv("V3_ORACLE_DIVERSITY.csv")) == 276
    assert len(load_csv("V3_RANK_CORRELATION.csv")) == 276
    assert len(load_csv("V3_TOPSET_OVERLAP.csv")) == 276 * 3
    assert len(load_csv("V3_PERSONALIZATION_GAP.csv")) == 24
    assert len(load_csv("V3_LOCAL_DIRECTION_AUDIT.csv")) == 24 * 4
    assert len(load_csv("V3_GLOBAL_DIRECTIONAL_AGREEMENT.csv")) == 1200
    assert len(load_csv("V3_NEAR_ORACLE_PLATEAU.csv")) == 24 * 3
    assert len(load_csv("V3_COMMON_NEAR_ORACLE_ANALYSIS.csv")) == 3
    assert len(load_csv("V3_SUBJECT_PARAMETER_ASSOCIATIONS.csv")) == 30


def test_v3_result_freeze_precedes_v2_comparison() -> None:
    result_freeze = load_json("V3_ANALYSIS_RESULT_FREEZE.json")
    comparison = load_json("V2_V3_PERSONALIZATION_COMPARISON.json")
    assert result_freeze["frozen_before_v2_result_artifacts_were_opened_by_analysis_execution"] is True
    assert comparison["v3_result_freeze_sha256"] == sha256(OUT / "V3_ANALYSIS_RESULT_FREEZE.json")
    assert len(result_freeze["v3_only_artifact_sha256"]) == 26
    for relative, expected in result_freeze["v3_only_artifact_sha256"].items():
        assert sha256(OUT / relative) == expected


def test_decision_is_strictly_reproduced_from_frozen_rule() -> None:
    metrics = load_json("V3_DECISION_METRICS.json")
    metadata = load_json("metadata.json")
    assert audit.classify(metrics) == metadata["outcome"]
    assert metadata["outcome"] == audit.OUTCOME_NOT_SUPPORTED
    assert metrics["unique_oracle_candidate_count"] == 1
    assert metrics["relative_common_regret"]["max"] == 0.0
    assert metrics["off_diagonal_relative_oracle_transfer_regret"]["max"] == 0.0
    assert metrics["universal_near_oracle_epsilon_0_001"] is True
    assert metrics["all_subjects_share_one_boundary_oracle"] is True
    assert metadata["recommended_next_stage"] == "MYOLEG_OBJECTIVE_AND_MUSCULOSKELETAL_HETEROGENEITY_DECISION_AUDIT_V1"
    assert metadata["next_stage_executed"] is False


def test_v2_v3_comparison_uses_proportion_aware_near_oracle_semantics() -> None:
    comparison = load_json("V2_V3_PERSONALIZATION_COMPARISON.json")
    assert comparison["raw_near_oracle_candidate_counts_compared"] is False
    assert comparison["v2"]["interaction_variance_percent"] == pytest.approx(0.03311433595538275)
    assert comparison["v3"]["interaction_variance_percent"] == pytest.approx(0.13507371763654394)
    assert comparison["v2"]["distinct_oracle_count"] == comparison["v3"]["distinct_oracle_count"] == 1
    assert comparison["v2"]["universal_near_oracle_epsilon_0_001"] is True
    assert comparison["v3"]["universal_near_oracle_epsilon_0_001"] is True


def test_all_eight_preregistered_figures_exist() -> None:
    figures = sorted((OUT / "figures").glob("*.png"))
    assert len(figures) == 8
    assert [path.name[:2] for path in figures] == [f"{index:02d}" for index in range(1, 9)]
    assert all(path.stat().st_size > 10_000 for path in figures)


def test_checksums_cover_every_artifact_without_self_reference() -> None:
    entries = {}
    for line in (OUT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual_paths = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    assert sorted(entries) == sorted(str(path.relative_to(OUT)) for path in actual_paths)
    for relative, expected in entries.items():
        assert sha256(OUT / relative) == expected


def test_report_preserves_scientific_claim_boundaries() -> None:
    report = (OUT / "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md").read_text(encoding="utf-8")
    assert "oracle upper-bound" in report
    assert "subject-specific mechanical trajectory preference" in report
    assert "HELD_OUT_SCIENTIFIC_ACCESS_COUNT = 0" in report
    assert "Do not execute it automatically" in report
    assert "patient preference" in report
    assert "not achieved algorithm benefit" in report.lower()
