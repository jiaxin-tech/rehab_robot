"""Regression gates for the development-only MyoLeg-V2 necessity audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from external_simulation.myoleg_v2_personalization_necessity_audit_v1.build_audit import (
    DevelopmentTruthStore,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1"
SOURCE = ROOT / "external_simulation/myoleg_v2_personalization_necessity_audit_v1/build_audit.py"
TRUTH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
CANDIDATES = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COHORT = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"

FROZEN_INPUT_SHA = {
    TRUTH: "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    CANDIDATES: "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    COHORT: "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
}
PROTOCOL_SHA = "f26663a71960c2f5cedb3d374cced98b0852fbfd718fe8235f0e1d9e6d102e6f"
POLICY_SHA = "b5e103a20869e626bd190479a91f09b3199791a895b6f91061c3d0363cfbef73"
HELD_OUT_IDS = {
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
}
EXPECTED_FIGURES = {
    "oracle_alpha_distribution.png",
    "development_rank_correlation_heatmap.png",
    "common_vs_oracle_regret_distribution.png",
    "oracle_cross_transfer_heatmap.png",
    "near_oracle_candidate_count.png",
    "representative_landscape_slices.png",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict[str, str]]:
    with (AUDIT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_frozen_inputs_protocol_and_access_policy_are_unchanged() -> None:
    assert {path: sha256(path) for path in FROZEN_INPUT_SHA} == FROZEN_INPUT_SHA
    assert sha256(AUDIT / "PERSONALIZATION_NECESSITY_PROTOCOL.json") == PROTOCOL_SHA
    assert sha256(AUDIT / "HELD_OUT_TRUTH_ACCESS_POLICY_V1.json") == POLICY_SHA


def test_protocol_was_frozen_before_development_truth_and_metrics_are_fixed() -> None:
    protocol = load_json(AUDIT / "PERSONALIZATION_NECESSITY_PROTOCOL.json")
    assert protocol["frozen_before_development_truth_reveal"] is True
    assert protocol["analysis_population"]["development_count"] == 24
    assert set(protocol["analysis_population"]["held_out_subject_ids"]) == HELD_OUT_IDS
    assert protocol["analysis_population"]["held_out_scientific_values_allowed"] is False
    assert protocol["oracle"]["tie_tolerance_j"] == 1.0e-12
    assert protocol["rank_similarity"]["subject_pair_count"] == 276
    assert protocol["top_sets"]["counts"] == {
        "top_1_percent": 167, "top_5_percent": 834, "top_10_percent": 1668,
    }
    assert protocol["near_oracle"]["epsilons_j"] == [0.0001, 0.0005, 0.001]
    assert protocol["personalization_gap"]["bootstrap"] == {
        "interval": "percentile 95% CI", "resamples": 10000,
        "seed": 20260830, "statistics": ["mean", "median"],
    }


def test_held_out_truth_stayed_sealed_and_store_fails_before_path_resolution() -> None:
    policy = load_json(AUDIT / "HELD_OUT_TRUTH_ACCESS_POLICY_V1.json")
    access = load_json(AUDIT / "TRUTH_ACCESS_AUDIT.json")
    protocol = load_json(AUDIT / "PERSONALIZATION_NECESSITY_PROTOCOL.json")
    assert policy["scientific_array_values_loaded"] is False
    assert policy["manifest_row_count"] == 8 * 16675
    assert policy["existing_post_freeze_oracle_summary_opened"] is False
    assert policy["existing_subject_landscape_summary_opened"] is False
    assert access["held_out_subject_ids_loaded"] == []
    assert access["held_out_scientific_values_read"] is False
    assert len(access["development_subject_ids_loaded"]) == 24
    store = DevelopmentTruthStore(
        {"chunks": []}, {"ordered_included_candidates": []},
        protocol["analysis_population"]["development_subject_ids"],
    )
    with pytest.raises(PermissionError, match="before path resolution"):
        store.load_subject("MYOLEG_VP_004")


def test_no_held_out_subject_appears_in_development_scientific_outputs() -> None:
    for path in sorted(AUDIT.glob("DEV_*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not any(subject_id in text for subject_id in HELD_OUT_IDS), path


def test_development_oracles_common_baselines_and_transfer_are_consistent() -> None:
    oracle = load_csv("DEV_ORACLE_SUMMARY.csv")
    gap = load_csv("DEV_PERSONALIZATION_GAP.csv")
    transfer = load_csv("DEV_ORACLE_TRANSFER_MATRIX.csv")
    baselines = load_json(AUDIT / "DEV_COMMON_BASELINES.json")
    assert len(oracle) == 24
    assert {row["candidate_id"] for row in oracle} == {"MYOLEG_V2_P20850"}
    assert {(float(row["alpha_hip_deg"]), float(row["alpha_knee_deg"]), float(row["alpha_phase"])) for row in oracle} == {(2.0, 0.5, -0.03)}
    assert all(row["any_candidate_domain_boundary"] == "True" for row in oracle)
    assert baselines["DEV_MEAN_OPTIMAL_COMMON"]["candidate_id"] == "MYOLEG_V2_P20850"
    assert baselines["DEV_WORSTCASE_OPTIMAL_COMMON"]["candidate_id"] == "MYOLEG_V2_P20850"
    assert all(float(row["common_regret"]) == 0.0 for row in gap)
    assert len(transfer) == 24 * 24
    assert all(float(row["regret_vs_recipient_oracle"]) == 0.0 for row in transfer)


def test_rank_topset_diversity_and_near_oracle_outputs_have_preregistered_shapes() -> None:
    diversity = load_csv("DEV_ORACLE_DIVERSITY.csv")
    ranks = load_csv("DEV_RANK_CORRELATION.csv")
    top = load_csv("DEV_TOPSET_OVERLAP.csv")
    common_near = load_csv("DEV_COMMON_NEAR_ORACLE_ANALYSIS.csv")
    assert len(diversity) == len(ranks) == math.comb(24, 2) == 276
    assert {row["classification"] for row in diversity} == {"EXACT_SAME_ORACLE"}
    assert min(float(row["spearman_rank_correlation"]) for row in ranks) >= 0.998
    assert len(top) == 3 * 276
    assert {int(row["top_count"]) for row in top} == {167, 834, 1668}
    epsilon_001 = next(row for row in common_near if float(row["epsilon_j"]) == 0.001)
    assert int(epsilon_001["maximum_subject_coverage"]) == 24
    assert epsilon_001["universal_near_oracle_candidate_exists"] == "True"


def test_descriptive_associations_are_finite_or_explicitly_undefined() -> None:
    rows = load_csv("DEV_PARAMETER_ORACLE_ASSOCIATIONS.csv")
    assert len(rows) == 30
    for row in rows:
        assert math.isfinite(float(row["raw_p_value"]))
        assert math.isfinite(float(row["bh_q_value_across_30_tests"]))
        if row["spearman_rho"]:
            assert math.isfinite(float(row["spearman_rho"]))
            assert row["undefined_reason"] == ""
        else:
            assert row["undefined_reason"] in {"CONSTANT_PARAMETER", "CONSTANT_OUTCOME", "CONSTANT_PARAMETER;CONSTANT_OUTCOME"}
            assert float(row["raw_p_value"]) == 1.0
        assert row["predictive_model_trained"] == "False"


def test_outcome_is_not_supported_and_no_model_or_robot_scope_was_entered() -> None:
    metadata = load_json(AUDIT / "metadata.json")
    protocol = load_json(AUDIT / "PERSONALIZATION_NECESSITY_PROTOCOL.json")
    report = (AUDIT / "MYOLEG_V2_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md").read_text(encoding="utf-8")
    assert metadata["outcome"] == "PERSONALIZATION_NECESSITY_NOT_SUPPORTED"
    assert metadata["held_out_truth_revealed"] is False
    assert metadata["scope"] == {
        "bo_run": False, "models_trained": False, "offline_only": True, "robot_or_hardware": False,
    }
    assert not any(protocol["scope_guards"].values())
    assert "`PERSONALIZATION_NECESSITY_NOT_SUPPORTED`" in report


def test_all_six_figures_are_present_and_valid_png_files() -> None:
    figures = AUDIT / "figures"
    assert {path.name for path in figures.glob("*.png")} == EXPECTED_FIGURES
    for path in figures.glob("*.png"):
        payload = path.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 10_000


def test_repairs_are_provenanced_and_final_checksums_verify() -> None:
    repair = load_json(AUDIT / "ANALYSIS_REPAIR_AUDIT.json")
    assert repair["decision_metrics_or_thresholds_changed"] is False
    assert repair["held_out_truth_access_changed"] is False
    assert repair["first_attempt_preserved_at"] == "attempt1_constant_input_warning/"
    assert repair["second_attempt"]["scientific_analysis_changed"] is False
    assert (AUDIT / repair["first_attempt_preserved_at"]).is_dir()
    assert (AUDIT / repair["second_attempt"]["preserved_at"]).is_dir()
    metadata = load_json(AUDIT / "metadata.json")
    assert metadata["analysis_code_sha256"] == sha256(SOURCE)
    for line in (AUDIT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(AUDIT / name.strip()) == expected
