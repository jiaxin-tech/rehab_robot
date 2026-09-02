"""Regression gates for the development-only MyoLeg-V2 root-cause audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from external_simulation.myoleg_v2_personalization_signal_root_cause_audit_v1.build_audit import (
    DevelopmentLandscapeStore,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "external_simulation_audits/myoleg_v2_personalization_signal_root_cause_audit_v1"
SOURCE = ROOT / "external_simulation/myoleg_v2_personalization_signal_root_cause_audit_v1/build_audit.py"
HELPER = ROOT / "external_simulation/myoleg_v2_personalization_signal_root_cause_audit_v1/generate_replay_cache.py"
TRUTH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
TRUTH_PROTOCOL = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/LANDSCAPE_GENERATION_PROTOCOL.json"
CANDIDATES = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COHORT = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
NECESSITY = ROOT / "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1"
REPLAY_CACHE = ROOT / "external_simulation/data/myoleg_v2_personalization_signal_root_cause_audit_v1/development_replay_subset.npz"

PROTOCOL_SHA = "2beac2ffb512783bcbe6dfcf60e8d64d9b6be8a5fe2122b8c77da876e6202bbb"
FROZEN_SHA = {
    TRUTH: "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    TRUTH_PROTOCOL: "2fe115d8c34685c70672bcc6a4d9752a88dfbb2cf12fb12d60df877755b7fdcc",
    CANDIDATES: "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    COHORT: "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    NECESSITY / "checksums.sha256": "0b4a449186b571b9c39207dc18cab1ec1366004821b8d9fd74e97792d87437d3",
}
HELD_OUT = {
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
}
EXPECTED_FIGURES = {
    "development_hip_slices.png", "development_knee_slices.png", "development_phase_slices.png",
    "common_candidate_effect_vs_interaction.png", "landscape_svd_explained_variance.png",
    "raw_vs_normalized_variability.png", "selected_time_resolved_torque.png",
    "force_component_decomposition.png",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict[str, str]]:
    with (AUDIT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_all_frozen_scientific_inputs_are_byte_unchanged() -> None:
    assert {path: sha256(path) for path in FROZEN_SHA} == FROZEN_SHA
    protocol = load_json(TRUTH_PROTOCOL)
    assert protocol["truth"]["objective"] == "sqrt(0.5*((hip_rms/subject_reference_hip_rms)^2+(knee_rms/subject_reference_knee_rms)^2))"


def test_every_personalization_necessity_artifact_is_unchanged() -> None:
    count = 0
    for line in (NECESSITY / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(NECESSITY / relative.strip()) == expected
        count += 1
    assert count == 60


def test_root_cause_protocol_was_frozen_before_outcome_analysis() -> None:
    path = AUDIT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json"
    protocol = load_json(path)
    freeze = load_json(AUDIT / "ANALYSIS_EXECUTION_FREEZE.json")
    assert sha256(path) == PROTOCOL_SHA
    assert protocol["frozen_before_development_outcome_matrix_read"] is True
    assert freeze["development_outcome_matrix_read_at_execution_freeze"] is False
    assert set(protocol["hypotheses"]) == {
        "H1_NORMALIZATION_CANCELLATION", "H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION",
        "H3_RMS_OBJECTIVE_COMPRESSION", "H4_CANDIDATE_DOMAIN_MONOTONICITY",
    }
    assert protocol["finite_difference"]["neighbors"] == {
        "hip": [-0.25, 0.25], "knee": [-0.25, 0.25], "phase": [-0.0025, 0.0025],
    }


def test_held_out_truth_stayed_sealed_and_store_rejects_before_path_resolution() -> None:
    protocol = load_json(AUDIT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json")
    access = load_json(AUDIT / "HELD_OUT_ACCESS_AUDIT.json")
    assert set(access["sealed_subject_ids"]) == HELD_OUT
    assert access["held_out_scientific_truth_access_count"] == 0
    assert access["np_load_held_out_count"] == 0
    assert access["held_out_j_oracle_rank_component_access_count"] == 0
    assert access["post_freeze_oracle_summary_opened"] is False
    assert access["subject_landscape_summary_opened"] is False
    assert access["local_shard_count_present"] == access["local_shard_count_sha256_verified"]
    store = DevelopmentLandscapeStore(
        {"chunks": []}, {"ordered_included_candidates": []},
        protocol["population"]["development_subject_ids"],
    )
    with pytest.raises(PermissionError, match="before path resolution"):
        store.load_subject("MYOLEG_VP_004")


def test_exact_development_matrix_and_replay_subset_are_enforced() -> None:
    integrity = load_json(AUDIT / "DEVELOPMENT_MATRIX_INTEGRITY.json")
    protocol = load_json(AUDIT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json")
    replay = load_json(AUDIT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json")
    assert integrity["shape"] == [24, 16675]
    assert len(integrity["development_subject_ids_loaded"]) == 24
    assert not set(integrity["development_subject_ids_loaded"]).intersection(HELD_OUT)
    assert integrity["held_out_scientific_truth_access_count"] == 0
    assert integrity["objective_reconstruction_max_abs_error"] <= 2.0e-15
    assert len(protocol["replay_subset"]["subject_rows"]) == 6
    assert len(protocol["replay_subset"]["candidate_rows"]) == 20
    assert replay["replay_pair_count"] == 120
    assert replay["held_out_scientific_truth_access_count"] == 0
    assert replay["warning_count_max"] == 0
    assert replay["runtime_environment"]["python"] == "3.10.19"
    if REPLAY_CACHE.is_file():
        assert sha256(REPLAY_CACHE) == replay["cache_sha256"]


def test_replay_generator_denies_held_out_and_uses_existing_api() -> None:
    source = HELPER.read_text(encoding="utf-8")
    assert "if subject_id in HELD_OUT:" in source
    assert "held-out replay denied before API call" in source
    assert "api.replay_subject_candidate(subject_id, candidate_id)" in source
    assert "ARRAY_NAMES" in source


def test_variance_and_hypothesis_decisions_are_internally_consistent() -> None:
    decomposition = load_json(AUDIT / "TWO_WAY_VARIANCE_DECOMPOSITION.json")
    decisions = load_json(AUDIT / "ROOT_CAUSE_HYPOTHESIS_DECISIONS.json")
    for matrix in decomposition["matrices"].values():
        fractions = (
            matrix["subject_main_variance_fraction"] + matrix["candidate_main_variance_fraction"]
            + matrix["subject_candidate_interaction_variance_fraction"]
        )
        assert math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=2.0e-12)
    j = decomposition["matrices"]["J"]
    assert j["candidate_main_variance_fraction"] > 0.999
    assert j["subject_candidate_interaction_variance_fraction"] < 0.001
    assert decisions["hypotheses"]["H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION"]["status"] == "SUPPORTED"
    assert decisions["hypotheses"]["H4_CANDIDATE_DOMAIN_MONOTONICITY"]["status"] == "SUPPORTED"
    assert decisions["overall_outcome"] == "PERSONALIZATION_SIGNAL_ROOT_CAUSE_IDENTIFIED"


def test_local_and_global_monotonicity_are_explicit_and_development_only() -> None:
    local = [row for row in load_csv("REFERENCE_FINITE_DIFFERENCE_AUDIT.csv") if row["record_type"] == "SUMMARY" and row["metric"] == "J"]
    assert {(row["dimension"], float(row["sign_agreement_fraction"])) for row in local} == {
        ("hip", 1.0), ("knee", 1.0), ("phase", 1.0),
    }
    direction = {row["dimension"]: (float(row["minus_improvement_fraction"]), float(row["plus_improvement_fraction"])) for row in local}
    assert direction == {"hip": (0.0, 1.0), "knee": (0.0, 1.0), "phase": (1.0, 0.0)}
    global_rows = [row for row in load_csv("GLOBAL_MONOTONICITY_AUDIT.csv") if row["record_type"] == "POOLED_GLOBAL" and row["metric"] == "J"]
    assert all(float(row["majority_sign_fraction"]) == 1.0 for row in global_rows)


def test_force_and_rms_diagnostics_use_frozen_components_without_objective_redesign() -> None:
    protocol = load_json(AUDIT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json")
    rms = load_json(AUDIT / "RMS_COMPRESSION_SUMMARY.json")
    components = [row for row in load_csv("FORCE_COMPONENT_DECOMPOSITION.csv") if row["record_type"] == "SUMMARY"]
    assert {row["component"] for row in components} == {
        "mass", "bias_gravity", "passive", "zero_control_actuator", "constraint",
    }
    assert {row["joint"] for row in components} == {"hip", "knee"}
    assert rms["knee"]["time_resolved_to_rms_heterogeneity_ratio"] > 2.0
    assert all(not value for value in protocol["scope_guards"].values())


def test_scientific_outputs_contain_no_held_out_subject_identifier() -> None:
    for path in sorted(AUDIT.glob("*.csv")):
        text = path.read_text(encoding="utf-8")
        assert not any(subject_id in text for subject_id in HELD_OUT), path


def test_no_learner_bo_or_robot_scope_was_entered() -> None:
    metadata = load_json(AUDIT / "metadata.json")
    assert metadata["held_out_scientific_truth_access_count"] == 0
    assert metadata["scope"] == {
        "bo_run": False, "candidate_or_generator_modified": False,
        "cohort_or_ranges_modified": False, "models_trained": False,
        "normalization_modified": False, "objective_modified": False,
        "offline_only": True, "robot_or_hardware": False,
    }
    assert metadata["analysis_code_sha256"] == sha256(SOURCE)


def test_eight_required_figures_are_valid_pngs() -> None:
    figures = AUDIT / "figures"
    assert {path.name for path in figures.glob("*.png")} == EXPECTED_FIGURES
    for path in figures.glob("*.png"):
        payload = path.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 10_000


def test_final_artifact_checksums_verify() -> None:
    for line in (AUDIT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(AUDIT / relative.strip()) == expected
