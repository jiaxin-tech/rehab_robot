"""Regression gates for the frozen compact MyoLeg-V2 truth landscape."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1"
DATA = ROOT / "external_simulation/data/myoleg_v2_truth_landscape_v1"
COHORT = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
CANDIDATES = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
SOURCE = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/build_truth_landscape.py"
REPLAY_API = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/replay_api.py"
ACCESS_API = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/truth_access.py"

COHORT_SHA = "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057"
CANDIDATE_SHA = "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7"
EXPECTED_PAIRS = 533600
REFERENCE = "MYOLEG_V2_P15012"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_manifest() -> dict:
    return load_json(AUDIT / "MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json")


def all_local_shards_available(chunks: list[dict]) -> bool:
    return all((ROOT / row["path"]).is_file() for row in chunks)


def test_frozen_cohort_and_candidate_manifests_are_unchanged() -> None:
    assert sha256(COHORT) == COHORT_SHA
    assert sha256(CANDIDATES) == CANDIDATE_SHA


def test_exact_evaluation_set_and_reference_identity() -> None:
    cohort = load_json(COHORT)
    candidates = load_json(CANDIDATES)
    assert len(cohort["subjects"]) == 32
    assert sum(row["split"] == "DEVELOPMENT" for row in cohort["subjects"]) == 24
    assert sum(row["split"] == "HELD_OUT" for row in cohort["subjects"]) == 8
    assert len(candidates["ordered_included_candidates"]) == 16675
    assert 32 * 16675 == EXPECTED_PAIRS
    reference = next(row for row in candidates["ordered_included_candidates"] if row["candidate_id"] == REFERENCE)
    assert reference == {"alpha": [0.0, 0.0, 0.0], "candidate_id": REFERENCE, "proposal_index": 15012}


def test_protocol_was_frozen_before_results_and_worker_count_by_benchmark() -> None:
    protocol = load_json(AUDIT / "LANDSCAPE_GENERATION_PROTOCOL.json")
    benchmark = load_json(AUDIT / "PARALLELISM_BENCHMARK.json")
    plan = load_json(AUDIT / "LANDSCAPE_EXECUTION_PLAN.json")
    assert protocol["frozen_before_landscape_outcomes"] is True
    assert [row["workers"] for row in benchmark["worker_results"]] == [1, 2, 4, 8]
    assert all(row["process_failure_count"] == 0 for row in benchmark["worker_results"])
    assert all(row["deterministic_equal_to_one_worker"] for row in benchmark["worker_results"])
    assert plan["frozen_before_landscape_outcomes"] is True
    assert plan["landscape_worker_count"] == benchmark["selected_worker_count"]


def test_chunk_manifest_has_exact_nonoverlapping_coverage_and_hashes() -> None:
    chunks = load_json(AUDIT / "LANDSCAPE_CHUNK_MANIFEST.json")
    assert chunks["expected_row_count"] == chunks["actual_row_count"] == EXPECTED_PAIRS
    assert chunks["chunk_count"] == 2144
    assert len({row["chunk_id"] for row in chunks["chunks"]}) == 2144
    assert sum(row["row_count"] for row in chunks["chunks"]) == EXPECTED_PAIRS
    for row in chunks["chunks"]:
        path = ROOT / row["path"]
        assert len(row["sha256"]) == 64
        if path.is_file():
            assert sha256(path) == row["sha256"]
        assert row["integrity_failure_count"] == 0


def test_no_duplicate_or_missing_pair_and_all_rows_finite() -> None:
    manifest = final_manifest()
    pair_count = 0
    seen: set[tuple[str, str]] = set()
    scalar_float_fields = [
        "alpha_hip_deg", "alpha_knee_deg", "alpha_phase", "hip_tau_rms_nm", "knee_tau_rms_nm",
        "hip_tau_peak_abs_nm", "knee_tau_peak_abs_nm", "subject_reference_hip_rms_nm",
        "subject_reference_knee_rms_nm", "j_truth", "source_equality_residual_max",
        "joint_limit_contribution_max_abs_nm", "joint_limit_contribution_max_relative",
    ]
    if all_local_shards_available(manifest["chunks"]):
        for chunk in manifest["chunks"]:
            with np.load(ROOT / chunk["path"], allow_pickle=False) as shard:
                assert np.all(shard["integrity_status"] == 1)
                assert all(np.isfinite(shard[key]).all() for key in scalar_float_fields)
                for subject_id, candidate_id in zip(shard["subject_id"], shard["candidate_id"]):
                    pair = (str(subject_id), str(candidate_id))
                    assert pair not in seen
                    seen.add(pair)
                pair_count += len(shard["candidate_id"])
        assert pair_count == len(seen) == EXPECTED_PAIRS
    else:
        by_subject: dict[str, list[dict]] = {}
        for chunk in manifest["chunks"]:
            by_subject.setdefault(chunk["subject_id"], []).append(chunk)
        assert len(by_subject) == 32
        for rows in by_subject.values():
            rows.sort(key=lambda row: row["candidate_start_rank"])
            assert rows[0]["candidate_start_rank"] == 0
            assert rows[-1]["candidate_end_rank_exclusive"] == 16675
            assert all(left["candidate_end_rank_exclusive"] == right["candidate_start_rank"] for left, right in zip(rows, rows[1:]))
            assert sum(row["row_count"] for row in rows) == 16675
            assert all(row["integrity_failure_count"] == 0 for row in rows)


def test_reference_is_subject_specifically_normalized_to_one() -> None:
    rows = list(csv.DictReader((AUDIT / "SUBJECT_LANDSCAPE_SUMMARY.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 32
    assert all(row["reference_candidate_id"] == REFERENCE for row in rows)
    assert all(abs(float(row["reference_j"]) - 1.0) <= 1.0e-12 for row in rows)
    assert final_manifest()["integrity_summary"]["all_subject_reference_j_within_tolerance"] is True


def test_subject_denominators_are_retained_per_pair() -> None:
    cohort = {row["subject_id"]: row for row in load_json(COHORT)["subjects"]}
    manifest = final_manifest()
    if all_local_shards_available(manifest["chunks"]):
        for subject_id, subject in cohort.items():
            first = next(row for row in manifest["chunks"] if row["subject_id"] == subject_id)
            with np.load(ROOT / first["path"], allow_pickle=False) as shard:
                assert np.all(shard["subject_reference_hip_rms_nm"] == subject["subject_reference_tau_hip_rms_nm"])
                assert np.all(shard["subject_reference_knee_rms_nm"] == subject["subject_reference_tau_knee_rms_nm"])
    else:
        schema = {row["name"] for row in manifest["storage_schema"]}
        assert {"subject_reference_hip_rms_nm", "subject_reference_knee_rms_nm"} <= schema
        source = SOURCE.read_text(encoding="utf-8")
        assert 'float(subject["subject_reference_tau_hip_rms_nm"])' in source
        assert 'float(subject["subject_reference_tau_knee_rms_nm"])' in source


def test_truth_semantics_and_compact_schema_are_frozen() -> None:
    manifest = final_manifest()
    assert manifest["truth_semantic_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    assert manifest["truth_field"] == "TAU_MY0LEG_REQUIRED_DRIVE"
    names = {row["name"] for row in manifest["storage_schema"]}
    assert {"subject_id", "candidate_id", "proposal_index", "alpha_hip_deg", "alpha_knee_deg", "alpha_phase", "hip_tau_rms_nm", "knee_tau_rms_nm", "hip_tau_peak_abs_nm", "knee_tau_peak_abs_nm", "j_truth", "integrity_status"} <= names
    assert not ({"q_rad", "dq_rad_s", "ddq_rad_s2", "tau_truth_nm", "actuator_force_n"} & names)


def test_bulk_storage_contains_no_full_time_series_schema() -> None:
    manifest = final_manifest()
    prohibited = {"q_rad", "dq_rad_s", "ddq_rad_s2", "tau_truth_nm", "mass_term_nm", "actuator_force_n", "muscle_torque_contribution_nm"}
    if all_local_shards_available(manifest["chunks"]):
        for chunk in manifest["chunks"]:
            with np.load(ROOT / chunk["path"], allow_pickle=False) as shard:
                assert not (prohibited & set(shard.files))
                assert all(np.asarray(shard[key]).ndim == 1 for key in shard.files)
    else:
        assert not (prohibited & {row["name"] for row in manifest["storage_schema"]})
    runtime = load_json(AUDIT / "RUNTIME_STORAGE_AUDIT.json")
    assert runtime["bulk_full_time_series_generated"] is False


def test_deterministic_on_demand_replay_api_is_independent_of_oracle() -> None:
    source = REPLAY_API.read_text(encoding="utf-8")
    assert "def replay_subject_candidate(subject_id: str, candidate_id: str)" in source
    assert "prescribed_truth" in source
    assert "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1" in source
    assert "POST_FREEZE_ORACLE_SUMMARY" not in source
    detailed = list(csv.DictReader((AUDIT / "DETAILED_VALIDATION_RESULTS.csv").open(newline="", encoding="utf-8")))
    assert detailed
    assert all(row["prescribed_repeat_array_equal"] == "True" for row in detailed)
    assert all(row["first_array_payload_sha256"] == row["second_array_payload_sha256"] for row in detailed)


def test_resume_skips_valid_chunks_and_only_recomputes_missing_or_invalid() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "if valid:\n            valid_existing += 1" in source
    assert 'if reason != "missing"' in source
    assert '"action": "RECOMPUTE"' in source
    events = [json.loads(line) for line in (AUDIT / "RESUME_EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [row["chunk_id"] for row in events if row["action"] == "COMPLETED"]
    completion_counts = {chunk_id: completed.count(chunk_id) for chunk_id in set(completed)}
    recomputed_invalid = {
        row["chunk_id"]
        for row in events
        if row["action"] == "RECOMPUTE" and row["reason"] == "pair_integrity_invalid"
    }
    duplicated = {chunk_id for chunk_id, count in completion_counts.items() if count > 1}
    assert duplicated == recomputed_invalid
    assert all(count == (2 if chunk_id in recomputed_invalid else 1) for chunk_id, count in completion_counts.items())


def test_detailed_subset_covers_every_subject_and_frozen_geometry_roles() -> None:
    protocol = load_json(AUDIT / "LANDSCAPE_GENERATION_PROTOCOL.json")
    detail = protocol["detailed_validation_subset"]
    roles = {row["role"] for row in detail["candidate_rows"]}
    assert len(detail["all_subject_ids"]) == 32
    assert {"REFERENCE", "GLOBAL_HIP_LOW", "GLOBAL_HIP_HIGH", "GLOBAL_KNEE_LOW", "GLOBAL_KNEE_HIGH", "GLOBAL_PHASE_LOW", "GLOBAL_PHASE_HIGH", "KNEE_TRUSTED_BOUND_NEIGHBOR", "FIXED_HASH_INTERIOR"} <= roles
    rows = list(csv.DictReader((AUDIT / "DETAILED_VALIDATION_RESULTS.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == detail["pair_count"]
    assert {row["subject_id"] for row in rows} == set(detail["all_subject_ids"])
    assert all(row["detailed_integrity_pass"] == "True" for row in rows)


def test_controlled_crosscheck_is_diagnostic_and_never_replaces_truth() -> None:
    policy = load_json(AUDIT / "LANDSCAPE_GENERATION_PROTOCOL.json")
    rows = list(csv.DictReader((AUDIT / "CONTROLLED_CROSSCHECK_RESULTS.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 2 * policy["controlled_crosscheck_subset"]["pair_count"]
    assert {row["joint"] for row in rows} == {"hip", "knee"}
    assert final_manifest()["integrity_summary"]["controlled_crosscheck_joint_row_count"] == len(rows)
    assert "never replaces prescribed truth" in policy["truth"]["controlled_method_role"]


def test_truth_access_policy_prevents_unqueried_oracle_use() -> None:
    policy = load_json(AUDIT / "TRUTH_ACCESS_POLICY_V1.json")
    prohibited = " ".join(policy["prohibited"])
    assert "PINN reads unqueried tau" in prohibited
    assert "BO reads unqueried J" in prohibited
    assert "candidate selection uses oracle rank" in prohibited
    source = ACCESS_API.read_text(encoding="utf-8")
    assert "return replay_subject_candidate(subject_id, candidate_id)" in source
    assert "_ALLOWED_PURPOSES" in source


def test_oracle_was_revealed_only_after_checksums_and_uses_frozen_tie_rule() -> None:
    protocol = load_json(AUDIT / "LANDSCAPE_GENERATION_PROTOCOL.json")
    freeze = load_json(AUDIT / "LANDSCAPE_DATA_FREEZE.json")
    manifest = final_manifest()
    oracle = list(csv.DictReader((AUDIT / "POST_FREEZE_ORACLE_SUMMARY.csv").open(newline="", encoding="utf-8")))
    assert freeze["oracle_reveal_occurred"] is False
    assert freeze["row_count"] == EXPECTED_PAIRS
    assert manifest["landscape_frozen_before_oracle_reveal"] is True
    assert manifest["oracle_reveal_policy"]["tie_tolerance_j"] == protocol["oracle"]["equivalence_tolerance_j"]
    assert len(oracle) == 32
    assert all(row["oracle_reveal_occurred_after_landscape_freeze"] == "True" for row in oracle)


def test_no_learner_bo_robot_or_hardware_scope_expansion() -> None:
    manifest = final_manifest()
    scope = manifest["scope"]
    assert scope == {"bo": False, "five_parameter": False, "learner_trained": False, "nn_or_pinn": False, "offline_only": True, "robot_or_hardware": False}
    protocol = load_json(AUDIT / "LANDSCAPE_GENERATION_PROTOCOL.json")
    assert protocol["scope_guards"]["human_ready"] is False
    assert protocol["scope_guards"]["robot_or_hardware"] is False


def test_final_artifact_checksums_and_per_subject_hashes_verify() -> None:
    manifest = final_manifest()
    chunks = load_json(AUDIT / "LANDSCAPE_CHUNK_MANIFEST.json")
    assert manifest["final_data_sha256"] == chunks["global_data_sha256"]
    assert manifest["subject_landscape_sha256"] == chunks["subject_landscape_sha256"]
    assert len(manifest["subject_landscape_sha256"]) == 32
    for line in (AUDIT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(AUDIT / name.strip()) == expected
