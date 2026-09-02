"""Regression gates for the development-only compact MyoLeg-V3 landscape."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1"
DATA = ROOT / "external_simulation/data/myoleg_v3_development_truth_landscape_v1"
V3_DESIGN = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1"
V3_MANIFEST = V3_DESIGN / "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
V3_TABLE = V3_DESIGN / "V3_KINEMATIC_CANDIDATE_TABLE.csv"
COHORT = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
V2_MANIFEST = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
SOURCE = ROOT / "external_simulation/myoleg_v3_development_truth_landscape_generation_v1/build_landscape.py"
REPLAY_API = ROOT / "external_simulation/myoleg_v3_development_truth_landscape_generation_v1/replay_api.py"
TRUTH_ACCESS = ROOT / "external_simulation/myoleg_v3_development_truth_landscape_generation_v1/truth_access.py"

V3_MANIFEST_SHA = "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745"
V3_TABLE_SHA = "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774"
COHORT_SHA = "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057"
V2_MANIFEST_SHA = "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515"
PROTOCOL_SHA = "837f287f75d353af69bdd0e9ade5a417777c6de60a69cda693f6be9f094f133d"
FINAL_MANIFEST_SHA = "c318700e161e857d3e059eadf2bc21364e21b74d8b39faf4124a26fc15d37c6e"
REFERENCE = "MYOLEG_V3_K0312"
EXPECTED_PAIRS = 15000

DEVELOPMENT_IDS = (
    "MYOLEG_VP_001", "MYOLEG_VP_002", "MYOLEG_VP_003", "MYOLEG_VP_005",
    "MYOLEG_VP_006", "MYOLEG_VP_007", "MYOLEG_VP_009", "MYOLEG_VP_010",
    "MYOLEG_VP_011", "MYOLEG_VP_013", "MYOLEG_VP_014", "MYOLEG_VP_015",
    "MYOLEG_VP_017", "MYOLEG_VP_018", "MYOLEG_VP_019", "MYOLEG_VP_021",
    "MYOLEG_VP_022", "MYOLEG_VP_023", "MYOLEG_VP_025", "MYOLEG_VP_026",
    "MYOLEG_VP_027", "MYOLEG_VP_029", "MYOLEG_VP_030", "MYOLEG_VP_031",
)
HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def manifest() -> dict:
    return load_json(AUDIT / "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1_MANIFEST.json")


def test_all_frozen_inputs_and_prior_results_are_unchanged() -> None:
    assert sha256(V3_MANIFEST) == V3_MANIFEST_SHA
    assert sha256(V3_TABLE) == V3_TABLE_SHA
    assert sha256(COHORT) == COHORT_SHA
    assert sha256(V2_MANIFEST) == V2_MANIFEST_SHA
    metadata = load_json(AUDIT / "metadata.json")
    assert metadata["frozen_inputs_before"] == metadata["frozen_inputs_after"]


def test_protocol_was_frozen_before_any_V3_truth_and_exact_set_is_development_only() -> None:
    protocol_path = AUDIT / "V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL.json"
    protocol = load_json(protocol_path)
    assert sha256(protocol_path) == PROTOCOL_SHA
    assert protocol["frozen_before_any_V3_truth_replay"] is True
    evaluation = protocol["evaluation_set"]
    assert tuple(evaluation["development_subject_ids"]) == DEVELOPMENT_IDS
    assert tuple(evaluation["held_out_subject_ids_excluded"]) == HELD_OUT_IDS
    assert evaluation["development_subject_count"] == 24
    assert evaluation["candidate_count"] == 625
    assert evaluation["expected_pair_count"] == EXPECTED_PAIRS
    assert evaluation["nominal_control_included"] is False
    assert evaluation["reference_candidate_id"] == REFERENCE
    assert len(evaluation["candidate_ids_in_order"]) == 625


def test_parallelism_benchmark_is_deterministic_and_worker_count_was_frozen() -> None:
    benchmark = load_json(AUDIT / "V3_PARALLELISM_BENCHMARK.json")
    plan = load_json(AUDIT / "V3_LANDSCAPE_EXECUTION_PLAN.json")
    rows = benchmark["worker_results"]
    assert [row["workers"] for row in rows] == [1, 2, 4, 8]
    assert all(row["stable"] and row["deterministic_equal_to_one_worker"] for row in rows)
    assert all(row["process_failure_count"] == 0 for row in rows)
    assert benchmark["selected_worker_count"] == plan["formal_worker_count"] == 4
    assert plan["frozen_before_formal_landscape_generation"] is True


def test_chunk_manifest_has_24_atomic_subject_shards_and_exact_hashes() -> None:
    chunks = load_json(AUDIT / "V3_LANDSCAPE_CHUNK_MANIFEST.json")
    assert chunks["expected_pair_count"] == chunks["actual_pair_count"] == EXPECTED_PAIRS
    assert chunks["chunk_count"] == 24
    assert len(chunks["chunks"]) == 24
    assert tuple(row["subject_id"] for row in chunks["chunks"]) == DEVELOPMENT_IDS
    assert sum(row["row_count"] for row in chunks["chunks"]) == EXPECTED_PAIRS
    assert all(row["row_count"] == 625 and row["integrity_failure_count"] == 0 for row in chunks["chunks"])
    assert chunks["oracle_revealed"] is False
    for row in chunks["chunks"]:
        path = ROOT / row["path"]
        sidecar = path.with_suffix(".npz.sha256")
        assert path.is_file() and sidecar.is_file()
        assert sha256(path) == row["sha256"] == sidecar.read_text(encoding="utf-8").split()[0]


def test_all_15000_pairs_are_unique_finite_complete_and_integrity_pass() -> None:
    final = manifest()
    expected_ids = np.asarray(load_json(V3_MANIFEST)["ordered_candidate_ids"])
    seen: set[tuple[str, str]] = set()
    float_fields = {
        "beta_flex", "beta_extend", "hip_tau_rms_nm", "knee_tau_rms_nm",
        "hip_tau_peak_abs_nm", "knee_tau_peak_abs_nm", "subject_reference_hip_rms_nm",
        "subject_reference_knee_rms_nm", "j_truth", "source_equality_residual_max",
        "joint_limit_contribution_max_abs_nm", "joint_limit_contribution_max_relative",
    }
    for row in final["chunks"]:
        with np.load(ROOT / row["path"], allow_pickle=False) as shard:
            assert len(shard["candidate_id"]) == 625
            assert np.array_equal(shard["candidate_id"], expected_ids)
            assert np.array_equal(shard["candidate_index"], np.arange(625, dtype=np.int32))
            assert np.all(shard["subject_id"] == row["subject_id"])
            assert all(np.isfinite(shard[key]).all() for key in float_fields)
            assert np.all(shard["task_invariant_status"] == 1)
            assert np.all(shard["integrity_status"] == 1)
            for candidate_id in shard["candidate_id"]:
                pair = (row["subject_id"], str(candidate_id))
                assert pair not in seen
                seen.add(pair)
    assert len(seen) == EXPECTED_PAIRS
    assert final["actual_pair_count"] == EXPECTED_PAIRS
    assert final["duplicate_pair_count"] == 0
    assert final["missing_pair_count"] == 0


def test_reference_exists_once_per_subject_and_J_is_one() -> None:
    rows = read_csv(AUDIT / "V3_REFERENCE_NORMALIZATION_AUDIT.csv")
    assert len(rows) == 24
    assert tuple(row["subject_id"] for row in rows) == DEVELOPMENT_IDS
    assert all(row["reference_candidate_id"] == REFERENCE for row in rows)
    assert all(int(row["reference_candidate_index"]) == 312 for row in rows)
    assert all(abs(float(row["reference_j_truth"]) - 1.0) <= 1e-12 for row in rows)
    assert max(float(row["abs_error_from_one"]) for row in rows) <= 1e-12
    assert manifest()["integrity_summary"]["all_subject_reference_j_within_tolerance"] is True


def test_compact_storage_has_no_bulk_time_series_fields() -> None:
    final = manifest()
    schema = {row["name"] for row in final["storage_schema"]}
    required = {
        "subject_id", "candidate_id", "candidate_index", "beta_flex", "beta_extend",
        "hip_tau_rms_nm", "knee_tau_rms_nm", "hip_tau_peak_abs_nm", "knee_tau_peak_abs_nm",
        "j_truth", "task_invariant_status", "integrity_status",
    }
    prohibited = {"q_rad", "dq_rad_s", "ddq_rad_s2", "tau_truth_nm", "actuator_force_n", "mass_term_nm"}
    assert required <= schema
    assert not (prohibited & schema)
    for row in final["chunks"]:
        with np.load(ROOT / row["path"], allow_pickle=False) as shard:
            assert not (prohibited & set(shard.files))
            assert all(np.asarray(shard[key]).ndim == 1 for key in shard.files)
    runtime = load_json(AUDIT / "V3_RUNTIME_STORAGE_AUDIT.json")
    assert runtime["compact_shard_count"] == 24
    assert runtime["bulk_full_time_series_generated"] is False
    assert runtime["full_401_sample_time_series_reconstructable_on_demand"] is True


def test_detailed_subset_covers_reference_all_subjects_and_frozen_geometry() -> None:
    protocol = load_json(AUDIT / "V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL.json")
    pairs = protocol["detailed_validation_subset"]["pairs"]
    assert len(pairs) == 36
    references = [row for row in pairs if row["candidate_id"] == REFERENCE]
    assert tuple(row["subject_id"] for row in references) == DEVELOPMENT_IDS
    roles = {row["selection_role"] for row in protocol["detailed_validation_subset"]["geometry_candidate_rows"]}
    assert {"REFERENCE", "CORNER_NEG_NEG", "CORNER_NEG_POS", "CORNER_POS_NEG", "CORNER_POS_POS", "FLEX_NEG_AXIS", "FLEX_POS_AXIS", "EXTEND_NEG_AXIS", "EXTEND_POS_AXIS", "INTERIOR_NEG_NEG", "INTERIOR_NEG_POS", "INTERIOR_POS_NEG", "INTERIOR_POS_POS"} == roles
    rows = read_csv(AUDIT / "V3_DETAILED_VALIDATION_RESULTS.csv")
    assert len(rows) == 36
    assert all(row["prescribed_repeat_array_equal"] == "True" for row in rows)
    assert all(row["first_array_payload_sha256"] == row["second_array_payload_sha256"] for row in rows)
    assert all(row["trusted_ROM_valid"] == "True" and row["detailed_integrity_pass"] == "True" for row in rows)
    assert max(float(row["compact_vs_full_j_abs_error"]) for row in rows) <= 1e-12
    assert max(float(row["max_extrema_rom_error_deg"]) for row in rows) <= 1e-3


def test_controlled_crosscheck_is_small_diagnostic_and_passes() -> None:
    protocol = load_json(AUDIT / "V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL.json")
    assert protocol["controlled_crosscheck_subset"]["pair_count"] == 2
    assert "never replaces prescribed truth" in protocol["truth"]["controlled_method_role"]
    rows = read_csv(AUDIT / "V3_CONTROLLED_CROSSCHECK_RESULTS.csv")
    assert len(rows) == 4
    assert {row["joint"] for row in rows} == {"hip", "knee"}
    assert all(row["controlled_consistency_pass"] == "True" for row in rows)
    assert all(int(row["controlled_solver_warning_count"]) == 0 for row in rows)


def test_on_demand_API_is_deterministic_and_blocks_held_out_before_replay() -> None:
    audit = load_json(AUDIT / "V3_REPLAY_API_VALIDATION.json")
    assert audit["pass"] is True
    assert audit["all_arrays_equal"] is True
    assert audit["first_array_payload_sha256"] == audit["second_array_payload_sha256"]
    assert audit["held_out_rejected_before_replay"] is True
    assert audit["compact_landscape_or_oracle_read"] is False
    source = REPLAY_API.read_text(encoding="utf-8")
    assert "def replay_v3_subject_candidate(subject_id: str, candidate_id: str)" in source
    assert 'if subject_id in held_out_ids:' in source
    assert "prescribed_truth" in source
    assert "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1" in source


def test_truth_access_policy_is_query_only_and_held_out_scientific_access_zero() -> None:
    policy = load_json(AUDIT / "V3_TRUTH_ACCESS_POLICY_V1.json")
    held_out = load_json(AUDIT / "HELD_OUT_ACCESS_AUDIT.json")
    assert policy["development_generation_stage_oracle_reveal"] is False
    assert "bulk compact arrays are not a learner interface" in policy["query_access"]
    assert held_out["held_out_scientific_access_count"] == 0
    assert held_out["held_out_replay_count"] == 0
    assert held_out["np_load_held_out_count"] == 0
    assert tuple(held_out["held_out_subject_ids"]) == HELD_OUT_IDS
    access_source = TRUTH_ACCESS.read_text(encoding="utf-8")
    assert "_ALLOWED_PURPOSES" in access_source
    assert "return replay_v3_subject_candidate(subject_id, candidate_id)" in access_source


def test_oracle_and_personalization_analysis_remain_unrevealed() -> None:
    final = manifest()
    freeze = load_json(AUDIT / "V3_LANDSCAPE_DATA_FREEZE.json")
    assert final["oracle_revealed_during_generation_stage"] is False
    assert final["oracle_summary_generated"] is False
    assert final["candidate_minimum_rank_topK_regret_cross_transfer_computed"] is False
    assert freeze["oracle_revealed"] is False
    assert not any("ORACLE" in path.name for path in AUDIT.iterdir())
    assert final["next_allowed_stage"] == "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_V1"
    assert final["next_stage_executed"] is False


def test_final_manifest_identity_scope_and_data_hash_are_frozen() -> None:
    final_path = AUDIT / "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1_MANIFEST.json"
    final = load_json(final_path)
    chunks = load_json(AUDIT / "V3_LANDSCAPE_CHUNK_MANIFEST.json")
    assert sha256(final_path) == FINAL_MANIFEST_SHA
    assert final["outcome"] == "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_VALID"
    assert final["global_data_sha256"] == chunks["global_data_sha256"]
    assert final["V3_candidate_manifest_sha256"] == V3_MANIFEST_SHA
    assert tuple(final["development_subject_ids"]) == DEVELOPMENT_IDS
    assert tuple(final["held_out_subject_ids_excluded"]) == HELD_OUT_IDS
    assert final["scope"] == {
        "offline_only": True, "development_only": True, "learner_trained": False,
        "five_parameter": False, "nn_or_pinn": False, "bo": False,
        "robot_or_hardware": False, "not_human_ready": True, "not_robot_approved": True,
    }


def test_no_five_parameter_NN_PINN_BO_or_robot_scope_in_generation_source() -> None:
    protocol = load_json(AUDIT / "V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL.json")
    scope = protocol["scope_guards"]
    assert scope["five_parameter"] is False
    assert scope["nn_or_pinn"] is False
    assert scope["bo"] is False
    assert scope["robot_hardware_control_collection_safety"] is False
    assert scope["candidate_or_cohort_or_objective_change"] is False
    source = SOURCE.read_text(encoding="utf-8")
    assert "POST_FREEZE_ORACLE_SUMMARY" not in source
    assert "np.argmin" not in source


def test_all_formal_artifact_checksums_verify() -> None:
    lines = (AUDIT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 17
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = AUDIT / relative.strip()
        assert path.is_file()
        assert sha256(path) == expected
