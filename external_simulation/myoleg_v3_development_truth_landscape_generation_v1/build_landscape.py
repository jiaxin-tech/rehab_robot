"""Generate the frozen development-only compact MyoLeg-V3 truth landscape.

The protocol, data roles, candidate order, validation subsets, chunking, and
worker-selection rule are frozen before any V3 truth replay.  This stage never
loads held-out scientific arrays and never reveals an oracle or candidate
ranking.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_simulation.myoleg_v2_truth_landscape_generation_v1 import build_truth_landscape as v2infra
from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import parameterization


STAGE_ID = "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_GENERATION_V1"
PROTOCOL_ID = "V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL_V1"
LANDSCAPE_ID = "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1"
OUTCOME_VALID = "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_VALID"
OUTCOME_LIMITED = "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_VALID_WITH_LIMITATIONS"
OUTCOME_INVALID = "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_NOT_VALID"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"

OUTPUT = ROOT / "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1"
DATA = ROOT / "external_simulation/data/myoleg_v3_development_truth_landscape_v1"
SHARDS = DATA / "shards"
PROTOCOL_PATH = OUTPUT / "V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL.json"
ACCESS_POLICY_PATH = OUTPUT / "V3_TRUTH_ACCESS_POLICY_V1.json"
BENCHMARK_PATH = OUTPUT / "V3_PARALLELISM_BENCHMARK.json"
EXECUTION_PLAN_PATH = OUTPUT / "V3_LANDSCAPE_EXECUTION_PLAN.json"
CHUNK_MANIFEST_PATH = OUTPUT / "V3_LANDSCAPE_CHUNK_MANIFEST.json"
FINAL_MANIFEST_PATH = OUTPUT / "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1_MANIFEST.json"
REPLAY_API_VALIDATION_PATH = OUTPUT / "V3_REPLAY_API_VALIDATION.json"

V3_AUDIT = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1"
V3_MANIFEST_PATH = V3_AUDIT / "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
V3_CHECKSUMS_PATH = V3_AUDIT / "checksums.sha256"
V3_TABLE_PATH = V3_AUDIT / "V3_KINEMATIC_CANDIDATE_TABLE.csv"
V3_PARAMETERIZATION_PATH = ROOT / "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py"
COHORT_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
TRUTH_SEMANTICS_PATH = ROOT / "external_simulation_audits/myoleg_reference_trajectory_replay_v1/MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
V2_REFERENCE_PATH = ROOT / "external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv"
V2_CANDIDATE_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
V2_TRUTH_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
V2_TRUTH_CHECKSUMS_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/checksums.sha256"
CANDIDATE_BUILDER_PATH = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER_PATH = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
V2_INFRA_PATH = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/build_truth_landscape.py"

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

FROZEN_SHA = {
    "v3_candidate_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "v3_design_checksums": "b57f07633a06c808cc5945f9c46b62e32b899b4e10911594a2cb05d8b29c0714",
    "v3_candidate_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "v3_parameterization": "e830b5cadd6d970107e59eb9b346650af5ab254b42beecdfaf6b70a5985957ef",
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "v2_candidate_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "v2_truth_manifest": "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    "v2_truth_checksums": "fca04aa4284c2d05a88d943c78dbc22af74a8944b108d2666e77150780c84772",
    "candidate_builder": "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e",
    "replay_builder": "d60a9b1651b49307155b8b36bfdd881b595c604288f7c07a3237afe5f5feb32e",
    "v2_landscape_infrastructure": "269c63a33da4035653de1862d83bc24253be5a384586ff868ee17b12f42746d4",
}

REFERENCE_CANDIDATE_ID = "MYOLEG_V3_K0312"
EXPECTED_SUBJECTS = 24
EXPECTED_CANDIDATES = 625
EXPECTED_PAIRS = 15000
EXPECTED_CHUNKS = 24
REFERENCE_J_TOLERANCE = 1.0e-12
TASK_EXTREMA_TOLERANCE_DEG = 1.0e-3
Q_CLOSURE_MAX_RAD = 1.0e-10
DQ_CLOSURE_MAX_RAD_S = 1.0e-10
DDQ_CLOSURE_MAX_RAD_S2 = 1.0e-9
ABS_LIMIT_TORQUE_MAX_NM = 0.005
REL_LIMIT_CONTRIBUTION_MAX = 0.0005
SOURCE_EQUALITY_RESIDUAL_MAX = 0.001
ALGEBRAIC_RESIDUAL_MAX_NM = 1.0e-8
BENCHMARK_MIN_EFFICIENCY = 0.35
CONTROLLED_THRESHOLDS = {
    "rmse_max_nm": 5.0,
    "p95_max_nm": 10.0,
    "max_abs_max_nm": 65.0,
    "relative_rms_max": 0.20,
}

PAIR_SCHEMA = (
    ("subject_id", "<U13"),
    ("candidate_id", "<U16"),
    ("candidate_index", "<i4"),
    ("beta_flex", "<f8"),
    ("beta_extend", "<f8"),
    ("hip_tau_rms_nm", "<f8"),
    ("knee_tau_rms_nm", "<f8"),
    ("hip_tau_peak_abs_nm", "<f8"),
    ("knee_tau_peak_abs_nm", "<f8"),
    ("subject_reference_hip_rms_nm", "<f8"),
    ("subject_reference_knee_rms_nm", "<f8"),
    ("j_truth", "<f8"),
    ("source_equality_residual_max", "<f8"),
    ("joint_limit_contribution_max_abs_nm", "<f8"),
    ("joint_limit_contribution_max_relative", "<f8"),
    ("solver_warning_count", "<i4"),
    ("joint_limit_active_count_max", "<i2"),
    ("tendon_limit_active_count_max", "<i2"),
    ("contact_active_count_max", "<i2"),
    ("sample_count", "<i2"),
    ("task_invariant_status", "u1"),
    ("integrity_status", "u1"),
)

_CANDIDATE_BUILDER: Any = None
_REPLAY_BUILDER: Any = None
_REFERENCE: dict[str, Any] | None = None
_MODEL_CACHE_ID: str | None = None
_MODEL_CACHE: Any = None
_TRUSTED_DOMAIN: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise RuntimeError(f"cannot infer empty CSV schema: {path}")
    columns = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    v2infra.atomic_text(path, buffer.getvalue())


def frozen_paths() -> dict[str, Path]:
    return {
        "v3_candidate_manifest": V3_MANIFEST_PATH,
        "v3_design_checksums": V3_CHECKSUMS_PATH,
        "v3_candidate_table": V3_TABLE_PATH,
        "v3_parameterization": V3_PARAMETERIZATION_PATH,
        "cohort_manifest": COHORT_MANIFEST_PATH,
        "truth_semantics": TRUTH_SEMANTICS_PATH,
        "v2_reference": V2_REFERENCE_PATH,
        "v2_candidate_manifest": V2_CANDIDATE_MANIFEST_PATH,
        "v2_truth_manifest": V2_TRUTH_MANIFEST_PATH,
        "v2_truth_checksums": V2_TRUTH_CHECKSUMS_PATH,
        "candidate_builder": CANDIDATE_BUILDER_PATH,
        "replay_builder": REPLAY_BUILDER_PATH,
        "v2_landscape_infrastructure": V2_INFRA_PATH,
    }


def verify_checksum_manifest(directory: Path, manifest: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen artifact checksum mismatch: {path}")
        count += 1
    return count


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    actual = {name: sha256_file(path) for name, path in frozen_paths().items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    if verify_checksum_manifest(V3_AUDIT, V3_CHECKSUMS_PATH) < 20:
        raise RuntimeError("V3 design checksum manifest is incomplete")
    if verify_checksum_manifest(V2_TRUTH_MANIFEST_PATH.parent, V2_TRUTH_CHECKSUMS_PATH) < 15:
        raise RuntimeError("V2 landscape checksum manifest is incomplete")
    cohort = read_json(COHORT_MANIFEST_PATH)
    manifest = read_json(V3_MANIFEST_PATH)
    table_rows = read_csv(V3_TABLE_PATH)
    subjects = cohort["subjects"]
    development = [row for row in subjects if row["split"] == "DEVELOPMENT"]
    held_out = [row for row in subjects if row["split"] == "HELD_OUT"]
    if tuple(cohort["development_subject_ids"]) != DEVELOPMENT_IDS or tuple(cohort["held_out_subject_ids"]) != HELD_OUT_IDS:
        raise RuntimeError("frozen cohort split identity changed")
    if [row["subject_id"] for row in development] != list(DEVELOPMENT_IDS):
        raise RuntimeError("development subject order changed")
    if tuple(row["subject_id"] for row in held_out) != HELD_OUT_IDS:
        raise RuntimeError("held-out subject order changed")
    if not (
        len(development) == EXPECTED_SUBJECTS
        and len(held_out) == 8
        and manifest["candidate_count"] == EXPECTED_CANDIDATES
        and manifest["included_candidate_count"] == EXPECTED_CANDIDATES
        and len(table_rows) == EXPECTED_CANDIDATES
        and [row["candidate_id"] for row in table_rows] == manifest["ordered_candidate_ids"]
        and len({row["candidate_id"] for row in table_rows}) == EXPECTED_CANDIDATES
        and all(row["included"] == "True" and row["kinematic_gate_pass"] == "True" for row in table_rows)
    ):
        raise RuntimeError("V3 candidate identity/cardinality changed")
    reference = next((row for row in table_rows if row["candidate_id"] == REFERENCE_CANDIDATE_ID), None)
    if reference is None or int(reference["candidate_index"]) != 312 or float(reference["beta_flex"]) != 0.0 or float(reference["beta_extend"]) != 0.0:
        raise RuntimeError("V3 reference candidate missing or changed")
    semantics = read_json(TRUTH_SEMANTICS_PATH)
    if semantics.get("semantic_version") != TRUTH_SEMANTIC_VERSION or semantics.get("truth_field") != TRUTH_FIELD:
        raise RuntimeError("truth semantics changed")
    return cohort, manifest, [dict(row) for row in table_rows]


def runtime_environment() -> dict[str, Any]:
    return v2infra.runtime_environment()


def held_out_hash_audit(cohort: Mapping[str, Any]) -> dict[str, Any]:
    records = [row for row in cohort["subjects"] if row["subject_id"] in HELD_OUT_IDS]
    if len(records) != 8:
        raise RuntimeError("held-out record count changed")
    checked = []
    for row in records:
        for path_key, sha_key in (
            ("metadata_path", "metadata_sha256"),
            ("model_delta_path", "model_delta_sha256"),
            ("reference_replay_truth_path", "reference_replay_truth_sha256"),
        ):
            path = ROOT / row[path_key]
            actual = sha256_file(path)
            if actual != row[sha_key]:
                raise RuntimeError(f"held-out stream checksum mismatch: {path}")
            checked.append({
                "subject_id": row["subject_id"], "path_role": path_key,
                "path": row[path_key], "size_bytes": path.stat().st_size,
                "stream_sha256": actual,
            })
    return {
        "classification": "SEALED_CONFIRMATORY_TRUTH",
        "held_out_subject_ids": list(HELD_OUT_IDS),
        "held_out_subject_count": 8,
        "allowed_operations": ["file existence", "manifest identity", "file size", "streaming SHA-256"],
        "stream_hashed_file_count": len(checked),
        "stream_hashed_files": checked,
        "np_load_held_out_count": 0,
        "held_out_replay_count": 0,
        "held_out_scientific_access_count": 0,
        "held_out_J_tau_oracle_rank_access_count": 0,
    }


def candidate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "candidate_index": int(row["candidate_index"]),
        "beta_flex": float(row["beta_flex"]),
        "beta_extend": float(row["beta_extend"]),
    }


def validation_plan(v3_manifest: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["candidate_id"]: candidate_record(row) for row in candidates}
    smoke = [dict(row) for row in v3_manifest["smoke_selection"]]
    expected_roles = {
        "REFERENCE", "CORNER_NEG_NEG", "CORNER_NEG_POS", "CORNER_POS_NEG", "CORNER_POS_POS",
        "FLEX_NEG_AXIS", "FLEX_POS_AXIS", "EXTEND_NEG_AXIS", "EXTEND_POS_AXIS",
        "INTERIOR_NEG_NEG", "INTERIOR_NEG_POS", "INTERIOR_POS_NEG", "INTERIOR_POS_POS",
    }
    if {row["selection_role"] for row in smoke} != expected_roles:
        raise RuntimeError("frozen V3 geometry selection roles changed")
    reference = by_id[REFERENCE_CANDIDATE_ID]
    special = [row for row in smoke if row["candidate_id"] != REFERENCE_CANDIDATE_ID]
    detailed_pairs = [
        {"subject_id": subject_id, "candidate_id": REFERENCE_CANDIDATE_ID, "selection_role": "REFERENCE_ALL_SUBJECTS"}
        for subject_id in DEVELOPMENT_IDS
    ]
    detailed_pairs.extend(
        {"subject_id": DEVELOPMENT_IDS[index], "candidate_id": row["candidate_id"], "selection_role": row["selection_role"]}
        for index, row in enumerate(special)
    )
    benchmark = {
        "subject_ids": list(DEVELOPMENT_IDS[:8]),
        "candidate_ids": [REFERENCE_CANDIDATE_ID, "MYOLEG_V3_K0000", "MYOLEG_V3_K0024", "MYOLEG_V3_K0624"],
        "pair_count": 32,
    }
    controlled_pairs = [
        {"subject_id": DEVELOPMENT_IDS[0], "candidate_id": REFERENCE_CANDIDATE_ID, "selection_role": "REFERENCE"},
        {"subject_id": DEVELOPMENT_IDS[-1], "candidate_id": "MYOLEG_V3_K0624", "selection_role": "CORNER_POS_POS"},
    ]
    return {
        "reference_candidate": reference,
        "benchmark": benchmark,
        "detailed_pairs": detailed_pairs,
        "controlled_pairs": controlled_pairs,
        "geometry_candidate_rows": [{**by_id[row["candidate_id"]], "selection_role": row["selection_role"]} for row in smoke],
    }


def freeze_protocol() -> None:
    if OUTPUT.exists() or DATA.exists():
        raise RuntimeError("V3 landscape output/data already exists; refusing overwrite")
    cohort, manifest, candidates = verify_inputs()
    environment = runtime_environment()
    plan = validation_plan(manifest, candidates)
    bytes_per_row = sum(np.dtype(dtype).itemsize for _, dtype in PAIR_SCHEMA)
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_any_V3_truth_replay": True,
        "scientific_role": "HIDDEN_DEVELOPMENT_ONLY_OFFLINE_VIRTUAL_TRUTH_LANDSCAPE",
        "input_sha256": FROZEN_SHA,
        "evaluation_set": {
            "development_subject_ids": list(DEVELOPMENT_IDS),
            "development_subject_count": EXPECTED_SUBJECTS,
            "held_out_subject_ids_excluded": list(HELD_OUT_IDS),
            "nominal_control_included": False,
            "candidate_manifest_sha256": FROZEN_SHA["v3_candidate_manifest"],
            "candidate_count": EXPECTED_CANDIDATES,
            "candidate_ids_in_order": [row["candidate_id"] for row in candidates],
            "reference_candidate_id": REFERENCE_CANDIDATE_ID,
            "expected_pair_count": EXPECTED_PAIRS,
            "subject_or_candidate_replacement_allowed": False,
        },
        "truth": {
            "semantic_version": TRUTH_SEMANTIC_VERSION,
            "semantic_sha256": FROZEN_SHA["truth_semantics"],
            "field": TRUTH_FIELD,
            "primary_method": "prescribed-state inverse-dynamics replay",
            "controlled_method_role": "two-pair consistency diagnostic only; never replaces prescribed truth",
            "rms_quadrature": "sqrt(trapezoid(tau^2,time)/(time[-1]-time[0]))",
            "objective_formula_frozen_unchanged": "sqrt(0.5*((hip_tau_rms/subject_reference_hip_rms)^2+(knee_tau_rms/subject_reference_knee_rms)^2))",
            "reference_J_tolerance": REFERENCE_J_TOLERANCE,
        },
        "pair_integrity_gates": {
            "sample_count": 401,
            "duration_s": 24.0,
            "finite_q_dq_ddq_tau_J": True,
            "task_extrema_ROM_tolerance_deg": TASK_EXTREMA_TOLERANCE_DEG,
            "q_closure_max_rad": Q_CLOSURE_MAX_RAD,
            "dq_closure_max_rad_s": DQ_CLOSURE_MAX_RAD_S,
            "ddq_closure_max_rad_s2": DDQ_CLOSURE_MAX_RAD_S2,
            "solver_warning_count": 0,
            "source_equality_residual_max": SOURCE_EQUALITY_RESIDUAL_MAX,
            "joint_limit_contribution_max_abs_nm": ABS_LIMIT_TORQUE_MAX_NM,
            "joint_limit_contribution_max_relative": REL_LIMIT_CONTRIBUTION_MAX,
            "joint_limit_active_count_max": 1,
            "tendon_limit_active_count": 0,
            "contact_active_count": 0,
            "trusted_ROM_from_frozen_V2_simulator_domain": True,
            "failure_policy": "FAIL_CLOSED_NO_SUBJECT_OR_CANDIDATE_REPLACEMENT",
        },
        "storage": {
            "format": "deterministic compressed NPZ",
            "schema": [{"name": name, "dtype": dtype} for name, dtype in PAIR_SCHEMA],
            "uncompressed_bytes_per_pair": bytes_per_row,
            "estimated_uncompressed_bytes": bytes_per_row * EXPECTED_PAIRS,
            "full_401_sample_bulk_storage": False,
            "chunking": "one atomic shard per development subject in frozen candidate order",
            "chunk_size": EXPECTED_CANDIDATES,
            "chunk_count": EXPECTED_CHUNKS,
            "resume": "skip checksum/schema/identity-valid shard; recompute only missing or invalid shard and record event",
            "deterministic_zip_timestamp": "1980-01-01T00:00:00",
        },
        "parallelism": {
            "implementation": "process-based; each task reconstructs independent model/MjData state",
            "benchmark_worker_counts": [1, 2, 4, 8],
            "benchmark_fixed_set": plan["benchmark"],
            "selection_rule": f"highest-throughput stable deterministic count with efficiency >= {BENCHMARK_MIN_EFFICIENCY}; otherwise highest-throughput stable deterministic count",
            "formal_worker_count": "TO_BE_FROZEN_BY_BENCHMARK_BEFORE_FORMAL_GENERATION",
        },
        "detailed_validation_subset": {
            "selection_frozen_by_subject_and_parameter_geometry_before_truth": True,
            "pairs": plan["detailed_pairs"],
            "pair_count": len(plan["detailed_pairs"]),
            "repeat_prescribed_replay": 2,
            "geometry_candidate_rows": plan["geometry_candidate_rows"],
        },
        "controlled_crosscheck_subset": {
            "pairs": plan["controlled_pairs"],
            "pair_count": len(plan["controlled_pairs"]),
            "selection_frozen_before_truth": True,
            "thresholds": CONTROLLED_THRESHOLDS,
        },
        "oracle_and_analysis": {
            "oracle_reveal_during_generation_stage": False,
            "per_subject_minimum_or_ranking_allowed": False,
            "common_optimum_distinct_oracle_topK_regret_cross_transfer_allowed": False,
            "next_allowed_stage": "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_V1",
        },
        "scope_guards": {
            "development_only": True, "held_out_truth": False, "learner_training": False,
            "five_parameter": False, "nn_or_pinn": False, "bo": False,
            "candidate_or_cohort_or_objective_change": False,
            "robot_hardware_control_collection_safety": False,
        },
        "runtime_environment": environment,
    }
    access_policy = {
        "policy_id": "V3_TRUTH_ACCESS_POLICY_V1",
        "landscape_role": "hidden post-freeze development oracle artifact",
        "landscape_oracle_access_allowed_only_for": ["post-freeze personalization audit", "oracle/regret evaluation under separately frozen protocol"],
        "query_access": "future model/BO may receive only candidates explicitly executed through replay_v3_subject_candidate; bulk compact arrays are not a learner interface",
        "development_generation_stage_oracle_reveal": False,
        "held_out_access": "prohibited until final algorithm freeze; this API rejects all held-out IDs",
        "prohibited": [
            "learner reads unqueried J or torque", "BO reads unqueried candidate outcomes",
            "candidate selection uses landscape rank", "held-out replay/J/torque/oracle/rank",
        ],
        "query_api": "external_simulation/myoleg_v3_development_truth_landscape_generation_v1/replay_api.py",
        "access_boundary": "external_simulation/myoleg_v3_development_truth_landscape_generation_v1/truth_access.py",
    }
    OUTPUT.mkdir(parents=True)
    SHARDS.mkdir(parents=True)
    v2infra.atomic_json(PROTOCOL_PATH, protocol)
    v2infra.atomic_json(ACCESS_POLICY_PATH, access_policy)
    held_out = held_out_hash_audit(cohort)
    held_out["protocol_sha256"] = sha256_file(PROTOCOL_PATH)
    held_out["V3_truth_replay_started_at_audit"] = False
    v2infra.atomic_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json", held_out)
    print(json.dumps({
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "expected_pairs": EXPECTED_PAIRS,
        "detailed_pair_count": len(plan["detailed_pairs"]),
        "controlled_pair_count": len(plan["controlled_pairs"]),
        "held_out_scientific_access_count": 0,
    }, indent=2))


def worker_init() -> None:
    global _CANDIDATE_BUILDER, _REPLAY_BUILDER, _REFERENCE, _TRUSTED_DOMAIN
    _CANDIDATE_BUILDER = v2infra.load_module(CANDIDATE_BUILDER_PATH, "_myoleg_v3_candidate_builder_worker")
    _REPLAY_BUILDER = v2infra.load_module(REPLAY_BUILDER_PATH, "_myoleg_v3_replay_builder_worker")
    _REFERENCE = _CANDIDATE_BUILDER.load_reference_adapter()
    _TRUSTED_DOMAIN = read_json(V2_CANDIDATE_MANIFEST_PATH)["trusted_domain"]


def worker_model(subject: dict[str, Any]) -> Any:
    global _MODEL_CACHE_ID, _MODEL_CACHE
    if _CANDIDATE_BUILDER is None:
        worker_init()
    subject_id = subject["subject_id"]
    if subject_id not in DEVELOPMENT_IDS:
        raise RuntimeError("non-development subject blocked in V3 worker")
    if _MODEL_CACHE_ID != subject_id:
        rebuilt, split, model, _ = _CANDIDATE_BUILDER.model_from_record(subject)
        if rebuilt != subject_id or split != "DEVELOPMENT":
            raise RuntimeError("worker subject reconstruction/split mismatch")
        _MODEL_CACHE_ID = subject_id
        _MODEL_CACHE = model
    return _MODEL_CACHE


def generated_candidate(candidate: Mapping[str, Any]) -> tuple[parameterization.V3Trajectory, dict[str, Any]]:
    if _REFERENCE is None:
        worker_init()
    generated = parameterization.generate_v3_trajectory(
        _REFERENCE, float(candidate["beta_flex"]), float(candidate["beta_extend"])
    )
    replay_reference = {
        "time_s": _REFERENCE["time_s"], "q": generated.q, "dq": generated.dq,
        "ddq": generated.ddq, "phases": _REFERENCE["phases"], "rows": [],
    }
    return generated, replay_reference


def task_invariants(generated: parameterization.V3Trajectory) -> tuple[bool, dict[str, float]]:
    if _REFERENCE is None or _TRUSTED_DOMAIN is None:
        worker_init()
    q_ref = np.asarray(_REFERENCE["q"], dtype=float)
    q, dq, ddq = generated.q, generated.dq, generated.ddq
    q_deg, ref_deg = np.degrees(q), np.degrees(q_ref)
    errors = [
        abs(float(np.min(q_deg[:, joint]) - np.min(ref_deg[:, joint])))
        for joint in (0, 1)
    ] + [
        abs(float(np.max(q_deg[:, joint]) - np.max(ref_deg[:, joint])))
        for joint in (0, 1)
    ] + [
        abs(float(np.ptp(q_deg[:, joint]) - np.ptp(ref_deg[:, joint])))
        for joint in (0, 1)
    ]
    metrics = {
        "max_extrema_rom_error_deg": max(errors),
        "q_closure_error_rad": float(np.max(np.abs(q[-1] - q[0]))),
        "dq_closure_error_rad_s": float(np.max(np.abs(dq[-1] - dq[0]))),
        "ddq_closure_error_rad_s2": float(np.max(np.abs(ddq[-1] - ddq[0]))),
    }
    trusted = bool(
        np.min(q_deg[:, 0]) >= float(_TRUSTED_DOMAIN["trusted_hip_domain_deg"][0]) - 1e-12
        and np.max(q_deg[:, 0]) <= float(_TRUSTED_DOMAIN["trusted_hip_domain_deg"][1]) + 1e-12
        and np.min(q_deg[:, 1]) >= float(_TRUSTED_DOMAIN["trusted_knee_lower_deg"]) - 1e-12
        and np.max(q_deg[:, 1]) <= float(_TRUSTED_DOMAIN["trusted_knee_upper_deg"]) + 1e-12
    )
    passed = bool(
        np.isfinite(np.column_stack((q, dq, ddq))).all()
        and np.array_equal(q[:, 0], q_ref[:, 0])
        and metrics["max_extrema_rom_error_deg"] <= TASK_EXTREMA_TOLERANCE_DEG
        and metrics["q_closure_error_rad"] <= Q_CLOSURE_MAX_RAD
        and metrics["dq_closure_error_rad_s"] <= DQ_CLOSURE_MAX_RAD_S
        and metrics["ddq_closure_error_rad_s2"] <= DDQ_CLOSURE_MAX_RAD_S2
        and len(q) == 401 and float(_REFERENCE["time_s"][-1] - _REFERENCE["time_s"][0]) == 24.0
        and trusted
    )
    metrics["trusted_ROM_valid"] = float(trusted)
    return passed, metrics


def compact_replay(model: Any, candidate: Mapping[str, Any], subject: Mapping[str, Any]) -> dict[str, Any]:
    generated, _ = generated_candidate(candidate)
    task_pass, _ = task_invariants(generated)
    time_s = np.asarray(_REFERENCE["time_s"], dtype=float)
    data = mujoco.MjData(model)
    tau = np.empty((len(time_s), 2), dtype=float)
    warning_max = 0
    equality_max = 0.0
    limit_abs_max = 0.0
    limit_relative_max = 0.0
    joint_count_max = tendon_count_max = contact_count_max = 0
    finite = True
    denominators = np.asarray([
        float(subject["subject_reference_tau_hip_rms_nm"]),
        float(subject["subject_reference_tau_knee_rms_nm"]),
    ])
    for sample in range(len(time_s)):
        _REPLAY_BUILDER.reset_to_target_state(model, data, generated.q[sample], generated.dq[sample], generated.ddq[sample])
        desired_acceleration = np.asarray(data.qacc).copy()
        tangent = _REPLAY_BUILDER.independent_coordinate_tangent(model, data)
        mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
        actuator_internal = np.asarray(data.qfrc_actuator).copy()
        data.qacc[:] = desired_acceleration
        mujoco.mj_inverseSkip(model, data, mujoco.mjtStage.mjSTAGE_VEL, 1)
        required = tangent.T @ (np.asarray(data.qfrc_inverse) - actuator_internal)
        tau[sample] = required
        projected_signed, counts = v2infra.lightweight_constraint_metrics(model, data, tangent)
        projected = np.abs(projected_signed)
        relative = projected / np.maximum(np.abs(required), denominators)
        limit_abs_max = max(limit_abs_max, float(np.max(projected)))
        limit_relative_max = max(limit_relative_max, float(np.max(relative)))
        joint_count_max = max(joint_count_max, int(counts["joint_limit"]))
        tendon_count_max = max(tendon_count_max, int(counts["tendon_limit"]))
        contact_count_max = max(contact_count_max, int(counts["contact"]))
        equality, _ = _REPLAY_BUILDER.source_equality_metrics(model, data)
        equality_max = max(equality_max, float(equality))
        warning_max = max(warning_max, int(_REPLAY_BUILDER.warning_count(data)))
        finite = finite and all(
            bool(np.isfinite(value).all())
            for value in (required, data.qpos, data.qvel, data.qacc, data.qfrc_inverse, data.qfrc_constraint, data.actuator_force, data.ten_length)
        )
    duration = float(time_s[-1] - time_s[0])
    rms = np.sqrt(np.trapezoid(tau**2, time_s, axis=0) / duration)
    peak = np.max(np.abs(tau), axis=0)
    j_truth = float(np.sqrt(0.5 * np.sum((rms / denominators) ** 2)))
    integrity = bool(
        task_pass and finite and np.isfinite(rms).all() and np.isfinite(peak).all() and np.isfinite(j_truth)
        and warning_max == 0 and equality_max <= SOURCE_EQUALITY_RESIDUAL_MAX
        and limit_abs_max <= ABS_LIMIT_TORQUE_MAX_NM and limit_relative_max <= REL_LIMIT_CONTRIBUTION_MAX
        and joint_count_max <= 1 and tendon_count_max == 0 and contact_count_max == 0
        and len(time_s) == 401
    )
    return {
        "hip_tau_rms_nm": float(rms[0]), "knee_tau_rms_nm": float(rms[1]),
        "hip_tau_peak_abs_nm": float(peak[0]), "knee_tau_peak_abs_nm": float(peak[1]),
        "j_truth": j_truth, "source_equality_residual_max": equality_max,
        "joint_limit_contribution_max_abs_nm": limit_abs_max,
        "joint_limit_contribution_max_relative": limit_relative_max,
        "solver_warning_count": warning_max, "joint_limit_active_count_max": joint_count_max,
        "tendon_limit_active_count_max": tendon_count_max, "contact_active_count_max": contact_count_max,
        "sample_count": len(time_s), "task_invariant_status": int(task_pass), "integrity_status": int(integrity),
    }


def empty_arrays(count: int) -> dict[str, np.ndarray]:
    return {name: np.empty(count, dtype=np.dtype(dtype)) for name, dtype in PAIR_SCHEMA}


def subject_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    subject = task["subject"]
    candidates = task["candidates"]
    model = worker_model(subject)
    arrays = empty_arrays(len(candidates))
    for index, candidate in enumerate(candidates):
        metrics = compact_replay(model, candidate, subject)
        arrays["subject_id"][index] = subject["subject_id"]
        arrays["candidate_id"][index] = candidate["candidate_id"]
        arrays["candidate_index"][index] = int(candidate["candidate_index"])
        arrays["beta_flex"][index] = float(candidate["beta_flex"])
        arrays["beta_extend"][index] = float(candidate["beta_extend"])
        arrays["subject_reference_hip_rms_nm"][index] = float(subject["subject_reference_tau_hip_rms_nm"])
        arrays["subject_reference_knee_rms_nm"][index] = float(subject["subject_reference_tau_knee_rms_nm"])
        for key, value in metrics.items():
            arrays[key][index] = value
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        maximum_rss *= 1024
    return {
        "chunk_id": task["chunk_id"], "arrays": arrays,
        "wall_time_s": time.perf_counter() - started,
        "cpu_time_s": time.process_time() - cpu_started,
        "worker_peak_rss_bytes": maximum_rss,
    }


def arrays_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    return v2infra.arrays_fingerprint(dict(arrays))


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {executor.submit(subject_task, task): task for task in tasks}
        for future, task in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"chunk_id": task["chunk_id"], "error_type": type(exc).__name__, "error": str(exc)})
    return sorted(results, key=lambda row: row["chunk_id"]), failures


def benchmark_tasks(cohort: Mapping[str, Any], candidates: list[dict[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    subjects = {row["subject_id"]: row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"}
    by_candidate = {row["candidate_id"]: row for row in candidates}
    fixed = protocol["parallelism"]["benchmark_fixed_set"]
    return [
        {
            "chunk_id": f"BENCH_{subject_id}", "subject": subjects[subject_id],
            "candidates": [by_candidate[candidate_id] for candidate_id in fixed["candidate_ids"]],
        }
        for subject_id in fixed["subject_ids"]
    ]


def run_benchmark() -> None:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("generation protocol must be frozen first")
    if BENCHMARK_PATH.exists() or EXECUTION_PLAN_PATH.exists():
        raise RuntimeError("benchmark/plan already frozen")
    cohort, _, raw_candidates = verify_inputs()
    candidates = [candidate_record(row) for row in raw_candidates]
    protocol = read_json(PROTOCOL_PATH)
    tasks = benchmark_tasks(cohort, candidates, protocol)
    rows = []
    baseline = None
    reference_fingerprint = None
    for workers in protocol["parallelism"]["benchmark_worker_counts"]:
        started = time.perf_counter()
        results, failures = run_tasks(tasks, int(workers))
        wall = time.perf_counter() - started
        fingerprint = canonical_sha([(row["chunk_id"], arrays_fingerprint(row["arrays"])) for row in results])
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        pair_count = sum(len(row["arrays"]["candidate_id"]) for row in results)
        throughput = pair_count / max(wall, 1e-12)
        if baseline is None:
            baseline = throughput
        efficiency = throughput / max(baseline * int(workers), 1e-12)
        row = {
            "workers": int(workers), "pair_count": pair_count, "wall_time_s": wall,
            "throughput_pairs_per_s": throughput, "parallel_efficiency_vs_one": efficiency,
            "aggregate_worker_cpu_time_s": sum(float(item["cpu_time_s"]) for item in results),
            "maximum_worker_peak_rss_bytes": max((int(item["worker_peak_rss_bytes"]) for item in results), default=0),
            "process_failure_count": len(failures), "process_failures": failures,
            "result_fingerprint_sha256": fingerprint,
            "deterministic_equal_to_one_worker": fingerprint == reference_fingerprint,
            "stable": len(failures) == 0 and pair_count == protocol["parallelism"]["benchmark_fixed_set"]["pair_count"] and fingerprint == reference_fingerprint,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    eligible = [row for row in rows if row["stable"] and row["parallel_efficiency_vs_one"] >= BENCHMARK_MIN_EFFICIENCY]
    if not eligible:
        eligible = [row for row in rows if row["stable"]]
    if not eligible:
        raise RuntimeError("no stable deterministic benchmark worker count")
    selected = max(eligible, key=lambda row: (row["throughput_pairs_per_s"], -row["workers"]))
    benchmark = {
        "stage_id": STAGE_ID, "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "fixed_pair_set": protocol["parallelism"]["benchmark_fixed_set"],
        "worker_results": rows, "selection_rule_frozen_before_truth": protocol["parallelism"]["selection_rule"],
        "selected_worker_count": selected["workers"],
    }
    v2infra.atomic_json(BENCHMARK_PATH, benchmark)
    execution = {
        "plan_id": "MYOLEG_V3_DEVELOPMENT_LANDSCAPE_EXECUTION_PLAN_V1",
        "frozen_before_formal_landscape_generation": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "formal_worker_count": selected["workers"],
        "chunk_count": EXPECTED_CHUNKS, "chunk_size": EXPECTED_CANDIDATES,
        "resume_policy": protocol["storage"]["resume"],
    }
    v2infra.atomic_json(EXECUTION_PLAN_PATH, execution)
    print(json.dumps({"selected_worker_count": selected["workers"], "benchmark_sha256": sha256_file(BENCHMARK_PATH), "execution_plan_sha256": sha256_file(EXECUTION_PLAN_PATH)}, indent=2))


def formal_tasks(cohort: Mapping[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subjects = {row["subject_id"]: row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"}
    tasks = [
        {"chunk_id": subject_id, "subject": subjects[subject_id], "candidates": candidates}
        for subject_id in DEVELOPMENT_IDS
    ]
    if len(tasks) != EXPECTED_CHUNKS or sum(len(task["candidates"]) for task in tasks) != EXPECTED_PAIRS:
        raise RuntimeError("formal chunk coverage plan failure")
    return tasks


def chunk_path(chunk_id: str) -> Path:
    return SHARDS / f"{chunk_id}.npz"


def validate_existing(task: Mapping[str, Any]) -> tuple[bool, str]:
    path = chunk_path(str(task["chunk_id"]))
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False, "missing"
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        return False, "checksum_invalid"
    try:
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != {name for name, _ in PAIR_SCHEMA}:
                return False, "schema_invalid"
            candidates = task["candidates"]
            if len(shard["candidate_id"]) != EXPECTED_CANDIDATES:
                return False, "row_count_invalid"
            if not np.array_equal(shard["candidate_id"], np.asarray([row["candidate_id"] for row in candidates])):
                return False, "candidate_identity_invalid"
            if not np.array_equal(shard["candidate_index"], np.asarray([row["candidate_index"] for row in candidates], dtype=np.int32)):
                return False, "candidate_index_invalid"
            if not np.all(shard["subject_id"] == task["subject"]["subject_id"]):
                return False, "subject_identity_invalid"
            if not np.all(shard["integrity_status"] == 1):
                return False, "pair_integrity_invalid"
    except Exception as exc:
        return False, f"load_invalid:{type(exc).__name__}"
    return True, "valid"


def append_resume(event: Mapping[str, Any]) -> None:
    path = OUTPUT / "V3_RESUME_EVENTS.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def generate_landscape() -> None:
    if not EXECUTION_PLAN_PATH.is_file():
        raise RuntimeError("benchmark-frozen execution plan required")
    cohort, _, raw_candidates = verify_inputs()
    candidates = [candidate_record(row) for row in raw_candidates]
    tasks = formal_tasks(cohort, candidates)
    plan = read_json(EXECUTION_PLAN_PATH)
    workers = int(plan["formal_worker_count"])
    pending = []
    valid_existing = invalid_existing = 0
    for task in tasks:
        valid, reason = validate_existing(task)
        if valid:
            valid_existing += 1
        else:
            pending.append(task)
            invalid_existing += int(reason != "missing")
            append_resume({"chunk_id": task["chunk_id"], "action": "RECOMPUTE", "reason": reason})
    print(json.dumps({"valid_existing_chunks": valid_existing, "pending_chunks": len(pending), "invalid_existing_chunks": invalid_existing, "workers": workers}), flush=True)
    started = time.perf_counter()
    completed = 0
    runtime_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
            iterator = iter(pending)
            active: dict[Any, dict[str, Any]] = {}
            for _ in range(min(len(pending), workers * 2)):
                task = next(iterator, None)
                if task is not None:
                    active[executor.submit(subject_task, task)] = task
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        result = future.result()
                        path = chunk_path(task["chunk_id"])
                        digest = v2infra.atomic_npz(path, result["arrays"])
                        append_resume({"chunk_id": task["chunk_id"], "action": "COMPLETED", "sha256": digest, "row_count": len(result["arrays"]["candidate_id"])})
                        runtime_rows.append({
                            "chunk_id": task["chunk_id"], "subject_id": task["subject"]["subject_id"],
                            "row_count": len(result["arrays"]["candidate_id"]), "wall_time_s": result["wall_time_s"],
                            "cpu_time_s": result["cpu_time_s"], "worker_peak_rss_bytes": result["worker_peak_rss_bytes"],
                        })
                        completed += 1
                        elapsed = time.perf_counter() - started
                        print(json.dumps({"completed_subject_shards": completed, "pending_total": len(pending), "elapsed_s": elapsed, "new_pairs_per_s": sum(row["row_count"] for row in runtime_rows) / max(elapsed, 1e-12)}), flush=True)
                    except Exception as exc:
                        failures.append({"chunk_id": task["chunk_id"], "error_type": type(exc).__name__, "error": str(exc)})
                    replacement = next(iterator, None)
                    if replacement is not None:
                        active[executor.submit(subject_task, replacement)] = replacement
    generation = {
        "formal_worker_count": workers, "resume_valid_chunk_count": valid_existing,
        "recomputed_chunk_count": completed, "invalid_existing_chunk_count": invalid_existing,
        "failed_chunk_count": len(failures), "failed_chunks": failures,
        "generation_command_wall_time_s": time.perf_counter() - started,
        "new_chunk_runtime": runtime_rows,
    }
    v2infra.atomic_json(OUTPUT / "V3_GENERATION_RUNTIME_RAW.json", generation)
    if failures:
        raise RuntimeError(f"formal generation failed closed: {failures}")
    freeze_chunk_manifest(tasks, plan)


def freeze_chunk_manifest(tasks: list[dict[str, Any]], plan: Mapping[str, Any]) -> None:
    chunks = []
    failures = []
    for task in tasks:
        valid, reason = validate_existing(task)
        if not valid:
            failures.append({"chunk_id": task["chunk_id"], "reason": reason})
            continue
        path = chunk_path(task["chunk_id"])
        with np.load(path, allow_pickle=False) as shard:
            status = np.asarray(shard["integrity_status"], dtype=np.uint8)
            failure_count = int(np.sum(status != 1))
            if failure_count:
                for index in np.flatnonzero(status != 1):
                    failures.append({"subject_id": str(shard["subject_id"][index]), "candidate_id": str(shard["candidate_id"][index]), "reason": "pair_integrity_failure"})
        chunks.append({
            "chunk_id": task["chunk_id"], "subject_id": task["subject"]["subject_id"],
            "candidate_start_index": 0, "candidate_end_index_exclusive": EXPECTED_CANDIDATES,
            "row_count": EXPECTED_CANDIDATES, "integrity_failure_count": failure_count,
            "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        })
    if failures:
        write_csv(OUTPUT / "V3_LANDSCAPE_FAILURES.csv", failures)
        raise RuntimeError(f"V3_LANDSCAPE_GENERATION_BLOCKED: {len(failures)} failures")
    if len(chunks) != EXPECTED_CHUNKS or sum(row["row_count"] for row in chunks) != EXPECTED_PAIRS:
        raise RuntimeError("V3 landscape coverage mismatch")
    manifest = {
        "manifest_id": "V3_LANDSCAPE_CHUNK_MANIFEST_V1", "stage_id": STAGE_ID,
        "created_before_validation_and_oracle_reveal": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "execution_plan_sha256": sha256_file(EXECUTION_PLAN_PATH),
        "formal_worker_count": int(plan["formal_worker_count"]),
        "expected_pair_count": EXPECTED_PAIRS, "actual_pair_count": EXPECTED_PAIRS,
        "chunk_count": EXPECTED_CHUNKS, "chunk_size": EXPECTED_CANDIDATES,
        "chunks": chunks,
        "subject_landscape_sha256": {row["subject_id"]: row["sha256"] for row in chunks},
        "global_data_sha256": canonical_sha([(row["path"], row["sha256"]) for row in chunks]),
        "all_pair_integrity_pass": True,
        "oracle_revealed": False,
    }
    v2infra.atomic_json(CHUNK_MANIFEST_PATH, manifest)
    v2infra.atomic_json(OUTPUT / "V3_LANDSCAPE_DATA_FREEZE.json", {
        "freeze_id": "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_DATA_FREEZE_V1",
        "chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST_PATH),
        "global_data_sha256": manifest["global_data_sha256"],
        "pair_count": EXPECTED_PAIRS, "oracle_revealed": False,
    })
    print(json.dumps({"chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST_PATH), "global_data_sha256": manifest["global_data_sha256"], "pairs": EXPECTED_PAIRS, "oracle_revealed": False}, indent=2))


def array_payload_sha(arrays: Mapping[str, np.ndarray]) -> str:
    return arrays_fingerprint({key: np.asarray(value) for key, value in arrays.items()})


def detailed_task(task: dict[str, Any]) -> dict[str, Any]:
    subject = task["subject"]
    candidate = task["candidate"]
    model = worker_model(subject)
    generated, replay_reference = generated_candidate(candidate)
    first, first_runtime = _REPLAY_BUILDER.prescribed_truth(model, replay_reference)
    second, second_runtime = _REPLAY_BUILDER.prescribed_truth(model, replay_reference)
    compact = compact_replay(model, candidate, subject)
    tau = np.asarray(first["tau_truth_nm"], dtype=float)
    time_s = np.asarray(_REFERENCE["time_s"], dtype=float)
    rms = np.sqrt(np.trapezoid(tau**2, time_s, axis=0) / (time_s[-1] - time_s[0]))
    denominators = np.asarray([
        subject["subject_reference_tau_hip_rms_nm"], subject["subject_reference_tau_knee_rms_nm"]
    ], dtype=float)
    j_truth = float(np.sqrt(0.5 * np.sum((rms / denominators) ** 2)))
    repeat_equal = set(first) == set(second) and all(np.array_equal(first[key], second[key]) for key in first)
    residual = max(
        float(np.max(np.abs(first[key])))
        for key in ("inverse_formula_residual_nm", "decomposition_residual_nm", "muscle_reconstruction_residual_nm")
    )
    task_pass, invariant = task_invariants(generated)
    warnings = int(np.max(first["warning_count"]))
    equality = float(np.max(np.abs(first["source_equality_residual"])))
    joint_count = int(np.max(first["constraint_joint_limit_active_count"]))
    tendon_count = int(np.max(first["constraint_tendon_limit_active_count"]))
    contact_count = int(np.max(first["constraint_contact_active_count"]))
    compact_j_error = abs(j_truth - float(compact["j_truth"]))
    compact_hip_error = abs(float(rms[0]) - float(compact["hip_tau_rms_nm"]))
    compact_knee_error = abs(float(rms[1]) - float(compact["knee_tau_rms_nm"]))
    passed = bool(
        repeat_equal and task_pass and np.isfinite(tau).all() and np.isfinite(j_truth)
        and residual <= ALGEBRAIC_RESIDUAL_MAX_NM and equality <= SOURCE_EQUALITY_RESIDUAL_MAX
        and warnings == 0 and joint_count <= 1 and tendon_count == 0 and contact_count == 0
        and compact_j_error <= 1e-12 and compact_hip_error <= 1e-12 and compact_knee_error <= 1e-12
        and bool(compact["integrity_status"])
    )
    return {
        "subject_id": subject["subject_id"], "split": subject["split"],
        "candidate_id": candidate["candidate_id"], "candidate_index": int(candidate["candidate_index"]),
        "beta_flex": float(candidate["beta_flex"]), "beta_extend": float(candidate["beta_extend"]),
        "selection_role": task["selection_role"], "sample_count": len(tau),
        "hip_tau_rms_nm": float(rms[0]), "knee_tau_rms_nm": float(rms[1]), "j_truth": j_truth,
        "compact_vs_full_j_abs_error": compact_j_error,
        "compact_vs_full_hip_rms_abs_error_nm": compact_hip_error,
        "compact_vs_full_knee_rms_abs_error_nm": compact_knee_error,
        "inverse_formula_residual_max_abs_nm": float(np.max(np.abs(first["inverse_formula_residual_nm"]))),
        "decomposition_residual_max_abs_nm": float(np.max(np.abs(first["decomposition_residual_nm"]))),
        "muscle_reconstruction_residual_max_abs_nm": float(np.max(np.abs(first["muscle_reconstruction_residual_nm"]))),
        "source_equality_residual_max": equality,
        "solver_warning_count": warnings, "joint_limit_active_count_max": joint_count,
        "tendon_limit_active_count_max": tendon_count, "contact_active_count_max": contact_count,
        "max_extrema_rom_error_deg": invariant["max_extrema_rom_error_deg"],
        "q_closure_error_rad": invariant["q_closure_error_rad"],
        "dq_closure_error_rad_s": invariant["dq_closure_error_rad_s"],
        "ddq_closure_error_rad_s2": invariant["ddq_closure_error_rad_s2"],
        "trusted_ROM_valid": bool(invariant["trusted_ROM_valid"]),
        "prescribed_repeat_array_equal": repeat_equal,
        "first_array_payload_sha256": array_payload_sha(first),
        "second_array_payload_sha256": array_payload_sha(second),
        "first_prescribed_wall_time_s": float(first_runtime["wall_time_s"]),
        "second_prescribed_wall_time_s": float(second_runtime["wall_time_s"]),
        "detailed_integrity_pass": passed,
    }


def controlled_task(task: dict[str, Any]) -> dict[str, Any]:
    subject = task["subject"]
    candidate = task["candidate"]
    model = worker_model(subject)
    _, replay_reference = generated_candidate(candidate)
    prescribed, prescribed_runtime = _REPLAY_BUILDER.prescribed_truth(model, replay_reference)
    controlled, controlled_runtime = _REPLAY_BUILDER.controlled_replay(model, replay_reference)
    rows = []
    controlled_warning = int(np.max(controlled["warning_count"]))
    for joint, label in enumerate(("hip", "knee")):
        truth = np.asarray(prescribed["tau_truth_nm"][:, joint], dtype=float)
        reconstructed = np.asarray(controlled["force_balance_reconstruction_nm"][:, joint], dtype=float)
        difference = reconstructed - truth
        truth_rms = float(np.sqrt(np.mean(truth**2)))
        rmse = float(np.sqrt(np.mean(difference**2)))
        p95 = float(np.percentile(np.abs(difference), 95.0))
        maximum = float(np.max(np.abs(difference)))
        relative = rmse / max(truth_rms, 1e-12)
        passed = bool(
            np.isfinite(truth).all() and np.isfinite(reconstructed).all() and controlled_warning == 0
            and rmse <= CONTROLLED_THRESHOLDS["rmse_max_nm"]
            and p95 <= CONTROLLED_THRESHOLDS["p95_max_nm"]
            and maximum <= CONTROLLED_THRESHOLDS["max_abs_max_nm"]
            and relative <= CONTROLLED_THRESHOLDS["relative_rms_max"]
        )
        rows.append({
            "subject_id": subject["subject_id"], "split": subject["split"],
            "candidate_id": candidate["candidate_id"], "candidate_index": int(candidate["candidate_index"]),
            "beta_flex": float(candidate["beta_flex"]), "beta_extend": float(candidate["beta_extend"]),
            "selection_role": task["selection_role"], "joint": label,
            "difference_rmse_nm": rmse, "difference_p95_abs_nm": p95,
            "difference_max_abs_nm": maximum, "difference_relative_rms": relative,
            "correlation": float(np.corrcoef(reconstructed, truth)[0, 1]),
            "controlled_solver_warning_count": controlled_warning,
            "prescribed_wall_time_s": float(prescribed_runtime["wall_time_s"]),
            "controlled_wall_time_s": float(controlled_runtime["wall_time_s"]),
            "controlled_consistency_pass": passed,
        })
    return {"rows": rows}


def run_generic(function: Any, tasks: list[dict[str, Any]], workers: int) -> list[Any]:
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {executor.submit(function, task): task for task in tasks}
        for future, task in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"validation failed closed {task['subject']['subject_id']} x {task['candidate']['candidate_id']}: {exc}"
                ) from exc
    return results


def load_subject_shard(subject_id: str, chunk_manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    row = next(item for item in chunk_manifest["chunks"] if item["subject_id"] == subject_id)
    with np.load(ROOT / row["path"], allow_pickle=False) as shard:
        return {key: np.asarray(shard[key]).copy() for key in shard.files}


def validate_complete_without_oracle(
    cohort: Mapping[str, Any], candidates: list[dict[str, Any]], chunk_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_ids = np.asarray([row["candidate_id"] for row in candidates])
    candidate_indexes = np.asarray([row["candidate_index"] for row in candidates], dtype=np.int32)
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0
    reference_rows = []
    finite_fields = [
        "beta_flex", "beta_extend", "hip_tau_rms_nm", "knee_tau_rms_nm",
        "hip_tau_peak_abs_nm", "knee_tau_peak_abs_nm", "subject_reference_hip_rms_nm",
        "subject_reference_knee_rms_nm", "j_truth", "source_equality_residual_max",
        "joint_limit_contribution_max_abs_nm", "joint_limit_contribution_max_relative",
    ]
    subjects = {row["subject_id"]: row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"}
    all_finite = all_integrity = all_task = True
    for subject_id in DEVELOPMENT_IDS:
        arrays = load_subject_shard(subject_id, chunk_manifest)
        if not np.array_equal(arrays["candidate_id"], candidate_ids) or not np.array_equal(arrays["candidate_index"], candidate_indexes):
            raise RuntimeError(f"candidate identity/order mismatch for {subject_id}")
        if not np.all(arrays["subject_id"] == subject_id):
            raise RuntimeError(f"subject identity mismatch in shard {subject_id}")
        all_finite = all_finite and all(bool(np.isfinite(arrays[key]).all()) for key in finite_fields)
        all_integrity = all_integrity and bool(np.all(arrays["integrity_status"] == 1))
        all_task = all_task and bool(np.all(arrays["task_invariant_status"] == 1))
        for candidate_id in arrays["candidate_id"]:
            pair = (subject_id, str(candidate_id))
            duplicate_count += int(pair in seen)
            seen.add(pair)
        reference_indexes = np.flatnonzero(arrays["candidate_id"] == REFERENCE_CANDIDATE_ID)
        if len(reference_indexes) != 1:
            raise RuntimeError(f"reference multiplicity failure for {subject_id}")
        index = int(reference_indexes[0])
        reference_j = float(arrays["j_truth"][index])
        reference_rows.append({
            "subject_id": subject_id, "split": "DEVELOPMENT",
            "reference_candidate_id": REFERENCE_CANDIDATE_ID,
            "reference_candidate_index": int(arrays["candidate_index"][index]),
            "subject_reference_hip_rms_nm": float(subjects[subject_id]["subject_reference_tau_hip_rms_nm"]),
            "subject_reference_knee_rms_nm": float(subjects[subject_id]["subject_reference_tau_knee_rms_nm"]),
            "reference_hip_tau_rms_nm": float(arrays["hip_tau_rms_nm"][index]),
            "reference_knee_tau_rms_nm": float(arrays["knee_tau_rms_nm"][index]),
            "reference_j_truth": reference_j,
            "abs_error_from_one": abs(reference_j - 1.0),
            "within_frozen_tolerance": abs(reference_j - 1.0) <= REFERENCE_J_TOLERANCE,
        })
    summary = {
        "actual_unique_pair_count": len(seen), "duplicate_pair_count": duplicate_count,
        "missing_pair_count": EXPECTED_PAIRS - len(seen),
        "all_scalar_values_finite": all_finite,
        "all_pair_integrity_pass": all_integrity,
        "all_pair_task_invariants_pass": all_task,
        "all_subject_reference_j_within_tolerance": all(row["within_frozen_tolerance"] for row in reference_rows),
        "oracle_minimum_rank_or_preference_computed": False,
    }
    if summary != {
        "actual_unique_pair_count": EXPECTED_PAIRS, "duplicate_pair_count": 0, "missing_pair_count": 0,
        "all_scalar_values_finite": True, "all_pair_integrity_pass": True,
        "all_pair_task_invariants_pass": True, "all_subject_reference_j_within_tolerance": True,
        "oracle_minimum_rank_or_preference_computed": False,
    }:
        raise RuntimeError(f"complete landscape validation failed closed: {summary}")
    return reference_rows, summary


def finalize_landscape() -> None:
    if not CHUNK_MANIFEST_PATH.is_file():
        raise RuntimeError("complete pre-validation chunk manifest required")
    if not REPLAY_API_VALIDATION_PATH.is_file():
        raise RuntimeError("deterministic development-only replay API validation required")
    cohort, v3_manifest, raw_candidates = verify_inputs()
    candidates = [candidate_record(row) for row in raw_candidates]
    protocol = read_json(PROTOCOL_PATH)
    execution = read_json(EXECUTION_PLAN_PATH)
    chunks = read_json(CHUNK_MANIFEST_PATH)
    if not (
        chunks["actual_pair_count"] == EXPECTED_PAIRS
        and chunks["all_pair_integrity_pass"]
        and chunks["created_before_validation_and_oracle_reveal"]
        and chunks["oracle_revealed"] is False
    ):
        raise RuntimeError("chunk landscape is not complete and oracle-sealed")
    subject_map = {row["subject_id"]: row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"}
    candidate_map = {row["candidate_id"]: row for row in candidates}
    detailed_tasks = [
        {
            "subject": subject_map[row["subject_id"]], "candidate": candidate_map[row["candidate_id"]],
            "selection_role": row["selection_role"],
        }
        for row in protocol["detailed_validation_subset"]["pairs"]
    ]
    workers = int(execution["formal_worker_count"])
    detailed_rows = run_generic(detailed_task, detailed_tasks, workers)
    detailed_rows.sort(key=lambda row: (row["subject_id"], row["candidate_index"]))
    write_csv(OUTPUT / "V3_DETAILED_VALIDATION_RESULTS.csv", detailed_rows)
    if not all(bool(row["detailed_integrity_pass"]) for row in detailed_rows):
        raise RuntimeError("detailed prescribed-repeat validation failed closed")

    controlled_tasks = [
        {
            "subject": subject_map[row["subject_id"]], "candidate": candidate_map[row["candidate_id"]],
            "selection_role": row["selection_role"],
        }
        for row in protocol["controlled_crosscheck_subset"]["pairs"]
    ]
    controlled_results = run_generic(controlled_task, controlled_tasks, min(workers, len(controlled_tasks)))
    controlled_rows = [row for result in controlled_results for row in result["rows"]]
    controlled_rows.sort(key=lambda row: (row["subject_id"], row["candidate_index"], row["joint"]))
    write_csv(OUTPUT / "V3_CONTROLLED_CROSSCHECK_RESULTS.csv", controlled_rows)
    controlled_pass = all(bool(row["controlled_consistency_pass"]) for row in controlled_rows)

    reference_rows, complete = validate_complete_without_oracle(cohort, candidates, chunks)
    write_csv(OUTPUT / "V3_REFERENCE_NORMALIZATION_AUDIT.csv", reference_rows)
    generation = read_json(OUTPUT / "V3_GENERATION_RUNTIME_RAW.json")
    shard_bytes = sum(int(row["size_bytes"]) for row in chunks["chunks"])
    sidecar_bytes = sum((ROOT / row["path"]).with_suffix(".npz.sha256").stat().st_size for row in chunks["chunks"])
    runtime_storage = {
        "formal_worker_count": workers,
        "generation_command_wall_time_s": float(generation["generation_command_wall_time_s"]),
        "aggregate_new_shard_worker_wall_time_s": sum(float(row["wall_time_s"]) for row in generation["new_chunk_runtime"]),
        "aggregate_new_shard_cpu_time_s": sum(float(row["cpu_time_s"]) for row in generation["new_chunk_runtime"]),
        "actual_generation_throughput_pairs_per_s": EXPECTED_PAIRS / max(float(generation["generation_command_wall_time_s"]), 1e-12),
        "compact_shard_count": EXPECTED_CHUNKS, "compact_shard_bytes": shard_bytes,
        "checksum_sidecar_bytes": sidecar_bytes,
        "total_compact_data_and_sidecar_bytes": shard_bytes + sidecar_bytes,
        "uncompressed_schema_bytes_per_pair": sum(np.dtype(dtype).itemsize for _, dtype in PAIR_SCHEMA),
        "estimated_uncompressed_landscape_bytes": sum(np.dtype(dtype).itemsize for _, dtype in PAIR_SCHEMA) * EXPECTED_PAIRS,
        "bulk_full_time_series_generated": False,
        "full_401_sample_time_series_reconstructable_on_demand": True,
    }
    v2infra.atomic_json(OUTPUT / "V3_RUNTIME_STORAGE_AUDIT.json", runtime_storage)

    outcome = OUTCOME_VALID if controlled_pass else OUTCOME_LIMITED
    access = read_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json")
    final_manifest = {
        "manifest_id": LANDSCAPE_ID, "stage_id": STAGE_ID, "outcome": outcome,
        "V3_candidate_manifest_sha256": FROZEN_SHA["v3_candidate_manifest"],
        "V3_candidate_table_sha256": FROZEN_SHA["v3_candidate_table"],
        "cohort_manifest_sha256": FROZEN_SHA["cohort_manifest"],
        "V2_frozen_truth_manifest_sha256": FROZEN_SHA["v2_truth_manifest"],
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_semantic_sha256": FROZEN_SHA["truth_semantics"],
        "truth_field": TRUTH_FIELD,
        "reference_identity": {
            "candidate_id": REFERENCE_CANDIDATE_ID, "candidate_index": 312,
            "beta_flex": 0.0, "beta_extend": 0.0,
            "reference_sha256": FROZEN_SHA["v2_reference"],
        },
        "development_subject_ids": list(DEVELOPMENT_IDS),
        "held_out_subject_ids_excluded": list(HELD_OUT_IDS),
        "development_subject_count": EXPECTED_SUBJECTS,
        "candidate_count": EXPECTED_CANDIDATES,
        "expected_pair_count": EXPECTED_PAIRS,
        "actual_pair_count": complete["actual_unique_pair_count"],
        "duplicate_pair_count": complete["duplicate_pair_count"],
        "missing_pair_count": complete["missing_pair_count"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parallelism_benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "execution_plan_sha256": sha256_file(EXECUTION_PLAN_PATH),
        "chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST_PATH),
        "global_data_sha256": chunks["global_data_sha256"],
        "chunks": chunks["chunks"],
        "subject_landscape_sha256": chunks["subject_landscape_sha256"],
        "formal_worker_count": workers,
        "storage_schema": protocol["storage"]["schema"],
        "integrity_summary": {
            **complete,
            "detailed_validation_pair_count": len(detailed_rows),
            "detailed_validation_pass": True,
            "controlled_crosscheck_pair_count": len(controlled_tasks),
            "controlled_crosscheck_joint_row_count": len(controlled_rows),
            "controlled_crosscheck_pass": controlled_pass,
        },
        "held_out_scientific_access_count": access["held_out_scientific_access_count"],
        "oracle_revealed_during_generation_stage": False,
        "oracle_summary_generated": False,
        "candidate_minimum_rank_topK_regret_cross_transfer_computed": False,
        "scope": {
            "offline_only": True, "development_only": True, "learner_trained": False,
            "five_parameter": False, "nn_or_pinn": False, "bo": False,
            "robot_or_hardware": False, "not_human_ready": True, "not_robot_approved": True,
        },
        "runtime_storage_audit_sha256": sha256_file(OUTPUT / "V3_RUNTIME_STORAGE_AUDIT.json"),
        "deterministic_replay_api_validation_sha256": sha256_file(REPLAY_API_VALIDATION_PATH),
        "next_allowed_stage": "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_V1",
        "next_stage_executed": False,
    }
    v2infra.atomic_json(FINAL_MANIFEST_PATH, final_manifest)
    write_report(final_manifest, runtime_storage, controlled_rows)
    write_metadata_and_checksums(final_manifest)
    print(json.dumps({
        "outcome": outcome, "final_manifest_sha256": sha256_file(FINAL_MANIFEST_PATH),
        "pairs": EXPECTED_PAIRS, "controlled_pass": controlled_pass,
        "held_out_scientific_access_count": 0, "oracle_revealed": False,
    }, indent=2))


def write_report(manifest: Mapping[str, Any], runtime: Mapping[str, Any], controlled: list[dict[str, Any]]) -> None:
    integrity = manifest["integrity_summary"]
    text = f"""# MyoLeg V3 Development Truth Landscape Generation V1

## Decision

`{manifest['outcome']}`

This is a hidden **development-only offline virtual truth landscape**. It is not a human result, robot-motion approval, clinical result, learner training set, personalization conclusion, or safety validation.

## Frozen identity and coverage

- V3 candidate manifest SHA-256: `{manifest['V3_candidate_manifest_sha256']}`.
- Cohort manifest SHA-256: `{manifest['cohort_manifest_sha256']}`.
- Truth semantic: `{TRUTH_SEMANTIC_VERSION}` / `{TRUTH_FIELD}`.
- Development subjects: `{manifest['development_subject_count']}`; candidates: `{manifest['candidate_count']}`.
- Evaluated pairs: `{manifest['actual_pair_count']:,}` / `{manifest['expected_pair_count']:,}`.
- Duplicate pairs: `{manifest['duplicate_pair_count']}`; missing pairs: `{manifest['missing_pair_count']}`.
- All 24 reference objectives are within `{REFERENCE_J_TOLERANCE:g}` of 1: `{integrity['all_subject_reference_j_within_tolerance']}`.

## Compact storage and runtime

The frozen dataset contains `{runtime['compact_shard_count']}` deterministic compressed NPZ subject shards. It stores only one-dimensional compact scalar columns and occupies `{runtime['compact_shard_bytes']:,}` bytes plus `{runtime['checksum_sidecar_bytes']:,}` bytes of checksum sidecars. No bulk 401-point replay arrays were stored; full prescribed arrays remain reproducible through the development-only on-demand replay API.

The on-demand API was invoked twice for the frozen development reference pair and returned array-identical full payloads; a held-out ID was rejected before simulator replay.

- Formal workers: `{runtime['formal_worker_count']}`.
- Formal generation wall time: `{runtime['generation_command_wall_time_s']:.3f}` s.
- Throughput: `{runtime['actual_generation_throughput_pairs_per_s']:.3f}` pairs/s.

## Integrity validation

- All-pair compact integrity: `{integrity['all_pair_integrity_pass']}`.
- All-pair task invariants: `{integrity['all_pair_task_invariants_pass']}`.
- Detailed repeated prescribed replays: `{integrity['detailed_validation_pair_count']}` / pass `{integrity['detailed_validation_pass']}`.
- Controlled diagnostic: `{integrity['controlled_crosscheck_pair_count']}` pairs and `{integrity['controlled_crosscheck_joint_row_count']}` joint rows / pass `{integrity['controlled_crosscheck_pass']}`.

Prescribed replay remains primary truth. Controlled replay was used only for the two frozen consistency diagnostics and did not replace or alter any compact truth value.

## Information boundary

Held-out scientific truth access was exactly `0`. No held-out replay, J, torque, oracle, rank, or candidate preference was read. No per-subject minimum, candidate ranking, common optimum, distinct-oracle count, Top-K overlap, regret, cross-transfer, or V2/V3 performance comparison was computed. `ORACLE_NOT_REVEALED_DURING_GENERATION_STAGE = true`.

## Boundaries and next step

No Five-parameter model, NN/PINN, BO, cohort/domain/objective modification, hardware/control/collection/safety change, or personalization interpretation occurred. The only allowed next stage is `MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_V1`, under a separately frozen protocol; it was not executed here.
"""
    v2infra.atomic_text(OUTPUT / "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_GENERATION_REPORT.md", text)


def write_metadata_and_checksums(manifest: Mapping[str, Any]) -> None:
    artifact_paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name not in {"checksums.sha256", "metadata.json"})
    metadata = {
        "stage_id": STAGE_ID, "outcome": manifest["outcome"],
        "formal_manifest_sha256": sha256_file(FINAL_MANIFEST_PATH),
        "artifact_count_excluding_metadata_and_checksums": len(artifact_paths),
        "data_directory": str(DATA.relative_to(ROOT)),
        "data_global_sha256": manifest["global_data_sha256"],
        "held_out_scientific_access_count": 0, "oracle_revealed": False,
        "offline_only": True, "development_only": True,
        "not_human_ready": True, "not_robot_approved": True,
        "frozen_inputs_before": FROZEN_SHA,
        "frozen_inputs_after": {name: sha256_file(path) for name, path in frozen_paths().items()},
    }
    v2infra.atomic_json(OUTPUT / "metadata.json", metadata)
    paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    v2infra.atomic_text(OUTPUT / "checksums.sha256", "\n".join(lines) + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--generate", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--all", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.freeze_protocol:
        freeze_protocol()
    elif args.benchmark:
        run_benchmark()
    elif args.generate:
        generate_landscape()
    elif args.finalize:
        finalize_landscape()
    else:
        freeze_protocol(); run_benchmark(); generate_landscape(); finalize_landscape()


if __name__ == "__main__":
    main()
