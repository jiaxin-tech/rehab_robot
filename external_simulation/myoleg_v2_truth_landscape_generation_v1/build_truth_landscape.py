"""Build the frozen compact MyoLeg-V2 prescribed-replay truth landscape.

The implementation is intentionally offline-only.  It uses deterministic
subject/candidate chunks, process isolation, atomic writes, checksum-aware
resume, and an explicit pre-oracle landscape freeze.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Iterable
import zipfile

import mujoco
import numpy as np


STAGE_ID = "MYOLEG_V2_TRUTH_LANDSCAPE_GENERATION_V1"
PROTOCOL_ID = "MYOLEG_V2_TRUTH_LANDSCAPE_GENERATION_PROTOCOL_V1"
LANDSCAPE_ID = "MYOLEG_V2_TRUTH_LANDSCAPE_V1"
OUTCOME_VALID = "MYOLEG_V2_TRUTH_LANDSCAPE_VALID"
OUTCOME_LIMITED = "MYOLEG_V2_TRUTH_LANDSCAPE_VALID_WITH_LIMITATIONS"
OUTCOME_INVALID = "MYOLEG_V2_TRUTH_LANDSCAPE_NOT_VALID"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1"
DATA = ROOT / "external_simulation/data/myoleg_v2_truth_landscape_v1"
SHARDS = DATA / "shards"
COHORT_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
CANDIDATE_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
CANDIDATE_ADMISSION_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/V2_CANDIDATE_ADMISSION.csv"
CANDIDATE_BUILDER_PATH = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER_PATH = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
REPLAY_API_PATH = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/replay_api.py"
TRUTH_ACCESS_PATH = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/truth_access.py"
TRUTH_SEMANTICS_PATH = ROOT / "external_simulation_audits/myoleg_reference_trajectory_replay_v1/MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
V2_REFERENCE_PATH = ROOT / "external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv"

PROTOCOL_PATH = OUTPUT / "LANDSCAPE_GENERATION_PROTOCOL.json"
ACCESS_POLICY_PATH = OUTPUT / "TRUTH_ACCESS_POLICY_V1.json"
BENCHMARK_PATH = OUTPUT / "PARALLELISM_BENCHMARK.json"
EXECUTION_PLAN_PATH = OUTPUT / "LANDSCAPE_EXECUTION_PLAN.json"
CHUNK_MANIFEST_PATH = OUTPUT / "LANDSCAPE_CHUNK_MANIFEST.json"
FINAL_MANIFEST_PATH = OUTPUT / "MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"

FROZEN_SHA = {
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "candidate_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
}
REFERENCE_CANDIDATE_ID = "MYOLEG_V2_P15012"
EXPECTED_SUBJECTS = 32
EXPECTED_CANDIDATES = 16675
EXPECTED_PAIRS = 533600
CHUNK_SIZE = 250
EXPECTED_CHUNKS_PER_SUBJECT = math.ceil(EXPECTED_CANDIDATES / CHUNK_SIZE)
EXPECTED_CHUNKS = EXPECTED_SUBJECTS * EXPECTED_CHUNKS_PER_SUBJECT
REFERENCE_J_TOLERANCE = 1.0e-12
ORACLE_EQUIVALENCE_TOLERANCE = 1.0e-12
BENCHMARK_MIN_EFFICIENCY = 0.35
ABS_LIMIT_TORQUE_MAX_NM = 0.005
REL_LIMIT_CONTRIBUTION_MAX = 0.0005
SOURCE_EQUALITY_RESIDUAL_MAX = 0.001
ALGEBRAIC_RESIDUAL_MAX_NM = 1.0e-8
CONTROLLED_THRESHOLDS = {
    "rmse_max_nm": 5.0,
    "p95_max_nm": 10.0,
    "max_abs_max_nm": 65.0,
    "relative_rms_max": 0.20,
}

PAIR_SCHEMA = (
    ("subject_id", "<U13"),
    ("candidate_id", "<U16"),
    ("proposal_index", "<i4"),
    ("alpha_hip_deg", "<f8"),
    ("alpha_knee_deg", "<f8"),
    ("alpha_phase", "<f8"),
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
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise RuntimeError(f"cannot infer empty CSV schema: {path}")
    columns = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = deterministic_npz_bytes(arrays)
    digest = hashlib.sha256(payload).hexdigest()
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    actual = {
        "cohort_manifest": sha256_file(COHORT_MANIFEST_PATH),
        "candidate_manifest": sha256_file(CANDIDATE_MANIFEST_PATH),
        "v2_reference": sha256_file(V2_REFERENCE_PATH),
        "truth_semantics": sha256_file(TRUTH_SEMANTICS_PATH),
    }
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    cohort = read_json(COHORT_MANIFEST_PATH)
    candidates = read_json(CANDIDATE_MANIFEST_PATH)
    subject_rows = cohort["subjects"]
    candidate_rows = candidates["ordered_included_candidates"]
    if not (
        cohort["cohort_size"] == EXPECTED_SUBJECTS
        and cohort["development_count"] == 24
        and cohort["held_out_count"] == 8
        and len(subject_rows) == EXPECTED_SUBJECTS
        and len({row["subject_id"] for row in subject_rows}) == EXPECTED_SUBJECTS
        and candidates["admissible_candidate_count"] == EXPECTED_CANDIDATES
        and len(candidate_rows) == EXPECTED_CANDIDATES
        and len({row["candidate_id"] for row in candidate_rows}) == EXPECTED_CANDIDATES
        and len({int(row["proposal_index"]) for row in candidate_rows}) == EXPECTED_CANDIDATES
        and EXPECTED_SUBJECTS * EXPECTED_CANDIDATES == EXPECTED_PAIRS
    ):
        raise RuntimeError("frozen evaluation-set cardinality/uniqueness failure")
    reference = next((row for row in candidate_rows if row["candidate_id"] == REFERENCE_CANDIDATE_ID), None)
    if reference is None or int(reference["proposal_index"]) != 15012 or list(reference["alpha"]) != [0.0, 0.0, 0.0]:
        raise RuntimeError("frozen reference candidate missing or changed")
    truth = read_json(TRUTH_SEMANTICS_PATH)
    if truth.get("semantic_version") != TRUTH_SEMANTIC_VERSION or truth.get("truth_field") != TRUTH_FIELD:
        raise RuntimeError("frozen truth semantics changed")
    return cohort, candidates


def runtime_environment() -> dict[str, Any]:
    values = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "myosuite": importlib.metadata.version("myosuite"),
        "logical_cpu_count": os.cpu_count(),
    }
    expected = {"python": "3.10.19", "numpy": "2.2.6", "mujoco": "3.6.0", "myosuite": "2.12.2"}
    values["frozen_expected"] = expected
    values["frozen_match"] = all(values[key] == expected[key] for key in expected)
    if not values["frozen_match"]:
        raise RuntimeError(f"frozen MyoLeg runtime mismatch: {values}")
    return values


def candidate_geometry_rows() -> list[dict[str, str]]:
    with CANDIDATE_ADMISSION_PATH.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["included"] == "True"]


def select_validation_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["candidate_id"]: row for row in candidates}
    selected: dict[str, str] = {REFERENCE_CANDIDATE_ID: "REFERENCE"}
    keys = ("delta_hip_amp_deg", "delta_knee_amp_deg", "knee_phase_shift")
    aliases = ("HIP", "KNEE", "PHASE")
    converted = [
        {
            **row,
            "delta_hip_amp_deg": float(row["alpha"][0]),
            "delta_knee_amp_deg": float(row["alpha"][1]),
            "knee_phase_shift": float(row["alpha"][2]),
        }
        for row in candidates
    ]
    for key, alias in zip(keys, aliases):
        for side, value in (("LOW", min(row[key] for row in converted)), ("HIGH", max(row[key] for row in converted))):
            matches = [row for row in converted if row[key] == value]
            other = [item for item in keys if item != key]
            chosen = min(matches, key=lambda row: (sum(abs(row[item]) for item in other), int(row["proposal_index"])))
            selected.setdefault(chosen["candidate_id"], f"GLOBAL_{alias}_{side}")
    geometry = sorted(
        candidate_geometry_rows(),
        key=lambda row: (-float(row["q_knee_max_deg"]), int(row["proposal_index"])),
    )
    for row in geometry[:2]:
        selected.setdefault(row["candidate_id"], "KNEE_TRUSTED_BOUND_NEIGHBOR")
    interior = [row for row in converted if -3.0 <= row["delta_hip_amp_deg"] <= 0.0 and -3.0 <= row["delta_knee_amp_deg"] <= 0.0 and abs(row["knee_phase_shift"]) <= 0.015]
    interior.sort(key=lambda row: (hashlib.sha256(row["candidate_id"].encode()).hexdigest(), int(row["proposal_index"])))
    for row in interior[:2]:
        selected.setdefault(row["candidate_id"], "FIXED_HASH_INTERIOR")
    result = []
    for candidate_id, role in selected.items():
        row = lookup[candidate_id]
        result.append({"candidate_id": candidate_id, "proposal_index": int(row["proposal_index"]), "alpha": list(map(float, row["alpha"])), "role": role})
    return sorted(result, key=lambda row: row["proposal_index"])


def benchmark_pairs(cohort: dict[str, Any], detailed: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = sorted(cohort["subjects"], key=lambda row: row["subject_id"])
    development = [row for row in subjects if row["split"] == "DEVELOPMENT"][:4]
    held_out = [row for row in subjects if row["split"] == "HELD_OUT"][:4]
    candidate_ids = [REFERENCE_CANDIDATE_ID]
    candidate_ids.extend(row["candidate_id"] for row in detailed if row["candidate_id"] != REFERENCE_CANDIDATE_ID)
    candidate_ids = candidate_ids[:4]
    return {"subject_ids": [row["subject_id"] for row in development + held_out], "candidate_ids": candidate_ids, "pair_count": 32}


def controlled_subset(cohort: dict[str, Any], detailed: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = sorted(cohort["subjects"], key=lambda row: row["subject_id"])
    chosen_subjects = [
        [row for row in subjects if row["split"] == "DEVELOPMENT"][0],
        [row for row in subjects if row["split"] == "DEVELOPMENT"][-1],
        [row for row in subjects if row["split"] == "HELD_OUT"][0],
        [row for row in subjects if row["split"] == "HELD_OUT"][-1],
    ]
    chosen_candidates = [next(row for row in detailed if row["candidate_id"] == REFERENCE_CANDIDATE_ID)]
    knee = next(row for row in detailed if row["role"] == "KNEE_TRUSTED_BOUND_NEIGHBOR")
    interior = next(row for row in detailed if row["role"] == "FIXED_HASH_INTERIOR")
    chosen_candidates.extend([knee, interior])
    return {"subject_ids": [row["subject_id"] for row in chosen_subjects], "candidate_ids": [row["candidate_id"] for row in chosen_candidates], "pair_count": 12}


def freeze_protocol() -> None:
    if OUTPUT.exists() or DATA.exists():
        raise RuntimeError("truth-landscape output already exists; refusing to overwrite a freeze")
    cohort, candidate_manifest = verify_inputs()
    environment = runtime_environment()
    candidates = candidate_manifest["ordered_included_candidates"]
    detailed = select_validation_candidates(candidates)
    benchmark = benchmark_pairs(cohort, detailed)
    controlled = controlled_subset(cohort, detailed)
    bytes_per_row = sum(np.dtype(dtype).itemsize for _, dtype in PAIR_SCHEMA)
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_landscape_outcomes": True,
        "scientific_role": "HIDDEN_OFFLINE_VIRTUAL_TRUTH_LANDSCAPE",
        "input_sha256": FROZEN_SHA,
        "evaluation_set": {
            "subject_count": EXPECTED_SUBJECTS,
            "development_count": 24,
            "held_out_count": 8,
            "nominal_control_included": False,
            "candidate_count": EXPECTED_CANDIDATES,
            "expected_pair_count": EXPECTED_PAIRS,
            "reference_candidate_id": REFERENCE_CANDIDATE_ID,
            "subject_specific_candidate_filtering": False,
            "replacement_or_skip_allowed": False,
        },
        "truth": {
            "semantic_version": TRUTH_SEMANTIC_VERSION,
            "field": TRUTH_FIELD,
            "primary_method": "prescribed-state inverse dynamics replay",
            "controlled_method_role": "small frozen diagnostic subset only; never replaces prescribed truth",
            "rms_quadrature": "sqrt(trapezoid(tau^2,time)/(time[-1]-time[0]))",
            "objective": "sqrt(0.5*((hip_rms/subject_reference_hip_rms)^2+(knee_rms/subject_reference_knee_rms)^2))",
        },
        "pair_integrity_gates": {
            "sample_count": 401,
            "finite_tau_and_j": True,
            "solver_warning_count": 0,
            "source_equality_residual_max": SOURCE_EQUALITY_RESIDUAL_MAX,
            "joint_limit_contribution_max_abs_nm": ABS_LIMIT_TORQUE_MAX_NM,
            "joint_limit_contribution_max_relative": REL_LIMIT_CONTRIBUTION_MAX,
            "tendon_limit_active_count": 0,
            "contact_active_count": 0,
            "trusted_ROM_from_frozen_candidate_admission": True,
        },
        "storage": {
            "format": "deterministic compressed NPZ shards",
            "schema": [{"name": key, "dtype": dtype} for key, dtype in PAIR_SCHEMA],
            "uncompressed_bytes_per_row": bytes_per_row,
            "estimated_uncompressed_landscape_bytes": bytes_per_row * EXPECTED_PAIRS,
            "prohibited_full_replay_storage_estimate_gb": 386,
            "chunking": "subject then stable candidate rank blocks",
            "chunk_size": CHUNK_SIZE,
            "chunks_per_subject": EXPECTED_CHUNKS_PER_SUBJECT,
            "total_chunks": EXPECTED_CHUNKS,
            "zip_timestamp": "1980-01-01T00:00:00 for deterministic binary identity",
        },
        "parallelism": {
            "implementation": "process-based isolation",
            "benchmark_worker_counts": [1, 2, 4, 8],
            "benchmark_fixed_set": benchmark,
            "selection_rule": f"highest-throughput stable deterministic count with efficiency >= {BENCHMARK_MIN_EFFICIENCY}; otherwise highest-throughput stable deterministic count",
            "worker_owns_independent_model_and_MjData": True,
            "formal_worker_count": "TO_BE_FROZEN_BY_BENCHMARK_BEFORE_LANDSCAPE",
        },
        "detailed_validation_subset": {
            "selection_uses_geometry_and_candidate_id_only": True,
            "candidate_rows": detailed,
            "all_subject_ids": [row["subject_id"] for row in cohort["subjects"]],
            "pair_count": len(detailed) * EXPECTED_SUBJECTS,
            "repeat_prescribed_replay": 2,
        },
        "controlled_crosscheck_subset": controlled,
        "controlled_thresholds": CONTROLLED_THRESHOLDS,
        "oracle": {
            "allowed_only_after_complete_checksummed_landscape_freeze": True,
            "equivalence_tolerance_j": ORACLE_EQUIVALENCE_TOLERANCE,
            "tie_rule": "minimum float64 J; values within tolerance are equivalent; lowest original proposal_index wins",
            "personalization_interpretation_in_this_stage": False,
        },
        "scope_guards": {
            "learner_training": False,
            "five_parameter_features_or_fit": False,
            "nn_or_pinn": False,
            "bo": False,
            "robot_or_hardware": False,
            "human_ready": False,
        },
        "runtime_environment": environment,
    }
    access_policy = {
        "policy_id": "TRUTH_ACCESS_POLICY_V1",
        "landscape_role": "hidden post-freeze oracle artifact",
        "oracle_access_allowed_for": ["post_hoc_evaluation", "oracle", "regret", "personalization_analysis"],
        "query_access": "future algorithms reveal only replay_subject_candidate(subject_id,candidate_id) through query(subject_id,candidate_id)",
        "prohibited": [
            "PINN reads unqueried tau", "BO reads unqueried J", "candidate selection uses oracle rank",
            "full landscape becomes a learner feature", "held-out outcome changes cohort/domain/ranges",
        ],
        "implementation": {
            "query_replay_api": str(REPLAY_API_PATH.relative_to(ROOT)),
            "access_boundary": str(TRUTH_ACCESS_PATH.relative_to(ROOT)),
        },
    }
    OUTPUT.mkdir(parents=True)
    SHARDS.mkdir(parents=True)
    atomic_json(PROTOCOL_PATH, protocol)
    atomic_json(ACCESS_POLICY_PATH, access_policy)
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "detailed_candidates": len(detailed), "expected_chunks": EXPECTED_CHUNKS}, indent=2))


def worker_init() -> None:
    global _CANDIDATE_BUILDER, _REPLAY_BUILDER, _REFERENCE, _TRUSTED_DOMAIN
    _CANDIDATE_BUILDER = load_module(CANDIDATE_BUILDER_PATH, "_myoleg_candidate_builder_worker")
    _REPLAY_BUILDER = load_module(REPLAY_BUILDER_PATH, "_myoleg_replay_builder_worker")
    _REFERENCE = _CANDIDATE_BUILDER.load_reference_adapter()
    _TRUSTED_DOMAIN = read_json(CANDIDATE_MANIFEST_PATH)["trusted_domain"]


def worker_model(subject: dict[str, Any]) -> Any:
    global _MODEL_CACHE_ID, _MODEL_CACHE
    if _CANDIDATE_BUILDER is None:
        worker_init()
    subject_id = subject["subject_id"]
    if _MODEL_CACHE_ID != subject_id:
        rebuilt, _, model, _ = _CANDIDATE_BUILDER.model_from_record(subject)
        if rebuilt != subject_id:
            raise RuntimeError("worker subject reconstruction mismatch")
        _MODEL_CACHE_ID = subject_id
        _MODEL_CACHE = model
    return _MODEL_CACHE


def generated_candidate(candidate: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if _REFERENCE is None:
        worker_init()
    generated = _CANDIDATE_BUILDER.generate_candidate(_REFERENCE, *map(float, candidate["alpha"]))
    reference = {
        "time_s": _REFERENCE["time_s"], "q": generated["q"], "dq": generated["dq"],
        "ddq": generated["ddq"], "phases": _REFERENCE["phases"], "rows": [],
    }
    return generated, reference


def lightweight_constraint_metrics(model: Any, data: Any, tangent: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Project only the constraint information required by the pair gate.

    This is algebraically identical to selecting ``joint_limit`` from the
    frozen full grouping helper, but it does not materialize a dense Jacobian
    for equality/contact/friction groups that are not stored in compact rows.
    """

    if data.nefc == 0:
        return np.zeros(2, dtype=float), {"joint_limit": 0, "tendon_limit": 0, "contact": 0}
    types = np.asarray(data.efc_type, dtype=int)
    joint_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    tendon_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
    contact_types = np.asarray(
        [
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
        ],
        dtype=int,
    )
    joint_rows = np.flatnonzero(types == joint_type)
    counts = {
        "joint_limit": int(joint_rows.size),
        "tendon_limit": int(np.sum(types == tendon_type)),
        "contact": int(np.sum(np.isin(types, contact_types))),
    }
    if joint_rows.size == 0:
        return np.zeros(2, dtype=float), counts
    forces = np.asarray(data.efc_force)
    if not mujoco.mj_isSparse(model):
        jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
        projected = (jacobian[joint_rows] @ tangent).T @ forces[joint_rows]
    else:
        projected = np.zeros(2, dtype=float)
        values = np.asarray(data.efc_J)
        for row in joint_rows:
            start = int(data.efc_J_rowadr[row])
            count = int(data.efc_J_rownnz[row])
            columns = np.asarray(data.efc_J_colind[start : start + count], dtype=int)
            projected += (values[start : start + count] @ tangent[columns]) * float(forces[row])
    return projected, counts


def compact_replay(model: Any, candidate: dict[str, Any], subject: dict[str, Any]) -> dict[str, Any]:
    generated, _ = generated_candidate(candidate)
    time_s = np.asarray(_REFERENCE["time_s"], dtype=float)
    sample_count = len(time_s)
    data = mujoco.MjData(model)
    tau = np.empty((sample_count, 2), dtype=float)
    warning_max = 0
    equality_max = 0.0
    joint_limit_abs_max = 0.0
    joint_limit_relative_max = 0.0
    joint_limit_count_max = 0
    tendon_limit_count_max = 0
    contact_count_max = 0
    finite = True
    denominators = np.asarray([
        float(subject["subject_reference_tau_hip_rms_nm"]),
        float(subject["subject_reference_tau_knee_rms_nm"]),
    ])
    for sample in range(sample_count):
        _REPLAY_BUILDER.reset_to_target_state(model, data, generated["q"][sample], generated["dq"][sample], generated["ddq"][sample])
        desired_acceleration = np.asarray(data.qacc).copy()
        tangent = _REPLAY_BUILDER.independent_coordinate_tangent(model, data)
        # No compact-landscape field reads sensors; skip only sensor evaluation.
        mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
        actuator_internal = np.asarray(data.qfrc_actuator).copy()
        data.qacc[:] = desired_acceleration
        # qpos/qvel-dependent stages were just computed by mj_forward and are
        # unchanged. The frozen prescribed inverse dynamics are therefore
        # exactly reproduced by evaluating only the acceleration stage.
        mujoco.mj_inverseSkip(model, data, mujoco.mjtStage.mjSTAGE_VEL, 1)
        required = tangent.T @ (np.asarray(data.qfrc_inverse) - actuator_internal)
        tau[sample] = required
        projected_limit_signed, counts = lightweight_constraint_metrics(model, data, tangent)
        projected_limit = np.abs(projected_limit_signed)
        relative = projected_limit / np.maximum(np.abs(required), denominators)
        joint_limit_abs_max = max(joint_limit_abs_max, float(np.max(projected_limit)))
        joint_limit_relative_max = max(joint_limit_relative_max, float(np.max(relative)))
        joint_limit_count_max = max(joint_limit_count_max, int(counts["joint_limit"]))
        tendon_limit_count_max = max(tendon_limit_count_max, int(counts["tendon_limit"]))
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
    if _TRUSTED_DOMAIN is None:
        raise RuntimeError("worker trusted-domain identity was not initialized")
    q_deg = np.degrees(generated["q"])
    candidate_admission_respected = bool(
        np.isfinite(generated["q"]).all()
        and np.min(q_deg[:, 0]) >= float(_TRUSTED_DOMAIN["trusted_hip_domain_deg"][0]) - 1.0e-12
        and np.max(q_deg[:, 0]) <= float(_TRUSTED_DOMAIN["trusted_hip_domain_deg"][1]) + 1.0e-12
        and np.min(q_deg[:, 1]) >= float(_TRUSTED_DOMAIN["trusted_knee_lower_deg"]) - 1.0e-12
        and np.max(q_deg[:, 1]) <= float(_TRUSTED_DOMAIN["trusted_knee_upper_deg"]) + 1.0e-12
    )
    integrity = bool(
        finite and np.isfinite(rms).all() and np.isfinite(peak).all() and np.isfinite(j_truth)
        and warning_max == 0 and equality_max <= SOURCE_EQUALITY_RESIDUAL_MAX
        and joint_limit_abs_max <= ABS_LIMIT_TORQUE_MAX_NM
        and joint_limit_relative_max <= REL_LIMIT_CONTRIBUTION_MAX
        and joint_limit_count_max <= 1 and tendon_limit_count_max == 0 and contact_count_max == 0
        and candidate_admission_respected and sample_count == 401
    )
    return {
        "hip_tau_rms_nm": float(rms[0]), "knee_tau_rms_nm": float(rms[1]),
        "hip_tau_peak_abs_nm": float(peak[0]), "knee_tau_peak_abs_nm": float(peak[1]),
        "j_truth": j_truth, "source_equality_residual_max": equality_max,
        "joint_limit_contribution_max_abs_nm": joint_limit_abs_max,
        "joint_limit_contribution_max_relative": joint_limit_relative_max,
        "solver_warning_count": warning_max, "joint_limit_active_count_max": joint_limit_count_max,
        "tendon_limit_active_count_max": tendon_limit_count_max, "contact_active_count_max": contact_count_max,
        "sample_count": sample_count, "integrity_status": int(integrity),
    }


def empty_chunk_arrays(count: int) -> dict[str, np.ndarray]:
    return {key: np.empty(count, dtype=np.dtype(dtype)) for key, dtype in PAIR_SCHEMA}


def compact_chunk_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    subject = task["subject"]
    candidates = task["candidates"]
    model = worker_model(subject)
    arrays = empty_chunk_arrays(len(candidates))
    for index, candidate in enumerate(candidates):
        metrics = compact_replay(model, candidate, subject)
        arrays["subject_id"][index] = subject["subject_id"]
        arrays["candidate_id"][index] = candidate["candidate_id"]
        arrays["proposal_index"][index] = int(candidate["proposal_index"])
        arrays["alpha_hip_deg"][index], arrays["alpha_knee_deg"][index], arrays["alpha_phase"][index] = map(float, candidate["alpha"])
        arrays["subject_reference_hip_rms_nm"][index] = float(subject["subject_reference_tau_hip_rms_nm"])
        arrays["subject_reference_knee_rms_nm"][index] = float(subject["subject_reference_tau_knee_rms_nm"])
        for key, value in metrics.items():
            arrays[key][index] = value
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        maximum_rss *= 1024
    return {
        "chunk_id": task["chunk_id"], "arrays": arrays,
        "wall_time_s": time.perf_counter() - started, "cpu_time_s": time.process_time() - cpu_started,
        "worker_peak_rss_bytes": maximum_rss,
    }


def arrays_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        future_map = {executor.submit(compact_chunk_task, task): task for task in tasks}
        for future in future_map:
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"chunk_id": task["chunk_id"], "error_type": type(exc).__name__, "error": str(exc)})
    return sorted(results, key=lambda row: row["chunk_id"]), failures


def run_benchmark() -> None:
    cohort, candidate_manifest = verify_inputs()
    protocol = read_json(PROTOCOL_PATH)
    if BENCHMARK_PATH.exists() or EXECUTION_PLAN_PATH.exists():
        raise RuntimeError("parallelism benchmark already frozen")
    by_subject = {row["subject_id"]: row for row in cohort["subjects"]}
    by_candidate = {row["candidate_id"]: row for row in candidate_manifest["ordered_included_candidates"]}
    fixed = protocol["parallelism"]["benchmark_fixed_set"]
    tasks = []
    for subject_id in fixed["subject_ids"]:
        tasks.append({
            "chunk_id": f"BENCH_{subject_id}", "subject": by_subject[subject_id],
            "candidates": [by_candidate[candidate_id] for candidate_id in fixed["candidate_ids"]],
        })
    rows = []
    baseline_throughput = None
    reference_fingerprint = None
    for workers in (1, 2, 4, 8):
        started = time.perf_counter()
        results, failures = run_tasks(tasks, workers)
        wall = time.perf_counter() - started
        fingerprint = canonical_sha([(row["chunk_id"], arrays_fingerprint(row["arrays"])) for row in results])
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        pair_count = sum(len(row["arrays"]["candidate_id"]) for row in results)
        throughput = pair_count / wall if wall else 0.0
        if baseline_throughput is None:
            baseline_throughput = throughput
        efficiency = throughput / (baseline_throughput * workers) if baseline_throughput else 0.0
        aggregate_cpu = sum(float(row["cpu_time_s"]) for row in results)
        rows.append({
            "workers": workers, "pair_count": pair_count, "wall_time_s": wall,
            "throughput_pairs_per_s": throughput, "parallel_efficiency_vs_one": efficiency,
            "aggregate_worker_cpu_time_s": aggregate_cpu,
            "estimated_cpu_utilization_percent": 100.0 * aggregate_cpu / max(wall * workers, 1.0e-12),
            "maximum_worker_peak_rss_bytes": max((row["worker_peak_rss_bytes"] for row in results), default=0),
            "process_failure_count": len(failures), "process_failures": failures,
            "result_fingerprint_sha256": fingerprint,
            "deterministic_equal_to_one_worker": fingerprint == reference_fingerprint,
            "stable": len(failures) == 0 and pair_count == fixed["pair_count"] and fingerprint == reference_fingerprint,
        })
        print(json.dumps(rows[-1], sort_keys=True))
    eligible = [row for row in rows if row["stable"] and row["parallel_efficiency_vs_one"] >= BENCHMARK_MIN_EFFICIENCY]
    if not eligible:
        eligible = [row for row in rows if row["stable"]]
    if not eligible:
        raise RuntimeError("no stable deterministic process count")
    selected = max(eligible, key=lambda row: (row["throughput_pairs_per_s"], -row["workers"]))
    benchmark = {
        "stage_id": STAGE_ID, "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "fixed_pair_set": fixed, "worker_results": rows,
        "selection_rule_frozen_before_landscape": protocol["parallelism"]["selection_rule"],
        "selected_worker_count": selected["workers"],
    }
    plan = {
        "plan_id": "MYOLEG_V2_LANDSCAPE_EXECUTION_PLAN_V1",
        "frozen_before_landscape_outcomes": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "benchmark_sha256": "SET_AFTER_BENCHMARK_WRITE",
        "landscape_worker_count": selected["workers"],
        "chunk_size": CHUNK_SIZE, "total_chunks": EXPECTED_CHUNKS,
        "resume_policy": "skip valid checksum-and-schema chunks; recompute and record only missing or checksum-invalid chunks",
    }
    atomic_json(BENCHMARK_PATH, benchmark)
    plan["benchmark_sha256"] = sha256_file(BENCHMARK_PATH)
    atomic_json(EXECUTION_PLAN_PATH, plan)
    print(json.dumps({"selected_worker_count": selected["workers"], "benchmark_sha256": sha256_file(BENCHMARK_PATH), "execution_plan_sha256": sha256_file(EXECUTION_PLAN_PATH)}, indent=2))


def chunk_tasks(cohort: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for subject in sorted(cohort["subjects"], key=lambda row: row["subject_id"]):
        for chunk_rank, start in enumerate(range(0, len(candidates), CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, len(candidates))
            chunk_id = f"{subject['subject_id']}_C{chunk_rank:03d}"
            tasks.append({
                "chunk_id": chunk_id, "subject": subject, "candidates": candidates[start:end],
                "candidate_start_rank": start, "candidate_end_rank_exclusive": end,
            })
    if len(tasks) != EXPECTED_CHUNKS or sum(len(task["candidates"]) for task in tasks) != EXPECTED_PAIRS:
        raise RuntimeError("deterministic chunk plan coverage failure")
    return tasks


def chunk_path(chunk_id: str) -> Path:
    subject_id = chunk_id.rsplit("_C", 1)[0]
    return SHARDS / subject_id / f"{chunk_id}.npz"


def validate_existing_chunk(task: dict[str, Any]) -> tuple[bool, str]:
    path = chunk_path(task["chunk_id"])
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False, "missing"
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        return False, "checksum_invalid"
    try:
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != {key for key, _ in PAIR_SCHEMA}:
                return False, "schema_invalid"
            expected_candidates = task["candidates"]
            if len(shard["candidate_id"]) != len(expected_candidates):
                return False, "row_count_invalid"
            if not np.array_equal(shard["candidate_id"], np.asarray([row["candidate_id"] for row in expected_candidates])):
                return False, "candidate_identity_invalid"
            if not np.array_equal(shard["proposal_index"], np.asarray([row["proposal_index"] for row in expected_candidates], dtype=np.int32)):
                return False, "proposal_identity_invalid"
            if not np.all(shard["subject_id"] == task["subject"]["subject_id"]):
                return False, "subject_identity_invalid"
            if not np.all(shard["integrity_status"] == 1):
                return False, "pair_integrity_invalid"
    except Exception as exc:
        return False, f"load_invalid:{type(exc).__name__}"
    return True, "valid"


def append_resume_event(event: dict[str, Any]) -> None:
    path = OUTPUT / "RESUME_EVENTS.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def generate_landscape() -> None:
    cohort, candidate_manifest = verify_inputs()
    plan = read_json(EXECUTION_PLAN_PATH)
    workers = int(plan["landscape_worker_count"])
    tasks = chunk_tasks(cohort, candidate_manifest["ordered_included_candidates"])
    pending = []
    valid_existing = 0
    invalid_existing = 0
    for task in tasks:
        valid, reason = validate_existing_chunk(task)
        if valid:
            valid_existing += 1
        else:
            pending.append(task)
            if reason != "missing":
                invalid_existing += 1
            append_resume_event({"chunk_id": task["chunk_id"], "action": "RECOMPUTE", "reason": reason})
    print(json.dumps({"valid_existing_chunks": valid_existing, "pending_chunks": len(pending), "checksum_or_schema_invalid_chunks": invalid_existing, "workers": workers}))
    started = time.perf_counter()
    completed = 0
    failed_tasks: list[dict[str, Any]] = []
    task_runtime: list[dict[str, Any]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
            iterator = iter(pending)
            active: dict[Any, dict[str, Any]] = {}
            for _ in range(min(len(pending), workers * 2)):
                task = next(iterator, None)
                if task is not None:
                    active[executor.submit(compact_chunk_task, task)] = task
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        result = future.result()
                        path = chunk_path(task["chunk_id"])
                        digest = atomic_npz(path, result["arrays"])
                        append_resume_event({"chunk_id": task["chunk_id"], "action": "COMPLETED", "sha256": digest, "row_count": len(result["arrays"]["candidate_id"])})
                        task_runtime.append({
                            "chunk_id": task["chunk_id"], "subject_id": task["subject"]["subject_id"],
                            "row_count": len(result["arrays"]["candidate_id"]), "wall_time_s": result["wall_time_s"],
                            "cpu_time_s": result["cpu_time_s"], "worker_peak_rss_bytes": result["worker_peak_rss_bytes"],
                        })
                        completed += 1
                        if completed % 10 == 0 or completed == len(pending):
                            elapsed = time.perf_counter() - started
                            print(json.dumps({"completed_new_chunks": completed, "pending_total": len(pending), "elapsed_s": elapsed, "new_pairs_per_s": sum(row["row_count"] for row in task_runtime) / elapsed}), flush=True)
                    except Exception as exc:
                        failed_tasks.append({"chunk_id": task["chunk_id"], "error_type": type(exc).__name__, "error": str(exc)})
                    replacement = next(iterator, None)
                    if replacement is not None:
                        active[executor.submit(compact_chunk_task, replacement)] = replacement
    generation = {
        "formal_worker_count": workers, "resume_valid_chunk_count": valid_existing,
        "recomputed_chunk_count": completed, "invalid_existing_chunk_count": invalid_existing,
        "failed_chunk_count": len(failed_tasks), "failed_chunks": failed_tasks,
        "generation_command_wall_time_s": time.perf_counter() - started,
        "new_chunk_runtime": task_runtime,
    }
    atomic_json(OUTPUT / "GENERATION_RUNTIME_RAW.json", generation)
    if failed_tasks:
        raise RuntimeError(f"landscape chunk generation failures: {failed_tasks[:3]}")
    freeze_chunk_manifest(tasks, plan)


def freeze_chunk_manifest(tasks: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    chunks = []
    failure_rows = []
    for task in tasks:
        valid, reason = validate_existing_chunk(task)
        if not valid:
            failure_rows.append({"chunk_id": task["chunk_id"], "reason": reason})
            continue
        path = chunk_path(task["chunk_id"])
        with np.load(path, allow_pickle=False) as shard:
            status = np.asarray(shard["integrity_status"], dtype=np.uint8)
            failure_count = int(np.sum(status != 1))
            if failure_count:
                indexes = np.flatnonzero(status != 1)
                for index in indexes:
                    failure_rows.append({"chunk_id": task["chunk_id"], "reason": "pair_integrity_failure", "subject_id": str(shard["subject_id"][index]), "candidate_id": str(shard["candidate_id"][index])})
            row_count = len(status)
        chunks.append({
            "chunk_id": task["chunk_id"], "subject_id": task["subject"]["subject_id"],
            "candidate_start_rank": task["candidate_start_rank"], "candidate_end_rank_exclusive": task["candidate_end_rank_exclusive"],
            "row_count": row_count, "integrity_failure_count": failure_count,
            "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        })
    if failure_rows:
        write_csv(OUTPUT / "LANDSCAPE_FAILURES.csv", failure_rows)
        raise RuntimeError(f"LANDSCAPE_GENERATION_BLOCKED: {len(failure_rows)} missing/invalid pairs or chunks")
    if len(chunks) != EXPECTED_CHUNKS or sum(row["row_count"] for row in chunks) != EXPECTED_PAIRS:
        raise RuntimeError("LANDSCAPE_GENERATION_BLOCKED: final chunk coverage mismatch")
    subject_hashes = {}
    for subject_id in sorted({row["subject_id"] for row in chunks}):
        rows = [row for row in chunks if row["subject_id"] == subject_id]
        subject_hashes[subject_id] = canonical_sha([(row["chunk_id"], row["sha256"]) for row in rows])
    manifest = {
        "manifest_id": "LANDSCAPE_CHUNK_MANIFEST_V1", "stage_id": STAGE_ID,
        "created_before_oracle_reveal": True, "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "execution_plan_sha256": sha256_file(EXECUTION_PLAN_PATH),
        "formal_worker_count": int(plan["landscape_worker_count"]),
        "expected_row_count": EXPECTED_PAIRS, "actual_row_count": EXPECTED_PAIRS,
        "chunk_count": len(chunks), "chunk_size": CHUNK_SIZE, "chunks": chunks,
        "subject_landscape_sha256": subject_hashes,
        "global_data_sha256": canonical_sha([(row["path"], row["sha256"]) for row in chunks]),
        "all_pair_integrity_pass": True,
    }
    atomic_json(CHUNK_MANIFEST_PATH, manifest)
    atomic_json(OUTPUT / "LANDSCAPE_DATA_FREEZE.json", {
        "freeze_id": "MYOLEG_V2_TRUTH_LANDSCAPE_DATA_FREEZE_V1",
        "oracle_reveal_occurred": False,
        "chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST_PATH),
        "global_data_sha256": manifest["global_data_sha256"],
        "row_count": EXPECTED_PAIRS,
    })
    print(json.dumps({"chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST_PATH), "global_data_sha256": manifest["global_data_sha256"], "rows": EXPECTED_PAIRS}, indent=2))


def array_payload_sha(arrays: dict[str, np.ndarray]) -> str:
    return arrays_fingerprint({key: np.asarray(value) for key, value in arrays.items()})


def detailed_task(task: dict[str, Any]) -> dict[str, Any]:
    subject = task["subject"]
    candidate = task["candidate"]
    model = worker_model(subject)
    generated, replay_reference = generated_candidate(candidate)
    first, _ = _REPLAY_BUILDER.prescribed_truth(model, replay_reference)
    second, _ = _REPLAY_BUILDER.prescribed_truth(model, replay_reference)
    tau = np.asarray(first["tau_truth_nm"], dtype=float)
    time_s = np.asarray(_REFERENCE["time_s"], dtype=float)
    duration = float(time_s[-1] - time_s[0])
    rms = np.sqrt(np.trapezoid(tau**2, time_s, axis=0) / duration)
    denominators = np.asarray([subject["subject_reference_tau_hip_rms_nm"], subject["subject_reference_tau_knee_rms_nm"]], dtype=float)
    j_truth = float(np.sqrt(0.5 * np.sum((rms / denominators) ** 2)))
    repeat_equal = set(first) == set(second) and all(np.array_equal(first[key], second[key]) for key in first)
    residual = max(float(np.max(np.abs(first[key]))) for key in ("inverse_formula_residual_nm", "decomposition_residual_nm", "muscle_reconstruction_residual_nm"))
    return {
        "subject_id": subject["subject_id"], "split": subject["split"],
        "candidate_id": candidate["candidate_id"], "proposal_index": int(candidate["proposal_index"]),
        "alpha_hip_deg": float(candidate["alpha"][0]), "alpha_knee_deg": float(candidate["alpha"][1]), "alpha_phase": float(candidate["alpha"][2]),
        "selection_role": task["role"], "sample_count": len(tau),
        "hip_tau_rms_nm": float(rms[0]), "knee_tau_rms_nm": float(rms[1]), "j_truth": j_truth,
        "inverse_formula_residual_max_abs_nm": float(np.max(np.abs(first["inverse_formula_residual_nm"]))),
        "decomposition_residual_max_abs_nm": float(np.max(np.abs(first["decomposition_residual_nm"]))),
        "muscle_reconstruction_residual_max_abs_nm": float(np.max(np.abs(first["muscle_reconstruction_residual_nm"]))),
        "source_equality_residual_max": float(np.max(np.abs(first["source_equality_residual"]))),
        "mass_term_rms_nm": float(np.sqrt(np.mean(np.asarray(first["mass_term_nm"]) ** 2))),
        "bias_term_rms_nm": float(np.sqrt(np.mean(np.asarray(first["bias_term_nm"]) ** 2))),
        "passive_internal_rms_nm": float(np.sqrt(np.mean(np.asarray(first["passive_internal_nm"]) ** 2))),
        "actuator_internal_rms_nm": float(np.sqrt(np.mean(np.asarray(first["actuator_internal_nm"]) ** 2))),
        "constraint_internal_rms_nm": float(np.sqrt(np.mean(np.asarray(first["constraint_internal_nm"]) ** 2))),
        "solver_warning_count": int(np.max(first["warning_count"])),
        "prescribed_repeat_array_equal": repeat_equal,
        "first_array_payload_sha256": array_payload_sha(first), "second_array_payload_sha256": array_payload_sha(second),
        "detailed_integrity_pass": bool(repeat_equal and residual <= ALGEBRAIC_RESIDUAL_MAX_NM and float(np.max(np.abs(first["source_equality_residual"]))) <= SOURCE_EQUALITY_RESIDUAL_MAX and int(np.max(first["warning_count"])) == 0 and np.isfinite(tau).all()),
    }


def controlled_task(task: dict[str, Any]) -> dict[str, Any]:
    subject = task["subject"]
    candidate = task["candidate"]
    model = worker_model(subject)
    _, replay_reference = generated_candidate(candidate)
    prescribed, _ = _REPLAY_BUILDER.prescribed_truth(model, replay_reference)
    controlled, _ = _REPLAY_BUILDER.controlled_replay(model, replay_reference)
    rows = []
    for joint, label in enumerate(("hip", "knee")):
        truth = np.asarray(prescribed["tau_truth_nm"][:, joint], dtype=float)
        difference = np.asarray(controlled["force_balance_reconstruction_nm"][:, joint], dtype=float) - truth
        truth_rms = float(np.sqrt(np.mean(truth**2)))
        rmse = float(np.sqrt(np.mean(difference**2)))
        p95 = float(np.percentile(np.abs(difference), 95.0))
        maximum = float(np.max(np.abs(difference)))
        relative = rmse / max(truth_rms, 1.0e-12)
        passed = bool(rmse <= CONTROLLED_THRESHOLDS["rmse_max_nm"] and p95 <= CONTROLLED_THRESHOLDS["p95_max_nm"] and maximum <= CONTROLLED_THRESHOLDS["max_abs_max_nm"] and relative <= CONTROLLED_THRESHOLDS["relative_rms_max"])
        rows.append({
            "subject_id": subject["subject_id"], "split": subject["split"], "candidate_id": candidate["candidate_id"],
            "proposal_index": int(candidate["proposal_index"]), "joint": label,
            "difference_rmse_nm": rmse, "difference_p95_abs_nm": p95, "difference_max_abs_nm": maximum,
            "difference_relative_rms": relative, "correlation": float(np.corrcoef(controlled["force_balance_reconstruction_nm"][:, joint], truth)[0, 1]),
            "controlled_consistency_pass": passed,
        })
    return {"rows": rows}


def run_generic_process_tasks(function: Any, tasks: list[dict[str, Any]], workers: int) -> list[Any]:
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        future_map = {executor.submit(function, task): task for task in tasks}
        for future, task in future_map.items():
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"validation task failed {task.get('subject', {}).get('subject_id')} {task.get('candidate', {}).get('candidate_id')}: {exc}") from exc
    return results


def load_compact_subject(subject_id: str, chunk_manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {}
    rows = sorted((row for row in chunk_manifest["chunks"] if row["subject_id"] == subject_id), key=lambda row: row["candidate_start_rank"])
    for row in rows:
        with np.load(ROOT / row["path"], allow_pickle=False) as shard:
            for key in shard.files:
                columns.setdefault(key, []).append(np.asarray(shard[key]))
    return {key: np.concatenate(values) for key, values in columns.items()}


def finalize_landscape() -> None:
    cohort, candidate_manifest = verify_inputs()
    protocol = read_json(PROTOCOL_PATH)
    plan = read_json(EXECUTION_PLAN_PATH)
    chunk_manifest = read_json(CHUNK_MANIFEST_PATH)
    if not (chunk_manifest["actual_row_count"] == EXPECTED_PAIRS and chunk_manifest["all_pair_integrity_pass"] and chunk_manifest["created_before_oracle_reveal"]):
        raise RuntimeError("landscape is not complete/frozen before validation and oracle")
    subjects = {row["subject_id"]: row for row in cohort["subjects"]}
    candidates = {row["candidate_id"]: row for row in candidate_manifest["ordered_included_candidates"]}
    detailed_rows = protocol["detailed_validation_subset"]["candidate_rows"]
    detailed_tasks = [
        {"subject": subjects[subject_id], "candidate": candidates[row["candidate_id"]], "role": row["role"]}
        for subject_id in protocol["detailed_validation_subset"]["all_subject_ids"] for row in detailed_rows
    ]
    detailed_results = run_generic_process_tasks(detailed_task, detailed_tasks, int(plan["landscape_worker_count"]))
    detailed_results.sort(key=lambda row: (row["subject_id"], row["proposal_index"]))
    write_csv(OUTPUT / "DETAILED_VALIDATION_RESULTS.csv", detailed_results)

    controlled_spec = protocol["controlled_crosscheck_subset"]
    controlled_tasks = [
        {"subject": subjects[subject_id], "candidate": candidates[candidate_id]}
        for subject_id in controlled_spec["subject_ids"] for candidate_id in controlled_spec["candidate_ids"]
    ]
    controlled_results = run_generic_process_tasks(controlled_task, controlled_tasks, int(plan["landscape_worker_count"]))
    controlled_rows = [row for result in controlled_results for row in result["rows"]]
    controlled_rows.sort(key=lambda row: (row["subject_id"], row["proposal_index"], row["joint"]))
    write_csv(OUTPUT / "CONTROLLED_CROSSCHECK_RESULTS.csv", controlled_rows)

    # The complete data/chunk identity was frozen above.  Only now reveal minima.
    oracle_rows = []
    subject_summaries = []
    reference_failures = []
    duplicate_pair_count = 0
    global_pairs: set[tuple[str, str]] = set()
    for subject_id in sorted(subjects):
        arrays = load_compact_subject(subject_id, chunk_manifest)
        candidate_ids = arrays["candidate_id"]
        proposal = arrays["proposal_index"]
        j_values = arrays["j_truth"]
        for candidate_id in candidate_ids:
            key = (subject_id, str(candidate_id))
            duplicate_pair_count += int(key in global_pairs)
            global_pairs.add(key)
        reference_index = int(np.flatnonzero(candidate_ids == REFERENCE_CANDIDATE_ID)[0])
        reference_j = float(j_values[reference_index])
        if abs(reference_j - 1.0) > REFERENCE_J_TOLERANCE:
            reference_failures.append({"subject_id": subject_id, "reference_j": reference_j})
        minimum = float(np.min(j_values))
        equivalent = np.flatnonzero(j_values <= minimum + ORACLE_EQUIVALENCE_TOLERANCE)
        winner = int(equivalent[np.argmin(proposal[equivalent])])
        oracle_rows.append({
            "subject_id": subject_id, "split": subjects[subject_id]["split"],
            "oracle_candidate_id": str(candidate_ids[winner]), "oracle_proposal_index": int(proposal[winner]),
            "oracle_alpha_hip_deg": float(arrays["alpha_hip_deg"][winner]), "oracle_alpha_knee_deg": float(arrays["alpha_knee_deg"][winner]),
            "oracle_alpha_phase": float(arrays["alpha_phase"][winner]), "oracle_j_truth": float(j_values[winner]),
            "minimum_float64_j": minimum, "equivalent_candidate_count": len(equivalent),
            "oracle_reveal_occurred_after_landscape_freeze": True,
        })
        subject_summaries.append({
            "subject_id": subject_id, "split": subjects[subject_id]["split"], "row_count": len(j_values),
            "j_min": minimum, "j_median": float(np.median(j_values)), "j_max": float(np.max(j_values)),
            "reference_candidate_id": REFERENCE_CANDIDATE_ID, "reference_j": reference_j,
            "oracle_candidate_id": str(candidate_ids[winner]), "oracle_j": float(j_values[winner]),
            "subject_landscape_sha256": chunk_manifest["subject_landscape_sha256"][subject_id],
        })
    if len(global_pairs) != EXPECTED_PAIRS or duplicate_pair_count != 0:
        raise RuntimeError("pair uniqueness/completeness failure at oracle reveal")
    write_csv(OUTPUT / "POST_FREEZE_ORACLE_SUMMARY.csv", oracle_rows)
    write_csv(OUTPUT / "SUBJECT_LANDSCAPE_SUMMARY.csv", subject_summaries)

    generation_runtime = read_json(OUTPUT / "GENERATION_RUNTIME_RAW.json")
    all_chunk_bytes = sum(int(row["size_bytes"]) for row in chunk_manifest["chunks"])
    all_sidecar_bytes = sum((ROOT / row["path"]).with_suffix(".npz.sha256").stat().st_size for row in chunk_manifest["chunks"])
    worker_cpu = sum(float(row["cpu_time_s"]) for row in generation_runtime["new_chunk_runtime"])
    replay_wall = sum(float(row["wall_time_s"]) for row in generation_runtime["new_chunk_runtime"])
    formal_wall = float(generation_runtime["generation_command_wall_time_s"])
    interrupted = read_json(OUTPUT / "INTERRUPTION_AND_RESUME_AUDIT.json")
    new_pair_count = sum(int(row["row_count"]) for row in generation_runtime["new_chunk_runtime"])
    runtime_storage = {
        "serial_baseline_estimate_hours": 25.20, "prior_eight_worker_estimate_hours": 4.20,
        "formal_worker_count": int(plan["landscape_worker_count"]),
        "generation_command_wall_time_s": formal_wall, "aggregate_new_chunk_worker_wall_time_s": replay_wall,
        "aggregate_new_chunk_cpu_time_s": worker_cpu,
        "prior_interrupted_generation_wall_time_lower_bound_s": interrupted["prior_interrupted_wall_time_lower_bound_s"],
        "total_observed_generation_wall_time_lower_bound_including_interrupted_attempts_s": formal_wall + float(interrupted["prior_interrupted_wall_time_lower_bound_s"]),
        "actual_generation_throughput_pairs_per_s": new_pair_count / max(formal_wall, 1.0e-12),
        "compact_shard_count": len(chunk_manifest["chunks"]), "compact_shard_bytes": all_chunk_bytes,
        "checksum_sidecar_bytes": all_sidecar_bytes, "total_compact_data_and_sidecar_bytes": all_chunk_bytes + all_sidecar_bytes,
        "uncompressed_schema_bytes_per_row": sum(np.dtype(dtype).itemsize for _, dtype in PAIR_SCHEMA),
        "bulk_full_time_series_generated": False, "bulk_full_replay_storage_estimate_gb_avoided": 386,
    }
    atomic_json(OUTPUT / "RUNTIME_STORAGE_AUDIT.json", runtime_storage)

    detailed_pass = all(bool(row["detailed_integrity_pass"]) for row in detailed_results)
    controlled_pass = all(bool(row["controlled_consistency_pass"]) for row in controlled_rows)
    if reference_failures or not detailed_pass:
        outcome = OUTCOME_INVALID
    elif not controlled_pass:
        outcome = OUTCOME_LIMITED
    else:
        outcome = OUTCOME_VALID
    final_manifest = {
        "manifest_id": LANDSCAPE_ID + "_MANIFEST", "stage_id": STAGE_ID, "outcome": outcome,
        "cohort_manifest_sha256": FROZEN_SHA["cohort_manifest"], "candidate_manifest_sha256": FROZEN_SHA["candidate_manifest"],
        "v2_reference_sha256": FROZEN_SHA["v2_reference"], "truth_semantics_sha256": FROZEN_SHA["truth_semantics"],
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION, "truth_field": TRUTH_FIELD,
        "generation_code_sha256": sha256_file(Path(__file__)),
        "replay_api_sha256": sha256_file(REPLAY_API_PATH), "truth_access_code_sha256": sha256_file(TRUTH_ACCESS_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "truth_access_policy_sha256": sha256_file(ACCESS_POLICY_PATH),
        "parallelism_benchmark_sha256": sha256_file(BENCHMARK_PATH), "execution_plan_sha256": sha256_file(EXECUTION_PLAN_PATH),
        "landscape_chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST_PATH),
        "landscape_data_freeze_sha256": sha256_file(OUTPUT / "LANDSCAPE_DATA_FREEZE.json"),
        "formal_worker_count": int(plan["landscape_worker_count"]), "chunking_scheme": "subject then stable candidate rank blocks of 250",
        "expected_row_count": EXPECTED_PAIRS, "actual_row_count": len(global_pairs), "duplicate_pair_count": duplicate_pair_count,
        "storage_schema": [{"name": key, "dtype": dtype} for key, dtype in PAIR_SCHEMA],
        "chunks": chunk_manifest["chunks"], "subject_landscape_sha256": chunk_manifest["subject_landscape_sha256"],
        "final_data_sha256": chunk_manifest["global_data_sha256"],
        "integrity_summary": {
            "all_pair_integrity_pass": chunk_manifest["all_pair_integrity_pass"],
            "reference_normalization_tolerance": REFERENCE_J_TOLERANCE,
            "all_subject_reference_j_within_tolerance": not reference_failures,
            "reference_normalization_failures": reference_failures,
            "detailed_validation_pair_count": len(detailed_results), "detailed_validation_pass": detailed_pass,
            "controlled_crosscheck_pair_count": len(controlled_tasks), "controlled_crosscheck_joint_row_count": len(controlled_rows),
            "controlled_crosscheck_pass": controlled_pass,
        },
        "landscape_frozen_before_oracle_reveal": True,
        "oracle_reveal_policy": {
            "reveal_occurred_after_landscape_freeze": True,
            "tie_tolerance_j": ORACLE_EQUIVALENCE_TOLERANCE,
            "tie_break": "lowest original proposal_index",
            "post_freeze_oracle_summary_sha256": sha256_file(OUTPUT / "POST_FREEZE_ORACLE_SUMMARY.csv"),
        },
        "scope": {"offline_only": True, "learner_trained": False, "five_parameter": False, "nn_or_pinn": False, "bo": False, "robot_or_hardware": False},
        "runtime_storage_audit_sha256": sha256_file(OUTPUT / "RUNTIME_STORAGE_AUDIT.json"),
    }
    atomic_json(FINAL_MANIFEST_PATH, final_manifest)
    write_report(final_manifest, runtime_storage, subject_summaries, controlled_rows)
    write_metadata_and_checksums(final_manifest)
    print(json.dumps({"outcome": outcome, "final_manifest_sha256": sha256_file(FINAL_MANIFEST_PATH), "rows": len(global_pairs), "controlled_pass": controlled_pass}, indent=2))


def write_report(manifest: dict[str, Any], runtime: dict[str, Any], subjects: list[dict[str, Any]], controlled: list[dict[str, Any]]) -> None:
    integrity = manifest["integrity_summary"]
    text = f"""# MyoLeg V2 Truth Landscape Generation V1

## Decision

`{manifest['outcome']}`

This artifact is a hidden offline virtual truth landscape. It is not a human result, robot-motion approval, clinical result, learner training set, or safety validation.

## Frozen identity and coverage

- Cohort manifest SHA-256: `{manifest['cohort_manifest_sha256']}`.
- Candidate manifest SHA-256: `{manifest['candidate_manifest_sha256']}`.
- Truth semantic: `{TRUTH_SEMANTIC_VERSION}` / `{TRUTH_FIELD}`.
- Evaluated pairs: `{manifest['actual_row_count']:,}` / `{manifest['expected_row_count']:,}`.
- Duplicate pairs: `{manifest['duplicate_pair_count']}`.
- Reference candidate per subject: `{REFERENCE_CANDIDATE_ID}`.
- All subject reference objectives within {REFERENCE_J_TOLERANCE:g} of 1: `{integrity['all_subject_reference_j_within_tolerance']}`.

## Compact storage

The primary data are {runtime['compact_shard_count']:,} deterministic compressed NPZ shards using {runtime['uncompressed_schema_bytes_per_row']} uncompressed schema bytes per pair. Shards occupy {runtime['compact_shard_bytes']:,} bytes; checksum sidecars occupy {runtime['checksum_sidecar_bytes']:,} bytes. No bulk 401-sample replay schema was generated, avoiding the approximately 386 GB design.

## Runtime and integrity

- Formal process workers: `{runtime['formal_worker_count']}`.
- Formal generation command wall time: `{runtime['generation_command_wall_time_s']:.3f}` s.
- All-pair lightweight integrity: `{integrity['all_pair_integrity_pass']}`.
- Detailed prescribed-repeat pairs: `{integrity['detailed_validation_pair_count']}`; pass: `{integrity['detailed_validation_pass']}`.
- Controlled diagnostic pairs: `{integrity['controlled_crosscheck_pair_count']}` ({integrity['controlled_crosscheck_joint_row_count']} joint rows); pass: `{integrity['controlled_crosscheck_pass']}`.

## Access and oracle ordering

Future algorithms must reveal executed candidates through `query(subject_id, candidate_id)`, which regenerates full prescribed replay arrays. The compact table is restricted to post-hoc evaluation/oracle/regret/personalization analysis. The chunk/data freeze SHA existed before minima were read. Oracle tie handling was preregistered as float64 minimum, {ORACLE_EQUIVALENCE_TOLERANCE:g} equivalence, then lowest original proposal index. No personalization interpretation was performed here.

## Boundaries

No Five-parameter model, NN/PINN, BO, candidate-domain change, cohort change, robot/hardware connection, or human-ready claim was made. The next stage may audit personalization necessity only if the final decision above is valid or valid with limitations.
"""
    atomic_text(OUTPUT / "MYOLEG_V2_TRUTH_LANDSCAPE_GENERATION_REPORT.md", text)


def write_metadata_and_checksums(manifest: dict[str, Any]) -> None:
    artifact_paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name not in {"checksums.sha256", "metadata.json"})
    metadata = {
        "stage_id": STAGE_ID, "outcome": manifest["outcome"],
        "formal_manifest_sha256": sha256_file(FINAL_MANIFEST_PATH),
        "artifact_count_excluding_metadata_and_checksums": len(artifact_paths),
        "data_directory": str(DATA.relative_to(ROOT)), "data_global_sha256": manifest["final_data_sha256"],
        "offline_only": True, "not_human_ready": True, "not_robot_approved": True,
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--generate", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol:
        freeze_protocol()
    elif args.benchmark:
        run_benchmark()
    elif args.generate:
        generate_landscape()
    elif args.finalize:
        finalize_landscape()
    else:
        freeze_protocol()
        run_benchmark()
        generate_landscape()
        finalize_landscape()


if __name__ == "__main__":
    main()
