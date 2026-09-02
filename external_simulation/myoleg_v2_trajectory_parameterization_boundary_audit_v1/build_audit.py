"""Build the development-only MyoLeg-V2 trajectory-boundary audit.

The protocol is frozen before development landscape values are loaded.  The
stage audits the exact frozen V2 generator, produces kinematic-only prototype
trajectories, and never modifies or expands the frozen candidate domain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rehab_robot_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from external_simulation.myoleg_v2_trajectory_parameterization_boundary_audit_v1 import prototype_parameterizations as prototypes


STAGE_ID = "MYOLEG_V2_TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_V1"
PROTOCOL_ID = "TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_PROTOCOL_V1"
OUTCOME = "TRAJECTORY_PARAMETERIZATION_ROOT_CAUSE_SUPPORTED"
V2_DECISION = "NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION"
NEXT_STAGE = "MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_DESIGN_V1"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_trajectory_parameterization_boundary_audit_v1"
FIGURES = OUTPUT / "figures"
PROTOCOL_PATH = OUTPUT / "TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_PROTOCOL.json"
TRUTH_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
TRUTH_PROTOCOL_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/LANDSCAPE_GENERATION_PROTOCOL.json"
CANDIDATE_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
CANDIDATE_ADMISSION_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/V2_CANDIDATE_ADMISSION.csv"
TRUSTED_ROM_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/TRUSTED_ROM_DECISION.json"
COHORT_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
ROOT_CAUSE_OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_personalization_signal_root_cause_audit_v1"
ROOT_CAUSE_CHECKSUMS_PATH = ROOT_CAUSE_OUTPUT / "checksums.sha256"
ROOT_CAUSE_PROTOCOL_PATH = ROOT_CAUSE_OUTPUT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json"
ROOT_CAUSE_METADATA_PATH = ROOT_CAUSE_OUTPUT / "metadata.json"
ROOT_CAUSE_REPLAY_MANIFEST_PATH = ROOT_CAUSE_OUTPUT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json"
ROOT_CAUSE_REPLAY_CACHE_PATH = ROOT / "external_simulation/data/myoleg_v2_personalization_signal_root_cause_audit_v1/development_replay_subset.npz"
GENERATOR_SOURCE_PATH = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
V2_REFERENCE_PATH = ROOT / "external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv"
FORMAL_REFERENCE_PATH = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
PROTOTYPE_SOURCE_PATH = Path(prototypes.__file__).resolve()

EXPECTED_CANDIDATES = 16675
EXPECTED_DEVELOPMENT = 24
EXPECTED_HELD_OUT = 8
HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)
REFERENCE_ID = "MYOLEG_V2_P15012"
COMMON_ORACLE_ID = "MYOLEG_V2_P20850"
ROM_EQUIVALENCE_TOLERANCE_DEG = 1.0e-3
KINEMATIC_EQUALITY_TOLERANCE = 1.0e-9
PROTOTYPE_EXTREMA_TOLERANCE_DEG = 1.0e-3
PHASE_REPLAY_VALUES = (-0.03, -0.0025, 0.0, 0.0025, 0.03)
PHASE_AMPLITUDE_PAIRS = (
    (0.0, 0.0), (2.0, 0.5), (-5.0, -5.0),
    (-2.0, -2.0), (2.0, -2.0), (-2.0, 0.5),
)
EXPECTED_PROTOCOL_SHA256 = "5699e75a73a28d9df037a01de2c047a241c9cb5a4c0598e16d66e8f8d2f708a5"

FROZEN_SHA = {
    "truth_landscape_manifest": "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    "truth_landscape_protocol": "2fe115d8c34685c70672bcc6a4d9752a88dfbb2cf12fb12d60df877755b7fdcc",
    "candidate_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "candidate_admission": "13ff61daf55560953d1b1ff7a590af7f132b2f88d880dad0202cd3fb79e017b2",
    "trusted_rom_decision": "0383984f3b49835dd014d8de13f54d69d258fe8099a02d46dd398cd045a6e902",
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "root_cause_checksums": "d30910fa6ea2b6ce2346212cfb1b1013a07ff0c4c90d4db99a19abf66b170c3f",
    "root_cause_protocol": "2beac2ffb512783bcbe6dfcf60e8d64d9b6be8a5fe2122b8c77da876e6202bbb",
    "root_cause_metadata": "7a7e63f8ab03584484ba41ae29be2236fad719f0545972f477c0b3a15bb9e6a6",
    "root_cause_replay_manifest": "9956ebca4746778a634d1ef32231bf63b9c91464f16a69fe5b326ac186749298",
    "root_cause_replay_cache": "6e74603d52b6dfd7ef6bbc3919f7de34dcbbf57cd5d73fe0404ce3205d3d7afe",
    "generator_source": "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise RuntimeError(f"cannot infer CSV schema: {path}")
    columns = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def verify_checksum_manifest(directory: Path) -> int:
    count = 0
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen artifact checksum mismatch: {path}")
        count += 1
    return count


def frozen_paths() -> dict[str, Path]:
    return {
        "truth_landscape_manifest": TRUTH_MANIFEST_PATH,
        "truth_landscape_protocol": TRUTH_PROTOCOL_PATH,
        "candidate_manifest": CANDIDATE_MANIFEST_PATH,
        "candidate_admission": CANDIDATE_ADMISSION_PATH,
        "trusted_rom_decision": TRUSTED_ROM_PATH,
        "cohort_manifest": COHORT_MANIFEST_PATH,
        "root_cause_checksums": ROOT_CAUSE_CHECKSUMS_PATH,
        "root_cause_protocol": ROOT_CAUSE_PROTOCOL_PATH,
        "root_cause_metadata": ROOT_CAUSE_METADATA_PATH,
        "root_cause_replay_manifest": ROOT_CAUSE_REPLAY_MANIFEST_PATH,
        "root_cause_replay_cache": ROOT_CAUSE_REPLAY_CACHE_PATH,
        "generator_source": GENERATOR_SOURCE_PATH,
        "v2_reference": V2_REFERENCE_PATH,
        "formal_reference": FORMAL_REFERENCE_PATH,
    }


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    actual = {name: sha256_file(path) for name, path in frozen_paths().items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    verified_root_files = verify_checksum_manifest(ROOT_CAUSE_OUTPUT)
    truth = read_json(TRUTH_MANIFEST_PATH)
    candidates = read_json(CANDIDATE_MANIFEST_PATH)
    cohort = read_json(COHORT_MANIFEST_PATH)
    root_metadata = read_json(ROOT_CAUSE_METADATA_PATH)
    if not (
        truth["outcome"] == "MYOLEG_V2_TRUTH_LANDSCAPE_VALID"
        and truth["actual_row_count"] == 533600
        and len(candidates["ordered_included_candidates"]) == EXPECTED_CANDIDATES
        and len(cohort["development_subject_ids"]) == EXPECTED_DEVELOPMENT
        and tuple(cohort["held_out_subject_ids"]) == HELD_OUT_IDS
        and root_metadata["outcome"] == "PERSONALIZATION_SIGNAL_ROOT_CAUSE_IDENTIFIED"
        and root_metadata["held_out_scientific_truth_access_count"] == 0
        and verified_root_files >= 40
    ):
        raise RuntimeError("frozen identity or prior-stage status changed")
    return truth, candidates, cohort


def held_out_hash_audit(truth: dict[str, Any]) -> dict[str, Any]:
    chunks = [row for row in truth["chunks"] if row["subject_id"] in HELD_OUT_IDS]
    if len(chunks) != 536 or sum(int(row["row_count"]) for row in chunks) != EXPECTED_HELD_OUT * EXPECTED_CANDIDATES:
        raise RuntimeError("held-out manifest coverage mismatch")
    present = verified = byte_count = 0
    for row in chunks:
        path = ROOT / row["path"]
        if path.is_file():
            present += 1
            byte_count += path.stat().st_size
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"held-out raw-byte SHA mismatch: {path}")
            verified += 1
    return {
        "classification": "SEALED_CONFIRMATORY_TRUTH",
        "sealed_subject_ids": list(HELD_OUT_IDS),
        "sealed_subject_count": EXPECTED_HELD_OUT,
        "manifest_chunk_count": len(chunks),
        "manifest_row_count": sum(int(row["row_count"]) for row in chunks),
        "local_shard_count_present": present,
        "local_shard_count_sha256_verified": verified,
        "local_shard_bytes_stream_hashed": byte_count,
        "allowed_operations": ["file existence", "file size", "streaming SHA-256", "manifest row count"],
        "np_load_held_out_count": 0,
        "held_out_scientific_truth_access_count": 0,
        "held_out_j_oracle_rank_component_access_count": 0,
    }


def structural_option_rows() -> list[dict[str, Any]]:
    criteria = (
        "low_dimensionality", "interpretability", "prescribed_task_preservation", "C2_feasibility",
        "closed_cycle_feasibility", "no_clipping_generation", "branch_asymmetry_control",
        "coordination_shape_expressiveness", "extrema_preservation", "simulator_compatibility",
        "smooth_dq_ddq", "reversible_semantics", "local_reference_perturbation",
        "low_budget_BO_suitability", "future_robot_auditability",
    )
    scores = {
        "P1_PHASE_COORDINATION_ONLY": (2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2),
        "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION": (2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1),
        "P3_JOINT_SPACE_NORMAL_DISPLACEMENT": (2, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 0),
        "P4_BRANCH_AWARE_COORDINATION_FUNCTION": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    }
    dimensions = {
        "P1_PHASE_COORDINATION_ONLY": 1,
        "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION": 4,
        "P3_JOINT_SPACE_NORMAL_DISPLACEMENT": 2,
        "P4_BRANCH_AWARE_COORDINATION_FUNCTION": 2,
    }
    rows: list[dict[str, Any]] = []
    for option_id, values in scores.items():
        row: dict[str, Any] = {"parameterization_id": option_id, "dimension": dimensions[option_id]}
        row.update(dict(zip(criteria, values)))
        row["structural_score"] = int(sum(values))
        row["maximum_score"] = int(2 * len(criteria))
        rows.append(row)
    return rows


def protocol_payload(candidates: dict[str, Any], cohort: dict[str, Any]) -> dict[str, Any]:
    available = {tuple(map(float, row["alpha"])) for row in candidates["ordered_included_candidates"]}
    for hip, knee in PHASE_AMPLITUDE_PAIRS:
        for phase in np.round(np.arange(-0.03, 0.0301, 0.0025), 10):
            if (hip, knee, float(phase)) not in available:
                raise RuntimeError(f"pre-frozen phase slice is absent: {(hip, knee, phase)}")
    option_rows = structural_option_rows()
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_new_development_scientific_values_read": True,
        "protocol_freeze_inputs": FROZEN_SHA,
        "prior_frozen_conclusion": "PERSONALIZATION_SIGNAL_ROOT_CAUSE_IDENTIFIED",
        "population": {
            "development_subject_ids": list(cohort["development_subject_ids"]),
            "development_count": EXPECTED_DEVELOPMENT,
            "held_out_subject_ids": list(HELD_OUT_IDS),
            "held_out_scientific_values_allowed": False,
            "candidate_count": EXPECTED_CANDIDATES,
        },
        "hypotheses": {
            "H_PARAMETER_TASK_MIXING": "amplitude parameters change prescribed extrema/ROM rather than path shape alone",
            "H_ROM_COMMON_EFFECT": "simple ROM/extrema descriptors explain most common candidate effect",
            "H_FIXED_ROM_LIMITED": "the frozen grid has only one independent fixed-ROM path/timing dimension",
            "H_PHASE_COMMON_MONOTONIC": "phase-only variation remains common-monotonic across development subjects",
            "H_BOUNDARY_MIXED_SOURCE": "oracle coordinates mix arbitrary proposal and simulator-validity boundaries",
        },
        "generator_audit": {
            "exact_source_path": str(GENERATOR_SOURCE_PATH.relative_to(ROOT)),
            "exact_source_sha256": FROZEN_SHA["generator_source"],
            "functions_executed": ["load_reference_adapter", "generate_candidate"],
            "no_reimplementation_of_frozen_generator": True,
        },
        "descriptor_definitions": {
            "ROM_extrema_deg": ["hip_min", "hip_max", "hip_range", "knee_min", "knee_max", "knee_range"],
            "timing_s": ["cycle_duration", "flexion_duration", "extension_duration", "flexion_midpoint_lag", "extension_midpoint_lag"],
            "path_geometry": ["joint_space_path_length_deg", "signed_qspace_area_deg2", "absolute_qspace_area_deg2", "normalized_path_length", "branch_coordination_slope"],
            "kinematics": ["peak_abs_dq", "time_weighted_rms_dq", "peak_abs_ddq", "time_weighted_rms_ddq"],
            "continuity": ["q_closure", "dq_closure", "ddq_closure"],
        },
        "matched_ROM": {
            "equivalence_tolerance_deg": ROM_EQUIVALENCE_TOLERANCE_DEG,
            "grouping": "quantize all four joint extrema by tolerance; require at least five phase-distinct candidates",
            "minimum_independent_path_dimensions_for_broad_identification": 2,
            "metrics": ["pairwise Kendall rank similarity", "rank inversion", "subject-candidate interaction fraction", "shared optimum count", "common regret"],
        },
        "phase_only": {
            "amplitude_pairs": [list(pair) for pair in PHASE_AMPLITUDE_PAIRS],
            "phase_values": np.round(np.arange(-0.03, 0.0301, 0.0025), 10).tolist(),
            "common_monotonic_rule": "at least 23/24 subjects have all 24 adjacent J differences with the same nonzero sign for every preregistered amplitude pair",
        },
        "phase_mechanism": {
            "reused_prior_deterministic_cache_sha256": FROZEN_SHA["root_cause_replay_cache"],
            "subject_count": 6,
            "phase_values": list(PHASE_REPLAY_VALUES),
            "components": ["mass", "bias_gravity", "passive", "zero_control_actuator", "constraint", "total_tau"],
            "selection_modified_after_results": False,
        },
        "candidate_effect_diagnostics": {
            "targets": ["mean_development_J", "candidate_main_effect"],
            "univariate": "Pearson and Spearman association for preregistered descriptors",
            "primary_ROM_model": "ordinary least squares with hip ROM, knee ROM, their squares, and interaction",
            "secondary_timing_model": "ordinary least squares quadratic in hip ROM, knee ROM, and phase including pair interactions",
            "full_kinematic_model": "ordinary least squares over fixed standardized descriptor list; descriptive only",
            "ROM_EXTREMA_EXPLAINED_COMMON_EFFECT": "R2 of primary_ROM_model on candidate-main effect",
            "dominant_threshold": 0.90,
            "substantial_threshold": 0.75,
            "causal_interpretation_forbidden": True,
            "production_learner": False,
        },
        "prototype_basis": {
            "kinematic_only": True,
            "diagnostic_coefficient_magnitude_deg": 0.25,
            "diagnostic_value_is_not_a_frozen_range": True,
            "extrema_equivalence_tolerance_deg": PROTOTYPE_EXTREMA_TOLERANCE_DEG,
            "requirements": ["closure", "C2", "fixed duration", "fixed branch endpoints", "no clipping", "finite q/dq/ddq"],
            "definitions": prototypes.prototype_definitions(),
        },
        "structural_comparison": {
            "score_values": {"0": "structurally unsupported/poor", "1": "conditional/moderate", "2": "direct/strong"},
            "weights": "equal and frozen",
            "rows": option_rows,
            "outcome_diversity_or_new_J_used": False,
            "dimension_target": "2-4 preferred, <=5 audited, >10 TOO_HIGH_DIMENSIONAL_FOR_LOW_BUDGET_PERSONALIZATION_V1",
        },
        "decision_rules": {
            "ROM_effect": "DOMINANT if R2>=0.90; SUBSTANTIAL if >=0.75; otherwise LIMITED",
            "fixed_ROM_identifiability": "CURRENT_GRID_CANNOT_IDENTIFY_FIXED_ROM_PATH_EFFECT if fewer than two independent path-shape coordinates vary within matched-ROM groups",
            "V2_inadequate": "task amplitude changes AND broad fixed-ROM path effect is not identifiable AND phase-only variation remains common-monotonic",
            "V3_primary": "highest preregistered structural score that preserves extrema, closure, C2 and branch asymmetry with 2-4 parameters",
            "V3_fallback": "next highest structurally distinct option satisfying those invariants",
            "next_branch": "Branch A when V2 is inadequate and a structurally better parameterization exists; otherwise Branch B if objective remains primary, else Branch C if heterogeneity remains primary",
        },
        "scope_guards": {
            "candidate_domain_modified": False, "cohort_modified": False, "truth_landscape_modified": False,
            "reference_modified": False, "objective_modified": False, "normalization_modified": False,
            "subject_ranges_modified": False, "myoleg_model_modified": False, "five_parameter": False,
            "nn_or_pinn": False, "bo": False, "held_out_truth": False,
            "new_v3_landscape": False, "robot_or_hardware": False,
        },
    }


def freeze_protocol() -> None:
    if OUTPUT.exists():
        raise RuntimeError("audit output already exists; refusing to overwrite protocol freeze")
    truth, candidates, cohort = verify_frozen_inputs()
    access = held_out_hash_audit(truth)
    protocol = protocol_payload(candidates, cohort)
    OUTPUT.mkdir(parents=True)
    atomic_json(PROTOCOL_PATH, protocol)
    access.update({
        "stage_id": STAGE_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_scientific_values_read_at_freeze": False,
    })
    atomic_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json", access)
    print(json.dumps({
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_count": EXPECTED_CANDIDATES,
        "development_scientific_values_read": False,
        "held_out_scientific_truth_access_count": 0,
    }, indent=2))


def load_frozen_generator() -> Any:
    """Import the exact frozen source while disabling unused simulator import.

    Only the kinematic functions listed in the protocol are called.  The source
    identity is verified first.  A minimal ``mujoco`` module is installed only
    when the analysis Python lacks MyoLeg; no simulator-dependent function is
    then reachable from this stage.
    """

    if sha256_file(GENERATOR_SOURCE_PATH) != FROZEN_SHA["generator_source"]:
        raise RuntimeError("frozen generator source changed")
    prior = sys.modules.get("mujoco")
    inserted = False
    if prior is None:
        try:
            __import__("mujoco")
        except ModuleNotFoundError:
            stub = types.ModuleType("mujoco")
            stub.__version__ = "KINEMATIC_ONLY_NOT_LOADED"
            sys.modules["mujoco"] = stub
            inserted = True
    spec = importlib.util.spec_from_file_location("_frozen_v2_candidate_generator", GENERATOR_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen candidate generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.modules.pop("mujoco", None)
    return module


def time_weighted_rms(time_s: np.ndarray, values: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    return float(np.sqrt(np.trapezoid(np.asarray(values, dtype=float) ** 2, time_s) / duration))


def _midpoint_time(time_s: np.ndarray, values: np.ndarray, indices: np.ndarray) -> float:
    branch_values = np.asarray(values[indices], dtype=float)
    start, end = float(branch_values[0]), float(branch_values[-1])
    if abs(end - start) <= 1.0e-12:
        return math.nan
    progress = (branch_values - start) / (end - start)
    order = np.argsort(progress, kind="stable")
    unique, unique_index = np.unique(progress[order], return_index=True)
    ordered_time = time_s[indices][order][unique_index]
    return float(np.interp(0.5, unique, ordered_time))


def trajectory_descriptors(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, float | bool]:
    time_s = np.asarray(reference["time_s"], dtype=float)
    phases = np.asarray(reference["phases"])
    q = np.asarray(candidate["q"], dtype=float)
    dq = np.asarray(candidate["dq"], dtype=float)
    ddq = np.asarray(candidate["ddq"], dtype=float)
    q_deg = np.degrees(q)
    dq_deg = np.degrees(dq)
    ddq_deg = np.degrees(ddq)
    flex = np.flatnonzero(phases == "flexion")
    extension = np.flatnonzero(phases == "extension")
    diff = np.diff(q_deg, axis=0)
    path_length = float(np.sum(np.linalg.norm(diff, axis=1)))
    signed_area = float(0.5 * np.sum(q_deg[:-1, 0] * q_deg[1:, 1] - q_deg[1:, 0] * q_deg[:-1, 1]))
    hip_range = float(np.ptp(q_deg[:, 0]))
    knee_range = float(np.ptp(q_deg[:, 1]))
    direct_scale = max(float(np.hypot(hip_range, knee_range)), np.finfo(float).tiny)

    slopes: dict[str, float] = {}
    for branch, indices in (("flexion", flex), ("extension", extension)):
        x = q_deg[indices, 0]
        y = q_deg[indices, 1]
        design = np.column_stack((x, np.ones_like(x)))
        slopes[branch] = float(np.linalg.lstsq(design, y, rcond=None)[0][0])

    finite = bool(np.isfinite(np.column_stack((q, dq, ddq))).all())
    return {
        "hip_min_deg": float(np.min(q_deg[:, 0])),
        "hip_max_deg": float(np.max(q_deg[:, 0])),
        "hip_range_deg": hip_range,
        "knee_min_deg": float(np.min(q_deg[:, 1])),
        "knee_max_deg": float(np.max(q_deg[:, 1])),
        "knee_range_deg": knee_range,
        "cycle_duration_s": float(time_s[-1] - time_s[0]),
        "flexion_duration_s": float(time_s[flex[-1]] - time_s[flex[0]]),
        "extension_duration_s": float(time_s[extension[-1]] - time_s[extension[0]]),
        "joint_space_path_length_deg": path_length,
        "normalized_joint_space_path_length": path_length / direct_scale,
        "signed_qspace_area_deg2": signed_area,
        "absolute_qspace_area_deg2": abs(signed_area),
        "flexion_coordination_slope": slopes["flexion"],
        "extension_coordination_slope": slopes["extension"],
        "flexion_midpoint_lag_s": _midpoint_time(time_s, q_deg[:, 1], flex) - _midpoint_time(time_s, q_deg[:, 0], flex),
        "extension_midpoint_lag_s": _midpoint_time(time_s, q_deg[:, 1], extension) - _midpoint_time(time_s, q_deg[:, 0], extension),
        "hip_peak_abs_dq_deg_s": float(np.max(np.abs(dq_deg[:, 0]))),
        "knee_peak_abs_dq_deg_s": float(np.max(np.abs(dq_deg[:, 1]))),
        "hip_rms_dq_deg_s": time_weighted_rms(time_s, dq_deg[:, 0]),
        "knee_rms_dq_deg_s": time_weighted_rms(time_s, dq_deg[:, 1]),
        "hip_peak_abs_ddq_deg_s2": float(np.max(np.abs(ddq_deg[:, 0]))),
        "knee_peak_abs_ddq_deg_s2": float(np.max(np.abs(ddq_deg[:, 1]))),
        "hip_rms_ddq_deg_s2": time_weighted_rms(time_s, ddq_deg[:, 0]),
        "knee_rms_ddq_deg_s2": time_weighted_rms(time_s, ddq_deg[:, 1]),
        "q_closure_error_rad": float(np.max(np.abs(q[-1] - q[0]))),
        "dq_closure_error_rad_s": float(np.max(np.abs(dq[-1] - dq[0]))),
        "ddq_closure_error_rad_s2": float(np.max(np.abs(ddq[-1] - ddq[0]))),
        "finite": finite,
    }


def build_descriptor_rows(generator: Any, candidate_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = generator.load_reference_adapter()
    rows: list[dict[str, Any]] = []
    reference_q = np.asarray(reference["q"])
    time_s = np.asarray(reference["time_s"])
    for record in candidate_manifest["ordered_included_candidates"]:
        hip, knee, phase = map(float, record["alpha"])
        candidate = generator.generate_candidate(reference, hip, knee, phase)
        descriptor = trajectory_descriptors(candidate, reference)
        q_delta_deg = np.degrees(np.asarray(candidate["q"]) - reference_q)
        descriptor.update({
            "qspace_rms_deviation_from_reference_deg": float(np.sqrt(np.mean(np.sum(q_delta_deg**2, axis=1)))),
            "qspace_peak_deviation_from_reference_deg": float(np.max(np.linalg.norm(q_delta_deg, axis=1))),
            "phase_warp_min_derivative": float(np.min(candidate["warp_first"])),
            "phase_warp_max_derivative": float(np.max(candidate["warp_first"])),
        })
        rows.append({
            "candidate_id": record["candidate_id"],
            "proposal_index": int(record["proposal_index"]),
            "alpha_hip_deg": hip,
            "alpha_knee_deg": knee,
            "alpha_phase": phase,
            **descriptor,
        })
    if len(rows) != EXPECTED_CANDIDATES or any(not row["finite"] for row in rows):
        raise RuntimeError("candidate descriptor generation failed")
    reference_descriptor = trajectory_descriptors(
        {"q": reference["q"], "dq": reference["dq"], "ddq": reference["ddq"]}, reference
    )
    return rows, {"reference": reference, "reference_descriptor": reference_descriptor, "time_s": time_s}


def alpha_lookup(rows: list[dict[str, Any]]) -> dict[tuple[float, float, float], dict[str, Any]]:
    return {
        tuple(round(float(row[key]), 10) for key in ("alpha_hip_deg", "alpha_knee_deg", "alpha_phase")): row
        for row in rows
    }


def parameter_semantics_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = alpha_lookup(rows)
    definitions = (
        ("delta_hip_amp", -5.0, 2.0, "q_hip=q_hip_ref+radians(alpha_hip)*minimum_jerk_amplitude_basis"),
        ("delta_knee_amp", -5.0, 0.5, "q_knee=q_knee_ref(warped_phase)+radians(alpha_knee)*minimum_jerk_amplitude_basis"),
        ("knee_phase_shift", -0.03, 0.03, "knee branch phase r'=r+alpha_phase*64*r^3*(1-r)^3"),
    )
    output: list[dict[str, Any]] = []
    selected_metrics = (
        "hip_min_deg", "hip_max_deg", "hip_range_deg", "knee_min_deg", "knee_max_deg", "knee_range_deg",
        "joint_space_path_length_deg", "signed_qspace_area_deg2", "flexion_midpoint_lag_s", "extension_midpoint_lag_s",
        "hip_peak_abs_dq_deg_s", "knee_peak_abs_dq_deg_s", "hip_rms_dq_deg_s", "knee_rms_dq_deg_s",
        "hip_peak_abs_ddq_deg_s2", "knee_peak_abs_ddq_deg_s2", "hip_rms_ddq_deg_s2", "knee_rms_ddq_deg_s2",
    )
    for index, (name, lower, upper, formula) in enumerate(definitions):
        low_alpha = [0.0, 0.0, 0.0]
        high_alpha = [0.0, 0.0, 0.0]
        low_alpha[index], high_alpha[index] = lower, upper
        low = lookup[tuple(low_alpha)]
        high = lookup[tuple(high_alpha)]
        row: dict[str, Any] = {
            "parameter": name,
            "admitted_lower": lower,
            "admitted_upper": upper,
            "actual_generator_formula": formula,
            "changes_cycle_duration": False,
            "changes_flexion_extension_branch_duration": False,
            "changes_cycle_endpoint": False,
            "changes_turning_point": name != "knee_phase_shift",
            "preserves_closure": True,
            "preserves_C2": True,
            "changes_rehabilitation_task_amplitude": name in {"delta_hip_amp", "delta_knee_amp"},
        }
        for metric in selected_metrics:
            delta = float(high[metric]) - float(low[metric])
            row[f"low_{metric}"] = float(low[metric])
            row[f"high_{metric}"] = float(high[metric])
            row[f"delta_{metric}"] = delta
        output.append(row)
    return output


def prototype_audit(generator: Any, reference: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    magnitude = 0.25
    p1_raw = generator.generate_candidate(reference, 0.0, 0.0, 0.0025)
    p1 = prototypes.PrototypeTrajectory(
        q=p1_raw["q"], dq=p1_raw["dq"], ddq=p1_raw["ddq"],
        parameterization_id="P1_PHASE_COORDINATION_ONLY", parameters={"knee_phase_shift": 0.0025},
    )
    p2 = prototypes.p2_interior_bspline(
        reference, hip_flex_deg=magnitude, hip_extension_deg=-magnitude,
        knee_flex_deg=-magnitude, knee_extension_deg=magnitude,
    )
    p3, degeneracy = prototypes.p3_joint_space_normal(
        reference, flex_normal_deg=magnitude, extension_normal_deg=-magnitude,
    )
    p4 = prototypes.p4_coordination_function(
        reference, flex_knee_coordination_deg=magnitude, extension_knee_coordination_deg=-magnitude,
    )
    reference_descriptor = trajectory_descriptors(
        {"q": reference["q"], "dq": reference["dq"], "ddq": reference["ddq"]}, reference
    )
    output: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for item in (p1, p2, p3, p4):
        descriptor = trajectory_descriptors({"q": item.q, "dq": item.dq, "ddq": item.ddq}, reference)
        extrema_delta = max(
            abs(float(descriptor[key]) - float(reference_descriptor[key]))
            for key in ("hip_min_deg", "hip_max_deg", "knee_min_deg", "knee_max_deg")
        )
        output.append({
            "parameterization_id": item.parameterization_id,
            "dimension": next(row["dimension"] for row in prototypes.prototype_definitions() if row["parameterization_id"] == item.parameterization_id),
            "diagnostic_parameters_json": json.dumps(item.parameters, sort_keys=True),
            "diagnostic_values_are_formal_bounds": False,
            "finite": bool(descriptor["finite"]),
            "q_closure_error_rad": descriptor["q_closure_error_rad"],
            "dq_closure_error_rad_s": descriptor["dq_closure_error_rad_s"],
            "ddq_closure_error_rad_s2": descriptor["ddq_closure_error_rad_s2"],
            "maximum_extrema_change_deg": extrema_delta,
            "extrema_invariant_at_diagnostic_point": extrema_delta <= PROTOTYPE_EXTREMA_TOLERANCE_DEG,
            "cycle_duration_change_s": float(descriptor["cycle_duration_s"]) - float(reference_descriptor["cycle_duration_s"]),
            "path_length_change_deg": float(descriptor["joint_space_path_length_deg"]) - float(reference_descriptor["joint_space_path_length_deg"]),
            "peak_dq_ratio": max(
                float(descriptor["hip_peak_abs_dq_deg_s"]) / float(reference_descriptor["hip_peak_abs_dq_deg_s"]),
                float(descriptor["knee_peak_abs_dq_deg_s"]) / float(reference_descriptor["knee_peak_abs_dq_deg_s"]),
            ),
            "peak_ddq_ratio": max(
                float(descriptor["hip_peak_abs_ddq_deg_s2"]) / float(reference_descriptor["hip_peak_abs_ddq_deg_s2"]),
                float(descriptor["knee_peak_abs_ddq_deg_s2"]) / float(reference_descriptor["knee_peak_abs_ddq_deg_s2"]),
            ),
            "normal_low_speed_degeneracy_fraction": degeneracy if item.parameterization_id.startswith("P3_") else 0.0,
            "post_generation_clipping_used": False,
            "myoleg_objective_evaluated": False,
        })
        trajectories[item.parameterization_id] = {"q": item.q, "dq": item.dq, "ddq": item.ddq}
    return output, trajectories


def load_root_cause_module() -> Any:
    path = ROOT / "external_simulation/myoleg_v2_personalization_signal_root_cause_audit_v1/build_audit.py"
    expected = read_json(ROOT_CAUSE_METADATA_PATH)["analysis_code_sha256"]
    if sha256_file(path) != expected:
        raise RuntimeError("frozen root-cause analysis source changed")
    spec = importlib.util.spec_from_file_location("_frozen_root_cause_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen root-cause analysis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_development_matrix(
    truth: dict[str, Any], candidates: dict[str, Any], cohort: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    root_cause = load_root_cause_module()
    development_ids = list(cohort["development_subject_ids"])
    store = root_cause.DevelopmentLandscapeStore(truth, candidates, development_ids)
    loaded = [store.load_subject(subject_id) for subject_id in development_ids]
    if store.accessed_subject_ids != development_ids:
        raise RuntimeError("development access audit mismatch")
    j_matrix = np.vstack([row["j_truth"] for row in loaded])
    fields = {
        name: np.vstack([row[name] for row in loaded])
        for name in (
            "hip_tau_rms_nm", "knee_tau_rms_nm", "subject_reference_hip_rms_nm",
            "subject_reference_knee_rms_nm",
        )
    }
    if j_matrix.shape != (EXPECTED_DEVELOPMENT, EXPECTED_CANDIDATES) or not np.isfinite(j_matrix).all():
        raise RuntimeError("development matrix integrity failure")
    return j_matrix, fields, development_ids


def two_way_metrics(values: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(values, dtype=float)
    grand = float(np.mean(matrix))
    subject = np.mean(matrix, axis=1) - grand
    candidate = np.mean(matrix, axis=0) - grand
    interaction = matrix - grand - subject[:, None] - candidate[None, :]
    centered = matrix - grand
    total = max(float(np.sum(centered**2)), np.finfo(float).tiny)
    return {
        "grand_mean": grand,
        "subject_variance_fraction": float(matrix.shape[1] * np.sum(subject**2) / total),
        "candidate_variance_fraction": float(matrix.shape[0] * np.sum(candidate**2) / total),
        "interaction_variance_fraction": float(np.sum(interaction**2) / total),
        "candidate_main_rms": float(np.sqrt(np.mean(candidate**2))),
        "interaction_rms": float(np.sqrt(np.mean(interaction**2))),
    }


def standardize_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
    return (matrix - mean) / scale, mean, scale


def fit_ols(name: str, target_name: str, features: np.ndarray, feature_names: list[str], target: np.ndarray) -> dict[str, Any]:
    standardized, _, _ = standardize_columns(features)
    design = np.column_stack((np.ones(len(standardized)), standardized))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    residual = target - predicted
    sse = float(np.sum(residual**2))
    sst = max(float(np.sum((target - np.mean(target)) ** 2)), np.finfo(float).tiny)
    r2 = 1.0 - sse / sst
    n = len(target)
    p = int(rank) - 1
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "analysis_type": "MULTIVARIATE_DESCRIPTIVE_OLS",
        "target": target_name,
        "descriptor": ";".join(feature_names),
        "method": name,
        "sample_count": n,
        "pearson_r": "",
        "spearman_rho": "",
        "raw_p_value": "",
        "r2": r2,
        "adjusted_r2": adjusted,
        "rmse": rmse,
        "normalized_rmse_by_target_sd": rmse / max(float(np.std(target)), np.finfo(float).tiny),
        "feature_count": len(feature_names),
        "design_rank_including_intercept": int(rank),
        "causal_claim": False,
        "production_learner": False,
    }


def common_effect_audit(
    descriptor_rows: list[dict[str, Any]], j_matrix: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    mean_j = np.mean(j_matrix, axis=0)
    candidate_main = mean_j - float(np.mean(mean_j))
    descriptors = (
        "hip_max_deg", "hip_range_deg", "knee_max_deg", "knee_range_deg", "alpha_phase",
        "joint_space_path_length_deg", "absolute_qspace_area_deg2", "flexion_midpoint_lag_s",
        "extension_midpoint_lag_s", "hip_rms_dq_deg_s", "knee_rms_dq_deg_s",
        "hip_rms_ddq_deg_s2", "knee_rms_ddq_deg_s2",
    )
    rows: list[dict[str, Any]] = []
    for target_name, target in (("mean_development_J", mean_j), ("candidate_main_effect", candidate_main)):
        for descriptor in descriptors:
            x = np.asarray([float(row[descriptor]) for row in descriptor_rows])
            pearson = stats.pearsonr(x, target)
            spearman = stats.spearmanr(x, target)
            rows.append({
                "analysis_type": "UNIVARIATE_ASSOCIATION",
                "target": target_name,
                "descriptor": descriptor,
                "method": "Pearson_and_Spearman",
                "sample_count": len(x),
                "pearson_r": float(pearson.statistic),
                "spearman_rho": float(spearman.statistic),
                "raw_p_value": float(spearman.pvalue),
                "r2": float(pearson.statistic**2),
                "adjusted_r2": "",
                "rmse": "",
                "normalized_rmse_by_target_sd": "",
                "feature_count": 1,
                "design_rank_including_intercept": 2,
                "causal_claim": False,
                "production_learner": False,
            })

    hip = np.asarray([float(row["hip_range_deg"]) for row in descriptor_rows])
    knee = np.asarray([float(row["knee_range_deg"]) for row in descriptor_rows])
    phase = np.asarray([float(row["alpha_phase"]) for row in descriptor_rows])
    h, _, _ = standardize_columns(hip[:, None]); h = h[:, 0]
    k, _, _ = standardize_columns(knee[:, None]); k = k[:, 0]
    p, _, _ = standardize_columns(phase[:, None]); p = p[:, 0]
    rom_features = np.column_stack((h, k, h**2, k**2, h*k))
    rom_names = ["hip_ROM_z", "knee_ROM_z", "hip_ROM_z2", "knee_ROM_z2", "hip_ROM_x_knee_ROM"]
    timing_features = np.column_stack((h, k, p, h**2, k**2, p**2, h*k, h*p, k*p))
    timing_names = ["hip_ROM_z", "knee_ROM_z", "phase_z", "hip_ROM_z2", "knee_ROM_z2", "phase_z2", "hip_x_knee", "hip_x_phase", "knee_x_phase"]
    full_names = [
        "hip_range_deg", "knee_range_deg", "alpha_phase", "joint_space_path_length_deg",
        "absolute_qspace_area_deg2", "flexion_midpoint_lag_s", "extension_midpoint_lag_s",
        "hip_rms_dq_deg_s", "knee_rms_dq_deg_s", "hip_rms_ddq_deg_s2", "knee_rms_ddq_deg_s2",
    ]
    full_features = np.column_stack([[float(row[name]) for row in descriptor_rows] for name in full_names])
    model_rows: list[dict[str, Any]] = []
    for target_name, target in (("mean_development_J", mean_j), ("candidate_main_effect", candidate_main)):
        model_rows.extend((
            fit_ols("ROM_EXTREMA_QUADRATIC", target_name, rom_features, rom_names, target),
            fit_ols("ROM_EXTREMA_PLUS_PHASE_QUADRATIC", target_name, timing_features, timing_names, target),
            fit_ols("FIXED_KINEMATIC_DESCRIPTOR_LINEAR", target_name, full_features, full_names, target),
        ))
    rows.extend(model_rows)
    primary = next(row for row in model_rows if row["target"] == "candidate_main_effect" and row["method"] == "ROM_EXTREMA_QUADRATIC")
    r2 = float(primary["r2"])
    status = "DOMINANT" if r2 >= 0.90 else "SUBSTANTIAL" if r2 >= 0.75 else "LIMITED"
    summary = {
        "audit_id": "ROM_EXTREMA_COMMON_EFFECT_AUDIT_V1",
        "ROM_EXTREMA_EXPLAINED_COMMON_EFFECT": r2,
        "interpretation": status,
        "primary_model": "ROM_EXTREMA_QUADRATIC",
        "primary_model_features": rom_names,
        "candidate_count": EXPECTED_CANDIDATES,
        "development_subject_count": EXPECTED_DEVELOPMENT,
        "association_is_not_causality": True,
        "production_learner_trained": False,
        "secondary_model_results": {
            row["method"]: float(row["r2"])
            for row in model_rows if row["target"] == "candidate_main_effect"
        },
    }
    return rows, summary, mean_j, candidate_main


def pairwise_rank_metrics(matrix: np.ndarray) -> tuple[float, float]:
    values = np.asarray(matrix, dtype=float)
    upper_candidate = np.triu_indices(values.shape[1], 1)
    signs = np.sign(values[:, upper_candidate[0]] - values[:, upper_candidate[1]])
    # Ties are extremely rare; exact pair agreement remains explicit if present.
    subject_upper = np.triu_indices(values.shape[0], 1)
    taus: list[float] = []
    inversions: list[float] = []
    for left, right in zip(*subject_upper):
        comparable = (signs[left] != 0) & (signs[right] != 0)
        if not np.any(comparable):
            taus.append(0.0); inversions.append(0.0)
            continue
        disagreement = float(np.mean(signs[left, comparable] != signs[right, comparable]))
        inversions.append(disagreement)
        taus.append(1.0 - 2.0 * disagreement)
    return float(np.mean(taus)), float(np.mean(inversions))


def matched_rom_analysis(
    descriptor_rows: list[dict[str, Any]], j_matrix: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[list[int]] = []
    keys = ("hip_min_deg", "hip_max_deg", "knee_min_deg", "knee_max_deg")
    for index, row in enumerate(descriptor_rows):
        if not groups:
            groups.append([index])
            continue
        representative = descriptor_rows[groups[-1][0]]
        if all(abs(float(row[name]) - float(representative[name])) <= ROM_EQUIVALENCE_TOLERANCE_DEG for name in keys):
            groups[-1].append(index)
        else:
            groups.append([index])
    output: list[dict[str, Any]] = []
    for group_index, indices in enumerate(groups):
        phases = np.asarray([float(descriptor_rows[index]["alpha_phase"]) for index in indices])
        if len(indices) < 5 or len(np.unique(phases)) < 5:
            continue
        values = j_matrix[:, indices]
        decomposition = two_way_metrics(values)
        mean_tau, inversion = pairwise_rank_metrics(values)
        mean_values = np.mean(values, axis=0)
        common_local = int(np.argmin(mean_values))
        individual_local = np.argmin(values, axis=1)
        regret = values[:, common_local] - np.min(values, axis=1)
        independent = sum(
            float(np.ptp([float(descriptor_rows[index][name]) for index in indices])) > KINEMATIC_EQUALITY_TOLERANCE
            for name in ("alpha_hip_deg", "alpha_knee_deg", "alpha_phase")
        )
        output.append({
            "matched_rom_group_id": f"MATCHED_ROM_{group_index:04d}",
            "candidate_count": len(indices),
            "alpha_hip_deg": float(descriptor_rows[indices[0]]["alpha_hip_deg"]),
            "alpha_knee_deg": float(descriptor_rows[indices[0]]["alpha_knee_deg"]),
            "phase_min": float(np.min(phases)),
            "phase_max": float(np.max(phases)),
            "hip_extrema_span_within_group_deg": max(
                float(np.ptp([float(descriptor_rows[index][name]) for index in indices])) for name in ("hip_min_deg", "hip_max_deg")
            ),
            "knee_extrema_span_within_group_deg": max(
                float(np.ptp([float(descriptor_rows[index][name]) for index in indices])) for name in ("knee_min_deg", "knee_max_deg")
            ),
            "independent_alpha_dimensions_varying": independent,
            "path_length_range_deg": float(np.ptp([float(descriptor_rows[index]["joint_space_path_length_deg"]) for index in indices])),
            "qspace_area_range_deg2": float(np.ptp([float(descriptor_rows[index]["signed_qspace_area_deg2"]) for index in indices])),
            "flexion_midpoint_lag_range_s": float(np.ptp([float(descriptor_rows[index]["flexion_midpoint_lag_s"]) for index in indices])),
            "pairwise_subject_kendall_tau_mean": mean_tau,
            "pairwise_rank_inversion_fraction_mean": inversion,
            "subject_candidate_interaction_variance_fraction": decomposition["interaction_variance_fraction"],
            "candidate_main_variance_fraction": decomposition["candidate_variance_fraction"],
            "common_optimum_candidate_id": descriptor_rows[indices[common_local]]["candidate_id"],
            "common_optimum_phase": float(phases[common_local]),
            "subjects_sharing_common_optimum": int(np.sum(individual_local == common_local)),
            "mean_common_optimum_regret": float(np.mean(regret)),
            "max_common_optimum_regret": float(np.max(regret)),
        })
    if not output:
        raise RuntimeError("no matched-ROM groups found")
    maximum_dimensions = max(int(row["independent_alpha_dimensions_varying"]) for row in output)
    status = (
        "CURRENT_GRID_CANNOT_IDENTIFY_FIXED_ROM_PATH_EFFECT"
        if maximum_dimensions < 2 else "CURRENT_GRID_CONTAINS_MULTIDIMENSIONAL_FIXED_ROM_PATH_VARIATION"
    )
    summary = {
        "matched_group_count": len(output),
        "candidate_count_per_complete_group_min": min(int(row["candidate_count"]) for row in output),
        "candidate_count_per_complete_group_max": max(int(row["candidate_count"]) for row in output),
        "maximum_independent_path_shape_dimensions_within_matched_ROM": maximum_dimensions,
        "status": status,
        "qualified_statement": "the grid identifies its one-dimensional phase warp at fixed ROM, but cannot identify broader fixed-ROM path-shape effects",
    }
    return output, summary


def phase_only_analysis(
    descriptor_rows: list[dict[str, Any]], j_matrix: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = alpha_lookup(descriptor_rows)
    output: list[dict[str, Any]] = []
    phase_axis = np.round(np.arange(-0.03, 0.0301, 0.0025), 10)
    common_pass = True
    for pair_index, (hip, knee) in enumerate(PHASE_AMPLITUDE_PAIRS):
        indices = [int(lookup[(hip, knee, float(phase))]["proposal_index"]) for phase in phase_axis]
        # Proposal indices are not compact candidate ranks; resolve by descriptor order.
        ids = [lookup[(hip, knee, float(phase))]["candidate_id"] for phase in phase_axis]
        id_to_rank = {row["candidate_id"]: rank for rank, row in enumerate(descriptor_rows)}
        ranks = [id_to_rank[candidate_id] for candidate_id in ids]
        values = j_matrix[:, ranks]
        differences = np.diff(values, axis=1)
        increasing_subject_count = int(np.sum(np.all(differences > 0.0, axis=1)))
        decreasing_subject_count = int(np.sum(np.all(differences < 0.0, axis=1)))
        common_pass = common_pass and increasing_subject_count >= 23
        decomposition = two_way_metrics(values)
        mean_tau, inversion = pairwise_rank_metrics(values)
        mean_values = np.mean(values, axis=0)
        common_local = int(np.argmin(mean_values))
        individual = np.argmin(values, axis=1)
        reference_phase_index = int(np.flatnonzero(np.isclose(phase_axis, 0.0))[0])
        output.append({
            "phase_slice_id": f"PHASE_SLICE_{pair_index + 1:02d}",
            "alpha_hip_deg": hip,
            "alpha_knee_deg": knee,
            "phase_value_count": len(phase_axis),
            "common_oracle_phase": float(phase_axis[common_local]),
            "subjects_sharing_common_oracle": int(np.sum(individual == common_local)),
            "unique_subject_oracle_phases_json": json.dumps(sorted(set(float(phase_axis[index]) for index in individual))),
            "subjects_monotonic_J_increasing_with_phase": increasing_subject_count,
            "subjects_monotonic_J_decreasing_with_phase": decreasing_subject_count,
            "positive_adjacent_difference_fraction": float(np.mean(differences > 0.0)),
            "negative_adjacent_difference_fraction": float(np.mean(differences < 0.0)),
            "pairwise_subject_kendall_tau_mean": mean_tau,
            "pairwise_rank_inversion_fraction_mean": inversion,
            "subject_phase_interaction_variance_fraction": decomposition["interaction_variance_fraction"],
            "candidate_phase_main_variance_fraction": decomposition["candidate_variance_fraction"],
            "mean_J_phase_low_minus_reference": float(np.mean(values[:, 0] - values[:, reference_phase_index])),
            "mean_J_phase_high_minus_reference": float(np.mean(values[:, -1] - values[:, reference_phase_index])),
            "mean_J_phase_range": float(np.ptp(mean_values)),
        })
    summary = {
        "phase_slice_count": len(output),
        "phase_value_count_per_slice": len(phase_axis),
        "all_preregistered_slices_common_monotonic": common_pass,
        "direction": "phase increase worsens J" if common_pass else "not uniform",
        "development_subject_count": EXPECTED_DEVELOPMENT,
    }
    return output, summary


def phase_mechanistic_decomposition(descriptor_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(ROOT_CAUSE_REPLAY_CACHE_PATH) != FROZEN_SHA["root_cause_replay_cache"]:
        raise RuntimeError("frozen development replay cache changed")
    descriptor_lookup = alpha_lookup(descriptor_rows)
    with np.load(ROOT_CAUSE_REPLAY_CACHE_PATH, allow_pickle=False) as cache:
        subject_ids = [str(value) for value in cache["subject_ids"]]
        candidate_ids = [str(value) for value in cache["candidate_ids"]]
        alpha = np.asarray(cache["alpha"], dtype=float)
        time_s = np.asarray(cache["time_s"], dtype=float)
        arrays = {
            "total_tau": np.asarray(cache["tau_truth_nm"], dtype=float),
            "mass": np.asarray(cache["mass_term_nm"], dtype=float),
            "bias_gravity": np.asarray(cache["bias_term_nm"], dtype=float),
            "passive": np.asarray(cache["passive_internal_nm"], dtype=float),
            "zero_control_actuator": np.asarray(cache["actuator_internal_nm"], dtype=float),
            "constraint": np.asarray(cache["constraint_internal_nm"], dtype=float),
        }
    selected: dict[float, int] = {}
    for phase in PHASE_REPLAY_VALUES:
        matches = np.flatnonzero(
            np.isclose(alpha[:, 0], 0.0) & np.isclose(alpha[:, 1], 0.0) & np.isclose(alpha[:, 2], phase)
        )
        if len(matches) != 1:
            raise RuntimeError(f"frozen replay phase candidate unavailable: {phase}")
        selected[phase] = int(matches[0])
    reference_index = selected[0.0]
    output: list[dict[str, Any]] = []
    for subject_index, subject_id in enumerate(subject_ids):
        reference_rms = {
            (name, joint): time_weighted_rms(time_s, values[subject_index, reference_index, :, joint])
            for name, values in arrays.items() for joint in range(2)
        }
        for phase, candidate_index in selected.items():
            descriptor = descriptor_lookup[(0.0, 0.0, phase)]
            for joint, joint_name in enumerate(("hip", "knee")):
                row: dict[str, Any] = {
                    "subject_id": subject_id,
                    "candidate_id": candidate_ids[candidate_index],
                    "alpha_phase": phase,
                    "joint": joint_name,
                    "q_min_deg": descriptor[f"{joint_name}_min_deg"],
                    "q_max_deg": descriptor[f"{joint_name}_max_deg"],
                    "peak_abs_dq_deg_s": descriptor[f"{joint_name}_peak_abs_dq_deg_s"],
                    "rms_dq_deg_s": descriptor[f"{joint_name}_rms_dq_deg_s"],
                    "peak_abs_ddq_deg_s2": descriptor[f"{joint_name}_peak_abs_ddq_deg_s2"],
                    "rms_ddq_deg_s2": descriptor[f"{joint_name}_rms_ddq_deg_s2"],
                    "flexion_midpoint_lag_s": descriptor["flexion_midpoint_lag_s"],
                    "extension_midpoint_lag_s": descriptor["extension_midpoint_lag_s"],
                }
                for name, values in arrays.items():
                    rms = time_weighted_rms(time_s, values[subject_index, candidate_index, :, joint])
                    row[f"{name}_rms_nm"] = rms
                    row[f"{name}_rms_delta_from_phase0_nm"] = rms - reference_rms[(name, joint)]
                output.append(row)
    summary_rows = [row for row in output if float(row["alpha_phase"]) in (-0.03, 0.03)]
    summary: dict[str, Any] = {
        "subject_count": len(subject_ids),
        "phase_value_count": len(selected),
        "row_count": len(output),
        "replay_pair_count_used": len(subject_ids) * len(selected),
        "held_out_replay_pair_count": 0,
        "component_extreme_deltas": {},
    }
    for joint in ("hip", "knee"):
        for component in arrays:
            low = [row for row in summary_rows if row["joint"] == joint and float(row["alpha_phase"]) == -0.03]
            high = [row for row in summary_rows if row["joint"] == joint and float(row["alpha_phase"]) == 0.03]
            summary["component_extreme_deltas"][f"{joint}_{component}_high_minus_low_rms_nm"] = float(
                np.mean([float(row[f"{component}_rms_nm"]) for row in high])
                - np.mean([float(row[f"{component}_rms_nm"]) for row in low])
            )
    return output, summary


def boundary_source_payload(descriptor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    trusted = read_json(TRUSTED_ROM_PATH)
    lookup = alpha_lookup(descriptor_rows)
    oracle = lookup[(2.0, 0.5, -0.03)]
    return {
        "audit_id": "MYOLEG_V2_BOUNDARY_SOURCE_AUDIT_V1",
        "frozen_common_oracle": {"candidate_id": COMMON_ORACLE_ID, "alpha": [2.0, 0.5, -0.03]},
        "dimensions": [
            {
                "dimension": "hip amplitude upper",
                "oracle_coordinate": 2.0,
                "classification": "A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE",
                "evidence": {
                    "original_proposal_upper": 2.0,
                    "oracle_hip_max_deg": oracle["hip_max_deg"],
                    "trusted_MyoLeg_hip_upper_deg": trusted["trusted_hip_domain_deg"][1],
                    "separation_from_trusted_upper_deg": float(trusted["trusted_hip_domain_deg"][1]) - float(oracle["hip_max_deg"]),
                },
                "prescribed_task_constraint": False,
                "out_of_domain_optimum_claimed": False,
            },
            {
                "dimension": "knee amplitude upper",
                "oracle_coordinate": 0.5,
                "classification": "B_MYOLEG_SIMULATOR_VALIDITY_LIMIT",
                "evidence": {
                    "original_proposal_upper": 2.0,
                    "admitted_upper": 0.5,
                    "oracle_knee_max_deg": oracle["knee_max_deg"],
                    "trusted_MyoLeg_knee_upper_deg": trusted["trusted_knee_upper_deg"],
                    "proposals_excluded_by_knee_upper_trusted_bound": 4350,
                },
                "prescribed_task_constraint": False,
                "out_of_domain_optimum_claimed": False,
            },
            {
                "dimension": "knee phase lower",
                "oracle_coordinate": -0.03,
                "classification": "A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE",
                "evidence": {
                    "original_proposal_lower": -0.03,
                    "all_25_phase_values_pass_frozen_warp_integrity": True,
                    "minimum_phase_warp_derivative_at_boundary": oracle["phase_warp_min_derivative"],
                },
                "prescribed_task_constraint": False,
                "out_of_domain_optimum_claimed": False,
            },
        ],
        "task_constraint_boundary_count": 0,
        "interpretation": "the three active boundaries do not have one common source; two are original proposal bounds and one is the frozen MyoLeg validity bound",
        "widen_current_bounds_as_primary_fix": False,
    }


def invariants_payload() -> dict[str, Any]:
    return {
        "invariant_set_id": "FUTURE_PATH_SHAPE_INVARIANTS_V1",
        "research_question": "subject-specific path/coordination optimization at fixed prescribed rehabilitation task",
        "MUST_PRESERVE": [
            {"quantity": "hip minimum and maximum", "reason": "keeps prescribed hip ROM dosage fixed"},
            {"quantity": "knee minimum and maximum", "reason": "keeps prescribed knee ROM dosage fixed"},
            {"quantity": "cycle duration", "reason": "prevents speed/dose from replacing path-shape personalization"},
            {"quantity": "flexion and extension branch endpoints", "reason": "retains measured asymmetric task and explicit turning states"},
            {"quantity": "cycle q/dq/ddq closure and C2 continuity", "reason": "supports repeatable prescribed execution without boundary impulses"},
            {"quantity": "finite native MyoLeg simulator validity", "reason": "future truth evaluation must remain interpretable"},
            {"quantity": "no pointwise clipping", "reason": "clipping destroys reversible parameter semantics and derivative continuity"},
        ],
        "SHOULD_PRESERVE": [
            {"quantity": "flexion/extension branch durations", "reason": "separates path shape from global branch timing unless timing is an explicitly distinct factor"},
            {"quantity": "measured flexion/extension asymmetry", "reason": "keeps the active reference's empirical structure"},
            {"quantity": "bounded smooth dq/ddq envelopes", "reason": "enables later deterministic pre-execution audit"},
            {"quantity": "locality around the frozen reference", "reason": "supports low-budget inference without changing the task class"},
        ],
        "OPTIONAL": [
            {"quantity": "relative hip-knee phase within each branch", "reason": "a legitimate coordination variable when extrema and durations stay fixed"},
            {"quantity": "interior q-space curvature or loop area", "reason": "candidate subject-specific path-shape coordinates"},
            {"quantity": "branch-specific coordination perturbations", "reason": "may personalize measured asymmetry while retaining endpoints"},
        ],
        "theta_shank_definition": "q_hip - q_knee",
        "formal_ranges_frozen": False,
    }


def option_comparison_rows(prototype_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prototype_by_id = {row["parameterization_id"]: row for row in prototype_rows}
    roles = {
        "P4_BRANCH_AWARE_COORDINATION_FUNCTION": ("V3_PRIMARY_PARAMETERIZATION", "directly varies branch-specific hip-knee coordination while hip task coordinate and all endpoints stay fixed", "requires an independent future range/validity stage"),
        "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION": ("V3_FALLBACK_PARAMETERIZATION", "four reversible branch/joint coefficients provide richer interior shape control", "extrema invariance is conditional on constrained local coefficient ranges"),
        "P1_PHASE_COORDINATION_ONLY": ("LIMITED_EXISTING_BASELINE", "simple, smooth, deterministic, and task preserving", "one shared warp cannot express independent branch-shape changes and is already common-monotonic in development"),
        "P3_JOINT_SPACE_NORMAL_DISPLACEMENT": ("NOT_RECOMMENDED_PRIMARY", "two-dimensional local geometric displacement", "angle-space normal has scale/meaning ambiguity and degenerates near turning points"),
    }
    rows = structural_option_rows()
    ranked = sorted(rows, key=lambda row: (-int(row["structural_score"]), row["parameterization_id"]))
    ranks = {row["parameterization_id"]: index + 1 for index, row in enumerate(ranked)}
    output: list[dict[str, Any]] = []
    for row in rows:
        option_id = row["parameterization_id"]
        role, strength, limitation = roles[option_id]
        prototype = prototype_by_id[option_id]
        output.append({
            **row,
            "structural_rank": ranks[option_id],
            "recommendation_role": role,
            "key_strength": strength,
            "key_limitation": limitation,
            "kinematic_prototype_finite": prototype["finite"],
            "kinematic_prototype_closure_max": max(
                float(prototype["q_closure_error_rad"]),
                float(prototype["dq_closure_error_rad_s"]),
                float(prototype["ddq_closure_error_rad_s2"]),
            ),
            "diagnostic_extrema_change_deg": prototype["maximum_extrema_change_deg"],
            "diagnostic_values_are_formal_bounds": False,
            "new_MyoLeg_J_used_for_selection": False,
        })
    return output


def v3_recommendation_payload(option_rows: list[dict[str, Any]], matched_summary: dict[str, Any], phase_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "MYOLEG_V3_PARAMETERIZATION_RECOMMENDATION_V1",
        "current_v2_decision": V2_DECISION,
        "decision_basis": {
            "current_amplitudes_change_task_ROM": True,
            "matched_ROM_status": matched_summary["status"],
            "phase_only_common_monotonic": phase_summary["all_preregistered_slices_common_monotonic"],
            "selection_used_new_outcome_diversity": False,
            "selection_used_held_out_truth": False,
            "structural_scores_frozen_in_protocol": True,
        },
        "V3_PRIMARY_PARAMETERIZATION": {
            "id": "P4_BRANCH_AWARE_COORDINATION_FUNCTION",
            "parameters": ["flexion interior knee coordination coefficient", "extension interior knee coordination coefficient"],
            "dimension": 2,
            "generation": "q_hip remains unchanged; q_knee receives a branch-wise C2 interior basis perturbation as a function of normalized hip-reference branch phase",
            "invariants": ["joint extrema", "duration", "branch endpoints", "closure", "C2", "measured branch asymmetry", "no clipping"],
            "bounds_strategy": "future independent kinematic and simulator-validity audit; local reversible coefficients centered at zero",
            "numeric_range_frozen": False,
        },
        "V3_FALLBACK_PARAMETERIZATION": {
            "id": "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION",
            "parameters": ["hip flexion interior", "hip extension interior", "knee flexion interior", "knee extension interior"],
            "dimension": 4,
            "generation": "branch-wise degree-6 interior Bezier/B-spline basis with zero value, first derivative, and second derivative at endpoints",
            "invariants": ["duration", "branch endpoints", "closure", "C2", "no clipping"],
            "bounds_strategy": "future constrained local feasibility audit that proves extrema invariance without clipping",
            "numeric_range_frozen": False,
        },
        "parameterization_option_scores": {
            row["parameterization_id"]: row["structural_score"] for row in option_rows
        },
        "next_branch": "BRANCH_A",
        "next_stage": NEXT_STAGE,
        "execute_next_stage_now": False,
    }


def task_invariance_markdown(semantics: list[dict[str, Any]], matched_summary: dict[str, Any]) -> str:
    by_name = {row["parameter"]: row for row in semantics}
    hip = by_name["delta_hip_amp"]
    knee = by_name["delta_knee_amp"]
    phase = by_name["knee_phase_shift"]
    return f"""# Rehabilitation-task invariance audit

## Formal distinction

`TRAJECTORY PERSONALIZATION` should change the interior hip-knee coordination/path while the prescribed rehabilitation task remains fixed. `TASK / ROM DOSAGE MODIFICATION` changes a requested joint extreme or total excursion. Those are scientifically different questions even if both produce valid trajectories.

## What frozen V2 actually changes

- `delta_hip_amp` changes hip maximum by `{float(hip['delta_hip_max_deg']):.6f}` deg and hip ROM by `{float(hip['delta_hip_range_deg']):.6f}` deg across its admitted axis, while cycle duration remains fixed.
- `delta_knee_amp` changes knee maximum by `{float(knee['delta_knee_max_deg']):.6f}` deg and knee ROM by `{float(knee['delta_knee_range_deg']):.6f}` deg across its admitted axis, while cycle duration remains fixed.
- `knee_phase_shift` changes knee timing/path geometry but changes sampled knee maximum by only `{float(phase['delta_knee_max_deg']):.9f}` deg across the full phase axis (below the frozen matched-ROM tolerance); joint extrema are mathematically fixed at branch endpoints.

Therefore:

`CURRENT_PARAMETERIZATION_CHANGES_REHABILITATION_TASK_AMPLITUDE = true`

The two amplitude variables are task/dose variables, not pure path-shape variables. The current matched-ROM grid status is `{matched_summary['status']}`: it supports a one-dimensional phase experiment, not a broad fixed-ROM path-shape experiment.

## Methodological recommendation

For the stated paper question, hip and knee extrema/ROM should be treated as **fixed prescribed-task constraints**. Future personalization should vary only smooth interior coordination/path quantities. This is a design recommendation; it does not modify V2, the active reference, the objective, or any robot setting.
"""


def _figure_path(name: str) -> Path:
    return FIGURES / name


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_axis_slice(
    path: Path,
    descriptor_rows: list[dict[str, Any]],
    j_matrix: np.ndarray,
    axis: str,
    title: str,
    x_label: str,
) -> None:
    alpha_key_name = {"hip": "alpha_hip_deg", "knee": "alpha_knee_deg", "phase": "alpha_phase"}[axis]
    fixed = {
        "hip": ("alpha_knee_deg", "alpha_phase"),
        "knee": ("alpha_hip_deg", "alpha_phase"),
        "phase": ("alpha_hip_deg", "alpha_knee_deg"),
    }[axis]
    indices = [
        index for index, row in enumerate(descriptor_rows)
        if all(abs(float(row[name])) <= 1.0e-12 for name in fixed)
    ]
    indices.sort(key=lambda index: float(descriptor_rows[index][alpha_key_name]))
    x = np.asarray([float(descriptor_rows[index][alpha_key_name]) for index in indices])
    values = j_matrix[:, indices]
    plt.figure(figsize=(7.2, 4.4))
    for subject in range(values.shape[0]):
        plt.plot(x, values[subject], color="0.75", linewidth=0.7, alpha=0.55)
    plt.fill_between(x, np.min(values, axis=0), np.max(values, axis=0), color="#7aa6c2", alpha=0.22, label="development range")
    plt.plot(x, np.mean(values, axis=0), color="#164e63", linewidth=2.2, marker="o", markersize=3.2, label="development mean")
    plt.xlabel(x_label)
    plt.ylabel("Mechanical objective J (dimensionless)")
    plt.title(title)
    plt.gca().ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    _save_figure(path)


def plot_rom_common_effect(path: Path, descriptor_rows: list[dict[str, Any]], candidate_main: np.ndarray) -> None:
    hip = np.asarray([float(row["hip_range_deg"]) for row in descriptor_rows])
    knee = np.asarray([float(row["knee_range_deg"]) for row in descriptor_rows])
    phase = np.asarray([float(row["alpha_phase"]) for row in descriptor_rows])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    first = axes[0].scatter(hip, candidate_main, c=phase, s=5, alpha=0.35, cmap="coolwarm", rasterized=True)
    axes[0].set_xlabel("Hip ROM (deg)")
    axes[0].set_ylabel("Candidate main effect on J")
    axes[0].set_title("Hip ROM and common candidate effect")
    second = axes[1].scatter(knee, candidate_main, c=phase, s=5, alpha=0.35, cmap="coolwarm", rasterized=True)
    axes[1].set_xlabel("Knee ROM (deg)")
    axes[1].set_title("Knee ROM and common candidate effect")
    for axis in axes:
        axis.grid(alpha=0.18)
    colorbar = figure.colorbar(second, ax=axes, shrink=0.88)
    colorbar.set_label("Knee phase shift")
    figure.suptitle("ROM/extrema association with the frozen common effect")
    figure.subplots_adjust(top=0.84, right=0.9, wspace=0.14)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_phase_subjects(path: Path, descriptor_rows: list[dict[str, Any]], j_matrix: np.ndarray) -> None:
    lookup = alpha_lookup(descriptor_rows)
    phase = np.round(np.arange(-0.03, 0.0301, 0.0025), 10)
    id_to_rank = {row["candidate_id"]: index for index, row in enumerate(descriptor_rows)}
    ranks = [id_to_rank[lookup[(0.0, 0.0, float(value))]["candidate_id"]] for value in phase]
    values = j_matrix[:, ranks]
    plt.figure(figsize=(7.2, 4.5))
    for subject in range(values.shape[0]):
        plt.plot(phase, values[subject], linewidth=0.85, alpha=0.55)
    plt.plot(phase, np.mean(values, axis=0), color="black", linewidth=2.4, label="development mean")
    plt.xlabel("Knee phase shift")
    plt.ylabel("Mechanical objective J (dimensionless)")
    plt.title("Fixed-ROM phase-only landscapes: all 24 development subjects")
    plt.gca().ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    _save_figure(path)


def plot_current_qspace_family(path: Path, generator: Any, reference: Mapping[str, np.ndarray]) -> None:
    points = [
        ((0.0, 0.0, 0.0), "reference"), ((-5.0, -5.0, 0.0), "lower amplitudes"),
        ((2.0, 0.5, 0.0), "upper admitted amplitudes"), ((0.0, 0.0, -0.03), "phase lower"),
        ((0.0, 0.0, 0.03), "phase upper"), ((2.0, -2.0, 0.0), "mixed amplitude A"),
        ((-2.0, 0.5, 0.0), "mixed amplitude B"),
    ]
    plt.figure(figsize=(6.8, 5.4))
    for alpha, label in points:
        candidate = generator.generate_candidate(reference, *alpha)
        q = np.degrees(candidate["q"])
        linewidth = 2.5 if label == "reference" else 1.1
        plt.plot(q[:, 0], q[:, 1], linewidth=linewidth, label=label)
    plt.xlabel("Hip flexion (deg)")
    plt.ylabel("Knee flexion (deg)")
    plt.title("Frozen V2 q-space family mixes extrema and timing changes")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, fontsize=8)
    _save_figure(path)


def plot_prototype_comparison(
    path: Path, reference: Mapping[str, np.ndarray], trajectories: dict[str, dict[str, np.ndarray]]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.4), sharex=True)
    ref = np.asarray(reference["q"])
    cycle_phase = np.asarray(reference["global_phase"], dtype=float)
    for axis, option_id in zip(axes.ravel(), sorted(trajectories)):
        delta = np.degrees(trajectories[option_id]["q"] - ref)
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.plot(cycle_phase, delta[:, 0], color="#1d4ed8", linewidth=1.8, label="hip interior change")
        axis.plot(cycle_phase, delta[:, 1], color="#0f766e", linewidth=1.8, label="knee interior change")
        axis.set_title(option_id.replace("_", " "), fontsize=9)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized cycle phase")
    for axis in axes[:, 0]:
        axis.set_ylabel("Change from fixed reference (deg)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("P1-P4 interior perturbations (kinematic-only; no MyoLeg J)")
    figure.subplots_adjust(top=0.9)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_v3_schematic(path: Path, reference: Mapping[str, np.ndarray]) -> None:
    plus = prototypes.p4_coordination_function(
        reference, flex_knee_coordination_deg=0.25, extension_knee_coordination_deg=-0.25
    )
    minus = prototypes.p4_coordination_function(
        reference, flex_knee_coordination_deg=-0.25, extension_knee_coordination_deg=0.25
    )
    ref = np.degrees(np.asarray(reference["q"]))
    q_plus = np.degrees(plus.q); q_minus = np.degrees(minus.q)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.6))
    left.plot(ref[:, 0], ref[:, 1], color="black", linewidth=2.4, label="fixed prescribed reference")
    left.plot(q_plus[:, 0], q_plus[:, 1], color="#0f766e", linewidth=1.7, label="P4 positive")
    left.plot(q_minus[:, 0], q_minus[:, 1], color="#b45309", linewidth=1.7, label="P4 negative")
    for index in (0, int(np.argmax(ref[:, 0]))):
        left.scatter([ref[index, 0]], [ref[index, 1]], color="black", s=32, zorder=5)
    left.annotate("same cycle endpoint", (ref[0, 0], ref[0, 1]), xytext=(ref[0, 0] + 4, ref[0, 1] + 2), arrowprops={"arrowstyle": "->", "lw": 0.8})
    peak = int(np.argmax(ref[:, 0]))
    left.annotate("same turning extrema", (ref[peak, 0], ref[peak, 1]), xytext=(ref[peak, 0] - 37, ref[peak, 1] - 4), arrowprops={"arrowstyle": "->", "lw": 0.8})
    left.set_xlabel("Fixed-task hip flexion (deg)")
    left.set_ylabel("Knee flexion (deg)")
    left.set_title("q-space path")
    left.grid(alpha=0.2)
    left.legend(frameon=False, fontsize=8)
    phase = np.asarray(reference["global_phase"], dtype=float)
    right.axhline(0.0, color="black", linewidth=1.1)
    right.plot(phase, q_plus[:, 1] - ref[:, 1], color="#0f766e", linewidth=2.0, label="positive branch coefficients")
    right.plot(phase, q_minus[:, 1] - ref[:, 1], color="#b45309", linewidth=2.0, label="negative branch coefficients")
    right.set_xlabel("Normalized cycle phase")
    right.set_ylabel("Interior knee coordination change (deg)")
    right.set_title("zero at cycle and turning endpoints")
    right.grid(alpha=0.2)
    right.legend(frameon=False, fontsize=8)
    figure.suptitle("Recommended V3 schematic: change interior coordination, not ROM")
    figure.subplots_adjust(top=0.85, wspace=0.28)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_figures(
    descriptor_rows: list[dict[str, Any]], j_matrix: np.ndarray, candidate_main: np.ndarray,
    generator: Any, reference: Mapping[str, np.ndarray], trajectories: dict[str, dict[str, np.ndarray]],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_axis_slice(_figure_path("01_mean_j_vs_hip_amplitude.png"), descriptor_rows, j_matrix, "hip", "Development J versus hip amplitude", "Hip amplitude increment (deg)")
    plot_axis_slice(_figure_path("02_mean_j_vs_knee_amplitude.png"), descriptor_rows, j_matrix, "knee", "Development J versus knee amplitude", "Knee amplitude increment (deg)")
    plot_axis_slice(_figure_path("03_mean_j_vs_phase.png"), descriptor_rows, j_matrix, "phase", "Development J versus knee phase", "Knee phase shift")
    plot_rom_common_effect(_figure_path("04_rom_extrema_vs_candidate_main.png"), descriptor_rows, candidate_main)
    plot_phase_subjects(_figure_path("05_phase_only_landscapes_24_development.png"), descriptor_rows, j_matrix)
    plot_current_qspace_family(_figure_path("06_current_qspace_candidate_family.png"), generator, reference)
    plot_prototype_comparison(_figure_path("07_p1_p4_kinematic_prototype_comparison.png"), reference, trajectories)
    plot_v3_schematic(_figure_path("08_recommended_v3_parameterization_schematic.png"), reference)


def build_report(
    common: dict[str, Any], matched: dict[str, Any], phase: dict[str, Any], mechanism: dict[str, Any],
    boundary: dict[str, Any], prototype_rows: list[dict[str, Any]], option_rows: list[dict[str, Any]],
    protocol_sha: str, runtime_s: float,
) -> str:
    option = {row["parameterization_id"]: row for row in option_rows}
    proto = {row["parameterization_id"]: row for row in prototype_rows}
    component = mechanism["component_extreme_deltas"]
    return f"""# MyoLeg V2 trajectory parameterization boundary audit V1

## Status and scope

- Stage: `{STAGE_ID}`
- Protocol SHA-256: `{protocol_sha}`
- Development subjects: `24`
- Frozen candidates: `16,675`
- Held-out scientific truth access: `0`
- Outcome: `{OUTCOME}`
- Current V2 decision: `{V2_DECISION}`
- Offline diagnosis/design only; no V3 landscape, learner, BO, robot, or hardware operation was run.

## Q1 — What the three current parameters actually change

`delta_hip_amp` adds the frozen minimum-jerk amplitude basis to hip. It leaves the cycle start/end, duration, branch durations, closure and C2 continuity fixed, but moves the hip turning maximum and therefore changes hip ROM, q-space geometry, velocity and acceleration.

`delta_knee_amp` does the same for knee on top of the phase-warped knee trajectory. It moves the knee turning maximum and knee ROM. Its admitted upper coordinate is only `+0.5 deg`, because higher original proposals cross the frozen MyoLeg 120-degree trusted-domain gate.

`knee_phase_shift` applies `r' = r + shift*64*r^3*(1-r)^3` within both branches. It fixes branch endpoints, extrema, duration, closure and C2, but changes knee timing relative to hip, q-space loop geometry, dq and ddq. It is the only current approximately fixed-ROM coordinate.

## Q2 — How much common candidate effect is associated with ROM/extrema?

`ROM_EXTREMA_EXPLAINED_COMMON_EFFECT = {common['ROM_EXTREMA_EXPLAINED_COMMON_EFFECT']:.6f}` (`{common['interpretation']}`). This is the preregistered ROM-only quadratic descriptive OLS R2. Adding phase raises R2 to `{common['secondary_model_results']['ROM_EXTREMA_PLUS_PHASE_QUADRATIC']:.6f}`, and the fixed kinematic descriptor model reaches `{common['secondary_model_results']['FIXED_KINEMATIC_DESCRIPTOR_LINEAR']:.6f}`.

This is an **association/decomposition result, not a causal estimate**. Nevertheless, together with the exact generator semantics it shows that the dominant common ordering is largely aligned with changing task amplitude/ROM rather than isolated subject-specific path shape.

## Q3 — What remains when ROM/extrema are fixed?

There are `{matched['matched_group_count']}` complete matched-ROM groups, each containing `{matched['candidate_count_per_complete_group_min']}` phase values. But only one independent alpha coordinate varies within any group. Formal status:

`{matched['status']}`

The current grid can identify the behavior of its single phase warp at fixed ROM; it cannot identify whether a richer two-or-more-dimensional interior path family would be subject-specific.

## Q4 — Is phase-only variation common-monotonic, and why?

Across all `{phase['phase_slice_count']}` preregistered amplitude pairs, phase-only J is common-monotonic: `{phase['all_preregistered_slices_common_monotonic']}`. Direction: `{phase['direction']}`. The six-subject deterministic replay subset shows that moving from phase `-0.03` to `+0.03` changes the time placement of q/dq/ddq while extrema remain fixed. Mean high-minus-low component RMS changes include hip bias/gravity `{component['hip_bias_gravity_high_minus_low_rms_nm']:+.6f} Nm`, knee bias/gravity `{component['knee_bias_gravity_high_minus_low_rms_nm']:+.6f} Nm`, hip inertia `{component['hip_mass_high_minus_low_rms_nm']:+.6f} Nm`, and knee inertia `{component['knee_mass_high_minus_low_rms_nm']:+.6f} Nm`.

Thus phase monotonicity is consistent with a shared deterministic timing effect on gravity/bias and inertial demand under the unchanged RMS objective. This mechanism audit does not change phase semantics and does not claim physiology.

## Q5 — Why is the oracle on three boundaries?

- Hip `+2`: `A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE`; the generated hip maximum remains separated from the native/trusted hip upper range.
- Knee `+0.5`: `B_MYOLEG_SIMULATOR_VALIDITY_LIMIT`; original proposals extended to `+2`, but the frozen all-model knee upper-domain admission gate removes them.
- Phase `-0.03`: `A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE`; all 25 frozen phase values pass the phase-warp integrity checks.

None of the three is classified as a prescribed rehabilitation-task constraint. The audit makes no claim about an optimum outside the frozen domain and does not recommend bound widening as the primary fix.

## Q6 — Is V2 optimizing task amplitude rather than pure path shape?

Yes. `CURRENT_PARAMETERIZATION_CHANGES_REHABILITATION_TASK_AMPLITUDE = true`. Two of three coordinates explicitly change target maxima and ROM. V2 therefore mixes task/ROM dosage modification with timing/path modification.

## Q7 — What should be invariant in a cleaner problem?

The future fixed-task problem should **MUST PRESERVE** hip and knee extrema, duration, branch endpoints, q/dq/ddq closure, C2 continuity, finite native simulator validity, and no-clipping generation. It **SHOULD PRESERVE** branch duration and the measured flexion/extension asymmetry. Interior relative coordination, curvature/loop area, and branch-specific smooth deviations may vary.

## Q8 — Which low-dimensional parameterization is structurally best?

The preregistered equal-weight structural scores are P1 `{option['P1_PHASE_COORDINATION_ONLY']['structural_score']}/30`, P2 `{option['P2_INTERIOR_BSPLINE_JOINT_PERTURBATION']['structural_score']}/30`, P3 `{option['P3_JOINT_SPACE_NORMAL_DISPLACEMENT']['structural_score']}/30`, and P4 `{option['P4_BRANCH_AWARE_COORDINATION_FUNCTION']['structural_score']}/30`.

Primary: `P4_BRANCH_AWARE_COORDINATION_FUNCTION` (2 parameters). It leaves hip unchanged as the prescribed task coordinate and changes only knee's smooth branch-interior coordination relation using separate normalized-phase functions for flexion and extension. Direct `q_knee=f(q_hip)` is not used because the measured hip branch is not strictly single-valued. At the diagnostic kinematic point its maximum extrema change is `{float(proto['P4_BRANCH_AWARE_COORDINATION_FUNCTION']['maximum_extrema_change_deg']):.9f} deg`; no clipping or objective evaluation was used.

Fallback: `P2_INTERIOR_BSPLINE_JOINT_PERTURBATION` (4 parameters). Its branch/joint coefficients are interpretable and C2-compatible, but a future independent stage must constrain its local coefficient range so extrema remain fixed without clipping.

P3 is not primary because the Euclidean angle-space normal has coordinate-scale/mechanical-meaning ambiguity and a measured low-speed normal degeneracy fraction of `{float(proto['P3_JOINT_SPACE_NORMAL_DISPLACEMENT']['normal_low_speed_degeneracy_fraction']):.6f}`. P1 is retained as a baseline but is too limited to answer the broader path-shape question.

## Q9 — Is current V2 adequate for the stated personalization question?

`{V2_DECISION}`

V2 remains interpretable as a task-amplitude/timing optimization family, but it is not scientifically adequate as a clean test of subject-specific path optimization at fixed rehabilitation ROM.

## Q10 — Next branch

Recommend Branch A: `{NEXT_STAGE}`. The next stage should formally design and preflight the P4 primary and P2 fallback ranges using kinematic/simulator validity only. It must not be executed automatically and must not select ranges by new oracle diversity.

## Integrity

- Frozen root-cause, truth-landscape, candidate-domain, cohort, objective, normalization, reference and model inputs passed their SHA checks.
- Development-only scientific access: `24` subjects.
- Held-out scientific access: `0`.
- Five-parameter / NN / PINN / BO: not run.
- Robot / hardware: not accessed.
- Runtime: `{runtime_s:.3f} s`.
"""


def write_checksums() -> int:
    files = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in files]
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(lines) + "\n")
    return len(files)


def analyze() -> None:
    started = time.perf_counter()
    if not PROTOCOL_PATH.is_file() or sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("protocol is missing or changed; analysis is fail-closed")
    truth, candidates, cohort = verify_frozen_inputs()
    access = read_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json")
    if access["held_out_scientific_truth_access_count"] != 0:
        raise RuntimeError("held-out sealing failed")

    generator = load_frozen_generator()
    descriptor_rows, kinematic_context = build_descriptor_rows(generator, candidates)
    write_csv(OUTPUT / "CANDIDATE_KINEMATIC_DESCRIPTOR_TABLE.csv", descriptor_rows)
    semantics = parameter_semantics_rows(descriptor_rows)
    write_csv(OUTPUT / "CURRENT_PARAMETER_SEMANTICS_AUDIT.csv", semantics)
    prototype_rows, trajectories = prototype_audit(generator, kinematic_context["reference"])
    write_csv(OUTPUT / "KINEMATIC_PROTOTYPE_AUDIT.csv", prototype_rows)
    atomic_json(OUTPUT / "KINEMATIC_PROTOTYPE_DEFINITIONS.json", {
        "prototype_id": prototypes.PROTOTYPE_ID,
        "source_path": str(PROTOTYPE_SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": sha256_file(PROTOTYPE_SOURCE_PATH),
        "definitions": prototypes.prototype_definitions(),
        "formal_numeric_ranges_frozen": False,
        "objective_evaluated": False,
    })

    # First new scientific-value access occurs here, after protocol SHA lock.
    j_matrix, _, development_ids = load_development_matrix(truth, candidates, cohort)
    common_rows, common_summary, _, candidate_main = common_effect_audit(descriptor_rows, j_matrix)
    write_csv(OUTPUT / "ROM_EXTREMA_COMMON_EFFECT_AUDIT.csv", common_rows)
    atomic_json(OUTPUT / "ROM_EXTREMA_COMMON_EFFECT_AUDIT.json", common_summary)
    matched_rows, matched_summary = matched_rom_analysis(descriptor_rows, j_matrix)
    write_csv(OUTPUT / "MATCHED_ROM_ANALYSIS.csv", matched_rows)
    atomic_json(OUTPUT / "MATCHED_ROM_ANALYSIS_SUMMARY.json", matched_summary)
    phase_rows, phase_summary = phase_only_analysis(descriptor_rows, j_matrix)
    write_csv(OUTPUT / "PHASE_ONLY_SUBSPACE_AUDIT.csv", phase_rows)
    atomic_json(OUTPUT / "PHASE_ONLY_SUBSPACE_SUMMARY.json", phase_summary)
    mechanism_rows, mechanism_summary = phase_mechanistic_decomposition(descriptor_rows)
    write_csv(OUTPUT / "PHASE_MECHANISTIC_DECOMPOSITION.csv", mechanism_rows)
    atomic_json(OUTPUT / "PHASE_MECHANISTIC_DECOMPOSITION_SUMMARY.json", mechanism_summary)

    boundary = boundary_source_payload(descriptor_rows)
    atomic_json(OUTPUT / "BOUNDARY_SOURCE_AUDIT.json", boundary)
    atomic_json(OUTPUT / "FUTURE_PATH_SHAPE_INVARIANTS.json", invariants_payload())
    atomic_text(OUTPUT / "REHABILITATION_TASK_INVARIANCE_AUDIT.md", task_invariance_markdown(semantics, matched_summary))
    option_rows = option_comparison_rows(prototype_rows)
    write_csv(OUTPUT / "PARAMETERIZATION_OPTION_COMPARISON.csv", option_rows)
    recommendation = v3_recommendation_payload(option_rows, matched_summary, phase_summary)
    atomic_json(OUTPUT / "V3_PARAMETERIZATION_RECOMMENDATION.json", recommendation)

    make_figures(
        descriptor_rows, j_matrix, candidate_main, generator,
        kinematic_context["reference"], trajectories,
    )
    runtime_s = time.perf_counter() - started
    report = build_report(
        common_summary, matched_summary, phase_summary, mechanism_summary,
        boundary, prototype_rows, option_rows, EXPECTED_PROTOCOL_SHA256, runtime_s,
    )
    atomic_text(OUTPUT / "MYOLEG_V2_TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_REPORT.md", report)

    access.update({
        "development_scientific_values_read_after_protocol_freeze": True,
        "development_subject_ids_accessed": development_ids,
        "development_subject_count_accessed": len(development_ids),
        "development_candidate_values_accessed": int(j_matrix.size),
        "held_out_scientific_truth_access_count": 0,
        "held_out_replay_pair_count": 0,
        "post_analysis_protocol_sha256": sha256_file(PROTOCOL_PATH),
    })
    atomic_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json", access)

    frozen_after = {name: sha256_file(path) for name, path in frozen_paths().items()}
    if frozen_after != FROZEN_SHA:
        raise RuntimeError("a frozen input changed during analysis")
    artifact_files = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"}
    )
    metadata = {
        "stage_id": STAGE_ID,
        "outcome": OUTCOME,
        "current_v2_parameterization_decision": V2_DECISION,
        "next_stage_recommendation": NEXT_STAGE,
        "next_stage_executed": False,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "frozen_input_sha256_before_and_after": frozen_after,
        "candidate_count": EXPECTED_CANDIDATES,
        "development_subject_count": EXPECTED_DEVELOPMENT,
        "held_out_subject_count": EXPECTED_HELD_OUT,
        "held_out_scientific_truth_access_count": 0,
        "ROM_EXTREMA_EXPLAINED_COMMON_EFFECT": common_summary["ROM_EXTREMA_EXPLAINED_COMMON_EFFECT"],
        "matched_ROM_status": matched_summary["status"],
        "phase_only_common_monotonic": phase_summary["all_preregistered_slices_common_monotonic"],
        "CURRENT_PARAMETERIZATION_CHANGES_REHABILITATION_TASK_AMPLITUDE": True,
        "V3_PRIMARY_PARAMETERIZATION": recommendation["V3_PRIMARY_PARAMETERIZATION"]["id"],
        "V3_FALLBACK_PARAMETERIZATION": recommendation["V3_FALLBACK_PARAMETERIZATION"]["id"],
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "prototype_code_sha256": sha256_file(PROTOTYPE_SOURCE_PATH),
        "artifact_sha256": {path.relative_to(OUTPUT).as_posix(): sha256_file(path) for path in artifact_files},
        "scope": {
            "offline_only": True, "development_only": True, "frozen_v2_modified": False,
            "new_v3_landscape": False, "new_MyoLeg_objective_replay": False,
            "five_parameter": False, "nn_or_pinn": False, "bo": False,
            "held_out_truth": False, "robot_or_hardware": False,
        },
        "runtime_s": runtime_s,
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    checksum_count = write_checksums()
    print(json.dumps({
        "outcome": OUTCOME,
        "v2_decision": V2_DECISION,
        "ROM_EXTREMA_EXPLAINED_COMMON_EFFECT": common_summary["ROM_EXTREMA_EXPLAINED_COMMON_EFFECT"],
        "matched_ROM_status": matched_summary["status"],
        "phase_common_monotonic": phase_summary["all_preregistered_slices_common_monotonic"],
        "next_stage": NEXT_STAGE,
        "held_out_scientific_truth_access_count": 0,
        "artifact_checksum_count": checksum_count,
        "runtime_s": runtime_s,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.freeze_protocol == args.analyze:
        parser.error("choose exactly one of --freeze-protocol or --analyze")
    if args.freeze_protocol:
        freeze_protocol()
    else:
        analyze()


if __name__ == "__main__":
    main()
