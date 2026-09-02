"""Build the outcome-free MyoLeg-V3 trajectory-parameterization design.

The protocol is frozen before the wide kinematic sweep.  This stage reads no
development or held-out scientific truth, computes no mechanical objective,
and performs only a sparse nominal-model simulator-integrity smoke after the
kinematic candidate-domain manifest is frozen.
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
import types
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rehab_robot_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import parameterization


STAGE_ID = "MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_DESIGN_V1"
PROTOCOL_ID = "MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_DESIGN_PROTOCOL_V1"
MANIFEST_ID = "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1"
OUTCOME = "MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_VALID_WITH_LIMITATIONS"
NEXT_STAGE = "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_GENERATION_V1"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1"
FIGURES = OUTPUT / "figures"
PROTOCOL_PATH = OUTPUT / "V3_PARAMETERIZATION_DESIGN_PROTOCOL.json"
MANIFEST_PATH = OUTPUT / "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
SMOKE_PATH = OUTPUT / "V3_NOMINAL_MYOLEG_SMOKE.csv"

PRIOR_OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_trajectory_parameterization_boundary_audit_v1"
PRIOR_CHECKSUMS = PRIOR_OUTPUT / "checksums.sha256"
PRIOR_METADATA = PRIOR_OUTPUT / "metadata.json"
PRIOR_RECOMMENDATION = PRIOR_OUTPUT / "V3_PARAMETERIZATION_RECOMMENDATION.json"
TRUTH_MANIFEST = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
CANDIDATE_MANIFEST = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
REFERENCE_PATH = ROOT / "external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv"
FORMAL_REFERENCE_PATH = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
FROZEN_GENERATOR = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
FORMAL_MANIFEST = ROOT / "config/formal_experiment_manifest.json"

HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)

FROZEN_SHA = {
    "prior_checksums": "485e3f81659500e287fc9cc165472ab4971c2e4c30200a27c5024029fa9cc111",
    "prior_metadata": "8341cd876f594a9d8d26af22616c219367ed8c15b8cc681e7b71eed480e4929b",
    "prior_recommendation": "504e1dac8219dc0dfe97fc27a36034eecfd05b23f62e20129db448f0f3cbe41b",
    "truth_manifest": "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    "candidate_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "frozen_generator": "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
}

# Filled only after the protocol is frozen.  Tests require identity thereafter.
EXPECTED_PROTOCOL_SHA256 = "debe43f5a5c9e0766f79656646b2409706fdd47e205d1e8924650d477768d1f1"

WIDE_BETA_MIN = -0.25
WIDE_BETA_MAX = 0.25
WIDE_BETA_STEP = 0.0025
EXTREMA_TOLERANCE_DEG = 1.0e-3
Q_CLOSURE_MAX_RAD = 1.0e-10
DQ_CLOSURE_MAX_RAD_S = 1.0e-10
DDQ_CLOSURE_MAX_RAD_S2 = 1.0e-9

# Structural research gates only.  They are neither robot nor human safety
# thresholds and are frozen before the wide sweep.
KINEMATIC_GATES = {
    "minimum_warp_derivative": 0.85,
    "maximum_knee_displacement_fraction_of_reference_rom": 0.05,
    "rms_knee_displacement_fraction_of_reference_rom": 0.015,
    "peak_dq_ratio": 1.10,
    "rms_dq_ratio": 1.10,
    "peak_ddq_ratio": 1.25,
    "rms_ddq_ratio": 1.15,
    "extrema_tolerance_deg": EXTREMA_TOLERANCE_DEG,
    "q_closure_max_rad": Q_CLOSURE_MAX_RAD,
    "dq_closure_max_rad_s": DQ_CLOSURE_MAX_RAD_S,
    "ddq_closure_max_rad_s2": DDQ_CLOSURE_MAX_RAD_S2,
}

GRID_PLANS = (
    ("COARSE", 0.01),
    ("MEDIUM", 0.005),
    ("FINE", 0.0025),
)
GRID_LOCAL_RESOLUTION_MAX_FRACTION_OF_KNEE_ROM = 0.005
GRID_MAX_CANDIDATES = 1000
SMOKE_SELECTION_COUNT = 13


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


def frozen_paths() -> dict[str, Path]:
    return {
        "prior_checksums": PRIOR_CHECKSUMS,
        "prior_metadata": PRIOR_METADATA,
        "prior_recommendation": PRIOR_RECOMMENDATION,
        "truth_manifest": TRUTH_MANIFEST,
        "candidate_manifest": CANDIDATE_MANIFEST,
        "cohort_manifest": COHORT_MANIFEST,
        "v2_reference": REFERENCE_PATH,
        "formal_reference": FORMAL_REFERENCE_PATH,
        "frozen_generator": FROZEN_GENERATOR,
        "formal_manifest": FORMAL_MANIFEST,
    }


def verify_prior_checksums() -> int:
    count = 0
    for line in PRIOR_CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = PRIOR_OUTPUT / relative.strip()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"prior frozen artifact changed: {path}")
        count += 1
    return count


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    actual = {name: sha256_file(path) for name, path in frozen_paths().items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    if verify_prior_checksums() < 20:
        raise RuntimeError("prior checksum manifest unexpectedly incomplete")
    truth = read_json(TRUTH_MANIFEST)
    cohort = read_json(COHORT_MANIFEST)
    recommendation = read_json(PRIOR_RECOMMENDATION)
    if recommendation["current_v2_decision"] != "NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION":
        raise RuntimeError("prior V2 decision changed")
    if recommendation["V3_PRIMARY_PARAMETERIZATION"]["id"] != "P4_BRANCH_AWARE_COORDINATION_FUNCTION":
        raise RuntimeError("prior primary recommendation changed")
    if tuple(cohort["held_out_subject_ids"]) != HELD_OUT_IDS:
        raise RuntimeError("held-out split identity changed")
    if truth["actual_row_count"] != 533600:
        raise RuntimeError("truth manifest identity changed")
    return truth, cohort


def held_out_hash_audit(truth: Mapping[str, Any]) -> dict[str, Any]:
    chunks = [row for row in truth["chunks"] if row["subject_id"] in HELD_OUT_IDS]
    if len(chunks) != 536 or sum(int(row["row_count"]) for row in chunks) != 8 * 16675:
        raise RuntimeError("held-out hash-only manifest coverage changed")
    present = verified = byte_count = 0
    for row in chunks:
        path = ROOT / row["path"]
        if path.is_file():
            present += 1
            byte_count += path.stat().st_size
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"held-out raw-byte checksum mismatch: {path}")
            verified += 1
    return {
        "classification": "SEALED_CONFIRMATORY_TRUTH",
        "held_out_subject_ids": list(HELD_OUT_IDS),
        "held_out_subject_count": 8,
        "manifest_chunk_count": len(chunks),
        "manifest_row_count": sum(int(row["row_count"]) for row in chunks),
        "local_shards_present": present,
        "local_shards_stream_sha256_verified": verified,
        "local_shard_bytes_stream_hashed": byte_count,
        "allowed_operations": ["manifest identity", "file existence", "file size", "stream SHA-256"],
        "np_load_held_out_count": 0,
        "held_out_scientific_truth_access_count": 0,
        "held_out_J_oracle_rank_component_access_count": 0,
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_wide_kinematic_sweep": True,
        "frozen_before_nominal_simulator_smoke": True,
        "prior_frozen_conclusions": [
            "TRAJECTORY_PARAMETERIZATION_ROOT_CAUSE_SUPPORTED",
            "NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION",
        ],
        "frozen_inputs": FROZEN_SHA,
        "parameterization": parameterization.semantics_payload(),
        "reference_invariants": {
            "source": str(REFERENCE_PATH.relative_to(ROOT)),
            "sample_count": 401,
            "duration_s": 24.0,
            "preserve_q_hip_array_exactly": True,
            "preserve_hip_and_knee_min_max_rom_within_deg": EXTREMA_TOLERANCE_DEG,
            "preserve_cycle_endpoints": True,
            "preserve_branch_endpoints": True,
            "preserve_asymmetric_measured_flexion_and_extension": True,
            "closure_and_C2_required": True,
            "pointwise_clipping_forbidden": True,
        },
        "wide_sweep": {
            "beta_min": WIDE_BETA_MIN,
            "beta_max": WIDE_BETA_MAX,
            "step": WIDE_BETA_STEP,
            "axis_only": True,
            "selection_uses_only_kinematics": True,
            "range_rule": "largest origin-connected symmetric interval whose every +/- axis candidate passes all preregistered structural gates",
        },
        "structural_kinematic_gates_not_safety_thresholds": KINEMATIC_GATES,
        "grid_resolution_comparison": {
            "plans": [{"name": name, "step": step} for name, step in GRID_PLANS],
            "maximum_adjacent_trajectory_change_fraction_of_knee_rom": GRID_LOCAL_RESOLUTION_MAX_FRACTION_OF_KNEE_ROM,
            "maximum_total_candidates": GRID_MAX_CANDIDATES,
            "selection_rule": "finest listed grid passing local-resolution and count rules; report all three",
        },
        "candidate_ids": "MYOLEG_V3_K{zero_padded_index_4}; beta_flex outer ascending, beta_extend inner ascending",
        "nominal_smoke": {
            "runs_after_manifest_freeze": True,
            "model": "unmodified nominal MyoLeg V2 model",
            "candidate_count": SMOKE_SELECTION_COUNT,
            "selection": ["reference", "four corners", "four axis extremes", "four deterministic interior geometry points"],
            "selection_uses_outcomes": False,
            "purpose": "simulator integrity only; no ranking, objective, efficacy, safety, or human inference",
            "gates_reused_from_frozen_V2_candidate_domain": {
                "absolute_joint_limit_knee_contribution_nm": 0.005,
                "relative_joint_limit_contribution": 0.0005,
                "source_equality_residual": 0.001,
                "algebraic_residual_nm": 1.0e-8,
                "solver_warnings": 0,
                "contact_active_count": 0,
                "tendon_limit_active_count": 0,
                "joint_limit_active_count_max": 1,
            },
        },
        "structural_fallback_comparison": {
            "primary": "P4_BRANCH_AWARE_COORDINATION_FUNCTION",
            "fallback": "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION",
            "no_objective_or_truth_used": True,
        },
        "scientific_data_access": {
            "development_truth_allowed": False,
            "held_out_truth_allowed": False,
            "mechanical_objective_allowed": False,
            "subject_model_allowed": False,
        },
        "forbidden": {
            "new_24_subject_landscape": True,
            "five_parameter_model": True,
            "NN_or_PINN": True,
            "BO": True,
            "cohort_change": True,
            "objective_change": True,
            "hardware_control_collection_safety": True,
        },
    }


def freeze_protocol() -> None:
    if OUTPUT.exists():
        raise RuntimeError("output already exists; refusing to overwrite protocol freeze")
    truth, _ = verify_frozen_inputs()
    OUTPUT.mkdir(parents=True)
    atomic_json(PROTOCOL_PATH, protocol_payload())
    access = held_out_hash_audit(truth)
    access.update({
        "stage_id": STAGE_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "wide_sweep_completed_at_freeze": False,
        "nominal_smoke_completed_at_freeze": False,
    })
    atomic_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json", access)
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "held_out_scientific_truth_access_count": 0}, indent=2))


def load_reference() -> dict[str, Any]:
    """Reuse the exact frozen V2 adapter without importing simulator code."""
    if sha256_file(FROZEN_GENERATOR) != FROZEN_SHA["frozen_generator"]:
        raise RuntimeError("frozen generator source changed")
    prior = sys.modules.get("mujoco")
    inserted = False
    if prior is None:
        try:
            __import__("mujoco")
        except ModuleNotFoundError:
            sys.modules["mujoco"] = types.ModuleType("mujoco")
            inserted = True
    spec = importlib.util.spec_from_file_location("_frozen_v2_candidate_builder_for_v3", FROZEN_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen reference adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        reference = module.load_reference_adapter()
    finally:
        if inserted:
            sys.modules.pop("mujoco", None)
    return reference


def time_weighted_rms(time_s: np.ndarray, values: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(np.asarray(values, dtype=float) ** 2, time_s) / (time_s[-1] - time_s[0])))


def midpoint_time(time_s: np.ndarray, values: np.ndarray, indices: np.ndarray) -> float:
    branch = np.asarray(values[indices], dtype=float)
    progress = (branch - branch[0]) / (branch[-1] - branch[0])
    order = np.argsort(progress, kind="stable")
    unique, first = np.unique(progress[order], return_index=True)
    return float(np.interp(0.5, unique, time_s[indices][order][first]))


def trajectory_metrics(reference: Mapping[str, Any], candidate: parameterization.V3Trajectory) -> dict[str, Any]:
    time_s = np.asarray(reference["time_s"], dtype=float)
    q_ref = np.asarray(reference["q"], dtype=float)
    dq_ref = np.asarray(reference["dq"], dtype=float)
    ddq_ref = np.asarray(reference["ddq"], dtype=float)
    q, dq, ddq = candidate.q, candidate.dq, candidate.ddq
    q_deg, q_ref_deg = np.degrees(q), np.degrees(q_ref)
    dq_deg, dq_ref_deg = np.degrees(dq), np.degrees(dq_ref)
    ddq_deg, ddq_ref_deg = np.degrees(ddq), np.degrees(ddq_ref)
    jerk_deg = np.gradient(ddq_deg, time_s, axis=0, edge_order=2)
    jerk_ref_deg = np.gradient(ddq_ref_deg, time_s, axis=0, edge_order=2)
    flex = np.flatnonzero(np.asarray(reference["phases"]) == "flexion")
    extend = np.flatnonzero(np.asarray(reference["phases"]) == "extension")
    difference_deg = q_deg - q_ref_deg
    knee_rom = float(np.ptp(q_ref_deg[:, 1]))
    extrema_errors = {
        "hip_min_error_deg": float(abs(np.min(q_deg[:, 0]) - np.min(q_ref_deg[:, 0]))),
        "hip_max_error_deg": float(abs(np.max(q_deg[:, 0]) - np.max(q_ref_deg[:, 0]))),
        "hip_rom_error_deg": float(abs(np.ptp(q_deg[:, 0]) - np.ptp(q_ref_deg[:, 0]))),
        "knee_min_error_deg": float(abs(np.min(q_deg[:, 1]) - np.min(q_ref_deg[:, 1]))),
        "knee_max_error_deg": float(abs(np.max(q_deg[:, 1]) - np.max(q_ref_deg[:, 1]))),
        "knee_rom_error_deg": float(abs(np.ptp(q_deg[:, 1]) - np.ptp(q_ref_deg[:, 1]))),
    }
    path_length = float(np.sum(np.linalg.norm(np.diff(q_deg, axis=0), axis=1)))
    signed_area = float(0.5 * np.sum(q_deg[:-1, 0] * q_deg[1:, 1] - q_deg[1:, 0] * q_deg[:-1, 1]))
    slope: dict[str, float] = {}
    for label, indices in (("flex", flex), ("extend", extend)):
        design = np.column_stack((q_deg[indices, 0], np.ones(len(indices))))
        slope[label] = float(np.linalg.lstsq(design, q_deg[indices, 1], rcond=None)[0][0])
    peak_dq_ref = float(np.max(np.abs(dq_ref_deg[:, 1])))
    rms_dq_ref = time_weighted_rms(time_s, dq_ref_deg[:, 1])
    peak_ddq_ref = float(np.max(np.abs(ddq_ref_deg[:, 1])))
    rms_ddq_ref = time_weighted_rms(time_s, ddq_ref_deg[:, 1])
    metrics: dict[str, Any] = {
        "beta_flex": candidate.beta_flex,
        "beta_extend": candidate.beta_extend,
        "finite": bool(np.isfinite(np.column_stack((q, dq, ddq))).all()),
        "duration_s": float(time_s[-1] - time_s[0]),
        "sample_count": len(time_s),
        "hip_min_deg": float(np.min(q_deg[:, 0])),
        "hip_max_deg": float(np.max(q_deg[:, 0])),
        "hip_rom_deg": float(np.ptp(q_deg[:, 0])),
        "knee_min_deg": float(np.min(q_deg[:, 1])),
        "knee_max_deg": float(np.max(q_deg[:, 1])),
        "knee_rom_deg": float(np.ptp(q_deg[:, 1])),
        **extrema_errors,
        "hip_q_max_abs_error_rad": float(np.max(np.abs(q[:, 0] - q_ref[:, 0]))),
        "hip_dq_max_abs_error_rad_s": float(np.max(np.abs(dq[:, 0] - dq_ref[:, 0]))),
        "hip_ddq_max_abs_error_rad_s2": float(np.max(np.abs(ddq[:, 0] - ddq_ref[:, 0]))),
        "max_knee_displacement_deg": float(np.max(np.abs(difference_deg[:, 1]))),
        "rms_knee_displacement_deg": time_weighted_rms(time_s, difference_deg[:, 1]),
        "max_knee_displacement_fraction_of_rom": float(np.max(np.abs(difference_deg[:, 1])) / knee_rom),
        "rms_knee_displacement_fraction_of_rom": time_weighted_rms(time_s, difference_deg[:, 1]) / knee_rom,
        "joint_space_path_length_deg": path_length,
        "signed_qspace_area_deg2": signed_area,
        "flex_coordination_slope": slope["flex"],
        "extension_coordination_slope": slope["extend"],
        "flex_knee_midpoint_time_s": midpoint_time(time_s, q_deg[:, 1], flex),
        "extension_knee_midpoint_time_s": midpoint_time(time_s, q_deg[:, 1], extend),
        "peak_abs_knee_dq_deg_s": float(np.max(np.abs(dq_deg[:, 1]))),
        "rms_knee_dq_deg_s": time_weighted_rms(time_s, dq_deg[:, 1]),
        "peak_abs_knee_ddq_deg_s2": float(np.max(np.abs(ddq_deg[:, 1]))),
        "rms_knee_ddq_deg_s2": time_weighted_rms(time_s, ddq_deg[:, 1]),
        "peak_abs_knee_jerk_deg_s3": float(np.max(np.abs(jerk_deg[:, 1]))),
        "rms_knee_jerk_deg_s3": time_weighted_rms(time_s, jerk_deg[:, 1]),
        "peak_dq_ratio": float(np.max(np.abs(dq_deg[:, 1])) / peak_dq_ref),
        "rms_dq_ratio": time_weighted_rms(time_s, dq_deg[:, 1]) / rms_dq_ref,
        "peak_ddq_ratio": float(np.max(np.abs(ddq_deg[:, 1])) / peak_ddq_ref),
        "rms_ddq_ratio": time_weighted_rms(time_s, ddq_deg[:, 1]) / rms_ddq_ref,
        "peak_jerk_ratio": float(np.max(np.abs(jerk_deg[:, 1])) / np.max(np.abs(jerk_ref_deg[:, 1]))),
        "minimum_warp_derivative": float(np.min(candidate.warp_first_derivative)),
        "warped_phase_min": float(np.min(candidate.warped_segment_phase)),
        "warped_phase_max": float(np.max(candidate.warped_segment_phase)),
        "q_closure_error_rad": float(np.max(np.abs(q[-1] - q[0]))),
        "dq_closure_error_rad_s": float(np.max(np.abs(dq[-1] - dq[0]))),
        "ddq_closure_error_rad_s2": float(np.max(np.abs(ddq[-1] - ddq[0]))),
        "branch_anchor_q_max_error_rad": 0.0,
        "branch_anchor_dq_max_error_rad_s": 0.0,
        "branch_anchor_ddq_max_error_rad_s2": 0.0,
    }
    anchors = np.isclose(reference["segment_phase"], 0.0, atol=1e-15) | np.isclose(reference["segment_phase"], 1.0, atol=1e-15)
    metrics["branch_anchor_q_max_error_rad"] = float(np.max(np.abs(q[anchors] - q_ref[anchors])))
    metrics["branch_anchor_dq_max_error_rad_s"] = float(np.max(np.abs(dq[anchors] - dq_ref[anchors])))
    metrics["branch_anchor_ddq_max_error_rad_s2"] = float(np.max(np.abs(ddq[anchors] - ddq_ref[anchors])))
    failures: list[str] = []
    if not metrics["finite"]:
        failures.append("nonfinite")
    if metrics["minimum_warp_derivative"] < KINEMATIC_GATES["minimum_warp_derivative"]:
        failures.append("warp_monotonic_reserve")
    if metrics["warped_phase_min"] < -1e-12 or metrics["warped_phase_max"] > 1 + 1e-12:
        failures.append("warped_phase_out_of_branch")
    if max(extrema_errors.values()) > EXTREMA_TOLERANCE_DEG:
        failures.append("reference_extrema_or_rom")
    if metrics["max_knee_displacement_fraction_of_rom"] > KINEMATIC_GATES["maximum_knee_displacement_fraction_of_reference_rom"]:
        failures.append("maximum_knee_displacement")
    if metrics["rms_knee_displacement_fraction_of_rom"] > KINEMATIC_GATES["rms_knee_displacement_fraction_of_reference_rom"]:
        failures.append("rms_knee_displacement")
    for key in ("peak_dq_ratio", "rms_dq_ratio", "peak_ddq_ratio", "rms_ddq_ratio"):
        if metrics[key] > KINEMATIC_GATES[key]:
            failures.append(key)
    if metrics["hip_q_max_abs_error_rad"] != 0.0 or metrics["hip_dq_max_abs_error_rad_s"] != 0.0 or metrics["hip_ddq_max_abs_error_rad_s2"] != 0.0:
        failures.append("hip_not_exact_reference")
    if metrics["q_closure_error_rad"] > Q_CLOSURE_MAX_RAD or metrics["dq_closure_error_rad_s"] > DQ_CLOSURE_MAX_RAD_S or metrics["ddq_closure_error_rad_s2"] > DDQ_CLOSURE_MAX_RAD_S2:
        failures.append("closure")
    if metrics["branch_anchor_q_max_error_rad"] != 0.0 or metrics["branch_anchor_dq_max_error_rad_s"] != 0.0 or metrics["branch_anchor_ddq_max_error_rad_s2"] != 0.0:
        failures.append("branch_endpoint_identity")
    metrics["kinematic_gate_pass"] = not failures
    metrics["exclusion_reason"] = "" if not failures else ";".join(failures)
    return metrics


def axis_sweep(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = np.round(np.arange(WIDE_BETA_MIN, WIDE_BETA_MAX + WIDE_BETA_STEP / 2, WIDE_BETA_STEP), 10)
    rows: list[dict[str, Any]] = []
    for branch in ("flex", "extend"):
        for beta in values:
            candidate = parameterization.generate_v3_trajectory(reference, beta if branch == "flex" else 0.0, beta if branch == "extend" else 0.0)
            rows.append({"swept_branch": branch, "swept_beta": float(beta), **trajectory_metrics(reference, candidate)})
    return rows


def select_symmetric_range(rows: list[dict[str, Any]]) -> float:
    by_branch_beta = {(row["swept_branch"], round(abs(float(row["swept_beta"])), 10), math.copysign(1, float(row["swept_beta"])) if float(row["swept_beta"]) else 0): row for row in rows}
    magnitudes = np.round(np.arange(0.0, WIDE_BETA_MAX + WIDE_BETA_STEP / 2, WIDE_BETA_STEP), 10)
    selected = 0.0
    for magnitude in magnitudes:
        required = []
        for branch in ("flex", "extend"):
            if magnitude == 0:
                required.append(by_branch_beta[(branch, 0.0, 0)])
            else:
                required.extend((by_branch_beta[(branch, float(magnitude), -1.0)], by_branch_beta[(branch, float(magnitude), 1.0)]))
        if not all(bool(row["kinematic_gate_pass"]) for row in required):
            break
        selected = float(magnitude)
    if selected <= 0:
        raise RuntimeError("no nonzero symmetric V3 beta range passed")
    return selected


def grid_axis(beta_max: float, step: float) -> np.ndarray:
    count = int(round(2 * beta_max / step)) + 1
    return np.round(np.linspace(-beta_max, beta_max, count), 10)


def grid_resolution_audit(reference: Mapping[str, Any], beta_max: float) -> list[dict[str, Any]]:
    knee_rom = float(np.ptp(np.degrees(np.asarray(reference["q"])[:, 1])))
    rows = []
    for name, step in GRID_PLANS:
        axis = grid_axis(beta_max, step)
        max_neighbor = 0.0
        for branch in ("flex", "extend"):
            trajectories = [parameterization.generate_v3_trajectory(reference, beta if branch == "flex" else 0.0, beta if branch == "extend" else 0.0).q[:, 1] for beta in axis]
            max_neighbor = max(max_neighbor, max(float(np.max(np.abs(np.degrees(trajectories[index + 1] - trajectories[index])))) for index in range(len(axis) - 1)))
        count = len(axis) ** 2
        resolution_pass = max_neighbor / knee_rom <= GRID_LOCAL_RESOLUTION_MAX_FRACTION_OF_KNEE_ROM
        count_pass = count <= GRID_MAX_CANDIDATES
        rows.append({
            "grid_name": name,
            "beta_step": step,
            "axis_count": len(axis),
            "candidate_count": count,
            "max_adjacent_knee_trajectory_change_deg": max_neighbor,
            "max_adjacent_change_fraction_of_reference_knee_rom": max_neighbor / knee_rom,
            "local_resolution_gate": GRID_LOCAL_RESOLUTION_MAX_FRACTION_OF_KNEE_ROM,
            "resolution_pass": resolution_pass,
            "count_gate": GRID_MAX_CANDIDATES,
            "count_pass": count_pass,
            "grid_pass": resolution_pass and count_pass,
        })
    if not rows[-1]["grid_pass"]:
        raise RuntimeError("fine V3 grid does not pass preregistered resolution rule")
    return rows


def candidate_domain(reference: Mapping[str, Any], beta_max: float, step: float) -> tuple[list[dict[str, Any]], dict[str, parameterization.V3Trajectory]]:
    axis = grid_axis(beta_max, step)
    rows: list[dict[str, Any]] = []
    trajectories: dict[str, parameterization.V3Trajectory] = {}
    index = 0
    for beta_flex in axis:
        for beta_extend in axis:
            candidate_id = f"MYOLEG_V3_K{index:04d}"
            candidate = parameterization.generate_v3_trajectory(reference, float(beta_flex), float(beta_extend))
            metrics = trajectory_metrics(reference, candidate)
            included = bool(metrics["kinematic_gate_pass"])
            rows.append({
                "candidate_index": index,
                "candidate_id": candidate_id,
                "beta_flex": float(beta_flex),
                "beta_extend": float(beta_extend),
                "included": included,
                "inclusion_reason": "all_preregistered_kinematic_gates_pass" if included else "",
                **metrics,
            })
            trajectories[candidate_id] = candidate
            index += 1
    if not all(row["included"] for row in rows):
        raise RuntimeError("frozen local grid unexpectedly contains excluded candidates")
    return rows, trajectories


def smoke_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(round(float(row["beta_flex"]), 10), round(float(row["beta_extend"]), 10)): row for row in rows}
    lo = min(float(row["beta_flex"]) for row in rows)
    hi = max(float(row["beta_flex"]) for row in rows)
    half = round(hi / 2, 10)
    targets = [
        (0.0, 0.0, "REFERENCE"),
        (lo, lo, "CORNER_NEG_NEG"), (lo, hi, "CORNER_NEG_POS"),
        (hi, lo, "CORNER_POS_NEG"), (hi, hi, "CORNER_POS_POS"),
        (lo, 0.0, "FLEX_NEG_AXIS"), (hi, 0.0, "FLEX_POS_AXIS"),
        (0.0, lo, "EXTEND_NEG_AXIS"), (0.0, hi, "EXTEND_POS_AXIS"),
        (-half, -half, "INTERIOR_NEG_NEG"), (-half, half, "INTERIOR_NEG_POS"),
        (half, -half, "INTERIOR_POS_NEG"), (half, half, "INTERIOR_POS_POS"),
    ]
    selected = []
    for rank, (flex, extend, role) in enumerate(targets):
        row = lookup[(round(flex, 10), round(extend, 10))]
        selected.append({"smoke_rank": rank, "selection_role": role, "candidate_id": row["candidate_id"], "candidate_index": row["candidate_index"], "beta_flex": flex, "beta_extend": extend})
    if len(selected) != SMOKE_SELECTION_COUNT:
        raise RuntimeError("smoke selection count changed")
    return selected


def reference_recovery(reference: Mapping[str, Any]) -> dict[str, Any]:
    candidate = parameterization.generate_v3_trajectory(reference, 0.0, 0.0)
    fields = {
        "q": np.asarray(reference["q"]),
        "dq": np.asarray(reference["dq"]),
        "ddq": np.asarray(reference["ddq"]),
    }
    checks = {
        "q_array_equal": bool(np.array_equal(candidate.q, fields["q"])),
        "dq_array_equal": bool(np.array_equal(candidate.dq, fields["dq"])),
        "ddq_array_equal": bool(np.array_equal(candidate.ddq, fields["ddq"])),
        "q_max_abs_error": float(np.max(np.abs(candidate.q - fields["q"]))),
        "dq_max_abs_error": float(np.max(np.abs(candidate.dq - fields["dq"]))),
        "ddq_max_abs_error": float(np.max(np.abs(candidate.ddq - fields["ddq"]))),
    }
    checks["pass"] = all(checks[key] for key in ("q_array_equal", "dq_array_equal", "ddq_array_equal"))
    if not checks["pass"]:
        raise RuntimeError("[0,0] does not exactly recover frozen reference")
    return checks


def structural_comparison() -> list[dict[str, Any]]:
    return [
        {
            "option": "P4_BRANCH_AWARE_COORDINATION_FUNCTION",
            "role": "PRIMARY_IMPLEMENTED",
            "dimension": 2,
            "preserves_hip_exactly": True,
            "preserves_extrema_by_construction_and_audit": True,
            "separate_flexion_extension_control": True,
            "C2_endpoint_identity": True,
            "local_interpretability": "high: advance/delay knee progression per measured branch",
            "parameter_interaction": "disjoint branch interiors",
            "low_budget_future_search": "structurally favorable",
            "limitations": "one scalar warp mode per branch; cannot express multiple interior bends",
            "selection_used_J_or_subject_truth": False,
        },
        {
            "option": "P2_INTERIOR_BSPLINE_JOINT_PERTURBATION",
            "role": "FALLBACK_NOT_IMPLEMENTED",
            "dimension": 4,
            "preserves_hip_exactly": True,
            "preserves_extrema_by_construction_and_audit": "requires constrained basis",
            "separate_flexion_extension_control": True,
            "C2_endpoint_identity": "achievable with constrained basis",
            "local_interpretability": "moderate: control-point displacement",
            "parameter_interaction": "multiple overlapping basis supports",
            "low_budget_future_search": "less favorable than two-dimensional P4",
            "limitations": "more parameters, knot semantics and interaction need additional freeze",
            "selection_used_J_or_subject_truth": False,
        },
    ]


def make_figures(reference: Mapping[str, Any], trajectories: Mapping[str, parameterization.V3Trajectory], rows: list[dict[str, Any]], beta_max: float, sweep: list[dict[str, Any]], grid_audit: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    lookup = {(round(float(row["beta_flex"]), 10), round(float(row["beta_extend"]), 10)): row["candidate_id"] for row in rows}
    q_ref = np.degrees(np.asarray(reference["q"]))
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.plot(q_ref[:, 0], q_ref[:, 1], color="black", linewidth=2.8, label="frozen reference")
    colors = {(-beta_max, -beta_max): "#0072B2", (-beta_max, beta_max): "#56B4E9", (beta_max, -beta_max): "#D55E00", (beta_max, beta_max): "#CC79A7"}
    for pair, color in colors.items():
        q = np.degrees(trajectories[lookup[(round(pair[0], 10), round(pair[1], 10))]].q)
        ax.plot(q[:, 0], q[:, 1], color=color, linewidth=1.8, label=f"βf={pair[0]:+.3f}, βe={pair[1]:+.3f}")
    ax.set_xlabel("Hip flexion (deg)")
    ax.set_ylabel("Knee flexion (deg)")
    ax.set_title("V3 changes interior hip-knee coordination, not task extrema")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_v3_joint_space_paths.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    time_s = np.asarray(reference["time_s"])
    for branch, axis in (("flex", axes[0]), ("extend", axes[1])):
        for beta, color in ((-beta_max, "#0072B2"), (0.0, "black"), (beta_max, "#D55E00")):
            candidate = parameterization.generate_v3_trajectory(reference, beta if branch == "flex" else 0.0, beta if branch == "extend" else 0.0)
            axis.plot(time_s, np.degrees(candidate.q[:, 1]), color=color, linewidth=2 if beta == 0 else 1.6, label=f"β={beta:+.3f}")
        axis.axvline(float(time_s[np.flatnonzero(np.asarray(reference["phases"]) == "flexion")[-1]]), color="0.6", linestyle="--", linewidth=1)
        axis.set_ylabel("Knee flexion (deg)")
        axis.set_title(f"{branch.capitalize()}-branch coefficient acts only on its branch interior")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_branch_specific_time_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for branch, color in (("flex", "#0072B2"), ("extend", "#D55E00")):
        selected = sorted((row for row in sweep if row["swept_branch"] == branch), key=lambda row: row["swept_beta"])
        axes[0].plot([row["swept_beta"] for row in selected], [row["max_knee_displacement_fraction_of_rom"] for row in selected], label=branch, color=color)
        axes[1].plot([row["swept_beta"] for row in selected], [row["minimum_warp_derivative"] for row in selected], label=branch, color=color)
    axes[0].axhline(KINEMATIC_GATES["maximum_knee_displacement_fraction_of_reference_rom"], color="black", linestyle="--", label="frozen gate")
    axes[1].axhline(KINEMATIC_GATES["minimum_warp_derivative"], color="black", linestyle="--", label="frozen gate")
    for axis in axes:
        axis.axvspan(-beta_max, beta_max, color="#009E73", alpha=0.12, label="frozen local range")
        axis.set_xlabel("β")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Max knee displacement / reference ROM")
    axes[1].set_ylabel("Minimum warp derivative")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Outcome-free wide sweep and selected symmetric local range")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_beta_range_audit.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
    names = [row["grid_name"] for row in grid_audit]
    fractions = [100 * float(row["max_adjacent_change_fraction_of_reference_knee_rom"]) for row in grid_audit]
    counts = [int(row["candidate_count"]) for row in grid_audit]
    x = np.arange(len(names))
    ax1.bar(x - 0.17, fractions, 0.34, color="#56B4E9", label="max adjacent path change (% ROM)")
    ax1.axhline(100 * GRID_LOCAL_RESOLUTION_MAX_FRACTION_OF_KNEE_ROM, color="black", linestyle="--", label="resolution gate")
    ax1.set_ylabel("Adjacent change (% reference knee ROM)")
    ax1.set_xticks(x, names)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.17, counts, 0.34, color="#E69F00", label="candidate count")
    ax2.set_ylabel("2-D candidate count")
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels, loc="upper left", fontsize=8)
    ax1.set_title("Coarse / medium / fine grid resolution audit")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_grid_resolution_audit.png", dpi=180)
    plt.close(fig)


def build_domain() -> None:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("protocol must be frozen first")
    if EXPECTED_PROTOCOL_SHA256 == "TO_BE_FROZEN" or sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol SHA constant is not established or protocol changed")
    verify_frozen_inputs()
    if MANIFEST_PATH.exists():
        raise RuntimeError("candidate manifest already exists; refusing overwrite")
    reference = load_reference()
    recovery = reference_recovery(reference)
    atomic_json(OUTPUT / "REFERENCE_RECOVERY_AUDIT.json", recovery)
    atomic_json(OUTPUT / "V3_PARAMETERIZATION_SEMANTICS.json", parameterization.semantics_payload())
    sweep = axis_sweep(reference)
    write_csv(OUTPUT / "V3_BETA_RANGE_AUDIT.csv", sweep)
    beta_max = select_symmetric_range(sweep)
    grid_audit = grid_resolution_audit(reference, beta_max)
    write_csv(OUTPUT / "V3_GRID_RESOLUTION_AUDIT.csv", grid_audit)
    selected_grid = next(row for row in reversed(grid_audit) if row["grid_pass"])
    rows, trajectories = candidate_domain(reference, beta_max, float(selected_grid["beta_step"]))
    write_csv(OUTPUT / "V3_KINEMATIC_CANDIDATE_TABLE.csv", rows)
    selection = smoke_selection(rows)
    write_csv(OUTPUT / "V3_NOMINAL_SMOKE_SELECTION.csv", selection)
    comparison = structural_comparison()
    write_csv(OUTPUT / "V3_P4_VS_P2_STRUCTURAL_COMPARISON.csv", comparison)

    task_fields = [
        "candidate_id", "beta_flex", "beta_extend", "duration_s", "sample_count",
        "hip_min_deg", "hip_max_deg", "hip_rom_deg", "knee_min_deg", "knee_max_deg", "knee_rom_deg",
        "hip_min_error_deg", "hip_max_error_deg", "hip_rom_error_deg", "knee_min_error_deg", "knee_max_error_deg", "knee_rom_error_deg",
        "hip_q_max_abs_error_rad", "hip_dq_max_abs_error_rad_s", "hip_ddq_max_abs_error_rad_s2",
        "max_knee_displacement_deg", "rms_knee_displacement_deg", "joint_space_path_length_deg", "signed_qspace_area_deg2",
        "flex_coordination_slope", "extension_coordination_slope", "flex_knee_midpoint_time_s", "extension_knee_midpoint_time_s",
        "kinematic_gate_pass", "exclusion_reason",
    ]
    write_csv(OUTPUT / "V3_TASK_INVARIANCE_AUDIT.csv", [{key: row[key] for key in task_fields} for row in rows], task_fields)
    continuity_fields = [
        "candidate_id", "beta_flex", "beta_extend", "minimum_warp_derivative", "warped_phase_min", "warped_phase_max",
        "q_closure_error_rad", "dq_closure_error_rad_s", "ddq_closure_error_rad_s2",
        "branch_anchor_q_max_error_rad", "branch_anchor_dq_max_error_rad_s", "branch_anchor_ddq_max_error_rad_s2",
        "peak_abs_knee_dq_deg_s", "rms_knee_dq_deg_s", "peak_abs_knee_ddq_deg_s2", "rms_knee_ddq_deg_s2",
        "peak_abs_knee_jerk_deg_s3", "rms_knee_jerk_deg_s3", "kinematic_gate_pass",
    ]
    write_csv(OUTPUT / "V3_C2_CLOSURE_AUDIT.csv", [{key: row[key] for key in continuity_fields} for row in rows], continuity_fields)
    make_figures(reference, trajectories, rows, beta_max, sweep, grid_audit)

    reference_deg = np.degrees(np.asarray(reference["q"]))
    manifest = {
        "manifest_id": MANIFEST_ID,
        "stage_id": STAGE_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "parameterization_semantics_sha256": sha256_file(OUTPUT / "V3_PARAMETERIZATION_SEMANTICS.json"),
        "parameterization_source_sha256": sha256_file(Path(parameterization.__file__).resolve()),
        "reference_id": "MYOLEG_V2_NATIVE_ROM_REFERENCE_CANDIDATE",
        "reference_sha256": FROZEN_SHA["v2_reference"],
        "formal_parent_reference_sha256": FROZEN_SHA["formal_reference"],
        "parameter_order": ["beta_flex", "beta_extend"],
        "beta_range": [-beta_max, beta_max],
        "beta_step": float(selected_grid["beta_step"]),
        "grid_name": selected_grid["grid_name"],
        "axis_count": int(selected_grid["axis_count"]),
        "candidate_count": len(rows),
        "included_candidate_count": sum(bool(row["included"]) for row in rows),
        "ordered_candidate_ids": [row["candidate_id"] for row in rows],
        "candidate_table_sha256": sha256_file(OUTPUT / "V3_KINEMATIC_CANDIDATE_TABLE.csv"),
        "reference_task": {
            "duration_s": 24.0,
            "sample_count": 401,
            "hip_min_deg": float(np.min(reference_deg[:, 0])),
            "hip_max_deg": float(np.max(reference_deg[:, 0])),
            "hip_rom_deg": float(np.ptp(reference_deg[:, 0])),
            "knee_min_deg": float(np.min(reference_deg[:, 1])),
            "knee_max_deg": float(np.max(reference_deg[:, 1])),
            "knee_rom_deg": float(np.ptp(reference_deg[:, 1])),
        },
        "all_candidates_pass_kinematic_gates": all(bool(row["kinematic_gate_pass"]) for row in rows),
        "identity_reference_recovery_exact": recovery["pass"],
        "smoke_selection": selection,
        "smoke_completed_at_manifest_freeze": False,
        "mechanical_objective_evaluated": False,
        "development_truth_access_count": 0,
        "held_out_scientific_truth_access_count": 0,
        "candidate_domain_ready_for_next_development_stage": True,
        "not_human_ready": True,
        "not_robot_approved": True,
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps({"manifest_sha256": sha256_file(MANIFEST_PATH), "beta_range": [-beta_max, beta_max], "candidate_count": len(rows), "reference_recovery_exact": True}, indent=2))


def summary_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "all_pass": all(row["smoke_integrity_pass"] == "True" for row in rows),
        "max_absolute_joint_limit_knee_contribution_nm": max(float(row["absolute_joint_limit_knee_contribution_nm"]) for row in rows),
        "max_relative_joint_limit_contribution": max(float(row["relative_joint_limit_contribution"]) for row in rows),
        "max_source_equality_residual": max(float(row["source_equality_residual_max"]) for row in rows),
        "max_algebraic_residual_nm": max(float(row["algebraic_residual_max_nm"]) for row in rows),
        "max_solver_warning_count": max(int(row["solver_warning_count"]) for row in rows),
    }


def finalize() -> None:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("protocol changed")
    if not MANIFEST_PATH.is_file() or not SMOKE_PATH.is_file():
        raise RuntimeError("manifest freeze and nominal smoke are required before finalize")
    verify_frozen_inputs()
    manifest = read_json(MANIFEST_PATH)
    smoke_rows = read_csv(SMOKE_PATH)
    smoke = summary_stats(smoke_rows)
    if len(smoke_rows) != SMOKE_SELECTION_COUNT or not smoke["all_pass"]:
        raise RuntimeError("nominal sparse integrity smoke did not pass")
    candidate_rows = read_csv(OUTPUT / "V3_KINEMATIC_CANDIDATE_TABLE.csv")
    sweep_rows = read_csv(OUTPUT / "V3_BETA_RANGE_AUDIT.csv")
    grid_rows = read_csv(OUTPUT / "V3_GRID_RESOLUTION_AUDIT.csv")
    beta_min, beta_max = map(float, manifest["beta_range"])
    variation = {
        key: (min(float(row[key]) for row in candidate_rows), max(float(row[key]) for row in candidate_rows))
        for key in (
            "max_knee_displacement_deg", "rms_knee_displacement_deg",
            "joint_space_path_length_deg", "signed_qspace_area_deg2",
            "flex_knee_midpoint_time_s", "extension_knee_midpoint_time_s",
        )
    }
    first_failed_positive: dict[str, float | None] = {}
    for branch in ("flex", "extend"):
        positive = sorted((row for row in sweep_rows if row["swept_branch"] == branch and float(row["swept_beta"]) > beta_max), key=lambda row: float(row["swept_beta"]))
        first_failed_positive[branch] = next((float(row["swept_beta"]) for row in positive if row["kinematic_gate_pass"] == "False"), None)
    report = f"""# MyoLeg V3 Trajectory Parameterization Design V1

## Formal result

`{OUTCOME}`

This is an **offline, default-off, kinematic design result**. It is not a human result, robot-motion approval, clinical finding, safety validation, or personalized-trajectory outcome. No development or held-out scientific truth, mechanical objective, subject model, Five-parameter model, learner, PINN, NN, BO, hardware, control, collection, or safety code was used.

## Frozen semantics

The two coordinates are `beta_flex` and `beta_extend`. For each measured branch, the normalized phase is transformed by

`w(s; beta) = s + beta * 64 s^3 (1-s)^3`.

Positive beta advances knee progression along that measured branch relative to the frozen hip/reference phase; negative beta delays it. The hip trajectory is copied exactly. The basis and its first two derivatives are zero at both branch endpoints, so the map returns to identity through second derivative. Pointwise clipping is absent.

## Reference recovery and task invariance

At `[0, 0]`, q, dq and ddq are array-exact copies of the frozen V2 reference. The cycle remains 24 s and 401 samples. Across all {manifest['candidate_count']} candidates, the frozen hip/knee extrema and ROM are preserved within {EXTREMA_TOLERANCE_DEG:g} deg; hip q/dq/ddq remain exactly unchanged; branch anchors, cycle closure and C2 endpoint conditions pass. The parameterization changes only the interior hip-knee path and timing relationship.

## Range and grid

The preregistered outcome-free axis sweep covered [{WIDE_BETA_MIN:+.2f}, {WIDE_BETA_MAX:+.2f}] at step {WIDE_BETA_STEP}. The largest origin-connected symmetric interval satisfying all frozen kinematic gates is [{beta_min:+.3f}, {beta_max:+.3f}]. The first failed positive axis value was {first_failed_positive}; the table records every failure reason. These bounds are structural research bounds, not human or robot safety thresholds.

Coarse, medium and fine grids were compared before candidate-domain freeze. The selected `{manifest['grid_name']}` grid has step {manifest['beta_step']}, {manifest['axis_count']} values per axis, and {manifest['candidate_count']} total candidates. Stable IDs follow beta-flex outer ascending and beta-extend inner ascending order.

## Actual interior variation

Because hip is exact, matched-phase joint-space displacement from reference is entirely the knee displacement. Across the domain, maximum displacement spans {variation['max_knee_displacement_deg'][0]:.6f}--{variation['max_knee_displacement_deg'][1]:.6f} deg and time-weighted RMS displacement spans {variation['rms_knee_displacement_deg'][0]:.6f}--{variation['rms_knee_displacement_deg'][1]:.6f} deg. Joint-space path length spans {variation['joint_space_path_length_deg'][0]:.6f}--{variation['joint_space_path_length_deg'][1]:.6f} deg, signed path area spans {variation['signed_qspace_area_deg2'][0]:.6f}--{variation['signed_qspace_area_deg2'][1]:.6f} deg2, flexion knee-midpoint time spans {variation['flex_knee_midpoint_time_s'][0]:.6f}--{variation['flex_knee_midpoint_time_s'][1]:.6f} s, and extension knee-midpoint time spans {variation['extension_knee_midpoint_time_s'][0]:.6f}--{variation['extension_knee_midpoint_time_s'][1]:.6f} s. Thus the domain produces nonzero interior coordination/path variation without amplitude variation.

## Nominal MyoLeg smoke

After manifest freeze, {len(smoke_rows)} geometry-selected candidates (reference, corners, axes and interior points) were replayed once on the unmodified nominal MyoLeg model. All passed the frozen simulator-artifact integrity screen. Maximum joint-limit knee contribution was {smoke['max_absolute_joint_limit_knee_contribution_nm']:.6g} N m, maximum relative contribution {smoke['max_relative_joint_limit_contribution']:.6g}, maximum source equality residual {smoke['max_source_equality_residual']:.6g}, maximum algebraic residual {smoke['max_algebraic_residual_nm']:.6g} N m, and maximum solver warnings {smoke['max_solver_warning_count']}. No trajectory was ranked and no objective was evaluated.

## P4 versus P2

P4 remains the primary structure because two branch-specific coefficients provide direct, reversible semantics, disjoint branch support, exact hip preservation and low dimensionality. P2 remains a plausible fallback if later development evidence shows that one scalar warp mode per branch is too restrictive; it would require a new knot/basis/range freeze and a larger four-dimensional domain. That fallback was not implemented here.

## Limitations and next stage

The result is `VALID_WITH_LIMITATIONS` because simulator coverage is deliberately sparse and nominal-only, and because no development truth landscape has been generated. It establishes a valid candidate space, not personalization benefit or subject specificity. The only recommended next stage is `{NEXT_STAGE}`, using the frozen manifest SHA. That stage was not run here. Held-out scientific access remained zero.
"""
    atomic_text(OUTPUT / "MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_DESIGN_REPORT.md", report)

    access = read_json(OUTPUT / "HELD_OUT_ACCESS_AUDIT.json")
    artifacts = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"})
    metadata = {
        "stage_id": STAGE_ID,
        "outcome": OUTCOME,
        "status": ["OFFLINE_ONLY", "DEFAULT_OFF", "NOT_HUMAN_READY", "NOT_ROBOT_APPROVED"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "candidate_manifest_sha256": sha256_file(MANIFEST_PATH),
        "candidate_count": manifest["candidate_count"],
        "beta_range": manifest["beta_range"],
        "beta_step": manifest["beta_step"],
        "nominal_smoke": smoke,
        "development_truth_access_count": 0,
        "held_out_scientific_truth_access_count": access["held_out_scientific_truth_access_count"],
        "mechanical_objective_evaluated": False,
        "full_V3_landscape_generated": False,
        "recommended_next_stage": NEXT_STAGE,
        "next_stage_executed": False,
        "frozen_inputs_before": FROZEN_SHA,
        "frozen_inputs_after": {name: sha256_file(path) for name, path in frozen_paths().items()},
        "artifact_sha256": {path.relative_to(OUTPUT).as_posix(): sha256_file(path) for path in artifacts},
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    checksum_files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in checksum_files]
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(lines) + "\n")
    print(json.dumps({"outcome": OUTCOME, "candidate_count": manifest["candidate_count"], "smoke_count": len(smoke_rows), "artifact_count": len(lines) + 1}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--build-domain", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze_protocol:
        freeze_protocol()
    elif args.build_domain:
        build_domain()
    else:
        finalize()


if __name__ == "__main__":
    main()
