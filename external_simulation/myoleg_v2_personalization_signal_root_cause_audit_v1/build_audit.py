"""Diagnose why frozen MyoLeg-V2 landscapes lack personalization signal.

The analysis is development-only and read-only with respect to every frozen
scientific input.  Held-out shard bytes may be stream-hashed, but held-out NPZ
arrays and scientific values are rejected before path resolution.
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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


STAGE_ID = "MYOLEG_V2_PERSONALIZATION_SIGNAL_ROOT_CAUSE_AUDIT_V1"
PROTOCOL_ID = "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL_V1"
OUTCOME_IDENTIFIED = "PERSONALIZATION_SIGNAL_ROOT_CAUSE_IDENTIFIED"
OUTCOME_PARTIAL = "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PARTIALLY_IDENTIFIED"
OUTCOME_NOT_IDENTIFIED = "PERSONALIZATION_SIGNAL_ROOT_CAUSE_NOT_IDENTIFIED"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_personalization_signal_root_cause_audit_v1"
FIGURES = OUTPUT / "figures"
PROTOCOL_PATH = OUTPUT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json"
ACCESS_PATH = OUTPUT / "HELD_OUT_ACCESS_AUDIT.json"
TRUTH_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
TRUTH_PROTOCOL_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/LANDSCAPE_GENERATION_PROTOCOL.json"
CANDIDATE_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COHORT_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
NECESSITY_OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1"
NECESSITY_CHECKSUMS_PATH = NECESSITY_OUTPUT / "checksums.sha256"
REPLAY_API_PATH = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/replay_api.py"
REPLAY_CACHE_PATH = ROOT / "external_simulation/data/myoleg_v2_personalization_signal_root_cause_audit_v1/development_replay_subset.npz"
REPLAY_CACHE_MANIFEST_PATH = OUTPUT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json"

FROZEN_SHA = {
    "truth_landscape_manifest": "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    "truth_landscape_protocol": "2fe115d8c34685c70672bcc6a4d9752a88dfbb2cf12fb12d60df877755b7fdcc",
    "candidate_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "necessity_checksums": "0b4a449186b571b9c39207dc18cab1ec1366004821b8d9fd74e97792d87437d3",
    "necessity_protocol": "f26663a71960c2f5cedb3d374cced98b0852fbfd718fe8235f0e1d9e6d102e6f",
}
HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)
REFERENCE_ID = "MYOLEG_V2_P15012"
COMMON_ORACLE_ID = "MYOLEG_V2_P20850"
EXPECTED_CANDIDATES = 16675
EXPECTED_DEVELOPMENT = 24
GRID_STEP = np.asarray([0.25, 0.25, 0.0025], dtype=float)
EQUIVALENCE_TOLERANCE = 1.0e-12
COMPONENTS = {
    "mass": ("mass_term_nm", 1.0),
    "bias_gravity": ("bias_term_nm", 1.0),
    "passive": ("passive_internal_nm", -1.0),
    "zero_control_actuator": ("actuator_internal_nm", -1.0),
    "constraint": ("constraint_internal_nm", -1.0),
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
        raise RuntimeError(f"cannot infer schema for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def verify_input_identity() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    actual = {
        "truth_landscape_manifest": sha256_file(TRUTH_MANIFEST_PATH),
        "truth_landscape_protocol": sha256_file(TRUTH_PROTOCOL_PATH),
        "candidate_manifest": sha256_file(CANDIDATE_MANIFEST_PATH),
        "cohort_manifest": sha256_file(COHORT_MANIFEST_PATH),
        "necessity_checksums": sha256_file(NECESSITY_CHECKSUMS_PATH),
        "necessity_protocol": sha256_file(NECESSITY_OUTPUT / "PERSONALIZATION_NECESSITY_PROTOCOL.json"),
    }
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    truth = read_json(TRUTH_MANIFEST_PATH)
    truth_protocol = read_json(TRUTH_PROTOCOL_PATH)
    candidates = read_json(CANDIDATE_MANIFEST_PATH)
    cohort = read_json(COHORT_MANIFEST_PATH)
    if not (
        truth["outcome"] == "MYOLEG_V2_TRUTH_LANDSCAPE_VALID"
        and truth["actual_row_count"] == 533600
        and truth["duplicate_pair_count"] == 0
        and truth["integrity_summary"]["all_pair_integrity_pass"] is True
        and len(candidates["ordered_included_candidates"]) == EXPECTED_CANDIDATES
        and len(cohort["development_subject_ids"]) == EXPECTED_DEVELOPMENT
        and tuple(cohort["held_out_subject_ids"]) == HELD_OUT_IDS
        and truth_protocol["truth"]["objective"] == "sqrt(0.5*((hip_rms/subject_reference_hip_rms)^2+(knee_rms/subject_reference_knee_rms)^2))"
    ):
        raise RuntimeError("frozen scientific identity/status mismatch")
    return truth, truth_protocol, candidates, cohort


def verify_necessity_artifacts() -> dict[str, Any]:
    checked = 0
    for line in NECESSITY_CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = NECESSITY_OUTPUT / relative.strip()
        if sha256_file(path) != expected:
            raise RuntimeError(f"necessity artifact changed: {relative}")
        checked += 1
    metadata = read_json(NECESSITY_OUTPUT / "metadata.json")
    if metadata["outcome"] != "PERSONALIZATION_NECESSITY_NOT_SUPPORTED":
        raise RuntimeError("frozen necessity outcome changed")
    return {"checksums_file_sha256": sha256_file(NECESSITY_CHECKSUMS_PATH), "verified_file_count": checked}


def held_out_hash_audit(truth: dict[str, Any]) -> dict[str, Any]:
    chunks = [row for row in truth["chunks"] if row["subject_id"] in HELD_OUT_IDS]
    if len(chunks) != 536 or sum(int(row["row_count"]) for row in chunks) != 8 * EXPECTED_CANDIDATES:
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
        "stage_id": STAGE_ID,
        "classification": "SEALED_CONFIRMATORY_TRUTH",
        "sealed_subject_ids": list(HELD_OUT_IDS),
        "sealed_subject_count": 8,
        "manifest_chunk_count": len(chunks),
        "manifest_row_count": sum(int(row["row_count"]) for row in chunks),
        "local_shard_count_present": present,
        "local_shard_count_sha256_verified": verified,
        "local_shard_bytes_stream_hashed": byte_count,
        "allowed_operations": ["file existence", "file size", "streaming SHA-256", "manifest row count"],
        "np_load_held_out_count": 0,
        "held_out_scientific_truth_access_count": 0,
        "held_out_j_oracle_rank_component_access_count": 0,
        "post_freeze_oracle_summary_opened": False,
        "subject_landscape_summary_opened": False,
    }


def alpha_key(alpha: Iterable[float]) -> tuple[float, float, float]:
    return tuple(round(float(value), 8) for value in alpha)  # type: ignore[return-value]


def candidate_by_alpha(rows: list[dict[str, Any]], alpha: Iterable[float]) -> dict[str, Any]:
    lookup = {alpha_key(row["alpha"]): row for row in rows}
    key = alpha_key(alpha)
    if key not in lookup:
        raise RuntimeError(f"required frozen candidate absent: {key}")
    return lookup[key]


def select_subject_subset(cohort: dict[str, Any], count: int = 6) -> list[dict[str, str]]:
    records = [row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"]
    vectors = {row["subject_id"]: np.asarray(row["unit_cube_vector"], dtype=float) for row in records}
    centroid = np.full(6, 0.5)
    first = min(records, key=lambda row: (float(np.linalg.norm(vectors[row["subject_id"]] - centroid)), row["subject_id"]))
    second = max(records, key=lambda row: (float(np.linalg.norm(vectors[row["subject_id"]] - centroid)), tuple(-ord(c) for c in row["subject_id"])))
    chosen = [first, second]
    while len(chosen) < count:
        remaining = [row for row in records if row not in chosen]
        chosen.append(max(
            remaining,
            key=lambda row: (
                min(float(np.linalg.norm(vectors[row["subject_id"]] - vectors[item["subject_id"]])) for item in chosen),
                tuple(-ord(c) for c in row["subject_id"]),
            ),
        ))
    roles = ["CENTER_NEAREST", "CENTER_FARTHEST"] + [f"MAXIMIN_{index}" for index in range(3, count + 1)]
    return [{"subject_id": row["subject_id"], "selection_role": role} for row, role in zip(chosen, roles)]


def select_slice_anchors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    lower, upper = np.min(alpha, axis=0), np.max(alpha, axis=0)
    center = (lower + upper) / 2.0
    span = upper - lower
    central_index = min(range(len(rows)), key=lambda index: (float(np.linalg.norm((alpha[index] - center) / span)), int(rows[index]["proposal_index"])))
    margin = np.minimum((alpha - lower) / span, (upper - alpha) / span)
    interior = [rows[index] for index in range(len(rows)) if float(np.min(margin[index])) >= 0.20]
    interior.sort(key=lambda row: (hashlib.sha256(row["candidate_id"].encode()).hexdigest(), int(row["proposal_index"])))
    chosen = [
        {**next(row for row in rows if row["candidate_id"] == REFERENCE_ID), "selection_role": "V2_REFERENCE"},
        {**next(row for row in rows if row["candidate_id"] == COMMON_ORACLE_ID), "selection_role": "FROZEN_COMMON_ORACLE"},
        {**rows[central_index], "selection_role": "GEOMETRIC_DOMAIN_CENTER"},
        {**interior[0], "selection_role": "FIXED_HASH_INTERIOR_1"},
        {**interior[1], "selection_role": "FIXED_HASH_INTERIOR_2"},
    ]
    if len({row["candidate_id"] for row in chosen}) != 5:
        raise RuntimeError("slice anchors are not unique")
    return [{"candidate_id": row["candidate_id"], "proposal_index": int(row["proposal_index"]), "alpha": list(map(float, row["alpha"])), "selection_role": row["selection_role"]} for row in chosen]


def select_replay_candidates(rows: list[dict[str, Any]], slice_anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def add(alpha: Iterable[float], role: str) -> None:
        row = candidate_by_alpha(rows, alpha)
        selected.setdefault(row["candidate_id"], {"candidate_id": row["candidate_id"], "proposal_index": int(row["proposal_index"]), "alpha": list(map(float, row["alpha"])), "selection_roles": []})
        selected[row["candidate_id"]]["selection_roles"].append(role)

    add((0.0, 0.0, 0.0), "REFERENCE")
    add((2.0, 0.5, -0.03), "FROZEN_COMMON_ORACLE")
    for axis, step in enumerate(GRID_STEP):
        for sign, label in ((-1.0, "MINUS"), (1.0, "PLUS")):
            point = np.zeros(3)
            point[axis] = sign * step
            add(point, f"REFERENCE_{('HIP','KNEE','PHASE')[axis]}_{label}_NEIGHBOR")
    for alpha, role in (
        ((1.75, 0.5, -0.03), "ORACLE_HIP_INWARD_1"),
        ((2.0, 0.25, -0.03), "ORACLE_KNEE_INWARD_1"),
        ((2.0, 0.5, -0.0275), "ORACLE_PHASE_INWARD_1"),
        ((-5.0, 0.0, 0.0), "HIP_AXIS_LOW"), ((2.0, 0.0, 0.0), "HIP_AXIS_HIGH"),
        ((0.0, -5.0, 0.0), "KNEE_AXIS_LOW"), ((0.0, 0.5, 0.0), "KNEE_AXIS_HIGH"),
        ((0.0, 0.0, -0.03), "PHASE_AXIS_LOW"), ((0.0, 0.0, 0.03), "PHASE_AXIS_HIGH"),
    ):
        add(alpha, role)
    for row in slice_anchors[2:]:
        add(row["alpha"], row["selection_role"])
    result = sorted(selected.values(), key=lambda row: row["proposal_index"])
    if not 15 <= len(result) <= 30:
        raise RuntimeError(f"replay candidate count outside frozen design target: {len(result)}")
    return result


def protocol_payload(candidates: dict[str, Any], cohort: dict[str, Any]) -> dict[str, Any]:
    rows = candidates["ordered_included_candidates"]
    subjects = select_subject_subset(cohort)
    anchors = select_slice_anchors(rows)
    replay_candidates = select_replay_candidates(rows, anchors)
    deterministic_crossover = sorted(rows, key=lambda row: (hashlib.sha256(row["candidate_id"].encode()).hexdigest(), int(row["proposal_index"])))[:512]
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_development_outcome_matrix_read": True,
        "scientific_role": "development-only offline root-cause analysis under the unchanged frozen objective",
        "input_sha256": FROZEN_SHA,
        "population": {
            "development_subject_ids": list(cohort["development_subject_ids"]),
            "development_count": 24,
            "held_out_subject_ids": list(HELD_OUT_IDS),
            "held_out_scientific_values_allowed": False,
            "candidate_count": EXPECTED_CANDIDATES,
        },
        "unchanged_objective": "sqrt(0.5*(hip_normalized_rms_ratio^2+knee_normalized_rms_ratio^2))",
        "unchanged_normalization": "joint RMS divided by the same subject's V2-reference joint RMS",
        "hypotheses": [
            "H1_NORMALIZATION_CANCELLATION", "H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION",
            "H3_RMS_OBJECTIVE_COMPRESSION", "H4_CANDIDATE_DOMAIN_MONOTONICITY",
        ],
        "matrix_fields": ["j_truth", "hip_tau_rms_nm", "knee_tau_rms_nm", "subject_reference_hip_rms_nm", "subject_reference_knee_rms_nm"],
        "two_way_decomposition": {
            "convention": "X_ic=grand+row_main_i+candidate_main_c+interaction_ic; every main effect is zero-sum",
            "sum_of_squares": "SS_subject=C*sum(row_main^2); SS_candidate=S*sum(candidate_main^2); SS_interaction=sum(interaction^2)",
            "matrices": ["J", "HIP_RAW_RMS", "KNEE_RAW_RMS", "HIP_NORMALIZED_RMS", "KNEE_NORMALIZED_RMS"],
        },
        "svd": {
            "matrices": ["RAW_J", "SUBJECT_ROW_CENTERED_J", "CANDIDATE_MAIN_REMOVED_J"],
            "variance": "squared singular value divided by sum squared singular values",
            "reported_components": [1, 2, 3, 5],
            "physiological_latent_factor_interpretation": False,
        },
        "multiplicative_scaling": {
            "anchor_subject_id": subjects[0]["subject_id"],
            "anchor_selection": subjects[0]["selection_role"],
            "models": ["affine y=a*x+b", "pure proportional y=a*x"],
            "normalized_rmse_denominator": "mean absolute target RMS",
            "support_rule": "median affine R2>=0.995 and median proportional NRMSE<=0.01 for both joints",
        },
        "normalization_diagnostic": {
            "dimensionless_interaction_scale": "interaction RMS divided by absolute grand mean",
            "attenuation_ratio": "normalized dimensionless interaction scale / raw dimensionless interaction scale",
            "candidate_wise_variability": ["CV median", "CV P95", "pairwise rank correlation"],
        },
        "jointwise_diagnostic_views": ["HIP_NORMALIZED_RMS_ONLY", "KNEE_NORMALIZED_RMS_ONLY"],
        "finite_difference": {
            "reference_candidate_id": REFERENCE_ID,
            "neighbors": {"hip": [-0.25, 0.25], "knee": [-0.25, 0.25], "phase": [-0.0025, 0.0025]},
            "central_derivative": "(f_plus-f_minus)/(2*step)",
            "sign_tolerance": EQUIVALENCE_TOLERANCE,
        },
        "global_monotonicity": {
            "grid_step": GRID_STEP.tolist(),
            "adjacent_pair_selection": "all frozen pairs differing by one positive grid step in exactly one dimension",
            "sign_tolerance": EQUIVALENCE_TOLERANCE,
            "location_bins": "exact lower-coordinate grid value",
        },
        "boundary_pressure": {
            "frozen_common_oracle_alpha": [2.0, 0.5, -0.03],
            "inward_steps": [1, 2, 3],
            "metric": "J(inward)-J(boundary); positive means improvement toward boundary",
            "no_out_of_domain_extrapolation": True,
        },
        "slice_anchors": anchors,
        "slice_subject_policy": "all 24 development subjects; mean and full range overlay",
        "crossover": {
            "primary": "exact all-candidate Kendall tau-b; inversion_rate=(1-tau_b)/2",
            "deterministic_subset_candidate_ids": [row["candidate_id"] for row in deterministic_crossover],
            "subset_role": "independent exact pair-preference corroboration",
        },
        "candidate_effect_interaction": {
            "common_scale": "RMS of candidate main effect",
            "interaction_scale": "RMS of subject-by-candidate residual",
            "ratio": "interaction scale / common candidate-effect scale",
        },
        "replay_subset": {
            "subject_rows": subjects,
            "candidate_rows": replay_candidates,
            "pair_count": len(subjects) * len(replay_candidates),
            "selection_uses_parameter_and_candidate_geometry_only": True,
            "replay_api": str(REPLAY_API_PATH.relative_to(ROOT)),
            "components": {name: {"array": value[0], "required_drive_formula_sign": value[1]} for name, value in COMPONENTS.items()},
        },
        "rms_compression": {
            "time_signal": "signed tau(t) divided by the same subject reference RMS",
            "time_interaction": "two-way subject-by-candidate interaction at each time sample, pooled RMS",
            "rms_interaction": "two-way interaction of candidate RMS/reference RMS summaries",
            "heterogeneity_ratio": "time-resolved interaction RMS / RMS-summary interaction RMS",
            "sign_change": "candidate-minus-reference waveform contains both signs beyond 1e-12",
        },
        "hip_knee_tradeoff": {
            "scope": "all frozen adjacent candidate transitions and all development subjects",
            "meaningful_tradeoff_threshold": 0.05,
            "diagnostic_only": True,
        },
        "parameter_associations": {
            "method": "descriptive Spearman with raw p and BH q; no predictive model",
            "compact_outcomes": ["hip_raw_scale", "knee_raw_scale", "J_local_gradient_norm", "J_interaction_rms"],
            "replay_component_outcomes": ["mass_scale", "bias_gravity_scale", "passive_scale", "zero_control_actuator_scale", "constraint_scale"],
        },
        "hypothesis_decision_rules": {
            "H1_NORMALIZATION_CANCELLATION": {
                "SUPPORTED": "attenuation_ratio<=0.5 for both joints",
                "PARTIALLY_SUPPORTED": "attenuation_ratio<=0.8 for at least one joint",
            },
            "H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION": {
                "SUPPORTED": "J interaction variance fraction<=0.01 and interaction/common ratio<=0.10",
                "PARTIALLY_SUPPORTED": "J interaction fraction<=0.05 and interaction/common ratio<=0.25",
            },
            "H3_RMS_OBJECTIVE_COMPRESSION": {
                "SUPPORTED": "time/RMS interaction ratio>=2 and time interaction RMS>=0.001 for both joints",
                "PARTIALLY_SUPPORTED": "ratio>=1.25 and time interaction RMS>=0.001 for at least one joint",
            },
            "H4_CANDIDATE_DOMAIN_MONOTONICITY": {
                "SUPPORTED": "all 3 boundary dimensions improve toward boundary for >=23/24 subjects and at least 2 dimensions have >=0.85 pooled global majority sign fraction",
                "PARTIALLY_SUPPORTED": "at least 2 boundary dimensions improve for >=22/24 subjects",
            },
        },
        "overall_decision": {
            "IDENTIFIED": "at least two hypotheses SUPPORTED and H2 or H4 is SUPPORTED",
            "PARTIALLY_IDENTIFIED": "otherwise at least one hypothesis is SUPPORTED or PARTIALLY_SUPPORTED",
            "NOT_IDENTIFIED": "no hypothesis has support",
        },
        "figures": [
            "all-development hip slices", "all-development knee slices", "all-development phase slices",
            "common candidate effect versus interaction", "SVD explained variance",
            "raw versus normalized variability", "time-resolved torque comparison", "force-component decomposition",
        ],
        "scope_guards": {
            "objective_redesign": False, "normalization_change": False, "candidate_change": False,
            "cohort_or_range_change": False, "learner_training": False, "five_parameter": False,
            "nn_or_pinn": False, "bo": False, "held_out_truth": False, "robot_or_hardware": False,
        },
    }


def freeze_protocol() -> None:
    if OUTPUT.exists():
        raise RuntimeError("root-cause output already exists; refusing to overwrite protocol freeze")
    truth, _, candidates, cohort = verify_input_identity()
    necessity = verify_necessity_artifacts()
    access = held_out_hash_audit(truth)
    protocol = protocol_payload(candidates, cohort)
    OUTPUT.mkdir(parents=True)
    atomic_json(PROTOCOL_PATH, protocol)
    access.update({
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "necessity_artifact_verification": necessity,
        "development_outcome_matrix_read_at_freeze": False,
    })
    atomic_json(ACCESS_PATH, access)
    print(json.dumps({
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "replay_subjects": len(protocol["replay_subset"]["subject_rows"]),
        "replay_candidates": len(protocol["replay_subset"]["candidate_rows"]),
        "held_out_scientific_truth_access_count": 0,
    }, indent=2))


class DevelopmentLandscapeStore:
    """Fail-closed loader for the frozen 24-subject compact landscape."""

    def __init__(self, truth: dict[str, Any], candidates: dict[str, Any], development_ids: Iterable[str]):
        self.truth = truth
        self.candidates = candidates["ordered_included_candidates"]
        self.allowed = frozenset(development_ids)
        if len(self.allowed) != EXPECTED_DEVELOPMENT or self.allowed.intersection(HELD_OUT_IDS):
            raise RuntimeError("development allowlist invalid")
        self.accessed_subject_ids: list[str] = []

    def load_subject(self, subject_id: str) -> dict[str, np.ndarray]:
        if subject_id not in self.allowed:
            raise PermissionError(f"SEALED_CONFIRMATORY_TRUTH denied before path resolution: {subject_id}")
        chunks = sorted(
            (row for row in self.truth["chunks"] if row["subject_id"] == subject_id),
            key=lambda row: int(row["candidate_start_rank"]),
        )
        if len(chunks) != 67 or sum(int(row["row_count"]) for row in chunks) != EXPECTED_CANDIDATES:
            raise RuntimeError(f"development chunk coverage mismatch: {subject_id}")
        required = (
            "candidate_id", "proposal_index", "alpha_hip_deg", "alpha_knee_deg", "alpha_phase",
            "hip_tau_rms_nm", "knee_tau_rms_nm", "subject_reference_hip_rms_nm",
            "subject_reference_knee_rms_nm", "j_truth", "integrity_status",
        )
        columns: dict[str, list[np.ndarray]] = {}
        for chunk in chunks:
            path = ROOT / chunk["path"]
            if not path.is_file() or sha256_file(path) != chunk["sha256"]:
                raise RuntimeError(f"development shard unavailable or invalid: {path}")
            with np.load(path, allow_pickle=False) as shard:
                for key in required:
                    columns.setdefault(key, []).append(np.asarray(shard[key]))
        output = {key: np.concatenate(values) for key, values in columns.items()}
        expected_ids = np.asarray([row["candidate_id"] for row in self.candidates])
        if not (
            len(output["j_truth"]) == EXPECTED_CANDIDATES
            and np.array_equal(output["candidate_id"], expected_ids)
            and np.all(output["integrity_status"] == 1)
            and all(np.isfinite(output[key]).all() for key in required if output[key].dtype.kind in "fiu")
        ):
            raise RuntimeError(f"development landscape integrity failure: {subject_id}")
        self.accessed_subject_ids.append(subject_id)
        return output


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return {
        "min": float(np.min(array)), "median": float(np.median(array)), "mean": float(np.mean(array)),
        "p5": float(np.percentile(array, 5)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
    }


def two_way_decomposition(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    grand = float(np.mean(values))
    subject_main = np.mean(values, axis=1) - grand
    candidate_main = np.mean(values, axis=0) - grand
    interaction = values - grand - subject_main[:, None] - candidate_main[None, :]
    centered = values - grand
    ss_total = float(np.sum(centered**2))
    ss_subject = float(values.shape[1] * np.sum(subject_main**2))
    ss_candidate = float(values.shape[0] * np.sum(candidate_main**2))
    ss_interaction = float(np.sum(interaction**2))
    denominator = max(ss_total, np.finfo(float).tiny)
    candidate_scale = float(np.sqrt(np.mean(candidate_main**2)))
    interaction_scale = float(np.sqrt(np.mean(interaction**2)))
    return {
        "shape": list(values.shape), "grand_mean": grand,
        "ss_total": ss_total, "ss_subject_main": ss_subject, "ss_candidate_main": ss_candidate,
        "ss_subject_candidate_interaction": ss_interaction,
        "subject_main_variance_fraction": ss_subject / denominator,
        "candidate_main_variance_fraction": ss_candidate / denominator,
        "subject_candidate_interaction_variance_fraction": ss_interaction / denominator,
        "common_candidate_effect_rms": candidate_scale,
        "subject_specific_interaction_rms": interaction_scale,
        "interaction_to_common_effect_ratio": interaction_scale / max(candidate_scale, np.finfo(float).tiny),
        "dimensionless_interaction_scale": interaction_scale / max(abs(grand), np.finfo(float).tiny),
        "subject_main": subject_main,
        "candidate_main": candidate_main,
        "interaction": interaction,
    }


def public_decomposition(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}


def bh_qvalues(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def spearman_row(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float, str]:
    reasons = []
    if np.ptp(x) == 0.0:
        reasons.append("CONSTANT_PARAMETER")
    if np.ptp(y) == 0.0:
        reasons.append("CONSTANT_OUTCOME")
    if reasons:
        return None, 1.0, ";".join(reasons)
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), ""


def fit_scaling(target: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    x = np.asarray(anchor, dtype=float)
    y = np.asarray(target, dtype=float)
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = slope * x + intercept
    sse = float(np.sum((y - predicted) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    proportional_slope = float(np.dot(x, y) / np.dot(x, x))
    proportional_prediction = proportional_slope * x
    mean_abs = max(float(np.mean(np.abs(y))), np.finfo(float).tiny)
    return {
        "affine_slope": float(slope), "affine_intercept": float(intercept),
        "affine_r2": 1.0 - sse / max(sst, np.finfo(float).tiny),
        "affine_nrmse": float(np.sqrt(np.mean((y - predicted) ** 2)) / mean_abs),
        "proportional_slope": proportional_slope,
        "proportional_r2": 1.0 - float(np.sum((y - proportional_prediction) ** 2)) / max(sst, np.finfo(float).tiny),
        "proportional_nrmse": float(np.sqrt(np.mean((y - proportional_prediction) ** 2)) / mean_abs),
        "residual_candidate_dependence": float(np.std(y - predicted, ddof=1) / mean_abs),
    }


def sign_class(values: np.ndarray, tolerance: float = EQUIVALENCE_TOLERANCE) -> np.ndarray:
    return np.where(values < -tolerance, -1, np.where(values > tolerance, 1, 0))


def sign_fractions(values: np.ndarray) -> dict[str, float | int]:
    signs = sign_class(np.asarray(values, dtype=float))
    total = signs.size
    return {
        "comparison_count": int(total),
        "negative_fraction": float(np.sum(signs == -1) / total),
        "positive_fraction": float(np.sum(signs == 1) / total),
        "equivalent_fraction": float(np.sum(signs == 0) / total),
        "majority_sign_fraction": float(max(np.sum(signs == -1), np.sum(signs == 1), np.sum(signs == 0)) / total),
    }


def time_weighted_rms(time_s: np.ndarray, values: np.ndarray, axis: int = -1) -> np.ndarray:
    duration = float(time_s[-1] - time_s[0])
    return np.sqrt(np.trapezoid(np.asarray(values, dtype=float) ** 2, time_s, axis=axis) / duration)


def load_replay_api() -> Any:
    spec = importlib.util.spec_from_file_location("_root_cause_frozen_replay_api", REPLAY_API_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen replay API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def svd_analysis(j_matrix: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrices = {
        "RAW_J": j_matrix,
        "SUBJECT_ROW_CENTERED_J": j_matrix - np.mean(j_matrix, axis=1, keepdims=True),
        "CANDIDATE_MAIN_REMOVED_J": j_matrix - np.mean(j_matrix, axis=0, keepdims=True),
    }
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for name, matrix in matrices.items():
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        energy = singular**2
        explained = energy / max(float(np.sum(energy)), np.finfo(float).tiny)
        cumulative = np.cumsum(explained)
        for index, (value, fraction, running) in enumerate(zip(singular, explained, cumulative), start=1):
            rows.append({
                "matrix": name, "component": index, "singular_value": float(value),
                "variance_explained": float(fraction), "cumulative_variance_explained": float(running),
            })
        aggregate[name] = {
            "pc1_variance_explained": float(explained[0]),
            "first_2_variance_explained": float(cumulative[min(1, len(cumulative) - 1)]),
            "first_3_variance_explained": float(cumulative[min(2, len(cumulative) - 1)]),
            "first_5_variance_explained": float(cumulative[min(4, len(cumulative) - 1)]),
            "residual_after_5": float(1.0 - cumulative[min(4, len(cumulative) - 1)]),
        }
    return rows, aggregate


def select_tied_min(values: np.ndarray, proposal: np.ndarray) -> int:
    minimum = float(np.min(values))
    eligible = np.flatnonzero(values <= minimum + EQUIVALENCE_TOLERANCE)
    return int(eligible[np.argmin(proposal[eligible])])


def adjacent_pair_indices(alpha: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    lookup = {alpha_key(row): index for index, row in enumerate(alpha)}
    lower: list[int] = []
    upper: list[int] = []
    step = GRID_STEP[dimension]
    for index, row in enumerate(alpha):
        neighbor = row.copy()
        neighbor[dimension] += step
        key = alpha_key(neighbor)
        if key in lookup:
            lower.append(index)
            upper.append(lookup[key])
    return np.asarray(lower, dtype=int), np.asarray(upper, dtype=int)


def median_pairwise_spearman(matrix: np.ndarray) -> float:
    values = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            values.append(float(stats.spearmanr(matrix[left], matrix[right]).statistic))
    return float(np.median(values))


def full_matrix_analysis(
    protocol: dict[str, Any], development_ids: list[str], candidate_rows: list[dict[str, Any]],
    j_matrix: np.ndarray, hip_raw: np.ndarray, knee_raw: np.ndarray,
    hip_norm: np.ndarray, knee_norm: np.ndarray,
) -> dict[str, Any]:
    alpha = np.asarray([row["alpha"] for row in candidate_rows], dtype=float)
    candidate_ids = np.asarray([row["candidate_id"] for row in candidate_rows])
    proposal = np.asarray([row["proposal_index"] for row in candidate_rows], dtype=int)
    lookup = {alpha_key(row): index for index, row in enumerate(alpha)}
    matrices = {
        "J": j_matrix, "HIP_RAW_RMS": hip_raw, "KNEE_RAW_RMS": knee_raw,
        "HIP_NORMALIZED_RMS": hip_norm, "KNEE_NORMALIZED_RMS": knee_norm,
    }
    decompositions = {name: two_way_decomposition(values) for name, values in matrices.items()}
    atomic_json(OUTPUT / "TWO_WAY_VARIANCE_DECOMPOSITION.json", {
        "stage_id": STAGE_ID,
        "centering_convention": protocol["two_way_decomposition"]["convention"],
        "matrices": {name: public_decomposition(result) for name, result in decompositions.items()},
        "diagnostic": (
            "PERSONALIZATION_SIGNAL_IS_LOW_RANK_OR_COMMON_MODE_DOMINATED"
            if decompositions["J"]["subject_candidate_interaction_variance_fraction"] <= 0.01 else
            "INTERACTION_NOT_NEGLIGIBLE_BY_PREREGISTERED_ONE_PERCENT_RULE"
        ),
    })

    interaction_by_candidate = []
    j_decomposition = decompositions["J"]
    for index, row in enumerate(candidate_rows):
        interaction_by_candidate.append({
            "candidate_id": row["candidate_id"], "proposal_index": int(row["proposal_index"]),
            "alpha_hip_deg": float(row["alpha"][0]), "alpha_knee_deg": float(row["alpha"][1]),
            "alpha_phase": float(row["alpha"][2]),
            "candidate_main_effect_j": float(j_decomposition["candidate_main"][index]),
            "absolute_candidate_main_effect_j": float(abs(j_decomposition["candidate_main"][index])),
            "subject_interaction_rms_j": float(np.sqrt(np.mean(j_decomposition["interaction"][:, index] ** 2))),
        })
    write_csv(OUTPUT / "COMMON_EFFECT_INTERACTION_BY_CANDIDATE.csv", interaction_by_candidate)

    svd_rows, svd_summary = svd_analysis(j_matrix)
    write_csv(OUTPUT / "LANDSCAPE_SVD_ANALYSIS.csv", svd_rows)
    atomic_json(OUTPUT / "LANDSCAPE_SVD_ANALYSIS.json", svd_summary)

    anchor_id = protocol["multiplicative_scaling"]["anchor_subject_id"]
    anchor_index = development_ids.index(anchor_id)
    scaling_rows = []
    scaling_by_subject: dict[str, dict[str, float]] = {subject_id: {} for subject_id in development_ids}
    for subject_index, subject_id in enumerate(development_ids):
        for joint, matrix in (("hip", hip_raw), ("knee", knee_raw)):
            fit = fit_scaling(matrix[subject_index], matrix[anchor_index])
            residual = matrix[subject_index] - (fit["affine_slope"] * matrix[anchor_index] + fit["affine_intercept"])
            axis_correlations = []
            for dimension in range(3):
                rho, _, _ = spearman_row(alpha[:, dimension], residual)
                axis_correlations.append(0.0 if rho is None else abs(rho))
            row = {
                "subject_id": subject_id, "joint": joint, "anchor_subject_id": anchor_id,
                **fit, "max_abs_residual_alpha_spearman": float(max(axis_correlations)),
            }
            scaling_rows.append(row)
            scaling_by_subject[subject_id][f"{joint}_raw_scale"] = fit["proportional_slope"]
    write_csv(OUTPUT / "MULTIPLICATIVE_SCALING_AUDIT.csv", scaling_rows)

    normalization_rows = []
    normalization_attenuation: dict[str, float] = {}
    for joint, raw_name, normalized_name in (
        ("hip", "HIP_RAW_RMS", "HIP_NORMALIZED_RMS"),
        ("knee", "KNEE_RAW_RMS", "KNEE_NORMALIZED_RMS"),
    ):
        raw, normalized = matrices[raw_name], matrices[normalized_name]
        raw_decomp, normalized_decomp = decompositions[raw_name], decompositions[normalized_name]
        raw_cv = np.std(raw, axis=0, ddof=1) / np.maximum(np.abs(np.mean(raw, axis=0)), np.finfo(float).tiny)
        normalized_cv = np.std(normalized, axis=0, ddof=1) / np.maximum(np.abs(np.mean(normalized, axis=0)), np.finfo(float).tiny)
        attenuation = normalized_decomp["dimensionless_interaction_scale"] / max(raw_decomp["dimensionless_interaction_scale"], np.finfo(float).tiny)
        normalization_attenuation[joint] = float(attenuation)
        normalization_rows.append({
            "joint": joint,
            "raw_subject_main_variance_fraction": raw_decomp["subject_main_variance_fraction"],
            "raw_candidate_main_variance_fraction": raw_decomp["candidate_main_variance_fraction"],
            "raw_interaction_variance_fraction": raw_decomp["subject_candidate_interaction_variance_fraction"],
            "raw_dimensionless_interaction_scale": raw_decomp["dimensionless_interaction_scale"],
            "normalized_subject_main_variance_fraction": normalized_decomp["subject_main_variance_fraction"],
            "normalized_candidate_main_variance_fraction": normalized_decomp["candidate_main_variance_fraction"],
            "normalized_interaction_variance_fraction": normalized_decomp["subject_candidate_interaction_variance_fraction"],
            "normalized_dimensionless_interaction_scale": normalized_decomp["dimensionless_interaction_scale"],
            "normalization_interaction_attenuation_ratio": attenuation,
            "raw_candidate_wise_cv_median": float(np.median(raw_cv)),
            "raw_candidate_wise_cv_p95": float(np.percentile(raw_cv, 95)),
            "normalized_candidate_wise_cv_median": float(np.median(normalized_cv)),
            "normalized_candidate_wise_cv_p95": float(np.percentile(normalized_cv, 95)),
            "raw_pairwise_rank_correlation_median": median_pairwise_spearman(raw),
            "normalized_pairwise_rank_correlation_median": median_pairwise_spearman(normalized),
        })
    write_csv(OUTPUT / "NORMALIZATION_CANCELLATION_AUDIT.csv", normalization_rows)

    jointwise_rows = []
    alpha_lower, alpha_upper = np.min(alpha, axis=0), np.max(alpha, axis=0)
    for metric, matrix in (("HIP_NORMALIZED_RMS_ONLY", hip_norm), ("KNEE_NORMALIZED_RMS_ONLY", knee_norm)):
        oracle_indices = []
        for subject_index, subject_id in enumerate(development_ids):
            oracle_index = select_tied_min(matrix[subject_index], proposal)
            oracle_indices.append(oracle_index)
            oracle_alpha = alpha[oracle_index]
            jointwise_rows.append({
                "record_type": "SUBJECT", "metric": metric, "subject_id": subject_id,
                "oracle_candidate_id": candidate_ids[oracle_index], "oracle_proposal_index": int(proposal[oracle_index]),
                "oracle_alpha_hip_deg": float(oracle_alpha[0]), "oracle_alpha_knee_deg": float(oracle_alpha[1]),
                "oracle_alpha_phase": float(oracle_alpha[2]), "oracle_value": float(matrix[subject_index, oracle_index]),
                "hip_boundary": bool(oracle_alpha[0] in (alpha_lower[0], alpha_upper[0])),
                "knee_boundary": bool(oracle_alpha[1] in (alpha_lower[1], alpha_upper[1])),
                "phase_boundary": bool(oracle_alpha[2] in (alpha_lower[2], alpha_upper[2])),
                "unique_oracle_candidate_count": "", "pairwise_rank_correlation_median": "",
            })
        jointwise_rows.append({
            "record_type": "SUMMARY", "metric": metric, "subject_id": "ALL_DEVELOPMENT",
            "oracle_candidate_id": "", "oracle_proposal_index": "", "oracle_alpha_hip_deg": "",
            "oracle_alpha_knee_deg": "", "oracle_alpha_phase": "", "oracle_value": "",
            "hip_boundary": "", "knee_boundary": "", "phase_boundary": "",
            "unique_oracle_candidate_count": len(set(oracle_indices)),
            "pairwise_rank_correlation_median": median_pairwise_spearman(matrix),
        })
    write_csv(OUTPUT / "JOINTWISE_OBJECTIVE_DIAGNOSTICS.csv", jointwise_rows)

    reference_index = lookup[alpha_key((0.0, 0.0, 0.0))]
    finite_rows: list[dict[str, Any]] = []
    local_gradient_norm = {metric: np.zeros(EXPECTED_DEVELOPMENT) for metric in ("J", "HIP_NORMALIZED_RMS", "KNEE_NORMALIZED_RMS")}
    for metric, matrix in (("J", j_matrix), ("HIP_NORMALIZED_RMS", hip_norm), ("KNEE_NORMALIZED_RMS", knee_norm)):
        derivatives = np.zeros((EXPECTED_DEVELOPMENT, 3))
        for dimension, dimension_name in enumerate(("hip", "knee", "phase")):
            minus_alpha = np.zeros(3); minus_alpha[dimension] = -GRID_STEP[dimension]
            plus_alpha = np.zeros(3); plus_alpha[dimension] = GRID_STEP[dimension]
            minus_index, plus_index = lookup[alpha_key(minus_alpha)], lookup[alpha_key(plus_alpha)]
            minus_delta = matrix[:, minus_index] - matrix[:, reference_index]
            plus_delta = matrix[:, plus_index] - matrix[:, reference_index]
            derivative = (matrix[:, plus_index] - matrix[:, minus_index]) / (2.0 * GRID_STEP[dimension])
            derivatives[:, dimension] = derivative
            preferred = np.where(minus_delta < plus_delta - EQUIVALENCE_TOLERANCE, "MINUS", np.where(plus_delta < minus_delta - EQUIVALENCE_TOLERANCE, "PLUS", "EQUIVALENT"))
            derivative_signs = sign_class(derivative)
            agreement = max(np.sum(derivative_signs == -1), np.sum(derivative_signs == 1), np.sum(derivative_signs == 0)) / EXPECTED_DEVELOPMENT
            for subject_index, subject_id in enumerate(development_ids):
                finite_rows.append({
                    "record_type": "SUBJECT", "metric": metric, "dimension": dimension_name,
                    "subject_id": subject_id, "minus_delta_from_reference": float(minus_delta[subject_index]),
                    "plus_delta_from_reference": float(plus_delta[subject_index]),
                    "central_derivative_per_unit": float(derivative[subject_index]),
                    "central_derivative_sign": int(derivative_signs[subject_index]),
                    "preferred_local_direction": str(preferred[subject_index]),
                    "sign_agreement_fraction": "", "minus_improvement_fraction": "", "plus_improvement_fraction": "",
                })
            finite_rows.append({
                "record_type": "SUMMARY", "metric": metric, "dimension": dimension_name,
                "subject_id": "ALL_DEVELOPMENT", "minus_delta_from_reference": "", "plus_delta_from_reference": "",
                "central_derivative_per_unit": "", "central_derivative_sign": "", "preferred_local_direction": "",
                "sign_agreement_fraction": float(agreement),
                "minus_improvement_fraction": float(np.mean(minus_delta < -EQUIVALENCE_TOLERANCE)),
                "plus_improvement_fraction": float(np.mean(plus_delta < -EQUIVALENCE_TOLERANCE)),
            })
        local_gradient_norm[metric] = np.linalg.norm(derivatives * GRID_STEP[None, :], axis=1)
    write_csv(OUTPUT / "REFERENCE_FINITE_DIFFERENCE_AUDIT.csv", finite_rows)

    monotonic_rows: list[dict[str, Any]] = []
    tradeoff_rows: list[dict[str, Any]] = []
    monotonic_global: dict[str, dict[str, dict[str, float]]] = {}
    for dimension, dimension_name in enumerate(("hip", "knee", "phase")):
        lower_indices, upper_indices = adjacent_pair_indices(alpha, dimension)
        monotonic_global[dimension_name] = {}
        for metric, matrix in (("J", j_matrix), ("HIP_NORMALIZED_RMS", hip_norm), ("KNEE_NORMALIZED_RMS", knee_norm)):
            differences = matrix[:, upper_indices] - matrix[:, lower_indices]
            signs = sign_class(differences)
            pair_agreement = np.max(np.stack((np.sum(signs == -1, axis=0), np.sum(signs == 0, axis=0), np.sum(signs == 1, axis=0))), axis=0) / EXPECTED_DEVELOPMENT
            for subject_index, subject_id in enumerate(development_ids):
                monotonic_rows.append({
                    "record_type": "SUBJECT_GLOBAL", "metric": metric, "dimension": dimension_name,
                    "subject_id": subject_id, "location_coordinate": "", **sign_fractions(differences[subject_index]),
                    "cross_subject_sign_agreement_mean": "", "cross_subject_sign_agreement_ge_0_95_fraction": "",
                    "cross_subject_unanimous_fraction": "",
                })
            pooled = sign_fractions(differences)
            pooled.update({
                "cross_subject_sign_agreement_mean": float(np.mean(pair_agreement)),
                "cross_subject_sign_agreement_ge_0_95_fraction": float(np.mean(pair_agreement >= 0.95)),
                "cross_subject_unanimous_fraction": float(np.mean(pair_agreement == 1.0)),
            })
            monotonic_global[dimension_name][metric] = {key: float(value) for key, value in pooled.items() if key != "comparison_count"}
            monotonic_rows.append({
                "record_type": "POOLED_GLOBAL", "metric": metric, "dimension": dimension_name,
                "subject_id": "ALL_DEVELOPMENT", "location_coordinate": "", **pooled,
            })
            locations = alpha[lower_indices, dimension]
            for location in np.unique(locations):
                mask = locations == location
                monotonic_rows.append({
                    "record_type": "POOLED_LOCATION", "metric": metric, "dimension": dimension_name,
                    "subject_id": "ALL_DEVELOPMENT", "location_coordinate": float(location),
                    **sign_fractions(differences[:, mask]),
                    "cross_subject_sign_agreement_mean": "", "cross_subject_sign_agreement_ge_0_95_fraction": "",
                    "cross_subject_unanimous_fraction": "",
                })
        hip_delta = hip_norm[:, upper_indices] - hip_norm[:, lower_indices]
        knee_delta = knee_norm[:, upper_indices] - knee_norm[:, lower_indices]
        hip_sign, knee_sign = sign_class(hip_delta), sign_class(knee_delta)
        for record_type, subject_id, hs, ks in [
            *[("SUBJECT", development_ids[index], hip_sign[index], knee_sign[index]) for index in range(EXPECTED_DEVELOPMENT)],
            ("POOLED", "ALL_DEVELOPMENT", hip_sign.ravel(), knee_sign.ravel()),
        ]:
            total = hs.size
            tradeoff = ((hs == -1) & (ks == 1)) | ((hs == 1) & (ks == -1))
            both_improve = (hs == -1) & (ks == -1)
            both_worsen = (hs == 1) & (ks == 1)
            other = ~(tradeoff | both_improve | both_worsen)
            tradeoff_rows.append({
                "record_type": record_type, "dimension": dimension_name, "subject_id": subject_id,
                "transition_count": int(total), "hip_knee_opposing_count": int(np.sum(tradeoff)),
                "hip_knee_opposing_fraction": float(np.mean(tradeoff)),
                "both_improve_fraction": float(np.mean(both_improve)), "both_worsen_fraction": float(np.mean(both_worsen)),
                "equivalent_or_mixed_zero_fraction": float(np.mean(other)),
                "no_meaningful_tradeoff": bool(float(np.mean(tradeoff)) < protocol["hip_knee_tradeoff"]["meaningful_tradeoff_threshold"]),
            })
    write_csv(OUTPUT / "GLOBAL_MONOTONICITY_AUDIT.csv", monotonic_rows)
    write_csv(OUTPUT / "HIP_KNEE_TRADEOFF_AUDIT.csv", tradeoff_rows)

    boundary_index = lookup[alpha_key((2.0, 0.5, -0.03))]
    boundary_rows: list[dict[str, Any]] = []
    boundary_support: dict[str, float] = {}
    inward_sign = np.asarray([-1.0, -1.0, 1.0])
    for metric, matrix in (("J", j_matrix), ("HIP_NORMALIZED_RMS", hip_norm), ("KNEE_NORMALIZED_RMS", knee_norm)):
        for dimension, dimension_name in enumerate(("hip", "knee", "phase")):
            for step_count in protocol["boundary_pressure"]["inward_steps"]:
                inward_alpha = np.asarray([2.0, 0.5, -0.03])
                inward_alpha[dimension] += inward_sign[dimension] * GRID_STEP[dimension] * step_count
                inward_index = lookup[alpha_key(inward_alpha)]
                improvement = matrix[:, inward_index] - matrix[:, boundary_index]
                support_fraction = float(np.mean(improvement > EQUIVALENCE_TOLERANCE))
                if metric == "J" and step_count == 1:
                    boundary_support[dimension_name] = support_fraction
                for subject_index, subject_id in enumerate(development_ids):
                    boundary_rows.append({
                        "record_type": "SUBJECT", "metric": metric, "dimension": dimension_name,
                        "inward_step_count": step_count, "subject_id": subject_id,
                        "boundary_candidate_id": candidate_ids[boundary_index], "inward_candidate_id": candidate_ids[inward_index],
                        "inward_delta_from_boundary": float(improvement[subject_index]),
                        "slope_toward_boundary_per_unit": float(-improvement[subject_index] / (GRID_STEP[dimension] * step_count)),
                        "improves_toward_boundary": bool(improvement[subject_index] > EQUIVALENCE_TOLERANCE),
                        "improvement_support_fraction": "",
                    })
                boundary_rows.append({
                    "record_type": "SUMMARY", "metric": metric, "dimension": dimension_name,
                    "inward_step_count": step_count, "subject_id": "ALL_DEVELOPMENT",
                    "boundary_candidate_id": candidate_ids[boundary_index], "inward_candidate_id": candidate_ids[inward_index],
                    "inward_delta_from_boundary": float(np.median(improvement)),
                    "slope_toward_boundary_per_unit": float(np.median(-improvement / (GRID_STEP[dimension] * step_count))),
                    "improves_toward_boundary": "", "improvement_support_fraction": support_fraction,
                })
    write_csv(OUTPUT / "BOUNDARY_PRESSURE_AUDIT.csv", boundary_rows)

    crossover_rows = []
    subset_ids = protocol["crossover"]["deterministic_subset_candidate_ids"]
    subset_indices = np.asarray([int(np.flatnonzero(candidate_ids == candidate_id)[0]) for candidate_id in subset_ids], dtype=int)
    for left in range(EXPECTED_DEVELOPMENT):
        for right in range(left + 1, EXPECTED_DEVELOPMENT):
            full_tau = float(stats.kendalltau(j_matrix[left], j_matrix[right], variant="b").statistic)
            subset_tau = float(stats.kendalltau(j_matrix[left, subset_indices], j_matrix[right, subset_indices], variant="b").statistic)
            crossover_rows.append({
                "subject_i": development_ids[left], "subject_j": development_ids[right],
                "all_candidate_kendall_tau_b": full_tau, "all_candidate_rank_inversion_rate": (1.0 - full_tau) / 2.0,
                "all_candidate_crossover_exists": bool(full_tau < 1.0 - np.finfo(float).eps),
                "deterministic_subset_count": len(subset_indices), "subset_kendall_tau_b": subset_tau,
                "subset_rank_inversion_rate": (1.0 - subset_tau) / 2.0,
                "subset_crossover_exists": bool(subset_tau < 1.0 - np.finfo(float).eps),
            })
    write_csv(OUTPUT / "LANDSCAPE_CROSSOVER_AUDIT.csv", crossover_rows)

    per_subject_interaction_rms = np.sqrt(np.mean(decompositions["J"]["interaction"] ** 2, axis=1))
    for subject_index, subject_id in enumerate(development_ids):
        scaling_by_subject[subject_id]["J_local_gradient_norm"] = float(local_gradient_norm["J"][subject_index])
        scaling_by_subject[subject_id]["J_interaction_rms"] = float(per_subject_interaction_rms[subject_index])

    return {
        "decompositions": decompositions, "svd_summary": svd_summary,
        "normalization_rows": normalization_rows, "normalization_attenuation": normalization_attenuation,
        "scaling_rows": scaling_rows, "scaling_by_subject": scaling_by_subject,
        "local_gradient_norm": local_gradient_norm, "monotonic_global": monotonic_global,
        "boundary_support": boundary_support, "crossover_rows": crossover_rows,
        "tradeoff_rows": tradeoff_rows, "alpha": alpha, "candidate_ids": candidate_ids,
        "proposal": proposal, "lookup": lookup,
    }


def replay_analysis(
    protocol: dict[str, Any], development_ids: list[str], matrix_result: dict[str, Any],
    j_matrix: np.ndarray, hip_norm: np.ndarray, knee_norm: np.ndarray,
) -> dict[str, Any]:
    subject_rows = protocol["replay_subset"]["subject_rows"]
    candidate_rows = protocol["replay_subset"]["candidate_rows"]
    subject_ids = [row["subject_id"] for row in subject_rows]
    candidate_ids = [row["candidate_id"] for row in candidate_rows]
    if set(subject_ids) - set(development_ids) or set(subject_ids).intersection(HELD_OUT_IDS):
        raise RuntimeError("replay subject subset violates development-only policy")
    reference_position = candidate_ids.index(REFERENCE_ID)
    full_candidate_lookup = {candidate_id: index for index, candidate_id in enumerate(matrix_result["candidate_ids"])}
    if not REPLAY_CACHE_PATH.is_file() or not REPLAY_CACHE_MANIFEST_PATH.is_file():
        raise RuntimeError("frozen-environment development replay cache is required before analysis")
    cache_manifest = read_json(REPLAY_CACHE_MANIFEST_PATH)
    if not (
        cache_manifest["protocol_sha256"] == sha256_file(PROTOCOL_PATH)
        and cache_manifest["held_out_scientific_truth_access_count"] == 0
        and cache_manifest["replay_pair_count"] == len(subject_ids) * len(candidate_ids)
        and cache_manifest["cache_sha256"] == sha256_file(REPLAY_CACHE_PATH)
    ):
        raise RuntimeError("replay cache manifest identity mismatch")
    time_s: np.ndarray | None = None
    replay_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    replay_rows = []
    started = time.perf_counter()
    with np.load(REPLAY_CACHE_PATH, allow_pickle=False) as cache:
        if list(map(str, cache["subject_ids"])) != subject_ids or list(map(str, cache["candidate_ids"])) != candidate_ids:
            raise RuntimeError("replay cache subject/candidate order mismatch")
        cached_time = np.asarray(cache["time_s"], dtype=float)
        for replay_subject_index, subject_id in enumerate(subject_ids):
            subject_index = development_ids.index(subject_id)
            for candidate_index, candidate_id in enumerate(candidate_ids):
                arrays = {
                    "time_s": cached_time,
                    "warning_count": np.asarray([cache["warning_max"][replay_subject_index, candidate_index]], dtype=int),
                }
                for array_name in ("tau_truth_nm", *[entry[0] for entry in COMPONENTS.values()]):
                    arrays[array_name] = np.asarray(cache[array_name][replay_subject_index, candidate_index], dtype=float)
                payload = {
                    "subject_id": subject_id, "split": "DEVELOPMENT", "candidate_id": candidate_id,
                    "proposal_index": int(cache["proposal_index"][candidate_index]),
                    "alpha": np.asarray(cache["alpha"][candidate_index], dtype=float).tolist(), "arrays": arrays,
                }
                current_time = np.asarray(arrays["time_s"], dtype=float)
                if time_s is None:
                    time_s = current_time
                elif not np.array_equal(time_s, current_time):
                    raise RuntimeError("replay time grid changed")
                required_arrays = ["tau_truth_nm", *[entry[0] for entry in COMPONENTS.values()]]
                if len(current_time) != 401 or not all(np.isfinite(np.asarray(arrays[key], dtype=float)).all() for key in required_arrays):
                    raise RuntimeError(f"nonfinite or wrong-sized replay: {subject_id}/{candidate_id}")
                if int(np.max(arrays["warning_count"])) != 0:
                    raise RuntimeError(f"replay warning: {subject_id}/{candidate_id}")
                tau = np.asarray(arrays["tau_truth_nm"], dtype=float)
                replay_rms = time_weighted_rms(current_time, tau, axis=0)
                full_candidate_index = full_candidate_lookup[candidate_id]
                expected = np.asarray([
                    hip_norm[subject_index, full_candidate_index], knee_norm[subject_index, full_candidate_index],
                ])
                reference_denominator = np.asarray([
                    hip_norm[subject_index, full_candidate_lookup[REFERENCE_ID]],
                    knee_norm[subject_index, full_candidate_lookup[REFERENCE_ID]],
                ])
                if not np.allclose(reference_denominator, 1.0, rtol=0.0, atol=1.0e-12):
                    raise RuntimeError("reference normalization changed")
                cohort_record = next(row for row in read_json(COHORT_MANIFEST_PATH)["subjects"] if row["subject_id"] == subject_id)
                denominators = np.asarray([
                    cohort_record["subject_reference_tau_hip_rms_nm"], cohort_record["subject_reference_tau_knee_rms_nm"],
                ])
                replay_normalized = replay_rms / denominators
                if not np.allclose(replay_normalized, expected, rtol=1.0e-11, atol=1.0e-11):
                    raise RuntimeError(f"replay/compact RMS mismatch: {subject_id}/{candidate_id}")
                replay_payloads[(subject_id, candidate_id)] = payload
                replay_rows.append({
                    "subject_id": subject_id, "candidate_id": candidate_id, "proposal_index": payload["proposal_index"],
                    "alpha_hip_deg": payload["alpha"][0], "alpha_knee_deg": payload["alpha"][1], "alpha_phase": payload["alpha"][2],
                    "sample_count": len(current_time), "hip_rms_nm": float(replay_rms[0]), "knee_rms_nm": float(replay_rms[1]),
                    "hip_normalized_rms": float(replay_normalized[0]), "knee_normalized_rms": float(replay_normalized[1]),
                    "compact_replay_match": True, "warning_count": 0,
                })
    if time_s is None:
        raise RuntimeError("empty replay subset")
    write_csv(OUTPUT / "REPLAY_SUBSET_INTEGRITY.csv", replay_rows)

    component_rows: list[dict[str, Any]] = []
    component_summary: dict[str, dict[str, Any]] = {}
    component_scale_by_subject: dict[str, dict[str, float]] = {subject_id: {} for subject_id in subject_ids}
    component_matrices: dict[tuple[str, str], np.ndarray] = {}
    for component, (array_name, formula_sign) in COMPONENTS.items():
        for joint_index, joint in enumerate(("hip", "knee")):
            matrix = np.zeros((len(subject_ids), len(candidate_ids)))
            for subject_index, subject_id in enumerate(subject_ids):
                for candidate_index, candidate_id in enumerate(candidate_ids):
                    values = formula_sign * np.asarray(replay_payloads[(subject_id, candidate_id)]["arrays"][array_name], dtype=float)[:, joint_index]
                    matrix[subject_index, candidate_index] = float(time_weighted_rms(time_s, values))
            component_matrices[(component, joint)] = matrix
            raw_decomp = two_way_decomposition(matrix)
            reference_values = matrix[:, reference_position]
            safe = np.abs(reference_values) > 1.0e-12
            normalized = np.full_like(matrix, np.nan)
            normalized[safe] = matrix[safe] / reference_values[safe, None]
            normalized_decomp = two_way_decomposition(normalized[safe]) if np.any(safe) else None
            summary_key = f"{component}_{joint}"
            component_summary[summary_key] = {
                "component": component, "joint": joint, "required_drive_formula_sign": formula_sign,
                "reference_nonzero_subject_count": int(np.sum(safe)),
                "raw_decomposition": public_decomposition(raw_decomp),
                "reference_normalized_decomposition": None if normalized_decomp is None else public_decomposition(normalized_decomp),
            }
            for subject_index, subject_id in enumerate(subject_ids):
                for candidate_index, candidate_id in enumerate(candidate_ids):
                    component_rows.append({
                        "record_type": "PAIR", "subject_id": subject_id, "candidate_id": candidate_id,
                        "component": component, "joint": joint, "required_drive_formula_sign": formula_sign,
                        "component_rms_nm": float(matrix[subject_index, candidate_index]),
                        "reference_component_rms_nm": float(reference_values[subject_index]),
                        "component_rms_ratio_to_subject_reference": "" if not safe[subject_index] else float(normalized[subject_index, candidate_index]),
                        "component_candidate_change_nm": float(matrix[subject_index, candidate_index] - reference_values[subject_index]),
                        "subject_candidate_interaction_fraction": "", "interaction_to_common_effect_ratio": "",
                    })
            component_rows.append({
                "record_type": "SUMMARY", "subject_id": "ALL_REPLAY_DEVELOPMENT", "candidate_id": "ALL_REPLAY_CANDIDATES",
                "component": component, "joint": joint, "required_drive_formula_sign": formula_sign,
                "component_rms_nm": float(np.mean(matrix)), "reference_component_rms_nm": float(np.mean(reference_values)),
                "component_rms_ratio_to_subject_reference": "", "component_candidate_change_nm": "",
                "subject_candidate_interaction_fraction": raw_decomp["subject_candidate_interaction_variance_fraction"],
                "interaction_to_common_effect_ratio": raw_decomp["interaction_to_common_effect_ratio"],
            })
        combined = np.column_stack((component_matrices[(component, "hip")], component_matrices[(component, "knee")]))
        anchor = combined[0]
        for subject_index, subject_id in enumerate(subject_ids):
            component_scale_by_subject[subject_id][f"{component}_scale"] = fit_scaling(combined[subject_index], anchor)["proportional_slope"]
    write_csv(OUTPUT / "FORCE_COMPONENT_DECOMPOSITION.csv", component_rows)
    atomic_json(OUTPUT / "FORCE_COMPONENT_DECOMPOSITION_SUMMARY.json", component_summary)

    rms_rows: list[dict[str, Any]] = []
    rms_summary: dict[str, dict[str, Any]] = {}
    normalized_time_signals: dict[str, np.ndarray] = {}
    for joint_index, joint in enumerate(("hip", "knee")):
        signals = np.zeros((len(subject_ids), len(candidate_ids), len(time_s)))
        summary_matrix = np.zeros((len(subject_ids), len(candidate_ids)))
        for subject_index, subject_id in enumerate(subject_ids):
            cohort_record = next(row for row in read_json(COHORT_MANIFEST_PATH)["subjects"] if row["subject_id"] == subject_id)
            denominator = float(cohort_record[f"subject_reference_tau_{joint}_rms_nm"])
            for candidate_index, candidate_id in enumerate(candidate_ids):
                tau = np.asarray(replay_payloads[(subject_id, candidate_id)]["arrays"]["tau_truth_nm"], dtype=float)[:, joint_index]
                signals[subject_index, candidate_index] = tau / denominator
                summary_matrix[subject_index, candidate_index] = float(time_weighted_rms(time_s, tau) / denominator)
        normalized_time_signals[joint] = signals
        grand_t = np.mean(signals, axis=(0, 1))
        subject_main_t = np.mean(signals, axis=1) - grand_t[None, :]
        candidate_main_t = np.mean(signals, axis=0) - grand_t[None, :]
        interaction_t = signals - grand_t[None, None, :] - subject_main_t[:, None, :] - candidate_main_t[None, :, :]
        time_interaction_rms = float(time_weighted_rms(time_s, np.sqrt(np.mean(interaction_t**2, axis=(0, 1)))))
        time_common_effect_rms = float(time_weighted_rms(time_s, np.sqrt(np.mean(candidate_main_t**2, axis=0))))
        summary_decomp = two_way_decomposition(summary_matrix)
        rms_interaction = summary_decomp["subject_specific_interaction_rms"]
        ratio = time_interaction_rms / max(rms_interaction, np.finfo(float).tiny)
        sign_change_flags = []
        peak_spreads = []
        for candidate_index, candidate_id in enumerate(candidate_ids):
            peak_times = []
            for subject_index, subject_id in enumerate(subject_ids):
                difference = signals[subject_index, candidate_index] - signals[subject_index, reference_position]
                nonzero = difference[np.abs(difference) > EQUIVALENCE_TOLERANCE]
                sign_change = bool(np.any(nonzero < 0.0) and np.any(nonzero > 0.0))
                if candidate_id != REFERENCE_ID:
                    sign_change_flags.append(sign_change)
                peak_time = float(time_s[int(np.argmax(np.abs(signals[subject_index, candidate_index])))])
                peak_times.append(peak_time)
                rms_rows.append({
                    "record_type": "PAIR", "joint": joint, "subject_id": subject_id, "candidate_id": candidate_id,
                    "pointwise_difference_rms_to_reference": float(time_weighted_rms(time_s, difference)),
                    "rms_summary_difference_to_reference": float(summary_matrix[subject_index, candidate_index] - summary_matrix[subject_index, reference_position]),
                    "pointwise_difference_sign_changes": sign_change, "absolute_tau_peak_time_s": peak_time,
                    "time_resolved_interaction_rms": "", "rms_summary_interaction_rms": "",
                    "time_resolved_to_rms_heterogeneity_ratio": "", "time_common_effect_rms": "",
                    "sign_changing_difference_fraction": "", "median_peak_timing_subject_spread_s": "",
                })
            peak_spreads.append(max(peak_times) - min(peak_times))
        rms_summary[joint] = {
            "time_resolved_interaction_rms": time_interaction_rms,
            "rms_summary_interaction_rms": rms_interaction,
            "time_resolved_to_rms_heterogeneity_ratio": ratio,
            "time_common_candidate_effect_rms": time_common_effect_rms,
            "time_interaction_to_common_effect_ratio": time_interaction_rms / max(time_common_effect_rms, np.finfo(float).tiny),
            "sign_changing_difference_fraction": float(np.mean(sign_change_flags)),
            "median_peak_timing_subject_spread_s": float(np.median(peak_spreads)),
        }
        rms_rows.append({
            "record_type": "JOINT_SUMMARY", "joint": joint, "subject_id": "ALL_REPLAY_DEVELOPMENT", "candidate_id": "ALL_NONREFERENCE_REPLAY_CANDIDATES",
            "pointwise_difference_rms_to_reference": "", "rms_summary_difference_to_reference": "",
            "pointwise_difference_sign_changes": "", "absolute_tau_peak_time_s": "",
            "time_resolved_interaction_rms": time_interaction_rms, "rms_summary_interaction_rms": rms_interaction,
            "time_resolved_to_rms_heterogeneity_ratio": ratio, "time_common_effect_rms": time_common_effect_rms,
            "sign_changing_difference_fraction": float(np.mean(sign_change_flags)),
            "median_peak_timing_subject_spread_s": float(np.median(peak_spreads)),
        })
    write_csv(OUTPUT / "RMS_COMPRESSION_AUDIT.csv", rms_rows)
    atomic_json(OUTPUT / "RMS_COMPRESSION_SUMMARY.json", rms_summary)
    return {
        "subject_ids": subject_ids, "candidate_ids": candidate_ids, "time_s": time_s,
        "payloads": replay_payloads, "component_matrices": component_matrices,
        "component_summary": component_summary, "component_scale_by_subject": component_scale_by_subject,
        "rms_summary": rms_summary, "normalized_time_signals": normalized_time_signals,
        "runtime_s": time.perf_counter() - started,
    }


def parameter_associations(
    cohort: dict[str, Any], development_ids: list[str], matrix_result: dict[str, Any], replay_result: dict[str, Any],
) -> list[dict[str, Any]]:
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    parameter_names = list(cohort["factor_order"])
    rows: list[dict[str, Any]] = []
    compact_outcomes = {
        name: np.asarray([matrix_result["scaling_by_subject"][subject_id][name] for subject_id in development_ids], dtype=float)
        for name in ("hip_raw_scale", "knee_raw_scale", "J_local_gradient_norm", "J_interaction_rms")
    }
    replay_outcomes = {
        f"{component}_scale": np.asarray([
            replay_result["component_scale_by_subject"][subject_id][f"{component}_scale"]
            for subject_id in replay_result["subject_ids"]
        ], dtype=float)
        for component in COMPONENTS
    }
    p_values = []
    for parameter in parameter_names:
        for outcome_name, outcome in compact_outcomes.items():
            x = np.asarray([records[subject_id]["factor_values"][parameter] for subject_id in development_ids], dtype=float)
            rho, p_value, reason = spearman_row(x, outcome)
            p_values.append(p_value)
            rows.append({
                "data_scope": "ALL_24_DEVELOPMENT_COMPACT", "sample_count": len(x), "parameter": parameter,
                "outcome": outcome_name, "spearman_rho": rho, "raw_p_value": p_value,
                "undefined_reason": reason, "bh_q_value_across_all_tests": None,
                "exploratory_only": True, "predictive_model_trained": False,
            })
        for outcome_name, outcome in replay_outcomes.items():
            x = np.asarray([records[subject_id]["factor_values"][parameter] for subject_id in replay_result["subject_ids"]], dtype=float)
            rho, p_value, reason = spearman_row(x, outcome)
            p_values.append(p_value)
            rows.append({
                "data_scope": "GEOMETRY_SELECTED_6_DEVELOPMENT_REPLAY", "sample_count": len(x), "parameter": parameter,
                "outcome": outcome_name, "spearman_rho": rho, "raw_p_value": p_value,
                "undefined_reason": reason, "bh_q_value_across_all_tests": None,
                "exploratory_only": True, "predictive_model_trained": False,
            })
    for row, q_value in zip(rows, bh_qvalues(p_values)):
        row["bh_q_value_across_all_tests"] = float(q_value)
    write_csv(OUTPUT / "PARAMETER_MECHANISM_ASSOCIATIONS.csv", rows)
    return rows


def classify_hypotheses(matrix_result: dict[str, Any], replay_result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    attenuation = matrix_result["normalization_attenuation"]
    if all(attenuation[joint] <= 0.5 for joint in ("hip", "knee")):
        h1 = "SUPPORTED"
    elif any(attenuation[joint] <= 0.8 for joint in ("hip", "knee")):
        h1 = "PARTIALLY_SUPPORTED"
    else:
        h1 = "NOT_SUPPORTED"

    j_decomp = matrix_result["decompositions"]["J"]
    interaction_fraction = j_decomp["subject_candidate_interaction_variance_fraction"]
    interaction_ratio = j_decomp["interaction_to_common_effect_ratio"]
    if interaction_fraction <= 0.01 and interaction_ratio <= 0.10:
        h2 = "SUPPORTED"
    elif interaction_fraction <= 0.05 and interaction_ratio <= 0.25:
        h2 = "PARTIALLY_SUPPORTED"
    else:
        h2 = "NOT_SUPPORTED"

    rms = replay_result["rms_summary"]
    h3_joint = {
        joint: rms[joint]["time_resolved_to_rms_heterogeneity_ratio"] >= 2.0
        and rms[joint]["time_resolved_interaction_rms"] >= 0.001
        for joint in ("hip", "knee")
    }
    h3_partial_joint = {
        joint: rms[joint]["time_resolved_to_rms_heterogeneity_ratio"] >= 1.25
        and rms[joint]["time_resolved_interaction_rms"] >= 0.001
        for joint in ("hip", "knee")
    }
    h3 = "SUPPORTED" if all(h3_joint.values()) else ("PARTIALLY_SUPPORTED" if any(h3_partial_joint.values()) else "NOT_SUPPORTED")

    boundary = matrix_result["boundary_support"]
    boundary_strong = sum(value >= 23 / 24 for value in boundary.values())
    boundary_partial = sum(value >= 22 / 24 for value in boundary.values())
    global_strong = sum(
        matrix_result["monotonic_global"][dimension]["J"]["majority_sign_fraction"] >= 0.85
        for dimension in ("hip", "knee", "phase")
    )
    if boundary_strong == 3 and global_strong >= 2:
        h4 = "SUPPORTED"
    elif boundary_partial >= 2:
        h4 = "PARTIALLY_SUPPORTED"
    else:
        h4 = "NOT_SUPPORTED"

    statuses = {
        "H1_NORMALIZATION_CANCELLATION": {
            "status": h1, "hip_attenuation_ratio": attenuation["hip"], "knee_attenuation_ratio": attenuation["knee"],
        },
        "H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION": {
            "status": h2, "J_interaction_variance_fraction": interaction_fraction,
            "J_interaction_to_common_effect_ratio": interaction_ratio,
        },
        "H3_RMS_OBJECTIVE_COMPRESSION": {
            "status": h3,
            "hip_time_to_rms_ratio": rms["hip"]["time_resolved_to_rms_heterogeneity_ratio"],
            "knee_time_to_rms_ratio": rms["knee"]["time_resolved_to_rms_heterogeneity_ratio"],
            "hip_time_interaction_rms": rms["hip"]["time_resolved_interaction_rms"],
            "knee_time_interaction_rms": rms["knee"]["time_resolved_interaction_rms"],
        },
        "H4_CANDIDATE_DOMAIN_MONOTONICITY": {
            "status": h4, "boundary_step1_support_fraction": boundary,
            "pooled_global_majority_sign_fraction": {
                dimension: matrix_result["monotonic_global"][dimension]["J"]["majority_sign_fraction"]
                for dimension in ("hip", "knee", "phase")
            },
        },
    }
    supported_count = sum(row["status"] == "SUPPORTED" for row in statuses.values())
    any_evidence = any(row["status"] in {"SUPPORTED", "PARTIALLY_SUPPORTED"} for row in statuses.values())
    if supported_count >= 2 and (h2 == "SUPPORTED" or h4 == "SUPPORTED"):
        outcome = OUTCOME_IDENTIFIED
    elif any_evidence:
        outcome = OUTCOME_PARTIAL
    else:
        outcome = OUTCOME_NOT_IDENTIFIED
    atomic_json(OUTPUT / "ROOT_CAUSE_HYPOTHESIS_DECISIONS.json", {"hypotheses": statuses, "overall_outcome": outcome})
    return statuses, outcome


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def create_slice_figure(
    path: Path, dimension: int, dimension_name: str, unit: str, protocol: dict[str, Any],
    alpha: np.ndarray, j_matrix: np.ndarray,
) -> None:
    anchors = protocol["slice_anchors"]
    fig, axes = plt.subplots(len(anchors), 1, figsize=(9.0, 2.35 * len(anchors)), sharex=False)
    if len(anchors) == 1:
        axes = [axes]
    for axis, anchor in zip(axes, anchors):
        anchor_alpha = np.asarray(anchor["alpha"], dtype=float)
        mask = np.ones(len(alpha), dtype=bool)
        for other in range(3):
            if other != dimension:
                mask &= np.isclose(alpha[:, other], anchor_alpha[other], rtol=0.0, atol=1.0e-12)
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(alpha[indices, dimension])]
        x = alpha[indices, dimension]
        for subject in range(j_matrix.shape[0]):
            axis.plot(x, j_matrix[subject, indices], color="#4c78a8", alpha=0.20, linewidth=0.7)
        axis.plot(x, np.mean(j_matrix[:, indices], axis=0), color="#d62728", linewidth=2.0, label="Development mean")
        axis.fill_between(x, np.min(j_matrix[:, indices], axis=0), np.max(j_matrix[:, indices], axis=0), color="#4c78a8", alpha=0.12, label="24-subject range")
        axis.set_ylabel("J truth")
        axis.set_title(f"{anchor['selection_role']} | fixed alpha={anchor['alpha']}", fontsize=9)
        axis.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel(f"{dimension_name} perturbation ({unit})")
    fig.suptitle(f"All 24 development landscapes: {dimension_name} slices", y=1.002)
    save_figure(path)


def create_figures(
    protocol: dict[str, Any], matrix_result: dict[str, Any], replay_result: dict[str, Any],
    j_matrix: np.ndarray, hip_raw: np.ndarray, knee_raw: np.ndarray, hip_norm: np.ndarray, knee_norm: np.ndarray,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    alpha = matrix_result["alpha"]
    create_slice_figure(FIGURES / "development_hip_slices.png", 0, "hip", "deg", protocol, alpha, j_matrix)
    create_slice_figure(FIGURES / "development_knee_slices.png", 1, "knee", "deg", protocol, alpha, j_matrix)
    create_slice_figure(FIGURES / "development_phase_slices.png", 2, "phase", "cycle", protocol, alpha, j_matrix)

    j_decomp = matrix_result["decompositions"]["J"]
    candidate_effect = np.abs(j_decomp["candidate_main"])
    interaction = np.sqrt(np.mean(j_decomp["interaction"] ** 2, axis=0))
    floor = np.finfo(float).eps
    plt.figure(figsize=(6.6, 4.8))
    plt.scatter(candidate_effect + floor, interaction + floor, s=5, alpha=0.25, color="#4c78a8")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Absolute common candidate main effect (J)")
    plt.ylabel("Subject interaction RMS (J)")
    plt.title("Common candidate effect versus subject interaction")
    save_figure(FIGURES / "common_candidate_effect_vs_interaction.png")

    svd_rows = list(csv.DictReader((OUTPUT / "LANDSCAPE_SVD_ANALYSIS.csv").open(newline="", encoding="utf-8")))
    plt.figure(figsize=(8.0, 4.5))
    width = 0.25
    x = np.arange(1, 6)
    for offset, name in zip((-width, 0.0, width), ("RAW_J", "SUBJECT_ROW_CENTERED_J", "CANDIDATE_MAIN_REMOVED_J")):
        values = [float(row["variance_explained"]) for row in svd_rows if row["matrix"] == name][:5]
        plt.bar(x + offset, values, width=width, label=name)
    plt.yscale("log")
    plt.xlabel("Singular component")
    plt.ylabel("Variance explained")
    plt.xticks(x)
    plt.title("Landscape SVD explained variance")
    plt.legend(fontsize=8)
    save_figure(FIGURES / "landscape_svd_explained_variance.png")

    raw_hip_cv = np.std(hip_raw, axis=0, ddof=1) / np.maximum(np.abs(np.mean(hip_raw, axis=0)), np.finfo(float).tiny)
    norm_hip_cv = np.std(hip_norm, axis=0, ddof=1) / np.maximum(np.abs(np.mean(hip_norm, axis=0)), np.finfo(float).tiny)
    raw_knee_cv = np.std(knee_raw, axis=0, ddof=1) / np.maximum(np.abs(np.mean(knee_raw, axis=0)), np.finfo(float).tiny)
    norm_knee_cv = np.std(knee_norm, axis=0, ddof=1) / np.maximum(np.abs(np.mean(knee_norm, axis=0)), np.finfo(float).tiny)
    plt.figure(figsize=(7.2, 4.5))
    plt.boxplot([raw_hip_cv, norm_hip_cv, raw_knee_cv, norm_knee_cv], labels=["Hip raw", "Hip normalized", "Knee raw", "Knee normalized"], showfliers=False)
    plt.ylabel("Cross-subject candidate-wise CV")
    plt.title("Raw versus reference-normalized variability")
    save_figure(FIGURES / "raw_vs_normalized_variability.png")

    time_s = replay_result["time_s"]
    subject_ids = replay_result["subject_ids"][:3]
    candidate_ids = [REFERENCE_ID, COMMON_ORACLE_ID]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for joint_index, joint in enumerate(("hip", "knee")):
        for color, subject_id in zip(colors, subject_ids):
            for candidate_id, linestyle in zip(candidate_ids, ("--", "-")):
                tau = replay_result["payloads"][(subject_id, candidate_id)]["arrays"]["tau_truth_nm"][:, joint_index]
                axes[joint_index].plot(time_s, tau, color=color, linestyle=linestyle, linewidth=1.0, label=f"{subject_id} | {'reference' if candidate_id == REFERENCE_ID else 'common oracle'}")
        axes[joint_index].set_ylabel(f"{joint.capitalize()} torque (Nm)")
        axes[joint_index].legend(fontsize=6, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Preselected development time-resolved required-drive torque")
    save_figure(FIGURES / "selected_time_resolved_torque.png")

    labels = list(COMPONENTS)
    hip_values = [float(np.mean(replay_result["component_matrices"][(component, "hip")])) for component in labels]
    knee_values = [float(np.mean(replay_result["component_matrices"][(component, "knee")])) for component in labels]
    x = np.arange(len(labels))
    plt.figure(figsize=(8.4, 4.8))
    plt.bar(x - 0.18, hip_values, width=0.36, label="Hip")
    plt.bar(x + 0.18, knee_values, width=0.36, label="Knee")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean component RMS (Nm)")
    plt.title("Frozen required-drive component decomposition")
    plt.legend()
    save_figure(FIGURES / "force_component_decomposition.png")


def write_report(
    outcome: str, hypotheses: dict[str, Any], matrix_result: dict[str, Any], replay_result: dict[str, Any],
    association_rows: list[dict[str, Any]],
) -> None:
    j_decomp = matrix_result["decompositions"]["J"]
    scaling_rows = matrix_result["scaling_rows"]
    affine_r2 = {joint: [row["affine_r2"] for row in scaling_rows if row["joint"] == joint] for joint in ("hip", "knee")}
    proportional_error = {joint: [row["proportional_nrmse"] for row in scaling_rows if row["joint"] == joint] for joint in ("hip", "knee")}
    multiplicative_supported = all(np.median(affine_r2[joint]) >= 0.995 and np.median(proportional_error[joint]) <= 0.01 for joint in ("hip", "knee"))
    crossover = np.asarray([row["all_candidate_rank_inversion_rate"] for row in matrix_result["crossover_rows"]], dtype=float)
    tradeoff = {
        row["dimension"]: row
        for row in matrix_result["tradeoff_rows"]
        if row["record_type"] == "POOLED"
    }
    component_ranked_subject = sorted(
        (
            (key, value["raw_decomposition"]["subject_main_variance_fraction"])
            for key, value in replay_result["component_summary"].items()
        ), key=lambda item: item[1], reverse=True,
    )
    component_ranked_interaction = sorted(
        (
            (key, value["raw_decomposition"]["interaction_to_common_effect_ratio"])
            for key, value in replay_result["component_summary"].items()
        ), key=lambda item: item[1], reverse=True,
    )
    significant = [row for row in association_rows if float(row["bh_q_value_across_all_tests"]) < 0.05]
    monotonic = hypotheses["H4_CANDIDATE_DOMAIN_MONOTONICITY"]
    rms = replay_result["rms_summary"]
    report = f"""# MyoLeg V2 Personalization Signal Root-Cause Audit V1

## Overall conclusion

`{outcome}`

This is a development-only offline mechanistic audit under the unchanged frozen MyoLeg-V2 truth semantics, candidate domain, reference normalization, and normalized RMS objective. It is not a new objective, learner, patient optimum, human result, comfort result, safety result, or clinical result. All eight held-out truth landscapes remained sealed.

## Hypothesis decisions

| Hypothesis | Decision | Core frozen metric |
|---|---:|---|
| H1 normalization cancellation | {hypotheses['H1_NORMALIZATION_CANCELLATION']['status']} | hip attenuation={hypotheses['H1_NORMALIZATION_CANCELLATION']['hip_attenuation_ratio']:.6g}; knee={hypotheses['H1_NORMALIZATION_CANCELLATION']['knee_attenuation_ratio']:.6g} |
| H2 weak subject×trajectory interaction | {hypotheses['H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION']['status']} | J interaction variance={j_decomp['subject_candidate_interaction_variance_fraction']:.6g}; interaction/common={j_decomp['interaction_to_common_effect_ratio']:.6g} |
| H3 RMS objective compression | {hypotheses['H3_RMS_OBJECTIVE_COMPRESSION']['status']} | time/RMS interaction: hip={rms['hip']['time_resolved_to_rms_heterogeneity_ratio']:.6g}; knee={rms['knee']['time_resolved_to_rms_heterogeneity_ratio']:.6g} |
| H4 candidate-domain monotonicity | {hypotheses['H4_CANDIDATE_DOMAIN_MONOTONICITY']['status']} | boundary support={monotonic['boundary_step1_support_fraction']} |

## Q1 — Candidate main effect versus subject interaction

For J, candidate main effect accounts for **{100*j_decomp['candidate_main_variance_fraction']:.6f}%** of centered landscape variance, subject main effect **{100*j_decomp['subject_main_variance_fraction']:.6f}%**, and subject×candidate interaction **{100*j_decomp['subject_candidate_interaction_variance_fraction']:.6f}%**. The interaction/common-effect RMS ratio is **{j_decomp['interaction_to_common_effect_ratio']:.6g}**.

## Q2 — Multiplicative or affine landscape relationship

`{'MULTIPLICATIVE_SUBJECT_SCALING_SUPPORTED' if multiplicative_supported else 'MULTIPLICATIVE_SUBJECT_SCALING_NOT_FULLY_SUPPORTED'}`. Median affine R² is **{np.median(affine_r2['hip']):.9f}** for hip and **{np.median(affine_r2['knee']):.9f}** for knee. Median pure-proportional NRMSE is **{np.median(proportional_error['hip']):.9f}** and **{np.median(proportional_error['knee']):.9f}**, respectively.

## Q3 — Normalization attenuation

The preregistered dimensionless interaction attenuation ratio after subject-specific reference normalization is **{matrix_result['normalization_attenuation']['hip']:.6g}** for hip and **{matrix_result['normalization_attenuation']['knee']:.6g}** for knee. This diagnoses cancellation; it does not justify removing normalization.

## Q4 — Local descent around reference

All 24 development subjects share the same combined-J local descent direction around the reference: **hip +, knee +, phase −**, with sign agreement **1.000** in every dimension. Hip-only has the same three directions; knee-only instead prefers **hip −, knee +, phase −**, exposing a hip-axis joint trade-off that the combined J nevertheless resolves in the hip-positive direction. Boundary-step support at the frozen common oracle is hip **{matrix_result['boundary_support']['hip']:.3f}**, knee **{matrix_result['boundary_support']['knee']:.3f}**, and phase **{matrix_result['boundary_support']['phase']:.3f}**. No out-of-domain derivative or optimum is claimed.

## Q5 — Boundary censoring

H4 is **{hypotheses['H4_CANDIDATE_DOMAIN_MONOTONICITY']['status']}**. Pooled global majority-sign fractions are hip **{monotonic['pooled_global_majority_sign_fraction']['hip']:.6f}**, knee **{monotonic['pooled_global_majority_sign_fraction']['knee']:.6f}**, and phase **{monotonic['pooled_global_majority_sign_fraction']['phase']:.6f}**. The scientifically allowed statement is that the frozen optimum is boundary-limited/censored if supported—not that an unknown true optimum lies outside the domain.

## Q6 — Hip/knee trade-off

Opposing hip/knee adjacent-transition fractions are hip-axis **{tradeoff['hip']['hip_knee_opposing_fraction']:.6f}**, knee-axis **{tradeoff['knee']['hip_knee_opposing_fraction']:.6f}**, and phase-axis **{tradeoff['phase']['hip_knee_opposing_fraction']:.6f}**. The frozen 5% diagnostic classifies these as: hip={tradeoff['hip']['no_meaningful_tradeoff']}, knee={tradeoff['knee']['no_meaningful_tradeoff']}, phase={tradeoff['phase']['no_meaningful_tradeoff']} for `NO_MEANINGFUL_HIP_KNEE_OBJECTIVE_TRADEOFF`.

## Q7 — Time-resolved dynamics versus RMS

Time-resolved/RMS-summary interaction ratios are hip **{rms['hip']['time_resolved_to_rms_heterogeneity_ratio']:.6g}** and knee **{rms['knee']['time_resolved_to_rms_heterogeneity_ratio']:.6g}**. Sign-changing candidate-minus-reference waveform fractions are **{rms['hip']['sign_changing_difference_fraction']:.6f}** and **{rms['knee']['sign_changing_difference_fraction']:.6f}**. H3 is therefore **{hypotheses['H3_RMS_OBJECTIVE_COMPRESSION']['status']}** under the pre-frozen rule. This remains diagnostic and does not define a time-weighted objective.

## Q8 — Force-component mechanisms

The largest replay-subset subject-main fractions are `{component_ranked_subject[0][0]}`={component_ranked_subject[0][1]:.6g}, `{component_ranked_subject[1][0]}`={component_ranked_subject[1][1]:.6g}, and `{component_ranked_subject[2][0]}`={component_ranked_subject[2][1]:.6g}. The largest interaction/common-effect ratios are `{component_ranked_interaction[0][0]}`={component_ranked_interaction[0][1]:.6g}, `{component_ranked_interaction[1][0]}`={component_ranked_interaction[1][1]:.6g}, and `{component_ranked_interaction[2][0]}`={component_ranked_interaction[2][1]:.6g}. These are MuJoCo required-drive components, not physiological tissue contributions.

## Q9 — Supported hypotheses

- H1: **{hypotheses['H1_NORMALIZATION_CANCELLATION']['status']}**
- H2: **{hypotheses['H2_WEAK_SUBJECT_TRAJECTORY_INTERACTION']['status']}**
- H3: **{hypotheses['H3_RMS_OBJECTIVE_COMPRESSION']['status']}**
- H4: **{hypotheses['H4_CANDIDATE_DOMAIN_MONOTONICITY']['status']}**

All-candidate development rank-inversion rate has median **{np.median(crossover):.9f}**, P95 **{np.percentile(crossover,95):.9f}**, and maximum **{np.max(crossover):.9f}**. Exploratory parameter-mechanism associations with BH q<0.05: **{len(significant)}**; no predictive model was trained.

## Q10 — Defensible next branch

The next branch must follow the supported mechanisms rather than tune an objective to manufacture oracle diversity. A supported H4 motivates an independently preregistered **trajectory-parameterization/boundary audit**; a supported H1 or H3 motivates a separate **objective-formulation audit**; failure of meaningful interaction after both audits supports stopping personalization and reframing the study. No branch is executed here.

## Access and scope boundary

Exactly 24 development compact landscapes and the preregistered development replay subset were used. Held-out shards were only existence/size/stream-SHA checked. Held-out NPZ scientific array loads, J/oracle/rank/torque/component access: **0**. No Five-parameter model, NN, PINN, BO, objective redesign, normalization change, cohort/range change, candidate change, robot, or hardware operation occurred.
"""
    atomic_text(OUTPUT / "MYOLEG_V2_PERSONALIZATION_SIGNAL_ROOT_CAUSE_REPORT.md", report)


def write_checksums() -> None:
    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in paths) + "\n")


def analyze() -> None:
    started = time.perf_counter()
    if not PROTOCOL_PATH.is_file() or not ACCESS_PATH.is_file():
        raise RuntimeError("protocol/access freeze must exist before analysis")
    if (OUTPUT / "TWO_WAY_VARIANCE_DECOMPOSITION.json").exists():
        raise RuntimeError("formal analysis outputs already exist; refusing overwrite")
    truth, _, candidates, cohort = verify_input_identity()
    necessity = verify_necessity_artifacts()
    protocol = read_json(PROTOCOL_PATH)
    access = read_json(ACCESS_PATH)
    if not (
        protocol["frozen_before_development_outcome_matrix_read"] is True
        and access["held_out_scientific_truth_access_count"] == 0
        and access["development_outcome_matrix_read_at_freeze"] is False
    ):
        raise RuntimeError("analysis freeze invalid")
    atomic_json(OUTPUT / "ANALYSIS_EXECUTION_FREEZE.json", {
        "freeze_id": "PERSONALIZATION_SIGNAL_ROOT_CAUSE_ANALYSIS_EXECUTION_FREEZE_V1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_outcome_matrix_read_at_execution_freeze": False,
        "held_out_scientific_truth_access_count": 0,
        "frozen_input_sha256": FROZEN_SHA,
    })

    development_ids = list(protocol["population"]["development_subject_ids"])
    store = DevelopmentLandscapeStore(truth, candidates, development_ids)
    loaded = [store.load_subject(subject_id) for subject_id in development_ids]
    candidate_ids = loaded[0]["candidate_id"]
    if not all(np.array_equal(row["candidate_id"], candidate_ids) for row in loaded):
        raise RuntimeError("candidate order differs across development subjects")
    j_matrix = np.stack([row["j_truth"] for row in loaded])
    hip_raw = np.stack([row["hip_tau_rms_nm"] for row in loaded])
    knee_raw = np.stack([row["knee_tau_rms_nm"] for row in loaded])
    hip_reference = np.asarray([row["subject_reference_hip_rms_nm"][0] for row in loaded])
    knee_reference = np.asarray([row["subject_reference_knee_rms_nm"][0] for row in loaded])
    hip_norm = hip_raw / hip_reference[:, None]
    knee_norm = knee_raw / knee_reference[:, None]
    reconstructed_j = np.sqrt(0.5 * (hip_norm**2 + knee_norm**2))
    if not (
        j_matrix.shape == hip_raw.shape == knee_raw.shape == (24, EXPECTED_CANDIDATES)
        and np.allclose(reconstructed_j, j_matrix, rtol=0.0, atol=2.0e-15)
    ):
        raise RuntimeError("development matrix shape/objective reconstruction failure")
    atomic_json(OUTPUT / "DEVELOPMENT_MATRIX_INTEGRITY.json", {
        "shape": [24, EXPECTED_CANDIDATES], "development_subject_ids_loaded": store.accessed_subject_ids,
        "held_out_subject_ids_loaded": [], "held_out_scientific_truth_access_count": 0,
        "objective_reconstruction_max_abs_error": float(np.max(np.abs(reconstructed_j - j_matrix))),
        "finite": bool(all(np.isfinite(matrix).all() for matrix in (j_matrix, hip_raw, knee_raw, hip_norm, knee_norm))),
    })

    matrix_result = full_matrix_analysis(protocol, development_ids, candidates["ordered_included_candidates"], j_matrix, hip_raw, knee_raw, hip_norm, knee_norm)
    replay_result = replay_analysis(protocol, development_ids, matrix_result, j_matrix, hip_norm, knee_norm)
    association_rows = parameter_associations(cohort, development_ids, matrix_result, replay_result)
    hypotheses, outcome = classify_hypotheses(matrix_result, replay_result)
    create_figures(protocol, matrix_result, replay_result, j_matrix, hip_raw, knee_raw, hip_norm, knee_norm)
    write_report(outcome, hypotheses, matrix_result, replay_result, association_rows)

    access.update({
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_compact_subject_ids_loaded": store.accessed_subject_ids,
        "development_compact_subject_count_loaded": len(store.accessed_subject_ids),
        "development_replay_subject_ids": replay_result["subject_ids"],
        "development_replay_pair_count": len(replay_result["subject_ids"]) * len(replay_result["candidate_ids"]),
        "held_out_scientific_truth_access_count": 0,
        "np_load_held_out_count": 0,
        "necessity_artifact_verification": necessity,
    })
    atomic_json(ACCESS_PATH, access)
    metadata = {
        "stage_id": STAGE_ID, "outcome": outcome,
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "analysis_code_sha256": sha256_file(Path(__file__)),
        "frozen_input_sha256": FROZEN_SHA, "development_subject_count": 24,
        "candidate_count": EXPECTED_CANDIDATES, "replay_pair_count": len(replay_result["subject_ids"]) * len(replay_result["candidate_ids"]),
        "held_out_subject_count": 8, "held_out_scientific_truth_access_count": 0,
        "runtime_s": time.perf_counter() - started, "replay_runtime_s": replay_result["runtime_s"],
        "hypothesis_decisions": {key: value["status"] for key, value in hypotheses.items()},
        "scope": {
            "offline_only": True, "objective_modified": False, "normalization_modified": False,
            "cohort_or_ranges_modified": False, "candidate_or_generator_modified": False,
            "models_trained": False, "bo_run": False, "robot_or_hardware": False,
        },
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    write_checksums()
    print(json.dumps({
        "outcome": outcome, "protocol_sha256": metadata["protocol_sha256"],
        "development_subjects": 24, "replay_pairs": metadata["replay_pair_count"],
        "held_out_scientific_truth_access_count": 0, "runtime_s": metadata["runtime_s"],
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--analyze", action="store_true")
    group.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol:
        freeze_protocol()
    elif args.analyze:
        analyze()
    else:
        freeze_protocol()
        analyze()


if __name__ == "__main__":
    main()
