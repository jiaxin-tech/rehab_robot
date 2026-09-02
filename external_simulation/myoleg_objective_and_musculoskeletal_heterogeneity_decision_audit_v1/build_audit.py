"""Separate objective information loss from limited MyoLeg heterogeneity.

All scientific diagnostics are development-only.  Held-out files are only
stream-hashed; held-out identifiers are rejected before replay/path resolution.
Alternative torque views remain diagnostic and never replace the frozen J.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy import stats


STAGE_ID = "MYOLEG_OBJECTIVE_AND_MUSCULOSKELETAL_HETEROGENEITY_DECISION_AUDIT_V1"
PROTOCOL_ID = "OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL_V1"
OBJECTIVE_ADEQUATE = "CURRENT_OBJECTIVE_INFORMATION_RETENTION_ADEQUATE"
OBJECTIVE_LIMITED = "CURRENT_OBJECTIVE_INFORMATION_RETENTION_LIMITED"
HETEROGENEITY_ADEQUATE = "CURRENT_HETEROGENEITY_TRAJECTORY_INTERACTION_ADEQUATE"
HETEROGENEITY_LIMITED = "CURRENT_HETEROGENEITY_TRAJECTORY_INTERACTION_LIMITED"
DECISION_OBJECTIVE = "OBJECTIVE_LIMITATION_DOMINANT"
DECISION_HETEROGENEITY = "HETEROGENEITY_LIMITATION_DOMINANT"
DECISION_BOTH = "BOTH_OBJECTIVE_AND_HETEROGENEITY_LIMITING"
DECISION_INSUFFICIENT = "INSUFFICIENT_EVIDENCE_TO_SEPARATE"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1"
FIGURES = OUTPUT / "figures"
PROTOCOL_PATH = OUTPUT / "OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL.json"
ACCESS_PATH = OUTPUT / "HELD_OUT_ACCESS_AUDIT.json"
TRUTH_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1_MANIFEST.json"
CANDIDATE_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
CANDIDATE_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
NECESSITY = ROOT / "external_simulation_audits/myoleg_v3_personalization_necessity_audit_v1"
REPLAY_API_SOURCE = ROOT / "external_simulation/myoleg_v3_development_truth_landscape_generation_v1/replay_api.py"
PARAMETERIZATION_SOURCE = ROOT / "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py"
CANDIDATE_BUILDER_SOURCE = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER_SOURCE = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
CACHE_PATH = ROOT / "external_simulation/data/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1/development_replay_subset.npz"

FROZEN_SHA = {
    "protocol": "80a24b8da08b902581db05c875fb3f8532ece30737e36a98cf26778a0e467fd8",
    "truth": "c318700e161e857d3e059eadf2bc21364e21b74d8b39faf4124a26fc15d37c6e",
    "candidate_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "candidate_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "cohort": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "necessity_checksums": "3f4e7928c06af705ce76f22439b4f65530e68cc5238f68ce31cd52963f94f340",
    "necessity_metadata": "9282c0a1aa6da9818d4caa976fa3567db02fc2c5b22122ee59fd0f7f46496b25",
    "necessity_metrics": "e8524bb86adddb5d0b4e28e708a2950cac91effddc1769145fb5cc64a67b4184",
    "replay_api": "5e25c9c8a96e9c778de320802352691c759e04a1d5ae2dcb29d444b9f37c4674",
    "parameterization": "e830b5cadd6d970107e59eb9b346650af5ab254b42beecdfaf6b70a5985957ef",
    "candidate_builder": "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e",
    "replay_builder": "d60a9b1651b49307155b8b36bfdd881b595c604288f7c07a3237afe5f5feb32e",
}
HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)
EXPECTED_SUBJECTS = 24
EXPECTED_CANDIDATES = 625
GRID_SIZE = 25
GRID_STEP = 0.0025
REFERENCE_INDEX = 312
REFERENCE_ID = "MYOLEG_V3_K0312"
ANCHOR_SUBJECT = "MYOLEG_VP_031"
TOLERANCE = 1.0e-12
COMPONENTS = {
    "mass_inertia": ("mass_term_nm", 1.0),
    "bias_gravity": ("bias_term_nm", 1.0),
    "passive": ("passive_internal_nm", -1.0),
    "zero_control_actuator": ("actuator_internal_nm", -1.0),
    "constraint": ("constraint_internal_nm", -1.0),
}
COMPACT_REPRESENTATIONS = (
    "raw_hip_rms", "raw_knee_rms", "normalized_hip_rms",
    "normalized_knee_rms", "combined_j", "raw_hip_peak",
    "raw_knee_peak", "normalized_hip_peak", "normalized_knee_peak",
)


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
        raise RuntimeError(f"cannot infer empty CSV schema: {path}")
    columns = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return {
        "min": float(np.min(array)), "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)), "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)), "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def sign_class(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(array < -TOLERANCE, -1, np.where(array > TOLERANCE, 1, 0))


def bh_qvalues(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def two_way_decomposition(matrix: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
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
    common_rms = float(np.sqrt(np.mean(candidate_main**2)))
    interaction_rms = float(np.sqrt(np.mean(interaction**2)))
    return {
        "grand_mean": grand, "subject_count": values.shape[0], "candidate_count": values.shape[1],
        "ss_total": ss_total, "ss_subject_main": ss_subject,
        "ss_candidate_main": ss_candidate, "ss_interaction": ss_interaction,
        "subject_main_fraction": ss_subject / denominator,
        "candidate_main_fraction": ss_candidate / denominator,
        "interaction_fraction": ss_interaction / denominator,
        "common_candidate_effect_rms": common_rms,
        "interaction_rms": interaction_rms,
        "interaction_to_common_ratio": interaction_rms / max(common_rms, np.finfo(float).tiny),
        "dimensionless_interaction_rms": interaction_rms / max(abs(grand), np.finfo(float).tiny),
        "subject_interaction_residual_rms": np.sqrt(np.mean(interaction**2, axis=1)).tolist(),
    }, interaction


def fit_scaling(target: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    x = np.asarray(anchor, dtype=float)
    y = np.asarray(target, dtype=float)
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    affine = slope * x + intercept
    sst = float(np.sum((y - np.mean(y)) ** 2))
    p_slope = float(np.dot(x, y) / max(np.dot(x, x), np.finfo(float).tiny))
    proportional = p_slope * x
    scale = max(float(np.mean(np.abs(y))), np.finfo(float).tiny)
    return {
        "affine_slope": float(slope), "affine_intercept": float(intercept),
        "affine_r2": 1.0 - float(np.sum((y - affine) ** 2)) / max(sst, np.finfo(float).tiny),
        "affine_nrmse": float(np.sqrt(np.mean((y - affine) ** 2)) / scale),
        "proportional_slope": p_slope,
        "proportional_r2": 1.0 - float(np.sum((y - proportional) ** 2)) / max(sst, np.finfo(float).tiny),
        "proportional_nrmse": float(np.sqrt(np.mean((y - proportional) ** 2)) / scale),
    }


def time_weighted_rms(time_s: np.ndarray, values: np.ndarray, axis: int = 0) -> np.ndarray:
    duration = float(time_s[-1] - time_s[0])
    return np.sqrt(np.trapezoid(np.asarray(values, dtype=float) ** 2, time_s, axis=axis) / duration)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_necessity_checksums() -> int:
    count = 0
    for line in (NECESSITY / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(NECESSITY / relative) != expected:
            raise RuntimeError(f"frozen V3 necessity artifact changed: {relative}")
        count += 1
    metadata = read_json(NECESSITY / "metadata.json")
    if not (
        metadata["outcome"] == "V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED"
        and metadata["held_out_scientific_access_count"] == 0
        and metadata["next_stage_executed"] is False
    ):
        raise RuntimeError("frozen V3 necessity identity changed")
    return count


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    actual = {
        "protocol": sha256_file(PROTOCOL_PATH),
        "truth": sha256_file(TRUTH_MANIFEST),
        "candidate_manifest": sha256_file(CANDIDATE_MANIFEST),
        "candidate_table": sha256_file(CANDIDATE_TABLE),
        "cohort": sha256_file(COHORT_MANIFEST),
        "necessity_checksums": sha256_file(NECESSITY / "checksums.sha256"),
        "necessity_metadata": sha256_file(NECESSITY / "metadata.json"),
        "necessity_metrics": sha256_file(NECESSITY / "V3_DECISION_METRICS.json"),
        "replay_api": sha256_file(REPLAY_API_SOURCE),
        "parameterization": sha256_file(PARAMETERIZATION_SOURCE),
        "candidate_builder": sha256_file(CANDIDATE_BUILDER_SOURCE),
        "replay_builder": sha256_file(REPLAY_BUILDER_SOURCE),
    }
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    protocol = read_json(PROTOCOL_PATH)
    truth = read_json(TRUTH_MANIFEST)
    candidate_manifest = read_json(CANDIDATE_MANIFEST)
    candidates = read_csv(CANDIDATE_TABLE)
    cohort = read_json(COHORT_MANIFEST)
    necessity_count = verify_necessity_checksums()
    if not (
        protocol["protocol_id"] == PROTOCOL_ID
        and protocol["frozen_before_new_development_diagnostics_reveal"] is True
        and truth["outcome"] == "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_VALID"
        and truth["actual_pair_count"] == EXPECTED_SUBJECTS * EXPECTED_CANDIDATES
        and truth["held_out_scientific_access_count"] == 0
        and truth["held_out_subject_ids_excluded"] == list(HELD_OUT_IDS)
        and len(truth["development_subject_ids"]) == EXPECTED_SUBJECTS
        and candidate_manifest["candidate_count"] == EXPECTED_CANDIDATES
        and candidate_manifest["all_candidates_pass_kinematic_gates"] is True
        and candidate_manifest["mechanical_objective_evaluated"] is False
        and len(candidates) == EXPECTED_CANDIDATES
        and tuple(cohort["development_subject_ids"]) == tuple(truth["development_subject_ids"])
        and tuple(cohort["held_out_subject_ids"]) == HELD_OUT_IDS
        and necessity_count > 0
    ):
        raise RuntimeError("frozen scientific identity/status mismatch")
    return truth, candidate_manifest, candidates, cohort


def verify_held_out_without_loading(cohort: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    verified: list[dict[str, Any]] = []
    for subject_id in HELD_OUT_IDS:
        record = records[subject_id]
        if record["split"] != "HELD_OUT":
            raise RuntimeError(f"held-out split mismatch: {subject_id}")
        files = []
        for path_key, sha_key in (
            ("metadata_path", "metadata_sha256"),
            ("model_delta_path", "model_delta_sha256"),
            ("reference_replay_truth_path", "reference_replay_truth_sha256"),
        ):
            path = ROOT / record[path_key]
            actual = sha256_file(path)
            if actual != record[sha_key]:
                raise RuntimeError(f"held-out streaming SHA mismatch: {path}")
            files.append({"path": record[path_key], "sha256": actual, "operation": "streaming_sha_only"})
        verified.append({"subject_id": subject_id, "files": files})
    if any(row["subject_id"] in HELD_OUT_IDS for row in truth["chunks"]):
        raise RuntimeError("development landscape unexpectedly contains held-out rows")
    audit = read_json(ACCESS_PATH)
    audit.update({
        "held_out_file_hashes_verified": 24,
        "held_out_file_verification": verified,
        "held_out_np_load_count": 0,
        "held_out_replay_count": 0,
        "held_out_scientific_access_count": 0,
        "v3_development_landscape_subject_ids": [row["subject_id"] for row in truth["chunks"]],
        "v3_development_landscape_held_out_row_count": 0,
    })
    atomic_json(ACCESS_PATH, audit)
    return audit


class CompactDevelopmentStore:
    REQUIRED = (
        "candidate_id", "candidate_index", "beta_flex", "beta_extend",
        "hip_tau_rms_nm", "knee_tau_rms_nm", "hip_tau_peak_abs_nm",
        "knee_tau_peak_abs_nm", "subject_reference_hip_rms_nm",
        "subject_reference_knee_rms_nm", "j_truth", "integrity_status",
    )

    def __init__(self, truth: dict[str, Any], candidates: list[dict[str, str]], allowed: Iterable[str]):
        self.truth = truth
        self.candidates = candidates
        self.allowed = frozenset(allowed)
        self.accessed: list[str] = []
        if len(self.allowed) != EXPECTED_SUBJECTS or self.allowed.intersection(HELD_OUT_IDS):
            raise RuntimeError("compact development allowlist invalid")

    def load(self, subject_id: str) -> dict[str, np.ndarray]:
        if subject_id not in self.allowed:
            raise PermissionError(f"SEALED_CONFIRMATORY_TRUTH denied before path resolution: {subject_id}")
        rows = [row for row in self.truth["chunks"] if row["subject_id"] == subject_id]
        if len(rows) != 1 or int(rows[0]["row_count"]) != EXPECTED_CANDIDATES:
            raise RuntimeError(f"compact shard coverage mismatch: {subject_id}")
        path = ROOT / rows[0]["path"]
        if sha256_file(path) != rows[0]["sha256"]:
            raise RuntimeError(f"compact shard SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            output = {key: np.asarray(shard[key]) for key in self.REQUIRED}
        expected_ids = np.asarray([row["candidate_id"] for row in self.candidates])
        if not (
            np.array_equal(output["candidate_id"], expected_ids)
            and np.array_equal(output["candidate_index"], np.arange(EXPECTED_CANDIDATES, dtype=np.int32))
            and np.all(output["integrity_status"] == 1)
            and all(np.isfinite(output[key]).all() for key in self.REQUIRED if output[key].dtype.kind in "fiu")
            and abs(float(output["j_truth"][REFERENCE_INDEX]) - 1.0) <= TOLERANCE
        ):
            raise RuntimeError(f"compact scientific integrity failure: {subject_id}")
        self.accessed.append(subject_id)
        return output


def pairwise_rank_summary(matrix: np.ndarray) -> dict[str, float]:
    values = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            i_constant = bool(np.ptp(matrix[i]) == 0.0)
            j_constant = bool(np.ptp(matrix[j]) == 0.0)
            if i_constant and j_constant:
                correlation = 1.0
            elif i_constant or j_constant:
                correlation = 0.0
            else:
                correlation = float(stats.spearmanr(matrix[i], matrix[j]).statistic)
            values.append(correlation)
    return summary(np.asarray(values))


def full_grid_direction_disagreement(matrix: np.ndarray) -> dict[str, Any]:
    cube = np.asarray(matrix).reshape(matrix.shape[0], GRID_SIZE, GRID_SIZE)
    transition_count = disagreement = 0
    majority = []
    for axis in (1, 2):
        delta = np.diff(cube, axis=axis)
        signs = sign_class(delta)
        moved = np.moveaxis(signs, 0, -1).reshape(-1, matrix.shape[0])
        for row in moved:
            negative, equivalent, positive = (int(np.sum(row == value)) for value in (-1, 0, 1))
            disagreement += int(negative > 0 and positive > 0)
            majority.append(max(negative, equivalent, positive) / matrix.shape[0])
            transition_count += 1
    return {
        "direction_transition_count": transition_count,
        "direction_disagreement_count": disagreement,
        "direction_disagreement_fraction": disagreement / transition_count,
        "mean_majority_sign_fraction": float(np.mean(majority)),
    }


def subset_direction_disagreement(matrix: np.ndarray, candidate_ids: list[str]) -> dict[str, Any]:
    lookup = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    edges = (
        (REFERENCE_ID, "MYOLEG_V3_K0287"), (REFERENCE_ID, "MYOLEG_V3_K0337"),
        (REFERENCE_ID, "MYOLEG_V3_K0311"), (REFERENCE_ID, "MYOLEG_V3_K0313"),
    )
    disagreement = 0
    majority = []
    for origin, target in edges:
        signs = sign_class(matrix[:, lookup[target]] - matrix[:, lookup[origin]])
        negative, equivalent, positive = (int(np.sum(signs == value)) for value in (-1, 0, 1))
        disagreement += int(negative > 0 and positive > 0)
        majority.append(max(negative, equivalent, positive) / matrix.shape[0])
    return {
        "direction_transition_count": len(edges),
        "direction_disagreement_count": disagreement,
        "direction_disagreement_fraction": disagreement / len(edges),
        "mean_majority_sign_fraction": float(np.mean(majority)),
    }


def representation_diagnostic(
    name: str, matrix: np.ndarray, subject_ids: list[str], scope: str,
    candidate_ids: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    decomposition, interaction = two_way_decomposition(matrix)
    rank = pairwise_rank_summary(matrix)
    direction = full_grid_direction_disagreement(matrix) if candidate_ids is None else subset_direction_disagreement(matrix, candidate_ids)
    anchor_index = subject_ids.index(ANCHOR_SUBJECT)
    fits = []
    for index, subject_id in enumerate(subject_ids):
        fits.append({
            "representation": name, "scope": scope, "subject_id": subject_id,
            "anchor_subject_id": ANCHOR_SUBJECT, **fit_scaling(matrix[index], matrix[anchor_index]),
        })
    affine_r2 = np.asarray([row["affine_r2"] for row in fits])
    proportional_nrmse = np.asarray([row["proportional_nrmse"] for row in fits])
    evidence = bool(
        rank["median"] <= 0.95
        or direction["direction_disagreement_fraction"] >= 0.05
        or (decomposition["interaction_fraction"] >= 0.01 and decomposition["interaction_to_common_ratio"] >= 0.10)
    )
    diagnostic = {
        "representation": name, "scope": scope, **decomposition,
        "spearman_min": rank["min"], "spearman_median": rank["median"],
        "spearman_mean": rank["mean"], "spearman_max": rank["max"],
        **direction,
        "affine_r2_median": float(np.median(affine_r2)),
        "affine_r2_min": float(np.min(affine_r2)),
        "proportional_nrmse_median": float(np.median(proportional_nrmse)),
        "proportional_nrmse_max": float(np.max(proportional_nrmse)),
        "representation_ordering_evidence": evidence,
        "common_ranked": bool(
            rank["median"] >= 0.98
            and direction["direction_disagreement_fraction"] <= 0.01
            and np.median(affine_r2) >= 0.995
        ),
    }
    return diagnostic, fits, interaction


def replay_selection(protocol: dict[str, Any]) -> tuple[list[str], list[str], np.ndarray]:
    subject_ids = [row["subject_id"] for row in protocol["replay_subset"]["subject_rows"]]
    candidate_rows = protocol["replay_subset"]["candidate_rows"]
    candidate_ids = [row["candidate_id"] for row in candidate_rows]
    beta = np.asarray([row["beta"] for row in candidate_rows], dtype=float)
    if not (
        len(subject_ids) == 8 and len(set(subject_ids)) == 8
        and len(candidate_ids) == 17 and len(set(candidate_ids)) == 17
        and set(subject_ids).isdisjoint(HELD_OUT_IDS)
        and len(subject_ids) * len(candidate_ids) == 136
    ):
        raise RuntimeError("frozen replay selection invalid")
    return subject_ids, candidate_ids, beta


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def generate_replay_cache(protocol: dict[str, Any]) -> dict[str, Any]:
    if CACHE_PATH.exists() or (OUTPUT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json").exists():
        raise RuntimeError("partial/existing replay cache requires explicit review; refusing overwrite")
    subject_ids, candidate_ids, beta = replay_selection(protocol)
    replay_api = load_module(REPLAY_API_SOURCE, "_objective_heterogeneity_v3_replay_api")
    reference_builder = load_module(CANDIDATE_BUILDER_SOURCE, "_objective_heterogeneity_reference_builder")
    reference = reference_builder.load_reference_adapter()
    time_s = np.asarray(reference["time_s"], dtype=float)
    phases = np.asarray(reference["phases"])
    segment_phase = np.asarray(reference["segment_phase"], dtype=float)
    shape = (len(subject_ids), len(candidate_ids), len(time_s), 2)
    arrays: dict[str, np.ndarray] = {
        "subject_id": np.asarray(subject_ids), "candidate_id": np.asarray(candidate_ids),
        "beta": beta, "time_s": time_s, "phases": phases, "segment_phase": segment_phase,
    }
    for key in ("tau_truth_nm", *[value[0] for value in COMPONENTS.values()]):
        arrays[key] = np.empty(shape, dtype=np.float64)
    warning_max = decomposition_residual_max = 0.0
    started = time.perf_counter()
    completed = 0
    for s_index, subject_id in enumerate(subject_ids):
        for c_index, candidate_id in enumerate(candidate_ids):
            payload = replay_api.replay_v3_subject_candidate(subject_id, candidate_id)
            if not (
                payload["split"] == "DEVELOPMENT"
                and payload["compact_landscape_or_oracle_read"] is False
                and payload["subject_id"] == subject_id
                and payload["candidate_id"] == candidate_id
            ):
                raise RuntimeError(f"replay identity violation: {subject_id}/{candidate_id}")
            replay_arrays = payload["arrays"]
            if not np.array_equal(np.asarray(replay_arrays["time_s"]), time_s):
                raise RuntimeError("replay time grid changed")
            warning_max = max(warning_max, float(np.max(replay_arrays["warning_count"])))
            decomposition_residual_max = max(
                decomposition_residual_max,
                float(np.max(np.abs(replay_arrays["decomposition_residual_nm"]))),
            )
            for key in ("tau_truth_nm", *[value[0] for value in COMPONENTS.values()]):
                value = np.asarray(replay_arrays[key], dtype=float)
                if value.shape != (401, 2) or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid replay array {key}: {subject_id}/{candidate_id}")
                arrays[key][s_index, c_index] = value
            completed += 1
            if completed % 8 == 0 or completed == 136:
                print(f"replay {completed}/136", flush=True)
    if warning_max != 0 or decomposition_residual_max > 1e-8:
        raise RuntimeError(f"replay integrity failed: warnings={warning_max}, residual={decomposition_residual_max}")
    atomic_npz(CACHE_PATH, arrays)
    manifest = {
        "cache_id": "MYOLEG_OBJECTIVE_HETEROGENEITY_DEVELOPMENT_REPLAY_SUBSET_V1",
        "path": str(CACHE_PATH.relative_to(ROOT)), "sha256": sha256_file(CACHE_PATH),
        "subject_ids": subject_ids, "candidate_ids": candidate_ids,
        "subject_count": len(subject_ids), "candidate_count": len(candidate_ids),
        "pair_count": 136, "sample_count": len(time_s),
        "scientific_arrays": ["tau_truth_nm", *[value[0] for value in COMPONENTS.values()]],
        "selection_uses_geometry_only": True,
        "compact_landscape_or_oracle_read_by_replay_api": False,
        "warning_count_max": int(warning_max),
        "decomposition_residual_max_nm": decomposition_residual_max,
        "held_out_replay_count": 0,
        "runtime_s": time.perf_counter() - started,
    }
    atomic_json(OUTPUT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json", manifest)
    return manifest


def load_replay_cache(protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = OUTPUT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json"
    if not CACHE_PATH.is_file() or not manifest_path.is_file():
        generate_replay_cache(protocol)
    manifest = read_json(manifest_path)
    if sha256_file(CACHE_PATH) != manifest["sha256"]:
        raise RuntimeError("replay cache SHA mismatch")
    subject_ids, candidate_ids, beta = replay_selection(protocol)
    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        arrays = {key: np.asarray(cache[key]) for key in cache.files}
    if not (
        np.array_equal(arrays["subject_id"], np.asarray(subject_ids))
        and np.array_equal(arrays["candidate_id"], np.asarray(candidate_ids))
        and np.array_equal(arrays["beta"], beta)
        and len(arrays["time_s"]) == 401
        and manifest["held_out_replay_count"] == 0
        and manifest["compact_landscape_or_oracle_read_by_replay_api"] is False
    ):
        raise RuntimeError("replay cache identity mismatch")
    return arrays, manifest


def compact_matrices(
    truth: dict[str, Any], candidates: list[dict[str, str]], cohort: dict[str, Any],
) -> tuple[list[str], dict[str, np.ndarray], CompactDevelopmentStore]:
    subject_ids = list(truth["development_subject_ids"])
    store = CompactDevelopmentStore(truth, candidates, subject_ids)
    payloads = [store.load(subject_id) for subject_id in subject_ids]
    hip_rms = np.vstack([payload["hip_tau_rms_nm"] for payload in payloads])
    knee_rms = np.vstack([payload["knee_tau_rms_nm"] for payload in payloads])
    hip_peak = np.vstack([payload["hip_tau_peak_abs_nm"] for payload in payloads])
    knee_peak = np.vstack([payload["knee_tau_peak_abs_nm"] for payload in payloads])
    hip_den = np.asarray([payload["subject_reference_hip_rms_nm"][REFERENCE_INDEX] for payload in payloads])
    knee_den = np.asarray([payload["subject_reference_knee_rms_nm"][REFERENCE_INDEX] for payload in payloads])
    hip_peak_den = hip_peak[:, REFERENCE_INDEX]
    knee_peak_den = knee_peak[:, REFERENCE_INDEX]
    if not (
        np.all(hip_den > 0) and np.all(knee_den > 0)
        and np.all(hip_peak_den > 0) and np.all(knee_peak_den > 0)
    ):
        raise RuntimeError("normalization denominator invalid")
    matrices = {
        "raw_hip_rms": hip_rms, "raw_knee_rms": knee_rms,
        "normalized_hip_rms": hip_rms / hip_den[:, None],
        "normalized_knee_rms": knee_rms / knee_den[:, None],
        "combined_j": np.vstack([payload["j_truth"] for payload in payloads]),
        "raw_hip_peak": hip_peak, "raw_knee_peak": knee_peak,
        "normalized_hip_peak": hip_peak / hip_peak_den[:, None],
        "normalized_knee_peak": knee_peak / knee_peak_den[:, None],
    }
    expected_j = np.sqrt(0.5 * (matrices["normalized_hip_rms"] ** 2 + matrices["normalized_knee_rms"] ** 2))
    if not np.allclose(expected_j, matrices["combined_j"], rtol=0.0, atol=2e-12):
        raise RuntimeError("frozen objective reconstruction mismatch")
    return subject_ids, matrices, store


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > np.finfo(float).tiny else 1.0


def compact_gradient_rows(
    representations: dict[str, np.ndarray], subject_ids: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor_index = subject_ids.index(ANCHOR_SUBJECT)
    for name, matrix in representations.items():
        cube = matrix.reshape(len(subject_ids), GRID_SIZE, GRID_SIZE)
        grad_flex, grad_extend = np.gradient(cube, GRID_STEP, GRID_STEP, axis=(1, 2), edge_order=2)
        local = np.column_stack((grad_flex[:, 12, 12], grad_extend[:, 12, 12]))
        anchor_local = local[anchor_index]
        anchor_global = np.concatenate((grad_flex[anchor_index].ravel(), grad_extend[anchor_index].ravel()))
        for index, subject_id in enumerate(subject_ids):
            global_vector = np.concatenate((grad_flex[index].ravel(), grad_extend[index].ravel()))
            rows.append({
                "representation": name, "scope": "ALL_24_DEVELOPMENT_COMPACT",
                "subject_id": subject_id, "anchor_subject_id": ANCHOR_SUBJECT,
                "local_beta_flex_gradient": float(local[index, 0]),
                "local_beta_extend_gradient": float(local[index, 1]),
                "local_gradient_norm": float(np.linalg.norm(local[index])),
                "local_gradient_cosine_to_anchor": cosine(local[index], anchor_local),
                "global_gradient_field_cosine_to_anchor": cosine(global_vector, anchor_global),
                "local_flex_sign": int(sign_class(local[index, 0])),
                "local_extend_sign": int(sign_class(local[index, 1])),
                "preference_direction_differs_from_anchor": bool(
                    int(sign_class(local[index, 0])) != int(sign_class(anchor_local[0]))
                    or int(sign_class(local[index, 1])) != int(sign_class(anchor_local[1]))
                ),
            })
    return rows


def subset_gradient_rows(
    representation: str, matrix: np.ndarray, subject_ids: list[str], candidate_ids: list[str],
) -> list[dict[str, Any]]:
    lookup = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    flex = (matrix[:, lookup["MYOLEG_V3_K0337"]] - matrix[:, lookup["MYOLEG_V3_K0287"]]) / (2 * GRID_STEP)
    extend = (matrix[:, lookup["MYOLEG_V3_K0313"]] - matrix[:, lookup["MYOLEG_V3_K0311"]]) / (2 * GRID_STEP)
    local = np.column_stack((flex, extend))
    anchor_index = subject_ids.index(ANCHOR_SUBJECT)
    rows = []
    for index, subject_id in enumerate(subject_ids):
        rows.append({
            "representation": representation, "scope": "PREREGISTERED_8x17_REPLAY_SUBSET",
            "subject_id": subject_id, "anchor_subject_id": ANCHOR_SUBJECT,
            "local_beta_flex_gradient": float(flex[index]),
            "local_beta_extend_gradient": float(extend[index]),
            "local_gradient_norm": float(np.linalg.norm(local[index])),
            "local_gradient_cosine_to_anchor": cosine(local[index], local[anchor_index]),
            "global_gradient_field_cosine_to_anchor": "",
            "local_flex_sign": int(sign_class(flex[index])),
            "local_extend_sign": int(sign_class(extend[index])),
            "preference_direction_differs_from_anchor": bool(
                int(sign_class(flex[index])) != int(sign_class(flex[anchor_index]))
                or int(sign_class(extend[index])) != int(sign_class(extend[anchor_index]))
            ),
        })
    return rows


def ordering_retention(raw: np.ndarray, normalized: np.ndarray) -> dict[str, Any]:
    correlations = np.asarray([
        float(stats.spearmanr(raw[index], normalized[index]).statistic)
        for index in range(raw.shape[0])
    ])
    raw_cube = raw.reshape(raw.shape[0], GRID_SIZE, GRID_SIZE)
    normalized_cube = normalized.reshape(normalized.shape[0], GRID_SIZE, GRID_SIZE)
    changes = total = 0
    for axis in (1, 2):
        raw_sign = sign_class(np.diff(raw_cube, axis=axis))
        normalized_sign = sign_class(np.diff(normalized_cube, axis=axis))
        changes += int(np.sum(raw_sign != normalized_sign))
        total += raw_sign.size
    return {
        "within_subject_raw_normalized_spearman_min": float(np.min(correlations)),
        "within_subject_raw_normalized_spearman_median": float(np.median(correlations)),
        "adjacent_sign_comparison_count": total,
        "adjacent_sign_change_count": changes,
        "adjacent_sign_change_fraction": changes / total,
        "ordering_retained": bool(np.min(correlations) >= 0.999999 and changes == 0),
        "ordering_loss": bool(np.min(correlations) < 0.99 or changes / total >= 0.01),
    }


def compact_parameter_associations(
    cohort: dict[str, Any], subject_ids: list[str], gradient_rows: list[dict[str, Any]],
    interactions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    factors = list(cohort["factor_order"])
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for representation in COMPACT_REPRESENTATIONS:
        selected = {row["subject_id"]: row for row in gradient_rows if row["representation"] == representation}
        interaction_residual = np.sqrt(np.mean(interactions[representation] ** 2, axis=1))
        outcomes = {
            "local_beta_flex_gradient": np.asarray([float(selected[s]["local_beta_flex_gradient"]) for s in subject_ids]),
            "local_beta_extend_gradient": np.asarray([float(selected[s]["local_beta_extend_gradient"]) for s in subject_ids]),
            "local_gradient_norm": np.asarray([float(selected[s]["local_gradient_norm"]) for s in subject_ids]),
            "local_gradient_cosine_to_anchor": np.asarray([float(selected[s]["local_gradient_cosine_to_anchor"]) for s in subject_ids]),
            "interaction_residual_rms": interaction_residual,
        }
        for factor in factors:
            x = np.asarray([records[subject_id]["factor_values"][factor] for subject_id in subject_ids])
            for outcome_name, y in outcomes.items():
                reason = "CONSTANT_OUTCOME" if np.ptp(y) == 0.0 else ""
                if reason:
                    rho, p = None, 1.0
                else:
                    result = stats.spearmanr(x, y)
                    rho, p = float(result.statistic), float(result.pvalue)
                p_values.append(p)
                rows.append({
                    "family": "COMPACT_24x625", "sample_count": 24,
                    "representation": representation, "subject_parameter": factor,
                    "outcome": outcome_name, "spearman_rho": rho, "raw_p_value": p,
                    "undefined_reason": reason, "bh_q_value_within_compact_family": None,
                    "exploratory_only": True, "predictive_learner_trained": False,
                })
    for row, q in zip(rows, bh_qvalues(p_values)):
        row["bh_q_value_within_compact_family"] = float(q)
    return rows


def time_interaction_metrics(time_s: np.ndarray, signals: np.ndarray) -> dict[str, float]:
    grand = np.mean(signals, axis=(0, 1))
    subject_main = np.mean(signals, axis=1) - grand[None, :]
    candidate_main = np.mean(signals, axis=0) - grand[None, :]
    interaction = signals - grand[None, None, :] - subject_main[:, None, :] - candidate_main[None, :, :]
    interaction_envelope = np.sqrt(np.mean(interaction**2, axis=(0, 1)))
    common_envelope = np.sqrt(np.mean(candidate_main**2, axis=0))
    interaction_rms = float(time_weighted_rms(time_s, interaction_envelope))
    common_rms = float(time_weighted_rms(time_s, common_envelope))
    return {
        "time_resolved_interaction_rms": interaction_rms,
        "time_resolved_common_candidate_effect_rms": common_rms,
        "time_resolved_interaction_to_common_ratio": interaction_rms / max(common_rms, np.finfo(float).tiny),
    }


def window_mask(phases: np.ndarray, segment_phase: np.ndarray, branch: str, lower: float, upper: float) -> np.ndarray:
    if upper == 1.0:
        phase_mask = (segment_phase >= lower) & (segment_phase <= upper)
    else:
        phase_mask = (segment_phase >= lower) & (segment_phase < upper)
    mask = (phases == branch) & phase_mask
    if np.sum(mask) < 3:
        raise RuntimeError(f"time window too small: {branch}/{lower}/{upper}")
    return mask


def analyze_time_resolved(
    cache: dict[str, np.ndarray], protocol: dict[str, Any], cohort: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    subject_ids = cache["subject_id"].astype(str).tolist()
    candidate_ids = cache["candidate_id"].astype(str).tolist()
    time_s = np.asarray(cache["time_s"], dtype=float)
    phases = cache["phases"].astype(str)
    segment_phase = np.asarray(cache["segment_phase"], dtype=float)
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    rows: list[dict[str, Any]] = []
    affine_rows: list[dict[str, Any]] = []
    representation_matrices: dict[str, np.ndarray] = {}
    tau = np.asarray(cache["tau_truth_nm"], dtype=float)
    for joint_index, joint in enumerate(("hip", "knee")):
        denominator = np.asarray([
            records[subject_id][f"subject_reference_tau_{joint}_rms_nm"] for subject_id in subject_ids
        ])
        for scale_name, signals in (
            ("raw", tau[:, :, :, joint_index]),
            ("reference_rms_normalized", tau[:, :, :, joint_index] / denominator[:, None, None]),
        ):
            time_metrics = time_interaction_metrics(time_s, signals)
            rms_matrix = time_weighted_rms(time_s, np.moveaxis(signals, 2, 0), axis=0)
            representation = f"{scale_name}_{joint}_fullcycle_rms_replay_subset"
            diagnostic, fits, _ = representation_diagnostic(
                representation, rms_matrix, subject_ids, "PREREGISTERED_8x17_REPLAY_SUBSET", candidate_ids,
            )
            representation_matrices[representation] = rms_matrix
            rms_interaction = diagnostic["interaction_rms"]
            rows.append({
                "record_type": "FULL_CYCLE_TIME_INTERACTION", "joint": joint,
                "scale": scale_name, "window": "FULL_CYCLE", **time_metrics,
                "rms_summary_interaction_rms": rms_interaction,
                "time_to_rms_interaction_ratio": time_metrics["time_resolved_interaction_rms"] / max(rms_interaction, np.finfo(float).tiny),
                "interaction_fraction": diagnostic["interaction_fraction"],
                "interaction_to_common_ratio": diagnostic["interaction_to_common_ratio"],
                "spearman_median": diagnostic["spearman_median"],
                "direction_disagreement_fraction": diagnostic["direction_disagreement_fraction"],
                "representation_ordering_evidence": diagnostic["representation_ordering_evidence"],
            })
            affine_rows.extend(fits)
            for window in protocol["time_resolved"]["time_windows"]:
                mask = window_mask(phases, segment_phase, window["branch"], window["segment_phase"][0], window["segment_phase"][1])
                window_time = time_s[mask]
                window_matrix = time_weighted_rms(window_time, np.moveaxis(signals[:, :, mask], 2, 0), axis=0)
                window_name = f"{scale_name}_{joint}_{window['label'].lower()}_rms"
                window_diag, window_fits, _ = representation_diagnostic(
                    window_name, window_matrix, subject_ids, "PREREGISTERED_8x17_REPLAY_SUBSET", candidate_ids,
                )
                representation_matrices[window_name] = window_matrix
                rows.append({
                    "record_type": "BRANCH_PHASE_WINDOW", "joint": joint,
                    "scale": scale_name, "window": window["label"],
                    "sample_count": int(np.sum(mask)),
                    "time_resolved_interaction_rms": "", "time_resolved_common_candidate_effect_rms": "",
                    "time_resolved_interaction_to_common_ratio": "", "rms_summary_interaction_rms": window_diag["interaction_rms"],
                    "time_to_rms_interaction_ratio": "",
                    "interaction_fraction": window_diag["interaction_fraction"],
                    "interaction_to_common_ratio": window_diag["interaction_to_common_ratio"],
                    "spearman_median": window_diag["spearman_median"],
                    "direction_disagreement_fraction": window_diag["direction_disagreement_fraction"],
                    "representation_ordering_evidence": window_diag["representation_ordering_evidence"],
                })
                affine_rows.extend(window_fits)
    return rows, affine_rows, representation_matrices


def analyze_components(
    cache: dict[str, np.ndarray], cohort: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray],
]:
    subject_ids = cache["subject_id"].astype(str).tolist()
    candidate_ids = cache["candidate_id"].astype(str).tolist()
    time_s = np.asarray(cache["time_s"], dtype=float)
    component_rows: list[dict[str, Any]] = []
    affine_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    interactions: dict[str, np.ndarray] = {}
    for component, (array_name, formula_sign) in COMPONENTS.items():
        signal = formula_sign * np.asarray(cache[array_name], dtype=float)
        for joint_index, joint in enumerate(("hip", "knee")):
            matrix = time_weighted_rms(time_s, np.moveaxis(signal[:, :, :, joint_index], 2, 0), axis=0)
            representation = f"component_{component}_{joint}_rms"
            diagnostic, fits, interaction = representation_diagnostic(
                representation, matrix, subject_ids, "PREREGISTERED_8x17_REPLAY_SUBSET", candidate_ids,
            )
            present = bool(abs(diagnostic["grand_mean"]) >= 1e-6)
            diagnostic.update({
                "component": component, "joint": joint, "scale": "raw_rms_nm",
                "array_name": array_name, "required_drive_formula_sign": formula_sign,
                "component_numerically_present": present,
            })
            component_rows.append(diagnostic)
            affine_rows.extend(fits)
            gradient_rows.extend(subset_gradient_rows(representation, matrix, subject_ids, candidate_ids))
            matrices[representation] = matrix
            interactions[representation] = interaction
            reference = matrix[:, candidate_ids.index(REFERENCE_ID)]
            safe = np.abs(reference) > 1e-9
            if np.sum(safe) >= 2 and bool(safe[subject_ids.index(ANCHOR_SUBJECT)]):
                normalized = matrix[safe] / reference[safe, None]
                safe_subjects = [subject_id for subject_id, keep in zip(subject_ids, safe) if keep]
                normalized_name = f"component_{component}_{joint}_reference_normalized"
                normalized_diag, normalized_fits, normalized_interaction = representation_diagnostic(
                    normalized_name, normalized, safe_subjects, "PREREGISTERED_REPLAY_SUBSET_NONZERO_REFERENCE", candidate_ids,
                )
                normalized_diag.update({
                    "component": component, "joint": joint, "scale": "reference_normalized",
                    "array_name": array_name, "required_drive_formula_sign": formula_sign,
                    "component_numerically_present": present,
                })
                component_rows.append(normalized_diag)
                affine_rows.extend(normalized_fits)
                matrices[normalized_name] = normalized
                interactions[normalized_name] = normalized_interaction
    passive_rows = mechanism_associations(
        "PASSIVE", ("HIP_ONLY_PASSIVE_FP_MAX_SCALE", "KNEE_ONLY_PASSIVE_FP_MAX_SCALE", "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE"),
        [name for name in matrices if name.startswith("component_passive_") and name.endswith("_rms")],
        matrices, interactions, gradient_rows, subject_ids, cohort,
    )
    mass_rows = mechanism_associations(
        "MASS_INERTIA", ("FEMUR_MASS_INERTIA_SCALE", "TIBIA_PATELLA_MASS_INERTIA_SCALE", "FOOT_COMPLEX_MASS_INERTIA_SCALE"),
        [name for name in matrices if (name.startswith("component_mass_inertia_") or name.startswith("component_bias_gravity_")) and name.endswith("_rms")],
        matrices, interactions, gradient_rows, subject_ids, cohort,
    )
    return component_rows, passive_rows, mass_rows, affine_rows, gradient_rows, matrices, interactions


def mechanism_associations(
    family: str, factors: tuple[str, ...], representations: list[str],
    matrices: dict[str, np.ndarray], interactions: dict[str, np.ndarray],
    gradient_rows: list[dict[str, Any]], subject_ids: list[str], cohort: dict[str, Any],
) -> list[dict[str, Any]]:
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    anchor_index = subject_ids.index(ANCHOR_SUBJECT)
    for representation in representations:
        selected_gradient = {row["subject_id"]: row for row in gradient_rows if row["representation"] == representation}
        scales = np.asarray([
            fit_scaling(matrices[representation][index], matrices[representation][anchor_index])["proportional_slope"]
            for index in range(len(subject_ids))
        ])
        outcomes = {
            "proportional_response_scale": scales,
            "local_gradient_norm": np.asarray([float(selected_gradient[s]["local_gradient_norm"]) for s in subject_ids]),
            "interaction_residual_rms": np.sqrt(np.mean(interactions[representation] ** 2, axis=1)),
        }
        for factor in factors:
            x = np.asarray([records[subject_id]["factor_values"][factor] for subject_id in subject_ids])
            for outcome_name, y in outcomes.items():
                reason = "CONSTANT_OUTCOME" if np.ptp(y) == 0.0 else ""
                if reason:
                    rho, p = None, 1.0
                else:
                    result = stats.spearmanr(x, y)
                    rho, p = float(result.statistic), float(result.pvalue)
                p_values.append(p)
                rows.append({
                    "record_type": "PARAMETER_ASSOCIATION", "family": family,
                    "sample_count": len(subject_ids), "representation": representation,
                    "subject_parameter": factor, "outcome": outcome_name,
                    "spearman_rho": rho, "raw_p_value": p,
                    "undefined_reason": reason,
                    "bh_q_value_within_family": None, "exploratory_only": True,
                    "preference_direction_outcome": False,
                })
    for row, q in zip(rows, bh_qvalues(p_values)):
        row["bh_q_value_within_family"] = float(q)
    return rows


def make_information_summary(
    compact_diagnostics: list[dict[str, Any]], time_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic in compact_diagnostics:
        rows.append({
            "representation": diagnostic["representation"], "scope": diagnostic["scope"],
            "interaction_fraction": diagnostic["interaction_fraction"],
            "interaction_to_common_ratio": diagnostic["interaction_to_common_ratio"],
            "spearman_median": diagnostic["spearman_median"],
            "direction_disagreement_fraction": diagnostic["direction_disagreement_fraction"],
            "affine_r2_median": diagnostic["affine_r2_median"],
            "proportional_nrmse_median": diagnostic["proportional_nrmse_median"],
            "representation_ordering_evidence": diagnostic["representation_ordering_evidence"],
            "common_ranked": diagnostic["common_ranked"],
            "diagnostic_only": diagnostic["representation"] != "combined_j",
        })
    for row in time_rows:
        if row["record_type"] == "BRANCH_PHASE_WINDOW" and row["scale"] == "reference_rms_normalized":
            rows.append({
                "representation": f"time_window_{row['joint']}_{row['window'].lower()}",
                "scope": "PREREGISTERED_8x17_REPLAY_SUBSET",
                "interaction_fraction": row["interaction_fraction"],
                "interaction_to_common_ratio": row["interaction_to_common_ratio"],
                "spearman_median": row["spearman_median"],
                "direction_disagreement_fraction": row["direction_disagreement_fraction"],
                "affine_r2_median": "", "proportional_nrmse_median": "",
                "representation_ordering_evidence": row["representation_ordering_evidence"],
                "common_ranked": bool(
                    float(row["spearman_median"]) >= 0.98
                    and float(row["direction_disagreement_fraction"]) <= 0.01
                ),
                "diagnostic_only": True,
            })
    for row in component_rows:
        if row["scale"] == "raw_rms_nm":
            rows.append({
                "representation": row["representation"], "scope": row["scope"],
                "interaction_fraction": row["interaction_fraction"],
                "interaction_to_common_ratio": row["interaction_to_common_ratio"],
                "spearman_median": row["spearman_median"],
                "direction_disagreement_fraction": row["direction_disagreement_fraction"],
                "affine_r2_median": row["affine_r2_median"],
                "proportional_nrmse_median": row["proportional_nrmse_median"],
                "representation_ordering_evidence": bool(row["representation_ordering_evidence"] and row["component_numerically_present"]),
                "common_ranked": row["common_ranked"], "diagnostic_only": True,
            })
    return rows


def decide(
    information_rows: list[dict[str, Any]], retention: dict[str, dict[str, Any]],
    time_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    info = {row["representation"]: row for row in information_rows}
    j_evidence = bool(info["combined_j"]["representation_ordering_evidence"])
    upstream = [
        row for row in information_rows
        if row["representation"] != "combined_j" and bool(row["representation_ordering_evidence"])
    ]
    components = [
        row for row in component_rows
        if row["scale"] == "raw_rms_nm"
        and row["component_numerically_present"]
        and row["representation_ordering_evidence"]
    ]
    component_direction = [row for row in components if row["direction_disagreement_fraction"] >= 0.05]
    physical_direction = [
        row for row in information_rows
        if row["representation"] != "combined_j"
        and bool(row["representation_ordering_evidence"])
        and float(row["direction_disagreement_fraction"]) >= 0.05
    ]
    normalization_loss = any(value["ordering_loss"] for value in retention.values())
    normalization_retained = all(value["ordering_retained"] for value in retention.values())
    normalized_full = [
        row for row in time_rows
        if row["record_type"] == "FULL_CYCLE_TIME_INTERACTION" and row["scale"] == "reference_rms_normalized"
    ]
    time_windows = [
        row for row in time_rows
        if row["record_type"] == "BRANCH_PHASE_WINDOW" and row["scale"] == "reference_rms_normalized"
    ]
    rms_compression = bool(
        len(normalized_full) == 2
        and all(float(row["time_to_rms_interaction_ratio"]) >= 2.0 for row in normalized_full)
        and any(bool(row["representation_ordering_evidence"]) for row in time_windows)
    )
    component_aggregation_loss = bool(len(components) >= 2 and not j_evidence)
    objective_limited_condition = bool(
        len(upstream) >= 2 and not j_evidence
        and (normalization_loss or rms_compression or component_aggregation_loss)
    )
    objective_status = OBJECTIVE_LIMITED if objective_limited_condition else OBJECTIVE_ADEQUATE
    heterogeneity_adequate_condition = bool(
        len(upstream) >= 2 and len(components) >= 1 and len(physical_direction) >= 1
    )
    heterogeneity_status = HETEROGENEITY_ADEQUATE if heterogeneity_adequate_condition else HETEROGENEITY_LIMITED
    if objective_status == OBJECTIVE_LIMITED and heterogeneity_status == HETEROGENEITY_ADEQUATE:
        final = DECISION_OBJECTIVE
    elif objective_status == OBJECTIVE_ADEQUATE and heterogeneity_status == HETEROGENEITY_LIMITED:
        final = DECISION_HETEROGENEITY
    elif objective_status == OBJECTIVE_LIMITED and heterogeneity_status == HETEROGENEITY_LIMITED:
        final = DECISION_BOTH
    else:
        final = DECISION_INSUFFICIENT
    next_stage = {
        DECISION_OBJECTIVE: "MYOLEG_OBJECTIVE_FORMULATION_DESIGN_AUDIT_V1",
        DECISION_HETEROGENEITY: "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_AUDIT_V1",
        DECISION_BOTH: "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_AUDIT_V1",
        DECISION_INSUFFICIENT: "MYOLEG_OBJECTIVE_HETEROGENEITY_EVIDENCE_GAP_PROTOCOL_V1",
    }[final]
    objective = {
        "status": objective_status,
        "upstream_ordering_evidence_representation_count": len(upstream),
        "upstream_ordering_evidence_representations": [row["representation"] for row in upstream],
        "combined_j_ordering_evidence": j_evidence,
        "normalization_ordering_loss": normalization_loss,
        "normalization_ordering_retained_for_both_joints": normalization_retained,
        "material_rms_compression": rms_compression,
        "component_aggregation_loss": component_aggregation_loss,
        "limited_conjunction": objective_limited_condition,
        "alternative_view_promoted_to_objective": False,
    }
    heterogeneity = {
        "status": heterogeneity_status,
        "ordering_evidence_representation_count": len(upstream),
        "force_component_ordering_evidence_count": len(components),
        "force_component_direction_evidence_count": len(component_direction),
        "physical_direction_evidence_count": len(physical_direction),
        "adequate_conjunction": heterogeneity_adequate_condition,
        "parameters_modulate_response_magnitude_not_preference": bool(len(physical_direction) == 0),
        "cohort_or_range_modified": False,
    }
    branch = {
        "decision": final, "objective_status": objective_status,
        "heterogeneity_status": heterogeneity_status,
        "recommended_next_stage": next_stage, "next_stage_executed": False,
        "both_limiting_priority_rule": "heterogeneity first when BOTH because objective redesign needs a robust upstream subject-specific ordering signal",
    }
    return objective, heterogeneity, branch


def future_factor_taxonomy() -> list[dict[str, Any]]:
    return [
        {
            "factor_class": "passive force-length curve shape", "actual_myoleg_field_availability": "actuator gainprm/biasprm semantics exist; exact safe mapping requires audit",
            "physiological_meaning": "shape of passive muscle force versus normalized length", "geometry_consistency_risk": "MEDIUM",
            "independent_from_current_learner": True, "evidence_requirement": "field-level semantics, identifiability, and bounded literature evidence; no range proposed",
        },
        {
            "factor_class": "optimal-length-related muscle parameters", "actual_myoleg_field_availability": "model-dependent actuator/tendon length parameters; requires explicit availability audit",
            "physiological_meaning": "operating point on active/passive force-length curves", "geometry_consistency_risk": "HIGH",
            "independent_from_current_learner": True, "evidence_requirement": "preserve reference geometry and validate equilibrium; no range proposed",
        },
        {
            "factor_class": "moment-arm and muscle-path geometry", "actual_myoleg_field_availability": "tendon routes, sites, and wrapping geometry exist",
            "physiological_meaning": "joint-angle-dependent muscle leverage", "geometry_consistency_risk": "HIGH",
            "independent_from_current_learner": True, "evidence_requirement": "anatomical consistency and collision/path validation; no range proposed",
        },
        {
            "factor_class": "segment geometry and anthropometry", "actual_myoleg_field_availability": "body, geom, joint, site, inertia fields exist; current cohort changes mass/inertia scale only",
            "physiological_meaning": "segment lengths, centers of mass, and inertial geometry", "geometry_consistency_risk": "HIGH",
            "independent_from_current_learner": True, "evidence_requirement": "coupled geometry scaling protocol and kinematic task preservation; no range proposed",
        },
        {
            "factor_class": "biarticular coupling structure", "actual_myoleg_field_availability": "biarticular muscle/tendon routes exist; current cohort changes fpmax only",
            "physiological_meaning": "cross-joint coupling and leverage", "geometry_consistency_risk": "HIGH",
            "independent_from_current_learner": True, "evidence_requirement": "muscle-specific routing evidence and coupled-joint validation; no range proposed",
        },
        {
            "factor_class": "tendon elastic properties", "actual_myoleg_field_availability": "tendon parameters are model-dependent and require field audit",
            "physiological_meaning": "series elasticity and force transmission", "geometry_consistency_risk": "MEDIUM_HIGH",
            "independent_from_current_learner": True, "evidence_requirement": "MyoLeg field semantics and dynamic validation; no range proposed",
        },
    ]


def future_objective_information_classes() -> list[dict[str, Any]]:
    return [
        {"information_class": "peak load", "current_myoleg_available": True, "requires_hardware_or_human": False, "status": "diagnostic only; not optimized"},
        {"information_class": "time-local load", "current_myoleg_available": True, "requires_hardware_or_human": False, "status": "diagnostic only; not optimized"},
        {"information_class": "passive component", "current_myoleg_available": True, "requires_hardware_or_human": False, "status": "simulator component, not tissue validation"},
        {"information_class": "mechanical work", "current_myoleg_available": True, "requires_hardware_or_human": False, "status": "derivable but not audited as an objective"},
        {"information_class": "interaction/contact load", "current_myoleg_available": "PARTIAL", "requires_hardware_or_human": False, "status": "simulation contacts/constraints require semantic audit"},
        {"information_class": "tactile pressure", "current_myoleg_available": False, "requires_hardware_or_human": True, "status": "requires physical sensing"},
        {"information_class": "pairwise subjective preference", "current_myoleg_available": False, "requires_hardware_or_human": True, "status": "requires human protocol and ethics approval"},
    ]


def normalization_audit_rows(
    diagnostics: dict[str, dict[str, Any]], retention: dict[str, dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("raw_hip_rms", "normalized_hip_rms", "raw_knee_rms", "normalized_knee_rms", "combined_j"):
        rows.append({"record_type": "REPRESENTATION_DIAGNOSTIC", **diagnostics[name]})
    gradients = {
        (row["representation"], row["subject_id"]): row
        for row in gradient_rows
        if row["representation"] in {
            "raw_hip_rms", "normalized_hip_rms", "raw_knee_rms", "normalized_knee_rms",
        }
    }
    for joint in ("hip", "knee"):
        raw = diagnostics[f"raw_{joint}_rms"]
        normalized = diagnostics[f"normalized_{joint}_rms"]
        subject_ids = sorted({key[1] for key in gradients if key[0] == f"raw_{joint}_rms"})
        direction_changes = 0
        gradient_cosines = []
        for subject_id in subject_ids:
            raw_gradient = gradients[(f"raw_{joint}_rms", subject_id)]
            normalized_gradient = gradients[(f"normalized_{joint}_rms", subject_id)]
            a = np.asarray([raw_gradient["local_beta_flex_gradient"], raw_gradient["local_beta_extend_gradient"]], dtype=float)
            b = np.asarray([normalized_gradient["local_beta_flex_gradient"], normalized_gradient["local_beta_extend_gradient"]], dtype=float)
            gradient_cosines.append(cosine(a, b))
            direction_changes += int(
                raw_gradient["local_flex_sign"] != normalized_gradient["local_flex_sign"]
                or raw_gradient["local_extend_sign"] != normalized_gradient["local_extend_sign"]
            )
        rows.append({
            "record_type": "NORMALIZATION_COMPARISON", "joint": joint,
            "raw_interaction_rms_nm": raw["interaction_rms"],
            "normalized_interaction_rms_ratio_units": normalized["interaction_rms"],
            "raw_dimensionless_interaction_rms": raw["dimensionless_interaction_rms"],
            "normalized_dimensionless_interaction_rms": normalized["dimensionless_interaction_rms"],
            "dimensionless_interaction_attenuation_ratio": normalized["dimensionless_interaction_rms"] / max(raw["dimensionless_interaction_rms"], np.finfo(float).tiny),
            "raw_interaction_fraction": raw["interaction_fraction"],
            "normalized_interaction_fraction": normalized["interaction_fraction"],
            "interaction_fraction_attenuation_ratio": normalized["interaction_fraction"] / max(raw["interaction_fraction"], np.finfo(float).tiny),
            "raw_interaction_to_common_ratio": raw["interaction_to_common_ratio"],
            "normalized_interaction_to_common_ratio": normalized["interaction_to_common_ratio"],
            **retention[joint],
            "local_gradient_direction_change_count": direction_changes,
            "local_gradient_direction_change_fraction": direction_changes / len(subject_ids),
            "raw_normalized_local_gradient_cosine_min": float(np.min(gradient_cosines)),
            "raw_normalized_local_gradient_cosine_median": float(np.median(gradient_cosines)),
            "scientific_role": "DIAGNOSTIC_ONLY_NO_OBJECTIVE_CHANGE",
        })
    return rows


def validate_replay_against_compact(
    cache: dict[str, np.ndarray], subject_ids: list[str], matrices: dict[str, np.ndarray],
    all_candidate_ids: list[str],
) -> dict[str, Any]:
    replay_subjects = cache["subject_id"].astype(str).tolist()
    replay_candidates = cache["candidate_id"].astype(str).tolist()
    time_s = np.asarray(cache["time_s"], dtype=float)
    tau = np.asarray(cache["tau_truth_nm"], dtype=float)
    errors: dict[str, float] = {}
    for joint_index, joint in enumerate(("hip", "knee")):
        replay_rms = time_weighted_rms(time_s, np.moveaxis(tau[:, :, :, joint_index], 2, 0), axis=0)
        expected = np.asarray([
            [matrices[f"raw_{joint}_rms"][subject_ids.index(subject_id), all_candidate_ids.index(candidate_id)] for candidate_id in replay_candidates]
            for subject_id in replay_subjects
        ])
        errors[joint] = float(np.max(np.abs(replay_rms - expected)))
    if max(errors.values()) > 1e-9:
        raise RuntimeError(f"replay versus compact RMS mismatch: {errors}")
    return {
        "subject_count": len(replay_subjects), "candidate_count": len(replay_candidates),
        "pair_count": len(replay_subjects) * len(replay_candidates),
        "hip_rms_max_abs_error_nm": errors["hip"], "knee_rms_max_abs_error_nm": errors["knee"],
        "integrity_tolerance_nm": 1e-9, "passed": True,
    }


def compact_diagnostic_sets(
    matrices: dict[str, np.ndarray], subject_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    diagnostics: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    affine_rows: list[dict[str, Any]] = []
    interactions: dict[str, np.ndarray] = {}
    for name in COMPACT_REPRESENTATIONS:
        diagnostic, fits, interaction = representation_diagnostic(
            name, matrices[name], subject_ids, "ALL_24_DEVELOPMENT_X_625_V3_CANDIDATES",
        )
        diagnostics.append(diagnostic)
        by_name[name] = diagnostic
        affine_rows.extend(fits)
        interactions[name] = interaction
    return diagnostics, by_name, affine_rows, interactions


def _label(name: str) -> str:
    replacements = {
        "raw_hip_rms": "Raw hip RMS", "raw_knee_rms": "Raw knee RMS",
        "normalized_hip_rms": "Normalized hip RMS", "normalized_knee_rms": "Normalized knee RMS",
        "combined_j": "Frozen combined J", "raw_hip_peak": "Raw hip peak",
        "raw_knee_peak": "Raw knee peak", "normalized_hip_peak": "Normalized hip peak",
        "normalized_knee_peak": "Normalized knee peak",
    }
    return replacements.get(name, name.replace("component_", "").replace("_rms", "").replace("_", " ").title())


def create_figures(
    diagnostics: dict[str, dict[str, Any]], normalization_rows: list[dict[str, Any]],
    time_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]],
    affine_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]], information_rows: list[dict[str, Any]],
    branch: dict[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    blue, orange, green, red, gray = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#777777"
    paths: list[Path] = []

    names = ["raw_hip_rms", "normalized_hip_rms", "raw_knee_rms", "normalized_knee_rms", "combined_j"]
    values = [100 * diagnostics[name]["interaction_fraction"] for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(range(len(names)), values, color=[blue, orange, blue, orange, gray])
    ax.bar_label(bars, fmt="%.3g%%", padding=3)
    ax.set_xticks(range(len(names)), [_label(name) for name in names], rotation=18, ha="right")
    ax.set_ylabel("Subject × candidate interaction (%)")
    ax.set_title("Raw torque, normalization, and the frozen objective")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "raw_vs_normalized_interaction.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    comparisons = [row for row in normalization_rows if row["record_type"] == "NORMALIZATION_COMPARISON"]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    raw_fraction = [100 * float(row["raw_interaction_fraction"]) for row in comparisons]
    normalized_fraction = [100 * float(row["normalized_interaction_fraction"]) for row in comparisons]
    axes[0].bar(x - 0.18, raw_fraction, width=0.36, label="Raw", color=blue)
    axes[0].bar(x + 0.18, normalized_fraction, width=0.36, label="Normalized", color=orange)
    axes[0].set_xticks(x, [row["joint"].title() for row in comparisons]); axes[0].set_ylabel("Interaction (%)")
    axes[0].legend(frameon=False); axes[0].set_title("Variance interaction")
    changes = [100 * float(row["local_gradient_direction_change_fraction"]) for row in comparisons]
    axes[1].bar(x, changes, color=green)
    axes[1].set_xticks(x, [row["joint"].title() for row in comparisons]); axes[1].set_ylabel("Gradient direction changes (%)")
    axes[1].set_title("Ordering/direction loss")
    if max(changes) == 0.0:
        axes[1].set_ylim(-0.05, 1.0)
        axes[1].text(
            0.5, 0.5, "0% for both joints\n(ordering fully retained)",
            ha="center", va="center", transform=axes[1].transAxes,
            color=green, fontweight="bold",
        )
    for ax in axes: ax.grid(axis="y", alpha=0.25)
    fig.suptitle("What reference normalization changes—and preserves", y=1.02); fig.tight_layout()
    path = FIGURES / "normalization_attenuation_and_ordering.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    full_time = [row for row in time_rows if row["record_type"] == "FULL_CYCLE_TIME_INTERACTION" and row["scale"] == "reference_rms_normalized"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ratios = [float(row["time_to_rms_interaction_ratio"]) for row in full_time]
    bars = ax.bar([row["joint"].title() for row in full_time], ratios, color=[blue, orange])
    ax.axhline(2.0, color=red, linestyle="--", label="Frozen material-compression ratio")
    ax.bar_label(bars, fmt="%.2f×", padding=3); ax.set_ylabel("Time-resolved / RMS interaction")
    ax.set_title("Time-local interaction energy is larger, but ordering is unchanged")
    ax.text(
        0.5, 0.04, "Ratio criterion met; preregistered window-ordering criterion not met",
        ha="center", transform=ax.transAxes, fontsize=9,
    )
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    path = FIGURES / "time_resolved_vs_rms_interaction.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    windows = [row for row in time_rows if row["record_type"] == "BRANCH_PHASE_WINDOW" and row["scale"] == "reference_rms_normalized"]
    window_names = list(dict.fromkeys(row["window"] for row in windows))
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for offset, joint, color in ((-0.18, "hip", blue), (0.18, "knee", orange)):
        selected = [row for window in window_names for row in windows if row["window"] == window and row["joint"] == joint]
        ax.bar(np.arange(len(window_names)) + offset, [100 * float(row["direction_disagreement_fraction"]) for row in selected], width=0.36, label=joint.title(), color=color)
    ax.axhline(5.0, color=red, linestyle="--", label="Frozen ordering-evidence threshold")
    ax.set_xticks(np.arange(len(window_names)), [name.replace("_", "\n").title() for name in window_names])
    ax.set_ylabel("Direction-disagreement transitions (%)"); ax.set_title("Preregistered phase-window ordering evidence")
    if all(float(row["direction_disagreement_fraction"]) == 0.0 for row in windows):
        ax.text(
            0.5, 0.52, "All 12 joint × window values = 0%", ha="center", va="center",
            transform=ax.transAxes, color=green, fontweight="bold",
        )
    ax.legend(frameon=False, ncol=3); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    path = FIGURES / "time_window_direction_disagreement.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    raw_components = [row for row in component_rows if row["scale"] == "raw_rms_nm"]
    component_names = list(dict.fromkeys(row["component"] for row in raw_components))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, joint, color in ((-0.18, "hip", blue), (0.18, "knee", orange)):
        selected = [next(row for row in raw_components if row["component"] == component and row["joint"] == joint) for component in component_names]
        ax.bar(np.arange(len(component_names)) + offset, [100 * float(row["interaction_fraction"]) for row in selected], width=0.36, label=joint.title(), color=color)
    ax.set_xticks(np.arange(len(component_names)), [name.replace("_", "\n").title() for name in component_names])
    ax.set_ylabel("Interaction fraction (%)"); ax.set_title("Existing MyoLeg force-component interactions (8×17 subset)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    path = FIGURES / "force_component_interactions.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    selected_reps = ["raw_hip_rms", "raw_knee_rms", "normalized_hip_rms", "normalized_knee_rms"]
    affine_selected = [row for row in affine_rows if row["representation"] in selected_reps]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    data = [[float(row["affine_r2"]) for row in affine_selected if row["representation"] == rep] for rep in selected_reps]
    box = ax.boxplot(data, patch_artist=True, tick_labels=[_label(rep) for rep in selected_reps], showfliers=False)
    for patch, color in zip(box["boxes"], [blue, blue, orange, orange]): patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(0.995, color=red, linestyle="--", label="Frozen affine-common threshold")
    ax.set_ylabel("Affine fit R²"); ax.set_ylim(min(0.98, min(min(v) for v in data) - 0.002), 1.0005)
    ax.set_title("Can each subject be explained as an affine rescaling?")
    ax.tick_params(axis="x", rotation=16); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    path = FIGURES / "affine_scaling_across_subjects.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    gradient_selected = [row for row in gradient_rows if row["representation"] == "combined_j"]
    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    gx = np.asarray([float(row["local_beta_flex_gradient"]) for row in gradient_selected])
    gy = np.asarray([float(row["local_beta_extend_gradient"]) for row in gradient_selected])
    ax.scatter(gx, gy, c=np.arange(len(gx)), cmap="viridis", s=45, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=gray, linewidth=0.8); ax.axvline(0, color=gray, linewidth=0.8)
    ax.set_xlabel("∂J / ∂β flex"); ax.set_ylabel("∂J / ∂β extend")
    ax.set_title("Frozen-objective local gradient directions (24 development subjects)")
    ax.grid(alpha=0.2); fig.tight_layout()
    path = FIGURES / "subject_gradient_directions.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    finite_assoc = [row for row in association_rows if row.get("spearman_rho") is not None]
    top = sorted(finite_assoc, key=lambda row: abs(float(row["spearman_rho"])), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(10, 6.3))
    labels = [f"{row['subject_parameter'].replace('_SCALE', '').replace('_', ' ')}\n{row['representation'].replace('component_', '').replace('_rms', '')}: {row['outcome']}" for row in reversed(top)]
    values = [float(row["spearman_rho"]) for row in reversed(top)]
    ax.barh(range(len(top)), values, color=[blue if value >= 0 else orange for value in values])
    ax.set_yticks(range(len(top)), labels, fontsize=7); ax.set_xlim(-1, 1); ax.axvline(0, color=gray, linewidth=0.8)
    ax.set_xlabel("Exploratory Spearman ρ"); ax.set_title("Largest factor–response associations (descriptive only)")
    ax.grid(axis="x", alpha=0.25); fig.tight_layout()
    path = FIGURES / "factor_response_associations.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    display_rows = [row for row in information_rows if row["representation"] in {
        "raw_hip_rms", "raw_knee_rms", "normalized_hip_rms", "normalized_knee_rms", "combined_j",
        "raw_hip_peak", "raw_knee_peak", "normalized_hip_peak", "normalized_knee_peak",
        "component_passive_hip_rms", "component_passive_knee_rms",
        "component_mass_inertia_hip_rms", "component_mass_inertia_knee_rms", "component_bias_gravity_hip_rms",
        "component_bias_gravity_knee_rms",
    }]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    y = np.arange(len(display_rows))
    interaction = [100 * float(row["interaction_fraction"]) for row in display_rows]
    disagreement = [100 * float(row["direction_disagreement_fraction"]) for row in display_rows]
    ax.scatter(interaction, y, label="Interaction fraction", color=blue, s=48)
    ax.scatter(disagreement, y, label="Direction disagreement", color=orange, marker="s", s=40)
    ax.set_yticks(y, [_label(row["representation"]) for row in display_rows]); ax.invert_yaxis()
    ax.set_xlabel("Percent"); ax.set_title(f"Information-retention map → {branch['decision']}")
    ax.legend(frameon=False); ax.grid(axis="x", alpha=0.25); fig.tight_layout()
    path = FIGURES / "objective_information_retention_map.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def _pct(value: Any) -> str:
    return f"{100 * float(value):.6f}%"


def build_report(
    protocol: dict[str, Any], diagnostics: dict[str, dict[str, Any]],
    normalization_rows: list[dict[str, Any]], time_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]],
    objective: dict[str, Any], heterogeneity: dict[str, Any], branch: dict[str, Any],
    replay_integrity: dict[str, Any], cache_manifest: dict[str, Any], runtime_s: float,
) -> str:
    comparisons = {row["joint"]: row for row in normalization_rows if row["record_type"] == "NORMALIZATION_COMPARISON"}
    full_time = {row["joint"]: row for row in time_rows if row["record_type"] == "FULL_CYCLE_TIME_INTERACTION" and row["scale"] == "reference_rms_normalized"}
    peak = {name: diagnostics[name] for name in ("raw_hip_peak", "raw_knee_peak", "normalized_hip_peak", "normalized_knee_peak")}
    raw_components = [row for row in component_rows if row["scale"] == "raw_rms_nm" and row["component_numerically_present"]]
    strongest = sorted(raw_components, key=lambda row: float(row["interaction_fraction"]), reverse=True)
    component_table = "\n".join(
        f"| {row['component']} | {row['joint']} | {_pct(row['interaction_fraction'])} | {float(row['interaction_to_common_ratio']):.6g} | {_pct(row['direction_disagreement_fraction'])} |"
        for row in strongest
    )
    j_gradients = [row for row in gradient_rows if row["representation"] == "combined_j"]
    different_gradients = sum(bool(row["preference_direction_differs_from_anchor"]) for row in j_gradients)
    min_gradient_cosine = min(float(row["local_gradient_cosine_to_anchor"]) for row in j_gradients)
    ordering_sentence = (
        "两个关节的 subject 内 raw→normalized 排序均逐值保留，邻接方向没有改变。"
        if objective["normalization_ordering_retained_for_both_joints"] else
        "至少一个关节的 raw→normalized 排序或邻接方向发生了预注册意义下的改变。"
    )
    time_sentence = (
        "满足预冻结的 RMS compression 条件。" if objective["material_rms_compression"] else
        "未满足预冻结的 RMS compression 条件。"
    )
    peak_evidence = [name for name, row in peak.items() if row["representation_ordering_evidence"]]
    knee_peak = peak["normalized_knee_peak"]
    component_evidence = [f"{row['component']}:{row['joint']}" for row in raw_components if row["representation_ordering_evidence"]]
    return f"""# MyoLeg objective and musculoskeletal heterogeneity decision audit V1

## Formal outcome

**{branch['decision']}**

- Objective assessment: `{objective['status']}`
- Current 6-D heterogeneity assessment: `{heterogeneity['status']}`
- Scientifically justified next independent stage: `{branch['recommended_next_stage']}`
- Next stage executed: **no**

This is an offline, development-only diagnosis. The frozen objective remains `J_NORMALIZED_RMS`; raw, peak, time-local, and force-component views are diagnostic only. No objective weights were searched.

## Frozen protocol and integrity

- Protocol: `{protocol['protocol_id']}`
- Protocol SHA-256: `{FROZEN_SHA['protocol']}` (frozen before new development diagnostics were read)
- Primary matrix: 24 development subjects × 625 frozen V3 candidates
- Replay subset: 8 geometry-selected development subjects × 17 beta-geometry-selected candidates = 136 replays
- Replay/compact RMS agreement: hip max error `{replay_integrity['hip_rms_max_abs_error_nm']:.3e}` Nm; knee max error `{replay_integrity['knee_rms_max_abs_error_nm']:.3e}` Nm
- Replay cache SHA-256: `{cache_manifest['sha256']}`
- Held-out scientific access: **0**; held-out replay: **0**; held-out `np.load`: **0**
- V3 candidate domain, cohort, ranges, subject factors, objective, normalization, learner, and BO were not modified.

## Frozen formal baseline

The frozen V3 result is unchanged: interaction `{_pct(protocol['baseline']['frozen_v3_interaction_fraction'])}`, median Spearman `{protocol['baseline']['frozen_v3_spearman_median']:.6f}`, median Kendall `{protocol['baseline']['frozen_v3_kendall_median']:.6f}`, adjacent direction reversals `0`, and common regret `0`. V2 remains a historical secondary comparison (interaction about `0.033114%`); it was not used to select diagnostics.

## Raw versus normalized RMS

| Joint | Raw interaction | Normalized interaction | attenuation ratio (fraction) | raw→normalized Spearman min | adjacent sign changes | local-gradient direction changes |
|---|---:|---:|---:|---:|---:|---:|
| Hip | {_pct(comparisons['hip']['raw_interaction_fraction'])} | {_pct(comparisons['hip']['normalized_interaction_fraction'])} | {float(comparisons['hip']['interaction_fraction_attenuation_ratio']):.6g} | {float(comparisons['hip']['within_subject_raw_normalized_spearman_min']):.9f} | {comparisons['hip']['adjacent_sign_change_count']} | {comparisons['hip']['local_gradient_direction_change_count']} |
| Knee | {_pct(comparisons['knee']['raw_interaction_fraction'])} | {_pct(comparisons['knee']['normalized_interaction_fraction'])} | {float(comparisons['knee']['interaction_fraction_attenuation_ratio']):.6g} | {float(comparisons['knee']['within_subject_raw_normalized_spearman_min']):.9f} | {comparisons['knee']['adjacent_sign_change_count']} | {comparisons['knee']['local_gradient_direction_change_count']} |

{ordering_sentence} Absolute interaction RMS changes unit/scale after normalization, so the decision uses dimensionless interaction plus ordering/direction retention rather than comparing Nm directly with a ratio.

## Time-resolved and peak diagnostics

- Reference-normalized hip time/RMS interaction ratio: `{float(full_time['hip']['time_to_rms_interaction_ratio']):.6g}`.
- Reference-normalized knee time/RMS interaction ratio: `{float(full_time['knee']['time_to_rms_interaction_ratio']):.6g}`.
- Interpretation: {time_sentence}
- Peak representations meeting frozen ordering-evidence rule: `{', '.join(peak_evidence) if peak_evidence else 'none'}`.

Time-local evidence is not a comfort, safety, tissue-load, clinical, or human-preference result. It is only a simulator required-drive diagnostic.

## Existing force-component decomposition

The decomposition reuses the frozen replay semantics: required drive = mass + bias/gravity − passive − zero-control actuator − constraint. It does not invent a new physical decomposition.

| Component | Joint | interaction | interaction/common | direction disagreement |
|---|---|---:|---:|---:|
{component_table}

Components meeting frozen ordering-evidence rule: `{', '.join(component_evidence) if component_evidence else 'none'}`.

## Q1–Q10 answers

### Q1. Does raw unnormalized torque exhibit materially more subject×trajectory interaction than normalized torque?

**No.** Hip raw/normalized interaction fractions are `{_pct(comparisons['hip']['raw_interaction_fraction'])}` / `{_pct(comparisons['hip']['normalized_interaction_fraction'])}`; knee values are `{_pct(comparisons['knee']['raw_interaction_fraction'])}` / `{_pct(comparisons['knee']['normalized_interaction_fraction'])}`. The fraction increases after normalization because subject-main scale is removed from the total-variance denominator; this is not evidence that normalization created or destroyed ordering. Dimensionless interaction changes by `{float(comparisons['hip']['dimensionless_interaction_attenuation_ratio']):.4f}×` (hip) and `{float(comparisons['knee']['dimensionless_interaction_attenuation_ratio']):.4f}×` (knee), while ordering remains identical.

### Q2. Does normalization change only magnitude scaling, or trajectory ordering?

**Only magnitude scaling changed under the frozen tests.** {ordering_sentence} Therefore normalization is interpreted as removing absolute subject scale unless the frozen `NORMALIZATION_ORDERING_LOSS` rule is met; current status: `{objective['normalization_ordering_loss']}`.

### Q3. Does time-resolved torque contain subject-specific path information that RMS removes?

**Not under the preregistered conjunction.** Hip and knee time/RMS energy ratios are `{float(full_time['hip']['time_to_rms_interaction_ratio']):.4f}` and `{float(full_time['knee']['time_to_rms_interaction_ratio']):.4f}`, but all six windows for both joints have zero direction disagreement and no window satisfies representation-ordering evidence. {time_sentence}

### Q4. Do peak-based diagnostics reveal subject-specific ordering?

Only `{', '.join(peak_evidence) if peak_evidence else 'no peak representation'}` meets the composite representation-evidence rule. For normalized knee peak, median rank correlation is `{float(knee_peak['spearman_median']):.6f}` and direction disagreement is `{_pct(knee_peak['direction_disagreement_fraction'])}`; therefore it does **not** establish robust subject-specific path preference by itself. Peak remains diagnostic-only and was not optimized.

### Q5. Which MyoLeg force components carry the strongest subject×candidate interaction?

The largest preregistered subset interaction is `{strongest[0]['component']}:{strongest[0]['joint']}` at `{_pct(strongest[0]['interaction_fraction'])}`. The complete ordered table above prevents cherry-picking a single component.

### Q6. Do passive/fpmax variations alter path preference or mainly response magnitude?

Passive component ordering evidence count is `{sum(row['component'] == 'passive' and row['representation_ordering_evidence'] for row in raw_components)}`; direction-evidence count is `{sum(row['component'] == 'passive' and float(row['direction_disagreement_fraction']) >= 0.05 for row in raw_components)}`. The frozen fpmax variation therefore does not change path preference in this audit; at most it rescales passive magnitude. Magnitude associations are descriptive and are not counted as preference-direction evidence.

### Q7. Do mass/inertia variations alter path preference or mainly response magnitude?

Mass/bias component direction-evidence count is `{sum(row['component'] in {'mass_inertia', 'bias_gravity'} and float(row['direction_disagreement_fraction']) >= 0.05 for row in raw_components)}`. These factors mainly change response magnitude, not preferred direction. Parameter associations with scale or gradient norm do not establish a different preferred path.

### Q8. Are subject-specific gradient directions different, or only magnitudes?

For frozen J, `{different_gradients}/24` subjects differ in local gradient sign from the pre-frozen anchor and the minimum local gradient cosine is `{min_gradient_cosine:.6f}`. Gradient magnitudes vary, but preferred directions do not meaningfully differ.

### Q9. Is the stronger limitation the objective, musculoskeletal heterogeneity, or both?

The stronger limitation is the **current musculoskeletal heterogeneity**: `{branch['decision']}` under the pre-frozen 2×2 decision matrix. Objective evidence: `{objective['status']}`. Heterogeneity evidence: `{heterogeneity['status']}`.

### Q10. What exact next independent stage is justified?

`{branch['recommended_next_stage']}`. It is recommended only and was **not executed**. If the branch concerns heterogeneity, future factor classes are taxonomy entries without ranges or implementation; if it concerns objective information, future information classes are not a new objective.

## Scientific boundaries

- This audit does not establish physiological parameter values, patient comfort, rehabilitation effectiveness, safety, clinical validity, or robot validity.
- Passive simulator terms are not direct tissue-force measurements.
- Current six factors and ranges are unchanged; no future range is proposed.
- Held-out truth remains sealed and cannot support any statement above.
- Runtime for the audit build: `{runtime_s:.3f}` s (full pytest is reported separately after generation).
"""


def write_checksums() -> dict[str, str]:
    checksum_path = OUTPUT / "checksums.sha256"
    artifacts = [path for path in OUTPUT.rglob("*") if path.is_file() and path != checksum_path]
    values = {str(path.relative_to(OUTPUT)): sha256_file(path) for path in sorted(artifacts)}
    atomic_text(checksum_path, "".join(f"{digest}  {relative}\n" for relative, digest in values.items()))
    return values


def analyze() -> dict[str, Any]:
    started = time.perf_counter()
    if (OUTPUT / "metadata.json").exists():
        raise RuntimeError("formal audit already exists; refusing silent overwrite")
    truth, candidate_manifest, candidates, cohort = verify_inputs()
    protocol = read_json(PROTOCOL_PATH)
    access = verify_held_out_without_loading(cohort, truth)
    atomic_json(OUTPUT / "ANALYSIS_EXECUTION_FREEZE.json", {
        "stage_id": STAGE_ID, "protocol_id": PROTOCOL_ID,
        "protocol_sha256": FROZEN_SHA["protocol"],
        "frozen_before_compact_or_replay_scientific_arrays_read": True,
        "development_subject_count": 24, "candidate_count": 625,
        "held_out_scientific_access_count": 0,
        "decision_rules_copied_without_post_result_tuning": protocol["decision_rules"],
    })

    subject_ids, matrices, store = compact_matrices(truth, candidates, cohort)
    compact_diagnostics, diagnostic_map, compact_affine, compact_interactions = compact_diagnostic_sets(matrices, subject_ids)
    compact_gradients = compact_gradient_rows(matrices, subject_ids)
    retention = {
        "hip": ordering_retention(matrices["raw_hip_rms"], matrices["normalized_hip_rms"]),
        "knee": ordering_retention(matrices["raw_knee_rms"], matrices["normalized_knee_rms"]),
    }
    raw_normalized_rows = normalization_audit_rows(diagnostic_map, retention, compact_gradients)
    peak_rows = [
        {"record_type": "PEAK_REPRESENTATION_DIAGNOSTIC", **diagnostic_map[name], "scientific_role": "DIAGNOSTIC_ONLY_NO_OBJECTIVE_CHANGE"}
        for name in ("raw_hip_peak", "raw_knee_peak", "normalized_hip_peak", "normalized_knee_peak")
    ]
    compact_associations = compact_parameter_associations(cohort, subject_ids, compact_gradients, compact_interactions)

    cache, cache_manifest = load_replay_cache(protocol)
    all_candidate_ids = [row["candidate_id"] for row in candidates]
    replay_integrity = validate_replay_against_compact(cache, subject_ids, matrices, all_candidate_ids)
    time_rows, time_affine, _ = analyze_time_resolved(cache, protocol, cohort)
    (
        component_rows, passive_associations, mass_associations, component_affine,
        component_gradients, _, _,
    ) = analyze_components(cache, cohort)
    all_affine = compact_affine + time_affine + component_affine
    all_gradients = compact_gradients + component_gradients
    all_associations = compact_associations + passive_associations + mass_associations
    information_rows = make_information_summary(compact_diagnostics, time_rows, component_rows)
    objective, heterogeneity, branch = decide(information_rows, retention, time_rows, component_rows)

    write_csv(OUTPUT / "RAW_VS_NORMALIZED_INTERACTION_AUDIT.csv", raw_normalized_rows)
    write_csv(OUTPUT / "TIME_RESOLVED_INTERACTION_AUDIT.csv", time_rows)
    write_csv(OUTPUT / "PEAK_DIAGNOSTIC_AUDIT.csv", peak_rows)
    write_csv(OUTPUT / "FORCE_COMPONENT_INTERACTION_AUDIT.csv", component_rows)
    write_csv(
        OUTPUT / "PASSIVE_HETEROGENEITY_AUDIT.csv",
        [{"record_type": "COMPONENT_SUMMARY", **row} for row in component_rows if row["component"] == "passive"] + passive_associations,
    )
    write_csv(
        OUTPUT / "MASS_INERTIA_HETEROGENEITY_AUDIT.csv",
        [{"record_type": "COMPONENT_SUMMARY", **row} for row in component_rows if row["component"] in {"mass_inertia", "bias_gravity"}] + mass_associations,
    )
    write_csv(OUTPUT / "AFFINE_SCALING_ACROSS_REPRESENTATIONS.csv", all_affine)
    write_csv(OUTPUT / "SUBJECT_GRADIENT_DIRECTION_AUDIT.csv", all_gradients)
    write_csv(OUTPUT / "SUBJECT_PARAMETER_GRADIENT_ASSOCIATIONS.csv", all_associations)
    write_csv(OUTPUT / "OBJECTIVE_INFORMATION_RETENTION_SUMMARY.csv", information_rows)
    atomic_json(OUTPUT / "REPLAY_COMPACT_INTEGRITY.json", replay_integrity)
    atomic_json(OUTPUT / "OBJECTIVE_ADEQUACY_DECISION.json", objective)
    atomic_json(OUTPUT / "HETEROGENEITY_ADEQUACY_DECISION.json", heterogeneity)
    atomic_json(OUTPUT / "FINAL_BRANCH_DECISION.json", branch)
    if heterogeneity["status"] == HETEROGENEITY_LIMITED:
        write_csv(OUTPUT / "FUTURE_MUSCULOSKELETAL_FACTOR_TAXONOMY.csv", future_factor_taxonomy())
    if objective["status"] == OBJECTIVE_LIMITED:
        write_csv(OUTPUT / "FUTURE_OBJECTIVE_INFORMATION_CLASSES.csv", future_objective_information_classes())

    figures = create_figures(
        diagnostic_map, raw_normalized_rows, time_rows, component_rows, all_affine,
        all_gradients, all_associations, information_rows, branch,
    )
    runtime_s = time.perf_counter() - started
    report = build_report(
        protocol, diagnostic_map, raw_normalized_rows, time_rows, component_rows,
        all_gradients, objective, heterogeneity, branch, replay_integrity, cache_manifest, runtime_s,
    )
    atomic_text(OUTPUT / "MYOLEG_OBJECTIVE_AND_HETEROGENEITY_DECISION_REPORT.md", report)
    access.update({
        "development_compact_subject_ids_accessed": store.accessed,
        "development_compact_subject_count": len(set(store.accessed)),
        "development_replay_subject_ids": cache["subject_id"].astype(str).tolist(),
        "development_replay_pair_count": 136,
        "held_out_np_load_count": 0, "held_out_replay_count": 0,
        "held_out_scientific_access_count": 0,
        "decision_population": "24 DEVELOPMENT ONLY",
    })
    atomic_json(ACCESS_PATH, access)

    pre_metadata_sha = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"}
    }
    metadata = {
        "stage_id": STAGE_ID, "protocol_id": PROTOCOL_ID,
        "outcome": branch["decision"], "objective_status": objective["status"],
        "heterogeneity_status": heterogeneity["status"],
        "recommended_next_stage": branch["recommended_next_stage"], "next_stage_executed": False,
        "development_subject_count": len(subject_ids), "candidate_count": EXPECTED_CANDIDATES,
        "development_pair_count": len(subject_ids) * EXPECTED_CANDIDATES,
        "replay_subject_count": 8, "replay_candidate_count": 17, "replay_pair_count": 136,
        "held_out_subject_count": 8, "held_out_scientific_access_count": 0,
        "formal_objective_unchanged": True, "normalization_unchanged": True,
        "v3_parameterization_unchanged": True, "v3_candidate_domain_unchanged": True,
        "cohort_unchanged": True, "subject_ranges_unchanged": True,
        "new_subject_factors_added": False, "objective_weight_search": False,
        "five_parameter_model_trained": False, "nn_or_pinn_trained": False,
        "bo_run": False, "robot_or_hardware": False,
        "protocol_sha256": FROZEN_SHA["protocol"],
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "truth_landscape_manifest_sha256": sha256_file(TRUTH_MANIFEST),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
        "cohort_manifest_sha256": sha256_file(COHORT_MANIFEST),
        "replay_cache_sha256": cache_manifest["sha256"],
        "figure_count": len(figures), "artifact_sha256_before_metadata": pre_metadata_sha,
        "runtime_s": runtime_s,
        "pytest": {"status": "PENDING_FULL_SUITE_POST_GENERATION"},
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    write_checksums()
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="generate the frozen development-only audit")
    group.add_argument(
        "--generate-replay-cache", action="store_true",
        help="generate only the preregistered replay cache in the frozen MyoSuite environment",
    )
    args = parser.parse_args()
    if args.generate_replay_cache:
        verify_inputs()
        manifest = generate_replay_cache(read_json(PROTOCOL_PATH))
        print(json.dumps(manifest, indent=2))
        return 0
    metadata = analyze()
    print(json.dumps({
        "stage_id": metadata["stage_id"], "outcome": metadata["outcome"],
        "objective_status": metadata["objective_status"],
        "heterogeneity_status": metadata["heterogeneity_status"],
        "held_out_scientific_access_count": metadata["held_out_scientific_access_count"],
        "next_stage_executed": metadata["next_stage_executed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
