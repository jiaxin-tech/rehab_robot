"""Development-only MyoLeg-V3 personalization-necessity audit.

The analysis protocol is an immutable input to this program.  This module never
loads a held-out NPZ and its truth store rejects non-development identifiers
before resolving a shard path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


STAGE_ID = "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_V1"
PROTOCOL_ID = "MYOLEG_V3_PERSONALIZATION_NECESSITY_PROTOCOL_V1"
OUTCOME_SUPPORTED = "V3_PERSONALIZATION_NECESSITY_SUPPORTED"
OUTCOME_WEAK = "V3_PERSONALIZATION_NECESSITY_WEAK_OR_LIMITED"
OUTCOME_NOT_SUPPORTED = "V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v3_personalization_necessity_audit_v1"
FIGURES = OUTPUT / "figures"
TRUTH_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1_MANIFEST.json"
CANDIDATE_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
CANDIDATE_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
V2_PROTOCOL = ROOT / "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1/PERSONALIZATION_NECESSITY_PROTOCOL.json"
V2_NECESSITY = ROOT / "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1"
V2_ROOT_CAUSE = ROOT / "external_simulation_audits/myoleg_v2_personalization_signal_root_cause_audit_v1"
PROTOCOL_PATH = OUTPUT / "V3_PERSONALIZATION_NECESSITY_PROTOCOL.json"
HELD_OUT_AUDIT_PATH = OUTPUT / "HELD_OUT_ACCESS_AUDIT.json"

FROZEN_SHA = {
    "truth": "c318700e161e857d3e059eadf2bc21364e21b74d8b39faf4124a26fc15d37c6e",
    "candidate_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "candidate_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "cohort": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "v2_protocol": "f26663a71960c2f5cedb3d374cced98b0852fbfd718fe8235f0e1d9e6d102e6f",
    "protocol": "d21506672ed006e5015bb92ad8ec50dce15ea2762b1b421083664ae6321b3eb3",
}
HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)
EXPECTED_SUBJECTS = 24
EXPECTED_CANDIDATES = 625
REFERENCE_INDEX = 312
REFERENCE_ID = "MYOLEG_V3_K0312"
GRID_SIZE = 25
GRID_STEP = 0.0025
BETA_RANGE = np.asarray([0.06, 0.06])
TIE_TOLERANCE = 1.0e-12
EQUIVALENCE_TOLERANCE = 1.0e-12
NEAR_EPSILONS = (1.0e-4, 5.0e-4, 1.0e-3)
TOP_FRACTIONS = (0.01, 0.05, 0.10)
TOP_COUNTS = (7, 32, 63)
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260830


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
        raise RuntimeError(f"cannot infer empty CSV schema: {path}")
    columns = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return {
        "min": float(np.min(array)), "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)), "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
    }


def select_tied_min(values: np.ndarray) -> int:
    minimum = float(np.min(values))
    eligible = np.flatnonzero(values <= minimum + TIE_TOLERANCE)
    return int(np.min(eligible))


def sign_class(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(array < -EQUIVALENCE_TOLERANCE, -1, np.where(array > EQUIVALENCE_TOLERANCE, 1, 0))


def effect_band(relative: float) -> str:
    if relative < 0.001:
        return "NEGLIGIBLE"
    if relative < 0.005:
        return "VERY_SMALL"
    if relative < 0.01:
        return "SMALL"
    if relative < 0.02:
        return "MODERATE_ENGINEERING_SEPARATION"
    return "LARGER_ENGINEERING_SEPARATION"


def bootstrap_intervals(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))]
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_95_percentile_ci": np.percentile(np.mean(samples, axis=1), [2.5, 97.5]).tolist(),
        "median_95_percentile_ci": np.percentile(np.median(samples, axis=1), [2.5, 97.5]).tolist(),
    }


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
    public = {
        "shape": list(values.shape), "grand_mean": grand,
        "ss_total": ss_total, "ss_subject_main": ss_subject,
        "ss_candidate_main": ss_candidate,
        "ss_subject_candidate_interaction": ss_interaction,
        "subject_main_variance_fraction": ss_subject / denominator,
        "candidate_main_variance_fraction": ss_candidate / denominator,
        "subject_candidate_interaction_variance_fraction": ss_interaction / denominator,
        "common_candidate_effect_rms": common_rms,
        "subject_specific_interaction_rms": interaction_rms,
        "interaction_to_common_effect_ratio": interaction_rms / max(common_rms, np.finfo(float).tiny),
        "dimensionless_interaction_scale": interaction_rms / max(abs(grand), np.finfo(float).tiny),
        "subject_interaction_residual_rms": np.sqrt(np.mean(interaction**2, axis=1)).tolist(),
        "centering_convention": "J_ic=grand+subject_main_i+candidate_main_c+interaction_ic",
    }
    return public, interaction


def fit_scaling(target: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    x = np.asarray(anchor, dtype=float)
    y = np.asarray(target, dtype=float)
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    affine_prediction = slope * x + intercept
    sst = float(np.sum((y - np.mean(y)) ** 2))
    proportional_slope = float(np.dot(x, y) / np.dot(x, x))
    proportional_prediction = proportional_slope * x
    scale = max(float(np.mean(np.abs(y))), np.finfo(float).tiny)
    return {
        "affine_slope": float(slope), "affine_intercept": float(intercept),
        "affine_r2": 1.0 - float(np.sum((y - affine_prediction) ** 2)) / max(sst, np.finfo(float).tiny),
        "affine_nrmse": float(np.sqrt(np.mean((y - affine_prediction) ** 2)) / scale),
        "proportional_slope": proportional_slope,
        "proportional_r2": 1.0 - float(np.sum((y - proportional_prediction) ** 2)) / max(sst, np.finfo(float).tiny),
        "proportional_nrmse": float(np.sqrt(np.mean((y - proportional_prediction) ** 2)) / scale),
    }


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    actual = {
        "truth": sha256_file(TRUTH_MANIFEST),
        "candidate_manifest": sha256_file(CANDIDATE_MANIFEST),
        "candidate_table": sha256_file(CANDIDATE_TABLE),
        "cohort": sha256_file(COHORT_MANIFEST),
        "v2_protocol": sha256_file(V2_PROTOCOL),
        "protocol": sha256_file(PROTOCOL_PATH),
    }
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    protocol = read_json(PROTOCOL_PATH)
    truth = read_json(TRUTH_MANIFEST)
    candidate_manifest = read_json(CANDIDATE_MANIFEST)
    cohort = read_json(COHORT_MANIFEST)
    candidates = read_csv(CANDIDATE_TABLE)
    dev_ids = tuple(truth["development_subject_ids"])
    if not (
        protocol["protocol_id"] == PROTOCOL_ID
        and protocol["frozen_before_development_truth_reveal"] is True
        and truth["outcome"] == "MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_VALID"
        and truth["actual_pair_count"] == EXPECTED_SUBJECTS * EXPECTED_CANDIDATES
        and truth["held_out_scientific_access_count"] == 0
        and truth["held_out_subject_ids_excluded"] == list(HELD_OUT_IDS)
        and len(dev_ids) == EXPECTED_SUBJECTS
        and not set(dev_ids).intersection(HELD_OUT_IDS)
        and candidate_manifest["candidate_count"] == EXPECTED_CANDIDATES
        and candidate_manifest["all_candidates_pass_kinematic_gates"] is True
        and candidate_manifest["mechanical_objective_evaluated"] is False
        and len(candidates) == EXPECTED_CANDIDATES
        and tuple(cohort["development_subject_ids"]) == dev_ids
        and tuple(cohort["held_out_subject_ids"]) == HELD_OUT_IDS
    ):
        raise RuntimeError("frozen stage identity/status mismatch")
    for index, row in enumerate(candidates):
        expected_flex = -0.03 + (index // GRID_SIZE) * GRID_STEP
        expected_extend = -0.03 + (index % GRID_SIZE) * GRID_STEP
        if not (
            int(row["candidate_index"]) == index
            and row["candidate_id"] == f"MYOLEG_V3_K{index:04d}"
            and abs(float(row["beta_flex"]) - expected_flex) <= 1e-14
            and abs(float(row["beta_extend"]) - expected_extend) <= 1e-14
            and row["included"] == "True"
        ):
            raise RuntimeError(f"candidate grid mismatch at {index}")
    return truth, candidate_manifest, cohort, candidates


def verify_held_out_without_loading(cohort: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    verified: list[dict[str, Any]] = []
    for subject_id in HELD_OUT_IDS:
        record = records[subject_id]
        if record["split"] != "HELD_OUT":
            raise RuntimeError(f"held-out split mismatch: {subject_id}")
        file_rows = []
        for path_key, sha_key in (
            ("metadata_path", "metadata_sha256"),
            ("model_delta_path", "model_delta_sha256"),
            ("reference_replay_truth_path", "reference_replay_truth_sha256"),
        ):
            path = ROOT / record[path_key]
            actual = sha256_file(path)
            if actual != record[sha_key]:
                raise RuntimeError(f"sealed file hash mismatch: {path}")
            file_rows.append({"path": record[path_key], "sha256": actual, "operation": "streaming_hash_only"})
        verified.append({"subject_id": subject_id, "files": file_rows})
    if any(row["subject_id"] in HELD_OUT_IDS for row in truth["chunks"]):
        raise RuntimeError("V3 development manifest contains held-out shard")
    audit = read_json(HELD_OUT_AUDIT_PATH)
    audit.update({
        "held_out_file_hash_verification": verified,
        "held_out_files_sha256_verified": 3 * len(HELD_OUT_IDS),
        "held_out_files_opened_via_np_load": 0,
        "held_out_replay_count": 0,
        "held_out_scientific_access_count": 0,
        "v3_development_chunk_subject_ids": [row["subject_id"] for row in truth["chunks"]],
    })
    atomic_json(HELD_OUT_AUDIT_PATH, audit)
    return audit


class DevelopmentTruthStore:
    def __init__(self, truth: dict[str, Any], development_ids: Iterable[str], candidates: list[dict[str, str]]):
        self.allowed = frozenset(development_ids)
        self.truth = truth
        self.candidates = candidates
        self.accessed: list[str] = []
        if len(self.allowed) != EXPECTED_SUBJECTS or self.allowed.intersection(HELD_OUT_IDS):
            raise RuntimeError("development allowlist invalid")

    def load_subject(self, subject_id: str) -> dict[str, np.ndarray]:
        if subject_id not in self.allowed:
            raise PermissionError(f"SEALED_CONFIRMATORY_TRUTH access denied before path resolution: {subject_id}")
        rows = [row for row in self.truth["chunks"] if row["subject_id"] == subject_id]
        if len(rows) != 1 or rows[0]["row_count"] != EXPECTED_CANDIDATES:
            raise RuntimeError(f"development shard coverage mismatch: {subject_id}")
        path = ROOT / rows[0]["path"]
        if sha256_file(path) != rows[0]["sha256"]:
            raise RuntimeError(f"development shard hash mismatch: {path}")
        required = ("candidate_id", "candidate_index", "beta_flex", "beta_extend", "j_truth", "integrity_status")
        with np.load(path, allow_pickle=False) as shard:
            output = {key: np.asarray(shard[key]) for key in required}
        expected_ids = np.asarray([row["candidate_id"] for row in self.candidates])
        expected_indices = np.arange(EXPECTED_CANDIDATES, dtype=np.int32)
        expected_flex = np.asarray([float(row["beta_flex"]) for row in self.candidates])
        expected_extend = np.asarray([float(row["beta_extend"]) for row in self.candidates])
        if not (
            np.array_equal(output["candidate_id"], expected_ids)
            and np.array_equal(output["candidate_index"], expected_indices)
            and np.array_equal(output["beta_flex"], expected_flex)
            and np.array_equal(output["beta_extend"], expected_extend)
            and np.all(output["integrity_status"] == 1)
            and np.isfinite(output["j_truth"]).all()
            and abs(float(output["j_truth"][REFERENCE_INDEX]) - 1.0) <= 1e-12
        ):
            raise RuntimeError(f"development landscape integrity failure: {subject_id}")
        self.accessed.append(subject_id)
        return output


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def oracle_and_diversity(
    subject_ids: list[str], candidate_ids: np.ndarray, beta: np.ndarray, j_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    oracle_indices = np.asarray([select_tied_min(row) for row in j_matrix], dtype=int)
    oracle_beta = beta[oracle_indices]
    oracle_j = j_matrix[np.arange(len(subject_ids)), oracle_indices]
    counts: dict[str, int] = {}
    for index in oracle_indices:
        key = str(candidate_ids[index])
        counts[key] = counts.get(key, 0) + 1
    modal_count = max(counts.values())
    oracle_rows: list[dict[str, Any]] = []
    for row_index, subject_id in enumerate(subject_ids):
        index = int(oracle_indices[row_index])
        flex = float(oracle_beta[row_index, 0])
        extend = float(oracle_beta[row_index, 1])
        flex_lower = bool(np.isclose(flex, -0.03, atol=1e-14))
        flex_upper = bool(np.isclose(flex, 0.03, atol=1e-14))
        extend_lower = bool(np.isclose(extend, -0.03, atol=1e-14))
        extend_upper = bool(np.isclose(extend, 0.03, atol=1e-14))
        oracle_rows.append({
            "subject_id": subject_id, "oracle_candidate_id": str(candidate_ids[index]),
            "oracle_candidate_index": index, "beta_flex_oracle": flex,
            "beta_extend_oracle": extend, "oracle_j": float(oracle_j[row_index]),
            "tie_equivalent_candidate_count": int(np.sum(j_matrix[row_index] <= oracle_j[row_index] + TIE_TOLERANCE)),
            "beta_flex_lower_boundary": flex_lower, "beta_flex_upper_boundary": flex_upper,
            "beta_extend_lower_boundary": extend_lower, "beta_extend_upper_boundary": extend_upper,
            "any_boundary": bool(flex_lower or flex_upper or extend_lower or extend_upper),
            "interior": bool(not (flex_lower or flex_upper or extend_lower or extend_upper)),
            "shares_modal_oracle": counts[str(candidate_ids[index])] == modal_count,
        })
    diversity_rows: list[dict[str, Any]] = []
    for i in range(len(subject_ids)):
        for j in range(i + 1, len(subject_ids)):
            diff = np.abs(oracle_beta[i] - oracle_beta[j])
            grid_diff = np.rint(diff / GRID_STEP).astype(int)
            exact = int(oracle_indices[i]) == int(oracle_indices[j])
            immediate = bool(not exact and np.max(grid_diff) <= 1)
            classification = "EXACT_SAME_ORACLE" if exact else ("IMMEDIATE_GRID_NEIGHBOR" if immediate else "CLEARLY_SEPARATED")
            diversity_rows.append({
                "subject_i": subject_ids[i], "subject_j": subject_ids[j],
                "oracle_candidate_i": str(candidate_ids[oracle_indices[i]]),
                "oracle_candidate_j": str(candidate_ids[oracle_indices[j]]),
                "absolute_beta_flex_difference": float(diff[0]),
                "absolute_beta_extend_difference": float(diff[1]),
                "normalized_2d_beta_distance": float(np.sqrt(np.mean((diff / BETA_RANGE) ** 2))),
                "beta_flex_grid_step_difference": int(grid_diff[0]),
                "beta_extend_grid_step_difference": int(grid_diff[1]),
                "classification": classification,
            })
    return oracle_indices, oracle_beta, oracle_j, oracle_rows, diversity_rows


def rank_and_topsets(
    subject_ids: list[str], j_matrix: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rank_rows: list[dict[str, Any]] = []
    spearman_matrix = np.eye(len(subject_ids))
    kendall_matrix = np.eye(len(subject_ids))
    orders = [np.lexsort((np.arange(EXPECTED_CANDIDATES), j_matrix[i])) for i in range(len(subject_ids))]
    top_rows: list[dict[str, Any]] = []
    for i in range(len(subject_ids)):
        for j in range(i + 1, len(subject_ids)):
            spearman = float(stats.spearmanr(j_matrix[i], j_matrix[j]).statistic)
            kendall = float(stats.kendalltau(j_matrix[i], j_matrix[j], variant="b").statistic)
            spearman_matrix[i, j] = spearman_matrix[j, i] = spearman
            kendall_matrix[i, j] = kendall_matrix[j, i] = kendall
            rank_rows.append({
                "subject_i": subject_ids[i], "subject_j": subject_ids[j],
                "spearman_rank_correlation": spearman, "kendall_tau_b": kendall,
            })
            for fraction, count in zip(TOP_FRACTIONS, TOP_COUNTS):
                set_i = set(orders[i][:count].tolist())
                set_j = set(orders[j][:count].tolist())
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                top_rows.append({
                    "subject_i": subject_ids[i], "subject_j": subject_ids[j],
                    "top_fraction": fraction, "candidate_count": count,
                    "intersection_count": intersection, "union_count": union,
                    "jaccard": intersection / union,
                    "overlap_coefficient": intersection / min(len(set_i), len(set_j)),
                })
    return rank_rows, spearman_matrix, kendall_matrix, top_rows


def common_baselines_and_gaps(
    subject_ids: list[str], candidate_ids: np.ndarray, beta: np.ndarray,
    j_matrix: np.ndarray, oracle_j: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    mean_by_candidate = np.mean(j_matrix, axis=0)
    worst_by_candidate = np.max(j_matrix, axis=0)
    mean_index = select_tied_min(mean_by_candidate)
    worst_index = select_tied_min(worst_by_candidate)
    baselines: dict[str, Any] = {
        "V3_DEV_MEAN_OPTIMAL_COMMON": {
            "candidate_id": str(candidate_ids[mean_index]), "candidate_index": mean_index,
            "beta_flex": float(beta[mean_index, 0]), "beta_extend": float(beta[mean_index, 1]),
            "mean_j": float(mean_by_candidate[mean_index]),
            "median_j": float(np.median(j_matrix[:, mean_index])),
            "worst_case_j": float(np.max(j_matrix[:, mean_index])),
        },
        "V3_DEV_WORSTCASE_OPTIMAL_COMMON": {
            "candidate_id": str(candidate_ids[worst_index]), "candidate_index": worst_index,
            "beta_flex": float(beta[worst_index, 0]), "beta_extend": float(beta[worst_index, 1]),
            "mean_j": float(mean_by_candidate[worst_index]),
            "median_j": float(np.median(j_matrix[:, worst_index])),
            "worst_case_j": float(worst_by_candidate[worst_index]),
        },
        "REFERENCE": {"candidate_id": REFERENCE_ID, "candidate_index": REFERENCE_INDEX, "beta_flex": 0.0, "beta_extend": 0.0, "j": 1.0},
    }
    common_j = j_matrix[:, mean_index]
    common_regret = common_j - oracle_j
    relative_gap = common_regret / common_j
    gap_rows = [
        {
            "subject_id": subject_id, "oracle_j": float(oracle_j[i]),
            "common_candidate_id": str(candidate_ids[mean_index]), "common_j": float(common_j[i]),
            "common_regret": float(common_regret[i]),
            "relative_personalization_gap": float(relative_gap[i]),
            "engineering_effect_band": effect_band(float(relative_gap[i])),
        }
        for i, subject_id in enumerate(subject_ids)
    ]
    baselines["personalization_gap_bootstrap"] = bootstrap_intervals(relative_gap)
    return baselines, common_regret, relative_gap, gap_rows


def local_direction(subject_ids: list[str], candidate_ids: np.ndarray, j_matrix: np.ndarray) -> list[dict[str, Any]]:
    neighbors = (
        (287, "beta_flex_negative", -GRID_STEP),
        (337, "beta_flex_positive", GRID_STEP),
        (311, "beta_extend_negative", -GRID_STEP),
        (313, "beta_extend_positive", GRID_STEP),
    )
    rows: list[dict[str, Any]] = []
    for subject_index, subject_id in enumerate(subject_ids):
        reference_j = float(j_matrix[subject_index, REFERENCE_INDEX])
        for candidate_index, direction, signed_step in neighbors:
            delta = float(j_matrix[subject_index, candidate_index] - reference_j)
            sign = int(sign_class(delta))
            rows.append({
                "subject_id": subject_id, "direction": direction,
                "reference_candidate_id": REFERENCE_ID,
                "neighbor_candidate_id": str(candidate_ids[candidate_index]),
                "signed_beta_step": signed_step, "reference_j": reference_j,
                "neighbor_j": float(j_matrix[subject_index, candidate_index]),
                "delta_j_neighbor_minus_reference": delta,
                "finite_difference_dj_dbeta": delta / signed_step,
                "delta_sign": sign,
                "neighbor_is_descent": bool(sign < 0),
            })
    return rows


def global_directional_agreement(j_matrix: np.ndarray) -> list[dict[str, Any]]:
    cube = j_matrix.reshape(EXPECTED_SUBJECTS, GRID_SIZE, GRID_SIZE)
    rows: list[dict[str, Any]] = []
    for axis in ("beta_flex", "beta_extend"):
        for lower in range(GRID_SIZE - 1):
            for fixed in range(GRID_SIZE):
                if axis == "beta_flex":
                    delta = cube[:, lower + 1, fixed] - cube[:, lower, fixed]
                    lower_index, upper_index = lower * GRID_SIZE + fixed, (lower + 1) * GRID_SIZE + fixed
                else:
                    delta = cube[:, fixed, lower + 1] - cube[:, fixed, lower]
                    lower_index, upper_index = fixed * GRID_SIZE + lower, fixed * GRID_SIZE + lower + 1
                signs = sign_class(delta)
                negative = int(np.sum(signs == -1))
                equivalent = int(np.sum(signs == 0))
                positive = int(np.sum(signs == 1))
                opposite_pairs = negative * positive
                rows.append({
                    "axis": axis, "transition_index": lower,
                    "fixed_other_axis_index": fixed,
                    "lower_candidate_index": lower_index, "upper_candidate_index": upper_index,
                    "negative_subject_count": negative, "equivalent_subject_count": equivalent,
                    "positive_subject_count": positive,
                    "same_sign_fraction": max(negative, equivalent, positive) / EXPECTED_SUBJECTS,
                    "non_equivalent_sign_disagreement": bool(negative > 0 and positive > 0),
                    "rank_inversion_subject_pair_count": opposite_pairs,
                    "rank_inversion_fraction_of_276": opposite_pairs / 276.0,
                    "delta_j_min": float(np.min(delta)), "delta_j_median": float(np.median(delta)),
                    "delta_j_mean": float(np.mean(delta)), "delta_j_max": float(np.max(delta)),
                })
    return rows


def transfer_analysis(
    subject_ids: list[str], candidate_ids: np.ndarray, j_matrix: np.ndarray,
    oracle_indices: np.ndarray, oracle_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    transfer = j_matrix[:, oracle_indices]
    regret = transfer - oracle_j[:, None]
    relative = regret / transfer
    matrix_rows: list[dict[str, Any]] = []
    for i, subject_id in enumerate(subject_ids):
        row: dict[str, Any] = {"recipient_subject_id": subject_id}
        for j, donor_id in enumerate(subject_ids):
            row[f"donor_{donor_id}"] = float(transfer[i, j])
        matrix_rows.append(row)
    summary_rows: list[dict[str, Any]] = []
    for i, subject_id in enumerate(subject_ids):
        foreign_mask = np.arange(EXPECTED_SUBJECTS) != i
        foreign = transfer[i, foreign_mask]
        foreign_regret = regret[i, foreign_mask]
        summary_rows.append({
            "recipient_subject_id": subject_id, "own_oracle_candidate_id": str(candidate_ids[oracle_indices[i]]),
            "own_oracle_j": float(oracle_j[i]), "best_foreign_oracle_j": float(np.min(foreign)),
            "mean_foreign_oracle_j": float(np.mean(foreign)), "worst_foreign_oracle_j": float(np.max(foreign)),
            "median_foreign_regret": float(np.median(foreign_regret)),
            "max_foreign_regret": float(np.max(foreign_regret)),
            "foreign_oracle_equivalent_fraction": float(np.mean(np.abs(foreign_regret) <= TIE_TOLERANCE)),
        })
    return transfer, regret, relative, matrix_rows, summary_rows


def near_oracle_analysis(
    subject_ids: list[str], candidate_ids: np.ndarray, beta: np.ndarray,
    j_matrix: np.ndarray, oracle_j: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plateau_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for epsilon in NEAR_EPSILONS:
        mask = j_matrix <= oracle_j[:, None] + epsilon
        for i, subject_id in enumerate(subject_ids):
            selected = beta[mask[i]]
            lower = np.min(selected, axis=0)
            upper = np.max(selected, axis=0)
            ranges = upper - lower
            bbox_grid_count = int((round(ranges[0] / GRID_STEP) + 1) * (round(ranges[1] / GRID_STEP) + 1))
            plateau_rows.append({
                "subject_id": subject_id, "epsilon_j": epsilon,
                "near_oracle_candidate_count": int(np.sum(mask[i])),
                "near_oracle_fraction": float(np.mean(mask[i])),
                "beta_flex_min": float(lower[0]), "beta_flex_max": float(upper[0]),
                "beta_flex_range": float(ranges[0]),
                "beta_extend_min": float(lower[1]), "beta_extend_max": float(upper[1]),
                "beta_extend_range": float(ranges[1]),
                "normalized_beta_bbox_diagonal": float(np.sqrt(np.mean((ranges / BETA_RANGE) ** 2))),
                "bounding_box_grid_candidate_count": bbox_grid_count,
                "bounding_box_grid_fraction": bbox_grid_count / EXPECTED_CANDIDATES,
            })
        coverage = np.sum(mask, axis=0)
        maximum = int(np.max(coverage))
        best_index = int(np.flatnonzero(coverage == maximum)[0])
        common_rows.append({
            "epsilon_j": epsilon, "maximum_subject_coverage": maximum,
            "best_coverage_candidate_id": str(candidate_ids[best_index]),
            "best_coverage_candidate_index": best_index,
            "best_coverage_beta_flex": float(beta[best_index, 0]),
            "best_coverage_beta_extend": float(beta[best_index, 1]),
            "candidate_count_coverage_24": int(np.sum(coverage >= 24)),
            "candidate_count_coverage_at_least_22": int(np.sum(coverage >= 22)),
            "candidate_count_coverage_at_least_18": int(np.sum(coverage >= 18)),
            "universal_near_oracle_candidate_exists": bool(np.any(coverage == 24)),
        })
    return plateau_rows, common_rows


def landscape_contrast(
    subject_ids: list[str], j_matrix: np.ndarray, oracle_j: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top5_count = TOP_COUNTS[1]
    for i, subject_id in enumerate(subject_ids):
        ordered = np.lexsort((np.arange(EXPECTED_CANDIDATES), j_matrix[i]))
        boundary_j = float(j_matrix[i, ordered[top5_count - 1]])
        rows.append({
            "subject_id": subject_id, "oracle_j": float(oracle_j[i]),
            "reference_j": float(j_matrix[i, REFERENCE_INDEX]),
            "reference_minus_oracle_j": float(j_matrix[i, REFERENCE_INDEX] - oracle_j[i]),
            "landscape_min_j": float(np.min(j_matrix[i])),
            "landscape_median_j": float(np.median(j_matrix[i])),
            "landscape_max_j": float(np.max(j_matrix[i])),
            "landscape_dynamic_range_j": float(np.ptp(j_matrix[i])),
            "top5_percent_boundary_j": boundary_j,
            "top5_percent_to_oracle_spread_j": boundary_j - float(oracle_j[i]),
        })
    return rows


def multiplicative_similarity(subject_ids: list[str], j_matrix: np.ndarray) -> list[dict[str, Any]]:
    anchor_id = "MYOLEG_VP_031"
    anchor_index = subject_ids.index(anchor_id)
    rows: list[dict[str, Any]] = []
    for i, subject_id in enumerate(subject_ids):
        rows.append({"subject_id": subject_id, "anchor_subject_id": anchor_id, **fit_scaling(j_matrix[i], j_matrix[anchor_index])})
    return rows


def parameter_associations(
    subject_ids: list[str], cohort: dict[str, Any], oracle_beta: np.ndarray,
    oracle_j: np.ndarray, common_regret: np.ndarray, interaction_residual_rms: np.ndarray,
) -> list[dict[str, Any]]:
    records = {row["subject_id"]: row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"}
    parameter_names = tuple(cohort["factor_order"])
    parameter_matrix = np.asarray([[records[subject_id]["factor_values"][name] for name in parameter_names] for subject_id in subject_ids])
    outcome_names = (
        "oracle_beta_flex", "oracle_beta_extend", "oracle_j",
        "common_regret", "subject_interaction_residual_rms",
    )
    outcomes = np.column_stack((oracle_beta, oracle_j, common_regret, interaction_residual_rms))
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for p_index, parameter in enumerate(parameter_names):
        for o_index, outcome_name in enumerate(outcome_names):
            x = parameter_matrix[:, p_index]
            y = outcomes[:, o_index]
            reasons = []
            if np.ptp(x) == 0.0:
                reasons.append("CONSTANT_PARAMETER")
            if np.ptp(y) == 0.0:
                reasons.append("CONSTANT_OUTCOME")
            if reasons:
                rho, raw_p = None, 1.0
            else:
                result = stats.spearmanr(x, y)
                rho, raw_p = float(result.statistic), float(result.pvalue)
            p_values.append(raw_p)
            rows.append({
                "subject_parameter": parameter, "outcome": outcome_name,
                "spearman_rho": rho, "raw_p_value": raw_p,
                "undefined_reason": ";".join(reasons),
                "exploratory_only": True, "predictive_learner_trained": False,
            })
    for row, q_value in zip(rows, bh_qvalues(p_values)):
        row["bh_q_value_across_30_tests"] = float(q_value)
    return rows


def decision_metrics(
    oracle_indices: np.ndarray, oracle_beta: np.ndarray, oracle_rows: list[dict[str, Any]],
    diversity_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]],
    relative_gap: np.ndarray, decomposition: dict[str, Any], transfer_relative: np.ndarray,
    common_near_rows: list[dict[str, Any]], plateau_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    classifications = [row["classification"] for row in diversity_rows]
    distances = np.asarray([row["normalized_2d_beta_distance"] for row in diversity_rows])
    spearman = np.asarray([row["spearman_rank_correlation"] for row in rank_rows])
    kendall = np.asarray([row["kendall_tau_b"] for row in rank_rows])
    top5_jaccard = np.asarray([row["jaccard"] for row in top_rows if row["top_fraction"] == 0.05])
    common_001 = next(row for row in common_near_rows if row["epsilon_j"] == 0.001)
    plateau_001 = np.asarray([row["near_oracle_candidate_count"] for row in plateau_rows if row["epsilon_j"] == 0.001])
    offdiag = transfer_relative[~np.eye(EXPECTED_SUBJECTS, dtype=bool)]
    material_interaction = decomposition["subject_candidate_interaction_variance_fraction"] >= 0.0025
    global_disagreement = np.asarray([row["non_equivalent_sign_disagreement"] for row in global_rows], dtype=bool)
    modal_count = max(np.bincount(oracle_indices, minlength=EXPECTED_CANDIDATES))
    return {
        "unique_oracle_candidate_count": int(len(np.unique(oracle_indices))),
        "unique_beta_flex_oracle_count": int(len(np.unique(oracle_beta[:, 0]))),
        "unique_beta_extend_oracle_count": int(len(np.unique(oracle_beta[:, 1]))),
        "modal_oracle_subject_count": int(modal_count),
        "modal_oracle_subject_fraction": float(modal_count / EXPECTED_SUBJECTS),
        "boundary_oracle_fraction": float(np.mean([row["any_boundary"] for row in oracle_rows])),
        "all_subjects_share_one_boundary_oracle": bool(len(np.unique(oracle_indices)) == 1 and all(row["any_boundary"] for row in oracle_rows)),
        "oracle_pair_classification_counts": {label: classifications.count(label) for label in ("EXACT_SAME_ORACLE", "IMMEDIATE_GRID_NEIGHBOR", "CLEARLY_SEPARATED")},
        "clearly_separated_oracle_pair_fraction": classifications.count("CLEARLY_SEPARATED") / 276.0,
        "normalized_oracle_distance": summary(distances),
        "spearman_rank_correlation": summary(spearman),
        "kendall_tau_b": summary(kendall),
        "top5_jaccard": summary(top5_jaccard),
        "relative_common_regret": summary(relative_gap),
        "candidate_main_variance_fraction": decomposition["candidate_main_variance_fraction"],
        "subject_main_variance_fraction": decomposition["subject_main_variance_fraction"],
        "interaction_variance_fraction": decomposition["subject_candidate_interaction_variance_fraction"],
        "interaction_variance_percent": 100.0 * decomposition["subject_candidate_interaction_variance_fraction"],
        "interaction_to_common_effect_ratio": decomposition["interaction_to_common_effect_ratio"],
        "material_interaction_increase_vs_v2": bool(material_interaction),
        "interaction_fold_change_vs_v2": decomposition["subject_candidate_interaction_variance_fraction"] / 0.00033114,
        "off_diagonal_relative_oracle_transfer_regret": summary(offdiag),
        "meaningful_median_transfer_loss": bool(np.median(offdiag) >= 0.001),
        "max_common_near_oracle_coverage_epsilon_0_001": int(common_001["maximum_subject_coverage"]),
        "universal_near_oracle_epsilon_0_001": bool(common_001["universal_near_oracle_candidate_exists"]),
        "near_oracle_count_epsilon_0_001": summary(plateau_001),
        "broad_plateau_rule_triggered": bool(np.median(plateau_001) >= 63),
        "global_transition_count": len(global_rows),
        "global_direction_disagreement_transition_count": int(np.sum(global_disagreement)),
        "global_direction_disagreement_transition_fraction": float(np.mean(global_disagreement)),
    }


def classify(metrics: dict[str, Any]) -> str:
    supported = bool(
        metrics["unique_oracle_candidate_count"] >= 4
        and metrics["clearly_separated_oracle_pair_fraction"] >= 0.50
        and metrics["relative_common_regret"]["median"] >= 0.005
        and metrics["relative_common_regret"]["p75"] >= 0.01
        and (metrics["spearman_rank_correlation"]["median"] <= 0.95 or metrics["top5_jaccard"]["median"] <= 0.75)
        and metrics["max_common_near_oracle_coverage_epsilon_0_001"] < 22
        and metrics["material_interaction_increase_vs_v2"]
        and metrics["off_diagonal_relative_oracle_transfer_regret"]["median"] >= 0.001
    )
    not_supported = bool(
        metrics["unique_oracle_candidate_count"] <= 2
        and metrics["relative_common_regret"]["median"] < 0.001
        and metrics["relative_common_regret"]["p95"] < 0.005
        and metrics["spearman_rank_correlation"]["median"] >= 0.98
        and metrics["universal_near_oracle_epsilon_0_001"]
    )
    if supported:
        return OUTCOME_SUPPORTED
    if not_supported:
        return OUTCOME_NOT_SUPPORTED
    return OUTCOME_WEAK


def matrix_rows(subject_ids: list[str], matrix: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, subject_id in enumerate(subject_ids):
        row: dict[str, Any] = {"subject_id": subject_id}
        for j, other_id in enumerate(subject_ids):
            row[f"{prefix}_{other_id}"] = float(matrix[i, j])
        rows.append(row)
    return rows


def create_v3_figures(
    protocol: dict[str, Any], subject_ids: list[str], beta: np.ndarray,
    oracle_beta: np.ndarray, j_matrix: np.ndarray, spearman_matrix: np.ndarray,
    relative_gap: np.ndarray, transfer_regret: np.ndarray,
    cohort: dict[str, Any], interaction_residual: np.ndarray,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(6.3, 5.4))
    unique, counts = np.unique(oracle_beta, axis=0, return_counts=True)
    plt.scatter(unique[:, 0], unique[:, 1], s=45 + 28 * counts, c=counts, cmap="viridis", edgecolors="black", linewidths=0.6)
    for point, count in zip(unique, counts):
        x_offset = -8 if point[0] > 0.025 else 5
        horizontal = "right" if point[0] > 0.025 else "left"
        plt.annotate(str(int(count)), point, xytext=(x_offset, 5), textcoords="offset points", fontsize=9, ha=horizontal)
    plt.xlabel("Oracle beta_flex")
    plt.ylabel("Oracle beta_extend")
    plt.xlim(-0.035, 0.035)
    plt.ylim(-0.035, 0.035)
    plt.title("Development-subject V3 oracle coordination paths")
    save_figure(FIGURES / "01_oracle_beta_scatter.png")

    mean_landscape = np.mean(j_matrix, axis=0).reshape(GRID_SIZE, GRID_SIZE)
    plt.figure(figsize=(6.3, 5.4))
    image = plt.imshow(mean_landscape.T, origin="lower", extent=(-0.03, 0.03, -0.03, 0.03), aspect="equal", cmap="viridis")
    plt.colorbar(image, label="Mean normalized torque objective J")
    plt.xlabel("beta_flex")
    plt.ylabel("beta_extend")
    plt.title("V3 development mean landscape")
    save_figure(FIGURES / "02_common_mean_landscape.png")

    selected = protocol["representative_landscape_subjects"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.8), sharex=True, sharey=True)
    for axis, selected_row in zip(axes.ravel(), selected):
        subject_index = subject_ids.index(selected_row["subject_id"])
        image = axis.imshow(j_matrix[subject_index].reshape(GRID_SIZE, GRID_SIZE).T, origin="lower", extent=(-0.03, 0.03, -0.03, 0.03), aspect="equal", cmap="viridis")
        axis.set_title(f"{selected_row['subject_id']}\n{selected_row['selection_role']}", fontsize=9)
        axis.set_xlabel("beta_flex")
        axis.set_ylabel("beta_extend")
        fig.colorbar(image, ax=axis, label="J")
    fig.suptitle("Preregistered V3 development-subject landscapes", fontsize=11)
    save_figure(FIGURES / "03_preselected_subject_landscapes.png")

    plt.figure(figsize=(7.8, 6.8))
    image = plt.imshow(spearman_matrix, vmin=float(np.min(spearman_matrix)), vmax=1.0, cmap="viridis", aspect="equal")
    plt.colorbar(image, label="Spearman rank correlation")
    plt.xticks(range(24), subject_ids, rotation=90, fontsize=6)
    plt.yticks(range(24), subject_ids, fontsize=6)
    plt.title("V3 development candidate-ranking similarity")
    save_figure(FIGURES / "04_rank_correlation_heatmap.png")

    plt.figure(figsize=(6.5, 4.2))
    plt.hist(100 * relative_gap, bins=12, color="#4c78a8", edgecolor="white")
    for value in (0.1, 0.5, 1.0, 2.0):
        plt.axvline(value, color="black", linewidth=0.7, linestyle="--")
    plt.xlabel("Relative common-trajectory regret (%)")
    plt.ylabel("Development subjects")
    plt.xlim(-0.05, 2.10)
    plt.title("Oracle upper-bound mechanical personalization gap")
    save_figure(FIGURES / "05_common_regret_distribution.png")

    plt.figure(figsize=(7.8, 6.8))
    transfer_max = float(np.max(transfer_regret))
    image = plt.imshow(transfer_regret, cmap="magma", aspect="equal", vmin=0.0, vmax=max(transfer_max, 1e-6))
    plt.colorbar(image, label="Regret versus recipient oracle (J)")
    plt.xticks(range(24), subject_ids, rotation=90, fontsize=6)
    plt.yticks(range(24), subject_ids, fontsize=6)
    plt.xlabel("Donor oracle subject")
    plt.ylabel("Recipient subject")
    plt.title("V3 cross-subject oracle transfer")
    if transfer_max <= TIE_TOLERANCE:
        plt.text(11.5, 11.5, "All transfer regrets = 0", color="white", ha="center", va="center", fontsize=10)
    save_figure(FIGURES / "06_oracle_transfer_heatmap.png")

    records = {row["subject_id"]: row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"}
    factor_names = tuple(cohort["factor_order"])
    factor_labels = {
        "FEMUR_MASS_INERTIA_SCALE": "Femur mass/inertia scale",
        "TIBIA_PATELLA_MASS_INERTIA_SCALE": "Tibia/patella mass/inertia scale",
        "FOOT_COMPLEX_MASS_INERTIA_SCALE": "Foot-complex mass/inertia scale",
        "HIP_ONLY_PASSIVE_FP_MAX_SCALE": "Hip-only passive scale",
        "KNEE_ONLY_PASSIVE_FP_MAX_SCALE": "Knee-only passive scale",
        "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE": "Biarticular passive scale",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.1))
    for axis, factor in zip(axes.ravel(), factor_names):
        x = np.asarray([records[subject_id]["factor_values"][factor] for subject_id in subject_ids])
        axis.scatter(x, interaction_residual, color="#4c78a8", s=32, edgecolors="black", linewidths=0.3)
        axis.set_xlabel(factor_labels[factor], fontsize=8)
        axis.set_ylabel("Interaction residual RMS")
        axis.tick_params(labelsize=7)
    fig.suptitle("Frozen subject factors versus V3 path-interaction residual", fontsize=11)
    fig.subplots_adjust(top=0.90, wspace=0.36, hspace=0.40)
    plt.savefig(FIGURES / "08_subject_factor_path_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def freeze_v3_results(paths: list[Path], metrics: dict[str, Any], decision: str) -> dict[str, Any]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze incomplete V3-only analysis: {missing}")
    payload = {
        "freeze_id": "MYOLEG_V3_PERSONALIZATION_NECESSITY_V3_ONLY_RESULT_FREEZE_V1",
        "frozen_before_v2_result_artifacts_were_opened_by_analysis_execution": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "decision": decision,
        "decision_metrics": metrics,
        "v3_only_artifact_sha256": {str(path.relative_to(OUTPUT)): sha256_file(path) for path in sorted(paths)},
    }
    atomic_json(OUTPUT / "V3_ANALYSIS_RESULT_FREEZE.json", payload)
    return payload


def v2_v3_comparison(metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v2_paths = {
        "protocol": V2_PROTOCOL,
        "oracle": V2_NECESSITY / "DEV_ORACLE_SUMMARY.csv",
        "gap": V2_NECESSITY / "DEV_PERSONALIZATION_GAP.csv",
        "rank": V2_NECESSITY / "DEV_RANK_CORRELATION.csv",
        "transfer": V2_NECESSITY / "DEV_ORACLE_TRANSFER_MATRIX.csv",
        "common_near": V2_NECESSITY / "DEV_COMMON_NEAR_ORACLE_ANALYSIS.csv",
        "decomposition": V2_ROOT_CAUSE / "TWO_WAY_VARIANCE_DECOMPOSITION.json",
    }
    hashes = {name: sha256_file(path) for name, path in v2_paths.items()}
    v2_oracle = read_csv(v2_paths["oracle"])
    v2_gap = read_csv(v2_paths["gap"])
    v2_rank = read_csv(v2_paths["rank"])
    v2_transfer = read_csv(v2_paths["transfer"])
    v2_common = read_csv(v2_paths["common_near"])
    v2_decomp = read_json(v2_paths["decomposition"])["matrices"]["J"]
    v2_spearman = np.asarray([float(row["spearman_rank_correlation"]) for row in v2_rank])
    v2_kendall = np.asarray([float(row["kendall_tau_b"]) for row in v2_rank])
    v2_relative_gap = np.asarray([float(row["relative_personalization_gap"]) for row in v2_gap])
    v2_transfer_offdiag = np.asarray([
        float(row["relative_regret_vs_transfer_j"])
        for row in v2_transfer if row["own_oracle_diagonal"] == "False"
    ])
    v2_001 = next(row for row in v2_common if abs(float(row["epsilon_j"]) - 0.001) <= 1e-15)
    v2 = {
        "candidate_dimensions": 3,
        "candidate_count": 16675,
        "rom_invariant": False,
        "distinct_oracle_count": len({row["candidate_id"] for row in v2_oracle}),
        "boundary_oracle_fraction": float(np.mean([row["any_candidate_domain_boundary"] == "True" for row in v2_oracle])),
        "spearman_median": float(np.median(v2_spearman)),
        "kendall_median": float(np.median(v2_kendall)),
        "candidate_main_variance_percent": 100 * float(v2_decomp["candidate_main_variance_fraction"]),
        "interaction_variance_percent": 100 * float(v2_decomp["subject_candidate_interaction_variance_fraction"]),
        "interaction_to_common_ratio": float(v2_decomp["interaction_to_common_effect_ratio"]),
        "common_regret_mean": float(np.mean(v2_relative_gap)),
        "common_regret_max": float(np.max(v2_relative_gap)),
        "off_diagonal_transfer_relative_regret_median": float(np.median(v2_transfer_offdiag)),
        "off_diagonal_transfer_relative_regret_max": float(np.max(v2_transfer_offdiag)),
        "universal_near_oracle_epsilon_0_001": v2_001["universal_near_oracle_candidate_exists"] == "True",
        "maximum_common_near_oracle_coverage_epsilon_0_001": int(v2_001["maximum_subject_coverage"]),
    }
    if not (
        v2["distinct_oracle_count"] == 1
        and v2["boundary_oracle_fraction"] == 1.0
        and abs(v2["spearman_median"] - 0.999834) < 1e-6
        and abs(v2["kendall_median"] - 0.989722) < 1e-6
        and abs(v2["interaction_variance_percent"] - 0.033114) < 1e-6
        and np.all(v2_relative_gap == 0.0)
        and np.all(v2_transfer_offdiag == 0.0)
        and v2["universal_near_oracle_epsilon_0_001"]
    ):
        raise RuntimeError(f"V2 frozen result identity mismatch: {v2}")
    v3 = {
        "candidate_dimensions": 2,
        "candidate_count": EXPECTED_CANDIDATES,
        "rom_invariant": True,
        "distinct_oracle_count": metrics["unique_oracle_candidate_count"],
        "boundary_oracle_fraction": metrics["boundary_oracle_fraction"],
        "spearman_median": metrics["spearman_rank_correlation"]["median"],
        "kendall_median": metrics["kendall_tau_b"]["median"],
        "candidate_main_variance_percent": 100.0 * metrics["candidate_main_variance_fraction"],
        "interaction_variance_percent": metrics["interaction_variance_percent"],
        "interaction_to_common_ratio": metrics["interaction_to_common_effect_ratio"],
        "common_regret_mean": metrics["relative_common_regret"]["mean"],
        "common_regret_max": metrics["relative_common_regret"]["max"],
        "off_diagonal_transfer_relative_regret_median": metrics["off_diagonal_relative_oracle_transfer_regret"]["median"],
        "off_diagonal_transfer_relative_regret_max": metrics["off_diagonal_relative_oracle_transfer_regret"]["max"],
        "universal_near_oracle_epsilon_0_001": metrics["universal_near_oracle_epsilon_0_001"],
        "maximum_common_near_oracle_coverage_epsilon_0_001": metrics["max_common_near_oracle_coverage_epsilon_0_001"],
    }
    units = {
        "candidate_dimensions": "count", "candidate_count": "count", "rom_invariant": "boolean",
        "distinct_oracle_count": "count", "boundary_oracle_fraction": "fraction",
        "spearman_median": "correlation", "kendall_median": "correlation",
        "candidate_main_variance_percent": "percent", "interaction_variance_percent": "percent",
        "interaction_to_common_ratio": "ratio", "common_regret_mean": "relative J fraction",
        "common_regret_max": "relative J fraction",
        "off_diagonal_transfer_relative_regret_median": "relative J fraction",
        "off_diagonal_transfer_relative_regret_max": "relative J fraction",
        "universal_near_oracle_epsilon_0_001": "boolean",
        "maximum_common_near_oracle_coverage_epsilon_0_001": "subjects",
    }
    rows = [
        {"metric": name, "v2": v2[name], "v3": v3[name], "unit": units[name], "comparison_frozen_after_v3_only_result_freeze": True}
        for name in v2
    ]
    payload = {
        "comparison_id": "MYOLEG_V2_V3_PERSONALIZATION_COMPARISON_V1",
        "v3_result_freeze_sha256": sha256_file(OUTPUT / "V3_ANALYSIS_RESULT_FREEZE.json"),
        "v2_source_artifact_sha256": hashes,
        "raw_near_oracle_candidate_counts_compared": False,
        "v2": v2, "v3": v3,
    }
    return rows, payload


def create_comparison_figure(comparison: dict[str, Any]) -> None:
    v2, v3 = comparison["v2"], comparison["v3"]
    labels = ("Interaction\nvariance (%)", "Rank\ndissimilarity (%)", "Mean common\nregret (%)", "Max transfer\nregret (%)")
    v2_values = (
        v2["interaction_variance_percent"],
        100 * (1 - v2["spearman_median"]),
        100 * v2["common_regret_mean"],
        100 * v2["off_diagonal_transfer_relative_regret_max"],
    )
    v3_values = (
        v3["interaction_variance_percent"],
        100 * (1 - v3["spearman_median"]),
        100 * v3["common_regret_mean"],
        100 * v3["off_diagonal_transfer_relative_regret_max"],
    )
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8.4, 4.6))
    plt.bar(x - width / 2, v2_values, width, label="V2")
    plt.bar(x + width / 2, v3_values, width, label="V3")
    plt.xticks(x, labels)
    plt.ylabel("Percent-scale diagnostic")
    plt.title("Frozen development-only V2 versus V3 diagnostics")
    plt.legend(frameon=False)
    save_figure(FIGURES / "07_v2_v3_comparison.png")


def recommended_branch(decision: str) -> str:
    if decision == OUTCOME_SUPPORTED:
        return "MYOLEG_V3_SUBJECT_MODEL_BENCHMARK_DESIGN_V1"
    if decision == OUTCOME_WEAK:
        return "MYOLEG_V3_PERSONALIZATION_EFFECT_SIZE_AND_OBJECTIVE_AUDIT_V1"
    return "MYOLEG_OBJECTIVE_AND_MUSCULOSKELETAL_HETEROGENEITY_DECISION_AUDIT_V1"


def write_report(
    decision: str, metrics: dict[str, Any], baselines: dict[str, Any],
    oracle_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]], common_near_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]], comparison: dict[str, Any],
) -> None:
    modal = max(set(row["oracle_candidate_id"] for row in oracle_rows), key=lambda item: sum(r["oracle_candidate_id"] == item for r in oracle_rows))
    common_001 = next(row for row in common_near_rows if row["epsilon_j"] == 0.001)
    significant = [row for row in association_rows if row["bh_q_value_across_30_tests"] < 0.05]
    local_summary = {}
    for direction in ("beta_flex_negative", "beta_flex_positive", "beta_extend_negative", "beta_extend_positive"):
        signs = [row["delta_sign"] for row in local_rows if row["direction"] == direction]
        local_summary[direction] = {
            "descent": signs.count(-1), "equivalent": signs.count(0), "increase": signs.count(1),
            "majority_fraction": max(signs.count(-1), signs.count(0), signs.count(1)) / 24,
        }
    branch = recommended_branch(decision)
    report = f"""# MyoLeg V3 Personalization Necessity Audit V1

## Formal decision

`{decision}`

Recommended next scientific branch: `{branch}`. This report does not execute that stage.

This is a development-only offline oracle upper-bound audit under the frozen normalized torque objective. It measures subject-specific mechanical trajectory preference in heterogeneous musculoskeletal virtual subjects. It is not achieved algorithm benefit, patient preference, comfort, clinical improvement, or safety evidence.

## Q1 — Distinct V3 oracle coordination paths

There are **{metrics['unique_oracle_candidate_count']}** distinct exact oracle paths among 24 development subjects, with **{metrics['unique_beta_flex_oracle_count']}** beta_flex values and **{metrics['unique_beta_extend_oracle_count']}** beta_extend values. The modal oracle is `{modal}` and is shared by **{metrics['modal_oracle_subject_count']}/24 ({metrics['modal_oracle_subject_fraction']:.1%})**.

Boundary-oracle fraction is **{metrics['boundary_oracle_fraction']:.1%}**. All 24 share one boundary oracle: **{metrics['all_subjects_share_one_boundary_oracle']}**. The beta domain was not expanded.

## Q2 — Mechanical size versus grid/equivalence effects

Of 276 subject pairs, **{metrics['oracle_pair_classification_counts']['EXACT_SAME_ORACLE']}** share the exact oracle, **{metrics['oracle_pair_classification_counts']['IMMEDIATE_GRID_NEIGHBOR']}** are immediate grid neighbors, and **{metrics['oracle_pair_classification_counts']['CLEARLY_SEPARATED']}** are separated. The median / P95 / maximum normalized 2-D oracle distance is **{metrics['normalized_oracle_distance']['median']:.6g} / {metrics['normalized_oracle_distance']['p95']:.6g} / {metrics['normalized_oracle_distance']['max']:.6g}**.

## Q3 — Subject-by-candidate interaction

V3 candidate-main variance is **{comparison['v3']['candidate_main_variance_percent']:.6f}%** and subject-by-candidate interaction is **{metrics['interaction_variance_percent']:.6f}%**, versus frozen V2 **0.033114%**. The V3/V2 interaction fold change is **{metrics['interaction_fold_change_vs_v2']:.3f}x** and interaction/common-effect RMS ratio is **{metrics['interaction_to_common_effect_ratio']:.6g}**. Under the preregistered >=0.25% material-increase rule, material increase is **{metrics['material_interaction_increase_vs_v2']}**.

## Q4 — Candidate-ranking similarity

Pairwise Spearman median / range is **{metrics['spearman_rank_correlation']['median']:.6f} / [{metrics['spearman_rank_correlation']['min']:.6f}, {metrics['spearman_rank_correlation']['max']:.6f}]**, versus V2 median 0.999834. Kendall median is **{metrics['kendall_tau_b']['median']:.6f}**, versus V2 0.989722. Top-5% Jaccard median is **{metrics['top5_jaccard']['median']:.6f}**.

## Q5 — Common-trajectory regret

The development mean-optimal common candidate is `{baselines['V3_DEV_MEAN_OPTIMAL_COMMON']['candidate_id']}` at beta **[{baselines['V3_DEV_MEAN_OPTIMAL_COMMON']['beta_flex']}, {baselines['V3_DEV_MEAN_OPTIMAL_COMMON']['beta_extend']}]**. Relative common regret median / mean / P75 / P95 / max is **{metrics['relative_common_regret']['median']:.6%} / {metrics['relative_common_regret']['mean']:.6%} / {metrics['relative_common_regret']['p75']:.6%} / {metrics['relative_common_regret']['p95']:.6%} / {metrics['relative_common_regret']['max']:.6%}**. This is an oracle upper bound on potential mechanical personalization, not an achieved algorithm benefit.

## Q6 — Universal near-oracle solution

At epsilon=0.001 the maximum shared coverage is **{common_001['maximum_subject_coverage']}/24** and a 24/24 universal near-oracle candidate exists: **{common_001['universal_near_oracle_candidate_exists']}**. The median near-oracle plateau contains **{metrics['near_oracle_count_epsilon_0_001']['median']:.0f}/625** candidates; the preregistered broad-plateau rule is **{metrics['broad_plateau_rule_triggered']}**.

## Q7 — Oracle transferability

Off-diagonal relative oracle-transfer regret median / P95 / max is **{metrics['off_diagonal_relative_oracle_transfer_regret']['median']:.6%} / {metrics['off_diagonal_relative_oracle_transfer_regret']['p95']:.6%} / {metrics['off_diagonal_relative_oracle_transfer_regret']['max']:.6%}**. The preregistered meaningful median-transfer-loss condition is **{metrics['meaningful_median_transfer_loss']}**.

## Q8 — Frozen subject-factor associations

The 30 preregistered exploratory Spearman tests produced **{len(significant)}** BH-q<0.05 associations. No predictive learner was trained. Any associations remain exploratory structural diagnostics, not causal or physiological parameter identification.

## Q9 — Fixed-ROM coordination-path personalization

Local sign counts (descent/equivalent/increase) are `{json.dumps(local_summary, sort_keys=True)}`. Across 1,200 global adjacent transitions, **{metrics['global_direction_disagreement_transition_count']} ({metrics['global_direction_disagreement_transition_fraction']:.3%})** show non-equivalent cross-subject direction disagreement under the frozen 1e-12 tolerance.

The formal decision is `{decision}` because the preregistered conjunction, not any single dramatic subject or metric, determines whether the fixed-ROM V3 task constitutes an algorithmically meaningful mechanical personalization problem.

## Q10 — Next branch

Proceed, if authorized, to `{branch}`. Do not execute it automatically.

## Integrity boundary

- Protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`; frozen before V3 development outcome reveal.
- V3-only numeric outputs were frozen before V2 result artifacts were opened by this execution.
- V3 truth landscape, candidate manifest/table, cohort, objective, normalization, and V2 frozen artifacts were read-only.
- Exactly 24 development subjects were read. `HELD_OUT_SCIENTIFIC_ACCESS_COUNT = 0`.
- No held-out replay/J/oracle/ranking/torque/beta statistic was accessed.
- No Five-parameter model, NN/PINN, BO, robot, hardware, human, or clinical stage was run.
"""
    atomic_text(OUTPUT / "MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md", report)


def write_checksums() -> None:
    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in paths) + "\n")


def analyze() -> None:
    started = time.perf_counter()
    if (OUTPUT / "V3_ANALYSIS_RESULT_FREEZE.json").exists():
        raise RuntimeError("V3 analysis result is already frozen; refusing to overwrite")
    truth, candidate_manifest, cohort, candidates = verify_inputs()
    protocol = read_json(PROTOCOL_PATH)
    held_out_audit = verify_held_out_without_loading(cohort, truth)
    development_ids = list(truth["development_subject_ids"])
    execution_freeze = {
        "freeze_id": "MYOLEG_V3_PERSONALIZATION_NECESSITY_ANALYSIS_EXECUTION_FREEZE_V1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "input_sha256": FROZEN_SHA,
        "development_subject_ids": development_ids,
        "held_out_subject_ids": list(HELD_OUT_IDS),
        "held_out_scientific_access_count_before_analysis": 0,
        "v2_result_artifacts_opened_before_v3_result_freeze": False,
    }
    atomic_json(OUTPUT / "ANALYSIS_EXECUTION_FREEZE.json", execution_freeze)

    store = DevelopmentTruthStore(truth, development_ids, candidates)
    j_rows: list[np.ndarray] = []
    for subject_id in development_ids:
        j_rows.append(np.asarray(store.load_subject(subject_id)["j_truth"], dtype=float))
    j_matrix = np.vstack(j_rows)
    candidate_ids = np.asarray([row["candidate_id"] for row in candidates])
    beta = np.asarray([[float(row["beta_flex"]), float(row["beta_extend"])] for row in candidates])

    oracle_indices, oracle_beta, oracle_j, oracle_rows, diversity_rows = oracle_and_diversity(
        development_ids, candidate_ids, beta, j_matrix,
    )
    rank_rows, spearman_matrix, kendall_matrix, top_rows = rank_and_topsets(development_ids, j_matrix)
    baselines, common_regret, relative_gap, gap_rows = common_baselines_and_gaps(
        development_ids, candidate_ids, beta, j_matrix, oracle_j,
    )
    decomposition, interaction = two_way_decomposition(j_matrix)
    multiplicative_rows = multiplicative_similarity(development_ids, j_matrix)
    local_rows = local_direction(development_ids, candidate_ids, j_matrix)
    global_rows = global_directional_agreement(j_matrix)
    transfer, transfer_regret, transfer_relative, transfer_rows, transfer_summary_rows = transfer_analysis(
        development_ids, candidate_ids, j_matrix, oracle_indices, oracle_j,
    )
    plateau_rows, common_near_rows = near_oracle_analysis(
        development_ids, candidate_ids, beta, j_matrix, oracle_j,
    )
    contrast_rows = landscape_contrast(development_ids, j_matrix, oracle_j)
    interaction_residual = np.sqrt(np.mean(interaction**2, axis=1))
    association_rows = parameter_associations(
        development_ids, cohort, oracle_beta, oracle_j, common_regret, interaction_residual,
    )
    metrics = decision_metrics(
        oracle_indices, oracle_beta, oracle_rows, diversity_rows, rank_rows, top_rows,
        relative_gap, decomposition, transfer_relative, common_near_rows, plateau_rows, global_rows,
    )
    decision = classify(metrics)

    write_csv(OUTPUT / "V3_DEV_ORACLE_SUMMARY.csv", oracle_rows)
    write_csv(OUTPUT / "V3_ORACLE_DIVERSITY.csv", diversity_rows)
    write_csv(OUTPUT / "V3_RANK_CORRELATION.csv", rank_rows)
    write_csv(OUTPUT / "V3_RANK_CORRELATION_MATRIX.csv", matrix_rows(development_ids, spearman_matrix, "spearman"))
    write_csv(OUTPUT / "V3_KENDALL_CORRELATION_MATRIX.csv", matrix_rows(development_ids, kendall_matrix, "kendall"))
    write_csv(OUTPUT / "V3_TOPSET_OVERLAP.csv", top_rows)
    atomic_json(OUTPUT / "V3_COMMON_BASELINE.json", baselines)
    write_csv(OUTPUT / "V3_PERSONALIZATION_GAP.csv", gap_rows)
    atomic_json(OUTPUT / "V3_TWO_WAY_VARIANCE_DECOMPOSITION.json", decomposition)
    write_csv(OUTPUT / "V3_MULTIPLICATIVE_SIMILARITY_AUDIT.csv", multiplicative_rows)
    write_csv(OUTPUT / "V3_LOCAL_DIRECTION_AUDIT.csv", local_rows)
    write_csv(OUTPUT / "V3_GLOBAL_DIRECTIONAL_AGREEMENT.csv", global_rows)
    write_csv(OUTPUT / "V3_ORACLE_TRANSFER_MATRIX.csv", transfer_rows)
    write_csv(OUTPUT / "V3_ORACLE_TRANSFER_SUMMARY.csv", transfer_summary_rows)
    write_csv(OUTPUT / "V3_NEAR_ORACLE_PLATEAU.csv", plateau_rows)
    write_csv(OUTPUT / "V3_COMMON_NEAR_ORACLE_ANALYSIS.csv", common_near_rows)
    write_csv(OUTPUT / "V3_SUBJECT_PARAMETER_ASSOCIATIONS.csv", association_rows)
    write_csv(OUTPUT / "V3_LANDSCAPE_CONTRAST.csv", contrast_rows)
    atomic_json(OUTPUT / "V3_DECISION_METRICS.json", metrics)

    create_v3_figures(
        protocol, development_ids, beta, oracle_beta, j_matrix, spearman_matrix,
        relative_gap, transfer_regret, cohort, interaction_residual,
    )
    held_out_audit.update({
        "development_subject_ids_scientifically_loaded": store.accessed,
        "development_subject_count_scientifically_loaded": len(store.accessed),
        "held_out_subject_ids_scientifically_loaded": [],
        "held_out_scientific_access_count": 0,
        "held_out_scientific_array_values_loaded": False,
    })
    atomic_json(HELD_OUT_AUDIT_PATH, held_out_audit)

    v3_only_paths = [
        OUTPUT / name for name in (
            "V3_DEV_ORACLE_SUMMARY.csv", "V3_ORACLE_DIVERSITY.csv",
            "V3_RANK_CORRELATION.csv", "V3_RANK_CORRELATION_MATRIX.csv",
            "V3_KENDALL_CORRELATION_MATRIX.csv", "V3_TOPSET_OVERLAP.csv",
            "V3_COMMON_BASELINE.json", "V3_PERSONALIZATION_GAP.csv",
            "V3_TWO_WAY_VARIANCE_DECOMPOSITION.json", "V3_MULTIPLICATIVE_SIMILARITY_AUDIT.csv",
            "V3_LOCAL_DIRECTION_AUDIT.csv", "V3_GLOBAL_DIRECTIONAL_AGREEMENT.csv",
            "V3_ORACLE_TRANSFER_MATRIX.csv", "V3_ORACLE_TRANSFER_SUMMARY.csv",
            "V3_NEAR_ORACLE_PLATEAU.csv", "V3_COMMON_NEAR_ORACLE_ANALYSIS.csv",
            "V3_SUBJECT_PARAMETER_ASSOCIATIONS.csv", "V3_LANDSCAPE_CONTRAST.csv",
            "V3_DECISION_METRICS.json",
        )
    ] + [FIGURES / name for name in (
        "01_oracle_beta_scatter.png", "02_common_mean_landscape.png",
        "03_preselected_subject_landscapes.png", "04_rank_correlation_heatmap.png",
        "05_common_regret_distribution.png", "06_oracle_transfer_heatmap.png",
        "08_subject_factor_path_diagnostic.png",
    )]
    result_freeze = freeze_v3_results(v3_only_paths, metrics, decision)

    comparison_rows, comparison = v2_v3_comparison(metrics)
    write_csv(OUTPUT / "V2_V3_PERSONALIZATION_COMPARISON.csv", comparison_rows)
    atomic_json(OUTPUT / "V2_V3_PERSONALIZATION_COMPARISON.json", comparison)
    create_comparison_figure(comparison)
    write_report(
        decision, metrics, baselines, oracle_rows, local_rows, global_rows,
        common_near_rows, association_rows, comparison,
    )

    metadata = {
        "stage_id": STAGE_ID, "outcome": decision,
        "recommended_next_stage": recommended_branch(decision),
        "next_stage_executed": False,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "analysis_execution_freeze_sha256": sha256_file(OUTPUT / "ANALYSIS_EXECUTION_FREEZE.json"),
        "v3_analysis_result_freeze_sha256": sha256_file(OUTPUT / "V3_ANALYSIS_RESULT_FREEZE.json"),
        "v3_truth_manifest_sha256": sha256_file(TRUTH_MANIFEST),
        "v3_candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
        "v3_candidate_table_sha256": sha256_file(CANDIDATE_TABLE),
        "cohort_manifest_sha256": sha256_file(COHORT_MANIFEST),
        "v2_frozen_artifact_sha256": comparison["v2_source_artifact_sha256"],
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "development_subject_count": len(store.accessed),
        "held_out_subject_count": len(HELD_OUT_IDS),
        "held_out_scientific_access_count": 0,
        "candidate_count": EXPECTED_CANDIDATES,
        "runtime_s": time.perf_counter() - started,
        "scope": {
            "offline_only": True, "models_trained": False, "nn_or_pinn": False,
            "bo_run": False, "robot_or_hardware": False, "human_or_clinical": False,
            "objective_modified": False, "normalization_modified": False,
            "parameterization_modified": False, "candidate_domain_modified": False,
        },
        "frozen_input_integrity": {
            "truth_unchanged": True, "candidate_manifest_unchanged": True,
            "candidate_table_unchanged": True, "cohort_unchanged": True,
            "v2_results_read_only": True,
        },
        "result_freeze_artifact_count": len(result_freeze["v3_only_artifact_sha256"]),
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    write_checksums()
    print(json.dumps({
        "outcome": decision, "recommended_next_stage": metadata["recommended_next_stage"],
        "protocol_sha256": metadata["protocol_sha256"],
        "distinct_oracles": metrics["unique_oracle_candidate_count"],
        "interaction_variance_percent": metrics["interaction_variance_percent"],
        "held_out_scientific_access_count": 0,
        "runtime_s": metadata["runtime_s"],
    }, indent=2))


def verify_only() -> None:
    truth, _, cohort, candidates = verify_inputs()
    if (OUTPUT / "metadata.json").exists():
        metadata = read_json(OUTPUT / "metadata.json")
        if metadata["held_out_scientific_access_count"] != 0:
            raise RuntimeError("held-out scientific access invariant failed")
    print(json.dumps({
        "inputs_verified": True, "development_subject_count": len(truth["development_subject_ids"]),
        "candidate_count": len(candidates), "held_out_subject_count": len(cohort["held_out_subject_ids"]),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }, indent=2))


def repair_figures() -> None:
    """Repair presentation only; fail if any frozen numeric artifact changes."""
    freeze_path = OUTPUT / "V3_ANALYSIS_RESULT_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("cannot repair figures before V3 result freeze")
    truth, _, cohort, candidates = verify_inputs()
    protocol = read_json(PROTOCOL_PATH)
    freeze = read_json(freeze_path)
    numeric_relative = [
        relative for relative in freeze["v3_only_artifact_sha256"]
        if not relative.startswith("figures/")
    ]
    numeric_before = {relative: sha256_file(OUTPUT / relative) for relative in numeric_relative}
    figure_before = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(FIGURES.glob("*.png"))
    }
    initial_freeze_sha = sha256_file(freeze_path)

    subject_ids = list(truth["development_subject_ids"])
    store = DevelopmentTruthStore(truth, subject_ids, candidates)
    j_matrix = np.vstack([store.load_subject(subject_id)["j_truth"] for subject_id in subject_ids])
    beta = np.asarray([[float(row["beta_flex"]), float(row["beta_extend"])] for row in candidates])
    candidate_ids = np.asarray([row["candidate_id"] for row in candidates])
    oracle_indices, oracle_beta, oracle_j, _, _ = oracle_and_diversity(subject_ids, candidate_ids, beta, j_matrix)
    _, spearman_matrix, _, _ = rank_and_topsets(subject_ids, j_matrix)
    _, _, relative_gap, _ = common_baselines_and_gaps(subject_ids, candidate_ids, beta, j_matrix, oracle_j)
    _, interaction = two_way_decomposition(j_matrix)
    interaction_residual = np.sqrt(np.mean(interaction**2, axis=1))
    _, transfer_regret, _, _, _ = transfer_analysis(subject_ids, candidate_ids, j_matrix, oracle_indices, oracle_j)
    create_v3_figures(
        protocol, subject_ids, beta, oracle_beta, j_matrix, spearman_matrix,
        relative_gap, transfer_regret, cohort, interaction_residual,
    )
    comparison = read_json(OUTPUT / "V2_V3_PERSONALIZATION_COMPARISON.json")
    create_comparison_figure(comparison)

    numeric_after = {relative: sha256_file(OUTPUT / relative) for relative in numeric_relative}
    if numeric_before != numeric_after:
        raise RuntimeError("visualization repair changed a frozen numeric artifact")
    figure_after = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(FIGURES.glob("*.png"))
    }
    for relative, digest in figure_after.items():
        if relative in freeze["v3_only_artifact_sha256"]:
            freeze["v3_only_artifact_sha256"][relative] = digest
    freeze["visualization_only_repair"] = {
        "scientific_numeric_artifact_changes": 0,
        "initial_freeze_sha256": initial_freeze_sha,
        "reason": "visual QA: boundary annotation and uninformative/overlapping color scales",
    }
    atomic_json(freeze_path, freeze)
    updated_freeze_sha = sha256_file(freeze_path)
    comparison["v3_result_freeze_sha256"] = updated_freeze_sha
    atomic_json(OUTPUT / "V2_V3_PERSONALIZATION_COMPARISON.json", comparison)
    repair = {
        "audit_id": "MYOLEG_V3_PERSONALIZATION_NECESSITY_VISUALIZATION_REPAIR_V1",
        "repair_scope": "presentation_only",
        "reason": [
            "oracle boundary count annotation was too close to plot edge",
            "constant-zero transfer matrix used an uninformative symmetric autoscale",
            "constant oracle beta color scale overlapped factor diagnostic panels",
        ],
        "numeric_artifact_sha256_before": numeric_before,
        "numeric_artifact_sha256_after": numeric_after,
        "numeric_artifacts_changed": False,
        "formal_decision_changed": False,
        "protocol_changed": False,
        "held_out_scientific_access_count": 0,
        "figure_sha256_before": figure_before,
        "figure_sha256_after": figure_after,
        "initial_v3_result_freeze_sha256": initial_freeze_sha,
        "updated_v3_result_freeze_sha256": updated_freeze_sha,
    }
    atomic_json(OUTPUT / "VISUALIZATION_REPAIR_AUDIT.json", repair)
    metadata = read_json(OUTPUT / "metadata.json")
    metadata["analysis_code_sha256"] = sha256_file(Path(__file__))
    metadata["v3_analysis_result_freeze_sha256"] = updated_freeze_sha
    metadata["visualization_repair_audit_sha256"] = sha256_file(OUTPUT / "VISUALIZATION_REPAIR_AUDIT.json")
    metadata["visualization_only_repair_numeric_changes"] = 0
    atomic_json(OUTPUT / "metadata.json", metadata)
    write_checksums()
    print(json.dumps({
        "visualization_repair": "PASS", "numeric_artifacts_changed": False,
        "formal_decision_changed": False, "held_out_scientific_access_count": 0,
        "updated_v3_result_freeze_sha256": updated_freeze_sha,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analyze", action="store_true")
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--repair-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze:
        analyze()
    elif args.repair_figures:
        repair_figures()
    else:
        verify_only()


if __name__ == "__main__":
    main()
