"""Audit whether the frozen MyoLeg-V2 objective supports personalization.

Held-out subjects are fail-closed: their shard bytes may be hashed, but their
NPZ arrays and every scientific outcome field are inaccessible in this stage.
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
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


STAGE_ID = "MYOLEG_V2_PERSONALIZATION_NECESSITY_AUDIT_V1"
PROTOCOL_ID = "MYOLEG_V2_PERSONALIZATION_NECESSITY_PROTOCOL_V1"
POLICY_ID = "HELD_OUT_TRUTH_ACCESS_POLICY_V1"
OUTCOME_SUPPORTED = "PERSONALIZATION_NECESSITY_SUPPORTED"
OUTCOME_WEAK = "PERSONALIZATION_NECESSITY_WEAK_OR_LIMITED"
OUTCOME_NOT_SUPPORTED = "PERSONALIZATION_NECESSITY_NOT_SUPPORTED"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1"
FIGURES = OUTPUT / "figures"
TRUTH_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"
CANDIDATE_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COHORT_MANIFEST_PATH = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
PROTOCOL_PATH = OUTPUT / "PERSONALIZATION_NECESSITY_PROTOCOL.json"
POLICY_PATH = OUTPUT / "HELD_OUT_TRUTH_ACCESS_POLICY_V1.json"
EXECUTION_FREEZE_PATH = OUTPUT / "ANALYSIS_EXECUTION_FREEZE.json"

FROZEN_SHA = {
    "truth_landscape_manifest": "4ea893b479099ebd39906f4b9bb140b6ba07ee58d74baadbd58b78113129f515",
    "candidate_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
}
HELD_OUT_IDS = (
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
)
REFERENCE_ID = "MYOLEG_V2_P15012"
EXPECTED_CANDIDATES = 16675
ORACLE_TIE_TOLERANCE = 1.0e-12
ALPHA_GRID_STEP = np.asarray([0.25, 0.25, 0.0025], dtype=float)
NEAR_ORACLE_EPSILONS = (1.0e-4, 5.0e-4, 1.0e-3)
TOP_FRACTIONS = (0.01, 0.05, 0.10)
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260830


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


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


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    actual = {
        "truth_landscape_manifest": sha256_file(TRUTH_MANIFEST_PATH),
        "candidate_manifest": sha256_file(CANDIDATE_MANIFEST_PATH),
        "cohort_manifest": sha256_file(COHORT_MANIFEST_PATH),
    }
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    truth = read_json(TRUTH_MANIFEST_PATH)
    candidates = read_json(CANDIDATE_MANIFEST_PATH)
    cohort = read_json(COHORT_MANIFEST_PATH)
    if not (
        truth["outcome"] == "MYOLEG_V2_TRUTH_LANDSCAPE_VALID"
        and truth["actual_row_count"] == 533600
        and truth["duplicate_pair_count"] == 0
        and truth["integrity_summary"]["all_pair_integrity_pass"] is True
        and len(candidates["ordered_included_candidates"]) == EXPECTED_CANDIDATES
        and len(cohort["development_subject_ids"]) == 24
        and tuple(cohort["held_out_subject_ids"]) == HELD_OUT_IDS
    ):
        raise RuntimeError("frozen audit identity/status mismatch")
    return truth, candidates, cohort


def select_representative_subjects(cohort: dict[str, Any]) -> list[dict[str, str]]:
    records = [row for row in cohort["subjects"] if row["split"] == "DEVELOPMENT"]
    vectors = {row["subject_id"]: np.asarray(row["unit_cube_vector"], dtype=float) for row in records}
    centroid = np.full(6, 0.5)
    first = min(records, key=lambda row: (float(np.linalg.norm(vectors[row["subject_id"]] - centroid)), row["subject_id"]))
    second = max(records, key=lambda row: (float(np.linalg.norm(vectors[row["subject_id"]] - centroid)), tuple(-ord(c) for c in row["subject_id"])))
    chosen = [first, second]
    while len(chosen) < 4:
        remaining = [row for row in records if row not in chosen]
        next_row = max(
            remaining,
            key=lambda row: (
                min(float(np.linalg.norm(vectors[row["subject_id"]] - vectors[item["subject_id"]])) for item in chosen),
                tuple(-ord(c) for c in row["subject_id"]),
            ),
        )
        chosen.append(next_row)
    roles = ("NEAREST_PARAMETER_CENTROID", "FARTHEST_PARAMETER_CENTROID", "MAXIMIN_3", "MAXIMIN_4")
    return [{"subject_id": row["subject_id"], "selection_role": role} for row, role in zip(chosen, roles)]


def verify_held_out_hashes_without_loading(truth: dict[str, Any], cohort: dict[str, Any]) -> dict[str, Any]:
    chunks = [row for row in truth["chunks"] if row["subject_id"] in HELD_OUT_IDS]
    if len(chunks) != 8 * 67 or sum(int(row["row_count"]) for row in chunks) != 8 * EXPECTED_CANDIDATES:
        raise RuntimeError("held-out manifest coverage mismatch")
    local_present = 0
    local_verified = 0
    local_bytes = 0
    for row in chunks:
        path = ROOT / row["path"]
        if path.is_file():
            local_present += 1
            local_bytes += path.stat().st_size
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"sealed held-out shard hash mismatch: {path}")
            local_verified += 1
    return {
        "sealed_subject_ids": list(HELD_OUT_IDS),
        "sealed_subject_count": 8,
        "manifest_chunk_count": len(chunks),
        "manifest_row_count": sum(int(row["row_count"]) for row in chunks),
        "manifest_subject_landscape_sha256": {subject_id: truth["subject_landscape_sha256"][subject_id] for subject_id in HELD_OUT_IDS},
        "local_shard_count_present": local_present,
        "local_shard_count_sha256_verified": local_verified,
        "local_shard_bytes_hashed_without_np_load": local_bytes,
        "scientific_array_values_loaded": False,
        "existing_post_freeze_oracle_summary_opened": False,
        "existing_subject_landscape_summary_opened": False,
    }


def protocol_payload(candidates: dict[str, Any], cohort: dict[str, Any]) -> dict[str, Any]:
    rows = candidates["ordered_included_candidates"]
    alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    alpha_min = np.min(alpha, axis=0)
    alpha_max = np.max(alpha, axis=0)
    top_counts = {f"top_{int(fraction * 100)}_percent": int(math.ceil(EXPECTED_CANDIDATES * fraction)) for fraction in TOP_FRACTIONS}
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_development_truth_reveal": True,
        "scientific_role": "development-only offline oracle upper-bound audit under the frozen normalized torque objective",
        "input_sha256": FROZEN_SHA,
        "analysis_population": {
            "development_subject_ids": list(cohort["development_subject_ids"]),
            "development_count": 24,
            "held_out_subject_ids": list(HELD_OUT_IDS),
            "held_out_count": 8,
            "candidate_count": EXPECTED_CANDIDATES,
            "held_out_scientific_values_allowed": False,
        },
        "oracle": {
            "rule": "minimum float64 J; candidates within tie tolerance are equivalent; lowest original proposal_index wins",
            "tie_tolerance_j": ORACLE_TIE_TOLERANCE,
        },
        "alpha_geometry": {
            "order": ["alpha_hip_deg", "alpha_knee_deg", "alpha_phase"],
            "grid_step": ALPHA_GRID_STEP.tolist(),
            "admitted_min": alpha_min.tolist(),
            "admitted_max": alpha_max.tolist(),
            "normalization_range": (alpha_max - alpha_min).tolist(),
            "normalized_distance": "sqrt(mean(((alpha_i-alpha_j)/(admitted_max-admitted_min))^2))",
            "exact_same_oracle": "same candidate_id",
            "immediate_grid_neighbor": "not exact and maximum absolute integer grid-index difference <= 1",
            "clearly_separated": "neither exact nor immediate-grid-neighbor",
        },
        "rank_similarity": {"subject_pair_count": 276, "metrics": ["Spearman rank correlation", "Kendall tau-b"]},
        "top_sets": {
            "fractions": list(TOP_FRACTIONS), "counts": top_counts,
            "selection": "exact K ordered by (J, original proposal_index); no outcome-dependent K change",
            "metrics": ["Jaccard", "overlap coefficient"],
        },
        "common_baselines": {
            "primary": "DEV_MEAN_OPTIMAL_COMMON: argmin candidate of mean development J",
            "secondary": "DEV_WORSTCASE_OPTIMAL_COMMON: argmin candidate of maximum development J",
            "tie_tolerance_j": ORACLE_TIE_TOLERANCE,
            "tie_break": "lowest original proposal_index",
            "reference_candidate_id": REFERENCE_ID,
        },
        "personalization_gap": {
            "absolute": "J_i(DEV_MEAN_OPTIMAL_COMMON)-J_i(subject_oracle)",
            "relative": "absolute/J_i(DEV_MEAN_OPTIMAL_COMMON)",
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "interval": "percentile 95% CI", "statistics": ["mean", "median"]},
        },
        "engineering_effect_bands_relative_j": [
            {"label": "NEGLIGIBLE", "lower_inclusive": 0.0, "upper_exclusive": 0.001},
            {"label": "VERY_SMALL", "lower_inclusive": 0.001, "upper_exclusive": 0.005},
            {"label": "SMALL", "lower_inclusive": 0.005, "upper_exclusive": 0.01},
            {"label": "MODERATE_ENGINEERING_SEPARATION", "lower_inclusive": 0.01, "upper_exclusive": 0.02},
            {"label": "LARGER_ENGINEERING_SEPARATION", "lower_inclusive": 0.02, "upper_exclusive": None},
        ],
        "effect_band_language": "descriptive normalized-objective engineering bands only; not clinical, comfort, or safety thresholds",
        "transfer": "TRANSFER[i,j]=development subject i truth J at development subject j oracle; diagonal own oracle; all off-diagonal metrics frozen",
        "candidate_variability": ["mean", "sample SD", "CV=SD/abs(mean)", "min", "max", "range"],
        "landscape_contrast": {"top5_to_oracle_spread": "J at deterministic Top-5% rank K minus oracle J"},
        "near_oracle": {
            "epsilons_j": list(NEAR_ORACLE_EPSILONS),
            "subject_metrics": ["candidate count", "per-axis min/max/range", "normalized bounding-box diagonal"],
            "broad_plateau_rule": "median count at epsilon=0.001 >= ceil(10% of 16675)=1668",
            "common_coverage_thresholds": [24, 22, 18],
        },
        "parameter_associations": {
            "frozen_parameter_count": 6,
            "outcomes": ["oracle alpha hip", "oracle alpha knee", "oracle alpha phase", "oracle J", "common regret"],
            "method": "descriptive Spearman; raw p and Benjamini-Hochberg q across 30 exploratory tests; no predictive model",
        },
        "decision_logic": {
            "supported_all_required": {
                "unique_oracle_candidates_at_least": 4,
                "clearly_separated_oracle_pair_fraction_at_least": 0.50,
                "median_relative_common_regret_at_least": 0.005,
                "p75_relative_common_regret_at_least": 0.01,
                "rank_or_topset_subject_dependence": "median Spearman <=0.95 OR median Top-5% Jaccard <=0.75",
                "max_common_near_oracle_coverage_at_epsilon_0.001_below": 22,
            },
            "not_supported_all_required": {
                "unique_oracle_candidates_at_most": 2,
                "median_relative_common_regret_below": 0.001,
                "p95_relative_common_regret_below": 0.005,
                "median_spearman_at_least": 0.98,
                "universal_24_of_24_near_oracle_at_epsilon_0.001": True,
            },
            "otherwise": OUTCOME_WEAK,
            "classification": "operational research rule frozen before development outcomes; not a clinical threshold",
        },
        "representative_landscape_subjects": select_representative_subjects(cohort),
        "figures": [
            "oracle alpha scatter/distributions", "development rank-correlation heatmap",
            "common-vs-oracle regret distribution", "oracle cross-transfer regret heatmap",
            "near-oracle count by subject and epsilon", "phase=0 landscape slices for frozen geometry-selected subjects",
        ],
        "scope_guards": {
            "five_parameter_training": False, "nn_or_pinn": False, "bo": False,
            "candidate_domain_modification": False, "cohort_modification": False,
            "objective_modification": False, "robot_or_hardware": False,
            "human_or_clinical_claim": False,
        },
    }


def freeze_protocol() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists; refusing overwrite: {OUTPUT}")
    truth, candidates, cohort = verify_inputs()
    held_out_audit = verify_held_out_hashes_without_loading(truth, cohort)
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir()
    protocol = protocol_payload(candidates, cohort)
    policy = {
        "policy_id": POLICY_ID,
        "stage_id": STAGE_ID,
        "effective_until": "final algorithm freeze and a separately authorized confirmatory evaluation",
        "classification": "SEALED_CONFIRMATORY_TRUTH",
        **held_out_audit,
        "allowed_operations": ["manifest identity check", "raw file existence/size check", "streaming SHA-256 check", "manifest row-count check"],
        "forbidden_operations": [
            "np.load on a held-out shard", "held-out candidate J read", "held-out oracle/rank/alpha/J read",
            "held-out landscape plot or statistic", "held-out outcome use in metric/threshold/model selection",
        ],
        "forbidden_existing_summary_files_for_this_stage": [
            "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/POST_FREEZE_ORACLE_SUMMARY.csv",
            "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/SUBJECT_LANDSCAPE_SUMMARY.csv",
        ],
        "enforcement": "DevelopmentTruthStore rejects every subject not in the frozen 24-development allowlist before path resolution or np.load",
    }
    atomic_json(PROTOCOL_PATH, protocol)
    atomic_json(POLICY_PATH, policy)
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "held_out_policy_sha256": sha256_file(POLICY_PATH), "held_out_hashes_verified_without_loading": held_out_audit["local_shard_count_sha256_verified"]}, indent=2))


class DevelopmentTruthStore:
    def __init__(self, truth: dict[str, Any], candidates: dict[str, Any], development_ids: Iterable[str]):
        self.truth = truth
        self.candidates = candidates["ordered_included_candidates"]
        self.allowed = frozenset(development_ids)
        if len(self.allowed) != 24 or self.allowed.intersection(HELD_OUT_IDS):
            raise RuntimeError("development truth allowlist invalid")
        self.accessed_subject_ids: list[str] = []

    def load_subject(self, subject_id: str) -> dict[str, np.ndarray]:
        if subject_id not in self.allowed:
            raise PermissionError(f"SEALED_CONFIRMATORY_TRUTH access denied before path resolution: {subject_id}")
        chunks = sorted(
            (row for row in self.truth["chunks"] if row["subject_id"] == subject_id),
            key=lambda row: int(row["candidate_start_rank"]),
        )
        if len(chunks) != 67 or sum(int(row["row_count"]) for row in chunks) != EXPECTED_CANDIDATES:
            raise RuntimeError(f"development chunk coverage mismatch: {subject_id}")
        columns: dict[str, list[np.ndarray]] = {}
        required = ("candidate_id", "proposal_index", "alpha_hip_deg", "alpha_knee_deg", "alpha_phase", "j_truth", "integrity_status")
        for chunk in chunks:
            path = ROOT / chunk["path"]
            if not path.is_file() or sha256_file(path) != chunk["sha256"]:
                raise RuntimeError(f"development shard unavailable or hash invalid: {path}")
            with np.load(path, allow_pickle=False) as shard:
                for key in required:
                    columns.setdefault(key, []).append(np.asarray(shard[key]))
        output = {key: np.concatenate(values) for key, values in columns.items()}
        if len(output["j_truth"]) != EXPECTED_CANDIDATES or not np.all(output["integrity_status"] == 1) or not np.isfinite(output["j_truth"]).all():
            raise RuntimeError(f"development truth integrity failure: {subject_id}")
        expected_ids = np.asarray([row["candidate_id"] for row in self.candidates])
        if not np.array_equal(output["candidate_id"], expected_ids):
            raise RuntimeError(f"candidate identity mismatch: {subject_id}")
        self.accessed_subject_ids.append(subject_id)
        return output


def select_tied_min(values: np.ndarray, proposal: np.ndarray, tolerance: float = ORACLE_TIE_TOLERANCE) -> int:
    minimum = float(np.min(values))
    eligible = np.flatnonzero(values <= minimum + tolerance)
    return int(eligible[np.argmin(proposal[eligible])])


def summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)), "median": float(np.median(array)), "mean": float(np.mean(array)),
        "p5": float(np.percentile(array, 5)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
    }


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
    samples = np.asarray(values, dtype=float)[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))]
    means = np.mean(samples, axis=1)
    medians = np.median(samples, axis=1)
    return {
        "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
        "mean_95_percentile_ci": np.percentile(means, [2.5, 97.5]).tolist(),
        "median_95_percentile_ci": np.percentile(medians, [2.5, 97.5]).tolist(),
    }


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def analyze() -> None:
    started = time.perf_counter()
    truth, candidate_manifest, cohort = verify_inputs()
    protocol = read_json(PROTOCOL_PATH)
    policy = read_json(POLICY_PATH)
    if not protocol.get("frozen_before_development_truth_reveal") or policy.get("scientific_array_values_loaded") is not False:
        raise RuntimeError("protocol/policy was not frozen before development truth reveal")
    if any(path.exists() for path in (OUTPUT / "DEV_ORACLE_SUMMARY.csv", OUTPUT / "DEV_COMMON_BASELINES.json")):
        raise RuntimeError("development outcomes already exist; refusing post-outcome rerun")
    atomic_json(EXECUTION_FREEZE_PATH, {
        "freeze_id": "MYOLEG_V2_PERSONALIZATION_NECESSITY_ANALYSIS_EXECUTION_FREEZE_V1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "held_out_policy_sha256": sha256_file(POLICY_PATH),
        "development_truth_values_read_at_freeze_time": False, "held_out_truth_values_read": False,
    })

    development_ids = list(protocol["analysis_population"]["development_subject_ids"])
    records = {row["subject_id"]: row for row in cohort["subjects"]}
    store = DevelopmentTruthStore(truth, candidate_manifest, development_ids)
    loaded = [store.load_subject(subject_id) for subject_id in development_ids]
    j_matrix = np.vstack([row["j_truth"] for row in loaded])
    candidate_ids = np.asarray(loaded[0]["candidate_id"])
    proposal = np.asarray(loaded[0]["proposal_index"], dtype=int)
    alpha = np.column_stack((loaded[0]["alpha_hip_deg"], loaded[0]["alpha_knee_deg"], loaded[0]["alpha_phase"]))
    if not all(np.array_equal(row["candidate_id"], candidate_ids) and np.array_equal(row["proposal_index"], proposal) for row in loaded):
        raise RuntimeError("development candidates are not identical")
    alpha_min = np.asarray(protocol["alpha_geometry"]["admitted_min"], dtype=float)
    alpha_max = np.asarray(protocol["alpha_geometry"]["admitted_max"], dtype=float)
    alpha_range = alpha_max - alpha_min
    reference_index = int(np.flatnonzero(candidate_ids == REFERENCE_ID)[0])
    if np.max(np.abs(j_matrix[:, reference_index] - 1.0)) > 1.0e-12:
        raise RuntimeError("development reference normalization changed")

    oracle_indices = np.asarray([select_tied_min(row, proposal) for row in j_matrix], dtype=int)
    oracle_j = j_matrix[np.arange(24), oracle_indices]
    oracle_alpha = alpha[oracle_indices]
    oracle_rows = []
    for index, subject_id in enumerate(development_ids):
        at_lower = np.isclose(oracle_alpha[index], alpha_min, atol=1.0e-15)
        at_upper = np.isclose(oracle_alpha[index], alpha_max, atol=1.0e-15)
        oracle_rows.append({
            "subject_id": subject_id, "candidate_id": str(candidate_ids[oracle_indices[index]]),
            "proposal_index": int(proposal[oracle_indices[index]]),
            "alpha_hip_deg": float(oracle_alpha[index, 0]), "alpha_knee_deg": float(oracle_alpha[index, 1]), "alpha_phase": float(oracle_alpha[index, 2]),
            "oracle_j": float(oracle_j[index]), "equivalent_float64_min_count": int(np.sum(j_matrix[index] <= oracle_j[index] + ORACLE_TIE_TOLERANCE)),
            "hip_lower_edge": bool(at_lower[0]), "hip_upper_edge": bool(at_upper[0]),
            "knee_lower_edge": bool(at_lower[1]), "knee_upper_edge": bool(at_upper[1]),
            "phase_lower_edge": bool(at_lower[2]), "phase_upper_edge": bool(at_upper[2]),
            "any_candidate_domain_boundary": bool(np.any(at_lower | at_upper)),
        })
    write_csv(OUTPUT / "DEV_ORACLE_SUMMARY.csv", oracle_rows)

    diversity_rows = []
    classifications = []
    for i in range(24):
        for j in range(i + 1, 24):
            difference = np.abs(oracle_alpha[i] - oracle_alpha[j])
            normalized = float(np.sqrt(np.mean((difference / alpha_range) ** 2)))
            grid_difference = np.rint(difference / ALPHA_GRID_STEP).astype(int)
            if oracle_indices[i] == oracle_indices[j]:
                classification = "EXACT_SAME_ORACLE"
            elif int(np.max(grid_difference)) <= 1:
                classification = "IMMEDIATE_GRID_NEIGHBOR"
            else:
                classification = "CLEARLY_SEPARATED"
            classifications.append(classification)
            diversity_rows.append({
                "subject_i": development_ids[i], "subject_j": development_ids[j],
                "abs_delta_hip_deg": float(difference[0]), "abs_delta_knee_deg": float(difference[1]), "abs_delta_phase": float(difference[2]),
                "normalized_3d_alpha_distance": normalized, "hip_grid_steps": int(grid_difference[0]),
                "knee_grid_steps": int(grid_difference[1]), "phase_grid_steps": int(grid_difference[2]), "classification": classification,
            })
    write_csv(OUTPUT / "DEV_ORACLE_DIVERSITY.csv", diversity_rows)

    ranks = np.vstack([stats.rankdata(row, method="average") for row in j_matrix])
    spearman_matrix = np.corrcoef(ranks)
    kendall_matrix = np.eye(24)
    rank_rows = []
    for i in range(24):
        for j in range(i + 1, 24):
            kendall = float(stats.kendalltau(j_matrix[i], j_matrix[j], variant="b").statistic)
            kendall_matrix[i, j] = kendall_matrix[j, i] = kendall
            rank_rows.append({"subject_i": development_ids[i], "subject_j": development_ids[j], "spearman_rank_correlation": float(spearman_matrix[i, j]), "kendall_tau_b": kendall})
    write_csv(OUTPUT / "DEV_RANK_CORRELATION.csv", rank_rows)

    top_sets: dict[float, list[set[int]]] = {}
    top_rows = []
    for fraction in TOP_FRACTIONS:
        count = int(math.ceil(EXPECTED_CANDIDATES * fraction))
        sets = [set(np.lexsort((proposal, row))[:count].tolist()) for row in j_matrix]
        top_sets[fraction] = sets
        for i in range(24):
            for j in range(i + 1, 24):
                intersection = len(sets[i].intersection(sets[j]))
                union = len(sets[i].union(sets[j]))
                top_rows.append({
                    "top_fraction": fraction, "top_count": count, "subject_i": development_ids[i], "subject_j": development_ids[j],
                    "intersection_count": intersection, "jaccard": intersection / union, "overlap_coefficient": intersection / count,
                })
    write_csv(OUTPUT / "DEV_TOPSET_OVERLAP.csv", top_rows)

    candidate_mean = np.mean(j_matrix, axis=0)
    candidate_max = np.max(j_matrix, axis=0)
    common_mean_index = select_tied_min(candidate_mean, proposal)
    common_worst_index = select_tied_min(candidate_max, proposal)
    common_j = j_matrix[:, common_mean_index]
    common_regret = common_j - oracle_j
    relative_gap = common_regret / common_j
    gap_rows = []
    for index, subject_id in enumerate(development_ids):
        gap_rows.append({
            "subject_id": subject_id, "oracle_j": float(oracle_j[index]), "dev_mean_common_j": float(common_j[index]),
            "common_regret": float(common_regret[index]), "relative_personalization_gap": float(relative_gap[index]),
            "engineering_effect_band": effect_band(float(relative_gap[index])),
        })
    write_csv(OUTPUT / "DEV_PERSONALIZATION_GAP.csv", gap_rows)
    baselines = {
        "DEV_MEAN_OPTIMAL_COMMON": {
            "candidate_id": str(candidate_ids[common_mean_index]), "proposal_index": int(proposal[common_mean_index]), "alpha": alpha[common_mean_index].tolist(),
            "mean_j": float(np.mean(common_j)), "median_j": float(np.median(common_j)), "worst_j": float(np.max(common_j)),
        },
        "DEV_WORSTCASE_OPTIMAL_COMMON": {
            "candidate_id": str(candidate_ids[common_worst_index]), "proposal_index": int(proposal[common_worst_index]), "alpha": alpha[common_worst_index].tolist(),
            "mean_j": float(np.mean(j_matrix[:, common_worst_index])), "median_j": float(np.median(j_matrix[:, common_worst_index])), "worst_j": float(np.max(j_matrix[:, common_worst_index])),
        },
        "REFERENCE": {"candidate_id": REFERENCE_ID, "alpha": alpha[reference_index].tolist(), "j_all_development": 1.0},
        "common_regret_summary": summary(common_regret), "relative_gap_summary": summary(relative_gap),
        "relative_gap_bootstrap": bootstrap_intervals(relative_gap),
    }
    atomic_json(OUTPUT / "DEV_COMMON_BASELINES.json", baselines)

    transfer_rows = []
    transfer_j = j_matrix[:, oracle_indices]
    transfer_regret = transfer_j - oracle_j[:, None]
    for i, recipient in enumerate(development_ids):
        for j, donor in enumerate(development_ids):
            transfer_rows.append({
                "recipient_subject_id": recipient, "donor_oracle_subject_id": donor,
                "donor_oracle_candidate_id": str(candidate_ids[oracle_indices[j]]), "transfer_j": float(transfer_j[i, j]),
                "regret_vs_recipient_oracle": float(transfer_regret[i, j]), "relative_regret_vs_transfer_j": float(transfer_regret[i, j] / transfer_j[i, j]),
                "own_oracle_diagonal": i == j, "within_0_001_j_of_own_oracle": bool(transfer_j[i, j] <= oracle_j[i] + 0.001),
            })
    write_csv(OUTPUT / "DEV_ORACLE_TRANSFER_MATRIX.csv", transfer_rows)

    candidate_sd = np.std(j_matrix, axis=0, ddof=1)
    candidate_min = np.min(j_matrix, axis=0)
    candidate_maximum = np.max(j_matrix, axis=0)
    variability_rows = []
    for index in range(EXPECTED_CANDIDATES):
        variability_rows.append({
            "candidate_id": str(candidate_ids[index]), "proposal_index": int(proposal[index]),
            "alpha_hip_deg": float(alpha[index, 0]), "alpha_knee_deg": float(alpha[index, 1]), "alpha_phase": float(alpha[index, 2]),
            "mean_j": float(candidate_mean[index]), "sample_sd_j": float(candidate_sd[index]), "cv_j": float(candidate_sd[index] / abs(candidate_mean[index])),
            "min_j": float(candidate_min[index]), "max_j": float(candidate_maximum[index]), "range_j": float(candidate_maximum[index] - candidate_min[index]),
            "is_reference": index == reference_index, "is_dev_mean_common": index == common_mean_index,
        })
    write_csv(OUTPUT / "DEV_CANDIDATE_VARIABILITY.csv", variability_rows)

    top5_count = int(math.ceil(EXPECTED_CANDIDATES * 0.05))
    contrast_rows = []
    for index, subject_id in enumerate(development_ids):
        ordered = np.lexsort((proposal, j_matrix[index]))
        contrast_rows.append({
            "subject_id": subject_id, "oracle_j": float(oracle_j[index]), "reference_j": 1.0,
            "reference_to_oracle_improvement": float(1.0 - oracle_j[index]),
            "median_landscape_j": float(np.median(j_matrix[index])),
            "landscape_min_j": float(np.min(j_matrix[index])), "landscape_max_j": float(np.max(j_matrix[index])),
            "landscape_dynamic_range": float(np.max(j_matrix[index]) - np.min(j_matrix[index])),
            "top5_boundary_j": float(j_matrix[index, ordered[top5_count - 1]]),
            "top5_to_oracle_spread": float(j_matrix[index, ordered[top5_count - 1]] - oracle_j[index]),
        })
    write_csv(OUTPUT / "DEV_LANDSCAPE_CONTRAST.csv", contrast_rows)

    near_rows = []
    common_near_rows = []
    near_masks: dict[float, np.ndarray] = {}
    for epsilon in NEAR_ORACLE_EPSILONS:
        mask = j_matrix <= oracle_j[:, None] + epsilon
        near_masks[epsilon] = mask
        for index, subject_id in enumerate(development_ids):
            selected = alpha[mask[index]]
            lower = np.min(selected, axis=0)
            upper = np.max(selected, axis=0)
            ranges = upper - lower
            near_rows.append({
                "subject_id": subject_id, "epsilon_j": epsilon, "near_oracle_candidate_count": len(selected),
                "near_oracle_fraction": len(selected) / EXPECTED_CANDIDATES,
                "alpha_hip_min": float(lower[0]), "alpha_hip_max": float(upper[0]), "alpha_hip_range": float(ranges[0]),
                "alpha_knee_min": float(lower[1]), "alpha_knee_max": float(upper[1]), "alpha_knee_range": float(ranges[1]),
                "alpha_phase_min": float(lower[2]), "alpha_phase_max": float(upper[2]), "alpha_phase_range": float(ranges[2]),
                "normalized_alpha_bbox_diagonal": float(np.sqrt(np.mean((ranges / alpha_range) ** 2))),
            })
        coverage = np.sum(mask, axis=0)
        max_coverage = int(np.max(coverage))
        best = np.flatnonzero(coverage == max_coverage)
        best_index = int(best[np.argmin(proposal[best])])
        common_near_rows.append({
            "epsilon_j": epsilon, "maximum_subject_coverage": max_coverage,
            "best_coverage_candidate_id": str(candidate_ids[best_index]), "best_coverage_proposal_index": int(proposal[best_index]),
            "best_coverage_alpha_hip_deg": float(alpha[best_index, 0]), "best_coverage_alpha_knee_deg": float(alpha[best_index, 1]), "best_coverage_alpha_phase": float(alpha[best_index, 2]),
            "candidate_count_coverage_24": int(np.sum(coverage >= 24)), "candidate_count_coverage_at_least_22": int(np.sum(coverage >= 22)),
            "candidate_count_coverage_at_least_18": int(np.sum(coverage >= 18)),
            "universal_near_oracle_candidate_exists": bool(np.any(coverage == 24)),
        })
    write_csv(OUTPUT / "DEV_NEAR_ORACLE_PLATEAU.csv", near_rows)
    write_csv(OUTPUT / "DEV_COMMON_NEAR_ORACLE_ANALYSIS.csv", common_near_rows)

    parameter_names = tuple(next(iter(records.values()))["factor_values"])
    parameter_matrix = np.asarray([[records[subject_id]["factor_values"][name] for name in parameter_names] for subject_id in development_ids], dtype=float)
    outcome_names = ("oracle_alpha_hip_deg", "oracle_alpha_knee_deg", "oracle_alpha_phase", "oracle_j", "common_regret")
    outcomes = np.column_stack((oracle_alpha, oracle_j, common_regret))
    association_rows = []
    p_values = []
    for p_index, parameter in enumerate(parameter_names):
        for o_index, outcome_name in enumerate(outcome_names):
            parameter_values = parameter_matrix[:, p_index]
            outcome_values = outcomes[:, o_index]
            undefined_reasons = []
            if np.ptp(parameter_values) == 0.0:
                undefined_reasons.append("CONSTANT_PARAMETER")
            if np.ptp(outcome_values) == 0.0:
                undefined_reasons.append("CONSTANT_OUTCOME")
            if undefined_reasons:
                rho = None
                raw_p_value = 1.0
            else:
                result = stats.spearmanr(parameter_values, outcome_values)
                rho = float(result.statistic)
                raw_p_value = float(result.pvalue)
            p_values.append(raw_p_value)
            association_rows.append({
                "parameter": parameter, "outcome": outcome_name, "spearman_rho": rho,
                "raw_p_value": raw_p_value, "undefined_reason": ";".join(undefined_reasons),
                "exploratory_only": True, "predictive_model_trained": False,
            })
    q_values = bh_qvalues(np.asarray(p_values))
    for row, q_value in zip(association_rows, q_values):
        row["bh_q_value_across_30_tests"] = float(q_value)
    write_csv(OUTPUT / "DEV_PARAMETER_ORACLE_ASSOCIATIONS.csv", association_rows)

    metrics = calculate_decision_metrics(
        oracle_indices, oracle_alpha, diversity_rows, classifications, rank_rows, top_rows,
        relative_gap, common_near_rows, near_rows, transfer_regret,
    )
    decision = classify(metrics)
    create_figures(protocol, development_ids, oracle_alpha, oracle_j, spearman_matrix, relative_gap, transfer_regret, near_rows, alpha, j_matrix)
    write_report(decision, metrics, baselines, oracle_rows, rank_rows, top_rows, common_near_rows, association_rows, contrast_rows, variability_rows, transfer_regret)
    access_audit = {
        "development_subject_ids_loaded": store.accessed_subject_ids,
        "development_subject_count_loaded": len(store.accessed_subject_ids),
        "held_out_subject_ids_loaded": [], "held_out_scientific_values_read": False,
        "held_out_policy_sha256": sha256_file(POLICY_PATH),
        "forbidden_existing_oracle_or_subject_summary_opened": False,
    }
    atomic_json(OUTPUT / "TRUTH_ACCESS_AUDIT.json", access_audit)
    metadata = {
        "stage_id": STAGE_ID, "outcome": decision, "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "held_out_policy_sha256": sha256_file(POLICY_PATH), "analysis_code_sha256": sha256_file(Path(__file__)),
        "truth_landscape_manifest_sha256": FROZEN_SHA["truth_landscape_manifest"],
        "development_subject_count": 24, "held_out_subject_count": 8, "held_out_truth_revealed": False,
        "candidate_count": EXPECTED_CANDIDATES, "runtime_s": time.perf_counter() - started,
        "scope": {"offline_only": True, "models_trained": False, "bo_run": False, "robot_or_hardware": False},
    }
    atomic_json(OUTPUT / "metadata.json", metadata)
    write_checksums()
    print(json.dumps({"outcome": decision, "protocol_sha256": sha256_file(PROTOCOL_PATH), "development_subjects": 24, "held_out_truth_revealed": False, "runtime_s": metadata["runtime_s"]}, indent=2))


def calculate_decision_metrics(
    oracle_indices: np.ndarray, oracle_alpha: np.ndarray, diversity_rows: list[dict[str, Any]], classifications: list[str],
    rank_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]], relative_gap: np.ndarray,
    common_near_rows: list[dict[str, Any]], near_rows: list[dict[str, Any]], transfer_regret: np.ndarray,
) -> dict[str, Any]:
    spearman = np.asarray([row["spearman_rank_correlation"] for row in rank_rows])
    kendall = np.asarray([row["kendall_tau_b"] for row in rank_rows])
    top5 = np.asarray([row["jaccard"] for row in top_rows if row["top_fraction"] == 0.05])
    diversity_distance = np.asarray([row["normalized_3d_alpha_distance"] for row in diversity_rows])
    epsilon_common = next(row for row in common_near_rows if row["epsilon_j"] == 0.001)
    near_001 = np.asarray([row["near_oracle_candidate_count"] for row in near_rows if row["epsilon_j"] == 0.001])
    off_diagonal = transfer_regret[~np.eye(24, dtype=bool)]
    return {
        "unique_oracle_candidate_count": int(len(np.unique(oracle_indices))),
        "unique_oracle_alpha_hip_count": int(len(np.unique(oracle_alpha[:, 0]))),
        "unique_oracle_alpha_knee_count": int(len(np.unique(oracle_alpha[:, 1]))),
        "unique_oracle_alpha_phase_count": int(len(np.unique(oracle_alpha[:, 2]))),
        "oracle_pair_classification_counts": {label: classifications.count(label) for label in ("EXACT_SAME_ORACLE", "IMMEDIATE_GRID_NEIGHBOR", "CLEARLY_SEPARATED")},
        "clearly_separated_oracle_pair_fraction": classifications.count("CLEARLY_SEPARATED") / len(classifications),
        "normalized_oracle_alpha_distance": summary(diversity_distance),
        "spearman_rank_correlation": summary(spearman), "kendall_tau_b": summary(kendall),
        "top5_jaccard": summary(top5), "relative_common_regret": summary(relative_gap),
        "max_common_near_oracle_coverage_epsilon_0_001": int(epsilon_common["maximum_subject_coverage"]),
        "universal_near_oracle_epsilon_0_001": bool(epsilon_common["universal_near_oracle_candidate_exists"]),
        "near_oracle_count_epsilon_0_001": summary(near_001),
        "broad_plateau_rule_triggered": bool(np.median(near_001) >= 1668),
        "off_diagonal_oracle_transfer_regret": summary(off_diagonal),
    }


def classify(metrics: dict[str, Any]) -> str:
    supported = bool(
        metrics["unique_oracle_candidate_count"] >= 4
        and metrics["clearly_separated_oracle_pair_fraction"] >= 0.50
        and metrics["relative_common_regret"]["median"] >= 0.005
        and metrics["relative_common_regret"]["p75"] >= 0.01
        and (metrics["spearman_rank_correlation"]["median"] <= 0.95 or metrics["top5_jaccard"]["median"] <= 0.75)
        and metrics["max_common_near_oracle_coverage_epsilon_0_001"] < 22
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


def create_figures(
    protocol: dict[str, Any], subject_ids: list[str], oracle_alpha: np.ndarray, oracle_j: np.ndarray,
    spearman: np.ndarray, relative_gap: np.ndarray, transfer_regret: np.ndarray,
    near_rows: list[dict[str, Any]], alpha: np.ndarray, j_matrix: np.ndarray,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    scatter = axes[0].scatter(oracle_alpha[:, 0], oracle_alpha[:, 1], c=oracle_alpha[:, 2], cmap="coolwarm", s=42, edgecolors="black", linewidths=0.4)
    axes[0].set(xlabel="Hip amplitude delta (deg)", ylabel="Knee amplitude delta (deg)", title="Oracle alpha")
    fig.colorbar(scatter, ax=axes[0], label="Phase shift")
    for axis, values, label in zip(axes[1:], oracle_alpha.T, ("Hip delta (deg)", "Knee delta (deg)", "Phase shift")):
        axis.hist(values, bins=min(12, len(np.unique(values))), color="#4c78a8", edgecolor="white")
        axis.set(xlabel=label, ylabel="Subjects")
    save_figure(FIGURES / "oracle_alpha_distribution.png")

    plt.figure(figsize=(8.2, 7.2))
    image = plt.imshow(spearman, vmin=np.min(spearman), vmax=1.0, cmap="viridis", aspect="equal")
    plt.colorbar(image, label="Spearman rank correlation")
    plt.xticks(range(24), subject_ids, rotation=90, fontsize=6)
    plt.yticks(range(24), subject_ids, fontsize=6)
    plt.title("Development-subject candidate-ranking similarity")
    save_figure(FIGURES / "development_rank_correlation_heatmap.png")

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(relative_gap * 100.0, bins=12, color="#f58518", edgecolor="white")
    for value, label in ((0.1, "0.1%"), (0.5, "0.5%"), (1.0, "1%"), (2.0, "2%")):
        plt.axvline(value, color="black", linewidth=0.7, linestyle="--")
        plt.text(value, plt.ylim()[1] * 0.94, label, rotation=90, va="top", ha="right", fontsize=7)
    plt.xlabel("Common trajectory regret relative to common J (%)")
    plt.ylabel("Development subjects")
    plt.title("Oracle upper-bound personalization opportunity")
    save_figure(FIGURES / "common_vs_oracle_regret_distribution.png")

    plt.figure(figsize=(8.2, 7.2))
    image = plt.imshow(transfer_regret, cmap="magma", aspect="equal")
    plt.colorbar(image, label="Regret versus recipient oracle (J)")
    plt.xticks(range(24), subject_ids, rotation=90, fontsize=6)
    plt.yticks(range(24), subject_ids, fontsize=6)
    plt.xlabel("Donor oracle subject")
    plt.ylabel("Recipient subject")
    plt.title("Development oracle cross-transfer")
    save_figure(FIGURES / "oracle_cross_transfer_heatmap.png")

    plt.figure(figsize=(10.5, 4.4))
    x = np.arange(24)
    for epsilon, marker in zip(NEAR_ORACLE_EPSILONS, ("o", "s", "^")):
        values = [next(row["near_oracle_candidate_count"] for row in near_rows if row["subject_id"] == subject_id and row["epsilon_j"] == epsilon) for subject_id in subject_ids]
        plt.plot(x, values, marker=marker, linewidth=1.2, markersize=3.5, label=f"epsilon={epsilon:g}")
    plt.xticks(x, subject_ids, rotation=90, fontsize=7)
    plt.ylabel("Near-oracle candidate count")
    plt.xlabel("Development subject")
    plt.title("Near-oracle plateau size")
    plt.legend(frameon=False)
    save_figure(FIGURES / "near_oracle_candidate_count.png")

    selected = protocol["representative_landscape_subjects"]
    phase_zero = np.isclose(alpha[:, 2], 0.0, atol=1.0e-15)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), sharex=True, sharey=True)
    for axis, selected_row in zip(axes.ravel(), selected):
        index = subject_ids.index(selected_row["subject_id"])
        scatter = axis.scatter(alpha[phase_zero, 0], alpha[phase_zero, 1], c=j_matrix[index, phase_zero], cmap="viridis", s=18)
        axis.set_title(f"{selected_row['subject_id']}\n{selected_row['selection_role']}", fontsize=9)
        axis.set_xlabel("Hip amplitude delta (deg)")
        axis.set_ylabel("Knee amplitude delta (deg)")
        fig.colorbar(scatter, ax=axis, label="J truth")
    fig.suptitle("Preregistered development landscape slices at phase shift = 0", fontsize=11)
    save_figure(FIGURES / "representative_landscape_slices.png")


def write_report(
    decision: str, metrics: dict[str, Any], baselines: dict[str, Any], oracle_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]], common_near_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]], contrast_rows: list[dict[str, Any]], variability_rows: list[dict[str, Any]],
    transfer_regret: np.ndarray,
) -> None:
    boundary = sum(bool(row["any_candidate_domain_boundary"]) for row in oracle_rows) / 24
    hip_lower = sum(bool(row["hip_lower_edge"]) for row in oracle_rows) / 24
    hip_upper = sum(bool(row["hip_upper_edge"]) for row in oracle_rows) / 24
    knee_lower = sum(bool(row["knee_lower_edge"]) for row in oracle_rows) / 24
    knee_upper = sum(bool(row["knee_upper_edge"]) for row in oracle_rows) / 24
    phase_edge = sum(bool(row["phase_lower_edge"] or row["phase_upper_edge"]) for row in oracle_rows) / 24
    significant_associations = [row for row in association_rows if row["bh_q_value_across_30_tests"] < 0.05]
    top_variability = sorted(variability_rows, key=lambda row: (-row["range_j"], row["proposal_index"]))[:5]
    common_near_001 = next(row for row in common_near_rows if row["epsilon_j"] == 0.001)
    foreign = transfer_regret[~np.eye(24, dtype=bool)]
    report = f"""# MyoLeg V2 Personalization Necessity Audit V1

## Decision

`{decision}`

This is a development-only offline oracle upper-bound audit under the frozen normalized torque objective. It is not achieved algorithm performance, a patient optimum, human evidence, clinical benefit, comfort, or safety evidence. All eight confirmatory subjects remained sealed.

## Development oracle geometry

- Distinct exact oracle candidates: **{metrics['unique_oracle_candidate_count']} / 24**.
- Unique hip / knee / phase oracle values: **{metrics['unique_oracle_alpha_hip_count']} / {metrics['unique_oracle_alpha_knee_count']} / {metrics['unique_oracle_alpha_phase_count']}**.
- Any candidate-domain boundary fraction: **{boundary:.3f}**.
- Hip lower / upper edge fractions: **{hip_lower:.3f} / {hip_upper:.3f}**.
- Knee lower / upper edge fractions: **{knee_lower:.3f} / {knee_upper:.3f}**.
- Phase edge fraction: **{phase_edge:.3f}**.
- Clearly separated oracle-pair fraction: **{metrics['clearly_separated_oracle_pair_fraction']:.3f}**.

Different exact candidate IDs are not by themselves evidence of useful personalization. The pair classifications and near-oracle plateaus below determine whether the differences exceed grid/tie effects.

## Ranking and top sets

- Pairwise Spearman: median **{metrics['spearman_rank_correlation']['median']:.6f}**, mean **{metrics['spearman_rank_correlation']['mean']:.6f}**, range **[{metrics['spearman_rank_correlation']['min']:.6f}, {metrics['spearman_rank_correlation']['max']:.6f}]**.
- Kendall tau-b: median **{metrics['kendall_tau_b']['median']:.6f}**.
- Top-5% Jaccard: median **{metrics['top5_jaccard']['median']:.6f}**, P5/P95 **{metrics['top5_jaccard']['p5']:.6f}/{metrics['top5_jaccard']['p95']:.6f}**.

## Common trajectory versus subject oracle

- DEV_MEAN_OPTIMAL_COMMON: `{baselines['DEV_MEAN_OPTIMAL_COMMON']['candidate_id']}`, alpha `{baselines['DEV_MEAN_OPTIMAL_COMMON']['alpha']}`.
- Common J mean / median / worst: **{baselines['DEV_MEAN_OPTIMAL_COMMON']['mean_j']:.9f} / {baselines['DEV_MEAN_OPTIMAL_COMMON']['median_j']:.9f} / {baselines['DEV_MEAN_OPTIMAL_COMMON']['worst_j']:.9f}**.
- Relative common regret median / mean / P75 / P95 / max: **{metrics['relative_common_regret']['median']:.6%} / {metrics['relative_common_regret']['mean']:.6%} / {metrics['relative_common_regret']['p75']:.6%} / {metrics['relative_common_regret']['p95']:.6%} / {metrics['relative_common_regret']['max']:.6%}**.

These gaps are oracle upper bounds on potential subject-specific mechanical-objective benefit, not achieved few-trial improvement.

## Near-oracle plateau and transfer

- At epsilon=0.001, maximum common coverage: **{common_near_001['maximum_subject_coverage']} / 24**.
- Universal near-oracle candidate exists: **{common_near_001['universal_near_oracle_candidate_exists']}**.
- Near-oracle count at epsilon=0.001, median / P95 / max: **{metrics['near_oracle_count_epsilon_0_001']['median']:.0f} / {metrics['near_oracle_count_epsilon_0_001']['p95']:.0f} / {metrics['near_oracle_count_epsilon_0_001']['max']:.0f}**.
- Broad plateau rule triggered: **{metrics['broad_plateau_rule_triggered']}**.
- Foreign-oracle regret median / P95 / max: **{float(np.median(foreign)):.9f} / {float(np.percentile(foreign,95)):.9f} / {float(np.max(foreign)):.9f}** J.

## Parameter associations

Thirty preregistered descriptive Spearman tests were run across six frozen parameters and five oracle/gap outcomes. Associations with BH q<0.05: **{len(significant_associations)}**. These are exploratory associations only; no predictive model was trained and no cohort or range was changed.

## Interpretation boundary

The decision follows the protocol frozen before development truth reveal. Held-out shards were only stream-hashed and row-count checked; no held-out NPZ array, oracle row, J value, ranking, figure, or statistic was read. No Five-parameter model, NN/PINN, BO, candidate-domain update, objective update, cohort update, robot, or hardware operation occurred.
"""
    atomic_text(OUTPUT / "MYOLEG_V2_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md", report)


def write_checksums() -> None:
    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in paths]
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(lines) + "\n")


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
