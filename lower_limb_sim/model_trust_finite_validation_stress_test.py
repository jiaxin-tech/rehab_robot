"""Frozen, default-off helpers for finite-validation model-trust stress testing.

This module contains no virtual-truth generator.  It only freezes candidate
identities, defines preregistered comparators, and computes post-freeze metrics.
The executable stage lives in
``run_model_trust_finite_validation_stress_test.py`` and reuses the frozen V1
sequential evaluator rather than implementing a second Top-3 policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STAGE_ID = "MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_V1"
PROTOCOL_ID = "STRESS_TEST_PROTOCOL_V1"
BASELINE_MANIFEST_ID = "FROZEN_BASELINE_MANIFEST_V1"
DEFAULT_ENABLED = False
OFFLINE_ONLY = "OFFLINE_ONLY"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"
N_RANDOM_REPEATS = 100
RANDOM_BASE_SEED = 20260828
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_REPEATS = 20_000
NEAR_OPTIMAL_TOLERANCES = (0.001, 0.0025, 0.005)
MACHINE_COMPARISON_TOLERANCE = 1e-12

BASELINES = (
    "B0_REFERENCE",
    "B1_MODEL_ONLY",
    "B2_RANDOM3_FINITE_VALIDATION",
    "B3_MODEL_TOP1_VALIDATION",
    "B4_FROZEN_TOP3_SEQUENTIAL",
    "B5_ORACLE",
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def deterministic_seed(case_id: str, repeat: int) -> int:
    """Derive a stable uint32 seed without Python's randomized hash()."""

    payload = f"{RANDOM_BASE_SEED}|{case_id}|{int(repeat)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _forbidden_selection_columns(columns: Sequence[str]) -> list[str]:
    forbidden = []
    for column in columns:
        lowered = str(column).lower()
        if lowered in {"j_pred", "j_truth"} or "truth" in lowered:
            forbidden.append(str(column))
    return forbidden


def geometry_candidate_universe(parameter_lattice: pd.DataFrame) -> pd.DataFrame:
    """Return the shared non-zero geometry-valid identity universe.

    No prediction or truth column is accepted.  Reference remains a separate
    fallback and therefore is not sampled as a candidate.
    """

    forbidden = _forbidden_selection_columns(parameter_lattice.columns)
    if forbidden:
        raise PermissionError(
            f"candidate-universe construction received forbidden columns: {forbidden}"
        )
    required = {
        "trajectory_id",
        "trajectory_sha256",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "geometrically_admissible",
    }
    missing = required.difference(parameter_lattice.columns)
    if missing:
        raise ValueError(f"candidate universe missing columns: {sorted(missing)}")
    finite = np.isfinite(
        parameter_lattice.loc[
            :, ["hip_delta", "knee_delta", "phase_delta"]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    neutral = (
        np.isclose(parameter_lattice["hip_delta"].to_numpy(dtype=float), 0.0)
        & np.isclose(parameter_lattice["knee_delta"].to_numpy(dtype=float), 0.0)
        & np.isclose(parameter_lattice["phase_delta"].to_numpy(dtype=float), 0.0)
    )
    universe = parameter_lattice.loc[
        finite
        & ~neutral
        & parameter_lattice["geometrically_admissible"].astype(bool).to_numpy(),
        [
            "trajectory_id",
            "trajectory_sha256",
            "hip_delta",
            "knee_delta",
            "phase_delta",
            "geometrically_admissible",
        ],
    ].copy()
    if universe.empty or universe["trajectory_id"].astype(str).duplicated().any():
        raise RuntimeError("shared geometry-valid candidate universe is invalid")
    return universe.sort_values("trajectory_id", kind="mergesort").reset_index(
        drop=True
    )


def freeze_random3_candidates(
    identity_universe: pd.DataFrame,
    *,
    case_id: str,
    repeats: int = N_RANDOM_REPEATS,
) -> pd.DataFrame:
    """Freeze deterministic Random-3 identities without prediction or truth."""

    forbidden = _forbidden_selection_columns(identity_universe.columns)
    if forbidden:
        raise PermissionError(
            f"Random-3 sampling received forbidden columns: {forbidden}"
        )
    if repeats < 30:
        raise ValueError("formal Random-3 baseline requires at least 30 repeats")
    if len(identity_universe) < 3:
        raise ValueError("Random-3 universe contains fewer than three candidates")
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for repeat in range(repeats):
        seed = deterministic_seed(str(case_id), repeat)
        generator = np.random.default_rng(seed)
        indices = generator.choice(len(identity_universe), size=3, replace=False)
        selected = identity_universe.iloc[indices]
        for ordinal, (_, candidate) in enumerate(selected.iterrows(), start=1):
            logical_order += 1
            rows.append(
                {
                    "case_id": str(case_id),
                    "baseline_id": "B2_RANDOM3_FINITE_VALIDATION",
                    "random_repeat": int(repeat),
                    "seed": int(seed),
                    "candidate_ordinal": ordinal,
                    "candidate_id": f"R{ordinal}",
                    "trajectory_id": str(candidate["trajectory_id"]),
                    "trajectory_sha256": str(candidate["trajectory_sha256"]),
                    "hip_delta": float(candidate["hip_delta"]),
                    "knee_delta": float(candidate["knee_delta"]),
                    "phase_delta": float(candidate["phase_delta"]),
                    "prediction_rank": None,
                    "freeze_logical_order_within_case": logical_order,
                    "freeze_timestamp": f"PRETRUTH_LOGICAL_T{logical_order:06d}",
                    "truth_read_before_freeze": False,
                    "sampling_uses_J_pred": False,
                    "sampling_uses_J_truth": False,
                }
            )
    output = pd.DataFrame(rows)
    counts = output.groupby("random_repeat")["trajectory_id"].nunique()
    if len(counts) != repeats or not counts.eq(3).all():
        raise RuntimeError("Random-3 freeze did not produce three unique candidates")
    return output


@dataclass
class PersistedTruthGate:
    """Fail closed until both preregistered JSON files are persisted."""

    protocol_sha256: str = ""
    baseline_manifest_sha256: str = ""
    persisted: bool = False
    opened: bool = False

    def mark_persisted(
        self, *, protocol_sha256: str, baseline_manifest_sha256: str
    ) -> None:
        if self.opened:
            raise RuntimeError("cannot change frozen manifests after truth opens")
        if len(protocol_sha256) != 64 or len(baseline_manifest_sha256) != 64:
            raise ValueError("persisted manifest SHA-256 is invalid")
        self.protocol_sha256 = str(protocol_sha256)
        self.baseline_manifest_sha256 = str(baseline_manifest_sha256)
        self.persisted = True

    def authorize_truth(self) -> str:
        if not self.persisted:
            raise PermissionError(
                "STRESS_TEST_PROTOCOL_INTEGRITY = FAIL: truth requested before freeze"
            )
        payload = (
            f"{self.protocol_sha256}|{self.baseline_manifest_sha256}|TRUTH_OPEN"
        )
        self.opened = True
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def final_regret(final_j: float, oracle_j: float) -> float:
    regret = float(final_j) - float(oracle_j)
    if regret < -MACHINE_COMPARISON_TOLERANCE:
        raise ValueError("final regret cannot be below the oracle")
    return max(regret, 0.0)


def select_model_only(c1: Mapping[str, Any]) -> dict[str, Any]:
    """Return C1 without reference fallback, even if it is harmful."""

    return dict(c1)


def select_validated_with_reference(candidates: pd.DataFrame) -> dict[str, Any]:
    """Select truth minimum from Reference plus frozen measured candidates."""

    required = {"trajectory_id", "J_truth"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"validated candidates missing columns: {sorted(missing)}")
    reference = {
        "trajectory_id": "REFERENCE",
        "J_truth": 1.0,
        "hip_delta": 0.0,
        "knee_delta": 0.0,
        "phase_delta": 0.0,
    }
    records = [reference, *candidates.to_dict(orient="records")]
    return min(records, key=lambda row: (float(row["J_truth"]), str(row["trajectory_id"])))


def empirical_p95(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("P95 requires non-empty finite values")
    return float(np.quantile(array, 0.95, method="linear"))


def bootstrap_mean_ci(
    paired_differences: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    values = np.asarray(paired_differences, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires non-empty finite differences")
    generator = np.random.default_rng(seed)
    # Chunking bounds peak memory while preserving deterministic samples.
    means: list[np.ndarray] = []
    remaining = int(repeats)
    while remaining:
        count = min(remaining, 2_000)
        samples = generator.choice(values, size=(count, len(values)), replace=True)
        means.append(samples.mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    return (
        float(np.quantile(distribution, 0.025, method="linear")),
        float(np.quantile(distribution, 0.975, method="linear")),
    )


def near_optimal_rates(regrets: Sequence[float]) -> dict[str, float]:
    values = np.asarray(regrets, dtype=float)
    return {
        f"near_optimal_at_{tolerance:g}": float(
            np.mean(values <= tolerance + MACHINE_COMPARISON_TOLERANCE)
        )
        for tolerance in NEAR_OPTIMAL_TOLERANCES
    }


def truth_rank_percentile(
    truth_landscape: pd.DataFrame, trajectory_id: str
) -> tuple[int, float]:
    ranked = truth_landscape.sort_values(
        ["J_truth", "trajectory_id"], kind="mergesort"
    ).reset_index(drop=True)
    positions = np.flatnonzero(
        ranked["trajectory_id"].astype(str).to_numpy() == str(trajectory_id)
    )
    if len(positions) != 1:
        raise RuntimeError("candidate identity is not unique in truth landscape")
    rank = int(positions[0]) + 1
    return rank, 100.0 * rank / len(ranked)


__all__ = [
    "BASELINE_MANIFEST_ID",
    "BASELINES",
    "BOOTSTRAP_REPEATS",
    "BOOTSTRAP_SEED",
    "DEFAULT_ENABLED",
    "MACHINE_COMPARISON_TOLERANCE",
    "NEAR_OPTIMAL_TOLERANCES",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_APPROVED",
    "N_RANDOM_REPEATS",
    "OFFLINE_ONLY",
    "PROTOCOL_ID",
    "PersistedTruthGate",
    "RANDOM_BASE_SEED",
    "STAGE_ID",
    "bootstrap_mean_ci",
    "canonical_json_bytes",
    "deterministic_seed",
    "empirical_p95",
    "final_regret",
    "freeze_random3_candidates",
    "geometry_candidate_universe",
    "near_optimal_rates",
    "select_model_only",
    "select_validated_with_reference",
    "truth_rank_percentile",
]
