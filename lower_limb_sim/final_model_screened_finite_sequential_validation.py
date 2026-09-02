"""Default-off model-screened finite sequential validation research prototype.

The module deliberately separates prediction-only shortlist freezing from
virtual-truth access.  It does not import or call P2 Explore/Exploit, bundle, or
adaptive-horizon policies, and it contains no hardware boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE


METHOD_ID = "MODEL_SCREENED_FINITE_SEQUENTIAL_VALIDATION_V1"
MANIFEST_ID = "FINAL_METHOD_MANIFEST_V1"
MAX_MODEL_SCREENED_CANDIDATES = 3
MAX_VALIDATION_TRIALS = 3
DEFAULT_ENABLED = False
OFFLINE_ONLY = "OFFLINE_ONLY"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"
BEST_VALIDATED_TRAJECTORY = "BEST_VALIDATED_TRAJECTORY"
SHORTLIST_SELECTION_RULE = (
    "ascending_J_pred_then_trajectory_id_one_representative_per_frozen_"
    "0.005_predicted_objective_equivalence_band"
)
SUPPORT_ROLE = "DATA_PROVENANCE_AND_APPLICABILITY_NOT_RELIABILITY_SCORE"
TRUTH_ROLE = "POST_SHORTLIST_FREEZE_VIRTUAL_VALIDATION_ONLY"
FINAL_STATUS_RULE = {
    "supported": (
        "B2_mean_J_no_worse_than_B1_and_B2_mean_regret_no_worse_than_B1_and_"
        "zero_false_improvements_and_all_truth_global_optima_in_shortlist_and_"
        "best_validated_full_alpha_diversity_exceeds_old_bundle5"
    ),
    "supported_with_limitations": (
        "B2_mean_J_no_worse_than_B1_and_B2_mean_regret_no_worse_than_B1_and_"
        "best_validated_full_alpha_diversity_exceeds_old_bundle5"
    ),
    "not_supported": "otherwise",
    "comparison_tolerance": "machine_scale_1e-12_not_a_scientific_threshold",
}


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


def _alpha_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        round(float(row["hip_delta"]), 12),
        round(float(row["knee_delta"]), 12),
        round(float(row["phase_delta"]), 12),
    )


def _truth_columns(columns: Sequence[str]) -> list[str]:
    return [column for column in columns if "truth" in str(column).lower()]


@dataclass(frozen=True)
class FrozenCandidate:
    shortlist_ordinal: int
    trajectory_id: str
    hip_delta: float
    knee_delta: float
    phase_delta: float
    initial_prediction_rank: int
    predicted_equivalence_band: int
    initial_J_pred: float
    initial_domain_coverage: float
    geometrically_admissible: bool
    model_supported: bool
    trajectory_sha256: str

    @property
    def alpha_key(self) -> tuple[float, float, float]:
        return (
            round(self.hip_delta, 12),
            round(self.knee_delta, 12),
            round(self.phase_delta, 12),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "shortlist_ordinal": self.shortlist_ordinal,
            "candidate_id": f"C{self.shortlist_ordinal}",
            "trajectory_id": self.trajectory_id,
            "hip_delta": self.hip_delta,
            "knee_delta": self.knee_delta,
            "phase_delta": self.phase_delta,
            "initial_prediction_rank": self.initial_prediction_rank,
            "predicted_equivalence_band": self.predicted_equivalence_band,
            "initial_J_pred": self.initial_J_pred,
            "initial_domain_coverage": self.initial_domain_coverage,
            "geometrically_admissible": self.geometrically_admissible,
            "model_supported": self.model_supported,
            "trajectory_sha256": self.trajectory_sha256,
        }


@dataclass(frozen=True)
class FrozenShortlist:
    case_id: str
    candidates: tuple[FrozenCandidate, ...]
    freeze_token: str
    selection_rule: str = SHORTLIST_SELECTION_RULE
    max_candidates: int = MAX_MODEL_SCREENED_CANDIDATES
    truth_read_before_freeze: bool = False

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(candidate.trajectory_id for candidate in self.candidates)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "freeze_token": self.freeze_token,
            "selection_rule": self.selection_rule,
            "max_candidates": self.max_candidates,
            "candidate_count": len(self.candidates),
            "truth_read_before_freeze": self.truth_read_before_freeze,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def assert_complete_candidate_trajectory(
    trajectory: pd.DataFrame, *, expected_trajectory_id: str
) -> None:
    """Fail closed unless the offline object is one complete trajectory.

    ``formal_execution_allowed`` is intentionally *not* promoted here: the
    generator keeps it false for personalized offline candidates because real
    robot execution remains separately unapproved.
    """

    required = {
        "trajectory_id",
        "time_s",
        "q_hip_rad",
        "q_knee_rad",
        "theta_shank_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "trajectory_sample_valid",
        "formal_execution_allowed",
        "closure_valid",
        "rom_valid",
        "workspace_valid",
        "jacobian_valid",
        "force_mapping_valid",
        "velocity_valid",
        "acceleration_valid",
        "asymmetry_valid",
        "finite_valid",
    }
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(f"candidate trajectory missing columns: {sorted(missing)}")
    if len(trajectory) < 3:
        raise ValueError("candidate execution must contain a complete trajectory")
    identities = set(trajectory["trajectory_id"].astype(str))
    if identities != {str(expected_trajectory_id)}:
        raise PermissionError("candidate trajectory identity differs from freeze token")
    time_s = trajectory["time_s"].to_numpy(dtype=float)
    if not np.isfinite(time_s).all() or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("candidate trajectory time must be finite and increasing")
    if not float(time_s[-1]) > float(time_s[0]):
        raise ValueError("candidate trajectory has no whole-cycle duration")
    non_domain_gates = (
        "closure_valid",
        "rom_valid",
        "workspace_valid",
        "jacobian_valid",
        "force_mapping_valid",
        "velocity_valid",
        "acceleration_valid",
        "asymmetry_valid",
        "finite_valid",
    )
    failed = [
        column
        for column in non_domain_gates
        if not trajectory[column].astype(bool).all()
    ]
    if failed:
        raise ValueError(f"candidate fails non-domain geometry gates: {failed}")
    expected_theta = trajectory["q_hip_rad"].to_numpy(dtype=float) - trajectory[
        "q_knee_rad"
    ].to_numpy(dtype=float)
    if not np.allclose(
        trajectory["theta_shank_rad"].to_numpy(dtype=float),
        expected_theta,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("theta_shank must equal q_hip - q_knee")


def freeze_model_screened_shortlist(
    prediction_map: pd.DataFrame,
    *,
    case_id: str,
    max_candidates: int = MAX_MODEL_SCREENED_CANDIDATES,
) -> FrozenShortlist:
    """Freeze at most three deterministic candidates before any truth column exists.

    The 0.005 objective-equivalence tolerance is reused only to avoid selecting
    three numerically adjacent predictions from the same formal equivalence
    band.  It is not changed, tuned, or interpreted as a reliability score.
    """

    if max_candidates < 0 or max_candidates > MAX_MODEL_SCREENED_CANDIDATES:
        raise ValueError("shortlist candidate count exceeds the frozen maximum")
    truth = _truth_columns(list(prediction_map.columns))
    if truth:
        raise PermissionError(
            f"shortlist must freeze before truth columns are attached: {truth}"
        )
    required = {
        "trajectory_id",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "J_pred",
        "domain_coverage",
        "model_supported",
        "geometrically_admissible",
    }
    missing = required.difference(prediction_map.columns)
    if missing:
        raise ValueError(f"prediction map missing shortlist columns: {sorted(missing)}")
    if prediction_map["trajectory_id"].astype(str).duplicated().any():
        raise ValueError("prediction map contains duplicate trajectory identities")
    alpha = prediction_map.loc[:, ["hip_delta", "knee_delta", "phase_delta"]]
    if alpha.duplicated().any():
        raise ValueError("prediction map contains duplicate alpha points")
    finite = np.isfinite(
        prediction_map.loc[:, ["hip_delta", "knee_delta", "phase_delta", "J_pred"]]
        .to_numpy(dtype=float)
    ).all(axis=1)
    neutral = (
        np.isclose(prediction_map["hip_delta"].to_numpy(dtype=float), 0.0)
        & np.isclose(prediction_map["knee_delta"].to_numpy(dtype=float), 0.0)
        & np.isclose(prediction_map["phase_delta"].to_numpy(dtype=float), 0.0)
    )
    eligible = prediction_map.loc[
        finite
        & ~neutral
        & prediction_map["geometrically_admissible"].astype(bool).to_numpy()
        & prediction_map["model_supported"].astype(bool).to_numpy()
        & prediction_map["J_pred"].lt(1.0).to_numpy()
    ].sort_values(["J_pred", "trajectory_id"], kind="mergesort")
    eligible = eligible.copy()
    eligible["initial_prediction_rank"] = np.arange(1, len(eligible) + 1)

    selected_rows: list[pd.Series] = []
    selected_bands: set[int] = set()
    if not eligible.empty and max_candidates:
        minimum = float(eligible.iloc[0]["J_pred"])
        for _, row in eligible.iterrows():
            difference = max(0.0, float(row["J_pred"]) - minimum)
            band = int(
                math.floor(
                    (difference + 1e-14) / OBJECTIVE_EQUIVALENCE_TOLERANCE
                )
            )
            if band in selected_bands:
                continue
            selected_bands.add(band)
            copy = row.copy()
            copy["predicted_equivalence_band"] = band
            selected_rows.append(copy)
            if len(selected_rows) == max_candidates:
                break

    candidates: list[FrozenCandidate] = []
    for ordinal, row in enumerate(selected_rows, start=1):
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=float(row["hip_delta"]),
            knee_amplitude_delta_deg=float(row["knee_delta"]),
            knee_phase_shift=float(row["phase_delta"]),
        )
        trajectory = generated.trajectory.copy(deep=True)
        trajectory_id = str(row["trajectory_id"])
        trajectory["trajectory_id"] = trajectory_id
        assert_complete_candidate_trajectory(
            trajectory, expected_trajectory_id=trajectory_id
        )
        candidates.append(
            FrozenCandidate(
                shortlist_ordinal=ordinal,
                trajectory_id=trajectory_id,
                hip_delta=float(row["hip_delta"]),
                knee_delta=float(row["knee_delta"]),
                phase_delta=float(row["phase_delta"]),
                initial_prediction_rank=int(row["initial_prediction_rank"]),
                predicted_equivalence_band=int(row["predicted_equivalence_band"]),
                initial_J_pred=float(row["J_pred"]),
                initial_domain_coverage=float(row["domain_coverage"]),
                geometrically_admissible=bool(row["geometrically_admissible"]),
                model_supported=bool(row["model_supported"]),
                trajectory_sha256=str(generated.metadata["trajectory_sha256"]),
            )
        )
    token_payload = {
        "method_id": METHOD_ID,
        "case_id": str(case_id),
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "selection_rule": SHORTLIST_SELECTION_RULE,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "max_candidates": max_candidates,
        "candidates": [candidate.as_dict() for candidate in candidates],
        "truth_read_before_freeze": False,
    }
    token = hashlib.sha256(canonical_json_bytes(token_payload)).hexdigest()
    return FrozenShortlist(
        case_id=str(case_id), candidates=tuple(candidates), freeze_token=token
    )


def rerank_remaining_frozen_candidates(
    shortlist: FrozenShortlist,
    prediction_map: pd.DataFrame,
    *,
    executed_trajectory_ids: Sequence[str],
) -> pd.DataFrame:
    """Rank only unexecuted members of an immutable shortlist."""

    truth = _truth_columns(list(prediction_map.columns))
    if truth:
        raise PermissionError("reranking must not receive truth columns")
    executed = {str(value) for value in executed_trajectory_ids}
    frozen = set(shortlist.trajectory_ids)
    if not executed.issubset(frozen):
        raise PermissionError("executed candidate is outside the frozen shortlist")
    remaining = frozen - executed
    rows = prediction_map.loc[
        prediction_map["trajectory_id"].astype(str).isin(remaining)
    ].copy()
    if len(rows) != len(remaining) or set(rows["trajectory_id"].astype(str)) != remaining:
        raise RuntimeError("full prediction map does not contain the frozen shortlist")
    ordinal = {
        candidate.trajectory_id: candidate.shortlist_ordinal
        for candidate in shortlist.candidates
    }
    rows["shortlist_ordinal"] = rows["trajectory_id"].astype(str).map(ordinal)
    rows = rows.sort_values(["J_pred", "trajectory_id"], kind="mergesort")
    rows["current_frozen_rank"] = np.arange(1, len(rows) + 1)
    rows["candidate_addition_allowed"] = False
    return rows.reset_index(drop=True)


class FrozenShortlistTruthGate:
    """Authorize virtual truth exactly once per persisted frozen candidate."""

    def __init__(
        self,
        shortlist: FrozenShortlist,
        *,
        global_manifest_sha256: str,
        manifest_persisted: bool,
    ) -> None:
        if not manifest_persisted:
            raise PermissionError("candidate truth requires a persisted method manifest")
        if len(global_manifest_sha256) != 64:
            raise ValueError("global manifest SHA-256 is invalid")
        self.shortlist = shortlist
        self.global_manifest_sha256 = str(global_manifest_sha256)
        self._executed: set[str] = set()
        self._pending: tuple[str, str] | None = None

    @property
    def executed(self) -> tuple[str, ...]:
        return tuple(sorted(self._executed))

    def authorize(self, trajectory_id: str) -> str:
        identifier = str(trajectory_id)
        if self._pending is not None:
            raise RuntimeError("only one complete candidate may be pending")
        if identifier not in set(self.shortlist.trajectory_ids):
            raise PermissionError("new candidate cannot enter the frozen shortlist")
        if identifier in self._executed:
            raise PermissionError("frozen candidate can be executed only once")
        payload = (
            f"{self.global_manifest_sha256}|{self.shortlist.freeze_token}|"
            f"{identifier}|{len(self._executed) + 1}"
        )
        token = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._pending = (identifier, token)
        return token

    def complete(self, trajectory_id: str, token: str) -> None:
        expected = (str(trajectory_id), str(token))
        if self._pending != expected:
            raise PermissionError("candidate truth completion token mismatch")
        self._executed.add(str(trajectory_id))
        self._pending = None


def select_best_validated(validated: pd.DataFrame) -> pd.Series:
    """Select only by measured/virtual-truth whole-trajectory J."""

    required = {"trajectory_id", "validated_J", "validation_role"}
    missing = required.difference(validated.columns)
    if missing:
        raise ValueError(f"validated table missing columns: {sorted(missing)}")
    if "J_pred" in validated.columns:
        raise PermissionError("final selection must not use prediction columns")
    if validated.empty or not np.isfinite(validated["validated_J"]).all():
        raise ValueError("validated trajectory J values must be finite")
    return validated.sort_values(
        ["validated_J", "trajectory_id"], kind="mergesort"
    ).iloc[0].copy()


def method_manifest_payload(
    *,
    checkpoint: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    shortlists: Sequence[FrozenShortlist],
) -> dict[str, Any]:
    return {
        "manifest_id": MANIFEST_ID,
        "method_id": METHOD_ID,
        "default_enabled": DEFAULT_ENABLED,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "checkpoint": dict(checkpoint),
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "max_model_screened_candidates": MAX_MODEL_SCREENED_CANDIDATES,
        "max_validation_trials": MAX_VALIDATION_TRIALS,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "shortlist_selection_rule": SHORTLIST_SELECTION_RULE,
        "support_role": SUPPORT_ROLE,
        "truth_role": TRUTH_ROLE,
        "final_status_rule_frozen_before_candidate_truth": FINAL_STATUS_RULE,
        "shortlist_frozen_before_candidate_truth": True,
        "new_candidate_after_freeze_allowed": False,
        "new_predicted_optimum_after_refit_execution_allowed": False,
        "whole_trajectory_trial": True,
        "P2_explore_exploit_invoked": False,
        "bundle_invoked": False,
        "adaptive_horizon_invoked": False,
        "calibration_cases_used_for_rule_tuning": False,
        "held_out_final_test_read": False,
        "new_prospective_cohort_generated": False,
        "source_sha256": dict(source_hashes),
        "case_shortlists": [shortlist.as_manifest() for shortlist in shortlists],
    }
