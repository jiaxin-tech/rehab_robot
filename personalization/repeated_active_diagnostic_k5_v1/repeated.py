"""Repeated active diagnostics and cumulative prequential arbitration for K=5."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..active_diagnostic_v1.diagnostic import (
    ARBITRATION_MARGIN,
    DIAGNOSTIC_SCORE_ID,
    DiagnosticSelection,
    gaussian_kl_divergence,
    select_active_diagnostic_candidate,
)
from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from ..predictive_failover_v1.evidence import (
    PHYSICS_MODE,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
    capture_pretrial_predictions,
    gaussian_log_predictive_density,
)
from ..selectors.base import unexecuted_candidates


PRIMARY_RULE_ID = "REPEATED_ACTIVE_DIAGNOSTIC_ARBITRATION_K5_V1"
PRIMARY_BUDGET = 5
RULE_PATH = Path(__file__).with_name(
    "PRIMARY_REPEATED_ACTIVE_DIAGNOSTIC_K5_V1.json"
)


def verify_frozen_rule() -> dict[str, Any]:
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if payload["primary_rule_id"] != PRIMARY_RULE_ID:
        raise RuntimeError("repeated active diagnostic rule identifier drifted")
    if payload["primary_budget"] != PRIMARY_BUDGET:
        raise RuntimeError("repeated active diagnostic budget drifted")
    if payload["diagnostic_candidate"]["score_id"] != DIAGNOSTIC_SCORE_ID:
        raise RuntimeError("repeated active diagnostic score drifted")
    if payload["arbitration"]["evidence_margin"] != ARBITRATION_MARGIN:
        raise RuntimeError("repeated active diagnostic margin drifted")
    if payload["arbitration"]["inconclusive_policy"] != PHYSICS_MODE:
        raise RuntimeError("repeated diagnostic inconclusive policy drifted")
    if payload["score"]["variance_floor_std"] != 0.05:
        raise RuntimeError("repeated active diagnostic variance floor drifted")
    if payload["selected_expert_fixed_after_trial_3"] is not True:
        raise RuntimeError("repeated diagnostic expert must remain fixed")
    return payload


def causal_running_physics_offset(
    prediction_observations: Iterable[
        tuple[PretrialPredictionSnapshot, EpisodeObservation]
    ],
    *,
    before_trial: int,
) -> float:
    residuals = []
    for snapshot, observation in prediction_observations:
        if snapshot.trial_index >= before_trial or observation.trial_index >= before_trial:
            raise ValueError("physics offset cannot use current or future observations")
        if snapshot.trial_index != observation.trial_index:
            raise ValueError("physics offset prediction/observation trial mismatch")
        if snapshot.candidate_id != observation.candidate_id:
            raise ValueError("physics offset prediction/observation candidate mismatch")
        if (
            observation.valid
            and observation.endpoint_value is not None
            and snapshot.physics_bo_prediction_valid
            and math.isfinite(snapshot.physics_bo_mean_before)
        ):
            residuals.append(
                float(observation.endpoint_value)
                - snapshot.physics_bo_mean_before
            )
    return float(np.median(residuals)) if residuals else 0.0


def select_repeated_diagnostic_candidate(
    history: list[EpisodeObservation],
    domain: V3CandidateDomain,
    *,
    trial_index: int,
    standard_model: SequentialModel,
    physics_model: SequentialModel,
    physics_offset: float,
) -> DiagnosticSelection:
    """Select diagnostic 1 from D1 or diagnostic 2 from D2."""

    if trial_index == 2:
        return select_active_diagnostic_candidate(
            history,
            domain,
            standard_model=standard_model,
            physics_model=physics_model,
            physics_offset=physics_offset,
        )
    if trial_index != 3 or len(history) != 2:
        raise ValueError("repeated diagnostic selection is defined for Trial 2/D1 or Trial 3/D2")
    if [item.trial_index for item in history] != [1, 2]:
        raise ValueError("Trial 3 diagnostic must use contiguous D2")

    scored: list[DiagnosticSelection] = []
    for candidate in unexecuted_candidates(history, domain):
        predictions = capture_pretrial_predictions(
            trial_index=3,
            candidate=candidate,
            history_size=2,
            standard_model=standard_model,
            physics_model=physics_model,
        )
        if not (
            predictions.standard_bo_prediction_valid
            and predictions.physics_bo_prediction_valid
        ):
            continue
        adjusted_physics_mean = predictions.physics_bo_mean_before + physics_offset
        forward = gaussian_kl_divergence(
            predictions.standard_bo_mean_before,
            predictions.standard_bo_std_before,
            adjusted_physics_mean,
            predictions.physics_bo_std_before,
        )
        reverse = gaussian_kl_divergence(
            adjusted_physics_mean,
            predictions.physics_bo_std_before,
            predictions.standard_bo_mean_before,
            predictions.standard_bo_std_before,
        )
        scored.append(
            DiagnosticSelection(
                candidate=candidate,
                predictions=predictions,
                physics_offset=float(physics_offset),
                adjusted_physics_mean=float(adjusted_physics_mean),
                diagnostic_score=float(0.5 * (forward + reverse)),
                standard_to_physics_kl=forward,
                physics_to_standard_kl=reverse,
            )
        )
    if not scored:
        raise RuntimeError("no valid unexecuted Trial 3 diagnostic candidate")
    return min(
        scored,
        key=lambda item: (-item.diagnostic_score, item.candidate.candidate_index),
    )


@dataclass(frozen=True)
class RepeatedDiagnosticEvidence:
    trial_index: int
    selection: DiagnosticSelection
    observation_valid: bool
    observed_value: float | None
    standard_log_score: float | None
    physics_log_score: float | None
    score_difference: float | None
    evidence_sign: str
    update_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "selection": self.selection.as_dict(),
            "observation_valid": self.observation_valid,
            "observed_value": self.observed_value,
            "standard_log_score": self.standard_log_score,
            "physics_log_score": self.physics_log_score,
            "score_difference": self.score_difference,
            "evidence_sign": self.evidence_sign,
            "update_reason": self.update_reason,
        }


def score_repeated_diagnostic_observation(
    selection: DiagnosticSelection,
    observation: EpisodeObservation,
) -> RepeatedDiagnosticEvidence:
    trial_index = selection.predictions.trial_index
    if trial_index not in {2, 3} or observation.trial_index != trial_index:
        raise ValueError("diagnostic score trial mismatch")
    if selection.candidate.candidate_id != observation.candidate_id:
        raise ValueError("diagnostic prediction and observation candidate mismatch")
    if not observation.valid or observation.endpoint_value is None:
        return RepeatedDiagnosticEvidence(
            trial_index=trial_index,
            selection=selection,
            observation_valid=False,
            observed_value=None,
            standard_log_score=None,
            physics_log_score=None,
            score_difference=None,
            evidence_sign="INVALID_NO_EVIDENCE",
            update_reason="INVALID_DIAGNOSTIC_CONSUMED_NO_SCORE",
        )
    observed = float(observation.endpoint_value)
    observation_std = max(float(observation.endpoint_uncertainty or 0.0), 0.0)
    standard_score = gaussian_log_predictive_density(
        observed,
        selection.predictions.standard_bo_mean_before,
        selection.predictions.standard_bo_std_before,
        observation_std,
    )
    physics_score = gaussian_log_predictive_density(
        observed,
        selection.adjusted_physics_mean,
        selection.predictions.physics_bo_std_before,
        observation_std,
    )
    difference = physics_score - standard_score
    sign = (
        "FAVORS_PHYSICS"
        if difference > 0.0
        else "FAVORS_STANDARD"
        if difference < 0.0
        else "NEUTRAL"
    )
    return RepeatedDiagnosticEvidence(
        trial_index=trial_index,
        selection=selection,
        observation_valid=True,
        observed_value=observed,
        standard_log_score=standard_score,
        physics_log_score=physics_score,
        score_difference=difference,
        evidence_sign=sign,
        update_reason="VALID_PREQUENTIAL_GAUSSIAN_LOG_SCORE",
    )


def evidence_consistency(
    evidence: tuple[RepeatedDiagnosticEvidence, RepeatedDiagnosticEvidence],
) -> str:
    signs = tuple(item.evidence_sign for item in evidence)
    if "INVALID_NO_EVIDENCE" in signs:
        return "INCOMPLETE_INVALID_EVIDENCE"
    if signs == ("FAVORS_PHYSICS", "FAVORS_PHYSICS"):
        return "CONSISTENT_FAVORS_PHYSICS"
    if signs == ("FAVORS_STANDARD", "FAVORS_STANDARD"):
        return "CONSISTENT_FAVORS_STANDARD"
    if set(signs) == {"FAVORS_PHYSICS", "FAVORS_STANDARD"}:
        return "CONFLICTING_EVIDENCE"
    return "NEUTRAL_OR_PARTIALLY_NEUTRAL_EVIDENCE"


@dataclass(frozen=True)
class RepeatedDiagnosticArbitrationDecision:
    selected_expert: str
    decision: str
    cumulative_evidence: float
    valid_evidence_count: int
    evidence_consistency: str
    evidence_margin: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_expert": self.selected_expert,
            "decision": self.decision,
            "cumulative_evidence": self.cumulative_evidence,
            "valid_evidence_count": self.valid_evidence_count,
            "evidence_consistency": self.evidence_consistency,
            "evidence_margin": self.evidence_margin,
        }


def arbitrate_after_trial_3(
    evidence: tuple[RepeatedDiagnosticEvidence, RepeatedDiagnosticEvidence],
) -> RepeatedDiagnosticArbitrationDecision:
    if tuple(item.trial_index for item in evidence) != (2, 3):
        raise ValueError("repeated arbitration requires Trial 2 and Trial 3 evidence")
    valid_differences = [
        float(item.score_difference)
        for item in evidence
        if item.score_difference is not None
    ]
    cumulative = float(sum(valid_differences))
    if cumulative > ARBITRATION_MARGIN:
        selected = PHYSICS_MODE
        decision = "PHYSICS_SELECTED_BY_CUMULATIVE_DIAGNOSTIC_EVIDENCE"
    elif cumulative < -ARBITRATION_MARGIN:
        selected = STANDARD_MODE
        decision = "STANDARD_SELECTED_BY_CUMULATIVE_DIAGNOSTIC_EVIDENCE"
    else:
        selected = PHYSICS_MODE
        decision = "INCONCLUSIVE_DEFAULT_PHYSICS"
    return RepeatedDiagnosticArbitrationDecision(
        selected_expert=selected,
        decision=decision,
        cumulative_evidence=cumulative,
        valid_evidence_count=len(valid_differences),
        evidence_consistency=evidence_consistency(evidence),
        evidence_margin=ARBITRATION_MARGIN,
    )
