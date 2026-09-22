"""Active Gaussian-distribution diagnostic and one-shot model arbitration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..candidates import Candidate, V3CandidateDomain
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


PRIMARY_RULE_ID = "PRIMARY_ACTIVE_SYMMETRIC_KL_SINGLE_ARBITRATION_V1"
DIAGNOSTIC_SCORE_ID = "AVERAGE_SYMMETRIC_KL_GAUSSIAN"
VARIANCE_FLOOR_STD = 0.05
ARBITRATION_MARGIN = math.log(10.0)
INCONCLUSIVE_EXPERT = PHYSICS_MODE
RULE_PATH = Path(__file__).with_name("PRIMARY_ACTIVE_DIAGNOSTIC_RULE_V1.json")


def verify_frozen_rule() -> dict[str, Any]:
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if payload["primary_rule_id"] != PRIMARY_RULE_ID:
        raise RuntimeError("active diagnostic rule identifier drifted")
    if payload["primary_budget"] != 4:
        raise RuntimeError("active diagnostic budget drifted")
    if payload["diagnostic_candidate"]["score_id"] != DIAGNOSTIC_SCORE_ID:
        raise RuntimeError("active diagnostic score drifted")
    if payload["score"]["variance_floor_std"] != VARIANCE_FLOOR_STD:
        raise RuntimeError("active diagnostic variance floor drifted")
    if payload["arbitration"]["evidence_margin"] != ARBITRATION_MARGIN:
        raise RuntimeError("active diagnostic arbitration margin drifted")
    if payload["inconclusive_policy"] != INCONCLUSIVE_EXPERT:
        raise RuntimeError("active diagnostic inconclusive policy drifted")
    if payload["selected_expert_fixed_after_trial_2"] is not True:
        raise RuntimeError("active diagnostic expert must remain fixed")
    return payload


def _variance(std: float) -> float:
    if not math.isfinite(float(std)) or std < 0.0:
        raise ValueError("predictive standard deviation must be finite and non-negative")
    return max(float(std) ** 2, VARIANCE_FLOOR_STD**2)


def gaussian_kl_divergence(
    mean_p: float,
    std_p: float,
    mean_q: float,
    std_q: float,
) -> float:
    """KL[p||q] for univariate Gaussians with the frozen variance floor."""

    if not math.isfinite(float(mean_p)) or not math.isfinite(float(mean_q)):
        raise ValueError("predictive means must be finite")
    variance_p = _variance(std_p)
    variance_q = _variance(std_q)
    delta = float(mean_p) - float(mean_q)
    value = 0.5 * (
        math.log(variance_q / variance_p)
        + (variance_p + delta**2) / variance_q
        - 1.0
    )
    return float(max(value, 0.0))


def average_symmetric_kl_gaussian(
    standard_mean: float,
    standard_std: float,
    physics_mean: float,
    physics_std: float,
) -> float:
    """Average bidirectional KL, sensitive to both means and uncertainty."""

    forward = gaussian_kl_divergence(
        standard_mean, standard_std, physics_mean, physics_std
    )
    reverse = gaussian_kl_divergence(
        physics_mean, physics_std, standard_mean, standard_std
    )
    return float(0.5 * (forward + reverse))


def causal_trial_1_physics_offset(
    snapshot: PretrialPredictionSnapshot,
    observation: EpisodeObservation,
) -> float:
    if snapshot.trial_index != 1 or observation.trial_index != 1:
        raise ValueError("active diagnostic offset must be derived from Trial 1")
    if snapshot.history_size_before != 0:
        raise ValueError("Trial 1 offset forecast must use empty history")
    if snapshot.candidate_id != observation.candidate_id:
        raise ValueError("Trial 1 prediction and observation candidate mismatch")
    if (
        not observation.valid
        or observation.endpoint_value is None
        or not snapshot.physics_bo_prediction_valid
        or not math.isfinite(snapshot.physics_bo_mean_before)
    ):
        return 0.0
    return float(observation.endpoint_value) - snapshot.physics_bo_mean_before


@dataclass(frozen=True)
class DiagnosticSelection:
    candidate: Candidate
    predictions: PretrialPredictionSnapshot
    physics_offset: float
    adjusted_physics_mean: float
    diagnostic_score: float
    standard_to_physics_kl: float
    physics_to_standard_kl: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "predictions": self.predictions.as_dict(),
            "physics_offset": self.physics_offset,
            "adjusted_physics_mean": self.adjusted_physics_mean,
            "diagnostic_score": self.diagnostic_score,
            "diagnostic_score_id": DIAGNOSTIC_SCORE_ID,
            "standard_to_physics_kl": self.standard_to_physics_kl,
            "physics_to_standard_kl": self.physics_to_standard_kl,
        }


def select_active_diagnostic_candidate(
    history: list[EpisodeObservation],
    domain: V3CandidateDomain,
    *,
    standard_model: SequentialModel,
    physics_model: SequentialModel,
    physics_offset: float,
) -> DiagnosticSelection:
    """Choose Trial 2 solely from the two D1 predictive distributions."""

    if len(history) != 1 or history[0].trial_index != 1:
        raise ValueError("active diagnostic Trial 2 selection requires exactly D1")
    available = unexecuted_candidates(history, domain)
    scored: list[DiagnosticSelection] = []
    for candidate in available:
        predictions = capture_pretrial_predictions(
            trial_index=2,
            candidate=candidate,
            history_size=1,
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
        raise RuntimeError("no valid unexecuted active diagnostic candidate")
    return min(
        scored,
        key=lambda item: (-item.diagnostic_score, item.candidate.candidate_index),
    )


@dataclass(frozen=True)
class DiagnosticArbitrationDecision:
    selected_expert: str
    decision: str
    standard_log_score: float | None
    physics_log_score: float | None
    score_difference: float | None
    evidence_margin: float
    observation_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_expert": self.selected_expert,
            "decision": self.decision,
            "standard_log_score": self.standard_log_score,
            "physics_log_score": self.physics_log_score,
            "score_difference": self.score_difference,
            "evidence_margin": self.evidence_margin,
            "observation_valid": self.observation_valid,
        }


def arbitrate_after_diagnostic(
    selection: DiagnosticSelection,
    observation: EpisodeObservation,
) -> DiagnosticArbitrationDecision:
    """Make the only model-arbitration decision after revealing Trial 2."""

    if observation.trial_index != 2 or selection.predictions.trial_index != 2:
        raise ValueError("model arbitration is only defined after Trial 2")
    if observation.candidate_id != selection.candidate.candidate_id:
        raise ValueError("diagnostic prediction and observation candidate mismatch")
    if not observation.valid or observation.endpoint_value is None:
        return DiagnosticArbitrationDecision(
            selected_expert=INCONCLUSIVE_EXPERT,
            decision="INVALID_DIAGNOSTIC_INCONCLUSIVE_DEFAULT_PHYSICS",
            standard_log_score=None,
            physics_log_score=None,
            score_difference=None,
            evidence_margin=ARBITRATION_MARGIN,
            observation_valid=False,
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
    if difference < -ARBITRATION_MARGIN:
        selected = STANDARD_MODE
        decision = "STANDARD_SELECTED_BY_DIAGNOSTIC_EVIDENCE"
    elif difference > ARBITRATION_MARGIN:
        selected = PHYSICS_MODE
        decision = "PHYSICS_SELECTED_BY_DIAGNOSTIC_EVIDENCE"
    else:
        selected = INCONCLUSIVE_EXPERT
        decision = "INCONCLUSIVE_DEFAULT_PHYSICS"
    return DiagnosticArbitrationDecision(
        selected_expert=selected,
        decision=decision,
        standard_log_score=standard_score,
        physics_log_score=physics_score,
        score_difference=difference,
        evidence_margin=ARBITRATION_MARGIN,
        observation_valid=True,
    )
