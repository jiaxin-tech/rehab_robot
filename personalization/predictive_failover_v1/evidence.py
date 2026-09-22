"""Causal prequential evidence and one-way expert arbitration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..candidates import Candidate
from ..models.base import SequentialModel
from ..observations import EpisodeObservation


PHYSICS_MODE = "PHYSICS_BO"
STANDARD_MODE = "STANDARD_BO"
VARIANCE_FLOOR_STD = 0.05
FAILOVER_THRESHOLD = -math.log(10.0)
SCORE_START_TRIAL = 2
PRIMARY_RULE_ID = "PRIMARY_PREQUENTIAL_GAUSSIAN_LOG_SCORE_ONE_WAY_FAILOVER_V1"
RULE_PATH = Path(__file__).with_name("PRIMARY_FAILOVER_RULE_V1.json")


def verify_frozen_rule() -> dict[str, Any]:
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if payload["primary_rule_id"] != PRIMARY_RULE_ID:
        raise RuntimeError("primary failover rule identifier drifted")
    if payload["primary_budget"] != 4:
        raise RuntimeError("primary failover budget drifted")
    if payload["score"]["variance_floor_std"] != VARIANCE_FLOOR_STD:
        raise RuntimeError("predictive-score variance floor drifted")
    if payload["failover"]["threshold"] != FAILOVER_THRESHOLD:
        raise RuntimeError("primary failover threshold drifted")
    if payload["score_start_trial"] != SCORE_START_TRIAL:
        raise RuntimeError("predictive evidence start trial drifted")
    if payload["initial_mode"] != PHYSICS_MODE:
        raise RuntimeError("initial expert mode drifted")
    if payload["calibration"]["mode"] != (
        "CAUSAL_RUNNING_MEDIAN_PHYSICS_OFFSET"
    ):
        raise RuntimeError("physics-score calibration rule drifted")
    if payload["selection_semantics"]["continuous_mixture_used"] is not False:
        raise RuntimeError("continuous mixture is prohibited in failover V1")
    return payload


def gaussian_log_predictive_density(
    observed: float,
    mean: float,
    model_std: float,
    observation_std: float = 0.0,
) -> float:
    """Gaussian log density with frozen total-variance semantics."""

    values = (observed, mean, model_std, observation_std)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("predictive score inputs must be finite")
    if model_std < 0.0 or observation_std < 0.0:
        raise ValueError("predictive standard deviations must be non-negative")
    variance = max(
        float(model_std) ** 2 + float(observation_std) ** 2,
        VARIANCE_FLOOR_STD**2,
    )
    residual = float(observed) - float(mean)
    return float(-0.5 * (math.log(2.0 * math.pi * variance) + residual**2 / variance))


@dataclass(frozen=True)
class PretrialPredictionSnapshot:
    """Both expert forecasts frozen before the current observation is revealed."""

    trial_index: int
    candidate_id: str
    beta_flex: float
    beta_extend: float
    history_size_before: int
    standard_bo_mean_before: float
    standard_bo_std_before: float
    standard_bo_prediction_valid: bool
    physics_bo_mean_before: float
    physics_bo_std_before: float
    physics_bo_prediction_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "candidate_id": self.candidate_id,
            "beta": [self.beta_flex, self.beta_extend],
            "history_size_before": self.history_size_before,
            "standard_bo_mean_before": self.standard_bo_mean_before,
            "standard_bo_std_before": self.standard_bo_std_before,
            "standard_bo_prediction_valid": self.standard_bo_prediction_valid,
            "physics_bo_mean_before": self.physics_bo_mean_before,
            "physics_bo_std_before": self.physics_bo_std_before,
            "physics_bo_prediction_valid": self.physics_bo_prediction_valid,
        }


def capture_pretrial_predictions(
    *,
    trial_index: int,
    candidate: Candidate,
    history_size: int,
    standard_model: SequentialModel,
    physics_model: SequentialModel,
) -> PretrialPredictionSnapshot:
    if history_size != trial_index - 1:
        raise ValueError("pretrial prediction must use exactly D_(k-1)")
    standard = standard_model.predict(candidate)
    physics = physics_model.predict(candidate)
    return PretrialPredictionSnapshot(
        trial_index=trial_index,
        candidate_id=candidate.candidate_id,
        beta_flex=candidate.beta_flex,
        beta_extend=candidate.beta_extend,
        history_size_before=history_size,
        standard_bo_mean_before=float(standard.mean),
        standard_bo_std_before=float(standard.std),
        standard_bo_prediction_valid=bool(standard.valid),
        physics_bo_mean_before=float(physics.mean),
        physics_bo_std_before=float(physics.std),
        physics_bo_prediction_valid=bool(physics.valid),
    )


@dataclass(frozen=True)
class PredictiveEvidenceRecord:
    trial_index: int
    snapshot: PretrialPredictionSnapshot
    observation_valid: bool
    observed_value: float | None
    observation_std: float | None
    physics_offset_before: float
    physics_scoring_mean: float
    standard_log_score: float | None
    physics_log_score: float | None
    score_difference: float | None
    cumulative_evidence_before: float
    cumulative_evidence_after: float
    scored_evidence_count: int
    update_reason: str
    mode_before: str
    mode_after: str
    failover_triggered_this_trial: bool
    failover_trial: int | None

    @property
    def usable_for_future_offset(self) -> bool:
        return (
            self.observation_valid
            and self.observed_value is not None
            and self.snapshot.physics_bo_prediction_valid
            and math.isfinite(self.snapshot.physics_bo_mean_before)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.snapshot.as_dict(),
            "observation_valid": self.observation_valid,
            "observed_value": self.observed_value,
            "observation_std": self.observation_std,
            "physics_offset_before": self.physics_offset_before,
            "physics_scoring_mean": self.physics_scoring_mean,
            "standard_log_score": self.standard_log_score,
            "physics_log_score": self.physics_log_score,
            "score_difference": self.score_difference,
            "cumulative_evidence_before": self.cumulative_evidence_before,
            "cumulative_evidence_after": self.cumulative_evidence_after,
            "scored_evidence_count": self.scored_evidence_count,
            "update_reason": self.update_reason,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "failover_triggered_this_trial": self.failover_triggered_this_trial,
            "failover_trial": self.failover_trial,
        }


@dataclass(frozen=True)
class PredictiveEvidenceState:
    trial_index: int = 0
    mode: str = PHYSICS_MODE
    cumulative_evidence: float = 0.0
    scored_evidence_count: int = 0
    failed_over: bool = False
    failover_trial: int | None = None
    records: tuple[PredictiveEvidenceRecord, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "mode": self.mode,
            "cumulative_evidence": self.cumulative_evidence,
            "scored_evidence_count": self.scored_evidence_count,
            "failed_over": self.failed_over,
            "failover_trial": self.failover_trial,
            "records": [record.as_dict() for record in self.records],
        }


class PredictiveEvidenceArbitrator:
    """Frozen one-way physics-to-standard failover rule."""

    def __init__(self) -> None:
        verify_frozen_rule()

    @staticmethod
    def initial_state() -> PredictiveEvidenceState:
        return PredictiveEvidenceState()

    @staticmethod
    def _physics_offset(state: PredictiveEvidenceState) -> float:
        residuals = [
            float(record.observed_value) - record.snapshot.physics_bo_mean_before
            for record in state.records
            if record.usable_for_future_offset
        ]
        return float(np.median(residuals)) if residuals else 0.0

    def update(
        self,
        state: PredictiveEvidenceState,
        snapshot: PretrialPredictionSnapshot,
        observation: EpisodeObservation,
    ) -> PredictiveEvidenceState:
        expected_trial = state.trial_index + 1
        if snapshot.trial_index != expected_trial or observation.trial_index != expected_trial:
            raise ValueError("predictive evidence trials must be contiguous")
        if snapshot.history_size_before != state.trial_index:
            raise ValueError("snapshot does not represent D_(k-1)")
        if snapshot.candidate_id != observation.candidate_id:
            raise ValueError("prediction and observation candidate mismatch")
        if not math.isclose(snapshot.beta_flex, observation.beta_flex) or not math.isclose(
            snapshot.beta_extend, observation.beta_extend
        ):
            raise ValueError("prediction and observation beta mismatch")

        offset = self._physics_offset(state)
        physics_scoring_mean = snapshot.physics_bo_mean_before + offset
        standard_score = physics_score = difference = None
        cumulative = state.cumulative_evidence
        score_count = state.scored_evidence_count

        prediction_valid = (
            snapshot.standard_bo_prediction_valid
            and snapshot.physics_bo_prediction_valid
            and math.isfinite(snapshot.standard_bo_mean_before)
            and math.isfinite(snapshot.standard_bo_std_before)
            and snapshot.standard_bo_std_before >= 0.0
            and math.isfinite(snapshot.physics_bo_mean_before)
            and math.isfinite(snapshot.physics_bo_std_before)
            and snapshot.physics_bo_std_before >= 0.0
            and math.isfinite(physics_scoring_mean)
        )
        if snapshot.trial_index < SCORE_START_TRIAL:
            reason = "TRIAL_1_CALIBRATION_ONLY_NO_MODEL_COMPARISON_SCORE"
        elif not observation.valid or observation.endpoint_value is None:
            reason = "INVALID_OBSERVATION_NO_SCORE_NO_EVIDENCE"
        elif not prediction_valid:
            reason = "INVALID_PRETRIAL_PREDICTION_NO_SCORE_NO_EVIDENCE"
        else:
            observed = float(observation.endpoint_value)
            observation_std = max(float(observation.endpoint_uncertainty or 0.0), 0.0)
            standard_score = gaussian_log_predictive_density(
                observed,
                snapshot.standard_bo_mean_before,
                snapshot.standard_bo_std_before,
                observation_std,
            )
            physics_score = gaussian_log_predictive_density(
                observed,
                physics_scoring_mean,
                snapshot.physics_bo_std_before,
                observation_std,
            )
            difference = physics_score - standard_score
            cumulative += difference
            score_count += 1
            reason = "VALID_PREQUENTIAL_GAUSSIAN_LOG_SCORE"

        newly_failed = (
            not state.failed_over
            and snapshot.trial_index >= SCORE_START_TRIAL
            and difference is not None
            and cumulative < FAILOVER_THRESHOLD
        )
        failed_over = state.failed_over or newly_failed
        failover_trial = snapshot.trial_index if newly_failed else state.failover_trial
        mode_after = STANDARD_MODE if failed_over else PHYSICS_MODE
        record = PredictiveEvidenceRecord(
            trial_index=snapshot.trial_index,
            snapshot=snapshot,
            observation_valid=bool(observation.valid),
            observed_value=(
                float(observation.endpoint_value)
                if observation.valid and observation.endpoint_value is not None
                else None
            ),
            observation_std=(
                max(float(observation.endpoint_uncertainty or 0.0), 0.0)
                if observation.valid
                else None
            ),
            physics_offset_before=offset,
            physics_scoring_mean=physics_scoring_mean,
            standard_log_score=standard_score,
            physics_log_score=physics_score,
            score_difference=difference,
            cumulative_evidence_before=state.cumulative_evidence,
            cumulative_evidence_after=cumulative,
            scored_evidence_count=score_count,
            update_reason=reason,
            mode_before=state.mode,
            mode_after=mode_after,
            failover_triggered_this_trial=newly_failed,
            failover_trial=failover_trial,
        )
        return PredictiveEvidenceState(
            trial_index=snapshot.trial_index,
            mode=mode_after,
            cumulative_evidence=cumulative,
            scored_evidence_count=score_count,
            failed_over=failed_over,
            failover_trial=failover_trial,
            records=(*state.records, record),
        )
