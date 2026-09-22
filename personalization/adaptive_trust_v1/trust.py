"""Causal small-K physics-prior trust estimator frozen before benchmarking."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


RULE_PATH = Path(__file__).with_name("PRIMARY_TRUST_RULE_V1.json")
EXPECTED_RULE_SHA256 = "326007fe40d752f52d411d04d2c732ed3a2a1ced543d618ac817af683813c67d"
PRIMARY_TRUST_RULE_ID = "PRIMARY_CAUSAL_OFFSET_INVARIANT_RANKING_TRUST_V1"
ALGORITHMIC_PHYSICS_PRIOR_TRUST_WEIGHT = "ALGORITHMIC_PHYSICS_PRIOR_TRUST_WEIGHT"
NO_ORACLE_TRUST_UPDATE = "NO_ORACLE_TRUST_UPDATE"
INITIAL_TRUST_SCORE = 1.0
RANKING_WEIGHT = 0.75
CALIBRATION_WEIGHT = 0.25
FALLBACK_DOMINANT_THRESHOLD = 0.5


def verify_frozen_rule() -> dict[str, Any]:
    actual = hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_RULE_SHA256:
        raise RuntimeError("PRIMARY_TRUST_RULE_V1.json changed after rule freeze")
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if payload["primary_rule_id"] != PRIMARY_TRUST_RULE_ID:
        raise RuntimeError("primary trust rule identity mismatch")
    return payload


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class PhysicsPriorTrustEvidence:
    """One valid executed observation and its pre-observation physics forecast."""

    trial_index: int
    candidate_id: str
    beta_flex: float
    beta_extend: float
    observed_value: float
    physics_prediction: float
    observation_uncertainty: float

    def __post_init__(self) -> None:
        if self.trial_index < 1:
            raise ValueError("trust evidence trial_index must be >= 1")
        numeric = (
            self.beta_flex,
            self.beta_extend,
            self.observed_value,
            self.physics_prediction,
            self.observation_uncertainty,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("trust evidence values must be finite")
        if self.observation_uncertainty < 0.0:
            raise ValueError("observation uncertainty must be non-negative")

    @property
    def prediction_residual(self) -> float:
        return self.observed_value - self.physics_prediction

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "candidate_id": self.candidate_id,
            "beta_flex": self.beta_flex,
            "beta_extend": self.beta_extend,
            "observed_value": self.observed_value,
            "physics_prediction": self.physics_prediction,
            "observation_uncertainty": self.observation_uncertainty,
            "prediction_residual": self.prediction_residual,
        }


@dataclass(frozen=True)
class PhysicsPriorTrustState:
    """Versioned algorithmic weight state; not biological or clinical confidence."""

    trial_index: int
    trust_score: float
    evidence_count: int
    prediction_error_metrics: Mapping[str, Any]
    ranking_consistency_metrics: Mapping[str, Any]
    previous_trust_score: float
    update_reason: str
    valid: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError("trust_score must be bounded in [0,1]")
        if not 0.0 <= self.previous_trust_score <= 1.0:
            raise ValueError("previous_trust_score must be bounded in [0,1]")
        if self.trial_index < 0 or self.evidence_count < 0:
            raise ValueError("trust state indices must be non-negative")
        object.__setattr__(
            self, "prediction_error_metrics", _immutable_mapping(self.prediction_error_metrics)
        )
        object.__setattr__(
            self,
            "ranking_consistency_metrics",
            _immutable_mapping(self.ranking_consistency_metrics),
        )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "trust_score": self.trust_score,
            "evidence_count": self.evidence_count,
            "prediction_error_metrics": dict(self.prediction_error_metrics),
            "ranking_consistency_metrics": dict(self.ranking_consistency_metrics),
            "previous_trust_score": self.previous_trust_score,
            "update_reason": self.update_reason,
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }


class PhysicsPriorTrustEstimator:
    """Primary offset-invariant, ranking-led trust rule for at most K=4."""

    rule_id = PRIMARY_TRUST_RULE_ID

    def __init__(self) -> None:
        self.rule = verify_frozen_rule()

    def initial_state(self) -> PhysicsPriorTrustState:
        return PhysicsPriorTrustState(
            trial_index=0,
            trust_score=INITIAL_TRUST_SCORE,
            evidence_count=0,
            prediction_error_metrics={
                "calibration_score": 1.0,
                "offset_removed_nrmse": 0.0,
                "estimated_constant_offset": None,
                "normalization_scale": None,
            },
            ranking_consistency_metrics={
                "ranking_score": 1.0,
                "comparable_pair_count": 0,
                "concordant_pair_count": 0,
                "discordant_pair_count": 0,
            },
            previous_trust_score=INITIAL_TRUST_SCORE,
            update_reason="FIXED_INITIAL_TRUST_POLICY_NO_SUBJECT_EVIDENCE",
            valid=True,
            metadata=self._metadata(),
        )

    @staticmethod
    def _metadata() -> dict[str, Any]:
        return {
            "version": 1,
            "semantic_scope": ALGORITHMIC_PHYSICS_PRIOR_TRUST_WEIGHT,
            "not_biological_probability": True,
            "not_patient_or_clinical_confidence": True,
            "causal_policy": NO_ORACLE_TRUST_UPDATE,
            "primary_rule_sha256": EXPECTED_RULE_SHA256,
        }

    @staticmethod
    def _prediction_metrics(
        evidence: Sequence[PhysicsPriorTrustEvidence],
    ) -> dict[str, Any]:
        if len(evidence) < 2:
            offset = evidence[0].prediction_residual if evidence else None
            return {
                "calibration_score": 1.0,
                "offset_removed_nrmse": 0.0,
                "estimated_constant_offset": offset,
                "normalization_scale": None,
                "minimum_valid_evidence_met": False,
            }
        observed = np.asarray([item.observed_value for item in evidence], dtype=float)
        physics = np.asarray([item.physics_prediction for item in evidence], dtype=float)
        uncertainty = np.asarray(
            [item.observation_uncertainty for item in evidence], dtype=float
        )
        residual = observed - physics
        offset = float(np.median(residual))
        centered = residual - offset
        scale = max(
            float(np.ptp(observed)),
            float(np.ptp(physics)),
            4.0 * float(np.median(uncertainty)),
            1.0e-6,
        )
        nrmse = float(np.sqrt(np.mean(centered**2)) / scale)
        return {
            "calibration_score": float(math.exp(-nrmse)),
            "offset_removed_nrmse": nrmse,
            "estimated_constant_offset": offset,
            "normalization_scale": scale,
            "minimum_valid_evidence_met": True,
        }

    @staticmethod
    def _ranking_metrics(
        evidence: Sequence[PhysicsPriorTrustEvidence],
    ) -> dict[str, Any]:
        comparable = concordant = discordant = 0
        for right_index, right in enumerate(evidence):
            for left in evidence[:right_index]:
                observed_delta = right.observed_value - left.observed_value
                physics_delta = right.physics_prediction - left.physics_prediction
                uncertainty_deadband = 2.0 * math.hypot(
                    right.observation_uncertainty, left.observation_uncertainty
                )
                if (
                    abs(observed_delta) <= uncertainty_deadband
                    or abs(physics_delta) <= 1.0e-12
                ):
                    continue
                comparable += 1
                if observed_delta * physics_delta > 0.0:
                    concordant += 1
                else:
                    discordant += 1
        ranking_score = (
            (concordant + 0.5) / (comparable + 1.0) if comparable else 1.0
        )
        return {
            "ranking_score": float(ranking_score),
            "comparable_pair_count": comparable,
            "concordant_pair_count": concordant,
            "discordant_pair_count": discordant,
            "pair_uncertainty_deadband_multiplier": 2.0,
        }

    def update(
        self,
        previous: PhysicsPriorTrustState,
        evidence: Sequence[PhysicsPriorTrustEvidence],
        *,
        trial_index: int,
        current_observation_valid: bool,
        invalid_reason: str | None = None,
    ) -> PhysicsPriorTrustState:
        if trial_index != previous.trial_index + 1:
            raise ValueError("trust updates must follow contiguous trial indices")
        ordered = tuple(evidence)
        if len({item.trial_index for item in ordered}) != len(ordered):
            raise ValueError("trust evidence trial indices must be unique")
        if any(item.trial_index > trial_index for item in ordered):
            raise ValueError("future trust evidence is prohibited")
        if not current_observation_valid:
            if len(ordered) != previous.evidence_count:
                raise ValueError("invalid observations cannot add trust evidence")
            return PhysicsPriorTrustState(
                trial_index=trial_index,
                trust_score=previous.trust_score,
                evidence_count=previous.evidence_count,
                prediction_error_metrics=previous.prediction_error_metrics,
                ranking_consistency_metrics=previous.ranking_consistency_metrics,
                previous_trust_score=previous.trust_score,
                update_reason=f"INVALID_OBSERVATION_IGNORED:{invalid_reason or 'UNSPECIFIED'}",
                valid=True,
                metadata=self._metadata(),
            )
        if len(ordered) != previous.evidence_count + 1:
            raise ValueError("one valid observation must add exactly one trust evidence item")
        prediction = self._prediction_metrics(ordered)
        ranking = self._ranking_metrics(ordered)
        score = float(
            np.clip(
                RANKING_WEIGHT * float(ranking["ranking_score"])
                + CALIBRATION_WEIGHT * float(prediction["calibration_score"]),
                0.0,
                1.0,
            )
        )
        reason = (
            "VALID_EVIDENCE_NO_COMPARABLE_PAIR"
            if int(ranking["comparable_pair_count"]) == 0
            else "RANKING_AND_OFFSET_REMOVED_CALIBRATION_UPDATED"
        )
        return PhysicsPriorTrustState(
            trial_index=trial_index,
            trust_score=score,
            evidence_count=len(ordered),
            prediction_error_metrics=prediction,
            ranking_consistency_metrics=ranking,
            previous_trust_score=previous.trust_score,
            update_reason=reason,
            valid=True,
            metadata=self._metadata(),
        )


class FixedTrustEstimator(PhysicsPriorTrustEstimator):
    """Limit-case test utility, never the primary adaptive benchmark rule."""

    def __init__(self, trust_score: float) -> None:
        if not 0.0 <= trust_score <= 1.0:
            raise ValueError("fixed trust must be in [0,1]")
        self.fixed_trust_score = float(trust_score)
        self.rule = verify_frozen_rule()

    def initial_state(self) -> PhysicsPriorTrustState:
        base = super().initial_state()
        return PhysicsPriorTrustState(
            trial_index=0,
            trust_score=self.fixed_trust_score,
            evidence_count=0,
            prediction_error_metrics=base.prediction_error_metrics,
            ranking_consistency_metrics=base.ranking_consistency_metrics,
            previous_trust_score=self.fixed_trust_score,
            update_reason="FIXED_TRUST_LIMIT_CASE_TEST_ONLY",
            valid=True,
            metadata={**self._metadata(), "test_only_fixed_trust": True},
        )

    def update(
        self,
        previous: PhysicsPriorTrustState,
        evidence: Sequence[PhysicsPriorTrustEvidence],
        *,
        trial_index: int,
        current_observation_valid: bool,
        invalid_reason: str | None = None,
    ) -> PhysicsPriorTrustState:
        del invalid_reason
        if trial_index != previous.trial_index + 1:
            raise ValueError("trust updates must follow contiguous trial indices")
        if any(item.trial_index > trial_index for item in evidence):
            raise ValueError("future trust evidence is prohibited")
        expected_count = previous.evidence_count + int(current_observation_valid)
        if len(evidence) != expected_count:
            raise ValueError("fixed-trust evidence accounting mismatch")
        return PhysicsPriorTrustState(
            trial_index=trial_index,
            trust_score=self.fixed_trust_score,
            evidence_count=len(evidence),
            prediction_error_metrics=previous.prediction_error_metrics,
            ranking_consistency_metrics=previous.ranking_consistency_metrics,
            previous_trust_score=self.fixed_trust_score,
            update_reason="FIXED_TRUST_LIMIT_CASE_TEST_ONLY",
            valid=True,
            metadata={**self._metadata(), "test_only_fixed_trust": True},
        )
