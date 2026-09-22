"""Causal K=4 adaptive-trust sequential personalization loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..candidates import Candidate, V3CandidateDomain
from ..environment import PersonalizationEnvironment
from ..ledger import ExecutedCandidateLedger, LedgerEntry
from ..models.physics_graybox import PhysicsSubjectModel
from ..observations import EpisodeObservation, valid_observations
from ..selectors import LowerConfidenceBoundSelector
from .model import AdaptivePhysicsMixtureModel
from .trust import (
    FALLBACK_DOMINANT_THRESHOLD,
    PhysicsPriorTrustEstimator,
    PhysicsPriorTrustEvidence,
    PhysicsPriorTrustState,
)


ADAPTIVE_TRUST_METHOD = "Adaptive-Trust Physics BO"
PRIMARY_ADAPTATION_BUDGET = 4


@dataclass
class AdaptiveTrustLedgerEntry(LedgerEntry):
    """Trust fields added to the existing Stage 1 ledger entry contract."""

    physics_prediction_before_observation: float | None
    prediction_residual: float | None
    physics_prior_trust_before: float
    physics_prior_trust_after: float
    trust_evidence_summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            **super().as_dict(),
            "physics_prediction_before_observation": (
                self.physics_prediction_before_observation
            ),
            "prediction_residual": self.prediction_residual,
            "physics_prior_trust_before": self.physics_prior_trust_before,
            "physics_prior_trust_after": self.physics_prior_trust_after,
            "trust_evidence_summary": dict(self.trust_evidence_summary),
        }


@dataclass
class AdaptiveTrustSequentialRunResult:
    method: str
    budget: int
    ledger: ExecutedCandidateLedger
    best_observed_candidate: Candidate | None
    model_recommended_final_candidate: Candidate | None
    final_model_summary: dict[str, Any]
    trust_states: tuple[PhysicsPriorTrustState, ...]
    fallback_dominant_selection_frequency: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "budget": self.budget,
            "best_observed_candidate": (
                self.best_observed_candidate.as_dict()
                if self.best_observed_candidate is not None
                else None
            ),
            "model_recommended_final_candidate": (
                self.model_recommended_final_candidate.as_dict()
                if self.model_recommended_final_candidate is not None
                else None
            ),
            "final_model_summary": self.final_model_summary,
            "trust_states": [state.as_dict() for state in self.trust_states],
            "final_trust": self.trust_states[-1].trust_score,
            "fallback_dominant_selection_frequency": (
                self.fallback_dominant_selection_frequency
            ),
            "ledger": self.ledger.as_dict(),
        }


def _best_observed(
    history: list[EpisodeObservation], domain: V3CandidateDomain
) -> Candidate | None:
    usable = valid_observations(history)
    if not usable:
        return None
    best = min(usable, key=lambda item: (float(item.endpoint_value), item.trial_index))
    return domain.by_id(best.candidate_id)


def _final_recommendation(
    model: AdaptivePhysicsMixtureModel,
    domain: V3CandidateDomain,
) -> Candidate:
    return min(domain, key=lambda item: (model.predict(item).mean, item.candidate_index))


def run_adaptive_trust_personalization(
    environment: PersonalizationEnvironment,
    domain: V3CandidateDomain,
    *,
    physics_model: PhysicsSubjectModel,
    trust_estimator: PhysicsPriorTrustEstimator | None = None,
    budget: int = PRIMARY_ADAPTATION_BUDGET,
    kappa: float = 1.5,
) -> AdaptiveTrustSequentialRunResult:
    """Run the frozen K=4 adaptive-trust method without environment oracle access."""

    if budget != PRIMARY_ADAPTATION_BUDGET:
        raise ValueError("Adaptive-Trust Physics BO primary budget is frozen at K=4")
    estimator = trust_estimator or PhysicsPriorTrustEstimator()
    state = estimator.initial_state()
    model = AdaptivePhysicsMixtureModel(
        physics_model, trust_score=state.trust_score
    )
    selector = LowerConfidenceBoundSelector(name=ADAPTIVE_TRUST_METHOD, kappa=kappa)
    ledger = ExecutedCandidateLedger()
    evidence: list[PhysicsPriorTrustEvidence] = []
    states: list[PhysicsPriorTrustState] = []
    fallback_dominant_selections = 0
    current = domain.reference

    for trial_index in range(1, budget + 1):
        trust_before = state.trust_score
        physics_prediction = model.physics.predict(current)
        observation = environment.evaluate(current, trial_index)
        evidence_valid = (
            observation.valid
            and physics_prediction.valid
            and math.isfinite(physics_prediction.mean)
        )
        if evidence_valid:
            assert observation.endpoint_value is not None
            evidence.append(
                PhysicsPriorTrustEvidence(
                    trial_index=trial_index,
                    candidate_id=current.candidate_id,
                    beta_flex=current.beta_flex,
                    beta_extend=current.beta_extend,
                    observed_value=float(observation.endpoint_value),
                    physics_prediction=float(physics_prediction.mean),
                    observation_uncertainty=float(
                        observation.endpoint_uncertainty or 0.0
                    ),
                )
            )
        invalid_reason = observation.invalid_reason
        if observation.valid and not evidence_valid:
            invalid_reason = "PHYSICS_PREDICTION_INVALID_FOR_TRUST"
        state = estimator.update(
            state,
            tuple(evidence),
            trial_index=trial_index,
            current_observation_valid=evidence_valid,
            invalid_reason=invalid_reason,
        )
        states.append(state)
        model.set_trust(state.trust_score)
        history = ledger.observations + [observation]
        model.fit(history)
        selection = None
        if trial_index < budget:
            selection = selector.select_next(history, domain, model)
            if state.trust_score < FALLBACK_DOMINANT_THRESHOLD:
                fallback_dominant_selections += 1
        prediction_residual = (
            float(observation.endpoint_value) - physics_prediction.mean
            if evidence_valid and observation.endpoint_value is not None
            else None
        )
        ledger.append(
            AdaptiveTrustLedgerEntry(
                trial_index=trial_index,
                candidate=current,
                observation=observation,
                physics_model_state_summary=model.physics.state_summary(),
                residual_model_state_summary=model.state_summary(),
                selector=selector.name,
                acquisition_value=(selection.acquisition_value if selection else None),
                selected_next_candidate=(selection.candidate if selection else None),
                physics_prediction_before_observation=(
                    physics_prediction.mean if physics_prediction.valid else None
                ),
                prediction_residual=prediction_residual,
                physics_prior_trust_before=trust_before,
                physics_prior_trust_after=state.trust_score,
                trust_evidence_summary={
                    "evidence_count": state.evidence_count,
                    "prediction_error_metrics": dict(
                        state.prediction_error_metrics
                    ),
                    "ranking_consistency_metrics": dict(
                        state.ranking_consistency_metrics
                    ),
                    "update_reason": state.update_reason,
                    "used_to_select_next_trial": trial_index < budget,
                },
            )
        )
        if selection is not None:
            current = selection.candidate

    best_observed = _best_observed(ledger.observations, domain)
    recommendation = _final_recommendation(model, domain)
    return AdaptiveTrustSequentialRunResult(
        method=ADAPTIVE_TRUST_METHOD,
        budget=budget,
        ledger=ledger,
        best_observed_candidate=best_observed,
        model_recommended_final_candidate=recommendation,
        final_model_summary=model.state_summary(),
        trust_states=tuple(states),
        fallback_dominant_selection_frequency=(
            fallback_dominant_selections / (budget - 1)
        ),
    )
