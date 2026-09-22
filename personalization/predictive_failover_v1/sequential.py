"""K=4 predictive-evidence arbitration between two frozen BO experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..candidates import Candidate, V3CandidateDomain
from ..environment import PersonalizationEnvironment
from ..ledger import ExecutedCandidateLedger, LedgerEntry
from ..models.base import SequentialModel
from ..models.physics_graybox import PhysicsSubjectModel
from ..models.residual_gp import PhysicsInformedResidualModel
from ..models.standard_gp import StandardGaussianProcess
from ..observations import EpisodeObservation, valid_observations
from ..selectors import LowerConfidenceBoundSelector
from .evidence import (
    PHYSICS_MODE,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
    PredictiveEvidenceArbitrator,
    PredictiveEvidenceState,
    capture_pretrial_predictions,
    verify_frozen_rule,
)


PREDICTIVE_FAILOVER_METHOD = "Predictive-Evidence Failover V1"
PRIMARY_ADAPTATION_BUDGET = 4


@dataclass
class PredictiveFailoverLedgerEntry(LedgerEntry):
    """Pre-observation forecasts and post-observation evidence for one trial."""

    pretrial_predictions: PretrialPredictionSnapshot
    physics_offset_before: float
    physics_scoring_mean: float
    standard_log_score: float | None
    physics_log_score: float | None
    score_difference: float | None
    cumulative_evidence_before: float
    cumulative_evidence_after: float
    mode_before_observation: str
    mode_after_observation: str
    failover_triggered_this_trial: bool
    failover_trial: int | None
    evidence_update_reason: str

    @property
    def standard_bo_mean_before(self) -> float:
        return self.pretrial_predictions.standard_bo_mean_before

    @property
    def standard_bo_std_before(self) -> float:
        return self.pretrial_predictions.standard_bo_std_before

    @property
    def physics_bo_mean_before(self) -> float:
        return self.pretrial_predictions.physics_bo_mean_before

    @property
    def physics_bo_std_before(self) -> float:
        return self.pretrial_predictions.physics_bo_std_before

    def as_dict(self) -> dict[str, Any]:
        snapshot = self.pretrial_predictions
        return {
            **super().as_dict(),
            "pretrial_predictions": snapshot.as_dict(),
            "standard_bo_mean_before": snapshot.standard_bo_mean_before,
            "standard_bo_std_before": snapshot.standard_bo_std_before,
            "physics_bo_mean_before": snapshot.physics_bo_mean_before,
            "physics_bo_std_before": snapshot.physics_bo_std_before,
            "physics_offset_before": self.physics_offset_before,
            "physics_scoring_mean": self.physics_scoring_mean,
            "standard_log_score": self.standard_log_score,
            "physics_log_score": self.physics_log_score,
            "score_difference": self.score_difference,
            "cumulative_evidence_before": self.cumulative_evidence_before,
            "cumulative_evidence_after": self.cumulative_evidence_after,
            "mode_before_observation": self.mode_before_observation,
            "mode_after_observation": self.mode_after_observation,
            "failover_triggered_this_trial": self.failover_triggered_this_trial,
            "failover_trial": self.failover_trial,
            "evidence_update_reason": self.evidence_update_reason,
        }


@dataclass
class PredictiveFailoverSequentialRunResult:
    method: str
    budget: int
    ledger: ExecutedCandidateLedger
    best_observed_candidate: Candidate | None
    model_recommended_final_candidate: Candidate | None
    final_model_summary: dict[str, Any]
    evidence_states: tuple[PredictiveEvidenceState, ...]
    failed_over: bool
    failover_trial: int | None

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
            "failed_over": self.failed_over,
            "failover_trial": self.failover_trial,
            "evidence_states": [state.as_dict() for state in self.evidence_states],
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


def _model_recommendation(
    model: SequentialModel,
    domain: V3CandidateDomain,
) -> Candidate:
    return min(domain, key=lambda item: (model.predict(item).mean, item.candidate_index))


def run_predictive_evidence_failover(
    environment: PersonalizationEnvironment,
    domain: V3CandidateDomain,
    *,
    physics_model: PhysicsSubjectModel,
    budget: int = PRIMARY_ADAPTATION_BUDGET,
    kappa: float = 1.5,
) -> PredictiveFailoverSequentialRunResult:
    """Run the preregistered causal one-way failover method."""

    verify_frozen_rule()
    if budget != PRIMARY_ADAPTATION_BUDGET:
        raise ValueError("Predictive-Evidence Failover V1 primary budget is frozen at K=4")

    standard_expert = StandardGaussianProcess()
    physics_expert = PhysicsInformedResidualModel(physics_model)
    standard_selector = LowerConfidenceBoundSelector(name="Standard BO", kappa=kappa)
    physics_selector = LowerConfidenceBoundSelector(
        name="Physics-Informed BO", kappa=kappa
    )
    arbitrator = PredictiveEvidenceArbitrator()
    state = arbitrator.initial_state()
    states: list[PredictiveEvidenceState] = []
    ledger = ExecutedCandidateLedger()
    current = domain.reference

    for trial_index in range(1, budget + 1):
        mode_before = state.mode
        snapshot = capture_pretrial_predictions(
            trial_index=trial_index,
            candidate=current,
            history_size=len(ledger.entries),
            standard_model=standard_expert,
            physics_model=physics_expert,
        )

        # This is the only observation reveal. Both forecasts above are frozen first.
        observation = environment.evaluate(current, trial_index)
        state = arbitrator.update(state, snapshot, observation)
        states.append(state)
        evidence_record = state.records[-1]

        history = [*ledger.observations, observation]
        standard_expert.fit(history)
        physics_expert.fit(history)

        active_model: SequentialModel
        if state.mode == STANDARD_MODE:
            active_model = standard_expert
            active_selector = standard_selector
        elif state.mode == PHYSICS_MODE:
            active_model = physics_expert
            active_selector = physics_selector
        else:
            raise RuntimeError(f"unknown arbitration mode: {state.mode}")

        selection = None
        if trial_index < budget:
            selection = active_selector.select_next(history, domain, active_model)

        ledger.append(
            PredictiveFailoverLedgerEntry(
                trial_index=trial_index,
                candidate=current,
                observation=observation,
                physics_model_state_summary=physics_expert.physics.state_summary(),
                residual_model_state_summary={
                    "standard_bo": standard_expert.state_summary(),
                    "physics_bo": physics_expert.state_summary(),
                },
                selector=active_selector.name,
                acquisition_value=(selection.acquisition_value if selection else None),
                selected_next_candidate=(selection.candidate if selection else None),
                pretrial_predictions=snapshot,
                physics_offset_before=evidence_record.physics_offset_before,
                physics_scoring_mean=evidence_record.physics_scoring_mean,
                standard_log_score=evidence_record.standard_log_score,
                physics_log_score=evidence_record.physics_log_score,
                score_difference=evidence_record.score_difference,
                cumulative_evidence_before=(
                    evidence_record.cumulative_evidence_before
                ),
                cumulative_evidence_after=evidence_record.cumulative_evidence_after,
                mode_before_observation=mode_before,
                mode_after_observation=state.mode,
                failover_triggered_this_trial=(
                    evidence_record.failover_triggered_this_trial
                ),
                failover_trial=state.failover_trial,
                evidence_update_reason=evidence_record.update_reason,
            )
        )
        if selection is not None:
            current = selection.candidate

    best_observed = _best_observed(ledger.observations, domain)
    final_model: SequentialModel = (
        standard_expert if state.mode == STANDARD_MODE else physics_expert
    )
    recommendation = _model_recommendation(final_model, domain)
    return PredictiveFailoverSequentialRunResult(
        method=PREDICTIVE_FAILOVER_METHOD,
        budget=budget,
        ledger=ledger,
        best_observed_candidate=best_observed,
        model_recommended_final_candidate=recommendation,
        final_model_summary={
            "model": "prequential_predictive_evidence_one_way_expert_arbitration",
            "final_mode": state.mode,
            "failed_over": state.failed_over,
            "failover_trial": state.failover_trial,
            "cumulative_evidence": state.cumulative_evidence,
            "scored_evidence_count": state.scored_evidence_count,
            "continuous_mixture_used": False,
            "standard_expert": standard_expert.state_summary(),
            "physics_expert": physics_expert.state_summary(),
            "selection_semantics": (
                "EXACT_STANDARD_BO_FROM_CURRENT_HISTORY"
                if state.failed_over
                else "EXACT_FIXED_PHYSICS_BO_FROM_CURRENT_HISTORY"
            ),
        },
        evidence_states=tuple(states),
        failed_over=state.failed_over,
        failover_trial=state.failover_trial,
    )
