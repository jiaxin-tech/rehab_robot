"""K=4 active prior diagnostic followed by one-shot expert arbitration."""

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
from ..predictive_failover_v1.evidence import (
    PHYSICS_MODE,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
    capture_pretrial_predictions,
)
from ..selectors import LowerConfidenceBoundSelector
from .diagnostic import (
    DIAGNOSTIC_SCORE_ID,
    DiagnosticArbitrationDecision,
    DiagnosticSelection,
    arbitrate_after_diagnostic,
    causal_trial_1_physics_offset,
    select_active_diagnostic_candidate,
    verify_frozen_rule,
)


ACTIVE_DIAGNOSTIC_METHOD = "Active Prior Diagnostic Arbitration V1"
PRIMARY_ADAPTATION_BUDGET = 4


@dataclass
class ActiveDiagnosticLedgerEntry(LedgerEntry):
    trial_role: str
    pretrial_predictions: PretrialPredictionSnapshot | None
    diagnostic_score: float | None
    diagnostic_score_id: str | None
    physics_offset: float | None
    adjusted_physics_mean: float | None
    arbitration_decision: str | None
    selected_expert_after_trial_2: str | None
    standard_log_score: float | None
    physics_log_score: float | None
    score_difference: float | None

    def as_dict(self) -> dict[str, Any]:
        snapshot = self.pretrial_predictions
        return {
            **super().as_dict(),
            "trial_role": self.trial_role,
            "pretrial_predictions": snapshot.as_dict() if snapshot else None,
            "standard_bo_mean_before": (
                snapshot.standard_bo_mean_before if snapshot else None
            ),
            "standard_bo_std_before": (
                snapshot.standard_bo_std_before if snapshot else None
            ),
            "physics_bo_mean_before": (
                snapshot.physics_bo_mean_before if snapshot else None
            ),
            "physics_bo_std_before": (
                snapshot.physics_bo_std_before if snapshot else None
            ),
            "diagnostic_score": self.diagnostic_score,
            "diagnostic_score_id": self.diagnostic_score_id,
            "physics_offset": self.physics_offset,
            "adjusted_physics_mean": self.adjusted_physics_mean,
            "arbitration_decision": self.arbitration_decision,
            "selected_expert_after_trial_2": self.selected_expert_after_trial_2,
            "standard_log_score": self.standard_log_score,
            "physics_log_score": self.physics_log_score,
            "score_difference": self.score_difference,
        }


@dataclass
class ActiveDiagnosticSequentialRunResult:
    method: str
    budget: int
    ledger: ExecutedCandidateLedger
    best_observed_candidate: Candidate | None
    model_recommended_final_candidate: Candidate | None
    final_model_summary: dict[str, Any]
    diagnostic_selection: DiagnosticSelection
    arbitration: DiagnosticArbitrationDecision
    selected_expert: str

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
            "diagnostic_selection": self.diagnostic_selection.as_dict(),
            "arbitration": self.arbitration.as_dict(),
            "selected_expert": self.selected_expert,
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


def _model_recommendation(model: SequentialModel, domain: V3CandidateDomain) -> Candidate:
    return min(domain, key=lambda item: (model.predict(item).mean, item.candidate_index))


def _model_summaries(
    standard: StandardGaussianProcess,
    physics: PhysicsInformedResidualModel,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return physics.physics.state_summary(), {
        "standard_bo": standard.state_summary(),
        "physics_bo": physics.state_summary(),
    }


def run_active_prior_diagnostic_arbitration(
    environment: PersonalizationEnvironment,
    domain: V3CandidateDomain,
    *,
    physics_model: PhysicsSubjectModel,
    budget: int = PRIMARY_ADAPTATION_BUDGET,
    kappa: float = 1.5,
) -> ActiveDiagnosticSequentialRunResult:
    """Run reference, active diagnostic, then two fixed-expert BO trials."""

    verify_frozen_rule()
    if budget != PRIMARY_ADAPTATION_BUDGET:
        raise ValueError("Active Diagnostic Arbitration V1 is frozen at K=4")

    standard_expert = StandardGaussianProcess()
    physics_expert = PhysicsInformedResidualModel(physics_model)
    standard_selector = LowerConfidenceBoundSelector(name="Standard BO", kappa=kappa)
    physics_selector = LowerConfidenceBoundSelector(
        name="Physics-Informed BO", kappa=kappa
    )
    ledger = ExecutedCandidateLedger()

    # Trial 1 is always the frozen-domain reference.
    reference = domain.reference
    trial_1_predictions = capture_pretrial_predictions(
        trial_index=1,
        candidate=reference,
        history_size=0,
        standard_model=standard_expert,
        physics_model=physics_expert,
    )
    trial_1_observation = environment.evaluate(reference, 1)
    trial_1_history = [trial_1_observation]
    standard_expert.fit(trial_1_history)
    physics_expert.fit(trial_1_history)
    physics_offset = causal_trial_1_physics_offset(
        trial_1_predictions, trial_1_observation
    )
    diagnostic = select_active_diagnostic_candidate(
        trial_1_history,
        domain,
        standard_model=standard_expert,
        physics_model=physics_expert,
        physics_offset=physics_offset,
    )
    physics_summary, residual_summary = _model_summaries(
        standard_expert, physics_expert
    )
    ledger.append(
        ActiveDiagnosticLedgerEntry(
            trial_index=1,
            candidate=reference,
            observation=trial_1_observation,
            physics_model_state_summary=physics_summary,
            residual_model_state_summary=residual_summary,
            selector=DIAGNOSTIC_SCORE_ID,
            acquisition_value=diagnostic.diagnostic_score,
            selected_next_candidate=diagnostic.candidate,
            trial_role="REFERENCE_AND_DIAGNOSTIC_DESIGN",
            pretrial_predictions=trial_1_predictions,
            diagnostic_score=diagnostic.diagnostic_score,
            diagnostic_score_id=DIAGNOSTIC_SCORE_ID,
            physics_offset=physics_offset,
            adjusted_physics_mean=None,
            arbitration_decision=None,
            selected_expert_after_trial_2=None,
            standard_log_score=None,
            physics_log_score=None,
            score_difference=None,
        )
    )

    # Trial 2 prediction and divergence were frozen above, before this reveal.
    trial_2_observation = environment.evaluate(diagnostic.candidate, 2)
    arbitration = arbitrate_after_diagnostic(diagnostic, trial_2_observation)
    selected_expert = arbitration.selected_expert
    history = [trial_1_observation, trial_2_observation]
    standard_expert.fit(history)
    physics_expert.fit(history)
    if selected_expert == STANDARD_MODE:
        active_model: SequentialModel = standard_expert
        active_selector = standard_selector
    elif selected_expert == PHYSICS_MODE:
        active_model = physics_expert
        active_selector = physics_selector
    else:
        raise RuntimeError(f"unknown selected expert: {selected_expert}")
    trial_3_selection = active_selector.select_next(history, domain, active_model)
    physics_summary, residual_summary = _model_summaries(
        standard_expert, physics_expert
    )
    ledger.append(
        ActiveDiagnosticLedgerEntry(
            trial_index=2,
            candidate=diagnostic.candidate,
            observation=trial_2_observation,
            physics_model_state_summary=physics_summary,
            residual_model_state_summary=residual_summary,
            selector=active_selector.name,
            acquisition_value=trial_3_selection.acquisition_value,
            selected_next_candidate=trial_3_selection.candidate,
            trial_role="ACTIVE_PRIOR_DIAGNOSTIC_AND_MODEL_ARBITRATION",
            pretrial_predictions=diagnostic.predictions,
            diagnostic_score=diagnostic.diagnostic_score,
            diagnostic_score_id=DIAGNOSTIC_SCORE_ID,
            physics_offset=diagnostic.physics_offset,
            adjusted_physics_mean=diagnostic.adjusted_physics_mean,
            arbitration_decision=arbitration.decision,
            selected_expert_after_trial_2=selected_expert,
            standard_log_score=arbitration.standard_log_score,
            physics_log_score=arbitration.physics_log_score,
            score_difference=arbitration.score_difference,
        )
    )

    current = trial_3_selection.candidate
    for trial_index in (3, 4):
        observation = environment.evaluate(current, trial_index)
        history = [*ledger.observations, observation]
        standard_expert.fit(history)
        physics_expert.fit(history)
        selection = None
        if trial_index == 3:
            selection = active_selector.select_next(history, domain, active_model)
        physics_summary, residual_summary = _model_summaries(
            standard_expert, physics_expert
        )
        ledger.append(
            ActiveDiagnosticLedgerEntry(
                trial_index=trial_index,
                candidate=current,
                observation=observation,
                physics_model_state_summary=physics_summary,
                residual_model_state_summary=residual_summary,
                selector=active_selector.name,
                acquisition_value=(selection.acquisition_value if selection else None),
                selected_next_candidate=(selection.candidate if selection else None),
                trial_role="FIXED_SELECTED_EXPERT_OPTIMIZATION",
                pretrial_predictions=None,
                diagnostic_score=None,
                diagnostic_score_id=None,
                physics_offset=None,
                adjusted_physics_mean=None,
                arbitration_decision="EXPERT_FIXED_AFTER_TRIAL_2",
                selected_expert_after_trial_2=selected_expert,
                standard_log_score=None,
                physics_log_score=None,
                score_difference=None,
            )
        )
        if selection is not None:
            current = selection.candidate

    best_observed = _best_observed(ledger.observations, domain)
    recommendation = _model_recommendation(active_model, domain)
    return ActiveDiagnosticSequentialRunResult(
        method=ACTIVE_DIAGNOSTIC_METHOD,
        budget=budget,
        ledger=ledger,
        best_observed_candidate=best_observed,
        model_recommended_final_candidate=recommendation,
        final_model_summary={
            "model": "active_prior_diagnostic_single_expert_arbitration",
            "diagnostic_score_id": DIAGNOSTIC_SCORE_ID,
            "selected_expert": selected_expert,
            "arbitration_decision": arbitration.decision,
            "physics_offset_from_trial_1": physics_offset,
            "continuous_mixture_used": False,
            "selected_expert_fixed_after_trial_2": True,
            "selection_semantics": (
                "EXACT_STANDARD_BO_FROM_POST_DIAGNOSTIC_HISTORY"
                if selected_expert == STANDARD_MODE
                else "EXACT_FIXED_PHYSICS_BO_FROM_POST_DIAGNOSTIC_HISTORY"
            ),
            "standard_expert": standard_expert.state_summary(),
            "physics_expert": physics_expert.state_summary(),
        },
        diagnostic_selection=diagnostic,
        arbitration=arbitration,
        selected_expert=selected_expert,
    )
