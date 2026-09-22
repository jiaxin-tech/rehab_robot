"""K=5 reference, two active diagnostics, then fixed-expert optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..active_diagnostic_v1 import DIAGNOSTIC_SCORE_ID, DiagnosticSelection
from ..candidates import Candidate, V3CandidateDomain
from ..environment import PersonalizationEnvironment
from ..ledger import ExecutedCandidateLedger, LedgerEntry
from ..models.base import SequentialModel
from ..models.physics_graybox import PhysicsSubjectModel
from ..models.residual_gp import PhysicsInformedResidualModel
from ..models.standard_gp import StandardGaussianProcess
from ..observations import EpisodeObservation, valid_observations
from ..predictive_failover_v1 import (
    PHYSICS_MODE,
    STANDARD_MODE,
    PretrialPredictionSnapshot,
    capture_pretrial_predictions,
)
from ..selectors import LowerConfidenceBoundSelector
from .repeated import (
    PRIMARY_BUDGET,
    RepeatedDiagnosticArbitrationDecision,
    RepeatedDiagnosticEvidence,
    arbitrate_after_trial_3,
    causal_running_physics_offset,
    score_repeated_diagnostic_observation,
    select_repeated_diagnostic_candidate,
    verify_frozen_rule,
)


REPEATED_ACTIVE_DIAGNOSTIC_METHOD = "Repeated Active Diagnostic Arbitration K5 V1"


@dataclass
class RepeatedActiveDiagnosticLedgerEntry(LedgerEntry):
    trial_role: str
    pretrial_predictions: PretrialPredictionSnapshot | None
    diagnostic_score: float | None
    physics_offset: float | None
    adjusted_physics_mean: float | None
    standard_log_score: float | None
    physics_log_score: float | None
    score_difference: float | None
    evidence_sign: str | None
    cumulative_evidence_after: float | None
    arbitration_decision: str | None
    selected_expert_after_trial_3: str | None

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
            "diagnostic_score_id": (
                DIAGNOSTIC_SCORE_ID if self.diagnostic_score is not None else None
            ),
            "physics_offset": self.physics_offset,
            "adjusted_physics_mean": self.adjusted_physics_mean,
            "standard_log_score": self.standard_log_score,
            "physics_log_score": self.physics_log_score,
            "score_difference": self.score_difference,
            "evidence_sign": self.evidence_sign,
            "cumulative_evidence_after": self.cumulative_evidence_after,
            "arbitration_decision": self.arbitration_decision,
            "selected_expert_after_trial_3": self.selected_expert_after_trial_3,
        }


@dataclass
class RepeatedActiveDiagnosticSequentialRunResult:
    method: str
    budget: int
    ledger: ExecutedCandidateLedger
    best_observed_candidate: Candidate | None
    model_recommended_final_candidate: Candidate | None
    final_model_summary: dict[str, Any]
    diagnostic_selections: tuple[DiagnosticSelection, DiagnosticSelection]
    diagnostic_evidence: tuple[
        RepeatedDiagnosticEvidence, RepeatedDiagnosticEvidence
    ]
    arbitration: RepeatedDiagnosticArbitrationDecision
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
            "diagnostic_selections": [
                selection.as_dict() for selection in self.diagnostic_selections
            ],
            "diagnostic_evidence": [
                item.as_dict() for item in self.diagnostic_evidence
            ],
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


def _recommendation(model: SequentialModel, domain: V3CandidateDomain) -> Candidate:
    return min(domain, key=lambda item: (model.predict(item).mean, item.candidate_index))


def _summaries(
    standard: StandardGaussianProcess,
    physics: PhysicsInformedResidualModel,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return physics.physics.state_summary(), {
        "standard_bo": standard.state_summary(),
        "physics_bo": physics.state_summary(),
    }


def run_repeated_active_diagnostic_arbitration_k5(
    environment: PersonalizationEnvironment,
    domain: V3CandidateDomain,
    *,
    physics_model: PhysicsSubjectModel,
    budget: int = PRIMARY_BUDGET,
    kappa: float = 1.5,
) -> RepeatedActiveDiagnosticSequentialRunResult:
    """Run the frozen K=5 repeated-diagnostic information-budget variant."""

    verify_frozen_rule()
    if budget != PRIMARY_BUDGET:
        raise ValueError("Repeated Active Diagnostic Arbitration is frozen at K=5")

    standard_expert = StandardGaussianProcess()
    physics_expert = PhysicsInformedResidualModel(physics_model)
    standard_selector = LowerConfidenceBoundSelector(name="Standard BO", kappa=kappa)
    physics_selector = LowerConfidenceBoundSelector(
        name="Physics-Informed BO", kappa=kappa
    )
    ledger = ExecutedCandidateLedger()

    reference = domain.reference
    trial_1_predictions = capture_pretrial_predictions(
        trial_index=1,
        candidate=reference,
        history_size=0,
        standard_model=standard_expert,
        physics_model=physics_expert,
    )
    trial_1_observation = environment.evaluate(reference, 1)
    history = [trial_1_observation]
    standard_expert.fit(history)
    physics_expert.fit(history)
    prediction_observations = [(trial_1_predictions, trial_1_observation)]
    offset_2 = causal_running_physics_offset(
        prediction_observations, before_trial=2
    )
    diagnostic_2 = select_repeated_diagnostic_candidate(
        history,
        domain,
        trial_index=2,
        standard_model=standard_expert,
        physics_model=physics_expert,
        physics_offset=offset_2,
    )
    physics_summary, residual_summary = _summaries(standard_expert, physics_expert)
    ledger.append(
        RepeatedActiveDiagnosticLedgerEntry(
            trial_index=1,
            candidate=reference,
            observation=trial_1_observation,
            physics_model_state_summary=physics_summary,
            residual_model_state_summary=residual_summary,
            selector=DIAGNOSTIC_SCORE_ID,
            acquisition_value=diagnostic_2.diagnostic_score,
            selected_next_candidate=diagnostic_2.candidate,
            trial_role="REFERENCE_AND_DIAGNOSTIC_1_DESIGN",
            pretrial_predictions=trial_1_predictions,
            diagnostic_score=diagnostic_2.diagnostic_score,
            physics_offset=offset_2,
            adjusted_physics_mean=None,
            standard_log_score=None,
            physics_log_score=None,
            score_difference=None,
            evidence_sign=None,
            cumulative_evidence_after=0.0,
            arbitration_decision="DEFERRED_UNTIL_AFTER_TRIAL_3",
            selected_expert_after_trial_3=None,
        )
    )

    trial_2_observation = environment.evaluate(diagnostic_2.candidate, 2)
    evidence_2 = score_repeated_diagnostic_observation(
        diagnostic_2, trial_2_observation
    )
    history = [trial_1_observation, trial_2_observation]
    standard_expert.fit(history)
    physics_expert.fit(history)
    prediction_observations.append(
        (diagnostic_2.predictions, trial_2_observation)
    )
    offset_3 = causal_running_physics_offset(
        prediction_observations, before_trial=3
    )
    diagnostic_3 = select_repeated_diagnostic_candidate(
        history,
        domain,
        trial_index=3,
        standard_model=standard_expert,
        physics_model=physics_expert,
        physics_offset=offset_3,
    )
    physics_summary, residual_summary = _summaries(standard_expert, physics_expert)
    ledger.append(
        RepeatedActiveDiagnosticLedgerEntry(
            trial_index=2,
            candidate=diagnostic_2.candidate,
            observation=trial_2_observation,
            physics_model_state_summary=physics_summary,
            residual_model_state_summary=residual_summary,
            selector=DIAGNOSTIC_SCORE_ID,
            acquisition_value=diagnostic_3.diagnostic_score,
            selected_next_candidate=diagnostic_3.candidate,
            trial_role="ACTIVE_PRIOR_DIAGNOSTIC_1_AND_DIAGNOSTIC_2_DESIGN",
            pretrial_predictions=diagnostic_2.predictions,
            diagnostic_score=diagnostic_2.diagnostic_score,
            physics_offset=diagnostic_2.physics_offset,
            adjusted_physics_mean=diagnostic_2.adjusted_physics_mean,
            standard_log_score=evidence_2.standard_log_score,
            physics_log_score=evidence_2.physics_log_score,
            score_difference=evidence_2.score_difference,
            evidence_sign=evidence_2.evidence_sign,
            cumulative_evidence_after=(evidence_2.score_difference or 0.0),
            arbitration_decision="DEFERRED_UNTIL_AFTER_TRIAL_3",
            selected_expert_after_trial_3=None,
        )
    )

    trial_3_observation = environment.evaluate(diagnostic_3.candidate, 3)
    evidence_3 = score_repeated_diagnostic_observation(
        diagnostic_3, trial_3_observation
    )
    evidence = (evidence_2, evidence_3)
    arbitration = arbitrate_after_trial_3(evidence)
    selected_expert = arbitration.selected_expert
    history = [trial_1_observation, trial_2_observation, trial_3_observation]
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
    trial_4_selection = active_selector.select_next(history, domain, active_model)
    physics_summary, residual_summary = _summaries(standard_expert, physics_expert)
    ledger.append(
        RepeatedActiveDiagnosticLedgerEntry(
            trial_index=3,
            candidate=diagnostic_3.candidate,
            observation=trial_3_observation,
            physics_model_state_summary=physics_summary,
            residual_model_state_summary=residual_summary,
            selector=active_selector.name,
            acquisition_value=trial_4_selection.acquisition_value,
            selected_next_candidate=trial_4_selection.candidate,
            trial_role="ACTIVE_PRIOR_DIAGNOSTIC_2_AND_MODEL_ARBITRATION",
            pretrial_predictions=diagnostic_3.predictions,
            diagnostic_score=diagnostic_3.diagnostic_score,
            physics_offset=diagnostic_3.physics_offset,
            adjusted_physics_mean=diagnostic_3.adjusted_physics_mean,
            standard_log_score=evidence_3.standard_log_score,
            physics_log_score=evidence_3.physics_log_score,
            score_difference=evidence_3.score_difference,
            evidence_sign=evidence_3.evidence_sign,
            cumulative_evidence_after=arbitration.cumulative_evidence,
            arbitration_decision=arbitration.decision,
            selected_expert_after_trial_3=selected_expert,
        )
    )

    current = trial_4_selection.candidate
    for trial_index in (4, 5):
        observation = environment.evaluate(current, trial_index)
        history = [*ledger.observations, observation]
        standard_expert.fit(history)
        physics_expert.fit(history)
        selection = None
        if trial_index == 4:
            selection = active_selector.select_next(history, domain, active_model)
        physics_summary, residual_summary = _summaries(standard_expert, physics_expert)
        ledger.append(
            RepeatedActiveDiagnosticLedgerEntry(
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
                physics_offset=None,
                adjusted_physics_mean=None,
                standard_log_score=None,
                physics_log_score=None,
                score_difference=None,
                evidence_sign=None,
                cumulative_evidence_after=arbitration.cumulative_evidence,
                arbitration_decision="EXPERT_FIXED_AFTER_TRIAL_3",
                selected_expert_after_trial_3=selected_expert,
            )
        )
        if selection is not None:
            current = selection.candidate

    best_observed = _best_observed(ledger.observations, domain)
    recommendation = _recommendation(active_model, domain)
    return RepeatedActiveDiagnosticSequentialRunResult(
        method=REPEATED_ACTIVE_DIAGNOSTIC_METHOD,
        budget=PRIMARY_BUDGET,
        ledger=ledger,
        best_observed_candidate=best_observed,
        model_recommended_final_candidate=recommendation,
        final_model_summary={
            "model": "repeated_active_diagnostic_cumulative_expert_arbitration",
            "selected_expert": selected_expert,
            "arbitration_decision": arbitration.decision,
            "cumulative_evidence": arbitration.cumulative_evidence,
            "valid_diagnostic_evidence_count": arbitration.valid_evidence_count,
            "evidence_consistency": arbitration.evidence_consistency,
            "continuous_mixture_used": False,
            "selected_expert_fixed_after_trial_3": True,
            "selection_semantics": (
                "EXACT_STANDARD_BO_FROM_POST_DIAGNOSTIC_HISTORY"
                if selected_expert == STANDARD_MODE
                else "EXACT_FIXED_PHYSICS_BO_FROM_POST_DIAGNOSTIC_HISTORY"
            ),
            "standard_expert": standard_expert.state_summary(),
            "physics_expert": physics_expert.state_summary(),
        },
        diagnostic_selections=(diagnostic_2, diagnostic_3),
        diagnostic_evidence=evidence,
        arbitration=arbitration,
        selected_expert=selected_expert,
    )
