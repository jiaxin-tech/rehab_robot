"""Causal fixed-budget sequential personalization loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidates import Candidate, V3CandidateDomain
from .environment import PersonalizationEnvironment
from .ledger import ExecutedCandidateLedger, LedgerEntry
from .models.base import SequentialModel
from .models.physics_graybox import PhysicsSubjectModel
from .models.residual_gp import PhysicsInformedResidualModel
from .models.standard_gp import StandardGaussianProcess
from .observations import EpisodeObservation, valid_observations
from .selectors import (
    LowerConfidenceBoundSelector,
    ModelOnlyGreedySelector,
    RandomSelector,
    ReferenceSelector,
    SpaceFillingSelector,
)


METHODS = (
    "Reference",
    "Random",
    "Space Filling",
    "Model-Only Greedy",
    "Standard BO",
    "Physics-Informed BO",
)


@dataclass
class SequentialRunResult:
    method: str
    budget: int
    ledger: ExecutedCandidateLedger
    best_observed_candidate: Candidate | None
    model_recommended_final_candidate: Candidate | None
    final_model_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "budget": self.budget,
            "best_observed_candidate": (
                self.best_observed_candidate.as_dict()
                if self.best_observed_candidate
                else None
            ),
            "model_recommended_final_candidate": (
                self.model_recommended_final_candidate.as_dict()
                if self.model_recommended_final_candidate
                else None
            ),
            "final_model_summary": self.final_model_summary,
            "ledger": self.ledger.as_dict(),
        }


def _components(
    method: str,
    *,
    seed: int,
    physics_model: PhysicsSubjectModel | None,
    kappa: float,
):
    if method == "Reference":
        return ReferenceSelector(), None
    if method == "Random":
        return RandomSelector(seed), None
    if method == "Space Filling":
        return SpaceFillingSelector(), None
    if method == "Model-Only Greedy":
        if physics_model is None:
            raise ValueError("Model-Only Greedy requires physics_model")
        return ModelOnlyGreedySelector(), physics_model
    if method == "Standard BO":
        return (
            LowerConfidenceBoundSelector(name=method, kappa=kappa),
            StandardGaussianProcess(),
        )
    if method == "Physics-Informed BO":
        if physics_model is None:
            raise ValueError("Physics-Informed BO requires physics_model")
        return (
            LowerConfidenceBoundSelector(name=method, kappa=kappa),
            PhysicsInformedResidualModel(physics_model),
        )
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def _best_observed(
    history: list[EpisodeObservation], domain: V3CandidateDomain
) -> Candidate | None:
    usable = valid_observations(history)
    if not usable:
        return None
    best = min(usable, key=lambda item: (float(item.endpoint_value), item.trial_index))
    return domain.by_id(best.candidate_id)


def _model_recommendation(
    model: SequentialModel | None,
    best_observed: Candidate | None,
    domain: V3CandidateDomain,
) -> Candidate | None:
    if model is None:
        return best_observed
    return min(domain, key=lambda item: (model.predict(item).mean, item.candidate_index))


def run_sequential_personalization(
    environment: PersonalizationEnvironment,
    domain: V3CandidateDomain,
    *,
    method: str,
    budget: int = 4,
    seed: int = 0,
    physics_model: PhysicsSubjectModel | None = None,
    kappa: float = 1.5,
) -> SequentialRunResult:
    """Run exactly K adaptation trials with no future-data or oracle access."""

    if budget < 1:
        raise ValueError("budget must be >= 1")
    selector, model = _components(
        method, seed=seed, physics_model=physics_model, kappa=kappa
    )
    ledger = ExecutedCandidateLedger()
    current = domain.reference
    for trial_index in range(1, budget + 1):
        # Selection of current was completed before this observation exists.
        observation = environment.evaluate(current, trial_index)
        history = ledger.observations + [observation]
        if model is not None:
            model.fit(history)
        selection = None
        if trial_index < budget:
            selection = selector.select_next(history, domain, model)
        physics_summary: dict[str, Any] = {}
        residual_summary: dict[str, Any] = {}
        if isinstance(model, PhysicsSubjectModel):
            physics_summary = model.state_summary()
        elif isinstance(model, PhysicsInformedResidualModel):
            physics_summary = model.physics.state_summary()
            residual_summary = model.residual_gp.state_summary()
        elif model is not None:
            residual_summary = model.state_summary()
        ledger.append(
            LedgerEntry(
                trial_index=trial_index,
                candidate=current,
                observation=observation,
                physics_model_state_summary=physics_summary,
                residual_model_state_summary=residual_summary,
                selector=selector.name,
                acquisition_value=(selection.acquisition_value if selection else None),
                selected_next_candidate=(selection.candidate if selection else None),
            ),
            allow_reference_repeat=selector.allows_reference_repeat,
        )
        if selection is not None:
            current = selection.candidate

    best_observed = _best_observed(ledger.observations, domain)
    recommendation = _model_recommendation(model, best_observed, domain)
    return SequentialRunResult(
        method=method,
        budget=budget,
        ledger=ledger,
        best_observed_candidate=best_observed,
        model_recommended_final_candidate=recommendation,
        final_model_summary=model.state_summary() if model is not None else {},
    )
