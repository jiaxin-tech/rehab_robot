from __future__ import annotations

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from .base import Selection, unexecuted_candidates


class LowerConfidenceBoundSelector:
    allows_reference_repeat = False

    def __init__(self, *, name: str, kappa: float = 1.5) -> None:
        self.name = name
        self.kappa = float(kappa)

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection:
        if model is None:
            raise ValueError("BO requires a predictive model")
        available = unexecuted_candidates(history, domain)
        scored = []
        for item in available:
            prediction = model.predict(item)
            lcb = prediction.mean - self.kappa * prediction.std
            scored.append((lcb, item.candidate_index, item))
        value, _, selected = min(scored)
        return Selection(selected, float(value))
