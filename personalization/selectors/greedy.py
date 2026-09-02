from __future__ import annotations

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from .base import Selection, unexecuted_candidates


class ModelOnlyGreedySelector:
    name = "Model-Only Greedy"
    allows_reference_repeat = False

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection:
        if model is None:
            raise ValueError("model-only greedy requires a physics model")
        available = unexecuted_candidates(history, domain)
        scored = [(model.predict(item).mean, item.candidate_index, item) for item in available]
        value, _, selected = min(scored)
        return Selection(selected, float(value))
