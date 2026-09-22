from __future__ import annotations

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from .base import Selection, valid_unexecuted_predictions, acquisition_metadata


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
        scored = valid_unexecuted_predictions(history, domain, model)
        selected, prediction = min(scored, key=lambda row: (row[1].mean, row[0].candidate_index))
        return Selection(selected, prediction.mean, acquisition_metadata(
            history, selected, prediction, "MODEL_MEAN", prediction.mean))
