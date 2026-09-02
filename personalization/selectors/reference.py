from __future__ import annotations

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from .base import Selection


class ReferenceSelector:
    name = "Reference"
    allows_reference_repeat = True

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection:
        del history, model
        return Selection(domain.reference, None)
