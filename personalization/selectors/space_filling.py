from __future__ import annotations

import math

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from .base import Selection, unexecuted_candidates


class SpaceFillingSelector:
    name = "Space Filling"
    allows_reference_repeat = False

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection:
        del model
        available = unexecuted_candidates(history, domain)
        executed = [(item.beta_flex, item.beta_extend) for item in history]
        if not executed:
            return Selection(domain.reference, None)

        def minimum_distance(candidate):
            return min(
                math.hypot(candidate.beta_flex - x, candidate.beta_extend - z)
                for x, z in executed
            )

        selected = min(
            available,
            key=lambda item: (-minimum_distance(item), item.candidate_index),
        )
        return Selection(selected, minimum_distance(selected))
