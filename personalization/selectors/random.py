from __future__ import annotations

import hashlib
import random

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
from .base import Selection, unexecuted_candidates


class RandomSelector:
    name = "Random"
    allows_reference_repeat = False

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection:
        del model
        available = unexecuted_candidates(history, domain)
        payload = f"{self.seed}|{len(history) + 1}".encode("utf-8")
        child_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return Selection(random.Random(child_seed).choice(available), None)
