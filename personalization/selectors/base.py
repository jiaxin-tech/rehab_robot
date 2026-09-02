"""Unified selector interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..candidates import Candidate, V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    acquisition_value: float | None


class Selector(Protocol):
    name: str
    allows_reference_repeat: bool

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection: ...


def unexecuted_candidates(
    history: list[EpisodeObservation], domain: V3CandidateDomain
) -> list[Candidate]:
    executed = {item.candidate_id for item in history}
    available = [item for item in domain if item.candidate_id not in executed]
    if not available:
        raise RuntimeError("candidate domain exhausted under no-duplicate policy")
    return available
