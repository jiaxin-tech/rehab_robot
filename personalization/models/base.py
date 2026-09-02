"""Shared prediction contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..candidates import Candidate
from ..observations import EpisodeObservation


@dataclass(frozen=True)
class Prediction:
    mean: float
    std: float
    valid: bool
    metadata: dict[str, Any]


class SequentialModel(Protocol):
    def fit(self, history: list[EpisodeObservation]) -> None: ...

    def predict(self, candidate: Candidate) -> Prediction: ...

    def state_summary(self) -> dict[str, Any]: ...
