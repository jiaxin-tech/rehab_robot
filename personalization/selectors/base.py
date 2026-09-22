"""Unified selector interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
from typing import Protocol

from ..candidates import Candidate, V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    acquisition_value: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


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


def valid_unexecuted_predictions(history, domain, model):
    """Prediction validity only; this is not a physical safety gate."""
    predictions = []
    for candidate in unexecuted_candidates(history, domain):
        prediction = model.predict(candidate)
        if (prediction.valid and math.isfinite(prediction.mean)
                and math.isfinite(prediction.std) and prediction.std >= 0):
            predictions.append((candidate, prediction))
    if not predictions:
        raise RuntimeError("NO_VALID_UNEXECUTED_PREDICTION")
    return predictions


def acquisition_metadata(history, candidate, prediction, kind, value, **parameters):
    measured = [float(item.endpoint_value) for item in history if item.valid]
    return {
        "acquisition_type": kind, "acquisition_value": float(value),
        "predictive_mean": prediction.mean, "predictive_std": prediction.std,
        "incumbent_measured_best": min(measured) if measured else None,
        "candidate_id": candidate.candidate_id, "beta": list(candidate.beta),
        "trial_index": max((item.trial_index for item in history), default=0) + 1,
        **parameters,
    }
