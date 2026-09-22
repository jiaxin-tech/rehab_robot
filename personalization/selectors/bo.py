from __future__ import annotations

from ..candidates import V3CandidateDomain
from ..models.base import SequentialModel
from ..observations import EpisodeObservation
import math
from scipy.special import ndtr
from .base import Selection, valid_unexecuted_predictions, acquisition_metadata


class LowerConfidenceBoundSelector:
    allows_reference_repeat = False

    def __init__(self, *, name: str, kappa: float = 1.5) -> None:
        self.name = name
        self.kappa = float(kappa)
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")

    def select_next(
        self,
        history: list[EpisodeObservation],
        domain: V3CandidateDomain,
        model: SequentialModel | None,
    ) -> Selection:
        if model is None:
            raise ValueError("BO requires a predictive model")
        scored = []
        for item, prediction in valid_unexecuted_predictions(history, domain, model):
            lcb = prediction.mean - self.kappa * prediction.std
            scored.append((lcb, item.candidate_index, item, prediction))
        value, _, selected, prediction = min(scored, key=lambda row: row[:2])
        return Selection(selected, float(value), acquisition_metadata(
            history, selected, prediction, "LCB", value, kappa=self.kappa))


def expected_improvement(mean: float, std: float, incumbent: float, xi: float = 0.0) -> float:
    """Minimization EI; near-zero sigma uses the deterministic improvement limit."""
    if not all(math.isfinite(x) for x in (mean, std, incumbent, xi)) or std < 0 or xi < 0:
        raise ValueError("INVALID_EI_ARGUMENTS")
    improvement = incumbent - mean - xi
    if std <= 1.0e-12:
        return max(improvement, 0.0)
    z = improvement / std
    return max(float(improvement * ndtr(z) + std * math.exp(-0.5*z*z) / math.sqrt(2*math.pi)), 0.0)


class ExpectedImprovementSelector:
    allows_reference_repeat = False

    def __init__(self, *, name="MODEL_INFORMED_BO_EI", xi=0.0):
        if not math.isfinite(xi) or xi < 0:
            raise ValueError("xi must be finite and nonnegative")
        self.name, self.xi = name, float(xi)

    def select_next(self, history, domain, model):
        measured = [float(item.endpoint_value) for item in history if item.valid]
        if not measured:
            raise RuntimeError("EI_REQUIRES_VALID_INCUMBENT")
        if model is None:
            raise ValueError("EI requires a predictive model")
        incumbent = min(measured)
        scored = []
        for candidate, prediction in valid_unexecuted_predictions(history, domain, model):
            value = expected_improvement(prediction.mean, prediction.std, incumbent, self.xi)
            scored.append((value, candidate, prediction))
        value, candidate, prediction = min(scored, key=lambda row: (-row[0], row[1].candidate_index))
        return Selection(candidate, value, acquisition_metadata(
            history, candidate, prediction, "EI", value, xi=self.xi))
