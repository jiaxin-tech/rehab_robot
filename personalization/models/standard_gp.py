"""Outcome-only GP baseline with no physics access."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..candidates import Candidate
from ..observations import EpisodeObservation, valid_observations
from .base import Prediction
from .residual_gp import ResidualGaussianProcess


class StandardGaussianProcess:
    uses_physics = False

    def __init__(self) -> None:
        self.gp = ResidualGaussianProcess()
        self._mean = 0.0

    def fit(self, history: list[EpisodeObservation]) -> None:
        usable = valid_observations(history)
        values = np.asarray([float(item.endpoint_value) for item in usable])
        self._mean = float(np.mean(values)) if len(values) else 0.0
        x = np.asarray(
            [[item.beta_flex / 0.03, item.beta_extend / 0.03] for item in usable],
            dtype=float,
        ).reshape((-1, 2))
        noise = np.asarray(
            [float(item.endpoint_uncertainty or 0.0) for item in usable], dtype=float
        )
        self.gp.fit_arrays(x, values - self._mean, noise)

    def predict(self, candidate: Candidate) -> Prediction:
        mean, std = self.gp.predict_beta(*candidate.beta)
        return Prediction(
            mean=self._mean + mean,
            std=std,
            valid=True,
            metadata={"physics_access": False, "outcome_only_gp": True},
        )

    def state_summary(self) -> dict[str, Any]:
        return {
            "model": "standard_outcome_only_gp",
            "physics_access": False,
            "training_mean": self._mean,
            **self.gp.state_summary(),
        }
