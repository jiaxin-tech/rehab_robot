"""Finite predictive mixture of existing Standard and Physics-Informed BO."""

from __future__ import annotations

import math
from typing import Any

from ..candidates import Candidate
from ..models.base import Prediction
from ..models.physics_graybox import PhysicsSubjectModel
from ..models.residual_gp import PhysicsInformedResidualModel
from ..models.standard_gp import StandardGaussianProcess
from ..observations import EpisodeObservation


class AdaptivePhysicsMixtureModel:
    """Moment representation of a two-component predictive mixture.

    This is not claimed to be one exact Gaussian-process posterior.  The two
    existing component models remain unchanged.
    """

    def __init__(self, physics_model: PhysicsSubjectModel, *, trust_score: float) -> None:
        self.standard = StandardGaussianProcess()
        self.physics_informed = PhysicsInformedResidualModel(physics_model)
        self.set_trust(trust_score)

    @property
    def physics(self) -> PhysicsSubjectModel:
        return self.physics_informed.physics

    def set_trust(self, trust_score: float) -> None:
        if not 0.0 <= trust_score <= 1.0:
            raise ValueError("adaptive mixture trust must be in [0,1]")
        self.trust_score = float(trust_score)

    def fit(self, history: list[EpisodeObservation]) -> None:
        self.standard.fit(history)
        if self.trust_score > 0.0:
            self.physics_informed.fit(history)

    def predict(self, candidate: Candidate) -> Prediction:
        standard = self.standard.predict(candidate)
        if self.trust_score == 0.0:
            return Prediction(
                mean=standard.mean,
                std=standard.std,
                valid=standard.valid,
                metadata={
                    "mixture_semantics": "EXACT_STANDARD_BO_LIMIT",
                    "trust_score": 0.0,
                    "standard_mean": standard.mean,
                    "standard_std": standard.std,
                    "physics_contribution_disabled": True,
                },
            )
        physics = self.physics_informed.predict(candidate)
        if self.trust_score == 1.0:
            return Prediction(
                mean=physics.mean,
                std=physics.std,
                valid=physics.valid,
                metadata={
                    **physics.metadata,
                    "mixture_semantics": "EXACT_FIXED_PHYSICS_BO_LIMIT",
                    "trust_score": 1.0,
                    "physics_contribution_disabled": False,
                },
            )
        weight = self.trust_score
        mean = (1.0 - weight) * standard.mean + weight * physics.mean
        variance = (1.0 - weight) * (
            standard.std**2 + (standard.mean - mean) ** 2
        ) + weight * (physics.std**2 + (physics.mean - mean) ** 2)
        return Prediction(
            mean=float(mean),
            std=float(math.sqrt(max(variance, 0.0))),
            valid=standard.valid and physics.valid,
            metadata={
                "mixture_semantics": "FINITE_MIXTURE_MOMENTS_NOT_SINGLE_GP_POSTERIOR",
                "trust_score": weight,
                "standard_mean": standard.mean,
                "standard_std": standard.std,
                "physics_informed_mean": physics.mean,
                "physics_informed_std": physics.std,
                "between_component_variance_included": True,
                "physics_contribution_disabled": False,
            },
        )

    def state_summary(self) -> dict[str, Any]:
        return {
            "model": "adaptive_standard_and_physics_informed_predictive_mixture",
            "trust_score": self.trust_score,
            "mixture_is_single_exact_gp_posterior": False,
            "standard_component": self.standard.state_summary(),
            "physics_informed_component": self.physics_informed.state_summary(),
            "limit_semantics": {
                "trust_0": "EXACT_STANDARD_BO",
                "trust_1": "EXACT_FIXED_PHYSICS_INFORMED_BO",
            },
        }
