"""Deterministic low-dimensional residual Gaussian process."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..candidates import Candidate
from ..observations import EpisodeObservation, valid_observations
from .base import Prediction
from .physics_graybox import PhysicsSubjectModel


class ResidualGaussianProcess:
    """Fixed Matern-5/2 GP; no case-specific kernel or oracle tuning."""

    def __init__(
        self,
        *,
        length_scale: float = 0.7,
        signal_std: float = 0.6,
        jitter: float = 1.0e-9,
    ) -> None:
        self.length_scale = float(length_scale)
        self.signal_std = float(signal_std)
        self.jitter = float(jitter)
        self._x = np.empty((0, 2), dtype=float)
        self._y = np.empty(0, dtype=float)
        self._noise = np.empty(0, dtype=float)
        self._chol: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    @staticmethod
    def _features(beta_flex: float, beta_extend: float) -> np.ndarray:
        return np.asarray([beta_flex / 0.03, beta_extend / 0.03], dtype=float)

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(
            a[:, None, :] - b[None, :, :], axis=2
        ) / self.length_scale
        scaled = math.sqrt(5.0) * distance
        return self.signal_std**2 * (1.0 + scaled + scaled**2 / 3.0) * np.exp(
            -scaled
        )

    def fit_arrays(
        self,
        x: np.ndarray,
        y: np.ndarray,
        noise_std: np.ndarray | None = None,
    ) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != (len(y), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("GP training arrays must be finite Nx2 and N")
        noise = (
            np.full(len(y), 1.0e-4, dtype=float)
            if noise_std is None
            else np.maximum(np.asarray(noise_std, dtype=float), 1.0e-6)
        )
        self._x, self._y, self._noise = x.copy(), y.copy(), noise.copy()
        if not len(y):
            self._chol = self._alpha = None
            return
        covariance = self._kernel(x, x) + np.diag(noise**2 + self.jitter)
        extra = self.jitter
        for _ in range(6):
            try:
                self._chol = np.linalg.cholesky(covariance + extra * np.eye(len(y)))
                break
            except np.linalg.LinAlgError:
                extra *= 10.0
        else:
            raise RuntimeError("GP covariance remained non-positive-definite")
        self._alpha = np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, y)
        )

    def predict_beta(self, beta_flex: float, beta_extend: float) -> tuple[float, float]:
        point = self._features(beta_flex, beta_extend)[None, :]
        prior_std = self.signal_std
        if self._chol is None or self._alpha is None or not len(self._x):
            return 0.0, prior_std
        cross = self._kernel(self._x, point)[:, 0]
        mean = float(cross @ self._alpha)
        projected = np.linalg.solve(self._chol, cross)
        variance = max(self.signal_std**2 - float(projected @ projected), 0.0)
        return mean, math.sqrt(variance)

    def state_summary(self) -> dict[str, Any]:
        return {
            "model": "fixed_matern52_gaussian_process",
            "training_count": len(self._y),
            "length_scale_normalized_beta": self.length_scale,
            "signal_std": self.signal_std,
            "hyperparameter_optimization": "disabled_fixed_rule",
        }


class PhysicsInformedResidualModel:
    """Total prediction = physics prediction + residual GP posterior."""

    uses_physics = True

    def __init__(
        self,
        physics: PhysicsSubjectModel,
        residual_gp: ResidualGaussianProcess | None = None,
    ) -> None:
        self.physics = physics
        self.residual_gp = residual_gp or ResidualGaussianProcess()

    def fit(self, history: list[EpisodeObservation]) -> None:
        self.physics.fit(history)
        usable = valid_observations(history)
        x, residuals, noise = [], [], []
        for item in usable:
            candidate = Candidate(
                item.candidate_id,
                item.beta_flex,
                item.beta_extend,
                item.trial_index,
            )
            physics_value = self.physics.predict(candidate).mean
            x.append([item.beta_flex / 0.03, item.beta_extend / 0.03])
            residuals.append(float(item.endpoint_value) - physics_value)
            noise.append(float(item.endpoint_uncertainty or 0.0))
        self.residual_gp.fit_arrays(
            np.asarray(x, dtype=float).reshape((-1, 2)),
            np.asarray(residuals, dtype=float),
            np.asarray(noise, dtype=float),
        )

    def predict(self, candidate: Candidate) -> Prediction:
        physics = self.physics.predict(candidate)
        residual_mean, residual_std = self.residual_gp.predict_beta(*candidate.beta)
        return Prediction(
            mean=physics.mean + residual_mean,
            std=residual_std,
            valid=physics.valid,
            metadata={
                "physics_mean": physics.mean,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "total_prediction_identity": "physics_mean + residual_mean",
            },
        )

    def state_summary(self) -> dict[str, Any]:
        return {
            "model": "physics_plus_residual_gp",
            "physics": self.physics.state_summary(),
            "residual": self.residual_gp.state_summary(),
        }
