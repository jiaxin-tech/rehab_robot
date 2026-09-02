"""Strict observation boundary for offline algorithm development."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .candidates import Candidate, V3CandidateDomain
from .observations import EpisodeObservation


OFFLINE_ALGORITHM_TEST_CASE = "OFFLINE_ALGORITHM_TEST_CASE"
OFFLINE_ALGORITHM_DEVELOPMENT_EVALUATION = (
    "OFFLINE_ALGORITHM_DEVELOPMENT_EVALUATION"
)
REAL_ROBOT_ENVIRONMENT_DISABLED = "REAL_ROBOT_ENVIRONMENT_DISABLED"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"


class PersonalizationEnvironment(ABC):
    """The algorithm's sole source of an episode observation."""

    @abstractmethod
    def evaluate(self, candidate: Candidate, trial_index: int) -> EpisodeObservation:
        raise NotImplementedError


def _stable_normal(seed: int, candidate_id: str, trial_index: int) -> float:
    payload = f"{seed}|{candidate_id}|{trial_index}".encode("utf-8")
    raw = hashlib.sha256(payload).digest()
    child_seed = int.from_bytes(raw[:8], "big", signed=False)
    return float(np.random.default_rng(child_seed).normal())


@dataclass(frozen=True)
class AnalyticCase:
    name: str
    landscape: str
    optimum_beta: tuple[float, float]
    noise_std: float = 0.0
    invalid_candidate_ids: frozenset[str] = frozenset()
    outlier_candidate_ids: frozenset[str] = frozenset()
    outlier_magnitude: float = 1.5


class AnalyticBenchmarkEnvironment(PersonalizationEnvironment):
    """Known-truth deterministic-seed benchmark; never a virtual patient."""

    endpoint_name = "offline_synthetic_mechanical_cost"
    endpoint_unit = "normalized_cost"

    def __init__(
        self,
        domain: V3CandidateDomain,
        case: AnalyticCase,
        *,
        seed: int,
    ) -> None:
        self.domain = domain
        self.case = case
        self.seed = int(seed)
        self.oracle_access_count = 0
        self._optimum = min(domain, key=lambda item: self._truth(item))
        self._optimum_value = self._truth(self._optimum)

    def _truth(self, candidate: Candidate) -> float:
        x = candidate.beta_flex / 0.03
        z = candidate.beta_extend / 0.03
        ox = self.case.optimum_beta[0] / 0.03
        oz = self.case.optimum_beta[1] / 0.03
        dx, dz = x - ox, z - oz
        if self.case.landscape == "smooth_convex":
            return 0.55 * dx * dx + 0.45 * dz * dz - 0.25
        if self.case.landscape == "anisotropic":
            return 1.2 * dx * dx + 0.16 * dz * dz - 0.35
        if self.case.landscape == "rotated":
            u = (dx + dz) / math.sqrt(2.0)
            v = (dx - dz) / math.sqrt(2.0)
            return 0.9 * u * u + 0.2 * v * v - 0.3
        if self.case.landscape == "mildly_nonlinear":
            return (
                0.5 * dx * dx
                + 0.35 * dz * dz
                + 0.09 * math.sin(3.0 * x - 1.5 * z)
                - 0.3
            )
        raise ValueError(f"unknown analytic landscape: {self.case.landscape}")

    def evaluate(self, candidate: Candidate, trial_index: int) -> EpisodeObservation:
        metadata = {
            "classification": OFFLINE_ALGORITHM_TEST_CASE,
            "case": self.case.name,
            "seed": self.seed,
            "truth_hidden_from_selector": True,
        }
        if candidate.candidate_id in self.case.invalid_candidate_ids:
            return EpisodeObservation(
                episode_id=f"{self.case.name}:{self.seed}:{trial_index}",
                trial_index=trial_index,
                candidate_id=candidate.candidate_id,
                beta_flex=candidate.beta_flex,
                beta_extend=candidate.beta_extend,
                endpoint_name=self.endpoint_name,
                endpoint_value=None,
                endpoint_unit=self.endpoint_unit,
                endpoint_uncertainty=self.case.noise_std,
                valid=False,
                invalid_reason="INJECTED_INVALID_OFFLINE_EPISODE",
                metadata=metadata,
            )
        noise = self.case.noise_std * _stable_normal(
            self.seed, candidate.candidate_id, trial_index
        )
        outlier = (
            self.case.outlier_magnitude
            if candidate.candidate_id in self.case.outlier_candidate_ids
            else 0.0
        )
        return EpisodeObservation(
            episode_id=f"{self.case.name}:{self.seed}:{trial_index}",
            trial_index=trial_index,
            candidate_id=candidate.candidate_id,
            beta_flex=candidate.beta_flex,
            beta_extend=candidate.beta_extend,
            endpoint_name=self.endpoint_name,
            endpoint_value=self._truth(candidate) + noise + outlier,
            endpoint_unit=self.endpoint_unit,
            endpoint_uncertainty=self.case.noise_std,
            valid=True,
            metadata={**metadata, "outlier_injected": bool(outlier)},
        )

    # These methods belong to the post-run evaluator, never to a selector/model.
    def oracle_value(self, candidate: Candidate) -> float:
        self.oracle_access_count += 1
        return self._truth(candidate)

    def oracle_optimum(self) -> tuple[Candidate, float]:
        self.oracle_access_count += 1
        return self._optimum, self._optimum_value


class FrozenOfflineReplayEnvironment(PersonalizationEnvironment):
    """Replay table owner; only the requested executed row is revealed."""

    def __init__(
        self,
        values: Mapping[str, float | None],
        *,
        endpoint_name: str,
        endpoint_unit: str,
    ) -> None:
        self._values = dict(values)
        self.endpoint_name = endpoint_name
        self.endpoint_unit = endpoint_unit
        self.revealed_candidate_ids: list[str] = []

    def evaluate(self, candidate: Candidate, trial_index: int) -> EpisodeObservation:
        if candidate.candidate_id not in self._values:
            raise KeyError("candidate is absent from frozen replay artifact")
        self.revealed_candidate_ids.append(candidate.candidate_id)
        value = self._values[candidate.candidate_id]
        valid = value is not None and math.isfinite(float(value))
        return EpisodeObservation(
            episode_id=f"frozen-replay:{trial_index}",
            trial_index=trial_index,
            candidate_id=candidate.candidate_id,
            beta_flex=candidate.beta_flex,
            beta_extend=candidate.beta_extend,
            endpoint_name=self.endpoint_name,
            endpoint_value=float(value) if valid else None,
            endpoint_unit=self.endpoint_unit,
            endpoint_uncertainty=None,
            valid=valid,
            invalid_reason=None if valid else "FROZEN_REPLAY_VALUE_MISSING_OR_INVALID",
            metadata={"classification": "FROZEN_DEVELOPMENT_REPLAY"},
        )


class RealRobotEnvironment(PersonalizationEnvironment):
    """Fail-closed placeholder.  It intentionally imports no hardware API."""

    def evaluate(self, candidate: Candidate, trial_index: int) -> EpisodeObservation:
        del candidate, trial_index
        raise RuntimeError(
            f"{REAL_ROBOT_ENVIRONMENT_DISABLED}: {NOT_ROBOT_APPROVED}"
        )


def make_primary_cases(noise_std: float = 0.0) -> tuple[AnalyticCase, ...]:
    return (
        AnalyticCase("smooth_convex", "smooth_convex", (0.0125, -0.01), noise_std),
        AnalyticCase("anisotropic", "anisotropic", (-0.015, 0.0175), noise_std),
        AnalyticCase("rotated", "rotated", (0.0175, 0.0125), noise_std),
        AnalyticCase(
            "mildly_nonlinear", "mildly_nonlinear", (-0.0125, -0.0175), noise_std
        ),
    )
