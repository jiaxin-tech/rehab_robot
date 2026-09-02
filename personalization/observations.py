"""Hardware-independent episode observation contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeObservation:
    episode_id: str
    trial_index: int
    candidate_id: str
    beta_flex: float
    beta_extend: float
    endpoint_name: str
    endpoint_value: float | None
    endpoint_unit: str
    endpoint_uncertainty: float | None
    valid: bool
    invalid_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trial_index < 1:
            raise ValueError("trial_index must be >= 1")
        if not self.endpoint_name or self.endpoint_name.lower() == "comfort":
            raise ValueError("endpoint must be explicit and cannot be hard-coded comfort")
        if self.valid:
            if self.endpoint_value is None or not math.isfinite(self.endpoint_value):
                raise ValueError("valid observations require a finite endpoint")
            if self.invalid_reason:
                raise ValueError("valid observations cannot have invalid_reason")
        else:
            if self.endpoint_value is not None:
                raise ValueError("invalid observations must retain a missing endpoint, not zero")
            if not self.invalid_reason:
                raise ValueError("invalid observations require invalid_reason")
        if self.endpoint_uncertainty is not None and (
            not math.isfinite(self.endpoint_uncertainty)
            or self.endpoint_uncertainty < 0.0
        ):
            raise ValueError("endpoint uncertainty must be finite and non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trial_index": self.trial_index,
            "candidate_id": self.candidate_id,
            "beta_flex": self.beta_flex,
            "beta_extend": self.beta_extend,
            "endpoint_name": self.endpoint_name,
            "endpoint_value": self.endpoint_value,
            "endpoint_unit": self.endpoint_unit,
            "endpoint_uncertainty": self.endpoint_uncertainty,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "metadata": dict(self.metadata),
        }


def valid_observations(history: list[EpisodeObservation]) -> list[EpisodeObservation]:
    return [item for item in history if item.valid]
