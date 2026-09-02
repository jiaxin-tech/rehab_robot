"""Auditable causal ledger for executed candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .candidates import Candidate
from .observations import EpisodeObservation


INVALID_EPISODE_POLICY = (
    "INVALID_CONSUMES_BUDGET_EXCLUDED_FROM_FIT_NO_COVERT_RETRY"
)
NO_DUPLICATE_CANDIDATE = "NO_DUPLICATE_CANDIDATE"


@dataclass
class LedgerEntry:
    trial_index: int
    candidate: Candidate
    observation: EpisodeObservation
    physics_model_state_summary: dict[str, Any]
    residual_model_state_summary: dict[str, Any]
    selector: str
    acquisition_value: float | None
    selected_next_candidate: Candidate | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "candidate": self.candidate.as_dict(),
            "observation": self.observation.as_dict(),
            "validity": self.observation.valid,
            "physics_model_state_summary": self.physics_model_state_summary,
            "residual_model_state_summary": self.residual_model_state_summary,
            "selector": self.selector,
            "acquisition_value": self.acquisition_value,
            "selected_next_candidate": (
                self.selected_next_candidate.as_dict()
                if self.selected_next_candidate is not None
                else None
            ),
        }


@dataclass
class ExecutedCandidateLedger:
    entries: list[LedgerEntry] = field(default_factory=list)
    invalid_episode_policy: str = INVALID_EPISODE_POLICY
    duplicate_candidate_policy: str = NO_DUPLICATE_CANDIDATE

    @property
    def executed_candidate_ids(self) -> tuple[str, ...]:
        return tuple(entry.candidate.candidate_id for entry in self.entries)

    @property
    def observations(self) -> list[EpisodeObservation]:
        return [entry.observation for entry in self.entries]

    def append(self, entry: LedgerEntry, *, allow_reference_repeat: bool = False) -> None:
        if (
            entry.candidate.candidate_id in self.executed_candidate_ids
            and not allow_reference_repeat
        ):
            raise RuntimeError("duplicate candidate rejected by frozen policy")
        if entry.trial_index != len(self.entries) + 1:
            raise RuntimeError("ledger trial indices must be contiguous and causal")
        self.entries.append(entry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "invalid_episode_policy": self.invalid_episode_policy,
            "duplicate_candidate_policy": self.duplicate_candidate_policy,
            "entries": [entry.as_dict() for entry in self.entries],
        }
