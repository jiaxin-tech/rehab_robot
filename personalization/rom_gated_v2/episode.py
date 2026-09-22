"""Frozen-ROM handoff into the unchanged V1 sequential core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..environment import PersonalizationEnvironment
from ..models.physics_graybox import PhysicsSubjectModel
from ..sequential import SequentialRunResult, run_sequential_personalization
from .reference import SubjectSpecificV3CandidateDomain
from .rom import SubjectROMProfile


PRIMARY_PERSONALIZATION_ADAPTATION_BUDGET = 4
PERSONALIZATION_ADAPTATION_BUDGET_SENSITIVITY = (3, 5)
PERSONALIZATION_TRIAL_LEDGER = "PERSONALIZATION_TRIAL_LEDGER"


@dataclass(frozen=True)
class ROMGatedRunResult:
    episode_id: str
    rom_profile_id: str
    rom_profile_fingerprint: str
    sequential_result: SequentialRunResult

    def as_dict(self) -> dict[str, Any]:
        payload = self.sequential_result.as_dict()
        ledger = payload.pop("ledger")
        return {
            "episode_id": self.episode_id,
            "rom_profile_id": self.rom_profile_id,
            "rom_profile_fingerprint": self.rom_profile_fingerprint,
            "adaptation_budget": self.sequential_result.budget,
            "ledger_type": PERSONALIZATION_TRIAL_LEDGER,
            "personalization_trial_ledger": ledger,
            **payload,
        }


class ROMGatedPersonalizationEpisode:
    """One immutable-domain Stage 1 episode.

    A changed ROM never mutates this object.  ``restart_for_changed_rom``
    closes it and returns a distinct episode bound to the new profile/domain.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        profile: SubjectROMProfile,
        domain: SubjectSpecificV3CandidateDomain,
        adaptation_budget: int = PRIMARY_PERSONALIZATION_ADAPTATION_BUDGET,
    ) -> None:
        if not profile.frozen:
            raise RuntimeError("Stage 1 cannot start until ROM_PROFILE_FROZEN=true")
        if domain.profile.fingerprint != profile.fingerprint:
            raise RuntimeError("Stage 1 domain does not match its frozen ROM profile")
        if adaptation_budget < 1:
            raise ValueError("personalization adaptation budget must be >= 1")
        self.episode_id = episode_id
        self._profile = profile
        self.domain = domain
        self.adaptation_budget = int(adaptation_budget)
        self.status = "READY"
        self.closed_reason: str | None = None
        self.result: ROMGatedRunResult | None = None

    @property
    def profile(self) -> SubjectROMProfile:
        return self._profile

    def run(
        self,
        environment: PersonalizationEnvironment,
        *,
        method: str,
        seed: int = 0,
        physics_model: PhysicsSubjectModel | None = None,
        kappa: float = 1.5,
    ) -> ROMGatedRunResult:
        if self.status != "READY":
            raise RuntimeError(f"personalization episode is not runnable: {self.status}")
        sequential = run_sequential_personalization(
            environment,
            self.domain,
            method=method,
            budget=self.adaptation_budget,
            seed=seed,
            physics_model=physics_model,
            kappa=kappa,
        )
        if len(sequential.ledger.entries) != self.adaptation_budget:
            raise RuntimeError("Stage 1 ledger does not match its independent K budget")
        self.status = "COMPLETED"
        self.result = ROMGatedRunResult(
            episode_id=self.episode_id,
            rom_profile_id=self.profile.profile_id,
            rom_profile_fingerprint=self.profile.fingerprint,
            sequential_result=sequential,
        )
        return self.result

    def restart_for_changed_rom(
        self,
        *,
        new_episode_id: str,
        new_profile: SubjectROMProfile,
        new_domain: SubjectSpecificV3CandidateDomain,
    ) -> "ROMGatedPersonalizationEpisode":
        if new_profile.fingerprint == self.profile.fingerprint:
            raise ValueError("restart requires a changed ROM profile")
        if new_profile.version <= self.profile.version:
            raise ValueError("changed ROM profile must advance its version")
        if not new_profile.frozen:
            raise RuntimeError("replacement ROM profile must be frozen")
        if new_domain.profile.fingerprint != new_profile.fingerprint:
            raise RuntimeError("replacement domain/profile mismatch")
        self.status = "CLOSED_ROM_PROFILE_CHANGED"
        self.closed_reason = "ROM_CHANGE_INVALIDATES_PERSONALIZATION_EPISODE"
        return ROMGatedPersonalizationEpisode(
            episode_id=new_episode_id,
            profile=new_profile,
            domain=new_domain,
            adaptation_budget=self.adaptation_budget,
        )


def assert_same_frozen_rom_comparison(
    results: Mapping[str, ROMGatedRunResult],
) -> str:
    """Reject a baseline comparison whose methods used different task ROMs."""

    if not results:
        raise ValueError("at least one method result is required")
    fingerprints = {result.rom_profile_fingerprint for result in results.values()}
    if len(fingerprints) != 1:
        raise RuntimeError("baseline comparison requires one identical frozen ROM")
    return next(iter(fingerprints))
