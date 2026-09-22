"""Frozen-ROM wrapper for Predictive-Evidence Failover V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..environment import PersonalizationEnvironment
from ..models.physics_graybox import PhysicsSubjectModel
from ..rom_gated_v2.episode import PERSONALIZATION_TRIAL_LEDGER
from ..rom_gated_v2.reference import SubjectSpecificV3CandidateDomain
from ..rom_gated_v2.rom import SubjectROMProfile
from .sequential import (
    PRIMARY_ADAPTATION_BUDGET,
    PredictiveFailoverSequentialRunResult,
    run_predictive_evidence_failover,
)


@dataclass(frozen=True)
class ROMGatedPredictiveFailoverRunResult:
    episode_id: str
    rom_profile_id: str
    rom_profile_fingerprint: str
    sequential_result: PredictiveFailoverSequentialRunResult

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


class ROMGatedPredictiveFailoverEpisode:
    """One K=4 failover episode bound to one immutable ROM profile."""

    def __init__(
        self,
        *,
        episode_id: str,
        profile: SubjectROMProfile,
        domain: SubjectSpecificV3CandidateDomain,
        adaptation_budget: int = PRIMARY_ADAPTATION_BUDGET,
    ) -> None:
        if not profile.frozen:
            raise RuntimeError("predictive failover requires ROM_PROFILE_FROZEN=true")
        if domain.profile.fingerprint != profile.fingerprint:
            raise RuntimeError("predictive-failover domain/ROM fingerprint mismatch")
        if adaptation_budget != PRIMARY_ADAPTATION_BUDGET:
            raise ValueError("predictive-failover primary episode requires K=4")
        self.episode_id = episode_id
        self._profile = profile
        self.domain = domain
        self.adaptation_budget = adaptation_budget
        self.status = "READY"
        self.closed_reason: str | None = None
        self.result: ROMGatedPredictiveFailoverRunResult | None = None

    @property
    def profile(self) -> SubjectROMProfile:
        return self._profile

    def run(
        self,
        environment: PersonalizationEnvironment,
        *,
        physics_model: PhysicsSubjectModel,
        kappa: float = 1.5,
    ) -> ROMGatedPredictiveFailoverRunResult:
        if self.status != "READY":
            raise RuntimeError(f"predictive-failover episode is not runnable: {self.status}")
        sequential = run_predictive_evidence_failover(
            environment,
            self.domain,
            physics_model=physics_model,
            budget=self.adaptation_budget,
            kappa=kappa,
        )
        self.status = "COMPLETED"
        self.result = ROMGatedPredictiveFailoverRunResult(
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
    ) -> "ROMGatedPredictiveFailoverEpisode":
        if new_profile.fingerprint == self.profile.fingerprint:
            raise ValueError("restart requires a changed ROM profile")
        if new_profile.version <= self.profile.version:
            raise ValueError("changed ROM must advance the profile version")
        if not new_profile.frozen:
            raise RuntimeError("replacement ROM profile must be frozen")
        if new_domain.profile.fingerprint != new_profile.fingerprint:
            raise RuntimeError("replacement predictive-failover domain/profile mismatch")
        self.status = "CLOSED_ROM_PROFILE_CHANGED"
        self.closed_reason = "ROM_CHANGE_INVALIDATES_PERSONALIZATION_EPISODE"
        return ROMGatedPredictiveFailoverEpisode(
            episode_id=new_episode_id,
            profile=new_profile,
            domain=new_domain,
            adaptation_budget=self.adaptation_budget,
        )

