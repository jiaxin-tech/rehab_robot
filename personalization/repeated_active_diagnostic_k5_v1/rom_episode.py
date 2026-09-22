"""Frozen-ROM wrapper for repeated active diagnostics at K=5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..environment import PersonalizationEnvironment
from ..models.physics_graybox import PhysicsSubjectModel
from ..rom_gated_v2.episode import PERSONALIZATION_TRIAL_LEDGER
from ..rom_gated_v2.reference import SubjectSpecificV3CandidateDomain
from ..rom_gated_v2.rom import SubjectROMProfile
from .repeated import PRIMARY_BUDGET
from .sequential import (
    RepeatedActiveDiagnosticSequentialRunResult,
    run_repeated_active_diagnostic_arbitration_k5,
)


@dataclass(frozen=True)
class ROMGatedRepeatedDiagnosticRunResult:
    episode_id: str
    rom_profile_id: str
    rom_profile_fingerprint: str
    sequential_result: RepeatedActiveDiagnosticSequentialRunResult

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


class ROMGatedRepeatedDiagnosticEpisode:
    """One K=5 repeated-diagnostic episode bound to an immutable ROM."""

    def __init__(
        self,
        *,
        episode_id: str,
        profile: SubjectROMProfile,
        domain: SubjectSpecificV3CandidateDomain,
        adaptation_budget: int = PRIMARY_BUDGET,
    ) -> None:
        if not profile.frozen:
            raise RuntimeError("repeated active diagnostic requires ROM_PROFILE_FROZEN=true")
        if domain.profile.fingerprint != profile.fingerprint:
            raise RuntimeError("repeated-diagnostic domain/ROM fingerprint mismatch")
        if adaptation_budget != PRIMARY_BUDGET:
            raise ValueError("repeated-diagnostic episode requires K=5")
        self.episode_id = episode_id
        self._profile = profile
        self.domain = domain
        self.adaptation_budget = adaptation_budget
        self.status = "READY"
        self.closed_reason: str | None = None
        self.result: ROMGatedRepeatedDiagnosticRunResult | None = None

    @property
    def profile(self) -> SubjectROMProfile:
        return self._profile

    def run(
        self,
        environment: PersonalizationEnvironment,
        *,
        physics_model: PhysicsSubjectModel,
        kappa: float = 1.5,
    ) -> ROMGatedRepeatedDiagnosticRunResult:
        if self.status != "READY":
            raise RuntimeError(f"repeated-diagnostic episode is not runnable: {self.status}")
        sequential = run_repeated_active_diagnostic_arbitration_k5(
            environment,
            self.domain,
            physics_model=physics_model,
            budget=self.adaptation_budget,
            kappa=kappa,
        )
        self.status = "COMPLETED"
        self.result = ROMGatedRepeatedDiagnosticRunResult(
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
    ) -> "ROMGatedRepeatedDiagnosticEpisode":
        if new_profile.fingerprint == self.profile.fingerprint:
            raise ValueError("restart requires a changed ROM profile")
        if new_profile.version <= self.profile.version:
            raise ValueError("changed ROM must advance the profile version")
        if not new_profile.frozen:
            raise RuntimeError("replacement ROM profile must be frozen")
        if new_domain.profile.fingerprint != new_profile.fingerprint:
            raise RuntimeError("replacement repeated-diagnostic domain/profile mismatch")
        self.status = "CLOSED_ROM_PROFILE_CHANGED"
        self.closed_reason = "ROM_CHANGE_INVALIDATES_PERSONALIZATION_EPISODE"
        return ROMGatedRepeatedDiagnosticEpisode(
            episode_id=new_episode_id,
            profile=new_profile,
            domain=new_domain,
            adaptation_budget=self.adaptation_budget,
        )

