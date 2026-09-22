"""Offline adaptive physics-prior trust for K=4 personalization."""

from .model import AdaptivePhysicsMixtureModel
from .rom_episode import (
    ROMGatedAdaptiveTrustEpisode,
    ROMGatedAdaptiveTrustRunResult,
)
from .sequential import (
    ADAPTIVE_TRUST_METHOD,
    AdaptiveTrustLedgerEntry,
    AdaptiveTrustSequentialRunResult,
    run_adaptive_trust_personalization,
)
from .trust import (
    EXPECTED_RULE_SHA256,
    FALLBACK_DOMINANT_THRESHOLD,
    FixedTrustEstimator,
    PhysicsPriorTrustEstimator,
    PhysicsPriorTrustEvidence,
    PhysicsPriorTrustState,
    verify_frozen_rule,
)

__all__ = [
    "ADAPTIVE_TRUST_METHOD",
    "AdaptivePhysicsMixtureModel",
    "AdaptiveTrustLedgerEntry",
    "AdaptiveTrustSequentialRunResult",
    "EXPECTED_RULE_SHA256",
    "FALLBACK_DOMINANT_THRESHOLD",
    "FixedTrustEstimator",
    "PhysicsPriorTrustEstimator",
    "PhysicsPriorTrustEvidence",
    "PhysicsPriorTrustState",
    "ROMGatedAdaptiveTrustEpisode",
    "ROMGatedAdaptiveTrustRunResult",
    "run_adaptive_trust_personalization",
    "verify_frozen_rule",
]
