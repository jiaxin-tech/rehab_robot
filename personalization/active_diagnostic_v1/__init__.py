"""Active prior diagnostic arbitration for offline K=4 personalization."""

from .diagnostic import (
    ARBITRATION_MARGIN,
    DIAGNOSTIC_SCORE_ID,
    INCONCLUSIVE_EXPERT,
    PHYSICS_MODE,
    PRIMARY_RULE_ID,
    STANDARD_MODE,
    VARIANCE_FLOOR_STD,
    DiagnosticArbitrationDecision,
    DiagnosticSelection,
    arbitrate_after_diagnostic,
    average_symmetric_kl_gaussian,
    causal_trial_1_physics_offset,
    gaussian_kl_divergence,
    select_active_diagnostic_candidate,
    verify_frozen_rule,
)
from .rom_episode import (
    ROMGatedActiveDiagnosticEpisode,
    ROMGatedActiveDiagnosticRunResult,
)
from .sequential import (
    ACTIVE_DIAGNOSTIC_METHOD,
    PRIMARY_ADAPTATION_BUDGET,
    ActiveDiagnosticLedgerEntry,
    ActiveDiagnosticSequentialRunResult,
    run_active_prior_diagnostic_arbitration,
)

__all__ = [
    "ACTIVE_DIAGNOSTIC_METHOD",
    "ARBITRATION_MARGIN",
    "DIAGNOSTIC_SCORE_ID",
    "INCONCLUSIVE_EXPERT",
    "PHYSICS_MODE",
    "PRIMARY_ADAPTATION_BUDGET",
    "PRIMARY_RULE_ID",
    "STANDARD_MODE",
    "VARIANCE_FLOOR_STD",
    "ActiveDiagnosticLedgerEntry",
    "ActiveDiagnosticSequentialRunResult",
    "DiagnosticArbitrationDecision",
    "DiagnosticSelection",
    "ROMGatedActiveDiagnosticEpisode",
    "ROMGatedActiveDiagnosticRunResult",
    "arbitrate_after_diagnostic",
    "average_symmetric_kl_gaussian",
    "causal_trial_1_physics_offset",
    "gaussian_kl_divergence",
    "run_active_prior_diagnostic_arbitration",
    "select_active_diagnostic_candidate",
    "verify_frozen_rule",
]
