"""Predictive-evidence physics-prior failover for offline K=4 personalization."""

from .evidence import (
    FAILOVER_THRESHOLD,
    PHYSICS_MODE,
    PRIMARY_RULE_ID,
    SCORE_START_TRIAL,
    STANDARD_MODE,
    VARIANCE_FLOOR_STD,
    PretrialPredictionSnapshot,
    PredictiveEvidenceArbitrator,
    PredictiveEvidenceRecord,
    PredictiveEvidenceState,
    capture_pretrial_predictions,
    gaussian_log_predictive_density,
    verify_frozen_rule,
)
from .rom_episode import (
    ROMGatedPredictiveFailoverEpisode,
    ROMGatedPredictiveFailoverRunResult,
)
from .sequential import (
    PREDICTIVE_FAILOVER_METHOD,
    PRIMARY_ADAPTATION_BUDGET,
    PredictiveFailoverLedgerEntry,
    PredictiveFailoverSequentialRunResult,
    run_predictive_evidence_failover,
)

__all__ = [
    "FAILOVER_THRESHOLD",
    "PHYSICS_MODE",
    "PREDICTIVE_FAILOVER_METHOD",
    "PRIMARY_ADAPTATION_BUDGET",
    "PRIMARY_RULE_ID",
    "PretrialPredictionSnapshot",
    "PredictiveEvidenceArbitrator",
    "PredictiveEvidenceRecord",
    "PredictiveEvidenceState",
    "PredictiveFailoverLedgerEntry",
    "PredictiveFailoverSequentialRunResult",
    "ROMGatedPredictiveFailoverEpisode",
    "ROMGatedPredictiveFailoverRunResult",
    "SCORE_START_TRIAL",
    "STANDARD_MODE",
    "VARIANCE_FLOOR_STD",
    "capture_pretrial_predictions",
    "gaussian_log_predictive_density",
    "run_predictive_evidence_failover",
    "verify_frozen_rule",
]
