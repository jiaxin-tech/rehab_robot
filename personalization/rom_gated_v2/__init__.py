"""Subject-specific ROM-gated sequential personalization V2.

All functionality is offline-only.  The existing personalization V1 package
and its historical benchmark artifacts remain unchanged.
"""

from .development import (
    OfflineROMDevelopmentCase,
    SubjectSpecificOfflineMechanicalEnvironment,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
    make_subject_physics_model,
)
from .episode import (
    PERSONALIZATION_ADAPTATION_BUDGET_SENSITIVITY,
    PRIMARY_PERSONALIZATION_ADAPTATION_BUDGET,
    ROMGatedPersonalizationEpisode,
    ROMGatedRunResult,
    assert_same_frozen_rom_comparison,
)
from .physics import SubjectSpecificFullDynamicsGrayBoxEndpointAdapter
from .reference import (
    SubjectSpecificCandidate,
    SubjectSpecificReference,
    SubjectSpecificReferenceAdapter,
    SubjectSpecificV3CandidateDomain,
)
from .rom import (
    CompositeROMSafetyGate,
    ROMBoundaryObservation,
    ROMCalibrationLedger,
    ROMDeterminationController,
    ROMGateResult,
    ROMGateStatus,
    ROMSafetyGate,
    RealROMDeterminationInterface,
    SubjectROMProfile,
    SyntheticThresholdGate,
    UnavailableValidatedSafetyGate,
)

__all__ = [
    "CompositeROMSafetyGate",
    "OfflineROMDevelopmentCase",
    "PERSONALIZATION_ADAPTATION_BUDGET_SENSITIVITY",
    "PRIMARY_PERSONALIZATION_ADAPTATION_BUDGET",
    "ROMBoundaryObservation",
    "ROMCalibrationLedger",
    "ROMDeterminationController",
    "ROMGateResult",
    "ROMGateStatus",
    "ROMGatedPersonalizationEpisode",
    "ROMGatedRunResult",
    "ROMSafetyGate",
    "RealROMDeterminationInterface",
    "SubjectROMProfile",
    "SubjectSpecificCandidate",
    "SubjectSpecificFullDynamicsGrayBoxEndpointAdapter",
    "SubjectSpecificOfflineMechanicalEnvironment",
    "SubjectSpecificReference",
    "SubjectSpecificReferenceAdapter",
    "SubjectSpecificV3CandidateDomain",
    "SyntheticThresholdGate",
    "UnavailableValidatedSafetyGate",
    "assert_same_frozen_rom_comparison",
    "determine_synthetic_rom",
    "make_offline_rom_development_cases",
    "make_subject_physics_model",
]
